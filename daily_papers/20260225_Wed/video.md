# 计算机视觉 cs.CV

- **最新发布 124 篇**

- **更新 79 篇**

## 最新发布

#### [new 001] DA-Cal: Towards Cross-Domain Calibration in Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于语义分割的跨域适应任务，解决UDA方法中网络校准不足的问题，提出DA-Cal框架优化软伪标签，提升目标域校准精度。**

- **链接: [https://arxiv.org/pdf/2602.20860v1](https://arxiv.org/pdf/2602.20860v1)**

> **作者:** Wangkai Li; Rui Sun; Zhaoyang Li; Yujia Chen; Tianzhu Zhang
>
> **摘要:** While existing unsupervised domain adaptation (UDA) methods greatly enhance target domain performance in semantic segmentation, they often neglect network calibration quality, resulting in misalignment between prediction confidence and actual accuracy -- a significant risk in safety-critical applications. Our key insight emerges from observing that performance degrades substantially when soft pseudo-labels replace hard pseudo-labels in cross-domain scenarios due to poor calibration, despite the theoretical equivalence of perfectly calibrated soft pseudo-labels to hard pseudo-labels. Based on this finding, we propose DA-Cal, a dedicated cross-domain calibration framework that transforms target domain calibration into soft pseudo-label optimization. DA-Cal introduces a Meta Temperature Network to generate pixel-level calibration parameters and employs bi-level optimization to establish the relationship between soft pseudo-labels and UDA supervision, while utilizing complementary domain-mixing strategies to prevent overfitting and reduce domain discrepancies. Experiments demonstrate that DA-Cal seamlessly integrates with existing self-training frameworks across multiple UDA segmentation benchmarks, significantly improving target domain calibration while delivering performance gains without inference overhead. The code will be released.
>
---
#### [new 002] Computing a Characteristic Orientation for Rotation-Independent Image Analysis
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决旋转不变性问题。通过引入GID方法，在不修改网络结构的情况下提升模型对旋转的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.20930v1](https://arxiv.org/pdf/2602.20930v1)**

> **作者:** Cristian Valero-Abundio; Emilio Sansano-Sansano; Raúl Montoliu; Marina Martínez García
>
> **备注:** Accepted for publication at the 21st International Conference on Computer Vision Theory and Applications (VISAPP 2026). 8 pages
>
> **摘要:** Handling geometric transformations, particularly rotations, remains a challenge in deep learning for computer vision. Standard neural networks lack inherent rotation invariance and typically rely on data augmentation or architectural modifications to improve robustness. Although effective, these approaches increase computational demands, require specialised implementations, or alter network structures, limiting their applicability. This paper introduces General Intensity Direction (GID), a preprocessing method that improves rotation robustness without modifying the network architecture. The method estimates a global orientation for each image and aligns it to a canonical reference frame, allowing standard models to process inputs more consistently across different rotations. Unlike moment-based approaches that extract invariant descriptors, this method directly transforms the image while preserving spatial structure, making it compatible with convolutional networks. Experimental evaluation on the rotated MNIST dataset shows that the proposed method achieves higher accuracy than state-of-the-art rotation-invariant architectures. Additional experiments on the CIFAR-10 dataset, confirm that the method remains effective under more complex conditions.
>
---
#### [new 003] Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景重建任务，旨在解决无人机影像实时高保真重建问题。通过整合视频流、传感器融合与3DGS优化，实现低延迟、高效率的实时重建与可视化。**

- **链接: [https://arxiv.org/pdf/2602.20342v1](https://arxiv.org/pdf/2602.20342v1)**

> **作者:** Christos Maikos; Georgios Angelidis; Georgios Th. Papadopoulos
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** In this study, we present an end-to-end pipeline capable of converting drone-captured video streams into high-fidelity 3D reconstructions with minimal latency. Unmanned aerial vehicles (UAVs) are extensively used in aerial real-time perception applications. Moreover, recent advances in 3D Gaussian Splatting (3DGS) have demonstrated significant potential for real-time neural rendering. However, their integration into end-to-end UAV-based reconstruction and visualization systems remains underexplored. Our goal is to propose an efficient architecture that combines live video acquisition via RTMP streaming, synchronized sensor fusion, camera pose estimation, and 3DGS optimization, achieving continuous model updates and low-latency deployment within interactive visualization environments that supports immersive augmented and virtual reality (AR/VR) applications. Experimental results demonstrate that the proposed method achieves competitive visual fidelity, while delivering significantly higher rendering performance and substantially reduced end-to-end latency, compared to NeRF-based approaches. Reconstruction quality remains within 4-7\% of high-fidelity offline references, confirming the suitability of the proposed system for real-time, scalable augmented perception from aerial platforms.
>
---
#### [new 004] 3DSPA: A 3D Semantic Point Autoencoder for Evaluating Video Realism
- **分类: cs.CV**

- **简介: 该论文提出3DSPA，用于自动评估视频真实性。解决生成视频质量评估问题，通过融合3D语义和运动信息实现更准确的自动化评估。**

- **链接: [https://arxiv.org/pdf/2602.20354v1](https://arxiv.org/pdf/2602.20354v1)**

> **作者:** Bhavik Chandna; Kelsey R. Allen
>
> **摘要:** AI video generation is evolving rapidly. For video generators to be useful for applications ranging from robotics to film-making, they must consistently produce realistic videos. However, evaluating the realism of generated videos remains a largely manual process -- requiring human annotation or bespoke evaluation datasets which have restricted scope. Here we develop an automated evaluation framework for video realism which captures both semantics and coherent 3D structure and which does not require access to a reference video. Our method, 3DSPA, is a 3D spatiotemporal point autoencoder which integrates 3D point trajectories, depth cues, and DINO semantic features into a unified representation for video evaluation. 3DSPA models how objects move and what is happening in the scene, enabling robust assessments of realism, temporal consistency, and physical plausibility. Experiments show that 3DSPA reliably identifies videos which violate physical laws, is more sensitive to motion artifacts, and aligns more closely with human judgments of video quality and realism across multiple datasets. Our results demonstrate that enriching trajectory-based representations with 3D semantics offers a stronger foundation for benchmarking generative video models, and implicitly captures physical rule violations. The code and pretrained model weights will be available at https://github.com/TheProParadox/3dspa_code.
>
---
#### [new 005] An interactive enhanced driving dataset for autonomous driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决VLA模型数据不足与对齐问题。提出IEDD数据集，通过挖掘交互片段和生成合成视频提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2602.20575v1](https://arxiv.org/pdf/2602.20575v1)**

> **作者:** Haojie Feng; Peizhi Zhang; Mengjie Tian; Xinrui Zhang; Zhuoren Li; Junpeng Huang; Xiurong Wang; Junfan Zhu; Jianzhou Wang; Dongxiao Yin; Lu Xiong
>
> **摘要:** The evolution of autonomous driving towards full automation demands robust interactive capabilities; however, the development of Vision-Language-Action (VLA) models is constrained by the sparsity of interactive scenarios and inadequate multimodal alignment in existing data. To this end, this paper proposes the Interactive Enhanced Driving Dataset (IEDD). We develop a scalable pipeline to mine million-level interactive segments from naturalistic driving data based on interactive trajectories, and design metrics to quantify the interaction processes. Furthermore, the IEDD-VQA dataset is constructed by generating synthetic Bird's Eye View (BEV) videos where semantic actions are strictly aligned with structured language. Benchmark results evaluating ten mainstream Vision Language Models (VLMs) are provided to demonstrate the dataset's reuse value in assessing and fine-tuning the reasoning capabilities of autonomous driving models.
>
---
#### [new 006] Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决长期监测中因外观变化导致的多会话模型对齐问题。通过联合SfM重建和特征匹配，实现跨时间的连贯3D模型重建。**

- **链接: [https://arxiv.org/pdf/2602.20584v1](https://arxiv.org/pdf/2602.20584v1)**

> **作者:** Beverley Gorry; Tobias Fischer; Michael Milford; Alejandro Fontan
>
> **摘要:** Long-term environmental monitoring requires the ability to reconstruct and align 3D models across repeated site visits separated by months or years. However, existing Structure-from-Motion (SfM) pipelines implicitly assume near-simultaneous image capture and limited appearance change, and therefore fail when applied to long-term monitoring scenarios such as coral reef surveys, where substantial visual and structural change is common. In this paper, we show that the primary limitation of current approaches lies in their reliance on post-hoc alignment of independently reconstructed sessions, which is insufficient under large temporal appearance change. We address this limitation by enforcing cross-session correspondences directly within a joint SfM reconstruction. Our approach combines complementary handcrafted and learned visual features to robustly establish correspondences across large temporal gaps, enabling the reconstruction of a single coherent 3D model from imagery captured years apart, where standard independent and joint SfM pipelines break down. We evaluate our method on long-term coral reef datasets exhibiting significant real-world change, and demonstrate consistent joint reconstruction across sessions in cases where existing methods fail to produce coherent reconstructions. To ensure scalability to large datasets, we further restrict expensive learned feature matching to a small set of likely cross-session image pairs identified via visual place recognition, which reduces computational cost and improves alignment robustness.
>
---
#### [new 007] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决冗余视觉token导致的推理成本高的问题。通过最优传输实现分布对齐的视觉token剪枝，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.20205v1](https://arxiv.org/pdf/2602.20205v1)**

> **作者:** Xiwen Chen; Wenhui Zhu; Gen Li; Xuanzhao Dong; Yujian Xiong; Hao Wang; Peijie Qiu; Qingquan Song; Zhipeng Wang; Shao Tang; Yalin Wang; Abolfazl Razi
>
> **备注:** Accepted by CVPR2026 (Findings). arXiv admin note: text overlap with arXiv:2503.02175 by other authors
>
> **摘要:** Multi-modal large language models (MLLMs) achieve strong visual-language reasoning but suffer from high inference cost due to redundant visual tokens. Recent work explores visual token pruning to accelerate inference, while existing pruning methods overlook the underlying distributional structure of visual representations. We propose OTPrune, a training-free framework that formulates pruning as distribution alignment via optimal transport (OT). By minimizing the 2-Wasserstein distance between the full and pruned token distributions, OTPrune preserves both local diversity and global representativeness while reducing inference cost. Moreover, we derive a tractable submodular objective that enables efficient optimization, and theoretically prove its monotonicity and submodularity, providing a principled foundation for stable and efficient pruning. We further provide a comprehensive analysis that explains how distributional alignment contributes to stable and semantically faithful pruning. Comprehensive experiments on wider benchmarks demonstrate that OTPrune achieves superior performance-efficiency tradeoffs compared to state-of-the-art methods. The code is available at https://github.com/xiwenc1/OTPrune.
>
---
#### [new 008] Path-Decoupled Hyperbolic Flow Matching for Few-Shot Adaptation
- **分类: cs.CV**

- **简介: 该论文属于少样本适应任务，旨在解决欧几里得流匹配中的路径纠缠问题。提出路径解耦的双曲流匹配方法，通过双曲几何实现更优的语义对齐与轨迹分离。**

- **链接: [https://arxiv.org/pdf/2602.20479v1](https://arxiv.org/pdf/2602.20479v1)**

> **作者:** Lin Li; Ziqi Jiang; Gefan Ye; Zhenqi He; Jiahui Li; Jun Xiao; Kwang-Ting Cheng; Long Chen
>
> **摘要:** Recent advances in cross-modal few-shot adaptation treat visual-semantic alignment as a continuous feature transport problem via Flow Matching (FM). However, we argue that Euclidean-based FM overlooks fundamental limitations of flat geometry, where polynomial volume growth fails to accommodate diverse feature distributions, leading to severe path entanglement. To this end, we propose path-decoupled Hyperbolic Flow Matching (HFM), leveraging the Lorentz manifold's exponential expansion for trajectory decoupling. HFM structures the transport via two key designs: 1) Centripetal hyperbolic alignment: It constructs a centripetal hierarchy by anchoring textual roots, which pushes visual leaves to the boundary to initialize orderly flows. 2) Path-decoupled objective: It acts as a ``semantic guardrail'' rigidly confining trajectories within isolated class-specific geodesic corridors via step-wise supervision. Furthermore, we devise an adaptive diameter-based stopping to prevent over-transportation into the crowded origin based on the intrinsic semantic scale. Extensive ablations on 11 benchmarks have shown that HFM establishes a new state-of-the-art, consistently outperforming its Euclidean counterparts. Our codes and models will be released.
>
---
#### [new 009] A Lightweight Vision-Language Fusion Framework for Predicting App Ratings from User Interfaces and Metadata
- **分类: cs.CV**

- **简介: 该论文属于应用评价预测任务，旨在通过融合UI和语义信息提升App评分预测效果。提出轻量级视觉-语言框架，结合MobileNetV3和DistilBERT进行特征提取与融合。**

- **链接: [https://arxiv.org/pdf/2602.20531v1](https://arxiv.org/pdf/2602.20531v1)**

> **作者:** Azrin Sultana; Firoz Ahmed
>
> **备注:** 24 pages, 10 figures
>
> **摘要:** App ratings are among the most significant indicators of the quality, usability, and overall user satisfaction of mobile applications. However, existing app rating prediction models are largely limited to textual data or user interface (UI) features, overlooking the importance of jointly leveraging UI and semantic information. To address these limitations, this study proposes a lightweight vision--language framework that integrates both mobile UI and semantic information for app rating prediction. The framework combines MobileNetV3 to extract visual features from UI layouts and DistilBERT to extract textual features. These multimodal features are fused through a gated fusion module with Swish activations, followed by a multilayer perceptron (MLP) regression head. The proposed model is evaluated using mean absolute error (MAE), root mean square error (RMSE), mean squared error (MSE), coefficient of determination (R2), and Pearson correlation. After training for 20 epochs, the model achieves an MAE of 0.1060, an RMSE of 0.1433, an MSE of 0.0205, an R2 of 0.8529, and a Pearson correlation of 0.9251. Extensive ablation studies further demonstrate the effectiveness of different combinations of visual and textual encoders. Overall, the proposed lightweight framework provides valuable insights for developers and end users, supports sustainable app development, and enables efficient deployment on edge devices.
>
---
#### [new 010] SynthRender and IRIS: Open-Source Framework and Dataset for Bidirectional Sim-Real Transfer in Industrial Object Perception
- **分类: cs.CV**

- **简介: 该论文属于工业物体感知任务，解决真实与仿真数据迁移问题。提出SynthRender框架和IRIS数据集，生成高质量合成数据以提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21141v1](https://arxiv.org/pdf/2602.21141v1)**

> **作者:** Jose Moises Araya-Martinez; Thushar Tom; Adrián Sanchis Reig; Pablo Rey Valiente; Jens Lambrecht; Jörg Krüger
>
> **摘要:** Object perception is fundamental for tasks such as robotic material handling and quality inspection. However, modern supervised deep-learning perception models require large datasets for robust automation under semi-uncontrolled conditions. The cost of acquiring and annotating such data for proprietary parts is a major barrier for widespread deployment. In this context, we release SynthRender, an open source framework for synthetic image generation with Guided Domain Randomization capabilities. Furthermore, we benchmark recent Reality-to-Simulation techniques for 3D asset creation from 2D images of real parts. Combined with Domain Randomization, these synthetic assets provide low-overhead, transferable data even for parts lacking 3D files. We also introduce IRIS, the Industrial Real-Sim Imagery Set, containing 32 categories with diverse textures, intra-class variation, strong inter-class similarities and about 20,000 labels. Ablations on multiple benchmarks outline guidelines for efficient data generation with SynthRender. Our method surpasses existing approaches, achieving 99.1% mAP@50 on a public robotics dataset, 98.3% mAP@50 on an automotive benchmark, and 95.3% mAP@50 on IRIS.
>
---
#### [new 011] VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于模型自评估任务，旨在解决LVLM在视觉条件下的不确定性量化问题。提出VAUQ框架，通过视觉信息得分和核心区域掩码策略，提升模型输出的可靠性。**

- **链接: [https://arxiv.org/pdf/2602.21054v1](https://arxiv.org/pdf/2602.21054v1)**

> **作者:** Seongheon Park; Changdae Oh; Hyeong Kyu Choi; Xuefeng Du; Sharon Li
>
> **摘要:** Large Vision-Language Models (LVLMs) frequently hallucinate, limiting their safe deployment in real-world applications. Existing LLM self-evaluation methods rely on a model's ability to estimate the correctness of its own outputs, which can improve deployment reliability; however, they depend heavily on language priors and are therefore ill-suited for evaluating vision-conditioned predictions. We propose VAUQ, a vision-aware uncertainty quantification framework for LVLM self-evaluation that explicitly measures how strongly a model's output depends on visual evidence. VAUQ introduces the Image-Information Score (IS), which captures the reduction in predictive uncertainty attributable to visual input, and an unsupervised core-region masking strategy that amplifies the influence of salient regions. Combining predictive entropy with this core-masked IS yields a training-free scoring function that reliably reflects answer correctness. Comprehensive experiments show that VAUQ consistently outperforms existing self-evaluation methods across multiple datasets.
>
---
#### [new 012] MIP Candy: A Modular PyTorch Framework for Medical Image Processing
- **分类: cs.CV; cs.AI; cs.LG; cs.SE**

- **简介: 该论文提出MIP Candy框架，解决医学图像处理中软件灵活性与集成难度问题，提供模块化、可配置的PyTorch工具链。**

- **链接: [https://arxiv.org/pdf/2602.21033v1](https://arxiv.org/pdf/2602.21033v1)**

> **作者:** Tianhao Fu; Yucheng Chen
>
> **摘要:** Medical image processing demands specialized software that handles high-dimensional volumetric data, heterogeneous file formats, and domain-specific training procedures. Existing frameworks either provide low-level components that require substantial integration effort or impose rigid, monolithic pipelines that resist modification. We present MIP Candy (MIPCandy), a freely available, PyTorch-based framework designed specifically for medical image processing. MIPCandy provides a complete, modular pipeline spanning data loading, training, inference, and evaluation, allowing researchers to obtain a fully functional process workflow by implementing a single method, $\texttt{build_network}$, while retaining fine-grained control over every component. Central to the design is $\texttt{LayerT}$, a deferred configuration mechanism that enables runtime substitution of convolution, normalization, and activation modules without subclassing. The framework further offers built-in $k$-fold cross-validation, dataset inspection with automatic region-of-interest detection, deep supervision, exponential moving average, multi-frontend experiment tracking (Weights & Biases, Notion, MLflow), training state recovery, and validation score prediction via quotient regression. An extensible bundle ecosystem provides pre-built model implementations that follow a consistent trainer--predictor pattern and integrate with the core framework without modification. MIPCandy is open-source under the Apache-2.0 license and requires Python~3.12 or later. Source code and documentation are available at https://github.com/ProjectNeura/MIPCandy.
>
---
#### [new 013] Training-Free Multi-Concept Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决无需训练的多概念编辑问题。通过结合优化和LoRA技术，实现无需训练即可控制多个视觉概念的图像生成。**

- **链接: [https://arxiv.org/pdf/2602.20839v1](https://arxiv.org/pdf/2602.20839v1)**

> **作者:** Niki Foteinopoulou; Ignas Budvytis; Stephan Liwicki
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** Editing images with diffusion models without training remains challenging. While recent optimisation-based methods achieve strong zero-shot edits from text, they struggle to preserve identity or capture details that language alone cannot express. Many visual concepts such as facial structure, material texture, or object geometry are impossible to express purely through text prompts alone. To address this gap, we introduce a training-free framework for concept-based image editing, which unifies Optimised DDS with LoRA-driven concept composition, where the training data of the LoRA represent the concept. Our approach enables combining and controlling multiple visual concepts directly within the diffusion process, integrating semantic guidance from text with low-level cues from pretrained concept adapters. We further refine DDS for stability and controllability through ordered timesteps, regularisation, and negative-prompt guidance. Quantitative and qualitative results demonstrate consistent improvements over existing training-free diffusion editing methods on InstructPix2Pix and ComposLoRA benchmarks. Code will be made publicly available.
>
---
#### [new 014] N4MC: Neural 4D Mesh Compression
- **分类: cs.CV**

- **简介: 该论文提出N4MC，属于4D网格压缩任务，解决时间变化网格序列的高效压缩问题。通过学习运动补偿和时空相关性，实现更优的压缩效果。**

- **链接: [https://arxiv.org/pdf/2602.20312v1](https://arxiv.org/pdf/2602.20312v1)**

> **作者:** Guodong Chen; Huanshuo Dong; Mallesham Dasari
>
> **摘要:** We present N4MC, the first 4D neural compression framework to efficiently compress time-varying mesh sequences by exploiting their temporal redundancy. Unlike prior neural mesh compression methods that treat each mesh frame independently, N4MC takes inspiration from inter-frame compression in 2D video codecs, and learns motion compensation in long mesh sequences. Specifically, N4MC converts consecutive irregular mesh frames into regular 4D tensors to provide a uniform and compact representation. These tensors are then condensed using an auto-decoder, which captures both spatial and temporal correlations for redundancy removal. To enhance temporal coherence, we introduce a transformer-based interpolation model that predicts intermediate mesh frames conditioned on latent embeddings derived from tracked volume centers, eliminating motion ambiguities. Extensive evaluations show that N4MC outperforms state-of-the-art in rate-distortion performance, while enabling real-time decoding of 4D mesh sequences. The implementation of our method is available at: https://github.com/frozzzen3/N4MC.
>
---
#### [new 015] Communication-Inspired Tokenization for Structured Image Representations
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出COMiT，用于学习结构化离散视觉令牌序列，解决传统方法侧重重建而缺乏语义结构的问题。属于视觉表示学习任务，通过迭代优化提升对象中心的结构和组合推理能力。**

- **链接: [https://arxiv.org/pdf/2602.20731v1](https://arxiv.org/pdf/2602.20731v1)**

> **作者:** Aram Davtyan; Yusuf Sahin; Yasaman Haghighi; Sebastian Stapf; Pablo Acuaviva; Alexandre Alahi; Paolo Favaro
>
> **备注:** Project website: https://araachie.github.io/comit/
>
> **摘要:** Discrete image tokenizers have emerged as a key component of modern vision and multimodal systems, providing a sequential interface for transformer-based architectures. However, most existing approaches remain primarily optimized for reconstruction and compression, often yielding tokens that capture local texture rather than object-level semantic structure. Inspired by the incremental and compositional nature of human communication, we introduce COMmunication inspired Tokenization (COMiT), a framework for learning structured discrete visual token sequences. COMiT constructs a latent message within a fixed token budget by iteratively observing localized image crops and recurrently updating its discrete representation. At each step, the model integrates new visual information while refining and reorganizing the existing token sequence. After several encoding iterations, the final message conditions a flow-matching decoder that reconstructs the full image. Both encoding and decoding are implemented within a single transformer model and trained end-to-end using a combination of flow-matching reconstruction and semantic representation alignment losses. Our experiments demonstrate that while semantic alignment provides grounding, attentive sequential tokenization is critical for inducing interpretable, object-centric token structure and substantially improving compositional generalization and relational reasoning over prior methods.
>
---
#### [new 016] Aesthetic Camera Viewpoint Suggestion with 3D Aesthetic Field
- **分类: cs.CV**

- **简介: 该论文属于视觉美学任务，解决相机视角建议问题。提出3D美学场概念，利用稀疏图像生成高效美学视角建议，避免复杂计算。**

- **链接: [https://arxiv.org/pdf/2602.20363v1](https://arxiv.org/pdf/2602.20363v1)**

> **作者:** Sheyang Tang; Armin Shafiee Sarvestani; Jialu Xu; Xiaoyu Xu; Zhou Wang
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** The aesthetic quality of a scene depends strongly on camera viewpoint. Existing approaches for aesthetic viewpoint suggestion are either single-view adjustments, predicting limited camera adjustments from a single image without understanding scene geometry, or 3D exploration approaches, which rely on dense captures or prebuilt 3D environments coupled with costly reinforcement learning (RL) searches. In this work, we introduce the notion of 3D aesthetic field that enables geometry-grounded aesthetic reasoning in 3D with sparse captures, allowing efficient viewpoint suggestions in contrast to costly RL searches. We opt to learn this 3D aesthetic field using a feedforward 3D Gaussian Splatting network that distills high-level aesthetic knowledge from a pretrained 2D aesthetic model into 3D space, enabling aesthetic prediction for novel viewpoints from only sparse input views. Building on this field, we propose a two-stage search pipeline that combines coarse viewpoint sampling with gradient-based refinement, efficiently identifying aesthetically appealing viewpoints without dense captures or RL exploration. Extensive experiments show that our method consistently suggests viewpoints with superior framing and composition compared to existing approaches, establishing a new direction toward 3D-aware aesthetic modeling.
>
---
#### [new 017] TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering
- **分类: cs.CV**

- **简介: 该论文属于视觉文本渲染任务，解决文本结构异常检测与优化问题。提出TextPecker方法，提升文本生成的结构准确性和语义对齐度。**

- **链接: [https://arxiv.org/pdf/2602.20903v1](https://arxiv.org/pdf/2602.20903v1)**

> **作者:** Hanshen Zhu; Yuliang Liu; Xuecheng Wu; An-Lan Wang; Hao Feng; Dingkang Yang; Chao Feng; Can Huang; Jingqun Tang; Xiang Bai
>
> **备注:** Code: https://github.com/CIawevy/TextPecker
>
> **摘要:** Visual Text Rendering (VTR) remains a critical challenge in text-to-image generation, where even advanced models frequently produce text with structural anomalies such as distortion, blurriness, and misalignment. However, we find that leading MLLMs and specialist OCR models largely fail to perceive these structural anomalies, creating a critical bottleneck for both VTR evaluation and RL-based optimization. As a result, even state-of-the-art generators (e.g., SeedDream4.0, Qwen-Image) still struggle to render structurally faithful text. To address this, we propose TextPecker, a plug-and-play structural anomaly perceptive RL strategy that mitigates noisy reward signals and works with any textto-image generator. To enable this capability, we construct a recognition dataset with character-level structural-anomaly annotations and develop a stroke-editing synthesis engine to expand structural-error coverage. Experiments show that TextPecker consistently improves diverse text-to-image models; even on the well-optimized Qwen-Image, it significantly yields average gains of 4% in structural fidelity and 8.7% in semantic alignment for Chinese text rendering, establishing a new state-of-the-art in high-fidelity VTR. Our work fills a gap in VTR optimization, providing a foundational step towards reliable and structural faithful visual text generation.
>
---
#### [new 018] Hybrid Fusion: One-Minute Efficient Training for Zero-Shot Cross-Domain Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于图像融合任务，旨在解决传统方法适应性差与深度学习训练效率低的问题。提出一种混合框架，结合U-Net与拉普拉斯金字塔，实现高效全分辨率训练和零样本泛化。**

- **链接: [https://arxiv.org/pdf/2602.20851v1](https://arxiv.org/pdf/2602.20851v1)**

> **作者:** Ran Zhang; Xuanhua He; Liu Liu
>
> **摘要:** Image fusion seeks to integrate complementary information from multiple sources into a single, superior image. While traditional methods are fast, they lack adaptability and performance. Conversely, deep learning approaches achieve state-of-the-art (SOTA) results but suffer from critical inefficiencies: their reliance on slow, resource-intensive, patch-based training introduces a significant gap with full-resolution inference. We propose a novel hybrid framework that resolves this trade-off. Our method utilizes a learnable U-Net to generate a dynamic guidance map that directs a classic, fixed Laplacian pyramid fusion kernel. This decoupling of policy learning from pixel synthesis enables remarkably efficient full-resolution training, eliminating the train-inference gap. Consequently, our model achieves SOTA-comparable performance in about one minute on a RTX 4090 or two minutes on a consumer laptop GPU from scratch without any external model and demonstrates powerful zero-shot generalization across diverse tasks, from infrared-visible to medical imaging. By design, the fused output is linearly constructed solely from source information, ensuring high faithfulness for critical applications. The codes are available at https://github.com/Zirconium233/HybridFusion
>
---
#### [new 019] Optimizing Occupancy Sensor Placement in Smart Environments
- **分类: cs.CV**

- **简介: 该论文属于智能环境中的传感器布局优化任务，旨在解决如何高效布置占用传感器以提高计数准确性。通过模拟和整数线性规划方法，实现最优传感器位置选择。**

- **链接: [https://arxiv.org/pdf/2602.21098v1](https://arxiv.org/pdf/2602.21098v1)**

> **作者:** Hao Lu; Richard J. Radke
>
> **摘要:** Understanding the locations of occupants in a commercial built environment is critical for realizing energy savings by delivering lighting, heating, and cooling only where it is needed. The key to achieving this goal is being able to recognize zone occupancy in real time, without impeding occupants' activities or compromising privacy. While low-resolution, privacy-preserving time-of-flight (ToF) sensor networks have demonstrated good performance in zone counting, the performance depends on careful sensor placement. To address this issue, we propose an automatic sensor placement method that determines optimal sensor layouts for a given number of sensors, and can predict the counting accuracy of such a layout. In particular, given the geometric constraints of an office environment, we simulate a large number of occupant trajectories. We then formulate the sensor placement problem as an integer linear programming (ILP) problem and solve it with the branch and bound method. We demonstrate the effectiveness of the proposed method based on simulations of several different office environments.
>
---
#### [new 020] UDVideoQA: A Traffic Video Question Answering Dataset for Multi-Object Spatio-Temporal Reasoning in Urban Dynamics
- **分类: cs.CV**

- **简介: 该论文提出UDVideoQA数据集，用于解决城市交通视频中的多对象时空推理问题，通过高质量标注和多样化任务评估视觉与因果推理能力。**

- **链接: [https://arxiv.org/pdf/2602.21137v1](https://arxiv.org/pdf/2602.21137v1)**

> **作者:** Joseph Raj Vishal; Nagasiri Poluri; Katha Naik; Rutuja Patil; Kashyap Hegde Kota; Krishna Vinod; Prithvi Jai Ramesh; Mohammad Farhadi; Yezhou Yang; Bharatesh Chakravarthi
>
> **摘要:** Understanding the complex, multi-agent dynamics of urban traffic remains a fundamental challenge for video language models. This paper introduces Urban Dynamics VideoQA, a benchmark dataset that captures the unscripted real-world behavior of dynamic urban scenes. UDVideoQA is curated from 16 hours of traffic footage recorded at multiple city intersections under diverse traffic, weather, and lighting conditions. It employs an event-driven dynamic blur technique to ensure privacy preservation without compromising scene fidelity. Using a unified annotation pipeline, the dataset contains 28K question-answer pairs generated across 8 hours of densely annotated video, averaging one question per second. Its taxonomy follows a hierarchical reasoning level, spanning basic understanding and attribution to event reasoning, reverse reasoning, and counterfactual inference, enabling systematic evaluation of both visual grounding and causal reasoning. Comprehensive experiments benchmark 10 SOTA VideoLMs on UDVideoQA and 8 models on a complementary video question generation benchmark. Results reveal a persistent perception-reasoning gap, showing models that excel in abstract inference often fail with fundamental visual grounding. While models like Gemini Pro achieve the highest zero-shot accuracy, fine-tuning the smaller Qwen2.5-VL 7B model on UDVideoQA bridges this gap, achieving performance comparable to proprietary systems. In VideoQGen, Gemini 2.5 Pro, and Qwen3 Max generate the most relevant and complex questions, though all models exhibit limited linguistic diversity, underscoring the need for human-centric evaluation. The UDVideoQA suite, including the dataset, annotation tools, and benchmarks for both VideoQA and VideoQGen, provides a foundation for advancing robust, privacy-aware, and real-world multimodal reasoning. UDVideoQA is available at https://ud-videoqa.github.io/UD-VideoQA/UD-VideoQA/.
>
---
#### [new 021] UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling
- **分类: cs.CV**

- **简介: 该论文属于动态驾驶场景建模任务，解决长序列3D重建效率与精度问题。提出UFO方法，结合优化与前馈方法，实现高效4D场景重构。**

- **链接: [https://arxiv.org/pdf/2602.20943v1](https://arxiv.org/pdf/2602.20943v1)**

> **作者:** Kaiyuan Tan; Yingying Shen; Mingfei Tu; Haohui Zhu; Bing Wang; Guang Chen; Hangjun Ye; Haiyang Sun
>
> **摘要:** Dynamic driving scene reconstruction is critical for autonomous driving simulation and closed-loop learning. While recent feed-forward methods have shown promise for 3D reconstruction, they struggle with long-range driving sequences due to quadratic complexity in sequence length and challenges in modeling dynamic objects over extended durations. We propose UFO, a novel recurrent paradigm that combines the benefits of optimization-based and feed-forward methods for efficient long-range 4D reconstruction. Our approach maintains a 4D scene representation that is iteratively refined as new observations arrive, using a visibility-based filtering mechanism to select informative scene tokens and enable efficient processing of long sequences. For dynamic objects, we introduce an object pose-guided modeling approach that supports accurate long-range motion capture. Experiments on the Waymo Open Dataset demonstrate that our method significantly outperforms both per-scene optimization and existing feed-forward methods across various sequence lengths. Notably, our approach can reconstruct 16-second driving logs within 0.5 second while maintaining superior visual quality and geometric accuracy.
>
---
#### [new 022] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification
- **分类: cs.CV**

- **简介: 该论文属于少样本病理图像分类任务，解决标注数据稀缺导致的模型泛化问题。提出MUSE框架，通过样本级语义优化和多视图生成提升分类性能。**

- **链接: [https://arxiv.org/pdf/2602.20873v1](https://arxiv.org/pdf/2602.20873v1)**

> **作者:** Jiahao Xu; Sheng Huang; Xin Zhang; Zhixiong Nan; Jiajun Dong; Nankun Mu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** In computational pathology, few-shot whole slide image classification is primarily driven by the extreme scarcity of expert-labeled slides. Recent vision-language methods incorporate textual semantics generated by large language models, but treat these descriptions as static class-level priors that are shared across all samples and lack sample-wise refinement. This limits both the diversity and precision of visual-semantic alignment, hindering generalization under limited supervision. To overcome this, we propose the stochastic MUlti-view Semantic Enhancement (MUSE), a framework that first refines semantic precision via sample-wise adaptation and then enhances semantic richness through retrieval-augmented multi-view generation. Specifically, MUSE introduces Sample-wise Fine-grained Semantic Enhancement (SFSE), which yields a fine-grained semantic prior for each sample through MoE-based adaptive visual-semantic interaction. Guided by this prior, Stochastic Multi-view Model Optimization (SMMO) constructs an LLM-generated knowledge base of diverse pathological descriptions per class, then retrieves and stochastically integrates multiple matched textual views during training. These dynamically selected texts serve as enriched semantic supervisions to stochastically optimize the vision-language model, promoting robustness and mitigating overfitting. Experiments on three benchmark WSI datasets show that MUSE consistently outperforms existing vision-language baselines in few-shot settings, demonstrating that effective few-shot pathology learning requires not only richer semantic sources but also their active and sample-aware semantic optimization. Our code is available at: https://github.com/JiahaoXu-god/CVPR2026_MUSE.
>
---
#### [new 023] CAD-Prompted SAM3: Geometry-Conditioned Instance Segmentation for Industrial Objects
- **分类: cs.CV**

- **简介: 该论文属于实例分割任务，解决工业场景中对象描述困难的问题。通过CAD模型生成几何提示，提升分割准确性。**

- **链接: [https://arxiv.org/pdf/2602.20551v1](https://arxiv.org/pdf/2602.20551v1)**

> **作者:** Zhenran Tang; Rohan Nagabhirava; Changliu Liu
>
> **摘要:** Verbal-prompted segmentation is inherently limited by the expressiveness of natural language and struggles with uncommon, instance-specific, or difficult-to-describe objects: scenarios frequently encountered in manufacturing and 3D printing environments. While image exemplars provide an alternative, they primarily encode appearance cues such as color and texture, which are often unrelated to a part's geometric identity. In industrial settings, a single component may be produced in different materials, finishes, or colors, making appearance-based prompting unreliable. In contrast, such objects are typically defined by precise CAD models that capture their canonical geometry. We propose a CAD-prompted segmentation framework built on SAM3 that uses canonical multi-view renderings of a CAD model as prompt input. The rendered views provide geometry-based conditioning independent of surface appearance. The model is trained using synthetic data generated from mesh renderings in simulation under diverse viewpoints and scene contexts. Our approach enables single-stage, CAD-prompted mask prediction, extending promptable segmentation to objects that cannot be robustly described by language or appearance alone.
>
---
#### [new 024] De-rendering, Reasoning, and Repairing Charts with Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于可视化设计任务，旨在解决图表错误导致的误解问题。通过结合图像解析与语言模型，识别并修复图表设计缺陷，提供结构化改进建议。**

- **链接: [https://arxiv.org/pdf/2602.20291v1](https://arxiv.org/pdf/2602.20291v1)**

> **作者:** Valentin Bonas; Martin Sinnona; Viviana Siless; Emmanuel Iarussi
>
> **摘要:** Data visualizations are central to scientific communication, journalism, and everyday decision-making, yet they are frequently prone to errors that can distort interpretation or mislead audiences. Rule-based visualization linters can flag violations, but they miss context and do not suggest meaningful design changes. Directly querying general-purpose LLMs about visualization quality is unreliable: lacking training to follow visualization design principles, they often produce inconsistent or incorrect feedback. In this work, we introduce a framework that combines chart de-rendering, automated analysis, and iterative improvement to deliver actionable, interpretable feedback on visualization design. Our system reconstructs the structure of a chart from an image, identifies design flaws using vision-language reasoning, and proposes concrete modifications supported by established principles in visualization research. Users can selectively apply these improvements and re-render updated figures, creating a feedback loop that promotes both higher-quality visualizations and the development of visualization literacy. In our evaluation on 1,000 charts from the Chart2Code benchmark, the system generated 10,452 design recommendations, which clustered into 10 coherent categories (e.g., axis formatting, color accessibility, legend consistency). These results highlight the promise of LLM-driven recommendation systems for delivering structured, principle-based feedback on visualization design, opening the door to more intelligent and accessible authoring tools.
>
---
#### [new 025] Interaction-aware Representation Modeling with Co-occurrence Consistency for Egocentric Hand-Object Parsing
- **分类: cs.CV**

- **简介: 该论文属于第一人称视角下手物交互解析任务，旨在解决手物分割中的交互感知不足、物理不一致等问题，提出InterFormer模型提升准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.20597v1](https://arxiv.org/pdf/2602.20597v1)**

> **作者:** Yuejiao Su; Yi Wang; Lei Yao; Yawen Cui; Lap-Pui Chau
>
> **摘要:** A fine-grained understanding of egocentric human-environment interactions is crucial for developing next-generation embodied agents. One fundamental challenge in this area involves accurately parsing hands and active objects. While transformer-based architectures have demonstrated considerable potential for such tasks, several key limitations remain unaddressed: 1) existing query initialization mechanisms rely primarily on semantic cues or learnable parameters, demonstrating limited adaptability to changing active objects across varying input scenes; 2) previous transformer-based methods utilize pixel-level semantic features to iteratively refine queries during mask generation, which may introduce interaction-irrelevant content into the final embeddings; and 3) prevailing models are susceptible to "interaction illusion", producing physically inconsistent predictions. To address these issues, we propose an end-to-end Interaction-aware Transformer (InterFormer), which integrates three key components, i.e., a Dynamic Query Generator (DQG), a Dual-context Feature Selector (DFS), and the Conditional Co-occurrence (CoCo) loss. The DQG explicitly grounds query initialization in the spatial dynamics of hand-object contact, enabling targeted generation of interaction-aware queries for hands and various active objects. The DFS fuses coarse interactive cues with semantic features, thereby suppressing interaction-irrelevant noise and emphasizing the learning of interactive relationships. The CoCo loss incorporates hand-object relationship constraints to enhance physical consistency in prediction. Our model achieves state-of-the-art performance on both the EgoHOS and the challenging out-of-distribution mini-HOI4D datasets, demonstrating its effectiveness and strong generalization ability. Code and models are publicly available at https://github.com/yuggiehk/InterFormer.
>
---
#### [new 026] GSNR: Graph Smooth Null-Space Representation for Inverse Problems
- **分类: cs.CV; eess.IV; math.OC**

- **简介: 该论文提出GSNR方法，用于解决成像中的逆问题。通过构建图平滑的零空间表示，提升图像重建质量，改善收敛性与预测性。**

- **链接: [https://arxiv.org/pdf/2602.20328v1](https://arxiv.org/pdf/2602.20328v1)**

> **作者:** Romario Gualdrón-Hurtado; Roman Jacome; Rafael S. Suarez; Henry Arguello
>
> **备注:** 23 pages, 24 figures, Accepted to The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026
>
> **摘要:** Inverse problems in imaging are ill-posed, leading to infinitely many solutions consistent with the measurements due to the non-trivial null-space of the sensing matrix. Common image priors promote solutions on the general image manifold, such as sparsity, smoothness, or score function. However, as these priors do not constrain the null-space component, they can bias the reconstruction. Thus, we aim to incorporate meaningful null-space information in the reconstruction framework. Inspired by smooth image representation on graphs, we propose Graph-Smooth Null-Space Representation (GSNR), a mechanism that imposes structure only into the invisible component. Particularly, given a graph Laplacian, we construct a null-restricted Laplacian that encodes similarity between neighboring pixels in the null-space signal, and we design a low-dimensional projection matrix from the $p$-smoothest spectral graph modes (lowest graph frequencies). This approach has strong theoretical and practical implications: i) improved convergence via a null-only graph regularizer, ii) better coverage, how much null-space variance is captured by $p$ modes, and iii) high predictability, how well these modes can be inferred from the measurements. GSNR is incorporated into well-known inverse problem solvers, e.g., PnP, DIP, and diffusion solvers, in four scenarios: image deblurring, compressed sensing, demosaicing, and image super-resolution, providing consistent improvement of up to 4.3 dB over baseline formulations and up to 1 dB compared with end-to-end learned models in terms of PSNR.
>
---
#### [new 027] AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents
- **分类: cs.CV**

- **简介: 该论文属于文档伪造检测任务，旨在解决AI生成内容对金融和表单文档的篡改问题。通过构建包含4061张伪造图像的基准数据集，验证现有检测方法效果显著下降，揭示新挑战。**

- **链接: [https://arxiv.org/pdf/2602.20569v1](https://arxiv.org/pdf/2602.20569v1)**

> **作者:** Jiaqi Wu; Yuchen Zhou; Muduo Xu; Zisheng Liang; Simiao Ren; Jiayu Xue; Meige Yang; Siying Chen; Jingheng Huan
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** We present AIForge-Doc, the first dedicated benchmark targeting exclusively diffusion-model-based inpainting in financial and form documents with pixel-level annotation. Existing document forgery datasets rely on traditional digital editing tools (e.g., Adobe Photoshop, GIMP), creating a critical gap: state-of-the-art detectors are blind to the rapidly growing threat of AI-forged document fraud. AIForge-Doc addresses this gap by systematically forging numeric fields in real-world receipt and form images using two AI inpainting APIs -- Gemini 2.5 Flash Image and Ideogram v2 Edit -- yielding 4,061 forged images from four public document datasets (CORD, WildReceipt, SROIE, XFUND) across nine languages, annotated with pixel-precise tampered-region masks in DocTamper-compatible format. We benchmark three representative detectors -- TruFor, DocTamper, and a zero-shot GPT-4o judge -- and find that all existing methods degrade substantially: TruFor achieves AUC=0.751 (zero-shot, out-of-distribution) vs. AUC=0.96 on NIST16; DocTamper achieves AUC=0.563 vs. AUC=0.98 in-distribution, with pixel-level IoU=0.020; GPT-4o achieves only 0.509 -- essentially at chance -- confirming that AI-forged values are indistinguishable to automated detectors and VLMs. These results demonstrate that AIForge-Doc represents a qualitatively new and unsolved challenge for document forensics.
>
---
#### [new 028] gQIR: Generative Quanta Image Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于图像重建任务，解决在极低光条件下从稀疏光子数据中恢复高质量图像的问题。通过改进扩散模型以适应光子统计特性，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2602.20417v1](https://arxiv.org/pdf/2602.20417v1)**

> **作者:** Aryan Garg; Sizhuo Ma; Mohit Gupta
>
> **备注:** CVPR 2026
>
> **摘要:** Capturing high-quality images from only a few detected photons is a fundamental challenge in computational imaging. Single-photon avalanche diode (SPAD) sensors promise high-quality imaging in regimes where conventional cameras fail, but raw \emph{quanta frames} contain only sparse, noisy, binary photon detections. Recovering a coherent image from a burst of such frames requires handling alignment, denoising, and demosaicing (for color) under noise statistics far outside those assumed by standard restoration pipelines or modern generative models. We present an approach that adapts large text-to-image latent diffusion models to the photon-limited domain of quanta burst imaging. Our method leverages the structural and semantic priors of internet-scale diffusion models while introducing mechanisms to handle Bernoulli photon statistics. By integrating latent-space restoration with burst-level spatio-temporal reasoning, our approach produces reconstructions that are both photometrically faithful and perceptually pleasing, even under high-speed motion. We evaluate the method on synthetic benchmarks and new real-world datasets, including the first color SPAD burst dataset and a challenging \textit{Deforming (XD)} video benchmark. Across all settings, the approach substantially improves perceptual quality over classical and modern learning-based baselines, demonstrating the promise of adapting large generative priors to extreme photon-limited sensing. Code at \href{https://github.com/Aryan-Garg/gQIR}{https://github.com/Aryan-Garg/gQIR}.
>
---
#### [new 029] WildGHand: Learning Anti-Perturbation Gaussian Hand Avatars from Monocular In-the-Wild Videos
- **分类: cs.CV**

- **简介: 该论文属于3D手部重建任务，解决真实场景下手部姿态和形状的高精度重建问题。提出WildGHand框架，通过动态扰动解耦和感知扰动优化策略，提升在复杂环境中的重建效果。**

- **链接: [https://arxiv.org/pdf/2602.20556v1](https://arxiv.org/pdf/2602.20556v1)**

> **作者:** Hanhui Li; Xuan Huang; Wanquan Liu; Yuhao Cheng; Long Chen; Yiqiang Yan; Xiaodan Liang; Chenqiang Gao
>
> **摘要:** Despite recent progress in 3D hand reconstruction from monocular videos, most existing methods rely on data captured in well-controlled environments and therefore degrade in real-world settings with severe perturbations, such as hand-object interactions, extreme poses, illumination changes, and motion blur. To tackle these issues, we introduce WildGHand, an optimization-based framework that enables self-adaptive 3D Gaussian splatting on in-the-wild videos and produces high-fidelity hand avatars. WildGHand incorporates two key components: (i) a dynamic perturbation disentanglement module that explicitly represents perturbations as time-varying biases on 3D Gaussian attributes during optimization, and (ii) a perturbation-aware optimization strategy that generates per-frame anisotropic weighted masks to guide optimization. Together, these components allow the framework to identify and suppress perturbations across both spatial and temporal dimensions. We further curate a dataset of monocular hand videos captured under diverse perturbations to benchmark in-the-wild hand avatar reconstruction. Extensive experiments on this dataset and two public datasets demonstrate that WildGHand achieves state-of-the-art performance and substantially improves over its base model across multiple metrics (e.g., up to a $15.8\%$ relative gain in PSNR and a $23.1\%$ relative reduction in LPIPS). Our implementation and dataset are available at https://github.com/XuanHuang0/WildGHand.
>
---
#### [new 030] GA-Drive: Geometry-Appearance Decoupled Modeling for Free-viewpoint Driving Scene Generatio
- **分类: cs.CV**

- **简介: 该论文提出GA-Drive，用于生成自由视角的驾驶场景。任务是提升驾驶模拟器的逼真度与可编辑性，通过几何与外观解耦及扩散生成技术实现高质量新视角合成。**

- **链接: [https://arxiv.org/pdf/2602.20673v1](https://arxiv.org/pdf/2602.20673v1)**

> **作者:** Hao Zhang; Lue Fan; Qitai Wang; Wenbo Li; Zehuan Wu; Lewei Lu; Zhaoxiang Zhang; Hongsheng Li
>
> **摘要:** A free-viewpoint, editable, and high-fidelity driving simulator is crucial for training and evaluating end-to-end autonomous driving systems. In this paper, we present GA-Drive, a novel simulation framework capable of generating camera views along user-specified novel trajectories through Geometry-Appearance Decoupling and Diffusion-Based Generation. Given a set of images captured along a recorded trajectory and the corresponding scene geometry, GA-Drive synthesizes novel pseudo-views using geometry information. These pseudo-views are then transformed into photorealistic views using a trained video diffusion model. In this way, we decouple the geometry and appearance of scenes. An advantage of such decoupling is its support for appearance editing via state-of-the-art video-to-video editing techniques, while preserving the underlying geometry, enabling consistent edits across both original and novel trajectories. Extensive experiments demonstrate that GA-Drive substantially outperforms existing methods in terms of NTA-IoU, NTL-IoU, and FID scores.
>
---
#### [new 031] LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型计算成本高的问题。通过提出LESA框架，利用多阶段预测提升加速效果，保持生成质量。**

- **链接: [https://arxiv.org/pdf/2602.20497v1](https://arxiv.org/pdf/2602.20497v1)**

> **作者:** Peiliang Cai; Jiacheng Liu; Haowen Xu; Xinyu Wang; Chang Zou; Linfeng Zhang
>
> **摘要:** Diffusion models have achieved remarkable success in image and video generation tasks. However, the high computational demands of Diffusion Transformers (DiTs) pose a significant challenge to their practical deployment. While feature caching is a promising acceleration strategy, existing methods based on simple reusing or training-free forecasting struggle to adapt to the complex, stage-dependent dynamics of the diffusion process, often resulting in quality degradation and failing to maintain consistency with the standard denoising process. To address this, we propose a LEarnable Stage-Aware (LESA) predictor framework based on two-stage training. Our approach leverages a Kolmogorov-Arnold Network (KAN) to accurately learn temporal feature mappings from data. We further introduce a multi-stage, multi-expert architecture that assigns specialized predictors to different noise-level stages, enabling more precise and robust feature forecasting. Extensive experiments show our method achieves significant acceleration while maintaining high-fidelity generation. Experiments demonstrate 5.00x acceleration on FLUX.1-dev with minimal quality degradation (1.0% drop), 6.25x speedup on Qwen-Image with a 20.2% quality improvement over the previous SOTA (TaylorSeer), and 5.00x acceleration on HunyuanVideo with a 24.7% PSNR improvement over TaylorSeer. State-of-the-art performance on both text-to-image and text-to-video synthesis validates the effectiveness and generalization capability of our training-based framework across different models. Our code is included in the supplementary materials and will be released on GitHub.
>
---
#### [new 032] Object-Scene-Camera Decomposition and Recomposition for Data-Efficient Monocular 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单目3D目标检测任务，旨在解决训练数据不足和过拟合问题。通过分解并重新组合物体、场景和相机姿态，提升数据效率。**

- **链接: [https://arxiv.org/pdf/2602.20627v1](https://arxiv.org/pdf/2602.20627v1)**

> **作者:** Zhaonian Kuang; Rui Ding; Meng Yang; Xinhu Zheng; Gang Hua
>
> **备注:** IJCV
>
> **摘要:** Monocular 3D object detection (M3OD) is intrinsically ill-posed, hence training a high-performance deep learning based M3OD model requires a humongous amount of labeled data with complicated visual variation from diverse scenes, variety of objects and camera poses.However, we observe that, due to strong human bias, the three independent entities, i.e., object, scene, and camera pose, are always tightly entangled when an image is captured to construct training data. More specifically, specific 3D objects are always captured in particular scenes with fixed camera poses, and hence lacks necessary diversity. Such tight entanglement induces the challenging issues of insufficient utilization and overfitting to uniform training data. To mitigate this, we propose an online object-scene-camera decomposition and recomposition data manipulation scheme to more efficiently exploit the training data. We first fully decompose training images into textured 3D object point models and background scenes in an efficient computation and storage manner. We then continuously recompose new training images in each epoch by inserting the 3D objects into the freespace of the background scenes, and rendering them with perturbed camera poses from textured 3D point representation. In this way, the refreshed training data in all epochs can cover the full spectrum of independent object, scene, and camera pose combinations. This scheme can serve as a plug-and-play component to boost M3OD models, working flexibly with both fully and sparsely supervised settings. In the sparsely-supervised setting, objects closest to the ego-camera for all instances are sparsely annotated. We then can flexibly increase the annotated objects to control annotation cost. For validation, our method is widely applied to five representative M3OD models and evaluated on both the KITTI and the more complicated Waymo datasets.
>
---
#### [new 033] Beyond Human Performance: A Vision-Language Multi-Agent Approach for Quality Control in Pharmaceutical Manufacturing
- **分类: cs.CV**

- **简介: 该论文属于微生物检测任务，解决药品生产中菌落计数的自动化问题。通过结合深度学习与视觉语言模型，提升检测准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2602.20543v1](https://arxiv.org/pdf/2602.20543v1)**

> **作者:** Subhra Jyoti Mandal; Lara Rachidi; Puneet Jain; Matthieu Duvinage; Sander W. Timmer
>
> **摘要:** Colony-forming unit (CFU) detection is critical in pharmaceutical manufacturing, serving as a key component of Environmental Monitoring programs and ensuring compliance with stringent quality standards. Manual counting is labor-intensive and error-prone, while deep learning (DL) approaches, though accurate, remain vulnerable to sample quality variations and artifacts. Building on our earlier CNN-based framework (Beznik et al., 2020), we evaluated YOLOv5, YOLOv7, and YOLOv8 for CFU detection; however, these achieved only 97.08 percent accuracy, insufficient for pharmaceutical-grade requirements. A custom Detectron2 model trained on GSK's dataset of over 50,000 Petri dish images achieved 99 percent detection rate with 2 percent false positives and 0.6 percent false negatives. Despite high validation accuracy, Detectron2 performance degrades on outlier cases including contaminated plates, plastic artifacts, or poor optical clarity. To address this, we developed a multi-agent framework combining DL with vision-language models (VLMs). The VLM agent first classifies plates as valid or invalid. For valid samples, both DL and VLM agents independently estimate colony counts. When predictions align within 5 percent, results are automatically recorded in Postgres and SAP; otherwise, samples are routed for expert review. Expert feedback enables continuous retraining and self-improvement. Initial DL-based automation reduced human verification by 50 percent across vaccine manufacturing sites. With VLM integration, this increased to 85 percent, delivering significant operational savings. The proposed system provides a scalable, auditable, and regulation-ready solution for microbiological quality control, advancing automation in biopharmaceutical production.
>
---
#### [new 034] Cycle-Consistent Tuning for Layered Image Decomposition
- **分类: cs.CV**

- **简介: 该论文属于图像分解任务，旨在解决真实图像中视觉层的解耦问题，特别是Logo与背景的分离。通过微调扩散模型并引入循环一致性策略，提升分解与重构的准确性。**

- **链接: [https://arxiv.org/pdf/2602.20989v1](https://arxiv.org/pdf/2602.20989v1)**

> **作者:** Zheng Gu; Min Lu; Zhida Sun; Dani Lischinski; Daniel Cohen-O; Hui Huang
>
> **备注:** Accepted to CVPR 2026. Project page: https://vcc.tech/research/2026/ImgDecom
>
> **摘要:** Disentangling visual layers in real-world images is a persistent challenge in vision and graphics, as such layers often involve non-linear and globally coupled interactions, including shading, reflection, and perspective distortion. In this work, we present an in-context image decomposition framework that leverages large diffusion foundation models for layered separation. We focus on the challenging case of logo-object decomposition, where the goal is to disentangle a logo from the surface on which it appears while faithfully preserving both layers. Our method fine-tunes a pretrained diffusion model via lightweight LoRA adaptation and introduces a cycle-consistent tuning strategy that jointly trains decomposition and composition models, enforcing reconstruction consistency between decomposed and recomposed images. This bidirectional supervision substantially enhances robustness in cases where the layers exhibit complex interactions. Furthermore, we introduce a progressive self-improving process, which iteratively augments the training set with high-quality model-generated examples to refine performance. Extensive experiments demonstrate that our approach achieves accurate and coherent decompositions and also generalizes effectively across other decomposition types, suggesting its potential as a unified framework for layered image decomposition.
>
---
#### [new 035] OmniOCR: Generalist OCR for Ethnic Minority Languages
- **分类: cs.CV**

- **简介: 该论文提出OmniOCR，解决民族语言OCR的低资源和零样本问题。通过动态LoRA和稀疏正则化，提升模型适应性与效率。**

- **链接: [https://arxiv.org/pdf/2602.21042v1](https://arxiv.org/pdf/2602.21042v1)**

> **作者:** Bonan Liu; Zeyu Zhang; Bingbing Meng; Han Wang; Hanshuo Zhang; Chengping Wang; Daji Ergu; Ying Cai
>
> **摘要:** Optical character recognition (OCR) has advanced rapidly with deep learning and multimodal models, yet most methods focus on well-resourced scripts such as Latin and Chinese. Ethnic minority languages remain underexplored due to complex writing systems, scarce annotations, and diverse historical and modern forms, making generalization in low-resource or zero-shot settings challenging. To address these challenges, we present OmniOCR, a universal framework for ethnic minority scripts. OmniOCR introduces Dynamic Low-Rank Adaptation (Dynamic LoRA) to allocate model capacity across layers and scripts, enabling effective adaptation while preserving knowledge.A sparsity regularization prunes redundant updates, ensuring compact and efficient adaptation without extra inference cost. Evaluations on TibetanMNIST, Shui, ancient Yi, and Dongba show that OmniOCR outperforms zero-shot foundation models and standard post training, achieving state-of-the-art accuracy with superior parameter efficiency, and compared with the state-of-the-art baseline models, it improves accuracy by 39%-66% on these four datasets. Code: https://github.com/AIGeeksGroup/OmniOCR.
>
---
#### [new 036] How Do Inpainting Artifacts Propagate to Language?
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉修复误差如何影响多模态语言生成。任务为视觉-语言模型中的图像修复对文本生成的影响。解决的问题是修复质量如何影响生成文本的准确性。工作包括构建诊断框架，分析重建质量与文本质量的关系。**

- **链接: [https://arxiv.org/pdf/2602.20520v1](https://arxiv.org/pdf/2602.20520v1)**

> **作者:** Pratham Yashwante; Davit Abrahamyan; Shresth Grover; Sukruth Rao
>
> **摘要:** We study how visual artifacts introduced by diffusion-based inpainting affect language generation in vision-language models. We use a two-stage diagnostic setup in which masked image regions are reconstructed and then provided to captioning models, enabling controlled comparisons between captions generated from original and reconstructed inputs. Across multiple datasets, we analyze the relationship between reconstruction fidelity and downstream caption quality. We observe consistent associations between pixel-level and perceptual reconstruction metrics and both lexical and semantic captioning performance. Additional analysis of intermediate visual representations and attention patterns shows that inpainting artifacts lead to systematic, layer-dependent changes in model behavior. Together, these results provide a practical diagnostic framework for examining how visual reconstruction quality influences language generation in multimodal systems.
>
---
#### [new 037] LUMEN: Longitudinal Multi-Modal Radiology Model for Prognosis and Diagnosis
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出LUMEN模型，用于纵向胸片的诊断与预后分析，解决放射学中时间变化分析效率低的问题。通过多图像和多任务微调提升性能。**

- **链接: [https://arxiv.org/pdf/2602.21142v1](https://arxiv.org/pdf/2602.21142v1)**

> **作者:** Zhifan Jiang; Dong Yang; Vishwesh Nath; Abhijeet Parida; Nishad P. Kulkarni; Ziyue Xu; Daguang Xu; Syed Muhammad Anwar; Holger R. Roth; Marius George Linguraru
>
> **备注:** Accepted to IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** Large vision-language models (VLMs) have evolved from general-purpose applications to specialized use cases such as in the clinical domain, demonstrating potential for decision support in radiology. One promising application is assisting radiologists in decision-making by the analysis of radiology imaging data such as chest X-rays (CXR) via a visual and natural language question-answering (VQA) interface. When longitudinal imaging is available, radiologists analyze temporal changes, which are essential for accurate diagnosis and prognosis. The manual longitudinal analysis is a time-consuming process, motivating the development of a training framework that can provide prognostic capabilities. We introduce a novel training framework LUMEN, that is optimized for longitudinal CXR interpretation, leveraging multi-image and multi-task instruction fine-tuning to enhance prognostic and diagnostic performance. We conduct experiments on the publicly available MIMIC-CXR and its associated Medical-Diff-VQA datasets. We further formulate and construct a novel instruction-following dataset incorporating longitudinal studies, enabling the development of a prognostic VQA task. Our method demonstrates significant improvements over baseline models in diagnostic VQA tasks, and more importantly, shows promising potential for prognostic capabilities. These results underscore the value of well-designed, instruction-tuned VLMs in enabling more accurate and clinically meaningful radiological interpretation of longitudinal radiological imaging data.
>
---
#### [new 038] From Perception to Action: An Interactive Benchmark for Vision Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉推理任务，旨在解决VLM在物理结构理解与长期规划上的不足。提出CHAIN基准，评估模型在物理约束下的动作规划与执行能力。**

- **链接: [https://arxiv.org/pdf/2602.21015v1](https://arxiv.org/pdf/2602.21015v1)**

> **作者:** Yuhao Wu; Maojia Song; Yihuai Lan; Lei Wang; Zhiqiang Hu; Yao Xiao; Heng Zhou; Weihua Zheng; Dylan Raharja; Soujanya Poria; Roy Ka-Wei Lee
>
> **备注:** Work in processing. Website: https://social-ai-studio.github.io/CHAIN/
>
> **摘要:** Understanding the physical structure is essential for real-world applications such as embodied agents, interactive design, and long-horizon manipulation. Yet, prevailing Vision-Language Model (VLM) evaluations still center on structure-agnostic, single-turn setups (e.g., VQA), which fail to assess agents' ability to reason about how geometry, contact, and support relations jointly constrain what actions are possible in a dynamic environment. To address this gap, we introduce the Causal Hierarchy of Actions and Interactions (CHAIN) benchmark, an interactive 3D, physics-driven testbed designed to evaluate whether models can understand, plan, and execute structured action sequences grounded in physical constraints. CHAIN shifts evaluation from passive perception to active problem solving, spanning tasks such as interlocking mechanical puzzles and 3D stacking and packing. We conduct a comprehensive study of state-of-the-art VLMs and diffusion-based models under unified interactive settings. Our results show that top-performing models still struggle to internalize physical structure and causal constraints, often failing to produce reliable long-horizon plans and cannot robustly translate perceived structure into effective actions. The project is available at https://social-ai-studio.github.io/CHAIN/.
>
---
#### [new 039] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DCQ框架，解决图像数据集存储过大问题。通过减少颜色空间冗余，实现数据集压缩，同时保持训练所需信息。属于数据集压缩任务。**

- **链接: [https://arxiv.org/pdf/2602.20650v1](https://arxiv.org/pdf/2602.20650v1)**

> **作者:** Chenyue Yu; Lingao Xiao; Jinhong Deng; Ivor W. Tsang; Yang He
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large-scale image datasets are fundamental to deep learning, but their high storage demands pose challenges for deployment in resource-constrained environments. While existing approaches reduce dataset size by discarding samples, they often ignore the significant redundancy within each image -- particularly in the color space. To address this, we propose Dataset Color Quantization (DCQ), a unified framework that compresses visual datasets by reducing color-space redundancy while preserving information crucial for model training. DCQ achieves this by enforcing consistent palette representations across similar images, selectively retaining semantically important colors guided by model perception, and maintaining structural details necessary for effective feature learning. Extensive experiments across CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet-1K show that DCQ significantly improves training performance under aggressive compression, offering a scalable and robust solution for dataset-level storage reduction. Code is available at \href{https://github.com/he-y/Dataset-Color-Quantization}{https://github.com/he-y/Dataset-Color-Quantization}.
>
---
#### [new 040] BoxSplitGen: A Generative Model for 3D Part Bounding Boxes in Varying Granularity
- **分类: cs.CV**

- **简介: 该论文提出BoxSplitGen，用于3D形状生成任务，解决从粗到细的细节生成问题。通过迭代分割边界框和生成形状，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.20666v1](https://arxiv.org/pdf/2602.20666v1)**

> **作者:** Juil Koo; Wei-Tung Lin; Chanho Park; Chanhyeok Park; Minhyuk Sung
>
> **备注:** Project page: https://boxsplitgen.github.io
>
> **摘要:** Human creativity follows a perceptual process, moving from abstract ideas to finer details during creation. While 3D generative models have advanced dramatically, models specifically designed to assist human imagination in 3D creation -- particularly for detailing abstractions from coarse to fine -- have not been explored. We propose a framework that enables intuitive and interactive 3D shape generation by iteratively splitting bounding boxes to refine the set of bounding boxes. The main technical components of our framework are two generative models: the box-splitting generative model and the box-to-shape generative model. The first model, named BoxSplitGen, generates a collection of 3D part bounding boxes with varying granularity by iteratively splitting coarse bounding boxes. It utilizes part bounding boxes created through agglomerative merging and learns the reverse of the merging process -- the splitting sequences. The model consists of two main components: the first learns the categorical distribution of the box to be split, and the second learns the distribution of the two new boxes, given the set of boxes and the indication of which box to split. The second model, the box-to-shape generative model, is trained by leveraging the 3D shape priors learned by an existing 3D diffusion model while adapting the model to incorporate bounding box conditioning. In our experiments, we demonstrate that the box-splitting generative model outperforms token prediction models and the inpainting approach with an unconditional diffusion model. Also, we show that our box-to-shape model, based on a state-of-the-art 3D diffusion model, provides superior results compared to a previous model.
>
---
#### [new 041] SurgAtt-Tracker: Online Surgical Attention Tracking via Temporal Proposal Reranking and Motion-Aware Refinement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于手术注意力跟踪任务，解决术中视野引导问题。提出SurgAtt-Tracker框架，通过时序重排序和运动感知优化，实现稳定注意力追踪，并构建大规模基准数据集。**

- **链接: [https://arxiv.org/pdf/2602.20636v1](https://arxiv.org/pdf/2602.20636v1)**

> **作者:** Rulin Zhou; Guankun Wang; An Wang; Yujie Ma; Lixin Ouyang; Bolin Cui; Junyan Li; Chaowei Zhu; Mingyang Li; Ming Chen; Xiaopin Zhong; Peng Lu; Jiankun Wang; Xianming Liu; Hongliang Ren
>
> **摘要:** Accurate and stable field-of-view (FoV) guidance is critical for safe and efficient minimally invasive surgery, yet existing approaches often conflate visual attention estimation with downstream camera control or rely on direct object-centric assumptions. In this work, we formulate surgical attention tracking as a spatio-temporal learning problem and model surgeon focus as a dense attention heatmap, enabling continuous and interpretable frame-wise FoV guidance. We propose SurgAtt-Tracker, a holistic framework that robustly tracks surgical attention by exploiting temporal coherence through proposal-level reranking and motion-aware refinement, rather than direct regression. To support systematic training and evaluation, we introduce SurgAtt-1.16M, a large-scale benchmark with a clinically grounded annotation protocol that enables comprehensive heatmap-based attention analysis across procedures and institutions. Extensive experiments on multiple surgical datasets demonstrate that SurgAtt-Tracker consistently achieves state-of-the-art performance and strong robustness under occlusion, multi-instrument interference, and cross-domain settings. Beyond attention tracking, our approach provides a frame-wise FoV guidance signal that can directly support downstream robotic FoV planning and automatic camera control.
>
---
#### [new 042] Robust Spiking Neural Networks Against Adversarial Attacks
- **分类: cs.CV**

- **简介: 该论文属于对抗攻击防御任务，旨在提升SNN的鲁棒性。针对阈值附近神经元易受干扰的问题，提出TGO方法，通过优化损失函数和引入噪声增强稳定性。**

- **链接: [https://arxiv.org/pdf/2602.20548v1](https://arxiv.org/pdf/2602.20548v1)**

> **作者:** Shuai Wang; Malu Zhang; Yulin Jiang; Dehao Zhang; Ammar Belatreche; Yu Liang; Yimeng Shan; Zijian Zhou; Yang Yang; Haizhou Li
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Spiking Neural Networks (SNNs) represent a promising paradigm for energy-efficient neuromorphic computing due to their bio-plausible and spike-driven characteristics. However, the robustness of SNNs in complex adversarial environments remains significantly constrained. In this study, we theoretically demonstrate that those threshold-neighboring spiking neurons are the key factors limiting the robustness of directly trained SNNs. We find that these neurons set the upper limits for the maximum potential strength of adversarial attacks and are prone to state-flipping under minor disturbances. To address this challenge, we propose a Threshold Guarding Optimization (TGO) method, which comprises two key aspects. First, we incorporate additional constraints into the loss function to move neurons' membrane potentials away from their thresholds. It increases SNNs' gradient sparsity, thereby reducing the theoretical upper bound of adversarial attacks. Second, we introduce noisy spiking neurons to transition the neuronal firing mechanism from deterministic to probabilistic, decreasing their state-flipping probability due to minor disturbances. Extensive experiments conducted in standard adversarial scenarios prove that our method significantly enhances the robustness of directly trained SNNs. These findings pave the way for advancing more reliable and secure neuromorphic computing in real-world applications.
>
---
#### [new 043] On the Explainability of Vision-Language Models in Art History
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型在艺术史中的可解释性，探讨XAI方法如何揭示CLIP的视觉推理，旨在提升其在艺术分析中的透明度与可信度。**

- **链接: [https://arxiv.org/pdf/2602.20853v1](https://arxiv.org/pdf/2602.20853v1)**

> **作者:** Stefanie Schneider
>
> **摘要:** Vision-Language Models (VLMs) transfer visual and textual data into a shared embedding space. In so doing, they enable a wide range of multimodal tasks, while also raising critical questions about the nature of machine 'understanding.' In this paper, we examine how Explainable Artificial Intelligence (XAI) methods can render the visual reasoning of a VLM - namely, CLIP - legible in art-historical contexts. To this end, we evaluate seven methods, combining zero-shot localization experiments with human interpretability studies. Our results indicate that, while these methods capture some aspects of human interpretation, their effectiveness hinges on the conceptual stability and representational availability of the examined categories.
>
---
#### [new 044] AnimeAgent: Is the Multi-Agent via Image-to-Video models a Good Disney Storytelling Artist?
- **分类: cs.CV**

- **简介: 该论文属于定制分镜生成任务，解决静态模型表达不足、无法迭代修正和评估不准确的问题。提出AnimeAgent框架，利用图像到视频模型提升一致性与表现力。**

- **链接: [https://arxiv.org/pdf/2602.20664v1](https://arxiv.org/pdf/2602.20664v1)**

> **作者:** Hailong Yan; Shice Liu; Tao Wang; Xiangtao Zhang; Yijie Zhong; Jinwei Chen; Le Zhang; Bo Li
>
> **备注:** Tech Report
>
> **摘要:** Custom Storyboard Generation (CSG) aims to produce high-quality, multi-character consistent storytelling. Current approaches based on static diffusion models, whether used in a one-shot manner or within multi-agent frameworks, face three key limitations: (1) Static models lack dynamic expressiveness and often resort to "copy-paste" pattern. (2) One-shot inference cannot iteratively correct missing attributes or poor prompt adherence. (3) Multi-agents rely on non-robust evaluators, ill-suited for assessing stylized, non-realistic animation. To address these, we propose AnimeAgent, the first Image-to-Video (I2V)-based multi-agent framework for CSG. Inspired by Disney's "Combination of Straight Ahead and Pose to Pose" workflow, AnimeAgent leverages I2V's implicit motion prior to enhance consistency and expressiveness, while a mixed subjective-objective reviewer enables reliable iterative refinement. We also collect a human-annotated CSG benchmark with ground-truth. Experiments show AnimeAgent achieves SOTA performance in consistency, prompt fidelity, and stylization.
>
---
#### [new 045] SPRITETOMESH: Automatic Mesh Generation for 2D Skeletal Animation Using Learned Segmentation and Contour-Aware Vertex Placement
- **分类: cs.CV**

- **简介: 该论文属于2D动画网格生成任务，解决手动创建骨骼动画网格耗时问题。通过结合学习分割与算法定位，实现自动网格生成。**

- **链接: [https://arxiv.org/pdf/2602.21153v1](https://arxiv.org/pdf/2602.21153v1)**

> **作者:** Bastien Gimbert
>
> **备注:** 11 pages, 17 figures. Code available at https://github.com/BastienGimbert/SpriteToMesh
>
> **摘要:** We present SPRITETOMESH, a fully automatic pipeline for converting 2D game sprite images into triangle meshes compatible with skeletal animation frameworks such as Spine2D. Creating animation-ready meshes is traditionally a tedious manual process requiring artists to carefully place vertices along visual boundaries, a task that typically takes 15-60 minutes per sprite. Our method addresses this through a hybrid learned-algorithmic approach. A segmentation network (EfficientNet-B0 encoder with U-Net decoder) trained on over 100,000 sprite-mask pairs from 172 games achieves an IoU of 0.87, providing accurate binary masks from arbitrary input images. From these masks, we extract exterior contour vertices using Douglas-Peucker simplification with adaptive arc subdivision, and interior vertices along visual boundaries detected via bilateral-filtered multi-channel Canny edge detection with contour-following placement. Delaunay triangulation with mask-based centroid filtering produces the final mesh. Through controlled experiments, we demonstrate that direct vertex position prediction via neural network heatmap regression is fundamentally not viable for this task: the heatmap decoder consistently fails to converge (loss plateau at 0.061) while the segmentation decoder trains normally under identical conditions. We attribute this to the inherently artistic nature of vertex placement - the same sprite can be meshed validly in many different ways. This negative result validates our hybrid design: learned segmentation where ground truth is unambiguous, algorithmic placement where domain heuristics are appropriate. The complete pipeline processes a sprite in under 3 seconds, representing a speedup of 300x-1200x over manual creation. We release our trained model to the game development community.
>
---
#### [new 046] PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning
- **分类: cs.CV**

- **简介: 该论文提出PFGNet，用于高效时空预测任务。针对传统模型效率低、固定感受野的问题，引入频率引导的门控机制，实现动态感知运动模式，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2602.20537v1](https://arxiv.org/pdf/2602.20537v1)**

> **作者:** Xinyong Cai; Changbin Sun; Yong Wang; Hongyu Yang; Yuankai Wu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Spatiotemporal predictive learning (STPL) aims to forecast future frames from past observations and is essential across a wide range of applications. Compared with recurrent or hybrid architectures, pure convolutional models offer superior efficiency and full parallelism, yet their fixed receptive fields limit their ability to adaptively capture spatially varying motion patterns. Inspired by biological center-surround organization and frequency-selective signal processing, we propose PFGNet, a fully convolutional framework that dynamically modulates receptive fields through pixel-wise frequency-guided gating. The core Peripheral Frequency Gating (PFG) block extracts localized spectral cues and adaptively fuses multi-scale large-kernel peripheral responses with learnable center suppression, effectively forming spatially adaptive band-pass filters. To maintain efficiency, all large kernels are decomposed into separable 1D convolutions ($1 \times k$ followed by $k \times 1$), reducing per-channel computational cost from $O(k^2)$ to $O(2k)$. PFGNet enables structure-aware spatiotemporal modeling without recurrence or attention. Experiments on Moving MNIST, TaxiBJ, Human3.6M, and KTH show that PFGNet delivers SOTA or near-SOTA forecasting performance with substantially fewer parameters and FLOPs. Our code is available at https://github.com/fhjdqaq/PFGNet.
>
---
#### [new 047] RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于4D场景重建任务，旨在解决动态环境中SLAM的不确定性问题。通过引入时间因素和动态感知机制，提升跟踪与重建精度。**

- **链接: [https://arxiv.org/pdf/2602.20807v1](https://arxiv.org/pdf/2602.20807v1)**

> **作者:** Yangfan Zhao; Hanwei Zhang; Ke Huang; Qiufeng Wang; Zhenzhou Shao; Dengyu Wu
>
> **摘要:** Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io
>
---
#### [new 048] VII: Visual Instruction Injection for Jailbreaking Image-to-Video Generation Models
- **分类: cs.CV**

- **简介: 该论文属于图像到视频生成任务，旨在解决视觉指令注入带来的安全风险。提出VII框架，通过视觉指令隐匿恶意意图，提升攻击成功率并降低拒绝率。**

- **链接: [https://arxiv.org/pdf/2602.20999v1](https://arxiv.org/pdf/2602.20999v1)**

> **作者:** Bowen Zheng; Yongli Xiang; Ziming Hong; Zerong Lin; Chaojian Yu; Tongliang Liu; Xinge You
>
> **备注:** Project page: https://Zbwwwwwwww.github.io/VII
>
> **摘要:** Image-to-Video (I2V) generation models, which condition video generation on reference images, have shown emerging visual instruction-following capability, allowing certain visual cues in reference images to act as implicit control signals for video generation. However, this capability also introduces a previously overlooked risk: adversaries may exploit visual instructions to inject malicious intent through the image modality. In this work, we uncover this risk by proposing Visual Instruction Injection (VII), a training-free and transferable jailbreaking framework that intentionally disguises the malicious intent of unsafe text prompts as benign visual instructions in the safe reference image. Specifically, VII coordinates a Malicious Intent Reprogramming module to distill malicious intent from unsafe text prompts while minimizing their static harmfulness, and a Visual Instruction Grounding module to ground the distilled intent onto a safe input image by rendering visual instructions that preserve semantic consistency with the original unsafe text prompt, thereby inducing harmful content during I2V generation. Empirically, our extensive experiments on four state-of-the-art commercial I2V models (Kling-v2.5-turbo, Gemini Veo-3.1, Seedance-1.5-pro, and PixVerse-V5) demonstrate that VII achieves Attack Success Rates of up to 83.5% while reducing Refusal Rates to near zero, significantly outperforming existing baselines.
>
---
#### [new 049] NGL-Prompter: Training-Free Sewing Pattern Estimation from a Single Image
- **分类: cs.CV**

- **简介: 该论文属于服装设计任务，旨在从单张图像中无训练地估计缝制图案。针对数据不足和泛化能力差的问题，提出NGL-Prompter方法，通过自然语言桥梁实现准确的缝制参数提取。**

- **链接: [https://arxiv.org/pdf/2602.20700v1](https://arxiv.org/pdf/2602.20700v1)**

> **作者:** Anna Badalyan; Pratheba Selvaraju; Giorgio Becherini; Omid Taheri; Victoria Fernandez Abrevaya; Michael Black
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Estimating sewing patterns from images is a practical approach for creating high-quality 3D garments. Due to the lack of real-world pattern-image paired data, prior approaches fine-tune large vision language models (VLMs) on synthetic garment datasets generated by randomly sampling from a parametric garment model GarmentCode. However, these methods often struggle to generalize to in-the-wild images, fail to capture real-world correlations between garment parts, and are typically restricted to single-layer outfits. In contrast, we observe that VLMs are effective at describing garments in natural language, yet perform poorly when asked to directly regress GarmentCode parameters from images. To bridge this gap, we propose NGL (Natural Garment Language), a novel intermediate language that restructures GarmentCode into a representation more understandable to language models. Leveraging this language, we introduce NGL-Prompter, a training-free pipeline that queries large VLMs to extract structured garment parameters, which are then deterministically mapped to valid GarmentCode. We evaluate our method on the Dress4D, CloSe and a newly collected dataset of approximately 5,000 in-the-wild fashion images. Our approach achieves state-of-the-art performance on standard geometry metrics and is strongly preferred in both human and GPT-based perceptual evaluations compared to existing baselines. Furthermore, NGL-prompter can recover multi-layer outfits whereas competing methods focus mostly on single-layer garments, highlighting its strong generalization to real-world images even with occluded parts. These results demonstrate that accurate sewing pattern reconstruction is possible without costly model training. Our code and data will be released for research use.
>
---
#### [new 050] Efficient and Explainable End-to-End Autonomous Driving via Masked Vision-Language-Action Diffusion
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，解决推理延迟、动作精度和可解释性问题。提出MVLAD-AD框架，通过掩码视觉-语言-动作扩散模型实现高效且可解释的路径规划。**

- **链接: [https://arxiv.org/pdf/2602.20577v1](https://arxiv.org/pdf/2602.20577v1)**

> **作者:** Jiaru Zhang; Manav Gagvani; Can Cui; Juntong Peng; Ruqi Zhang; Ziran Wang
>
> **摘要:** Large Language Models (LLMs) and Vision-Language Models (VLMs) have emerged as promising candidates for end-to-end autonomous driving. However, these models typically face challenges in inference latency, action precision, and explainability. Existing autoregressive approaches struggle with slow token-by-token generation, while prior diffusion-based planners often rely on verbose, general-purpose language tokens that lack explicit geometric structure. In this work, we propose Masked Vision-Language-Action Diffusion for Autonomous Driving (MVLAD-AD), a novel framework designed to bridge the gap between efficient planning and semantic explainability via a masked vision-language-action diffusion model. Unlike methods that force actions into the language space, we introduce a discrete action tokenization strategy that constructs a compact codebook of kinematically feasible waypoints from real-world driving distributions. Moreover, we propose geometry-aware embedding learning to ensure that embeddings in the latent space approximate physical geometric metrics. Finally, an action-priority decoding strategy is introduced to prioritize trajectory generation. Extensive experiments on nuScenes and derived benchmarks demonstrate that MVLAD-AD achieves superior efficiency and outperforms state-of-the-art autoregressive and diffusion baselines in planning precision, while providing high-fidelity and explainable reasoning.
>
---
#### [new 051] SD4R: Sparse-to-Dense Learning for 3D Object Detection with 4D Radar
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决4D雷达点云稀疏和噪声问题。提出SD4R框架，通过点云增强和特征提取提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.20653v1](https://arxiv.org/pdf/2602.20653v1)**

> **作者:** Xiaokai Bai; Jiahao Cheng; Songkai Wang; Yixuan Luo; Lianqing Zheng; Xiaohan Zhang; Si-Yuan Cao; Hui-Liang Shen
>
> **备注:** 7 pages, 5 figures, 4 tables
>
> **摘要:** 4D radar measurements offer an affordable and weather-robust solution for 3D perception. However, the inherent sparsity and noise of radar point clouds present significant challenges for accurate 3D object detection, underscoring the need for effective and robust point clouds densification. Despite recent progress, existing densification methods often fail to address the extreme sparsity of 4D radar point clouds and exhibit limited robustness when processing scenes with a small number of points. In this paper, we propose SD4R, a novel framework that transforms sparse radar point clouds into dense representations. SD4R begins by utilizing a foreground point generator (FPG) to mitigate noise propagation and produce densified point clouds. Subsequently, a logit-query encoder (LQE) enhances conventional pillarization, resulting in robust feature representations. Through these innovations, our SD4R demonstrates strong capability in both noise reduction and foreground point densification. Extensive experiments conducted on the publicly available View-of-Delft dataset demonstrate that SD4R achieves state-of-the-art performance. Source code is available at https://github.com/lancelot0805/SD4R.
>
---
#### [new 052] VISION-ICE: Video-based Interpretation and Spatial Identification of Arrhythmia Origins via Neural Networks in Intracardiac Echocardiography
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于心律失常定位任务，旨在解决传统方法耗时费力的问题。通过AI分析ICE视频，实现心律失常源的自动分类与定位。**

- **链接: [https://arxiv.org/pdf/2602.20165v1](https://arxiv.org/pdf/2602.20165v1)**

> **作者:** Dorsa EPMoghaddam; Feng Gao; Drew Bernard; Kavya Sinha; Mehdi Razavi; Behnaam Aazhang
>
> **备注:** 8 pages, 3 figures, 3 tabels
>
> **摘要:** Contemporary high-density mapping techniques and preoperative CT/MRI remain time and resource intensive in localizing arrhythmias. AI has been validated as a clinical decision aid in providing accurate, rapid real-time analysis of echocardiographic images. Building on this, we propose an AI-enabled framework that leverages intracardiac echocardiography (ICE), a routine part of electrophysiology procedures, to guide clinicians toward areas of arrhythmogenesis and potentially reduce procedural time. Arrhythmia source localization is formulated as a three-class classification task, distinguishing normal sinus rhythm, left-sided, and right-sided arrhythmias, based on ICE video data. We developed a 3D Convolutional Neural Network trained to discriminate among the three aforementioned classes. In ten-fold cross-validation, the model achieved a mean accuracy of 66.2% when evaluated on four previously unseen patients (substantially outperforming the 33.3% random baseline). These results demonstrate the feasibility and clinical promise of using ICE videos combined with deep learning for automated arrhythmia localization. Leveraging ICE imaging could enable faster, more targeted electrophysiological interventions and reduce the procedural burden of cardiac ablation. Future work will focus on expanding the dataset to improve model robustness and generalizability across diverse patient populations.
>
---
#### [new 053] SceMoS: Scene-Aware 3D Human Motion Synthesis by Planning with Geometry-Grounded Tokens
- **分类: cs.CV**

- **简介: 该论文属于3D人体运动合成任务，解决文本驱动运动在真实场景中的语义与物理可行性问题。提出SceMoS框架，利用2D场景表示替代3D数据，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2602.20476v1](https://arxiv.org/pdf/2602.20476v1)**

> **作者:** Anindita Ghosh; Vladislav Golyanik; Taku Komura; Philipp Slusallek; Christian Theobalt; Rishabh Dabral
>
> **备注:** 13 pages, 6 figures, 4 tables
>
> **摘要:** Synthesizing text-driven 3D human motion within realistic scenes requires learning both semantic intent ("walk to the couch") and physical feasibility (e.g., avoiding collisions). Current methods use generative frameworks that simultaneously learn high-level planning and low-level contact reasoning, and rely on computationally expensive 3D scene data such as point clouds or voxel occupancy grids. We propose SceMoS, a scene-aware motion synthesis framework that shows that structured 2D scene representations can serve as a powerful alternative to full 3D supervision in physically grounded motion synthesis. SceMoS disentangles global planning from local execution using lightweight 2D cues and relying on (1) a text-conditioned autoregressive global motion planner that operates on a bird's-eye-view (BEV) image rendered from an elevated corner of the scene, encoded with DINOv2 features, as the scene representation, and (2) a geometry-grounded motion tokenizer trained via a conditional VQ-VAE, that uses 2D local scene heightmap, thus embedding surface physics directly into a discrete vocabulary. This 2D factorization reaches an efficiency-fidelity trade-off: BEV semantics capture spatial layout and affordance for global reasoning, while local heightmaps enforce fine-grained physical adherence without full 3D volumetric reasoning. SceMoS achieves state-of-the-art motion realism and contact accuracy on the TRUMANS benchmark, reducing the number of trainable parameters for scene encoding by over 50%, showing that 2D scene cues can effectively ground 3D human-scene interaction.
>
---
#### [new 054] Federated Learning for Cross-Modality Medical Image Segmentation via Augmentation-Driven Generalization
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决跨模态泛化问题。通过增强策略提升联邦学习模型在不同影像模态间的性能，实现隐私保护下的高效分割。**

- **链接: [https://arxiv.org/pdf/2602.20773v1](https://arxiv.org/pdf/2602.20773v1)**

> **作者:** Sachin Dudda Nagaraju; Ashkan Moradi; Bendik Skarre Abrahamsen; Mattijs Elschot
>
> **备注:** Submitted to IEEE JBHI
>
> **摘要:** Artificial intelligence has emerged as a transformative tool in medical image analysis, yet developing robust and generalizable segmentation models remains difficult due to fragmented, privacy-constrained imaging data siloed across institutions. While federated learning (FL) enables collaborative model training without centralizing data, cross-modality domain shifts pose a critical challenge, particularly when models trained on one modality fail to generalize to another. Many existing solutions require paired multimodal data per patient or rely on complex architectures, both of which are impractical in real clinical settings. In this work, we consider a realistic FL scenario where each client holds single-modality data (CT or MRI), and systematically investigate augmentation strategies for cross-modality generalization. Using abdominal organ segmentation and whole-heart segmentation as representative multi-class and binary segmentation benchmarks, we evaluate convolution-based spatial augmentation, frequency-domain manipulation, domain-specific normalization, and global intensity nonlinear (GIN) augmentation. Our results show that GIN consistently outperforms alternatives in both centralized and federated settings by simulating cross-modality appearance variations while preserving anatomical structure. For the pancreas, Dice score improved from 0.073 to 0.437, a 498% gain. Our federated approach achieves 93-98% of centralized training accuracy, demonstrating strong cross-modality generalization without compromising data privacy, pointing toward feasible federated AI deployment across diverse healthcare systems.
>
---
#### [new 055] Spa3R: Predictive Spatial Field Modeling for 3D Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文提出Spa3R，解决VLM在3D空间理解上的不足，通过自监督学习构建统一的三维场景表示，提升视觉问答性能。**

- **链接: [https://arxiv.org/pdf/2602.21186v1](https://arxiv.org/pdf/2602.21186v1)**

> **作者:** Haoyi Jiang; Liu Liu; Xinjie Wang; Yonghao He; Wei Sui; Zhizhong Su; Wenyu Liu; Xinggang Wang
>
> **摘要:** While Vision-Language Models (VLMs) exhibit exceptional 2D visual understanding, their ability to comprehend and reason about 3D space--a cornerstone of spatial intelligence--remains superficial. Current methodologies attempt to bridge this domain gap either by relying on explicit 3D modalities or by augmenting VLMs with partial, view-conditioned geometric priors. However, such approaches hinder scalability and ultimately burden the language model with the ill-posed task of implicitly reconstructing holistic 3D geometry from sparse cues. In this paper, we argue that spatial intelligence can emerge inherently from 2D vision alone, rather than being imposed via explicit spatial instruction tuning. To this end, we introduce Spa3R, a self-supervised framework that learns a unified, view-invariant spatial representation directly from unposed multi-view images. Spa3R is built upon the proposed Predictive Spatial Field Modeling (PSFM) paradigm, where Spa3R learns to synthesize feature fields for arbitrary unseen views conditioned on a compact latent representation, thereby internalizing a holistic and coherent understanding of the underlying 3D scene. We further integrate the pre-trained Spa3R Encoder into existing VLMs via a lightweight adapter to form Spa3-VLM, effectively grounding language reasoning in a global spatial context. Experiments on the challenging VSI-Bench demonstrate that Spa3-VLM achieves state-of-the-art accuracy of 58.6% on 3D VQA, significantly outperforming prior methods. These results highlight PSFM as a scalable path toward advancing spatial intelligence. Code is available at https://github.com/hustvl/Spa3R.
>
---
#### [new 056] Bridging Physically Based Rendering and Diffusion Models with Stochastic Differential Equation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型缺乏物理控制的问题。通过构建随机微分方程，将物理渲染与扩散模型统一，实现对材质和光照的物理控制。**

- **链接: [https://arxiv.org/pdf/2602.20725v1](https://arxiv.org/pdf/2602.20725v1)**

> **作者:** Junwei Shu; Wenjie Liu; Changgu Chen; Hantang Liu; Yang Li; Changbo Wang
>
> **备注:** preprint
>
> **摘要:** Diffusion-based image generators excel at producing realistic content from text or image conditions, but they offer only limited explicit control over low-level, physically grounded shading and material properties. In contrast, physically based rendering (PBR) offers fine-grained physical control but lacks prompt-driven flexibility. Although these two paradigms originate from distinct communities, both share a common evolution -- from noisy observations to clean images. In this paper, we propose a unified stochastic formulation that bridges Monte Carlo rendering and diffusion-based generative modeling. First, a general stochastic differential equation (SDE) formulation for Monte Carlo integration under the Central Limit Theorem is modeled. Through instantiation via physically based path tracing, we convert it into a physically grounded SDE representation. Moreover, we provide a systematic analysis of how the physical characteristics of path tracing can be extended to existing diffusion models from the perspective of noise variance. Extensive experiments across multiple tasks show that our method can exert physically grounded control over diffusion-generated results, covering tasks such as rendering and material editing.
>
---
#### [new 057] Pip-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching
- **分类: cs.CV**

- **简介: 该论文属于立体匹配任务，旨在解决迭代方法在边缘设备部署效率低的问题。通过引入渐进式剪枝、协同单目先验传输和优化RNN结构，提升了实时性和精度。**

- **链接: [https://arxiv.org/pdf/2602.20496v1](https://arxiv.org/pdf/2602.20496v1)**

> **作者:** Jintu Zheng; Qizhe Liu; HuangXin Xu; Zhuojie Chen
>
> **备注:** Accepted to CVPR 2026 (3D vision track)
>
> **摘要:** While iterative stereo matching achieves high accuracy, its dependence on Recurrent Neural Networks (RNN) hinders edge deployment, a challenge underexplored in existing researches. We analyze iterative refinement and reveal that disparity updates are spatially sparse and temporally redundant. First, we introduce a progressive iteration pruning strategy that suppresses redundant update steps, effectively collapsing the recursive computation into a near-single-pass inference. Second, we propose a collaborative monocular prior transfer framework that implicitly embeds depth priors without requiring a dedicated monocular encoder, thereby eliminating its associated computational burden. Third, we develop FlashGRU, a hardware-aware RNN operator leveraging structured sparsity and I/O-conscious design, achieving a 7.28$\times$ speedup, 76.6\% memory peak reduction and 80.9\% global memory requests reduction over natvie ConvGRUs under 2K resolution. Our PipStereo enables real-time, high-fidelity stereo matching on edge hardware: it processes 320$\times$640 frames in just 75ms on an NVIDIA Jetson Orin NX (FP16) and 19ms on RTX 4090, matching the accuracy of large iterative based models, and our generalization ability and accuracy far exceeds that of existing real-time methods. Our embedded AI projects will be updated at: https://github.com/XPENG-Aridge-AI.
>
---
#### [new 058] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出VGGDrive，解决自动驾驶中视觉-语言模型缺乏跨视角几何建模的问题，通过引入3D几何增强模块提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.20794v1](https://arxiv.org/pdf/2602.20794v1)**

> **作者:** Jie Wang; Guang Li; Zhijian Huang; Chenxu Dang; Hangjun Ye; Yahong Han; Long Chen
>
> **备注:** CVPR 2026
>
> **摘要:** The significance of cross-view 3D geometric modeling capabilities for autonomous driving is self-evident, yet existing Vision-Language Models (VLMs) inherently lack this capability, resulting in their mediocre performance. While some promising approaches attempt to mitigate this by constructing Q&A data for auxiliary training, they still fail to fundamentally equip VLMs with the ability to comprehensively handle diverse evaluation protocols. We thus chart a new course, advocating for the infusion of VLMs with the cross-view geometric grounding of mature 3D foundation models, closing this critical capability gap in autonomous driving. In this spirit, we propose a novel architecture, VGGDrive, which empowers Vision-language models with cross-view Geometric Grounding for autonomous Driving. Concretely, to bridge the cross-view 3D geometric features from the frozen visual 3D model with the VLM's 2D visual features, we introduce a plug-and-play Cross-View 3D Geometric Enabler (CVGE). The CVGE decouples the base VLM architecture and effectively empowers the VLM with 3D features through a hierarchical adaptive injection mechanism. Extensive experiments show that VGGDrive enhances base VLM performance across five autonomous driving benchmarks, including tasks like cross-view risk perception, motion prediction, and trajectory planning. It's our belief that mature 3D foundation models can empower autonomous driving tasks through effective integration, and we hope our initial exploration demonstrates the potential of this paradigm to the autonomous driving community.
>
---
#### [new 059] Le-DETR: Revisiting Real-Time Detection Transformer with Efficient Encoder Design
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决实时检测模型预训练成本高的问题。通过设计高效编码器，提出Le-DETR模型，在保持高精度的同时显著降低预训练开销。**

- **链接: [https://arxiv.org/pdf/2602.21010v1](https://arxiv.org/pdf/2602.21010v1)**

> **作者:** Jiannan Huang; Aditya Kane; Fengzhe Zhou; Yunchao Wei; Humphrey Shi
>
> **备注:** CVPR Findings
>
> **摘要:** Real-time object detection is crucial for real-world applications as it requires high accuracy with low latency. While Detection Transformers (DETR) have demonstrated significant performance improvements, current real-time DETR models are challenging to reproduce from scratch due to excessive pre-training overheads on the backbone, constraining research advancements by hindering the exploration of novel backbone architectures. In this paper, we want to show that by using general good design, it is possible to have \textbf{high performance} with \textbf{low pre-training cost}. After a thorough study of the backbone architecture, we propose EfficientNAT at various scales, which incorporates modern efficient convolution and local attention mechanisms. Moreover, we re-design the hybrid encoder with local attention, significantly enhancing both performance and inference speed. Based on these advancements, we present Le-DETR (\textbf{L}ow-cost and \textbf{E}fficient \textbf{DE}tection \textbf{TR}ansformer), which achieves a new \textbf{SOTA} in real-time detection using only ImageNet1K and COCO2017 training datasets, saving about 80\% images in pre-training stage compared with previous methods. We demonstrate that with well-designed, real-time DETR models can achieve strong performance without the need for complex and computationally expensive pretraining. Extensive experiments show that Le-DETR-M/L/X achieves \textbf{52.9/54.3/55.1 mAP} on COCO Val2017 with \textbf{4.45/5.01/6.68 ms} on an RTX4090. It surpasses YOLOv12-L/X by \textbf{+0.6/-0.1 mAP} while achieving similar speed and \textbf{+20\%} speedup. Compared with DEIM-D-FINE, Le-DETR-M achieves \textbf{+0.2 mAP} with slightly faster inference, and surpasses DEIM-D-FINE-L by \textbf{+0.4 mAP} with only \textbf{0.4 ms} additional latency. Code and weights will be open-sourced.
>
---
#### [new 060] FLIM Networks with Bag of Feature Points
- **分类: cs.CV**

- **简介: 该论文属于显著目标检测任务，解决传统标注成本高的问题。提出FLIM-BoFP方法，提升滤波器估计效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.20845v1](https://arxiv.org/pdf/2602.20845v1)**

> **作者:** João Deltregia Martinelli; Marcelo Luis Rodrigues Filho; Felipe Crispim da Rocha Salvagnini; Gilson Junior Soares; Jefersson A. dos Santos; Alexandre X. Falcão
>
> **备注:** Accepted at the 28th Iberoamerican Congress on Pattern Recognition (CIARP 2025). To appear in Lecture Notes in Computer Science (LNCS), Springer
>
> **摘要:** Convolutional networks require extensive image annotation, which can be costly and time-consuming. Feature Learning from Image Markers (FLIM) tackles this challenge by estimating encoder filters (i.e., kernel weights) from user-drawn markers on discriminative regions of a few representative images without traditional optimization. Such an encoder combined with an adaptive decoder comprises a FLIM network fully trained without backpropagation. Prior research has demonstrated their effectiveness in Salient Object Detection (SOD), being significantly lighter than existing lightweight models. This study revisits FLIM SOD and introduces FLIM-Bag of Feature Points (FLIM-BoFP), a considerably faster filter estimation method. The previous approach, FLIM-Cluster, derives filters through patch clustering at each encoder's block, leading to computational overhead and reduced control over filter locations. FLIM-BoFP streamlines this process by performing a single clustering at the input block, creating a bag of feature points, and defining filters directly from mapped feature points across all blocks. The paper evaluates the benefits in efficiency, effectiveness, and generalization of FLIM-BoFP compared to FLIM-Cluster and other state-of-the-art baselines for parasite detection in optical microscopy images.
>
---
#### [new 061] From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection
- **分类: cs.CV**

- **简介: 该论文属于关键点检测任务，解决传统方法在序列中跟踪关键点能力不足的问题。提出TraqPoint框架，通过强化学习优化关键点的可跟踪性。**

- **链接: [https://arxiv.org/pdf/2602.20630v1](https://arxiv.org/pdf/2602.20630v1)**

> **作者:** Yepeng Liu; Hao Li; Liwen Yang; Fangzhen Li; Xudi Ge; Yuliang Gu; kuang Gao; Bing Wang; Guang Chen; Hangjun Ye; Yongchao Xu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Keypoint-based matching is a fundamental component of modern 3D vision systems, such as Structure-from-Motion (SfM) and SLAM. Most existing learning-based methods are trained on image pairs, a paradigm that fails to explicitly optimize for the long-term trackability of keypoints across sequences under challenging viewpoint and illumination changes. In this paper, we reframe keypoint detection as a sequential decision-making problem. We introduce TraqPoint, a novel, end-to-end Reinforcement Learning (RL) framework designed to optimize the \textbf{Tra}ck-\textbf{q}uality (Traq) of keypoints directly on image sequences. Our core innovation is a track-aware reward mechanism that jointly encourages the consistency and distinctiveness of keypoints across multiple views, guided by a policy gradient method. Extensive evaluations on sparse matching benchmarks, including relative pose estimation and 3D reconstruction, demonstrate that TraqPoint significantly outperforms some state-of-the-art (SOTA) keypoint detection and description methods.
>
---
#### [new 062] CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于3D点云域适应任务，解决从合成到真实数据的域偏移问题。通过结合CLIP和轻量编码器，提出CLIPoint3D框架，提升适应效率与准确率。**

- **链接: [https://arxiv.org/pdf/2602.20409v1](https://arxiv.org/pdf/2602.20409v1)**

> **作者:** Mainak Singha; Sarthak Mehrotra; Paolo Casari; Subhasis Chaudhuri; Elisa Ricci; Biplab Banerjee
>
> **备注:** Accepted in CVPR 2026
>
> **摘要:** Recent vision-language models (VLMs) such as CLIP demonstrate impressive cross-modal reasoning, extending beyond images to 3D perception. Yet, these models remain fragile under domain shifts, especially when adapting from synthetic to real-world point clouds. Conventional 3D domain adaptation approaches rely on heavy trainable encoders, yielding strong accuracy but at the cost of efficiency. We introduce CLIPoint3D, the first framework for few-shot unsupervised 3D point cloud domain adaptation built upon CLIP. Our approach projects 3D samples into multiple depth maps and exploits the frozen CLIP backbone, refined through a knowledge-driven prompt tuning scheme that integrates high-level language priors with geometric cues from a lightweight 3D encoder. To adapt task-specific features effectively, we apply parameter-efficient fine-tuning to CLIP's encoders and design an entropy-guided view sampling strategy for selecting confident projections. Furthermore, an optimal transport-based alignment loss and an uncertainty-aware prototype alignment loss collaboratively bridge source-target distribution gaps while maintaining class separability. Extensive experiments on PointDA-10 and GraspNetPC-10 benchmarks show that CLIPoint3D achieves consistent 3-16% accuracy gains over both CLIP-based and conventional encoder-based baselines. Codes are available at https://github.com/SarthakM320/CLIPoint3D.
>
---
#### [new 063] Vision-Language Models for Ergonomic Assessment of Manual Lifting Tasks: Estimating Horizontal and Vertical Hand Distances from RGB Video
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **简介: 论文探讨了使用视觉语言模型从RGB视频中估算手动搬运任务的水平和垂直手部距离，以支持 ergonomic 评估。该研究旨在解决传统测量方法在实际应用中的局限性，通过开发两种VLM管道提升估计精度。**

- **链接: [https://arxiv.org/pdf/2602.20658v1](https://arxiv.org/pdf/2602.20658v1)**

> **作者:** Mohammad Sadra Rajabi; Aanuoluwapo Ojelade; Sunwook Kim; Maury A. Nussbaum
>
> **摘要:** Manual lifting tasks are a major contributor to work-related musculoskeletal disorders, and effective ergonomic risk assessment is essential for quantifying physical exposure and informing ergonomic interventions. The Revised NIOSH Lifting Equation (RNLE) is a widely used ergonomic risk assessment tool for lifting tasks that relies on six task variables, including horizontal (H) and vertical (V) hand distances; such distances are typically obtained through manual measurement or specialized sensing systems and are difficult to use in real-world environments. We evaluated the feasibility of using innovative vision-language models (VLMs) to non-invasively estimate H and V from RGB video streams. Two multi-stage VLM-based pipelines were developed: a text-guided detection-only pipeline and a detection-plus-segmentation pipeline. Both pipelines used text-guided localization of task-relevant regions of interest, visual feature extraction from those regions, and transformer-based temporal regression to estimate H and V at the start and end of a lift. For a range of lifting tasks, estimation performance was evaluated using leave-one-subject-out validation across the two pipelines and seven camera view conditions. Results varied significantly across pipelines and camera view conditions, with the segmentation-based, multi-view pipeline consistently yielding the smallest errors, achieving mean absolute errors of approximately 6-8 cm when estimating H and 5-8 cm when estimating V. Across pipelines and camera view configurations, pixel-level segmentation reduced estimation error by approximately 20-30% for H and 35-40% for V relative to the detection-only pipeline. These findings support the feasibility of VLM-based pipelines for video-based estimation of RNLE distance parameters.
>
---
#### [new 064] Region of Interest Segmentation and Morphological Analysis for Membranes in Cryo-Electron Tomography
- **分类: cs.CV**

- **简介: 该论文属于生物图像分析任务，旨在解决膜结构ROI分割与形态分析问题。提出TomoROIS-SurfORA框架，实现直接、自动的膜结构分割与定量分析。**

- **链接: [https://arxiv.org/pdf/2602.21195v1](https://arxiv.org/pdf/2602.21195v1)**

> **作者:** Xingyi Cheng; Julien Maufront; Aurélie Di Cicco; Daniël M. Pelt; Manuela Dezi; Daniel Lévy
>
> **摘要:** Cryo-electron tomography (cryo-ET) enables high resolution, three-dimensional reconstruction of biological structures, including membranes and membrane proteins. Identification of regions of interest (ROIs) is central to scientific imaging, as it enables isolation and quantitative analysis of specific structural features within complex datasets. In practice, however, ROIs are typically derived indirectly through full structure segmentation followed by post hoc analysis. This limitation is especially apparent for continuous and geometrically complex structures such as membranes, which are segmented as single entities. Here, we developed TomoROIS-SurfORA, a two step framework for direct, shape-agnostic ROI segmentation and morphological surface analysis. TomoROIS performs deep learning-based ROI segmentation and can be trained from scratch using small annotated datasets, enabling practical application across diverse imaging data. SurfORA processes segmented structures as point clouds and surface meshes to extract quantitative morphological features, including inter-membrane distances, curvature, and surface roughness. It supports both closed and open surfaces, with specific considerations for open surfaces, which are common in cryo-ET due to the missing wedge effect. We demonstrate both tools using in vitro reconstituted membrane systems containing deformable vesicles with complex geometries, enabling automatic quantitative analysis of membrane contact sites and remodeling events such as invagination. While demonstrated here on cryo-ET membrane data, the combined approach is applicable to ROI detection and surface analysis in broader scientific imaging contexts.
>
---
#### [new 065] Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决稀疏视角下的过拟合问题。提出DropAnSH-GS方法，通过随机删除高阶球谐系数和邻近高斯点，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.20933v1](https://arxiv.org/pdf/2602.20933v1)**

> **作者:** Shuangkang Fang; I-Chao Shen; Xuanyang Zhang; Zesheng Wang; Yufeng Wang; Wenrui Ding; Gang Yu; Takeo Igarashi
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Recent 3D Gaussian Splatting (3DGS) Dropout methods address overfitting under sparse-view conditions by randomly nullifying Gaussian opacities. However, we identify a neighbor compensation effect in these approaches: dropped Gaussians are often compensated by their neighbors, weakening the intended regularization. Moreover, these methods overlook the contribution of high-degree spherical harmonic coefficients (SH) to overfitting. To address these issues, we propose DropAnSH-GS, a novel anchor-based Dropout strategy. Rather than dropping Gaussians independently, our method randomly selects certain Gaussians as anchors and simultaneously removes their spatial neighbors. This effectively disrupts local redundancies near anchors and encourages the model to learn more robust, globally informed representations. Furthermore, we extend the Dropout to color attributes by randomly dropping higher-degree SH to concentrate appearance information in lower-degree SH. This strategy further mitigates overfitting and enables flexible post-training model compression via SH truncation. Experimental results demonstrate that DropAnSH-GS substantially outperforms existing Dropout methods with negligible computational overhead, and can be readily integrated into various 3DGS variants to enhance their performances. Project Website: https://sk-fun.fun/DropAnSH-GS
>
---
#### [new 066] Mask-HybridGNet: Graph-based segmentation with emergent anatomical correspondence from pixel-level supervision
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决缺乏手动标注对应点的问题。通过像素级掩码训练，实现解剖结构的自动对应与分割。**

- **链接: [https://arxiv.org/pdf/2602.21179v1](https://arxiv.org/pdf/2602.21179v1)**

> **作者:** Nicolás Gaggion; Maria J. Ledesma-Carbayo; Stergios Christodoulidis; Maria Vakalopoulou; Enzo Ferrante
>
> **摘要:** Graph-based medical image segmentation represents anatomical structures using boundary graphs, providing fixed-topology landmarks and inherent population-level correspondences. However, their clinical adoption has been hindered by a major requirement: training datasets with manually annotated landmarks that maintain point-to-point correspondences across patients rarely exist in practice. We introduce Mask-HybridGNet, a framework that trains graph-based models directly using standard pixel-wise masks, eliminating the need for manual landmark annotations. Our approach aligns variable-length ground truth boundaries with fixed-length landmark predictions by combining Chamfer distance supervision and edge-based regularization to ensure local smoothness and regular landmark distribution, further refined via differentiable rasterization. A significant emergent property of this framework is that predicted landmark positions become consistently associated with specific anatomical locations across patients without explicit correspondence supervision. This implicit atlas learning enables temporal tracking, cross-slice reconstruction, and morphological population analyses. Beyond direct segmentation, Mask-HybridGNet can extract correspondences from existing segmentation masks, allowing it to generate stable anatomical atlases from any high-quality pixel-based model. Experiments across chest radiography, cardiac ultrasound, cardiac MRI, and fetal imaging demonstrate that our model achieves competitive results against state-of-the-art pixel-based methods, while ensuring anatomical plausibility by enforcing boundary connectivity through a fixed graph adjacency matrix. This framework leverages the vast availability of standard segmentation masks to build structured models that maintain topological integrity and provide implicit correspondences.
>
---
#### [new 067] LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于长视频理解任务，解决低算力下高效视频分析问题。提出LongVideo-R1模型，通过推理模块和主动导航策略，提升视频问答的准确率与效率。**

- **链接: [https://arxiv.org/pdf/2602.20913v1](https://arxiv.org/pdf/2602.20913v1)**

> **作者:** Jihao Qiu; Lingxi Xie; Xinyue Huo; Qi Tian; Qixiang Ye
>
> **备注:** 17 pages, 9 figures, 8 tables, accepted to CVPR 2026
>
> **摘要:** This paper addresses the critical and underexplored challenge of long video understanding with low computational budgets. We propose LongVideo-R1, an active, reasoning-equipped multimodal large language model (MLLM) agent designed for efficient video context navigation, avoiding the redundancy of exhaustive search. At the core of LongVideo-R1 lies a reasoning module that leverages high-level visual cues to infer the most informative video clip for subsequent processing. During inference, the agent initiates traversal from top-level visual summaries and iteratively refines its focus, immediately halting the exploration process upon acquiring sufficient knowledge to answer the query. To facilitate training, we first extract hierarchical video captions from CGBench, a video corpus with grounding annotations, and guide GPT-5 to generate 33K high-quality chain-of-thought-with-tool trajectories. The LongVideo-R1 agent is fine-tuned upon the Qwen-3-8B model through a two-stage paradigm: supervised fine-tuning (SFT) followed by reinforcement learning (RL), where RL employs a specifically designed reward function to maximize selective and efficient clip navigation. Experiments on multiple long video benchmarks validate the effectiveness of name, which enjoys superior tradeoff between QA accuracy and efficiency. All curated data and source code are provided in the supplementary material and will be made publicly available. Code and data are available at: https://github.com/qiujihao19/LongVideo-R1
>
---
#### [new 068] BiRQA: Bidirectional Robust Quality Assessment for Images
- **分类: cs.CV**

- **简介: 该论文提出BiRQA，用于图像质量评估任务，解决现有模型速度慢、易受攻击的问题。通过双向金字塔结构和对抗训练提升准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.20351v1](https://arxiv.org/pdf/2602.20351v1)**

> **作者:** Aleksandr Gushchin; Dmitriy S. Vatolin; Anastasia Antsiferova
>
> **摘要:** Full-Reference image quality assessment (FR IQA) is important for image compression, restoration and generative modeling, yet current neural metrics remain slow and vulnerable to adversarial perturbations. We present BiRQA, a compact FR IQA metric model that processes four fast complementary features within a bidirectional multiscale pyramid. A bottom-up attention module injects fine-scale cues into coarse levels through an uncertainty-aware gate, while a top-down cross-gating block routes semantic context back to high resolution. To enhance robustness, we introduce Anchored Adversarial Training, a theoretically grounded strategy that uses clean "anchor" samples and a ranking loss to bound pointwise prediction error under attacks. On five public FR IQA benchmarks BiRQA outperforms or matches the previous state of the art (SOTA) while running ~3x faster than previous SOTA models. Under unseen white-box attacks it lifts SROCC from 0.30-0.57 to 0.60-0.84 on KADID-10k, demonstrating substantial robustness gains. To our knowledge, BiRQA is the only FR IQA model combining competitive accuracy with real-time throughput and strong adversarial resilience.
>
---
#### [new 069] BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在从多视角图像中恢复B-rep模型。提出BrepGaussian框架，结合高斯泼溅与分阶段学习，提升重建精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21105v1](https://arxiv.org/pdf/2602.21105v1)**

> **作者:** Jiaxing Yu; Dongyang Ren; Hangyu Xu; Zhouyuxiao Yang; Yuanqi Li; Jie Guo; Zhengkang Zhou; Yanwen Guo
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** The boundary representation (B-rep) models a 3D solid as its explicit boundaries: trimmed corners, edges, and faces. Recovering B-rep representation from unstructured data is a challenging and valuable task of computer vision and graphics. Recent advances in deep learning have greatly improved the recovery of 3D shape geometry, but still depend on dense and clean point clouds and struggle to generalize to novel shapes. We propose B-rep Gaussian Splatting (BrepGaussian), a novel framework that learns 3D parametric representations from 2D images. We employ a Gaussian Splatting renderer with learnable features, followed by a specific fitting strategy. To disentangle geometry reconstruction and feature learning, we introduce a two-stage learning framework that first captures geometry and edges and then refines patch features to achieve clean geometry and coherent instance representations. Extensive experiments demonstrate the superior performance of our approach to state-of-the-art methods. We will release our code and datasets upon acceptance.
>
---
#### [new 070] Boosting Instance Awareness via Cross-View Correlation with 4D Radar and Camera for 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决4D雷达与相机融合中的实例感知不足问题。通过引入SIFormer模型，提升实例识别能力，结合图像语义与雷达几何信息，提高检测精度。**

- **链接: [https://arxiv.org/pdf/2602.20632v1](https://arxiv.org/pdf/2602.20632v1)**

> **作者:** Xiaokai Bai; Lianqing Zheng; Si-Yuan Cao; Xiaohan Zhang; Zhe Wu; Beinan Yu; Fang Wang; Jie Bai; Hui-Liang Shen
>
> **备注:** 14 pages, 10 figures, 13 tables
>
> **摘要:** 4D millimeter-wave radar has emerged as a promising sensing modality for autonomous driving due to its robustness and affordability. However, its sparse and weak geometric cues make reliable instance activation difficult, limiting the effectiveness of existing radar-camera fusion paradigms. BEV-level fusion offers global scene understanding but suffers from weak instance focus, while perspective-level fusion captures instance details but lacks holistic context. To address these limitations, we propose SIFormer, a scene-instance aware transformer for 3D object detection using 4D radar and camera. SIFormer first suppresses background noise during view transformation through segmentation- and depth-guided localization. It then introduces a cross-view activation mechanism that injects 2D instance cues into BEV space, enabling reliable instance awareness under weak radar geometry. Finally, a transformer-based fusion module aggregates complementary image semantics and radar geometry for robust perception. As a result, with the aim of enhancing instance awareness, SIFormer bridges the gap between the two paradigms, combining their complementary strengths to address inherent sparse nature of radar and improve detection accuracy. Experiments demonstrate that SIFormer achieves state-of-the-art performance on View-of-Delft, TJ4DRadSet and NuScenes datasets. Source code is available at github.com/shawnnnkb/SIFormer.
>
---
#### [new 071] Not Just What's There: Enabling CLIP to Comprehend Negated Visual Descriptions Without Fine-tuning
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视觉语言理解任务，解决CLIP模型对否定描述理解不足的问题。通过提出CLIPGlasses框架，增强模型对否定语义的识别能力。**

- **链接: [https://arxiv.org/pdf/2602.21035v1](https://arxiv.org/pdf/2602.21035v1)**

> **作者:** Junhao Xiao; Zhiyu Wu; Hao Lin; Yi Chen; Yahui Liu; Xiaoran Zhao; Zixu Wang; Zejiang He
>
> **摘要:** Vision-Language Models (VLMs) like CLIP struggle to understand negation, often embedding affirmatives and negatives similarly (e.g., matching "no dog" with dog images). Existing methods refine negation understanding via fine-tuning CLIP's text encoder, risking overfitting. In this work, we propose CLIPGlasses, a plug-and-play framework that enhances CLIP's ability to comprehend negated visual descriptions. CLIPGlasses adopts a dual-stage design: a Lens module disentangles negated semantics from text embeddings, and a Frame module predicts context-aware repulsion strength, which is integrated into a modified similarity computation to penalize alignment with negated semantics, thereby reducing false positive matches. Experiments show that CLIP equipped with CLIPGlasses achieves competitive in-domain performance and outperforms state-of-the-art methods in cross-domain generalization. Its superiority is especially evident under low-resource conditions, indicating stronger robustness across domains.
>
---
#### [new 072] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images
- **分类: cs.CV**

- **简介: 该论文属于图像伪造检测任务，旨在解决模型泛化能力差的问题。通过学习真实图像分布，提出SimLBR方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.20412v1](https://arxiv.org/pdf/2602.20412v1)**

> **作者:** Aayush Dhakal; Subash Khanal; Srikumar Sastry; Jacob Arndt; Philipe Ambrozio Dias; Dalton Lunga; Nathan Jacobs
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** The rapid advancement of generative models has made the detection of AI-generated images a critical challenge for both research and society. Recent works have shown that most state-of-the-art fake image detection methods overfit to their training data and catastrophically fail when evaluated on curated hard test sets with strong distribution shifts. In this work, we argue that it is more principled to learn a tight decision boundary around the real image distribution and treat the fake category as a sink class. To this end, we propose SimLBR, a simple and efficient framework for fake image detection using Latent Blending Regularization (LBR). Our method significantly improves cross-generator generalization, achieving up to +24.85\% accuracy and +69.62\% recall on the challenging Chameleon benchmark. SimLBR is also highly efficient, training orders of magnitude faster than existing approaches. Furthermore, we emphasize the need for reliability-oriented evaluation in fake image detection, introducing risk-adjusted metrics and worst-case estimates to better assess model robustness. All code and models will be released on HuggingFace and GitHub.
>
---
#### [new 073] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频到音频生成任务，旨在解决模型在短视频上训练后无法有效生成长音频的问题。通过提出MMHNet模型，实现超过5分钟的长音频生成。**

- **链接: [https://arxiv.org/pdf/2602.20981v1](https://arxiv.org/pdf/2602.20981v1)**

> **作者:** Christian Simon; MAsato Ishii; Wei-Yao Wang; Koichi Saito; Akio Hayakawa; Dongseok Shim; Zhi Zhong; Shuyang Cui; Shusuke Takahashi; Takashi Shibuya; Yuki Mitsufuji
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Scaling multimodal alignment between video and audio is challenging, particularly due to limited data and the mismatch between text descriptions and frame-level video information. In this work, we tackle the scaling challenge in multimodal-to-audio generation, examining whether models trained on short instances can generalize to longer ones during testing. To tackle this challenge, we present multimodal hierarchical networks so-called MMHNet, an enhanced extension of state-of-the-art video-to-audio models. Our approach integrates a hierarchical method and non-causal Mamba to support long-form audio generation. Our proposed method significantly improves long audio generation up to more than 5 minutes. We also prove that training short and testing long is possible in the video-to-audio generation tasks without training on the longer durations. We show in our experiments that our proposed method could achieve remarkable results on long-video to audio benchmarks, beating prior works in video-to-audio tasks. Moreover, we showcase our model capability in generating more than 5 minutes, while prior video-to-audio methods fall short in generating with long durations.
>
---
#### [new 074] Knowing the Unknown: Interpretable Open-World Object Detection via Concept Decomposition Model
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于开放世界目标检测任务，旨在解决已知与未知类别混淆问题。提出IPOW框架，通过概念分解提升可解释性与检测可靠性。**

- **链接: [https://arxiv.org/pdf/2602.20616v1](https://arxiv.org/pdf/2602.20616v1)**

> **作者:** Xueqiang Lv; Shizhou Zhang; Yinghui Xing; Di Xu; Peng Wang; Yanning Zhang
>
> **摘要:** Open-world object detection (OWOD) requires incrementally detecting known categories while reliably identifying unknown objects. Existing methods primarily focus on improving unknown recall, yet overlook interpretability, often leading to known-unknown confusion and reduced prediction reliability. This paper aims to make the entire OWOD framework interpretable, enabling the detector to truly "knowing the unknown". To this end, we propose a concept-driven InterPretable OWOD framework(IPOW) by introducing a Concept Decomposition Model (CDM) for OWOD, which explicitly decomposes the coupled RoI features in Faster R-CNN into discriminative, shared, and background concepts. Discriminative concepts identify the most discriminative features to enlarge the distances between known categories, while shared and background concepts, due to their strong generalization ability, can be readily transferred to detect unknown categories. Leveraging the interpretable framework, we identify that known-unknown confusion arises when unknown objects fall into the discriminative space of known classes. To address this, we propose Concept-Guided Rectification (CGR) to further resolve such confusion. Extensive experiments show that IPOW significantly improves unknown recall while mitigating confusion, and provides concept-level interpretability for both known and unknown predictions.
>
---
#### [new 075] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer
- **分类: cs.CV**

- **简介: 该论文提出EW-DETR框架，解决动态环境中物体检测问题，通过增量学习、域适应和未知检测实现高效目标检测。**

- **链接: [https://arxiv.org/pdf/2602.20985v1](https://arxiv.org/pdf/2602.20985v1)**

> **作者:** Munish Monga; Vishal Chudasama; Pankaj Wasnik; C. V. Jawahar
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Real-world object detection must operate in evolving environments where new classes emerge, domains shift, and unseen objects must be identified as "unknown": all without accessing prior data. We introduce Evolving World Object Detection (EWOD), a paradigm coupling incremental learning, domain adaptation, and unknown detection under exemplar-free constraints. To tackle EWOD, we propose EW-DETR framework that augments DETR-based detectors with three synergistic modules: Incremental LoRA Adapters for exemplar-free incremental learning under evolving domains; a Query-Norm Objectness Adapter that decouples objectness-aware features from DETR decoder queries; and Entropy-Aware Unknown Mixing for calibrated unknown detection. This framework generalises across DETR-based detectors, enabling state-of-the-art RF-DETR to operate effectively in evolving-world settings. We also introduce FOGS (Forgetting, Openness, Generalisation Score) to holistically evaluate performance across these dimensions. Extensive experiments on Pascal Series and Diverse Weather benchmarks show EW-DETR outperforms other methods, improving FOGS by 57.24%.
>
---
#### [new 076] Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决高速飞行无人机的运动模糊和位姿漂移问题。通过融合事件流和模糊图像，优化NeRF并提升位姿估计，实现高精度辐射场重建。**

- **链接: [https://arxiv.org/pdf/2602.21101v1](https://arxiv.org/pdf/2602.21101v1)**

> **作者:** Rong Zou; Marco Cannici; Davide Scaramuzza
>
> **摘要:** Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.
>
---
#### [new 077] Are Multimodal Large Language Models Good Annotators for Image Tagging?
- **分类: cs.CV**

- **简介: 该论文属于图像标注任务，旨在解决人工标注成本高、效率低的问题。通过分析MLLM的标注能力，提出TagLLM框架以提升其标注质量与下游任务表现。**

- **链接: [https://arxiv.org/pdf/2602.20972v1](https://arxiv.org/pdf/2602.20972v1)**

> **作者:** Ming-Kun Xie; Jia-Hao Xiao; Zhiqiang Kou; Zhongnian Li; Gang Niu; Masashi Sugiyama
>
> **摘要:** Image tagging, a fundamental vision task, traditionally relies on human-annotated datasets to train multi-label classifiers, which incurs significant labor and costs. While Multimodal Large Language Models (MLLMs) offer promising potential to automate annotation, their capability to replace human annotators remains underexplored. This paper aims to analyze the gap between MLLM-generated and human annotations and to propose an effective solution that enables MLLM-based annotation to replace manual labeling. Our analysis of MLLM annotations reveals that, under a conservative estimate, MLLMs can reduce annotation cost to as low as one-thousandth of the human cost, mainly accounting for GPU usage, which is nearly negligible compared to manual efforts. Their annotation quality reaches about 50\% to 80\% of human performance, while achieving over 90\% performance on downstream training tasks.Motivated by these findings, we propose TagLLM, a novel framework for image tagging, which aims to narrow the gap between MLLM-generated and human annotations. TagLLM comprises two components: Candidates generation, which employs structured group-wise prompting to efficiently produce a compact candidate set that covers as many true labels as possible while reducing subsequent annotation workload; and label disambiguation, which interactively calibrates the semantic concept of categories in the prompts and effectively refines the candidate labels. Extensive experiments show that TagLLM substantially narrows the gap between MLLM-generated and human annotations, especially in downstream training performance, where it closes about 60\% to 80\% of the difference.
>
---
#### [new 078] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces
- **分类: cs.CV**

- **简介: 该论文属于图像篡改检测任务，旨在解决AI生成内容导致的面部篡改问题。提出RecoverMark框架，实现鲁棒的篡改定位、内容恢复和版权验证。**

- **链接: [https://arxiv.org/pdf/2602.20618v1](https://arxiv.org/pdf/2602.20618v1)**

> **作者:** Haonan An; Xiaohui Ye; Guang Hua; Yihang Tao; Hangcheng Cao; Xiangyu Yu; Yuguang Fang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** The proliferation of AI-generated content has facilitated sophisticated face manipulation, severely undermining visual integrity and posing unprecedented challenges to intellectual property. In response, a common proactive defense leverages fragile watermarks to detect, localize, or even recover manipulated regions. However, these methods always assume an adversary unaware of the embedded watermark, overlooking their inherent vulnerability to watermark removal attacks. Furthermore, this fragility is exacerbated in the commonly used dual-watermark strategy that adds a robust watermark for image ownership verification, where mutual interference and limited embedding capacity reduce the fragile watermark's effectiveness. To address the gap, we propose RecoverMark, a watermarking framework that achieves robust manipulation localization, content recovery, and ownership verification simultaneously. Our key insight is twofold. First, we exploit a critical real-world constraint: an adversary must preserve the background's semantic consistency to avoid visual detection, even if they apply global, imperceptible watermark removal attacks. Second, using the image's own content (face, in this paper) as the watermark enhances extraction robustness. Based on these insights, RecoverMark treats the protected face content itself as the watermark and embeds it into the surrounding background. By designing a robust two-stage training paradigm with carefully crafted distortion layers that simulate comprehensive potential attacks and a progressive training strategy, RecoverMark achieves a robust watermark embedding in no fragile manner for image manipulation localization, recovery, and image IP protection simultaneously. Extensive experiments demonstrate the proposed RecoverMark's robustness against both seen and unseen attacks and its generalizability to in-distribution and out-of-distribution data.
>
---
#### [new 079] MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision
- **分类: cs.CV**

- **简介: 该论文属于边缘检测任务，旨在解决传统方法依赖非可微后处理的问题。提出一种轻量级端到端匹配监督模块，提升边缘检测精度。**

- **链接: [https://arxiv.org/pdf/2602.20689v1](https://arxiv.org/pdf/2602.20689v1)**

> **作者:** Bedrettin Cetinkaya; Sinan Kalkan; Emre Akbas
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Generating crisp, i.e., one-pixel-wide, edge maps remains one of the fundamental challenges in edge detection, affecting both traditional and learning-based methods. To obtain crisp edges, most existing approaches rely on two hand-crafted post-processing algorithms, Non-Maximum Suppression (NMS) and skeleton-based thinning, which are non-differentiable and hinder end-to-end optimization. Moreover, all existing crisp edge detection methods still depend on such post-processing to achieve satisfactory results. To address this limitation, we propose \MethodLPP, a lightweight, only $\sim$21K additional parameters, and plug-and-play matching-based supervision module that can be appended to any edge detection model for joint end-to-end learning of crisp edges. At each training iteration, \MethodLPP performs one-to-one matching between predicted and ground-truth edges based on spatial distance and confidence, ensuring consistency between training and testing protocols. Extensive experiments on four popular datasets demonstrate that integrating \MethodLPP substantially improves the performance of existing edge detection models. In particular, \MethodLPP increases the Average Crispness (AC) metric by up to 2--4$\times$ compared to baseline models. Under the crispness-emphasized evaluation (CEval), \MethodLPP further boosts baseline performance by up to 20--35\% in ODS and achieves similar gains in OIS and AP, achieving SOTA performance that matches or surpasses standard post-processing for the first time. Code is available at https://cvpr26-matched.github.io.
>
---
#### [new 080] PropFly: Learning to Propagate via On-the-Fly Supervision from Pre-trained Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，解决缺乏配对数据的问题。提出PropFly框架，利用预训练视频扩散模型实时生成训练样本，实现高效视频传播编辑。**

- **链接: [https://arxiv.org/pdf/2602.20583v1](https://arxiv.org/pdf/2602.20583v1)**

> **作者:** Wonyong Seo; Jaeho Moon; Jaehyup Lee; Soo Ye Kim; Munchurl Kim
>
> **备注:** The first two authors contributed equally to this work (equal contribution)
>
> **摘要:** Propagation-based video editing enables precise user control by propagating a single edited frame into following frames while maintaining the original context such as motion and structures. However, training such models requires large-scale, paired (source and edited) video datasets, which are costly and complex to acquire. Hence, we propose the PropFly, a training pipeline for Propagation-based video editing, relying on on-the-Fly supervision from pre-trained video diffusion models (VDMs) instead of requiring off-the-shelf or precomputed paired video editing datasets. Specifically, our PropFly leverages one-step clean latent estimations from intermediate noised latents with varying Classifier-Free Guidance (CFG) scales to synthesize diverse pairs of 'source' (low-CFG) and 'edited' (high-CFG) latents on-the-fly. The source latent serves as structural information of the video, while the edited latent provides the target transformation for learning propagation. Our pipeline enables an additional adapter attached to the pre-trained VDM to learn to propagate edits via Guidance-Modulated Flow Matching (GMFM) loss, which guides the model to replicate the target transformation. Our on-the-fly supervision ensures the model to learn temporally consistent and dynamic transformations. Extensive experiments demonstrate that our PropFly significantly outperforms the state-of-the-art methods on various video editing tasks, producing high-quality editing results.
>
---
#### [new 081] SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking
- **分类: cs.CV**

- **简介: 该论文提出SIMSPINE框架，解决3D脊柱运动标注与基准问题。通过生物力学模拟生成脊椎关键点，构建首个公开数据集，提升脊柱运动估计精度。**

- **链接: [https://arxiv.org/pdf/2602.20792v1](https://arxiv.org/pdf/2602.20792v1)**

> **作者:** Muhammad Saif Ullah Khan; Didier Stricker
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Modeling spinal motion is fundamental to understanding human biomechanics, yet remains underexplored in computer vision due to the spine's complex multi-joint kinematics and the lack of large-scale 3D annotations. We present a biomechanics-aware keypoint simulation framework that augments existing human pose datasets with anatomically consistent 3D spinal keypoints derived from musculoskeletal modeling. Using this framework, we create the first open dataset, named SIMSPINE, which provides sparse vertebra-level 3D spinal annotations for natural full-body motions in indoor multi-camera capture without external restraints. With 2.14 million frames, this enables data-driven learning of vertebral kinematics from subtle posture variations and bridges the gap between musculoskeletal simulation and computer vision. In addition, we release pretrained baselines covering fine-tuned 2D detectors, monocular 3D pose lifting models, and multi-view reconstruction pipelines, establishing a unified benchmark for biomechanically valid spine motion estimation. Specifically, our 2D spine baselines improve the state-of-the-art from 0.63 to 0.80 AUC in controlled environments, and from 0.91 to 0.93 AP for in-the-wild spine tracking. Together, the simulation framework and SIMSPINE dataset advance research in vision-based biomechanics, motion analysis, and digital human modeling by enabling reproducible, anatomically grounded 3D spine estimation under natural conditions.
>
---
#### [new 082] OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI Interpretation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出OrthoDiffusion，用于骨科MRI的多任务分析，解决诊断准确性和效率问题。通过扩散模型实现结构分割和异常检测，并具备良好的泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.20752v1](https://arxiv.org/pdf/2602.20752v1)**

> **作者:** Tian Lan; Lei Xu; Zimu Yuan; Shanggui Liu; Jiajun Liu; Jiaxin Liu; Weilai Xiang; Hongyu Yang; Dong Jiang; Jianxin Yin; Dingyu Wang
>
> **摘要:** Musculoskeletal disorders represent a significant global health burden and are a leading cause of disability worldwide. While MRI is essential for accurate diagnosis, its interpretation remains exceptionally challenging. Radiologists must identify multiple potential abnormalities within complex anatomical structures across different imaging planes, a process that requires significant expertise and is prone to variability. We developed OrthoDiffusion, a unified diffusion-based foundation model designed for multi-task musculoskeletal MRI interpretation. The framework utilizes three orientation-specific 3D diffusion models, pre-trained in a self-supervised manner on 15,948 unlabeled knee MRI scans, to learn robust anatomical features from sagittal, coronal, and axial views. These view-specific representations are integrated to support diverse clinical tasks, including anatomical segmentation and multi-label diagnosis. Our evaluation demonstrates that OrthoDiffusion achieves excellent performance in the segmentation of 11 knee structures and the detection of 8 knee abnormalities. The model exhibited remarkable robustness across different clinical centers and MRI field strengths, consistently outperforming traditional supervised models. Notably, in settings where labeled data was scarce, OrthoDiffusion maintained high diagnostic precision using only 10\% of training labels. Furthermore, the anatomical representations learned from knee imaging proved highly transferable to other joints, achieving strong diagnostic performance across 11 diseases of the ankle and shoulder. These findings suggest that diffusion-based foundation models can serve as a unified platform for multi-disease diagnosis and anatomical segmentation, potentially improving the efficiency and accuracy of musculoskeletal MRI interpretation in real-world clinical workflows.
>
---
#### [new 083] CrystaL: Spontaneous Emergence of Visual Latents in MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CrystaL框架，解决MLLMs中视觉信息在隐状态中丢失的问题，通过双路径对齐提升视觉语义理解与推理能力。**

- **链接: [https://arxiv.org/pdf/2602.20980v1](https://arxiv.org/pdf/2602.20980v1)**

> **作者:** Yang Zhang; Danyang Li; Yuxuan Li; Xin Zhang; Tianyu Xie; Mingming Cheng; Xiang Li
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable performance by integrating powerful language backbones with large-scale visual encoders. Among these, latent Chain-of-Thought (CoT) methods enable implicit reasoning in continuous hidden states, facilitating seamless vision-language integration and faster inference. However, existing heuristically predefined supervision signals in latent CoT provide limited guidance for preserving critical visual information in intermediate latent states. To address this limitation, we propose CrystaL (Crystallized Latent Reasoning), a single-stage framework with two paths to process intact and corrupted images, respectively. By explicitly aligning the attention patterns and prediction distributions across the two paths, CrystaL crystallizes latent representations into task-relevant visual semantics, without relying on auxiliary annotations or external modules. Extensive experiments on perception-intensive benchmarks demonstrate that CrystaL consistently outperforms state-of-the-art baselines, achieving substantial gains in fine-grained visual understanding while maintaining robust reasoning capabilities.
>
---
#### [new 084] Probing and Bridging Geometry-Interaction Cues for Affordance Reasoning in Vision Foundation Models
- **分类: cs.CV**

- **简介: 该论文研究视觉基础模型中的可操作性推理任务，旨在理解物体如何被交互。通过分析几何与交互感知，提出融合方法提升可操作性估计效果。**

- **链接: [https://arxiv.org/pdf/2602.20501v1](https://arxiv.org/pdf/2602.20501v1)**

> **作者:** Qing Zhang; Xuesong Li; Jing Zhang
>
> **备注:** 11 pages, 12 figures, Accepted to CVPR 2026
>
> **摘要:** What does it mean for a visual system to truly understand affordance? We argue that this understanding hinges on two complementary capacities: geometric perception, which identifies the structural parts of objects that enable interaction, and interaction perception, which models how an agent's actions engage with those parts. To test this hypothesis, we conduct a systematic probing of Visual Foundation Models (VFMs). We find that models like DINO inherently encode part-level geometric structures, while generative models like Flux contain rich, verb-conditioned spatial attention maps that serve as implicit interaction priors. Crucially, we demonstrate that these two dimensions are not merely correlated but are composable elements of affordance. By simply fusing DINO's geometric prototypes with Flux's interaction maps in a training-free and zero-shot manner, we achieve affordance estimation competitive with weakly-supervised methods. This final fusion experiment confirms that geometric and interaction perception are the fundamental building blocks of affordance understanding in VFMs, providing a mechanistic account of how perception grounds action.
>
---
#### [new 085] Human Video Generation from a Single Image with 3D Pose and View Control
- **分类: cs.CV**

- **简介: 该论文属于图像到视频生成任务，旨在解决从单张图像生成高质量、视角一致的人类视频的问题。提出HVG模型，通过3D姿态和视角控制实现多视角、时空连贯的视频生成。**

- **链接: [https://arxiv.org/pdf/2602.21188v1](https://arxiv.org/pdf/2602.21188v1)**

> **作者:** Tiantian Wang; Chun-Han Yao; Tao Hu; Mallikarjun Byrasandra Ramalinga Reddy; Ming-Hsuan Yang; Varun Jampani
>
> **摘要:** Recent diffusion methods have made significant progress in generating videos from single images due to their powerful visual generation capabilities. However, challenges persist in image-to-video synthesis, particularly in human video generation, where inferring view-consistent, motion-dependent clothing wrinkles from a single image remains a formidable problem. In this paper, we present Human Video Generation in 4D (HVG), a latent video diffusion model capable of generating high-quality, multi-view, spatiotemporally coherent human videos from a single image with 3D pose and view control. HVG achieves this through three key designs: (i) Articulated Pose Modulation, which captures the anatomical relationships of 3D joints via a novel dual-dimensional bone map and resolves self-occlusions across views by introducing 3D information; (ii) View and Temporal Alignment, which ensures multi-view consistency and alignment between a reference image and pose sequences for frame-to-frame stability; and (iii) Progressive Spatio-Temporal Sampling with temporal alignment to maintain smooth transitions in long multi-view animations. Extensive experiments on image-to-video tasks demonstrate that HVG outperforms existing methods in generating high-quality 4D human videos from diverse human images and pose inputs.
>
---
#### [new 086] Seeing Through Words: Controlling Visual Retrieval Quality with Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言学习任务，旨在解决短而模糊的文本查询导致的图像检索质量不佳问题。通过语言模型扩展查询，提升检索结果的语义和质量可控性。**

- **链接: [https://arxiv.org/pdf/2602.21175v1](https://arxiv.org/pdf/2602.21175v1)**

> **作者:** Jianglin Lu; Simon Jenni; Kushal Kafle; Jing Shi; Handong Zhao; Yun Fu
>
> **摘要:** Text-to-image retrieval is a fundamental task in vision-language learning, yet in real-world scenarios it is often challenged by short and underspecified user queries. Such queries are typically only one or two words long, rendering them semantically ambiguous, prone to collisions across diverse visual interpretations, and lacking explicit control over the quality of retrieved images. To address these issues, we propose a new paradigm of quality-controllable retrieval, which enriches short queries with contextual details while incorporating explicit notions of image quality. Our key idea is to leverage a generative language model as a query completion function, extending underspecified queries into descriptive forms that capture fine-grained visual attributes such as pose, scene, and aesthetics. We introduce a general framework that conditions query completion on discretized quality levels, derived from relevance and aesthetic scoring models, so that query enrichment is not only semantically meaningful but also quality-aware. The resulting system provides three key advantages: 1) flexibility, it is compatible with any pretrained vision-language model (VLMs) without modification; 2) transparency, enriched queries are explicitly interpretable by users; and 3) controllability, enabling retrieval results to be steered toward user-preferred quality levels. Extensive experiments demonstrate that our proposed approach significantly improves retrieval results and provides effective quality control, bridging the gap between the expressive capacity of modern VLMs and the underspecified nature of short user queries. Our code is available at https://github.com/Jianglin954/QCQC.
>
---
#### [new 087] BBQ-to-Image: Numeric Bounding Box and Qolor Control in Large-Scale Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文提出BBQ模型，解决文本到图像生成中缺乏精确数值控制的问题，通过引入数值边界框和颜色信息，提升生成精度。**

- **链接: [https://arxiv.org/pdf/2602.20672v1](https://arxiv.org/pdf/2602.20672v1)**

> **作者:** Eliran Kachlon; Alexander Visheratin; Nimrod Sarid; Tal Hacham; Eyal Gutflaish; Saar Huberman; Hezi Zisman; David Ruppin; Ron Mokady
>
> **摘要:** Text-to-image models have rapidly advanced in realism and controllability, with recent approaches leveraging long, detailed captions to support fine-grained generation. However, a fundamental parametric gap remains: existing models rely on descriptive language, whereas professional workflows require precise numeric control over object location, size, and color. In this work, we introduce BBQ, a large-scale text-to-image model that directly conditions on numeric bounding boxes and RGB triplets within a unified structured-text framework. We obtain precise spatial and chromatic control by training on captions enriched with parametric annotations, without architectural modifications or inference-time optimization. This also enables intuitive user interfaces such as object dragging and color pickers, replacing ambiguous iterative prompting with precise, familiar controls. Across comprehensive evaluations, BBQ achieves strong box alignment and improves RGB color fidelity over state-of-the-art baselines. More broadly, our results support a new paradigm in which user intent is translated into an intermediate structured language, consumed by a flow-based transformer acting as a renderer and naturally accommodating numeric parameters.
>
---
#### [new 088] GatedCLIP: Gated Multimodal Fusion for Hateful Memes Detection
- **分类: cs.CV**

- **简介: 该论文属于 hateful memes 检测任务，解决多模态内容中隐含仇恨信息的识别问题。通过改进 CLIP 模型，引入动态门控融合和对比学习，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.20818v1](https://arxiv.org/pdf/2602.20818v1)**

> **作者:** Yingying Guo; Ke Zhang; Zirong Zeng
>
> **备注:** Preprint
>
> **摘要:** Detecting hateful content in multimodal memes presents unique challenges, as harmful messages often emerge from the complex interplay between benign images and text. We propose GatedCLIP, a Vision-Language model that enhances CLIP's multimodal capabilities with specialized architectural improvements for hateful memes detection. Our approach introduces learned projection heads that map CLIP embeddings to a task-optimized semantic space, a dynamic gated fusion mechanism that adaptively weights visual and textual features, and a contrastive learning objective that maintains cross-modal semantic alignment. Experiments on the Hateful Memes dataset demonstrate that GatedCLIP achieves an AUROC of 0.66, substantially outperforming the CLIP baseline (AUROC 0.49) while maintaining computational efficiency with only 350K trainable parameters.
>
---
#### [new 089] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成的安全防护任务，解决多类别有害内容冲突问题。提出CASG框架，动态识别并引导安全方向，降低有害内容生成率。**

- **链接: [https://arxiv.org/pdf/2602.20880v1](https://arxiv.org/pdf/2602.20880v1)**

> **作者:** Yongli Xiang; Ziming Hong; Zhaoqing Wang; Xiangyu Zhao; Bo Han; Tongliang Liu
>
> **备注:** CVPR 2026; Code is released at https://github.com/tmllab/2026_CVPR_CASG
>
> **摘要:** Text-to-Image (T2I) diffusion models have demonstrated significant advancements in generating high-quality images, while raising potential safety concerns regarding harmful content generation. Safety-guidance-based methods have been proposed to mitigate harmful outputs by steering generation away from harmful zones, where the zones are averaged across multiple harmful categories based on predefined keywords. However, these approaches fail to capture the complex interplay among different harm categories, leading to "harmful conflicts" where mitigating one type of harm may inadvertently amplify another, thus increasing overall harmful rate. To address this issue, we propose Conflict-aware Adaptive Safety Guidance (CASG), a training-free framework that dynamically identifies and applies the category-aligned safety direction during generation. CASG is composed of two components: (i) Conflict-aware Category Identification (CaCI), which identifies the harmful category most aligned with the model's evolving generative state, and (ii) Conflict-resolving Guidance Application (CrGA), which applies safety steering solely along the identified category to avoid multi-category interference. CASG can be applied to both latent-space and text-space safeguards. Experiments on T2I safety benchmarks demonstrate CASG's state-of-the-art performance, reducing the harmful rate by up to 15.4% compared to existing methods.
>
---
#### [new 090] The Finite Primitive Basis Theorem for Computational Imaging: Formal Foundations of the OperatorGraph Representation
- **分类: cs.CV**

- **简介: 该论文提出有限原始基定理，将计算成像模型表示为特定结构的有向无环图，解决成像模型统一表示问题，通过11个基本算子构建高效近似模型。**

- **链接: [https://arxiv.org/pdf/2602.20550v1](https://arxiv.org/pdf/2602.20550v1)**

> **作者:** Chengshuai Yang
>
> **摘要:** Computational imaging forward models, from coded aperture spectral cameras to MRI scanners, are traditionally implemented as monolithic, modality-specific codes. We prove that every forward model in a broad, precisely defined operator class Cimg (encompassing clinical, scientific, and industrial imaging modalities, both linear and nonlinear) admits an epsilon-approximate representation as a typed directed acyclic graph (DAG) whose nodes are drawn from a library of exactly 11 canonical primitives: Propagate, Modulate, Project, Encode, Convolve, Accumulate, Detect, Sample, Disperse, Scatter, and Transform. We call this the Finite Primitive Basis Theorem. The proof is constructive: we provide an algorithm that, given any H in Cimg, produces a DAG G with relative operator error at most epsilon and graph complexity within prescribed bounds. We further prove that the library is minimal: removing any single primitive causes at least one modality to lose its epsilon-approximate representation. A systematic analysis of nonlinearities in imaging physics shows they fall into two structural categories: pointwise scalar functions (handled by Transform) and self-consistent iterations (unrolled into existing linear primitives). Empirical validation on 31 linear modalities confirms eimg below 0.01 with at most 5 nodes and depth 5, and we provide constructive DAG decompositions for 9 additional nonlinear modalities. These results establish mathematical foundations for the Physics World Model (PWM) framework.
>
---
#### [new 091] RAYNOVA: 3D-Geometry-Free Auto-Regressive Driving World Modeling with Unified Spatio-Temporal Representation
- **分类: cs.CV**

- **简介: 该论文提出RAYNOVA，一种无需3D几何的自回归世界建模方法，解决多视角视频生成问题。通过统一时空表示和递归训练，提升生成质量和泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.20685v1](https://arxiv.org/pdf/2602.20685v1)**

> **作者:** Yichen Xie; Chensheng Peng; Mazen Abdelfattah; Yihan Hu; Jiezhi Yang; Eric Higgins; Ryan Brigden; Masayoshi Tomizuka; Wei Zhan
>
> **备注:** Accepted by CVPR 2026; Project website: http://yichen928.github.io/raynova
>
> **摘要:** World foundation models aim to simulate the evolution of the real world with physically plausible behavior. Unlike prior methods that handle spatial and temporal correlations separately, we propose RAYNOVA, a geometry-free world model that employs a dual-causal autoregressive framework. It follows both scale-wise and temporal topological orders in the autoregressive process, and leverages global attention for unified 4D spatio-temporal reasoning. Different from existing works that impose strong 3D geometric priors, RAYNOVA constructs an isotropic spatio-temporal representation across views, frames, and scales based on relative Plücker-ray positional encoding, enabling robust generalization to diverse camera setups and ego motions. We further introduce a recurrent training paradigm to alleviate distribution drift in long-horizon video generation. RAYNOVA achieves state-of-the-art multi-view video generation results on nuScenes, while offering higher throughput and strong controllability under diverse input conditions, generalizing to novel views and camera configurations without explicit 3D scene representation. Our code will be released at http://yichen928.github.io/raynova.
>
---
#### [new 092] Skullptor: High Fidelity 3D Head Reconstruction in Seconds with Multi-View Normal Prediction
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D人脸重建任务，解决单图与多图重建的精度与效率问题。提出混合方法，结合多视角法线预测与优化框架，实现高保真快速重建。**

- **链接: [https://arxiv.org/pdf/2602.21100v1](https://arxiv.org/pdf/2602.21100v1)**

> **作者:** Noé Artru; Rukhshanda Hussain; Emeline Got; Alexandre Messier; David B. Lindell; Abdallah Dib
>
> **备注:** 14 pages, 8 figures, to be published in proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
>
> **摘要:** Reconstructing high-fidelity 3D head geometry from images is critical for a wide range of applications, yet existing methods face fundamental limitations. Traditional photogrammetry achieves exceptional detail but requires extensive camera arrays (25-200+ views), substantial computation, and manual cleanup in challenging areas like facial hair. Recent alternatives present a fundamental trade-off: foundation models enable efficient single-image reconstruction but lack fine geometric detail, while optimization-based methods achieve higher fidelity but require dense views and expensive computation. We bridge this gap with a hybrid approach that combines the strengths of both paradigms. Our method introduces a multi-view surface normal prediction model that extends monocular foundation models with cross-view attention to produce geometrically consistent normals in a feed-forward pass. We then leverage these predictions as strong geometric priors within an inverse rendering optimization framework to recover high-frequency surface details. Our approach outperforms state-of-the-art single-image and multi-view methods, achieving high-fidelity reconstruction on par with dense-view photogrammetry while reducing camera requirements and computational cost. The code and model will be released.
>
---
#### [new 093] Leveraging Causal Reasoning Method for Explaining Medical Image Segmentation Models
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决模型解释性不足的问题。通过因果推理方法，量化输入区域和网络组件对分割结果的影响，提升模型可信度。**

- **链接: [https://arxiv.org/pdf/2602.20511v1](https://arxiv.org/pdf/2602.20511v1)**

> **作者:** Limai Jiang; Ruitao Xie; Bokai Yang; Huazhen Huang; Juan He; Yufu Huo; Zikai Wang; Yang Wei; Yunpeng Cai
>
> **备注:** Preprint
>
> **摘要:** Medical image segmentation plays a vital role in clinical decision-making, enabling precise localization of lesions and guiding interventions. Despite significant advances in segmentation accuracy, the black-box nature of most deep models has raised growing concerns about their trustworthiness in high-stakes medical scenarios. Current explanation techniques have primarily focused on classification tasks, leaving the segmentation domain relatively underexplored. We introduced an explanation model for segmentation task which employs the causal inference framework and backpropagates the average treatment effect (ATE) into a quantification metric to determine the influence of input regions, as well as network components, on target segmentation areas. Through comparison with recent segmentation explainability techniques on two representative medical imaging datasets, we demonstrated that our approach provides more faithful explanations than existing approaches. Furthermore, we carried out a systematic causal analysis of multiple foundational segmentation models using our method, which reveals significant heterogeneity in perceptual strategies across different models, and even between different inputs for the same model. Suggesting the potential of our method to provide notable insights for optimizing segmentation models. Our code can be found at https://github.com/lcmmai/PdCR.
>
---
#### [new 094] See and Fix the Flaws: Enabling VLMs and Diffusion Models to Comprehend Visual Artifacts via Agentic Data Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决AI生成图像中的视觉伪影问题。提出ArtiAgent，自动合成带伪影的图像数据集，提升模型对伪影的理解与修复能力。**

- **链接: [https://arxiv.org/pdf/2602.20951v1](https://arxiv.org/pdf/2602.20951v1)**

> **作者:** Jaehyun Park; Minyoung Ahn; Minkyu Kim; Jonghyun Lee; Jae-Gil Lee; Dongmin Park
>
> **摘要:** Despite recent advances in diffusion models, AI generated images still often contain visual artifacts that compromise realism. Although more thorough pre-training and bigger models might reduce artifacts, there is no assurance that they can be completely eliminated, which makes artifact mitigation a highly crucial area of study. Previous artifact-aware methodologies depend on human-labeled artifact datasets, which are costly and difficult to scale, underscoring the need for an automated approach to reliably acquire artifact-annotated datasets. In this paper, we propose ArtiAgent, which efficiently creates pairs of real and artifact-injected images. It comprises three agents: a perception agent that recognizes and grounds entities and subentities from real images, a synthesis agent that introduces artifacts via artifact injection tools through novel patch-wise embedding manipulation within a diffusion transformer, and a curation agent that filters the synthesized artifacts and generates both local and global explanations for each instance. Using ArtiAgent, we synthesize 100K images with rich artifact annotations and demonstrate both efficacy and versatility across diverse applications. Code is available at link.
>
---
#### [new 095] Monocular Endoscopic Tissue 3D Reconstruction with Multi-Level Geometry Regularization
- **分类: cs.CV**

- **简介: 该论文属于内窥镜组织3D重建任务，旨在解决软组织表面重建不一致和实时渲染不足的问题。通过引入多层级几何约束，提升重建质量和实时性。**

- **链接: [https://arxiv.org/pdf/2602.20718v1](https://arxiv.org/pdf/2602.20718v1)**

> **作者:** Yangsen Chen; Hao Wang
>
> **备注:** ijcnn 2025
>
> **摘要:** Reconstructing deformable endoscopic tissues is crucial for achieving robot-assisted surgery. However, 3D Gaussian Splatting-based approaches encounter challenges in achieving consistent tissue surface reconstruction, while existing NeRF-based methods lack real-time rendering capabilities. In pursuit of both smooth deformable surfaces and real-time rendering, we introduce a novel approach based on 3D Gaussian Splatting. Specifically, we introduce surface-aware reconstruction, initially employing a Sign Distance Field-based method to construct a mesh, subsequently utilizing this mesh to constrain the Gaussian Splatting reconstruction process. Furthermore, to ensure the generation of physically plausible deformations, we incorporate local rigidity and global non-rigidity restrictions to guide Gaussian deformation, tailored for the highly deformable nature of soft endoscopic tissue. Based on 3D Gaussian Splatting, our proposed method delivers a fast rendering process and smooth surface appearances. Quantitative and qualitative analysis against alternative methodologies shows that our approach achieves solid reconstruction quality in both textures and geometries.
>
---
#### [new 096] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出SpatiaLQA基准，用于评估视觉语言模型的空间逻辑推理能力。针对现有模型在复杂场景中空间关系和步骤依赖理解不足的问题，构建了包含9605个问答对的数据集，并提出递归场景图辅助推理方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.20901v1](https://arxiv.org/pdf/2602.20901v1)**

> **作者:** Yuechen Xie; Xiaoyan Zhang; Yicheng Shan; Hao Zhu; Rui Tang; Rong Wei; Mingli Song; Yuanyu Wan; Jie Song
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Vision-Language Models (VLMs) have been increasingly applied in real-world scenarios due to their outstanding understanding and reasoning capabilities. Although VLMs have already demonstrated impressive capabilities in common visual question answering and logical reasoning, they still lack the ability to make reasonable decisions in complex real-world environments. We define this ability as spatial logical reasoning, which not only requires understanding the spatial relationships among objects in complex scenes, but also the logical dependencies between steps in multi-step tasks. To bridge this gap, we introduce Spatial Logical Question Answering (SpatiaLQA), a benchmark designed to evaluate the spatial logical reasoning capabilities of VLMs. SpatiaLQA consists of 9,605 question answer pairs derived from 241 real-world indoor scenes. We conduct extensive experiments on 41 mainstream VLMs, and the results show that even the most advanced models still struggle with spatial logical reasoning. To address this issue, we propose a method called recursive scene graph assisted reasoning, which leverages visual foundation models to progressively decompose complex scenes into task-relevant scene graphs, thereby enhancing the spatial logical reasoning ability of VLMs, outperforming all previous methods. Code and dataset are available at https://github.com/xieyc99/SpatiaLQA.
>
---
#### [new 097] OCR-Agent: Agentic OCR with Capability and Memory Reflection
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，解决模型缺乏自我修正机制的问题。通过引入能力与记忆反思框架，提升模型的推理鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.21053v1](https://arxiv.org/pdf/2602.21053v1)**

> **作者:** Shimin Wen; Zeyu Zhang; Xingdou Bian; Hongjie Zhu; Lulu He; Layi Shama; Daji Ergu; Ying Cai
>
> **摘要:** Large Vision-Language Models (VLMs) have demonstrated significant potential on complex visual understanding tasks through iterative optimization methods.However, these models generally lack effective self-correction mechanisms, making it difficult for them to independently rectify cognitive biases. Consequently, during multi-turn revisions, they often fall into repetitive and ineffective attempts, failing to achieve stable improvements in answer quality.To address this issue, we propose a novel iterative self-correction framework that endows models with two key capabilities: Capability Reflection and Memory Reflection. This framework guides the model to first diagnose errors and generate a correction plan via Capability Reflection, then leverage Memory Reflection to review past attempts to avoid repetition and explore new solutions, and finally, optimize the answer through rigorous re-reasoning. Experiments on the challenging OCRBench v2 benchmark show that OCR-Agent outperforms the current open-source SOTA model InternVL3-8B by +2.0 on English and +1.2 on Chinese subsets, while achieving state-of-the-art results in Visual Understanding (79.9) and Reasoning (66.5) - surpassing even larger fine-tuned models. Our method demonstrates that structured, self-aware reflection can significantly enhance VLMs' reasoning robustness without additional training. Code: https://github.com/AIGeeksGroup/OCR-Agent.
>
---
#### [new 098] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉语言模型解释任务，旨在解决VLMs透明度不足的问题。通过构建电路追踪框架，分析多模态推理机制，验证其因果性和可控性。**

- **链接: [https://arxiv.org/pdf/2602.20330v1](https://arxiv.org/pdf/2602.20330v1)**

> **作者:** Jingcheng Yang; Tianhu Xiong; Shengyi Qian; Klara Nahrstedt; Mingyuan Wu
>
> **备注:** To appear in the Findings of CVPR 2026
>
> **摘要:** Vision-language models (VLMs) are powerful but remain opaque black boxes. We introduce the first framework for transparent circuit tracing in VLMs to systematically analyze multimodal reasoning. By utilizing transcoders, attribution graphs, and attention-based methods, we uncover how VLMs hierarchically integrate visual and semantic concepts. We reveal that distinct visual feature circuits can handle mathematical reasoning and support cross-modal associations. Validated through feature steering and circuit patching, our framework proves these circuits are causal and controllable, laying the groundwork for more explainable and reliable VLMs.
>
---
#### [new 099] VAGNet: Grounding 3D Affordance from Human-Object Interactions in Videos
- **分类: cs.CV**

- **简介: 该论文提出VAGNet，解决3D物体交互区域定位问题，通过视频动态信息提升静态方法的不足。**

- **链接: [https://arxiv.org/pdf/2602.20608v1](https://arxiv.org/pdf/2602.20608v1)**

> **作者:** Aihua Mao; Kaihang Huang; Yong-Jin Liu; Chee Seng Chan; Ying He
>
> **摘要:** 3D object affordance grounding aims to identify regions on 3D objects that support human-object interaction (HOI), a capability essential to embodied visual reasoning. However, most existing approaches rely on static visual or textual cues, neglecting that affordances are inherently defined by dynamic actions. As a result, they often struggle to localize the true contact regions involved in real interactions. We take a different perspective. Humans learn how to use objects by observing and imitating actions, not just by examining shapes. Motivated by this intuition, we introduce video-guided 3D affordance grounding, which leverages dynamic interaction sequences to provide functional supervision. To achieve this, we propose VAGNet, a framework that aligns video-derived interaction cues with 3D structure to resolve ambiguities that static cues cannot address. To support this new setting, we introduce PVAD, the first HOI video-3D pairing affordance dataset, providing functional supervision unavailable in prior works. Extensive experiments on PVAD show that VAGNet achieves state-of-the-art performance, significantly outperforming static-based baselines. The code and dataset will be open publicly.
>
---
#### [new 100] CleanStyle: Plug-and-Play Style Conditioning Purification for Text-to-Image Stylization
- **分类: cs.CV**

- **简介: 该论文属于文本到图像风格化任务，解决风格迁移中的内容泄露问题。提出CleanStyle框架，通过过滤风格嵌入中的噪声提升生成质量与风格一致性。**

- **链接: [https://arxiv.org/pdf/2602.20721v1](https://arxiv.org/pdf/2602.20721v1)**

> **作者:** Xiaoman Feng; Mingkun Lei; Yang Wang; Dingwen Fu; Chi Zhang
>
> **备注:** 26 pages
>
> **摘要:** Style transfer in diffusion models enables controllable visual generation by injecting the style of a reference image. However, recent encoder-based methods, while efficient and tuning-free, often suffer from content leakage, where semantic elements from the style image undesirably appear in the output, impairing prompt fidelity and stylistic consistency. In this work, we introduce CleanStyle, a plug-and-play framework that filters out content-related noise from the style embedding without retraining. Motivated by empirical analysis, we observe that such leakage predominantly stems from the tail components of the style embedding, which are isolated via Singular Value Decomposition (SVD). To address this, we propose CleanStyleSVD (CS-SVD), which dynamically suppresses tail components using a time-aware exponential schedule, providing clean, style-preserving conditional embeddings throughout the denoising process. Furthermore, we present Style-Specific Classifier-Free Guidance (SS-CFG), which reuses the suppressed tail components to construct style-aware unconditional inputs. Unlike conventional methods that use generic negative embeddings (e.g., zero vectors), SS-CFG introduces targeted negative signals that reflect style-specific but prompt-irrelevant visual elements. This enables the model to effectively suppress these distracting patterns during generation, thereby improving prompt fidelity and enhancing the overall visual quality of stylized outputs. Our approach is lightweight, interpretable, and can be seamlessly integrated into existing encoder-based diffusion models without retraining. Extensive experiments demonstrate that CleanStyle substantially reduces content leakage, improves stylization quality and improves prompt alignment across a wide range of style references and prompts.
>
---
#### [new 101] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学图像分割任务，旨在解决标注数据少、领域差异大等问题。提出MedCLIPSeg框架，利用CLIP模型实现高效、可解释的分割。**

- **链接: [https://arxiv.org/pdf/2602.20423v1](https://arxiv.org/pdf/2602.20423v1)**

> **作者:** Taha Koleilat; Hojat Asgariandehkordi; Omid Nejati Manzari; Berardino Barile; Yiming Xiao; Hassan Rivaz
>
> **备注:** CVPR 2026; Project Page: https://tahakoleilat.github.io/MedCLIPSeg
>
> **摘要:** Medical image segmentation remains challenging due to limited annotations for training, ambiguous anatomical features, and domain shifts. While vision-language models such as CLIP offer strong cross-modal representations, their potential for dense, text-guided medical image segmentation remains underexplored. We present MedCLIPSeg, a novel framework that adapts CLIP for robust, data-efficient, and uncertainty-aware medical image segmentation. Our approach leverages patch-level CLIP embeddings through probabilistic cross-modal attention, enabling bidirectional interaction between image and text tokens and explicit modeling of predictive uncertainty. Together with a soft patch-level contrastive loss that encourages more nuanced semantic learning across diverse textual prompts, MedCLIPSeg effectively improves data efficiency and domain generalizability. Extensive experiments across 16 datasets spanning five imaging modalities and six organs demonstrate that MedCLIPSeg outperforms prior methods in accuracy, efficiency, and robustness, while providing interpretable uncertainty maps that highlight local reliability of segmentation results. This work demonstrates the potential of probabilistic vision-language modeling for text-driven medical image segmentation.
>
---
#### [new 102] XMorph: Explainable Brain Tumor Analysis Via LLM-Assisted Hybrid Deep Intelligence
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出XMorph，用于脑肿瘤分类任务，解决AI模型可解释性差和计算效率低的问题，通过结合边界分析与AI解释模块提升诊断可靠性。**

- **链接: [https://arxiv.org/pdf/2602.21178v1](https://arxiv.org/pdf/2602.21178v1)**

> **作者:** Sepehr Salem Ghahfarokhi; M. Moein Esfahani; Raj Sunderraman; Vince Calhoun; Mohammed Alser
>
> **备注:** Accepted in ICCABS 2026: The 14th International Conference on Computational Advances in Bio and Medical Sciences
>
> **摘要:** Deep learning has significantly advanced automated brain tumor diagnosis, yet clinical adoption remains limited by interpretability and computational constraints. Conventional models often act as opaque ''black boxes'' and fail to quantify the complex, irregular tumor boundaries that characterize malignant growth. To address these challenges, we present XMorph, an explainable and computationally efficient framework for fine-grained classification of three prominent brain tumor types: glioma, meningioma, and pituitary tumors. We propose an Information-Weighted Boundary Normalization (IWBN) mechanism that emphasizes diagnostically relevant boundary regions alongside nonlinear chaotic and clinically validated features, enabling a richer morphological representation of tumor growth. A dual-channel explainable AI module combines GradCAM++ visual cues with LLM-generated textual rationales, translating model reasoning into clinically interpretable insights. The proposed framework achieves a classification accuracy of 96.0%, demonstrating that explainability and high performance can co-exist in AI-based medical imaging systems. The source code and materials for XMorph are all publicly available at: https://github.com/ALSER-Lab/XMorph.
>
---
#### [new 103] Real-time Motion Segmentation with Event-based Normal Flow
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于运动分割任务，旨在解决事件相机在实时场景中运动分割效率低的问题。通过引入法线流作为中间表示，提出一种基于法线流的运动分割框架，提升处理速度与准确性。**

- **链接: [https://arxiv.org/pdf/2602.20790v1](https://arxiv.org/pdf/2602.20790v1)**

> **作者:** Sheng Zhong; Zhongyang Ren; Xiya Zhu; Dehao Yuan; Cornelia Fermuller; Yi Zhou
>
> **摘要:** Event-based cameras are bio-inspired sensors with pixels that independently and asynchronously respond to brightness changes at microsecond resolution, offering the potential to handle visual tasks in challenging scenarios. However, due to the sparse information content in individual events, directly processing the raw event data to solve vision tasks is highly inefficient, which severely limits the applicability of state-of-the-art methods in real-time tasks, such as motion segmentation, a fundamental task for dynamic scene understanding. Incorporating normal flow as an intermediate representation to compress motion information from event clusters within a localized region provides a more effective solution. In this work, we propose a normal flow-based motion segmentation framework for event-based vision. Leveraging the dense normal flow directly learned from event neighborhoods as input, we formulate the motion segmentation task as an energy minimization problem solved via graph cuts, and optimize it iteratively with normal flow clustering and motion model fitting. By using a normal flow-based motion model initialization and fitting method, the proposed system is able to efficiently estimate the motion models of independently moving objects with only a limited number of candidate models, which significantly reduces the computational complexity and ensures real-time performance, achieving nearly a 800x speedup in comparison to the open-source state-of-the-art method. Extensive evaluations on multiple public datasets fully demonstrate the accuracy and efficiency of our framework.
>
---
#### [new 104] Onboard-Targeted Segmentation of Straylight in Space Camera Sensors
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分割任务，旨在解决空间相机中杂散光的识别问题。通过AI方法进行语义分割，提升模型在资源受限环境下的性能。**

- **链接: [https://arxiv.org/pdf/2602.20709v1](https://arxiv.org/pdf/2602.20709v1)**

> **作者:** Riccardo Gallon; Fabian Schiemenz; Alessandra Menicucci; Eberhard Gill
>
> **备注:** Submitted to Aerospace Science and Technology
>
> **摘要:** This study details an artificial intelligence (AI)-based methodology for the semantic segmentation of space camera faults. Specifically, we address the segmentation of straylight effects induced by solar presence around the camera's Field of View (FoV). Anomalous images are sourced from our published dataset. Our approach emphasizes generalization across diverse flare textures, leveraging pre-training on a public dataset (Flare7k++) including flares in various non-space contexts to mitigate the scarcity of realistic space-specific data. A DeepLabV3 model with MobileNetV3 backbone performs the segmentation task. The model design targets deployment in spacecraft resource-constrained hardware. Finally, based on a proposed interface between our model and the onboard navigation pipeline, we develop custom metrics to assess the model's performance in the system-level context.
>
---
#### [new 105] Sample-efficient evidence estimation of score based priors for model selection
- **分类: cs.LG; cs.CV; stat.ME**

- **简介: 该论文属于模型选择任务，解决扩散先验下模型证据估计问题。提出一种高效估算方法，利用后验采样中间样本，仅需少量样本即可准确估计模型证据，用于选择合适先验并诊断先验不匹配。**

- **链接: [https://arxiv.org/pdf/2602.20549v1](https://arxiv.org/pdf/2602.20549v1)**

> **作者:** Frederic Wang; Katherine L. Bouman
>
> **备注:** ICLR 2026
>
> **摘要:** The choice of prior is central to solving ill-posed imaging inverse problems, making it essential to select one consistent with the measurements $y$ to avoid severe bias. In Bayesian inverse problems, this could be achieved by evaluating the model evidence $p(y \mid M)$ under different models $M$ that specify the prior and then selecting the one with the highest value. Diffusion models are the state-of-the-art approach to solving inverse problems with a data-driven prior; however, directly computing the model evidence with respect to a diffusion prior is intractable. Furthermore, most existing model evidence estimators require either many pointwise evaluations of the unnormalized prior density or an accurate clean prior score. We propose \method, an estimator of the model evidence of a diffusion prior by integrating over the time-marginals of posterior sampling methods. Our method leverages the large amount of intermediate samples naturally obtained during the reverse diffusion sampling process to obtain an accurate estimation of the model evidence using only a handful of posterior samples (e.g., 20). We also demonstrate how to implement our estimator in tandem with recent diffusion posterior sampling methods. Empirically, our estimator matches the model evidence when it can be computed analytically, and it is able to both select the correct diffusion model prior and diagnose prior misfit under different highly ill-conditioned, non-linear inverse problems, including a real-world black hole imaging problem.
>
---
#### [new 106] Estimation of Confidence Bounds in Binary Classification using Wilson Score Kernel Density Estimation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于二分类任务，旨在解决可靠置信度边界估计问题。提出Wilson Score核密度分类方法，用于提高分类系统的可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2602.20947v1](https://arxiv.org/pdf/2602.20947v1)**

> **作者:** Thorbjørn Mosekjær Iversen; Zebin Duan; Frederik Hagelskjær
>
> **摘要:** The performance and ease of use of deep learning-based binary classifiers have improved significantly in recent years. This has opened up the potential for automating critical inspection tasks, which have traditionally only been trusted to be done manually. However, the application of binary classifiers in critical operations depends on the estimation of reliable confidence bounds such that system performance can be ensured up to a given statistical significance. We present Wilson Score Kernel Density Classification, which is a novel kernel-based method for estimating confidence bounds in binary classification. The core of our method is the Wilson Score Kernel Density Estimator, which is a function estimator for estimating confidence bounds in Binomial experiments with conditionally varying success probabilities. Our method is evaluated in the context of selective classification on four different datasets, illustrating its use as a classification head of any feature extractor, including vision foundation models. Our proposed method shows similar performance to Gaussian Process Classification, but at a lower computational complexity.
>
---
#### [new 107] Multimodal MRI Report Findings Supervised Brain Lesion Segmentation with Substructures
- **分类: eess.IV; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决脑肿瘤分割中标签不全的问题。通过引入一种新的报告监督方法（MS-RSuper），结合多模态MRI信息与报告中的定量和定性线索，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2602.20994v1](https://arxiv.org/pdf/2602.20994v1)**

> **作者:** Yubin Ge; Yongsong Huang; Xiaofeng Liu
>
> **备注:** IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** Report-supervised (RSuper) learning seeks to alleviate the need for dense tumor voxel labels with constraints derived from radiology reports (e.g., volumes, counts, sizes, locations). In MRI studies of brain tumors, however, we often involve multi-parametric scans and substructures. Here, fine-grained modality/parameter-wise reports are usually provided along with global findings and are correlated with different substructures. Moreover, the reports often describe only the largest lesion and provide qualitative or uncertain cues (``mild,'' ``possible''). Classical RSuper losses (e.g., sum volume consistency) can over-constrain or hallucinate unreported findings under such incompleteness, and are unable to utilize these hierarchical findings or exploit the priors of varied lesion types in a merged dataset. We explicitly parse the global quantitative and modality-wise qualitative findings and introduce a unified, one-sided, uncertainty-aware formulation (MS-RSuper) that: (i) aligns modality-specific qualitative cues (e.g., T1c enhancement, FLAIR edema) with their corresponding substructures using existence and absence losses; (ii) enforces one-sided lower-bounds for partial quantitative cues (e.g., largest lesion size, minimal multiplicity); and (iii) adds extra- vs. intra-axial anatomical priors to respect cohort differences. Certainty tokens scale penalties; missing cues are down-weighted. On 1238 report-labeled BraTS-MET/MEN scans, our MS-RSuper largely outperforms both a sparsely-supervised baseline and a naive RSuper method.
>
---
#### [new 108] Test-Time Training with KV Binding Is Secretly Linear Attention
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究测试时训练（TTT）任务，解决其机制解释问题。工作表明TTT实际是线性注意力机制，而非记忆映射，带来简化架构和效率提升。**

- **链接: [https://arxiv.org/pdf/2602.21204v1](https://arxiv.org/pdf/2602.21204v1)**

> **作者:** Junchen Liu; Sven Elflein; Or Litany; Zan Gojcic; Ruilong Li
>
> **备注:** Webpage: https://research.nvidia.com/labs/sil/projects/tttla/
>
> **摘要:** Test-time training (TTT) with KV binding as sequence modeling layer is commonly interpreted as a form of online meta-learning that memorizes a key-value mapping at test time. However, our analysis reveals multiple phenomena that contradict this memorization-based interpretation. Motivated by these findings, we revisit the formulation of TTT and show that a broad class of TTT architectures can be expressed as a form of learned linear attention operator. Beyond explaining previously puzzling model behaviors, this perspective yields multiple practical benefits: it enables principled architectural simplifications, admits fully parallel formulations that preserve performance while improving efficiency, and provides a systematic reduction of diverse TTT variants to a standard linear attention form. Overall, our results reframe TTT not as test-time memorization, but as learned linear attention with enhanced representational capacity.
>
---
#### [new 109] LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决动态大场景下热红外相机定位与建图的难题。通过特征学习、运动跟踪和优化方法提升系统鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2602.20925v1](https://arxiv.org/pdf/2602.20925v1)**

> **作者:** Zeyu Jiang; Kuan Xu; Changhao Chen
>
> **备注:** ICRA 2026
>
> **摘要:** Thermal cameras offer strong potential for robot perception under challenging illumination and weather conditions. However, thermal Simultaneous Localization and Mapping (SLAM) remains difficult due to unreliable feature extraction, unstable motion tracking, and inconsistent global pose and map construction, particularly in dynamic large-scale outdoor environments. To address these challenges, we propose LST-SLAM, a novel large-scale stereo thermal SLAM system that achieves robust performance in complex, dynamic scenes. Our approach combines self-supervised thermal feature learning, stereo dual-level motion tracking, and geometric pose optimization. We also introduce a semantic-geometric hybrid constraint that suppresses potentially dynamic features lacking strong inter-frame geometric consistency. Furthermore, we develop an online incremental bag-of-words model for loop closure detection, coupled with global pose optimization to mitigate accumulated drift. Extensive experiments on kilometer-scale dynamic thermal datasets show that LST-SLAM significantly outperforms recent representative SLAM systems, including AirSLAM and DROID-SLAM, in both robustness and accuracy.
>
---
#### [new 110] Inspectorch: Efficient rare event exploration in solar observations
- **分类: astro-ph.SR; cs.CV**

- **简介: 该论文属于异常检测任务，旨在解决太阳观测中罕见事件识别问题。通过引入Inspectorch框架，利用流模型进行密度估计，高效识别异常光谱特征。**

- **链接: [https://arxiv.org/pdf/2602.20316v1](https://arxiv.org/pdf/2602.20316v1)**

> **作者:** C. J. Díaz Baso; I. J. Soler Poquet; C. Kuckein; M. van Noort; N. Poirier
>
> **备注:** Comments: 12+1 pages, 11+2 figures, submitted to A&A
>
> **摘要:** The Sun is observed in unprecedented detail, enabling studies of its activity on very small spatiotemporal scales. However, the large volume of data collected by our telescopes cannot be fully analyzed with conventional methods. Popular machine learning methods identify general trends from observations, but tend to overlook unusual events due to their low frequency of occurrence. We study the applicability of unsupervised probabilistic methods to efficiently identify rare events in multidimensional solar observations and optimize our computational resources to the study of these extreme phenomena. We introduce Inspectorch, an open-source framework that utilizes flow-based models: flexible density estimators capable of learning the multidimensional distribution of solar observations. Once optimized, it assigns a probability to each sample, allowing us to identify unusual events. We apply this approach by applying it to observations from the Hinode Spectro-Polarimeter, the Interface Region Imaging Spectrograph, the Microlensed Hyperspectral Imager at Swedish 1-m Solar Telescope, the Atmospheric Imaging Assembly on board the Solar Dynamics Observatory and the Extreme Ultraviolet Imager on board Solar Orbiter. We find that the algorithm assigns consistently lower probabilities to spectra that exhibit unusual features. For example, it identifies profiles with very strong Doppler shifts, uncommon broadening, and temporal dynamics associated with small-scale reconnection events, among others. As a result, Inspectorch demonstrates that density estimation using flow-based models offers a powerful approach to identifying rare events in large solar datasets. The resulting probabilistic anomaly scores allow computational resources to be focused on the most informative and physically relevant events. We make our Python package publicly available at https://github.com/cdiazbas/inspectorch.
>
---
#### [new 111] From Isolation to Integration: Building an Adaptive Expert Forest for Pre-Trained Model-based Class-Incremental Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于类增量学习任务，旨在解决模型在学习新类时遗忘旧类的问题。通过构建结构化的专家森林，实现知识共享与高效预测。**

- **链接: [https://arxiv.org/pdf/2602.20911v1](https://arxiv.org/pdf/2602.20911v1)**

> **作者:** Ruiqi Liu; Boyu Diao; Hangda Liu; Zhulin An; Fei Wang; Yongjun Xu
>
> **摘要:** Class-Incremental Learning (CIL) requires models to learn new classes without forgetting old ones. A common method is to freeze a pre-trained model and train a new, lightweight adapter for each task. While this prevents forgetting, it treats the learned knowledge as a simple, unstructured collection and fails to use the relationships between tasks. To this end, we propose the Semantic-guided Adaptive Expert Forest (SAEF), a new method that organizes adapters into a structured hierarchy for better knowledge sharing. SAEF first groups tasks into conceptual clusters based on their semantic relationships. Then, within each cluster, it builds a balanced expert tree by creating new adapters from merging the adapters of similar tasks. At inference time, SAEF finds and activates a set of relevant experts from the forest for any given input. The final prediction is made by combining the outputs of these activated experts, weighted by how confident each expert is. Experiments on several benchmark datasets show that SAEF achieves SOTA performance.
>
---
#### [new 112] BFA++: Hierarchical Best-Feature-Aware Token Prune for Multi-View Vision Language Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出BFA++，解决多视角视觉语言动作模型中的实时性问题，通过动态令牌剪枝提升计算效率和操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.20566v1](https://arxiv.org/pdf/2602.20566v1)**

> **作者:** Haosheng Li; Weixin Mao; Zihan Lan; Hongwei Xiong; Hongan Wang; Chenyang Si; Ziwei Liu; Xiaoming Deng; Hua Chen
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** Vision-Language-Action (VLA) models have achieved significant breakthroughs by leveraging Large Vision Language Models (VLMs) to jointly interpret instructions and visual inputs. However, the substantial increase in visual tokens, particularly from multi-view inputs, poses serious challenges to real-time robotic manipulation. Existing acceleration techniques for VLMs, such as token pruning, often result in degraded performance when directly applied to VLA models, as they overlook the relationships between different views and fail to account for the dynamic and task-specific characteristics of robotic operation. To address this, we propose BFA++, a dynamic token pruning framework designed specifically for VLA models. BFA++ introduces a hierarchical pruning strategy guided by two-level importance predictors: an intra-view predictor highlights task-relevant regions within each image to suppress spatial noise, while an inter-view predictor identifies critical camera views throughout different manipulation phases to reduce cross-view redundancy. This design enables efficient token selection while preserving essential visual cues, resulting in improved computational efficiency and higher manipulation success rates. Evaluations on the RoboTwin benchmark and real-world robotic tasks demonstrate that BFA++ consistently outperforms existing methods. BFA++ improves the success rate by about 10% on both the π0 and RDT models, achieving speedup of 1.8X and 1.5X, respectively. Our results highlight that context-sensitive and task-aware token pruning serves as a more effective strategy than full visual processing, enabling faster inference and improved manipulation accuracy in real-world robotic systems.
>
---
#### [new 113] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于联邦半监督学习任务，旨在解决数据异质性问题。通过引入代理框架ProxyFL，同时缓解客户端间和客户端内的分布差异。**

- **链接: [https://arxiv.org/pdf/2602.21078v1](https://arxiv.org/pdf/2602.21078v1)**

> **作者:** Duowen Chen; Yan Wang
>
> **备注:** CVPR 2026. code: https://github.com/DuowenC/FSSLlib
>
> **摘要:** Federated Semi-Supervised Learning (FSSL) aims to collaboratively train a global model across clients by leveraging partially-annotated local data in a privacy-preserving manner. In FSSL, data heterogeneity is a challenging issue, which exists both across clients and within clients. External heterogeneity refers to the data distribution discrepancy across different clients, while internal heterogeneity represents the mismatch between labeled and unlabeled data within clients. Most FSSL methods typically design fixed or dynamic parameter aggregation strategies to collect client knowledge on the server (external) and / or filter out low-confidence unlabeled samples to reduce mistakes in local client (internal). But, the former is hard to precisely fit the ideal global distribution via direct weights, and the latter results in fewer data participation into FL training. To this end, we propose a proxy-guided framework called ProxyFL that focuses on simultaneously mitigating external and internal heterogeneity via a unified proxy. I.e., we consider the learnable weights of classifier as proxy to simulate the category distribution both locally and globally. For external, we explicitly optimize global proxy against outliers instead of direct weights; for internal, we re-include the discarded samples into training by a positive-negative proxy pool to mitigate the impact of potentially-incorrect pseudo-labels. Insight experiments & theoretical analysis show our significant performance and convergence in FSSL.
>
---
#### [new 114] Progressive Per-Branch Depth Optimization for DEFOM-Stereo and SAM3 Joint Analysis in UAV Forestry Applications
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决复杂林冠下树木分支的精确重建问题。通过整合多阶段优化方法，提升点云质量，以支持无人机自主修剪。**

- **链接: [https://arxiv.org/pdf/2602.20539v1](https://arxiv.org/pdf/2602.20539v1)**

> **作者:** Yida Lin; Bing Xue; Mengjie Zhang; Sam Schofield; Richard Green
>
> **摘要:** Accurate per-branch 3D reconstruction is a prerequisite for autonomous UAV-based tree pruning; however, dense disparity maps from modern stereo matchers often remain too noisy for individual branch analysis in complex forest canopies. This paper introduces a progressive pipeline integrating DEFOM-Stereo foundation-model disparity estimation, SAM3 instance segmentation, and multi-stage depth optimization to deliver robust per-branch point clouds. Starting from a naive baseline, we systematically identify and resolve three error families through successive refinements. Mask boundary contamination is first addressed through morphological erosion and subsequently refined via a skeleton-preserving variant to safeguard thin-branch topology. Segmentation inaccuracy is then mitigated using LAB-space Mahalanobis color validation coupled with cross-branch overlap arbitration. Finally, depth noise - the most persistent error source - is initially reduced by outlier removal and median filtering, before being superseded by a robust five-stage scheme comprising MAD global detection, spatial density consensus, local MAD filtering, RGB-guided filtering, and adaptive bilateral filtering. Evaluated on 1920x1080 stereo imagery of Radiata pine (Pinus radiata) acquired with a ZED Mini camera (63 mm baseline) from a UAV in Canterbury, New Zealand, the proposed pipeline reduces the average per-branch depth standard deviation by 82% while retaining edge fidelity. The result is geometrically coherent 3D point clouds suitable for autonomous pruning tool positioning. All code and processed data are publicly released to facilitate further UAV forestry research.
>
---
#### [new 115] Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文研究机器人学习任务，旨在解决Embodied LLMs无法反思错误的问题。通过引入两种反射机制，提升机器人在执行任务中的自我修正能力。**

- **链接: [https://arxiv.org/pdf/2602.21198v1](https://arxiv.org/pdf/2602.21198v1)**

> **作者:** Yining Hong; Huang Huang; Manling Li; Li Fei-Fei; Jiajun Wu; Yejin Choi
>
> **摘要:** Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with ablative studies validating the complementary roles of reflection-in-action and reflection-on-action. Qualitative analyses, including real-robot trials, highlight behavioral correction through reflection.
>
---
#### [new 116] Multi-Vector Index Compression in Any Modality
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文属于多模态信息检索任务，解决长文档检索中计算与存储成本高的问题。通过四种压缩方法提升索引效率，其中注意力引导聚类表现最佳。**

- **链接: [https://arxiv.org/pdf/2602.21202v1](https://arxiv.org/pdf/2602.21202v1)**

> **作者:** Hanxiang Qin; Alexander Martin; Rohan Jha; Chunsheng Zuo; Reno Kriz; Benjamin Van Durme
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** We study efficient multi-vector retrieval for late interaction in any modality. Late interaction has emerged as a dominant paradigm for information retrieval in text, images, visual documents, and videos, but its computation and storage costs grow linearly with document length, making it costly for image-, video-, and audio-rich corpora. To address this limitation, we explore query-agnostic methods for compressing multi-vector document representations under a constant vector budget. We introduce four approaches for index compression: sequence resizing, memory tokens, hierarchical pooling, and a novel attention-guided clustering (AGC). AGC uses an attention-guided mechanism to identify the most semantically salient regions of a document as cluster centroids and to weight token aggregation. Evaluating these methods on retrieval tasks spanning text (BEIR), visual-document (ViDoRe), and video (MSR-VTT, MultiVENT 2.0), we show that attention-guided clustering consistently outperforms other parameterized compression methods (sequence resizing and memory tokens), provides greater flexibility in index size than non-parametric hierarchical clustering, and achieves competitive or improved performance compared to a full, uncompressed index. The source code is available at: github.com/hanxiangqin/omni-col-press.
>
---
#### [new 117] Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于视觉强化学习领域，解决机器人模拟到现实的高效训练问题。提出Squint方法，通过并行仿真和优化策略，提升训练速度与效果。**

- **链接: [https://arxiv.org/pdf/2602.21203v1](https://arxiv.org/pdf/2602.21203v1)**

> **作者:** Abdulaziz Almuzairee; Henrik I. Christensen
>
> **备注:** For website and code, see https://aalmuzairee.github.io/squint
>
> **摘要:** Visual reinforcement learning is appealing for robotics but expensive -- off-policy methods are sample-efficient yet slow; on-policy methods parallelize well but waste samples. Recent work has shown that off-policy methods can train faster than on-policy methods in wall-clock time for state-based control. Extending this to vision remains challenging, where high-dimensional input images complicate training dynamics and introduce substantial storage and encoding overhead. To address these challenges, we introduce Squint, a visual Soft Actor Critic method that achieves faster wall-clock training than prior visual off-policy and on-policy methods. Squint achieves this via parallel simulation, a distributional critic, resolution squinting, layer normalization, a tuned update-to-data ratio, and an optimized implementation. We evaluate on the SO-101 Task Set, a new suite of eight manipulation tasks in ManiSkill3 with heavy domain randomization, and demonstrate sim-to-real transfer to a real SO-101 robot. We train policies for 15 minutes on a single RTX 3090 GPU, with most tasks converging in under 6 minutes.
>
---
#### [new 118] Motivation is Something You Need
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器学习任务，旨在提升模型训练效率。通过引入动机机制，设计双模型框架，实现更高效的学习和性能优化。**

- **链接: [https://arxiv.org/pdf/2602.21064v1](https://arxiv.org/pdf/2602.21064v1)**

> **作者:** Mehdi Acheli; Walid Gaaloul
>
> **摘要:** This work introduces a novel training paradigm that draws from affective neuroscience. Inspired by the interplay of emotions and cognition in the human brain and more specifically the SEEKING motivational state, we design a dual-model framework where a smaller base model is trained continuously, while a larger motivated model is activated intermittently during predefined "motivation conditions". The framework mimics the emotional state of high curiosity and anticipation of reward in which broader brain regions are recruited to enhance cognitive performance. Exploiting scalable architectures where larger models extend smaller ones, our method enables shared weight updates and selective expansion of network capacity during noteworthy training steps. Empirical evaluation on the image classification task demonstrates that, not only does the alternating training scheme efficiently and effectively enhance the base model compared to a traditional scheme, in some cases, the motivational model also surpasses its standalone counterpart despite seeing less data per epoch. This opens the possibility of simultaneously training two models tailored to different deployment constraints with competitive or superior performance while keeping training cost lower than when training the larger model.
>
---
#### [new 119] PyVision-RL: Forging Open Agentic Vision Models via RL
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出PyVision-RL框架，解决多模态代理模型训练中的交互崩溃问题，通过强化学习提升工具使用和多轮推理能力。**

- **链接: [https://arxiv.org/pdf/2602.20739v1](https://arxiv.org/pdf/2602.20739v1)**

> **作者:** Shitian Zhao; Shaoheng Lin; Ming Li; Haoquan Zhang; Wenshuo Peng; Kaipeng Zhang; Chen Wei
>
> **备注:** preprint
>
> **摘要:** Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce PyVision-RL, a reinforcement learning framework for open-weight multimodal models that stabilizes training and sustains interaction. Our approach combines an oversampling-filtering-ranking rollout strategy with an accumulative tool reward to prevent collapse and encourage multi-turn tool use. Using a unified training pipeline, we develop PyVision-Image and PyVision-Video for image and video understanding. For video reasoning, PyVision-Video employs on-demand context construction, selectively sampling task-relevant frames during reasoning to significantly reduce visual token usage. Experiments show strong performance and improved efficiency, demonstrating that sustained interaction and on-demand visual processing are critical for scalable multimodal agents.
>
---
#### [new 120] UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，解决RGB视频缺乏3D结构的问题。提出UniLACT模型，结合深度信息进行统一潜在动作学习，提升操作精度。**

- **链接: [https://arxiv.org/pdf/2602.20231v1](https://arxiv.org/pdf/2602.20231v1)**

> **作者:** Manish Kumar Govind; Dominick Reilly; Pu Wang; Srijan Das
>
> **备注:** https://manishgovind.github.io/unilact-vla/
>
> **摘要:** Latent action representations learned from unlabeled videos have recently emerged as a promising paradigm for pretraining vision-language-action (VLA) models without explicit robot action supervision. However, latent actions derived solely from RGB observations primarily encode appearance-driven dynamics and lack explicit 3D geometric structure, which is essential for precise and contact-rich manipulation. To address this limitation, we introduce UniLACT, a transformer-based VLA model that incorporates geometric structure through depth-aware latent pretraining, enabling downstream policies to inherit stronger spatial priors. To facilitate this process, we propose UniLARN, a unified latent action learning framework based on inverse and forward dynamics objectives that learns a shared embedding space for RGB and depth while explicitly modeling their cross-modal interactions. This formulation produces modality-specific and unified latent action representations that serve as pseudo-labels for the depth-aware pretraining of UniLACT. Extensive experiments in both simulation and real-world settings demonstrate the effectiveness of depth-aware unified latent action representations. UniLACT consistently outperforms RGB-based latent action baselines under in-domain and out-of-domain pretraining regimes, as well as on both seen and unseen manipulation tasks.
>
---
#### [new 121] Strategy-Supervised Autonomous Laparoscopic Camera Control via Event-Driven Graph Mining
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主腹腔镜摄像控制任务，解决手术中视角稳定与安全问题。通过事件驱动图挖掘提取策略，结合视觉语言模型实现智能控制。**

- **链接: [https://arxiv.org/pdf/2602.20500v1](https://arxiv.org/pdf/2602.20500v1)**

> **作者:** Keyu Zhou; Peisen Xu; Yahao Wu; Jiming Chen; Gaofeng Li; Shunlei Li
>
> **备注:** Submitted to IEEE Transactions on Robotics (T-RO). 19 pages, 9 figures
>
> **摘要:** Autonomous laparoscopic camera control must maintain a stable and safe surgical view under rapid tool-tissue interactions while remaining interpretable to surgeons. We present a strategy-grounded framework that couples high-level vision-language inference with low-level closed-loop control. Offline, raw surgical videos are parsed into camera-relevant temporal events (e.g., interaction, working-distance deviation, and view-quality degradation) and structured as attributed event graphs. Mining these graphs yields a compact set of reusable camera-handling strategy primitives, which provide structured supervision for learning. Online, a fine-tuned Vision-Language Model (VLM) processes the live laparoscopic view to predict the dominant strategy and discrete image-based motion commands, executed by an IBVS-RCM controller under strict safety constraints; optional speech input enables intuitive human-in-the-loop conditioning. On a surgeon-annotated dataset, event parsing achieves reliable temporal localization (F1-score 0.86), and the mined strategies show strong semantic alignment with expert interpretation (cluster purity 0.81). Extensive ex vivo experiments on silicone phantoms and porcine tissues demonstrate that the proposed system outperforms junior surgeons in standardized camera-handling evaluations, reducing field-of-view centering error by 35.26% and image shaking by 62.33%, while preserving smooth motion and stable working-distance regulation.
>
---
#### [new 122] Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在生成动作时效率低和鲁棒性差的问题。通过引入双记忆机制提升性能与速度。**

- **链接: [https://arxiv.org/pdf/2602.20200v1](https://arxiv.org/pdf/2602.20200v1)**

> **作者:** Zaijing Li; Bing Hu; Rui Shao; Gongwei Chen; Dongmei Jiang; Pengwei Xie; Jianye Hao; Liqiang Nie
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Hierarchical Vision-Language-Action (VLA) models have rapidly become a dominant paradigm for robotic manipulation. It typically comprising a Vision-Language backbone for perception and understanding, together with a generative policy for action generation. However, its performance is increasingly bottlenecked by the action generation proceess. (i) Low inference efficiency. A pronounced distributional gap between isotropic noise priors and target action distributions, which increases denoising steps and the incidence of infeasible samples. (ii) Poor robustness. Existing policies condition solely on the current observation, neglecting the constraint of history sequence and thus lacking awareness of task progress and temporal consistency. To address these issues, we introduce OptimusVLA, a dual-memory VLA framework with Global Prior Memory (GPM) and Local Consistency Memory (LCM). GPM replaces Gaussian noise with task-level priors retrieved from semantically similar trajectories, thereby shortening the generative path and reducing the umber of function evaluations (NFE). LCM dynamically models executed action sequence to infer task progress and injects a learned consistency constraint that enforces temporal coherence and smoothness of trajectory. Across three simulation benchmarks, OptimusVLA consistently outperforms strong baselines: it achieves 98.6% average success rate on LIBERO, improves over pi_0 by 13.5% on CALVIN, and attains 38% average success rate on RoboTwin 2.0 Hard. In Real-World evaluation, OptimusVLA ranks best on Generalization and Long-horizon suites, surpassing pi_0 by 42.9% and 52.4%, respectively, while delivering 2.9x inference speedup.
>
---
#### [new 123] Momentum Guidance: Plug-and-Play Guidance for Flow Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出Momentum Guidance（MG），用于流模型的生成任务，解决条件生成中样本质量与多样性不足的问题。MG通过利用ODE轨迹提升生成效果，无需额外计算成本。**

- **链接: [https://arxiv.org/pdf/2602.20360v1](https://arxiv.org/pdf/2602.20360v1)**

> **作者:** Runlong Liao; Jian Yu; Baiyu Su; Chi Zhang; Lizhang Chen; Qiang Liu
>
> **摘要:** Flow-based generative models have become a strong framework for high-quality generative modeling, yet pretrained models are rarely used in their vanilla conditional form: conditional samples without guidance often appear diffuse and lack fine-grained detail due to the smoothing effects of neural networks. Existing guidance techniques such as classifier-free guidance (CFG) improve fidelity but double the inference cost and typically reduce sample diversity. We introduce Momentum Guidance (MG), a new dimension of guidance that leverages the ODE trajectory itself. MG extrapolates the current velocity using an exponential moving average of past velocities and preserves the standard one-evaluation-per-step cost. It matches the effect of standard guidance without extra computation and can further improve quality when combined with CFG. Experiments demonstrate MG's effectiveness across benchmarks. Specifically, on ImageNet-256, MG achieves average improvements in FID of 36.68% without CFG and 25.52% with CFG across various sampling settings, attaining an FID of 1.597 at 64 sampling steps. Evaluations on large flow-based models like Stable Diffusion 3 and FLUX.1-dev further confirm consistent quality enhancements across standard metrics.
>
---
#### [new 124] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出NoRD模型，解决VLA模型数据和标注成本高的问题。通过减少数据量和无需推理标注，提升自动驾驶效率。**

- **链接: [https://arxiv.org/pdf/2602.21172v1](https://arxiv.org/pdf/2602.21172v1)**

> **作者:** Ishaan Rawal; Shubh Gupta; Yihan Hu; Wei Zhan
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Vision-Language-Action (VLA) models are advancing autonomous driving by replacing modular pipelines with unified end-to-end architectures. However, current VLAs face two expensive requirements: (1) massive dataset collection, and (2) dense reasoning annotations. In this work, we address both challenges with \modelname (\textbf{No} \textbf{R}easoning for \textbf{D}riving). Compared to existing VLAs, \modelname achieves competitive performance while being fine-tuned on $<$60\% of the data and no reasoning annotations, resulting in 3$\times$ fewer tokens. We identify that standard Group Relative Policy Optimization (GRPO) fails to yield significant improvements when applied to policies trained on such small, reasoning-free datasets. We show that this limitation stems from difficulty bias, which disproportionately penalizes reward signals from scenarios that produce high-variance rollouts within GRPO. \modelname overcomes this by incorporating Dr.~GRPO, a recent algorithm designed to mitigate difficulty bias in LLMs. As a result, \modelname achieves competitive performance on Waymo and NAVSIM with a fraction of the training data and no reasoning overhead, enabling more efficient autonomous systems.
>
---
## 更新

#### [replaced 001] Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.06878v2](https://arxiv.org/pdf/2508.06878v2)**

> **作者:** Maoxun Yuan; Duanni Meng; Ziteng Xi; Tianyi Zhao; Shiji Zhao; Yimian Dai; Xingxing Wei
>
> **摘要:** Infrared small target detection and segmentation (IRSTDS) is a critical yet challenging task in defense and civilian applications, owing to the dim, shapeless appearance of targets and severe background clutter. Recent CNN-based methods have achieved promising target perception results, but they only focus on enhancing feature representation to offset the impact of noise, which results in the increased false alarm problem. In this paper, through analyzing the problem from the frequency domain, we pioneer in improving performance from noise suppression perspective and propose a novel noise-suppression feature pyramid network (NS-FPN), which integrates a low-frequency guided feature purification (LFP) module and a spiral-aware feature sampling (SFS) module into the original FPN structure. The LFP module suppresses the noise features by purifying high-frequency components to achieve feature enhancement devoid of noise interference, while the SFS module further adopts spiral sampling to fuse target-relevant features in feature fusion process. Our NS-FPN is designed to be lightweight yet effective and can be easily plugged into existing IRSTDS frameworks. Extensive experiments on the IRSTD-1k and NUAA-SIRST datasets demonstrate that our method significantly reduces false alarms and achieves superior performance on IRSTDS task.
>
---
#### [replaced 002] Erased, But Not Forgotten: Erased Rectified Flow Transformers Still Remain Unsafe Under Concept Attack
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00635v3](https://arxiv.org/pdf/2510.00635v3)**

> **作者:** Nanxiang Jiang; Zhaoxin Fan; Enhan Kang; Daiheng Gao; Yun Zhou; Yanxia Chang; Zheng Zhu; Yeying Jin; Wenjun Wu
>
> **摘要:** Recent advances in text-to-image (T2I) diffusion models have enabled impressive generative capabilities, but they also raise significant safety concerns due to the potential to produce harmful or undesirable content. While concept erasure has been explored as a mitigation strategy, most existing approaches and corresponding attack evaluations are tailored to Stable Diffusion (SD) and exhibit limited effectiveness when transferred to next-generation rectified flow transformers such as Flux. In this work, we present ReFlux, the first concept attack method specifically designed to assess the robustness of concept erasure in the latest rectified flow-based T2I framework. Our approach is motivated by the observation that existing concept erasure techniques, when applied to Flux, fundamentally rely on a phenomenon known as attention localization. Building on this insight, we propose a simple yet effective attack strategy that specifically targets this property. At its core, a reverse-attention optimization strategy is introduced to effectively reactivate suppressed signals while stabilizing attention. This is further reinforced by a velocity-guided dynamic that enhances the robustness of concept reactivation by steering the flow matching process, and a consistency-preserving objective that maintains the global layout and preserves unrelated content. Extensive experiments consistently demonstrate the effectiveness and efficiency of the proposed attack method, establishing a reliable benchmark for evaluating the robustness of concept erasure strategies in rectified flow transformers.
>
---
#### [replaced 003] Learning Unified Representations from Heterogeneous Data for Robust Heart Rate Modeling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.21785v3](https://arxiv.org/pdf/2508.21785v3)**

> **作者:** Zhengdong Huang; Zicheng Xie; Wentao Tian; Jingyu Liu; Lunhong Dong; Peng Yang
>
> **摘要:** Heart rate prediction is vital for personalized health monitoring and fitness, while it frequently faces a critical challenge in real-world deployment: data heterogeneity. We classify it in two key dimensions: source heterogeneity from fragmented device markets with varying feature sets, and user heterogeneity reflecting distinct physiological patterns across individuals and activities. Existing methods either discard device-specific information, or fail to model user-specific differences, limiting their real-world performance. To address this, we propose a framework that learns latent representations agnostic to both heterogeneity,enabling downstream predictors to work consistently under heterogeneous data patterns. Specifically, we introduce a random feature dropout strategy to handle source heterogeneity, making the model robust to various feature sets. To manage user heterogeneity, we employ a history-aware attention module to capture long-term physiological traits and use a contrastive learning objective to build a discriminative representation space. To reflect the heterogeneous nature of real-world data, we created a new benchmark dataset, PARROTAO. Evaluations on both PARROTAO and the public FitRec dataset show that our model significantly outperforms existing baselines by 17.5% and 10.4% in terms of test MSE, respectively. Furthermore, analysis of the learned representations demonstrates their strong discriminative power,and two downstream application tasks confirm the practical value of our model.
>
---
#### [replaced 004] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; eess.IV**

- **链接: [https://arxiv.org/pdf/2602.08550v3](https://arxiv.org/pdf/2602.08550v3)**

> **作者:** Shih-Fang Chen; Jun-Cheng Chen; I-Hong Jhuo; Yen-Yu Lin
>
> **备注:** ICLR 2026
>
> **摘要:** Human perception for effective object tracking in 2D video streams arises from the implicit use of prior 3D knowledge and semantic reasoning. In contrast, most generic object tracking (GOT) methods primarily rely on 2D features of the target and its surroundings, while neglecting 3D geometric cues, making them susceptible to partial occlusion, distractors, and variations in geometry and appearance. To address this limitation, we introduce GOT-Edit, an online cross-modality model editing approach that integrates geometry-aware cues into a generic object tracker from a 2D video stream. Our approach leverages features from a pre-trained Visual Geometry Grounded Transformer to infer geometric cues from only a few 2D images. To address the challenge of seamlessly combining geometry and semantics, GOT-Edit performs online model editing. By leveraging null-space constraints during model updates, it incorporates geometric information while preserving semantic discrimination, yielding consistently better performance across diverse scenarios. Extensive experiments on multiple GOT benchmarks demonstrate that GOT-Edit achieves superior robustness and accuracy, particularly under occlusion and clutter, establishing a new paradigm for combining 2D semantics with 3D geometric reasoning for generic object tracking. The project page is available at https://chenshihfang.github.io/GOT-EDIT.
>
---
#### [replaced 005] PaCo-FR: Patch-Pixel Aligned End-to-End Codebook Learning for Facial Representation Pre-training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09691v2](https://arxiv.org/pdf/2508.09691v2)**

> **作者:** Yin Xie; Zhichao Chen; Zeyu Xiao; Yongle Zhao; Xiang An; Kaicheng Yang; Zimin Ran; Jia Guo; Ziyong Feng; Jiankang Deng
>
> **摘要:** Facial representation pre-training is crucial for tasks like facial recognition, expression analysis, and virtual reality. However, existing methods face three key challenges: (1) failing to capture distinct facial features and fine-grained semantics, (2) ignoring the spatial structure inherent to facial anatomy, and (3) inefficiently utilizing limited labeled data. To overcome these, we introduce PaCo-FR, an unsupervised framework that combines masked image modeling with patch-pixel alignment. Our approach integrates three innovative components: (1) a structured masking strategy that preserves spatial coherence by aligning with semantically meaningful facial regions, (2) a novel patch-based codebook that enhances feature discrimination with multiple candidate tokens, and (3) spatial consistency constraints that preserve geometric relationships between facial components. PaCo-FR achieves state-of-the-art performance across several facial analysis tasks with just 2 million unlabeled images for pre-training. Our method demonstrates significant improvements, particularly in scenarios with varying poses, occlusions, and lighting conditions. We believe this work advances facial representation learning and offers a scalable, efficient solution that reduces reliance on expensive annotated datasets, driving more effective facial analysis systems.
>
---
#### [replaced 006] The devil is in the details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20340v2](https://arxiv.org/pdf/2512.20340v2)**

> **作者:** Qingdong He; Xueqin Chen; Yanjie Pan; Peng Tang; Pengcheng Xu; Zhenye Gan; Chengjie Wang; Xiaobin Hu; Jiangning Zhang; Yabiao Wang
>
> **备注:** Accepted by CVPR 2026 (Main Conference)
>
> **摘要:** Although diffusion transformer (DiT)-based video virtual try-on (VVT) has made significant progress in synthesizing realistic videos, existing methods still struggle to capture fine-grained garment dynamics and preserve background integrity across video frames. They also incur high computational costs due to additional interaction modules introduced into DiTs, while the limited scale and quality of existing public datasets also restrict model generalization and effective training. To address these challenges, we propose a novel framework, KeyTailor, along with a large-scale, high-definition dataset, ViT-HD. The core idea of KeyTailor is a keyframe-driven details injection strategy, motivated by the fact that keyframes inherently contain both foreground dynamics and background consistency. Specifically, KeyTailor adopts an instruction-guided keyframe sampling strategy to filter informative frames from the input video. Subsequently,two tailored keyframe-driven modules, the garment details enhancement module and the collaborative background optimization module, are employed to distill garment dynamics into garment-related latents and to optimize the integrity of background latents, both guided by keyframes.These enriched details are then injected into standard DiT blocks together with pose, mask, and noise latents, enabling efficient and realistic try-on video synthesis. This design ensures consistency without explicitly modifying the DiT architecture, while simultaneously avoiding additional complexity. In addition, our dataset ViT-HD comprises 15, 070 high-quality video samples at a resolution of 810*1080, covering diverse garments. Extensive experiments demonstrate that KeyTailor outperforms state-of-the-art baselines in terms of garment fidelity and background integrity across both dynamic and static scenarios.
>
---
#### [replaced 007] CuriGS: Curriculum-Guided Gaussian Splatting for Sparse View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16030v2](https://arxiv.org/pdf/2511.16030v2)**

> **作者:** Zijian Wu; Mingfeng Jiang; Zidian Lin; Ying Song; Hanjie Ma; Qun Wu; Dongping Zhang; Guiyang Pu
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as an efficient, high-fidelity representation for real-time scene reconstruction and rendering. However, extending 3DGS to sparse-view settings remains challenging because of supervision scarcity and overfitting caused by limited viewpoint coverage. In this paper, we present CuriGS, a curriculum-guided framework for sparse-view 3D reconstruction using 3DGS. CuriGS addresses the core challenge of sparse-view synthesis by introducing student views: pseudo-views sampled around ground-truth poses (teacher). For each teacher, we generate multiple groups of student views with different perturbation levels. During training, we follow a curriculum schedule that gradually unlocks higher perturbation level, randomly sampling candidate students from the active level to assist training. Each sampled student is regularized via depth-correlation and co-regularization, and evaluated using a multi-signal metric that combines SSIM, LPIPS, and an image-quality measure. For every teacher and perturbation level, we periodically retain the best-performing students and promote those that satisfy a predefined quality threshold to the training set, resulting in a stable augmentation of sparse training views. Experimental results show that CuriGS outperforms state-of-the-art baselines in both rendering fidelity and geometric consistency across various synthetic and real sparse-view scenes. Project page: https://zijian1026.github.io/CuriGS/
>
---
#### [replaced 008] Ecological mapping with geospatial foundation models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.10720v2](https://arxiv.org/pdf/2602.10720v2)**

> **作者:** Craig Mahlasi; Gciniwe S. Baloyi; Zaheed Gaffoor; Levente Klein; Anne Jones; Etienne Vos; Michal Muszynski; Geoffrey Dawson; Campbell Watson
>
> **备注:** Revised abstract
>
> **摘要:** The value of Earth observation foundation models for high-impact ecological applications remains insufficiently characterized. This study is one of the first to systematically evaluate the performance, limitations and practical considerations across three common ecological use cases: forest functional trait estimation, land use and land cover mapping and peatland detection. We fine-tune two pretrained models (Prithvi-EO-2.0 and TerraMind) and benchmark them against a ResNet-101 baseline using datasets collected from open sources. Across all tasks, Prithvi-EO-2.0 and TerraMind consistently outperform the ResNet baseline, demonstrating improved generalization and transfer across ecological domains. TerraMind marginally exceeds Prithvi-EO-2.0 in unimodal settings and shows substantial gains when additional modalities are incorporated. However, performance is sensitive to divergence between downstream inputs and pretraining modalities, underscoring the need for careful dataset alignment. Results also indicate that higher-resolution inputs and more accurate pixel-level labels remain critical for capturing fine-scale ecological dynamics.
>
---
#### [replaced 009] Learning Humanoid End-Effector Control for Open-Vocabulary Visual Loco-Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人形机器人操作任务，旨在解决复杂环境中物体操控的泛化问题。通过结合视觉模型与控制策略，提升末端执行器的跟踪精度，实现多样化物体的可靠操作。**

- **链接: [https://arxiv.org/pdf/2602.16705v2](https://arxiv.org/pdf/2602.16705v2)**

> **作者:** Runpei Dong; Ziyan Li; Xialin He; Saurabh Gupta
>
> **备注:** Project page: https://hero-humanoid.github.io/
>
> **摘要:** Visual loco-manipulation of arbitrary objects in the wild with humanoid robots requires accurate end-effector (EE) control and a generalizable understanding of the scene via visual inputs (e.g., RGB-D images). Existing approaches are based on real-world imitation learning and exhibit limited generalization due to the difficulty in collecting large-scale training datasets. This paper presents a new paradigm, HERO, for object loco-manipulation with humanoid robots that combines the strong generalization and open-vocabulary understanding of large vision models with strong control performance from simulated training. We achieve this by designing an accurate residual-aware EE tracking policy. This EE tracking policy combines classical robotics with machine learning. It uses a) inverse kinematics to convert residual end-effector targets into reference trajectories, b) a learned neural forward model for accurate forward kinematics, c) goal adjustment, and d) replanning. Together, these innovations help us cut down the end-effector tracking error by 3.2x. We use this accurate end-effector tracker to build a modular system for loco-manipulation, where we use open-vocabulary large vision models for strong visual generalization. Our system is able to operate in diverse real-world environments, from offices to coffee shops, where the robot is able to reliably manipulate various everyday objects (e.g., mugs, apples, toys) on surfaces ranging from 43cm to 92cm in height. Systematic modular and end-to-end tests in simulation and the real world demonstrate the effectiveness of our proposed design. We believe the advances in this paper can open up new ways of training humanoid robots to interact with daily objects.
>
---
#### [replaced 010] Enhancing Out-of-Distribution Detection with Extended Logit Normalization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.11434v2](https://arxiv.org/pdf/2504.11434v2)**

> **作者:** Yifan Ding; Xixi Liu; Jonas Unger; Gabriel Eilertsen
>
> **备注:** CVPR 2026
>
> **摘要:** \noindent Out-of-distribution (OOD) detection is essential for the safe deployment of machine learning models. Extensive work has focused on devising various scoring functions for detecting OOD samples, while only a few studies focus on training neural networks using certain model calibration objectives, which often lead to a compromise in predictive accuracy and support only limited choices of scoring functions. In this work, we first identify the feature collapse phenomena in Logit Normalization (LogitNorm), then propose a novel hyperparameter-free formulation that significantly benefits a wide range of post-hoc detection methods. To be specific, we devise a feature distance-awareness loss term in addition to LogitNorm, termed $\textbf{ELogitNorm}$, which enables improved OOD detection and in-distribution (ID) confidence calibration. Extensive experiments across standard benchmarks demonstrate that our approach outperforms state-of-the-art training-time methods in OOD detection while maintaining strong ID classification accuracy. Our code is available on: https://github.com/limchaos/ElogitNorm.
>
---
#### [replaced 011] SpecAware: A Spectral-Content Aware Foundation Model for Unifying Multi-Sensor Learning in Hyperspectral Remote Sensing Mapping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.27219v2](https://arxiv.org/pdf/2510.27219v2)**

> **作者:** Renjie Ji; Xue Wang; Chao Niu; Wen Zhang; Yong Mei; Kun Tan
>
> **摘要:** Hyperspectral imaging (HSI) is a critical technique for fine-grained land-use and land-cover (LULC) mapping. However, the inherent heterogeneity of HSI data, particularly the variation in spectral channels across sensors, has long constrained the development of model generalization via transfer learning or joint training. Existing HSI foundation models show promise for different downstream tasks, but typically underutilize the critical guiding role of sensor meta-attributes and image semantic features, resulting in limited adaptability to cross-sensor joint learning. To address these issues, we propose SpecAware, which is a novel hyperspectral spectral-content aware foundation model for unifying multi-sensor learning for HSI mapping. To support this work, we constructed the Hyper-400K dataset, which is a new large-scale pre-training dataset with over 400\,k high-quality patches from diverse airborne AVIRIS sensors that cover two data processing levels (L1 and L2). The core of SpecAware is a hypernetwork-driven unified image embedding process for HSI data. Firstly, we designed a meta-content aware module to generate a unique conditional input for each HSI sample, tailored to each spectral band by fusing the sensor meta-attributes and its own image content. Secondly, we designed the HyperEmbedding module, where a sample-conditioned hypernetwork dynamically generates a pair of matrix factors for channel-wise encoding. This process implements two-step matrix factorization, consisting of adaptive spatial pattern extraction and latent semantic feature projection, yielding a unified hyperspectral token representation. Thus, SpecAware learns to capture and interpret spatial-spectral features across diverse scenes and sensors, enabling adaptive processing of variable spectral channels within a unified multi-sensor joint pre-training framework.
>
---
#### [replaced 012] Learning Hierarchical Sparse Transform Coding for 3DGS Compression
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2505.22908v4](https://arxiv.org/pdf/2505.22908v4)**

> **作者:** Hao Xu; Xiaolin Wu; Xi Zhang
>
> **备注:** Our code will be released at \href{https://github.com/hxu160/SHTC_for_3DGS_compression}{here}
>
> **摘要:** Current 3DGS compression methods largely forego the neural analysis-synthesis transform, which is a crucial component in learned signal compression systems. As a result, redundancy removal is left solely to the entropy coder, overburdening the entropy coding module and reducing rate-distortion (R-D) performance. To fix this critical omission, we propose a training-time transform coding (TTC) method that adds the analysis-synthesis transform and optimizes it jointly with the 3DGS representation and entropy model. Concretely, we adopt a hierarchical design: a channel-wise KLT for decorrelation and energy compaction, followed by a sparsity-aware neural transform that reconstructs the KLT residuals with minimal parameter and computational overhead. Experiments show that our method delivers strong R-D performance with fast decoding, offering a favorable BD-rate-decoding-time trade-off over SOTA 3DGS compressors.
>
---
#### [replaced 013] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文提出HoloLLM，解决多模态感知与语言理解问题，整合LiDAR、红外等非视觉数据，提升智能体在复杂环境中的行为识别能力。**

- **链接: [https://arxiv.org/pdf/2505.17645v2](https://arxiv.org/pdf/2505.17645v2)**

> **作者:** Chuhao Zhou; Jianfei Yang
>
> **备注:** Camera-ready version. Accepted at NeurIPS 2025
>
> **摘要:** Embodied agents operating in smart homes must understand human behavior through diverse sensory inputs and communicate via natural language. While Vision-Language Models (VLMs) have enabled impressive language-grounded perception, their reliance on visual data limits robustness in real-world scenarios with occlusions, poor lighting, or privacy constraints. In this paper, we introduce HoloLLM, a Multimodal Large Language Model (MLLM) that integrates uncommon but powerful sensing modalities, such as LiDAR, infrared, mmWave radar, and WiFi, to enable seamless human perception and reasoning across heterogeneous environments. We address two key challenges: (1) the scarcity of aligned modality-text data for rare sensors, and (2) the heterogeneity of their physical signal representations. To overcome these, we design a Universal Modality-Injection Projector (UMIP) that enhances pre-aligned modality embeddings with fine-grained, text-aligned features from tailored encoders via coarse-to-fine cross-attention without introducing significant alignment overhead. We further introduce a human-VLM collaborative data curation pipeline to generate paired textual annotations for sensing datasets. Extensive experiments on two newly constructed benchmarks show that HoloLLM significantly outperforms existing MLLMs, improving language-grounded human sensing accuracy by up to 30%. This work establishes a new foundation for real-world, language-informed multisensory embodied intelligence.
>
---
#### [replaced 014] Knee or ROC
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2401.07390v2](https://arxiv.org/pdf/2401.07390v2)**

> **作者:** Veronica Wendt; Jacob Steiner; Byunggu Yu; Caleb Kelly; Justin Kim
>
> **备注:** 9 pages
>
> **摘要:** Self-attention transformers have demonstrated accuracy for image classification with smaller data sets. However, a limitation is that tests to-date are based upon single class image detection with known representation of image populations. For instances where the input image classes may be greater than one and test sets that lack full information on representation of image populations, accuracy calculations must adapt. The Receiver Operating Characteristic (ROC) accuracy threshold can address the instances of multiclass input images. However, this approach is unsuitable in instances where image population representation is unknown. We then consider calculating accuracy using the knee method to determine threshold values on an ad-hoc basis. Results of ROC curve and knee thresholds for a multi-class data set, created from CIFAR-10 images, are discussed for multiclass image detection.
>
---
#### [replaced 015] Earth Embeddings as Products: Taxonomy, Ecosystem, and Standardized Access
- **分类: cs.SE; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.13134v2](https://arxiv.org/pdf/2601.13134v2)**

> **作者:** Heng Fang; Adam J. Stewart; Isaac Corley; Xiao Xiang Zhu; Hossein Azizpour
>
> **摘要:** Geospatial Foundation Models (GFMs) provide powerful representations, but high compute costs hinder their widespread use. Pre-computed embedding data products offer a practical "frozen" alternative, yet they currently exist in a fragmented ecosystem of incompatible formats and resolutions. This lack of standardization creates an engineering bottleneck that prevents meaningful model comparison and reproducibility. We formalize this landscape through a three-layer taxonomy: Data, Tools, and Value. We survey existing products to identify interoperability barriers. To bridge this gap, we extend TorchGeo with a unified API that standardizes the loading and querying of diverse embedding products. By treating embeddings as first-class geospatial datasets, we decouple downstream analysis from model-specific engineering, providing a roadmap for more transparent and accessible Earth observation workflows.
>
---
#### [replaced 016] UniPart: Part-Level 3D Generation with Unified 3D Geom-Seg Latents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.09435v2](https://arxiv.org/pdf/2512.09435v2)**

> **作者:** Xufan He; Yushuang Wu; Xiaoyang Guo; Chongjie Ye; Jiaqing Zhou; Tianlei Hu; Xiaoguang Han; Dong Du
>
> **备注:** Project page: https://xfanhe.github.io/projects/unipart/
>
> **摘要:** Part-level 3D generation is essential for applications requiring decomposable and structured 3D synthesis. However, existing methods either rely on implicit part segmentation with limited granularity control or depend on strong external segmenters trained on large annotated datasets. In this work, we observe that part awareness emerges naturally during whole-object geometry learning and propose Geom-Seg VecSet, a unified geometry-segmentation latent representation that jointly encodes object geometry and part-level structure. Building on this representation, we introduce UniPart, a two-stage latent diffusion framework for image-guided part-level 3D generation. The first stage performs joint geometry generation and latent part segmentation, while the second stage conditions part-level diffusion on both whole-object and part-specific latents. A dual-space generation scheme further enhances geometric fidelity by predicting part latents in both global and canonical spaces. Extensive experiments demonstrate that UniPart achieves superior segmentation controllability and part-level geometric quality compared with existing approaches. Project page: https://xfanhe.github.io/projects/unipart/
>
---
#### [replaced 017] Keep it SymPL: Symbolic Projective Layout for Allocentric Spatial Reasoning in Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19117v2](https://arxiv.org/pdf/2602.19117v2)**

> **作者:** Jaeyun Jang; Seunghui Shin; Taeho Park; Hyoseok Hwang
>
> **摘要:** Perspective-aware spatial reasoning involves understanding spatial relationships from specific viewpoints-either egocentric (observer-centered) or allocentric (object-centered). While vision-language models (VLMs) perform well in egocentric settings, their performance deteriorates when reasoning from allocentric viewpoints, where spatial relations must be inferred from the perspective of objects within the scene. In this study, we address this underexplored challenge by introducing Symbolic Projective Layout (SymPL), a framework that reformulates allocentric reasoning into symbolic-layout forms that VLMs inherently handle well. By leveraging four key factors-projection, abstraction, bipartition, and localization-SymPL converts allocentric questions into structured symbolic-layout representations. Extensive experiments demonstrate that this reformulation substantially improves performance in both allocentric and egocentric tasks, enhances robustness under visual illusions and multi-view scenarios, and that each component contributes critically to these gains. These results show that SymPL provides an effective and principled approach for addressing complex perspective-aware spatial reasoning.
>
---
#### [replaced 018] UniGenBench++: A Unified Semantic Evaluation Benchmark for Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18701v2](https://arxiv.org/pdf/2510.18701v2)**

> **作者:** Yibin Wang; Zhimin Li; Yuhang Zang; Jiazi Bu; Yujie Zhou; Yi Xin; Junjun He; Chunyu Wang; Qinglin Lu; Cheng Jin; Jiaqi Wang
>
> **备注:** Project page: codegoat24.github.io/UniGenBench/
>
> **摘要:** Recent progress in text-to-image (T2I) generation underscores the importance of reliable benchmarks in evaluating how accurately generated images reflect the semantics of their textual prompt. However, (1) existing benchmarks lack the diversity of prompt scenarios and multilingual support, both essential for real-world applicability; (2) they offer only coarse evaluations across primary dimensions, covering a narrow range of sub-dimensions, and fall short in fine-grained sub-dimension assessment. To address these limitations, we introduce UniGenBench++, a unified semantic assessment benchmark for T2I generation. Specifically, it comprises 600 prompts organized hierarchically to ensure both coverage and efficiency: (1) spans across diverse real-world scenarios, i.e., 5 main prompt themes and 20 subthemes; (2) comprehensively probes T2I models' semantic consistency over 10 primary and 27 sub evaluation criteria, with each prompt assessing multiple testpoints. To rigorously assess model robustness to variations in language and prompt length, we provide both English and Chinese versions of each prompt in short and long forms. Leveraging the general world knowledge and fine-grained image understanding capabilities of a closed-source Multi-modal Large Language Model (MLLM), i.e., Gemini-2.5-Pro, an effective pipeline is developed for reliable benchmark construction and streamlined model assessment. Moreover, to further facilitate community use, we train a robust evaluation model that enables offline assessment of T2I model outputs. Through comprehensive benchmarking of both open- and closed-sourced T2I models, we systematically reveal their strengths and weaknesses across various aspects.
>
---
#### [replaced 019] WonderVerse: Extendable 3D Scene Generation with Video Generative Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.09160v4](https://arxiv.org/pdf/2503.09160v4)**

> **作者:** Hao Feng; Zhi Zuo; Jia-Hui Pan; Ka-Hei Hui; Qi Dou; Jingyu Hu; Zhengzhe Liu
>
> **备注:** Accepted at CVM 2026
>
> **摘要:** We introduce \textit{WonderVerse}, a simple but effective framework for generating extendable 3D scenes. Unlike existing methods that rely on iterative depth estimation and image inpainting, often leading to geometric distortions and inconsistencies, WonderVerse leverages the powerful world-level priors embedded within video generative foundation models to create highly immersive and geometrically coherent 3D environments. Furthermore, we propose a new technique for controllable 3D scene extension to substantially increase the scale of the generated environments. Besides, we introduce a novel abnormal sequence detection module that utilizes camera trajectory to address geometric inconsistency in the generated videos. Finally, WonderVerse is compatible with various 3D reconstruction methods, allowing both efficient and high-quality generation. Extensive experiments on 3D scene generation demonstrate that our WonderVerse, with an elegant and simple pipeline, delivers extendable and highly-realistic 3D scenes, markedly outperforming existing works that rely on more complex architectures.
>
---
#### [replaced 020] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13394v3](https://arxiv.org/pdf/2510.13394v3)**

> **作者:** Xinmiao Huang; Qisong He; Zhenglin Huang; Boxuan Wang; Zhuoyun Li; Guangliang Cheng; Yi Dong; Xiaowei Huang
>
> **备注:** ICLR 2026 Accepted Project Page: https://shinmohuang.github.io/spatialdise_page/
>
> **摘要:** Spatial reasoning ability is crucial for Vision Language Models (VLMs) to support real-world applications in diverse domains including robotics, augmented reality, and autonomous navigation. Unfortunately, existing benchmarks are inadequate in assessing spatial reasoning ability, especially the \emph{intrinsic-dynamic} spatial reasoning which is a fundamental aspect of human spatial cognition. In this paper, we propose a unified benchmark, \textbf{Spatial-DISE}, based on a cognitively grounded taxonomy that categorizes tasks into four fundamental quadrants: \textbf{I}ntrinsic-\textbf{S}tatic, Intrinsic-\textbf{D}ynamic, \textbf{E}xtrinsic-Static, and Extrinsic-Dynamic spatial reasoning. Moreover, to address the issue of data scarcity, we develop a scalable and automated pipeline to generate diverse and verifiable spatial reasoning questions, resulting in a new \textbf{Spatial-DISE} dataset that includes Spatial-DISE Bench (559 evaluation VQA pairs) and Spatial-DISE-12K (12K+ training VQA pairs). Our comprehensive evaluation across 28 state-of-the-art VLMs reveals that, current VLMs have a large and consistent gap to human competence, especially on multi-step multi-view spatial reasoning. Spatial-DISE offers a robust framework, valuable dataset, and clear direction for future research toward human-like spatial intelligence. Benchmark, dataset, and code will be publicly released.
>
---
#### [replaced 021] Defending Unauthorized Model Merging via Dual-Stage Weight Protection
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2511.11851v2](https://arxiv.org/pdf/2511.11851v2)**

> **作者:** Wei-Jia Chen; Min-Yen Tsai; Cheng-Yi Lee; Chia-Mu Yu
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** The rapid proliferation of pretrained models and open repositories has made model merging a convenient yet risky practice, allowing free-riders to combine fine-tuned models into a new multi-capability model without authorization. Such unauthorized model merging not only violates intellectual property rights but also undermines model ownership and accountability. To address this issue, we present MergeGuard, a proactive dual-stage weight protection framework that disrupts merging compatibility while maintaining task fidelity. In the first stage, we redistribute task-relevant information across layers via L2-regularized optimization, ensuring that important gradients are evenly dispersed. In the second stage, we inject structured perturbations to misalign task subspaces, breaking curvature compatibility in the loss landscape. Together, these stages reshape the model's parameter geometry such that merged models collapse into destructive interference while the protected model remains fully functional. Extensive experiments on both vision (ViT-L-14) and language (Llama2, Gemma2, Mistral) models demonstrate that MergeGuard reduces merged model accuracy by up to 90% with less than 1.5% performance loss on the protected model.
>
---
#### [replaced 022] Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19863v2](https://arxiv.org/pdf/2602.19863v2)**

> **作者:** Filip Wolf; Blaž Rolih; Luka Čehovin Zajc
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Foundation models are transforming Earth Observation (EO), yet the diversity of EO sensors and modalities makes a single universal model unrealistic. Multiple specialized EO foundation models (EOFMs) will likely coexist, making efficient knowledge transfer across modalities essential. Most existing EO pretraining relies on masked image modeling, which emphasizes local reconstruction but provides limited control over global semantic structure. To address this, we propose a dual-teacher contrastive distillation framework for multispectral imagery that aligns the student's pretraining objective with the contrastive self-distillation paradigm of modern optical vision foundation models (VFMs). Our approach combines a multispectral teacher with an optical VFM teacher, enabling coherent cross-modal representation learning. Experiments across diverse optical and multispectral benchmarks show that our model adapts to multispectral data without compromising performance on optical-only inputs, achieving state-of-the-art results in both settings, with an average improvement of 3.64 percentage points in semantic segmentation, 1.2 in change detection, and 1.31 in classification tasks. This demonstrates that contrastive distillation provides a principled and efficient approach to scalable representation learning across heterogeneous EO data sources. Project page: \textcolor{magenta}{https://wolfilip.github.io/DEO/}.
>
---
#### [replaced 023] RegTrack: Simplicity Beneath Complexity in Robust Multi-Modal 3D Multi-Object Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.00618v3](https://arxiv.org/pdf/2409.00618v3)**

> **作者:** Lipeng Gu; Xuefeng Yan; Song Wang; Mingqiang Wei
>
> **摘要:** Existing 3D multi-object tracking (MOT) methods often sacrifice efficiency and generalizability for robustness, largely relying on complex association metrics derived from multi-modal architectures and class-specific motion priors. Challenging the rooted belief that greater complexity necessarily yields greater robustness, we propose a robust, efficient, and generalizable method for multi-modal 3D MOT, dubbed RegTrack. Inspired by Yang-Mills gauge theory, RegTrack is built upon a unified tri-cue encoder (UTEnc), comprising three tightly coupled components: a local-global point cloud encoder (LG-PEnc), a mixture-of-experts-based geometry encoder (MoE-GEnc), and an image encoder from a well-pretrained visual-language model. LG-PEnc efficiently encodes the spatial and structural information of point clouds to produce foundational representations for each object, whose pairwise similarities serve as the sole association metric. MoE-GEnc seamlessly interacts with LG-PEnc to model inter-object geometric relationships across frames, adaptively compensating for inter-frame object motion without relying on any class-specific priors. The image encoder is kept frozen and is used exclusively during training to provide a well-pretrained representation space. Point cloud representations are aligned to this space to supervise the motion compensation process, encouraging representation invariance across frames for the same object while enhancing discriminability among different objects. Through this formulation, RegTrack attains robust, efficient, and generalizable inference using only point cloud inputs, requiring just 2.6M parameters. Extensive experiments on KITTI and nuScenes show that RegTrack outperforms its thirty-five competitors.
>
---
#### [replaced 024] SpikePingpong: Spike Vision-based Fast-Slow Pingpong Robot System
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人高精度操控任务，旨在解决动态环境中高速物体的精准控制问题。通过结合视觉与学习策略，实现乒乓球的快速准确击打。**

- **链接: [https://arxiv.org/pdf/2506.06690v2](https://arxiv.org/pdf/2506.06690v2)**

> **作者:** Hao Wang; Chengkai Hou; Xianglong Li; Yankai Fu; Chenxuan Li; Ning Chen; Gaole Dai; Jiaming Liu; Tiejun Huang; Shanghang Zhang
>
> **摘要:** Learning to control high-speed objects in dynamic environments represents a fundamental challenge in robotics. Table tennis serves as an ideal testbed for advancing robotic capabilities in dynamic environments. This task presents two fundamental challenges: it requires a high-precision vision system capable of accurately predicting ball trajectories under complex dynamics, and it necessitates intelligent control strategies to ensure precise ball striking to target regions. High-speed object manipulation typically demands advanced visual perception hardware capable of capturing rapid motion with exceptional temporal resolution. Drawing inspiration from Kahneman's dual-system theory, where fast intuitive processing complements slower deliberate reasoning, there exists an opportunity to develop more robust perception architectures that can handle high-speed dynamics while maintaining accuracy. To this end, we present \textit{\textbf{SpikePingpong}}, a novel system that integrates spike-based vision with imitation learning for high-precision robotic table tennis. We develop a Fast-Slow system architecture where System 1 provides rapid ball detection and preliminary trajectory prediction with millisecond-level responses, while System 2 employs spike-oriented neural calibration for precise hittable position corrections. For strategic ball striking, we introduce Imitation-based Motion Planning And Control Technology, which learns optimal robotic arm striking policies through demonstration-based learning. Experimental results demonstrate that \textit{\textbf{SpikePingpong}} achieves a remarkable 92\% success rate for 30 cm accuracy zones and 70\% in the more challenging 20 cm precision targeting. This work demonstrates the potential of Fast-Slow architectures for advancing robotic capabilities in time-critical manipulation tasks.
>
---
#### [replaced 025] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks
- **分类: cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2510.22129v3](https://arxiv.org/pdf/2510.22129v3)**

> **作者:** Matthias Jammot; Björn Braun; Paul Streli; Rafael Wampfler; Christian Holz
>
> **备注:** Accepted for publication at NeurIPS 2025
>
> **摘要:** Understanding affect is central to anticipating human behavior, yet current egocentric vision benchmarks largely ignore the person's emotional states that shape their decisions and actions. Existing tasks in egocentric perception focus on physical activities, hand-object interactions, and attention modeling - assuming neutral affect and uniform personality. This limits the ability of vision systems to capture key internal drivers of behavior. In this paper, we present egoEMOTION, the first dataset that couples egocentric visual and physiological signals with dense self-reports of emotion and personality across controlled and real-world scenarios. Our dataset includes over 50 hours of recordings from 43 participants, captured using Meta's Project Aria glasses. Each session provides synchronized eye-tracking video, headmounted photoplethysmography, inertial motion data, and physiological baselines for reference. Participants completed emotion-elicitation tasks and naturalistic activities while self-reporting their affective state using the Circumplex Model and Mikels' Wheel as well as their personality via the Big Five model. We define three benchmark tasks: (1) continuous affect classification (valence, arousal, dominance); (2) discrete emotion classification; and (3) trait-level personality inference. We show that a classical learning-based method, as a simple baseline in real-world affect prediction, produces better estimates from signals captured on egocentric vision systems than processing physiological signals. Our dataset establishes emotion and personality as core dimensions in egocentric perception and opens new directions in affect-driven modeling of behavior, intent, and interaction.
>
---
#### [replaced 026] HSSBench: Benchmarking Humanities and Social Sciences Ability for Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态语言模型评估任务，旨在解决HSS领域基准缺失的问题。提出HSSBench，涵盖多语言HSS任务，评估模型跨学科推理能力。**

- **链接: [https://arxiv.org/pdf/2506.03922v2](https://arxiv.org/pdf/2506.03922v2)**

> **作者:** Zhaolu Kang; Junhao Gong; Jiaxu Yan; Wanke Xia; Yian Wang; Ziwen Wang; Huaxuan Ding; Zhuo Cheng; Wenhao Cao; Zhiyuan Feng; Siqi He; Shannan Yan; Junzhe Chen; Xiaomin He; Chaoya Jiang; Wei Ye; Kaidong Yu; Xuelong Li
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated significant potential to advance a broad range of domains. However, current benchmarks for evaluating MLLMs primarily emphasize general knowledge and vertical step-by-step reasoning typical of STEM disciplines, while overlooking the distinct needs and potential of the Humanities and Social Sciences (HSS). Tasks in the HSS domain require more horizontal, interdisciplinary thinking and a deep integration of knowledge across related fields, which presents unique challenges for MLLMs, particularly in linking abstract concepts with corresponding visual representations. Addressing this gap, we present HSSBench, a dedicated benchmark designed to assess the capabilities of MLLMs on HSS tasks in multiple languages, including the six official languages of the United Nations. We also introduce a novel data generation pipeline tailored for HSS scenarios, in which multiple domain experts and automated agents collaborate to generate and iteratively refine each sample. HSSBench contains over 13,000 meticulously designed samples, covering six key categories. We benchmark more than 20 mainstream MLLMs on HSSBench and demonstrate that it poses significant challenges even for state-of-the-art models. We hope that this benchmark will inspire further research into enhancing the cross-disciplinary reasoning abilities of MLLMs, especially their capacity to internalize and connect knowledge across fields.
>
---
#### [replaced 027] Reproducing and Improving CheXNet: Deep Learning for Chest X-ray Disease Classification
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.06646v3](https://arxiv.org/pdf/2505.06646v3)**

> **作者:** Daniel J. Strick; Carlos Garcia; Anthony Huang; Thomas Gardos
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Deep learning for radiologic image analysis is a rapidly growing field in biomedical research and is likely to become a standard practice in modern medicine. On the publicly available NIH ChestX-ray14 dataset, containing X-ray images that are classified by the presence or absence of 14 different diseases, we reproduced an algorithm known as CheXNet, as well as explored other algorithms that outperform CheXNet's baseline metrics. Model performance was primarily evaluated using the F1 score and AUC-ROC, both of which are critical metrics for imbalanced, multi-label classification tasks in medical imaging. The best model achieved an average AUC-ROC score of 0.85 and an average F1 score of 0.39 across all 14 disease classifications present in the dataset.
>
---
#### [replaced 028] Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20161v2](https://arxiv.org/pdf/2602.20161v2)**

> **作者:** Abdelrahman Shaker; Ahmed Heakl; Jaseel Muhammad; Ritesh Thawkar; Omkar Thawakar; Senmao Li; Hisham Cholakkal; Ian Reid; Eric P. Xing; Salman Khan; Fahad Shahbaz Khan
>
> **备注:** Project page: https://amshaker.github.io/Mobile-O/
>
> **摘要:** Unified multimodal models can both understand and generate visual content within a single architecture. Existing models, however, remain data-hungry and too heavy for deployment on edge devices. We present Mobile-O, a compact vision-language-diffusion model that brings unified multimodal intelligence to a mobile device. Its core module, the Mobile Conditioning Projector (MCP), fuses vision-language features with a diffusion generator using depthwise-separable convolutions and layerwise alignment. This design enables efficient cross-modal conditioning with minimal computational cost. Trained on only a few million samples and post-trained in a novel quadruplet format (generation prompt, image, question, answer), Mobile-O jointly enhances both visual understanding and generation capabilities. Despite its efficiency, Mobile-O attains competitive or superior performance compared to other unified models, achieving 74% on GenEval and outperforming Show-O and JanusFlow by 5% and 11%, while running 6x and 11x faster, respectively. For visual understanding, Mobile-O surpasses them by 15.3% and 5.1% averaged across seven benchmarks. Running in only ~3s per 512x512 image on an iPhone, Mobile-O establishes the first practical framework for real-time unified multimodal understanding and generation on edge devices. We hope Mobile-O will ease future research in real-time unified multimodal intelligence running entirely on-device with no cloud dependency. Our code, models, datasets, and mobile application are publicly available at https://amshaker.github.io/Mobile-O/
>
---
#### [replaced 029] VISIONLOGIC: From Neuron Activations to Causally Grounded Concept Rules for Vision Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.10547v2](https://arxiv.org/pdf/2503.10547v2)**

> **作者:** Chuqin Geng; Yuhe Jiang; Ziyu Zhao; Haolin Ye; Anqi Xing; Li Zhang; Xujie Si
>
> **备注:** 27 pages, 18 figures
>
> **摘要:** While concept-based explanations improve interpretability over local attributions, they often rely on correlational signals and lack causal validation. We introduce VisionLogic, a novel neural-symbolic framework that produces faithful, hierarchical explanations as global logical rules over causally validated concepts. VisionLogic first learns activation thresholds that abstract neuron activations into predicates, then induces class-level logical rules from these predicates. It then grounds predicates to visual concepts via ablation-based causal tests with iterative region refinement, ensuring that discovered concepts correspond to features that are causal for predicate activation. Across different vision architectures such as CNNs and ViTs, it produces interpretable concepts and compact rules that largely preserve the original model's predictive performance. In our large-scale human evaluations, VisionLogic's concept explanations significantly improve participants' understanding of model behavior over prior concept-based methods.
>
---
#### [replaced 030] Towards Attributions of Input Variables in a Coalition
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2309.13411v3](https://arxiv.org/pdf/2309.13411v3)**

> **作者:** Xinhao Zheng; Huiqi Deng; Quanshi Zhang
>
> **备注:** Accepted to the 2025 International Conference on Machine Learning (ICML 2025)
>
> **摘要:** This paper focuses on the fundamental challenge of partitioning input variables in attribution methods for Explainable AI, particularly in Shapley value-based approaches. Previous methods always compute attributions given a predefined partition but lack theoretical guidance on how to form meaningful variable partitions. We identify that attribution conflicts arise when the attribution of a coalition differs from the sum of its individual variables' attributions. To address this, we analyze the numerical effects of AND-OR interactions in AI models and extend the Shapley value to a new attribution metric for variable coalitions. Our theoretical findings reveal that specific interactions cause attribution conflicts, and we propose three metrics to evaluate coalition faithfulness. Experiments on synthetic data, NLP, image classification, and the game of Go validate our approach, demonstrating consistency with human intuition and practical applicability.
>
---
#### [replaced 031] Tree crop mapping of South America reveals links to deforestation and conservation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17372v2](https://arxiv.org/pdf/2602.17372v2)**

> **作者:** Yuchang Jiang; Anton Raichuk; Xiaoye Tong; Vivien Sainte Fare Garnot; Daniel Ortiz-Gonzalo; Dan Morris; Konrad Schindler; Jan Dirk Wegner; Maxim Neumann
>
> **摘要:** Monitoring tree crop expansion is vital for zero-deforestation policies like the European Union's Regulation on Deforestation-free Products (EUDR). However, these efforts are hindered by a lack of highresolution data distinguishing diverse agricultural systems from forests. Here, we present the first 10m-resolution tree crop map for South America, generated using a multi-modal, spatio-temporal deep learning model trained on Sentinel-1 and Sentinel-2 satellite imagery time series. The map identifies approximately 11 million hectares of tree crops, 23% of which is linked to 2000-2020 forest cover loss. Critically, our analysis reveals that existing regulatory maps supporting the EUDR often classify established agriculture, particularly smallholder agroforestry, as "forest". This discrepancy risks false deforestation alerts and unfair penalties for small-scale farmers. Our work mitigates this risk by providing a high-resolution baseline, supporting conservation policies that are effective, inclusive, and equitable.
>
---
#### [replaced 032] On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究视觉-语言-动作模型在多模态扰动下的鲁棒性，解决实际部署中多模态干扰问题，提出RobustVLA方法提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.00037v4](https://arxiv.org/pdf/2510.00037v4)**

> **作者:** Jianing Guo; Zhenhong Wu; Chang Tu; Yiyao Ma; Xiangqi Kong; Zhiqian Liu; Jiaming Ji; Shuning Zhang; Yuanpei Chen; Kai Chen; Qi Dou; Yaodong Yang; Xianglong Liu; Huijie Zhao; Weifeng Lv; Simin Li
>
> **摘要:** In Vision-Language-Actionf(VLA) models, robustness to real-world perturbations is critical for deployment. Existing methods target simple visual disturbances, overlooking the broader multi-modal perturbations that arise in actions, instructions, environments, and observations. Here, we first evaluate the robustness of mainstream VLAs under 17 perturbations across four modalities. We find (1) actions as the most fragile modality, (2) Existing visual-robust VLA do not gain robustness in other modality, and (3) pi0 demonstrates superior robustness. To build multi-modal robust VLAs, we propose RobustVLA against perturbations in VLA inputs and outputs. For output robustness, we perform offline robust optimization against worst-case action noise that maximizes mismatch in flow matching objective. This can be seen as adversarial training, label smoothing, and outlier penalization. For input robustness, we enforce consistent actions across input variations that preserve task semantics. To account for multiple perturbations, we formulate robustness as a multi-armed bandit problem and apply an upper confidence bound algorithm to automatically identify the most harmful noise. Experiments on LIBERO demonstrate our RobustVLA delivers absolute gains over baselines of 12.6% on the pi0 backbone and 10.4% on the OpenVLA backbone across all 17 perturbations, achieving 50.6x faster inference than existing visual-robust BYOVLA that requires external LLMs, and a 10.4% gain under mixed perturbations. On the real-world FR5 robot, under four types of multimodal perturbations, RobustVLA shows strong low-data performance, outperforming pi0 by 65.6% success rate with 25 demonstrations. Even with abundant demos, our method still outperform pi0 by 30% success rate. Code and demo videos available at https://github.com/gakakulicc/RobustVLA.
>
---
#### [replaced 033] A Cognitive Process-Inspired Architecture for Subject-Agnostic Brain Visual Decoding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.02565v2](https://arxiv.org/pdf/2511.02565v2)**

> **作者:** Jingyu Lu; Haonan Wang; Qixiang Zhang; Xiaomeng Li
>
> **备注:** Accepted at the International Conference on Learning Representations (ICLR), 2026
>
> **摘要:** Subject-agnostic brain decoding, which aims to reconstruct continuous visual experiences from fMRI without subject-specific training, holds great potential for clinical applications. However, this direction remains underexplored due to challenges in cross-subject generalization and the complex nature of brain signals. In this work, we propose Visual Cortex Flow Architecture (VCFlow), a novel hierarchical decoding framework that explicitly models the ventral-dorsal architecture of the human visual system to learn multi-dimensional representations. By disentangling and leveraging features from early visual cortex, ventral, and dorsal streams, VCFlow captures diverse and complementary cognitive information essential for visual reconstruction. Furthermore, we introduce a feature-level contrastive learning strategy to enhance the extraction of subject-invariant semantic representations, thereby enhancing subject-agnostic applicability to previously unseen subjects. Unlike conventional pipelines that need more than 12 hours of per-subject data and heavy computation, VCFlow sacrifices only 7\% accuracy on average yet generates each reconstructed video in 10 seconds without any retraining, offering a fast and clinically scalable solution. The source code will be released upon acceptance of the paper.
>
---
#### [replaced 034] Pareto-Guided Optimization for Uncertainty-Aware Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19365v2](https://arxiv.org/pdf/2601.19365v2)**

> **作者:** Jinming Zhang; Youpeng Yang; Xi Yang; Haosen Shi; Yuyao Yan; Qiufeng Wang; Guangliang Cheng; Kaizhu Huang
>
> **摘要:** Uncertainty in medical image segmentation is inherently non-uniform, with boundary regions exhibiting substantially higher ambiguity than interior areas. Conventional training treats all pixels equally, leading to unstable optimization during early epochs when predictions are unreliable. We argue that this instability hinders convergence toward Pareto-optimal solutions and propose a region-wise curriculum strategy that prioritizes learning from certain regions and gradually incorporates uncertain ones, reducing gradient variance. Methodologically, we introduce a Pareto-consistent loss that balances trade-offs between regional uncertainties by adaptively reshaping the loss landscape and constraining convergence dynamics between interior and boundary regions; this guides the model toward Pareto-approximate solutions. To address boundary ambiguity, we further develop a fuzzy labeling mechanism that maintains binary confidence in non-boundary areas while enabling smooth transitions near boundaries, stabilizing gradients, and expanding flat regions in the loss surface. Experiments on brain metastasis and non-metastatic tumor segmentation show consistent improvements across multiple configurations, with our method outperforming traditional crisp-set approaches in all tumor subregions.
>
---
#### [replaced 035] CRAFT-LoRA: Content-Style Personalization via Rank-Constrained Adaptation and Training-Free Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18936v2](https://arxiv.org/pdf/2602.18936v2)**

> **作者:** Yu Li; Yujun Cai; Chi Zhang
>
> **摘要:** Personalized image generation requires effectively balancing content fidelity with stylistic consistency when synthesizing images based on text and reference examples. Low-Rank Adaptation (LoRA) offers an efficient personalization approach, with potential for precise control through combining LoRA weights on different concepts. However, existing combination techniques face persistent challenges: entanglement between content and style representations, insufficient guidance for controlling elements' influence, and unstable weight fusion that often require additional training. We address these limitations through CRAFT-LoRA, with complementary components: (1) rank-constrained backbone fine-tuning that injects low-rank projection residuals to encourage learning decoupled content and style subspaces; (2) a prompt-guided approach featuring an expert encoder with specialized branches that enables semantic extension and precise control through selective adapter aggregation; and (3) a training-free, timestep-dependent classifier-free guidance scheme that enhances generation stability by strategically adjusting noise predictions across diffusion steps. Our method significantly improves content-style disentanglement, enables flexible semantic control over LoRA module combinations, and achieves high-fidelity generation without additional retraining overhead.
>
---
#### [replaced 036] An Efficient LiDAR-Camera Fusion Network for Multi-Class 3D Dynamic Object Detection and Trajectory Prediction
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于3D动态目标检测与轨迹预测任务，旨在提升服务机器人在复杂环境中的感知能力。提出一种高效融合LiDAR与相机的框架，实现快速准确的目标检测与轨迹预测。**

- **链接: [https://arxiv.org/pdf/2504.13647v2](https://arxiv.org/pdf/2504.13647v2)**

> **作者:** Yushen He; Lei Zhao; Tianchen Deng; Zipeng Fang; Weidong Chen
>
> **摘要:** Service mobile robots are often required to avoid dynamic objects while performing their tasks, but they usually have only limited computational resources. To further advance the practical application of service robots in complex dynamic environments, we propose an efficient multi-modal framework for 3D object detection and trajectory prediction, which synergistically integrates LiDAR and camera inputs to achieve real-time perception of pedestrians, vehicles, and riders in 3D space.The framework incorporates two novel models: 1) a Unified modality detector with Mamba and Transformer (UniMT) for object detection, which achieves high-accuracy object detection with fast inference speed, and 2) a Reference Trajectory-based Multi-Class Transformer (RTMCT) for efficient and diverse trajectory prediction of multi-class objects with flexible-length trajectories. Evaluations on the CODa benchmark demonstrate that our method outperforms existing ones in both detection (+3.71\% in mAP) and trajectory prediction (-0.408m in minADE$_5$ of pedestrians) metrics. Furthermore, on the challenging nuScenes detection benchmark, our detection model achieves competitive performance among LiDAR-camera fusion methods, with a mAP of 72.7\% and NDS of 75.3\%. Remarkably, the system demonstrates exceptional generalizability and practical deployment potential. When transferred and implemented on a wheelchair robot with an entry-level NVIDIA RTX 3060 GPU, it achieves real-time inference at 13.9 frames per second (FPS) with satisfactory accuracy. To facilitate reproducibility and practical deployment, we release the related code of the method at \href{https://github.com/TossherO/3D_Perception}{https://github.com/TossherO/3D\_Perception} and its ROS inference version at \href{https://github.com/TossherO/ros_packages}{https://github.com/TossherO/ros\_packages}.
>
---
#### [replaced 037] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16156v2](https://arxiv.org/pdf/2511.16156v2)**

> **作者:** Jian Ma; Qirong Peng; Xujie Zhu; Peixing Xie; Chen Chen; Haonan Lu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Diffusion Transformers (DiTs) have shown exceptional performance in image generation, yet their large parameter counts incur high computational costs, impeding deployment in resource-constrained settings. To address this, we propose Pluggable Pruning with Contiguous Layer Distillation (PPCL), a flexible structured pruning framework specifically designed for DiT architectures. First, we identify redundant layer intervals through a linear probing mechanism combined with the first-order differential trend analysis of similarity metrics. Subsequently, we propose a plug-and-play teacher-student alternating distillation scheme tailored to integrate depth-wise and width-wise pruning within a single training phase. This distillation framework enables flexible knowledge transfer across diverse pruning ratios, eliminating the need for per-configuration retraining. Extensive experiments on multiple Multi-Modal Diffusion Transformer architecture models demonstrate that PPCL achieves a 50\% reduction in parameter count compared to the full model, with less than 3\% degradation in key objective metrics. Notably, our method maintains high-quality image generation capabilities while achieving higher compression ratios, rendering it well-suited for resource-constrained environments. The open-source code, checkpoints for PPCL can be found at the following link: https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning.
>
---
#### [replaced 038] Universal 3D Shape Matching via Coarse-to-Fine Language Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19112v2](https://arxiv.org/pdf/2602.19112v2)**

> **作者:** Qinfeng Xiao; Guofeng Mei; Bo Yang; Liying Zhang; Jian Zhang; Kit-lun Yick
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Establishing dense correspondences between shapes is a crucial task in computer vision and graphics, while prior approaches depend on near-isometric assumptions and homogeneous subject types (i.e., only operate for human shapes). However, building semantic correspondences for cross-category objects remains challenging and has received relatively little attention. To achieve this, we propose UniMatch, a semantic-aware, coarse-to-fine framework for constructing dense semantic correspondences between strongly non-isometric shapes without restricting object categories. The key insight is to lift "coarse" semantic cues into "fine" correspondence, which is achieved through two stages. In the "coarse" stage, we perform class-agnostic 3D segmentation to obtain non-overlapping semantic parts and prompt multimodal large language models (MLLMs) to identify part names. Then, we employ pretrained vision language models (VLMs) to extract text embeddings, enabling the construction of matched semantic parts. In the "fine" stage, we leverage these coarse correspondences to guide the learning of dense correspondences through a dedicated rank-based contrastive scheme. Thanks to class-agnostic segmentation, language guiding, and rank-based contrastive learning, our method is versatile for universal object categories and requires no predefined part proposals, enabling universal matching for inter-class and non-isometric shapes. Extensive experiments demonstrate UniMatch consistently outperforms competing methods in various challenging scenarios.
>
---
#### [replaced 039] Decouple, Reorganize, and Fuse: A Multimodal Framework for Cancer Survival Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.18632v2](https://arxiv.org/pdf/2508.18632v2)**

> **作者:** Huayi Wang; Haochao Ying; Yuyang Xu; Qibo Qiu; Cheng Zhang; Danny Z. Chen; Ying Sun; Jian Wu
>
> **备注:** 13 pages
>
> **摘要:** Cancer survival analysis commonly integrates information across diverse medical modalities to make survival-time predictions. Existing methods primarily focus on extracting different decoupled features of modalities and performing fusion operations such as concatenation, attention, and \revm{Mixture-of-Experts (MoE)-based fusion. However, these methods still face two key challenges: i) Fixed fusion schemes (concatenation and attention) can lead to model over-reliance on predefined feature combinations, limiting the dynamic fusion of decoupled features; ii) in MoE-based fusion methods, each expert network handles separate decoupled features, which limits information interaction among the decoupled features. To address these challenges, we propose a novel Decoupling-Reorganization-Fusion framework (DeReF), which devises a random feature reorganization strategy between modalities decoupling and dynamic MoE fusion modules.Its advantages are: i) it increases the diversity of feature combinations and granularity, enhancing the generalization ability of the subsequent expert networks; ii) it overcomes the problem of information closure and helps expert networks better capture information among decoupled features. Additionally, we incorporate a regional cross-attention network within the modality decoupling module to improve the representation quality of decoupled features. Extensive experimental results on our in-house Liver Cancer (LC) and three widely used TCGA public datasets confirm the effectiveness of our proposed method. Codes are available at https://github.com/ZJUMAI/DeReF.
>
---
#### [replaced 040] RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.13289v3](https://arxiv.org/pdf/2505.13289v3)**

> **作者:** Alonso Urbano; David W. Romero; Max Zimmer; Sebastian Pokutta
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Real world data often exhibits unknown, instance-specific symmetries that rarely exactly match a transformation group $G$ fixed a priori. Class-pose decompositions aim to create disentangled representations by factoring inputs into invariant features and a pose $g\in G$ defined relative to a training-dependent, \emph{arbitrary} canonical representation. We introduce RECON, a class-pose agnostic \emph{canonical orientation normalization} that corrects arbitrary canonicals via a simple right translation, yielding \emph{natural}, data-aligned canonicalizations. This enables (i) unsupervised discovery of instance-specific pose distributions, (ii) detection of out-of-distribution poses and (iii) a plug-and-play \emph{test-time canonicalization layer}. This layer can be attached on top of any pre-trained model to infuse group invariance, improving its performance without retraining. We validate on images and molecular ensembles, demonstrating accurate symmetry discovery, and matching or outperforming other canonicalizations in downstream classification.
>
---
#### [replaced 041] Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.10611v2](https://arxiv.org/pdf/2601.10611v2)**

> **作者:** Christopher Clark; Jieyu Zhang; Zixian Ma; Jae Sung Park; Mohammadreza Salehi; Rohun Tripathi; Sangho Lee; Zhongzheng Ren; Chris Dongjoo Kim; Yinuo Yang; Vincent Shao; Yue Yang; Weikai Huang; Ziqi Gao; Taira Anderson; Jianrui Zhang; Jitesh Jain; George Stoica; Winson Han; Ali Farhadi; Ranjay Krishna
>
> **备注:** Updated acknowledgements
>
> **摘要:** Today's strongest video-language models (VLMs) remain proprietary. The strongest open-weight models either rely on synthetic data from proprietary VLMs, effectively distilling from them, or do not disclose their training data or recipe. As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models. Crucially, many downstream applications require more than just high-level video understanding; they require grounding -- either by pointing or by tracking in pixels. Even proprietary models lack this capability. We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks. Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance. Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos. On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).
>
---
#### [replaced 042] NRSeg: Noise-Resilient Learning for BEV Semantic Segmentation via Driving World Models
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文针对BEV语义分割任务，解决标注数据分布单一和合成数据噪声问题，提出NRSeg框架提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2507.04002v2](https://arxiv.org/pdf/2507.04002v2)**

> **作者:** Siyu Li; Fei Teng; Yihong Cao; Kailun Yang; Zhiyong Li; Yaonan Wang
>
> **备注:** Accepted to IEEE Transactions on Image Processing (TIP). The source code will be made publicly available at https://github.com/lynn-yu/NRSeg
>
> **摘要:** Birds' Eye View (BEV) semantic segmentation is an indispensable perception task in end-to-end autonomous driving systems. Unsupervised and semi-supervised learning for BEV tasks, as pivotal for real-world applications, underperform due to the homogeneous distribution of the labeled data. In this work, we explore the potential of synthetic data from driving world models to enhance the diversity of labeled data for robustifying BEV segmentation. Yet, our preliminary findings reveal that generation noise in synthetic data compromises efficient BEV model learning. To fully harness the potential of synthetic data from world models, this paper proposes NRSeg, a noise-resilient learning framework for BEV semantic segmentation. Specifically, a Perspective-Geometry Consistency Metric (PGCM) is proposed to quantitatively evaluate the guidance capability of generated data for model learning. This metric originates from the alignment measure between the perspective road mask of generated data and the mask projected from the BEV labels. Moreover, a Bi-Distribution Parallel Prediction (BiDPP) is designed to enhance the inherent robustness of the model, where the learning process is constrained through parallel prediction of multinomial and Dirichlet distributions. The former efficiently predicts semantic probabilities, whereas the latter adopts evidential deep learning to realize uncertainty quantification. Furthermore, a Hierarchical Local Semantic Exclusion (HLSE) module is designed to address the non-mutual exclusivity inherent in BEV semantic segmentation tasks. Experimental results demonstrate that NRSeg achieves state-of-the-art performance, yielding the highest improvements in mIoU of 13.8% and 11.4% in unsupervised and semi-supervised BEV segmentation tasks, respectively. The source code will be made publicly available at https://github.com/lynn-yu/NRSeg.
>
---
#### [replaced 043] All-Optical Segmentation via Diffractive Neural Networks for Autonomous Driving
- **分类: cs.CV; cs.ET**

- **链接: [https://arxiv.org/pdf/2602.07717v2](https://arxiv.org/pdf/2602.07717v2)**

> **作者:** Yingjie Li; Daniel Robinson; Weilu Gao; Cunxi Yu
>
> **摘要:** Semantic segmentation and lane detection are crucial tasks in autonomous driving systems. Conventional approaches predominantly rely on deep neural networks (DNNs), which incur high energy costs due to extensive analog-to-digital conversions and large-scale image computations required for low-latency, real-time responses. Diffractive optical neural networks (DONNs) have shown promising advantages over conventional DNNs on digital or optoelectronic computing platforms in energy efficiency. By performing all-optical image processing via light diffraction at the speed of light, DONNs save computation energy costs while reducing the overhead associated with analog-to-digital conversions by all-optical encoding and computing. In this work, we propose a novel all-optical computing framework for RGB image segmentation and lane detection in autonomous driving applications. Our experimental results demonstrate the effectiveness of the DONN system for image segmentation on the CityScapes dataset. Additionally, we conduct case studies on lane detection using a customized indoor track dataset and simulated driving scenarios in CARLA, where we further evaluate the model's generalizability under diverse environmental conditions.
>
---
#### [replaced 044] Sound Source Localization for Spatial Mapping of Surgical Actions in Dynamic Scenes
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文属于手术场景理解任务，旨在解决动态场景中声源定位问题。通过融合3D音频与视觉信息，实现手术动作的时空定位与建模。**

- **链接: [https://arxiv.org/pdf/2510.24332v2](https://arxiv.org/pdf/2510.24332v2)**

> **作者:** Jonas Hein; Lazaros Vlachopoulos; Maurits Geert Laurent Olthof; Bastian Sigrist; Philipp Fürnstahl; Matthias Seibold
>
> **摘要:** Purpose: Surgical scene understanding is key to advancing computer-aided and intelligent surgical systems. Current approaches predominantly rely on visual data or end-to-end learning, which limits fine-grained contextual modeling. This work aims to enhance surgical scene representations by integrating 3D acoustic information, enabling temporally and spatially aware multimodal understanding of surgical environments. Methods: We propose a novel framework for generating 4D audio-visual representations of surgical scenes by projecting acoustic localization information from a phased microphone array onto dynamic point clouds from an RGB-D camera. A transformer-based acoustic event detection module identifies relevant temporal segments containing tool-tissue interactions which are spatially localized in the audio-visual scene representation. The system was experimentally evaluated in a realistic operating room setup during simulated surgical procedures performed by experts. Results: The proposed method successfully localizes surgical acoustic events in 3D space and associates them with visual scene elements. Experimental evaluation demonstrates accurate spatial sound localization and robust fusion of multimodal data, providing a comprehensive, dynamic representation of surgical activity. Conclusion: This work introduces the first approach for spatial sound localization in dynamic surgical scenes, marking a significant advancement toward multimodal surgical scene representations. By integrating acoustic and visual data, the proposed framework enables richer contextual understanding and provides a foundation for future intelligent and autonomous surgical systems.
>
---
#### [replaced 045] MIRROR: Multimodal Iterative Reasoning via Reflection on Visual Regions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18746v2](https://arxiv.org/pdf/2602.18746v2)**

> **作者:** Haoyu Zhang; Yuwei Wu; Pengxiang Li; Xintong Zhang; Zhi Gao; Rui Gao; Mingyang Gao; Che Sun; Yunde Jia
>
> **摘要:** In the era of Vision-Language Models (VLMs), enhancing multimodal reasoning capabilities remains a critical challenge, particularly in handling ambiguous or complex visual inputs, where initial inferences often lead to hallucinations or logic errors. Existing VLMs often produce plausible yet ungrounded answers, and even when prompted to "reflect", their corrections may remain detached from the image evidence. To address this, we propose the MIRROR framework for Multimodal Iterative Reasoning via Reflection On visual Regions. By embedding visual reflection as a core mechanism, MIRROR is formulated as a closed-loop process comprising draft, critique, region-based verification, and revision, which are repeated until the output is visually grounded. To facilitate training of this model, we construct **ReflectV**, a visual reflective dataset for multi-turn supervision that explicitly contains reflection triggers, region-based verification actions, and answer revision grounded in visual evidence. Experiments on both general vision-language benchmarks and representative vision-language reasoning benchmarks show that MIRROR improves correctness and reduces visual hallucinations, demonstrating the value of training reflection as an evidence-seeking, region-aware verification process rather than a purely textual revision step.
>
---
#### [replaced 046] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.06437v2](https://arxiv.org/pdf/2503.06437v2)**

> **作者:** Juhyeon Park; Peter Yongho Kim; Jiook Cha; Shinjae Yoo; Taesup Moon
>
> **备注:** ICLR 2026
>
> **摘要:** We present SEED (Semantic Evaluation for Visual Brain Decoding), a novel metric for evaluating the semantic decoding performance of visual brain decoding models. It integrates three complementary metrics, each capturing a different aspect of semantic similarity between images inspired by neuroscientific findings. Using carefully crowd-sourced human evaluation data, we demonstrate that SEED achieves the highest alignment with human evaluation, outperforming other widely used metrics. Through the evaluation of existing visual brain decoding models with SEED, we further reveal that crucial information is often lost in translation, even in the state-of-the-art models that achieve near-perfect scores on existing metrics. This finding highlights the limitations of current evaluation practices and provides guidance for future improvements in decoding models. Finally, to facilitate further research, we open-source the human evaluation data, encouraging the development of more advanced evaluation methods for brain decoding. Our code and the human evaluation data are available at https://github.com/Concarne2/SEED.
>
---
#### [replaced 047] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作推理任务，解决推理延迟高问题。提出Fast-ThinkAct框架，通过可表述的潜在推理实现高效规划，降低延迟并保持性能。**

- **链接: [https://arxiv.org/pdf/2601.09708v2](https://arxiv.org/pdf/2601.09708v2)**

> **作者:** Chi-Pin Huang; Yunze Man; Zhiding Yu; Min-Hung Chen; Jan Kautz; Yu-Chiang Frank Wang; Fu-En Yang
>
> **备注:** CVPR 2026. Project page: https://jasper0314-huang.github.io/fast-thinkact/
>
> **摘要:** Vision-Language-Action (VLA) tasks require reasoning over complex visual scenes and executing adaptive actions in dynamic environments. While recent studies on reasoning VLAs show that explicit chain-of-thought (CoT) can improve generalization, they suffer from high inference latency due to lengthy reasoning traces. We propose Fast-ThinkAct, an efficient reasoning framework that achieves compact yet performant planning through verbalizable latent reasoning. Fast-ThinkAct learns to reason efficiently with latent CoTs by distilling from a teacher, driven by a preference-guided objective to align manipulation trajectories that transfers both linguistic and visual planning capabilities for embodied control. This enables reasoning-enhanced policy learning that effectively connects compact reasoning to action execution. Extensive experiments across diverse embodied manipulation and reasoning benchmarks demonstrate that Fast-ThinkAct achieves strong performance with up to 89.3% reduced inference latency over state-of-the-art reasoning VLAs, while maintaining effective long-horizon planning, few-shot adaptation, and failure recovery.
>
---
#### [replaced 048] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.01874v3](https://arxiv.org/pdf/2601.01874v3)**

> **作者:** Shuhang Chen; Yunqiu Xu; Junjie Xie; Aojun Lu; Tao Feng; Zeying Huang; Ning Zhang; Yi Sun; Yi Yang; Hangjie Yuan
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Despite significant progress, multimodal large language models continue to struggle with visual mathematical problem solving. Some recent works recognize that visual perception is a bottleneck in visual mathematical reasoning, but their solutions are limited to improving the extraction and interpretation of visual inputs. Notably, they all ignore the key issue of whether the extracted visual cues are faithfully integrated and properly utilized in subsequent reasoning. Motivated by this, we present CogFlow, a novel cognitive-inspired three-stage framework that incorporates a knowledge internalization stage, explicitly simulating the hierarchical flow of human reasoning: perception$\Rightarrow$internalization$\Rightarrow$reasoning. In line with this hierarchical flow, we holistically enhance all its stages. We devise Synergistic Visual Rewards to boost perception capabilities in parametric and semantic spaces, jointly improving visual information extraction from symbols and diagrams. To guarantee faithful integration of extracted visual cues into subsequent reasoning, we introduce a Knowledge Internalization Reward model in the internalization stage, bridging perception and reasoning. Moreover, we design a Visual-Gated Policy Optimization algorithm to further enforce the reasoning is grounded with the visual knowledge, preventing models seeking shortcuts that appear coherent but are visually ungrounded reasoning chains. Moreover, we contribute a new dataset MathCog for model training, which contains samples with over 120K high-quality perception-reasoning aligned annotations. Comprehensive experiments and analysis on commonly used visual mathematical reasoning benchmarks validate the superiority of the proposed CogFlow. Project page: https://shchen233.github.io/cogflow.
>
---
#### [replaced 049] Sim2Radar: Toward Bridging the Radar Sim-to-Real Gap with VLM-Guided Scene Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.13314v3](https://arxiv.org/pdf/2602.13314v3)**

> **作者:** Emily Bejerano; Federico Tondolo; Ayaan Qayyum; Xiaofan Yu; Xiaofan Jiang
>
> **摘要:** Millimeter-wave (mmWave) radar provides reliable perception in visually degraded indoor environments (e.g., smoke, dust, and low light), but learning-based radar perception is bottlenecked by the scarcity and cost of collecting and annotating large-scale radar datasets. We present Sim2Radar, an end-to-end framework that synthesizes training radar data directly from single-view RGB images, enabling scalable data generation without manual scene modeling. Sim2Radar reconstructs a material-aware 3D scene by combining monocular depth estimation, segmentation, and vision-language reasoning to infer object materials, then simulates mmWave propagation with a configurable physics-based ray tracer using Fresnel reflection models parameterized by ITU-R electromagnetic properties. Evaluated on real-world indoor scenes, Sim2Radar improves downstream 3D radar perception via transfer learning: pre-training a radar point-cloud object detection model on synthetic data and fine-tuning on real radar yields up to +3.7 3D AP (IoU 0.3), with gains driven primarily by improved spatial localization. These results suggest that physics-based, vision-driven radar simulation can provide effective geometric priors for radar learning and measurably improve performance under limited real-data supervision.
>
---
#### [replaced 050] DVLA-RL: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.00795v2](https://arxiv.org/pdf/2602.00795v2)**

> **作者:** Wenhao Li; Xianjing Meng; Qiangchang Wang; Zhongyi Han; Zhibin Wu; Yilong Yin
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Few-shot learning (FSL) aims to generalize to novel categories with only a few samples. Recent approaches incorporate large language models (LLMs) to enrich visual representations with semantic embeddings derived from class names. However, they overlook progressive and adaptive alignment between vision and language from low-level to high-level semantics, resulting in limited semantic gains. To address these challenges, we propose Dual-level Vision-Language Alignment with Reinforcement Learning gating (DVLA-RL), which consists of Dual-level Semantic Construction (DSC) and RL-gated Attention (RLA). Specifically, DSC conditions LLMs on both class names and support samples to generate discriminative attributes, progressively selects the most relevant ones, and then synthesizes them into coherent class descriptions. This process provides complementary low-level attributes and high-level descriptions, enabling both fine-grained grounding and holistic class understanding. To dynamically integrate dual-level semantics along with the visual network layers, RLA formulates cross-modal fusion as a sequential decision process. A lightweight policy trained with episodic REINFORCE adaptively adjusts the contributions of self-attention and cross-attention to integrate textual and visual tokens. As a result, shallow layers refine local attributes and deep layers emphasize global semantics, enabling more precise cross-modal alignment. This achieves class-specific discrimination and generalized representations with merely a few support samples. DVLA-RL achieves new state-of-the-art performance across nine benchmarks in three diverse FSL scenarios.
>
---
#### [replaced 051] SAS-Net: Scene-Appearance Separation Network for Robust Spatiotemporal Registration in Bidirectional Photoacoustic Microscopy
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.09050v2](https://arxiv.org/pdf/2602.09050v2)**

> **作者:** Jiahao Qin
>
> **备注:** 21 pages, 6 figures, 3 tables
>
> **摘要:** High-speed optical-resolution photoacoustic microscopy (OR-PAM) with bidirectional scanning enables rapid functional brain imaging but introduces severe spatiotemporal misalignment from coupled scan-direction-dependent domain shift and geometric distortion. Conventional registration methods rely on brightness constancy, an assumption violated under bidirectional scanning, leading to unreliable alignment. A unified scene-appearance separation framework is proposed to jointly address domain shift and spatial misalignment. The proposed architecture separates domain-invariant scene content from domain-specific appearance characteristics, enabling cross-domain reconstruction with geometric preservation. A scene consistency loss promotes geometric correspondence in the latent space, linking domain shift correction with spatial registration within a single framework. For in vivo mouse brain vasculature imaging, the proposed method achieves normalized cross-correlation (NCC) of 0.961 and structural similarity index (SSIM) of 0.894, substantially outperforming conventional methods. Ablation studies demonstrate that domain alignment loss is critical, with its removal causing 82% NCC reduction (0.961 to 0.175), while scene consistency and cycle consistency losses provide complementary regularization for optimal performance. The method achieves 11.2 ms inference time per frame (86 fps), substantially exceeding typical OR-PAM acquisition rates and enabling real-time processing. These results suggest that the proposed framework enables robust high-speed bidirectional OR-PAM for reliable quantitative and longitudinal functional imaging. The code will be publicly available at https://github.com/D-ST-Sword/SAS-Net
>
---
#### [replaced 052] DeLTa: Demonstration and Language-Guided Novel Transparent Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决透明物体长时序、高精度操控问题。提出DeLTa框架，结合语言指令与视觉感知，实现无需额外训练的透明物体操作。**

- **链接: [https://arxiv.org/pdf/2510.05662v2](https://arxiv.org/pdf/2510.05662v2)**

> **作者:** Taeyeop Lee; Gyuree Kang; Bowen Wen; Youngho Kim; Seunghyeok Back; In So Kweon; David Hyunchul Shim; Kuk-Jin Yoon
>
> **备注:** Project page: https://sites.google.com/view/DeLTa25/
>
> **摘要:** Despite the prevalence of transparent object interactions in human everyday life, transparent robotic manipulation research remains limited to short-horizon tasks and basic grasping capabilities. Although some methods have partially addressed these issues, most of them have limitations in generalization to novel objects and are insufficient for precise long-horizon robot manipulation. To address this limitation, we propose DeLTa (Demonstration and Language-Guided Novel Transparent Object Manipulation), a novel framework that integrates depth estimation, 6D pose estimation, and vision-language planning for precise long-horizon manipulation of transparent objects guided by natural language task instructions. A key advantage of our method is its single-demonstration approach, which generalizes 6D trajectories to novel transparent objects without requiring category-level priors or additional training. Additionally, we present a task planner that refines the VLM-generated plan to account for the constraints of a single-arm, eye-in-hand robot for long-horizon object manipulation tasks. Through comprehensive evaluation, we demonstrate that our method significantly outperforms existing transparent object manipulation approaches, particularly in long-horizon scenarios requiring precise manipulation capabilities. Project page: https://sites.google.com/view/DeLTa25/
>
---
#### [replaced 053] Two Models for Surface Segmentation using the Total Variation of the Normal Vector
- **分类: cs.CV; math.NA**

- **链接: [https://arxiv.org/pdf/2412.00445v2](https://arxiv.org/pdf/2412.00445v2)**

> **作者:** Manuel Weiß; Lukas Baumgärtner; Laura Weigl; Ronny Bergmann; Stephan Schmidt; Roland Herzog
>
> **摘要:** We consider the problem of surface segmentation, where the goal is to partition a surface represented by a triangular mesh. The segmentation is based on the similarity of the normal vector field to a given set of label vectors. We propose a variational approach and compare two different regularizers, both based on a total variation measure. The first regularizer penalizes the total variation of the assignment function directly, while the second regularizer penalizes the total variation in the label space. In order to solve the resulting optimization problems, we use variations of the split Bregman (ADMM) iteration adapted to the problem at hand. While computationally more expensive, the second regularizer yields better results in our experiments. In particular it removes noise more reliably in regions of constant curvature. In order to mitigate the computational cost, we present a manifold Newton scheme for the most expensive subproblem, which is related to the Riemannian center of mass on a sphere. This significantly improves the computational cost.
>
---
#### [replaced 054] Revisiting the Generalization Problem of Low-level Vision Models Through the Lens of Image Deraining
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.12600v3](https://arxiv.org/pdf/2502.12600v3)**

> **作者:** Jinfan Hu; Zhiyuan You; Jinjin Gu; Kaiwen Zhu; Tianfan Xue; Chao Dong
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2305.15134
>
> **摘要:** Generalization to unseen degradations remains a fundamental challenge for low-level vision models. This paper aims to investigate the underlying mechanism of this failure, using image deraining as a primary case study due to its well-defined and decoupled structure. Through systematic experiments, we reveal that generalization issues are not primarily caused by limited network capacity, but rather by a ``shortcut learning'' phenomenon driven by the relative complexity between image content and degradation patterns. We find that when background content is excessively complex, networks preferentially overfit the simpler degradation characteristics to minimize training loss, thereby failing to learn the underlying image distribution. To address this, we propose two principled strategies: (1) balancing the complexity of training data (backgrounds vs. degradations) to redirect the network's focus toward content reconstruction, and (2) leveraging strong content priors from pre-trained generative models to physically constrain the network onto a high-quality image manifold. Extensive experiments on image deraining, denoising, and deblurring validate our theoretical insights. Our work provides an interpretability-driven perspective and a principled methodology for improving the robustness and generalization of low-level vision models.
>
---
#### [replaced 055] MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.17404v3](https://arxiv.org/pdf/2508.17404v3)**

> **作者:** Haoyu Wang; Hao Tang; Donglin Di; Zhilu Zhang; Wangmeng Zuo; Feng Gao; Siwei Ma; Shiliang Zhang
>
> **备注:** Accepted by ICLR 2026. Project: https://hywang2002.github.io/MoSA
>
> **摘要:** Existing video generation models predominantly emphasize appearance fidelity while exhibiting limited ability to synthesize complex human motions, such as whole-body movements, long-range dynamics, and fine-grained human-environment interactions. This often leads to unrealistic or physically implausible movements with inadequate structural coherence. To conquer these challenges, we propose MoSA, which decouples the process of human video generation into two components, i.e., structure generation and appearance generation. MoSA first employs a 3D structure transformer to generate a human motion sequence from the text prompt. The remaining video appearance is then synthesized under the guidance of this structural sequence. We achieve fine-grained control over the sparse human structures by introducing Human-Aware Dynamic Control modules with a dense tracking constraint during training. The modeling of human-environment interactions is improved through the proposed contact constraint. Those two components work comprehensively to ensure the structural and appearance fidelity across the generated videos. This paper also contributes a large-scale human video dataset, which features more complex and diverse motions than existing human video datasets. We conduct comprehensive comparisons between MoSA and a variety of approaches, including general video generation models, human video generation models, and human animation models. Experiments demonstrate that MoSA substantially outperforms existing approaches across the majority of evaluation metrics.
>
---
#### [replaced 056] Is Exchangeability better than I.I.D to handle Data Distribution Shifts while Pooling Data for Data-scarce Medical image segmentation?
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.19575v2](https://arxiv.org/pdf/2507.19575v2)**

> **作者:** Ayush Roy; Samin Enam; Jun Xia; Won Hwa Kim; Vishnu Suresh Lokhande
>
> **备注:** MIDL 2026
>
> **摘要:** Data scarcity is a major challenge in medical imaging, particularly for deep learning models. While data pooling (combining datasets from multiple sources) and data addition (adding more data from a new dataset) have been shown to enhance model performance, they are not without complications. Specifically, increasing the size of the training dataset through pooling or addition can induce distributional shifts, negatively affecting downstream model performance, a phenomenon known as the "Data Addition Dilemma". While the traditional i.i.d. assumption may not hold in multi-source contexts, assuming exchangeability across datasets provides a more practical framework for data pooling. In this work, we investigate medical image segmentation under these conditions, drawing insights from causal frameworks to propose a method for controlling foreground-background feature discrepancies across all layers of deep networks. This approach improves feature representations, which are crucial in data-addition scenarios. Our method achieves state-of-the-art segmentation performance on histopathology and ultrasound images across five datasets, including a novel ultrasound dataset that we have curated and contributed. Qualitative results demonstrate more refined and accurate segmentation maps compared to prominent baselines across three model architectures.
>
---
#### [replaced 057] Principal Component Analysis-Based Terahertz Self-Supervised Denoising and Deblurring Deep Neural Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.12149v2](https://arxiv.org/pdf/2601.12149v2)**

> **作者:** Pengfei Zhu; Stefano Sfarra; Hai Zhang; Carlo Santulli; Elana Pivarciova; Fabrizio Sarasini; Xavier Maldague
>
> **摘要:** Terahertz (THz) systems inherently introduce frequency-dependent degradation effects, resulting in low-frequency blurring and high-frequency noise in amplitude images. Conventional image processing techniques cannot simultaneously address both issues, and manual intervention is often required due to the unknown boundary between denoising and deblurring. To tackle this challenge, we propose a principal component analysis (PCA)-based THz self-supervised denoising and deblurring network (THz-SSDD). The network employs a Recorrupted-to-Recorrupted self-supervised learning strategy to capture the intrinsic features of noise by exploiting invariance under repeated corruption. PCA decomposition and reconstruction are then applied to restore images across both low and high frequencies. The performance of the THz-SSDD network was evaluated on four types of samples. Training requires only a small set of unlabeled noisy images, and testing across samples with different material properties and measurement modes demonstrates effective denoising and deblurring. Quantitative analysis further validates the network feasibility, showing improvements in image quality while preserving the physical characteristics of the original signals.
>
---
#### [replaced 058] Unbiased Object Detection Beyond Frequency with Visually Prompted Image Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18229v2](https://arxiv.org/pdf/2510.18229v2)**

> **作者:** Xinhao Cai; Liulei Li; Gensheng Pei; Tao Chen; Jinshan Pan; Yazhou Yao; Wenguan Wang
>
> **摘要:** This paper presents a generation-based debiasing framework for object detection. Prior debiasing methods are often limited by the representation diversity of samples, while naive generative augmentation often preserves the biases it aims to solve. Moreover, our analysis reveals that simply generating more data for rare classes is suboptimal due to two core issues: i) instance frequency is an incomplete proxy for the true data needs of a model, and ii) current layout-to-image synthesis lacks the fidelity and control to generate high-quality, complex scenes. To overcome this, we introduce the representation score (RS) to diagnose representational gaps beyond mere frequency, guiding the creation of new, unbiased layouts. To ensure high-quality synthesis, we replace ambiguous text prompts with a precise visual blueprint and employ a generative alignment strategy, which fosters communication between the detector and generator. Our method significantly narrows the performance gap for underrepresented object groups, \eg, improving large/rare instances by 4.4/3.6 mAP over the baseline, and surpassing prior L2I synthesis models by 15.9 mAP for layout accuracy in generated images.
>
---
#### [replaced 059] DriveMamba: Task-Centric Scalable State Space Model for Efficient End-to-End Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.13301v2](https://arxiv.org/pdf/2602.13301v2)**

> **作者:** Haisheng Su; Wei Wu; Feixiang Song; Junjie Zhang; Zhenjie Yang; Junchi Yan
>
> **备注:** Accepted to ICLR2026
>
> **摘要:** Recent advances towards End-to-End Autonomous Driving (E2E-AD) have been often devoted on integrating modular designs into a unified framework for joint optimization e.g. UniAD, which follow a sequential paradigm (i.e., perception-prediction-planning) based on separable Transformer decoders and rely on dense BEV features to encode scene representations. However, such manual ordering design can inevitably cause information loss and cumulative errors, lacking flexible and diverse relation modeling among different modules and sensors. Meanwhile, insufficient training of image backbone and quadratic-complexity of attention mechanism also hinder the scalability and efficiency of E2E-AD system to handle spatiotemporal input. To this end, we propose DriveMamba, a Task-Centric Scalable paradigm for efficient E2E-AD, which integrates dynamic task relation modeling, implicit view correspondence learning and long-term temporal fusion into a single-stage Unified Mamba decoder. Specifically, both extracted image features and expected task outputs are converted into token-level sparse representations in advance, which are then sorted by their instantiated positions in 3D space. The linear-complexity operator enables efficient long-context sequential token modeling to capture task-related inter-dependencies simultaneously. Additionally, a bidirectional trajectory-guided "local-to-global" scan method is designed to preserve spatial locality from ego-perspective, thus facilitating the ego-planning. Extensive experiments conducted on nuScenes and Bench2Drive datasets demonstrate the superiority, generalizability and great efficiency of DriveMamba.
>
---
#### [replaced 060] Probability-Invariant Random Walk Learning on Gyral Folding-Based Cortical Similarity Networks for Alzheimer's and Lewy Body Dementia Diagnosis
- **分类: q-bio.NC; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17557v2](https://arxiv.org/pdf/2602.17557v2)**

> **作者:** Minheng Chen; Tong Chen; Chao Cao; Jing Zhang; Tianming Liu; Li Su; Dajiang Zhu
>
> **摘要:** Alzheimer's disease (AD) and Lewy body dementia (LBD) present overlapping clinical features yet require distinct diagnostic strategies. While neuroimaging-based brain network analysis is promising, atlas-based representations may obscure individualized anatomy. Gyral folding-based networks using three-hinge gyri provide a biologically grounded alternative, but inter-individual variability in cortical folding results in inconsistent landmark correspondence and highly irregular network sizes, violating the fixed-topology and node-alignment assumptions of most existing graph learning methods, particularly in clinical datasets where pathological changes further amplify anatomical heterogeneity. We therefore propose a probability-invariant random-walk-based framework that classifies individualized gyral folding networks without explicit node alignment. Cortical similarity networks are built from local morphometric features and represented by distributions of anonymized random walks, with an anatomy-aware encoding that preserves permutation invariance. Experiments on a large clinical cohort of AD and LBD subjects show consistent improvements over existing gyral folding and atlas-based models, demonstrating robustness and potential for dementia diagnosis.
>
---
#### [replaced 061] Seeing What Matters: Visual Preference Policy Optimization for Visual Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18719v3](https://arxiv.org/pdf/2511.18719v3)**

> **作者:** Ziqi Ni; Yuanzhi Liang; Rui Li; Yi Zhou; Haibin Huang; Chi Zhang; Xuelong Li
>
> **摘要:** Reinforcement learning (RL) has become a powerful tool for post-training visual generative models, with Group Relative Policy Optimization (GRPO) increasingly used to align generators with human preferences. However, existing GRPO pipelines rely on a single scalar reward per sample, treating each image or video as a holistic entity and ignoring the rich spatial and temporal structure of visual content. This coarse supervision hinders the correction of localized artifacts and the modeling of fine-grained perceptual cues. We introduce Visual Preference Policy Optimization (ViPO), a GRPO variant that lifts scalar feedback into structured, pixel-level advantages. ViPO employs a Perceptual Structuring Module that uses pretrained vision backbones to construct spatially and temporally aware advantage maps, redistributing optimization pressure toward perceptually important regions while preserving the stability of standard GRPO. Across both image and video benchmarks, ViPO consistently outperforms vanilla GRPO, improving in-domain alignment with human-preference rewards and enhancing generalization on out-of-domain evaluations. The method is architecture-agnostic, lightweight, and fully compatible with existing GRPO training pipelines, providing a more expressive and informative learning signal for visual generation.
>
---
#### [replaced 062] Improving Motion in Image-to-Video Models via Adaptive Low-Pass Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08456v2](https://arxiv.org/pdf/2506.08456v2)**

> **作者:** June Suk Choi; Kyungmin Lee; Sihyun Yu; Yisol Choi; Jinwoo Shin; Kimin Lee
>
> **备注:** Project page: http://choi403.github.io/ALG
>
> **摘要:** Recent text-to-video (T2V) models have demonstrated strong capabilities in producing high-quality, dynamic videos. To improve the visual controllability, recent works have considered fine-tuning pre-trained T2V models to support image-to-video (I2V) generation. However, such adaptation frequently suppresses motion dynamics of generated outputs, resulting in more static videos compared to their T2V counterparts. In this work, we analyze this phenomenon and identify that it stems from the premature exposure to high-frequency details in the input image, which biases the sampling process toward a shortcut trajectory that overfits to the static appearance of the reference image. To address this, we propose adaptive low-pass guidance (ALG), a simple training-free fix to the I2V model sampling procedure to generate more dynamic videos without compromising per-frame image quality. Specifically, ALG adaptively modulates the frequency content of the conditioning image by applying a low-pass filter at the early stage of denoising. Extensive experiments show ALG significantly improves the temporal dynamics of generated videos, while preserving or even improving image fidelity and text alignment. For instance, on the VBench test suite, ALG achieves a 33% average improvement across models in dynamic degree while maintaining the original video quality. For additional visualizations and source code, see the project page.
>
---
#### [replaced 063] UI-Venus-1.5 Technical Report
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出UI-Venus-1.5，一个统一的GUI代理模型，用于自动化数字环境交互。解决GUI自动化中泛化能力和任务性能不足的问题，通过改进训练策略和模型结构，提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2602.09082v2](https://arxiv.org/pdf/2602.09082v2)**

> **作者:** Venus Team; Changlong Gao; Zhangxuan Gu; Yulin Liu; Xinyu Qiu; Shuheng Shen; Yue Wen; Tianyu Xia; Zhenyu Xu; Zhengwen Zeng; Beitong Zhou; Xingran Zhou; Weizhi Chen; Sunhao Dai; Jingya Dou; Yichen Gong; Yuan Guo; Zhenlin Guo; Feng Li; Qian Li; Jinzhen Lin; Yuqi Zhou; Linchao Zhu; Liang Chen; Zhenyu Guo; Changhua Meng; Weiqiang Wang
>
> **摘要:** GUI agents have emerged as a powerful paradigm for automating interactions in digital environments, yet achieving both broad generality and consistently strong task performance remains challenging. In this report, we present UI-Venus-1.5, a unified, end-to-end GUI Agent designed for robust real-world applications. The proposed model family comprises two dense variants (2B and 8B) and one mixture-of-experts variant (30B-A3B) to meet various downstream application scenarios. Compared to our previous version, UI-Venus-1.5 introduces three key technical advances: (1) a comprehensive Mid-Training stage leveraging 10 billion tokens across 30+ datasets to establish foundational GUI semantics; (2) Online Reinforcement Learning with full-trajectory rollouts, aligning training objectives with long-horizon, dynamic navigation in large-scale environments; and (3) a single unified GUI Agent constructed via Model Merging, which synthesizes domain-specific models (grounding, web, and mobile) into one cohesive checkpoint. Extensive evaluations demonstrate that UI-Venus-1.5 establishes new state-of-the-art performance on benchmarks such as ScreenSpot-Pro (69.6%), VenusBench-GD (75.0%), and AndroidWorld (77.6%), significantly outperforming previous strong baselines. In addition, UI-Venus-1.5 demonstrates robust navigation capabilities across a variety of Chinese mobile apps, effectively executing user instructions in real-world scenarios. Code: https://github.com/inclusionAI/UI-Venus; Model: https://huggingface.co/collections/inclusionAI/ui-venus
>
---
#### [replaced 064] EAGLE: Expert-Augmented Attention Guidance for Tuning-Free Industrial Anomaly Detection in Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17419v2](https://arxiv.org/pdf/2602.17419v2)**

> **作者:** Xiaomeng Peng; Xilang Huang; Seon Han Choi
>
> **摘要:** Industrial anomaly detection is important for smart manufacturing, but many deep learning approaches produce only binary decisions and provide limited semantic explanations. Multimodal large language models (MLLMs) can potentially generate fine-grained, language-based analyses, yet existing methods often require costly fine-tuning and do not consistently improve anomaly detection accuracy compared to lightweight specialist detectors. We propose expert-augmented attention guidance for industrial anomaly detection in MLLMs (EAGLE), a tuning-free framework that integrates outputs from expert model to guide MLLMs toward both accurate detection and interpretable anomaly descriptions. We further study how EAGLE affects MLLMs internals by examining the attention distribution of MLLMs to the anomalous image regions in the intermediate layers. We observe that successful anomaly detection is associated with increased attention concentration on anomalous regions, and EAGLE tends to encourage this alignment. Experiments on MVTec-AD and VisA show that EAGLE improves anomaly detection performance across multiple MLLMs without any parameter updates, achieving results comparable to fine-tuning based methods. Code is available at
>
---
#### [replaced 065] Generating metamers of human scene understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.11675v3](https://arxiv.org/pdf/2601.11675v3)**

> **作者:** Ritik Raina; Abe Leite; Alexandros Graikos; Seoyoung Ahn; Dimitris Samaras; Gregory J. Zelinsky
>
> **摘要:** Human vision combines low-resolution "gist" information from the visual periphery with sparse but high-resolution information from fixated locations to construct a coherent understanding of a visual scene. In this paper, we introduce MetamerGen, a tool for generating scenes that are aligned with latent human scene representations. MetamerGen is a latent diffusion model that combines peripherally obtained scene gist information with information obtained from scene-viewing fixations to generate image metamers for what humans understand after viewing a scene. Generating images from both high and low resolution (i.e. "foveated") inputs constitutes a novel image-to-image synthesis problem, which we tackle by introducing a dual-stream representation of the foveated scenes consisting of DINOv2 tokens that fuse detailed features from fixated areas with peripherally degraded features capturing scene context. To evaluate the perceptual alignment of MetamerGen generated images to latent human scene representations, we conducted a same-different behavioral experiment where participants were asked for a "same" or "different" response between the generated and the original image. With that, we identify scene generations that are indeed metamers for the latent scene representations formed by the viewers. MetamerGen is a powerful tool for understanding scene understanding. Our proof-of-concept analyses uncovered specific features at multiple levels of visual processing that contributed to human judgments. While it can generate metamers even conditioned on random fixations, we find that high-level semantic alignment most strongly predicts metamerism when the generated scenes are conditioned on viewers' own fixated regions.
>
---
#### [replaced 066] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12370v3](https://arxiv.org/pdf/2511.12370v3)**

> **作者:** Chamuditha Jayanga Galappaththige; Jason Lai; Lloyd Windrim; Donald Dansereau; Niko Sünderhauf; Dimity Miller
>
> **备注:** Accepted at CVPR 2026. Project Page: https://chumsy0725.github.io/O-SCD/
>
> **摘要:** Online Scene Change Detection (SCD) is an extremely challenging problem that requires an agent to detect relevant changes on the fly while observing the scene from unconstrained viewpoints. Existing online SCD methods are significantly less accurate than offline approaches. We present the first online SCD approach that is pose-agnostic, label-free, and ensures multi-view consistency, while operating at over 10 FPS and achieving new state-of-the-art performance, surpassing even the best offline approaches. Our method introduces a new self-supervised fusion loss to infer scene changes from multiple cues and observations, PnP-based fast pose estimation against the reference scene, and a fast change-guided update strategy for the 3D Gaussian Splatting scene representation. Extensive experiments on complex real-world datasets demonstrate that our approach outperforms both online and offline baselines.
>
---
#### [replaced 067] CER-HV: A Human-in-the-Loop Framework for Cleaning Datasets Applied to Arabic-Script HTR
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16713v3](https://arxiv.org/pdf/2601.16713v3)**

> **作者:** Sana Al-azzawi; Elisa Barney; Marcus Liwicki
>
> **摘要:** Handwritten text recognition (HTR) for Arabic-script languages still lags behind Latin-script HTR, despite recent advances in model architectures, datasets, and benchmarks. We show that data quality is a significant limiting factor in many published datasets and propose CER-HV (CER-based Ranking with Human Verification) as a framework to detect and clean label errors. CER-HV combines a CER-based noise detector, built on a carefully configured Convolutional Recurrent Neural Network (CRNN) with early stopping to avoid overfitting noisy samples, and a human-in-the-loop (HITL) step that verifies high-ranking samples. The framework reveals that several existing datasets contain previously underreported problems, including transcription, segmentation, orientation, and non-text content errors. These have been identified with up to 90 percent precision in the Muharaf and 80-86 percent in the PHTI datasets. We also show that our CRNN achieves state-of-the-art performance across five of the six evaluated datasets, reaching 8.45 percent Character Error Rate (CER) on KHATT (Arabic), 8.26 percent on PHTI (Pashto), 10.66 percent on Ajami, and 10.11 percent on Muharaf (Arabic), all without any data cleaning. We establish a new baseline of 11.3 percent CER on the PHTD (Persian) dataset. Applying CER-HV improves the evaluation CER by 0.3-0.6 percent on the cleaner datasets and 1.0-1.8 percent on the noisier ones. Although our experiments focus on documents written in an Arabic-script language, including Arabic, Persian, Urdu, Ajami, and Pashto, the framework is general and can be applied to other text recognition datasets.
>
---
#### [replaced 068] Light of Normals: Unified Feature Representation for Universal Photometric Stereo
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.18882v5](https://arxiv.org/pdf/2506.18882v5)**

> **作者:** Houyuan Chen; Hong Li; Chongjie Ye; Zhaoxi Chen; Bohan Li; Shaocong Xu; Xianda Guo; Xuhui Liu; Yikai Wang; Baochang Zhang; Satoshi Ikehata; Boxin Shi; Anyi Rao; Hao Zhao
>
> **备注:** Home: https://houyuanchen111.github.io/lino.github.io Github: https://github.com/houyuanchen111/LINO_UniPS HuggingFace
>
> **摘要:** Universal photometric stereo (PS) is defined by two factors: it must (i) operate under arbitrary, unknown lighting conditions and (ii) avoid reliance on specific illumination models. Despite progress (e.g., SDM UniPS), two challenges remain. First, current encoders cannot guarantee that illumination and normal information are decoupled. To enforce decoupling, we introduce LINO UniPS with two key components: (i) Light Register Tokens with light alignment supervision to aggregate point, direction, and environment lights; (ii) Interleaved Attention Block featuring global cross-image attention that takes all lighting conditions together so the encoder can factor out lighting while retaining normal-related evidence. Second, high-frequency geometric details are easily lost. We address this with (i) a Wavelet-based Dual-branch Architecture and (ii) a Normal-gradient Perception Loss. These techniques yield a unified feature space in which lighting is explicitly represented by register tokens, while normal details are preserved via wavelet branch. We further introduce PS-Verse, a large-scale synthetic dataset graded by geometric complexity and lighting diversity, and adopt curriculum training from simple to complex scenes. Extensive experiments show new state-of-the-art results on public benchmarks (e.g., DiLiGenT, Luces), stronger generalization to real materials, and improved efficiency; ablations confirm that Light Register Tokens + Interleaved Attention Block drive better feature decoupling, while Wavelet-based Dual-branch Architecture + Normal-gradient Perception Loss recover finer details.
>
---
#### [replaced 069] A Very Big Video Reasoning Suite
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 该论文属于视频推理任务，旨在解决视频模型推理能力不足的问题。提出VBVR数据集和评估框架，推动视频推理研究的发展。**

- **链接: [https://arxiv.org/pdf/2602.20159v2](https://arxiv.org/pdf/2602.20159v2)**

> **作者:** Maijunxian Wang; Ruisi Wang; Juyi Lin; Ran Ji; Thaddäus Wiedemer; Qingying Gao; Dezhi Luo; Yaoyao Qian; Lianyu Huang; Zelong Hong; Jiahui Ge; Qianli Ma; Hang He; Yifan Zhou; Lingzi Guo; Lantao Mei; Jiachen Li; Hanwen Xing; Tianqi Zhao; Fengyuan Yu; Weihang Xiao; Yizheng Jiao; Jianheng Hou; Danyang Zhang; Pengcheng Xu; Boyang Zhong; Zehong Zhao; Gaoyun Fang; John Kitaoka; Yile Xu; Hua Xu; Kenton Blacutt; Tin Nguyen; Siyuan Song; Haoran Sun; Shaoyue Wen; Linyang He; Runming Wang; Yanzhi Wang; Mengyue Yang; Ziqiao Ma; Raphaël Millière; Freda Shi; Nuno Vasconcelos; Daniel Khashabi; Alan Yuille; Yilun Du; Ziming Liu; Bo Li; Dahua Lin; Ziwei Liu; Vikash Kumar; Yijiang Li; Lei Yang; Zhongang Cai; Hokin Deng
>
> **备注:** Homepage: https://video-reason.com/
>
> **摘要:** Rapid progress in video models has largely focused on visual quality, leaving their reasoning capabilities underexplored. Video reasoning grounds intelligence in spatiotemporally consistent visual environments that go beyond what text can naturally capture, enabling intuitive reasoning over spatiotemporal structure such as continuity, interaction, and causality. However, systematically studying video reasoning and its scaling behavior is hindered by the lack of large-scale training data. To address this gap, we introduce the Very Big Video Reasoning (VBVR) Dataset, an unprecedentedly large-scale resource spanning 200 curated reasoning tasks following a principled taxonomy and over one million video clips, approximately three orders of magnitude larger than existing datasets. We further present VBVR-Bench, a verifiable evaluation framework that moves beyond model-based judging by incorporating rule-based, human-aligned scorers, enabling reproducible and interpretable diagnosis of video reasoning capabilities. Leveraging the VBVR suite, we conduct one of the first large-scale scaling studies of video reasoning and observe early signs of emergent generalization to unseen reasoning tasks. Together, VBVR lays a foundation for the next stage of research in generalizable video reasoning. The data, benchmark toolkit, and models are publicly available at https://video-reason.com/ .
>
---
#### [replaced 070] Affinity Contrastive Learning for Skeleton-based Human Activity Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16694v2](https://arxiv.org/pdf/2601.16694v2)**

> **作者:** Hongda Liu; Yunfan Liu; Min Ren; Lin Sui; Yunlong Wang; Zhenan Sun
>
> **备注:** Accepted by TBIOM
>
> **摘要:** In skeleton-based human activity understanding, existing methods often adopt the contrastive learning paradigm to construct a discriminative feature space. However, many of these approaches fail to exploit the structural inter-class similarities and overlook the impact of anomalous positive samples. In this study, we introduce ACLNet, an Affinity Contrastive Learning Network that explores the intricate clustering relationships among human activity classes to improve feature discrimination. Specifically, we propose an affinity metric to refine similarity measurements, thereby forming activity superclasses that provide more informative contrastive signals. A dynamic temperature schedule is also introduced to adaptively adjust the penalty strength for various superclasses. In addition, we employ a margin-based contrastive strategy to improve the separation of hard positive and negative samples within classes. Extensive experiments on NTU RGB+D 60, NTU RGB+D 120, Kinetics-Skeleton, PKU-MMD, FineGYM, and CASIA-B demonstrate the superiority of our method in skeleton-based action recognition, gait recognition, and person re-identification. The source code is available at https://github.com/firework8/ACLNet.
>
---
#### [replaced 071] Coherent and Multi-modality Image Inpainting via Latent Space Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.08019v2](https://arxiv.org/pdf/2407.08019v2)**

> **作者:** Lingzhi Pan; Tong Zhang; Bingyuan Chen; Qi Zhou; Wei Ke; Sabine Süsstrunk; Mathieu Salzmann
>
> **摘要:** With the advancements in denoising diffusion probabilistic models (DDPMs), image inpainting has significantly evolved from merely filling information based on nearby regions to generating content conditioned on various prompts such as text, exemplar images, and sketches. However, existing methods, such as model fine-tuning and simple concatenation of latent vectors, often result in generation failures due to overfitting and inconsistency between the inpainted region and the background. In this paper, we argue that the current large diffusion models are sufficiently powerful to generate realistic images without further tuning. Hence, we introduce PILOT (in\textbf{P}ainting v\textbf{I}a \textbf{L}atent \textbf{O}p\textbf{T}imization), an optimization approach grounded on a novel \textit{semantic centralization} and \textit{background preservation loss}. Our method searches latent spaces capable of generating inpainted regions that exhibit high fidelity to user-provided prompts while maintaining coherence with the background. Furthermore, we propose a strategy to balance optimization expense and image quality, significantly enhancing generation efficiency. Our method seamlessly integrates with any pre-trained model, including ControlNet and DreamBooth, making it suitable for deployment in multi-modal editing tools. Our qualitative and quantitative evaluations demonstrate that PILOT outperforms existing approaches by generating more coherent, diverse, and faithful inpainted regions in response to provided prompts.
>
---
#### [replaced 072] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.25774v3](https://arxiv.org/pdf/2509.25774v3)**

> **作者:** Jeongjae Lee; Jong Chul Ye
>
> **备注:** 35 pages, 20 figures. ICLR 2026
>
> **摘要:** While reinforcement learning has advanced the alignment of text-to-image (T2I) models, state-of-the-art policy gradient methods are still hampered by training instability and high variance, hindering convergence speed and compromising image quality. Our analysis identifies a key cause of this instability: disproportionate credit assignment, in which the mathematical structure of the generative sampler produces volatile and non-proportional feedback across timesteps. To address this, we introduce Proportionate Credit Policy Optimization (PCPO), a framework that enforces proportional credit assignment through a stable objective reformulation and a principled reweighting of timesteps. This correction stabilizes the training process, leading to significantly accelerated convergence and superior image quality. The improvement in quality is a direct result of mitigating model collapse, a common failure mode in recursive training. PCPO substantially outperforms existing policy gradient baselines on all fronts, including the state-of-the-art DanceGRPO. Code is available at https://github.com/jaylee2000/pcpo/.
>
---
#### [replaced 073] TraceVision: Trajectory-Aware Vision-Language Model for Human-Like Spatial Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19768v2](https://arxiv.org/pdf/2602.19768v2)**

> **作者:** Fan Yang; Shurong Zheng; Hongyin Zhao; Yufei Zhan; Xin Li; Yousong Zhu; Chaoyang Zhao Ming Tang; Jinqiao Wang
>
> **摘要:** Recent Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities in image understanding and natural language generation. However, current approaches focus predominantly on global image understanding, struggling to simulate human visual attention trajectories and explain associations between descriptions and specific regions. We propose TraceVision, a unified vision-language model integrating trajectory-aware spatial understanding in an end-to-end framework. TraceVision employs a Trajectory-aware Visual Perception (TVP) module for bidirectional fusion of visual features and trajectory information. We design geometric simplification to extract semantic keypoints from raw trajectories and propose a three-stage training pipeline where trajectories guide description generation and region localization. We extend TraceVision to trajectory-guided segmentation and video scene understanding, enabling cross-frame tracking and temporal attention analysis. We construct the Reasoning-based Interactive Localized Narratives (RILN) dataset to enhance logical reasoning and interpretability. Extensive experiments on trajectory-guided captioning, text-guided trajectory prediction, understanding, and segmentation demonstrate that TraceVision achieves state-of-the-art performance, establishing a foundation for intuitive spatial interaction and interpretable visual understanding.
>
---
#### [replaced 074] XtraLight-MedMamba for Classification of Neoplastic Tubular Adenomas
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.04819v2](https://arxiv.org/pdf/2602.04819v2)**

> **作者:** Aqsa Sultana; Rayan Afsar; Ahmed Rahu; Surendra P. Singh; Brian Shula; Brandon Combs; Derrick Forchetti; Vijayan K. Asari
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Accurate risk stratification of precancerous polyps during routine colonoscopy screenings is essential for lowering the risk of developing colorectal cancer (CRC). However, assessment of low-grade dysplasia remains limited by subjective histopathologic interpretation. Advancements in digital pathology and deep learning provide new opportunities to identify subtle and fine morphologic patterns associated with malignant progression that may be imperceptible to the human eye. In this work, we propose XtraLight-MedMamba, an ultra-lightweight state-space-based deep learning framework for classifying neoplastic tubular adenomas from whole-slide images (WSIs). The architecture is a blend of ConvNext based shallow feature extractor with parallel vision mamba to efficiently model both long- and short-range dependencies and image generalization. An integration of Spatial and Channel Attention Bridge (SCAB) module enhances multiscale feature extraction, while Fixed Non-Negative Orthogonal Classifier (FNOClassifier) enables substantial parameter reduction and improved generalization. The model was evaluated on a curated dataset acquired from patients with low-grade tubular adenomas, stratified into case and control cohorts based on subsequent CRC development. XtraLight-MedMamba achieved an accuracy of 97.18% and an F1-score of 0.9767 using approximately 32,000 parameters, outperforming transformer-based and conventional Mamba architectures with significantly higher model complexity.
>
---
#### [replaced 075] When Pretty Isn't Useful: Investigating Why Modern Text-to-Image Models Fail as Reliable Training Data Generators
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.19946v2](https://arxiv.org/pdf/2602.19946v2)**

> **作者:** Krzysztof Adamkiewicz; Brian Moser; Stanislav Frolov; Tobias Christian Nauen; Federico Raue; Andreas Dengel
>
> **摘要:** Recent text-to-image (T2I) diffusion models produce visually stunning images and demonstrate excellent prompt following. But do they perform well as synthetic vision data generators? In this work, we revisit the promise of synthetic data as a scalable substitute for real training sets and uncover a surprising performance regression. We generate large-scale synthetic datasets using state-of-the-art T2I models released between 2022 and 2025, train standard classifiers solely on this synthetic data, and evaluate them on real test data. Despite observable advances in visual fidelity and prompt adherence, classification accuracy on real test data consistently declines with newer T2I models as training data generators. Our analysis reveals a hidden trend: These models collapse to a narrow, aesthetic-centric distribution that undermines diversity and label-image alignment. Overall, our findings challenge a growing assumption in vision research, namely that progress in generative realism implies progress in data realism. We thus highlight an urgent need to rethink the capabilities of modern T2I models as reliable training data generators.
>
---
#### [replaced 076] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.14856v2](https://arxiv.org/pdf/2506.14856v2)**

> **作者:** Zhengquan Zhang; Feng Xu; Mengmi Zhang
>
> **备注:** 10 pages, 4 figures in the main text. Published at ICLR 2026
>
> **摘要:** Some perspectives naturally provide more information than others. How can an AI system determine which viewpoint offers the most valuable insight for accurate and efficient 3D object reconstruction? Active view selection (AVS) for 3D reconstruction remains a fundamental challenge in computer vision. The aim is to identify the minimal set of views that yields the most accurate 3D reconstruction. Instead of learning radiance fields, like NeRF or 3D Gaussian Splatting, from a current observation and computing uncertainty for each candidate viewpoint, we introduce a novel AVS approach guided by neural uncertainty maps predicted by a lightweight feedforward deep neural network, named UPNet. UPNet takes a single input image of a 3D object and outputs a predicted uncertainty map, representing uncertainty values across all possible candidate viewpoints. By leveraging heuristics derived from observing many natural objects and their associated uncertainty patterns, we train UPNet to learn a direct mapping from viewpoint appearance to uncertainty in the underlying volumetric representations. Next, our approach aggregates all previously predicted neural uncertainty maps to suppress redundant candidate viewpoints and effectively select the most informative one. Using these selected viewpoints, we train 3D neural rendering models and evaluate the quality of novel view synthesis against other competitive AVS methods. Remarkably, despite using half of the viewpoints than the upper bound, our method achieves comparable reconstruction accuracy. In addition, it significantly reduces computational overhead during AVS, achieving up to a 400 times speedup along with over 50\% reductions in CPU, RAM, and GPU usage compared to baseline methods. Notably, our approach generalizes effectively to AVS tasks involving novel object categories, without requiring any additional training.
>
---
#### [replaced 077] A deep learning framework for efficient pathology image analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.13027v2](https://arxiv.org/pdf/2502.13027v2)**

> **作者:** Peter Neidlinger; Tim Lenz; Sebastian Foersch; Chiara M. L. Loeffler; Jan Clusmann; Marco Gustav; Lawrence A. Shaktah; Rupert Langer; Bastian Dislich; Lisa A. Boardman; Amy J. French; Ellen L. Goode; Andrea Gsur; Stefanie Brezina; Marc J. Gunter; Robert Steinfelder; Hans-Michael Behrens; Christoph Röcken; Tabitha Harrison; Ulrike Peters; Amanda I. Phipps; Giuseppe Curigliano; Nicola Fusco; Antonio Marra; Michael Hoffmeister; Hermann Brenner; Jakob Nikolas Kather
>
> **摘要:** Artificial intelligence (AI) has transformed digital pathology by enabling biomarker prediction from high-resolution whole-slide images (WSIs). However, current methods are computationally inefficient, processing thousands of redundant tiles per WSI and requiring complex aggregator models. We introduce EAGLE (Efficient Approach for Guided Local Examination), a deep learning framework that emulates pathologists by selectively analyzing informative regions. EAGLE incorporates two foundation models: CHIEF for efficient tile selection and Virchow2 for extracting high-quality features. Benchmarking was conducted against leading slide- and tile-level foundation models across 43 tasks from nine cancer types, spanning morphology, biomarker prediction, treatment response and prognosis. EAGLE outperformed state-of-the-art patch aggregation methods by up to 23% and achieved the highest AUROC overall. It processed a slide in 2.27 seconds, reducing computational time by more than 99% compared to existing models. This efficiency enables real-time workflows, allows rapid review of the exact tiles used for each prediction, and reduces dependence on high-performance computing, making AI-powered pathology more accessible. By reliably identifying meaningful regions and minimizing artifacts, EAGLE provides robust and auditable outputs, supported by systematic negative controls and attention concentration analyses. Its unified embedding enables rapid slide searches, integration into multi-omics pipelines and emerging clinical foundation models.
>
---
#### [replaced 078] Less is More: Data-Efficient Adaptation for Controllable Text-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.17844v3](https://arxiv.org/pdf/2511.17844v3)**

> **作者:** Shihan Cheng; Nilesh Kulkarni; David Hyde; Dmitriy Smirnov
>
> **摘要:** Fine-tuning large-scale text-to-video diffusion models to add new generative controls, such as those over physical camera parameters (e.g., shutter speed or aperture), typically requires vast, high-fidelity datasets that are difficult to acquire. In this work, we propose a data-efficient fine-tuning strategy that learns these controls from sparse, low-quality synthetic data. We show that not only does fine-tuning on such simple data enable the desired controls, it actually yields superior results to models fine-tuned on photorealistic "real" data. Beyond demonstrating these results, we provide a framework that justifies this phenomenon both intuitively and quantitatively.
>
---
#### [replaced 079] Trajectory-aware Shifted State Space Models for Online Video Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10453v2](https://arxiv.org/pdf/2508.10453v2)**

> **作者:** Qiang Zhu; Xiandong Meng; Yuxian Jiang; Fan Zhang; David Bull; Shuyuan Zhu; Bing Zeng; Ronggang Wang
>
> **备注:** ICLR2026
>
> **摘要:** Online video super-resolution (VSR) is an important technique for many real-world video processing applications, which aims to restore the current high-resolution video frame based on temporally previous frames. Most of the existing online VSR methods solely employ one neighboring previous frame to achieve temporal alignment, which limits long-range temporal modeling of videos. Recently, state space models (SSMs) have been proposed with linear computational complexity and a global receptive field, which significantly improve computational efficiency and performance. In this context, this paper presents a novel online VSR method based on Trajectory-aware Shifted SSMs (TS-Mamba), leveraging both long-term trajectory modeling and low-complexity Mamba to achieve efficient spatio-temporal information aggregation. Specifically, TS-Mamba first constructs the trajectories within a video to select the most similar tokens from the previous frames. Then, a Trajectory-aware Shifted Mamba Aggregation (TSMA) module consisting of proposed shifted SSMs blocks is employed to aggregate the selected tokens. The shifted SSMs blocks are designed based on Hilbert scannings and corresponding shift operations to compensate for scanning losses and strengthen the spatial continuity of Mamba. Additionally, we propose a trajectory-aware loss function to supervise the trajectory generation, ensuring the accuracy of token selection when training our model. Extensive experiments on three widely used VSR test datasets demonstrate that compared with six online VSR benchmark models, our TS-Mamba achieves state-of-the-art performance in most cases and over 22.7% complexity reduction (in MACs).
>
---
