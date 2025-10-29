# 计算机视觉 cs.CV

- **最新发布 89 篇**

- **更新 80 篇**

## 最新发布

#### [new 001] TeleEgo: Benchmarking Egocentric AI Assistants in the Wild
- **分类: cs.CV**

- **简介: 该论文提出TeleEgo，一个面向真实场景的长时序、多模态AI助手评测基准。针对现有评估缺乏真实流式场景与长期记忆测试的问题，构建了含14小时同步视频、音频、文本数据的跨领域数据集，设计12项诊断任务与实时准确率、记忆持续时间等指标，全面评估记忆、理解与跨记忆推理能力。**

- **链接: [http://arxiv.org/pdf/2510.23981v1](http://arxiv.org/pdf/2510.23981v1)**

> **作者:** Jiaqi Yan; Ruilong Ren; Jingren Liu; Shuning Xu; Ling Wang; Yiheng Wang; Yun Wang; Long Zhang; Xiangyu Chen; Changzhi Sun; Jixiang Luo; Dell Zhang; Hao Sun; Chi Zhang; Xuelong Li
>
> **摘要:** Egocentric AI assistants in real-world settings must process multi-modal inputs (video, audio, text), respond in real time, and retain evolving long-term memory. However, existing benchmarks typically evaluate these abilities in isolation, lack realistic streaming scenarios, or support only short-term tasks. We introduce \textbf{TeleEgo}, a long-duration, streaming, omni-modal benchmark for evaluating egocentric AI assistants in realistic daily contexts. The dataset features over 14 hours per participant of synchronized egocentric video, audio, and text across four domains: work \& study, lifestyle \& routines, social activities, and outings \& culture. All data is aligned on a unified global timeline and includes high-quality visual narrations and speech transcripts, curated through human refinement.TeleEgo defines 12 diagnostic subtasks across three core capabilities: Memory (recalling past events), Understanding (interpreting the current moment), and Cross-Memory Reasoning (linking distant events). It contains 3,291 human-verified QA items spanning multiple question formats (single-choice, binary, multi-choice, and open-ended), evaluated strictly in a streaming setting. We propose two key metrics -- Real-Time Accuracy and Memory Persistence Time -- to jointly assess correctness, temporal responsiveness, and long-term retention. TeleEgo provides a realistic and comprehensive evaluation to advance the development of practical AI assistants.
>
---
#### [new 002] Physics-Inspired Gaussian Kolmogorov-Arnold Networks for X-ray Scatter Correction in Cone-Beam CT
- **分类: cs.CV; I.4.5; I.5**

- **简介: 该论文针对锥束CT成像中的散射伪影问题，提出一种基于物理先验的高斯柯尔莫戈洛夫-阿诺德网络方法。利用旋转对称性建模散射分布，结合高斯RBF与KAN网络，高效学习高维散射特征，显著提升图像重建质量，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24579v1](http://arxiv.org/pdf/2510.24579v1)**

> **作者:** Xu Jiang; Huiying Pan; Ligen Shi; Jianing Sun; Wenfeng Xu; Xing Zhao
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Cone-beam CT (CBCT) employs a flat-panel detector to achieve three-dimensional imaging with high spatial resolution. However, CBCT is susceptible to scatter during data acquisition, which introduces CT value bias and reduced tissue contrast in the reconstructed images, ultimately degrading diagnostic accuracy. To address this issue, we propose a deep learning-based scatter artifact correction method inspired by physical prior knowledge. Leveraging the fact that the observed point scatter probability density distribution exhibits rotational symmetry in the projection domain. The method uses Gaussian Radial Basis Functions (RBF) to model the point scatter function and embeds it into the Kolmogorov-Arnold Networks (KAN) layer, which provides efficient nonlinear mapping capabilities for learning high-dimensional scatter features. By incorporating the physical characteristics of the scattered photon distribution together with the complex function mapping capacity of KAN, the model improves its ability to accurately represent scatter. The effectiveness of the method is validated through both synthetic and real-scan experiments. Experimental results show that the model can effectively correct the scatter artifacts in the reconstructed images and is superior to the current methods in terms of quantitative metrics.
>
---
#### [new 003] Uniform Discrete Diffusion with Metric Path for Video Generation
- **分类: cs.CV**

- **简介: 该论文提出URSA框架，解决离散视频生成中误差累积与长序列不一致问题。通过线性度量路径和分辨率自适应时间步调整，实现高效高分辨率视频生成，并支持多种任务统一建模，性能媲美连续扩散方法。**

- **链接: [http://arxiv.org/pdf/2510.24717v1](http://arxiv.org/pdf/2510.24717v1)**

> **作者:** Haoge Deng; Ting Pan; Fan Zhang; Yang Liu; Zhuoyan Luo; Yufeng Cui; Wenxuan Wang; Chunhua Shen; Shiguang Shan; Zhaoxiang Zhang; Xinlong Wang
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Continuous-space video generation has advanced rapidly, while discrete approaches lag behind due to error accumulation and long-context inconsistency. In this work, we revisit discrete generative modeling and present Uniform discRete diffuSion with metric pAth (URSA), a simple yet powerful framework that bridges the gap with continuous approaches for the scalable video generation. At its core, URSA formulates the video generation task as an iterative global refinement of discrete spatiotemporal tokens. It integrates two key designs: a Linearized Metric Path and a Resolution-dependent Timestep Shifting mechanism. These designs enable URSA to scale efficiently to high-resolution image synthesis and long-duration video generation, while requiring significantly fewer inference steps. Additionally, we introduce an asynchronous temporal fine-tuning strategy that unifies versatile tasks within a single model, including interpolation and image-to-video generation. Extensive experiments on challenging video and image generation benchmarks demonstrate that URSA consistently outperforms existing discrete methods and achieves performance comparable to state-of-the-art continuous diffusion methods. Code and models are available at https://github.com/baaivision/URSA
>
---
#### [new 004] Stroke Lesion Segmentation in Clinical Workflows: A Modular, Lightweight, and Deployment-Ready Tool
- **分类: cs.CV**

- **简介: 该论文针对脑卒中病灶分割的临床部署难题，提出轻量级模块化工具StrokeSeg。通过解耦预处理、推理与后处理，结合ONNX Runtime与浮点16量化，实现模型尺寸减半，支持图形与命令行界面，可在临床工作流中便捷使用，性能与原研究模型相当。**

- **链接: [http://arxiv.org/pdf/2510.24378v1](http://arxiv.org/pdf/2510.24378v1)**

> **作者:** Yann Kerverdo; Florent Leray; Youwan Mahé; Stéphanie Leplaideur; Francesca Galassi
>
> **摘要:** Deep learning frameworks such as nnU-Net achieve state-of-the-art performance in brain lesion segmentation but remain difficult to deploy clinically due to heavy dependencies and monolithic design. We introduce \textit{StrokeSeg}, a modular and lightweight framework that translates research-grade stroke lesion segmentation models into deployable applications. Preprocessing, inference, and postprocessing are decoupled: preprocessing relies on the Anima toolbox with BIDS-compliant outputs, and inference uses ONNX Runtime with \texttt{Float16} quantisation, reducing model size by about 50\%. \textit{StrokeSeg} provides both graphical and command-line interfaces and is distributed as Python scripts and as a standalone Windows executable. On a held-out set of 300 sub-acute and chronic stroke subjects, segmentation performance was equivalent to the original PyTorch pipeline (Dice difference $<10^{-3}$), demonstrating that high-performing research pipelines can be transformed into portable, clinically usable tools.
>
---
#### [new 005] TRELLISWorld: Training-Free World Generation from Object Generators
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出TRELLISWorld，一种无需训练的3D场景生成方法。针对现有方法受限于单物体、需训练或缺乏全景视图的问题，利用文本到3D物体扩散模型作为模块化瓦片生成器，通过重叠区域加权融合实现大规模、语义一致的场景合成，支持灵活编辑与高效生成。**

- **链接: [http://arxiv.org/pdf/2510.23880v1](http://arxiv.org/pdf/2510.23880v1)**

> **作者:** Hanke Chen; Yuan Liu; Minchen Li
>
> **摘要:** Text-driven 3D scene generation holds promise for a wide range of applications, from virtual prototyping to AR/VR and simulation. However, existing methods are often constrained to single-object generation, require domain-specific training, or lack support for full 360-degree viewability. In this work, we present a training-free approach to 3D scene synthesis by repurposing general-purpose text-to-3D object diffusion models as modular tile generators. We reformulate scene generation as a multi-tile denoising problem, where overlapping 3D regions are independently generated and seamlessly blended via weighted averaging. This enables scalable synthesis of large, coherent scenes while preserving local semantic control. Our method eliminates the need for scene-level datasets or retraining, relies on minimal heuristics, and inherits the generalization capabilities of object-level priors. We demonstrate that our approach supports diverse scene layouts, efficient generation, and flexible editing, establishing a simple yet powerful foundation for general-purpose, language-driven 3D scene construction.
>
---
#### [new 006] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Mars-Bench，首个面向火星科学任务的基准测试，解决缺乏标准化评估框架的问题。针对火星地质特征识别任务，构建20个涵盖分类、分割、检测的多源数据集，提供基线模型与评估标准，推动火星专用基础模型发展。**

- **链接: [http://arxiv.org/pdf/2510.24010v1](http://arxiv.org/pdf/2510.24010v1)**

> **作者:** Mirali Purohit; Bimal Gajera; Vatsal Malaviya; Irish Mehta; Kunal Kasodekar; Jacob Adler; Steven Lu; Umaa Rebbapragada; Hannah Kerner
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Foundation models have enabled rapid progress across many specialized domains by leveraging large-scale pre-training on unlabeled data, demonstrating strong generalization to a variety of downstream tasks. While such models have gained significant attention in fields like Earth Observation, their application to Mars science remains limited. A key enabler of progress in other domains has been the availability of standardized benchmarks that support systematic evaluation. In contrast, Mars science lacks such benchmarks and standardized evaluation frameworks, which have limited progress toward developing foundation models for Martian tasks. To address this gap, we introduce Mars-Bench, the first benchmark designed to systematically evaluate models across a broad range of Mars-related tasks using both orbital and surface imagery. Mars-Bench comprises 20 datasets spanning classification, segmentation, and object detection, focused on key geologic features such as craters, cones, boulders, and frost. We provide standardized, ready-to-use datasets and baseline evaluations using models pre-trained on natural images, Earth satellite data, and state-of-the-art vision-language models. Results from all analyses suggest that Mars-specific foundation models may offer advantages over general-domain counterparts, motivating further exploration of domain-adapted pre-training. Mars-Bench aims to establish a standardized foundation for developing and comparing machine learning models for Mars science. Our data, models, and code are available at: https://mars-bench.github.io/.
>
---
#### [new 007] DynaStride: Dynamic Stride Windowing with MMCoT for Instructional Multi-Scene Captioning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出DynaStride，用于指令视频的场景级字幕生成任务。针对现有方法难以捕捉时间结构导致字幕不连贯的问题，通过动态步长窗口与多模态思维链，自适应采样并融合视觉与文本信息，生成更具时序一致性和信息量的综合字幕。**

- **链接: [http://arxiv.org/pdf/2510.23907v1](http://arxiv.org/pdf/2510.23907v1)**

> **作者:** Eddison Pham; Prisha Priyadarshini; Adrian Maliackel; Kanishk Bandi; Cristian Meo; Kevin Zhu
>
> **备注:** 16 pages, 15 figures, 5 Tables, submitted to AAAI AI4ED Workshop 2026
>
> **摘要:** Scene-level captioning in instructional videos can enhance learning by requiring an understanding of both visual cues and temporal structure. By aligning visual cues with textual guidance, this understanding supports procedural learning and multimodal reasoning, providing a richer context for skill acquisition. However, captions that fail to capture this structure may lack coherence and quality, which can create confusion and undermine the video's educational intent. To address this gap, we introduce DynaStride, a pipeline to generate coherent, scene-level captions without requiring manual scene segmentation. Using the YouCookII dataset's scene annotations, DynaStride performs adaptive frame sampling and multimodal windowing to capture key transitions within each scene. It then employs a multimodal chain-of-thought process to produce multiple action-object pairs, which are refined and fused using a dynamic stride window selection algorithm that adaptively balances temporal context and redundancy. The final scene-level caption integrates visual semantics and temporal reasoning in a single instructional caption. Empirical evaluations against strong baselines, including VLLaMA3 and GPT-4o, demonstrate consistent gains on both N-gram-based metrics (BLEU, METEOR) and semantic similarity measures (BERTScore, CLIPScore). Qualitative analyses further show that DynaStride produces captions that are more temporally coherent and informative, suggesting a promising direction for improving AI-powered instructional content generation.
>
---
#### [new 008] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodel LLMs
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型中视觉令牌冗余导致的计算开销问题，提出SCOPE方法，通过联合建模显著性与覆盖度，迭代选择最具信息量的视觉令牌，提升语义完整性。实验表明，该方法在多个基准上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24214v1](http://arxiv.org/pdf/2510.24214v1)**

> **作者:** Jinhong Deng; Wen Li; Joey Tianyi Zhou; Yang He
>
> **备注:** NeurIPS 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) typically process a large number of visual tokens, leading to considerable computational overhead, even though many of these tokens are redundant. Existing visual token pruning methods primarily focus on selecting the most salient tokens based on attention scores, resulting in the semantic incompleteness of the selected tokens. In this paper, we propose a novel visual token pruning strategy, called \textbf{S}aliency-\textbf{C}overage \textbf{O}riented token \textbf{P}runing for \textbf{E}fficient MLLMs (SCOPE), to jointly model both the saliency and coverage of the selected visual tokens to better preserve semantic completeness. Specifically, we introduce a set-coverage for a given set of selected tokens, computed based on the token relationships. We then define a token-coverage gain for each unselected token, quantifying how much additional coverage would be obtained by including it. By integrating the saliency score into the token-coverage gain, we propose our SCOPE score and iteratively select the token with the highest SCOPE score. We conduct extensive experiments on multiple vision-language understanding benchmarks using the LLaVA-1.5 and LLaVA-Next models. Experimental results demonstrate that our method consistently outperforms prior approaches. Our code is available at \href{https://github.com/kinredon/SCOPE}{https://github.com/kinredon/SCOPE}.
>
---
#### [new 009] Kineo: Calibration-Free Metric Motion Capture From Sparse RGB Cameras
- **分类: cs.CV**

- **简介: 该论文提出Kineo，一个无需标定的多人动作捕捉方法，解决多视角无标记动作捕捉中依赖精确相机标定的问题。通过2D关键点实现相机自动标定与3D重建，结合时空采样和图优化，提升精度与效率，显著降低误差并支持实时处理。**

- **链接: [http://arxiv.org/pdf/2510.24464v1](http://arxiv.org/pdf/2510.24464v1)**

> **作者:** Charles Javerliat; Pierre Raimbaud; Guillaume Lavoué
>
> **摘要:** Markerless multiview motion capture is often constrained by the need for precise camera calibration, limiting accessibility for non-experts and in-the-wild captures. Existing calibration-free approaches mitigate this requirement but suffer from high computational cost and reduced reconstruction accuracy. We present Kineo, a fully automatic, calibration-free pipeline for markerless motion capture from videos captured by unsynchronized, uncalibrated, consumer-grade RGB cameras. Kineo leverages 2D keypoints from off-the-shelf detectors to simultaneously calibrate cameras, including Brown-Conrady distortion coefficients, and reconstruct 3D keypoints and dense scene point maps at metric scale. A confidence-driven spatio-temporal keypoint sampling strategy, combined with graph-based global optimization, ensures robust calibration at a fixed computational cost independent of sequence length. We further introduce a pairwise reprojection consensus score to quantify 3D reconstruction reliability for downstream tasks. Evaluations on EgoHumans and Human3.6M demonstrate substantial improvements over prior calibration-free methods. Compared to previous state-of-the-art approaches, Kineo reduces camera translation error by approximately 83-85%, camera angular error by 86-92%, and world mean-per-joint error (W-MPJPE) by 83-91%. Kineo is also efficient in real-world scenarios, processing multi-view sequences faster than their duration in specific configuration (e.g., 36min to process 1h20min of footage). The full pipeline and evaluation code are openly released to promote reproducibility and practical adoption at https://liris-xr.github.io/kineo/.
>
---
#### [new 010] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors
- **分类: cs.CV**

- **简介: 该论文针对室内场景3D高保真重建任务，解决传统3DGS在大平面、低纹理区域因光度损失模糊导致几何失真的问题。提出PlanarGS框架，引入视觉语言引导的平面先验（LP3），结合平面一致性与几何线索优化3D高斯分布，显著提升重建精度与细节表现。**

- **链接: [http://arxiv.org/pdf/2510.23930v1](http://arxiv.org/pdf/2510.23930v1)**

> **作者:** Xirui Jin; Renbiao Jin; Boying Li; Danping Zou; Wenxian Yu
>
> **备注:** Accepted by NeurIPS 2025. Project page: https://planargs.github.io
>
> **摘要:** Three-dimensional Gaussian Splatting (3DGS) has recently emerged as an efficient representation for novel-view synthesis, achieving impressive visual quality. However, in scenes dominated by large and low-texture regions, common in indoor environments, the photometric loss used to optimize 3DGS yields ambiguous geometry and fails to recover high-fidelity 3D surfaces. To overcome this limitation, we introduce PlanarGS, a 3DGS-based framework tailored for indoor scene reconstruction. Specifically, we design a pipeline for Language-Prompted Planar Priors (LP3) that employs a pretrained vision-language segmentation model and refines its region proposals via cross-view fusion and inspection with geometric priors. 3D Gaussians in our framework are optimized with two additional terms: a planar prior supervision term that enforces planar consistency, and a geometric prior supervision term that steers the Gaussians toward the depth and normal cues. We have conducted extensive experiments on standard indoor benchmarks. The results show that PlanarGS reconstructs accurate and detailed 3D surfaces, consistently outperforming state-of-the-art methods by a large margin. Project page: https://planargs.github.io
>
---
#### [new 011] OmniText: A Training-Free Generalist for Controllable Text-Image Manipulation
- **分类: cs.CV**

- **简介: 该论文提出OmniText，一个无需训练的通用文本图像编辑框架。针对现有方法在文本移除、风格控制和重复字符生成方面的局限，通过自注意力反演与交叉注意力重分配实现精准文本编辑，并引入新型损失函数提升内容准确性和风格可控性。研究还构建了OmniText-Bench基准数据集，验证其在多种任务上的先进性能。**

- **链接: [http://arxiv.org/pdf/2510.24093v1](http://arxiv.org/pdf/2510.24093v1)**

> **作者:** Agus Gunawan; Samuel Teodoro; Yun Chen; Soo Ye Kim; Jihyong Oh; Munchurl Kim
>
> **备注:** The first two authors contributed equally to this work. The last two authors are co-corresponding authors
>
> **摘要:** Recent advancements in diffusion-based text synthesis have demonstrated significant performance in inserting and editing text within images via inpainting. However, despite the potential of text inpainting methods, three key limitations hinder their applicability to broader Text Image Manipulation (TIM) tasks: (i) the inability to remove text, (ii) the lack of control over the style of rendered text, and (iii) a tendency to generate duplicated letters. To address these challenges, we propose OmniText, a training-free generalist capable of performing a wide range of TIM tasks. Specifically, we investigate two key properties of cross- and self-attention mechanisms to enable text removal and to provide control over both text styles and content. Our findings reveal that text removal can be achieved by applying self-attention inversion, which mitigates the model's tendency to focus on surrounding text, thus reducing text hallucinations. Additionally, we redistribute cross-attention, as increasing the probability of certain text tokens reduces text hallucination. For controllable inpainting, we introduce novel loss functions in a latent optimization framework: a cross-attention content loss to improve text rendering accuracy and a self-attention style loss to facilitate style customization. Furthermore, we present OmniText-Bench, a benchmark dataset for evaluating diverse TIM tasks. It includes input images, target text with masks, and style references, covering diverse applications such as text removal, rescaling, repositioning, and insertion and editing with various styles. Our OmniText framework is the first generalist method capable of performing diverse TIM tasks. It achieves state-of-the-art performance across multiple tasks and metrics compared to other text inpainting methods and is comparable with specialist methods.
>
---
#### [new 012] A geometric and deep learning reproducible pipeline for monitoring floating anthropogenic debris in urban rivers using in situ cameras
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对城市河流中漂浮人工垃圾的监测问题，提出基于固定摄像头的几何与深度学习融合方法。通过深度学习实现垃圾的自动识别与量化，并结合相机几何参数进行真实尺寸估计，解决了复杂环境下精度与速度的平衡及数据泄露等问题，构建了可复现的自动化监测流程。**

- **链接: [http://arxiv.org/pdf/2510.23798v1](http://arxiv.org/pdf/2510.23798v1)**

> **作者:** Gauthier Grimmer; Romain Wenger; Clément Flint; Germain Forestier; Gilles Rixhon; Valentin Chardon
>
> **摘要:** The proliferation of floating anthropogenic debris in rivers has emerged as a pressing environmental concern, exerting a detrimental influence on biodiversity, water quality, and human activities such as navigation and recreation. The present study proposes a novel methodological framework for the monitoring the aforementioned waste, utilising fixed, in-situ cameras. This study provides two key contributions: (i) the continuous quantification and monitoring of floating debris using deep learning and (ii) the identification of the most suitable deep learning model in terms of accuracy and inference speed under complex environmental conditions. These models are tested in a range of environmental conditions and learning configurations, including experiments on biases related to data leakage. Furthermore, a geometric model is implemented to estimate the actual size of detected objects from a 2D image. This model takes advantage of both intrinsic and extrinsic characteristics of the camera. The findings of this study underscore the significance of the dataset constitution protocol, particularly with respect to the integration of negative images and the consideration of temporal leakage. In conclusion, the feasibility of metric object estimation using projective geometry coupled with regression corrections is demonstrated. This approach paves the way for the development of robust, low-cost, automated monitoring systems for urban aquatic environments.
>
---
#### [new 013] Explainable Detection of AI-Generated Images with Artifact Localization Using Faster-Than-Lies and Vision-Language Models for Edge Devices
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文针对低分辨率AI生成图像的真实性检测任务，提出结合轻量级分类器与视觉语言模型的可解释检测方法。通过重建误差热图定位伪造痕迹，并生成语义化解释，实现高精度（96.5%）与低延迟（175ms）检测，适用于边缘设备部署。**

- **链接: [http://arxiv.org/pdf/2510.23775v1](http://arxiv.org/pdf/2510.23775v1)**

> **作者:** Aryan Mathur; Asaduddin Ahmed; Pushti Amit Vasoya; Simeon Kandan Sonar; Yasir Z; Madesh Kuppusamy
>
> **摘要:** The increasing realism of AI-generated imagery poses challenges for verifying visual authenticity. We present an explainable image authenticity detection system that combines a lightweight convolutional classifier ("Faster-Than-Lies") with a Vision-Language Model (Qwen2-VL-7B) to classify, localize, and explain artifacts in 32x32 images. Our model achieves 96.5% accuracy on the extended CiFAKE dataset augmented with adversarial perturbations and maintains an inference time of 175ms on 8-core CPUs, enabling deployment on local or edge devices. Using autoencoder-based reconstruction error maps, we generate artifact localization heatmaps, which enhance interpretability for both humans and the VLM. We further categorize 70 visual artifact types into eight semantic groups and demonstrate explainable text generation for each detected anomaly. This work highlights the feasibility of combining visual and linguistic reasoning for interpretable authenticity detection in low-resolution imagery and outlines potential cross-domain applications in forensics, industrial inspection, and social media moderation.
>
---
#### [new 014] ViPER: Empowering the Self-Evolution of Visual Perception Abilities in Vision-Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视觉语言模型（VLM）在细粒度视觉感知能力上的局限性，提出ViPER框架。通过自举式双阶段强化学习，实现图像与实例级重建的闭环训练，促进模型自我批判与进化，显著提升感知能力，同时保持通用性。**

- **链接: [http://arxiv.org/pdf/2510.24285v1](http://arxiv.org/pdf/2510.24285v1)**

> **作者:** Juntian Zhang; Song Jin; Chuanqi Cheng; Yuhan Liu; Yankai Lin; Xun Zhang; Yufei Zhang; Fei Jiang; Guojun Yin; Wei Lin; Rui Yan
>
> **摘要:** The limited capacity for fine-grained visual perception presents a critical bottleneck for Vision-Language Models (VLMs) in real-world applications. Addressing this is challenging due to the scarcity of high-quality data and the limitations of existing methods: supervised fine-tuning (SFT) often compromises general capabilities, while reinforcement fine-tuning (RFT) prioritizes textual reasoning over visual perception. To bridge this gap, we propose a novel two-stage task that structures visual perception learning as a coarse-to-fine progressive process. Based on this task formulation, we develop ViPER, a self-bootstrapping framework specifically designed to enable iterative evolution through self-critiquing and self-prediction. By synergistically integrating image-level and instance-level reconstruction with a two-stage reinforcement learning strategy, ViPER establishes a closed-loop training paradigm, where internally synthesized data directly fuel the enhancement of perceptual ability. Applied to the Qwen2.5-VL family, ViPER produces the Qwen-Viper series. With an average gain of 1.7% on seven comprehensive benchmarks spanning various tasks and up to 6.0% on fine-grained perception, Qwen-Viper consistently demonstrates superior performance across different vision-language scenarios while maintaining generalizability. Beyond enabling self-improvement in perceptual capabilities, ViPER provides concrete evidence for the reciprocal relationship between generation and understanding, a breakthrough to developing more autonomous and capable VLMs.
>
---
#### [new 015] AdvBlur: Adversarial Blur for Robust Diabetic Retinopathy Classification and Cross-Domain Generalization
- **分类: cs.CV**

- **简介: 该论文针对糖尿病视网膜病变（DR）分类中因设备、人群和成像条件差异导致的模型泛化能力差的问题，提出AdvBlur方法。通过引入对抗性模糊图像并结合双损失函数，提升模型在跨域数据上的鲁棒性，显著改善了未知分布下的分类性能。**

- **链接: [http://arxiv.org/pdf/2510.24000v1](http://arxiv.org/pdf/2510.24000v1)**

> **作者:** Heethanjan Kanagalingam; Thenukan Pathmanathan; Mokeeshan Vathanakumar; Tharmakulasingam Mukunthan
>
> **摘要:** Diabetic retinopathy (DR) is a leading cause of vision loss worldwide, yet early and accurate detection can significantly improve treatment outcomes. While numerous Deep learning (DL) models have been developed to predict DR from fundus images, many face challenges in maintaining robustness due to distributional variations caused by differences in acquisition devices, demographic disparities, and imaging conditions. This paper addresses this critical limitation by proposing a novel DR classification approach, a method called AdvBlur. Our method integrates adversarial blurred images into the dataset and employs a dual-loss function framework to address domain generalization. This approach effectively mitigates the impact of unseen distributional variations, as evidenced by comprehensive evaluations across multiple datasets. Additionally, we conduct extensive experiments to explore the effects of factors such as camera type, low-quality images, and dataset size. Furthermore, we perform ablation studies on blurred images and the loss function to ensure the validity of our choices. The experimental results demonstrate the effectiveness of our proposed method, achieving competitive performance compared to state-of-the-art domain generalization DR models on unseen external datasets.
>
---
#### [new 016] Eye-Tracking, Mouse Tracking, Stimulus Tracking,and Decision-Making Datasets in Digital Pathology
- **分类: cs.CV; cs.HC; J.3**

- **简介: 该论文针对数字病理诊断中诊断一致性低的问题，构建了PathoGaze1.0数据集，通过眼动、鼠标轨迹等行为数据记录19名病理医生对397张全切片图像的诊断过程，旨在揭示诊断错误与不一致的成因，并为提升病理医生培训及辅助AI系统提供数据支持。**

- **链接: [http://arxiv.org/pdf/2510.24653v1](http://arxiv.org/pdf/2510.24653v1)**

> **作者:** Veronica Thai; Rui Li; Meng Ling; Shuning Jiang; Jeremy Wolfe; Raghu Machiraju; Yan Hu; Zaibo Li; Anil Parwani; Jian Chen
>
> **备注:** 16 pages, 9 figures, submitted to Nature Scientific Data
>
> **摘要:** Interpretation of giga-pixel whole-slide images (WSIs) is an important but difficult task for pathologists. Their diagnostic accuracy is estimated to average around 70%. Adding a second pathologist does not substantially improve decision consistency. The field lacks adequate behavioral data to explain diagnostic errors and inconsistencies. To fill in this gap, we present PathoGaze1.0, a comprehensive behavioral dataset capturing the dynamic visual search and decision-making processes of the full diagnostic workflow during cancer diagnosis. The dataset comprises 18.69 hours of eye-tracking, mouse interaction, stimulus tracking, viewport navigation, and diagnostic decision data (EMSVD) collected from 19 pathologists interpreting 397 WSIs. The data collection process emphasizes ecological validity through an application-grounded testbed, called PTAH. In total, we recorded 171,909 fixations, 263,320 saccades, and 1,867,362 mouse interaction events. In addition, such data could also be used to improve the training of both pathologists and AI systems that might support human experts. All experiments were preregistered at https://osf.io/hj9a7, and the complete dataset along with analysis code is available at https://go.osu.edu/pathogaze.
>
---
#### [new 017] SAGE: Structure-Aware Generative Video Transitions between Diverse Clips
- **分类: cs.CV**

- **简介: 该论文提出SAGE，一种零样本视频过渡生成方法，旨在解决跨多样视频片段间生成视觉连贯、语义一致中间帧的问题。通过结合线条图与运动流的结构引导与生成合成，实现无需微调的高质量过渡，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24667v1](http://arxiv.org/pdf/2510.24667v1)**

> **作者:** Mia Kan; Yilin Liu; Niloy Mitra
>
> **备注:** Website: https://kan32501.github.io/sage.github.io/
>
> **摘要:** Video transitions aim to synthesize intermediate frames between two clips, but naive approaches such as linear blending introduce artifacts that limit professional use or break temporal coherence. Traditional techniques (cross-fades, morphing, frame interpolation) and recent generative inbetweening methods can produce high-quality plausible intermediates, but they struggle with bridging diverse clips involving large temporal gaps or significant semantic differences, leaving a gap for content-aware and visually coherent transitions. We address this challenge by drawing on artistic workflows, distilling strategies such as aligning silhouettes and interpolating salient features to preserve structure and perceptual continuity. Building on this, we propose SAGE (Structure-Aware Generative vidEo transitions) as a zeroshot approach that combines structural guidance, provided via line maps and motion flow, with generative synthesis, enabling smooth, semantically consistent transitions without fine-tuning. Extensive experiments and comparison with current alternatives, namely [FILM, TVG, DiffMorpher, VACE, GI], demonstrate that SAGE outperforms both classical and generative baselines on quantitative metrics and user studies for producing transitions between diverse clips. Code to be released on acceptance.
>
---
#### [new 018] Decoupling What to Count and Where to See for Referring Expression Counting
- **分类: cs.CV**

- **简介: 该论文聚焦于指代表达计数（REC）任务，解决现有方法因标注点集中在类别代表性区域而忽略属性信息的问题。提出W2-Net框架，通过双查询机制分离“计数对象”与“关注区域”，结合可区分匹配策略，显著提升计数与定位精度。**

- **链接: [http://arxiv.org/pdf/2510.24374v1](http://arxiv.org/pdf/2510.24374v1)**

> **作者:** Yuda Zou; Zijian Zhang; Yongchao Xu
>
> **摘要:** Referring Expression Counting (REC) extends class-level object counting to the fine-grained subclass-level, aiming to enumerate objects matching a textual expression that specifies both the class and distinguishing attribute. A fundamental challenge, however, has been overlooked: annotation points are typically placed on class-representative locations (e.g., heads), forcing models to focus on class-level features while neglecting attribute information from other visual regions (e.g., legs for "walking"). To address this, we propose W2-Net, a novel framework that explicitly decouples the problem into "what to count" and "where to see" via a dual-query mechanism. Specifically, alongside the standard what-to-count (w2c) queries that localize the object, we introduce dedicated where-to-see (w2s) queries. The w2s queries are guided to seek and extract features from attribute-specific visual regions, enabling precise subclass discrimination. Furthermore, we introduce Subclass Separable Matching (SSM), a novel matching strategy that incorporates a repulsive force to enhance inter-subclass separability during label assignment. W2-Net significantly outperforms the state-of-the-art on the REC-8K dataset, reducing counting error by 22.5% (validation) and 18.0% (test), and improving localization F1 by 7% and 8%, respectively. Code will be available.
>
---
#### [new 019] Few-Shot Remote Sensing Image Scene Classification with CLIP and Prompt Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感图像场景分类中标签数据稀缺的问题，提出基于CLIP与提示学习的少样本分类方法。通过多种提示学习策略，有效缩小视觉与文本模态间的领域差距，显著提升跨域泛化能力，优于零样本和线性探测基线。**

- **链接: [http://arxiv.org/pdf/2510.24321v1](http://arxiv.org/pdf/2510.24321v1)**

> **作者:** Ivica Dimitrovski; Vlatko Spasev; Ivan Kitanovski
>
> **摘要:** Remote sensing applications increasingly rely on deep learning for scene classification. However, their performance is often constrained by the scarcity of labeled data and the high cost of annotation across diverse geographic and sensor domains. While recent vision-language models like CLIP have shown promise by learning transferable representations at scale by aligning visual and textual modalities, their direct application to remote sensing remains suboptimal due to significant domain gaps and the need for task-specific semantic adaptation. To address this critical challenge, we systematically explore prompt learning as a lightweight and efficient adaptation strategy for few-shot remote sensing image scene classification. We evaluate several representative methods, including Context Optimization, Conditional Context Optimization, Multi-modal Prompt Learning, and Prompting with Self-Regulating Constraints. These approaches reflect complementary design philosophies: from static context optimization to conditional prompts for enhanced generalization, multi-modal prompts for joint vision-language adaptation, and semantically regularized prompts for stable learning without forgetting. We benchmark these prompt-learning methods against two standard baselines: zero-shot CLIP with hand-crafted prompts and a linear probe trained on frozen CLIP features. Through extensive experiments on multiple benchmark remote sensing datasets, including cross-dataset generalization tests, we demonstrate that prompt learning consistently outperforms both baselines in few-shot scenarios. Notably, Prompting with Self-Regulating Constraints achieves the most robust cross-domain performance. Our findings underscore prompt learning as a scalable and efficient solution for bridging the domain gap in satellite and aerial imagery, providing a strong foundation for future research in this field.
>
---
#### [new 020] TurboPortrait3D: Single-step diffusion-based fast portrait novel-view synthesis
- **分类: cs.CV**

- **简介: 该论文提出TurboPortrait3D，用于高效的人像新视角合成。针对现有方法在细节保留与多视角一致性上的不足，结合单步扩散模型与3D感知生成，仅需一张正面图像即可快速生成高质量、一致的多视角人像渲染，显著提升视觉质量与运行效率。**

- **链接: [http://arxiv.org/pdf/2510.23929v1](http://arxiv.org/pdf/2510.23929v1)**

> **作者:** Emily Kim; Julieta Martinez; Timur Bagautdinov; Jessica Hodgins
>
> **摘要:** We introduce TurboPortrait3D: a method for low-latency novel-view synthesis of human portraits. Our approach builds on the observation that existing image-to-3D models for portrait generation, while capable of producing renderable 3D representations, are prone to visual artifacts, often lack of detail, and tend to fail at fully preserving the identity of the subject. On the other hand, image diffusion models excel at generating high-quality images, but besides being computationally expensive, are not grounded in 3D and thus are not directly capable of producing multi-view consistent outputs. In this work, we demonstrate that image-space diffusion models can be used to significantly enhance the quality of existing image-to-avatar methods, while maintaining 3D-awareness and running with low-latency. Our method takes a single frontal image of a subject as input, and applies a feedforward image-to-avatar generation pipeline to obtain an initial 3D representation and corresponding noisy renders. These noisy renders are then fed to a single-step diffusion model which is conditioned on input image(s), and is specifically trained to refine the renders in a multi-view consistent way. Moreover, we introduce a novel effective training strategy that includes pre-training on a large corpus of synthetic multi-view data, followed by fine-tuning on high-quality real images. We demonstrate that our approach both qualitatively and quantitatively outperforms current state-of-the-art for portrait novel-view synthesis, while being efficient in time.
>
---
#### [new 021] Kernelized Sparse Fine-Tuning with Bi-level Parameter Competition for Vision Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉模型参数高效微调中的高内存占用与权重定位不准问题，提出一阶段方法SNELLA。通过核化低秩分解增强表达能力，结合双层自适应稀疏分配机制，实现高效稀疏更新，显著降低内存消耗并提升性能，在分类、分割等任务上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2510.24037v1](http://arxiv.org/pdf/2510.24037v1)**

> **作者:** Shufan Shen; Junshu Sun; Shuhui Wang; Qingming Huang
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) aims to adapt pre-trained vision models to downstream tasks. Among PEFT paradigms, sparse tuning achieves remarkable performance by adjusting only the weights most relevant to downstream tasks, rather than densely tuning the entire weight matrix. Current methods follow a two-stage paradigm. First, it locates task-relevant weights by gradient information, which overlooks the parameter adjustments during fine-tuning and limits the performance. Second, it updates only the located weights by applying a sparse mask to the gradient of the weight matrix, which results in high memory usage due to the storage of all weight matrices in the optimizer. In this paper, we propose a one-stage method named SNELLA to overcome the above limitations. For memory usage, SNELLA selectively updates the weight matrix by adding it to another sparse matrix that is merged by two low-rank learnable matrices. We extend the low-rank decomposition by introducing nonlinear kernel functions, thereby increasing the rank of the resulting merged matrix to prevent the interdependency among weight updates, enabling better adaptation to downstream tasks. For locating task-relevant weights, we propose an adaptive bi-level sparsity allocation mechanism that encourages weights to compete across and inside layers based on their importance scores in an end-to-end manner. Extensive experiments are conducted on classification, segmentation, and generation tasks using different pre-trained vision models. The results show that SNELLA achieves SOTA performance with low memory usage. Notably, SNELLA obtains 1.8% (91.9% v.s. 90.1%) higher Top-1 accuracy on the FGVC benchmark compared to SPT-LoRA. Compared to previous methods, SNELLA achieves a memory reduction of 31.1%-39.9% across models with parameter scales from 86M to 632M. Our source codes are available at https://github.com/ssfgunner/SNELL.
>
---
#### [new 022] Improving Visual Discriminability of CLIP for Training-Free Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对训练-free开放词汇语义分割任务，解决CLIP模型因图像级预训练导致像素级视觉判别力不足的问题。提出LHT-CLIP框架，通过重加权、选择性头增强和异常令牌替换，提升多层、多头、多标记的视觉判别能力，显著改善分割性能。**

- **链接: [http://arxiv.org/pdf/2510.23894v1](http://arxiv.org/pdf/2510.23894v1)**

> **作者:** Jinxin Zhou; Jiachen Jiang; Zhihui Zhu
>
> **备注:** 23 pages, 10 figures, 14 tables
>
> **摘要:** Extending CLIP models to semantic segmentation remains challenging due to the misalignment between their image-level pre-training objectives and the pixel-level visual understanding required for dense prediction. While prior efforts have achieved encouraging results by reorganizing the final layer and features, they often inherit the global alignment bias of preceding layers, leading to suboptimal segmentation performance. In this work, we propose LHT-CLIP, a novel training-free framework that systematically exploits the visual discriminability of CLIP across layer, head, and token levels. Through comprehensive analysis, we reveal three key insights: (i) the final layers primarily strengthen image-text alignment with sacrifice of visual discriminability (e.g., last 3 layers in ViT-B/16 and 8 layers in ViT-L/14), partly due to the emergence of anomalous tokens; (ii) a subset of attention heads (e.g., 10 out of 144 in ViT-B/16) display consistently strong visual discriminability across datasets; (iii) abnormal tokens display sparse and consistent activation pattern compared to normal tokens. Based on these findings, we propose three complementary techniques: semantic-spatial reweighting, selective head enhancement, and abnormal token replacement to effectively restore visual discriminability and improve segmentation performance without any additional training, auxiliary pre-trained networks, or extensive hyperparameter tuning. Extensive experiments on 8 common semantic segmentation benchmarks demonstrate that LHT-CLIP achieves state-of-the-art performance across diverse scenarios, highlighting its effectiveness and practicality for real-world deployment.
>
---
#### [new 023] Adaptive Training of INRs via Pruning and Densification
- **分类: cs.CV**

- **简介: 该论文针对隐式神经表示（INR）中频率与架构选择困难、参数冗余问题，提出AIRe自适应训练框架。通过神经元剪枝减少冗余，频谱密集化增强表达能力，在保持或提升重建质量的同时压缩模型规模，适用于图像与SDF建模任务。**

- **链接: [http://arxiv.org/pdf/2510.23943v1](http://arxiv.org/pdf/2510.23943v1)**

> **作者:** Diana Aldana; João Paulo Lima; Daniel Csillag; Daniel Perazzo; Haoan Feng; Luiz Velho; Tiago Novello
>
> **摘要:** Encoding input coordinates with sinusoidal functions into multilayer perceptrons (MLPs) has proven effective for implicit neural representations (INRs) of low-dimensional signals, enabling the modeling of high-frequency details. However, selecting appropriate input frequencies and architectures while managing parameter redundancy remains an open challenge, often addressed through heuristics and heavy hyperparameter optimization schemes. In this paper, we introduce AIRe ($\textbf{A}$daptive $\textbf{I}$mplicit neural $\textbf{Re}$presentation), an adaptive training scheme that refines the INR architecture over the course of optimization. Our method uses a neuron pruning mechanism to avoid redundancy and input frequency densification to improve representation capacity, leading to an improved trade-off between network size and reconstruction quality. For pruning, we first identify less-contributory neurons and apply a targeted weight decay to transfer their information to the remaining neurons, followed by structured pruning. Next, the densification stage adds input frequencies to spectrum regions where the signal underfits, expanding the representational basis. Through experiments on images and SDFs, we show that AIRe reduces model size while preserving, or even improving, reconstruction quality. Code and pretrained models will be released for public use.
>
---
#### [new 024] Fast and accurate neural reflectance transformation imaging through knowledge distillation
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对神经反射变换成像（NeuralRTI）渲染慢的问题，提出基于知识蒸馏的高效方法DisK-NeuralRTI。通过蒸馏大模型知识到小模型，实现快速准确的交互式光照渲染，显著降低计算成本，同时保持高质量图像输出，适用于有限硬件环境下的大规模图像处理。**

- **链接: [http://arxiv.org/pdf/2510.24486v1](http://arxiv.org/pdf/2510.24486v1)**

> **作者:** Tinsae G. Dulecha; Leonardo Righetto; Ruggero Pintus; Enrico Gobbetti; Andrea Giachetti
>
> **备注:** 18 pages
>
> **摘要:** Reflectance Transformation Imaging (RTI) is very popular for its ability to visually analyze surfaces by enhancing surface details through interactive relighting, starting from only a few tens of photographs taken with a fixed camera and variable illumination. Traditional methods like Polynomial Texture Maps (PTM) and Hemispherical Harmonics (HSH) are compact and fast, but struggle to accurately capture complex reflectance fields using few per-pixel coefficients and fixed bases, leading to artifacts, especially in highly reflective or shadowed areas. The NeuralRTI approach, which exploits a neural autoencoder to learn a compact function that better approximates the local reflectance as a function of light directions, has been shown to produce superior quality at comparable storage cost. However, as it performs interactive relighting with custom decoder networks with many parameters, the rendering step is computationally expensive and not feasible at full resolution for large images on limited hardware. Earlier attempts to reduce costs by directly training smaller networks have failed to produce valid results. For this reason, we propose to reduce its computational cost through a novel solution based on Knowledge Distillation (DisK-NeuralRTI). ...
>
---
#### [new 025] MIC-BEV: Multi-Infrastructure Camera Bird's-Eye-View Transformer with Relation-Aware Fusion for 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文针对基础设施感知中的多相机3D目标检测任务，解决多视角、异构相机配置及传感器退化等问题。提出MIC-BEV框架，基于Transformer的BEV融合模块利用相机与BEV单元间的几何关系，实现鲁棒的多视图特征融合，并构建合成数据集M2I用于训练评估。**

- **链接: [http://arxiv.org/pdf/2510.24688v1](http://arxiv.org/pdf/2510.24688v1)**

> **作者:** Yun Zhang; Zhaoliang Zheng; Johnson Liu; Zhiyu Huang; Zewei Zhou; Zonglin Meng; Tianhui Cai; Jiaqi Ma
>
> **摘要:** Infrastructure-based perception plays a crucial role in intelligent transportation systems, offering global situational awareness and enabling cooperative autonomy. However, existing camera-based detection models often underperform in such scenarios due to challenges such as multi-view infrastructure setup, diverse camera configurations, degraded visual inputs, and various road layouts. We introduce MIC-BEV, a Transformer-based bird's-eye-view (BEV) perception framework for infrastructure-based multi-camera 3D object detection. MIC-BEV flexibly supports a variable number of cameras with heterogeneous intrinsic and extrinsic parameters and demonstrates strong robustness under sensor degradation. The proposed graph-enhanced fusion module in MIC-BEV integrates multi-view image features into the BEV space by exploiting geometric relationships between cameras and BEV cells alongside latent visual cues. To support training and evaluation, we introduce M2I, a synthetic dataset for infrastructure-based object detection, featuring diverse camera configurations, road layouts, and environmental conditions. Extensive experiments on both M2I and the real-world dataset RoScenes demonstrate that MIC-BEV achieves state-of-the-art performance in 3D object detection. It also remains robust under challenging conditions, including extreme weather and sensor degradation. These results highlight the potential of MIC-BEV for real-world deployment. The dataset and source code are available at: https://github.com/HandsomeYun/MIC-BEV.
>
---
#### [new 026] Enhancing Vision-Language Models for Autonomous Driving through Task-Specific Prompting and Spatial Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶场景下的视觉语言模型（VLM）理解任务，解决多任务间干扰与空间推理不足问题。提出基于提示路由、任务专用提示、视觉组装与参数调优的框架，提升感知、预测、规划与异常检测性能，在清洁与损坏数据上分别达70.87%与72.85%准确率。**

- **链接: [http://arxiv.org/pdf/2510.24152v1](http://arxiv.org/pdf/2510.24152v1)**

> **作者:** Aodi Wu; Xubo Luo
>
> **备注:** RoboSense Challenge with IROS 2025
>
> **摘要:** This technical report presents our solution for the RoboSense Challenge at IROS 2025, which evaluates Vision-Language Models (VLMs) on autonomous driving scene understanding across perception, prediction, planning, and corruption detection tasks. We propose a systematic framework built on four core components. First, a Mixture-of-Prompts router classifies questions and dispatches them to task-specific expert prompts, eliminating interference across diverse question types. Second, task-specific prompts embed explicit coordinate systems, spatial reasoning rules, role-playing, Chain-of-Thought/Tree-of-Thought reasoning, and few-shot examples tailored to each task. Third, a visual assembly module composes multi-view images with object crops, magenta markers, and adaptive historical frames based on question requirements. Fourth, we configure model inference parameters (temperature, top-p, message roles) per task to optimize output quality. Implemented on Qwen2.5-VL-72B, our approach achieves 70.87% average accuracy on Phase-1 (clean data) and 72.85% on Phase-2 (corrupted data), demonstrating that structured prompting and spatial grounding substantially enhance VLM performance on safety-critical autonomous driving tasks. Code and prompt are available at https://github.com/wuaodi/UCAS-CSU-phase2.
>
---
#### [new 027] Group Relative Attention Guidance for Image Editing
- **分类: cs.CV**

- **简介: 该论文针对扩散模型驱动的图像编辑任务，解决现有方法难以精细控制编辑强度的问题。提出Group Relative Attention Guidance（GRAG），通过重加权注意力中的内容差异信号，实现无需调参的连续、细粒度编辑强度控制，显著提升编辑质量和可控性。**

- **链接: [http://arxiv.org/pdf/2510.24657v1](http://arxiv.org/pdf/2510.24657v1)**

> **作者:** Xuanpu Zhang; Xuesong Niu; Ruidong Chen; Dan Song; Jianhao Zeng; Penghui Du; Haoxiang Cao; Kai Wu; An-an Liu
>
> **摘要:** Recently, image editing based on Diffusion-in-Transformer models has undergone rapid development. However, existing editing methods often lack effective control over the degree of editing, limiting their ability to achieve more customized results. To address this limitation, we investigate the MM-Attention mechanism within the DiT model and observe that the Query and Key tokens share a bias vector that is only layer-dependent. We interpret this bias as representing the model's inherent editing behavior, while the delta between each token and its corresponding bias encodes the content-specific editing signals. Based on this insight, we propose Group Relative Attention Guidance, a simple yet effective method that reweights the delta values of different tokens to modulate the focus of the model on the input image relative to the editing instruction, enabling continuous and fine-grained control over editing intensity without any tuning. Extensive experiments conducted on existing image editing frameworks demonstrate that GRAG can be integrated with as few as four lines of code, consistently enhancing editing quality. Moreover, compared to the commonly used Classifier-Free Guidance, GRAG achieves smoother and more precise control over the degree of editing. Our code will be released at https://github.com/little-misfit/GRAG-Image-Editing.
>
---
#### [new 028] Neural USD: An object-centric framework for iterative editing and control
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Neural USD，一种基于对象的迭代编辑框架，旨在解决生成图像中精确、局部编辑难的问题。通过借鉴计算机图形学中的USD标准，构建分层结构化场景表示，实现对物体外观、几何与姿态的解耦控制，支持可迭代的精细编辑。**

- **链接: [http://arxiv.org/pdf/2510.23956v1](http://arxiv.org/pdf/2510.23956v1)**

> **作者:** Alejandro Escontrela; Shrinu Kushagra; Sjoerd van Steenkiste; Yulia Rubanova; Aleksander Holynski; Kelsey Allen; Kevin Murphy; Thomas Kipf
>
> **备注:** 22 pages, 16 figures, 1 table
>
> **摘要:** Amazing progress has been made in controllable generative modeling, especially over the last few years. However, some challenges remain. One of them is precise and iterative object editing. In many of the current methods, trying to edit the generated image (for example, changing the color of a particular object in the scene or changing the background while keeping other elements unchanged) by changing the conditioning signals often leads to unintended global changes in the scene. In this work, we take the first steps to address the above challenges. Taking inspiration from the Universal Scene Descriptor (USD) standard developed in the computer graphics community, we introduce the "Neural Universal Scene Descriptor" or Neural USD. In this framework, we represent scenes and objects in a structured, hierarchical manner. This accommodates diverse signals, minimizes model-specific constraints, and enables per-object control over appearance, geometry, and pose. We further apply a fine-tuning approach which ensures that the above control signals are disentangled from one another. We evaluate several design considerations for our framework, demonstrating how Neural USD enables iterative and incremental workflows. More information at: https://escontrela.me/neural_usd .
>
---
#### [new 029] DeshadowMamba: Deshadowing as 1D Sequential Similarity
- **分类: cs.CV**

- **简介: 该论文针对图像去阴影任务，解决现有模型因固定注意力机制导致结构扭曲与颜色不一致的问题。提出DeshadowMamba，结合Mamba的序列建模能力与自研CrossGate机制，增强阴影区域上下文感知；引入ColorShift正则化，提升颜色恢复准确性，显著改善视觉质量与定量指标。**

- **链接: [http://arxiv.org/pdf/2510.24260v1](http://arxiv.org/pdf/2510.24260v1)**

> **作者:** Zhaotong Yang; Yi Chen; Yanying Li; Shengfeng He; Yangyang Xu; Junyu Dong; Jian Yang; Yong Du
>
> **摘要:** Recent deep models for image shadow removal often rely on attention-based architectures to capture long-range dependencies. However, their fixed attention patterns tend to mix illumination cues from irrelevant regions, leading to distorted structures and inconsistent colors. In this work, we revisit shadow removal from a sequence modeling perspective and explore the use of Mamba, a selective state space model that propagates global context through directional state transitions. These transitions yield an efficient global receptive field while preserving positional continuity. Despite its potential, directly applying Mamba to image data is suboptimal, since it lacks awareness of shadow-non-shadow semantics and remains susceptible to color interference from nearby regions. To address these limitations, we propose CrossGate, a directional modulation mechanism that injects shadow-aware similarity into Mamba's input gate, allowing selective integration of relevant context along transition axes. To further ensure appearance fidelity, we introduce ColorShift regularization, a contrastive learning objective driven by global color statistics. By synthesizing structured informative negatives, it guides the model to suppress color contamination and achieve robust color restoration. Together, these components adapt sequence modeling to the structural integrity and chromatic consistency required for shadow removal. Extensive experiments on public benchmarks demonstrate that DeshadowMamba achieves state-of-the-art visual quality and strong quantitative performance.
>
---
#### [new 030] ResNet: Enabling Deep Convolutional Neural Networks through Residual Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对深度卷积神经网络训练中的梯度消失问题，提出残差网络（ResNet），通过引入跳跃连接使梯度可直接传播。该方法显著提升网络深度与训练稳定性，在CIFAR-10上实现89.9%准确率，优于传统深层网络。任务为图像分类。**

- **链接: [http://arxiv.org/pdf/2510.24036v1](http://arxiv.org/pdf/2510.24036v1)**

> **作者:** Xingyu Liu; Kun Ming Goh
>
> **备注:** 3 pages, 5 figures, 1 table
>
> **摘要:** Convolutional Neural Networks (CNNs) has revolutionized computer vision, but training very deep networks has been challenging due to the vanishing gradient problem. This paper explores Residual Networks (ResNet), introduced by He et al. (2015), which overcomes this limitation by using skip connections. ResNet enables the training of networks with hundreds of layers by allowing gradients to flow directly through shortcut connections that bypass intermediate layers. In our implementation on the CIFAR-10 dataset, ResNet-18 achieves 89.9% accuracy compared to 84.1% for a traditional deep CNN of similar depth, while also converging faster and training more stably.
>
---
#### [new 031] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
- **分类: cs.CV; cs.AI; cs.LG; q-bio.NC**

- **简介: 该论文研究视觉变换器（ViT）中对象绑定是否自然涌现。针对“预训练ViT能否自发形成对象绑定”问题，通过设计相似性探测器验证，发现自监督模型（如DINO、MAE）能高效识别同一对象的图像块，且该能力与注意力机制和预训练目标协同，表明对象绑定是特定预训练下自然涌现的认知能力。**

- **链接: [http://arxiv.org/pdf/2510.24709v1](http://arxiv.org/pdf/2510.24709v1)**

> **作者:** Yihao Li; Saeed Salehi; Lyle Ungar; Konrad P. Kording
>
> **备注:** Accepted as a Spotlight at NeurIPS 2025
>
> **摘要:** Object binding, the brain's ability to bind the many features that collectively represent an object into a coherent whole, is central to human cognition. It groups low-level perceptual features into high-level object representations, stores those objects efficiently and compositionally in memory, and supports human reasoning about individual object instances. While prior work often imposes object-centric attention (e.g., Slot Attention) explicitly to probe these benefits, it remains unclear whether this ability naturally emerges in pre-trained Vision Transformers (ViTs). Intuitively, they could: recognizing which patches belong to the same object should be useful for downstream prediction and thus guide attention. Motivated by the quadratic nature of self-attention, we hypothesize that ViTs represent whether two patches belong to the same object, a property we term IsSameObject. We decode IsSameObject from patch embeddings across ViT layers using a similarity probe, which reaches over 90% accuracy. Crucially, this object-binding capability emerges reliably in self-supervised ViTs (DINO, MAE, CLIP), but markedly weaker in ImageNet-supervised models, suggesting that binding is not a trivial architectural artifact, but an ability acquired through specific pretraining objectives. We further discover that IsSameObject is encoded in a low-dimensional subspace on top of object features, and that this signal actively guides attention. Ablating IsSameObject from model activations degrades downstream performance and works against the learning objective, implying that emergent object binding naturally serves the pretraining objective. Our findings challenge the view that ViTs lack object binding and highlight how symbolic knowledge of "which parts belong together" emerges naturally in a connectionist system.
>
---
#### [new 032] Compositional Image Synthesis with Inference-Time Scaling
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本生成图像中的组合性问题，提出一种无需训练的推理时缩放框架。通过大语言模型生成显式布局，并利用视觉语言模型迭代优化生成结果，提升场景与提示的一致性，同时保持图像美感。**

- **链接: [http://arxiv.org/pdf/2510.24133v1](http://arxiv.org/pdf/2510.24133v1)**

> **作者:** Minsuk Ji; Sanghyeok Lee; Namhyuk Ahn
>
> **备注:** projcet page: https://github.com/gcl-inha/ReFocus
>
> **摘要:** Despite their impressive realism, modern text-to-image models still struggle with compositionality, often failing to render accurate object counts, attributes, and spatial relations. To address this challenge, we present a training-free framework that combines an object-centric approach with self-refinement to improve layout faithfulness while preserving aesthetic quality. Specifically, we leverage large language models (LLMs) to synthesize explicit layouts from input prompts, and we inject these layouts into the image generation process, where a object-centric vision-language model (VLM) judge reranks multiple candidates to select the most prompt-aligned outcome iteratively. By unifying explicit layout-grounding with self-refine-based inference-time scaling, our framework achieves stronger scene alignment with prompts compared to recent text-to-image models. The code are available at https://github.com/gcl-inha/ReFocus.
>
---
#### [new 033] A Luminance-Aware Multi-Scale Network for Polarization Image Fusion with a Multi-Scene Dataset
- **分类: cs.CV**

- **简介: 该论文针对复杂光照下偏振图像融合任务，提出亮度感知多尺度网络（MLSN），通过亮度分支动态加权特征、全局-局部融合机制及亮度增强模块，提升融合质量。构建了1000对的多场景偏振图像数据集MSP，实验证明方法在主观与客观指标上均优于现有技术。**

- **链接: [http://arxiv.org/pdf/2510.24379v1](http://arxiv.org/pdf/2510.24379v1)**

> **作者:** Zhuangfan Huang; Xiaosong Li; Gao Wang; Tao Ye; Haishu Tan; Huafeng Li
>
> **摘要:** Polarization image fusion combines S0 and DOLP images to reveal surface roughness and material properties through complementary texture features, which has important applications in camouflage recognition, tissue pathology analysis, surface defect detection and other fields. To intergrate coL-Splementary information from different polarized images in complex luminance environment, we propose a luminance-aware multi-scale network (MLSN). In the encoder stage, we propose a multi-scale spatial weight matrix through a brightness-branch , which dynamically weighted inject the luminance into the feature maps, solving the problem of inherent contrast difference in polarized images. The global-local feature fusion mechanism is designed at the bottleneck layer to perform windowed self-attention computation, to balance the global context and local details through residual linking in the feature dimension restructuring stage. In the decoder stage, to further improve the adaptability to complex lighting, we propose a Brightness-Enhancement module, establishing the mapping relationship between luminance distribution and texture features, realizing the nonlinear luminance correction of the fusion result. We also present MSP, an 1000 pairs of polarized images that covers 17 types of indoor and outdoor complex lighting scenes. MSP provides four-direction polarization raw maps, solving the scarcity of high-quality datasets in polarization image fusion. Extensive experiment on MSP, PIF and GAND datasets verify that the proposed MLSN outperms the state-of-the-art methods in subjective and objective evaluations, and the MS-SSIM and SD metircs are higher than the average values of other methods by 8.57%, 60.64%, 10.26%, 63.53%, 22.21%, and 54.31%, respectively. The source code and dataset is avalable at https://github.com/1hzf/MLS-UNet.
>
---
#### [new 034] Enhancing CLIP Robustness via Cross-Modality Alignment
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型CLIP在对抗攻击下鲁棒性差的问题，提出无需训练的跨模态对齐方法COLA。通过最优传输机制，增强图像与文本特征在全局和局部结构上的对齐，有效抑制非语义扰动，显著提升零样本分类在对抗攻击下的性能。**

- **链接: [http://arxiv.org/pdf/2510.24038v1](http://arxiv.org/pdf/2510.24038v1)**

> **作者:** Xingyu Zhu; Beier Zhu; Shuo Wang; Kesen Zhao; Hanwang Zhang
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Vision-language models (VLMs) such as CLIP demonstrate strong generalization in zero-shot classification but remain highly vulnerable to adversarial perturbations. Existing methods primarily focus on adversarial fine-tuning or prompt optimization; they often overlook the gaps in CLIP's encoded features, which is shown as the text and image features lie far apart from each other. This misalignment is significantly amplified under adversarial perturbations, leading to severe degradation in classification performance. To address this problem, we propose Cross-modality Alignment, dubbed COLA, an optimal transport-based framework that explicitly addresses adversarial misalignment by restoring both global image-text alignment and local structural consistency in the feature space. (1) COLA first projects adversarial image embeddings onto a subspace spanned by class text features, effectively filtering out non-semantic distortions while preserving discriminative information. (2) It then models images and texts as discrete distributions over multiple augmented views and refines their alignment via OT, with the subspace projection seamlessly integrated into the cost computation. This design ensures stable cross-modal alignment even under adversarial conditions. COLA is training-free and compatible with existing fine-tuned models. Extensive evaluations across 14 zero-shot classification benchmarks demonstrate the effectiveness of COLA, especially with an average improvement of 6.7% on ImageNet and its variants under PGD adversarial attacks, while maintaining high accuracy on clean samples.
>
---
#### [new 035] Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance
- **分类: cs.CV**

- **简介: 该论文针对扩散模型中MoE应用效果不佳的问题，提出ProMoE框架。通过两阶段显式路由引导，区分条件与无条件图像令牌，并基于语义原型优化分配，增强专家专业化。引入对比损失提升路由一致性与多样性，在ImageNet上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24711v1](http://arxiv.org/pdf/2510.24711v1)**

> **作者:** Yujie Wei; Shiwei Zhang; Hangjie Yuan; Yujin Han; Zhekai Chen; Jiayu Wang; Difan Zou; Xihui Liu; Yingya Zhang; Yu Liu; Hongming Shan
>
> **摘要:** Mixture-of-Experts (MoE) has emerged as a powerful paradigm for scaling model capacity while preserving computational efficiency. Despite its notable success in large language models (LLMs), existing attempts to apply MoE to Diffusion Transformers (DiTs) have yielded limited gains. We attribute this gap to fundamental differences between language and visual tokens. Language tokens are semantically dense with pronounced inter-token variation, while visual tokens exhibit spatial redundancy and functional heterogeneity, hindering expert specialization in vision MoE. To this end, we present ProMoE, an MoE framework featuring a two-step router with explicit routing guidance that promotes expert specialization. Specifically, this guidance encourages the router to partition image tokens into conditional and unconditional sets via conditional routing according to their functional roles, and refine the assignments of conditional image tokens through prototypical routing with learnable prototypes based on semantic content. Moreover, the similarity-based expert allocation in latent space enabled by prototypical routing offers a natural mechanism for incorporating explicit semantic guidance, and we validate that such guidance is crucial for vision MoE. Building on this, we propose a routing contrastive loss that explicitly enhances the prototypical routing process, promoting intra-expert coherence and inter-expert diversity. Extensive experiments on ImageNet benchmark demonstrate that ProMoE surpasses state-of-the-art methods under both Rectified Flow and DDPM training objectives. Code and models will be made publicly available.
>
---
#### [new 036] A Dual-Branch CNN for Robust Detection of AI-Generated Facial Forgeries
- **分类: cs.CV**

- **简介: 该论文针对AI生成人脸伪造检测任务，解决伪造图像难以识别的问题。提出双分支CNN模型，融合空间与频率域特征，引入通道注意力与联合损失函数，提升检测鲁棒性与泛化能力，在多类伪造数据上优于人类水平。**

- **链接: [http://arxiv.org/pdf/2510.24640v1](http://arxiv.org/pdf/2510.24640v1)**

> **作者:** Xin Zhang; Yuqi Song; Fei Zuo
>
> **摘要:** The rapid advancement of generative AI has enabled the creation of highly realistic forged facial images, posing significant threats to AI security, digital media integrity, and public trust. Face forgery techniques, ranging from face swapping and attribute editing to powerful diffusion-based image synthesis, are increasingly being used for malicious purposes such as misinformation, identity fraud, and defamation. This growing challenge underscores the urgent need for robust and generalizable face forgery detection methods as a critical component of AI security infrastructure. In this work, we propose a novel dual-branch convolutional neural network for face forgery detection that leverages complementary cues from both spatial and frequency domains. The RGB branch captures semantic information, while the frequency branch focuses on high-frequency artifacts that are difficult for generative models to suppress. A channel attention module is introduced to adaptively fuse these heterogeneous features, highlighting the most informative channels for forgery discrimination. To guide the network's learning process, we design a unified loss function, FSC Loss, that combines focal loss, supervised contrastive loss, and a frequency center margin loss to enhance class separability and robustness. We evaluate our model on the DiFF benchmark, which includes forged images generated from four representative methods: text-to-image, image-to-image, face swap, and face edit. Our method achieves strong performance across all categories and outperforms average human accuracy. These results demonstrate the model's effectiveness and its potential contribution to safeguarding AI ecosystems against visual forgery attacks.
>
---
#### [new 037] Deeply-Conditioned Image Compression via Self-Generated Priors
- **分类: cs.CV**

- **简介: 该论文属于图像压缩任务，针对现有学习型压缩方法在低比特率下产生几何失真的问题。提出基于自生成先验的深度条件压缩框架（DCIC-sgp），通过分解结构与细节信息流，利用先验深度调控编码器，使模型专注捕捉高熵细节，显著降低失真并提升率失真性能。**

- **链接: [http://arxiv.org/pdf/2510.24437v1](http://arxiv.org/pdf/2510.24437v1)**

> **作者:** Zhineng Zhao; Zhihai He; Zikun Zhou; Siwei Ma; Yaowei Wang
>
> **摘要:** Learned image compression (LIC) has shown great promise for achieving high rate-distortion performance. However, current LIC methods are often limited in their capability to model the complex correlation structures inherent in natural images, particularly the entanglement of invariant global structures with transient local textures within a single monolithic representation. This limitation precipitates severe geometric deformation at low bitrates. To address this, we introduce a framework predicated on functional decomposition, which we term Deeply-Conditioned Image Compression via self-generated priors (DCIC-sgp). Our central idea is to first encode a potent, self-generated prior to encapsulate the image's structural backbone. This prior is subsequently utilized not as mere side-information, but to holistically modulate the entire compression pipeline. This deep conditioning, most critically of the analysis transform, liberates it to dedicate its representational capacity to the residual, high-entropy details. This hierarchical, dependency-driven approach achieves an effective disentanglement of information streams. Our extensive experiments validate this assertion; visual analysis demonstrates that our method substantially mitigates the geometric deformation artifacts that plague conventional codecs at low bitrates. Quantitatively, our framework establishes highly competitive performance, achieving significant BD-rate reductions of 14.4%, 15.7%, and 15.1% against the VVC test model VTM-12.1 on the Kodak, CLIC, and Tecnick datasets.
>
---
#### [new 038] VC4VG: Optimizing Video Captions for Text-to-Video Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对文本到视频生成（T2V）中视频描述质量影响模型性能的问题，提出VC4VG框架，通过多维度分析与优化视频标题，构建专用评估基准VC4VG-Bench，实验证明高质量标题显著提升生成效果，推动T2V训练数据优化。**

- **链接: [http://arxiv.org/pdf/2510.24134v1](http://arxiv.org/pdf/2510.24134v1)**

> **作者:** Yang Du; Zhuoran Lin; Kaiqiang Song; Biao Wang; Zhicheng Zheng; Tiezheng Ge; Bo Zheng; Qin Jin
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Recent advances in text-to-video (T2V) generation highlight the critical role of high-quality video-text pairs in training models capable of producing coherent and instruction-aligned videos. However, strategies for optimizing video captions specifically for T2V training remain underexplored. In this paper, we introduce VC4VG (Video Captioning for Video Generation), a comprehensive caption optimization framework tailored to the needs of T2V models.We begin by analyzing caption content from a T2V perspective, decomposing the essential elements required for video reconstruction into multiple dimensions, and proposing a principled caption design methodology. To support evaluation, we construct VC4VG-Bench, a new benchmark featuring fine-grained, multi-dimensional, and necessity-graded metrics aligned with T2V-specific requirements.Extensive T2V fine-tuning experiments demonstrate a strong correlation between improved caption quality and video generation performance, validating the effectiveness of our approach. We release all benchmark tools and code at https://github.com/qyr0403/VC4VG to support further research.
>
---
#### [new 039] Rethinking Visual Intelligence: Insights from Video Pretraining
- **分类: cs.CV; cs.AI; 68T07, 68T45, 68T20; I.2.10; I.4.8; I.5.1; I.2.6**

- **简介: 该论文研究视觉智能，旨在解决视觉模型在组合理解、样本效率和通用问题求解上的不足。通过对比预训练语言模型与视频扩散模型（VDMs），发现VDMs在多任务上表现出更高数据效率，表明视频预训练能提供支持视觉基础模型的强归纳偏置。**

- **链接: [http://arxiv.org/pdf/2510.24448v1](http://arxiv.org/pdf/2510.24448v1)**

> **作者:** Pablo Acuaviva; Aram Davtyan; Mariam Hassan; Sebastian Stapf; Ahmad Rahimi; Alexandre Alahi; Paolo Favaro
>
> **备注:** Updated version from preprint arXiv:2506.07280 (Gen2Gen) focused on visual intelligence. This work can be considered as v2
>
> **摘要:** Large language models (LLMs) have demonstrated that large-scale pretraining enables systems to adapt rapidly to new problems with little supervision in the language domain. This success, however, has not translated as effectively to the visual domain, where models, including LLMs, continue to struggle with compositional understanding, sample efficiency, and general-purpose problem-solving. We investigate Video Diffusion Models (VDMs) as a promising direction for bridging this gap. Pretraining on spatiotemporal data endows these models with strong inductive biases for structure and dynamics, which we hypothesize can support broad task adaptability. To test this, we design a controlled evaluation in which both a pretrained LLM and a pretrained VDM are equipped with lightweight adapters and presented with tasks in their natural modalities. Across benchmarks including ARC-AGI, ConceptARC, visual games, route planning, and cellular automata, VDMs demonstrate higher data efficiency than their language counterparts. Taken together, our results indicate that video pretraining offers inductive biases that support progress toward visual foundation models.
>
---
#### [new 040] XAI Evaluation Framework for Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对语义分割任务中的可解释性问题，提出一种系统化XAI评估框架。通过像素级评价策略与定制指标，解决现有XAI评估在空间与上下文复杂性上不足的问题，验证了基于CAM方法的可靠性与有效性，推动高可信语义分割模型发展。**

- **链接: [http://arxiv.org/pdf/2510.24414v1](http://arxiv.org/pdf/2510.24414v1)**

> **作者:** Reem Hammoud; Abdul karim Gizzini; Ali J. Ghandour
>
> **摘要:** Ensuring transparency and trust in artificial intelligence (AI) models is essential, particularly as they are increasingly applied in safety-critical and high-stakes domains. Explainable AI (XAI) has emerged as a promising approach to address this challenge, yet the rigorous evaluation of XAI methods remains crucial for optimizing the trade-offs between model complexity, predictive performance, and interpretability. While extensive progress has been achieved in evaluating XAI techniques for classification tasks, evaluation strategies tailored to semantic segmentation remain relatively underexplored. This work introduces a comprehensive and systematic evaluation framework specifically designed for assessing XAI in semantic segmentation, explicitly accounting for both spatial and contextual task complexities. The framework employs pixel-level evaluation strategies and carefully designed metrics to provide fine-grained interpretability insights. Simulation results using recently adapted class activation mapping (CAM)-based XAI schemes demonstrate the efficiency, robustness, and reliability of the proposed methodology. These findings contribute to advancing transparent, trustworthy, and accountable semantic segmentation models.
>
---
#### [new 041] Generative View Stitching
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出生成式视图拼接（GVS），用于相机轨迹引导的视频生成。针对自回归模型无法利用未来信息导致的碰撞与崩溃问题，GVS采用并行采样与全景引导机制，实现全序列一致性与长程连贯性，支持复杂路径如“不可能楼梯”。**

- **链接: [http://arxiv.org/pdf/2510.24718v1](http://arxiv.org/pdf/2510.24718v1)**

> **作者:** Chonghyuk Song; Michal Stary; Boyuan Chen; George Kopanas; Vincent Sitzmann
>
> **备注:** Project website: https://andrewsonga.github.io/gvs
>
> **摘要:** Autoregressive video diffusion models are capable of long rollouts that are stable and consistent with history, but they are unable to guide the current generation with conditioning from the future. In camera-guided video generation with a predefined camera trajectory, this limitation leads to collisions with the generated scene, after which autoregression quickly collapses. To address this, we propose Generative View Stitching (GVS), which samples the entire sequence in parallel such that the generated scene is faithful to every part of the predefined camera trajectory. Our main contribution is a sampling algorithm that extends prior work on diffusion stitching for robot planning to video generation. While such stitching methods usually require a specially trained model, GVS is compatible with any off-the-shelf video model trained with Diffusion Forcing, a prevalent sequence diffusion framework that we show already provides the affordances necessary for stitching. We then introduce Omni Guidance, a technique that enhances the temporal consistency in stitching by conditioning on both the past and future, and that enables our proposed loop-closing mechanism for delivering long-range coherence. Overall, GVS achieves camera-guided video generation that is stable, collision-free, frame-to-frame consistent, and closes loops for a variety of predefined camera paths, including Oscar Reutersv\"ard's Impossible Staircase. Results are best viewed as videos at https://andrewsonga.github.io/gvs.
>
---
#### [new 042] Reasoning Visual Language Model for Chest X-Ray Analysis
- **分类: cs.CV**

- **简介: 该论文针对胸部X光片分析任务，提出一种具备链式思维推理能力的视觉语言模型。旨在解决现有模型预测不透明、缺乏临床可审计性的问题。通过两阶段训练（监督微调与强化学习），使模型输出符合放射科医生系统化思考过程的可解释推理路径，提升诊断可信度与人机协作效率。**

- **链接: [http://arxiv.org/pdf/2510.23968v1](http://arxiv.org/pdf/2510.23968v1)**

> **作者:** Andriy Myronenko; Dong Yang; Baris Turkbey; Mariam Aboian; Sena Azamat; Esra Akcicek; Hongxu Yin; Pavlo Molchanov; Marc Edgar; Yufan He; Pengfei Guo; Yucheng Tang; Daguang Xu
>
> **备注:** NV-Reason-CXR-3B
>
> **摘要:** Vision-language models (VLMs) have shown strong promise for medical image analysis, but most remain opaque, offering predictions without the transparent, stepwise reasoning clinicians rely on. We present a framework that brings chain-of-thought (CoT) reasoning to chest X-ray interpretation. Inspired by reasoning-first training paradigms, our approach is designed to learn how experts reason, not just what they conclude, by aligning intermediate steps with observable image evidence and radiology workflow. Beyond accuracy, the explicit reasoning traces support clinical auditability: they reveal why a conclusion was reached, which alternatives were considered, and where uncertainty remains, enabling quality assurance, error analysis, and safer human-AI collaboration. Our model couples high-fidelity visual encoding with a two-stage training recipe: a reasoning-style supervised fine-tuning (SFT) followed by reinforcement learning (RL) that uses verifiable rewards over a list of X-ray abnormalities. The model outputs reasoning that mirrors radiologists systematic thought process, uncertainty, and differential diagnosis. In out-of-distribution evaluation, the approach achieves competitive multi-label classification while improving interpretability. In a reader study with expert radiologists, full reasoning traces increased confidence, supported error auditing, and reduced time to finalize reports. We release code and the model NV-Reason-CXR-3B to support community progress toward trustworthy, explainable AI in chest radiography and other medical imaging tasks where reasoning quality is as critical as prediction quality.
>
---
#### [new 043] OSWorld-MCP: Benchmarking MCP Tool Invocation In Computer-Use Agents
- **分类: cs.CV**

- **简介: 该论文提出OSWorld-MCP基准，针对多模态计算机使用智能体的工具调用能力进行评估。解决现有评测偏重GUI交互而忽视工具调用的问题。构建158个高质量真实工具，设计自动化生成与验证流程，实现对决策、操作与工具调用的综合评估，推动智能体在复杂任务中的公平评测。**

- **链接: [http://arxiv.org/pdf/2510.24563v1](http://arxiv.org/pdf/2510.24563v1)**

> **作者:** Hongrui Jia; Jitong Liao; Xi Zhang; Haiyang Xu; Tianbao Xie; Chaoya Jiang; Ming Yan; Si Liu; Wei Ye; Fei Huang
>
> **摘要:** With advances in decision-making and reasoning capabilities, multimodal agents show strong potential in computer application scenarios. Past evaluations have mainly assessed GUI interaction skills, while tool invocation abilities, such as those enabled by the Model Context Protocol (MCP), have been largely overlooked. Comparing agents with integrated tool invocation to those evaluated only on GUI interaction is inherently unfair. We present OSWorld-MCP, the first comprehensive and fair benchmark for assessing computer-use agents' tool invocation, GUI operation, and decision-making abilities in a real-world environment. We design a novel automated code-generation pipeline to create tools and combine them with a curated selection from existing tools. Rigorous manual validation yields 158 high-quality tools (covering 7 common applications), each verified for correct functionality, practical applicability, and versatility. Extensive evaluations of state-of-the-art multimodal agents on OSWorld-MCP show that MCP tools generally improve task success rates (e.g., from 8.3% to 20.4% for OpenAI o3 at 15 steps, from 40.1% to 43.3% for Claude 4 Sonnet at 50 steps), underscoring the importance of assessing tool invocation capabilities. However, even the strongest models have relatively low tool invocation rates, Only 36.3%, indicating room for improvement and highlighting the benchmark's challenge. By explicitly measuring MCP tool usage skills, OSWorld-MCP deepens understanding of multimodal agents and sets a new standard for evaluating performance in complex, tool-assisted environments. Our code, environment, and data are publicly available at https://osworld-mcp.github.io.
>
---
#### [new 044] Beyond Inference Intervention: Identity-Decoupled Diffusion for Face Anonymization
- **分类: cs.CV**

- **简介: 该论文针对人脸匿名化任务，解决现有扩散模型依赖推理阶段干预导致的图像质量下降与身份泄露问题。提出ID²Face框架，通过训练阶段显式解耦身份与非身份特征，实现无需后处理的可控匿名化，提升视觉质量和隐私保护效果。**

- **链接: [http://arxiv.org/pdf/2510.24213v1](http://arxiv.org/pdf/2510.24213v1)**

> **作者:** Haoxin Yang; Yihong Lin; Jingdan Kang; Xuemiao Xu; Yue Li; Cheng Xu; Shengfeng He
>
> **摘要:** Face anonymization aims to conceal identity information while preserving non-identity attributes. Mainstream diffusion models rely on inference-time interventions such as negative guidance or energy-based optimization, which are applied post-training to suppress identity features. These interventions often introduce distribution shifts and entangle identity with non-identity attributes, degrading visual fidelity and data utility. To address this, we propose \textbf{ID\textsuperscript{2}Face}, a training-centric anonymization framework that removes the need for inference-time optimization. The rationale of our method is to learn a structured latent space where identity and non-identity information are explicitly disentangled, enabling direct and controllable anonymization at inference. To this end, we design a conditional diffusion model with an identity-masked learning scheme. An Identity-Decoupled Latent Recomposer uses an Identity Variational Autoencoder to model identity features, while non-identity attributes are extracted from same-identity pairs and aligned through bidirectional latent alignment. An Identity-Guided Latent Harmonizer then fuses these representations via soft-gating conditioned on noisy feature prediction. The model is trained with a recomposition-based reconstruction loss to enforce disentanglement. At inference, anonymization is achieved by sampling a random identity vector from the learned identity space. To further suppress identity leakage, we introduce an Orthogonal Identity Mapping strategy that enforces orthogonality between sampled and source identity vectors. Experiments demonstrate that ID\textsuperscript{2}Face outperforms existing methods in visual quality, identity suppression, and utility preservation.
>
---
#### [new 045] Beyond Objects: Contextual Synthetic Data Generation for Fine-Grained Classification
- **分类: cs.CV**

- **简介: 该论文针对细粒度分类中的低样本合成数据生成问题，提出BOB方法。通过提取并条件化类无关属性（如背景、姿态），在微调T2I模型时减少过拟合与类别混淆，生成更多样、高质量的合成数据。实验表明，该方法显著提升分类性能，在多个基准上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24078v1](http://arxiv.org/pdf/2510.24078v1)**

> **作者:** William Yang; Xindi Wu; Zhiwei Deng; Esin Tureci; Olga Russakovsky
>
> **摘要:** Text-to-image (T2I) models are increasingly used for synthetic dataset generation, but generating effective synthetic training data for classification remains challenging. Fine-tuning a T2I model with a few real examples can help improve the quality of synthetic training data; however, it may also cause overfitting and reduce diversity in the generated samples. We propose a fine-tuning strategy BOB (BeyondOBjects) to mitigate these concerns for fine-grained classification. Given a small set of real examples, we first extract class-agnostic attributes such as scene background and object pose. We then explicitly condition on these attributes during fine-tuning of the T2I model and marginalize them out during generation. This design mitigates overfitting, preserves the T2I model's generative prior, reduces estimation errors, and further minimizes unintended inter-class associations. Extensive experiments across multiple T2I models, backbones, and datasets show that our method achieves state-of-the-art performance in low-shot fine-grained classification when augmented with synthetic data. Concretely, BOB outperforms DataDream by 7.4% on the Aircraft dataset (from 50.0% to 57.4% when fine-tuning a CLIP classifier with five real images augmented with 100 synthetic images). In three of the four benchmarks, fine-tuning downstream models with 5 real images augmented with BOB achieves better performance than fine-tuning with 10 real images. Collectively, BOB outperforms prior art in 18 of 24 experimental settings, with 2+% accuracy improvements in 14 of these settings.
>
---
#### [new 046] Towards the Automatic Segmentation, Modeling and Meshing of the Aortic Vessel Tree from Multicenter Acquisitions: An Overview of the SEG.A. 2023 Segmentation of the Aorta Challenge
- **分类: cs.CV**

- **简介: 该论文聚焦于主动脉血管树自动分割任务，旨在解决临床影像分析中缺乏高质量共享数据的问题。通过发起SEG.A. 2023挑战，构建多中心公开数据集，评估深度学习算法性能，发现3D U-Net主导并验证模型集成优势，推动自动化分析工具向临床应用发展。**

- **链接: [http://arxiv.org/pdf/2510.24009v1](http://arxiv.org/pdf/2510.24009v1)**

> **作者:** Yuan Jin; Antonio Pepe; Gian Marco Melito; Yuxuan Chen; Yunsu Byeon; Hyeseong Kim; Kyungwon Kim; Doohyun Park; Euijoon Choi; Dosik Hwang; Andriy Myronenko; Dong Yang; Yufan He; Daguang Xu; Ayman El-Ghotni; Mohamed Nabil; Hossam El-Kady; Ahmed Ayyad; Amr Nasr; Marek Wodzinski; Henning Müller; Hyeongyu Kim; Yejee Shin; Abbas Khan; Muhammad Asad; Alexander Zolotarev; Caroline Roney; Anthony Mathur; Martin Benning; Gregory Slabaugh; Theodoros Panagiotis Vagenas; Konstantinos Georgas; George K. Matsopoulos; Jihan Zhang; Zhen Zhang; Liqin Huang; Christian Mayer; Heinrich Mächler; Jan Egger
>
> **摘要:** The automated analysis of the aortic vessel tree (AVT) from computed tomography angiography (CTA) holds immense clinical potential, but its development has been impeded by a lack of shared, high-quality data. We launched the SEG.A. challenge to catalyze progress in this field by introducing a large, publicly available, multi-institutional dataset for AVT segmentation. The challenge benchmarked automated algorithms on a hidden test set, with subsequent optional tasks in surface meshing for computational simulations. Our findings reveal a clear convergence on deep learning methodologies, with 3D U-Net architectures dominating the top submissions. A key result was that an ensemble of the highest-ranking algorithms significantly outperformed individual models, highlighting the benefits of model fusion. Performance was strongly linked to algorithmic design, particularly the use of customized post-processing steps, and the characteristics of the training data. This initiative not only establishes a new performance benchmark but also provides a lasting resource to drive future innovation toward robust, clinically translatable tools.
>
---
#### [new 047] ETC: training-free diffusion models acceleration with Error-aware Trend Consistency
- **分类: cs.CV**

- **简介: 该论文针对扩散模型采样效率低的问题，提出无需训练的加速框架ETC。通过引入误差感知的趋势一致性机制，利用去噪轨迹的连续性预测稳定方向，并自适应确定模型特异性误差阈值，实现多步重用下的高效且一致的生成，相比FLUX提升2.65倍速度，质量损失极小。**

- **链接: [http://arxiv.org/pdf/2510.24129v1](http://arxiv.org/pdf/2510.24129v1)**

> **作者:** Jiajian Xie; Hubery Yin; Chen Li; Zhou Zhao; Shengyu Zhang
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Diffusion models have achieved remarkable generative quality but remain bottlenecked by costly iterative sampling. Recent training-free methods accelerate diffusion process by reusing model outputs. However, these methods ignore denoising trends and lack error control for model-specific tolerance, leading to trajectory deviations under multi-step reuse and exacerbating inconsistencies in the generated results. To address these issues, we introduce Error-aware Trend Consistency (ETC), a framework that (1) introduces a consistent trend predictor that leverages the smooth continuity of diffusion trajectories, projecting historical denoising patterns into stable future directions and progressively distributing them across multiple approximation steps to achieve acceleration without deviating; (2) proposes a model-specific error tolerance search mechanism that derives corrective thresholds by identifying transition points from volatile semantic planning to stable quality refinement. Experiments show that ETC achieves a 2.65x acceleration over FLUX with negligible (-0.074 SSIM score) degradation of consistency.
>
---
#### [new 048] 50 Years of Water Body Monitoring: The Case of Qaraaoun Reservoir, Lebanon
- **分类: cs.CV**

- **简介: 该论文针对黎巴嫩最大水库Qaraaoun的储水监测难题，提出无需传感器的遥感监测方法。通过融合卫星影像、新水体分割指数与支持向量回归模型，实现近实时水面面积与体积估算，精度高（误差<1.5%，R²>0.98），解决了设备故障与维护难问题，可推广至其他水体，支撑长期环境与气候变化研究。**

- **链接: [http://arxiv.org/pdf/2510.24413v1](http://arxiv.org/pdf/2510.24413v1)**

> **作者:** Ali Ahmad Faour; Nabil Amacha; Ali J. Ghandour
>
> **摘要:** The sustainable management of the Qaraaoun Reservoir, the largest surface water body in Lebanon located in the Bekaa Plain, depends on reliable monitoring of its storage volume despite frequent sensor malfunctions and limited maintenance capacity. This study introduces a sensor-free approach that integrates open-source satellite imagery, advanced water-extent segmentation, and machine learning to estimate the reservoir surface area and volume in near real time. Sentinel-2 and Landsat images are processed, where surface water is delineated using a newly proposed water segmentation index. A machine learning model based on Support Vector Regression (SVR) is trained on a curated dataset that includes water surface area, water level, and water volume calculations using a reservoir bathymetry survey. The model is then able to estimate reservoir volume relying solely on surface area extracted from satellite imagery, without the need for ground measurements. Water segmentation using the proposed index aligns with ground truth for more than 95 percent of the shoreline. Hyperparameter tuning with GridSearchCV yields an optimized SVR performance with error under 1.5 percent of full reservoir capacity and coefficients of determination exceeding 0.98. These results demonstrate the robustness and cost-effectiveness of the method, offering a practical solution for continuous, sensor-independent monitoring of reservoir storage. The proposed methodology can be replicated for other water bodies, and the resulting 50 years of time-series data is valuable for research on climate change and environmental patterns.
>
---
#### [new 049] CLFSeg: A Fuzzy-Logic based Solution for Boundary Clarity and Uncertainty Reduction in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割中边界模糊与不确定性问题，提出CLFSeg框架，融合模糊逻辑与卷积网络，增强特征提取并降低噪声与模糊。通过改进损失函数处理类别不平衡，显著提升分割精度与效率，在多个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24202v1](http://arxiv.org/pdf/2510.24202v1)**

> **作者:** Anshul Kaushal; Kunal Jangid; Vinod K. Kurmi
>
> **备注:** The 36th British Machine Vision Conference (BMVC) 2025
>
> **摘要:** Accurate polyp and cardiac segmentation for early detection and treatment is essential for the diagnosis and treatment planning of cancer-like diseases. Traditional convolutional neural network (CNN) based models have represented limited generalizability, robustness, and inability to handle uncertainty, which affects the segmentation performance. To solve these problems, this paper introduces CLFSeg, an encoder-decoder based framework that aggregates the Fuzzy-Convolutional (FC) module leveraging convolutional layers and fuzzy logic. This module enhances the segmentation performance by identifying local and global features while minimizing the uncertainty, noise, and ambiguity in boundary regions, ensuring computing efficiency. In order to handle class imbalance problem while focusing on the areas of interest with tiny and boundary regions, binary cross-entropy (BCE) with dice loss is incorporated. Our proposed model exhibits exceptional performance on four publicly available datasets, including CVC-ColonDB, CVC-ClinicDB, EtisLaribPolypDB, and ACDC. Extensive experiments and visual studies show CLFSeg surpasses the existing SOTA performance and focuses on relevant regions of interest in anatomical structures. The proposed CLFSeg improves performance while ensuring computing efficiency, which makes it a potential solution for real-world medical diagnostic scenarios. Project page is available at https://visdomlab.github.io/CLFSeg/
>
---
#### [new 050] Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy
- **分类: cs.CV**

- **简介: 该论文针对图像恢复与目标检测级联框架中的稳定性问题，指出两者在Lipschitz连续性上的不匹配导致检测敏感性增强。提出LROD框架，通过在检测网络中引入恢复模块并正则化其Lipschitz连续性，实现两任务协同优化。实验表明，该方法提升了恶劣条件下的检测稳定性和精度。**

- **链接: [http://arxiv.org/pdf/2510.24232v1](http://arxiv.org/pdf/2510.24232v1)**

> **作者:** Qing Zhao; Weijian Deng; Pengxu Wei; ZiYi Dong; Hannan Lu; Xiangyang Ji; Liang Lin
>
> **备注:** NeurIPS 2025
>
> **摘要:** To improve detection robustness in adverse conditions (e.g., haze and low light), image restoration is commonly applied as a pre-processing step to enhance image quality for the detector. However, the functional mismatch between restoration and detection networks can introduce instability and hinder effective integration -- an issue that remains underexplored. We revisit this limitation through the lens of Lipschitz continuity, analyzing the functional differences between restoration and detection networks in both the input space and the parameter space. Our analysis shows that restoration networks perform smooth, continuous transformations, while object detectors operate with discontinuous decision boundaries, making them highly sensitive to minor perturbations. This mismatch introduces instability in traditional cascade frameworks, where even imperceptible noise from restoration is amplified during detection, disrupting gradient flow and hindering optimization. To address this, we propose Lipschitz-regularized object detection (LROD), a simple yet effective framework that integrates image restoration directly into the detector's feature learning, harmonizing the Lipschitz continuity of both tasks during training. We implement this framework as Lipschitz-regularized YOLO (LR-YOLO), extending seamlessly to existing YOLO detectors. Extensive experiments on haze and low-light benchmarks demonstrate that LR-YOLO consistently improves detection stability, optimization smoothness, and overall accuracy.
>
---
#### [new 051] Unsupervised Detection of Post-Stroke Brain Abnormalities
- **分类: cs.CV**

- **简介: 该论文属于医学图像异常检测任务，旨在解决传统方法难以捕捉卒中后非病灶性脑结构异常的问题。提出基于流的生成模型REFLECT，利用健康对照数据训练，实现对卒中后脑部异常（如萎缩、脑室扩大）的无监督检测，显著提升对非病灶异常的敏感性。**

- **链接: [http://arxiv.org/pdf/2510.24398v1](http://arxiv.org/pdf/2510.24398v1)**

> **作者:** Youwan Mahé; Elise Bannier; Stéphanie Leplaideur; Elisa Fromont; Francesca Galassi
>
> **摘要:** Post-stroke MRI not only delineates focal lesions but also reveals secondary structural changes, such as atrophy and ventricular enlargement. These abnormalities, increasingly recognised as imaging biomarkers of recovery and outcome, remain poorly captured by supervised segmentation methods. We evaluate REFLECT, a flow-based generative model, for unsupervised detection of both focal and non-lesional abnormalities in post-stroke patients. Using dual-expert central-slice annotations on ATLAS data, performance was assessed at the object level with Free-Response ROC analysis for anomaly maps. Two models were trained on lesion-free slices from stroke patients (ATLAS) and on healthy controls (IXI) to test the effect of training data. On ATLAS test subjects, the IXI-trained model achieved higher lesion segmentation (Dice = 0.37 vs 0.27) and improved sensitivity to non-lesional abnormalities (FROC = 0.62 vs 0.43). Training on fully healthy anatomy improves the modelling of normal variability, enabling broader and more reliable detection of structural abnormalities.
>
---
#### [new 052] Benchmarking Microsaccade Recognition with Event Cameras: A Novel Dataset and Evaluation
- **分类: cs.CV**

- **简介: 该论文针对微眼动识别任务，解决传统方法成本高、分辨率低的问题。构建首个基于事件相机的微眼动数据集，模拟0.5–2.0°角位移的七类微眼动，生成高精度事件流。提出Spiking-VGG16Flow模型，在SpikingJelly中实现90%平均准确率，验证了脉冲神经网络在细粒度运动识别中的潜力。**

- **链接: [http://arxiv.org/pdf/2510.24231v1](http://arxiv.org/pdf/2510.24231v1)**

> **作者:** Waseem Shariff; Timothy Hanley; Maciej Stec; Hossein Javidnia; Peter Corcoran
>
> **备注:** Accepted in British Machine Vision Conference (BMVC) 2025, Main Conference
>
> **摘要:** Microsaccades are small, involuntary eye movements vital for visual perception and neural processing. Traditional microsaccade studies typically use eye trackers or frame-based analysis, which, while precise, are costly and limited in scalability and temporal resolution. Event-based sensing offers a high-speed, low-latency alternative by capturing fine-grained spatiotemporal changes efficiently. This work introduces a pioneering event-based microsaccade dataset to support research on small eye movement dynamics in cognitive computing. Using Blender, we render high-fidelity eye movement scenarios and simulate microsaccades with angular displacements from 0.5 to 2.0 degrees, divided into seven distinct classes. These are converted to event streams using v2e, preserving the natural temporal dynamics of microsaccades, with durations ranging from 0.25 ms to 2.25 ms. We evaluate the dataset using Spiking-VGG11, Spiking-VGG13, and Spiking-VGG16, and propose Spiking-VGG16Flow, an optical-flow-enhanced variant implemented in SpikingJelly. The models achieve around 90 percent average accuracy, successfully classifying microsaccades by angular displacement, independent of event count or duration. These results demonstrate the potential of spiking neural networks for fine motion recognition and establish a benchmark for event-based vision research. The dataset, code, and trained models will be publicly available at https://waseemshariff126.github.io/microsaccades/ .
>
---
#### [new 053] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts
- **分类: cs.CV**

- **简介: 该论文针对文本生成图像模型的安全漏洞，提出AutoPrompt框架，通过大语言模型自动生成可绕过过滤机制的自然语言对抗性后缀。解决现有方法依赖白盒访问、生成无意义提示及易被拦截的问题，实现黑盒高效红队测试，显著提升零样本迁移能力与真实场景适用性。**

- **链接: [http://arxiv.org/pdf/2510.24034v1](http://arxiv.org/pdf/2510.24034v1)**

> **作者:** Yufan Liu; Wanqian Zhang; Huashan Chen; Lin Wang; Xiaojun Jia; Zheng Lin; Weiping Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Despite rapid advancements in text-to-image (T2I) models, their safety mechanisms are vulnerable to adversarial prompts, which maliciously generate unsafe images. Current red-teaming methods for proactively assessing such vulnerabilities usually require white-box access to T2I models, and rely on inefficient per-prompt optimization, as well as inevitably generate semantically meaningless prompts easily blocked by filters. In this paper, we propose APT (AutoPrompT), a black-box framework that leverages large language models (LLMs) to automatically generate human-readable adversarial suffixes for benign prompts. We first introduce an alternating optimization-finetuning pipeline between adversarial suffix optimization and fine-tuning the LLM utilizing the optimized suffix. Furthermore, we integrates a dual-evasion strategy in optimization phase, enabling the bypass of both perplexity-based filter and blacklist word filter: (1) we constrain the LLM generating human-readable prompts through an auxiliary LLM perplexity scoring, which starkly contrasts with prior token-level gibberish, and (2) we also introduce banned-token penalties to suppress the explicit generation of banned-tokens in blacklist. Extensive experiments demonstrate the excellent red-teaming performance of our human-readable, filter-resistant adversarial prompts, as well as superior zero-shot transferability which enables instant adaptation to unseen prompts and exposes critical vulnerabilities even in commercial APIs (e.g., Leonardo.Ai.).
>
---
#### [new 054] Decoupled MeanFlow: Turning Flow Models into Flow Maps for Accelerated Sampling
- **分类: cs.CV**

- **简介: 该论文针对扩散模型采样慢的问题，提出Decoupled MeanFlow方法，将预训练流模型无需修改架构地转化为流图模型，通过条件化最终层实现高效加速。仅需1-4步即可生成高质量图像，显著提升推理速度且性能超越现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24474v1](http://arxiv.org/pdf/2510.24474v1)**

> **作者:** Kyungmin Lee; Sihyun Yu; Jinwoo Shin
>
> **摘要:** Denoising generative models, such as diffusion and flow-based models, produce high-quality samples but require many denoising steps due to discretization error. Flow maps, which estimate the average velocity between timesteps, mitigate this error and enable faster sampling. However, their training typically demands architectural changes that limit compatibility with pretrained flow models. We introduce Decoupled MeanFlow, a simple decoding strategy that converts flow models into flow map models without architectural modifications. Our method conditions the final blocks of diffusion transformers on the subsequent timestep, allowing pretrained flow models to be directly repurposed as flow maps. Combined with enhanced training techniques, this design enables high-quality generation in as few as 1 to 4 steps. Notably, we find that training flow models and subsequently converting them is more efficient and effective than training flow maps from scratch. On ImageNet 256x256 and 512x512, our models attain 1-step FID of 2.16 and 2.12, respectively, surpassing prior art by a large margin. Furthermore, we achieve FID of 1.51 and 1.68 when increasing the steps to 4, which nearly matches the performance of flow models while delivering over 100x faster inference.
>
---
#### [new 055] UHKD: A Unified Framework for Heterogeneous Knowledge Distillation via Frequency-Domain Representations
- **分类: cs.CV**

- **简介: 该论文提出UHKD框架，解决异构模型间知识蒸馏因架构差异导致的语义不匹配问题。通过频域转换提取全局特征，利用特征变换与对齐模块实现跨架构中间层知识迁移，结合损失函数优化，显著提升异构场景下的模型压缩效果。**

- **链接: [http://arxiv.org/pdf/2510.24116v1](http://arxiv.org/pdf/2510.24116v1)**

> **作者:** Fengming Yu; Haiwei Pan; Kejia Zhang; Jian Guan; Haiying Jiang
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Knowledge distillation (KD) is an effective model compression technique that transfers knowledge from a high-performance teacher to a lightweight student, reducing cost while maintaining accuracy. In visual applications, where large-scale image models are widely used, KD enables efficient deployment. However, architectural diversity introduces semantic discrepancies that hinder the use of intermediate representations. Most existing KD methods are designed for homogeneous models and degrade in heterogeneous scenarios, especially when intermediate features are involved. Prior studies mainly focus on the logits space, making limited use of the semantic information in intermediate layers. To address this limitation, Unified Heterogeneous Knowledge Distillation (UHKD) is proposed as a framework that leverages intermediate features in the frequency domain for cross-architecture transfer. Fourier transform is applied to capture global feature information, alleviating representational discrepancies between heterogeneous teacher-student pairs. A Feature Transformation Module (FTM) produces compact frequency-domain representations of teacher features, while a learnable Feature Alignment Module (FAM) projects student features and aligns them via multi-level matching. Training is guided by a joint objective combining mean squared error on intermediate features with Kullback-Leibler divergence on logits. Experiments on CIFAR-100 and ImageNet-1K demonstrate gains of 5.59% and 0.83% over the latest method, highlighting UHKD as an effective approach for unifying heterogeneous representations and enabling efficient utilization of visual knowledge
>
---
#### [new 056] DogMo: A Large-Scale Multi-View RGB-D Dataset for 4D Canine Motion Recovery
- **分类: cs.CV**

- **简介: 该论文提出DogMo数据集，用于4D犬类运动恢复任务。针对现有数据缺乏多视角、真实3D信息及多样性的问题，构建了1.2k序列的多视图RGB-D数据集，并设计三阶段优化方法，实现高精度姿态与形状重建，推动计算机视觉与动物行为建模研究。**

- **链接: [http://arxiv.org/pdf/2510.24117v1](http://arxiv.org/pdf/2510.24117v1)**

> **作者:** Zan Wang; Siyu Chen; Luya Mo; Xinfeng Gao; Yuxin Shen; Lebin Ding; Wei Liang
>
> **备注:** 19 pages
>
> **摘要:** We present DogMo, a large-scale multi-view RGB-D video dataset capturing diverse canine movements for the task of motion recovery from images. DogMo comprises 1.2k motion sequences collected from 10 unique dogs, offering rich variation in both motion and breed. It addresses key limitations of existing dog motion datasets, including the lack of multi-view and real 3D data, as well as limited scale and diversity. Leveraging DogMo, we establish four motion recovery benchmark settings that support systematic evaluation across monocular and multi-view, RGB and RGB-D inputs. To facilitate accurate motion recovery, we further introduce a three-stage, instance-specific optimization pipeline that fits the SMAL model to the motion sequences. Our method progressively refines body shape and pose through coarse alignment, dense correspondence supervision, and temporal regularization. Our dataset and method provide a principled foundation for advancing research in dog motion recovery and open up new directions at the intersection of computer vision, computer graphics, and animal behavior modeling.
>
---
#### [new 057] Efficient Cost-and-Quality Controllable Arbitrary-scale Super-resolution with Fourier Constraints
- **分类: cs.CV**

- **简介: 该论文聚焦任意尺度超分辨率任务，针对现有方法因逐个预测傅里叶分量导致效率低、质量差的问题，提出联合预测多个傅里叶成分的新方法，有效提升重建质量和计算效率。**

- **链接: [http://arxiv.org/pdf/2510.23978v1](http://arxiv.org/pdf/2510.23978v1)**

> **作者:** Kazutoshi Akita; Norimichi Ukita
>
> **备注:** 9 pages
>
> **摘要:** Cost-and-Quality (CQ) controllability in arbitrary-scale super-resolution is crucial. Existing methods predict Fourier components one by one using a recurrent neural network. However, this approach leads to performance degradation and inefficiency due to independent prediction. This paper proposes predicting multiple components jointly to improve both quality and efficiency.
>
---
#### [new 058] When are radiology reports useful for training medical image classifiers?
- **分类: cs.CV**

- **简介: 该论文研究如何利用放射科报告提升医学图像分类模型性能。针对“何时及如何使用文本辅助训练图像模型”这一问题，系统评估了报告在预训练与微调阶段的作用，发现文本对标签强关联任务有益，且微调阶段使用报告效果显著。**

- **链接: [http://arxiv.org/pdf/2510.24385v1](http://arxiv.org/pdf/2510.24385v1)**

> **作者:** Herman Bergström; Zhongqi Yue; Fredrik D. Johansson
>
> **摘要:** Medical images used to train machine learning models are often accompanied by radiology reports containing rich expert annotations. However, relying on these reports as inputs for clinical prediction requires the timely manual work of a trained radiologist. This raises a natural question: when can radiology reports be leveraged during training to improve image-only classification? Prior works are limited to evaluating pre-trained image representations by fine-tuning them to predict diagnostic labels, often extracted from reports, ignoring tasks with labels that are weakly associated with the text. To address this gap, we conduct a systematic study of how radiology reports can be used during both pre-training and fine-tuning, across diagnostic and prognostic tasks (e.g., 12-month readmission), and under varying training set sizes. Our findings reveal that: (1) Leveraging reports during pre-training is beneficial for downstream classification tasks where the label is well-represented in the text; however, pre-training through explicit image-text alignment can be detrimental in settings where it's not; (2) Fine-tuning with reports can lead to significant improvements and even have a larger impact than the pre-training method in certain settings. These results provide actionable insights into when and how to leverage privileged text data to train medical image classifiers while highlighting gaps in current research.
>
---
#### [new 059] MC-SJD : Maximal Coupling Speculative Jacobi Decoding for Autoregressive Visual Generation Acceleration
- **分类: cs.CV**

- **简介: 该论文针对自回归视觉生成中推理速度慢的问题，提出MC-SJD框架。通过信息论耦合机制最大化连续迭代间草稿令牌的一致性，提升接受率，实现无损并行解码。仅需一行代码修改，即在图像生成上提速4.2倍，视频生成上提速13.3倍。**

- **链接: [http://arxiv.org/pdf/2510.24211v1](http://arxiv.org/pdf/2510.24211v1)**

> **作者:** Junhyuk So; Hyunho Kook; Chaeyeon Jang; Eunhyeok Park
>
> **摘要:** While autoregressive (AR) modeling has recently emerged as a new paradigm in visual generation, its practical adoption is severely constrained by the slow inference speed of per-token generation, which often requires thousands of steps to produce a single sample. To address this challenge, we propose MC-SJD, a training-free, lossless parallel decoding framework designed to accelerate AR visual generation by extending the recently introduced Speculative Jacobi Decoding (SJD). Although SJD shows strong potential for accelerating AR generation, we demonstrate that token instability across iterations significantly reduces the acceptance rate, a limitation that primarily arises from the independent sampling process used during draft token generation. To overcome this, we introduce MC-SJD, an information-theoretic approach based on coupling, which substantially accelerates standard SJD by maximizing the probability of sampling identical draft tokens across consecutive iterations, all while preserving its lossless property. Remarkably, this method requires only a single-line modification to the existing algorithm, yet achieves substantial performance gains, delivering up to a ~4.2x acceleration in image generation and ~13.3x acceleration in video generation compared to standard AR decoding, without any degradation in output quality.
>
---
#### [new 060] Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2
- **分类: cs.CV**

- **简介: 该论文针对视频图像分割模型SAM2的鲁棒性问题，提出首个跨提示通用对抗攻击方法UAP-SAM2。针对提示引导与帧间语义耦合带来的挑战，设计目标扫描策略与双语义偏差框架，提升攻击迁移性与有效性，在多个数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24195v1](http://arxiv.org/pdf/2510.24195v1)**

> **作者:** Ziqi Zhou; Yifan Hu; Yufei Song; Zijing Li; Shengshan Hu; Leo Yu Zhang; Dezhong Yao; Long Zheng; Hai Jin
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent studies reveal the vulnerability of the image segmentation foundation model SAM to adversarial examples. Its successor, SAM2, has attracted significant attention due to its strong generalization capability in video segmentation. However, its robustness remains unexplored, and it is unclear whether existing attacks on SAM can be directly transferred to SAM2. In this paper, we first analyze the performance gap of existing attacks between SAM and SAM2 and highlight two key challenges arising from their architectural differences: directional guidance from the prompt and semantic entanglement across consecutive frames. To address these issues, we propose UAP-SAM2, the first cross-prompt universal adversarial attack against SAM2 driven by dual semantic deviation. For cross-prompt transferability, we begin by designing a target-scanning strategy that divides each frame into k regions, each randomly assigned a prompt, to reduce prompt dependency during optimization. For effectiveness, we design a dual semantic deviation framework that optimizes a UAP by distorting the semantics within the current frame and disrupting the semantic consistency across consecutive frames. Extensive experiments on six datasets across two segmentation tasks demonstrate the effectiveness of the proposed method for SAM2. The comparative results show that UAP-SAM2 significantly outperforms state-of-the-art (SOTA) attacks by a large margin.
>
---
#### [new 061] A Critical Study towards the Detection of Parkinsons Disease using ML Technologies
- **分类: cs.CV**

- **简介: 该论文属于目标检测与分割任务，旨在识别茶树叶病（红锈病、Helopeltis害虫、红蜘蛛螨）。采用SSD MobileNet V2、Faster R-CNN ResNet50 V1进行检测，用Mask R-CNN实现病变区域分割，优化了病害面积计算方法。Faster R-CNN表现更优，mAP达25%。**

- **链接: [http://arxiv.org/pdf/2510.24456v1](http://arxiv.org/pdf/2510.24456v1)**

> **作者:** Vivek Chetia; Abdul Taher Khan; Rahish Gogoi; David Kapsian Khual; Purnendu Bikash; Sajal Saha
>
> **摘要:** The proposed solution is Deep Learning Technique that will be able classify three types of tea leaves diseases from which two diseases are caused by the pests and one due to pathogens (infectious organisms) and environmental conditions and also show the area damaged by a disease in leaves. Namely Red Rust, Helopeltis and Red spider mite respectively. In this paper we have evaluated two models namely SSD MobileNet V2 and Faster R-CNN ResNet50 V1 for the object detection. The SSD MobileNet V2 gave precision of 0.209 for IOU range of 0.50:0.95 with recall of 0.02 on IOU 0.50:0.95 and final mAP of 20.9%. While Faster R-CNN ResNet50 V1 has precision of 0.252 on IOU range of 0.50:0.95 and recall of 0.044 on IOU of 0.50:0.95 with a mAP of 25%, which is better than SSD. Also used Mask R-CNN for Object Instance Segmentation where we have implemented our custom method to calculate the damaged diseased portion of leaves. Keywords: Tea Leaf Disease, Deep Learning, Red Rust, Helopeltis and Red Spider Mite, SSD MobileNet V2, Faster R-CNN ResNet50 V1 and Mask RCNN.
>
---
#### [new 062] RareFlow: Physics-Aware Flow-Matching for Cross-Sensor Super-Resolution of Rare-Earth Features
- **分类: cs.CV**

- **简介: 该论文针对遥感图像超分辨率在分布外（OOD）场景下物理不准确的问题，提出RareFlow框架。通过双条件控制架构与多维度物理损失，融合几何保真与语义引导，实现跨传感器稀有地物的高保真重建，并量化预测不确定性以减少幻觉。**

- **链接: [http://arxiv.org/pdf/2510.23816v1](http://arxiv.org/pdf/2510.23816v1)**

> **作者:** Forouzan Fallah; Wenwen Li; Chia-Yu Hsu; Hyunho Lee; Yezhou Yang
>
> **摘要:** Super-resolution (SR) for remote sensing imagery often fails under out-of-distribution (OOD) conditions, such as rare geomorphic features captured by diverse sensors, producing visually plausible but physically inaccurate results. We present RareFlow, a physics-aware SR framework designed for OOD robustness. RareFlow's core is a dual-conditioning architecture. A Gated ControlNet preserves fine-grained geometric fidelity from the low-resolution input, while textual prompts provide semantic guidance for synthesizing complex features. To ensure physically sound outputs, we introduce a multifaceted loss function that enforces both spectral and radiometric consistency with sensor properties. Furthermore, the framework quantifies its own predictive uncertainty by employing a stochastic forward pass approach; the resulting output variance directly identifies unfamiliar inputs, mitigating feature hallucination. We validate RareFlow on a new, curated benchmark of multi-sensor satellite imagery. In blind evaluations, geophysical experts rated our model's outputs as approaching the fidelity of ground truth imagery, significantly outperforming state-of-the-art baselines. This qualitative superiority is corroborated by quantitative gains in perceptual metrics, including a nearly 40\% reduction in FID. RareFlow provides a robust framework for high-fidelity synthesis in data-scarce scientific domains and offers a new paradigm for controlled generation under severe domain shift.
>
---
#### [new 063] Enhancing Pre-trained Representation Classifiability can Boost its Interpretability
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究预训练视觉模型的表示可解释性与分类能力的关系。提出固有可解释性评分（IIS）量化可解释性，发现二者正相关。通过最大化可解释性微调模型，可在提升分类性能的同时保持预测准确性，实现两者协同优化。**

- **链接: [http://arxiv.org/pdf/2510.24105v1](http://arxiv.org/pdf/2510.24105v1)**

> **作者:** Shufan Shen; Zhaobo Qi; Junshu Sun; Qingming Huang; Qi Tian; Shuhui Wang
>
> **备注:** ICLR 2025 (Spotlight)
>
> **摘要:** The visual representation of a pre-trained model prioritizes the classifiability on downstream tasks, while the widespread applications for pre-trained visual models have posed new requirements for representation interpretability. However, it remains unclear whether the pre-trained representations can achieve high interpretability and classifiability simultaneously. To answer this question, we quantify the representation interpretability by leveraging its correlation with the ratio of interpretable semantics within the representations. Given the pre-trained representations, only the interpretable semantics can be captured by interpretations, whereas the uninterpretable part leads to information loss. Based on this fact, we propose the Inherent Interpretability Score (IIS) that evaluates the information loss, measures the ratio of interpretable semantics, and quantifies the representation interpretability. In the evaluation of the representation interpretability with different classifiability, we surprisingly discover that the interpretability and classifiability are positively correlated, i.e., representations with higher classifiability provide more interpretable semantics that can be captured in the interpretations. This observation further supports two benefits to the pre-trained representations. First, the classifiability of representations can be further improved by fine-tuning with interpretability maximization. Second, with the classifiability improvement for the representations, we obtain predictions based on their interpretations with less accuracy degradation. The discovered positive correlation and corresponding applications show that practitioners can unify the improvements in interpretability and classifiability for pre-trained vision models. Codes are available at https://github.com/ssfgunner/IIS.
>
---
#### [new 064] Training-free Source Attribution of AI-generated Images via Resynthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对少样本/零样本条件下AI生成图像的来源归属问题，提出一种无需训练的基于图像重合成的溯源方法。通过生成图像描述并用各候选模型重合成图像，以特征空间相似度判定源模型。同时构建了包含商业与开源模型生成人脸的新数据集，为评估新方法提供基准。**

- **链接: [http://arxiv.org/pdf/2510.24278v1](http://arxiv.org/pdf/2510.24278v1)**

> **作者:** Pietro Bongini; Valentina Molinari; Andrea Costanzo; Benedetta Tondi; Mauro Barni
>
> **备注:** 14 pages, 4 figures, 1 table, accepted at "The 17th IEEE INTERNATIONAL WORKSHOP ON INFORMATION FORENSICS AND SECURITY (WIFS2025)", Perth, Australia
>
> **摘要:** Synthetic image source attribution is a challenging task, especially in data scarcity conditions requiring few-shot or zero-shot classification capabilities. We present a new training-free one-shot attribution method based on image resynthesis. A prompt describing the image under analysis is generated, then it is used to resynthesize the image with all the candidate sources. The image is attributed to the model which produced the resynthesis closest to the original image in a proper feature space. We also introduce a new dataset for synthetic image attribution consisting of face images from commercial and open-source text-to-image generators. The dataset provides a challenging attribution framework, useful for developing new attribution models and testing their capabilities on different generative architectures. The dataset structure allows to test approaches based on resynthesis and to compare them to few-shot methods. Results from state-of-the-art few-shot approaches and other baselines show that the proposed resynthesis method outperforms existing techniques when only a few samples are available for training or fine-tuning. The experiments also demonstrate that the new dataset is a challenging one and represents a valuable benchmark for developing and evaluating future few-shot and zero-shot methods.
>
---
#### [new 065] Adaptive Knowledge Transferring with Switching Dual-Student Framework for Semi-Supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对半监督医学图像分割任务，解决教师-学生框架中知识传递不可靠、易误差累积的问题。提出切换双学生架构，动态选择最优学生协同学习，并引入损失感知指数移动平均策略，提升伪标签质量，显著增强分割精度。**

- **链接: [http://arxiv.org/pdf/2510.24366v1](http://arxiv.org/pdf/2510.24366v1)**

> **作者:** Thanh-Huy Nguyen; Hoang-Thien Nguyen; Ba-Thinh Lam; Vi Vu; Bach X. Nguyen; Jianhua Xing; Tianyang Wang; Xingjian Li; Min Xu
>
> **备注:** The paper is under review at Pattern Recognition Journal
>
> **摘要:** Teacher-student frameworks have emerged as a leading approach in semi-supervised medical image segmentation, demonstrating strong performance across various tasks. However, the learning effects are still limited by the strong correlation and unreliable knowledge transfer process between teacher and student networks. To overcome this limitation, we introduce a novel switching Dual-Student architecture that strategically selects the most reliable student at each iteration to enhance dual-student collaboration and prevent error reinforcement. We also introduce a strategy of Loss-Aware Exponential Moving Average to dynamically ensure that the teacher absorbs meaningful information from students, improving the quality of pseudo-labels. Our plug-and-play framework is extensively evaluated on 3D medical image segmentation datasets, where it outperforms state-of-the-art semi-supervised methods, demonstrating its effectiveness in improving segmentation accuracy under limited supervision.
>
---
#### [new 066] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出UtilGen框架，解决传统数据增强忽视下游任务需求的问题。通过双层优化机制，基于任务反馈动态调整生成策略，使合成数据更契合任务需求。实验表明，该方法在8个基准数据集上显著提升模型性能，平均准确率提高3.87%，实现从视觉质量导向到任务效用导向的范式转变。**

- **链接: [http://arxiv.org/pdf/2510.24262v1](http://arxiv.org/pdf/2510.24262v1)**

> **作者:** Jiyu Guo; Shuo Yang; Yiming Huang; Yancheng Long; Xiaobo Xia; Xiu Su; Bo Zhao; Zeke Xie; Liqiang Nie
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Data augmentation using generative models has emerged as a powerful paradigm for enhancing performance in computer vision tasks. However, most existing augmentation approaches primarily focus on optimizing intrinsic data attributes -- such as fidelity and diversity -- to generate visually high-quality synthetic data, while often neglecting task-specific requirements. Yet, it is essential for data generators to account for the needs of downstream tasks, as training data requirements can vary significantly across different tasks and network architectures. To address these limitations, we propose UtilGen, a novel utility-centric data augmentation framework that adaptively optimizes the data generation process to produce task-specific, high-utility training data via downstream task feedback. Specifically, we first introduce a weight allocation network to evaluate the task-specific utility of each synthetic sample. Guided by these evaluations, UtilGen iteratively refines the data generation process using a dual-level optimization strategy to maximize the synthetic data utility: (1) model-level optimization tailors the generative model to the downstream task, and (2) instance-level optimization adjusts generation policies -- such as prompt embeddings and initial noise -- at each generation round. Extensive experiments on eight benchmark datasets of varying complexity and granularity demonstrate that UtilGen consistently achieves superior performance, with an average accuracy improvement of 3.87% over previous SOTA. Further analysis of data influence and distribution reveals that UtilGen produces more impactful and task-relevant synthetic data, validating the effectiveness of the paradigm shift from visual characteristics-centric to task utility-centric data augmentation.
>
---
#### [new 067] CountFormer: A Transformer Framework for Learning Visual Repetition and Structure in Class-Agnostic Object Counting
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CountFormer，一种用于无类别物体计数的Transformer框架。针对现有模型在复杂结构或密集场景下计数不准的问题，引入DINOv2作为编码器并融合位置嵌入，提升对视觉重复与结构关系的感知能力。实验表明，该方法在FSC-147数据集上表现优异，更接近人类计数方式。**

- **链接: [http://arxiv.org/pdf/2510.23785v1](http://arxiv.org/pdf/2510.23785v1)**

> **作者:** Md Tanvir Hossain; Akif Islam; Mohd Ruhul Ameen
>
> **备注:** 6 pages, 2 tables, 6 figures. Submitted to IEEE 5th International Conference on Electrical, Computer and Telecommunication Engineering (ICECTE 2025)
>
> **摘要:** Humans can effortlessly count diverse objects by perceiving visual repetition and structural relationships rather than relying on class identity. However, most existing counting models fail to replicate this ability; they often miscount when objects exhibit complex shapes, internal symmetry, or overlapping components. In this work, we introduce CountFormer, a transformer-based framework that learns to recognize repetition and structural coherence for class-agnostic object counting. Built upon the CounTR architecture, our model replaces its visual encoder with the self-supervised foundation model DINOv2, which produces richer and spatially consistent feature representations. We further incorporate positional embedding fusion to preserve geometric relationships before decoding these features into density maps through a lightweight convolutional decoder. Evaluated on the FSC-147 dataset, our model achieves performance comparable to current state-of-the-art methods while demonstrating superior accuracy on structurally intricate or densely packed scenes. Our findings indicate that integrating foundation models such as DINOv2 enables counting systems to approach human-like structural perception, advancing toward a truly general and exemplar-free counting paradigm.
>
---
#### [new 068] MSRANetV2: An Explainable Deep Learning Architecture for Multi-class Classification of Colorectal Histopathological Images
- **分类: cs.CV**

- **简介: 该论文提出MSRANetV2模型，用于结直肠癌组织病理图像的多分类任务。针对传统诊断主观、耗时的问题，通过融合残差注意力与SE模块，实现多尺度特征融合，提升分类精度与可解释性。在两个公开数据集上均取得优异性能，并结合Grad-CAM增强模型可解释性。**

- **链接: [http://arxiv.org/pdf/2510.24136v1](http://arxiv.org/pdf/2510.24136v1)**

> **作者:** Ovi Sarkar; Md Shafiuzzaman; Md. Faysal Ahamed; Golam Mahmud; Muhammad E. H. Chowdhury
>
> **摘要:** Colorectal cancer (CRC) is a leading worldwide cause of cancer-related mortality, and the role of prompt precise detection is of paramount interest in improving patient outcomes. Conventional diagnostic methods such as colonoscopy and histological examination routinely exhibit subjectivity, are extremely time-consuming, and are susceptible to variation. Through the development of digital pathology, deep learning algorithms have become a powerful approach in enhancing diagnostic precision and efficiency. In our work, we proposed a convolutional neural network architecture named MSRANetV2, specially optimized for the classification of colorectal tissue images. The model employs a ResNet50V2 backbone, extended with residual attention mechanisms and squeeze-and-excitation (SE) blocks, to extract deep semantic and fine-grained spatial features. With channel alignment and upsampling operations, MSRANetV2 effectively fuses multi-scale representations, thereby enhancing the robustness of the classification. We evaluated our model on a five-fold stratified cross-validation strategy on two publicly available datasets: CRC-VAL-HE-7K and NCT-CRC-HE-100K. The proposed model achieved remarkable average Precision, recall, F1-score, AUC, and test accuracy were 0.9884 plus-minus 0.0151, 0.9900 plus-minus 0.0151, 0.9900 plus-minus 0.0145, 0.9999 plus-minus 0.00006, and 0.9905 plus-minus 0.0025 on the 7K dataset. On the 100K dataset, they were 0.9904 plus-minus 0.0091, 0.9900 plus-minus 0.0071, 0.9900 plus-minus 0.0071, 0.9997 plus-minus 0.00016, and 0.9902 plus-minus 0.0006. Additionally, Grad-CAM visualizations were incorporated to enhance model interpretability by highlighting tissue areas that are medically relevant. These findings validate that MSRANetV2 is a reliable, interpretable, and high-performing architectural model for classifying CRC tissues.
>
---
#### [new 069] SafeVision: Efficient Image Guardrail with Robust Policy Adherence and Explainability
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文提出SafeVision，一种高效、可解释的图像安全防护模型。针对传统模型分类不准、难适应新威胁的问题，引入类人推理与动态政策对齐机制，结合自建数据集VisionHarm，实现精准风险识别与解释，显著优于现有方法且速度更快。**

- **链接: [http://arxiv.org/pdf/2510.23960v1](http://arxiv.org/pdf/2510.23960v1)**

> **作者:** Peiyang Xu; Minzhou Pan; Zhaorun Chen; Shuang Yang; Chaowei Xiao; Bo Li
>
> **备注:** 42 pages, 9 figures
>
> **摘要:** With the rapid proliferation of digital media, the need for efficient and transparent safeguards against unsafe content is more critical than ever. Traditional image guardrail models, constrained by predefined categories, often misclassify content due to their pure feature-based learning without semantic reasoning. Moreover, these models struggle to adapt to emerging threats, requiring costly retraining for new threats. To address these limitations, we introduce SafeVision, a novel image guardrail that integrates human-like reasoning to enhance adaptability and transparency. Our approach incorporates an effective data collection and generation framework, a policy-following training pipeline, and a customized loss function. We also propose a diverse QA generation and training strategy to enhance learning effectiveness. SafeVision dynamically aligns with evolving safety policies at inference time, eliminating the need for retraining while ensuring precise risk assessments and explanations. Recognizing the limitations of existing unsafe image benchmarks, which either lack granularity or cover limited risks, we introduce VisionHarm, a high-quality dataset comprising two subsets: VisionHarm Third-party (VisionHarm-T) and VisionHarm Comprehensive(VisionHarm-C), spanning diverse harmful categories. Through extensive experiments, we show that SafeVision achieves state-of-the-art performance on different benchmarks. SafeVision outperforms GPT-4o by 8.6% on VisionHarm-T and by 15.5% on VisionHarm-C, while being over 16x faster. SafeVision sets a comprehensive, policy-following, and explainable image guardrail with dynamic adaptation to emerging threats.
>
---
#### [new 070] GenTrack: A New Generation of Multi-Object Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GenTrack，一种新型多目标跟踪方法，旨在解决目标数量未知、动态变化及遮挡下的身份一致性问题。通过融合随机与确定性追踪，引入粒子群优化与社会交互建模，提升弱检测条件下的跟踪鲁棒性，并构建了包含三种变体的开源基准实现，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24399v1](http://arxiv.org/pdf/2510.24399v1)**

> **作者:** Toan Van Nguyen; Rasmus G. K. Christiansen; Dirk Kraft; Leon Bodenhagen
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper introduces a novel multi-object tracking (MOT) method, dubbed GenTrack, whose main contributions include: a hybrid tracking approach employing both stochastic and deterministic manners to robustly handle unknown and time-varying numbers of targets, particularly in maintaining target identity (ID) consistency and managing nonlinear dynamics, leveraging particle swarm optimization (PSO) with some proposed fitness measures to guide stochastic particles toward their target distribution modes, enabling effective tracking even with weak and noisy object detectors, integration of social interactions among targets to enhance PSO-guided particles as well as improve continuous updates of both strong (matched) and weak (unmatched) tracks, thereby reducing ID switches and track loss, especially during occlusions, a GenTrack-based redefined visual MOT baseline incorporating a comprehensive state and observation model based on space consistency, appearance, detection confidence, track penalties, and social scores for systematic and efficient target updates, and the first-ever publicly available source-code reference implementation with minimal dependencies, featuring three variants, including GenTrack Basic, PSO, and PSO-Social, facilitating flexible reimplementation. Experimental results have shown that GenTrack provides superior performance on standard benchmarks and real-world scenarios compared to state-of-the-art trackers, with integrated implementations of baselines for fair comparison. Potential directions for future work are also discussed. The source-code reference implementations of both the proposed method and compared-trackers are provided on GitHub: https://github.com/SDU-VelKoTek/GenTrack
>
---
#### [new 071] A Hybrid Approach for Visual Multi-Object Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种混合视觉多目标跟踪方法，融合随机与确定性机制，解决未知且动态变化目标数下的身份一致性问题。通过粒子滤波结合粒子群优化处理非线性动态与噪声，利用时空一致性和外观相似性提升跟踪精度；通过确定性关联与轨迹平滑更新策略，增强弱跟踪鲁棒性，适用于视频与实时流。**

- **链接: [http://arxiv.org/pdf/2510.24410v1](http://arxiv.org/pdf/2510.24410v1)**

> **作者:** Toan Van Nguyen; Rasmus G. K. Christiansen; Dirk Kraft; Leon Bodenhagen
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper proposes a visual multi-object tracking method that jointly employs stochastic and deterministic mechanisms to ensure identifier consistency for unknown and time-varying target numbers under nonlinear dynamics. A stochastic particle filter addresses nonlinear dynamics and non-Gaussian noise, with support from particle swarm optimization (PSO) to guide particles toward state distribution modes and mitigate divergence through proposed fitness measures incorporating motion consistency, appearance similarity, and social-interaction cues with neighboring targets. Deterministic association further enforces identifier consistency via a proposed cost matrix incorporating spatial consistency between particles and current detections, detection confidences, and track penalties. Subsequently, a novel scheme is proposed for the smooth updating of target states while preserving their identities, particularly for weak tracks during interactions with other targets and prolonged occlusions. Moreover, velocity regression over past states provides trend-seed velocities, enhancing particle sampling and state updates. The proposed tracker is designed to operate flexibly for both pre-recorded videos and camera live streams, where future frames are unavailable. Experimental results confirm superior performance compared to state-of-the-art trackers. The source-code reference implementations of both the proposed method and compared-trackers are provided on GitHub: https://github.com/SDU-VelKoTek/GenTrack2
>
---
#### [new 072] Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Latent Sketchpad框架，解决MLLMs在复杂视觉规划与想象任务中表现不足的问题。通过引入内嵌视觉思维的生成机制，使模型在推理过程中交替进行文本与视觉潜表示生成，提升多模态推理能力，并实现可解释的草图输出。**

- **链接: [http://arxiv.org/pdf/2510.24514v1](http://arxiv.org/pdf/2510.24514v1)**

> **作者:** Huanyu Zhang; Wenshan Wu; Chengzu Li; Ning Shang; Yan Xia; Yangyu Huang; Yifan Zhang; Li Dong; Zhang Zhang; Liang Wang; Tieniu Tan; Furu Wei
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at visual understanding, they often struggle in complex scenarios that require visual planning and imagination. Inspired by how humans use sketching as a form of visual thinking to develop and communicate ideas, we introduce Latent Sketchpad, a framework that equips MLLMs with an internal visual scratchpad. The internal visual representations of MLLMs have traditionally been confined to perceptual understanding. We repurpose them to support generative visual thought without compromising reasoning ability. Building on frontier MLLMs, our approach integrates visual generation directly into their native autoregressive reasoning process. It allows the model to interleave textual reasoning with the generation of visual latents. These latents guide the internal thought process and can be translated into sketch images for interpretability. To realize this, we introduce two components: a Context-Aware Vision Head autoregressively produces visual representations, and a pretrained Sketch Decoder renders these into human-interpretable images. We evaluate the framework on our new dataset MazePlanning. Experiments across various MLLMs show that Latent Sketchpad delivers comparable or even superior reasoning performance to their backbone. It further generalizes across distinct frontier MLLMs, including Gemma3 and Qwen2.5-VL. By extending model's textual reasoning to visual thinking, our framework opens new opportunities for richer human-computer interaction and broader applications. More details and resources are available on our project page: https://latent-sketchpad.github.io/.
>
---
#### [new 073] SPARTA: Evaluating Reasoning Segmentation Robustness through Black-Box Adversarial Paraphrasing in Text Autoencoder Latent Space
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对视觉语言任务中的推理分割，研究文本语义不变但能降低模型性能的对抗性改写问题。提出SPARTA方法，在文本自编码器隐空间中通过强化学习生成高保真对抗性改写，有效提升攻击成功率，揭示当前模型对语义等价文本扰动的脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.24446v1](http://arxiv.org/pdf/2510.24446v1)**

> **作者:** Viktoriia Zinkovich; Anton Antonov; Andrei Spiridonov; Denis Shepelev; Andrey Moskalenko; Daria Pugacheva; Elena Tutubalina; Andrey Kuznetsov; Vlad Shakhuro
>
> **摘要:** Multimodal large language models (MLLMs) have shown impressive capabilities in vision-language tasks such as reasoning segmentation, where models generate segmentation masks based on textual queries. While prior work has primarily focused on perturbing image inputs, semantically equivalent textual paraphrases-crucial in real-world applications where users express the same intent in varied ways-remain underexplored. To address this gap, we introduce a novel adversarial paraphrasing task: generating grammatically correct paraphrases that preserve the original query meaning while degrading segmentation performance. To evaluate the quality of adversarial paraphrases, we develop a comprehensive automatic evaluation protocol validated with human studies. Furthermore, we introduce SPARTA-a black-box, sentence-level optimization method that operates in the low-dimensional semantic latent space of a text autoencoder, guided by reinforcement learning. SPARTA achieves significantly higher success rates, outperforming prior methods by up to 2x on both the ReasonSeg and LLMSeg-40k datasets. We use SPARTA and competitive baselines to assess the robustness of advanced reasoning segmentation models. We reveal that they remain vulnerable to adversarial paraphrasing-even under strict semantic and grammatical constraints. All code and data will be released publicly upon acceptance.
>
---
#### [new 074] What do vision-language models see in the context? Investigating multimodal in-context learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究视觉语言模型（VLMs）在上下文学习（ICL）中的表现，针对多模态ICL效果不佳的问题，系统评估7个VLMs在图像描述任务上的表现，分析提示设计、架构与训练策略的影响，并揭示当前模型过度依赖文本线索、忽视视觉信息的局限性。**

- **链接: [http://arxiv.org/pdf/2510.24331v1](http://arxiv.org/pdf/2510.24331v1)**

> **作者:** Gabriel O. dos Santos; Esther Colombini; Sandra Avila
>
> **摘要:** In-context learning (ICL) enables Large Language Models (LLMs) to learn tasks from demonstration examples without parameter updates. Although it has been extensively studied in LLMs, its effectiveness in Vision-Language Models (VLMs) remains underexplored. In this work, we present a systematic study of ICL in VLMs, evaluating seven models spanning four architectures on three image captioning benchmarks. We analyze how prompt design, architectural choices, and training strategies influence multimodal ICL. To our knowledge, we are the first to analyze how attention patterns in VLMs vary with an increasing number of in-context demonstrations. Our results reveal that training on imag-text interleaved data enhances ICL performance but does not imply effective integration of visual and textual information from demonstration examples. In contrast, instruction tuning improves instruction-following but can reduce reliance on in-context demonstrations, suggesting a trade-off between instruction alignment and in-context adaptation. Attention analyses further show that current VLMs primarily focus on textual cues and fail to leverage visual information, suggesting a limited capacity for multimodal integration. These findings highlight key limitations in the ICL abilities of current VLMs and provide insights for enhancing their ability to learn from multimodal in-context examples.
>
---
#### [new 075] ZTRS: Zero-Imitation End-to-end Autonomous Driving with Trajectory Scoring
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ZTRS框架，解决端到端自动驾驶中依赖模仿学习导致的性能瓶颈问题。通过纯强化学习结合轨迹评分机制，直接从高维传感器数据学习驾驶策略，无需专家示范，实现零模仿、全端到端的鲁棒规划，在多个基准上达到领先性能。**

- **链接: [http://arxiv.org/pdf/2510.24108v1](http://arxiv.org/pdf/2510.24108v1)**

> **作者:** Zhenxin Li; Wenhao Yao; Zi Wang; Xinglong Sun; Jingde Chen; Nadine Chang; Maying Shen; Jingyu Song; Zuxuan Wu; Shiyi Lan; Jose M. Alvarez
>
> **摘要:** End-to-end autonomous driving maps raw sensor inputs directly into ego-vehicle trajectories to avoid cascading errors from perception modules and to leverage rich semantic cues. Existing frameworks largely rely on Imitation Learning (IL), which can be limited by sub-optimal expert demonstrations and covariate shift during deployment. On the other hand, Reinforcement Learning (RL) has recently shown potential in scaling up with simulations, but is typically confined to low-dimensional symbolic inputs (e.g. 3D objects and maps), falling short of full end-to-end learning from raw sensor data. We introduce ZTRS (Zero-Imitation End-to-End Autonomous Driving with Trajectory Scoring), a framework that combines the strengths of both worlds: sensor inputs without losing information and RL training for robust planning. To the best of our knowledge, ZTRS is the first framework that eliminates IL entirely by only learning from rewards while operating directly on high-dimensional sensor data. ZTRS utilizes offline reinforcement learning with our proposed Exhaustive Policy Optimization (EPO), a variant of policy gradient tailored for enumerable actions and rewards. ZTRS demonstrates strong performance across three benchmarks: Navtest (generic real-world open-loop planning), Navhard (open-loop planning in challenging real-world and synthetic scenarios), and HUGSIM (simulated closed-loop driving). Specifically, ZTRS achieves the state-of-the-art result on Navhard and outperforms IL-based baselines on HUGSIM. Code will be available at https://github.com/woxihuanjiangguo/ZTRS.
>
---
#### [new 076] RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出RoboOmni框架，面向多模态情境下的主动机器人操作任务。针对现实交互中用户不常下达明确指令的问题，通过融合语音、视觉与环境声音，实现意图的主动推断与协作。构建了包含140k场景的OmniAction数据集，推动了无需显式命令的机器人智能发展。**

- **链接: [http://arxiv.org/pdf/2510.23763v1](http://arxiv.org/pdf/2510.23763v1)**

> **作者:** Siyin Wang; Jinlan Fu; Feihong Liu; Xinzhe He; Huangxuan Wu; Junhao Shi; Kexin Huang; Zhaoye Fei; Jingjing Gong; Zuxuan Wu; Yugang Jiang; See-Kiong Ng; Tat-Seng Chua; Xipeng Qiu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.
>
---
#### [new 077] Quanvolutional Neural Networks for Pneumonia Detection: An Efficient Quantum-Assisted Feature Extraction Paradigm
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对肺炎检测任务，旨在解决传统CNN计算成本高、小样本下泛化能力差的问题。提出一种混合量子-经典模型，利用参数化量子电路处理图像块，提取非经典特征，提升分类精度与样本效率。**

- **链接: [http://arxiv.org/pdf/2510.23660v1](http://arxiv.org/pdf/2510.23660v1)**

> **作者:** Gazi Tanbhir; Md. Farhan Shahriyar; Abdullah Md Raihan Chy
>
> **摘要:** Pneumonia poses a significant global health challenge, demanding accurate and timely diagnosis. While deep learning, particularly Convolutional Neural Networks (CNNs), has shown promise in medical image analysis for pneumonia detection, CNNs often suffer from high computational costs, limitations in feature representation, and challenges in generalizing from smaller datasets. To address these limitations, we explore the application of Quanvolutional Neural Networks (QNNs), leveraging quantum computing for enhanced feature extraction. This paper introduces a novel hybrid quantum-classical model for pneumonia detection using the PneumoniaMNIST dataset. Our approach utilizes a quanvolutional layer with a parameterized quantum circuit (PQC) to process 2x2 image patches, employing rotational Y-gates for data encoding and entangling layers to generate non-classical feature representations. These quantum-extracted features are then fed into a classical neural network for classification. Experimental results demonstrate that the proposed QNN achieves a higher validation accuracy of 83.33 percent compared to a comparable classical CNN which achieves 73.33 percent. This enhanced convergence and sample efficiency highlight the potential of QNNs for medical image analysis, particularly in scenarios with limited labeled data. This research lays the foundation for integrating quantum computing into deep-learning-driven medical diagnostic systems, offering a computationally efficient alternative to traditional approaches.
>
---
#### [new 078] OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文针对移动GUI代理的安全性问题，提出OS-Sentinel框架，融合形式化验证与视觉语言模型的上下文判断，实现对系统级违规和上下文风险的混合检测。基于动态沙箱与真实操作轨迹构建基准，显著提升安全检测效果。**

- **链接: [http://arxiv.org/pdf/2510.24411v1](http://arxiv.org/pdf/2510.24411v1)**

> **作者:** Qiushi Sun; Mukai Li; Zhoumianze Liu; Zhihui Xie; Fangzhi Xu; Zhangyue Yin; Kanzhi Cheng; Zehao Li; Zichen Ding; Qi Liu; Zhiyong Wu; Zhuosheng Zhang; Ben Kao; Lingpeng Kong
>
> **备注:** work in progress
>
> **摘要:** Computer-using agents powered by Vision-Language Models (VLMs) have demonstrated human-like capabilities in operating digital environments like mobile platforms. While these agents hold great promise for advancing digital automation, their potential for unsafe operations, such as system compromise and privacy leakage, is raising significant concerns. Detecting these safety concerns across the vast and complex operational space of mobile environments presents a formidable challenge that remains critically underexplored. To establish a foundation for mobile agent safety research, we introduce MobileRisk-Live, a dynamic sandbox environment accompanied by a safety detection benchmark comprising realistic trajectories with fine-grained annotations. Built upon this, we propose OS-Sentinel, a novel hybrid safety detection framework that synergistically combines a Formal Verifier for detecting explicit system-level violations with a VLM-based Contextual Judge for assessing contextual risks and agent actions. Experiments show that OS-Sentinel achieves 10%-30% improvements over existing approaches across multiple metrics. Further analysis provides critical insights that foster the development of safer and more reliable autonomous mobile agents.
>
---
#### [new 079] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出DynaRend，一种用于机器人操作的3D动态表征学习框架。针对真实数据稀缺导致策略泛化难的问题，通过掩码未来渲染与可微体素渲染，联合学习几何、语义与动态信息。在多视图RGB-D视频上预训练，提升策略成功率与环境扰动下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.24261v1](http://arxiv.org/pdf/2510.24261v1)**

> **作者:** Jingyi Tian; Le Wang; Sanping Zhou; Sen Wang; Jiayi Li; Gang Hua
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Learning generalizable robotic manipulation policies remains a key challenge due to the scarcity of diverse real-world training data. While recent approaches have attempted to mitigate this through self-supervised representation learning, most either rely on 2D vision pretraining paradigms such as masked image modeling, which primarily focus on static semantics or scene geometry, or utilize large-scale video prediction models that emphasize 2D dynamics, thus failing to jointly learn the geometry, semantics, and dynamics required for effective manipulation. In this paper, we present DynaRend, a representation learning framework that learns 3D-aware and dynamics-informed triplane features via masked reconstruction and future prediction using differentiable volumetric rendering. By pretraining on multi-view RGB-D video data, DynaRend jointly captures spatial geometry, future dynamics, and task semantics in a unified triplane representation. The learned representations can be effectively transferred to downstream robotic manipulation tasks via action value map prediction. We evaluate DynaRend on two challenging benchmarks, RLBench and Colosseum, as well as in real-world robotic experiments, demonstrating substantial improvements in policy success rate, generalization to environmental perturbations, and real-world applicability across diverse manipulation tasks.
>
---
#### [new 080] Listening without Looking: Modality Bias in Audio-Visual Captioning
- **分类: eess.AS; cs.CV; eess.IV**

- **简介: 该论文研究音频-视觉字幕生成任务，针对现有模型对音频模态的偏倚问题，通过消融实验和新数据集AudioVisualCaps评估模态互补性与鲁棒性，发现并缓解了模型对音频的过度依赖。**

- **链接: [http://arxiv.org/pdf/2510.24024v1](http://arxiv.org/pdf/2510.24024v1)**

> **作者:** Yuchi Ishikawa; Toranosuke Manabe; Tatsuya Komatsu; Yoshimitsu Aoki
>
> **备注:** under review
>
> **摘要:** Audio-visual captioning aims to generate holistic scene descriptions by jointly modeling sound and vision. While recent methods have improved performance through sophisticated modality fusion, it remains unclear to what extent the two modalities are complementary in current audio-visual captioning models and how robust these models are when one modality is degraded. We address these questions by conducting systematic modality robustness tests on LAVCap, a state-of-the-art audio-visual captioning model, in which we selectively suppress or corrupt the audio or visual streams to quantify sensitivity and complementarity. The analysis reveals a pronounced bias toward the audio stream in LAVCap. To evaluate how balanced audio-visual captioning models are in their use of both modalities, we augment AudioCaps with textual annotations that jointly describe the audio and visual streams, yielding the AudioVisualCaps dataset. In our experiments, we report LAVCap baseline results on AudioVisualCaps. We also evaluate the model under modality robustness tests on AudioVisualCaps and the results indicate that LAVCap trained on AudioVisualCaps exhibits less modality bias than when trained on AudioCaps.
>
---
#### [new 081] Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对动态环境中3D场景重建的实时性与精度问题，提出一种自适应关键帧选择方法。通过结合基于误差的筛选与动态阈值调整机制，优化数据流压缩，提升重建质量。在Spann3r和CUT3R上验证，显著优于传统静态策略。**

- **链接: [http://arxiv.org/pdf/2510.23928v1](http://arxiv.org/pdf/2510.23928v1)**

> **作者:** Raman Jha; Yang Zhou; Giuseppe Loianno
>
> **备注:** Under Review for ROBOVIS 2026
>
> **摘要:** In this paper, we propose an adaptive keyframe selection method for improved 3D scene reconstruction in dynamic environments. The proposed method integrates two complementary modules: an error-based selection module utilizing photometric and structural similarity (SSIM) errors, and a momentum-based update module that dynamically adjusts keyframe selection thresholds according to scene motion dynamics. By dynamically curating the most informative frames, our approach addresses a key data bottleneck in real-time perception. This allows for the creation of high-quality 3D world representations from a compressed data stream, a critical step towards scalable robot learning and deployment in complex, dynamic environments. Experimental results demonstrate significant improvements over traditional static keyframe selection strategies, such as fixed temporal intervals or uniform frame skipping. These findings highlight a meaningful advancement toward adaptive perception systems that can dynamically respond to complex and evolving visual scenes. We evaluate our proposed adaptive keyframe selection module on two recent state-of-the-art 3D reconstruction networks, Spann3r and CUT3R, and observe consistent improvements in reconstruction quality across both frameworks. Furthermore, an extensive ablation study confirms the effectiveness of each individual component in our method, underlining their contribution to the overall performance gains.
>
---
#### [new 082] Synergistic Neural Forecasting of Air Pollution with Stochastic Sampling
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出SynCast模型，针对空气污染预测中极端事件易被低估的问题，融合气象与空气质量数据，采用区域自适应Transformer与扩散随机优化模块，提升对PM₁、PM₂.₅、PM₁₀的高分辨率预测精度，尤其改善极端污染事件的预警能力。**

- **链接: [http://arxiv.org/pdf/2510.23977v1](http://arxiv.org/pdf/2510.23977v1)**

> **作者:** Yohan Abeysinghe; Muhammad Akhtar Munir; Sanoojan Baliah; Ron Sarafian; Fahad Shahbaz Khan; Yinon Rudich; Salman Khan
>
> **摘要:** Air pollution remains a leading global health and environmental risk, particularly in regions vulnerable to episodic air pollution spikes due to wildfires, urban haze and dust storms. Accurate forecasting of particulate matter (PM) concentrations is essential to enable timely public health warnings and interventions, yet existing models often underestimate rare but hazardous pollution events. Here, we present SynCast, a high-resolution neural forecasting model that integrates meteorological and air composition data to improve predictions of both average and extreme pollution levels. Built on a regionally adapted transformer backbone and enhanced with a diffusion-based stochastic refinement module, SynCast captures the nonlinear dynamics driving PM spikes more accurately than existing approaches. Leveraging on harmonized ERA5 and CAMS datasets, our model shows substantial gains in forecasting fidelity across multiple PM variables (PM$_1$, PM$_{2.5}$, PM$_{10}$), especially under extreme conditions. We demonstrate that conventional loss functions underrepresent distributional tails (rare pollution events) and show that SynCast, guided by domain-aware objectives and extreme value theory, significantly enhances performance in highly impacted regions without compromising global accuracy. This approach provides a scalable foundation for next-generation air quality early warning systems and supports climate-health risk mitigation in vulnerable regions.
>
---
#### [new 083] GroundLoc: Efficient Large-Scale Outdoor LiDAR-Only Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GroundLoc，一种基于LiDAR的室外大场景定位方法。针对高精度、高效定位难题，利用BEV投影与R2D2/SIFT关键点匹配实现地图注册，支持多传感器，地图仅需4MB/km²，实现在多个数据集上低于50cm的轨迹误差，满足实时性要求。**

- **链接: [http://arxiv.org/pdf/2510.24623v1](http://arxiv.org/pdf/2510.24623v1)**

> **作者:** Nicolai Steinke; Daniel Goehring
>
> **摘要:** In this letter, we introduce GroundLoc, a LiDAR-only localization pipeline designed to localize a mobile robot in large-scale outdoor environments using prior maps. GroundLoc employs a Bird's-Eye View (BEV) image projection focusing on the perceived ground area and utilizes the place recognition network R2D2, or alternatively, the non-learning approach Scale-Invariant Feature Transform (SIFT), to identify and select keypoints for BEV image map registration. Our results demonstrate that GroundLoc outperforms state-of-the-art methods on the SemanticKITTI and HeLiPR datasets across various sensors. In the multi-session localization evaluation, GroundLoc reaches an Average Trajectory Error (ATE) well below 50 cm on all Ouster OS2 128 sequences while meeting online runtime requirements. The system supports various sensor models, as evidenced by evaluations conducted with Velodyne HDL-64E, Ouster OS2 128, Aeva Aeries II, and Livox Avia sensors. The prior maps are stored as 2D raster image maps, which can be created from a single drive and require only 4 MB of storage per square kilometer. The source code is available at https://github.com/dcmlr/groundloc.
>
---
#### [new 084] Local Performance vs. Out-of-Distribution Generalization: An Empirical Analysis of Personalized Federated Learning in Heterogeneous Data Environments
- **分类: cs.LG; cs.AI; cs.CV; cs.DC; cs.MA**

- **简介: 该论文研究个性化联邦学习在异构数据下的性能与泛化能力。针对本地模型偏离全局最优的问题，提出FLIU算法，通过自适应个性化因子提升本地性能，同时评估模型在分布外样本上的泛化能力，实验证明其在多种异构环境下优于传统方法。**

- **链接: [http://arxiv.org/pdf/2510.24503v1](http://arxiv.org/pdf/2510.24503v1)**

> **作者:** Mortesa Hussaini; Jan Theiß; Anthony Stein
>
> **摘要:** In the context of Federated Learning with heterogeneous data environments, local models tend to converge to their own local model optima during local training steps, deviating from the overall data distributions. Aggregation of these local updates, e.g., with FedAvg, often does not align with the global model optimum (client drift), resulting in an update that is suboptimal for most clients. Personalized Federated Learning approaches address this challenge by exclusively focusing on the average local performances of clients' models on their own data distribution. Generalization to out-of-distribution samples, which is a substantial benefit of FedAvg and represents a significant component of robustness, appears to be inadequately incorporated into the assessment and evaluation processes. This study involves a thorough evaluation of Federated Learning approaches, encompassing both their local performance and their generalization capabilities. Therefore, we examine different stages within a single communication round to enable a more nuanced understanding of the considered metrics. Furthermore, we propose and incorporate a modified approach of FedAvg, designated as Federated Learning with Individualized Updates (FLIU), extending the algorithm by a straightforward individualization step with an adaptive personalization factor. We evaluate and compare the approaches empirically using MNIST and CIFAR-10 under various distributional conditions, including benchmark IID and pathological non-IID, as well as additional novel test environments with Dirichlet distribution specifically developed to stress the algorithms on complex data heterogeneity.
>
---
#### [new 085] NVSim: Novel View Synthesis Simulator for Large Scale Indoor Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出NVSim，用于从图像序列自动生成大规模可导航室内模拟环境。针对传统3D扫描成本高、扩展性差的问题，采用改进的3D高斯泼溅技术，结合地面感知与无网格可通行性分析，构建清晰的可行走平面与拓扑导航图，实现高效、真实的室内场景重建与导航模拟。**

- **链接: [http://arxiv.org/pdf/2510.24335v1](http://arxiv.org/pdf/2510.24335v1)**

> **作者:** Mingyu Jeong; Eunsung Kim; Sehun Park; Andrew Jaeyong Choi
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** We present NVSim, a framework that automatically constructs large-scale, navigable indoor simulators from only common image sequences, overcoming the cost and scalability limitations of traditional 3D scanning. Our approach adapts 3D Gaussian Splatting to address visual artifacts on sparsely observed floors a common issue in robotic traversal data. We introduce Floor-Aware Gaussian Splatting to ensure a clean, navigable ground plane, and a novel mesh-free traversability checking algorithm that constructs a topological graph by directly analyzing rendered views. We demonstrate our system's ability to generate valid, large-scale navigation graphs from real-world data. A video demonstration is avilable at https://youtu.be/tTiIQt6nXC8
>
---
#### [new 086] Quantum Machine Learning for Image Classification: A Hybrid Model of Residual Network with Quantum Support Vector Machine
- **分类: cs.LG; cs.CV; cs.ET**

- **简介: 该论文针对图像分类任务，解决高维复杂数据下疾病检测精度不足的问题。提出融合ResNet-50与量子支持向量机的混合模型，利用ResNet提取特征，经PCA降维后输入不同量子映射的QSVM进行分类，实验表明Z映射QSVM达99.23%准确率，优于传统模型。**

- **链接: [http://arxiv.org/pdf/2510.23659v1](http://arxiv.org/pdf/2510.23659v1)**

> **作者:** Md. Farhan Shahriyar; Gazi Tanbhir; Abdullah Md Raihan Chy
>
> **摘要:** Recently, there has been growing attention on combining quantum machine learning (QML) with classical deep learning approaches, as computational techniques are key to improving the performance of image classification tasks. This study presents a hybrid approach that uses ResNet-50 (Residual Network) for feature extraction and Quantum Support Vector Machines (QSVM) for classification in the context of potato disease detection. Classical machine learning as well as deep learning models often struggle with high-dimensional and complex datasets, necessitating advanced techniques like quantum computing to improve classification efficiency. In our research, we use ResNet-50 to extract deep feature representations from RGB images of potato diseases. These features are then subjected to dimensionality reduction using Principal Component Analysis (PCA). The resulting features are processed through QSVM models which apply various quantum feature maps such as ZZ, Z, and Pauli-X to transform classical data into quantum states. To assess the model performance, we compared it with classical machine learning algorithms such as Support Vector Machine (SVM) and Random Forest (RF) using five-fold stratified cross-validation for comprehensive evaluation. The experimental results demonstrate that the Z-feature map-based QSVM outperforms classical models, achieving an accuracy of 99.23 percent, surpassing both SVM and RF models. This research highlights the advantages of integrating quantum computing into image classification and provides a potential disease detection solution through hybrid quantum-classical modeling.
>
---
#### [new 087] Sound Source Localization for Spatial Mapping of Surgical Actions in Dynamic Scenes
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文属于多模态手术场景理解任务，旨在解决动态手术环境中细粒度上下文建模难题。通过融合相控阵麦克风与RGB-D相机数据，提出4D音视频表示框架，实现手术工具-组织交互的3D声源定位与时空关联，提升手术活动的动态感知能力。**

- **链接: [http://arxiv.org/pdf/2510.24332v1](http://arxiv.org/pdf/2510.24332v1)**

> **作者:** Jonas Hein; Lazaros Vlachopoulos; Maurits Geert Laurent Olthof; Bastian Sigrist; Philipp Fürnstahl; Matthias Seibold
>
> **摘要:** Purpose: Surgical scene understanding is key to advancing computer-aided and intelligent surgical systems. Current approaches predominantly rely on visual data or end-to-end learning, which limits fine-grained contextual modeling. This work aims to enhance surgical scene representations by integrating 3D acoustic information, enabling temporally and spatially aware multimodal understanding of surgical environments. Methods: We propose a novel framework for generating 4D audio-visual representations of surgical scenes by projecting acoustic localization information from a phased microphone array onto dynamic point clouds from an RGB-D camera. A transformer-based acoustic event detection module identifies relevant temporal segments containing tool-tissue interactions which are spatially localized in the audio-visual scene representation. The system was experimentally evaluated in a realistic operating room setup during simulated surgical procedures performed by experts. Results: The proposed method successfully localizes surgical acoustic events in 3D space and associates them with visual scene elements. Experimental evaluation demonstrates accurate spatial sound localization and robust fusion of multimodal data, providing a comprehensive, dynamic representation of surgical activity. Conclusion: This work introduces the first approach for spatial sound localization in dynamic surgical scenes, marking a significant advancement toward multimodal surgical scene representations. By integrating acoustic and visual data, the proposed framework enables richer contextual understanding and provides a foundation for future intelligent and autonomous surgical systems.
>
---
#### [new 088] Why Foundation Models in Pathology Are Failing
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，针对病理学领域基础模型（FMs）性能不佳的问题。指出其因生物复杂性、自监督无效、过拟合、架构冗余等七方面原因导致诊断不准、鲁棒性差。提出需重新思考现有范式，以适配组织形态学本质。**

- **链接: [http://arxiv.org/pdf/2510.23807v1](http://arxiv.org/pdf/2510.23807v1)**

> **作者:** Hamid R. Tizhoosh
>
> **摘要:** In non-medical domains, foundation models (FMs) have revolutionized computer vision and language processing through large-scale self-supervised and multimodal learning. Consequently, their rapid adoption in computational pathology was expected to deliver comparable breakthroughs in cancer diagnosis, prognostication, and multimodal retrieval. However, recent systematic evaluations reveal fundamental weaknesses: low diagnostic accuracy, poor robustness, geometric instability, heavy computational demands, and concerning safety vulnerabilities. This short paper examines these shortcomings and argues that they stem from deeper conceptual mismatches between the assumptions underlying generic foundation modeling in mainstream AI and the intrinsic complexity of human tissue. Seven interrelated causes are identified: biological complexity, ineffective self-supervision, overgeneralization, excessive architectural complexity, lack of domain-specific innovation, insufficient data, and a fundamental design flaw related to tissue patch size. These findings suggest that current pathology foundation models remain conceptually misaligned with the nature of tissue morphology and call for a fundamental rethinking of the paradigm itself.
>
---
#### [new 089] Noise is All You Need: Solving Linear Inverse Problems by Noise Combination Sampling with Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **简介: 该论文针对线性逆问题求解，提出噪声组合采样方法，通过合成最优噪声向量替代扩散模型中的噪声项，实现条件信息的自然嵌入。无需迭代调参，在少步生成下显著提升性能与稳定性，适用于图像压缩等多种任务。**

- **链接: [http://arxiv.org/pdf/2510.23633v1](http://arxiv.org/pdf/2510.23633v1)**

> **作者:** Xun Su; Hiroyuki Kasai
>
> **备注:** 9 pages
>
> **摘要:** Pretrained diffusion models have demonstrated strong capabilities in zero-shot inverse problem solving by incorporating observation information into the generation process of the diffusion models. However, this presents an inherent dilemma: excessive integration can disrupt the generative process, while insufficient integration fails to emphasize the constraints imposed by the inverse problem. To address this, we propose \emph{Noise Combination Sampling}, a novel method that synthesizes an optimal noise vector from a noise subspace to approximate the measurement score, replacing the noise term in the standard Denoising Diffusion Probabilistic Models process. This enables conditional information to be naturally embedded into the generation process without reliance on step-wise hyperparameter tuning. Our method can be applied to a wide range of inverse problem solvers, including image compression, and, particularly when the number of generation steps $T$ is small, achieves superior performance with negligible computational overhead, significantly improving robustness and stability.
>
---
## 更新

#### [replaced 001] FRBNet: Revisiting Low-Light Vision through Frequency-Domain Radial Basis Network
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.23444v2](http://arxiv.org/pdf/2510.23444v2)**

> **作者:** Fangtong Sun; Congyu Li; Ke Yang; Yuchen Pan; Hanwen Yu; Xichuan Zhang; Yiying Li
>
> **摘要:** Low-light vision remains a fundamental challenge in computer vision due to severe illumination degradation, which significantly affects the performance of downstream tasks such as detection and segmentation. While recent state-of-the-art methods have improved performance through invariant feature learning modules, they still fall short due to incomplete modeling of low-light conditions. Therefore, we revisit low-light image formation and extend the classical Lambertian model to better characterize low-light conditions. By shifting our analysis to the frequency domain, we theoretically prove that the frequency-domain channel ratio can be leveraged to extract illumination-invariant features via a structured filtering process. We then propose a novel and end-to-end trainable module named \textbf{F}requency-domain \textbf{R}adial \textbf{B}asis \textbf{Net}work (\textbf{FRBNet}), which integrates the frequency-domain channel ratio operation with a learnable frequency domain filter for the overall illumination-invariant feature enhancement. As a plug-and-play module, FRBNet can be integrated into existing networks for low-light downstream tasks without modifying loss functions. Extensive experiments across various downstream tasks demonstrate that FRBNet achieves superior performance, including +2.2 mAP for dark object detection and +2.9 mIoU for nighttime segmentation. Code is available at: https://github.com/Sing-Forevet/FRBNet.
>
---
#### [replaced 002] Learning to See and Act: Task-Aware View Planning for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05186v3](http://arxiv.org/pdf/2508.05186v3)**

> **作者:** Yongjie Bai; Zhouxia Wang; Yang Liu; Weixing Chen; Ziliang Chen; Mingtong Dai; Yongsen Zheng; Lingbo Liu; Guanbin Li; Liang Lin
>
> **备注:** 14 pages, 8 figures, project page: https://hcplab-sysu.github.io/TAVP
>
> **摘要:** Recent vision-language-action (VLA) models for multi-task robotic manipulation commonly rely on static viewpoints and shared visual encoders, which limit 3D perception and cause task interference, hindering robustness and generalization. In this work, we propose Task-Aware View Planning (TAVP), a framework designed to overcome these challenges by integrating active view planning with task-specific representation learning. TAVP employs an efficient exploration policy, accelerated by a novel pseudo-environment, to actively acquire informative views. Furthermore, we introduce a Mixture-of-Experts (MoE) visual encoder to disentangle features across different tasks, boosting both representation fidelity and task generalization. By learning to see the world in a task-aware way, TAVP generates more complete and discriminative visual representations, demonstrating significantly enhanced action prediction across a wide array of manipulation challenges. Extensive experiments on RLBench tasks show that our proposed TAVP model achieves superior performance over state-of-the-art fixed-view approaches. Visual results and code are provided at: https://hcplab-sysu.github.io/TAVP.
>
---
#### [replaced 003] ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.20234v3](http://arxiv.org/pdf/2509.20234v3)**

> **作者:** Tom Burgert; Oliver Stoll; Paolo Rota; Begüm Demir
>
> **备注:** Accepted at NeurIPS 2025 (oral)
>
> **摘要:** The hypothesis that Convolutional Neural Networks (CNNs) are inherently texture-biased has shaped much of the discourse on feature use in deep learning. We revisit this hypothesis by examining limitations in the cue-conflict experiment by Geirhos et al. To address these limitations, we propose a domain-agnostic framework that quantifies feature reliance through systematic suppression of shape, texture, and color cues, avoiding the confounds of forced-choice conflicts. By evaluating humans and neural networks under controlled suppression conditions, we find that CNNs are not inherently texture-biased but predominantly rely on local shape features. Nonetheless, this reliance can be substantially mitigated through modern training strategies or architectures (ConvNeXt, ViTs). We further extend the analysis across computer vision, medical imaging, and remote sensing, revealing that reliance patterns differ systematically: computer vision models prioritize shape, medical imaging models emphasize color, and remote sensing models exhibit a stronger reliance on texture. Code is available at https://github.com/tomburgert/feature-reliance.
>
---
#### [replaced 004] GRAID: Enhancing Spatial Reasoning of VLMs Through High-Fidelity Data Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.22118v2](http://arxiv.org/pdf/2510.22118v2)**

> **作者:** Karim Elmaaroufi; Liheng Lai; Justin Svegliato; Yutong Bai; Sanjit A. Seshia; Matei Zaharia
>
> **备注:** 22 pages, 3 figures, 3 tables, project page: https://ke7.github.io/graid/
>
> **摘要:** Vision Language Models (VLMs) achieve strong performance on many vision-language tasks but often struggle with spatial reasoning$\unicode{x2014}$a prerequisite for many applications. Empirically, we find that a dataset produced by a current training data generation pipeline has a 57.6% human validation rate. These rates stem from current limitations: single-image 3D reconstruction introduces cascading modeling errors and requires wide answer tolerances, while caption-based methods require hyper-detailed annotations and suffer from generative hallucinations. We present GRAID, built on the key insight that qualitative spatial relationships can be reliably determined from 2D geometric primitives alone. By operating exclusively on 2D bounding boxes from standard object detectors, GRAID avoids both 3D reconstruction errors and generative hallucinations, resulting in datasets that are of higher quality than existing tools that produce similar datasets as validated by human evaluations. We apply our framework to the BDD100k, NuImages, and Waymo datasets, generating over 8.5 million high-quality VQA pairs creating questions spanning spatial relations, counting, ranking, and size comparisons. We evaluate one of the datasets and find it achieves 91.16% human-validated accuracy$\unicode{x2014}$compared to 57.6% on a dataset generated by recent work. Critically, we demonstrate that when trained on GRAID data, models learn spatial reasoning concepts that generalize: models fine-tuned on 6 question types improve on over 10 held-out types, with accuracy gains of 47.5% on BDD and 37.9% on NuImages for Llama 3.2B 11B, and when trained on all questions types, achieve improvements on several existing benchmarks such as BLINK. The GRAID framework, datasets, and additional information can be found $\href{this https URL}{here}$.
>
---
#### [replaced 005] VSA: Faster Video Diffusion with Trainable Sparse Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13389v5](http://arxiv.org/pdf/2505.13389v5)**

> **作者:** Peiyuan Zhang; Yongqi Chen; Haofeng Huang; Will Lin; Zhengzhong Liu; Ion Stoica; Eric Xing; Hao Zhang
>
> **备注:** Accepted by Neurips 2025
>
> **摘要:** Scaling video diffusion transformers (DiTs) is limited by their quadratic 3D attention, even though most of the attention mass concentrates on a small subset of positions. We turn this observation into VSA, a trainable, hardware-efficient sparse attention that replaces full attention at \emph{both} training and inference. In VSA, a lightweight coarse stage pools tokens into tiles and identifies high-weight \emph{critical tokens}; a fine stage computes token-level attention only inside those tiles subjecting to block computing layout to ensure hard efficiency. This leads to a single differentiable kernel that trains end-to-end, requires no post-hoc profiling, and sustains 85\% of FlashAttention3 MFU. We perform a large sweep of ablation studies and scaling-law experiments by pretraining DiTs from 60M to 1.4B parameters. VSA reaches a Pareto point that cuts training FLOPS by 2.53$\times$ with no drop in diffusion loss. Retrofitting the open-source Wan-2.1 model speeds up attention time by 6$\times$ and lowers end-to-end generation time from 31s to 18s with comparable quality. These results establish trainable sparse attention as a practical alternative to full attention and a key enabler for further scaling of video diffusion models. Code will be available at https://github.com/hao-ai-lab/FastVideo.
>
---
#### [replaced 006] Does CLIP perceive art the same way we do?
- **分类: cs.CV; cs.MM; 68T45, 68T07 (Primary) 68T50, 68U10 (Secondary); I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.05229v2](http://arxiv.org/pdf/2505.05229v2)**

> **作者:** Andrea Asperti; Leonardo Dessì; Maria Chiara Tonetti; Nico Wu
>
> **摘要:** CLIP has emerged as a powerful multimodal model capable of connecting images and text through joint embeddings, but to what extent does it 'see' the same way humans do - especially when interpreting artworks? In this paper, we investigate CLIP's ability to extract high-level semantic and stylistic information from paintings, including both human-created and AI-generated imagery. We evaluate its perception across multiple dimensions: content, scene understanding, artistic style, historical period, and the presence of visual deformations or artifacts. By designing targeted probing tasks and comparing CLIP's responses to human annotations and expert benchmarks, we explore its alignment with human perceptual and contextual understanding. Our findings reveal both strengths and limitations in CLIP's visual representations, particularly in relation to aesthetic cues and artistic intent. We further discuss the implications of these insights for using CLIP as a guidance mechanism during generative processes, such as style transfer or prompt-based image synthesis. Our work highlights the need for deeper interpretability in multimodal systems, especially when applied to creative domains where nuance and subjectivity play a central role.
>
---
#### [replaced 007] Unveiling Concept Attribution in Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.02542v3](http://arxiv.org/pdf/2412.02542v3)**

> **作者:** Quang H. Nguyen; Hoang Phan; Khoa D. Doan
>
> **摘要:** Diffusion models have shown remarkable abilities in generating realistic and high-quality images from text prompts. However, a trained model remains largely black-box; little do we know about the roles of its components in exhibiting a concept such as objects or styles. Recent works employ causal tracing to localize knowledge-storing layers in generative models without showing how other layers contribute to the target concept. In this work, we approach diffusion models' interpretability problem from a more general perspective and pose a question: \textit{``How do model components work jointly to demonstrate knowledge?''}. To answer this question, we decompose diffusion models using component attribution, systematically unveiling the importance of each component (specifically the model parameter) in generating a concept. The proposed framework, called \textbf{C}omponent \textbf{A}ttribution for \textbf{D}iffusion Model (CAD), discovers the localization of concept-inducing (positive) components, while interestingly uncovers another type of components that contribute negatively to generating a concept, which is missing in the previous knowledge localization work. Based on this holistic understanding of diffusion models, we introduce two fast, inference-time model editing algorithms, CAD-Erase and CAD-Amplify; in particular, CAD-Erase enables erasure and CAD-Amplify allows amplification of a generated concept by ablating the positive and negative components, respectively, while retaining knowledge of other concepts. Extensive experimental results validate the significance of both positive and negative components pinpointed by our framework, demonstrating the potential of providing a complete view of interpreting generative models. Our code is available \href{https://github.com/mail-research/CAD-attribution4diffusion}{here}.
>
---
#### [replaced 008] CAUSAL3D: A Comprehensive Benchmark for Causal Learning from Visual Data
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04852v2](http://arxiv.org/pdf/2503.04852v2)**

> **作者:** Disheng Liu; Yiran Qiao; Wuche Liu; Yiren Lu; Yunlai Zhou; Tuo Liang; Yu Yin; Jing Ma
>
> **摘要:** True intelligence hinges on the ability to uncover and leverage hidden causal relations. Despite significant progress in AI and computer vision (CV), there remains a lack of benchmarks for assessing models' abilities to infer latent causality from complex visual data. In this paper, we introduce \textsc{\textbf{Causal3D}}, a novel and comprehensive benchmark that integrates structured data (tables) with corresponding visual representations (images) to evaluate causal reasoning. Designed within a systematic framework, Causal3D comprises 19 3D-scene datasets capturing diverse causal relations, views, and backgrounds, enabling evaluations across scenes of varying complexity. We assess multiple state-of-the-art methods, including classical causal discovery, causal representation learning, and large/vision-language models (LLMs/VLMs). Our experiments show that as causal structures grow more complex without prior knowledge, performance declines significantly, highlighting the challenges even advanced methods face in complex causal scenarios. Causal3D serves as a vital resource for advancing causal reasoning in CV and fostering trustworthy AI in critical domains.
>
---
#### [replaced 009] DArFace: Deformation Aware Robustness for Low Quality Face Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08423v4](http://arxiv.org/pdf/2505.08423v4)**

> **作者:** Sadaf Gulshad; Abdullah Aldahlawi
>
> **摘要:** Facial recognition systems have achieved remarkable success by leveraging deep neural networks, advanced loss functions, and large-scale datasets. However, their performance often deteriorates in real-world scenarios involving low-quality facial images. Such degradations, common in surveillance footage or standoff imaging include low resolution, motion blur, and various distortions, resulting in a substantial domain gap from the high-quality data typically used during training. While existing approaches attempt to address robustness by modifying network architectures or modeling global spatial transformations, they frequently overlook local, non-rigid deformations that are inherently present in real-world settings. In this work, we introduce \textbf{DArFace}, a \textbf{D}eformation-\textbf{A}ware \textbf{r}obust \textbf{Face} recognition framework that enhances robustness to such degradations without requiring paired high- and low-quality training samples. Our method adversarially integrates both global transformations (e.g., rotation, translation) and local elastic deformations during training to simulate realistic low-quality conditions. Moreover, we introduce a contrastive objective to enforce identity consistency across different deformed views. Extensive evaluations on low-quality benchmarks including TinyFace, IJB-B, and IJB-C demonstrate that DArFace surpasses state-of-the-art methods, with significant gains attributed to the inclusion of local deformation modeling.
>
---
#### [replaced 010] Navigation with VLM framework: Towards Going to Any Language
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02787v2](http://arxiv.org/pdf/2410.02787v2)**

> **作者:** Zecheng Yin; Chonghao Cheng; and Yao Guo; Zhen Li
>
> **备注:** under review
>
> **摘要:** Navigating towards fully open language goals and exploring open scenes in an intelligent way have always raised significant challenges. Recently, Vision Language Models (VLMs) have demonstrated remarkable capabilities to reason with both language and visual data. Although many works have focused on leveraging VLMs for navigation in open scenes, they often require high computational cost, rely on object-centric approaches, or depend on environmental priors in detailed human instructions. We introduce Navigation with VLM (NavVLM), a training-free framework that harnesses open-source VLMs to enable robots to navigate effectively, even for human-friendly language goal such as abstract places, actions, or specific objects in open scenes. NavVLM leverages the VLM as its cognitive core to perceive environmental information and constantly provides exploration guidance achieving intelligent navigation with only a neat target rather than a detailed instruction with environment prior. We evaluated and validated NavVLM in both simulation and real-world experiments. In simulation, our framework achieves state-of-the-art performance in Success weighted by Path Length (SPL) on object-specifc tasks in richly detailed environments from Matterport 3D (MP3D), Habitat Matterport 3D (HM3D) and Gibson. With navigation episode reported, NavVLM demonstrates the capabilities to navigate towards any open-set languages. In real-world validation, we validated our framework's effectiveness in real-world robot at indoor scene.
>
---
#### [replaced 011] UniMedVL: Unifying Medical Multimodal Understanding And Generation Through Observation-Knowledge-Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.15710v2](http://arxiv.org/pdf/2510.15710v2)**

> **作者:** Junzhi Ning; Wei Li; Cheng Tang; Jiashi Lin; Chenglong Ma; Chaoyang Zhang; Jiyao Liu; Ying Chen; Shujian Gao; Lihao Liu; Yuandong Pu; Huihui Xu; Chenhui Gou; Ziyan Huang; Yi Xin; Qi Qin; Zhongying Deng; Diping Song; Bin Fu; Guang Yang; Yuanfeng Ji; Tianbin Li; Yanzhou Su; Jin Ye; Shixiang Tang; Ming Hu; Junjun He
>
> **摘要:** Medical diagnostic applications require models that can process multimodal medical inputs (images, patient histories, lab results) and generate diverse outputs including both textual reports and visual content (annotations, segmentation masks, and images). Despite this need, existing medical AI systems disrupt this unified process: medical image understanding models interpret images but cannot generate visual outputs, while medical image generation models synthesize images but cannot provide textual explanations. This leads to gaps in data representation, feature integration, and task-level multimodal capabilities. To this end, we propose a multi-level framework that draws inspiration from diagnostic workflows through the Observation-Knowledge-Analysis (OKA) paradigm. Specifically, at the observation level, we construct UniMed-5M, a dataset comprising over 5.6M samples that reformat diverse unimodal data into multimodal pairs for foundational observation. At the knowledge level, we propose Progressive Curriculum Learning that systematically introduces medical multimodal knowledge. At the analysis level, we introduce UniMedVL, the first medical unified multimodal model for the simultaneous analysis of image understanding and generation tasks within a single architecture. UniMedVL achieves superior performance on five medical image understanding benchmarks, while matching specialized models in generation quality across eight medical imaging modalities. Crucially, our unified architecture enables bidirectional knowledge sharing: generation tasks enhance visual understanding features, demonstrating that integrating traditionally separate capabilities within a single medical framework unlocks improvements across diverse medical vision-language tasks. Code is available at https://github.com/uni-medical/UniMedVL.
>
---
#### [replaced 012] PRISM-Bench: A Benchmark of Puzzle-Based Visual Tasks with CoT Error Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23594v2](http://arxiv.org/pdf/2510.23594v2)**

> **作者:** Yusu Qian; Cheng Wan; Chao Jia; Yinfei Yang; Qingyu Zhao; Zhe Gan
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable progress on vision-language tasks, yet their reasoning processes remain sometimes unreliable. We introduce PRISM-Bench, a benchmark of puzzle-based visual challenges designed to evaluate not only whether models can solve problems, but how their reasoning unfolds. Unlike prior evaluations that measure only final-answer accuracy, PRISM-Bench introduces a diagnostic task: given a visual puzzle and a step-by-step chain-of-thought (CoT) containing exactly one error, models must identify the first incorrect step. This setting enables fine-grained assessment of logical consistency, error detection, and visual reasoning. The puzzles in PRISM-Bench require multi-step symbolic, geometric, and analogical reasoning, resisting shortcuts based on superficial pattern matching. Evaluations across state-of-the-art MLLMs reveal a persistent gap between fluent generation and faithful reasoning: models that produce plausible CoTs often fail to locate simple logical faults. By disentangling answer generation from reasoning verification, PRISM-Bench offers a sharper lens on multimodal reasoning competence and underscores the need for diagnostic evaluation protocols in the development of trustworthy MLLMs.
>
---
#### [replaced 013] LongCat-Video Technical Report
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22200v2](http://arxiv.org/pdf/2510.22200v2)**

> **作者:** Meituan LongCat Team; Xunliang Cai; Qilong Huang; Zhuoliang Kang; Hongyu Li; Shijun Liang; Liya Ma; Siyu Ren; Xiaoming Wei; Rixu Xie; Tong Zhang
>
> **摘要:** Video generation is a critical pathway toward world models, with efficient long video inference as a key capability. Toward this end, we introduce LongCat-Video, a foundational video generation model with 13.6B parameters, delivering strong performance across multiple video generation tasks. It particularly excels in efficient and high-quality long video generation, representing our first step toward world models. Key features include: Unified architecture for multiple tasks: Built on the Diffusion Transformer (DiT) framework, LongCat-Video supports Text-to-Video, Image-to-Video, and Video-Continuation tasks with a single model; Long video generation: Pretraining on Video-Continuation tasks enables LongCat-Video to maintain high quality and temporal coherence in the generation of minutes-long videos; Efficient inference: LongCat-Video generates 720p, 30fps videos within minutes by employing a coarse-to-fine generation strategy along both the temporal and spatial axes. Block Sparse Attention further enhances efficiency, particularly at high resolutions; Strong performance with multi-reward RLHF: Multi-reward RLHF training enables LongCat-Video to achieve performance on par with the latest closed-source and leading open-source models. Code and model weights are publicly available to accelerate progress in the field.
>
---
#### [replaced 014] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00129v2](http://arxiv.org/pdf/2506.00129v2)**

> **作者:** Edward Fish; Richard Bowden
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Recent progress in Sign Language Translation (SLT) has focussed primarily on improving the representational capacity of large language models to incorporate Sign Language features. This work explores an alternative direction: enhancing the geometric properties of skeletal representations themselves. We propose Geo-Sign, a method that leverages the properties of hyperbolic geometry to model the hierarchical structure inherent in sign language kinematics. By projecting skeletal features derived from Spatio-Temporal Graph Convolutional Networks (ST-GCNs) into the Poincar\'e ball model, we aim to create more discriminative embeddings, particularly for fine-grained motions like finger articulations. We introduce a hyperbolic projection layer, a weighted Fr\'echet mean aggregation scheme, and a geometric contrastive loss operating directly in hyperbolic space. These components are integrated into an end-to-end translation framework as a regularisation function, to enhance the representations within the language model. This work demonstrates the potential of hyperbolic geometry to improve skeletal representations for Sign Language Translation, improving on SOTA RGB methods while preserving privacy and improving computational efficiency. Code available here: https://github.com/ed-fish/geo-sign.
>
---
#### [replaced 015] Frequency-Aware Vision Transformers for High-Fidelity Super-Resolution of Earth System Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12427v4](http://arxiv.org/pdf/2502.12427v4)**

> **作者:** Ehsan Zeraatkar; Salah A Faroughi; Jelena Tešić
>
> **摘要:** Super-resolution (SR) is crucial for enhancing the spatial fidelity of Earth System Model (ESM) outputs, allowing fine-scale structures vital to climate science to be recovered from coarse simulations. However, traditional deep super-resolution methods, including convolutional and transformer-based models, tend to exhibit spectral bias, reconstructing low-frequency content more readily than valuable high-frequency details. In this work, we introduce two frequency-aware frameworks: the Vision Transformer-Tuned Sinusoidal Implicit Representation (ViSIR), combining Vision Transformers and sinusoidal activations to mitigate spectral bias, and the Vision Transformer Fourier Representation Network (ViFOR), which integrates explicit Fourier-based filtering for independent low- and high-frequency learning. Evaluated on the E3SM-HR Earth system dataset across surface temperature, shortwave, and longwave fluxes, these models outperform leading CNN, GAN, and vanilla transformer baselines, with ViFOR demonstrating up to 2.6~dB improvements in PSNR and significantly higher SSIM. Detailed ablation and scaling studies highlight the benefit of full-field training, the impact of frequency hyperparameters, and the potential for generalization. The results establish ViFOR as a state-of-the-art, scalable solution for climate data downscaling. Future extensions will address temporal super-resolution, multimodal climate variables, automated parameter selection, and integration of physical conservation constraints to broaden scientific applicability.
>
---
#### [replaced 016] LiDAR Remote Sensing Meets Weak Supervision: Concepts, Methods, and Perspectives
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18384v2](http://arxiv.org/pdf/2503.18384v2)**

> **作者:** Yuan Gao; Shaobo Xia; Pu Wang; Xiaohuan Xi; Sheng Nie; Cheng Wang
>
> **摘要:** Light detection and ranging (LiDAR) remote sensing encompasses two major directions: data interpretation and parameter inversion. However, both directions rely heavily on costly and labor-intensive labeled data and field measurements, which constrains their scalability and spatiotemporal adaptability. Weakly Supervised Learning (WSL) provides a unified framework to address these limitations. This paper departs from the traditional view that treats interpretation and inversion as separate tasks and offers a systematic review of recent advances in LiDAR remote sensing from a unified WSL perspective. We cover typical WSL settings including incomplete supervision(e.g., sparse point labels), inexact supervision (e.g., scene-level tags), inaccurate supervision (e.g., noisy labels), and cross-domain supervision (e.g., domain adaptation/generalization) and corresponding techniques such as pseudo-labeling, consistency regularization, self-training, and label refinement, which collectively enable robust learning from limited and weak annotations.We further analyze LiDAR-specific challenges (e.g., irregular geometry, data sparsity, domain heterogeneity) that require tailored weak supervision, and examine how sparse LiDAR observations can guide joint learning with other remote-sensing data for continuous surface-parameter retrieval. Finally, we highlight future directions where WSL acts as a bridge between LiDAR and foundation models to leverage large-scale multimodal datasets and reduce labeling costs, while also enabling broader WSL-driven advances in generalization, open-world adaptation, and scalable LiDAR remote sensing.
>
---
#### [replaced 017] Riemannian-Geometric Fingerprints of Generative Models
- **分类: cs.LG; cs.CR; cs.CV; I.2.6**

- **链接: [http://arxiv.org/pdf/2506.22802v2](http://arxiv.org/pdf/2506.22802v2)**

> **作者:** Hae Jin Song; Laurent Itti
>
> **备注:** ICCV 2025 Highlight paper
>
> **摘要:** Recent breakthroughs and rapid integration of generative models (GMs) have sparked interest in the problem of model attribution and their fingerprints. For instance, service providers need reliable methods of authenticating their models to protect their IP, while users and law enforcement seek to verify the source of generated content for accountability and trust. In addition, a growing threat of model collapse is arising, as more model-generated data are being fed back into sources (e.g., YouTube) that are often harvested for training ("regurgitative training"), heightening the need to differentiate synthetic from human data. Yet, a gap still exists in understanding generative models' fingerprints, we believe, stemming from the lack of a formal framework that can define, represent, and analyze the fingerprints in a principled way. To address this gap, we take a geometric approach and propose a new definition of artifact and fingerprint of GMs using Riemannian geometry, which allows us to leverage the rich theory of differential geometry. Our new definition generalizes previous work (Song et al., 2024) to non-Euclidean manifolds by learning Riemannian metrics from data and replacing the Euclidean distances and nearest-neighbor search with geodesic distances and kNN-based Riemannian center of mass. We apply our theory to a new gradient-based algorithm for computing the fingerprints in practice. Results show that it is more effective in distinguishing a large array of GMs, spanning across 4 different datasets in 2 different resolutions (64 by 64, 256 by 256), 27 model architectures, and 2 modalities (Vision, Vision-Language). Using our proposed definition significantly improves the performance on model attribution, as well as a generalization to unseen datasets, model types, and modalities, suggesting its practical efficacy.
>
---
#### [replaced 018] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.20072v2](http://arxiv.org/pdf/2508.20072v2)**

> **作者:** Zhixuan Liang; Yizhuo Li; Tianshuo Yang; Chengyue Wu; Sitong Mao; Tian Nian; Liuao Pei; Shunbo Zhou; Xiaokang Yang; Jiangmiao Pang; Yao Mu; Ping Luo
>
> **备注:** 16 pages
>
> **摘要:** Vision-Language-Action (VLA) models adapt large vision-language backbones to map images and instructions into robot actions. However, prevailing VLAs either generate actions auto-regressively in a fixed left-to-right order or attach separate MLP or diffusion heads outside the backbone, leading to fragmented information pathways and specialized training requirements that hinder a unified, scalable architecture. We present Discrete Diffusion VLA, a unified-transformer policy that models discretized action chunks with discrete diffusion. The design retains diffusion's progressive refinement paradigm while remaining natively compatible with the discrete token interface of VLMs. Our method achieves an adaptive decoding order that resolves easy action elements before harder ones and uses secondary re-masking to revisit uncertain predictions across refinement rounds, which improves consistency and enables robust error correction. This unified decoder preserves pre-trained vision-language priors, supports parallel decoding, breaks the autoregressive bottleneck, and reduces the number of function evaluations. Discrete Diffusion VLA achieves 96.3% avg. success rates on LIBERO, 71.2% visual matching on SimplerEnv-Fractal and 54.2% overall on SimplerEnv-Bridge, improving over autoregressive, MLP decoder and continuous diffusion baselines. These findings indicate that discrete-diffusion VLA supports precise action modeling and consistent training, laying groundwork for scaling VLA to larger models and datasets. Our project page is https://github.com/Liang-ZX/DiscreteDiffusionVLA
>
---
#### [replaced 019] Seeing Symbols, Missing Cultures: Probing Vision-Language Models' Reasoning on Fire Imagery and Cultural Meaning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23311v2](http://arxiv.org/pdf/2509.23311v2)**

> **作者:** Haorui Yu; Yang Zhao; Yijia Chu; Qiufeng Yi
>
> **备注:** 8 pages, 5 figures, 4 tables. Submitted to WiNLP 2025 Workshop at COLING 2025
>
> **摘要:** Vision-Language Models (VLMs) often appear culturally competent but rely on superficial pattern matching rather than genuine cultural understanding. We introduce a diagnostic framework to probe VLM reasoning on fire-themed cultural imagery through both classification and explanation analysis. Testing multiple models on Western festivals, non-Western traditions, and emergency scenes reveals systematic biases: models correctly identify prominent Western festivals but struggle with underrepresented cultural events, frequently offering vague labels or dangerously misclassifying emergencies as celebrations. These failures expose the risks of symbolic shortcuts and highlight the need for cultural evaluation beyond accuracy metrics to ensure interpretable and fair multimodal systems.
>
---
#### [replaced 020] Normal and Abnormal Pathology Knowledge-Augmented Vision-Language Model for Anomaly Detection in Pathology Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15256v2](http://arxiv.org/pdf/2508.15256v2)**

> **作者:** Jinsol Song; Jiamu Wang; Anh Tien Nguyen; Keunho Byeon; Sangjeong Ahn; Sung Hak Lee; Jin Tae Kwak
>
> **备注:** Accepted to ICCV 2025. Code is available at: https://github.com/QuIIL/ICCV2025_Ano-NAViLa
>
> **摘要:** Anomaly detection in computational pathology aims to identify rare and scarce anomalies where disease-related data are often limited or missing. Existing anomaly detection methods, primarily designed for industrial settings, face limitations in pathology due to computational constraints, diverse tissue structures, and lack of interpretability. To address these challenges, we propose Ano-NAViLa, a Normal and Abnormal pathology knowledge-augmented Vision-Language model for Anomaly detection in pathology images. Ano-NAViLa is built on a pre-trained vision-language model with a lightweight trainable MLP. By incorporating both normal and abnormal pathology knowledge, Ano-NAViLa enhances accuracy and robustness to variability in pathology images and provides interpretability through image-text associations. Evaluated on two lymph node datasets from different organs, Ano-NAViLa achieves the state-of-the-art performance in anomaly detection and localization, outperforming competing models.
>
---
#### [replaced 021] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views
- **分类: cs.CV; cs.CL; cs.RO; I.2.10; I.2.9; I.2.7; H.5.2**

- **链接: [http://arxiv.org/pdf/2510.22672v2](http://arxiv.org/pdf/2510.22672v2)**

> **作者:** Anna Deichler; Jonas Beskow
>
> **备注:** 10 pages, 6 figures, 2 tables. Accepted to the NeurIPS 2025 Workshop on SPACE in Vision, Language, and Embodied AI (SpaVLE). Dataset: https://huggingface.co/datasets/annadeichler/KTH-ARIA-referential
>
> **摘要:** We introduce Look and Tell, a multimodal dataset for studying referential communication across egocentric and exocentric perspectives. Using Meta Project Aria smart glasses and stationary cameras, we recorded synchronized gaze, speech, and video as 25 participants instructed a partner to identify ingredients in a kitchen. Combined with 3D scene reconstructions, this setup provides a benchmark for evaluating how different spatial representations (2D vs. 3D; ego vs. exo) affect multimodal grounding. The dataset contains 3.67 hours of recordings, including 2,707 richly annotated referential expressions, and is designed to advance the development of embodied agents that can understand and engage in situated dialogue.
>
---
#### [replaced 022] InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.16691v2](http://arxiv.org/pdf/2509.16691v2)**

> **作者:** Qiang Xiang; Shuang Sun; Binglei Li; Dejia Song; Huaxia Li; Nemo Chen; Xu Tang; Yao Hu; Junping Zhang
>
> **备注:** Accepted in NeurIPS 2025
>
> **摘要:** Diffusion models have demonstrated remarkable capabilities in generating high-quality images. Recent advancements in Layout-to-Image (L2I) generation have leveraged positional conditions and textual descriptions to facilitate precise and controllable image synthesis. Despite overall progress, current L2I methods still exhibit suboptimal performance. Therefore, we propose InstanceAssemble, a novel architecture that incorporates layout conditions via instance-assembling attention, enabling position control with bounding boxes (bbox) and multimodal content control including texts and additional visual content. Our method achieves flexible adaption to existing DiT-based T2I models through light-weighted LoRA modules. Additionally, we propose a Layout-to-Image benchmark, Denselayout, a comprehensive benchmark for layout-to-image generation, containing 5k images with 90k instances in total. We further introduce Layout Grounding Score (LGS), an interpretable evaluation metric to more precisely assess the accuracy of L2I generation. Experiments demonstrate that our InstanceAssemble method achieves state-of-the-art performance under complex layout conditions, while exhibiting strong compatibility with diverse style LoRA modules. The code and pretrained models are publicly available at https://github.com/FireRedTeam/InstanceAssemble.
>
---
#### [replaced 023] Acoustic Neural 3D Reconstruction Under Pose Drift
- **分类: eess.SP; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08930v2](http://arxiv.org/pdf/2503.08930v2)**

> **作者:** Tianxiang Lin; Mohamad Qadri; Kevin Zhang; Adithya Pediredla; Christopher A. Metzler; Michael Kaess
>
> **备注:** 8 pages, 8 figures. This paper is accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** We consider the problem of optimizing neural implicit surfaces for 3D reconstruction using acoustic images collected with drifting sensor poses. The accuracy of current state-of-the-art 3D acoustic modeling algorithms is highly dependent on accurate pose estimation; small errors in sensor pose can lead to severe reconstruction artifacts. In this paper, we propose an algorithm that jointly optimizes the neural scene representation and sonar poses. Our algorithm does so by parameterizing the 6DoF poses as learnable parameters and backpropagating gradients through the neural renderer and implicit representation. We validated our algorithm on both real and simulated datasets. It produces high-fidelity 3D reconstructions even under significant pose drift.
>
---
#### [replaced 024] Mano Technical Report
- **分类: cs.MM; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.17336v2](http://arxiv.org/pdf/2509.17336v2)**

> **作者:** Tianyu Fu; Anyang Su; Chenxu Zhao; Hanning Wang; Minghui Wu; Zhe Yu; Fei Hu; Mingjia Shi; Wei Dong; Jiayao Wang; Yuyang Chen; Ruiyang Yu; Siran Peng; Menglin Li; Nan Huang; Haitian Wei; Jiawei Yu; Yi Xin; Xilin Zhao; Kai Gu; Ping Jiang; Sifan Zhou; Shuo Wang
>
> **摘要:** Graphical user interfaces (GUIs) are the primary medium for human-computer interaction, yet automating GUI interactions remains challenging due to the complexity of visual elements, dynamic environments, and the need for multi-step reasoning. Existing methods based on vision-language models (VLMs) often suffer from limited resolution, domain mismatch, and insufficient sequential decisionmaking capability. To address these issues, we propose Mano, a robust GUI agent built upon a multi-modal foundation model pre-trained on extensive web and computer system data. Our approach integrates a novel simulated environment for high-fidelity data generation, a three-stage training pipeline (supervised fine-tuning, offline reinforcement learning, and online reinforcement learning), and a verification module for error recovery. Mano demonstrates state-of-the-art performance on multiple GUI benchmarks, including Mind2Web and OSWorld, achieving significant improvements in success rate and operational accuracy. Our work provides new insights into the effective integration of reinforcement learning with VLMs for practical GUI agent deployment, highlighting the importance of domain-specific data, iterative training, and holistic reward design.
>
---
#### [replaced 025] IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22706v2](http://arxiv.org/pdf/2510.22706v2)**

> **作者:** Hao Li; Zhengyu Zou; Fangfu Liu; Xuanyang Zhang; Fangzhou Hong; Yukang Cao; Yushi Lan; Manyuan Zhang; Gang Yu; Dingwen Zhang; Ziwei Liu
>
> **备注:** https://github.com/lifuguan/IGGT_official
>
> **摘要:** Humans naturally perceive the geometric structure and semantic content of a 3D world as intertwined dimensions, enabling coherent and accurate understanding of complex scenes. However, most prior approaches prioritize training large geometry models for low-level 3D reconstruction and treat high-level spatial understanding in isolation, overlooking the crucial interplay between these two fundamental aspects of 3D-scene analysis, thereby limiting generalization and leading to poor performance in downstream 3D understanding tasks. Recent attempts have mitigated this issue by simply aligning 3D models with specific language models, thus restricting perception to the aligned model's capacity and limiting adaptability to downstream tasks. In this paper, we propose InstanceGrounded Geometry Transformer (IGGT), an end-to-end large unified transformer to unify the knowledge for both spatial reconstruction and instance-level contextual understanding. Specifically, we design a 3D-Consistent Contrastive Learning strategy that guides IGGT to encode a unified representation with geometric structures and instance-grounded clustering through only 2D visual inputs. This representation supports consistent lifting of 2D visual inputs into a coherent 3D scene with explicitly distinct object instances. To facilitate this task, we further construct InsScene-15K, a large-scale dataset with high-quality RGB images, poses, depth maps, and 3D-consistent instance-level mask annotations with a novel data curation pipeline.
>
---
#### [replaced 026] Global urban visual perception varies across demographics and personalities
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12758v4](http://arxiv.org/pdf/2505.12758v4)**

> **作者:** Matias Quintana; Youlong Gu; Xiucheng Liang; Yujun Hou; Koichi Ito; Yihan Zhu; Mahmoud Abdelrahman; Filip Biljecki
>
> **摘要:** Understanding people's preferences is crucial for urban planning, yet current approaches often combine responses from multi-cultural populations, obscuring demographic differences and risking amplifying biases. We conducted a largescale urban visual perception survey of streetscapes worldwide using street view imagery, examining how demographics -- including gender, age, income, education, race and ethnicity, and personality traits -- shape perceptions among 1,000 participants with balanced demographics from five countries and 45 nationalities. This dataset, Street Perception Evaluation Considering Socioeconomics (SPECS), reveals demographic- and personality-based differences across six traditional indicators -- safe, lively, wealthy, beautiful, boring, depressing -- and four new ones -- live nearby, walk, cycle, green. Location-based sentiments further shape these preferences. Machine learning models trained on existing global datasets tend to overestimate positive indicators and underestimate negative ones compared to human responses, underscoring the need for local context. Our study aspires to rectify the myopic treatment of street perception, which rarely considers demographics or personality traits.
>
---
#### [replaced 027] MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.02885v2](http://arxiv.org/pdf/2501.02885v2)**

> **作者:** Hui Sun; Shiyin Lu; Huanyu Wang; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang; Ming Li
>
> **备注:** 26 pages, 14 figures
>
> **摘要:** Video large language models (Video-LLMs) have made significant progress in understanding videos. However, processing multiple frames leads to lengthy visual token sequences, presenting challenges such as the limited context length cannot accommodate the entire video, and the inclusion of irrelevant frames hinders visual perception. Hence, effective frame selection is crucial. This paper emphasizes that frame selection should follow three key principles: query relevance, list-wise diversity, and sequentiality. Existing methods, such as uniform frame sampling and query-frame matching, do not capture all of these principles. Thus, we propose Markov decision determinantal point process with dynamic programming (MDP3) for frame selection, a training-free and model-agnostic method that can be seamlessly integrated into existing Video-LLMs. Our method first estimates frame similarities conditioned on the query using a conditional Gaussian kernel within the reproducing kernel Hilbert space~(RKHS). We then apply the determinantal point process~(DPP) to the similarity matrix to capture both query relevance and list-wise diversity. To incorporate sequentiality, we segment the video and apply DPP within each segment, conditioned on the preceding segment selection, modeled as a Markov decision process~(MDP) for allocating selection sizes across segments. Theoretically, MDP3 provides a \((1 - 1/e)\)-approximate solution to the NP-hard list-wise frame selection problem with pseudo-polynomial time complexity, demonstrating its efficiency. Empirically, MDP3 significantly outperforms existing methods, verifying its effectiveness and robustness.
>
---
#### [replaced 028] DRBD-Mamba for Robust and Efficient Brain Tumor Segmentation with Analytical Insights
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14383v2](http://arxiv.org/pdf/2510.14383v2)**

> **作者:** Danish Ali; Ajmal Mian; Naveed Akhtar; Ghulam Mubashar Hassan
>
> **摘要:** Accurate brain tumor segmentation is significant for clinical diagnosis and treatment but remains challenging due to tumor heterogeneity. Mamba-based State Space Models have demonstrated promising performance. However, despite their computational efficiency over other neural architectures, they incur considerable overhead for this task due to their sequential feature computation across multiple spatial axes. Moreover, their robustness across diverse BraTS data partitions remains largely unexplored, leaving a critical gap in reliable evaluation. To address this, we first propose a dual-resolution bi-directional Mamba (DRBD-Mamba), an efficient 3D segmentation model that captures multi-scale long-range dependencies with minimal computational overhead. We leverage a space-filling curve to preserve spatial locality during 3D-to-1D feature mapping, thereby reducing reliance on computationally expensive multi-axial feature scans. To enrich feature representation, we propose a gated fusion module that adaptively integrates forward and reverse contexts, along with a quantization block that improves robustness. We further propose five systematic folds on BraTS2023 for rigorous evaluation of segmentation techniques under diverse conditions and present analysis of common failure scenarios. On the 20% test set used by recent methods, our model achieves Dice improvements of 0.10% for whole tumor, 1.75% for tumor core, and 0.93% for enhancing tumor. Evaluations on the proposed systematic folds demonstrate that our model maintains competitive whole tumor accuracy while achieving clear average Dice gains of 1.16% for tumor core and 1.68% for enhancing tumor over existing state-of-the-art. Furthermore, our model achieves a 15x efficiency improvement while maintaining high segmentation accuracy, highlighting its robustness and computational advantage over existing methods.
>
---
#### [replaced 029] Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02076v3](http://arxiv.org/pdf/2412.02076v3)**

> **作者:** Bo Wen; Haochen Zhang; Dirk-Uwe G. Bartsch; William R. Freeman; Truong Q. Nguyen; Cheolhong An
>
> **摘要:** Topological correctness is critical for segmentation of tubular structures, which pervade in biomedical images. Existing topological segmentation loss functions are primarily based on the persistent homology of the image. They match the persistent features from the segmentation with the persistent features from the ground truth and minimize the difference between them. However, these methods suffer from an ambiguous matching problem since the matching only relies on the information in the topological space. In this work, we propose an effective and efficient Spatial-Aware Topological Loss Function that further leverages the information in the original spatial domain of the image to assist the matching of persistent features. Extensive experiments on images of various types of tubular structures show that the proposed method has superior performance in improving the topological accuracy of the segmentation compared with state-of-the-art methods. Code is available at https://github.com/JRC-VPLab/SATLoss.
>
---
#### [replaced 030] GaussianFusion: Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00034v2](http://arxiv.org/pdf/2506.00034v2)**

> **作者:** Shuai Liu; Quanmin Liang; Zefeng Li; Boyang Li; Kai Huang
>
> **备注:** Accepted at NeurIPS2025 (Spotlight)
>
> **摘要:** Multi-sensor fusion is crucial for improving the performance and robustness of end-to-end autonomous driving systems. Existing methods predominantly adopt either attention-based flatten fusion or bird's eye view fusion through geometric transformations. However, these approaches often suffer from limited interpretability or dense computational overhead. In this paper, we introduce GaussianFusion, a Gaussian-based multi-sensor fusion framework for end-to-end autonomous driving. Our method employs intuitive and compact Gaussian representations as intermediate carriers to aggregate information from diverse sensors. Specifically, we initialize a set of 2D Gaussians uniformly across the driving scene, where each Gaussian is parameterized by physical attributes and equipped with explicit and implicit features. These Gaussians are progressively refined by integrating multi-modal features. The explicit features capture rich semantic and spatial information about the traffic scene, while the implicit features provide complementary cues beneficial for trajectory planning. To fully exploit rich spatial and semantic information in Gaussians, we design a cascade planning head that iteratively refines trajectory predictions through interactions with Gaussians. Extensive experiments on the NAVSIM and Bench2Drive benchmarks demonstrate the effectiveness and robustness of the proposed GaussianFusion framework. The source code will be released at https://github.com/Say2L/GaussianFusion.
>
---
#### [replaced 031] MTFL: Multi-Timescale Feature Learning for Weakly-Supervised Anomaly Detection in Surveillance Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.05900v2](http://arxiv.org/pdf/2410.05900v2)**

> **作者:** Yiling Zhang; Erkut Akdag; Egor Bondarev; Peter H. N. De With
>
> **摘要:** Detection of anomaly events is relevant for public safety and requires a combination of fine-grained motion information and contextual events at variable time-scales. To this end, we propose a Multi-Timescale Feature Learning (MTFL) method to enhance the representation of anomaly features. Short, medium, and long temporal tubelets are employed to extract spatio-temporal video features using a Video Swin Transformer. Experimental results demonstrate that MTFL outperforms state-of-the-art methods on the UCF-Crime dataset, achieving an anomaly detection performance 89.78% AUC. Moreover, it performs complementary to SotA with 95.32% AUC on the ShanghaiTech and 84.57% AP on the XD-Violence dataset. Furthermore, we generate an extended dataset of the UCF-Crime for development and evaluation on a wider range of anomalies, namely Video Anomaly Detection Dataset (VADD), involving 2,591 videos in 18 classes with extensive coverage of realistic anomalies.
>
---
#### [replaced 032] Multispectral State-Space Feature Fusion: Bridging Shared and Cross-Parametric Interactions for Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.14643v2](http://arxiv.org/pdf/2507.14643v2)**

> **作者:** Jifeng Shen; Haibo Zhan; Shaohua Dong; Xin Zuo; Wankou Yang; Haibin Ling
>
> **备注:** submitted on 30/4/2025, Accepted by Information Fusion
>
> **摘要:** Modern multispectral feature fusion for object detection faces two critical limitations: (1) Excessive preference for local complementary features over cross-modal shared semantics adversely affects generalization performance; and (2) The trade-off between the receptive field size and computational complexity present critical bottlenecks for scalable feature modeling. Addressing these issues, a novel Multispectral State-Space Feature Fusion framework, dubbed MS2Fusion, is proposed based on the state space model (SSM), achieving efficient and effective fusion through a dual-path parametric interaction mechanism. More specifically, the first cross-parameter interaction branch inherits the advantage of cross-attention in mining complementary information with cross-modal hidden state decoding in SSM. The second shared-parameter branch explores cross-modal alignment with joint embedding to obtain cross-modal similar semantic features and structures through parameter sharing in SSM. Finally, these two paths are jointly optimized with SSM for fusing multispectral features in a unified framework, allowing our MS2Fusion to enjoy both functional complementarity and shared semantic space. In our extensive experiments on mainstream benchmarks including FLIR, M3FD and LLVIP, our MS2Fusion significantly outperforms other state-of-the-art multispectral object detection methods, evidencing its superiority. Moreover, MS2Fusion is general and applicable to other multispectral perception tasks. We show that, even without specific design, MS2Fusion achieves state-of-the-art results on RGB-T semantic segmentation and RGBT salient object detection, showing its generality. The source code will be available at https://github.com/61s61min/MS2Fusion.git.
>
---
#### [replaced 033] Superpowering Open-Vocabulary Object Detectors for X-ray Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17071v2](http://arxiv.org/pdf/2503.17071v2)**

> **作者:** Pablo Garcia-Fernandez; Lorenzo Vaquero; Mingxuan Liu; Feng Xue; Daniel Cores; Nicu Sebe; Manuel Mucientes; Elisa Ricci
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Open-vocabulary object detection (OvOD) is set to revolutionize security screening by enabling systems to recognize any item in X-ray scans. However, developing effective OvOD models for X-ray imaging presents unique challenges due to data scarcity and the modality gap that prevents direct adoption of RGB-based solutions. To overcome these limitations, we propose RAXO, a training-free framework that repurposes off-the-shelf RGB OvOD detectors for robust X-ray detection. RAXO builds high-quality X-ray class descriptors using a dual-source retrieval strategy. It gathers relevant RGB images from the web and enriches them via a novel X-ray material transfer mechanism, eliminating the need for labeled databases. These visual descriptors replace text-based classification in OvOD, leveraging intra-modal feature distances for robust detection. Extensive experiments demonstrate that RAXO consistently improves OvOD performance, providing an average mAP increase of up to 17.0 points over base detectors. To further support research in this emerging field, we also introduce DET-COMPASS, a new benchmark featuring bounding box annotations for over 300 object categories, enabling large-scale evaluation of OvOD in X-ray. Code and dataset available at: https://github.com/PAGF188/RAXO.
>
---
#### [replaced 034] CustomVideo: Customizing Text-to-Video Generation with Multiple Subjects
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.09962v3](http://arxiv.org/pdf/2401.09962v3)**

> **作者:** Zhao Wang; Aoxue Li; Lingting Zhu; Yong Guo; Qi Dou; Zhenguo Li
>
> **备注:** IEEE TMM 2025
>
> **摘要:** Customized text-to-video generation aims to generate high-quality videos guided by text prompts and subject references. Current approaches for personalizing text-to-video generation suffer from tackling multiple subjects, which is a more challenging and practical scenario. In this work, our aim is to promote multi-subject guided text-to-video customization. We propose CustomVideo, a novel framework that can generate identity-preserving videos with the guidance of multiple subjects. To be specific, firstly, we encourage the co-occurrence of multiple subjects via composing them in a single image. Further, upon a basic text-to-video diffusion model, we design a simple yet effective attention control strategy to disentangle different subjects in the latent space of diffusion model. Moreover, to help the model focus on the specific area of the object, we segment the object from given reference images and provide a corresponding object mask for attention learning. Also, we collect a multi-subject text-to-video generation dataset as a comprehensive benchmark. Extensive qualitative, quantitative, and user study results demonstrate the superiority of our method compared to previous state-of-the-art approaches. The project page is https://kyfafyd.wang/projects/customvideo.
>
---
#### [replaced 035] ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07862v2](http://arxiv.org/pdf/2502.07862v2)**

> **作者:** Jason Wu; Yuyang Yuan; Kang Yang; Lance Kaplan; Mani Srivastava
>
> **备注:** Accepted to Neurips 2025
>
> **摘要:** Multimodal deep learning systems are deployed in dynamic scenarios due to the robustness afforded by multiple sensing modalities. Nevertheless, they struggle with varying compute resource availability (due to multi-tenancy, device heterogeneity, etc.) and fluctuating quality of inputs (from sensor feed corruption, environmental noise, etc.). Statically provisioned multimodal systems cannot adapt when compute resources change over time, while existing dynamic networks struggle with strict compute budgets. Additionally, both systems often neglect the impact of variations in modality quality. Consequently, modalities suffering substantial corruption may needlessly consume resources better allocated towards other modalities. We propose ADMN, a layer-wise Adaptive Depth Multimodal Network capable of tackling both challenges: it adjusts the total number of active layers across all modalities to meet strict compute resource constraints and continually reallocates layers across input modalities according to their modality quality. Our evaluations showcase ADMN can match the accuracy of state-of-the-art networks while reducing up to 75% of their floating-point operations.
>
---
#### [replaced 036] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20426v3](http://arxiv.org/pdf/2505.20426v3)**

> **作者:** Yolo Yunlong Tang; Pinxin Liu; Zhangyun Tan; Mingqian Feng; Rui Mao; Chao Huang; Jing Bi; Yunzhong Xiao; Susan Liang; Hang Hua; Ali Vosoughi; Luchuan Song; Zeliang Zhang; Chenliang Xu
>
> **备注:** Accepted to NeurIPS 2025 DB Track
>
> **摘要:** Understanding perspective is fundamental to human visual perception, yet the extent to which multimodal large language models (MLLMs) internalize perspective geometry remains unclear. We introduce MMPerspective, the first benchmark specifically designed to systematically evaluate MLLMs' understanding of perspective through 10 carefully crafted tasks across three complementary dimensions: Perspective Perception, Reasoning, and Robustness. Our benchmark comprises 2,711 real-world and synthetic image instances with 5,083 question-answer pairs that probe key capabilities, such as vanishing point perception and counting, perspective type reasoning, line relationship understanding in 3D space, invariance to perspective-preserving transformations, etc. Through a comprehensive evaluation of 43 state-of-the-art MLLMs, we uncover significant limitations: while models demonstrate competence on surface-level perceptual tasks, they struggle with compositional reasoning and maintaining spatial consistency under perturbations. Our analysis further reveals intriguing patterns between model architecture, scale, and perspective capabilities, highlighting both robustness bottlenecks and the benefits of chain-of-thought prompting. MMPerspective establishes a valuable testbed for diagnosing and advancing spatial understanding in vision-language systems. Resources available at: https://yunlong10.github.io/MMPerspective/
>
---
#### [replaced 037] On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.00037v3](http://arxiv.org/pdf/2510.00037v3)**

> **作者:** Jianing Guo; Zhenhong Wu; Chang Tu; Yiyao Ma; Xiangqi Kong; Zhiqian Liu; Jiaming Ji; Shuning Zhang; Yuanpei Chen; Kai Chen; Qi Dou; Yaodong Yang; Xianglong Liu; Huijie Zhao; Weifeng Lv; Simin Li
>
> **摘要:** In Vision-Language-Action (VLA) models, robustness to real-world perturbations is critical for deployment. Existing methods target simple visual disturbances, overlooking the broader multi-modal perturbations that arise in actions, instructions, environments, and observations. Here, we first evaluate the robustness of mainstream VLAs under 17 perturbations across four modalities. We find (1) actions as the most fragile modality, (2) Existing visual-robust VLA do not gain robustness in other modality, and (3) pi0 demonstrates superior robustness with a diffusion-based action head. To build multi-modal robust VLAs, we propose RobustVLA against perturbations in VLA inputs and outputs. For output robustness, we perform offline robust optimization against worst-case action noise that maximizes mismatch in flow matching objective. This can be seen as adversarial training, label smoothing, and outlier penalization. For input robustness, we enforce consistent actions across input variations that preserve task semantics. To account for multiple perturbations, we formulate robustness as a multi-armed bandit problem and apply an upper confidence bound algorithm to automatically identify the most harmful noise. Experiments on LIBERO demonstrate our RobustVLA delivers absolute gains over baselines of 12.6% on the pi0 backbone and 10.4% on the OpenVLA backbone across all 17 perturbations, achieving 50.6x faster inference than existing visual-robust VLAs, and a 10.4% gain under mixed perturbations. Our RobustVLA is particularly effective on real-world FR5 robot with limited demonstrations, showing absolute gains by 65.6% under perturbations of four modalities.
>
---
#### [replaced 038] Polygonal network disorder and the turning distance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06415v2](http://arxiv.org/pdf/2503.06415v2)**

> **作者:** Alex Dolce; Ryan Lavelle; Bernard Scott; Ashlyn Urbanski; Joseph Klobusicky
>
> **摘要:** The turning distance is a well-studied metric for measuring the similarity between two polygons. This metric is constructed by taking an $L^p$ distance between step functions which track each shape's tangent angle of a path tracing its boundary. In this study, we introduce \textit{turning disorders} for polygonal planar networks, defined by averaging turning distances between network faces and "ordered" shapes (regular polygons or circles). We derive closed-form expressions of turning distances for special classes of regular polygons, related to the divisibility of $m$ and $n$, and also between regular polygons and circles. These formulas are used to show that the time for computing the 2-turning distances reduces to $O((m+n) \log(m+n))$ when both shapes are regular polygons, an improvement from $O(mn\log(mn))$ operations needed to compute distances between general polygons of $n$ and $m$ sides. We also apply these formulas to several examples of network microstructure with varying disorder. For Archimedean lattices, a class of regular tilings, we can express turning disorders with exact expressions. We also consider turning disorders applied to two examples of stochastic processes on networks: spring networks evolving under T1 moves and polygonal rupture processes. We find that the two aspects of defining different turning disorders, the choice of ordered shape and whether to apply area-weighting, can capture different notions of network disorder.
>
---
#### [replaced 039] Bridging the gap to real-world language-grounded visual concept learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.21412v2](http://arxiv.org/pdf/2510.21412v2)**

> **作者:** Whie Jung; Semin Kim; Junee Kim; Seunghoon Hong
>
> **摘要:** Human intelligence effortlessly interprets visual scenes along a rich spectrum of semantic dimensions. However, existing approaches to language-grounded visual concept learning are limited to a few predefined primitive axes, such as color and shape, and are typically explored in synthetic datasets. In this work, we propose a scalable framework that adaptively identifies image-related concept axes and grounds visual concepts along these axes in real-world scenes. Leveraging a pretrained vision-language model and our universal prompting strategy, our framework identifies a diverse image-related axes without any prior knowledge. Our universal concept encoder adaptively binds visual features to the discovered axes without introducing additional model parameters for each concept. To ground visual concepts along the discovered axes, we optimize a compositional anchoring objective, which ensures that each axis can be independently manipulated without affecting others. We demonstrate the effectiveness of our framework on subsets of ImageNet, CelebA-HQ, and AFHQ, showcasing superior editing capabilities across diverse real-world concepts that are too varied to be manually predefined. Our method also exhibits strong compositional generalization, outperforming existing visual concept learning and text-based editing methods. The code is available at https://github.com/whieya/Language-grounded-VCL.
>
---
#### [replaced 040] Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05177v3](http://arxiv.org/pdf/2502.05177v3)**

> **作者:** Yunhang Shen; Chaoyou Fu; Shaoqi Dong; Xiong Wang; Yi-Fan Zhang; Peixian Chen; Mengdan Zhang; Haoyu Cao; Ke Li; Shaohui Lin; Xiawu Zheng; Yan Zhang; Yiyi Zhou; Ran He; Caifeng Shan; Rongrong Ji; Xing Sun
>
> **备注:** https://github.com/VITA-MLLM/Long-VITA
>
> **摘要:** We introduce Long-VITA, a simple yet effective large multi-modal model for long-context visual-language understanding tasks. It is adept at concurrently processing and analyzing modalities of image, video, and text over 4K frames or 1M tokens while delivering advanced performances on short-context multi-modal tasks. We propose an effective multi-modal training schema that starts with large language models and proceeds through vision-language alignment, general knowledge learning, and two sequential stages of long-sequence fine-tuning. We further implement context-parallelism distributed inference and logits-masked language modeling head to scale Long-VITA to infinitely long inputs of images and texts during model inference. Regarding training data, Long-VITA is built on a mix of 17M samples from public datasets only and demonstrates state-of-the-art performance on various multi-modal benchmarks, compared against recent cutting-edge models with internal data. Long-VITA is fully open-source and reproducible.. By leveraging our inference designs, Long-VITA models achieve a remarkable 2x prefill speedup and 4x context length extension in a single node with 8 GPUs. We hope Long-VITA can serve as a competitive baseline and offer valuable insights for the open-source community in advancing long-context multi-modal understanding.
>
---
#### [replaced 041] Switchable Token-Specific Codebook Quantization For Face Image Compression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22943v2](http://arxiv.org/pdf/2510.22943v2)**

> **作者:** Yongbo Wang; Haonan Wang; Guodong Mu; Ruixin Zhang; Jiaqi Chen; Jingyun Zhang; Jun Wang; Yuan Xie; Zhizhong Zhang; Shouhong Ding
>
> **备注:** NeurIPS 2025 accepted
>
> **摘要:** With the ever-increasing volume of visual data, the efficient and lossless transmission, along with its subsequent interpretation and understanding, has become a critical bottleneck in modern information systems. The emerged codebook-based solution utilize a globally shared codebook to quantize and dequantize each token, controlling the bpp by adjusting the number of tokens or the codebook size. However, for facial images, which are rich in attributes, such global codebook strategies overlook both the category-specific correlations within images and the semantic differences among tokens, resulting in suboptimal performance, especially at low bpp. Motivated by these observations, we propose a Switchable Token-Specific Codebook Quantization for face image compression, which learns distinct codebook groups for different image categories and assigns an independent codebook to each token. By recording the codebook group to which each token belongs with a small number of bits, our method can reduce the loss incurred when decreasing the size of each codebook group. This enables a larger total number of codebooks under a lower overall bpp, thereby enhancing the expressive capability and improving reconstruction performance. Owing to its generalizable design, our method can be integrated into any existing codebook-based representation learning approach and has demonstrated its effectiveness on face recognition datasets, achieving an average accuracy of 93.51% for reconstructed images at 0.05 bpp.
>
---
#### [replaced 042] Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12702v2](http://arxiv.org/pdf/2505.12702v2)**

> **作者:** Tianming Liang; Haichao Jiang; Yuting Yang; Chaolei Tan; Shuai Li; Wei-Shi Zheng; Jian-Fang Hu
>
> **备注:** Project Page: \url{https://isee-laboratory.github.io/Long-RVOS}
>
> **摘要:** Referring video object segmentation (RVOS) aims to identify, track and segment the objects in a video based on language descriptions, which has received great attention in recent years. However, existing datasets remain focus on short video clips within several seconds, with salient objects visible in most frames. To advance the task towards more practical scenarios, we introduce \textbf{Long-RVOS}, a large-scale benchmark for long-term referring video object segmentation. Long-RVOS contains 2,000+ videos of an average duration exceeding 60 seconds, covering a variety of objects that undergo occlusion, disappearance-reappearance and shot changing. The objects are manually annotated with three different types of descriptions to individually evaluate the understanding of static attributes, motion patterns and spatiotemporal relationships. Moreover, unlike previous benchmarks that rely solely on the per-frame spatial evaluation, we introduce two new metrics to assess the temporal and spatiotemporal consistency. We benchmark 6 state-of-the-art methods on Long-RVOS. The results show that current approaches struggle severely with the long-video challenges. To address this, we further propose ReferMo, a promising baseline method that integrates motion information to expand the temporal receptive field, and employs a local-to-global architecture to capture both short-term dynamics and long-term dependencies. Despite simplicity, ReferMo achieves significant improvements over current methods in long-term scenarios. We hope that Long-RVOS and our baseline can drive future RVOS research towards tackling more realistic and long-form videos.
>
---
#### [replaced 043] VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23497v2](http://arxiv.org/pdf/2510.23497v2)**

> **作者:** Walid Bousselham; Hilde Kuehne; Cordelia Schmid
>
> **备注:** www.walidbousselham.com/VOLD/
>
> **摘要:** Training vision-language models (VLMs) for complex reasoning remains a challenging task, i.a. due to the scarcity of high-quality image-text reasoning data. Conversely, text-based reasoning resources are abundant and scalable, but it is still an open question how to leveraging them for VLM reasoning. To address this problem, we propose VOLD, a framework to transfer reasoning capabilities from text-only teacher models to VLM student models. To this end, VOLD combines reinforcement learning via Group Relative Policy Optimization (GRPO) with on-policy distillation, which allows the student reasoning traces to be guided by the teacher model, resulting in a significant gain over using GRPO alone. We further show that a cold-start alignment is essential for an effective transfer during the online training phase in this scenario and that without sufficient distributional alignment between teacher and student, on-policy distillation fails to provide meaningful guidance. We evaluate VOLD across diverse benchmarks including MMMU-Pro, MathVision, MathVista, and LogicVista, showing that VOLD outperforms the baseline model significantly and improves over the state of the art by a margin. Our ablation shows the importance of a cold-start alignment via SFT for on-policy distillation with a text-only teacher.
>
---
#### [replaced 044] Is It Certainly a Deepfake? Reliability Analysis in Detection & Generation Ecosystem
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.17550v3](http://arxiv.org/pdf/2509.17550v3)**

> **作者:** Neslihan Kose; Anthony Rhodes; Umur Aybars Ciftci; Ilke Demir
>
> **备注:** Accepted for publication at the ICCV 2025 workshop - STREAM
>
> **摘要:** As generative models are advancing in quality and quantity for creating synthetic content, deepfakes begin to cause online mistrust. Deepfake detectors are proposed to counter this effect, however, misuse of detectors claiming fake content as real or vice versa further fuels this misinformation problem. We present the first comprehensive uncertainty analysis of deepfake detectors, systematically investigating how generative artifacts influence prediction confidence. As reflected in detectors' responses, deepfake generators also contribute to this uncertainty as their generative residues vary, so we cross the uncertainty analysis of deepfake detectors and generators. Based on our observations, the uncertainty manifold holds enough consistent information to leverage uncertainty for deepfake source detection. Our approach leverages Bayesian Neural Networks and Monte Carlo dropout to quantify both aleatoric and epistemic uncertainties across diverse detector architectures. We evaluate uncertainty on two datasets with nine generators, with four blind and two biological detectors, compare different uncertainty methods, explore region- and pixel-based uncertainty, and conduct ablation studies. We conduct and analyze binary real/fake, multi-class real/fake, source detection, and leave-one-out experiments between the generator/detector combinations to share their generalization capability, model calibration, uncertainty, and robustness against adversarial attacks. We further introduce uncertainty maps that localize prediction confidence at the pixel level, revealing distinct patterns correlated with generator-specific artifacts. Our analysis provides critical insights for deploying reliable deepfake detection systems and establishes uncertainty quantification as a fundamental requirement for trustworthy synthetic media detection.
>
---
#### [replaced 045] MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition
- **分类: cs.CV; I.5.4; I.2.6; C.3**

- **链接: [http://arxiv.org/pdf/2505.20744v2](http://arxiv.org/pdf/2505.20744v2)**

> **作者:** Hao Zhang; Zhan Zhuang; Xuehao Wang; Xiaodong Yang; Yu Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Human Activity Recognition (HAR) with wearable sensors is challenged by limited interpretability, which significantly impacts cross-dataset generalization. To address this challenge, we propose Motion-Primitive Transformer (MoPFormer), a novel self-supervised framework that enhances interpretability by tokenizing inertial measurement unit signals into semantically meaningful motion primitives and leverages a Transformer architecture to learn rich temporal representations. MoPFormer comprises two stages. The first stage is to partition multi-channel sensor streams into short segments and quantize them into discrete ``motion primitive'' codewords, while the second stage enriches those tokenized sequences through a context-aware embedding module and then processes them with a Transformer encoder. The proposed MoPFormer can be pre-trained using a masked motion-modeling objective that reconstructs missing primitives, enabling it to develop robust representations across diverse sensor configurations. Experiments on six HAR benchmarks demonstrate that MoPFormer not only outperforms state-of-the-art methods but also successfully generalizes across multiple datasets. More importantly, the learned motion primitives significantly enhance both interpretability and cross-dataset performance by capturing fundamental movement patterns that remain consistent across similar activities, regardless of dataset origin.
>
---
#### [replaced 046] Unsupervised Monocular Depth Estimation Based on Hierarchical Feature-Guided Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.09782v3](http://arxiv.org/pdf/2406.09782v3)**

> **作者:** Runze Liu; Dongchen Zhu; Guanghui Zhang; Yue Xu; Wenjun Shi; Xiaolin Zhang; Lei Wang; Jiamao Li
>
> **摘要:** Unsupervised monocular depth estimation has received widespread attention because of its capability to train without ground truth. In real-world scenarios, the images may be blurry or noisy due to the influence of weather conditions and inherent limitations of the camera. Therefore, it is particularly important to develop a robust depth estimation model. Benefiting from the training strategies of generative networks, generative-based methods often exhibit enhanced robustness. In light of this, we employ a well-converging diffusion model among generative networks for unsupervised monocular depth estimation. Additionally, we propose a hierarchical feature-guided denoising module. This model significantly enriches the model's capacity for learning and interpreting depth distribution by fully leveraging image features to guide the denoising process. Furthermore, we explore the implicit depth within reprojection and design an implicit depth consistency loss. This loss function serves to enhance the performance of the model and ensure the scale consistency of depth within a video sequence. We conduct experiments on the KITTI, Make3D, and our self-collected SIMIT datasets. The results indicate that our approach stands out among generative-based models, while also showcasing remarkable robustness.
>
---
#### [replaced 047] TraceTrans: Translation and Spatial Tracing for Surgical Prediction
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.22379v2](http://arxiv.org/pdf/2510.22379v2)**

> **作者:** Xiyu Luo; Haodong Li; Xinxing Cheng; He Zhao; Yang Hu; Xuan Song; Tianyang Zhang
>
> **摘要:** Image-to-image translation models have achieved notable success in converting images across visual domains and are increasingly used for medical tasks such as predicting post-operative outcomes and modeling disease progression. However, most existing methods primarily aim to match the target distribution and often neglect spatial correspondences between the source and translated images. This limitation can lead to structural inconsistencies and hallucinations, undermining the reliability and interpretability of the predictions. These challenges are accentuated in clinical applications by the stringent requirement for anatomical accuracy. In this work, we present TraceTrans, a novel deformable image translation model designed for post-operative prediction that generates images aligned with the target distribution while explicitly revealing spatial correspondences with the pre-operative input. The framework employs an encoder for feature extraction and dual decoders for predicting spatial deformations and synthesizing the translated image. The predicted deformation field imposes spatial constraints on the generated output, ensuring anatomical consistency with the source. Extensive experiments on medical cosmetology and brain MRI datasets demonstrate that TraceTrans delivers accurate and interpretable post-operative predictions, highlighting its potential for reliable clinical deployment.
>
---
#### [replaced 048] RapVerse: Coherent Vocals and Whole-Body Motions Generations from Text
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2405.20336v2](http://arxiv.org/pdf/2405.20336v2)**

> **作者:** Jiaben Chen; Xin Yan; Yihang Chen; Siyuan Cen; Zixin Wang; Qinwei Ma; Haoyu Zhen; Kaizhi Qian; Lie Lu; Chuang Gan
>
> **备注:** ICCV 2025, Project website: https://jiabenchen.github.io/RapVerse/
>
> **摘要:** In this work, we introduce a challenging task for simultaneously generating 3D holistic body motions and singing vocals directly from textual lyrics inputs, advancing beyond existing works that typically address these two modalities in isolation. To facilitate this, we first collect the RapVerse dataset, a large dataset containing synchronous rapping vocals, lyrics, and high-quality 3D holistic body meshes. With the RapVerse dataset, we investigate the extent to which scaling autoregressive multimodal transformers across language, audio, and motion can enhance the coherent and realistic generation of vocals and whole-body human motions. For modality unification, a vector-quantized variational autoencoder is employed to encode whole-body motion sequences into discrete motion tokens, while a vocal-to-unit model is leveraged to obtain quantized audio tokens preserving content, prosodic information and singer identity. By jointly performing transformer modeling on these three modalities in a unified way, our framework ensures a seamless and realistic blend of vocals and human motions. Extensive experiments demonstrate that our unified generation framework not only produces coherent and realistic singing vocals alongside human motions directly from textual inputs, but also rivals the performance of specialized single-modality generation systems, establishing new benchmarks for joint vocal-motion generation.
>
---
#### [replaced 049] A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13043v2](http://arxiv.org/pdf/2505.13043v2)**

> **作者:** Hao-Ran Yang; Xiaohui Chen; Chuan-Xian Ren
>
> **备注:** NeurIPS 2025
>
> **摘要:** Aiming to generalize the well-trained gaze estimation model to new target domains, Cross-domain Gaze Estimation (CDGE) is developed for real-world application scenarios. Existing CDGE methods typically extract the domain-invariant features to mitigate domain shift in feature space, which is proved insufficient by Generalized Label Shift (GLS) theory. In this paper, we introduce a novel GLS perspective to CDGE and modelize the cross-domain problem by label and conditional shift problem. A GLS correction framework is presented and a feasible realization is proposed, in which a importance reweighting strategy based on truncated Gaussian distribution is introduced to overcome the continuity challenges in label shift correction. To embed the reweighted source distribution to conditional invariant learning, we further derive a probability-aware estimation of conditional operator discrepancy. Extensive experiments on standard CDGE tasks with different backbone models validate the superior generalization capability across domain and applicability on various models of proposed method.
>
---
#### [replaced 050] OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.15870v2](http://arxiv.org/pdf/2510.15870v2)**

> **作者:** Hanrong Ye; Chao-Han Huck Yang; Arushi Goel; Wei Huang; Ligeng Zhu; Yuanhang Su; Sean Lin; An-Chieh Cheng; Zhen Wan; Jinchuan Tian; Yuming Lou; Dong Yang; Zhijian Liu; Yukang Chen; Ambrish Dantrey; Ehsan Jahangiri; Sreyan Ghosh; Daguang Xu; Ehsan Hosseini-Asl; Danial Mohseni Taheri; Vidya Murali; Sifei Liu; Yao Lu; Oluwatobi Olabiyi; Yu-Chiang Frank Wang; Rafael Valle; Bryan Catanzaro; Andrew Tao; Song Han; Jan Kautz; Hongxu Yin; Pavlo Molchanov
>
> **备注:** Technical Report. Code: https://github.com/NVlabs/OmniVinci
>
> **摘要:** Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.
>
---
#### [replaced 051] From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04897v3](http://arxiv.org/pdf/2506.04897v3)**

> **作者:** Tianxu Wang; Zhuofan Zhang; Ziyu Zhu; Yue Fan; Jing Xiong; Pengxiang Li; Xiaojian Ma; Qing Li
>
> **备注:** Update v3 of the NeurIPS 2025 Datasets and Benchmarks paper (v2), including additional evaluations of state-of-the-art multimodal large language models. Project page: https://anywhere-3d.github.io/
>
> **摘要:** 3D visual grounding has made notable progress in localizing objects within complex 3D scenes. However, grounding referring expressions beyond objects in 3D scenes remains unexplored. In this paper, we introduce Anywhere3D-Bench, a holistic 3D visual grounding benchmark consisting of 2,886 referring expression-3D bounding box pairs spanning four different grounding levels: human-activity areas, unoccupied space beyond objects, individual objects in the scene, and fine-grained object parts. We assess a range of state-of-the-art 3D visual grounding methods alongside large language models (LLMs) and multimodal LLMs (MLLMs) on Anywhere3D-Bench. Experimental results reveal that space-level and part-level visual grounding pose the greatest challenges: space-level tasks require a more comprehensive spatial reasoning ability, for example, modeling distances and spatial relations within 3D space, while part-level tasks demand fine-grained perception of object composition. Even the best-performing models, Google Gemini-2.5-Pro and OpenAI o3, achieve just around 30% accuracy on space-level tasks and around 40% on part-level tasks, significantly lower than its performance on area-level and object-level tasks. These findings underscore a critical gap in current models' capacity to understand and reason about 3D scenes beyond object-level semantics.
>
---
#### [replaced 052] Caption-Driven Explainability: Probing CNNs for Bias via CLIP
- **分类: cs.CV; eess.IV; I.2.6; I.2.8; I.2.10; I.4.8**

- **链接: [http://arxiv.org/pdf/2510.22035v2](http://arxiv.org/pdf/2510.22035v2)**

> **作者:** Patrick Koller; Amil V. Dravid; Guido M. Schuster; Aggelos K. Katsaggelos
>
> **备注:** Accepted and presented at the IEEE ICIP 2025 Satellite Workshop "Generative AI for World Simulations and Communications & Celebrating 40 Years of Excellence in Education: Honoring Professor Aggelos Katsaggelos", Anchorage, Alaska, USA, September 14, 2025. Camera-ready preprint; the official IEEE Xplore publication will follow. Code is available at <https://github.com/patch0816/caption-driven-xai>
>
> **摘要:** Robustness has become one of the most critical problems in machine learning (ML). The science of interpreting ML models to understand their behavior and improve their robustness is referred to as explainable artificial intelligence (XAI). One of the state-of-the-art XAI methods for computer vision problems is to generate saliency maps. A saliency map highlights the pixel space of an image that excites the ML model the most. However, this property could be misleading if spurious and salient features are present in overlapping pixel spaces. In this paper, we propose a caption-based XAI method, which integrates a standalone model to be explained into the contrastive language-image pre-training (CLIP) model using a novel network surgery approach. The resulting caption-based XAI model identifies the dominant concept that contributes the most to the models prediction. This explanation minimizes the risk of the standalone model falling for a covariate shift and contributes significantly towards developing robust ML models. Our code is available at <https://github.com/patch0816/caption-driven-xai>.
>
---
#### [replaced 053] RETTA: Retrieval-Enhanced Test-Time Adaptation for Zero-Shot Video Captioning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.07046v3](http://arxiv.org/pdf/2405.07046v3)**

> **作者:** Yunchuan Ma; Laiyun Qing; Guorong Li; Yuankai Qi; Amin Beheshti; Quan Z. Sheng; Qingming Huang
>
> **备注:** Published in Pattern Recognition
>
> **摘要:** Despite the significant progress of fully-supervised video captioning, zero-shot methods remain much less explored. In this paper, we propose a novel zero-shot video captioning framework named Retrieval-Enhanced Test-Time Adaptation (RETTA), which takes advantage of existing pretrained large-scale vision and language models to directly generate captions with test-time adaptation. Specifically, we bridge video and text using four key models: a general video-text retrieval model XCLIP, a general image-text matching model CLIP, a text alignment model AnglE, and a text generation model GPT-2, due to their source-code availability. The main challenge is how to enable the text generation model to be sufficiently aware of the content in a given video so as to generate corresponding captions. To address this problem, we propose using learnable tokens as a communication medium among these four frozen models GPT-2, XCLIP, CLIP, and AnglE. Different from the conventional way that trains these tokens with training data, we propose to learn these tokens with soft targets of the inference data under several carefully crafted loss functions, which enable the tokens to absorb video information catered for GPT-2. This procedure can be efficiently done in just a few iterations (we use 16 iterations in the experiments) and does not require ground truth data. Extensive experimental results on three widely used datasets, MSR-VTT, MSVD, and VATEX, show absolute 5.1%-32.4% improvements in terms of the main metric CIDEr compared to several state-of-the-art zero-shot video captioning methods.
>
---
#### [replaced 054] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.26386v2](http://arxiv.org/pdf/2509.26386v2)**

> **作者:** Zhiwei Yang; Chen Gao; Mike Zheng Shou
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Video anomaly detection (VAD) is a critical yet challenging task due to the complex and diverse nature of real-world scenarios. Previous methods typically rely on domain-specific training data and manual adjustments when applying to new scenarios and unseen anomaly types, suffering from high labor costs and limited generalization. Therefore, we aim to achieve generalist VAD, \ie, automatically handle any scene and any anomaly types without training data or human involvement. In this work, we propose PANDA, an agentic AI engineer based on MLLMs. Specifically, we achieve PANDA by comprehensively devising four key capabilities: (1) self-adaptive scene-aware strategy planning, (2) goal-driven heuristic reasoning, (3) tool-augmented self-reflection, and (4) self-improving chain-of-memory. Concretely, we develop a self-adaptive scene-aware RAG mechanism, enabling PANDA to retrieve anomaly-specific knowledge for anomaly detection strategy planning. Next, we introduce a latent anomaly-guided heuristic prompt strategy to enhance reasoning precision. Furthermore, PANDA employs a progressive reflection mechanism alongside a suite of context-aware tools to iteratively refine decision-making in complex scenarios. Finally, a chain-of-memory mechanism enables PANDA to leverage historical experiences for continual performance improvement. Extensive experiments demonstrate that PANDA achieves state-of-the-art performance in multi-scenario, open-set, and complex scenario settings without training and manual involvement, validating its generalizable and robust anomaly detection capability. Code is released at https://github.com/showlab/PANDA.
>
---
#### [replaced 055] MsEdF: A Multi-stream Encoder-decoder Framework for Remote Sensing Image Captioning
- **分类: cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09282v4](http://arxiv.org/pdf/2502.09282v4)**

> **作者:** Swadhin Das; Raksha Sharma
>
> **摘要:** Remote sensing images contain complex spatial patterns and semantic structures, which makes the captioning model difficult to accurately describe. Encoder-decoder architectures have become the widely used approach for RSIC by translating visual content into descriptive text. However, many existing methods rely on a single-stream architecture, which weakens the model to accurately describe the image. Such single-stream architectures typically struggle to extract diverse spatial features or capture complex semantic relationships, limiting their effectiveness in scenes with high intraclass similarity or contextual ambiguity. In this work, we propose a novel Multi-stream Encoder-decoder Framework (MsEdF) which improves the performance of RSIC by optimizing both the spatial representation and language generation of encoder-decoder architecture. The encoder fuses information from two complementary image encoders, thereby promoting feature diversity through the integration of multiscale and structurally distinct cues. To improve the capture of context-aware descriptions, we refine the input sequence's semantic modeling on the decoder side using a stacked GRU architecture with an element-wise aggregation scheme. Experiments on three benchmark RSIC datasets show that MsEdF outperforms several baseline models.
>
---
#### [replaced 056] FaceCloak: Learning to Protect Face Templates
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06131v2](http://arxiv.org/pdf/2504.06131v2)**

> **作者:** Sudipta Banerjee; Anubhav Jain; Chinmay Hegde; Nasir Memon
>
> **备注:** Accepted in IEEE International Conference on Automatic Face and Gesture Recognition (FG 2025)
>
> **摘要:** Generative models can reconstruct face images from encoded representations (templates) bearing remarkable likeness to the original face, raising security and privacy concerns. We present \textsc{FaceCloak}, a neural network framework that protects face templates by generating smart, renewable binary cloaks. Our method proactively thwarts inversion attacks by cloaking face templates with unique disruptors synthesized from a single face template on the fly while provably retaining biometric utility and unlinkability. Our cloaked templates can suppress sensitive attributes while generalizing to novel feature extraction schemes and outperform leading baselines in terms of biometric matching and resiliency to reconstruction attacks. \textsc{FaceCloak}-based matching is extremely fast (inference time =0.28 ms) and light (0.57 MB). We have released our \href{https://github.com/sudban3089/FaceCloak.git}{code} for reproducible research.
>
---
#### [replaced 057] OmniResponse: Online Multimodal Conversational Response Generation in Dyadic Interactions
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.21724v2](http://arxiv.org/pdf/2505.21724v2)**

> **作者:** Cheng Luo; Jianghui Wang; Bing Li; Siyang Song; Bernard Ghanem
>
> **备注:** 25 pages, 9 figures
>
> **摘要:** In this paper, we introduce Online Multimodal Conversational Response Generation (OMCRG), a novel task designed to produce synchronized verbal and non-verbal listener feedback online, based on the speaker's multimodal inputs. OMCRG captures natural dyadic interactions and introduces new challenges in aligning generated audio with listeners' facial responses. To tackle these challenges, we incorporate text as an intermediate modality to connect audio and facial responses. We propose OmniResponse, a Multimodal Large Language Model (MLLM) that autoregressively generates accurate multimodal listener responses. OmniResponse leverages a pretrained LLM enhanced with two core components: Chrono-Text Markup, which precisely timestamps generated text tokens, and TempoVoice, a controllable online text-to-speech (TTS) module that outputs speech synchronized with facial responses. To advance OMCRG research, we offer ResponseNet, a dataset of 696 detailed dyadic interactions featuring synchronized split-screen videos, multichannel audio, transcripts, and annotated facial behaviors. Comprehensive evaluations on ResponseNet demonstrate that OmniResponse outperforms baseline models in terms of semantic speech content, audio-visual synchronization, and generation quality. Our dataset, code, and models are publicly available.
>
---
#### [replaced 058] Through the Lens: Benchmarking Deepfake Detectors Against Moiré-Induced Distortions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23225v2](http://arxiv.org/pdf/2510.23225v2)**

> **作者:** Razaib Tariq; Minji Heo; Simon S. Woo; Shahroz Tariq
>
> **备注:** 48 Pages, 29 Figures, 15 Tables
>
> **摘要:** Deepfake detection remains a pressing challenge, particularly in real-world settings where smartphone-captured media from digital screens often introduces Moir\'e artifacts that can distort detection outcomes. This study systematically evaluates state-of-the-art (SOTA) deepfake detectors on Moir\'e-affected videos, an issue that has received little attention. We collected a dataset of 12,832 videos, spanning 35.64 hours, from the Celeb-DF, DFD, DFDC, UADFV, and FF++ datasets, capturing footage under diverse real-world conditions, including varying screens, smartphones, lighting setups, and camera angles. To further examine the influence of Moir\'e patterns on deepfake detection, we conducted additional experiments using our DeepMoir\'eFake, referred to as (DMF) dataset and two synthetic Moir\'e generation techniques. Across 15 top-performing detectors, our results show that Moir\'e artifacts degrade performance by as much as 25.4%, while synthetically generated Moir\'e patterns lead to a 21.4% drop in accuracy. Surprisingly, demoir\'eing methods, intended as a mitigation approach, instead worsened the problem, reducing accuracy by up to 17.2%. These findings underscore the urgent need for detection models that can robustly handle Moir\'e distortions alongside other realworld challenges, such as compression, sharpening, and blurring. By introducing the DMF dataset, we aim to drive future research toward closing the gap between controlled experiments and practical deepfake detection.
>
---
#### [replaced 059] Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.02293v2](http://arxiv.org/pdf/2508.02293v2)**

> **作者:** Muhammad Aqeel; Shakiba Sharifi; Marco Cristani; Francesco Setti
>
> **备注:** Accepted to IEEE/CVF International Conference on Computer Vision (ICCV2025)
>
> **摘要:** So-called unsupervised anomaly detection is better described as semi-supervised, as it assumes all training data are nominal. This assumption simplifies training but requires manual data curation, introducing bias and limiting adaptability. We propose Confident Meta-learning (CoMet), a novel training strategy that enables deep anomaly detection models to learn from uncurated datasets where nominal and anomalous samples coexist, eliminating the need for explicit filtering. Our approach integrates Soft Confident Learning, which assigns lower weights to low-confidence samples, and Meta-Learning, which stabilizes training by regularizing updates based on training validation loss covariance. This prevents overfitting and enhances robustness to noisy data. CoMet is model-agnostic and can be applied to any anomaly detection method trainable via gradient descent. Experiments on MVTec-AD, VIADUCT, and KSDD2 with two state-of-the-art models demonstrate the effectiveness of our approach, consistently improving over the baseline methods, remaining insensitive to anomalies in the training set, and setting a new state-of-the-art across all datasets. Code is available at https://github.com/aqeeelmirza/CoMet
>
---
#### [replaced 060] GRASP: Geospatial pixel Reasoning viA Structured Policy learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.17102v2](http://arxiv.org/pdf/2508.17102v2)**

> **作者:** Chengjie Jiang; Yunqi Zhou; Jiafeng Yan; Jing Li; Jiayang Li; Yue Zhou; Hongjie He; Jonathan Li
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Geospatial pixel reasoning aims to generate segmentation masks in remote sensing imagery directly from natural-language instructions. Most existing approaches follow a paradigm that fine-tunes multimodal large language models under supervision with dense pixel-level masks as ground truth. While effective within the training data distribution, this design suffers from two main drawbacks: (1) the high cost of large-scale dense mask annotation, and (2) the limited generalization capability of supervised fine-tuning in out-of-domain scenarios. To address these issues, we propose GRASP, a structured policy-learning framework that integrates a multimodal large language model with a pretrained segmentation model in a cascaded manner. To enhance generalization, we introduce PRIME, a training paradigm that replaces supervised fine-tuning with reinforcement learning to better align reasoning and grounding behaviors with task objectives. To reduce annotation costs, we design BoP-Rewards, which substitutes dense mask labels with bounding box and positive points. It further verifies outputs through two complementary signals: format, which constrains the reasoning and grounding structure to remain syntactically parsable, and accuracy, which evaluates the quality of predicted boxes and points. For evaluation, we train our method and all baselines on EarthReason and GeoPixInstruct, constructing an in-domain benchmark by merging their test sets. We further release GRASP-1k, a fully out-of-domain benchmark with reasoning-intensive queries, reasoning traces, and fine-grained masks. Experimental results demonstrate state-of-the-art (SOTA) in-domain performance and up to 54\% improvement in out-of-domain scenarios, confirming that reinforcement learning with cost-aware rewards provides a robust and scalable paradigm for geospatial pixel reasoning. All code and datasets will be released publicly.
>
---
#### [replaced 061] Real-Time Neural Video Compression with Unified Intra and Inter Coding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14431v2](http://arxiv.org/pdf/2510.14431v2)**

> **作者:** Hui Xiang; Yifan Bian; Li Li; Jingran Wu; Xianguo Zhang; Dong Liu
>
> **备注:** 10 pages
>
> **摘要:** Neural video compression (NVC) technologies have advanced rapidly in recent years, yielding state-of-the-art schemes such as DCVC-RT that offer superior compression efficiency to H.266/VVC and real-time encoding/decoding capabilities. Nonetheless, existing NVC schemes have several limitations, including inefficiency in dealing with disocclusion and new content, interframe error propagation and accumulation, among others. To eliminate these limitations, we borrow the idea from classic video coding schemes, which allow intra coding within inter-coded frames. With the intra coding tool enabled, disocclusion and new content are properly handled, and interframe error propagation is naturally intercepted without the need for manual refresh mechanisms. We present an NVC framework with unified intra and inter coding, where every frame is processed by a single model that is trained to perform intra/inter coding adaptively. Moreover, we propose a simultaneous two-frame compression design to exploit interframe redundancy not only forwardly but also backwardly. Experimental results show that our scheme outperforms DCVC-RT by an average of 10.7\% BD-rate reduction, delivers more stable bitrate and quality per frame, and retains real-time encoding/decoding performances. Code and models will be released.
>
---
#### [replaced 062] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11842v3](http://arxiv.org/pdf/2505.11842v3)**

> **作者:** Xuannan Liu; Zekun Li; Zheqi He; Peipei Li; Shuhan Xia; Xing Cui; Huaibo Huang; Xi Yang; Ran He
>
> **备注:** Accepted by NeurIPS 2025 Dataset and Benchmark Track, Project page: https://liuxuannan.github.io/Video-SafetyBench.github.io/
>
> **摘要:** The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.
>
---
#### [replaced 063] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24424v2](http://arxiv.org/pdf/2505.24424v2)**

> **作者:** Amit Peleg; Naman Deep Singh; Matthias Hein
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Vision-language models like CLIP have demonstrated remarkable zero-shot capabilities in classification and retrieval. However, these models often struggle with compositional reasoning - the ability to understand the relationships between concepts. A recent benchmark, SugarCrepe++, reveals that previous works on improving compositionality have mainly improved lexical sensitivity but neglected semantic understanding. In addition, downstream retrieval performance often deteriorates, although one would expect that improving compositionality should enhance retrieval. In this work, we introduce CLIC (Compositionally-aware Learning in CLIP), a fine-tuning method based on a novel training technique combining multiple images and their associated captions. CLIC improves compositionality across architectures as well as differently pre-trained CLIP models, both in terms of lexical and semantic understanding, and achieves consistent gains in retrieval performance. This even applies to the recent CLIPS, which achieves SOTA retrieval performance. Nevertheless, the short fine-tuning with CLIC leads to an improvement in retrieval and to the best compositional CLIP model on SugarCrepe++. All our models and code are available at https://clic-compositional-clip.github.io
>
---
#### [replaced 064] DynCIM: Dynamic Curriculum for Imbalanced Multimodal Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06456v3](http://arxiv.org/pdf/2503.06456v3)**

> **作者:** Chengxuan Qian; Kai Han; Jiaxin Liu; Zhenlong Yuan; Zhengzhong Zhu; Jingchao Wang; Chongwen Lyu; Jun Chen; Zhe Liu
>
> **摘要:** Multimodal learning integrates complementary information from diverse modalities to enhance the decision-making process. However, the potential of multimodal collaboration remains under-exploited due to disparities in data quality and modality representation capabilities. To address this, we introduce DynCIM, a novel dynamic curriculum learning framework designed to quantify the inherent imbalances from both sample and modality perspectives. DynCIM employs a sample-level curriculum to dynamically assess each sample's difficulty according to prediction deviation, consistency, and stability, while a modality-level curriculum measures modality contributions from global and local. Furthermore, a gating-based dynamic fusion mechanism is introduced to adaptively adjust modality contributions, minimizing redundancy and optimizing fusion effectiveness. Extensive experiments on six multimodal benchmarking datasets, spanning both bimodal and trimodal scenarios, demonstrate that DynCIM consistently outperforms state-of-the-art methods. Our approach effectively mitigates modality and sample imbalances while enhancing adaptability and robustness in multimodal learning tasks. Our code is available at https://github.com/Raymond-Qiancx/DynCIM.
>
---
#### [replaced 065] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity-Aware Tree
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22693v2](http://arxiv.org/pdf/2510.22693v2)**

> **作者:** Wenlong Li; Yifei Xu; Yuan Rao; Zhenhua Wang; Shuiguang Deng
>
> **备注:** NeurIPS 2025 poster
>
> **摘要:** Video anomaly detection (VAD) focuses on identifying anomalies in videos. Supervised methods demand substantial in-domain training data and fail to deliver clear explanations for anomalies. In contrast, training-free methods leverage the knowledge reserves and language interactivity of large pre-trained models to detect anomalies. However, the current fixed-length temporal window sampling approaches struggle to accurately capture anomalies with varying temporal spans. Therefore, we propose VADTree that utilizes a Hierarchical Granularityaware Tree (HGTree) structure for flexible sampling in VAD. VADTree leverages the knowledge embedded in a pre-trained Generic Event Boundary Detection (GEBD) model to characterize potential anomaly event boundaries. Specifically, VADTree decomposes the video into generic event nodes based on boundary confidence, and performs adaptive coarse-fine hierarchical structuring and redundancy removal to construct the HGTree. Then, the multi-dimensional priors are injected into the visual language models (VLMs) to enhance the node-wise anomaly perception, and anomaly reasoning for generic event nodes is achieved via large language models (LLMs). Finally, an inter-cluster node correlation method is used to integrate the multi-granularity anomaly scores. Extensive experiments on three challenging datasets demonstrate that VADTree achieves state-of-the-art performance in training-free settings while drastically reducing the number of sampled video segments. The code will be available at https://github.com/wenlongli10/VADTree.
>
---
#### [replaced 066] Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.03520v2](http://arxiv.org/pdf/2405.03520v2)**

> **作者:** Zheng Zhu; Xiaofeng Wang; Wangbo Zhao; Chen Min; Bohan Li; Nianchen Deng; Min Dou; Yuqi Wang; Botian Shi; Kai Wang; Chi Zhang; Yang You; Zhaoxiang Zhang; Dawei Zhao; Liang Xiao; Jian Zhao; Jiwen Lu; Guan Huang
>
> **备注:** This survey will be regularly updated at: https://github.com/GigaAI-research/General-World-Models-Survey
>
> **摘要:** General world models represent a crucial pathway toward achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications ranging from virtual environments to decision-making systems. Recently, the emergence of the Sora model has attained significant attention due to its remarkable simulation capabilities, which exhibits an incipient comprehension of physical laws. In this survey, we embark on a comprehensive exploration of the latest advancements in world models. Our analysis navigates through the forefront of generative methodologies in video generation, where world models stand as pivotal constructs facilitating the synthesis of highly realistic visual content. Additionally, we scrutinize the burgeoning field of autonomous-driving world models, meticulously delineating their indispensable role in reshaping transportation and urban mobility. Furthermore, we delve into the intricacies inherent in world models deployed within autonomous agents, shedding light on their profound significance in enabling intelligent interactions within dynamic environmental contexts. At last, we examine challenges and limitations of world models, and discuss their potential future directions. We hope this survey can serve as a foundational reference for the research community and inspire continued innovation. This survey will be regularly updated at: https://github.com/GigaAI-research/General-World-Models-Survey.
>
---
#### [replaced 067] UMCFuse: A Unified Multiple Complex Scenes Infrared and Visible Image Fusion Framework
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.02096v2](http://arxiv.org/pdf/2402.02096v2)**

> **作者:** Xilai Li; Xiaosong Li; Tianshu Tan; Huafeng Li; Tao Ye
>
> **备注:** Published in IEEE-TIP 2025
>
> **摘要:** Infrared and visible image fusion has emerged as a prominent research area in computer vision. However, little attention has been paid to the fusion task in complex scenes, leading to sub-optimal results under interference. To fill this gap, we propose a unified framework for infrared and visible images fusion in complex scenes, termed UMCFuse. Specifically, we classify the pixels of visible images from the degree of scattering of light transmission, allowing us to separate fine details from overall intensity. Maintaining a balance between interference removal and detail preservation is essential for the generalization capacity of the proposed method. Therefore, we propose an adaptive denoising strategy for the fusion of detail layers. Meanwhile, we fuse the energy features from different modalities by analyzing them from multiple directions. Extensive fusion experiments on real and synthetic complex scenes datasets cover adverse weather conditions, noise, blur, overexposure, fire, as well as downstream tasks including semantic segmentation, object detection, salient object detection, and depth estimation, consistently indicate the superiority of the proposed method compared with the recent representative methods. Our code is available at https://github.com/ixilai/UMCFuse.
>
---
#### [replaced 068] Task-Agnostic Fusion of Time Series and Imagery for Earth Observation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23118v2](http://arxiv.org/pdf/2510.23118v2)**

> **作者:** Gianfranco Basile; Johannes Jakubik; Benedikt Blumenstiel; Thomas Brunschwiler; Juan Bernabe Moreno
>
> **摘要:** We propose a task-agnostic framework for multimodal fusion of time series and single timestamp images, enabling cross-modal generation and robust downstream performance. Our approach explores deterministic and learned strategies for time series quantization and then leverages a masked correlation learning objective, aligning discrete image and time series tokens in a unified representation space. Instantiated in the Earth observation domain, the pretrained model generates consistent global temperature profiles from satellite imagery and is validated through counterfactual experiments. Across downstream tasks, our task-agnostic pretraining outperforms task-specific fusion by 6% in R^2 and 2% in RMSE on average, and exceeds baseline methods by 50\% in R$^2$ and 12\% in RMSE. Finally, we analyze gradient sensitivity across modalities, providing insights into model robustness. Code, data, and weights will be released under a permissive license.
>
---
#### [replaced 069] Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark
- **分类: cs.CL; cs.AI; cs.CV; cs.DL**

- **链接: [http://arxiv.org/pdf/2510.19585v2](http://arxiv.org/pdf/2510.19585v2)**

> **作者:** Yu Wu; Ke Shu; Jonas Fischer; Lidia Pivovarova; David Rosson; Eetu Mäkelä; Mikko Tolonen
>
> **备注:** Under review. Both the dataset and code will be published
>
> **摘要:** This paper presents a novel task of extracting Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary models is achievable. Our study provides the first comprehensive analysis of these models' capabilities and limits for this task.
>
---
#### [replaced 070] Federated Learning with Partially Labeled Data: A Conditional Distillation Approach
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.18833v2](http://arxiv.org/pdf/2412.18833v2)**

> **作者:** Pochuan Wang; Chen Shen; Masahiro Oda; Chiou-Shann Fuh; Kensaku Mori; Weichung Wang; Holger R. Roth
>
> **备注:** This manuscript was submitted to IEEE JBHI and is currently under peer review
>
> **摘要:** In medical imaging, developing generalized segmentation models that can handle multiple organs and lesions is crucial. However, the scarcity of fully annotated datasets and strict privacy regulations present significant barriers to data sharing. Federated Learning (FL) allows decentralized model training, but existing FL methods often struggle with partial labeling, leading to model divergence and catastrophic forgetting. We propose ConDistFL, a novel FL framework incorporating conditional distillation to address these challenges. ConDistFL enables effective learning from partially labeled datasets, significantly improving segmentation accuracy across distributed and non-uniform datasets. In addition to its superior segmentation performance, ConDistFL maintains computational and communication efficiency, ensuring its scalability for real-world applications. Furthermore, ConDistFL demonstrates remarkable generalizability, significantly outperforming existing FL methods in out-of-federation tests, even adapting to unseen contrast phases (e.g., non-contrast CT images) in our experiments. Extensive evaluations on 3D CT and 2D chest X-ray datasets show that ConDistFL is an efficient, adaptable solution for collaborative medical image segmentation in privacy-constrained settings.
>
---
#### [replaced 071] GEMeX-RMCoT: An Enhanced Med-VQA Dataset for Region-Aware Multimodal Chain-of-Thought Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.17939v2](http://arxiv.org/pdf/2506.17939v2)**

> **作者:** Bo Liu; Xiangyu Zhao; Along He; Yidi Chen; Huazhu Fu; Xiao-Ming Wu
>
> **备注:** Accepted at ACM MM 2025 (also known as GEMeX-ThinkVG)
>
> **摘要:** Medical visual question answering aims to support clinical decision-making by enabling models to answer natural language questions based on medical images. While recent advances in multi-modal learning have significantly improved performance, current methods still suffer from limited answer reliability and poor interpretability, impairing the ability of clinicians and patients to understand and trust model outputs. To address these limitations, this work first proposes a Region-Aware Multimodal Chain-of-Thought (RMCoT) dataset, in which the process of producing an answer is preceded by a sequence of intermediate reasoning steps that explicitly ground relevant visual regions of the medical image, thereby providing fine-grained explainability. Furthermore, we introduce a novel verifiable reward mechanism for reinforcement learning to guide post-training, improving the alignment between the model's reasoning process and its final answer. Remarkably, our method achieves comparable performance using only one-eighth of the training data, demonstrating the efficiency and effectiveness of the proposal. The dataset is available at https://www.med-vqa.com/GEMeX/.
>
---
#### [replaced 072] Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Model
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.23502v3](http://arxiv.org/pdf/2503.23502v3)**

> **作者:** Jannik Endres; Oliver Hahn; Charles Corbière; Simone Schaub-Meyer; Stefan Roth; Alexandre Alahi
>
> **备注:** Accepted at IROS 2025. Project page: https://vita-epfl.github.io/DFI-OmniStereo-website/
>
> **摘要:** Omnidirectional depth perception is essential for mobile robotics applications that require scene understanding across a full 360{\deg} field of view. Camera-based setups offer a cost-effective option by using stereo depth estimation to generate dense, high-resolution depth maps without relying on expensive active sensing. However, existing omnidirectional stereo matching approaches achieve only limited depth accuracy across diverse environments, depth ranges, and lighting conditions, due to the scarcity of real-world data. We present DFI-OmniStereo, a novel omnidirectional stereo matching method that leverages a large-scale pre-trained foundation model for relative monocular depth estimation within an iterative optimization-based stereo matching architecture. We introduce a dedicated two-stage training strategy to utilize the relative monocular depth features for our omnidirectional stereo matching before scale-invariant fine-tuning. DFI-OmniStereo achieves state-of-the-art results on the real-world Helvipad dataset, reducing disparity MAE by approximately 16% compared to the previous best omnidirectional stereo method.
>
---
#### [replaced 073] CPathAgent: An Agent-based Foundation Model for Interpretable High-Resolution Pathology Image Analysis Mimicking Pathologists' Diagnostic Logic
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20510v2](http://arxiv.org/pdf/2505.20510v2)**

> **作者:** Yuxuan Sun; Yixuan Si; Chenglu Zhu; Kai Zhang; Zhongyi Shui; Bowen Ding; Tao Lin; Lin Yang
>
> **备注:** 52 pages, 34 figures
>
> **摘要:** Recent advances in computational pathology have led to the emergence of numerous foundation models. These models typically rely on general-purpose encoders with multi-instance learning for whole slide image (WSI) classification or apply multimodal approaches to generate reports directly from images. However, these models cannot emulate the diagnostic approach of pathologists, who systematically examine slides at low magnification to obtain an overview before progressively zooming in on suspicious regions to formulate comprehensive diagnoses. Instead, existing models directly output final diagnoses without revealing the underlying reasoning process. To address this gap, we introduce CPathAgent, an innovative agent-based approach that mimics pathologists' diagnostic workflow by autonomously navigating across WSI based on observed visual features, thereby generating substantially more transparent and interpretable diagnostic summaries. To achieve this, we develop a multi-stage training strategy that unifies patch-level, region-level, and WSI-level capabilities within a single model, which is essential for replicating how pathologists understand and reason across diverse image scales. Additionally, we construct PathMMU-HR2, the first expert-validated benchmark for large region analysis. This represents a critical intermediate scale between patches and whole slides, reflecting a key clinical reality where pathologists typically examine several key large regions rather than entire slides at once. Extensive experiments demonstrate that CPathAgent consistently outperforms existing approaches across benchmarks at three different image scales, validating the effectiveness of our agent-based diagnostic approach and highlighting a promising direction for computational pathology.
>
---
#### [replaced 074] GS4: Generalizable Sparse Splatting Semantic SLAM
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06517v2](http://arxiv.org/pdf/2506.06517v2)**

> **作者:** Mingqi Jiang; Chanho Kim; Chen Ziwen; Li Fuxin
>
> **备注:** 17 pages, 6 figures
>
> **摘要:** Traditional SLAM algorithms excel at camera tracking, but typically produce incomplete and low-resolution maps that are not tightly integrated with semantics prediction. Recent work integrates Gaussian Splatting (GS) into SLAM to enable dense, photorealistic 3D mapping, yet existing GS-based SLAM methods require per-scene optimization that is slow and consumes an excessive number of Gaussians. We present GS4, the first generalizable GS-based semantic SLAM system. Compared with prior approaches, GS4 runs 10x faster, uses 10x fewer Gaussians, and achieves state-of-the-art performance across color, depth, semantic mapping and camera tracking. From an RGB-D video stream, GS4 incrementally builds and updates a set of 3D Gaussians using a feed-forward network. First, the Gaussian Prediction Model estimates a sparse set of Gaussian parameters from input frame, which integrates both color and semantic prediction with the same backbone. Then, the Gaussian Refinement Network merges new Gaussians with the existing set while avoiding redundancy. Finally, we propose to optimize GS for only 1-5 iterations that corrects drift and floaters when significant pose changes are detected. Experiments on the real-world ScanNet and ScanNet++ benchmarks demonstrate state-of-the-art semantic SLAM performance, with strong generalization capability shown through zero-shot transfer to the NYUv2 and TUM RGB-D datasets.
>
---
#### [replaced 075] DWaste: Greener AI for Waste Sorting using Mobile and Edge Devices
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.18513v2](http://arxiv.org/pdf/2510.18513v2)**

> **作者:** Suman Kunwar
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** The rise of convenience packaging has led to generation of enormous waste, making efficient waste sorting crucial for sustainable waste management. To address this, we developed DWaste, a computer vision-powered platform designed for real-time waste sorting on resource-constrained smartphones and edge devices, including offline functionality. We benchmarked various image classification models (EfficientNetV2S/M, ResNet50/101, MobileNet) and object detection (YOLOv8n, YOLOv11n) including our purposed YOLOv8n-CBAM model using our annotated dataset designed for recycling. We found a clear trade-off between accuracy and resource consumption: the best classifier, EfficientNetV2S, achieved high accuracy(~ 96%) but suffered from high latency (~ 0.22s) and elevated carbon emissions. In contrast, lightweight object detection models delivered strong performance (up to 80% mAP) with ultra-fast inference (~ 0.03s) and significantly smaller model sizes (< 7MB ), making them ideal for real-time, low-power use. Model quantization further maximized efficiency, substantially reducing model size and VRAM usage by up to 75%. Our work demonstrates the successful implementation of "Greener AI" models to support real-time, sustainable waste sorting on edge devices.
>
---
#### [replaced 076] AnyCap Project: A Unified Framework, Dataset, and Benchmark for Controllable Omni-modal Captioning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12841v2](http://arxiv.org/pdf/2507.12841v2)**

> **作者:** Yiming Ren; Zhiqiang Lin; Yu Li; Gao Meng; Weiyun Wang; Junjie Wang; Zicheng Lin; Jifeng Dai; Yujiu Yang; Wenhai Wang; Ruihang Chu
>
> **摘要:** Controllable captioning is essential for precise multimodal alignment and instruction following, yet existing models often lack fine-grained control and reliable evaluation protocols. To address this gap, we present the AnyCap Project, an integrated solution spanning model, dataset, and evaluation. We introduce AnyCapModel (ACM), a lightweight plug-and-play framework that enhances the controllability of existing foundation models for omni-modal captioning without retraining the base model. ACM reuses the original captions from base models while incorporating user instructions and modality features to generate improved captions. To remedy the data scarcity in controllable multimodal captioning, we build AnyCapDataset (ACD), covering three modalities, 28 user-instruction types, and 300\,k high-quality data entries. We further propose AnyCapEval, a new benchmark that provides more reliable evaluation metrics for controllable captioning by decoupling content accuracy and stylistic fidelity. ACM markedly improves caption quality across a diverse set of base models on AnyCapEval. Notably, ACM-8B raises GPT-4o\'s content scores by 45\% and style scores by 12\%, and it also achieves substantial gains on widely used benchmarks such as MIA-Bench and VidCapBench.
>
---
#### [replaced 077] Radar and Event Camera Fusion for Agile Robot Ego-Motion Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18443v2](http://arxiv.org/pdf/2506.18443v2)**

> **作者:** Yang Lyu; Zhenghao Zou; Yanfeng Li; Xiaohu Guo; Chunhui Zhao; Quan Pan
>
> **备注:** 2025.10.28 version v2 for TwistEstimator
>
> **摘要:** Achieving reliable ego motion estimation for agile robots, e.g., aerobatic aircraft, remains challenging because most robot sensors fail to respond timely and clearly to highly dynamic robot motions, often resulting in measurement blurring, distortion, and delays. In this paper, we propose an IMU-free and feature-association-free framework to achieve aggressive ego-motion velocity estimation of a robot platform in highly dynamic scenarios by combining two types of exteroceptive sensors, an event camera and a millimeter wave radar, First, we used instantaneous raw events and Doppler measurements to derive rotational and translational velocities directly. Without a sophisticated association process between measurement frames, the proposed method is more robust in texture-less and structureless environments and is more computationally efficient for edge computing devices. Then, in the back-end, we propose a continuous-time state-space model to fuse the hybrid time-based and event-based measurements to estimate the ego-motion velocity in a fixed-lagged smoother fashion. In the end, we validate our velometer framework extensively in self-collected experiment datasets. The results indicate that our IMU-free and association-free ego motion estimation framework can achieve reliable and efficient velocity output in challenging environments. The source code, illustrative video and dataset are available at https://github.com/ZzhYgwh/TwistEstimator.
>
---
#### [replaced 078] Stealthy Patch-Wise Backdoor Attack in 3D Point Cloud via Curvature Awareness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09336v3](http://arxiv.org/pdf/2503.09336v3)**

> **作者:** Yu Feng; Dingxin Zhang; Runkai Zhao; Yong Xia; Heng Huang; Weidong Cai
>
> **备注:** 13 pages, 6 figures, 11 tables
>
> **摘要:** Backdoor attacks pose a severe threat to deep neural networks (DNNs) by implanting hidden backdoors that can be activated with predefined triggers to manipulate model behaviors maliciously. Existing 3D point cloud backdoor attacks primarily rely on sample-wise global modifications, which suffer from low imperceptibility. Although optimization can improve stealthiness, optimizing sample-wise triggers significantly increases computational cost. To address these limitations, we propose the Stealthy Patch-Wise Backdoor Attack (SPBA), the first patch-wise backdoor attack framework for 3D point clouds. Specifically, SPBA decomposes point clouds into local patches and employs a curvature-based imperceptibility score to guide trigger injection into visually less sensitive patches. By optimizing a unified patch-wise trigger that perturbs spectral features of selected patches, SPBA significantly enhances optimization efficiency while maintaining high stealthiness. Extensive experiments on ModelNet40 and ShapeNetPart further demonstrate that SPBA surpasses prior state-of-the-art backdoor attacks in both attack effectiveness and resistance to defense methods. The code is available at https://github.com/HazardFY/SPBA.
>
---
#### [replaced 079] One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.22366v5](http://arxiv.org/pdf/2410.22366v5)**

> **作者:** Viacheslav Surkov; Chris Wendler; Antonio Mari; Mikhail Terekhov; Justin Deschenaux; Robert West; Caglar Gulcehre; David Bau
>
> **摘要:** For large language models (LLMs), sparse autoencoders (SAEs) have been shown to decompose intermediate representations that often are not interpretable directly into sparse sums of interpretable features, facilitating better control and subsequent analysis. However, similar analyses and approaches have been lacking for text-to-image models. We investigate the possibility of using SAEs to learn interpretable features for SDXL Turbo, a few-step text-to-image diffusion model. To this end, we train SAEs on the updates performed by transformer blocks within SDXL Turbo's denoising U-net in its 1-step setting. Interestingly, we find that they generalize to 4-step SDXL Turbo and even to the multi-step SDXL base model (i.e., a different model) without additional training. In addition, we show that their learned features are interpretable, causally influence the generation process, and reveal specialization among the blocks. We do so by creating RIEBench, a representation-based image editing benchmark, for editing images while they are generated by turning on and off individual SAE features. This allows us to track which transformer blocks' features are the most impactful depending on the edit category. Our work is the first investigation of SAEs for interpretability in text-to-image diffusion models and our results establish SAEs as a promising approach for understanding and manipulating the internal mechanisms of text-to-image models.
>
---
#### [replaced 080] Faces of Fairness: Examining Bias in Facial Expression Recognition Datasets and Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11049v2](http://arxiv.org/pdf/2502.11049v2)**

> **作者:** Mohammad Mehdi Hosseini; Ali Pourramezan Fard; Mohammad H. Mahoor
>
> **摘要:** Building AI systems, including Facial Expression Recognition (FER), involves two critical aspects: data and model design. Both components significantly influence bias and fairness in FER tasks. Issues related to bias and fairness in FER datasets and models remain underexplored. This study investigates bias sources in FER datasets and models. Four common FER datasets--AffectNet, ExpW, Fer2013, and RAF-DB--are analyzed. The findings demonstrate that AffectNet and ExpW exhibit high generalizability despite data imbalances. Additionally, this research evaluates the bias and fairness of six deep models, including three state-of-the-art convolutional neural network (CNN) models: MobileNet, ResNet, XceptionNet, as well as three transformer-based models: ViT, CLIP, and GPT-4o-mini. Experimental results reveal that while GPT-4o-mini and ViT achieve the highest accuracy scores, they also display the highest levels of bias. These findings underscore the urgent need for developing new methodologies to mitigate bias and ensure fairness in datasets and models, particularly in affective computing applications. See our implementation details at https://github.com/MMHosseini/bias_in_FER.
>
---
