# 计算机视觉 cs.CV

- **最新发布 105 篇**

- **更新 58 篇**

## 最新发布

#### [new 001] SmokeBench: Evaluating Multimodal Large Language Models for Wildfire Smoke Detection
- **分类: cs.CV**

- **简介: 该论文属于视觉安全监测任务，旨在解决野火烟雾早期检测难题。作者提出SmokeBench基准，评估多模态大模型在烟雾分类与定位上的表现，发现现有模型在早期小范围烟雾定位上普遍表现不佳，凸显改进必要性。**

- **链接: [https://arxiv.org/pdf/2512.11215v1](https://arxiv.org/pdf/2512.11215v1)**

> **作者:** Tianye Qi; Weihao Li; Nick Barnes
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Wildfire smoke is transparent, amorphous, and often visually confounded with clouds, making early-stage detection particularly challenging. In this work, we introduce a benchmark, called SmokeBench, to evaluate the ability of multimodal large language models (MLLMs) to recognize and localize wildfire smoke in images. The benchmark consists of four tasks: (1) smoke classification, (2) tile-based smoke localization, (3) grid-based smoke localization, and (4) smoke detection. We evaluate several MLLMs, including Idefics2, Qwen2.5-VL, InternVL3, Unified-IO 2, Grounding DINO, GPT-4o, and Gemini-2.5 Pro. Our results show that while some models can classify the presence of smoke when it covers a large area, all models struggle with accurate localization, especially in the early stages. Further analysis reveals that smoke volume is strongly correlated with model performance, whereas contrast plays a comparatively minor role. These findings highlight critical limitations of current MLLMs for safety-critical wildfire monitoring and underscore the need for methods that improve early-stage smoke localization.
>
---
#### [new 002] Physics-Informed Video Flare Synthesis and Removal Leveraging Motion Independence between Flare and Scene
- **分类: cs.CV**

- **简介: 该论文研究视频镜头眩光去除，解决动态眩光与场景运动独立导致的去除难题。提出物理引导的动态眩光合成与去除网络，构建首个视频眩光数据集，通过注意力机制和Mamba时序建模实现高质量、时空一致的眩光去除。**

- **链接: [https://arxiv.org/pdf/2512.11327v1](https://arxiv.org/pdf/2512.11327v1)**

> **作者:** Junqiao Wang; Yuanfei Huang; Hua Huang
>
> **摘要:** Lens flare is a degradation phenomenon caused by strong light sources. Existing researches on flare removal have mainly focused on images, while the spatiotemporal characteristics of video flare remain largely unexplored. Video flare synthesis and removal pose significantly greater challenges than in image, owing to the complex and mutually independent motion of flare, light sources, and scene content. This motion independence further affects restoration performance, often resulting in flicker and artifacts. To address this issue, we propose a physics-informed dynamic flare synthesis pipeline, which simulates light source motion using optical flow and models the temporal behaviors of both scattering and reflective flares. Meanwhile, we design a video flare removal network that employs an attention module to spatially suppress flare regions and incorporates a Mamba-based temporal modeling component to capture long range spatio-temporal dependencies. This motion-independent spatiotemporal representation effectively eliminates the need for multi-frame alignment, alleviating temporal aliasing between flares and scene content and thereby improving video flare removal performance. Building upon this, we construct the first video flare dataset to comprehensively evaluate our method, which includes a large set of synthetic paired videos and additional real-world videos collected from the Internet to assess generalization capability. Extensive experiments demonstrate that our method consistently outperforms existing video-based restoration and image-based flare removal methods on both real and synthetic videos, effectively removing dynamic flares while preserving light source integrity and maintaining spatiotemporal consistency of scene.
>
---
#### [new 003] Reconstruction as a Bridge for Event-Based Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文研究事件相机与多模态大语言模型的结合，旨在解决事件数据特性保留与帧基模型兼容性的权衡问题。提出FRT和ART两种重建方法，并构建首个真实世界事件问答基准EvQA，验证了重建作为桥梁的有效性。**

- **链接: [https://arxiv.org/pdf/2512.11510v1](https://arxiv.org/pdf/2512.11510v1)**

> **作者:** Hanyue Lou; Jiayi Zhou; Yang Zhang; Boyu Li; Yi Wang; Guangnan Ye; Boxin Shi
>
> **摘要:** Integrating event cameras with Multimodal Large Language Models (MLLMs) promises general scene understanding in challenging visual conditions, yet requires navigating a trade-off between preserving the unique advantages of event data and ensuring compatibility with frame-based models. We address this challenge by using reconstruction as a bridge, proposing a straightforward Frame-based Reconstruction and Tokenization (FRT) method and designing an efficient Adaptive Reconstruction and Tokenization (ART) method that leverages event sparsity. For robust evaluation, we introduce EvQA, the first objective, real-world benchmark for event-based MLLMs, comprising 1,000 event-Q&A pairs from 22 public datasets. Our experiments demonstrate that our methods achieve state-of-the-art performance on EvQA, highlighting the significant potential of MLLMs in event-based vision.
>
---
#### [new 004] Flowception: Temporally Expansive Flow Matching for Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Flowception，用于视频生成任务，解决现有方法在长视频生成中的误差累积与计算成本高问题。其通过交替进行离散帧插入与连续去噪，实现非自回归、可变长度生成，降低训练开销，并统一支持多种视频生成任务。**

- **链接: [https://arxiv.org/pdf/2512.11438v1](https://arxiv.org/pdf/2512.11438v1)**

> **作者:** Tariq Berrada Ifriqi; John Nguyen; Karteek Alahari; Jakob Verbeek; Ricky T. Q. Chen
>
> **摘要:** We present Flowception, a novel non-autoregressive and variable-length video generation framework. Flowception learns a probability path that interleaves discrete frame insertions with continuous frame denoising. Compared to autoregressive methods, Flowception alleviates error accumulation/drift as the frame insertion mechanism during sampling serves as an efficient compression mechanism to handle long-term context. Compared to full-sequence flows, our method reduces FLOPs for training three-fold, while also being more amenable to local attention variants, and allowing to learn the length of videos jointly with their content. Quantitative experimental results show improved FVD and VBench metrics over autoregressive and full-sequence baselines, which is further validated with qualitative results. Finally, by learning to insert and denoise frames in a sequence, Flowception seamlessly integrates different tasks such as image-to-video generation and video interpolation.
>
---
#### [new 005] E-CHUM: Event-based Cameras for Human Detection and Urban Monitoring
- **分类: cs.CV; eess.IV**

- **简介: 该论文探讨事件相机在人群检测与城市动态监测中的应用，旨在解决传统监控隐私泄露与低光照适应性差的问题。作者分析其优势与挑战，提出融合多传感器提升性能，并倡导利用事件相机结合机器学习实现高效、隐私保护的城市感知。**

- **链接: [https://arxiv.org/pdf/2512.11076v1](https://arxiv.org/pdf/2512.11076v1)**

> **作者:** Jack Brady; Andrew Dailey; Kristen Schang; Zo Vic Shong
>
> **摘要:** Understanding human movement and city dynamics has always been challenging. From traditional methods of manually observing the city's inhabitant, to using cameras, to now using sensors and more complex technology, the field of urban monitoring has evolved greatly. Still, there are more that can be done to unlock better practices for understanding city dynamics. This paper surveys how the landscape of urban dynamics studying has evolved with a particular focus on event-based cameras. Event-based cameras capture changes in light intensity instead of the RGB values that traditional cameras do. They offer unique abilities, like the ability to work in low-light, that can make them advantageous compared to other sensors. Through an analysis of event-based cameras, their applications, their advantages and challenges, and machine learning applications, we propose event-based cameras as a medium for capturing information to study urban dynamics. They offer the ability to capture important information while maintaining privacy. We also suggest multi-sensor fusion of event-based cameras and other sensors in the study of urban dynamics. Combining event-based cameras and infrared, event-LiDAR, or vibration has to potential to enhance the ability of event-based cameras and overcome the challenges that event-based cameras have.
>
---
#### [new 006] VGent: Visual Grounding via Modular Design for Disentangling Reasoning and Prediction
- **分类: cs.CV**

- **简介: 该论文研究多目标视觉定位任务，旨在解决现有方法因自回归生成导致的延迟与幻觉问题。提出VGent，采用模块化编码器-解码器架构，分离推理与定位，结合冻结MLLM与检测器提案，提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2512.11099v1](https://arxiv.org/pdf/2512.11099v1)**

> **作者:** Weitai Kang; Jason Kuen; Mengwei Ren; Zijun Wei; Yan Yan; Kangning Liu
>
> **备注:** 8 pages
>
> **摘要:** Current visual grounding models are either based on a Multimodal Large Language Model (MLLM) that performs auto-regressive decoding, which is slow and risks hallucinations, or on re-aligning an LLM with vision features to learn new special or object tokens for grounding, which may undermine the LLM's pretrained reasoning ability. In contrast, we propose VGent, a modular encoder-decoder architecture that explicitly disentangles high-level reasoning and low-level bounding box prediction. Specifically, a frozen MLLM serves as the encoder to provide untouched powerful reasoning capabilities, while a decoder takes high-quality boxes proposed by detectors as queries and selects target box(es) via cross-attending on encoder's hidden states. This design fully leverages advances in both object detection and MLLM, avoids the pitfalls of auto-regressive decoding, and enables fast inference. Moreover, it supports modular upgrades of both the encoder and decoder to benefit the whole system: we introduce (i) QuadThinker, an RL-based training paradigm for enhancing multi-target reasoning ability of the encoder; (ii) mask-aware label for resolving detection-segmentation ambiguity; and (iii) global target recognition to improve the recognition of all the targets which benefits the selection among augmented proposals. Experiments on multi-target visual grounding benchmarks show that VGent achieves a new state-of-the-art with +20.6% F1 improvement over prior methods, and further boosts gIoU by +8.2% and cIoU by +5.8% under visual reference challenges, while maintaining constant, fast inference latency.
>
---
#### [new 007] Fast and Explicit: Slice-to-Volume Reconstruction via 3D Gaussian Primitives with Analytic Point Spread Function Modeling
- **分类: cs.CV**

- **简介: 该论文研究胎儿MRI中的快速三维重建任务，旨在解决传统隐式神经表示因蒙特卡洛采样导致的计算瓶颈问题。作者提出基于各向异性高斯基元的显式表示方法，利用高斯函数卷积闭包性，推导出点扩散函数的解析解，实现快速精准的切片到体积重建，速度提升5–10倍。**

- **链接: [https://arxiv.org/pdf/2512.11624v1](https://arxiv.org/pdf/2512.11624v1)**

> **作者:** Maik Dannecker; Steven Jia; Nil Stolt-Ansó; Nadine Girard; Guillaume Auzias; François Rousseau; Daniel Rueckert
>
> **备注:** Under Review for MIDL 2026
>
> **摘要:** Recovering high-fidelity 3D images from sparse or degraded 2D images is a fundamental challenge in medical imaging, with broad applications ranging from 3D ultrasound reconstruction to MRI super-resolution. In the context of fetal MRI, high-resolution 3D reconstruction of the brain from motion-corrupted low-resolution 2D acquisitions is a prerequisite for accurate neurodevelopmental diagnosis. While implicit neural representations (INRs) have recently established state-of-the-art performance in self-supervised slice-to-volume reconstruction (SVR), they suffer from a critical computational bottleneck: accurately modeling the image acquisition physics requires expensive stochastic Monte Carlo sampling to approximate the point spread function (PSF). In this work, we propose a shift from neural network based implicit representations to Gaussian based explicit representations. By parameterizing the HR 3D image volume as a field of anisotropic Gaussian primitives, we leverage the property of Gaussians being closed under convolution and thus derive a \textit{closed-form analytical solution} for the forward model. This formulation reduces the previously intractable acquisition integral to an exact covariance addition ($\mathbfΣ_{obs} = \mathbfΣ_{HR} + \mathbfΣ_{PSF}$), effectively bypassing the need for compute-intensive stochastic sampling while ensuring exact gradient propagation. We demonstrate that our approach matches the reconstruction quality of self-supervised state-of-the-art SVR frameworks while delivering a 5$\times$--10$\times$ speed-up on neonatal and fetal data. With convergence often reached in under 30 seconds, our framework paves the way towards translation into clinical routine of real-time fetal 3D MRI. Code will be public at {https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR}.
>
---
#### [new 008] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition
- **分类: cs.CV**

- **简介: 该论文针对少样本动作识别，提出TS-FSAR框架，解决现有方法忽略非线性关系和任务特定信息、以及适配大模型难优化的问题。通过设计侧网络、任务特定距离相关匹配及正则化模块，提升小样本下的识别性能。**

- **链接: [https://arxiv.org/pdf/2512.11340v1](https://arxiv.org/pdf/2512.11340v1)**

> **作者:** Fei Long; Yao Zhang; Jiaming Lv; Jiangtao Xie; Peihua Li
>
> **备注:** 9 pages. 4 figures, conference
>
> **摘要:** Few-shot action recognition (FSAR) has recently made notable progress through set matching and efficient adaptation of large-scale pre-trained models. However, two key limitations persist. First, existing set matching metrics typically rely on cosine similarity to measure inter-frame linear dependencies and then perform matching with only instance-level information, thus failing to capture more complex patterns such as nonlinear relationships and overlooking task-specific cues. Second, for efficient adaptation of CLIP to FSAR, recent work performing fine-tuning via skip-fusion layers (which we refer to as side layers) has significantly reduced memory cost. However, the newly introduced side layers are often difficult to optimize under limited data conditions. To address these limitations, we propose TS-FSAR, a framework comprising three components: (1) a visual Ladder Side Network (LSN) for efficient CLIP fine-tuning; (2) a metric called Task-Specific Distance Correlation Matching (TS-DCM), which uses $α$-distance correlation to model both linear and nonlinear inter-frame dependencies and leverages a task prototype to enable task-specific matching; and (3) a Guiding LSN with Adapted CLIP (GLAC) module, which regularizes LSN using the adapted frozen CLIP to improve training for better $α$-distance correlation estimation under limited supervision. Extensive experiments on five widely-used benchmarks demonstrate that our TS-FSAR yields superior performance compared to prior state-of-the-arts.
>
---
#### [new 009] Evaluating the Efficacy of Sentinel-2 versus Aerial Imagery in Serrated Tussock Classification
- **分类: cs.CV**

- **简介: 该论文属遥感分类任务，旨在解决大范围监测入侵植物锯齿状针茅的问题。通过构建多时相Sentinel-2与航拍影像模型，比较其分类效果，验证了卫星影像在成本效益和 scalability 上的潜力。**

- **链接: [https://arxiv.org/pdf/2512.11267v1](https://arxiv.org/pdf/2512.11267v1)**

> **作者:** Rezwana Sultana; Manzur Murshed; Kathryn Sheffield; Singarayer Florentine; Tsz-Kwan Lee; Shyh Wei Teng
>
> **备注:** Accepted in Earthsense 2025 (IEEE INTERNATIONAL CONFERENCE ON NEXT-GEN TECHNOLOGIES OF ARTIFICIAL INTELLIGENCE AND GEOSCIENCE REMOTE SENSING)
>
> **摘要:** Invasive species pose major global threats to ecosystems and agriculture. Serrated tussock (\textit{Nassella trichotoma}) is a highly competitive invasive grass species that disrupts native grasslands, reduces pasture productivity, and increases land management costs. In Victoria, Australia, it presents a major challenge due to its aggressive spread and ecological impact. While current ground surveys and subsequent management practices are effective at small scales, they are not feasible for landscape-scale monitoring. Although aerial imagery offers high spatial resolution suitable for detailed classification, its high cost limits scalability. Satellite-based remote sensing provides a more cost-effective and scalable alternative, though often with lower spatial resolution. This study evaluates whether multi-temporal Sentinel-2 imagery, despite its lower spatial resolution, can provide a comparable and cost-effective alternative for landscape-scale monitoring of serrated tussock by leveraging its higher spectral resolution and seasonal phenological information. A total of eleven models have been developed using various combinations of spectral bands, texture features, vegetation indices, and seasonal data. Using a random forest classifier, the best-performing Sentinel-2 model (M76*) has achieved an Overall Accuracy (OA) of 68\% and an Overall Kappa (OK) of 0.55, slightly outperforming the best-performing aerial imaging model's OA of 67\% and OK of 0.52 on the same dataset. These findings highlight the potential of multi-seasonal feature-enhanced satellite-based models for scalable invasive species classification.
>
---
#### [new 010] Reducing Domain Gap with Diffusion-Based Domain Adaptation for Cell Counting
- **分类: cs.CV**

- **简介: 该论文属于细胞计数任务，旨在解决合成显微图像与真实图像间域差距大的问题。作者改进InST框架，结合自适应实例归一化与扩散模型反演，迁移真实图像风格至合成图像，显著缩小域差，提升模型性能，减少标注依赖。**

- **链接: [https://arxiv.org/pdf/2512.11763v1](https://arxiv.org/pdf/2512.11763v1)**

> **作者:** Mohammad Dehghanmanshadi; Wallapak Tavanapong
>
> **备注:** Accepted at ICMLA 2025
>
> **摘要:** Generating realistic synthetic microscopy images is critical for training deep learning models in label-scarce environments, such as cell counting with many cells per image. However, traditional domain adaptation methods often struggle to bridge the domain gap when synthetic images lack the complex textures and visual patterns of real samples. In this work, we adapt the Inversion-Based Style Transfer (InST) framework originally designed for artistic style transfer to biomedical microscopy images. Our method combines latent-space Adaptive Instance Normalization with stochastic inversion in a diffusion model to transfer the style from real fluorescence microscopy images to synthetic ones, while weakly preserving content structure. We evaluate the effectiveness of our InST-based synthetic dataset for downstream cell counting by pre-training and fine-tuning EfficientNet-B0 models on various data sources, including real data, hard-coded synthetic data, and the public Cell200-s dataset. Models trained with our InST-synthesized images achieve up to 37\% lower Mean Absolute Error (MAE) compared to models trained on hard-coded synthetic data, and a 52\% reduction in MAE compared to models trained on Cell200-s (from 53.70 to 25.95 MAE). Notably, our approach also outperforms models trained on real data alone (25.95 vs. 27.74 MAE). Further improvements are achieved when combining InST-synthesized data with lightweight domain adaptation techniques such as DACS with CutMix. These findings demonstrate that InST-based style transfer most effectively reduces the domain gap between synthetic and real microscopy data. Our approach offers a scalable path for enhancing cell counting performance while minimizing manual labeling effort. The source code and resources are publicly available at: https://github.com/MohammadDehghan/InST-Microscopy.
>
---
#### [new 011] Cross-modal Context-aware Learning for Visual Prompt Guided Multimodal Image Understanding in Remote Sensing
- **分类: cs.CV**

- **简介: 该论文研究遥感图像的多模态理解任务，旨在通过视觉提示（如框选区域）引导模型精准识别用户关注区域。针对现有方法难以聚焦目标及对象间相似性高的问题，提出CLV-Net，结合上下文感知掩码解码与跨模态对齐机制，提升分割与描述的准确性。**

- **链接: [https://arxiv.org/pdf/2512.11680v1](https://arxiv.org/pdf/2512.11680v1)**

> **作者:** Xu Zhang; Jiabin Fang; Zhuoming Ding; Jin Yuan; Xuan Liu; Qianjun Zhang; Zhiyong Li
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Recent advances in image understanding have enabled methods that leverage large language models for multimodal reasoning in remote sensing. However, existing approaches still struggle to steer models to the user-relevant regions when only simple, generic text prompts are available. Moreover, in large-scale aerial imagery many objects exhibit highly similar visual appearances and carry rich inter-object relationships, which further complicates accurate recognition. To address these challenges, we propose Cross-modal Context-aware Learning for Visual Prompt-Guided Multimodal Image Understanding (CLV-Net). CLV-Net lets users supply a simple visual cue, a bounding box, to indicate a region of interest, and uses that cue to guide the model to generate correlated segmentation masks and captions that faithfully reflect user intent. Central to our design is a Context-Aware Mask Decoder that models and integrates inter-object relationships to strengthen target representations and improve mask quality. In addition, we introduce a Semantic and Relationship Alignment module: a Cross-modal Semantic Consistency Loss enhances fine-grained discrimination among visually similar targets, while a Relationship Consistency Loss enforces alignment between textual relations and visual interactions. Comprehensive experiments on two benchmark datasets show that CLV-Net outperforms existing methods and establishes new state-of-the-art results. The model effectively captures user intent and produces precise, intention-aligned multimodal outputs.
>
---
#### [new 012] EditMGT: Unleashing Potentials of Masked Generative Transformers in Image Editing
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 该论文研究图像编辑任务，旨在解决扩散模型易误改非目标区域的问题。作者提出EditMGT，首次将掩码生成Transformer用于图像编辑，通过注意力引导的局部化和区域保持采样，实现精准编辑，兼顾高质量与高效率。**

- **链接: [https://arxiv.org/pdf/2512.11715v1](https://arxiv.org/pdf/2512.11715v1)**

> **作者:** Wei Chow; Linfeng Li; Lingdong Kong; Zefeng Li; Qi Xu; Hang Song; Tian Ye; Xian Wang; Jinbin Bai; Shilin Xu; Xiangtai Li; Junting Pan; Shaoteng Liu; Ran Zhou; Tianshu Yang; Songhua Liu
>
> **摘要:** Recent advances in diffusion models (DMs) have achieved exceptional visual quality in image editing tasks. However, the global denoising dynamics of DMs inherently conflate local editing targets with the full-image context, leading to unintended modifications in non-target regions. In this paper, we shift our attention beyond DMs and turn to Masked Generative Transformers (MGTs) as an alternative approach to tackle this challenge. By predicting multiple masked tokens rather than holistic refinement, MGTs exhibit a localized decoding paradigm that endows them with the inherent capacity to explicitly preserve non-relevant regions during the editing process. Building upon this insight, we introduce the first MGT-based image editing framework, termed EditMGT. We first demonstrate that MGT's cross-attention maps provide informative localization signals for localizing edit-relevant regions and devise a multi-layer attention consolidation scheme that refines these maps to achieve fine-grained and precise localization. On top of these adaptive localization results, we introduce region-hold sampling, which restricts token flipping within low-attention areas to suppress spurious edits, thereby confining modifications to the intended target regions and preserving the integrity of surrounding non-target areas. To train EditMGT, we construct CrispEdit-2M, a high-resolution dataset spanning seven diverse editing categories. Without introducing additional parameters, we adapt a pre-trained text-to-image MGT into an image editing model through attention injection. Extensive experiments across four standard benchmarks demonstrate that, with fewer than 1B parameters, our model achieves similarity performance while enabling 6 times faster editing. Moreover, it delivers comparable or superior editing quality, with improvements of 3.6% and 17.6% on style change and style transfer tasks, respectively.
>
---
#### [new 013] VFMF: World Modeling by Forecasting Vision Foundation Model Features
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究世界模型中的视觉预测任务，旨在解决现有方法在预测效率与不确定性建模间的权衡问题。作者提出VFMF，通过在视觉基础模型特征的紧凑隐空间中进行生成式扩散预测，实现多模态、高精度且可解释的未来状态预报。**

- **链接: [https://arxiv.org/pdf/2512.11225v1](https://arxiv.org/pdf/2512.11225v1)**

> **作者:** Gabrijel Boduljak; Yushi Lan; Christian Rupprecht; Andrea Vedaldi
>
> **摘要:** Forecasting from partial observations is central to world modeling. Many recent methods represent the world through images, and reduce forecasting to stochastic video generation. Although such methods excel at realism and visual fidelity, predicting pixels is computationally intensive and not directly useful in many applications, as it requires translating RGB into signals useful for decision making. An alternative approach uses features from vision foundation models (VFMs) as world representations, performing deterministic regression to predict future world states. These features can be directly translated into actionable signals such as semantic segmentation and depth, while remaining computationally efficient. However, deterministic regression averages over multiple plausible futures, undermining forecast accuracy by failing to capture uncertainty. To address this crucial limitation, we introduce a generative forecaster that performs autoregressive flow matching in VFM feature space. Our key insight is that generative modeling in this space requires encoding VFM features into a compact latent space suitable for diffusion. We show that this latent space preserves information more effectively than previously used PCA-based alternatives, both for forecasting and other applications, such as image generation. Our latent predictions can be easily decoded into multiple useful and interpretable output modalities: semantic segmentation, depth, surface normals, and even RGB. With matched architecture and compute, our method produces sharper and more accurate predictions than regression across all modalities. Our results suggest that stochastic conditional generation of VFM features offers a promising and scalable foundation for future world models.
>
---
#### [new 014] YawDD+: Frame-level Annotations for Accurate Yawn Prediction
- **分类: cs.CV**

- **简介: 该论文属驾驶员疲劳检测任务，旨在解决因视频级标注导致的时序噪声问题。作者提出YawDD+数据集，采用半自动标注与人工校验，提升帧级标注精度，显著提高打哈欠检测的准确率，并实现在边缘设备上的实时监测。**

- **链接: [https://arxiv.org/pdf/2512.11446v1](https://arxiv.org/pdf/2512.11446v1)**

> **作者:** Ahmed Mujtaba; Gleb Radchenko; Marc Masana; Radu Prodan
>
> **备注:** This paper is submitted at European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, 2026
>
> **摘要:** Driver fatigue remains a leading cause of road accidents, with 24\% of crashes involving drowsy drivers. While yawning serves as an early behavioral indicator of fatigue, existing machine learning approaches face significant challenges due to video-annotated datasets that introduce systematic noise from coarse temporal annotations. We develop a semi-automated labeling pipeline with human-in-the-loop verification, which we apply to YawDD, enabling more accurate model training. Training the established MNasNet classifier and YOLOv11 detector architectures on YawDD+ improves frame accuracy by up to 6\% and mAP by 5\% over video-level supervision, achieving 99.34\% classification accuracy and 95.69\% detection mAP. The resulting approach deliver up to 59.8 FPS on edge AI hardware (NVIDIA Jetson Nano), confirming that enhanced data quality alone supports on-device yawning monitoring without server-side computation.
>
---
#### [new 015] A Multi-Mode Structured Light 3D Imaging System with Multi-Source Information Fusion for Underwater Pipeline Detection
- **分类: cs.CV**

- **简介: 该论文针对水下管道检测任务，解决腐蚀缺陷高精度三维成像难题。提出多模态结构光系统，融合声学与光学信息，实现快速畸变校正、鲁棒位姿估计与缺陷重建，提升检测的准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2512.11354v1](https://arxiv.org/pdf/2512.11354v1)**

> **作者:** Qinghan Hu; Haijiang Zhu; Na Sun; Lei Chen; Zhengqiang Fan; Zhiqing Li
>
> **摘要:** Underwater pipelines are highly susceptible to corrosion, which not only shorten their service life but also pose significant safety risks. Compared with manual inspection, the intelligent real-time imaging system for underwater pipeline detection has become a more reliable and practical solution. Among various underwater imaging techniques, structured light 3D imaging can restore the sufficient spatial detail for precise defect characterization. Therefore, this paper develops a multi-mode underwater structured light 3D imaging system for pipeline detection (UW-SLD system) based on multi-source information fusion. First, a rapid distortion correction (FDC) method is employed for efficient underwater image rectification. To overcome the challenges of extrinsic calibration among underwater sensors, a factor graph-based parameter optimization method is proposed to estimate the transformation matrix between the structured light and acoustic sensors. Furthermore, a multi-mode 3D imaging strategy is introduced to adapt to the geometric variability of underwater pipelines. Given the presence of numerous disturbances in underwater environments, a multi-source information fusion strategy and an adaptive extended Kalman filter (AEKF) are designed to ensure stable pose estimation and high-accuracy measurements. In particular, an edge detection-based ICP (ED-ICP) algorithm is proposed. This algorithm integrates pipeline edge detection network with enhanced point cloud registration to achieve robust and high-fidelity reconstruction of defect structures even under variable motion conditions. Extensive experiments are conducted under different operation modes, velocities, and depths. The results demonstrate that the developed system achieves superior accuracy, adaptability and robustness, providing a solid foundation for autonomous underwater pipeline detection.
>
---
#### [new 016] Super-Resolved Canopy Height Mapping from Sentinel-2 Time Series Using LiDAR HD Reference Data across Metropolitan France
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感与森林监测任务，旨在解决细尺度树高制图难题。作者提出THREASURE-Net模型，利用Sentinel-2时序数据和LiDAR高度参考数据，实现无需预训练的端到端树高超分辨率反演，生成高精度年度树高图。**

- **链接: [https://arxiv.org/pdf/2512.11524v1](https://arxiv.org/pdf/2512.11524v1)**

> **作者:** Ekaterina Kalinicheva; Florian Helen; Stéphane Mermoz; Florian Mouret; Milena Planells
>
> **摘要:** Fine-scale forest monitoring is essential for understanding canopy structure and its dynamics, which are key indicators of carbon stocks, biodiversity, and forest health. Deep learning is particularly effective for this task, as it integrates spectral, temporal, and spatial signals that jointly reflect the canopy structure. To address this need, we introduce THREASURE-Net, a novel end-to-end framework for Tree Height Regression And Super-Resolution. The model is trained on Sentinel-2 time series using reference height metrics derived from LiDAR HD data at multiple spatial resolutions over Metropolitan France to produce annual height maps. We evaluate three model variants, producing tree-height predictions at 2.5 m, 5 m, and 10 m resolution. THREASURE-Net does not rely on any pretrained model nor on reference very high resolution optical imagery to train its super-resolution module; instead, it learns solely from LiDAR-derived height information. Our approach outperforms existing state-of-the-art methods based on Sentinel data and is competitive with methods based on very high resolution imagery. It can be deployed to generate high-precision annual canopy-height maps, achieving mean absolute errors of 2.62 m, 2.72 m, and 2.88 m at 2.5 m, 5 m, and 10 m resolution, respectively. These results highlight the potential of THREASURE-Net for scalable and cost-effective structural monitoring of temperate forests using only freely available satellite data. The source code for THREASURE-Net is available at: https://github.com/Global-Earth-Observation/threasure-net.
>
---
#### [new 017] Reliable Detection of Minute Targets in High-Resolution Aerial Imagery across Temporal Shifts
- **分类: cs.CV**

- **简介: 该论文属目标检测任务，旨在解决高分辨率航拍影像中因目标微小和时序变化导致的水稻幼苗检测难题。作者采用迁移学习初始化Faster R-CNN模型，构建无人机数据集，并在多个时间点测试集上验证模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.11360v1](https://arxiv.org/pdf/2512.11360v1)**

> **作者:** Mohammad Sadegh Gholizadeh; Amir Arsalan Rezapour; Hamidreza Shayegh; Ehsan Pazouki
>
> **摘要:** Efficient crop detection via Unmanned Aerial Vehicles is critical for scaling precision agriculture, yet it remains challenging due to the small scale of targets and environmental variability. This paper addresses the detection of rice seedlings in paddy fields by leveraging a Faster R-CNN architecture initialized via transfer learning. To overcome the specific difficulties of detecting minute objects in high-resolution aerial imagery, we curate a significant UAV dataset for training and rigorously evaluate the model's generalization capabilities. Specifically, we validate performance across three distinct test sets acquired at different temporal intervals, thereby assessing robustness against varying imaging conditions. Our empirical results demonstrate that transfer learning not only facilitates the rapid convergence of object detection models in agricultural contexts but also yields consistent performance despite domain shifts in image acquisition.
>
---
#### [new 018] Evaluating Foundation Models' 3D Understanding Through Multi-View Correspondence Analysis
- **分类: cs.CV**

- **简介: 该论文属于3D理解评估任务，旨在解决现有方法依赖微调、难以衡量模型内在3D推理能力的问题。作者提出一种无需微调的新基准，基于多视角图像匹配分析，直接评估基础模型的密集特征在3D场景中的表现，并在MVImgNet上对8个模型进行了评测。**

- **链接: [https://arxiv.org/pdf/2512.11574v1](https://arxiv.org/pdf/2512.11574v1)**

> **作者:** Valentina Lilova; Toyesh Chakravorty; Julian I. Bibo; Emma Boccaletti; Brandon Li; Lívia Baxová; Cees G. M. Snoek; Mohammadreza Salehi
>
> **备注:** NeurIPS 2025 UniReps workshop
>
> **摘要:** Benchmarking 3D spatial understanding of foundation models is essential for real-world applications such as robotics and autonomous driving. Existing evaluations often rely on downstream finetuning with linear heads or task-specific decoders, making it difficult to isolate the intrinsic 3D reasoning ability of pretrained encoders. In this work, we introduce a novel benchmark for in-context 3D scene understanding that requires no finetuning and directly probes the quality of dense visual features. Building on the Hummingbird framework, which evaluates in-context 2D scene understanding, we extend the setup to the 3D Multi-View ImageNet (MVImgNet) dataset. Given a set of images from objects in specific angles (keys), we benchmark the performance of segmenting novel views (queries) and report the scores in 4 categories of easy, medium, hard, and extreme based on the key-query view contrast. We benchmark 8 state-of-the-art foundation models and show DINO-based encoders remain competitive across large viewpoint shifts, while 3D-aware models like VGGT require dedicated multi-view adjustments. Our code is publicly available at https://github.com/ToyeshC/open-hummingbird-3d-eval .
>
---
#### [new 019] Boosting Skeleton-based Zero-Shot Action Recognition with Training-Free Test-Time Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究骨架数据的零样本动作识别，提出无需训练的测试时自适应框架Skeleton-Cache。它通过构建非参数化缓存存储全局与局部骨架描述符，并利用大语言模型生成语义权重，动态融合预测，提升对未见动作的泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.11458v1](https://arxiv.org/pdf/2512.11458v1)**

> **作者:** Jingmin Zhu; Anqi Zhu; Hossein Rahmani; Jun Liu; Mohammed Bennamoun; Qiuhong Ke
>
> **摘要:** We introduce Skeleton-Cache, the first training-free test-time adaptation framework for skeleton-based zero-shot action recognition (SZAR), aimed at improving model generalization to unseen actions during inference. Skeleton-Cache reformulates inference as a lightweight retrieval process over a non-parametric cache that stores structured skeleton representations, combining both global and fine-grained local descriptors. To guide the fusion of descriptor-wise predictions, we leverage the semantic reasoning capabilities of large language models (LLMs) to assign class-specific importance weights. By integrating these structured descriptors with LLM-guided semantic priors, Skeleton-Cache dynamically adapts to unseen actions without any additional training or access to training data. Extensive experiments on NTU RGB+D 60/120 and PKU-MMD II demonstrate that Skeleton-Cache consistently boosts the performance of various SZAR backbones under both zero-shot and generalized zero-shot settings. The code is publicly available at https://github.com/Alchemist0754/Skeleton-Cache.
>
---
#### [new 020] AutoRefiner: Improving Autoregressive Video Diffusion Models via Reflective Refinement Over the Stochastic Sampling Path
- **分类: cs.CV**

- **简介: 该论文针对自回归视频扩散模型（AR-VDM）生成质量不足的问题，提出AutoRefiner，通过路径式噪声 refinement 和反射式KV缓存，在不更新模型参数下提升采样保真度，实现高效即插即用的推理优化。**

- **链接: [https://arxiv.org/pdf/2512.11203v1](https://arxiv.org/pdf/2512.11203v1)**

> **作者:** Zhengyang Yu; Akio Hayakawa; Masato Ishii; Qingtao Yu; Takashi Shibuya; Jing Zhang; Yuki Mitsufuji
>
> **摘要:** Autoregressive video diffusion models (AR-VDMs) show strong promise as scalable alternatives to bidirectional VDMs, enabling real-time and interactive applications. Yet there remains room for improvement in their sample fidelity. A promising solution is inference-time alignment, which optimizes the noise space to improve sample fidelity without updating model parameters. Yet, optimization- or search-based methods are computationally impractical for AR-VDMs. Recent text-to-image (T2I) works address this via feedforward noise refiners that modulate sampled noises in a single forward pass. Can such noise refiners be extended to AR-VDMs? We identify the failure of naively extending T2I noise refiners to AR-VDMs and propose AutoRefiner-a noise refiner tailored for AR-VDMs, with two key designs: pathwise noise refinement and a reflective KV-cache. Experiments demonstrate that AutoRefiner serves as an efficient plug-in for AR-VDMs, effectively enhancing sample fidelity by refining noise along stochastic denoising paths.
>
---
#### [new 021] MultiEgo: A Multi-View Egocentric Video Dataset for 4D Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文聚焦多视角第一人称视频的4D动态场景重建任务，旨在解决现有数据集缺乏多视角第一人称同步数据的问题。作者构建了MultiEgo数据集，包含五种社交场景的同步第一人称视频与精确位姿标注，并设计采集系统与处理流程，验证了其在自由视点视频中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.11301v1](https://arxiv.org/pdf/2512.11301v1)**

> **作者:** Bate Li; Houqiang Zhong; Zhengxue Cheng; Qiang Hu; Qiang Wang; Li Song; Wenjun Zhang
>
> **备注:** ACM MM 2025 Dataset Track
>
> **摘要:** Multi-view egocentric dynamic scene reconstruction holds significant research value for applications in holographic documentation of social interactions. However, existing reconstruction datasets focus on static multi-view or single-egocentric view setups, lacking multi-view egocentric datasets for dynamic scene reconstruction. Therefore, we present MultiEgo, the first multi-view egocentric dataset for 4D dynamic scene reconstruction. The dataset comprises five canonical social interaction scenes: meetings, performances, and a presentation. Each scene provides five authentic egocentric videos captured by participants wearing AR glasses. We design a hardware-based data acquisition system and processing pipeline, achieving sub-millisecond temporal synchronization across views, coupled with accurate pose annotations. Experiment validation demonstrates the practical utility and effectiveness of our dataset for free-viewpoint video (FVV) applications, establishing MultiEgo as a foundational resource for advancing multi-view egocentric dynamic scene reconstruction research.
>
---
#### [new 022] SVG-T2I: Scaling Up Text-to-Image Latent Diffusion Model Without Variational Autoencoder
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决在视觉基础模型（VFM）表征空间中直接训练大尺度扩散模型的问题。作者提出SVG-T2I框架，无需变分自编码器即可在VFM特征空间内实现高质量文本到图像生成，并开源全部代码与模型。**

- **链接: [https://arxiv.org/pdf/2512.11749v1](https://arxiv.org/pdf/2512.11749v1)**

> **作者:** Minglei Shi; Haolin Wang; Borui Zhang; Wenzhao Zheng; Bohan Zeng; Ziyang Yuan; Xiaoshi Wu; Yuanxing Zhang; Huan Yang; Xintao Wang; Pengfei Wan; Kun Gai; Jie Zhou; Jiwen Lu
>
> **备注:** Code Repository: https://github.com/KlingTeam/SVG-T2I; Model Weights: https://huggingface.co/KlingTeam/SVG-T2I
>
> **摘要:** Visual generation grounded in Visual Foundation Model (VFM) representations offers a highly promising unified pathway for integrating visual understanding, perception, and generation. Despite this potential, training large-scale text-to-image diffusion models entirely within the VFM representation space remains largely unexplored. To bridge this gap, we scale the SVG (Self-supervised representations for Visual Generation) framework, proposing SVG-T2I to support high-quality text-to-image synthesis directly in the VFM feature domain. By leveraging a standard text-to-image diffusion pipeline, SVG-T2I achieves competitive performance, reaching 0.75 on GenEval and 85.78 on DPG-Bench. This performance validates the intrinsic representational power of VFMs for generative tasks. We fully open-source the project, including the autoencoder and generation model, together with their training, inference, evaluation pipelines, and pre-trained weights, to facilitate further research in representation-driven visual generation.
>
---
#### [new 023] DentalGPT: Incentivizing Multimodal Complex Reasoning in Dentistry
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出DentalGPT，旨在提升牙科多模态复杂推理能力。针对现有模型在牙科视觉细节理解和推理上的不足，构建了12万+标注数据集，并结合领域知识注入与强化学习，显著提升疾病分类与牙科视觉问答性能。**

- **链接: [https://arxiv.org/pdf/2512.11558v1](https://arxiv.org/pdf/2512.11558v1)**

> **作者:** Zhenyang Cai; Jiaming Zhang; Junjie Zhao; Ziyi Zeng; Yanchao Li; Jingyi Liang; Junying Chen; Yunjin Yang; Jiajun You; Shuzhi Deng; Tongfei Wang; Wanting Chen; Chunxiu Hao; Ruiqi Xie; Zhenwei Wen; Xiangyi Feng; Zou Ting; Jin Zou Lin; Jianquan Li; Guangjun Yu; Liangyi Chen; Junwen Wang; Shan Jiang; Benyou Wang
>
> **摘要:** Reliable interpretation of multimodal data in dentistry is essential for automated oral healthcare, yet current multimodal large language models (MLLMs) struggle to capture fine-grained dental visual details and lack sufficient reasoning ability for precise diagnosis. To address these limitations, we present DentalGPT, a specialized dental MLLM developed through high-quality domain knowledge injection and reinforcement learning. Specifically, the largest annotated multimodal dataset for dentistry to date was constructed by aggregating over 120k dental images paired with detailed descriptions that highlight diagnostically relevant visual features, making it the multimodal dataset with the most extensive collection of dental images to date. Training on this dataset significantly enhances the MLLM's visual understanding of dental conditions, while the subsequent reinforcement learning stage further strengthens its capability for multimodal complex reasoning. Comprehensive evaluations on intraoral and panoramic benchmarks, along with dental subsets of medical VQA benchmarks, show that DentalGPT achieves superior performance in disease classification and dental VQA tasks, outperforming many state-of-the-art MLLMs despite having only 7B parameters. These results demonstrate that high-quality dental data combined with staged adaptation provides an effective pathway for building capable and domain-specialized dental MLLMs.
>
---
#### [new 024] Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于立体匹配任务，旨在解决现有模型无法兼顾实时性与零样本泛化能力的问题。作者提出Fast-FoundationStereo，通过知识蒸馏、神经架构搜索和结构化剪枝实现高效加速，并构建真实场景数据集提升性能，首次实现实时零样本高精度立体匹配。**

- **链接: [https://arxiv.org/pdf/2512.11130v1](https://arxiv.org/pdf/2512.11130v1)**

> **作者:** Bowen Wen; Shaurya Dewan; Stan Birchfield
>
> **摘要:** Stereo foundation models achieve strong zero-shot generalization but remain computationally prohibitive for real-time applications. Efficient stereo architectures, on the other hand, sacrifice robustness for speed and require costly per-domain fine-tuning. To bridge this gap, we present Fast-FoundationStereo, a family of architectures that achieve, for the first time, strong zero-shot generalization at real-time frame rate. We employ a divide-and-conquer acceleration strategy with three components: (1) knowledge distillation to compress the hybrid backbone into a single efficient student; (2) blockwise neural architecture search for automatically discovering optimal cost filtering designs under latency budgets, reducing search complexity exponentially; and (3) structured pruning for eliminating redundancy in the iterative refinement module. Furthermore, we introduce an automatic pseudo-labeling pipeline used to curate 1.4M in-the-wild stereo pairs to supplement synthetic training data and facilitate knowledge distillation. The resulting model can run over 10x faster than FoundationStereo while closely matching its zero-shot accuracy, thus establishing a new state-of-the-art among real-time methods. Project page: https://nvlabs.github.io/Fast-FoundationStereo/
>
---
#### [new 025] Depth-Copy-Paste: Multimodal and Depth-Aware Compositing for Robust Face Detection
- **分类: cs.CV**

- **简介: 该论文属人脸检测任务，旨在解决传统复制粘贴数据增强中语义不一致、几何不合理的问题。提出Depth Copy Paste框架，结合多模态检索、精确分割与深度感知粘贴，生成语义兼容、几何合理的训练样本，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2512.11683v1](https://arxiv.org/pdf/2512.11683v1)**

> **作者:** Qiushi Guo
>
> **摘要:** Data augmentation is crucial for improving the robustness of face detection systems, especially under challenging conditions such as occlusion, illumination variation, and complex environments. Traditional copy paste augmentation often produces unrealistic composites due to inaccurate foreground extraction, inconsistent scene geometry, and mismatched background semantics. To address these limitations, we propose Depth Copy Paste, a multimodal and depth aware augmentation framework that generates diverse and physically consistent face detection training samples by copying full body person instances and pasting them into semantically compatible scenes. Our approach first employs BLIP and CLIP to jointly assess semantic and visual coherence, enabling automatic retrieval of the most suitable background images for the given foreground person. To ensure high quality foreground masks that preserve facial details, we integrate SAM3 for precise segmentation and Depth-Anything to extract only the non occluded visible person regions, preventing corrupted facial textures from being used in augmentation. For geometric realism, we introduce a depth guided sliding window placement mechanism that searches over the background depth map to identify paste locations with optimal depth continuity and scale alignment. The resulting composites exhibit natural depth relationships and improved visual plausibility. Extensive experiments show that Depth Copy Paste provides more diverse and realistic training data, leading to significant performance improvements in downstream face detection tasks compared with traditional copy paste and depth free augmentation methods.
>
---
#### [new 026] Learning from a Generative Oracle: Domain Adaptation for Restoration
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究图像恢复的域自适应任务，旨在解决预训练模型在真实场景中因域差异导致性能下降的问题。作者提出LEGO框架，利用生成模型生成伪真值，通过伪监督学习实现无需配对数据的后训练自适应，提升模型在未知域上的表现。**

- **链接: [https://arxiv.org/pdf/2512.11121v1](https://arxiv.org/pdf/2512.11121v1)**

> **作者:** Yuyang Hu; Mojtaba Sahraee-Ardakan; Arpit Bansal; Kangfu Mei; Christian Qi; Peyman Milanfar; Mauricio Delbracio
>
> **摘要:** Pre-trained image restoration models often fail on real-world, out-of-distribution degradations due to significant domain gaps. Adapting to these unseen domains is challenging, as out-of-distribution data lacks ground truth, and traditional adaptation methods often require complex architectural changes. We propose LEGO (Learning from a Generative Oracle), a practical three-stage framework for post-training domain adaptation without paired data. LEGO converts this unsupervised challenge into a tractable pseudo-supervised one. First, we obtain initial restorations from the pre-trained model. Second, we leverage a frozen, large-scale generative oracle to refine these estimates into high-quality pseudo-ground-truths. Third, we fine-tune the original model using a mixed-supervision strategy combining in-distribution data with these new pseudo-pairs. This approach adapts the model to the new distribution without sacrificing its original robustness or requiring architectural modifications. Experiments demonstrate that LEGO effectively bridges the domain gap, significantly improving performance on diverse real-world benchmarks.
>
---
#### [new 027] RoomPilot: Controllable Synthesis of Interactive Indoor Environments via Multimodal Semantic Parsing
- **分类: cs.CV**

- **简介: 该论文提出RoomPilot，旨在解决多模态输入下室内场景生成的可控性与交互性问题。通过构建室内领域特定语言（IDSL），统一解析文本或CAD输入，实现高质量、具物理一致性和交互语义的三维室内场景生成。**

- **链接: [https://arxiv.org/pdf/2512.11234v1](https://arxiv.org/pdf/2512.11234v1)**

> **作者:** Wentang Chen; Shougao Zhang; Yiman Zhang; Tianhao Zhou; Ruihui Li
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Generating controllable and interactive indoor scenes is fundamental to applications in game development, architectural visualization, and embodied AI training. Yet existing approaches either handle a narrow range of input modalities or rely on stochastic processes that hinder controllability. To overcome these limitations, we introduce RoomPilot, a unified framework that parses diverse multi-modal inputs--textual descriptions or CAD floor plans--into an Indoor Domain-Specific Language (IDSL) for indoor structured scene generation. The key insight is that a well-designed IDSL can act as a shared semantic representation, enabling coherent, high-quality scene synthesis from any single modality while maintaining interaction semantics. In contrast to conventional procedural methods that produce visually plausible but functionally inert layouts, RoomPilot leverages a curated dataset of interaction-annotated assets to synthesize environments exhibiting realistic object behaviors. Extensive experiments further validate its strong multi-modal understanding, fine-grained controllability in scene generation, and superior physical consistency and visual fidelity, marking a significant step toward general-purpose controllable 3D indoor scene generation.
>
---
#### [new 028] PersonaLive! Expressive Portrait Image Animation for Live Streaming
- **分类: cs.CV**

- **简介: 该论文研究实时人像动画生成任务，旨在解决现有扩散模型在直播场景中延迟高、难以实时生成的问题。提出PersonaLive框架，通过多阶段训练、隐式控制信号、外观蒸馏和流式生成机制，实现高效、低延迟的高质量表情动画生成。**

- **链接: [https://arxiv.org/pdf/2512.11253v1](https://arxiv.org/pdf/2512.11253v1)**

> **作者:** Zhiyuan Li; Chi-Man Pun; Chen Fang; Jue Wang; Xiaodong Cun
>
> **摘要:** Current diffusion-based portrait animation models predominantly focus on enhancing visual quality and expression realism, while overlooking generation latency and real-time performance, which restricts their application range in the live streaming scenario. We propose PersonaLive, a novel diffusion-based framework towards streaming real-time portrait animation with multi-stage training recipes. Specifically, we first adopt hybrid implicit signals, namely implicit facial representations and 3D implicit keypoints, to achieve expressive image-level motion control. Then, a fewer-step appearance distillation strategy is proposed to eliminate appearance redundancy in the denoising process, greatly improving inference efficiency. Finally, we introduce an autoregressive micro-chunk streaming generation paradigm equipped with a sliding training strategy and a historical keyframe mechanism to enable low-latency and stable long-term video generation. Extensive experiments demonstrate that PersonaLive achieves state-of-the-art performance with up to 7-22x speedup over prior diffusion-based portrait animation models.
>
---
#### [new 029] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出UFVideo，首个支持统一细粒度协作理解的视频大模型，旨在解决现有模型难以兼顾全局、时序与像素级多粒度视频理解的问题。通过视觉-语言对齐设计，实现跨尺度联合推理，并构建新评测集验证其优越性。**

- **链接: [https://arxiv.org/pdf/2512.11336v1](https://arxiv.org/pdf/2512.11336v1)**

> **作者:** Hewen Pan; Cong Wei; Dashuang Liang; Zepeng Huang; Pengfei Gao; Ziqi Zhou; Lulu Xue; Pengfei Yan; Xiaoming Wei; Minghui Li; Shengshan Hu
>
> **备注:** 22 pages, 13 figures, technical report
>
> **摘要:** With the advancement of multi-modal Large Language Models (LLMs), Video LLMs have been further developed to perform on holistic and specialized video understanding. However, existing works are limited to specialized video understanding tasks, failing to achieve a comprehensive and multi-grained video perception. To bridge this gap, we introduce UFVideo, the first Video LLM with unified multi-grained cooperative understanding capabilities. Specifically, we design unified visual-language guided alignment to flexibly handle video understanding across global, pixel and temporal scales within a single model. UFVideo dynamically encodes the visual and text inputs of different tasks and generates the textual response, temporal localization, or grounded mask. Additionally, to evaluate challenging multi-grained video understanding tasks, we construct the UFVideo-Bench consisting of three distinct collaborative tasks within the scales, which demonstrates UFVideo's flexibility and advantages over GPT-4o. Furthermore, we validate the effectiveness of our model across 9 public benchmarks covering various common video understanding tasks, providing valuable insights for future Video LLMs.
>
---
#### [new 030] MLLM Machine Unlearning via Visual Knowledge Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究MLLM机器遗忘任务，旨在安全删除模型中的敏感视觉知识。提出视觉知识蒸馏方法，利用中间视觉表征监督遗忘过程，仅微调视觉组件，有效提升遗忘效果、模型效用与效率，并验证了抗重学习攻击的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.11325v1](https://arxiv.org/pdf/2512.11325v1)**

> **作者:** Yuhang Wang; Zhenxing Niu; Haoxuan Ji; Guangyu He; Haichang Gao; Gang Hua
>
> **摘要:** Recently, machine unlearning approaches have been proposed to remove sensitive information from well-trained large models. However, most existing methods are tailored for LLMs, while MLLM-oriented unlearning remains at its early stage. Inspired by recent studies exploring the internal mechanisms of MLLMs, we propose to disentangle the visual and textual knowledge embedded within MLLMs and introduce a dedicated approach to selectively erase target visual knowledge while preserving textual knowledge. Unlike previous unlearning methods that rely on output-level supervision, our approach introduces a Visual Knowledge Distillation (VKD) scheme, which leverages intermediate visual representations within the MLLM as supervision signals. This design substantially enhances both unlearning effectiveness and model utility. Moreover, since our method only fine-tunes the visual components of the MLLM, it offers significant efficiency advantages. Extensive experiments demonstrate that our approach outperforms state-of-the-art unlearning methods in terms of both effectiveness and efficiency. Moreover, we are the first to evaluate the robustness of MLLM unlearning against relearning attacks.
>
---
#### [new 031] Kinetic Mining in Context: Few-Shot Action Synthesis via Text-to-Motion Distillation
- **分类: cs.CV**

- **简介: 该论文属人体动作识别任务，旨在解决标注动作数据稀缺问题。提出KineMIC框架，通过文本编码空间的语义对齐，将通用文本到动作模型迁移为适用于小样本动作合成的专用生成器，实现高效数据增强，显著提升识别精度。**

- **链接: [https://arxiv.org/pdf/2512.11654v1](https://arxiv.org/pdf/2512.11654v1)**

> **作者:** Luca Cazzola; Ahed Alboody
>
> **摘要:** The acquisition cost for large, annotated motion datasets remains a critical bottleneck for skeletal-based Human Activity Recognition (HAR). Although Text-to-Motion (T2M) generative models offer a compelling, scalable source of synthetic data, their training objectives, which emphasize general artistic motion, and dataset structures fundamentally differ from HAR's requirements for kinematically precise, class-discriminative actions. This disparity creates a significant domain gap, making generalist T2M models ill-equipped for generating motions suitable for HAR classifiers. To address this challenge, we propose KineMIC (Kinetic Mining In Context), a transfer learning framework for few-shot action synthesis. KineMIC adapts a T2M diffusion model to an HAR domain by hypothesizing that semantic correspondences in the text encoding space can provide soft supervision for kinematic distillation. We operationalize this via a kinetic mining strategy that leverages CLIP text embeddings to establish correspondences between sparse HAR labels and T2M source data. This process guides fine-tuning, transforming the generalist T2M backbone into a specialized few-shot Action-to-Motion generator. We validate KineMIC using HumanML3D as the source T2M dataset and a subset of NTU RGB+D 120 as the target HAR domain, randomly selecting just 10 samples per action class. Our approach generates significantly more coherent motions, providing a robust data augmentation source that delivers a +23.1% accuracy points improvement. Animated illustrations and supplementary materials are available at (https://lucazzola.github.io/publications/kinemic).
>
---
#### [new 032] Leveraging Text Guidance for Enhancing Demographic Fairness in Gender Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究性别分类中的 demographic fairness 问题，提出利用文本引导（如图像描述）增强模型公平性。通过图像-文本匹配与融合策略，在无须 demographic 标签的情况下提升跨群体准确性与公平性，缓解偏见。**

- **链接: [https://arxiv.org/pdf/2512.11015v1](https://arxiv.org/pdf/2512.11015v1)**

> **作者:** Anoop Krishnan
>
> **摘要:** In the quest for fairness in artificial intelligence, novel approaches to enhance it in facial image based gender classification algorithms using text guided methodologies are presented. The core methodology involves leveraging semantic information from image captions during model training to improve generalization capabilities. Two key strategies are presented: Image Text Matching (ITM) guidance and Image Text fusion. ITM guidance trains the model to discern fine grained alignments between images and texts to obtain enhanced multimodal representations. Image text fusion combines both modalities into comprehensive representations for improved fairness. Exensive experiments conducted on benchmark datasets demonstrate these approaches effectively mitigate bias and improve accuracy across gender racial groups compared to existing methods. Additionally, the unique integration of textual guidance underscores an interpretable and intuitive training paradigm for computer vision systems. By scrutinizing the extent to which semantic information reduces disparities, this research offers valuable insights into cultivating more equitable facial analysis algorithms. The proposed methodologies contribute to addressing the pivotal challenge of demographic bias in gender classification from facial images. Furthermore, this technique operates in the absence of demographic labels and is application agnostic.
>
---
#### [new 033] SSA3D: Text-Conditioned Assisted Self-Supervised Framework for Automatic Dental Abutment Design
- **分类: cs.CV**

- **简介: 该论文研究自动牙冠基台设计，解决因标注数据少导致的AI应用难题。提出SSA3D框架，结合自监督学习与文本提示，通过双分支结构共享特征，省去预训练微调过程，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2512.11507v1](https://arxiv.org/pdf/2512.11507v1)**

> **作者:** Mianjie Zheng; Xinquan Yang; Along He; Xuguang Li; Feilie Zhong; Xuefen Liu; Kun Tang; Zhicheng Zhang; Linlin Shen
>
> **摘要:** Abutment design is a critical step in dental implant restoration. However, manual design involves tedious measurement and fitting, and research on automating this process with AI is limited, due to the unavailability of large annotated datasets. Although self-supervised learning (SSL) can alleviate data scarcity, its need for pre-training and fine-tuning results in high computational costs and long training times. In this paper, we propose a Self-supervised assisted automatic abutment design framework (SS$A^3$D), which employs a dual-branch architecture with a reconstruction branch and a regression branch. The reconstruction branch learns to restore masked intraoral scan data and transfers the learned structural information to the regression branch. The regression branch then predicts the abutment parameters under supervised learning, which eliminates the separate pre-training and fine-tuning process. We also design a Text-Conditioned Prompt (TCP) module to incorporate clinical information (such as implant location, system, and series) into SS$A^3$D. This guides the network to focus on relevant regions and constrains the parameter predictions. Extensive experiments on a collected dataset show that SS$A^3$D saves half of the training time and achieves higher accuracy than traditional SSL methods. It also achieves state-of-the-art performance compared to other methods, significantly improving the accuracy and efficiency of automated abutment design.
>
---
#### [new 034] Few-Shot VLM-Based G-Code and HMI Verification in CNC Machining
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文针对CNC加工中手动G代码生成的验证问题，提出一种基于视觉语言模型（VLM）的少样本方法，联合检测G代码与HMI界面的错误。通过引入配对文本与截图数据及结构化提示，提升对多模态错误的识别能力，适用于CNC教学中的综合调试验证。**

- **链接: [https://arxiv.org/pdf/2512.11296v1](https://arxiv.org/pdf/2512.11296v1)**

> **作者:** Yasaman Hashem Pour; Nazanin Mahjourian; Vinh Nguyen
>
> **摘要:** Manual generation of G-code is important for learning the operation of CNC machines. Prior work in G-code verification uses Large-Language Models (LLMs), which primarily examine errors in the written programming. However, CNC machining requires extensive use and knowledge of the Human-Machine Interface (HMI), which displays machine status and errors. LLMs currently lack the capability to leverage knowledge of HMIs due to their inability to access the vision modality. This paper proposes a few-shot VLM-based verification approach that simultaneously evaluates the G-code and the HMI display for errors and safety status. The input dataset includes paired G-code text and associated HMI screenshots from a 15-slant-PRO lathe, including both correct and error-prone cases. To enable few-shot learning, the VLM is provided with a structured JSON schema based on prior heuristic knowledge. After determining the prompts, instances of G-code and HMI that either contain errors or are error free are used as few-shot examples to guide the VLM. The model was then evaluated in comparison to a zero-shot VLM through multiple scenarios of incorrect G-code and HMI errors with respect to per-slot accuracy. The VLM showed that few-shot prompting led to overall enhancement of detecting HMI errors and discrepancies with the G-code for more comprehensive debugging. Therefore, the proposed framework was demonstrated to be suitable for verification of manually generated G-code that is typically developed in CNC training.
>
---
#### [new 035] Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models
- **分类: cs.CV**

- **简介: 该论文研究文本到图像生成中的组合对齐问题，评估VAR与扩散模型在属性、空间关系等任务上的表现。通过多模型对比，揭示VAR模型在组合性方面的优势，建立未来研究的统一基线。**

- **链接: [https://arxiv.org/pdf/2512.11542v1](https://arxiv.org/pdf/2512.11542v1)**

> **作者:** Hossein Shahabadi; Niki Sepasian; Arash Marioriyad; Ali Sharifi-Zarchi; Mahdieh Soleymani Baghshah
>
> **摘要:** Achieving compositional alignment between textual descriptions and generated images - covering objects, attributes, and spatial relationships - remains a core challenge for modern text-to-image (T2I) models. Although diffusion-based architectures have been widely studied, the compositional behavior of emerging Visual Autoregressive (VAR) models is still largely unexamined. We benchmark six diverse T2I systems - SDXL, PixArt-$α$, Flux-Dev, Flux-Schnell, Infinity-2B, and Infinity-8B - across the full T2I-CompBench++ and GenEval suites, evaluating alignment in color and attribute binding, spatial relations, numeracy, and complex multi-object prompts. Across both benchmarks, Infinity-8B achieves the strongest overall compositional alignment, while Infinity-2B also matches or exceeds larger diffusion models in several categories, highlighting favorable efficiency-performance trade-offs. In contrast, SDXL and PixArt-$α$ show persistent weaknesses in attribute-sensitive and spatial tasks. These results provide the first systematic comparison of VAR and diffusion approaches to compositional alignment and establish unified baselines for the future development of the T2I model.
>
---
#### [new 036] SoccerMaster: A Vision Foundation Model for Soccer Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SoccerMaster，首个面向足球理解的视觉基础模型，旨在统一处理多种足球视觉任务。通过多任务预训练和新建数据集SoccerFactory，实现从目标检测到事件分类的全面优化，超越专用模型性能。**

- **链接: [https://arxiv.org/pdf/2512.11016v1](https://arxiv.org/pdf/2512.11016v1)**

> **作者:** Haolin Yang; Jiayuan Rao; Haoning Wu; Weidi Xie
>
> **摘要:** Soccer understanding has recently garnered growing research interest due to its domain-specific complexity and unique challenges. Unlike prior works that typically rely on isolated, task-specific expert models, this work aims to propose a unified model to handle diverse soccer visual understanding tasks, ranging from fine-grained perception (e.g., athlete detection) to semantic reasoning (e.g., event classification). Specifically, our contributions are threefold: (i) we present SoccerMaster, the first soccer-specific vision foundation model that unifies diverse understanding tasks within a single framework via supervised multi-task pretraining; (ii) we develop an automated data curation pipeline to generate scalable spatial annotations, and integrate them with various existing soccer video datasets to construct SoccerFactory, a comprehensive pretraining data resource; and (iii) we conduct extensive evaluations demonstrating that SoccerMaster consistently outperforms task-specific expert models across diverse downstream tasks, highlighting its breadth and superiority. The data, code, and model will be publicly available.
>
---
#### [new 037] Referring Change Detection in Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文研究遥感图像中的指代变化检测（RCD），旨在通过自然语言描述实现按需的特定变化识别。针对数据稀缺与类别不平衡问题，提出RCDNet网络与基于扩散的合成数据生成方法RCDGen，实现可扩展、精准的跨模态变化检测。**

- **链接: [https://arxiv.org/pdf/2512.11719v1](https://arxiv.org/pdf/2512.11719v1)**

> **作者:** Yilmaz Korkmaz; Jay N. Paranjape; Celso M. de Melo; Vishal M. Patel
>
> **备注:** 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
>
> **摘要:** Change detection in remote sensing imagery is essential for applications such as urban planning, environmental monitoring, and disaster management. Traditional change detection methods typically identify all changes between two temporal images without distinguishing the types of transitions, which can lead to results that may not align with specific user needs. Although semantic change detection methods have attempted to address this by categorizing changes into predefined classes, these methods rely on rigid class definitions and fixed model architectures, making it difficult to mix datasets with different label sets or reuse models across tasks, as the output channels are tightly coupled with the number and type of semantic classes. To overcome these limitations, we introduce Referring Change Detection (RCD), which leverages natural language prompts to detect specific classes of changes in remote sensing images. By integrating language understanding with visual analysis, our approach allows users to specify the exact type of change they are interested in. However, training models for RCD is challenging due to the limited availability of annotated data and severe class imbalance in existing datasets. To address this, we propose a two-stage framework consisting of (I) \textbf{RCDNet}, a cross-modal fusion network designed for referring change detection, and (II) \textbf{RCDGen}, a diffusion-based synthetic data generation pipeline that produces realistic post-change images and change maps for a specified category using only pre-change image, without relying on semantic segmentation masks and thereby significantly lowering the barrier to scalable data creation. Experiments across multiple datasets show that our framework enables scalable and targeted change detection. Project website is here: https://yilmazkorkmaz1.github.io/RCD.
>
---
#### [new 038] Weak-to-Strong Generalization Enables Fully Automated De Novo Training of Multi-head Mask-RCNN Model for Segmenting Densely Overlapping Cell Nuclei in Multiplex Whole-slice Brain Images
- **分类: cs.CV**

- **简介: 该论文针对多重全片脑图像中密集重叠细胞核的分割难题，提出弱到强泛化方法，实现无需人工标注的自动模型训练。扩展伪标签并提升覆盖，结合自诊断指标验证质量，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.11722v1](https://arxiv.org/pdf/2512.11722v1)**

> **作者:** Lin Bai; Xiaoyang Li; Liqiang Huang; Quynh Nguyen; Hien Van Nguyen; Saurabh Prasad; Dragan Maric; John Redell; Pramod Dash; Badrinath Roysam
>
> **摘要:** We present a weak to strong generalization methodology for fully automated training of a multi-head extension of the Mask-RCNN method with efficient channel attention for reliable segmentation of overlapping cell nuclei in multiplex cyclic immunofluorescent (IF) whole-slide images (WSI), and present evidence for pseudo-label correction and coverage expansion, the key phenomena underlying weak to strong generalization. This method can learn to segment de novo a new class of images from a new instrument and/or a new imaging protocol without the need for human annotations. We also present metrics for automated self-diagnosis of segmentation quality in production environments, where human visual proofreading of massive WSI images is unaffordable. Our method was benchmarked against five current widely used methods and showed a significant improvement. The code, sample WSI images, and high-resolution segmentation results are provided in open form for community adoption and adaptation.
>
---
#### [new 039] Autoregressive Video Autoencoder with Decoupled Temporal and Spatial Context
- **分类: cs.CV**

- **简介: 该论文研究视频自编码器，旨在解决现有方法时空信息耦合导致时序不一致的问题。提出ARVAE模型，通过解耦时空上下文，以自回归方式压缩与重建视频，提升重建质量与压缩效率，适用于任意长度视频，并在小数据下表现优异。**

- **链接: [https://arxiv.org/pdf/2512.11293v1](https://arxiv.org/pdf/2512.11293v1)**

> **作者:** Cuifeng Shen; Lumin Xu; Xingguo Zhu; Gengdai Liu
>
> **摘要:** Video autoencoders compress videos into compact latent representations for efficient reconstruction, playing a vital role in enhancing the quality and efficiency of video generation. However, existing video autoencoders often entangle spatial and temporal information, limiting their ability to capture temporal consistency and leading to suboptimal performance. To address this, we propose Autoregressive Video Autoencoder (ARVAE), which compresses and reconstructs each frame conditioned on its predecessor in an autoregressive manner, allowing flexible processing of videos with arbitrary lengths. ARVAE introduces a temporal-spatial decoupled representation that combines downsampled flow field for temporal coherence with spatial relative compensation for newly emerged content, achieving high compression efficiency without information loss. Specifically, the encoder compresses the current and previous frames into the temporal motion and spatial supplement, while the decoder reconstructs the original frame from the latent representations given the preceding frame. A multi-stage training strategy is employed to progressively optimize the model. Extensive experiments demonstrate that ARVAE achieves superior reconstruction quality with extremely lightweight models and small-scale training data. Moreover, evaluations on video generation tasks highlight its strong potential for downstream applications.
>
---
#### [new 040] Do We Need Reformer for Vision? An Experimental Comparison with Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决Vision Transformers计算复杂度高的问题。作者采用Reformer架构，通过LSH注意力降低复杂度，实验比较其与ViT在不同数据集上的效率与精度，发现ViT在实际应用中仍更高效。**

- **链接: [https://arxiv.org/pdf/2512.11260v1](https://arxiv.org/pdf/2512.11260v1)**

> **作者:** Ali El Bellaj; Mohammed-Amine Cheddadi; Rhassan Berber
>
> **摘要:** Transformers have recently demonstrated strong performance in computer vision, with Vision Transformers (ViTs) leveraging self-attention to capture both low-level and high-level image features. However, standard ViTs remain computationally expensive, since global self-attention scales quadratically with the number of tokens, which limits their practicality for high-resolution inputs and resource-constrained settings. In this work, we investigate the Reformer architecture as an alternative vision backbone. By combining patch-based tokenization with locality-sensitive hashing (LSH) attention, our model approximates global self-attention while reducing its theoretical time complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$ in the sequence length $n$. We evaluate the proposed Reformer-based vision model on CIFAR-10 to assess its behavior on small-scale datasets, on ImageNet-100 to study its accuracy--efficiency trade-off in a more realistic setting, and on a high-resolution medical imaging dataset to evaluate the model under longer token sequences. While the Reformer achieves higher accuracy on CIFAR-10 compared to our ViT-style baseline, the ViT model consistently outperforms the Reformer in our experiments in terms of practical efficiency and end-to-end computation time across the larger and higher-resolution settings. These results suggest that, despite the theoretical advantages of LSH-based attention, meaningful computation gains require sequence lengths substantially longer than those produced by typical high-resolution images.
>
---
#### [new 041] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator
- **分类: cs.CV**

- **简介: 该论文属于视频抠图任务，旨在解决现有数据集规模小、细节缺失的问题。提出学习型质量评估器（MQE）用于训练反馈与数据筛选，构建大规模真实数据集VMReal，并引入参考帧训练策略，显著提升长视频抠图效果。**

- **链接: [https://arxiv.org/pdf/2512.11782v1](https://arxiv.org/pdf/2512.11782v1)**

> **作者:** Peiqing Yang; Shangchen Zhou; Kai Hao; Qingyi Tao
>
> **备注:** Project page: https://pq-yang.github.io/projects/MatAnyone2/
>
> **摘要:** Video matting remains limited by the scale and realism of existing datasets. While leveraging segmentation data can enhance semantic stability, the lack of effective boundary supervision often leads to segmentation-like mattes lacking fine details. To this end, we introduce a learned Matting Quality Evaluator (MQE) that assesses semantic and boundary quality of alpha mattes without ground truth. It produces a pixel-wise evaluation map that identifies reliable and erroneous regions, enabling fine-grained quality assessment. The MQE scales up video matting in two ways: (1) as an online matting-quality feedback during training to suppress erroneous regions, providing comprehensive supervision, and (2) as an offline selection module for data curation, improving annotation quality by combining the strengths of leading video and image matting models. This process allows us to build a large-scale real-world video matting dataset, VMReal, containing 28K clips and 2.4M frames. To handle large appearance variations in long videos, we introduce a reference-frame training strategy that incorporates long-range frames beyond the local window for effective training. Our MatAnyone 2 achieves state-of-the-art performance on both synthetic and real-world benchmarks, surpassing prior methods across all metrics.
>
---
#### [new 042] Smudged Fingerprints: A Systematic Evaluation of the Robustness of AI Image Fingerprints
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI生成图像溯源任务，旨在评估现有模型指纹技术在对抗环境下的鲁棒性。研究构建了清除与伪造指纹的攻击方法，对14种指纹技术进行系统评测，发现当前方法在对抗攻击下性能显著下降，准确率与鲁棒性存在权衡，尚无方法能全面兼顾。**

- **链接: [https://arxiv.org/pdf/2512.11771v1](https://arxiv.org/pdf/2512.11771v1)**

> **作者:** Kai Yao; Marc Juarez
>
> **备注:** This work has been accepted for publication in the 4th IEEE Conference on Secure and Trustworthy Machine Learning (IEEE SaTML 2026). The final version will be available on IEEE Xplore
>
> **摘要:** Model fingerprint detection techniques have emerged as a promising approach for attributing AI-generated images to their source models, but their robustness under adversarial conditions remains largely unexplored. We present the first systematic security evaluation of these techniques, formalizing threat models that encompass both white- and black-box access and two attack goals: fingerprint removal, which erases identifying traces to evade attribution, and fingerprint forgery, which seeks to cause misattribution to a target model. We implement five attack strategies and evaluate 14 representative fingerprinting methods across RGB, frequency, and learned-feature domains on 12 state-of-the-art image generators. Our experiments reveal a pronounced gap between clean and adversarial performance. Removal attacks are highly effective, often achieving success rates above 80% in white-box settings and over 50% under constrained black-box access. While forgery is more challenging than removal, its success significantly varies across targeted models. We also identify a utility-robustness trade-off: methods with the highest attribution accuracy are often vulnerable to attacks. Although some techniques exhibit robustness in specific settings, none achieves high robustness and accuracy across all evaluated threat models. These findings highlight the need for techniques balancing robustness and accuracy, and identify the most promising approaches for advancing this goal.
>
---
#### [new 043] Structure From Tracking: Distilling Structure-Preserving Motion for Video Generation
- **分类: cs.CV**

- **简介: 该论文属视频生成任务，旨在解决生成过程中结构失真问题。提出SAM2VideoX，通过双向特征融合模块和局部Gram Flow损失，从跟踪模型蒸馏结构保持运动先验，提升生成视频的结构一致性与真实感。**

- **链接: [https://arxiv.org/pdf/2512.11792v1](https://arxiv.org/pdf/2512.11792v1)**

> **作者:** Yang Fei; George Stoica; Jingyuan Liu; Qifeng Chen; Ranjay Krishna; Xiaojuan Wang; Benlin Liu
>
> **备注:** Project Website: https://sam2videox.github.io/
>
> **摘要:** Reality is a dance between rigid constraints and deformable structures. For video models, that means generating motion that preserves fidelity as well as structure. Despite progress in diffusion models, producing realistic structure-preserving motion remains challenging, especially for articulated and deformable objects such as humans and animals. Scaling training data alone, so far, has failed to resolve physically implausible transitions. Existing approaches rely on conditioning with noisy motion representations, such as optical flow or skeletons extracted using an external imperfect model. To address these challenges, we introduce an algorithm to distill structure-preserving motion priors from an autoregressive video tracking model (SAM2) into a bidirectional video diffusion model (CogVideoX). With our method, we train SAM2VideoX, which contains two innovations: (1) a bidirectional feature fusion module that extracts global structure-preserving motion priors from a recurrent model like SAM2; (2) a Local Gram Flow loss that aligns how local features move together. Experiments on VBench and in human studies show that SAM2VideoX delivers consistent gains (+2.60\% on VBench, 21-22\% lower FVD, and 71.4\% human preference) over prior baselines. Specifically, on VBench, we achieve 95.51\%, surpassing REPA (92.91\%) by 2.60\%, and reduce FVD to 360.57, a 21.20\% and 22.46\% improvement over REPA- and LoRA-finetuning, respectively. The project website can be found at https://sam2videox.github.io/ .
>
---
#### [new 044] CADKnitter: Compositional CAD Generation from Text and Geometry Guidance
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出CADKnitter，解决多部件CAD模型组合生成任务。针对现有方法难以满足几何与语义约束的问题，设计几何引导扩散采样框架，并构建含31万样本的KnitCAD数据集，实现文本和几何引导下的可编辑CAD部件生成。**

- **链接: [https://arxiv.org/pdf/2512.11199v1](https://arxiv.org/pdf/2512.11199v1)**

> **作者:** Tri Le; Khang Nguyen; Baoru Huang; Tung D. Ta; Anh Nguyen
>
> **摘要:** Crafting computer-aided design (CAD) models has long been a painstaking and time-intensive task, demanding both precision and expertise from designers. With the emergence of 3D generation, this task has undergone a transformative impact, shifting not only from visual fidelity to functional utility but also enabling editable CAD designs. Prior works have achieved early success in single-part CAD generation, which is not well-suited for real-world applications, as multiple parts need to be assembled under semantic and geometric constraints. In this paper, we propose CADKnitter, a compositional CAD generation framework with a geometry-guided diffusion sampling strategy. CADKnitter is able to generate a complementary CAD part that follows both the geometric constraints of the given CAD model and the semantic constraints of the desired design text prompt. We also curate a dataset, so-called KnitCAD, containing over 310,000 samples of CAD models, along with textual prompts and assembly metadata that provide semantic and geometric constraints. Intensive experiments demonstrate that our proposed method outperforms other state-of-the-art baselines by a clear margin.
>
---
#### [new 045] Exploring MLLM-Diffusion Information Transfer with MetaCanvas
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属多模态生成任务，旨在解决MLLM在生成中推理能力未充分利用的问题。提出MetaCanvas框架，使MLLM直接在潜空间进行空间与时空规划，增强对扩散模型的结构化控制，提升生成精确性。**

- **链接: [https://arxiv.org/pdf/2512.11464v1](https://arxiv.org/pdf/2512.11464v1)**

> **作者:** Han Lin; Xichen Pan; Ziqi Huang; Ji Hou; Jialiang Wang; Weifeng Chen; Zecheng He; Felix Juefei-Xu; Junzhe Sun; Zhipeng Fan; Ali Thabet; Mohit Bansal; Chu Wang
>
> **备注:** Project page: https://metacanvas.github.io
>
> **摘要:** Multimodal learning has rapidly advanced visual understanding, largely via multimodal large language models (MLLMs) that use powerful LLMs as cognitive cores. In visual generation, however, these powerful core models are typically reduced to global text encoders for diffusion models, leaving most of their reasoning and planning ability unused. This creates a gap: current multimodal LLMs can parse complex layouts, attributes, and knowledge-intensive scenes, yet struggle to generate images or videos with equally precise and structured control. We propose MetaCanvas, a lightweight framework that lets MLLMs reason and plan directly in spatial and spatiotemporal latent spaces and interface tightly with diffusion generators. We empirically implement MetaCanvas on three different diffusion backbones and evaluate it across six tasks, including text-to-image generation, text/image-to-video generation, image/video editing, and in-context video generation, each requiring precise layouts, robust attribute binding, and reasoning-intensive control. MetaCanvas consistently outperforms global-conditioning baselines, suggesting that treating MLLMs as latent-space planners is a promising direction for narrowing the gap between multimodal understanding and generation.
>
---
#### [new 046] WildCap: Facial Appearance Capture in the Wild via Hybrid Inverse Rendering
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于面部外观捕捉任务，旨在解决野外复杂光照下难以高质量重建面部反射属性的问题。提出WildCap方法，结合数据驱动预处理与基于物理的逆向渲染，引入纹理网格光照模型，有效分离光照与材质，提升野外视频中面部外观重建质量。**

- **链接: [https://arxiv.org/pdf/2512.11237v1](https://arxiv.org/pdf/2512.11237v1)**

> **作者:** Yuxuan Han; Xin Ming; Tianxiao Li; Zhuofan Shen; Qixuan Zhang; Lan Xu; Feng Xu
>
> **备注:** Technical report. project page: https://yxuhan.github.io/WildCap/index.html; code: https://github.com/yxuhan/WildCap
>
> **摘要:** Existing methods achieve high-quality facial appearance capture under controllable lighting, which increases capture cost and limits usability. We propose WildCap, a novel method for high-quality facial appearance capture from a smartphone video recorded in the wild. To disentangle high-quality reflectance from complex lighting effects in in-the-wild captures, we propose a novel hybrid inverse rendering framework. Specifically, we first apply a data-driven method, i.e., SwitchLight, to convert the captured images into more constrained conditions and then adopt model-based inverse rendering. However, unavoidable local artifacts in network predictions, such as shadow-baking, are non-physical and thus hinder accurate inverse rendering of lighting and material. To address this, we propose a novel texel grid lighting model to explain non-physical effects as clean albedo illuminated by local physical lighting. During optimization, we jointly sample a diffusion prior for reflectance maps and optimize the lighting, effectively resolving scale ambiguity between local lights and albedo. Our method achieves significantly better results than prior arts in the same capture setup, closing the quality gap between in-the-wild and controllable recordings by a large margin. Our code will be released \href{https://yxuhan.github.io/WildCap/index.html}{\textcolor{magenta}{here}}.
>
---
#### [new 047] Lightweight 3D Gaussian Splatting Compression via Video Codec
- **分类: cs.CV**

- **简介: 该论文研究3D高斯点阵的轻量级压缩，旨在解决现有方法计算开销大、耗时长的问题。提出基于视频编码的LGSCV方法，采用双阶段Morton扫描与MiniPLAS优化排序，结合SH降维，显著提升压缩效率与速率失真性能。**

- **链接: [https://arxiv.org/pdf/2512.11186v1](https://arxiv.org/pdf/2512.11186v1)**

> **作者:** Qi Yang; Geert Van Der Auwera; Zhu Li
>
> **备注:** Accepted by DCC2026 Oral
>
> **摘要:** Current video-based GS compression methods rely on using Parallel Linear Assignment Sorting (PLAS) to convert 3D GS into smooth 2D maps, which are computationally expensive and time-consuming, limiting the application of GS on lightweight devices. In this paper, we propose a Lightweight 3D Gaussian Splatting (GS) Compression method based on Video codec (LGSCV). First, a two-stage Morton scan is proposed to generate blockwise 2D maps that are friendly for canonical video codecs in which the coding units (CU) are square blocks. A 3D Morton scan is used to permute GS primitives, followed by a 2D Morton scan to map the ordered GS primitives to 2D maps in a blockwise style. However, although the blockwise 2D maps report close performance to the PLAS map in high-bitrate regions, they show a quality collapse at medium-to-low bitrates. Therefore, a principal component analysis (PCA) is used to reduce the dimensionality of spherical harmonics (SH), and a MiniPLAS, which is flexible and fast, is designed to permute the primitives within certain block sizes. Incorporating SH PCA and MiniPLAS leads to a significant gain in rate-distortion (RD) performance, especially at medium and low bitrates. MiniPLAS can also guide the setting of the codec CU size configuration and significantly reduce encoding time. Experimental results on the MPEG dataset demonstrate that the proposed LGSCV achieves over 20% RD gain compared with state-of-the-art methods, while reducing 2D map generation time to approximately 1 second and cutting encoding time by 50%. The code is available at https://github.com/Qi-Yangsjtu/LGSCV .
>
---
#### [new 048] Uncertainty-Aware Domain Adaptation for Vitiligo Segmentation in Clinical Photographs
- **分类: cs.CV**

- **简介: 该论文研究白癜风病灶分割任务，旨在提升临床照片中病损区域量化精度。提出一种融合域自适应预训练、频域感知网络结构与不确定性估计的框架，有效抑制背景干扰，精准分割边界，生成可信度图辅助医生判读。**

- **链接: [https://arxiv.org/pdf/2512.11791v1](https://arxiv.org/pdf/2512.11791v1)**

> **作者:** Wentao Jiang; Vamsi Varra; Caitlin Perez-Stable; Harrison Zhu; Meredith Apicella; Nicole Nyamongo
>
> **摘要:** Accurately quantifying vitiligo extent in routine clinical photographs is crucial for longitudinal monitoring of treatment response. We propose a trustworthy, frequency-aware segmentation framework built on three synergistic pillars: (1) a data-efficient training strategy combining domain-adaptive pre-training on the ISIC 2019 dataset with an ROI-constrained dual-task loss to suppress background noise; (2) an architectural refinement via a ConvNeXt V2-based encoder enhanced with a novel High-Frequency Spectral Gating (HFSG) module and stem-skip connections to capture subtle textures; and (3) a clinical trust mechanism employing K-fold ensemble and Test-Time Augmentation (TTA) to generate pixel-wise uncertainty maps. Extensive validation on an expert-annotated clinical cohort demonstrates superior performance, achieving a Dice score of 85.05% and significantly reducing boundary error (95% Hausdorff Distance improved from 44.79 px to 29.95 px), consistently outperforming strong CNN (ResNet-50 and UNet++) and Transformer (MiT-B5) baselines. Notably, our framework demonstrates high reliability with zero catastrophic failures and provides interpretable entropy maps to identify ambiguous regions for clinician review. Our approach suggests that the proposed framework establishes a robust and reliable standard for automated vitiligo assessment.
>
---
#### [new 049] Using GUI Agent for Electronic Design Automation
- **分类: cs.CV; cs.AR**

- **简介: 该论文研究GUI代理在电子设计自动化（EDA）中的应用，旨在解决现有GUI代理在专业CAD软件中性能不足的问题。工作包括构建GUI-EDA数据集、评估主流GUI代理，并提出专用指标EDAgent，首次在工业软件中超越电气工程博士生表现。**

- **链接: [https://arxiv.org/pdf/2512.11611v1](https://arxiv.org/pdf/2512.11611v1)**

> **作者:** Chunyi Li; Longfei Li; Zicheng Zhang; Xiaohong Liu; Min Tang; Weisi Lin; Guangtao Zhai
>
> **备注:** 17 pages, 15 figures, 8 tables
>
> **摘要:** Graphical User Interface (GUI) agents adopt an end-to-end paradigm that maps a screenshot to an action sequence, thereby automating repetitive tasks in virtual environments. However, existing GUI agents are evaluated almost exclusively on commodity software such as Microsoft Word and Excel. Professional Computer-Aided Design (CAD) suites promise an order-of-magnitude higher economic return, yet remain the weakest performance domain for existing agents and are still far from replacing expert Electronic-Design-Automation (EDA) engineers. We therefore present the first systematic study that deploys GUI agents for EDA workflows. Our contributions are: (1) a large-scale dataset named GUI-EDA, including 5 CAD tools and 5 physical domains, comprising 2,000+ high-quality screenshot-answer-action pairs recorded by EDA scientists and engineers during real-world component design; (2) a comprehensive benchmark that evaluates 30+ mainstream GUI agents, demonstrating that EDA tasks constitute a major, unsolved challenge; and (3) an EDA-specialized metric named EDAgent, equipped with a reflection mechanism that achieves reliable performance on industrial CAD software and, for the first time, outperforms Ph.D. students majored in Electrical Engineering. This work extends GUI agents from generic office automation to specialized, high-value engineering domains and offers a new avenue for advancing EDA productivity. The dataset will be released at: https://github.com/aiben-ch/GUI-EDA.
>
---
#### [new 050] Moment-Based 3D Gaussian Splatting: Resolving Volumetric Occlusion with Order-Independent Transmittance
- **分类: cs.CV; cs.GR**

- **简介: 该论文属新型视图合成任务，旨在解决3D高斯点阵渲染中因顺序依赖混合导致的透明重叠物体渲染失真问题。提出基于矩的光透射计算方法，用统计矩建模光线上的密度分布，实现无需排序的高质量半透明渲染。**

- **链接: [https://arxiv.org/pdf/2512.11800v1](https://arxiv.org/pdf/2512.11800v1)**

> **作者:** Jan U. Müller; Robin Tim Landsgesell; Leif Van Holland; Patrick Stotko; Reinhard Klein
>
> **摘要:** The recent success of 3D Gaussian Splatting (3DGS) has reshaped novel view synthesis by enabling fast optimization and real-time rendering of high-quality radiance fields. However, it relies on simplified, order-dependent alpha blending and coarse approximations of the density integral within the rasterizer, thereby limiting its ability to render complex, overlapping semi-transparent objects. In this paper, we extend rasterization-based rendering of 3D Gaussian representations with a novel method for high-fidelity transmittance computation, entirely avoiding the need for ray tracing or per-pixel sample sorting. Building on prior work in moment-based order-independent transparency, our key idea is to characterize the density distribution along each camera ray with a compact and continuous representation based on statistical moments. To this end, we analytically derive and compute a set of per-pixel moments from all contributing 3D Gaussians. From these moments, a continuous transmittance function is reconstructed for each ray, which is then independently sampled within each Gaussian. As a result, our method bridges the gap between rasterization and physical accuracy by modeling light attenuation in complex translucent media, significantly improving overall reconstruction and rendering quality.
>
---
#### [new 051] KeyframeFace: From Text to Expressive Facial Keyframes
- **分类: cs.CV**

- **简介: 该论文研究文本到动态3D面部动画生成，旨在解决现有方法缺乏语义对齐与时序结构的问题。作者构建了KeyframeFace数据集，提供带关键帧标注的多模态数据，并提出基于大语言模型的框架，实现可解释、高保真的表情动画合成。**

- **链接: [https://arxiv.org/pdf/2512.11321v1](https://arxiv.org/pdf/2512.11321v1)**

> **作者:** Jingchao Wu; Zejian Kang; Haibo Liu; Yuanchen Fei; Xiangru Huang
>
> **摘要:** Generating dynamic 3D facial animation from natural language requires understanding both temporally structured semantics and fine-grained expression changes. Existing datasets and methods mainly focus on speech-driven animation or unstructured expression sequences and therefore lack the semantic grounding and temporal structures needed for expressive human performance generation. In this work, we introduce KeyframeFace, a large-scale multimodal dataset designed for text-to-animation research through keyframe-level supervision. KeyframeFace provides 2,100 expressive scripts paired with monocular videos, per-frame ARKit coefficients, contextual backgrounds, complex emotions, manually defined keyframes, and multi-perspective annotations based on ARKit coefficients and images via Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Beyond the dataset, we propose the first text-to-animation framework that explicitly leverages LLM priors for interpretable facial motion synthesis. This design aligns the semantic understanding capabilities of LLMs with the interpretable structure of ARKit's coefficients, enabling high-fidelity expressive animation. KeyframeFace and our LLM-based framework together establish a new foundation for interpretable, keyframe-guided, and context-aware text-to-animation. Code and data are available at https://github.com/wjc12345123/KeyframeFace.
>
---
#### [new 052] FreqDINO: Frequency-Guided Adaptation for Generalized Boundary-Aware Ultrasound Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对超声图像分割中边界退化问题，提出FreqDINO框架。基于DINOv3引入多尺度频域特征对齐、频域引导边界细化和多任务解码器，增强边界感知与结构一致性，提升分割精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.11335v1](https://arxiv.org/pdf/2512.11335v1)**

> **作者:** Yixuan Zhang; Qing Xu; Yue Li; Xiangjian He; Qian Zhang; Mainul Haque; Rong Qu; Wenting Duan; Zhen Chen
>
> **摘要:** Ultrasound image segmentation is pivotal for clinical diagnosis, yet challenged by speckle noise and imaging artifacts. Recently, DINOv3 has shown remarkable promise in medical image segmentation with its powerful representation capabilities. However, DINOv3, pre-trained on natural images, lacks sensitivity to ultrasound-specific boundary degradation. To address this limitation, we propose FreqDINO, a frequency-guided segmentation framework that enhances boundary perception and structural consistency. Specifically, we devise a Multi-scale Frequency Extraction and Alignment (MFEA) strategy to separate low-frequency structures and multi-scale high-frequency boundary details, and align them via learnable attention. We also introduce a Frequency-Guided Boundary Refinement (FGBR) module that extracts boundary prototypes from high-frequency components and refines spatial features. Furthermore, we design a Multi-task Boundary-Guided Decoder (MBGD) to ensure spatial coherence between boundary and semantic predictions. Extensive experiments demonstrate that FreqDINO surpasses state-of-the-art methods with superior achieves remarkable generalization capability. The code is at https://github.com/MingLang-FD/FreqDINO.
>
---
#### [new 053] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决多镜头视频中角色与场景跨镜头不一致、长度受限的问题。提出FilmWeaver框架，通过缓存引导的自回归扩散模型，结合双级缓存机制实现长时序、高一致性多镜头视频生成。**

- **链接: [https://arxiv.org/pdf/2512.11274v1](https://arxiv.org/pdf/2512.11274v1)**

> **作者:** Xiangyang Luo; Qingyu Li; Xiaokun Liu; Wenyu Qin; Miao Yang; Meng Wang; Pengfei Wan; Di Zhang; Kun Gai; Shao-Lun Huang
>
> **备注:** AAAI-2026
>
> **摘要:** Current video generation models perform well at single-shot synthesis but struggle with multi-shot videos, facing critical challenges in maintaining character and background consistency across shots and flexibly generating videos of arbitrary length and shot count. To address these limitations, we introduce \textbf{FilmWeaver}, a novel framework designed to generate consistent, multi-shot videos of arbitrary length. First, it employs an autoregressive diffusion paradigm to achieve arbitrary-length video generation. To address the challenge of consistency, our key insight is to decouple the problem into inter-shot consistency and intra-shot coherence. We achieve this through a dual-level cache mechanism: a shot memory caches keyframes from preceding shots to maintain character and scene identity, while a temporal memory retains a history of frames from the current shot to ensure smooth, continuous motion. The proposed framework allows for flexible, multi-round user interaction to create multi-shot videos. Furthermore, due to this decoupled design, our method demonstrates high versatility by supporting downstream tasks such as multi-concept injection and video extension. To facilitate the training of our consistency-aware method, we also developed a comprehensive pipeline to construct a high-quality multi-shot video dataset. Extensive experimental results demonstrate that our method surpasses existing approaches on metrics for both consistency and aesthetic quality, opening up new possibilities for creating more consistent, controllable, and narrative-driven video content. Project Page: https://filmweaver.github.io
>
---
#### [new 054] The N-Body Problem: Parallel Execution from Single-Person Egocentric Video
- **分类: cs.CV**

- **简介: 该论文提出“N-Body问题”：从单人第一视角视频中，推理多人并行执行相同任务的合理分工。旨在提升效率同时避免空间、物体和时序冲突。通过结构化提示引导视觉语言模型建模3D环境与因果关系，在多指标上显著优于基线。**

- **链接: [https://arxiv.org/pdf/2512.11393v1](https://arxiv.org/pdf/2512.11393v1)**

> **作者:** Zhifan Zhu; Yifei Huang; Yoichi Sato; Dima Damen
>
> **备注:** project webpage: https://zhifanzhu.github.io/ego-nbody
>
> **摘要:** Humans can intuitively parallelise complex activities, but can a model learn this from observing a single person? Given one egocentric video, we introduce the N-Body Problem: how N individuals, can hypothetically perform the same set of tasks observed in this video. The goal is to maximise speed-up, but naive assignment of video segments to individuals often violates real-world constraints, leading to physically impossible scenarios like two people using the same object or occupying the same space. To address this, we formalise the N-Body Problem and propose a suite of metrics to evaluate both performance (speed-up, task coverage) and feasibility (spatial collisions, object conflicts and causal constraints). We then introduce a structured prompting strategy that guides a Vision-Language Model (VLM) to reason about the 3D environment, object usage, and temporal dependencies to produce a viable parallel execution. On 100 videos from EPIC-Kitchens and HD-EPIC, our method for N = 2 boosts action coverage by 45% over a baseline prompt for Gemini 2.5 Pro, while simultaneously slashing collision rates, object and causal conflicts by 55%, 45% and 55% respectively.
>
---
#### [new 055] Information-driven Fusion of Pathology Foundation Models for Enhanced Disease Characterization
- **分类: cs.CV**

- **简介: 该论文聚焦病理学基础模型融合任务，旨在提升癌症分级与分期性能。针对模型间冗余与互补性不明确的问题，提出一种基于相关性的智能融合方法，通过剪枝冗余特征整合多模型信息，增强表征能力与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.11104v1](https://arxiv.org/pdf/2512.11104v1)**

> **作者:** Brennan Flannery; Thomas DeSilvio; Jane Nguyen; Satish E. Viswanath
>
> **备注:** 29 Pages, 10 figures
>
> **摘要:** Foundation models (FMs) have demonstrated strong performance across diverse pathology tasks. While there are similarities in the pre-training objectives of FMs, there is still limited understanding of their complementarity, redundancy in embedding spaces, or biological interpretation of features. In this study, we propose an information-driven, intelligent fusion strategy for integrating multiple pathology FMs into a unified representation and systematically evaluate its performance for cancer grading and staging across three distinct diseases. Diagnostic H&E whole-slide images from kidney (519 slides), prostate (490 slides), and rectal (200 slides) cancers were dichotomized into low versus high grade or stage. Both tile-level FMs (Conch v1.5, MUSK, Virchow2, H-Optimus1, Prov-Gigapath) and slide-level FMs (TITAN, CHIEF, MADELEINE) were considered to train downstream classifiers. We then evaluated three FM fusion schemes at both tile and slide levels: majority-vote ensembling, naive feature concatenation, and intelligent fusion based on correlation-guided pruning of redundant features. Under patient-stratified cross-validation with hold-out testing, intelligent fusion of tile-level embeddings yielded consistent gains in classification performance across all three cancers compared with the best single FMs and naive fusion. Global similarity metrics revealed substantial alignment of FM embedding spaces, contrasted by lower local neighborhood agreement, indicating complementary fine-grained information across FMs. Attention maps showed that intelligent fusion yielded concentrated attention on tumor regions while reducing spurious focus on benign regions. Our findings suggest that intelligent, correlation-guided fusion of pathology FMs can yield compact, task-tailored representations that enhance both predictive performance and interpretability in downstream computational pathology tasks.
>
---
#### [new 056] 3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation
- **分类: cs.CV**

- **简介: 该论文针对3D牙齿分割任务，提出3DTeethSAM方法，通过渲染多视角图像利用SAM2进行2D分割，并设计三个轻量模块优化提示生成、掩码细化与分类，结合DGAP提升精度与训练速度，在3DTeethSeg上达到91.90% IoU，实现新SOTA。**

- **链接: [https://arxiv.org/pdf/2512.11557v1](https://arxiv.org/pdf/2512.11557v1)**

> **作者:** Zhiguo Lu; Jianwen Lou; Mingjun Ma; Hairong Jin; Youyi Zheng; Kun Zhou
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** 3D teeth segmentation, involving the localization of tooth instances and their semantic categorization in 3D dental models, is a critical yet challenging task in digital dentistry due to the complexity of real-world dentition. In this paper, we propose 3DTeethSAM, an adaptation of the Segment Anything Model 2 (SAM2) for 3D teeth segmentation. SAM2 is a pretrained foundation model for image and video segmentation, demonstrating a strong backbone in various downstream scenarios. To adapt SAM2 for 3D teeth data, we render images of 3D teeth models from predefined views, apply SAM2 for 2D segmentation, and reconstruct 3D results using 2D-3D projections. Since SAM2's performance depends on input prompts and its initial outputs often have deficiencies, and given its class-agnostic nature, we introduce three light-weight learnable modules: (1) a prompt embedding generator to derive prompt embeddings from image embeddings for accurate mask decoding, (2) a mask refiner to enhance SAM2's initial segmentation results, and (3) a mask classifier to categorize the generated masks. Additionally, we incorporate Deformable Global Attention Plugins (DGAP) into SAM2's image encoder. The DGAP enhances both the segmentation accuracy and the speed of the training process. Our method has been validated on the 3DTeethSeg benchmark, achieving an IoU of 91.90% on high-resolution 3D teeth meshes, establishing a new state-of-the-art in the field.
>
---
#### [new 057] Collaborative Reconstruction and Repair for Multi-class Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文研究多类工业异常检测，旨在解决重建模型易出现的恒等映射问题。提出协作重建与修复（CRR）框架，通过修复合成异常、特征随机掩码和分割网络优化，提升异常定位与检测性能。**

- **链接: [https://arxiv.org/pdf/2512.11401v1](https://arxiv.org/pdf/2512.11401v1)**

> **作者:** Qishan Wang; Haofeng Wang; Shuyong Gao; Jia Guo; Li Xiong; Jiaqi Li; Dengxuan Bai; Wenqiang Zhang
>
> **备注:** Accepted to Data Intelligence 2025
>
> **摘要:** Industrial anomaly detection is a challenging open-set task that aims to identify unknown anomalous patterns deviating from normal data distribution. To avoid the significant memory consumption and limited generalizability brought by building separate models per class, we focus on developing a unified framework for multi-class anomaly detection. However, under this challenging setting, conventional reconstruction-based networks often suffer from an identity mapping problem, where they directly replicate input features regardless of whether they are normal or anomalous, resulting in detection failures. To address this issue, this study proposes a novel framework termed Collaborative Reconstruction and Repair (CRR), which transforms the reconstruction to repairation. First, we optimize the decoder to reconstruct normal samples while repairing synthesized anomalies. Consequently, it generates distinct representations for anomalous regions and similar representations for normal areas compared to the encoder's output. Second, we implement feature-level random masking to ensure that the representations from decoder contain sufficient local information. Finally, to minimize detection errors arising from the discrepancies between feature representations from the encoder and decoder, we train a segmentation network supervised by synthetic anomaly masks, thereby enhancing localization performance. Extensive experiments on industrial datasets that CRR effectively mitigates the identity mapping issue and achieves state-of-the-art performance in multi-class industrial anomaly detection.
>
---
#### [new 058] JoyAvatar: Real-time and Infinite Audio-Driven Avatar Generation with Autoregressive Diffusion
- **分类: cs.CV**

- **简介: 该论文研究音频驱动的虚拟形象生成，旨在解决现有方法在长视频生成中计算量大、错误累积和质量下降的问题。提出JoyAvatar模型，通过渐进去噪、运动条件注入和无限位置编码，实现高质量、实时、无限长度的视频生成。**

- **链接: [https://arxiv.org/pdf/2512.11423v1](https://arxiv.org/pdf/2512.11423v1)**

> **作者:** Chaochao Li; Ruikui Wang; Liangbo Zhou; Jinheng Feng; Huaishao Luo; Huan Zhang; Youzheng Wu; Xiaodong He
>
> **摘要:** Existing DiT-based audio-driven avatar generation methods have achieved considerable progress, yet their broader application is constrained by limitations such as high computational overhead and the inability to synthesize long-duration videos. Autoregressive methods address this problem by applying block-wise autoregressive diffusion methods. However, these methods suffer from the problem of error accumulation and quality degradation. To address this, we propose JoyAvatar, an audio-driven autoregressive model capable of real-time inference and infinite-length video generation with the following contributions: (1) Progressive Step Bootstrapping (PSB), which allocates more denoising steps to initial frames to stabilize generation and reduce error accumulation; (2) Motion Condition Injection (MCI), enhancing temporal coherence by injecting noise-corrupted previous frames as motion condition; and (3) Unbounded RoPE via Cache-Resetting (URCR), enabling infinite-length generation through dynamic positional encoding. Our 1.3B-parameter causal model achieves 16 FPS on a single GPU and achieves competitive results in visual quality, temporal consistency, and lip synchronization.
>
---
#### [new 059] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文研究无监督工业异常检测，旨在解决传统自编码器对不同规模缺陷抑制不彻底及细节丢失的问题。提出递归重建框架RcAE，通过迭代重构逐步抑制异常并保留细节，结合跨递归检测模块提升检测精度，显著提高效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.11284v1](https://arxiv.org/pdf/2512.11284v1)**

> **作者:** Rongcheng Wu; Hao Zhu; Shiying Zhang; Mingzhe Wang; Zhidong Li; Hui Li; Jianlong Zhou; Jiangtao Cui; Fang Chen; Pingyang Sun; Qiyu Liao; Ye Lin
>
> **备注:** 19 pages, 7 figures, to be published in AAAI-26
>
> **摘要:** Unsupervised industrial anomaly detection requires accurately identifying defects without labeled data. Traditional autoencoder-based methods often struggle with incomplete anomaly suppression and loss of fine details, as their single-pass decoding fails to effectively handle anomalies with varying severity and scale. We propose a recursive architecture for autoencoder (RcAE), which performs reconstruction iteratively to progressively suppress anomalies while refining normal structures. Unlike traditional single-pass models, this recursive design naturally produces a sequence of reconstructions, progressively exposing suppressed abnormal patterns. To leverage this reconstruction dynamics, we introduce a Cross Recursion Detection (CRD) module that tracks inconsistencies across recursion steps, enhancing detection of both subtle and large-scale anomalies. Additionally, we incorporate a Detail Preservation Network (DPN) to recover high-frequency textures typically lost during reconstruction. Extensive experiments demonstrate that our method significantly outperforms existing non-diffusion methods, and achieves performance on par with recent diffusion models with only 10% of their parameters and offering substantially faster inference. These results highlight the practicality and efficiency of our approach for real-world applications.
>
---
#### [new 060] Synthetic Vasculature and Pathology Enhance Vision-Language Model Reasoning
- **分类: cs.CV**

- **简介: 该论文属医学视觉语言模型任务，旨在解决专业医疗图像（如OCTA）缺乏精细图文标注的问题。作者提出SVR框架，合成带糖尿病视网膜病变特征的视网膜血管图像及细粒度推理文本，构建OCTA-100K-SVR数据集，提升模型在真实图像上的分类与解释能力。**

- **链接: [https://arxiv.org/pdf/2512.11060v1](https://arxiv.org/pdf/2512.11060v1)**

> **作者:** Chenjun Li; Cheng Wan; Laurin Lux; Alexander Berger; Richard B. Rosen; Martin J. Menten; Johannes C. Paetzold
>
> **备注:** 23 pages, 8 figures, 6 tables. Full paper under review for MIDL 2026 (Medical Imaging with Deep Learning)
>
> **摘要:** Vision-Language Models (VLMs) offer a promising path toward interpretable medical diagnosis by allowing users to ask about clinical explanations alongside predictions and across different modalities. However, training VLMs for detailed reasoning requires large-scale image-text datasets. In many specialized domains, for example in reading Optical Coherence Tomography Angiography (OCTA) images, such precise text with grounded description of pathologies is scarce or even non-existent. To overcome this bottleneck, we introduce Synthetic Vasculature Reasoning (SVR), a framework that controllably synthesizes images and corresponding text, specifically: realistic retinal vasculature with Diabetic Retinopathy (DR) features: capillary dropout, microaneurysms, neovascularization, and tortuosity, while automatically generating granular reasoning texts. Based on this we curate OCTA-100K-SVR, an OCTA image-reasoning dataset with 100,000 pairs. Our experiments show that a general-purpose VLM (Qwen3-VL-8b) trained on the dataset achieves a zero-shot balanced classification accuracy of 89.67% on real OCTA images, outperforming supervised baselines. Through human expert evaluation we also demonstrate that it significantly enhances explanation quality and pathology localization on clinical data.
>
---
#### [new 061] SSL-MedSAM2: A Semi-supervised Medical Image Segmentation Framework Powered by Few-shot Learning of SAM2
- **分类: cs.CV**

- **简介: 该论文研究医学图像分割任务，旨在解决标注成本高的问题。提出SSL-MedSAM2框架，结合SAM2的少样本伪标签生成与nnUNet的迭代优化，在少量标注下实现高性能肝脏分割。**

- **链接: [https://arxiv.org/pdf/2512.11548v1](https://arxiv.org/pdf/2512.11548v1)**

> **作者:** Zhendi Gong; Xin Chen
>
> **备注:** Accepted by MICCAI 2025 CARE Challenge, waiting for publication
>
> **摘要:** Despite the success of deep learning based models in medical image segmentation, most state-of-the-art (SOTA) methods perform fully-supervised learning, which commonly rely on large scale annotated training datasets. However, medical image annotation is highly time-consuming, hindering its clinical applications. Semi-supervised learning (SSL) has been emerged as an appealing strategy in training with limited annotations, largely reducing the labelling cost. We propose a novel SSL framework SSL-MedSAM2, which contains a training-free few-shot learning branch TFFS-MedSAM2 based on the pretrained large foundation model Segment Anything Model 2 (SAM2) for pseudo label generation, and an iterative fully-supervised learning branch FSL-nnUNet based on nnUNet for pseudo label refinement. The results on MICCAI2025 challenge CARE-LiSeg (Liver Segmentation) demonstrate an outstanding performance of SSL-MedSAM2 among other methods. The average dice scores on the test set in GED4 and T1 MRI are 0.9710 and 0.9648 respectively, and the Hausdorff distances are 20.07 and 21.97 respectively. The code is available via https://github.com/naisops/SSL-MedSAM2/tree/main.
>
---
#### [new 062] Embodied Image Compression
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出“具身图像压缩”任务，旨在解决具身智能体在低码率下实时通信与任务执行的问题。作者构建了EmbodiedComp基准，验证现有模型在超低码率下的性能瓶颈，推动面向具身智能的专用压缩技术发展。**

- **链接: [https://arxiv.org/pdf/2512.11612v1](https://arxiv.org/pdf/2512.11612v1)**

> **作者:** Chunyi Li; Rui Qing; Jianbo Zhang; Yuan Tian; Xiangyang Zhu; Zicheng Zhang; Xiaohong Liu; Weisi Lin; Guangtao Zhai
>
> **备注:** 15 pages, 12 figures, 3 tables
>
> **摘要:** Image Compression for Machines (ICM) has emerged as a pivotal research direction in the field of visual data compression. However, with the rapid evolution of machine intelligence, the target of compression has shifted from task-specific virtual models to Embodied agents operating in real-world environments. To address the communication constraints of Embodied AI in multi-agent systems and ensure real-time task execution, this paper introduces, for the first time, the scientific problem of Embodied Image Compression. We establish a standardized benchmark, EmbodiedComp, to facilitate systematic evaluation under ultra-low bitrate conditions in a closed-loop setting. Through extensive empirical studies in both simulated and real-world settings, we demonstrate that existing Vision-Language-Action models (VLAs) fail to reliably perform even simple manipulation tasks when compressed below the Embodied bitrate threshold. We anticipate that EmbodiedComp will catalyze the development of domain-specific compression tailored for Embodied agents , thereby accelerating the Embodied AI deployment in the Real-world.
>
---
#### [new 063] CADMorph: Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop
- **分类: cs.CV**

- **简介: 该论文研究几何驱动的参数化CAD编辑任务，旨在解决编辑过程中结构保持、语义有效与形状保真难题。作者提出CADMorph框架，利用预训练模型通过“规划-生成-验证”循环实现高质量编辑，无需三元组数据，提升编辑效果与应用能力。**

- **链接: [https://arxiv.org/pdf/2512.11480v1](https://arxiv.org/pdf/2512.11480v1)**

> **作者:** Weijian Ma; Shizhao Sun; Ruiyu Wang; Jiang Bian
>
> **备注:** NeurIPS 2025
>
> **摘要:** A Computer-Aided Design (CAD) model encodes an object in two coupled forms: a parametric construction sequence and its resulting visible geometric shape. During iterative design, adjustments to the geometric shape inevitably require synchronized edits to the underlying parametric sequence, called geometry-driven parametric CAD editing. The task calls for 1) preserving the original sequence's structure, 2) ensuring each edit's semantic validity, and 3) maintaining high shape fidelity to the target shape, all under scarce editing data triplets. We present CADMorph, an iterative plan-generate-verify framework that orchestrates pretrained domain-specific foundation models during inference: a parameter-to-shape (P2S) latent diffusion model and a masked-parameter-prediction (MPP) model. In the planning stage, cross-attention maps from the P2S model pinpoint the segments that need modification and offer editing masks. The MPP model then infills these masks with semantically valid edits in the generation stage. During verification, the P2S model embeds each candidate sequence in shape-latent space, measures its distance to the target shape, and selects the closest one. The three stages leverage the inherent geometric consciousness and design knowledge in pretrained priors, and thus tackle structure preservation, semantic validity, and shape fidelity respectively. Besides, both P2S and MPP models are trained without triplet data, bypassing the data-scarcity bottleneck. CADMorph surpasses GPT-4o and specialized CAD baselines, and supports downstream applications such as iterative editing and reverse-engineering enhancement.
>
---
#### [new 064] Multi-task Learning with Extended Temporal Shift Module for Temporal Action Localization
- **分类: cs.CV**

- **简介: 该论文针对多视角多模态视频中的时序动作定位（TAL）任务，提出扩展时序位移模块（TSM），引入背景类并结合多任务学习联合优化场景分类与TAL，通过加权集成提升性能，在ICCV 2025 BinEgo-360挑战赛中取得第一。**

- **链接: [https://arxiv.org/pdf/2512.11189v1](https://arxiv.org/pdf/2512.11189v1)**

> **作者:** Anh-Kiet Duong; Petra Gomez-Krämer
>
> **备注:** BinEgo360@ICCV25
>
> **摘要:** We present our solution to the BinEgo-360 Challenge at ICCV 2025, which focuses on temporal action localization (TAL) in multi-perspective and multi-modal video settings. The challenge provides a dataset containing panoramic, third-person, and egocentric recordings, annotated with fine-grained action classes. Our approach is built on the Temporal Shift Module (TSM), which we extend to handle TAL by introducing a background class and classifying fixed-length non-overlapping intervals. We employ a multi-task learning framework that jointly optimizes for scene classification and TAL, leveraging contextual cues between actions and environments. Finally, we integrate multiple models through a weighted ensemble strategy, which improves robustness and consistency of predictions. Our method is ranked first in both the initial and extended rounds of the competition, demonstrating the effectiveness of combining multi-task learning, an efficient backbone, and ensemble learning for TAL.
>
---
#### [new 065] On Geometric Understanding and Learned Data Priors in VGGT
- **分类: cs.CV**

- **简介: 该论文研究VGGT模型在3D场景理解中是否隐含几何理解，还是依赖外观先验。通过分析特征、注意力与干预实验，发现其虽无显式几何训练，但仍隐式学习了对应匹配与极线几何，并探讨了其对数据先验的依赖及鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.11508v1](https://arxiv.org/pdf/2512.11508v1)**

> **作者:** Jelena Bratulić; Sudhanshu Mittal; Thomas Brox; Christian Rupprecht
>
> **摘要:** The Visual Geometry Grounded Transformer (VGGT) is a 3D foundation model that infers camera geometry and scene structure in a single feed-forward pass. Trained in a supervised, single-step fashion on large datasets, VGGT raises a key question: does it build upon geometric concepts like traditional multi-view methods, or does it rely primarily on learned appearance-based data-driven priors? In this work, we conduct a systematic analysis of VGGT's internal mechanisms to uncover whether geometric understanding emerges within its representations. By probing intermediate features, analyzing attention patterns, and performing interventions, we examine how the model implements its functionality. Our findings reveal that VGGT implicitly performs correspondence matching within its global attention layers and encodes epipolar geometry, despite being trained without explicit geometric constraints. We further investigate VGGT's dependence on its learned data priors. Using spatial input masking and perturbation experiments, we assess its robustness to occlusions, appearance variations, and camera configurations, comparing it with classical multi-stage pipelines. Together, these insights highlight how VGGT internalizes geometric structure while using learned data-driven priors.
>
---
#### [new 066] VDAWorld: World Modelling via VLM-Directed Abstraction and Simulation
- **分类: cs.CV**

- **简介: 该论文属于世界建模任务，旨在解决生成式视频模型违背物理逻辑、缺乏交互性的问题。提出VDAWorld框架，利用视觉语言模型选择视觉工具构建场景，并匹配物理模拟器进行动态预测，实现可解释、可查询的高质量世界模拟。**

- **链接: [https://arxiv.org/pdf/2512.11061v1](https://arxiv.org/pdf/2512.11061v1)**

> **作者:** Felix O'Mahony; Roberto Cipolla; Ayush Tewari
>
> **备注:** Website: https://felixomahony.github.io/vdaworld/
>
> **摘要:** Generative video models, a leading approach to world modeling, face fundamental limitations. They often violate physical and logical rules, lack interactivity, and operate as opaque black boxes ill-suited for building structured, queryable worlds. To overcome these challenges, we propose a new paradigm focused on distilling an image caption pair into a tractable, abstract representation optimized for simulation. We introduce VDAWorld, a framework where a Vision-Language Model (VLM) acts as an intelligent agent to orchestrate this process. The VLM autonomously constructs a grounded (2D or 3D) scene representation by selecting from a suite of vision tools, and accordingly chooses a compatible physics simulator (e.g., rigid body, fluid) to act upon it. VDAWorld can then infer latent dynamics from the static scene to predict plausible future states. Our experiments show that this combination of intelligent abstraction and adaptive simulation results in a versatile world model capable of producing high quality simulations across a wide range of dynamic scenarios.
>
---
#### [new 067] Assisted Refinement Network Based on Channel Information Interaction for Camouflaged and Salient Object Detection
- **分类: cs.CV**

- **简介: 该论文研究伪装与显著物体检测任务，旨在解决解码阶段跨通道信息交互不足及边界与区域信息协同建模困难问题。提出通道信息交互模块和先验引导的协同解码结构，提升特征表达与边界定位精度，并验证了模型在多任务上的适用性。**

- **链接: [https://arxiv.org/pdf/2512.11369v1](https://arxiv.org/pdf/2512.11369v1)**

> **作者:** Kuan Wang; Yanjun Qin; Mengge Lu; Liejun Wang; Xiaoming Tao
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Camouflaged Object Detection (COD) stands as a significant challenge in computer vision, dedicated to identifying and segmenting objects visually highly integrated with their backgrounds. Current mainstream methods have made progress in cross-layer feature fusion, but two critical issues persist during the decoding stage. The first is insufficient cross-channel information interaction within the same-layer features, limiting feature expressiveness. The second is the inability to effectively co-model boundary and region information, making it difficult to accurately reconstruct complete regions and sharp boundaries of objects. To address the first issue, we propose the Channel Information Interaction Module (CIIM), which introduces a horizontal-vertical integration mechanism in the channel dimension. This module performs feature reorganization and interaction across channels to effectively capture complementary cross-channel information. To address the second issue, we construct a collaborative decoding architecture guided by prior knowledge. This architecture generates boundary priors and object localization maps through Boundary Extraction (BE) and Region Extraction (RE) modules, then employs hybrid attention to collaboratively calibrate decoded features, effectively overcoming semantic ambiguity and imprecise boundaries. Additionally, the Multi-scale Enhancement (MSE) module enriches contextual feature representations. Extensive experiments on four COD benchmark datasets validate the effectiveness and state-of-the-art performance of the proposed model. We further transferred our model to the Salient Object Detection (SOD) task and demonstrated its adaptability across downstream tasks, including polyp segmentation, transparent object detection, and industrial and road defect detection. Code and experimental results are publicly available at: https://github.com/akuan1234/ARNet-v2.
>
---
#### [new 068] FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing
- **分类: cs.CV**

- **简介: 该论文研究文本引导的复杂图像编辑任务，旨在解决多目标编辑中语义对齐与源图像一致性难以平衡的问题。提出FlowDC方法，通过解耦编辑效果并行处理，并分解衰减正交速度分量以提升编辑质量和源结构保持。**

- **链接: [https://arxiv.org/pdf/2512.11395v1](https://arxiv.org/pdf/2512.11395v1)**

> **作者:** Yilei Jiang; Zhen Wang; Yanghao Wang; Jun Yu; Yueting Zhuang; Jun Xiao; Long Chen
>
> **摘要:** With the surge of pre-trained text-to-image flow matching models, text-based image editing performance has gained remarkable improvement, especially for \underline{simple editing} that only contains a single editing target. To satisfy the exploding editing requirements, the \underline{complex editing} which contains multiple editing targets has posed as a more challenging task. However, current complex editing solutions: single-round and multi-round editing are limited by long text following and cumulative inconsistency, respectively. Thus, they struggle to strike a balance between semantic alignment and source consistency. In this paper, we propose \textbf{FlowDC}, which decouples the complex editing into multiple sub-editing effects and superposes them in parallel during the editing process. Meanwhile, we observed that the velocity quantity that is orthogonal to the editing displacement harms the source structure preserving. Thus, we decompose the velocity and decay the orthogonal part for better source consistency. To evaluate the effectiveness of complex editing settings, we construct a complex editing benchmark: Complex-PIE-Bench. On two benchmarks, FlowDC shows superior results compared with existing methods. We also detail the ablations of our module designs.
>
---
#### [new 069] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文研究不完整多模态情感识别，旨在解决模态性能差异与缺失数据下的模态欠优化问题。提出跨模态提示方法，通过生成动态提示、传播一致语义信息及动态重加权输出，提升各模态表征能力与整体识别精度。**

- **链接: [https://arxiv.org/pdf/2512.11239v1](https://arxiv.org/pdf/2512.11239v1)**

> **作者:** Wen-Jue He; Xiaofeng Zhu; Zheng Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Incomplete multi-modal emotion recognition (IMER) aims at understanding human intentions and sentiments by comprehensively exploring the partially observed multi-source data. Although the multi-modal data is expected to provide more abundant information, the performance gap and modality under-optimization problem hinder effective multi-modal learning in practice, and are exacerbated in the confrontation of the missing data. To address this issue, we devise a novel Cross-modal Prompting (ComP) method, which emphasizes coherent information by enhancing modality-specific features and improves the overall recognition accuracy by boosting each modality's performance. Specifically, a progressive prompt generation module with a dynamic gradient modulator is proposed to produce concise and consistent modality semantic cues. Meanwhile, cross-modal knowledge propagation selectively amplifies the consistent information in modality features with the delivered prompts to enhance the discrimination of the modality-specific output. Additionally, a coordinator is designed to dynamically re-weight the modality outputs as a complement to the balance strategy to improve the model's efficacy. Extensive experiments on 4 datasets with 7 SOTA methods under different missing rates validate the effectiveness of our proposed method.
>
---
#### [new 070] Learning complete and explainable visual representations from itemized text supervision
- **分类: cs.CV**

- **简介: 该论文针对非物体中心视觉领域中多条独立文本标注的问题，提出ItemizedCLIP框架，通过交叉注意力和定制目标学习完整、可解释的视觉表示，实现更好的零样本性能与细粒度可解释性。**

- **链接: [https://arxiv.org/pdf/2512.11141v1](https://arxiv.org/pdf/2512.11141v1)**

> **作者:** Yiwei Lyu; Chenhui Zhao; Soumyanil Banerjee; Shixuan Liu; Akshay Rao; Akhil Kondepudi; Honglak Lee; Todd C. Hollon
>
> **摘要:** Training vision models with language supervision enables general and transferable representations. However, many visual domains, especially non-object-centric domains such as medical imaging and remote sensing, contain itemized text annotations: multiple text items describing distinct and semantically independent findings within a single image. Such supervision differs from standard multi-caption supervision, where captions are redundant or highly overlapping. Here, we introduce ItemizedCLIP, a framework for learning complete and explainable visual representations from itemized text supervision. ItemizedCLIP employs a cross-attention module to produce text item-conditioned visual embeddings and a set of tailored objectives that jointly enforce item independence (distinct regions for distinct items) and representation completeness (coverage of all items). Across four domains with naturally itemized text supervision (brain MRI, head CT, chest CT, remote sensing) and one additional synthetically itemized dataset, ItemizedCLIP achieves substantial improvements in zero-shot performance and fine-grained interpretability over baselines. The resulting ItemizedCLIP representations are semantically grounded, item-differentiable, complete, and visually interpretable. Our code is available at https://github.com/MLNeurosurg/ItemizedCLIP.
>
---
#### [new 071] HFS: Holistic Query-Aware Frame Selection for Efficient Video Reasoning
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文研究视频理解中的关键帧选择任务，旨在解决传统方法因独立打分导致的冗余和静态伪标签问题。提出端到端可训练框架HFS，通过查询感知的隐式向量、集级别优化与师生互学习，实现任务自适应的高效帧选择。**

- **链接: [https://arxiv.org/pdf/2512.11534v1](https://arxiv.org/pdf/2512.11534v1)**

> **作者:** Yiqing Yang; Kin-Man Lam
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Key frame selection in video understanding presents significant challenges. Traditional top-K selection methods, which score frames independently, often fail to optimize the selection as a whole. This independent scoring frequently results in selecting frames that are temporally clustered and visually redundant. Additionally, training lightweight selectors using pseudo labels generated offline by Multimodal Large Language Models (MLLMs) prevents the supervisory signal from dynamically adapting to task objectives. To address these limitations, we propose an end-to-end trainable, task-adaptive framework for frame selection. A Chain-of-Thought approach guides a Small Language Model (SLM) to generate task-specific implicit query vectors, which are combined with multimodal features to enable dynamic frame scoring. We further define a continuous set-level objective function that incorporates relevance, coverage, and redundancy, enabling differentiable optimization via Gumbel-Softmax to select optimal frame combinations at the set level. Finally, student-teacher mutual learning is employed, where the student selector (SLM) and teacher reasoner (MLLM) are trained to align their frame importance distributions via KL divergence. Combined with cross-entropy loss, this enables end-to-end optimization, eliminating reliance on static pseudo labels. Experiments across various benchmarks, including Video-MME, LongVideoBench, MLVU, and NExT-QA, demonstrate that our method significantly outperforms existing approaches.
>
---
#### [new 072] FactorPortrait: Controllable Portrait Animation via Disentangled Expression, Pose, and Viewpoint
- **分类: cs.CV**

- **简介: 该论文研究可控人像动画生成，旨在解耦表情、姿态与视角控制。提出FactorPortrait方法，通过预训练编码器提取表情潜变量，结合Plücker射线图与法线图实现多维度控制，在合成数据上训练，实现了高真实感、精准控制的动态人像生成。**

- **链接: [https://arxiv.org/pdf/2512.11645v1](https://arxiv.org/pdf/2512.11645v1)**

> **作者:** Jiapeng Tang; Kai Li; Chengxiang Yin; Liuhao Ge; Fei Jiang; Jiu Xu; Matthias Nießner; Christian Häne; Timur Bagautdinov; Egor Zakharov; Peihong Guo
>
> **备注:** Project page: https://tangjiapeng.github.io/FactorPortrait/
>
> **摘要:** We introduce FactorPortrait, a video diffusion method for controllable portrait animation that enables lifelike synthesis from disentangled control signals of facial expressions, head movement, and camera viewpoints. Given a single portrait image, a driving video, and camera trajectories, our method animates the portrait by transferring facial expressions and head movements from the driving video while simultaneously enabling novel view synthesis from arbitrary viewpoints. We utilize a pre-trained image encoder to extract facial expression latents from the driving video as control signals for animation generation. Such latents implicitly capture nuanced facial expression dynamics with identity and pose information disentangled, and they are efficiently injected into the video diffusion transformer through our proposed expression controller. For camera and head pose control, we employ Plücker ray maps and normal maps rendered from 3D body mesh tracking. To train our model, we curate a large-scale synthetic dataset containing diverse combinations of camera viewpoints, head poses, and facial expression dynamics. Extensive experiments demonstrate that our method outperforms existing approaches in realism, expressiveness, control accuracy, and view consistency.
>
---
#### [new 073] FutureX: Enhance End-to-End Autonomous Driving via Latent Chain-of-Thought World Model
- **分类: cs.CV**

- **简介: 该论文针对端到端自动驾驶在动态环境中决策不足的问题，提出FutureX框架。通过引入潜在思维链世界模型，实现对未来场景的推理与轨迹优化，结合自动切换机制提升复杂场景下的决策质量与安全性。**

- **链接: [https://arxiv.org/pdf/2512.11226v1](https://arxiv.org/pdf/2512.11226v1)**

> **作者:** Hongbin Lin; Yiming Yang; Yifan Zhang; Chaoda Zheng; Jie Feng; Sheng Wang; Zhennan Wang; Shijia Chen; Boyang Wang; Yu Zhang; Xianming Liu; Shuguang Cui; Zhen Li
>
> **摘要:** In autonomous driving, end-to-end planners learn scene representations from raw sensor data and utilize them to generate a motion plan or control actions. However, exclusive reliance on the current scene for motion planning may result in suboptimal responses in highly dynamic traffic environments where ego actions further alter the future scene. To model the evolution of future scenes, we leverage the World Model to represent how the ego vehicle and its environment interact and change over time, which entails complex reasoning. The Chain of Thought (CoT) offers a promising solution by forecasting a sequence of future thoughts that subsequently guide trajectory refinement. In this paper, we propose FutureX, a CoT-driven pipeline that enhances end-to-end planners to perform complex motion planning via future scene latent reasoning and trajectory refinement. Specifically, the Auto-think Switch examines the current scene and decides whether additional reasoning is required to yield a higher-quality motion plan. Once FutureX enters the Thinking mode, the Latent World Model conducts a CoT-guided rollout to predict future scene representation, enabling the Summarizer Module to further refine the motion plan. Otherwise, FutureX operates in an Instant mode to generate motion plans in a forward pass for relatively simple scenes. Extensive experiments demonstrate that FutureX enhances existing methods by producing more rational motion plans and fewer collisions without compromising efficiency, thereby achieving substantial overall performance gains, e.g., 6.2 PDMS improvement for TransFuser on NAVSIM. Code will be released.
>
---
#### [new 074] Multi-temporal Calving Front Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦冰川崩解前沿分割任务，旨在解决季节性干扰导致的分割不准确问题。通过在Tyrion模型中引入多时相特征融合，利用时间序列信息提升稳定性，显著提高了CaFFe数据集上的性能，达到新的SOTA水平。**

- **链接: [https://arxiv.org/pdf/2512.11560v1](https://arxiv.org/pdf/2512.11560v1)**

> **作者:** Marcel Dreier; Nora Gourmelon; Dakota Pyles; Fei Wu; Matthias Braun; Thorsten Seehaus; Andreas Maier; Vincent Christlein
>
> **摘要:** The calving fronts of marine-terminating glaciers undergo constant changes. These changes significantly affect the glacier's mass and dynamics, demanding continuous monitoring. To address this need, deep learning models were developed that can automatically delineate the calving front in Synthetic Aperture Radar imagery. However, these models often struggle to correctly classify areas affected by seasonal conditions such as ice melange or snow-covered surfaces. To address this issue, we propose to process multiple frames from a satellite image time series of the same glacier in parallel and exchange temporal information between the corresponding feature maps to stabilize each prediction. We integrate our approach into the current state-of-the-art architecture Tyrion and accomplish a new state-of-the-art performance on the CaFFe benchmark dataset. In particular, we achieve a Mean Distance Error of 184.4 m and a mean Intersection over Union of 83.6.
>
---
#### [new 075] Vision-Language Models for Infrared Industrial Sensing in Additive Manufacturing Scene Description
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究零样本工业红外感知任务，解决现有视觉语言模型无法理解红外图像的问题。提出VLM-IRIS框架，将红外图像转为RGB兼容的伪彩色图像，结合CLIP模型实现无需训练的工件存在检测，适用于增材制造中的无标签热监控。**

- **链接: [https://arxiv.org/pdf/2512.11098v1](https://arxiv.org/pdf/2512.11098v1)**

> **作者:** Nazanin Mahjourian; Vinh Nguyen
>
> **摘要:** Many manufacturing environments operate in low-light conditions or within enclosed machines where conventional vision systems struggle. Infrared cameras provide complementary advantages in such environments. Simultaneously, supervised AI systems require large labeled datasets, which makes zero-shot learning frameworks more practical for applications including infrared cameras. Recent advances in vision-language foundation models (VLMs) offer a new path in zero-shot predictions from paired image-text representations. However, current VLMs cannot understand infrared camera data since they are trained on RGB data. This work introduces VLM-IRIS (Vision-Language Models for InfraRed Industrial Sensing), a zero-shot framework that adapts VLMs to infrared data by preprocessing infrared images captured by a FLIR Boson sensor into RGB-compatible inputs suitable for CLIP-based encoders. We demonstrate zero-shot workpiece presence detection on a 3D printer bed where temperature differences between the build plate and workpieces make the task well-suited for thermal imaging. VLM-IRIS converts the infrared images to magma representation and applies centroid prompt ensembling with a CLIP ViT-B/32 encoder to achieve high accuracy on infrared images without any model retraining. These findings demonstrate that the proposed improvements to VLMs can be effectively extended to thermal applications for label-free monitoring.
>
---
#### [new 076] Surveillance Video-Based Traffic Accident Detection Using Transformer Architecture
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于交通监控任务，旨在提升交通事故检测的准确性和泛化能力。针对现有方法时空建模不足和数据集局限的问题，作者构建了多样化数据集，提出融合RGB与光流特征的Transformer模型，并验证其优于现有VLM方法。**

- **链接: [https://arxiv.org/pdf/2512.11350v1](https://arxiv.org/pdf/2512.11350v1)**

> **作者:** Tanu Singh; Pranamesh Chakraborty; Long T. Truong
>
> **摘要:** Road traffic accidents represent a leading cause of mortality globally, with incidence rates rising due to increasing population, urbanization, and motorization. Rising accident rates raise concerns about traffic surveillance effectiveness. Traditional computer vision methods for accident detection struggle with limited spatiotemporal understanding and poor cross-domain generalization. Recent advances in transformer architectures excel at modeling global spatial-temporal dependencies and parallel computation. However, applying these models to automated traffic accident detection is limited by small, non-diverse datasets, hindering the development of robust, generalizable systems. To address this gap, we curated a comprehensive and balanced dataset that captures a wide spectrum of traffic environments, accident types, and contextual variations. Utilizing the curated dataset, we propose an accident detection model based on a transformer architecture using pre-extracted spatial video features. The architecture employs convolutional layers to extract local correlations across diverse patterns within a frame, while leveraging transformers to capture sequential-temporal dependencies among the retrieved features. Moreover, most existing studies neglect the integration of motion cues, which are essential for understanding dynamic scenes, especially during accidents. These approaches typically rely on static features or coarse temporal information. In this study, multiple methods for incorporating motion cues were evaluated to identify the most effective strategy. Among the tested input approaches, concatenating RGB features with optical flow achieved the highest accuracy at 88.3%. The results were further compared with vision language models (VLM) such as GPT, Gemini, and LLaVA-NeXT-Video to assess the effectiveness of the proposed method.
>
---
#### [new 077] Image Tiling for High-Resolution Reasoning: Balancing Local Detail with Global Context
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于高分辨率图像理解任务，旨在解决局部细节与全局上下文的平衡问题。作者复现并分析了Monkey VLM的图像分块方法，验证其有效性，并探究全局信息的影响，揭示结果受任务类型和分块粒度影响。**

- **链接: [https://arxiv.org/pdf/2512.11167v1](https://arxiv.org/pdf/2512.11167v1)**

> **作者:** Anatole Jacquin de Margerie; Alexis Roger; Irina Rish
>
> **备注:** Accepted in AAAI 2025 Workshop on Reproducible AI
>
> **摘要:** Reproducibility remains a cornerstone of scientific progress, yet complex multimodal models often lack transparent implementation details and accessible training infrastructure. In this work, we present a detailed reproduction and critical analysis of the Monkey Vision-Language Model (VLM) (Li et al. 2023b) published in CVPR24, a recent approach to high-resolution image understanding via image tiling. The original paper proposed splitting large images into tiles to recover fine-grained visual details while maintaining computational efficiency. Our study replicates this strategy using open checkpoints and reimplements the training pipeline. We confirm the key finding of the original Monkey VLM work, namely that tiling effectively recovers local details. We then extend this work further, by investigating the effect of the inclusion of the global context, which provide practical insights for future high-resolution multimodal modeling. However, we also report deviations in the results, with the magnitude of these effects depending heavily on task type and tile granularity.
>
---
#### [new 078] Particulate: Feed-Forward 3D Object Articulation
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文提出Particulate，解决单张静态3D网格的物体关节结构推断问题。基于Transformer网络，直接预测部件、运动结构与约束，实现快速、端到端的3D物体关节约束建模，支持多关节，适用于真实与生成数据。**

- **链接: [https://arxiv.org/pdf/2512.11798v1](https://arxiv.org/pdf/2512.11798v1)**

> **作者:** Ruining Li; Yuxin Yao; Chuanxia Zheng; Christian Rupprecht; Joan Lasenby; Shangzhe Wu; Andrea Vedaldi
>
> **备注:** Project page: https://ruiningli.com/particulate
>
> **摘要:** We present Particulate, a feed-forward approach that, given a single static 3D mesh of an everyday object, directly infers all attributes of the underlying articulated structure, including its 3D parts, kinematic structure, and motion constraints. At its core is a transformer network, Part Articulation Transformer, which processes a point cloud of the input mesh using a flexible and scalable architecture to predict all the aforementioned attributes with native multi-joint support. We train the network end-to-end on a diverse collection of articulated 3D assets from public datasets. During inference, Particulate lifts the network's feed-forward prediction to the input mesh, yielding a fully articulated 3D model in seconds, much faster than prior approaches that require per-object optimization. Particulate can also accurately infer the articulated structure of AI-generated 3D assets, enabling full-fledged extraction of articulated 3D objects from a single (real or synthetic) image when combined with an off-the-shelf image-to-3D generator. We further introduce a new challenging benchmark for 3D articulation estimation curated from high-quality public 3D assets, and redesign the evaluation protocol to be more consistent with human preferences. Quantitative and qualitative results show that Particulate significantly outperforms state-of-the-art approaches.
>
---
#### [new 079] VLM2GeoVec: Toward Universal Multimodal Embeddings for Remote Sensing
- **分类: cs.CV; cs.IR**

- **简介: 该论文提出VLM2GeoVec，旨在解决遥感中多模态表征割裂问题。通过单编码器统一图像、文本、框和坐标嵌入，支持检索与区域推理。构建新基准RSMEB验证其在多种任务上的优越性能。**

- **链接: [https://arxiv.org/pdf/2512.11490v1](https://arxiv.org/pdf/2512.11490v1)**

> **作者:** Emanuel Sánchez Aimar; Gulnaz Zhambulova; Fahad Shahbaz Khan; Yonghao Xu; Michael Felsberg
>
> **备注:** 21 pages, 7 figures, under review
>
> **摘要:** Satellite imagery differs fundamentally from natural images: its aerial viewpoint, very high resolution, diverse scale variations, and abundance of small objects demand both region-level spatial reasoning and holistic scene understanding. Current remote-sensing approaches remain fragmented between dual-encoder retrieval models, which excel at large-scale cross-modal search but cannot interleave modalities, and generative assistants, which support region-level interpretation but lack scalable retrieval capabilities. We propose $\textbf{VLM2GeoVec}$, an instruction-following, single-encoder vision-language model trained contrastively to embed interleaved inputs (images, text, bounding boxes, and geographic coordinates) in a unified vector space. Our single encoder interleaves all inputs into one joint embedding trained with a contrastive loss, eliminating multi-stage pipelines and task-specific modules. To evaluate its versatility, we introduce $\textbf{RSMEB}$, a novel benchmark covering key remote-sensing embedding applications: scene classification; cross-modal search; compositional retrieval; visual-question answering; visual grounding and region-level reasoning; and semantic geospatial retrieval. On RSMEB, it achieves $\textbf{26.6%}$ P@1 on region-caption retrieval (+25 pp vs. dual-encoder baselines), $\textbf{32.5%}$ P@1 on referring-expression retrieval (+19 pp), and $\textbf{17.8%}$ P@1 on semantic geo-localization retrieval (over $3\times$ prior best), while matching or exceeding specialized baselines on conventional tasks such as scene classification and cross-modal retrieval. VLM2GeoVec unifies scalable retrieval with region-level spatial reasoning, enabling cohesive multimodal analysis in remote sensing. We will publicly release the code, checkpoints, and data upon acceptance.
>
---
#### [new 080] V-RGBX: Video Editing with Accurate Controls over Intrinsic Properties
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，旨在解决现有方法缺乏对固有属性（如反照率、法线等）精确控制的问题。作者提出V-RGBX框架，首次实现端到端的基于固有属性的视频编辑，支持逆向渲染、合成与关键帧编辑，提升编辑的物理合理性和时序一致性。**

- **链接: [https://arxiv.org/pdf/2512.11799v1](https://arxiv.org/pdf/2512.11799v1)**

> **作者:** Ye Fang; Tong Wu; Valentin Deschaintre; Duygu Ceylan; Iliyan Georgiev; Chun-Hao Paul Huang; Yiwei Hu; Xuelin Chen; Tuanfeng Yang Wang
>
> **备注:** Project Page: https://aleafy.github.io/vrgbx
>
> **摘要:** Large-scale video generation models have shown remarkable potential in modeling photorealistic appearance and lighting interactions in real-world scenes. However, a closed-loop framework that jointly understands intrinsic scene properties (e.g., albedo, normal, material, and irradiance), leverages them for video synthesis, and supports editable intrinsic representations remains unexplored. We present V-RGBX, the first end-to-end framework for intrinsic-aware video editing. V-RGBX unifies three key capabilities: (1) video inverse rendering into intrinsic channels, (2) photorealistic video synthesis from these intrinsic representations, and (3) keyframe-based video editing conditioned on intrinsic channels. At the core of V-RGBX is an interleaved conditioning mechanism that enables intuitive, physically grounded video editing through user-selected keyframes, supporting flexible manipulation of any intrinsic modality. Extensive qualitative and quantitative results show that V-RGBX produces temporally consistent, photorealistic videos while propagating keyframe edits across sequences in a physically plausible manner. We demonstrate its effectiveness in diverse applications, including object appearance editing and scene-level relighting, surpassing the performance of prior methods.
>
---
#### [new 081] SATMapTR: Satellite Image Enhanced Online HD Map Construction
- **分类: cs.CV**

- **简介: 该论文研究在线高精地图构建任务，旨在解决车载传感器数据质量低和卫星图像受遮挡导致的建图不准确问题。提出SATMapTR模型，通过门控特征优化和几何感知融合模块，有效融合卫星与BEV特征，显著提升精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.11319v1](https://arxiv.org/pdf/2512.11319v1)**

> **作者:** Bingyuan Huang; Guanyi Zhao; Qian Xu; Yang Lou; Yung-Hui Li; Jianping Wang
>
> **备注:** 9 pages (+ 3 pages of Appendix)
>
> **摘要:** High-definition (HD) maps are evolving from pre-annotated to real-time construction to better support autonomous driving in diverse scenarios. However, this process is hindered by low-quality input data caused by onboard sensors limited capability and frequent occlusions, leading to incomplete, noisy, or missing data, and thus reduced mapping accuracy and robustness. Recent efforts have introduced satellite images as auxiliary input, offering a stable, wide-area view to complement the limited ego perspective. However, satellite images in Bird's Eye View are often degraded by shadows and occlusions from vegetation and buildings. Prior methods using basic feature extraction and fusion remain ineffective. To address these challenges, we propose SATMapTR, a novel online map construction model that effectively fuses satellite image through two key components: (1) a gated feature refinement module that adaptively filters satellite image features by integrating high-level semantics with low-level structural cues to extract high signal-to-noise ratio map-relevant representations; and (2) a geometry-aware fusion module that consistently fuse satellite and BEV features at a grid-to-grid level, minimizing interference from irrelevant regions and low-quality inputs. Experimental results on the nuScenes dataset show that SATMapTR achieves the highest mean average precision (mAP) of 73.8, outperforming state-of-the-art satellite-enhanced models by up to 14.2 mAP. It also shows lower mAP degradation under adverse weather and sensor failures, and achieves nearly 3 times higher mAP at extended perception ranges.
>
---
#### [new 082] TSkel-Mamba: Temporal Dynamic Modeling via State Space Model for Human Skeleton-based Action Recognition
- **分类: cs.CV**

- **简介: 该论文研究基于骨架的动作识别任务，旨在提升时序建模能力。针对Mamba模型缺乏跨通道交互的问题，提出TSkel-Mamba框架，引入含多尺度时序交互模块的时序动态建模块，增强跨通道时序依赖学习，兼顾性能与效率。**

- **链接: [https://arxiv.org/pdf/2512.11503v1](https://arxiv.org/pdf/2512.11503v1)**

> **作者:** Yanan Liu; Jun Liu; Hao Zhang; Dan Xu; Hossein Rahmani; Mohammed Bennamoun; Qiuhong Ke
>
> **摘要:** Skeleton-based action recognition has garnered significant attention in the computer vision community. Inspired by the recent success of the selective state-space model (SSM) Mamba in modeling 1D temporal sequences, we propose TSkel-Mamba, a hybrid Transformer-Mamba framework that effectively captures both spatial and temporal dynamics. In particular, our approach leverages Spatial Transformer for spatial feature learning while utilizing Mamba for temporal modeling. Mamba, however, employs separate SSM blocks for individual channels, which inherently limits its ability to model inter-channel dependencies. To better adapt Mamba for skeleton data and enhance Mamba`s ability to model temporal dependencies, we introduce a Temporal Dynamic Modeling (TDM) block, which is a versatile plug-and-play component that integrates a novel Multi-scale Temporal Interaction (MTI) module. The MTI module employs multi-scale Cycle operators to capture cross-channel temporal interactions, a critical factor in action recognition. Extensive experiments on NTU-RGB+D 60, NTU-RGB+D 120, NW-UCLA and UAV-Human datasets demonstrate that TSkel-Mamba achieves state-of-the-art performance while maintaining low inference time, making it both efficient and highly effective.
>
---
#### [new 083] Reframing Music-Driven 2D Dance Pose Generation as Multi-Channel Image Generation
- **分类: cs.CV**

- **简介: 该论文研究音乐驱动的2D舞蹈姿态生成，旨在解决动作与音乐节奏对齐及姿态多样性问题。作者将姿态序列视为多通道图像，采用图像生成框架建模，并提出时序同步机制与参考姿态条件策略，提升生成质量与一致性。**

- **链接: [https://arxiv.org/pdf/2512.11720v1](https://arxiv.org/pdf/2512.11720v1)**

> **作者:** Yan Zhang; Han Zou; Lincong Feng; Cong Xie; Ruiqi Yu; Zhenpeng Zhan
>
> **摘要:** Recent pose-to-video models can translate 2D pose sequences into photorealistic, identity-preserving dance videos, so the key challenge is to generate temporally coherent, rhythm-aligned 2D poses from music, especially under complex, high-variance in-the-wild distributions. We address this by reframing music-to-dance generation as a music-token-conditioned multi-channel image synthesis problem: 2D pose sequences are encoded as one-hot images, compressed by a pretrained image VAE, and modeled with a DiT-style backbone, allowing us to inherit architectural and training advances from modern text-to-image models and better capture high-variance 2D pose distributions. On top of this formulation, we introduce (i) a time-shared temporal indexing scheme that explicitly synchronizes music tokens and pose latents over time and (ii) a reference-pose conditioning strategy that preserves subject-specific body proportions and on-screen scale while enabling long-horizon segment-and-stitch generation. Experiments on a large in-the-wild 2D dance corpus and the calibrated AIST++2D benchmark show consistent improvements over representative music-to-dance methods in pose- and video-space metrics and human preference, and ablations validate the contributions of the representation, temporal indexing, and reference conditioning. See supplementary videos at https://hot-dance.github.io
>
---
#### [new 084] Text images processing system using artificial intelligence models
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出一种基于AI的文本图像分类系统，旨在解决复杂条件下文本识别与分类问题。通过DBNet++检测文本，BART模型分类，实现对发票、表单、信件和报告四类文档的高效识别，适应多变光照、角度、低分辨率等实际场景。**

- **链接: [https://arxiv.org/pdf/2512.11691v1](https://arxiv.org/pdf/2512.11691v1)**

> **作者:** Aya Kaysan Bahjat
>
> **备注:** 8 pages, 12 figures, article
>
> **摘要:** This is to present a text image classifier device that identifies textual content in images and then categorizes each image into one of four predefined categories, including Invoice, Form, Letter, or Report. The device supports a gallery mode, in which users browse files on flash disks, hard disk drives, or microSD cards, and a live mode which renders feeds of cameras connected to it. Its design is specifically aimed at addressing pragmatic challenges, such as changing light, random orientation, curvature or partial coverage of text, low resolution, and slightly visible text. The steps of the processing process are divided into four steps: image acquisition and preprocessing, textual elements detection with the help of DBNet++ (Differentiable Binarization Network Plus) model, BART (Bidirectional Auto-Regressive Transformers) model that classifies detected textual elements, and the presentation of the results through a user interface written in Python and PyQt5. All the stages are connected in such a way that they form a smooth workflow. The system achieved a text recognition rate of about 94.62% when tested over ten hours on the mentioned Total-Text dataset, that includes high resolution images, created so as to represent a wide range of problematic conditions. These experimental results support the effectiveness of the suggested methodology to practice, mixed-source text categorization, even in uncontrolled imaging conditions.
>
---
#### [new 085] In-Context Learning for Seismic Data Processing
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究地震数据去多次波任务，旨在解决传统方法与深度学习模型在空间一致性及用户控制方面的不足。提出ContextSeisNet，利用上下文学习机制，通过邻近地震道集示例实现无需重训练的自适应处理，提升横向一致性和数据效率。**

- **链接: [https://arxiv.org/pdf/2512.11575v1](https://arxiv.org/pdf/2512.11575v1)**

> **作者:** Fabian Fuchs; Mario Ruben Fernandez; Norman Ettrich; Janis Keuper
>
> **备注:** Source code available under https://codeberg.org/fuchsfa/in-context-learning-seismic
>
> **摘要:** Seismic processing transforms raw data into subsurface images essential for geophysical applications. Traditional methods face challenges, such as noisy data, and manual parameter tuning, among others. Recently deep learning approaches have proposed alternative solutions to some of these problems. However, important challenges of existing deep learning approaches are spatially inconsistent results across neighboring seismic gathers and lack of user-control. We address these limitations by introducing ContextSeisNet, an in-context learning model, to seismic demultiple processing. Our approach conditions predictions on a support set of spatially related example pairs: neighboring common-depth point gathers from the same seismic line and their corresponding labels. This allows the model to learn task-specific processing behavior at inference time by observing how similar gathers should be processed, without any retraining. This method provides both flexibility through user-defined examples and improved lateral consistency across seismic lines. On synthetic data, ContextSeisNet outperforms a U-Net baseline quantitatively and demonstrates enhanced spatial coherence between neighboring gathers. On field data, our model achieves superior lateral consistency compared to both traditional Radon demultiple and the U-Net baseline. Relative to the U-Net, ContextSeisNet also delivers improved near-offset performance and more complete multiple removal. Notably, ContextSeisNet achieves comparable field data performance despite being trained on 90% less data, demonstrating substantial data efficiency. These results establish ContextSeisNet as a practical approach for spatially consistent seismic demultiple with potential applicability to other seismic processing tasks.
>
---
#### [new 086] Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video
- **分类: cs.CV**

- **简介: 该论文研究从普通单目视频中重建动态场景。针对现有方法在细节和运动一致性上的不足，提出增强先验的高斯点阵方法，利用分割与误差图生成精确掩膜，优化深度与轨迹，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2512.11356v1](https://arxiv.org/pdf/2512.11356v1)**

> **作者:** Meng-Li Shih; Ying-Huan Chen; Yu-Lun Liu; Brian Curless
>
> **摘要:** We introduce a fully automatic pipeline for dynamic scene reconstruction from casually captured monocular RGB videos. Rather than designing a new scene representation, we enhance the priors that drive Dynamic Gaussian Splatting. Video segmentation combined with epipolar-error maps yields object-level masks that closely follow thin structures; these masks (i) guide an object-depth loss that sharpens the consistent video depth, and (ii) support skeleton-based sampling plus mask-guided re-identification to produce reliable, comprehensive 2-D tracks. Two additional objectives embed the refined priors in the reconstruction stage: a virtual-view depth loss removes floaters, and a scaffold-projection loss ties motion nodes to the tracks, preserving fine geometry and coherent motion. The resulting system surpasses previous monocular dynamic scene reconstruction methods and delivers visibly superior renderings
>
---
#### [new 087] Weakly Supervised Tuberculosis Localization in Chest X-rays through Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文研究弱监督结核病定位任务，旨在解决缺乏精细标注和模型泛化性差的问题。通过知识蒸馏框架，利用分类标签训练学生模型，实现无需边界框标注的病灶定位，在TBX11k数据集上取得良好效果。**

- **链接: [https://arxiv.org/pdf/2512.11057v1](https://arxiv.org/pdf/2512.11057v1)**

> **作者:** Marshal Ashif Shawkat; Moidul Hasan; Taufiq Hasan
>
> **备注:** 18 pages, 9 figures, 4 tables
>
> **摘要:** Tuberculosis (TB) remains one of the leading causes of mortality worldwide, particularly in resource-limited countries. Chest X-ray (CXR) imaging serves as an accessible and cost-effective diagnostic tool but requires expert interpretation, which is often unavailable. Although machine learning models have shown high performance in TB classification, they often depend on spurious correlations and fail to generalize. Besides, building large datasets featuring high-quality annotations for medical images demands substantial resources and input from domain specialists, and typically involves several annotators reaching agreement, which results in enormous financial and logistical expenses. This study repurposes knowledge distillation technique to train CNN models reducing spurious correlations and localize TB-related abnormalities without requiring bounding-box annotations. By leveraging a teacher-student framework with ResNet50 architecture, the proposed method trained on TBX11k dataset achieve impressive 0.2428 mIOU score. Experimental results further reveal that the student model consistently outperforms the teacher, underscoring improved robustness and potential for broader clinical deployment in diverse settings.
>
---
#### [new 088] Out-of-Distribution Segmentation via Wasserstein-Based Evidential Uncertainty
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究开放世界中的语义分割，旨在识别并分割模型未见过的未知物体。提出基于Wasserstein损失的证据分割框架，结合KL正则化与Dice一致性，提升分布外物体的检测性能。**

- **链接: [https://arxiv.org/pdf/2512.11373v1](https://arxiv.org/pdf/2512.11373v1)**

> **作者:** Arnold Brosch; Abdelrahman Eldesokey; Michael Felsberg; Kira Maag
>
> **摘要:** Deep neural networks achieve superior performance in semantic segmentation, but are limited to a predefined set of classes, which leads to failures when they encounter unknown objects in open-world scenarios. Recognizing and segmenting these out-of-distribution (OOD) objects is crucial for safety-critical applications such as automated driving. In this work, we present an evidence segmentation framework using a Wasserstein loss, which captures distributional distances while respecting the probability simplex geometry. Combined with Kullback-Leibler regularization and Dice structural consistency terms, our approach leads to improved OOD segmentation performance compared to uncertainty-based approaches.
>
---
#### [new 089] REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation
- **分类: cs.CV; cs.SD**

- **简介: 该论文研究语音驱动的实时流式说话人头生成任务，旨在解决扩散模型推理慢、难以实时生成的问题。提出REST框架，通过紧凑潜在空间、ID-Context缓存机制和异步流式蒸馏策略，实现快速、连贯的端到端生成。**

- **链接: [https://arxiv.org/pdf/2512.11229v1](https://arxiv.org/pdf/2512.11229v1)**

> **作者:** Haotian Wang; Yuzhe Weng; Xinyi Yu; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Qingfeng Liu
>
> **备注:** 10pages, 4 figures
>
> **摘要:** Diffusion models have significantly advanced the field of talking head generation. However, the slow inference speeds and non-autoregressive paradigms severely constrain the application of diffusion-based THG models. In this study, we propose REST, the first diffusion-based, real-time, end-to-end streaming audio-driven talking head generation framework. To support real-time end-to-end generation, a compact video latent space is first learned through high spatiotemporal VAE compression. Additionally, to enable autoregressive streaming within the compact video latent space, we introduce an ID-Context Cache mechanism, which integrates ID-Sink and Context-Cache principles to key-value caching for maintaining temporal consistency and identity coherence during long-time streaming generation. Furthermore, an Asynchronous Streaming Distillation (ASD) training strategy is proposed to mitigate error accumulation in autoregressive generation and enhance temporal consistency, which leverages a non-streaming teacher with an asynchronous noise schedule to supervise the training of the streaming student model. REST bridges the gap between autoregressive and diffusion-based approaches, demonstrating substantial value for applications requiring real-time talking head generation. Experimental results demonstrate that REST outperforms state-of-the-art methods in both generation speed and overall performance.
>
---
#### [new 090] DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属自监督3D点云表征学习，旨在解决不规则几何、语义不平衡等问题。提出DOS框架，通过在可观测点蒸馏软映射并引入Zipfian原型与Zipf-Sinkhorn算法，增强监督信号，提升分割与检测性能。**

- **链接: [https://arxiv.org/pdf/2512.11465v1](https://arxiv.org/pdf/2512.11465v1)**

> **作者:** Mohamed Abdelsamad; Michael Ulrich; Bin Yang; Miao Zhang; Yakov Miron; Abhinav Valada
>
> **备注:** AAAI-26
>
> **摘要:** Recent advances in self-supervised learning (SSL) have shown tremendous potential for learning 3D point cloud representations without human annotations. However, SSL for 3D point clouds still faces critical challenges due to irregular geometry, shortcut-prone reconstruction, and unbalanced semantics distribution. In this work, we propose DOS (Distilling Observable Softmaps), a novel SSL framework that self-distills semantic relevance softmaps only at observable (unmasked) points. This strategy prevents information leakage from masked regions and provides richer supervision than discrete token-to-prototype assignments. To address the challenge of unbalanced semantics in an unsupervised setting, we introduce Zipfian prototypes and incorporate them using a modified Sinkhorn-Knopp algorithm, Zipf-Sinkhorn, which enforces a power-law prior over prototype usage and modulates the sharpness of the target softmap during training. DOS outperforms current state-of-the-art methods on semantic segmentation and 3D object detection across multiple benchmarks, including nuScenes, Waymo, SemanticKITTI, ScanNet, and ScanNet200, without relying on extra data or annotations. Our results demonstrate that observable-point softmaps distillation offers a scalable and effective paradigm for learning robust 3D representations.
>
---
#### [new 091] Minimal Clips, Maximum Salience: Long Video Summarization via Key Moment Extraction
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视频摘要任务，旨在解决长视频中关键信息易丢失且计算成本高的问题。作者提出一种基于关键片段提取的方法，通过轻量级模型生成片段描述，利用大语言模型选择最具信息量的片段，实现高效、准确的多模态视频摘要。**

- **链接: [https://arxiv.org/pdf/2512.11399v1](https://arxiv.org/pdf/2512.11399v1)**

> **作者:** Galann Pennec; Zhengyuan Liu; Nicholas Asher; Philippe Muller; Nancy F. Chen
>
> **摘要:** Vision-Language Models (VLMs) are able to process increasingly longer videos. Yet, important visual information is easily lost throughout the entire context and missed by VLMs. Also, it is important to design tools that enable cost-effective analysis of lengthy video content. In this paper, we propose a clip selection method that targets key video moments to be included in a multimodal summary. We divide the video into short clips and generate compact visual descriptions of each using a lightweight video captioning model. These are then passed to a large language model (LLM), which selects the K clips containing the most relevant visual information for a multimodal summary. We evaluate our approach on reference clips for the task, automatically derived from full human-annotated screenplays and summaries in the MovieSum dataset. We further show that these reference clips (less than 6% of the movie) are sufficient to build a complete multimodal summary of the movies in MovieSum. Using our clip selection method, we achieve a summarization performance close to that of these reference clips while capturing substantially more relevant video information than random clip selection. Importantly, we maintain low computational cost by relying on a lightweight captioning model.
>
---
#### [new 092] Task-Aware Multi-Expert Architecture For Lifelong Deep Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究持续学习任务，旨在解决神经网络在序贯学习中遗忘旧知识的问题。提出TAME方法，通过任务感知的专家选择、共享预测层、回放缓冲区和注意力机制，实现新任务适应与旧知识保留的平衡。**

- **链接: [https://arxiv.org/pdf/2512.11243v1](https://arxiv.org/pdf/2512.11243v1)**

> **作者:** Jianyu Wang; Jacob Nean-Hua Sheikh; Cat P. Le; Hoda Bidkhori
>
> **摘要:** Lifelong deep learning (LDL) trains neural networks to learn sequentially across tasks while preserving prior knowledge. We propose Task-Aware Multi-Expert (TAME), a continual learning algorithm that leverages task similarity to guide expert selection and knowledge transfer. TAME maintains a pool of pretrained neural networks and activates the most relevant expert for each new task. A shared dense layer integrates features from the chosen expert to generate predictions. To reduce catastrophic forgetting, TAME uses a replay buffer that stores representative samples and embeddings from previous tasks and reuses them during training. An attention mechanism further prioritizes the most relevant stored information for each prediction. Together, these components allow TAME to adapt flexibly while retaining important knowledge across evolving task sequences. Experiments on binary classification tasks derived from CIFAR-100 show that TAME improves accuracy on new tasks while sustaining performance on earlier ones, highlighting its effectiveness in balancing adaptation and retention in lifelong learning settings.
>
---
#### [new 093] TV2TV: A Unified Framework for Interleaved Language and Video Generation
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出TV2TV框架，属于视频生成任务，旨在解决复杂语义推理与多分支视频生成难题。通过融合语言与视频的交错生成，利用语言模型进行高层推理，提升生成质量与可控性。**

- **链接: [https://arxiv.org/pdf/2512.05103v2](https://arxiv.org/pdf/2512.05103v2)**

> **作者:** Xiaochuang Han; Youssef Emad; Melissa Hall; John Nguyen; Karthik Padthe; Liam Robbins; Amir Bar; Delong Chen; Michal Drozdzal; Maha Elbayad; Yushi Hu; Shang-Wen Li; Sreya Dutta Roy; Jakob Verbeek; XuDong Wang; Marjan Ghazvininejad; Luke Zettlemoyer; Emily Dinan
>
> **摘要:** Video generation models are rapidly advancing, but can still struggle with complex video outputs that require significant semantic branching or repeated high-level reasoning about what should happen next. In this paper, we introduce a new class of omni video-text models that integrate ideas from recent LM reasoning advances to address this challenge. More specifically, we present TV2TV, a unified generative modeling framework which decomposes video generation into an interleaved text and video generation process. TV2TV jointly learns language modeling (next-token prediction) and video flow matching (next-frame prediction) using a Mixture-of-Transformers (MoT) architecture. At inference time, TV2TV decides when to alternate between generating text and video frames, allowing the model to "think in words" about subsequent content before ``acting in pixels'' to produce frames. This design offloads much of the responsibility for deciding what should happen next to the language modeling tower, enabling improved visual quality and prompt alignment of generated videos. It also enables fine-grained controllability, allowing users to modify the video generation trajectory through text interventions at any point in the process. In controlled experiments on video game data, TV2TV demonstrates substantial improvements in both visual quality and controllability. TV2TV also scales to natural videos, as we show by augmenting sports videos with interleaved natural language action descriptions using vision-language models (VLMs). Training TV2TV on this corpus yields strong visual quality and prompt alignment, showcasing the model's ability to reason about and generate complex real-world action sequences. Together, these results highlight TV2TV as a promising step toward video generation with open-ended textual reasoning and control.
>
---
#### [new 094] Back to the Baseline: Examining Baseline Effects on Explainability Metrics
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于可解释人工智能（XAI）任务，旨在解决归因方法评估中基线选择偏差问题。作者分析现有基线在信息移除与分布合理性间的权衡缺陷，并提出一种新型模型相关基线，以更好平衡二者，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2512.11433v1](https://arxiv.org/pdf/2512.11433v1)**

> **作者:** Agustin Martin Picard; Thibaut Boissin; Varshini Subhash; Rémi Cadène; Thomas Fel
>
> **摘要:** Attribution methods are among the most prevalent techniques in Explainable Artificial Intelligence (XAI) and are usually evaluated and compared using Fidelity metrics, with Insertion and Deletion being the most popular. These metrics rely on a baseline function to alter the pixels of the input image that the attribution map deems most important. In this work, we highlight a critical problem with these metrics: the choice of a given baseline will inevitably favour certain attribution methods over others. More concerningly, even a simple linear model with commonly used baselines contradicts itself by designating different optimal methods. A question then arises: which baseline should we use? We propose to study this problem through two desirable properties of a baseline: (i) that it removes information and (ii) that it does not produce overly out-of-distribution (OOD) images. We first show that none of the tested baselines satisfy both criteria, and there appears to be a trade-off among current baselines: either they remove information or they produce a sequence of OOD images. Finally, we introduce a novel baseline by leveraging recent work in feature visualisation to artificially produce a model-dependent baseline that removes information without being overly OOD, thus improving on the trade-off when compared to other existing baselines. Our code is available at https://github.com/deel-ai-papers/Back-to-the-Baseline
>
---
#### [new 095] Beyond Memorization: Gradient Projection Enables Selective Learning in Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对扩散模型中敏感概念的记忆化问题，提出一种梯度投影框架，通过在训练中将梯度投影到敏感特征的正交空间，实现概念级选择性遗忘，在避免数据浪费的同时有效防止知识产权与隐私泄露。**

- **链接: [https://arxiv.org/pdf/2512.11194v1](https://arxiv.org/pdf/2512.11194v1)**

> **作者:** Divya Kothandaraman; Jaclyn Pytlarz
>
> **摘要:** Memorization in large-scale text-to-image diffusion models poses significant security and intellectual property risks, enabling adversarial attribute extraction and the unauthorized reproduction of sensitive or proprietary features. While conventional dememorization techniques, such as regularization and data filtering, limit overfitting to specific training examples, they fail to systematically prevent the internalization of prohibited concept-level features. Simply discarding all images containing a sensitive feature wastes invaluable training data, necessitating a method for selective unlearning at the concept level. To address this, we introduce a Gradient Projection Framework designed to enforce a stringent requirement of concept-level feature exclusion. Our defense operates during backpropagation by systematically identifying and excising training signals aligned with embeddings of prohibited attributes. Specifically, we project each gradient update onto the orthogonal complement of the sensitive feature's embedding space, thereby zeroing out its influence on the model's weights. Our method integrates seamlessly into standard diffusion model training pipelines and complements existing defenses. We analyze our method against an adversary aiming for feature extraction. In extensive experiments, we demonstrate that our framework drastically reduces memorization while rigorously preserving generation quality and semantic fidelity. By reframing memorization control as selective learning, our approach establishes a new paradigm for IP-safe and privacy-preserving generative AI.
>
---
#### [new 096] Autoencoder-based Semi-Supervised Dimensionality Reduction and Clustering for Scientific Ensembles
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属无监督学习任务，旨在解决高维科学集合数据的降维与聚类难题。提出结合软轮廓损失和对比损失的自编码器框架，利用EfficientNetV2生成伪标签，优化隐空间结构，并用UMAP可视化，提升特征提取与聚类效果。**

- **链接: [https://arxiv.org/pdf/2512.11145v1](https://arxiv.org/pdf/2512.11145v1)**

> **作者:** Lennard Manuel; Hamid Gadirov; Steffen Frey
>
> **备注:** Research Internship Project
>
> **摘要:** Analyzing and visualizing scientific ensemble datasets with high dimensionality and complexity poses significant challenges. Dimensionality reduction techniques and autoencoders are powerful tools for extracting features, but they often struggle with such high-dimensional data. This paper presents an enhanced autoencoder framework that incorporates a clustering loss, based on the soft silhouette score, alongside a contrastive loss to improve the visualization and interpretability of ensemble datasets. First, EfficientNetV2 is used to generate pseudo-labels for the unlabeled portions of the scientific ensemble datasets. By jointly optimizing the reconstruction, clustering, and contrastive objectives, our method encourages similar data points to group together while separating distinct clusters in the latent space. UMAP is subsequently applied to this latent representation to produce 2D projections, which are evaluated using the silhouette score. Multiple types of autoencoders are evaluated and compared based on their ability to extract meaningful features. Experiments on two scientific ensemble datasets - channel structures in soil derived from Markov chain Monte Carlo, and droplet-on-film impact dynamics - show that models incorporating clustering or contrastive loss marginally outperform the baseline approaches.
>
---
#### [new 097] AnchorDream: Repurposing Video Diffusion for Embodiment-Aware Robot Data Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人模仿学习数据生成任务，旨在解决真实数据采集成本高、仿真数据多样性不足的问题。作者提出AnchorDream，利用预训练视频扩散模型，以机器人动作为锚点生成具身一致的多样化演示数据，显著提升策略学习性能。**

- **链接: [https://arxiv.org/pdf/2512.11797v1](https://arxiv.org/pdf/2512.11797v1)**

> **作者:** Junjie Ye; Rong Xue; Basile Van Hoorick; Pavel Tokmakov; Muhammad Zubair Irshad; Yue Wang; Vitor Guizilini
>
> **备注:** Project page: https://jay-ye.github.io/AnchorDream/
>
> **摘要:** The collection of large-scale and diverse robot demonstrations remains a major bottleneck for imitation learning, as real-world data acquisition is costly and simulators offer limited diversity and fidelity with pronounced sim-to-real gaps. While generative models present an attractive solution, existing methods often alter only visual appearances without creating new behaviors, or suffer from embodiment inconsistencies that yield implausible motions. To address these limitations, we introduce AnchorDream, an embodiment-aware world model that repurposes pretrained video diffusion models for robot data synthesis. AnchorDream conditions the diffusion process on robot motion renderings, anchoring the embodiment to prevent hallucination while synthesizing objects and environments consistent with the robot's kinematics. Starting from only a handful of human teleoperation demonstrations, our method scales them into large, diverse, high-quality datasets without requiring explicit environment modeling. Experiments show that the generated data leads to consistent improvements in downstream policy learning, with relative gains of 36.4% in simulator benchmarks and nearly double performance in real-world studies. These results suggest that grounding generative world models in robot motion provides a practical path toward scaling imitation learning.
>
---
#### [new 098] Multimodal Fusion of Regional Brain Experts for Interpretable Alzheimer's Disease Diagnosis
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **简介: 该论文研究阿尔茨海默病的多模态诊断，旨在解决传统融合方法无法自适应平衡不同脑区生物标志物贡献的问题。提出MREF-AD模型，通过区域专家与双层门控机制实现可解释的模态融合，在提升诊断性能的同时提供模态与脑区级解释性。**

- **链接: [https://arxiv.org/pdf/2512.10966v1](https://arxiv.org/pdf/2512.10966v1)**

> **作者:** Farica Zhuang; Dinara Aliyeva; Shu Yang; Zixuan Wen; Duy Duong-Tran; Christos Davatzikos; Tianlong Chen; Song Wang; Li Shen
>
> **摘要:** Accurate and early diagnosis of Alzheimer's disease (AD) can benefit from integrating complementary information from multiple modalities, mirroring clinical practice. However, conventional fusion approaches often rely on simple concatenation of features, which cannot adaptively balance the contributions of biomarkers such as amyloid PET and MRI across brain regions. In this work, we propose MREF-AD, a Multimodal Regional Expert Fusion model for AD diagnosis. It is a Mixture-of-Experts (MoE) framework that models meso-scale brain regions in each modality as an independent expert and employs two-level gating networks to learn subject-specific fusion weights. Beyond improving diagnostic performance, MREF-AD provides modality- and region-level insight into how structural and molecular imaging jointly contribute to disease diagnosis. Using data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), MREF-AD achieves state-of-the-art performance over baselines while providing enhanced interpretability of brain region-specific biomarker relevance, underscoring its utility as a general framework for adaptive and interpretable multimodal fusion in neuroimaging.
>
---
#### [new 099] WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文聚焦人形机器人全身运动-操作控制任务，旨在解决现有方法在操作感知行走和大空间运动中的局限。提出统一的隐式学习框架WholeBodyVLA，结合视觉-语言-动作模型与专用强化学习策略，提升运动精度与泛化能力，实现高效大规模数据训练与复杂任务执行。**

- **链接: [https://arxiv.org/pdf/2512.11047v1](https://arxiv.org/pdf/2512.11047v1)**

> **作者:** Haoran Jiang; Jin Chen; Qingwen Bu; Li Chen; Modi Shi; Yanjie Zhang; Delong Li; Chuanzhe Suo; Chuang Wang; Zhihui Peng; Hongyang Li
>
> **摘要:** Humanoid robots require precise locomotion and dexterous manipulation to perform challenging loco-manipulation tasks. Yet existing approaches, modular or end-to-end, are deficient in manipulation-aware locomotion. This confines the robot to a limited workspace, preventing it from performing large-space loco-manipulation. We attribute this to: (1) the challenge of acquiring loco-manipulation knowledge due to the scarcity of humanoid teleoperation data, and (2) the difficulty of faithfully and reliably executing locomotion commands, stemming from the limited precision and stability of existing RL controllers. To acquire richer loco-manipulation knowledge, we propose a unified latent learning framework that enables Vision-Language-Action (VLA) system to learn from low-cost action-free egocentric videos. Moreover, an efficient human data collection pipeline is devised to augment the dataset and scale the benefits. To more precisely execute the desired locomotion commands, we present a loco-manipulation-oriented (LMO) RL policy specifically tailored for accurate and stable core loco-manipulation movements, such as advancing, turning, and squatting. Building on these components, we introduce WholeBodyVLA, a unified framework for humanoid loco-manipulation. To the best of our knowledge, WholeBodyVLA is one of its kind enabling large-space humanoid loco-manipulation. It is verified via comprehensive experiments on the AgiBot X2 humanoid, outperforming prior baseline by 21.3%. It also demonstrates strong generalization and high extensibility across a broad range of tasks.
>
---
#### [new 100] Stochastics of shapes and Kunita flows
- **分类: math.PR; cs.CV**

- **简介: 该论文研究形状演化随机过程的数学构造，旨在解决非线性无限维形状空间中随机建模的难题。通过引入Kunita流生成符合形状结构特性的随机过程，并结合桥接采样实现基于观测数据的统计推断。**

- **链接: [https://arxiv.org/pdf/2512.11676v1](https://arxiv.org/pdf/2512.11676v1)**

> **作者:** Stefan Sommer; Gefan Yang; Elizabeth Louise Baker
>
> **摘要:** Stochastic processes of evolving shapes are used in applications including evolutionary biology, where morphology changes stochastically as a function of evolutionary processes. Due to the non-linear and often infinite-dimensional nature of shape spaces, the mathematical construction of suitable stochastic shape processes is far from immediate. We define and formalize properties that stochastic shape processes should ideally satisfy to be compatible with the shape structure, and we link this to Kunita flows that, when acting on shape spaces, induce stochastic processes that satisfy these criteria by their construction. We couple this with a survey of other relevant shape stochastic processes and show how bridge sampling techniques can be used to condition shape stochastic processes on observed data thereby allowing for statistical inference of parameters of the stochastic dynamics.
>
---
#### [new 101] Seeing to Act, Prompting to Specify: A Bayesian Factorization of Vision Language Action Policy
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉-语言-动作（VLA）模型的泛化问题，解决因模态不平衡导致的语言遗忘。提出BayesVLA，通过贝叶斯分解将策略分为视觉-动作先验和语言条件似然，提升指令跟随与跨场景泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.11218v1](https://arxiv.org/pdf/2512.11218v1)**

> **作者:** Kechun Xu; Zhenjie Zhu; Anzhe Chen; Shuqi Zhao; Qing Huang; Yifei Yang; Haojian Lu; Rong Xiong; Masayoshi Tomizuka; Yue Wang
>
> **摘要:** The pursuit of out-of-distribution generalization in Vision-Language-Action (VLA) models is often hindered by catastrophic forgetting of the Vision-Language Model (VLM) backbone during fine-tuning. While co-training with external reasoning data helps, it requires experienced tuning and data-related overhead. Beyond such external dependencies, we identify an intrinsic cause within VLA datasets: modality imbalance, where language diversity is much lower than visual and action diversity. This imbalance biases the model toward visual shortcuts and language forgetting. To address this, we introduce BayesVLA, a Bayesian factorization that decomposes the policy into a visual-action prior, supporting seeing-to-act, and a language-conditioned likelihood, enabling prompt-to-specify. This inherently preserves generalization and promotes instruction following. We further incorporate pre- and post-contact phases to better leverage pre-trained foundation models. Information-theoretic analysis formally validates our effectiveness in mitigating shortcut learning. Extensive experiments show superior generalization to unseen instructions, objects, and environments compared to existing methods. Project page is available at: https://xukechun.github.io/papers/BayesVLA.
>
---
#### [new 102] Particle Image Velocimetry Refinement via Consensus ADMM
- **分类: physics.flu-dyn; cs.CV; eess.IV; math.OC**

- **简介: 该论文属流场测速任务，旨在提升粒子图像测速（PIV）精度。针对传统方法依赖参数调优、机器学习泛化差的问题，提出基于共识ADMM的并行多算法融合框架，结合物理先验，显著降低误差，并支持高效部署与比较。**

- **链接: [https://arxiv.org/pdf/2512.11695v1](https://arxiv.org/pdf/2512.11695v1)**

> **作者:** Alan Bonomi; Francesco Banelli; Antonio Terpin
>
> **备注:** Code: https://github.com/antonioterpin/flowgym
>
> **摘要:** Particle Image Velocimetry (PIV) is an imaging technique in experimental fluid dynamics that quantifies flow fields around bluff bodies by analyzing the displacement of neutrally buoyant tracer particles immersed in the fluid. Traditional PIV approaches typically depend on tuning parameters specific to the imaging setup, making the performance sensitive to variations in illumination, flow conditions, and seeding density. On the other hand, even state-of-the-art machine learning methods for flow quantification are fragile outside their training set. In our experiments, we observed that flow quantification would improve if different tunings (or algorithms) were applied to different regions of the same image pair. In this work, we parallelize the instantaneous flow quantification with multiple algorithms and adopt a consensus framework based on the alternating direction method of multipliers, seamlessly incorporating priors such as smoothness and incompressibility. We perform several numerical experiments to demonstrate the benefits of this approach. For instance, we achieve a decrease in end-point-error of up to 20% of a dense-inverse-search estimator at an inference rate of 60Hz, and we show how this performance boost can be increased further with outlier rejection. Our method is implemented in JAX, effectively exploiting hardware acceleration, and integrated in Flow Gym, enabling (i) reproducible comparisons with the state-of-the-art, (ii) testing different base algorithms, (iii) straightforward deployment for active fluids control applications.
>
---
#### [new 103] Parallax: Runtime Parallelization for Operator Fallbacks in Heterogeneous Edge Systems
- **分类: cs.DC; cs.AI; cs.CV**

- **简介: 该论文针对边缘设备上DNN推理中算子回退导致的CPU利用率低、延迟高等问题，提出Parallax框架。通过计算图分割、内存优化与自适应调度，在不修改模型的前提下实现并行化加速，显著降低延迟与能耗。**

- **链接: [https://arxiv.org/pdf/2512.11532v1](https://arxiv.org/pdf/2512.11532v1)**

> **作者:** Chong Tang; Hao Dai; Jagmohan Chauhan
>
> **摘要:** The growing demand for real-time DNN applications on edge devices necessitates faster inference of increasingly complex models. Although many devices include specialized accelerators (e.g., mobile GPUs), dynamic control-flow operators and unsupported kernels often fall back to CPU execution. Existing frameworks handle these fallbacks poorly, leaving CPU cores idle and causing high latency and memory spikes. We introduce Parallax, a framework that accelerates mobile DNN inference without model refactoring or custom operator implementations. Parallax first partitions the computation DAG to expose parallelism, then employs branch-aware memory management with dedicated arenas and buffer reuse to reduce runtime footprint. An adaptive scheduler executes branches according to device memory constraints, meanwhile, fine-grained subgraph control enables heterogeneous inference of dynamic models. By evaluating on five representative DNNs across three different mobile devices, Parallax achieves up to 46% latency reduction, maintains controlled memory overhead (26.5% on average), and delivers up to 30% energy savings compared with state-of-the-art frameworks, offering improvements aligned with the responsiveness demands of real-time mobile inference.
>
---
#### [new 104] Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model
- **分类: cs.LG; cs.CV; q-bio.NC**

- **简介: 该论文属于脑成像表征学习任务，旨在解决fMRI数据噪声大、现有模型泛化差的问题。提出Brain-Semantoks框架，通过语义分词器和自蒸馏目标学习稳定的脑功能网络抽象表征，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2512.11582v1](https://arxiv.org/pdf/2512.11582v1)**

> **作者:** Sam Gijsen; Marc-Andre Schulz; Kerstin Ritter
>
> **备注:** Code and pretrained models available at https://github.com/SamGijsen/Brain-Semantoks
>
> **摘要:** The development of foundation models for functional magnetic resonance imaging (fMRI) time series holds significant promise for predicting phenotypes related to disease and cognition. Current models, however, are often trained using a mask-and-reconstruct objective on small brain regions. This focus on low-level information leads to representations that are sensitive to noise and temporal fluctuations, necessitating extensive fine-tuning for downstream tasks. We introduce Brain-Semantoks, a self-supervised framework designed specifically to learn abstract representations of brain dynamics. Its architecture is built on two core innovations: a semantic tokenizer that aggregates noisy regional signals into robust tokens representing functional networks, and a self-distillation objective that enforces representational stability across time. We show that this objective is stabilized through a novel training curriculum, ensuring the model robustly learns meaningful features from low signal-to-noise time series. We demonstrate that learned representations enable strong performance on a variety of downstream tasks even when only using a linear probe. Furthermore, we provide comprehensive scaling analyses indicating more unlabeled data reliably results in out-of-distribution performance gains without domain adaptation.
>
---
#### [new 105] mViSE: A Visual Search Engine for Analyzing Multiplex IHC Brain Tissue Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出mViSE，一种无需编程的视觉搜索引擎，用于分析多重免疫组化脑组织图像。旨在解决全片多通道图像分析难题，通过分而治之策略与自监督学习，实现细胞、微环境及脑区结构的检索与探索，已作为开源QuPath插件提供。**

- **链接: [https://arxiv.org/pdf/2512.11745v1](https://arxiv.org/pdf/2512.11745v1)**

> **作者:** Liqiang Huang; Rachel W. Mills; Saikiran Mandula; Lin Bai; Mahtab Jeyhani; John Redell; Hien Van Nguyen; Saurabh Prasad; Dragan Maric; Badrinath Roysam
>
> **摘要:** Whole-slide multiplex imaging of brain tissue generates massive information-dense images that are challenging to analyze and require custom software. We present an alternative query-driven programming-free strategy using a multiplex visual search engine (mViSE) that learns the multifaceted brain tissue chemoarchitecture, cytoarchitecture, and myeloarchitecture. Our divide-and-conquer strategy organizes the data into panels of related molecular markers and uses self-supervised learning to train a multiplex encoder for each panel with explicit visual confirmation of successful learning. Multiple panels can be combined to process visual queries for retrieving similar communities of individual cells or multicellular niches using information-theoretic methods. The retrievals can be used for diverse purposes including tissue exploration, delineating brain regions and cortical cell layers, profiling and comparing brain regions without computer programming. We validated mViSE's ability to retrieve single cells, proximal cell pairs, tissue patches, delineate cortical layers, brain regions and sub-regions. mViSE is provided as an open-source QuPath plug-in.
>
---
## 更新

#### [replaced 001] SpecDETR: A transformer-based hyperspectral point object detection network
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.10148v4](https://arxiv.org/pdf/2405.10148v4)**

> **作者:** Zhaoxu Li; Wei An; Gaowei Guo; Longguang Wang; Yingqian Wang; Zaiping Lin
>
> **摘要:** Hyperspectral target detection (HTD) aims to identify specific materials based on spectral information in hyperspectral imagery and can detect extremely small-sized objects, some of which occupy a smaller than one-pixel area. However, existing HTD methods are developed based on per-pixel binary classification, neglecting the three-dimensional cube structure of hyperspectral images (HSIs) that integrates both spatial and spectral dimensions. The synergistic existence of spatial and spectral features in HSIs enable objects to simultaneously exhibit both, yet the per-pixel HTD framework limits the joint expression of these features. In this paper, we rethink HTD from the perspective of spatial-spectral synergistic representation and propose hyperspectral point object detection as an innovative task framework. We introduce SpecDETR, the first specialized network for hyperspectral multi-class point object detection, which eliminates dependence on pre-trained backbone networks commonly required by vision-based object detectors. SpecDETR uses a multi-layer Transformer encoder with self-excited subpixel-scale attention modules to directly extract deep spatial-spectral joint features from hyperspectral cubes. We develop a simulated hyperspectral point object detection benchmark termed SPOD, and for the first time, evaluate and compare the performance of visual object detection networks and HTD methods on hyperspectral point object detection. Extensive experiments demonstrate that our proposed SpecDETR outperforms SOTA visual object detection networks and HTD methods. Our code and dataset are available at https://github.com/ZhaoxuLi123/SpecDETR.
>
---
#### [replaced 002] MetaVoxel: Joint Diffusion Modeling of Imaging and Clinical Metadata
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.10041v2](https://arxiv.org/pdf/2512.10041v2)**

> **作者:** Yihao Liu; Chenyu Gao; Lianrui Zuo; Michael E. Kim; Brian D. Boyd; Lisa L. Barnes; Walter A. Kukull; Lori L. Beason-Held; Susan M. Resnick; Timothy J. Hohman; Warren D. Taylor; Bennett A. Landman
>
> **摘要:** Modern deep learning methods have achieved impressive results across tasks from disease classification, estimating continuous biomarkers, to generating realistic medical images. Most of these approaches are trained to model conditional distributions defined by a specific predictive direction with a specific set of input variables. We introduce MetaVoxel, a generative joint diffusion modeling framework that models the joint distribution over imaging data and clinical metadata by learning a single diffusion process spanning all variables. By capturing the joint distribution, MetaVoxel unifies tasks that traditionally require separate conditional models and supports flexible zero-shot inference using arbitrary subsets of inputs without task-specific retraining. Using more than 10,000 T1-weighted MRI scans paired with clinical metadata from nine datasets, we show that a single MetaVoxel model can perform image generation, age estimation, and sex prediction, achieving performance comparable to established task-specific baselines. Additional experiments highlight its capabilities for flexible inference. Together, these findings demonstrate that joint multimodal diffusion offers a promising direction for unifying medical AI models and enabling broader clinical applicability.
>
---
#### [replaced 003] SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08710v3](https://arxiv.org/pdf/2506.08710v3)**

> **作者:** Mengjiao Ma; Qi Ma; Yue Li; Jiahuan Cheng; Runyi Yang; Bin Ren; Nikola Popovic; Mingqiang Wei; Nicu Sebe; Luc Van Gool; Theo Gevers; Martin R. Oswald; Danda Pani Paudel
>
> **备注:** 15 pages, codes, data and benchmark are released
>
> **摘要:** 3D Gaussian Splatting (3DGS) serves as a highly performant and efficient encoding of scene geometry, appearance, and semantics. Moreover, grounding language in 3D scenes has proven to be an effective strategy for 3D scene understanding. Current Language Gaussian Splatting line of work fall into three main groups: (i) per-scene optimization-based, (ii) per-scene optimization-free, and (iii) generalizable approach. However, most of them are evaluated only on rendered 2D views of a handful of scenes and viewpoints close to the training views, limiting ability and insight into holistic 3D understanding. To address this gap, we propose the first large-scale benchmark that systematically assesses these three groups of methods directly in 3D space, evaluating on 1060 scenes across three indoor datasets and one outdoor dataset. Benchmark results demonstrate a clear advantage of the generalizable paradigm, particularly in relaxing the scene-specific limitation, enabling fast feed-forward inference on novel scenes, and achieving superior segmentation performance. We further introduce GaussianWorld-49K a carefully curated 3DGS dataset comprising around 49K diverse indoor and outdoor scenes obtained from multiple sources, with which we demonstrate the generalizable approach could harness strong data priors. Our codes, benchmark, and datasets are released at https://scenesplatpp.gaussianworld.ai/.
>
---
#### [replaced 004] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.11640v3](https://arxiv.org/pdf/2505.11640v3)**

> **作者:** Pandula Thennakoon; Avishka Ranasinghe; Mario De Silva; Buwaneka Epakanda; Roshan Godaliyadda; Parakrama Ekanayake; Vijitha Herath
>
> **备注:** Submitted as a conference paper to ICLR 2026
>
> **摘要:** Implicit neural representations (INRs) are a powerful paradigm for modeling data, offering a continuous alternative to discrete signal representations. Their ability to compactly encode complex signals has led to strong performance in many vision tasks. Prior work shows INR performance is highly sensitive to the choice of activation function in the underlying multilayer perceptron, yet the theoretical reasons remain unclear. Key limitations also persist, including spectral bias (reduced sensitivity to high-frequency content), limited robustness to noise, and difficulty capturing local and global structure jointly. We analyze INR signal representation using harmonic analysis and Chebyshev polynomials. We prove that modulating activation functions with a complex sinusoidal term yields richer and more complete spectral support throughout the network. Building on this, we introduce a new activation function tailored to INRs and validate our theory using Chebyshev analysis and extensive experiments. We additionally use a regularized deep prior, extracted from a task-specific model, to adapt the activations, further improving convergence speed and stability. Across image reconstruction (average PSNR gain of +5.67 dB over the nearest counterpart on a diverse dataset), denoising (+0.46 dB PSNR), super-resolution (+0.64 dB over the nearest SOTA method for 6X upscaling), inpainting, and 3D shape reconstruction, our activation consistently outperforms existing state-of-the-art alternatives.
>
---
#### [replaced 005] InterAgent: Physics-based Multi-agent Command Execution via Diffusion on Interaction Graphs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07410v2](https://arxiv.org/pdf/2512.07410v2)**

> **作者:** Bin Li; Ruichi Zhang; Han Liang; Jingyan Zhang; Juze Zhang; Xin Chen; Lan Xu; Jingyi Yu; Jingya Wang
>
> **备注:** Project page: https://binlee26.github.io/InterAgent-Page
>
> **摘要:** Humanoid agents are expected to emulate the complex coordination inherent in human social behaviors. However, existing methods are largely confined to single-agent scenarios, overlooking the physically plausible interplay essential for multi-agent interactions. To bridge this gap, we propose InterAgent, the first end-to-end framework for text-driven physics-based multi-agent humanoid control. At its core, we introduce an autoregressive diffusion transformer equipped with multi-stream blocks, which decouples proprioception, exteroception, and action to mitigate cross-modal interference while enabling synergistic coordination. We further propose a novel interaction graph exteroception representation that explicitly captures fine-grained joint-to-joint spatial dependencies to facilitate network learning. Additionally, within it we devise a sparse edge-based attention mechanism that dynamically prunes redundant connections and emphasizes critical inter-agent spatial relations, thereby enhancing the robustness of interaction modeling. Extensive experiments demonstrate that InterAgent consistently outperforms multiple strong baselines, achieving state-of-the-art performance. It enables producing coherent, physically plausible, and semantically faithful multi-agent behaviors from only text prompts. Our code and data will be released to facilitate future research.
>
---
#### [replaced 006] Defense That Attacks: How Robust Models Become Better Attackers
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.02830v3](https://arxiv.org/pdf/2512.02830v3)**

> **作者:** Mohamed Awad; Mahmoud Akrm; Walid Gomaa
>
> **摘要:** Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples.
>
---
#### [replaced 007] From Macro to Micro: Benchmarking Microscopic Spatial Intelligence on Molecules via Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10867v2](https://arxiv.org/pdf/2512.10867v2)**

> **作者:** Zongzhao Li; Xiangzhe Kong; Jiahui Su; Zongyang Ma; Mingze Li; Songyou Li; Yuelin Zhang; Yu Rong; Tingyang Xu; Deli Zhao; Wenbing Huang
>
> **摘要:** This paper introduces the concept of Microscopic Spatial Intelligence (MiSI), the capability to perceive and reason about the spatial relationships of invisible microscopic entities, which is fundamental to scientific discovery. To assess the potential of Vision-Language Models (VLMs) in this domain, we propose a systematic benchmark framework MiSI-Bench. This framework features over 163,000 question-answer pairs and 587,000 images derived from approximately 4,000 molecular structures, covering nine complementary tasks that evaluate abilities ranging from elementary spatial transformations to complex relational identifications. Experimental results reveal that current state-of-the-art VLMs perform significantly below human level on this benchmark. However, a fine-tuned 7B model demonstrates substantial potential, even surpassing humans in spatial transformation tasks, while its poor performance in scientifically-grounded tasks like hydrogen bond recognition underscores the necessity of integrating explicit domain knowledge for progress toward scientific AGI. The datasets are available at https://huggingface.co/datasets/zongzhao/MiSI-bench.
>
---
#### [replaced 008] Free-Lunch Color-Texture Disentanglement for Stylized Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.14275v4](https://arxiv.org/pdf/2503.14275v4)**

> **作者:** Jiang Qin; Senmao Li; Alexandra Gomez-Villa; Shiqi Yang; Yaxing Wang; Kai Wang; Joost van de Weijer
>
> **备注:** Accepted by NeurIPS2025. Code is available at https://deepffff.github.io/sadis.github.io/
>
> **摘要:** Recent advances in Text-to-Image (T2I) diffusion models have transformed image generation, enabling significant progress in stylized generation using only a few style reference images. However, current diffusion-based methods struggle with fine-grained style customization due to challenges in controlling multiple style attributes, such as color and texture. This paper introduces the first tuning-free approach to achieve free-lunch color-texture disentanglement in stylized T2I generation, addressing the need for independently controlled style elements for the Disentangled Stylized Image Generation (DisIG) problem. Our approach leverages the Image-Prompt Additivity property in the CLIP image embedding space to develop techniques for separating and extracting Color-Texture Embeddings (CTE) from individual color and texture reference images. To ensure that the color palette of the generated image aligns closely with the color reference, we apply a whitening and coloring transformation to enhance color consistency. Additionally, to prevent texture loss due to the signal-leak bias inherent in diffusion training, we introduce a noise term that preserves textural fidelity during the Regularized Whitening and Coloring Transformation (RegWCT). Through these methods, our Style Attributes Disentanglement approach (SADis) delivers a more precise and customizable solution for stylized image generation. Experiments on images from the WikiArt and StyleDrop datasets demonstrate that, both qualitatively and quantitatively, SADis surpasses state-of-the-art stylization methods in the DisIG task.Code is released at https://deepffff.github.io/sadis.github.io/.
>
---
#### [replaced 009] Counterfactual Segmentation Reasoning: Diagnosing and Mitigating Pixel-Grounding Hallucination
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究分割视觉语言模型的像素错觉问题，提出反事实分割推理任务及新基准HalluSegBench，诊断并缓解视觉驱动的幻觉。通过反事实微调训练RobustSeg模型，在减少30%幻觉的同时提升分割性能。**

- **链接: [https://arxiv.org/pdf/2506.21546v3](https://arxiv.org/pdf/2506.21546v3)**

> **作者:** Xinzhuo Li; Adheesh Juvekar; Jiaxun Zhang; Xingyou Liu; Muntasir Wahed; Kiet A. Nguyen; Yifan Shen; Tianjiao Yu; Ismini Lourentzou
>
> **备注:** Project webpage: https://plan-lab.github.io/hallusegbench/
>
> **摘要:** Segmentation Vision-Language Models (VLMs) have significantly advanced grounded visual understanding, yet they remain prone to pixel-grounding hallucinations, producing masks for incorrect objects or for objects that are entirely absent. Existing evaluations rely almost entirely on text- or label-based perturbations, which check only whether the predicted mask matches the queried label. Such evaluations overlook the spatial footprint and severity of hallucination and therefore fail to reveal vision-driven hallucinations, which are more challenging and more prevalent. To address this gap, we formalize the task of Counterfactual Segmentation Reasoning (CSR), where a model must segment the referenced object in the factual image and abstain in its counterfactual counterpart. To support this task, we curate HalluSegBench, the first large-scale benchmark to diagnose referring and reasoning expression segmentation hallucinations using controlled visual counterfactuals, alongside new evaluation metrics that measure hallucination severity and disentangle vision- and language-driven failure modes. We further introduce RobustSeg, a segmentation VLM trained with counterfactual fine-tuning (CFT) to learn when to segment and when to abstain. Experimental results confirm RobustSeg reduces hallucinations by 30%, while improving segmentation performance on FP-RefCOCO(+/g).
>
---
#### [replaced 010] Beyond Endpoints: Path-Centric Reasoning for Vectorized Off-Road Network Extraction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.10416v2](https://arxiv.org/pdf/2512.10416v2)**

> **作者:** Wenfei Guan; Jilin Mei; Tong Shen; Xumin Wu; Shuo Wang; Cheng Min; Yu Hu
>
> **备注:** v2: Corrected the abstract to accurately reflect the paper content. Updated the project link to the correct repository<a href="https://github.com/xiaofei-guan/MaGRoad" target="_blank" rel="noopener noreferrer nofollow"></a>. No changes to the main text
>
> **摘要:** Deep learning has advanced vectorized road extraction in urban settings, yet off-road environments remain underexplored and challenging. A significant domain gap causes advanced models to fail in wild terrains due to two key issues: lack of large-scale vectorized datasets and structural weakness in prevailing methods. Models such as SAM-Road employ a node-centric paradigm that reasons at sparse endpoints, making them fragile to occlusions and ambiguous junctions in off-road scenes, leading to topological errors. This work addresses these limitations in two complementary ways. First, we release WildRoad, a global off-road road network dataset constructed efficiently with a dedicated interactive annotation tool tailored for road-network labeling. Second, we introduce MaGRoad (Mask-aware Geodesic Road network extractor), a path-centric framework that aggregates multi-scale visual evidence along candidate paths to infer connectivity robustly. Extensive experiments show that MaGRoad achieves state-of-the-art performance on our challenging WildRoad benchmark while generalizing well to urban datasets. A streamlined pipeline also yields roughly 2.5x faster inference, improving practical applicability. Together, the dataset and path-centric paradigm provide a stronger foundation for mapping roads in the wild. We release both the dataset and code at https://github.com/xiaofei-guan/MaGRoad.
>
---
#### [replaced 011] Denoising Diffusion Models for Anomaly Localization in Medical Images
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2410.23834v2](https://arxiv.org/pdf/2410.23834v2)**

> **作者:** Cosmin I. Bercea; Philippe C. Cattin; Julia A. Schnabel; Julia Wolleb
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) https://melba-journal.org/2025:030
>
> **摘要:** This review explores anomaly localization in medical images using denoising diffusion models. After providing a brief methodological background of these models, including their application to image reconstruction and their conditioning using guidance mechanisms, we provide an overview of available datasets and evaluation metrics suitable for their application to anomaly localization in medical images. In this context, we discuss supervision schemes ranging from fully supervised segmentation to semi-supervised, weakly supervised, self-supervised, and unsupervised methods, and provide insights into the effectiveness and limitations of these approaches. Furthermore, we highlight open challenges in anomaly localization, including detection bias, domain shift, computational cost, and model interpretability. Our goal is to provide an overview of the current state of the art in the field, outline research gaps, and highlight the potential of diffusion models for robust anomaly localization in medical images.
>
---
#### [replaced 012] MMAP: A Multi-Magnification and Prototype-Aware Architecture for Predicting Spatial Gene Expression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.11344v2](https://arxiv.org/pdf/2510.11344v2)**

> **作者:** Hai Dang Nguyen; Nguyen Dang Huy Pham; The Minh Duc Nguyen; Dac Thai Nguyen; Hang Thi Nguyen; Duong M. Nguyen
>
> **备注:** Received Best Paper Award at the 2025 Pacific Rim International Conference on Artificial Intelligence (PRICAI 2025)
>
> **摘要:** Spatial Transcriptomics (ST) enables the measurement of gene expression while preserving spatial information, offering critical insights into tissue architecture and disease pathology. Recent developments have explored the use of hematoxylin and eosin (H&E)-stained whole-slide images (WSIs) to predict transcriptome-wide gene expression profiles through deep neural networks. This task is commonly framed as a regression problem, where each input corresponds to a localized image patch extracted from the WSI. However, predicting spatial gene expression from histological images remains a challenging problem due to the significant modality gap between visual features and molecular signals. Recent studies have attempted to incorporate both local and global information into predictive models. Nevertheless, existing methods still suffer from two key limitations: (1) insufficient granularity in local feature extraction, and (2) inadequate coverage of global spatial context. In this work, we propose a novel framework, MMAP (Multi-MAgnification and Prototype-enhanced architecture), that addresses both challenges simultaneously. To enhance local feature granularity, MMAP leverages multi-magnification patch representations that capture fine-grained histological details. To improve global contextual understanding, it learns a set of latent prototype embeddings that serve as compact representations of slide-level information. Extensive experimental results demonstrate that MMAP consistently outperforms all existing state-of-the-art methods across multiple evaluation metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), and Pearson Correlation Coefficient (PCC).
>
---
#### [replaced 013] Spec-Gloss Surfels and Normal-Diffuse Priors for Relightable Glossy Objects
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.02069v3](https://arxiv.org/pdf/2510.02069v3)**

> **作者:** Georgios Kouros; Minye Wu; Tinne Tuytelaars
>
> **摘要:** Accurate reconstruction and relighting of glossy objects remains a longstanding challenge, as object shape, material properties, and illumination are inherently difficult to disentangle. Existing neural rendering approaches often rely on simplified BRDF models or parameterizations that couple diffuse and specular components, which restrict faithful material recovery and limit relighting fidelity. We propose a relightable framework that integrates a microfacet BRDF with the specular-glossiness parameterization into 2D Gaussian Splatting with deferred shading. This formulation enables more physically consistent material decomposition, while diffusion-based priors for surface normals and diffuse color guide early-stage optimization and mitigate ambiguity. A coarse-to-fine environment map optimization accelerates convergence, and negative-only environment map clipping preserves high-dynamic-range specular reflections. Extensive experiments on complex, glossy scenes demonstrate that our method achieves high-quality geometry and material reconstruction, delivering substantially more realistic and consistent relighting under novel illumination compared to existing Gaussian splatting methods.
>
---
#### [replaced 014] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.17982v5](https://arxiv.org/pdf/2505.17982v5)**

> **作者:** Bryan Wong; Jong Woo Kim; Huazhu Fu; Mun Yong Yi
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Vision-language models (VLMs) have recently been integrated into multiple instance learning (MIL) frameworks to address the challenge of few-shot, weakly supervised classification of whole slide images (WSIs). A key trend involves leveraging multi-scale information to better represent hierarchical tissue structures. However, existing methods often face two key limitations: (1) insufficient modeling of interactions within the same modalities across scales (e.g., 5x and 20x) and (2) inadequate alignment between visual and textual modalities on the same scale. To address these gaps, we propose HiVE-MIL, a hierarchical vision-language framework that constructs a unified graph consisting of (1) parent-child links between coarse (5x) and fine (20x) visual/textual nodes to capture hierarchical relationships, and (2) heterogeneous intra-scale edges linking visual and textual nodes on the same scale. To further enhance semantic consistency, HiVE-MIL incorporates a two-stage, text-guided dynamic filtering mechanism that removes weakly correlated patch-text pairs, and introduces a hierarchical contrastive loss to align textual semantics across scales. Extensive experiments on TCGA breast, lung, and kidney cancer datasets demonstrate that HiVE-MIL consistently outperforms both traditional MIL and recent VLM-based MIL approaches, achieving gains of up to 4.1% in macro F1 under 16-shot settings. Our results demonstrate the value of jointly modeling hierarchical structure and multimodal alignment for efficient and scalable learning from limited pathology data. The code is available at https://github.com/bryanwong17/HiVE-MIL.
>
---
#### [replaced 015] Estimating Object Physical Properties from RGB-D Vision and Depth Robot Sensors Using Deep Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.05029v2](https://arxiv.org/pdf/2507.05029v2)**

> **作者:** Ricardo Cardoso; Plinio Moreno
>
> **摘要:** Inertial mass plays a crucial role in robotic applications such as object grasping, manipulation, and simulation, providing a strong prior for planning and control. Accurately estimating an object's mass before interaction can significantly enhance the performance of various robotic tasks. However, mass estimation using only vision sensors is a relatively underexplored area. This paper proposes a novel approach combining sparse point-cloud data from depth images with RGB images to estimate the mass of objects. We evaluate a range of point-cloud processing architectures, alongside RGB-only methods. To overcome the limited availability of training data, we create a synthetic dataset using ShapeNetSem 3D models, simulating RGBD images via a Kinect camera. This synthetic data is used to train an image generation model for estimating dense depth maps, which we then use to augment an existing dataset of images paired with mass values. Our approach significantly outperforms existing benchmarks across all evaluated metrics. The data generation (https://github.com/RavineWindteer/ShapenetSem-to-RGBD) as well as the training of the depth estimator (https://github.com/RavineWindteer/GLPDepth-Edited) and the mass estimator (https://github.com/RavineWindteer/Depth-mass-estimator) are available online.
>
---
#### [replaced 016] Fine-grained Defocus Blur Control for Generative Image Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.06215v2](https://arxiv.org/pdf/2510.06215v2)**

> **作者:** Ayush Shrivastava; Connelly Barnes; Xuaner Zhang; Lingzhi Zhang; Andrew Owens; Sohrab Amirghodsi; Eli Shechtman
>
> **备注:** Project link: https://www.ayshrv.com/defocus-blur-gen
>
> **摘要:** Current text-to-image diffusion models excel at generating diverse, high-quality images, yet they struggle to incorporate fine-grained camera metadata such as precise aperture settings. In this work, we introduce a novel text-to-image diffusion framework that leverages camera metadata, or EXIF data, which is often embedded in image files, with an emphasis on generating controllable lens blur. Our method mimics the physical image formation process by first generating an all-in-focus image, estimating its monocular depth, predicting a plausible focus distance with a novel focus distance transformer, and then forming a defocused image with an existing differentiable lens blur model. Gradients flow backwards through this whole process, allowing us to learn without explicit supervision to generate defocus effects based on content elements and the provided EXIF data. At inference time, this enables precise interactive user control over defocus effects while preserving scene contents, which is not achievable with existing diffusion models. Experimental results demonstrate that our model enables superior fine-grained control without altering the depicted scene.
>
---
#### [replaced 017] SOF: Sorted Opacity Fields for Fast Unbounded Surface Reconstruction
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.19139v2](https://arxiv.org/pdf/2506.19139v2)**

> **作者:** Lukas Radl; Felix Windisch; Thomas Deixelberger; Jozef Hladky; Michael Steiner; Dieter Schmalstieg; Markus Steinberger
>
> **备注:** SIGGRAPH Asia 2025; Project Page: https://r4dl.github.io/SOF/
>
> **摘要:** Recent advances in 3D Gaussian representations have significantly improved the quality and efficiency of image-based scene reconstruction. Their explicit nature facilitates real-time rendering and fast optimization, yet extracting accurate surfaces - particularly in large-scale, unbounded environments - remains a difficult task. Many existing methods rely on approximate depth estimates and global sorting heuristics, which can introduce artifacts and limit the fidelity of the reconstructed mesh. In this paper, we present Sorted Opacity Fields (SOF), a method designed to recover detailed surfaces from 3D Gaussians with both speed and precision. Our approach improves upon prior work by introducing hierarchical resorting and a robust formulation of Gaussian depth, which better aligns with the level-set. To enhance mesh quality, we incorporate a level-set regularizer operating on the opacity field and introduce losses that encourage geometrically-consistent primitive shapes. In addition, we develop a parallelized Marching Tetrahedra algorithm tailored to our opacity formulation, reducing meshing time by up to an order of magnitude. As demonstrated by our quantitative evaluation, SOF achieves higher reconstruction accuracy while cutting total processing time by more than a factor of three. These results mark a step forward in turning efficient Gaussian-based rendering into equally efficient geometry extraction.
>
---
#### [replaced 018] Generalized Denoising Diffusion Codebook Models (gDDCM): Tokenizing images using a pre-trained diffusion model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.13387v3](https://arxiv.org/pdf/2511.13387v3)**

> **作者:** Fei Kong
>
> **备注:** in Chinese language
>
> **摘要:** Denoising diffusion models have emerged as a dominant paradigm in image generation. Discretizing image data into tokens is a critical step for effectively integrating images with Transformer and other architectures. Although the Denoising Diffusion Codebook Models (DDCM) pioneered the use of pre-trained diffusion models for image tokenization, it strictly relies on the traditional discrete-time DDPM architecture. Consequently, it fails to adapt to modern continuous-time variants-such as Flow Matching and Consistency Models-and suffers from inefficient sampling in high-noise regions. To address these limitations, this paper proposes the Generalized Denoising Diffusion Codebook Models (gDDCM). We establish a unified theoretical framework and introduce a generic "De-noise and Back-trace" sampling strategy. By integrating a deterministic ODE denoising step with a residual-aligned noise injection step, our method resolves the challenge of adaptation. Furthermore, we introduce a backtracking parameter $p$ and significantly enhance tokenization ability. Extensive experiments on CIFAR10 and LSUN Bedroom datasets demonstrate that gDDCM achieves comprehensive compatibility with mainstream diffusion variants and significantly outperforms DDCM in terms of reconstruction quality and perceptual fidelity.
>
---
#### [replaced 019] UStyle: Waterbody Style Transfer of Underwater Scenes by Depth-Guided Feature Synthesis
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2503.11893v3](https://arxiv.org/pdf/2503.11893v3)**

> **作者:** Md Abu Bakr Siddique; Vaishnav Ramesh; Junliang Liu; Piyush Singh; Md Jahidul Islam
>
> **摘要:** The concept of waterbody style transfer remains largely unexplored in the underwater imaging and vision literature. Traditional image style transfer (STx) methods primarily focus on artistic and photorealistic blending, often failing to preserve object and scene geometry in images captured in high-scattering mediums such as underwater. The wavelength-dependent nonlinear attenuation and depth-dependent backscattering artifacts further complicate learning underwater image STx from unpaired data. This paper introduces UStyle, the first data-driven learning framework for transferring waterbody styles across underwater images without requiring prior reference images or scene information. We propose a novel depth-aware whitening and coloring transform (DA-WCT) mechanism that integrates physics-based waterbody synthesis to ensure perceptually consistent stylization while preserving scene structure. To enhance style transfer quality, we incorporate carefully designed loss functions that guide UStyle to maintain colorfulness, lightness, structural integrity, and frequency-domain characteristics, as well as high-level content in VGG and CLIP (contrastive language-image pretraining) feature spaces. By addressing domain-specific challenges, UStyle provides a robust framework for no-reference underwater image STx, surpassing state-of-the-art (SOTA) methods that rely solely on end-to-end reconstruction loss. Furthermore, we introduce the UF7D dataset, a curated collection of high-resolution underwater images spanning seven distinct waterbody styles, establishing a benchmark to support future research in underwater image STx. The UStyle inference pipeline and UF7D dataset are released at: https://github.com/uf-robopi/UStyle.
>
---
#### [replaced 020] Two Datasets Are Better Than One: Method of Double Moments for 3-D Reconstruction in Cryo-EM
- **分类: cs.CV; math.NA; stat.ME**

- **链接: [https://arxiv.org/pdf/2511.07438v2](https://arxiv.org/pdf/2511.07438v2)**

> **作者:** Joe Kileel; Oscar Mickelin; Amit Singer; Sheng Xu
>
> **摘要:** Cryo-electron microscopy (cryo-EM) is a powerful imaging technique for reconstructing three-dimensional molecular structures from noisy tomographic projection images of randomly oriented particles. We introduce a new data fusion framework, termed the method of double moments (MoDM), which reconstructs molecular structures from two instances of the second-order moment of projection images obtained under distinct orientation distributions: one uniform, the other non-uniform and unknown. We prove that these moments generically uniquely determine the underlying structure, up to a global rotation and reflection, and we develop a convex-relaxation-based algorithm that achieves accurate recovery using only second-order statistics. Our results demonstrate the advantage of collecting and modeling multiple datasets under different experimental conditions, illustrating that leveraging dataset diversity can substantially enhance reconstruction quality in computational imaging tasks.
>
---
#### [replaced 021] Visual-Friendly Concept Protection via Selective Adversarial Perturbations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.08518v3](https://arxiv.org/pdf/2408.08518v3)**

> **作者:** Xiaoyue Mi; Fan Tang; You Wu; Juan Cao; Peng Li; Yang Liu
>
> **备注:** AAAI AISI 26
>
> **摘要:** Personalized concept generation by tuning diffusion models with a few images raises potential legal and ethical concerns regarding privacy and intellectual property rights. Researchers attempt to prevent malicious personalization using adversarial perturbations. However, previous efforts have mainly focused on the effectiveness of protection while neglecting the visibility of perturbations. They utilize global adversarial perturbations, which introduce noticeable alterations to original images and significantly degrade visual quality. In this work, we propose the Visual-Friendly Concept Protection (VCPro) framework, which prioritizes the protection of key concepts chosen by the image owner through adversarial perturbations with lower perceptibility. To ensure these perturbations are as inconspicuous as possible, we introduce a relaxed optimization objective to identify the least perceptible yet effective adversarial perturbations, solved using the Lagrangian multiplier method. Qualitative and quantitative experiments validate that VCPro achieves a better trade-off between the visibility of perturbations and protection effectiveness, effectively prioritizing the protection of target concepts in images with less perceptible perturbations.
>
---
#### [replaced 022] Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15351v2](https://arxiv.org/pdf/2511.15351v2)**

> **作者:** Yifu Guo; Zishan Xu; Zhiyuan Yao; Yuquan Lu; Jiaye Lin; Sen Hu; Zhenheng Tang; Huacan Wang; Ronghao Chen
>
> **摘要:** Existing multimodal reasoning models and frameworks suffer from fundamental architectural limitations: most lack the human-like ability to autonomously explore diverse reasoning pathways-whether in direct inference, tool-driven visual exploration, programmatic visual manipulation, or intrinsic visual imagination. Consequently, they struggle to adapt to dynamically changing capability requirements in real-world tasks. Meanwhile, humans exhibit a complementary set of thinking abilities when addressing such tasks, whereas existing methods typically cover only a subset of these dimensions. Inspired by this, we propose Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration, a new paradigm for multimodal agentic reasoning. We define six core capabilities essential for multimodal reasoning and organize a comprehensive evaluation benchmark, Octopus-Bench, accordingly. Octopus is capable of autonomously exploring during reasoning and dynamically selecting the most appropriate capability based on the current state. Experimental results show that Octopus achieves the best performance on the vast majority of tasks in Octopus-Bench, highlighting the crucial role of capability coordination in agentic multimodal reasoning.
>
---
#### [replaced 023] 3D-LATTE: Latent Space 3D Editing from Textual Instructions
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.00269v3](https://arxiv.org/pdf/2509.00269v3)**

> **作者:** Maria Parelli; Michael Oechsle; Michael Niemeyer; Federico Tombari; Andreas Geiger
>
> **摘要:** Despite the recent success of multi-view diffusion models for text/image-based 3D asset generation, instruction-based editing of 3D assets lacks surprisingly far behind the quality of generation models. The main reason is that recent approaches using 2D priors suffer from view-inconsistent editing signals. Going beyond 2D prior distillation methods and multi-view editing strategies, we propose a training-free editing method that operates within the latent space of a native 3D diffusion model, allowing us to directly manipulate 3D geometry. We guide the edit synthesis by blending 3D attention maps from the generation with the source object. Coupled with geometry-aware regularization guidance, a spectral modulation strategy in the Fourier domain and a refinement step for 3D enhancement, our method outperforms previous 3D editing methods enabling high-fidelity and precise edits across a wide range of shapes and semantic manipulations. Our project webpage is https://mparelli.github.io/3d-latte
>
---
#### [replaced 024] MoCA-Video: Motion-Aware Concept Alignment for Consistent Video Editing
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.01004v2](https://arxiv.org/pdf/2506.01004v2)**

> **作者:** Tong Zhang; Juan C Leon Alcazar; Victor Escorcia; Bernard Ghanem
>
> **摘要:** We present MoCA-Video, a training-free framework for semantic mixing in videos. Operating in the latent space of a frozen video diffusion model, MoCA-Video utilizes class-agnostic segmentation with diagonal denoising scheduler to localize and track the target object across frames. To ensure temporal stability under semantic shifts, we introduce momentum-based correction to approximate novel hybrid distributions beyond trained data distribution, alongside a light gamma residual module that smooths out visual artifacts. We evaluate model's performance using SSIM, LPIPS, and a proposed metric, \metricnameabbr, which quantifies semantic alignment between reference and output. Extensive evaluation demonstrates that our model consistently outperforms both training-free and trained baselines, achieving superior semantic mixing and temporal coherence without retraining. Results establish that structured manipulation of diffusion noise trajectories enables controllable and high-quality video editing under semantic shifts.
>
---
#### [replaced 025] Any2Caption:Interpreting Any Condition to Caption for Controllable Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.24379v2](https://arxiv.org/pdf/2503.24379v2)**

> **作者:** Shengqiong Wu; Weicai Ye; Jiahao Wang; Quande Liu; Xintao Wang; Pengfei Wan; Di Zhang; Kun Gai; Shuicheng Yan; Hao Fei; Tat-Seng Chua
>
> **备注:** Project Page: https://sqwu.top/Any2Cap/
>
> **摘要:** To address the bottleneck of accurate user intent interpretation within the current video generation community, we present Any2Caption, a novel framework for controllable video generation under any condition. The key idea is to decouple various condition interpretation steps from the video synthesis step. By leveraging modern multimodal large language models (MLLMs), Any2Caption interprets diverse inputs--text, images, videos, and specialized cues such as region, motion, and camera poses--into dense, structured captions that offer backbone video generators with better guidance. We also introduce Any2CapIns, a large-scale dataset with 337K instances and 407K conditions for any-condition-to-caption instruction tuning. Comprehensive evaluations demonstrate significant improvements of our system in controllability and video quality across various aspects of existing video generation models. Project Page: https://sqwu.top/Any2Cap/
>
---
#### [replaced 026] Time-Series at the Edge: Tiny Separable CNNs for Wearable Gait Detection and Optimal Sensor Placement
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.00396v2](https://arxiv.org/pdf/2512.00396v2)**

> **作者:** Andrea Procopio; Marco Esposito; Sara Raggiunto; Andrey Gizdov; Alberto Belli; Paola Pierleoni
>
> **摘要:** We study on-device time-series analysis for gait detection in Parkinson's disease (PD) from short windows of triaxial acceleration, targeting resource-constrained wearables and edge nodes. We compare magnitude thresholding to three 1D CNNs for time-series analysis: a literature baseline (separable convolutions) and two ultra-light models - one purely separable and one with residual connections. Using the BioStampRC21 dataset, 2 s windows at 30 Hz, and subject-independent leave-one-subject-out (LOSO) validation on 16 PwPD with chest-worn IMUs, our residual separable model (Model 2, 533 params) attains PR-AUC = 94.5%, F1 = 91.2%, MCC = 89.4%, matching or surpassing the baseline (5,552 params; PR-AUC = 93.7%, F1 = 90.5%, MCC = 88.5%) with approximately 10x fewer parameters. The smallest model (Model 1, 305 params) reaches PR-AUC = 94.0%, F1 = 91.0%, MCC = 89.1%. Thresholding obtains high recall (89.0%) but low precision (76.5%), yielding many false positives and high inter-subject variance. Sensor-position analysis (train-on-all) shows chest and thighs are most reliable; forearms degrade precision/recall due to non-gait arm motion; naive fusion of all sites does not outperform the best single site. Both compact CNNs execute within tight memory/latency budgets on STM32-class MCUs (sub-10 ms on low-power boards), enabling on-sensor gating of transmission/storage. Overall, ultra-light separable CNNs provide a superior accuracy-efficiency-generalization trade-off to fixed thresholds for wearable PD gait detection and underscore the value of tailored time-series models for edge deployment.
>
---
#### [replaced 027] An Efficient Test-Time Scaling Approach for Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08985v2](https://arxiv.org/pdf/2512.08985v2)**

> **作者:** Vignesh Sundaresha; Akash Haridas; Vikram Appia; Lav R. Varshney
>
> **备注:** 11 pages
>
> **摘要:** Image generation has emerged as a mainstream application of large generative AI models. Just as test-time compute and reasoning have helped language models improve their capabilities, similar benefits have also been observed with image generation models. In particular, searching over noise samples for diffusion and flow models has shown to scale well with test-time compute. While recent works have explored allocating non-uniform inference-compute budgets across different denoising steps, they rely on greedy algorithms and allocate the compute budget ineffectively. In this work, we study this problem and propose solutions to fix it. We propose the Verifier-Threshold method which automatically reallocates test-time compute and delivers substantial efficiency improvements. For the same performance on the GenEval benchmark, we achieve a 2-4x reduction in computational time over the state-of-the-art method.
>
---
#### [replaced 028] More Than Memory Savings: Zeroth-Order Optimization Mitigates Forgetting in Continual Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21019v2](https://arxiv.org/pdf/2510.21019v2)**

> **作者:** Wanhao Yu; Zheng Wang; Shuteng Niu; Sen Lin; Li Yang
>
> **摘要:** Zeroth-order (ZO) optimization has gained attention as a memory-efficient alternative to first-order (FO) methods, particularly in settings where gradient computation is expensive or even impractical. Beyond its memory efficiency, in this work, we investigate ZO optimization for continual learning (CL) as a novel approach to address the plasticity-stability-efficiency trilemma. Through theoretical analysis and empirical evidence, we show that ZO optimization naturally leads to flatter loss landscapes, which in turn reduce forgetting in CL. However, this stability comes at a cost of plasticity: due to its imprecise gradient estimates and slower convergence, ZO optimization tends to be less effective than FO in acquiring new task-specific knowledge, particularly under constrained training budgets. To better understand this trade-off, we conduct a holistic evaluation of ZO optimization applied to various existing CL methods. Our findings reveal that ZO optimization enhances stability but often undermines plasticity, particularly when used with learnable classifiers. Motivated by this insight, we propose ZO-FC, a simple but effective approach that applies ZO optimization to a single adapter-based PEFT module with FO optimized classifier. This design leverages the stability benefits of ZO while preserving the adaptability of FO updates with negligible memory overhead. Experiments demonstrate that ZO-FC achieves an effective balance between stability and plasticity, offering a practical and memory-efficient solution for on-device CL.
>
---
#### [replaced 029] Temporal In-Context Fine-Tuning with Temporal Reasoning for Versatile Control of Video Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.00996v2](https://arxiv.org/pdf/2506.00996v2)**

> **作者:** Kinam Kim; Junha Hyung; Jaegul Choo
>
> **备注:** project page: https://kinam0252.github.io/TIC-FT/
>
> **摘要:** Recent advances in text-to-video diffusion models have enabled high-quality video synthesis, but controllable generation remains challenging, particularly under limited data and compute. Existing fine-tuning methods for conditional generation often rely on external encoders or architectural modifications, which demand large datasets and are typically restricted to spatially aligned conditioning, limiting flexibility and scalability. In this work, we introduce Temporal In-Context Fine-Tuning (TIC-FT), an efficient and versatile approach for adapting pretrained video diffusion models to diverse conditional generation tasks. Our key idea is to concatenate condition and target frames along the temporal axis and insert intermediate buffer frames with progressively increasing noise levels. These buffer frames enable smooth transitions, aligning the fine-tuning process with the pretrained model's temporal dynamics. TIC-FT requires no architectural changes and achieves strong performance with as few as 10-30 training samples. We validate our method across a range of tasks, including image-to-video and video-to-video generation, using large-scale base models such as CogVideoX-5B and Wan-14B. Extensive experiments show that TIC-FT outperforms existing baselines in both condition fidelity and visual quality, while remaining highly efficient in both training and inference. For additional results, visit https://kinam0252.github.io/TIC-FT/
>
---
#### [replaced 030] MultiMotion: Multi Subject Video Motion Transfer via Video Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07500v2](https://arxiv.org/pdf/2512.07500v2)**

> **作者:** Penghui Liu; Jiangshan Wang; Yutong Shen; Shanhui Mo; Chenyang Qi; Yue Ma
>
> **摘要:** Multi-object video motion transfer poses significant challenges for Diffusion Transformer (DiT) architectures due to inherent motion entanglement and lack of object-level control. We present MultiMotion, a novel unified framework that overcomes these limitations. Our core innovation is Maskaware Attention Motion Flow (AMF), which utilizes SAM2 masks to explicitly disentangle and control motion features for multiple objects within the DiT pipeline. Furthermore, we introduce RectPC, a high-order predictor-corrector solver for efficient and accurate sampling, particularly beneficial for multi-entity generation. To facilitate rigorous evaluation, we construct the first benchmark dataset specifically for DiT-based multi-object motion transfer. MultiMotion demonstrably achieves precise, semantically aligned, and temporally coherent motion transfer for multiple distinct objects, maintaining DiT's high quality and scalability. The code is in the supp.
>
---
#### [replaced 031] The Finer the Better: Towards Granular-aware Open-set Domain Generalization
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.16979v2](https://arxiv.org/pdf/2511.16979v2)**

> **作者:** Yunyun Wang; Zheng Duan; Xinyue Liao; Ke-Jia Chen; Songcan Chen
>
> **备注:** 9 pages,3 figures,aaai2026
>
> **摘要:** Open-Set Domain Generalization (OSDG) tackles the realistic scenario where deployed models encounter both domain shifts and novel object categories. Despite impressive progress with vision-language models like CLIP, existing methods still fall into the dilemma between structural risk of known-classes and open-space risk from unknown-classes, and easily suffers from over-confidence, especially when distinguishing ``hard unknowns" that share fine-grained visual similarities with known classes. To this end, we propose a Semantic-enhanced CLIP (SeeCLIP) framework that explicitly addresses this dilemma through fine-grained semantic enhancement. In SeeCLIP, we propose a semantic-aware prompt enhancement module to decompose images into discriminative semantic tokens, enabling nuanced vision-language alignment beyond coarse category labels. To position unknown prompts effectively, we introduce duplex contrastive learning with complementary objectives, that is, repulsion to maintain separability from known classes, and cohesion to preserve semantic proximity. Further, our semantic-guided diffusion module synthesizes pseudo-unknowns by perturbing extracted semantic tokens, generating challenging samples that are visually similar to known classes yet exhibit key local differences. These hard negatives force the model to learn finer decision boundaries. Extensive experiments across five benchmarks demonstrate consistent improvements of 3% accuracy and 5% H-score over state-of-the-art methods.
>
---
#### [replaced 032] MM-SeR: Multimodal Self-Refinement for Lightweight Image Captioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.21451v4](https://arxiv.org/pdf/2508.21451v4)**

> **作者:** Junha Song; Yongsik Jo; So Yeon Min; Quanting Xie; Taehwan Kim; Yonatan Bisk; Jaegul Choo
>
> **备注:** Project page: https://sites.google.com/view/junha/mm-ser
>
> **摘要:** Systems such as video chatbots and navigation robots often depend on streaming image captioning to interpret visual inputs. Existing approaches typically employ large multimodal language models (MLLMs) for this purpose, but their substantial computational cost hinders practical application. This limitation motivates our development of a lightweight captioning model. Our investigation begins by replacing the large-scale language component in MLLMs with a compact 125M-parameter model. Surprisingly, this compact model, despite a 93x reduction in size, achieves comparable performance to MLLMs, suggesting that factual image captioning does not significantly require the complex reasoning abilities of LLMs. Despite this promising result, our lightweight model still lacks reliability. To address this, we draw inspiration from the human visual process: perceiving a global and coarse understanding of the scene before attending to finer details. Accordingly, we propose a multimodal self-refinement framework that guides the model to utilize features from salient regions, identified by referencing the previous coarse caption, and to produce a refined description. Experimental results demonstrate the superiority of our model in both single-sentence and detailed captioning, extending even to long-range video QA tasks.
>
---
#### [replaced 033] Enhancing Few-Shot Classification of Benchmark and Disaster Imagery with ATTBHFA-Net
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18326v2](https://arxiv.org/pdf/2510.18326v2)**

> **作者:** Gao Yu Lee; Tanmoy Dam; Md Meftahul Ferdaus; Daniel Puiu Poenar; Vu Duong
>
> **备注:** Submitted to a journal. Re-uploaded again after revising the mathematical derivation sections to clear up some errors in the steps
>
> **摘要:** The increasing frequency of natural and human-induced disasters necessitates advanced visual recognition techniques capable of analyzing critical photographic data. With progress in artificial intelligence and resilient computational systems, rapid and accurate disaster classification has become crucial for efficient rescue operations. However, visual recognition in disaster contexts faces significant challenges due to limited and diverse data from the difficulties in collecting and curating comprehensive, high-quality disaster imagery. Few-Shot Learning (FSL) provides a promising approach to data scarcity, yet current FSL research mainly relies on generic benchmark datasets lacking remote-sensing disaster imagery, limiting its practical effectiveness. Moreover, disaster images exhibit high intra-class variation and inter-class similarity, hindering the performance of conventional metric-based FSL methods. To address these issues, this paper introduces the Attention-based Bhattacharyya-Hellinger Feature Aggregation Network (ATTBHFA-Net), which linearly combines the Bhattacharyya coefficient and Hellinger distances to compare and aggregate feature probability distributions for robust prototype formation. The Bhattacharyya coefficient serves as a contrastive margin that enhances inter-class separability, while the Hellinger distance regularizes same-class alignment. This framework parallels contrastive learning but operates over probability distributions rather than embedded feature points. Furthermore, a Bhattacharyya-Hellinger distance-based contrastive loss is proposed as a distributional counterpart to cosine similarity loss, used jointly with categorical cross-entropy to significantly improve FSL performance. Experiments on four FSL benchmarks and two disaster image datasets demonstrate the superior effectiveness and generalization of ATTBHFA-Net compared to existing approaches.
>
---
#### [replaced 034] Multimodal Learning for Scalable Representation of High-Dimensional Medical Data
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2409.13115v2](https://arxiv.org/pdf/2409.13115v2)**

> **作者:** Areej Alsaafin; Abubakr Shafique; Saghir Alfasly; Krishna R. Kalari; H. R. Tizhoosh
>
> **摘要:** Integrating artificial intelligence (AI) with healthcare data is rapidly transforming medical diagnostics and driving progress toward precision medicine. However, effectively leveraging multimodal data, particularly digital pathology whole slide images (WSIs) and genomic sequencing, remains a significant challenge due to the intrinsic heterogeneity of these modalities and the need for scalable and interpretable frameworks. Existing diagnostic models typically operate on unimodal data, overlooking critical cross-modal interactions that can yield richer clinical insights. We introduce MarbliX (Multimodal Association and Retrieval with Binary Latent Indexed matriX), a self-supervised framework that learns to embed WSIs and immunogenomic profiles into compact, scalable binary codes, termed ``monogram.'' By optimizing a triplet contrastive objective across modalities, MarbliX captures high-resolution patient similarity in a unified latent space, enabling efficient retrieval of clinically relevant cases and facilitating case-based reasoning. \textcolor{black}{In lung cancer, MarbliX achieves 85-89\% across all evaluation metrics, outperforming histopathology (69-71\%) and immunogenomics (73-76\%). In kidney cancer, real-valued monograms yield the strongest performance (F1: 80-83\%, Accuracy: 87-90\%), with binary monograms slightly lower (F1: 78-82\%).
>
---
#### [replaced 035] FlowDirector: Training-Free Flow Steering for Precise Text-to-Video Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.05046v2](https://arxiv.org/pdf/2506.05046v2)**

> **作者:** Guangzhao Li; Yanming Yang; Chenxi Song; Chi Zhang
>
> **备注:** Project Page is https://flowdirector-edit.github.io
>
> **摘要:** Text-driven video editing aims to modify video content based on natural language instructions. While recent training-free methods have leveraged pretrained diffusion models, they often rely on an inversion-editing paradigm. This paradigm maps the video to a latent space before editing. However, the inversion process is not perfectly accurate, often compromising appearance fidelity and motion consistency. To address this, we introduce FlowDirector, a novel training-free and inversion-free video editing framework. Our framework models the editing process as a direct evolution in the data space. It guides the video to transition smoothly along its inherent spatio-temporal manifold using an ordinary differential equation (ODE), thereby avoiding the inaccurate inversion step. From this foundation, we introduce three flow correction strategies for appearance, motion, and stability: 1) Direction-aware flow correction amplifies components that oppose the source direction and removes irrelevant terms, breaking conservative streamlines and enabling stronger structural and textural changes. 2) Motion-appearance decoupling optimizes motion agreement as an energy term at each timestep, significantly improving consistency and motion transfer. 3) Differential averaging guidance strategy leverages differences among multiple candidate flows to approximate a low variance regime at low cost, suppressing artifacts and stabilizing the trajectory. Extensive experiments across various editing tasks and benchmarks demonstrate that FlowDirector achieves state-of-the-art performance in instruction following, temporal consistency, and background preservation, establishing an efficient new paradigm for coherent video editing without inversion.
>
---
#### [replaced 036] Efficient Action Counting with Dynamic Queries
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.01543v4](https://arxiv.org/pdf/2403.01543v4)**

> **作者:** Xiaoxuan Ma; Zishi Li; Qiuyan Shang; Wentao Zhu; Hai Ci; Yu Qiao; Yizhou Wang
>
> **备注:** code: https://github.com/lizishi/DeTRC, proj page: https://shirleymaxx.github.io/DeTRC/
>
> **摘要:** Temporal repetition counting aims to quantify the repeated action cycles within a video. The majority of existing methods rely on the similarity correlation matrix to characterize the repetitiveness of actions, but their scalability is hindered due to the quadratic computational complexity. In this work, we introduce a novel approach that employs an action query representation to localize repeated action cycles with linear computational complexity. Based on this representation, we further develop two key components to tackle the essential challenges of temporal repetition counting. Firstly, to facilitate open-set action counting, we propose the dynamic update scheme on action queries. Unlike static action queries, this approach dynamically embeds video features into action queries, offering a more flexible and generalizable representation. Secondly, to distinguish between actions of interest and background noise actions, we incorporate inter-query contrastive learning to regularize the video representations corresponding to different action queries. As a result, our method significantly outperforms previous works, particularly in terms of long video sequences, unseen actions, and actions at various speeds. On the challenging RepCountA benchmark, we outperform the state-of-the-art method TransRAC by 26.5% in OBO accuracy, with a 22.7% mean error decrease and 94.1% computational burden reduction. Code is available at https://github.com/lizishi/DeTRC.
>
---
#### [replaced 037] Enhancing Object Discovery for Unsupervised Instance Segmentation and Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02386v2](https://arxiv.org/pdf/2508.02386v2)**

> **作者:** Xingyu Feng; Hebei Gao; Hong Li
>
> **摘要:** We propose Cut-Once-and-LEaRn (COLER), a simple approach for unsupervised instance segmentation and object detection. COLER first uses our developed CutOnce to generate coarse pseudo labels, then enables the detector to learn from these masks. CutOnce applies Normalized Cut (NCut) only once and does not rely on any clustering methods (e.g., K-Means), but it can generate multiple object masks in an image. Our work opens a new direction for NCut algorithm in multi-object segmentation. We have designed several novel yet simple modules that not only allow CutOnce to fully leverage the object discovery capabilities of self-supervised model, but also free it from reliance on mask post-processing. During training, COLER achieves strong performance without requiring specially designed loss functions for pseudo labels, and its performance is further improved through self-training. COLER is a zero-shot unsupervised model that outperforms previous state-of-the-art methods on multiple benchmarks. We believe our method can help advance the field of unsupervised object localization. Code is available at: https://github.com/Quantumcraft616/COLER.
>
---
#### [replaced 038] Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究语言条件下的行为克隆任务，旨在解决序列动作中累积误差导致的执行不连续与语义-物理错位问题。提出CCoL框架，通过视觉-语言-动作连续协同学习与双向跨注意力实现语义-物理对齐，提升动作克隆的连贯性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.14396v3](https://arxiv.org/pdf/2511.14396v3)**

> **作者:** Xiuxiu Qi; Yu Yang; Jiannong Cao; Luyao Bai; Chongshan Fan; Chengtai Cao; Hongpeng Wang
>
> **备注:** Accepted at AAAI 2026, the Project website is available at https://qhemu.github.io/CCoL/
>
> **摘要:** Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.
>
---
#### [replaced 039] Exploring Diffusion with Test-Time Training on Efficient Image Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.14541v3](https://arxiv.org/pdf/2506.14541v3)**

> **作者:** Rongchang Lu; Tianduo Luo; Yunzhi Jiang; Conghan Yue; Pei Yang; Guibao Liu; Changyang Gu
>
> **备注:** We withdraw this paper due to erroneous experiment data in the ablation study, which was inadvertently copied from our preprint "Ultra-Lightweight Semantic-Injected Imagery Super-Resolution for Real-Time UAV Remote Sensing" This nearly constituted academic misconduct. We sincerely apologize and thank those who alerted us
>
> **摘要:** Image restoration faces challenges including ineffective feature fusion, computational bottlenecks and inefficient diffusion processes. To address these, we propose DiffRWKVIR, a novel framework unifying Test-Time Training (TTT) with efficient diffusion. Our approach introduces three key innovations: (1) Omni-Scale 2D State Evolution extends RWKV's location-dependent parameterization to hierarchical multi-directional 2D scanning, enabling global contextual awareness with linear complexity O(L); (2) Chunk-Optimized Flash Processing accelerates intra-chunk parallelism by 3.2x via contiguous chunk processing (O(LCd) complexity), reducing sequential dependencies and computational overhead; (3) Prior-Guided Efficient Diffusion extracts a compact Image Prior Representation (IPR) in only 5-20 steps, proving 45% faster training/inference than DiffIR while solving computational inefficiency in denoising. Evaluated across super-resolution and inpainting benchmarks (Set5, Set14, BSD100, Urban100, Places365), DiffRWKVIR outperforms SwinIR, HAT, and MambaIR/v2 in PSNR, SSIM, LPIPS, and efficiency metrics. Our method establishes a new paradigm for adaptive, high-efficiency image restoration with optimized hardware utilization.
>
---
#### [replaced 040] Wukong's 72 Transformations: High-fidelity Textured 3D Morphing via Flow Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22425v3](https://arxiv.org/pdf/2511.22425v3)**

> **作者:** Minghao Yin; Yukang Cao; Kai Han
>
> **摘要:** We present WUKONG, a novel training-free framework for high-fidelity textured 3D morphing that takes a pair of source and target prompts (image or text) as input. Unlike conventional methods -- which rely on manual correspondence matching and deformation trajectory estimation (limiting generalization and requiring costly preprocessing) -- WUKONG leverages the generative prior of flow-based transformers to produce high-fidelity 3D transitions with rich texture details. To ensure smooth shape transitions, we exploit the inherent continuity of flow-based generative processes and formulate morphing as an optimal transport barycenter problem. We further introduce a sequential initialization strategy to prevent abrupt geometric distortions and preserve identity coherence. For faithful texture preservation, we propose a similarity-guided semantic consistency mechanism that selectively retains high-frequency details and enables precise control over blending dynamics. This avoids common artifacts like oversmoothing while maintaining semantic fidelity. Extensive quantitative and qualitative evaluations demonstrate that WUKONG significantly outperforms state-of-the-art methods, achieving superior results across diverse geometry and texture variations.
>
---
#### [replaced 041] Enhancing Supervised Composed Image Retrieval via Reasoning-Augmented Representation Engineering
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.11272v2](https://arxiv.org/pdf/2508.11272v2)**

> **作者:** Jun Li; Hongjian Dou; Zhenyu Zhang; Kai Li; Shaoguo Liu; Tingting Gao
>
> **摘要:** Composed Image Retrieval (CIR) presents a significant challenge as it requires jointly understanding a reference image and a modified textual instruction to find relevant target images. Some existing methods attempt to use a two-stage approach to further refine retrieval results. However, this often requires additional training of a ranking model. Despite the success of Chain-of-Thought (CoT) techniques in reducing training costs for language models, their application in CIR tasks remains limited -- compressing visual information into text or relying on elaborate prompt designs. Besides, existing works only utilize it for zero-shot CIR, as it is challenging to achieve satisfactory results in supervised CIR with a well-trained model. In this work, we proposed a framework that includes the Pyramid Matching Model with Training-Free Refinement (PMTFR) to address these challenges. Through a simple but effective module called Pyramid Patcher, we enhanced the Pyramid Matching Model's understanding of visual information at different granularities. Inspired by representation engineering, we extracted representations from COT data and injected them into the LVLMs. This approach allowed us to obtain refined retrieval scores in the Training-Free Refinement paradigm without relying on explicit textual reasoning, further enhancing performance. Extensive experiments on CIR benchmarks demonstrate that PMTFR surpasses state-of-the-art methods in supervised CIR tasks. The code will be made public.
>
---
#### [replaced 042] MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.20629v2](https://arxiv.org/pdf/2511.20629v2)**

> **作者:** Chieh-Yun Chen; Zhonghao Wang; Qi Chen; Zhifan Ye; Min Shi; Yue Zhao; Yinan Zhao; Hui Qu; Wei-An Lin; Yiru Shen; Ajinkya Kale; Irfan Essa; Humphrey Shi
>
> **摘要:** Reinforcement learning from human feedback (RLHF) with reward models has advanced alignment of generative models to human aesthetic and perceptual preferences. However, jointly optimizing multiple rewards often incurs an alignment tax, improving one dimension while degrading others. To address this, we introduce two complementary methods: MapReduce LoRA and Reward-aware Token Embedding (RaTE). MapReduce LoRA trains preference-specific LoRA experts in parallel and iteratively merges them to refine a shared base model; RaTE learns reward-specific token embeddings that compose at inference for flexible preference control. Experiments on Text-to-Image generation (Stable Diffusion 3.5 Medium and FLUX.1-dev) show improvements of 36.1%, 4.6%, and 55.7%, and 32.7%, 4.3%, and 67.1% on GenEval, PickScore, and OCR, respectively. On Text-to-Video generation (HunyuanVideo), visual and motion quality improve by 48.1% and 90.0%, respectively. On the language task, Helpful Assistant, with Llama-2 7B, helpful and harmless improve by 43.4% and 136.7%, respectively. Our framework sets a new state-of-the-art multi-preference alignment recipe across modalities.
>
---
#### [replaced 043] VADER: Towards Causal Video Anomaly Understanding with Relation-Aware Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07299v2](https://arxiv.org/pdf/2511.07299v2)**

> **作者:** Ying Cheng; Yu-Ho Lin; Min-Hung Chen; Fu-En Yang; Shang-Hong Lai
>
> **备注:** Accepted to WACV 2026. Project page available at: https://vader-vau.github.io/
>
> **摘要:** Video anomaly understanding (VAU) aims to provide detailed interpretation and semantic comprehension of anomalous events within videos, addressing limitations of traditional methods that focus solely on detecting and localizing anomalies. However, existing approaches often neglect the deeper causal relationships and interactions between objects, which are critical for understanding anomalous behaviors. In this paper, we propose VADER, an LLM-driven framework for Video Anomaly unDErstanding, which integrates keyframe object Relation features with visual cues to enhance anomaly comprehension from video. Specifically, VADER first applies an Anomaly Scorer to assign per-frame anomaly scores, followed by a Context-AwarE Sampling (CAES) strategy to capture the causal context of each anomalous event. A Relation Feature Extractor and a COntrastive Relation Encoder (CORE) jointly model dynamic object interactions, producing compact relational representations for downstream reasoning. These visual and relational cues are integrated with LLMs to generate detailed, causally grounded descriptions and support robust anomaly-related question answering. Experiments on multiple real-world VAU benchmarks demonstrate that VADER achieves strong results across anomaly description, explanation, and causal reasoning tasks, advancing the frontier of explainable video anomaly analysis.
>
---
#### [replaced 044] BrainExplore: Large-Scale Discovery of Interpretable Visual Representations in the Human Brain
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08560v2](https://arxiv.org/pdf/2512.08560v2)**

> **作者:** Navve Wasserman; Matias Cosarinsky; Yuval Golbari; Aude Oliva; Antonio Torralba; Tamar Rott Shaham; Michal Irani
>
> **摘要:** Understanding how the human brain represents visual concepts, and in which brain regions these representations are encoded, remains a long-standing challenge. Decades of work have advanced our understanding of visual representations, yet brain signals remain large and complex, and the space of possible visual concepts is vast. As a result, most studies remain small-scale, rely on manual inspection, focus on specific regions and properties, and rarely include systematic validation. We present a large-scale, automated framework for discovering and explaining visual representations across the human cortex. Our method comprises two main stages. First, we discover candidate interpretable patterns in fMRI activity through unsupervised, data-driven decomposition methods. Next, we explain each pattern by identifying the set of natural images that most strongly elicit it and generating a natural-language description of their shared visual meaning. To scale this process, we introduce an automated pipeline that tests multiple candidate explanations, assigns quantitative reliability scores, and selects the most consistent description for each voxel pattern. Our framework reveals thousands of interpretable patterns spanning many distinct visual concepts, including fine-grained representations previously unreported.
>
---
#### [replaced 045] Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.17186v2](https://arxiv.org/pdf/2508.17186v2)**

> **作者:** Zhenghui Zhao; Chen Wu; Di Wang; Hongruixuan Chen; Cuiqun Chen; Zhuo Zheng; Bo Du; Liangpei Zhang
>
> **备注:** Accepted by IEEE Transactions on Image Processing
>
> **摘要:** Weakly-Supervised Change Detection (WSCD) aims to distinguish specific object changes (e.g., objects appearing or disappearing) from background variations (e.g., environmental changes due to light, weather, or seasonal shifts) in paired satellite images, relying only on paired image (i.e., image-level) classification labels. This technique significantly reduces the need for dense annotations required in fully-supervised change detection. However, as image-level supervision only indicates whether objects have changed in a scene, WSCD methods often misclassify background variations as object changes, especially in complex remote-sensing scenarios. In this work, we propose an Adversarial Class Prompting (AdvCP) method to address this co-occurring noise problem, including two phases: a) Adversarial Prompt Mining: After each training iteration, we introduce adversarial prompting perturbations, using incorrect one-hot image-level labels to activate erroneous feature mappings. This process reveals co-occurring adversarial samples under weak supervision, namely background variation features that are likely to be misclassified as object changes. b) Adversarial Sample Rectification: We integrate these adversarially prompt-activated pixel samples into training by constructing an online global prototype. This prototype is built from an exponentially weighted moving average of the current batch and all historical training data. Our AdvCP can be seamlessly integrated into current WSCD methods without adding additional inference cost. Experiments on ConvNet, Transformer, and Segment Anything Model (SAM)-based baselines demonstrate significant performance enhancements. Furthermore, we demonstrate the generalizability of AdvCP to other multi-class weakly-supervised dense prediction scenarios. Code is available at https://github.com/zhenghuizhao/AdvCP
>
---
#### [replaced 046] MADrive: Memory-Augmented Driving Scene Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21520v2](https://arxiv.org/pdf/2506.21520v2)**

> **作者:** Polina Karpikova; Daniil Selikhanovych; Kirill Struminsky; Ruslan Musaev; Maria Golitsyna; Dmitry Baranchuk
>
> **摘要:** Recent advances in scene reconstruction have pushed toward highly realistic modeling of autonomous driving (AD) environments using 3D Gaussian splatting. However, the resulting reconstructions remain closely tied to the original observations and struggle to support photorealistic synthesis of significantly altered or novel driving scenarios. This work introduces MADrive, a memory-augmented reconstruction framework designed to extend the capabilities of existing scene reconstruction methods by replacing observed vehicles with visually similar 3D assets retrieved from a large-scale external memory bank. Specifically, we release MAD-Cars, a curated dataset of ${\sim}70$K 360° car videos captured in the wild and present a retrieval module that finds the most similar car instances in the memory bank, reconstructs the corresponding 3D assets from video, and integrates them into the target scene through orientation alignment and relighting. The resulting replacements provide complete multi-view representations of vehicles in the scene, enabling photorealistic synthesis of substantially altered configurations, as demonstrated in our experiments. Project page: https://yandex-research.github.io/madrive/
>
---
#### [replaced 047] Annotation-Free Reinforcement Learning Query Rewriting via Verifiable Search Reward
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究检索增强生成（RAG）中的查询重写任务，旨在解决依赖人工标注数据的问题。提出RL-QR框架，利用可验证的搜索奖励实现免标注强化学习，提升多模态检索性能。**

- **链接: [https://arxiv.org/pdf/2507.23242v2](https://arxiv.org/pdf/2507.23242v2)**

> **作者:** Sungguk Cha; DongWook Kim; Taeseung Hahn; Mintae Kim; Youngsub Han; Byoung-Ki Jeon
>
> **摘要:** Optimizing queries for Retrieval-Augmented Generation (RAG) systems poses a significant challenge, particularly across diverse modal indices. We introduce RL-QR, a novel annotation-free reinforcement learning framework for query rewriting that eliminates the need for costly human-annotated data. By leveraging verifiable search rewards derived from index-aligned synthetic queries, RL-QR overcomes human-annotation dependencies, extending its applicability to various modalities and index domains. Experimental results demonstrate the framework's robustness, achieving substantial retrieval performance gains of up to 3.9$\times$ on lexical retrievers and 3.5$\times$ on semantic retrievers on the MTEB VIDORE V2 benchmark for unstructured visual documents, along with consistent 5\% to 10\% improvements on MS MARCO v2.1 and internal industrial datasets.
>
---
#### [replaced 048] Open-World Object Counting in Videos
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.15368v2](https://arxiv.org/pdf/2506.15368v2)**

> **作者:** Niki Amini-Naieni; Andrew Zisserman
>
> **备注:** AAAI 2026
>
> **摘要:** We introduce a new task of open-world object counting in videos: given a text description, or an image example, that specifies the target object, the objective is to enumerate all the unique instances of the target objects in the video. This task is especially challenging in crowded scenes with occlusions and objects of similar appearance, where avoiding double counting and identifying reappearances is crucial. To this end, we make the following contributions: we introduce a model, CountVid, for this task. It leverages an image-based counting model, and a promptable video segmentation and tracking model, to enable automated open-world object counting across video frames. To evaluate its performance, we introduce VideoCount, a new dataset for this novel task built from the TAO and MOT20 tracking datasets, as well as from videos of penguins and metal alloy crystallization captured by x-rays. Using this dataset, we demonstrate that CountVid provides accurate object counts, and significantly outperforms strong baselines. The VideoCount dataset, the CountVid model, and all the code are available at https://www.robots.ox.ac.uk/~vgg/research/countvid/.
>
---
#### [replaced 049] MOAT: Evaluating LMMs for Capability Integration and Instruction Grounding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出MOAT基准，评估大视觉语言模型在复杂现实任务中的能力整合与指令 grounding 问题。针对现有模型在多能力协同和复杂指令理解上的不足，构建了含1005题的测试集，揭示当前最优模型准确率仅44%，并分析瓶颈原因，推动模型改进。**

- **链接: [https://arxiv.org/pdf/2503.09348v2](https://arxiv.org/pdf/2503.09348v2)**

> **作者:** Zhoutong Ye; Mingze Sun; Huan-ang Gao; Xutong Wang; Xiangyang Wang; Yu Mei; Chang Liu; Qinwei Li; Chengwen Zhang; Qinghuan Lan; Chun Yu; Yuanchun Shi
>
> **备注:** Project page: https://cambrian-yzt.github.io/MOAT
>
> **摘要:** Large multimodal models (LMMs) have demonstrated significant potential as generalists in vision-language (VL) tasks. However, adoption of LMMs in real-world tasks is hindered by their poor performance in tasks that require a combination of VL capabilities, as well as in tasks that involve the grounding of complex text or visual instructions. To thoroughly investigate this gap and its underlying causes, we propose MOAT, a diverse benchmark with 1005 complex real-world vision questions that are straightforward for humans but challenging for LMMs. Specifically, the tasks in MOAT require LMMs to engage in generalist problem solving by integrating VL capabilities such as reading text, counting, understanding spatial relations, grounding textual and visual instructions, etc. All these abilities fit into a taxonomy proposed by us that contains 9 VL capabilities, enabling MOAT to provide a fine-grained view of LMMs' strengths and weaknesses. Besides, MOAT is the first benchmark to explicitly evaluate LMMs' ability to ground complex text and visual instructions, which is essential for many real-world applications. We evaluated 17 proprietary and open source LMMs, finding that the best performing LMM (Gemini 2.5 Pro) achieved only 44% accuracy, far below what would be acceptable in real-world applications. To guide future model development, we analyze common trends in our results and discuss the underlying causes of poor performance, focusing on the impact of text-centric reasoning, which VL capabilities form bottlenecks in complex tasks, and the potential harmful effects of tiling. Code and data are available at https://cambrian-yzt.github.io/MOAT/.
>
---
#### [replaced 050] ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.04678v2](https://arxiv.org/pdf/2507.04678v2)**

> **作者:** Zhenghui Zhao; Chen Wu; Xiangyong Cao; Di Wang; Hongruixuan Chen; Datao Tang; Liangpei Zhang; Zhuo Zheng
>
> **摘要:** Spatiotemporal image generation is a highly meaningful task, which can generate future scenes conditioned on given observations. However, existing change generation methods can only handle event-driven changes (e.g., new buildings) and fail to model cross-temporal variations (e.g., seasonal shifts). In this work, we propose ChangeBridge, a conditional spatiotemporal image generation model for remote sensing. Given pre-event images and multimodal event controls, ChangeBridge generates post-event scenes that are both spatially and temporally coherent. The core idea is a drift-asynchronous diffusion bridge. Specifically, it consists of three main modules: a) Composed bridge initialization, which replaces noise initialization. It starts the diffusion from a composed pre-event state, modeling a diffusion bridge process. b) Asynchronous Drift Diffusion, which uses a pixel-wise drift map, assigning different drift magnitudes to event and temporal evolution. This enables differentiated generation during the pre-to-post transition. c) Drift-Aware Denoising, which embeds the drift map into the denoising network, guiding drift-aware reconstruction. Experiments show that ChangeBridge can generate better cross-spatiotemporal aligned scenarios compared to state-of-the-art methods. Additionally, ChangeBridge shows great potential for land-use planning and as a data generation engine for a series of change detection tasks.
>
---
#### [replaced 051] UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.06165v4](https://arxiv.org/pdf/2509.06165v4)**

> **作者:** Huy Le; Nhat Chung; Tung Kieu; Jingkang Yang; Ngan Le
>
> **备注:** 11 pages, 7 figures. Accepted at WACV 2026
>
> **摘要:** Video Scene Graph Generation (VidSGG) aims to represent dynamic visual content by detecting objects and modeling their temporal interactions as structured graphs. Prior studies typically target either coarse-grained box-level or fine-grained panoptic pixel-level VidSGG, often requiring task-specific architectures and multi-stage training pipelines. In this paper, we present UNO (UNified Object-centric VidSGG), a single-stage, unified framework that jointly addresses both tasks within an end-to-end architecture. UNO is designed to minimize task-specific modifications and maximize parameter sharing, enabling generalization across different levels of visual granularity. The core of UNO is an extended slot attention mechanism that decomposes visual features into object and relation slots. To ensure robust temporal modeling, we introduce object temporal consistency learning, which enforces consistent object representations across frames without relying on explicit tracking modules. Additionally, a dynamic triplet prediction module links relation slots to corresponding object pairs, capturing evolving interactions over time. We evaluate UNO on standard box-level and pixel-level VidSGG benchmarks. Results demonstrate that UNO not only achieves competitive performance across both tasks but also offers improved efficiency through a unified, object-centric design.
>
---
#### [replaced 052] Tera-MIND: Tera-scale mouse brain simulation via spatial mRNA-guided diffusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.01220v3](https://arxiv.org/pdf/2503.01220v3)**

> **作者:** Jiqing Wu; Ingrid Berg; Yawei Li; Ender Konukoglu; Viktor H. Koelzer
>
> **摘要:** Holistic 3D modeling of molecularly defined brain structures is crucial for understanding complex brain functions. Using emerging tissue profiling technologies, researchers charted comprehensive atlases of mammalian brain with sub-cellular resolution and spatially resolved transcriptomic data. However, these tera-scale volumetric atlases pose computational challenges for modeling intricate brain structures within the native spatial context. We propose \textbf{Tera-MIND}, a novel generative framework capable of simulating \textbf{Tera}-scale \textbf{M}ouse bra\textbf{IN}s in 3D using a patch-based and boundary-aware \textbf{D}iffusion model. Taking spatial gene expression as conditional input, we generate virtual mouse brains with comprehensive cellular morphological detail at teravoxel scale. Through the lens of 3D \textit{gene}-\textit{gene} self-attention, we identify spatial molecular interactions for key transcriptomic pathways, including glutamatergic and dopaminergic neuronal systems. Lastly, we showcase the translational applicability of Tera-MIND on previously unseen human brain samples. Tera-MIND offers an efficient generative modeling of whole virtual organisms, paving the way for integrative applications in biomedical research. Project website: https://musikisomorphie.github.io/Tera-MIND.html
>
---
#### [replaced 053] Equivariant symmetry-aware head pose estimation for fetal MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04890v3](https://arxiv.org/pdf/2512.04890v3)**

> **作者:** Ramya Muthukrishnan; Borjan Gagoski; Aryn Lee; P. Ellen Grant; Elfar Adalsteinsson; Polina Golland; Benjamin Billot
>
> **摘要:** We present E(3)-Pose, a novel fast pose estimation method that jointly and explicitly models rotation equivariance and object symmetry. Our work is motivated by the challenging problem of accounting for fetal head motion during a diagnostic MRI scan. We aim to enable automatic adaptive prescription of 2D diagnostic MRI slices with 6-DoF head pose estimation, supported by 3D MRI volumes rapidly acquired before each 2D slice. Existing methods struggle to generalize to clinical volumes, due to pose ambiguities induced by inherent anatomical symmetries, as well as low resolution, noise, and artifacts. In contrast, E(3)-Pose captures anatomical symmetries and rigid pose equivariance by construction, and yields robust estimates of the fetal head pose. Our experiments on publicly available and representative clinical fetal MRI datasets demonstrate the superior robustness and generalization of our method across domains. Crucially, E(3)-Pose achieves state-of-the-art accuracy on clinical MRI volumes, paving the way for clinical translation. Our implementation is available at github.com/ramyamut/E3-Pose.
>
---
#### [replaced 054] Gaze on the Prize: Shaping Visual Attention with Return-Guided Contrastive Learning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉强化学习任务，旨在解决高维图像中无关像素干扰导致的样本效率低问题。作者提出“Gaze on the Prize”框架，通过返回引导的对比学习，训练可学习的注视机制聚焦任务相关特征，提升样本效率并解决基线难以收敛的任务。**

- **链接: [https://arxiv.org/pdf/2510.08442v2](https://arxiv.org/pdf/2510.08442v2)**

> **作者:** Andrew Lee; Ian Chuang; Dechen Gao; Kai Fukazawa; Iman Soltani
>
> **备注:** Project page: https://andrewcwlee.github.io/gaze-on-the-prize
>
> **摘要:** Visual Reinforcement Learning (RL) agents must learn to act based on high-dimensional image data where only a small fraction of the pixels is task-relevant. This forces agents to waste exploration and computational resources on irrelevant features, leading to sample-inefficient and unstable learning. To address this, inspired by human visual foveation, we introduce Gaze on the Prize. This framework augments visual RL with a learnable foveal attention mechanism (Gaze), guided by a self-supervised signal derived from the agent's experience pursuing higher returns (the Prize). Our key insight is that return differences reveal what matters most: If two similar representations produce different outcomes, their distinguishing features are likely task-relevant, and the gaze should focus on them accordingly. This is realized through return-guided contrastive learning that trains the attention to distinguish between the features relevant to success and failure. We group similar visual representations into positives and negatives based on their return differences and use the resulting labels to construct contrastive triplets. These triplets provide the training signal that teaches the attention mechanism to produce distinguishable representations for states associated with different outcomes. Our method achieves up to 2.52x improvement in sample efficiency and can solve challenging tasks from the ManiSkill3 benchmark that the baseline fails to learn, without modifying the underlying algorithm or hyperparameters.
>
---
#### [replaced 055] Unconsciously Forget: Mitigating Memorization; Without Knowing What is being Memorized
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.09687v2](https://arxiv.org/pdf/2512.09687v2)**

> **作者:** Er Jin; Yang Zhang; Yongli Mou; Yanfei Dong; Stefan Decker; Kenji Kawaguchi; Johannes Stegmaier
>
> **摘要:** Recent advances in generative models have demonstrated an exceptional ability to produce highly realistic images. However, previous studies show that generated images often resemble the training data, and this problem becomes more severe as the model size increases. Memorizing training data can lead to legal challenges, including copyright infringement, violations of portrait rights, and trademark violations. Existing approaches to mitigating memorization mainly focus on manipulating the denoising sampling process to steer image embeddings away from the memorized embedding space or employ unlearning methods that require training on datasets containing specific sets of memorized concepts. However, existing methods often incur substantial computational overhead during sampling, or focus narrowly on removing one or more groups of target concepts, imposing a significant limitation on their scalability. To understand and mitigate these problems, our work, UniForget, offers a new perspective on understanding the root cause of memorization. Our work demonstrates that specific parts of the model are responsible for copyrighted content generation. By applying model pruning, we can effectively suppress the probability of generating copyrighted content without targeting specific concepts while preserving the general generative capabilities of the model. Additionally, we show that our approach is both orthogonal and complementary to existing unlearning methods, thereby highlighting its potential to improve current unlearning and de-memorization techniques.
>
---
#### [replaced 056] Noise Matters: Optimizing Matching Noise for Diffusion Classifiers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11330v2](https://arxiv.org/pdf/2508.11330v2)**

> **作者:** Yanghao Wang; Long Chen
>
> **摘要:** Although today's pretrained discriminative vision-language models (e.g., CLIP) have demonstrated strong perception abilities, such as zero-shot image classification, they also suffer from the bag-of-words problem and spurious bias. To mitigate these problems, some pioneering studies leverage powerful generative models (e.g., pretrained diffusion models) to realize generalizable image classification, dubbed Diffusion Classifier (DC). Specifically, by randomly sampling a Gaussian noise, DC utilizes the differences of denoising effects with different category conditions to classify categories. Unfortunately, an inherent and notorious weakness of existing DCs is noise instability: different random sampled noises lead to significant performance changes. To achieve stable classification performance, existing DCs always ensemble the results of hundreds of sampled noises, which significantly reduces the classification speed. To this end, we firstly explore the role of noise in DC, and conclude that: there are some ``good noises'' that can relieve the instability. Meanwhile, we argue that these good noises should meet two principles: Frequency Matching and Spatial Matching. Regarding both principles, we propose a novel Noise Optimization method to learn matching (i.e., good) noise for DCs: NoOp. For frequency matching, NoOp first optimizes a dataset-specific noise: Given a dataset and a timestep t, optimize one randomly initialized parameterized noise. For Spatial Matching, NoOp trains a Meta-Network that adopts an image as input and outputs image-specific noise offset. The sum of optimized noise and noise offset will be used in DC to replace random noise. Extensive ablations on various datasets demonstrated the effectiveness of NoOp.
>
---
#### [replaced 057] Class-wise Balancing Data Replay for Federated Class-Incremental Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.07712v4](https://arxiv.org/pdf/2507.07712v4)**

> **作者:** Zhuang Qi; Ying-Peng Tang; Lei Meng; Han Yu; Xiaoxiao Li; Xiangxu Meng
>
> **备注:** NeurIPS'25 Accepted, Oral
>
> **摘要:** Federated Class Incremental Learning (FCIL) aims to collaboratively process continuously increasing incoming tasks across multiple clients. Among various approaches, data replay has become a promising solution, which can alleviate forgetting by reintroducing representative samples from previous tasks. However, their performance is typically limited by class imbalance, both within the replay buffer due to limited global awareness and between replayed and newly arrived classes. To address this issue, we propose a class wise balancing data replay method for FCIL (FedCBDR), which employs a global coordination mechanism for class-level memory construction and reweights the learning objective to alleviate the aforementioned imbalances. Specifically, FedCBDR has two key components: 1) the global-perspective data replay module reconstructs global representations of prior task in a privacy-preserving manner, which then guides a class-aware and importance-sensitive sampling strategy to achieve balanced replay; 2) Subsequently, to handle class imbalance across tasks, the task aware temperature scaling module adaptively adjusts the temperature of logits at both class and instance levels based on task dynamics, which reduces the model's overconfidence in majority classes while enhancing its sensitivity to minority classes. Experimental results verified that FedCBDR achieves balanced class-wise sampling under heterogeneous data distributions and improves generalization under task imbalance between earlier and recent tasks, yielding a 2%-15% Top-1 accuracy improvement over six state-of-the-art methods.
>
---
#### [replaced 058] Conditional Text-to-Image Generation with Reference Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16713v2](https://arxiv.org/pdf/2411.16713v2)**

> **作者:** Taewook Kim; Ze Wang; Zhengyuan Yang; Jiang Wang; Lijuan Wang; Zicheng Liu; Qiang Qiu
>
> **备注:** WACV 2026
>
> **摘要:** Text-to-image diffusion models have demonstrated tremendous success in synthesizing visually stunning images given textual instructions. Despite remarkable progress in creating high-fidelity visuals, text-to-image models can still struggle with precisely rendering subjects, such as text spelling. To address this challenge, this paper explores using additional conditions of an image that provides visual guidance of the particular subjects for diffusion models to generate. In addition, this reference condition empowers the model to be conditioned in ways that the vocabularies of the text tokenizer cannot adequately represent, and further extends the model's generalization to novel capabilities such as generating non-English text spellings. We develop several small-scale expert plugins that efficiently endow a Stable Diffusion model with the capability to take different references. Each plugin is trained with auxiliary networks and loss functions customized for applications such as English scene-text generation, multi-lingual scene-text generation, and logo-image generation. Our expert plugins demonstrate superior results than the existing methods on all tasks, each containing only 28.55M trainable parameters.
>
---
