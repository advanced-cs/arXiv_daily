# 计算机视觉 cs.CV

- **最新发布 123 篇**

- **更新 67 篇**

## 最新发布

#### [new 001] PatchPoison: Poisoning Multi-View Datasets to Degrade 3D Reconstruction
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文属于3D重建安全任务，旨在防止未经授权的3D模型重建。提出PatchPoison方法，在多视角数据中注入小而隐蔽的对抗补丁，干扰SfM流程，提升重建误差。**

- **链接: [https://arxiv.org/pdf/2604.13153](https://arxiv.org/pdf/2604.13153)**

> **作者:** Prajas Wadekar; Venkata Sai Pranav Bachina; Kunal Bhosikar; Ankit Gangwal; Charu Sharma
>
> **备注:** CVPR Workshop on Security, Privacy, and Adversarial Robustness in 3D Generative Vision Models (SPAR-3D), 2026
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently enabled highly photorealistic 3D reconstruction from casually captured multi-view images. However, this accessibility raises a privacy concern: publicly available images or videos can be exploited to reconstruct detailed 3D models of scenes or objects without the owner's consent. We present PatchPoison, a lightweight dataset-poisoning method that prevents unauthorized 3D reconstruction. Unlike global perturbations, PatchPoison injects a small high-frequency adversarial patch, a structured checkerboard, into the periphery of each image in a multi-view dataset. The patch is designed to corrupt the feature-matching stage of Structure-from-Motion (SfM) pipelines such as COLMAP by introducing spurious correspondences that systematically misalign estimated camera poses. Consequently, downstream 3DGS optimization diverges from the correct scene geometry. On the NeRF-Synthetic benchmark, inserting a 12 X 12 pixel patch increases reconstruction error by 6.8x in LPIPS, while the poisoned images remain unobtrusive to human viewers. PatchPoison requires no pipeline modifications, offering a practical, "drop-in" preprocessing step for content creators to protect their multi-view data.
>
---
#### [new 002] Gaslight, Gatekeep, V1-V3: Early Visual Cortex Alignment Shields Vision-Language Models from Sycophantic Manipulation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型安全研究，旨在解决模型易受操控的问题。通过分析12个模型的视觉表征与人类大脑的对齐程度，发现早期视觉皮层对齐度越高，模型越不易被操纵。**

- **链接: [https://arxiv.org/pdf/2604.13803](https://arxiv.org/pdf/2604.13803)**

> **作者:** Arya Shah; Vaibhav Tripathi; Mayank Singh; Chaklam Silpasuwanchai
>
> **备注:** 28 pages, 9 figures, 13 tables
>
> **摘要:** Vision-language models are increasingly deployed in high-stakes settings, yet their susceptibility to sycophantic manipulation remains poorly understood, particularly in relation to how these models represent visual information internally. Whether models whose visual representations more closely mirror human neural processing are also more resistant to adversarial pressure is an open question with implications for both neuroscience and AI safety. We investigate this question by evaluating 12 open-weight vision-language models spanning 6 architecture families and a 40$\times$ parameter range (256M--10B) along two axes: brain alignment, measured by predicting fMRI responses from the Natural Scenes Dataset across 8 human subjects and 6 visual cortex regions of interest, and sycophancy, measured through 76,800 two-turn gaslighting prompts spanning 5 categories and 10 difficulty levels. Region-of-interest analysis reveals that alignment specifically in early visual cortex (V1--V3) is a reliable negative predictor of sycophancy ($r = -0.441$, BCa 95\% CI $[-0.740, -0.031]$), with all 12 leave-one-out correlations negative and the strongest effect for existence denial attacks ($r = -0.597$, $p = 0.040$). This anatomically specific relationship is absent in higher-order category-selective regions, suggesting that faithful low-level visual encoding provides a measurable anchor against adversarial linguistic override in vision-language models. We release our code on \href{this https URL}{GitHub} and dataset on \href{this https URL}{Hugging Face}
>
---
#### [new 003] MApLe: Multi-instance Alignment of Diagnostic Reports and Large Medical Images
- **分类: cs.CV**

- **简介: 该论文提出MApLe模型，解决医学影像与诊断报告之间的多实例对齐问题。通过分离解剖区域和诊断发现，实现图像局部信息与文本的精准关联。**

- **链接: [https://arxiv.org/pdf/2604.13970](https://arxiv.org/pdf/2604.13970)**

> **作者:** Felicia Bader; Philipp Seeböck; Anastasia Bartashova; Ulrike Attenberger; Georg Langs
>
> **备注:** Accepted for MIDL 2026; Reviews available at this https URL
>
> **摘要:** In diagnostic reports, experts encode complex imaging data into clinically actionable information. They describe subtle pathological findings that are meaningful in their anatomical context. Reports follow relatively consistent structures, expressing diagnostic information with few words that are often associated with tiny but consequential image observations. Standard vision language models struggle to identify the associations between these informative text components and small locations in the images. Here, we propose "MApLe", a multi-task, multi-instance vision language alignment approach that overcomes these limitations. It disentangles the concepts of anatomical region and diagnostic finding, and links local image information to sentences in a patch-wise approach. Our method consists of a text embedding trained to capture anatomical and diagnostic concepts in sentences, a patch-wise image encoder conditioned on anatomical structures, and a multi-instance alignment of these representations. We demonstrate that MApLe can successfully align different image regions and multiple diagnostic findings in free-text reports. We show that our model improves the alignment performance compared to state-of-the-art baseline models when evaluated on several downstream tasks. The code is available at this https URL.
>
---
#### [new 004] ROSE: Retrieval-Oriented Segmentation Enhancement
- **分类: cs.CV**

- **简介: 该论文提出ROSE框架，解决多模态大语言模型在分割新出现实体时的不足。通过检索增强提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.14147](https://arxiv.org/pdf/2604.14147)**

> **作者:** Song Tang; Guangquan Jie; Henghui Ding; Yu-Gang Jiang
>
> **备注:** CVPR 2026 Findings, Project Page: this https URL
>
> **摘要:** Existing segmentation models based on multimodal large language models (MLLMs), such as LISA, often struggle with novel or emerging entities due to their inability to incorporate up-to-date knowledge. To address this challenge, we introduce the Novel Emerging Segmentation Task (NEST), which focuses on segmenting (i) novel entities that MLLMs fail to recognize due to their absence from training data, and (ii) emerging entities that exist within the model's knowledge but demand up-to-date external information for accurate recognition. To support the study of NEST, we construct a NEST benchmark using an automated pipeline that generates news-related data samples for comprehensive evaluation. Additionally, we propose ROSE: Retrieval-Oriented Segmentation Enhancement, a plug-and-play framework designed to augment any MLLM-based segmentation model. ROSE comprises four key components. First, an Internet Retrieval-Augmented Generation module is introduced to employ user-provided multimodal inputs to retrieve real-time web information. Then, a Textual Prompt Enhancer enriches the model with up-to-date information and rich background knowledge, improving the model's perception ability for emerging entities. Furthermore, a Visual Prompt Enhancer is proposed to compensate for MLLMs' lack of exposure to novel entities by leveraging internet-sourced images. To maintain efficiency, a WebSense module is introduced to intelligently decide when to invoke retrieval mechanisms based on user input. Experimental results demonstrate that ROSE significantly boosts performance on the NEST benchmark, outperforming a strong Gemini-2.0 Flash-based retrieval baseline by 19.2 in gIoU.
>
---
#### [new 005] Depth-Aware Image and Video Orientation Estimation
- **分类: cs.CV**

- **简介: 该论文属于图像和视频方向估计任务，旨在解决自然场景中方向准确估计的问题。通过分析深度分布及引入深度梯度一致性与水平对称性分析，提升方向校正的精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.13995](https://arxiv.org/pdf/2604.13995)**

> **作者:** Muhammad Z. Alam; Larry Stetsiuk; M. Umair Mukati; Zeeshan Kaleem
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** This paper introduces a novel approach for image and video orientation estimation by leveraging depth distribution in natural images. The proposed method estimates the orientation based on the depth distribution across different quadrants of the image, providing a robust framework for orientation estimation suited for applications such as virtual reality (VR), augmented reality (AR), autonomous navigation, and interactive surveillance systems. To further enhance fine-scale perceptual alignment, we incorporate depth gradient consistency (DGC) and horizontal symmetry analysis (HSA), enabling precise orientation correction. This hybrid strategy effectively exploits depth cues to support spatial coherence and perceptual stability in immersive visual content. Qualitative and quantitative evaluations demonstrate the robustness and accuracy of the proposed approach, outperforming existing techniques across diverse scenarios.
>
---
#### [new 006] Free Lunch for Unified Multimodal Models: Enhancing Generation via Reflective Rectification with Inherent Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态生成任务，旨在解决统一多模态模型生成能力不足的问题。通过引入反思修正机制，提升模型生成质量。**

- **链接: [https://arxiv.org/pdf/2604.13540](https://arxiv.org/pdf/2604.13540)**

> **作者:** Yibo Jiang; Tao Wu; Rui Jiang; Yehao Lu; Chaoxiang Cai; Zequn Qin; Xi Li
>
> **摘要:** Unified Multimodal Models (UMMs) aim to integrate visual understanding and generation within a single structure. However, these models exhibit a notable capability mismatch, where their understanding capability significantly outperforms their generation. This mismatch indicates that the model's rich internal knowledge, while effective for understanding tasks, remains underactivated during generation. To address this, we draw inspiration from the human ``Thinking-While-Drawing'' paradigm, where humans continuously reflect to activate their knowledge and rectify intermediate results. In this paper, we propose UniRect-CoT, a training-free unified rectification chain-of-thought framework. Our approach unlocks the ``free lunch'' hidden in the UMM's powerful inherent understanding to continuously reflect, activating its internal knowledge and rectifying intermediate results during this http URL regard the diffusion denoising process in UMMs as an intrinsic visual reasoning process and align the intermediate results with the target instruction understood by the model, serving as a self-supervisory signal to rectify UMM this http URL experiments demonstrate that UniRect-CoT can be easily integrated into existing UMMs, significantly enhancing generation quality across diverse complex tasks.
>
---
#### [new 007] Any3DAvatar: Fast and High-Quality Full-Head 3D Avatar Reconstruction from Single Portrait Image
- **分类: cs.CV**

- **简介: 该论文属于单图像3D头像重建任务，解决高质量与速度的矛盾问题。提出Any3DAvatar方法，实现快速且高保真重建。**

- **链接: [https://arxiv.org/pdf/2604.13856](https://arxiv.org/pdf/2604.13856)**

> **作者:** Yujie Gao; Yao Xiao; Xiangnan Zhu; Ya Li; Yiyi Zhang; Liqing Zhang; Jianfu Zhang
>
> **摘要:** Reconstructing a complete 3D head from a single portrait remains challenging because existing methods still face a sharp quality-speed trade-off: high-fidelity pipelines often rely on multi-stage processing and per-subject optimization, while fast feed-forward models struggle with complete geometry and fine appearance details. To bridge this gap, we propose Any3DAvatar, a fast and high-quality method for single-image 3D Gaussian head avatar generation, whose fastest setting reconstructs a full head in under one second while preserving high-fidelity geometry and texture. First, we build AnyHead, a unified data suite that combines identity diversity, dense multi-view supervision, and realistic accessories, filling the main gaps of existing head data in coverage, full-head geometry, and complex appearance. Second, rather than sampling unstructured noise, we initialize from a Plücker-aware structured 3D Gaussian scaffold and perform one-step conditional denoising, formulating full-head reconstruction into a single forward pass while retaining high fidelity. Third, we introduce auxiliary view-conditioned appearance supervision on the same latent tokens alongside 3D Gaussian reconstruction, improving novel-view texture details at zero extra inference cost. Experiments show that Any3DAvatar outperforms prior single-image full-head reconstruction methods in rendering fidelity while remaining substantially faster.
>
---
#### [new 008] SemiFA: An Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis Report Generation
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于半导体故障分析任务，旨在自动化生成结构化FA报告。通过多模态框架SemiFA，融合图像、设备数据与历史记录，提升分析效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.13236](https://arxiv.org/pdf/2604.13236)**

> **作者:** Shivam Chand Kaushik
>
> **备注:** 11 pages, 6 figures, 8 tables. Dataset available at this https URL. Code available at this https URL
>
> **摘要:** Semiconductor failure analysis (FA) requires engineers to examine inspection images, correlate equipment telemetry, consult historical defect records, and write structured reports, a process that can consume several hours of expert time per case. We present SemiFA, an agentic multi-modal framework that autonomously generates structured FA reports from semiconductor inspection images in under one minute. SemiFA decomposes FA into a four-agent LangGraph pipeline: a DefectDescriber that classifies and narrates defect morphology using DINOv2 and LLaVA-1.6, a RootCauseAnalyzer that fuses SECS/GEM equipment telemetry with historically similar defects retrieved from a Qdrant vector database, a SeverityClassifier that assigns severity and estimates yield impact, and a RecipeAdvisor that proposes corrective process adjustments. A fifth node assembles a PDF report. We introduce SemiFA-930, a dataset of 930 annotated semiconductor defect images paired with structured FA narratives across nine defect classes, drawn from procedural synthesis, WM-811K, and MixedWM38. Our DINOv2-based classifier achieves 92.1% accuracy on 140 validation images (macro F1 = 0.917), and the full pipeline produces complete FA reports in 48 seconds on an NVIDIA A100-SXM4-40 GB GPU. A GPT-4o judge ablation across four modality conditions demonstrates that multi-modal fusion improves root cause reasoning by +0.86 composite points (1-5 scale) over an image-only baseline, with equipment telemetry as the more load-bearing modality. To our knowledge, SemiFA is the first system to integrate SECS/GEM equipment telemetry into a vision-language model pipeline for autonomous FA report generation.
>
---
#### [new 009] 3DRealHead: Few-Shot Detailed Head Avatar
- **分类: cs.CV**

- **简介: 该论文提出3DRealHead，解决3D头像重建与表情控制问题。通过少量图像和单目视频生成高保真头像，提升表情表现力。**

- **链接: [https://arxiv.org/pdf/2604.13171](https://arxiv.org/pdf/2604.13171)**

> **作者:** Jalees Nehvi; Timo Bolkart; Thabo Beeler; Justus Thies
>
> **摘要:** The human face is central to communication. For immersive applications, the digital presence of a person should mirror the physical reality, capturing the users idiosyncrasies and detailed facial expressions. However, current 3D head avatar methods often struggle to faithfully reproduce the identity and facial expressions, despite having multi-view data or learned priors. Learning priors that capture the diversity of human appearances, especially, for regions with highly person-specific features, like the mouth and teeth region is challenging as the underlying training data is limited. In addition, many of the avatar methods are purely relying on 3D morphable model-based expression control which strongly limits expressivity. To address these challenges, we are introducing 3DRealHead, a few-shot head avatar reconstruction method with a novel expression control signal that is extracted from a monocular video stream of the subject. Specifically, the subject can take a few pictures of themselves, recover a 3D head avatar and drive it with a consumer-level webcam. The avatar reconstruction is enabled via a novel few-shot inversion process of a 3D human head prior which is represented as a Style U-Net that emits 3D Gaussian primitives which can be rendered under novel views. The prior is learned on the NeRSemble dataset. For animating the avatar, the U-Net is conditioned on 3DMM-based facial expression signals, as well as features of the mouth region extracted from the driving video. These additional mouth features allow us to recover facial expressions that cannot be represented by the 3DMM leading to a higher expressivity and closer resemblance to the physical reality.
>
---
#### [new 010] CausalDisenSeg: A Causality-Guided Disentanglement Framework with Counterfactual Reasoning for Robust Brain Tumor Segmentation Under Missing Modalities
- **分类: cs.CV**

- **简介: 该论文属于脑肿瘤分割任务，解决缺失模态下模型鲁棒性问题。提出CausalDisenSeg框架，通过因果解耦和反事实推理提升分割准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.13409](https://arxiv.org/pdf/2604.13409)**

> **作者:** Bo Liu; Yulong Zou; Jin Hong
>
> **摘要:** In clinical practice, the robustness of deep learning models for multimodal brain tumor segmentation is severely compromised by incomplete MRI data. This vulnerability stems primarily from modality bias, where models exploit spurious correlations as shortcuts rather than learning true anatomical structures. Existing feature fusion methods fail to fundamentally eliminate this dependency. To address this, we propose CausalDisenSeg, a novel Structural Causal Model (SCM)-grounded framework that achieves robust segmentation via causality-guided disentanglement and counterfactual reasoning. We reframe the problem as isolating the anatomical Causal Factor from the stylistic Bias Factor. Our framework implements a three-stage causal intervention: (1) Explicit Causal Disentanglement: A Conditional Variational Autoencoder (CVAE) coupled with an HSIC constraint mathematically enforces statistical orthogonality between anatomical and style features. (2) Causal Representation Reinforcement: A Region Causality Module (RCM) explicitly grounds causal features in physical tumor regions. (3) Counterfactual Reasoning: A dual-adversarial strategy actively suppresses the residual Natural Direct Effect (NDE) of the bias, forcing its spatial attention to be mutually exclusive from the causal path. Extensive experiments on the BraTS 2020 dataset demonstrate that CausalDisenSeg significantly outperforms state-of-the-art methods in accuracy and consistency across severe missing-modality scenarios. Furthermore, cross-dataset evaluation on BraTS 2023 under the same protocol yields a state-of-the-art macro-average DSC of 84.49.
>
---
#### [new 011] UI-Zoomer: Uncertainty-Driven Adaptive Zoom-In for GUI Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于GUI接地任务，解决小图标和密集布局下的定位难题。提出UI-Zoomer，通过不确定性驱动自适应缩放提升定位精度。**

- **链接: [https://arxiv.org/pdf/2604.14113](https://arxiv.org/pdf/2604.14113)**

> **作者:** Fei Tang; Bofan Chen; Zhengxi Lu; Tongbo Chen; Songqin Nong; Tao Jiang; Wenhao Xu; Weiming Lu; Jun Xiao; Yueting Zhuang; Yongliang Shen
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** GUI grounding, which localizes interface elements from screenshots given natural language queries, remains challenging for small icons and dense layouts. Test-time zoom-in methods improve localization by cropping and re-running inference at higher resolution, but apply cropping uniformly across all instances with fixed crop sizes, ignoring whether the model is actually uncertain on each case. We propose \textbf{UI-Zoomer}, a training-free adaptive zoom-in framework that treats both the trigger and scale of zoom-in as a prediction uncertainty quantification problem. A confidence-aware gate fuses spatial consensus among stochastic candidates with token-level generation confidence to selectively trigger zoom-in only when localization is uncertain. When triggered, an uncertainty-driven crop sizing module decomposes prediction variance into inter-sample positional spread and intra-sample box extent, deriving a per-instance crop radius via the law of total variance. Extensive experiments on ScreenSpot-Pro, UI-Vision, and ScreenSpot-v2 demonstrate consistent improvements over strong baselines across multiple model architectures, achieving gains of up to +13.4\%, +10.3\%, and +4.2\% respectively, with no additional training required.
>
---
#### [new 012] Towards Unconstrained Human-Object Interaction
- **分类: cs.CV**

- **简介: 该论文属于人-物交互检测任务，旨在解决传统模型依赖预定义交互列表的问题。通过引入多模态大语言模型，提出无约束HOI检测方法，实现更灵活的交互识别。**

- **链接: [https://arxiv.org/pdf/2604.14069](https://arxiv.org/pdf/2604.14069)**

> **作者:** Francesco Tonini; Alessandro Conti; Lorenzo Vaquero; Cigdem Beyan; Elisa Ricci
>
> **备注:** Accepted to the 20th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2026)
>
> **摘要:** Human-Object Interaction (HOI) detection is a longstanding computer vision problem concerned with predicting the interaction between humans and objects. Current HOI models rely on a vocabulary of interactions at training and inference time, limiting their applicability to static environments. With the advent of Multimodal Large Language Models (MLLMs), it has become feasible to explore more flexible paradigms for interaction recognition. In this work, we revisit HOI detection through the lens of MLLMs and apply them to in-the-wild HOI detection. We define the Unconstrained HOI (U-HOI) task, a novel HOI domain that removes the requirement for a predefined list of interactions at both training and inference. We evaluate a range of MLLMs on this setting and introduce a pipeline that includes test-time inference and language-to-graph conversion to extract structured interactions from free-form text. Our findings highlight the limitations of current HOI detectors and the value of MLLMs for U-HOI. Code will be available at this https URL
>
---
#### [new 013] VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决MLLM缺乏专业伪造知识和推理能力的问题。通过结合检索增强生成与强化学习，提出VRAG-DFD框架，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.13660](https://arxiv.org/pdf/2604.13660)**

> **作者:** Hui Han; Shunli Wang; Yandan Zhao; Taiping Yao; Shouhong Ding
>
> **摘要:** In Deepfake Detection (DFD) tasks, researchers proposed two types of MLLM-based methods: complementary combination with small DFD detectors, or static forgery knowledge this http URL lack of professional forgery knowledge hinders the performance of these this http URL solve this, we deeply considered two insightful issues: How to provide high-quality associated forgery knowledge for MLLMs? AND How to endow MLLMs with critical reasoning abilities given noisy reference information? Notably, we attempted to address above two questions with preliminary answers by leveraging the combination of Retrieval-Augmented Generation (RAG) and Reinforcement Learning (RL).Through RAG and RL techniques, we propose the VRAG-DFD framework with accurate dynamic forgery knowledge retrieval and powerful critical reasoning this http URL, in terms of data, we constructed two datasets with RAG: Forensic Knowledge Database (FKD) for DFD knowledge annotation, and Forensic Chain-of-Thought Dataset (F-CoT), for critical CoT this http URL terms of model training, we adopt a three-stage training method (Alignment->SFT->GRPO) to gradually cultivate the critical reasoning ability of the this http URL terms of performance, VRAG-DFD achieved SOTA and competitive performance on DFD generalization testing.
>
---
#### [new 014] SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态检索任务，旨在解决冻结MLLMs适应检索的问题。通过引入共享潜在查询，保持预训练表示并提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.13710](https://arxiv.org/pdf/2604.13710)**

> **作者:** Haoran Lou; Ziyan Liu; Chunxiao Fan; Yuexin Wu; Yue Ming
>
> **摘要:** Multimodal Large Language Models (MLLMs) exhibit strong reasoning and world knowledge, yet adapting them for retrieval remains challenging. Existing approaches rely on invasive parameter updates, such as full fine-tuning and LoRA, which may disrupt the pre-trained semantic space and impair the structured knowledge essential for reasoning. In this work, we argue that adapting MLLMs for retrieval should focus on eliciting pre-trained representations rather than overwriting them. To this end, we propose SLQ, an effective and efficient framework that adapts a frozen MLLM into a retriever through a small set of Shared Latent Queries. Appended to the end of both text and image token sequences, these queries leverage the model's native causal attention to serve as global aggregation interfaces, producing compact embeddings in a unified space while keeping the backbone unchanged. Furthermore, to better evaluate retrieval beyond superficial pattern matching, we construct KARR-Bench, a benchmark designed for knowledge-aware reasoning retrieval. Extensive experiments show that SLQ outperforms full fine-tuning and LoRA on COCO and Flickr30K, while achieving competitive performance on MMEB and yielding substantial gains on KARR-Bench. The results demonstrate that SLQ, which preserves pre-trained representations, provides an effective and efficient framework for adapting MLLMs to retrieval.
>
---
#### [new 015] SSD-GS: Scattering and Shadow Decomposition for Relightable 3D Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出SSD-GS，解决3D高斯点云的物理光照重建问题，通过分解反射为四个成分提升材质和光照的解耦效果，实现高质量重新光照。**

- **链接: [https://arxiv.org/pdf/2604.13333](https://arxiv.org/pdf/2604.13333)**

> **作者:** Iris Zheng; Guojun Tang; Alexander Doronin; Paul Teal; Fang-Lue Zhang
>
> **备注:** Accepted to ICLR 2026. Code available at: this https URL
>
> **摘要:** We present SSD-GS, a physically-based relighting framework built upon 3D Gaussian Splatting (3DGS) that achieves high-quality reconstruction and photorealistic relighting under novel lighting conditions. In physically-based relighting, accurately modeling light-material interactions is essential for faithful appearance reproduction. However, existing 3DGS-based relighting methods adopt coarse shading decompositions, either modeling only diffuse and specular reflections or relying on neural networks to approximate shadows and scattering. This leads to limited fidelity and poor physical interpretability, particularly for anisotropic metals and translucent materials. To address these limitations, SSD-GS decomposes reflectance into four components: diffuse, specular, shadow, and subsurface scattering. We introduce a learnable dipole-based scattering module for subsurface transport, an occlusion-aware shadow formulation that integrates visibility estimates with a refinement network, and an enhanced specular component with an anisotropic Fresnel-based model. Through progressive integration of all components during training, SSD-GS effectively disentangles lighting and material properties, even for unseen illumination conditions, as demonstrated on the challenging OLAT dataset. Experiments demonstrate superior quantitative and perceptual relighting quality compared to prior methods and pave the way for downstream tasks, including controllable light source editing and interactive scene relighting. The source code is available at: this https URL.
>
---
#### [new 016] ZoomSpec: A Physics-Guided Coarse-to-Fine Framework for Wideband Spectrum Sensing
- **分类: cs.CV**

- **简介: 该论文提出ZoomSpec，用于解决宽频谱感知中的窄带可见性问题。通过物理引导的粗到细框架，提升检测精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.13568](https://arxiv.org/pdf/2604.13568)**

> **作者:** Zhentao Yang; Yixiang Luomei; Zhuoyang Liu; Zhenyu Liu; Feng Xu
>
> **备注:** 14 pages, 8 figures, 5 tables
>
> **摘要:** Wideband spectrum sensing for low-altitude monitoring is critical yet challenging due to heterogeneous protocols,large bandwidths, and non-stationary SNR. Existing data-driven approaches treat spectrograms as natural images,suffering from domain mismatch: they neglect time-frequency resolution constraints and spectral leakage, leading topoor narrowband visibility. This paper proposes ZoomSpec, a physics-guided coarse-to-fine framework integrating signal processing priors with deep learning. We introduce a Log-Space STFT (LS-STFT) to overcome the geometric bottleneck of linear spectrograms, sharpening narrowband structures while maintaining constant relative resolution. A lightweight Coarse Proposal Net (CPN) rapidly screens the full band. To bridge coarse detection and fine recognition, we design an Adaptive Heterodyne Low-Pass (AHLP) module that executes center-frequency aligning, bandwidth-matched filtering, and safe decimation, purifying signals of out-of-band interference. A Fine Recognition Net (FRN) fuses purified time-domain I/Q with spectral magnitude via dual-domain attention to jointly refine temporal boundaries and modulation classification. Evaluations on the SpaceNet real-world dataset demonstrate state-of-the-art 78.1 mAP@0.5:0.95, surpassing existing leaderboard systems with superior stability across diverse modulation bandwidths.
>
---
#### [new 017] Beyond Voxel 3D Editing: Learning from 3D Masks and Self-Constructed Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D编辑任务，旨在解决现有方法在语义一致性和局部不变性上的不足，提出BVE框架及自建数据集，提升文本引导的3D生成质量。**

- **链接: [https://arxiv.org/pdf/2604.13688](https://arxiv.org/pdf/2604.13688)**

> **作者:** Yizhao Xu; Hongyuan Zhu; Caiyun Liu; Tianfu Wang; Keyu Chen; Sicheng Xu; Jiaolong Yang; Nicholas Jing Yuan; Qi Zhang
>
> **摘要:** 3D editing refers to the ability to apply local or global modifications to 3D assets. Effective 3D editing requires maintaining semantic consistency by performing localized changes according to prompts, while also preserving local invariance so that unchanged regions remain consistent with the original. However, existing approaches have significant limitations: multi-view editing methods incur losses when projecting back to 3D, while voxel-based editing is constrained in both the regions that can be modified and the scale of modifications. Moreover, the lack of sufficiently large editing datasets for training and evaluation remains a challenge. To address these challenges, we propose a Beyond Voxel 3D Editing (BVE) framework with a self-constructed large-scale dataset specifically tailored for 3D editing. Building upon this dataset, our model enhances a foundational image-to-3D generative architecture with lightweight, trainable modules, enabling efficient injection of textual semantics without the need for expensive full-model retraining. Furthermore, we introduce an annotation-free 3D masking strategy to preserve local invariance, maintaining the integrity of unchanged regions during editing. Extensive experiments demonstrate that BVE achieves superior performance in generating high-quality, text-aligned 3D assets, while faithfully retaining the visual characteristics of the original input.
>
---
#### [new 018] Towards Patient-Specific Deformable Registration in Laparoscopic Surgery
- **分类: cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决术中与术前模型不匹配的问题。通过引入患者特异性非刚性点云配准方法，提升手术可视化与安全性。**

- **链接: [https://arxiv.org/pdf/2604.13186](https://arxiv.org/pdf/2604.13186)**

> **作者:** Alberto Neri; Veronica Penza; Nazim Haouchine; Leonardo S. Mattos
>
> **摘要:** Unsafe surgical care is a critical health concern, often linked to limitations in surgeon experience, skills, and situational awareness. Integrating patient-specific 3D models into the surgical field can enhance visualization, provide real-time anatomical guidance, and reduce intraoperative complications. However, reliably registering these models in general surgery remains challenging due to mismatches between preoperative and intraoperative organ surfaces, such as deformations and noise. To overcome these challenges, we introduce the first patient-specific non-rigid point cloud registration method, which leverages a novel data generation strategy to optimize outcomes for individual patients. Our approach combines a Transformer encoder-decoder architecture with overlap estimation and a dedicated matching module to predict dense correspondences, followed by a physics-based algorithm for registration. Experimental results on both synthetic and real data demonstrate that our patient-specific method significantly outperforms traditional agnostic approaches, achieving 45% Matching Score with 92% Inlier Ratio on synthetic data, highlighting its potential to improve surgical care.
>
---
#### [new 019] Context Sensitivity Improves Human-Machine Visual Alignment
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉对齐任务，旨在解决机器学习模型与人类认知差异的问题。通过引入上下文敏感的相似性计算，提升奇偶识别任务的准确率。**

- **链接: [https://arxiv.org/pdf/2604.13883](https://arxiv.org/pdf/2604.13883)**

> **作者:** Frieda Born; Tom Neuhäuser; Lukas Muttenthaler; Brett D. Roads; Bernhard Spitzer; Andrew K. Lampinen; Matt Jones; Klaus-Robert Müller; Michael C. Mozer
>
> **摘要:** Modern machine learning models typically represent inputs as fixed points in a high-dimensional embedding space. While this approach has been proven powerful for a wide range of downstream tasks, it fundamentally differs from the way humans process information. Because humans are constantly adapting to their environment, they represent objects and their relationships in a highly context-sensitive manner. To address this gap, we propose a method for context-sensitive similarity computation from neural network embeddings, applied to modeling a triplet odd-one-out task with an anchor image serving as simultaneous context. Modeling context enables us to achieve up to a 15% improvement in odd-one-out accuracy over a context-insensitive model. We find that this improvement is consistent across both original and "human-aligned" vision foundation models.
>
---
#### [new 020] Deep Spatially-Regularized and Superpixel-Based Diffusion Learning for Unsupervised Hyperspectral Image Clustering
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于无监督高光谱图像聚类任务，旨在提升聚类精度。通过结合深度表示学习与扩散聚类，提出DS²DL算法，利用空间正则化和超像素分割优化扩散图构建。**

- **链接: [https://arxiv.org/pdf/2604.13307](https://arxiv.org/pdf/2604.13307)**

> **作者:** Vutichart Buranasiri; James M. Murphy
>
> **备注:** To appear in IEEE IGARSS 2026
>
> **摘要:** An unsupervised framework for hyperspectral image (HSI) clustering is proposed that incorporates masked deep representation learning with diffusion-based clustering, extending the Spatially-Regularized Superpixel-based Diffusion Learning ($S^2DL$) algorithm. Initially, a denoised latent representation of the original HSI is learned via an unsupervised masked autoencoder (UMAE) model with a Vision Transformer backbone. The UMAE takes spatial context and long-range spectral correlations into account and incorporates an efficient pretraining process via masking that utilizes only a small subset of training pixels. In the next stage, the entropy rate superpixel (ERS) algorithm is used to segment the image into superpixels, and a spatially regularized diffusion graph is constructed using Euclidean and diffusion distances within the compressed latent space instead of the HSI space. The proposed algorithm, Deep Spatially-Regularized Superpixel-based Diffusion Learning ($DS^2DL$), leverages more faithful diffusion distances and subsequent diffusion graph construction that better reflect the intrinsic geometry of the underlying data manifold, improving labeling accuracy and clustering quality. Experiments on Botswana and KSC datasets demonstrate the efficacy of $DS^2DL$.
>
---
#### [new 021] Rethinking Image-to-3D Generation with Sparse Queries: Efficiency, Capacity, and Input-View Bias
- **分类: cs.CV**

- **简介: 该论文属于图像到3D生成任务，旨在解决传统方法效率低、输入视图偏差大的问题。提出SparseGen框架，通过稀疏查询和扩展算子实现高效且多视角一致的3D生成。**

- **链接: [https://arxiv.org/pdf/2604.13905](https://arxiv.org/pdf/2604.13905)**

> **作者:** Zhiyuan Xu; Jiuming Liu; Yuxin Chen; Masayoshi Tomizuka; Chenfeng Xu; Chensheng Peng
>
> **备注:** Code is available at this https URL
>
> **摘要:** We present SparseGen, a novel framework for efficient image-to-3D generation, which exhibits low input-view bias while being significantly faster. Unlike traditional approaches that rely on dense volumetric grids, triplanes, or pixel-aligned primitives, we model scenes with a compact sparse set of learned 3D anchor queries and a learned expansion operator that decodes each transformed query into a small local set of 3D Gaussian primitives. Trained under a rectified-flow reconstruction objective without 3D supervision, our model learns to allocate representation capacity where geometry and appearance matter, achieving significant reductions in memory and inference time while preserving multi-view fidelity. We introduce quantitative measures of input-view bias and utilization to show that sparse queries reduce overfitting to conditioning views while being representationally efficient. Our results argue that sparse set-latent expansion is a principled, practical alternative for efficient 3D generative modeling.
>
---
#### [new 022] Towards Successful Implementation of Automated Raveling Detection: Effects of Training Data Size, Illumination Difference, and Spatial Shift
- **分类: cs.CV**

- **简介: 该论文属于道路检测任务，旨在解决raveling检测模型在实际应用中因数据多样性不足导致的性能下降问题。通过分析训练数据量、光照差异和空间位移的影响，提出RavelingArena基准以提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.13322](https://arxiv.org/pdf/2604.13322)**

> **作者:** Xinan Zhang; Haolin Wang; Zhongyu Yang; Yi-Chang; Tsai
>
> **备注:** Accepted and presented in TRBAM 2026
>
> **摘要:** Raveling, the loss of aggregates, is a major form of asphalt pavement surface distress, especially on highways. While research has shown that machine learning and deep learning-based methods yield promising results for raveling detection by classification on range images, their performance often degrades in large-scale deployments where more diverse inference data may originate from different runs, sensors, and environmental conditions. This degradation highlights the need of a more generalizable and robust solution for real-world implementation. Thus, the objectives of this study are to 1) identify and assess potential variations that impact model robustness, such as the quantity of training data, illumination difference, and spatial shift; and 2) leverage findings to enhance model robustness under real-world conditions. To this end, we propose RavelingArena, a benchmark designed to evaluate model robustness to variations in raveling detection. Instead of collecting extensive new data, it is built by augmenting an existing dataset with diverse, controlled variations, thereby enabling variation-controlled experiments to quantify the impact of each variation. Results demonstrate that both the quantity and diversity of training data are critical to the accuracy of models, achieving at least a 9.2% gain in accuracy under the most diverse conditions in experiments. Additionally, a case study applying these findings to a multi-year test section in Georgia, U.S., shows significant improvements in year-to-year consistency, laying foundations for future studies on temporal deterioration modeling. These insights provide guidance for more reliable model deployment in raveling detection and other real-world tasks that require adaptability to diverse conditions.
>
---
#### [new 023] UHR-BAT: Budget-Aware Token Compression Vision-Language model for Ultra-High-Resolution Remote Sensing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UHR-BAT，解决超高分辨率遥感图像中视觉令牌冗余问题，通过区域保留与合并策略实现高效特征提取。属于视觉-语言模型任务。**

- **链接: [https://arxiv.org/pdf/2604.13565](https://arxiv.org/pdf/2604.13565)**

> **作者:** Yunkai Dang; Minxin Dai; Yuekun Yang; Zhangnan Li; Wenbin Li; Feng Miao; Yang Gao
>
> **摘要:** Ultra-high-resolution (UHR) remote sensing imagery couples kilometer-scale context with query-critical evidence that may occupy only a few pixels. Such vast spatial scale leads to a quadratic explosion of visual tokens and hinders the extraction of information from small objects. Previous works utilize direct downsampling, dense tiling, or global top-k pruning, which either compromise query-critical image details or incur unpredictable compute. In this paper, we propose UHR-BAT, a query-guided and region-faithful token compression framework to efficiently select visual tokens under a strict context budget. Specifically, we leverage text-guided, multi-scale importance estimation for visual tokens, effectively tackling the challenge of achieving precise yet low-cost feature extraction. Furthermore, by introducing region-wise preserve and merge strategies, we mitigate visual token redundancy, further driving down the computational budget. Experimental results show that UHR-BAT achieves state-of-the-art performance across various benchmarks. Code will be available at this https URL.
>
---
#### [new 024] Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling
- **分类: cs.CV**

- **简介: 该论文属于MoE模型优化任务，旨在解决专家对称性和早期专业化不足的问题。通过语义聚类和SVD初始化专家，提升模型性能与多样性。**

- **链接: [https://arxiv.org/pdf/2604.13508](https://arxiv.org/pdf/2604.13508)**

> **作者:** Sanghyeok Chu; Pyunghwan Ahn; Gwangmo Song; SeungHwan Kim; Honglak Lee; Bohyung Han
>
> **备注:** Comments: Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Sparse Upcycling provides an efficient way to initialize a Mixture-of-Experts (MoE) model from pretrained dense weights instead of training from scratch. However, since all experts start from identical weights and the router is randomly initialized, the model suffers from expert symmetry and limited early specialization. We propose Cluster-aware Upcycling, a strategy that incorporates semantic structure into MoE initialization. Our method first partitions the dense model's input activations into semantic clusters. Each expert is then initialized using the subspace representations of its corresponding cluster via truncated SVD, while setting the router's initial weights to the cluster centroids. This cluster-aware initialization breaks expert symmetry and encourages early specialization aligned with the data distribution. Furthermore, we introduce an expert-ensemble self-distillation loss that stabilizes training by providing reliable routing guidance using an ensemble teacher. When evaluated on CLIP ViT-B/32 and ViT-B/16, Cluster-aware Upcycling consistently outperforms existing methods across both zero-shot and few-shot benchmarks. The proposed method also produces more diverse and disentangled expert representations, reduces inter-expert similarity, and leads to more confident routing behavior.
>
---
#### [new 025] PAT-VCM: Plug-and-Play Auxiliary Tokens for Video Coding for Machines
- **分类: cs.CV**

- **简介: 该论文提出PAT-VCM框架，解决视频编码与机器任务耦合问题。通过共享压缩流和轻量辅助令牌，提升多任务适应性与效率。属于视频编码领域。**

- **链接: [https://arxiv.org/pdf/2604.13294](https://arxiv.org/pdf/2604.13294)**

> **作者:** Wei Jiang; Wei Wang
>
> **备注:** 15 pages, 3 figures, 13 tables
>
> **摘要:** Existing video coding for machines is often trained for a specific downstream task and model. As a result, the compressed representation becomes tightly coupled to the end task, making it difficult to scale across multiple tasks or adapt to model updates. We propose PAT-VCM, a plug-and-play auxiliary-token framework for video coding for machines. PAT-VCM keeps a shared baseline compressed stream and augments it with lightweight task-aware auxiliary tokens, allowing different downstream tasks to recover the information they need without retraining a separate codec for each task. The framework supports three forms of auxiliary information: visual residual tokens, prompt/control tokens, and semantic tokens. We evaluate PAT-VCM on segmentation, depth estimation, and semantic recognition. A shared detection-oriented auxiliary branch provides a reusable first refinement, task-specific visual branches improve segmentation and depth, prompt tokens provide further segmentation gains at negligible bitrate, and semantic tokens achieve strong recognition performance with extremely low overhead. These results suggest that a shared compressed representation, combined with lightweight task-aware auxiliary tokens, is a practical and scalable alternative to tightly task-coupled VCM design.
>
---
#### [new 026] DiffMagicFace: Identity Consistent Facial Editing of Real Videos
- **分类: cs.CV**

- **简介: 该论文提出DiffMagicFace，解决视频人脸编辑中保持身份一致性的难题。通过融合文本与图像控制模型，实现高质量、连贯的视频编辑。**

- **链接: [https://arxiv.org/pdf/2604.13841](https://arxiv.org/pdf/2604.13841)**

> **作者:** Huanghao Yin; Shenkun Xu; Kanle Shi; Junhai Yong; Bin Wang
>
> **摘要:** Text-conditioned image editing has greatly benefitted from the advancements in Image Diffusion Models. However, extending these techniques to facial video editing introduces challenges in preserving facial identity throughout the source video and ensuring consistency of the edited subject across frames. In this paper, we introduce DiffMagicFace, a unique video editing framework that integrates two fine-tuned models for text and image control. These models operate concurrently during inference to produce video frames that maintain identity features while seamlessly aligning with the editing semantics. To ensure the consistency of the edited videos, we develop a dataset comprising images showcasing various facial perspectives for each edited subject. The creation of a data set is achieved through rendering techniques and the subsequent application of optimization algorithms. Remarkably, our approach does not depend on video datasets but still delivers high-quality results in both consistency and content. The excellent effect holds even for complex tasks like talking head videos and distinguishing closely related categories. The videos edited using our framework exhibit parity with videos that are made using traditional rendering software. Through comparative analysis with current state-of-the-art methods, our framework demonstrates superior performance in both visual appeal and quantitative metrics.
>
---
#### [new 027] Training-Free Semantic Multi-Object Tracking with Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，解决语义跟踪中依赖训练的问题。提出TF-SMOT方法，使用预训练组件实现无需训练的跟踪、摘要生成和交互识别。**

- **链接: [https://arxiv.org/pdf/2604.14074](https://arxiv.org/pdf/2604.14074)**

> **作者:** Laurence Bonat; Francesco Tonini; Elisa Ricci; Lorenzo Vaquero
>
> **备注:** Accepted to the 20th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2026)
>
> **摘要:** Semantic Multi-Object Tracking (SMOT) extends multi-object tracking with semantic outputs such as video summaries, instance-level captions, and interaction labels, aiming to move from trajectories to human-interpretable descriptions of dynamic scenes. Existing SMOT systems are trained end-to-end, coupling progress to expensive supervision, limiting the ability to rapidly adapt to new foundation models and new interactions. We propose TF-SMOT, a training-free SMOT pipeline that composes pretrained components for detection, mask-based tracking, and video-language generation. TF-SMOT combines D-FINE and the promptable SAM2 segmentation tracker to produce temporally consistent tracklets, uses contour grounding to generate video summaries and instance captions with InternVideo2.5, and aligns extracted interaction predicates to BenSMOT WordNet synsets via gloss-based semantic retrieval with LLM disambiguation. On BenSMOT, TF-SMOT achieves state-of-the-art tracking performance within the SMOT setting and improves summary and caption quality compared to prior art. Interaction recognition, however, remains challenging under strict exact-match evaluation on the fine-grained and long-tailed WordNet label space; our analysis and ablations indicate that semantic overlap and label granularity substantially affect measured performance.
>
---
#### [new 028] A Study of Failure Modes in Two-Stage Human-Object Interaction Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人-物交互检测任务，旨在分析两阶段模型的失败模式。通过分解场景配置，研究模型在复杂情况下的失效原因，揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2604.13448](https://arxiv.org/pdf/2604.13448)**

> **作者:** Lemeng Wang; Qinqian Lei; Vidhi Bakshi; Daniel Yi; Yifan Liu; Jiacheng Hou; Asher Seng Hao; Zheda Mai; Wei-Lun Chao; Robby T. Tan; Bo Wang
>
> **备注:** Accepted to SAUAFG Workshop at CVPR 2026
>
> **摘要:** Human-object interaction (HOI) detection aims to detect interactions between humans and objects in images. While recent advances have improved performance on existing benchmarks, their evaluations mainly focus on overall prediction accuracy and provide limited insight into the underlying causes of model failures. In particular, modern models often struggle in complex scenes involving multiple people and rare interaction combinations. In this work, we present a study to better understand the failure modes of two-stage HOI models, which form the basis of many current HOI detection approaches. Rather than constructing a large-scale benchmark, we instead decompose HOI detection into multiple interpretable perspectives and analyze model behavior across these dimensions to study different types of failure patterns. We curate a subset of images from an existing HOI dataset organized by human-object-interaction configurations (e.g., multi-person interactions and object sharing), and analyze model behavior under these configurations to examine different failure modes. This design allows us to analyze how these HOI models behave under different scene compositions and why their predictions fail. Importantly, high overall benchmark performance does not necessarily reflect robust visual reasoning about human-object relationships. We hope that this study can provide useful insights into the limitations of HOI models and offer observations for future research in this area.
>
---
#### [new 029] Dehaze-then-Splat: Generative Dehazing with Physics-Informed 3D Gaussian Splatting for Smoke-Free Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于多视角烟雾去除与新视角合成任务，解决单图像去雾后多视角不一致问题。通过两阶段方法提升合成质量。**

- **链接: [https://arxiv.org/pdf/2604.13589](https://arxiv.org/pdf/2604.13589)**

> **作者:** Yuchao Chen; Hanqing Wang
>
> **摘要:** We present Dehaze-then-Splat, a two-stage pipeline for multi-view smoke removal and novel view synthesis developed for Track~2 of the NTIRE 2026 3D Restoration and Reconstruction Challenge. In the first stage, we produce pseudo-clean training images via per-frame generative dehazing using Nano Banana Pro, followed by brightness normalization. In the second stage, we train 3D Gaussian Splatting (3DGS) with physics-informed auxiliary losses -- depth supervision via Pearson correlation with pseudo-depth, dark channel prior regularization, and dual-source gradient matching -- that compensate for cross-view inconsistencies inherent in frame-wise generative processing. We identify a fundamental tension in dehaze-then-reconstruct pipelines: per-image restoration quality does not guarantee multi-view consistency, and such inconsistency manifests as blurred renders and structural instability in downstream 3D this http URL analysis shows that MCMC-based densification with early stopping, combined with depth and haze-suppression priors, effectively mitigates these artifacts. On the Akikaze validation scene, our pipeline achieves 20.98\,dB PSNR and 0.683 SSIM for novel view synthesis, a +1.50\,dB improvement over the unregularized baseline.
>
---
#### [new 030] The Spectrascapes Dataset: Street-view imagery beyond the visible captured using a mobile platform
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Spectrascapes数据集，解决城市监测数据不足问题，通过多光谱成像获取高质量街景数据，支持城市规划与遥感研究。**

- **链接: [https://arxiv.org/pdf/2604.13315](https://arxiv.org/pdf/2604.13315)**

> **作者:** Akshit Gupta; Joris Timmermans; Filip Biljecki; Remko Uijlenhoet
>
> **备注:** Submitted, under-review
>
> **摘要:** High-resolution data in spatial and temporal contexts is imperative for developing climate resilient cities. Current datasets for monitoring urban parameters are developed primarily using manual inspections, embedded-sensing, remote sensing, or standard street-view imagery (RGB). These methods and datasets are often constrained respectively by poor scalability, inconsistent spatio-temporal resolutions, overhead views or low spectral information. We present a novel method and its open implementation: a multi-spectral terrestrial-view dataset that circumvents these limitations. This dataset consists of 17,718 street level multi-spectral images captured with RGB, Near-infrared, and Thermal imaging sensors on bikes, across diverse urban morphologies (village, town, small city, and big urban area) in the Netherlands. Strict emphasis is put on data calibration and quality while also providing the details of our data collection methodology (including the hardware and software details). To the best of our knowledge, Spectrascapes is the first open-access dataset of its kind. Finally, we demonstrate two downstream use-cases enabled using this dataset and provide potential research directions in the machine learning, urban planning and remote sensing domains.
>
---
#### [new 031] Neural 3D Reconstruction of Planetary Surfaces from Descent-Phase Wide-Angle Imagery
- **分类: cs.CV**

- **简介: 该论文属于行星表面3D重建任务，旨在解决从下降阶段广角影像中准确重建地形的问题。工作包括提出一种基于神经高度场的新方法，提升重建精度和覆盖范围。**

- **链接: [https://arxiv.org/pdf/2604.13235](https://arxiv.org/pdf/2604.13235)**

> **作者:** Melonie de Almeida; George Brydon; Divya M. Persaud; John H. Williamson; Paul Henderson
>
> **摘要:** Digital elevation modeling of planetary surfaces is essential for studying past and ongoing geological processes. Wide-angle imagery acquired during spacecraft descent promises to offer a low-cost option for high-resolution terrain reconstruction. However, accurate 3D reconstruction from such imagery is challenging due to strong radial distortion and limited parallax from vertically descending, predominantly nadir-facing cameras. Conventional multi-view stereo exhibits limited depth range and reduced fidelity under these conditions and also lacks domain-specific priors. We present the first study of modern neural reconstruction methods for planetary descent imaging. We also develop a novel approach that incorporates an explicit neural height field representation, which provides a strong prior since planetary surfaces are generally continuous, smooth, solid, and free from floating objects. This study demonstrates that neural approaches offer a strong and competitive alternative to traditional multi-view stereo (MVS) methods. Experiments on simulated descent sequences over high-fidelity lunar and Mars terrains demonstrate that the proposed approach achieves increased spatial coverage while maintaining satisfactory estimation accuracy.
>
---
#### [new 032] GeoLink: A 3D-Aware Framework Towards Better Generalization in Cross-View Geo-Localization
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于跨视角地理定位任务，旨在解决域迁移下的语义不一致问题。提出GeoLink框架，结合3D结构信息提升2D特征表示，增强跨视角匹配能力。**

- **链接: [https://arxiv.org/pdf/2604.13183](https://arxiv.org/pdf/2604.13183)**

> **作者:** Hongyang Zhang; Yinhao Liu; Haitao Zhang; Zhongyi Wen; Shuxian Liang; Xiansheng Hua
>
> **摘要:** Generalizable cross-view geo-localization aims to match the same location across views in unseen regions and conditions without GPS supervision. Its core difficulty lies in severe semantic inconsistency caused by viewpoint variation and poor generalization under domain shift. Existing methods mainly rely on 2D correspondence, but they are easily distracted by redundant shared information across views, leading to less transferable representations. To address this, we propose GeoLink, a 3D-aware semantic-consistent framework for Generalizable cross-view geo-localization. Specifically, we offline reconstruct scene point clouds from multi-view drone images using VGGT, providing stable structural priors. Based on these 3D anchors, we improve 2D representation learning in two complementary ways. A Geometric-aware Semantic Refinement module mitigates potentially redundant and view-biased dependencies in 2D features under 3D guidance. In addition, a Unified View Relation Distillation module transfers 3D structural relations to 2D features, improving cross-view alignment while preserving a 2D-only inference pipeline. Extensive experiments on multiple benchmarks show that GeoLink consistently outperforms state-of-the-art methods and achieves superior generalization across unseen domains and diverse weather environments.
>
---
#### [new 033] Enhanced Text-to-Image Generation by Fine-grained Multimodal Reasoning
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决统一多模态模型在细粒度控制上的不足。通过引入FiMR框架，实现对提示的细粒度分析与优化，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.13491](https://arxiv.org/pdf/2604.13491)**

> **作者:** Yongjin Kim; Yoonjin Oh; Yerin Kim; Hyomin Kim; Jeeyoung Yun; Yujung Heo; Minjun Kim; Sungwoong Kim
>
> **摘要:** With the rapid progress of Multimodal Large Language Models (MLLMs), unified MLLMs that jointly perform image understanding and generation have advanced significantly. However, despite the inherent reasoning capabilities of unified MLLMs for self-reflection and self-refinement, their use in text-to-image generation remains largely underexplored. Meanwhile, existing multimodal reasoning-based image generation methods mostly rely on holistic image-text alignment judgments, without fine-grained reflection and refinement of detailed prompt attributes, leading to limited fine-grained control. Therefore, we propose Fine-grained Multimodal Reasoning (FiMR), a framework that leverages decomposed visual question answering (VQA) to break down an input prompt into minimal semantic units-such as entities and attributes-and verify each unit via VQA to generate explicit, fine-grained feedback. Based on this feedback, FiMR then applies targeted, localized refinements. This fine-grained self-reasoning and self-refinement enable MLLMs to achieve more precise improvements in image-prompt alignment and overall generation quality at test time. Extensive experiments demonstrate that FiMR consistently outperforms image generation baselines, including reasoning-based methods, particularly on compositional text-to-image benchmarks.
>
---
#### [new 034] A Lightweight Multi-Metric No-Reference Image Quality Assessment Framework for UAV Imaging
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，解决无参考图像质量评估问题。提出MM-IQA框架，结合多种视觉特征生成质量评分，实现快速、轻量化的图像质量筛选。**

- **链接: [https://arxiv.org/pdf/2604.13112](https://arxiv.org/pdf/2604.13112)**

> **作者:** Koffi Titus Sergio Aglin; Anthony K. Muchiri; Celestin Nkundineza
>
> **备注:** 13 pages, 5 figures, article
>
> **摘要:** Reliable image quality assessment is essential in applications where large volumes of images are acquired automatically and must be filtered before further analysis. In many practical scenarios, a pristine reference image is unavailable, making no reference image quality assessment (NR-IQA) particularly important. This paper introduces Multi-Metric Image Quality Assessment (MM-IQA), a lightweight multi-metric framework for NR-IQA. It combines interpretable cues related to blur, edge structure, low resolution artifacts, exposure imbalance, noise, haze, and frequency content to produce a single quality score in the range [0,100].MM-IQA was evaluated on five benchmark datasets (KonIQ-10k, LIVE Challenge, KADID-10k, TID2013, and BIQ2021) and achieved SRCC values ranging from 0.647 to 0.830. Additional experiments on a synthetic agricultural dataset showed consistent behavior of the designed cues. The Python/OpenCV implementation required about 1.97 s per image. This method also has modest memory requirements because it stores only a limited number of intermediate grayscale, filtered, and frequency-domain representations, resulting in memory usage that scales linearly with image size. The results show that MM-IQA can be used for fast image quality screening with explicit distortion aware cues and modest computational cost.
>
---
#### [new 035] Multi-Agent Object Detection Framework Based on Raspberry Pi YOLO Detector and Slack-Ollama Natural Language Interface
- **分类: cs.CV**

- **简介: 论文提出一种基于树莓派的多智能体目标检测框架，整合YOLO视觉代理与自然语言接口，解决边缘计算中资源受限下的多智能体协作问题。**

- **链接: [https://arxiv.org/pdf/2604.13345](https://arxiv.org/pdf/2604.13345)**

> **作者:** Vladimir Kalušev; Branko Brkljač; Milan Brkljač
>
> **备注:** 19 pages, 7 figures, 2 tables, implementation code will be made available upon manuscript publication
>
> **摘要:** The paper presents design and prototype implementation of an edge based object detection system within the new paradigm of AI agents orchestration. It goes beyond traditional design approaches by leveraging on LLM based natural language interface for system control and communication and practically demonstrates integration of all system components into a single resource constrained hardware platform. The method is based on the proposed multi-agent object detection framework which tightly integrates different AI agents within the same task of providing object detection and tracking capabilities. The proposed design principles highlight the fast prototyping approach that is characteristic for transformational potential of generative AI systems, which are applied during both development and implementation stages. Instead of specialized communication and control interface, the system is made by using Slack channel chatbot agent and accompanying Ollama LLM reporting agent, which are both run locally on the same Raspberry Pi platform, alongside the dedicated YOLO based computer vision agent performing real time object detection and tracking. Agent orchestration is implemented through a specially designed event based message exchange subsystem, which represents an alternative to completely autonomous agent orchestration and control characteristic for contemporary LLM based frameworks like the recently proposed OpenClaw. Conducted experimental investigation provides valuable insights into limitations of the low cost testbed platforms in the design of completely centralized multi-agent AI systems. The paper also discusses comparative differences between presented approach and the solution that would require additional cloud based external resources.
>
---
#### [new 036] Physically-Guided Optical Inversion Enable Non-Contact Side-Channel Attack on Isolated Screens
- **分类: cs.CV**

- **简介: 该论文属于安全领域，解决非接触式屏幕内容窃取问题。提出IR4Net模型，应对投影映射不稳定和光传输压缩导致的重建难题，提升攻击精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.13419](https://arxiv.org/pdf/2604.13419)**

> **作者:** Zhiwen Zheng; Yuheng Qiao; Xiaoshuai Zhang; Zhao Huang; Tao Zhang; Huiyu Zhou; Shaowei Jiang; Jin Liu; Wenwen Tang; Xingru Huang
>
> **摘要:** Noncontact exfiltration of electronic screen content poses a security challenge, with side-channel incursions as the principal vector. We introduce an optical projection side-channel paradigm that confronts two core instabilities: (i) the near-singular Jacobian spectrum of projection mapping breaches Hadamard stability, rendering inversion hypersensitive to perturbations; (ii) irreversible compression in light transport obliterates global semantic cues, magnifying reconstruction ambiguity. Exploiting passive speckle patterns formed by diffuse reflection, our Irradiance Robust Radiometric Inversion Network (IR4Net) fuses a Physically Regularized Irradiance Approximation (PRIrr-Approximation), which embeds the radiative transfer equation in a learnable optimizer, with a contour-to-detail cross-scale reconstruction mechanism that arrests noise propagation. Moreover, an Irreversibility Constrained Semantic Reprojection (ICSR) module reinstates lost global structure through context-driven semantic mapping. Evaluated across four scene categories, IR4Net achieves fidelity beyond competing neural approaches while retaining resilience to illumination perturbations.
>
---
#### [new 037] MaMe & MaRe: Matrix-Based Token Merging and Restoration for Efficient Visual Perception and Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MaMe和MaRe方法，用于加速视觉Transformer，解决自注意力机制计算效率低的问题。通过矩阵操作实现高效token合并与恢复，提升模型推理速度。**

- **链接: [https://arxiv.org/pdf/2604.13432](https://arxiv.org/pdf/2604.13432)**

> **作者:** Simin Huo; Ning Li
>
> **备注:** 20 pages. Extended version of CVPR 2026 Findings paper. Neurocomputing (Elsevier) under review
>
> **摘要:** Token compression is crucial for mitigating the quadratic complexity of self-attention mechanisms in Vision Transformers (ViTs), which often involve numerous input tokens. Existing methods, such as ToMe, rely on GPU-inefficient operations (e.g., sorting, scattered writes), introducing overheads that limit their effectiveness. We introduce MaMe, a training-free, differentiable token merging method based entirely on matrix operations, which is GPU-friendly to accelerate ViTs. Additionally, we present MaRe, its inverse operation, for token restoration, forming a MaMe+MaRe pipeline for image synthesis. When applied to pre-trained models, MaMe doubles ViT-B throughput with a 2% accuracy drop. Notably, fine-tuning the last layer with MaMe boosts ViT-B accuracy by 1.0% at 1.1x speed. In SigLIP2-B@512 zero-shot classification, MaMe provides 1.3x acceleration with negligible performance degradation. In video tasks, MaMe accelerates VideoMAE-L by 48.5% on Kinetics-400 with only a 0.84% accuracy loss. Furthermore, MaMe achieves simultaneous improvements in both performance and speed on some tasks. In image synthesis, the MaMe+MaRe pipeline enhances quality while reducing Stable Diffusion v2.1 generation latency by 31%. Collectively, these results demonstrate MaMe's and MaRe's effectiveness in accelerating vision models. The code is available at this https URL}{this https URL.
>
---
#### [new 038] Radar-Informed 3D Multi-Object Tracking under Adverse Conditions
- **分类: cs.CV**

- **简介: 该论文属于3D多目标跟踪任务，旨在提升恶劣条件下的跟踪鲁棒性。提出RadarMOT框架，利用雷达数据增强状态估计与检测补全。**

- **链接: [https://arxiv.org/pdf/2604.13571](https://arxiv.org/pdf/2604.13571)**

> **作者:** Bingxue Xu; Emil Hedemalm; Ajinkya Khoche; Patric Jensfelt
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** The challenge of 3D multi-object tracking (3D MOT) is achieving robustness in real-world applications, for example under adverse conditions and maintaining consistency as distance increases. To overcome these challenges, sensor fusion approaches that combine LiDAR, cameras, and radar have emerged. However, existing multi-modal fusion methods usually treat radar as another learned feature inside the network. When the overall model degrades in difficult environmental conditions, the robustness advantages that radar could provide are also reduced. We propose RadarMOT, a radar-informed 3D MOT framework that explicitly uses radar point cloud data as additional observation to refine state estimation and recover detector misses at long ranges. Evaluations on the MAN-TruckScenes dataset show that RadarMOT consistently improves the Average Multi-Object Tracking Accuracy (AMOTA) with absolute 12.7% at long range and 10.3% in adverse weather. The code will be available at this https URL
>
---
#### [new 039] SEDTalker: Emotion-Aware 3D Facial Animation Using Frame-Level Speech Emotion Diarization
- **分类: cs.CV**

- **简介: 该论文提出SEDTalker，用于语音驱动的3D面部动画生成，解决情绪控制问题。通过帧级情感二分法，实现细腻的情感表达控制。**

- **链接: [https://arxiv.org/pdf/2604.13335](https://arxiv.org/pdf/2604.13335)**

> **作者:** Farzaneh Jafari; Stefano Berretti; Anup Basu
>
> **备注:** 15 pages; 4 figures; conference
>
> **摘要:** We introduce SEDTalker, an emotion-aware framework for speech-driven 3D facial animation that leverages frame-level speech emotion diarization to achieve fine-grained expressive control. Unlike prior approaches that rely on utterance-level or manually specified emotion labels, our method predicts temporally dense emotion categories and intensities directly from speech, enabling continuous modulation of facial expressions over time. The diarized emotion signals are encoded as learned embeddings and used to condition a speech-driven 3D animation model based on a hybrid Transformer-Mamba architecture. This design allows effective disentanglement of linguistic content and emotional style while preserving identity and temporal coherence. We evaluate our approach on a large-scale multi-corpus dataset for speech emotion diarization and on the EmoVOCA dataset for emotional 3D facial animation. Quantitative results demonstrate strong frame-level emotion recognition performance and low geometric and temporal reconstruction errors, while qualitative results show smooth emotion transitions and consistent expression control. These findings highlight the effectiveness of frame-level emotion diarization for expressive and controllable 3D talking head generation.
>
---
#### [new 040] Efficient Multi-View 3D Object Detection by Dynamic Token Selection and Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文属于多视角3D目标检测任务，旨在解决现有方法计算复杂度高和微调参数多的问题。提出动态令牌选择和参数高效微调策略，显著降低计算量并提升效率。**

- **链接: [https://arxiv.org/pdf/2604.13586](https://arxiv.org/pdf/2604.13586)**

> **作者:** Danish Nazir; Antoine Hanna-Asaad; Lucas Görnhardt; Jan Piewek; Thorsten Bagdonat; Tim Fingscheidt
>
> **摘要:** Existing multi-view three-dimensional (3D) object detection approaches widely adopt large-scale pre-trained vision transformer (ViT)-based foundation models as backbones, being computationally complex. To address this problem, current state-of-the-art (SOTA) \texttt{ToC3D} for efficient multi-view ViT-based 3D object detection employs ego-motion-based relevant token selection. However, there are two key limitations: (1) The fixed layer-individual token selection ratios limit computational efficiency during both training and inference. (2) Full end-to-end retraining of the ViT backbone is required for the multi-view 3D object detection method. In this work, we propose an image token compensator combined with a token selection for ViT backbones to accelerate multi-view 3D object detection. Unlike \texttt{ToC3D}, our approach enables dynamic layer-wise token selection within the ViT backbone. Furthermore, we introduce a parameter-efficient fine-tuning strategy, which trains only the proposed modules, thereby reducing the number of fine-tuned parameters from more than $300$ million (M) to only $1.6$ M. Experiments on the large-scale NuScenes dataset across three multi-view 3D object detection approaches demonstrate that our proposed method decreases computational complexity (GFLOPs) by $48\%$ ... $55\%$, inference latency (on an \texttt{NVIDIA-GV100} GPU) by $9\%$ ... $25\%$, while still improving mean average precision by $1.0\%$ ... $2.8\%$ absolute and NuScenes detection score by $0.4\%$ ... $1.2\%$ absolute compared to so-far SOTA \texttt{ToC3D}.
>
---
#### [new 041] CLIP Architecture for Abdominal CT Image-Text Alignment and Zero-Shot Learning: Investigating Batch Composition and Data Scaling
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究3D腹部CT图像与文本对齐的视觉-语言模型，解决零样本诊断问题。通过调整训练批次组成和数据规模，探索最佳训练策略。**

- **链接: [https://arxiv.org/pdf/2604.13561](https://arxiv.org/pdf/2604.13561)**

> **作者:** Shivika; Kartik Bose; Pankaj Gupta
>
> **摘要:** Vision-language models trained with contrastive learning on paired medical images and reports show strong zero-shot diagnostic capabilities, yet the effect of training batch composition on learned representations remains unexplored for 3D medical imaging. We reproduce Merlin, a dual-encoder model that aligns 3D abdominal CT volumes with radiology reports using symmetric InfoNCE loss, achieving a zero-shot macro F1 of 74.45% across 30 findings (original: 73.00%). We then investigate two axes of variation. First, we control the normal-to-abnormal ratio within training batches at 25:75, 50:50, and 75:25 using section-level balanced sampling on the full dataset. All three configurations underperform the unbalanced baseline by 2.4 to 2.8 points, with 75:25 achieving the best result (72.02%) among balanced variants. Second, we conduct data scaling ablations on a 4,362-study subset, training with 20%, 40%, and 100% of the data. Performance scales sub-linearly from 65.26% to 71.88%, with individual findings varying dramatically in data sensitivity. Enforcing 50:50 balanced sampling on the same subset further degrades performance to 68.01%, confirming that explicit class balancing hurts regardless of dataset or balancing granularity. Our results indicate that the stochastic diversity of random sampling, combined with Merlin's alternating batching over anatomical subsections, provides more effective regularization than engineered class ratios at the small batch sizes required by 3D medical volumes.
>
---
#### [new 042] ReConText3D: Replay-based Continual Text-to-3D Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到3D生成任务，解决模型在持续学习中遗忘旧知识的问题。提出ReConText3D框架，通过记忆回放实现新类别增量学习，保持生成质量。**

- **链接: [https://arxiv.org/pdf/2604.13730](https://arxiv.org/pdf/2604.13730)**

> **作者:** Muhammad Ahmed Ullah Khan; Muhammad Haris Bin Amir; Didier Stricker; Muhammad Zeshan Afzal
>
> **备注:** Accepted at CVPR Findings 2026
>
> **摘要:** Continual learning enables models to acquire new knowledge over time while retaining previously learned capabilities. However, its application to text-to-3D generation remains unexplored. We present ReConText3D, the first framework for continual text-to-3D generation. We first demonstrate that existing text-to-3D models suffer from catastrophic forgetting under incremental training. ReConText3D enables generative models to incrementally learn new 3D categories from textual descriptions while preserving the ability to synthesize previously seen assets. Our method constructs a compact and diverse replay memory through text-embedding k-Center selection, allowing representative rehearsal of prior knowledge without modifying the underlying architecture. To systematically evaluate continual text-to-3D learning, we introduce Toys4K-CL, a benchmark derived from the Toys4K dataset that provides balanced and semantically diverse class-incremental splits. Extensive experiments on the Toys4K-CL benchmark show that ReConText3D consistently outperforms all baselines across different generative backbones, maintaining high-quality generation for both old and new classes. To the best of our knowledge, this work establishes the first continual learning framework and benchmark for text-to-3D generation, opening a new direction for incremental 3D generative modeling. Project page is available at: this https URL.
>
---
#### [new 043] Geometric Context Transformer for Streaming 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于实时3D重建任务，解决视频流中相机位姿和点云的准确恢复问题。提出LingBot-Map模型，利用几何上下文Transformer架构实现高效稳定的实时重建。**

- **链接: [https://arxiv.org/pdf/2604.14141](https://arxiv.org/pdf/2604.14141)**

> **作者:** Lin-Zhuo Chen; Jian Gao; Yihang Chen; Ka Leong Cheng; Yipengjing Sun; Liangxiao Hu; Nan Xue; Xing Zhu; Yujun Shen; Yao Yao; Yinghao Xu
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** Streaming 3D reconstruction aims to recover 3D information, such as camera poses and point clouds, from a video stream, which necessitates geometric accuracy, temporal consistency, and computational efficiency. Motivated by the principles of Simultaneous Localization and Mapping (SLAM), we introduce LingBot-Map, a feed-forward 3D foundation model for reconstructing scenes from streaming data, built upon a geometric context transformer (GCT) architecture. A defining aspect of LingBot-Map lies in its carefully designed attention mechanism, which integrates an anchor context, a pose-reference window, and a trajectory memory to address coordinate grounding, dense geometric cues, and long-range drift correction, respectively. This design keeps the streaming state compact while retaining rich geometric context, enabling stable efficient inference at around 20 FPS on 518 x 378 resolution inputs over long sequences exceeding 10,000 frames. Extensive evaluations across a variety of benchmarks demonstrate that our approach achieves superior performance compared to both existing streaming and iterative optimization-based approaches.
>
---
#### [new 044] MSGS: Multispectral 3D Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出MSGS，扩展了3DGS以实现波长感知的视图合成。解决RGB信息不足的问题，通过引入多光谱辐射和双损失优化，提升渲染质量和光谱一致性。**

- **链接: [https://arxiv.org/pdf/2604.13340](https://arxiv.org/pdf/2604.13340)**

> **作者:** Iris Zheng; Guojun Tang; Alexander Doronin; Paul Teal; Fang-Lue Zhang
>
> **备注:** Published in IEEE ISMAR 2025 Adjunct
>
> **摘要:** We present a multispectral extension to 3D Gaussian Splatting (3DGS) for wavelength-aware view synthesis. Each Gaussian is augmented with spectral radiance, represented via per-band spherical harmonics, and optimized under a dual-loss supervision scheme combining RGB and multispectral signals. To improve rendering fidelity, we perform spectral-to-RGB conversion at the pixel level, allowing richer spectral cues to be retained during optimization. Our method is evaluated on both public and self-captured real-world datasets, demonstrating consistent improvements over the RGB-only 3DGS baseline in terms of image quality and spectral consistency. Notably, it excels in challenging scenes involving translucent materials and anisotropic reflections. The proposed approach maintains the compactness and real-time efficiency of 3DGS while laying the foundation for future integration with physically based shading models.
>
---
#### [new 045] DroneScan-YOLO: Redundancy-Aware Lightweight Detection for Tiny Objects in UAV Imagery
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于无人机图像中的小目标检测任务，针对小目标检测难、计算资源受限等问题，提出DroneScan-YOLO模型，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.13278](https://arxiv.org/pdf/2604.13278)**

> **作者:** Yann V. Bellec
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** Aerial object detection in UAV imagery presents unique challenges due to the high prevalence of tiny objects, adverse environmental conditions, and strict computational constraints. Standard YOLO-based detectors fail to address these jointly: their minimum detection stride of 8 pixels renders sub-32px objects nearly undetectable, their CIoU loss produces zero gradients for non-overlapping tiny boxes, and their architectures contain significant filter redundancy. We propose DroneScan-YOLO, a holistic system contribution that addresses these limitations through four coordinated design choices: (1) increased input resolution of 1280x1280 to maximize spatial detail for tiny objects, (2) RPA-Block, a dynamic filter pruning mechanism based on lazy cosine-similarity updates with a 10-epoch warm-up period, (3) MSFD, a lightweight P2 detection branch at stride 4 adding only 114,592 parameters (+1.1%), and (4) SAL-NWD, a hybrid loss combining Normalized Wasserstein Distance with size-adaptive CIoU weighting, integrated into YOLOv8's TaskAligned assignment pipeline. Evaluated on VisDrone2019-DET, DroneScan-YOLO achieves 55.3% mAP@50 and 35.6% mAP@50-95, outperforming the YOLOv8s baseline by +16.6 and +12.3 points respectively, improving recall from 0.374 to 0.518, and maintaining 96.7 FPS inference speed with only +4.1% parameters. Gains are most pronounced on tiny object classes: bicycle AP@50 improves from 0.114 to 0.328 (+187%), and awning-tricycle from 0.156 to 0.237 (+52%).
>
---
#### [new 046] HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出HiVLA系统，解决机器人操作中视觉-语言-动作模型的推理与控制冲突问题。通过分层架构分离语义规划与运动控制，提升长序列任务和细粒度操作性能。**

- **链接: [https://arxiv.org/pdf/2604.14125](https://arxiv.org/pdf/2604.14125)**

> **作者:** Tianshuo Yang; Guanyu Chen; Yutian Chen; Zhixuan Liang; Yitian Liu; Zanxin Chen; Chunpu Xu; Haotian Liang; Jiangmiao Pang; Yao Mu; Ping Luo
>
> **备注:** Project Page: this https URL
>
> **摘要:** While end-to-end Vision-Language-Action (VLA) models offer a promising paradigm for robotic manipulation, fine-tuning them on narrow control data often compromises the profound reasoning capabilities inherited from their base Vision-Language Models (VLMs). To resolve this fundamental trade-off, we propose HiVLA, a visual-grounded-centric hierarchical framework that explicitly decouples high-level semantic planning from low-level motor control. In high-level part, a VLM planner first performs task decomposition and visual grounding to generate structured plans, comprising a subtask instruction and a precise target bounding box. Then, to translate this plan into physical actions, we introduce a flow-matching Diffusion Transformer (DiT) action expert in low-level part equipped with a novel cascaded cross-attention mechanism. This design sequentially fuses global context, high-resolution object-centric crops and skill semantics, enabling the DiT to focus purely on robust execution. Our decoupled architecture preserves the VLM's zero-shot reasoning while allowing independent improvement of both components. Extensive experiments in simulation and the real world demonstrate that HiVLA significantly outperforms state-of-the-art end-to-end baselines, particularly excelling in long-horizon skill composition and the fine-grained manipulation of small objects in cluttered scenes.
>
---
#### [new 047] Why MLLMs Struggle to Determine Object Orientations
- **分类: cs.CV**

- **简介: 该论文研究MLLM在2D物体方向判断上的困难，通过实验发现编码器能恢复方向信息，但MLLM可能未能有效利用。**

- **链接: [https://arxiv.org/pdf/2604.13321](https://arxiv.org/pdf/2604.13321)**

> **作者:** Anju Gopinath; Nikhil Krishnaswamy; Bruce Draper
>
> **摘要:** Multimodal Large Language Models (MLLMs) struggle with tasks that require reasoning about 2D object orientation in images, as documented in prior work. Tong et al. and Nichols et al. hypothesize that these failures originate in the visual encoder, since commonly used encoders such as CLIP and SigLIP are trained for image-text semantic alignment rather than geometric reasoning. We design a controlled empirical protocol to test this claim by measuring whether rotations can be recovered from encoder representations. In particular, we examine SigLIP and ViT features from LLaVA OneVision and Qwen2.5-VL-7B-Instruct models, respectively, using full images, and examine CLIP representations in LLaVA 1.5 and 1.6 using rotated foreground patches against natural background images. Our null hypothesis is that orientation information is not preserved in the encoder embeddings and we test this by training linear regressors to predict object orientation from encoded features. Contrary to the hypothesis, we find that orientation information is recoverable from encoder representations: simple linear models accurately predict object orientations from embeddings. This contradicts the assumption that MLLM orientation failures originate in the visual encoder. Having rejected the accepted hypothesis that MLLMs struggle with 2D orientation tasks because of visual encoder limitations, we still don't know why they fail. Although a full explanation is beyond the scope of this paper, we show that although present, orientation information is spread diffusely across tens of thousands of features. This may or may not be while MLLMs fail to exploit the available orientation information.
>
---
#### [new 048] VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation
- **分类: cs.CV**

- **简介: 该论文聚焦于跨视角实例分割任务，解决因尺度、视角和遮挡导致的像素匹配不稳定问题。提出VGGT-Segmentor框架，结合几何建模与精确分割，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.13596](https://arxiv.org/pdf/2604.13596)**

> **作者:** Yulu Gao; Bohao Zhang; Zongheng Tang; Jitong Liao; Wenjun Wu; Si Liu
>
> **摘要:** Instance-level object segmentation across disparate egocentric and exocentric views is a fundamental challenge in visual understanding, critical for applications in embodied AI and remote collaboration. This task is exceptionally difficult due to severe changes in scale, perspective, and occlusion, which destabilize direct pixel-level matching. While recent geometry-aware models like VGGT provide a strong foundation for feature alignment, we find they often fail at dense prediction tasks due to significant pixel-level projection drift, even when their internal object-level textntion remains consistent. To bridge this gap, we introduce VGGT-Segmentor (VGGT-S), a framework that unifies robust geometric modeling with pixel-accurate semantic segmentation. VGGT-S leverages VGGT's powerful cross-view feature representation and introduces a novel Union Segmentation Head. This head operates in three stages: mask prompt fusion, point-guided prediction, and iterative mask refinement, effectively translating high-level feature alignment into a precise segmentation mask. Furthermore, we propose a single-image self-supervised training strategy that eliminates the need for paired annotations and enables strong generalization. On the Ego-Exo4D benchmark, VGGT-S sets a new state-of-the-art, achieving 67.7% and 68.0% average IoU for Ego to Exo and Exo to Ego tasks, respectively, significantly outperforming prior methods. Notably, our correspondence-free pretrained model surpasses most fully-supervised baselines, demonstrating the effectiveness and scalability of our approach.
>
---
#### [new 049] SocialMirror: Reconstructing 3D Human Interaction Behaviors from Monocular Videos with Semantic and Geometric Guidance
- **分类: cs.CV**

- **简介: 该论文属于3D人体行为重建任务，旨在解决单目视频中近距离交互场景下的遮挡问题。通过融合语义与几何信息，提出SocialMirror框架，提升重建精度与连贯性。**

- **链接: [https://arxiv.org/pdf/2604.13581](https://arxiv.org/pdf/2604.13581)**

> **作者:** Qi Xia; Peishan Cong; Ziyi Wang; Yujing Sun; Qin Sun; Xinge Zhu; Mao Ye; Ruigang Yang; Yuexin Ma
>
> **摘要:** Accurately reconstructing human behavior in close-interaction scenarios is crucial for enabling realistic virtual interactions in augmented reality, precise motion analysis in sports, and natural collaborative behavior in human-robot tasks. Reliable reconstruction in these contexts significantly enhances the realism and effectiveness of AI-driven interactive applications. However, human reconstruction from monocular videos in close-interaction scenarios remains challenging due to severe mutual occlusions, leading local motion ambiguity, disrupted temporal continuity and spatial relationship error. In this paper, we propose SocialMirror, a diffusion-based framework that integrates semantic and geometric cues to effectively address these issues. Specifically, we first leverage high-level interaction descriptions generated by a vision-language model to guide a semantic-guided motion infiller, hallucinating occluded bodies and resolving local pose ambiguities. Next, we propose a sequence-level temporal refiner that enforces smooth, jitter-free motions, while incorporating geometric constraints during sampling to ensure plausible contact and spatial relationships. Evaluations on multiple interaction benchmarks show that SocialMirror achieves state-of-the-art performance in reconstructing interactive human meshes, demonstrating strong generalization across unseen datasets and in-the-wild scenarios. The code will be released upon publication.
>
---
#### [new 050] DRG-Font: Dynamic Reference-Guided Few-shot Font Generation via Contrastive Style-Content Disentanglement
- **分类: cs.CV**

- **简介: 该论文属于字体生成任务，旨在解决少样本下保持字形局部特征的问题。提出DRG-Font方法，通过分解风格与内容空间实现高效字体生成。**

- **链接: [https://arxiv.org/pdf/2604.13797](https://arxiv.org/pdf/2604.13797)**

> **作者:** Rejoy Chakraborty; Prasun Roy; Saumik Bhattacharya; Umapada Pal
>
> **备注:** 11 pages
>
> **摘要:** Few-shot Font Generation aims to generate stylistically consistent glyphs from a few reference glyphs. However, capturing complex font styles from a few exemplars remains challenging, and the existing methods often struggle to retain discernible local characteristics in generated samples. This paper introduces DRG-Font, a contrastive font generation strategy that learns complex glyph attributes by decomposing style and content embedding spaces. For optimal style supervision, the proposed architecture incorporates a Reference Selection (RS) Module to dynamically select the best style reference from an available pool of candidates. The network learns to decompose glyph attributes into style and shape priors through a Multi-scale Style Head Block (MSHB) and a Multi-scale Content Head Block (MCHB). For style adaptation, a Multi-Fusion Upsampling Block (MFUB) produces the target glyph by combining the reference style prior and target content prior. The proposed method demonstrates significant improvements over state-of-the-art approaches across multiple visual and analytical benchmarks.
>
---
#### [new 051] Blind Bitstream-corrupted Video Recovery via Metadata-guided Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于视频恢复任务，解决无预设掩码的比特流损坏视频恢复问题。提出M-GDM模型，利用元数据引导扩散过程，实现损坏区域识别与内容恢复。**

- **链接: [https://arxiv.org/pdf/2604.13906](https://arxiv.org/pdf/2604.13906)**

> **作者:** Shuyun Wang; Hu Zhang; Xin Shen; Dadong Wang; Xin Yu
>
> **备注:** CVPR 2025
>
> **摘要:** Bitstream-corrupted video recovery aims to restore realistic content degraded during video storage or transmission. Existing methods typically assume that predefined masks of corrupted regions are available, but manually annotating these masks is labor-intensive and impractical in real-world scenarios. To address this limitation, we introduce a new blind video recovery setting that removes the reliance on predefined masks. This setting presents two major challenges: accurately identifying corrupted regions and recovering content from extensive and irregular degradations. We propose a Metadata-Guided Diffusion Model (M-GDM) to tackle these challenges. Specifically, intrinsic video metadata are leveraged as corruption indicators through a dual-stream metadata encoder that separately embeds motion vectors and frame types before fusing them into a unified representation. This representation interacts with corrupted latent features via cross-attention at each diffusion step. To preserve intact regions, we design a prior-driven mask predictor that generates pseudo masks using both metadata and diffusion priors, enabling the separation and recombination of intact and recovered regions through hard masking. To mitigate boundary artifacts caused by imperfect masks, a post-refinement module enhances consistency between intact and recovered regions. Extensive experiments demonstrate the effectiveness of our method and its superiority in blind video recovery. Code is available at: this https URL.
>
---
#### [new 052] PostureObjectstitch: Anomaly Image Generation Considering Assembly Relationships in Industrial Scenarios
- **分类: cs.CV**

- **简介: 该论文属于工业异常检测任务，解决装配关系忽略导致生成图像无效的问题。提出PostureObjectStitch方法，通过特征解耦和条件损失提升生成图像的语义与几何准确性。**

- **链接: [https://arxiv.org/pdf/2604.13863](https://arxiv.org/pdf/2604.13863)**

> **作者:** Zebei Tong; Hongchang Chen; Yujie Lei; Gang Chen; Yushi Liu; Zhi Zheng; Hao Chen; Jieming Zhang; Ying Li; Dongpu Cao
>
> **摘要:** Image generation technology can synthesize condition-specific images to supplement real-world industrial anomaly data and enhance anomaly detection model performance. Existing generation techniques rarely account for the pose and orientation of industrial components in assembly, making the generated images difficult to utilize for downstream application. To solve this, we propose a novel image synthesis approach, called PostureObjectStitch, that achieves accurate generation to meet the requirement of industrial assembly. A condition decoupling approach is introduced to separate input multi-view images into high-frequency, texture, and RGB features. The feature temporal modulation mechanism adapts these features across diffusion model time-steps, enabling progressive generation from coarse to fine details while maintaining consistency. To ensure semantic accuracy, we introduce a conditional loss that enhances critical industrial elements and a geometric prior that guides component positioning for correct assembly relationships. Comprehensive experimental results on the MureCom dataset, our newly contributed DreamAssembly dataset, and the downstream application validate the outstanding performance of our method.
>
---
#### [new 053] VibeFlow: Versatile Video Chroma-Lux Editing through Self-Supervised Learning
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，旨在解决视频色彩与光照修改中的结构和时间一致性问题。通过自监督学习，VibeFlow实现高效、无监督的视频色彩-光照编辑。**

- **链接: [https://arxiv.org/pdf/2604.13425](https://arxiv.org/pdf/2604.13425)**

> **作者:** Yifan Li; Pei Cheng; Bin Fu; Shuai Yang; Jiaying Liu
>
> **摘要:** Video chroma-lux editing, which aims to modify illumination and color while preserving structural and temporal fidelity, remains a significant challenge. Existing methods typically rely on expensive supervised training with synthetic paired data. This paper proposes VibeFlow, a novel self-supervised framework that unleashes the intrinsic physical understanding of pre-trained video generation models. Instead of learning color and light transitions from scratch, we introduce a disentangled data perturbation pipeline that enforces the model to adaptively recombine structure from source videos and color-illumination cues from reference images, enabling robust disentanglement in a self-supervised manner. Furthermore, to rectify discretization errors inherent in flow-based models, we introduce Residual Velocity Fields alongside a Structural Distortion Consistency Regularization, ensuring rigorous structural preservation and temporal coherence. Our framework eliminates the need for costly training resources and generalizes in a zero-shot manner to diverse applications, including video relighting, recoloring, low-light enhancement, day-night translation, and object-specific color editing. Extensive experiments demonstrate that VibeFlow achieves impressive visual quality with significantly reduced computational overhead. Our project is publicly available at this https URL.
>
---
#### [new 054] PBE-UNet: A light weight Progressive Boundary-Enhanced U-Net with Scale-Aware Aggregation for Ultrasound Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决超声图像中病灶边界模糊和尺度变化带来的分割难题。提出PBE-UNet模型，引入SAAM和BGFE模块提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.13791](https://arxiv.org/pdf/2604.13791)**

> **作者:** Chen Wang; Yixin Zhu; Yongbin Zhu; Fengyuan Shi; Qi Li; Jun Wang; Zuozhu Liu; Keli Hu
>
> **备注:** 14 pages, 14 figures
>
> **摘要:** Accurate lesion segmentation in ultrasound images is essential for preventive screening and clinical diagnosis, yet remains challenging due to low contrast, blurry boundaries, and significant scale variations. Although existing deep learning-based methods have achieved remarkable performance, these methods still struggle with scale variations and indistinct tumor boundaries. To address these challenges, we propose a progressive boundary enhanced U-Net (PBE-UNet). Specially, we first introduce a scale-aware aggregation module (SAAM) that dynamically adjusts its receptive field to capture robust multi-scale contextual information. Then, we propose a boundary-guided feature enhancement (BGFE) module to enhance the feature representations. We find that there are large gaps between the narrow boundary and the wide segmentation error areas. Unlike existing methods that treat boundaries as static masks, the BGFE module progressively expands the narrow boundary prediction into broader spatial attention maps. Thus, broader spatial attention maps could effectively cover the wider segmentation error regions and enhance the model's focus on these challenging areas. We conduct expensive experiments on four benchmark ultrasound datasets, BUSI, Dataset B, TN3K, and BP. The experimental results how that our proposed PBE-UNet outperforms state-of-the-art ultrasound image segmentation methods. The code is at this https URL.
>
---
#### [new 055] Graph Propagated Projection Unlearning: A Unified Framework for Vision and Audio Discriminative Models
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出GPPU，解决深度学习模型中类级信息擦除问题，通过图传播和投影实现高效、可逆的模型遗忘，适用于视觉和音频任务。**

- **链接: [https://arxiv.org/pdf/2604.13127](https://arxiv.org/pdf/2604.13127)**

> **作者:** Shreyansh Pathak; Jyotishman Das
>
> **摘要:** The need to selectively and efficiently erase learned information from deep neural networks is becoming increasingly important for privacy, regulatory compliance, and adaptive system design. We introduce Graph-Propagated Projection Unlearning (GPPU), a unified and scalable algorithm for class-level unlearning that operates across both vision and audio models. GPPU employs graph-based propagation to identify class-specific directions in the feature space and projects representations onto the orthogonal subspace, followed by targeted fine-tuning, to ensure that target class information is effectively and irreversibly removed. Through comprehensive evaluations on six vision datasets and two large-scale audio benchmarks spanning a variety of architectures including CNNs, Vision Transformers, and Audio Transformers, we demonstrate that GPPU achieves highly efficient unlearning, realizing 10-20x speedups over prior methodologies while preserving model utility on retained classes. Our framework provides a principled and modality-agnostic approach to machine unlearning, evaluated at a scale that has received limited attention in prior work, contributing toward more efficient and responsible deep learning.
>
---
#### [new 056] A Multimodal Clinically Informed Coarse-to-Fine Framework for Longitudinal CT Registration in Proton Therapy
- **分类: cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决质子治疗中纵向CT图像的准确配准问题。通过融合多模态临床信息，提出一种快速且临床相关的粗到细框架。**

- **链接: [https://arxiv.org/pdf/2604.13397](https://arxiv.org/pdf/2604.13397)**

> **作者:** Caiwen Jiang; Yuzhen Ding; Mi Jia; Samir H. Patel; Terence T. Sio; Jonathan B. Ashman; Lisa A. McGee; Jean-Claude M. Rwigema; William G. Rule; Sameer R. Keole; Sujay A. Vora; William W. Wong; Nathan Y. Yu; Michele Y. Halyard; Steven E. Schild; Dinggang Shen; Wei Liu
>
> **摘要:** Proton therapy offers superior organ-at-risk sparing but is highly sensitive to anatomical changes, making accurate deformable image registration (DIR) across longitudinal CT scans essential. Conventional DIR methods are often too slow for emerging online adaptive workflows, while existing deep learning-based approaches are primarily designed for generic benchmarks and underutilize clinically relevant information beyond images. To address this gap, we propose a clinically scalable coarse-to-fine deformable registration framework that integrates multimodal information from the proton radiotherapy workflow to accommodate diverse clinical scenarios. The model employs dual CNN-based encoders for hierarchical feature extraction and a transformer-based decoder to progressively refine deformation fields. Beyond CT intensities, clinically critical priors, including target and organ-at-risk contours, dose distributions, and treatment planning text, are incorporated through anatomy- and risk-guided attention, text-conditioned feature modulation, and foreground-aware optimization, enabling anatomically focused and clinically informed deformation estimation. We evaluate the proposed framework on a large-scale proton therapy DIR dataset comprising 1,222 paired planning and repeat CT scans across multiple anatomical regions and disease types. Extensive experiments demonstrate consistent improvements over state-of-the-art methods, enabling fast and robust clinically meaningful registration.
>
---
#### [new 057] A 3D SAM-Based Progressive Prompting Framework for Multi-Task Segmentation of Radiotherapy-induced Normal Tissue Injuries in Limited-Data Settings
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决放射治疗后正常组织损伤的多任务分割问题。针对数据有限和病变多样性，提出一种基于3D SAM的渐进提示框架，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.13367](https://arxiv.org/pdf/2604.13367)**

> **作者:** Caiwen Jiang; Lei Zeng; Wei Liu
>
> **摘要:** Radiotherapy-induced normal tissue injury is a clinically important complication, and accurate segmentation of injury regions from medical images could facilitate disease assessment, treatment planning, and longitudinal monitoring. However, automatic segmentation of these lesions remains largely unexplored because of limited voxel-level annotations and substantial heterogeneity across injury types, lesion size, and imaging modality. To address this gap, we curate a dedicated head-and-neck radiotherapy-induced normal tissue injury dataset covering three manifestations: osteoradionecrosis (ORN), cerebral edema (CE), and cerebral radiation necrosis (CRN). We further propose a 3D SAM-based progressive prompting framework for multi-task segmentation in limited-data settings. The framework progressively incorporates three complementary prompts: text prompts for task-aware adaptation, dose-guided box prompts for coarse localization, and click prompts for iterative refinement. A small-target focus loss is introduced to improve local prediction and boundary delineation for small and sparse lesions. Experiments on ORN, CE, and CRN demonstrate that the proposed method achieves reliable segmentation performance across diverse injury types and outperforms state-of-the-art methods.
>
---
#### [new 058] POINTS-Seeker: Towards Training a Multimodal Agentic Search Model from Scratch
- **分类: cs.CV**

- **简介: 该论文属于多模态搜索任务，旨在解决长序列交互中证据定位困难的问题。通过构建从零开始的多模态代理搜索模型，提出Agentic Seeding和V-Fold方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.14029](https://arxiv.org/pdf/2604.14029)**

> **作者:** Yikun Liu; Yuan Liu; Le Tian; Xiao Zhou; Jiangchao Yao; Yanfeng Wang; Weidi Xie
>
> **摘要:** While Large Multimodal Models (LMMs) demonstrate impressive visual perception, they remain epistemically constrained by their static parametric knowledge. To transcend these boundaries, multimodal search models have been adopted to actively interact with the external environment for evidence retrieval. Diverging from prevailing paradigms that merely retrofit general LMMs with search tools as modular extensions, we explore the potential of building a multimodal agentic search model from scratch. Specifically, we make the following contributions: (i) we introduce Agentic Seeding, a dedicated phase designed to weave the foundational precursors necessary for eliciting agentic behaviors; (ii) we uncover a performance bottleneck in long-horizon interactions, where the increasing volume of interaction history overwhelms the model's ability to locate ground-truth evidence. To mitigate this, we propose V-Fold, an adaptive history-aware compression scheme that preserves recent dialogue turns in high fidelity while folding historical context into the visual space via rendering; and (iii) we develop POINTS-Seeker-8B, a state-of-the-art multimodal agentic search model that consistently outperforms existing models across six diverse benchmarks, effectively resolving the challenges of long-horizon, knowledge-intensive visual reasoning.
>
---
#### [new 059] PartNerFace: Part-based Neural Radiance Fields for Animatable Facial Avatar Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于面部动画重建任务，旨在解决单目视频生成可动画化人脸模型的问题。提出PartNerFace方法，通过分部件建模实现更精细的面部运动捕捉与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.13918](https://arxiv.org/pdf/2604.13918)**

> **作者:** Xianggang Yu; Lingteng Qiu; Xiaohang Ren; Guanying Chen; Shuguang Cui; Xiaoguang Han; Baoyuan Wang
>
> **摘要:** We present PartNerFace, a part-based neural radiance fields approach, for reconstructing animatable facial avatar from monocular RGB videos. Existing solutions either simply condition the implicit network with the morphable model parameters or learn an imaginary canonical radiance field, making them fail to generalize to unseen facial expressions and capture fine-scale motion details. To address these challenges, we first apply inverse skinning based on a parametric head model to map an observed point to the canonical space, and then model fine-scale motions with a part-based deformation field. Our key insight is that the deformation of different facial parts should be modeled differently. Specifically, our part-based deformation field consists of multiple local MLPs to adaptively partition the canonical space into different parts, where the deformation of a 3D point is computed by aggregating the prediction of all local MLPs by a soft-weighting mechanism. Extensive experiments demonstrate that our method generalizes well to unseen expressions and is capable of modeling fine-scale facial motions, outperforming state-of-the-art methods both quantitatively and qualitatively.
>
---
#### [new 060] Bias at the End of the Score
- **分类: cs.CV**

- **简介: 该论文研究奖励模型在文本到图像生成中的偏见问题，揭示其编码性别和种族刻板印象，影响生成结果的公平性。任务为模型公平性分析，解决奖励模型潜在偏见问题。**

- **链接: [https://arxiv.org/pdf/2604.13305](https://arxiv.org/pdf/2604.13305)**

> **作者:** Salma Abdel Magid; Grace Guo; Esin Tureci; Amaya Dharmasiri; Vikram V. Ramaswamy; Hanspeter Pfister; Olga Russakovsky
>
> **备注:** Accepted to The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Reward models (RMs) are inherently non-neutral value functions designed and trained to encode specific objectives, such as human preferences or text-image alignment. RMs have become crucial components of text-to-image (T2I) generation systems where they are used at various stages for dataset filtering, as evaluation metrics, as a supervisory signal during optimization of parameters, and for post-generation safety and quality filtering of T2I outputs. While specific problems with the integration of RMs into the T2I pipeline have been studied (e.g. reward hacking or mode collapse), their robustness and fairness as scoring functions remains largely unknown. We conduct a large scale audit of RM robustness with respect to demographic biases during T2I model training and generation. We provide quantitative and qualitative evidence that while originally developed as quality measures, RMs encode demographic biases, which cause reward-guided optimization to disproportionately sexualize female image subjects reinforce gender/racial stereotypes, and collapse demographic diversity. These findings highlight shortcomings in current reward models, challenge their reliability as quality metrics, and underscore the need for improved data collection and training procedures to enable more robust scoring.
>
---
#### [new 061] Multitasking Embedding for Embryo Blastocyst Grading Prediction (MEmEBG)
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于胚胎质量评估任务，旨在解决传统人工评估主观性强的问题。通过多任务嵌入方法，自动识别并预测胚胎的多个关键结构。**

- **链接: [https://arxiv.org/pdf/2604.13217](https://arxiv.org/pdf/2604.13217)**

> **作者:** Nahid Khoshk Angabini; Mohsen Tajgardan; Mahesh Madhavan; Zahra Asghari Varzaneh; Reza Khoshkangini; Thomas Ebner
>
> **摘要:** Reliable evaluation of blastocyst quality is critical for the success of in vitro fertilization (IVF) treatments. Current embryo grading practices primarily rely on visual assessment of morphological features, which introduces subjectivity, inter-embryologist variability, and challenges in standardizing quality assurance. In this study, we propose a multitask embedding-based approach for the automated analysis and prediction of key blastocyst components, including the trophectoderm (TE), inner cell mass (ICM), and blastocyst expansion (EXP). The method leverages biological and physical characteristics extracted from images of day-5 human embryos. A pretrained ResNet-18 architecture, enhanced with an embedding layer, is employed to learn discriminative representations from a limited dataset and to automatically identify TE and ICM regions along with their corresponding grades, structures that are visually similar and inherently difficult to distinguish. Experimental results demonstrate the promise of the multitask embedding approach and potential for robust and consistent blastocyst quality assessment.
>
---
#### [new 062] ASTRA: Enhancing Multi-Subject Generation with Retrieval-Augmented Pose Guidance and Disentangled Position Embedding
- **分类: cs.CV**

- **简介: 该论文属于多主体图像生成任务，旨在解决多主体身份融合与姿态扭曲问题。提出ASTRA框架，通过检索增强姿态引导和解耦位置嵌入实现身份与姿态的分离生成。**

- **链接: [https://arxiv.org/pdf/2604.13938](https://arxiv.org/pdf/2604.13938)**

> **作者:** Tianze Xia; Zijian Ning; Zonglin Zhao; Mingjia Wang
>
> **摘要:** Subject-driven image generation has shown great success in creating personalized content, but its capabilities are largely confined to single subjects in common poses. Current approaches face a fundamental conflict when handling multiple subjects with complex, distinct actions: preserving individual identities while enforcing precise pose structures. This challenge often leads to identity fusion and pose distortion, as appearance and structure signals become entangled within the model's architecture. To resolve this conflict, we introduce ASTRA(Adaptive Synthesis through Targeted Retrieval Augmentation), a novel framework that architecturally disentangles subject appearance from pose structure within a unified Diffusion Transformer. ASTRA achieves this through a dual-pronged strategy. It first employs a Retrieval-Augmented Pose (RAG-Pose) pipeline to provide a clean, explicit structural prior from a curated database. Then, its core generative model learns to process these dual visual conditions using our Enhanced Universal Rotary Position Embedding (EURoPE), an asymmetric encoding mechanism that decouples identity tokens from spatial locations while binding pose tokens to the canvas. Concurrently, a Disentangled Semantic Modulation (DSM) adapter offloads the identity preservation task into the text conditioning stream. Extensive experiments demonstrate that our integrated approach achieves superior disentanglement. On our designed COCO-based complex pose benchmark, ASTRA achieves a new state-of-the-art in pose adherence, while maintaining high identity fidelity and text alignment in DreamBench.
>
---
#### [new 063] Design and Behavior of Sparse Mixture-of-Experts Layers in CNN-based Semantic Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究将稀疏混合专家层应用于CNN的语义分割任务，解决如何高效提升模型性能的问题。通过设计粗粒度的专家路由机制，实现性能提升且计算开销小。**

- **链接: [https://arxiv.org/pdf/2604.13761](https://arxiv.org/pdf/2604.13761)**

> **作者:** Svetlana Pavlitska; Haixi Fan; Konstantin Ditschuneit; J. Marius Zöllner
>
> **备注:** Accepted for publication at the SAIAD workshop at CVPR 2026
>
> **摘要:** Sparse mixture-of-experts (MoE) layers have been shown to substantially increase model capacity without a proportional increase in computational cost and are widely used in transformer architectures, where they typically replace feed-forward network blocks. In contrast, integrating sparse MoE layers into convolutional neural networks (CNNs) remains inconsistent, with most prior work focusing on fine-grained MoEs operating at the filter or channel levels. In this work, we investigate a coarser, patch-wise formulation of sparse MoE layers for semantic segmentation, where local regions are routed to a small subset of convolutional experts. Through experiments on the Cityscapes and BDD100K datasets using encoder-decoder and backbone-based CNNs, we conduct a design analysis to assess how architectural choices affect routing dynamics and expert specialization. Our results demonstrate consistent, architecture-dependent improvements (up to +3.9 mIoU) with little computational overhead, while revealing strong design sensitivity. Our work provides empirical insights into the design and internal dynamics of sparse MoE layers in CNN-based dense prediction. Our code is available at this https URL.
>
---
#### [new 064] A Multi-Stage Optimization Pipeline for Bethesda Cell Detection in Pap Smear Cytology
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在提升巴氏细胞检测的准确性。通过集成YOLO和U-Net模型，并结合优化步骤，提高了检测性能。**

- **链接: [https://arxiv.org/pdf/2604.13939](https://arxiv.org/pdf/2604.13939)**

> **作者:** Martin Amster; Camila María Polotto
>
> **备注:** ISBI 2026 Accepted Paper & Second Place Solution for the RIVA Cervical Cytology Challenge Track B
>
> **摘要:** Computer vision techniques have advanced significantly in recent years, finding diverse and impactful applications within the medical field. In this paper, we introduce a new framework for the detection of Bethesda cells in Pap smear images, developed for Track B of the Riva Cytology Challenge held in association with the International Symposium on Biomedical Imaging (ISBI). This work focuses on enhancing computer vision models for cell detection, with performance evaluated using the mAP50-95 metric. We propose a solution based on an ensemble of YOLO and U-Net architectures, followed by a refinement stage utilizing overlap removal techniques and a binary classifier. Our framework achieved second place with a mAP50-95 score of 0.5909 in the competition. The implementation and source code are available at the following repository: this http URL
>
---
#### [new 065] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization
- **分类: cs.CV**

- **简介: 该论文属于环境光照归一化任务，旨在解决复杂光照下图像恢复问题。提出UniBlendNet，融合全局、多尺度和区域自适应机制，提升恢复效果。**

- **链接: [https://arxiv.org/pdf/2604.13383](https://arxiv.org/pdf/2604.13383)**

> **作者:** Jiatao Dai; Wei Dong; Han Zhou; Chengzhou Tang; Jun Chen
>
> **备注:** Accepted to CVPR 2026 NTIRE Workshop on New Trends in Image Restoration and Enhancement. 8 pages, 4 figures
>
> **摘要:** Ambient Lighting Normalization (ALN) aims to restore images degraded by complex, spatially varying illumination conditions. Existing methods, such as IFBlend, leverage frequency-domain priors to model illumination variations, but still suffer from limited global context modeling and insufficient spatial adaptivity, leading to suboptimal restoration in challenging regions. In this paper, we propose UniBlendNet, a unified framework for ambient lighting normalization that jointly models global illumination, multi-scale structures, and region-adaptive refinement. Specifically, we enhance global illumination understanding by integrating a UniConvNet-based module to capture long-range dependencies. To better handle complex lighting variations, we introduce a Scale-Aware Aggregation Module (SAAM) that performs pyramid-based multi-scale feature aggregation with dynamic reweighting. Furthermore, we design a mask-guided residual refinement mechanism to enable region-adaptive correction, allowing the model to selectively enhance degraded regions while preserving well-exposed areas. This design effectively improves illumination consistency and structural fidelity under complex lighting conditions. Extensive experiments on the NTIRE Ambient Lighting Normalization benchmark demonstrate that UniBlendNet consistently outperforms the baseline IFBlend and achieves improved restoration quality, while producing visually more natural and stable restoration results.
>
---
#### [new 066] Seek-and-Solve: Benchmarking MLLMs for Visual Clue-Driven Reasoning in Daily Scenarios
- **分类: cs.CV**

- **简介: 该论文提出DailyClue基准，用于评估MLLM在日常场景中基于视觉线索的推理能力。旨在解决现有基准忽略推理能力的问题，通过真实场景和复杂任务测试模型的视觉线索识别与推理能力。**

- **链接: [https://arxiv.org/pdf/2604.14041](https://arxiv.org/pdf/2604.14041)**

> **作者:** Xiaomin Li; Tala Wang; Zichen Zhong; Ying Zhang; Zirui Zheng; Takashi Isobe; Dezhuang Li; Huchuan Lu; You He; Xu Jia
>
> **备注:** Project page: this https URL
>
> **摘要:** Daily scenarios are characterized by visual richness, requiring Multimodal Large Language Models (MLLMs) to filter noise and identify decisive visual clues for accurate reasoning. Yet, current benchmarks predominantly aim at evaluating MLLMs' pre-existing knowledge or perceptual understanding, often neglecting the critical capability of reasoning. To bridge this gap, we introduce DailyClue, a benchmark designed for visual clue-driven reasoning in daily scenarios. Our construction is guided by two core principles: (1) strict grounding in authentic daily activities, and (2) challenging query design that necessitates more than surface-level perception. Instead of simple recognition, our questions compel MLLMs to actively explore suitable visual clues and leverage them for subsequent reasoning. To this end, we curate a comprehensive dataset spanning four major daily domains and 16 distinct subtasks. Comprehensive evaluation across MLLMs and agentic models underscores the formidable challenge posed by our benchmark. Our analysis reveals several critical insights, emphasizing that the accurate identification of visual clues is essential for robust reasoning.
>
---
#### [new 067] Feed-Forward 3D Scene Modeling: A Problem-Driven Perspective
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文属于3D重建任务，旨在解决传统方法效率低、泛化能力差的问题。通过分析feed-forward模型的设计模式，提出新的分类体系，并探讨关键问题与应用方向。**

- **链接: [https://arxiv.org/pdf/2604.14025](https://arxiv.org/pdf/2604.14025)**

> **作者:** Weijie Wang; Qihang Cao; Sensen Gao; Donny Y. Chen; Haofei Xu; Wenjing Bian; Songyou Peng; Tat-Jen Cham; Chuanxia Zheng; Andreas Geiger; Jianfei Cai; Jia-Wang Bian; Bohan Zhuang
>
> **备注:** 67 pages, 395 references. Project page: this https URL. Code: this https URL. This work has been submitted to Springer for possible publication
>
> **摘要:** Reconstructing 3D representations from 2D inputs is a fundamental task in computer vision and graphics, serving as a cornerstone for understanding and interacting with the physical world. While traditional methods achieve high fidelity, they are limited by slow per-scene optimization or category-specific training, which hinders their practical deployment and scalability. Hence, generalizable feed-forward 3D reconstruction has witnessed rapid development in recent years. By learning a model that maps images directly to 3D representations in a single forward pass, these methods enable efficient reconstruction and robust cross-scene generalization. Our survey is motivated by a critical observation: despite the diverse geometric output representations, ranging from implicit fields to explicit primitives, existing feed-forward approaches share similar high-level architectural patterns, such as image feature extraction backbones, multi-view information fusion mechanisms, and geometry-aware design principles. Consequently, we abstract away from these representation differences and instead focus on model design, proposing a novel taxonomy centered on model design strategies that are agnostic to the output format. Our proposed taxonomy organizes the research directions into five key problems that drive recent research development: feature enhancement, geometry awareness, model efficiency, augmentation strategies and temporal-aware models. To support this taxonomy with empirical grounding and standardized evaluation, we further comprehensively review related benchmarks and datasets, and extensively discuss and categorize real-world applications based on feed-forward 3D models. Finally, we outline future directions to address open challenges such as scalability, evaluation standards, and world modeling.
>
---
#### [new 068] Don't Let the Video Speak: Audio-Contrastive Preference Optimization for Audio-Visual Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，解决AVLMs中视频驱动的音频幻觉问题。通过ACPO框架，增强音频真实性，提升模型对音频的准确理解。**

- **链接: [https://arxiv.org/pdf/2604.14129](https://arxiv.org/pdf/2604.14129)**

> **作者:** Ami Baid; Zihui Xue; Kristen Grauman
>
> **备注:** Project page: this https URL
>
> **摘要:** While Audio-Visual Language Models (AVLMs) have achieved remarkable progress over recent years, their reliability is bottlenecked by cross-modal hallucination. A particularly pervasive manifestation is video-driven audio hallucination: models routinely exploit visual shortcuts to hallucinate expected sounds, discarding true auditory evidence. To counteract this deeply ingrained visual dominance, we propose Audio-Contrastive Preference Optimization (ACPO). This dual-axis preference learning framework introduces an output-contrastive objective to penalize visual descriptions masquerading as audio facts, alongside an input-contrastive objective that swaps audio tracks to explicitly penalize generation invariant to the true auditory signal. Extensive experiments demonstrate that ACPO establishes highly faithful audio grounding and mitigates audio hallucination without compromising overarching multimodal capabilities.
>
---
#### [new 069] See&Say: Vision Language Guided Safe Zone Detection for Autonomous Package Delivery Drones
- **分类: cs.CV**

- **简介: 该论文属于无人机安全降落区检测任务，旨在解决复杂环境中准确识别安全包裹投放点的问题。通过融合视觉语言模型与深度信息，提升安全区域判断的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.13292](https://arxiv.org/pdf/2604.13292)**

> **作者:** Mahyar Ghazanfari; Peng Wei
>
> **摘要:** Autonomous drone delivery systems are rapidly advancing, but ensuring safe and reliable package drop-offs remains highly challenging in cluttered urban and suburban environments where accurately identifying suitable package drop zones is critical. Existing approaches typically rely on either geometry-based analysis or semantic segmentation alone, but these methods lack the integrated semantic reasoning required for robust decision-making. To address this gap, we propose See&Say, a novel framework that combines geometric safety cues with semantic perception, guided by a Vision-Language Model (VLM) for iterative refinement. The system fuses monocular depth gradients with open-vocabulary detection masks to produce safety maps, while the VLM dynamically adjusts object category prompts and refines hazard detection across time, enabling reliable reasoning under dynamic conditions during the final delivery phase. When the primary drop-pad is occupied or unsafe, the proposed See&Say also identifies alternative candidate zones for package delivery. We curated a dataset of urban delivery scenarios with moving objects and human activities to evaluate the approach. Experimental results show that See&Say outperforms all baselines, achieving the highest accuracy and IoU for safety map prediction as well as superior performance in alternative drop zone evaluation across multiple thresholds. These findings highlight the promise of VLM-guided segmentation-depth fusion for advancing safe and practical drone-based package delivery.
>
---
#### [new 070] A Resource-Efficient Hybrid CNN-LSTM network for image-based bean leaf disease classification
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决豆类叶片疾病自动诊断中的资源消耗高和空间依赖捕捉不足问题。通过融合CNN与LSTM，构建轻量级模型，提升分类精度并降低内存占用。**

- **链接: [https://arxiv.org/pdf/2604.13835](https://arxiv.org/pdf/2604.13835)**

> **作者:** Hye Jin Rhee; Joseph Damilola Akinyemi
>
> **摘要:** Accurate and resource-efficient automated diagnosis is a cornerstone of modern agricultural expert systems. While Convolutional Neural Networks (CNNs) have established benchmarks in plant pathology, their ability to capture long-range spatial dependencies is often limited by standard pooling layers, and their high memory footprint hinders deployment on portable devices. This paper proposes a lightweight hybrid CNN-LSTM system for bean leaf disease classification. By integrating an LSTM layer to model the spatial-sequential relationships within feature maps, our hybrid architecture achieves a 94.38% accuracy while maintaining an exceptionally small footprint of 1.86 MB; a 70% reduction in size compared to traditional CNN-based systems. Furthermore, we provide a systematic evaluation of image augmentation strategies, demonstrating that tailored transformations are superior to generic combinations for maintaining the integrity of diagnostic patterns. Results on the $\textit{ibean}$ dataset confirm that the proposed system achieves state-of-the-art F1 scores of 99.22% with EfficientNet-B7+LSTM, providing a robust and scalable framework for real-time agricultural decision support in resource-constrained environments. The code and augmented datasets used in this study are publicly available on this $\href{this https URL}{Github}$ repo.
>
---
#### [new 071] Heuristic Style Transfer for Real-Time, Efficient Weather Attribute Detection
- **分类: cs.CV**

- **简介: 该论文属于天气属性检测任务，旨在从RGB图像中高效识别天气类型及11个相关属性。通过引入轻量级架构和风格迁移技术，提升检测精度与实时性。**

- **链接: [https://arxiv.org/pdf/2604.13947](https://arxiv.org/pdf/2604.13947)**

> **作者:** Hamed Ouattara; Pierre Duthon; Pascal Houssam Salmane; Frédéric Bernardin; Omar Ait Aider
>
> **备注:** 32 pages, 18 figures
>
> **摘要:** We present lightweight and efficient architectures to detect weather conditions from RGB images, predicting the weather type (sunny, rain, snow, fog) and 11 complementary attributes such as intensity, visibility, and ground condition, for a total of 53 classes across the tasks. This work examines to what extent weather conditions manifest as variations in visual style. We investigate style-inspired techniques, including Gram matrices, a truncated ResNet-50 targeting lower and intermediate layers, and PatchGAN-style architectures, within a multi-task framework with attention mechanisms. Two families are introduced: RTM (ResNet50-Truncated-MultiTasks) and PMG (PatchGAN-MultiTasks-Gram), together with their variants. Our contributions include automation of Gram-matrix computation, integration of PatchGAN into supervised multi-task learning, and local style capture through local Gram for improved spatial coherence. We also release a dataset of 503,875 images annotated with 12 weather attributes under a Creative Commons Attribution (CC-BY) license. The models achieve F1 scores above 96 percent on our internal test set and above 78 percent in zero-shot evaluation on several external datasets, confirming their generalization ability. The PMG architecture, with fewer than 5 million parameters, runs in real time with a small memory footprint, making it suitable for embedded systems. The modular design of the models also allows style-related or weather-related tasks to be added or removed as needed.
>
---
#### [new 072] Why Multimodal In-Context Learning Lags Behind? Unveiling the Inner Mechanisms and Bottlenecks
- **分类: cs.CV**

- **简介: 该论文研究多模态上下文学习（ICL）的机制与瓶颈，旨在解决其在少样本情况下性能下降的问题。通过分析任务映射构建与迁移，提出改进方法以提升多模态适应效果。**

- **链接: [https://arxiv.org/pdf/2604.13403](https://arxiv.org/pdf/2604.13403)**

> **作者:** Yu Wang; Sharon Li
>
> **备注:** ACL Main 2026
>
> **摘要:** In-context learning (ICL) enables models to adapt to new tasks via inference-time demonstrations. Despite its success in large language models, the extension of ICL to multimodal settings remains poorly understood in terms of its internal mechanisms and how it differs from text-only ICL. In this work, we conduct a systematic analysis of ICL in multimodal large language models. Using identical task formulations across modalities, we show that multimodal ICL performs comparably to text-only ICL in zero-shot settings but degrades significantly under few-shot demonstrations. To understand this gap, we decompose multimodal ICL into task mapping construction and task mapping transfer, and analyze how models establish cross-modal task mappings, and transfer them to query samples across layers. Our analysis reveals that current models lack reasoning-level alignment between visual and textual representations, and fail to reliably transfer learned task mappings to queries. Guided by these findings, we further propose a simple inference-stage enhancement method that reinforces task mapping transfer. Our results provide new insights into the mechanisms and limitations of multimodal ICL and suggest directions for more effective multimodal adaptation. Our code is available \href{this https URL}{here}.
>
---
#### [new 073] DiT as Real-Time Rerenderer: Streaming Video Stylization with Autoregressive Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文属于视频风格化任务，解决长视频处理不稳定和计算成本高的问题。提出RTR-DiT框架，实现高效实时视频风格化与交互式风格切换。**

- **链接: [https://arxiv.org/pdf/2604.13509](https://arxiv.org/pdf/2604.13509)**

> **作者:** Hengye Lyu; Zisu Li; Yue Hong; Yueting Weng; Jiaxin Shi; Hanwang Zhang; Chen Liang
>
> **摘要:** Recent advances in video generation models has significantly accelerated video generation and related downstream tasks. Among these, video stylization holds important research value in areas such as immersive applications and artistic creation, attracting widespread attention. However, existing diffusion-based video stylization methods struggle to maintain stability and consistency when processing long videos, and their high computational cost and multi-step denoising make them difficult to apply in practical scenarios. In this work, we propose RTR-DiT (DiT as Real-Time Rerenderer), a steaming video stylization framework built upon Diffusion Transformer. We first fine-tune a bidirectional teacher model on a curated video stylization dataset, supporting both text-guided and reference-guided video stylization tasks, and subsequently distill it into a few-step autoregressive model via post-training with Self Forcing and Distribution Matching Distillation. Furthermore, we propose a reference-preserving KV cache update strategy that not only enables stable and consistent processing of long videos, but also supports real-time switching between text prompts and reference images. Experimental results show that RTR-DiT outperforms existing methods in both text-guided and reference-guided video stylization tasks, in terms of quantitative metrics and visual quality, and demonstrates excellent performance in real-time long video stylization and interactive style-switching applications.
>
---
#### [new 074] OneHOI: Unifying Human-Object Interaction Generation and Editing
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出OneHOI，解决人类-物体交互生成与编辑任务。针对现有方法在条件整合和多交互处理上的不足，引入统一的扩散Transformer框架，实现高效、灵活的HOI控制。**

- **链接: [https://arxiv.org/pdf/2604.14062](https://arxiv.org/pdf/2604.14062)**

> **作者:** Jiun Tian Hoe; Weipeng Hu; Xudong Jiang; Yap-Peng Tan; Chee Seng Chan
>
> **备注:** Accepted at CVPR2026. This paper moves toward unifying HOI generation and editing within a single model
>
> **摘要:** Human-Object Interaction (HOI) modelling captures how humans act upon and relate to objects, typically expressed as <person, action, object> triplets. Existing approaches split into two disjoint families: HOI generation synthesises scenes from structured triplets and layout, but fails to integrate mixed conditions like HOI and object-only entities; and HOI editing modifies interactions via text, yet struggles to decouple pose from physical contact and scale to multiple interactions. We introduce OneHOI, a unified diffusion transformer framework that consolidates HOI generation and editing into a single conditional denoising process driven by shared structured interaction representations. At its core, the Relational Diffusion Transformer (R-DiT) models verb-mediated relations through role- and instance-aware HOI tokens, layout-based spatial Action Grounding, a Structured HOI Attention to enforce interaction topology, and HOI RoPE to disentangle multi-HOI scenes. Trained jointly with modality dropout on our HOI-Edit-44K, along with HOI and object-centric datasets, OneHOI supports layout-guided, layout-free, arbitrary-mask, and mixed-condition control, achieving state-of-the-art results across both HOI generation and editing. Code is available at this https URL.
>
---
#### [new 075] Free Geometry: Refining 3D Reconstruction from Longer Versions of Itself
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决模型在测试时无法适应场景的问题。通过自监督学习，使模型在无真值情况下自我优化，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2604.14048](https://arxiv.org/pdf/2604.14048)**

> **作者:** Yuhang Dai; Xingyi Yang
>
> **备注:** Code is available at this https URL
>
> **摘要:** Feed-forward 3D reconstruction models are efficient but rigid: once trained, they perform inference in a zero-shot manner and cannot adapt to the test scene. As a result, visually plausible reconstructions often contain errors, particularly under occlusions, specularities, and ambiguous cues. To address this, we introduce Free Geometry, a framework that enables feed-forward 3D reconstruction models to self-evolve at test time without any 3D ground truth. Our key insight is that, when the model receives more views, it produces more reliable and view-consistent reconstructions. Leveraging this property, given a testing sequence, we mask a subset of frames to construct a self-supervised task. Free Geometry enforces cross-view feature consistency between representations from full and partial observations, while maintaining the pairwise relations implied by the held-out frames. This self-supervision allows for fast recalibration via lightweight LoRA updates, taking less than 2 minutes per dataset on a single GPU. Our approach consistently improves state-of-the-art foundation models, including Depth Anything 3 and VGGT, across 4 benchmark datasets, yielding an average improvement of 3.73% in camera pose accuracy and 2.88% in point map prediction. Code is available at this https URL .
>
---
#### [new 076] HiProto: Hierarchical Prototype Learning for Interpretable Object Detection Under Low-quality Conditions
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决低质量图像下的可解释性问题。通过层次化原型学习，提升语义区分能力和模型解释性，无需图像增强或复杂结构。**

- **链接: [https://arxiv.org/pdf/2604.13981](https://arxiv.org/pdf/2604.13981)**

> **作者:** Jianlin Xiang; Linhui Dai; Xue Yang; Chaolei Yang; Yanshan Li
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Interpretability is essential for deploying object detection systems in critical applications, especially under low-quality imaging conditions that degrade visual information and increase prediction uncertainty. Existing methods either enhance image quality or design complex architectures, but often lack interpretability and fail to improve semantic discrimination. In contrast, prototype learning enables interpretable modeling by associating features with class-centered semantics, which can provide more stable and interpretable representations under degradation. Motivated by this, we propose HiProto, a new paradigm for interpretable object detection based on hierarchical prototype learning. By constructing structured prototype representations across multiple feature levels, HiProto effectively models class-specific semantics, thereby enhancing both semantic discrimination and interpretability. Building upon prototype modeling, we first propose a Region-to-Prototype Contrastive Loss (RPC-Loss) to enhance the semantic focus of prototypes on target regions. Then, we propose a Prototype Regularization Loss (PR-Loss) to improve the distinctiveness among class prototypes. Finally, we propose a Scale-aware Pseudo Label Generation Strategy (SPLGS) to suppress mismatched supervision for RPC-Loss, thereby preserving the robustness of low-level prototype representations. Experiments on ExDark, RTTS, and VOC2012-FOG demonstrate that HiProto achieves competitive results while offering clear interpretability through prototype responses, without relying on image enhancement or complex architectures. Our code will be available at this https URL.
>
---
#### [new 077] Right Regions, Wrong Labels: Semantic Label Flips in Segmentation under Correlation Shift
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，研究在分布偏移下模型因相关性错误导致的语义标签翻转问题，提出诊断方法和风险评估指标以提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.13326](https://arxiv.org/pdf/2604.13326)**

> **作者:** Akshit Achara; Yovin Yathathugoda; Nick Byrne; Michela Antonelli; Esther Puyol Anton; Alexander Hammers; Andrew P. King
>
> **备注:** Accepted at the CAO Workshop, ICLR 2026
>
> **摘要:** The robustness of machine learning models can be compromised by spurious correlations between non-causal features in the input data and target labels. A common way to test for such correlations is to train on data where the label is strongly tied to some non-causal cue, then evaluate on examples where that tie no longer holds. This idea is well established for classification tasks, but for semantic segmentation the specific failure modes are not well understood. We show that a model may achieve reasonable overlap while assigning the wrong semantic label, swapping one plausible foreground class for another, even when object boundaries are largely correct. We focus on this semantic label-flip behaviour and quantify it with a simple diagnostic (Flip) that counts how often ground truth foreground pixels are assigned the wrong foreground identity while remaining predicted as foreground. In a setting where category and scene are correlated during training, increasing the correlation consistently widens the gap between common and rare test conditions and increases these within-object label swaps on counterfactual groups. Overall, our results motivate assessing segmentation robustness under distribution shift beyond overlap by decomposing foreground errors into correct pixels, flipped-identity pixels, and missed-to-background pixels. We also propose an entropy-based, ground truth label-free `flip-risk' score, which is computed from foreground identity uncertainty, and show that it can flag flip-prone cases at inference time. Code is available at this https URL.
>
---
#### [new 078] 4th Workshop on Maritime Computer Vision (MaCVi): Challenge Overview
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于计算机视觉领域，聚焦海事应用。论文介绍了MaCVi 2026挑战赛的设置、评估协议及结果分析，旨在提升海事场景下的视觉算法准确性与实时性。**

- **链接: [https://arxiv.org/pdf/2604.13244](https://arxiv.org/pdf/2604.13244)**

> **作者:** Benjamin Kiefer; Jan Lukas Augustin; Jon Muhovič; Mingi Jeong; Arnold Wiliem; Janez Pers; Matej Kristan; Alberto Quattrini Li; Matija Teršek; Josip Šarić; Arpita Vats; Dominik Hildebrand; Rafia Rahim; Mahmut Karaaslan; Arpit Vaishya; Steve Xie; Ersin Kaya; Akib Mashrur; Tze-Hsiang Tang; Chun-Ming Tsai; Jun-Wei Hsieh; Ming-Ching Chang; Wonwoo Jo; Doyeon Lee; Yusi Cao; Lingling Li; Vinayak Nageli; Arshad Jamal; Gorthi Rama Krishna Sai Subrahmanyam; Jemo Maeng; Seongju Lee; Kyoobin Lee; Xu Liu; LiCheng Jiao; Jannik Sheikh; Martin Weinmann; Ivan Martinović; Jose Mateus Raitz Persch; Rahul Harsha Cheppally; Mehmet E. Belviranli; Dimitris Gahtidis; Hyewon Chun; Sangmun Lee; Philipp Gorczak; Hansol Kim; Jeeyeon Jeon; Borja Carrillo Perez; Jiahui Wang; Sangmin Park; Andreas Michel; Jannick Kuester; Bettina Felten; Wolfgang Gross; Yuan Feng; Justin Davis
>
> **备注:** Accepted to CVPR 2026 Workshop Proceeding; Maritime Computer Vision Workshop
>
> **摘要:** The 4th Workshop on Maritime Computer Vision (MaCVi) is organized as part of CVPR 2026. This edition features five benchmark challenges with emphasis on both predictive accuracy and embedded real-time feasibility. This report summarizes the MaCVi 2026 challenge setup, evaluation protocols, datasets, and benchmark tracks, and presents quantitative results, qualitative comparisons, and cross-challenge analyses of emerging method trends. We also include technical reports from top-performing teams to highlight practical design choices and lessons learned across the benchmark suite. Datasets, leaderboards, and challenge resources are available at this https URL.
>
---
#### [new 079] From Synchrony to Sequence: Exo-to-Ego Generation via Interpolation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决同步数据中的时空不连续问题。通过序列建模和插值方法，提升第一人称视频生成效果，并统一处理双向生成任务。**

- **链接: [https://arxiv.org/pdf/2604.13793](https://arxiv.org/pdf/2604.13793)**

> **作者:** Mohammad Mahdi; Nedko Savov; Danda Pani Paudel; Luc Van Gool
>
> **摘要:** Exo-to-Ego video generation aims to synthesize a first-person video from a synchronized third-person view and corresponding camera poses. While paired supervision is available, synchronized exo-ego data inherently introduces substantial spatio-temporal and geometric discontinuities, violating the smooth-motion assumptions of standard video generation benchmarks. We identify this synchronization-induced jump as the central challenge and propose Syn2Seq-Forcing, a sequential formulation that interpolates between the source and target videos to form a single continuous signal. By reframing Exo2Ego as sequential signal modeling rather than a conventional condition-output task, our approach enables diffusion-based sequence models, e.g. Diffusion Forcing Transformers (DFoT), to capture coherent transitions across frames more effectively. Empirically, we show that interpolating only the videos, without performing pose interpolation already produces significant improvements, emphasizing that the dominant difficulty arises from spatio-temporal discontinuities. Beyond immediate performance gains, this formulation establishes a general and flexible framework capable of unifying both Exo2Ego and Ego2Exo generation within a single continuous sequence model, providing a principled foundation for future research in cross-view video synthesis.
>
---
#### [new 080] Reconstruction of a 3D wireframe from a single line drawing via generative depth estimation
- **分类: cs.CV**

- **简介: 该论文属于2D草图到3D模型的重建任务，旨在解决传统方法依赖预设CAD原型和符号逻辑的问题。通过生成式深度估计方法，实现从单张线稿生成3D线框模型。**

- **链接: [https://arxiv.org/pdf/2604.13549](https://arxiv.org/pdf/2604.13549)**

> **作者:** Elton Cao; Hod Lipson
>
> **摘要:** The conversion of 2D freehand sketches into 3D models remains a pivotal challenge in computer vision, bridging the gap between human creativity and digital fabrication. Traditional line drawing reconstruction relies on brittle symbolic logic, while modern approaches are constrained by rigid parametric modeling, limiting users to predefined CAD primitives. We propose a generative approach by framing reconstruction as a conditional dense depth estimation task. To achieve this, we implement a Latent Diffusion Model (LDM) with a ControlNet-style conditioning framework to resolve the inherent ambiguities of orthographic projections. To support an iterative "sketch-reconstruct-sketch" workflow, we introduce a graph-based BFS masking strategy to simulate partial depth cues. We train and evaluate our approach using a massive dataset of over one million image-depth pairs derived from the ABC Dataset. Our framework demonstrates robust performance across varying shape complexities, providing a scalable pipeline for converting sparse 2D line drawings into dense 3D representations, effectively allowing users to "draw in 3D" without the rigid constraints of traditional CAD.
>
---
#### [new 081] AI Powered Image Analysis for Phishing Detection
- **分类: cs.CV; cs.NI**

- **简介: 该论文属于图像分类任务，旨在解决视觉仿冒钓鱼网站的检测问题。通过使用深度学习模型分析网页截图，比较不同模型的性能，优化决策阈值以提高检测效果。**

- **链接: [https://arxiv.org/pdf/2604.13555](https://arxiv.org/pdf/2604.13555)**

> **作者:** K. Acharya; S. Ale; R. Kadel
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Phishing websites now rely heavily on visual imitation-copied logos, similar layouts, and matching colours-to avoid detection by text- and URL-based systems. This paper presents a deep learning approach that uses webpage screenshots for image-based phishing detection. Two vision models, ConvNeXt-Tiny and Vision Transformer (ViT-Base), were tested to see how well they handle visually deceptive phishing pages. The framework covers dataset creation, preprocessing, transfer learning with ImageNet weights, and evaluation using different decision thresholds. The results show that ConvNeXt-Tiny performs the best overall, achieving the highest F1-score at the optimised threshold and running more efficiently than ViT-Base. This highlights the strength of convolutional models for visual phishing detection and shows why threshold tuning is important for real-world deployment. As future work, the curated dataset used in this study will be released to support reproducibility and encourage further research in this area. Unlike many existing studies that primarily report accuracy, this work places greater emphasis on threshold-aware evaluation to better reflect real-world deployment conditions. By examining precision, recall, and F1-score across different decision thresholds, the study identifies operating points that balance detection performance and false-alarm control. In addition, the side-by-side comparison of ConvNeXt-Tiny and ViT-Base under the same experimental setup offers practical insights into how convolutional and transformer-based architectures differ in robustness and computational efficiency for visual phishing detection.
>
---
#### [new 082] One Token per Highly Selective Frame: Towards Extreme Compression for Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于长视频理解任务，旨在解决视频帧过多导致的token压缩与信息丢失问题。通过引入token级和帧级压缩方法，实现极端压缩并提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.14149](https://arxiv.org/pdf/2604.14149)**

> **作者:** Zheyu Zhang; Ziqi Pang; Shixing Chen; Xiang Hao; Vimal Bhat; Yu-Xiong Wang
>
> **摘要:** Long video understanding is inherently challenging for vision-language models (VLMs) because of the extensive number of frames. With each video frame typically expanding into tens or hundreds of tokens, the limited context length of large language models (LLMs) forces the VLMs to perceive the frames sparsely and lose temporal information. To address this, we explore extreme video token compression towards \emph{one token per frame} at the final LLM layer. Our key insight is that heuristic-based compression, widely adopted by previous methods, is prone to information loss, and this necessitates supervising LLM layers into \emph{learnable} and \emph{progressive} modules for \emph{token-level compression} (LP-Comp). Such compression enables our VLM to digest 2x-4x more frames with improved performance. To further increase the token efficiency, we investigate \emph{frame-level compression}, which selects the frames most relevant to the queries via the internal attention scores of the LLM layers, named \emph{question-conditioned compression} (QC-Comp). As a notable distinction from previous studies, we mitigate the position bias of LLM attention in long contexts, \emph{i.e.}, the over-concentration on the beginning and end of a sequence, by splitting long videos into short segments and employing local attention. Collectively, our combined \emph{token-level} and \emph{frame-level} leads to an e\textbf{x}treme compression model for long video understanding, named \textbf{\name}, achieving a significantly larger compression ratio and enabling denser frame sampling. Our \name is finetuned from VideoChat-Flash with a data-efficient \emph{supervised compression tuning} stage that only requires 2.5\% of the supervised fine-tuning data, yet boosts the accuracy from 42.9\% to 46.2\% on LVBench and enhances multiple other long video benchmarks.
>
---
#### [new 083] A High-Resolution Landscape Dataset for Concept-Based XAI With Application to Species Distribution Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于物种分布建模任务，旨在解决深度学习模型难以解释的问题。通过构建高分辨率景观数据集，结合概念可解释AI方法，提升模型的可解释性与生态意义。**

- **链接: [https://arxiv.org/pdf/2604.13240](https://arxiv.org/pdf/2604.13240)**

> **作者:** Augustin de la Brosse; Damien Garreau; Thomas Houet; Thomas Corpetti
>
> **摘要:** Mapping the spatial distribution of species is essential for conservation policy and invasive species management. Species distribution models (SDMs) are the primary tools for this task, serving two purposes: achieving robust predictive performance while providing ecological insights into the driving factors of distribution. However, the increasing complexity of deep learning SDMs has made extracting these insights more challenging. To reconcile these objectives, we propose the first implementation of concept-based Explainable AI (XAI) for SDMs. We leverage the Robust TCAV (Testing with Concept Activation Vectors) methodology to quantify the influence of landscape concepts on model predictions. To enable this, we provide a new open-access landscape concept dataset derived from high-resolution multispectral and LiDAR drone imagery. It includes 653 patches across 15 distinct landscape concepts and 1,450 random reference patches, designed to suit a wide range of species. We demonstrate this approach through a case study of two aquatic insects, Plecoptera and Trichoptera, using two Convolutional Neural Networks and one Vision Transformer. Results show that concept-based XAI helps validate SDMs against expert knowledge while uncovering novel associations that generate new ecological hypotheses. Robust TCAV also provides landscape-level information, useful for policy-making and land management. Code and datasets are publicly available.
>
---
#### [new 084] ADP-DiT: Text-Guided Diffusion Transformer for Brain Image Generation in Alzheimer's Disease Progression
- **分类: cs.CV**

- **简介: 该论文提出ADP-DiT，用于阿尔茨海默病纵向MRI生成，解决个体化影像合成问题，通过文本条件控制时间与临床信息。**

- **链接: [https://arxiv.org/pdf/2604.13495](https://arxiv.org/pdf/2604.13495)**

> **作者:** Juneyong Lee; Geonwoo Baek; Ikbeom Jang
>
> **备注:** 15 pages, 3 figures, accepted to ICPR 2026
>
> **摘要:** Alzheimer's disease (AD) progresses heterogeneously across individuals, motivating subject-specific synthesis of follow-up magnetic resonance imaging (MRI) to support progression assessment. While Diffusion Transformers (DiT), an emerging transformer-based diffusion model, offer a scalable backbone for image synthesis, longitudinal AD MRI generation with clinically interpretable control over follow-up time and participant metadata remains underexplored. We present ADP-DiT, an interval-aware, clinically text-conditioned diffusion transformer for longitudinal AD MRI synthesis. ADP-DiT encodes follow-up interval together with multi-domain demographic, diagnostic (CN/MCI/AD), and neuropsychological information as a natural-language prompt, enabling time-specific control beyond coarse diagnostic stages. To inject this conditioning effectively, we use dual text encoders-OpenCLIP for vision-language alignment and T5 for richer clinical-language understanding. Their embeddings are fused into DiT through cross-attention for fine-grained guidance and adaptive layer normalization for global modulation. We further enhance anatomical fidelity by applying rotary positional embeddings to image tokens and performing diffusion in a pre-trained SDXL-VAE latent space to enable efficient high-resolution reconstruction. On 3,321 longitudinal 3T T1-weighted scans from 712 participants (259,038 image slices), ADP-DiT achieves SSIM 0.8739 and PSNR 29.32 dB, improving over a DiT baseline by +0.1087 SSIM and +6.08 dB PSNR while capturing progression-related changes such as ventricular enlargement and shrinking hippocampus. These results suggest that integrating comprehensive, subject-specific clinical conditions with architectures can improve longitudinal AD MRI synthesis.
>
---
#### [new 085] What Are We Really Measuring? Rethinking Dataset Bias in Web-Scale Natural Image Collections via Unsupervised Semantic Clustering
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉领域，旨在解决数据集偏差评估问题。研究指出传统方法依赖的分类准确率可能受低级特征干扰，提出无监督聚类方法更准确衡量语义差异。**

- **链接: [https://arxiv.org/pdf/2604.13610](https://arxiv.org/pdf/2604.13610)**

> **作者:** Amir Hossein Saleknia; Mohammad Sabokrou
>
> **摘要:** In computer vision, a prevailing method for quantifying dataset bias is to train a model to distinguish between datasets. High classification accuracy is then interpreted as evidence of meaningful semantic differences. This approach assumes that standard image augmentations successfully suppress low-level, non-semantic cues, and that any remaining performance must therefore reflect true semantic divergence. We demonstrate that this fundamental assumption is flawed within the domain of large-scale natural image collections. High classification accuracy is often driven by resolution-based artifacts, which are structural fingerprints arising from native image resolution distributions and interpolation effects during resizing. These artifacts form robust, dataset-specific signatures that persist despite conventional image corruptions. Through controlled experiments, we show that models achieve strong dataset classification even on non-semantic, procedurally generated images, proving their reliance on superficial cues. To address this issue, we revisit this decades-old idea of dataset separability, but not with supervised classification. Instead, we introduce an unsupervised approach that measures true semantic separability. Our framework directly assesses semantic similarity by clustering semantically-rich features from foundational vision models, deliberately bypassing supervised classification on dataset labels. When applied to major web-scale datasets, the primary focus of this work, the high separability reported by supervised methods largely vanishes, with clustering accuracy dropping to near-chance levels. This reveals that conventional classification-based evaluation systematically overstates semantic bias by an overwhelming margin.
>
---
#### [new 086] From Pixels to Nucleotides: End-to-End Token-Based Video Compression for DNA Storage
- **分类: cs.CV; cs.ET**

- **简介: 该论文属于视频压缩任务，解决视频DNA存储问题。提出HELIX模型，首次实现视频压缩与DNA编码的端到端联合优化。**

- **链接: [https://arxiv.org/pdf/2604.13667](https://arxiv.org/pdf/2604.13667)**

> **作者:** Cihan Ruan; Lebin Zhou; Bingqing Zhao; Rongduo Han; Qiming Yuan; Chenchen Zhu; Linyi Han; Liang Yang; Wei Wang; Wei Jiang; Nam Ling
>
> **摘要:** DNA-based storage has emerged as a promising approach to the global data crisis, offering molecular-scale density and millennial-scale stability at low maintenance cost. Over the past decade, substantial progress has been made in storing text, images, and files in DNA -- yet video remains an open challenge. The difficulty is not merely technical: effective video DNA storage requires co-designing compression and molecular encoding from the ground up, a challenge that sits at the intersection of two fields that have largely evolved independently. In this work, we present HELIX, the first end-to-end neural network jointly optimizing video compression and DNA encoding -- prior approaches treat the two stages independently, leaving biochemical constraints and compression objectives fundamentally misaligned. Our key insight: token-based representations naturally align with DNA's quaternary alphabet -- discrete semantic units map directly to ATCG bases. We introduce TK-SCONE (Token-Kronecker Structured Constraint-Optimized Neural Encoding), which achieves 1.91 bits per nucleotide through Kronecker-structured mixing that breaks spatial correlations and FSM-based mapping that guarantees biochemical constraints. Unlike two-stage approaches, HELIX learns token distributions simultaneously optimized for visual quality, prediction under masking, and DNA synthesis efficiency. This work demonstrates for the first time that learned compression and molecular storage converge naturally at token representations -- suggesting a new paradigm where neural video codecs are designed for biological substrates from the ground up.
>
---
#### [new 087] SpatialEvo: Self-Evolving Spatial Intelligence via Deterministic Geometric Environments
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SpatialEvo，解决3D空间推理任务中的标注成本高和模型错误固化问题，通过确定性几何环境实现自进化训练。**

- **链接: [https://arxiv.org/pdf/2604.14144](https://arxiv.org/pdf/2604.14144)**

> **作者:** Dinging Li; Yingxiu Zhao; Xinrui Cheng; Kangheng Lin; Hongbo Peng; Hongxing Li; Zixuan Wang; Yuhong Dai; Haodong Li; Jia Wang; Yukang Shi; Liang Zhao; Jianjian Sun; Zheng Ge; Xiangyu Zhang; Weiming Lu; Jun Xiao; Yueting Zhuang; Yongliang Shen
>
> **摘要:** Spatial reasoning over three-dimensional scenes is a core capability for embodied intelligence, yet continuous model improvement remains bottlenecked by the cost of geometric annotation. The self-evolving paradigm offers a promising path, but its reliance on model consensus to construct pseudo-labels causes training to reinforce rather than correct the model's own geometric errors. We identify a property unique to 3D spatial reasoning that circumvents this limitation: ground truth is a deterministic consequence of the underlying geometry, computable exactly from point clouds and camera poses without any model involvement. Building on this insight, we present SpatialEvo, a self-evolving framework for 3D spatial reasoning, centered on the Deterministic Geometric Environment (DGE). The DGE formalizes 16 spatial reasoning task categories under explicit geometric validation rules and converts unannotated 3D scenes into zero-noise interactive oracles, replacing model consensus with objective physical feedback. A single shared-parameter policy co-evolves across questioner and solver roles under DGE constraints: the questioner generates physically valid spatial questions grounded in scene observations, while the solver derives precise answers against DGE-verified ground truth. A task-adaptive scheduler endogenously concentrates training on the model's weakest categories, producing a dynamic curriculum without manual design. Experiments across nine benchmarks demonstrate that SpatialEvo achieves the highest average score at both 3B and 7B scales, with consistent gains on spatial reasoning benchmarks and no degradation on general visual understanding.
>
---
#### [new 088] Temporally Consistent Long-Term Memory for 3D Single Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于3D单目标跟踪任务，解决长期跟踪中特征不一致和记忆冗余问题。提出ChronoTrack框架，通过长时记忆和双损失机制提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2604.13789](https://arxiv.org/pdf/2604.13789)**

> **作者:** Jaejoon Yoo; SuBeen Lee; Yerim Jeon; Miso Lee; Jae-Pil Heo
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** 3D Single Object Tracking (3D-SOT) aims to localize a target object across a sequence of LiDAR point clouds, given its 3D bounding box in the first frame. Recent methods have adopted a memory-based approach to utilize previously observed features of the target object, but remain limited to only a few recent frames. This work reveals that their temporal capacity is fundamentally constrained to short-term context due to severe temporal feature inconsistency and excessive memory overhead. To this end, we propose a robust long-term 3D-SOT framework, ChronoTrack, which preserves the temporal feature consistency while efficiently aggregating the diverse target features via long-term memory. Based on a compact set of learnable memory tokens, ChronoTrack leverages long-term information through two complementary objectives: a temporal consistency loss and a memory cycle consistency loss. The former enforces feature alignment across frames, alleviating temporal drift and improving the reliability of proposed long-term memory. In parallel, the latter encourages each token to encode diverse and discriminative target representations observed throughout the sequence via memory-point-memory cyclic walks. As a result, ChronoTrack achieves new state-of-the-art performance on multiple 3D-SOT benchmarks, demonstrating its effectiveness in long-term target modeling with compact memory while running at real-time speed of 42 FPS on a single RTX 4090 GPU. The code is available at this https URL
>
---
#### [new 089] DF3DV-1K: A Large-Scale Dataset and Benchmark for Distractor-Free Novel View Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DF3DV-1K数据集，用于无干扰的新视角合成任务，解决现有数据集缺乏干净与杂乱图像的问题，通过基准测试和方法优化推动该领域发展。**

- **链接: [https://arxiv.org/pdf/2604.13416](https://arxiv.org/pdf/2604.13416)**

> **作者:** Cheng-You Lu; Yi-Shan Hung; Wei-Ling Chi; Hao-Ping Wang; Charlie Li-Ting Tsai; Yu-Cheng Chang; Yu-Lun Liu; Thomas Do; Chin-Teng Lin
>
> **摘要:** Advances in radiance fields have enabled photorealistic novel view synthesis. In several domains, large-scale real-world datasets have been developed to support comprehensive benchmarking and to facilitate progress beyond scene-specific reconstruction. However, for distractor-free radiance fields, a large-scale dataset with clean and cluttered images per scene remains lacking, limiting the development. To address this gap, we introduce DF3DV-1K, a large-scale real-world dataset comprising 1,048 scenes, each providing clean and cluttered image sets for benchmarking. In total, the dataset contains 89,924 images captured using consumer cameras to mimic casual capture, spanning 128 distractor types and 161 scene themes across indoor and outdoor environments. A curated subset of 41 scenes, DF3DV-41, is systematically designed to evaluate the robustness of distractor-free radiance field methods under challenging scenarios. Using DF3DV-1K, we benchmark nine recent distractor-free radiance field methods and 3D Gaussian Splatting, identifying the most robust methods and the most challenging scenarios. Beyond benchmarking, we demonstrate an application of DF3DV-1K by fine-tuning a diffusion-based 2D enhancer to improve radiance field methods, achieving average improvements of 0.96 dB PSNR and 0.057 LPIPS on the held-out set (e.g., DF3DV-41) and the On-the-go dataset. We hope DF3DV-1K facilitates the development of distractor-free vision and promotes progress beyond scene-specific approaches.
>
---
#### [new 090] Med-CAM: Minimal Evidence for Explaining Medical Decision Making
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Med-CAM，用于医疗决策的可解释性分析。解决医疗AI黑箱问题，通过生成精确证据图，提升模型透明度与临床可信度。**

- **链接: [https://arxiv.org/pdf/2604.13695](https://arxiv.org/pdf/2604.13695)**

> **作者:** Pirzada Suhail; Aditya Anand; Amit Sethi
>
> **摘要:** Reliable and interpretable decision-making is essential in medical imaging, where diagnostic outcomes directly influence patient care. Despite advances in deep learning, most medical AI systems operate as opaque black boxes, providing little insight into why a particular diagnosis was reached. In this paper, we introduce Med-CAM, a framework for generating minimal and sharp maps as evidence-based explanations for Medical decision making via Classifier Activation Matching. Med-CAM trains a segmentation network from scratch to produce a mask that highlights the minimal evidence critical to model's decision for any seen or unseen image. This ensures that the explanation is both faithful to the network's behaviour and interpretable to clinicians. Experiments show, unlike prior spatial explanation methods, such as Grad-CAM and attention maps, which yield only fuzzy regions of relative importance, Med-CAM with its superior spatial awareness to shapes, textures, and boundaries, delivers conclusive, evidence-based explanations that faithfully replicate the model's prediction for any given image. By explicitly constraining explanations to be compact, consistent with model activations, and diagnostic alignment, Med-CAM advances transparent AI to foster clinician understanding and trust in high-stakes medical applications such as pathology and radiology.
>
---
#### [new 091] Explainable Fall Detection for Elderly Care via Temporally Stable SHAP in Skeleton-Based Human Activity Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于老年人跌倒检测任务，旨在解决传统方法解释性差的问题。通过结合LSTM与T-SHAP，提升分类准确性与解释稳定性。**

- **链接: [https://arxiv.org/pdf/2604.13279](https://arxiv.org/pdf/2604.13279)**

> **作者:** Mohammad Saleh; Azadeh Tabatabaei
>
> **摘要:** Fall detection in elderly care requires not only accurate classification but also reliable explanations that clinicians can trust. However, existing post-hoc explainability methods, when applied frame-by-frame to sequential data, produce temporally unstable attribution maps that clinicians cannot reliably act upon. To address this issue, we propose a lightweight and explainable framework for skeleton-based fall detection that combines an efficient LSTM model with T-SHAP, a temporally aware post-hoc aggregation strategy that stabilizes SHAP-based feature attributions over contiguous time windows. Unlike standard SHAP, which treats each frame independently, T-SHAP applies a linear smoothing operator to the attribution sequence, reducing high-frequency variance while preserving the theoretical guarantees of Shapley values, including local accuracy and consistency. Experiments on the NTU RGB+D Dataset demonstrate that the proposed framework achieves 94.3% classification accuracy with an end-to-end inference latency below 25 milliseconds, satisfying real-time constraints on mid-range hardware and indicating strong potential for deployment in clinical monitoring scenarios. Quantitative evaluation using perturbation-based faithfulness metrics shows that T-SHAP improves explanation reliability compared to standard SHAP (AUP: 0.89 vs. 0.91) and Grad-CAM (0.82), with consistent improvements observed across five-fold cross-validation, indicating enhanced explanation reliability. The resulting attributions consistently highlight biomechanically relevant motion patterns, including lower-limb instability and changes in spinal alignment, aligning with established clinical observations of fall dynamics and supporting their use as transparent decision aids in long-term care environments
>
---
#### [new 092] Decoding the Delta: Unifying Remote Sensing Change Detection and Understanding with Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，旨在解决多时相遥感数据理解中的“时间盲区”问题。通过提出Delta-LLaVA框架和Delta-QA基准，提升变化推理与精确定位能力。**

- **链接: [https://arxiv.org/pdf/2604.14044](https://arxiv.org/pdf/2604.14044)**

> **作者:** Xiaohe Li; Jiahao Li; Kaixin Zhang; Yuqiang Fang; Leilei Lin; Hong Wang; Haohua Wu; Zide Fan
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel in general vision-language tasks, their application to remote sensing change understanding is hindered by a fundamental "temporal blindness". Existing architectures lack intrinsic mechanisms for multi-temporal contrastive reasoning and struggle with precise spatial grounding. To address this, we first introduce Delta-QA, a comprehensive benchmark comprising 180k visual question-answering samples. Delta-QA unifies pixel-level segmentation and visual question answering across bi- and tri-temporal scenarios, structuring change interpretation into four progressive cognitive dimensions. Methodologically, we propose Delta-LLaVA, a novel MLLM framework explicitly tailored for multi-temporal remote sensing interpretation. It overcomes the limitations of naive feature concatenation through three core innovations: a Change-Enhanced Attention module that systematically isolates and amplifies visual differences, a Change-SEG module utilizing Change Prior Embedding to extract differentiable difference features as input for the LLM, and Local Causal Attention to prevent cross-temporal contextual leakage. Extensive experiments demonstrate that Delta-LLaVA decisively outperforms leading generalist MLLMs and specialized segmentation models in complex change deduction and high-precision boundary localization, establishing a unified framework for earth observation intelligence.
>
---
#### [new 093] SceneGlue: Scene-Aware Transformer for Feature Matching without Scene-Level Annotation
- **分类: cs.CV**

- **简介: 该论文提出SceneGlue，用于跨视图特征匹配任务，解决传统方法无法捕捉全局场景信息的问题。通过结合注意力机制和可见性估计，提升匹配精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.13941](https://arxiv.org/pdf/2604.13941)**

> **作者:** Songlin Du; Xiaoyong Lu; Yaping Yan; Guobao Xiao; Xiaobo Lu; Takeshi Ikenaga
>
> **摘要:** Local feature matching plays a critical role in understanding the correspondence between cross-view images. However, traditional methods are constrained by the inherent local nature of feature descriptors, limiting their ability to capture non-local scene information that is essential for accurate cross-view correspondence. In this paper, we introduce SceneGlue, a scene-aware feature matching framework designed to overcome these limitations. SceneGlue leverages a hybridizable matching paradigm that integrates implicit parallel attention and explicit cross-view visibility estimation. The parallel attention mechanism simultaneously exchanges information among local descriptors within and across images, enhancing the scene's global context. To further enrich the scene awareness, we propose the Visibility Transformer, which explicitly categorizes features into visible and invisible regions, providing an understanding of cross-view scene visibility. By combining explicit and implicit scene-level awareness, SceneGlue effectively compensates for the local descriptor constraints. Notably, SceneGlue is trained using only local feature matches, without requiring scene-level groundtruth annotations. This scene-aware approach not only improves accuracy and robustness but also enhances interpretability compared to traditional methods. Extensive experiments on applications such as homography estimation, pose estimation, image matching, and visual localization validate SceneGlue's superior performance. The source code is available at this https URL.
>
---
#### [new 094] ESCAPE: Episodic Spatial Memory and Adaptive Execution Policy for Long-Horizon Mobile Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ESCAPE系统，解决长时空中机器人导航与操作的协调问题，通过时空融合映射和自适应执行策略提升任务成功率与路径效率。**

- **链接: [https://arxiv.org/pdf/2604.13633](https://arxiv.org/pdf/2604.13633)**

> **作者:** Jingjing Qian; Zeyuan He; Chen Shi; Lei Xiao; Li Jiang
>
> **摘要:** Coordinating navigation and manipulation with robust performance is essential for embodied AI in complex indoor environments. However, as tasks extend over long horizons, existing methods often struggle due to catastrophic forgetting, spatial inconsistency, and rigid execution. To address these issues, we propose ESCAPE (Episodic Spatial Memory Coupled with an Adaptive Policy for Execution), operating through a tightly coupled perception-grounding-execution workflow. For robust perception, ESCAPE features a Spatio-Temporal Fusion Mapping module to autoregressively construct a depth-free, persistent 3D spatial memory, alongside a Memory-Driven Target Grounding module for precise interaction mask generation. To achieve flexible action, our Adaptive Execution Policy dynamically orchestrates proactive global navigation and reactive local manipulation to seize opportunistic targets. ESCAPE achieves state-of-the-art performance on the ALFRED benchmark, reaching 65.09% and 60.79% success rates in test seen and unseen environments with step-by-step instructions. By reducing redundant exploration, our ESCAPE attains substantial improvements in path-length-weighted metrics and maintains robust performance (61.24% / 56.04%) even without detailed guidance for long-horizon tasks.
>
---
#### [new 095] Remote Sensing Image Super-Resolution for Imbalanced Textures: A Texture-Aware Diffusion Framework
- **分类: cs.CV**

- **简介: 该论文属于遥感图像超分辨率任务，解决纹理分布不均导致的重建质量下降问题。提出TexADiff框架，通过纹理密度图提升模型对纹理的感知与生成能力。**

- **链接: [https://arxiv.org/pdf/2604.13994](https://arxiv.org/pdf/2604.13994)**

> **作者:** Enzhuo Zhang; Sijie Zhao; Dilxat Muhtar; Zhenshi Li; Xueliang Zhang; Pengfeng Xiao
>
> **备注:** 10 pages, 5 figures, 9 Tables
>
> **摘要:** Generative diffusion priors have recently achieved state-of-the-art performance in natural image super-resolution, demonstrating a powerful capability to synthesize photorealistic details. However, their direct application to remote sensing image super-resolution (RSISR) reveals significant shortcomings. Unlike natural images, remote sensing images exhibit a unique texture distribution where ground objects are globally stochastic yet locally clustered, leading to highly imbalanced textures. This imbalance severely hinders the model's spatial perception. To address this, we propose TexADiff, a novel framework that begins by estimating a Relative Texture Density Map (RTDM) to represent the texture distribution. TexADiff then leverages this RTDM in three synergistic ways: as an explicit spatial conditioning to guide the diffusion process, as a loss modulation term to prioritize texture-rich regions, and as a dynamic adapter for the sampling schedule. These modifications are designed to endow the model with explicit texture-aware capabilities. Experiments demonstrate that TexADiff achieves superior or competitive quantitative metrics. Furthermore, qualitative results show that our model generates faithful high-frequency details while effectively suppressing texture hallucinations. This improved reconstruction quality also results in significant gains in downstream task performance. The source code of our method can be found at this https URL.
>
---
#### [new 096] Rethinking Uncertainty in Segmentation: From Estimation to Decision
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分割任务，研究如何将不确定性转化为决策策略，解决不确定性未有效指导决策的问题。通过实验验证不同方法在提升分割安全性上的效果。**

- **链接: [https://arxiv.org/pdf/2604.13262](https://arxiv.org/pdf/2604.13262)**

> **作者:** Saket Maganti
>
> **备注:** 29 pages, 12 tables, 9 figures, Github repo: Saket-Maganti/medical-seg-uncertainity
>
> **摘要:** In medical image segmentation, uncertainty estimates are often reported but rarely used to guide decisions. We study the missing step: how uncertainty maps are converted into actionable policies such as accepting, flagging, or deferring predictions. We formulate segmentation as a two-stage pipeline, estimation followed by decision, and show that optimizing uncertainty alone fails to capture most of the achievable safety gains. Using retinal vessel segmentation benchmarks (DRIVE, STARE, CHASE_DB1), we evaluate two uncertainty sources (Monte Carlo Dropout and Test-Time Augmentation) combined with three deferral strategies, and introduce a simple confidence-aware deferral rule that prioritizes uncertain and low-confidence predictions. Our results show that the best method and policy combination removes up to 80 percent of segmentation errors at only 25 percent pixel deferral, while achieving strong cross-dataset robustness. We further show that calibration improvements do not translate to better decision quality, highlighting a disconnect between standard uncertainty metrics and real-world utility. These findings suggest that uncertainty should be evaluated based on the decisions it enables, rather than in isolation.
>
---
#### [new 097] Seedance 2.0: Advancing Video Generation for World Complexity
- **分类: cs.CV**

- **简介: 该论文介绍Seedance 2.0，一种多模态音视频生成模型，解决复杂世界视频生成问题，通过统一架构提升生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.14148](https://arxiv.org/pdf/2604.14148)**

> **作者:** Team Seedance; De Chen; Liyang Chen; Xin Chen; Ying Chen; Zhuo Chen; Zhuowei Chen; Feng Cheng; Tianheng Cheng; Yufeng Cheng; Mojie Chi; Xuyan Chi; Jian Cong; Qinpeng Cui; Fei Ding; Qide Dong; Yujiao Du; Haojie Duanmu; Junliang Fan; Jiarui Fang; Jing Fang; Zetao Fang; Chengjian Feng; Yu Gao; Diandian Gu; Dong Guo; Hanzhong Guo; Qiushan Guo; Boyang Hao; Hongxiang Hao; Haoxun He; Jiaao He; Qian He; Tuyen Hoang; Heng Hu; Ruoqing Hu; Yuxiang Hu; Jiancheng Huang; Weilin Huang; Zhaoyang Huang; Zhongyi Huang; Jishuo Jin; Ming Jing; Ashley Kim; Shanshan Lao; Yichong Leng; Bingchuan Li; Gen Li; Haifeng Li; Huixia Li; Jiashi Li; Ming Li; Xiaojie Li; Xingxing Li; Yameng Li; Yiying Li; Yu Li; Yueyan Li; Chao Liang; Han Liang; Jianzhong Liang; Ying Liang; Wang Liao; J. H. Lien; Shanchuan Lin; Xi Lin; Feng Ling; Yue Ling; Fangfang Liu; Jiawei Liu; Jihao Liu; Jingtuo Liu; Shu Liu; Sichao Liu; Wei Liu; Xue Liu; Zuxi Liu; Ruijie Lu; Lecheng Lyu; Jingting Ma; Tianxiang Ma; Xiaonan Nie; Jingzhe Ning; Junjie Pan; Xitong Pan; Ronggui Peng; Xueqiong Qu; Yuxi Ren; Yuchen Shen; Guang Shi; Lei Shi; Yinglong Song; Fan Sun; Li Sun; Renfei Sun; Wenjing Tang; Boyang Tao; Zirui Tao; Dongliang Wang; Feng Wang
>
> **备注:** Seedance 2.0 Model Card
>
> **摘要:** Seedance 2.0 is a new native multi-modal audio-video generation model, officially released in China in early February 2026. Compared with its predecessors, Seedance 1.0 and 1.5 Pro, Seedance 2.0 adopts a unified, highly efficient, and large-scale architecture for multi-modal audio-video joint generation. This allows it to support four input modalities: text, image, audio, and video, by integrating one of the most comprehensive suites of multi-modal content reference and editing capabilities available in the industry to date. It delivers substantial, well-rounded improvements across all key sub-dimensions of video and audio generation. In both expert evaluations and public user tests, the model has demonstrated performance on par with the leading levels in the field. Seedance 2.0 supports direct generation of audio-video content with durations ranging from 4 to 15 seconds, with native output resolutions of 480p and 720p. For multi-modal inputs as reference, its current open platform supports up to 3 video clips, 9 images, and 3 audio clips. In addition, we provide Seedance 2.0 Fast version, an accelerated variant of Seedance 2.0 designed to boost generation speed for low-latency scenarios. Seedance 2.0 has delivered significant improvements to its foundational generation capabilities and multi-modal generation performance, bringing an enhanced creative experience for end users.
>
---
#### [new 098] Granularity-Aware Transfer for Tree Instance Segmentation in Synthetic and Real Forests
- **分类: cs.CV**

- **简介: 该论文属于森林场景下的树实例分割任务，解决合成数据到真实数据的迁移问题。通过引入混合粒度数据集和结构先验迁移方法，提升小树和远距离树的分割效果。**

- **链接: [https://arxiv.org/pdf/2604.13722](https://arxiv.org/pdf/2604.13722)**

> **作者:** Pankaj Deoli; Atef Tej; Anmol Ashri; Anandatirtha JS; Karsten Berns
>
> **摘要:** We address the challenge of synthetic-to-real transfer in forestry perception where real data have only coarse Tree labels while synthetic data provide fine-grained trunk/crown annotations. We introduce MGTD, a mixed-granularity dataset with 53k synthetic and 3.6k real images, and a four-stage protocol isolating domain shift and granularity mismatch. Our core contribution is granularity-aware distillation, which transfers structural priors from fine-grained synthetic teachers to a coarse-label student via logit-space merging and mask unification. Experiments show consistent mask AP gains, especially for small/distant trees, establishing a testbed for Sim-Real transfer under label granularity constraints.
>
---
#### [new 099] Artificial intelligence application in lymphoma diagnosis with Vision Transformer using weakly supervised training
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决淋巴瘤诊断问题。通过改进训练方式，使用弱监督的Vision Transformer模型提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2604.13795](https://arxiv.org/pdf/2604.13795)**

> **作者:** Nghia; Nguyen; Amer Wahed; Andy Quesada; Yasir Ali; Hanadi El Achi; Y. Helen Zhang; Jocelyn Ursua; Alex Banerjee; Sahib Kalra; L. Jeffrey Medeiros; Jie Xu
>
> **备注:** 23 pages, 6 figures, 1 table
>
> **摘要:** Vision transformers (ViT) have been shown to allow for more flexible feature detection and can outperform convolutional neural network (CNN) when pre-trained on sufficient data. Due to their promising feature detection capabilities, we deployed ViTs for morphological classification of anaplastic large cell lymphoma (ALCL) versus classic Hodgkin lymphoma (cHL). We had previously designed a ViT model which was trained on a small dataset of 1,200 image patches in fully supervised training. That model achieved a diagnostic accuracy of 100% and an F1 score of 1.0 on the independent test set. Since fully supervised training is not a practical method due to lack of expertise resources in both the training and testing phases, we conducted a recent study on a modified approach to training data (weakly supervised training) and show that labeling training image patch automatically at the slide level of each whole-slide-image is a more practical solution for clinical use of Vision Transformer. Our ViT model, trained on a larger dataset of 100,000 image patches, yields evaluation metrics with significant accuracy, F1 score, and area under the curve (AUC) at 91.85%, 0.92, and 0.98, respectively. These are respectable values that qualify this ViT model, with weakly supervised training, as a suitable tool for a deep learning module in clinical model development using automated image patch extraction.
>
---
#### [new 100] Indexing Multimodal Language Models for Large-scale Image Retrieval
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文研究将多模态大语言模型用于大规模图像检索任务，解决其在视觉任务中的潜力不足问题。通过提示模型生成相似度评分，实现无需微调的零样本重排序。**

- **链接: [https://arxiv.org/pdf/2604.13268](https://arxiv.org/pdf/2604.13268)**

> **作者:** Bahey Tharwat; Giorgos Kordopatis-Zilos; Pavel Suma; Ian Reid; Giorgos Tolias
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated strong cross-modal reasoning capabilities, yet their potential for vision-only tasks remains underexplored. We investigate MLLMs as training-free similarity estimators for instance-level image-to-image retrieval. Our approach prompts the model with paired images and converts next-token probabilities into similarity scores, enabling zero-shot re-ranking within large-scale retrieval pipelines. This design avoids specialized architectures and fine-tuning, leveraging the rich visual discrimination learned during multimodal pre-training. We address scalability by combining MLLMs with memory-efficient indexing and top-$k$ candidate re-ranking. Experiments across diverse benchmarks show that MLLMs outperform task-specific re-rankers outside their native domains and exhibit superior robustness to clutter, occlusion, and small objects. Despite strong results, we identify failure modes under severe appearance changes, highlighting opportunities for future research. Our findings position MLLMs as a promising alternative for open-world large-scale image retrieval.
>
---
#### [new 101] Event-Adaptive State Transition and Gated Fusion for RGB-Event Object Tracking
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于RGB-事件目标跟踪任务，解决现有方法因静态状态转移矩阵导致的模型不平衡问题。提出MambaTrack框架，引入动态状态转移机制和门控融合模块，提升跨模态融合效果。**

- **链接: [https://arxiv.org/pdf/2604.13426](https://arxiv.org/pdf/2604.13426)**

> **作者:** Jinlin You; Muyu Li; Xudong Zhao
>
> **摘要:** Existing Vision Mamba-based RGB-Event(RGBE) tracking methods suffer from using static state transition matrices, which fail to adapt to variations in event sparsity. This rigidity leads to imbalanced modeling-underfitting sparse event streams and overfitting dense ones-thus degrading cross-modal fusion robustness. To address these limitations, we propose MambaTrack, a multimodal and efficient tracking framework built upon a Dynamic State Space Model(DSSM). Our contributions are twofold. First, we introduce an event-adaptive state transition mechanism that dynamically modulates the state transition matrix based on event stream density. A learnable scalar governs the state evolution rate, enabling differentiated modeling of sparse and dense event flows. Second, we develop a Gated Projection Fusion(GPF) module for robust cross-modal integration. This module projects RGB features into the event feature space and generates adaptive gates from event density and RGB confidence scores. These gates precisely control the fusion intensity, suppressing noise while preserving complementary information. Experiments show that MambaTrack achieves state-of-the-art performance on the FE108 and FELT datasets. Its lightweight design suggests potential for real-time embedded deployment.
>
---
#### [new 102] ClipGStream: Clip-Stream Gaussian Splatting for Any Length and Any Motion Multi-View Dynamic Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于动态三维场景重建任务，解决长序列、大范围运动下的重建难题。提出ClipGStream框架，结合剪辑级优化与残差补偿，提升重建质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.13746](https://arxiv.org/pdf/2604.13746)**

> **作者:** Jie Liang; Jiahao Wu; Chao Wang; Jiayu Yang; Xiaoyun Zheng; Kaiqiang Xiong; Zhanke Wang; Jinbo Yan; Feng Gao; Ronggang Wang
>
> **备注:** CVPR 2026, Project pages: this https URL
>
> **摘要:** Dynamic 3D scene reconstruction is essential for immersive media such as VR, MR, and XR, yet remains challenging for long multi-view sequences with large-scale motion. Existing dynamic Gaussian approaches are either Frame-Stream, offering scalability but poor temporal stability, or Clip, achieving local consistency at the cost of high memory and limited sequence length. We propose ClipGStream, a hybrid reconstruction framework that performs stream optimization at the clip level rather than the frame level. The sequence is divided into short clips, where dynamic motion is modeled using clip-independent spatio-temporal fields and residual anchor compensation to capture local variations efficiently, while inter-clip inherited anchors and decoders maintain structural consistency across clips. This Clip-Stream design enables scalable, flicker-free reconstruction of long dynamic videos with high temporal coherence and reduced memory overhead. Extensive experiments demonstrate that ClipGStream achieves state-of-the-art reconstruction quality and efficiency. The project page is available at: this https URL
>
---
#### [new 103] Can Cross-Layer Transcoders Replace Vision Transformer Activations? An Interpretable Perspective on Vision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型可解释性任务，旨在解决ViT内部激活难以解释的问题。通过引入跨层编解码器（CLTs），实现对ViT层间结构的可解释重构。**

- **链接: [https://arxiv.org/pdf/2604.13304](https://arxiv.org/pdf/2604.13304)**

> **作者:** Gerasimos Chatzoudis; Konstantinos D. Polyzos; Zhuowei Li; Difei Gu; Gemma E. Moran; Hao Wang; Dimitris N. Metaxas
>
> **摘要:** Understanding the internal activations of Vision Transformers (ViTs) is critical for building interpretable and trustworthy models. While Sparse Autoencoders (SAEs) have been used to extract human-interpretable features, they operate on individual layers and fail to capture the cross-layer computational structure of Transformers, as well as the relative significance of each layer in forming the last-layer representation. Alternatively, we introduce the adoption of Cross-Layer Transcoders (CLTs) as reliable, sparse, and depth-aware proxy models for MLP blocks in ViTs. CLTs use an encoder-decoder scheme to reconstruct each post-MLP activation from learned sparse embeddings of preceding layers, yielding a linear decomposition that transforms the final representation of ViTs from an opaque embedding into an additive, layer-resolved construction that enables faithful attribution and process-level interpretability. We train CLTs on CLIP ViT-B/32 and ViT-B/16 across CIFAR-100, COCO, and ImageNet-100. We show that CLTs achieve high reconstruction fidelity with post-MLP activations while preserving and even improving, in some cases, CLIP zero-shot classification accuracy. In terms of interpretability, we show that the cross-layer contribution scores provide faithful attribution, revealing that the final representation is concentrated in a smaller set of dominant layer-wise terms whose removal degrades performance and whose retention largely preserves it. These results showcase the significance of adopting CLTs as an alternative interpretable proxy of ViTs in the vision domain.
>
---
#### [new 104] Towards Multi-Object-Tracking with Radar on a Fast Moving Vehicle: On the Potential of Processing Radar in the Frequency Domain
- **分类: cs.RO; cs.AI; cs.CV; eess.IV; eess.SP**

- **简介: 该论文属于多目标跟踪任务，旨在提升雷达在高速移动车辆中的鲁棒性。通过频域处理雷达数据，解决噪声和结构误差问题，并提供场景中所有运动结构的信息。**

- **链接: [https://arxiv.org/pdf/2604.14013](https://arxiv.org/pdf/2604.14013)**

> **作者:** Tim Hansen; Arturo Gomez-Chavez; Ilya Shimchik; Andreas Birk
>
> **摘要:** We promote in this paper the processing of radar data in the frequency domain to achieve higher robustness against noise and structural errors, especially in comparison to feature-based methods. This holds also for high dynamics in the scene, i.e., ego-motion of the vehicle with the sensor plus the presence of an unknown number of other moving objects. In addition to the high robustness, the processing in the frequency domain has the so far neglected advantage that the underlying correlation based methods used for, e.g., registration, provide information about all moving structures in the scene. A typical automotive application case is overtaking maneuvers, which in the context of autonomous racing are used here as a motivating example. Initial experiments and results with Fourier SOFT in 2D (FS2D) are presented that use the Boreas dataset to demonstrate radar-only-odometry, i.e., radar-odometry without sensor-fusion, to support our arguments.
>
---
#### [new 105] MedRCube: A Multidimensional Framework for Fine-Grained and In-Depth Evaluation of MLLMs in Medical Imaging
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于医学影像领域，旨在解决MLLMs评估不足的问题。提出MedRCube框架，实现细粒度、深度评估，并揭示推理可信度与诊断性能的关系。**

- **链接: [https://arxiv.org/pdf/2604.13756](https://arxiv.org/pdf/2604.13756)**

> **作者:** Zhijie Bao; Fangke Chen; Licheng Bao; Chenhui Zhang; Wei Chen; Jiajie Peng; Zhongyu Wei
>
> **摘要:** The potential of Multimodal Large Language Models (MLLMs) in domain of medical imaging raise the demands of systematic and rigorous evaluation frameworks that are aligned with the real-world medical imaging practice. Existing practices that report single or coarse-grained metrics are lack the granularity required for specialized clinical support and fail to assess the reliability of reasoning mechanisms. To address this, we propose a paradigm shift toward multidimensional, fine-grained and in-depth evaluation. Based on a two-stage systematic construction pipeline designed for this paradigm, we instantiate it with MedRCube. We benchmark 33 MLLMs, \textit{Lingshu-32B} achieve top-tier performance. Crucially, MedRCube exposes a series of pronounced insights inaccessible under prior evaluation settings. Furthermore, we introduce a credibility evaluation subset to quantify reasoning credibility, uncover a highly significant positive association between shortcut behavior and diagnostic task performance, raising concerns for clinically trustworthy deployment. The resources of this work can be found at this https URL.
>
---
#### [new 106] Learning Class Difficulty in Imbalanced Histopathology Segmentation via Dynamic Focal Attention
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，解决类别不平衡问题。提出Dynamic Focal Attention机制，直接学习类别难度，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.13479](https://arxiv.org/pdf/2604.13479)**

> **作者:** Lakmali Nadeesha Kumari; Sen-Ching Samson Cheung
>
> **摘要:** Semantic segmentation of histopathology images under class imbalance is typically addressed through frequency-based loss reweighting, which implicitly assumes that rare classes are difficult. However, true difficulty also arises from morphological variability, boundary ambiguity, and contextual similarity-factors that frequency cannot capture. We propose Dynamic Focal Attention (DFA), a simple and efficient mechanism that learns class-specific difficulty directly within the cross-attention of query-based mask decoders. DFA introduces a learnable per-class bias to attention logits, enabling representation-level reweighting prior to prediction rather than gradient-level reweighting after prediction. Initialised from a log-frequency prior to prevent gradient starvation, the bias is optimised end-to-end, allowing the model to adaptively capture difficulty signals through training, effectively unifying frequency-based and difficulty-aware approaches under a common attention-bias framework. On three histopathology benchmarks (BDSA, BCSS, CRAG), DFA consistently improves Dice and IoU, matching or exceeding a difficulty-aware baseline without a separate estimator or additional training stage. These results demonstrate that encoding class difficulty at the representation level provides a principled alternative to conventional loss reweighting for imbalanced segmentation.
>
---
#### [new 107] Caption First, VQA Second: Knowledge Density, Not Task Format, Drives Multimodal Scaling
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文探讨了多模态大语言模型的扩展问题，指出任务格式不是瓶颈，而是训练数据的知识密度不足。通过提升知识密度，模型性能显著提升。**

- **链接: [https://arxiv.org/pdf/2604.13054](https://arxiv.org/pdf/2604.13054)**

> **作者:** Hongjian Zou; Yue Ge; Qi Ding; Yixuan Liao; Xiaoxin Chen
>
> **备注:** 23 pages, 4 figures, 10 tables. Preprint
>
> **摘要:** Multimodal large language models (MLLMs) have achieved rapid progress, yet their scaling behavior remains less clearly characterized and often less predictable than that of text-only LLMs. Increasing model size and task diversity often yields diminishing returns. In this work, we argue that the primary bottleneck in multimodal scaling is not task format, but knowledge density in training data. We first show that task-specific supervision such as Visual Question Answering (VQA) contributes little incremental semantic information beyond image captions: VQA signals can be reconstructed from captions with negligible performance loss. We then demonstrate that increasing knowledge density -- through structured caption enrichment and cross-modal knowledge injection -- leads to consistent performance improvements across multimodal and downstream benchmarks. Across controlled experiments, performance correlates more strongly with semantic coverage than with task diversity. These findings suggest that current MLLMs fail to scale primarily because training data lacks sufficient knowledge coverage. We advocate for knowledge-centric multimodal training as a principled foundation for scalable multimodal models.
>
---
#### [new 108] Multi-modal panoramic 3D outdoor datasets for place categorization
- **分类: cs.RO; cs.CV; cs.DB**

- **简介: 该论文属于场景分类任务，旨在解决户外环境的语义分类问题。作者构建了两个多模态3D数据集，用于提升场景分类性能。**

- **链接: [https://arxiv.org/pdf/2604.13142](https://arxiv.org/pdf/2604.13142)**

> **作者:** Hojung Jung; Yuki Oto; Oscar M. Mozos; Yumi Iwashita; Ryo Kurazume
>
> **备注:** This is the authors' manuscript. The final published article was presented at IROS 2026, and it is available at this https URL
>
> **摘要:** We present two multi-modal panoramic 3D outdoor (MPO) datasets for semantic place categorization with six categories: forest, coast, residential area, urban area and indoor/outdoor parking lot. The first dataset consists of 650 static panoramic scans of dense (9,000,000 points) 3D color and reflectance point clouds obtained using a FARO laser scanner with synchronized color images. The second dataset consists of 34,200 real-time panoramic scans of sparse (70,000 points) 3D reflectance point clouds obtained using a Velodyne laser scanner while driving a car. The datasets were obtained in the city of Fukuoka, Japan and are publicly available in [1], [2]. In addition, we compare several approaches for semantic place categorization with best results of 96.42% (dense) and 89.67% (sparse).
>
---
#### [new 109] A Unified Conditional Flow for Motion Generation, Editing, and Intra-Structural Retargeting
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于动作生成与编辑任务，解决传统方法在输入和表示上的不兼容问题。通过统一的条件流框架，实现文本驱动的动作生成、编辑和结构重定向。**

- **链接: [https://arxiv.org/pdf/2604.13427](https://arxiv.org/pdf/2604.13427)**

> **作者:** Junlin Li; Xinhao Song; Siqi Wang; Haibin Huang; Yili Zhao
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Text-driven motion editing and intra-structural retargeting, where source and target share topology but may differ in bone lengths, are traditionally handled by fragmented pipelines with incompatible inputs and representations: editing relies on specialized generative steering, while retargeting is deferred to geometric post-processing. We present a unifying perspective where both tasks are cast as instances of conditional transport within a single generative framework. By leveraging recent advances in flow matching, we demonstrate that editing and retargeting are fundamentally the same generative task, distinguished only by which conditioning signal, semantic or structural, is modulated during inference. We implement this vision via a rectified-flow motion model jointly conditioned on text prompts and target skeletal structures. Our architecture extends a DiT-style transformer with per-joint tokenization and explicit joint self-attention to strictly enforce kinematic dependencies, while a multi-condition classifier-free guidance strategy balances text adherence with skeletal conformity. Experiments on SnapMoGen and a multi-character Mixamo subset show that a single trained model supports text-to-motion generation, zero-shot editing, and zero-shot intra-structural retargeting. This unified approach simplifies deployment and improves structural consistency compared to task-specific baselines.
>
---
#### [new 110] Creo: From One-Shot Image Generation to Progressive, Co-Creative Ideation
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文提出Creo系统，解决文本到图像生成中用户控制力弱、创意受限的问题。通过多阶段生成和中间控制，提升用户参与度与输出多样性。**

- **链接: [https://arxiv.org/pdf/2604.13956](https://arxiv.org/pdf/2604.13956)**

> **作者:** Zoe De Simone; Angie Boggust; Fredo Durand; Ashia Wilson; Arvind Satyanarayan
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Text-to-image (T2I) systems enable rapid generation of high-fidelity imagery but are misaligned with how visual ideas develop. T2I systems generate outputs that make implicit visual decisions on behalf of the user, often introduce fine-grained details that can anchor users prematurely and limit their ability to keep options open early on, and cause unintended changes during editing that are difficult to correct and reduce users' sense of control. To address these concerns, we present Creo, a multi-stage T2I system that scaffolds image generation by progressing from rough sketches to high-resolution outputs, exposing intermediary abstractions where users can make incremental changes. Sketch-like abstractions invite user editing and allow users to keep design options open when ideas are still forming due to their provisional nature. Each stage in Creo can be modified with manual changes and AI-assisted operations, enabling fine-grained, step-wise control through a locking mechanism that preserves prior decisions so subsequent edits affect only specified regions or attributes. Users remain in the loop, making and verifying decisions across stages, while the system applies diffs instead of regenerating full images, reducing drift as fidelity increases. A comparative study with a one-shot baseline shows that participants felt stronger ownership over Creo outputs, as they were able to trace their decisions in building up the image. Furthermore, embedding-based analysis indicates that Creo outputs are less homogeneous than one-shot results. These findings suggest that multi-stage generation, combined with intermediate control and decision locking, is a key design principle for improving controllability, user agency, creativity, and output diversity in generative systems.
>
---
#### [new 111] Depth-Resolved Coral Reef Thermal Fields from Satellite SST and Sparse In-Situ Loggers Using Physics-Informed Neural Networks
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于海洋温度建模任务，解决卫星数据无法准确反映珊瑚礁深层温度的问题。通过物理信息神经网络融合卫星与现场数据，提升深度温度预测精度，改进珊瑚白化评估。**

- **链接: [https://arxiv.org/pdf/2604.13131](https://arxiv.org/pdf/2604.13131)**

> **作者:** Alzayat Saleh; Mostafa Rahimi Azghadi
>
> **备注:** 23 pages, 7 figures, submitted to Remote Sensing of Environment
>
> **摘要:** Satellite sea surface temperature (SST) products underpin global coral bleaching monitoring, yet they measure only the ocean skin. Corals inhabit depths from the shallows to beyond 20 metres, where temperatures can be 1-3°C cooler than the surface; applying satellite SST uniformly to all depths therefore overestimates subsurface thermal stress. We present a physics-informed neural network (PINN) that fuses NOAA Coral Reef Watch SST with sparse in-situ temperature loggers within the one-dimensional vertical heat equation, enforcing SST as a hard surface boundary condition and jointly learning effective thermal diffusivity (\k{appa}) and light attenuation (Kd). Validated across four Great Barrier Reef sites (30 holdout experiments), the PINN achieves 0.25-1.38°C RMSE at unseen depths. Under extreme sparsity (three training depths), the PINN maintains 0.27°C RMSE at the 5 metre holdout and 0.32°C at the 9.1 metre holdout, where statistical baselines collapse to >1.8°C; it outperforms a physics-only finite-difference baseline in 90% of experiments. Depth-resolved Degree Heating Day (DHD) profiles show that thermal stress attenuates with depth: at Davies Reef, DHD drops from 0.29 at the surface to zero by 10.7 metres, consistent with logger observations, while satellite DHD remains constant at 0.31 across all depths. However, the PINN underestimates absolute DHD at shallow depths because its smooth predictions attenuate the short-duration peaks that drive threshold exceedances; PINN DHD values should be interpreted as conservative lower bounds on depth-resolved stress. These results demonstrate that physics-constrained fusion of satellite SST with sparse loggers can extend bleaching assessment to the depth dimension using existing observational infrastructure.
>
---
#### [new 112] MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于多类分类任务，解决鸡胸肉肌病（正常、木质肌、面条肌）的低成本、非破坏性检测问题。提出MyoVision框架和NEATBoost-Attention模型，实现高精度分类。**

- **链接: [https://arxiv.org/pdf/2604.13456](https://arxiv.org/pdf/2604.13456)**

> **作者:** Chaitanya Pallerla; Siavash Mahmoudi; Dongyi Wang
>
> **备注:** Accepted at CVPR 2026 MetaFoods Workshop. 11 pages, 5 figures
>
> **摘要:** Woody Breast (WB) and Spaghetti Meat (SM) myopathies significantly impact poultry meat quality, yet current detection methods rely either on subjective manual evaluation or costly laboratory-grade imaging systems. We address the problem of low-cost, non-destructive multi-class myopathy classification using consumer smartphones. MyoVision is introduced as a mobile transillumination imaging framework in which 14-bit RAW images are captured and structural texture descriptors indicative of internal tissue abnormalities are extracted. To classify three categories (Normal, Woody Breast, Spaghetti Meat), we propose a NEATBoost-Attention Ensemble model, which is a neuroevolution-optimized weighted fusion of LightGBM and attention-based MLP models. Hyperparameters are automatically discovered using NeuroEvolution of Augmenting Topologies (NEAT), eliminating manual tuning and enabling architecture diversity for small tabular datasets. On a dataset of 336 fillets collected from a commercial processing facility, our method achieves 82.4% test accuracy (F1 = 0.83), outperforming conventional machine learning and deep learning baselines and matching performance reported by hyperspectral imaging systems costing orders of magnitude more. Beyond classification performance, MyoVision establishes a reproducible mobile RGB-D acquisition pipeline for multimodal meat quality research, demonstrating that consumer-grade imaging can support scalable internal tissue assessment.
>
---
#### [new 113] Who Gets Flagged? The Pluralistic Evaluation Gap in AI Content Watermarking
- **分类: cs.CY; cs.CL; cs.CR; cs.CV**

- **简介: 该论文属于AI内容验证任务，探讨水印技术在不同语言、文化及群体中的公平性问题，提出多维评估框架以解决检测偏差。**

- **链接: [https://arxiv.org/pdf/2604.13776](https://arxiv.org/pdf/2604.13776)**

> **作者:** Alexander Nemecek; Osama Zafar; Yuqiao Xu; Wenbiao Li; Erman Ayday
>
> **备注:** 7 pages
>
> **摘要:** Watermarking is becoming the default mechanism for AI content authentication, with governance policies and frameworks referencing it as infrastructure for content provenance. Yet across text, image, and audio modalities, watermark signal strength, detectability, and robustness depend on statistical properties of the content itself, properties that vary systematically across languages, cultural visual traditions, and demographic groups. We examine how this content dependence creates modality-specific pathways to bias. Reviewing the major watermarking benchmarks across modalities, we find that, with one exception, none report performance across languages, cultural content types, or population groups. To address this, we propose three concrete evaluation dimensions for pluralistic watermark benchmarking: cross-lingual detection parity, culturally diverse content coverage, and demographic disaggregation of detection metrics. We connect these to the governance frameworks currently mandating watermarking deployment and show that watermarking is held to a lower fairness standard than the generative systems it is meant to govern. Our position is that evaluation must precede deployment, and that the same bias auditing requirements applied to AI models should extend to the verification layer.
>
---
#### [new 114] Evolvable Embodied Agent for Robotic Manipulation via Long Short-Term Reflection and Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决机器人自适应与进化问题。提出EEAgent框架和LSTRO机制，提升机器人环境理解与任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.13533](https://arxiv.org/pdf/2604.13533)**

> **作者:** Jianzong Wang; Botao Zhao; Yayun He; Junqing Peng; Xulong Zhang
>
> **备注:** This work has been accepted for publication in the Proceedings of the 2026 International Joint Conference on Neural Networks (IJCNN 2026)
>
> **摘要:** Achieving general-purpose robotics requires empowering robots to adapt and evolve based on their environment and feedback. Traditional methods face limitations such as extensive training requirements, difficulties in cross-task generalization, and lack of interpretability. Prompt learning offers new opportunities for self-evolving robots without extensive training, but simply reflecting on past this http URL, extracting meaningful insights from task successes and failures remains a challenge. To this end, we propose the evolvable embodied agent (EEAgent) framework, which leverages large vision-language models (VLMs) for better environmental interpretation and policy planning. To enhance reflection on past experiences, we propose a long short-term reflective optimization (LSTRO) mechanism that dynamically refines prompts based on both past experiences and newly learned lessons, facilitating continuous self-evolution, thereby enhancing overall task success rates. Evaluations on six VIMA-Bench tasks reveal that our approach sets a new state-of-the-art, notably outperforming baselines in complex scenarios.
>
---
#### [new 115] Failure Identification in Imitation Learning Via Statistical and Semantic Filtering
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人故障检测任务，解决真实场景下模仿学习策略的失败识别问题。提出FIDeL模块，结合统计与语义过滤提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.13788](https://arxiv.org/pdf/2604.13788)**

> **作者:** Quentin Rolland; Fabrice Mayran de Chamisso; Jean-Baptiste Mouret
>
> **备注:** 8 pages, Appendix coming soon, accepted at ICRA 2026
>
> **摘要:** Imitation learning (IL) policies in robotics deliver strong performance in controlled settings but remain brittle in real-world deployments: rare events such as hardware faults, defective parts, unexpected human actions, or any state that lies outside the training distribution can lead to failed executions. Vision-based Anomaly Detection (AD) methods emerged as an appropriate solution to detect these anomalous failure states but do not distinguish failures from benign deviations. We introduce FIDeL (Failure Identification in Demonstration Learning), a policy-independent failure detection module. Leveraging recent AD methods, FIDeL builds a compact representation of nominal demonstrations and aligns incoming observations via optimal transport matching to produce anomaly scores and heatmaps. Spatio-temporal thresholds are derived with an extension of conformal prediction, and a Vision-Language Model (VLM) performs semantic filtering to discriminate benign anomalies from genuine failures. We also introduce BotFails, a multimodal dataset of real-world tasks for failure detection in robotics. FIDeL consistently outperforms state-of-the-art baselines, yielding +5.30% percent AUROC in anomaly detection and +17.38% percent failure-detection accuracy on BotFails compared to existing methods.
>
---
#### [new 116] MERRIN: A Benchmark for Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出MERRIN基准，用于评估搜索增强代理在噪声网络环境中的多模态证据检索与推理能力，解决多跳查询和异构信息处理问题。**

- **链接: [https://arxiv.org/pdf/2604.13418](https://arxiv.org/pdf/2604.13418)**

> **作者:** Han Wang; David Wan; Hyunji Lee; Thinh Pham; Mikaela Cankosyan; Weiyuan Chen; Elias Stengel-Eskin; Tu Vu; Mohit Bansal
>
> **备注:** First three authors contributed equally. Project Page: this https URL
>
> **摘要:** Motivated by the underspecified, multi-hop nature of search queries and the multimodal, heterogeneous, and often conflicting nature of real-world web results, we introduce MERRIN (Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments), a human-annotated benchmark for evaluating search-augmented agents. MERRIN measures AI agents' ability to identify relevant modalities, retrieve multimodal evidence, and perform multi-hop reasoning over noisy web sources. It differs from prior work in three important aspects: (1) using natural language queries without explicit modality cues, (2) incorporating underexplored modalities such as video and audio, and (3) requiring the retrieval of complex, often noisy or conflicting multimodal evidence during web search. We evaluate diverse search agents powered by ten models, including strong closed-source models (e.g., GPT-5.4-mini, Gemini 3/3.1 Flash/Pro) and open-weight models (Qwen3-4B/30B/235B), across three search settings (no search, native search, and agentic search). Our results show that MERRIN is highly challenging: the average accuracy across all agents is 22.3%, with the best-performing agent reaching only 40.1%. We further observe that while stronger agents like Gemini Deep Research achieve higher performance, gains are modest due to over-exploration; they take more steps and use more tools, but are often distracted by conflicting or partially relevant web content, leading to incorrect answers. Compared to humans, these agents consume more resources yet achieve lower accuracy, largely due to inefficient source selection and an overreliance on text modalities. These findings highlight the need for search agents capable of robust search and reasoning across diverse modalities in noisy web environments, making MERRIN a valuable testbed for evaluating such capabilities.
>
---
#### [new 117] Automatic Charge State Tuning of 300 mm FDSOI Quantum Dots Using Neural Network Segmentation of Charge Stability Diagram
- **分类: cond-mat.mes-hall; cs.CV; cs.LG**

- **简介: 该论文属于量子点调控任务，旨在解决硅基量子点电荷自动调谐难题。通过神经网络分割电荷稳定性图，实现高效电荷状态定位与优化。**

- **链接: [https://arxiv.org/pdf/2604.13662](https://arxiv.org/pdf/2604.13662)**

> **作者:** Peter Samaha; Amine Torki; Ysaline Renaud; Sam Fiette; Emmanuel Chanrion; Pierre-Andre Mortemousque; Yann Beilliard
>
> **备注:** 10 pages, 6 figures, supplementary materials available
>
> **摘要:** Tuning of gate-defined semiconductor quantum dots (QDs) is a major bottleneck for scaling spin qubit technologies. We present a deep learning (DL) driven, semantic-segmentation pipeline that performs charge auto-tuning by locating transition lines in full charge stability diagrams (CSDs) and returns gate voltage targets for the single charge regime. We assemble and manually annotate a large, heterogeneous dataset of 1015 experimental CSDs measured from silicon QD devices, spanning nine design geometries, multiple wafers, and fabrication runs. A U-Net style convolutional neural network (CNN) with a MobileNetV2 encoder is trained and validated through five-fold group cross validation. Our model achieves an overall offline tuning success of 80.0% in locating the single-charge regime, with peak performance exceeding 88% for some designs. We analyze dominant failure modes and propose targeted mitigations. Finally, wide-range diagram segmentation also naturally enables scalable physic-based feature extraction that can feed back to fabrication and design workflows and outline a roadmap for real-time integration in a cryogenic wafer prober. Overall, our results show that neural network (NN) based wide-diagram segmentation is a practical step toward automated, high-throughput charge tuning for silicon QD qubits.
>
---
#### [new 118] RadarSplat-RIO: Indoor Radar-Inertial Odometry with Gaussian Splatting-Based Radar Bundle Adjustment
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于室内雷达-惯性里程计任务，旨在解决雷达SLAM中漂移问题。提出基于高斯点云的雷达BA框架，联合优化位姿与场景几何，显著降低误差。**

- **链接: [https://arxiv.org/pdf/2604.13492](https://arxiv.org/pdf/2604.13492)**

> **作者:** Pou-Chun Kung; Yuan Tian; Zhengqin Li; Yue Liu; Eric Whitmire; Wolf Kienzle; Hrvoje Benko
>
> **摘要:** Radar is more resilient to adverse weather and lighting conditions than visual and Lidar simultaneous localization and mapping (SLAM). However, most radar SLAM pipelines still rely heavily on frame-to-frame odometry, which leads to substantial drift. While loop closure can correct long-term errors, it requires revisiting places and relies on robust place recognition. In contrast, visual odometry methods typically leverage bundle adjustment (BA) to jointly optimize poses and map within a local window. However, an equivalent BA formulation for radar has remained largely unexplored. We present the first radar BA framework enabled by Gaussian Splatting (GS), a dense and differentiable scene representation. Our method jointly optimizes radar sensor poses and scene geometry using full range-azimuth-Doppler data, bringing the benefits of multi-frame BA to radar for the first time. When integrated with an existing radar-inertial odometry frontend, our approach significantly reduces pose drift and improves robustness. Across multiple indoor scenes, our radar BA achieves substantial gains over the prior radar-inertial odometry, reducing average absolute translational and rotational errors by 90% and 80%, respectively.
>
---
#### [new 119] PersonaVLM: Long-Term Personalized Multimodal LLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出PersonaVLM，解决长期个性化问题。通过记忆、推理和响应对齐，提升多模态大模型的个性化能力。**

- **链接: [https://arxiv.org/pdf/2604.13074](https://arxiv.org/pdf/2604.13074)**

> **作者:** Chang Nie; Chaoyou Fu; Yifan Zhang; Haihua Yang; Caifeng Shan
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) serve as daily assistants for millions. However, their ability to generate responses aligned with individual preferences remains limited. Prior approaches enable only static, single-turn personalization through input augmentation or output alignment, and thus fail to capture users' evolving preferences and personality over time (see Fig.1). In this paper, we introduce PersonaVLM, an innovative personalized multimodal agent framework designed for long-term personalization. It transforms a general-purpose MLLM into a personalized assistant by integrating three key capabilities: (a) Remembering: It proactively extracts and summarizes chronological multimodal memories from interactions, consolidating them into a personalized database. (b) Reasoning: It conducts multi-turn reasoning by retrieving and integrating relevant memories from the database. (c) Response Alignment: It infers the user's evolving personality throughout long-term interactions to ensure outputs remain aligned with their unique characteristics. For evaluation, we establish Persona-MME, a comprehensive benchmark comprising over 2,000 curated interaction cases, designed to assess long-term MLLM personalization across seven key aspects and 14 fine-grained tasks. Extensive experiments validate our method's effectiveness, improving the baseline by 22.4% (Persona-MME) and 9.8% (PERSONAMEM) under a 128k context, while outperforming GPT-4o by 5.2% and 2.0%, respectively. Project page: this https URL.
>
---
#### [new 120] Reward Design for Physical Reasoning in Vision-Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型的物理推理任务，旨在提升模型在物理问题上的推理能力。通过设计不同奖励机制，研究奖励对模型表现的影响，发现准确率奖励效果最佳，注意力奖励有助于空间推理。**

- **链接: [https://arxiv.org/pdf/2604.13993](https://arxiv.org/pdf/2604.13993)**

> **作者:** Derek Lilienthal; Manisha Mukherjee; Sameera Horawalavithana
>
> **摘要:** Physical reasoning over visual inputs demands tight integration of visual perception, domain knowledge, and multi-step symbolic inference. Yet even state-of-the-art Vision Language Models (VLMs) fall far short of human performance on physics benchmarks. While post-training algorithms such as Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) have demonstrated strong reasoning gains in language models, how reward design shapes VLM physical reasoning behavior remains poorly understood. We present a systematic reward ablation study for GRPO-based VLM training on physical reasoning. We compare four reward signals of increasing semantic richness: format compliance, answer accuracy, a composite rubric reward (answer correctness, physics principle identification, and unit consistency), and a novel internal reward derived from model attention weights over input image regions. We evaluate on PhyX, a 3,000-problem benchmark spanning six physics domains and six reasoning types across multiple-choice and open-ended formats, using IBM Granite Vision 3.3 (2B). Across both formats, GRPO with accuracy-based rewards outperforms SFT on most domains, though gains vary substantially by reward type and domain. Reward design does not uniformly improve performance. Instead, it induces domain-specific reasoning behaviors. Accuracy-based rewards provide the strongest overall gains. Rubric rewards improve structured reasoning quality without consistent accuracy improvements. Attention-based rewards enhance spatial reasoning while degrading performance in symbolic domains. Our internal attention-weight reward requires no spatial annotations and improves spatial relation accuracy from 0.27 to 0.50, suggesting that supervising where the model attends during generation is a promising direction for visually grounded physical reasoning.
>
---
#### [new 121] C$^2$T: Captioning-Structure and LLM-Aligned Common-Sense Reward Learning for Traffic--Vehicle Coordination
- **分类: cs.MA; cs.CV; cs.RO**

- **简介: 该论文属于交通控制任务，旨在解决传统奖励机制无法体现人类目标的问题。提出C2T框架，利用LLM知识提升奖励函数，优化交通效率与安全。**

- **链接: [https://arxiv.org/pdf/2604.13098](https://arxiv.org/pdf/2604.13098)**

> **作者:** Yuyang Chen; Kaiyan Zhao; Yiming Wang; Ming Yang; Bin Rao; Zhenning Li
>
> **备注:** Accepted to CVPR 2026 Findings Track
>
> **摘要:** State-of-the-art (SOTA) urban traffic control increasingly employs Multi-Agent Reinforcement Learning (MARL) to coordinate Traffic Light Controllers (TLCs) and Connected Autonomous Vehicles (CAVs). However, the performance of these systems is fundamentally capped by their hand-crafted, myopic rewards (e.g., intersection pressure), which fail to capture high-level, human-centric goals like safety, flow stability, and comfort. To overcome this limitation, we introduce C2T, a novel framework that learns a common-sense coordination model from traffic-vehicle dynamics. C2T distills "common-sense" knowledge from a Large Language Model (LLM) into a learned intrinsic reward function. This new reward is then used to guide the coordination policy of a cooperative multi-intersection TLC MARL system on CityFlow-based multi-intersection benchmarks. Our framework significantly outperforms strong MARL baselines in traffic efficiency, safety, and an energy-related proxy. We further highlight C2T's flexibility in principle, allowing distinct "efficiency-focused" versus "safety-focused" policies by modifying the LLM prompt.
>
---
#### [new 122] RobotPan: A 360$^\circ$ Surround-View Robotic Vision System for Embodied Perception
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RobotPan系统，解决机器人环绕视觉感知问题，通过多相机与LiDAR融合，实现360°实时3D重建与渲染，提升导航与操作效率。**

- **链接: [https://arxiv.org/pdf/2604.13476](https://arxiv.org/pdf/2604.13476)**

> **作者:** Jiahao Ma; Qiang Zhang; Peiran Liu; Zeran Su; Pihai Sun; Gang Han; Wen Zhao; Wei Cui; Zhang Zhang; Zhiyuan Xu; Renjing Xu; Jian Tang; Miaomiao Liu; Yijie Guo
>
> **备注:** Project website: this https URL
>
> **摘要:** Surround-view perception is increasingly important for robotic navigation and loco-manipulation, especially in human-in-the-loop settings such as teleoperation, data collection, and emergency takeover. However, current robotic visual interfaces are often limited to narrow forward-facing views, or, when multiple on-board cameras are available, require cumbersome manual switching that interrupts the operator's workflow. Both configurations suffer from motion-induced jitter that causes simulator sickness in head-mounted displays. We introduce a surround-view robotic vision system that combines six cameras with LiDAR to provide full 360$^\circ$ visual coverage, while meeting the geometric and real-time constraints of embodied deployment. We further present \textsc{RobotPan}, a feed-forward framework that predicts \emph{metric-scaled} and \emph{compact} 3D Gaussians from calibrated sparse-view inputs for real-time rendering, reconstruction, and streaming. \textsc{RobotPan} lifts multi-view features into a unified spherical coordinate representation and decodes Gaussians using hierarchical spherical voxel priors, allocating fine resolution near the robot and coarser resolution at larger radii to reduce computational redundancy without sacrificing fidelity. To support long sequences, our online fusion updates dynamic content while preventing unbounded growth in static regions by selectively updating appearance. Finally, we release a multi-sensor dataset tailored to 360$^\circ$ novel view synthesis and metric 3D reconstruction for robotics, covering navigation, manipulation, and locomotion on real platforms. Experiments show that \textsc{RobotPan} achieves competitive quality against prior feed-forward reconstruction and view-synthesis methods while producing substantially fewer Gaussians, enabling practical real-time embodied deployment. Project website: this https URL
>
---
#### [new 123] ASTER: Latent Pseudo-Anomaly Generation for Unsupervised Time-Series Anomaly Detection
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于时间序列异常检测任务，旨在解决标注数据稀缺和复杂数据处理问题。提出ASTER框架，在潜在空间生成伪异常，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.13924](https://arxiv.org/pdf/2604.13924)**

> **作者:** Romain Hermary; Samet Hicsonmez; Dan Pineau; Abd El Rahman Shabayek; Djamila Aouada
>
> **摘要:** Time-series anomaly detection (TSAD) is critical in domains such as industrial monitoring, healthcare, and cybersecurity, but it remains challenging due to rare and heterogeneous anomalies and the scarcity of labelled data. This scarcity makes unsupervised approaches predominant, yet existing methods often rely on reconstruction or forecasting, which struggle with complex data, or on embedding-based approaches that require domain-specific anomaly synthesis and fixed distance metrics. We propose ASTER, a framework that generates pseudo-anomalies directly in the latent space, avoiding handcrafted anomaly injections and the need for domain expertise. A latent-space decoder produces tailored pseudo-anomalies to train a Transformer-based anomaly classifier, while a pre-trained LLM enriches the temporal and contextual representations of this space. Experiments on three benchmark datasets show that ASTER achieves state-of-the-art performance and sets a new standard for LLM-based TSAD.
>
---
## 更新

#### [replaced 001] Adaptive Multi-Scale Channel-Spatial Attention Aggregation Framework for 3D Indoor Semantic Scene Completion Toward Assisting Visually Impaired
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.16385](https://arxiv.org/pdf/2602.16385)**

> **作者:** Qi He; XiangXiang Wang; Jingtao Zhang; Yongbin Yu; Hongxiang Chu; Manping Fan; JingYe Cai; Zhenglin Yang
>
> **备注:** We need to optimize the experiment, the changes are quite significant
>
> **摘要:** Independent indoor mobility remains a critical challenge for individuals with visual impairments, largely due to the limited capability of existing assistive systems in detecting fine-grained hazardous objects such as chairs, tables, and small obstacles. These perceptual blind zones substantially increase the risk of collision in unfamiliar environments. To bridge the gap between monocular 3D vision research and practical assistive deployment, this paper proposes an Adaptive Multi-scale Attention Aggregation (AMAA) framework for monocular 3D semantic scene completion using only a wearable RGB camera. The proposed framework addresses two major limitations in 2D-to-3D feature lifting: noise diffusion during back-projection and structural instability in multi-scale fusion. A parallel channel--spatial attention mechanism is introduced to recalibrate lifted features along semantic and geometric dimensions, while a hierarchical adaptive gating strategy regulates cross-scale information flow to preserve fine-grained structural details. Experiments on the NYUv2 benchmark demonstrate that AMAA achieves an overall mIoU of 27.88%. Crucially, it yields significant relative improvements of 16.9% for small objects and 10.4% for tables over the MonoScene baseline. Furthermore, a wearable prototype based on an NVIDIA Jetson Orin NX and a ZED~2i camera validates stable real-time performance in indoor environments, demonstrating the feasibility of deploying monocular 3D scene completion for assistive navigation.
>
---
#### [replaced 002] A Function-Centric Perspective on Flat and Sharp Minima
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.12451](https://arxiv.org/pdf/2510.12451)**

> **作者:** Israel Mason-Williams; Gabryel Mason-Williams; Helen Yannakoudakis
>
> **备注:** 51 pages, 44 tables, 16 figures, pre-print
>
> **摘要:** Flat minima are strongly associated with improved generalisation in deep neural networks. However, this connection has proven nuanced in recent studies, with both theoretical counterexamples and empirical exceptions emerging in the literature. In this paper, we revisit the role of sharpness in model performance and argue that sharpness is better understood as a function-dependent property rather than an indicator of poor generalisation. We conduct extensive empirical studies ranging from single-objective optimisation, synthetic non-linear binary classification tasks, to modern image classification tasks. In single-objective optimisation, we show that flatness and sharpness are relative to the function being learned: equally optimal solutions can exhibit markedly different local geometry. In synthetic non-linear binary classification tasks, we show that increasing decision-boundary tightness can increase sharpness even when models generalise perfectly, indicating that sharpness is not reducible to memorisation alone. Finally, in large-scale experiments, we find that sharper minima often emerge when models are regularised (e.g., via weight decay, data augmentation, or SAM), and coincide with better generalisation, calibration, robustness, and functional consistency. Our findings suggest that function complexity, rather than flatness, shapes the geometry of solutions, and that sharper minima can reflect more appropriate inductive biases, calling for a function-centric reappraisal of minima geometry.
>
---
#### [replaced 003] Hybrid Approach for Enhancing Lesion Segmentation in Fundus Images
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.25549](https://arxiv.org/pdf/2509.25549)**

> **作者:** Mohammadmahdi Eshragh; Emad A. Mohammed; Behrouz Far; Ezekiel Weis; Carol L Shields; Sandor R Ferenczy; Trafford Crump
>
> **摘要:** Choroidal nevi are common benign pigmented lesions in the eye, with a small risk of transforming into melanoma. Early detection is critical to improving survival rates, but misdiagnosis or delayed diagnosis can lead to poor outcomes. Despite advancements in AI-based image analysis, diagnosing choroidal nevi in colour fundus images remains challenging, particularly for clinicians without specialized expertise. Existing datasets often suffer from low resolution and inconsistent labelling, limiting the effectiveness of segmentation models. This paper addresses the challenge of achieving precise segmentation of fundus lesions, a critical step toward developing robust diagnostic tools. While deep learning models like U-Net have demonstrated effectiveness, their accuracy heavily depends on the quality and quantity of annotated data. Previous mathematical/clustering segmentation methods, though accurate, required extensive human input, making them impractical for medical applications. This paper proposes a novel approach that combines mathematical/clustering segmentation models with insights from U-Net, leveraging the strengths of both methods. This hybrid model improves accuracy, reduces the need for large-scale training data, and achieves significant performance gains on high-resolution fundus images. The proposed model achieves a Dice coefficient of 89.7% and an IoU of 80.01% on 1024*1024 fundus images, outperforming the Attention U-Net model, which achieved 51.3% and 34.2%, respectively. It also demonstrated better generalizability on external datasets. This work forms a part of a broader effort to develop a decision support system for choroidal nevus diagnosis, with potential applications in automated lesion annotation to enhance the speed and accuracy of diagnosis and monitoring.
>
---
#### [replaced 004] The Second Challenge on Real-World Face Restoration at NTIRE 2026: Methods and Results
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10532](https://arxiv.org/pdf/2604.10532)**

> **作者:** Jingkai Wang; Jue Gong; Zheng Chen; Kai Liu; Jiatong Li; Yulun Zhang; Radu Timofte; Jiachen Tu; Yaokun Shi; Guoyi Xu; Yaoxin Jiang; Jiajia Liu; Yingsi Chen; Yijiao Liu; Hui Li; Yu Wang; Congchao Zhu; Alexandru-Gabriel Lefterache; Anamaria Radoi; Chuanyue Yan; Tao Lu; Yanduo Zhang; Kanghui Zhao; Jiaming Wang; Yuqi Li; WenBo Xiong; Yifei Chen; Xian Hu; Wei Deng; Daiguo Zhou; Sujith Roy V; Claudia Jesuraj; Vikas B; Spoorthi LC; Nikhil Akalwadi; Ramesh Ashok Tabib; Uma Mudenagudi; Yuxuan Jiang; Chengxi Zeng; Tianhao Peng; Fan Zhang; David Bull Wei Zhou; Linfeng Li; Hongyu Huang; Hoyoung Lee; SangYun Oh; ChangYoung Jeong; Axi Niu; Jinyang Zhang; Zhenguo Wu; Senyan Qing; Jinqiu Sun; Yanning Zhang
>
> **备注:** NTIRE 26: this https URL . NTIRE Real-World Face Restoration: this https URL . CVPR 2026 Workshop
>
> **摘要:** This paper provides a review of the NTIRE 2026 challenge on real-world face restoration, highlighting the proposed solutions and the resulting outcomes. The challenge focuses on generating natural and realistic outputs while maintaining identity consistency. Its goal is to advance state-of-the-art solutions for perceptual quality and realism, without imposing constraints on computational resources or training data. Performance is evaluated using a weighted image quality assessment (IQA) score and employs the AdaFace model as an identity checker. The competition attracted 96 registrants, with 10 teams submitting valid models; ultimately, 9 teams achieved valid scores in the final ranking. This collaborative effort advances the performance of real-world face restoration while offering an in-depth overview of the latest trends in the field.
>
---
#### [replaced 005] A Faster Path to Continual Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.11064](https://arxiv.org/pdf/2604.11064)**

> **作者:** Wei Li; Hangjie Yuan; Zixiang Zhao; Borui Kang; Ziwei Liu; Tao Feng
>
> **备注:** Update Author Affiliations
>
> **摘要:** Continual Learning (CL) aims to train neural networks on a dynamic stream of tasks without forgetting previously learned knowledge. Among optimization-based approaches, C-Flat has emerged as a promising solution due to its plug-and-play nature and its ability to encourage uniformly low-loss regions for both new and old tasks. However, C-Flat requires three additional gradient computations per iteration, imposing substantial overhead on the optimization process. In this work, we propose C-Flat Turbo, a faster yet stronger optimizer that significantly reduces the training cost. We show that the gradients associated with first-order flatness contain direction-invariant components relative to the proxy-model gradients, enabling us to skip redundant gradient computations in the perturbed ascent steps. Moreover, we observe that these flatness-promoting gradients progressively stabilize across tasks, which motivates a linear scheduling strategy with an adaptive trigger to allocate larger turbo steps for later tasks. Experiments show that C-Flat Turbo is 1.0$\times$ to 1.25$\times$ faster than C-Flat across a wide range of CL methods, while achieving comparable or even improved accuracy.
>
---
#### [replaced 006] GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02697](https://arxiv.org/pdf/2512.02697)**

> **作者:** Zixuan Song; Jing Zhang; Di Wang; Zidie Zhou; Wenbin Liu; Haonan Guo; En Wang; Bo Du
>
> **备注:** The paper is accepted by CVPR 2026! Code, dataset, and pretrained models will be released at this https URL
>
> **摘要:** Cross-view geo-localization infers a location by retrieving geo-tagged reference images that visually correspond to a query image. However, the traditional satellite-centric paradigm limits robustness when high-resolution or up-to-date satellite imagery is unavailable. It further underexploits complementary cues across views (\eg, drone, satellite, and street) and modalities (\eg, language and image). To address these challenges, we propose GeoBridge, a novel model that performs bidirectional matching across views and supports language-to-image retrieval. Going beyond traditional satellite-centric formulations, GeoBridge builds on a novel semantic-anchor mechanism that bridges multi-view features through textual descriptions for robust, flexible localization. In support of this task, we construct GeoLoc, the first large-scale, cross-modal, and multi-view aligned dataset comprising over 50,000 pairs of drone, street-view panorama, and satellite images as well as their textual descriptions, collected from 36 countries, ensuring both geographic and semantic alignment. We performed broad evaluations across multiple tasks. Experiments confirm that GeoLoc pre-training markedly improves geo-location accuracy for GeoBridge while promoting cross-domain generalization and cross-modal knowledge transfer. Code, dataset, and pretrained models will be released at this https URL.
>
---
#### [replaced 007] Lite Any Stereo: Efficient Zero-Shot Stereo Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16555](https://arxiv.org/pdf/2511.16555)**

> **作者:** Junpeng Jing; Weixun Luo; Ye Mao; Krystian Mikolajczyk
>
> **备注:** CVPR 2026
>
> **摘要:** Recent advances in stereo matching have focused on accuracy, often at the cost of significantly increased model size. Traditionally, the community has regarded efficient models as incapable of zero-shot ability due to their limited capacity. In this paper, we introduce Lite Any Stereo, a stereo depth estimation framework that achieves strong zero-shot generalization while remaining highly efficient. To this end, we design a compact yet expressive backbone to ensure scalability, along with a carefully crafted hybrid cost aggregation module. We further propose a three-stage training strategy on million-scale data to effectively bridge the sim-to-real gap. Together, these components demonstrate that an ultra-light model can deliver strong generalization, ranking 1st across four widely used real-world benchmarks. Remarkably, our model attains accuracy comparable to or exceeding state-of-the-art non-prior-based accurate methods while requiring less than 1% computational cost, setting a new standard for efficient stereo matching.
>
---
#### [replaced 008] Learning Sewing Patterns via Latent Flow Matching of Implicit Fields
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.17740](https://arxiv.org/pdf/2601.17740)**

> **作者:** Cong Cao; Ren Li; Corentin Dumery; Hao Li
>
> **备注:** SIGGRAPH 2026
>
> **摘要:** Sewing patterns define the structural foundation of garments and are essential for applications such as fashion design, fabrication, and physical simulation. Despite progress in automated pattern generation, accurately modeling sewing patterns remains difficult due to the broad variability in panel geometry and seam arrangements. In this work, we introduce a sewing pattern modeling method based on an implicit representation. We represent each panel using a signed distance field that defines its boundary and an unsigned distance field that identifies seam endpoints, and encode these fields into a continuous latent space that enables differentiable meshing. A latent flow matching model learns distributions over panel combinations in this representation, and a stitching prediction module recovers seam relations from extracted edge segments. This formulation allows accurate modeling and generation of sewing patterns with complex structures. We further show that it can be used to estimate sewing patterns from images with improved accuracy relative to existing approaches, and supports applications such as pattern completion and refitting, providing a practical tool for digital fashion design.
>
---
#### [replaced 009] Cyclic 2.5D Perceptual Loss for Cross-Modal 3D Medical Image Synthesis: T1w MRI to Tau PET
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2406.12632](https://arxiv.org/pdf/2406.12632)**

> **作者:** Junho Moon; Symac Kim; Haejun Chung; Ikbeom Jang
>
> **备注:** Published in Human Brain Mapping, available at this https URL
>
> **摘要:** Positron emission tomography (PET) provides molecular biomarkers for Alzheimer's disease and related dementias (ADRD) and is increasingly used for diagnosis, staging, and clinical trial enrichment. However, its use is limited by cost, regulatory restrictions, and the invasiveness of radiotracer injection. Although current frameworks emphasize multimodal biomarker assessment, including the amyloid/tau/neurodegeneration (A/T/N) scheme, these barriers constrain access to PET imaging. Cross-modal image synthesis may help address this gap by reconstructing unavailable modalities from routine scans. Because PET is clinically valuable for regional uptake patterns rather than exact voxel-wise intensities, perceptual losses that capture higher-level semantic features are well suited to PET synthesis. Existing 2D, 3D, and 2.5D perceptual losses for 3D synthesis each have limitations, including restricted volumetric context, scarcity of pretrained 3D models, and difficulty balancing optimization across anatomical planes. In this study, we synthesize tau PET from structural MRI by generating 3D pseudo-[18F]flortaucipir standardized uptake value ratio (SUVR) maps from 3D T1-weighted MR images. We propose a cyclic 2.5D perceptual loss that alternates optimization across axial, coronal, and sagittal planes during training to improve volumetric consistency. We also standardize PET SUVRs by scanner manufacturer, reducing inter-manufacturer variability and better preserving high-uptake regions. Using cohorts spanning the ADRD spectrum from the ADNI and the SCAN cohort, we show that the method generalizes across U-Net, UNETR, SwinUNETR, CycleGAN, and Pix2Pix, with strong performance. Notably, it improves agreement between synthesized SUVRs and measured PET in brain regions relevant to Alzheimer-type tau pathology. Code is publicly available at this https URL.
>
---
#### [replaced 010] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.03723](https://arxiv.org/pdf/2604.03723)**

> **作者:** Guiyu Zhang; Yabo Chen; Xunzhi Xiang; Junchao Huang; Zhongyu Wang; Li Jiang
>
> **备注:** CVPR 2026
>
> **摘要:** Controlling both camera motion and object dynamics is essential for coherent and expressive video generation, yet current methods typically handle only one motion type or rely on ambiguous 2D cues that entangle camera-induced parallax with true object movement. We present SymphoMotion, a unified motion-control framework that jointly governs camera trajectories and object dynamics within a single model. SymphoMotion features a Camera Trajectory Control mechanism that integrates explicit camera paths with geometry-aware cues to ensure stable, structurally consistent viewpoint transitions, and an Object Dynamics Control mechanism that combines 2D visual guidance with 3D trajectory embeddings to enable depth-aware, spatially coherent object manipulation. To support large-scale training and evaluation, we further construct RealCOD-25K, a comprehensive real-world dataset containing paired camera poses and object-level 3D trajectories across diverse indoor and outdoor scenes, addressing a key data gap in unified motion control. Extensive experiments and user studies show that SymphoMotion significantly outperforms existing methods in visual fidelity, camera controllability, and object-motion accuracy, establishing a new benchmark for unified motion control in video generation. Codes and data are publicly available at this https URL.
>
---
#### [replaced 011] Democratising Pathology Co-Pilots: An Open Pipeline and Dataset for Whole-Slide Vision-Language Modelling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17326](https://arxiv.org/pdf/2512.17326)**

> **作者:** Sander Moonemans; Sebastiaan Ram; Frédérique Meeuwsen; Carlijn Lems; Jeroen van der Laak; Geert Litjens; Francesco Ciompi
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Vision-language models (VLMs) have the potential to become co-pilots for pathologists. However, most VLMs either focus on small regions of interest within whole-slide images, provide only static slide-level outputs, or rely on data that is not publicly available, limiting reproducibility. Furthermore, training data containing WSIs paired with detailed clinical reports is scarce, restricting progress toward transparent and generalisable VLMs. We address these limitations with three main contributions. First, we introduce Polysome, a standardised tool for synthetic instruction generation. Second, we apply Polysome to the public HISTAI dataset, generating HISTAI-Instruct, a large whole-slide instruction tuning dataset spanning 24,259 slides and over 1.1 million instruction-response pairs. Finally, we use HISTAI-Instruct to train ANTONI-{\alpha}, a VLM capable of visual-question answering (VQA). We show that ANTONI-{\alpha} outperforms MedGemma on WSI-level VQA tasks of tissue identification, neoplasm detection, and differential diagnosis. We also compare the performance of multiple incarnations of ANTONI-{\alpha} trained with different amounts of data. All methods, data, and code are publicly available.
>
---
#### [replaced 012] Visual Self-Fulfilling Alignment: Shaping Safety-Oriented Personas via Threat-Related Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.08486](https://arxiv.org/pdf/2603.08486)**

> **作者:** Qishun Yang; Shu Yang; Lijie Hu; Di Wang
>
> **摘要:** Multimodal large language models (MLLMs) face safety misalignment, where visual inputs enable harmful outputs. To address this, existing methods require explicit safety labels or contrastive data; yet, threat-related concepts are concrete and visually depictable, while safety concepts, like helpfulness, are abstract and lack visual referents. Inspired by the Self-Fulfilling mechanism underlying emergent misalignment, we propose Visual Self-Fulfilling Alignment (VSFA). VSFA fine-tunes vision-language models (VLMs) on neutral VQA tasks constructed around threat-related images, without any safety labels. Through repeated exposure to threat-related visual content, models internalize the implicit semantics of vigilance and caution, shaping safety-oriented personas. Experiments across multiple VLMs and safety benchmarks demonstrate that VSFA reduces the attack success rate, improves response quality, and mitigates over-refusal while preserving general capabilities. Our work extends the self-fulfilling mechanism from text to visual modalities, offering a label-free approach to VLMs alignment.
>
---
#### [replaced 013] Abstract 3D Perception for Spatial Intelligence in Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10946](https://arxiv.org/pdf/2511.10946)**

> **作者:** Yifan Liu; Fangneng Zhan; Kaichen Zhou; Yilun Du; Paul Pu Liang; Hanspeter Pfister
>
> **摘要:** Vision-language models (VLMs) struggle with 3D-related tasks such as spatial cognition and physical understanding, which are crucial for real-world applications like robotics and embodied agents. We attribute this to a modality gap between the 3D tasks and the 2D training of VLM, which led to inefficient retrieval of 3D information from 2D input. To bridge this gap, we introduce SandboxVLM, a simple yet effective framework that leverages abstract bounding boxes to encode geometric structure and physical kinematics for VLM. Specifically, we design a 3D Sandbox reconstruction and perception pipeline comprising four stages: generating multi-view priors with abstract control, proxy elevation, multi-view voting and clustering, and 3D-aware reasoning. Evaluated in zero-shot settings across multiple benchmarks and VLM backbones, our approach consistently improves spatial intelligence, achieving an 8.3\% gain on SAT Real compared with baseline methods for instance. These results demonstrate that equipping VLMs with a 3D abstraction substantially enhances their 3D reasoning ability without additional training, suggesting new possibilities for general-purpose embodied intelligence.
>
---
#### [replaced 014] Aerial Vision-Language Navigation with a Unified Framework for Spatial, Temporal and Embodied Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.08639](https://arxiv.org/pdf/2512.08639)**

> **作者:** Huilin Xu; Zhuoyang Liu; Yixiang Luomei; Feng Xu
>
> **备注:** Under Review, 16 pages, 12 figures. Our code is publicly available at this https URL
>
> **摘要:** Aerial Vision-and-Language Navigation (VLN) aims to enable unmanned aerial vehicles (UAVs) to interpret natural language instructions and navigate complex urban environments using onboard visual observation. This task holds promise for real-world applications such as low-altitude inspection, search-and-rescue, and autonomous aerial delivery. Existing methods often rely on panoramic images, depth inputs, or odometry to support spatial reasoning and action planning. These requirements increase system cost and integration complexity, thus hindering practical deployment for lightweight UAVs. We present a unified aerial VLN framework that operates solely on egocentric monocular RGB observations and natural language instructions. The model formulates navigation as a next-token prediction problem, jointly optimizing spatial perception, trajectory reasoning, and action prediction through prompt-guided multi-task learning. Moreover, we propose a keyframe selection strategy to reduce visual redundancy by retaining semantically informative frames, along with an action merging and label reweighting mechanism that mitigates long-tailed supervision imbalance and facilitates stable multi-task co-training. Extensive experiments on the AerialVLN and OpenFly benchmark validate the effectiveness of our method. Under the challenging monocular RGB-only setting, our model achieves strong results across both seen and unseen environments. It significantly outperforms existing RGB-only baselines and narrows the performance gap with state-of-the-art panoramic RGB-D counterparts. Comprehensive ablation studies further demonstrate the contribution of our task design and architectural choices. Our code is publicly available at this https URL.
>
---
#### [replaced 015] Person Re-Identification via Generalized Class Prototypes
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2510.17043](https://arxiv.org/pdf/2510.17043)**

> **作者:** Md Ahmed Al Muzaddid; William J. Beksi
>
> **备注:** To be published in the 2026 International Conference on Pattern Recognition (ICPR)
>
> **摘要:** Advanced feature extraction methods have significantly contributed to enhancing the task of person re-identification. In addition, modifications to objective functions have been developed to further improve performance. Nonetheless, selecting better class representatives is an underexplored area of research that can also lead to advancements in re-identification performance. Although past works have experimented with using the centroid of a gallery image class during training, only a few have investigated alternative representations during the retrieval stage. In this paper, we demonstrate that these prior techniques yield suboptimal results in terms of re-identification metrics. To address the re-identification problem, we propose a generalized selection method that involves choosing representations that are not limited to class centroids. Our approach strikes a balance between accuracy and mean average precision, leading to improvements beyond the state of the art. For example, the actual number of representations per class can be adjusted to meet specific application requirements. We apply our methodology on top of multiple re-identification embeddings, and in all cases it substantially improves upon contemporary results.
>
---
#### [replaced 016] Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究AI理解幽默矛盾的能力，属于自然语言处理任务。针对AI在解析漫画中矛盾叙事上的不足，提出YesBut基准，评估模型在不同难度任务中的表现。**

- **链接: [https://arxiv.org/pdf/2405.19088](https://arxiv.org/pdf/2405.19088)**

> **作者:** Zhe Hu; Tuo Liang; Jing Li; Yiren Lu; Yunlai Zhou; Yiran Qiao; Jing Ma; Yu Yin
>
> **备注:** NeurIPS 2024 (Oral)
>
> **摘要:** Recent advancements in large multimodal language models have demonstrated remarkable proficiency across a wide range of tasks. Yet, these models still struggle with understanding the nuances of human humor through juxtaposition, particularly when it involves nonlinear narratives that underpin many jokes and humor cues. This paper investigates this challenge by focusing on comics with contradictory narratives, where each comic consists of two panels that create a humorous contradiction. We introduce the YesBut benchmark, which comprises tasks of varying difficulty aimed at assessing AI's capabilities in recognizing and interpreting these comics, ranging from literal content comprehension to deep narrative reasoning. Through extensive experimentation and analysis of recent commercial or open-sourced large (vision) language models, we assess their capability to comprehend the complex interplay of the narrative humor inherent in these comics. Our results show that even state-of-the-art models still lag behind human performance on this task. Our findings offer insights into the current limitations and potential improvements for AI in understanding human creative expressions.
>
---
#### [replaced 017] Detecting Diffusion-generated Images via Dynamic Assembly Forests
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.09106](https://arxiv.org/pdf/2604.09106)**

> **作者:** Mengxin Fu; Yuezun Li
>
> **摘要:** Diffusion models are known for generating high-quality images, causing serious security concerns. To combat this, most efforts rely on deep neural networks (e.g., CNNs and Transformers), while largely overlooking the potential of traditional machine learning models. In this paper, we freshly investigate such alternatives and proposes a novel Dynamic Assembly Forest model (DAF) to detect diffusion-generated images. Built upon the deep forest paradigm, DAF addresses the inherent limitations in feature learning and scalable training, making it an effective diffusion-generated image detector. Compared to existing DNN-based methods, DAF has significantly fewer parameters, much lower computational cost, and can be deployed without GPUs, while achieving competitive performance under standard evaluation protocols. These results highlight the strong potential of the proposed method as a practical substitute for heavyweight DNN models in resource-constrained scenarios. Our code and models are available at this https URL.
>
---
#### [replaced 018] RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows
- **分类: cs.MA; cs.CL; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决胸片解读中推理不透明、多模态信息融合不足及工具不一致的问题。提出RadAgents框架，实现临床流程的模块化推理与验证。**

- **链接: [https://arxiv.org/pdf/2509.20490](https://arxiv.org/pdf/2509.20490)**

> **作者:** Kai Zhang; Corey D Barrett; Jangwon Kim; Lichao Sun; Tara Taghavi; Krishnaram Kenthapadi
>
> **备注:** MIDL 2026
>
> **摘要:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework that couples clinical priors with task-aware multimodal reasoning and encodes a radiologist-style workflow into a modular, auditable pipeline. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.
>
---
#### [replaced 019] Tokenizing Semantic Segmentation with Run Length Encoding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21627](https://arxiv.org/pdf/2602.21627)**

> **作者:** Abhineet Singh; Justin Rozeboom; Nilanjan Ray
>
> **备注:** Code and models available at: this https URL
>
> **摘要:** This paper presents a new unified approach to semantic segmentation in both images and videos by using language modeling to output the masks as sequences of discrete tokens. We use run length encoding (RLE) to discretize the segmentation masks, and adapt the Pix2Seq framework to learn autoregressive models to output these tokens. We propose novel tokenization strategies to compress the lengths of the token sequences to make it practicable to extend this approach to videos. We also show how instance information can be incorporated into the tokenization process to perform panoptic segmentation. We evaluate our models on two domain-specific datasets to demonstrate their competitiveness with the state of the art in certain scenarios, in spite of being severely bottlenecked by our limited computational resources. We supplement these analyses by proposing several promising approaches to foster future competitiveness in general-purpose applications, and facilitate this by making our code and models publicly available.
>
---
#### [replaced 020] Getting the Numbers Right$\unicode{x2014}$Modelling Multi-Class Object Counting in Dense and Varied Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.02213](https://arxiv.org/pdf/2510.02213)**

> **作者:** Villanelle O'Reilly; Jonathan Cox; Georgios Leontidis; Marc Hanheide; Petra Bosilj; James M. Brown
>
> **备注:** 8 pages, 4 figures, 5 tables
>
> **摘要:** Density map estimation enables accurate object counting in heavily occluded, and densely packed scenes where detection-based counting fails. In multi-class density estimation, class awareness can be introduced by modelling classes non-exclusively, better reflecting crowded and visually ambiguous contexts. However, existing multi-class density estimators often degrade in less-dense scenes, while state-of-the-art detectors still struggle in the most congested settings. To bridge this gap, we propose the first vision-transformer-based approach to multi-class density estimation. Our model combines a Twins-SVT pyramid vision transformer backbone with a multiscale CNN decoder that leverages hierarchical features for robust counting across a wide range of densities. Further to that, the method adds an auxiliary segmentation task with the Category Focus Module to suppress inter-category interference at training time. The module improves the density estimation head without the need for constraining assumptions added by the application of the auxiliary task at inference time, as required in previous methods. Training and evaluation on the VisDrone and iSAID benchmarks demonstrates a leap in performance versus the previous state-of-the-art multi-class density estimation methods, attaining a 33%, 43%, and 64% reduction to MAE in testing evaluation. The method outperforms YOLO11 in less busy scenes, exceeding it by an order of magnitude in the most crowded testing samples. Code, and trained weights available at this https URL.
>
---
#### [replaced 021] From Instruction to Event: Sound-Triggered Mobile Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于移动操作任务，旨在解决传统指令驱动方式限制自主性的问题。通过声音触发机制，使智能体主动感知和响应声源，提升环境适应能力。**

- **链接: [https://arxiv.org/pdf/2601.21667](https://arxiv.org/pdf/2601.21667)**

> **作者:** Hao Ju; Shaofei Huang; Hongyu Li; Zihan Ding; Si Liu; Meng Wang; Zhedong Zheng
>
> **摘要:** Current mobile manipulation research predominantly follows an instruction-driven paradigm, where agents rely on predefined textual commands to execute tasks. However, this setting confines agents to a passive role, limiting their autonomy and ability to react to dynamic environmental events. To address these limitations, we introduce sound-triggered mobile manipulation, where agents must actively perceive and interact with sound-emitting objects without explicit action instructions. To support these tasks, we develop Habitat-Echo, a data platform that integrates acoustic rendering with physical interaction. We further propose a baseline comprising a high-level task planner and low-level policy models to complete these tasks. Extensive experiments show that the proposed baseline empowers agents to actively detect and respond to auditory events, eliminating the need for case-by-case instructions. Notably, in the challenging dual-source scenario, the agent successfully isolates the primary source from overlapping acoustic interference to execute the first interaction, and subsequently proceeds to manipulate the secondary object, verifying the robustness of the baseline.
>
---
#### [replaced 022] Heavy-Tailed Class-Conditional Priors for Long-Tailed Generative Modeling
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2509.02154](https://arxiv.org/pdf/2509.02154)**

> **作者:** Aymene Mohammed Bouayed; Samuel Deslauriers-Gauthier; Adrian Iaccovelli; David Naccache
>
> **摘要:** Variational Autoencoders (VAEs) with global priors trained under an imbalanced empirical class distribution can lead to underrepresentation of tail classes in the latent space. While $t^3$VAE improves robustness via heavy-tailed Student's $t$-distribution priors, its single global prior still allocates mass proportionally to class frequency. We address this latent geometric bias by introducing C-$t^3$VAE, which assigns a per-class Student's $t$ joint prior over latent and output variables. This design promotes uniform prior mass across class-conditioned components. To optimize our model we derive a closed-form objective from the $\gamma$-power divergence, and we introduce an equal-weight latent mixture for class-balanced generation. On SVHN-LT, CIFAR100-LT, and CelebA datasets, C-$t^3$VAE consistently attains lower FID scores than $t^3$VAE and Gaussian-based VAE baselines under severe class imbalance while remaining competitive in balanced or mildly imbalanced settings. In per-class F1 evaluations, our model outperforms the conditional Gaussian VAE across highly imbalanced settings. Moreover, we identify the mild imbalance threshold $\rho < 5$, for which Gaussian-based models remain competitive. However, for $\rho \geq 5$ our approach yields improved class-balanced generation and mode coverage.
>
---
#### [replaced 023] UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23332](https://arxiv.org/pdf/2511.23332)**

> **作者:** Shuo Ni; Di Wang; He Chen; Haonan Guo; Ning Zhang; Jing Zhang
>
> **备注:** Datasets and source code were released at this https URL
>
> **摘要:** Instruction-driven segmentation in remote sensing generates masks from guidance, offering great potential for accessible and generalizable applications. However, existing methods suffer from fragmented task formulations and limited instruction data, hindering effective understanding and generalization. To address these issues, we introduce GeoSeg-1M, the first million-scale dataset for remote sensing instruction-driven segmentation, constructed via an automatic mask filtering and instruction generation pipeline that synthesizes referring, interactive, and reasoning segmentation instructions from multiple public datasets. GeoSeg-1M contains 590K images, 117 categories, and 1.1M image-mask-instruction triplets. Building upon this foundation, we further curate GeoSeg-Bench, a challenging benchmark designed to evaluate contextual understanding and reasoning capabilities across diverse instruction-driven tasks and complex geospatial scenes. Furthermore, we present UniGeoSeg, a unified framework that serves as a strong baseline, incorporating task-aware text enhancement, latent knowledge memory, and a progressive training strategy to facilitate multi-task learning. Extensive experiments demonstrate the state-of-the-art performance of UniGeoSeg across GeoSeg-Bench and diverse public benchmarks, while exhibiting strong zero-shot generalization. Datasets and source code were released at this https URL.
>
---
#### [replaced 024] RoboTAG: End-to-end Robot Configuration Estimation via Topological Alignment Graph
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboTAG，解决单目RGB图像中机器人位姿估计问题，通过结合2D与3D分支，提升模型性能并减少对标注数据的依赖。**

- **链接: [https://arxiv.org/pdf/2511.07717](https://arxiv.org/pdf/2511.07717)**

> **作者:** Yifan Liu; Fangneng Zhan; Wanhua Li; Haowen Sun; Katerina Fragkiadaki; Hanspeter Pfister
>
> **摘要:** Estimating robot pose from a monocular RGB image is a challenge in robotics and computer vision. Existing methods typically build networks on top of 2D visual backbones and depend heavily on labeled data for training, which is often scarce in real-world scenarios, causing a sim-to-real gap. Moreover, these approaches reduce the 3D-based problem to 2D domain, neglecting the 3D priors. To address these, we propose Robot Topological Alignment Graph (RoboTAG), which incorporates a 3D branch to inject 3D priors while enabling co-evolution of the 2D and 3D representations, alleviating the reliance on labels. Specifically, the RoboTAG consists of a 3D branch and a 2D branch, where nodes represent the states of the camera and robot system, and edges capture the dependencies between these variables or denote alignments between them. Closed loops are then defined in the graph, on which a consistency supervision across branches can be applied. Experimental results demonstrate that our method is effective across robot types, suggesting new possibilities of alleviating the data bottleneck in robotics.
>
---
#### [replaced 025] LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20913](https://arxiv.org/pdf/2602.20913)**

> **作者:** Jihao Qiu; Lingxi Xie; Xinyue Huo; Qi Tian; Qixiang Ye
>
> **备注:** 17 pages, 9 figures, 8 tables, accepted to CVPR 2026
>
> **摘要:** This paper addresses the critical and underexplored challenge of long video understanding with low computational budgets. We propose LongVideo-R1, an active, reasoning-equipped multimodal large language model (MLLM) agent designed for efficient video context navigation, avoiding the redundancy of exhaustive search. At the core of LongVideo-R1 lies a reasoning module that leverages high-level visual cues to infer the most informative video clip for subsequent processing. During inference, the agent initiates traversal from top-level visual summaries and iteratively refines its focus, immediately halting the exploration process upon acquiring sufficient knowledge to answer the query. To facilitate training, we first extract hierarchical video captions from CGBench, a video corpus with grounding annotations, and guide GPT-5 to generate 33K high-quality chain-of-thought-with-tool trajectories. The LongVideo-R1 agent is fine-tuned upon the Qwen-3-8B model through a two-stage paradigm: supervised fine-tuning (SFT) followed by reinforcement learning (RL), where RL employs a specifically designed reward function to maximize selective and efficient clip navigation. Experiments on multiple long video benchmarks validate the effectiveness of name, which enjoys superior tradeoff between QA accuracy and efficiency. All curated data and source code are provided in the supplementary material and will be made publicly available. Code and data are available at: this https URL
>
---
#### [replaced 026] Visual Sparse Steering (VS2): Unsupervised Adaptation for Image Classification using Sparsity-Guided Steering Vectors
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.01247](https://arxiv.org/pdf/2506.01247)**

> **作者:** Gerasimos Chatzoudis; Zhuowei Li; Gemma E. Moran; Hao Wang; Dimitris N. Metaxas
>
> **摘要:** Steering vision foundation models at test time, without updating foundation-model weights or using labeled target data, is a desirable yet challenging goal. We present Visual Sparse Steering (VS2), a lightweight, label-free adaptation method that constructs a steering vector from sparse features extracted by a Sparse Autoencoder (SAE) trained on unlabeled in-domain training-split activations of the vision encoder. VS2 offers three key advantages over existing test-time adaptation methods: (1) a feature-level intervention space in sparse SAE representations; (2) efficiency, requiring only a forward pass with no test-time optimization or backpropagation; and (3) a reliability diagnostic based on SAE reconstruction loss that can skip steering when reconstruction is poor, enabling safe fallback to the baseline, a capability not standard in conventional steering vectors and test-time adaptation methods. Across CIFAR-100, CUB-200, and Tiny-ImageNet and two CLIP backbones (ViT-B/32, ViT-B/16), VS2 improves zero-shot top-1 accuracy by 3.45-4.12\%, 0.93-1.08\%, and 1.50-1.84\%, respectively, while remaining forward-only and adding minimal compute overhead. A retrieval-based upper-bound analysis suggests substantial headroom if task-relevant sparse features can be selected reliably, motivating future work on selective feature amplification for interpretable, efficient test-time steering.
>
---
#### [replaced 027] LPM 1.0: Video-based Character Performance Model
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2604.07823](https://arxiv.org/pdf/2604.07823)**

> **作者:** Ailing Zeng; Casper Yang; Chauncey Ge; Eddie Zhang; Garvey Xu; Gavin Lin; Gilbert Gu; Jeremy Pi; Leo Li; Mingyi Shi; Shawn Wang; Sheng Bi; Steven Tang; Thorn Hang; Tobey Guo; Vincent Li; Xin Tong; Yikang Li; Yuchen Sun; Yue Zhao; Yuhan Lu; Yuwei Li; Zane Zhang; Zeshi Yang; Zi Ye
>
> **备注:** 43 pages, 15 figures, 2 tables. Project page: this https URL
>
> **摘要:** Performance, the externalization of intent, emotion, and personality through visual, vocal, and temporal behavior, is what makes a character alive. Learning such performance from video is a promising alternative to traditional 3D pipelines. However, existing video models struggle to jointly achieve high expressiveness, real-time inference, and long-horizon identity stability, a tension we call the performance trilemma. Conversation is the most comprehensive performance scenario, as characters simultaneously speak, listen, react, and emote while maintaining identity over time. To address this, we present LPM 1.0 (Large Performance Model), focusing on single-person full-duplex audio-visual conversational performance. Concretely, we build a multimodal human-centric dataset through strict filtering, speaking-listening audio-video pairing, performance understanding, and identity-aware multi-reference extraction; train a 17B-parameter Diffusion Transformer (Base LPM) for highly controllable, identity-consistent performance through multimodal conditioning; and distill it into a causal streaming generator (Online LPM) for low-latency, infinite-length interaction. At inference, given a character image with identity-aware references, LPM 1.0 generates listening videos from user audio and speaking videos from synthesized audio, with text prompts for motion control, all at real-time speed with identity-stable, infinite-length generation. LPM 1.0 thus serves as a visual engine for conversational agents, live streaming characters, and game NPCs. To systematically evaluate this setting, we propose LPM-Bench, the first benchmark for interactive character performance. LPM 1.0 achieves state-of-the-art results across all evaluated dimensions while maintaining real-time inference.
>
---
#### [replaced 028] DBMF: A Dual-Branch Multimodal Framework for Out-of-Distribution Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.08261](https://arxiv.org/pdf/2604.08261)**

> **作者:** Jiangbei Yue; Darren Treanor; Venkataraman Subramanian; Sharib Ali
>
> **摘要:** The complex and dynamic real-world clinical environment demands reliable deep learning (DL) systems. Out-of-distribution (OOD) detection plays a critical role in enhancing the reliability and generalizability of DL models when encountering data that deviate from the training distribution, such as unseen disease cases. However, existing OOD detection methods typically rely either on a single visual modality or solely on image-text matching, failing to fully leverage multimodal information. To overcome the challenge, we propose a novel dual-branch multimodal framework by introducing a text-image branch and a vision branch. Our framework fully exploits multimodal representations to identify OOD samples through these two complementary branches. After training, we compute scores from the text-image branch ($S_t$) and vision branch ($S_v$), and integrate them to obtain the final OOD score $S$ that is compared with a threshold for OOD detection. Comprehensive experiments on publicly available endoscopic image datasets demonstrate that our proposed framework is robust across diverse backbones and improves state-of-the-art performance in OOD detection by up to 24.84%
>
---
#### [replaced 029] Grid2Matrix: Revealing Digital Agnosia in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.09687](https://arxiv.org/pdf/2604.09687)**

> **作者:** Yunkai Zhang; Linda Li; Yingxin Cui; Xiyuan Ruan; Zeyu Zheng; Kezhen Chen; Yi Zhang; Diji Yang
>
> **摘要:** Vision-Language Models (VLMs) excel on many multimodal reasoning benchmarks, but these evaluations often do not require an exhaustive readout of the image and can therefore obscure failures in faithfully capturing all visual details. We introduce Grid2Matrix (G2M), a controlled benchmark in which a model is shown a color grid and a color-to-number mapping, and must output the corresponding matrix. By varying grid size and the number of colors, G2M provides a simple way to increase visual complexity while minimizing semantic confounds. We find that VLMs exhibit a sharp early collapse in zero-shot end-to-end evaluation, failing on surprisingly small grids rather than degrading gradually as the task becomes denser. We probe the visual encoders of VLMs from two representative families and find that they preserve substantially more of the grid information than the corresponding end-to-end outputs. This suggests that the failure is not explained by visual encoding alone, but also reflects a gap between what remains recoverable from visual features and what is ultimately expressed in language. We term this gap \textit{Digital Agnosia}. Further analyses show that these errors are highly structured and depend strongly on how grid cells overlap with visual patch boundaries. We also find that common strategies such as model scaling and multimodal alignment do not fully eliminate this failure mode. We expect G2M to serve as a useful testbed for understanding where and how VLMs lose fine visual details, and for evaluating tasks where missing even small visual details can matter, such as tables, charts, forms, and GUIs.
>
---
#### [replaced 030] Hydra: Unifying Document Retrieval and Generation in a Single Vision-Language Model
- **分类: cs.CV; cs.AI; cs.IR**

- **链接: [https://arxiv.org/pdf/2603.28554](https://arxiv.org/pdf/2603.28554)**

> **作者:** Athos Georgiou
>
> **备注:** 18 pages, 2 figures, 7 tables, 1 algorithm. v2: lm_head alias via Qwen3.5 weight-tying cuts peak GPU memory 41% -> 48% (10.5 -> 9.2 GB); bitwise-identical outputs verified over 50+ greedy samples, 10 decodes at 1024 tokens, 50 mode-switch round-trips. Code: this http URL ; HF models under this http URL
>
> **摘要:** Visual document understanding typically requires separate retrieval and generation models, doubling memory and system complexity. We present Hydra, a dual-head approach that provides both ColBERT-style late-interaction retrieval and autoregressive generation from a single vision-language model (VLM). A single LoRA adapter, trained only for retrieval, is toggled at inference: enabling it produces multi-vector embeddings; disabling it recovers the base model's generation quality -- byte-identical outputs in 100% of 10,500 greedy and stochastic samples, with max delta-ANLS = 0.0044 across 15,301 samples on four VQA benchmarks (three informative; ChartQA is near-zero for both models under greedy decoding) when compared against an independent base-model pipeline. We identify three engineering requirements (attention-mode restoration, lm_head preservation, KV-cache-aware decoding) whose omission silently breaks generation despite correct weight recovery. On ViDoRe V1, Hydra (4B) is within 1 percentage point of a controlled single-head baseline in a single training run, with higher aggregate scores on V2 and V3 that are concentrated on a subset of tasks; multi-seed experiments are needed to confirm these trends. The single-model design cuts peak GPU memory from 17.9 GB (two-model baseline) to 9.2 GB -- a 48% reduction, though adapter switching introduces throughput overhead under concurrent serving loads. An ablation shows that GritLM-style joint training provides no benefit within the LoRA-based (r=16) training regime. A proof-of-concept extension to Qwen2.5-Omni-3B demonstrates that the mechanism generalizes to audio retrieval and video embedding, with speech generation.
>
---
#### [replaced 031] Script-a-Video: Deep Structured Audio-visual Captions via Factorized Streams and Relational Grounding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.11244](https://arxiv.org/pdf/2604.11244)**

> **作者:** Tencent Hunyuan Team
>
> **摘要:** Advances in Multimodal Large Language Models (MLLMs) are transforming video captioning from a descriptive endpoint into a semantic interface for both video understanding and generation. However, the dominant paradigm still casts videos as monolithic narrative paragraphs that entangle visual, auditory, and identity information. This dense coupling not only compromises representational fidelity but also limits scalability, since even local edits can trigger global rewrites. To address this structural bottleneck, we propose Multi-Stream Scene Script (MTSS), a novel paradigm that replaces monolithic text with factorized and explicitly grounded scene descriptions. MTSS is built on two core principles: Stream Factorization, which decouples a video into complementary streams (Reference, Shot, Event, and Global), and Relational Grounding, which reconnects these isolated streams through explicit identity and temporal links to maintain holistic video consistency. Extensive experiments demonstrate that MTSS consistently enhances video understanding across various models, achieving an average reduction of 25% in the total error rate on Video-SALMONN-2 and an average performance gain of 67% on the Daily-Omni reasoning benchmark. It also narrows the performance gap between smaller and larger MLLMs, indicating a substantially more learnable caption interface. Finally, even without architectural adaptation, replacing monolithic prompts with MTSS in multi-shot video generation yields substantial human-rated improvements: a 45% boost in cross-shot identity consistency, a 56% boost in audio-visual alignment, and a 71% boost in temporal controllability.
>
---
#### [replaced 032] Simplicity Prevails: The Emergence of Generalizable AIGI Detection in Visual Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01738](https://arxiv.org/pdf/2602.01738)**

> **作者:** Yue Zhou; Xinan He; Kaiqing Lin; Bing Fan; Feng Ding; Bin Li
>
> **摘要:** While specialized detectors for AI-Generated Images (AIGI) achieve near-perfect accuracy on curated benchmarks, they suffer from a dramatic performance collapse in realistic, in-the-wild scenarios. In this work, we demonstrate that simplicity prevails over complex architectural designs. A simple linear classifier trained on the frozen features of modern Vision Foundation Models , including Perception Encoder, MetaCLIP 2, and DINOv3, establishes a new state-of-the-art. Through a comprehensive evaluation spanning traditional benchmarks, unseen generators, and challenging in-the-wild distributions, we show that this baseline not only matches specialized detectors on standard benchmarks but also decisively outperforms them on in-the-wild datasets, boosting accuracy by striking margins of over 30\%. We posit that this superior capability is an emergent property driven by the massive scale of pre-training data containing synthetic content. We trace the source of this capability to two distinct manifestations of data exposure: Vision-Language Models internalize an explicit semantic concept of forgery, while Self-Supervised Learning models implicitly acquire discriminative forensic features from the pretraining data. However, we also reveal persistent limitations: these models suffer from performance degradation under recapture and transmission, remain blind to VAE reconstruction and localized editing. We conclude by advocating for a paradigm shift in AI forensics, moving from overfitting on static benchmarks to harnessing the evolving world knowledge of foundation models for real-world reliability.
>
---
#### [replaced 033] Multi-Dimensional Knowledge Profiling with Large-Scale Literature Database and Hierarchical Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.15170](https://arxiv.org/pdf/2601.15170)**

> **作者:** Zhucun Xue; Jiangning Zhang; Juntao Jiang; Jinzhuo Liu; Haoyang He; Teng Hu; Xiaobin Hu; Yong Liu; Shuicheng Yan
>
> **摘要:** The rapid expansion of research across machine learning, vision, and language has produced a volume of publications that is increasingly difficult to synthesize. Traditional bibliometric tools rely mainly on metadata and offer limited visibility into the semantic content of papers, making it hard to track how research themes evolve over time or how different areas influence one this http URL obtain a clearer picture of recent developments, we compile a unified corpus of more than 100,000 papers from 22 major conferences between 2020 and 2025 and construct a multidimensional profiling pipeline to organize and analyze their textual content. By combining topic clustering, LLM-assisted parsing, and structured retrieval, we derive a comprehensive representation of research activity that supports the study of topic lifecycles, methodological transitions, dataset and model usage patterns, and institutional research this http URL analysis highlights several notable shifts, including the growth of safety, multimodal reasoning, and agent-oriented studies, as well as the gradual stabilization of areas such as neural machine translation and graph-based methods. These findings provide an evidence-based view of how AI research is evolving and offer a resource for understanding broader trends and identifying emerging directions.
>
---
#### [replaced 034] Geometry-Aware Cross Modal Alignment for Light Field-LiDAR Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.06687](https://arxiv.org/pdf/2510.06687)**

> **作者:** Jie Luo; Yuxuan Jiang; Xin Jin; Mingyu Liu; Yihui Fan
>
> **摘要:** Semantic segmentation serves as a cornerstone of scene understanding in autonomous driving but continues to face significant challenges under complex conditions such as occlusion. Light field and LiDAR modalities provide complementary visual and spatial cues that are beneficial for robust perception; however, their effective integration is hindered by limited viewpoint diversity and inherent modality discrepancies. To address these challenges, the first multimodal semantic segmentation dataset integrating light field data and point cloud data is proposed. Based on this dataset, we proposed a multi-modal light field point-cloud fusion segmentation network(Mlpfseg), incorporating feature completion and depth perception to segment both camera images and LiDAR point clouds simultaneously. The feature completion module addresses the density mismatch between point clouds and image pixels by performing differential reconstruction of point-cloud feature maps, enhancing the fusion of these modalities. The depth perception module improves the segmentation of occluded objects by reinforcing attention scores for better occlusion awareness. Our method outperforms image-only segmentation by 1.71 Mean Intersection over Union(mIoU) and point cloud-only segmentation by 2.38 mIoU, demonstrating its effectiveness.
>
---
#### [replaced 035] SyncFix: Fixing 3D Reconstructions via Multi-View Synchronization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.11797](https://arxiv.org/pdf/2604.11797)**

> **作者:** Deming Li; Abhay Yadav; Cheng Peng; Rama Chellappa; Anand Bhattad
>
> **备注:** Project website: this https URL
>
> **摘要:** We present SyncFix, a framework that enforces cross-view consistency during the diffusion-based refinement of reconstructed scenes. SyncFix formulates refinement as a joint latent bridge matching problem, synchronizing distorted and clean representations across multiple views to fix the semantic and geometric inconsistencies. This means SyncFix learns a joint conditional over multiple views to enforce consistency throughout the denoising trajectory. Our training is done only on image pairs, but it generalizes naturally to an arbitrary number of views during inference. Moreover, reconstruction quality improves with additional views, with diminishing returns at higher view counts. Qualitative and quantitative results demonstrate that SyncFix consistently generates high-quality reconstructions and surpasses current state-of-the-art baselines, even in the absence of clean reference images. SyncFix achieves even higher fidelity when sparse references are available.
>
---
#### [replaced 036] PianoFlow: Music-Aware Streaming Piano Motion Generation with Bimanual Coordination
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.12856](https://arxiv.org/pdf/2604.12856)**

> **作者:** Xuan Wang; Kai Ruan; Jiayi Han; Kaiyue Zhou; Gaoang Wang
>
> **摘要:** Audio-driven bimanual piano motion generation requires precise modeling of complex musical structures and dynamic cross-hand coordination. However, existing methods often rely on acoustic-only representations lacking symbolic priors, employ inflexible interaction mechanisms, and are limited to computationally expensive short-sequence generation. To address these limitations, we propose PianoFlow, a flow-matching framework for precise and coordinated bimanual piano motion synthesis. Our approach strategically leverages MIDI as a privileged modality during training, distilling these structured musical priors to achieve deep semantic understanding while maintaining audio-only inference. Furthermore, we introduce an asymmetric role-gated interaction module to explicitly capture dynamic cross-hand coordination through role-aware attention and temporal gating. To enable real-time streaming generation for arbitrarily long sequences, we design an autoregressive flow continuation scheme that ensures seamless cross-chunk temporal coherence. Extensive experiments on the PianoMotion10M dataset demonstrate that PianoFlow achieves superior quantitative and qualitative performance, while accelerating inference by over 9\times compared to previous methods.
>
---
#### [replaced 037] Towards Generalizable Robotic Manipulation in Dynamic Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人动态操作任务，旨在解决VLA模型在动态环境中的表现不足问题。通过构建DOMINO数据集和提出PUMA架构，提升模型的时空推理能力。**

- **链接: [https://arxiv.org/pdf/2603.15620](https://arxiv.org/pdf/2603.15620)**

> **作者:** Heng Fang; Shangru Li; Shuhan Wang; Xuanyang Xi; Dingkang Liang; Xiang Bai
>
> **备注:** Project Page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models excel in static manipulation but struggle in dynamic environments with moving targets. This performance gap primarily stems from a scarcity of dynamic manipulation datasets and the reliance of mainstream VLAs on single-frame observations, restricting their spatiotemporal reasoning capabilities. To address this, we introduce DOMINO, a large-scale dataset and benchmark for generalizable dynamic manipulation, featuring 35 tasks with hierarchical complexities, over 110K expert trajectories, and a multi-dimensional evaluation suite. Through comprehensive experiments, we systematically evaluate existing VLAs on dynamic tasks, explore effective training strategies for dynamic awareness, and validate the generalizability of dynamic data. Furthermore, we propose PUMA, a dynamics-aware VLA architecture. By integrating scene-centric historical optical flow and specialized world queries to implicitly forecast object-centric future states, PUMA couples history-aware perception with short-horizon prediction. Results demonstrate that PUMA achieves state-of-the-art performance, yielding a 6.3% absolute improvement in success rate over baselines. Moreover, we show that training on dynamic data fosters robust spatiotemporal representations that transfer to static tasks. All code and data are available at this https URL.
>
---
#### [replaced 038] Spatial Atlas: Compute-Grounded Reasoning for Spatial-Aware Research Agent Benchmarks
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.12102](https://arxiv.org/pdf/2604.12102)**

> **作者:** Arun Sharma
>
> **备注:** 11 pages. Code: this https URL
>
> **摘要:** We introduce compute-grounded reasoning (CGR), a design paradigm for spatial-aware research agents in which every answerable sub-problem is resolved by deterministic computation before a language model is asked to generate. Spatial Atlas instantiates CGR as a single Agent-to-Agent (A2A) server that handles two challenging benchmarks: FieldWorkArena, a multimodal spatial question-answering benchmark spanning factory, warehouse, and retail environments, and MLE-Bench, a suite of 75 Kaggle machine learning competitions requiring end-to-end ML engineering. A structured spatial scene graph engine extracts entities and relations from vision descriptions, computes distances and safety violations deterministically, then feeds computed facts to large language models, thereby avoiding hallucinated spatial reasoning. Entropy-guided action selection maximizes information gain per step and routes queries across a three-tier frontier model stack (OpenAI + Anthropic). A self-healing ML pipeline with strategy-aware code generation, a score-driven iterative refinement loop, and a prompt-based leak audit registry round out the system. We evaluate across both benchmarks and show that CGR yields competitive accuracy while maintaining interpretability through structured intermediate representations and deterministic spatial computations.
>
---
#### [replaced 039] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14234](https://arxiv.org/pdf/2512.14234)**

> **作者:** Juze Zhang; Changan Chen; Xin Chen; Heng Yu; Tiange Xiang; Ali Sartaz Khan; Shrinidhi K. Lakshmikanth; Ehsan Adeli
>
> **备注:** Project page: this https URL. Accepted by CVPR 2026
>
> **摘要:** Human communication is inherently multimodal and social: words, prosody, and body language jointly carry intent. Yet most prior systems model human behavior as a translation task co-speech gesture or text-to-motion that maps a fixed utterance to motion clips-without requiring agentic decision-making about when to move, what to do, or how to adapt across multi-turn dialogue. This leads to brittle timing, weak social grounding, and fragmented stacks where speech, text, and motion are trained or inferred in isolation. We introduce ViBES (Voice in Behavioral Expression and Synchrony), a conversational 3D agent that jointly plans language and movement and executes dialogue-conditioned body actions. Concretely, ViBES is a speech-language-behavior (SLB) model with a mixture-of-modality-experts (MoME) backbone: modality-partitioned transformer experts for speech, facial expression, and body motion. The model processes interleaved multimodal token streams with hard routing by modality (parameters are split per expert), while sharing information through cross-expert attention. By leveraging strong pretrained speech-language models, the agent supports mixed-initiative interaction: users can speak, type, or issue body-action directives mid-conversation, and the system exposes controllable behavior hooks for streaming responses. We further benchmark on multi-turn conversation with automatic metrics of dialogue-motion alignment and behavior quality, and observe consistent gains over strong co-speech and text-to-motion baselines. ViBES goes beyond "speech-conditioned motion generation" toward agentic virtual bodies where language, prosody, and movement are jointly generated, enabling controllable, socially competent 3D interaction. Code and data will be made available at: this http URL
>
---
#### [replaced 040] When 'YES' Meets 'BUT': Can Large Models Comprehend Contradictory Humor Through Comparative Reasoning?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言理解任务，旨在解决大模型在理解矛盾幽默中的不足。通过构建基准数据集，评估模型在比较推理方面的能力，并探索提升方法。**

- **链接: [https://arxiv.org/pdf/2503.23137](https://arxiv.org/pdf/2503.23137)**

> **作者:** Tuo Liang; Zhe Hu; Jing Li; Hao Zhang; Yiren Lu; Yunlai Zhou; Yiran Qiao; Disheng Liu; Jeirui Peng; Jing Ma; Yu Yin
>
> **摘要:** Understanding humor-particularly when it involves complex, contradictory narratives that require comparative reasoning-remains a significant challenge for large vision-language models (VLMs). This limitation hinders AI's ability to engage in human-like reasoning and cultural expression. In this paper, we investigate this challenge through an in-depth analysis of comics that juxtapose panels to create humor through contradictions. We introduce the YesBut (V2), a novel benchmark with 1,262 comic images from diverse multilingual and multicultural contexts, featuring comprehensive annotations that capture various aspects of narrative understanding. Using this benchmark, we systematically evaluate a wide range of VLMs through four complementary tasks spanning from surface content comprehension to deep narrative reasoning, with particular emphasis on comparative reasoning between contradictory elements. Our extensive experiments reveal that even the most advanced models significantly underperform compared to humans, with common failures in visual perception, key element identification, comparative analysis and hallucinations. We further investigate text-based training strategies and social knowledge augmentation methods to enhance model performance. Our findings not only highlight critical weaknesses in VLMs' understanding of cultural and creative expressions but also provide pathways toward developing context-aware models capable of deeper narrative understanding though comparative reasoning.
>
---
#### [replaced 041] Action Images: End-to-End Policy Learning via Multiview Video Generation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人策略学习任务，旨在解决传统方法依赖分离模块或非像素对齐动作表示的问题。提出Action Images，通过多视角视频生成实现统一策略学习。**

- **链接: [https://arxiv.org/pdf/2604.06168](https://arxiv.org/pdf/2604.06168)**

> **作者:** Haoyu Zhen; Zixian Gao; Qiao Sun; Yilin Zhao; Yuncong Yang; Yilun Du; Pengsheng Guo; Tsun-Hsuan Wang; Yi-Ling Qiao; Chuang Gan
>
> **备注:** Project Page: this https URL
>
> **摘要:** World action models (WAMs) have emerged as a promising direction for robot policy learning, as they can leverage powerful video backbones to model the future states. However, existing approaches often rely on separate action modules, or use action representations that are not pixel-grounded, making it difficult to fully exploit the pretrained knowledge of video models and limiting transfer across viewpoints and environments. In this work, we present Action Images, a unified world action model that formulates policy learning as multiview video generation. Instead of encoding control as low-dimensional tokens, we translate 7-DoF robot actions into interpretable action images: multi-view action videos that are grounded in 2D pixels and explicitly track robot-arm motion. This pixel-grounded action representation allows the video backbone itself to act as a zero-shot policy, without a separate policy head or action module. Beyond control, the same unified model supports video-action joint generation, action-conditioned video generation, and action labeling under a shared representation. On RLBench and real-world evaluations, our model achieves the strongest zero-shot success rates and improves video-action joint generation quality over prior video-space world models, suggesting that interpretable action images are a promising route to policy learning.
>
---
#### [replaced 042] OPTED: Open Preprocessed Trachoma Eye Dataset Using Zero-Shot SAM 3 Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.06885](https://arxiv.org/pdf/2603.06885)**

> **作者:** Kibrom Gebremedhin; Hadush Hailu; Bruk Gebregziabher
>
> **备注:** 9 figure, 3 tables
>
> **摘要:** Trachoma remains the leading infectious cause of blindness worldwide, with Sub-Saharan Africa bearing over 85% of the global burden and Ethiopia alone accounting for more than half of all cases. Yet publicly available preprocessed datasets for automated trachoma classification are scarce, and none originate from the most affected region. Raw clinical photographs of eyelids contain significant background noise that hinders direct use in machine learning pipelines. We present OPTED, an open-source preprocessed trachoma eye dataset constructed using the Segment Anything Model 3 (SAM 3) for automated region-of-interest extraction. We describe a reproducible four-step pipeline: (1) text-prompt-based zero-shot segmentation of the tarsal conjunctiva using SAM 3, (2) background removal and bounding-box cropping with alignment, (3) quality filtering based on confidence scores, and (4) Lanczos resizing to 224x224 pixels. A separate prompt-selection stage identifies the optimal text prompt, and manual quality assurance verifies outputs. Through comparison of five candidate prompts on all 2,832 known-label images, we identify "inner surface of eyelid with red tissue" as optimal, achieving a mean confidence of 0.872 (std 0.070) and 99.5% detection rate (the remaining 13 images are recovered via fallback prompts). The pipeline produces outputs in two formats: cropped and aligned images preserving the original aspect ratio, and standardized 224x224 images ready for pre-trained architectures. The OPTED dataset, preprocessing code, and all experimental artifacts are released as open source to facilitate reproducible trachoma classification research.
>
---
#### [replaced 043] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.20981](https://arxiv.org/pdf/2602.20981)**

> **作者:** Christian Simon; Masato Ishii; Wei-Yao Wang; Koichi Saito; Akio Hayakawa; Dongseok Shim; Zhi Zhong; Shuyang Cui; Shusuke Takahashi; Takashi Shibuya; Yuki Mitsufuji
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Scaling multimodal alignment between video and audio is challenging, particularly due to limited data and the mismatch between text descriptions and frame-level video information. In this work, we tackle the scaling challenge in multimodal-to-audio generation, examining whether models trained on short instances can generalize to longer ones during testing. To tackle this challenge, we present multimodal hierarchical networks so-called MMHNet, an enhanced extension of state-of-the-art video-to-audio models. Our approach integrates a hierarchical method and non-causal Mamba to support long-form audio generation. Our proposed method significantly improves long audio generation up to more than 5 minutes. We also prove that training short and testing long is possible in the video-to-audio generation tasks without training on the longer durations. We show in our experiments that our proposed method could achieve remarkable results on long-video to audio benchmarks, beating prior works in video-to-audio tasks. Moreover, we showcase our model capability in generating more than 5 minutes, while prior video-to-audio methods fall short in generating with long durations.
>
---
#### [replaced 044] FieldWorkArena: Agentic AI Benchmark for Real Field Work Tasks
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19662](https://arxiv.org/pdf/2505.19662)**

> **作者:** Jun Takahashi; Atsunori Moteki; Akiyoshi Uchida; Shoichi Masui; Fan Yang; Kanji Uchino; Yueqi Song; Yonatan Bisk; Graham Neubig; Ikuo Kusajima; Yasuto Watanabe; Hiroyuki Ishida; Koki Nakagawa; Shan Jiang
>
> **备注:** 15 pages, 2 figures, 5 tables [ICPR 2026 Accepted] Changes from Version 2: 1) Added retail domain as third scenario; dataset scaled from 455 to 886 tasks, 2) Task taxonomy restructured (Planning/Perception/Action -> Perception/Decision Making/Combination), 3) Experiments updated: GPT-5.1/5.2, Gemini 2.5 Flash/Pro (replaced Claude); added human baseline and video chunking/Qwen3-VL experiments
>
> **摘要:** This paper introduces FieldWorkArena, a benchmark for agentic AI targeting real-world field work. With the recent increase in demand for agentic AI, they are built to detect and document safety hazards, procedural violations, and other critical incidents across real-world manufacturing and retail environments. Whereas most agentic AI benchmarks focus on performance in simulated or digital environments, our work addresses the fundamental challenge of evaluating agents in the real-world. In this paper, we improve the evaluation function from previous methods to assess the performance of agentic AI in diverse real-world tasks. Our dataset comprises on-site captured images/videos in factories, warehouses and retails. Tasks were meticulously developed through interviews with site workers and managers. Evaluation results confirmed that performance evaluation considering the characteristics of Multimodal LLM (MLLM) such as GPT-4o is feasible. Furthermore, this study identifies both the effectiveness and limitations of the proposed new evaluation methodology. The complete dataset and evaluation program are publicly accessible on the website (this https URL)
>
---
#### [replaced 045] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2211.16780](https://arxiv.org/pdf/2211.16780)**

> **作者:** Quyen Tran; Hai Nguyen; Hoang Phan; Quan Dao; Linh Ngo; Khoat Than; Dinh Phung; Dimitris Metaxas; Trung Le
>
> **备注:** CVPR2026
>
> **摘要:** In online incremental learning, data continuously arrives with substantial distributional shifts, creating a significant challenge because previous samples have limited replay value when learning a new task. Prior research has typically relied on either a single adaptive centroid or multiple fixed centroids to represent each class in the latent space. However, such methods struggle when class data streams are inherently multimodal and require continual centroid updates. To overcome this, we introduce an online Mixture Model learning framework grounded in Optimal Transport theory (MMOT), where centroids evolve incrementally with new data. This approach offers two main advantages: (i) it provides a more precise characterization of complex data streams, and (ii) it enables improved class similarity estimation for unseen samples during inference through MMOT-derived centroids. Furthermore, to strengthen representation learning and mitigate catastrophic forgetting, we design a Dynamic Preservation strategy that regulates the latent space and maintains class separability over time. Experimental evaluations on benchmark datasets confirm the superior effectiveness of our proposed method.
>
---
#### [replaced 046] SemAttNet: Towards Attention-based Semantic Aware Guided Depth Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2204.13635](https://arxiv.org/pdf/2204.13635)**

> **作者:** Danish Nazir; Marcus Liwicki; Didier Stricker; Muhammad Zeshan Afzal
>
> **备注:** accepted at IEEE Access
>
> **摘要:** Depth completion involves recovering a dense depth map from a sparse map and an RGB image. Recent approaches focus on utilizing color images as guidance images to recover depth at invalid pixels. However, color images alone are not enough to provide the necessary semantic understanding of the scene. Consequently, the depth completion task suffers from sudden illumination changes in RGB images (e.g., shadows). In this paper, we propose a novel three-branch backbone comprising color-guided, semantic-guided, and depth-guided branches. Specifically, the color-guided branch takes a sparse depth map and RGB image as an input and generates color depth which includes color cues (e.g., object boundaries) of the scene. The predicted dense depth map of color-guided branch along-with semantic image and sparse depth map is passed as input to semantic-guided branch for estimating semantic depth. The depth-guided branch takes sparse, color, and semantic depths to generate the dense depth map. The color depth, semantic depth, and guided depth are adaptively fused to produce the output of our proposed three-branch backbone. In addition, we also propose to apply semantic-aware multi-modal attention-based fusion block (SAMMAFB) to fuse features between all three branches. We further use CSPN++ with Atrous convolutions to refine the dense depth map produced by our three-branch backbone. Extensive experiments show that our model achieves state-of-the-art performance in the KITTI depth completion benchmark at the time of submission.
>
---
#### [replaced 047] SiLVR: A Simple Language-based Video Reasoning Framework
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.24869](https://arxiv.org/pdf/2505.24869)**

> **作者:** Ce Zhang; Yan-Bo Lin; Ziyang Wang; Mohit Bansal; Gedas Bertasius
>
> **备注:** Accepted by TMLR (01/2026)
>
> **摘要:** Recent advances in test-time optimization have led to remarkable reasoning capabilities in Large Language Models (LLMs), enabling them to solve highly complex problems in math and coding. However, the reasoning capabilities of multimodal LLMs (MLLMs) still significantly lag, especially for complex video-language tasks. To address this issue, we present SILVR, a Simple Language-based Video Reasoning framework that decomposes complex video understanding into two stages. In the first stage, SILVR transforms raw video into language-based representations using multisensory inputs, such as short clip captions and audio/speech subtitles. In the second stage, language descriptions are fed into a powerful reasoning LLM to solve complex video-language understanding tasks. To handle long-context multisensory inputs, we use an Adaptive Context Reduction scheme, which dynamically determines the temporal granularity with which to sample the tokens. Our simple, modular, and training-free video reasoning framework achieves the best-reported results on Video-MME (long), Video-MMMU (comprehension), Video-MMLU, CGBench, and EgoLife. Furthermore, our empirical study focused on video reasoning capabilities shows that, despite not being explicitly trained on video, strong reasoning LLMs can effectively aggregate multisensory input information from video, speech, and audio for complex temporal, causal, long-context, and knowledge acquisition reasoning tasks in video. More details can be found at this https URL.
>
---
#### [replaced 048] VLMs Need Words: Vision Language Models Ignore Visual Detail In Favor of Semantic Anchors
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型在细粒度视觉任务中的表现问题，发现其依赖语义标签而非视觉细节。通过实验验证，提出通过命名未知实体提升性能的方法。**

- **链接: [https://arxiv.org/pdf/2604.02486](https://arxiv.org/pdf/2604.02486)**

> **作者:** Haz Sameen Shahgir; Xiaofu Chen; Yu Fu; Erfan Shayegani; Nael Abu-Ghazaleh; Yova Kementchedjhieva; Yue Dong
>
> **摘要:** Vision-language models (VLMs) have achieved impressive performance across a wide range of multimodal tasks. However, they often fail on tasks that require fine-grained visual perception, even when the required information is still present in their internal representations. Prior work has attributed this ``hidden-in-plain-sight'' gap to the language model, but the cause remains unexplained. In this work, we demonstrate that this gap arises from the language model's lack of semantic labels for fine-grained visual details: when visual entities can be mapped to known concepts, VLMs bypass visual comparison and reason through language; when they cannot, VLMs resort to brittle and hallucinated descriptions. We verify this across semantic correspondence, synthetic shape matching, and face matching, and find that VLMs perform much better when the relevant entities are nameable than when they are unnamable. Mechanistically, Logit Lens analysis confirms that VLMs explicitly recover semantic labels for nameable entities and surface more unique tokens compared to unnameable entities. Furthermore, we show that this limitation can be addressed: teaching completely arbitrary names for unknown entities improves performance. More importantly, task-specific finetuning yields even stronger generalization without relying on language priors, i.e. through real visual perception. Our findings suggest that current VLM failures on visual tasks reflect a learned shortcut rather than a fundamental limitation of multimodal reasoning.
>
---
#### [replaced 049] HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决VLAs忽视历史上下文的问题。通过引入时间感知的时刻标记和轻量记忆模块，使模型具备历史感知能力，提升长周期任务性能。**

- **链接: [https://arxiv.org/pdf/2510.00695](https://arxiv.org/pdf/2510.00695)**

> **作者:** Myungkyu Koo; Daewon Choi; Taeyoung Kim; Kyungmin Lee; Changyeon Kim; Younggyo Seo; Jinwoo Shin
>
> **备注:** ICLR 2026. Project page: this https URL
>
> **摘要:** Inherently, robotic manipulation tasks are history-dependent: leveraging past context could be beneficial. However, most existing Vision-Language-Action models (VLAs) have been designed without considering this aspect, i.e., they rely solely on the current observation, ignoring preceding context. In this paper, we propose HAMLET, a scalable framework to adapt VLAs to attend to the historical context during action prediction. Specifically, we introduce moment tokens that compactly encode perceptual information at each timestep. Their representations are initialized with time-contrastive learning, allowing them to better capture temporally distinctive aspects. Next, we employ a lightweight memory module that integrates the moment tokens across past timesteps into memory features, which are then leveraged for action prediction. Through empirical evaluation, we show that HAMLET successfully transforms a state-of-the-art VLA into a history-aware policy, especially demonstrating significant improvements on long-horizon tasks that require historical context. In particular, on top of GR00T N1.5, HAMLET achieves an average success rate of 76.4% on history-dependent real-world tasks, surpassing the baseline performance by 47.2%. Furthermore, HAMLET pushes prior art performance from 64.1% to 66.4% on RoboCasa Kitchen (100-demo setup) and from 95.6% to 97.7% on LIBERO, highlighting its effectiveness even under generic robot-manipulation benchmarks.
>
---
#### [replaced 050] From Plausibility to Verifiability: Risk-Controlled Generative OCR with Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.19790](https://arxiv.org/pdf/2603.19790)**

> **作者:** Weile Gong; Yiping Zuo; Zijian Lu; Xin He; Weibei Fan; Lianyong Qi; Shi Jin
>
> **备注:** 10 pages, 5 figures, 5 tables
>
> **摘要:** Modern vision-language models (VLMs) can act as generative OCR engines, yet open-ended decoding can expose rare but consequential failures. We identify a core deployment misalignment in generative OCR. Autoregressive decoding favors semantic plausibility, whereas OCR requires outputs that are visually grounded and geometrically verifiable. This mismatch produces severe errors, especially over-generation and unsupported substitutions, creating deployment risk even when benchmark accuracy remains high. We therefore formulate frozen VLM OCR as a selective accept/abstain problem and propose a model-agnostic Geometric Risk Controller. The controller probes multiple structured views of the same input, applies lightweight structural screening, and accepts a transcription only when cross-view consensus and stability satisfy predefined criteria, yielding a small family of operating points. Experiments on frozen VLM backbones and standard OCR benchmarks show consistent reductions in extreme-error risk and catastrophic over-generation at predictable coverage costs. Reliable deployment of generative OCR with frozen VLMs benefits from explicit system-level risk control rather than unconstrained generation.
>
---
#### [replaced 051] Target-Bench: Can Video World Models Achieve Mapless Path Planning with Semantic Targets?
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Target-Bench，评估视频世界模型的语义推理与路径规划能力，解决现实视觉生成与语义理解之间的差距问题。**

- **链接: [https://arxiv.org/pdf/2511.17792](https://arxiv.org/pdf/2511.17792)**

> **作者:** Dingrui Wang; Zhihao Liang; Hongyuan Ye; Zhexiao Sun; Zhaowei Lu; Yuchen Zhang; Yuyu Zhao; Yuan Gao; Marvin Seegert; Finn Schäfer; Haotong Qin; Wei Li; Luigi Palmieri; Felix Jahncke; Mattia Piccinini; Johannes Betz
>
> **备注:** 19 pages
>
> **摘要:** While recent video world models can generate highly realistic videos, their ability to perform semantic reasoning and planning remains unclear and unquantified. We introduce Target-Bench, the first benchmark that enables comprehensive evaluation of video world models' semantic reasoning, spatial estimation, and planning capabilities. Target-Bench provides 450 robot-collected scenarios spanning 47 semantic categories, with SLAM-based trajectories serving as motion tendency references. Our benchmark reconstructs motion from generated videos with a metric scale recovery mechanism, enabling the evaluation of planning performance with five complementary metrics that focus on target-approaching capability and directional consistency. Our evaluation result shows that the best off-the-shelf model achieves only a 0.341 overall score, revealing a significant gap between realistic visual generation and semantic reasoning in current video world models. Furthermore, we demonstrate that fine-tuning process on a relatively small real-world robot dataset can significantly improve task-level planning performance.
>
---
#### [replaced 052] X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决人类与机器人动作不匹配的问题。通过X-Diffusion框架，利用噪声人类动作训练机器人策略，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.04671](https://arxiv.org/pdf/2511.04671)**

> **作者:** Maximus A. Pace; Prithwish Dan; Chuanruo Ning; Atiksh Bhardwaj; Audrey Du; Edward W. Duan; Wei-Chiu Ma; Kushal Kedia
>
> **备注:** ICRA 2026
>
> **摘要:** Human videos are a scalable source of training data for robot learning. However, humans and robots significantly differ in embodiment, making many human actions infeasible for direct execution on a robot. Still, these demonstrations convey rich object-interaction cues and task intent. Our goal is to learn from this coarse guidance without transferring embodiment-specific, infeasible execution strategies. Recent advances in generative modeling tackle a related problem of learning from low-quality data. In particular, Ambient Diffusion is a recent method for diffusion modeling that incorporates low-quality data only at high-noise timesteps of the forward diffusion process. Our key insight is to view human actions as noisy counterparts of robot actions. As noise increases along the forward diffusion process, embodiment-specific differences fade away while task-relevant guidance is preserved. Based on these observations, we present X-Diffusion, a cross-embodiment learning framework based on Ambient Diffusion that selectively trains diffusion policies on noised human actions. This enables effective use of easy-to-collect human videos without sacrificing robot feasibility. Across five real-world manipulation tasks, we show that X-Diffusion improves average success rates by 16% over naive co-training and manual data filtering. The project website is available at this https URL.
>
---
#### [replaced 053] What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.08101](https://arxiv.org/pdf/2407.08101)**

> **作者:** Sunny Panchal; Apratim Bhattacharyya; Guillaume Berger; Antoine Mercier; Cornelius Bohm; Florian Dietrichkeit; Reza Pourreza; Xuanlin Li; Pulkit Madan; Mingu Lee; Mark Todorovich; Ingo Bax; Roland Memisevic
>
> **备注:** Accepted to the 2024 NeurIPS Datasets and Benchmarks track; Data: this https URL Dataset quick start guide: this https URL and Stream-VLM code: this https URL
>
> **摘要:** Vision-language models have shown impressive progress in recent years. However, existing models are largely limited to turn-based interactions, where each turn must be stepped (i.e., prompted) by the user. Open-ended, asynchronous interactions, where an AI model may proactively deliver timely responses or feedback based on the unfolding situation in real-time, are an open challenge. In this work, we present the QEVD benchmark and dataset, which explores human-AI interaction in the challenging, yet controlled, real-world domain of fitness coaching -- a task which intrinsically requires monitoring live user activity and providing immediate feedback. The benchmark requires vision-language models to recognize complex human actions, identify possible mistakes, and provide appropriate feedback in real-time. Our experiments reveal the limitations of existing state-of-the-art vision-language models for such asynchronous situated interactions. Motivated by this, we propose a simple end-to-end streaming baseline that can respond asynchronously to human actions with appropriate feedback at the appropriate time.
>
---
#### [replaced 054] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型优化任务，解决模型过思考导致的效率低和准确率下降问题。提出GPRO方法，通过动态决策路径提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.04442](https://arxiv.org/pdf/2601.04442)**

> **作者:** Xingjian Diao; Zheyuan Liu; Chunhui Zhang; Weiyi Wu; Keyi Kong; Lin Shi; Kaize Ding; Soroush Vosoughi; Jiang Gui
>
> **备注:** Accepted to Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Large Vision-Language Models (LVLMs) have exhibited strong reasoning capabilities through chain-of-thought mechanisms that generate step-by-step rationales. However, such slow-thinking approaches often lead to overthinking, where models produce excessively verbose responses even for simple queries, resulting in test-time inefficiency and even degraded accuracy. Prior work has attempted to mitigate this issue via adaptive reasoning strategies, but these methods largely overlook a fundamental bottleneck: visual perception failures. We argue that stable reasoning critically depends on low-level visual grounding, and that reasoning errors often originate from imperfect perception rather than insufficient deliberation. To address this limitation, we propose Gated Perception-Reasoning Optimization (GPRO), a meta-reasoning controller that dynamically routes computation among three decision paths at each generation step: a lightweight fast path, a slow perception path for re-examining visual inputs, and a slow reasoning path for internal self-reflection. To learn this distinction, we derive large-scale failure attribution supervision from approximately 790k samples, using teacher models to distinguish perceptual hallucinations from reasoning errors. We then train the controller with multi-objective reinforcement learning to optimize the trade-off between task accuracy and computational cost under uncertainty. Experiments on five benchmarks demonstrate that GPRO substantially improves both accuracy and efficiency, outperforming recent slow-thinking methods while generating significantly shorter responses.
>
---
#### [replaced 055] The Gaussian Latent Machine: Efficient Prior and Posterior Sampling for Inverse Problems
- **分类: eess.IV; cs.CV; cs.LG; stat.ML**

- **链接: [https://arxiv.org/pdf/2505.12836](https://arxiv.org/pdf/2505.12836)**

> **作者:** Muhamed Kuric; Martin Zach; Andreas Habring; Michael Unser; Thomas Pock
>
> **摘要:** We consider the problem of sampling from a product-of-experts-type model that encompasses many standard prior and posterior distributions commonly found in Bayesian imaging. We show that this model can be easily lifted into a novel latent variable model, which we refer to as a Gaussian latent machine. This leads to a general sampling approach that unifies and generalizes many existing sampling algorithms in the literature. Most notably, it yields a highly efficient and effective two-block Gibbs sampling approach in the general case, while also specializing to direct sampling algorithms in particular cases. Finally, we present detailed numerical experiments that demonstrate the efficiency and effectiveness of our proposed sampling approach across a wide range of prior and posterior sampling problems from Bayesian imaging.
>
---
#### [replaced 056] Frozen Forecasting: A Unified Evaluation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.13942](https://arxiv.org/pdf/2507.13942)**

> **作者:** Jacob C Walker; Pedro Vélez; Luisa Polania Cabrera; Guangyao Zhou; Sayna Ebrahimi; Rishabh Kabra; Carl Doersch; Maks Ovsjanikov; João Carreira; Shiry Ginosar
>
> **备注:** New Title, Additional Author
>
> **摘要:** Forecasting future events is a fundamental capability for general-purpose systems that plan or act across different levels of abstraction. Yet, evaluating whether a forecast is "correct" remains challenging due to the inherent uncertainty of the future. We propose a unified evaluation framework for assessing the forecasting capabilities of frozen vision backbones across diverse tasks and abstraction levels. Rather than focusing on single time steps, our framework evaluates entire trajectories and incorporates distributional metrics that better capture the multimodal nature of future outcomes. Given a frozen vision model, we train latent diffusion models to forecast future features directly in its representation space, which are then decoded via lightweight, task-specific readouts. This enables consistent evaluation across a suite of diverse tasks while isolating the forecasting capacity of the backbone itself. We apply our framework to nine diverse vision models, spanning image and video pretraining, contrastive and generative objectives, and with or without language supervision, and evaluate them on four forecasting tasks, from low-level pixel predictions to high-level object motion. We find that forecasting performance strongly correlates with perceptual quality and that the forecasting abilities of video synthesis models are comparable or exceed those pretrained in masking regimes across all levels of abstraction. However, language supervision does not consistently improve forecasting. Notably, video-pretrained models consistently outperform image-based ones.
>
---
#### [replaced 057] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ChartNet，一个大规模多模态数据集，用于提升图表理解与推理能力。针对当前视觉语言模型在图表理解上的不足，通过合成与标注，构建包含150万样本的高质量数据集，支持多模态模型训练。**

- **链接: [https://arxiv.org/pdf/2603.27064](https://arxiv.org/pdf/2603.27064)**

> **作者:** Jovana Kondic; Pengyuan Li; Dhiraj Joshi; Isaac Sanchez; Ben Wiesel; Shafiq Abedin; Amit Alfassy; Eli Schwartz; Daniel Caraballo; Yagmur Gizem Cinar; Florian Scheidegger; Steven I. Ross; Daniel Karl I. Weidele; Hang Hua; Ekaterina Arutyunova; Roei Herzig; Zexue He; Zihan Wang; Xinyue Yu; Yunfei Zhao; Sicong Jiang; Minghao Liu; Qunshu Lin; Peter Staar; Luis Lastras; Aude Oliva; Rogerio Feris
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Understanding charts requires models to jointly reason over geometric visual patterns, structured numerical data, and natural language -- a capability where current vision-language models (VLMs) remain limited. We introduce ChartNet, a high-quality, million-scale multimodal dataset designed to advance chart interpretation and reasoning. ChartNet leverages a novel code-guided synthesis pipeline to generate 1.5 million diverse chart samples spanning 24 chart types and 6 plotting libraries. Each sample consists of five aligned components: plotting code, rendered chart image, data table, natural language summary, and question-answering with reasoning, providing fine-grained cross-modal alignment. To capture the full spectrum of chart comprehension, ChartNet additionally includes specialized subsets encompassing human annotated data, real-world data, safety, and grounding. Moreover, a rigorous quality-filtering pipeline ensures visual fidelity, semantic accuracy, and diversity across chart representations. Fine-tuning on ChartNet consistently improves results across benchmarks, demonstrating its utility as large-scale supervision for multimodal models. As the largest open-source dataset of its kind, ChartNet aims to support the development of foundation models with robust and generalizable capabilities for data visualization understanding. The dataset is publicly available at this https URL
>
---
#### [replaced 058] RSGMamba: Reliability-Aware Self-Gated State Space Model for Multimodal Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.12319](https://arxiv.org/pdf/2604.12319)**

> **作者:** Guoan Xu; Yang Xiao; Guangwei Gao; Dongchen Zhu; Guo-Jun Qi; Wenjing Jia
>
> **备注:** 7tables,9 figures
>
> **摘要:** Multimodal semantic segmentation has emerged as a powerful paradigm for enhancing scene understanding by leveraging complementary information from multiple sensing modalities (e.g., RGB, depth, and thermal). However, existing cross-modal fusion methods often implicitly assume that all modalities are equally reliable, which can lead to feature degradation when auxiliary modalities are noisy, misaligned, or incomplete. In this paper, we revisit cross-modal fusion from the perspective of modality reliability and propose a novel framework termed the Reliability-aware Self-Gated State Space Model (RSGMamba). At the core of our method is the Reliability-aware Self-Gated Mamba Block (RSGMB), which explicitly models modality reliability and dynamically regulates cross-modal interactions through a self-gating mechanism. Unlike conventional fusion strategies that indiscriminately exchange information across modalities, RSGMB enables reliability-aware feature selection and enhancing informative feature aggregation. In addition, a lightweight Local Cross-Gated Modulation (LCGM) is incorporated to refine fine-grained spatial details, complementing the global modeling capability of RSGMB. Extensive experiments demonstrate that RSGMamba achieves state-of-the-art performance on both RGB-D and RGB-T semantic segmentation benchmarks, resulting 58.8% / 54.0% mIoU on NYUDepth V2 and SUN-RGBD (+0.4% / +0.7% over prior best), and 61.1% / 88.9% mIoU on MFNet and PST900 (up to +1.6%), with only 48.6M parameters, thereby validating the effectiveness and superiority of the proposed approach.
>
---
#### [replaced 059] SinkSAM-Net: Knowledge-Driven Self-Supervised Sinkhole Segmentation Using Topographic Priors and Segment Anything Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.01473](https://arxiv.org/pdf/2410.01473)**

> **作者:** Osher Rafaeli; Tal Svoray; Ariel Nahlieli
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Soil sinkholes significantly influence soil degradation, infrastructure vulnerability, and landscape evolution. However, their irregular shapes, combined with interference from shadows and vegetation, make it challenging to accurately quantify their properties using remotely sensed data. In addition, manual annotation can be laborious and costly. In this study, we introduce a novel self-supervised framework for sinkhole segmentation, termed SinkSAM-Net, which integrates traditional topographic computations of closed depressions with an iterative, geometry-aware, prompt-based Segment Anything Model (SAM). We generate high-quality pseudo-labels through pixel-level refinement of sinkhole boundaries by integrating monocular depth information with random prompts augmentation technique named coordinate-wise bounding box jittering (CWBJ). These pseudo-labels iteratively enhance a lightweight EfficientNetV2-UNet target model, ultimately transferring knowledge to a prompt-free, low-parameter, and fast inference model. Our proposed approach achieves approximately 95\% of the performance obtained through manual supervision by human annotators. The framework's performance was evaluated on a large sinkhole database, covering diverse sinkhole dateset-induced sinkholes using both aerial and high-resolution drone imagery. This paper presents the first self-supervised framework for sinkhole segmentation, demonstrating the robustness of foundational models (such as SAM and Depth Anything V2) when combined with prior topographic and geometry knowledge and an iterative self-learning pipeline. SinkSAM-Net has the potential to be trained effectively on extensive unlabeled RGB sinkholes datasets, achieving comparable performance to a supervised model. The code and interactive demo for SinkSAM-Net are available at this https URL
>
---
#### [replaced 060] Why and When Visual Token Pruning Fails? A Study on Relevant Visual Information Shift in MLLMs Decoding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.12358](https://arxiv.org/pdf/2604.12358)**

> **作者:** Jiwan Kim; Kibum Kim; Wonjoong Kim; Byung-Kwan Lee; Chanyoung Park
>
> **备注:** Preprint, Project : this https URL
>
> **摘要:** Recently, visual token pruning has been studied to handle the vast number of visual tokens in Multimodal Large Language Models. However, we observe that while existing pruning methods perform reliably on simple visual understanding, they struggle to effectively generalize to complex visual reasoning tasks, a critical gap underexplored in previous studies. Through a systematic analysis, we identify Relevant Visual Information Shift (RVIS) during decoding as the primary failure driver. To address this, we propose Decoding-stage Shift-aware Token Pruning (DSTP), a training-free add-on framework that enables existing pruning methods to align visual tokens with shifting reasoning requirements during the decoding stage. Extensive experiments demonstrate that DSTP significantly mitigates performance degradation of pruning methods in complex reasoning tasks, while consistently yielding performance gains even across visual understanding benchmarks. Furthermore, DSTP demonstrates effectiveness across diverse state-of-the-art architectures, highlighting its generalizability and efficiency with minimal computational overhead.
>
---
#### [replaced 061] Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于工具交互任务，旨在提升大模型在错误后的修复能力。通过结构化反思机制，增强模型的错误诊断与修复能力，提高工具调用的可靠性。**

- **链接: [https://arxiv.org/pdf/2509.18847](https://arxiv.org/pdf/2509.18847)**

> **作者:** Junhao Su; Yuanliang Wan; Junwei Yang; Hengyu Shi; Tianyang Han; Junfeng Luo; Yurui Qiu
>
> **备注:** ACL
>
> **摘要:** Tool-augmented large language models (LLMs) are usually trained with supervised imitation or coarse-grained reinforcement learning that optimizes single tool calls. Current self-reflection practices rely on heuristic prompts or one-way reasoning: the model is urged to 'think more' instead of learning error diagnosis and repair. This is fragile in multi-turn interactions; after a failure the model often repeats the same mistake. We propose structured reflection, which turns the path from error to repair into an explicit, controllable, and trainable action. The agent produces a short yet precise reflection: it diagnoses the failure using evidence from the previous step and then proposes a correct, executable follow-up call. For training we combine DAPO and GSPO objectives with a reward scheme tailored to tool use, optimizing the stepwise strategy Reflect, then Call, then Final. To evaluate, we introduce Tool-Reflection-Bench, a lightweight benchmark that programmatically checks structural validity, executability, parameter correctness, and result consistency. Tasks are built as mini trajectories of erroneous call, reflection, and corrected call, with disjoint train and test splits. Experiments on BFCL v3 and Tool-Reflection-Bench show large gains in multi-turn tool-call success and error recovery, and a reduction of redundant calls. These results indicate that making reflection explicit and optimizing it directly improves the reliability of tool interaction and offers a reproducible path for agents to learn from failure.
>
---
#### [replaced 062] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25699](https://arxiv.org/pdf/2509.25699)**

> **作者:** Xiping Li; Jianghong Ma
>
> **备注:** Accepted by ACL 2026 Main Conference. 30 pages, 6 figures
>
> **摘要:** Interleaved-Modal Chain-of-Thought (I-MCoT) advances vision-language reasoning, such as Visual Question Answering (VQA). This paradigm integrates specially selected visual evidence from the input image into the context of Vision-Language Models (VLMs), enabling them to ground their reasoning logic in these details. Accordingly, the efficacy of an I-MCoT framework relies on identifying what to see (evidence selection) and when to see it (triggering of insertions). However, existing methods fall short in both aspects. First, for selection, they rely on attention signals, which are unreliable -- particularly under severe granularity imbalance between the brief textual query and the informative image. Second, for triggering, they adopt static triggers, which fail to capture the VLMs' dynamic needs for visual evidence. To this end, we propose a novel I-MCoT framework, Active Information-driven Multi-modal Chain-of-Thought (AIM-CoT), which aims to improve both evidence selection and insertion triggering via: (1) Context-enhanced Attention-map Generation (CAG) to mitigate granularity imbalance via textual context enhancement; (2) Active Visual Probing (AVP) to proactively select the most informative evidence via an information foraging process; and (3) Dynamic Attention-shift Trigger (DAT) to precisely activate insertions when VLM's attention shifts from text to visual context. Experiments across three benchmarks and four backbones demonstrate AIM-CoT's consistent superiority. Our code is available at this https URL.
>
---
#### [replaced 063] CoD-Lite: Real-Time Diffusion-Based Generative Image Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.12525](https://arxiv.org/pdf/2604.12525)**

> **作者:** Zhaoyang Jia; Naifu Xue; Zihan Zheng; Jiahao Li; Bin Li; Xiaoyi Zhang; Zongyu Guo; Yuan Zhang; Houqiang Li; Yan Lu
>
> **摘要:** Recent advanced diffusion methods typically derive strong generative priors by scaling diffusion transformers. However, scaling fails to generalize when adapted for real-time compression scenarios that demand lightweight models. In this paper, we explore the design of real-time and lightweight diffusion codecs by addressing two pivotal questions. First, does diffusion pre-training benefit lightweight diffusion codecs? Through systematic analysis, we find that generation-oriented pre-training is less effective at small model scales whereas compression-oriented pre-training yields consistently better performance. Second, are transformers essential? We find that while global attention is crucial for standard generation, lightweight convolutions suffice for compression-oriented diffusion when paired with distillation. Guided by these findings, we establish a one-step lightweight convolution diffusion codec that achieves real-time $60$~FPS encoding and $42$~FPS decoding at 1080p. Further enhanced by distillation and adversarial learning, the proposed codec reduces bitrate by 85\% at a comparable FID to MS-ILLM, bridging the gap between generative compression and practical real-time deployment. Codes are released at this https URL
>
---
#### [replaced 064] AudioX: A Unified Framework for Anything-to-Audio Generation
- **分类: cs.MM; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出AudioX，解决多模态到音频生成的任务，通过统一框架和高质量数据集，提升生成质量与跨模态对齐。**

- **链接: [https://arxiv.org/pdf/2503.10522](https://arxiv.org/pdf/2503.10522)**

> **作者:** Zeyue Tian; Zhaoyang Liu; Yizhu Jin; Ruibin Yuan; Liumeng Xue; Xu Tan; Qifeng Chen; Wei Xue; Yike Guo
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Audio and music generation based on flexible multimodal control signals is a widely applicable topic, with the following key challenges: 1) a unified multimodal modeling framework, and 2) large-scale, high-quality training data. As such, we propose AudioX, a unified framework for anything-to-audio generation that integrates varied multimodal conditions (i.e., text, video, and audio signals) in this work. The core design in this framework is a Multimodal Adaptive Fusion module, which enables the effective fusion of diverse multimodal inputs, enhancing cross-modal alignment and improving overall generation quality. To train this unified model, we construct a large-scale, high-quality dataset, IF-caps, comprising over 7 million samples curated through a structured data annotation pipeline. This dataset provides comprehensive supervision for multimodal-conditioned audio generation. We benchmark AudioX against state-of-the-art methods across a wide range of tasks, finding that our model achieves superior performance, especially in text-to-audio and text-to-music generation. These results demonstrate our method is capable of audio generation under multimodal control signals, showing powerful instruction-following potential. The code and datasets will be available at this https URL.
>
---
#### [replaced 065] Reading Between the Pixels: Linking Text-Image Embedding Alignment to Typographic Attack Success on Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.12371](https://arxiv.org/pdf/2604.12371)**

> **作者:** Ravikumar Balakrishnan; Sanket Mendapara; Ankit Garg
>
> **备注:** Accepted at ICLR 2026 Workshop on Agents in the Wild
>
> **摘要:** We study typographic prompt injection attacks on vision-language models (VLMs), where adversarial text is rendered as images to bypass safety mechanisms, posing a growing threat as VLMs serve as the perceptual backbone of autonomous agents, from browser automation and computer-use systems to camera-equipped embodied agents. In practice, the attack surface is heterogeneous: adversarial text appears at varying font sizes and under diverse visual conditions, while the growing ecosystem of VLMs exhibits substantial variation in vulnerability, complicating defensive approaches. Evaluating 1,000 prompts from SALAD-Bench across four VLMs, namely, GPT-4o, Claude Sonnet 4.5, Mistral-Large-3, and Qwen3-VL-4B-Instruct under varying font sizes (6--28px) and visual transformations (rotation, blur, noise, contrast changes), we find: (1) font size significantly affects attack success rate (ASR), with very small fonts (6px) yielding near-zero ASR while mid-range fonts achieve peak effectiveness; (2) text attacks are more effective than image attacks for GPT-4o (36% vs 8%) and Claude (47% vs 22%), while Qwen3-VL and Mistral show comparable ASR across modalities; (3) text-image embedding distance from two multimodal embedding models (JinaCLIP and Qwen3-VL-Embedding) shows strong negative correlation with ASR across all four models (r = -0.71 to -0.93, p < 0.01); (4) heavy degradations increase embedding distance by 10--12% and reduce ASR by 34--96%, while rotation asymmetrically affects models (Mistral drops 50%, GPT-4o unchanged). These findings highlight that model-specific robustness patterns preclude one-size-fits-all defenses and offer empirical guidance for practitioners selecting VLM backbones for agentic systems operating in adversarial environments.
>
---
#### [replaced 066] Delineate Anything Flow: Fast, Country-Level Field Boundary Detection from Any Source
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13417](https://arxiv.org/pdf/2511.13417)**

> **作者:** Mykola Lavreniuk; Nataliia Kussul; Andrii Shelestov; Yevhenii Salii; Volodymyr Kuzin; Sergii Skakun; Zoltan Szantoi
>
> **摘要:** Accurate delineation of agricultural field boundaries from satellite imagery is essential for land management and crop monitoring, yet existing methods often produce incomplete boundaries, merge adjacent fields, and struggle to scale. We present the Delineate Anything Flow (DelAnyFlow) methodology, a resolution-agnostic approach for large-scale field boundary mapping. DelAnyFlow combines the DelAny instance segmentation model, based on a YOLOv11 backbone and trained on the large-scale Field Boundary Instance Segmentation-22M (FBIS 22M) dataset, with a structured post-processing, merging, vectorization, and simplification to generate vector boundaries. FBIS 22M, the largest dataset of its kind, contains 672,909 multi-resolution image patches (0.25-10m) and 22.9million validated field instances. The DelAny model delivers state-of-the-art accuracy with over 100% higher mAP and 400x faster inference than SAM2. DelAny demonstrates strong zero-shot generalization and supports national-scale applications: using Sentinel 2 data for 2024, DelAnyFlow generated a complete field boundary layer for Ukraine (603,000km2) in under six hours on a single workstation. DelAnyFlow outputs significantly improve boundary completeness relative to operational products from Sinergise Solutions and NASA Harvest, particularly in smallholder and fragmented systems (0.25-1ha). For Ukraine, DelAnyFlow delineated 3.75M fields at 5m and 5.15M at 2.5m, compared to 2.66M detected by Sinergise Solutions and 1.69M by NASA Harvest. This work delivers a scalable, cost-effective methodology for field delineation in regions lacking digital cadastral data. A project landing page with links to model weights, code, national-scale vector outputs, and dataset is available at this https URL.
>
---
#### [replaced 067] UNBOX: Unveiling Black-box visual models with Natural-language
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.08639](https://arxiv.org/pdf/2603.08639)**

> **作者:** Simone Carnemolla; Chiara Russo; Simone Palazzo; Quentin Bouniot; Daniela Giordano; Zeynep Akata; Matteo Pennisi; Concetto Spampinato
>
> **备注:** Under review at IJCV
>
> **摘要:** Ensuring trustworthiness in open-world visual recognition requires models that are interpretable, fair, and robust to distribution shifts. Yet modern vision systems are increasingly deployed as proprietary black-box APIs, exposing only output probabilities and hiding architecture, parameters, gradients, and training data. This opacity prevents meaningful auditing, bias detection, and failure analysis. Existing explanation methods assume white- or gray-box access or knowledge of the training distribution, making them unusable in these real-world settings. We introduce UNBOX, a framework for class-wise model dissection under fully data-free, gradient-free, and backpropagation-free constraints. UNBOX leverages Large Language Models and text-to-image diffusion models to recast activation maximization as a purely semantic search driven by output probabilities. The method produces human-interpretable text descriptors that maximally activate each class, revealing the concepts a model has implicitly learned, the training distribution it reflects, and potential sources of bias. We evaluate UNBOX on ImageNet-1K, Waterbirds, and CelebA through semantic fidelity tests, visual-feature correlation analyses and slice-discovery auditing. Despite operating under the strictest black-box constraints, UNBOX performs competitively with state-of-the-art white-box interpretability methods. This demonstrates that meaningful insight into a model's internal reasoning can be recovered without any internal access, enabling more trustworthy and accountable visual recognition systems.
>
---
