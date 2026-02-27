# 计算机视觉 cs.CV

- **最新发布 139 篇**

- **更新 89 篇**

## 最新发布

#### [new 001] From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于多模态模型训练任务，旨在解决静态数据和固定方法难以发现模型弱点的问题。提出DPE框架，通过诊断驱动的数据生成与强化，实现持续优化。**

- **链接: [https://arxiv.org/pdf/2602.22859v1](https://arxiv.org/pdf/2602.22859v1)**

> **作者:** Hongrui Jia; Chaoya Jiang; Shikun Zhang; Wei Ye
>
> **摘要:** As Large Multimodal Models (LMMs) scale up and reinforcement learning (RL) methods mature, LMMs have made notable progress in complex reasoning and decision making. Yet training still relies on static data and fixed recipes, making it difficult to diagnose capability blind spots or provide dynamic, targeted reinforcement. Motivated by findings that test driven error exposure and feedback based correction outperform repetitive practice, we propose Diagnostic-driven Progressive Evolution (DPE), a spiral loop where diagnosis steers data generation and reinforcement, and each iteration re-diagnoses the updated model to drive the next round of targeted improvement. DPE has two key components. First, multiple agents annotate and quality control massive unlabeled multimodal data, using tools such as web search and image editing to produce diverse, realistic samples. Second, DPE attributes failures to specific weaknesses, dynamically adjusts the data mixture, and guides agents to generate weakness focused data for targeted reinforcement. Experiments on Qwen3-VL-8B-Instruct and Qwen2.5-VL-7B-Instruct show stable, continual gains across eleven benchmarks, indicating DPE as a scalable paradigm for continual LMM training under open task distributions. Our code, models, and data are publicly available at https://github.com/hongruijia/DPE.
>
---
#### [new 002] AgentVista: Evaluating Multimodal Agents in Ultra-Challenging Realistic Visual Scenarios
- **分类: cs.CV**

- **简介: 该论文提出AgentVista基准，用于评估多模态代理在复杂视觉场景中的长期工具使用能力。旨在解决现有基准无法全面反映现实挑战的问题。**

- **链接: [https://arxiv.org/pdf/2602.23166v1](https://arxiv.org/pdf/2602.23166v1)**

> **作者:** Zhaochen Su; Jincheng Gao; Hangyu Guo; Zhenhua Liu; Lueyang Zhang; Xinyu Geng; Shijue Huang; Peng Xia; Guanyu Jiang; Cheng Wang; Yue Zhang; Yi R. Fung; Junxian He
>
> **备注:** The project website is available at \url{https://agentvista-bench.github.io/}, and the code is available at \url{https://github.com/hkust-nlp/AgentVista}
>
> **摘要:** Real-world multimodal agents solve multi-step workflows grounded in visual evidence. For example, an agent can troubleshoot a device by linking a wiring photo to a schematic and validating the fix with online documentation, or plan a trip by interpreting a transit map and checking schedules under routing constraints. However, existing multimodal benchmarks mainly evaluate single-turn visual reasoning or specific tool skills, and they do not fully capture the realism, visual subtlety, and long-horizon tool use that practical agents require. We introduce AgentVista, a benchmark for generalist multimodal agents that spans 25 sub-domains across 7 categories, pairing realistic and detail-rich visual scenarios with natural hybrid tool use. Tasks require long-horizon tool interactions across modalities, including web search, image search, page navigation, and code-based operations for both image processing and general programming. Comprehensive evaluation of state-of-the-art models exposes significant gaps in their ability to carry out long-horizon multimodal tool use. Even the best model in our evaluation, Gemini-3-Pro with tools, achieves only 27.3% overall accuracy, and hard instances can require more than 25 tool-calling turns. We expect AgentVista to accelerate the development of more capable and reliable multimodal agents for realistic and ultra-challenging problem solving.
>
---
#### [new 003] Robust Human Trajectory Prediction via Self-Supervised Skeleton Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于人类轨迹预测任务，旨在解决真实场景中骨骼数据缺失导致的预测精度下降问题。通过自监督骨架表示学习提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.22791v1](https://arxiv.org/pdf/2602.22791v1)**

> **作者:** Taishu Arashima; Hiroshi Kera; Kazuhiko Kawamoto
>
> **备注:** 11 pages main, 5 pages supplementary material
>
> **摘要:** Human trajectory prediction plays a crucial role in applications such as autonomous navigation and video surveillance. While recent works have explored the integration of human skeleton sequences to complement trajectory information, skeleton data in real-world environments often suffer from missing joints caused by occlusions. These disturbances significantly degrade prediction accuracy, indicating the need for more robust skeleton representations. We propose a robust trajectory prediction method that incorporates a self-supervised skeleton representation model pretrained with masked autoencoding. Experimental results in occlusion-prone scenarios show that our method improves robustness to missing skeletal data without sacrificing prediction accuracy, and consistently outperforms baseline models in clean-to-moderate missingness regimes.
>
---
#### [new 004] Beyond Dominant Patches: Spatial Credit Redistribution For Grounded Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型任务，解决模型幻觉问题。提出空间信用重分配方法（SCR），通过重新分配激活信息降低幻觉，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2602.22469v1](https://arxiv.org/pdf/2602.22469v1)**

> **作者:** Niamul Hassan Samin; Md Arifur Rahman; Abdullah Ibne Hanif; Juena Ahmed Noshin; Md Ashikur Rahman
>
> **摘要:** Vision-language models (VLMs) frequently hallucinate objects absent from the input image. We trace this failure to spatial credit collapse: activation credit concentrating on sparse visual patches in early transformer layers, which suppresses contextual evidence and increases reliance on language priors. We introduce Spatial Credit Redistribution (SCR), a training-free inference-time intervention that redistributes hidden-state activation from high-attention source patches to their context, guided by low-entropy inputs. We evaluate six model families (Chameleon, LLaVA, and Qwen, including both Qwen-VL and Qwen2-VL) at scales of 7B, 13B, and 30B, on POPE and CHAIR benchmarks. SCR reduces hallucination by ~4.7-6.0 percentage points on POPE-Adversarial, cuts CHAIR-s by 3.7-5.2 percentage points (42-51 percent relative), and CHAIR-i by 2.7-4.4 percentage points (44-58 percent relative), and preserves CIDEr within 0.8 percentage points. Gains are largest for low-entropy inputs, consistent with the theoretical framework. SCR incurs only 43-56 ms overhead (small models: +43-46 ms; large models: +54-56 ms), roughly 3-6 times lower than OPERA and VCD and 1.3-1.7 times lower than OVCD (+72 ms), while Pareto-dominating all three on both hallucination rate and CIDEr, making it practical for real-time settings. A controlled ablation confirms that attention-guided source selection is essential: replacing it with uniform random selection reduces hallucination rate gains from ~4.7-6.0 percentage points to only ~2.6-3.4 percentage points, pointing to credit-collapse as the key driver.
>
---
#### [new 005] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散模型采样速度慢的问题。提出DPCache框架，通过全局路径规划优化采样步骤，提升效率并保持质量。**

- **链接: [https://arxiv.org/pdf/2602.22654v1](https://arxiv.org/pdf/2602.22654v1)**

> **作者:** Bowen Cui; Yuanbin Wang; Huajiang Xu; Biaolong Chen; Aixi Zhang; Hao Jiang; Zhengzheng Jin; Xu Liu; Pipei Huang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Diffusion models have demonstrated remarkable success in image and video generation, yet their practical deployment remains hindered by the substantial computational overhead of multi-step iterative sampling. Among acceleration strategies, caching-based methods offer a training-free and effective solution by reusing or predicting features across timesteps. However, existing approaches rely on fixed or locally adaptive schedules without considering the global structure of the denoising trajectory, often leading to error accumulation and visual artifacts. To overcome this limitation, we propose DPCache, a novel training-free acceleration framework that formulates diffusion sampling acceleration as a global path planning problem. DPCache constructs a Path-Aware Cost Tensor from a small calibration set to quantify the path-dependent error of skipping timesteps conditioned on the preceding key timestep. Leveraging this tensor, DPCache employs dynamic programming to select an optimal sequence of key timesteps that minimizes the total path cost while preserving trajectory fidelity. During inference, the model performs full computations only at these key timesteps, while intermediate outputs are efficiently predicted using cached features. Extensive experiments on DiT, FLUX, and HunyuanVideo demonstrate that DPCache achieves strong acceleration with minimal quality loss, outperforming prior acceleration methods by $+$0.031 ImageReward at 4.87$\times$ speedup and even surpassing the full-step baseline by $+$0.028 ImageReward at 3.54$\times$ speedup on FLUX, validating the effectiveness of our path-aware global scheduling framework. Code will be released at https://github.com/argsss/DPCache.
>
---
#### [new 006] ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在提升大语言模型的推理能力。针对现有模型推理不足的问题，提出ThinkOmni框架，无需训练即可增强多模态场景下的文本推理。**

- **链接: [https://arxiv.org/pdf/2602.23306v1](https://arxiv.org/pdf/2602.23306v1)**

> **作者:** Yiran Guan; Sifan Tu; Dingkang Liang; Linghao Zhu; Jianzhong Ju; Zhenbo Luo; Jian Luan; Yuliang Liu; Xiang Bai
>
> **备注:** Accept by ICLR 2026
>
> **摘要:** Omni-modal reasoning is essential for intelligent systems to understand and draw inferences from diverse data sources. While existing omni-modal large language models (OLLM) excel at perceiving diverse modalities, they lack the complex reasoning abilities of recent large reasoning models (LRM). However, enhancing the reasoning ability of OLLMs through additional training presents significant challenges, including the need for high-quality data, task-specific adaptation, and substantial computational costs. To address these limitations, we propose ThinkOmni, a training-free and data-free framework that lifts textual reasoning to omni-modal scenarios. ThinkOmni introduces two key components: 1) LRM-as-a-Guide, which leverages off-the-shelf LRMs to guide the OLLM decoding process; 2) Stepwise Contrastive Scaling, which adaptively balances perception and reasoning signals without manual hyperparameter tuning. Experiments on six multi-modal reasoning benchmarks demonstrate that ThinkOmni consistently delivers performance improvements, with main results achieving 70.2 on MathVista and 75.5 on MMAU. Overall, ThinkOmni offers a flexible and generalizable solution for omni-modal reasoning and provides new insights into the generalization and application of reasoning capabilities.
>
---
#### [new 007] ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive Integration of Mobility Proposals
- **分类: cs.CV**

- **简介: 该论文属于物体重建任务，旨在解决复杂多部件物体的自监督重建问题。提出ArtPro框架，通过自适应整合运动提议，提升重建精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.22666v1](https://arxiv.org/pdf/2602.22666v1)**

> **作者:** Xuelu Li; Zhaonan Wang; Xiaogang Wang; Lei Wu; Manyi Li; Changhe Tu
>
> **摘要:** Reconstructing articulated objects into high-fidelity digital twins is crucial for applications such as robotic manipulation and interactive simulation. Recent self-supervised methods using differentiable rendering frameworks like 3D Gaussian Splatting remain highly sensitive to the initial part segmentation. Their reliance on heuristic clustering or pre-trained models often causes optimization to converge to local minima, especially for complex multi-part objects. To address these limitations, we propose ArtPro, a novel self-supervised framework that introduces adaptive integration of mobility proposals. Our approach begins with an over-segmentation initialization guided by geometry features and motion priors, generating part proposals with plausible motion hypotheses. During optimization, we dynamically merge these proposals by analyzing motion consistency among spatial neighbors, while a collision-aware motion pruning mechanism prevents erroneous kinematic estimation. Extensive experiments on both synthetic and real-world objects demonstrate that ArtPro achieves robust reconstruction of complex multi-part objects, significantly outperforming existing methods in accuracy and stability.
>
---
#### [new 008] Plug, Play, and Fortify: A Low-Cost Module for Robust Multimodal Image Understanding Models
- **分类: cs.CV**

- **简介: 该论文属于多模态图像理解任务，解决缺失模态导致模型性能下降的问题。提出一种低开销模块，通过频率域分析动态平衡各模态贡献，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.22644v1](https://arxiv.org/pdf/2602.22644v1)**

> **作者:** Siqi Lu; Wanying Xu; Yongbin Zheng; Wenting Luan; Peng Sun; Jianhang Yao
>
> **摘要:** Missing modalities present a fundamental challenge in multimodal models, often causing catastrophic performance degradation. Our observations suggest that this fragility stems from an imbalanced learning process, where the model develops an implicit preference for certain modalities, leading to the under-optimization of others. We propose a simple yet efficient method to address this challenge. The central insight of our work is that the dominance relationship between modalities can be effectively discerned and quantified in the frequency domain. To leverage this principle, we first introduce a Frequency Ratio Metric (FRM) to quantify modality preference by analyzing features in the frequency domain. Guided by FRM, we then propose a Multimodal Weight Allocation Module, a plug-and-play component that dynamically re-balances the contribution of each branch during training, promoting a more holistic learning paradigm. Extensive experiments demonstrate that MWAM can be seamlessly integrated into diverse architectural backbones, such as those based on CNNs and ViTs. Furthermore, MWAM delivers consistent performance gains across a wide range of tasks and modality combinations. This advancement extends beyond merely optimizing the performance of the base model; it also manifests as further performance improvements to state-of-the-art methods addressing the missing modality problem.
>
---
#### [new 009] Interactive Medical-SAM2 GUI: A Napari-based semi-automatic annotation tool for medical images
- **分类: cs.CV**

- **简介: 该论文属于医学图像标注任务，旨在解决3D医学图像手动标注效率低的问题，提出一种基于Napari的半自动标注工具，支持高效标注与量化分析。**

- **链接: [https://arxiv.org/pdf/2602.22649v1](https://arxiv.org/pdf/2602.22649v1)**

> **作者:** Woojae Hong; Jong Ha Hwang; Jiyong Chung; Joongyeon Choi; Hyunngun Kim; Yong Hwy Kim
>
> **备注:** 6 pages, 2 figures, Planning to submit JOSS (Journal of Open Source Software)
>
> **摘要:** Interactive Medical-SAM2 GUI is an open-source desktop application for semi-automatic annotation of 2D and 3D medical images. Built on the Napari multi-dimensional viewer, box/point prompting is integrated with SAM2-style propagation by treating a 3D volume as a slice sequence, enabling mask propagation from sparse prompts using Medical-SAM2 on top of SAM2. Voxel-level annotation remains essential for developing and validating medical imaging algorithms, yet manual labeling is slow and expensive for 3D scans, and existing integrations frequently emphasize per-slice interaction without providing a unified, cohort-oriented workflow for navigation, propagation, interactive correction, and quantitative export in a single local pipeline. To address this practical limitation, a local-first Napari workflow is provided for efficient 3D annotation across multiple studies using standard DICOM series and/or NIfTI volumes. Users can annotate cases sequentially under a single root folder with explicit proceed/skip actions, initialize objects via box-first prompting (including first/last-slice initialization for single-object propagation), refine predictions with point prompts, and finalize labels through prompt-first correction prior to saving. During export, per-object volumetry and 3D volume rendering are supported, and image geometry is preserved via SimpleITK. The GUI is implemented in Python using Napari and PyTorch, with optional N4 bias-field correction, and is intended exclusively for research annotation workflows. The code is released on the project page: https://github.com/SKKU-IBE/Medical-SAM2GUI/.
>
---
#### [new 010] Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文提出SATtxt，解决卫星图像中RGB输入与多光谱信息对齐的问题，通过知识蒸馏和指令增强实现高效视觉-语言对齐。**

- **链接: [https://arxiv.org/pdf/2602.22613v1](https://arxiv.org/pdf/2602.22613v1)**

> **作者:** Minh Kha Do; Wei Xiang; Kang Han; Di Wu; Khoa Phan; Yi-Ping Phoebe Chen; Gaowen Liu; Ramana Rao Kompella
>
> **摘要:** Vision-language foundation models (VLFMs) promise zero-shot and retrieval understanding for Earth observation. While operational satellite systems often lack full multi-spectral coverage, making RGB-only inference highly desirable for scalable deployment, the adoption of VLFMs for satellite imagery remains hindered by two factors: (1) multi-spectral inputs are informative but difficult to exploit consistently due to band redundancy and misalignment; and (2) CLIP-style text encoders limit semantic expressiveness and weaken fine-grained alignment. We present SATtxt, a spectrum-aware VLFM that operates with RGB inputs only at inference while retaining spectral cues learned during training. Our framework comprises two stages. First, Spectral Representation Distillation transfers spectral priors from a frozen multi-spectral teacher to an RGB student via a lightweight projector. Second, Spectrally Grounded Alignment with Instruction-Augmented LLMs bridges the distilled visual space and an expressive LLM embedding space. Across EuroSAT, BigEarthNet, and ForestNet, SATtxt improves zero-shot classification on average by 4.2%, retrieval by 5.9%, and linear probing by 2.7% over baselines, showing an efficient path toward spectrum-aware vision-language learning for Earth observation. Project page: https://ikhado.github.io/sattxt/
>
---
#### [new 011] CMSA-Net: Causal Multi-scale Aggregation with Adaptive Multi-source Reference for Video Polyp Segmentation
- **分类: cs.CV**

- **简介: 论文提出CMSA-Net，用于视频息肉分割（VPS），解决息肉与黏膜相似、位置尺度变化大导致的分割困难问题。通过多尺度语义聚合和动态多源参考策略提升分割精度与实时性。**

- **链接: [https://arxiv.org/pdf/2602.22821v1](https://arxiv.org/pdf/2602.22821v1)**

> **作者:** Tong Wang; Yaolei Qi; Siwen Wang; Imran Razzak; Guanyu Yang; Yutong Xie
>
> **摘要:** Video polyp segmentation (VPS) is an important task in computer-aided colonoscopy, as it helps doctors accurately locate and track polyps during examinations. However, VPS remains challenging because polyps often look similar to surrounding mucosa, leading to weak semantic discrimination. In addition, large changes in polyp position and scale across video frames make stable and accurate segmentation difficult. To address these challenges, we propose a robust VPS framework named CMSA-Net. The proposed network introduces a Causal Multi-scale Aggregation (CMA) module to effectively gather semantic information from multiple historical frames at different scales. By using causal attention, CMA ensures that temporal feature propagation follows strict time order, which helps reduce noise and improve feature reliability. Furthermore, we design a Dynamic Multi-source Reference (DMR) strategy that adaptively selects informative and reliable reference frames based on semantic separability and prediction confidence. This strategy provides strong multi-frame guidance while keeping the model efficient for real-time inference. Extensive experiments on the SUN-SEG dataset demonstrate that CMSA-Net achieves state-of-the-art performance, offering a favorable balance between segmentation accuracy and real-time clinical applicability.
>
---
#### [new 012] CLIP Is Shortsighted: Paying Attention Beyond the First Sentence
- **分类: cs.CV**

- **简介: 该论文针对多模态学习任务，解决CLIP模型对长文本描述注意力不均的问题。通过改进训练策略，提升长文本检索效果。**

- **链接: [https://arxiv.org/pdf/2602.22419v1](https://arxiv.org/pdf/2602.22419v1)**

> **作者:** Marc-Antoine Lavoie; Anas Mahmoud; Aldo Zaimi; Arsene Fansi Tchango; Steven L. Waslander
>
> **备注:** 19 pages, 13 figures, to be published in the CVPR 2026 proceedings
>
> **摘要:** CLIP models learn transferable multi-modal features via image-text contrastive learning on internet-scale data. They are widely used in zero-shot classification, multi-modal retrieval, text-to-image diffusion, and as image encoders in large vision-language models. However, CLIP's pretraining is dominated by images paired with short captions, biasing the model toward encoding simple descriptions of salient objects and leading to coarse alignment on complex scenes and dense descriptions. While recent work mitigates this by fine-tuning on small-scale long-caption datasets, we identify an important common bias: both human- and LLM-generated long captions typically begin with a one-sentence summary followed by a detailed description. We show that this acts as a shortcut during training, concentrating attention on the opening sentence and early tokens and weakening alignment over the rest of the caption. To resolve this, we introduce DeBias-CLIP, which removes the summary sentence during training and applies sentence sub-sampling and text token padding to distribute supervision across all token positions. DeBias-CLIP achieves state-of-the-art long-text retrieval, improves short-text retrieval, and is less sensitive to sentence order permutations. It is a drop-in replacement for Long-CLIP with no additional trainable parameters.
>
---
#### [new 013] Large Multimodal Models as General In-Context Classifiers
- **分类: cs.CV**

- **简介: 论文探讨了大型多模态模型（LMM）在分类任务中的应用，旨在解决传统对比模型在零样本和开放世界场景下的局限性。研究显示LMM通过上下文学习可达到甚至超越对比模型效果，并提出CIRCLE方法提升其性能。**

- **链接: [https://arxiv.org/pdf/2602.23229v1](https://arxiv.org/pdf/2602.23229v1)**

> **作者:** Marco Garosi; Matteo Farina; Alessandro Conti; Massimiliano Mancini; Elisa Ricci
>
> **备注:** CVPR Findings 2026. Project website at https://circle-lmm.github.io/
>
> **摘要:** Which multimodal model should we use for classification? Previous studies suggest that the answer lies in CLIP-like contrastive Vision-Language Models (VLMs), due to their remarkable performance in zero-shot classification. In contrast, Large Multimodal Models (LMM) are more suitable for complex tasks. In this work, we argue that this answer overlooks an important capability of LMMs: in-context learning. We benchmark state-of-the-art LMMs on diverse datasets for closed-world classification and find that, although their zero-shot performance is lower than CLIP's, LMMs with a few in-context examples can match or even surpass contrastive VLMs with cache-based adapters, their "in-context" equivalent. We extend this analysis to the open-world setting, where the generative nature of LMMs makes them more suitable for the task. In this challenging scenario, LMMs struggle whenever provided with imperfect context information. To address this issue, we propose CIRCLE, a simple training-free method that assigns pseudo-labels to in-context examples, iteratively refining them with the available context itself. Through extensive experiments, we show that CIRCLE establishes a robust baseline for open-world classification, surpassing VLM counterparts and highlighting the potential of LMMs to serve as unified classifiers, and a flexible alternative to specialized models.
>
---
#### [new 014] Coded-E2LF: Coded Aperture Light Field Imaging from Events
- **分类: cs.CV**

- **简介: 该论文属于光场成像任务，旨在通过事件传感器和编码孔径重建4D光场。工作包括纯事件驱动方法、编码模式分析及硬件验证。**

- **链接: [https://arxiv.org/pdf/2602.22620v1](https://arxiv.org/pdf/2602.22620v1)**

> **作者:** Tomoya Tsuchida; Keita Takahashi; Chihiro Tsutake; Toshiaki Fujii; Hajime Nagahara
>
> **备注:** accepted to CVPR 2026
>
> **摘要:** We propose Coded-E2LF (coded event to light field), a computational imaging method for acquiring a 4-D light field using a coded aperture and a stationary event-only camera. In a previous work, an imaging system similar to ours was adopted, but both events and intensity images were captured and used for light field reconstruction. In contrast, our method is purely event-based, which relaxes restrictions for hardware implementation. We also introduce several advancements from the previous work that enable us to theoretically support and practically improve light field reconstruction from events alone. In particular, we clarify the key role of a black pattern in aperture coding patterns. We finally implemented our method on real imaging hardware to demonstrate its effectiveness in capturing real 3-D scenes. To the best of our knowledge, we are the first to demonstrate that a 4-D light field with pixel-level accuracy can be reconstructed from events alone. Our software and supplementary video are available from our project website.
>
---
#### [new 015] Phys-3D: Physics-Constrained Real-Time Crowd Tracking and Counting on Railway Platforms
- **分类: cs.CV**

- **简介: 论文提出Phys-3D框架，解决铁路平台实时人群计数问题。通过融合物理约束与深度学习，提升动态场景下的计数准确性。**

- **链接: [https://arxiv.org/pdf/2602.23177v1](https://arxiv.org/pdf/2602.23177v1)**

> **作者:** Bin Zeng; Johannes Künzel; Anna Hilsmann; Peter Eisert
>
> **备注:** published at VISAPP 2026
>
> **摘要:** Accurate, real-time crowd counting on railway platforms is essential for safety and capacity management. We propose to use a single camera mounted in a train, scanning the platform while arriving. While hardware constraints are simple, counting remains challenging due to dense occlusions, camera motion, and perspective distortions during train arrivals. Most existing tracking-by-detection approaches assume static cameras or ignore physical consistency in motion modeling, leading to unreliable counting under dynamic conditions. We propose a physics-constrained tracking framework that unifies detection, appearance, and 3D motion reasoning in a real-time pipeline. Our approach integrates a transfer-learned YOLOv11m detector with EfficientNet-B0 appearance encoding within DeepSORT, while introducing a physics-constrained Kalman model (Phys-3D) that enforces physically plausible 3D motion dynamics through pinhole geometry. To address counting brittleness under occlusions, we implement a virtual counting band with persistence. On our platform benchmark, MOT-RailwayPlatformCrowdHead Dataset(MOT-RPCH), our method reduces counting error to 2.97%, demonstrating robust performance despite motion and occlusions. Our results show that incorporating first-principles geometry and motion priors enables reliable crowd counting in safety-critical transportation scenarios, facilitating effective train scheduling and platform safety management.
>
---
#### [new 016] ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization
- **分类: cs.CV**

- **简介: 该论文属于视觉自回归模型优化任务，解决生成效率低的问题。通过三维熵分析和稀疏优化，提升生成速度并保持质量。**

- **链接: [https://arxiv.org/pdf/2602.22948v1](https://arxiv.org/pdf/2602.22948v1)**

> **作者:** Jiayu Chen; Ruoyu Lin; Zihao Zheng; Jingxin Li; Maoliang Li; Guojie Luo; Xiang chen
>
> **备注:** ToProVAR is honored to be accepted by ICLR 2026
>
> **摘要:** Visual Autoregressive(VAR) models enhance generation quality but face a critical efficiency bottleneck in later stages. In this paper, we present a novel optimization framework for VAR models that fundamentally differs from prior approaches such as FastVAR and SkipVAR. Instead of relying on heuristic skipping strategies, our method leverages attention entropy to characterize the semantic projections across different dimensions of the model architecture. This enables precise identification of parameter dynamics under varying token granularity levels, semantic scopes, and generation scales. Building on this analysis, we further uncover sparsity patterns along three critical dimensions-token, layer, and scale-and propose a set of fine-grained optimization strategies tailored to these patterns. Extensive evaluation demonstrates that our approach achieves aggressive acceleration of the generation process while significantly preserving semantic fidelity and fine details, outperforming traditional methods in both efficiency and quality. Experiments on Infinity-2B and Infinity-8B models demonstrate that ToProVAR achieves up to 3.4x acceleration with minimal quality loss, effectively mitigating the issues found in prior work. Our code will be made publicly available.
>
---
#### [new 017] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis
- **分类: cs.CV**

- **简介: 该论文属于手语识别任务，解决签字手模糊、训练损失不足和词汇外问题。提出OpenFS方法，实现多手识别与隐式签字手检测，并引入新损失函数和生成器提升性能。**

- **链接: [https://arxiv.org/pdf/2602.22949v1](https://arxiv.org/pdf/2602.22949v1)**

> **作者:** Junuk Cha; Jihyeon Kim; Han-Mu Park
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Fingerspelling is a component of sign languages in which words are spelled out letter by letter using specific hand poses. Automatic fingerspelling recognition plays a crucial role in bridging the communication gap between Deaf and hearing communities, yet it remains challenging due to the signing-hand ambiguity issue, the lack of appropriate training losses, and the out-of-vocabulary (OOV) problem. Prior fingerspelling recognition methods rely on explicit signing-hand detection, which often leads to recognition failures, and on a connectionist temporal classification (CTC) loss, which exhibits the peaky behavior problem. To address these issues, we develop OpenFS, an open-source approach for fingerspelling recognition and synthesis. We propose a multi-hand-capable fingerspelling recognizer that supports both single- and multi-hand inputs and performs implicit signing-hand detection by incorporating a dual-level positional encoding and a signing-hand focus (SF) loss. The SF loss encourages cross-attention to focus on the signing hand, enabling implicit signing-hand detection during recognition. Furthermore, without relying on the CTC loss, we introduce a monotonic alignment (MA) loss that enforces the output letter sequence to follow the temporal order of the input pose sequence through cross-attention regularization. In addition, we propose a frame-wise letter-conditioned generator that synthesizes realistic fingerspelling pose sequences for OOV words. This generator enables the construction of a new synthetic benchmark, called FSNeo. Through comprehensive experiments, we demonstrate that our approach achieves state-of-the-art performance in recognition and validate the effectiveness of the proposed recognizer and generator. Codes and data are available in: https://github.com/JunukCha/OpenFS.
>
---
#### [new 018] Skarimva: Skeleton-based Action Recognition is a Multi-view Application
- **分类: cs.CV**

- **简介: 该论文属于骨架动作识别任务，旨在提升模型性能。通过多视角数据提高骨架质量，证明其能显著改善识别效果，建议未来研究采用多视角方案。**

- **链接: [https://arxiv.org/pdf/2602.23231v1](https://arxiv.org/pdf/2602.23231v1)**

> **作者:** Daniel Bermuth; Alexander Poeppel; Wolfgang Reif
>
> **摘要:** Human action recognition plays an important role when developing intelligent interactions between humans and machines. While there is a lot of active research on improving the machine learning algorithms for skeleton-based action recognition, not much attention has been given to the quality of the input skeleton data itself. This work demonstrates that by making use of multiple camera views to triangulate more accurate 3D~skeletons, the performance of state-of-the-art action recognition models can be improved significantly. This suggests that the quality of the input data is currently a limiting factor for the performance of these models. Based on these results, it is argued that the cost-benefit ratio of using multiple cameras is very favorable in most practical use-cases, therefore future research in skeleton-based action recognition should consider multi-view applications as the standard setup.
>
---
#### [new 019] Efficient Encoder-Free Fourier-based 3D Large Multimodal Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Fase3D，一种无需编码器的高效3D多模态模型。针对3D数据无序性和大规模问题，通过傅里叶变换和序列化实现高效特征提取，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.23153v1](https://arxiv.org/pdf/2602.23153v1)**

> **作者:** Guofeng Mei; Wei Lin; Luigi Riz; Yujiao Wu; Yiming Wang; Fabio Poiesi
>
> **摘要:** Large Multimodal Models (LMMs) that process 3D data typically rely on heavy, pre-trained visual encoders to extract geometric features. While recent 2D LMMs have begun to eliminate such encoders for efficiency and scalability, extending this paradigm to 3D remains challenging due to the unordered and large-scale nature of point clouds. This leaves a critical unanswered question: How can we design an LMM that tokenizes unordered 3D data effectively and efficiently without a cumbersome encoder? We propose Fase3D, the first efficient encoder-free Fourier-based 3D scene LMM. Fase3D tackles the challenges of scalability and permutation invariance with a novel tokenizer that combines point cloud serialization and the Fast Fourier Transform (FFT) to approximate self-attention. This design enables an effective and computationally minimal architecture, built upon three key innovations: First, we represent large scenes compactly via structured superpoints. Second, our space-filling curve serialization followed by an FFT enables efficient global context modeling and graph-based token merging. Lastly, our Fourier-augmented LoRA adapters inject global frequency-aware interactions into the LLMs at a negligible cost. Fase3D achieves performance comparable to encoder-based 3D LMMs while being significantly more efficient in computation and parameters. Project website: https://tev-fbk.github.io/Fase3D.
>
---
#### [new 020] Retrieve and Segment: Are a Few Examples Enough to Bridge the Supervision Gap in Open-Vocabulary Segmentation?
- **分类: cs.CV**

- **简介: 该论文属于开放词汇分割任务，旨在解决监督不足下的分割问题。通过引入少量标注图像增强文本提示，提出一种融合文本与视觉特征的轻量级分类器，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2602.23339v1](https://arxiv.org/pdf/2602.23339v1)**

> **作者:** Tilemachos Aravanis; Vladan Stojnić; Bill Psomas; Nikos Komodakis; Giorgos Tolias
>
> **摘要:** Open-vocabulary segmentation (OVS) extends the zero-shot recognition capabilities of vision-language models (VLMs) to pixel-level prediction, enabling segmentation of arbitrary categories specified by text prompts. Despite recent progress, OVS lags behind fully supervised approaches due to two challenges: the coarse image-level supervision used to train VLMs and the semantic ambiguity of natural language. We address these limitations by introducing a few-shot setting that augments textual prompts with a support set of pixel-annotated images. Building on this, we propose a retrieval-augmented test-time adapter that learns a lightweight, per-image classifier by fusing textual and visual support features. Unlike prior methods relying on late, hand-crafted fusion, our approach performs learned, per-query fusion, achieving stronger synergy between modalities. The method supports continually expanding support sets, and applies to fine-grained tasks such as personalized segmentation. Experiments show that we significantly narrow the gap between zero-shot and supervised segmentation while preserving open-vocabulary ability.
>
---
#### [new 021] Uni-Animator: Towards Unified Visual Colorization
- **分类: cs.CV**

- **简介: 该论文提出Uni-Animator，解决图像与视频草图着色任务中的颜色传递不准确、细节丢失和时间不一致问题，通过引入视觉参考增强、物理细节强化和动态RoPE编码实现统一的高质量着色。**

- **链接: [https://arxiv.org/pdf/2602.23191v1](https://arxiv.org/pdf/2602.23191v1)**

> **作者:** Xinyuan Chen; Yao Xu; Shaowen Wang; Pengjie Song; Bowen Deng
>
> **备注:** 10 pages, 8 figures. Submitted to CVPR 2026
>
> **摘要:** We propose Uni-Animator, a novel Diffusion Transformer (DiT)-based framework for unified image and video sketch colorization. Existing sketch colorization methods struggle to unify image and video tasks, suffering from imprecise color transfer with single or multiple references, inadequate preservation of high-frequency physical details, and compromised temporal coherence with motion artifacts in large-motion scenes. To tackle imprecise color transfer, we introduce visual reference enhancement via instance patch embedding, enabling precise alignment and fusion of reference color information. To resolve insufficient physical detail preservation, we design physical detail reinforcement using physical features that effectively capture and retain high-frequency textures. To mitigate motion-induced temporal inconsistency, we propose sketch-based dynamic RoPE encoding that adaptively models motion-aware spatial-temporal dependencies. Extensive experimental results demonstrate that Uni-Animator achieves competitive performance on both image and video sketch colorization, matching that of task-specific methods while unlocking unified cross-domain capabilities with high detail fidelity and robust temporal consistency.
>
---
#### [new 022] Devling into Adversarial Transferability on Image Classification: Review, Benchmark, and Evaluation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于对抗攻击领域，旨在解决对抗样本转移性评估不统一的问题。通过分类攻击方法、构建评估框架并分析提升转移性的策略，推动公平比较与安全研究。**

- **链接: [https://arxiv.org/pdf/2602.23117v1](https://arxiv.org/pdf/2602.23117v1)**

> **作者:** Xiaosen Wang; Zhijin Ge; Bohan Liu; Zheng Fang; Fengfan Zhou; Ruixuan Zhang; Shaokang Wang; Yuyang Luo
>
> **备注:** Code is available at https://github.com/Trustworthy-AI-Group/TransferAttack
>
> **摘要:** Adversarial transferability refers to the capacity of adversarial examples generated on the surrogate model to deceive alternate, unexposed victim models. This property eliminates the need for direct access to the victim model during an attack, thereby raising considerable security concerns in practical applications and attracting substantial research attention recently. In this work, we discern a lack of a standardized framework and criteria for evaluating transfer-based attacks, leading to potentially biased assessments of existing approaches. To rectify this gap, we have conducted an exhaustive review of hundreds of related works, organizing various transfer-based attacks into six distinct categories. Subsequently, we propose a comprehensive framework designed to serve as a benchmark for evaluating these attacks. In addition, we delineate common strategies that enhance adversarial transferability and highlight prevalent issues that could lead to unfair comparisons. Finally, we provide a brief review of transfer-based attacks beyond image classification.
>
---
#### [new 023] PhotoAgent: Agentic Photo Editing with Exploratory Visual Aesthetic Planning
- **分类: cs.CV**

- **简介: 该论文提出PhotoAgent，解决自主图像编辑问题。通过美学规划和多步骤决策，提升编辑质量与指令遵循度。**

- **链接: [https://arxiv.org/pdf/2602.22809v1](https://arxiv.org/pdf/2602.22809v1)**

> **作者:** Mingde Yao; Zhiyuan You; Tam-King Man; Menglu Wang; Tianfan Xue
>
> **备注:** A fully automated, intelligent photo-editing agent that autonomously plans multi-step aesthetic enhancements, smartly chooses diverse editing tools, and enables everyday users to achieve professional-looking results without crafting complex prompts. Project page: https://github.com/mdyao/PhotoAgent
>
> **摘要:** With the recent fast development of generative models, instruction-based image editing has shown great potential in generating high-quality images. However, the quality of editing highly depends on carefully designed instructions, placing the burden of task decomposition and sequencing entirely on the user. To achieve autonomous image editing, we present PhotoAgent, a system that advances image editing through explicit aesthetic planning. Specifically, PhotoAgent formulates autonomous image editing as a long-horizon decision-making problem. It reasons over user aesthetic intent, plans multi-step editing actions via tree search, and iteratively refines results through closed-loop execution with memory and visual feedback, without requiring step-by-step user prompts. To support reliable evaluation in real-world scenarios, we introduce UGC-Edit, an aesthetic evaluation benchmark consisting of 7,000 photos and a learned aesthetic reward model. We also construct a test set containing 1,017 photos to systematically assess autonomous photo editing performance. Extensive experiments demonstrate that PhotoAgent consistently improves both instruction adherence and visual quality compared with baseline methods. The project page is https://github.com/mdyao/PhotoAgent.
>
---
#### [new 024] Beyond Detection: Multi-Scale Hidden-Code for Natural Image Deepfake Recovery and Factual Retrieval
- **分类: cs.CV**

- **简介: 该论文属于图像真实性任务，旨在解决深度伪造内容的恢复与事实检索问题。通过构建统一的隐码恢复框架，实现从水印中提取语义信息并进行重建。**

- **链接: [https://arxiv.org/pdf/2602.22759v1](https://arxiv.org/pdf/2602.22759v1)**

> **作者:** Yuan-Chih Chen; Chun-Shien Lu
>
> **摘要:** Recent advances in image authenticity have primarily focused on deepfake detection and localization, leaving recovery of tampered contents for factual retrieval relatively underexplored. We propose a unified hidden-code recovery framework that enables both retrieval and restoration from post-hoc and in-generation watermarking paradigms. Our method encodes semantic and perceptual information into a compact hidden-code representation, refined through multi-scale vector quantization, and enhances contextual reasoning via conditional Transformer modules. To enable systematic evaluation for natural images, we construct ImageNet-S, a benchmark that provides paired image-label factual retrieval tasks. Extensive experiments on ImageNet-S demonstrate that our method exhibits promising retrieval and reconstruction performance while remaining fully compatible with diverse watermarking pipelines. This framework establishes a foundation for general-purpose image recovery beyond detection and localization.
>
---
#### [new 025] IRSDE-Despeckle: A Physics-Grounded Diffusion Model for Generalizable Ultrasound Despeckling
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决超声图像去斑问题。通过构建物理引导的扩散模型，提升图像质量并量化预测不确定性。**

- **链接: [https://arxiv.org/pdf/2602.22717v1](https://arxiv.org/pdf/2602.22717v1)**

> **作者:** Shuoqi Chen; Yujia Wu; Geoffrey P. Luke
>
> **备注:** 12 pages main text + 6 pages appendix, 7 figures main + 3 figures appendix, 3 tables main + 1 table appendix. Preprint
>
> **摘要:** Ultrasound imaging is widely used for real-time, noninvasive diagnosis, but speckle and related artifacts reduce image quality and can hinder interpretation. We present a diffusion-based ultrasound despeckling method built on the Image Restoration Stochastic Differential Equations framework. To enable supervised training, we curate large paired datasets by simulating ultrasound images from speckle-free magnetic resonance images using the Matlab UltraSound Toolbox. The proposed model reconstructs speckle-suppressed images while preserving anatomically meaningful edges and contrast. On a held-out simulated test set, our approach consistently outperforms classical filters and recent learning-based despeckling baselines. We quantify prediction uncertainty via cross-model variance and show that higher uncertainty correlates with higher reconstruction error, providing a practical indicator of difficult or failure-prone regions. Finally, we evaluate sensitivity to simulation probe settings and observe domain shift, motivating diversified training and adaptation for robust clinical deployment.
>
---
#### [new 026] MM-NeuroOnco: A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像诊断任务，旨在解决脑肿瘤MRI诊断中语义标注不足的问题。通过构建多模态数据集和基准，提升模型的诊断理解能力。**

- **链接: [https://arxiv.org/pdf/2602.22955v1](https://arxiv.org/pdf/2602.22955v1)**

> **作者:** Feng Guo; Jiaxiang Liu; Yang Li; Qianqian Shi; Mingkun Xu
>
> **摘要:** Accurate brain tumor diagnosis requires models to not only detect lesions but also generate clinically interpretable reasoning grounded in imaging manifestations, yet existing public datasets remain limited in annotation richness and diagnostic semantics. To bridge this gap, we introduce MM-NeuroOnco, a large-scale multimodal benchmark and instruction-tuning dataset for brain tumor MRI understanding, consisting of 24,726 MRI slices from 20 data sources paired with approximately 200,000 semantically enriched multimodal instructions spanning diverse tumor subtypes and imaging modalities. To mitigate the scarcity and high cost of diagnostic semantic annotations, we develop a multi-model collaborative pipeline for automated medical information completion and quality control, enabling the generation of diagnosis-related semantics beyond mask-only annotations. Building upon this dataset, we further construct MM-NeuroOnco-Bench, a manually annotated evaluation benchmark with a rejection-aware setting to reduce biases inherent in closed-ended question formats. Evaluation across ten representative models shows that even the strongest baseline, Gemini 3 Flash, achieves only 41.88% accuracy on diagnosis-related questions, highlighting the substantial challenges of multimodal brain tumor diagnostic understanding. Leveraging MM-NeuroOnco, we further propose NeuroOnco-GPT, which achieves a 27% absolute accuracy improvement on diagnostic questions following fine-tuning. This result demonstrates the effectiveness of our dataset and benchmark in advancing clinically grounded multimodal diagnostic reasoning. Code and dataset are publicly available at: https://github.com/gfnnnb/MM-NeuroOnco
>
---
#### [new 027] Motion-aware Event Suppression for Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种运动感知事件抑制框架，用于过滤事件相机中由IMOs和自运动引起的噪声。任务是提升事件流处理的准确性与效率，通过实时分割与预测运动实现动态事件的提前抑制。**

- **链接: [https://arxiv.org/pdf/2602.23204v1](https://arxiv.org/pdf/2602.23204v1)**

> **作者:** Roberto Pellerito; Nico Messikommer; Giovanni Cioffi; Marco Cannici; Davide Scaramuzza
>
> **摘要:** In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.
>
---
#### [new 028] Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自主驾驶任务，解决E2E-AD在罕见场景下的泛化与安全性问题。提出RaWMPC框架，通过世界模型和风险评估实现安全决策。**

- **链接: [https://arxiv.org/pdf/2602.23259v1](https://arxiv.org/pdf/2602.23259v1)**

> **作者:** Jiangxin Sun; Feng Xue; Teng Long; Chang Liu; Jian-Fang Hu; Wei-Shi Zheng; Nicu Sebe
>
> **摘要:** With advances in imitation learning (IL) and large-scale driving datasets, end-to-end autonomous driving (E2E-AD) has made great progress recently. Currently, IL-based methods have become a mainstream paradigm: models rely on standard driving behaviors given by experts, and learn to minimize the discrepancy between their actions and expert actions. However, this objective of "only driving like the expert" suffers from limited generalization: when encountering rare or unseen long-tail scenarios outside the distribution of expert demonstrations, models tend to produce unsafe decisions in the absence of prior experience. This raises a fundamental question: Can an E2E-AD system make reliable decisions without any expert action supervision? Motivated by this, we propose a unified framework named Risk-aware World Model Predictive Control (RaWMPC) to address this generalization dilemma through robust control, without reliance on expert demonstrations. Practically, RaWMPC leverages a world model to predict the consequences of multiple candidate actions and selects low-risk actions through explicit risk evaluation. To endow the world model with the ability to predict the outcomes of risky driving behaviors, we design a risk-aware interaction strategy that systematically exposes the world model to hazardous behaviors, making catastrophic outcomes predictable and thus avoidable. Furthermore, to generate low-risk candidate actions at test time, we introduce a self-evaluation distillation method to distill riskavoidance capabilities from the well-trained world model into a generative action proposal network without any expert demonstration. Extensive experiments show that RaWMPC outperforms state-of-the-art methods in both in-distribution and out-of-distribution scenarios, while providing superior decision interpretability.
>
---
#### [new 029] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于隐私安全任务，解决无标注图像的成员推理问题。提出MoFit框架，通过模型拟合嵌入实现无需 captions 的攻击，提升隐私泄露检测效果。**

- **链接: [https://arxiv.org/pdf/2602.22689v1](https://arxiv.org/pdf/2602.22689v1)**

> **作者:** Joonsung Jeon; Woo Jae Kim; Suhyeon Ha; Sooel Son; Sung-Eui Yoon
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Latent diffusion models have achieved remarkable success in high-fidelity text-to-image generation, but their tendency to memorize training data raises critical privacy and intellectual property concerns. Membership inference attacks (MIAs) provide a principled way to audit such memorization by determining whether a given sample was included in training. However, existing approaches assume access to ground-truth captions. This assumption fails in realistic scenarios where only images are available and their textual annotations remain undisclosed, rendering prior methods ineffective when substituted with vision-language model (VLM) captions. In this work, we propose MoFit, a caption-free MIA framework that constructs synthetic conditioning inputs that are explicitly overfitted to the target model's generative manifold. Given a query image, MoFit proceeds in two stages: (i) model-fitted surrogate optimization, where a perturbation applied to the image is optimized to construct a surrogate in regions of the model's unconditional prior learned from member samples, and (ii) surrogate-driven embedding extraction, where a model-fitted embedding is derived from the surrogate and then used as a mismatched condition for the query image. This embedding amplifies conditional loss responses for member samples while leaving hold-outs relatively less affected, thereby enhancing separability in the absence of ground-truth captions. Our comprehensive experiments across multiple datasets and diffusion models demonstrate that MoFit consistently outperforms prior VLM-conditioned baselines and achieves performance competitive with caption-dependent methods.
>
---
#### [new 030] Spatio-Temporal Token Pruning for Efficient High-Resolution GUI Agents
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GUIPruner，解决高分辨率GUI代理的效率问题，通过时空剪枝提升性能，减少计算量并保持精度。**

- **链接: [https://arxiv.org/pdf/2602.23235v1](https://arxiv.org/pdf/2602.23235v1)**

> **作者:** Zhou Xu; Bowen Zhou; Qi Wang; Shuwen Feng; Jingyu Xiao
>
> **摘要:** Pure-vision GUI agents provide universal interaction capabilities but suffer from severe efficiency bottlenecks due to the massive spatiotemporal redundancy inherent in high-resolution screenshots and historical trajectories. We identify two critical misalignments in existing compression paradigms: the temporal mismatch, where uniform history encoding diverges from the agent's "fading memory" attention pattern, and the spatial topology conflict, where unstructured pruning compromises the grid integrity required for precise coordinate grounding, inducing spatial hallucinations. To address these challenges, we introduce GUIPruner, a training-free framework tailored for high-resolution GUI navigation. It synergizes Temporal-Adaptive Resolution (TAR), which eliminates historical redundancy via decay-based resizing, and Stratified Structure-aware Pruning (SSP), which prioritizes interactive foregrounds and semantic anchors while safeguarding global layout. Extensive evaluations across diverse benchmarks demonstrate that GUIPruner consistently achieves state-of-the-art performance, effectively preventing the collapse observed in large-scale models under high compression. Notably, on Qwen2-VL-2B, our method delivers a 3.4x reduction in FLOPs and a 3.3x speedup in vision encoding latency while retaining over 94% of the original performance, enabling real-time, high-precision navigation with minimal resource consumption.
>
---
#### [new 031] DisQ-HNet: A Disentangled Quantized Half-UNet for Interpretable Multimodal Image Synthesis Applications to Tau-PET Synthesis from T1 and FLAIR MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DisQ-HNet，用于从T1和FLAIR MRI合成tau-PET图像，解决阿尔茨海默病影像分析中的替代方案问题。通过解耦量化编码器和半UNet解码器，提升可解释性和疾病相关信号保留。**

- **链接: [https://arxiv.org/pdf/2602.22545v1](https://arxiv.org/pdf/2602.22545v1)**

> **作者:** Agamdeep S. Chopra; Caitlin Neher; Tianyi Ren; Juampablo E. Heras Rivera; Mehmet Kurt
>
> **备注:** 14 pages, 8 figures, 8 tables; includes PID guided vector quantized latent factorization and sobel edge conditioned Half-UNet decoder
>
> **摘要:** Tau positron emission tomography (tau-PET) provides an in vivo marker of Alzheimer's disease pathology, but cost and limited availability motivate MRI-based alternatives. We introduce DisQ-HNet (DQH), a framework that synthesizes tau-PET from paired T1-weighted and FLAIR MRI while exposing how each modality contributes to the prediction. The method combines (i) a Partial Information Decomposition (PID)-guided, vector-quantized encoder that partitions latent information into redundant, unique, and complementary components, and (ii) a Half-UNet decoder that preserves anatomical detail using pseudo-skip connections conditioned on structural edge cues rather than direct encoder feature reuse. Across multiple baselines (VAE, VQ-VAE, and UNet), DisQ-HNet maintains reconstruction fidelity and better preserves disease-relevant signal for downstream AD tasks, including Braak staging, tau localization, and classification. PID-based Shapley analysis provides modality-specific attribution of synthesized uptake patterns.
>
---
#### [new 032] UCM: Unifying Camera Control and Memory with Time-aware Positional Encoding Warping for World Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决场景重访一致性与精确相机控制问题。提出UCM框架，通过时间感知位置编码扭曲实现长期记忆与相机控制的统一。**

- **链接: [https://arxiv.org/pdf/2602.22960v1](https://arxiv.org/pdf/2602.22960v1)**

> **作者:** Tianxing Xu; Zixuan Wang; Guangyuan Wang; Li Hu; Zhongyi Zhang; Peng Zhang; Bang Zhang; Song-Hai Zhang
>
> **备注:** Project Page: https://humanaigc.github.io/ucm-webpage/
>
> **摘要:** World models based on video generation demonstrate remarkable potential for simulating interactive environments but face persistent difficulties in two key areas: maintaining long-term content consistency when scenes are revisited and enabling precise camera control from user-provided inputs. Existing methods based on explicit 3D reconstruction often compromise flexibility in unbounded scenarios and fine-grained structures. Alternative methods rely directly on previously generated frames without establishing explicit spatial correspondence, thereby constraining controllability and consistency. To address these limitations, we present UCM, a novel framework that unifies long-term memory and precise camera control via a time-aware positional encoding warping mechanism. To reduce computational overhead, we design an efficient dual-stream diffusion transformer for high-fidelity generation. Moreover, we introduce a scalable data curation strategy utilizing point-cloud-based rendering to simulate scene revisiting, facilitating training on over 500K monocular videos. Extensive experiments on real-world and synthetic benchmarks demonstrate that UCM significantly outperforms state-of-the-art methods in long-term scene consistency, while also achieving precise camera controllability in high-fidelity video generation.
>
---
#### [new 033] Guidance Matters: Rethinking the Evaluation Pitfall for Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有评估方法的偏差问题。研究揭示了引导尺度对评价结果的误导，提出新的评估框架并设计改进方法，但发现多数方法实际效果有限。**

- **链接: [https://arxiv.org/pdf/2602.22570v1](https://arxiv.org/pdf/2602.22570v1)**

> **作者:** Dian Xie; Shitong Shao; Lichen Bai; Zikai Zhou; Bojun Cheng; Shuo Yang; Jun Wu; Zeke Xie
>
> **摘要:** Classifier-free guidance (CFG) has helped diffusion models achieve great conditional generation in various fields. Recently, more diffusion guidance methods have emerged with improved generation quality and human preference. However, can these emerging diffusion guidance methods really achieve solid and significant improvements? In this paper, we rethink recent progress on diffusion guidance. Our work mainly consists of four contributions. First, we reveal a critical evaluation pitfall that common human preference models exhibit a strong bias towards large guidance scales. Simply increasing the CFG scale can easily improve quantitative evaluation scores due to strong semantic alignment, even if image quality is severely damaged (e.g., oversaturation and artifacts). Second, we introduce a novel guidance-aware evaluation (GA-Eval) framework that employs effective guidance scale calibration to enable fair comparison between current guidance methods and CFG by identifying the effects orthogonal and parallel to CFG effects. Third, motivated by the evaluation pitfall, we design Transcendent Diffusion Guidance (TDG) method that can significantly improve human preference scores in the conventional evaluation framework but actually does not work in practice. Fourth, in extensive experiments, we empirically evaluate recent eight diffusion guidance methods within the conventional evaluation framework and the proposed GA-Eval framework. Notably, simply increasing the CFG scales can compete with most studied diffusion guidance methods, while all methods suffer severely from winning rate degradation over standard CFG. Our work would strongly motivate the community to rethink the evaluation paradigm and future directions of this field.
>
---
#### [new 034] PRIMA: Pre-training with Risk-integrated Image-Metadata Alignment for Medical Diagnosis via LLM
- **分类: cs.CV**

- **简介: 该论文属于医学诊断任务，旨在解决视觉与临床数据融合不足的问题。通过整合风险-疾病知识，提出PRIMA框架，提升疾病分类性能。**

- **链接: [https://arxiv.org/pdf/2602.23297v1](https://arxiv.org/pdf/2602.23297v1)**

> **作者:** Yiqing Wang; Chunming He; Ming-Chen Lu; Mercy Pawar; Leslie Niziol; Maria Woodward; Sina Farsiu
>
> **摘要:** Medical diagnosis requires the effective synthesis of visual manifestations and clinical metadata. However, existing methods often treat metadata as isolated tags, failing to exploit the rich semantic knowledge embedded in clinical descriptions. We propose PRIMA (Pre-training with Risk-integrated Image-Metadata Alignment), a framework that integrates domain-specific knowledge into multi-modal representation learning. We first curate an expert corpus of risk-disease correlations via Retrieval-Augmented Generation (RAG) to refine Clinical ModernBERT, embedding diagnostic priors into the text encoder. To bridge the modality gap, we introduce a dual-encoder pre-training strategy utilizing DINOv3 and our refined BERT, optimized by a suite of four complementary loss functions. These losses are designed to capture multi-granular semantic alignment and handle the ambiguity of clinical correlations through soft labels. Finally, we leverage Qwen-3 to fuse these aligned features for precise disease classification. Extensive experiments demonstrate that PRIMA effectively harmonizes pixel-level features with abstract clinical expertise, significantly outperforming other state-of-the-art methods. Notably, our framework achieves superior robustness without the need for massive data collection or exhaustive computational resources. Our code will be made public upon acceptance.
>
---
#### [new 035] Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出LaGS方法，用于4D全景占据跟踪任务，解决动态环境中同时获取精确几何与时间关联的问题。通过融合多视角信息到3D体素网格，实现高效场景理解。**

- **链接: [https://arxiv.org/pdf/2602.23172v1](https://arxiv.org/pdf/2602.23172v1)**

> **作者:** Maximilian Luz; Rohit Mohan; Thomas Nürnberg; Yakov Miron; Daniele Cattaneo; Abhinav Valada
>
> **摘要:** Capturing 4D spatiotemporal surroundings is crucial for the safe and reliable operation of robots in dynamic environments. However, most existing methods address only one side of the problem: they either provide coarse geometric tracking via bounding boxes, or detailed 3D structures like voxel-based occupancy that lack explicit temporal association. In this work, we present Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking (LaGS) that advances spatiotemporal scene understanding in a holistic direction. Our approach incorporates camera-based end-to-end tracking with mask-based multi-view panoptic occupancy prediction, and addresses the key challenge of efficiently aggregating multi-view information into 3D voxel grids via a novel latent Gaussian splatting approach. Specifically, we first fuse observations into 3D Gaussians that serve as a sparse point-centric latent representation of the 3D scene, and then splat the aggregated features onto a 3D voxel grid that is decoded by a mask-based segmentation head. We evaluate LaGS on the Occ3D nuScenes and Waymo datasets, achieving state-of-the-art performance for 4D panoptic occupancy tracking. We make our code available at https://lags.cs.uni-freiburg.de/.
>
---
#### [new 036] BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D场景合成任务，旨在提升多样真实场景的新型视图合成质量。通过改进扩散模型的潜在空间，结合3D高斯泼溅技术，解决细节不一致和伪影问题。**

- **链接: [https://arxiv.org/pdf/2602.22596v1](https://arxiv.org/pdf/2602.22596v1)**

> **作者:** Yuci Han; Charles Toth; John E. Anderson; William J. Shuart; Alper Yilmaz
>
> **摘要:** We present BetterScene, an approach to enhance novel view synthesis (NVS) quality for diverse real-world scenes using extremely sparse, unconstrained photos. BetterScene leverages the production-ready Stable Video Diffusion (SVD) model pretrained on billions of frames as a strong backbone, aiming to mitigate artifacts and recover view-consistent details at inference time. Conventional methods have developed similar diffusion-based solutions to address these challenges of novel view synthesis. Despite significant improvements, these methods typically rely on off-the-shelf pretrained diffusion priors and fine-tune only the UNet module while keeping other components frozen, which still leads to inconsistent details and artifacts even when incorporating geometry-aware regularizations like depth or semantic conditions. To address this, we investigate the latent space of the diffusion model and introduce two components: (1) temporal equivariance regularization and (2) vision foundation model-aligned representation, both applied to the variational autoencoder (VAE) module within the SVD pipeline. BetterScene integrates a feed-forward 3D Gaussian Splatting (3DGS) model to render features as inputs for the SVD enhancer and generate continuous, artifact-free, consistent novel views. We evaluate on the challenging DL3DV-10K dataset and demonstrate superior performance compared to state-of-the-art methods.
>
---
#### [new 037] SUPERGLASSES: Benchmarking Vision Language Models as Intelligent Agents for AI Smart Glasses
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉问答任务，旨在解决智能眼镜场景下VQA模型性能不足的问题。通过构建真实数据集SUPERGLASSES并提出SUPERLENS模型提升效果。**

- **链接: [https://arxiv.org/pdf/2602.22683v1](https://arxiv.org/pdf/2602.22683v1)**

> **作者:** Zhuohang Jiang; Xu Yuan; Haohao Qu; Shanru Lin; Kanglong Liu; Wenqi Fan; Qing Li
>
> **摘要:** The rapid advancement of AI-powered smart glasses, one of the hottest wearable devices, has unlocked new frontiers for multimodal interaction, with Visual Question Answering (VQA) over external knowledge sources emerging as a core application. Existing Vision Language Models (VLMs) adapted to smart glasses are typically trained and evaluated on traditional multimodal datasets; however, these datasets lack the variety and realism needed to reflect smart glasses usage scenarios and diverge from their specific challenges, where accurately identifying the object of interest must precede any external knowledge retrieval. To bridge this gap, we introduce SUPERGLASSES, the first comprehensive VQA benchmark built on real-world data entirely collected by smart glasses devices. SUPERGLASSES comprises 2,422 egocentric image-question pairs spanning 14 image domains and 8 query categories, enriched with full search trajectories and reasoning annotations. We evaluate 26 representative VLMs on this benchmark, revealing significant performance gaps. To address the limitations of existing models, we further propose SUPERLENS, a multimodal smart glasses agent that enables retrieval-augmented answer generation by integrating automatic object detection, query decoupling, and multimodal web search. Our agent achieves state-of-the-art performance, surpassing GPT-4o by 2.19 percent, and highlights the need for task-specific solutions in smart glasses VQA scenarios.
>
---
#### [new 038] TrajTok: Learning Trajectory Tokens enables better Video Understanding
- **分类: cs.CV**

- **简介: 该论文提出TrajTok，解决视频模型中token冗余问题，通过动态轨迹分割提升视频理解效率与性能。属于视频理解任务。**

- **链接: [https://arxiv.org/pdf/2602.22779v1](https://arxiv.org/pdf/2602.22779v1)**

> **作者:** Chenhao Zheng; Jieyu Zhang; Jianing Zhang; Weikai Huang; Ashutosh Kumar; Quan Kong; Oncel Tuzel; Chun-Liang Li; Ranjay Krishna
>
> **备注:** CVPR 2026
>
> **摘要:** Tokenization in video models, typically through patchification, generates an excessive and redundant number of tokens. This severely limits video efficiency and scalability. While recent trajectory-based tokenizers offer a promising solution by decoupling video duration from token count, they rely on complex external segmentation and tracking pipelines that are slow and task-agnostic. We propose TrajTok, an end-to-end video tokenizer module that is fully integrated and co-trained with video models for a downstream objective, dynamically adapting its token granularity to semantic complexity, independent of video duration. TrajTok contains a unified segmenter that performs implicit clustering over pixels in both space and time to directly produce object trajectories in a single forward pass. By prioritizing downstream adaptability over pixel-perfect segmentation fidelity, TrajTok is lightweight and efficient, yet empirically improves video understanding performance. With TrajTok, we implement a video CLIP model trained from scratch (TrajViT2). It achieves the best accuracy at scale across both classification and retrieval benchmarks, while maintaining efficiency comparable to the best token-merging methods. TrajTok also proves to be a versatile component beyond its role as a tokenizer. We show that it can be seamlessly integrated as either a probing head for pretrained visual features (TrajAdapter) or an alignment connector in vision-language models (TrajVLM) with especially strong performance in long-video reasoning.
>
---
#### [new 039] PackUV: Packed Gaussian UV Maps for 4D Volumetric Video
- **分类: cs.CV**

- **简介: 该论文提出PackUV，解决4D体积视频存储与流传输问题，通过结构化UV图集实现高效编码和兼容传统编码器。**

- **链接: [https://arxiv.org/pdf/2602.23040v1](https://arxiv.org/pdf/2602.23040v1)**

> **作者:** Aashish Rai; Angela Xing; Anushka Agarwal; Xiaoyan Cong; Zekun Li; Tao Lu; Aayush Prakash; Srinath Sridhar
>
> **备注:** https://ivl.cs.brown.edu/packuv
>
> **摘要:** Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications. We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure. To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.
>
---
#### [new 040] Velocity and stroke rate reconstruction of canoe sprint team boats based on panned and zoomed video recordings
- **分类: cs.CV**

- **简介: 该论文属于运动分析任务，旨在通过视频重建皮划艇的速度和划频，解决无GPS时的性能评估问题。采用YOLOv8和U-net等方法实现自动检测与跟踪，提升训练反馈精度。**

- **链接: [https://arxiv.org/pdf/2602.22941v1](https://arxiv.org/pdf/2602.22941v1)**

> **作者:** Julian Ziegler; Daniel Matthes; Finn Gerdts; Patrick Frenzel; Torsten Warnke; Matthias Englert; Tina Koevari; Mirco Fuchs
>
> **摘要:** Pacing strategies, defined by velocity and stroke rate profiles, are essential for peak performance in canoe sprint. While GPS is the gold standard for analysis, its limited availability necessitates automated video-based solutions. This paper presents an extended framework for reconstructing performance metrics from panned and zoomed video recordings across all sprint disciplines (K1-K4, C1-C2) and distances (200m-500m). Our method utilizes YOLOv8 for buoy and athlete detection, leveraging the known buoy grid to estimate homographies. We generalized the estimation of the boat position by means of learning a boat-specific athlete offset using a U-net based boat tip calibration. Further, we implement a robust tracking scheme using optical flow to adapt to multi-athlete boat types. Finally, we introduce methods to extract stroke rate information from either pose estimations or the athlete bounding boxes themselves. Evaluation against GPS data from elite competitions yields a velocity RRMSE of 0.020 +- 0.011 (rho = 0.956) and a stroke rate RRMSE of 0.022 +- 0.024 (rho = 0.932). The methods provide coaches with highly accurate, automated feedback without requiring on-boat sensors or manual annotation.
>
---
#### [new 041] QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition
- **分类: cs.CV; math.NA; math.OC**

- **简介: 该论文属于视觉重建任务，旨在解决多相机同步问题。通过 Tucker 分解提出 QuadSync 算法，有效利用四焦点张量信息，实现相机参数的同步恢复。**

- **链接: [https://arxiv.org/pdf/2602.22639v1](https://arxiv.org/pdf/2602.22639v1)**

> **作者:** Daniel Miao; Gilad Lerman; Joe Kileel
>
> **备注:** 30 pages, accepted to CVPR 2026
>
> **摘要:** In structure from motion, quadrifocal tensors capture more information than their pairwise counterparts (essential matrices), yet they have often been thought of as impractical and only of theoretical interest. In this work, we challenge such beliefs by providing a new framework to recover $n$ cameras from the corresponding collection of quadrifocal tensors. We form the block quadrifocal tensor and show that it admits a Tucker decomposition whose factor matrices are the stacked camera matrices, and which thus has a multilinear rank of (4,~4,~4,~4) independent of $n$. We develop the first synchronization algorithm for quadrifocal tensors, using Tucker decomposition, alternating direction method of multipliers, and iteratively reweighted least squares. We further establish relationships between the block quadrifocal, trifocal, and bifocal tensors, and introduce an algorithm that jointly synchronizes these three entities. Numerical experiments demonstrate the effectiveness of our methods on modern datasets, indicating the potential and importance of using higher-order information in synchronization.
>
---
#### [new 042] DrivePTS: A Progressive Learning Framework with Textual and Structural Enhancement for Driving Scene Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于驾驶场景生成任务，解决现有方法在场景细节和可控性上的不足。提出DrivePTS框架，通过渐进学习、多视角文本引导和频率结构损失提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2602.22549v1](https://arxiv.org/pdf/2602.22549v1)**

> **作者:** Zhechao Wang; Yiming Zeng; Lufan Ma; Zeqing Fu; Chen Bai; Ziyao Lin; Cheng Lu
>
> **摘要:** Synthesis of diverse driving scenes serves as a crucial data augmentation technique for validating the robustness and generalizability of autonomous driving systems. Current methods aggregate high-definition (HD) maps and 3D bounding boxes as geometric conditions in diffusion models for conditional scene generation. However, implicit inter-condition dependency causes generation failures when control conditions change independently. Additionally, these methods suffer from insufficient details in both semantic and structural aspects. Specifically, brief and view-invariant captions restrict semantic contexts, resulting in weak background modeling. Meanwhile, the standard denoising loss with uniform spatial weighting neglects foreground structural details, causing visual distortions and blurriness. To address these challenges, we propose DrivePTS, which incorporates three key innovations. Firstly, our framework adopts a progressive learning strategy to mitigate inter-dependency between geometric conditions, reinforced by an explicit mutual information constraint. Secondly, a Vision-Language Model is utilized to generate multi-view hierarchical descriptions across six semantic aspects, providing fine-grained textual guidance. Thirdly, a frequency-guided structure loss is introduced to strengthen the model's sensitivity to high-frequency elements, improving foreground structural fidelity. Extensive experiments demonstrate that our DrivePTS achieves state-of-the-art fidelity and controllability in generating diverse driving scenes. Notably, DrivePTS successfully generates rare scenes where prior methods fail, highlighting its strong generalization ability.
>
---
#### [new 043] DyaDiT: A Multi-Modal Diffusion Transformer for Socially Favorable Dyadic Gesture Generation
- **分类: cs.CV**

- **简介: 该论文属于对话动作生成任务，解决单向音频到动作映射的不足，通过多模态扩散Transformer生成符合社交情境的双人手势。**

- **链接: [https://arxiv.org/pdf/2602.23165v1](https://arxiv.org/pdf/2602.23165v1)**

> **作者:** Yichen Peng; Jyun-Ting Song; Siyeol Jung; Ruofan Liu; Haiyang Liu; Xuangeng Chu; Ruicong Liu; Erwin Wu; Hideki Koike; Kris Kitani
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Generating realistic conversational gestures are essential for achieving natural, socially engaging interactions with digital humans. However, existing methods typically map a single audio stream to a single speaker's motion, without considering social context or modeling the mutual dynamics between two people engaging in conversation. We present DyaDiT, a multi-modal diffusion transformer that generates contextually appropriate human motion from dyadic audio signals. Trained on Seamless Interaction Dataset, DyaDiT takes dyadic audio with optional social-context tokens to produce context-appropriate motion. It fuses information from both speakers to capture interaction dynamics, uses a motion dictionary to encode motion priors, and can optionally utilize the conversational partner's gestures to produce more responsive motion. We evaluate DyaDiT on standard motion generation metrics and conduct quantitative user studies, demonstrating that it not only surpasses existing methods on objective metrics but is also strongly preferred by users, highlighting its robustness and socially favorable motion generation. Code and models will be released upon acceptance.
>
---
#### [new 044] SpectralMamba-UNet: Frequency-Disentangled State Space Modeling for Texture-Structure Consistent Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升结构与纹理的一致性。提出SpectralMamba-UNet，通过频域分解和注意力机制增强细粒度边界建模。**

- **链接: [https://arxiv.org/pdf/2602.23103v1](https://arxiv.org/pdf/2602.23103v1)**

> **作者:** Fuhao Zhang; Lei Liu; Jialin Zhang; Ya-Nan Zhang; Nan Mu
>
> **摘要:** Accurate medical image segmentation requires effective modeling of both global anatomical structures and fine-grained boundary details. Recent state space models (e.g., Vision Mamba) offer efficient long-range dependency modeling. However, their one-dimensional serialization weakens local spatial continuity and high-frequency representation. To this end, we propose SpectralMamba-UNet, a novel frequency-disentangled framework to decouple the learning of structural and textural information in the spectral domain. Our Spectral Decomposition and Modeling (SDM) module applies discrete cosine transform to decompose low- and high-frequency features, where low frequency contributes to global contextual modeling via a frequency-domain Mamba and high frequency preserves boundary-sensitive details. To balance spectral contributions, we introduce a Spectral Channel Reweighting (SCR) mechanism to form channel-wise frequency-aware attention, and a Spectral-Guided Fusion (SGF) module to achieve adaptively multi-scale fusion in the decoder. Experiments on five public benchmarks demonstrate consistent improvements across diverse modalities and segmentation targets, validating the effectiveness and generalizability of our approach.
>
---
#### [new 045] WARM-CAT: : Warm-Started Test-Time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于组合零样本学习任务，旨在解决测试时标签空间分布偏移导致的性能下降问题。通过多模态知识积累和自适应更新机制提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.23114v1](https://arxiv.org/pdf/2602.23114v1)**

> **作者:** Xudong Yan; Songhe Feng; Jiaxin Wang; Xin Su; Yi Jin
>
> **摘要:** Compositional Zero-Shot Learning (CZSL) aims to recognize novel attribute-object compositions based on the knowledge learned from seen ones. Existing methods suffer from performance degradation caused by the distribution shift of label space at test time, which stems from the inclusion of unseen compositions recombined from attributes and objects. To overcome the challenge, we propose a novel approach that accumulates comprehensive knowledge in both textual and visual modalities from unsupervised data to update multimodal prototypes at test time. Building on this, we further design an adaptive update weight to control the degree of prototype adjustment, enabling the model to flexibly adapt to distribution shift during testing. Moreover, a dynamic priority queue is introduced that stores high-confidence images to acquire visual prototypes from historical images for inference. Since the model tends to favor compositions already stored in the queue during testing, we warm-start the queue by initializing it with training images for visual prototypes of seen compositions and generating unseen visual prototypes using the mapping learned between seen and unseen textual prototypes. Considering the semantic consistency of multimodal knowledge, we align textual and visual prototypes by multimodal collaborative representation learning. To provide a more reliable evaluation for CZSL, we introduce a new benchmark dataset, C-Fashion, and refine the widely used but noisy MIT-States dataset. Extensive experiments indicate that our approach achieves state-of-the-art performance on four benchmark datasets under both closed-world and open-world settings. The source code and datasets are available at https://github.com/xud-yan/WARM-CAT .
>
---
#### [new 046] SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，旨在解决深度误差和多视角不一致问题。通过神经深度校正生成一致深度图，提升重建效率与质量。**

- **链接: [https://arxiv.org/pdf/2602.22565v1](https://arxiv.org/pdf/2602.22565v1)**

> **作者:** Kang Han; Wei Xiang; Lu Yu; Mathew Wyatt; Gaowen Liu; Ramana Rao Kompella
>
> **摘要:** Depth-guided 3D reconstruction has gained popularity as a fast alternative to optimization-heavy approaches, yet existing methods still suffer from scale drift, multi-view inconsistencies, and the need for substantial refinement to achieve high-fidelity geometry. Here, we propose SwiftNDC, a fast and general framework built around a Neural Depth Correction field that produces cross-view consistent depth maps. From these refined depths, we generate a dense point cloud through back-projection and robust reprojection-error filtering, obtaining a clean and uniformly distributed geometric initialization for downstream reconstruction. This reliable dense geometry substantially accelerates 3D Gaussian Splatting (3DGS) for mesh reconstruction, enabling high-quality surfaces with significantly fewer optimization iterations. For novel-view synthesis, SwiftNDC can also improve 3DGS rendering quality, highlighting the benefits of strong geometric initialization. We conduct a comprehensive study across five datasets, including two for mesh reconstruction, as well as three for novel-view synthesis. SwiftNDC consistently reduces running time for accurate mesh reconstruction and boosts rendering fidelity for view synthesis, demonstrating the effectiveness of combining neural depth refinement with robust geometric initialization for high-fidelity and efficient 3D reconstruction.
>
---
#### [new 047] Enhancing Renal Tumor Malignancy Prediction: Deep Learning with Automatic 3D CT Organ Focused Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于肾肿瘤恶性预测任务，旨在解决传统方法依赖手动分割的问题。提出一种无需分割的深度学习框架，利用器官聚焦注意力机制提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2602.22381v1](https://arxiv.org/pdf/2602.22381v1)**

> **作者:** Zhengkang Fan; Chengkun Sun; Russell Terry; Jie Xu; Longin Jan Latecki
>
> **备注:** 5 pages, 2 figures, Accepted at IEEE ISBI 2026
>
> **摘要:** Accurate prediction of malignancy in renal tumors is crucial for informing clinical decisions and optimizing treatment strategies. However, existing imaging modalities lack the necessary accuracy to reliably predict malignancy before surgical intervention. While deep learning has shown promise in malignancy prediction using 3D CT images, traditional approaches often rely on manual segmentation to isolate the tumor region and reduce noise, which enhances predictive performance. Manual segmentation, however, is labor-intensive, costly, and dependent on expert knowledge. In this study, a deep learning framework was developed utilizing an Organ Focused Attention (OFA) loss function to modify the attention of image patches so that organ patches attend only to other organ patches. Hence, no segmentation of 3D renal CT images is required at deployment time for malignancy prediction. The proposed framework achieved an AUC of 0.685 and an F1-score of 0.872 on a private dataset from the UF Integrated Data Repository (IDR), and an AUC of 0.760 and an F1-score of 0.852 on the publicly available KiTS21 dataset. These results surpass the performance of conventional models that rely on segmentation-based cropping for noise reduction, demonstrating the frameworks ability to enhance predictive accuracy without explicit segmentation input. The findings suggest that this approach offers a more efficient and reliable method for malignancy prediction, thereby enhancing clinical decision-making in renal cancer diagnosis.
>
---
#### [new 048] Towards Long-Form Spatio-Temporal Video Grounding
- **分类: cs.CV**

- **简介: 该论文属于长视频时空定位任务，解决长视频中目标定位困难的问题。提出ART-STVG模型，通过递归处理和记忆选择提升性能。**

- **链接: [https://arxiv.org/pdf/2602.23294v1](https://arxiv.org/pdf/2602.23294v1)**

> **作者:** Xin Gu; Bing Fan; Jiali Yao; Zhipeng Zhang; Yan Huang; Cheng Han; Heng Fan; Libo Zhang
>
> **摘要:** In real scenarios, videos can span several minutes or even hours. However, existing research on spatio-temporal video grounding (STVG), given a textual query, mainly focuses on localizing targets in short videos of tens of seconds, typically less than one minute, which limits real-world applications. In this paper, we explore Long-Form STVG (LF-STVG), which aims to locate targets in long-term videos. Compared with short videos, long-term videos contain much longer temporal spans and more irrelevant information, making it difficult for existing STVG methods that process all frames at once. To address this challenge, we propose an AutoRegressive Transformer architecture for LF-STVG, termed ART-STVG. Unlike conventional STVG methods that require the entire video sequence to make predictions at once, ART-STVG treats the video as streaming input and processes frames sequentially, enabling efficient handling of long videos. To model spatio-temporal context, we design spatial and temporal memory banks and apply them to the decoders. Since memories from different moments are not always relevant to the current frame, we introduce simple yet effective memory selection strategies to provide more relevant information to the decoders, significantly improving performance. Furthermore, instead of parallel spatial and temporal localization, we propose a cascaded spatio-temporal design that connects the spatial decoder to the temporal decoder, allowing fine-grained spatial cues to assist complex temporal localization in long videos. Experiments on newly extended LF-STVG datasets show that ART-STVG significantly outperforms state-of-the-art methods, while achieving competitive performance on conventional short-form STVG.
>
---
#### [new 049] UFO-DETR: Frequency-Guided End-to-End Detector for UAV Tiny Objects
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决无人机图像中小目标检测难题。提出UFO-DETR框架，优化检测性能与计算效率。**

- **链接: [https://arxiv.org/pdf/2602.22712v1](https://arxiv.org/pdf/2602.22712v1)**

> **作者:** Yuankai Chen; Kai Lin; Qihong Wu; Xinxuan Yang; Jiashuo Lai; Ruoen Chen; Haonan Shi; Minfan He; Meihua Wang
>
> **备注:** 6 pages, 6 figures, published to 2026 International Conference on Computer Supported Cooperative Work in Design
>
> **摘要:** Small target detection in UAV imagery faces significant challenges such as scale variations, dense distribution, and the dominance of small targets. Existing algorithms rely on manually designed components, and general-purpose detectors are not optimized for UAV images, making it difficult to balance accuracy and complexity. To address these challenges, this paper proposes an end-to-end object detection framework, UFO-DETR, which integrates an LSKNet-based backbone network to optimize the receptive field and reduce the number of parameters. By combining the DAttention and AIFI modules, the model flexibly models multi-scale spatial relationships, improving multi-scale target detection performance. Additionally, the DynFreq-C3 module is proposed to enhance small target detection capability through cross-space frequency feature enhancement. Experimental results show that, compared to RT-DETR-L, the proposed method offers significant advantages in both detection performance and computational efficiency, providing an efficient solution for UAV edge computing.
>
---
#### [new 050] MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MovieTeller，解决长视频摘要生成中角色一致性和叙事连贯性问题，通过工具增强和渐进抽象方法提升准确性。**

- **链接: [https://arxiv.org/pdf/2602.23228v1](https://arxiv.org/pdf/2602.23228v1)**

> **作者:** Yizhi Li; Xiaohan Chen; Miao Jiang; Wentao Tang; Gaoang Wang
>
> **备注:** 6 pages, CSCWD 2026
>
> **摘要:** With the explosive growth of digital entertainment, automated video summarization has become indispensable for applications such as content indexing, personalized recommendation, and efficient media archiving. Automatic synopsis generation for long-form videos, such as movies and TV series, presents a significant challenge for existing Vision-Language Models (VLMs). While proficient at single-image captioning, these general-purpose models often exhibit critical failures in long-duration contexts, primarily a lack of ID-consistent character identification and a fractured narrative coherence. To overcome these limitations, we propose MovieTeller, a novel framework for generating movie synopses via tool-augmented progressive abstraction. Our core contribution is a training-free, tool-augmented, fact-grounded generation process. Instead of requiring costly model fine-tuning, our framework directly leverages off-the-shelf models in a plug-and-play manner. We first invoke a specialized face recognition model as an external "tool" to establish Factual Groundings--precise character identities and their corresponding bounding boxes. These groundings are then injected into the prompt to steer the VLM's reasoning, ensuring the generated scene descriptions are anchored to verifiable facts. Furthermore, our progressive abstraction pipeline decomposes the summarization of a full-length movie into a multi-stage process, effectively mitigating the context length limitations of current VLMs. Experiments demonstrate that our approach yields significant improvements in factual accuracy, character consistency, and overall narrative coherence compared to end-to-end baselines.
>
---
#### [new 051] Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于医学图像重建任务，解决PnP扩散框架的稳态偏差与幻觉问题，提出双变量耦合和频谱同质化方法，提升重建精度与速度。**

- **链接: [https://arxiv.org/pdf/2602.23214v1](https://arxiv.org/pdf/2602.23214v1)**

> **作者:** Chenhe Du; Xuanyu Tian; Qing Wu; Muyu Liu; Jingyi Yu; Hongjiang Wei; Yuyao Zhang
>
> **摘要:** Plug-and-Play diffusion prior (PnPDP) frameworks have emerged as a powerful paradigm for solving imaging inverse problems by treating pretrained generative models as modular priors. However, we identify a critical flaw in prevailing PnP solvers (e.g., based on HQS or Proximal Gradient): they function as memoryless operators, updating estimates solely based on instantaneous gradients. This lack of historical tracking inevitably leads to non-vanishing steady-state bias, where the reconstruction fails to strictly satisfy physical measurements under heavy corruption. To resolve this, we propose Dual-Coupled PnP Diffusion, which restores the classical dual variable to provide integral feedback, theoretically guaranteeing asymptotic convergence to the exact data manifold. However, this rigorous geometric coupling introduces a secondary challenge: the accumulated dual residuals exhibit spectrally colored, structured artifacts that violate the Additive White Gaussian Noise (AWGN) assumption of diffusion priors, causing severe hallucinations. To bridge this gap, we introduce Spectral Homogenization (SH), a frequency-domain adaptation mechanism that modulates these structured residuals into statistically compliant pseudo-AWGN inputs. This effectively aligns the solver's rigorous optimization trajectory with the denoiser's valid statistical manifold. Extensive experiments on CT and MRI reconstruction demonstrate that our approach resolves the bias-hallucination trade-off, achieving state-of-the-art fidelity with significantly accelerated convergence.
>
---
#### [new 052] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于单视角无人机视频的4D动态重建任务，解决空中场景中深度模糊和运动估计不稳定的问题。提出AeroDGS框架，结合物理先验提升重建精度。**

- **链接: [https://arxiv.org/pdf/2602.22376v1](https://arxiv.org/pdf/2602.22376v1)**

> **作者:** Hanyang Liu; Rongjun Qin
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent advances in 4D scene reconstruction have significantly improved dynamic modeling across various domains. However, existing approaches remain limited under aerial conditions with single-view capture, wide spatial range, and dynamic objects of limited spatial footprint and large motion disparity. These challenges cause severe depth ambiguity and unstable motion estimation, making monocular aerial reconstruction inherently ill-posed. To this end, we present AeroDGS, a physics-guided 4D Gaussian splatting framework for monocular UAV videos. AeroDGS introduces a Monocular Geometry Lifting module that reconstructs reliable static and dynamic geometry from a single aerial sequence, providing a robust basis for dynamic estimation. To further resolve monocular ambiguity, we propose a Physics-Guided Optimization module that incorporates differentiable ground-support, upright-stability, and trajectory-smoothness priors, transforming ambiguous image cues into physically consistent motion. The framework jointly refines static backgrounds and dynamic entities with stable geometry and coherent temporal evolution. We additionally build a real-world UAV dataset that spans various altitudes and motion conditions to evaluate dynamic aerial reconstruction. Experiments on synthetic and real UAV scenes demonstrate that AeroDGS outperforms state-of-the-art methods, achieving superior reconstruction fidelity in dynamic aerial environments.
>
---
#### [new 053] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于图像对齐任务，旨在解决传统方法在遮挡和光照变化下的性能问题。提出DMAligner框架，通过扩散模型生成新视角实现更准确的对齐。**

- **链接: [https://arxiv.org/pdf/2602.23022v1](https://arxiv.org/pdf/2602.23022v1)**

> **作者:** Xinglong Luo; Ao Luo; Zhengning Wang; Yueqi Yang; Chaoyu Feng; Lei Lei; Bing Zeng; Shuaicheng Liu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at https://github.com/boomluo02/DMAligner.
>
---
#### [new 054] WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文提出WISER框架，解决零样本组合图像检索问题。通过融合文本到图像和图像到图像检索，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2602.23029v1](https://arxiv.org/pdf/2602.23029v1)**

> **作者:** Tianyue Wang; Leigang Qu; Tianyu Yang; Xiangzhao Hao; Yifan Xu; Haiyun Guo; Jinqiao Wang
>
> **摘要:** Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images given a multimodal query (comprising a reference image and a modification text), without training on annotated triplets. Existing methods typically convert the multimodal query into a single modality-either as an edited caption for Text-to-Image retrieval (T2I) or as an edited image for Image-to-Image retrieval (I2I). However, each paradigm has inherent limitations: T2I often loses fine-grained visual details, while I2I struggles with complex semantic modifications. To effectively leverage their complementary strengths under diverse query intents, we propose WISER, a training-free framework that unifies T2I and I2I via a "retrieve-verify-refine" pipeline, explicitly modeling intent awareness and uncertainty awareness. Specifically, WISER first performs Wider Search by generating both edited captions and images for parallel retrieval to broaden the candidate pool. Then, it conducts Adaptive Fusion with a verifier to assess retrieval confidence, triggering refinement for uncertain retrievals, and dynamically fusing the dual-path for reliable ones. For uncertain retrievals, WISER generates refinement suggestions through structured self-reflection to guide the next retrieval round toward Deeper Thinking. Extensive experiments demonstrate that WISER significantly outperforms previous methods across multiple benchmarks, achieving relative improvements of 45% on CIRCO (mAP@5) and 57% on CIRR (Recall@1) over existing training-free methods. Notably, it even surpasses many training-dependent methods, highlighting its superiority and generalization under diverse scenarios. Code will be released at https://github.com/Physicsmile/WISER.
>
---
#### [new 055] FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time
- **分类: cs.CV; cs.CG; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决单目视频中相机运动估计的问题。提出基于斐波那契格网的霍夫变换方法，提升在噪声和异常值下的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.23115v1](https://arxiv.org/pdf/2602.23115v1)**

> **作者:** David Dirnfeld; Fabien Delattre; Pedro Miraldo; Erik Learned-Miller
>
> **摘要:** Estimating camera motion from monocular video is a fundamental problem in computer vision, central to tasks such as SLAM, visual odometry, and structure-from-motion. Existing methods that recover the camera's heading under known rotation, whether from an IMU or an optimization algorithm, tend to perform well in low-noise, low-outlier conditions, but often decrease in accuracy or become computationally expensive as noise and outlier levels increase. To address these limitations, we propose a novel generalization of the Hough transform on the unit sphere (S(2)) to estimate the camera's heading. First, the method extracts correspondences between two frames and generates a great circle of directions compatible with each pair of correspondences. Then, by discretizing the unit sphere using a Fibonacci lattice as bin centers, each great circle casts votes for a range of directions, ensuring that features unaffected by noise or dynamic objects vote consistently for the correct motion direction. Experimental results on three datasets demonstrate that the proposed method is on the Pareto frontier of accuracy versus efficiency. Additionally, experiments on SLAM show that the proposed method reduces RMSE by correcting the heading during camera pose initialization.
>
---
#### [new 056] GFRRN: Explore the Gaps in Single Image Reflection Removal
- **分类: cs.CV**

- **简介: 该论文属于单图像反射去除任务，解决预训练模型与反射去除模型间的语义差异及标签不一致问题。通过参数高效微调、标签生成和自适应频率学习等方法，提出GFRRN模型。**

- **链接: [https://arxiv.org/pdf/2602.22695v1](https://arxiv.org/pdf/2602.22695v1)**

> **作者:** Yu Chen; Zewei He; Xingyu Liu; Zixuan Chen; Zheming Lu
>
> **备注:** CVPR26
>
> **摘要:** Prior dual-stream methods with the feature interaction mechanism have achieved remarkable performance in single image reflection removal (SIRR). However, they often struggle with (1) semantic understanding gap between the features of pre-trained models and those of reflection removal models, and (2) reflection label inconsistencies between synthetic and real-world training data. In this work, we first adopt the parameter efficient fine-tuning (PEFT) strategy by integrating several learnable Mona layers into the pre-trained model to align the training directions. Then, a label generator is designed to unify the reflection labels for both synthetic and real-world data. In addition, a Gaussian-based Adaptive Frequency Learning Block (G-AFLB) is proposed to adaptively learn and fuse the frequency priors, and a Dynamic Agent Attention (DAA) is employed as an alternative to window-based attention by dynamically modeling the significance levels across windows (inter-) and within an individual window (intra-). These components constitute our proposed Gap-Free Reflection Removal Network (GFRRN). Extensive experiments demonstrate the effectiveness of our GFRRN, achieving superior performance against state-of-the-art SIRR methods.
>
---
#### [new 057] Face Time Traveller : Travel Through Ages Without Losing Identity
- **分类: cs.CV**

- **简介: 该论文属于人脸年龄变换任务，旨在解决身份保持与视觉真实性的难题。提出FaceTT框架，结合属性感知提示、无调优反演和自适应注意力控制，提升变换效果。**

- **链接: [https://arxiv.org/pdf/2602.22819v1](https://arxiv.org/pdf/2602.22819v1)**

> **作者:** Purbayan Kar; Ayush Ghadiya; Vishal Chudasama; Pankaj Wasnik; C. V. Jawahar
>
> **备注:** Accepted at CVPR 2026 (Findings Track)
>
> **摘要:** Face aging, an ill-posed problem shaped by environmental and genetic factors, is vital in entertainment, forensics, and digital archiving, where realistic age transformations must preserve both identity and visual realism. However, existing works relying on numerical age representations overlook the interplay of biological and contextual cues. Despite progress in recent face aging models, they struggle with identity preservation in wide age transformations, also static attention and optimization-heavy inversion in diffusion limit adaptability, fine-grained control and background consistency. To address these challenges, we propose Face Time Traveller (FaceTT), a diffusion-based framework that achieves high-fidelity, identity-consistent age transformation. Here, we introduce a Face-Attribute-Aware Prompt Refinement strategy that encodes intrinsic (biological) and extrinsic (environmental) aging cues for context-aware conditioning. A tuning-free Angular Inversion method is proposed that efficiently maps real faces into the diffusion latent space for fast and accurate reconstruction. Moreover, an Adaptive Attention Control mechanism is introduced that dynamically balances cross-attention for semantic aging cues and self-attention for structural and identity preservation. Extensive experiments on benchmark datasets and in-the-wild testset demonstrate that FaceTT achieves superior identity retention, background preservation and aging realism over state-of-the-art (SOTA) methods.
>
---
#### [new 058] Decomposing Private Image Generation via Coarse-to-Fine Wavelet Modeling
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于图像生成任务，解决隐私泄露问题。通过分阶段的波浪模型，先保护低频结构，再用公开模型提升细节，平衡隐私与质量。**

- **链接: [https://arxiv.org/pdf/2602.23262v1](https://arxiv.org/pdf/2602.23262v1)**

> **作者:** Jasmine Bayrooti; Weiwei Kong; Natalia Ponomareva; Carlos Esteves; Ameesh Makadia; Amanda Prorok
>
> **摘要:** Generative models trained on sensitive image datasets risk memorizing and reproducing individual training examples, making strong privacy guarantees essential. While differential privacy (DP) provides a principled framework for such guarantees, standard DP finetuning (e.g., with DP-SGD) often results in severe degradation of image quality, particularly in high-frequency textures, due to the indiscriminate addition of noise across all model parameters. In this work, we propose a spectral DP framework based on the hypothesis that the most privacy-sensitive portions of an image are often low-frequency components in the wavelet space (e.g., facial features and object shapes) while high-frequency components are largely generic and public. Based on this hypothesis, we propose the following two-stage framework for DP image generation with coarse image intermediaries: (1) DP finetune an autoregressive spectral image tokenizer model on the low-resolution wavelet coefficients of the sensitive images, and (2) perform high-resolution upsampling using a publicly pretrained super-resolution model. By restricting the privacy budget to the global structures of the image in the first stage, and leveraging the post-processing property of DP for detail refinement, we achieve promising trade-offs between privacy and utility. Experiments on the MS-COCO and MM-CelebA-HQ datasets show that our method generates images with improved quality and style capture relative to other leading DP image frameworks.
>
---
#### [new 059] Chain of Flow: A Foundational Generative Framework for ECG-to-4D Cardiac Digital Twins
- **分类: cs.CV**

- **简介: 该论文提出COF框架，用于从ECG生成4D心脏数字孪生，解决传统方法仅限于特定任务的问题，实现个性化、可操作的虚拟心脏重建。**

- **链接: [https://arxiv.org/pdf/2602.22919v1](https://arxiv.org/pdf/2602.22919v1)**

> **作者:** Haofan Wu; Nay Aung; Theodoros N. Arvanitis; Joao A. C. Lima; Steffen E. Petersen; Le Zhang
>
> **备注:** 10 pages, 8 figures. Submitted to IEEE Transactions on Medical Imaging (TMI). Code will be released after review
>
> **摘要:** A clinically actionable Cardiac Digital Twin (CDT) should reconstruct individualised cardiac anatomy and physiology, update its internal state from multimodal signals, and enable a broad range of downstream simulations beyond isolated tasks. However, existing CDT frameworks remain limited to task-specific predictors rather than building a patient-specific, manipulable virtual heart. In this work, we introduce Chain of Flow (COF), a foundational ECG-driven generative framework that reconstructs full 4D cardiac structure and motion from a single cardiac cycle. The method integrates cine-CMR and 12-lead ECG during training to learn a unified representation of cardiac geometry, electrophysiology, and motion dynamics. We evaluate Chain of Flow on diverse cohorts and demonstrate accurate recovery of cardiac anatomy, chamber-wise function, and dynamic motion patterns. The reconstructed 4D hearts further support downstream CDT tasks such as volumetry, regional function analysis, and virtual cine synthesis. By enabling full 4D organ reconstruction directly from ECG, COF transforms cardiac digital twins from narrow predictive models into fully generative, patient-specific virtual hearts. Code will be released after review.
>
---
#### [new 060] No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors
- **分类: cs.CV**

- **简介: 该论文属于视频稳定任务，解决无标签数据下的在线视频稳定问题。提出一种基于经典先验的框架，克服数据不足、控制差和硬件效率低的问题。**

- **链接: [https://arxiv.org/pdf/2602.23141v1](https://arxiv.org/pdf/2602.23141v1)**

> **作者:** Tao Liu; Gang Wan; Kan Ren; Shibo Wen
>
> **备注:** CVPR2026
>
> **摘要:** We propose a new unsupervised framework for online video stabilization. Unlike methods based on deep learning that require paired stable and unstable datasets, our approach instantiates the classical stabilization pipeline with three stages and incorporates a multithreaded buffering mechanism. This design addresses three longstanding challenges in end-to-end learning: limited data, poor controllability, and inefficiency on hardware with constrained resources. Existing benchmarks focus mainly on handheld videos with a forward view in visible light, which restricts the applicability of stabilization to domains such as UAV nighttime remote sensing. To fill this gap, we introduce a new multimodal UAV aerial video dataset (UAV-Test). Experiments show that our method consistently outperforms state-of-the-art online stabilizers in both quantitative metrics and visual quality, while achieving performance comparable to offline methods.
>
---
#### [new 061] Sensor Generalization for Adaptive Sensing in Event-based Object Detection via Joint Distribution Training
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决事件相机数据适应性不足的问题。通过分析内在参数影响，提升模型对不同传感器的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.23357v1](https://arxiv.org/pdf/2602.23357v1)**

> **作者:** Aheli Saha; René Schuster; Didier Stricker
>
> **备注:** 12 pages, International Conference on Pattern Recognition Applications and Methods
>
> **摘要:** Bio-inspired event cameras have recently attracted significant research due to their asynchronous and low-latency capabilities. These features provide a high dynamic range and significantly reduce motion blur. However, because of the novelty in the nature of their output signals, there is a gap in the variability of available data and a lack of extensive analysis of the parameters characterizing their signals. This paper addresses these issues by providing readers with an in-depth understanding of how intrinsic parameters affect the performance of a model trained on event data, specifically for object detection. We also use our findings to expand the capabilities of the downstream model towards sensor-agnostic robustness.
>
---
#### [new 062] ProjFlow: Projection Sampling with Flow Matching for Zero-Shot Exact Spatial Motion Control
- **分类: cs.CV**

- **简介: 该论文提出ProjFlow，解决零样本精确空间运动控制问题。通过投影采样与流匹配，实现线性约束的准确满足，同时保持运动自然。**

- **链接: [https://arxiv.org/pdf/2602.22742v1](https://arxiv.org/pdf/2602.22742v1)**

> **作者:** Akihisa Watanabe; Qing Yu; Edgar Simo-Serra; Kent Fujiwara
>
> **摘要:** Generating human motion with precise spatial control is a challenging problem. Existing approaches often require task-specific training or slow optimization, and enforcing hard constraints frequently disrupts motion naturalness. Building on the observation that many animation tasks can be formulated as a linear inverse problem, we introduce ProjFlow, a training-free sampler that achieves zero-shot, exact satisfaction of linear spatial constraints while preserving motion realism. Our key advance is a novel kinematics-aware metric that encodes skeletal topology. This metric allows the sampler to enforce hard constraints by distributing corrections coherently across the entire skeleton, avoiding the unnatural artifacts of naive projection. Furthermore, for sparse inputs, such as filling in long gaps between a few keyframes, we introduce a time-varying formulation using pseudo-observations that fade during sampling. Extensive experiments on representative applications, motion inpainting, and 2D-to-3D lifting, demonstrate that ProjFlow achieves exact constraint satisfaction and matches or improves realism over zero-shot baselines, while remaining competitive with training-based controllers.
>
---
#### [new 063] D-FINE-seg: Object Detection and Instance Segmentation Framework with multi-backend deployment
- **分类: cs.CV**

- **简介: 该论文提出D-FINE-seg，解决实时实例分割问题，扩展D-FINE框架，提升分割性能并支持多后端部署。**

- **链接: [https://arxiv.org/pdf/2602.23043v1](https://arxiv.org/pdf/2602.23043v1)**

> **作者:** Argo Saakyan; Dmitry Solntsev
>
> **备注:** 6 pages, 4 figures, 5 tables
>
> **摘要:** Transformer-based real-time object detectors achieve strong accuracy-latency trade-offs, and D-FINE is among the top-performing recent architectures. However, real-time instance segmentation with transformers is still less common. We present D-FINE-seg, an instance segmentation extension of D-FINE that adds: a lightweight mask head, segmentation-aware training, including box cropped BCE and dice mask losses, auxiliary and denoising mask supervision, and adapted Hungarian matching cost. On the TACO dataset, D-FINE-seg improves F1-score over Ultralytics YOLO26 under a unified TensorRT FP16 end-to-end benchmarking protocol, while maintaining competitive latency. Second contribution is an end-to-end pipeline for training, exporting, and optimized inference across ONNX, TensorRT, OpenVINO for both object detection and instance segmentation tasks. This framework is released as open-source under the Apache-2.0 license. GitHub repository - https://github.com/ArgoHA/D-FINE-seg.
>
---
#### [new 064] Asymmetric Idiosyncrasies in Multimodal Models
- **分类: cs.CV**

- **简介: 该论文研究多模态模型中的独特性问题，分析图像生成模型是否保留文本描述的风格特征。任务属于多模态分析，旨在评估文本到图像模型的提示遵循能力。**

- **链接: [https://arxiv.org/pdf/2602.22734v1](https://arxiv.org/pdf/2602.22734v1)**

> **作者:** Muzi Tao; Chufan Shi; Huijuan Wang; Shengbang Tong; Xuezhe Ma
>
> **备注:** Project page: https://muzi-tao.github.io/asymmetric-idiosyncrasies/
>
> **摘要:** In this work, we study idiosyncrasies in the caption models and their downstream impact on text-to-image models. We design a systematic analysis: given either a generated caption or the corresponding image, we train neural networks to predict the originating caption model. Our results show that text classification yields very high accuracy (99.70\%), indicating that captioning models embed distinctive stylistic signatures. In contrast, these signatures largely disappear in the generated images, with classification accuracy dropping to at most 50\% even for the state-of-the-art Flux model. To better understand this cross-modal discrepancy, we further analyze the data and find that the generated images fail to preserve key variations present in captions, such as differences in the level of detail, emphasis on color and texture, and the distribution of objects within a scene. Overall, our classification-based framework provides a novel methodology for quantifying both the stylistic idiosyncrasies of caption models and the prompt-following ability of text-to-image systems.
>
---
#### [new 065] Small Object Detection Model with Spatial Laplacian Pyramid Attention and Multi-Scale Features Enhancement in Aerial Images
- **分类: cs.CV**

- **简介: 该论文属于小目标检测任务，旨在解决航拍图像中目标小、密集分布导致的检测效率低问题。提出SLPA和MSFEM模块，提升特征表示与融合效果。**

- **链接: [https://arxiv.org/pdf/2602.23031v1](https://arxiv.org/pdf/2602.23031v1)**

> **作者:** Zhangjian Ji; Huijia Yan; Shaotong Qiao; Kai Feng; Wei Wei
>
> **摘要:** Detecting objects in aerial images confronts some significant challenges, including small size, dense and non-uniform distribution of objects over high-resolution images, which makes detection inefficient. Thus, in this paper, we proposed a small object detection algorithm based on a Spatial Laplacian Pyramid Attention and Multi-Scale Feature Enhancement in aerial images. Firstly, in order to improve the feature representation of ResNet-50 on small objects, we presented a novel Spatial Laplacian Pyramid Attention (SLPA) module, which is integrated after each stage of ResNet-50 to identify and emphasize important local regions. Secondly, to enhance the model's semantic understanding and features representation, we designed a Multi-Scale Feature Enhancement Module (MSFEM), which is incorporated into the lateral connections of C5 layer for building Feature Pyramid Network (FPN). Finally, the features representation quality of traditional feature pyramid network will be affected because the features are not aligned when the upper and lower layers are fused. In order to handle it, we utilized deformable convolutions to align the features in the fusion processing of the upper and lower levels of the Feature Pyramid Network, which can help enhance the model's ability to detect and recognize small objects. The extensive experimental results on two benchmark datasets: VisDrone and DOTA demonstrate that our improved model performs better for small object detection in aerial images compared to the original algorithm.
>
---
#### [new 066] SO3UFormer: Learning Intrinsic Spherical Features for Rotation-Robust Panoramic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于全景分割任务，解决相机旋转导致的性能下降问题。提出SO3UFormer模型，通过几何方法提升对旋转的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.22867v1](https://arxiv.org/pdf/2602.22867v1)**

> **作者:** Qinfeng Zhu; Yunxi Jiang; Lei Fan
>
> **摘要:** Panoramic semantic segmentation models are typically trained under a strict gravity-aligned assumption. However, real-world captures often deviate from this canonical orientation due to unconstrained camera motions, such as the rotational jitter of handheld devices or the dynamic attitude shifts of aerial platforms. This discrepancy causes standard spherical Transformers to overfit global latitude cues, leading to performance collapse under 3D reorientations. To address this, we introduce SO3UFormer, a rotation-robust architecture designed to learn intrinsic spherical features that are less sensitive to the underlying coordinate frame. Our approach rests on three geometric pillars: (1) an intrinsic feature formulation that decouples the representation from the gravity vector by removing absolute latitude encoding; (2) quadrature-consistent spherical attention that accounts for non-uniform sampling densities; and (3) a gauge-aware relative positional mechanism that encodes local angular geometry using tangent-plane projected angles and discrete gauge pooling, avoiding reliance on global axes. We further use index-based spherical resampling together with a logit-level SO(3)-consistency regularizer during training. To rigorously benchmark robustness, we introduce Pose35, a dataset variant of Stanford2D3D perturbed by random rotations within $\pm 35^\circ$. Under the extreme test of arbitrary full SO(3) rotations, existing SOTAs fail catastrophically: the baseline SphereUFormer drops from 67.53 mIoU to 25.26 mIoU. In contrast, SO3UFormer demonstrates remarkable stability, achieving 72.03 mIoU on Pose35 and retaining 70.67 mIoU under full SO(3) rotations.
>
---
#### [new 067] LineGraph2Road: Structural Graph Reasoning on Line Graphs for Road Network Extraction
- **分类: cs.CV**

- **简介: 该论文属于道路网络提取任务，旨在解决卫星图像中道路自动识别的问题。通过构建线图并应用图Transformer，提升连接性预测效果。**

- **链接: [https://arxiv.org/pdf/2602.23290v1](https://arxiv.org/pdf/2602.23290v1)**

> **作者:** Zhengyang Wei; Renzhi Jing; Yiyi He; Jenny Suckale
>
> **摘要:** The accurate and automatic extraction of roads from satellite imagery is critical for applications in navigation and urban planning, significantly reducing the need for manual annotation. Many existing methods decompose this task into keypoint extraction and connectedness prediction, but often struggle to capture long-range dependencies and complex topologies. Here, we propose LineGraph2Road, a framework that improves connectedness prediction by formulating it as binary classification over edges in a constructed global but sparse Euclidean graph, where nodes are keypoints extracted from segmentation masks and edges connect node pairs within a predefined distance threshold, representing potential road segments. To better learn structural link representation, we transform the original graph into its corresponding line graph and apply a Graph Transformer on it for connectedness prediction. This formulation overcomes the limitations of endpoint-embedding fusion on set-isomorphic links, enabling rich link representations and effective relational reasoning over the global structure. Additionally, we introduce an overpass/underpass head to resolve multi-level crossings and a coupled NMS strategy to preserve critical connections. We evaluate LineGraph2Road on three benchmarks: City-scale, SpaceNet, and Global-scale, and show that it achieves state-of-the-art results on two key metrics, TOPO-F1 and APLS. It also captures fine visual details critical for real-world deployment. We will make our code publicly available.
>
---
#### [new 068] Don't let the information slip away
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决现有模型忽略背景信息的问题。通过提出Association DETR模型，融合背景上下文提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.22595v1](https://arxiv.org/pdf/2602.22595v1)**

> **作者:** Taozhe Li
>
> **备注:** 10
>
> **摘要:** Real-time object detection has advanced rapidly in recent years. The YOLO series of detectors is among the most well-known CNN-based object detection models and cannot be overlooked. The latest version, YOLOv26, was recently released, while YOLOv12 achieved state-of-the-art (SOTA) performance with 55.2 mAP on the COCO val2017 dataset. Meanwhile, transformer-based object detection models, also known as DEtection TRansformer (DETR), have demonstrated impressive performance. RT-DETR is an outstanding model that outperformed the YOLO series in both speed and accuracy when it was released. Its successor, RT-DETRv2, achieved 53.4 mAP on the COCO val2017 dataset. However, despite their remarkable performance, all these models let information to slip away. They primarily focus on the features of foreground objects while neglecting the contextual information provided by the background. We believe that background information can significantly aid object detection tasks. For example, cars are more likely to appear on roads rather than in offices, while wild animals are more likely to be found in forests or remote areas rather than on busy streets. To address this gap, we propose an object detection model called Association DETR, which achieves state-of-the-art results compared to other object detection models on the COCO val2017 dataset.
>
---
#### [new 069] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation
- **分类: cs.CV**

- **简介: 该论文提出SceneTransporter，用于从单图生成结构化3D场景。解决现有方法在组织3D部件为独立实例上的不足，通过优化运输目标实现结构约束。**

- **链接: [https://arxiv.org/pdf/2602.22785v1](https://arxiv.org/pdf/2602.22785v1)**

> **作者:** Ling Wang; Hao-Xiang Guo; Xinzhou Wang; Fuchun Sun; Kai Sun; Pengkun Liu; Hang Xiao; Zhong Wang; Guangyuan Fu; Eric Li; Yang Liu; Yikai Wang
>
> **备注:** published at iclr 2026
>
> **摘要:** We introduce SceneTransporter, an end-to-end framework for structured 3D scene generation from a single image. While existing methods generate part-level 3D objects, they often fail to organize these parts into distinct instances in open-world scenes. Through a debiased clustering probe, we reveal a critical insight: this failure stems from the lack of structural constraints within the model's internal assignment mechanism. Based on this finding, we reframe the task of structured 3D scene generation as a global correlation assignment problem. To solve this, SceneTransporter formulates and solves an entropic Optimal Transport (OT) objective within the denoising loop of the compositional DiT model. This formulation imposes two powerful structural constraints. First, the resulting transport plan gates cross-attention to enforce an exclusive, one-to-one routing of image patches to part-level 3D latents, preventing entanglement. Second, the competitive nature of the transport encourages the grouping of similar patches, a process that is further regularized by an edge-based cost, to form coherent objects and prevent fragmentation. Extensive experiments show that SceneTransporter outperforms existing methods on open-world scene generation, significantly improving instance-level coherence and geometric fidelity. Code and models will be publicly available at https://2019epwl.github.io/SceneTransporter/.
>
---
#### [new 070] GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决大气湍流导致的图像模糊和位移问题。通过结合光流引导的倾斜校正与高斯点云建模，提升长距离成像质量。**

- **链接: [https://arxiv.org/pdf/2602.22800v1](https://arxiv.org/pdf/2602.22800v1)**

> **作者:** Hanliang Du; Zhangji Lu; Zewei Cai; Qijian Tang; Qifeng Yu; Xiaoli Liu
>
> **摘要:** Atmospheric turbulence causes significant image degradation due to pixel displacement (tilt) and blur, particularly in long-range imaging applications. In this paper, we propose a novel framework for atmospheric turbulence mitigation, GSTurb, which integrates optical flow-guided tilt correction and Gaussian splatting for modeling non-isoplanatic blur. The framework employs Gaussian parameters to represent tilt and blur, and optimizes them across multiple frames to enhance restoration. Experimental results on the ATSyn-static dataset demonstrate the effectiveness of our method, achieving a peak PSNR of 27.67 dB and SSIM of 0.8735. Compared to the state-of-the-art method, GSTurb improves PSNR by 1.3 dB (a 4.5% increase) and SSIM by 0.048 (a 5.8% increase). Additionally, on real datasets, including the TSRWGAN Real-World and CLEAR datasets, GSTurb outperforms existing methods, showing significant improvements in both qualitative and quantitative performance. These results highlight that combining optical flow-guided tilt correction with Gaussian splatting effectively enhances image restoration under both synthetic and real-world turbulence conditions. The code for this method will be available at https://github.com/DuhlLiamz/3DGS_turbulence/tree/main.
>
---
#### [new 071] From Calibration to Refinement: Seeking Certainty via Probabilistic Evidence Propagation for Noisy-Label Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于人物重识别任务，解决噪声标签和样本稀疏的问题。提出CARE方法，通过概率证据传播实现校准与精炼，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.23133v1](https://arxiv.org/pdf/2602.23133v1)**

> **作者:** Xin Yuan; Zhiyong Zhang; Xin Xu; Zheng Wang; Chia-Wen Lin
>
> **备注:** Accepted by IEEE TMM 2026
>
> **摘要:** With the increasing demand for robust person Re-ID in unconstrained environments, learning from datasets with noisy labels and sparse per-identity samples remains a critical challenge. Existing noise-robust person Re-ID methods primarily rely on loss-correction or sample-selection strategies using softmax outputs. However, these methods suffer from two key limitations: 1) Softmax exhibits translation invariance, leading to over-confident and unreliable predictions on corrupted labels. 2) Conventional sample selection based on small-loss criteria often discards valuable hard positives that are crucial for learning discriminative features. To overcome these issues, we propose the CAlibration-to-REfinement (CARE) method, a two-stage framework that seeks certainty through probabilistic evidence propagation from calibration to refinement. In the calibration stage, we propose the probabilistic evidence calibration (PEC) that dismantles softmax translation invariance by injecting adaptive learnable parameters into the similarity function, and employs an evidential calibration loss to mitigate overconfidence on mislabeled samples. In the refinement stage, we design the evidence propagation refinement (EPR) that can more accurately distinguish between clean and noisy samples. Specifically, the EPR contains two steps: Firstly, the composite angular margin (CAM) metric is proposed to precisely distinguish clean but hard-to-learn positive samples from mislabeled ones in a hyperspherical space; Secondly, the certainty-oriented sphere weighting (COSW) is developed to dynamically allocate the importance of samples according to CAM, ensuring clean instances drive model updates. Extensive experimental results on Market1501, DukeMTMC-ReID, and CUHK03 datasets under both random and patterned noises show that CARE achieves competitive performance.
>
---
#### [new 072] Reflectance Multispectral Imaging for Soil Composition Estimation and USDA Texture Classification
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于土壤纹理分类任务，旨在解决传统方法耗时、昂贵的问题。通过多光谱成像与机器学习，实现土壤成分和纹理类别的准确预测。**

- **链接: [https://arxiv.org/pdf/2602.22829v1](https://arxiv.org/pdf/2602.22829v1)**

> **作者:** G. A. S. L Ranasinghe; J. A. S. T. Jayakody; M. C. L. De Silva; G. Thilakarathne; G. M. R. I. Godaliyadda; H. M. V. R. Herath; M. P. B. Ekanayake; S. K. Navaratnarajah
>
> **备注:** Under Review at IEEE Access. 17 pages, 15 figures
>
> **摘要:** Soil texture is a foundational attribute that governs water availability and erosion in agriculture, as well as load bearing capacity, deformation response, and shrink-swell risk in geotechnical engineering. Yet texture is still typically determined by slow and labour intensive laboratory particle size tests, while many sensing alternatives are either costly or too coarse to support routine field scale deployment. This paper proposes a robust and field deployable multispectral imaging (MSI) system and machine learning framework for predicting soil composition and the United States Department of Agriculture (USDA) texture classes. The proposed system uses a cost effective in-house MSI device operating from 365 nm to 940 nm to capture thirteen spectral bands, which effectively capture the spectral properties of soil texture. Regression models use the captured spectral properties to estimate clay, silt, and sand percentages, while a direct classifier predicts one of the twelve USDA textural classes. Indirect classification is obtained by mapping the regressed compositions to texture classes via the USDA soil texture triangle. The framework is evaluated on mixture data by mixing clay, silt, and sand in varying proportions, using the USDA classification triangle as a basis. Experimental results show that the proposed approach achieves a coefficient of determination R^2 up to 0.99 for composition prediction and over 99% accuracy for texture classification. These findings indicate that MSI combined with data-driven modeling can provide accurate, non-destructive, and field deployable soil texture characterization suitable for geotechnical screening and precision agriculture.
>
---
#### [new 073] LoR-LUT: Learning Compact 3D Lookup Tables via Low-Rank Residuals
- **分类: cs.CV**

- **简介: 该论文提出LoR-LUT，用于生成紧凑且可解释的3D LUT，解决图像增强与风格迁移问题。通过低秩残差修正，提升图像质量并减少参数量。**

- **链接: [https://arxiv.org/pdf/2602.22607v1](https://arxiv.org/pdf/2602.22607v1)**

> **作者:** Ziqi Zhao; Abhijit Mishra; Shounak Roychowdhury
>
> **摘要:** We present LoR-LUT, a unified low-rank formulation for compact and interpretable 3D lookup table (LUT) generation. Unlike conventional 3D-LUT-based techniques that rely on fusion of basis LUTs, which are usually dense tensors, our unified approach extends the current framework by jointly using residual corrections, which are in fact low-rank tensors, together with a set of basis LUTs. The approach described here improves the existing perceptual quality of an image, which is primarily due to the technique's novel use of residual corrections. At the same time, we achieve the same level of trilinear interpolation complexity, using a significantly smaller number of network, residual corrections, and LUT parameters. The experimental results obtained from LoR-LUT, which is trained on the MIT-Adobe FiveK dataset, reproduce expert-level retouching characteristics with high perceptual fidelity and a sub-megabyte model size. Furthermore, we introduce an interactive visualization tool, termed LoR-LUT Viewer, which transforms an input image into the LUT-adjusted output image, via a number of slidebars that control different parameters. The tool provides an effective way to enhance interpretability and user confidence in the visual results. Overall, our proposed formulation offers a compact, interpretable, and efficient direction for future LUT-based image enhancement and style transfer.
>
---
#### [new 074] SPMamba-YOLO: An Underwater Object Detection Network Based on Multi-Scale Feature Enhancement and Global Context Modeling
- **分类: cs.CV**

- **简介: 该论文属于 underwater object detection 任务，旨在解决水下目标检测中的光照衰减、颜色失真等问题。提出 SPMamba-YOLO 网络，融合多尺度特征增强与全局上下文建模，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.22674v1](https://arxiv.org/pdf/2602.22674v1)**

> **作者:** Guanghao Liao; Zhen Liu; Liyuan Cao; Yonghui Yang; Qi Li
>
> **备注:** 31 pages, 10 figures, 6 tables. This paper presents SPMamba-YOLO, an underwater object detection framework integrating multi-scale feature enhancement and global context modeling. The work is under review
>
> **摘要:** Underwater object detection is a critical yet challenging research problem owing to severe light attenuation, color distortion, background clutter, and the small scale of underwater targets. To address these challenges, we propose SPMamba-YOLO, a novel underwater object detection network that integrates multi-scale feature enhancement with global context modeling. Specifically, a Spatial Pyramid Pooling Enhanced Layer Aggregation Network (SPPELAN) module is introduced to strengthen multi-scale feature aggregation and expand the receptive field, while a Pyramid Split Attention (PSA) mechanism enhances feature discrimination by emphasizing informative regions and suppressing background interference. In addition, a Mamba-based state space modeling module is incorporated to efficiently capture long-range dependencies and global contextual information, thereby improving detection robustness in complex underwater environments. Extensive experiments on the URPC2022 dataset demonstrate that SPMamba-YOLO outperforms the YOLOv8n baseline by more than 4.9\% in mAP@0.5, particularly for small and densely distributed underwater objects, while maintaining a favorable balance between detection accuracy and computational cost.
>
---
#### [new 075] ColoDiff: Integrating Dynamic Consistency With Content Awareness for Colonoscopy Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视频生成任务，旨在解决数据稀缺和动态一致性问题。提出ColoDiff框架，结合时间一致性和内容感知机制，生成高质量结肠镜视频，辅助临床分析。**

- **链接: [https://arxiv.org/pdf/2602.23203v1](https://arxiv.org/pdf/2602.23203v1)**

> **作者:** Junhu Fu; Shuyu Liang; Wutong Li; Chen Ma; Peng Huang; Kehao Wang; Ke Chen; Shengli Lin; Pinghong Zhou; Zeju Li; Yuanyuan Wang; Yi Guo
>
> **摘要:** Colonoscopy video generation delivers dynamic, information-rich data critical for diagnosing intestinal diseases, particularly in data-scarce scenarios. High-quality video generation demands temporal consistency and precise control over clinical attributes, but faces challenges from irregular intestinal structures, diverse disease representations, and various imaging modalities. To this end, we propose ColoDiff, a diffusion-based framework that generates dynamic-consistent and content-aware colonoscopy videos, aiming to alleviate data shortage and assist clinical analysis. At the inter-frame level, our TimeStream module decouples temporal dependency from video sequences through a cross-frame tokenization mechanism, enabling intricate dynamic modeling despite irregular intestinal structures. At the intra-frame level, our Content-Aware module incorporates noise-injected embeddings and learnable prototypes to realize precise control over clinical attributes, breaking through the coarse guidance of diffusion models. Additionally, ColoDiff employs a non-Markovian sampling strategy that cuts steps by over 90% for real-time generation. ColoDiff is evaluated across three public datasets and one hospital database, based on both generation metrics and downstream tasks including disease diagnosis, modality discrimination, bowel preparation scoring, and lesion segmentation. Extensive experiments show ColoDiff generates videos with smooth transitions and rich dynamics. ColoDiff presents an effort in controllable colonoscopy video generation, revealing the potential of synthetic videos in complementing authentic representation and mitigating data scarcity in clinical settings.
>
---
#### [new 076] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在提升长视频的高效理解能力。通过联合优化MLLM和关键帧采样器，提升问答准确率。**

- **链接: [https://arxiv.org/pdf/2602.22932v1](https://arxiv.org/pdf/2602.22932v1)**

> **作者:** Wenhui Tan; Xiaoyi Yu; Jiaze Li; Yijing Chen; Jianzhong Ju; Zhenbo Luo; Ruihua Song; Jian Luan
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Efficiently understanding long-form videos remains a fundamental challenge for multimodal large language models (MLLMs). In this paper, we present MLLM-Sampler Joint Evolution (MSJoE), a novel framework that jointly evolves the MLLM and a lightweight key-frame sampler for efficient long-form video understanding. MSJoE builds upon a key assumption that only a small subset of key-frames is truly informative for answering each question to a video. Specifically, MSJoE first reasons out several queries, which describe diverse visual perspectives relevant to the question. Then, these queries interact with a frozen CLIP model to produce a query-frame similarity matrix. Finally, a lightweight sampler predicts key-frame sampling weights from this matrix, selecting a compact set of informative frames, which are then fed into the MLLM for answer generation. Both the MLLM and sampler are jointly optimized through reinforcement learning, enabling co-adaptation of query-reasoning, frame-sampling, and key-frame understanding. A new long-video QA dataset containing 2.8K videos with 7K question-answer pairs is collected to support the training process. Extensive experiments on VideoMME, LongVideoBench, LVBench, and MLVU show that MSJoE achieves 8.0\% accuracy gain upon the base MLLM, and 1.1\% higher accuracy than strongest baseline method.
>
---
#### [new 077] Cross-Task Benchmarking of CNN Architectures
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉领域，比较不同CNN架构在图像分类、分割和时序分析中的表现，旨在提升模型的准确性和效率。通过实验验证了动态卷积和注意力机制的优势。**

- **链接: [https://arxiv.org/pdf/2602.22945v1](https://arxiv.org/pdf/2602.22945v1)**

> **作者:** Kamal Sherawat; Vikrant Bhati
>
> **摘要:** This project provides a comparative study of dynamic convolutional neural networks (CNNs) for various tasks, including image classification, segmentation, and time series analysis. Based on the ResNet-18 architecture, we compare five variants of CNNs: the vanilla CNN, the hard attention-based CNN, the soft attention-based CNN with local (pixel-wise) and global (image-wise) feature attention, and the omni-directional CNN (ODConv). Experiments on Tiny ImageNet, Pascal VOC, and the UCR Time Series Classification Archive illustrate that attention mechanisms and dynamic convolution methods consistently exceed conventional CNNs in accuracy, efficiency, and computational performance. ODConv was especially effective on morphologically complex images by being able to dynamically adjust to varying spatial patterns. Dynamic CNNs enhanced feature representation and cross-task generalization through adaptive kernel modulation. This project provides perspectives on advanced CNN design architecture for multiplexed data modalities and indicates promising directions in neural network engineering.
>
---
#### [new 078] SPATIALALIGN: Aligning Dynamic Spatial Relationships in Video Generation
- **分类: cs.CV**

- **简介: 该论文属于文本生成视频任务，解决视频生成中空间关系对齐问题。提出SPATIALALIGN框架，通过DSR-SCORE评估指标和DPO优化方法，提升模型对动态空间关系的生成能力。**

- **链接: [https://arxiv.org/pdf/2602.22745v1](https://arxiv.org/pdf/2602.22745v1)**

> **作者:** Fengming Liu; Tat-Jen Cham; Chuanxia Zheng
>
> **摘要:** Most text-to-video (T2V) generators prioritize aesthetic quality, but often ignoring the spatial constraints in the generated videos. In this work, we present SPATIALALIGN, a self-improvement framework that enhances T2V models capabilities to depict Dynamic Spatial Relationships (DSR) specified in text prompts. We present a zeroth-order regularized Direct Preference Optimization (DPO) to fine-tune T2V models towards better alignment with DSR. Specifically, we design DSR-SCORE, a geometry-based metric that quantitatively measures the alignment between generated videos and the specified DSRs in prompts, which is a step forward from prior works that rely on VLM for evaluation. We also conduct a dataset of text-video pairs with diverse DSRs to facilitate the study. Extensive experiments demonstrate that our fine-tuned model significantly out performs the baseline in spatial relationships. The code will be released in Link.
>
---
#### [new 079] Optimizing Neural Network Architecture for Medical Image Segmentation Using Monte Carlo Tree Search
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升分割精度与效率。通过结合MCTS与NAS，提出MNAS-Unet框架，优化网络结构，减少搜索成本并提高性能。**

- **链接: [https://arxiv.org/pdf/2602.22361v1](https://arxiv.org/pdf/2602.22361v1)**

> **作者:** Liping Meng; Fan Nie; Yunyun Zhang; Chao Han
>
> **摘要:** This paper proposes a novel medical image segmentation framework, MNAS-Unet, which combines Monte Carlo Tree Search (MCTS) and Neural Architecture Search (NAS). MNAS-Unet dynamically explores promising network architectures through MCTS, significantly enhancing the efficiency and accuracy of architecture search. It also optimizes the DownSC and UpSC unit structures, enabling fast and precise model adjustments. Experimental results demonstrate that MNAS-Unet outperforms NAS-Unet and other state-of-the-art models in segmentation accuracy on several medical image datasets, including PROMISE12, Ultrasound Nerve, and CHAOS. Furthermore, compared with NAS-Unet, MNAS-Unet reduces the architecture search budget by 54% (early stopping at 139 epochs versus 300 epochs under the same search setting), while achieving a lightweight model with only 0.6M parameters and lower GPU memory consumption, which further improves its practical applicability. These results suggest that MNAS-Unet can improve search efficiency while maintaining competitive segmentation accuracy under practical resource constraints.
>
---
#### [new 080] Causal Motion Diffusion Models for Autoregressive Motion Generation
- **分类: cs.CV**

- **简介: 该论文属于动作生成任务，解决现有模型在时间因果性和实时性上的不足。提出CMDM框架，结合因果扩散Transformer和语义对齐潜空间，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2602.22594v1](https://arxiv.org/pdf/2602.22594v1)**

> **作者:** Qing Yu; Akihisa Watanabe; Kent Fujiwara
>
> **备注:** Accepted to CVPR 2026, Project website: https://yu1ut.com/CMDM-HP/
>
> **摘要:** Recent advances in motion diffusion models have substantially improved the realism of human motion synthesis. However, existing approaches either rely on full-sequence diffusion models with bidirectional generation, which limits temporal causality and real-time applicability, or autoregressive models that suffer from instability and cumulative errors. In this work, we present Causal Motion Diffusion Models (CMDM), a unified framework for autoregressive motion generation based on a causal diffusion transformer that operates in a semantically aligned latent space. CMDM builds upon a Motion-Language-Aligned Causal VAE (MAC-VAE), which encodes motion sequences into temporally causal latent representations. On top of this latent representation, an autoregressive diffusion transformer is trained using causal diffusion forcing to perform temporally ordered denoising across motion frames. To achieve fast inference, we introduce a frame-wise sampling schedule with causal uncertainty, where each subsequent frame is predicted from partially denoised previous frames. The resulting framework supports high-quality text-to-motion generation, streaming synthesis, and long-horizon motion generation at interactive rates. Experiments on HumanML3D and SnapMoGen demonstrate that CMDM outperforms existing diffusion and autoregressive models in both semantic fidelity and temporal smoothness, while substantially reducing inference latency.
>
---
#### [new 081] CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于无源域适应目标检测任务，解决在无源数据情况下将检测器适配到目标域的问题。提出CGSA框架，结合类引导的槽感知机制，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.22621v1](https://arxiv.org/pdf/2602.22621v1)**

> **作者:** Boyang Dai; Zeng Fan; Zihao Qi; Meng Lou; Yizhou Yu
>
> **备注:** The paper has been accepted by the conference ICLR 2026
>
> **摘要:** Source-Free Domain Adaptive Object Detection (SF-DAOD) aims to adapt a detector trained on a labeled source domain to an unlabeled target domain without retaining any source data. Despite recent progress, most popular approaches focus on tuning pseudo-label thresholds or refining the teacher-student framework, while overlooking object-level structural cues within cross-domain data. In this work, we present CGSA, the first framework that brings Object-Centric Learning (OCL) into SF-DAOD by integrating slot-aware adaptation into the DETR-based detector. Specifically, our approach integrates a Hierarchical Slot Awareness (HSA) module into the detector to progressively disentangle images into slot representations that act as visual priors. These slots are then guided toward class semantics via a Class-Guided Slot Contrast (CGSC) module, maintaining semantic consistency and prompting domain-invariant adaptation. Extensive experiments on multiple cross-domain datasets demonstrate that our approach outperforms previous SF-DAOD methods, with theoretical derivations and experimental analysis further demonstrating the effectiveness of the proposed components and the framework, thereby indicating the promise of object-centric design in privacy-sensitive adaptation scenarios. Code is released at https://github.com/Michael-McQueen/CGSA.
>
---
#### [new 082] MammoWise: Multi-Model Local RAG Pipeline for Mammography Report Generation
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于医学影像报告生成任务，旨在解决 mammography 报告自动化问题。提出 MammoWise 系统，利用本地多模型管道实现高效、可复现的报告生成与分类。**

- **链接: [https://arxiv.org/pdf/2602.22462v1](https://arxiv.org/pdf/2602.22462v1)**

> **作者:** Raiyan Jahangir; Nafiz Imtiaz Khan; Amritanand Sudheerkumar; Vladimir Filkov
>
> **备注:** arXiv preprint (submitted 25 Feb 2026). Local multi-model pipeline for mammography report generation + classification using prompting, multimodal RAG (ChromaDB), and QLoRA fine-tuning; evaluates MedGemma, LLaVA-Med, Qwen2.5-VL on VinDr-Mammo and DMID; reports BERTScore/ROUGE-L and classification metrics
>
> **摘要:** Screening mammography is high volume, time sensitive, and documentation heavy. Radiologists must translate subtle visual findings into consistent BI-RADS assessments, breast density categories, and structured narrative reports. While recent Vision Language Models (VLMs) enable image-to-text reporting, many rely on closed cloud systems or tightly coupled architectures that limit privacy, reproducibility, and adaptability. We present MammoWise, a local multi-model pipeline that transforms open source VLMs into mammogram report generators and multi-task classifiers. MammoWise supports any Ollama-hosted VLM and mammography dataset, and enables zero-shot, few-shot, and Chain-of-Thought prompting, with optional multimodal Retrieval Augmented Generation (RAG) using a vector database for case-specific context. We evaluate MedGemma, LLaVA-Med, and Qwen2.5-VL on VinDr-Mammo and DMID datasets, assessing report quality (BERTScore, ROUGE-L), BI-RADS classification, breast density, and key findings. Report generation is consistently strong and improves with few-shot prompting and RAG. Classification is feasible but sensitive to model and dataset choice. Parameter-efficient fine-tuning (QLoRA) of MedGemma improves reliability, achieving BI-RADS accuracy of 0.7545, density accuracy of 0.8840, and calcification accuracy of 0.9341 while preserving report quality. MammoWise provides a practical and extensible framework for deploying local VLMs for mammography reporting within a unified and reproducible workflow.
>
---
#### [new 083] WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出WaterVideoQA基准和NaviMind系统，解决自主水面航行器在动态水域环境中的认知与决策问题，提升其安全性和准确性。**

- **链接: [https://arxiv.org/pdf/2602.22923v1](https://arxiv.org/pdf/2602.22923v1)**

> **作者:** Runwei Guan; Shaofeng Liang; Ningwei Ouyang; Weichen Fei; Shanliang Yao; Wei Dai; Chenhao Ge; Penglei Sun; Xiaohui Zhu; Tao Huang; Ryan Wen Liu; Hui Xiong
>
> **备注:** 11 pages,8 figures
>
> **摘要:** While autonomous navigation has achieved remarkable success in passive perception (e.g., object detection and segmentation), it remains fundamentally constrained by a void in knowledge-driven, interactive environmental cognition. In the high-stakes domain of maritime navigation, the ability to bridge the gap between raw visual perception and complex cognitive reasoning is not merely an enhancement but a critical prerequisite for Autonomous Surface Vessels to execute safe and precise maneuvers. To this end, we present WaterVideoQA, the first large-scale, comprehensive Video Question Answering benchmark specifically engineered for all-waterway environments. This benchmark encompasses 3,029 video clips across six distinct waterway categories, integrating multifaceted variables such as volatile lighting and dynamic weather to rigorously stress-test ASV capabilities across a five-tier hierarchical cognitive framework. Furthermore, we introduce NaviMind, a pioneering multi-agent neuro-symbolic system designed for open-ended maritime reasoning. By synergizing Adaptive Semantic Routing, Situation-Aware Hierarchical Reasoning, and Autonomous Self-Reflective Verification, NaviMind transitions ASVs from superficial pattern matching to regulation-compliant, interpretable decision-making. Experimental results demonstrate that our framework significantly transcends existing baselines, establishing a new paradigm for intelligent, trustworthy interaction in dynamic maritime environments.
>
---
#### [new 084] Locally Adaptive Decay Surfaces for High-Speed Face and Landmark Detection with Event Cameras
- **分类: cs.CV**

- **简介: 该论文针对事件相机的高帧率人脸与关键点检测任务，提出局部自适应衰减表面（LADS），解决传统方法在动态与静态区域间的性能平衡问题。**

- **链接: [https://arxiv.org/pdf/2602.23101v1](https://arxiv.org/pdf/2602.23101v1)**

> **作者:** Paul Kielty; Timothy Hanley; Peter Corcoran
>
> **摘要:** Event cameras record luminance changes with microsecond resolution, but converting their sparse, asynchronous output into dense tensors that neural networks can exploit remains a core challenge. Conventional histograms or globally-decayed time-surface representations apply fixed temporal parameters across the entire image plane, which in practice creates a trade-off between preserving spatial structure during still periods and retaining sharp edges during rapid motion. We introduce Locally Adaptive Decay Surfaces (LADS), a family of event representations in which the temporal decay at each location is modulated according to local signal dynamics. Three strategies are explored, based on event rate, Laplacian-of-Gaussian response, and high-frequency spectral energy. These adaptive schemes preserve detail in quiescent regions while reducing blur in regions of dense activity. Extensive experiments on the public data show that LADS consistently improves both face detection and facial landmark accuracy compared to standard non-adaptive representations. At 30 Hz, LADS achieves higher detection accuracy and lower landmark error than either baseline, and at 240 Hz it mitigates the accuracy decline typically observed at higher frequencies, sustaining 2.44 % normalized mean error for landmarks and 0.966 mAP50 in face detection. These high-frequency results even surpass the accuracy reported in prior works operating at 30 Hz, setting new benchmarks for event-based face analysis. Moreover, by preserving spatial structure at the representation stage, LADS supports the use of much lighter network architectures while still retaining real-time performance. These results highlight the importance of context-aware temporal integration for neuromorphic vision and point toward real-time, high-frequency human-computer interaction systems that exploit the unique advantages of event cameras.
>
---
#### [new 085] A data- and compute-efficient chest X-ray foundation model beyond aggressive scaling
- **分类: cs.CV**

- **简介: 该论文提出CheXficient，解决医学影像基础模型训练中的数据冗余和计算效率问题，通过精选样本实现高效预训练。**

- **链接: [https://arxiv.org/pdf/2602.22843v1](https://arxiv.org/pdf/2602.22843v1)**

> **作者:** Chong Wang; Yabin Zhang; Yunhe Gao; Maya Varma; Clemence Mottez; Faidra Patsatzi; Jiaming Liu; Jin Long; Jean-Benoit Delbrouck; Sergios Gatidis; Akshay S. Chaudhari; Curtis P. Langlotz
>
> **摘要:** Foundation models for medical imaging are typically pretrained on increasingly large datasets, following a "scale-at-all-costs" paradigm. However, this strategy faces two critical challenges: large-scale medical datasets often contain substantial redundancy and severe class imbalance that bias representation learning toward over-represented patterns, and indiscriminate training regardless of heterogeneity in data quality incurs considerable computational inefficiency. Here we demonstrate that active, principled data curation during pretraining can serve as a viable, cost-effective alternative to brute-force dataset enlargement. We introduce CheXficient, a chest X-ray (CXR) foundation model that selectively prioritizes informative training samples. CheXficient is pretrained on only 22.7% of 1,235,004 paired CXR images and reports while consuming under 27.3% of the total compute budget, yet achieving comparable or superior performance to its full-data counterpart and other large-scale pretrained models. We assess CheXficient across 20 individual benchmarks spanning 5 task types, including non-adapted off-the-shelf evaluations (zero-shot findings classification and crossmodal retrieval) and adapted downstream tasks (disease prediction, semantic segmentation, and radiology report generation). Further analyses show that CheXficient systematically prioritizes under-represented training samples, improving generalizability on long-tailed or rare conditions. Overall, our work offers practical insights into the data and computation demands for efficient pretraining and downstream adaptation of medical vision-language foundation models.
>
---
#### [new 086] pMoE: Prompting Diverse Experts Together Wins More in Visual Adaptation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉适应任务，旨在解决单一模型知识局限性问题。提出pMoE方法，通过多专家提示调优融合不同领域知识，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.22938v1](https://arxiv.org/pdf/2602.22938v1)**

> **作者:** Shentong Mo; Xufang Luo; Dongsheng Li
>
> **摘要:** Parameter-efficient fine-tuning has demonstrated promising results across various visual adaptation tasks, such as classification and segmentation. Typically, prompt tuning techniques have harnessed knowledge from a single pre-trained model, whether from a general or a specialized medical domain. However, this approach typically overlooks the potential synergies that could arise from integrating diverse domain knowledge within the same tuning process. In this work, we propose a novel Mixture-of-Experts prompt tuning method called pMoE, which leverages the strengths of multiple expert domains through expert-specialized prompt tokens and the learnable dispatcher, effectively combining their expertise in a unified model framework. Our pMoE introduces expert-specific prompt tokens and utilizes a dynamic token dispatching mechanism at various prompt layers to optimize the contribution of each domain expert during the adaptation phase. By incorporating both domain knowledge from diverse experts, the proposed pMoE significantly enhances the model's versatility and applicability to a broad spectrum of tasks. We conduct extensive experiments across 47 adaptation tasks, including both classification and segmentation in general and medical domains. The results demonstrate that our pMoE not only achieves superior performance with a large margin of improvements but also offers an optimal trade-off between computational efficiency and adaptation effectiveness compared to existing methods.
>
---
#### [new 087] EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents
- **分类: cs.CV**

- **简介: 该论文提出EmbodMocap，用于在真实环境中重建人体与场景的4D数据。解决传统采集系统成本高、受限的问题，通过双iPhone实现低成本、便携的数据收集与场景一致的运动重建。**

- **链接: [https://arxiv.org/pdf/2602.23205v1](https://arxiv.org/pdf/2602.23205v1)**

> **作者:** Wenjia Wang; Liang Pan; Huaijin Pi; Yuke Lou; Xuqian Ren; Yifan Wu; Zhouyingcheng Liao; Lei Yang; Rishabh Dabral; Christian Theobalt; Taku Komura
>
> **摘要:** Human behaviors in the real world naturally encode rich, long-term contextual information that can be leveraged to train embodied agents for perception, understanding, and acting. However, existing capture systems typically rely on costly studio setups and wearable devices, limiting the large-scale collection of scene-conditioned human motion data in the wild. To address this, we propose EmbodMocap, a portable and affordable data collection pipeline using two moving iPhones. Our key idea is to jointly calibrate dual RGB-D sequences to reconstruct both humans and scenes within a unified metric world coordinate frame. The proposed method allows metric-scale and scene-consistent capture in everyday environments without static cameras or markers, bridging human motion and scene geometry seamlessly. Compared with optical capture ground truth, we demonstrate that the dual-view setting exhibits a remarkable ability to mitigate depth ambiguity, achieving superior alignment and reconstruction performance over single iphone or monocular models. Based on the collected data, we empower three embodied AI tasks: monocular human-scene-reconstruction, where we fine-tune on feedforward models that output metric-scale, world-space aligned humans and scenes; physics-based character animation, where we prove our data could be used to scale human-object interaction skills and scene-aware motion tracking; and robot motion control, where we train a humanoid robot via sim-to-real RL to replicate human motions depicted in videos. Experimental results validate the effectiveness of our pipeline and its contributions towards advancing embodied AI research.
>
---
#### [new 088] Towards Multimodal Domain Generalization with Few Labels
- **分类: cs.CV**

- **简介: 该论文提出SSMDG任务，解决少标签下多模态域泛化问题。通过统一框架提升模型鲁棒性，有效利用未标记数据和跨模态信息。**

- **链接: [https://arxiv.org/pdf/2602.22917v1](https://arxiv.org/pdf/2602.22917v1)**

> **作者:** Hongzhao Li; Hao Dong; Hualei Wan; Shupan Li; Mingliang Xu; Muhammad Haris Khan
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Multimodal models ideally should generalize to unseen domains while remaining data-efficient to reduce annotation costs. To this end, we introduce and study a new problem, Semi-Supervised Multimodal Domain Generalization (SSMDG), which aims to learn robust multimodal models from multi-source data with few labeled samples. We observe that existing approaches fail to address this setting effectively: multimodal domain generalization methods cannot exploit unlabeled data, semi-supervised multimodal learning methods ignore domain shifts, and semi-supervised domain generalization methods are confined to single-modality inputs. To overcome these limitations, we propose a unified framework featuring three key components: Consensus-Driven Consistency Regularization, which obtains reliable pseudo-labels through confident fused-unimodal consensus; Disagreement-Aware Regularization, which effectively utilizes ambiguous non-consensus samples; and Cross-Modal Prototype Alignment, which enforces domain- and modality-invariant representations while promoting robustness under missing modalities via cross-modal translation. We further establish the first SSMDG benchmarks, on which our method consistently outperforms strong baselines in both standard and missing-modality scenarios. Our benchmarks and code are available at https://github.com/lihongzhao99/SSMDG.
>
---
#### [new 089] MediX-R1: Open Ended Medical Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文提出MediX-R1，一个用于医疗多模态大语言模型的强化学习框架，解决开放性医学问答问题。通过多信号奖励机制提升模型的临床推理能力。**

- **链接: [https://arxiv.org/pdf/2602.23363v1](https://arxiv.org/pdf/2602.23363v1)**

> **作者:** Sahal Shaji Mullappilly; Mohammed Irfan Kurpath; Omair Mohamed; Mohamed Zidan; Fahad Khan; Salman Khan; Rao Anwer; Hisham Cholakkal
>
> **摘要:** We introduce MediX-R1, an open-ended Reinforcement Learning (RL) framework for medical multimodal large language models (MLLMs) that enables clinically grounded, free-form answers beyond multiple-choice formats. MediX-R1 fine-tunes a baseline vision-language backbone with Group Based RL and a composite reward tailored for medical reasoning: an LLM-based accuracy reward that judges semantic correctness with a strict YES/NO decision, a medical embedding-based semantic reward to capture paraphrases and terminology variants, and lightweight format and modality rewards that enforce interpretable reasoning and modality recognition. This multi-signal design provides stable, informative feedback for open-ended outputs where traditional verifiable or MCQ-only rewards fall short. To measure progress, we propose a unified evaluation framework for both text-only and image+text tasks that uses a Reference-based LLM-as-judge in place of brittle string-overlap metrics, capturing semantic correctness, reasoning, and contextual alignment. Despite using only $\sim51$K instruction examples, MediX-R1 achieves excellent results across standard medical LLM (text-only) and VLM (image + text) benchmarks, outperforming strong open-source baselines and delivering particularly large gains on open-ended clinical tasks. Our results demonstrate that open-ended RL with comprehensive reward signals and LLM-based evaluation is a practical path toward reliable medical reasoning in multimodal models. Our trained models, curated datasets and source code are available at https://medix.cvmbzuai.com
>
---
#### [new 090] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于异常检测任务，解决工业视觉中少量正常样本下的异常检测问题。提出SubspaceAD方法，通过PCA建模正常特征子空间，利用重构残差检测异常。**

- **链接: [https://arxiv.org/pdf/2602.23013v1](https://arxiv.org/pdf/2602.23013v1)**

> **作者:** Camile Lendering; Erkut Akdag; Egor Bondarev
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Detecting visual anomalies in industrial inspection often requires training with only a few normal images per category. Recent few-shot methods achieve strong results employing foundation-model features, but typically rely on memory banks, auxiliary datasets, or multi-modal tuning of vision-language models. We therefore question whether such complexity is necessary given the feature representations of vision foundation models. To answer this question, we introduce SubspaceAD, a training-free method, that operates in two simple stages. First, patch-level features are extracted from a small set of normal images by a frozen DINOv2 backbone. Second, a Principal Component Analysis (PCA) model is fit to these features to estimate the low-dimensional subspace of normal variations. At inference, anomalies are detected via the reconstruction residual with respect to this subspace, producing interpretable and statistically grounded anomaly scores. Despite its simplicity, SubspaceAD achieves state-of-the-art performance across one-shot and few-shot settings without training, prompt tuning, or memory banks. In the one-shot anomaly detection setting, SubspaceAD achieves image-level and pixel-level AUROC of 98.0% and 97.6% on the MVTec-AD dataset, and 93.3% and 98.3% on the VisA dataset, respectively, surpassing prior state-of-the-art results. Code and demo are available at https://github.com/CLendering/SubspaceAD.
>
---
#### [new 091] TriLite: Efficient Weakly Supervised Object Localization with Universal Visual Features and Tri-Region Disentanglement
- **分类: cs.CV**

- **简介: 该论文提出TriLite，解决弱监督目标定位问题，通过冻结预训练ViT和最小参数的TriHead模块，提升定位效果与效率。**

- **链接: [https://arxiv.org/pdf/2602.23120v1](https://arxiv.org/pdf/2602.23120v1)**

> **作者:** Arian Sabaghi; José Oramas
>
> **备注:** This paper consists of 8 pages including 6 figures. Accepted at CVPR 2026
>
> **摘要:** Weakly supervised object localization (WSOL) aims to localize target objects in images using only image-level labels. Despite recent progress, many approaches still rely on multi-stage pipelines or full fine-tuning of large backbones, which increases training cost, while the broader WSOL community continues to face the challenge of partial object coverage. We present TriLite, a single-stage WSOL framework that leverages a frozen Vision Transformer with Dinov2 pre-training in a self-supervised manner, and introduces only a minimal number of trainable parameters (fewer than 800K on ImageNet-1K) for both classification and localization. At its core is the proposed TriHead module, which decomposes patch features into foreground, background, and ambiguous regions, thereby improving object coverage while suppressing spurious activations. By disentangling classification and localization objectives, TriLite effectively exploits the universal representations learned by self-supervised ViTs without requiring expensive end-to-end training. Extensive experiments on CUB-200-2011, ImageNet-1K, and OpenImages demonstrate that TriLite sets a new state of the art, while remaining significantly more parameter-efficient and easier to train than prior methods. The code will be released soon.
>
---
#### [new 092] Can Agents Distinguish Visually Hard-to-Separate Diseases in a Zero-Shot Setting? A Pilot Study
- **分类: cs.CV**

- **简介: 该论文属于医学图像诊断任务，旨在解决视觉相似疾病在零样本条件下的区分问题。通过构建多智能体框架，提升诊断准确性并减少误判。**

- **链接: [https://arxiv.org/pdf/2602.22959v1](https://arxiv.org/pdf/2602.22959v1)**

> **作者:** Zihao Zhao; Frederik Hauke; Juliana De Castilhos; Sven Nebelung; Daniel Truhn
>
> **备注:** Code available at https://github.com/TruhnLab/Contrastive-Agent-Reasoning
>
> **摘要:** The rapid progress of multimodal large language models (MLLMs) has led to increasing interest in agent-based systems. While most prior work in medical imaging concentrates on automating routine clinical workflows, we study an underexplored yet clinically significant setting: distinguishing visually hard-to-separate diseases in a zero-shot setting. We benchmark representative agents on two imaging-only proxy diagnostic tasks, (1) melanoma vs. atypical nevus and (2) pulmonary edema vs. pneumonia, where visual features are highly confounded despite substantial differences in clinical management. We introduce a multi-agent framework based on contrastive adjudication. Experimental results show improved diagnostic performance (an 11-percentage-point gain in accuracy on dermoscopy data) and reduced unsupported claims on qualitative samples, although overall performance remains insufficient for clinical deployment. We acknowledge the inherent uncertainty in human annotations and the absence of clinical context, which further limit the translation to real-world settings. Within this controlled setting, this pilot study provides preliminary insights into zero-shot agent performance in visually confounded scenarios.
>
---
#### [new 093] Instruction-based Image Editing with Planning, Reasoning, and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像编辑任务，旨在解决指令驱动的图像编辑质量受限问题。通过多模态模型提升场景理解与生成能力，提出分步的规划、推理和编辑方法。**

- **链接: [https://arxiv.org/pdf/2602.22624v1](https://arxiv.org/pdf/2602.22624v1)**

> **作者:** Liya Ji; Chenyang Qi; Qifeng Chen
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Editing images via instruction provides a natural way to generate interactive content, but it is a big challenge due to the higher requirement of scene understanding and generation. Prior work utilizes a chain of large language models, object segmentation models, and editing models for this task. However, the understanding models provide only a single modality ability, restricting the editing quality. We aim to bridge understanding and generation via a new multi-modality model that provides the intelligent abilities to instruction-based image editing models for more complex cases. To achieve this goal, we individually separate the instruction editing task with the multi-modality chain of thought prompts, i.e., Chain-of-Thought (CoT) planning, editing region reasoning, and editing. For Chain-of-Thought planning, the large language model could reason the appropriate sub-prompts considering the instruction provided and the ability of the editing network. For editing region reasoning, we train an instruction-based editing region generation network with a multi-modal large language model. Finally, a hint-guided instruction-based editing network is proposed for editing image generations based on the sizeable text-to-image diffusion model to accept the hints for generation. Extensive experiments demonstrate that our method has competitive editing abilities on complex real-world images.
>
---
#### [new 094] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes
- **分类: cs.CV**

- **简介: 该论文属于室内场景的开放词汇占据预测任务，旨在解决室内环境复杂、语义细粒度的问题。通过几何监督和语言对齐方法，提升占据预测效果。**

- **链接: [https://arxiv.org/pdf/2602.22667v1](https://arxiv.org/pdf/2602.22667v1)**

> **作者:** Changqing Zhou; Yueru Luo; Han Zhang; Zeyu Jiang; Changhao Chen
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Open-vocabulary 3D occupancy is vital for embodied agents, which need to understand complex indoor environments where semantic categories are abundant and evolve beyond fixed taxonomies. While recent work has explored open-vocabulary occupancy in outdoor driving scenarios, such methods transfer poorly indoors, where geometry is denser, layouts are more intricate, and semantics are far more fine-grained. To address these challenges, we adopt a geometry-only supervision paradigm that uses only binary occupancy labels (occupied vs free). Our framework builds upon 3D Language-Embedded Gaussians, which serve as a unified intermediate representation coupling fine-grained 3D geometry with a language-aligned semantic embedding. On the geometry side, we find that existing Gaussian-to-Occupancy operators fail to converge under such weak supervision, and we introduce an opacity-aware, Poisson-based approach that stabilizes volumetric aggregation. On the semantic side, direct alignment between rendered features and open-vocabulary segmentation features suffers from feature mixing; we therefore propose a Progressive Temperature Decay schedule that gradually sharpens opacities during splatting, strengthening Gaussian-language alignment. On Occ-ScanNet, our framework achieves 59.50 IoU and 21.05 mIoU in the open-vocabulary setting, surpassing all existing occupancy methods in IoU and outperforming prior open-vocabulary approaches by a large margin in mIoU. Code will be released at https://github.com/JuIvyy/LegoOcc.
>
---
#### [new 095] Pix2Key: Controllable Open-Vocabulary Retrieval with Semantic Decomposition and Self-Supervised Visual Dictionary Learning
- **分类: cs.CV**

- **简介: 该论文提出Pix2Key，解决图像检索中的可控制开放词汇检索问题。通过视觉词典表示和自监督学习，提升检索精度与多样性。**

- **链接: [https://arxiv.org/pdf/2602.22510v1](https://arxiv.org/pdf/2602.22510v1)**

> **作者:** Guoyizhe Wei; Yang Jiao; Nan Xi; Zhishen Huang; Jingjing Meng; Rama Chellappa; Yan Gao
>
> **摘要:** Composed Image Retrieval (CIR) uses a reference image plus a natural-language edit to retrieve images that apply the requested change while preserving other relevant visual content. Classic fusion pipelines typically rely on supervised triplets and can lose fine-grained cues, while recent zero-shot approaches often caption the reference image and merge the caption with the edit, which may miss implicit user intent and return repetitive results. We present Pix2Key, which represents both queries and candidates as open-vocabulary visual dictionaries, enabling intent-aware constraint matching and diversity-aware reranking in a unified embedding space. A self-supervised pretraining component, V-Dict-AE, further improves the dictionary representation using only images, strengthening fine-grained attribute understanding without CIR-specific supervision. On the DFMM-Compose benchmark, Pix2Key improves Recall@10 up to 3.2 points, and adding V-Dict-AE yields an additional 2.3-point gain while improving intent consistency and maintaining high list diversity.
>
---
#### [new 096] VGG-T$^3$: Offline Feed-Forward 3D Reconstruction at Scale
- **分类: cs.CV**

- **简介: 该论文提出VGG-T$^3$，解决3D重建中在线方法计算成本高的问题，通过固定大小MLP实现线性扩展，提升重建效率与精度。**

- **链接: [https://arxiv.org/pdf/2602.23361v1](https://arxiv.org/pdf/2602.23361v1)**

> **作者:** Sven Elflein; Ruilong Li; Sérgio Agostinho; Zan Gojcic; Laura Leal-Taixé; Qunjie Zhou; Aljosa Osep
>
> **备注:** CVPR 2026, Project page: https://research.nvidia.com/labs/dvl/projects/vgg-ttt
>
> **摘要:** We present a scalable 3D reconstruction model that addresses a critical limitation in offline feed-forward methods: their computational and memory requirements grow quadratically w.r.t. the number of input images. Our approach is built on the key insight that this bottleneck stems from the varying-length Key-Value (KV) space representation of scene geometry, which we distill into a fixed-size Multi-Layer Perceptron (MLP) via test-time training. VGG-T$^3$ (Visual Geometry Grounded Test Time Training) scales linearly w.r.t. the number of input views, similar to online models, and reconstructs a $1k$ image collection in just $54$ seconds, achieving a $11.6\times$ speed-up over baselines that rely on softmax attention. Since our method retains global scene aggregation capability, our point map reconstruction error outperforming other linear-time methods by large margins. Finally, we demonstrate visual localization capabilities of our model by querying the scene representation with unseen images.
>
---
#### [new 097] SimpleOCR: Rendering Visualized Questions to Teach MLLMs to Read
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言理解任务，旨在解决MLLMs是否真正“阅读”图像中文字的问题。通过引入VQ设置和SimpleOCR方法，提升模型的视觉文本识别能力。**

- **链接: [https://arxiv.org/pdf/2602.22426v1](https://arxiv.org/pdf/2602.22426v1)**

> **作者:** Yibo Peng; Peng Xia; Ding Zhong; Kaide Zeng; Siwei Han; Yiyang Zhou; Jiaqi Liu; Ruiyi Zhang; Huaxiu Yao
>
> **摘要:** Despite the rapid advancements in Multimodal Large Language Models (MLLMs), a critical question regarding their visual grounding mechanism remains unanswered: do these models genuinely ``read'' text embedded in images, or do they merely rely on parametric shortcuts in the text prompt? In this work, we diagnose this issue by introducing the Visualized-Question (VQ) setting, where text queries are rendered directly onto images to structurally mandate visual engagement. Our diagnostic experiments on Qwen2.5-VL reveal a startling capability-utilization gap: despite possessing strong OCR capabilities, models suffer a performance degradation of up to 12.7% in the VQ setting, exposing a deep-seated ``modality laziness.'' To bridge this gap, we propose SimpleOCR, a plug-and-play training strategy that imposes a structural constraint on the learning process. By transforming training samples into the VQ format with randomized styles, SimpleOCR effectively invalidates text-based shortcuts, compelling the model to activate and optimize its visual text extraction pathways. Empirically, SimpleOCR yields robust gains without architectural modifications. On four representative OOD benchmarks, it surpasses the base model by 5.4% and GRPO based on original images by 2.7%, while exhibiting extreme data efficiency, achieving superior performance with 30x fewer samples (8.5K) than recent RL-based methods. Furthermore, its plug-and-play nature allows seamless integration with advanced RL strategies like NoisyRollout to yield complementary improvements. Code is available at https://github.com/aiming-lab/SimpleOCR.
>
---
#### [new 098] Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决模型对分布外退化情况泛化能力差的问题。提出BaryIR框架，通过Wasserstein barycenter空间学习通用特征表示。**

- **链接: [https://arxiv.org/pdf/2602.23169v1](https://arxiv.org/pdf/2602.23169v1)**

> **作者:** Xiaole Tang; Xiaoyi He; Jiayi Xu; Xiang Gu; Jian Sun
>
> **摘要:** Despite substantial advances in all-in-one image restoration for addressing diverse degradations within a unified model, existing methods remain vulnerable to out-of-distribution degradations, thereby limiting their generalization in real-world scenarios. To tackle the challenge, this work is motivated by the intuition that multisource degraded feature distributions are induced by different degradation-specific shifts from an underlying degradation-agnostic distribution, and recovering such a shared distribution is thus crucial for achieving generalization across degradations. With this insight, we propose BaryIR, a representation learning framework that aligns multisource degraded features in the Wasserstein barycenter (WB) space, which models a degradation-agnostic distribution by minimizing the average of Wasserstein distances to multisource degraded distributions. We further introduce residual subspaces, whose embeddings are mutually contrasted while remaining orthogonal to the WB embeddings. Consequently, BaryIR explicitly decouples two orthogonal spaces: a WB space that encodes the degradation-agnostic invariant contents shared across degradations, and residual subspaces that adaptively preserve the degradation-specific knowledge. This disentanglement mitigates overfitting to in-distribution degradations and enables adaptive restoration grounded on the degradation-agnostic shared invariance. Extensive experiments demonstrate that BaryIR performs competitively against state-of-the-art all-in-one methods. Notably, BaryIR generalizes well to unseen degradations (\textit{e.g.,} types and levels) and shows remarkable robustness in learning generalized features, even when trained on limited degradation types and evaluated on real-world data with mixed degradations.
>
---
#### [new 099] Vision Transformers Need More Than Registers
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉领域，旨在解决Vision Transformers中的冗余特征问题。通过分析发现其依赖无关背景块导致性能下降，提出改进方法提升模型表现。**

- **链接: [https://arxiv.org/pdf/2602.22394v1](https://arxiv.org/pdf/2602.22394v1)**

> **作者:** Cheng Shi; Yizhou Yu; Sibei Yang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Vision Transformers (ViTs), when pre-trained on large-scale data, provide general-purpose representations for diverse downstream tasks. However, artifacts in ViTs are widely observed across different supervision paradigms and downstream tasks. Through systematic analysis of artifacts in ViTs, we find that their fundamental mechanisms have yet to be sufficiently elucidated. In this paper, through systematic analysis, we conclude that these artifacts originate from a lazy aggregation behavior: ViT uses semantically irrelevant background patches as shortcuts to represent global semantics, driven by global attention and Coarse-grained semantic supervision. Our solution selectively integrates patch features into the CLS token, reducing the influence of background-dominated shortcuts and consistently improving performance across 12 benchmarks under label-, text-, and self-supervision. We hope this work offers a new perspective on ViT behavior.
>
---
#### [new 100] ViCLIP-OT: The First Foundation Vision-Language Model for Vietnamese Image-Text Retrieval with Optimal Transport
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像-文本检索任务，针对越南语等低资源语言的跨模态对齐问题，提出ViCLIP-OT模型，结合对比学习与最优传输损失，提升检索性能。**

- **链接: [https://arxiv.org/pdf/2602.22678v1](https://arxiv.org/pdf/2602.22678v1)**

> **作者:** Quoc-Khang Tran; Minh-Thien Nguyen; Nguyen-Khang Pham
>
> **备注:** Preprint submitted to Expert Systems with Applications
>
> **摘要:** Image-text retrieval has become a fundamental component in intelligent multimedia systems; however, most existing vision-language models are optimized for highresource languages and remain suboptimal for low-resource settings such as Vietnamese. This work introduces ViCLIP-OT, a foundation vision-language model specifically designed for Vietnamese image-text retrieval. The proposed framework integrates CLIP-style contrastive learning with a Similarity-Graph Regularized Optimal Transport (SIGROT) loss to enhance global cross-modal consistency and mitigate modality gap issues. Extensive experiments on three Vietnamese benchmarks (UITOpenViIC, KTVIC, and Crossmodal-3600) demonstrate that ViCLIP-OT consistently outperforms CLIP and SigLIP baselines in both in-domain and zero-shot settings. On UIT-OpenViIC, the model achieves an average Recall@K of 67.34%, improving upon CLIP by 5.75 percentage points. In zero-shot evaluation on Crossmodal-3600, ViCLIPOT surpasses CLIP by 11.72 percentage points. Embedding-space analysis further confirms improved alignment and reduced modality gap. The results indicate that integrating SIGROT provides an effective and scalable strategy for cross-modal retrieval in low-resource languages, offering practical implications for intelligent multimedia retrieval systems in Vietnamese and other underrepresented linguistic contexts.
>
---
#### [new 101] Cytoarchitecture in Words: Weakly Supervised Vision-Language Modeling for Human Brain Microscopy
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言建模任务，解决医学图像与自然语言描述不匹配的问题。通过标签生成合成文本，将视觉模型与语言模型结合，实现脑区的自然语言描述。**

- **链接: [https://arxiv.org/pdf/2602.23088v1](https://arxiv.org/pdf/2602.23088v1)**

> **作者:** Matthew Sutton; Katrin Amunts; Timo Dickscheid; Christian Schiffer
>
> **备注:** 8 pages, 3 figures, submitted for inclusion at a conference
>
> **摘要:** Foundation models increasingly offer potential to support interactive, agentic workflows that assist researchers during analysis and interpretation of image data. Such workflows often require coupling vision to language to provide a natural-language interface. However, paired image-text data needed to learn this coupling are scarce and difficult to obtain in many research and clinical settings. One such setting is microscopic analysis of cell-body-stained histological human brain sections, which enables the study of cytoarchitecture: cell density and morphology and their laminar and areal organization. Here, we propose a label-mediated method that generates meaningful captions from images by linking images and text only through a label, without requiring curated paired image-text data. Given the label, we automatically mine area descriptions from related literature and use them as synthetic captions reflecting canonical cytoarchitectonic attributes. An existing cytoarchitectonic vision foundation model (CytoNet) is then coupled to a large language model via an image-to-text training objective, enabling microscopy regions to be described in natural language. Across 57 brain areas, the resulting method produces plausible area-level descriptions and supports open-set use through explicit rejection of unseen areas. It matches the cytoarchitectonic reference label for in-scope patches with 90.6% accuracy and, with the area label masked, its descriptions remain discriminative enough to recover the area in an 8-way test with 68.6% accuracy. These results suggest that weak, label-mediated pairing can suffice to connect existing biomedical vision foundation models to language, providing a practical recipe for integrating natural-language in domains where fine-grained paired annotations are scarce.
>
---
#### [new 102] OSDaR-AR: Enhancing Railway Perception Datasets via Multi-modal Augmented Reality
- **分类: cs.CV**

- **简介: 该论文属于铁路感知任务，旨在解决数据稀缺与真实感不足的问题。通过多模态增强现实技术，将虚拟物体融入真实铁路场景，提升数据集的逼真度和实用性。**

- **链接: [https://arxiv.org/pdf/2602.22920v1](https://arxiv.org/pdf/2602.22920v1)**

> **作者:** Federico Nesti; Gianluca D'Amico; Mauro Marinoni; Giorgio Buttazzo
>
> **摘要:** Although deep learning has significantly advanced the perception capabilities of intelligent transportation systems, railway applications continue to suffer from a scarcity of high-quality, annotated data for safety-critical tasks like obstacle detection. While photorealistic simulators offer a solution, they often struggle with the ``sim-to-real" gap; conversely, simple image-masking techniques lack the spatio-temporal coherence required to obtain augmented single- and multi-frame scenes with the correct appearance and dimensions. This paper introduces a multi-modal augmented reality framework designed to bridge this gap by integrating photorealistic virtual objects into real-world railway sequences from the OSDaR23 dataset. Utilizing Unreal Engine 5 features, our pipeline leverages LiDAR point-clouds and INS/GNSS data to ensure accurate object placement and temporal stability across RGB frames. This paper also proposes a segmentation-based refinement strategy for INS/GNSS data to significantly improve the realism of the augmented sequences, as confirmed by the comparative study presented in the paper. Carefully designed augmented sequences are collected to produce OSDaR-AR, a public dataset designed to support the development of next-generation railway perception systems. The dataset is available at the following page: https://syndra.retis.santannapisa.it/osdarar.html
>
---
#### [new 103] Exploring Multimodal LMMs for Online Episodic Memory Question Answering on the Edge
- **分类: cs.CV**

- **简介: 该论文属于在线情景记忆问答任务，旨在解决可穿戴设备隐私和延迟问题。通过在边缘设备上部署多模态大语言模型，实现高效实时问答。**

- **链接: [https://arxiv.org/pdf/2602.22455v1](https://arxiv.org/pdf/2602.22455v1)**

> **作者:** Giuseppe Lando; Rosario Forte; Antonino Furnari
>
> **摘要:** We investigate the feasibility of using Multimodal Large Language Models (MLLMs) for real-time online episodic memory question answering. While cloud offloading is common, it raises privacy and latency concerns for wearable assistants, hence we investigate implementation on the edge. We integrated streaming constraints into our question answering pipeline, which is structured into two asynchronous threads: a Descriptor Thread that continuously converts video into a lightweight textual memory, and a Question Answering (QA) Thread that reasons over the textual memory to answer queries. Experiments on the QAEgo4D-Closed benchmark analyze the performance of Multimodal Large Language Models (MLLMs) within strict resource boundaries, showing promising results also when compared to clound-based solutions. Specifically, an end-to-end configuration running on a consumer-grade 8GB GPU achieves 51.76% accuracy with a Time-To-First-Token (TTFT) of 0.41s. Scaling to a local enterprise-grade server yields 54.40% accuracy with a TTFT of 0.88s. In comparison, a cloud-based solution obtains an accuracy of 56.00%. These competitive results highlight the potential of edge-based solutions for privacy-preserving episodic memory retrieval.
>
---
#### [new 104] Through BrokenEyes: How Eye Disorders Impact Face Detection?
- **分类: cs.CV**

- **简介: 该论文属于视觉感知研究任务，探讨眼疾对人脸识别的影响。通过模拟五种眼病，分析其对深度学习特征表示的干扰，揭示视觉退化与模型表现的关系。**

- **链接: [https://arxiv.org/pdf/2602.23212v1](https://arxiv.org/pdf/2602.23212v1)**

> **作者:** Prottay Kumar Adhikary
>
> **摘要:** Vision disorders significantly impact millions of lives, altering how visual information is processed and perceived. In this work, a computational framework was developed using the BrokenEyes system to simulate five common eye disorders: Age-related macular degeneration, cataract, glaucoma, refractive errors, and diabetic retinopathy and analyze their effects on neural-like feature representations in deep learning models. Leveraging a combination of human and non-human datasets, models trained under normal and disorder-specific conditions revealed critical disruptions in feature maps, particularly for cataract and glaucoma, which align with known neural processing challenges in these conditions. Evaluation metrics such as activation energy and cosine similarity quantified the severity of these distortions, providing insights into the interplay between degraded visual inputs and learned representations.
>
---
#### [new 105] Multidimensional Task Learning: A Unified Tensor Framework for Computer Vision Tasks
- **分类: cs.CV; math.NA**

- **简介: 该论文提出多维任务学习框架MTL，解决计算机视觉任务表达受限问题。通过张量操作替代矩阵，扩展任务空间，支持更复杂的任务配置。**

- **链接: [https://arxiv.org/pdf/2602.23217v1](https://arxiv.org/pdf/2602.23217v1)**

> **作者:** Alaa El Ichi; Khalide Jbilou
>
> **摘要:** This paper introduces Multidimensional Task Learning (MTL), a unified mathematical framework based on Generalized Einstein MLPs (GE-MLPs) that operate directly on tensors via the Einstein product. We argue that current computer vision task formulations are inherently constrained by matrix-based thinking: standard architectures rely on matrix-valued weights and vectorvalued biases, requiring structural flattening that restricts the space of naturally expressible tasks. GE-MLPs lift this constraint by operating with tensor-valued parameters, enabling explicit control over which dimensions are preserved or contracted without information loss. Through rigorous mathematical derivations, we demonstrate that classification, segmentation, and detection are special cases of MTL, differing only in their dimensional configuration within a formally defined task space. We further prove that this task space is strictly larger than what matrix-based formulations can natively express, enabling principled task configurations such as spatiotemporal or cross modal predictions that require destructive flattening under conventional approaches. This work provides a mathematical foundation for understanding, comparing, and designing computer vision tasks through the lens of tensor algebra.
>
---
#### [new 106] UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception
- **分类: cs.CV; cs.RO**

- **简介: 论文提出UniScale，用于机器人感知的多视角3D重建任务，解决环境结构准确提取问题。通过统一模型联合估计相机参数、深度和场景尺度，结合几何先验提升性能。**

- **链接: [https://arxiv.org/pdf/2602.23224v1](https://arxiv.org/pdf/2602.23224v1)**

> **作者:** Mohammad Mahdavian; Gordon Tan; Binbin Xu; Yuan Ren; Dongfeng Bai; Bingbing Liu
>
> **摘要:** We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.
>
---
#### [new 107] Enabling clinical use of foundation models in histopathology
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算病理学任务，旨在解决基础模型在临床应用中的鲁棒性问题。通过引入稳健损失函数，提升模型对技术变异的抗干扰能力，提高预测准确性。**

- **链接: [https://arxiv.org/pdf/2602.22347v1](https://arxiv.org/pdf/2602.22347v1)**

> **作者:** Audun L. Henriksen; Ole-Johan Skrede; Lisa van der Schee; Enric Domingo; Sepp De Raedt; Ilyá Kostolomov; Jennifer Hay; Karolina Cyll; Wanja Kildal; Joakim Kalsnes; Robert W. Williams; Manohar Pradhan; John Arne Nesheim; Hanne A. Askautrud; Maria X. Isaksen; Karmele Saez de Gordoa; Miriam Cuatrecasas; Joanne Edwards; TransSCOT group; Arild Nesbakken; Neil A. Shepherd; Ian Tomlinson; Daniel-Christoph Wagner; Rachel S. Kerr; Tarjei Sveinsgjerd Hveem; Knut Liestøl; Yoshiaki Nakamura; Marco Novelli; Masaaki Miyo; Sebastian Foersch; David N. Church; Miangela M. Lacle; David J. Kerr; Andreas Kleppe
>
> **摘要:** Foundation models in histopathology are expected to facilitate the development of high-performing and generalisable deep learning systems. However, current models capture not only biologically relevant features, but also pre-analytic and scanner-specific variation that bias the predictions of task-specific models trained from the foundation model features. Here we show that introducing novel robustness losses during training of downstream task-specific models reduces sensitivity to technical variability. A purpose-designed comprehensive experimentation setup with 27,042 WSIs from 6155 patients is used to train thousands of models from the features of eight popular foundation models for computational pathology. In addition to a substantial improvement in robustness, we observe that prediction accuracy improves by focusing on biologically relevant features. Our approach successfully mitigates robustness issues of foundation models for computational pathology without retraining the foundation models themselves, enabling development of robust computational pathology models applicable to real-world data in routine clinical practice.
>
---
#### [new 108] GeoWorld: Geometric World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GeoWorld，解决视觉规划中的长期预测和几何结构保留问题。通过超球面JEPA和几何强化学习，提升多步规划效果。**

- **链接: [https://arxiv.org/pdf/2602.23058v1](https://arxiv.org/pdf/2602.23058v1)**

> **作者:** Zeyu Zhang; Danning Li; Ian Reid; Richard Hartley
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Energy-based predictive world models provide a powerful approach for multi-step visual planning by reasoning over latent energy landscapes rather than generating pixels. However, existing approaches face two major challenges: (i) their latent representations are typically learned in Euclidean space, neglecting the underlying geometric and hierarchical structure among states, and (ii) they struggle with long-horizon prediction, which leads to rapid degradation across extended rollouts. To address these challenges, we introduce GeoWorld, a geometric world model that preserves geometric structure and hierarchical relations through a Hyperbolic JEPA, which maps latent representations from Euclidean space onto hyperbolic manifolds. We further introduce Geometric Reinforcement Learning for energy-based optimization, enabling stable multi-step planning in hyperbolic latent space. Extensive experiments on CrossTask and COIN demonstrate around 3% SR improvement in 3-step planning and 2% SR improvement in 4-step planning compared to the state-of-the-art V-JEPA 2. Project website: https://steve-zeyu-zhang.github.io/GeoWorld.
>
---
#### [new 109] GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views
- **分类: cs.CV**

- **简介: 该论文提出GIFSplat，解决从稀疏视角进行高效3D重建的问题。通过迭代优化和生成先验引导，提升重建质量与效率。**

- **链接: [https://arxiv.org/pdf/2602.22571v1](https://arxiv.org/pdf/2602.22571v1)**

> **作者:** Tianyu Chen; Wei Xiang; Kang Han; Yu Lu; Di Wu; Gaowen Liu; Ramana Rao Kompella
>
> **摘要:** Feed-forward 3D reconstruction offers substantial runtime advantages over per-scene optimization, which remains slow at inference and often fragile under sparse views. However, existing feed-forward methods still have potential for further performance gains, especially for out-of-domain data, and struggle to retain second-level inference time once a generative prior is introduced. These limitations stem from the one-shot prediction paradigm in existing feed-forward pipeline: models are strictly bounded by capacity, lack inference-time refinement, and are ill-suited for continuously injecting generative priors. We introduce GIFSplat, a purely feed-forward iterative refinement framework for 3D Gaussian Splatting from sparse unposed views. A small number of forward-only residual updates progressively refine current 3D scene using rendering evidence, achieve favorable balance between efficiency and quality. Furthermore, we distill a frozen diffusion prior into Gaussian-level cues from enhanced novel renderings without gradient backpropagation or ever-increasing view-set expansion, thereby enabling per-scene adaptation with generative prior while preserving feed-forward efficiency. Across DL3DV, RealEstate10K, and DTU, GIFSplat consistently outperforms state-of-the-art feed-forward baselines, improving PSNR by up to +2.1 dB, and it maintains second-scale inference time without requiring camera poses or any test-time gradient optimization.
>
---
#### [new 110] ManifoldGD: Training-Free Hierarchical Manifold Guidance for Diffusion-Based Dataset Distillation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出ManifoldGD，属于数据蒸馏任务，旨在通过无训练的扩散模型生成紧凑且具有代表性的数据集，解决传统方法在语义一致性和多样性上的不足。**

- **链接: [https://arxiv.org/pdf/2602.23295v1](https://arxiv.org/pdf/2602.23295v1)**

> **作者:** Ayush Roy; Wei-Yang Alex Lee; Rudrasis Chakraborty; Vishnu Suresh Lokhande
>
> **备注:** CVPE 2026
>
> **摘要:** In recent times, large datasets hinder efficient model training while also containing redundant concepts. Dataset distillation aims to synthesize compact datasets that preserve the knowledge of large-scale training sets while drastically reducing storage and computation. Recent advances in diffusion models have enabled training-free distillation by leveraging pre-trained generative priors; however, existing guidance strategies remain limited. Current score-based methods either perform unguided denoising or rely on simple mode-based guidance toward instance prototype centroids (IPC centroids), which often are rudimentary and suboptimal. We propose Manifold-Guided Distillation (ManifoldGD), a training-free diffusion-based framework that integrates manifold consistent guidance at every denoising timestep. Our method employs IPCs computed via a hierarchical, divisive clustering of VAE latent features, yielding a multi-scale coreset of IPCs that captures both coarse semantic modes and fine intra-class variability. Using a local neighborhood of the extracted IPC centroids, we create the latent manifold for each diffusion denoising timestep. At each denoising step, we project the mode-alignment vector onto the local tangent space of the estimated latent manifold, thus constraining the generation trajectory to remain manifold-faithful while preserving semantic consistency. This formulation improves representativeness, diversity, and image fidelity without requiring any model retraining. Empirical results demonstrate consistent gains over existing training-free and training-based baselines in terms of FID, l2 distance among real and synthetic dataset embeddings, and classification accuracy, establishing ManifoldGD as the first geometry-aware training-free data distillation framework.
>
---
#### [new 111] CRAG: Can 3D Generative Models Help 3D Assembly?
- **分类: cs.CV**

- **简介: 该论文属于3D装配任务，旨在解决传统方法仅依赖姿态估计而无法处理缺失几何的问题。工作上提出CRAG模型，联合生成与装配，同时生成完整形状并预测部件姿态。**

- **链接: [https://arxiv.org/pdf/2602.22629v1](https://arxiv.org/pdf/2602.22629v1)**

> **作者:** Zeyu Jiang; Sihang Li; Siqi Tan; Chenyang Xu; Juexiao Zhang; Julia Galway-Witham; Xue Wang; Scott A. Williams; Radu Iovita; Chen Feng; Jing Zhang
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Most existing 3D assembly methods treat the problem as pure pose estimation, rearranging observed parts via rigid transformations. In contrast, human assembly naturally couples structural reasoning with holistic shape inference. Inspired by this intuition, we reformulate 3D assembly as a joint problem of assembly and generation. We show that these two processes are mutually reinforcing: assembly provides part-level structural priors for generation, while generation injects holistic shape context that resolves ambiguities in assembly. Unlike prior methods that cannot synthesize missing geometry, we propose CRAG, which simultaneously generates plausible complete shapes and predicts poses for input parts. Extensive experiments demonstrate state-of-the-art performance across in-the-wild objects with diverse geometries, varying part counts, and missing pieces. Our code and models will be released.
>
---
#### [new 112] SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D多模态任务，旨在解决3D LVLMs中位置建模不足的问题。提出SoPE方法，通过球坐标映射提升空间感知，增强几何表示能力。**

- **链接: [https://arxiv.org/pdf/2602.22716v1](https://arxiv.org/pdf/2602.22716v1)**

> **作者:** Guanting Ye; Qiyan Zhao; Wenhao Yu; Liangyu Yuan; Mingkai Li; Xiaofeng Zhang; Jianmin Ji; Yanyong Zhang; Qing Jiang; Ka-Veng Yuen
>
> **备注:** CVPR 2026
>
> **摘要:** 3D Large Vision-Language Models (3D LVLMs) built upon Large Language Models (LLMs) have achieved remarkable progress across various multimodal tasks. However, their inherited position-dependent modeling mechanism, Rotary Position Embedding (RoPE), remains suboptimal for 3D multimodal understanding. The vanilla RoPE formulation fails to preserve essential three-dimensional spatial structures when encoding 3D tokens, and its relative distance computation overlooks angular dependencies, hindering the model's ability to capture directional variations in visual representations. To overcome these limitations, we introduce Spherical Coordinate-based Positional Embedding (SoPE). Our method maps point-cloud token indices into a 3D spherical coordinate space, enabling unified modeling of spatial locations and directional angles. This formulation preserves the inherent geometric structure of point-cloud data, enhances spatial awareness, and yields more consistent and expressive geometric representations for multimodal learning. In addition, we introduce a multi-scale frequency mixing strategy to fuse feature information across different frequency domains. Experimental results on multiple 3D scene benchmarks validate the effectiveness of our approach, while real-world deployment experiments further demonstrate its strong generalization capability.
>
---
#### [new 113] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于Referring Image Segmentation任务，旨在提升图像中由自然语言描述定位物体的准确性。提出AML方法，通过像素级视觉-语言对齐增强模型性能。**

- **链接: [https://arxiv.org/pdf/2602.22740v1](https://arxiv.org/pdf/2602.22740v1)**

> **作者:** Tongfei Chen; Shuo Yang; Yuguang Yang; Linlin Yang; Runtang Guo; Changbai Li; He Long; Chunyu Xie; Dawei Leng; Baochang Zhang
>
> **备注:** ICLR 2026 conference paper
>
> **摘要:** Referring Image Segmentation (RIS) aims to segment an object in an image identified by a natural language expression. The paper introduces Alignment-Aware Masked Learning (AML), a training strategy to enhance RIS by explicitly estimating pixel-level vision-language alignment, filtering out poorly aligned regions during optimization, and focusing on trustworthy cues. This approach results in state-of-the-art performance on RefCOCO datasets and also enhances robustness to diverse descriptions and scenarios
>
---
#### [new 114] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，解决3D布局下物体遮挡建模问题。提出SeeThrough3D模型，通过透明3D框和视觉令牌实现精确遮挡与相机控制。**

- **链接: [https://arxiv.org/pdf/2602.23359v1](https://arxiv.org/pdf/2602.23359v1)**

> **作者:** Vaibhav Agrawal; Rishubh Parihar; Pradhaan Bhat; Ravi Kiran Sarvadevabhatla; R. Venkatesh Babu
>
> **备注:** Project page: https://seethrough3d.github.io. Accepted at CVPR 2026
>
> **摘要:** We identify occlusion reasoning as a fundamental yet overlooked aspect for 3D layout-conditioned generation. It is essential for synthesizing partially occluded objects with depth-consistent geometry and scale. While existing methods can generate realistic scenes that follow input layouts, they often fail to model precise inter-object occlusions. We propose SeeThrough3D, a model for 3D layout conditioned generation that explicitly models occlusions. We introduce an occlusion-aware 3D scene representation (OSCR), where objects are depicted as translucent 3D boxes placed within a virtual environment and rendered from desired camera viewpoint. The transparency encodes hidden object regions, enabling the model to reason about occlusions, while the rendered viewpoint provides explicit camera control during generation. We condition a pretrained flow based text-to-image image generation model by introducing a set of visual tokens derived from our rendered 3D representation. Furthermore, we apply masked self-attention to accurately bind each object bounding box to its corresponding textual description, enabling accurate generation of multiple objects without object attribute mixing. To train the model, we construct a synthetic dataset with diverse multi-object scenes with strong inter-object occlusions. SeeThrough3D generalizes effectively to unseen object categories and enables precise 3D layout control with realistic occlusions and consistent camera control.
>
---
#### [new 115] Scaling Audio-Visual Quality Assessment Dataset via Crowdsourcing
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于音频视频质量评估任务，旨在解决现有数据集规模小、多样性不足的问题。通过众包方式构建大规模、多样化的AVQA数据集，提升模型训练和多模态研究支持。**

- **链接: [https://arxiv.org/pdf/2602.22659v1](https://arxiv.org/pdf/2602.22659v1)**

> **作者:** Renyu Yang; Jian Jin; Lili Meng; Meiqin Liu; Yilin Wang; Balu Adsumilli; Weisi Lin
>
> **备注:** Accepted to ICASSP 2026. 5 pages (main paper) + 8 pages (supplementary material)
>
> **摘要:** Audio-visual quality assessment (AVQA) research has been stalled by limitations of existing datasets: they are typically small in scale, with insufficient diversity in content and quality, and annotated only with overall scores. These shortcomings provide limited support for model development and multimodal perception research. We propose a practical approach for AVQA dataset construction. First, we design a crowdsourced subjective experiment framework for AVQA, breaks the constraints of in-lab settings and achieves reliable annotation across varied environments. Second, a systematic data preparation strategy is further employed to ensure broad coverage of both quality levels and semantic scenarios. Third, we extend the dataset with additional annotations, enabling research on multimodal perception mechanisms and their relation to content. Finally, we validate this approach through YT-NTU-AVQ, the largest and most diverse AVQA dataset to date, consisting of 1,620 user-generated audio and video (A/V) sequences. The dataset and platform code are available at https://github.com/renyu12/YT-NTU-AVQ
>
---
#### [new 116] FairQuant: Fairness-Aware Mixed-Precision Quantization for Medical Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医疗图像分类任务，旨在解决量化压缩中算法公平性问题。提出FairQuant框架，实现混合精度量化与公平性优化。**

- **链接: [https://arxiv.org/pdf/2602.23192v1](https://arxiv.org/pdf/2602.23192v1)**

> **作者:** Thomas Woergaard; Raghavendra Selvan
>
> **备注:** Source code available at https://github.com/saintslab/FairQuant
>
> **摘要:** Compressing neural networks by quantizing model parameters offers useful trade-off between performance and efficiency. Methods like quantization-aware training and post-training quantization strive to maintain the downstream performance of compressed models compared to the full precision models. However, these techniques do not explicitly consider the impact on algorithmic fairness. In this work, we study fairness-aware mixed-precision quantization schemes for medical image classification under explicit bit budgets. We introduce FairQuant, a framework that combines group-aware importance analysis, budgeted mixed-precision allocation, and a learnable Bit-Aware Quantization (BAQ) mode that jointly optimizes weights and per-unit bit allocations under bitrate and fairness regularization. We evaluate the method on Fitzpatrick17k and ISIC2019 across ResNet18/50, DeiT-Tiny, and TinyViT. Results show that FairQuant configurations with average precision near 4-6 bits recover much of the Uniform 8-bit accuracy while improving worst-group performance relative to Uniform 4- and 8-bit baselines, with comparable fairness metrics under shared budgets.
>
---
#### [new 117] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型中的幻觉消除任务，旨在解决模型生成虚假内容的问题。提出HulluEdit框架，通过单次传递的子空间编辑有效抑制幻觉，同时保留视觉基础。**

- **链接: [https://arxiv.org/pdf/2602.22727v1](https://arxiv.org/pdf/2602.22727v1)**

> **作者:** Yangguang Lin; Quan Fang; Yufei Li; Jiachen Sun; Junyu Gao; Jitao Sang
>
> **备注:** accepted at CVPR 2026
>
> **摘要:** Object hallucination in Large Vision-Language Models (LVLMs) significantly hinders their reliable deployment. Existing methods struggle to balance efficiency and accuracy: they often require expensive reference models and multiple forward passes, or apply static edits that risk suppressing genuine visual evidence. To address this, we introduce HulluEdit, a single-pass, reference-free intervention framework. Our core innovation is orthogonal subspace editing: we decompose the hidden states of the model into orthogonal subspaces - visual evidence, conflicting priors, and residual uncertainty - enabling selective suppression of hallucinatory patterns without interfering with visual grounding. This approach mathematically guarantees that edits applied to the prior subspace leave the visual component entirely unaffected. Extensive experiments show that HulluEdit achieves state-of-the-art hallucination reduction on benchmarks including POPE and CHAIR across diverse architectures, while preserving general capabilities on MME and maintaining efficient inference. Our method consistently outperforms contrastive decoding and static subspace editing baselines, offering a new pathway toward more trustworthy LVLMs.
>
---
#### [new 118] Quality-Aware Robust Multi-View Clustering for Heterogeneous Observation Noise
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多视图聚类任务，旨在解决真实场景中异质噪声干扰问题。提出QARMVC框架，通过质量感知机制提升聚类鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.22568v1](https://arxiv.org/pdf/2602.22568v1)**

> **作者:** Peihan Wu; Guanjie Cheng; Yufei Tong; Meng Xi; Shuiguang Deng
>
> **摘要:** Deep multi-view clustering has achieved remarkable progress but remains vulnerable to complex noise in real-world applications. Existing noisy robust methods predominantly rely on a simplified binary assumption, treating data as either perfectly clean or completely corrupted. This overlooks the prevalent existence of heterogeneous observation noise, where contamination intensity varies continuously across data. To bridge this gap, we propose a novel framework termed Quality-Aware Robust Multi-View Clustering (QARMVC). Specifically, QARMVC employs an information bottleneck mechanism to extract intrinsic semantics for view reconstruction. Leveraging the insight that noise disrupts semantic integrity and impedes reconstruction, we utilize the resulting reconstruction discrepancy to precisely quantify fine-grained contamination intensity and derive instance-level quality scores. These scores are integrated into a hierarchical learning strategy: at the feature level, a quality-weighted contrastive objective is designed to adaptively suppress the propagation of noise; at the fusion level, a high-quality global consensus is constructed via quality-weighted aggregation, which is subsequently utilized to align and rectify local views via mutual information maximization. Extensive experiments on five benchmark datasets demonstrate that QARMVC consistently outperforms state-of-the-art baselines, particularly in scenarios with heterogeneous noise intensities.
>
---
#### [new 119] PGVMS: A Prompt-Guided Unified Framework for Virtual Multiplex IHC Staining with Pathological Semantic Learning
- **分类: cs.CV**

- **简介: 该论文属于虚拟多通道IHC染色任务，旨在解决组织样本不足导致的分析限制。通过引入提示引导框架，提升语义指导、蛋白分布一致性和空间对齐性。**

- **链接: [https://arxiv.org/pdf/2602.23292v1](https://arxiv.org/pdf/2602.23292v1)**

> **作者:** Fuqiang Chen; Ranran Zhang; Wanming Hu; Deboch Eyob Abera; Yue Peng; Boyun Zheng; Yiwen Sun; Jing Cai; Wenjian Qin
>
> **备注:** Accepted by TMI
>
> **摘要:** Immunohistochemical (IHC) staining enables precise molecular profiling of protein expression, with over 200 clinically available antibody-based tests in modern pathology. However, comprehensive IHC analysis is frequently limited by insufficient tissue quantities in small biopsies. Therefore, virtual multiplex staining emerges as an innovative solution to digitally transform H&E images into multiple IHC representations, yet current methods still face three critical challenges: (1) inadequate semantic guidance for multi-staining, (2) inconsistent distribution of immunochemistry staining, and (3) spatial misalignment across different stain modalities. To overcome these limitations, we present a prompt-guided framework for virtual multiplex IHC staining using only uniplex training data (PGVMS). Our framework introduces three key innovations corresponding to each challenge: First, an adaptive prompt guidance mechanism employing a pathological visual language model dynamically adjusts staining prompts to resolve semantic guidance limitations (Challenge 1). Second, our protein-aware learning strategy (PALS) maintains precise protein expression patterns by direct quantification and constraint of protein distributions (Challenge 2). Third, the prototype-consistent learning strategy (PCLS) establishes cross-image semantic interaction to correct spatial misalignments (Challenge 3).
>
---
#### [new 120] Align then Adapt: Rethinking Parameter-Efficient Transfer Learning in 4D Perception
- **分类: cs.CV**

- **简介: 该论文属于4D感知任务，解决3D模型迁移至4D任务时的过拟合和模态差异问题。提出“Align then Adapt”方法，通过对齐和适配提升参数效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.23069v1](https://arxiv.org/pdf/2602.23069v1)**

> **作者:** Yiding Sun; Jihua Zhu; Haozhe Cheng; Chaoyi Lu; Zhichuan Yang; Lin Chen; Yaonan Wang
>
> **摘要:** Point cloud video understanding is critical for robotics as it accurately encodes motion and scene interaction. We recognize that 4D datasets are far scarcer than 3D ones, which hampers the scalability of self-supervised 4D models. A promising alternative is to transfer 3D pre-trained models to 4D perception tasks. However, rigorous empirical analysis reveals two critical limitations that impede transfer capability: overfitting and the modality gap. To overcome these challenges, we develop a novel "Align then Adapt" (PointATA) paradigm that decomposes parameter-efficient transfer learning into two sequential stages. Optimal-transport theory is employed to quantify the distributional discrepancy between 3D and 4D datasets, enabling our proposed point align embedder to be trained in Stage 1 to alleviate the underlying modality gap. To mitigate overfitting, an efficient point-video adapter and a spatial-context encoder are integrated into the frozen 3D backbone to enhance temporal modeling capacity in Stage 2. Notably, with the above engineering-oriented designs, PointATA enables a pre-trained 3D model without temporal knowledge to reason about dynamic video content at a smaller parameter cost compared to previous work. Extensive experiments show that PointATA can match or even outperform strong full fine-tuning models, whilst enjoying the advantage of parameter efficiency, e.g. 97.21 \% accuracy on 3D action recognition, $+8.7 \%$ on 4 D action segmentation, and 84.06\% on 4D semantic segmentation.
>
---
#### [new 121] HELMLAB: An Analytical, Data-Driven Color Space for Perceptual Distance in UI Design Systems
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出HELMLAB，一个用于UI设计系统的72参数颜色空间，解决感知距离建模问题。通过学习矩阵和修正方法提升颜色匹配精度，实现高跨数据集性能。**

- **链接: [https://arxiv.org/pdf/2602.23010v1](https://arxiv.org/pdf/2602.23010v1)**

> **作者:** Gorkem Yildiz
>
> **备注:** 9 pages, 6 figures. Code and demo available at: https://github.com/Grkmyldz148/helmlab
>
> **摘要:** We present HELMLAB, a 72-parameter analytical color space for UI design systems. The forward transform maps CIE XYZ to a perceptually-organized Lab representation through learned matrices, per-channel power compression, Fourier hue correction, and embedded Helmholtz-Kohlrausch lightness adjustment. A post-pipeline neutral correction guarantees that achromatic colors map to a=b=0 (chroma < 10^-6), and a rigid rotation of the chromatic plane improves hue-angle alignment without affecting the distance metric, which is invariant under isometries. On the COMBVD dataset (3,813 color pairs), HELMLAB achieves a STRESS of 23.22, a 20.4% reduction from CIEDE2000 (29.18). Cross-validation on He et al. 2022 and MacAdam 1974 shows competitive cross-dataset performance. The transform is invertible with round-trip errors below 10^-14. Gamut mapping, design-token export, and dark/light mode adaptation utilities are included for use in web and mobile design systems.
>
---
#### [new 122] Partial recovery of meter-scale surface weather
- **分类: cs.LG; cs.CV; physics.ao-ph**

- **简介: 该论文属于气象建模任务，解决地面天气在百米尺度上的可预测性问题。通过结合观测数据，推断高分辨率近地表气象场，提升预报精度。**

- **链接: [https://arxiv.org/pdf/2602.23146v1](https://arxiv.org/pdf/2602.23146v1)**

> **作者:** Jonathan Giezendanner; Qidong Yang; Eric Schmitt; Anirban Chandra; Daniel Salles Civitarese; Johannes Jakubik; Jeremy Vila; Detlef Hohl; Campbell Watson; Sherrie Wang
>
> **摘要:** Near-surface atmospheric conditions can differ sharply over tens to hundreds of meters due to land cover and topography, yet this variability is absent from current weather analyses and forecasts. It is unclear whether such meter-scale variability reflects irreducibly chaotic dynamics or contains a component predictable from surface characteristics and large-scale atmospheric forcing. Here we show that a substantial, physically coherent component of meter-scale near-surface weather is statistically recoverable from existing observations. By conditioning coarse atmospheric state on sparse surface station measurements and high-resolution Earth observation data, we infer spatially continuous fields of near-surface wind, temperature, and humidity at 10 m resolution across the contiguous United States. Relative to ERA5, the inferred fields reduce wind error by 29% and temperature and dewpoint error by 6%, while explaining substantially more spatial variance at fixed time steps. They also exhibit physically interpretable structure, including urban heat islands, evapotranspiration-driven humidity contrasts, and wind speed differences across land cover types. Our findings expand the frontier of weather modeling by demonstrating a computationally feasible approach to continental-scale meter-resolution inference. More broadly, they illustrate how conditioning coarse dynamical models on static fine-scale features can reveal previously unresolved components of the Earth system.
>
---
#### [new 123] A Dataset is Worth 1 MB
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于数据集传输优化任务，旨在降低通信成本。通过仅传输标签而非图像，利用参考数据集进行训练，实现高效任务知识迁移。**

- **链接: [https://arxiv.org/pdf/2602.23358v1](https://arxiv.org/pdf/2602.23358v1)**

> **作者:** Elad Kimchi Shoshani; Leeyam Gabay; Yedid Hoshen
>
> **备注:** 23 pages, 9 figures
>
> **摘要:** A dataset server must often distribute the same large payload to many clients, incurring massive communication costs. Since clients frequently operate on diverse hardware and software frameworks, transmitting a pre-trained model is often infeasible; instead, agents require raw data to train their own task-specific models locally. While dataset distillation attempts to compress training signals, current methods struggle to scale to high-resolution data and rarely achieve sufficiently small files. In this paper, we propose Pseudo-Labels as Data (PLADA), a method that completely eliminates pixel transmission. We assume agents are preloaded with a large, generic, unlabeled reference dataset (e.g., ImageNet-1K, ImageNet-21K) and communicate a new task by transmitting only the class labels for specific images. To address the distribution mismatch between the reference and target datasets, we introduce a pruning mechanism that filters the reference dataset to retain only the labels of the most semantically relevant images for the target task. This selection process simultaneously maximizes training efficiency and minimizes transmission payload. Experiments on 10 diverse datasets demonstrate that our approach can transfer task knowledge with a payload of less than 1 MB while retaining high classification accuracy, offering a promising solution for efficient dataset serving.
>
---
#### [new 124] Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于生态监测任务，旨在解决森林幼树三维重建与精准定位问题。通过融合NeRF、LiDAR SLAM和GNSS，实现幼树的高精度结构重建与长期跟踪。**

- **链接: [https://arxiv.org/pdf/2602.22731v1](https://arxiv.org/pdf/2602.22731v1)**

> **作者:** Miguel Ángel Muñoz-Bañón; Nived Chebrolu; Sruthi M. Krishna Moorthy; Yifu Tao; Fernando Torres; Roberto Salguero-Gómez; Maurice Fallon
>
> **摘要:** Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.
>
---
#### [new 125] Entropy-Controlled Flow Matching
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出熵控制流匹配（ECFM），解决生成模型中信息几何控制问题，通过约束优化提升模式覆盖率和稳定性。**

- **链接: [https://arxiv.org/pdf/2602.22265v1](https://arxiv.org/pdf/2602.22265v1)**

> **作者:** Chika Maduabuchi
>
> **摘要:** Modern vision generators transport a base distribution to data through time-indexed measures, implemented as deterministic flows (ODEs) or stochastic diffusions (SDEs). Despite strong empirical performance, standard flow-matching objectives do not directly control the information geometry of the trajectory, allowing low-entropy bottlenecks that can transiently deplete semantic modes. We propose Entropy-Controlled Flow Matching (ECFM): a constrained variational principle over continuity-equation paths enforcing a global entropy-rate budget d/dt H(mu_t) >= -lambda. ECFM is a convex optimization in Wasserstein space with a KKT/Pontryagin system, and admits a stochastic-control representation equivalent to a Schrodinger bridge with an explicit entropy multiplier. In the pure transport regime, ECFM recovers entropic OT geodesics and Gamma-converges to classical OT as lambda -> 0. We further obtain certificate-style mode-coverage and density-floor guarantees with Lipschitz stability, and construct near-optimal collapse counterexamples for unconstrained flow matching.
>
---
#### [new 126] DiffBMP: Differentiable Rendering with Bitmap Primitives
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出DiffBMP，解决位图图像的可微渲染问题，通过高效并行化管道实现快速优化，适用于创意工作流。**

- **链接: [https://arxiv.org/pdf/2602.22625v1](https://arxiv.org/pdf/2602.22625v1)**

> **作者:** Seongmin Hong; Junghun James Kim; Daehyeop Kim; Insoo Chung; Se Young Chun
>
> **备注:** Accepted to CVPR 2026, https://diffbmp.com
>
> **摘要:** We introduce DiffBMP, a scalable and efficient differentiable rendering engine for a collection of bitmap images. Our work addresses a limitation that traditional differentiable renderers are constrained to vector graphics, given that most images in the world are bitmaps. Our core contribution is a highly parallelized rendering pipeline, featuring a custom CUDA implementation for calculating gradients. This system can, for example, optimize the position, rotation, scale, color, and opacity of thousands of bitmap primitives all in under 1 min using a consumer GPU. We employ and validate several techniques to facilitate the optimization: soft rasterization via Gaussian blur, structure-aware initialization, noisy canvas, and specialized losses/heuristics for videos or spatially constrained images. We demonstrate DiffBMP is not just an isolated tool, but a practical one designed to integrate into creative workflows. It supports exporting compositions to a native, layered file format, and the entire framework is publicly accessible via an easy-to-hack Python package.
>
---
#### [new 127] Space Syntax-guided Post-training for Residential Floor Plan Generation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于住宅平面生成任务，旨在解决生成模型忽视空间结构优先的问题。通过引入空间句法知识，提出SSPT方法提升公共空间主导性和功能层次。**

- **链接: [https://arxiv.org/pdf/2602.22507v1](https://arxiv.org/pdf/2602.22507v1)**

> **作者:** Zhuoyang Jiang; Dongqing Zhang
>
> **摘要:** Pre-trained generative models for residential floor plans are typically optimized to fit large-scale data distributions, which can under-emphasize critical architectural priors such as the configurational dominance and connectivity of domestic public spaces (e.g., living rooms and foyers). This paper proposes Space Syntax-guided Post-training (SSPT), a post-training paradigm that explicitly injects space syntax knowledge into floor plan generation via a non-differentiable oracle. The oracle converts RPLAN-style layouts into rectangle-space graphs through greedy maximal-rectangle decomposition and door-mediated adjacency construction, and then computes integration-based measurements to quantify public space dominance and functional hierarchy. To enable consistent evaluation and diagnosis, we further introduce SSPT-Bench (Eval-8), an out-of-distribution benchmark that post-trains models using conditions capped at $\leq 7$ rooms while evaluating on 8-room programs, together with a unified metric suite for dominance, stability, and profile alignment. SSPT is instantiated with two strategies: (i) iterative retraining via space-syntax filtering and diffusion fine-tuning, and (ii) reinforcement learning via PPO with space-syntax rewards. Experiments show that both strategies improve public-space dominance and restore clearer functional hierarchy compared to distribution-fitted baselines, while PPO achieves stronger gains with substantially higher compute efficiency and reduced variance. SSPT provides a scalable pathway for integrating architectural theory into data-driven plan generation and is compatible with other generative backbones given a post-hoc evaluation oracle.
>
---
#### [new 128] $φ$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决大型多模态模型中的公平性问题。针对数据分布不平衡导致的模型偏差，提出φ-DPO方法，优化学习过程以减少遗忘并提升性能。**

- **链接: [https://arxiv.org/pdf/2602.22601v1](https://arxiv.org/pdf/2602.22601v1)**

> **作者:** Thanh-Dat Truong; Huu-Thien Tran; Jackson Cothren; Bhiksha Raj; Khoa Luu
>
> **备注:** Accepted to CVPR'26
>
> **摘要:** Fairness in Continual Learning for Large Multimodal Models (LMMs) is an emerging yet underexplored challenge, particularly in the presence of imbalanced data distributions that can lead to biased model updates and suboptimal performance across tasks. While recent continual learning studies have made progress in addressing catastrophic forgetting, the problem of fairness caused the imbalanced data remains largely underexplored. This paper presents a novel Fairness Direct Preference Optimization (FaiDPO or $φ$-DPO) framework for continual learning in LMMs. In particular, we first propose a new continual learning paradigm based on Direct Preference Optimization (DPO) to mitigate catastrophic forgetting by aligning learning with pairwise preference signals. Then, we identify the limitations of conventional DPO in imbalanced data and present a new $φ$-DPO loss that explicitly addresses distributional biases. We provide a comprehensive theoretical analysis demonstrating that our approach addresses both forgetting and data imbalance. Additionally, to enable $φ$-DPO-based continual learning, we construct pairwise preference annotations for existing benchmarks in the context of continual learning. Extensive experiments and ablation studies show the proposed $φ$-DPO achieves State-of-the-Art performance across multiple benchmarks, outperforming prior continual learning methods of LMMs.
>
---
#### [new 129] Certified Circuits: Stability Guarantees for Mechanistic Circuits
- **分类: cs.AI; cs.CV; cs.CY**

- **简介: 该论文属于神经网络可解释性任务，解决电路发现稳定性问题。通过引入认证电路框架，确保电路在数据扰动下保持稳定，提升准确性和简洁性。**

- **链接: [https://arxiv.org/pdf/2602.22968v1](https://arxiv.org/pdf/2602.22968v1)**

> **作者:** Alaa Anani; Tobias Lorenz; Bernt Schiele; Mario Fritz; Jonas Fischer
>
> **摘要:** Understanding how neural networks arrive at their predictions is essential for debugging, auditing, and deployment. Mechanistic interpretability pursues this goal by identifying circuits - minimal subnetworks responsible for specific behaviors. However, existing circuit discovery methods are brittle: circuits depend strongly on the chosen concept dataset and often fail to transfer out-of-distribution, raising doubts whether they capture concept or dataset-specific artifacts. We introduce Certified Circuits, which provide provable stability guarantees for circuit discovery. Our framework wraps any black-box discovery algorithm with randomized data subsampling to certify that circuit component inclusion decisions are invariant to bounded edit-distance perturbations of the concept dataset. Unstable neurons are abstained from, yielding circuits that are more compact and more accurate. On ImageNet and OOD datasets, certified circuits achieve up to 91% higher accuracy while using 45% fewer neurons, and remain reliable where baselines degrade. Certified Circuits puts circuit discovery on formal ground by producing mechanistic explanations that are provably stable and better aligned with the target concept. Code will be released soon!
>
---
#### [new 130] Scale Can't Overcome Pragmatics: The Impact of Reporting Bias on Vision-Language Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言推理任务，旨在解决VLMs缺乏推理能力的问题。研究发现训练数据的报告偏差导致推理技能不足，并提出通过精心标注数据提升性能。**

- **链接: [https://arxiv.org/pdf/2602.23351v1](https://arxiv.org/pdf/2602.23351v1)**

> **作者:** Amita Kamath; Jack Hessel; Khyathi Chandu; Jena D. Hwang; Kai-Wei Chang; Ranjay Krishna
>
> **备注:** TACL 2026
>
> **摘要:** The lack of reasoning capabilities in Vision-Language Models (VLMs) has remained at the forefront of research discourse. We posit that this behavior stems from a reporting bias in their training data. That is, how people communicate about visual content by default omits tacit information needed to supervise some types of reasoning; e.g., "at the game today!" is a more likely caption than "a photo of 37 people standing behind a field". We investigate the data underlying the popular VLMs OpenCLIP, LLaVA-1.5 and Molmo through the lens of theories from pragmatics, and find that reporting bias results in insufficient representation of four reasoning skills (spatial, temporal, negation, and counting), despite the corpora being of web-scale, and/or synthetically generated. With a set of curated benchmarks, we demonstrate that: (i) VLMs perform poorly on the aforementioned types of reasoning suppressed in the training data by reporting bias; (ii) contrary to popular belief, scaling data size, model size, and to multiple languages does not result in emergence of these skills by default; but, promisingly, (iii) incorporating annotations specifically collected to obtain tacit information is effective. Our findings highlight the need for more intentional training data curation methods, rather than counting on scale for emergence of reasoning capabilities.
>
---
#### [new 131] CrossLLM-Mamba: Multimodal State Space Fusion of LLMs for RNA Interaction Prediction
- **分类: q-bio.GN; cs.CV; cs.LG**

- **简介: 该论文属于RNA相互作用预测任务，旨在解决静态融合策略无法捕捉动态分子结合的问题。提出CrossLLM-Mamba框架，通过状态空间对齐实现多模态嵌入的深度交互。**

- **链接: [https://arxiv.org/pdf/2602.22236v1](https://arxiv.org/pdf/2602.22236v1)**

> **作者:** Rabeya Tus Sadia; Qiang Ye; Qiang Cheng
>
> **摘要:** Accurate prediction of RNA-associated interactions is essential for understanding cellular regulation and advancing drug discovery. While Biological Large Language Models (BioLLMs) such as ESM-2 and RiNALMo provide powerful sequence representations, existing methods rely on static fusion strategies that fail to capture the dynamic, context-dependent nature of molecular binding. We introduce CrossLLM-Mamba, a novel framework that reformulates interaction prediction as a state-space alignment problem. By leveraging bidirectional Mamba encoders, our approach enables deep ``crosstalk'' between modality-specific embeddings through hidden state propagation, modeling interactions as dynamic sequence transitions rather than static feature overlaps. The framework maintains linear computational complexity, making it scalable to high-dimensional BioLLM embeddings. We further incorporate Gaussian noise injection and Focal Loss to enhance robustness against hard-negative samples. Comprehensive experiments across three interaction categories, RNA-protein, RNA-small molecule, and RNA-RNA demonstrate that CrossLLM-Mamba achieves state-of-the-art performance. On the RPI1460 benchmark, our model attains an MCC of 0.892, surpassing the previous best by 5.2\%. For binding affinity prediction, we achieve Pearson correlations exceeding 0.95 on riboswitch and repeat RNA subtypes. These results establish state-space modeling as a powerful paradigm for multi-modal biological interaction prediction.
>
---
#### [new 132] Moral Preferences of LLMs Under Directed Contextual Influence
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.CY**

- **简介: 该论文研究LLMs在有导向性上下文影响下的道德决策，旨在揭示上下文如何改变模型的道德判断。任务属于AI伦理评估，解决上下文对模型决策的影响问题，通过实验分析不同情境下的行为变化。**

- **链接: [https://arxiv.org/pdf/2602.22831v1](https://arxiv.org/pdf/2602.22831v1)**

> **作者:** Phil Blandfort; Tushar Karayil; Urja Pawar; Robert Graham; Alex McKenzie; Dmitrii Krasheninnikov
>
> **摘要:** Moral benchmarks for LLMs typically use context-free prompts, implicitly assuming stable preferences. In deployment, however, prompts routinely include contextual signals such as user requests, cues on social norms, etc. that may steer decisions. We study how directed contextual influences reshape decisions in trolley-problem-style moral triage settings. We introduce a pilot evaluation harness for directed contextual influence in trolley-problem-style moral triage: for each demographic factor, we apply matched, direction-flipped contextual influences that differ only in which group they favor, enabling systematic measurement of directional response. We find that: (i) contextual influences often significantly shift decisions, even when only superficially relevant; (ii) baseline preferences are a poor predictor of directional steerability, as models can appear baseline-neutral yet exhibit systematic steerability asymmetry under influence; (iii) influences can backfire: models may explicitly claim neutrality or discount the contextual cue, yet their choices still shift, sometimes in the opposite direction; and (iv) reasoning reduces average sensitivity, but amplifies the effect of biased few-shot examples. Our findings motivate extending moral evaluations with controlled, direction-flipped context manipulations to better characterize model behavior.
>
---
#### [new 133] GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，旨在提升模仿学习策略的抓取精度和泛化能力。通过引入先验知识的扩散策略，提高抓取动作的准确性和适应性。**

- **链接: [https://arxiv.org/pdf/2602.22862v1](https://arxiv.org/pdf/2602.22862v1)**

> **作者:** Enda Xiang; Haoxiang Ma; Xinzhu Ma; Zicheng Liu; Di Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** This paper focuses on enhancing the grasping precision and generalization of manipulation policies learned via imitation learning. Diffusion-based policy learning methods have recently become the mainstream approach for robotic manipulation tasks. As grasping is a critical subtask in manipulation, the ability of imitation-learned policies to execute precise and generalizable grasps merits particular attention. Existing imitation learning techniques for grasping often suffer from imprecise grasp executions, limited spatial generalization, and poor object generalization. To address these challenges, we incorporate grasp prior knowledge into the diffusion policy framework. In particular, we employ a latent diffusion policy to guide action chunk decoding with grasp pose prior, ensuring that generated motion trajectories adhere closely to feasible grasp configurations. Furthermore, we introduce a self-supervised reconstruction objective during diffusion to embed the graspness prior: at each reverse diffusion step, we reconstruct wrist-camera images back-projected the graspness from the intermediate representations. Both simulation and real robot experiments demonstrate that our approach significantly outperforms baseline methods and exhibits strong dynamic grasping capabilities.
>
---
#### [new 134] OmniGAIA: Towards Native Omni-Modal AI Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MM**

- **简介: 该论文提出OmniGAIA，解决多模态AI助手缺乏统一认知能力的问题，通过构建多模态基准和训练代理，提升工具使用与跨模态推理能力。**

- **链接: [https://arxiv.org/pdf/2602.22897v1](https://arxiv.org/pdf/2602.22897v1)**

> **作者:** Xiaoxi Li; Wenxiang Jiao; Jiarui Jin; Shijian Wang; Guanting Dong; Jiajie Jin; Hao Wang; Yinuo Wang; Ji-Rong Wen; Yuan Lu; Zhicheng Dou
>
> **摘要:** Human intelligence naturally intertwines omni-modal perception -- spanning vision, audio, and language -- with complex reasoning and tool usage to interact with the world. However, current multi-modal LLMs are primarily confined to bi-modal interactions (e.g., vision-language), lacking the unified cognitive capabilities required for general AI assistants. To bridge this gap, we introduce OmniGAIA, a comprehensive benchmark designed to evaluate omni-modal agents on tasks necessitating deep reasoning and multi-turn tool execution across video, audio, and image modalities. Constructed via a novel omni-modal event graph approach, OmniGAIA synthesizes complex, multi-hop queries derived from real-world data that require cross-modal reasoning and external tool integration. Furthermore, we propose OmniAtlas, a native omni-modal foundation agent under tool-integrated reasoning paradigm with active omni-modal perception. Trained on trajectories synthesized via a hindsight-guided tree exploration strategy and OmniDPO for fine-grained error correction, OmniAtlas effectively enhances the tool-use capabilities of existing open-source models. This work marks a step towards next-generation native omni-modal AI assistants for real-world scenarios.
>
---
#### [new 135] MolFM-Lite: Multi-Modal Molecular Property Prediction with Conformer Ensemble Attention and Cross-Modal Fusion
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于分子属性预测任务，解决单一分子表示限制问题。提出MolFM-Lite模型，融合序列、图和构象集合信息，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2602.22405v1](https://arxiv.org/pdf/2602.22405v1)**

> **作者:** Syed Omer Shah; Mohammed Maqsood Ahmed; Danish Mohiuddin Mohammed; Shahnawaz Alam; Mohd Vahaj ur Rahman
>
> **摘要:** Most machine learning models for molecular property prediction rely on a single molecular representation (either a sequence, a graph, or a 3D structure) and treat molecular geometry as static. We present MolFM-Lite, a multi-modal model that jointly encodes SELFIES sequences (1D), molecular graphs (2D), and conformer ensembles (3D) through cross-attention fusion, while conditioning predictions on experimental context via Feature-wise Linear Modulation (FiLM). Our main methodological contributions are: (1) a conformer ensemble attention mechanism that combines learnable attention with Boltzmann-weighted priors over multiple RDKit-generated conformers, capturing the thermodynamic distribution of molecular shapes; and (2) a cross-modal fusion layer where each modality can attend to others, enabling complementary information sharing. We evaluate on four MoleculeNet scaffold-split benchmarks using our model's own splits, and report all baselines re-evaluated under the same protocol. Comprehensive ablation studies across all four datasets confirm that each architectural component contributes independently, with tri-modal fusion providing 7-11% AUC improvement over single-modality baselines and conformer ensembles adding approximately 2% over single-conformer variants. Pre-training on ZINC250K (~250K molecules) using cross-modal contrastive and masked-atom objectives enables effective weight initialization at modest compute cost. We release all code, trained models, and data splits to support reproducibility.
>
---
#### [new 136] HARU-Net: Hybrid Attention Residual U-Net for Edge-Preserving Denoising in Cone-Beam Computed Tomography
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于CBCT图像去噪任务，旨在解决低剂量CBCT中噪声干扰和边缘模糊问题。提出HARU-Net模型，结合注意力机制与残差结构，提升去噪效果。**

- **链接: [https://arxiv.org/pdf/2602.22544v1](https://arxiv.org/pdf/2602.22544v1)**

> **作者:** Khuram Naveed; Ruben Pauwels
>
> **摘要:** Cone-beam computed tomography (CBCT) is widely used in dental and maxillofacial imaging, but low-dose acquisition introduces strong, spatially varying noise that degrades soft-tissue visibility and obscures fine anatomical structures. Classical denoising methods struggle to suppress noise in CBCT while preserving edges. Although deep learning-based approaches offer high-fidelity restoration, their use in CBCT denoising is limited by the scarcity of high-resolution CBCT data for supervised training. To address this research gap, we propose a novel Hybrid Attention Residual U-Net (HARU-Net) for high-quality denoising of CBCT data, trained on a cadaver dataset of human hemimandibles acquired using a high-resolution protocol of the 3D Accuitomo 170 (J. Morita, Kyoto, Japan) CBCT system. The novel contribution of this approach is the integration of three complementary architectural components: (i) a hybrid attention transformer block (HAB) embedded within each skip connection to selectively emphasize salient anatomical features, (ii) a residual hybrid attention transformer group (RHAG) at the bottleneck to strengthen global contextual modeling and long-range feature interactions, and (iii) residual learning convolutional blocks to facilitate deeper, more stable feature extraction throughout the network. HARU-Net consistently outperforms state-of-the-art (SOTA) methods including SwinIR and Uformer, achieving the highest PSNR (37.52 dB), highest SSIM (0.9557), and lowest GMSD (0.1084). This effective and clinically reliable CBCT denoising is achieved at a computational cost significantly lower than that of the SOTA methods, offering a practical advancement toward improving diagnostic quality in low-dose CBCT imaging.
>
---
#### [new 137] Adaptive Prefiltering for High-Dimensional Similarity Search: A Frequency-Aware Approach
- **分类: cs.IR; cs.CV**

- **简介: 该论文属于高维相似性搜索任务，解决统一策略无法适应查询分布差异的问题。通过频率感知的自适应预过滤框架，优化计算资源分配，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.22214v1](https://arxiv.org/pdf/2602.22214v1)**

> **作者:** Teodor-Ioan Calin
>
> **摘要:** High-dimensional similarity search underpins modern retrieval systems, yet uniform search strategies fail to exploit the heterogeneous nature of real-world query distributions. We present an adaptive prefiltering framework that leverages query frequency patterns and cluster coherence metrics to dynamically allocate computational budgets. Our approach partitions the query space into frequency tiers following Zipfian distributions and assigns differentiated search policies based on historical access patterns and local density characteristics. Experiments on ImageNet-1k using CLIP embeddings demonstrate that frequency-aware budget allocation achieves equivalent recall with 20.4% fewer distance computations compared to static nprobe selection, while maintaining sub-millisecond latency on GPU-accelerated FAISS indices. The framework introduces minimal overhead through lightweight frequency tracking and provides graceful degradation for unseen queries through coherence-based fallback policies.
>
---
#### [new 138] DP-aware AdaLN-Zero: Taming Conditioning-Induced Heavy-Tailed Gradients in Differentially Private Diffusion
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于时间序列生成任务，解决差分隐私下条件注入导致的梯度重尾问题。提出DP-aware AdaLN-Zero机制，抑制极端梯度事件，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.22610v1](https://arxiv.org/pdf/2602.22610v1)**

> **作者:** Tao Huang; Jiayang Meng; Xu Yang; Chen Hou; Hong Chen
>
> **摘要:** Condition injection enables diffusion models to generate context-aware outputs, which is essential for many time-series tasks. However, heterogeneous conditional contexts (e.g., observed history, missingness patterns or outlier covariates) can induce heavy-tailed per-example gradients. Under Differentially Private Stochastic Gradient Descent (DP-SGD), these rare conditioning-driven heavy-tailed gradients disproportionately trigger global clipping, resulting in outlier-dominated updates, larger clipping bias, and degraded utility under a fixed privacy budget. In this paper, we propose DP-aware AdaLN-Zero, a drop-in sensitivity-aware conditioning mechanism for conditional diffusion transformers that limits conditioning-induced gain without modifying the DP-SGD mechanism. DP-aware AdaLN-Zero jointly constrains conditioning representation magnitude and AdaLN modulation parameters via bounded re-parameterization, suppressing extreme gradient tail events before gradient clipping and noise injection. Empirically, DP-SGD equipped with DP-aware AdaLN-Zero improves interpolation/imputation and forecasting under matched privacy settings. We observe consistent gains on a real-world power dataset and two public ETT benchmarks over vanilla DP-SGD. Moreover, gradient diagnostics attribute these improvements to conditioning-specific tail reshaping and reduced clipping distortion, while preserving expressiveness in non-private training. Overall, these results show that sensitivity-aware conditioning can substantially improve private conditional diffusion training without sacrificing standard performance.
>
---
#### [new 139] An automatic counting algorithm for the quantification and uncertainty analysis of the number of microglial cells trainable in small and heterogeneous datasets
- **分类: cs.CE; cs.CV; eess.IV; eess.SP; stat.ML**

- **简介: 该论文属于细胞计数任务，旨在解决小而异质数据集中的微胶质细胞自动计数问题。通过设计一种非参数核计数方法，实现高效、准确的计数及不确定性估计。**

- **链接: [https://arxiv.org/pdf/2602.22974v1](https://arxiv.org/pdf/2602.22974v1)**

> **作者:** L. Martino; M. M. Garcia; P. S. Paradas; E. Curbelo
>
> **摘要:** Counting immunopositive cells on biological tissues generally requires either manual annotation or (when available) automatic rough systems, for scanning signal surface and intensity in whole slide imaging. In this work, we tackle the problem of counting microglial cells in lumbar spinal cord cross-sections of rats by omitting cell detection and focusing only on the counting task. Manual cell counting is, however, a time-consuming task and additionally entails extensive personnel training. The classic automatic color-based methods roughly inform about the total labeled area and intensity (protein quantification) but do not specifically provide information on cell number. Since the images to be analyzed have a high resolution but a huge amount of pixels contain just noise or artifacts, we first perform a pre-processing generating several filtered images {(providing a tailored, efficient feature extraction)}. Then, we design an automatic kernel counter that is a non-parametric and non-linear method. The proposed scheme can be easily trained in small datasets since, in its basic version, it relies only on one hyper-parameter. However, being non-parametric and non-linear, the proposed algorithm is flexible enough to express all the information contained in rich and heterogeneous datasets as well (providing the maximum overfit if required). Furthermore, the proposed kernel counter also provides uncertainty estimation of the given prediction, and can directly tackle the case of receiving several expert opinions over the same image. Different numerical experiments with artificial and real datasets show very promising results. Related Matlab code is also provided.
>
---
## 更新

#### [replaced 001] MomentMix Augmentation with Length-Aware DETR for Temporally Robust Moment Retrieval
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2412.20816v3](https://arxiv.org/pdf/2412.20816v3)**

> **作者:** Seojeong Park; Jiho Choi; Kyungjune Baek; Hyunjung Shim
>
> **备注:** WACV 2026
>
> **摘要:** Video Moment Retrieval (MR) aims to localize moments within a video based on a given natural language query. Given the prevalent use of platforms like YouTube for information retrieval, the demand for MR techniques is significantly growing. Recent DETR-based models have made notable advances in performance but still struggle with accurately localizing short moments. Through data analysis, we identified limited feature diversity in short moments, which motivated the development of MomentMix. MomentMix generates new short-moment samples by employing two augmentation strategies: ForegroundMix and BackgroundMix, each enhancing the ability to understand the query-relevant and irrelevant frames, respectively. Additionally, our analysis of prediction bias revealed that short moments particularly struggle with accurately predicting their center positions and length of moments. To address this, we propose a Length-Aware Decoder, which conditions length through a novel bipartite matching process. Our extensive studies demonstrate the efficacy of our length-aware approach, especially in localizing short moments, leading to improved overall performance. Our method surpasses state-of-the-art DETR-based methods on benchmark datasets, achieving the highest R1 and mAP on QVHighlights and the highest R1@0.7 on TACoS and Charades-STA (such as a 9.62% gain in R1@0.7 and an 16.9% gain in mAP average for QVHighlights). The code is available at https://github.com/sjpark5800/LA-DETR.
>
---
#### [replaced 002] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21965v3](https://arxiv.org/pdf/2509.21965v3)**

> **作者:** Zhe Zhu; Le Wan; Rui Xu; Yiheng Zhang; Honghua Chen; Zhiyang Dou; Cheng Lin; Yuan Liu; Mingqiang Wei
>
> **备注:** ICLR 2026. Project Page: https://czvvd.github.io/PartSAMPage/
>
> **摘要:** Segmenting 3D objects into parts is a long-standing challenge in computer vision. To overcome taxonomy constraints and generalize to unseen 3D objects, recent works turn to open-world part segmentation. These approaches typically transfer supervision from 2D foundation models, such as SAM, by lifting multi-view masks into 3D. However, this indirect paradigm fails to capture intrinsic geometry, leading to surface-only understanding, uncontrolled decomposition, and limited generalization. We present PartSAM, the first promptable part segmentation model trained natively on large-scale 3D data. Following the design philosophy of SAM, PartSAM employs an encoder-decoder architecture in which a triplane-based dual-branch encoder produces spatially structured tokens for scalable part-aware representation learning. To enable large-scale supervision, we further introduce a model-in-the-loop annotation pipeline that curates over five million 3D shape-part pairs from online assets, providing diverse and fine-grained labels. This combination of scalable architecture and diverse 3D data yields emergent open-world capabilities: with a single prompt, PartSAM achieves highly accurate part identification, and in a Segment-Every-Part mode, it automatically decomposes shapes into both surface and internal structures. Extensive experiments show that PartSAM outperforms state-of-the-art methods by large margins across multiple benchmarks, marking a decisive step toward foundation models for 3D part understanding.
>
---
#### [replaced 003] OneVision-Encoder: Codec-Aligned Sparsity as a Foundational Principle for Multimodal Intelligence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.08683v3](https://arxiv.org/pdf/2602.08683v3)**

> **作者:** Feilong Tang; Xiang An; Yunyao Yan; Yin Xie; Bin Qin; Kaicheng Yang; Yifei Shen; Yuanhan Zhang; Chunyuan Li; Shikun Feng; Changrui Chen; Huajie Tan; Ming Hu; Manyuan Zhang; Bo Li; Ziyong Feng; Ziwei Liu; Zongyuan Ge; Jiankang Deng
>
> **摘要:** Hypothesis. Artificial general intelligence is, at its core, a compression problem. Effective compression demands resonance: deep learning scales best when its architecture aligns with the fundamental structure of the data. These are the fundamental principles. Yet, modern vision architectures have strayed from these truths: visual signals are highly redundant, while discriminative information, the surprise, is sparse. Current models process dense pixel grids uniformly, wasting vast compute on static background rather than focusing on the predictive residuals that define motion and meaning. We argue that to solve visual understanding, we must align our architectures with the information-theoretic principles of video, i.e., Codecs. Method. OneVision-Encoder encodes video by compressing predictive visual structure into semantic meaning. By adopting Codec Patchification, OV-Encoder abandons uniform computation to focus exclusively on the 3.1%-25% of regions rich in signal entropy. To unify spatial and temporal reasoning under irregular token layouts, OneVision-Encoder employs a shared 3D RoPE and is trained with a large-scale cluster discrimination objective over more than one million semantic concepts, jointly capturing object permanence and motion dynamics. Evidence. The results validate our core hypothesis: efficiency and accuracy are not a trade-off; they are positively correlated. When integrated into LLM, it consistently outperforms strong vision backbones such as Qwen3-ViT and SigLIP2 across 16 image, video, and document understanding benchmarks, despite using substantially fewer visual tokens and pretraining data. Notably, on video understanding tasks, OV-Encoder achieves an average improvement of 4.1% over Qwen3-ViT. Codec-aligned, patch-level sparsity is a foundational principle, enabling OV-Encoder as a scalable engine for next-generation visual generalists.
>
---
#### [replaced 004] G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.12099v2](https://arxiv.org/pdf/2510.12099v2)**

> **作者:** Junfeng Ni; Yixin Chen; Zhifei Yang; Yu Liu; Ruijie Lu; Song-Chun Zhu; Siyuan Huang
>
> **备注:** ICLR'26. Project page: https://dali-jack.github.io/g4splat-web/
>
> **摘要:** Despite recent advances in leveraging generative prior from pre-trained diffusion models for 3D scene reconstruction, existing methods still face two critical limitations. First, due to the lack of reliable geometric supervision, they struggle to produce high-quality reconstructions even in observed regions, let alone in unobserved areas. Second, they lack effective mechanisms to mitigate multi-view inconsistencies in the generated images, leading to severe shape-appearance ambiguities and degraded scene geometry. In this paper, we identify accurate geometry as the fundamental prerequisite for effectively exploiting generative models to enhance 3D scene reconstruction. We first propose to leverage the prevalence of planar structures to derive accurate metric-scale depth maps, providing reliable supervision in both observed and unobserved regions. Furthermore, we incorporate this geometry guidance throughout the generative pipeline to improve visibility mask estimation, guide novel view selection, and enhance multi-view consistency when inpainting with video diffusion models, resulting in accurate and consistent scene completion. Extensive experiments on Replica, ScanNet++, DeepBlending and Mip-NeRF 360 show that our method consistently outperforms existing baselines in both geometry and appearance reconstruction, particularly for unobserved regions. Moreover, our method naturally supports single-view inputs and unposed videos, with strong generalizability in both indoor and outdoor scenarios with practical real-world applicability. The project page is available at https://dali-jack.github.io/g4splat-web/.
>
---
#### [replaced 005] CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.10981v4](https://arxiv.org/pdf/2503.10981v4)**

> **作者:** Fawaz Sammani; Jonas Fischer; Nikos Deligiannis
>
> **备注:** CVPR 2026 (Findings)
>
> **摘要:** Concept Bottleneck Models (CBMs) map dense feature representations into human-interpretable concepts which are then combined linearly to make a prediction. However, modern CBMs rely on the CLIP model to obtain image-concept annotations, and it remains unclear how to design CBMs without the CLIP bottleneck. Methods that do not use CLIP instead require manual, labor intensive annotation to associate feature representations with concepts. Furthermore, all CBMs necessitate training a linear classifier to map the extracted concepts to class labels. In this work, we lift all three limitations simultaneously by proposing a method that converts any frozen visual classifier into a CBM without requiring image-concept labels (label-free), without relying on the CLIP model (CLIP-free), and by deriving the linear classifier in an unsupervised manner. Our method is formulated by aligning the original classifier's distribution (over discrete class indices) with its corresponding vision-language counterpart distribution derived from textual class names, while preserving the classifier's performance. The approach requires no ground-truth image-class annotations, and is highly data-efficient and preserves the classifier's reasoning process. Applied and tested on over 40 visual classifiers, our resulting unsupervised, label-free and CLIP-free CBM (U-F$^2$-CBM) sets a new state of the art, surpassing even supervised CLIP-based CBMs. We also show that our method can be used for zero-shot image captioning, outperforming existing methods based on CLIP, and achieving state-of-art.
>
---
#### [replaced 006] A.I.R.: Enabling Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04428v2](https://arxiv.org/pdf/2510.04428v2)**

> **作者:** Yuanhao Zou; Shengji Jin; Andong Deng; Youpeng Zhao; Jun Wang; Chen Chen
>
> **备注:** ICLR 2026 Paper
>
> **摘要:** Effectively applying Vision-Language Models (VLMs) to Video Question Answering (VideoQA) hinges on selecting a concise yet comprehensive set of frames, as processing entire videos is computationally infeasible. However, current frame selection methods face a critical trade-off: approaches relying on lightweight similarity models, such as CLIP, often fail to capture the nuances of complex queries, resulting in inaccurate similarity scores that cannot reflect the authentic query-frame relevance, which further undermines frame selection. Meanwhile, methods that leverage a VLM for deeper analysis achieve higher accuracy but incur prohibitive computational costs. To address these limitations, we propose A.I.R., a training-free approach for Adaptive, Iterative, and Reasoning-based frame selection. We leverage a powerful VLM to perform deep, semantic analysis on complex queries, and this analysis is deployed within a cost-effective iterative loop that processes only a small batch of the most high-potential frames at a time. Extensive experiments on various VideoQA benchmarks demonstrate that our approach outperforms existing frame selection methods, significantly boosts the performance of the foundation VLM, and achieves substantial gains in computational efficiency over other VLM-based techniques.
>
---
#### [replaced 007] Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04714v4](https://arxiv.org/pdf/2510.04714v4)**

> **作者:** KunHo Heo; GiHyun Kim; SuYeon Kim; MyeongAh Cho
>
> **备注:** Accepted by NeurIPS 2025. Code: https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes
>
> **摘要:** 3D Semantic Scene Graph Prediction aims to detect objects and their semantic relationships in 3D scenes, and has emerged as a crucial technology for robotics and AR/VR applications. While previous research has addressed dataset limitations and explored various approaches including Open-Vocabulary settings, they frequently fail to optimize the representational capacity of object and relationship features, showing excessive reliance on Graph Neural Networks despite insufficient discriminative capability. In this work, we demonstrate through extensive analysis that the quality of object features plays a critical role in determining overall scene graph accuracy. To address this challenge, we design a highly discriminative object feature encoder and employ a contrastive pretraining strategy that decouples object representation learning from the scene graph prediction. This design not only enhances object classification accuracy but also yields direct improvements in relationship prediction. Notably, when plugging in our pretrained encoder into existing frameworks, we observe substantial performance improvements across all evaluation metrics. Additionally, whereas existing approaches have not fully exploited the integration of relationship information, we effectively combine both geometric and semantic features to achieve superior relationship prediction. Comprehensive experiments on the 3DSSG dataset demonstrate that our approach significantly outperforms previous state-of-the-art methods. Our code is publicly available at https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes.
>
---
#### [replaced 008] Establishing Stochastic Object Models from Noisy Data via Ambient Measurement-Integrated Diffusion
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14187v2](https://arxiv.org/pdf/2512.14187v2)**

> **作者:** Jianwei Sun; Xiaoning Lei; Wenhao Cai; Xichen Xu; Yanshu Wang; Hu Gao
>
> **摘要:** Task-based measures of image quality (IQ) are critical for evaluating medical imaging systems, which must account for randomness including anatomical variability. Stochastic object models (SOMs) provide a statistical description of such variability, but conventional mathematical SOMs fail to capture realistic anatomy, while data-driven approaches typically require clean data rarely available in clinical tasks. To address this challenge, we propose AMID, an unsupervised Ambient Measurement-Integrated Diffusion with noise decoupling, which establishes clean SOMs directly from noisy measurements. AMID introduces a measurement-integrated strategy aligning measurement noise with the diffusion trajectory, and explicitly models coupling between measurement and diffusion noise across steps, an ambient loss is thus designed base on it to learn clean SOMs. Experiments on real CT and mammography datasets show that AMID outperforms existing methods in generation fidelity and yields more reliable task-based IQ evaluation, demonstrating its potential for unsupervised medical imaging analysis.
>
---
#### [replaced 009] EndoDDC: Learning Sparse to Dense Reconstruction for Endoscopic Robotic Navigation via Diffusion Depth Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21893v2](https://arxiv.org/pdf/2602.21893v2)**

> **作者:** Yinheng Lin; Yiming Huang; Beilei Cui; Long Bai; Huxin Gao; Hongliang Ren; Jiewen Lai
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Accurate depth estimation plays a critical role in the navigation of endoscopic surgical robots, forming the foundation for 3D reconstruction and safe instrument guidance. Fine-tuning pretrained models heavily relies on endoscopic surgical datasets with precise depth annotations. While existing self-supervised depth estimation techniques eliminate the need for accurate depth annotations, their performance degrades in environments with weak textures and variable lighting, leading to sparse reconstruction with invalid depth estimation. Depth completion using sparse depth maps can mitigate these issues and improve accuracy. Despite the advances in depth completion techniques in general fields, their application in endoscopy remains limited. To overcome these limitations, we propose EndoDDC, an endoscopy depth completion method that integrates images, sparse depth information with depth gradient features, and optimizes depth maps through a diffusion model, addressing the issues of weak texture and light reflection in endoscopic environments. Extensive experiments on two publicly available endoscopy datasets show that our approach outperforms state-of-the-art models in both depth accuracy and robustness. This demonstrates the potential of our method to reduce visual errors in complex endoscopic environments. Our code will be released at https://github.com/yinheng-lin/EndoDDC.
>
---
#### [replaced 010] GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.12099v2](https://arxiv.org/pdf/2602.12099v2)**

> **作者:** GigaBrain Team; Boyuan Wang; Bohan Li; Chaojun Ni; Guan Huang; Guosheng Zhao; Hao Li; Jie Li; Jindi Lv; Jingyu Liu; Lv Feng; Mingming Yu; Peng Li; Qiuping Deng; Tianze Liu; Xinyu Zhou; Xinze Chen; Xiaofeng Wang; Yang Wang; Yifan Li; Yifei Nie; Yilong Li; Yukun Zhou; Yun Ye; Zhichao Liu; Zheng Zhu
>
> **备注:** https://gigabrain05m.github.io/
>
> **摘要:** Vision-language-action (VLA) models that directly predict multi-step action chunks from current observations face inherent limitations due to constrained scene understanding and weak future anticipation capabilities. In contrast, video world models pre-trained on web-scale video corpora exhibit robust spatiotemporal reasoning and accurate future prediction, making them a natural foundation for enhancing VLA learning. Therefore, we propose \textit{GigaBrain-0.5M*}, a VLA model trained via world model-based reinforcement learning. Built upon \textit{GigaBrain-0.5}, which is pre-trained on over 10,000 hours of robotic manipulation data, whose intermediate version currently ranks first on the international RoboChallenge benchmark. \textit{GigaBrain-0.5M*} further integrates world model-based reinforcement learning via \textit{RAMP} (Reinforcement leArning via world Model-conditioned Policy) to enable robust cross-task adaptation. Empirical results demonstrate that \textit{RAMP} achieves substantial performance gains over the RECAP baseline, yielding improvements of approximately 30\% on challenging tasks including \texttt{Laundry Folding}, \texttt{Box Packing}, and \texttt{Espresso Preparation}. Critically, \textit{GigaBrain-0.5M$^*$} exhibits reliable long-horizon execution, consistently accomplishing complex manipulation tasks without failure as validated by real-world deployment videos on our \href{https://gigabrain05m.github.io}{project page}.
>
---
#### [replaced 011] Visible Light Positioning With Lamé Curve LEDs: A Generic Approach for Camera Pose Estimation
- **分类: eess.SP; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01577v2](https://arxiv.org/pdf/2602.01577v2)**

> **作者:** Wenxuan Pan; Yang Yang; Dong Wei; Zhiyu Zhu; Jintao Wang; Huan Wu; Yao Nie
>
> **备注:** Submitted to an IEEE journal for possible publication
>
> **摘要:** Camera-based visible light positioning (VLP) is a promising technique for accurate and low-cost indoor camera pose estimation (CPE). To reduce the number of required light-emitting diodes (LEDs), advanced methods commonly exploit LED shape features for positioning. Although interesting, they are typically restricted to a single LED geometry, leading to failure in heterogeneous LED-shape scenarios. To address this challenge, this paper investigates Lamé curves as a unified representation of common LED shapes and proposes a generic VLP algorithm using Lamé curve-shaped LEDs, termed LC-VLP. In the considered system, multiple ceiling-mounted Lamé curve-shaped LEDs periodically broadcast their curve parameters via visible light communication, which are captured by a camera-equipped receiver. Based on the received LED images and curve parameters, the receiver can estimate the camera pose using LC-VLP. Specifically, an LED database is constructed offline to store the curve parameters, while online positioning is formulated as a nonlinear least-squares problem and solved iteratively. To provide a reliable initialization, a correspondence-free perspective-n-points (FreePnP) algorithm is further developed, enabling approximate CPE without any pre-calibrated reference points. The performance of LC-VLP is verified by both simulations and experiments. Simulations show that LC-VLP outperforms state-of-the-art methods in both circular- and rectangular-LED scenarios, achieving reductions of over 40% in position error and 25% in rotation error. Experiments further show that LC-VLP can achieve an average position accuracy of less than 4 cm.
>
---
#### [replaced 012] ST-GS: Vision-Based 3D Semantic Occupancy Prediction with Spatial-Temporal Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，旨在解决多视角空间交互不足和多帧时间一致性差的问题。提出ST-GS框架，增强空间与时间建模能力。**

- **链接: [https://arxiv.org/pdf/2509.16552v2](https://arxiv.org/pdf/2509.16552v2)**

> **作者:** Xiaoyang Yan; Muleilan Pei; Shaojie Shen
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** 3D occupancy prediction is critical for comprehensive scene understanding in vision-centric autonomous driving. Recent advances have explored utilizing 3D semantic Gaussians to model occupancy while reducing computational overhead, but they remain constrained by insufficient multi-view spatial interaction and limited multi-frame temporal consistency. To overcome these issues, in this paper, we propose a novel Spatial-Temporal Gaussian Splatting (ST-GS) framework to enhance both spatial and temporal modeling in existing Gaussian-based pipelines. Specifically, we develop a guidance-informed spatial aggregation strategy within a dual-mode attention mechanism to strengthen spatial interaction in Gaussian representations. Furthermore, we introduce a geometry-aware temporal fusion scheme that effectively leverages historical context to improve temporal continuity in scene completion. Extensive experiments on the large-scale nuScenes occupancy prediction benchmark showcase that our proposed approach not only achieves state-of-the-art performance but also delivers markedly better temporal consistency compared to existing Gaussian-based methods.
>
---
#### [replaced 013] USF-Net: A Unified Spatiotemporal Fusion Network for Ground-Based Remote Sensing Cloud Image Sequence Extrapolation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09045v2](https://arxiv.org/pdf/2511.09045v2)**

> **作者:** Penghui Niu; Taotao Cai; Suqi Zhang; Junhua Gua; Ping Zhanga; Qiqi Liu; Jianxin Li
>
> **摘要:** Ground-based remote sensing cloud image sequence extrapolation is a key research area in the development of photovoltaic power systems. However, existing approaches exhibit several limitations:(1)they primarily rely on static kernels to augment feature information, lacking adaptive mechanisms to extract features at varying resolutions dynamically;(2)temporal guidance is insufficient, leading to suboptimal modeling of long-range spatiotemporal dependencies; and(3)the quadratic computational cost of attention mechanisms is often overlooked, limiting efficiency in practical deployment. To address these challenges, we propose USF-Net, a Unified Spatiotemporal Fusion Network that integrates adaptive large-kernel convolutions and a low-complexity attention mechanism, combining temporal flow information within an encoder-decoder framework. Specifically, the encoder employs three basic layers to extract features. Followed by the USTM, which comprises:(1)a SiB equipped with a SSM that dynamically captures multi-scale contextual information, and(2)a TiB featuring a TAM that effectively models long-range temporal dependencies while maintaining computational efficiency. In addition, a DSM with a TGM is introduced to enable unified modeling of temporally guided spatiotemporal dependencies. On the decoder side, a DUM is employed to address the common "ghosting effect." It utilizes the initial temporal state as an attention operator to preserve critical motion signatures. As a key contribution, we also introduce and release the ASI-CIS dataset. Extensive experiments on ASI-CIS demonstrate that USF-Net significantly outperforms state-of-the-art methods, establishing a superior balance between prediction accuracy and computational efficiency for ground-based cloud extrapolation. The dataset and source code will be available at https://github.com/she1110/ASI-CIS.
>
---
#### [replaced 014] Unveiling Deep Shadows: A Survey and Benchmark on Image and Video Shadow Detection, Removal, and Generation in the Deep Learning Era
- **分类: cs.CV; cs.GR; cs.MM**

- **链接: [https://arxiv.org/pdf/2409.02108v3](https://arxiv.org/pdf/2409.02108v3)**

> **作者:** Xiaowei Hu; Zhenghao Xing; Tianyu Wang; Chi-Wing Fu; Pheng-Ann Heng
>
> **备注:** Accepted by International Journal of Computer Vision (IJCV). Publicly available results, trained models, and evaluation metrics at https://github.com/xw-hu/Unveiling-Deep-Shadows
>
> **摘要:** Shadows, formed by the occlusion of light, play an essential role in visual perception and directly influence scene understanding, image quality, and visual realism. This paper presents a unified survey and benchmark of deep-learning-based shadow detection, removal, and generation across images and videos. We introduce consistent taxonomies for architectures, supervision strategies, and learning paradigms; review major datasets and evaluation protocols; and re-train representative methods under standardized settings to enable fair comparison. Our benchmark reveals key findings, including inconsistencies in prior reports, strong dependence on model design and resolution, and limited cross-dataset generalization due to dataset bias. By synthesizing insights across the three tasks, we highlight shared illumination cues and priors that connect detection, removal, and generation. We further outline future directions involving unified all-in-one frameworks, semantics- and geometry-aware reasoning, shadow-based AIGC authenticity analysis, and the integration of physics-guided priors into multimodal foundation models. Corrected datasets, trained models, and evaluation tools are released to support reproducible research.
>
---
#### [replaced 015] Motion-Aware Animatable Gaussian Avatars Deblurring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16758v2](https://arxiv.org/pdf/2411.16758v2)**

> **作者:** Muyao Niu; Yifan Zhan; Qingtian Zhu; Zhuoxiao Li; Wei Wang; Zhihang Zhong; Xiao Sun; Yinqiang Zheng
>
> **备注:** CVPR2026, https://github.com/MyNiuuu/MAD-Avatar
>
> **摘要:** The creation of 3D human avatars from multi-view videos is a significant yet challenging task in computer vision. However, existing techniques rely on high-quality, sharp images as input, which are often impractical to obtain in real-world scenarios due to variations in human motion speed and intensity. This paper introduces a novel method for directly reconstructing sharp 3D human Gaussian avatars from blurry videos. The proposed approach incorporates a 3D-aware, physics-based model of blur formation caused by human motion, together with a 3D human motion model designed to resolve ambiguities in motion-induced blur. This framework enables the joint optimization of the avatar representation and motion parameters from a coarse initialization. Comprehensive benchmarks are established using both a synthetic dataset and a real-world dataset captured with a 360-degree synchronous hybrid-exposure camera system. Extensive evaluations demonstrate the effectiveness and robustness of the model across diverse conditions.
>
---
#### [replaced 016] Diffusion Model in Latent Space for Medical Image Segmentation Task
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.01292v3](https://arxiv.org/pdf/2512.01292v3)**

> **作者:** Huynh Trinh Ngoc; Toan Nguyen Hai; Ba Luong Son; Long Tran Quoc
>
> **摘要:** Medical image segmentation is crucial for clinical diagnosis and treatment planning. Traditional methods typically produce a single segmentation mask, failing to capture inherent uncertainty. Recent generative models enable the creation of multiple plausible masks per image, mimicking the collaborative interpretation of several clinicians. However, these approaches remain computationally heavy. We propose MedSegLatDiff, a diffusion based framework that combines a variational autoencoder (VAE) with a latent diffusion model for efficient medical image segmentation. The VAE compresses the input into a low dimensional latent space, reducing noise and accelerating training, while the diffusion process operates directly in this compact representation. We further replace the conventional MSE loss with weighted cross entropy in the VAE mask reconstruction path to better preserve tiny structures such as small nodules. MedSegLatDiff is evaluated on ISIC-2018 (skin lesions), CVC-Clinic (polyps), and LIDC-IDRI (lung nodules). It achieves state of the art or highly competitive Dice and IoU scores while simultaneously generating diverse segmentation hypotheses and confidence maps. This provides enhanced interpretability and reliability compared to deterministic baselines, making the model particularly suitable for clinical deployment.
>
---
#### [replaced 017] From Open Vocabulary to Open World: Teaching Vision Language Models to Detect Novel Objects
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2411.18207v4](https://arxiv.org/pdf/2411.18207v4)**

> **作者:** Zizhao Li; Zhengkang Xiang; Joseph West; Kourosh Khoshelham
>
> **备注:** Accepted by BMVC 2025
>
> **摘要:** Traditional object detection methods operate under the closed-set assumption, where models can only detect a fixed number of objects predefined in the training set. Recent works on open vocabulary object detection (OVD) enable the detection of objects defined by an in-principle unbounded vocabulary, which reduces the cost of training models for specific tasks. However, OVD heavily relies on accurate prompts provided by an ``oracle'', which limits their use in critical applications such as driving scene perception. OVD models tend to misclassify near-out-of-distribution (NOOD) objects that have similar features to known classes, and ignore far-out-of-distribution (FOOD) objects. To address these limitations, we propose a framework that enables OVD models to operate in open world settings, by identifying and incrementally learning previously unseen objects. To detect FOOD objects, we propose Open World Embedding Learning (OWEL) and introduce the concept of Pseudo Unknown Embedding which infers the location of unknown classes in a continuous semantic space based on the information of known classes. We also propose Multi-Scale Contrastive Anchor Learning (MSCAL), which enables the identification of misclassified unknown objects by promoting the intra-class consistency of object embeddings at different scales. The proposed method achieves state-of-the-art performance on standard open world object detection and autonomous driving benchmarks while maintaining its open vocabulary object detection capability.
>
---
#### [replaced 018] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04504v2](https://arxiv.org/pdf/2510.04504v2)**

> **作者:** Zijing Hu; Yunze Tong; Fengda Zhang; Junkun Yuan; Jun Xiao; Kun Kuang
>
> **备注:** Accepted to ICLR 2026, 25 pages, 13 figures, 6 tables
>
> **摘要:** Diffusion models have achieved impressive results in generating high-quality images. Yet, they often struggle to faithfully align the generated images with the input prompts. This limitation is associated with synchronous denoising, where all pixels simultaneously evolve from random noise to clear images. As a result, during generation, the prompt-related regions can only reference the unrelated regions at the same noise level, failing to obtain clear context and ultimately impairing text-to-image alignment. To address this issue, we propose asynchronous diffusion models -- a novel framework that allocates distinct timesteps to different pixels and reformulates the pixel-wise denoising process. By dynamically modulating the timestep schedules of individual pixels, prompt-related regions are denoised more gradually than unrelated regions, thereby allowing them to leverage clearer inter-pixel context. Consequently, these prompt-related regions achieve better alignment in the final images. Extensive experiments demonstrate that our asynchronous diffusion models can significantly improve text-to-image alignment across diverse prompts. The code repository for this work is available at https://github.com/hu-zijing/AsynDM.
>
---
#### [replaced 019] Breaking the Visual Shortcuts in Multimodal Knowledge-Based Visual Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22843v2](https://arxiv.org/pdf/2511.22843v2)**

> **作者:** Dosung Lee; Sangwon Jung; Boyoung Kim; Minyoung Kim; Sungyeon Kim; Junyoung Sung; Paul Hongsuck Seo
>
> **摘要:** Existing Multimodal Knowledge-Based Visual Question Answering (MKB-VQA) benchmarks suffer from "visual shortcuts", as the query image typically matches the primary subject entity of the target document. We demonstrate that models can exploit these shortcuts, achieving comparable results using visual cues alone. To address this, we introduce Relational Entity Text-Image kNowledge Augmented (RETINA) benchmark, automatically constructed using an LLM-driven pipeline, consisting of 120k training and 2k human-curated test set. RETINA contains queries referencing secondary subjects (i.e. related entities) and pairs them with images of these related entities, removing the visual shortcut. When evaluated on RETINA existing models show significantly degraded performance, confirming their reliance on the shortcut. Furthermore, we propose Multi-Image MultImodal Retriever (MIMIR), which enriches document embeddings by augmenting images of multiple related entities, effectively handling RETINA, unlike prior work that uses only a single image per document. Our experiments validate the limitations of existing benchmarks and demonstrate the effectiveness of RETINA and MIMIR. Our project is available at: Project Page.
>
---
#### [replaced 020] Visual Instruction Pretraining for Domain-Specific Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17562v4](https://arxiv.org/pdf/2509.17562v4)**

> **作者:** Yuxuan Li; Yicheng Zhang; Wenhao Tang; Yimian Dai; Ming-Ming Cheng; Xiang Li; Jian Yang
>
> **摘要:** Modern computer vision is converging on a closed loop in which perception, reasoning and generation mutually reinforce each other. However, this loop remains incomplete: the top-down influence of high-level reasoning on the foundational learning of low-level perceptual features is not yet underexplored. This paper addresses this gap by proposing a new paradigm for pretraining foundation models in downstream domains. We introduce Visual insTruction Pretraining (ViTP), a novel approach that directly leverages reasoning to enhance perception. ViTP embeds a Vision Transformer (ViT) backbone within a Vision-Language Model and pretrains it end-to-end using a rich corpus of visual instruction data curated from target downstream domains. ViTP is powered by our proposed Visual Robustness Learning (VRL), which compels the ViT to learn robust and domain-relevant features from a sparse set of visual tokens. Extensive experiments on 16 challenging remote sensing and medical imaging benchmarks demonstrate that ViTP establishes new state-of-the-art performance across a diverse range of downstream tasks. The code is available at https://github.com/zcablii/ViTP.
>
---
#### [replaced 021] StableMaterials: Enhancing Diversity in Material Generation via Semi-Supervised Learning
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2406.09293v4](https://arxiv.org/pdf/2406.09293v4)**

> **作者:** Giuseppe Vecchio
>
> **摘要:** We introduce StableMaterials, a novel approach for generating photorealistic physical-based rendering (PBR) materials that integrate semi-supervised learning with Latent Diffusion Models (LDMs). Our method employs adversarial training to distill knowledge from existing large-scale image generation models, minimizing the reliance on annotated data and enhancing the diversity in generation. This distillation approach aligns the distribution of the generated materials with that of image textures from an SDXL model, enabling the generation of novel materials that are not present in the initial training dataset. Furthermore, we employ a diffusion-based refiner model to improve the visual quality of the samples and achieve high-resolution generation. Finally, we distill a latent consistency model for fast generation in just four steps and propose a new tileability technique that removes visual artifacts typically associated with fewer diffusion steps. We detail the architecture and training process of StableMaterials, the integration of semi-supervised training within existing LDM frameworks and show the advantages of our approach. Comparative evaluations with state-of-the-art methods show the effectiveness of StableMaterials, highlighting its potential applications in computer graphics and beyond. StableMaterials is publicly available at https://gvecchio.com/stablematerials.
>
---
#### [replaced 022] PCReg-Net: Progressive Contrast-Guided Registration for Cross-Domain Image Alignment
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.13304v2](https://arxiv.org/pdf/2602.13304v2)**

> **作者:** Jiahao Qin
>
> **备注:** 11 pages, 1 figure, 3 tables
>
> **摘要:** Deformable image registration across heterogeneous domains remains challenging because coupled appearance variation and geometric misalignment violate the brightness constancy assumption underlying conventional methods. We propose PCReg-Net, a progressive contrast-guided registration framework that performs coarse-to-fine alignment through four lightweight modules: (1)~a registration U-Net for initial coarse alignment, (2)~a reference feature extractor capturing multi-scale structural cues from the fixed image, (3)~a multi-scale contrast module that identifies residual misalignment by comparing coarse-registered and reference features, and (4)~a refinement U-Net with feature injection that produces the final high-fidelity output. We evaluate on the FIRE-Reg-256 retinal fundus benchmark, demonstrating improvements over both traditional and deep learning baselines. Additional experiments on two microscopy benchmarks further confirm cross-domain applicability. With only 2.56M parameters, PCReg-Net achieves real-time inference at 141 FPS. Code is available at https://github.com/JiahaoQin/PCReg-Net.
>
---
#### [replaced 023] Loc$^2$: Interpretable Cross-View Localization via Depth-Lifted Local Feature Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09792v3](https://arxiv.org/pdf/2509.09792v3)**

> **作者:** Zimin Xia; Chenghao Xu; Alexandre Alahi
>
> **摘要:** We propose an accurate and interpretable fine-grained cross-view localization method that estimates the 3 Degrees of Freedom (DoF) pose of a ground-level image by matching its local features with a reference aerial image. Unlike prior approaches that rely on global descriptors or bird's-eye-view (BEV) transformations, our method directly learns ground-aerial image-plane correspondences using weak supervision from camera poses. The matched ground points are lifted into BEV space with monocular depth predictions, and scale-aware Procrustes alignment is then applied to estimate camera rotation, translation, and optionally the scale between relative depth and the aerial metric space. This formulation is lightweight, end-to-end trainable, and requires no pixel-level annotations. Experiments show state-of-the-art accuracy in challenging scenarios such as cross-area testing and unknown orientation. Furthermore, our method offers strong interpretability: correspondence quality directly reflects localization accuracy and enables outlier rejection via RANSAC, while overlaying the re-scaled ground layout on the aerial image provides an intuitive visual cue of localization performance.
>
---
#### [replaced 024] Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21058v2](https://arxiv.org/pdf/2512.21058v2)**

> **作者:** Minghao Han; Yichen Liu; Yizhou Liu; Zizhi Chen; Jingqun Tang; Xuecheng Wu; Dingkang Yang; Lihua Zhang
>
> **备注:** accepted by CVPR 2026; 32 pages, 17 figures, and 6 tables
>
> **摘要:** In computational pathology, understanding and generation have evolved along disparate paths: advanced understanding models already exhibit diagnostic-level competence, whereas generative models largely simulate pixels. Progress remains hindered by three coupled factors: the scarcity of large, high-quality image-text corpora; the lack of precise, fine-grained semantic control, which forces reliance on non-semantic cues; and terminological heterogeneity, where diverse phrasings for the same diagnostic concept impede reliable text conditioning. We introduce UniPath, a semantics-driven pathology image generation framework that leverages mature diagnostic understanding to enable controllable generation. UniPath implements Multi-Stream Control: a Raw-Text stream; a High-Level Semantics stream that uses learnable queries to a frozen pathology MLLM to distill paraphrase-robust Diagnostic Semantic Tokens and to expand prompts into diagnosis-aware attribute bundles; and a Prototype stream that affords component-level morphological control via a prototype bank. On the data front, we curate a 2.65M image-text corpus and a finely annotated, high-quality 68K subset to alleviate data scarcity. For a comprehensive assessment, we establish a four-tier evaluation hierarchy tailored to pathology. Extensive experiments demonstrate UniPath's SOTA performance, including a Patho-FID of 80.9 (51% better than the second-best) and fine-grained semantic control achieving 98.7% of the real-image. The dataset and code can be obtained from https://github.com/Hanminghao/UniPath.
>
---
#### [replaced 025] ThinkRL-Edit: Thinking in Reinforcement Learning for Reasoning-Centric Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.03467v3](https://arxiv.org/pdf/2601.03467v3)**

> **作者:** Hengjia Li; Liming Jiang; Qing Yan; Yizhi Song; Hao Kang; Zichuan Liu; Xin Lu; Boxi Wu; Deng Cai
>
> **摘要:** Instruction-driven image editing with unified multimodal generative models has advanced rapidly, yet their underlying visual reasoning remains limited, leading to suboptimal performance on reasoning-centric edits. Reinforcement learning (RL) has been investigated for improving the quality of image editing, but it faces three key challenges: (1) limited reasoning exploration confined to denoising stochasticity, (2) biased reward fusion, and (3) unstable VLM-based instruction rewards. In this work, we propose ThinkRL-Edit, a reasoning-centric RL framework that decouples visual reasoning from image synthesis and expands reasoning exploration beyond denoising. To the end, we introduce Chain-of-Thought (CoT)-based reasoning sampling with planning and reflection stages prior to generation in online sampling, compelling the model to explore multiple semantic hypotheses and validate their plausibility before committing to a visual outcome. To avoid the failures of weighted aggregation, we propose an unbiased chain preference grouping strategy across multiple reward dimensions. Moreover, we replace interval-based VLM scores with a binary checklist, yielding more precise, lower-variance, and interpretable rewards for complex reasoning. Experiments show our method significantly outperforms prior work on reasoning-centric image editing, producing instruction-faithful, visually coherent, and semantically grounded edits.
>
---
#### [replaced 026] SuperQuadricOcc: Multi-Layer Gaussian Approximation of Superquadrics for Real-Time Self-Supervised Occupancy Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17361v3](https://arxiv.org/pdf/2511.17361v3)**

> **作者:** Seamie Hayes; Reenu Mohandas; Tim Brophy; Alexandre Boulch; Ganesh Sistu; Ciaran Eising
>
> **摘要:** Semantic occupancy estimation enables comprehensive scene understanding for automated driving, providing dense spatial and semantic information essential for perception and planning. While Gaussian representations have been widely adopted in self-supervised occupancy estimation, the deployment of a large number of Gaussian primitives drastically increases memory requirements and is not suitable for real-time inference. In contrast, superquadrics permit reduced primitive count and lower memory requirements due to their diverse shape set. However, implementation into a self-supervised occupancy model is nontrivial due to the absence of a superquadric rasterizer to enable model supervision. Our proposed method, SuperQuadricOcc, employs a superquadric-based scene representation. By leveraging a multi-layer icosphere-tessellated Gaussian approximation of superquadrics, we enable Gaussian rasterization for supervision during training. On the Occ3D dataset, SuperQuadricOcc achieves a 75% reduction in memory footprint, 124% faster inference, and a 5.9% improvement in mIoU compared to previous Gaussian-based methods, without the use of temporal labels. To our knowledge, this is the first occupancy model to enable real-time inference while maintaining competitive performance. The use of superquadrics reduces the number of primitives required for scene modeling by 84% relative to Gaussian-based approaches. Finally, evaluation against prior methods is facilitated by our fast superquadric voxelization module. The code will be made available at https://github.com/seamie6/SuperQuadricOcc.
>
---
#### [replaced 027] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.00037v2](https://arxiv.org/pdf/2504.00037v2)**

> **作者:** Guoyizhe Wei; Rama Chellappa
>
> **摘要:** Vision Transformers (ViTs) have delivered remarkable progress through global self-attention, yet their quadratic complexity can become prohibitive for high-resolution inputs. In this work, we present ViT-Linearizer, a cross-architecture distillation framework that transfers rich ViT representations into a linear-time, recurrent-style model. Our approach leverages 1) activation matching, an intermediate constraint that encourages student to align its token-wise dependencies with those produced by the teacher, and 2) masked prediction, a contextual reconstruction objective that requires the student to predict the teacher's representations for unseen (masked) tokens, to effectively distill the quadratic self-attention knowledge into the student while maintaining efficient complexity. Empirically, our method provides notable speedups particularly for high-resolution tasks, significantly addressing the hardware challenges in inference. Additionally, it also elevates Mamba-based architectures' performance on standard vision benchmarks, achieving a competitive 84.3% top-1 accuracy on ImageNet with a base-sized model. Our results underscore the good potential of RNN-based solutions for large-scale visual tasks, bridging the gap between theoretical efficiency and real-world practice.
>
---
#### [replaced 028] Multi-View Camera System for Variant-Aware Autonomous Vehicle Inspection and Defect Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.26454v3](https://arxiv.org/pdf/2509.26454v3)**

> **作者:** Yash Kulkarni; Raman Jha; Renu Kachhoria
>
> **摘要:** Ensuring that every vehicle leaving a modern production line is built to the correct \emph{variant} specification and is free from visible defects is an increasingly complex challenge. We present the \textbf{Automated Vehicle Inspection (AVI)} platform, an end-to-end, \emph{multi-view} perception system that couples deep-learning detectors with a semantic rule engine to deliver \emph{variant-aware} quality control in real time. Eleven synchronized cameras capture a full 360° sweep of each vehicle; task-specific views are then routed to specialised modules: YOLOv8 for part detection, EfficientNet for ICE/EV classification, Gemini-1.5 Flash for mascot OCR, and YOLOv8-Seg for scratch-and-dent segmentation. A view-aware fusion layer standardises evidence, while a VIN-conditioned rule engine compares detected features against the expected manifest, producing an interpretable pass/fail report in \(\approx\! 300\,\text{ms}\). On a mixed data set of Original Equipment Manufacturer(OEM) vehicle data sets of four distinct models plus public scratch/dent images, AVI achieves \textbf{ 93 \%} verification accuracy, \textbf{86 \%} defect-detection recall, and sustains \(\mathbf{3.3}\) vehicles/min, surpassing single-view or no segmentation baselines by large margins. To our knowledge, this is the first publicly reported system that unifies multi-camera feature validation with defect detection in a deployable automotive setting in industry.
>
---
#### [replaced 029] IV-tuning: Parameter-Efficient Transfer Learning for Infrared-Visible Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.16654v5](https://arxiv.org/pdf/2412.16654v5)**

> **作者:** Yaming Zhang; Chenqiang Gao; Fangcen Liu; Junjie Guo; Lan Wang; Xinggan Peng; Deyu Meng
>
> **摘要:** Existing infrared and visible (IR-VIS) methods inherit the general representations of Pre-trained Visual Models (PVMs) to facilitate complementary learning. However, our analysis indicates that under the full fine-tuning paradigm, the feature space becomes highly constrained and low-ranked, which has been proven to seriously impair generalization. One remedy is to freeze the parameters, which preserves pretrained knowledge and helps maintain feature diversity. To this end, we propose IV-tuning, to parameter-efficiently harness PVMs for various IR-VIS downstream tasks, including salient object detection, semantic segmentation, and object detection. Extensive experiments across various settings demonstrate that IV-tuning outperforms previous state-of-the-art methods, and exhibits superior generalization and scalability. Remarkably, with only a single backbone, IV-tuning effectively facilitates the complementary learning of infrared and visible modalities with merely 3% trainable backbone parameters, and achieves superior computational efficiency compared to conventional IR-VIS paradigms.
>
---
#### [replaced 030] RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.14377v5](https://arxiv.org/pdf/2502.14377v5)**

> **作者:** Ke Cao; Jing Wang; Ao Ma; Jiasong Feng; Xuanhua He; Run Ling; Haowei Liu; Jian Lu; Wei Feng; Haozhe Wang; Hongjuan Pei; Yihua Shao; Zhanjie Zhang; Jie Zhang
>
> **备注:** AAAI 2026
>
> **摘要:** The Diffusion Transformer plays a pivotal role in advancing text-to-image and text-to-video generation, owing primarily to its inherent scalability. However, existing controlled diffusion transformer methods incur significant parameter and computational overheads and suffer from inefficient resource allocation due to their failure to account for the varying relevance of control information across different transformer layers. To address this, we propose the Relevance-Guided Efficient Controllable Generation framework, RelaCtrl, enabling efficient and resource-optimized integration of control signals into the Diffusion Transformer. First, we evaluate the relevance of each layer in the Diffusion Transformer to the control information by assessing the "ControlNet Relevance Score"-i.e., the impact of skipping each control layer on both the quality of generation and the control effectiveness during inference. Based on the strength of the relevance, we then tailor the positioning, parameter scale, and modeling capacity of the control layers to reduce unnecessary parameters and redundant computations. Additionally, to further improve efficiency, we replace the self-attention and FFN in the commonly used copy block with the carefully designed Two-Dimensional Shuffle Mixer (TDSM), enabling efficient implementation of both the token mixer and channel mixer. Both qualitative and quantitative experimental results demonstrate that our approach achieves superior performance with only 15% of the parameters and computational complexity compared to PixArt-delta.
>
---
#### [replaced 031] Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决高速飞行无人机的运动模糊和位姿漂移问题。通过融合事件流和模糊图像，优化NeRF并提升位姿估计精度。**

- **链接: [https://arxiv.org/pdf/2602.21101v2](https://arxiv.org/pdf/2602.21101v2)**

> **作者:** Rong Zou; Marco Cannici; Davide Scaramuzza
>
> **摘要:** Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.
>
---
#### [replaced 032] Deforming Videos to Masks: Flow Matching for Referring Video Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.06139v2](https://arxiv.org/pdf/2510.06139v2)**

> **作者:** Zanyi Wang; Dengyang Jiang; Liuzhuozheng Li; Sizhe Dang; Chengzu Li; Harry Yang; Guang Dai; Mengmeng Wang; Jingdong Wang
>
> **摘要:** Referring Video Object Segmentation (RVOS) requires segmenting specific objects in a video guided by a natural language description. The core challenge of RVOS is to anchor abstract linguistic concepts onto a specific set of pixels and continuously segment them through the complex dynamics of a video. Faced with this difficulty, prior work has often decomposed the task into a pragmatic `locate-then-segment' pipeline. However, this cascaded design creates an information bottleneck by simplifying semantics into coarse geometric prompts (e.g, point), and struggles to maintain temporal consistency as the segmenting process is often decoupled from the initial language grounding. To overcome these fundamental limitations, we propose FlowRVS, a novel framework that reconceptualizes RVOS as a conditional continuous flow problem. This allows us to harness the inherent strengths of pretrained T2V models, fine-grained pixel control, text-video semantic alignment, and temporal coherence. Instead of conventional generating from noise to mask or directly predicting mask, we reformulate the task by learning a direct, language-guided deformation from a video's holistic representation to its target mask. Our one-stage, generative approach achieves new state-of-the-art results across all major RVOS benchmarks. Specifically, achieving a J&F of 51.1 in MeViS (+1.6 over prior SOTA) and 73.3 in the zero shot Ref-DAVIS17 (+2.7), demonstrating the significant potential of modeling video understanding tasks as continuous deformation processes.
>
---
#### [replaced 033] Bridging Geometric and Semantic Foundation Models for Generalized Monocular Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23400v2](https://arxiv.org/pdf/2505.23400v2)**

> **作者:** Sanggyun Ma; Wonjoon Choi; Jihun Park; Jaeyeul Kim; Seunghun Lee; Jiwan Seo; Sunghoon Im
>
> **摘要:** We present Bridging Geometric and Semantic (BriGeS), an effective method that fuses geometric and semantic information within foundation models to enhance Monocular Depth Estimation (MDE). Central to BriGeS is the Bridging Gate, which integrates the complementary strengths of depth and segmentation foundation models. This integration is further refined by our Attention Temperature Scaling technique. It finely adjusts the focus of the attention mechanisms to prevent over-concentration on specific features, thus ensuring balanced performance across diverse inputs. BriGeS capitalizes on pre-trained foundation models and adopts a strategy that focuses on training only the Bridging Gate. This method significantly reduces resource demands and training time while maintaining the model's ability to generalize effectively. Extensive experiments across multiple challenging datasets demonstrate that BriGeS outperforms state-of-the-art methods in MDE for complex scenes, effectively handling intricate structures and overlapping objects.
>
---
#### [replaced 034] VQ-Style: Disentangling Style and Content in Motion with Residual Quantized Representations
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.02334v2](https://arxiv.org/pdf/2602.02334v2)**

> **作者:** Fatemeh Zargarbashi; Dhruv Agrawal; Jakob Buhmann; Martin Guay; Stelian Coros; Robert W. Sumner
>
> **摘要:** Human motion data is inherently rich and complex, containing both semantic content and subtle stylistic features that are challenging to model. We propose a novel method for effective disentanglement of the style and content in human motion data to facilitate style transfer. Our approach is guided by the insight that content corresponds to coarse motion attributes while style captures the finer, expressive details. To model this hierarchy, we employ Residual Vector Quantized Variational Autoencoders (RVQ-VAEs) to learn a coarse-to-fine representation of motion. We further enhance the disentanglement by integrating codebook learning with contrastive learning and a novel information leakage loss to organize the content and the style across different codebooks. We harness this disentangled representation using our simple and effective inference-time technique Quantized Code Swapping, which enables motion style transfer without requiring any fine-tuning for unseen styles. Our framework demonstrates strong versatility across multiple inference applications, including style transfer, style removal, and motion blending.
>
---
#### [replaced 035] Enhancing Multi-Modal LLMs Reasoning via Difficulty-Aware Group Normalization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21743v2](https://arxiv.org/pdf/2602.21743v2)**

> **作者:** Jinghan Li; Junfeng Fang; Jinda Lu; Yuan Wang; Xiaoyan Guo; Tianyu Zhang; Xiang Wang; Xiangnan He
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) and Group Relative Policy Optimization (GRPO) have significantly advanced the reasoning capabilities of large language models. Extending these methods to multimodal settings, however, faces a critical challenge: the instability of std-based normalization, which is easily distorted by extreme samples with nearly positive or negative rewards. Unlike pure-text LLMs, multimodal models are particularly sensitive to such distortions, as both perceptual and reasoning errors influence their responses. To address this, we characterize each sample by its difficulty, defined through perceptual complexity (measured via visual entropy) and reasoning uncertainty (captured by model confidence). Building on this characterization, we propose difficulty-aware group normalization (Durian), which re-groups samples by difficulty levels and shares the std within each group. Our approach preserves GRPO's intra-group distinctions while eliminating sensitivity to extreme cases, yielding significant performance gains across multiple multimodal reasoning benchmarks.
>
---
#### [replaced 036] Solaris: Building a Multiplayer Video World Model in Minecraft
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22208v2](https://arxiv.org/pdf/2602.22208v2)**

> **作者:** Georgy Savva; Oscar Michel; Daohan Lu; Suppakit Waiwitlikhit; Timothy Meehan; Dhairya Mishra; Srivats Poddar; Jack Lu; Saining Xie
>
> **备注:** Project website: https://solaris-wm.github.io/
>
> **摘要:** Existing action-conditioned video generation models (video world models) are limited to single-agent perspectives, failing to capture the multi-agent interactions of real-world environments. We introduce Solaris, a multiplayer video world model that simulates consistent multi-view observations. To enable this, we develop a multiplayer data system designed for robust, continuous, and automated data collection on video games such as Minecraft. Unlike prior platforms built for single-player settings, our system supports coordinated multi-agent interaction and synchronized videos + actions capture. Using this system, we collect 12.64 million multiplayer frames and propose an evaluation framework for multiplayer movement, memory, grounding, building, and view consistency. We train Solaris using a staged pipeline that progressively transitions from single-player to multiplayer modeling, combining bidirectional, causal, and Self Forcing training. In the final stage, we introduce Checkpointed Self Forcing, a memory-efficient Self Forcing variant that enables a longer-horizon teacher. Results show our architecture and training design outperform existing baselines. Through open-sourcing our system and models, we hope to lay the groundwork for a new generation of multi-agent world models.
>
---
#### [replaced 037] Hepato-LLaVA: An Expert MLLM with Sparse Topo-Pack Attention for Hepatocellular Pathology Analysis on Whole Slide Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19424v2](https://arxiv.org/pdf/2602.19424v2)**

> **作者:** Yuxuan Yang; Zhonghao Yan; Yi Zhang; Bo Yun; Muxi Diao; Guowei Zhao; Kongming Liang; Wenbin Li; Zhanyu Ma
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Hepatocellular Carcinoma diagnosis relies heavily on the interpretation of gigapixel Whole Slide Images. However, current computational approaches are constrained by fixed-resolution processing mechanisms and inefficient feature aggregation, which inevitably lead to either severe information loss or high feature redundancy. To address these challenges, we propose Hepato-LLaVA, a specialized Multi-modal Large Language Model designed for fine-grained hepatocellular pathology analysis. We introduce a novel Sparse Topo-Pack Attention mechanism that explicitly models 2D tissue topology. This mechanism effectively aggregates local diagnostic evidence into semantic summary tokens while preserving global context. Furthermore, to overcome the lack of multi-scale data, we present HepatoPathoVQA, a clinically grounded dataset comprising 33K hierarchically structured question-answer pairs validated by expert pathologists. Our experiments demonstrate that Hepato-LLaVA achieves state-of-the-art performance on HCC diagnosis and captioning tasks, significantly outperforming existing methods. Our code and implementation details are available at https://pris-cv.github.io/Hepto-LLaVA/.
>
---
#### [replaced 038] RAP: Real-time Audio-driven Portrait Animation with Video Diffusion Transformer
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出RAP框架，解决实时音频驱动的人像动画生成问题。通过混合注意力机制和静态-动态训练策略，实现实时高质量视频生成。**

- **链接: [https://arxiv.org/pdf/2508.05115v2](https://arxiv.org/pdf/2508.05115v2)**

> **作者:** Fangyu Du; Taiqing Li; Qian Qiao; Tan Yu; Ziwei Zhang; Dingcheng Zhen; Xu Jia; Yang Yang; Shunshun Yin; Siyuan Liu
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Audio-driven portrait animation aims to synthesize realistic and natural talking head videos from an input audio signal and a single reference image. While existing methods achieve high-quality results by leveraging high-dimensional intermediate representations and explicitly modeling motion dynamics, their computational complexity renders them unsuitable for real-time deployment. Real-time inference imposes stringent latency and memory constraints, often necessitating the use of highly compressed latent representations. However, operating in such compact spaces hinders the preservation of fine-grained spatiotemporal details, thereby complicating audio-visual synchronization RAP (Real-time Audio-driven Portrait animation), a unified framework for generating high-quality talking portraits under real-time constraints. Specifically, RAP introduces a hybrid attention mechanism for fine-grained audio control, and a static-dynamic training-inference paradigm that avoids explicit motion supervision. Through these techniques, RAP achieves precise audio-driven control, mitigates long-term temporal drift, and maintains high visual fidelity. Extensive experiments demonstrate that RAP achieves state-of-the-art performance while operating under real-time constraints.
>
---
#### [replaced 039] Sparse Imagination for Efficient Visual World Model Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉世界模型规划任务，旨在解决机器人决策中的计算资源限制问题。通过稀疏想象方法，减少预测过程中的token数量，提升效率并保持控制精度。**

- **链接: [https://arxiv.org/pdf/2506.01392v2](https://arxiv.org/pdf/2506.01392v2)**

> **作者:** Junha Chun; Youngjoon Jeong; Taesup Kim
>
> **备注:** Accepted to ICLR 2026; Project Page: https://nikriz1.github.io/sparse_imagination/
>
> **摘要:** World model based planning has significantly improved decision-making in complex environments by enabling agents to simulate future states and make informed choices. This computational burden is particularly restrictive in robotics, where resources are severely constrained. To address this limitation, we propose a Sparse Imagination for Efficient Visual World Model Planning, which enhances computational efficiency by reducing the number of tokens processed during forward prediction. Our method leverages a sparsely trained vision-based world model based on transformers with randomized grouped attention strategy, allowing the model to flexibly adjust the number of tokens processed based on the computational resource. By enabling sparse imagination during latent rollout, our approach significantly accelerates planning while maintaining high control fidelity. Experimental results demonstrate that sparse imagination preserves task performance while dramatically improving inference efficiency. This general technique for visual planning is applicable from simple test-time trajectory optimization to complex real-world tasks with the latest VLAs, enabling the deployment of world models in real-time scenarios.
>
---
#### [replaced 040] Learning with less: label-efficient land cover classification at very high spatial resolution using self-supervised deep learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.03004v2](https://arxiv.org/pdf/2511.03004v2)**

> **作者:** Dakota Hester; Vitor S. Martins; Lucas B. Ferreira; Thainara M. A. Lima
>
> **备注:** 36 pages, 14 figures. Published in Science of Remote Sensing
>
> **摘要:** Deep learning semantic segmentation methods have shown promising performance for very high 1-m resolution land cover classification, but the challenge of collecting large volumes of representative training data creates a significant barrier to widespread adoption of such models for meter-scale land cover mapping over large areas. In this study, we present a novel label-efficient approach for statewide 1-m land cover classification using only 1,000 annotated reference image patches with self-supervised deep learning. We use the "Bootstrap Your Own Latent" pre-training strategy with a large amount of unlabeled color-infrared aerial images (377,921 patches of 256x256 pixels at 1-m resolution) to pre-train a ResNet-101 convolutional encoder. The learned encoder weights were subsequently transferred into multiple deep semantic segmentation architectures (FCN, U-Net, Attention U-Net, DeepLabV3+, UPerNet, PAN), which were then fine-tuned using very small training dataset sizes with cross-validation (250, 500, 750 patches). Among the fine-tuned models, we obtained 87.14% overall accuracy and 75.58% macro F1 score using an ensemble of the best-performing U-Net models for comprehensive 1-m, 8-class land cover mapping, covering more than 123 billion pixels over the state of Mississippi, USA. Detailed qualitative and quantitative analysis revealed accurate mapping of open water and forested areas, while highlighting challenges in accurate delineation between cropland, herbaceous, and barren land cover types. These results show that self-supervised learning is an effective strategy for reducing the need for large volumes of manually annotated data, directly addressing a major limitation to high spatial resolution land cover mapping at scale.
>
---
#### [replaced 041] Secure and reversible face anonymization with diffusion models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.01031v2](https://arxiv.org/pdf/2510.01031v2)**

> **作者:** Pol Labarbarie; Vincent Itier; William Puech
>
> **摘要:** Face anonymization aims to protect sensitive identity information by altering faces while preserving visual realism and utility for downstream computer vision tasks. Current methods struggle to simultaneously ensure high image quality, strong security guarantees, and controlled reversibility for authorized identity recovery at a later time. To improve the image quality of generated anonymized faces, recent methods have adopted diffusion models. However, these new diffusion-based anonymization methods do not provide a mechanism to restrict de-anonymization to trusted parties, limiting their real-world applicability. In this paper, we present the first diffusion-based framework for secure, reversible face anonymization via secret-key conditioning. Our method injects the secret key directly into the diffusion process, enabling anonymization and authorized face reconstruction while preventing unauthorized de-anonymization. The use of deterministic forward and reverse diffusion steps guarantees exact identity recovery when the correct secret key is available. Experiments on CelebA-HQ and LFW demonstrate that our approach achieves better anonymization and de-anonymization capabilities than prior work. We also show that our method remains robust to incorrect or adversarial key de-anonymization. Our code will be made publicly available.
>
---
#### [replaced 042] Towards Generating Realistic 3D Semantic Training Data for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.21449v2](https://arxiv.org/pdf/2503.21449v2)**

> **作者:** Lucas Nunes; Rodrigo Marcuzzi; Jens Behley; Cyrill Stachniss
>
> **摘要:** Semantic scene understanding is crucial for robotics and computer vision applications. In autonomous driving, 3D semantic segmentation plays an important role for enabling safe navigation. Despite significant advances in the field, the complexity of collecting and annotating 3D data is a bottleneck in this developments. To overcome that data annotation limitation, synthetic simulated data has been used to generate annotated data on demand. There is still, however, a domain gap between real and simulated data. More recently, diffusion models have been in the spotlight, enabling close-to-real data synthesis. Those generative models have been recently applied to the 3D data domain for generating scene-scale data with semantic annotations. Still, those methods either rely on image projection or decoupled models trained with different resolutions in a coarse-to-fine manner. Such intermediary representations impact the generated data quality due to errors added in those transformations. In this work, we propose a novel approach able to generate 3D semantic scene-scale data without relying on any projection or decoupled trained multi-resolution models, achieving more realistic semantic scene data generation compared to previous state-of-the-art methods. Besides improving 3D semantic scene-scale data synthesis, we thoroughly evaluate the use of the synthetic scene samples as labeled data to train a semantic segmentation network. In our experiments, we show that using the synthetic annotated data generated by our method as training data together with the real semantic segmentation labels, leads to an improvement in the semantic segmentation model performance. Our results show the potential of generated scene-scale point clouds to generate more training data to extend existing datasets, reducing the data annotation effort. Our code is available at https://github.com/PRBonn/3DiSS.
>
---
#### [replaced 043] Automated Disentangling Analysis of Skin Colour for Lesion Images
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19055v2](https://arxiv.org/pdf/2602.19055v2)**

> **作者:** Wenbo Yang; Eman Rezk; Walaa M. Moursi; Zhou Wang
>
> **摘要:** Machine-learning models applied to skin images often have degraded performance when the skin colour captured in images (SCCI) differs between training and deployment. These discrepancies arise from a combination of entangled environmental factors (e.g., illumination, camera settings) and intrinsic factors (e.g., skin tone) that cannot be accurately described by a single "skin tone" scalar -- a simplification commonly adopted by prior work. To mitigate such colour mismatches, we propose a skin-colour disentangling framework that adapts disentanglement-by-compression to learn a structured, manipulable latent space for SCCI from unlabelled dermatology images. To prevent information leakage that hinders proper learning of dark colour features, we introduce a randomized, mostly monotonic decolourization mapping. To suppress unintended colour shifts of localized patterns (e.g., ink marks, scars) during colour manipulation, we further propose a geometry-aligned post-processing step. Together, these components enable faithful counterfactual editing and answering an essential question: "What would this skin condition look like under a different SCCI?", as well as direct colour transfer between images and controlled traversal along physically meaningful directions (e.g., blood perfusion, camera white balance), enabling educational visualization of skin conditions under varying SCCI. We demonstrate that dataset-level augmentation and colour normalization based on our framework achieve competitive lesion classification performance. Ultimately, our work promotes equitable diagnosis through creating diverse training datasets that include different skin tones and image-capturing conditions.
>
---
#### [replaced 044] Aligning Few-Step Diffusion Models with Dense Reward Difference Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.11727v2](https://arxiv.org/pdf/2411.11727v2)**

> **作者:** Ziyi Zhang; Li Shen; Sen Zhang; Deheng Ye; Yong Luo; Miaojing Shi; Dongjing Shan; Bo Du; Dacheng Tao
>
> **备注:** Accepted by IEEE TPAMI
>
> **摘要:** Few-step diffusion models enable efficient high-resolution image synthesis but struggle to align with specific downstream objectives due to limitations of existing reinforcement learning (RL) methods in low-step regimes with limited state spaces and suboptimal sample quality. To address this, we propose Stepwise Diffusion Policy Optimization (SDPO), a novel RL framework tailored for few-step diffusion models. SDPO introduces a dual-state trajectory sampling mechanism, tracking both noisy and predicted clean states at each step to provide dense reward feedback and enable low-variance, mixed-step optimization. For further efficiency, we develop a latent similarity-based dense reward prediction strategy to minimize costly dense reward queries. Leveraging these dense rewards, SDPO optimizes a dense reward difference learning objective that enables more frequent and granular policy updates. Additional refinements, including stepwise advantage estimates, temporal importance weighting, and step-shuffled gradient updates, further enhance long-term dependency, low-step priority, and gradient stability. Our experiments demonstrate that SDPO consistently delivers superior reward-aligned results across diverse few-step settings and tasks. Code is available at https://github.com/ZiyiZhang27/sdpo.
>
---
#### [replaced 045] DICArt: Advancing Category-level Articulated Object Pose Estimation in Discrete State-Spaces
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.19565v2](https://arxiv.org/pdf/2602.19565v2)**

> **作者:** Li Zhang; Mingyu Mei; Ailing Wang; Xianhui Meng; Yan Zhong; Xinyuan Song; Liu Liu; Rujing Wang; Zaixing He; Cewu Lu
>
> **摘要:** Articulated object pose estimation is a core task in embodied AI. Existing methods typically regress poses in a continuous space, but often struggle with 1) navigating a large, complex search space and 2) failing to incorporate intrinsic kinematic constraints. In this work, we introduce DICArt (DIsCrete Diffusion for Articulation Pose Estimation), a novel framework that formulates pose estimation as a conditional discrete diffusion process. Instead of operating in a continuous domain, DICArt progressively denoises a noisy pose representation through a learned reverse diffusion procedure to recover the GT pose. To improve modeling fidelity, we propose a flexible flow decider that dynamically determines whether each token should be denoised or reset, effectively balancing the real and noise distributions during diffusion. Additionally, we incorporate a hierarchical kinematic coupling strategy, estimating the pose of each rigid part hierarchically to respect the object's kinematic structure. We validate DICArt on both synthetic and real-world datasets. Experimental results demonstrate its superior performance and robustness. By integrating discrete generative modeling with structural priors, DICArt offers a new paradigm for reliable category-level 6D pose estimation in complex environments.
>
---
#### [replaced 046] Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.20570v2](https://arxiv.org/pdf/2508.20570v2)**

> **作者:** Lorenz Hufe; Constantin Venhoff; Erblina Purelku; Maximilian Dreyer; Sebastian Lapuschkin; Wojciech Samek
>
> **摘要:** Typographic attacks exploit multi-modal systems by injecting text into images, leading to targeted misclassifications, malicious content generation and even Vision-Language Model jailbreaks. In this work, we analyze how CLIP vision encoders behave under typographic attacks, locating specialized attention heads in the latter half of the model's layers that causally extract and transmit typographic information to the cls token. Building on these insights, we introduce Dyslexify - a method to defend CLIP models against typographic attacks by selectively ablating a typographic circuit, consisting of attention heads. Without requiring finetuning, dyslexify improves performance by up to 22.06% on a typographic variant of ImageNet-100, while reducing standard ImageNet-100 accuracy by less than 1%, and demonstrate its utility in a medical foundation model for skin lesion diagnosis. Notably, our training-free approach remains competitive with current state-of-the-art typographic defenses that rely on finetuning. To this end, we release a family of dyslexic CLIP models which are significantly more robust against typographic attacks. These models serve as suitable drop-in replacements for a broad range of safety-critical applications, where the risks of text-based manipulation outweigh the utility of text recognition.
>
---
#### [replaced 047] CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22150v2](https://arxiv.org/pdf/2602.22150v2)**

> **作者:** YuXin Song; Yu Lu; Haoyuan Sun; Huanjin Yao; Fanglong Liu; Yifan Sun; Haocheng Feng; Hang Zhou; Jingdong Wang
>
> **备注:** Accepted by CVPR2026. 15 pages, 8 figures
>
> **摘要:** Unified conditional image generation remains difficult because different tasks depend on fundamentally different internal representations. Some require conceptual understanding for semantic synthesis, while others rely on localization cues for spatial precision. Forcing these heterogeneous tasks to share a single representation leads to concept-localization representational conflict. To address this issue, we propose CoLoGen, a unified diffusion framework that progressively learns and reconciles this concept-localization duality. CoLoGen uses a staged curriculum that first builds core conceptual and localization abilities, then adapts them to diverse visual conditions, and finally refines their synergy for complex instruction-driven tasks. Central to this process is the Progressive Representation Weaving (PRW) module, which dynamically routes features to specialized experts and stably integrates their outputs across stages. Experiments on editing, controllable generation, and customized generation show that CoLoGen achieves competitive or superior performance, offering a principled representational perspective for unified image generation.
>
---
#### [replaced 048] Towards Privacy-Guaranteed Label Unlearning in Vertical Federated Learning: Few-Shot Forgetting without Disclosure
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2410.10922v3](https://arxiv.org/pdf/2410.10922v3)**

> **作者:** Hanlin Gu; Hong Xi Tae; Chee Seng Chan; Lixin Fan
>
> **备注:** We introduce the first method for label unlearning in vertical federated learning (VFL), focused on preventing label leakage by the active party
>
> **摘要:** This paper addresses the critical challenge of unlearning in Vertical Federated Learning (VFL), a setting that has received far less attention than its horizontal counterpart. Specifically, we propose the first method tailored to \textit{label unlearning} in VFL, where labels play a dual role as both essential inputs and sensitive information. To this end, we employ a representation-level manifold mixup mechanism to generate synthetic embeddings for both unlearned and retained samples. This is to provide richer signals for the subsequent gradient-based label forgetting and recovery steps. These augmented embeddings are then subjected to gradient-based label forgetting, effectively removing the associated label information from the model. To recover performance on the retained data, we introduce a recovery-phase optimization step that refines the remaining embeddings. This design achieves effective label unlearning while maintaining computational efficiency. We validate our method through extensive experiments on diverse datasets, including MNIST, CIFAR-10, CIFAR-100, ModelNet, Brain Tumor MRI, COVID-19 Radiography, and Yahoo Answers demonstrate strong efficacy and scalability. Overall, this work establishes a new direction for unlearning in VFL, showing that re-imagining mixup as an efficient mechanism can unlock practical and utility-preserving unlearning. The code is publicly available at \href{https://github.com/bryanhx/Towards-Privacy-Guaranteed-Label-Unlearning-in-Vertical-Federated-Learning}{https://github.com/bryanhx/Towards-Privacy-Guaranteed-Label-Unlearning-in-Vertical-Federated-Learning}
>
---
#### [replaced 049] Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.02088v5](https://arxiv.org/pdf/2502.02088v5)**

> **作者:** Xiaomeng Yang; Mengping Yang; Jia Gong; Luozheng Qin; Zhiyu Tan; Hao Li
>
> **备注:** To appear in ICLR 2026, GitHub Code: https://github.com/SAIS-FUXI/IPO
>
> **摘要:** Recent advances in video generation have enabled thrilling experiences in producing realistic videos driven by scalable diffusion transformers. However, they usually fail to produce satisfactory outputs that are aligned to users' authentic demands and preferences. In this work, we introduce Dual-Iterative Optimization (Dual-IPO), an iterative paradigm that sequentially optimizes both the reward model and the video generation model for improved synthesis quality and human preference alignment. For the reward model, our framework ensures reliable and robust reward signals via CoT-guided reasoning, voting-based self-consistency, and preference certainty estimation. Given this, we optimize video foundation models with guidance of signals from reward model's feedback, thus improving the synthesis quality in subject consistency, motion smoothness and aesthetic quality, etc. The reward model and video generation model complement each other and are progressively improved in the multi-round iteration, without requiring tediously manual preference annotations. Comprehensive experiments demonstrate that the proposed Dual-IPO can effectively and consistently improve the video generation quality of base model with various architectures and sizes, even help a model with only 2B parameters surpass a 5B one. Moreover, our analysis experiments and ablation studies identify the rational of our systematic design and the efficacy of each component.
>
---
#### [replaced 050] GmNet: Revisiting Gating Mechanisms From A Frequency View
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.22841v3](https://arxiv.org/pdf/2503.22841v3)**

> **作者:** Yifan Wang; Xu Ma; Yitian Zhang; Zhongruo Wang; Sung-Cheol Kim; Vahid Mirjalili; Vidya Renganathan; Yun Fu
>
> **摘要:** Gating mechanisms have emerged as an effective strategy integrated into model designs beyond recurrent neural networks for addressing long-range dependency problems. In a broad understanding, it provides adaptive control over the information flow while maintaining computational efficiency. However, there is a lack of theoretical analysis on how the gating mechanism works in neural networks. In this paper, inspired by the \textit{convolution theorem}, we systematically explore the effect of gating mechanisms on the training dynamics of neural networks from a frequency perspective. We investigate the interact between the element-wise product and activation functions in managing the responses to different frequency components. Leveraging these insights, we propose a Gating Mechanism Network (GmNet), a lightweight model designed to efficiently utilize the information of various frequency components. It minimizes the low-frequency bias present in existing lightweight models. GmNet achieves impressive performance in terms of both effectiveness and efficiency in the image classification task.
>
---
#### [replaced 051] ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02686v2](https://arxiv.org/pdf/2512.02686v2)**

> **作者:** Yuxing Liu; Zheng Li; Huanhuan Liang; Ji Zhang; Zeyu Sun; Yong Liu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Anomaly segmentation seeks to detect and localize unknown or out-of-distribution (OoD) objects that fall outside predefined semantic classes a capability essential for safe autonomous driving. However, the scarcity and limited diversity of anomaly data severely constrain model generalization in open-world environments. Existing approaches mitigate this issue through synthetic data generation, either by copy-pasting external objects into driving scenes or by leveraging text-to-image diffusion models to inpaint anomalous regions. While these methods improve anomaly diversity, they often lack contextual coherence and physical realism, resulting in domain gaps between synthetic and real data. In this paper, we present ClimaDrive, a semantics-guided image-to-image framework for synthesizing semantically coherent, weather-diverse, and physically plausible OoD driving data. ClimaDrive unifies structure-guided multi-weather generation with prompt-driven anomaly inpainting, enabling the creation of visually realistic training data. Based on this framework, we construct ClimaOoD, a large-scale benchmark spanning six representative driving scenarios under both clear and adverse weather conditions. Extensive experiments on four state-of-the-art methods show that training with ClimaOoD leads to robust improvements in anomaly segmentation. Across all methods, AUROC, AP, and FPR95 show notable gains, with FPR95 dropping from 3.97 to 3.52 for RbA on Fishyscapes LAF. These results demonstrate that ClimaOoD enhances model robustness, offering valuable training data for better generalization in open-world anomaly detection.
>
---
#### [replaced 052] Depth from Defocus via Direct Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18509v2](https://arxiv.org/pdf/2602.18509v2)**

> **作者:** Holly Jackson; Caleb Adams; Ignacio Lopez-Francos; Benjamin Recht
>
> **摘要:** Though there exists a reasonable forward model for blur based on optical physics, recovering depth from a collection of defocused images remains a computationally challenging optimization problem. In this paper, we show that with contemporary optimization methods and reasonable computing resources, a global optimization approach to depth from defocus is feasible. Our approach rests on alternating minimization. When holding the depth map fixed, the forward model is linear with respect to the all-in-focus image. When holding the all-in-focus image fixed, the depth at each pixel can be computed independently, enabling embarrassingly parallel computation. We show that alternating between convex optimization and parallel grid search can effectively solve the depth-from-defocus problem at higher resolutions than current deep learning methods. We demonstrate our approach on benchmark datasets with synthetic and real defocus blur and show promising results compared to prior approaches. Our code is available at github.com/hollyjackson/dfd.
>
---
#### [replaced 053] Benchmarking Video Foundation Models for Remote Parkinson's Disease Screening
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.13507v2](https://arxiv.org/pdf/2602.13507v2)**

> **作者:** Md Saiful Islam; Ekram Hossain; Abdelrahman Abdelkader; Tariq Adnan; Fazla Rabbi Mashrur; Sooyong Park; Praveen Kumar; Qasim Sudais; Natalia Chunga; Nami Shah; Jan Freyberg; Christopher Kanan; Ruth Schneider; Ehsan Hoque
>
> **摘要:** Video-based assessments offer a scalable pathway for remote Parkinson's disease (PD) screening. While traditional approaches rely on handcrafted features mimicking clinical scales, recent advances in video foundation models (VFMs) enable representation learning without task-specific customization. However, the comparative effectiveness of different VFM architectures across diverse clinical tasks remains poorly understood. We present a large-scale systematic study using a novel video dataset from 1,888 participants (727 with PD), comprising 32,847 videos across 16 standardized clinical tasks. We evaluate seven state-of-the-art VFMs -- including VideoPrism, V-JEPA, ViViT, and VideoMAE -- to determine their robustness in clinical screening. By evaluating frozen embeddings with a linear classification head, we demonstrate that task saliency is highly model-dependent: VideoPrism excels in capturing visual speech kinematics (no audio) and facial expressivity, while V-JEPA proves superior for upper-limb motor tasks. Notably, TimeSformer remains highly competitive for rhythmic tasks like finger tapping. Our experiments yield AUCs of 76.4 - 85.3% and accuracies of 71.5 - 80.6%. While high specificity (up to 90.3%) suggests strong potential for ruling out healthy individuals, the lower sensitivity (43.2 - 57.3%) highlights the need for task-aware calibration and integration of multiple tasks and modalities. Overall, this work establishes a rigorous baseline for VFM-based PD screening and provides a roadmap for selecting suitable tasks and architectures in remote neurological monitoring. Code and anonymized structured data are publicly available: https://anonymous.4open.science/r/parkinson\_video\_benchmarking-A2C5
>
---
#### [replaced 054] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态任务，解决MLLM中的幻觉问题。通过GACD方法，估计并抑制文本与视觉的偏差，提升输出的视觉合理性。**

- **链接: [https://arxiv.org/pdf/2509.03113v4](https://arxiv.org/pdf/2509.03113v4)**

> **作者:** Shan Wang; Maying Shen; Nadine Chang; Chuong Nguyen; Hongdong Li; Jose M. Alvarez
>
> **摘要:** Multimodal large language models achieve strong performance across diverse tasks but remain prone to hallucinations, where outputs are not grounded in visual inputs. This issue can be attributed to two main biases: text-visual bias, the overreliance on prompts and prior outputs, and co-occurrence bias, spurious correlations between frequently paired objects. We propose Gradient-based Influence-Aware Constrained Decoding (GACD), an inference-based method, that addresses both biases without auxiliary models, and is readily applicable to existing models without finetuning. The core of our approach is bias estimation, which uses first-order Taylor gradients to understand the contribution of individual tokens-visual features and text tokens-to the current output. Based on this analysis, GACD mitigates hallucinations through two components: (1) suppressing spurious visual features correlated with the output objects, and (2) rebalancing cross-modal contributions by strengthening visual features relative to text. Experiments across multiple benchmarks demonstrate that GACD effectively reduces hallucinations and improves the visual grounding of MLLM outputs.
>
---
#### [replaced 055] TerraCodec: Compressing Optical Earth Observations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.12670v2](https://arxiv.org/pdf/2510.12670v2)**

> **作者:** Julen Costa-Watanabe; Isabelle Wittmann; Benedikt Blumenstiel; Konrad Schindler
>
> **摘要:** Earth observation (EO) satellites produce massive streams of multispectral image time series, posing pressing challenges for storage and transmission. Yet, learned EO compression remains fragmented and lacks publicly available, large-scale pretrained codecs. Moreover, prior work has largely focused on image compression, leaving temporal redundancy and EO video codecs underexplored. To address these gaps, we introduce TerraCodec (TEC), a family of learned codecs pretrained on Sentinel-2 EO data. TEC includes efficient multispectral image variants and a Temporal Transformer model (TEC-TT) that leverages dependencies across time. To overcome the fixed-rate setting of today's neural codecs, we present Latent Repacking, a novel method for training flexible-rate transformer models that operate on varying rate-distortion settings. TerraCodec outperforms classical codecs, achieving 3-10x higher compression at equivalent image quality. Beyond compression, TEC-TT enables zero-shot cloud inpainting, surpassing state-of-the-art methods on the AllClear benchmark. Our results establish neural codecs as a promising direction for Earth observation. Our code and models are publically available at https://github.com/IBM/TerraCodec.
>
---
#### [replaced 056] A Pragmatic VLA Foundation Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种视觉-语言-动作基础模型LingBot-VLA，旨在提升机器人操作的泛化能力和成本效率，通过大量真实数据训练并在多个平台上验证其性能。**

- **链接: [https://arxiv.org/pdf/2601.18692v2](https://arxiv.org/pdf/2601.18692v2)**

> **作者:** Wei Wu; Fan Lu; Yunnan Wang; Shuai Yang; Shi Liu; Fangjing Wang; Qian Zhu; He Sun; Yong Wang; Shuailei Ma; Yiyu Ren; Kejia Zhang; Hui Yu; Jingmei Zhao; Shuai Zhou; Zhenqi Qiu; Houlong Xiong; Ziyu Wang; Zechen Wang; Ran Cheng; Yong-Lu Li; Yongtao Huang; Xing Zhu; Yujun Shen; Kecheng Zheng
>
> **备注:** Project Webpage: https://technology.robbyant.com/lingbot-vla/, Code: https://github.com/Robbyant/lingbot-vla/, GM-100: https://huggingface.co/datasets/robbyant/lingbot-GM-100
>
> **摘要:** Offering great potential in robotic manipulation, a capable Vision-Language-Action (VLA) foundation model is expected to faithfully generalize across tasks and platforms while ensuring cost efficiency (e.g., data and GPU hours required for adaptation). To this end, we develop LingBot-VLA with around 20,000 hours of real-world data from 9 popular dual-arm robot configurations. Through a systematic assessment on 3 robotic platforms, each completing 100 tasks with 130 post-training episodes per task, our model achieves clear superiority over competitors, showcasing its strong performance and broad generalizability. We have also built an efficient codebase, which delivers a throughput of 261 samples per second with an 8-GPU training setup, representing a 1.5~2.8$\times$ (depending on the relied VLM base model) speedup over existing VLA-oriented codebases. The above features ensure that our model is well-suited for real-world deployment. To advance the field of robot learning, we provide open access to the code, base model, and benchmark data, with a focus on enabling more challenging tasks and promoting sound evaluation standards.
>
---
#### [replaced 057] PPT: Pretraining with Pseudo-Labeled Trajectories for Motion Forecasting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于运动预测任务，解决数据标注成本高、可扩展性差的问题。通过伪标签轨迹预训练，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2412.06491v3](https://arxiv.org/pdf/2412.06491v3)**

> **作者:** Yihong Xu; Yuan Yin; Éloi Zablocki; Tuan-Hung Vu; Alexandre Boulch; Matthieu Cord
>
> **备注:** 8 pages, 6 figures, accepted to ICRA 2026
>
> **摘要:** Accurately predicting how agents move in dynamic scenes is essential for safe autonomous driving. State-of-the-art motion forecasting models rely on datasets with manually annotated or post-processed trajectories. However, building these datasets is costly, generally manual, hard to scale, and lacks reproducibility. They also introduce domain gaps that limit generalization across environments. We introduce PPT (Pretraining with Pseudo-labeled Trajectories), a simple and scalable pretraining framework that uses unprocessed and diverse trajectories automatically generated from off-the-shelf 3D detectors and tracking. Unlike data annotation pipelines aiming for clean, single-label annotations, PPT is a pretraining framework embracing off-the-shelf trajectories as useful signals for learning robust representations. With optional finetuning on a small amount of labeled data, models pretrained with PPT achieve strong performance across standard benchmarks, particularly in low-data regimes, and in cross-domain, end-to-end, and multi-class settings. PPT is easy to implement and improves generalization in motion forecasting.
>
---
#### [replaced 058] Autoregressive Image Generation with Randomized Parallel Decoding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.10568v4](https://arxiv.org/pdf/2503.10568v4)**

> **作者:** Haopeng Li; Jinyue Yang; Guoqi Li; Huan Wang
>
> **备注:** The Fourteenth International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** We introduce ARPG, a novel visual Autoregressive model that enables Randomized Parallel Generation, addressing the inherent limitations of conventional raster-order approaches, which hinder inference efficiency and zero-shot generalization due to their sequential, predefined token generation order. Our key insight is that effective random-order modeling necessitates explicit guidance for determining the position of the next predicted token. To this end, we propose a novel decoupled decoding framework that decouples positional guidance from content representation, encoding them separately as queries and key-value pairs. By directly incorporating this guidance into the causal attention mechanism, our approach enables fully random-order training and generation, eliminating the need for bidirectional attention. Consequently, ARPG readily generalizes to zero-shot tasks such as image in-painting, out-painting, and resolution expansion. Furthermore, it supports parallel inference by concurrently processing multiple queries using a shared KV cache. On the ImageNet-1K 256 benchmark, our approach attains an FID of 1.83 with only 32 sampling steps, achieving over a 30 times speedup in inference and and a 75 percent reduction in memory consumption compared to representative recent autoregressive models at a similar scale.
>
---
#### [replaced 059] LinGuinE: Longitudinal Guidance Estimation for Volumetric Tumour Segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.06092v2](https://arxiv.org/pdf/2506.06092v2)**

> **作者:** Nadine Garibli; Mayank Patwari; Bence Csiba; Yi Wei; Kostantinos Sidiropoulos
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Longitudinal volumetric tumour segmentation is critical for radiotherapy planning and response assessment, yet this problem is underexplored and most methods produce single-timepoint semantic masks, lack lesion correspondence, and offer limited radiologist control. We introduce LinGuinE (Longitudinal Guidance Estimation), a PyTorch framework that combines image registration and guided segmentation to deliver lesion-level tracking and volumetric masks across all scans in a longitudinal study from a single radiologist prompt. LinGuinE is temporally direction agnostic, requires no training on longitudinal data, and allows any registration and semi-automatic segmentation algorithm to be repurposed for the task. We evaluate various combinations of registration and segmentation algorithms within the framework. LinGuinE achieves state-of-the-art segmentation and tracking performance across four datasets with a total of 456 longitudinal studies. Tumour segmentation performance shows minimal degradation with increasing temporal separation. We conduct ablation studies to determine the impact of autoregression, pathology specific finetuning, and the use of real radiologist prompts. We release our code and substantial public benchmarking for longitudinal segmentation, facilitating future research.
>
---
#### [replaced 060] LayerT2V: A Unified Multi-Layer Video Generation Framework
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2508.04228v2](https://arxiv.org/pdf/2508.04228v2)**

> **作者:** Guangzhao Li; Kangrui Cen; Baixuan Zhao; Yi Xin; Siqi Luo; Guangtao Zhai; Lei Zhang; Xiaohong Liu
>
> **备注:** Project Page is https://layert2v.github.io/
>
> **摘要:** Text-to-video generation has advanced rapidly, but existing methods typically output only the final composited video and lack editable layered representations, limiting their use in professional workflows. We propose \textbf{LayerT2V}, a unified multi-layer video generation framework that produces multiple semantically consistent outputs in a single inference pass: the full video, an independent background layer, and multiple foreground RGB layers with corresponding alpha mattes. Our key insight is that recent video generation backbones use high compression in both time and space, enabling us to serialize multiple layer representations along the temporal dimension and jointly model them on a shared generation trajectory. This turns cross-layer consistency into an intrinsic objective, improving semantic alignment and temporal coherence. To mitigate layer ambiguity and conditional leakage, we augment a shared DiT backbone with LayerAdaLN and layer-aware cross-attention modulation. LayerT2V is trained in three stages: alpha mask VAE adaptation, joint multi-layer learning, and multi-foreground extension. We also introduce \textbf{VidLayer}, the first large-scale dataset for multi-layer video generation. Extensive experiments demonstrate that LayerT2V substantially outperforms prior methods in visual fidelity, temporal consistency, and cross-layer coherence.
>
---
#### [replaced 061] HLGFA: High-Low Resolution Guided Feature Alignment for Unsupervised Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.09524v2](https://arxiv.org/pdf/2602.09524v2)**

> **作者:** Han Zhou; Yuxuan Gao; Yinchao Du; Xuezhe Zheng
>
> **备注:** 14 pages, 6 figures, references added
>
> **摘要:** Unsupervised industrial anomaly detection (UAD) is essential for modern manufacturing inspection, where defect samples are scarce and reliable detection is required. In this paper, we propose HLGFA, a high-low resolution guided feature alignment framework that learns normality by modeling cross-resolution feature consistency between high-resolution and low-resolution representations of normal samples, instead of relying on pixel-level reconstruction. Dual-resolution inputs are processed by a shared frozen backbone to extract multi-level features, and high-resolution representations are decomposed into structure and detail priors to guide the refinement of low-resolution features through conditional modulation and gated residual correction. During inference, anomalies are naturally identified as regions where cross-resolution alignment breaks down. In addition, a noise-aware data augmentation strategy is introduced to suppress nuisance-induced responses commonly observed in industrial environments. Extensive experiments on standard benchmarks demonstrate the effectiveness of HLGFA, achieving 97.9% pixel-level AUROC and 97.5% image-level AUROC on the MVTec AD dataset, outperforming representative reconstruction-based and feature-based methods.
>
---
#### [replaced 062] Self-adaptive Dataset Construction for Real-World Multimodal Safety Scenarios
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于多模态安全数据集构建任务，旨在解决现有数据集无法覆盖真实复杂场景的问题。提出一种图像导向的自适应构建方法，生成35k图像-文本对，并引入标准化评估指标。**

- **链接: [https://arxiv.org/pdf/2509.04403v2](https://arxiv.org/pdf/2509.04403v2)**

> **作者:** Jingen Qu; Lijun Li; Bo Zhang; Yichen Yan; Jing Shao
>
> **备注:** Accepted at EMNLP 2025 Findings
>
> **摘要:** Multimodal large language models (MLLMs) are rapidly evolving, presenting increasingly complex safety challenges. However, current dataset construction methods, which are risk-oriented, fail to cover the growing complexity of real-world multimodal safety scenarios (RMS). And due to the lack of a unified evaluation metric, their overall effectiveness remains unproven. This paper introduces a novel image-oriented self-adaptive dataset construction method for RMS, which starts with images and end constructing paired text and guidance responses. Using the image-oriented method, we automatically generate an RMS dataset comprising 35k image-text pairs with guidance responses. Additionally, we introduce a standardized safety dataset evaluation metric: fine-tuning a safety judge model and evaluating its capabilities on other safety datasets.Extensive experiments on various tasks demonstrate the effectiveness of the proposed image-oriented pipeline. The results confirm the scalability and effectiveness of the image-oriented approach, offering a new perspective for the construction of real-world multimodal safety datasets. The dataset is presented at https://huggingface.co/datasets/NewCityLetter/RMS2/tree/main.
>
---
#### [replaced 063] StruXLIP: Enhancing Vision-language Models with Multimodal Structural Cues
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.20089v2](https://arxiv.org/pdf/2602.20089v2)**

> **作者:** Zanxi Ruan; Qiuyu Kong; Songqun Gao; Yiming Wang; Marco Cristani
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Edge-based representations are fundamental cues for visual understanding, a principle rooted in early vision research and still central today. We extend this principle to vision-language alignment, showing that isolating and aligning structural cues across modalities can greatly benefit fine-tuning on long, detail-rich captions, with a specific focus on improving cross-modal retrieval. We introduce StruXLIP, a fine-tuning alignment paradigm that extracts edge maps (e.g., Canny), treating them as proxies for the visual structure of an image, and filters the corresponding captions to emphasize structural cues, making them "structure-centric". Fine-tuning augments the standard alignment loss with three structure-centric losses: (i) aligning edge maps with structural text, (ii) matching local edge regions to textual chunks, and (iii) connecting edge maps to color images to prevent representation drift. From a theoretical standpoint, while standard CLIP maximizes the mutual information between visual and textual embeddings, StruXLIP additionally maximizes the mutual information between multimodal structural representations. This auxiliary optimization is intrinsically harder, guiding the model toward more robust and semantically stable minima, enhancing vision-language alignment. Beyond outperforming current competitors on cross-modal retrieval in both general and specialized domains, our method serves as a general boosting recipe that can be integrated into future approaches in a plug-and-play manner. Code and pretrained models are publicly available at: https://github.com/intelligolabs/StruXLIP.
>
---
#### [replaced 064] Compact Hadamard Latent Codes for Efficient Spectral Rendering
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18741v2](https://arxiv.org/pdf/2602.18741v2)**

> **作者:** Jiaqi Yu; Dar'ya Guarnera; Giuseppe Claudio Guarnera
>
> **摘要:** Spectral rendering accurately reproduces wavelength-dependent appearance but is computationally expensive, as shading must be evaluated at many wavelength samples and scales roughly linearly with the number of samples. It also requires spectral textures and lights throughout the rendering pipeline. We propose Hadamard spectral codes, a compact latent representation that enables spectral rendering using standard RGB rendering operations. Spectral images are approximated with a small number of RGB rendering passes, followed by a decoding step. Our key requirement is latent linearity: scaling and addition in spectral space correspond to scaling and addition of codes, and the element-wise product of spectra (for example reflectance times illumination) is approximated by the element-wise product of their latent codes. We show that an exact low-dimensional algebra-preserving representation cannot exist for arbitrary spectra when the latent dimension k is smaller than the number of spectral samples n. We therefore introduce a learned non-negative linear encoder and decoder architecture that preserves scaling and addition exactly while encouraging approximate multiplicativity under the Hadamard product. With k = 6, we render k/3 = 2 RGB images per frame using an unmodified RGB renderer, reconstruct the latent image, and decode to high-resolution spectra or XYZ or RGB. Experiments on 3D scenes demonstrate that k = 6 significantly reduces color error compared to RGB baselines while being substantially faster than naive n-sample spectral rendering. Using k = 9 provides higher-quality reference results. We further introduce a lightweight neural upsampling network that maps RGB assets directly to latent codes, enabling integration of legacy RGB content into the spectral pipeline while maintaining perceptually accurate colors in rendered images.
>
---
#### [replaced 065] Diffusion or Non-Diffusion Adversarial Defenses: Rethinking the Relation between Classifier and Adversarial Purifier
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.16904v2](https://arxiv.org/pdf/2501.16904v2)**

> **作者:** Yuan-Chih Chen; Chun-Shien Lu
>
> **摘要:** Adversarial defense research continues to face challenges in combating against advanced adversarial attacks, yet with diffusion models increasingly favoring their defensive capabilities. Unlike most prior studies that focus on diffusion models for test-time defense, we explore the generalization loss in classifiers caused by diffusion models. We compare diffusion-based and non-diffusion-based adversarial purifiers, demonstrating that non-diffusion models can also achieve well performance under a practical setting of non-adaptive attack. While non-diffusion models show promising adversarial robustness, they particularly excel in defense transferability and color generalization without relying on additional data beyond the training set. Notably, a non-diffusion model trained on CIFAR-10 achieves state-of-the-art performance when tested directly on ImageNet, surpassing existing diffusion-based models trained specifically on ImageNet.
>
---
#### [replaced 066] SplatSDF: Boosting SDF-NeRF via Architecture-Level Fusion with Gaussian Splats
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于环境建模任务，旨在解决SDF-NeRF训练慢、收敛难的问题。通过融合3D高斯点云，提升收敛速度与精度。**

- **链接: [https://arxiv.org/pdf/2411.15468v2](https://arxiv.org/pdf/2411.15468v2)**

> **作者:** Runfa Blark Li; Keito Suzuki; Bang Du; Ki Myung Brian Lee; Nikolay Atanasov; Truong Nguyen
>
> **摘要:** Signed distance-radiance field (SDF-NeRF) is a promising environment representation that offers both photo-realistic rendering and geometric reasoning such as proximity queries for collision avoidance. However, the slow training speed and convergence of SDF-NeRF hinder their use in practical robotic systems. We propose SplatSDF, a novel SDF-NeRF architecture that accelerates convergence using 3D Gaussian splats (3DGS), which can be quickly pre-trained. Unlike prior approaches that introduce a consistency loss between separate 3DGS and SDF-NeRF models, SplatSDF directly fuses 3DGS at an architectural level by consuming it as an input to SDF-NeRF during training. This is achieved using a novel sparse 3DGS fusion strategy that injects neural embeddings of 3DGS into SDF-NeRF around the object surface, while also permitting inference without 3DGS for minimal operation. Experimental results show SplatSDF achieves 3X faster convergence to the same geometric accuracy than the best baseline, and outperforms state-of-the-art SDF-NeRF methods in terms of chamfer distance and peak signal to noise ratio, unlike consistency loss-based approaches that in fact provide limited gains. We also present computational techniques for accelerating gradient and Hessian steps by 3X. We expect these improvements will contribute to deploying SDF-NeRF on practical systems.
>
---
#### [replaced 067] LAMM-ViT: AI Face Detection via Layer-Aware Modulation of Region-Guided Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.07734v2](https://arxiv.org/pdf/2505.07734v2)**

> **作者:** Jiangling Zhang; Weijie Zhu; Jirui Huang; Yaxiong Chen
>
> **备注:** Accepted to ECAI 2025
>
> **摘要:** Detecting AI-synthetic faces presents a critical challenge: it is hard to capture consistent structural relationships between facial regions across diverse generation techniques. Current methods, which focus on specific artifacts rather than fundamental inconsistencies, often fail when confronted with novel generative models. To address this limitation, we introduce Layer-aware Mask Modulation Vision Transformer (LAMM-ViT), a Vision Transformer designed for robust facial forgery detection. This model integrates distinct Region-Guided Multi-Head Attention (RG-MHA) and Layer-aware Mask Modulation (LAMM) components within each layer. RG-MHA utilizes facial landmarks to create regional attention masks, guiding the model to scrutinize architectural inconsistencies across different facial areas. Crucially, the separate LAMM module dynamically generates layer-specific parameters, including mask weights and gating values, based on network context. These parameters then modulate the behavior of RG-MHA, enabling adaptive adjustment of regional focus across network depths. This architecture facilitates the capture of subtle, hierarchical forgery cues ubiquitous among diverse generation techniques, such as GANs and Diffusion Models. In cross-model generalization tests, LAMM-ViT demonstrates superior performance, achieving 94.09% mean ACC (a +5.45% improvement over SoTA) and 98.62% mean AP (a +3.09% improvement). These results demonstrate LAMM-ViT's exceptional ability to generalize and its potential for reliable deployment against evolving synthetic media threats.
>
---
#### [replaced 068] WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.02439v5](https://arxiv.org/pdf/2601.02439v5)**

> **作者:** Hao Bai; Alexey Taymanov; Tong Zhang; Aviral Kumar; Spencer Whitehead
>
> **备注:** Added link to GitHub repo
>
> **摘要:** We present WebGym, the largest-to-date open-source environment for training realistic visual web agents. Real websites are non-stationary and diverse, making artificial or small-scale task sets insufficient for robust policy learning. WebGym contains nearly 300,000 tasks with rubric-based evaluations across diverse, real-world websites and difficulty levels. We train agents with a simple reinforcement learning (RL) recipe, which trains on the agent's own interaction traces (rollouts), using task rewards as feedback to guide learning. To enable scaling RL, we speed up sampling of trajectories in WebGym by developing a high-throughput asynchronous rollout system, designed specifically for web agents. Our system achieves a 4-5x rollout speedup compared to naive implementations. Second, we scale the task set breadth, depth, and size, which results in continued performance improvement. Fine-tuning a strong base vision-language model, Qwen-3-VL-8B-Instruct, on WebGym results in an improvement in success rate on an out-of-distribution test set from 26.2% to 42.9%, significantly outperforming agents based on proprietary models such as GPT-4o and GPT-5-Thinking that achieve 27.1% and 29.8%, respectively. This improvement is substantial because our test set consists only of tasks on websites never seen during training, unlike many other prior works on training visual web agents.
>
---
#### [replaced 069] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21172v2](https://arxiv.org/pdf/2602.21172v2)**

> **作者:** Ishaan Rawal; Shubh Gupta; Yihan Hu; Wei Zhan
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Vision-Language-Action (VLA) models are advancing autonomous driving by replacing modular pipelines with unified end-to-end architectures. However, current VLAs face two expensive requirements: (1) massive dataset collection, and (2) dense reasoning annotations. In this work, we address both challenges with NORD (No Reasoning for Driving). Compared to existing VLAs, NORD achieves competitive performance while being fine-tuned on <60% of the data and no reasoning annotations, resulting in 3x fewer tokens. We identify that standard Group Relative Policy Optimization (GRPO) fails to yield significant improvements when applied to policies trained on such small, reasoning-free datasets. We show that this limitation stems from difficulty bias, which disproportionately penalizes reward signals from scenarios that produce high-variance rollouts within GRPO. NORD overcomes this by incorporating Dr. GRPO, a recent algorithm designed to mitigate difficulty bias in LLMs. As a result, NORD achieves competitive performance on Waymo and NAVSIM with a fraction of the training data and no reasoning overhead, enabling more efficient autonomous systems. Website: https://nord-vla-ai.github.io/
>
---
#### [replaced 070] Joint Optimization for 4D Human-Scene Reconstruction in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.02158v2](https://arxiv.org/pdf/2501.02158v2)**

> **作者:** Zhizheng Liu; Joe Lin; Wayne Wu; Bolei Zhou
>
> **备注:** Project Page: https://vail-ucla.github.io/JOSH/
>
> **摘要:** Reconstructing human motion and its surrounding environment is crucial for understanding human-scene interaction and predicting human movements in the scene. While much progress has been made in capturing human-scene interaction in constrained environments, those prior methods can hardly reconstruct the natural and diverse human motion and scene context from web videos. In this work, we propose JOSH, a novel optimization-based method for 4D human-scene reconstruction in the wild from monocular videos. JOSH uses techniques in both dense scene reconstruction and human mesh recovery as initialization, and then it leverages the human-scene contact constraints to jointly optimize the scene, the camera poses, and the human motion. Experiment results show JOSH achieves better results on both global human motion estimation and dense scene reconstruction by joint optimization of scene geometry and human motion. We further design a more efficient model, JOSH3R, and directly train it with pseudo-labels from web videos. JOSH3R outperforms other optimization-free methods by only training with labels predicted from JOSH, further demonstrating its accuracy and generalization ability.
>
---
#### [replaced 071] UniFuture: A 4D Driving World Model for Future Generation and Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.13587v2](https://arxiv.org/pdf/2503.13587v2)**

> **作者:** Dingkang Liang; Dingyuan Zhang; Xin Zhou; Sifan Tu; Tianrui Feng; Xiaofan Li; Yumeng Zhang; Mingyang Du; Xiao Tan; Xiang Bai
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** We present UniFuture, a unified 4D Driving World Model designed to simulate the dynamic evolution of the 3D physical world. Unlike existing driving world models that focus solely on 2D pixel-level video generation (lacking geometry) or static perception (lacking temporal dynamics), our approach bridges appearance and geometry to construct a holistic 4D representation. Specifically, we treat future RGB images and depth maps as coupled projections of the same 4D reality and model them jointly within a single framework. To achieve this, we introduce a Dual-Latent Sharing (DLS) scheme, which maps visual and geometric modalities into a shared spatio-temporal latent space, implicitly entangling texture with structure. Furthermore, we propose a Multi-scale Latent Interaction (MLI) mechanism, which enforces bidirectional consistency: geometry constrains visual synthesis to prevent structural hallucinations, while visual semantics refine geometric estimation. During inference, UniFuture can forecast high-fidelity, geometrically consistent 4D scene sequences (image-depth pairs) from a single current frame. Extensive experiments on the nuScenes and Waymo datasets demonstrate that our method outperforms specialized models in both future generation and geometry perception, highlighting the efficacy of unified 4D modeling for autonomous driving. The code is available at https://github.com/dk-liang/UniFuture.
>
---
#### [replaced 072] Abstracted Gaussian Prototypes for True One-Shot Concept Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2408.17251v2](https://arxiv.org/pdf/2408.17251v2)**

> **作者:** Chelsea Zou; Kenneth J. Kurtz
>
> **摘要:** We introduce a cluster-based generative image segmentation framework to encode higher-level representations of visual concepts based on one-shot learning inspired by the Omniglot Challenge. The inferred parameters of each component of a Gaussian Mixture Model (GMM) represent a distinct topological subpart of a visual concept. Sampling new data from these parameters generates augmented subparts to build a more robust prototype for each concept, i.e., the Abstracted Gaussian Prototype (AGP). This framework addresses one-shot classification tasks using a cognitively-inspired similarity metric and addresses one-shot generative tasks through a novel AGP-VAE pipeline employing variational autoencoders (VAEs) to generate new class variants. Results from human judges reveal that the generative pipeline produces novel examples and classes of visual concepts that are broadly indistinguishable from those made by humans. The proposed framework leads to impressive, but not state-of-the-art, classification accuracy; thus, the contribution is two-fold: 1) the system is low in theoretical and computational complexity yet achieves the standard of 'true' one-shot learning by operating in a fully standalone manner unlike existing approaches that draw heavily on pre-training or knowledge engineering; and 2) in contrast with existing neural network approaches, the AGP approach addresses the importance of broad task capability emphasized in the Omniglot challenge (successful performance on classification and generative tasks). These two points are critical in advancing our understanding of how learning and reasoning systems can produce viable, robust, and flexible concepts based on literally no more than a single example.
>
---
#### [replaced 073] Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method with Forgery Style Mixture
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.12791v3](https://arxiv.org/pdf/2408.12791v3)**

> **作者:** Chenqi Kong; Anwei Luo; Peijun Bao; Haoliang Li; Renjie Wan; Zengwei Zheng; Anderson Rocha; Alex C. Kot
>
> **摘要:** Open-set face forgery detection poses significant security threats and presents substantial challenges for existing detection models. These detectors primarily have two limitations: they cannot generalize across unknown forgery domains and inefficiently adapt to new data. To address these issues, we introduce an approach that is both general and parameter-efficient for face forgery detection. It builds on the assumption that different forgery source domains exhibit distinct style statistics. Previous methods typically require fully fine-tuning pre-trained networks, consuming substantial time and computational resources. In turn, we design a forgery-style mixture formulation that augments the diversity of forgery source domains, enhancing the model's generalizability across unseen domains. Drawing on recent advancements in vision transformers (ViT) for face forgery detection, we develop a parameter-efficient ViT-based detection model that includes lightweight forgery feature extraction modules and enables the model to extract global and local forgery clues simultaneously. We only optimize the inserted lightweight modules during training, maintaining the original ViT structure with its pre-trained ImageNet weights. This training strategy effectively preserves the informative pre-trained knowledge while flexibly adapting the model to the task of Deepfake detection. Extensive experimental results demonstrate that the designed model achieves state-of-the-art generalizability with significantly reduced trainable parameters, representing an important step toward open-set Deepfake detection in the wild.
>
---
#### [replaced 074] Denoising the Deep Sky: Physics-Based CCD Noise Formation for Astronomical Imaging
- **分类: astro-ph.IM; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.23276v2](https://arxiv.org/pdf/2601.23276v2)**

> **作者:** Shuhong Liu; Xining Ge; Ziying Gu; Quanfeng Xu; Lin Gu; Ziteng Cui; Xuangeng Chu; Jun Liu; Dong Li; Tatsuya Harada
>
> **摘要:** Astronomical imaging remains noise-limited under practical observing conditions. Standard calibration pipelines remove structured artifacts but largely leave stochastic noise unresolved. Although learning-based denoising has shown strong potential, progress is constrained by scarce paired training data and the requirement for physically interpretable models in scientific workflows. We propose a physics-based noise synthesis framework tailored to CCD noise formation in the telescope. The pipeline models photon shot noise, photo-response non-uniformity, dark-current noise, readout effects, and localized outliers arising from cosmic-ray hits and hot pixels. To obtain low-noise inputs for synthesis, we stack multiple unregistered exposures to produce high-SNR bases. Realistic noisy counterparts synthesized from these bases using our noise model enable the construction of abundant paired datasets for supervised learning. Extensive experiments on our real-world multi-band dataset curated from two ground-based telescopes demonstrate the effectiveness of our framework in both photometric and scientific accuracy.
>
---
#### [replaced 075] Unified Multimodal Models as Auto-Encoders
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09666v4](https://arxiv.org/pdf/2509.09666v4)**

> **作者:** Zhiyuan Yan; Kaiqing Lin; Zongjian Li; Junyan Ye; Hui Han; Haochen Wang; Zhendong Wang; Bin Lin; Hao Li; Xinyan Xiao; Jingdong Wang; Haifeng Wang; Li Yuan
>
> **摘要:** Image-to-text (I2T) understanding and text-to-image (T2I) generation are two fundamental, important yet traditionally isolated multimodal tasks. Despite their intrinsic connection, existing approaches typically optimize them independently, missing the opportunity for mutual enhancement. In this paper, we argue that the both tasks can be connected under a shared Auto-Encoder perspective, where text serves as the intermediate latent representation bridging the two directions - encoding images into textual semantics (I2T) and decoding text back into images (T2I). Our key insight is that if the encoder truly "understands" the image, it should capture all essential structure, and if the decoder truly "understands" the text, it should recover that structure faithfully. Building upon this principle, we propose Unified-GRPO, a post-training method based on reinforcement learning that jointly optimizes both modules through reconstructive rewards, maximizing the semantic consistency between the input and the generated images. Under this reconstruction objective, the encoder is encouraged to extract as much accurate and comprehensive semantic information from the input image to maximize reconstruction quality, while the decoder is simultaneously optimized to generate conditioned on the encoder's prior, enabling a self-evolving improvement. Empirically, we find that using text as the intermediate representation and training under a reconstructive RL paradigm effectively benefits both I2T and T2I. The I2T module gains stronger fine-grained visual perception, such as small-object recognition, grounding, etc, while its dense embeddings and language priors, in turn, provide richer semantic signals that improve T2I fidelity and complex instruction following. These results demonstrate that the reconstructive RL establishes a mutually reinforcing cross-modal synergy within the auto-encoding framework.
>
---
#### [replaced 076] Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.10611v3](https://arxiv.org/pdf/2601.10611v3)**

> **作者:** Christopher Clark; Jieyu Zhang; Zixian Ma; Jae Sung Park; Mohammadreza Salehi; Rohun Tripathi; Sangho Lee; Zhongzheng Ren; Chris Dongjoo Kim; Yinuo Yang; Vincent Shao; Yue Yang; Weikai Huang; Ziqi Gao; Taira Anderson; Jianrui Zhang; Jitesh Jain; George Stoica; Winson Han; Ali Farhadi; Ranjay Krishna
>
> **备注:** Fixed results in Table 7
>
> **摘要:** Today's strongest video-language models (VLMs) remain proprietary. The strongest open-weight models either rely on synthetic data from proprietary VLMs, effectively distilling from them, or do not disclose their training data or recipe. As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models. Crucially, many downstream applications require more than just high-level video understanding; they require grounding -- either by pointing or by tracking in pixels. Even proprietary models lack this capability. We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks. Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance. Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos. On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).
>
---
#### [replaced 077] SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21818v2](https://arxiv.org/pdf/2602.21818v2)**

> **作者:** Guibin Chen; Dixuan Lin; Jiangping Yang; Youqiang Zhang; Zhengcong Fei; Debang Li; Sheng Chen; Chaofeng Ao; Nuo Pang; Yiming Wang; Yikun Dou; Zheng Chen; Mingyuan Fan; Tuanhui Li; Mingshan Chang; Hao Zhang; Xiaopeng Sun; Jingtao Xu; Yuqiang Xie; Jiahua Wang; Zhiheng Xu; Weiming Xiong; Yuzhe Jin; Baoxuan Gu; Binjie Mao; Yunjie Yu; Jujie He; Yuhao Feng; Shiwen Tu; Chaojie Wang; Rui Yan; Wei Shen; Jingchen Wu; Peng Zhao; Xuanyue Zhong; Zhuangzhuang Liu; Kaifei Wang; Fuxiang Zhang; Weikai Xu; Wenyan Liu; Binglu Zhang; Yu Shen; Tianhui Xiong; Bin Peng; Liang Zeng; Xuchen Song; Haoxiang Guo; Peiyu Wang; Max W. Y. Lam; Chien-Hung Liu; Yahui Zhou
>
> **摘要:** SkyReels V4 is a unified multi modal video foundation model for joint video audio generation, inpainting, and editing. The model adopts a dual stream Multimodal Diffusion Transformer (MMDiT) architecture, where one branch synthesizes video and the other generates temporally aligned audio, while sharing a powerful text encoder based on the Multimodal Large Language Models (MMLM). SkyReels V4 accepts rich multi modal instructions, including text, images, video clips, masks, and audio references. By combining the MMLMs multi modal instruction following capability with in context learning in the video branch MMDiT, the model can inject fine grained visual guidance under complex conditioning, while the audio branch MMDiT simultaneously leverages audio references to guide sound generation. On the video side, we adopt a channel concatenation formulation that unifies a wide range of inpainting style tasks, such as image to video, video extension, and video editing under a single interface, and naturally extends to vision referenced inpainting and editing via multi modal prompts. SkyReels V4 supports up to 1080p resolution, 32 FPS, and 15 second duration, enabling high fidelity, multi shot, cinema level video generation with synchronized audio. To make such high resolution, long-duration generation computationally feasible, we introduce an efficiency strategy: Joint generation of low resolution full sequences and high-resolution keyframes, followed by dedicated super-resolution and frame interpolation models. To our knowledge, SkyReels V4 is the first video foundation model that simultaneously supports multi-modal input, joint video audio generation, and a unified treatment of generation, inpainting, and editing, while maintaining strong efficiency and quality at cinematic resolutions and durations.
>
---
#### [replaced 078] Adaptive Hybrid Caching for Efficient Text-to-Video Diffusion Model Acceleration
- **分类: cs.GR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.12691v2](https://arxiv.org/pdf/2508.12691v2)**

> **作者:** Yuanxin Wei; Lansong Diao; Bujiao Chen; Shenggan Cheng; Zhengping Qian; Wenyuan Yu; Nong Xiao; Wei Lin; Jiangsu Du
>
> **备注:** 9 pages, 12 figures
>
> **摘要:** Efficient video generation models are increasingly vital for multimedia synthetic content generation. Leveraging the Transformer architecture and the diffusion process, video DiT models have emerged as a dominant approach for high-quality video generation. However, their multi-step iterative denoising process incurs high computational cost and inference latency. Caching, a widely adopted optimization method in DiT models, leverages the redundancy in the diffusion process to skip computations in different granularities (e.g., step, cfg, block). Nevertheless, existing caching methods are limited to single-granularity strategies, struggling to balance generation quality and inference speed in a flexible manner. In this work, we propose MixCache, a training-free caching-based framework for efficient video DiT inference. It first distinguishes the interference and boundary between different caching strategies, and then introduces a context-aware cache triggering strategy to determine when caching should be enabled, along with an adaptive hybrid cache decision strategy for dynamically selecting the optimal caching granularity. Extensive experiments on diverse models demonstrate that, MixCache can significantly accelerate video generation (e.g., 1.94$\times$ speedup on Wan 14B, 1.97$\times$ speedup on HunyuanVideo) while delivering both superior generation quality and inference efficiency compared to baseline methods.
>
---
#### [replaced 079] Distractor-free Generalizable 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.17605v3](https://arxiv.org/pdf/2411.17605v3)**

> **作者:** Yanqi Bao; Jing Liao; Jing Huo; Yang Gao
>
> **摘要:** We present DGGS, a novel framework that addresses the previously unexplored challenge: $\textbf{Distractor-free Generalizable 3D Gaussian Splatting}$ (3DGS). It mitigates 3D inconsistency and training instability caused by distractor data in the cross-scenes generalizable train setting while enabling feedforward inference for 3DGS and distractor masks from references in the unseen scenes. To achieve these objectives, DGGS proposes a scene-agnostic reference-based mask prediction and refinement module during the training phase, effectively eliminating the impact of distractor on training stability. Moreover, we combat distractor-induced artifacts and holes at inference time through a novel two-stage inference framework for references scoring and re-selection, complemented by a distractor pruning mechanism that further removes residual distractor 3DGS-primitive influences. Extensive feedforward experiments on the real and our synthetic data show DGGS's reconstruction capability when dealing with novel distractor scenes. Moreover, our generalizable mask prediction even achieves an accuracy superior to existing scene-specific training methods. Homepage is https://github.com/bbbbby-99/DGGS.
>
---
#### [replaced 080] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.02700v4](https://arxiv.org/pdf/2512.02700v4)**

> **作者:** Zhenkai Wu; Xiaowen Ma; Zhenliang Ni; Dengming Zhang; Han Shu; Xin Jiang; Xinghao Chen
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Vision-language models (VLMs) excel at image understanding tasks, but the large number of visual tokens imposes significant computational costs, hindering deployment on mobile devices. Many pruning methods rely solely on token importance and thus overlook inter-token redundancy, retaining numerous duplicated tokens and wasting capacity. Although some redundancy-aware approaches have been proposed, they often ignore the spatial relationships among visual tokens. This can lead to overly sparse selections of retained tokens that fail to adequately cover the regions of target objects. To address these limitations, we propose VLM-Pruner, a training-free token pruning algorithm that explicitly balances redundancy and spatial sparsity. We introduce a centrifugal token pruning paradigm that enables near-to-far selection while prioritizing the preservation of fine-grained object details. Moreover, we design a Buffering for Spatial Sparsity (BSS) criterion that defers the selection of spatially distant tokens. We further adopt a parallel greedy strategy to conduct token selection efficiently. To mitigate information loss from pruning, we selectively fuse salient information from the discarded tokens into the retained ones. Comprehensive comparisons demonstrate that VLM-Pruner consistently outperforms strong baselines across five VLMs with an 88.9\% pruning rate, while delivering an end-to-end inference speedup. The code is available at https://github.com/Casey-bit/VLMPruner.
>
---
#### [replaced 081] TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20903v3](https://arxiv.org/pdf/2602.20903v3)**

> **作者:** Hanshen Zhu; Yuliang Liu; Xuecheng Wu; An-Lan Wang; Hao Feng; Dingkang Yang; Chao Feng; Can Huang; Jingqun Tang; Xiang Bai
>
> **备注:** Accepted by CVPR 2026; Code: https://github.com/CIawevy/TextPecker
>
> **摘要:** Visual Text Rendering (VTR) remains a critical challenge in text-to-image generation, where even advanced models frequently produce text with structural anomalies such as distortion, blurriness, and misalignment. However, we find that leading MLLMs and specialist OCR models largely fail to perceive these structural anomalies, creating a critical bottleneck for both VTR evaluation and RL-based optimization. As a result, even state-of-the-art generators (e.g., Seedream4.0, Qwen-Image) still struggle to render structurally faithful text. To address this, we propose TextPecker, a plug-and-play structural anomaly perceptive RL strategy that mitigates noisy reward signals and works with any textto-image generator. To enable this capability, we construct a recognition dataset with character-level structural-anomaly annotations and develop a stroke-editing synthesis engine to expand structural-error coverage. Experiments show that TextPecker consistently improves diverse text-to-image models; even on the well-optimized Qwen-Image, it significantly yields average gains of 4% in structural fidelity and 8.7% in semantic alignment for Chinese text rendering, establishing a new state-of-the-art in high-fidelity VTR. Our work fills a gap in VTR optimization, providing a foundational step towards reliable and structural faithful visual text generation.
>
---
#### [replaced 082] PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像描述评估任务，旨在解决现有评价指标不适用于长文本的问题。提出PoSh方法，利用场景图指导LLM评分，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2510.19060v3](https://arxiv.org/pdf/2510.19060v3)**

> **作者:** Amith Ananthram; Elias Stengel-Eskin; Lorena A. Bradford; Julia Demarest; Adam Purvis; Keith Krut; Robert Stein; Rina Elster Pantalony; Mohit Bansal; Kathleen McKeown
>
> **备注:** Accepted at ICLR 2026. 26 pages, 9 figures. Metric/benchmark available at https://github.com/amith-ananthram/posh
>
> **摘要:** While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $ρ$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.
>
---
#### [replaced 083] Index Light, Reason Deep: Deferred Visual Ingestion for Visual-Dense Document Question Answering
- **分类: cs.CL; cs.CV; cs.IR**

- **简介: 该论文属于文档问答任务，针对视觉密集工程文档的问答问题，提出DVI框架，通过延迟视觉处理提升准确率。**

- **链接: [https://arxiv.org/pdf/2602.14162v2](https://arxiv.org/pdf/2602.14162v2)**

> **作者:** Tao Xu
>
> **备注:** 24 pages, 4 figures, 7 tables
>
> **摘要:** Existing multimodal document question answering methods predominantly adopt a Pre-Ingestion (PI) strategy: during the indexing phase, a Vision Language Model (VLM) is called on every page to generate page descriptions that are then encoded into vectors, and questions are answered via embedding similarity retrieval. However, this approach faces a dual dilemma on visual-dense engineering documents: VLM blind descriptions inevitably lose critical visual details, and embedding retrieval systematically fails on highly similar documents. This paper proposes the Deferred Visual Ingestion (DVI) framework: zero VLM calls during preprocessing, leveraging only document structural information (table of contents, drawing numbers) to automatically build a hierarchical index through the HDNC (Hierarchical Drawing Number Clustering) algorithm; during inference, candidate pages are located via BM25 retrieval, and the original images along with the specific question are sent to a VLM for targeted analysis. Large-scale experiments on three datasets validate the effectiveness of DVI: on Bridge engineering drawings (1,323 questions), end-to-end QA accuracy reaches 65.6\% vs. PI's 24.3\% (+41.3pp); on Steel catalog (186 questions), 30.6\% vs. 16.1\% (+14.5pp); on CircuitVQA, a public benchmark (9,315 questions), retrieval ImgR@3 achieves 31.2\% vs. 0.7\%. On the Bridge dataset, we evaluated ColPali (ICLR 2025 visual retrieval SOTA), which achieved only 20.1\% PageR@3, demonstrating that the failure of embedding retrieval on homogeneous engineering documents is structural rather than due to insufficient model capability. Ablation studies show that HDNC zero-cost automatic indexing yields a +27.5pp retrieval improvement, and VLM conversion rate analysis confirms that the bottleneck lies on the retrieval side rather than the comprehension side.
>
---
#### [replaced 084] Towards Seamless Interaction: Causal Turn-Level Modeling of Interactive 3D Conversational Head Dynamics
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15340v2](https://arxiv.org/pdf/2512.15340v2)**

> **作者:** Junjie Chen; Fei Wang; Zhihao Huang; Qing Zhou; Kun Li; Dan Guo; Linfeng Zhang; Xun Yang
>
> **摘要:** Human conversation involves continuous exchanges of speech and nonverbal cues such as head nods, gaze shifts, and facial expressions that convey attention and emotion. Modeling these bidirectional dynamics in 3D is essential for building expressive avatars and interactive robots. However, existing frameworks often treat talking and listening as independent processes or rely on non-causal full-sequence modeling, hindering temporal coherence across turns. We present TIMAR (Turn-level Interleaved Masked AutoRegression), a causal framework for 3D conversational head generation that models dialogue as interleaved audio-visual contexts. It fuses multimodal information within each turn and applies turn-level causal attention to accumulate conversational history, while a lightweight diffusion head predicts continuous 3D head dynamics that captures both coordination and expressive variability. Experiments on the DualTalk benchmark show that TIMAR reduces Fréchet Distance and MSE by 15-30% on the test set, and achieves similar gains on out-of-distribution data. The source code has been released at https://github.com/CoderChen01/towards-seamless-interaction.
>
---
#### [replaced 085] Is Exchangeability better than I.I.D to handle Data Distribution Shifts while Pooling Data for Data-scarce Medical image segmentation?
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.19575v3](https://arxiv.org/pdf/2507.19575v3)**

> **作者:** Ayush Roy; Samin Enam; Jun Xia; Won Hwa Kim; Vishnu Suresh Lokhande
>
> **备注:** MIDL 2026
>
> **摘要:** Data scarcity is a major challenge in medical imaging, particularly for deep learning models. While data pooling (combining datasets from multiple sources) and data addition (adding more data from a new dataset) have been shown to enhance model performance, they are not without complications. Specifically, increasing the size of the training dataset through pooling or addition can induce distributional shifts, negatively affecting downstream model performance, a phenomenon known as the "Data Addition Dilemma". While the traditional i.i.d. assumption may not hold in multi-source contexts, assuming exchangeability across datasets provides a more practical framework for data pooling. In this work, we investigate medical image segmentation under these conditions, drawing insights from causal frameworks to propose a method for controlling foreground-background feature discrepancies across all layers of deep networks. This approach improves feature representations, which are crucial in data-addition scenarios. Our method achieves state-of-the-art segmentation performance on histopathology and ultrasound images across five datasets, including a novel ultrasound dataset that we have curated and contributed. Qualitative results demonstrate more refined and accurate segmentation maps compared to prominent baselines across three model architectures.
>
---
#### [replaced 086] MERGETUNE: Continued Fine-Tuning of Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.10497v3](https://arxiv.org/pdf/2601.10497v3)**

> **作者:** Wenqing Wang; Da Li; Xiatian Zhu; Josef Kittler
>
> **备注:** 20 pages, 5 figures
>
> **摘要:** Fine-tuning vision-language models (VLMs) such as CLIP often leads to catastrophic forgetting of pretrained knowledge. Prior work primarily aims to mitigate forgetting during adaptation; however, forgetting often remains inevitable during this process. We introduce a novel paradigm, continued fine-tuning (CFT), which seeks to recover pretrained knowledge after a zero-shot model has already been adapted. We propose a simple, model-agnostic CFT strategy (named MERGETUNE) guided by linear mode connectivity (LMC), which can be applied post hoc to existing fine-tuned models without requiring architectural changes. Given a fine-tuned model, we continue fine-tuning its trainable parameters (e.g., soft prompts or linear heads) to search for a continued model which has two low-loss paths to the zero-shot (e.g., CLIP) and the fine-tuned (e.g., CoOp) solutions. By exploiting the geometry of the loss landscape, the continued model implicitly merges the two solutions, restoring pretrained knowledge lost in the fine-tuned counterpart. A challenge is that the vanilla LMC constraint requires data replay from the pretraining task. We approximate this constraint for the zero-shot model via a second-order surrogate, eliminating the need for large-scale data replay. Experiments show that MERGETUNE improves the harmonic mean of CoOp by +5.6% on base-novel generalisation without adding parameters. On robust fine-tuning evaluations, the LMC-merged model from MERGETUNE surpasses ensemble baselines with lower inference cost, achieving further gains and state-of-the-art results when ensembled with the zero-shot model. Our code is available at https://github.com/Surrey-UP-Lab/MERGETUNE.
>
---
#### [replaced 087] FUSAR-GPT : A Spatiotemporal Feature-Embedded and Two-Stage Decoupled Visual Language Model for SAR Imagery
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.19190v2](https://arxiv.org/pdf/2602.19190v2)**

> **作者:** Xiaokun Zhang; Yi Yang; Ziqi Ye; Baiyun; Xiaorong Guo; Qingchen Fang; Ruyi Zhang; Xinpeng Zhou; Haipeng Wang
>
> **摘要:** Research on the intelligent interpretation of all-weather, all-time Synthetic Aperture Radar (SAR) is crucial for advancing remote sensing applications. In recent years, although Visual Language Models (VLMs) have demonstrated strong open-world understanding capabilities on RGB images, their performance is severely limited when directly applied to the SAR field due to the complexity of the imaging mechanism, sensitivity to scattering features, and the scarcity of high-quality text corpora. To systematically address this issue, we constructed the inaugural SAR Image-Text-AlphaEarth feature triplet dataset and developed FUSAR-GPT, a VLM specifically for SAR. FUSAR-GPT innovatively introduces a geospatial baseline model as a 'world knowledge' prior and embeds multi-source remote-sensing temporal features into the model's visual backbone via 'spatiotemporal anchors', enabling dynamic compensation for the sparse representation of targets in SAR images. Furthermore, we designed a two-stage SFT strategy to decouple the knowledge injection and task execution of large models. The spatiotemporal feature embedding and the two-stage decoupling paradigm enable FUSAR-GPT to achieve state-of-the-art performance across several typical remote sensing visual-language benchmark tests, significantly outperforming mainstream baseline models by over 12%.
>
---
#### [replaced 088] Q$^2$: Quantization-Aware Gradient Balancing and Attention Alignment for Low-Bit Quantization
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.05898v2](https://arxiv.org/pdf/2511.05898v2)**

> **作者:** Zhaoyang Wang; Dong Wang
>
> **备注:** 24 pages,6 figures
>
> **摘要:** Quantization-aware training (QAT) has achieved remarkable success in low-bit ($\leq$4-bit) quantization for classification networks. However, when applied to more complex visual tasks such as object detection and image segmentation, performance still suffers significant degradation. A key cause of this limitation has been largely overlooked in the literature. In this work, we revisit this phenomenon from a new perspective and identify a major failure factor: gradient imbalance at feature fusion stages, induced by accumulated quantization errors. This imbalance biases the optimization trajectory and impedes convergence under low-bit quantization. Based on this diagnosis, we propose Q$^2$, a two-pronged framework comprising: (1) Quantization-aware Gradient Balancing Fusion (Q-GBFusion), a closed-loop mechanism that dynamically rebalances gradient contributions during feature fusion; and (2) Quantization-aware Attention Distribution Alignment (Q-ADA), a parameter-free supervision strategy that reconstructs the supervision distribution using semantic relevance and quantization sensitivity, yielding more stable and reliable supervision to stabilize training and accelerate convergence. Extensive experiments show that our method, as a plug-and-play and general strategy, can be integrated into various state-of-the-art QAT pipelines, achieving an average +2.5\% mAP gain on object detection and a +3.7\% mDICE improvement on image segmentation. Notably, it is applied only during training and introduces no inference-time overhead, making it highly practical for real-world deployment.
>
---
#### [replaced 089] Proxy-GS: Unified Occlusion Priors for Training and Inference in Structured 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24421v3](https://arxiv.org/pdf/2509.24421v3)**

> **作者:** Yuanyuan Gao; Yuning Gong; Yifei Liu; Li Jingfeng; Dingwen Zhang; Yanci Zhang; Dan Xu; Xiao Sun; Zhihang Zhong
>
> **备注:** Project page: https://gyy456.github.io/Proxy-GS
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as an efficient approach for achieving photorealistic rendering. Recent MLP-based variants further improve visual fidelity but introduce substantial decoding overhead during rendering. To alleviate computation cost, several pruning strategies and level-of-detail (LOD) techniques have been introduced, aiming to effectively reduce the number of Gaussian primitives in large-scale scenes. However, our analysis reveals that significant redundancy still remains due to the lack of occlusion awareness. In this work, we propose Proxy-GS, a novel pipeline that exploits a proxy to introduce Gaussian occlusion awareness from any view. At the core of our approach is a fast proxy system capable of producing precise occlusion depth maps at a resolution of 1000x1000 under 1ms. This proxy serves two roles: first, it guides the culling of anchors and Gaussians to accelerate rendering speed. Second, it guides the densification towards surfaces during training, avoiding inconsistencies in occluded regions, and improving the rendering quality. In heavily occluded scenarios, such as the MatrixCity Streets dataset, Proxy-GS not only equips MLP-based Gaussian splatting with stronger rendering capability but also achieves faster rendering speed. Specifically, it achieves more than 2.5x speedup over Octree-GS, and consistently delivers substantially higher rendering quality. Code will be public upon acceptance.
>
---
