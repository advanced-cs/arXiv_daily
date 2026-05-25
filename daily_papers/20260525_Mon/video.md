# 计算机视觉 cs.CV

- **最新发布 109 篇**

- **更新 70 篇**

## 最新发布

#### [new 001] PathNavigate: A Training-Free Pathology Agent with Surprise-Guided Scan and Shared Slide Memory for Whole-Slide Image VQA
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于WSI-VQA任务，旨在解决病理图像中高效定位关键证据的问题。提出PathNavigate，通过扫描-搜索-读取流程，在不训练模型的情况下提升答案准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2605.23559](https://arxiv.org/pdf/2605.23559)**

> **作者:** Chunze Yang; Qidong Liu; Wenjie Zhao; Yue Tang; Jiusong Ge; Di Zhang; Jiashuai Liu; Lei Wu; Junbo Lu; Ni Zhang; Xian Wu; Zeyu Gao; Chen Li
>
> **摘要:** Whole-slide image visual question answering (WSI-VQA) frames pathology as an extreme-context search problem: to answer a free-form clinical query, a system must first navigate a gigapixel slide under a strict inspection budget to locate sparse, high-resolution evidence. Existing approaches largely fall into two paradigms: i) supervised pathology multimodal large language models (MLLMs) and agents can absorb localization and reasoning into learned modules, but they often couple navigation to task-specific supervision and retraining, limiting their practicality; ii) training-free pathology agents avoid this cost by keeping core models frozen, but often follow a question-first design, constructing the initial candidate set mainly from query-conditioned relevance. This can miss decisive morphology that is not named in the question, and force heavier inference-time scaffolding. To address this challenge, we introduce PathNavigate, a training-free pathology agent built around a scan-search-readout routine. Before question matching, PathNavigate scans the current slide at low magnification with a shared online memory module over frozen pathology features, producing a slide-specific surprise field that marks an abnormal-region pool. It then applies question-conditioned PLIP relevance only within this pool to select high-magnification search targets. Finally, it extracts local high-magnification evidence and answers with a frozen perceptor-adjudicator stack, using the same online memory as slide-level context. Experiments on WSI-VQA and SlideBench-BCNB show that the proposed scan-search-readout design improves answer accuracy and yields more interpretable evidence-selection trajectories with higher this http URL code is available online.
>
---
#### [new 002] Decomposing Queries into Tool Calls for Long-Video Keyframe Retrieval
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于长视频关键帧检索任务，旨在解决如何根据查询准确找到相关帧的问题。提出ToolMerge方法，通过分解查询为工具调用并合并结果，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.23826](https://arxiv.org/pdf/2605.23826)**

> **作者:** Michal Shlapentokh-Rothman; Prachi Garg; Yu-Xiong Wang; Derek Hoiem
>
> **摘要:** Keyframe selection is a direct way to provide verifiable visual evidence for long-video question answering (QA). Queries differ in what they require, and finding the right frames depends on knowing what to look for. Existing keyframe selectors either score every frame against a single query, or decompose the query into a fixed schema evaluated by a single visual tool. We propose ToolMerge, a keyframe retrieval method based on decomposition and merging: an Large Language Model (LLM) based planner decomposes the query into tool calls and specifies how their per-tool rankings are merged using boolean operators. To evaluate retrieval directly, we construct Molmo-2 Moments (M2M), a benchmark in which every question is anchored to a specific time interval by construction. Across QA, question retrieval, and caption retrieval, ToolMerge is competitive with prior keyframe selectors, most notably on caption retrieval, outperforming other methods by 5%. Code and data can be found at this https URL .
>
---
#### [new 003] Scene Reconstruction as Mapping Priors for 3D Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在利用地图先验提升检测效果。针对传统高精度地图成本高、难以大规模应用的问题，提出自动构建密集地图先验的方法，并设计融合多传感器的检测框架，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.22997](https://arxiv.org/pdf/2605.22997)**

> **作者:** Yang Fu; Yuliang Zou; Hao Xiang; Xin Huang; Yijing Bai; Chen Song; Weijing Shi; Govind Thattai; Dragomir Anguelov; Mingxing Tan; Yingwei Li
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** In autonomous driving, mapping is critical for motion planning but remains an under-utilized resource for perception tasks such as 3D object detection. Maps can provide robust structural priors of the static environment, helping resolve ambiguities and correct for sensor data sparsity or noise, especially for distant objects or under adverse weather conditions. However, conventional High-Definition (HD) maps are resource-intensive to obtain and maintain, which presents a challenge for efficient, large-scale deployment. In this paper, we propose a scalable solution to systematically leverage mapping to improve 3D detection by overcoming two primary challenges. First, we introduce a pipeline to automatically build dense mapping priors from aggregated sensor data, eliminating the need for human labeling. Second, we design a novel Mapping Priors Augmented 3D Detection (MPA3D) framework to effectively integrate mapping priors with different sensor modalities. Extensive experiments on the Waymo Open Dataset demonstrate that our approach achieves new state-of-the-art results, proving the effectiveness of scalable reconstructed scene priors for enhancing 3D detection.
>
---
#### [new 004] PhotoFlow: Agentic 3D Virtual Photography Missions
- **分类: cs.CV; cs.AI; cs.MA**

- **简介: 该论文提出PhotoFlow，解决虚拟摄影任务中的3D空间理解和美学判断问题，通过代理系统实现高效相机参数选择与渲染。**

- **链接: [https://arxiv.org/pdf/2605.23771](https://arxiv.org/pdf/2605.23771)**

> **作者:** Jiarui Guo; Haojia Wei; Yiming Zhang; Yifei Liu; Yuning Gong; Hongjie Zhang; Xue Yang; Zhihang Zhong
>
> **摘要:** Virtual photography asks an agent to enter a prepared 3D scene with no preselected camera pose or reference image, infer a suitable shot from scene information and a language intent, choose executable camera parameters, and render the final photograph. Recent progress in vision-language models makes this kind of spatial agent increasingly plausible, but the task stresses two capabilities that remain hard to evaluate together: complex 3D spatial understanding and abstract aesthetic judgment. We introduce PhotoFlow, a Director-Reviewer-Reflector agent for closed-loop camera search. The Director builds a soft photographic blueprint and proposes diverse candidate cameras; the Reviewer combines rule checks, visual critique, and pairwise incumbent selection; and the Reflector converts failures into region memory, dead-zone suppression, and high-explore relocation. We also introduce VPhotoBench, a benchmark of 47 open-license Blender scenes and 141 language-conditioned photography missions spanning subject placement, relational composition, and atmosphere/style. On held-out experiments, PhotoFlow achieves the strongest external quality-alignment composite and success rate among one-shot prediction, single-chain reflection, anchor-bank selection, and random search under a six-round rendering budget. To our knowledge, this is the first work to make language-conditioned virtual photography in arbitrary Blender scenes an executable agent task, and our results show that an LLM-centered spatial agent can already produce strong photographs in a setting designed to challenge both 3D reasoning and aesthetic choice.
>
---
#### [new 005] StereoGenBench: A Synthetic Multi-Camera Benchmark for Stereo Generation under Controlled Baseline Regimes
- **分类: cs.CV**

- **简介: 该论文提出StereoGenBench，一个用于立体生成的合成多摄像机基准数据集，解决可控基线条件下立体视觉任务的数据不足问题。**

- **链接: [https://arxiv.org/pdf/2605.23237](https://arxiv.org/pdf/2605.23237)**

> **作者:** Yangzhi Cui; Feng Qiao; Nathan Jacobs
>
> **摘要:** Stereo image and video generation, stereo geometry estimation, and condition-controlled view synthesis require paired data in which the variables that determine binocular geometry -- camera baseline, intrinsics, scene depth, and camera motion -- are known and controllable. Existing stereo resources provide subsets of these variables, but resources commonly used for stereo generation evaluation do not, to our knowledge, provide scene-paired, calibrated multi-baseline right-view ground truth with jointly recorded intrinsics, dense metric depth, and per-frame poses in a single controlled source. We introduce StereoGenBench, a synthetic Unreal Engine benchmark designed to make baseline-regime sensitivity and target-camera consistency measurable under matched scene content. Each scene is rendered with a rigid six-camera lateral array, yielding up to 15 calibrated view pairs; adjacent baselines are sampled from inter-pupillary to wide-baseline regimes; focal length is sampled independently; and every view is released with RGB, metric depth, intrinsics, per-pair baselines, and per-frame poses. The splits include two evaluation families for narrow and wide baseline regimes and a train-only family for broader all-pairs coverage. We release the dataset, evaluation code, reference results, Croissant metadata, and generation code/configuration for extension with compatible assets. The dataset is available at this https URL
>
---
#### [new 006] Seeing without Looking: Do Vision-Language Benchmarks Really Test Vision?
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，探讨基准测试是否真实评估视觉理解。研究发现现有基准无法有效衡量细粒度视觉定位，提出多维度分析验证模型对视觉证据的依赖程度。**

- **链接: [https://arxiv.org/pdf/2605.22903](https://arxiv.org/pdf/2605.22903)**

> **作者:** Zixuan Lan; Luzhe Sun; Matthew R. Walter; Jiawei Zhou
>
> **备注:** Accepted to GRAIL-V: Grounded Retrieval and Agentic Intelligence for Vision-Language, CVPR 2026 Workshop. accepted version
>
> **摘要:** Benchmark accuracy is often implicitly assumed to reflect grounded visual understanding in vision-language models (VLMs), yet it remains unclear to what extent such scores truly reflect reliance on visual evidence. Motivated by a surprising observation that removing a substantial fraction of image tokens only degrades model performance very slightly on a widely used hallucination benchmark, we systematically investigate this mismatch in a set of open-source VLMs. Our analysis spans multiple levels of granularity, spanning global visual degradation, localized occlusion, question reformulation, answer-space expansion, and decision-level analyses beyond standard accuracy. We further complement these behavioral results with a layer-wise analysis of vision-token geometry. Throughout the experiments, we find that although VLMs do incorporate visual input, their predictions are less sensitive to the loss of fine-grained visual evidence that standard accuracy should have suggested. Even when the final prediction remains unchanged, the model's internal support for the correct answer may already be weakened. We further complement a representation-level analysis, which shows increasing similarity among visual tokens in deeper layers, providing a possible explanation for our findings. Together, these results suggest that current benchmarks are not sufficient to reliably evaluate fine-grained visual grounding in VLMs.
>
---
#### [new 007] RoboSurg-VQA: A Multimodal Benchmark for Surgical Segmentation-Aware Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文提出RoboSurg-VQA，属于视觉问答任务，旨在解决手术中复杂视觉理解问题，通过构建多模态基准提升手术场景下的视觉问答性能。**

- **链接: [https://arxiv.org/pdf/2605.23068](https://arxiv.org/pdf/2605.23068)**

> **作者:** Chengyi Zhang; Zi Ye; Ziyang Wang
>
> **摘要:** Reliable visual understanding in robot-assisted and minimally invasive surgery (RMIS/MIS) demands more than accurate masks: in clinical practice, clinicians pose language-like questions about procedural context, visibility, artefacts, and the presence of anatomical structures and surgical instruments, often under degraded views caused by occlusion, smoke, bleeding, and specular highlights. We present \textbf{RoboSurg-VQA}, a segmentation-aware visual question answering (VQA) benchmark built by repurposing public surgical segmentation datasets under a shared schema. Each frame is paired with a fixed set of clinically motivated questions spanning procedure context, anatomy (including region), imaging modality/view, surgical artefacts, image quality, and basic visibility and spatial attributes, with closed answer sets to enable consistent evaluation. To scale annotation, we generate candidate answers via constrained prompting with automatic validity and consistency checks, followed by human auditing to improve plausibility and label consistency. We report benchmark statistics, sanity baselines, and common evaluation challenges under challenging surgical conditions. The code will be available on this https URL.
>
---
#### [new 008] FAST-ME: Foundation-aware Adaptive Stopping for Motion Estimation for Efficient IoT Video Analysis
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频分析任务，解决运动估计计算成本高的问题。通过结合语义信息与最优停止理论，提出高效运动估计方法，降低计算量并提升准确性。**

- **链接: [https://arxiv.org/pdf/2605.23428](https://arxiv.org/pdf/2605.23428)**

> **作者:** Kakia Panagidi; Stathes Hadjieftymiadis
>
> **摘要:** In modern multimedia systems, efficient video processing is critical, especially in resource-constrained environments such as IoT-based camera networks, autonomous platforms, and wireless sensor multimedia systems. A key bottleneck in video compression and understanding is block motion estimation (ME), a process that remains computationally expensive despite the development of fast search techniques. This work introduces an Optimal Stopping Theory (OST) algorithm for block motion estimation based on the assessment of spatiotemporal differences within and across video frames. It also proposes a semantic-aware motion estimation framework that integrates Foundation Models (FMs) with the OST-based decision process. By leveraging pretrained visual models such as Vision Transformers (ViT) and the Segment Anything Model (SAM), the framework extracts semantic attention scores that indicate the importance of motion within specific spatial regions. These scores are fused with traditional distortion-based metrics, such as the Sum of Absolute Differences (SAD), to guide a hybrid stopping criterion that jointly considers motion magnitude and semantic relevance. The resulting adaptive algorithm stops early in redundant regions while continuing the search in areas where motion is semantically significant. Experiments compare the proposed solution with widely used approaches from the literature on benchmark and multimodal video datasets. The proposed method achieves a significant reduction in computation with minimal accuracy loss and improved semantic coverage. The results highlight the benefits of bridging low-level motion analysis with high-level semantic reasoning, offering a promising direction for efficient multimodal video understanding in next-generation smart systems.
>
---
#### [new 009] Generator-Refiner-Examiner: A Tri-Module Data Augmentation Framework for 3D Human Avatar Learning from Monocular Videos
- **分类: cs.CV**

- **简介: 该论文属于3D人体动画生成任务，旨在解决单目视频重建高质量人体模型的问题。提出三模块框架TrioMan，增强数据多样性与质量，提升重建效果。**

- **链接: [https://arxiv.org/pdf/2605.23555](https://arxiv.org/pdf/2605.23555)**

> **作者:** Gangjian Zhang; Jian Shu; Sicheng Yu; Wenhao Shen; Yu Feng; Hao Wang
>
> **摘要:** This paper addresses the challenge of reconstructing photorealistic and animatable 3D human avatars from monocular videos. While existing methods rely on combining per-subject optimization with generic human priors, they often fail to capture fine-grained details when training frames are limited. To mitigate this data scarcity, we propose TrioMan, a systematic tri-module framework for augmented 3D avatar learning. Our approach comprises three synergistic components. The Generator creates diverse unseen samples by imposing Gaussian perturbations on pose and camera. The Refiner improves the quality of generated data through one-step diffusion guided by texture and geometry cues. The Examiner selects subject-consistent samples using a dual-branch attention-based similarity evaluation. Experiments on the X-Humans and NeuMan benchmarks show that TrioMan outperforms state-of-the-art methods.
>
---
#### [new 010] RS2AD-LiDAR: End-to-End Autonomous Driving LiDAR Data Generation from Roadside Sensor Observations
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶数据生成任务，旨在解决单车数据采集成本高、场景稀缺的问题。通过路边传感器生成车载LiDAR数据，提升模型训练效果。**

- **链接: [https://arxiv.org/pdf/2605.23406](https://arxiv.org/pdf/2605.23406)**

> **作者:** Runyi Huang; Ni Ding; Ruidan Xing; Yuheng Shi; Lei He; Keqiang Li
>
> **摘要:** End-to-end autonomous driving solutions, which directly process multimodal sensory data and output fine-grained control commands, have gradually become a mainstream direction with the development of autonomous driving technology. However, current methods in this category rely on single-vehicle data collection for model training and optimization, which suffers from high acquisition and annotation costs, scarcity of valuable scenarios, and data silos. To address these challenges, we propose RS2AD-LiDAR, a novel framework for reconstructing and generating vehicle-mounted LiDAR data from roadside sensor observations. Since no public dataset currently provides highly overlapping perception coverage between roadside and vehicle-mounted LiDAR sensors, which is essential for studying roadside-to-vehicle data generation, we constructed a dedicated dataset named R2V-LiDAR which is used solely for evaluation in this work. Specifically, our method transforms roadside LiDAR point clouds into the vehicle-mounted LiDAR coordinate system, and synthesizes high-fidelity vehicle-mounted data via virtual LiDAR modeling and point cloud resampling techniques. To the best of our knowledge, this is the first approach to reconstruct vehicle-mounted LiDAR data from roadside sensor inputs. Extensive experimental comparisons demonstrate the semantic similarity between the generated data and real data. Furthermore, object detection experiments show that incorporating the generated data into real data for model training improves both Bird's Eye View (BEV) and 3D detection accuracy, thereby validating the effectiveness of the proposed method.
>
---
#### [new 011] LangFlash: Feed-forward 3D Language Gaussian Splatting from Sparse Unposed Images
- **分类: cs.CV**

- **简介: 论文提出LangFlash，用于从稀疏无姿态图像中进行3D语言高斯点云重建。解决无姿态下3D场景重建与语义一致性问题，通过单次前向传播实现高效重建。**

- **链接: [https://arxiv.org/pdf/2605.23287](https://arxiv.org/pdf/2605.23287)**

> **作者:** Yilong Liu; Wanhua Li; Chen Zhu-Tian; Hanspeter Pfister
>
> **备注:** CVPRF 2026
>
> **摘要:** We present LangFlash, a feed-forward framework for 3D Language Gaussian Splatting that reconstructs 3D scenes parameterized by Gaussian primitives enriched with language-aligned semantic features from sparse unposed multi-view images. Unlike optimization-based 3D methods, LangFlash directly predicts the geometry and semantics in a single forward pass, enabling low-latency 3D reconstruction and language-consistent scene understanding. To support large-scale training, we enriched the RealEstate10k dataset with coherent and dense semantic information for 3D semantic supervision. Furthermore, we propose a sparse semantic encoding scheme that combines a global semantic dictionary with locally varying per-primitive weights, preserving high-level linguistic information, while reducing representation complexity. Experimental results show that LangFlash achieves superior novel view synthesis and semantic consistency compared with previous methods. This study establishes a new paradigm for pose-free, language-grounded 3D scene reconstruction, advancing generalizable 3D vision and multimodal scene understanding. Demo is available at this https URL.
>
---
#### [new 012] ETCHR: Editing To Clarify and Harness Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ETCHR，解决多模态大模型中视觉推理的瓶颈问题，通过解耦图像编辑与理解模型，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2605.23897](https://arxiv.org/pdf/2605.23897)**

> **作者:** Beichen Zhang; Yuhong Liu; Jinsong Li; Yuhang Zang; Jiaqi Wang; Dahua Lin
>
> **备注:** Code, model and data are open-sourced at this https URL
>
> **摘要:** Multimodal Large Language Models have advanced visual reasoning, yet a purely textual chain of thought remains a bottleneck for questions that require fine-grained focus or view transformations. The ''think with images'' paradigm narrows this gap, but existing approaches are either constrained by fixed predefined toolkits or produce noisy intermediate images from unified multimodal methods. We pursue a third option: using a dedicated image editing model and decouple it with an understanding model. However, off-the-shelf image editors fail as reasoning assistants with two complementary gaps: a language-side gap, where editors trained as passive instruction-followers cannot map an abstract question to an appropriate visual transformation, and a generation-side gap, where edit correctness degrades as reasoning depth grows. Guided by this analysis, we introduce ETCHR (Editing To Clarify and Harness Reasoning), a question-conditioned, reasoning-aware image editor decoupled from the downstream understanding model and trained with a two-stage recipe targeted at the two gaps: Reasoning Imitation via supervised fine-tuning on edit trajectories, followed by Reasoning Enhancement with VLM-derived rewards for edit correctness and downstream reasoning accuracy. Since the editor is decoupled, ETCHR plugs into different open- and closed-source MLLMs in a training-free manner. Across five task families (fine-grained perception, chart understanding, logic reasoning, jigsaw restoration, and 3D understanding), ETCHR raises average Pass@1 from 55.95 to 60.77 (+4.82) with Qwen3-VL-8B, from 65.08 to 70.55 (+5.47) with Gemini-3.1-Flash-Lite, and from 76.55 to 81.16 (+4.61) with the 1T-parameter MoE model Kimi K2.5.
>
---
#### [new 013] HorizonStream: Long-Horizon Attention for Streaming 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于在线3D重建任务，解决长序列中相机位姿和场景几何估计的问题。提出HorizonStream模型，通过长视野Transformer实现稳定、高效的实时重建。**

- **链接: [https://arxiv.org/pdf/2605.23889](https://arxiv.org/pdf/2605.23889)**

> **作者:** Chong Cheng; Peilin Tao; Nanjie Yao; Guanzhi Ding; Xianda Chen; Yuansen Du; Xiaoyang Guo; Wei Yin; Weiqiang Ren; Qian Zhang; Zhengqing Chen; Hao Wang
>
> **摘要:** Online 3D reconstruction requires estimating camera pose and scene geometry under strict causal and bounded-memory constraints. Existing methods often suffer from drift, jitter, or collapse on long sequences. We trace these failures to a fundamental mismatch. Streaming geometry is inherently temporally heterogeneous, with evidence ranging from short-lived correspondences to persistent global scale. However, current architectures impose uniform and pathological influence patterns. For example, sliding windows enforce hard cutoffs, while ungated recurrence and causal attention cause cache saturation and spike-like attention sinks. To resolve this, we formalize geometric propagation as an \emph{evidence influence kernel} and propose HorizonStream, a long-horizon Transformer that explicitly factorizes this kernel. For the long-range temporal factor, Geometric Linear Attention learns channel-wise decay rates to enable bounded, multi-timescale propagation of geometric evidence. For the short-range spatial factor, Geometric Local Attention with Spatiotemporal RoPE performs reliable 3D matching while suppressing attention sinks. Finally, Metric Readout Tokens recover stable scale and rigid pose directly from the persistent geometric state. Extensive experiments show that HorizonStream, trained on only 48-frame clips, generalizes stably to sequences exceeding 10,000\ frames with constant memory and linear time, achieving state-of-the-art streaming 3D reconstruction performance. Project Page: this https URL
>
---
#### [new 014] Recursive Block-Diagonal Coupling for Resource-Efficient Training of Vision Models
- **分类: cs.CV**

- **简介: 该论文属于视觉模型训练任务，解决资源效率问题。提出RBDC方法，通过递归耦合窄模型构建宽模型，提升训练效率，减少计算量。**

- **链接: [https://arxiv.org/pdf/2605.23656](https://arxiv.org/pdf/2605.23656)**

> **作者:** Maxim Henry; Adrien Deliège; Sébastien Piérard; Marc Van Droogenbroeck
>
> **备注:** 22 pages, 3 figures, 4 tables, and 34 references
>
> **摘要:** Training high-capacity vision models from scratch requires substantial computational resources. To improve training efficiency of a wide target model, existing growth methods often assume the availability of narrower models, obscuring the true computational cost of the entire pipeline. We propose an efficient training protocol, RBDC, that builds wide models by coupling in a parameter-free block-diagonal way narrower, independently trained models in a recursive way. This allows a flexible allocation of the training budget available across all the models involved. Evaluated with vision transformers (DeiT) and convolutional networks (ResNet) on ImageNet, our RBDC training protocol shows a much better efficiency than models trained from scratch with the standard protocol, yielding 30% FLOPs reduction at similar test accuracies. It also achieves higher performances at same training FLOPs than training protocols from the model growth literature. Finally, we show that our models can serve as better backbones than their original counterparts for downstream object detection and instance segmentation tasks.
>
---
#### [new 015] IntentionNav: A Benchmark for Intent-Driven Object Navigation from Implicit Human Instruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IntentionNav基准，解决从隐式指令中进行意图驱动的物体导航问题，通过分析语言理解、目标推理和定位效果，评估模型在复杂场景中的表现。**

- **链接: [https://arxiv.org/pdf/2605.23187](https://arxiv.org/pdf/2605.23187)**

> **作者:** Lin Qian; Shijie Li; Sihao Lin; Xuan Zhang; Bangya Liu; Yanran Li; Hujun Yin
>
> **备注:** preprint
>
> **摘要:** Existing object navigation benchmarks usually tell an embodied agent which object category to find, such as microwave or chair. Human-facing embodied AI is often asked something less direct: "I need something to warm this food" or "the room feels stuffy." The agent must infer the object that can satisfy the need, find a scene-grounded instance, and decide whether the goal has been reached. We study this setting as intent-driven object navigation and introduce IntentionNav, a diagnostic benchmark for active object search from implicit human instructions. Each episode provides a free-text intent, RGB-D observations, and pose, but withholds the target object name. IntentionNav contains 500 intents over 176 Isaac Sim scenes and 64 target categories. Each intent is rewritten in four controlled instruction styles and annotated with one of four intent modes, separating surface phrasing from semantic cue type under matched geometry. This paired design supports analysis of target inference, language robustness, neighborhood reachability, and terminal success rather than only aggregate success. We evaluated three VLMs using a fixed active-navigation agent. Models identify the intended target in 48.3 percent of episodes and enter its 2 m neighborhood in 68.7 percent, but terminate successfully in only 24.9 percent and achieve grounded 1 m success in 5.5 percent. Success is highest for event-script intents (28.7 percent) and lower for physical-state and affordance intents (19.2 percent and 18.5 percent), showing that indirect human intent remains a bottleneck for target selection, visual verification, and terminal localization in active embodied search.
>
---
#### [new 016] Calibration-Informative Region Selection for Online LiDAR--Camera Calibration in Agricultural Environments
- **分类: cs.CV**

- **简介: 该论文属于多模态标定任务，旨在解决LiDAR与相机在线标定中的噪声与模糊问题。通过提出支持图驱动方法，提升标定精度。**

- **链接: [https://arxiv.org/pdf/2605.23580](https://arxiv.org/pdf/2605.23580)**

> **作者:** Rajitha de Silva; Grzegorz Cielniak
>
> **备注:** Accepted to ICRA 2026 Workshop on Agricultural Robotics
>
> **摘要:** Reliable multi-modal calibration requires identifying which observations truly constrain the extrinsic parameters and which ones mainly add noise or ambiguity. In this paper, we propose a support-map-driven approach to multi-modal calibration that decouples four functional blocks: initial calibration, cross-modal residual extraction, support-map estimation, and support-aware refinement. We instantiate this formulation for online LiDAR--camera calibration using MDPCalib, a target-less LiDAR--camera calibration method based on motion and deep point correspondences, and CMRNext, a dense LiDAR--camera matching model that predicts optical-flow-like image-plane residuals. The key contribution is a dense calibration support map that aggregates cross-modal agreement over aligned observations and highlights where calibration evidence is consistently reliable. Across the Bacchus Long-Term (BLT) dataset and KITTI, we show that calibration evidence is spatially and semantically non-uniform, indicating that some semantic regions provide stronger cues for calibration than others. On KITTI, support-guided refinement improves the calibration performance with better translation accuracy while rotational gains remain limited.
>
---
#### [new 017] CRONOS: Benchmarking Counterfactual Physical Consistency in Video Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决模型是否理解物理因果关系的问题。通过构建CRONOS基准，系统评估模型在不同干预下的物理一致性表现。**

- **链接: [https://arxiv.org/pdf/2605.23699](https://arxiv.org/pdf/2605.23699)**

> **作者:** León Begiristain; Olaf Dünkel; Adam Kortylewski
>
> **备注:** 27 pages, 12 figures
>
> **摘要:** Video prediction is increasingly viewed as a path toward generalizable world models, yet it remains unclear whether these systems learn underlying causal structure or merely exploit superficial visual correlations for future prediction. We introduce CRONOS, an intervention-based benchmark designed to evaluate counterfactual physical consistency: whether a model's predictions of physical events respond appropriately to controlled changes in the visual input, such as variations of scene context, viewpoint, object appearance, and object category. Built in a photorealistic Unreal Engine environment, CRONOS enables controlled, high-fidelity generation of videos across diverse scenes and dynamics. In contrast to previous benchmarks, CRONOS systematically intervenes on four key factors - viewpoint, scene, object category, and object appearance - while keeping the underlying physical event type, such as a collision, occlusion, or fall, fixed. Our evaluation of recent open-source video generators reveals substantial failures in counterfactual physical consistency: prediction quality for the same physical event type is affected by appearance, environment, and, particularly by viewpoint changes. CRONOS provides a controlled and reproducible testbed for diagnosing how the quality of generated videos changes for different interventions, establishing a concrete target for developing models that perform consistently across changes of multiple conditions. The dataset and code are available at our project page.
>
---
#### [new 018] VideoOdyssey: A Benchmark for Ultra-Long-Context and Omni-Modal Video Understanding
- **分类: cs.CV**

- **简介: 该论文提出VideoOdyssey基准，解决超长视频理解和多模态任务中的连续推理问题，通过设计长时视频数据集和多级评估体系，评估模型在不同上下文长度下的表现。**

- **链接: [https://arxiv.org/pdf/2605.22907](https://arxiv.org/pdf/2605.22907)**

> **作者:** Haichen He; Jiayi Zhou; Sifeng Shang; Yihan Hu; Yuanhan Zhang; Kaiyang Zhou
>
> **摘要:** Real-world long video understanding requires models to perform continuous tracking, information integration and memory retention over massive temporal spans within extreme video durations. Mastering this intense cognitive load constitutes the fundamental bottleneck in long video understanding. While existing benchmarks have driven progress by scaling up video duration, their evaluation tasks often require comprehending only short and isolated video segments, falling short of capturing the challenge of ultra-long-context reasoning. To measure this cognitive load, we emphasize continuous certificate length, defined as the video length a human must continuously watch to definitively answer a given question. Driven by this metric, we introduce VideoOdyssey, a benchmark specifically designed for ultra-long-context and omni-modal video understanding. VideoOdyssey is characterized by three key features: 1) Extreme video duration and diversity: spanning 11 domains and 54 subcategories with an average video duration of 109 minutes; 2) Comprehensive evaluation scenarios: offering two subsets to address different research focuses, i.e., VideoOdyssey-V for probing the limits of visual understanding in MLLMs, and VideoOdyssey-AV for evaluating synchronized audio-visual understanding for omni-modal models; 3) Ultra-long and multi-level continuous certificates: extending the average continuous certificate to 16 minutes for VideoOdyssey-V and 12.8 minutes for VideoOdyssey-AV. Crucially, we design 5 granular levels from seconds to hours, providing a comprehensive diagnostic tool to evaluate models across varying context lengths and cognitive loads. Extensive evaluations show that bottlenecks of current MLLMs extend beyond simple retrieval to include struggles with continuous reasoning across varying context lengths, fine-grained perception, and non-verbal omni-modal understanding.
>
---
#### [new 019] Online Hand Gesture Recognition Using 3D Convolutional Neural Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于手势识别任务，旨在解决实时视频流中动态手部动作的检测与分类问题。通过3D卷积神经网络和滑动窗口方法提升系统鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.23409](https://arxiv.org/pdf/2605.23409)**

> **作者:** Yinghao Qin; Tijana Timotijevic
>
> **备注:** Master's dissertation work written in Autumn 2020
>
> **摘要:** In human computer interaction, real-time detection and classification of dynamic hand gestures is challenging as: 1) the system must run in a real-time video stream and there is no noticeable lag in response after performing a gesture; 2) there is a large difference in how people perform gestures, making recognition more difficult. In this paper, an online hand gesture recognition system is proposed, which is able to localize gestures in real-time video stream and recognize what these gestures are. To improve the robustness of the system, the sliding window approach is used to refine results from multiple windows. All of the models in my project are trained on Jester database, achieving 98+% accuracy for detector and 90+% accuracy for classifier. For the overall performance of the system, the best group can respond within three seconds and reach 37.5% Levenshtein accuracy on the homemade dataset. The project codes used in this work are publicly available.
>
---
#### [new 020] A Novel Approach for the Counting of Wood Logs Using cGANs and Image Processing Techniques
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决木材计数问题。通过cGAN和图像处理技术实现精准计数，提升林业管理效率。**

- **链接: [https://arxiv.org/pdf/2605.23775](https://arxiv.org/pdf/2605.23775)**

> **作者:** João VC Mazzochin; Giovani Bernardes Vitor; Gustavo Tiecker; Elioenai MF Diniz; Gilson A Oliveira; Marcelo Trentin; Érick O Rodrigues
>
> **摘要:** This study tackles the challenge of precise wood log counting, where applications of the proposed methodology can span from automated approaches for materials management, surveillance, and safety science to wood traffic monitoring, wood volume estimation, and others. We introduce an approach leveraging Conditional Generative Adversarial Networks (cGANs) for eucalyptus log segmentation in images, incorporating specialized image processing techniques to handle noise and intersections, coupled with the Connected Components Algorithm for efficient counting. To support this research, we created and made publicly available a comprehensive database of 466 images containing approximately 13,048 eucalyptus logs, which served for both training and validation purposes. Our method demonstrated robust performance, achieving an average Accuracy_pixel of 96.4% and Accuracy_logs of 92.3%, with additional measures such as F1 scores ranging from 0.879 to 0.933 and IoU values between 0.784 and 0.875, further validating its effectiveness. The implementation proves to be efficient with an average processing time of 0.713s per image on an NVIDIA T4 GPU, making it suitable for realtime applications. The practical implications of this method are significant for operational forestry, enabling more accurate inventory management, reducing human errors in manual counting, and optimizing resource allocation. Furthermore, the segmentation capabilities of the model provide a foundation for advanced applications such as eucalyptus stack volume estimation, contributing to a more comprehensive and refined analysis of forestry operations. The methodology's success in handling complex scenarios, including intersecting logs and varying environmental conditions, positions it as a valuable tool for practical applications across related industrial sectors.
>
---
#### [new 021] RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video
- **分类: cs.CV**

- **简介: 该论文属于动态3D场景重建任务，旨在解决单目视频中长期平滑运动与短期复杂变形的建模问题。提出RiGS方法，通过三种高斯基元分别表示静态、刚性与瞬态运动，提升动态场景的重建精度。**

- **链接: [https://arxiv.org/pdf/2605.23672](https://arxiv.org/pdf/2605.23672)**

> **作者:** Chenyu Wu; Wanhua Li; Zhu-Tian Chen; Hanspeter Pfister
>
> **摘要:** Reconstructing dynamic 3D scenes from monocular videos is a fundamental yet highly challenging task, as real-world motions often involve both long-term smooth transformations and short-term complex deformations. Existing methods either struggle to maintain temporal consistency or fail to capture high-frequency dynamics due to limited motion modeling capacity. In this work, we present Rigid-aware 4D Gaussian Splatting (RiGS), which simultaneously captures motions across multiple temporal scales. Specifically, RiGS introduces three types of Gaussian primitives: static, rigid, and transient, which represent static backgrounds, long-term low-frequency motions, and short-term high-frequency dynamics, respectively. An object-wise dynamic mask is proposed to aggregate long-range spatiotemporal motion information and guide the decomposition of static and dynamic regions. To jointly model motion across scales, rigid Gaussians are allowed to transition into transient Gaussians based on their temporal duration, and both are optimized under scene flow guidance, providing dense 3D motion supervision. Extensive experiments demonstrate that RiGS achieves state-of-the-art performance on novel view synthesis benchmarks. Code is available at \hyperlink{this https URL}{this https URL}.
>
---
#### [new 022] DualMem: Bypassing the Objectness Bottleneck for Calibrated Unknown-Stream Filtering in Open-World Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于开放世界目标检测任务，旨在解决未知物体过滤中的信息瓶颈问题。通过提出DualMem方法，提升未知预测的准确性。**

- **链接: [https://arxiv.org/pdf/2605.23634](https://arxiv.org/pdf/2605.23634)**

> **作者:** Yingjun Xiao; Xi Chen; Gang Fang; Siyuan Chen
>
> **摘要:** Open-world object detection (OWOD) requires detectors to localize known classes while identifying unknown objects for future incremental learning. We find that the unknown prediction streams of strong OWOD detectors are heavily polluted: on M-OWODB, across PROB, OW-DETR, and HypOW, future-task positive unknowns make up less than 10% of unknown predictions, whereas background false positives account for 46-71%. We show that this is not a missing-information problem, but an information bottleneck at the objectness head. On PROB Task 1, a linear probe on the 256-D decoder query achieves an AUROC of 0.908 for positive-versus-negative unknown discrimination, but the final one-dimensional objectness scalar drops to 0.642. A frozen SigLIP feature, without access to the detector, independently recovers much of this proposal-level separability at the filtering stage (AUROC = 0.871). Motivated by this finding, we propose DualMem, a calibrated post-hoc filter that assumes a small image-disjoint annotated calibration split of held-out future-task objects and performs a non-parametric likelihood ratio test in frozen SigLIP feature space. DualMem uses a k-nearest-neighbor positive memory to protect future-task objects and a negative memory to suppress background-like proposals. Its decision threshold is chosen by Neyman-Pearson calibration, giving users an explicit trade-off between false-unknown suppression and novel recall. Across PROB, OW-DETR, and HypOW on M-OWODB Task 1, DualMem reduces background-type false unknown proposals per image by 44.9%-66.3%, with a mean reduction of 56.6%. On PROB Task 1, it more than doubles the reduction achieved by a natural K-means prototype baseline, while leaving known-class mAP unchanged because known detections bypass the filter.
>
---
#### [new 023] Smart-Insertion-V: Photorealistic Video Insertion via a Closed-Loop Feedback Dual-Stream Framework
- **分类: cs.CV**

- **简介: 该论文属于视频对象插入任务，解决参考对象与源视频风格差异大导致的融合困难问题。提出Smart-Insertion-V框架，结合双流结构与闭环反馈，实现高质量视频插入。**

- **链接: [https://arxiv.org/pdf/2605.23891](https://arxiv.org/pdf/2605.23891)**

> **作者:** Xiao Cao; Yansong Qu; Xiangzhen; Chang; Wen Xiao; Jiakui Hu; Heyuan Li; Jialun Liu; Zhiyong Huang; Xuelong Li
>
> **摘要:** Mask-free video object insertion has emerged as a challenging task, requiring harmonious integration of reference objects into source videos. However, existing methods struggle when references exhibit severe stylistic domain gaps with the source scene. To overcome this, we propose \textit{\textbf{Smart-Insertion-V}}, an end-to-end \textbf{Dual-Stream} framework that concurrently conducts video insertion and image style transfer. Within this framework, the image stream synchronously guides the video generation process, while a \textbf{Closed-loop Feedback} mechanism is further incorporated to ensure robust insertion. Inevitably, integrating these diverse conditioning signals results in feature entanglement and style leakage. To tackle this issue, we design \textbf{Dual-World-View RoPE} to distinguish different signals via spatial-temporal offsets without incurring heavy training overhead. Furthermore, to facilitate spatial grounding and stylistic adaptation, we introduce a \textbf{Decoupled Guidance Module} that leverages a Vision-Language Model for semantic reasoning while preserving original temporal guidance with native text encoder. To bridge data gap for harmonious reference insertion task, we propose a data curation pipeline and will release an \textbf{open-source dataset}. Experiments demonstrate that our method can insert objects into plausible positions while achieving the most harmonious results.
>
---
#### [new 024] Learning a Particle Dynamics Model with Real-world Videos
- **分类: cs.CV**

- **简介: 该论文属于物理模拟任务，旨在解决真实世界视频中物体动力学建模问题。通过无标签视频训练粒子动力学模型，无需精确状态信息。**

- **链接: [https://arxiv.org/pdf/2605.23845](https://arxiv.org/pdf/2605.23845)**

> **作者:** Chanho Kim; Suhas V. Sumukh; Li Fuxin
>
> **备注:** CVPR 2026 Findings
>
> **摘要:** Data-driven learning approaches for physics simulation, sometimes referred to as world models, have emerged as promising alternatives to traditional physics simulators due to their differentiable nature. Prior work has demonstrated impressive results in predicting the motions of rigid and non-rigid objects in complex scenes involving multiple interacting bodies. However, these models are typically trained in simulated environments because obtaining perfect state information such as complete scene point clouds and point correspondences over time is challenging in real-world settings. This reliance on synthetic data can limit their applicability when the sim-to-real gap is large. In this work, we aim to overcome these limitations by introducing a novel framework for training neural object dynamics models directly from unlabeled real-world videos. Specifically, we propose to learn a particle-based dynamics model compatible with a Gaussian splatting framework, which operates on dense particles derived from Gaussians (i.e., particles with scales and rotations) and predicts their position and rotation changes over time. The model is trained via rendering supervision, enabling learning from real-world videos without requiring particle-level labeled states. Our model operates directly on dense Gaussians without relying on heuristic subsampling anchor points. To enable this study, we also present a real-world dataset consisting of about 500 videos capturing diverse object interactions.
>
---
#### [new 025] Millimeter-wave Imaging for Anthropometric Body Measurement
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人体测量任务，旨在解决传统测量方法需脱衣、保持姿势的问题。通过毫米波雷达和优化框架，实现无接触、隐私保护的3D人体建模与尺寸提取。**

- **链接: [https://arxiv.org/pdf/2605.23064](https://arxiv.org/pdf/2605.23064)**

> **作者:** Miriam Senne; Benjamin D. Killeen; Christoph Baur; Nassir Navab; Azade Farshad
>
> **摘要:** Body shape and circumferences are clinically informative biomarkers for risk stratification, including measures such as waist to hip ratio, limb and trunk girths, yet conventional tools such as manual tape measures and optical scanners often require undressing and sustained poses. These demands slow workflows, compromise dignity, and exclude many older adults and people with limited mobility. To make measurement fast and contactless, we leverage millimeter-wave (mmWave) radar, which preserves privacy and operates through typical clothing, enabling quick full-body acquisition. In this work, we present a new optimization-based framework to recover 3D human shape and extract a comprehensive set of anthropometric measurements from volumetric mmWave data. Our method introduces a weighted registration pipeline that fits a parametric body model (SMPL) directly to the noisy mmWave point cloud. The core of our contribution is a vertex-weighting strategy that modulates a Chamfer energy function for reliable surface alignment and noise elimination. We further stabilize the fit by incorporating a foot-ground plane constraint and pose priors, optimizing directly for the SMPL parameters. Together, these components enable a fast, privacy preserving workflow that delivers high fidelity body shape and measurements through clothing without cameras or disrobing and with minimal cooperation, supporting frequent risk oriented assessments in clinics and care facilities for patients of all ages and mobility levels.
>
---
#### [new 026] Composing People Together: Iterative Pose-Image Generation for Multi-Person Interaction Scenes
- **分类: cs.CV**

- **简介: 该论文属于多人物交互场景生成任务，旨在解决文本到图像生成中布局重复、姿势刻板等问题。通过引入双姿态图像表示和迭代构建方案，提升场景多样性和结构准确性。**

- **链接: [https://arxiv.org/pdf/2605.23178](https://arxiv.org/pdf/2605.23178)**

> **作者:** Wenxuan Peng; Bharath Hariharan; Hadar Averbuch-Elor
>
> **备注:** Accepted to SIGGRAPH Conference Papers 2026. 22 pages, 12 figures. Project page: this https URL
>
> **摘要:** Despite recent progress, text-to-image models still struggle to generate semantically diverse and compositionally accurate multi-person interaction scenes, often collapsing to repetitive layouts, stereotypical poses, and poorly grounded interactions. In this work, we bridge this gap by introducing a dual pose-image representation that brings person-centric structural priors into pretrained diffusion transformers. Our model jointly predicts a 2D pose visualization image and its corresponding RGB image, enabling structure and appearance to co-evolve during learning. At its core, a cross-modal alignment scheme binds text, pose, and image representations, ensuring consistent grounding across modalities. Furthermore, we design an iterative scene construction scheme, progressively generating complex multi-human interactions while effectively decomposing the overall generation complexity. Extensive experiments demonstrate that our method substantially improves prompt alignment and scene diversity in multi-person image generation.
>
---
#### [new 027] One-Forcing: Towards Stable One-Step Autoregressive Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决一阶自回归视频生成中的延迟高和质量差问题。提出One-Forcing方法，结合GAN损失提升生成质量和效率。**

- **链接: [https://arxiv.org/pdf/2605.23458](https://arxiv.org/pdf/2605.23458)**

> **作者:** Jiaqi Feng; Justin Cui; Yuanhao Ban; Cho-Jui Hsieh
>
> **备注:** Work in Progress. Project Page: this https URL, Code: this https URL
>
> **摘要:** Recent advances have substantially improved real-time interactive video generation in the autoregressive regime. However, most existing few-step autoregressive video generation methods, often distilled from a corresponding many-step teacher, default to a 4-step sampling configuration, which still incurs considerable latency during deployment and suffers from severe quality degradation when the number of sampling steps is further reduced, particularly in the one-step setting. Trajectory-style consistency distillation methods often produce videos with weak dynamics, while DMD-based approaches, such as Self-Forcing, tend to yield blurry frames. To address this challenge, we propose One-Forcing, a simple yet effective approach which augments the DMD objective with an auxiliary GAN loss for high-quality and efficient one-step video generation. Experiments on VBench show that One-Forcing achieves a total score of 83.76, establishing state-of-the-art performance among one-step causal video generation methods and remaining competitive with strong many-step approaches. We further demonstrate that one-step framewise autoregressive generation can be achieved stably with merely one-third of the training cost of the chunkwise model, a setting that prior methods have failed to achieve successfully.
>
---
#### [new 028] DepthAgent: Towards Better Universal Depth Estimation via Sample-wise Expert Selection
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决不同相机设置下深度估计效果不稳定的问题。通过样本级专家选择与融合，提升模型在复杂场景中的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.23281](https://arxiv.org/pdf/2605.23281)**

> **作者:** Jie Zhu; Girish Chandar Ganesan; Xiaoming Liu
>
> **摘要:** Monocular metric depth estimation has achieved strong progress with large-scale training and universal-camera modeling, yet robust deployment across diverse camera settings, such as perspective, fisheye, and panoramic images, remains challenging. Existing methods typically rely on a single depth estimator, overlooking that different models encode different camera assumptions and perform best under different input domains. In this paper, we show that depth experts exhibit strong sample-wise complementarity: model preference is highly correlated with camera geometry, and multi-model fusion brings the largest gains on difficult samples where individual experts are unreliable. Motivated by these observations, we propose \textbf{\ours}, a vision-language agent for adaptive monocular depth estimation. DepthAgent treats existing depth models as frozen tools and learns to analyze scene and camera cues, invoke suitable experts through multi-turn tool utilization, and select or fuse their predictions for each input. To optimize such discrete decision-making toward dense geometric quality, we design a multi-reward reinforcement fine-tuning scheme that jointly encourages valid tool execution, camera/scene analysis, expert-selection quality, and inference efficiency. Extensive experiments across perspective, fisheye, and panoramic benchmarks show that \ours consistently outperforms individual experts, fixed model fusion, and different selection strategies, with strong improvements on challenging samples, highlighting the critical role of expert selection and fusion. The code and model will be released upon publication.
>
---
#### [new 029] MuellerPT: Decomposition Driven Pretraining for Dense Learning in Mueller Polarimetry
- **分类: cs.CV**

- **简介: 该论文属于生物医学图像分析任务，旨在解决 Mueller 矩阵数据标注稀缺和域移位问题。通过预训练学习密集表示，提升分割和分类性能。**

- **链接: [https://arxiv.org/pdf/2605.23840](https://arxiv.org/pdf/2605.23840)**

> **作者:** Adam Tlemsani; Yingdian Li; Maxime Giot; Naim Slim; Christopher J. Peters; Abhijeet Ghosh; Daniel S. Elson
>
> **备注:** Accepted to 29th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2026)
>
> **摘要:** Mueller matrix imaging provides rich, physically meaningful contrast for biomedical tissue analysis, but supervised learning is hindered by scarce dense annotations and strong domain shifts across specimens and acquisition settings. We introduce MuellerPT, a physics guided pre-training approach that learns transferable dense representations by predicting Lu-Chipman decomposition maps from per-pixel 4x4 Mueller matrices. To scale pre-training, we collected a new large Multispectral Animal Polarimetric Organ dataset (MAP-Org). The pre-trained encoder is adapted with a segmentation head for grey vs. white matter segmentation in lamb brain. A classification head is used for colorectal cancer vs. non-cancer classification. Both segmentation and classification are evaluated across few-shot learning scenarios. In segmentation, MuellerPT improves label efficiency and cross specimen transfer compared to models without pre-training, achieving an absolute DICE gain of over 20% compared to the baseline trained from scratch when using 5% of the training data. In classification, MuellerPT also enhances label efficiency, improving overall accuracy by 8% compared to the baseline when using 1% of the training data. We demonstrate MuellerPT's robustness to domain shift with a qualitative evaluation of its predicted Lu-Chipman maps on an ex vivo human oesophagus sample. These results suggest that predicting Lu-Chipman decomposition is an effective and practical pretext task for robust biomedical inference from Mueller polarimetry and can pave the way for future work on label efficient Mueller imaging.
>
---
#### [new 030] GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GEM-4D，解决视频生成中点级运动不一致问题，提升机器人操作的物理可靠性。属于视频世界模型任务，通过几何监督和逆动力学模块实现更准确的视频预测与轨迹生成。**

- **链接: [https://arxiv.org/pdf/2605.22882](https://arxiv.org/pdf/2605.22882)**

> **作者:** Kaichen Zhou; Yuzhen Chen; Fangneng Zhan; Hang Hua; Grace Chen; Xinhai Chang; Ao Qu; Yilun Du; Zhuang Liu; Paul Pu Liang; Mengyu Wang
>
> **备注:** Robotic World Model, Video Generative Model
>
> **摘要:** Video world models can generate realistic futures from a single instruction, but they often fail to preserve consistent point-level motion over time. As a result, the generated videos appear plausible, yet lack the physical grounding required for reliable action execution, such as robot manipulation. We present GEM-4D, a geometry-grounded video world model that resolves this limitation by injecting dense 4D correspondence supervision, distilled from a pretrained geometry foundation model, into the video generative backbone during training. This supervision enables the model to jointly capture appearance and geometric structure while retaining a single-stream architecture with no additional inference cost. We further introduce an inverse dynamics module that converts correspondence-consistent video rollouts into executable robot trajectories, enabling direct deployment in both real-world and simulated manipulation. GEM-4D achieves state-of-the-art performance on both video prediction and geometric consistency across simulation and realistic scenarios and improves real-world manipulation success from 61% to 81%. Additional results are available at the project page: this https URL.
>
---
#### [new 031] General Hazard Detection
- **分类: cs.CV**

- **简介: 该论文属于安全危害检测任务，解决传统系统在抽象安全概念上的泛化不足问题。通过语言规则与视觉结合的方法，构建了CompliVision数据集，并提出一种基于视觉推理和人机反馈的通用检测框架。**

- **链接: [https://arxiv.org/pdf/2605.23304](https://arxiv.org/pdf/2605.23304)**

> **作者:** Stephanie Ng; CP Lim; SueJen Looi; Hendrik Zurlinden; David Nguyen; Lei Wei; Saeid Nahavandi; Hailing Zhou
>
> **备注:** 20 pages, 7 figures and 4 tables
>
> **摘要:** Hazard, as an abstract concept, is typically defined through cognitive-level logical reasoning rather than concrete examples. In contrast, existing hazard detection systems rely on predefined hazard categories and require intensive collection of labelled examples within detection or classification architectures. This approach faces three fundamental challenges when addressing abstract safety concepts: (1) noisy and sparse training data, (2) dynamically evolving definitions that change across contexts and time, and (3) limited generalisation to unseen or novel scenarios. To address these limitations, we present the CompliVision dataset, the first general-purpose hazard dataset designed for rule-based compliance assessment, along with a baseline framework for hazard evaluation. Our key innovation is decoupling the hazard concept from image-based examples by expressing safety requirements through language-based rules. We ground our approach in authoritative domain regulations and ISO standards to define diverse hazard concepts across multiple domains. The CompliVision dataset comprises 3,006 images spanning traffic, construction, and warehouse environments, with each image annotated for compliance against specific safety rules, accompanied by natural language explanations highlighting the supporting visual evidence. To achieve robust generalisation, we develop an active learning framework to more effectively guide and refine vision-language models in assessing hazard compliance. While state-of-the-art VLMs demonstrate strong capabilities, they struggle with the fine-grained, context-dependent interpretation required for accurate safety assessment. We proposed a general hazard detection framework to address this limitation which combines LLaVA-based visual reasoning with with human-in-the-loop feedback.
>
---
#### [new 032] Lipschitz Optimization for Formal Verification of Homographies
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于形式化验证任务，解决视觉神经网络在相机运动下的鲁棒性问题。通过Lipschitz优化，建立相机位姿与像素值的映射，实现对投影几何变换的正式验证。**

- **链接: [https://arxiv.org/pdf/2605.23203](https://arxiv.org/pdf/2605.23203)**

> **作者:** Jean-Guillaume Durand; Panagiotis Kouvaros; Maxime Gariel; Alessio Lomuscio
>
> **备注:** 18 pages, 13 figures, 6 tables, to be published at CVPR 2026
>
> **摘要:** The adoption of vision neural networks in regulated industries requires formal robustness guarantees, especially in safety-critical domains such as healthcare, autonomous vehicles, and aerospace. However, current approaches are confined to incomplete statistical verification or robustness to $\ell_p$-norm and affine transforms, which cover only a narrow subset of perturbations to the image formation process. In particular, robustness to camera motion remains an open problem despite being key to deploy many vision applications. We present a formal verification approach that targets robustness against 3D motion perturbations of the capturing camera. We first establish a closed-form mapping from camera pose to pixel values. By analyzing the continuity properties of the resulting homographies, we show that recent work on Lipschitz optimization and piecewise continuity can be extended to derive tight linear bounds on perturbed pixel values. Our approach applies to scenes with predominantly planar structure, such as ground planes in augmented reality, road markings and traffic signs in autonomous driving, or planar workspaces in robotic manipulation. This enables the first formal verification of projective geometry transforms, without complex simulation, surrogate networks, or explicit image-formation models. We validate our implementation and show up to 89% speedup and 7% tighter bounds over prior work. We then evaluate our method on the VNN-COMP benchmark and reveal systematic weaknesses to projective perturbations. Finally, we demonstrate a real-world case study on a safety-critical runway classifier, highlighting practical vulnerabilities to camera motion, and addressing a key challenge in the certification of learned models. Data and code are publicly available at this https URL .
>
---
#### [new 033] Decoupling Spatio-Temporal Adapter for Fine-Grained Badminton Action Localization
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于细粒度时间动作定位任务，旨在解决专业羽毛球视频中细微动作的精准定位问题。提出DSTA模型，有效捕捉时空特征，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.23355](https://arxiv.org/pdf/2605.23355)**

> **作者:** Tianyu Wang; Junjie Wu; Jingquan Gao; Shishuo Li
>
> **备注:** 11 pages, 11figures
>
> **摘要:** Temporal Action Localization (TAL) has been extensively studied in generic video understanding, while fine-grained sports scenarios, such as professional badminton, remain underexplored due to their complex and subtle spatio-temporal dynamics. In this paper, we focus on fine-grained TAL in professional badminton videos and introduce a new benchmark dataset, Fine-Badminton, which consists of 31 matches with 29 fine-grained stroke categories, covering 2104 rallies and 27597 annotated actions. To effectively capture the intricate motion patterns in such scenarios, we propose a Decoupling Spatio-Temporal Adapter (DSTA), which enables efficient modeling of spatio-temporal features within a parameter-efficient framework. Specifically, DSTA decomposes motion representation into three parallel branches, capturing temporal dynamics as well as vertical and horizontal spatial variations. The design allows the model to better distinguish subtle differences among fine-grained actions. Extensive experiments on both the Fine-Badminton dataset and the ShuttleSet benchmark demonstrate that the proposed method achieves state-of-the-art performance while introducing only a marginal increase in computational and parameter cost. These results validate the effectiveness and efficiency of the proposed approach for fine-grained temporal action localization.
>
---
#### [new 034] ComPose: When to Trust Hands for Object Pose Tracking
- **分类: cs.CV**

- **简介: 该论文提出ComPose，解决手部遮挡下的物体位姿跟踪问题。通过融合手部运动作为互补线索，实现鲁棒的6DoF物体跟踪。**

- **链接: [https://arxiv.org/pdf/2605.23523](https://arxiv.org/pdf/2605.23523)**

> **作者:** Jisu Shin; Junoh Lee; JunGyu Lee; Inhwan Bae; Dohyeon Lee; Hokyun Im; Youngwoon Lee; Hae-Gon Jeon
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** Reconstructing the motion of objects from videos is a key component for embodied AI and robot manipulation. While diverse approaches to object pose tracking have been studied, they rely heavily on strong external priors, such as depth data or 3D templates, and remain highly vulnerable to severe occlusions by hand grasps despite the use of explicit masks. In this work, we present ComPose, a 6DoF object tracking framework designed for hand-aware object pose estimation from RGB video. Rather than treating the hand purely as an occluder, our method harmonizes hand motions as a \textit{complementary cue} for object tracking. In detail, we recover a variety of object motions over time by combining object and hand cues from foundation models within a unified tracking pipeline. Here, ComPose adaptively selects informative hand joints, combines object- and hand-derived cues for motion estimation, and refines the resulting object motion using visible geometric evidence and a learned correction. We further enforce the temporal consistency over both rotation and translation, yielding stable 3D object trajectories over time without any external smoothing. Extensive experiments show that our method is accurate, efficient, and robust under severe hand occlusion and geometric ambiguity. In addition, the resulting trajectories can also effectively transfer to downstream robot manipulation by enabling robots to reconstruct human actions from online videos.
>
---
#### [new 035] B-GRTO: Bootstrapped Group Relative Tool Optimization for Referring Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文聚焦于引用分割任务，解决视觉语言系统与分割解码器协同优化问题。提出B-GRTO框架，通过联合优化策略与辅助目标，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2605.23500](https://arxiv.org/pdf/2605.23500)**

> **作者:** Mario Markov; Stefan Maria Ailuro; Mohammad Mahdi; Luc Van Gool; Danda Pani Paudel
>
> **摘要:** Segmentation is a fundamental task in computer vision, underpinning pixel-level scene understanding and serving as a cornerstone for applications ranging from autonomous perception to medical image analysis. For complex referring segmentation, recent methods pair large vision-language models with segmentation decoders: the former analyzes the image and prompt, while the latter predicts the target mask. Although reinforcement learning improves reasoning-intensive vision-language systems, trainable tools such as segmentation decoders are typically optimized separately with differentiable objectives, and the principled integration of such objectives into reinforcement learning remains underexplored. Thus, we introduce group relative tool optimization (GRTO), a mathematically grounded framework for jointly optimizing a policy with differentiable tool use. GRTO reuses group relative policy optimization (GRPO) rollouts to optimize the auxiliary tool objective, letting decoder gradients complement policy rewards. Further, we derive Bootstrapped-GRTO (B-GRTO), a pre-training method that cheaply bootstraps the tool, leading to faster convergence and superior performance. Across three challenging referring segmentation settings, B-GRTO results in substantial improvements over plain GRPO, matching or surpassing domain-specific state-of-the-art methods. This demonstrates the value of unifying reinforcement learning with differentiable auxiliary objectives for reasoning-intensive segmentation.
>
---
#### [new 036] VisAnalog: A Diagnostic Suite for Visual Concept Transfer on Natural Images
- **分类: cs.CV**

- **简介: 该论文提出VisAnalog，用于评估视觉概念迁移的基准。任务是测试模型在变换中保持和操作概念的能力，解决视觉关系推理与变换应用的问题。**

- **链接: [https://arxiv.org/pdf/2605.23141](https://arxiv.org/pdf/2605.23141)**

> **作者:** Zhaonan Li; Kyle R. Chickering; Bangzheng Li; Jacob Dineen; Xiao Ye; Zhikun Xu; Shijie Lu; Yuxi Huang; Ming Shen; Bach Nguyen; Jaya Adithya Pavuluri; Mau Son Nguyen; Sanika Chavan; Ngoc Minh Thu Le; Muhao Chen; Ben Zhou
>
> **备注:** Accepted to the Workshop on Visual Concepts at CVPR 2026 as a non-archival report
>
> **摘要:** A useful test of visual concept learning is not just whether a model can recognize a concept in a single image, but whether it can preserve and manipulate concept-level properties under transformation and transfer them to new scenes. We introduce VisAnalog, a controlled suite for this setting on natural images. Each example instantiates $A\!:\!B::C\!:\,?$: images $B$ and a hidden target image $D$ are produced by applying the same deterministic transformation sequence to source images $A$ and $C$. Given $A$, $B$, and $C$, a model must answer a multiple-choice question about $D$. The benchmark contains 617 human-validated questions spanning one- to four-step transformations such as zoom, quadrant swap, rotation, flip, and hue rotation. Across strong proprietary and open-source VLMs, end-to-end accuracy is substantially lower than oracle accuracy when $D$ is directly shown, and degrades sharply as transformation depth increases, while human performance remains near the ceiling. A program-conditioned evaluation further separates failures of relation inference from failures of transformation application, showing that inferring the visual relation from $A \rightarrow B$ is the dominant bottleneck, with additional application errors emerging on harder multi-step cases. The dataset is publicly available at this https URL.
>
---
#### [new 037] Machine learning applied to emerald gemstone grading: framework proposal and creation of a public dataset
- **分类: cs.CV**

- **简介: 该论文属于宝石分级任务，旨在解决人工评级主观性问题。提出机器学习框架并创建公开数据集，实现自动化分级。**

- **链接: [https://arxiv.org/pdf/2605.23777](https://arxiv.org/pdf/2605.23777)**

> **作者:** FB Pena; D Crabi; Sandro C Izidoro; Érick O Rodrigues; G Bernardes
>
> **摘要:** The grading of gemstones is currently a manual procedure performed by gemologists. A popular approach uses reference stones, where those are visually inspected by specialists that decide which one of the available reference stone is the most similar to the inspected stone. This procedure is very subjective as different specialists may end up with different grading choices. This work proposes a complete framework that entails the image acquisition and goes up to the final stone categorization. The proposal is able to automate the entire process apart from including the stone in the created chamber for the image acquisition. It discards the subjective decisions made by specialists. This is the first work to propose a machine learning approach coupled with image processing techniques for emerald grading. The proposed framework achieves 98% of accuracy (correctly categorized stones), outperforming a deep learning approach. Furthermore, we also create and publish the used dataset that contains 192 images of emerald stones along with their extracted and pre-processed features.
>
---
#### [new 038] EvalVerse: Pipeline-Aware and Expert-Calibrated Benchmarking for Professional Cinematic Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出EvalVerse，用于专业电影视频生成的评估。解决现有评估忽视艺术质量的问题，通过领域知识整合、专家标注和模型优化，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2605.23271](https://arxiv.org/pdf/2605.23271)**

> **作者:** Songlin Yang; Haobin Zhong; Ruilin Zhang; Xiaotong Zhao; Shuai Li; Kai Zheng; Xuyi Yang; Zhe Wang; Zhenchen Tang; Yang Li; Bohai Gu; Zhengwei Peng; Yidan Huang; Mengzhou Luo; Yihang Bo; Dalu Feng; Yujia Zhang; Juntao Ma; Ruiqi Wang; Lvmin Zhang; Yuwei Guo; Frank Guan; Maneesh Agrawala; Hongbo Fu; Alan Zhao; Anyi Rao
>
> **摘要:** The rapid evolution of generative video foundation models has propelled the field toward professional-grade cinematic synthesis. To achieve such demanding quality, the community transitions towards Reinforcement Learning (RL) and agentic workflows. However, reliable evaluation has emerged as a critical bottleneck. Existing benchmarks predominantly evaluate ''whether it is right'' (basic prompt-following) while fundamentally neglecting ''whether it is good'' (cinematic quality, acting, and aesthetics). Furthermore, current automated metrics lack the domain-specific rigor required to provide trustworthy signals, creating a severe credibility gap between human aesthetic perception and machine scoring. To bridge this gap, we introduce EvalVerse, a comprehensive, pipeline-aware, and expert-calibrated evaluation framework. We treat video generation assessment not merely as an engineering task, but as a core scientific problem: the systematic digitization of subjective cinematic expertise. First, we organize domain knowledge into an evaluation taxonomy aligned with the professional filmmaking workflow (pre-production, production, and post-production). Second, we distill human expert judgments into a curated dataset with large-scale human annotations. Third, we inject this knowledge into Vision-Language Models (VLMs) through an expert-calibrated fine-tuning strategy, enabling the VLM to perform explicit Chain-of-Thought reasoning. Compared to previous works, EvalVerse not only retains compatibility with foundational ''rightness'' metrics, but also significantly expands the criteria to ''goodness'' and broaden the task coverage to complex multi-shot sequencing and audio-visual integration. Consequently, by providing granular diagnostic signals, EvalVerse transcends a static leaderboard and establishes a fundamental infrastructure for future work, such as reward models and evaluator agent.
>
---
#### [new 039] VINS-120K: Ultra High-Resolution Image Editing with A Large-Scale Dataset
- **分类: cs.CV**

- **简介: 该论文聚焦于超高清图像编辑任务，解决数据不足与高频纹理建模难题。构建了首个大规模数据集VINS-120K，并提出改进模型策略以提升细节与真实感。**

- **链接: [https://arxiv.org/pdf/2605.23518](https://arxiv.org/pdf/2605.23518)**

> **作者:** Zhizhou Chen; Shanyan Guan; Zhanxin Gao; En Ci; Yanhao Ge; Wei Li; Zhenyu Zhang; Jian Yang; Ying Tai
>
> **摘要:** Directly editing ultra-high-resolution (UHR) images is valuable but underexplored, primarily due to the lack of high-quality data and the challenge in modeling high-frequency texture details. We introduce VINS-120K, the first large-scale dataset for instruction-based UHR image editing, comprising 120K carefully curated triplets of instruction, input image, and edited image. Each image exceeds 4K resolution ($\geq$4096 $\times$ 4096) and is filtered through a rigorous multi-stage pipeline to ensure visual quality, instruction alignment, and aesthetic fidelity. Built on VINS-120K, we further develop a high-frequency-aware post-adaptation strategy to extend pretrained non-high-resolution models to the UHR regime. We also present VINS-4KEval, a benchmark covering diverse editing types, to facilitate consistent evaluation in UHR settings. Experiments confirm that our work improves fine-grained detail synthesis and texture realism in UHR image editing.
>
---
#### [new 040] Suicide Risk Assessment from AI-powered Video Surveillance: An Interpretable Framework for Prevention in Metro Stations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自杀风险评估任务，旨在通过视频监控识别地铁站高风险行为。工作包括构建可解释框架，整合跟踪、活动识别与热力图建模，提升干预有效性。**

- **链接: [https://arxiv.org/pdf/2605.22904](https://arxiv.org/pdf/2605.22904)**

> **作者:** Safwen Naimi; Wassim Bouachir; Guillaume-Alexandre Bilodeau; Brian Mishara
>
> **备注:** 9 pages, 6 figures, 1 table. Accpted for Publication in IJCAI 26
>
> **摘要:** Understanding and monitoring human behavior in metro stations play an important role in supporting suicide prevention efforts, where early identification of high-risk situations can enable timely intervention. This requires assessing suicide risk from a surveillance video by jointly reasoning about the behavior of each passenger, his/her spatial context, and temporal dynamics. However, this assessment using videos captured by surveillance cameras is challenging, as it demands accurate perception of human motion, understanding of platform geometry, and aggregation of heterogeneous behavioral cues over time. In this work, we formalize the task of Suicide Risk Assessment (SRA) in metro stations and introduce the first interpretable framework that addresses this challenge. Unlike approaches that focus on isolated subtasks or attempt to infer intent directly, our formulation assesses suicide risk from accumulated evidence by incorporating person tracking, activity recognition, semantic segmentation of the platform, and trajectory-driven risk heatmap modeling. By formalizing SRA as a distinct task and benchmarking a complete operational pipeline achieving 83.2% ROC-AUC on real surveillance data, this work highlights the complexity of suicide risk assessment and opens new directions for research on interpretable AI systems for social good.
>
---
#### [new 041] MDS-DETR: DETR with Masked Duplicate Suppressor
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决DETR收敛慢、召回率低的问题。通过引入MDS-DETR，结合一对一和一对多监督，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.23507](https://arxiv.org/pdf/2605.23507)**

> **作者:** Chanho Lee; Seunghee Koh; Yunho Jeon; Junmo Kim
>
> **备注:** code is available at this https URL
>
> **摘要:** The DEtection TRansformer (DETR) is a powerful end-to-end object detector, yet its one-to-one matching strategy suffers from slow convergence and low recall. A common approach to address this issue is to use one-to-many label assignment to provide more positive samples. However, existing methods that use one-to-many matching as an auxiliary objective lead to increased training costs, with their auxiliary decoders discarded during inference. To address this limitation, we propose MDS-DETR, which leverages both one-to-one and one-to-many supervision within a single decoder. Specifically, we introduce a Masked Duplicate Suppressor (MDS) that injects asymmetry into self-attention via confidence-based causal masking. MDS filters out the duplicates generated by the one-to-many supervised layer, enables explainable, duplicate-free predictions in a fully end-to-end framework. MDS-DETR outperforms existing one-to-many DETR variants such as MS-DETR, this http URL and Relation-DETR, without relying on any additional queries or auxiliary decoders. Under a 12-epoch training schedule on MS COCO with a ResNet-50 backbone, MDS-DETR achieves a +2.8 mAP improvement over Deformable-DETR with only a 5\% increase in training time, and outperforms the state-of-the-art this http URL by +0.3 mAP while being even 20\% faster in training. Our code and models are available at \href{this https URL}{this https URL}.
>
---
#### [new 042] Good Token Hunting: A Hitchhiker's Guide to Token Selection for Visual Geometry Transformers
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 该论文针对视觉几何Transformer的计算效率问题，提出一种分阶段的token选择策略，以降低计算成本并提升性能。**

- **链接: [https://arxiv.org/pdf/2605.23892](https://arxiv.org/pdf/2605.23892)**

> **作者:** Shuhong Zheng; Michael Oechsle; Erik Sandström; Marie-Julie Rakotosaona; Federico Tombari; Igor Gilitschenski
>
> **备注:** Project Page: this https URL, Code: this https URL
>
> **摘要:** Visual geometry transformers have become powerful architectures for multi-view 3D reconstruction, enabling joint prediction of multiple 3D attributes in a feed-forward manner. However, their computational cost grows quadratically with the input sequence length due to the global attention layers inside these models. This limits both their scalability and efficiency. In this work, we address this challenge with a simple yet general strategy: restricting the number of key/value tokens that each query interacts with during global attention. To achieve effective token selection, we introduce a two-stage framework. First, an inter-frame selection step operates at the frame level to identify frames that should be preserved. Second, an intra-frame selection step further discards more redundant tokens within the selected frames. Our analysis highlights the advantage of a diversity-based strategy for inter-frame selection, which ensures broad coverage of the scene. For intra-frame selection, we show that layer-aware sparsification is necessary, with the selection process guided by the entropy of the global attention pattern. Our approach offers a superior speed-accuracy trade-off compared to existing solutions. Extensive experiments show that it accelerates visual geometry transformers by over 85% for scenes with 500 images while maintaining, or even improving, baseline performance, which hints that how our token selection strategy can play a crucial role in future applications of visual geometry transformers. Our project website is available at this https URL.
>
---
#### [new 043] Revitalizing Dense Material Segmentation: Stabilized Vision Transformers and the Generalization Paradox
- **分类: cs.CV**

- **简介: 该论文属于材料分割任务，旨在解决传统模型在无定形纹理区域表现不佳的问题。通过优化Vision Transformer，提出稳定训练方法，提升分割性能，并揭示了泛化悖论。**

- **链接: [https://arxiv.org/pdf/2605.23747](https://arxiv.org/pdf/2605.23747)**

> **作者:** Allan Kazakov; Duygu Cakir; Hilal Kurt İrfanoğlu; Yavuz İrfanoğlu
>
> **摘要:** Material segmentation, the pixel-wise classification of physical surface properties, remains a challenging problem in computer vision, requiring physicochemical understanding distinct from object-centric parsing. Despite the introduction of the rigorous Apple Dense Material Segmentation (DMS) dataset, the benchmark has suffered from attrition and stagnation, increasingly overshadowed by geometry-biased foundation models. In this paper, we revive the Apple-DMS benchmark to establish a modern Vision Transformer baseline. We conduct an exhaustive evaluation of SegFormer and Mask2Former architectures, revealing that standard training paradigms fail on amorphous texture fields due to high-variance gradients. To address this, we introduce a stabilized training recipe featuring High-Fidelity Logit Projection, Query Entropy Regularization, and a domain-specific, physics-compliant augmentation pipeline. Our optimized SegFormer-B5 achieves a new State-of-the-Art (SOTA) of 0.4572 mIoU on the original dataset split, significantly surpassing the prior convolutional baseline. Furthermore, we identify a critical "Generalization Paradox": while re-partitioning the dataset into a data-rich 80/10/10 split inflates the metric to 0.5276 mIoU, expert qualitative analysis reveals this induces distributional homogenization, severely degrading real-world, out-of-distribution performance. By releasing our recovered dataset index and robust training framework, we demonstrate that material perception is far from solved and urge the community to leverage the rigorous original split to drive genuine progress in physically grounded artificial intelligence.
>
---
#### [new 044] Inconsistency-aware Multimodal Schrödinger Bridge for Deepfake Localization
- **分类: cs.CV**

- **简介: 该论文属于深度伪造定位任务，解决跨模态噪声干扰问题。提出IaMSB方法，联合估计跨模态一致性并实现区间定位，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.23113](https://arxiv.org/pdf/2605.23113)**

> **作者:** Jiayu Xiong; Jing Wang; Qi Zhang; Wanlong Wang; Jun Xue
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Audio-visual deepfake localization demands interval-level outputs that serve as temporal evidence. Despite recent progress, symmetric fusion under single-sided or asynchronous forgeries propagates cross-modal noise, degrading high-precision localization. We present IaMSB, an inconsistency-aware multimodal Schrödinger Bridge (SB) that jointly estimates cross-modal consistency and performs interval-level localization. Unlike diffusion models, SB minimizes path-distribution discrepancy and yields consistency scores without explicit noise injection or denoising. With the Schrödinger Bridge (SB), IaMSB unifies consistency estimation, cross-modal information selection, and bridge-step scheduling in one framework. Specifically, a lightweight coarse bridge first proposes candidate intervals and estimates cross-modal consistency; these statistics select cross-modal witness signals and allocate bridge steps asymmetrically across modalities. A refinement bridge then performs step-tuned fusion and outputs refined, time-aligned intervals. IaMSB anticipates single-sided and asynchronous forgeries and, using bottlenecked cross-modal interaction with step allocation, suppresses noise transfer, avoids unnecessary iterations. Across benchmarks, IaMSB stabilizes strict-IoU boundary precision, raising AP@0.95 by 3%~10%, and yields improved high-precision localization, particularly for single-sided forgeries.
>
---
#### [new 045] Efficient One-Step Diffusion Restoration Model with Compact Token Compression and Linear Attention
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决高分辨率图像恢复中的计算效率问题。提出SANA-SR框架，通过压缩表示和线性注意力机制实现高效恢复。**

- **链接: [https://arxiv.org/pdf/2605.23451](https://arxiv.org/pdf/2605.23451)**

> **作者:** Bingtian Qiao; Yue Shi; Yingjie Zhou; Yong Guo; Guangtao Zhai; Jiezhang Cao
>
> **摘要:** Real-world image super-resolution aims to recover high-quality images from complex and unknown real-world degradations. However, existing generative Real-ISR methods largely inherit the dense latent representations and quadratic-cost global modeling paradigm developed for high-resolution image synthesis, causing computation, memory usage, and inference latency to scale unfavorably with resolution and thus limiting practical deployment. We argue that the key bottleneck lies not in insufficient restoration priors, but in excessive token redundancy and costly token interactions during high-resolution restoration. Motivated by this observation, we revisit Real-ISR from the perspectives of compact latent representation and linear-complexity modeling, and propose SANA-SR, an efficient one-step restoration framework. Specifically, SANA-SR employs a deep compression autoencoder with a 32x compression ratio to drastically reduce latent tokens while preserving restoration-relevant structures and textures. On top of this compact latent space, we introduce a linear-attention DiT with LoRA fine-tuning, enabling efficient high-resolution restoration with linear-complexity token mixing. Extensive experiments on all benchmark datasets demonstrate that SANA-SR achieves highly competitive and often superior quantitative performance against existing methods, while restoring clearer and more realistic textures. Moreover, after pruning, the deployed model runs in 0.019s with 407.95G MACs and 344M parameters, highlighting its strong potential for practical mobile deployment.
>
---
#### [new 046] Exploring deep learning for Event-Based Saliency Prediction with a Transformer-based model
- **分类: cs.CV**

- **简介: 该论文属于事件数据的显著性预测任务，旨在解决事件相机数据缺乏标注和基准的问题。提出SEST模型，结合预训练Transformer与轻量CNN，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2605.23790](https://arxiv.org/pdf/2605.23790)**

> **作者:** Romaric Mazna; Jean Martinet; Sai Deepesh Pokala
>
> **摘要:** Saliency prediction has been extensively studied in RGB images and videos as a computational model of human visual attention. In contrast, predicting saliency from event-based data remains largely unexplored, despite the biological inspiration and favorable sensing properties of event cameras. Two obstacles have held this direction back: the absence of large-scale event saliency datasets, and the lack of a strong baseline. In this paper, we introduce SEST (Swin Event-based Saliency Transformer), a transformer-based model for saliency prediction from event data, bridging the data scarcity barrier through event-native pretraining and synthetic supervision. SEST leverages a self-supervised pretrained event-based Swin Transformer backbone combined with a lightweight CNN decoder to produce dynamic saliency maps. To address the scarcity of annotated event-based saliency data, we introduce two new benchmark datasets, N-DHF1K and N-UCF Sports, generated from large-scale RGB saliency benchmarks. Experimental results show that SEST clearly outperforms existing event-based saliency methods and narrows the performance gap with state-of-the-art RGB models. Zero-shot evaluation on a real event camera dataset further demonstrates that our model trained on synthetic data remains transferable on real event streams. To the best of our knowledge, this work is the first to apply deep learning to event-based saliency prediction, opening a new research direction at the intersection of event-based vision and neuromorphic visual attention.
>
---
#### [new 047] DFSAttn: Dynamic Fine-grained Sparse Attention for Efficient Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决扩散Transformer中注意力机制计算成本高的问题。提出DFSAttn框架，实现高效稀疏注意力，提升速度与质量。**

- **链接: [https://arxiv.org/pdf/2605.23445](https://arxiv.org/pdf/2605.23445)**

> **作者:** Jie Hu; Zixiang Gao; Yutong He; Kun Yuan
>
> **备注:** ICML 2026; 17 pages, 8 figures;
>
> **摘要:** Diffusion transformers have achieved remarkable success in high-quality video generation, yet their reliance on spatiotemporal 3D full attention incurs prohibitive computational cost due to the quadratic complexity of attention. Block sparse attention is a common approach to mitigate this by focusing computation on important regions. However, attention maps in DiTs exhibit inherently dynamic and fine-grained sparsity, which causes existing block sparse attention methods to degrade significantly in quality, especially at high sparsity ratios. In this paper, we revisit block sparse attention and derive a theoretical lower bound on attention recall to characterize the key factors governing its effectiveness. Guided by these insights, we propose DFSAttn, a training-free sparse attention framework that enables dynamic, fine-grained sparsification efficiently. DFSAttn incorporates three core designs: Hilbert curve-based token reordering to achieve fine-grained sparsity while preserving efficient GPU execution, hierarchical block scoring for accurate block importance estimation, and sparse mask caching with adaptive ratios to balance accuracy and efficiency. Experimental results demonstrate that DFSAttn consistently outperforms prior methods under high sparsity, achieving up to 2.1$\times$ end-to-end speedup while maintaining high generation quality. Our code is open-sourced and available at this https URL.
>
---
#### [new 048] Dithering Defense: Adversarial Robustness of Vision Foundation Models via Multi-Level Floyd-Steinberg Dithering
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究如何通过多级Floyd-Steinberg抖动提升视觉基础模型的对抗鲁棒性，解决对抗攻击问题。工作包括多任务、多模型和多攻击评估，验证抖动的有效性。**

- **链接: [https://arxiv.org/pdf/2605.23065](https://arxiv.org/pdf/2605.23065)**

> **作者:** Yury Belousov; Brian Pulfer; Vitaliy Kinakh; Slava Voloshynovskiy
>
> **备注:** Paper accepted at the IEEE International Conference on Image Processing (ICIP 2026)
>
> **摘要:** Vision foundation models are widely used as frozen backbones across many downstream tasks, making them a single point of failure under adversarial attack. We study multi-level Floyd-Steinberg error-diffusion dithering as a lightweight, model-agnostic input transformation that disrupts adversarial perturbations while preserving semantic content. Unlike prior work, which was limited to binary dithering, grayscale CIFAR-10, and a single small model trained from scratch, we evaluate across six tasks (classification, segmentation, depth estimation, retrieval, captioning, visual question answering), two model families (DINOv2, PaliGemma), and three attacks of increasing strength (PGD, MI-FGSM, SIA), as well as an adaptive attacker using a straight-through estimator. Our results show that Floyd-Steinberg dithering at intermediate quantization levels, especially when combined with post-processing blur, exceeds or matches all tested baselines, including diffusion-based denoising, with substantially less degradation on clean inputs.
>
---
#### [new 049] The TIME Machine: On The Power of Motion for Efficient Perception
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频表示学习任务，旨在解决视频模型训练成本高和依赖语言的局限性。提出基于运动的自监督方法，构建TIME嵌入，提升时序理解与可扩展性。**

- **链接: [https://arxiv.org/pdf/2605.23045](https://arxiv.org/pdf/2605.23045)**

> **作者:** Mantas Skackauskas; Xinyue Hao; Laura Sevilla-Lara
>
> **摘要:** Video representation learning has seen tremendous progress in recent years. This has been driven by many factors, including the scale of training and the success of visual models trained contrastively with language. While these factors have pushed the boundaries of what video models can do, they also introduce their own set of limitations: first, scaling video models can reach prohibitive costs and second, learning from language restricts the range of concepts that can be learned to those in captions. As a result, video models still struggle with temporal understanding. In this paper we propose a novel approach that uses motion as the central modality for video representation. In particular, given the motion in a video in the form of point-tracks, we use a masked-autoencoder to mask some of the tracks and train the autoencoder to reconstruct the missing tracks. This allows us to learn a representation in a self-supervised manner. We show that using motion to represent videos actually addresses both of the core limitations of video technology. First, it allows us to massively reduce the scale of training data, as motion is inherently appearance-independent and hence needs fewer examples to generalize well. Second, motion allows us to bypass the language-dependent training paradigm, learning better fine-grained concepts. The result is an embedding that we call TIME (Temporally Informed Motion Embedding), a representation trained exclusively on synthetic motion data. We test this embedding on a wide set of tasks in a zero-shot manner. We observe that without bells and whistles, performance is on par with state-of-the-art models using up to 4 orders of magnitude less training data. This is a stepping stone towards a new paradigm of video models that are both more temporally aware as well as more scalable.
>
---
#### [new 050] Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks
- **分类: cs.CV**

- **简介: 该论文属于工业视觉检测任务，探讨自监督预训练模型（如DINOv3）与传统ImageNet预训练在RGB和X-ray任务中的迁移效果。研究比较了不同预训练方法在冻结和微调设置下的表现，分析其适应性与有效性。**

- **链接: [https://arxiv.org/pdf/2605.23472](https://arxiv.org/pdf/2605.23472)**

> **作者:** Mehdi Gharbage; Céline Teulière; Pierre Bouges; Thierry Chateau
>
> **备注:** Accepted to the CVPR 2026 Workshop on Vision Foundation Models for Industrial Inspection (VISION'26)
>
> **摘要:** Vision foundation models pretrained on web-scale data have recently shown strong transfer capabilities on many downstream tasks, but their effectiveness for industrial visual inspection remains unclear. Industrial data differ substantially from web-data and often require fine-grained dense prediction, raising the question of whether modern self-supervised pretraining can improve over the conventional transfer-learning paradigm based on supervised ImageNet initialization. In this work, we compare ConvNeXt backbones pretrained with supervised ImageNet classification or DINOv3 distillation, and relate them to the conventional ResNet-50 baseline. We evaluate semantic segmentation, instance segmentation, and object detection across four downstream datasets spanning RGB surface-defect inspection and X-ray defect detection. We further study both frozen and fully finetuned adaptation regimes. Our results show that DINOv3 offers no clear advantage in frozen transfer, but provides a stronger initialization after full finetuning on RGB tasks, yielding faster convergence and better final performance. Under X-ray modality shift, however, supervised ImageNet pretraining remains more effective in both frozen and finetuned settings. Overall, our findings suggest that modern vision foundation models are promising for supervised RGB industrial inspection, but their transferability is strongly conditioned by downstream adaptation and target modality.
>
---
#### [new 051] Exploiting Longitudinal Context in Clinician-Verified Interactive Lesion Tracking
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分割任务，解决肿瘤跟踪中的自动与人工验证难题。提出一种结合临床验证的跟踪框架，提升分割准确性。**

- **链接: [https://arxiv.org/pdf/2605.23118](https://arxiv.org/pdf/2605.23118)**

> **作者:** Yannick Kirchhoff; Maximilian Rokuss; Daniel Philipp Mertens; David Füller; Benjamin Hamm; Andreas Schreyer; Oliver Ritter; Klaus Maier-Hein
>
> **备注:** Accepted at MICCAI 2026
>
> **摘要:** Tracking tumor lesions across serial CT scans is essential for oncological response assessment. Existing automated methods face a fundamental trade-off: end-to-end trackers achieve high automation but offer no opportunity to correct silent tracking failures, while decoupled registration-segmentation pipelines permit user verification yet discard the lesion's prior appearance, limiting accuracy in ambiguous cases. In this work, we propose a Verified Tracking paradigm: a clinician verifies a registration-proposed prompt, which the model leverages alongside the baseline lesion appearance to resolve segmentation ambiguities. We present a unified framework combining early spatial prompt fusion with latent temporal difference weighting for longitudinally-informed segmentation. To address data scarcity, we leverage large-scale synthetic pretraining, proving essential for exploiting longitudinal context, improving performance by up to 4.5 Dice points over training from scratch. Our approach secured first place in the MICCAI autoPET IV challenge. We further curate and release PanTrack, a new longitudinal pancreatic cancer benchmark, to assess out-of-distribution generalization. Experiments show that our model outperforms prior work in both fully automatic and the proposed verified tracking setting offering a clinically safe middle ground between automation and control. Code, model and dataset will be released at this https URL
>
---
#### [new 052] LQ-rPPG: A Label-Quantized Coarse-to-Fine Learning Framework for Remote Physiological Measurement
- **分类: cs.CV**

- **简介: 该论文属于远程生理信号估计任务，解决标签噪声影响模型性能的问题。提出LQ-rPPG框架，通过标签量化和分层学习提升rPPG估计的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.23174](https://arxiv.org/pdf/2605.23174)**

> **作者:** Jun Seong Lee; Samyeul Noh; Changki Sung; Hyun Myung
>
> **摘要:** Remote photoplethysmography (rPPG) enables non-contact measurement of physiological signals from facial videos, offering strong potential for remote healthcare and daily health monitoring. Driven by this potential, various deep learning-based rPPG methods have been proposed to improve rPPG estimation. However, previous deep learning-based rPPG methods have paid little attention to the quality of training labels and their impact on model learning. Contact-based PPG signals used as training labels often contain noise and variability caused by motion artifacts, inconsistent sensor contact, and morphological distortions. Such label inconsistency can lead models to overfit to the label noise and variability and consequently degrade generalization performance. To address this issue, we propose LQ-rPPG, a label-quantized coarse-to-fine learning framework for robust rPPG estimation. LQ-rPPG consists of a label quantization module and a coarse-to-fine rPPG estimation model. The label quantization module transforms continuous PPG signals into multi-bit quantized pseudo labels with reduced noise and variability. The coarse-to-fine estimation model progressively refines rPPG signals under hierarchical supervision guided by the multi-bit pseudo labels. This design alleviates overfitting to label-specific variations and enables the model to learn structured and consistent representations. As a result, LQ-rPPG achieves robust and generalizable rPPG estimation even under challenging conditions. Experiments on multiple benchmark datasets demonstrate that LQ-rPPG achieves strong performance in both intra- and cross-dataset evaluations, while reducing parameters and multiply-accumulate operations by 88% and 29%, respectively, and increasing throughput by 191%. The code is available at this https URL.
>
---
#### [new 053] DDX-TRACE: A Benchmark for Medical Diagnostic Trajectories in VLMs
- **分类: cs.CV**

- **简介: 该论文提出DDX-TRACE，一个评估医学诊断轨迹的基准，解决传统AI评测仅关注最终答案的问题。通过模拟真实诊断过程，评估模型在证据支持下的决策质量。**

- **链接: [https://arxiv.org/pdf/2605.23629](https://arxiv.org/pdf/2605.23629)**

> **作者:** Jiazhen Pan; Weixiang Shen; Jun Li; Julian Canisius; Felix Bitzer; Paula Roßmüller; Jiancheng Yang; Virginie Kreutzinger; Daniel Rueckert; Benedikt Wiestler
>
> **备注:** 41 pages
>
> **摘要:** Medical diagnosis is not a single prediction from a fully specified vignette. It is a sequential workup: clinicians decide what evidence to obtain, revise a differential diagnosis, and stop when the diagnosis is sufficiently supported. Most medical AI benchmarks instead reveal the relevant context upfront and score only the final answer, making unsupported correct guesses, premature closure, inefficient workups, and poor uncertainty updating invisible. We introduce DDX-TRACE, a physician-adjudicated benchmark for multimodal neuroradiology that evaluates diagnostic trajectories under hidden evidence over 211 challenging cases. Each case begins with limited clinical history; models request imaging studies in free form, receive matched image bundles when available, update a probabilistic differential diagnosis after each turn, and stop with a localized final diagnosis. Evaluating state-of-the-art VLMs, we find that final diagnosis scores can substantially misrepresent workup quality: models may guess plausible diagnoses without essential evidence, request useful studies but misinterpret raw images, or acquire evidence inefficiently while updating uncertainty poorly. Controlled evidence variants isolate bottlenecks in planning, visual evidence extraction, and downstream differential reasoning. DDX-TRACE shifts medical AI evaluation from final answers to evidence-supported diagnostic trajectories.
>
---
#### [new 054] CoMoGen: COntrollable MOtion Dynamics and Interactions with Mask-Guided Video GENeration
- **分类: cs.CV**

- **简介: 该论文提出CoMoGen，属于可控视频生成任务，解决如何根据二值掩码序列生成真实交互动态的问题。通过引入MaskAdapter和运动层精调，提升运动精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.22996](https://arxiv.org/pdf/2605.22996)**

> **作者:** Adil Meric; Lin Geng Foo; Mert Kiray; Benjamin Busam; Rishabh Dabral; Christian Theobalt
>
> **摘要:** We present CoMoGen, a controllable video generation framework that generates realistic interactive dynamics from a single binary mask sequence conditioned on an input image. CoMoGen introduces a lightweight MaskAdapter that encodes binary mask sequences into a latent residual signal, injected into the Multi Modal Diffusion Transformer (MMDiT) model through a cosine-weighted schedule. Unlike the hierarchical coarse-to-fine design of UNet architectures, MMDiT operates as a sequence of uniform transformer blocks, making it difficult to identify which layers are responsible for the motion generation. Therefore, we propose a novel way to determine "Motion Layers" operating in the attention space of MMDiT. We fine-tune the model by using Low-Rank Adaptation (LoRA) to the Motion Layers, without requiring any architecture change in the MMDiT. This selective adaptation enables our method to focus on motion-critical components, yielding reduced computational cost. Despite its simplicity, CoMoGen enables precise subject motion and plausible interactions with surrounding humans, objects, and scenes. Comprehensive experiments on different datasets show that CoMoGen consistently outperforms prior controllable video generation methods and achieves state-of-the-art performance in motion fidelity and perceptual realism. Project page: this http URL.
>
---
#### [new 055] CaST-Bench: Benchmarking Causal Chain-Grounded Spatio-Temporal Reasoning for Video Question Answering
- **分类: cs.CV**

- **简介: 该论文属于视频问答任务，旨在解决视频中的因果推理问题。针对现有基准不足，提出CaST-Bench，构建包含因果链的高质量数据集，并设计评估指标以提升模型的因果理解能力。**

- **链接: [https://arxiv.org/pdf/2605.23216](https://arxiv.org/pdf/2605.23216)**

> **作者:** Mingfang Zhang; Jingjing Pan; Ashutosh Kumar; Rajat Saini; Mustafa Erdogan; Hsuan-Kung Yang; Caixin Kang; Yifei Huang; Yoichi Sato; Quan Kong
>
> **备注:** CVPR 2026
>
> **摘要:** Cause-and-effect reasoning in video is a significant challenge for Vision-Language Models (VLMs), as it requires going beyond surface-level perception to a deeper understanding of causal mechanisms. However, existing benchmarks rarely provide the fine-grained, grounded evidence needed to rigorously evaluate this capability. To address this gap, we introduce CaST-Bench, a benchmark for Causal Chain-Grounded Spatio-Temporal Video Reasoning. CaST-Bench presents complex causal questions that require models to identify and localize a chain of multiple spatio-temporal evidences. Through a human-AI collaborative pipeline, we construct a high-quality dataset of 2,066 questions over 1,015 videos, with causal chains annotated by temporal segments and bounding-box tracks. Furthermore, we design a comprehensive evaluation suite with novel metrics that assess not only answer correctness but also the capability for visual evidence grounded reasoning. This grounding is crucial for improving accuracy by mitigating spurious correlations and for enhancing user trust by making models more transparent. Our experiments show that current VLMs struggle with causal questions, largely due to their limited ability to construct precise and grounded causal chains. This highlights an important direction for improving future VLMs.
>
---
#### [new 056] SimInsert: Seamless Video Object Insertion via Regional Sparse Attention Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频对象插入任务，解决传统方法依赖运动工程或重训练的问题。提出SimInsert，通过单帧编辑和语义运动描述实现高效、高质量的视频对象插入。**

- **链接: [https://arxiv.org/pdf/2605.23245](https://arxiv.org/pdf/2605.23245)**

> **作者:** Xinyu Chen; Yuyi Qian; Jiang Lin; Shenyi Wang; Gao Wang; Zhiqiu Zhang; Jizhi Zhang; Mingjie Wang; Qiang Tang; Qian Wang; Song Wu; Zili Yi
>
> **备注:** Accepted by ICME2026
>
> **摘要:** Video object insertion requires ensuring spatio-temporal coherence and interactive realism, extending far beyond simple content placement. However, current approaches are often hindered by a reliance on explicit motion engineering or resource-intensive retraining, restricting their flexibility and generalization. To bridge this gap, we present \textit{SimInsert}, a training-free paradigm that efficiently decouples the task into intuitive single-frame editing and semantic motion description. By harnessing the robust generative priors of image-to-video diffusion models, SimInsert propagates edits temporally, strictly preserving background invariance while enabling plausible, text-driven interactions between the inserted object and the dynamic environment. Our approach hinges on non-invasive guidance mechanisms that enforce structural consistency, facilitate seamless boundary fusion, and counteract the fidelity drift that typically accumulates during the denoising trajectory. Extensive quantitative experiments validate our efficacy: SimInsert surpasses state-of-the-art methods with an 18.8\% gain in PSNR, 20.1\% in SSIM, and a 44.1\% decrease in LPIPS, offering a streamlined solution for high-fidelity video editing.
>
---
#### [new 057] ChainFlow-VLA: Causal Flow Planning with Vision-Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出ChainFlow-VLA，解决自动驾驶中因果推理与全局优化不统一的问题，通过结合自回归生成和扩散模型，提升轨迹规划的准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2605.23270](https://arxiv.org/pdf/2605.23270)**

> **作者:** Xiyang Wang; Xinlin Wang; Tingguang Zhou; Gong Chen; Xingtai Gui; Zhi Xu; Xiaolei Wu; Feiyang Tan; Hangning Zhou; Mu Yang
>
> **摘要:** Current end-to-end autonomous driving systems are fundamentally limited by a mismatch between temporal causal reasoning and global trajectory consistency. Autoregressive (AR) models capture interaction-aware temporal dependencies via causal factorization, but their step-wise decoding leads to error accumulation and suboptimal global structure. In contrast, diffusion models optimize trajectories globally but lack explicit causal constraints, making them unreliable in interactive and safety-critical scenarios. This dichotomy reveals a deeper issue: existing methods treat causal modeling and global optimization as separate paradigms, without a principled way to unify them within a single trajectory distribution. To address this, we propose ChainFlow-VLA, which unifies causal generation and global refinement within a unified probabilistic framework. We formulate planning as a mixture over AR-induced modes and learn Vision-Language Model (VLM)-conditioned residual distributions over these modes. An autoregressive generator (Chain) produces a discrete set of causal trajectory modes, followed by a diffusion-based refiner (Flow) that leverages VLM hidden states as semantic priors to perform mode-conditioned correction in residual space while preserving causal structure. This straightforward conditioning seamlessly injects high-level scene understanding into fine-grained trajectory adjustments. Experiments demonstrate that ChainFlow-VLA achieves robust planning in ambiguous and long-tail scenarios, achieving a state-of-the-art score of 94.85 on the NAVSIM v1 leaderboard, matching human-level performance (94.8). Code will be available at this https URL.
>
---
#### [new 058] PGT: Procedurally Generated Tasks for improving visual grounding in MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉基础任务，旨在解决MLLMs在细粒度理解上的不足。通过生成程序化任务（PGT），增强视觉接地能力并诊断感知失败原因。**

- **链接: [https://arxiv.org/pdf/2605.23883](https://arxiv.org/pdf/2605.23883)**

> **作者:** Rim Assouel; Amir Bar; Michal Drozdzal; Adriana Romero-Soriano
>
> **摘要:** Despite remarkable progress in Multimodal Large Language Models (MLLMs), these models still struggle with fine-grained understanding tasks. In this work, we propose Procedurally Generated Tasks (PGT), a simple data-driven framework that serves a dual purpose: inducing fine-grained visual understanding and acting as a low-cost diagnostic tool to identify the source of perception failures. By overlaying unambiguous geometric primitives on images, PGT generate additional dense supervision that disentangles visual grounding capability from semantic priors. Extensive experiments on relational, quantitative, and 3D/depth understanding benchmarks show that PGT yields remarkable gains across diverse architectures. Instruction tuning MLLMs on LLaVA-v1.5-Instruct augmented with PGT data results in improvements of up to +20% on the What'sUp benchmark and +13.3% on CV-Bench-2D, while maintaining general perception capabilities. Moreover, finetuning state-of-the-art MLLMs on PGT data leads to boosts of up to +5.5% on What'sUp and +8.3% on CV-Bench-2D. These findings demonstrate that PGT effectively address the bottleneck of fine-grained perception, revealing that many spatial reasoning deficits stem from inadequate supervision signals rather than inherent architectural or resolution limitations.
>
---
#### [new 059] Enhancing Blood Cells Classification using Hybrid Quantum Neural Networks
- **分类: cs.CV; quant-ph**

- **简介: 该论文属于医学图像分类任务，旨在提升血细胞分类的准确性。通过引入混合量子-经典神经网络，增强特征表示并改善分类性能。**

- **链接: [https://arxiv.org/pdf/2605.23324](https://arxiv.org/pdf/2605.23324)**

> **作者:** Guilherme Cruz; Nouhaila Innan; Alberto Marchisio; Gabriel Falcao; Muhammad Shafique
>
> **备注:** 11 pages, 13 figures
>
> **摘要:** Accurate classification of microscopic blood cells is still a critical task in medical image analysis, where subtle variations and limited data can challenge conventional deep learning models. As such, we investigate in this work the potential of Hybrid Quantum-Classical Neural Networks (HQNNs) to enhance feature representation and improve classification performance in this domain. We propose a modular architecture combining a pre-trained ResNet-50 backbone with a low-dimensional latent bottleneck and a variational quantum circuit, enabling a direct comparison between quantum-enhanced and purely classical transformation mechanisms. To isolate the contribution of the quantum component, we evaluate three architectures: a HQNN model, a Classical Matched Model with an additional nonlinear transformation layer of comparable capacity, and a baseline model without an intermediate transformation stage. Experiments conducted on two publicly available blood cell datasets, namely the Blood Cell Images dataset and the PBC dataset, demonstrate that HQNNs consistently achieve superior or more balanced performance across evaluation metrics. In the Blood Cell Images Dataset, the proposed approach improves macro F1-score by up to 3.7% compared to classical baselines, while improving the F1-score from 98.54% to 98.69% in the more challenging 8-class scenario with near-saturated performance. Additional evaluation on IBM quantum hardware shows that the model remains robust under noise, with only a modest performance degradation relative to simulated results. These results indicate that quantum feature transformations can enhance discriminative representations, particularly in challenging classification scenarios, and highlight the practical potential of HQNN models for medical imaging tasks.
>
---
#### [new 060] SCOPE: Simulating Cross-game Operations in Playable Environments for FPS World Models
- **分类: cs.CV**

- **简介: 该论文属于FPS游戏中的动作模拟任务，解决高频率控制信号冲突问题。提出SCOPE模型，通过局部视觉内容生成动作响应，实现跨游戏泛化。**

- **链接: [https://arxiv.org/pdf/2605.23345](https://arxiv.org/pdf/2605.23345)**

> **作者:** Zizhao Tong; Hongfeng Lai; Zeqing Wang; Zhaohu Xing; Kexu Cheng; Haoran Xu; Zhao Pu; Shangwen Zhu; Ruili Feng; Jian Zhao; Yan Zhang; Hao Tang; Yeying Jin; Ling Shao
>
> **备注:** Project page: this https URL. Code is available at this https URL
>
> **摘要:** Interactive world models for first-person shooter (FPS) games must resolve high-frequency overlapping control signals at every frame without disrupting unaffected regions. Existing methods inject actions globally and train on single titles, failing under dense FPS inputs. We observe that FPS actions are spatially selective: discrete events such as firing or reloading affect only a localized region around the weapon (the scope), while continuous camera and movement signals govern stable surroundings. We propose SCOPE, which inserts a conditioning module into each transformer block of a pretrained video diffusion model. It reshapes features into per-pixel temporal sequences so that each position computes its action response from local visual content. This separates in-scope effects from out-of-scope generation without segmentation labels. We also introduce CrossFPS, the first multi-game FPS dataset with frame-aligned action telemetry. It comprises 69K clips from 7 titles with 10-DoF controller signals, curated to remove gameplay bias. The model learns general visual-to-action mappings rather than game-specific patterns, enabling zero-shot transfer to unseen scenes. Experiments confirm strong action responsiveness, precise scope separation, and effective cross-game generalization.
>
---
#### [new 061] Weierstrass Positional Encoding for Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉Transformer任务，旨在解决位置编码无法有效保留图像二维结构的问题。提出Weierstrass位置编码，利用复数域数学方法构建更精确的二维位置表示。**

- **链接: [https://arxiv.org/pdf/2605.23719](https://arxiv.org/pdf/2605.23719)**

> **作者:** Zhihang Xin; Rui Wang; Xitong Hu; Xiaojun Wu
>
> **摘要:** Vision Transformers have achieved remarkable success in computer vision, but their common use of learnable one-dimensional positional encodings weakens the inherent two-dimensional spatial structure of images after patch flattening. Existing positional encodings often lack geometric constraints and do not preserve a monotonic relationship between Euclidean spatial distances and sequential index distances, limiting ViTs' ability to exploit spatial proximity priors. Motivated by the usefulness of periodicity in positional encoding, we propose Weierstrass elliptic Positional Encoding (WePE), a mathematically grounded method for encoding two-dimensional coordinates in the complex domain. WePE maps normalized 2D patch coordinates onto the complex plane and constructs compact four-dimensional positional features using the Weierstrass elliptic function and its derivative. The double periodicity provides a principled representation of 2D positions, and its intrinsic lattice structure naturally matches the regular geometry of image patch grids. Its nonlinear geometric properties help model spatial distance relationships more faithfully, while the algebraic addition formula enables relative positional information between arbitrary patch pairs to be derived directly from their absolute encodings. WePE is plug-and-play and resolution-agnostic, allowing seamless integration into existing ViTs. Extensive experiments show that WePE brings consistent performance gains in most settings. With precomputed lookup tables, these improvements introduce no noticeable computational or memory overhead. Additional analyses and ablation studies further validate the effectiveness of the proposed method.
>
---
#### [new 062] Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，解决传统重建方法的不足。通过分析流匹配模型中的速度不匹配，识别异常区域，无需额外优化或记忆。**

- **链接: [https://arxiv.org/pdf/2605.23070](https://arxiv.org/pdf/2605.23070)**

> **作者:** Shengzhe Chen; Mehrdad Moradi; Kamran Paynabar; Hao Yan
>
> **摘要:** We propose Flow Mismatching, an unsupervised anomaly detection method that deliberately avoids reconstruction-based paradigms. Instead, we treat flow matching as geometric dynamics and leverage a key insight: anomalies occur at places where the learned normal flow disagrees with the geometric path toward a test image. Given a flow matching model trained only on normal images, we probe its learned velocity field along affine paths from Gaussian noise to a target image. Along each path, we compare the model-predicted velocity, which follows normal generative dynamics, with the geometric velocity toward the target, which includes any anomalous content. Anomalies induce strong local disagreement between these velocities. Aggregating the mismatch over different time steps and multiple paths yields pixel-wise heatmaps and image-level scores without test-time optimization, feature memories, or additional calibration. Our analysis shows that the population mismatch decomposes into an irreducible denoising term and a Fisher-divergence term between the test-path and normal-path score functions, which identifies the score-gap component that drives anomaly separation and explains the effectiveness of robust path aggregation. Extensive experiments on MVTec-AD and VisA demonstrate superior performance compared with SOTA reconstruction-based and recent flow matching-based approaches.
>
---
#### [new 063] From Activation to Causality: Discovery of Causal Visual Representations in the Human Brain
- **分类: cs.CV**

- **简介: 该论文属于神经科学中的脑区功能定位任务，旨在解决如何准确识别大脑中表示特定视觉概念的区域。通过引入BrainCause框架，结合生成模型与脑成像数据，进行因果验证以排除干扰因素，提升定位准确性。**

- **链接: [https://arxiv.org/pdf/2605.23895](https://arxiv.org/pdf/2605.23895)**

> **作者:** Yuval Golbari; Navve Wasserman; Matias Cosarinsky; Roman Beliy; Aude Oliva; Antonio Torralba; Michal Irani; Tamar Rott Shaham
>
> **摘要:** Identifying which brain regions represent a visual concept in the human brain is a central challenge in neuroscience. Existing approaches have localized coarse functional regions (e.g., faces, places) through activation maximization, identifying regions that activate strongly for a target concept relative to other concepts. Yet strong activation alone does not establish that a region represents the concept itself, as responses may instead be driven by correlated visual or semantic cues. We introduce BrainCause, an automated framework that combines generative and brain models to synthesize controlled stimuli and validate neural representations through targeted causal testing. Given a query specifying a concept of interest, our framework constructs targeted stimulus sets comprising concept images, counterfactual edits that remove the target concept while preserving other image content, and images with candidate correlated distractors. It then uses an image-to-fMRI encoding model to predict brain responses and searches for representations that respond specifically to the target concept over correlated alternatives. BrainCause returns validated candidate representations and proposes follow-up fMRI experiments to further test or extend its discoveries. Our approach successfully recovers known functional localizations and identifies new candidate representations across dozens of concepts, validated on both predicted and measured fMRI data. Critically, we show that without causal validation, a large fraction of localizations would be false positives, confirming that activation alone is insufficient evidence of representation.
>
---
#### [new 064] GlowGS: Generative Semantic Feature Learning for 3D Gaussian Splatting in Nighttime Glow Scenes
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决夜间场景中结构特征缺失导致的渲染问题。通过生成语义特征和学习新视角语义，提升夜间场景的3D高斯溅射效果。**

- **链接: [https://arxiv.org/pdf/2605.23602](https://arxiv.org/pdf/2605.23602)**

> **作者:** Beibei Lin; Xiao Cao; Jingyuan Guo; Robby T. Tan
>
> **备注:** Accepted by CVPR Findings 2026
>
> **摘要:** Existing 3DGS methods effectively render high-quality novel views in clear-day scenes. However, they struggle with night scenes, particularly in glow regions, due to the lack of structural features such as textures and edges, which are key cues for splatting-based reconstruction. To address this problem, we leverage a diffusion model and a Vision Foundation Model (VFM) to compensate for missing structural cues. Our method consists of two key novel ideas: semantic feature generation and novel-view semantic learning. First, semantic feature generation produces high-quality semantic features as implicit structural cues for novel views. Specifically, a diffusion model synthesizes novel views with unknown camera poses from training views, while a VFM evaluates their quality. Once high-quality novel views are identified, the VFM extracts robust features to construct the semantic feature bank. Second, novel-view semantic learning enables 3DGS to optimize rendered novel views without requiring ground truth. It achieves this by extracting semantic features from a rendered novel view, searching the feature bank for the most similar features, and minimizing their distance. This process enforces implicit structural constraints, ensuring semantically coherent, artifact-free rendered views. Extensive experiments demonstrate the effectiveness of our GlowGS in generating semantically accurate 3D views, showing significant improvements over existing methods.
>
---
#### [new 065] CHASD: Language Increment-Calibrated Contrastive Decoding against Hallucination in LVLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型生成中的幻觉问题。提出CHASD方法，在推理时按需进行对比校准，提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2605.23344](https://arxiv.org/pdf/2605.23344)**

> **作者:** Xiaoyi Huang; Kejia Zhang; Zhiming Luo
>
> **摘要:** Large Vision-Language Models have shown strong multimodal reasoning capabilities, yet they remain susceptible to object hallucinations when language priors dominate insufficient or misaligned visual evidence. Training-free contrastive decoding methods mitigate this issue by comparing predictions from original and perturbed visual inputs, but existing approaches either apply global perturbations that may alter useful visual evidence or invoke an additional negative branch at every decoding step. In this paper, we observe that hallucination risks are transient and token-specific: visual attention shifts across generated tokens, while some functional tokens are produced with high confidence and do not require contrastive calibration. Based on this observation, we propose Contrastive Hallucination-Aware Step-wise Decoding (CHASD) for Large Vision-Language Models, an inference-time framework for "calibration on demand". CHASD uses an uncertainty-driven confidence gate to activate the contrastive branch only when the maximum probability of the next-token is less than the threshold, and constructs the negative branch through attention-guided localized perturbations of the currently salient visual tokens. This design reduces unnecessary negative-branch forward passes while preserving the original distribution for high-confidence steps. Experiments on POPE, AMBER, MME, MMHal-Bench, and CHAIR show that CHASD improves hallucination-related metrics over strong training-free baselines with competitive inference efficiency.
>
---
#### [new 066] VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决模型推理速度慢的问题。提出VDE方法，通过分解和估计速度实现无需训练的加速，提升效率并保持质量。**

- **链接: [https://arxiv.org/pdf/2605.23381](https://arxiv.org/pdf/2605.23381)**

> **作者:** Junwen Tan; Jinglin Liang; Hongyuan Chen; Shuangping Huang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Though rectified flow models have achieved remarkable performance in image, video, and 3D generation, their practical deployments are challenged by slow inference speeds. Prior acceleration methods reuse cached features from previous steps, which neglects the growing mismatch between static caches and the evolving input, leading to reduced output fidelity. This work proposes Velocity Decomposition and Estimation (VDE), a training-free acceleration method that shifts the paradigm from caching-and-reusing to decomposing-and-estimating. Specifically, VDE decomposes the model's velocity into components parallel and orthogonal to the input, exploiting their temporal predictability and directional stability for precise, input-adaptive estimation. To prevent error accumulation, it periodically anchors the model's state via full forward passes. Extensive experiments on image and video generation tasks demonstrate that VDE achieves substantial acceleration with minimal loss in visual quality. Notably, VDE accelerates Flux by 3.22 times and achieves an LPIPS of 0.069 on Qwen-Image, outperforming the best baseline with a 52.2% reduction.
>
---
#### [new 067] PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion
- **分类: cs.CV**

- **简介: 该论文提出PiD，解决高分辨率图像生成中的解码效率问题。通过像素扩散实现快速高质量图像生成，提升解码速度与效果。**

- **链接: [https://arxiv.org/pdf/2605.23902](https://arxiv.org/pdf/2605.23902)**

> **作者:** Yifan Lu; Qi Wu; Jay Zhangjie Wu; Zian Wang; Huan Ling; Sanja Fidler; Xuanchi Ren
>
> **备注:** Project Page: this https URL
>
> **摘要:** Most practical high-resolution text-to-image systems, including latent diffusion and autoregressive models, perform generation in a compact latent space, and a decoder maps the generated latents back to pixels. Yet the latent-to-pixel decoder is reconstruction-oriented, optimized to invert the encoder rather than synthesize more details, and becomes increasingly costly at megapixel scale. This drawback calls for a more expressive and efficient decoding paradigm. Motivated by recent progress in scalable pixel-space diffusion, we introduce PiD, a Pixel diffusion Decoder that reformulates latent decoding as conditional pixel diffusion, unifying decoding and upsampling into one generative module. By denoising directly in high-resolution pixel space, PiD synthesizes $4\times$ and even $8\times$ upscaled images with low latency. For latent conditioning, a lightweight sigma-aware adapter injects noise-corrupted latents into the pixel diffusion backbone, enabling PiD to decode partially denoised latents and terminate the latent diffusion process early. To further improve efficiency, we distill the model using DMD2, reducing inference to just 4 steps. PiD applies to both conventional VAE latents and semantic latents (e.g., SigLIP, DINOv2) used in recent RAE-based models. PiD decodes latents of $512 \times 512$ images into $2048 \times 2048$ pixels in under 1 second with 13 GB peak memory on a consumer RTX 5090, and as fast as 210 ms on a GB200 GPU, about $6\times$ faster than cascaded diffusion-based super-resolution pipelines with better visual fidelity.
>
---
#### [new 068] Beyond Normal References: Discriminative Few-Shot Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于少样本异常检测任务，解决现有方法仅使用正常样本的问题。提出IDEAL框架，同时利用正常和异常样本，学习通用的异常特征，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.23231](https://arxiv.org/pdf/2605.23231)**

> **作者:** Huan Wang; Jun Shen; Jun Yan; Guansong Pang
>
> **备注:** 31 pages
>
> **摘要:** This paper considers a practical few-shot anomaly detection (FSAD) setting, termed discriminative FSAD, where a limited number of both normal and anomalous examples are available as references during inference. Existing FSAD methods rely on normal-only references through normality matching, ignoring the discriminative clues in anomalous references, while directly fitting both references can overfit to the seen anomalies. We introduce IDEAL, an intrinsic deviation learning framework that leverages both reference types to learn intrinsic deviation patterns characterizing generalizable abnormality as deviations from normality. IDEAL decomposes the learning process into two novel components: 1) a Normal Variation Eraser to suppress nuisance normal variations that may lead to noisy deviations from normality, thereby highlighting anomaly-relevant deviation representations; 2) an Intrinsic Deviation Encoder to decompose these denoised deviation representations into intrinsic deviation vectors capturing the most discriminative orthogonal deviation directions. At inference, IDEAL scores query-to-normal deviations preserved after projection onto the learned intrinsic deviation vectors, enabling generalization for both seen and unseen anomalies. Extensive experiments on eight real-world datasets show that IDEAL generalizes effectively to unseen anomalies and consistently outperforms existing state-of-the-art FSAD methods. Code and data will be available at \href{this https URL}{this https URL}.
>
---
#### [new 069] CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels
- **分类: cs.CV**

- **简介: 该论文属于长尾分布下的可靠学习任务，解决标签噪声对模型性能的影响。提出CARE框架，通过多源监督和自适应共识机制，提升模型在长尾数据上的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.23254](https://arxiv.org/pdf/2605.23254)**

> **作者:** Mengke Li; Haiquan Ling; Lihao Chen; Yang Lu; Yiqun Zhang; Hui Huang
>
> **备注:** poster in ICML 2026
>
> **摘要:** Learning from real-world data is frequently hindered by the compound challenge of long-tailed class distributions and noisy annotations. Existing methods partially address these issues but typically ignore the non-uniform impact of label noise across classes, resulting in ineffective correction for tail classes and over-regularization for head classes. To address this issue, we propose Class-Adaptive Rectification with Experts (CARE), a parameter-efficient framework that leverages three complementary supervision sources from vision-language models (VLM): observed noisy labels, VLM text embeddings, and visual features. CARE introduces a class-adaptive expert consensus mechanism that enforces stricter agreement for tail classes and more permissive agreement for head classes based on class frequency. By aggregating high-confidence predictions across these sources, CARE filters unreliable signals and recalibrates class distributions, yielding more reliable rectification under long-tailed distributions. Extensive experiments on both synthetic and real-world benchmarks demonstrate that CARE consistently outperforms state-of-the-art methods, achieving up to 3.0\% performance gains. The source code is available at this https URL.
>
---
#### [new 070] Not Too Generative, Not Too Discriminative: The Human Alignment Sweet Spot
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉表示是否由判别或生成学习更好解释。通过JEM模型隔离目标影响，发现中间点最接近人类视觉。任务是理解人类对齐的最优学习目标。**

- **链接: [https://arxiv.org/pdf/2605.23819](https://arxiv.org/pdf/2605.23819)**

> **作者:** Jorge Chang Ortega; Bastien Le Lan; Thomas Serre; Victor Boutin
>
> **摘要:** A central question in computational vision is whether human-like visual representations are better explained by discriminative or generative learning. Existing comparisons, however, often confound the learning objective with architecture, scale, and training data, leaving open whether the objective itself drives alignment. We address this confound using Joint Energy-Based Models (JEMs), which interpolate continuously between discriminative and generative training within a fixed architecture. By varying a single mixing coefficient, we isolate the effect of the learning objective and evaluate the resulting models across six human-alignment benchmarks spanning perceptual similarity, gloss perception, human response uncertainty, robustness, shape-texture cue conflict, and diagnostic feature attribution. Across this diverse suite, human alignment is consistently maximized at intermediate points of the generative-discriminative continuum, rather than at either endpoint. Hybrid JEMs combine the categorical structure induced by discriminative learning with the sensitivity to input structure induced by generative learning, yielding more human-like behavior across multiple levels of vision. These results suggest that the generative-discriminative dichotomy is the wrong axis for understanding human-aligned vision: alignment emerges not from choosing one objective over the other, but from balancing both.
>
---
#### [new 071] Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像超分辨率任务，旨在解决生成先验导致的失真问题。通过引入Sobolev几何对齐方法，提升图像结构保真度与频谱一致性。**

- **链接: [https://arxiv.org/pdf/2605.23264](https://arxiv.org/pdf/2605.23264)**

> **作者:** Hongbo Wang; Huaibo Huang; Pin Wang; Jinhua Hao; Chao Zhou; Ran He
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Generative priors in Image Super-Resolution (SR) often compromise faithful restoration, we attribute this limitation to a fundamental spectral misalignment between isotropic objectives and the intrinsic natural image manifold. While Direct Preference Optimization offers a path to alignment, its reliance on spectrally flat Gaussian noise fails to distinguish authentic high-frequency details from hallucinations. To bridge this geometric gap, we propose ASASR, a theoretically grounded framework that recasts the generative flow into a Sobolev-induced Riemannian geometry by explicitly coloring the noise transition kernel to mirror natural spectral decay. Driving this geometric alignment, we integrate a parametric adversary grounded in the Riesz Representation Theorem, which synthesizes targeted negative samples equivalent to worst-case Sobolev gradients to direct optimization along the tangent space of plausible structural failures. Extensive evaluations demonstrate that ASASR outperforms leading generative baselines, particularly in preserving spectral consistency and structural fidelity, offering a robust solution that effectively mitigates artifacts.
>
---
#### [new 072] DRIVESPATIAL: A Benchmark for Spatiotemporal Intelligence in VLMs for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出DriveSpatial基准，用于评估视觉语言模型在自动驾驶中的时空智能。解决现有基准不足的问题，通过多视角、动态场景的问答对测试模型的场景构建、关系理解、时间推理等能力。**

- **链接: [https://arxiv.org/pdf/2605.23176](https://arxiv.org/pdf/2605.23176)**

> **作者:** Hao Vo; Khoa Vo; Phu Loc Nguyen; Sieu Tran; Duc Minh Nguyen; Ngo Xuan Cuong; Gladys Gawugah; Sreevenkata Anjani Tishita Godavarthi; Chase Rainwater; Nghi D. Q. Bui; Anh Nguyen; Duy Minh Ho Nguyen; Ngan Le
>
> **摘要:** Spatiotemporal intelligence in autonomous driving (AD) requires an agent to integrate multi-view observations into a coherent scene representation, maintain object continuity across viewpoints and time, and reason about spatial relations, interactions, and future dynamics. However, existing AD vision-language benchmarks largely focus on single-view, static, ego-centric, or single-source question answering, leaving it unclear whether current Vision-Language Models (VLMs) can truly construct and reason over dynamic driving scenes. We introduce DriveSpatial, a benchmark of 15.6K human-verified QA pairs across 20 tasks from five large-scale AD datasets. DriveSpatial evaluates four abilities: Cognitive Scene Construction, Multi-view Relational Understanding, Temporal Reasoning, and Generalization. Unlike prior benchmarks, DriveSpatial is generated from a dynamic multi-relational scene graph that encodes object states, spatial relations, interactions, camera visibility, and temporal correspondences, enabling QA pairs that enforce genuine cross-view and spatiotemporal reasoning. Evaluating 15 representative VLMs reveals a substantial human-model gap: the strongest model trails humans by 28.4 points, with Cognitive Scene Construction emerging as the key bottleneck. Further diagnostics show that language-only prompting is insufficient, while explicit BEV grounding consistently improves performance. These results suggest that current VLMs lack the scene-construction ability needed for reliable spatiotemporal driving intelligence. DriveSpatial and its construction pipeline will be released to support future research.
>
---
#### [new 073] ExpOS: Explainable Open-Surgery Skills Assessment Using 3D Hand Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出ExpOS，用于开放性手术技能的可解释评估。解决传统依赖专家评价的局限性，通过分析运动数据自动评估技能并提供反馈。**

- **链接: [https://arxiv.org/pdf/2605.23653](https://arxiv.org/pdf/2605.23653)**

> **作者:** Roi Papo; Idan Smoller; Shlomi Laufer
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Timely and transparent feedback is essential for effective surgical training, yet current assessment remains dependent on expert observation, limiting scalability and opportunities for autonomous practice. We present ExpOS, an explainable framework for data-driven assessment of open-surgery skills designed to enable automatic, feedback-oriented evaluation. Rather than relying on expert-defined metrics, ExpOS learns discriminative temporal patterns directly from motion data and identifies the segments and behaviors most predictive of skill level. We trained and evaluated the method on 221 videos of medical students performing three open-surgery tasks. Hand poses and tool detections were extracted from each frame to derive kinematic descriptors and global motion statistics. Spatiotemporal hand-tool dynamics were modeled using a temporal convolutional backbone with attention-based pooling to generate frame-level importance maps. These representations were fused with global motion statistics to predict skill level and to provide interpretable feedback. ExpOS provides multi-level explainability by identifying when informative events occur through attention weights and which motion characteristics most influence predictions through global feature analysis. Across tasks, the framework achieved strong correlation with expert ratings, with best performance on fascial closure (r = 0.778, R2 = 0.74). These results demonstrate that combining weakly-supervised temporal importance learning with interpretable motion statistics enables scalable and actionable surgical skill assessment.
>
---
#### [new 074] PhenoYieldNet: Learning Crop-Aware Phenological Responses for Multi-Crop Yield Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多作物产量预测任务，旨在解决现有方法难以泛化到不同作物的问题。通过构建作物感知的时序解码器，学习作物特有的物候响应，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2605.23478](https://arxiv.org/pdf/2605.23478)**

> **作者:** Yu Luo; Xiaogang Zhu; Shan Zeng; Wei Xiang; Thomas Francis Bishop; Zhiyong Wang; Kun Hu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Accurate crop yield prediction is crucial for sustainable agriculture and global food security. While existing methods are predominantly developed for single-crop prediction, they often struggle to generalize across diverse crop types, without addressing the unique crop phenological responses that are dynamically modulated by complex weather patterns. In this paper, we propose PhenoYieldNet, a multi-crop yield prediction framework that learns crop-specific phenology by explicitly modeling their responses with temporal drivers. Specifically, we develop a crop-aware temporal decoder consisting of a Crop Phenology Bank (CPB) and a Crop Phenology Attention (CPA) module. The CPB integrates a set of learnable embeddings, which leverage a query to guide the CPA module to learn the most relevant phenology patterns for the specific crop. And the CPA module explicitly captures multi-scale trend and variation components to construct temporal contexts, enabling the model to dynamically adjust the attention across different phenological stages. To learn robust and generalizable features for multi-crop prediction, the encoder is initialized with a pre-trained foundation model, and further adapted via a self-supervised Temporal Contrastive Adaptation strategy to align with agricultural temporal dynamics. Extensive experiments conducted on multi-crop datasets indicate that our proposed method significantly outperforms state-of-the-art methods, exhibiting strong generalization capabilities across different regions and crops.
>
---
#### [new 075] Occlusion-Aware Physics-Semantic Keyframe Selection for Robust Video Editing
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，解决 occlusion 和运动变化下的编辑不一致问题。提出物理语义关键帧选择框架，提升编辑的鲁棒性和一致性。**

- **链接: [https://arxiv.org/pdf/2605.23192](https://arxiv.org/pdf/2605.23192)**

> **作者:** Lin Liu; Zhihan Xiao; Haohang Xu; Rong Cong; Zhibo Zhang; Xiaopeng Zhang; Qi Tian
>
> **摘要:** Video editing has recently achieved remarkable progress with diffusion-based generative models, enabling diverse object-level manipulations from natural language instructions. However, existing methods often struggle under occlusion, viewpoint changes, and fast object motion, where unreliable visual observations lead to inaccurate localization, temporal flickering, and inconsistent edits. In this work, we identify the absence of reliable visual anchors as a fundamental bottleneck in occlusion-robust video editing. To address this issue, we propose an occlusion-aware physics-semantic keyframe selection framework that automatically identifies an optimal anchor frame for downstream editing. Specifically, our method evaluates candidate frames from three complementary perspectives: structural completeness for avoiding truncated observations, cycle-consistent tracking stability for measuring physical reliability, and vision-language-based attribute visibility for ensuring semantic clarity. The selected keyframe is then propagated through bidirectional tracking to generate dense spatiotemporal masks, which are used as auxiliary supervision for a diffusion-based video editing backbone. By transforming occlusion handling from explicit reconstruction into reliable anchor selection, our framework enables precise and temporally consistent editing without requiring manual annotations. Extensive experiments on challenging video editing benchmarks demonstrate the effectiveness and high-quality performance of our method.
>
---
#### [new 076] Joint Target-Less Intrinsic and Extrinsic Camera-LiDAR Calibration using Deep Point Correspondences
- **分类: cs.CV**

- **简介: 该论文属于多模态传感器标定任务，解决无目标的相机-LiDAR联合标定问题。通过深度点对应，同时估计相机内参和外参，提升标定精度。**

- **链接: [https://arxiv.org/pdf/2605.23397](https://arxiv.org/pdf/2605.23397)**

> **作者:** Simon Bultmann; Daniele Cattaneo; Abhinav Valada
>
> **备注:** presented at 2nd German Robotics Conference (GRC)
>
> **摘要:** Accurate camera-LiDAR calibration is a prerequisite for robust multi-modal perception in robotics. Recent target-less approaches based on deep point correspondences achieve remarkable performance for extrinsic calibration but assume rectified images with known intrinsics. In this work, we overcome this limitation and present the first fully target-less pipeline that jointly estimates camera intrinsics (pinhole model with radial-tangential distortion) and camera-LiDAR extrinsics with deep pixel-point correspondences. Our approach extends deep correspondence-based calibration by (i) automatic intrinsic initialization via structure-from-motion, (ii) generalizing camera-LiDAR matching to raw images with unknown intrinsics including distortion, and (iii) tightly coupling correspondence estimation with joint nonlinear optimization over both intrinsics and extrinsics. We evaluate our method on the KITTI dataset with unseen camera-LiDAR pairs and demonstrate that joint calibration achieves improved extrinsic accuracy while additionally recovering accurate intrinsics.
>
---
#### [new 077] Improved Vision-to-Chart Buoy Association with Learned World-to-Image Projection
- **分类: cs.CV**

- **简介: 该论文属于视觉-图表目标关联任务，解决从世界坐标到图像的几何投影问题。通过引入额外MLP预测浮标水线点，提升关联精度。**

- **链接: [https://arxiv.org/pdf/2605.22942](https://arxiv.org/pdf/2605.22942)**

> **作者:** Borja Carrillo-Perez
>
> **备注:** 5 pages, 3 figures. Technical report for the MaCVi 2026 Vision-to-Chart Data Association Challenge at the CVPR 2026 Workshop; 2nd place submission. Code: this https URL
>
> **摘要:** This report presents a lightweight modification to the DETR-based fusion transformer baseline for the MaCVi 2026 Vision-to-Chart data association challenge. The challenge baseline decoder receives per-buoy queries encoding world-space distance and bearing, forcing the transformer to implicitly learn the complex geometric projection from world coordinates to image pixels. Instead, this work trains an additional dedicated MLP, QueryMLP, to explicitly predict the buoy's waterline contact point in the image from chart measurements and IMU orientation data. The predicted pixel coordinates are appended to the baseline decoder query vector, providing a direct spatial prior per buoy and reducing the geometric reasoning burden on the transformer decoder. On the challenge leaderboard, the presented approach achieves an Overall score of 0.7386, with F1 = 0.8055 and mIoU = 0.6718, on the held-out test set, placing second among all submissions.
>
---
#### [new 078] Vision Transformers Need Better Token Interaction
- **分类: cs.CV**

- **简介: 该论文属于视觉Transformer任务，旨在解决密集预测中特征退化问题。通过引入稀疏注意力机制，提升token交互的准确性，改善分割性能。**

- **链接: [https://arxiv.org/pdf/2605.23868](https://arxiv.org/pdf/2605.23868)**

> **作者:** Linxiang Su
>
> **备注:** 7 pages
>
> **摘要:** Vision Transformers (ViTs) can learn strong image-level representations while their patch representations become less effective for dense prediction during prolonged training. We revisit this dense degradation phenomenon and argue that it is not fully explained by high-norm artifacts alone. Instead, we characterize \emph{semantic diffusion}: an optimization shortcut in which global semantic information spreads through patch tokens beyond what is locally justified. Our analysis shows that dense representation quality is not captured by locality alone: shallow features can remain better aligned with foreground regions yet underperform deeper features, and \texttt{[CLS]} features remain complementary for dense prediction. These observations suggest that the goal should not be to remove global context, but to make token interactions more selective. We therefore study sparse attention as a minimal intervention, replacing softmax attention with entmax-1.5 while preserving global token connectivity. On DINOv1 ViT-S/16 trained for 200 epochs on ImageNet-1K, this change preserves ImageNet linear probing accuracy and substantially improves semantic segmentation performance: VOC mIoU increases from 42.80 to 48.78, ADE20K from 19.85 to 21.97, and Cityscapes from 36.79 to 37.87. These results suggest that selective token mixing is a simple and effective bias for improving dense ViT representations.
>
---
#### [new 079] Multimodal Distribution Matching for Vision-Language Dataset Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言多模态任务，解决数据蒸馏中如何高效保留跨模态对齐与语义的问题。提出MDM框架，在数据、模型和损失层面进行优化，提升蒸馏效率与泛化性。**

- **链接: [https://arxiv.org/pdf/2605.23482](https://arxiv.org/pdf/2605.23482)**

> **作者:** Jongoh Jeong; Hoyong Kwon; Minseok Kim; Kuk-Jin Yoon
>
> **备注:** Accepted for publication at CVPR 2026. Project Page: this https URL
>
> **摘要:** Dataset distillation compresses large training sets into compact synthetic datasets while preserving downstream performance. As modern systems increasingly operate on paired vision-language inputs, multimodal distillation must preserve representation quality and cross-modal alignment under tight compute and memory budgets, yet prior methods often require heavy computes and overlook their correlations. To address this, we present Multimodal Distribution Matching (MDM), a geometry-aware framework for efficient and generalizable multimodal distillation. Specifically, MDM integrates complementary components at the data, model, and loss levels. At the data level, it initializes synthetic image-text pairs by sampling from clusters in the joint embedding space. At the model level, it forms a mixed teacher by interpolating independently fine-tuned models in weight space according to their angular deviation from the pretrained anchor. At the loss level, it matches joint distributions on the unit hypersphere using a geometry-aware matching objective that exploits the joint features in the cross-modal agreement and discrepancy directions along with symmetric contrastive learning. Across image-text retrieval benchmarks with cross-architecture evaluation, MDM yields compact synthetic sets that preserve multimodal semantics, substantially reduce distillation cost, and remain robust across architectures.
>
---
#### [new 080] GazeBehavior Annotation Toolkit (GBAT): AI-powered toolkit for automatic annotation of egocentric eye-tracking and video data of child-caregiver interaction
- **分类: cs.CV; cs.CE; cs.HC; cs.SE; q-bio.NC**

- **简介: 该论文提出GBAT工具，解决儿童与照料者互动视频中注意力标注效率低的问题，通过深度学习实现自动同步、标注和动作分类。**

- **链接: [https://arxiv.org/pdf/2605.22962](https://arxiv.org/pdf/2605.22962)**

> **作者:** Iba Baig; Kevin Li; Yanbin Xu; Seiji Cattelain; Marie Hallo; Hayato Ono; Sho Tsuji; Ming Bo Cai
>
> **备注:** submitted to IEEE International Conference on Development and Learning (ICDL), 2026
>
> **摘要:** Video recordings of child-caregiver interactions enable investigation of attentional dynamics during naturalistic behavior. Such multimodal recording also allows researchers to examine how attention interacts with action and language use in real time. However, manual annotation of such data is time-consuming. Here, we introduce GazeBehavior Annotation Toolkit, a deep-learning-based toolkit designed to facilitate three key processes in data preprocessing and feature extraction: post-hoc synchronization across multiple videos, semi-automatic annotation of gaze target categories, and categorization of participants' poses and hand actions. This toolkit improves the efficiency and scalability of feature extraction from human egocentric eye-tracking and video data. Such improvement is critical in supporting large-scale and longitudinal investigations of attentional dynamics and naturalistic behavior in human early development.
>
---
#### [new 081] EM-Vid: Training-Free Entity-Centric Memory for Efficient and Consistent Multi-Shot Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多镜头视频生成任务，旨在解决实体一致性与提示忠实度问题。提出实体中心记忆机制，提升效率与控制精度。**

- **链接: [https://arxiv.org/pdf/2605.23610](https://arxiv.org/pdf/2605.23610)**

> **作者:** Jente Vandersanden; Matheus Gadelha; Chun-Hao P. Huang; Hyeonho Jeong; Yulia Gryaditskaya
>
> **摘要:** Multi-shot video generation requires maintaining a consistent appearance of recurring entities across shots while remaining faithful to shot-specific text prompts. Recent autoregressive methods reuse previously generated frames as memory. However, full-frame storage entangles persistent entity information with transient scene context, leading to irrelevant information leakage and high computational cost. We propose an entity-centric memory in the form of an entity-indexed bank of latent patches. We introduce sparse token conditioning compatible with pretrained models, restricting self-attention to entity-relevant tokens and reducing computational cost. To support this, we introduce a structured multi-shot script format. We additionally propose a budgeted memory update strategy to maintain a compact, evolving memory. Finally, we equip the entity representation with a noise-injection mechanism that enables fine-grained appearance control, preventing leakage of irrelevant information. Our method improves prompt adherence and efficiency while preserving subject consistency.
>
---
#### [new 082] Geo-Align: Video Generation Alignment via Metric Geometry Reward
- **分类: cs.CV**

- **简介: 该论文提出Geo-Align，解决真实视频生成中的相机控制问题，通过强化学习优化模型，提升视频的物理尺度和轨迹准确性。**

- **链接: [https://arxiv.org/pdf/2605.23903](https://arxiv.org/pdf/2605.23903)**

> **作者:** Zizun Li; Haoyu Guo; Runzhe Teng; Chunhua Shen; Tong He
>
> **摘要:** Camera-controlled video generation has achieved remarkable progress in recent years. However, existing video-to-video re-rendering methods primarily rely on Supervised Fine-Tuning using synthetic datasets. At present, there is an extreme scarcity of synchronized, multi-view real-world video data. Consequently, the prevailing paradigm often exhibits limited generalization when processing out-of-distribution real-world videos, with models struggling to accurately adhere to physical scales and camera trajectories. To bridge this gap, we propose Geo-Align, the first Reinforcement Learning framework specifically designed for camera-controlled video re-rendering. Built upon a pretrained model, we optimize the model through a scale-aware perceptual reward mechanism. Specifically, we introduce a metric 3D estimator to extract precise camera trajectories from generated videos, explicitly penalizing deviations in rotation and translation. Furthermore, we meticulously designed a data pipeline strategy based on real-world conditioning videos and target camera trajectories derived from synthetic data, eliminating the reliance on paired data. Extensive experiments demonstrate that Geo-Align consistently outperforms existing supervised learning baselines in both precise camera controllability and visual fidelity, indicating the effectiveness of our method.
>
---
#### [new 083] GFSR: Geometric Fidelity and Spatial Refinement for Reliable Lane Detection
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的车道检测任务，旨在解决现有方法在复杂场景下几何质量与分类置信度不匹配及采样点相关性弱的问题。提出GFSR框架，提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.23327](https://arxiv.org/pdf/2605.23327)**

> **作者:** Tiancheng Wang; Zhaolu Ding; Richeng Xu; Tianhui Zheng; Hui Liu; Hanyu Xuan; Zhiliang Wu; Guanghui Yue
>
> **备注:** Submitted to IEEE Transactions on Intelligent Transportation Systems (under review). 12 pages, 6 figures
>
> **摘要:** Lane detection stands as a crucial perception task in autonomous driving and advanced driver assistance systems. However, existing methods still degrade in complex real scenarios due to two major limitations. First, classification confidence only characterizes the categorical existence of lane candidates and has no strong correlation with geometric quality. If threshold filtering and NMS are conducted merely based on this confidence, the model tends to retain lane priors with high confidence while eliminating those with lower confidence but superior geometric representation. Secondly, existing regression modules weaken correlations among sampling points, hindering fine-grained optimization of distant, high-curvature and complex-topology lanes and causing underfitting. To address these issues, we propose Geometric Fidelity and Spatial Refinement (GFSR), a framework consisting of LaneIoU-guided Confidence Calibration (LCC) and Adaptive Gated Location Refinement (AGLR). Specifically, LCC adopts LaneIoU as soft supervision to explicitly estimate geometric fidelity of lane priors, which is further fused with classification confidence to construct the collaborative reliability index (CRI). This index guides threshold filtering and NMS, effectively retaining lane priors with high classification confidence and favorable geometric quality. Meanwhile, cooperating with regression heads in each refinement stage, AGLR predicts sampling point lateral offsets and adopts a gating mechanism to adaptively regulate correction magnitude, strengthen inter-point correlations and boost model adaptability as well as robustness toward complex lane scenarios. Extensive experiments on CULane and CurveLanes demonstrate that our GFSR achieves state-of-the-art performance on CULane, with F1@50 and F1@75 scores of 81.46% and 65.01%, and reaches 87.35% F1@50 on CurveLanes.
>
---
#### [new 084] PixIE: Prompted Pixel-Space Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光照图像增强任务，旨在解决噪声、对比度损失和语义模糊问题。提出PixIE框架，结合语义提示和高效特征处理，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2605.23531](https://arxiv.org/pdf/2605.23531)**

> **作者:** Ruirui Lin; Guoxi Huang; David Bull; Nantheera Anantrasirichai
>
> **摘要:** Low-light images exhibit severe noise, contrast loss, and semantic ambiguity, making enhancement a joint problem of denoising and detail recovery. We propose PixIE, a feed-forward pixel-space LLIE framework semantically-prompted by a vision foundation model. PixIE first performs a cross-scale denoising to suppress noise and preserve structure, then refines details with DINO-Prompted Pixel Blocks (DPPB) that inject intermediate DINOv3 features via patch-conditioned, spatially continuous per-pixel modulation. We introduce a Spatial-Channel Compaction (SCC), which folds features into a compact spatial grid and compresses in the channel dimension, so pixel-attention is computed efficiently with bounded cost across scales. We further propose Multi-Receptive-Field Pixel Embedding (MRPE) to provide neighborhood-aware pixel representations before semantic prompting, improving robustness to signal-dependent noise beyond point-wise embeddings. Experiments on LLIE benchmarks show that PixIE improves the average PSNR by 1.9-15.0% over recent state-of-the-art methods and reduces LPIPS by 8.5-44.4%. Qualitative comparisons further demonstrate that PixIE recovers sharper details and more stable textures, resulting in improved reconstruction fidelity and perceptual quality.
>
---
#### [new 085] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文提出CVSearch，解决高分辨率图像感知中视觉搜索的覆盖与效率矛盾，通过自适应策略提升多模态大模型性能。**

- **链接: [https://arxiv.org/pdf/2605.23655](https://arxiv.org/pdf/2605.23655)**

> **作者:** Liupeng Li; Haoqian Kang; Zhenyu Lu; Jinpeng Wang; Bin Chen; Ke Chen; Yaowei Wang
>
> **备注:** Accepted by ICML 2026. 22 pages, 12 figures, 7 tables
>
> **摘要:** High-resolution (HR) image perception presents a key bottleneck for multimodal large language models (MLLMs). While visual search offers a promising solution, existing methods struggle with the trade-off between coverage and efficiency. Visual expert-assisted search is efficient but prone to blind spots when proposals fail, whereas scan-based search guarantees coverage at the cost of computational redundancy and semantic fragmentation. To address this dilemma, we introduce CVSearch, a training-free adaptive framework that dynamically schedules search strategies via an Assess-then-Search workflow. Specifically, CVSearch first invokes expert-assisted search when global information is insufficient, and only triggers a novel semantic-aware scanning mechanism upon failure. Distinct from rigid grid partitioning, this efficient scanning paradigm incorporates Semantic Guided Adaptive Patching to decompose images into semantically consistent regions, effectively mitigating object fragmentation. Furthermore, we devise a Dynamic Bottom-Up Search strategy driven by a Visual Complexity prior to enable efficient and precise iterative exploration of local details. Extensive experiments on HR benchmarks demonstrate that CVSearch achieves state-of-the-art accuracy while substantially improving search efficiency. Code is released at this https URL.
>
---
#### [new 086] CoReVAD: A Contextual Reasoning Framework for Training-Free Video Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频异常检测任务，旨在解决传统方法依赖训练、解释性差的问题。提出CoReVAD框架，无需训练即可生成异常评分和可解释描述。**

- **链接: [https://arxiv.org/pdf/2605.23116](https://arxiv.org/pdf/2605.23116)**

> **作者:** Hyeongmuk Lim; Youngbum Hur
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Existing Video Anomaly Detection (VAD) methods typically rely on task-specific training, leading to strong domain dependency and high training costs. Moreover, most existing methods output only scalar anomaly scores, providing limited insight into why specific events are considered abnormal. Recent advances in Vision-Language Models (VLMs) have enabled both anomaly detection and human-interpretable reasoning. However, many VLM-based approaches still require additional training steps (e.g., instruction tuning or verbalized learning) or external Large Language Models (LLMs), incurring further training costs and inference overhead. To address these challenges, we propose CoReVAD, a contextual reasoning framework for training-free video anomaly detection that operates with a single frozen VLM. CoReVAD directly generates anomaly scores and temporal descriptions from the VLM. To mitigate noise in generative outputs, we introduce a Local Response Cleaning (LRC) module based on local vision-text alignment. Furthermore, global temporal context and progression are incorporated through softmax-based refinement, Gaussian smoothing, and position weighting. Experiments on UCF-Crime and XD-Violence demonstrate that CoReVAD achieves competitive performance among training-free methods while providing reliable and interpretable explanations. Our official code is available at: this https URL
>
---
#### [new 087] GenRecon: Bridging Generative Priors for Multi-View 3D Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，解决多视角图像生成高保真3D场景的问题。通过结合生成先验与场景重建，提升重建精度与一致性。**

- **链接: [https://arxiv.org/pdf/2605.23888](https://arxiv.org/pdf/2605.23888)**

> **作者:** Katharina Schmid; Nicolas von Lützow; Jozef Hladký; Angela Dai; Matthias Nießner
>
> **备注:** Project page: this https URL
>
> **摘要:** We introduce a new approach to high-fidelity 3D scene reconstruction from multi-view RGB images that tightly couples reconstruction with a strong generative 3D prior. We cast scene reconstruction as conditional 3D generation over a set of spatially-localized, overlapping chunks that together tile the scene, scaling generation to large scene extents. Crucially, we inherit the fidelity and completeness of state-of-the-art generative shape models -- we use Trellis.2 as an example -- which we generalize to the scene level. To this end, we propose a projection-based conditioning mechanism that lifts posed multi-view image features into a coherent 3D representation aligned with the generative model, independent of view ordering and spatially anchored to the scene, yielding high-fidelity, multi-view consistent generated geometry. This enables lifting the strong object-level prior of Trellis.2 to multi-view, scene-scale generation, producing faithful, editable PBR mesh reconstructions of indoor environments. As a result, we obtain high-fidelity results that outperform cutting-edge reconstruction methods by 16%.
>
---
#### [new 088] LaMo: Self-Supervised Latent Motion Priors for Physical Realism in Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在提升视频的物理真实性。通过自监督学习从未标注视频中提取运动信息，提出LaMo模型增强运动一致性。**

- **链接: [https://arxiv.org/pdf/2605.23878](https://arxiv.org/pdf/2605.23878)**

> **作者:** Bo Jiang; Depu Meng; Yihan Hu; Yichen Xie; Tianshuo Xu; Wei Zhan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Modern video generators produce visually compelling clips but still struggle with physical and motion consistency, limiting their use as reliable world simulators. Existing remedies often rely on external simulators, teacher models, or curated physics-focused data. We explore a complementary self-supervised direction: extracting motion cues from the unlabeled videos already used to train video diffusion models. We propose LaMo, which formulates a latent motion prior over frame-to-frame latent changes conditioned on the current latent and prompt. This prior is exposed through two lightweight readouts: a macro motion drift used during training as a Motion Drift Loss, and a learned micro motion field used during sampling as Motion Prior Guidance. Both components are plug-and-play with existing video diffusion backbones, requiring no architectural or I/O changes. On VideoPhy and VideoPhy2, LaMo improves CogVideoX backbones and outperforms recent physics-aware baselines that use external supervision. On VBench, it preserves overall generation quality while improving motion-related dimensions. These results suggest that unlabeled video contains useful motion supervision for improving physical fidelity in modern video diffusion models.
>
---
#### [new 089] U-CESE: Unified Clip-based Event Search Engine for AI Challenge HCMC 2025
- **分类: cs.CV**

- **简介: 该论文提出U-CESE，解决多模态视频事件检索任务。针对大规模视频数据中的时间、空间和多模态信息复杂问题，整合模块、优化剪辑算法，并引入关键帧提取和时序描述方法，提升检索效率与一致性。**

- **链接: [https://arxiv.org/pdf/2605.23274](https://arxiv.org/pdf/2605.23274)**

> **作者:** Duc-Nhuan Le; Hoang-Phuc Nguyen; Thanh-Duy Lam; Minh-Nhut Dang; Minh-Hoang Le
>
> **备注:** Accepted for publication in the Proceedings of the 14th International Symposium on Information and Communication Technology (SOICT 2025)
>
> **摘要:** Retrieving events from large-scale video datasets is challenging due to complex temporal, spatial, and multimodal information. This paper presents U-CESE, our solution for the AI Challenge HCMC 2025, a Unified Clip-based Event Search Engine for multimodal event retrieval across diverse video sources. Building on CESE, U-CESE integrates its three modules into a single cohesive framework, ensuring consistent processing and retrieval across query types. A core component is the Unified Clipping Algorithm, which merges separate clipping algorithms into one efficient pipeline. To handle large-scale data, we propose DAKE, a lightweight, training-free keyframe extraction method using JPEG file size variations to identify significant scene changes. Finally, we introduce ReCap, a temporally consistent captioning framework inspired by Recurrent Neural Network, generating detailed and context-aware textual descriptions. Experiments show that U-CESE delivers robust, consistent, and efficient performance in large-scale multimodal event retrieval.
>
---
#### [new 090] Spatio-Temporal Similarity Volume Aggregation for Open-Vocabulary Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于开放词汇动作识别任务，旨在解决传统方法丢失局部时空信息的问题。提出SimVA框架，构建4D相似性体积，提升动作识别性能。**

- **链接: [https://arxiv.org/pdf/2605.23288](https://arxiv.org/pdf/2605.23288)**

> **作者:** Yerim So; Jiyeong Kim; Jiwon Yoon; Dongbo Min
>
> **摘要:** Recent Open-Vocabulary Action Recognition (OVAR) methods typically aggregate visual features into a global representation before computing text alignment, a process that obscures local patch information and fine-grained spatio-temporal cues. We propose Similarity Volume Aggregation (SimVA), a framework that constructs a dense 4D spatio-temporal similarity volume from patch-level visual-text similarities. SimVA constructs a spatio-temporal similarity volume over local video tokens and action classes, and employs class sampling to ensure similarity aggregation scalable to large vocabularies. The similarity volume is refined by spatial aggregation, which contextualizes local similarity patterns to improve intra-frame consistency. Motion-aware modulation further injects inter-frame variation cues, highlighting dynamically changing regions. Mamba-based temporal aggregation then models the evolution of class-conditioned similarity patterns across frames. By maintaining dense visual-text correspondence, SimVA effectively transfers CLIP to video action recognition, achieving competitive performance across zero-shot, few-shot, and base-to-novel benchmarks.
>
---
#### [new 091] SLIP-RS: Structured-Attribute Language-Image Pre-Training for Remote Sensing Object Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感目标检测任务，解决数据稀缺下细粒度表示学习问题。提出SLIP-RS，通过结构化属性解耦与对比学习，提升检测性能与跨域泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.23144](https://arxiv.org/pdf/2605.23144)**

> **作者:** Chenxu Wang; Yuxuan Li; Yunheng Li; Xiang Li; Jingyuan Xia; Qibin Hou
>
> **摘要:** Existing language-image pre-training for remote sensing object detection is constrained by Monolithic Label Learning, which relies on exhaustively enumerating open-set categories via black-box data to acquire fine-grained representations, creating a dependency incompatible with the domain's inherent data scarcity. To transcend this bottleneck, we propose SLIP-RS, establishing a Structured-Attribute Decoupling Paradigm that maps the open-ended category space into a finite, physically meaningful attribute space, unlocking fine-grained discriminability via explicit structural logic. This paradigm is realized via two technical pillars: (1) Structured-Attribute Contrastive Learning, which enforces the learning of decoupled intrinsic visual logic via combinatorial attribute augmentation; and (2) Conformal Attribute Reliability Engine, which leverages conformal prediction theory to rigorously distill high-fidelity supervision from noisy sources, yielding RS-Attribute-15M, the largest dataset with over 15 million attribute annotations. Extensive experiments demonstrate that SLIP-RS establishes unprecedented performance in fine-grained detection and cross-domain generalization, validating structured attributes as a vital foundation for remote sensing. Code: this https URL.
>
---
#### [new 092] STAMBRIDGE: Spectral-Temporal Amplitude-aware Mid-Feature Bridge for EEG Visual Decoding
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于EEG视觉解码任务，解决神经信号与视觉空间对齐不稳定的问题。提出STAMBRIDGE框架，通过特征调节和跨模态对齐实现高效解码。**

- **链接: [https://arxiv.org/pdf/2605.23137](https://arxiv.org/pdf/2605.23137)**

> **作者:** Jiahe Meng; Weiming Zeng; Yueyang Li; Bo Chai; Hongjie Yan; Zhiguo Zhang; Wai Ting Siok; Nizhuan Wang
>
> **摘要:** Electroencephalography (EEG) visual decoding remains challenging due to the modality gap between low-SNR neural signals and highly structured vision--language spaces, making direct cross-modal alignment unstable. To address this, we propose STAMBRIDGE, a versatile two-stage framework that sequentially tackles feature conditioning and cross-modal alignment. First, we introduce a Spectral-Temporal Amplitude-aware Modulation (STAM) to extract well-conditioned EEG representations. By replacing hard frequency masking with amplitude-derived soft channel weighting and multi-scale temporal convolutions, STAM explicitly preserves frequency-aware transients while reducing the risk of time-domain ringing artifacts. Building upon these robust neural features, we further introduce a model-agnostic Mid-Feature Semantic Bridge (MFSB) that constructs a regularized intermediate space through directed cross-modal interactions, enabling staged distillation and more stable semantic alignment. Experiments on the THINGS-EEG benchmark show competitive 200-way zero-shot retrieval performance, with 34.50\% Top-1 and 65.95\% Top-5 accuracy. In addition, embeddings learned by STAMBRIDGE produce semantically coherent image reconstructions with a diffusion model, demonstrating robust EEG-to-vision semantic alignment. The code is available at: this https URL.
>
---
#### [new 093] DrawVideo: Generating Long Video from Storyboard Keyframe Sketches
- **分类: cs.GR; cs.AI; cs.CV; cs.MM; eess.IV**

- **简介: 该论文属于视频生成任务，旨在解决长视频生成中控制不足的问题。提出DrawVideo框架，通过草图引导生成结构可控的长视频。**

- **链接: [https://arxiv.org/pdf/2605.23508](https://arxiv.org/pdf/2605.23508)**

> **作者:** Chuanzhi Xu; Huiqi Liang; Bang Shi; Huiming Zhang; Yifan Xiao; Guangcheng Lin; Haodong Chen; Qiang Qu; Zhicheng Lu; Weidong Cai
>
> **备注:** 45 pages, 19 figures
>
> **摘要:** Long video generation requires high-fidelity synthesis, coherent narrative structure, and user control over extended time spans. Existing text-to-video methods often rely on a single long prompt, limiting control over pose, composition, layout, and motion. We propose DrawVideo, a sketch-guided, storyboard-driven framework for controllable long-video generation. DrawVideo decomposes long videos into independently controllable shots, each defined by a black-and-white sketch, an appearance prompt, and a motion prompt. The sketch controls pose and layout, the appearance prompt defines identity, scene, and style, and the motion prompt guides temporal dynamics. DrawVideo follows a hierarchical 'global multi-shot, local single-sketch' strategy: it first generates a structure-aligned reference keyframe, then expands the motion prompt into derivative keyframes representing action states, and finally synthesizes clips between adjacent keyframes to build each shot. We also introduce SketchLongVideo, the first dataset for sketch-guided text-to-long-video generation, constructed from animation videos via shot detection, keyframe extraction, vision-language recognition, prompt decomposition, and sketch conversion. Experiments show that DrawVideo achieves strong structural controllability, appearance consistency, visual stability, and coherent long-video generation.
>
---
#### [new 094] RADAR: Relative Angular Divergence Across Representations
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出RADAR，用于评估基础模型的跨域迁移能力，解决负迁移问题。通过分析表示空间的几何特性，提升跨域任务性能。**

- **链接: [https://arxiv.org/pdf/2605.23028](https://arxiv.org/pdf/2605.23028)**

> **作者:** Xavier Cadet; Mateusz Nowak; Peter Chin
>
> **备注:** 27 pages; 8 figures; 10 tables
>
> **摘要:** Machine learning methods rely on data. However, gathering suitable data can be challenging due to availability constraints, cost, or the need for domain expertise. Expanding datasets with additional sources is a common response to limited data, yet this practice does not always improve downstream performance and can sometimes lead to a loss of performance, known as negative transfer. We propose RADAR, a simple, geometrically grounded metric for estimating cross-domain transferability in foundation models. RADAR analyzes the layer-wise evolution of representations by measuring angular alignments and relative changes in distance along layer-to-layer displacement trajectories, and by comparing empirical distributions of within-domain and cross-domain dynamics. We hypothesize that domain transferability is related to the divergence between these trajectory distributions. We evaluate the metric across multiple modalities, including cross-lingual sentiment classification with text embedding models and cross-domain image classification with foundation vision models. Across several settings, RADAR provides competitive predictive performance relative to existing transferability metrics on several vision and text benchmarks, with particularly strong results when domain transitions are smooth or cleanly separated. Our ablations further suggest that the effectiveness of transferability estimation depends on the geometry of the model's internal representation space, with different modalities favoring different topological formulations.
>
---
#### [new 095] Extending Deep Event Visual Odometry with Sparse Point-Cloud Export
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉里程计任务，旨在提升DEVO系统以输出稀疏点云。通过暴露内部3D结构并转换为点云，实现可视化与后续处理。**

- **链接: [https://arxiv.org/pdf/2605.22890](https://arxiv.org/pdf/2605.22890)**

> **作者:** Alireza Safdari; Sajad Ashraf
>
> **备注:** 9 Pages, 4 figures, 5 tabel
>
> **摘要:** Event cameras are well suited for visual odometry under high-speed motion and challenging lighting conditions due to their low latency, high temporal resolution, and high dynamic range. Deep Event Visual Odometry (DEVO) demonstrated that monocular event-only odometry can achieve strong performance by combining sparse patch tracking, learned patch selection, recurrent correspondence refinement, and differentiable bundle adjustment. In this project, we extend DEVO with a sparse point-cloud export pipeline. Rather than modifying the core odometry formulation, our approach exposes the internal 3D structure already estimated by DEVO and converts it into an explicit point-cloud representation for visualization and further processing. In addition, we implement a practical workflow for data export, format conversion, and point-cloud cleanup. The resulting system preserves the original visual odometry pipeline while enabling sparse geometric scene output. Experiments on the BOARD SLOW sequence show that the exported sparse cloud is locally consistent with EMVS reconstructions, achieving high precision at a 5 cm threshold, while also highlighting the expected limitations in density, completeness, and sensitivity to accumulated odometry noise.
>
---
#### [new 096] GMENet: Generative Mixture of Experts Network for Multi-Center Glioma Diagnosis with Incomplete Imaging Sequences
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于多中心胶质瘤诊断任务，解决因影像序列不完整导致的临床数据浪费和模型泛化能力差的问题。提出GMENet网络，通过生成和融合机制提升诊断效果。**

- **链接: [https://arxiv.org/pdf/2605.23183](https://arxiv.org/pdf/2605.23183)**

> **作者:** Pengfei Song; Fangjin Liu; Wenwen Zeng; Yonghuang Wu; Chengqian Zhao; Feiyu Yin; Xuan Xie; Jinhua Yu
>
> **备注:** IJCAI Accept
>
> **摘要:** Contemporary glioma diagnosis integrates molecular features with histopathology to guide clinical decision-making. However, in clinical settings, divergent imaging protocols result in incomplete MRI sequences, leading to two primary challenges: forcing existing frameworks to discard a large portion of clinical data during training and consequently limiting their clinical applicability. To address these limitations, we propose GMENet, a Generative Mixture of Experts Network for multi-center glioma diagnosis with incomplete imaging sequences. Firstly, we design a Cross-attention-based Gated Generation Module that synthesizes missing sequence features from available sequences via cross-attention and dynamic gating mechanisms, incorporating a cycle-consistency loss to preserve semantic integrity. Secondly, we introduce a Dynamically Weighted Experts Fusion Module that performs mixture-of-experts interaction and confidence-aware fusion over original and synthesized dual-sequence features for multi-task prediction. We evaluate GMENet on a multi-center cohort of 1,241 subjects from four in-house datasets and two public repositories. Experiments show that GMENet expands clinically usable training data by 97\%, relative to complete-sequence-only data. Furthermore, it consistently outperforms state-of-the-art methods trained on complete data, demonstrating improved robustness under cross-center distribution shifts.
>
---
#### [new 097] Cost-Effective Model Evaluation with Meta-Learning
- **分类: cs.LG; cs.AI; cs.CV; cs.ET; cs.PF**

- **简介: 该论文属于模型评估任务，旨在解决无标签数据下模型可靠性验证问题。提出MetaEvaluator框架，通过元学习实现高效、无需标注的模型评估。**

- **链接: [https://arxiv.org/pdf/2605.23595](https://arxiv.org/pdf/2605.23595)**

> **作者:** Trinh Pham; Viet Huynh; Hongzhi Yin; Quoc Viet Hung Nguyen; Thanh Tam Nguyen
>
> **摘要:** The rapid growth of machine learning has produced an ever-expanding ecosystem of models, making it increasingly challenging to verify the reliability of newly released models on unseen, unlabeled data. Conventional evaluation pipelines depend on expensive annotation, repeated fine-tuning, or narrow assumptions that fail to transfer across model families. We present MetaEvaluator, a cost-effective, model-agnostic framework for rapid, label-free assessment of unseen models spanning diverse architectures and modalities. MetaEvaluator leverages meta-learning over a pool of reference models to obtain a transferable initialization, enabling accurate evaluation of new models while amortizing cost across the pool and removing the need for per-model retraining. To the best of our knowledge, this is the first model-agnostic framework capable of evaluating new models on entirely unlabeled datasets. Extensive experiments show that MetaEvaluator produces stable and accurate performance estimates at substantially reduced cost compared to conventional approaches, making scalable benchmarking of emerging models on unlabeled data practical.
>
---
#### [new 098] Efficient Learned Image Compression without Entropy Coding
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决传统方法中熵编码带来的延迟问题。通过引入无熵编码的框架，提升压缩效率与速度。**

- **链接: [https://arxiv.org/pdf/2605.23323](https://arxiv.org/pdf/2605.23323)**

> **作者:** Hao Cao; Wenqi Guo; Zhijin Qin; Jungong Han
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Entropy coding is widely used in typical learned image compression (LIC) that converts latents into a compact bitstream. However, entropy coding is typically sequential and becomes the coding latency bottleneck. To overcome it, we present Entropy-Coding Free Learned Image Compression (EF-LIC), a multi-rate framework that generates compact representation by removing statistical and correlation redundancy with low coding latency. First, we introduce unconstrained vector quantization and prove that its index distribution approaches the maximum-entropy bound, yielding minimal statistical redundancy. Second, we propose a context-conditioned autoregressive transform that directly reparameterizes the latents to reduce inter-dependency. Theoretical analysis shows that EF-LIC can remove correlation redundancy as effectively as typical LIC with entropy coding, leading to comparable compression performance. Experiments show EF-LIC achieves up to 67.86% bitrate reduction over MS-ILLM on Kodak with LPIPS. Ablation studies further show EF-LIC matches the compression performance of its entropy-coding based variant while achieving over $3\times$ faster encoding and $5\times$ faster decoding.
>
---
#### [new 099] Precise: SDE-Consistent Stochastic Sampling for RL Post-Training of Flow-Matching Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于强化学习后训练任务，旨在解决流匹配模型中采样器设计问题，通过提出Precise采样器提升奖励优化效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.23522](https://arxiv.org/pdf/2605.23522)**

> **作者:** Jade Zou; Tao Huang; Weijie Kong; Junzhe Li; Yue Wu; Qi Tian; Jiangfeng Xiong; Jianwei Zhang; Liefeng Bo; Zhao Zhong
>
> **摘要:** Reinforcement learning (RL) has become an effective way to improve prompt alignment and perceptual quality in diffusion and flow-matching generators. A critical step for applying online RL to flow matching is turning the deterministic sampling trajectory into a stochastic policy, typically by replacing the reverse-time Ordinary Differential Equation (ODE) with a Stochastic Differential Equation (SDE). The stochastic sampler, controlling the exploration behavior and denoising dynamics, is thus part of the policy, and its design can significantly affect the reward optimization performance. We break down the sampler design into two interdependent components: choosing the right amount of stochastic exploration, and discretizing the resulting SDE faithfully at the small step counts used in RL. To address the first component, we analyze the inherent tension between exploration and stability in denoising and derive an SDE schedule that balances the two. Turning to the discretization challenge, we use a toy example to show that existing samplers can deviate from the flow-matching process, either by introducing excessive discretization noise or by relying on heuristic rules that do not guarantee convergence to the data distribution. To address these issues, we propose Precise, a new stochastic sampler that balances effective exploration with stability. Crucially, Precise keeps the denoising trajectory SDE-consistent through a novel approximation that freezes the clean-latent posterior mean, resolving the excess noise issue in standard samplers. Extensive experiments demonstrate that this formulation leads to significantly faster and more stable reward optimization via reinforcement learning, achieving state-of-the-art alignment scores (e.g., PickScore, HPSv2.1) while requiring 13.1-53.2% less wall-clock training time to match the best in-domain performance of prior samplers.
>
---
#### [new 100] Semantic-Aware Guided Drone Exploration for Language-Conditioned 3D Indoor Mapping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SAGE系统，用于3D室内环境的语义引导探索，解决未知环境中高效对象发现与覆盖的问题。通过融合CLIP模型提升探索效果。**

- **链接: [https://arxiv.org/pdf/2605.23160](https://arxiv.org/pdf/2605.23160)**

> **作者:** Nitin Vegesna; Avideh Zakhor
>
> **备注:** 10 pages, 6 figures, 4 tables. To be presented at the 2nd 3D-LLM/VLA Workshop at CVPR 2026 (non-archival workshop)
>
> **摘要:** We present Semantic-Aware Guided Exploration, SAGE, a system for open-vocabulary exploration in unknown 3D indoor environments that preserves coverage-oriented behavior while allowing semantic cues to reprioritize frontier selection. Building on the FALCON volumetric explorer, SAGE integrates Contrastive Language-Image Pre-training (CLIP) via four key components: object-centric embedding storage, a temporal cache that projects recent observations onto the free-unknown boundary, object frontiers for high-similarity detections, and a unified semantic-geometric planning cost. This cost function bounds semantic reweighting influence, ensuring frontiers are prioritized without sacrificing total coverage. In Matterport3D-based simulations, SAGE outperforms FALCON and a semantic-only ablation in object discovery across map-query pairs. Compared to Finding Things in the Unknown (FTU), SAGE completes exploration 9.0 to 25.9 times faster across the nine shared map-query pairs, achieving a mean speedup of 13.7. Furthermore, SAGE achieves substantially higher volumetric throughput than FTU. Finally, we deploy SAGE in five real-world flights in two environments on a Modal AI Starling 2 quadrotor with onboard sensing and planning, and offboard CLIP inference. Comparing SAGE and FALCON, we find that while FALCON results in faster exploration and shorter mapping trajectories, SAGE outperforms FALCON in terms of object discovery.
>
---
#### [new 101] Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于图像去模糊任务，解决病理显微镜中空间变化和局部不连续的模糊问题。提出DGNO模型，结合局部性与全局一致性，提升去模糊效果。**

- **链接: [https://arxiv.org/pdf/2605.23282](https://arxiv.org/pdf/2605.23282)**

> **作者:** Shaoqing Duan; Haofei Song; Xintian Mao; Qingli Li; Yan Wang
>
> **备注:** 17 pages, 9 figures. Accepted by ICML 2026
>
> **摘要:** Defocus deblurring in pathological microscopy remains challenging due to the spatially varying and locally discontinuous nature of optical blur induced by a position-dependent integral imaging process. Existing deep learning methods, constrained by shift-invariance assumptions and limited interpretability, are not well suited to such heterogeneous blur patterns. Neural operators provide a principled alternative by modeling defocus formation directly as an integral operator, offering a new perspective on defocus deblurring. However, most existing neural operator architectures for low-level vision rely on globally parameterized kernels that assume smoothness and stationarity, limiting their ability to model heterogeneous and locally discontinuous blur patterns. To address this limitation, we propose the Discontinuous Galerkin Neural Operator (DGNO), which parameterizes the integral kernel using a discontinuous Galerkin formulation with element-local volume operators and interface numerical fluxes. DGNO provides a principled combination of locality, heterogeneity modeling, and global coherence while preserving the underlying physics of optical image formation. Extensive and insightful experiments demonstrate that DGNO surpasses state-of-the-arts, delivering sharper reconstructions, robust handling of spatially varying blur, and scalable high-resolution performance. The code will be released at this https URL.
>
---
#### [new 102] Do Synthetic Brain MRIs Reliably Improve Tumour Classification? A StyleGAN2-ADA Class-Plane Augmentation Study on BRISC 2025
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文研究生成合成脑部MRI对肿瘤分类的影响，属于医学图像增强任务。通过StyleGAN2-ADA生成数据，测试其对不同分类器性能的提升效果。**

- **链接: [https://arxiv.org/pdf/2605.23094](https://arxiv.org/pdf/2605.23094)**

> **作者:** José Rafael Noriega Cedeño
>
> **备注:** 18 pages, 16 figures
>
> **摘要:** Generative augmentation is often proposed as a remedy for small medical-image datasets, but synthetic images are only useful when they improve downstream task performance. "Augmentation" here means synthetic supplementation: GAN-generated samples added to the real training pool, not geometric or photometric transforms of existing images. Twelve class-plane StyleGAN2-ADA generators were trained on constrained BRISC 2025 partitions to test whether their output, with or without InceptionV3 feature-space filtering, improves held-out tumour classification across three classifier families: a random forest (RF) on InceptionV3 features, a compact two-headed convolutional neural network (CNN), and MobileViTV2, a mobile hybrid convolutional-transformer. Each was evaluated at 1:1 and 1:2 real-to-synthetic ratios. An independent GPT-5.5 blind test placed gated real-versus-synthetic discrimination at 57.73% (95% CI: 54.48--60.92%) on the model-legible subset -- modestly above chance. The RF classifier did not benefit from the synthetic MRIs. The CNN showed consistent mean gains that did not survive Holm correction. MobileViTV2 showed the clearest benefit: filtered 1:1 augmentation improved tumour classification accuracy by 1.02% absolute (95% CI: 0.54--1.54%; Holm-corrected p = 0.0104). A secondary efficiency analysis found that every augmented CNN condition selected its checkpoint 42--64% earlier than baseline, while compute-matched MobileViTV2 runs reached selection after 50--67% fewer real-data epochs. Overall, augmentation utility was found to be architecture- and ratio-dependent, not guaranteed by visual fidelity alone.
>
---
#### [new 103] Leveraging Foundation Models for Causal Generative Modeling
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于因果生成建模任务，旨在解决传统方法缺乏统一框架的问题。提出FM-CGM框架，结合预训练模型实现零样本因果推理与图像生成。**

- **链接: [https://arxiv.org/pdf/2605.23861](https://arxiv.org/pdf/2605.23861)**

> **作者:** Aneesh Komanduri; Xintao Wu
>
> **摘要:** Causal generative modeling is essential for developing reliable and transparent AI systems capable of counterfactual reasoning. While existing approaches focus on integrating causal constraints during the training of generative models, they often lack a unified framework to leverage the zero-shot reasoning capabilities of pretrained foundation models. We introduce FM-CGM, a modular framework for end-to-end visual causal reasoning using pretrained foundation models. FM-CGM formalizes the causal pipeline through three core components: a concept extractor, a concept manipulator, and a counterfactual generator. By leveraging a large reasoning model for causal inference and a text-to-image diffusion model for generation, our approach enables zero-shot causal discovery, intervention, and counterfactual generation. We then develop Causal Semantic Guidance (CSG), a cross-attention-based mechanism that ensures semantic interventions propagate to descendant concepts while preserving invariant regions. We empirically show that our approach can identify plausible causal structures and is suitable for faithful counterfactual image generation.
>
---
#### [new 104] Sample-wise Targeted Adversarial Attacks on Test-time Adaptation
- **分类: cs.LG; cs.CR; cs.CV**

- **简介: 该论文属于对抗攻击任务，针对测试时自适应（TTA）中的样本级目标攻击问题。提出一种基于元学习的攻击方法，实现仅误导带触发器的样本，同时保持整体标签分布，提高攻击隐蔽性。**

- **链接: [https://arxiv.org/pdf/2605.23411](https://arxiv.org/pdf/2605.23411)**

> **作者:** Phuc Duc Nguyen; Quang Duc Nguyen
>
> **备注:** 32 pages, 17 figures
>
> **摘要:** Test-time adaptation (TTA) effectively counters distribution shifts but exposes models to adversarial manipulation via the unlabeled test stream. Existing class-wise targeted attacks remain impractical for stealthy exploitation in this setting: since TTA operates on batches, forcing a subset of samples toward a target label unintentionally pulls similar benign samples along, resulting in a conspicuously high frequency of the target label that is easy to detect. To capture a more realistic threat, we introduce a sample-wise targeted attack. Unlike prior approaches, the attacker aims to misclassify only inputs carrying an attacker-chosen trigger, while preserving the global label distribution of benign queries to evade detection. To achieve this, we propose a meta-learning-based attack with a novel priority-aware gradient alignment strategy that explicitly prioritizes attack success. The strategy formulates the gradient update as an ellipsoidal trust-region problem, mitigating the misalignment between attack success and distributional stealth, while providing theoretical guarantees for effective optimization of the attack objective in the presence of gradient misalignment. Extensive experiments on CIFAR-10-C, CIFAR-100-C, and ImageNet-C across TTA protocols demonstrate that our method achieves high targeted success rates while maintaining a label distribution that is consistent with the no-attack baseline, making it difficult to detect in unlabeled TTA deployment scenarios. Furthermore, we demonstrate that our attack shows strong robustness against existing defenses.
>
---
#### [new 105] MedExpMem: Adapting Experience Memory for Differential Diagnosis
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出MedExpMem，用于医学微调诊断，解决模型缺乏经验积累的问题。通过记忆诊断失败案例，提升鉴别诊断能力。**

- **链接: [https://arxiv.org/pdf/2605.22872](https://arxiv.org/pdf/2605.22872)**

> **作者:** Qianhan Feng; Zhongzhen Huang; Yakun Zhu; Yannian Gu; Winnie Chiu Wing Chu; Xiaofan Zhang; Qi Dou
>
> **备注:** MICCAI 2026 Early Accept. Submission Version
>
> **摘要:** Experienced physicians develop diagnostic expertise through clinical practice, acquiring not only disease knowledge but also the ability to differentiate confusable conditions. Current medical vision-language models (VLMs) lack this capability -- their parameters encode static knowledge that does not evolve across diagnostic encounters. We propose MedExpMem, an experience memory framework enabling VLM-based diagnostic agents to accumulate differential diagnosis expertise. Unlike retrieval-augmented generation, which retrieves encyclopedic disease descriptions, MedExpMem memorizes discriminative experience derived from the agent's own diagnostic failures and organizes them as pairwise differential notes encoding key discriminators, actionable decision rules and reasoning error patterns. The framework adopts a two-phase construction process mirroring physician learning: initial practice exposes knowledge gaps, and reflective re-diagnosis refines understanding. When encountering new cases, the agent retrieves experience memory to guide differential reasoning. We evaluate MedExpMem on a radiology benchmark spanning 11 subspecialties. Results demonstrate consistent accuracy improvements, maximum 7.0%, across diverse models and scales. Analytical experiments validate experience quality and robustness, demonstrating MedExpMem as a competitive method addresses medical adaptation needs beyond the reach of parameteric learning.
>
---
#### [new 106] Turning Adaptation into Assets: Cross-Domain Bridging for Online Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决在线适应中的灾难性遗忘和负迁移问题。提出IDEA框架，通过积累和组合资产实现跨域桥梁构建。**

- **链接: [https://arxiv.org/pdf/2605.23257](https://arxiv.org/pdf/2605.23257)**

> **作者:** Zixuan Hu; Xuantuo Huang; Yancheng Li; Yichun Hu; Shengyong Xu; Ling-Yu Duan
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Navigating under non-stationary environment shifts poses a critical challenge for a Vision-and-Language Navigation (VLN) agent deployed in the wild. Yet, existing Test-Time Adaptation (TTA) methods for VLN largely treat online adaptation as transient, isolated updates, leading to catastrophic forgetting and negative transfer. To overcome these issues, we propose Inter-Domain BridgE with Historical Assets (IDEA), a novel TTA framework that transforms adaptation into the accumulation and composition of assets. Specifically, IDEA introduces soft prompts optimized via a Fisher-guided weighting scheme to capture the transferable knowledge. These optimized prompts are then augmented with domain coordinates to form a dynamic asset library. Leveraging this library, IDEA constructs a cross-domain bridge by projecting the target domain onto the convex hull of historical knowledge. These designs form a complementary loop: the evolving library underpins bridge construction, while the bridge provides superior initialization to accelerate asset optimization. Extensive experiments across REVERIE, R2R, and R2R-CE benchmarks demonstrate the consistent superiority of IDEA over existing methods, showcasing its ability to enable training-free adaptation via asset sharing.
>
---
#### [new 107] Debiased Negative Mining Improves Out-of-distribution Detection with Pre-trained Vision-Language Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于OOD检测任务，旨在解决预训练视觉-语言模型中负样本挖掘的偏差问题，提出一种去偏负样本挖掘方法以提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.23797](https://arxiv.org/pdf/2605.23797)**

> **作者:** Bo Peng; Jie Lu; Guangquan Zhang; Zhen Fang
>
> **备注:** KDD 2026
>
> **摘要:** Aiming at identifying unexpected inputs from unknown classes, out-of-distribution (OOD) detection has emerged as a pivotal approach to enhancing the reliability of machine learning models. This paper focuses on the burgeoning paradigm of post-hoc OOD detection with pre-trained vision-language models (VLMs), where a popular pipeline is to detect OOD inputs by examining their affinities between ID labels and negative labels, i.e., those semantically different from ID labels. Due to the unavailability of target OOD labels, existing works predominantly rely on heuristic rules to mine negative labels from unlabeled wild corpus data. Despite the empirical success, we argue that the power of VLM-based OOD detection has yet to be fully unleashed since the notorious false negative problem is far from addressed in the literature. With this motivation, we are interested in addressing the challenge of mining true negative labels for OOD scoring. To this end, we develop a theoretical framework for correcting the sampling bias of negatives labels by indirectly approximating the distribution of negative labels. Perhaps surprisingly, we show that the debiased negative mining can be naturally converted into Monte-Carlo sampling based on ID labels and the unlabeled wild corpus data. Extensive experiments empirically manifest that our method establishes a new state-of-the-art in a variety of OOD detection setups. Code is publicly available at \href{this https URL}{\textcolor{red}{here}}.
>
---
#### [new 108] Commutator-Induced Uncertainty in VAEs
- **分类: cs.LG; cs.CV; math.AG**

- **简介: 该论文属于生成模型任务，旨在解决VAEs中非对易结构表示不足的问题。通过引入Lie Group VAE框架，诊断并反映潜在空间的非对易性，提升重建质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.23449](https://arxiv.org/pdf/2605.23449)**

> **作者:** Tahereh Dehdarirad; Michael Felsberg; Gabriel Eilertsen; Ziliang Xiong
>
> **摘要:** Variational autoencoders (VAEs) often struggle to represent non-commutative structure in learned latent spaces. Symmetry-aware VAEs commonly address this issue by enforcing commutativity through algebraic regularization, which is appropriate for commutative transformation groups but can suppress meaningful non-commutative structure when it is intrinsic to the data. We argue that non-commutativity should instead be explicitly diagnosed and reflected in reconstruction behavior. We introduce a Lie Group VAE framework that combines geometric and algebraic perspectives on uncertainty while separating discrete generative factors from continuous geometric transformations. In a first phase, the model is trained without structural constraints while algebraic non-commutativity is measured through finite Baker-Campbell-Hausdorff deviations and decoder order sensitivity is measured through reconstruction order-swap tests. These diagnostics reveal a scale mismatch between latent non-commutativity and reconstruction behavior under unconstrained training. In a second phase, we introduce a deformation-stability constraint with a data-driven calibration constant that aligns decoder sensitivity with algebraic non-commutativity. We evaluate the framework on dSprites, 3DShapes, 3DCars, and CelebA against generic and symmetry-aware baselines, including beta-VAE, CLG-VAE, and CFASL. Across synthetic benchmarks, the method improves reconstruction quality and yields decoder-level behavior more consistent with latent non-commutative structure. Qualitative analyses show clearer order-dependent latent compositions and more stable reconstructions. On CelebA, the model yields more faithful reconstructions and factor-specific latent traversals than CFASL, while also exhibiting meaningful order-dependent interactions between learned latent directions.
>
---
#### [new 109] What Linear Probes Miss: Multi-View Probing for Weight-Space Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于模型分析任务，旨在解决单视角探针无法捕捉高阶相关性的不足。提出MVProbe框架，融合多视角信息以提升权重空间学习效果。**

- **链接: [https://arxiv.org/pdf/2605.23410](https://arxiv.org/pdf/2605.23410)**

> **作者:** Eunwoo Heo; Kyeongkook Seo; Jaejun Yoo
>
> **备注:** Accepted at ICML 2026. Code: this https URL ; Project page: this https URL
>
> **摘要:** The explosive growth of open-source model repositories has created a Model Jungle, where checkpoints are frequently shared without adequate documentation or metadata. While weight-space learning offers a pathway to identify and analyze these models directly from their parameters, processing full-scale weights is computationally prohibitive. Probing-based methods have emerged as a lightweight alternative, extracting permutation-equivariant representations via learnable probe vectors. However, existing probing methods are limited by a single-view design: they capture first-order structures but fail to encode the rich, higher-order correlation patterns inherent in row-column interactions. To bridge this gap, we introduce MVProbe, a multi-perspective probing framework that synthesizes first-order signals with interaction-aware (Gram-based) views. Our approach is theoretically grounded; we analyze the scaling laws of different probing orders to derive a principled standardization and fusion strategy that ensures balanced contributions from all branches. On the Model Jungle benchmark, MVProbe consistently outperforms the state-of-the-art ProbeX across diverse architectures, including discriminative backbones (ResNet, SupViT, MAE, DINO) and large-scale generative LoRA adapters (Stable Diffusion LoRA).
>
---
## 更新

#### [replaced 001] Visually-Guided Policy Optimization for Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言模型的多模态推理任务，旨在解决模型视觉忠实度不足的问题。通过引入VGPO框架，增强视觉关注与记忆，提升多模态任务表现。**

- **链接: [https://arxiv.org/pdf/2604.09349](https://arxiv.org/pdf/2604.09349)**

> **作者:** Zengbin Wang; Feng Xiong; Liang Lin; Xuecai Hu; Yong Wang; Yanlin Wang; Man Zhang; Xiangxiang Chu
>
> **备注:** Accepted to ACL 2026, this https URL
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning ability of vision-language models (VLMs). However, the inherent text-dominated nature of VLMs often leads to insufficient visual faithfulness, characterized by sparse attention activation to visual tokens. More importantly, our empirical analysis reveals that temporal visual forgetting along reasoning steps exacerbates this deficiency. To bridge this gap, we propose Visually-Guided Policy Optimization (VGPO), a novel framework to reinforce visual focus during policy optimization. Specifically, VGPO initially introduces a Visual Attention Compensation mechanism that leverages visual similarity to localize and amplify visual cues, while progressively elevating visual expectations in later steps to counteract visual forgetting. Building on this mechanism, we implement a dual-grained advantage re-weighting strategy: the intra-trajectory level highlights tokens exhibiting relatively high visual activation, while the inter-trajectory level prioritizes trajectories demonstrating superior visual accumulation. Extensive experiments demonstrate that VGPO achieves better visual activation and superior performance in mathematical multimodal reasoning and visual-dependent tasks. The code has been released at this https URL.
>
---
#### [replaced 002] PixelPonder: Dynamic Patch Adaptation for Enhanced Multi-Conditional Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.06684](https://arxiv.org/pdf/2503.06684)**

> **作者:** Yanjie Pan; Qingdong He; Zhengkai Jiang; Pengcheng Xu; Chaoyi Wang; Jinlong Peng; Haoxuan Wang; Yun Cao; Zhenye Gan; Mingmin Chi; Bo Peng; Yabiao Wang
>
> **摘要:** Recent advances in diffusion-based text-to-image generation have demonstrated promising results through visual condition control. However, existing ControlNet-like methods struggle with compositional visual conditioning - simultaneously preserving semantic fidelity across multiple heterogeneous control signals while maintaining high visual quality, where they employ separate control branches that often introduce conflicting guidance during the denoising process, leading to structural distortions and artifacts in generated images. To address this issue, we present PixelPonder, a novel unified control framework, which allows for effective control of multiple visual conditions under a single control structure. Specifically, we design a patch-level adaptive condition selection mechanism that dynamically prioritizes spatially relevant control signals at the sub-region level, enabling precise local guidance without global interference. Additionally, a time-aware control injection scheme is deployed to modulate condition influence according to denoising timesteps, progressively transitioning from structural preservation to texture refinement and fully utilizing the control information from different categories to promote more harmonious image generation. Extensive experiments demonstrate that PixelPonder surpasses previous methods across different benchmark datasets, showing superior improvement in spatial alignment accuracy while maintaining high textual semantic consistency.
>
---
#### [replaced 003] VGAS: Value-Guided Action-Chunk Selection for Few-Shot Vision-Language-Action Adaptation
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07399](https://arxiv.org/pdf/2602.07399)**

> **作者:** Changhua Xu; En Yu; Junyu Xuan; Jie Lu
>
> **备注:** Preprint
>
> **摘要:** Vision--Language--Action (VLA) models bridge multimodal reasoning with physical control, but adapting them to new tasks with scarce demonstrations remains unreliable. While fine-tuned VLA policies often produce semantically plausible trajectories, failures often arise from unresolved geometric ambiguities, where near-miss actions lead to divergent execution outcomes under limited supervision. We study few-shot VLA adaptation from a \emph{generation--selection} perspective and propose a novel framework \textbf{VGAS} (\textbf{V}alue-\textbf{G}uided \textbf{A}ction-chunk \textbf{S}election). It performs inference-time best-of-$N$ selection to identify action chunks that are both semantically faithful and geometrically precise. Specifically, \textbf{VGAS} employs a finetuned VLA as a high-recall proposal generator and introduces the \textrm{Q-Chunk-Former}, a geometrically grounded Transformer critic to resolve fine-grained geometric ambiguities. In addition, we propose \textit{Explicit Geometric Regularization} (\texttt{EGR}), which shapes a discriminative value landscape to preserve action ranking resolution among near-miss candidates while mitigating value instability under scarce supervision. Experiments and theoretical analysis demonstrate that \textbf{VGAS} consistently improves success rates and robustness under limited demonstrations and distribution shifts. Our code is available at this https URL.
>
---
#### [replaced 004] Mitigating Object Hallucinations via Sentence-Level Early Intervention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.12455](https://arxiv.org/pdf/2507.12455)**

> **作者:** Shangpin Peng; Senqiao Yang; Li Jiang; Zhuotao Tian
>
> **摘要:** Multimodal large language models (MLLMs) have revolutionized cross-modal understanding but continue to struggle with hallucinations - fabricated content contradicting visual inputs. Existing hallucination mitigation methods either incur prohibitive computational costs or introduce distribution mismatches between training data and model outputs. We identify a critical insight: hallucinations predominantly emerge at the early stages of text generation and propagate through subsequent outputs. To address this, we propose SENTINEL (Sentence-level Early iNtervention Through IN-domain prEference Learning), a framework that eliminates dependency on human annotations. Specifically, we first bootstrap high-quality in-domain preference pairs by iteratively sampling model outputs, validating object existence through cross-checking with two open-vocabulary detectors, and classifying sentences into hallucinated/non-hallucinated categories. Subsequently, we use context-coherent positive samples and hallucinated negative samples to build context-aware preference data iteratively. Finally, we train models using a context-aware preference loss (C-DPO) that emphasizes discriminative learning at the sentence level where hallucinations initially manifest. Experimental results show that SENTINEL can reduce hallucinations by over 90% compared to the original model and outperforms the previous state-of-the-art method on both hallucination benchmarks and general capabilities benchmarks, demonstrating its superiority and generalization ability. The models, datasets, and code are available at this https URL.
>
---
#### [replaced 005] Universal CT Representations from Anatomy to Disease Phenotype through Agglomerative Pretraining
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.21906](https://arxiv.org/pdf/2605.21906)**

> **作者:** Yuheng Li; Yuan Gao; Haoyu Dong; Yuxiang Lai; Shansong Wang; Mojtaba Safari; James E. Baciak; Xiaofeng Yang
>
> **摘要:** Computed tomography (CT) is a central to three-dimensional medical imaging, yet CT-based artificial intelligence remains fragmented across task-specific models for segmentation, classification, registration, and report analysis. Here we present FlexiCT, a family of CT foundation models trained by agglomerative continual pretraining on 266,227 CT volumes from 56 publicly available datasets, forming a large-scale public resource for CT representation learning. FlexiCT uses agglomerative pretraining across three stages: two-dimensional axial pretraining, three-dimensional anatomical pretraining and report-guided semantic alignment. This training strategy supports slice-level, volume-level and vision-language analysis. Across five downstream task families (segmentation, classification, registration, vision-language understanding and clinical retrieval), FlexiCT matches or exceeds prior task-specific approaches on multiple benchmarks. Its embeddings further organize CT scans along gradients associated with various tumor stages, suggesting that CT foundation models can capture imaging features relevant to disease phenotype characterization. Project page and code are available at: this https URL and this https URL.
>
---
#### [replaced 006] Uni-Edit: Intelligent Editing Is A General Task For Unified Model Tuning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.21487](https://arxiv.org/pdf/2605.21487)**

> **作者:** Dian Zheng; Manyuan Zhang; Hongyu Li; Hongbo Liu; Kai Zou; Kaituo Feng; Hongsheng Li
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** Currently, enhancing Unified Multimodal Models (UMMs) with image understanding, generation, and editing capabilities mainly relies on mixed multi-task training. Due to inherent task conflicts, such strategy requires complex multi-stage pipelines, massive data mixing, and balancing tricks, merely resulting in a performance trade-off rather than true mutual reinforcement. To break this paradigm, we propose Uni-Edit, an intelligent image editing task that serves as the first general task for UMM tuning. Unlike complex mixed pipelines, Uni-Edit improves performance across all three abilities at once using only one task, one training stage, and one dataset. Specifically, we first identify image editing as an inherently ideal general task, as it naturally demands both visual understanding and generation. However, existing editing data relies on simplistic instructions that severely underutilize a model's understanding capacity. To address this, we introduce the first automated and scalable data synthesis pipeline for intelligent editing, transforming diverse VQA data into complex and effective editing instructions with embedded questions and nested logic. This yields Uni-Edit-148k, pairing diverse reasoning-intensive instructions with high-quality edited images. Extensive experiments on BAGEL and Janus-Pro demonstrate that tuning solely on Uni-Edit achieves comprehensive enhancements across all three capabilities without any auxiliary operations.
>
---
#### [replaced 007] Gen-Searcher: Reinforcing Agentic Search for Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.28767](https://arxiv.org/pdf/2603.28767)**

> **作者:** Kaituo Feng; Manyuan Zhang; Shuang Chen; Yunlong Lin; Kaixuan Fan; Yilei Jiang; Hongyu Li; Dian Zheng; Chenyang Wang; Xiangyu Yue
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** Recent image generation models have shown strong capabilities in generating high-fidelity and photorealistic images. However, they are fundamentally constrained by frozen internal knowledge, thus often failing on real-world scenarios that are knowledge-intensive or require up-to-date information. In this paper, we present Gen-Searcher, as the first attempt to train a search-augmented image generation agent, which performs multi-hop reasoning and search to collect the textual knowledge and reference images needed for grounded generation. To achieve this, we construct a tailored data pipeline and curate two high-quality datasets, Gen-Searcher-SFT-10k and Gen-Searcher-RL-6k, containing diverse search-intensive prompts and corresponding ground-truth synthesis images. We further introduce KnowGen, a comprehensive benchmark that explicitly requires search-grounded external knowledge for image generation and evaluates models from multiple dimensions. Based on these resources, we train Gen-Searcher with SFT followed by agentic reinforcement learning with dual reward feedback, which combines text-based and image-based rewards to provide more stable and informative learning signals for GRPO training. Experiments show that Gen-Searcher brings substantial gains, improving Qwen-Image by around 16 points on KnowGen and 15 points on WISE. We hope this work can serve as an open foundation for search agents in image generation, and we fully open-source our data, models, and code.
>
---
#### [replaced 008] GT-SVJ: Generative-Transformer-Based Self-Supervised Video Judge For Efficient Video Reward Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05202](https://arxiv.org/pdf/2602.05202)**

> **作者:** Shivanshu Shekhar; Uttaran Bhattacharya; Raghavendra Addanki; Mehrab Tanjim; Somdeb Sarkhel; Tong Zhang
>
> **摘要:** Aligning video generative models with human preferences remains challenging: current approaches rely on Vision-Language Models (VLMs) for reward modeling, but these models struggle to capture subtle temporal dynamics. We propose a fundamentally different approach: repurposing video generative models, which are inherently designed to model temporal structure, as reward models. We present the Generative-Transformer-based Self-Supervised Video Judge (\modelname), a novel evaluation model that transforms state-of-the-art video generation models into powerful temporally-aware reward models. Our key insight is that generative models can be reformulated as energy-based models (EBMs) that assign low energy to high-quality videos and high energy to degraded ones, enabling them to discriminate video quality with remarkable precision when trained via contrastive objectives. To prevent the model from exploiting superficial differences between real and generated videos, we design challenging synthetic negative videos through controlled latent-space perturbations: temporal slicing, feature swapping, and frame shuffling, which simulate realistic but subtle visual degradations. This forces the model to learn meaningful spatiotemporal features rather than trivial artifacts. \modelname achieves state-of-the-art performance on GenAI-Bench and MonteBench using only 30K human-annotations: $6\times$ to $65\times$ fewer than existing VLM-based approaches.
>
---
#### [replaced 009] CLEAR-HPV: Interpretable concept discovery for human-papillomavirus-associated morphology in whole-slide histology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05126](https://arxiv.org/pdf/2602.05126)**

> **作者:** Weiyi Qin; Yingci Liu-Swetz; Shiwei Tan; Hao Wang
>
> **摘要:** Human papillomavirus (HPV) status is a critical determinant of prognosis and treatment response in head and neck and cervical cancers. Although attention-based multiple instance learning (MIL) achieves strong slide-level prediction for HPV-related whole-slide histopathology, it provides limited morphologic interpretability. To address this limitation, we introduce Concept-Level Explainable Attention-guided Representation for HPV (CLEAR-HPV), a framework that restructures the MIL latent space using attention to enable concept discovery without requiring concept labels during training. Operating in an attention-weighted latent space, CLEAR-HPV automatically discovers keratinizing, basaloid, and stromal morphologic concepts, generates spatial concept maps, and represents each slide using a compact concept-fraction vector. CLEAR-HPV's concept-fraction vectors preserve the predictive information of the original MIL embeddings while reducing the high-dimensional feature space (e.g., 1536 dimensions) to only 10 interpretable concepts. CLEAR-HPV generalizes consistently across TCGA-HNSCC, TCGA-CESC, and CPTAC-HNSCC, providing compact, concept-level interpretability through a general, backbone-agnostic framework for attention-based MIL models of whole-slide histopathology.
>
---
#### [replaced 010] Beyond Defenses: Manifold-Aligned Regularization for Intrinsic 3D Point Cloud Robustness
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.07590](https://arxiv.org/pdf/2605.07590)**

> **作者:** Pedro Alonso; Chongshou Li; Tianrui Li
>
> **摘要:** Despite extensive progress in point cloud robustness, existing methods primarily rely on augmentation strategies or defense mechanisms while overlooking the geometric nature of adversarial fragility. We hypothesize that adversarial vulnerability in 3D networks arises from a manifold misalignment between the latent geometry learned by the model and the intrinsic geometry of the underlying surface. Small, geometry-preserving perturbations along the input manifold often induce disproportionate distortions in feature space, potentially leading to misclassifications. We formalize this phenomenon by developing a geometric interpretation of 3D robustness that links classical adversarial theory to the intrinsic structure of point clouds. Motivated by this analysis, we introduce Manifold-Aligned Point Recognition (MAPR), a framework that regularizes the latent geometry by aligning predictions across intrinsic perturbations. MAPR augments each point cloud with intrinsic features capturing local curvature and diffusion structure, and applies a consistency loss that preserves invariance to intrinsic, geometry-preserving perturbations. Without relying on adversarial training or additional data, MAPR consistently improves robustness under multiple adversarial attacks across several datasets, achieving average robustness gains of +20.02 and +8.83 percentage points over vanilla models on ModelNet40 and ScanObjectNN, respectively.
>
---
#### [replaced 011] Enhancing 3D Semantic Scene Completion with a Refinement Module
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18363](https://arxiv.org/pdf/2512.18363)**

> **作者:** Dunxing Zhang; Jiachen Lu; Han Yang; Lei Bao; Bo Song
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** We propose ESSC-RM, a plug-and-play Enhancing framework for Semantic Scene Completion with a Refinement Module, which can be seamlessly integrated into existing SSC models. ESSC-RM operates in two phases: a baseline SSC network first produces a coarse voxel prediction, which is subsequently refined by a 3D U-Net-based Prediction Noise-Aware Module (PNAM) and Voxel-level Local Geometry Module (VLGM) under multiscale supervision. Experiments on SemanticKITTI show that ESSC-RM consistently improves semantic prediction performance. When integrated into CGFormer and MonoScene, the mean IoU increases from 16.87% to 17.27% and from 11.08% to 11.51%, respectively. These results demonstrate that ESSC-RM serves as a general refinement framework applicable to a wide range of SSC models.
>
---
#### [replaced 012] EgoInteract: Synthetic Egocentric Videos Generation for Interaction Understanding and Anticipation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18214](https://arxiv.org/pdf/2605.18214)**

> **作者:** Rosario Leonardi; Francesco Ragusa; Daniele Materia; Alessandro Passanisi; James Fort; Jakob Engel; Giovanni Maria Farinella
>
> **摘要:** Collecting large-scale egocentric video datasets with dense spatial and temporal annotations is costly, slow, and often constrained by environmental biases, privacy constraints, and limited coverage of interaction patterns. While synthetic data has shown strong potential in several vision domains, its use for egocentric perception remains relatively underexplored, especially for tasks requiring temporally coherent human-object interactions. In this work, we introduce EgoInteract, a controllable simulator for egocentric video generation designed to model fine-grained egocentric interactions and their temporal dynamics. The simulator enables precise control over camera, human body and hand motion, object manipulation, and scene composition across diverse environments. Building on this framework, we generate a synthetic egocentric video dataset with dense spatial and temporal annotations for temporal action segmentation, next-active object detection, interaction anticipation, and hand-object interaction detection. We evaluate models trained with simulated data on multiple real-world egocentric benchmarks spanning diverse environments, object categories, and interaction patterns. Results show consistent improvements over strong baselines across tasks and datasets, demonstrating the effectiveness and transferability of our simulation-based approach.
>
---
#### [replaced 013] MirrorCheck: Efficient Adversarial Defense for Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2406.09250](https://arxiv.org/pdf/2406.09250)**

> **作者:** Samar Fares; Klea Ziu; Toluwani Aremu; Nikita Durasov; Martin Takáč; Pascal Fua; Karthik Nandakumar; Ivan Laptev
>
> **摘要:** Vision-Language Models (VLMs) are increasingly susceptible to sophisticated adversarial attacks, including adaptive strategies specifically designed to bypass existing defenses. To address this vulnerability, we propose MirrorCheck, a robust and model-agnostic detection framework that operates effectively in both unimodal and multimodal settings. MirrorCheck leverages Text-to-Image (T2I) models to regenerate visual content from captions produced by the target model and assesses semantic consistency by comparing feature-space embeddings between the original and synthesized images. To enhance robustness against adaptive attacks, MirrorCheck introduces a stochastic defense strategy that randomly selects T2I generators and image encoders from a diverse model zoo. Additionally, we incorporate a novel One-Time-Use (OTU) perturbation applied to the selected encoder embeddings, regulated by a scaling factor, which decreases the effectiveness of adaptive attacks. Extensive experiments across multiple threat scenarios demonstrate that MirrorCheck consistently outperforms baseline methods, and maintains its utility even under strong adaptive adversarial conditions.
>
---
#### [replaced 014] Few-Shot Left Atrial Wall Segmentation in 3D LGE MRI via Meta-Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24985](https://arxiv.org/pdf/2603.24985)**

> **作者:** Yusri Al-Sanaani; Rebecca Thornhill; Pablo Nery; Elena Pena; Robert deKemp; Calum Redpath; David Birnie; Sreeraman Rajan
>
> **备注:** Accepted to IEEE EMBC 2026
>
> **摘要:** Segmenting the left atrial (LA) wall from late gadolinium enhancement magnetic resonance imaging (LGE-MRI) is challenging because of its thin geometry, low contrast, and limited expert annotations. We propose a model-agnostic meta-learning (MAML) framework with a 3D residual U-Net backbone for K-shot (K = 5, 10, 20) LA wall segmentation. The framework is meta-trained on LA wall tasks together with auxiliary LA and right atrial (RA) cavity tasks and uses a boundary-aware composite loss to improve thin-structure delineation. We evaluated MAML on a held-out clean test set and assessed its robustness under an unseen synthetic domain shift and on a local cohort. On the held-out clean test set, MAML outperformed the K-shot fine-tuning baseline at 5-shot, achieving Dice coefficient (DSC) = 0.54 versus 0.48 and Hausdorff distance (HD95) = 4.60 versus 6.40 mm. At 20-shot, MAML approached the fully supervised model trained from scratch, with DSC = 0.59 versus 0.61. Under unseen shifts, performance decreased relative to clean testing but improved consistently as K increased. At 5-shot, MAML achieved DSC = 0.52 and HD95 = 5.02 mm under the unseen synthetic shift, and DSC = 0.50 and HD95 = 5.43 mm on the local cohort. These results suggest that meta-learning can improve thin-wall delineation in low-shot adaptation and may reduce the annotation burden for atrial remodeling assessment.
>
---
#### [replaced 015] Edge Assisted Multi-Camera Vehicle Tracking Framework for Real-Time and Scalable Deployment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13904](https://arxiv.org/pdf/2511.13904)**

> **作者:** Yuqiang Lin; Sam Lockyer; Shucheng Zhang; Florian Stanek; Markus Zarbock; Adrian Evans; Wenbin Li; Yinhai Wang; Nic Zhang
>
> **摘要:** Cameras are a core sensing modality in modern intelligent transportation systems (ITS), providing rich visual information on road-user activities. Multi-Camera Vehicle Tracking (MCVT) uses this data to reconstruct vehicle trajectories across camera networks, supporting applications such as traffic flow prediction and optimisation. However, most existing MCVT studies emphasise tracking accuracy while paying limited attention to real-time performance and scalability, both essential for real-world and city-scale deployment. To address this gap, we propose Edge-Assisted, Scalable and Efficient MCVT (EASE-MCVT), a distributed edge--server framework designed for real-time throughput and scalable operation. On the edge side, each camera stream is processed through object detection, single-camera tracking, geo-mapping and feature extraction, while only lightweight metadata, including vehicle locations and appearance features, is sent to the central server for cross-camera association. To improve both tracking accuracy and system efficiency, EASE-MCVT is optimised from algorithmic and system perspectives. Algorithmically, it introduces a dynamic workload scheme for tracklet-level feature extraction, a server-side re-match module to reconnect fragmented tracklets, and a self-supervised camera link model that learns spatio-temporal constraints to accelerate and stabilise cross-camera association. Systemically, it integrates production-oriented data engineering components to standardise deployment and data exchange for large-scale operation. To the best of our knowledge, EASE-MCVT is the first MCVT framework explicitly designed to address both real-time performance and scalability in a distributed edge--server setting. Experiments on the RoundaboutHD and CityFlow datasets demonstrate real-time throughput with competitive tracking accuracy, paving the way for city-wide real-time traffic management.
>
---
#### [replaced 016] VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.13596](https://arxiv.org/pdf/2604.13596)**

> **作者:** Yulu Gao; Bohao Zhang; Zongheng Tang; Jitong Liao; Wenjun Wu; Si Liu
>
> **摘要:** Instance-level object segmentation across disparate egocentric and exocentric views is a fundamental challenge in visual understanding, critical for applications in embodied AI and remote collaboration. This task is exceptionally difficult due to severe changes in scale, perspective, and occlusion, which destabilize direct pixel-level matching. While recent geometry-aware models like VGGT provide a strong foundation for feature alignment, we find they often fail at dense prediction tasks due to significant pixel-level projection drift, even when their internal object-level attention remains consistent. To bridge this gap, we introduce VGGT-Segmentor (VGGT-S), a framework that unifies robust geometric modeling with pixel-accurate semantic segmentation. VGGT-S leverages VGGT's powerful cross-view feature representation and introduces a novel Union Segmentation Head. This head operates in three stages: mask prompt fusion, point-guided prediction, and iterative mask refinement, effectively translating high-level feature alignment into a precise segmentation mask. Furthermore, we propose a single-image self-supervised training strategy that eliminates the need for paired annotations and enables strong generalization. On the Ego-Exo4D benchmark, VGGT-S sets a new state-of-the-art, achieving 67.7% and 68.0% average IoU for Ego to Exo and Exo to Ego tasks, respectively, significantly outperforming prior methods. Notably, our correspondence-free pretrained model surpasses most fully-supervised baselines, demonstrating the effectiveness and scalability of our approach. Code is publicly available at: this https URL.
>
---
#### [replaced 017] Moment-Reenacting: Inverse Motion Degradation with Cross-shutter Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.22423](https://arxiv.org/pdf/2605.22423)**

> **作者:** Xiang Ji; Guixu Lin; Zhengwei Yin; Jiancheng Zhao; Yinqiang Zheng
>
> **备注:** Accepted by TPAMI
>
> **摘要:** Motion degradation, manifested as blur in global shutter (GS) images or rolling shutter (RS) distortion in RS counterparts, remains a fundamental challenge in computational imaging, especially under fast motion or low-light conditions. While prior works have treated blur decomposition and RS temporal super-resolution as separate tasks, this separation fails to exploit their intrinsic complementarity. In this paper, we propose a unified framework to invert motion degradation and reenact imaging moment by jointly leveraging the complementary characteristics of GS blur and RS distortion. To this end, we introduce a novel dual-shutter setup that captures synchronized blur-RS image pairs and demonstrate that this combination effectively resolves temporal and spatial ambiguities inherent in both modalities. For allowing flexible performance-cost trade-offs, we further extend this dual-shutter setup to a stereo Blur-RS configuration with a narrow baseline. In addition, we construct a triaxial imaging system to collect a real-world dataset with aligned GS-RS pairs and ground-truth high-speed frames, enabling robust training and evaluation beyond synthetic data. Our proposed network explicitly disentangles motion into context-aware and temporally-sensitive representations via a dual-stream motion interpretation module, followed by a self-prompted frame reconstruction stage. Extensive experiments validate the superiority and generalizability of our approach, establishing a new paradigm for realistic high-speed video reconstruction under complex motion degradations. Codes and more resources are available at this https URL.
>
---
#### [replaced 018] GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决动态场景建模与动作精准性问题。提出GAF方法，通过4D高斯动作场实现更准确的场景重建和动作预测。**

- **链接: [https://arxiv.org/pdf/2506.14135](https://arxiv.org/pdf/2506.14135)**

> **作者:** Ying Chai; Litao Deng; Ruizhi Shao; Jiajun Zhang; Kangchen Lv; Liangjun Xing; Xiang Li; Hongwen Zhang; Yebin Liu
>
> **备注:** this https URL
>
> **摘要:** Accurate scene perception is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we adopt a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing 4D modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF provides three interrelated outputs: reconstruction of the current scene, prediction of future frames, and estimation of init action via Gaussian motion. Furthermore, we employ an action-vision-aligned denoising framework, conditioned on a unified representation that combines the init action and the Gaussian perception, both generated by the GAF, to further obtain more precise actions. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR, +0.3864 SSIM and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average +7.3% success rate in robotic manipulation tasks over state-of-the-art methods.
>
---
#### [replaced 019] VFM$^{4}$SDG: Unveiling the Power of VFMs for Single-Domain Generalized Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.21502](https://arxiv.org/pdf/2604.21502)**

> **作者:** Yupeng Zhang; Ruize Han; Ningnan Guo; Wei Feng; Song Wang; Liang Wan
>
> **摘要:** Real-world weather, illumination, and imaging variations often induce severe domain shifts, degrading single-source detectors in unseen environments. Existing single-domain generalized object detection (SDGOD) methods mainly rely on data augmentation or domain-invariant learning, while largely overlooking how domain shift disrupts detector prediction stability. Through analytical experiments, we find that performance degradation is mainly dominated by increasing missed detections. Further analysis shows that this phenomenon stems from reduced cross-domain stability in DETR-style detectors: domain shift disrupts encoder-side object-background and inter-instance relations, and further weakens the semantic-spatial binding between decoder queries and real objects. Motivated by this, we find that vision foundation models (VFMs) still preserve stable relational structures and object responses under severe shifts, making them suitable cross-domain stability priors to compensate for detector degradation. To this end, we propose VFM$^{4}$SDG, a dual-prior learning framework for SDGOD, which introduces a frozen VFM into encoder representation learning and decoder query modeling. Specifically, we propose Cross-domain Stable Relational Prior Distillation to distill stable object-background and inter-instance relations from the VFM into the encoder, compensating for relational degradation. Meanwhile, we propose Semantic-Contextual Prior-based Query Enhancement, which injects category semantic prototypes and global object context into queries before they enter the decoder layer, enhancing semantic-spatial query-object binding stability. Extensive experiments show that VFM$^{4}$SDG significantly outperforms existing advanced methods on standard SDGOD benchmarks and two mainstream DETR-based detection frameworks, demonstrating its effectiveness, robustness, and generality.
>
---
#### [replaced 020] World-R1: Reinforcing 3D Constraints for Text-to-Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.24764](https://arxiv.org/pdf/2604.24764)**

> **作者:** Weijie Wang; Xiaoxuan He; Youping Gu; Yifan Yang; Zeyu Zhang; Yefei He; Yanbo Ding; Xirui Hu; Donny Y. Chen; Zhiyuan He; Yuqing Yang; Bohan Zhuang
>
> **备注:** ICML 2026, Project Page: this https URL, Code: this https URL
>
> **摘要:** Recent video foundation models demonstrate impressive visual synthesis but frequently suffer from geometric inconsistencies. While existing methods attempt to inject 3D priors via architectural modifications, they often incur high computational costs and limit scalability. We propose World-R1, a framework that aligns video generation with 3D constraints through reinforcement learning. To facilitate this alignment, we introduce a specialized pure text dataset tailored for world simulation. Utilizing Flow-GRPO, we optimize the model using feedback from pre-trained 3D foundation models and vision-language models to enforce structural coherence without altering the underlying architecture. We further employ a periodic decoupled training strategy to balance rigid geometric consistency with dynamic scene fluidity. Extensive evaluations reveal that our approach significantly enhances 3D consistency while preserving the original visual quality of the foundation model, effectively bridging the gap between video generation and scalable world simulation.
>
---
#### [replaced 021] GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.21605](https://arxiv.org/pdf/2605.21605)**

> **作者:** Sixiang Chen; Zhaohu Xing; Tian Ye; Xinyu Geng; Yunlong Lin; Jianyu Lai; Xuanhua He; Fuxiang Zhai; Jialin Gao; Lei Zhu
>
> **摘要:** Open-ended image generation is no longer a simple prompt-to-image problem. High-quality generation often requires an agent to combine a model's internal generative ability with external resources. As requests become more diverse and demanding, we aim to develop a general image-generation agent that can self-evolve through trajectories and use tools more effectively across varied generation challenges. To this end, we propose GenEvolve, a self-evolving framework based on Tool-Orchestrated Visual Experience Distillation. In GenEvolve, each generation attempt is modeled as a tool-orchestrated trajectory, where the agent gathers evidence, selects references, invokes generation skills, and composes them into a prompt-reference program. Unlike existing agentic generation methods that mainly rely on image-level scalar rewards, GenEvolve compares multiple trajectories for the same request and abstracts best-worst differences into structured visual experience, provided only to a privileged teacher branch. Inspired by on-policy self-distillation, Visual Experience Distillation provides dense token-level supervision, helping the student internalize better search, knowledge activation, reference selection, and prompt construction. We further construct GenEvolve-Data and GenEvolve-Bench. Experiments on public benchmarks and GenEvolve-Bench show substantial gains over strong baselines, achieving state-of-the-art performance among current image-generation frameworks. Our website is as follows: this https URL
>
---
#### [replaced 022] Eyes on VLM: Benchmarking Gaze Following and Social Gaze Prediction in Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.19859](https://arxiv.org/pdf/2605.19859)**

> **作者:** Hengfei Wang; Anshul Gupta; Pierre Vuillecard; Jean-Marc Odobez
>
> **备注:** Under review
>
> **摘要:** Vision-language models (VLMs) have rapidly evolved into general-purpose multimodal reasoners with strong zero-shot generalization. In this context, VLMs could greatly benefit the analysis of human gaze and attention, a central task in human behavior understanding that requires reasoning about the physical scene as well as the activity, interactions, and social context. However, the extent to which VLMs can reliably understand human gaze and related attentional behaviors remains largely unexplored. In this work, we present EyeVLM, a systematic evaluation framework for gaze understanding in VLMs across two complementary dimensions: tasks and models. To assess gaze understanding capabilities, we focus on two core tasks. The first, gaze following, i.e., predicting the 2D location where a person is looking, has a geometric and visual processing focus, requiring a precise understanding of the human face, attention direction, 3D scene structure, and spatial grounding of attended targets. The second, social gaze prediction, requires social and relational reasoning over multi-person interactions (e.g., mutual gaze and shared attention), and may benefit more from the LLM semantic reasoning capabilities within VLMs. Regarding models, EyeVLM evaluates these tasks in two ways: a zero-shot setting with a diverse set of state-of-the-art open- and closed-source VLMs, exploring different prompting strategies; and a fine-tuning approach based on task-specific QA pairs, studying the impact of model scale and data scale. As benchmarks, we rely on existing gaze understanding datasets and perform a systematic comparison with state-of-the-art purely visual models. Overall, our results show that current VLMs lack precise gaze understanding capabilities. While standard training helps reduce the gap with visual models, significant improvements are still needed.
>
---
#### [replaced 023] UniReg: A Universal Model for Controllable CT Image Registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.12868](https://arxiv.org/pdf/2503.12868)**

> **作者:** Zi Li; Jianpeng Zhang; Tai Ma; Tony C. W. Mok; Yan-Jie Zhou; Zeli Chen; Xianghua Ye; Le Lu; Cheng Chen; Dakai Jin
>
> **摘要:** Learning-based medical image registration has matched the accuracy of conventional methods while offering superior computational efficiency. However, existing approaches suffer from poor generalization across diverse clinical scenarios, requiring the laborious development of multiple isolated networks for specific registration tasks, e.g., inter-/intra-subject registration or anatomical region-specific alignment, leading to cumbersome development pipelines. To overcome this limitation, we propose UniReg, the first conditional unified model for multi-scenario CT image registration, which combines the precision advantages of task-specific learning methods with the generalization of traditional optimization methods. Our key innovation is a unified registration framework that adaptively estimates deformation fields conditioned on: (1) anatomical structure priors, (2) registration type constraints (inter/intra-subject), and (3) instance-specific features, enabling optimal alignment across heterogeneous scenarios within a single model. Through comprehensive experiments on multiple CT/MR registration datasets, UniReg achieves superior average registration accuracy compared with current state-of-the-art learning-based methods while exhibiting strong cross-scenario generalization. Moreover, by replacing multiple isolated task-specific models with a compact unified model, UniReg substantially reduces the overall training burden in terms of total training cost and model redundancy.
>
---
#### [replaced 024] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于在线高精地图构建任务，旨在减少标注依赖。通过地理对比学习增强特征表示，提升地图感知性能。**

- **链接: [https://arxiv.org/pdf/2603.10688](https://arxiv.org/pdf/2603.10688)**

> **作者:** Jonas Merkert; Alexander Blumberg; Jan-Hendrik Pauls; Christoph Stiller
>
> **摘要:** Autonomous vehicles rely on map information to understand the world around them. However, the creation and maintenance of offline high-definition (HD) maps remains costly. A more scalable alternative lies in online HD map construction, which only requires map annotations at training time. To further reduce the need for annotating vast training labels, self-supervised training provides an alternative. This work focuses on improving the latent birds-eye-view (BEV) feature grid representation within a vectorized online HD map construction model by enforcing geospatial consistency between overlapping BEV feature grids as part of a contrastive loss function. To ensure geospatial overlap for contrastive pairs, we introduce an approach to analyze the overlap between traversals within a given dataset and generate subsidiary dataset splits following adjustable multi-traversal requirements. We train the same model supervised using a reduced set of single-traversal labeled data and self-supervised on a broader unlabeled set of data following our multi-traversal requirements, effectively implementing a semi-supervised approach. Our approach outperforms the supervised baseline across the board, both quantitatively in terms of the downstream tasks vectorized map perception performance and qualitatively in terms of segmentation in the principal component analysis (PCA) visualization of the BEV feature space.
>
---
#### [replaced 025] PROGRESSLM: Towards Progress Reasoning in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型的进度推理任务，旨在解决从部分观察中推断任务进展的问题。研究提出Progress-Bench基准和ProgressLM数据集，探索两种推理方法，发现现有模型在此任务上表现不佳。**

- **链接: [https://arxiv.org/pdf/2601.15224](https://arxiv.org/pdf/2601.15224)**

> **作者:** Jianshu Zhang; Chengxuan Qian; Haosen Sun; Haoran Lu; Dingcheng Wang; Letian Xue; Han Liu
>
> **备注:** ACL 2026 Camera Ready Version
>
> **摘要:** Estimating task progress requires reasoning over long-horizon dynamics rather than recognizing static visual content. While modern Vision-Language Models (VLMs) excel at describing what is visible, it remains unclear whether they can infer how far a task has progressed from partial observations. To this end, we introduce Progress-Bench, a benchmark for systematically evaluating progress reasoning in VLMs. Beyond benchmarking, we further explore a human-inspired two-stage progress reasoning paradigm through both training-free prompting and training-based approach based on curated dataset ProgressLM-45K. Experiments on 14 VLMs show that most models are not yet ready for task progress estimation, exhibiting sensitivity to demonstration modality and viewpoint changes, as well as poor handling of unanswerable cases. While training-free prompting that enforces structured progress reasoning yields limited and model-dependent gains, the training-based ProgressLM-3B achieves consistent improvements even at a small model scale, despite being trained on a task set fully disjoint from the evaluation tasks. Further analyses reveal characteristic error patterns and clarify when and why progress reasoning succeeds or fails. Website: this https URL
>
---
#### [replaced 026] Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.11146](https://arxiv.org/pdf/2602.11146)**

> **作者:** Gongye Liu; Bo Yang; Yida Zhi; Zhizhou Zhong; Lei Ke; Didan Deng; Han Gao; Yongxiang Huang; Kaihao Zhang; Hongbo Fu; Wenhan Luo
>
> **备注:** Accepted by ICML 2026. Code: this https URL
>
> **摘要:** Preference optimization for diffusion and flow-matching models relies on reward functions that are both discriminatively robust and computationally efficient. Vision-Language Models (VLMs) have emerged as the primary reward provider, leveraging their rich multimodal priors to guide alignment. However, their computation and memory cost can be substantial, and optimizing a latent diffusion generator through a pixel-space reward introduces a domain mismatch that complicates alignment. In this paper, we propose DiNa-LRM, a diffusion-native latent reward model that formulates preference learning directly on noisy diffusion states. Our method introduces a noise-calibrated Thurstone likelihood with diffusion-noise-dependent uncertainty. DiNa-LRM leverages a pretrained latent diffusion backbone with a timestep-conditioned reward head, and supports inference-time noise ensembling, providing a diffusion-native mechanism for test-time scaling and robust rewarding. Across image alignment benchmarks, DiNa-LRM substantially outperforms existing diffusion-based reward baselines and achieves performance competitive with state-of-the-art VLMs at a fraction of the computational cost. In preference optimization, we demonstrate that DiNa-LRM improves preference optimization dynamics, enabling faster and more resource-efficient model alignment.
>
---
#### [replaced 027] Towards Generalizable Mapping of Hedges and Linear Woody Features from Earth Observation Data: a national Product for Germany
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.27247](https://arxiv.org/pdf/2604.27247)**

> **作者:** Thorsten Hoeser; Verena Huber-Garcia; Sarah Asam; Ursula Gessner; Claudia Kuenzer
>
> **备注:** 33 pages, 17 figures
>
> **摘要:** Hedges and other linear woody features provide valuable ecosystem services, particularly within intensively managed agricultural landscapes. They are key elements for climate adaptation and biodiversity amongst others not only due to a largely varying flora, but also as a feeding-, resting-, and nesting place for many animals and insects including valuable pollinators. Therefore, they require dedicated management, preservation, and attention. Thus, systematic and large-scale mapping of these features from Earth observation data is of high importance. However, transferable and reusable workflows for linear woody feature mapping remain a key methodological challenge, given the diversity of sensor types, spatial resolutions, data acquisition conditions, and complex landscape variability encountered across study areas. We introduce a modular workflow built around two independently optimizable components. Firstly, a flexible input data interface that consolidates heterogeneous Earth observation data into a binary woody vegetation mask, and secondly, a deep neural network trained to separate linear from non-linear shapes within these masks. We demonstrate the workflow by deriving three national-scale linear woody feature maps for all of Germany from three input sources with 0.73 m, 1 m and 3 m spatial resolution, respectively, by using a single trained model without retraining. Evaluation against refined reference data from four federal state biotope mapping campaigns and comparison with two existing linear woody feature maps demonstrate that the workflow produces competitive results across all evaluation sites on a national level. The modular design and its demonstrated applicability at national scale provide a foundation for scalable and generalizable linear woody feature mapping beyond Germany.
>
---
#### [replaced 028] ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02897](https://arxiv.org/pdf/2603.02897)**

> **作者:** Hao Cao; Chengbin Liang; Wenqi Guo; Zhijin Qin; Jungong Han
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** Recent advances in generative image compression (GIC) have delivered remarkable improvements in perceptual quality. However, many GICs rely on large-scale and rigid models, which severely constrain their utility for flexible transmission and practical deployment in low-bitrate scenarios. To address these issues, we propose Progressive Generative Image Compression (ProGIC), a compact codec built on residual vector quantization (RVQ). In RVQ, a sequence of vector quantizers encodes the residuals stage by stage, each with its own codebook. The resulting codewords sum to a coarse-to-fine reconstruction and a progressive bitstream, enabling previews from partial data. We pair this with a lightweight backbone based on depthwise-separable convolutions and small attention blocks, enabling practical deployment on both GPUs and CPU-only devices. Experimental results show that ProGIC attains comparable compression performance compared with previous methods. It achieves bitrate savings of up to 57.57% on DISTS and 58.83% on LPIPS compared to MS-ILLM on the Kodak dataset. Beyond perceptual quality, ProGIC enables progressive transmission for flexibility, and also delivers over 10 times faster encoding and decoding compared with MS-ILLM on GPUs for efficiency.
>
---
#### [replaced 029] UniEmo: Unifying Emotional Understanding and Generation with Learnable Expert Queries
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.23372](https://arxiv.org/pdf/2507.23372)**

> **作者:** Yijie Zhu; Lingsen Zhang; Zitong Yu; Rui Shao; Tao Tan; Liqiang Nie
>
> **备注:** Accepted to TIP 2026
>
> **摘要:** Emotional understanding and generation are often treated as separate tasks, yet they are inherently complementary and can mutually enhance each other. In this paper, we propose the UniEmo, a unified framework that seamlessly integrates these two tasks. The key challenge lies in the abstract nature of emotions, necessitating the extraction of visual representations beneficial for both tasks. To address this, we propose a hierarchical emotional understanding chain with learnable expert queries that progressively extracts multi-scale emotional features, thereby serving as a foundational step for unification. Simultaneously, we fuse these expert queries and emotional representations to guide the diffusion model in generating emotion-evoking images. To enhance the diversity and fidelity of the generated emotional images, we further introduce the emotional correlation coefficient and emotional condition loss into the fusion process. This step facilitates fusion and alignment for emotional generation guided by the understanding. In turn, we demonstrate that joint training allows the generation component to provide implicit feedback to the understanding part. Furthermore, we propose a novel data filtering algorithm to select high-quality and diverse emotional images generated by the well-trained model, which explicitly feedback into the understanding part. Together, these generation-driven dual feedback processes enhance the model's understanding capacity. Extensive experiments show that UniEmo significantly outperforms state-of-the-art methods in both emotional understanding and generation tasks. The code for the proposed method is available at this https URL.
>
---
#### [replaced 030] On the Provable Importance of Gradients for Language-Assisted Image Clustering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.16335](https://arxiv.org/pdf/2510.16335)**

> **作者:** Bo Peng; Jie Lu; Guangquan Zhang; Zhen Fang
>
> **备注:** revised and extended version of ICCV2025
>
> **摘要:** This paper investigates the recently emerged problem of Language-assisted Image Clustering (LaIC), where textual semantics are leveraged to improve the discriminability of visual representations to facilitate image clustering. Due to the unavailability of true class names, one of core challenges of LaIC lies in how to filter positive nouns, i.e., those semantically close to the images of interest, from unlabeled wild corpus data. Existing filtering strategies are predominantly based on the off-the-shelf feature space learned by CLIP; however, despite being intuitive, these strategies lack a rigorous theoretical foundation. To fill this gap, we propose a novel gradient-based framework, termed as GradNorm, which is theoretically guaranteed and shows strong empirical performance. In particular, we measure the positiveness of each noun based on the magnitude of gradients back-propagated from the cross-entropy between the predicted target distribution and the softmax output. Theoretically, we provide a rigorous error bound to quantify the separability of positive nouns by GradNorm and prove that GradNorm naturally subsumes existing filtering strategies as extremely special cases of itself. Empirically, extensive experiments show that GradNorm achieves the state-of-the-art clustering performance on various benchmarks. Code is publicly available at \href{this https URL}{here}.
>
---
#### [replaced 031] InfVSR: Toward Consistency-Driven Streaming Generative Video Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00948](https://arxiv.org/pdf/2510.00948)**

> **作者:** Ziqing Zhang; Kai Liu; Zheng Chen; Xi Li; Yucong Chen; Bingnan Duan; Linghe Kong; Yulun Zhang
>
> **备注:** Code and model are available at this https URL
>
> **摘要:** Real-world videos often extend over thousands of frames. Existing generative video super-resolution (VSR) approaches, however, face two persistent challenges when processing long sequences: (1) inefficiency due to the heavy cost of multi-step denoising for full-length sequences; and (2) poor consistency is hindered by temporal decomposition that causes artifacts and discontinuities. To break these limits, we propose InfVSR, which reformulates VSR as an autoregressive-one-step-diffusion paradigm, and enables streaming inference with video diffusion priors. First, we adapt the pretrained DiT into a causal structure, maintaining both local and global coherence via rolling KV-cache and joint visual guidance. Second, we distill the diffusion process into a single step efficiently, with patch-wise pixel supervision and cross-chunk distribution matching. To fill the gap in long-form video evaluation, we build a new benchmark tailored for extended sequences and further introduce semantic-level metrics to comprehensively assess temporal consistency. Our method pushes the frontier of long-form VSR, achieves state-of-the-art quality with enhanced semantic consistency, and delivers up to 58x speed-up over existing methods such as MGLD-VSR. Our code and models are available at this https URL.
>
---
#### [replaced 032] Variance Reduction for Expectations with Diffusion Teachers
- **分类: cs.LG; cs.AI; cs.CV; stat.CO; stat.ML**

- **链接: [https://arxiv.org/pdf/2605.21489](https://arxiv.org/pdf/2605.21489)**

> **作者:** Jesse Bettencourt; Xindi Wu; Matan Atzmon; James Lucas; Jonathan Lorraine
>
> **备注:** Project page: this https URL
>
> **摘要:** Pretrained diffusion models serve as frozen teachers feeding downstream pipelines such as text-to-3D, single-step distillation, and data attribution. The teacher gradients these pipelines consume are Monte Carlo (MC) expectations over noise levels and Gaussian noise samples; their estimator variance dominates compute cost because each draw requires expensive upstream work (rendering, simulation, encoding). We introduce CARV, a compute-aware variance-accounting framework that motivates a hierarchical MC estimator: amortize the expensive upstream computation over cheap diffusion-noise resamples, sharpened by timestep importance sampling and a stratified-inverse-CDF construction. In our text-to-3D distillation and attribution experiments, CARV delivers 2-3x effective compute multipliers (most from amortized reuse; ~25% additional from IS+stratification) without changing the objective; in single-step distillation, the same techniques cut gradient variance by an order of magnitude but do not improve downstream FID, marking the regime where MC variance is no longer the bottleneck.
>
---
#### [replaced 033] NeuralBoneReg: An Instance-Specific Label-Free Point Cloud-Based Method for Multi-Modal Bone Surface Registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14286](https://arxiv.org/pdf/2511.14286)**

> **作者:** Luohong Wu; Matthias Seibold; Nicola A. Cavalcanti; Yunke Ao; Roman Flepp; Aidana Massalimova; Lilian Calvet; Philipp Fürnstahl
>
> **摘要:** In computer- and robot-assisted orthopedic surgery (CAOS), patient-specific surgical plans derived from preoperative imaging define target locations and implant trajectories. During surgery, these plans must be accurately transferred, relying on precise cross-registration between preoperative and intraoperative data. However, substantial modality heterogeneity across imaging modalities makes this registration challenging and error-prone. Robust, automatic, and modality-agnostic bone surface registration is therefore clinically important. We propose NeuralBoneReg, a self-supervised, surface-based framework that registers bone surfaces using 3D point clouds as a modality-agnostic representation. NeuralBoneReg includes two modules: an implicit neural unsigned distance field (UDF) that learns the preoperative bone model, and an MLP-based registration module that performs global initialization and local refinement by generating transformation hypotheses to align the intraoperative point cloud with the neural UDF. Unlike SOTA supervised methods, NeuralBoneReg operates in a self-supervised manner, without requiring inter-subject training data. We evaluated NeuralBoneReg against baseline methods on two publicly available multi-modal datasets: a CT-ultrasound dataset of the fibula and tibia (UltraBones100k) and a CT-RGB-D dataset of spinal vertebrae (SpineDepth). The evaluation also includes a newly introduced CT-ultrasound dataset of cadaveric subjects containing femur and pelvis (UltraBones-Hip), which will be made publicly available. NeuralBoneReg matches or surpasses existing methods across all datasets, achieving mean RRE/RTE of 1.83°/2.02 mm on UltraBones100k, 1.90°/1.56 mm on UltraBones-Hip, and 3.78°/2.80 mm on SpineDepth. These results demonstrate strong generalizability across anatomies and modalities, providing robust and accurate cross-modal alignment for CAOS.
>
---
#### [replaced 034] DocVAL: Validated Chain-of-Thought Distillation for Grounded Document VQA
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.22521](https://arxiv.org/pdf/2511.22521)**

> **作者:** Pinaki Prasad Guha Neogi; Ahmad Mohammadshirazi; Ser-Nam Lim; Rajiv Ramnath
>
> **摘要:** Document visual question answering requires models not only to answer questions correctly, but also to precisely localize answers within complex document layouts. While large vision-language models (VLMs) achieve strong spatial grounding, their inference cost and latency limit real-world deployment. Compact VLMs are more efficient, but they often suffer substantial localization degradation under standard fine-tuning or distillation. To address this gap, we propose DocVAL, a validated chain-of-thought (CoT) distillation framework that transfers explicit spatial reasoning from large teacher models to compact, deployable student VLMs. DocVAL combines (1) teacher-generated spatial CoT supervision, (2) a rule-based dual-mode validator that filters low-quality training signals and provides fine-grained, pixel-level corrective feedback, and (3) a validation-driven two-stage training procedure with iterative refinement. Text detection is used only as training-time scaffolding for supervision and validation, enabling the final student to operate as a pure VLM without OCR or detection at inference. Across multiple document understanding benchmarks, DocVAL yields consistent improvements of up to 6-7 ANLS points over comparable compact VLMs. We further introduce mean Average Precision (mAP) as a localization metric for document question answering and report strong spatial grounding performance under this new evaluation. We release 95K validator-verified CoT traces and show that high-quality, validated supervision is more effective than scaling unfiltered data, enabling efficient and trustworthy document grounding. Code/Data: this https URL
>
---
#### [replaced 035] Anatomy-Guided Vision-Language Learning with Angular Prototype Separation for Multi-Label Video Capsule Endoscopy Classification Under Class Imbalance
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.17879](https://arxiv.org/pdf/2603.17879)**

> **作者:** Podakanti Satyajith Chary; Nagarajan Ganapathy
>
> **备注:** 12 pages, 1 figure, ICPR 2026 RARE-VISION Competition
>
> **摘要:** This work presents a multi-label temporal event detection framework for video capsule endoscopy (VCE) that addresses the extreme class imbalance inherent in the Galar dataset by combining two principal contributions: an Angular Separation Loss on class prototypes and a Biological State Machine temporal decoder. The backbone remains BiomedCLIP, a biomedical vision-language foundation model. Three consecutive frames are fused through a Local Differencing Attention module that amplifies transient pathological signals by suppressing static temporal redundancy. An Anatomy Context Head then conditions pathological predictions on soft anatomical activations, exploiting the known spatial co-occurrence structure of GI findings. Learnable text-feature prompts and prototype-based logit augmentation are trained alongside an Angular Separation Loss that penalizes off-diagonal cosine similarity between class prototypes, preventing the prototype collapse that afflicts rare classes under extreme imbalance. To counteract the skewed label distribution, the training regime combines asymmetric focal loss, inverse-frequency weighted sampling, temporal Mixup, Exponential Moving Average, and per-class threshold calibration. The Biological State Machine decoder replaces naive gap merging with a physiologically grounded forward-only state transition over anatomy labels, eliminating the fragmentation artefact that produced hundreds of spurious anatomy events per video in the prior approach and reducing per-video anatomy output to 2--3 clinically realistic events. On the held-out RARE-VISION test set comprising three NaviCam examinations (161,025 frames), the updated pipeline achieves an overall temporal mAP@0.5 of 0.3597 and mAP@0.95 of 0.3399, representing a relative improvement of 46% and 44% respectively over the prior submission, with total inference completed in approximately 21 minutes on a single GPU.
>
---
#### [replaced 036] HorizonDrive: Self-Corrective Autoregressive World Model for Long-horizon Driving Simulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.11596](https://arxiv.org/pdf/2605.11596)**

> **作者:** Conglang Zhang; Yifan Zhan; Qingjie Wang; Zhanpeng Ouyang; Yu Li; Zihao Yang; Xiaoyang Guo; Weiqiang Ren; Qian Zhang; Zhen Dong; Yinqiang Zheng; Wei Yin; Zhengqing Chen
>
> **备注:** Comments: 22 pages, 14 figures. Project page: this https URL Code: this https URL
>
> **摘要:** Closed-loop driving simulation requires real-time interaction beyond short offline clips, pushing current driving world models toward autoregressive (AR) rollout. Existing AR distillation approaches typically rely on frame sinks or student-side degradation training. The former transfers poorly to driving due to fast ego-motion and rapid scene changes, while the latter remains bounded by the teacher's single-pass output length and thus provides only a limited supervision horizon. A natural question is: can the teacher itself be extended via AR rollout to provide unbounded-horizon supervision at bounded memory cost? The key difficulty is that a standard teacher drifts under its own predictions, contaminating the supervision it provides. Our key insight is to make the teacher rollout-capable, ensuring reliable supervision from its own AR rollouts. This is instantiated as HorizonDrive, an anti-drifting training-and-distillation framework for AR driving simulation. First, scheduled rollout recovery (SRR) trains the base model to reconstruct ground-truth future clips from prediction-corrupted histories, yielding a teacher that remains stable across long AR rollouts. Second, the rollout-capable teacher is extended via AR rollout, providing long-horizon distribution-matching supervision under bounded memory, while a short-window student aligns to it with teacher rollout DMD (TRD) for efficient real-time deployment. HorizonDrive natively supports minute-scale AR rollout under bounded memory; on nuScenes, HorizonDrive reduces FID by 52% and FVD by 37%, and lowers ARE and DTW by 21% and 9% relative to the strongest long-horizon streaming baselines, while remaining competitive with single-pass driving video generators.
>
---
#### [replaced 037] Not All Tasks Quantize Equally: Fisher-Guided Quantization for Visual Geometry Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.15828](https://arxiv.org/pdf/2605.15828)**

> **作者:** Yipu Zhang; Jintao Cheng; Weilun Feng; Jiehao Luo; Chuanguang Yang; Zhulin An; Yongjun Xu; Wei Zhang
>
> **摘要:** Feed-forward 3D reconstruction models, represented by Visual Geometry Grounded Transformer (VGGT), jointly predict multiple visual geometry tasks such as depth estimation, camera pose prediction, and point cloud reconstruction in a single forward pass. They have been widely adopted in 3D vision applications, but their billion-scale parameters bring substantial memory and computation overhead, posing challenges for on-device deployment. Post-Training Quantization (PTQ) is an effective technique to reduce this overhead. Existing PTQ methods for feed-forward 3D models mainly focus on handling heavy-tailed activation distributions and constructing diverse calibration datasets. However, we observe that feed-forward 3D models predict multiple geometric attributes through a shared backbone, where different transformer blocks and hidden channels contribute distinctly to each task, resulting in substantially different sensitivities to quantization errors across tasks, blocks, and channels. Consequently, treating all tasks equally over-emphasizes insensitive tasks and causes significant accuracy loss on the sensitive ones. To address this issue, we propose Fisher-Guided Quantization (FGQ) for feed-forward 3D reconstruction models. Specifically, FGQ uses the diagonal Fisher information matrix to quantify the different sensitivities across tasks, blocks, and channels, and incorporates these sensitivities into the Learnable Affine Transformation during calibration to better preserve the channels and blocks most critical to each task. Extensive experiments across camera pose estimation, point map reconstruction, and depth estimation show that FGQ consistently outperforms state-of-the-art quantization baselines on VGGT, achieving up to 39% relative improvement under the 4-bit quantization. Code is available at this https URL.
>
---
#### [replaced 038] VISD: Enhancing Video Reasoning via Structured Self-Distillation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.06094](https://arxiv.org/pdf/2605.06094)**

> **作者:** Hao Lin; Kunyang Lv; Xu Jiang; Jingqi Tian; Zhongjing Du; Jiayu Ding; Qiaoman Zhang; Hongbo Jin
>
> **摘要:** Training VideoLLMs for complex reasoning remains challenging due to sparse sequence level rewards and the lack of fine grained credit assignment over long, temporally grounded reasoning trajectories. While reinforcement learning with verifiable rewards (RLVR) provides reliable supervision, it fails to capture token level contributions, leading to inefficient learning. Conversely, existing self distillation methods offer dense supervision but lack structure and diagnostic specificity, and often interact unstably with reinforcement learning. In this work, we propose VISD, a structured self distillation framework that introduces diagnostically meaningful privileged information for video reasoning. VISD employs a video aware judge model to decompose reasoning quality into multiple dimensions, including answer correctness, logical consistency, and spatio-temporal grounding, and uses this structured feedback to guide a teacher policy for token level supervision. To stably integrate dense supervision with RL, we introduce a direction magnitude decoupling mechanism, where rollout level advantages computed from rewards determine update direction, while structured privileged signals modulate token level update magnitudes. This design enables semantically aligned and fine grained credit assignment, improving both reasoning faithfulness and training efficiency. Additionally, VISD incorporates curriculum scheduling and EMA based teacher stabilization to support robust optimization over long video sequences. Experiments on diverse benchmarks show that VISD consistently outperforms strong baselines, improving answer accuracy and spatio temporal grounding quality. Notably, VISD reaches these gains with nearly 2x faster convergence in optimization steps, highlighting the effectiveness of structured self supervision in improving both performance and sample efficiency for VideoLLMs.
>
---
#### [replaced 039] Benchmarking and Enhancing VLM for Compressed Image Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20901](https://arxiv.org/pdf/2512.20901)**

> **作者:** Zifu Zhang; Tongda Xu; Siqi Li; Shengxi Li; Yue Zhang; Mai Xu; Yan Wang
>
> **备注:** The paper is accepted by ICML 2026
>
> **摘要:** With the rapid development of Vision-Language Models (VLMs) and the growing demand for their applications, efficient compression of the image inputs has become increasingly important. Existing VLMs predominantly digest and understand high-bitrate compressed images, while their ability to interpret low-bitrate compressed images has yet to be explored by far. In this paper, we introduce the first comprehensive benchmark to evaluate the ability of VLM against compressed images, varying existing widely used image codecs and diverse set of tasks, encompassing over one million compressed images in our benchmark. Next, we analyse the source of performance gap, by categorising the gap from a) the information loss during compression and b) generalisation failure of VLM. We visualize these gaps with concrete examples and identify that for compressed images, only the generalization gap can be mitigated. Finally, we propose a universal VLM adaptor to enhance model performance on images compressed by existing codecs. Consequently, we demonstrate that a single adaptor can improve VLM performance across images with varying codecs and bitrates by 10%-30%. We believe that our benchmark and enhancement method provide valuable insights and contribute toward bridging the gap between VLMs and compressed images. The source code is available at this https URL.
>
---
#### [replaced 040] 4DThinker: Thinking with 4D Imagery for Dynamic Spatial Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.05997](https://arxiv.org/pdf/2605.05997)**

> **作者:** Zhangquan Chen; Manyuan Zhang; Xinlei Yu; Xiang An; Bo Li; Xin Xie; ZiDong Wang; Mingze Sun; Shuang Chen; Hongyu Li; Xiaobin Hu; Ruqi Huang
>
> **备注:** 21 pages, 16 figures
>
> **摘要:** Dynamic spatial reasoning from monocular video is essential for bridging visual intelligence and the physical world, yet remains challenging for vision-language models (VLMs). Prior approaches either verbalize spatial-temporal reasoning entirely as text, which is inherently verbose and imprecise for complex dynamics, or rely on external geometric modules that increase inference complexity without fostering intrinsic model capability. In this paper, we present 4DThinker, the first framework that enables VLMs to "think with 4D" through dynamic latent mental imagery, i.e., internally simulating how scenes evolve within the continuous hidden space. Specifically, we first introduce a scalable, annotation-free data generation pipeline that synthesizes 4D reasoning data from raw videos. We then propose Dynamic-Imagery Fine-Tuning (DIFT), which jointly supervises textual tokens and 4D latents to ground the model in dynamic visual semantics. Building on this, 4D Reinforcement Learning (4DRL) further tackles complex reasoning tasks via outcome-based rewards, restricting policy gradients to text tokens to ensure stable optimization. Extensive experiments across multiple dynamic spatial reasoning benchmarks demonstrate that 4DThinker consistently outperforms strong baselines and offers a new perspective toward 4D reasoning in VLMs. Our code is available at this https URL.
>
---
#### [replaced 041] Imagine2Real: Towards Zero-shot Humanoid-Object Interaction via Video Generative Priors
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人形机器人与物体交互任务，解决3D数据稀缺和迁移复杂问题。提出Imagine2Real框架，通过4D点轨迹和关键点追踪实现零样本物理部署。**

- **链接: [https://arxiv.org/pdf/2605.22272](https://arxiv.org/pdf/2605.22272)**

> **作者:** Jiahe Chen; ZiRui Wang; Feiyu Jia; Xiao Chen; Xiaojie Niu; Weishuai Zeng; Tianfan Xue; Xiaowei Zhou; Jiangmiao Pang; Jingbo Wang
>
> **摘要:** Whole-body Humanoid-Object Interaction (HOI) is bottlenecked by the scarcity of high-fidelity 3D data. While video generative priors offer a promising alternative, existing methods suffer from \textit{Representation Misalignment} due to their reliance on geometric priors (e.g., explicit CAD models), and \textit{Retargeting Complexity} arising from intensive morphing and morphological mismatch. We propose Imagine2Real, a zero-shot HOI framework for flexible, geometry-free interaction. To resolve misalignment, we formulate robot and object motions as unified 4D point trajectories. To overcome retargeting complexity, our Keypoints Tracker tracks only sparse critical points (base, hands, and object), entirely bypassing the error-amplifying retargeting process. To maintain natural gaits despite these sparse signals, we utilize the latent space of a Behavior Foundation Model (BFM) as the tracker's search domain. Using a progressive training strategy, Imagine2Real learns robust behaviors with simple tracking rewards, enabling zero-shot physical deployment within a motion capture(mocap) system.
>
---
#### [replaced 042] The Double Dilemma in Multi-Task Radiology Report Generation: A Gradient Dynamics Analysis and Solution
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于多任务放射报告生成任务，解决线性标量化策略在平衡临床监督与报告平滑性上的不足。提出CAME-Grad优化器，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22635](https://arxiv.org/pdf/2605.22635)**

> **作者:** Erjian Zhang; Yatong Hao; Liejun Wang; Zhiqing Guo
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While multi-task learning based automatic radiology report generation (RRG) is widely adopted to ensure clinical consistency, most focus on architectural designs yet remain limited to coarse linear scalarization strategies. These strategies cannot effectively balance the hard constraints of discriminative clinical supervision with the smoothness requirements of report generation. To address these problems, we analyze the failure mechanism of linear scalarization from the perspective of gradient dynamics, utilizing the stochastic differential equation (SDE) framework to characterize it as a "Double Dilemma" of drift term deviation and diffusion term decay. Based on this, we propose a backbone-agnostic optimizer named Conflict-Averse Magnitude-Enhanced Gradient Descent (CAME-Grad). Through conflict-averse direction rectification and magnitude-enhanced energy injection, the algorithm not only ensures geometric validity, but also avoids local optimal solutions. Then, the adaptive gradient fusion mechanism is used to establish a dynamic balance between the theoretical optimal direction and the task-specific inductive bias. Experiments show that as a universal plug-and-play optimizer, CAME-Grad brings substantial and consistent improvements across eight diverse RRG methods, elevating overall clinical efficacy performance by an average of 2.3% on MIMIC-CXR and 1.9% on IU X-Ray. Our code is available at this https URL.
>
---
#### [replaced 043] Using Ensemble Diffusion to Estimate Uncertainty for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决轨迹规划中的不确定性问题。提出EnDfuser系统，利用扩散模型生成多候选轨迹，提升决策的不确定性感知能力。**

- **链接: [https://arxiv.org/pdf/2506.00560](https://arxiv.org/pdf/2506.00560)**

> **作者:** Florian Wintel; Sigmund H. Høeg; Gabriel Kiss; Frank Lindseth
>
> **备注:** Accepted at NLDL 2026
>
> **摘要:** End-to-end planning systems for autonomous driving are rapidly improving, especially in closed-loop simulation environments like CARLA. Many such driving systems either do not consider uncertainty as part of the plan itself or obtain it by using specialized representations that do not generalize. In this paper, we propose EnDfuser, an end-to-end driving system that uses a diffusion model as the trajectory planner. EnDfuser effectively leverages complex perception information like fused camera and LiDAR features, through combining attention pooling and trajectory planning into a single diffusion transformer module. Instead of committing to a single plan, EnDfuser produces a distribution of candidate trajectories (128 for our case) from a single perception frame through ensemble diffusion. By observing the full set of candidate trajectories, EnDfuser provides interpretability for uncertain, multimodal future trajectory spaces. Using this information we design a simplistic safety-rule that improves the system's driving score by 1.7% on the LAV benchmark. Our findings suggest that ensemble diffusion, used as a drop-in replacement for traditional point-estimate trajectory planning modules, can contribute to an uncertainty-aware decision making process in End-to-End driving policies by modeling the uncertainty of the posterior trajectory distribution.
>
---
#### [replaced 044] VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking-with-Videos
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.07801](https://arxiv.org/pdf/2602.07801)**

> **作者:** Wenqi Liu; Yunxiao Wang; Shijie Ma; Meng Liu; Qile Su; Tianke Zhang; Haonan Fan; Changyi Liu; Kaiyu Jiang; Jiankang Chen; Kaiyu Tang; Bin Wen; Fan Yang; Tingting Gao; Han Li; Yinwei Wei; Xuemeng Song
>
> **备注:** ICML 2026
>
> **摘要:** In long-video understanding, conventional uniform frame sampling often fails to capture key visual evidence, leading to degraded performance and increased hallucinations. To address this, recent agentic thinking-with-videos paradigms have emerged, adopting a localize-clip-answer pipeline in which the model actively identifies relevant video segments, performs dense sampling within those clips, and then produces answers. However, existing methods remain inefficient, suffer from weak localization, and adhere to rigid workflows. To solve these issues, we propose VideoTemp-o3, a unified agentic thinking-with-videos framework that jointly models video grounding and question answering. VideoTemp-o3 exhibits strong localization capability, supports on-demand clipping, and can refine inaccurate localizations. Specifically, in the supervised fine-tuning stage, we design a unified masking mechanism that encourages exploration while preventing noise. For reinforcement learning, we introduce dedicated rewards to mitigate reward hacking. Besides, from the data perspective, we develop an effective pipeline to construct high-quality long video grounded QA data, along with a corresponding benchmark for systematic evaluation across various video durations. Experimental results demonstrate that our method achieves remarkable performance on both long video understanding and grounding.
>
---
#### [replaced 045] WildTableBench: Benchmarking Multimodal Foundation Models on Table Understanding In the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.01018](https://arxiv.org/pdf/2605.01018)**

> **作者:** Junzhe Huang; Xiaoxiao Sun; Yan Yang; Yuxuan Hou; Ruotian Zhang; Sirui Li; Hehe Fan; Serena Yeung-Levy; Xin Yu
>
> **摘要:** Using multimodal foundation models to analyze table images is a high-value yet challenging application in consumer and enterprise scenarios. Despite its importance, current evaluations rely largely on structured-text tables or clean rendered images, leaving the visual complexity of in-the-wild table images underexplored. Such images feature varied layouts and diverse domains that demand sophisticated structural perception and numerical reasoning. To bridge this gap, we introduce WildTableBench, the first question-answering benchmark for naturally occurring table images from real-world settings. WildTableBench comprises 402 high-information-density table images collected from online forums and websites across diverse domains, together with 928 manually annotated and verified questions spanning 17 subtypes across five categories. We evaluate 21 frontier proprietary and open-source multimodal foundation models on this benchmark. Only one model exceeds 50% accuracy, while all remaining models range from 4.1% to 49.9%. We further conduct diagnostic analyses to characterize model failures and reveal persistent weaknesses in structural perception and reasoning. These results and analyses provide useful insights into current model capabilities and establish WildTableBench as a valuable diagnostic benchmark for table image understanding. Dataset: this https URL Code: this https URL Leaderboard: this https URL
>
---
#### [replaced 046] Towards Brain MRI Foundation Models for the Clinic: Findings from the FOMO25 Challenge
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.11679](https://arxiv.org/pdf/2604.11679)**

> **作者:** Asbjørn Munk; Stefano Cerri; Vardan Nersesjan; Christian Hedeager Krag; Jakob Ambsdorf; Pablo Rocamora García; Julia Machnio; Peirong Liu; Suhyun Ahn; Nasrin Akbari; Yasmina Al Khalil; Kimberly Amador; Sina Amirrajab; Tal Arbel; Meritxell Bach Cuadra; Ujjwal Baid; Bhakti Baheti; Jaume Banus; Kamil Barbierik; Christoph Brune; Yansong Bu; Baptiste Callard; Yuhan Chen; Cornelius Crijnen; Corentin Dancette; Peter Drotar; Prasad Dutande; Nils D. Forkert; Saurabh Garg; Jakub Gazda; Matej Gazda; Benoît Gérin; Partha Ghosh; Weikang Gong; Pedro M. Gordaliza; Sam Hashemi; Tobias Heimann; Fucang Jia; Jiexin Jiang; Emily Kaczmarek; Chris Kang; Seung Kwan Kang; Mohammad Khazaei; Julien Khlaut; Petros Koutsouvelis; Jae Sung Lee; Yuchong Li; Mengye Lyu; Mingchen Ma; Anant Madabhushi; Klaus H. Maier-Hein; Pierre Manceron; Andrés Martínez Mora; Moona Mazher; Felix Meister; Nataliia Molchanova; Steven A. Niederer; Leonard Nürnberg; Jinah Park; Abdul Qayyum; Jonas Richiardi; Antoine Saporta; Branislav Setlak; Ning Shen; Justin Szeto; Constantin Ulrich; Puru Vaish; Vibujithan Vigneshwaran; Leroy Volmer; Zihao Wang; Siqi Wei; Anthony Winder; Jelmer M. Wolterink; Maxence Wynen; Chang Yang; Si Young Yie; Mostafa Mehdipour Ghazi; Akshay Pai; Espen Jimenez Solem; Sebastian Nørgaard Llambias; Mikael Boesen; Michael Eriksen Benros; Juan Eugenio Iglesias; Mads Nielsen
>
> **摘要:** Clinical deployment of automated brain MRI analysis faces a fundamental challenge: clinical data is heterogeneous and noisy, and high-quality labels are prohibitively costly to obtain. Self-supervised learning (SSL) can address this by leveraging the vast amounts of unlabeled data produced in clinical workflows to train robust \textit{foundation models} that adapt out-of-domain with minimal supervision. However, the development of foundation models for brain MRI has been limited by small pretraining datasets and in-domain benchmarking focused on high-quality, research-grade data. To address this gap, we organized the FOMO25 challenge as a satellite event at MICCAI 2025. FOMO25 provided participants with a large pretraining dataset, FOMO60K, and evaluated models on data sourced directly from clinical workflows in few-shot and out-of-domain settings. Tasks covered infarct classification, meningioma segmentation, and brain age regression, and considered both models trained on FOMO60K (method track) and any data (open track). Nineteen foundation models from sixteen teams were evaluated using a standardized containerized pipeline. Results show that (a) self-supervised pretraining improves generalization on clinical data under domain shift, with the strongest models trained \textit{out-of-domain} surpassing supervised baselines trained \textit{in-domain}. (b) No single pretraining objective benefits all tasks: MAE favors segmentation, hybrid reconstruction-contrastive objectives favor classification, and (c) strong performance was achieved by small pretrained models, and improvements from scaling model size and training duration did not yield reliable benefits.
>
---
#### [replaced 047] Broken Memories: Detecting and Mitigating Memorization in Diffusion Models with Degraded Generations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.22050](https://arxiv.org/pdf/2605.22050)**

> **作者:** Yuanmin Huang; Mi Zhang; Chen Chen; Feifei Li; Geng Hong; Xiaoyu You; Min Yang
>
> **备注:** KDD 2026, extended version
>
> **摘要:** While diffusion models excel at generating high-quality images, their tendency to memorize training data poses significant privacy and copyright risks. In this work, we for the first time identify that memorization induces internal numerical instability, often manifesting as visually ``broken'' artifacts. Inspired by stability analysis in numerical methods, we introduce empirical stability regions based on latent update norms to quantitatively characterize stable behavior during generation. Leveraging this, we propose a principled, on-the-fly framework for step-wise detection and adaptive mitigation. Our approach suppresses memorization without altering prompts or guidance, thereby preserving semantic fidelity and image quality. Extensive experiments on Stable Diffusion 1.4 demonstrate that our method achieves an AUC $>0.999$ detection performance and a $0.0\%$ memorization rate after mitigation with negligible overhead ($\approx0.01$s per image).
>
---
#### [replaced 048] How Far Are We from Generating Missing Modalities with Foundation Models?
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态生成任务，旨在解决缺失模态重建问题。通过分析现有模型的不足，提出一种新的框架以提升生成质量。**

- **链接: [https://arxiv.org/pdf/2506.03530](https://arxiv.org/pdf/2506.03530)**

> **作者:** Guanzhou Ke; Bo Wang; Guoqing Chao; Weiming Hu; Shengfeng He
>
> **备注:** T-PAMI
>
> **摘要:** Multimodal foundation models have demonstrated impressive capabilities across diverse tasks. However, their potential as plug-and-play solutions for missing modality reconstruction remains underexplored. To bridge this gap, we identify and formalize three potential paradigms for missing modality reconstruction, and perform a comprehensive evaluation across these paradigms, covering 42 model variants in terms of reconstruction accuracy and adaptability to downstream tasks. Our analysis reveals that current foundation models often fall short in two critical aspects: (i) fine-grained semantic extraction from the available modalities, and (ii) robust validation of generated modalities. These limitations lead to suboptimal and, at times, misaligned generations. To address these challenges, we propose an agentic framework tailored for missing modality reconstruction. This framework dynamically formulates modality-aware mining strategies based on the input context, facilitating the extraction of richer and more discriminative semantic features. In addition, we introduce a self-refinement mechanism, which iteratively verifies and enhances the quality of generated modalities through internal feedback. Experimental results show that our method reduces FID for missing image reconstruction by at least 14\% and MER for missing text reconstruction by at least 10\% compared to baselines. Code are released at: this https URL.
>
---
#### [replaced 049] Investigating Robot Control Policy Learning for Autonomous X-ray-guided Spine Procedures
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于机器人控制策略学习任务，旨在解决X射线引导下脊柱手术的自主操作问题。通过仿真与数据训练，探索模仿学习在稀疏输入场景中的应用与挑战。**

- **链接: [https://arxiv.org/pdf/2511.03882](https://arxiv.org/pdf/2511.03882)**

> **作者:** Florence Klitzner; Blanca Inigo; Benjamin D. Killeen; Lalithkumar Seenivasan; Michelle Song; Axel Krieger; Mathias Unberath
>
> **摘要:** Imitation learning-based robot control policies are enjoying renewed interest in video-based robotics. However, it remains unclear whether this approach applies to X-ray-guided procedures, such as spine instrumentation, with sparse inputs. We examine the feasibility, opportunities and challenges for imitation policy learning in bi-plane-guided cannula insertion. We develop an in silico sandbox for scalable, automated simulation of X-ray-guided spine procedures with a high degree of realism. We curate a dataset of correct trajectories and corresponding bi-planar X-ray sequences that emulate the stepwise alignment of providers. We then train imitation learning policies for planning and open-loop control that iteratively align a cannula in a vertebroplasty setting solely based on visual information. This precisely controlled setup offers insights into limitations and capabilities of this method. Our policy succeeded on the first attempt in 68.5% of cases, maintaining safe intra-pedicular trajectories across diverse vertebral levels. The policy transferred to complex anatomy, including fractures, as well as varied anatomies and initializations. Rollouts on real X-ray indicate that partial sim-to-real transfer with plausible trajectories is possible. While these preliminary results are promising, we also identify limitations, especially in entry point precision. The current results present a clear benchmark for future efforts, while with more robust priors and domain knowledge, such models may provide a foundation for future efforts toward lightweight and CT-free robotic intra-operative spinal navigation.
>
---
#### [replaced 050] Progressive $\mathcal{J}$-Invariant Self-supervised Learning for Low-Dose CT Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.14180](https://arxiv.org/pdf/2601.14180)**

> **作者:** Yichao Liu; Zongru Shao; Yueyang Teng; Junwen Guo
>
> **摘要:** Self-supervised learning has been increasingly investigated for low-dose computed tomography (LDCT) image denoising, as it alleviates the dependence on paired normal-dose CT (NDCT) data, which are often difficult to collect. However, many existing self-supervised blind-spot denoising methods suffer from training inefficiencies and suboptimal performance due to restricted receptive fields. To mitigate this issue, we propose a novel Progressive $\mathcal{J}$-invariant Learning that maximizes the use of $\mathcal{J}$-invariant to enhance LDCT denoising performance. We introduce a step-wise blind-spot denoising mechanism that enforces conditional independence in a progressive manner, enabling more fine-grained learning for denoising. Furthermore, we explicitly inject a combination of controlled Gaussian and Poisson noise during training to regularize the denoising process and mitigate overfitting. Extensive experiments on the Mayo LDCT dataset demonstrate that the proposed method consistently outperforms existing self-supervised approaches and achieves performance comparable to, or better than, several representative supervised denoising methods.
>
---
#### [replaced 051] BVI-RLV: A Fully Registered Dataset for Low-Light Video Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.03535](https://arxiv.org/pdf/2407.03535)**

> **作者:** Ruirui Lin; Guoxi Huang; Joanne Lin; Qi Sun; Alexandra Malyugina; David R Bull; Nantheera Anantrasirichai
>
> **备注:** arXiv admin note: text overlap with arXiv:2402.01970
>
> **摘要:** Low-light videos often exhibit spatiotemporally incoherent noise, compromising visibility and degrading performance in computer vision applications. A major challenge for enhancing such content using deep learning lies in the scarcity of pixel-aligned, high-quality training data. We introduce BVI-RLV, a fully registered low-light video dataset comprising over 30k paired frames from 40 diverse scenes under two low-light conditions, each aligned with normal-light ground truth. Unlike existing datasets that rely on neutral density (ND) filters or suffer from misalignment issues, BVI-RLV achieves sub-pixel registration for 99.24% of data at full HD resolution across dynamic motion scenarios using a motorized dolly and image-based refinement. The dataset covers a wide range of motion types and realistic temporal noise. We also provide baseline implementations using four representative architectures: Convolutional Neural Network (CNN), Transformer, State Space Model (Mamba), and Diffusion Model (DM). Experiments demonstrate that registration is crucial for supervised learning, yielding up to 5.85 dB PSNR improvement compared to unregistered training. Models trained on BVI-RLV outperform those trained on existing datasets in cross-dataset evaluations, achieving superior performance even in real-world outdoor scenes. Our dataset is publicly available at this https URL.
>
---
#### [replaced 052] DFIR-DETR: Frequency-Domain Iterative Refinement and Dynamic Feature Aggregation for Small Object Detection
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.07078](https://arxiv.org/pdf/2512.07078)**

> **作者:** Bo Gao; Jingcheng Tong; Xingsheng Chen; Han Yu; Zichen Li
>
> **摘要:** Small object detection in complex scenes exposes a fundamental tension in neural network design: backbone attention distributes computation uniformly regardless of content, pyramid necks inflate activation magnitudes during upsampling without norm compensation, and bottleneck convolutions progressively smooth high-frequency edge components through accumulated spatial filtering. In response, we develop DFIR-DETR by tracing each proposed module back to a specific, measurable deficiency in the RT-DETR baseline: uniform attention that ignores spatial complexity, norm drift that destabilises upsampled features, and spatial convolutions that progressively suppress the high-frequency components small objects depend on. On NEU-DET and VisDrone, DFIR-DETR achieves 92.9% and 51.6% mAP50 with only 11.7M parameters and 47.2 GFLOPs, demonstrating consistent gains across two qualitatively different detection domains.
>
---
#### [replaced 053] A solution to generalized learning from small training sets found in infant repeated visual experiences of individual objects
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.15060](https://arxiv.org/pdf/2510.15060)**

> **作者:** Frangil Ramirez; Elizabeth Clerkin; David J. Crandall; Linda B. Smith
>
> **备注:** 28 pages, 7 figures, 3 tables
>
> **摘要:** One-year-old infants rapidly form and generalize categories of the everyday objects they encounter. Here we provide evidence on infants daily-life visual experiences for 8 early-learned object categories. Using a corpus of infant head-camera images recorded at mealtimes (87 mealtimes captured by 14 infants), we measure the frequency of the unique instances of each category and the variability of the visual experiences of each instance. The distribution of instances is highly skewed, containing, for each infant and category, many images of the same few objects along with fewer images of other instances. Graph theoretic measures of the similarity structure for individual categories reveal a lumpy mix of high similarity and high variability, organized into multiple but interconnected clusters of high-similarity images. In computational experiments, we show that artificially-created training sets characterized by a lumpy distribution of similarities support generalization to novel instances after very few training experiences. We discuss implications for visual object recognition, and for learning more generally, by both humans and machines.
>
---
#### [replaced 054] A drone-based framework for coral habitat mapping via weakly supervised segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.18958](https://arxiv.org/pdf/2508.18958)**

> **作者:** Matteo Contini; Victor Illien; Sylvain Poulain; Serge Bernard; Julien Barde; Sylvain Bonhommeau; Alexis Joly
>
> **备注:** Extended journal version of "The Point is the Mask: Scaling coral reef segmentation with weak supervision"
>
> **摘要:** Obtaining pixel-level annotations over large spatial extents remains a major bottleneck for deploying machine learning in ecological applications. Here we present a multi-scale weakly supervised semantic segmentation (WSSS) framework that enables training high-resolution segmentation models from dense, classification-based outputs. Our method combines fine-scale, multi-label predictions from underwater imagery with broad-coverage aerial data. We convert these point-level classifications into coarse supervision masks that can be used to train a semantic segmentation model on Unmanned Aerial Vehicle (UAV) orthophotos. A second training step using the model's own refined predictions is then used to further improve spatial accuracy without requiring additional annotations. We demonstrate the approach on coral reef imagery, enabling large-area segmentation of coral morphotypes and illustrating its flexibility in integrating new classes. The final model achieves 86.07% pixel accuracy and 52.23% mean Intersection over Union (mIoU) on manually annotated reef zones, demonstrating that accurate large-scale coral segmentation can be obtained without pixel-level annotations. By bridging image classification and segmentation across scales and modalities, this method provides an efficient solution for deploying segmentation models in settings where annotations are unavailable and opens opportunities for scalable, efficient monitoring in ecology and beyond.
>
---
#### [replaced 055] PipeMFL-240K: A Large-scale Dataset and Benchmark for Object Detection in Pipeline Magnetic Flux Leakage Imaging
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.07044](https://arxiv.org/pdf/2602.07044)**

> **作者:** Tianyi Qu; Songxiao Yang; Haolin Wang; Huadong Song; Xiaoting Guo; Wenguang Hu; Guanlin Liu; Honghe Chen; Yafei Ou
>
> **备注:** Accepted by ACM KDD 2026 Datasets and Benchmarks Track
>
> **摘要:** Pipeline integrity is critical to industrial safety and environmental protection, with Magnetic Flux Leakage (MFL) detection being a primary non-destructive testing technology. Despite the promise of deep learning for automating MFL interpretation, progress toward reliable models has been constrained by the absence of a large-scale public dataset and benchmark, making fair comparison and reproducible evaluation difficult. We introduce \textbf{PipeMFL-240K}, a large-scale, meticulously annotated dataset and benchmark for complex object detection in pipeline MFL pseudo-color images. PipeMFL-240K reflects real-world inspection complexity and poses several unique challenges: (i) an extremely long-tailed distribution over \textbf{12} categories, (ii) a high prevalence of tiny objects that often comprise only a handful of pixels and (iii) substantial intra-class variability. The dataset contains \textbf{249,320} images and \textbf{200,020} high-quality bounding-box annotations, collected from 12 pipelines spanning approximately \textbf{1,530} km. Extensive experiments are conducted with state-of-the-art object detectors to establish baselines. Results show that modern detectors still struggle with the intrinsic properties of MFL data, highlighting considerable headroom for improvement, while PipeMFL-240K provides a reliable and challenging testbed to drive future research. As the first public dataset and the first benchmark of this scale and scope for pipeline MFL inspection, it provides a critical foundation for efficient pipeline diagnostics as well as maintenance planning and is expected to accelerate algorithmic innovation and reproducible research in MFL-based pipeline integrity assessment.
>
---
#### [replaced 056] Quantum-Inspired Robust and Scalable SAR Object Classification
- **分类: quant-ph; cs.CV; physics.comp-ph**

- **链接: [https://arxiv.org/pdf/2604.25755](https://arxiv.org/pdf/2604.25755)**

> **作者:** Maximilian Scharf; Marco Trenti; Felix Bock; Padraig Davidson; Tobias Brosch; Benjamin Rodrigues de Miranda; Sigurd Huber; Timo Felser
>
> **备注:** 6 pages, 6 figures, EUSAR 2026 conference
>
> **摘要:** SAR image classification naturally has to deal with huge noise and a high dynamic range particularly requiring robust classification models. Additionally, the deployment of these models on edge devices, such as drones and military aircraft, requires a careful balance between model size and classification accuracy. This study explores the potential of tensor networks to meet these robustness requirements, specifically evaluating their resilience to data poisoning. Unlike previous works that concentrated on conventional neural networks for SAR object detection, this research focuses on the robustness and model reduction capabilities of tensor networks in object classification. Our findings indicate that tensor networks are adept at addressing both the challenges of robustness and the need for model efficiency, thereby contributing valuable insights to the ongoing discourse in radar applications and deep learning methodologies in general.
>
---
#### [replaced 057] Dynamic Weight-based Temporal Aggregation for Low-light Video Enhancement Under Extreme Noise
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.09450](https://arxiv.org/pdf/2510.09450)**

> **作者:** Ruirui Lin; Guoxi Huang; Nantheera Anantrasirichai
>
> **摘要:** Low-light video enhancement (LLVE) is challenging due to noise, low contrast, and color degradation. While learning-based methods enable fast inference, they often fail under heavy real-world noise because they do not sufficiently exploit long-term temporal cues. We propose DWTA-Net, a novel deep-learning recurrent LLVE framework with a recurrent design. DWTA-Net adopts an integrated two-stage architecture: Stage I restores local structure and color via multi-frame alignment for temporally consistent Mamba-based enhancement, while Stage II performs recurrent refinement using a novel dynamic weight-based temporal aggregation guided by optical flow, functioning as a recurrent denoiser that adapts to motion. We further introduce a texture-adaptive loss that preserves fine details in textured regions while suppressing noise in homogeneous areas. Experiments on real-world low-light footage show that DWTA-Net achieves stronger noise suppression and fewer artifacts, delivering superior visual quality compared with state-of-the-art methods.
>
---
#### [replaced 058] RT-NeRV: Rethinking Hybrid Neural Representations for Video via Residual Tokenization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.12401](https://arxiv.org/pdf/2403.12401)**

> **作者:** Yunjie Xu; Xiang Feng; Chengkai Wang; Alan Wee-Chung Liew; Xuefei Yin; Yanming Zhu
>
> **备注:** Under Review
>
> **摘要:** Neural Representations for Videos(NeRV) have emerged as a promising paradigm for video compression by representing videos as compact neural networks with efficient decoding. Hybrid NeRV methods further improve reconstruction quality through content adaptive embeddings, but still struggle to preserve fine details at low bitrates. A key limitation is that shallow residual support in formation, although highly beneficial for reconstruction, is costly to transmit in its continuous form and is therefore underutilized. In this paper, we rethink hybrid NeRV and present RT-NeRV, a residual tokenization framework for hybrid neural video representations. The core idea is to discretize shallow residual features and inter-frame residual cues into compact residual tokens, allowing informative reconstruction support to be transmitted efficiently and exploited by the decoder. To this end, we design a residual tokenizer together with a residual-aware codebook learning strategy that improves token utilization and stabilizes training. RT-NeRV can be readily integrated into modern hybrid NeRV hosts, consistently enhancing detail preservation, reconstruction quality, and bitrate quality trade-offs. Extensive experiments on video regression and related restoration tasks show that RT-NeRV outperforms strong hybrid NeRV baselines and remains competitive with recent INR based video compression methods. These results demonstrate that residual tokenization is an effective and complementary direction for advancing hybrid neural video representations
>
---
#### [replaced 059] NP-LoRA: Null Space Projection for Subject-Style LoRA Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11051](https://arxiv.org/pdf/2511.11051)**

> **作者:** Chuheng Chen; Xiaofei Zhou; Geyuan Zhang; Yong Huang
>
> **摘要:** Low-Rank Adaptation (LoRA) fusion enables the composition of subject and style representations for controllable generation without retraining. However, existing approaches primarily operate through weight-level merging, without explicitly modeling how independently trained LoRAs interact in the shared parameter space. We adopt a geometric perspective on LoRA fusion, interpreting content and style LoRAs as occupying overlapping, non-orthogonal low-rank subspaces, where such overlap can lead to conflicting parameter updates that affect generation quality. This observation motivates us to reformulate LoRA fusion not merely as parameter combination, but as a problem of controlling how updates from overlapping subspaces are combined. Based on this insight, we propose Null Space Projection LoRA (NP-LoRA), a training-free framework that employs projection as a fusion operator to explicitly modulate cross-LoRA interactions. Specifically, NP-LoRA uses principal directions of the style LoRA to define a projection subspace and projects the content LoRA onto the complementary subspace (i.e., the null space of the style LoRA), suppressing interference along dominant style directions while preserving complementary information. To avoid the overly aggressive suppression of hard projection, we further formulate soft projection as a regularized optimization problem that balances content preservation against style-subspace suppression. This objective admits a closed-form solution, yielding a projection operator controlled by a single parameter that continuously interpolates between linear merging and hard projection. Extensive experiments across multiple pretrained LoRA pairs show that NP-LoRA achieves more balanced content-style composition compared to strong baselines, without requiring retraining.
>
---
#### [replaced 060] Distill to Think, Foresee to Act: Cognitive-Physical Reinforcement Learning for Autonomous Driving
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.21139](https://arxiv.org/pdf/2605.21139)**

> **作者:** Yang Wu; Qiang Meng; Zhaojiang Liu; Youquan Liu; Jian Yang; Jin Xie
>
> **摘要:** Current end-to-end autonomous driving models are fundamentally constrained by the behavioral cloning ceiling of imitation learning. While reinforcement learning offers a path to smarter autonomy, it demands two missing pieces of infrastructure: (1) a cognitive foundation that understands traffic semantics and driving intent, and (2) a foresighted physical environment that can anticipate the consequences of candidate actions. To this end, we propose CoPhy, a CognitivePhysical reinforcement learning framework for autonomous driving. To distill to think, we distill VLM knowledge into the BEV encoder and then discard the VLM entirely, retaining cognitive ability at zero inference cost while releasing the cognitive channel as a pluggable interface for optional human language commands. To foresee to act, we build an auto-regressive BEV world model that explicitly predicts future semantic maps conditioned on candidate actions, serving as an interpretable physical sandbox from which safety metrics are directly derived. Built upon this dual infrastructure, we optimize the driving policy via GRPO with a novel dual-reward mechanism: a physical reward derived from BEV rollouts enforces hard safety constraints, while a cognitive reward from a language-aligned scorer ensures intent compliance. Extensive experiments demonstrate that CoPhy not only achieves state-of-the-art results on NAVSIM v1 and v2 benchmarks, but also enables safer driving via cognitively informed scene compliance and flexible intent control through user-defined language instructions.
>
---
#### [replaced 061] ForeSplat: Optimization-Aware Foresight for Feed-Forward 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.22020](https://arxiv.org/pdf/2605.22020)**

> **作者:** Yuke Li; Weihang Liu; Cheng Zhang; Yuefeng Zhang; Jiadi Cui; Zixuan Wang; Junran Ding; Haoyu Wu; Yujiao Shi; Jingyi Yu; Xin Lou
>
> **摘要:** Feed-forward 3D Gaussian Splatting models offer fast single-pass reconstruction,but scaling them to match per-scene optimization quality is fundamentally hindered by the scarcity of large-scale 3D annotations. A practical compromise is predict-then-refine,where post-prediction optimization compensates for the limited capacity of the feed-forward network. However,standard feed-forward 3DGS is trained solely for zero-step rendering error,ignoring whether its output constitutes a good initialization for the downstream optimizer. We present ForeSplat,an optimization-aware training framework that equips feed-forward 3DGS models to produce initializations explicitly designed for rapid,effective refinement. By offloading part of the scene-modeling burden to the optimizer,ForeSplat substantially reduces the capacity pressure on the feed-forward model,making high-quality reconstruction feasible even with compact networks. At its core is MetaGrad,a lightweight multi-anchor meta-gradient training rule that bypasses costly higher-order differentiation through the 3DGS optimizer. MetaGrad unrolls a short inner-loop refinement trajectory,samples anchor states,and back-propagates aggregated first-order gradients to the prediction head as a surrogate optimization-aware signal. This fine-tuning adds no inference cost and enables high-quality reconstruction within seconds after a few refinement steps. We instantiate ForeSplat on diverse backbones,including AnySplat,Pi3X,and a distilled variant tailored for edge deployment. Across all tested architectures,a ForeSplat-trained initialization converges in fewer refinement steps and reaches a higher peak reconstruction quality than its vanilla counterpart,even fully converged. The framework consistently bridges the gap between amortized prediction and per-scene optimization,establishing a practical path toward lightweight,high-fidelity 3D reconstruction.
>
---
#### [replaced 062] A European Multi-Center Breast Cancer MRI Dataset
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.00474](https://arxiv.org/pdf/2506.00474)**

> **作者:** Gustav Müller-Franzes; Lorena Escudero Sánchez; Nicholas Payne; Alexandra Athanasiou; Michael Kalogeropoulos; Aitor Lopez; Alfredo Miguel Soro Busto; Julia Camps Herrero; Nika Rasoolzadeh; Tianyu Zhang; Ritse Mann; Debora Jutz; Maike Bode; Christiane Kuhl; Yuan Gao; Wouter Veldhuis; Oliver Lester Saldanha; JieFu Zhu; Jakob Nikolas Kather; Daniel Truhn; Fiona J. Gilbert
>
> **摘要:** Early detection of breast cancer is critical for improving patient outcomes. While mammography remains the primary screening modality, magnetic resonance imaging (MRI) is increasingly recommended as a supplemental tool for women with dense breast tissue and those at elevated risk. However, the acquisition and interpretation of multiparametric breast MRI are time-consuming and require specialized expertise, limiting scalability in clinical practice. Artificial intelligence (AI) methods have shown promise in supporting breast MRI interpretation, but their development is hindered by the limited availability of large, diverse, and publicly accessible datasets. To address this gap, we present a publicly available, multi-centre breast MRI dataset collected across six clinical institutions in five European countries. The dataset comprises 741 examinations from women undergoing screening or diagnostic breast MRI and includes malignant, benign, and non-lesion cases. Data were acquired using heterogeneous scanners, field strengths, and acquisition protocols, reflecting real-world clinical variability. In addition, we report baseline benchmark experiments using a transformer-based model to illustrate potential use cases of the dataset and to provide reference performance for future methodological comparisons.
>
---
#### [replaced 063] A Robust Semantic Segmentation Pipeline for the CVPR 2026 8th UG2+ Challenge Track 2
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.22216](https://arxiv.org/pdf/2605.22216)**

> **作者:** Jinming Chai; Libo Yan; Licheng Jiao; Fang Liu
>
> **摘要:** This report presents our solution for the WeatherProof Dataset Challenge, namely CVPR 2026 8th UG2+ Challenge Track 2: Semantic Segmentation in Adverse Weather. For the semantic segmentation task under adverse weather conditions, we propose a semi-supervised segmentation pipeline. Our method is trained exclusively on the WeatherProof dataset, without using any additional external data. Specifically, we adopt UniMatch V2 as the baseline model and treat all degraded-weather images as unlabeled data for semi-supervised training, thereby fully exploiting the data distribution provided by the challenge. During inference, we further apply test-time augmentation to improve the robustness and segmentation accuracy of the final predictions. The code is publicly available at: this https URL.
>
---
#### [replaced 064] MedVIGIL: Evaluating Trustworthy Medical VLMs Under Broken Visual Evidence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.07919](https://arxiv.org/pdf/2605.07919)**

> **作者:** Hanqi Jiang; Junhao Chen; Mingyu Kang; Hyeokjae Kwon; Yi Pan; Lifeng Chen; Weihang You; Haozhen Gong; Ruiyu Yan; Jinglei Lv; Lin Zhao; Hui Ren; Quanzheng Li; Tianming Liu; Xiang Li
>
> **摘要:** Medical vision--language models (VLMs) are usually evaluated on intact image--question pairs, but trustworthy clinical use requires a stronger property: a model must recognise when the evidential basis for an answer has failed. We study this through silent failures under perturbed evidence, where a vision-required medical question is paired with a false premise, wording perturbation, knowledge-only rewrite, or ROI-corrupted image, yet the model returns a fluent non-refusal answer. We introduce medvigil, a 300-case evaluation suite drawn from four public medical VQA sources, supervised end to end by four board-certified radiologists: every gold answer, refusal option, candidate-answer set, paraphrase, false-premise trap, ROI box, and clinical risk tier is clinician-authored. Two attending radiologists annotate every case in parallel, a senior radiologist consolidates the released manifest, and a separate fourth radiologist independent of construction answers every probe to provide the human reference baseline. The release contains 2556 MCQ probes, 240 counterfactual triplets, physician-adjudicated risk-tier and answerability flags, ROI boxes, and a paired open-ended variant. We report seven correctness-conditioned audit metrics that summarise into the medvigil Composite Score (MCS), and audit 16 vision-capable models plus two text-only baselines. The independent radiologist scores MCS 83.3 at silent-failure rate 5.8%, leaving a 14.1-point composite headroom above the strongest audited model (Claude Opus 4.7 at 69.2). The benchmark and evaluation harness are publicly released.
>
---
#### [replaced 065] OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.06088](https://arxiv.org/pdf/2605.06088)**

> **作者:** Kunyi Li; Michael Niemeyer; Sen Wang; Stefano Gasperini; Nassir Navab; Federico Tombari
>
> **摘要:** Understanding open-vocabulary 3D scenes with Gaussian-based representations remains challenging due to fragmented and spatially inconsistent semantic predictions across multi-view observations. In this paper, we present OpenGaFF, a novel framework for open-vocabulary 3D scene understanding built upon 3D Gaussian Splatting. At the core of our method is a Gaussian Feature Field that models semantics as a continuous function of Gaussian geometry and appearance. By explicitly conditioning semantic predictions on geometric structure, this formulation strengthens the coupling between geometry and semantics, leading to improved spatial coherence across similar structures in 3D space. To further enforce object-level semantic consistency, we introduce a structured codebook that serves as a set of shared semantic primitives. Furthermore, a codebook-guided attention mechanism is proposed to retrieve language features via similarity matching between query embeddings and learned codebook entries, enabling robust open-vocabulary reasoning while reducing intra-object feature variance. Extensive experiments on standard 2D and 3D open-vocabulary benchmarks demonstrate that our method consistently outperforms prior approaches, achieving improved segmentation quality, stronger 3D semantic consistency and a semantically interpretable codebook that provides insight into the learned representation.
>
---
#### [replaced 066] Sparser Block-Sparse Attention via Token Permutation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于自然语言处理任务，旨在解决长序列中自注意力机制计算效率低的问题。通过引入令牌排列策略，提升块稀疏注意力的效率与效果。**

- **链接: [https://arxiv.org/pdf/2510.21270](https://arxiv.org/pdf/2510.21270)**

> **作者:** Xinghao Wang; Pengyu Wang; Dong Zhang; Chenkun Tan; Shaojun Zhou; Zhaoxiang Liu; Shiguo Lian; Fangxu Liu; Kai Song; Xipeng Qiu
>
> **备注:** ICML 2026
>
> **摘要:** Scaling the context length of large language models (LLMs) offers significant benefits but is computationally expensive. This expense stems primarily from the self-attention mechanism, whose $O(N^2)$ complexity with respect to sequence length presents a major bottleneck for both memory and latency. Fortunately, the attention matrix is often sparse, particularly for long sequences, suggesting an opportunity for optimization. Block-sparse attention has emerged as a promising solution that partitions sequences into blocks and skips computation for a subset of these blocks. However, the effectiveness of this method is highly dependent on the underlying attention patterns, which can lead to sub-optimal block-level sparsity. For instance, important key tokens for queries within a single block may be scattered across numerous other blocks, leading to computational redundancy. In this work, we propose Permuted Block-Sparse Attention (\textbf{PBS-Attn}), a plug-and-play method that leverages the permutation properties of attention to increase block-level sparsity and enhance the computational efficiency of LLM prefilling. We conduct comprehensive experiments on challenging real-world long-context datasets, demonstrating that PBS-Attn consistently outperforms existing block-sparse attention methods in model accuracy and closely matches the full attention baseline. Powered by our custom permuted-FlashAttention kernels, PBS-Attn achieves an end-to-end speedup of up to $2.75\times$ in long-context prefilling, confirming its practical viability. Code available at this https URL
>
---
#### [replaced 067] DocRevive: A Unified Pipeline for Document Text Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10077](https://arxiv.org/pdf/2604.10077)**

> **作者:** Kunal Purkayastha; Ayan Banerjee; Josep Llados; Umapada Pal
>
> **摘要:** In Document Understanding, the challenge of reconstructing damaged, occluded, or incomplete text remains a critical yet unexplored problem. Subsequent document understanding tasks can benefit from a document reconstruction process. In response, this paper presents a novel unified pipeline combining state-of-the-art Optical Character Recognition (OCR), advanced image analysis, masked language modeling, and diffusion-based models to restore and reconstruct text while preserving visual integrity. We create a synthetic dataset of 30{,}078 degraded document images that simulates diverse document degradation scenarios, setting a benchmark for restoration tasks. Our pipeline detects and recognizes text, identifies degradation with an occlusion detector, and uses an inpainting model for semantically coherent reconstruction. A diffusion-based module seamlessly reintegrates text, matching font, size, and alignment. To evaluate restoration quality, we propose a Unified Context Similarity Metric (UCSM), incorporating edit, semantic, and length similarities with a contextual predictability measure that penalizes deviations when the correct text is contextually obvious. Our work advances document restoration, benefiting archival research and digital preservation while setting a new standard for text reconstruction. The OPRB dataset and code are available at \href{this https URL}{Hugging Face} and \href{this https URL}{Github} respectively.
>
---
#### [replaced 068] Lost in the Folds: When Cross-Validation Is Not a Deep Ensemble for Uncertainty Estimation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.18329](https://arxiv.org/pdf/2605.18329)**

> **作者:** Tristan Kirscher; Markus Bujotzek; Yannick Kirchhoff; Maximilian Rokuss; Fabian Isensee; Kim-Celine Kahl; Balint Kovacs; Klaus Maier-Hein
>
> **备注:** Accepted for publication at MICCAI 2026
>
> **摘要:** Ensemble disagreement is widely used as a proxy for epistemic uncertainty in medical image segmentation. In practice, many studies form ensembles via K-fold cross-validation (CV), yet refer to them as ``deep ensembles'' (DE). Because CV members are trained on different data subsets, their disagreement mixes seed-driven variability with data-exposure effects, which can change how uncertainty should be interpreted. We audit recent segmentation uncertainty studies and find that terminology--implementation mismatches are common. We then compare a standard 5-fold CV ensemble to a 5-member DE (fixed training set, different random seeds) under otherwise identical configurations on three multi-rater segmentation datasets spanning three modalities. We evaluate uncertainty for calibration, failure detection, ambiguity modeling, and robustness under distribution shift. DE match segmentation accuracy while improving calibration and failure detection, whereas CV ensembles sometimes correlate more strongly with inter-rater variability on the studied datasets. Thus, ensemble construction should be chosen to match the research question: DE for reliability-oriented use (e.g., selective referral/failure detection) and CV ensembles as a proxy for ambiguity. We provide a lightweight nnU-Net modification enabling DE training within the default pipeline.
>
---
#### [replaced 069] Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07615](https://arxiv.org/pdf/2603.07615)**

> **作者:** Zongyu Guo; Jiajun He; Zhaoyang Jia; Xiaoyi Zhang; Jiahao Li; Xiao Li; Bin Li; José Miguel Hernández-Lobato; Yan Lu
>
> **备注:** ICML 2026
>
> **摘要:** Modern visual generative models acquire rich visual knowledge through large-scale training, yet existing visual representations (such as pixels, latents, or tokens) remain external to the model and cannot directly exploit this knowledge for compact storage or reuse. In this work, we introduce a new visual representation framework that encodes a signal as a function, which is parametrized by low-rank adaptations attached to a frozen visual generative model. Such implicit representations of visual signals, \textit{e.g.}, an 81-frame video, can further be hashed into a single compact vector, achieving strong perceptual video compression at extremely low bitrates. Beyond basic compression, the functional nature of this representation enables inference-time scaling and control, allowing additional refinement on the compression performance. More broadly, as the implicit representations directly act as a function of the generation process, this suggests a unified framework bridging visual compression and generation.
>
---
#### [replaced 070] Multi-SpatialMLLM: Multi-Frame Spatial Understanding with Multi-Modal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在解决MLLM在多帧空间理解上的不足。通过引入新数据集和框架，提升模型的多帧感知能力，并验证其在机器人等场景的应用效果。**

- **链接: [https://arxiv.org/pdf/2505.17015](https://arxiv.org/pdf/2505.17015)**

> **作者:** Runsen Xu; Weiyao Wang; Hao Tang; Xingyu Chen; Xiaodong Wang; Fu-Jen Chu; Matt Feiszli; Kevin J. Liang
>
> **备注:** CVPR 2026 Camera Ready. 27 pages. Project page: this https URL
>
> **摘要:** Multi-modal large language models (MLLMs) have rapidly advanced in visual tasks, yet their spatial understanding remains limited to single images, leaving them ill-suited for physical-world applications that require multi-frame reasoning. In this paper, we propose a framework to equip MLLMs with multi-frame spatial understanding by integrating fundamental spatial skills, including depth perception, visual correspondence, and dynamic perception. We design a novel data pipeline and collect the MultiSPA dataset of more than 27 million samples spanning diverse 3D and 4D scenes to enable training. Alongside MultiSPA, we introduce a comprehensive benchmark that tests a wide spectrum of spatial tasks under uniform metrics. Our resulting model, Multi-SpatialMLLM, achieves significant gains over baselines and proprietary systems, demonstrating scalable and generalizable multi-frame perception. We further observe multi-task benefits and emergent spatial capabilities in challenging scenarios, and showcase how our model can serve as a multi-frame reward annotator for robotics.
>
---
