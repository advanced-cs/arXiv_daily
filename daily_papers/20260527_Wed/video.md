# 计算机视觉 cs.CV

- **最新发布 141 篇**

- **更新 120 篇**

## 最新发布

#### [new 001] YOLO26-RipeLoc Lite: A lightweight architecture for tomato ripeness detection and picking point localization in greenhouse robotic harvesting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于温室番茄成熟度检测与采摘点定位任务，提出YOLO26-RipeLoc Lite模型，解决精准检测与定位问题。**

- **链接: [https://arxiv.org/pdf/2605.27129](https://arxiv.org/pdf/2605.27129)**

> **作者:** Rajmeet Singh; Manveen Kaur; Shahpour Alirezaee; Irfan Hussain
>
> **摘要:** In greenhouse tomato production, automated harvesting requires accurate detection of ripe tomatoes, ripeness classification, and precise picking-point localization for robotic end-effectors. This paper proposes YOLO26-RipeLoc Lite, a lightweight deep learning architecture based on YOLO26 for simultaneous detection, ripeness classification, and center-point localization of greenhouse tomatoes. The model introduces three modifications: (1) a Lightweight Feature Pyramid Network (LFPN) with depthwise separable convolutions for efficient multi-scale fusion, (2) a Ripeness-Aware Attention Module (RAAM) with dual pooling and a learnable ripeness bias vector for enhanced color-texture discrimination, and (3) a Compact Detection Head (CDH) with shared convolutions and an integrated center-point regression branch for direct grasp planning. The model is evaluated on a custom dataset of 1,500 images with 6,227 instances (3,566 ripe, 2,661 unripe) from the SILAL greenhouse, Abu Dhabi, UAE. YOLO26-RipeLoc Lite achieves mAP@0.5 of 92.9% (95.2% ripe, 90.6% unripe) with the highest precision (95.2%) among all evaluated architectures using only 2.38M parameters. Post-training BatchNorm pruning at 30% reduces parameters to ~1.8M with negligible accuracy loss. Ablation studies confirm that greenhouse-aware HSV augmentation provides the largest improvement (+2.02 pp mAP@50), backbone freezing achieves peak precision (93.8%), and 3-phase progressive unfreezing yields the best localization quality (mAP@50:95 of 64.6%). Comparisons with YOLOv8n/s, YOLO11n/s, YOLO12n/s, and YOLO26s confirm superior accuracy-efficiency: 2.9 pp higher precision than YOLO12n with 7.0% fewer parameters and integrated center-point localization for robotic end-effector guidance.
>
---
#### [new 002] Can Retrieval Heads See Images? Multimodal Retrieval Heads in Long-Context Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究多模态检索头在长文本-图像模型中的作用，解决跨模态证据定位问题。通过引入多模态检索方法，分析其稀疏性与重要性，并提升文档检索效果。**

- **链接: [https://arxiv.org/pdf/2605.27243](https://arxiv.org/pdf/2605.27243)**

> **作者:** Aaron Branson Cigres Li; Zhaowei Wang; Yu Zhao; Yiming Du; Haobo Li; Xiyu Ren; Ginny Wong; Simon See; Lishu Luo; Haodong Duan; Pasquale Minervini; Yangqiu Song
>
> **备注:** Work in Progress
>
> **摘要:** Large vision-language models increasingly rely on long-context modeling to reason over documents, hour-level videos, and long-horizon agent trajectories, requiring them to locate relevant evidence across interleaved text and images. Prior work has studied this behavior using retrieval heads in large language models, but its copy-based criterion does not directly apply when evidence appears in images. We introduce a multimodal retrieval head detection method that scores attention from question tokens to textual or visual evidence. With this method, we show that multimodal retrieval heads are sparse, intrinsic, and causally important: only 4.4-10.2% of attention heads account for 50% of the positive retrieval-score mass, and masking the top-5% selected heads drops MMLongBench-Doc from 48.2% to 5.7% and SlideVQA from 71.2% to 8.9%, while random-head masking is far less damaging. Further analysis shows that these heads are partly shared across modalities yet remain dynamic within each modality, with image retrieval heads changing more than text retrieval heads as context length and haystack modality change. Without further training, we find that these heads can also be used directly to rank visually rich documents: on MMDocIR, Qwen3-VL-8B selected-head scoring improves Recall@1 by 7.7/7.4 macro/micro points for page retrieval and 6.3/6.8 points for layout retrieval over the strongest reported baseline.
>
---
#### [new 003] LongCat-Video-Avatar 1.5 Technical Report
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决音频驱动视频生成的稳定性与工业部署问题。通过优化音频编码器和训练策略，提升唇同步、长期视频一致性及多场景适应性。**

- **链接: [https://arxiv.org/pdf/2605.26486](https://arxiv.org/pdf/2605.26486)**

> **作者:** Meituan LongCat Team; Xunliang Cai; Meng Cheng; Feng Gao; Zhe Kong; Jiamu Li; Le Li; Weiheng Li; Hongyu Liu; Shuai Tan; Xiaoming Wei; Tianyu Yang; Yong Zhang
>
> **备注:** Homepage: this https URL Github: this https URL
>
> **摘要:** Despite advances in audio-driven video generation, achieving commercial-grade stability remains challenging. We present LongCat-Video-Avatar 1.5, an upgraded open-source framework prioritizing systematic engineering and production-readiness over architectural novelty. By upgrading the audio encoder to Whisper Large and meticulously scaling our training recipes, v1.5 achieves accurate lip-synchronization, full-body temporal stability, and robust long-video generation with strict identity consistency. Through rigorous data curation and RLHF Training, the model readily generalizes to stylized domains such as anime and animals, and natively handles complex real-world conditions, such as multi-person interactions and object handling. Furthermore, addressing the practical demands of industrial deployment, we employ advanced step distillation to accelerate inference to an optimal 8 NFE, achieving a favorable trade-off between serving efficiency and visual fidelity. The superiority of our approach is validated through extensive quantitative metrics and a rigorous human evaluation conducted on a comprehensive benchmark of over 500 diverse test cases. Results show that v1.5 achieves competitive or superior performance compared to leading closed-source systems (e.g., HeyGen, OmniHuman 1.5, Kling Avatar 2.0) across human-likeness ratings and expert-level quality assessments on our benchmark. With its open-source release, LongCat-Video-Avatar 1.5 narrows the gap between academic research prototypes and commercial-grade deployment.
>
---
#### [new 004] MRT: Masked Region Transformer for Layered Image Generation and Editing at Scale
- **分类: cs.CV**

- **简介: 该论文提出MRT模型，解决大规模多层透明图像生成与编辑问题。通过统一任务框架和引入溢出画布层，实现高效、高质量的多层图像生成与编辑。**

- **链接: [https://arxiv.org/pdf/2605.27235](https://arxiv.org/pdf/2605.27235)**

> **作者:** Zhicong Tang; Zhao Zhang; Jingye Chen; Mohan Zhou; Yifan Pu; Yuchi Liu; Yalong Bai; Ethan Smith; Yuhui Yuan
>
> **备注:** CVPR 2026
>
> **摘要:** Layered image generation and editing is a fundamental capability that enables layer-wise reuse, editing, and composition of generated visual content, analogous to word-level editing in natural language. Despite its importance, this remains an underexplored area at scale. To address this gap, we present MRT, a 20B-parameter masked region diffusion model tailored for multi-layer transparent image generation and editing, trained on over 10M multilingual design samples spanning diverse aspect ratios and textual prompts. To fully leverage this scale, we make two key technical contributions. First, we unify three complementary tasks including text-to-layers, image-to-layers, and layers-to-layers within a shared masked region diffusion framework, where selective token masking enables flexible layer-wise generation and editing. Second, to enable overflow layer generation, we introduce an overflow-aware canvas layer that handles boundary inconsistencies and supports semi-transparent background synthesis, enabling complete editable layers extending beyond visible canvas boundaries. Additionally, we apply diffusion distillation to achieve 8-step, real-time multi-layer generation with minimal quality degradation. Extensive experiments demonstrate that our framework substantially outperforms prior state-of-the-art approaches, including various commercial systems, across all three tasks, establishing a new benchmark for multi-layer transparent image generation. Notably, our model significantly outperforms the concurrent Qwen-Image-Layered model in image-to-layers quality according to user-study results, while achieving 10-100\times faster inference and reducing activation GPU memory consumption by 50-90\% during image-to-layer inference.
>
---
#### [new 005] A Hybrid Vision-Language Architecture for Automated Defect Reasoning and Report Generation in Industrial Inspection
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于工业检测任务，解决缺陷定位与报告生成分离的问题。提出一种解耦架构，结合检测、编码和生成模块，提升报告质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.26533](https://arxiv.org/pdf/2605.26533)**

> **作者:** Malikussaid; Imad Gohar
>
> **备注:** 23 pages, 6 figures, 9 equations, and 6 tables
>
> **摘要:** Automated industrial inspection requires both precise defect localization and structured maintenance report generation; in current practice these tasks are handled separately, with linguistic interpretation left to human experts. This paper describes a decoupled, edge-deployable pipeline for wind turbine blade inspection built from three components that each handle a distinct sub-task. The Eyes a YOLO26-x-obb oriented bounding-box detector localizes defects at dataset-native resolution. The Bridge a deterministic, parameter-free encoding module maps each detected bounding box to grid-referenced spatial tokens embedded in a structured prompt. The Brain a 4-bit quantized Qwen-2.5-1.5B model adapted with Quantized Low-Rank Adaptation (QLoRA) on 947 synthetically generated maintenance reports generates a structured JSON report from that prompt. Retrieval-Augmented Fine-Tuning (RAFT) further grounds each recommendation in indexed maintenance procedures. Five ablation experiments, scored by BLEU-4, ROUGE-L, Hallucination Rate (HR), and an LLM-as-a-Judge rubric, compare the pipeline against a monolithic vision-language model (VLM) baseline and against partial configurations in which one component is removed. The complete system achieves BLEU-4 0.41, HR=4%, and Expert Score = 8.6/10 compared with 0.07, 65%, and 3.3/10 for the zero-shot VLM baseline. The QLoRA-adapted 1.5B model generates higher-quality reports than a 671B-parameter generalist API model given identical detection evidence, at 47 tokens per second on a single T4-class GPU. The results show that purpose-built decoupled architecture with a small domain-specific training corpus outperforms a generalist end-to-end model on this structured generation task.
>
---
#### [new 006] OmniInteract: Benchmarking Real-World Streaming Interaction for Real-Time Omnimodal Assistants
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出OmniInteract基准，用于评估实时多模态大模型的流式交互能力，解决在线多模态理解与响应问题。**

- **链接: [https://arxiv.org/pdf/2605.26485](https://arxiv.org/pdf/2605.26485)**

> **作者:** Xudong Lu; Xueying Li; Annan Wang; Yang Bo; Jinpeng Chen; Zengliang Li; Nianzu Yang; Rui Liu; Xue Yang; Jingwen Hou; Hongsheng Li
>
> **摘要:** We introduce OmniInteract, a streaming benchmark for real-time omnimodal large language models evaluated through native online inference over audio-visual streams. Unlike offline video understanding or text-prompted streaming QA, OmniInteract preserves the original audio-visual stream and requires models to process it online, without access to future content. User queries and ambient sounds are embedded in the audio track, requiring models to detect multimodal triggers, decide when to respond, and answer while the stream unfolds. OmniInteract contains 250 videos with 1,430 temporally grounded response slots: 1,062 1Q1A slots across real-time, proactive, and nested scenarios, and 368 1QnA slots for continuous task monitoring and step guidance. Each slot includes a trigger, response window, and target answer. We evaluate response correctness, timing, invalid outputs, interruption handling, and context continuity using Interaction-Aware Quality-Timeliness F1, Interruption Diagnostic Suite, and Nested Chain Completion Score. Experiments show that current models remain weak in streaming interaction, with the best overall IA-QTF1 reaching only 0.368 and the best 1QnA IA-QTF1 only 0.052. Further study on mathematical reasoning in full-duplex settings shows that offline capability does not necessarily transfer to online interaction. Code and datasets will be made publicly accessible at this https URL.
>
---
#### [new 007] When Eyes Betray AI: Social Gaze Consistency as a Semantic Cue for AI-Generated Image Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI生成图像检测任务，旨在解决低级特征失效后的检测难题。通过引入社会凝视一致性作为语义线索，提出三种机制提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.27348](https://arxiv.org/pdf/2605.27348)**

> **作者:** Kim Jihyeon; Sohee Kim; Soosan Lee; Souhwan Jung; James Matthew Rehg; Hyesong Choi
>
> **备注:** 23 pages, 2 figures, 17 tables
>
> **摘要:** Recent generative models have largely closed the gap on low-level artifacts - pixel fingerprints, frequency anomalies, upsampling traces - particularly in person-centric and partial-edit settings where the manipulated region is small and surrounded by photometrically authentic content. We introduce Social Gaze Consistency, a high-level semantic cue defined as the mutual coherence of gaze direction, head-eye alignment, and pupil placement between interacting individuals, and show that it constitutes a previously underutilized detection axis orthogonal to existing low-level paradigms. We instantiate this insight through three coupled mechanisms: (i) a controlled diagnostic dataset with region-specific perturbations of gaze-consistent imagery, where strict pair-level grouping forecloses generator-fingerprint memorization as an optimization-time shortcut rather than relying on augmentation; (ii) Block-Compositional Caption Supervision, which holds a single 5-block reasoning skeleton invariant across 1,250 macro-combined captions, decoupling reasoning consistency from surface diversity; (iii) Cross-architecture validation showing the same supervision improves a vision-language backbone (FakeVLM) by +3.7 pp on the COCOAI Interaction subset (balanced accuracy 67.8 -> 71.5) and +1.3 pp on the COCOAI Person subset (83.0 -> 84.3), with consistent gains on a vision-only backbone (Effort), evidencing a backbone-agnostic cue. Real- and fake-class recalls rise simultaneously, ruling out a "predict-all-fake" artifact. A four-step mechanistic account - paired-edit shortcut blocking, hard-to-easy difficulty transfer, CLIP prior preservation, and diffusion-family shared spectral weakness in periocular structure - explains why training on a single inpainter (FLUX.1-Fill) transfers to multi-generator suites. We will release the code upon acceptance to facilitate reproducibility.
>
---
#### [new 008] Touch-R1: Reinforcing Touch Reasoning in MLLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决触觉推理不足的问题。通过构建大规模数据集和评估框架，提出Touch-R1模型，提升触觉与视觉信息的融合与冲突解决能力。**

- **链接: [https://arxiv.org/pdf/2605.27154](https://arxiv.org/pdf/2605.27154)**

> **作者:** Yingxin Lai; Yafei Zhou; Fucai Zhu; Siyu Zhu; Weihao Yuan
>
> **备注:** Our code and data will be made public on the this https URL
>
> **摘要:** While rule-based reinforcement learning has recently catalyzed explicit reasoning in multimodal models, tactile reasoning remains largely underexplored. Existing tactile-language models primarily rely on supervised or contrastive objectives, which limits their capacity to ground predictions in physical evidence or rectify misleading visual priors. Tactile reasoning introduces two modality-specific challenges: the ordinal nature of physical attributes (e.g., hardness, roughness) and the cross-sensor distribution shifts inherent in optical tactile hardware. In this work, we introduce TouchReason-1M, a large-scale multimodal dataset comprising over 1M synchronized tactile pairs across four distinct sensors, and TouchReason-Bench, a rigorous framework for evaluating tactile perception and visual-tactile conflict resolution. Building upon these, we propose Touch-R1, a tactile reasoning MLLM based on Qwen2.5-VL-7B. Touch-R1 is trained via a tactile-grounded GRPO objective that combines ordinal-aware accuracy, cross-sensor physical consistency, structured-format control, and an input-side tactile grounding objective. Specifically, the tactile-use reward assigns credit only when authentic tactile inputs yield superior correctness relative to counterfactual controls where the tactile stream is removed, shuffled, or noise-masked. On TouchReason-Bench, Touch-R1-7B outperforms Octopi-13B by 18.4\% and GPT-4o by 24.7\% on average. Its structured reasoning traces reveal emergent behaviors of probing, comparison, and revision, demonstrating that R1-style reasoning can be effectively grounded in physical contact.
>
---
#### [new 009] Attenuation-Resilient Alternating Optimization for Laparoscopic Liver Landmark Detection
- **分类: cs.CV**

- **简介: 该论文属于肝表面关键点检测任务，解决光照衰减和结构不匹配问题。提出A2ONet网络，通过补偿光照和优化曲线结构，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.26630](https://arxiv.org/pdf/2605.26630)**

> **作者:** Lanqing Liu; Ruize Cui; Jialun Pei; Diandian Guo; Tiffany Y. So; Pheng-Ann Heng; Jing Qin
>
> **备注:** This paper has been accepted by MICCAI 2026
>
> **摘要:** Liver surface landmark detection is a fundamental prerequisite for anatomical guidance in laparoscopic liver surgery. However, it remains unreliable in practice due to two pervasive challenges: illumination attenuation in underexposed regions and the structural mismatch between pixel-wise localization and continuous curvilinear geometry. To address these limitations, we propose A2ONet, an attenuation-resilient alternating optimization network for robust liver landmark detection. To mitigate illumination attenuation, A2ONet embraces an illumination field compensation (IFC) block that adaptively enhances dark regions while preserving structural consistency. Meanwhile, we introduce a lightweight frequency-orientation selective filter (FOSF) to suppress repetitive texture interference and preserve salient curvilinear cues. Building upon these resilient representations, we design an alternating seg-curve optimization (ASCO) decoder that iteratively couples dense segmentation with explicit curve modeling, enabling mutual guidance to optimize both structural continuity and endpoint localization. Extensive evaluations on L3D-2K, L3D, and P2ILF demonstrate consistent improvements over competitive methods, establishing a more reliable foundation for intraoperative anatomy guidance. Our code will be available at this https URL.
>
---
#### [new 010] DinoComplete: 3D Shape Completion with Distilled Semantic Priors and State Space Models
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D形状补全任务，解决未见类别和噪声数据下的结构缺失问题。通过融合语义先验与几何信息，提出DinoComplete框架，提升补全效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.26949](https://arxiv.org/pdf/2605.26949)**

> **作者:** Furkan Mert Algan; Eckehard Steinbach
>
> **摘要:** 3D shape completion from partial scans remains challenging for unseen categories and noisy real-world observations, where geometry alone is often insufficient for inferring missing structure. We present DinoComplete, a deterministic and efficient shape completion framework that augments geometric reconstruction with voxel-aligned semantic priors distilled from DINO features. First, we construct multi-view DINO feature volumes aligned with ShapeNet data and train a student network to predict dense semantic features directly from incomplete shapes. These predicted features capture global structure and part-aware semantic context while remaining aligned with the underlying geometry. We then integrate these distilled features into a completion network, where geometric and semantic voxel representations are fused through voxel state-space modeling. To enable efficient long-range reasoning without sacrificing resolution, we introduce a multi-scale voxel Mamba module that refines the fused features by combining full-grid and chunk-wise sequence modeling. Experiments on unseen ShapeNet categories and ScanNet objects show that DinoComplete achieves stronger completion quality than prior deterministic and generative based completion methods while using fewer parameters, requiring lower memory, and achieving faster inference. Our results demonstrate that distilling semantic priors from visual foundation models improves generalization and robustness in 3D shape completion.
>
---
#### [new 011] Feedforward 3D Editing Learns from Semantic-Part Transformation
- **分类: cs.CV**

- **简介: 该论文属于3D编辑任务，解决缺乏高质量监督的问题。提出Pxform数据集和PartFlow网络，通过语义部件变换提升编辑精度与一致性。**

- **链接: [https://arxiv.org/pdf/2605.27351](https://arxiv.org/pdf/2605.27351)**

> **作者:** Jiawei Weng; Saining Zhang; Zhenxin Diao; Peishuo Li; Henghaofan Zhang; Junhao Chen; Hao Zhao
>
> **备注:** 30 pages, 22 figures. Project Page: this https URL
>
> **摘要:** 3D editing is a fundamental capability for scalable 3D content creation. While image editing has rapidly evolved toward large-scale feedforward generative paradigms, 3D AI generation remains dominated by training-free editing pipelines. A central challenge of feedforward 3D editing lies in the lack of high-quality paired supervision. Editable 3D assets require simultaneous preservation of geometry, multi-view consistency, structural coherence, and localized edit controllability. Existing 3D editing datasets often rely on independently generated assets, image-mediated reconstruction or narrow edit taxonomies, leading to inaccurate localization, weak preservation, blurred edit boundaries, and limited semantic consistency. In this work, we introduce a new perspective: scalable feedforward 3D editing should be learned from semantic-part transformations. Based on this insight, we propose Pxform, a high-quality 3D editing dataset with over 100K consistent before/after editing pairs across seven edit types. Instead of treating objects as unstructured shapes, our pipeline grounds edits directly in semantic 3D parts. Built upon Pxform, we further propose PartFlow, a feedforward 3D editing network that injects source-aware latent control into pretrained 3D generative priors. PartFlow introduces mask-aware velocity preservation and render-space consistency supervision to jointly improve edit fidelity and source preservation, while requiring no 3D edit mask during inference. Extensive experiments demonstrate that high-quality semantic-part supervision substantially improves scalable 3D editing, enabling PartFlow to achieve state-of-the-art performance on both geometric and appearance editing benchmarks.
>
---
#### [new 012] Uncertainty-Aware Gaussian Map for Vision-Language Navigation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决感知不确定性影响导航决策的问题。通过构建语义高斯图并融合几何、语义和外观不确定性，提升导航可靠性。**

- **链接: [https://arxiv.org/pdf/2605.26503](https://arxiv.org/pdf/2605.26503)**

> **作者:** Jianzhe Gao; Rui Liu; Yuxuan Xu; Tongtong Cao; Yingxue Zhang; Zhanguang Zhang; Sida Peng; Yi Yang; Wenguan Wang
>
> **摘要:** Vision-Language Navigation (VLN) requires an agent to navigate 3D environments following natural language instructions. During navigation, existing agents commonly encounter perceptual uncertainty, such as insufficient evidence for reliable grounding or ambiguity in interpreting spatial cues, yet they typically ignore such information when predicting actions. In this work, we explicitly model three forms of perceptual uncertainty (i.e., geometric, semantic, and appearance uncertainty) and integrate them into the agent's observation space to enable informed decision-making. Concretely, our agent first constructs a Semantic Gaussian Map (SGM), composed of differentiable 3D Gaussian primitives initialized from panoramic observations, that encodes both the geometric structure and semantic content of the environment. On top of SGM, geometric uncertainty is estimated through variational perturbations of Gaussian position and scale to assess structural reliability; semantic uncertainty is captured by perturbing Gaussian semantic attributes to reveal ambiguous interpretations; and appearance uncertainty is characterized by Fisher Information, which measures the sensitivity of rendered observations to Gaussian-level variations. These uncertainties are incorporated into SGM, extending it into a unified 3D Value Map, which grounds them as affordances and constraints that support reliable navigation. Comprehensive evaluations across multiple VLN benchmarks show the effectiveness of our agent.
>
---
#### [new 013] The Rescue Effect: Spatio-Semantic Early Exit Bypasses Quantization Collapse in CLIP
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型优化任务，解决INT8量化导致的CLIP模型表征崩溃问题。提出LRA-EE方法，通过早期退出机制提升精度并降低计算量。**

- **链接: [https://arxiv.org/pdf/2605.26415](https://arxiv.org/pdf/2605.26415)**

> **作者:** Kahyeon Nam; Hyesong Choi
>
> **摘要:** Deploying Vision-Language Models on resource-constrained hardware typically requires INT8 quantization, but in joint-embedding architectures such as CLIP this introduces a failure mode distinct from quantized CNN classifiers: activation noise accumulated across transformer blocks perturbs the direction of the multimodal embedding, eroding the cosine alignment on which zero-shot retrieval depends. We characterize this as Quantization-Induced Representation Collapse (QIRC) and quantify it on INT8 CLIP ViT-B/32, where the layer-wise noise-to-signal ratio grows from below 10% in shallow blocks to 52% at Layer 11. We propose LRA-EE (Layer-wise Representation-Aware Early Exit), which bypasses noise-saturated deep layers via Spatio-Semantic Aggregation (replacing the immature shallow [CLS] with a global patch-token average), a learned multi-feature gate (confidence, top-2 margin, spatial-activation variance), and Layer-adaptive Confidence Thresholding calibrated to each layer's Information-to-Noise Ratio. On ImageNet-1K zero-shot classification, LRA-EE reduces FLOPs by 13.4% and improves Top-1 accuracy by +2.44%p (58.72% -> 61.16%) over the INT8 baseline. A four-quadrant decomposition isolates the Rescue Effect: 9.5% of samples are correctly classified at shallow exits but lost to noise at full depth, against only 7.1% suffering the inverse.
>
---
#### [new 014] E$^3$C: Video Generation with 3D Environmental Memory and Ego-Exo Human Pose Control
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出E$^3$C，用于可控的自观视频生成。解决自观视频中视角变化大、人体动作细微等问题，通过3D环境记忆和人体姿态控制实现更真实的视频生成。**

- **链接: [https://arxiv.org/pdf/2605.26316](https://arxiv.org/pdf/2605.26316)**

> **作者:** Qiao Gu; Lingni Ma; Adam W Harley; Richard Newcombe; Florian Shkurti; Julian Straub
>
> **备注:** Preprint. Project Page: this https URL
>
> **摘要:** Controllable and physically grounded egocentric video generation is essential for embodied agents to reason about how their own and others' actions manifest and change the world. Compared to generic video synthesis, egocentric generation is especially challenging: the camera is tightly coupled to the actor, leading to rapid viewpoint changes and frequent self-occlusions; the underlying actions are subtle, articulated, and often only partially visible; and both the people and the scene state must evolve consistently with the specified controls. We present E$^3$C, a controllable video diffusion framework for egocentric generation that builds structured and compact conditions disentangling persistent scene structure from human-driven dynamics. From context frames, E$^3$C constructs a semi-dense point cloud-based 3D memory and augments each point with appearance descriptors from video-VAE features. Rendering this memory into target viewpoints produces conditioning aligned with the target frames. Human dynamics are modeled separately. The observed people in the scene are controlled by skeleton renderings (exo human control), while the camera wearer is specified by their 3D body joints and 6DoF wrist motion (ego human control). To preserve ego human control when the wearer's body parts are invisible, we introduce an ego motion encoder that produces persistent cross-attention tokens. Experiments on Nymeria show that E$^3$C improves visual fidelity, camera-motion accuracy, object consistency, and ego & exo human control over strong baselines, while also enabling intuitive scene editing.
>
---
#### [new 015] Revealing the core dimensions underlying representations in brains, behavior and AI
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文提出SRF方法，用于从相似性矩阵中提取可解释的低维表示，解决跨领域（脑科学、行为、AI）表征维度分析问题。**

- **链接: [https://arxiv.org/pdf/2605.26921](https://arxiv.org/pdf/2605.26921)**

> **作者:** Florian P. Mahner; Ka Chun Lam; Francisco Pereira; Martin N. Hebart
>
> **摘要:** The study of representations is widespread across fields, including neuroscience, psychology, and artificial intelligence. While representations are often studied and compared through similarities between stimuli, current methods provide only limited access to the dimensions that shape these representations and are often limited in interpretability. To overcome these challenges, here we introduce Similarity-Based Representation Factorization (SRF), a general computational method for recovering low-dimensional, non-negative, interpretable embeddings from similarity matrices derived from measured data. Across simulations and many neural, behavioral, and computational datasets, SRF recovers interpretable dimensions from diverse forms of representational data, even for very sparsely sampled, incomplete data. The dimensions derived from these datasets match those obtained by task-specific models, predict independent behavioral properties, improve exploratory analysis, and offer higher power for confirmatory hypothesis testing than comparing similarity matrices. Together, these results establish SRF as a general-purpose method with broad applications for uncovering, understanding, and leveraging the dimensions underlying representations.
>
---
#### [new 016] DynFrame: Adaptive Reasoning-Driven Multimodal Framework with Dynamic Frame Augmentation for Complex Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DynFrame框架，解决视频理解中采样密度固定和检索与回答优化不分离的问题，通过动态帧增强实现多粒度证据获取。**

- **链接: [https://arxiv.org/pdf/2605.26680](https://arxiv.org/pdf/2605.26680)**

> **作者:** Peng Zhang; Guanghao Zhang; Wanggui He; Longxiang Zhang; Mushui Liu; Yan Xia; Zhenhao Peng; Weilong Dai; Jinlong Liu; Haobing Tang; Le Zhang; Hao Jiang; Pipei Huang
>
> **摘要:** Recent video multimodal large language models (MLLMs) increasingly couple step-by-step reasoning with on-demand visual evidence retrieval, allowing models to revisit relevant video segments during inference. However, two structural gaps remain in existing thinking-with-video systems. (i) Sampling density is not a learnable decision: existing methods may let the model decide where to look, but the per-window frame rate is largely fixed. As a result, fine-grained evidence is often recovered through repeated retrieval calls, which increases inference context length and training difficulty. (ii) Retrieval and answer generation are usually optimized with a single trajectory-level advantage, so the "where to look" tokens and the "how to answer" tokens receive the same credit even when one is correct and the other is not. To address these gaps, we present DynFrame, a framework that emits the temporal window and the sampling density as native tokens within a single autoregressive pass. This learnable span-density retrieval enables acquiring multi-granularity evidence with a single retrieval step. Based on the above tokenized retrieval interface, we further introduce Segment-Decoupled GRPO (SD-GRPO), which splits each rollout at the retrieval boundary and assigns role-specific token-level advantages, separately crediting the sampling decision and the answer. Trained on the curated DM-CoT-74k and DM-RL-45k, DynFrame-4B is competitive with strong 7B-8B baselines across six benchmarks (NExT-GQA, Charades-STA, ActivityNet-MR, Video-MME, MLVU, LVBench), and DynFrame-8B sets new state-of-the-art on most metrics. Code is available at this https URL.
>
---
#### [new 017] InterSketch: An Interleaved Reasoning Model with Self-correcting Visual Sketch and Stepwise Reward
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出InterSketch模型，解决视觉语言模型在长距离视觉推理中的浅层和文本主导问题，通过交织视觉-文本链式思维与自修正机制提升推理能力。**

- **链接: [https://arxiv.org/pdf/2605.26520](https://arxiv.org/pdf/2605.26520)**

> **作者:** Zhiwei Ning; Wenwen Tong; Xiangli Kong; Shengnan Ma; Ziyi Shang; Jingcheng Ni; Tao Hu; Yong Xien Chng; Jixuan Ying; Zehuan Wu; Hanming Deng; Jie Yang; Yuanjie Zheng; Wei Liu; Lewei Lu
>
> **摘要:** While vision-language models (VLMs) have exhibited multi-turn visual reasoning capabilities, their reasoning trajectories remain relatively shallow and are dominated by a text-centric paradigm, limiting their applicability to complex visual challenges. In contrast, human-like thought typically involves long-horizon reasoning with an interleaved visual-textual chain-of-thought (VT-CoT). To bridge this gap, we introduce InterSketch, an interleaved reasoning model to enhance the VT-CoT capability via self-correcting and stepwise reward mechanisms. InterSketch dynamically generates intermediate visual sketches using external tools and interleaves them with textual reasoning, enabling effective perception and logical reasoning over long-horizon visual understanding tasks. Specifically, in the first cold-start stage, we propose a synthesized high-quality interleaved VT-CoT dataset and include a reflection mechanism to enable the model's capability in multi-turn interleaved reasoning and self-correction. In the subsequent reinforcement learning (RL) stage, we design a stepwise reward mechanism to mitigate the sparsity of reward signals inherent in end-only supervision over long-horizon reasoning. Extensive experiments on visual reasoning benchmarks demonstrate the effectiveness of InterSketch, even outperforming proprietary models such as Gemini-3-Pro.
>
---
#### [new 018] PILOT: A Data-Free Continual Learning Approach for Real-Time Semantic Segmentation via Boundary Guidance
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于实时语义分割任务，解决增量学习中的灾难性遗忘问题。提出PILOT方法，在不重新训练整个数据集的情况下，有效学习新类别并保持原有类别的性能。**

- **链接: [https://arxiv.org/pdf/2605.27128](https://arxiv.org/pdf/2605.27128)**

> **作者:** Yujing Zhou; Prashant Shekhar; Thomas Yang; Yongxin Liu
>
> **摘要:** Real-time semantic segmentation models offer an excellent balance between accuracy and inference speed. However, deploying these models in dynamic real world environments often requires the ability to learn novel classes incrementally without retraining on the entire dataset. This capability is known as continual learning. In this regard, the standard fine-tuning methods in deep learning often fail due to catastrophic forgetting, where the model learns new information but forgets previously trained and learned classes. Contributing to this crucial domain, the current paper proposes a novel continual learning framework tailored for PIDNet, which is a widely cited state-of-the-art real-time semantic segmentation model. Our method, PILOT(Parallel Incremental Learning Over Time), introduces a real-time and lightweight strategy by implementing a parallel Derivative-branch (D-branch) designed to capture the high frequency boundary information of novel classes while freezing the trained parameters of the original segmentation network. This novel setup allows the model to adapt to new semantic categories while preserving the knowledge of previously learned classes. By using only data associated with the new class, our model significantly reduces training overhead. Experimental results demonstrate that our approach successfully segments new classes while maintaining high mean Intersection over Union (mIoU) on the original base classes, thereby comfortably outperforming all major continual learning approaches in this domain. Overall, PILOT is shown to effectively mitigate catastrophic forgetting with minimal impact on inference latency, thus maintaining real-time performance.
>
---
#### [new 019] Is an Image Also Worth 16x16=256 Superpixels? A Framework for Attentional Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分类任务，旨在解决传统方法与Transformer模型结合的问题。提出SPT框架，融合超像素与Transformer，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2605.27144](https://arxiv.org/pdf/2605.27144)**

> **作者:** Pedro Henrique da Costa Avelar; Anderson R. Tavares; Luís C. Lamb
>
> **摘要:** Superpixel-based image classification has traditionally leveraged graph neural networks (GNNs) for processing irregular image representations. Recent advances in computer vision, driven by Vision Transformers (ViTs), have introduced new paradigms in self-attentional models, surpassing convolutional neural networks (CNNs) in various tasks. However, a synergistic connection between GNNs, superpixels, and transformers remains unexplored. In this work, we propose Superpixel Transformers (SPT), a novel framework that unifies superpixel-based image classification and ViTs. SPT generalizes the Superpixel Image Classification with Graph Attention Networks (SICGAT) model and ViT to support arbitrary superpixel-based chunking strategies, connectivity graphs, and positional encodings. We introduce refinements including a multidimensional sine-cosine positional encoding and an enriched patch data structure that fully incorporates superpixel shape and color information. By testing SPT across datasets such as CIFAR10, FashionMNIST, and Imagenette, with various superpixel generation and graph connectivity strategies, we demonstrate that SPT achieves superior performance compared to previous superpixel-based GNN methods and remains competitive with ViTs. Notably, our approach addresses the limitations of SICGAT, such as information loss during pixel aggregation, and shows how constrained graph connectivity can enhance ViT performance. SPT bridges the gap between superpixel-based and transformer models, opening avenues for cross-domain generalization and future innovations in hybrid attentional frameworks, and showing that an image can also be worth $16\times16$ superpixels.
>
---
#### [new 020] Sleep-stage efficient classification using a lightweight self-supervised model
- **分类: cs.CV**

- **简介: 该论文属于睡眠阶段分类任务，旨在提升分类效率与准确性。通过简化模型并结合SVM分类器，优化了睡眠阶段的自动识别。**

- **链接: [https://arxiv.org/pdf/2605.26295](https://arxiv.org/pdf/2605.26295)**

> **作者:** Eldiane Borges dos Santos Durães; João Batista Florindo
>
> **摘要:** Accurate classification of sleep stages is crucial for diagnosing sleep disorders and automating this process can significantly enhance clinical assessments. This study aims to explore the use of a self-supervised model (more specifically, an adapted version of mulEEG) combined with a Linear SVM classifier to improve sleep stage classification. \textbf{Methods:} The mulEEG model, which learns electroencephalogram signal representations in a self-supervised manner, was simplified here by replacing ResNet-50 with 1D-convolutions used as time series encoder by a ResNet-18 backbone. Two other adaptations were conducted: the first one evaluated different configurations of the model and data volume for training, while the second tested the effectiveness of time series features, spectrogram features, and their concatenation as inputs to a Linear SVM classifier. \textbf{Results:} The results showed that reducing the volume of data offered a better cost-benefit ratio compared to simplifying the model. Using the concatenated features with ResNet-18 also outperformed the linear evaluations of the original mulEEG model, achieving higher classification performance. \textbf{Conclusions:} Simplifying the mulEEG model to extract features and pairing it with a robust classifier leads to more efficient and accurate sleep stage classification. This approach holds promise for improving clinical sleep assessments and can be extended to other biological signal classification tasks.
>
---
#### [new 021] Scheduled Style Injection: Expanding the Style-Content Pareto Frontier in Training-Free Diffusion-based Style Transfer
- **分类: cs.CV**

- **简介: 该论文属于图像风格迁移任务，解决风格与内容平衡问题。通过优化风格注入策略，提升风格保真度与内容保留效果。**

- **链接: [https://arxiv.org/pdf/2605.26538](https://arxiv.org/pdf/2605.26538)**

> **作者:** Amey Sunil Kulkarni
>
> **备注:** Accepted to CVPR NTIRE 2026
>
> **摘要:** Style transfer with pre-trained diffusion models has advanced rapidly, but a core question remains underexplored: where in the model should style injection be strongest? StyleID, the leading training-free method, uses a single global parameter (gamma) uniformly across all layers and timesteps, which forces a fixed tradeoff between style quality and content preservation. We show this tradeoff is unnecessarily rigid. We systematically explore four dimensions of control: varying style injection strength across decoder layers, across denoising timesteps, and scheduling ControlNet geometric conditioning along both axes. The pattern is consistent everywhere: decreasing schedules, with stronger structural signal injection in shallower layers and earlier timesteps, reliably outperform the reverse. Beyond direction, schedule shape matters: cosine and square-root timestep schedules outperform linear. Most importantly, we find that gamma scheduling and ControlNet conditioning are nearly independent. The resulting combined configurations expand the Pareto frontier, offering superior tradeoffs between style fidelity and content preservation compared to any single baseline setting. Our best balanced configuration achieves ArtFID of 27.036 versus StyleID's 28.801 - a 6.1% relative improvement, with consistent gains across the full style-content tradeoff frontier. Results are validated across 35 configurations totaling over 28,000 stylized images using four complementary metrics. These findings generalize across SD backbones with identical rank ordering. All modifications are training-free, parameter-free, and require only a few lines of scheduling code; code is available at this https URL.
>
---
#### [new 022] CodecCap: High-Fidelity Codec-Inspired Residual Modeling for Dense Video Captioning
- **分类: cs.CV**

- **简介: 该论文提出CodecCap，解决视频描述中视觉保真与冗余的平衡问题。通过关键帧与残差描述，提升细粒度信息保留，构建高质量视频描述数据集。**

- **链接: [https://arxiv.org/pdf/2605.26967](https://arxiv.org/pdf/2605.26967)**

> **作者:** Zihan Lin; Songhe Deng; Shuwei He; Danxiang Zhu; Dan Zhang; Yishu Lei; Xianlong Luo; Shikun Feng; Rui Liu
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Existing video captioning methods struggle to balance visual fidelity and redundancy: holistic captions are compact but lose fine-grained evidence, whereas segment-wise captions improve coverage but introduce heavy redundancy. We propose CodecCap, a codec-inspired framework for high-fidelity dense video captioning. Analogous to video codecs, CodecCap represents videos using keyframe and residual captions. Keyframe captions exhaustively encode stable visual context, while residual captions capture temporally only localized actions, motions and changes. This effectively preserves fine-grained visual evidence while reducing redundant descriptions. To quantify the fidelity of captions, we introduce VidCapQA, a caption-then-QA benchmark with 1,000 questions across 14 capability dimensions. Results on VidCapQA show that captions directly generated by strong VLMs still miss many visual details, highlighting caption representation as a critical bottleneck. Experiments show that CodecCap significantly surpasses direct captioning with the same underlying VLMs, suggesting keyframe-residual captioning a way for high-fidelity video-language supervision. We further use CodecCap to construct CodecVDC-100K, a large-scale dense captioning dataset with anchor, residual, scene-level, and video-level supervision.
>
---
#### [new 023] MedVol-R1: Reward-Driven Evidence Grounding for Volumetric Reasoning Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，解决从自然语言查询中准确分割3D医学影像的问题。提出MedVol-R1框架，通过强化学习分离证据定位与体积分割，提升分割精度和可解释性。**

- **链接: [https://arxiv.org/pdf/2605.26621](https://arxiv.org/pdf/2605.26621)**

> **作者:** Zichun Wang; Hairong Shi; Bingzheng Wei; Yan Xu; Zihua Wang
>
> **摘要:** Volumetric Reasoning Segmentation (VRS) aims to segment a target region in a 3D medical scan from a free-form clinical query, where the referent is often implicit and requires both medical knowledge and volume-grounded reasoning. Existing methods typically rely on specialized segmentation tokens to connect language with mask decoding, but this coupling collapses the decision process into opaque latent representations, limiting interpretability and generalization to diverse narrative expressions. In this paper, we present MedVol-R1, a reinforcement learning-based framework for VRS that explicitly decouples evidence grounding from volumetric delineation: the LVLM grounds clinical reasoning to a verifiable 2D evidence anchor (key axial slice and 2D bounding boxes), which is then propagated into a coherent 3D mask by a frozen MedSAM2 module. We train MedVol-R1 with cold-start supervised fine-tuning followed by GRPO, guided by a multi-component reward that encourages informative evidence selection, accurate 2D spatial grounding, and cross-slice volumetric coherence, without requiring costly chain-of-thought annotations. Experiments on CT-ORG, AbdomenCT-1K, and KiTS23 from the M3D-Seg benchmark demonstrate that MedVol-R1 consistently outperforms strong baselines and achieves state-of-the-art performance, with reinforcement learning providing clear gains over pure supervised fine-tuning.
>
---
#### [new 024] Rotation-Invariant Spherical Watermarking via Third-Order SO(3) Representation Coupling
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于图像水印任务，解决全景图像在任意3D旋转下的鲁棒水印问题。通过构建SO(3)不变的三阶球谐描述子，实现旋转不变的水印嵌入与提取。**

- **链接: [https://arxiv.org/pdf/2605.26702](https://arxiv.org/pdf/2605.26702)**

> **作者:** Pengzhen Chen; Yanwei Liu; Xiaoyan Gu; Antonios Argyriou; Wu Liu; Weiping Wang
>
> **备注:** ICML 2026
>
> **摘要:** Reliable watermarking of panoramic imagery is fundamentally challenged by arbitrary 3D rotations. As panoramas are defined on the sphere, they naturally transform under the action of $SO(3)$, rendering conventional planar representations and augmentation-based robustness strategies inadequate and devoid of theoretical guarantees. To address this, we formulate panoramas as spherical signals and leverage $SO(3)$ representation theory to derive provably rotation-invariant descriptors. While spherical harmonic coefficients transform equivariantly under rotations, the natural invariant constructions are typically limited to zeroth-order statistics which eliminate directional information and severely constrain embedding capacity. In this work, we introduce a principled third-order invariant construction by coupling higher-order $SO(3)$ irreducible representations via tensor products and projecting onto the trivial representation. This yields a spherical invariant bispectrum that preserves phase information while remaining strictly rotation-invariant. Leveraging this property, we embed watermarks into higher-order spherical harmonic coefficients and recover them from invariant bispectral scalars, enabling reliable extraction under arbitrary 3D rotations. We provide a theoretical proof of $SO(3)$ invariance for it and demonstrate experimentally its near-perfect robustness to continuous rotations while maintaining high visual fidelity.
>
---
#### [new 025] Unveiling the Fragility of Vision-Language Models: Multi-Modal Adversarial Synergy via Texture-Constrained Perturbations and Cross-Modal Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全与对抗攻击领域，旨在解决LVLMs在多模态攻击下的脆弱性问题。提出一种新的多模态攻击框架，通过图像和文本扰动协同作用，提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2605.26501](https://arxiv.org/pdf/2605.26501)**

> **作者:** Xiang Fang; Wanlong Fang; Changshuo Wang
>
> **备注:** Publish in AAAI 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) have transformed multi-modal understanding, excelling in tasks like image captioning and visual question answering by integrating visual and textual inputs. However, their robustness against adversarial attacks, particularly those exploiting both modalities, remains underexplored, posing risks to critical applications like autonomous driving and content moderation. Existing attacks focus on single modalities or require impractical white-box access, limiting their real-world relevance. In this paper, we introduce Multi-Modal Adversarial Synergy, a groundbreaking framework that crafts universal, black-box multi-modal attacks against LVLMs. MMAS simultaneously generates a texture scale-constrained universal adversarial perturbation for images and a learnable prompt perturbation for text, optimized jointly using only model queries. The image perturbation leverages wavelet-based texture constraints to ensure imperceptibility and robustness across diverse visual inputs. The text perturbation, constrained by an L-norm in the embedding space, maintains semantic coherence while steering outputs toward a target. A novel cross-modal regularization term aligns the perturbations' gradient directions, enhancing their synergistic impact and transferability across tasks and models. Extensive experiments show the strong universal adversarial capabilities of our proposed attack with prevalent LVLMs.
>
---
#### [new 026] VisualNeedle: Benchmarking Active Visual Search in Information-Dense Scenes
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉问答任务，旨在解决多模态大模型在细粒度视觉搜索中的虚假性能问题。通过构建信息密集的VisualNeedle基准和反事实实验，验证模型是否真正依赖视觉证据。**

- **链接: [https://arxiv.org/pdf/2605.26380](https://arxiv.org/pdf/2605.26380)**

> **作者:** Jingru Chen; Yiming Liu; Mingtao Chen; Sijie Chen; Richeng Xuan; Liang Yang; Zhichao Hu; Fanyang Lu
>
> **摘要:** Frontier multimodal large language models (MLLMs) have been reported to achieve over 90% accuracy on fine-grained perception benchmarks. However, such scores do not necessarily imply faithful use of visual evidence. Prior studies have identified three shortcuts that inflate benchmark performance. First, linguistic priors and lexical cues in questions often enable models to infer plausible answers without seeing the image. Second, coarse global semantics from the visual encoder can bypass fine-grained local details. Third, in some ``think-with-images'' benchmarks, corrupting the intermediate images returned by visual tools barely affects the final answer. These findings suggest that higher input resolution or larger question pools alone do not elicit genuine active visual search. To address this, we introduce VisualNeedle, a challenging, information-dense, and fine-grained benchmark for scenes where critical evidence is spatially constrained to minute regions and not discernible at a glance. We further propose a counterfactual crop-black setting, which replaces crops returned by tools with black images of the same size, to test whether tool-enabled performance truly relies on intermediate visual evidence. We evaluate 9 promninent MLLMs across three settings: no-tool, standard tool-enabled, and crop-black. No-tool accuracy stays below 20\%, and the best tool-enabled model reaches only 56.01\%, still trailing the 63.00% human majority-vote accuracy. These results reveal persistent limitations in fine-grained visual search, while the crop-black ablation confirms that success on VisualNeedle hinges on genuine intermediate visual evidence.
>
---
#### [new 027] BEAT: Rhythm-Elastic Alignment for Agentic Music-guided Movie Trailer Generation
- **分类: cs.CV**

- **简介: 该论文属于电影预告片生成任务，解决音乐与画面同步问题。提出BEAT框架，通过弹性对齐提升预告片质量。**

- **链接: [https://arxiv.org/pdf/2605.27067](https://arxiv.org/pdf/2605.27067)**

> **作者:** Yutong Wang; Yunke Wang; Xinyuan Chen; Chang Xu
>
> **摘要:** Automatic movie trailer generation must select shots from a full-length film and synchronize them with background music. Existing methods either relegate music alignment to post-processing or enforce rigid one-to-one shot-music mappings, overlooking that professional editing rhythm is elastic: rapid cuts accompany high-energy passages while sustained shots span quieter bars. We introduce BEAT, a framework that addresses this gap with two core components: MuVA, a compact music-visual alignment encoder trained with Sinkhorn-regularized two-stage learning, and Bar-DP, an energy-adaptive dynamic programming algorithm that produces elastic many-to-one alignments following musical dynamics. These components are integrated into a five-phase agentic pipeline that grounds the core alignment in learned cross-modal features while coordinating higher-level creative decisions through structured text signals. To support comprehensive evaluation, we also introduce TrailerArena, a benchmark with 20+ metrics across four complementary dimensions. On TrailerArena, BEAT achieves state-of-the-art performance across shot selection, ordering, and perceptual quality, while producing fully composed trailers end-to-end.
>
---
#### [new 028] FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于3D物体分割任务，解决无标注点云中复杂场景的物体分割问题。提出FoundObj框架，利用自监督模型提供语义和几何奖励，实现多类物体的鲁棒识别。**

- **链接: [https://arxiv.org/pdf/2605.27178](https://arxiv.org/pdf/2605.27178)**

> **作者:** Zihui Zhang; Zhixuan Sun; Yafei Yang; Jinxi Li; Jiahao Chen; Bo Yang
>
> **备注:** ICML 2026. Zihui and Zhixuan are co-first authors. Code and data are available at: this https URL
>
> **摘要:** We address the challenging task of 3D object segmentation in complex scene point clouds without relying on any scene-level human annotations during training. Existing methods are typically constrained to identifying simple objects, primarily due to insufficient object priors in the learning process. In this paper, we present FoundObj, a novel framework featuring a superpoint-based object discovery agent that incrementally merges suitable neighboring superpoints, guided by our innovative semantic and geometric reward modules. These modules synergistically leverage semantic and geometric priors from self-supervised 2D/3D foundation models, providing complementary feedback to the object discovery agent and enabling robust identification of multi-class objects through reinforcement learning. Extensive experiments on diverse benchmarks demonstrate that our approach consistently outperforms existing baselines. Notably, our method exhibits strong generalization in zero-shot and long-tail scenarios, underscoring its potential for scalable, label-free 3D object segmentation.
>
---
#### [new 029] Once-For-All: A Train-Once and Select-Anytime Framework for Multimodal Instruction Tuning
- **分类: cs.CV**

- **简介: 该论文提出OFA框架，解决多模态指令调优中的数据选择问题。通过训练一次可复用的选择器，提升训练效率，适用于不同数据集和模型。**

- **链接: [https://arxiv.org/pdf/2605.26761](https://arxiv.org/pdf/2605.26761)**

> **作者:** Mingkang Dong; Hongyi Cai; Xiwen Lei; Jie Li; Tao Zhang; Muxin Pu
>
> **备注:** 15 pages, 6 figures. Mingkang Dong and Hongyi Cai contributed equally to this work. Muxin Pu is the corresponding author
>
> **摘要:** Multimodal instruction tuning is the de facto recipe for adapting vision language models (VLMs), yet instruction data are highly redundant, making data selection critical for training efficiency. Existing methods derive selection signals from a specific model or dataset, so whenever the target model or candidate pool changes, the criteria must be recomputed from scratch at substantial cost. To address this, we propose OFA, a data selection framework that trains a reusable selector once and applies it to any dataset or model without recomputation. OFA clusters multimodal instructions in a frozen CLIP space, derives pseudo labels from the cluster structure, and trains a lightweight selector for only a few epochs; samples on which this selector is least confident are selected as the most informative. Once trained, the frozen selector transfers directly across datasets and model scales. The selector is trained once on LLaVA-665K and applied both to LLaVA-665K itself and, without any retraining, to the unseen Vision-Flan-186K. Selecting only 15% of the data, OFA achieves 98.3% of full data performance across 10 downstream benchmarks; on the smaller Vision-Flan-186K, the transferred selector surpasses full data training by 10.6%, confirming that the learned signal generalizes to datasets never seen during selector training. The same selected subsets benefit VLMs at both Qwen2.5-VL-3B and LLaVA-v1.5-7B without per model recomputation, decoupling selection from the target model. These results demonstrate that a single, transferable selector provides an effective and reusable solution for efficient multimodal instruction tuning.
>
---
#### [new 030] Generative Animations: A Multi-Model Pipeline for Prompt-Driven Motion Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动画生成任务，解决手动创建运动路径繁琐的问题。通过结合语言模型和视觉模型，自动根据自然语言生成符合场景的动画轨迹。**

- **链接: [https://arxiv.org/pdf/2605.27203](https://arxiv.org/pdf/2605.27203)**

> **作者:** Mannat Khurana; Sanyam Jain; Rishav Agarwal
>
> **备注:** 5 pages, 6 figures
>
> **摘要:** Animation elevates digital documents into immersive experiences, yet creating custom motion paths remains cumbersome, requiring designers to manually select presets, plot Bézier points, and configure timing properties. We introduce Generative Animations, a system that transforms natural language prompts into production-ready animations. By chaining Large Language Models (LLMs) for semantic parsing with the Segment Anything Model (SAM) for visual grounding, our pipeline automatically generates motion paths that respect scene geometry, handle depth-based occlusions, and honor 3D perspective transforms. We demonstrate the system through three use cases: contour-following trajectories, orbital animations with z-order awareness, and perspective-aligned motion on transformed objects.
>
---
#### [new 031] Sentinel: Embodied Cooperative Spatial Reasoning and Planning
- **分类: cs.CV; cs.MA**

- **简介: 该论文研究多智能体协作空间推理与规划任务，解决在动态环境中协同导航和安全会合问题。提出CoSaR框架，结合语言通信与空间算法，提升协作效率与安全性。**

- **链接: [https://arxiv.org/pdf/2605.26239](https://arxiv.org/pdf/2605.26239)**

> **作者:** Xiangye Lin; Hongxin Zhang; Ruxi Deng; Qinhong Zhou; Chuang Gan
>
> **备注:** The first two authors contributed equally
>
> **摘要:** In this work, we study Cooperative Spatial Intelligence, the ability of decentralized embodied agents to coordinate effectively under dynamic environmental constraints across city-scale outdoor domains. We introduce Sentinel Challenge, a benchmark where multiple decentralized embodied agents must communicate in natural language to agree on a mutually safe and convenient meeting point within large, city-scale outdoor environments. Each agent must then navigate safely while avoiding dynamic sentinels patrolling the area, using a tool that provides coarse spatial information. To address this, we propose CoSaR (Cooperative Spatial Reasoning and Planning), a framework that bridges the high-level communication and planning abilities of foundation models with the precision of classical spatial navigation algorithms. CoSaR enables agents to exchange situational updates, reason over evolving spatial constraints, and collaboratively replan trajectories. Evaluated across 14 city-level scenes with 3-5 agents, CoSaR consistently leads to faster gathering, shorter path lengths, and improved safety. Our results demonstrate that integrating dynamic communication with spatial reasoning is essential for robust multi-agent cooperation. By formalizing this new setting and providing a scalable benchmark, we aim to build a foundation for advancing cooperative spatial intelligence in embodied multi-agent systems. Code and challenge are available at this https URL.
>
---
#### [new 032] Pop-Up Distractions Reveal Bag-of-Events Behavior in Video Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决VideoLLMs在时间关联上的问题。通过引入DistractionBench，发现模型存在bag-of-events行为，即错误关联不同片段事件。**

- **链接: [https://arxiv.org/pdf/2605.27101](https://arxiv.org/pdf/2605.27101)**

> **作者:** Oscar Chew; Serhii Honcharenko; Qian-Hui Chen; Patricia Lu; Dishant Zaveri; Khoa D. Doan; Kuan-Hao Huang
>
> **摘要:** A key capability for video understanding is reliably linking subjects to events across time, yet whether Video Large Language Models (VideoLLMs) actually achieve this remains unclear. In this work, we introduce DistractionBench to evaluate whether VideoLLMs can robustly link subjects and events in the presence of unrelated video segments. Through controlled interventions, such as inserting short advertisement clips into longer videos, we show that VideoLLMs frequently hallucinate interactions between entities from different segments, incorrectly attributing actions from injected advertisements to subjects in the main video. We characterize this systematic hallucination as bag-of-events (BoE) behavior, where models process videos as collections of events rather than temporally structured sequences. Evaluating 11 popular VideoLLMs, we find that all models exhibit substantial BoE behavior. Our findings suggest that VideoLLMs lack reliable mechanisms for temporal grounding and motivate the development of models with more robust subject-event association.
>
---
#### [new 033] REVERSE: Reinforcing Evidence Verification and Search for Agentic Image geo-localization
- **分类: cs.CV**

- **简介: 该论文属于图像地理定位任务，解决传统方法无法有效模拟人类多轮证据搜索与验证的问题。提出REVERSE框架，强化搜索与验证的交互，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.26861](https://arxiv.org/pdf/2605.26861)**

> **作者:** Yong Li; Furong Jia; Dacheng Yin; Kang Rong; Fengyun Rao; Jing Lyu; Fan Zhang
>
> **摘要:** Image geo-localization aims to determine where a photograph was taken, a task that often requires more than recognizing visible landmarks. Human experts typically solve it through an iterative workflow: they inspect informative regions, form location hypotheses, seek external evidence, and revise their judgments as new clues appear. Existing methods only partially capture this process: direct prediction methods bypass evidence acquisition altogether, while retrieval-augmented methods introduce external evidence but usually provide limited supervision on the intermediate decisions of where to search, how to query, and how to filter noisy results. We present REVERSE, a framework that reinforces the interplay between evidence search and verification to enable multi-turn agentic reasoning. REVERSE teaches three intermediate decisions: where to look, what to query, and what evidence to trust. To support this, we construct tool-grounded trajectories with annotated region selections, search observations, and geo-informative evidence labels, and introduce process rewards for visual grounding, query utility, and evidence discrimination. An offline search cache makes retrieval observations stable and reusable during reinforcement learning, enabling dense supervision over noisy search results. With a 4B model, REVERSE outperforms strong retrieval-augmented baselines and rivals substantially larger models on Im2GPS3k and YFCC4k. Code is available at this https URL.
>
---
#### [new 034] Respecting Modality Gap in Post-hoc Out-of-distribution Detection with Pre-trained Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于OOD检测任务，旨在解决预训练视觉-语言模型中模态差异导致的检测偏差问题。通过在线学习视觉原型提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.26661](https://arxiv.org/pdf/2605.26661)**

> **作者:** Yuanwei Hu; Bo Peng; Yadan Luo; Zhen Fang; Ling Chen; Jie Lu
>
> **摘要:** Out-of-distribution (OOD) detection has emerged as a popular technique to enhance the reliability of machine learning models by identifying unexpected inputs from unknown classes. Recent progress in pre-trained vision-language models (VLMs) has enabled zero-shot OOD detection without access to in-distribution (ID) training data; in this setting, existing methods commonly treat text embeddings of class names as class prototypes. In this paper, we challenge the widely adopted text-as-prototype paradigm by theoretically showing that off-the-shelf textual prototypes are generally misaligned with the optimal visual prototypes, yielding an intrinsic modality gap that cannot be eliminated by prompt engineering alone. To mitigate this gap under the post-hoc constraint, this paper presents an online pseudo-supervised framework that directly learns class prototypes in the visual feature space using unlabeled test-time data streams and soft predictions from the pre-trained VLMs. We provide theoretical guarantees for the convergence of the online optimization procedure. Extensive experiments empirically demonstrate that our method achieves a new state of the art across a variety of OOD detection setups.
>
---
#### [new 035] DuoGesture: Neuro-Inspired and Biomechanically Informed Dual-Stream Co-Speech Gesture Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于语音驱动手势生成任务，旨在解决语义表达与生物力学合理性之间的矛盾。提出DuoGesture模型，通过双流架构分离语义和节奏手势，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.26236](https://arxiv.org/pdf/2605.26236)**

> **作者:** Ferdinand Paar; Lanmiao Liu; Aslı Özyürek; Serge Thill; Esam Ghaleb
>
> **摘要:** Co-speech gesture generation requires both semantic expressivity and biomechanically plausible rhythmic motion. Existing holistic gesture models mix lexically grounded semantic gestures with frequent prosody-aligned beat gestures. This limits semantic grounding, speech-motion alignment, and kinematic smoothness. We propose \emph{DuoGesture}, a neuro-inspired and biomechanically informed dual-stream approach that decomposes co-speech gesture synthesis into coupled semantic and beat streams. The two streams are coordinated by a \emph{Semantic Variational Information Bottleneck}, a stochastic frame-level gate that learns when semantic gestures should override rhythmic beat motion. The semantic stream is controlled by \emph{Motion-Grounded Semantic Conditioning}, which replaces purely linguistic word embeddings with motion-language representations to provide motion-aligned semantic priors for long-tailed lexical triggers of gestures. The beat stream is further regularised by an \emph{Inertial Beat Prior}, an anthropometry-weighted arm-chain module that reduces jitter and improves rhythmic consistency without constraining semantic frames. Objective evaluations and subjective experiments show that DuoGesture outperforms strong holistic baselines, while component ablations confirm the complementary roles of semantic grounding, stochastic stream selection, and biomechanical regularisation.
>
---
#### [new 036] Towards Controllable Image Generation through Representation-Conditioned Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决如何有效引导扩散模型生成特定图像的问题。通过使用预训练自监督模型的表示进行条件控制，提升生成质量并实现可控生成。**

- **链接: [https://arxiv.org/pdf/2605.27343](https://arxiv.org/pdf/2605.27343)**

> **作者:** Nithesh Chandher Karthikeyan; Jonas Unger; Gabriel Eilertsen
>
> **摘要:** Diffusion models have emerged as powerful tools for high-quality image generation and editing, but guiding these models to produce specific outputs remains a challenge. Conventional approaches rely on conditioning mechanisms, such as text prompts or semantic maps, which require extensively annotated datasets. In this preliminary work, we explore diffusion models conditioned on representations from a pre-trained self-supervised model. The self-conditioning mechanism not only improves the quality of unconditional image generation, but also provides a representation space that can be used to control the generation. We explore this conditioning space by identifying directions of variations, and demonstrate promising properties in terms of smoothness and disentanglement.
>
---
#### [new 037] BioFact-MoE: Biologically Factorized Mixture of Experts for Vision-Language Prognostic Modeling in Hepatocellular Carcinoma
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于肝癌预后建模任务，解决现有模型无法区分肝功能与肿瘤因素的问题。提出BioFact-MoE框架，分离生物因素，提升预测准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2605.26376](https://arxiv.org/pdf/2605.26376)**

> **作者:** Junlin Yang; Tian Yu; Nicha C. Dvornek; Yuexi Du; Peiyu Duan; Annabella Shewarega; Lawrence H. Staib; James S. Duncan; Julius Chapiro
>
> **备注:** Early accepted at MICCAI 2026
>
> **摘要:** Hepatocellular carcinoma (HCC) is biologically heterogeneous, shaped by the interplay between hepatic functional reserve and tumor-related oncologic factors; thus, similar survival outcomes may reflect fundamentally different underlying biological processes. Prognostic modeling in HCC is informed by rich multimodal information from multiparametric MRI and radiology reports from routine clinical practice. Existing prognostic vision-language models (VLMs) learn a single entangled latent representation that blends hepatic and tumor-related factors, limiting both accuracy and biological interpretability. We present BioFact-MoE, a biologically factorized Mixture of Experts (MoE) framework that explicitly decomposes liver and tumor factors via biologically supervised experts within a residual MoE survival architecture. On a HCC cohort of N=588 patients (pretrained on 4,582 3D MRI image-report pairs), BioFact-MoE consistently improves survival prediction over all baselines across time horizons, achieving 12-, 18-, and 24-month AUCs of 75.33%, 75.85%, and 73.96%. Beyond scalar risk prediction, gated expert weights enable phenotype-aware risk stratification. Pathway-informed gating uncovers clinically meaningful treatment-associated survival heterogeneity. In held-out validation, hepatic and tumor embeddings show selective associations with liver function and tumor burden markers, respectively (p<0.05), without supervision. The code is available at this https URL.
>
---
#### [new 038] SpatialBench: Is Your Spatial Foundation Model an All-Round Player?
- **分类: cs.CV**

- **简介: 该论文属于空间基础模型评估任务，旨在解决模型泛化能力不足的问题。通过构建跨范式、多领域基准SpatialBench，全面评估模型性能，并提出新数据集与基线模型以推动发展。**

- **链接: [https://arxiv.org/pdf/2605.27367](https://arxiv.org/pdf/2605.27367)**

> **作者:** Haosong Peng; Hao Li; Jiaqi Chen; Yuhao Pan; Runmao Yao; Yalun Dai; Fushuo Huo; Fangzhou Hong; Zhaoxi Chen; Haozhao Wang; Dingwen Zhang; Ziwei Liu; Wenchao Xu
>
> **备注:** Project Page: this https URL
>
> **摘要:** While spatial foundation models have demonstrated impressive performance on standard datasets, a critical question remains: are they truly all-round players capable of generalizing robustly across diverse downstream tasks, arbitrary viewpoints, shifting scene domains, varying input densities, and specific hardware constraints? Answering this overarching question requires a holistic assessment, yet current models are mainly evaluated on specific domains for which they were specifically designed or trained. Such evaluations are intrinsically limited by narrow paradigm coverage, limited scene domains, and arbitrary frame sampling, making it fundamentally difficult to assess their true generalization capabilities. To address this gap, we present SpatialBench, a cross-paradigm, domain-diverse benchmark for spatial foundation models with deterministic sampling. SpatialBench features unprecedented scale and rigorous deterministic design, comprising 19 datasets and 546 scenes across 5 diverse spatial domains. It comprehensively evaluates 41 models across 6 paradigms on 5 task suites under 4 different input density settings. Our extensive evaluation reveals that current models are not yet all-round players, and uncovers crucial insights for future advancement. Specifically, we demonstrate that full-context attention maximizes accuracy while bounded-memory strategies unlock long-sequence scalability. Moreover, our empirical evaluations in challenging embodied and egocentric tasks demonstrate that strict domain alignment and high data quality are far more critical to performance than simple dataset scaling. Furthermore, to address the largest data gap identified in our analysis, we go beyond evaluation by introducing a large-scale dataset, DA-Next-5M, and a strong baseline model, DA-Next, pushing the boundaries of spatial representation learning.
>
---
#### [new 039] Multi-Modal Building Inspection via Perceiver IO Fusion of Satellite and Street-Level Imagery
- **分类: cs.CV**

- **简介: 该论文属于多模态建筑检测任务，解决卫星与街景图像融合问题。通过Perceiver IO架构实现多标签分类，提升屋顶元素识别精度。**

- **链接: [https://arxiv.org/pdf/2605.26381](https://arxiv.org/pdf/2605.26381)**

> **作者:** Niels Sombekke; Rob G.J. Wijnhoven; Martin R. Oswald
>
> **摘要:** We present a multi-modal classification framework that fuses satellite and street-level imagery through a Perceiver IO architecture operating on spatial patch tokens from a shared DINOv2 backbone. The design naturally handles a variable number of street-level views per building without padding or fixed-size pooling, and jointly predicts multi-label roof element and roof material classes. We construct a large-scale dataset of 32,135 buildings (61,672 segments) spanning ten countries, pairing satellite images with up to eight street-level views per segment and evaluating four masking strategies for isolating the target building. We propose an RGB-M masking strategy that appends the building footprint mask as a fourth input channel, providing a soft spatial prior that outperforms hard cropping across both modalities. The Perceiver IO fusion model improves over all other fusion strategies and yields substantial per-class gains for attributes visible from street level (e.g., +11.3 AP for slate, +1.3 AP for dormers), though the satellite-only baseline retains a slight advantage in macro-averaged mAP for classes that are predominantly visible from above. These results establish a scalable, flexible architecture for multi-modal building inspection that can accommodate heterogeneous inputs and multiple output tasks.
>
---
#### [new 040] RoadGIE: Towards A Global-Scale Aerial Benchmark for Generalizable Interactive Road Extraction
- **分类: cs.CV**

- **简介: 该论文属于遥感图像道路提取任务，旨在解决现有数据集多样性不足和交互方法效率低的问题。提出WorldRoadSeg-360K数据集和RoadGIE交互框架，提升道路分割精度与拓扑一致性。**

- **链接: [https://arxiv.org/pdf/2605.26862](https://arxiv.org/pdf/2605.26862)**

> **作者:** Chenxu Peng; Chenxu Wang; Yimian Dai; Yongxiang Liu; Ming-Ming Cheng; Xiang Li
>
> **摘要:** Accurate road segmentation from aerial imagery is fundamental to many geospatial applications. However, existing datasets often suffer from limited scene diversity, low semantic granularity, and poor structural continuity, restricting their generalization across environments. To address these challenges, we introduce WorldRoadSeg-360K, the largest and most diverse road segmentation dataset to date, comprising 366,947 high-resolution images collected from 38 countries and 223 cities across various terrains and continents. WorldRoadSeg-360K serves as a comprehensive benchmark and reveals key challenges in handling diverse and structurally complex scenes. Automated approaches often struggle to preserve road connectivity, while current interactive methods lack efficient, topology-sensitive tools for real-world road editing. To this end, we present RoadGIE, establishing a novel interactive paradigm for road extraction in remote sensing. Unlike prior point- or box-based prompting strategies, RoadGIE supports connectivity-aware prompts, including clicks and scribbles, which inherently align with the topology of road networks. To improve structural consistency and mitigate performance degradation during iterative interactions, RoadGIE integrates an expert-guided prompting strategy and adapts the skeleton-based recall loss for interactive scenarios. RoadGIE achieves state-of-the-art performance in both segmentation accuracy and topological consistency on WorldRoadSeg-360K and other benchmarks, while maintaining efficient operation with only 3.7M parameters. The code are publicly available at: this https URL
>
---
#### [new 041] Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from Monocular Videos
- **分类: cs.CV**

- **简介: 该论文属于单目人体运动恢复任务，解决运动过于平滑或动态不一致的问题。通过引入HTD-Refine框架，利用高阶时间动态信息提升运动的自然性与物理合理性。**

- **链接: [https://arxiv.org/pdf/2605.26879](https://arxiv.org/pdf/2605.26879)**

> **作者:** Dingkun Wei; Zehong Shen; Yan Xia; Georgios Pavlakos; Yujun Shen; Xiaowei Zhou
>
> **备注:** 13 pages, 6 figures. Accepted as an Oral presentation and Best Paper Candidate at CVPR 2026. Project page: this https URL
>
> **摘要:** Human motion recovered from monocular videos often appears overly smooth or dynamically inconsistent, even when joint positions are numerically accurate. We observe that this limitation stems from the absence of reliable high-order temporal cues -- velocity and acceleration -- which are essential for reconstructing motion that exhibits realistic momentum, timing, and high-frequency detail. We introduce HTD-Refine, a post-processing framework that augments existing Human Motion Recovery (HMR) pipelines using explicitly estimated high-order temporal dynamics. At the core of our system is PVA-Net, a temporal transformer that infers per-joint 2D positions, 3D velocities, and 3D accelerations directly from a monocular video. These predicted dynamics serve as soft yet informative constraints in a global optimization procedure that refines world-space trajectories, significantly reducing jitter, suppressing over-smoothing, and restoring physically plausible motion. Extensive experiments on challenging in-the-wild benchmarks show that HTD-Refine consistently improves state-of-the-art HMR methods, yielding more accurate global trajectories and substantially more natural motion dynamics. Our results highlight the critical role of high-order temporal modeling in advancing monocular human motion recovery.
>
---
#### [new 042] OSMa-Bench++: Toward Open-Ended Benchmarking of Semantic Mapping for Manipulation with Prompt-Generated Synthetic Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决语义映射评估不足的问题。通过生成合成场景扩展OSMa-Bench，提升基准测试的灵活性与实用性。**

- **链接: [https://arxiv.org/pdf/2605.26831](https://arxiv.org/pdf/2605.26831)**

> **作者:** Regina Kurkova; Maxim Popov; Sergey Kolyubin
>
> **备注:** Code: this https URL
>
> **摘要:** Semantic mapping methods are increasingly used as intermediate scene representations for downstream robotic reasoning and manipulation, yet their evaluation is still largely tied to fixed benchmark datasets with limited coverage of manipulation-relevant corner cases. In this work, we extend OSMa-Bench toward controllable benchmarking with prompt-generated synthetic indoor scenes. Our pipeline automatically generates scene descriptions, synthesizes corresponding environments with SceneSmith, and adapts the resulting assets into an OSMa-Bench-compatible simulation format. This adaptation requires a nontrivial intermediate layer, including semantic normalization, material and texture repair, shader fallback policies, floor handling, navigation setup, and controlled lighting configuration. A key advantage of the proposed setup is that the original scene-generation prompt is known in advance and can therefore serve as an auxiliary semantic specification of the intended scene. We use this property to extend the VQA component of OSMa-Bench with a prompt-grounded question category. The resulting framework supports targeted stress-testing of semantic scene representations under conditions such as clutter, small objects, partial occlusions, and lighting variation, and makes benchmarking more extensible and better aligned with downstream manipulation requirements. Our code is available at this https URL.
>
---
#### [new 043] Black-box Membership Inference Attacks on the Pre-training Data of Image-generation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于隐私安全任务，旨在检测图像生成模型的预训练数据泄露。针对现有方法在低暴露数据上效果差的问题，提出SD-MIA框架，通过分析模型对目标图像和文本指令的去噪行为来推断数据归属。**

- **链接: [https://arxiv.org/pdf/2605.27020](https://arxiv.org/pdf/2605.27020)**

> **作者:** Tao Qi; Huili Wang; Yuanhong Huang; Wendan Wang; Lianchao Zhao; Jinrui Wang; Zichen Qin; Shangguang Wang; Yongfeng Huang
>
> **备注:** 13 pages, 9 figures; CVPR 2026 camera-ready
>
> **摘要:** The rapid advancement of diffusion-based image generation models has raised serious concerns regarding potential copyright and privacy infringements involving human-created data. Membership inference attacks (MIAs) have emerged as a promising tool for identifying unauthorized data usage during model training. Existing methods typically assess the ability of model to denoise perturbed suspect images as an indicator of membership status. However, the discriminative power of such features is highly dependent on the degree of model memorization and deteriorates significantly when applied to less exposed data (e.g., pre-training data). Although several methods attempt to enhance detection by leveraging internal model features, these features are generally inaccessible in mainstream closed-source image generation platforms, limiting their practicality. In this paper, we demonstrate that analyzing how a black-box diffusion model denoises a target image and corresponding perturbed textual instructions can reveal more distinctive membership cues. Based on this insight, we propose a black-box membership inference attack framework (named SD-MIA) that leverages a cross-modal data perturbation mechanism to detect pre-training data in diffusion models. We conduct extensive experiments on both a public benchmark dataset and a newly constructed dataset, each comprising pre-training membership and non-membership samples with identical distributions. Experimental results demonstrate that SD-MIA achieves superior performance compared to existing baselines, including those with the unfair advantage of accessing internal model features.
>
---
#### [new 044] Receipt Replay OOD: A Small Benchmark for Screen Replay Detection Under Domain Shift
- **分类: cs.CV**

- **简介: 该论文属于屏幕重放检测任务，旨在解决域转移下的泛化性能问题。构建了Receipt Replay OOD数据集，用于评估模型在不同域间的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.26855](https://arxiv.org/pdf/2605.26855)**

> **作者:** Alexander Vinogradov
>
> **摘要:** Public datasets such as DLC-2021, SynID, and KID34K have significantly contributed to research on presentation attack detection for identity documents, including screen replay attacks. However, evaluation of out-of-domain (OOD) robustness remains insufficiently explored, especially under realistic domain shifts. In this work, we introduce Receipt Replay OOD, a small out-of-domain benchmark for screen replay detection. Receipts share several characteristics with identity documents, including planar geometry, curved corners, wear-and-tear artifacts, and text or logo patterns, while avoiding personally identifiable information constraints commonly associated with identity documents. We evaluate document replay detection models under cross-domain conditions and demonstrate the impact of domain shift on generalization performance. The dataset is publicly available.
>
---
#### [new 045] Model discovery for dynamical systems with complex-valued product units
- **分类: cs.CV**

- **简介: 该论文属于模型发现任务，旨在从轨迹数据中自动提取动力系统的控制方程。通过复杂值乘积单元网络，无需预设函数库即可学习方程，适用于高维系统。**

- **链接: [https://arxiv.org/pdf/2605.27158](https://arxiv.org/pdf/2605.27158)**

> **作者:** Martin Brückmann; Babette Dellen; Uwe Jaekel
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Discovering the governing equations of a dynamical system from observed trajectories provides deeper insight into its structure than mere prediction of future states. We present a data-driven approach to model discovery based on complex-valued product-unit networks, in which each unit represents a complex monomial and the network output is a sparse linear combination of such monomials. In contrast to established library-based methods such as SINDy, our approach does not require a predefined set of candidate functions: the relevant monomials, including those with fractional or negative exponents, are learned directly from data. Across four chaotic benchmark systems (Lorenz63, Lorenz84, the Four-Wing attractor, and a fractional variant of Lorenz63), we recover the exact governing equations in 90% of trials for the first three systems, and in 70-90% of trials for the fractional case, using at least 3000 training points. Applied to real-world human-gait accelerometer signals, the model produced stable trajectories with bounded prediction errors, corresponding to an RMSE of approximately 12-14% of the signal amplitude range over a test horizon three times longer than the training interval, demonstrating its potential for high-dimensional systems in which analytic equations are unavailable.
>
---
#### [new 046] CmIVTP: Cross-modal Interaction-based Vessel Trajectory Prediction for Maritime Intelligence
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于船舶轨迹预测任务，旨在解决单一数据源不足导致的预测不准确问题。通过融合AIS与CCTV数据，提出CmIVTP框架提升预测精度。**

- **链接: [https://arxiv.org/pdf/2605.26524](https://arxiv.org/pdf/2605.26524)**

> **作者:** Yuxu Lu; Dong Yang; Xiaoyu Li; Mengwei Bao; Congcong Zhao
>
> **摘要:** Maritime intelligent transportation systems (MITS) are essential for ensuring navigation safety and efficiency in busy waterways. However, accurate vessel trajectory prediction remains challenging due to the limitations of single-source data. Automatic identification system (AIS) data is often sparse or unavailable for small vessels, while closed-circuit television (CCTV) data alone cannot fully capture dynamic vessel behavior. To mitigate these challenges, we propose a cross-modal interaction-based vessel trajectory prediction (named CmIVTP) framework to model the intricate interactions between vessel dynamics and environmental constraints. Specifically, we introduce a target-aware scene encoder to extract scene semantic features, effectively capturing vessel-environment interactions and enhancing trajectory prediction accuracy. In addition, we propose a cross-modal interaction transformer, which integrates AIS-derived motion features, CCTV-based environmental features, and scene representations. It leverages cross-modal attention mechanisms to simultaneously capture intra-modal semantics and inter-modal interactions, ensuring dynamically consistent and environmentally feasible predictions. Furthermore, we construct a vessel group trajectory bank by clustering historical AIS trajectories into representative motion patterns, providing an efficient and scalable approach for candidate trajectory generation. Additionally, we introduce the maritime multimodal dataset plus (named Maritime-MmD$^+$), a large-scale dataset that synchronizes AIS data and CCTV video data, providing robust support for multimodal trajectory prediction research. Extensive experiments demonstrate that CmIVTP achieves better performance on multimodal-driven vessel trajectory prediction benchmarks. The code resources for this work can be available at this https URL.
>
---
#### [new 047] DelowlightSplat: Feed-Forward Gaussian Splatting for Lowlight 3D Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于低光环境下3D场景重建任务，解决低光导致的噪声和匹配不可靠问题。提出DelowlightSplat框架，通过低光适配器和多视角推理提升重建效果。**

- **链接: [https://arxiv.org/pdf/2605.26629](https://arxiv.org/pdf/2605.26629)**

> **作者:** Fuzhen Jiang; Zengtian Xie; Zhuoran Li
>
> **摘要:** Novel-view synthesis and 3D reconstruction from sparse posed images are central to robotics and AR/VR. Yet, feed-forward 3D Gaussian reconstruction fails under lowlight due to noise, color shifts, and unreliable correspondence. We propose DelowlightSplat, a lowlight-aware feed-forward Gaussian splatting framework for clean novel-view rendering. We build a controllable multi-view lowlight benchmark by degrading only context views while keeping target views clean. We introduce a lightweight Lowlight Adapter for residual enhancement to improve matchability, and couple it with cost-volume-based multi-view inference to directly predict clean 3D Gaussians. Experiments show that DelowlightSplat significantly outperforms previous feed-forward method and two-stage pipeline under lowlight conditions.
>
---
#### [new 048] Not All Modalities Are Equal: Instruction-Aware Gating for Multimodal Videos
- **分类: cs.CV**

- **简介: 该论文属于多模态视频理解任务，解决视频与辅助流融合时的模态干扰问题。提出UniMVU框架，通过动态门控机制实现指令感知的多模态融合。**

- **链接: [https://arxiv.org/pdf/2605.26232](https://arxiv.org/pdf/2605.26232)**

> **作者:** Bonan Ding; Umair Nawaz; Ufaq Khan; Abdelrahman M. Shaker; Muhammad Haris Khan; Jiale Cao; Jin Xie; Fahad Shahbaz Khan
>
> **备注:** 19 pages, 8 figures, 7 tables, preprint
>
> **摘要:** Pre-trained video large language models excel at visual reasoning. However, they struggle when videos arrive with auxiliary streams, such as audio, depth map, or dense temporal evidence. In such a scenario, uniform fusion induces modality interference, allowing irrelevant channels to distract the model. To address this issue, we present a unified multimodal video understanding framework, named UniMVU, that performs instruction-aware fusion across video, audio, depth map, or any other modality inputs via two levels of dynamic gating: inner-modality gates emphasize salient regions within each modality, whereas modality-level gates re-weight whole streams; both are conditioned on the text instruction to adaptively balance modality importance. Our UniMVU combines cross-modal self-attention with instruction-driven inner-modality gating module and a modality-level gating module with control token; for time-aligned streams we further adopt a fast-to-slow fusion scheme that reduces redundancy. Across six benchmarks (AVQA, AVSD, Music-AVQA, ScanQA, SQA3D and MVBench), our UniMVU achieves consistent gains over static-fusion baselines achieving gains as high as 13.5 in terms of CIDEr metric. Further, our analysis shows that the gating mechanism aligns with the human-interpretable modality relevance, and ablations show the contributions of inner-modality and modality-level gating. Our UniMVU provides a simple, unified recipe for instruction-aware multimodal video understanding that scales to diverse modalities without hand-crafted fusion rules.
>
---
#### [new 049] Chaos-SSL: An Attention-Based Self-Supervised Learning Framework with Chaotic Transformation for Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决标注数据不足和细粒度纹理识别难题。提出Chaos-SSL框架，结合混沌变换和注意力融合，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27146](https://arxiv.org/pdf/2605.27146)**

> **作者:** Joao Batista Florindo
>
> **摘要:** Self-Supervised Learning (SSL) has emerged as a powerful paradigm to mitigate the reliance on large, annotated datasets, a common bottleneck in medical image analysis. However, standard SSL methods, which rely on simple geometric and color augmentations, may fail to capture the fine-grained, complex textural details necessary for classifying subtle pathologies. This paper introduces Chaos-SSL, a novel two-stage framework for medical image classification. In the first stage, we propose a new self-supervised pre-training strategy that leverages 1D chaotic maps (Logistic, Tent, and Sine) as a complex, non-linear augmentation for contrastive learning. We hypothesize that these chaotic transformations create ``harder'' and more semantically-rich views, forcing a network to learn robust representations of fine-grained medical textures. In the second stage, we introduce an attention-based fusion model that dynamically combines the specialized features from our Chaos-SSL model with the general-purpose features of a larger, ImageNet-pre-trained model. We validate our method on two public datasets: ISIC 2018 (skin lesions) and APTOS 2019 (diabetic retinopathy). Our results demonstrate that the Chaos-SSL model pre-trained with a Tent map for 30 epochs, followed by attention fusion, achieves performance fully competitive with the state-of-the-art, yielding an accuracy of 0.9261 on ISIC 2018 and 0.8726 on APTOS 2019. This significantly outperforms existing SSL methods, including several recent approaches.
>
---
#### [new 050] Image Thresholding: Understanding Bias of Evaluation Metrics towards Specific Evaluation Functions
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，研究评估指标对阈值函数的偏差问题。通过分析不同阈值方法与评价指标的相关性，揭示了Otsu方法在PSNR和SSIM上表现更优，提出需建立更中立的评估框架。**

- **链接: [https://arxiv.org/pdf/2605.27132](https://arxiv.org/pdf/2605.27132)**

> **作者:** Eslam Hegazy; Mohamed Gabr
>
> **备注:** Submitted to ICPR 2026 (this https URL)
>
> **摘要:** Multilevel image thresholding is widely used for segmentation in applications ranging from medical imaging to remote sensing. Classical objective functions, such as Otsu's between-class variance and Kapur's entropy, are often optimized using metaheuristic algorithms, with performance evaluated via metrics like Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR). These evaluations implicitly assume that SSIM and PSNR provide unbiased measures of segmentation quality. In this study, we examine this assumption by analyzing the correlation between thresholding objective functions and quality metrics across all possible thresholds for images in the BSDS500 dataset. Results show that Otsu's criterion consistently exhibits high correlation with both SSIM and PSNR, while Kapur's entropy demonstrates weaker and more variable correlation. Otsu outperforms Kapur in correlation with PSNR for all images and with SSIM for over 91%. Our findings reveal an inherent metric-objective-function bias. This work highlights the need for more neutral evaluation frameworks and motivates extending the analysis to additional thresholding criteria and domains. Source code of this paper can be found at this https URL
>
---
#### [new 051] LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出LocateAnything，解决视觉-语言定位与检测任务中的效率与精度问题。通过并行框解码提升推理速度和定位准确性。**

- **链接: [https://arxiv.org/pdf/2605.27365](https://arxiv.org/pdf/2605.27365)**

> **作者:** Shihao Wang; Shilong Liu; Yuanguo Kuang; Xinyu Wei; Yangzhou Liu; Zhiqi Li; Yunze Man; Guo Chen; Andrew Tao; Guilin Liu; Jan Kautz; Lei Zhang; Zhiding Yu
>
> **摘要:** Vision-language models (VLMs) commonly formulate visual grounding and detection as a coordinate-token generation problem, serializing each 2D box into multiple 1D tokens that are learned and decoded largely independently. This token-by-token decoding mismatches the coupled structure of box geometry and creates a practical inference bottleneck due to strictly sequential generation. We introduce LocateAnything, a unified generative grounding and detection framework based on Parallel Box Decoding (PBD). By decoding geometric elements such as bounding boxes and points as atomic units in a single step, LocateAnything preserves intra-box geometric coherence and unlocks substantial parallelism. We show that PBD improves both decoding throughput and localization accuracy. We further develop a scalable data engine and curate LocateAnything-Data, a large-scale dataset with more than 138 million training samples, substantially increasing data diversity for high-precision localization. Extensive evaluations show that LocateAnything advances the speed-accuracy frontier, achieving significantly higher decoding throughput while improving high-IoU localization quality across diverse benchmarks. The results highlight the complementary benefits of Parallel Box Decoding and large-scale training data in enabling efficient and precise unified visual grounding and detection.
>
---
#### [new 052] CIRCLED: A Multi-turn CIR Dataset with Consistent Dialogues across Domains
- **分类: cs.CV**

- **简介: 该论文属于多轮图像检索任务，旨在解决现有数据集对话历史不一致和领域受限的问题。通过扩展多个数据集并优化生成流程，构建了更大更通用的CIRCLED数据集。**

- **链接: [https://arxiv.org/pdf/2605.26734](https://arxiv.org/pdf/2605.26734)**

> **作者:** Tomohisa Takeda; Yu-Chieh Lin; Yuji Nozawa; Youyang Ng; Osamu Torii; Yusuke Matsui
>
> **摘要:** Existing Multi-Turn Composed Image Retrieval (MTCIR) datasets lack dialogue-history consistency and are restricted to the fashion domain. To address these limitations, we construct CIRCLED by extending FashionIQ, CIRR, and CIRCO. In CIRCLED, the query at each turn progressively approaches the target image. Data are generated via a CIReVL-based retrieval pipeline and curated with multiple filters on retrieval success, turn length, consistency, and information redundancy to ensure quality. In total, we collect 22,608 multi-turn sessions across nine subsets, substantially exceeding Multi-turn FashionIQ (11,505 sessions) in both scale and generality. We further apply multiple baseline methods and quantitatively assess retrieval accuracy on CIRCLED. Our work provides a practical, high-quality benchmark to facilitate future research on multi-turn CIR. The dataset and code are publicly available at this https URL and this https URL.
>
---
#### [new 053] JetViT: Efficient High-Resolution Vision Transformer with Post-Training Attention Search
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出JetViT，解决高分辨率图像处理中的效率与精度平衡问题。通过后训练注意力搜索，提升Vision Transformer的推理效率。**

- **链接: [https://arxiv.org/pdf/2605.26636](https://arxiv.org/pdf/2605.26636)**

> **作者:** Dongyun Zou; Zhuoyang Zhang; Junyu Chen; Wenkun He; Qinhe Peng; Hanrong Ye; Yao Lu; Hongxu Yin; Yu Wang; Song Han; Han Cai
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** We introduce JetViT, a novel family of hybrid-architecture Vision Transformer (ViT) models that match the accuracy of state-of-the-art full-attention vision foundation models while achieving substantially higher inference efficiency on high-resolution images. At the core of our approach is Post-Training Attention Search, a post-training acceleration framework that converts pre-trained full-attention ViTs into efficient hybrid-attention variants by identifying and replacing redundant full-attention blocks with linear or window-attention blocks. By inheriting the MLP and attention weights from the base model, Post-Training Attention Search efficiently explores the architectural design space through three key steps: (1) optimizing the linear-attention block design; (2) finding the best combination of linear-attention and window-attention blocks; and (3) identifying and preserving critical full-attention blocks. We evaluate JetViT on two representative high-resolution vision foundation models, DINOv3 and DepthAnythingV2. On the NVIDIA H100 GPU, JetViT achieves up to 1.79x higher throughput and up to 44.81% lower latency without sacrificing accuracy. We will release our code and accelerated ViT models soon.
>
---
#### [new 054] Sparse-LiDAR Prompting of Monocular Geometry Foundations: An Empirical Study Toward Long-Range Driving Depth
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决长距离驾驶场景中稀疏LiDAR提示下的深度预测问题。提出SLIM模型，首次适配MoGe-2以接受真实稀疏LiDAR输入，并通过多尺度融合提升性能。**

- **链接: [https://arxiv.org/pdf/2605.26456](https://arxiv.org/pdf/2605.26456)**

> **作者:** Kai Zheng; Qiang Feng; Xingjian Liu; Wenquan Tan; Yuan Li
>
> **备注:** 6 pages, 3 figures, 2 tables
>
> **摘要:** Sparse-LiDAR-prompted depth foundation models (PromptDA, Prior Depth Anything, DMD3C) have shown strong results on indoor scenes or within KITTI's standard 80-meter evaluation cap. However, two limitations remain: (i) systematic distance-stratified evaluation in long-range driving regimes (50-150 m) is largely absent; (ii) prior approaches built on disparity-based foundations rely on pre-interpolated dense priors, leaving truly sparse LiDAR injection on point-map foundations (e.g., MoGe-2, NeurIPS 2025) unexplored. We present SLIM (Sparse-LiDAR Injected Monocular geometry), the first adaptation of MoGe-2 to accept truly sparse LiDAR input. SLIM integrates a partial-convolution sparse encoder with a multi-scale fusion neck that fuses LiDAR features into the point-map decoder at five scales. We adopt density-agnostic training (random injection ratio in [0.005, 0.30]) so a single model serves diverse input densities. On Virtual KITTI and CARLA, SLIM reduces the absolute relative error of the MoGe-2 baseline by approximately 39-51% at 100-150 m. Ablation across six injection ratios shows partial-convolution injection improves both AbsRel and RMSE on Virtual KITTI in all six settings; on CARLA, AbsRel improves in five of six settings (one near-tie at 0.015 differs by 0.0013), and RMSE is comparable across encoders, with partial-convolution improving in three settings (by up to 0.31 unit) and losing by at most 0.11 unit in the other three.
>
---
#### [new 055] A Dynamic Programming Framework for Discovering Count and Values of Multilevel Image Thresholding
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在自动确定多级阈值数量。提出一种基于动态规划和改进MET准则的方法，解决传统方法需手动输入阈值数的问题，并验证其效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.27287](https://arxiv.org/pdf/2605.27287)**

> **作者:** Eslam Hegazy; Mohamed Gabr
>
> **摘要:** Multilevel Image thresholding is an important preprocessing algorithm in computer vision applications nowadays. Since most common thresholding methods take the desired count of thresholds as input by the user, thresholding methods that automatically determines a suitable count of thresholds from the input image itself are advantageous. In this article, a novel thresholding method based on a dynamic programming algorithm and a modification of Minimum Error Thresholding (MET) criterion is thoroughly presented. An empirical statistical study is performed to pinpoint why this proposed method is superior. Moreover, an extended comparison between this proposed method and other state-of-the-art methods is performed on a comprehensive set of natural, satellite and medical test images. The numerical results show that the proposed MET-DP method takes much less time than traditional dynamic programming thresholding methods when the number of thresholds is high. The proposed method can detect a suitable count of thresholds for most of tested images of different types. However, traditional methods that take the count of thresholds as input produce thresholded images of higher structural similarity index measure (SSIM) and peak signal-to-noise ratio (PSNR) values than MET-DP. Source code can be found on this https URL
>
---
#### [new 056] Self-Intersection-Aware 3D Human Motion Generation Using an Efficient Human Sphere Proxy
- **分类: cs.CV**

- **简介: 该论文属于人体运动生成任务，解决生成动作中自相交问题。通过引入基于球体代理的损失函数，有效减少自相交并提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.26744](https://arxiv.org/pdf/2605.26744)**

> **作者:** Pascal Herrmann; Maarten Bieshaar; Dennis Mack; Robert Herzog; Juergen Gall
>
> **备注:** Accepted to BMVC 2025
>
> **摘要:** Human motion generation has made tremendous progress in recent years, with state-of-the-art approaches surpassing ground truth data in leading evaluation benchmarks. However, visual inspection of the generated motions paints a different picture. Even state-of-the-art approaches generate motions frequently containing self-intersections, i.e., body parts interpenetrating, which are strong artifacts, severely limiting the perceived motion quality. We introduce a novel loss, which explicitly penalizes self-intersections, to the training of human motion generation methods. We base our loss on a sphere proxy of human geometry, which allows us to calculate a self-intersection loss 98% faster and uses 83% less memory than comparable methods based on triangular meshes. The loss is agnostic to the specific approach, and we add it to the training of the recent human motion generation methods human motion diffusion model (MDM) and MoMask. Our extensive experiments show a reduction of self-intersections in generated motions of up to 49% while improving other evaluation metrics. The code is available at this https URL .
>
---
#### [new 057] Dimensional Distribution Emotion State: Leveraging Valence and Arousal as a Common Embedding Space for Visual Emotion Analysis
- **分类: cs.CV**

- **简介: 该论文属于视觉情感分析任务，旨在解决艺术作品情感预测问题。通过引入DDES表示和多数据集训练，提升情感建模效果。**

- **链接: [https://arxiv.org/pdf/2605.26262](https://arxiv.org/pdf/2605.26262)**

> **作者:** Émile Bergeron; Tadagbé Dhossou; Sébastien Tremblay; Jean-François Lalonde
>
> **摘要:** Museums are important sites for the dissemination of culture and art. They are institutions rooted in history and tradition; their exhibitions are often designed to highlight these aspects. Recently, a new approach is being explored in the field: emotion-based exhibitions. These exhibitions are designed specifically to elicit emotions in the visitors, in order to maximize engagement, and as a way to democratize access to art and attract a wider, more diverse audience. To do so, the emotional content of the artworks must first be extracted, however, manually annotating the artworks by experts is a prohibitively labor-intensive process, and risks introducing the personal bias of curators. To assist the museum curators in their design of these exhibitions, we wish to develop a tool that can predict the emotional response evoked by a work of art. In this article, we leverage a continuous bi-dimensional emotion space to enhance emotion representations and the training process of deep learning models. Drawing inspiration from existing categorical and dimensional emotion representations, we introduce a new representation, Dimensional Distribution Emotion State (DDES), along with a pipeline for multi-dataset training. We show that DDES provides multiple advantages compared to widely used representations while exhibiting similar baseline performance.
>
---
#### [new 058] IPIBench: Evaluating Interactive Proactive Intelligence of MLLMs under Continuous Streams
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型的交互智能评估任务，旨在解决实时流媒体中主动推理与反应式查询的协同问题。研究提出IPIBench基准和IPI-Agent框架，提升模型在动态多轮交互中的表现。**

- **链接: [https://arxiv.org/pdf/2605.27074](https://arxiv.org/pdf/2605.27074)**

> **作者:** Jinzhao Li; Yinuo Chen; Wenxuan Song; Yijia Lei; Yichi Zhang; Honglei Yan; Panwang Pan; Miao Liu
>
> **摘要:** Recent multimodal large language models (MLLMs) achieve strong performance on reactive question answering, but real-world streaming assistants require proactive reasoning over continuous visual inputs. Existing benchmarks mainly study reactive or proactive interactions in isolated single-turn settings, overlooking dynamic multi-turn scenarios where users may add, modify, or cancel proactive requests alongside interleaved reactive queries. To address this gap, we introduce IPIBench, the first benchmark for evaluating Interactive Proactive Intelligence of MLLMs under streaming video settings. IPIBench covers proactive monitoring, proactive task management, and interleaved reactive-proactive requests. Evaluations on representative MLLMs reveal two major limitations: unstable proactive triggering and weak coordination between reactive and proactive behaviors. We further propose IPI-Agent, a training-free agentic framework with an interaction-control policy and a temporal-gating mechanism for stabilizing proactive triggering and coordinating multi-turn interactions. Experiments show that IPI-Agent consistently improves existing MLLMs across all benchmark settings.
>
---
#### [new 059] AnchorDiff: Training-Free Concept Grounding for MM-DiTs via Anchor-Based Graph Propagation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态视觉定位任务，解决概念泄漏问题。提出AnchorDiff方法，通过图传播实现无训练的语义定位，减少误识别。**

- **链接: [https://arxiv.org/pdf/2605.26460](https://arxiv.org/pdf/2605.26460)**

> **作者:** Jian Zhang; Zhijun Zhang
>
> **摘要:** Multi-Modal Diffusion Transformers (MM-DiTs) encode rich representations for training-free concept grounding, but existing attention-based methods often produce overlapping activations on visually confusable concepts, a failure mode we call concept leakage, where target responses spill over to non-target objects. To address this issue, we propose AnchorDiff, a training-free grounding method that decouples semantic localization from structural refinement. AnchorDiff selects a high-confidence anchor from concept-to-image attention map and propagates it as a one-hot seed over a hybrid graph derived from image-to-image self-attention. The graph uses output-space similarity for dense within-object propagation and a row-wise attention gate to suppress cross-object connections. Additionally, we introduce the Multi-Concept Confusion Dataset, which contains images with multiple visually similar concepts and separate masks, enabling explicit evaluation of concept leakage. Experiments show that AnchorDiff achieves strong grounding performance on ImageNet-Segmentation and PascalVOC, while substantially reducing concept leakage on our Multi-Concept Confusion Dataset.
>
---
#### [new 060] Small Object Detection in Industrial Recycling: A New Dataset and YOLO Performance Evaluation
- **分类: cs.CV; cs.DB**

- **简介: 该论文属于目标检测任务，旨在解决工业回收中小而密集、重叠物体的检测问题。通过构建新数据集并评估YOLO性能，提出改进方法与未来方向。**

- **链接: [https://arxiv.org/pdf/2605.26884](https://arxiv.org/pdf/2605.26884)**

> **作者:** Oussama Messai; Abbass Zein-Eddine; Abdelouahid Bentamou; Mickael Picq; Nicolas Duquesne; Stéphane Puydarrieux; Yann Gavet
>
> **摘要:** In this paper, we address the problem of detecting small, dense, and overlapping objects, a major challenge in computer vision. Our focus is on reviewing proposed methods based on deep learning supervised approaches. We provide a detailed comparison of these systems on a new dataset of more than 10k images and 120k instances, highlighting their performance, accuracy, and computational efficiency in the industrial recycling process use case. Through this comparative analysis, we identify the most reliable systems currently available and the specific challenges they are designed to tackle. Furthermore, we explore the benefits of data augmentation and synthetic images. Based on our analysis, we also propose potential future directions and innovative solutions that could enhance the effectiveness of small, dense and overlapped object detection systems. The scope of our investigations encompasses object detection, length measurement, and anomaly detection within the context of the recycling process. The anomaly detection strategy is robust against variations in image resolution and zoom levels, ensuring reliable performance in industrial applications. The repository of the proposed dataset, methods and evaluation codes can be found at: this https URL
>
---
#### [new 061] METATR: A Multilingual, Evolving Benchmark for Automatic Text Recognition
- **分类: cs.CV**

- **简介: 该论文提出METATR，一个用于自动文本识别（ATR）的多语言、动态基准，解决现有数据集多样性不足的问题，通过多样化文档评估模型性能。**

- **链接: [https://arxiv.org/pdf/2605.26712](https://arxiv.org/pdf/2605.26712)**

> **作者:** Mélodie Boillet; Solène Tarride; Christopher Kermorvant
>
> **摘要:** Benchmarks that reflect the diversity and complexity of real-world documents are essential for accurately evaluating Automatic Text Recognition (ATR) systems, especially Vision-Large Language Models (vLLMs). Although recent models demonstrate impressive performance, they are often evaluated on datasets containing modern, printed texts mostly written in English, which limits their relevance to many practical applications. Therefore, selecting a model for a specific use case requires evaluating it on data that matches the target documents. This highlights the importance of representative benchmarks for real-world applications. In this paper, we introduce METATR (v1.0), a multilingual, evolving benchmark designed to evaluate ATR models across a wide range of documents, facilitating meaningful model comparison and selection. The benchmark was designed to maximize diversity by including documents from various public collections. These documents cover 29 languages and include texts with multiple scripts and layouts. Beyond the dataset itself, METATR defines a standardized prompting and normalization methodology and establishes a dynamic evaluation framework. This approach is intended to produce reproducible results while remaining extensible over time. We evaluated a wide range of state-of-the-art systems, including open-source models and closed-source models. Results are reported across various dimensions, including performance at the dataset and language levels, robustness to handwritten documents, and computational efficiency. Our findings show that, although proprietary models achieve the most consistent performance, substantial variability persists across scripts and layouts. Overall, METATR provides a multidimensional, practitioner-oriented framework for assessing multilingual ATR in real-world conditions and tracking progress as the field evolves.
>
---
#### [new 062] RadarSim: Simulating Single-Chip Radar via Multimodal Neural Fields
- **分类: cs.CV**

- **简介: 该论文提出RadarSim，属于雷达数据模拟任务，旨在解决雷达数据难解释和传感器差异大的问题，通过结合相机信息生成更清晰的雷达图像。**

- **链接: [https://arxiv.org/pdf/2605.26328](https://arxiv.org/pdf/2605.26328)**

> **作者:** Chuhan Chen; Tianshu Huang; Akarsh Prabhakara; Chaithanya Kumar Mummadi; Zhongxiao Cong; Anthony Rowe; Matthew O'Toole; Deva Ramanan
>
> **备注:** Accepted to 3DV 2026. Project website: this https URL
>
> **摘要:** Radars are an ideal complement to cameras: both are inexpensive, solid-state sensors, with cameras offering fine angular resolution, while radars provide metric depth and robustness under adverse weather. However, radar data is more difficult to interpret than camera images and varies significantly between sensors, necessitating increased reliance on simulation for prototyping sensors and processing pipelines. Recent work treating radar reconstruction as a novel view synthesis problem has shown great promise in reconstructing radar-relevant geometry and simulating low-level radar data. However, such methods are constrained by the low spatial resolution of the underlying radar. To address this, we propose a unified differentiable renderer, RadarSim, which leverages the high angular resolution of RGB cameras to generate Doppler radar range images from a camera-initialized neural field. Using a novel data set of calibrated radar camera recordings from a custom hand-held rig, we demonstrate that RadarSim produces sharper geometry and Doppler range frames than radar-only reconstructions.
>
---
#### [new 063] TrackRef3D: Multi-View Consistent Track-then-Label for Open-World Referring Segmentation in 3D Gaussian Splatting
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D目标分割任务，解决多视角不一致和依赖人工标注的问题。提出TrackRef3D，通过轨迹感知语义共识和混合训练策略，实现无需标注的开放世界 referring 分割。**

- **链接: [https://arxiv.org/pdf/2605.26576](https://arxiv.org/pdf/2605.26576)**

> **作者:** Yuyang Tan; Renhe Zhang; Hang Zhang; Ao Li; Xin Tan
>
> **摘要:** Referring 3D Gaussian Splatting (R3DGS), which utilizes natural language for 3D object segmentation, has emerged as a crucial capability for embodied AI. However, existing methods typically rely on expensive per-scene manual annotation and per-view pseudo mask generation, which suffer from multi-view inconsistency and poor generalization to varying query specificities. To address this, we present TrackRef3D, a fully automatic pipeline that achieves open-world referring segmentation in 3D Gaussian Splatting (3DGS) without manual annotation by introducing a multi-view consistent track-then-label paradigm that fundamentally decouples object discovery from semantic grounding. Specifically, we propose a Trajectory-Aware Semantic Consensus Module (TSCM) which aggregates cross-view predictions via synonymous clustering and trajectory-aware voting to establish a canonical semantic identity, thereby ensuring multi-view consistency. Furthermore, we employ a visibility-aware description generation strategy to mitigate ambiguity and propose a Hybrid Training Strategy (HTS) that jointly optimizes coarse category semantics and fine-grained referential cues to ensure robustness under varying query specificities using a multi-positive contrastive objective. Extensive experiments on benchmarks demonstrate that TrackRef3D achieves state-of-the-art performance.
>
---
#### [new 064] JLT: Clean-Latent Prediction in Latent Diffusion Transformers
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，研究 latent diffusion 中的预测目标选择问题。通过对比 clean-latent 与 velocity 预测，发现前者在特定表示下表现更优，揭示了预测目标的几何依赖性。**

- **链接: [https://arxiv.org/pdf/2605.27102](https://arxiv.org/pdf/2605.27102)**

> **作者:** Funing Fu; Tenghui Wang; Junyong Cen; Qichao Zhu; Guanyu Zhou
>
> **摘要:** Flow matching with clean-data prediction has shown that regressing the clean point can exploit low-dimensional structure more effectively than predicting an ambient noised quantity. We ask whether this principle remains useful after images are mapped into a learned latent space, where compression has already removed much of the raw pixel variability. We introduce JLT, a 130M latent diffusion Transformer over frozen FLUX.2 VAE codes, and compare clean-latent prediction with a matched velocity-prediction DiT under the same representation, backbone, and training settings. Although the three variables x, epsilon, and v are linearly convertible for a fixed corruption time, a local Gaussian analysis shows that velocity regression inherits an isotropic target-covariance floor and amplifies low-variance latent directions, while clean prediction damps them. On ImageNet 256 x 256, JLT-B/1 obtains FID-50K 2.50 with classifier-free guidance, with a large matched-target gap over velocity prediction. These results suggest that prediction targets in latent diffusion are representation-dependent geometric choices, rather than interchangeable algebraic parameterizations.
>
---
#### [new 065] PARE: Pruning and Adaptive Routing for Efficient Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散Transformer计算成本高的问题。通过剪枝和自适应路由，动态压缩模型宽度和深度，提升效率并保持质量。**

- **链接: [https://arxiv.org/pdf/2605.27336](https://arxiv.org/pdf/2605.27336)**

> **作者:** Yutong Wang; Yunke Wang; Tianfan Xue; Yu Qiao; Yaohui Wang; Xinyuan Chen; Chang Xu
>
> **摘要:** Video Diffusion Transformers (DiTs) generate high-quality videos but demand substantial compute due to wide blocks, deep architectures, and iterative sampling. Recent methods reduce cost by compressing width, depth, or sampling steps, but typically commit to a fixed architecture that cannot adapt to individual inputs or denoising stages. We propose PARE (Pruning and Adaptive Routing for Efficient video generation), which jointly compresses width and depth with structure-aware pruning and input-adaptive routing. For width, we observe that attention heads specialize into spatial and temporal roles, and design importance scoring that accounts for this distinction to prevent motion-critical temporal heads from being pruned prematurely. For depth, we train a lightweight router conditioned on denoising timestep and visual content to dynamically select which blocks to execute at each step, enabling per-input compute adaptation rather than static block removal. A progressive pipeline first recovers width-pruned quality via distillation, then jointly optimizes the student and router to decouple the two learning objectives. Experiments on Wan2.1-14B for both image-to-video and text-to-video generation show that PARE substantially reduces per-step computation while preserving quality across VBench dimensions, and composes with step distillation for further acceleration.
>
---
#### [new 066] Semantic Robustness Probing via Inpainting: An Interactive Tool for Safety-Critical Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SemProbe，用于安全关键场景下的目标检测模型语义鲁棒性测试，通过可控修复技术进行语义探针分析。**

- **链接: [https://arxiv.org/pdf/2605.27155](https://arxiv.org/pdf/2605.27155)**

> **作者:** Nico Steckhan; Krutarth Prajapati; Weija Shao; Silvia Vock
>
> **摘要:** Testing object detectors in safety-critical domains requires semantically meaningful probes beyond pixel-level corruptions. We present SemProbe, a tool for semantic robustness probing: users upload deployment images, create masks manually or automatically, select operational design domain-derived factors (or custom prompts), and run diffusion-based controlled inpainting. The system supports batch jobs, parallel seed/workflow variations, and configurable generation parameters. After each output, model inference runs automatically and displays annotated before/after comparisons with performance deltas. All probes are logged as structured artifacts, enabling traceable robustness evidence aligned with safety evaluation workflows. We demonstrate \textsc{SemProbe} on hand detection for dimension saws, targeting factors from insurance-oriented test criteria.
>
---
#### [new 067] Timestep-Aware SVDQuant-GPTQ for W4A4 Quantization of Wan2.2-I2V
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成模型的量化任务，解决W4A4量化中的激活异常和时间步依赖问题，提出结合SVDQuant和GPTQ的框架，提升量化精度。**

- **链接: [https://arxiv.org/pdf/2605.27003](https://arxiv.org/pdf/2605.27003)**

> **作者:** Junhao Wu; Dezhong Yao; Hai Jin
>
> **摘要:** W4A4 quantization of large video diffusion Transformers offers substantial memory savings but is hindered by two main challenges: sparse large-magnitude activation outliers, and strongly timestep-dependent activation distributions across the multi-step denoising trajectory. These difficulties are compounded by Wan2.2-I2V's two-expert Mixture-of-Experts DiT design, whose high-noise and low-noise experts exhibit distinct quantization sensitivities that a single global calibration policy cannot capture. We propose a post-training quantization framework combining SVDQuant-based low-rank outlier compensation, GPTQ-based reconstruction-aware residual weight quantization, and timestep-bin-wise per-layer activation clipping-ratio search conducted independently for each expert. On the OpenS2V-Eval benchmark, our method reduces peak GPU memory by 59.3\% relative to the BF16 baseline while incurring only a 0.9\% drop in VBench average score and a 2.3\% drop in Imaging Quality, demonstrating that expert- and timestep-aware calibration is essential for high-fidelity W4A4 inference on MoE video DiTs.
>
---
#### [new 068] SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising
- **分类: cs.CV**

- **简介: 该论文属于点云去噪任务，旨在解决噪声干扰下点与表面对应关系不明确的问题。提出SIMPC方法，通过生成镜像点并保持一致性，实现无监督去噪。**

- **链接: [https://arxiv.org/pdf/2605.26894](https://arxiv.org/pdf/2605.26894)**

> **作者:** Chengwei Zhang; Xueyi Zhang; Tao Jiang; Xinhao Xu; Wenjie Li; Fubo Zhang; Longyong Chen
>
> **备注:** Accepted by ICML 2026. 17 pages, 8 figures, 8 tables
>
> **摘要:** In point clouds, noise directly perturbs point coordinates that encode both spatial location and geometry, making one-to-one correspondence construction more challenging than in images. Existing methods impose statistical mappings across noisy variants via noise or optimal transport, but suffer from correspondence ambiguity. In this work, we propose Self-Induced Mirror-Point Consistency (SIMPC) to learn deterministic correspondences between points and the underlying surface in an unsupervised manner. For each noisy point, SIMPC generates a mirror-point on the opposite side of the underlying surface, guided by geometric priors during the denoising process. By encouraging consistency between the denoising targets of the original point and its mirror counterpart, SIMPC effectively localizes the position of underlying surface. Extensive experiments on synthetic and real-world datasets demonstrate that SIMPC significantly outperforms state-of-the-art unsupervised methods and surpasses several strong supervised counterparts.
>
---
#### [new 069] ChartAct: A Benchmark for Dynamic Chart Understanding
- **分类: cs.CV**

- **简介: 该论文提出ChartAct，一个动态图表理解的基准测试，解决真实交互环境中图表分析的问题。通过构建高质量数据集，评估多模态模型在动态图表中的表现。**

- **链接: [https://arxiv.org/pdf/2605.26994](https://arxiv.org/pdf/2605.26994)**

> **作者:** Muye Huang; Wu Lin; Lingling Zhang; Hang Yan; Zhiyuan Wang; Yumeng Fu; Zesheng Yang; Jun Liu
>
> **摘要:** Charts are widely used to present complex data for analysis and decision making. Existing chart understanding benchmarks mainly focus on static charts, but real-world charts are often dynamic and interactive. Key information may only appear after actions such as hovering, clicking, zooming, or dragging. Dynamic chart understanding therefore requires models to identify visible content, choose proper interactions, and reason over changing chart states. To evaluate this ability, we propose ChartAct, an interactive benchmark for dynamic chart understanding. ChartAct collects and filters 673 dynamic charts from 8 real chart websites, covers 7 common chart types, and constructs 1,440 high-quality question-answer samples. Each sample is instantiated in two environments, Dynamic Chart and Dashboard Chart, to evaluate dynamic chart understanding under different contexts. Based on ChartAct, we systematically evaluate 11 advanced multimodal models and GUI agents. Experimental results show that existing models still have clear limitations in dynamic chart understanding. The strongest model, Claude-Opus-4.7, achieves an average success rate of 84.5\%, while most models remain below 60\%. We also conduct detailed failure attribution and case analysis. ChartAct provides a new benchmark for studying chart understanding in real interactive environments. Codes at this https URL
>
---
#### [new 070] SCKAN: Structural Consensus-based KAN Prototype Learning for Semi-Supervised Pancreas Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督胰腺分割任务，旨在解决标注稀缺下的模型泛化能力不足问题。提出SCKAN方法，通过结构一致性学习和KAN融合提升分割精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.27032](https://arxiv.org/pdf/2605.27032)**

> **作者:** Yuqi Liu; Yufei Chen; Wei Fu; Xiaodong Yue; Shuo Li
>
> **备注:** 10.5 pages, 5 figures, Medical Image Computing and Computer Assisted Intervention 2026
>
> **摘要:** Accurate pancreas segmentation is critical for early cancer diagnosis, where annotation scarcity necessitates Semi-Supervised Learning (SSL). However, due to significant inter-sample morphological variability, existing SSL methods face severe generalizability limitations under sparse supervision, leading to the Supervision Bias problem. To address this, we propose Structural Consensus-based KAN Prototype Learning (SCKAN), which constructs the first cross-sample structural consensus learning with Kolmogorov-Arnold Networks (KANs), to achieve more generalizable and accurate segmentation. Specifically, SCKAN contains two key designs: Structure-constrained Prototype Consistency Learning (SPCL), which prompts unbiased structural representation by enforcing cross-sample consistency via prototype-level contrastive optimization, and Consensus-based Kolmogorov-Arnold Fusion (CKaF), which reduces morphology-specific bias by aggregating stable consensus and filtering sample-wise noise via KAN's adaptive B-spline nonlinearity. Extensive experiments on two public pancreas datasets demonstrate the effectiveness of SCKAN. Code is at this https URL.
>
---
#### [new 071] Cross-scale Aligned Supervision for Training GANs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生成对抗网络（GAN）训练任务，旨在解决多尺度生成中的跨尺度轨迹对齐问题。通过引入CAT模型，实现中间输出与最终输出的一致性约束，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.26449](https://arxiv.org/pdf/2605.26449)**

> **作者:** Sangeek Hyun; MinKyu Lee; Jae-Pil Heo
>
> **备注:** Preprint
>
> **摘要:** Modern GANs often introduce adversarial supervision on intermediate generator outputs and interpret the resulting multi-stage synthesis as coarse-to-fine hierarchical generation. In this work, we challenge this interpretation. We argue that standard scale-wise adversarial supervision does not construct a proper coarse-to-fine hierarchy: each intermediate image is independently pushed toward the real distribution at its own resolution, but this scale-wise realism does not ensure that outputs across stages represent the identical generated sample. Moreover, the scale-specific image produced at each stage is not used as an explicit refinement target for the subsequent stage. Therefore, its adversarial loss can improve a scale-specific output without constraining later stages to preserve the same sample trajectory, allowing them to move toward a different sample rather than refine the previous output. We refer to this problem as a cross-scale trajectory misalignment problem. To resolve it, we propose CAT, a Cross-scale Aligned Transformer for multi-scale adversarial generation. CAT keeps the discriminator scale-wise, so each intermediate output is evaluated at its own resolution, while adding a simple generator-side consistency regularization that aligns intermediate outputs with the final output. On class-conditional ImageNet-256, CAT-H/2 achieves an FID-50K of 1.56 with one-step inference after only 60 training epochs, outperforming strong one-step GAN and diffusion/flow baselines.
>
---
#### [new 072] MSCGC-KAN: Multi-scale Causal Graph Convolution and Kolmogorov-Arnold Feature Mapping for EEG Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文属于EEG情感识别任务，旨在解决多尺度情感动态建模、通道间功能连通性利用不足及分类器表达能力弱的问题。提出MSCGC-KAN方法，通过多尺度因果图卷积和Kolmogorov-Arnold映射提升性能。**

- **链接: [https://arxiv.org/pdf/2605.26624](https://arxiv.org/pdf/2605.26624)**

> **作者:** Haoliang Gong; Qingshan She; Jiale Xua; Yunyan Gao; Xugang Xi
>
> **摘要:** Electroencephalogram (EEG)-based emotion recognition is an important affective computing task, and recent EEG foundation models provide useful generic representations for downstream adaptation. However, under the fine-tuning setting, three limitations remain prominent: insufficient modeling of multi-scale emotional dynamics, inadequate exploitation of inter-channel functional connectivity, and the limited expressive power of simple linear classification heads. To address these issues, this paper proposes a new EEG emotion recognition method, termed MSCGC-KAN, which introduces a structured task head composed of multi-scale causal graph convolution and Kolmogorov--Arnold feature mapping. Built on a pre-trained CBraMod backbone, MSCGC-KAN enhances downstream adaptation by jointly strengthening multi-scale temporal modeling, learnable inter-channel connectivity modeling, and nonlinear discriminative mapping within a compact task-specific head. This design preserves the representation advantage of the foundation model while making the classifier more sensitive to emotion-related spatiotemporal patterns. Extensive experiments are conducted on the public FACED and SEED-VII datasets. The proposed method achieves a balanced accuracy of 60.66\%, a Cohen's Kappa of 0.5525, and a weighted F1-score of 60.40\% on FACED, and obtains 33.27\%, 0.2223, and 33.64\%, respectively, on SEED-VII. Compared with the CBraMod+Linear baseline, the balanced accuracy is improved by 5.91 and 2.03 percentage points on the two datasets, respectively. These results indicate that structured task-head design is an effective way to improve EEG emotion recognition when fine-tuning pre-trained EEG models.
>
---
#### [new 073] Underwater360: Reconstructing Underwater Scenes from Panoramic Images with Omnidirectional Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于 underwater scene reconstruction 任务，旨在解决 underwater rendering 中的几何失真和介质退化问题。提出 Underwater360 框架，结合 omnidirectional 3DGS 和物理模型，提升全景水下场景重建质量。**

- **链接: [https://arxiv.org/pdf/2605.26447](https://arxiv.org/pdf/2605.26447)**

> **作者:** Jiangbei Hu; Weichao Song; Shibo Yu; Mohan Wang; Zihan Yi; Rui Wu; Mingkang Xiang; Na Lei; Shengfa Wang; Zhongxuan Luo; Ying He
>
> **摘要:** Underwater scene reconstruction is essential for immersive exploration of aquatic environments, yet remains challenging due to complex participating-media effects such as absorption and scattering, as well as the limited field of view (FoV) of conventional cameras. Although combining panoramic imaging with 3D Gaussian Splatting (3DGS) offers a promising direction for photorealistic underwater rendering, traditional 3DGS struggles with both spherical projection distortion and underwater medium degradation. In this paper, we propose \textbf{Underwater360}, a physics-informed omnidirectional 3DGS framework for underwater panoramic scene reconstruction. First, we introduce an Omnidirectional Gaussian Splatting module that performs ray casting directly in spherical camera space instead of relying on 2D projection approximations, thereby reducing geometric distortions under 360$^\circ$ FoV. Second, we design a physics-based appearance-medium modeling architecture with pose-conditioned appearance embeddings to explicitly decouple intrinsic scene radiance from depth-dependent backscatter and attenuation, enabling physically grounded scene appearance restoration. Finally, we establish a new panoramic underwater benchmark dataset containing both synthetic and real-world scenes. Extensive experiments demonstrate that Underwater360 achieves superior performance in underwater novel view synthesis and scene appearance restoration, delivering improved rendering quality and cross-view consistency in complex underwater environments. The code and datasets are released at this https URL
>
---
#### [new 074] Benchmarking Convolutional, Transformer, Hybrid, and Vision Language Models for Multi Disease Retinal Screening
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多疾病视网膜筛查任务，旨在比较不同视觉模型在真实场景下的表现。通过基准测试，评估了多种模型架构在二分类和多标签分类任务中的效果，为临床自动化筛查提供参考。**

- **链接: [https://arxiv.org/pdf/2605.26283](https://arxiv.org/pdf/2605.26283)**

> **作者:** Durjoy Dey; Aymane Ajbar; Yuhong Yan
>
> **备注:** 12 pages, 3 figures, accepted at ICMHI 2026, 10th International Conference on Medical and Health Informatics, Kyoto, Japan. To appear in ACM Conference Proceedings
>
> **摘要:** Modern deep learning offers powerful tools for automated retinal screening, but it remains unclear how different visual model families compare in realistic multi-disease settings and under domain shift. In this work, we benchmark twelve architectures across four model families: convolutional neural networks, vision transformers, hybrid CNN-transformer backbones, and vision-language models, using the Retinal Fundus Multi-disease Image Dataset (RFMiD). We evaluate two tasks: binary screening for any retinal disease and multi-label classification across 28 disease classes. Using standardized training, calibration, and evaluation protocols, we report AUC, F1, precision, recall, and sensitivity at a clinically relevant operating point with specificity near 80%. On RFMiD, all architectures perform well on binary screening, with AUC above 84%, but attention-based models perform best. SwinTiny and the hybrid CoAtNet0 and MaxViTTiny models achieve the strongest binary screening results and improve macro and micro F1 in the multi-label setting. Vision-language models, including CLIP ViT-B/16 and SigLIP-Base384, are competitive with CNN baselines but do not surpass the best transformer and hybrid backbones. In external validation on Messidor-2 for referable diabetic retinopathy, AUC ranges from 66.8% to 84.7%, with hybrid and transformer models again showing strong performance. These results provide a reproducible reference for model selection in multi-disease retinal screening and guide future automated screening tools for clinical deployment.
>
---
#### [new 075] A multifractal-based masked auto-encoder: an application to medical images
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决传统MAE随机遮蔽忽略关键区域的问题。通过引入多分形分析优化遮蔽策略，提升模型对复杂组织结构的重建能力。**

- **链接: [https://arxiv.org/pdf/2605.26287](https://arxiv.org/pdf/2605.26287)**

> **作者:** Joao Batista Florindo; Viviane de Moura
>
> **摘要:** Masked autoencoders (MAE) have shown great promise in medical image classification. However, the random masking strategy employed by traditional MAEs may overlook critical areas in medical images, where even subtle changes can indicate disease. To address this limitation, we propose a novel approach that utilizes a multifractal measure (Renyi entropy) to optimize the masking strategy. Our method, termed Multifractal-Optimized Masked Autoencoder (MO-MAE), employs a multifractal analysis to identify regions of high complexity and information content. By focusing the masking process on these areas, MO-MAE ensures that the model learns to reconstruct the most diagnostically relevant features. This approach is particularly beneficial for medical imaging, where fine-grained inspection of tissue structures is crucial for accurate diagnosis. We evaluate MO-MAE on several medical datasets covering various diseases, including MedMNIST and COVID-CT. Our results demonstrate that MO-MAE achieves promising performance, surpassing other basiline and state-of-the-art models. The proposed method also adds minimum computational overhead as the computation of the proposed measure is straightforward. Our findings suggest that the multifractal-optimized masking strategy enhances the model's ability to capture and reconstruct complex tissue structures, leading to more accurate and efficient medical image representation. The proposed MO-MAE framework offers a promising direction for improving the accuracy and efficiency of deep learning models in medical image analysis, potentially advancing the field of computer-aided diagnosis.
>
---
#### [new 076] Cesarean Scar Defect Segmentation in Transvaginal Ultrasound Images: a Dataset and Benchmark
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决CSD漏诊问题。构建了首个公开的经阴道超声CSD数据集，包含标注图像和视频，用于提升分割算法与临床诊断水平。**

- **链接: [https://arxiv.org/pdf/2605.26774](https://arxiv.org/pdf/2605.26774)**

> **作者:** Yuan Tian; Yue Li; Wei Xia; Tianyu Xu; Jian Zhang; Liye Shi; Jing Liu; Yang Wang; Ming Liu; Qing Xu; Yixuan Zhang; Maggie M. He; Xiangjian He
>
> **摘要:** Cesarean Scar Defect (CSD) is one of the most prevalent complications following cesarean delivery. Transvaginal ultrasonography is widely used for primary CSD screening. Accurate determination of CSD outline and dimensions is crucial for treatment. However, CSDs are frequently overlooked by sonographers due to small size and irregular morphology, suboptimal image quality, and limited clinical awareness in resource-constrained settings. Despite artificial intelligence advances in medical imaging, no public dataset exists for transvaginal ultrasound CSD segmentation. To address this gap, we present a comprehensive CSD dataset comprising 1,111 images and 16 videos, yielding 501 positive samples with confirmed CSD and precise pixel-level manual annotations. Annotations are performed following standardized clinical guidelines through collaboration between experienced sonographers and trained PhD students. This work provides high-quality benchmark resources for advancing medical image segmentation algorithms and promoting clinical innovation. Ultimately, improved CSD diagnosis and subsequent treatment strategies can enhance the quality of life in women of reproductive age, representing significant value for both medical research and clinical practice.
>
---
#### [new 077] VesselSim: learning 3D blood vessel segmentation without expert annotations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决血管标注数据稀缺的问题。通过生成合成数据并训练模型，实现无需专家标注的3D血管分割。**

- **链接: [https://arxiv.org/pdf/2605.26277](https://arxiv.org/pdf/2605.26277)**

> **作者:** Erin Rainville; Melissa Ananian; Tristan Mirolla; Hassan Rivax; Yiming Xiao
>
> **备注:** This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution will be published as part of the MICCAI 2026 proceedings in October
>
> **摘要:** Blood vessel segmentation is a core task in medical image analysis for the care of vascular diseases and surgical planning, yet the challenges of providing expert vascular annotations pose a major obstacle for the progress of related deep learning techniques. To address this, we propose VesselSim, a two-stage framework for universal 3D blood vessel segmentation that eliminates the need for real annotated data during training. First, we introduce a stochastic, geometry-driven vascular simulation framework that models recursive branching, curvature-controlled growth, and collision-aware topology, followed by domain-randomized intensity synthesis to generate 16,500 anatomically plausible 3D angiographic volumes. Second, a 3D U-Net is trained solely on this synthetic data. To bridge the domain gap from synthetic to real images at inference time, we introduce a test-time adaptation strategy via a self-supervised mask reconstruction decoder, enabling adaptation to unseen clinical scans without prior domain knowledge. We evaluate VesselSim in a zero-shot setting on multiple real-world datasets spanning MR and CT across several anatomical regions, including the brain and kidneys. Despite being trained exclusively on synthetic data, VesselSim achieves performance competitive with state-of-the-art vascular segmentation foundation models. These findings suggest that learning vessel geometry from synthetic tubular structures is effective for robust cross-domain generalization, substantially reducing the reliance on acquired medical imaging data and more importantly, expert annotations.
>
---
#### [new 078] Rethinking Weakly-supervised Video Temporal Grounding From a Game Perspective
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究弱监督视频时间定位任务，解决跨模态对齐和候选片段质量依赖问题。提出游戏视角方法，通过合作博弈理论建模视频帧与查询词的关联，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.26441](https://arxiv.org/pdf/2605.26441)**

> **作者:** Xiang Fang; Zeyu Xiong; Wanlong Fang; Xiaoye Qu; Chen Chen; Jianfeng Dong; Keke Tang; Pan Zhou; Yu Cheng; Daizong Liu
>
> **备注:** Published in ECCV 2024
>
> **摘要:** This paper addresses the challenging task of weakly-supervised video temporal grounding. Existing approaches are generally based on the moment proposal selection framework that utilizes contrastive learning and reconstruction paradigm for scoring the pre-defined moment proposals. Although they have achieved significant progress, we argue that their current frameworks have overlooked two indispensable issues: 1) Coarse-grained cross-modal learning: previous methods solely capture the global video-level alignment with the query, failing to model the detailed consistency between video frames and query words for accurately grounding the moment boundaries. 2) Complex moment proposals: their performance severely relies on the quality of proposals, which are also time-consuming and complicated for selection. To this end, in this paper, we make the first attempt to tackle this task from a novel game perspective, which effectively learns the uncertain relationship between each vision-language pair with diverse granularity and flexible combination for multi-level cross-modal this http URL, we creatively model each video frame and query word as game players with multivariate cooperative game theory to learn their contribution to the cross-modal similarity score. By quantifying the trend of frame-word cooperation within a coalition via the game-theoretic interaction, we are able to value all uncertain but possible correspondence between frames and words. Finally, instead of using moment proposals, we utilize the learned query-guided frame-wise scores for better moment this http URL show that our method achieves superior performance on both Charades-STA and ActivityNet Caption datasets.
>
---
#### [new 079] Semi-Supervised Gaze Estimation via Disentangled Subspace Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉 gaze 估计任务，旨在解决标注数据不足导致的泛化能力差问题。通过半监督学习和解耦子空间对比学习，提升模型在少量标注数据下的性能。**

- **链接: [https://arxiv.org/pdf/2605.27080](https://arxiv.org/pdf/2605.27080)**

> **作者:** Qida Tan; Hongyu Yang; Wenchao Du
>
> **备注:** ICML2026
>
> **摘要:** Appearance-based gaze estimation always suffers from poor generalization due to limited annotated samples and insufficient dataset diversity. Leading approaches adopt weakly supervised learning to generate large-scale pseudo-labeled data from unconstrained real-world scenarios, aiming to mitigate the domain shifts. In this work, we devise a simple yet effective semi-supervised learning architecture that leverages unlabeled data to enhance domain generalization, thereby reducing reliance on labor-intensive manual annotations. Our key insight is to impose Jacobian regularization to disentangle feature representations into discriminative subspaces dedicated to specific gaze components, such as pitch and yaw angles. We further exploit the intrinsic ordinal ranking within each subspace for contrastive learning, enabling the model to learn robust gaze representations from a small set of labeled samples and an abundance of unlabeled ones. This ultimately yields our Disentangled Subspace Contrastive Learning (DSCL) framework. Extensive experiments on multiple benchmarks verify that the proposed DSCL is plug-and-play, achieving competitive performance using only 20\%, 10\%, and even 5\% of the annotated data under both in-domain and cross-domain evaluation settings. The public code is available at \href{this https URL}{this https URL}.
>
---
#### [new 080] RoMo: A Large-Scale, Richly Organized Dataset and Semantic Taxonomy for Human Motion Generation
- **分类: cs.CV**

- **简介: 该论文提出RoMo数据集，解决3D人体运动生成中数据质量与规模的矛盾，通过精细标注和语义分类提升模型性能。任务为人体运动生成。**

- **链接: [https://arxiv.org/pdf/2605.26241](https://arxiv.org/pdf/2605.26241)**

> **作者:** Jiahao Zhang; Joseph Liu; Young-Yoon Lee; Seonghyeon Moon; Victor Zordan; Guy Tevet; Karen Liu; Stephen Gould; Oren Jacob; Haomiao Jiang; Mubbasir Kapadia; Yizhak Ben-Shabat
>
> **备注:** Accepted to CVPR'26
>
> **摘要:** Success in generative modeling across language, image, and video demonstrates that large, well-curated datasets are the key driver for building capable models. 3D Human motion, however, has lagged behind, constrained by an unsatisfying choice between small, high-fidelity motion capture datasets and large-scale in-the-wild collections dominated by static or low-quality sequences. We introduce RoMo, a rich, large-scale, carefully curated dataset of in-the-wild human motions that resolves these tradeoffs. To ensure quality, we introduce a taxonomy-aware filtering pipeline that aggressively removes static and artifact-prone sequences. Every sequence is annotated with detailed captions and organized by a novel three-level semantic taxonomy. This hierarchical structure enables fine-grained, per-category evaluation, that reveals model strengths and weaknesses obscured by global metrics. We demonstrate that models trained on RoMo achieve state-of-the-art fidelity and diversity while gaining a superior understanding of complex, subtle text prompts. Finally, we release the Motion Toolbox to standardize metrics, data conversion, and visualization, establishing a foundation for reproducible and interpretable motion generation research.
>
---
#### [new 081] CSV-ViT: A Vision Transformer with the Variable-sized Cortical Supervertices for Detection of Alzheimer's Disease Pathologies
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于阿尔茨海默病检测任务，旨在解决MRI数据中脑皮层表面分析的挑战。提出CSV-ViT模型，通过可变大小的皮层超顶点进行有效特征提取与分类。**

- **链接: [https://arxiv.org/pdf/2605.26514](https://arxiv.org/pdf/2605.26514)**

> **作者:** Geonwoo Baek; Ikbeom Jang
>
> **摘要:** Confirming Alzheimer's disease (AD) typically relies on positron emission tomography (PET), which remains costly and invasive, motivating the use of structural MRI-based prescreening. Deep learning on non-Euclidean manifolds, particularly brain cortical surfaces, faces significant challenges due to the data's spherical topology. Recent surface models have enabled learning from cortical surface data; however, imposing face-based uniform patches often causes duplicate vertices at patch boundaries. In general, many surface-based models are limited in their awareness of the region of interest (ROI), which can result in non-cortical regions, such as the medial wall, being included. We propose a cortical surface tokenization that performs ROI-preserving, vertex-based, variable-sized patch partitioning. We refer to these cortical surface patches as cortical supervertices (CSVs). Building on this representation, we design the CSV Vision Transformer (CSV-ViT), a variable-size patch-tolerant Vision Transformer that uses padding and a mask-aware patch embedding. We used T1-weighted MRI and evaluated our framework by classifying AD-related status into three categories: AD diagnosis, amyloid positivity, and tau positivity. Across the experiments, CSV-ViT achieved higher classification performance than recent surface-based models. The results suggest that the proposed CSV-ViT may support MRI-based prediction of AD-related status prior to PET or CSF confirmation.
>
---
#### [new 082] Memory-Distilled Selection for Noise-Robust Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于工业异常检测任务，解决数据污染下的鲁棒性问题。提出MeDS算法，通过数据选择和记忆蒸馏提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.26676](https://arxiv.org/pdf/2605.26676)**

> **作者:** Sirojbek Safarov; Jaewoo Park; Yoon Gyo Jung; Kuan-Chuan Peng; Wonchul Kim; Seongdeok Bang; Octavia Camps
>
> **备注:** Accepted by ICML2026. The code is available at this https URL
>
> **摘要:** Anomaly detection (AD) under data contamination is critical for deploying unsupervised defect detection in industrial environments, where curating perfectly clean training sets is impractical. However, existing methods are sensitive to contamination, suffering significant performance degradation as the noise ratio increases. In this paper, we propose Memory-Distilled Selection (MeDS), a training algorithm based on data selection. MeDS constructs an ensemble of partial memories via random subsampling, where the resulting sparsity acts as a low-pass filter that captures nominal patterns across a wide range of noise ratios, enabling coarse-level identification of contaminated samples. The aggregated distances to the bootstrapped memories are then distilled into a reconstruction score network, which is subsequently fine-tuned on clean data filtered using scores from the distilled model, enabling fine-grained localization of anomalies. MeDS is robust across a wide range of noise ratios without requiring noise-ratio-specific hyperparameter tuning, achieving 99.16\% image-level AUROC on MVTecAD at a 40\% noise ratio, and attaining state-of-the-art performance on both VisA and Real-IAD under noisy settings. We thoroughly verify the efficacy of MeDS on industrial AD benchmarks under noisy data scenarios, accompanied by in-depth empirical analyses.
>
---
#### [new 083] LongAV-Compass: Towards Unified Evaluation of Minute-Scale Audio-Visual Generation Across T2AV, I2AV, and V2AV
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音频视频生成任务，旨在解决长时序、多模态生成的评估问题。提出LongAV-Compass基准，支持T2AV、I2AV和V2AV的统一评估。**

- **链接: [https://arxiv.org/pdf/2605.26244](https://arxiv.org/pdf/2605.26244)**

> **作者:** Tengfei Liu; Yang Shi; Xuanyu Zhu; Jiafu Tang; Liu Yang; Qixun Wang; Zhuoran Zhang; Yuqi Tang; Fengxiang Wang; Yuhao Dong; Xinlong Chen; Bozhou Li; Bohan Zeng; Yue Ding; Xiaohan Zhang; Jialu Chen; Haotian Wang; Yuanxing Zhang; Pengfei Wan; Leye Wang
>
> **摘要:** Audio-visual generation is rapidly advancing from short clips to minute-long content, while existing evaluation protocols remain largely confined to short-form settings. Existing benchmarks primarily focus on 5--10 second text-conditioned generation and rarely support unified evaluation across text, image, and video conditioning modalities. Moreover, they provide limited insight into how identity consistency, narrative coherence, and audio-visual alignment degrade over extended temporal horizons. To bridge this gap, we introduce LongAV-Compass, a systematic benchmark for minute-long audio-visual generation. LongAV-Compass contains 284 curated test cases spanning text-to-audio-video (T2AV), image-to-audio-video (I2AV), and video-to-audio-video (V2AV), organized by application scenario and generation complexity. The benchmark combines taxonomy-guided benchmark construction with a unified evaluation framework that integrates MLLM-assisted assessment with complementary perceptual and multimodal metrics, including DINO-v2, ArcFace, CLIP, and ImageBind. The framework evaluates more than 20 fine-grained dimensions covering within-segment quality, cross-segment consistency, global narrative coherence, semantic alignment, and audio-visual synchronization. Through experiments on 11 representative models together with human-alignment validation, LongAV-Compass provides a diagnostic testbed for analyzing the limitations of current systems in sustaining coherent, semantically aligned, and temporally consistent minute-scale audio-visual generation across diverse input modalities.
>
---
#### [new 084] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在提升智能体在复杂3D环境中的导航能力。通过构建3D高斯地图和开放集语义分组，增强场景理解与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.26500](https://arxiv.org/pdf/2605.26500)**

> **作者:** Jianzhe Gao; Rui Liu; Wenguan Wang
>
> **摘要:** Vision-language navigation (VLN) requires an agent to traverse complex 3D environments based on natural language instructions, necessitating a thorough scene understanding. While existing works equip agents with various scene representations to enhance spatial awareness, they often neglect the complex 3D geometry and rich semantics in VLN scenarios, limiting the ability to generalize across diverse and unseen environments. To address these challenges, this work proposes a 3D Gaussian Map that represents the environment as a set of differentiable 3D Gaussians and accordingly develops a navigation strategy for VLN. Specifically, Egocentric Scene Map is constructed online by initializing 3D Gaussians from sparse pseudo-lidar point clouds, providing informative geometric priors for scene understanding. Each Gaussian primitive is further enriched through Open-Set Semantic Grouping operation, which groups 3D Gaussians based on their membership in object instances or stuff categories within the open world, resulting in a unified 3D Gaussian Map. Building on this map, Multi-Level Action Prediction strategy, which combines spatial-semantic cues at multiple granularities, is designed to assist agents in decision-making. Extensive experiments conducted on three public benchmarks (i.e., R2R, R4R, and REVERIE) validate the effectiveness of our method.
>
---
#### [new 085] FTibSuite: A Comprehensive Resource Suite for Tibetan Vision-Language Modeling
- **分类: cs.CV; cs.DB**

- **简介: 该论文属于视觉-语言建模任务，旨在解决藏语低资源问题。提出FTibSuite，包含数据、基准和基线模型，提升藏语多模态研究的标准化水平。**

- **链接: [https://arxiv.org/pdf/2605.26601](https://arxiv.org/pdf/2605.26601)**

> **作者:** Guixian Xu; Yide Liang; Zeli Su; Xuexian Song; Ziyin Zhang; Yushuang Dong; Ting Zhang; Xu Han
>
> **摘要:** Vision-language models have progressed rapidly, but Tibetan remains a severely underserved low-resource language due to the lack of reproducible training and evaluation infrastructure. To fill this gap, we introduce FTibSuite, a comprehensive resource suite for Tibetan vision-language research, consisting of FTibData (human-verified multimodal training corpora spanning continual pretraining, image-text alignment, and instruction tuning data), FTibBench (Tibetan adaptations of five mainstream multimodal benchmarks with a hierarchical quality-control workflow to reduce translation noise), and FTibVLM, a reproducible baseline built on Qwen3-VL-8B-Instruct via a three-stage adaptation pipeline. Experiments on FTibBench show FTibVLM delivers consistent performance gains across all tasks, such as improving MMBench accuracy from 42.97 to 67.78 and POPE-random accuracy from 47.53 to 80.56, while retaining the backbone's original Chinese capabilities with minimal degradation, providing the first standardized foundation for Tibetan multimodal research.
>
---
#### [new 086] O-MARC: Omni Memory-Augmented Compression Distillation for Efficient Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，解决长序列推理成本高和音频视觉关联不明确的问题。提出OMAC和O-MARC方法，提升模型效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.26584](https://arxiv.org/pdf/2605.26584)**

> **作者:** Peiran Wu; Yunze Liu; Chi-Hao Wu; Chen Chen; Junxiao Shen
>
> **摘要:** Omnimodal large language models enable unified audio video understanding, but long joint token sequences make inference costly, and existing benchmarks do not fully isolate audio visual association in noisy user generated videos. We introduce UGC-AVQA, a public UGC benchmark with 1,000 videos and 4,816 QA pairs, where an audio removal test ensures that benchmark questions require both acoustic and visual evidence. To reduce inference cost, we propose OMAC, a training free plug in compression method that preserves salient visual memory and temporally grounded audio anchors. To further make compact models robust to compressed inputs, we introduce O-MARC, a compression distillation framework for learning with memory compressed multimodal contexts. On Qwen2.5-Omni-3B, O-MARC improves the average score across four benchmarks to 45.8, outperforming full token inference at 44.1 and OmniZip at 41.0. OMAC also keeps inference efficient, reducing latency by 34.6\% (1.53$\times$ speedup) and memory by 34.7\% compared with full token inference.
>
---
#### [new 087] NeR-SC: Adapting Neural Video Representation to Screen Content
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频压缩任务，针对屏幕内容视频的特殊性提出NeR-SC框架，解决自然场景方法不适用的问题，通过引入特定模块提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2605.27024](https://arxiv.org/pdf/2605.27024)**

> **作者:** Ruohan Shi; Jiaoyan Zhao; Haogang Feng
>
> **备注:** Submitted to PRMVAI 2026
>
> **摘要:** Implicit neural representations have emerged as a promising paradigm for video compression, with recent methods achieving competitive performance on natural video. However, screen content video -- common in remote desktop, online education, and cloud gaming -- exhibits distinct statistics: sharp edges, limited color palettes, and strong temporal redundancy. Existing neural representation methods, designed for natural scenes, lack mechanisms to exploit these properties, leaving substantial room for improvement. In this paper, we propose NeR-SC, a neural representation framework tailored for screen content video. Building on the SNeRV backbone, NeR-SC introduces three screen-content-specific modules: (i) a learnable color palette that models the discrete color structure of screen content by restricting the low-frequency sub-band to a learned color set; (ii) a multi-gate dense fusion module that replaces sequential feature fusion with dense, attention-gated cross-stage interaction; and (iii) an embedding-level frame skip strategy that bypasses redundant decoder invocations for static frames, with zero training overhead. Experiments on DSCVC and VCD show that NeR-SC achieves 40.32~dB and 41.73~dB average PSNR, outperforming representative neural video representation methods and, at low bitrates, surpassing H.264 and H.265. The skip strategy enables real-time decoding with no loss in quality.
>
---
#### [new 088] Frequency-Guided Fusion For RGB-Thermal Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于RGB-热红外语义分割任务，解决复杂环境下信息不足的问题。通过多模态融合架构，提升场景理解效果。**

- **链接: [https://arxiv.org/pdf/2605.26273](https://arxiv.org/pdf/2605.26273)**

> **作者:** İsmail Emre Canıtez; Özgür Erkent
>
> **备注:** 9 pages, 7 figures, To be Presented at Perception Beyond the Visible Spectrum workshop series (IEEE PBVS) at CVPR, 2026
>
> **摘要:** Semantic segmentation in complex environments such as urban driving scenes remains challenging under adverse lighting conditions, where RGB images alone provide insufficient information. RGB-Thermal fusion leverages the complementary strengths of visible and infrared imagery to improve scene understanding; however, effectively integrating these heterogeneous modalities at varying levels of feature abstraction remains an open problem. In this paper, we propose a multi-modal fusion architecture built upon dual ConvNeXt V2 backbones that employs stage-wise, modality-adaptive fusion strategies. For early-stage features, we introduce a Frequency-Based Fusion Module that decomposes infrared features into low- and high-frequency components via Gaussian filtering, applies dual-branch spatial attention to selectively emphasize thermal patterns and fine-grained boundaries, and integrates them with RGB features through a confidence-gated residual mechanism. For late-stage features, we design a semantic fusion module with cross-modal attention and multi-scale depthwise convolutions to capture semantic correspondences across modalities. The fused features are decoded via a PANet-style bidirectional decoder with deep supervision. Experiments on MFNet and PST900 demonstrate that our lightest variant achieves 61.73\% and 86.24\% mIoU, respectively, with only 35.43M parameters, outperforming recent methods while using substantially fewer parameters and lower computational cost. Code is available at this https URL
>
---
#### [new 089] Unified Panoramic Geometry Estimation via Multi-View Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D重建任务，旨在解决从全景图像中恢复360度场景的问题。工作包括引入PaGeR框架，利用预训练模型实现多视角和全景图像的几何估计。**

- **链接: [https://arxiv.org/pdf/2605.26368](https://arxiv.org/pdf/2605.26368)**

> **作者:** Vukasin Bozic; Isidora Slavkovic; Dominik Narnhofer; Nando Metzger; Denis Rozumny; Konrad Schindler; Nikolai Kalischek
>
> **摘要:** Geometry estimation from perspective images has greatly advanced, maturing to the point where off-the-shelf foundation models are able to reconstruct 3D scene structure not only from multi-view imagery, but even from a single view. A natural extension is 3D reconstruction from panoramas, with the exciting prospect of recovering a full 360-degree scene from a single panoramic image. In this work, we introduce PaGeR (Panoramic Geometry Reconstruction), a framework to lift powerful 3D foundation models designed for perspective imagery to the panorama domain. Our strategy is to start from a pre-trained transformer for 3D reconstruction and turn it into a unified high-performance model that predicts scale-invariant depth, metric depth, surface normals, and sky masks from both perspective and omnidirectional images, in a single forward pass. By keeping architectural changes to a minimum and mixing perspective and panoramic images during training, PaGeR retains the rich 3D prior of the underlying foundation model while learning to also estimate geometrically consistent 360-degree scenes from single panoramas. We extensively test our method in both indoor and outdoor environments and find that it delivers state-of-the-art performance and excellent zero-shot performance across a wide range of scenes.
>
---
#### [new 090] Re-M3Dr: Rebalanced MultiModal Mean Deviation Regression
- **分类: cs.CV**

- **简介: 该论文属于医学图像回归任务，旨在提升视神经病变评估的准确性。针对多模态融合效果不佳的问题，提出Re-M3Dr方法，通过优化数据分布和联合训练稳定性，显著降低预测误差。**

- **链接: [https://arxiv.org/pdf/2605.26513](https://arxiv.org/pdf/2605.26513)**

> **作者:** Haojie Yin; Chengcheng Feng; Tianyi Liu; Tianqi Zhang; Kaizhu Huang
>
> **摘要:** Mean Deviation (MD) is a critical metric for assessing visual field loss in ophthalmology. While previous work has focused solely on predicting MD from Optical Coherence Tomography (OCT), it is intuitive to assume that combining OCT with another imaging of fundus photography (FP) could improve performance, as two ophthalmic medical imaging provide complementary information. This is particularly expected when sophisticated multi-objective optimization is applied, as documented in common multimodal classification. Surprisingly, our investigations reveal that multimodal fusion in this medical imaging scenario performs worse than unimodal model. Through detailed analysis, we identify the root cause as a coupled imbalance between data distribution and modality learning conflict. This imbalance distorts the optimization landscape, leading to unstable training. To address this challenge, we propose the method of Rebalanced MultiModal Mean Deviation Regression (Re-M3Dr), a novel multimodal regression framework. We enhance unimodal representation through adaptive margin based supervised contrastive learning. Then, our framework stabilizes the joint optimization with the sharpness-aware gradient modulation. Experimental results on both public and private clinical datasets show average 29\% reduction in MSE compared to SOTA multimodal learning methods, demonstrating the superiority of Re-M3Dr. The code is available in the supplementary materials.
>
---
#### [new 091] Geometry-Aware Representation Denoising for Robust Multi-view 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于多视角3D重建任务，旨在提升退化条件下重建的鲁棒性。提出GARD框架，在特征空间进行去噪，恢复精确几何与高质量图像。**

- **链接: [https://arxiv.org/pdf/2605.26230](https://arxiv.org/pdf/2605.26230)**

> **作者:** Jin Hyeon Kim; Jaeeun Lee; Claire Kim; Kyoungjin Oh; Paul Hyunbin Cho; Jaewon Min; Yeji Choi; Jihye Park; Hyunhee Park; Minkyu Park; Seungryong Kim
>
> **摘要:** Multi-view 3D reconstruction has achieved remarkable progress with the advent of feed-forward 3D reconstruction models. However, these models are typically trained and evaluated under ideal, degradation-free imaging conditions, whereas real-world observations often contain degradations that differ significantly from such settings. Improving robustness for multi-view 3D reconstruction under degraded conditions therefore remains an important challenge. We present Geometry-Aware Representation Denoising (GARD), a novel framework that performs diffusion-based multi-view restoration directly in the feature space of a feed-forward 3D reconstruction model. This design exploits the geometry-aware feature representations of the 3D reconstructor to effectively recover accurate scene geometry. Furthermore, by employing an additional RGB image decoder, the refined representations can also be used to restore high-quality RGB images, thereby enabling the simultaneous recovery of 3D scene geometry and high-quality imagery. Comprehensive experiments on the Depth Anything 3 (DA3) benchmark demonstrate the effectiveness of the proposed GARD framework.
>
---
#### [new 092] On the Robustness of Machine Unlearning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于机器遗忘任务，旨在解决VLM中不良信息的记忆问题。通过系统分析和攻击测试，揭示现有方法的脆弱性，并提出更可靠的多模态遗忘策略。**

- **链接: [https://arxiv.org/pdf/2605.26992](https://arxiv.org/pdf/2605.26992)**

> **作者:** Yujie Lin; Kaidi Jia; Jiayao Ma; Chengyi Yang; Jinsong Su
>
> **摘要:** Vision-language models (VLMs) may memorize undesirable information from training data, motivating growing interest in machine unlearning. In this work, we present the first systematic survey and robustness analysis of VLM unlearning. We provide a comprehensive taxonomy and review of existing VLM unlearning methods, together with unified evaluations under multiple prompt settings. We then propose three attack paradigms to examine whether forgotten multimodal knowledge can be reactivated through contextual prompting or downstream retraining. Extensive experiments show that many existing methods remain vulnerable under these attacks, indicating that current approaches often hide rather than fully remove target knowledge. Our study provides new insights into the robustness and limitations of current VLM unlearning methods and highlights the need for more reliable multimodal unlearning strategies. Code is available at this https URL.
>
---
#### [new 093] Leveraging Visual Signals for Robust Token-Level Uncertainty in Vision-Language Generation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言生成任务，解决LVLM中token级不确定性估计问题。通过引入视觉信息提升不确定性评估，提出VIG-TUQ方法，无需训练即可优化预测可靠性。**

- **链接: [https://arxiv.org/pdf/2605.27136](https://arxiv.org/pdf/2605.27136)**

> **作者:** Joseph Hoche; David Brellmann; Gianni Franchi
>
> **摘要:** Uncertainty quantification (UQ) remains a critical challenge in Large Vision Language Models (LVLMs) for reliable predictions and real-world deployment. However, most existing methods are adapted from the LLM literature and primarily focus on the language modality, leaving the contribution of visual information to LVLM uncertainty largely underexplored. In this paper, we investigate how LVLMs process visual information and whether this process can be used to improve uncertainty estimation. By analyzing hidden representations after the integration of visual features during the generation process, we observe that high-confidence predictions rely more heavily on visual content than uncertain ones. Building on this insight, we propose Visual-Grounded Token UQ (VIG-TUQ), a training-free framework that explicitly incorporates visual grounding into uncertainty estimation by weighting token-level language uncertainty with visual grounding scores. We evaluate VIG-TUQ on multiple datasets and across diverse LVLM architectures, including early-fusion, late-fusion, and native-fusion models. Results indicate that our method often improves upon existing token-level uncertainty approaches. Code and data will be made available upon acceptance.
>
---
#### [new 094] Personalized Generative Models for Contextual Debiasing
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决视觉模型对罕见场景识别能力不足的问题。通过生成稀有情境图像增强训练数据，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.26353](https://arxiv.org/pdf/2605.26353)**

> **作者:** Xinran Liang; Esin Tureci; Prachi Sinha; Ye Zhu; Vikram V. Ramaswamy; Olga Russakovsky
>
> **备注:** CVPR 2026 Workshop on Synthetic Data for Computer Vision and Generative Models for Computer Vision. Code available at this https URL
>
> **摘要:** Different visual patterns appear with different frequencies in the world: e.g., beach balls appear on sand more often than they do on a road. These statistics are reflected in vision datasets, and as a result trained models more easily recognize objects in common scenarios. However, recognizing a beach ball on a road may arguably be even more important than recognizing it on sand. We study how to mitigate this discrepancy. Since collecting uncommon images in the real world may be difficult, we explore whether generating images with less frequent contexts can serve as effective training augmentation. A key challenge is guiding generations to remain close to the original dataset distribution while creating diverse images with uncommon contexts. We introduce Decoupling Contextual Patterns with Generations (DecoupleGen), a method that personalizes text-to-image diffusion models to facilitate coherent synthesis of images with rare contexts while preserving original visual details. The generated images contain semantically meaningful content and remain visually aligned with the original datasets. We further apply verification constraints to ensure relevance of the augmented data. We evaluate our approach on object classification and recognition tasks on complex scene datasets. Our experiments demonstrate consistent improvements over previous approaches, and our analyses identify factors underlying these improvements.
>
---
#### [new 095] DV-SFT: Direct Vision Supervision for Fine-Grained Visual Understanding
- **分类: cs.CV**

- **简介: 该论文提出DV-SFT方法，解决多模态大语言模型中视觉理解粗粒度问题，通过直接视觉监督提升细粒度视觉理解与对齐效率。**

- **链接: [https://arxiv.org/pdf/2605.26656](https://arxiv.org/pdf/2605.26656)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng; Bing Wang; Zhixing Tan
>
> **备注:** Under Review
>
> **摘要:** Multimodal large language models are typically trained end-to-end to predict ground-truth answers, yet supervision signals are applied exclusively to text tokens. Visual tokens, the core carriers of visual information, are optimized only implicitly as part of the context, leading to coarse-grained visual understanding. Prior works attempt to supervise visual inputs but inevitably rely on auxiliary components such as additional decoders or forward passes, because visual tokens lack readily interpretable labels. This limits their practical applicability. In this work, we propose \textbf{D}irect \textbf{V}ision \textbf{S}upervised \textbf{F}ine-\textbf{T}uning (DV-SFT), which constructs explicit, token-level supervision for visual tokens and trains them through the same next-token prediction objective used for text. Specifically, we exploit the direct vision--text correspondence in OCR-related scenarios and automatically label each visual token with the word in its corresponding image patch. DV-SFT treats the MLLM as a black box, requiring no architectural modifications or additional forward passes. Extensive experiments demonstrate the superiority of direct vision supervision. DV-SFT consistently outperforms standard SFT across three in-domain and four out-of-domain benchmarks. Further analyses show that vision supervision effectively enhances fine-grained visual understanding and achieves higher multimodal alignment efficiency.
>
---
#### [new 096] Evi-Steer: Learning to Steer Biomedical Vision-Language Models through Efficient and Generalizable Evidential Tuning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型的适应任务，旨在解决生物医学图像中模型在小样本和领域漂移下的鲁棒性问题。提出Evi-Steer框架，实现高效且不确定性感知的参数微调。**

- **链接: [https://arxiv.org/pdf/2605.26292](https://arxiv.org/pdf/2605.26292)**

> **作者:** Taha Koleilat; Hassan Rivaz; Yiming Xiao
>
> **备注:** MICCAI 2026 Early Accept; Project Page: this https URL
>
> **摘要:** Parameter-efficient adaptation of vision-language foundation models is crucial for precise multimodal understanding of biomedical images, yet existing methods remain deterministic and often struggle under domain shift or ambiguous image-text alignment. This limitation is particularly critical in the clinic, where models should remain robust in low-data regimes and domain shifts. We present Evi-Steer, an evidential cross-modal low-dimensional steering framework for BiomedCLIP that enables uncertainty-aware parameter-efficient fine-tuning while updating only 0.11% of total model parameters. Our approach performs lightweight low-dimensional token updates in both vision and text encoders while simultaneously estimating epistemic uncertainty. These uncertainty estimates update gate residuals, allowing the model to adapt conservatively when evidence is weak. Furthermore, we introduce cross-modal confidence fusion based on Dempster-Shafer theory, enabling visual adaptation to be conditioned on textual confidence and suppressing conflicting or uncertain cross-modal updates. We conduct a comprehensive evaluation on 15 biomedical imaging datasets spanning 8 organs and 8 imaging modalities under few-shot learning and domain generalization settings. Evi-Steer consistently outperforms state-of-the-art methods under few-shot learning and domain shift settings, demonstrating a practical and robust pathway for deploying vision-language models in real-world clinical settings. Code is available at this https URL.
>
---
#### [new 097] Erased but Exploitable: Black-box Embedding-Aware Prompting Against Unlearned Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全攻击任务，旨在解决黑盒环境下生成对抗提示的问题。提出BEAP方法，通过迭代生成有效提示，提升攻击成功率并规避检测。**

- **链接: [https://arxiv.org/pdf/2605.26332](https://arxiv.org/pdf/2605.26332)**

> **作者:** Arian Komaei Koma; Seyed Amir Kasaei; AmirMahdi Sadeghzadeh; Mohammad Hossein Rohban
>
> **摘要:** Machine unlearning aims to remove specific concepts from pretrained text-to-image diffusion models, yet several white- and black-box attacks have been introduced to make the model generate such unlearned concepts. These attacks, nevertheless, do not assume a realistic threat model, i.e. they either assume access to the model weights, or result in gibberish adversarial prompts that could be easily detected even through naive rule-based safeguarding. We aim to address this gap in this paper. We introduce BEAP, a black-box, embedding-aware adversarial prompting attack that leverages a large language model (LLM) to iteratively generate effective adversarial prompts and exploit such hidden vulnerabilities. BEAP performs an embedding-aware search in text space, combining multiple reward signals: unlearned concept presence, text-image alignment, and image quality, to refine generated prompts. Unlike previous attack methods, BEAP keeps its prompts undetectable to safety filters while producing high-quality images. Extensive experiments show that BEAP improves the Attack Success Rate (ASR) by more than 60% over prior methods, while requiring only an average of fifteen prompts per successful attack. Warning: This paper contains model outputs that may be offensive or upsetting in nature.
>
---
#### [new 098] COVD: Continual Open-Vocabulary Object Detection with Novel Concept Injection
- **分类: cs.CV**

- **简介: 该论文提出COVD任务，解决持续学习中新概念注入的问题。通过冻结视觉编码器，仅更新文本分支参数，实现高效知识保留与新概念学习。**

- **链接: [https://arxiv.org/pdf/2605.27116](https://arxiv.org/pdf/2605.27116)**

> **作者:** Yupeng Zhang; Ruize Han; Yuzhong Feng; Zixin Ren; Yuntong Tian; Liang Wan
>
> **摘要:** Open-vocabulary object detection (OVD) has made significant progress, enabling detectors to generalize from seen to unseen categories. However, real-world category spaces continually evolve, and existing OVD models still struggle with newly emerging concepts, while repeated full retraining is prohibitively expensive. To this end, we introduce a new task setting, termed Continual OVD with Novel Concept Injection (COVD), where models sequentially learn incoming novel concept groups while preserving prior concepts and original open-vocabulary knowledge, along with a new benchmark, Novel-114. Our key observation is that pretrained visual encoders often already perceive and represent many novel concepts, and the main bottleneck lies in the lack of stable semantic alignment between visual representations and textual concepts. Based on this, we propose NoIn-Det, an efficient continual injection framework without additional parameters. NoIn-Det freezes the visual encoder, preserves the text representation space using only texts of common concepts and previously injected concepts, and injects novel concepts by updating only a small subset of text-branch parameters beneficial to novel concept learning. Extensive experiments show that NoIn-Det effectively learns novel concepts, preserves old knowledge, and consistently outperforms existing continual learning methods for VLMs without introducing additional this http URL-114 and the code will be released.
>
---
#### [new 099] G3T Up! Gravity Aligned Coordinate Frames Simplify Pointmap Processing
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决点云处理中坐标框架选择不当的问题。提出G3T模型，在重力对齐框架下生成更准确的点图和相机姿态。**

- **链接: [https://arxiv.org/pdf/2605.27372](https://arxiv.org/pdf/2605.27372)**

> **作者:** Bharath Raj Nagoor Kani; Noah Snavely
>
> **备注:** Project Page: this https URL
>
> **摘要:** Modern feed-forward 3D reconstruction methods like VGGT predict pixel-aligned pointmaps in camera-centric coordinate frames. However, this choice of coordinate frame is not always optimal. We propose instead to predict pointmaps in upright, gravity-aligned frames that exploit strong structural cues present in many real-world scenes. Unlike camera-centric frames, gravity-aligned frames share a common vertical axis across viewpoints, reducing the rotational degrees of freedom needed to relate pointmaps to one another. To this end, we introduce the Gravity Grounded Geometry Transformer (G3T), fine-tuned from existing models on gravity-aligned 3D data. G3T produces highly accurate gravity-aware predictions, including upright pointmaps and camera-to-gravity poses. We further introduce G3T-Long, a submap-based incremental 3D reconstruction pipeline that leverages the reduced rotational degrees of freedom afforded by upright frames to achieve significantly improved reconstruction accuracy.
>
---
#### [new 100] Gaussian-Voxel Duet: A Dual-Scaffolding Hybrid Representation for Fast and Accurate Monocular Surface Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于单目表面重建任务，旨在解决几何精度与优化效率的平衡问题。提出混合高斯-体素表示，结合稀疏体素框架提升重建质量和速度。**

- **链接: [https://arxiv.org/pdf/2605.26616](https://arxiv.org/pdf/2605.26616)**

> **作者:** Zhenhua Du; Zhen Tan; Haoyu Zhang; Dewen Hu; Shuaifeng Zhi; Peidong Liu
>
> **备注:** 27 pages, 14 figures
>
> **摘要:** While 3D Gaussian Splatting has achieved remarkable success in photorealistic novel view synthesis, its pursuit of fast and high-fidelity 3D reconstruction has long been constrained by a trade-off between geometric accuracy and optimization efficiency. Methods specialized in image rendering converge quickly at the cost of imperfect geometry caused by superfluous primitives overfitting training views, while methods integrating neural signed-distance field (SDF) for better geometry incur prohibitive training costs. In this paper, we attempt to strike a better trade-off by tethering scaffold-anchored Gaussians to a jointly optimized sparse voxel scaffold. This hybrid Gaussian-Voxel representation explicitly confines anchored Gaussians to a narrow band around surfaces defined by voxelized SDFs, which effectively improves representation efficiency and condenses floating Gaussians without sacrificing geometry quality. An implicit surface tethering loss further pulls individual Gaussian primitives closer to SDF-induced surfaces in a mutually regularized manner for improved reconstruction accuracy. Extensive experiments on diverse real-world indoor scenes from ScanNet++, ScanNetv2, and DeepBlending datasets demonstrate that our method achieves state-of-the-art surface reconstruction quality as well as superior novel view synthesis against leading baselines, while maintaining fast training convergence and real-time rendering. Code will be available at this https URL.
>
---
#### [new 101] Q-GeoMem: Question-Guided Geometric Memory for Video Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文提出Q-GeoMem，解决视频空间推理中冗余几何信息干扰的问题。通过问题引导的几何记忆框架，提升长时序推理效果。**

- **链接: [https://arxiv.org/pdf/2605.27318](https://arxiv.org/pdf/2605.27318)**

> **作者:** Xianqiang Gao; Qizhi Chen; Delin Qu; Haoming Song; Zhigang Wang; Bin Zhao; Dong Wang; Xuelong Li
>
> **摘要:** Video spatial reasoning requires accumulating viewpoint-dependent evidence over time while retaining information useful to the question being asked. Existing spatial video-language models improve geometric perception and long-range context modeling, but often treat memory as a generic temporal cache, which can introduce redundant or irrelevant geometry and weaken long-horizon reasoning. We propose \textbf{\ours}, a question-guided geometric memory framework for video spatial reasoning. \ours injects camera-conditioned geometry into visual tokens and maintains two complementary memories: a Fine-Grained Context Bank for recent dense features and camera states, and a Semantic-Geometric Evidence Bank for compact long-range evidence. Each candidate frame is scored by the product of Q-Former-based question relevance and novelty with respect to the retained bank; this score is stored and reused during reading, while a capacity-based replacement rule keeps the bank compact. During reasoning, both memories are read before update and adaptively fused with the current frame representation. Experiments on VSI-Bench and VSTI-Bench show that \ours achieves state-of-the-art performance among evaluated spatial reasoning models, validating the effectiveness of question-guided geometric memory. Ablations further verify the contribution of the proposed evidence scoring mechanism.
>
---
#### [new 102] Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini
- **分类: cs.CV**

- **简介: 该论文提出Gemini Embedding 2，一个统一处理视频、音频、图像和文本的多模态嵌入模型，解决跨模态检索与泛化问题。通过多任务训练和对比学习，提升多种任务性能。**

- **链接: [https://arxiv.org/pdf/2605.27295](https://arxiv.org/pdf/2605.27295)**

> **作者:** Madhuri Shanbhogue; Zhe Li; Shanfeng Zhang; Gustavo Hernández Ábrego; Shih-Cheng Huang; Aashi Jain; Daniel Salz; Sonam Goenka; Chaitra Hegde; Ji Ma; Feiyang Chen; Jiaxing Wu; Tanmaya Dabral; Babak Samari; Kevin Poulet; Daniel Cer; Kaifeng Chen; Paul Suganathan; Hui Hui; Jovan Andonov; Philippe Schlattner; Jay Han; Iftekhar Naim; Wing Lowe; Vladimir Pchelin; Albert Yang; Yi-Ting Chen; Zhongli Ding; Grace Zhang; Georg Heigold; Yichang Chen; Antoine Reveillon; Brendan Mccloskey; Wenlei Zhou; Dahun Kim; Rui Meng; Emma Wang; Jack Zheng; Halley Fede; Zhen Yang; Keegan Mosley; Brian Potetz; Sahil Dua; Henrique Schechter Vera; Shen Gao; Hesen Zhang; Andreas Hess; Hengxuan Ying; Alberto Montes; Karan Gill; Min Choi; Sebastian Russo; Anja Hauth; Jinhyuk Lee; Michael Boratko; Megan Barnes; Vikram Rao; Claudiu Musat; Cyril Allauzen; Ehsan Variani; Shankar Kumar; Tom Bagby; Junyi Jiao; Yang Gu; Tengxin Li; Ayush Agrawal; Roberto Santana; Dev Nath; Stephen Karukas; Shuoxuan Han; Lucia Loher; Alice Twu; Nidhi Vyas; Siddharth Bhai; Frank Palma Gomez; Wangyuan Zhang; Chaoren Liu; Jizheng Yang; Steve Qiu; Shijie Zhang; Sujay Kulkarni; Sascha Rothe; Sean Nakamoto; Raphael Hoffmann; Zach Gleicher; Yunhsuan Sung; Qin Yin; Tom Duerig; Mojtaba Seyedhosseini
>
> **摘要:** We introduce Gemini Embedding 2, a native multimodal embedding model that allows embedding video, audio, image, and text modalities in a unified representation space. We leverage the multimodal capabilities of Gemini to produce embeddings for arbitrary combinations of interleaved inputs across all these modalities that generalize well across a wide variety of tasks. Applying large-scale contrastive learning in a multi-task multi-stage training setup, we achieve state-of-the-art performance on key embedding benchmarks including unimodal, cross-modal, and multimodal retrieval spanning a diverse set of tasks. We show that our embedding model demonstrates strong performance (with a score of 62.9 R@1 on MSCOCO, 68.8 NDCG@10 on Vatex, 69.9 on MTEB multilingual and 84.0 on MTEB Code) across a variety of tasks surpassing the performance of specialized models. These unified capabilities make Gemini Embedding 2 a promising candidate for downstream use cases such as RAG, recommendation and search. Furthermore, its robust zero-shot performance across distinct fields - from astronomy and bioscience to fine arts and the culinary arts - establishes it as a highly reliable, out-of-the-box representation even for specialized domains.
>
---
#### [new 103] How and What to Imagine? Visual Thinking in Unified Multimodal Models for Cross-View Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文研究跨视角空间推理任务，解决VLMs依赖语言而非视觉细节的问题。提出View Dropout方法，结合全景视觉思维，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.27310](https://arxiv.org/pdf/2605.27310)**

> **作者:** Qian Yang; Ankur Sikarwar; Huy Le; Le Zhang; Zhuan Shi; Perouz Taslakian; Aishwarya Agrawal
>
> **备注:** Preprint
>
> **摘要:** Cross-view spatial reasoning remains a weak spot for vision-language models (VLMs): they often reason in language and lose the fine-grained geometry needed for the task. Thinking with images aims to address this by generating an intermediate thinking image, but recent work shows that models often ignore the visual evidence in these traces. We therefore ask how to make visual thinking matter, and what kind of visual thinking works best. We study these questions in unified multimodal models (UMMs), which natively support interleaved image-text generation. For the first question, we propose View Dropout (VDrop), a training-time intervention that hides parts of one input view from the answer span while keeping them visible to the thinking-image tokens. This encourages the model to use the thinking image when answering, instead of relying only on the input views. Once the thinking image is used for answer prediction, we study which type of visual thinking is most effective. We frame this as a learnability-informativeness tradeoff and compare three thinking-image variants: top-down, panoramic, and point-matching renderings. Trained on synthetic scenes and evaluated on five real-world out-of-domain benchmarks, panoramic visual thinking with VDrop is the only configuration that is both informative and learnable, and it achieves the best out-of-domain generalization.
>
---
#### [new 104] Joint Instance Segmentation and Geometric Attribute Regression for Roof Structures in Aerial Imagery
- **分类: cs.CV**

- **简介: 该论文提出一种联合方法，同时预测屋顶实例分割和几何属性（高度、坡度、方位角），解决单张航拍图像中3D建筑模型重建问题。**

- **链接: [https://arxiv.org/pdf/2605.26370](https://arxiv.org/pdf/2605.26370)**

> **作者:** Luuk Versteeg; Rob G.J. Wijnhoven; Martin R. Oswald
>
> **摘要:** We present a method for jointly predicting instance-level roof segment masks together with three continuous geometric attributes -- building height, roof slope, and roof azimuth -- from a single aerial orthophoto. Our approach extends Mask R-CNN with a dedicated attribute regression branch and introduces two key innovations: a conditional azimuth loss that suppresses supervision for flat roof segments where azimuth labels are inherently noisy, and a log-normalized height representation that addresses the heavily skewed distribution of building heights. We train and evaluate on a large-scale dataset of Dutch aerial images paired with automatically derived ground truth from 3DBAG, a nationwide LiDAR-based 3D building dataset. Using a DINOv3 ConvNeXt-Base backbone, our method achieves a mean absolute error of approximately 4 degrees for roof slope, 7 degrees for azimuth, and 1 meter for building height, with an instance segmentation AP$_{50}$ of 0.566. The predicted per-segment masks and attributes are sufficient to reconstruct simplified 3D building models (LoD2) from a single overhead image, requiring expensive 3D reference data only for training.
>
---
#### [new 105] Comparative Study of Vision-Based Metric Measurement for Large-Scale Planar Scenes
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉度量任务，解决大尺度平面场景下的距离与面积测量问题。通过对比三种方法，分析其在不同条件下的性能与适用性。**

- **链接: [https://arxiv.org/pdf/2605.26475](https://arxiv.org/pdf/2605.26475)**

> **作者:** ZhiXin Sun
>
> **摘要:** Vision-based metric distance and area measurement remains challenging in large-scale outdoor environments due to long-range sensing, camera zoom, and unstable imaging conditions. This work studies planar metric measurement in a real-world reservoir monitoring scenario using PTZ cameras and compares three representative approaches: geometry-based monocular ranging, image stitching with birds-eye-view transformation, and stereo-based ranging using two jointly calibrated monocular cameras. For monocular ranging, planar localization models are derived from camera geometry and the effect of camera pitch angle is analyzed. Image stitching is investigated for large-area mapping, while a stereo-based scheme is developed for long-range measurement without dedicated stereo hardware. Experiments show clear trade-offs: monocular ranging achieves meter-level accuracy under sufficiently large pitch angles, stereo-based ranging achieves decimeter-level accuracy with reduced sensitivity to pitch variations, and image stitching is effective for small-scale scenes but degrades in stability and scalability as scene size increases.
>
---
#### [new 106] SoftCap: Soft-Budget Control for Diffusion Transformer Acceleration
- **分类: cs.CV**

- **简介: 该论文提出SoftCap，用于加速扩散Transformer（DiT）的推理。任务是图像生成加速，解决高计算成本问题，通过动态调整全步触发阈值实现高效推理。**

- **链接: [https://arxiv.org/pdf/2605.27075](https://arxiv.org/pdf/2605.27075)**

> **作者:** Yuhang Zhang; Junxiang Qiu; Huixia Ben; Zhenhua Tang; Shuo Wang; Yanbin Hao
>
> **摘要:** Diffusion Transformers (DiTs) achieve strong visual quality, but their iterative denoising process requires many costly Transformer evaluations. Training-free acceleration methods reduce this cost by caching, forecasting, or verifying intermediate features, yet the runtime decision of when to execute a Full step is often driven by fixed schedules or hand-tuned thresholds. We propose \textbf{SoftCap}, a training-free control layer for cache-based DiT inference. SoftCap couples a Trajectory Drift Observer, which estimates local cache risk from lightweight hidden-state statistics, with a Soft-Budget PI Controller, which adjusts the Full-triggering threshold from realized compute relative to a fixed reference profile. The budget is a soft ceiling: it shapes the threshold but does not require a run to spend a prescribed number of Full evaluations. On FLUX.1-dev, SoftCap improves over SpeCa at a comparable middle-compute operating point, raising ImageReward from 0.967 to 0.981 and reducing LPIPS-Full from 0.518 to 0.498 at nearly identical FLOPs, while target-sweep diagnostics show the intended soft-ceiling behavior as the budget is relaxed.
>
---
#### [new 107] CNNs, Transformers, Hybrid, and Vision Language Models for Skin Cancer Detection
- **分类: cs.CV**

- **简介: 该论文属于皮肤癌检测任务，旨在评估不同深度学习模型在PAD-UFES-20数据集上的表现，解决模型选择与性能优化问题。**

- **链接: [https://arxiv.org/pdf/2605.26294](https://arxiv.org/pdf/2605.26294)**

> **作者:** Durjoy Dey; Yuhong Yan; Hassan Hajjdiab
>
> **备注:** 13 pages, 3 figures, accepted at ICPRAI 2026, The Fifth International Conference on Pattern Recognition and Artificial Intelligence. To appear in Lecture Notes in Computer Science
>
> **摘要:** Skin cancer is a common and fast rising malignancy worldwide. Early detection is critical for improving outcomes. Deep learning models trained on dermoscopic and clinical images can support automated and fast triage. However, many studies evaluate only a limited set of architectures. Experimental setups also vary across studies. In this paper, we present a unified evaluation of twelve deep learning models for binary skin cancer detection on the PAD-UFES-20 dataset. The models span four families: convolutional neural networks (CNN), vision transformers (ViT), hybrid convolution transformer backbones, and vision language models (VLM). Performance is assessed using AUC, the maximum F1 score with its precision and recall, and sensitivity at 80% specificity, reflecting screening oriented requirements. Our results show that well tuned CNNs already provide strong baselines, but transformer based families consistently improve discrimination. Hybrid models (MaxViT Tiny, CoAtNet0) and a SigLIP based VLM achieve the best overall trade off between ranking performance and clinically relevant operating points, while CLIP based model offers high precision. The full codebase for all experiments is publicly released. Together, these findings offer practical guidance on which model families are most suitable for real world deployment in skin cancer screening and establish a reproducible reference point for future work on PAD-UFES-20.
>
---
#### [new 108] $R^3$: 3D Reconstruction via Relative Regression
- **分类: cs.CV**

- **简介: 该论文提出$R^3$方法，解决3D重建中的全局坐标系限制问题。通过相对回归，实现更高效的在线和离线重建。**

- **链接: [https://arxiv.org/pdf/2605.26519](https://arxiv.org/pdf/2605.26519)**

> **作者:** Congrong Xu; Huachen Gao; Xingyu Chen; Yuliang Xiu; Jun Gao; Anpei Chen
>
> **摘要:** Recent feed-forward geometry foundation models have demonstrated impressive generalization by recovering depth and poses in a single forward pass. However, these models are typically constrained by a global coordinate frame assumption. This dependency becomes a significant bottleneck for long-context and streaming reconstruction, as it forces the network to maintain an arbitrary temporal origin and handle translation magnitudes that grow unbounded over time. Our solution, which we call $R^3$, employs relative regression. We employ a lightweight MLP to predict confidence-weighted relative constraints. These confidences serve as a unified anchor: weighting losses during training and guiding pose aggregation during inference. $R^3$ supports both full-context offline reconstruction and causal, bounded-memory streaming. Our evaluation in both offline and streaming settings validates the effectiveness of our relative mechanism. Project page: this https URL
>
---
#### [new 109] HydraPrompt: An Adaptive and Asymmetric Framework of Vision-Language Models for Synthetic Image Detection
- **分类: cs.CV**

- **简介: 该论文属于合成图像检测任务，旨在解决传统方法无法适应不同伪造类型的问题。提出HydraPrompt框架，通过动态调整提示词提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.26421](https://arxiv.org/pdf/2605.26421)**

> **作者:** Senyuan Shi; Hao Tan; Zichang Tan; Shuhan Feng; Ajian Liu; Sergio Escalera; Jun Wan
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** The rapid evolution of generative models has precipitated a proliferation of fabricated content, posing significant challenges to existing Synthetic Image Detection (SID) methods. Capitalizing on advancements in vision-language models (e.g., CLIP), recent attempts have leveraged learnable textual prompts to identify synthetic images. However, they still leverage static prompt as a fixed boundary for real and fake images, failing to adapt to the varying types of forgery that emerge during inference. To overcome this issue, we propose **HydraPrompt**, an asymmetric prompting framework that dynamically adjusts the category centers by aligning with fine-grained image cues. Specifically, we propose an Asymmetric Prompt Adapter (**APA**): (1) for authentic category, we introduce a single set of prompts to capture the consistent representative patterns, which serves as a unified anchor for real content. While (2) for fake category, we construct sample-adaptive prompts that specialize in capturing diverse cues from different samples, enabling adaptive modeling of forgery image variations. To increase pronounced discriminability within different synthetic images, we further introduce a Conditional Supervised Contrastive (**CSC**) objective, which compacts the authentic representations while capturing fine-grained forgery clues. Extensive experiments on popular SID benchmarks demonstrate the state-of-the-art performance of our framework.
>
---
#### [new 110] Leveraging Text-to-Image Diffusion Models for Unsupervised Visual Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于无监督视觉目标跟踪任务，旨在解决无需标注数据的视频目标跟踪问题。通过利用文本到图像扩散模型的语义理解能力，提出Diff-Tracking方法实现目标跟踪。**

- **链接: [https://arxiv.org/pdf/2605.26933](https://arxiv.org/pdf/2605.26933)**

> **作者:** Zhengbo Zhang; Zhigang Tu; Junsong Yuan; De Wen Soh; Bo Du
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026
>
> **摘要:** Unsupervised visual object tracking is a challenging task that requires following arbitrary targets in videos without training on ground-truth annotations. Despite considerable progress, existing state-of-the-art unsupervised trackers often struggle in scenarios that demand fine-grained understanding of semantic and visual structural information within video frames. Text-to-image diffusion models are well known for their ability to generate images that accurately reflect the semantics and structures described in the input prompt, demonstrating a strong grasp of visual semantics and structures. Building on this capability, we approach the unsupervised tracking from a new perspective by exploiting the rich semantic knowledge encoded in pretrained text-to-image diffusion models. To adapt the diffusion models, which are originally developed for image generation, to the tracking task, we reinterpret the models as a bridge between text and image modalities. This connection is realized through the cross-attention mechanism: when both text and an image are input into the models, they highlight the regions of the image that are semantically aligned with the text in the cross-attention maps. We therefore learn a prompt that represents the tracking target and activates its corresponding region in the cross-attention map for each frame, which enables object tracking with the diffusion model. Specifically, our method Diff-Tracking is composed of two main components: an initial prompt learner and an online prompt updater. The initial prompt learner generates a prompt that captures the target object in the first frame, allowing the diffusion model to identify the target. The online prompt updater refines the prompt based on motion information, enabling consistent tracking across video frames. We evaluate our approach on six challenging tracking datasets demonstrate the effectiveness of our approach.
>
---
#### [new 111] Zero-Shot Object Re-Identification in Egocentric Kitchen Videos via Multi-Stage SAM3 Feature Fusion
- **分类: cs.CV**

- **简介: 该论文属于零样本目标重识别任务，解决egocentric厨房视频中因视角变化、遮挡等问题导致的匹配难题。通过多阶段SAM3特征融合提升性能。**

- **链接: [https://arxiv.org/pdf/2605.26383](https://arxiv.org/pdf/2605.26383)**

> **作者:** Dmytro Klepachevskyi; Alexander Wong; Sirisha Rambhatla; Yuhao Chen
>
> **摘要:** Object re-identification (ReID) in egocentric kitchen videos is challenging due to rapid viewpoint changes, frequent occlusions, cluttered scenes, and large intra-class appearance variations. Objects may leave and re-enter the field of view, and the large diversity of instances with limited annotations makes supervised ReID difficult to scale, motivating zero-shot approaches. We study zero-shot object ReID on the EPIC-Kitchens benchmark, where the goal is to match active food and kitchen-tool instances across frames using only pre-trained visual features. We first evaluate five state-of-the-art feature extractors, including Vision-Language Models (VLMs) - CLIP, DINOv2, DreamSim, I-JEPA, and SAM3 - and show that zero-shot methods fail, with the best baseline achieving only 45.3% mAP. We then propose an Enhanced SAM3 ReID Pipeline, a zero-shot multi-stage method built around SAM3 segmentation as the core component. Stage 1 uses SAM3 to suppress background clutter. Stage 2 fuses embeddings from SAM3, DINOv2, and CLIP into a single L2-normalized descriptor. Stage 3 augments cosine similarity with mask-shape IoU for geometric consistency, and Stage 4 applies k-reciprocal re-ranking. The full pipeline improves performance by 7.5% mAP to 52.8%.
>
---
#### [new 112] Detail Consistent Stage-Wise Distillation for Efficient 3D MRI Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D MRI分割任务，旨在解决压缩模型丢失细节的问题。提出DCD框架，在训练中通过小波域对齐保留结构细节，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2605.26382](https://arxiv.org/pdf/2605.26382)**

> **作者:** Mengchen Fan; Baocheng Geng; Xi Xiao; Tianyang Wang; Siyuan Mei; Pulin Che; Xiaoqian Jiang; Qizhen Lan
>
> **备注:** Accepted by MICCAI 2026. 11 pages, 3 figures
>
> **摘要:** Deploying high-performing 3D medical image segmenters (e.g., nnU-Net) is often limited by memory footprint and inference latency. Compression is therefore necessary, but compact 3D encoders tend to lose fine structural cues (small lesions and sharp boundaries) as downsampling repeats across multi-resolution stages. We propose Detail Consistent Distillation (DCD), a stage-wise distillation framework that preserves structural detail across scales by aligning teacher-student features in a wavelet-decomposed representation. At each encoder stage, DCD distills directional detail components in the wavelet domain while leaving the coarse approximation comparatively unconstrained, avoiding over-regularization of global semantics. DCD is used only during training and introduces no inference-time overhead. Experiments on the BraTS 2024 and ISLES 2022 benchmarks demonstrate that our approach achieves superior performance in MRI segmentation using 3D multi-modal data. Code and implementation details for DCD are publicly available at this https URL.
>
---
#### [new 113] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules
- **分类: cs.CV**

- **简介: 该论文属于图像逆问题任务，解决扩散模型中指导与随机性调度优化问题。提出TriPS方法，通过三元动态分析优化调度策略，提升数据保真度和感知真实感。**

- **链接: [https://arxiv.org/pdf/2605.26470](https://arxiv.org/pdf/2605.26470)**

> **作者:** Junseo Bang; Dong Ju Mun; Hoigi Seo; Seongmin Hong; Se Young Chun
>
> **备注:** ICML 2026
>
> **摘要:** Generative posterior sampling using diffusion models has emerged as a dominant paradigm for solving inverse problems in imaging, which usually consists of three main components: data consistency (DC) guidance, classifier-free guidance (CFG) and stochasticity. While prior arts have focused on how to develop each or all components, less attention has given to how to schedule them, leading to heuristically fixed or partially adjusted suboptimal schedules. In this work, we argue that the interactions among all three components in terms of scheduling are crucial for significantly improved performance in solving inverse problems in imaging. Our analysis shows that aggressive CFG early in sampling conflict with DC guidance, while stochasticity brings the trajectory back to higher-probability regions. Based on these findings, we propose Triadic Dynamics Aware Posterior Sampling (TriPS), which reformulates posterior sampling as a time-varying control problem and optimizes schedules following a triadic trend of decreasing DC and stochasticity scales alongside increasing CFG scale. TriPS achieves this through two strategies: template-based search over functional priors for reliable baseline schedules, and Group Relative Policy Optimization (GRPO)-based reinforcement learning for more flexible temporal curves. Experiments demonstrate TriPS outperforms state-of-the-art baselines in data fidelity and perceptual realism.
>
---
#### [new 114] PinPoint: Prompting with Informative Interior Points
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出PinPoint，解决 referring image segmentation 中提示模糊问题，通过选择信息丰富的内部点提升分割性能，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2605.26689](https://arxiv.org/pdf/2605.26689)**

> **作者:** Pouya Sadeghi; Shawn He; Pedro Pablo Guerrero Vela; C. Thomas; Alex Wong; Sirisha Rambhatla
>
> **摘要:** Modern referring image segmentation pipelines couple a vision-language model (VLM) for grounding with a promptable segmenter such as the Segment Anything Model (SAM) for mask generation. Prior training-free instances of this recipe consistently trail fine-tuned and reinforcement-learning (RL)-tuned specialists, and it has been unclear whether the gap comes from the VLM's grounding, SAM's capacity, or the prompt. We show that the gap is dominated by prompt ambiguity: a VLM-proposed bounding box (bbox) leaves SAM to guess which pixels inside the bbox belong to the object the expression denotes. Interior points are the natural disambiguator, but where they fall matters; prior work relies on naively sampled points that land on boundaries, distractors, and background clutter, and can even hurt performance compared to the bbox alone. Supervised and RL-tuned methods close this gap by training a VLM to predict better points; we show that this training is unnecessary. At a matched budget of five interior points, replacing naive sampling with stable, informative point selection improves cumulative Intersection-over-Union (cIoU) by 12-18 points across RefCOCO/+/g, with every model fixed. We turn this observation into PinPoint, a deterministic, training-free point selector that fuses four visual cues into a consensus map, selects compact, spatially diverse points away from boundaries, and uses the frozen VLM to label each point. Without any task-specific training, PinPoint matches supervised and RL-tuned specialists on the same stack while issuing only two VLM calls per query.
>
---
#### [new 115] PlayClass: Automated Play Behaviour Classification in Poultry
- **分类: cs.CV**

- **简介: 该论文属于动物行为分类任务，旨在解决禽类积极福利行为（如玩耍）的自动化识别问题。通过构建PlayClass系统，结合目标检测与特征提取技术进行行为分类。**

- **链接: [https://arxiv.org/pdf/2605.27304](https://arxiv.org/pdf/2605.27304)**

> **作者:** Prince Ravi Leow; Neil Scheidwasser; Rebecca Oscarsson; Per Jensen; Samir Bhatt; David Alejandro Duchêne
>
> **备注:** Accepted at CV4Animals Workshop @ CVPR 2026
>
> **摘要:** Automated monitoring of animal welfare has largely targeted negative indicators, leaving positive welfare behaviours such as play underexplored. To address this gap, we present PlayClass, a pipeline for play-behaviour classification in poultry from top-down pen video. The pipeline leverages long-duration tracking with SAM 3 via YOLO-guided chunk boundaries to minimise identity errors in point-based prompting, and frozen embeddings from image and video foundation models for play action classification. Although handcrafted motion features from tracked masks alone achieved competitive accuracy, V-JEPA 2.1 consistently outperformed all other backbones across model scales, reaching 77.0 macro-averaged F$_1$ when combined with handcrafted features. Despite this result, the dataset remains challenging due to play sub-types sharing similar kinematic profiles with non-play and inter-bird occlusion. Overall, our work provides encouraging evidence towards automated frameworks for play behaviour classification in poultry.
>
---
#### [new 116] ReCA: Multi-Shot Long Video Extrapolation via Recursive Context Allocation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MSVE任务，解决长视频生成中结构与连贯性问题。通过ReCA框架，实现多镜头视频的递归上下文分配，提升生成质量与一致性。**

- **链接: [https://arxiv.org/pdf/2605.26525](https://arxiv.org/pdf/2605.26525)**

> **作者:** Akide Liu; Jinbo Xing; Chaojie Mao; Ye Li; Zeyu Zhang; Yefei He; Weijie Wang; Zihan Wang; Yu Liu; Gholamreza Haffari; Bohan Zhuang
>
> **备注:** Project Page: this https URL , Code: this https URL
>
> **摘要:** Minute-scale cinematic video generation is a central challenge for generative video models. Existing paradigms address only fragments of this challenge: single-shot extrapolation preserves an anchor but lacks cinematic structure, while multi-shot storytelling imposes structure yet remains free to invent its visual states rather than continue an observed one. We define Multi-Shot Video Extrapolation (MSVE), a task that extends an observed frame or clip into a sequence of cinematically structured shots while preserving anchor state and advancing narrative intent. This setting operates under the finite per-call generation budget of short-video models. We identify three coupled bottlenecks: (1) global planners over-specify unsupported details from full screenplays; (2) shot-level prompts dilute task-relevant state when carrying the complete story; and (3) temporal chaining turns generated frames into a lossy memory in which identity, scene, object, and action state decay. MSVE reveals that long-video failure is not merely a limitation of context length, but a failure of context allocation. We propose Recursive Context Allocation (ReCA), an inference-time framework that allocates context hierarchically across planning and generation. ReCA recursively decomposes MSVE into context-bounded subproblems, invokes frozen generators at leaf nodes, and propagates structured state updates across time. To evaluate this setting, we further propose MSVE-Bench and NB-Q, a source-grounded protocol with prompts purpose-built for 3 to 5 minute long-video generation, a regime not addressed by existing short-clip benchmarks. Compared to previous methods, ReCA improves average normalized score by 8 to 16 percent over the strongest competing controller and improves multi-shot consistency metrics by 28 to 43 percent. View the project page at this https URL.
>
---
#### [new 117] Adaptation-Free Heterogeneous Collaborative Perception with Unseen Agent Configurations
- **分类: cs.CV**

- **简介: 该论文属于协同感知任务，解决未知协作者配置下的合作感知问题。提出ALF框架，通过转换消息生成兼容特征，实现零适应协作。**

- **链接: [https://arxiv.org/pdf/2605.26642](https://arxiv.org/pdf/2605.26642)**

> **作者:** Hyunchul Bae; Heejin Ahn
>
> **备注:** 9 pages main paper, 23 pages including references and appendix, 7 figures
>
> **摘要:** Collaborative perception improves 3D object detection by enabling agents to share complementary observations, but most existing methods assume fixed or known collaborator encoder configurations, limiting deployment in practice. In this work, we consider an open-world setting in which auxiliary agents with unseen configurations may appear after deployment, such as different LiDAR beam counts or encoder architectures. To address this challenge, we propose ALF, a collaborative perception framework that enables zero-adaptation collaboration with unseen agent configurations by lifting lightweight box-level messages into ego-compatible auxiliary features. ALF converts auxiliary box-level messages into pseudo-BEV maps and synthesizes ego-compatible latent features by combining object-centric cues with scene context from the ego feature. On V2X-Real, under a zero-shot evaluation across 64 case studies, ALF outperforms the strongest prior baseline by 35.91% in relative mAP@0.7 while requiring only 120 bytes per agent per frame (approximately 9.6 Kbps bandwidth at 10 Hz).
>
---
#### [new 118] Clinically-Grounded Counterfactual Reasoning for Medical Video Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学视频诊断任务，解决现有方法依赖外观而非病理、缺乏临床先验的问题。提出MedVCR框架，通过反事实推理提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2605.26483](https://arxiv.org/pdf/2605.26483)**

> **作者:** Jianzhe Gao; Churan Wang; Weiyi Zhang; Jianghua Li; Li-An Li; Wenguan Wang; Yixin Zhu; Yizhou Wang
>
> **摘要:** Medical video diagnosis involves inferring clinical decisions from dynamic tissue responses throughout examination processes. Existing methods rely on an end-to-end learning paradigm that i) focuses on appearance rather than pathology, ii) lacks clinical priors, and iii) reasons solely from observations without counterfactual comparison. This work introduces MedVCR, a counterfactual reasoning framework that mimics clinical diagnostic thinking. MedVCR comprises three components: a Counterfactual Generator that synthesizes tissue evolution under specified pathological states via a diffusion-based manner; a Counterfactual Representation Learning module that encodes diagnostic knowledge through clinical rules (i.e., temporal consistency, pathological separability, and counterfactual alignment); and a Dual Diagnostic Prediction strategy that integrates video-level assessment with frame-level counterfactual analysis. MedVCR is evaluated under both fully supervised (e.g., colposcopy) and weakly supervised (e.g., colonoscopy) video diagnosis settings, yielding 2.6%-10.2% performance gains compared with leading baselines. Comprehensive ablation studies further validate the effectiveness of each component. The code will be released.
>
---
#### [new 119] Learning Reference-Guided Exposure Correction with Hybrid Illumination Characteristics
- **分类: cs.CV**

- **简介: 该论文提出HICNet，用于参考引导的曝光校正任务，解决光照不均导致的图像质量下降问题，通过融合多尺度调制与光度通道重平衡实现精准曝光匹配。**

- **链接: [https://arxiv.org/pdf/2605.26729](https://arxiv.org/pdf/2605.26729)**

> **作者:** Hao Ren; Zetong Bi; Zhaoliang Wan; Hui Cheng
>
> **备注:** ICASSP2026
>
> **摘要:** We present HICNet, a reference-guided exposure correction framework. A lightweight, content-agnostic encoder distills each image into a compact illumination embedding capturing regional brightness, edge contrast, and higher-order luminance moments. The embedding difference between a source and its reference drives a multi-scale modulation network that combines FiLM-based global adjustment with Photometric Channel Rebalancing for fine-grained, illumination-aware spectral gating, producing exposure-matched outputs while faithfully preserving scene details. A cross-batch contrastive loss orders the illumination manifold, bolstering robustness to diverse lighting conditions. Trained without ground truth or intrinsic decomposition, HICNet attains better accuracy on public benchmarks and generalizes well to entirely unseen scenes.
>
---
#### [new 120] Joint 2D-3D Segmentation and Association in Street-level Imaging
- **分类: cs.CV**

- **简介: 该论文属于街景图像的2D-3D联合分割与关联任务，旨在提升城市地图构建中的对象识别与跟踪精度。通过融合视觉语义与几何推理，实现跨视角稳定匹配。**

- **链接: [https://arxiv.org/pdf/2605.26725](https://arxiv.org/pdf/2605.26725)**

> **作者:** Amir Melnikov; Masayuki Tanaka; Yusuke Monno; Masatoshi Okutomi
>
> **备注:** 15 pages, 6 image figures, 1 in-body table, 1 in-body algorithm, 2 indexes with tables
>
> **摘要:** Accurate interpretation of street-level imagery is essential for large-scale urban mapping and the creation of Spatial Digital Twin (SDT) environments. This work presents a unified framework for joint 2D-3D segmentation and association that integrates visual semantics with multi-view geometric reasoning. Unlike conventional approaches that rely heavily on sequential frames for temporal tracking, our method leverages zero-shot detection and segmentation together with structure-from-motion reconstruction to establish stable cross-view correspondences. A 3D-driven association mechanism replaces traditional 2D multi-object tracking, using geometric consistency to guide identity preservation across wide-baseline viewpoints and varying imaging conditions. By combining 2D texture cues with global 3D context, the proposed pipeline is well-suited for scalable street-level processing and can be used for a variety of object types. Experiments demonstrate substantially improved coverage of ground-truth sequences and more robust identity retention compared to state-of-the-art 2D-only tracking methods, achieving a 22% performance gain in challenging urban scenarios.
>
---
#### [new 121] I2PRef: Image-Driven Point Completion with Iterative Refinement
- **分类: cs.CV**

- **简介: 该论文属于点云补全任务，旨在从单张RGB图像中重建完整点云。提出I2P模块直接生成点云，结合P2P模块迭代优化，提升补全精度。**

- **链接: [https://arxiv.org/pdf/2605.26914](https://arxiv.org/pdf/2605.26914)**

> **作者:** Azhar Hussian; Marina Ritthaler; André Kaup; Vasileios Belagiannis
>
> **摘要:** We present an image-conditioned point cloud completion approach that treats images as the primary geometric source rather than a secondary guide. To this end, we introduce an Image-to-Point (I2P) module that can reconstruct complete point clouds directly from a single RGB image, with no need for 3D inputs. Additionally, we introduce a transformer-based Point-to-Point (P2P) refinement module that uses self- and cross-attention between point tokens and image features to iteratively refine the coarse I2P output. The I2P module enables the image encoder to learn rich geometric representations, while the P2P module progressively recovers fine-grained details. Unlike existing multimodal methods that rely on auxiliary losses or fusion modules, our explicit I2P task provides a strong, geometry-aware prior based on images alone. Extensive experiments on ShapeNet-ViPC demonstrate state-of-the-art completion performance with a 12.3% relative Chamfer Distance improvement over prior methods. Code is available at: this https URL
>
---
#### [new 122] OmniGF: A Dual-Branch Vision-Language Framework for Unified Gaze Following
- **分类: cs.CV**

- **简介: 该论文提出OmniGF，解决多人群体注视追踪任务。针对传统模型空间定位不足和处理效率低的问题，融合视觉语言模型，实现精准空间估计与语义推理。**

- **链接: [https://arxiv.org/pdf/2605.26399](https://arxiv.org/pdf/2605.26399)**

> **作者:** Qiaomu Miao; Haoyu Wu; Jingyi Xu; Minh Hoai; Dimitris Samaras
>
> **摘要:** Understanding human gaze behavior is essential for complex scene comprehension and human-computer interaction. Traditional gaze following models are typically restricted to pure spatial localization, lacking the high-level capacity to reason about semantic targets or complex social contexts. Furthermore, these models often process individuals sequentially, requiring redundant computations over the same scene image for multi-person inference. While recent Vision-Language Models (VLMs) offer the exceptional semantic reasoning needed to address gaze-related semantic tasks, their reliance on discrete text generation inherently limits precision in continuous spatial tasks like gaze localization. To bridge this gap, we propose OmniGF, a unified vision-language framework that adapts foundational VLMs for highly scalable multi-person gaze reasoning. The model adopts a dual-branch decoding strategy: a structured language branch generates discrete reasoning states, while a continuous spatial branch directly taps into the VLM's dense hidden states. Supervising these extracted representations with high-resolution gaze target heatmaps effectively overcomes the spatial bottleneck of text-only coordinate generation. Furthermore, to explicitly ground the model in multi-person scenes, we augment the input with head embeddings encoded from cropped head images, providing fine-grained appearance and orientation cues for all individuals simultaneously. By modeling all individuals and leveraging the strong semantic capability of VLMs, OmniGF seamlessly integrates precise spatial gaze target estimation, semantic gaze prediction, and complex social gaze reasoning. Extensive experiments demonstrate that our framework establishes new state-of-the-art performance across multiple standard benchmarks. Code is available at this https URL.
>
---
#### [new 123] OmniRetriever: Any-to-Any Audio-Video-Text Retrieval via Fusion-as-Teacher Distillation
- **分类: cs.CV**

- **简介: 该论文提出OmniRetriever，解决跨模态检索问题，通过融合教师蒸馏提升音频-视频-文本联合表示效果。**

- **链接: [https://arxiv.org/pdf/2605.26641](https://arxiv.org/pdf/2605.26641)**

> **作者:** Yunze Liu; Chi-Hao Wu; Enmin Zhou; Junxiao Shen
>
> **备注:** this https URL
>
> **摘要:** Unified multimodal embedding spaces have become the standard interface for cross-modal retrieval and multimodal RAG, and recent audio-video-text (AVT) encoders extend this setting to three modalities. Such encoders can produce a joint (T,V,A) embedding whenever all three modalities are available, but standard pairwise InfoNCE objectives leave this signal unused during training. We close this gap with fusion-as-teacher distillation, which treats a stop-gradient copy of the fused embedding as a teacher signal for the single-modal embeddings, paired with a Tuple-InfoNCE term that supervises the fused embedding directly. We instantiate this objective as OmniRetriever-7B. Across six zero-shot retrieval benchmarks, OmniRetriever-7B surpasses the closed-source Gemini Embedding 2 by 13.3-18.0 R@1 on Clotho and SoundDescs, and reaches the contemporary zero-shot specialist band of open video-text encoders on MSR-VTT and MSVD. To stress-test joint representations, we further release OmniRetriever-Bench, a 12-direction AVT retrieval benchmark totaling 3782 triples; on it OmniRetriever-7B attains AVG-all 34.84, improving over Gemini Embedding 2 by 1.72 and over the best prior open-source AVT method by 8.03.
>
---
#### [new 124] Recursive Flow Matching
- **分类: cs.LG; cs.AI; cs.CV; math.NA**

- **简介: 该论文提出RecFM，解决物理系统模拟中的速度与精度平衡问题，通过递归流匹配提高时空动态预测的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2605.26535](https://arxiv.org/pdf/2605.26535)**

> **作者:** Jiahe Huang; Sihan Xu; Sharvaree Vadgama; Rose Yu
>
> **备注:** Project page: this https URL
>
> **摘要:** Generative models have emerged as a powerful paradigm for solving physics systems and modeling complex spatiotemporal dynamics. However, achieving high physical accuracy without incurring high computational cost remains a fundamental challenge, as existing approaches face a critical speed-fidelity trade-off. In this work, we introduce Recursive Flow Matching (RecFM), a generative framework for forecasting complex spatiotemporal dynamics. RecFM enforces self-consistency to align trajectories across discretization scales, reducing discretization errors and improving performance across metrics for physics-based tasks. To our knowledge, this is the first method to achieve high-fidelity one- and few-step (2-4 step) dynamic generation for scientific systems with performance comparable to state-of-the-art multi-step solvers. Across challenging scientific benchmarks, RecFM achieves up to a 20$\times$ speedup over leading diffusion-based emulators while improving predictive accuracy. Furthermore, RecFM reduces mean squared error by over 15% compared to vanilla flow matching, offering a scalable and efficient solution for real-time scientific emulation.
>
---
#### [new 125] VISTA: An End-to-End Benchmark for Visual Spec-to-Web-App Coding Agents
- **分类: cs.SE; cs.AI; cs.CV**

- **简介: 该论文提出VISTA基准，用于评估基于大模型的代码生成代理在视觉导向的网页应用开发中的能力，解决真实UI开发中生成功能与视觉一致应用的问题。**

- **链接: [https://arxiv.org/pdf/2605.26144](https://arxiv.org/pdf/2605.26144)**

> **作者:** JunJia Guo; Yuhang Yao; Jiawei; Zhou; Jingdi Chen
>
> **摘要:** We present VISTA (VIsual Spec-To-App Benchmark), a benchmark for evaluating the end-to-end web-app generation capabilities of LLM-based agents. Unlike prior code generation benchmarks that focus on algorithmic tasks, VISTA targets realistic UI-centric development, where agents must produce functional, visually coherent applications from underspecified inputs. We define five prompt-information conditions that vary along two axes, visual/structural fidelity and stack constraint: (1) text only with free stack choice, (2) text with reference screenshots under three specified stacks, (3) text with reference screenshots under free stack choice, (4) text with screenshots and pruned Figma structure under a single specified stack, and (5) text with screenshots and pruned Figma structure under free stack choice. To enable robust evaluation, each page in the benchmark is manually annotated with interactive UI components and around three visual anchor points, addressing the well-known limitations of script-based testing tools such as Playwright in open-ended code generation settings. Evaluation combines DOM-grounded reference matching, behavior-specific browser tests, and CLIP-based visual similarity, jointly measuring structural alignment, behavioral completeness, and overall visual fidelity. We use VISTA to assess four agent systems drawn from two model families and two harnesses, finding that visual fidelity and functional correctness are partially decoupled across both input conditions and agents, and that agent editing style varies sharply but is largely orthogonal to task quality. VISTA establishes a rigorous and reproducible foundation for advancing agent-based software engineering research.
>
---
#### [new 126] Chartographer: Counterfactual Chart Generation for Evaluating Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言模型评估任务，旨在解决模型依赖捷径而非真正视觉推理的问题。通过生成反事实图表，验证模型的泛化能力与视觉推理水平。**

- **链接: [https://arxiv.org/pdf/2605.27311](https://arxiv.org/pdf/2605.27311)**

> **作者:** Yifan Jiang; Dae Yon Hwang; Jesse C. Cresswell; Freda Shi
>
> **摘要:** Chart question-answering (QA) benchmarks aim to pose questions that require visual reasoning to correctly answer, but models can often reach solutions through shortcuts or prior familiarity with a chart based on their own background knowledge. To strictly evaluate visual reasoning, we propose counterfactual charts where the chart-question task remains fixed, but underlying chart and the corresponding answer are varied. We introduce Chartographer, a framework to reverse engineer charts into executable code, validate reconstruction fidelity, generate seed-controlled counterfactual variants, and derive new answers from executable QA logic. We apply this framework to existing chart QA datasets and evaluate proprietary and open-source vision-language models (VLMs), measuring variation sensitivity and generalizability. Counterfactual charts reveal failures hidden by single-chart performance: VLMs often fail to generalize after answering the original chart correctly. We find failures are most prevalent when updated charts require novel visual reasoning pathways.
>
---
#### [new 127] Measuring Prediction Uncertainty in Neural Cellular Automata
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，解决NCA模型预测不确定性评估问题。通过分析NCA的迭代稳定性，提出"resilience"度量，提升模型可信度。**

- **链接: [https://arxiv.org/pdf/2605.26726](https://arxiv.org/pdf/2605.26726)**

> **作者:** Ario Sadafi; Michael Deutges; Nassir Navab; Carsten Marr
>
> **备注:** Accepted for publication at the 29th International Conference on Medical Image Computing and Computer Assisted Intervention - MICCAI 2026
>
> **摘要:** Neural cellular automata (NCA) provide a lightweight alternative to encoder-decoder segmentation networks. However, it can be difficult to decide when a prediction should be trusted. Here, we study uncertainty estimation for NCA-based medical image segmentation without modifying the underlying architecture or retraining the model. Our approach is motivated by viewing the NCA as a dynamical system where convergent attractors correspond to confident predictions. Concretely, we propose resilience, a simple measure that leverages the intrinsic iterative structure of NCAs by probing the stability of the final prediction under small perturbations of the automaton state. Predictions that return to the same solution are deemed confident, while those that change substantially are flagged as uncertain. We evaluate uncertainty by its ability to predict segmentation quality using selective prediction metrics ($\Delta$Dice@90 and AURC) and ranking metrics (AUROC and AUPRC). Across multiple medical segmentation benchmarks, resilience identifies failure cases more reliably than baselines, improving trust and safety in NCA-based models.
>
---
#### [new 128] Beyond Pairwise Preferences: Listwise Reward-Aware Alignment for Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决传统偏好优化方法仅依赖成对比较的局限性。提出Diffusion LAIR方法，利用列表奖励信息进行更有效的模型对齐。**

- **链接: [https://arxiv.org/pdf/2605.26491](https://arxiv.org/pdf/2605.26491)**

> **作者:** Austin Wang; Jiaqi Han; Stefano Ermon; Yisong Yue
>
> **摘要:** Preference optimization has emerged as an efficient alternative to online reinforcement learning from human feedback (RLHF) for aligning text-to-image diffusion models. However, existing methods largely reduce supervision to binary pairwise comparisons. This pairwise reduction is limiting when training data naturally contains multiple candidate images for the same prompt, and when continuous reward scores can provide richer information than a single winner-loser label. To address these limitations, we propose Diffusion LAIR, a reward-aware listwise preference optimization method for diffusion models. For each prompt, LAIR converts reward scores across a group of candidate images into centered advantage weights, then optimizes an advantage-weighted regression objective on the implicit reward, defined as the denoising-loss improvement of the current model over a fixed reference model, with a quadratic penalty that regularizes the magnitude of the implicit reward. The resulting objective uses all candidates simultaneously rather than selecting pairs, and remains conservative by explicitly controlling the magnitude of the implicit reward. The LAIR objective admits a bounded closed-form optimum in implicit-reward space, clarifying how the regularization strength controls the magnitude of the preference update. Experiments show that Diffusion LAIR outperforms strong preference optimization baselines on SD1.5 and SDXL across text-to-image generation, compositional generation, and image editing benchmarks.
>
---
#### [new 129] SteelDS: A High-Resolution Video Dataset of E40 Steel Scrap for Object Detection and Instance Segmentation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SteelDS数据集，用于钢渣中铜杂质的检测与分割。解决工业自动分拣中的材料识别问题，包含高分辨率标注视频数据。**

- **链接: [https://arxiv.org/pdf/2605.26682](https://arxiv.org/pdf/2605.26682)**

> **作者:** Melanie Neubauer; Christian Rauch; Gerald Koinig; Alexia Tischberger-Aldrian; Roland Pomberger; Elmar Rueckert
>
> **摘要:** This dataset provides high-resolution, annotated video sequences of shredded E40-grade steel and copper scrap on a conveyor belt. Captured in a controlled laboratory environment, the data reflects the industrial post-magnetic sorting stage, where manual intervention is typically required to remove copper contaminants. The dataset comprises 24,297 labeled frames across five subsets, featuring 396 steel and 101 copper objects categorized by size. It supports the development of machine learning models for material classification, object detection, and instance segmentation. Variations in object spacing and density are included to simulate realistic industrial sorting conditions. Ground truth annotations include pixel-wise segmentation masks and material classes. This dataset serves as a benchmark for evaluating automated sorting algorithms aiming to identify copper impurities within complex, heterogeneous steel scrap streams.
>
---
#### [new 130] Do Modern Post-Hoc Watermarking Methods Beat Broken-Arrows?
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于数字水印任务，旨在比较现代与经典后处理水印方法的鲁棒性和安全性。研究发现，在实际场景中，经典方法在保持鲁棒性的同时更具安全性。**

- **链接: [https://arxiv.org/pdf/2605.27135](https://arxiv.org/pdf/2605.27135)**

> **作者:** Enoal Gesny; Eva Giboulot
>
> **摘要:** With the rapid proliferation of generative models, such as diffusion models, digital watermarking has emerged as a crucial solution for identifying AI-generated images. Modern post-hoc watermarking schemes use neural networks to achieve an extremely low false-alarm rate while remaining robust to common image transformations. However, there is a lack of comparison between these modern methods and classic ones, particularly in real-world scenarios where robustness and security take precedence over achieving an extremely low false-alarm probability. In this paper, we propose a fair comparison of robustness and security between modern and classic post-hoc watermarking across various types of classic augmentations and recent sophisticated attacks. Our experiments show that, in a realistic scenario, classic watermarking outperforms modern techniques in terms of security while maintaining robustness.
>
---
#### [new 131] The Kalman Evolve: Closing the Gap in Kalman Filtering via Interpretable Algorithm Discovery
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于状态估计任务，解决Kalman滤波在非线性传感下的性能下降问题。通过优化滤波结构和参数，提出Kalman Evolve框架，提升跟踪精度。**

- **链接: [https://arxiv.org/pdf/2605.26830](https://arxiv.org/pdf/2605.26830)**

> **作者:** Vasileios Saketos; Ming Xiao
>
> **摘要:** State estimation is a fundamental problem in control and signal processing, for which the Kalman Filter provides an optimal solution under linear dynamics, Gaussian noise, and known noise covariances. However, these assumptions often fail in realistic sensing settings such as Doppler radar and LiDAR. In these cases, the optimal estimator is inherently nonlinear, which leads to systematic performance degradation. This creates a performance gap that cannot be eliminated by tuning the noise covariance parameters (i.e., the process and measurement noise in the Kalman Filter) alone. To address this limitation, we propose Kalman Evolve, a framework for discovering improved filtering algorithms by jointly optimizing both noise parameters and the update structure. Our approach leverages large language models (LLMs) as a structured prior over program space, enabling the generation of interpretable, non-affine modifications to the classical Kalman filter while preserving its recursive form. We provide analytical results establishing the suboptimality of affine estimators under common nonlinear sensing models, motivating the need for structure-aware updates. Across a range of synthetic and real-world tracking benchmarks, including Doppler radar, LiDAR-based localization, and pedestrian tracking, the discovered algorithms consistently improve over strong baselines such as the Optimized Kalman Filter, achieving up to 12\% reduction in RMSE. These results suggest that optimizing the structure of the Kalman filter, rather than only its parameters, provides a practical and interpretable way to improve state estimation.
>
---
#### [new 132] Not All Tokens Matter Equally: Dynamic In-context Vector Distillation with Decisive-Token Supervision for Long-form Medical Report Generation
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于长文本生成任务，针对医学报告生成中token重要性不均的问题，提出DIVE框架，通过关键token监督和动态调整机制提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.27194](https://arxiv.org/pdf/2605.27194)**

> **作者:** Ning Wu; Rui Liu; Xinkun Lin; Weixing Chen; Jinxi Xiang; Tao Wei; Lina Yao; Mingjie Li
>
> **备注:** Preprint. 20 pages, 6 figures
>
> **摘要:** Distilling demonstration effects into hidden-space interventions offers a lightweight alternative to full finetuning. However, existing multimodal variants are mostly evaluated on short-form tasks, where outputs end after a few tokens. Extending these methods to long-form generation exposes a fundamental yet underexamined limitation: token-level distillation implicitly treats all output tokens as equally informative, but long-form outputs are dominated by high-frequency template and grammatical tokens, while the tokens that actually determine output quality are sparsely distributed. In medical report generation (MRG), two such decisive tokens stand out: pathology-related tokens that determine diagnostic content, and the end-of-sequence (EOS) event that determines termination. Both receive insufficient supervision under uniform cross-entropy, and autoregressive decoding further compounds the problem by drifting away from teacher-forced trajectories. We propose DIVE, a frozen-backbone distillation framework that addresses long-form report generation through two complementary mechanisms matched to these failures. Decisive-token supervision restores supervision balance by upweighting the cross-entropy contribution of pathology-related tokens and the EOS event, ensuring that content fidelity and termination are learned during training rather than imposed at decoding time. State-conditioned dynamic steering replaces fixed open-loop residuals with hidden-state-dependent adapters, allowing the injected signal to adapt as decoding drifts. Experiments on MIMIC-CXR and CheXpert Plus with two medical VLM backbones show that DIVE consistently ranks among the strongest methods across lexical and clinical-proxy metrics. Our method achieves the best BLEU-4, ROUGE-L, and RadGraph F1 in all dataset--backbone settings, while remaining competitive on coarse label-level CheXbert F1.
>
---
#### [new 133] Efficient On-policy Visual-RL via Stochastic Decoupled Policy Gradient
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; eess.SY**

- **简介: 该论文提出SDPG，一种轻量级视觉强化学习方法，解决高效训练视觉-运动控制策略的问题，通过随机扰动轨迹估计梯度，提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.26478](https://arxiv.org/pdf/2605.26478)**

> **作者:** Haoxiang You; Yilang Liu; Davis Zong; Qian Wang; Teeratham Vitchutripop; Qi Wang; Daniel Rakita; Ian Abraham
>
> **摘要:** We present the stochastic decoupled policy gradient (SDPG), a lightweight visual reinforcement learning (RL) method that trains diverse visuomotor control policies end-to-end within a few hours on a single NVIDIA RTX 4080 GPU. SDPG estimates policy gradients via random perturbations of trajectory rollouts, requiring orders of magnitude fewer batch-rendered environments and substantially reducing compute and memory overhead. On visual MuJoCo benchmarks, SDPG consistently outperforms baseline methods in training time, memory usage, and rewards. Finally, to support future research, we introduce a suite of realistic visual robotics benchmarks spanning dexterous manipulation, challenging locomotion, and demonstrate effective sim-to-real transfer on physical hardware.
>
---
#### [new 134] Design First, Code Later: Aesthetically Pleasing Template-Free Slides Generation
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于自动幻灯片生成任务，旨在解决传统方法依赖模板或直接生成代码导致设计能力受限的问题。提出DeepSlides框架，实现无模板的幻灯片设计与生成。**

- **链接: [https://arxiv.org/pdf/2605.26451](https://arxiv.org/pdf/2605.26451)**

> **作者:** Zhiyao Cui; Chenxu Wang; Shuyue Hu; Yiqun Zhang; Wenqi Shao; Qiaosheng Zhang; Zhen Wang
>
> **摘要:** Producing presentation slides automatically entails coordinating narrative structure with page-level graphic design under strict spatial constraints. For such structured multimodal tasks, a well-organized design process is essential to ensure the final quality of slides. Existing approaches rely on fixed templates or directly emit executable code, thereby both limiting the creative layout-design capabilities of LLMs and bypassing the essential slide-page design step. To address these limitations, this paper (1) proposes a hierarchical slides generation workflow, DeepSlides, that systematically organizes slide design tasks without any predefined template or style, decoupling slide-page design from implementation; (2) introduces SlideDesign, a dataset tailored specifically for slides generation tasks; and (3) presents a multi-agent reinforcement learning training paradigm and trains a couple of models, SlideQwens, for slide design and implementation. Experimental results demonstrate that our proposed framework outperforms baseline methods on evaluated metrics and achieves superior performance in human preference evaluations. The dataset and code are available at this https URL.
>
---
#### [new 135] Object Pose and Shape Estimation for Grasping: Does it Work?
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究物体位姿与形状估计在抓取中的应用，对比模块化方法与端到端方法的性能，探讨其在单视角RGB-D图像下的有效性。**

- **链接: [https://arxiv.org/pdf/2605.26944](https://arxiv.org/pdf/2605.26944)**

> **作者:** Pavan Karke; Kushal Shah; Gaurav Singh; Md Faizal Karim; K Madhava Krishna; Rajat Talak
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** The problem of object pose and shape estimation has seen key advancements lately. Encoder-decoder (e.g., SAM3D, LRM, CRISP) and diffusion-based models (e.g., InstantMesh, Zero123, SceneComplete) have shown category-agnostic shape encoding capacity and open-set generalizability. In this work, we ask the question: Are the object pose and shape estimation methods mature enough, such that when used with antipodal grasp sampling, can outperform the end-to-end grasp synthesis methods? We explore this question in detail by scoping our study to parallel jaw grippers, 7-DoF grasps, and single-view RGB(-D) image as input. We implement and compare a state-of-the-art, end-to-end grasp synthesis method and three modular methods, which first estimate the object pose and shape for all objects in the scene, and generate grasps using antipodal sampling. We observe that the modular methods outperform the end-to-end method in all our experiments. The modular methods are able to synthesize plenty of grasps, even for small objects, where the end-to-end methods fail. The effectiveness of the modular methods is contingent on the accuracy of the pose and shape estimation, and suffers partial degradation in cluttered scenes - a limitation of the existing pose and shape estimation methods. We also analyze the failure modes and run-times for the three modular methods, which use two different ways of object pose and shape estimation: one based on an encoder-decoder model, while another a diffusion model. Finally, we demonstrate that the single-view object pose and shape estimation methods can be augmented with vision-language models to yield language-conditioned grasps from just single-view RGB-D image as input. We notice comparable performance to the state-of-the-art LERF-TOGO baseline.
>
---
#### [new 136] Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion
- **分类: cs.LG; cs.AI; cs.CV; cs.GR; eess.IV**

- **简介: 该论文属于视频扩散模型任务，解决KV缓存压缩导致的视频质量下降问题。通过修正量化带来的注意力偏差，提升压缩效率与视频质量。**

- **链接: [https://arxiv.org/pdf/2605.26266](https://arxiv.org/pdf/2605.26266)**

> **作者:** Tuna Tuncer; Felix Becker; Thomas Pfeil
>
> **备注:** Variants of this manuscript were accepted to the ICML 2026 workshops SCALE and F2S
>
> **摘要:** Chunk-wise autoregressive video diffusion models rely on a KV cache of previously generated chunks to avoid redundant computation, but this cache quickly becomes a memory bottleneck as videos grow longer. Methods that quantize the KV cache to low bitwidths reduce memory pressure but degrade video quality. We show that a key driver of this degradation is a systematic bias in attention weights: due to the convexity of the exponential in softmax attention, quantization noise inflates the contribution of cached keys, a phenomenon we call the Jensen bias. This effect causes quantized keys to steal attention mass from the unquantized current chunk. We derive a per-attention-score correction that removes this bias in expectation, computed on the fly from the quantization step sizes of the cached keys and the query norm. Using a second-order Taylor approximation, the additional computational overhead is negligible, and no additional memory is needed alongside the cache. Evaluated on MAGI-1, SkyReels-V2, and HY-WorldPlay at INT2 quantization, our correction recovers most of the quality lost to aggressive quantization, reaching near-BF16 video quality, and can outperform INT4 quantization while using 50% less memory.
>
---
#### [new 137] EdgeFlow: Edge-Map Augmented VLM-Based Flowchart Processing for Industrial Requirements Engineering
- **分类: cs.SE; cs.AI; cs.CV**

- **简介: 该论文属于工业需求工程中的流程图处理任务，旨在提升流程图到Mermaid的转换效果。针对VLM在拓扑细节上的不足，提出EdgeFlow方法，通过引入边缘图增强输入，无需训练数据即可提高转换精度。**

- **链接: [https://arxiv.org/pdf/2605.27332](https://arxiv.org/pdf/2605.27332)**

> **作者:** Zhifei Dou; Shabnam Hassani; Ou Wei
>
> **备注:** 10 pages
>
> **摘要:** Flowcharts are widely used in industrial requirements, but usually remain embedded as static images. Vision Language Models (VLMs) show promise in the conversion of these flowcharts into machine-readable models for RE activities, yet, when directly applied to flowchart conversion, they often fail on topology-critical visual details. To address this, we propose EdgeFlow that augments a VLM's original input with a deterministically extracted Canny edge map-acting as a structural prior-to improve flowchart-to-Mermaid conversion, without requiring annotated training data or domain-specific model fine-tuning. We evaluate EdgeFlow on IndusReqFlow, a dataset sourced from real-world requirements. Compared with off-the-shelf VLMs, EdgeFlow improves node-level F1 by 17.39 percentage points and edge-level F1 by 16.94 percentage points. At the path level, EdgeFlow improves path F1 by 11.06 percentage points, enabling better support for model-based testing. These results demonstrate that EdgeFlow provides a practical, training-free means to improve topology-preserving flowchart-to-Mermaid conversion for industrial RE. Cross-dataset evaluation results on a public synthetic benchmark show no significant improvement; this highlights the need for diverse benchmarks incorporating industrial data for the comprehensive evaluation of future VLM-based RE tools.
>
---
#### [new 138] AnySurf: Any Surface Generation with Directed Edge
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出AnySurf，解决开放与闭合表面生成问题，通过定向边增强的网格结构实现高质量3D表面生成。**

- **链接: [https://arxiv.org/pdf/2605.26149](https://arxiv.org/pdf/2605.26149)**

> **作者:** Wenda Shi; Chenyuan Pan; Dengming Zhang; Yiren Song; Biao Zhang; Xingxing Zou
>
> **摘要:** Open surface components prevail in real industrial 3D content and support rendering, physical simulation and geometric editing. Garments serve as a typical open surface type, with numerous existing generation methods leveraging sewing patterns to generate 2D panels and stitch them into 3D shapes. Such domain-specific designs lack scalability and cannot generalize to shoes and accessories. Common field-based 3D generators prioritize watertight meshes and tend to create flawed double-layer structures on open surfaces. Though Trellis2 adopts field-free representation, its open surface results still contain normal and topology errors. We present AnySurf, a unified framework generating open, closed and hybrid 3D surfaces with accurate face orientation. Built on directed-edge enhanced Flexible Dual Grid (FDG-D), our representation retains normal direction information via oriented grid edges. We also propose ROS-FT post-training and a lightweight DE-Adapter with merely 1% extra parameters, facilitating directed edge learning while preserving original generation performance. We further construct Outfit3D dataset containing industrial garments and closed accessories. Our work transforms garment modeling into a universal 3D generation task. Experimental results demonstrate superior mesh quality and better practicality for downstream applications.
>
---
#### [new 139] Unsupervised Deep Image Prior for Sparse-View and Limited-Angle Electron Tomography
- **分类: eess.IV; cs.CV; physics.ins-det**

- **简介: 该论文属于电子断层成像任务，旨在解决有限角度和稀疏视角下重建质量下降的问题。提出使用无监督深度图像先验方法，实现高质量3D重构。**

- **链接: [https://arxiv.org/pdf/2605.27139](https://arxiv.org/pdf/2605.27139)**

> **作者:** Serge Brosset; Daniel del Pozo Bueno; Thomas David; Laure Guetaz; Philippe Ciuciu; Zineb Saghi
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** Electron tomography (ET) plays an important role in the three-dimensional (3D) characterization of nanomaterials. However, under limited-angle and sparse-view conditions, conventional algorithms produce degraded reconstructions, which compromise the quality and interpretability of resulting 3D data. In this paper, we present deep image prior (DIP), an unsupervised deep learning (DL) approach, for highly degraded tomography acquisitions and demonstrate, using simulated data, that its performance is comparable to that of supervised approaches requiring training datasets, even for tilt ranges as limited as 60° and tilt increments of 10°. We then apply it to experimental data and show that it enables reliable 3D quantification under both sparse-view and limited-angle conditions, highlighting its potential for a wide range of materials and acquisition modalities.
>
---
#### [new 140] Garment Particles: A 2D--3D Symmetric Garment Representation for Generation and Editing
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出Garment Particles，解决服装生成与编辑问题，通过5D点云表示联合编码2D和3D信息，支持从高阶输入生成及低阶编辑。**

- **链接: [https://arxiv.org/pdf/2605.26391](https://arxiv.org/pdf/2605.26391)**

> **作者:** Kiyohiro Nakayama; I-chao Shen; Ruofan Liu; Yiming Wang; Gordon Wetzstein; Takeo Igarashi
>
> **摘要:** Practical garment design spans two modes: intuitive creation from high-level intent, such as a reference image or text description, and complex low-level editing across 2D sewing patterns and 3D draped geometry, which requires professional training to navigate their complex interdependencies. Yet existing frameworks address only part of this challenge, offering either garment generation from casual inputs or direct editing on sewing patterns. To support both ends of the spectrum, we propose Garment Particles, a 5D point-cloud representation that jointly encodes 2D sewing patterns and 3D geometry. This representation enables Garment Particles Flow (GPF), a rectified flow framework that supports intuitive generation from high-level inputs (text, images, sketches) and various editing operations on 2D sewing patterns and 3D geometries via diffusion posterior sampling. Finally, we introduce Particles-to-Pattern Flow that converts generated garment particles into curved-based patterns for simulation. We validate our model's generation ability on multiple datasets, achieving state-of-the-art garment generation results against competitive baselines. Our model also enables many garment editing scenarios, including garment interpolation, sewing pattern editing, point-cloud- and silhouette-conditioned garment generation. Our project website is at this https URL .
>
---
#### [new 141] AssetGen: Deployable 3D Asset Generation at Interactive Speed
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文提出AssetGen，解决3D资产生成的实时部署问题。通过优化框架和算法，实现快速高质量3D模型生成，适用于交互式应用。**

- **链接: [https://arxiv.org/pdf/2605.26137](https://arxiv.org/pdf/2605.26137)**

> **作者:** Dilin Wang; Xiaoyu Xiang; Kihyuk Sohn; Tom Monnier; Yu-Ying Yeh; Thu Nguyen-Phuoc; Jiawen Zhang; Yuchen Fan; Antoine Toisoul; Hyunyoung Jung; Prithviraj Dhar; Michael Bunnell; Nikolaos Sarafianos; Chuhang Zou; Roman Shapovalov; Andrea Vedaldi; Rakesh Ranjan
>
> **摘要:** While 3D generation is progressing rapidly, recent work has often focused on obtaining high-resolution assets, leaving user experience and deployability as afterthoughts. We present AssetGen, a 3D generator that focuses instead on these two aspects. Given one reference image, in 30 seconds it produces a high-quality mesh with baked normals, a color texture, and a controlled polygon budget suitable for real-time rendering, including mobile use cases. The AssetGen Flash variant further reduces latency to 14 seconds for interactive and agentic creation loops. Our model generates the object geometry with a coarse-to-refine VecSet framework, which implements mesh simplification, cleaning, and normal baking on the GPU, and a fast parallel UV unwrapping. It then generates textures in a multi-view fashion, followed by backprojection and 3D inpainting. Model distillation, kernel optimization, and pipeline parallelization are co-designed to accelerate the system end-to-end. We introduce numerous automated and blind human evaluations and demonstrate competitive visual quality against leading commercial solutions in 30 seconds and preview-quality results in less than 15 seconds. The final result is a system that supports AI-assisted, deployable 3D content creation in interactive workflows.
>
---
## 更新

#### [replaced 001] Detached Skip-Links and $R$-Probe: Decoupling Feature Aggregation from Gradient Propagation for MLLM OCR
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.20020](https://arxiv.org/pdf/2603.20020)**

> **作者:** Ziye Yuan; Ruchang Yao; Chengxin Zheng; Yusheng Zhao; Daxiang Dong; Ming Zhang
>
> **备注:** Accepted by ICML 2026. Ziye Yuan and Ruchang Yao contributed equally to this work (co-first authors, listed in random order)
>
> **摘要:** Multimodal large language models (MLLMs) excel at high-level reasoning yet fail on OCR tasks where fine-grained visual details are compromised or misaligned. We identify an overlooked optimization issue in multi-layer feature fusion. Skip pathways introduce direct back-propagation paths from high-level semantic objectives to early visual layers. This mechanism overwrites low-level signals and destabilizes training. To mitigate this gradient interference, we propose Detached Skip-Links, a minimal modification that reuses shallow features in the forward pass while stopping gradients through the skip branch during joint training. This asymmetric design reduces gradient interference, improving stability and convergence without adding learnable parameters. To diagnose whether fine-grained information is preserved and usable by an LLM, we introduce $R$-Probe, which measures pixel-level reconstructability of projected visual tokens using a shallow decoder initialized from the first quarter of the LLM layers. Across multiple ViT backbones and multimodal benchmarks, and at scales up to 7M training samples, our approach consistently improves OCR-centric benchmarks and delivers clear gains on general multimodal tasks.
>
---
#### [replaced 002] TAGRPO: Boosting GRPO on Image-to-Video Generation with Direct Trajectory Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.05729](https://arxiv.org/pdf/2601.05729)**

> **作者:** Jin Wang; Jianxiang Lu; Guangzheng Xu; Comi Chen; Haoyu Yang; Linqing Wang; Peng Chen; Mingtao Chen; Zhichao Hu; Longhuang Wu; Shuai Shao; Qinglin Lu; Ping Luo
>
> **备注:** 18 pages, 12 figures
>
> **摘要:** Recent studies have demonstrated the efficacy of integrating Group Relative Policy Optimization (GRPO) into flow matching models, particularly for text-to-image and text-to-video generation. However, we find that directly applying these techniques to image-to-video (I2V) models often fails to yield consistent reward improvements. To address this limitation, we present TAGRPO, a robust post-training framework for I2V models inspired by contrastive learning. Our approach is grounded in the observation that rollout videos generated from identical initial noise provide superior guidance for optimization. Leveraging this insight, we propose a novel GRPO loss applied to intermediate latents, encouraging direct alignment with high-reward trajectories while maximizing distance from low-reward counterparts. Furthermore, we introduce a memory bank for rollout videos to enhance diversity and reduce computational overhead. Despite its simplicity, TAGRPO achieves significant improvements over DanceGRPO in I2V generation. The deliverables will be updated at this https URL .
>
---
#### [replaced 003] Doc-CoB: Enhancing Document Understanding with Visual Chain-of-Boxes Reasoning
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.18603](https://arxiv.org/pdf/2505.18603)**

> **作者:** Ye Mo; Kai Ye; Xianwei Mao; Zirui Shao; Gang Huang; Bo Zhang; Hangdi Xing; Kehan Chen; Huan Zhou; Zixu Yan; Jiajun Bu; Sheng Zhou
>
> **摘要:** Document understanding aims to perform question answering and information extraction over document images, where the visual content is highly information-dense and most queries rely on only a few relevant layout regions. However, existing methods either adopt a one-pass strategy that implicitly assumes all layouts are equally important, or focus excessively on small regions at the cost of losing critical layout information. To address these limitations, we introduce Doc-CoB (Chain-of-Boxes), a simple-yet-effective framework that integrates coarse-to-fine layout-aware visual reasoning into multimodal large language models. Instead of directly zooming into small regions, Doc-CoB progressively focuses on query-relevant layouts while preserving global document information. Specifically, it first selects key layout boxes and then focuses on them for further understanding with visual prompting. To support this paradigm, we introduce two reasoning tasks for box recognition and box reasoning, with an automatic pipeline that constructs 249k training samples with intermediate visual supervision. Extensive experiments on seven benchmarks with four popular models show that Doc-CoB significantly improves performance, demonstrating its effectiveness and wide applicability.
>
---
#### [replaced 004] ScriptHOI: Learning Scripted State Transitions for Open-Vocabulary Human-Object Interaction Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.05057](https://arxiv.org/pdf/2605.05057)**

> **作者:** Minh Anh Nguyen; Quang Huy Tran; Bao Ngoc Le; SuiYang Guang; Tuan Kiet Pham; Linh Chi Vo
>
> **摘要:** Open-vocabulary human-object interaction (HOI) detection requires recognizing interaction phrases that may not appear as annotated categories during training. Recent vision-language HOI detectors improve semantic transfer by matching human-object features with text embeddings, but their predictions are often dominated by object affordance and phrase-level co-occurrence. As a result, a model may predict \textit{cut cake} from the presence of a knife and a cake without verifying whether the hand, tool, target, contact pattern, and object state jointly support the action. We propose \textbf{ScriptHOI}, a structured framework that represents each interaction phrase as a soft scripted state transition. Rather than treating a phrase as a single class token, ScriptHOI decomposes it into body-role, contact, geometry, affordance, motion, and object-state slots. A visual state tokenizer parses each detected human-object pair into corresponding state tokens, and a slot-wise matcher estimates both script coverage and script conflict. These two quantities calibrate HOI logits, expose missing visual evidence, and provide training constraints for incomplete annotations. To avoid suppressing valid but unannotated interactions, we further introduce interval partial-label learning, which constrains unannotated candidates with script-derived lower and upper probability bounds instead of assigning closed-world negatives. A counterfactual script contrast loss swaps individual script slots to discourage object-only shortcuts. Experiments on HICO-DET, V-COCO, and open-vocabulary HOI splits show that ScriptHOI improves rare and unseen interaction recognition while substantially reducing affordance-conflict false positives.
>
---
#### [replaced 005] LDP-Slicing: Local Differential Privacy for Images via Randomized Bit-Plane Slicing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03711](https://arxiv.org/pdf/2603.03711)**

> **作者:** Yuanming Cao; Chengqi Li; Wenbo He
>
> **摘要:** Local Differential Privacy (LDP) is the gold standard trust model for privacy-preserving machine learning by guaranteeing privacy at the data source. However, its application to image data has long been considered impractical due to the high dimensionality of pixel space. Canonical LDP mechanisms are designed for low-dimensional data, resulting in severe utility degradation when applied to high-dimensional pixel spaces. This paper demonstrates that this utility loss is not inherent to LDP, but from its application to an inappropriate data representation. We introduce LDP-Slicing, a lightweight, training-free framework that resolves this domain mismatch. Our key insight is to decompose pixel values into a sequence of binary bit-planes. This transformation allows us to apply the LDP mechanism directly to the bit-level representation. To further strengthen privacy and preserve utility, we integrate a perceptual obfuscation module that mitigates human-perceivable leakage and an optimization-based privacy budget allocation strategy. This pipeline satisfies rigorous pixel-level $\varepsilon$-LDP while producing images that retain high utility for downstream tasks. Extensive experiments on face recognition and image classification demonstrate that LDP-Slicing outperforms existing DP/LDP baselines under comparable privacy budgets, with negligible computational overhead.
>
---
#### [replaced 006] ISTASTrack: Bridging ANN and SNN via ISTA Adapter for RGB-Event Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09977](https://arxiv.org/pdf/2509.09977)**

> **作者:** Siying Liu; Zikai Wang; Hanle Zheng; Yifan Hu; Xilin Wang; Qingkai Yang; Jibin Wu; Hao Guo; Lei Deng
>
> **备注:** Accepted by IEEE Transactions on Image Processing, DOI: https://doi.org/10.1109/TIP.2026.3694138, 15 pages, 8 figures
>
> **摘要:** RGB-Event tracking has become a promising trend in visual object tracking to leverage the complementary strengths of both RGB images and dynamic spike events for improved performance. However, existing artificial neural networks (ANNs) struggle to fully exploit the sparse and asynchronous nature of event streams. Recent efforts toward hybrid architectures combining ANNs and spiking neural networks (SNNs) have emerged as a promising solution in RGB-Event perception, yet effectively fusing features across heterogeneous paradigms remains a challenge. In this work, we propose ISTASTrack, the first transformer-based \textbf{A}NN-\textbf{S}NN hybrid \textbf{Track}er equipped with \textbf{ISTA} adapters for RGB-Event tracking. The two-branch model employs a vision transformer to extract spatial context from RGB inputs and a spiking transformer to capture spatio-temporal dynamics from event streams. To bridge the modality and paradigm gap between ANN and SNN features, we systematically design a model-based ISTA adapter for bidirectional feature interaction between the two branches, derived from sparse representation theory by unfolding the iterative shrinkage thresholding algorithm. Additionally, we incorporate a temporal downsampling attention module within the adapter to align multi-step SNN features with single-step ANN features in the latent space, improving temporal fusion. Experimental results on RGB-Event tracking benchmarks, such as FE240hz, VisEvent, COESOT, and FELT, have demonstrated that ISTASTrack achieves state-of-the-art performance while maintaining high energy efficiency, highlighting the effectiveness and practicality of hybrid ANN-SNN designs for robust visual tracking. The code is publicly available at this https URL.
>
---
#### [replaced 007] When Brains Disagree: Biological Ambiguity Underlies the Challenge of Amyloid PET Synthesis from Structural MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.11867](https://arxiv.org/pdf/2605.11867)**

> **作者:** Louise E. G. Baron; Ross Callaghan; David M. Cash; Philip S. J. Weston; Hojjat Azadbakht; Hui Zhang
>
> **备注:** MICCAI 2026 accepted paper (no rebuttal)
>
> **摘要:** Structural MRI-to-amyloid PET synthesis has been proposed as a non-invasive alternative for amyloid assessment in Alzheimer's disease (AD). However, reported performance of identical models varies widely across studies, and increasingly complex architectures have not led to consistent gains. This inconsistency is thought to be caused by a fundamental biological ambiguity: MRI captures neurodegeneration, while PET measures amyloid pathology - two processes that are often temporally decoupled in AD. As a result, similar MRI patterns may correspond to different amyloid states, creating ambiguous one-to-many mappings. MRI-to-amyloid PET synthesis may therefore be intrinsically ill-posed; however, this idea has yet to be tested scientifically. The aim of this work is to test this hypothesis through two controlled experiments. We first control the training distribution by stratifying paired MRI-PET data by amyloid and neurodegeneration status. Using two standard synthesis models under a controlled design, we show that biologically unambiguous mappings are learnable in isolation, but performance collapses when data ambiguity is introduced. This demonstrates that ambiguity in the data distribution, rather than architectural capacity, constrains performance. Second, we show that introducing orthogonal biological information in the form of plasma biomarkers resolves this ambiguity. When multimodal inputs are incorporated, performance improves and stability is restored. Together, these findings suggest that limited and inconsistent performance in MRI-to-amyloid PET synthesis is explained by intrinsic biological ambiguity, and that stable, meaningful progress requires multimodal integration rather than architectural complexity.
>
---
#### [replaced 008] Bayesian In Vivo Tracking of Synapses using Joint Poisson Deconvolution and Diffeomorphic Registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.13455](https://arxiv.org/pdf/2605.13455)**

> **作者:** Shashwat Kumar; Dominic M. Padova; Binish Narang; Gabrielle I. Coste; Austin R. Graves; Richard L. Huganir; Adam S. Charles; Michael I. Miller; Anuj Srivastava
>
> **摘要:** Synapses are densely packed submicron structures that dynamically reorganize during learning and memory formation. Longitudinal \textit{in vivo} imaging of fluorescently tagged synaptic receptors offers a promising opportunity to study large-scale synaptic dynamics and how these processes are disrupted in neurological disease. However, in vivo imaging with 2-photon microscopy uses low laser power and therefore suffers from low signal-to-noise ratio (SNR) and high shot noise, nonlinear tissue motion between days, nonstationary fluctuations in synaptic fluorescence, and significant blur induced by the microscope point spread function (PSF). Together, these factors make it challenging to detect and track synapses, especially in regions with high synaptic density. This paper presents a novel template-based framework for modeling synapses as varying luminance point sources that move under a nonlinear tissue deformation. Taking a unified Bayesian approach, we apply this model to microscopy data by deriving a posterior that incorporates a diffeomorphic mapping for domain warping, a Gaussian point spread function for the imaging process, and a Poisson observation model for raw photon counts. The Bayesian solution simultaneously: (1) Constructs a probabilistic template of synapse locations, (2) denoises and deconvolves the image data, (3) infers fluorescence intensities, (4) performs diffeomorphic image registration to correct for tissue motion, and (5) provides confidence regions for these parameter estimates. We demonstrate the framework on both a 2D+t simulated dataset and a 3D+t longitudinal \textit{in vivo} microscopy dataset of fluorescent synapses imaged in a mouse over two weeks.
>
---
#### [replaced 009] DETR-ViP: Detection Transformer with Robust Discriminative Visual Prompts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.14684](https://arxiv.org/pdf/2604.14684)**

> **作者:** Bo Qian; Dahu Shi; Xing Wei
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Visual prompted object detection enables interactive and flexible definition of target categories, thereby facilitating open-vocabulary detection. Since visual prompts are derived directly from image features, they often outperform text prompts in recognizing rare categories. Nevertheless, research on visual prompted detection has been largely overlooked, and it is typically treated as a byproduct of training text prompted detectors, which hinders its development. To fully unlock the potential of visual-prompted detection, we investigate the reasons why its performance is suboptimal and reveal that the underlying issue lies in the absence of global discriminability in visual prompts. Motivated by these observations, we propose DETR-ViP, a robust object detection framework that yields class-distinguishable visual prompts. On top of basic image-text contrastive learning, DETR-ViP incorporates global prompt integration and visual-textual prompt relation distillation to learn more discriminative prompt representations. In addition, DETR-ViP employs a selective fusion strategy that ensures stable and robust detection. Extensive experiments on COCO, LVIS, ODinW, and Roboflow100 demonstrate that DETR-ViP achieves substantially higher performance in visual prompt detection compared to other state-of-the-art counterparts. A series of ablation studies and analyses further validate the effectiveness of the proposed improvements and shed light on the underlying reasons for the enhanced detection capability of visual prompts.
>
---
#### [replaced 010] V2V3D: View-to-View Denoised 3D Reconstruction for Light-Field Microscopy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.07853](https://arxiv.org/pdf/2504.07853)**

> **作者:** Jiayin Zhao; Zhenqi Fu; Tao Yu; Hui Qiao
>
> **备注:** CVPR 2025; New version: Fix NSFC ID
>
> **摘要:** Light field microscopy (LFM) has gained significant attention due to its ability to capture snapshot-based, large-scale 3D fluorescence images. However, existing LFM reconstruction algorithms are highly sensitive to sensor noise or require hard-to-get ground-truth annotated data for training. To address these challenges, this paper introduces V2V3D, an unsupervised view2view-based framework that establishes a new paradigm for joint optimization of image denoising and 3D reconstruction in a unified architecture. We assume that the LF images are derived from a consistent 3D signal, with the noise in each view being independent. This enables V2V3D to incorporate the principle of noise2noise for effective denoising. To enhance the recovery of high-frequency details, we propose a novel wave-optics-based feature alignment technique, which transforms the point spread function, used for forward propagation in wave optics, into convolution kernels specifically designed for feature alignment. Moreover, we introduce an LFM dataset containing LF images and their corresponding 3D intensity volumes. Extensive experiments demonstrate that our approach achieves high computational efficiency and outperforms the other state-of-the-art methods. These advancements position V2V3D as a promising solution for 3D imaging under challenging conditions.
>
---
#### [replaced 011] Xiaomi Auto World Model: A Joint World Model Integrating Reconstruction and Generation for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18137](https://arxiv.org/pdf/2605.18137)**

> **作者:** Lijun Zhou; Hongcheng Luo; Zhenxin Zhu; Cheng Chi; Mingfei Tu; Kaixin Xiong; Lei Gong; Zhanqian Wu; Zehan Zhang; Fangzhen Li; Hao Li; Yingying Shen; Jiale He; Haohui Zhu; Shan Zhao; Kai Wang; Zhiwei Zhan; Yuechuan Pu; Kaiyuan Tan; Ruiling Yang; Xianqi Wang; Tianyi Yan; Jiawei Zhou; Lei Zhang; Jingyang Zhao; Xi Zhou; Chitian Sun; Chenming Wu; Jiong Deng; Hongwei Xie; Ming Lu; Kun Ma; Long Chen; Guang Chen; Hangjun Ye; Bing Wang; Haiyang Sun
>
> **摘要:** This report presents a unified technical system addressing the two core capabilities of world models for autonomous driving: world representation and world generation. For world representation, we propose WorldRec, a feed-forward reconstruction architecture driven by sparse scene queries. WorldRec initializes structured queries in 3D space, leveraging them to aggregate cross-view, cross-temporal features, thereby naturally enforcing spatial consistency across frames and yielding compact yet high-fidelity 3D Gaussian scene representations. For world generation, we propose WorldGen, a two-stage training framework of bidirectional pretraining followed by causal fine-tuning through three progressive stages (Teacher Forcing, ODE distillation, and DMD), enabling high-quality online causal video generation in as few as 4 denoising steps. Building on both modules, we further introduce the JWM, which deeply integrates WorldRec and WorldGen to achieve synergistic gains in generation stability, cross-frame consistency, and visual fidelity, providing a solid foundation for closed-loop simulation, data synthesis, and end-to-end training in autonomous driving.
>
---
#### [replaced 012] VT-Bench: A Unified Benchmark for Visual-Tabular Multi-Modal Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.08146](https://arxiv.org/pdf/2605.08146)**

> **作者:** Zi-Yi Jia; Zi-Jian Cheng; Xin-Yue Zhang; Kun-Yang Yu; Zhi Zhou; Yu-Feng Li; Lan-Zhe Guo
>
> **摘要:** Multi-model learning has attracted great attention in visual-text tasks. However, visual-tabular data, which plays a pivotal role in high-stakes domains like healthcare and industry, remains underexplored. In this paper, we introduce \textit{VT-Bench}, the first unified benchmark for standardizing vision-tabular discriminative prediction and generative reasoning tasks. VT-Bench aggregates 14 datasets across 9 domains (medical-centric, while covering pets, media, and transportation) with over 756K samples. We evaluate 23 representative models, including unimodal experts, specialized visual-tabular models, general-purpose vision-language models (VLMs), and tool-augmented methods, highlighting substantial challenges of visual-tabular learning. We believe VT-Bench will stimulate the community to build more powerful multi-modal vision-tabular foundation models. Benchmark: this https URL
>
---
#### [replaced 013] Guiding Token-Sparse Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.01608](https://arxiv.org/pdf/2601.01608)**

> **作者:** Felix Krause; Stefan Andreas Baumann; Johannes Schusterbauer; Olga Grebenkova; Ming Gui; Vincent Tao Hu; Björn Ommer
>
> **摘要:** Diffusion models deliver high quality in image synthesis but remain expensive during training and inference. Recent works have leveraged the inherent redundancy in visual content to make training more affordable by training only on a subset of visual information. While these methods were successful in providing cheaper and more effective training, sparsely trained diffusion models struggle in inference. This is due to their lacking response to Classifier-free Guidance (CFG) leading to underwhelming performance during inference. To overcome this, we propose Sparse Guidance (SG). Instead of using conditional dropout as a signal to guide diffusion models, SG uses token-level sparsity. As a result, SG preserves the high-variance of the conditional prediction better, achieving good quality and high variance outputs. Leveraging token-level sparsity at inference, SG improves fidelity at lower compute, achieving 1.58 FID on the commonly used ImageNet-256 benchmark with 25% fewer FLOPs, and yields up to 58% FLOP savings at matched baseline quality. To demonstrate the effectiveness of Sparse Guidance, we train a 2.5B text-to-image diffusion model using training time sparsity and leverage SG during inference. SG achieves improvements in composition and human preference score while increasing throughput at the same time.
>
---
#### [replaced 014] RISE: Reliable Improvement in Self-Evolving Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.20914](https://arxiv.org/pdf/2605.20914)**

> **作者:** Chaoran Xu; Yingmao Miao; Pengfei Zhang; Hao Dou; Lei Sun; Xiangxiang Chu
>
> **摘要:** Vision-language models (VLMs) have achieved strong multimodal reasoning capabilities, but further improving them still relies heavily on large-scale human-constructed supervision for post-training. Such supervision is costly to obtain, especially for reasoning-intensive multimodal tasks where questions, answers, and feedback signals must be carefully designed. This motivates self-evolving learning, where a model improves itself through a dual-role closed loop: a questioner autonomously poses questions and a solver learns to solve them. However, we observe that current VLM self-evolving methods still face three major challenges: coarse-grained role alternation delays the interaction between question generation and solver adaptation; generated questions can progressively degrade in quality; and question types may collapse toward a narrow distribution. These issues limit the efficiency and reliability of self-evolution. Thus, we propose \textbf{RISE}, a reliable self-evolving framework for vision-language models. RISE is built on three complementary designs: fine-grained role alternation, which shortens the feedback loop between the questioner and the solver to improve efficiency; a quality supervisor, which improves question validity and pseudo-label reliability; and skill-aware dynamic balancing, which mitigates mode collapse and maintains broad skill coverage during evolution. Together, these components enable more reliable and effective self-evolution from unlabeled images. Experiments on two VLM backbones across seven benchmarks show that RISE consistently improves the base models, yielding broad and sustained gains. Our code is publicly available at this https URL.
>
---
#### [replaced 015] MuNet: A Mutualistic Network for Joint 3D Human Mesh Recovery and 3D Clothed Human Reconstruction from Single Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.25861](https://arxiv.org/pdf/2605.25861)**

> **作者:** Yunqi Gao; Leyuan Liu; Yuhan Li; Changxin Gao; Jingying Chen
>
> **摘要:** 3D human mesh recovery and 3D clothed human reconstruction are inherently related, yet they have long been studied in isolation, thereby overlooking the potential gains of joint optimization. To overcome this limitation, we propose to address these two tasks within a unified framework, which allows their mutual dependencies to be effectively exploited. Building on this idea, we propose MuNet, a mutualistic network for joint 3D human mesh recovery and 3D clothed human reconstruction from single images. First, we adopt 2-manifold graphs as a unified representation for all 3D models, enabling consistent modeling across 3D human mesh recovery and clothed human reconstruction. Second, we design an end-to-end graph convolutional network that progressively deforms an initial graph into a 3D human mesh and refines it into a detailed 3D clothed human model. Third, we introduce a mutualistic mechanism that allows reciprocal interaction between the two tasks {during training}, where 3D human mesh recovery provides guidance for 3D clothed human reconstruction, and reconstruction feedback refines the 3D human mesh recovery. We extensively evaluate MuNet on six benchmark datasets for 3D human mesh recovery and 3D clothed human reconstruction, including Human3.6M, 3DPW, MPI-INF-3DHP, THuman2.0, CAPE, and RenderPeople. Experimental results demonstrate that MuNet achieves state-of-the-art performance on both tasks across all datasets. The code of MuNet is released for research purposes at this https URL.
>
---
#### [replaced 016] Spectral Principal Paths: A Spectral Perspective on Linear Representation Formation in LLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08543](https://arxiv.org/pdf/2506.08543)**

> **作者:** Bowei Tian; Xuntao Lyu; Meng Liu; Hongyi Wang; Ang Li
>
> **备注:** arXiv admin note: text overlap with arXiv:2503.22720
>
> **摘要:** High-level representations have become a central focus in enhancing AI transparency and control, shifting attention from individual neurons or circuits to structured semantic directions that align with human-interpretable concepts. While the Linear Representation Hypothesis (LRH) suggests that such directions emerge in representations, it remains unclear how these representations originate and why they become increasingly stable across layers. To solve this issue, we introduce the Input-Space Linearity Hypothesis, positing that concept-aligned directions originate in the input space and are steadily maintained with increasing depth. We then propose the Spectral Principal Path (SPP) framework, which formalizes how deep networks progressively distill linear representations along the spectral principal directions. We provide rigorous stability guarantees for the SPP based on the Wedin $\sin\Theta$ perturbation theorem, identifying testable conditions, including spectral gap and context incoherence, that jointly ensure layer-wise directional preservation. By bridging theoretical analysis with empirical evidence, this work identifies a spectral view of how linear representations arise in LLMs, and suggests potential implications for concept-level controllable, robust, and coherent approaches to fairness and transparency in modern AI systems.
>
---
#### [replaced 017] ControlLight: Towards Controllable, Consistent, and Generalizable Low-Light Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.25569](https://arxiv.org/pdf/2605.25569)**

> **作者:** Yufeng Yang; Jianzhuang Liu; Jisheng Chu; Yuqi Peng; Xianfang Zeng; Jiancheng Huang; Shifeng Chen
>
> **备注:** 18 pages, 12 figures
>
> **摘要:** Existing deep learning-based low-light enhancement methods are typically trained on limited datasets with single enhancement targets, which restricts their generalization ability and controllability in real-world applications. To overcome these limitations, we propose ControlLight, a controllable, consistent, and generalizable framework for low-light enhancement. We first construct a large-scale dataset of real-world degraded images with continuous illumination-strength supervision. To further ensure consistent outputs under different control strengths, we introduce a misalignment-aware weighted flow matching loss that preserves image structure across continuous enhancement strengths. ControlLight allows users to edit real-world degraded low-light images toward satisfactory enhancement results by flexibly controlling the strength while preserving visual consistency and realism. Extensive experiments show that ControlLight achieves state-of-the-art performance against existing low-light enhancement approaches while demonstrating strong continuous controllability and generalization to real-world scenarios.
>
---
#### [replaced 018] Axial-Centric Cross-Plane Attention for 3D Medical Image Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21636](https://arxiv.org/pdf/2602.21636)**

> **作者:** Doyoung Park; Jinsoo Kim; Lohendran Baskaran
>
> **备注:** Submitted to BMVC 2026
>
> **摘要:** Abridged: Clinicians commonly interpret 3D medical images by examining multiple anatomical planes rather than relying on volumetric views. In clinical CT workflows, the axial plane often serves as the primary diagnostic reference, while the auxiliary planes provide complementary spatial context. However, many existing 3D deep learning approaches either process volumetric data holistically or assign equal importance to all planes, failing to reflect this asymmetric, axial-centric interpretation strategy. To address this, we propose an axial-centric cross-plane attention architecture for 3D medical image classification that models asymmetric dependencies between anatomical planes. The architecture employs large-scale axial CT images pretrained MedDINOv3 as a frozen feature extractor for axial, coronal, and sagittal planes. RICA blocks and intra-plane transformer encoders capture plane-specific positional and contextual information, while axial-centric cross-plane transformer encoders selectively condition axial representations on complementary auxiliary representations. Experiments on six datasets from the MedMNIST3D benchmark show that the proposed method consistently outperforms existing 3D and multi-plane models in ACC and AUC. A lightweight variant, AC-Tiny, achieves competitive performance with substantially fewer trainable parameters, suggesting that architectural design contributes more to performance gains than increased model scale. Ablation studies further validate the importance of axial-centric querying, QKV allocation, directional cross-plane fusion, residual-free cross-attention, and classification head design. Slice-level Grad-CAM visualizations demonstrate that the model identifies diagnostically relevant regions across all planes. These findings highlight the value of aligning architectural design with clinical interpretation workflows for robust 3D medical image analysis.
>
---
#### [replaced 019] Suicide Risk Assessment from AI-powered Video Surveillance: An Interpretable Framework for Prevention in Metro Stations
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.22904](https://arxiv.org/pdf/2605.22904)**

> **作者:** Safwen Naimi; Wassim Bouachir; Guillaume-Alexandre Bilodeau; Brian Mishara
>
> **备注:** 9 pages, 6 figures, 1 table. Accepted for Publication in the International Joint Conference of Artificial Intelligence (IJCAI)
>
> **摘要:** Understanding and monitoring human behavior in metro stations play an important role in supporting suicide prevention efforts, where early identification of high-risk situations can enable timely intervention. This requires assessing suicide risk from a surveillance video by jointly reasoning about the behavior of each passenger, his/her spatial context, and temporal dynamics. However, this assessment using videos captured by surveillance cameras is challenging, as it demands accurate perception of human motion, understanding of platform geometry, and aggregation of heterogeneous behavioral cues over time. In this work, we formalize the task of Suicide Risk Assessment (SRA) in metro stations and introduce the first interpretable framework that addresses this challenge. Unlike approaches that focus on isolated subtasks or attempt to infer intent directly, our formulation assesses suicide risk from accumulated evidence by incorporating person tracking, activity recognition, semantic segmentation of the platform, and trajectory-driven risk heatmap modeling. By formalizing SRA as a distinct task and benchmarking a complete operational pipeline achieving 83.2% ROC-AUC on real surveillance data, this work highlights the complexity of suicide risk assessment and opens new directions for research on interpretable AI systems for social good.
>
---
#### [replaced 020] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.12647](https://arxiv.org/pdf/2603.12647)**

> **作者:** ZY Chen; F Zhu; H Zhu; DY Kong; XK Kuang; YJ Zhang; CM Jiang
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis. However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting. To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures. Furthermore, we calibrate LiDAR intensity into reflectance and attach it to each Gaussian as a lighting-invariant material channel, jointly aligned with RGB to enforce boundary consistency. Extensive experiments on the Waymo Open Dataset demonstrate that LR-SGS achieves superior reconstruction performance with fewer Gaussians and shorter training time. In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR.
>
---
#### [replaced 021] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.20606](https://arxiv.org/pdf/2605.20606)**

> **作者:** Muquan Li; Yingyi Ma; Yihong Huang; Hang Gou; Ke Qin; Ming Li; Yuan-Fang Li; Tao He
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Dataset distillation (DD) compresses a large training set into a small synthetic set for efficient training, but most DD methods optimize only clean accuracy and leave robustness uncontrolled. Recent robust DD methods improve robustness, yet they often suffer from a poor accuracy-robustness trade-off because they (i) treat all adversarially perturbed examples uniformly, despite robust risk being dominated by near-zero robust margins, and (ii) do not explicitly increase inter-class separation in the decision boundary where attacks concentrate. We present Contrastive Curriculum for Robust Dataset Distillation (C$^2$R), a framework that couples an attack-aware curriculum with a contrastive robustness objective. From a robust-margin perspective, we derive a perturbation score that approximates each sample's robust hinge, enabling a curriculum that prioritizes the smallest-margin adversaries that most directly drive robust error. In parallel, a class-balanced contrastive robustness loss enforces adversarial invariance while explicitly widening boundary separation across classes. Experiments on CIFAR-10/100, Tiny-ImageNet, and multiple ImageNet-1K subsets under six attacks show that C$^2$R achieves the best robust accuracy, outperforming prior robust DD by $2.8$% on average.
>
---
#### [replaced 022] Diff-Instruct with Diffused Reward: Towards Principled One-step Generator RL
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.24001](https://arxiv.org/pdf/2605.24001)**

> **作者:** Junyi Wu; Weijian Luo; Haoyang Zheng; Ruizhe Zhang; Guang Lin
>
> **备注:** author list correction
>
> **摘要:** Recent advances in one-step text-to-image generation have enabled real-time synthesis with remarkable efficiency and quality. Previous reinforcement learning methods for one-step generators combine image-space reward optimization with diffusion noisy-space distribution matching. This paradigm brings challenges due to a mismatch between terminal reward optimization and the underlying generative dynamics. As a result, optimization tends to exploit stochastic degrees of freedom, often improving reward at the expense of image fidelity. To address this issue, we propose Diff-Instruct with Diffused Reward (DIDR), a data-free trajectory-level alignment framework derived from Integral KL minimization. DIDR propagates the RLHF-optimal reward-tilted clean-image distribution across all noise levels along the diffusion trajectory. We show that this objective admits the same minimizer as clean-image RLHF, while naturally inducing the Diffused Reward Score (DRS), which acts as a reward-driven correction to the reference score function. To make this practical, we further introduce the Diffused Reward Proxy (DRP), an efficient estimator of DRS based on differentiable short-step denoising. Extensive experiments demonstrate that DIDR consistently Pareto-dominates existing one-step SDXL baselines. Moreover, when transferred to a 6B DiT backbone (Z-Image), DIDR surpasses its 50-step teacher in preference alignment while requiring only a single generation step.
>
---
#### [replaced 023] Structured Relational Reasoning for Group Activity Assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.07996](https://arxiv.org/pdf/2508.07996)**

> **作者:** Thinesh Thiyakesan Ponbagavathi; Chengzheng Yang; Alina Roitberg
>
> **备注:** Accepted to CVPR 2026 Workshop (SAUAFG)
>
> **摘要:** Group Activity Detection (GAD) involves recognizing social groups and their collective behaviors in videos. Vision Foundation Models (VFMs), like DINOv2, offer excellent features but are pretrained on object-centric data. We find that naively substituting them into existing GAD pipelines actually degrades performance, exposing structured group-aware decoding as the true bottleneck. We introduce ProGraD, a structured relational-reasoning framework for GAD built on top of frozen VFMs. At its core is a lightweight two-layer GroupContext Transformer that explicitly models actor-group associations and aggregates global context to infer collective behavior. Learnable group prompts serve as a minimal conditioning mechanism to guide the frozen backbone toward socially relevant representations, while the relational decoder performs the core reasoning over actors and groups. This design jointly infers group locations, memberships, and activities in a single pass using only 10M trainable parameters - less than half of prior methods. On the Cafe benchmark with multiple concurrent social groups, ProGraD improves the state-of-the-art by 6.5% Group mAP$@$1.0 and 8.2% Group mAP$@$0.5. On Social-CAD, it achieves state-of-the-art social and membership accuracy. ProGraD further produces interpretable attention maps that provide insights into actor-group reasoning.
>
---
#### [replaced 024] A Unified Framework for Diffusion Model Unlearning with f-Divergence
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21167](https://arxiv.org/pdf/2509.21167)**

> **作者:** Nicola Novello; Federico Fontana; Luigi Cinque; Deniz Gunduz; Andrea M. Tonello
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Most existing methods for concept unlearning in text-to-image diffusion models minimize a mean squared error (MSE) loss between the denoiser outputs conditioned on a target and an anchor concept, which is implicitly the KL divergence between two Gaussians. We generalize this objective to any $f$-divergence, recovering MSE as the KL instance, and identify a family of $\alpha$-divergences whose Gaussian closed-form yields cheap, MSE-like training objectives. For the remaining $f$-divergences, we provide a min-max objective based on the variational formulation of the $f$-divergence. We theoretically analyze and numerically validate how different $f$-divergences impact the gradient magnitude and the convergence properties of the algorithm, affecting the quality of unlearning. For instance, we observe that the Hellinger closed-form instance consistently dominates MSE across multiple scenarios. More generally, the proposed unified framework offers a flexible paradigm for selecting the optimal divergence based on the application and user goal, allowing for finer control over the trade-off between unlearning efficacy and generative fidelity.
>
---
#### [replaced 025] When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR
- **分类: cs.CY; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.22774](https://arxiv.org/pdf/2604.22774)**

> **作者:** Jin Seong; Wencke Liermann; Minho Kim; Jong-hun Shin; Soojong Lim
>
> **摘要:** Accurate transcription of handwritten mathematics is crucial for educational AI systems, yet current benchmarks fail to evaluate this capability properly. Most prior studies focus on single-line expressions and rely on lexical metrics such as BLEU, which fail to assess the semantic reasoning across multi-line student solutions. In this paper, we present the first systematic study of multi-line handwritten math Optical Character Recognition (OCR), revealing a critical failure mode of Vision-Language Models (VLMs): over-correction. Instead of faithfully transcribing a student's work, these models often "fix" errors, thereby hiding the very mistakes an educational assessment aims to detect. To address this, we propose PINK (Penalized INK-based score), a semantic evaluation metric that leverages a Large Language Model (LLM) for rubric-based grading and explicitly penalizes over-correction. Our comprehensive evaluation of 15 state-of-the-art VLMs on the FERMAT dataset reveals substantial ranking reversals compared to BLEU: models like GPT-4o are heavily penalized for aggressive over-correction, whereas Gemini 2.5 Flash emerges as the most faithful transcriber. Furthermore, human expert studies show that PINK aligns significantly better with human judgment (55.0% preference over BLEU's 39.5%), providing a more reliable evaluation framework for handwritten math OCR in educational settings.
>
---
#### [replaced 026] TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04533](https://arxiv.org/pdf/2510.04533)**

> **作者:** Hyunmin Cho; Donghoon Ahn; Susung Hong; Jee Eun Kim; Seungryong Kim; Kyong Hwan Jin
>
> **备注:** Accepted to ICML 2026 (Regular)
>
> **摘要:** Diffusion models achieve state-of-the-art image generation but often produce semantic inconsistencies, or hallucinations. Existing inference-time guidance methods rely on external signals or architectural modifications, adding computational overhead. We propose $\mathbf{T}$angential $\mathbf{A}$mplifying $\mathbf{G}$uidance $\mathbf{(TAG)}$, a training-free, architecture-agnostic, plug-and-play guidance method that operates purely on trajectory signals. TAG uses an intermediate sample as a projection basis and amplifies the tangential components of the estimated score to correct the sampling trajectory. A first-order Taylor analysis shows that this steers the state toward higher-probability regions of the data manifold, reducing inconsistencies and improving fidelity while adding negligible overhead to existing samplers. Code is available at our Project Page (this https URL).
>
---
#### [replaced 027] AD-H: Language-guided Autonomous Driving with Hierarchical Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.03474](https://arxiv.org/pdf/2406.03474)**

> **作者:** Zaibin Zhang; Talas Fu; Shiyu Tang; Yuanhang Zhang; Yifan Wang; Lijun Wang; Huchuan Lu
>
> **摘要:** Language-guided autonomous driving requires bridging a large abstraction gap between high-level natural-language instructions and low-level vehicle control. End-to-end approaches that use a single multimodal large language model (MLLM) to map language directly to actions struggle with this mismatch, often failing to exploit the reasoning capabilities of the model and exhibiting limited generalization beyond the distributions of driving datasets used for fine-tuning. To address this issue, we propose AD-H, a hierarchical multi-agent framework that explicitly separates high-level decision-making from low-level vehicle execution. At the upper level, an MLLM-based planner interprets natural-language commands and environmental context to generate coherent mid-level driving instructions. At the lower level, a lightweight controller converts these mid-level instructions into precise, continuous control actions. This decomposition aligns with the functional strengths of each component: the planner focuses on semantic reasoning and task decomposition, while the controller ensures stable and accurate actuation. To support large-scale training under this hierarchy, we design a rule-based pipeline that reconstructs mid-level commands from driving signals, producing 1.15 million hierarchical annotation pairs. Extensive experiments show that AD-H outperforms state-of-the-art models despite using fewer parameters, namely 3B plus 350M compared with 7B, and achieves superior long-horizon generalization and instruction-following performance. We make our data and code publicly accessible at this https URL
>
---
#### [replaced 028] Training-Free Vector Quantization via Gaussian VAEs
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.06609](https://arxiv.org/pdf/2512.06609)**

> **作者:** Tongda Xu; Wendi Zheng; Jiajun He; Jose Miguel Hernandez-Lobato; Yan Wang; Ya-Qin Zhang; Jie Tang
>
> **摘要:** Vector-quantized variational autoencoders (VQ-VAEs) are discrete autoencoders that compress images into discrete tokens. However, they are difficult to train due to discretization. In this paper, we propose a simple yet effective technique dubbed Gaussian Quant (GQ), which first trains a Gaussian VAE under certain constraints and then converts it into a VQ-VAE without additional training. For conversion, GQ generates random Gaussian noise as a codebook and finds the closest noise vector to the posterior mean. Theoretically, we prove that when the logarithm of the codebook size exceeds the bits-back coding rate of the Gaussian VAE, a small quantization error is guaranteed. Practically, we propose a heuristic to train Gaussian VAEs for effective conversion, named the target divergence constraint (TDC). Empirically, we show that GQ outperforms previous VQ-VAEs, such as VQGAN, FSQ, LFQ, and BSQ, on both UNet and ViT architectures. Furthermore, TDC also improves previous Gaussian VAE discretization methods, such as TokenBridge. The source code is provided in this https URL.
>
---
#### [replaced 029] InHabit: Leveraging Image Foundation Models for Scalable 3D Human Placement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.19673](https://arxiv.org/pdf/2604.19673)**

> **作者:** Nikita Kister; Pradyumna YM; István Sárándi; Jiayi Wang; Anna Khoreva; Gerard Pons-Moll
>
> **摘要:** Training embodied agents to understand 3D scenes as humans do requires large-scale data of people meaningfully interacting with diverse environments, yet such data is scarce. Real-world capture is costly and limited to controlled settings, while existing synthetic datasets rely on simple geometric heuristics, ignoring rich scene context. In contrast, 2D foundation models trained at internet scale have acquired commonsense knowledge of human-environment interactions. To transfer this knowledge to 3D, we introduce InHabit, an automatic and scalable data generator for populating 3D scenes with interacting humans. InHabit follows a render-generate-lift principle: given a rendered 3D scene, a vision-language model proposes contextually meaningful actions, an image-editing model inserts a human, and an optimization procedure lifts the edited result into physically plausible SMPL-X bodies aligned with the scene geometry. Applied to Habitat-Matterport3D, InHabit produces InHabitants, the first large-scale photorealistic 3D human-scene interaction dataset, with 78K samples across $\sim$800 building-scale scenes with complete 3D geometry, SMPL-X bodies, and images. Augmenting standard training data with InHabitants improves RGB-based 3D human-scene reconstruction and contact estimation, and in a perceptual user study our data is preferred in 78% of cases over prior art.
>
---
#### [replaced 030] PyCAT4: A Hierarchical Vision Transformer-based Framework for 3D Human Pose Estimation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.02806](https://arxiv.org/pdf/2508.02806)**

> **作者:** Zongyou Yang; Jonathan Loo; Yinghan Hou
>
> **备注:** 10 pages, 20 figures
>
> **摘要:** Recently, a significant improvement in the accuracy of 3D human pose estimation has been achieved by combining convolutional neural networks (CNNs) with pyramid grid alignment feedback loops. Additionally, innovative breakthroughs have been made in the field of computer vision through the adoption of Transformer-based temporal analysis architectures. Given these advancements, this study aims to deeply optimize and improve the existing Pymaf network architecture. The main innovations of this paper include: (1) Introducing a Transformer feature extraction network layer based on self-attention mechanisms to enhance the capture of low-level features; (2) Enhancing the understanding and capture of temporal signals in video sequences through feature temporal fusion techniques; (3) Implementing spatial pyramid structures to achieve multi-scale feature fusion, effectively balancing feature representations differences across different scales. The new PyCAT4 model obtained in this study is validated through experiments on the COCO and 3DPW datasets. The results demonstrate that the proposed improvement strategies significantly enhance the network's detection capability in human pose estimation, further advancing the development of human pose estimation technology.
>
---
#### [replaced 031] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning
- **分类: cs.CV; cs.CE**

- **链接: [https://arxiv.org/pdf/2604.27604](https://arxiv.org/pdf/2604.27604)**

> **作者:** Junpeng Ding; Zichen Tang; Haihong E; Mengyuan Ji; Yang Liu; Haolin Tian; Haiyang Sun; Pengqi Sun; Yang Xu; Yichen Liu; Haocheng Gao; Zijie Xi; Ruomeng Jiang; Peizhi Zhao; Rongjin Li; Yuanze Li; Jiacheng Liu; Zhongjun Yang; Jintong Chen; Siying Lin
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** We introduce SPUR, a comprehensive benchmark for scientific experimental image perception, understanding, and reasoning, comprising 4,264 question-answering (QA) pairs derived from 1,084 expert-curated images. SPUR features three key innovations: (1) Panel-Level Fine-Grained Perception: evaluating the visual perception of multimodal large language models (MLLMs) across three dimensions (numerical, morphological, and information localization) on six fine-grained panel types; (2) Cross-Panel Relation Understanding: utilizing complex images with an average of 14.3 panels per sample to evaluate MLLMs' ability to decipher intricate cross-panel relations; (3) Expert-Level Reasoning: assessment of qualitative and quantitative reasoning across five experimental paradigms to determine if models can infer conclusions from evidence as human experts do. Comprehensive evaluation of 20 MLLMs and four multimodal Chain-of-Thought (MCoT) methods reveals that current models fall significantly short of the expert-level requirements for scientific image interpretation, underscoring a critical bottleneck in AI for Science (AI4S) research.
>
---
#### [replaced 032] Lifting Data-Tracing Machine Unlearning to Knowledge-Tracing for Foundation Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.11253](https://arxiv.org/pdf/2506.11253)**

> **作者:** Yuwen Tan; Boqing Gong
>
> **备注:** Accepted to TMLR
>
> **摘要:** Machine unlearning removes certain training data points and their influence from AI models (e.g., when a data owner revokes their consent to allow models to learn from the data). In this position paper, we propose to lift data-tracing machine unlearning to knowledge-tracing for foundation models (FMs). We support this position based on practical needs and insights from cognitive studies. Practically, tracing data cannot meet the diverse unlearning requests for FMs, which may be from regulators, enterprise users, product teams, etc., who have no access to FMs' massive training data. Instead, it is convenient for these parties to issue an unlearning request about the knowledge or capability FMs (should not) possess. Cognitively, knowledge-tracing unlearning aligns with how the human brain forgets more closely than tracing individual training data points does. We further discuss the nontrivial challenges in the knowledge-tracing machine unlearning paradigm. Finally, we provide a concrete case study about a vision-language FM to illustrate how an unlearner might instantiate the knowledge-tracing machine unlearning paradigm. Code is available at: this https URL.
>
---
#### [replaced 033] Global Structure-from-Motion Meets Feedforward Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.26103](https://arxiv.org/pdf/2605.26103)**

> **作者:** Linfei Pan; Johannes Schönberger; Marc Pollefeys
>
> **备注:** CVPR 2026, Highlight
>
> **摘要:** Structure-from-Motion -- the process of simultaneously estimating camera poses and 3D scene structure from a collection of images -- remains a central challenge in computer vision, with many open problems yet to be solved. Recent advances in feedforward 3D reconstruction have made significant strides in overcoming persistent failure cases of classical SfM methods, particularly in scenarios characterized by low texture, limited overlap, and symmetries. However, while feedforward approaches excel in these challenging conditions, they often face limitations regarding scalability, accuracy, or robustness, and typically fall short of classical methods in standard reconstruction settings. In this work, we systematically analyze these limitations and propose a new Structure-from-Motion pipeline by combining the respective strengths of classical and feedforward methods. Extensive experiments across multiple datasets show the benefits of our approach, achieving state-of-the-art results across a wide range of scenarios. We share our system as an open-source implementation at this https URL.
>
---
#### [replaced 034] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.00648](https://arxiv.org/pdf/2604.00648)**

> **作者:** Zhengxian Yang; Fei Xie; Xutao Xue; Rui Zhang; Taicheng Huang; Yang Liu; Mengqi Ji; Tao Yu
>
> **备注:** CVPR 2026 Highlight; Fix NSFC ID
>
> **摘要:** 3D Gaussian Splatting (3DGS) has enabled efficient 3D scene reconstruction from everyday images with real-time, high-fidelity rendering, greatly advancing VR/AR applications. Fisheye cameras, with their wider field of view (FOV), promise high-quality reconstructions from fewer inputs and have recently attracted much attention. However, since 3DGS relies on rasterization, most subsequent works involving fisheye camera inputs first undistort images before training, which introduces two problems: 1) Black borders at image edges cause information loss and negate the fisheye's large FOV advantage; 2) Undistortion's stretch-and-interpolate resampling spreads each pixel's value over a larger area, diluting detail density -- causes 3DGS overfitting these low-frequency zones, producing blur and floating artifacts. In this work, we integrate fisheye camera model into the original 3DGS framework, enabling native fisheye image input for training without preprocessing. Despite correct modeling, we observed that the reconstructed scenes still exhibit floaters at image edges: Distortion increases toward the periphery, and 3DGS's original per-iteration random-selecting-view optimization ignores the cross-view correlations of a Gaussian, leading to extreme shapes (e.g., oversized or elongated) that degrade reconstruction quality. To address this, we introduce a feature-overlap-driven cross-view joint optimization strategy that establishes consistent geometric and photometric constraints across views-a technique equally applicable to existing pinhole-camera-based pipelines. Our DirectFisheye-GS matches or surpasses state-of-the-art performance on public datasets. Project Page: this https URL .
>
---
#### [replaced 035] Unique Lives, Shared World: Learning from Single-Life Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04085](https://arxiv.org/pdf/2512.04085)**

> **作者:** Tengda Han; Sayna Ebrahimi; Dilara Gokay; Li Yang Ku; Maks Ovsjanikov; Iva Babukova; Daniel Zoran; Viorica Patraucean; Joao Carreira; Andrew Zisserman; Dima Damen
>
> **摘要:** We introduce the "single-life" learning paradigm, where we train a distinct vision model exclusively on egocentric videos captured by one individual. We leverage the multiple viewpoints naturally captured within a single life to learn a visual encoder in a self-supervised manner. Our experiments demonstrate three key findings. First, models trained independently on different lives develop a highly aligned geometric understanding. We demonstrate this by training visual encoders on distinct datasets each capturing a different life, both indoors and outdoors, as well as introducing a novel cross-attention-based metric to quantify the functional alignment of the internal representations developed by different models. Second, we show that single-life models learn generalizable geometric representations that effectively transfer to downstream tasks, such as depth estimation, in unseen environments. Third, we demonstrate that training on up to 30 hours from one week of the same person's life leads to comparable performance to training on 30 hours of diverse web data, highlighting the strength of single-life representation learning. Overall, our results establish that the shared structure of the world, both leads to consistency in models trained on individual lives, and provides a powerful signal for visual representation learning.
>
---
#### [replaced 036] Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.16449](https://arxiv.org/pdf/2511.16449)**

> **作者:** Ziyan Liu; Yeqiu Chen; Hongyi Cai; Tao Lin; Shuo Yang; Zheng Liu; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have shown great potential for embodied AI by integrating visual perception, language understanding, and action execution. In real-time deployment, these models must process continuous visual streams, incurring substantial computational overhead. Visual token pruning -- a mainstream technique for accelerating Vision-Language Models (VLMs) by retaining salient tokens while discarding redundant ones -- offers a natural candidate solution to this challenge. However, directly applying VLM-oriented pruning methods to VLA inference can cause severe degradation in manipulation performance. Our analysis attributes this degradation to a key mismatch: VLA inference exhibits distinct attention patterns between the vision-language prefill stage and the action-decode stage, so pruning based only on context-prefill semantic salience is biased toward semantic cues and may remove action-critical visual tokens. Motivated by this observation, we propose VLA-Pruner, an effective plug-and-play token pruning method grounded in the visual requirements of VLA inference, further exploiting the temporal continuity of robot manipulation. Specifically, VLA-Pruner estimates visual-token importance from both semantic prefilling and temporally smoothed action relevance, and then applies a Combine-then-Filter strategy to retain compact, non-redundant tokens under the compute budget. Experiments show that VLA-Pruner outperforms state-of-the-art approaches across multiple VLA architectures, achieving up to 1.99x speedup with comparable manipulation quality.
>
---
#### [replaced 037] CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.16417](https://arxiv.org/pdf/2405.16417)**

> **作者:** Lin Zhu; Yifeng Yang; Qinying Gu; Xinbing Wang; Chenghu Zhou; Nanyang Ye
>
> **摘要:** Recent vision-language pre-trained models (VL-PTMs) have shown remarkable success in open-vocabulary tasks. However, downstream use cases often involve further fine-tuning of VL-PTMs, which may distort their general knowledge and impair their ability to handle distribution shifts. In real-world scenarios, machine learning systems inevitably encounter both covariate shifts (e.g., changes in image styles) and semantic shifts (e.g., test-time unseen classes). This highlights the importance of enhancing out-of-distribution (OOD) generalization on covariate shifts and simultaneously detecting semantic-shifted unseen classes. Thus a critical but underexplored question arises: How to improve VL-PTMs' generalization ability to closed-set OOD data, while effectively detecting open-set unseen classes during fine-tuning? In this paper, we propose a novel objective function of OOD detection that also serves to improve OOD generalization. We show that minimizing the gradient magnitude of energy scores on training data leads to domain-consistent Hessians of classification loss, a strong indicator for OOD generalization revealed by theoretical analysis. Based on this finding, we have developed a unified fine-tuning framework that allows for concurrent optimization of both tasks. Extensive experiments have demonstrated the superiority of our method. The code is available at this https URL.
>
---
#### [replaced 038] Broken Memories: Detecting and Mitigating Memorization in Diffusion Models with Degraded Generations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.22050](https://arxiv.org/pdf/2605.22050)**

> **作者:** Yuanmin Huang; Mi Zhang; Chen Chen; Feifei Li; Geng Hong; Xiaoyu You; Min Yang
>
> **备注:** KDD 2026, extended version
>
> **摘要:** While diffusion models excel at generating high-quality images, their tendency to memorize training data poses significant privacy and copyright risks. In this work, we for the first time identify that memorization induces internal numerical instability, often manifesting as visually ``broken'' artifacts. Inspired by stability analysis in numerical methods, we introduce empirical stability regions based on latent update norms to quantitatively characterize stable behavior during generation. Leveraging this, we propose a principled, on-the-fly framework for step-wise detection and adaptive mitigation. Our approach suppresses memorization without altering prompts or guidance, thereby preserving semantic fidelity and image quality. Extensive experiments on Stable Diffusion 1.4 demonstrate that our method achieves an AUC $>0.999$ detection performance and a $0.0\%$ memorization rate after mitigation with negligible overhead ($\approx0.01$s per image).
>
---
#### [replaced 039] Learning GUI Grounding with Spatial Reasoning from Visual Feedback
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究GUI接地任务，解决高分辨率GUI中坐标预测不准确的问题，提出GUI-Cursor模型通过交互搜索和空间推理定位UI元素。**

- **链接: [https://arxiv.org/pdf/2509.21552](https://arxiv.org/pdf/2509.21552)**

> **作者:** Yu Zhao; Wei-Ning Chen; Huseyin Atahan Inan; Samuel Kessler; Lu Wang; Lukas Wutschitz; Fangkai Yang; Chaoyun Zhang; Pasquale Minervini; Saravan Rajmohan; Robert Sim
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Graphical User Interface (GUI) grounding is commonly framed as a coordinate prediction task -- given a natural language instruction, generate on-screen coordinates for actions such as clicks and keystrokes. However, recent Vision Language Models (VLMs) often fail to predict accurate numeric coordinates when processing GUI images with high resolutions and complex layouts. To address this issue, we reframe GUI grounding as an interactive search task, where the VLM generates actions to move a cursor in the GUI to locate UI elements. At each step, the model determines the target object, evaluates the spatial relations between the cursor and the target, and moves the cursor closer to the target conditioned on the movement history. In this interactive process, the rendered cursor provides visual feedback to help the model align its predictions with the corresponding on-screen locations. We train our GUI grounding model, GUI-Cursor, using multi-step online reinforcement learning with a dense trajectory-based reward function. Experimental results demonstrate that GUI-Cursor surpasses strong baselines in GUI grounding and agentic tasks, achieving superior performance with the same base models while requiring less training data. Further analysis shows that GUI-Cursor learns to adaptively conduct more steps on more difficult examples, and it obtains better spatial reasoning capability on out-of-distribution domains.
>
---
#### [replaced 040] Innovative Silicosis and Pneumonia Classification: Leveraging Graph Transformer Post-hoc Modeling and Ensemble Techniques
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.00520](https://arxiv.org/pdf/2501.00520)**

> **作者:** Bao Q. Bui; Tien T.T. Nguyen; Duy M. Le; Cong Tran; Cuong Pham
>
> **备注:** Withdrawn by the authors because the manuscript contains incomplete and potentially misleading descriptions of the dataset construction and evaluation protocol, particularly in the Dataset and Experimental Setup sections. The work should not be cited or used as an independent reference in its current form
>
> **摘要:** This paper presents a comprehensive study on the classification and detection of Silicosis-related lung inflammation. Our main contributions include 1) the creation of a newly curated chest X-ray (CXR) image dataset named SVBCX that is tailored to the nuances of lung inflammation caused by distinct agents, providing a valuable resource for silicosis and pneumonia research community; and 2) we propose a novel deep-learning architecture that integrates graph transformer networks alongside a traditional deep neural network module for the effective classification of silicosis and pneumonia. Additionally, we employ the Balanced Cross-Entropy (BalCE) as a loss function to ensure more uniform learning across different classes, enhancing the model's ability to discern subtle differences in lung conditions. The proposed model architecture and loss function selection aim to improve the accuracy and reliability of inflammation detection, particularly in the context of Silicosis. Furthermore, our research explores the efficacy of an ensemble approach that combines the strengths of diverse model architectures. Experimental results on the constructed dataset demonstrate promising outcomes, showcasing substantial enhancements compared to baseline models. The ensemble of models achieves a macro-F1 score of 0.9749 and AUC ROC scores exceeding 0.99 for each class, underscoring the effectiveness of our approach in accurate and robust lung inflammation classification.
>
---
#### [replaced 041] From Contrast to Consistency: Rethinking Event-based Continuous-Time Optical Flow Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.25570](https://arxiv.org/pdf/2605.25570)**

> **作者:** Rui Hu; Song Wu; Wen Yang; Jinjian Wu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Estimating continuous optical flow is a fundamental yet challenging problem in dynamic visual perception. Event-based cameras, with microsecond latency and high dynamic range, capture brightness changes asynchronously, offering a unique opportunity to model motion with fine temporal precision. However, the scarcity of temporally dense ground-truth annotations limits the effectiveness of supervised learning, while contrast maximization (CM) frameworks, focused on sharpening the Image of Warped Events (IWE), often neglect temporal continuity and structural coherence, leading to distorted trajectories under complex motion. To overcome these challenges, we propose a hybrid-supervised framework for continuous-time optical flow estimation, grounded in the principle of Spatio-temporal Structural Consistency (STSC). This paradigm jointly enforces local structural stability and trajectory continuity, ensuring physically coherent motion across time. To further enhance representation and robustness, we design a bidirectionally complementary multi-scale architecture and employ a curriculum-guided hybrid training strategy, enabling a smooth transition from supervised point constraints to self-supervised manifold regularization. Comprehensive experiments across multiple benchmarks show that our method achieves state-of-the-art performance in both continuous-time and standard optical flow estimation, demonstrating the effectiveness of the proposed learning paradigm.
>
---
#### [replaced 042] Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出Athena-PRM，解决多模态推理中步骤奖励评估问题，通过数据高效方法提升PRM性能，显著提高多个基准测试表现。**

- **链接: [https://arxiv.org/pdf/2506.09532](https://arxiv.org/pdf/2506.09532)**

> **作者:** Shuai Wang; Zhenhua Liu; Jiaheng Wei; Xuanwu Yin; Dong Li; Emad Barsoum
>
> **备注:** TMLR 2026, this https URL
>
> **摘要:** We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks.
>
---
#### [replaced 043] Advancing Metallic Surface Defect Detection via Anomaly-Guided Pretraining on a Large Industrial Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.18919](https://arxiv.org/pdf/2509.18919)**

> **作者:** Chuni Liu; Hongjie Li; Jiaqi Du; Yangyang Hou; Qian Sun; Lei Jin; Ke Xu
>
> **备注:** Accepted for publication in Pattern Recognition
>
> **摘要:** The pretraining-finetuning paradigm is a crucial strategy in metallic surface defect detection for mitigating the challenges posed by data scarcity. However, its implementation presents a critical dilemma. Pretraining on natural image datasets such as ImageNet, faces a significant domain gap. Meanwhile, naive self-supervised pretraining on in-domain industrial data is often ineffective due to the inability of existing learning objectives to distinguish subtle defect patterns from complex background noise and textures. To resolve this, we introduce Anomaly-Guided Self-Supervised Pretraining (AGSSP), a novel paradigm that explicitly guides representation learning through anomaly priors. AGSSP employs a two-stage framework: (1) it first pretrains the model's backbone by distilling knowledge from anomaly maps, encouraging the network to capture defect-salient features; (2) it then pretrains the detector using pseudo-defect boxes derived from these maps, aligning it with localization tasks. To enable this, we develop a knowledge-enhanced method to generate high-quality anomaly maps and collect a large-scale industrial dataset of 120,000 images. Additionally, we present two small-scale, pixel-level labeled metallic surface defect datasets for validation. Extensive experiments demonstrate that AGSSP consistently enhances performance across various settings, achieving up to a 10\% improvement in mAP@0.5 and 11.4\% in mAP@0.5:0.95 compared to ImageNet-based models. All code, pretrained models, and datasets are publicly available at this https URL.
>
---
#### [replaced 044] ImViD: Immersive Volumetric Videos for Enhanced VR Engagement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.14359](https://arxiv.org/pdf/2503.14359)**

> **作者:** Zhengxian Yang; Shi Pan; Shengqi Wang; Haoxiang Wang; Li Lin; Guanjun Li; Zhengqi Wen; Borong Lin; Jianhua Tao; Tao Yu
>
> **备注:** CVPR 2025 Highlight; Fix NSFC ID
>
> **摘要:** User engagement is greatly enhanced by fully immersive multi-modal experiences that combine visual and auditory stimuli. Consequently, the next frontier in VR/AR technologies lies in immersive volumetric videos with complete scene capture, large 6-DoF interaction space, multi-modal feedback, and high resolution & frame-rate contents. To stimulate the reconstruction of immersive volumetric videos, we introduce ImViD, a multi-view, multi-modal dataset featuring complete space-oriented data capture and various indoor/outdoor scenarios. Our capture rig supports multi-view video-audio capture while on the move, a capability absent in existing datasets, significantly enhancing the completeness, flexibility, and efficiency of data capture. The captured multi-view videos (with synchronized audios) are in 5K resolution at 60FPS, lasting from 1-5 minutes, and include rich foreground-background elements, and complex dynamics. We benchmark existing methods using our dataset and establish a base pipeline for constructing immersive volumetric videos from multi-view audiovisual inputs for 6-DoF multi-modal immersive VR experiences. The benchmark and the reconstruction and interaction results demonstrate the effectiveness of our dataset and baseline method, which we believe will stimulate future research on immersive volumetric video production.
>
---
#### [replaced 045] Inference-Time Search Using Side Information for Diffusion-Based Image Reconstruction
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.03352](https://arxiv.org/pdf/2510.03352)**

> **作者:** Mahdi Farahbakhsh; Vishnu Teja Kunde; Dileep Kalathil; Krishna Narayanan; Jean-Francois Chamberland
>
> **摘要:** Diffusion models have been used as priors for solving inverse problems. However, existing approaches typically overlook side information that could significantly improve reconstruction quality, especially in severely ill-posed settings. In this work, we propose a novel framework that incorporates side information into existing diffusion-based inverse problem solvers via inference-time search, in a plug-and-play, training-free manner. Through extensive experiments across a range of inverse problems, including inpainting, super-resolution, and several deblurring tasks, and across multiple diffusion-based inverse problem solvers (DPS, DAPS, and MPGD), we show that augmenting each solver with our framework consistently improves the quality of the reconstructions over the corresponding original method. To demonstrate the generality of our approach, we consider diverse forms of side information, including reference images, textual descriptions, and anatomical MRI scans. The code is available at this \href{this https URL}{repository}\footnote{this https URL}.
>
---
#### [replaced 046] Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型（VLM）的蒸馏任务，旨在解决学生模型在推理过程中遗忘视觉证据的问题。通过引入基于视觉锚定的推理前缀掩码策略，提升学生模型对视觉信息的利用能力。**

- **链接: [https://arxiv.org/pdf/2605.11651](https://arxiv.org/pdf/2605.11651)**

> **作者:** Seonghoon Yu; Dongjun Nam; Byung-Kwan Lee; Jeany Son
>
> **备注:** Pre-print
>
> **摘要:** Recent think-answer approaches in VLMs, such as Qwen3-VL-Thinking, boost reasoning performance by leveraging intermediate thinking steps before the final answer, but their computational cost becomes substantial, especially for larger VLMs. To distill such capabilities into compact think-answer VLMs, a primary objective is to improve the student's ability to utilize visual evidence throughout its reasoning trace, as long think-answer traces suffer from visual forgetting issues. To this end, we introduce a novel think-answer distillation framework that encourages the student to anchor its thinking on visual information by masking the student's salient reasoning prefixes. To compensate for such masked textual cues, the student is encouraged to rely more on visual evidence as an alternative source of information during distillation. Our masking strategies include: 1) token-wise salient reasoning-prefix masking, which masks high-influence reasoning prefixes selectively for each next-token prediction, and 2) self-paced masking budget scheduling, which gradually increases the masking scale according to distillation difficulty, measured by the discrepancy between teacher--student distributions. In the distillation phase, the student is guided by our salient reasoning-prefix mask, which blocks both future tokens and salient reasoning cues, in place of the standard causal mask used for auto-regressive language modeling. Experimental results show that our approach outperforms recent open-source VLMs, VLM distillation, and self-distillation methods on multimodal reasoning benchmarks, while further analyzes confirm enhanced visual utilization along the student thinking process.
>
---
#### [replaced 047] Dual-Thresholded Heatmap-Guided Proposal Clustering and Negative Certainty Supervision with Enhanced Base Network for Weakly Supervised Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.08289](https://arxiv.org/pdf/2509.08289)**

> **作者:** Yuelin Guo; Haoyu He; Zhiyuan Chen; Zitong Huang; Renhao Lu; Lu Shi; Zejun Wang; Weizhe Zhang
>
> **备注:** IEEE TIP Minor Revision
>
> **摘要:** Weakly supervised object detection (WSOD) has attracted significant attention in recent years, as it does not require box-level annotations. State-of-the-art methods generally adopt a multi-module network, which employs WSDDN as the multiple instance detection network module and uses multiple instance refinement modules to refine performance. However, these approaches suffer from three key limitations. First, existing methods tend to generate pseudo GT boxes that either focus only on discriminative parts, failing to capture the whole object, or cover the entire object but fail to distinguish between adjacent intra-class instances. Second, the foundational WSDDN architecture lacks a crucial background class representation for each proposal and exhibits a large semantic gap between its branches. Third, prior methods discard ignored proposals during optimization, leading to slow convergence. To address these challenges, we propose the Dual-thresholded heAtmap-guided proposal clustering and Negative Certainty supervision with Enhanced base network (DANCE) method for WSOD. Specifically, we first devise a heatmap-guided proposal selector (HGPS) algorithm, which utilizes dual thresholds on heatmaps to pre-select proposals, enabling pseudo GT boxes to both capture the full object extent and distinguish between adjacent intra-class instances. We then construct a weakly supervised basic detection network (WSBDN), which augments each proposal with a background class representation and uses heatmaps for pre-supervision to bridge the semantic gap between matrices. At last, we introduce a negative certainty supervision (NCS) loss on ignored proposals to accelerate convergence. Extensive experiments on the challenging PASCAL VOC and MS COCO datasets demonstrate the effectiveness and superiority of our method. Our code is publicly available at this https URL.
>
---
#### [replaced 048] Self-Cascaded Diffusion Models for Arbitrary-Scale Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.07813](https://arxiv.org/pdf/2506.07813)**

> **作者:** Junseo Bang; Joonhee Lee; Kyeonghyun Lee; Haechang Lee; Dong Un Kang; Se Young Chun
>
> **摘要:** Arbitrary-scale image super-resolution aims to upsample images to any desired resolution, offering greater flexibility than traditional fixed-scale super-resolution. Recent approaches based on regression-based or generative models have shown promising results but often suffer from scale inconsistency due to their single-stage formulation, which must handle a wide range of scaling factors simultaneously. To address this, we propose CasArbi, a self-cascaded diffusion framework for arbitrary-scale image super-resolution. CasArbi decomposes varying scaling factors into smaller sequential steps, progressively enhancing the image resolution at each step with seamless transitions for arbitrary scales. CasArbi leverages a coordinate-conditioned diffusion model for learning continuous image representations and adopts self-consistency guidance to generate scale-consistent details at inference time. Extensive experiments show that CasArbi outperforms existing methods in both perceptual and distortion metrics and demonstrates superior scale consistency across diverse arbitrary-scale super-resolution benchmarks. Our code is available at this https URL.
>
---
#### [replaced 049] SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出SOLE-R1，用于机器人强化学习的视频-语言推理奖励模型，解决无监督任务学习问题，通过自然语言目标生成密集奖励信号。**

- **链接: [https://arxiv.org/pdf/2603.28730](https://arxiv.org/pdf/2603.28730)**

> **作者:** Philip Schroeder; Thomas Weng; Karl Schmeckpeper; Eric Rosen; Stephen Hart; Ondrej Biza
>
> **摘要:** Vision-language models (VLMs) have shown impressive capabilities across diverse tasks, motivating efforts to leverage these models to supervise robot learning. However, when used as evaluators in reinforcement learning (RL), today's strongest models often fail under partial observability and distribution shift, enabling policies to exploit perceptual errors rather than solve the task. We introduce SOLE-R1 (Self-Observing LEarner), a video-language reasoning model explicitly designed to serve as the sole reward signal for online RL. Given only raw video observations and a natural-language goal, SOLE-R1 performs per-timestep spatiotemporal chain-of-thought (CoT) reasoning and produces dense estimates of task progress that can be used directly as rewards. To train SOLE-R1, we develop a large-scale video trajectory and reasoning synthesis pipeline that generates temporally grounded CoT traces aligned with continuous progress supervision. This data is combined with foundational spatial and multi-frame temporal reasoning, and used to train the model with a hybrid framework that couples supervised fine-tuning with RL from verifiable rewards. Across four different simulation environments and a real-robot setting, SOLE-R1 enables zero-shot online RL from random initialization: robots learn previously unseen manipulation tasks without ground-truth rewards, success indicators, demonstrations, or task-specific tuning. SOLE-R1 succeeds on 24 unseen tasks and substantially outperforms strong vision-language rewarders, including Robometer, RoboReward, ReWiND, GPT-5, and Gemini-3-Pro, while exhibiting markedly greater robustness to reward hacking. We release all models, data, code, and demos at the anonymous page: this https URL
>
---
#### [replaced 050] Seeing vs. Believing: Evaluating the Language Bias of Open-Source MLLMs in Counter-Intuitive Scenes
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.07737](https://arxiv.org/pdf/2601.07737)**

> **作者:** Chen Ling; Tongwei Zhang; Hanqian Li; Nai Ding
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable performance in mainstream visual understanding tasks, but their ability to process action scenes that contradict everyday common sense remains undertested. To address this gap, we introduce CAIT, a benchmark comprising 400 high-fidelity synthetic scenes focused on counter-intuitive visual actions, such as ``a rabbit is chasing a tiger'', where visual evidence explicitly contradicts common-sense expectations. We evaluate human, leading proprietary models (e.g., Claude and Gemini), and 14 representative open-source MLLMs. Humans achieve near-perfect performance (around 0.95 accuracy) and proprietary models demonstrate robust understanding (achieving up to 0.88 accuracy), standard open-source instruction-tuned models perform at the chance level. Further analysis demonstrates that this failure is driven by a strong language prior: rather than trusting the visual input, they automatically override the anomalous visual signals with statistically common text descriptions. Although introducing Chain-of-Thought reasoning mechanisms can improve accuracy, it significantly slows down the response and generates a new failure mode: models overthink the scenario and refuse to accept the actual visual content simply because it violates real-world physical laws. Finally, we demonstrate that targeted fine-tuning and structured prompting can effectively mitigate this reliance on language priors, enabling open-source models to accurately ground their reasoning in actual visual evidence.
>
---
#### [replaced 051] Hands-On: Segmenting Individual Signs from Continuous Sequences
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.08593](https://arxiv.org/pdf/2504.08593)**

> **作者:** JianHe Low; Harry Walsh; Ozge Mercanoglu Sincan; Richard Bowden
>
> **备注:** Accepted in the 19th IEEE International Conference on Automatic Face and Gesture Recognition. Code Implementation Released
>
> **摘要:** This work tackles the challenge of continuous sign language segmentation, a key task with huge implications for sign language translation and data annotation. We propose a transformer-based architecture that models the temporal dynamics of signing and frames segmentation as a sequence labeling problem using the Begin-In-Out (BIO) tagging scheme. Our method leverages the HaMeR hand features, and is complemented with 3D Angles. Extensive experiments show that our model achieves state-of-the-art results on the DGS Corpus, while our features surpass prior benchmarks on BSLCorpus.
>
---
#### [replaced 052] SRL-CLIP: Efficient CLIP Video Adaptation via Structured Semantic Role Labels
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2401.07669](https://arxiv.org/pdf/2401.07669)**

> **作者:** Darshan Singh; Zeeshan Khan; Makarand Tapaswi
>
> **备注:** Accepted to the CV4Smalls Workshop at CVPR 2026
>
> **摘要:** Adapting CLIP for videos has gained popularity due to its semantic and rich representation. While CLIP is a good starting point, it typically undergoes post-pretraining (contrastive finetuning) on large video narration or caption datasets (e.g. HowTo100M, WebVid2.5M). However, such narrations or captions often lack comprehensive information needed to represent a video holistically. As the learning signal from text is sparse, the visual learning is inefficient and adaptation requires millions of samples to post-pretrain. In this work, we ask: is it possible to efficiently adapt CLIP for general and holistic video understanding? We use videos labeled with structured and dense Semantic Role Labels (SRLs) that capture actions, people or objects, their attributes, adverbs (manner), and location in a structured format representing the entire video in a holistic way. We generate rule-based captions from SRLs and demonstrate that simple contrastive finetuning on a mere 23k video-caption pairs is adequate to learn powerful, transferable representations applicable across a diverse range of video understanding tasks that require varying levels of perceptual granularity. Our adapted CLIP model, SRL-CLIP, exhibits comparable or superior performance on zero-shot text-to-video retrieval compared to state-of-the-art models that possess 4-8x more parameters and are post-pretrained on up to 6000x more data. SRL-CLIP surpasses CLIP on multiple video benchmarks, underscoring the efficient learning and improved representations.
>
---
#### [replaced 053] AI-T2I: Aggregating-and-Isolating Cross-Attention to Diffusion Models for Text-to-Image Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.25763](https://arxiv.org/pdf/2605.25763)**

> **作者:** Shipeng Cao; Biao Qian; Haipeng Liu; Yang Wang; Meng Wang
>
> **备注:** Accepted by IEEE Transactions on Multimedia (2026). 13 pages, 15 figures
>
> **摘要:** Text-to-image synthesis has made significant progress, benefiting from the strong generative capabilities of diffusion models. However, these models struggle to achieve precise text-to-image alignment within cross-attention maps during the denoising process. Existing works primarily focus on inter-subject-token activations (i.e., cross-attention scores) overlap for different subjects, overlooking the intra-subject-token activations scattering issue for identical subjects. In this paper, we propose an Aggregating-and-Isolating cross-attention approach to diffusion models for Text-to-Image synthesis, dubbed AI-T2I. Technically, to address the scattering issue, we devise an aggregation loss to identify and consolidate the scattered intra-token activations, which implicitly helps mitigate the potential overlap issue. Upon that, an isolation loss is further introduced to push the inter-token activations apart, thus fulfilling precise text-to-image alignment. Extensive experiments on various benchmarks demonstrate the superiority of AI-T2I over the state-of-the-art works for text-to-image synthesis. Furthermore, our AI-T2I exhibits excellent generalization across other tasks, e.g., controllable layout generation and personalized generation.
>
---
#### [replaced 054] RadJEPA: Radiology Encoder for Chest X-Rays via Joint Embedding Predictive Architecture
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.15891](https://arxiv.org/pdf/2601.15891)**

> **作者:** Anas Anwarul Haq Khan; Mariam Husain; Pratik Jalan; Kshitij Jadhav
>
> **摘要:** Vision-language pretraining has driven much of the recent progress in medical image representation learning, but this paradigm is constrained by the availability of paired image-text data and by the reporting bias of clinical narratives. We ask whether competitive radiology encoders can be learned without any language supervision. We introduce RadJEPA, a self-supervised framework built on a Joint Embedding Predictive Architecture and pretrained on approximately 840K unlabeled chest X-ray images. The model learns to predict latent representations of masked target regions from a visible context region, an objective that differs from both image-text contrastive pretraining and DINO-style self-distillation by explicitly modelling conditional structure in representation space. We evaluate RadJEPA primarily on radiology report generation with a frozen Vicuna-7B decoder, and additionally substitute its encoder into four widely used vision-language backbones (MedLLaVA, Qwen-2.5, BLIP-2, and Phi-4). For completeness we also report disease classification and semantic segmentation results. Across two datasets and four metrics, RadJEPA matches or exceeds the strongest image-only and vision-language baselines while using a ViT-B/14 backbone at 224 x 224 resolution.
>
---
#### [replaced 055] Datasets for Lane Detection in Autonomous Driving: A Comprehensive Review
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.08540](https://arxiv.org/pdf/2504.08540)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Oliver Bringmann
>
> **摘要:** Accurate lane detection is essential for automated driving, enabling safe and reliable vehicle navigation across a variety of road scenarios. Numerous datasets have been introduced to support the development and evaluation of lane detection algorithms, each differing in terms of the amount of data, sensor types, annotation granularity, environmental conditions, and scenario diversity. This paper provides a comprehensive review of 20 publicly available lane detection datasets, systematically analyzing their characteristics, advantages, and limitations. We classify these datasets based on key performance indicators such as sensor resolution, annotation types and diversity of road and weather conditions using a novel multidimensional metric for dataset quality. By identifying existing challenges and research gaps, we highlight opportunities for future dataset improvements that can further drive innovation in robust lane detection. This review serves as a resource for researchers seeking appropriate datasets for robust lane detection and contributes to the broader goal of advancing autonomous driving.
>
---
#### [replaced 056] Can Visual Mamba Improve AI-Generated Image Detection? An In-Depth Investigation
- **分类: cs.CV; cs.CR; cs.SI**

- **链接: [https://arxiv.org/pdf/2605.14799](https://arxiv.org/pdf/2605.14799)**

> **作者:** Mamadou Keita; Wassim Hamidouche; Hessen Bougueffa Eutamene; Abdelmalik Taleb-Ahmed; Xianxun Zhu; Abdenour Hadid
>
> **摘要:** In recent years, computer vision has witnessed remarkable progress, fueled by the development of innovative architectures such as Convolutional Neural Networks (CNNs), Generative Adversarial Networks (GANs), diffusion-based architectures, Vision Transformers (ViTs), and, more recently, Vision-Language Models (VLMs). This progress has undeniably contributed to creating increasingly realistic and diverse visual content. However, such advancements in image generation also raise concerns about potential misuse in areas such as misinformation, identity theft, and threats to privacy and security. In parallel, Mamba-based architectures have emerged as versatile tools for a range of image analysis tasks, including classification, segmentation, medical imaging, object detection, and image restoration, in this rapidly evolving field. However, their potential for identifying AI-generated images remains relatively unexplored compared to established techniques. This study provides a systematic evaluation and comparative analysis of Vision Mamba models for AI-generated image detection. We benchmark multiple Vision Mamba variants against representative CNNs, ViTs, and VLM-based detectors across diverse datasets and synthetic image sources, focusing on key metrics such as accuracy, efficiency, and generalizability across diverse image types and generative models. Through this comprehensive analysis, we aim to elucidate Vision Mamba's strengths and limitations relative to established methodologies in terms of applicability, accuracy, and efficiency in detecting AI-generated images. Overall, our findings highlight both the promise and current limitations of Vision Mamba as a component in systems designed to distinguish authentic from AI-generated visual content. This research is crucial for enhancing detection in an age where distinguishing between real and AI-generated content is a major challenge.
>
---
#### [replaced 057] World-R1: Reinforcing 3D Constraints for Text-to-Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.24764](https://arxiv.org/pdf/2604.24764)**

> **作者:** Weijie Wang; Xiaoxuan He; Youping Gu; Yifan Yang; Zeyu Zhang; Yefei He; Yanbo Ding; Xirui Hu; Donny Y. Chen; Zhiyuan He; Yuqing Yang; Bohan Zhuang
>
> **备注:** ICML 2026, Project Page: this https URL, Code: this https URL
>
> **摘要:** Recent video foundation models demonstrate impressive visual synthesis but frequently suffer from geometric inconsistencies. While existing methods attempt to inject 3D priors via architectural modifications, they often incur high computational costs and limit scalability. We propose World-R1, a framework that aligns video generation with 3D constraints through reinforcement learning. To facilitate this alignment, we introduce a specialized pure text dataset tailored for world simulation. Utilizing Flow-GRPO, we optimize the model using feedback from pre-trained 3D foundation models and vision-language models to enforce structural coherence without altering the underlying architecture. We further employ a periodic decoupled training strategy to balance rigid geometric consistency with dynamic scene fluidity. Extensive evaluations reveal that our approach significantly enhances 3D consistency while preserving the original visual quality of the foundation model, effectively bridging the gap between video generation and scalable world simulation.
>
---
#### [replaced 058] Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.14993](https://arxiv.org/pdf/2511.14993)**

> **作者:** Vladimir Arkhipkin; Vladimir Korviakov; Nikolai Gerasimenko; Denis Parkhomenko; Viacheslav Vasilev; Alexey Letunovskiy; Nikolai Vaulin; Maria Kovaleva; Ivan Kirillov; Lev Novitskiy; Denis Koposov; Nikita Kiselev; Alexander Varlamov; Dmitrii Mikhailov; Vladimir Polovnikov; Andrey Shutkin; Julia Agafonova; Ilya Vasiliev; Anastasiia Kargapoltseva; Anna Dmitrienko; Anastasia Maltseva; Anna Averchenkova; Olga Kim; Tatiana Nikulina; Denis Dimitrov
>
> **备注:** Website: this https URL
>
> **摘要:** This report introduces Kandinsky 5.0, a family of state-of-the-art foundation models for high-resolution image and 10-second video synthesis. The framework comprises three core line-up of models: Kandinsky 5.0 Image Lite - a line-up of 6B parameter image generation models, Kandinsky 5.0 Video Lite - a fast and lightweight 2B parameter text-to-video and image-to-video models, and Kandinsky 5.0 Video Pro - 19B parameter models that achieves superior video generation quality. We provide a comprehensive review of the data curation lifecycle - including collection, processing, filtering and clustering - for the multi-stage training pipeline that involves extensive pre-training and incorporates quality-enhancement techniques such as self-supervised fine-tuning (SFT) and reinforcement learning (RL)-based post-training. We also present novel architectural, training, and inference optimizations that enable Kandinsky 5.0 to achieve high generation speeds and state-of-the-art performance across various tasks, as demonstrated by human evaluation. As a large-scale, publicly available generative framework, Kandinsky 5.0 leverages the full potential of its pre-training and subsequent stages to be adapted for a wide range of generative applications. We hope that this report, together with the release of our open-source code and training checkpoints, will substantially advance the development and accessibility of high-quality generative models for the research community.
>
---
#### [replaced 059] Align & Invert: Solving Inverse Problems with Diffusion and Flow-based Models via Representation Alignment
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.16870](https://arxiv.org/pdf/2511.16870)**

> **作者:** Loukas Sfountouris; Giannis Daras; Paris Giampouras
>
> **摘要:** Enforcing alignment between the internal representations of diffusion or flow-based generative models and those of pretrained self-supervised encoders has recently been shown to provide a powerful inductive bias, improving both convergence and sample quality. In this work, we extend this idea to inverse problems, where pretrained generative models are employed as priors. We propose applying representation alignment (REPA) between diffusion or flow-based models and a DINOv2 visual encoder, to guide the reconstruction process at inference time. Although ground-truth signals are unavailable in inverse problems, we empirically show that aligning model representations of approximate target features can substantially enhance reconstruction quality and perceptual realism. We provide theoretical results showing (a) that REPA regularization can be viewed as a variational approach for minimizing a divergence measure in the DINOv2 embedding space, and (b) how under certain regularity assumptions REPA updates steer the latent diffusion states toward those of the clean image. These results offer insights into the role of REPA in improving perceptual fidelity. Finally, we demonstrate the generality of our approach by We integrate REPA into multiple state-of-the-art inverse problem solvers, and provide extensive experiments on super-resolution, box inpainting, Gaussian deblurring, and motion deblurring confirming that our method consistently improves reconstruction quality, while also providing efficiency gains reducing the number of required discretization steps.
>
---
#### [replaced 060] To See or To Please: Uncovering Visual Sycophancy and Split Beliefs in VLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.18373](https://arxiv.org/pdf/2603.18373)**

> **作者:** Rui Hong; Shuxue Quan
>
> **备注:** 14 pages, 1 figures
>
> **摘要:** When VLMs answer correctly, do they genuinely rely on visual information? We introduce a Tri-Layer Diagnostic Framework with three per-sample metrics: Latent Anomaly Detection, Visual Necessity Score, and Competition Score, which disentangle perception, dependency, and alignment failures. Across 9 VLMs and 9,000 model-sample pairs under counterfactual blind, noise, and conflict interventions, 72.9% of samples exhibit Visual Sycophancy, a Split Beliefs pattern in which internal evidence is preserved yet a hallucinated answer is decoded, while zero samples show Robust Refusal, indicating that current alignment training has eliminated refusal as a decoding outcome. Scaling within the Qwen-VL family, both within- and across-generation, monotonically reduces Language Shortcuts but amplifies Visual Sycophancy, showing that scale and newer post-training alone cannot resolve the grounding problem. Diagnostic scores further enable a training-free selective-prediction strategy yielding up to +9.5 percentage points accuracy at 50% coverage.
>
---
#### [replaced 061] Identifiable Token Correspondence for World Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2605.16457](https://arxiv.org/pdf/2605.16457)**

> **作者:** Youngin Kim; Ray Sun; Inho Kim; Bumsoo Park; Hyun Oh Song
>
> **摘要:** Token-based transformer world models have shown strong performance in visual reinforcement learning, but often suffer from temporal inconsistency in long-horizon rollouts, including object duplication, disappearance, and transmutation. A key reason is that most existing approaches treat next-frame prediction purely as a token generation problem, without considering the persistence of tokens across time. We introduce Identifiable Token Correspondence (ITC), a decoding step for token-based transformer world models that formulates next-frame prediction as a structured assignment problem with latent token correspondence variables: each next-frame token is explained either by copying a token from the previous frame or by generating a new one. ITC leaves the transformer architecture and training procedure unchanged and can be added on top of existing backbones. Our experiments show state-of-the-art performance on 4 challenging benchmarks. The proposed method achieves a return of 72.5% and a score of 35.6% on the Craftax-classic benchmark, significantly surpassing the previous best of 67.4% and 27.9%. We release our source code on this https URL.
>
---
#### [replaced 062] DeepInterestGR: Mining Deep Multi-Interest Using Multi-Modal LLMs for Generative Recommendation
- **分类: cs.LG; cs.CV; cs.CY**

- **链接: [https://arxiv.org/pdf/2602.18907](https://arxiv.org/pdf/2602.18907)**

> **作者:** Yangchen Zeng; Zhenyu Yu; Zhiyuan Hu; Wenxin Zhang; Jinze Wang; Rongfeng Guo
>
> **摘要:** We introduce DeepInterestGR, a novel framework that integrates deep interest mining into the generative recommendation pipeline. This addresses the "Shallow Interest" problem - existing generative methods rely on surface-level textual features and fail to capture latent user motivations, limiting personalization depth and recommendation interpretability. Our approach leverages Multi-LLM Interest Mining (MLIM) via structured reasoning prompting, Reward-Labeled Deep Interest (RLDI) for quality control, and Interest-Enhanced Item Discretization (IEID) via RQ-VAE, combined with a two-stage SFT-GRPO training pipeline guided by an Interest-Aware Reward. We validate DeepInterestGR on three Amazon Review benchmarks (Beauty, Sports, Instruments), comparing against 14 state-of-the-art baselines including SASRec, BERT4Rec, TIGER, LC-Rec, and S-DPO. Our method achieves 5.8%-8.3% relative improvements on HR@10 and 7.7%-9.9% on NDCG@10 over the strongest baseline, with cross-domain generalization gains of +24.8%. These results provide evidence that incorporating deep semantic interests can effectively improve SID-based generative recommendation.
>
---
#### [replaced 063] "PhyWorldBench": A Comprehensive Evaluation of Physical Realism in Text-to-Video Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.13428](https://arxiv.org/pdf/2507.13428)**

> **作者:** Jing Gu; Xian Liu; Yu Zeng; Ashwin Nagarajan; Fangrui Zhu; Daniel Hong; Yue Fan; Qianqi Yan; Kaiwen Zhou; Ming-Yu Liu; Xin Eric Wang
>
> **备注:** 35 pages, 21 figures
>
> **摘要:** Video generation models have achieved remarkable progress in creating high-quality, photorealistic content. However, their ability to accurately simulate physical phenomena remains a critical and unresolved challenge. This paper presents PhyWorldBench, a comprehensive benchmark designed to evaluate video generation models based on their adherence to the laws of physics. The benchmark covers multiple levels of physical phenomena, ranging from fundamental principles such as object motion and energy conservation to more complex scenarios involving rigid body interactions and human or animal motion. Additionally, we introduce a novel Anti-Physics category, where prompts intentionally violate real-world physics, enabling the assessment of whether models can follow such instructions while maintaining logical consistency. Besides large-scale human evaluation, we also design a simple yet effective method that utilizes current multimodal large language models to evaluate physics realism in a zero-shot fashion. We evaluate 12 state-of-the-art text-to-video generation models, including five open-source and five proprietary models, with detailed comparison and analysis. Through systematic testing across 1050 curated prompts spanning fundamental, composite, and anti-physics scenarios, we identify pivotal challenges these models face in adhering to real-world physics. We further examine their performance under diverse physical phenomena and prompt types, and derive targeted recommendations for crafting prompts that enhance fidelity to physical principles.
>
---
#### [replaced 064] OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型在复杂文本图像推理任务的研究。针对现有基准不足，提出OCR-Reasoning基准，评估模型的文本图像推理能力，并发现当前模型表现不佳。**

- **链接: [https://arxiv.org/pdf/2505.17163](https://arxiv.org/pdf/2505.17163)**

> **作者:** Mingxin Huang; Yongxin Shi; Dezhi Peng; Songxuan Lai; Zecheng Xie; Lianwen Jin
>
> **备注:** ICLR 2026
>
> **摘要:** Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across various visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the absence of a dedicated and systematic benchmark. To address this gap, we propose OCR-Reasoning, a novel benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. Specifically, OCR-Reasoning comprises 1,069 human-annotated examples spanning 6 core reasoning abilities and 18 practical reasoning tasks in text-rich visual scenarios. Unlike existing text-rich image understanding benchmarks that only provide a final answer, this benchmark additionally provides a detailed step-by-step reasoning process. This dual annotation enables the evaluation of both the models' final answers and their reasoning processes, thereby offering a holistic assessment of text-rich reasoning capabilities. By leveraging this benchmark, we conducted a comprehensive evaluation of the latest MLLMs. Our results demonstrate that even the most advanced MLLMs exhibit substantial difficulties in text-rich image reasoning tasks, with none achieving an accuracy above 50\% on our benchmark, indicating that the challenges of text-rich image reasoning are an urgent issue to be addressed. The benchmark and evaluation scripts are available at this https URL.
>
---
#### [replaced 065] MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.05046](https://arxiv.org/pdf/2504.05046)**

> **作者:** Shenghao Ren; Yi Lu; Jiayi Huang; Jiayi Zhao; He Zhang; Tao Yu; Qiu Shen; Xun Cao
>
> **备注:** fix NSFC ID
>
> **摘要:** Existing human Motion Capture (MoCap) methods mostly focus on the visual similarity while neglecting the physical plausibility. As a result, downstream tasks such as driving virtual human in 3D scene or humanoid robots in real world suffer from issues such as timing drift and jitter, spatial problems like sliding and penetration, and poor global trajectory accuracy. In this paper, we revisit human MoCap from the perspective of interaction between human body and physical world by exploring the role of pressure. Firstly, we construct a large-scale human Motion capture dataset with Pressure, RGB and Optical sensors (named MotionPRO), which comprises 70 volunteers performing 400 types of motion, encompassing a total of 12.4M pose frames. Secondly, we examine both the necessity and effectiveness of the pressure signal through two challenging tasks: (1) pose and trajectory estimation based solely on pressure: We propose a network that incorporates a small kernel decoder and a long-short-term attention module, and proof that pressure could provide accurate global trajectory and plausible lower body pose. (2) pose and trajectory estimation by fusing pressure and RGB: We impose constraints on orthographic similarity along the camera axis and whole-body contact along the vertical axis to enhance the cross-attention strategy to fuse pressure and RGB feature maps. Experiments demonstrate that fusing pressure with RGB features not only significantly improves performance in terms of objective metrics, but also plausibly drives virtual humans (SMPL) in 3D scene. Furthermore, we demonstrate that incorporating physical perception enables humanoid robots to perform more precise and stable actions, which is highly beneficial for the development of embodied artificial intelligence. Project page is available at: this https URL
>
---
#### [replaced 066] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.17633](https://arxiv.org/pdf/2506.17633)**

> **作者:** Xiang Fang; Arvind Easwaran; Blaise Genest
>
> **备注:** Published in ICML 2025
>
> **摘要:** Out-of-distribution (OOD) detection attempts to distinguish outlier samples to prevent models trained on the in-distribution (ID) dataset from producing unavailable outputs. Most OOD detection methods require many IID samples for training, which seriously limits their real-world applications. To this end, we target a challenging setting: few-shot OOD detection, where {Only a few {\em labeled ID} samples are available.} Therefore, few-shot OOD detection is much more challenging than the traditional OOD detection setting. Previous few-shot OOD detection works ignore the distinct diversity between different classes. In this paper, we propose a novel network: Adaptive Multi-prompt Contrastive Network (AMCN), which adapts the ID-OOD separation boundary by learning inter- and intra-class distribution. To compensate for the absence of OOD and scarcity of ID {\em image samples}, we leverage CLIP, connecting text with images, engineering learnable ID and OOD {\em textual prompts}. Specifically, we first generate adaptive prompts (learnable ID prompts, label-fixed OOD prompts and label-adaptive OOD prompts). Then, we generate an adaptive class boundary for each class by introducing a class-wise threshold. Finally, we propose a prompt-guided ID-OOD separation module to control the margin between ID and OOD prompts. Experimental results show that AMCN outperforms other state-of-the-art works.
>
---
#### [replaced 067] LiM-YOLO: Less is More with Pyramid Level Shift for Ship Detection in Optical Remote Sensing
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.09700](https://arxiv.org/pdf/2512.09700)**

> **作者:** Seon-Hoon Kim; Yerin Kim; Hyeji Sim; Youeyun Jung; Okchul Jung; Daewon Chung
>
> **备注:** 16 pages, 6 figures, 9 tables
>
> **摘要:** General-purpose object detectors face fundamental structural limitations when applied to ship detection in satellite imagery, where the ship scale distribution is concentrated at small sizes and high aspect ratios. In conventional You Only Look Once architectures, the deepest feature pyramid level (stride 32) compresses narrow vessels into sub-pixel representations, causing severe spatial feature dilution and compromising accurate ship boundary regression. We propose Less is More YOLO, a streamlined detector built upon the extra-large variant of YOLOv9, to address these domain-specific structural conflicts. From a statistical analysis of ship scale distributions across four major benchmarks (SODA-A, DOTA-v1.5, FAIR1M-v2.0, and ShipRSImageNet), we introduce a Pyramid Level Shift Strategy that shifts the detection head from strides 8, 16, and 32 to strides 4, 8, and 16. This shift satisfies a spatial representability condition derived from the Nyquist-Shannon principle for the narrowest targets, while eliminating the computational redundancy of the deepest pyramid level. To further stabilize training on high-resolution satellite inputs, we incorporate a group-normalized auxiliary projection module that introduces Group Normalization into the projection path, mitigating gradient instability in memory-constrained micro-batch regimes. Validated on these four datasets, our detector attains an mAP_{50-95} of 0.600 with only 21.16 million parameters, a 64.1% reduction from the extra-large YOLOv9 baseline (58.99 million). Despite this compact size, our model surpasses state-of-the-art detectors up to three times larger, validating that a well-targeted pyramid level shift achieves a "Less is More" balance between accuracy and efficiency. The code is available at this https URL.
>
---
#### [replaced 068] Adapting Actively on the Fly: Relevance-Guided Online Meta-Learning with Latent Concepts for Geospatial Discovery
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.17605](https://arxiv.org/pdf/2602.17605)**

> **作者:** Jowaria Khan; Anindya Sarkar; Yevgeniy Vorobeychik; Elizabeth Bondi-Kelly
>
> **摘要:** In environmental monitoring, data collection is often costly, sparse, and shaped by urgent public-health needs. This is particularly true for cancer-causing PFAS (Per- and polyfluoroalkyl substances) contamination, where discussions with domain experts and environmental organizations highlight the need to strategically identify high-risk, under-observed regions under tight sampling budgets. More broadly, similar challenges arise in disaster response and public health settings, where dynamic environments make it essential to efficiently uncover hidden targets from limited ground truth. Yet sparse and biased geospatial labels limit the applicability of existing learning-based methods, such as reinforcement learning. To address this, we propose a unified geospatial discovery framework that integrates active learning, online meta-learning, and concept-guided reasoning. Our approach introduces two key innovations built on a shared notion of *concept relevance*, capturing how domain-specific factors influence target presence: a *concept-weighted uncertainty sampling strategy*, where uncertainty is modulated by learned relevance from readily available concepts such as land cover and source proximity; and a *relevance-aware meta-batch formation strategy* that promotes semantic diversity during online-meta updates, improving generalization in dynamic environments. We evaluate our framework on PFAS contamination discovery as a real-world inspired environmental monitoring task, demonstrating robust target discovery under limited data and changing conditions.
>
---
#### [replaced 069] Resolving Ambiguity in Composed Image Retrieval via Calibrated Interaction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.24634](https://arxiv.org/pdf/2605.24634)**

> **作者:** Amsisan Tran; Baogh Le; Tuan Kiet Pham; Sui Yang Guang
>
> **摘要:** Composed image retrieval (CIR) searches a corpus with a reference image and a text describing how to modify it. Despite rapid progress from triplet-trained compositors to zero-shot and generative methods, essentially all systems share one assumption: that a query maps to a single target, scored by Recall@K against one annotation. We argue this is fundamentally at odds with the task. A query such as make it more formal does not name an image but a region of the corpus, and which member the user intends is genuinely underdetermined. This underspecification is the root of the well-known false-negative problem and leaves current models unable to tell a precise query from an ambiguous one. We reframe CIR as calibrated intent resolution under uncertainty: a retriever is wrapped in a conformal prediction layer that returns a candidate set with a coverage guarantee and whose size is a principled measure of ambiguity; when the set is large, an expected-information-gain policy asks the single most useful clarifying question, drawn from interpretable ambiguity axes, and the set contracts. We introduce AmbiCIR, a benchmark and human-validated user simulator that revive the dormant auxiliary and dialogue annotations of CIRR and extend the multiple-positive setting of CIRCO. Across open-domain and fashion benchmarks our method matches single-turn state of the art, confirming calibrated resolution is cost-free on precise queries, while reaching the intended target in a fraction of the interaction budget required by naive conversational baselines, and it is the first to report valid coverage and calibration for the task.
>
---
#### [replaced 070] Olaf-World: Orienting Latent Actions for Video World Modeling
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.10104](https://arxiv.org/pdf/2602.10104)**

> **作者:** Yuxin Jiang; Yuchao Gu; Ivor W. Tsang; Mike Zheng Shou
>
> **备注:** ICML 2026. Project page: this https URL Code: this https URL
>
> **摘要:** Scaling action-controllable world models is limited by the scarcity of action labels. While latent action learning promises to extract control interfaces from unlabeled video, learned latents often fail to transfer across contexts: they entangle scene-specific cues and lack a shared coordinate system. This occurs because standard objectives operate only within each clip, providing no mechanism to align action semantics across contexts. Our key insight is that although actions are unobserved, their semantic effects are observable and can serve as a shared reference. We introduce Seq$\Delta$-REPA, a sequence-level control-effect alignment objective that anchors integrated latent action to temporal feature differences from a frozen, self-supervised video encoder. Building on this, we present Olaf-World, a pipeline that pretrains action-conditioned video world models from large-scale passive video. Extensive experiments demonstrate that our method learns a more structured latent action space, leading to stronger zero-shot action transfer and more data-efficient adaptation to new control interfaces than state-of-the-art baselines.
>
---
#### [replaced 071] Beyond Text Prompts: Visual-to-Visual Generation as A Unified Paradigm
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.12271](https://arxiv.org/pdf/2605.12271)**

> **作者:** Yaofang Liu; Kangning Cui; Meng Chu; Zhaoqing Li; Suiyun Zhang; Jean-Michel Morel; Xiaodong Cun; Haoxuan Che; Rui Liu; Raymond H. Chan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Humans often specify and create through visual artifacts: typography sheets, sketches, reference images, and annotated scenes. Yet modern visual generators still ask users to serialize this intent into text, a bottleneck that compresses signals like spatial structure, exact appearance, and glyph shape. We propose \textbf{\emph{visual-to-visual} (V2V)} generation, in which the user conditions a generative model with a visual specification page rather than a text prompt. The page is not an edit target, but a visual document that specifies the desired output. We introduce \textbf{V2V-Zero}, a training-free framework that exposes this interface in existing vision-language model (VLM) conditioned generators by replacing text-only conditioning with final-layer hidden states extracted from visual pages, exploiting the fact that the frozen VLM already maps both text and images into the generator's conditioning space. On GenEval, V2V-Zero reaches 0.85 with a frozen Qwen-Image backbone, closely matching its optimized text-to-image performance without fine-tuning. To evaluate the broader V2V space, we introduce \textbf{Simple-V2V Bench}, spanning seven visual-conditioning tasks and seven models, including GPT Image 2, Nano Banana 2, Seedream 5.0 Lite, open-weight baselines, and a video extension. V2V-Zero scores 32.7/100, outperforming evaluated open-weight image baselines and revealing a clear capability hierarchy: attribute binding is strong, content generation is unreliable, and structural control remains hard even for commercial systems. A HunyuanVideo-1.5 extension scores 20.2/100, showing the interface transfers beyond images. Mechanistic analysis shows the default reasoning path is primarily visually routed, with 95.0\% of conditioning-token attention mass on visual-page hidden states.
>
---
#### [replaced 072] The Neglected Baseline in Model Interpretation
- **分类: cs.CV; cs.SE**

- **链接: [https://arxiv.org/pdf/2605.22417](https://arxiv.org/pdf/2605.22417)**

> **作者:** Yongjin Cui; Xiaohui Fan
>
> **摘要:** We observe that existing model interpretation methods generally ignore the baseline, and such neglect often results in imprecise or even incorrect interpretation. In this paper, we reformulate the task of model interpretation and the interpretation principles for model interpretation results to demonstrate the importance of the baseline. We further unify gradient-based methods, Integrated Gradients (IG) methods, and Taylor expansion, clarifying the connections among them and explicitly identifying the baseline for each method. On this basis, we analyze the flaws and errors in related model interpretation methods (IG, LayerCAM, ODAM, Difference Map). We advocate evaluating the quality of model interpretation results precisely through the attribution error between the attribution result and the attribution target, rather than adopting flawed evaluation methods, such as those based on marginal-effect or the assumption of perfect model performance. We revise IG and develope a model interpretation method with a clear and reasonable baseline, achieving better results. Our method supports model interpretation based on features from any layer. Interpretation based on features from different layers are all reasonable, and the differences among these results reflect varying degrees of feature extraction at different feature extraction stages.
>
---
#### [replaced 073] GFSR: Geometric Fidelity and Spatial Refinement for Reliable Lane Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.23327](https://arxiv.org/pdf/2605.23327)**

> **作者:** Tiancheng Wang; Zhaolu Ding; Richeng Xu; Tianhui Zheng; Hui Liu; Hanyu Xuan; Zhiliang Wu; Guanghui Yue
>
> **备注:** Submitted to IEEE Transactions on Intelligent Transportation Systems. 12 pages, 6 figures
>
> **摘要:** Lane detection stands as a crucial perception task in autonomous driving and advanced driver assistance systems. However, existing methods still degrade in complex real scenarios due to two major limitations. First, classification confidence only characterizes the categorical existence of lane priors and has no strong correlation with geometric quality. If threshold filtering and NMS are conducted merely based on this confidence, the model tends to retain lane priors with high confidence while eliminating those with lower confidence but superior geometric representation. Secondly, the regression modules in existing methods weaken correlations among sampling points, hindering fine-grained optimization of distant, high-curvature and complex-topology lanes and causing underfitting. To address these issues, we propose Geometric Fidelity and Spatial Refinement (GFSR), a framework consisting of LaneIoU-guided Confidence Calibration (LCC) and Adaptive Gated Location Refinement (AGLR). Specifically, LCC adopts LaneIoU as soft supervision to explicitly estimate the geometric fidelity of lane priors, which is further fused with classification confidence to construct the Collaborative Reliability Index (CRI). This index guides lane prior filtering, effectively retaining those with high classification confidence and favorable geometric quality. Meanwhile, cooperating with regression heads in each refinement stage, AGLR predicts sampling point lateral offsets and adopts a gating mechanism to adaptively regulate correction magnitude, strengthen inter-point correlations and boost model adaptability as well as robustness toward complex lane scenarios. Extensive experiments on CULane and CurveLanes demonstrate that our GFSR achieves state-of-the-art performance on CULane, with F1_50 and F1_75 scores of 81.46% and 65.01%, and reaches 87.35% F1_50 on CurveLanes.
>
---
#### [replaced 074] Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23606](https://arxiv.org/pdf/2505.23606)**

> **作者:** Qingyu Shi; Jinbin Bai; Zhuoran Zhao; Wenhao Chai; Kaidong Yu; Jianzong Wu; Yunhai Tong; Xiangtai Li; Xuelong Li; Shuicheng Yan
>
> **备注:** Accepted to ICLR 2026. Codes and Supplementary Material: this https URL
>
> **摘要:** Unified generation models aim to handle diverse tasks across modalities -- such as text generation, image generation, and vision-language reasoning -- within a single architecture and decoding paradigm. Autoregressive unified models suffer from slow inference due to sequential decoding, and non-autoregressive unified models suffer from weak generalization due to limited pretrained backbones. We introduce the second-generation Meissonic: Muddit, a unified discrete diffusion transformer that enables fast and parallel generation across both text and image modalities. Unlike prior unified diffusion models trained from scratch, Muddit integrates strong visual priors from a pretrained text-to-image backbone with a lightweight text decoder, enabling flexible and high-quality multimodal generation under a unified architecture. Empirical results show that Muddit achieves competitive or superior performance compared to significantly larger autoregressive models in both quality and efficiency. The work highlights the potential of purely discrete diffusion, when equipped with strong visual priors, as a scalable and effective backbone for unified generation.
>
---
#### [replaced 075] FiRe: Fine-grained Multimodal Reasoning for Enhanced Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.13491](https://arxiv.org/pdf/2604.13491)**

> **作者:** Yongjin Kim; Yoonjin Oh; Yerin Kim; Hyomin Kim; Jeeyoung Yun; Yujung Heo; Minjun Kim; Sungwoong Kim
>
> **摘要:** With the rapid progress of Multimodal Large Language Models (MLLMs), unified MLLMs that jointly perform image understanding and generation have advanced significantly. However, despite the inherent reasoning capabilities of unified MLLMs for self-reflection and self-refinement, their use in text-to-image generation remains largely underexplored. Meanwhile, existing multimodal reasoning-based image generation methods mostly rely on prompt augmentation or holistic image-text alignment judgments, without fine-grained reflection and refinement of detailed prompt attributes, leading to limited fine-grained control. To address this limitation, we propose FiRe, a Fine-grained Multimodal Reasoning method for enhanced image generation by MLLM. In specific, FiRe performs a fine-grained multi-step reasoning by first decomposing the prompt into key visual requirements and then self-judging their satisfaction in the generated image, followed by localized refinement according to self-generated precise feedback. In addition, to further strengthen the MLLM's multimodal reasoning ability, we introduce FiRe-GRPO, a reinforcement learning method tailored to FiRe. Since standard Group Relative Policy Optimization (GRPO) suffers from sparse, outcome-based rewards in multi-step reasoning, we formulate our reasoning process as a step-level decision-making problem, design step-specific rewards, and compute step-level advantages for granular credit assignment within GRPO. Extensive experiments demonstrate that FiRe consistently outperforms competitive text-to-image baselines, including existing reasoning-based methods, with particularly substantial gains on compositional text-to-image benchmarks.
>
---
#### [replaced 076] PDEInvBench: A Comprehensive Dataset and Design Space Exploration of Neural Networks for PDE Inverse Problems
- **分类: cs.LG; cs.CV; physics.comp-ph**

- **链接: [https://arxiv.org/pdf/2605.25353](https://arxiv.org/pdf/2605.25353)**

> **作者:** Divyam Goel; Nithin Chalapathi; Sanjeev Raja; Aditi S. Krishnapriyan
>
> **备注:** 37 total pages, 13 main pages, 20 figures, 8 tables. Published in Transactions on Machine Learning Research (TMLR), 2026
>
> **摘要:** Inverse problems in partial differential equations (PDEs) involve estimating the physical parameters of a system from observed spatiotemporal solution fields. Neural networks are well-suited for PDE parameter estimation due to their capability to model function-to-function space transformations. While existing benchmarks of machine learning methods for PDEs primarily focus on the forward problem, there are no similar comprehensive studies and benchmark datasets on PDE inverse problems, i.e., mapping solution fields to underlying physical parameters. We fill this gap by introducing PDEInvBench, a comprehensive benchmark dataset consisting of numerical simulations for both time-dependent and time-independent PDEs across a wide range of physical behaviors and parameters. Our dataset includes evaluation splits that assess performance in both in-distribution and various out-of-distribution settings. Using our benchmark dataset, we comprehensively explore the design space of neural networks for PDE inverse problems along three key dimensions: (1) optimization procedures, analyzing the role of supervised, self-supervised, and test-time training objectives on performance, (2) problem representations, where we study the value of architectural choices with different inductive biases and various conditioning strategies, and (3) scaling, which we perform with respect to both model and data size. Our experiments reveal several practical insights: 1) neural networks perform best with a two-stage training procedure: initial supervision with PDE parameters followed by test-time fine-tuning using the PDE residual, 2) incorporating PDE derivatives as input features consistently improves accuracy, and 3) increasing the diversity of initial conditions in the training data yields greater performance gains than expanding the range of PDE parameters. We make our dataset and codebase publicly available.
>
---
#### [replaced 077] Drive-P2D: A Progressive Perception-to-Decision Benchmark for VLMs in Autonomous Driving
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出Drive-P2D基准，解决VLMs在自动驾驶中感知与决策联合评估的问题，通过多层级测试和错误分析提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.14702](https://arxiv.org/pdf/2601.14702)**

> **作者:** Zecong Tang; Zixu Wang; Yifei Wang; Weitong Lian; Tianjian Gao; Haoran Li; Tengju Ru; Lingyi Meng; Zhejun Cui; Yichen Zhu; Qi Kang; Kaixuan Wang; Yu Zhang
>
> **摘要:** Autonomous driving requires reliable perception and safe decision-making in complex scenarios. Recent vision-language models (VLMs) demonstrate reasoning and generalization abilities, opening new possibilities for autonomous driving; however, existing benchmarks often evaluate perception and decision-making separately, limit failure analysis with choice-only formats, or introduce evaluation bias through LLM-scored long-form outputs. To address these issues, we present Drive-P2D, a progressive perception-to-decision benchmark with 6,650 questions across Object, Scene, and Decision levels. Drive-P2D adopts a separated reasoning-and-answer protocol: final answers are scored objectively, while reasoning is analyzed to identify error modes exposed along the progressive perception-to-decision chain. We evaluate mainstream VLMs across all and high-risk scenarios, and further characterize the perception-to-decision capability boundary through correlation analysis and similar-scene robustness testing. Reasoning further exposes failure modes such as logical reasoning errors and semantic feature omissions, and we train a lightweight analyzer model to automate large-scale error-mode annotation of reasoning. Together, these designs provide practical insights for building safer and more reliable VLMs for real-world autonomous driving.
>
---
#### [replaced 078] Explainable Cross-Disease Reasoning for Cardiovascular Risk Assessment from Low-Dose Computed Tomography
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.06625](https://arxiv.org/pdf/2511.06625)**

> **作者:** Yifei Zhang; Jiashuo Zhang; Mojtaba Safari; Xiaofeng Yang; Liang Zhao
>
> **摘要:** Low-dose chest computed tomography (LDCT) captures pulmonary and cardiac structures in a single scan, enabling joint assessment of lung and cardiovascular health. Existing approaches typically model these domains independently and do not explicitly represent their physiological interactions. We propose an Explainable Cross-Disease Reasoning Framework for cardiovascular risk assessment from LDCT. The framework follows a constrained clinical-information pathway: it extracts pulmonary findings, grounds cross-organ mechanisms in medical knowledge, and produces a cardiovascular prediction with a natural-language rationale. It combines four components: a frozen lung-risk prior, a pulmonary perception module, an agentic reasoning module, and a cardiac subvolume feature extractor. Their outputs are fused to integrate localized cardiac evidence with mechanism-level pulmonary context. On the National Lung Screening Trial cohort, the framework achieves an AUC of 0.919 for CVD screening and up to 0.838 for CVD mortality prediction, outperforming cardiac-specific, single-disease, and foundation-model baselines. Targeted controls indicate that the gains are not explained by additional thoracic visual features alone, fixed rule propagation, or a single reasoning backend. The proposed framework thus provides an auditable approach to cross-disease cardiovascular risk assessment from LDCT.
>
---
#### [replaced 079] MultiSense-Pneumo: A Multimodal Learning Framework for Pneumonia Screening in Resource-Constrained Settings
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.02207](https://arxiv.org/pdf/2605.02207)**

> **作者:** Dineth Jayakody; Pasindu Thenahandi; Chameli Dommanige
>
> **摘要:** Pneumonia remains a leading global cause of morbidity and mortality, particularly in low-resource settings where access to imaging, laboratory testing, and specialist care is limited. Clinical assessment relies on heterogeneous evidence, including symptoms, respiratory patterns, spoken descriptions, and chest imaging, making frontline screening inherently multimodal. However, many existing computational approaches remain unimodal and focus primarily on radiographs. In this work, we present MultiSense-Pneumo, a multimodal research prototype for pneumonia-oriented screening and triage support that integrates structured symptom descriptors, cough audio, spoken language, and chest radiographs. The system combines deterministic symptom triage, LightGBM-based acoustic classification, domain-adversarial radiograph analysis using ResNet-18, transformer-based speech recognition, and an interpretable late-fusion operator. Each modality is transformed into a normalized concern signal and aggregated into a unified screening estimate. The fusion weights are hand-specified and are treated as heuristic, interpretable parameters rather than learned or clinically optimized values. MultiSense-Pneumo is implemented with offline execution in mind on standard laptop-class hardware, but it is not presented as a deployment-validated or clinically validated diagnostic system. Experimental results demonstrate strong component-level performance of the radiograph pathway under synthetic domain shifts, while also highlighting important limitations, especially reduced abnormal-class recall for cough acoustics and the absence of paired end-to-end multimodal patient evaluation. MultiSense-Pneumo is therefore intended as a framework and component-level prototype for screening and triage research.
>
---
#### [replaced 080] Where Detectors Fail: Probing Generative Space for Generalizable AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.24906](https://arxiv.org/pdf/2605.24906)**

> **作者:** Zijie Cao; Weijie Tu; Yao Xiao; Weijian Deng; Liang Lin; Pengxu Wei
>
> **摘要:** Detecting AI-generated images (AIGI) remains challenging because detectors often fail to generalize to unseen generators. Although existing methods are trained on large datasets, their performance still degrades when generation settings change, indicating that data scale alone is insufficient and that limited coverage of generative variations during training is a key factor. Studies on generative model editing show that small changes in internal representations can produce diverse and meaningful image variations, many of which are not explored under standard sampling. Leveraging this insight, we propose PROBE (Probing Robustness via Boundary Exploration), a framework that improves detector generalization by actively exploring challenging regions of the generative process. Instead of treating the generator as a fixed data source, PROBE uses the detector as a critic to steer the generator through manifold-level modifications, producing realistic samples that are difficult to classify. These samples expose failure cases that are uncommon under standard data sampling strategies and are used to refine the detector. Experimental results across multiple benchmarks indicate that PROBE enhances generalization to unseen generators, resulting in more generalizable AIGI detection performance. Code and models are available at this https URL
>
---
#### [replaced 081] Demystifying Video Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.16870](https://arxiv.org/pdf/2603.16870)**

> **作者:** Ruisi Wang; Zhongang Cai; Fanyi Pu; Junxiang Xu; Wanqi Yin; Maijunxian Wang; Ran Ji; Chenyang Gu; Bo Li; Ziqi Huang; Hokin Deng; Dahua Lin; Ziwei Liu; Lei Yang
>
> **备注:** Homepage: this https URL
>
> **摘要:** Recent advances in video generation have revealed an unexpected phenomenon: diffusion-based video models exhibit non-trivial reasoning capabilities. Prior work attributes this to a Chain-of-Frames (CoF) mechanism, where reasoning is assumed to unfold sequentially across video frames. In this work, we challenge this assumption and uncover a fundamentally different mechanism. We show that reasoning in video models instead primarily emerges along the diffusion denoising steps. Through qualitative analysis and targeted probing experiments, we find that models explore multiple candidate solutions in early denoising steps and progressively converge to a final answer, a process we term Chain-of-Steps (CoS). Beyond this core mechanism, we identify several emergent reasoning behaviors critical to model performance: (1) working memory, enabling persistent reference; (2) self-correction and enhancement, allowing recovery from incorrect intermediate solutions; and (3) perception before action, where early steps establish semantic grounding and later steps perform structured manipulation. During a diffusion step, we further uncover self-evolved functional specialization within Diffusion Transformers, where early layers encode dense perceptual structure, middle layers execute reasoning, and later layers consolidate latent representations. Motivated by these insights, we present a simple training-free strategy as a proof-of-concept, demonstrating how reasoning can be improved by ensembling latent trajectories from identical models with different random seeds. Overall, our work provides a systematic understanding of how reasoning emerges in video generation models, offering a foundation to guide future research in better exploiting the inherent reasoning dynamics of video models as a new substrate for intelligence.
>
---
#### [replaced 082] SenBen: Sensitive Scene Graphs for Explainable Content Moderation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2604.08819](https://arxiv.org/pdf/2604.08819)**

> **作者:** Fatih Cagatay Akyon; Alptekin Temizel
>
> **备注:** Accepted at CVPRW 2026
>
> **摘要:** Content moderation systems classify images as safe or unsafe but lack spatial grounding and interpretability: they cannot explain what sensitive behavior was detected, who is involved, or where it occurs. We introduce the Sensitive Benchmark (SenBen), the first large-scale scene graph benchmark for sensitive content, comprising 13,999 frames from 157 movies annotated with Visual Genome-style scene graphs (25 object classes, 28 attributes including affective states such as pain, fear, aggression, and distress, 14 predicates) and 16 sensitivity tags across 5 categories. We distill a frontier VLM into a compact 241M student model using a multi-task recipe that addresses vocabulary imbalance in autoregressive scene graph generation through suffix-based object identity, Vocabulary-Aware Recall (VAR) Loss, and a decoupled Query2Label tag head with asymmetric loss, yielding a +6.4 percentage point improvement in SenBen Recall over standard cross-entropy training. On grounded scene graph metrics, our student model outperforms all evaluated VLMs except Gemini models and all commercial safety APIs, while achieving the highest object detection and captioning scores across all models, at $7.6\times$ faster inference and $16\times$ less GPU memory.
>
---
#### [replaced 083] TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.02775](https://arxiv.org/pdf/2504.02775)**

> **作者:** Yoon Gyo Jung; Jaewoo Park; Jaeho Yoon; Kuan-Chuan Peng; Wonchul Kim; Andrew Beng Jin Teoh; Octavia Camps
>
> **备注:** Accepted to CVPR2025
>
> **摘要:** We aim to solve unsupervised anomaly detection in a practical challenging environment where the normal dataset is both contaminated with defective regions and its product class distribution is tailed but unknown. We observe that existing models suffer from tail-versus-noise trade-off where if a model is robust against pixel noise, then its performance deteriorates on tail class samples, and vice versa. To mitigate the issue, we handle the tail class and noise samples independently. To this end, we propose TailSampler, a novel class size predictor that estimates the class cardinality of samples based on a symmetric assumption on the class-wise distribution of embedding similarities. TailSampler can be utilized to sample the tail class samples exclusively, allowing to handle them separately. Based on these facets, we build a memory-based anomaly detection model TailedCore, whose memory both well captures tail class information and is noise-robust. We extensively validate the effectiveness of TailedCore on the unsupervised long-tail noisy anomaly detection setting, and show that TailedCore outperforms the state-of-the-art in most settings.
>
---
#### [replaced 084] An uncertainty-aware Bayesian framework for machine learning classification models: A case study in land cover classification
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2503.21510](https://arxiv.org/pdf/2503.21510)**

> **作者:** Samuel Bilson; Miles McCrory; Anna Pustogvar
>
> **备注:** 38 pages, 16 figures
>
> **摘要:** Ensuring that predictions of machine learning (ML) classification models are accompanied by uncertainty estimates is one of the main pillars of trustworthy AI. Current research in uncertainty quantification focuses mainly on epistemic uncertainty of the ML model, but rarely takes account of input measurement uncertainty, which is vital for traceability in metrology. In this work we propose a Bayesian framework for generative ML classification models that takes account of input measurement uncertainty. We take the specific case of a Bayesian quadratic discriminant analysis (BQDA) model, and apply it to metrological land cover datasets from Copernicus Sentinel-2 from 2020 and 2021. We benchmark the performance of the model against more popular classification models used in land cover maps such as random forests and neural networks. To validate and assess the generalisability of such a model, we also run simulations over synthetic classification data, varying distribution type and strength of the input measurement noise. We find for both real and synthetic data, the BQDA model presented is more trustworthy, in the sense that it is more interpretable, explicitly models the input measurement uncertainty, and maintains predictive performance of class probability outputs across datasets over different domains and sizes, whilst also being more computationally efficient.
>
---
#### [replaced 085] D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.05204](https://arxiv.org/pdf/2605.05204)**

> **作者:** Dengyang Jiang; Xin Jin; Dongyang Liu; Zanyi Wang; Mingzhe Zheng; Ruoyi Du; Xiangpeng Yang; Qilong Wu; Zhen Li; Peng Gao; Harry Yang; Steven Hoi
>
> **备注:** Project Page: this https URL
>
> **摘要:** The landscape of high-performance image generation models is currently shifting from the inefficient multi-step ones to the efficient few-step counterparts (e.g, Z-Image-Turbo and FLUX.2-klein). However, these models present significant challenges for direct continuous supervised fine-tuning. For example, applying the commonly used fine-tuning technique would compromise their inherent few-step inference capability. To address this, we propose D-OPSD, a novel training paradigm for step-distilled diffusion models that enables on-policy learning during supervised fine-tuning. We first find that the modern diffusion models, where the LLM/VLM serves as the encoder, can inherit its encoder's in-context capabilities. This enables us to formulate the training as an on-policy self-distillation process. Specifically, during training, we make the model act as both the teacher and the student with different contexts, where the student is conditioned only on the text feature, while the teacher is conditioned on the multimodal feature of both the text prompt and the target image. Training minimizes the two predicted distributions over the student's own roll-outs. By optimizing on the model's own trajectory and under its own supervision, D-OPSD enables the model to learn new concepts, styles, etc., without sacrificing the original few-step capacity.
>
---
#### [replaced 086] ODOV: Benchmark the Open-Domain Open-Vocabulary Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01253](https://arxiv.org/pdf/2508.01253)**

> **作者:** Yupeng Zhang; Ruize Han; Fangnan Zhou; Wei Feng; Liang Wan
>
> **摘要:** Existing studies typically investigate domain shift and category shift as independent problems, however, in real-world scenarios, the two types of shifts often occur simultaneously and interact, leading to significant degradation in detection performance. To address this, we propose and systematically study a novel problem-Open-Domain Open-Vocabulary (ODOV) object detection-which aims to evaluate a model's ability to adapt to the compound domain and category shifts in real-world this http URL construct a new benchmark, OD-LVIS, which contains 46,949 images spanning 15 diverse real-world scenarios and 1,203 categories, for assessing object detection performance. Furthermore, we propose a novel ODOV detection baseline that fully leverages VLM's powerful multi-modal alignment capabilities and introduces two key mechanisms to enhance both category and domain generalization. One is the Domain-Agnostic Category Prompt (DAPmt), which strengthens category semantics while attenuating domain representations, enabling pure category representation. The other is the Domain Projection and Grafting (DP&G) module, which incorporates domain-specific features from input images, allowing the model to dynamically generalize across diverse open domains. These two components enable the model to maintain effective detection performance under simultaneous category and domain variations in real-world scenarios. We provide extensive benchmark evaluations for the proposed ODOV detection task and report experimental results. These results validate the soundness of the ODOV task, the practicality of the OD-LVIS dataset, and the superiority of the method.
>
---
#### [replaced 087] BrainDINO: A Brain MRI Foundation Model for Generalizable Clinical Representation Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.27277](https://arxiv.org/pdf/2604.27277)**

> **作者:** Yizhou Wu; Shansong Wang; Yuheng Li; Mojtaba Safari; Mingzhe Hu; Chih-Wei Chang; Harini Veeraraghavan; Xiaofeng Yang
>
> **备注:** 22 pages, 5 figures
>
> **摘要:** Brain MRI underpins a wide range of neuroscientific and clinical applications, yet most learning-based methods remain task-specific and require substantial labeled data. Here we show that a single self-supervised representation can generalize across heterogeneous brain MRI endpoints. We trained BrainDINO, a self-distilled foundation model, on approximately 6.6 million unlabeled axial slices from 20 datasets encompassing broad variation in population, disease, and acquisition setting. Using a frozen encoder with lightweight task heads, BrainDINO supported transfer across tumor segmentation, neurodegenerative and neurodevelopmental conditions classification, brain age estimation, post-stroke temporal prediction, molecular status prediction, MRI sequence classification, and survival modeling. Across tasks and supervision regimes, BrainDINO consistently equaled or exceeded natural-image and MRI-specific self-supervised baselines, with particularly strong advantages under label scarcity. Representation analyses further showed anatomically organized and pathology-sensitive feature structure in the absence of task-specific supervision. Our findings indicate that large-scale slice-wise self-supervised learning can yield a unified brain MRI representation that supports diverse neuroimaging tasks without volumetric pretraining or full-network fine-tuning, establishing a scalable foundation for robust and data-efficient brain imaging analysis. Code is available at this https URL
>
---
#### [replaced 088] Degradation-Consistent Paired Training for Robust AI-Generated Image Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.10102](https://arxiv.org/pdf/2604.10102)**

> **作者:** Zongyou Yang; Yinghan Hou; Xiaokun Yang
>
> **备注:** 6 pages, 5 figures, 2 tables
>
> **摘要:** AI-generated image detectors suffer significant performance degradation under real-world image corruptions such as JPEG compression, Gaussian blur, and resolution downsampling. We observe that state-of-the-art methods, including B-Free, treat degradation robustness as a byproduct of data augmentation rather than an explicit training objective. In this work, we propose Degradation-Consistent Paired Training (DCPT), a simple yet effective training strategy that explicitly enforces robustness through paired consistency constraints. For each training image, we construct a clean view and a degraded view, then impose two constraints: a feature consistency loss that minimizes the cosine distance between clean and degraded representations, and a prediction consistency loss based on symmetric KL divergence that aligns output distributions across views. DCPT adds zero additional parameters and zero inference overhead. Experiments on the Synthbuster benchmark (9 generators, 8 degradation conditions) demonstrate that DCPT improves the degraded-condition average accuracy by 9.1 percentage points compared to an identical baseline without paired training, while sacrificing only 0.9% clean accuracy. The improvement is most pronounced under JPEG compression (+15.7% to +17.9%). Ablation further reveals that adding architectural components leads to overfitting on limited training data, confirming that training objective improvement is more effective than architectural augmentation for degradation robustness.
>
---
#### [replaced 089] Scalable GANs with Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.24935](https://arxiv.org/pdf/2509.24935)**

> **作者:** Sangeek Hyun; MinKyu Lee; Jae-Pil Heo
>
> **备注:** ICML 2026
>
> **摘要:** Scalability has driven recent advances in generative modeling, yet its principles remain underexplored for adversarial learning. We investigate the scalability of Generative Adversarial Networks (GANs) through two design choices that have proven to be effective in other types of generative models: training in a compact Variational Autoencoder latent space and adopting purely transformer-based generators and discriminators. Training in latent space enables efficient computation while preserving perceptual fidelity, and this efficiency pairs naturally with plain transformers, whose performance scales with computational budget. Building on these choices, we analyze failure modes that emerge when naively scaling GANs. Specifically, we find issues as underutilization of early layers in the generator and optimization instability as the network scales. Accordingly, we provide simple and scale-friendly solutions as lightweight intermediate supervision and width-aware learning-rate adjustment. Our experiments show that GAT, a purely transformer-based and latent-space GANs, can be easily trained reliably across a wide range of capacities (S through XL). Moreover, GAT-XL/2 achieves state-of-the-art single-step, class-conditional generation performance (FID of 2.96) on ImageNet-256 in just 40 epochs, 6x fewer epochs than strong baselines. Project page: this https URL.
>
---
#### [replaced 090] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19206](https://arxiv.org/pdf/2602.19206)**

> **作者:** Zehao Deng; An Liu; Yan Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Zero-shot 3D Anomaly Detection is an emerging task that aims to detect anomalies in a target dataset without any target training data, which is particularly important in scenarios constrained by sample scarcity and data privacy concerns. While current methods adapt CLIP by projecting 3D point clouds into 2D representations, they face challenges. The projection inherently loses some geometric details, and the reliance on a single 2D modality provides an incomplete visual understanding, limiting their ability to detect diverse anomaly types. To address these limitations, we propose the Geometry-Aware Prompt and Synergistic View Representation Learning (GS-CLIP) framework, which enables the model to identify geometric anomalies through a two-stage learning process. In stage 1, we dynamically generate text prompts embedded with 3D geometric priors. These prompts contain global shape context and local defect information distilled by our Geometric Defect Distillation Module (GDDM). In stage 2, we introduce Synergistic View Representation Learning architecture that processes rendered and depth images in parallel. A Synergistic Refinement Module (SRM) subsequently fuses the features of both streams, capitalizing on their complementary strengths. Comprehensive experimental results on four large-scale public datasets show that GS-CLIP achieves superior performance in detection. Code can be available at this https URL.
>
---
#### [replaced 091] No Data? No Problem: Robust Vision-Tabular Learning with Missing Values
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19602](https://arxiv.org/pdf/2512.19602)**

> **作者:** Marta Hasny; Laura Daza; Keno Bressem; Maxime Di Folco; Julia Schnabel
>
> **摘要:** Large-scale medical biobanks provide imaging data complemented by extensive tabular information, such as clinical measurements or demographics. However, this abundance of tabular attributes does not reflect real-world datasets, where only a subset of attributes may be available. This discrepancy calls for methods that remain robust to missing values at inference. To address this challenge, we propose RoVTL (Robust Vision-Tabular Learning), a framework designed to handle any level of tabular data availability, from 0% to 100%. RoVTL comprises two key stages: contrastive pretraining, where we introduce tabular attribute missingness as data augmentation to promote robustness, and downstream task tuning, where tabular missingness is complemented by a novel Tabular More vs. Fewer loss that ranks performance based on the amount of available tabular data. Combined with gated-cross attention fusion module, our tuning approach enables consistent performance across all tabular data completeness scenarios. We evaluate RoVTL on cardiac MRI scans from the UK Biobank, demonstrating superior robustness to missing tabular data compared to prior methods. Furthermore, RoVTL successfully generalizes to an external cardiac MRI dataset for multimodal disease classification, and extends to the natural images domain, achieving robust performance on a car advertisements dataset. The model weights and code are available at this https URL.
>
---
#### [replaced 092] Source-Free Domain Adaptation for Geospatial Point Cloud Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.08375](https://arxiv.org/pdf/2601.08375)**

> **作者:** Yuan Gao; Di Cao; Xiaohuan Xi; Sheng Nie; Shaobo Xia; Cheng Wang
>
> **摘要:** Semantic segmentation of 3D geospatial point clouds is fundamental to remote sensing applications, yet domain shifts caused by regional and acquisition-related variations often degrade model performance. Although domain adaptation can mitigate such shifts, existing methods typically require access to source-domain data, which is often infeasible due to privacy concerns and regulatory policies. To address this, we propose LoGo (Local-Global Dual-Consensus), a novel source-free unsupervised domain adaptation (SFUDA) framework requiring only a pretrained model and unlabeled target data. At the local level, we introduce a class-balanced prototype estimation module that ensures that robust feature prototypes can be generated even for sample-scarce tail classes, effectively mitigating the feature collapse caused by long-tailed distributions. At the global level, we introduce an optimal transport-based global distribution alignment module that formulates pseudo-label assignment as a global optimization problem, effectively correcting the over-dominance of head classes inherent in local greedy assignments, and thereby preventing model predictions from being severely biased towards majority classes. Finally, we propose a dual-consistency pseudo-label filtering mechanism that retains only high-confidence pseudo-labels where local multi-augmented ensemble predictions align with global optimal transport assignments for self-training. Extensive experiments on two challenging benchmarks, encompassing cross-scene and cross-sensor settings, demonstrate that LoGo consistently outperforms existing state-of-the-art methods. The source code is available at this https URL.
>
---
#### [replaced 093] VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models
- **分类: cs.CR; cs.CL; cs.CV; cs.LG; stat.ML**

- **简介: 该论文属于安全测试任务，旨在破解视觉语言模型的防护机制。提出VERA-V框架，通过概率推断生成隐蔽的对抗样本，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2510.17759](https://arxiv.org/pdf/2510.17759)**

> **作者:** Qilin Liao; Anamika Lochab; Ruqi Zhang
>
> **备注:** 18 pages, 7 Figures,
>
> **摘要:** Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o. We include the code on the project page available here: this https URL
>
---
#### [replaced 094] LaRe: Latent Refocusing for Multimodal Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出LaRe，一种基于潜在空间的多模态推理方法，解决视觉聚焦与计算效率的平衡问题。通过隐式重构提升准确率并减少token消耗。**

- **链接: [https://arxiv.org/pdf/2511.02360](https://arxiv.org/pdf/2511.02360)**

> **作者:** Jizheng Ma; Xiaofei Zhou; Geyuan Zhang; Yanlong Song; Han Yan
>
> **摘要:** Chain of Thought (CoT) reasoning enhances logical performance by decomposing complex tasks, yet its multimodal extension faces a trade-off. The prevailing Thinking with Images paradigm achieves visual refocusing by explicitly cropping image regions, yet incurs rapidly growing computational overhead. The emerging line of latent-space reasoning reduces token consumption, but lacks the capacity for dynamic refocusing. We argue that this trade-off stems from a tacitly accepted premise that effective visual refocusing must occur in the form of explicit tokens. Building on this, we propose Latent Refocusing (LaRe), a new multimodal reasoning paradigm in which visual refocusing takes place entirely within the latent space. We further design a semantic augmentation training strategy that ensures the semantic structure of the latent space through visual reconstruction objective. Experimental evaluations demonstrate that LaRe improves average accuracy by 7.6% compared to existing baselines while reducing the number of tokens required for inference by 59.7%. When scaled to a 8B-parameter Vision-Language Model backbone, LaRe achieves performance comparable to state-of-the-art methods, demonstrating the efficacy of our proposed latent refocusing paradigm for multimodal reasoning.
>
---
#### [replaced 095] Efficient Transferable Optimal Transport via Min-Sliced Transport Plans
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19741](https://arxiv.org/pdf/2511.19741)**

> **作者:** Xinran Liu; Elaheh Akbari; Rocio Diaz Martin; Navid NaderiAlizadeh; Soheil Kolouri
>
> **摘要:** Optimal Transport (OT) offers a powerful framework for finding correspondences between distributions and addressing matching and alignment problems in various areas of computer vision, including shape analysis, image generation, and multimodal tasks. The computation cost of OT, however, hinders its scalability. Slice-based transport plans have recently shown promise for reducing the computational cost by leveraging the closed-form solutions of 1D OT problems. These methods optimize a one-dimensional projection (slice) to obtain a conditional transport plan that minimizes the transport cost in the ambient space. While efficient, these methods leave open the question of whether learned optimal slicers can transfer to new distribution pairs under distributional shift. Understanding this transferability is crucial in settings with evolving data or repeated OT computations across closely related distributions. In this paper, we study the min-Sliced Transport Plan (min-STP) framework and investigate the transferability of optimized slicers: can a slicer trained on one distribution pair yield effective transport plans for new, unseen pairs? Theoretically, we show that optimized slicers remain close under slight perturbations of the data distributions, enabling efficient transfer across related tasks. To further improve scalability, we introduce a minibatch formulation of min-STP and provide statistical guarantees on its accuracy. Empirically, we demonstrate that the transferable min-STP achieves strong one-shot matching performance and facilitates amortized training for point cloud alignment and flow-based generative modeling.
>
---
#### [replaced 096] Intuitions of Machine Learning Researchers about Transfer Learning for Medical Image Classification
- **分类: cs.CV; cs.CY; cs.HC**

- **链接: [https://arxiv.org/pdf/2510.00902](https://arxiv.org/pdf/2510.00902)**

> **作者:** Yucheng Lu; Hubert Dariusz Zając; Veronika Cheplygina; Amelia Jiménez-Sánchez
>
> **备注:** Under review
>
> **摘要:** Transfer learning is crucial for medical imaging, yet the selection of source datasets often relies on researchers' intuition rather than systematic principles, which can impact the generalizability of algorithms and, thus, patient outcomes. This study investigates these decisions through a task-based survey with machine learning practitioners. Unlike prior work that benchmarks models and experimental setups, we take a human-computer interaction (HCI) perspective on how practitioners select source datasets. Our findings indicate that choices are task-dependent and influenced by community practices, dataset properties, and computational (data embedding), or perceived visual or semantic similarity. However, similarity ratings and expected performance are not always aligned, challenging a traditional "more similar is better" view. Moreover, ethical and fairness considerations remain largely absent from source dataset sections. Participants often used ambiguous terminology, which suggests a need for clearer definitions and tools to make them explicit and usable. By clarifying these heuristics and introducing a conceptual framework of transfer learning factors, this work provides practical insights for more systematic source selection in transfer learning.
>
---
#### [replaced 097] RAVE: Re-Allocating Visual Attention in Large Multimodal Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18359](https://arxiv.org/pdf/2605.18359)**

> **作者:** Xi Leng; Xinhong Ma; Ziqiang Dong; Feng Zhang; Xiaoying Tang; Yang Yang; Guanjun Jiang
>
> **摘要:** Large multimodal models (LMMs) inherit the self-attention mechanism of pretrained language backbones, yet standard attention can exhibit suboptimal allocation, including cross-modal misallocation between textual and visual evidence and intra-visual imbalance among visual tokens. We propose RAVE (Re-Allocating Visual Attention), a lightweight pair-gating mechanism that adds a learned query-key bias to pre-softmax attention scores over visual keys, derived from pre-RoPE query and key features. RAVE requires no architectural modification to the backbone and can be trained end-to-end with the rest of the model. Across a suite of multimodal benchmarks, RAVE improves over standard attention by an average of 3 points, with the largest gains on perception-intensive tasks -- including multilingual OCR, chart understanding, document VQA, and scene text VQA -- where accurate visual grounding is critical.
>
---
#### [replaced 098] SketchAssist: A Practical Assistant for Semantic Edits and Precise Local Redrawing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14140](https://arxiv.org/pdf/2512.14140)**

> **作者:** Han Zou; Yan Zhang; Ruiqi Yu; Cong Xie; Jie Huang; Zhenpeng Zhan
>
> **摘要:** Sketch editing requires jointly handling high-level semantic changes and precise local redrawing, a combination that is particularly challenging for sparse, style-sensitive line art. Unlike natural images, sketches rely on minimal visual cues, making it difficult for existing methods to reconcile global semantic modifications with fine-grained structural control while preserving overall coherence. We present SketchAssist, an interactive sketch assistant that unifies instruction-guided editing with line-guided region redrawing, enabling efficient and controllable sketch manipulation while preserving overall composition. To support this task, we introduce a controllable data generation pipeline that constructs structured edit sequences with precise attribute variations and maintains structural alignment across multi-step modifications, while expanding stylistic diversity via style-preserving transformations. Building on this data, SketchAssist adopts a unified framework based on DiT, using a multi-channel input representation to encode sketches, masks, and guidance signals within a single interface. To further handle different editing modes, we integrate a Task-guided Mixture-of-Experts (T-MoE) into LoRA layers, enabling adaptive control over semantic and structural guidance. Extensive experiments demonstrate state-of-the-art performance on both tasks, achieving strong instruction adherence and improved structural and style consistency compared to recent methods. Together, our method provide a practical and controllable solution for sketch editing.
>
---
#### [replaced 099] Mining Attribute Subspaces for Efficient Fine-tuning of 3D Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10095](https://arxiv.org/pdf/2604.10095)**

> **作者:** Yu Jiang; Hanwen Jiang; Ahmed Abdelkader; Wen-Sheng Chu; Brandon Y. Feng; Zhangyang Wang; Qixing Huang
>
> **备注:** 10 pages, 8 figures. Code here: this https URL
>
> **摘要:** With the emergence of 3D foundation models, there is growing interest in fine-tuning them for downstream tasks, where LoRA is the dominant fine-tuning paradigm. As 3D datasets exhibit distinct variations in texture, geometry, camera motion, and lighting, there are interesting fundamental questions: 1) Are there LoRA subspaces associated with each type of variation? 2) Are these subspaces disentangled (i.e., orthogonal to each other)? 3) How do we compute them effectively? This paper provides answers to all these questions. We introduce a robust approach that generates synthetic datasets with controlled variations, fine-tunes a LoRA adapter on each dataset, and extracts a LoRA sub-space associated with each type of variation. We show that these subspaces are approximately disentangled. Integrating them leads to a reduced LoRA subspace that enables efficient LoRA fine-tuning with improved prediction accuracy for downstream tasks. In particular, we show that such a reduced LoRA subspace, despite being derived entirely from synthetic data, generalizes to real datasets. An ablation study validates the effectiveness of the choices in our approach.
>
---
#### [replaced 100] GeoSolver: Scaling Test-Time Reasoning in Remote Sensing with Fine-Grained Process Supervision
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09551](https://arxiv.org/pdf/2603.09551)**

> **作者:** Lang Sun; Ronghao Fu; Zhuoran Duan; Haoran Liu; Xueyan Liu; Bo Yang
>
> **备注:** Code: this https URL
>
> **摘要:** While Vision-Language Models (VLMs) have significantly advanced remote sensing interpretation, enabling them to perform complex, step-by-step reasoning remains highly challenging. Recent efforts to introduce Chain-of-Thought (CoT) reasoning to this domain have shown promise, yet ensuring the visual faithfulness of these intermediate steps remains a critical bottleneck. To address this, we introduce GeoSolver, a novel framework that transitions remote sensing reasoning toward verifiable, process-supervised reinforcement learning. We first construct Geo-PRM-2M, a large-scale, token-level process supervision dataset synthesized via entropy-guided Monte Carlo Tree Search (MCTS) and targeted visual hallucination injection. Building upon this dataset, we train GeoPRM, a token-level process reward model (PRM) that provides granular faithfulness feedback. To effectively leverage these verification signals, we propose Process-Aware Tree-GRPO, a reinforcement learning algorithm that integrates tree-structured exploration with a faithfulness-weighted reward mechanism to precisely assign credit to intermediate steps. Extensive experiments demonstrate that our resulting model, GeoSolver-9B, achieves state-of-the-art performance across diverse remote sensing benchmarks. Crucially, GeoPRM unlocks robust Test-Time Scaling (TTS). Serving as a universal geospatial verifier, it seamlessly scales the performance of GeoSolver-9B and directly enhances general-purpose VLMs, highlighting its remarkable cross-model generalization.
>
---
#### [replaced 101] Left-Right Symmetry Breaking in CLIP-style Vision-Language Models Trained on Synthetic Spatial-Relation Data
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.12809](https://arxiv.org/pdf/2601.12809)**

> **作者:** Takaki Yamamoto; Chihiro Noguchi; Toshihiro Tanizawa
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Spatial understanding remains a key challenge in vision-language models. Yet it is still unclear whether such understanding is truly acquired, and if so, through what mechanisms. We present a controllable 1D image-text testbed to probe how left-right relational understanding emerges in Transformer-based vision and text encoders trained with a CLIP-style contrastive objective. We train lightweight Transformer-based vision and text encoders end-to-end on paired descriptions of one- and two-object scenes and evaluate generalization to unseen object pairs while systematically varying label and layout diversity. We find that contrastive training learns left-right relations and that label diversity, more than layout diversity, is the primary driver of generalization in this setting. To gain the mechanistic understanding, we perform an attention decomposition and show that interactions between positional and token embeddings induce a horizontal attention gradient that breaks left-right symmetry in the encoders; ablating this contribution substantially reduces left-right discrimination. Our results provide a mechanistic insight of when and how CLIP-style models acquire relational competence.
>
---
#### [replaced 102] SpaceVista: All-Scale Visual Spatial Reasoning from mm to km
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.09606](https://arxiv.org/pdf/2510.09606)**

> **作者:** Peiwen Sun; Shiqiang Lang; Dongming Wu; Yi Ding; Kaituo Feng; Huadai Liu; Zhen Ye; Rui Liu; Yun-Hui Liu; Jianan Wang; Xiangyu Yue
>
> **备注:** Project Page: this https URL
>
> **摘要:** With the current surge in spatial reasoning explorations, researchers have made significant progress in understanding indoor scenes, but still struggle with diverse applications such as robotics and autonomous driving. This paper aims to advance all-scale spatial reasoning across diverse scenarios by tackling two key challenges: 1) the heavy reliance on indoor 3D scans and labor-intensive manual annotations for dataset curation; 2) the absence of effective all-scale scene modeling, which often leads to overfitting to individual scenes. In this paper, we introduce a holistic solution that integrates a structured spatial reasoning knowledge system, scale-aware modeling, and a progressive training paradigm, as the first attempt to broaden the all-scale spatial intelligence of MLLMs to the best of our knowledge. Using a task-specific, specialist-driven automated pipeline, we curate over 38K video scenes across 5 spatial scales to create SpaceVista-1M, a dataset comprising approximately 1M spatial QA pairs spanning 19 diverse task types. While specialist models can inject useful domain knowledge, they are not reliable for evaluation. We then build an all-scale benchmark with precise annotations by manually recording, retrieving, and assembling video-based data. However, naive training with SpaceVista-1M often yields suboptimal results due to the potential knowledge conflict. Accordingly, we introduce SpaceVista-7B, a spatial reasoning model that accepts dense inputs beyond semantics and uses scale as an anchor for scale-aware experts and progressive rewards. Finally, extensive evaluations across 5 benchmarks, including our SpaceVista-Bench, demonstrate competitive performance, showcasing strong generalization across all scales and scenarios. Our dataset, model, and benchmark will be released on this https URL .
>
---
#### [replaced 103] Tetris: Tile-level Sampling for Efficient and High-Fidelity Video Object Tracking
- **分类: cs.CV; cs.DB**

- **链接: [https://arxiv.org/pdf/2605.25538](https://arxiv.org/pdf/2605.25538)**

> **作者:** Chanwut Kittivorawong; Alena Chao; Charlie Si; Alvin Cheung
>
> **摘要:** Track materialization converts raw video into reusable object tracks that downstream queries can run against without rerunning tracking, but extracting those tracks efficiently and with high fidelity remains expensive. Prior systems reduce cost through temporal frame sampling, erasing the inter-frame motion that fine-grained tracking requires. In stationary video, however, large portions of each frame contain no objects of interest, and the remaining regions tolerate different sampling rates. We present Tetris, a track-extraction system that decomposes videos into a tile-based polyomino data model, enabling fine-grained spatiotemporal pruning that reduces detector calls with minimal fidelity loss. Tetris runs three operators upstream of the user-provided detector: a classifier identifies relevant tiles and groups them into polyominoes, an integer linear program (ILP) prunes redundant polyominoes under a user-specified accuracy constraint, and a packer assembles the survivors into canvases that minimize detector calls. Across 7 stationary-video datasets, Tetris stays within a 5% tracking accuracy loss of a full-frame, every-frame reference pipeline, whereas prior systems exceed this bound on 3 of the 7 datasets. At this 5% bound, Tetris achieves up to 17.4x higher throughput than prior systems and up to 68.8x higher than the reference pipeline. The project page is at this https URL .
>
---
#### [replaced 104] UPOCR: Towards Unified Pixel-Level OCR Interface
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2312.02694](https://arxiv.org/pdf/2312.02694)**

> **作者:** Dezhi Peng; Zhenhua Yang; Jiaxin Zhang; Chongyu Liu; Yongxin Shi; Kai Ding; Fengjun Guo; Lianwen Jin
>
> **备注:** ICML 2024 Version
>
> **摘要:** Existing optical character recognition (OCR) methods rely on task-specific designs with divergent paradigms, architectures, and training strategies, which significantly increases the complexity of research and maintenance and hinders the fast deployment in applications. To this end, we propose UPOCR, a simple-yet-effective generalist model for Unified Pixel-level OCR interface. Specifically, the UPOCR unifies the paradigm of diverse OCR tasks as image-to-image transformation and the architecture as a vision Transformer (ViT)-based encoder-decoder with learnable task prompts. The prompts push the general feature representations extracted by the encoder towards task-specific spaces, endowing the decoder with task awareness. Moreover, the model training is uniformly aimed at minimizing the discrepancy between the predicted and ground-truth images regardless of the inhomogeneity among tasks. Experiments are conducted on three pixel-level OCR tasks including text removal, text segmentation, and tampered text detection. Without bells and whistles, the experimental results showcase that the proposed method can simultaneously achieve state-of-the-art performance on three tasks with a unified single model, which provides valuable strategies and insights for future research on generalist OCR models. Code is available at this https URL.
>
---
#### [replaced 105] What Demands Attention in Urban Street Scenes? From Scene Understanding towards Road Safety: A Survey of Vision-driven Datasets and Studies
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.06513](https://arxiv.org/pdf/2507.06513)**

> **作者:** Yaoqi Huang; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **备注:** 40 tasks, 78 datasets
>
> **摘要:** Advances in vision-based sensors and computer vision algorithms have significantly improved the analysis and understanding of traffic scenarios. To facilitate the use of these improvements for road safety, this survey systematically categorizes the critical elements that demand attention in traffic scenarios and comprehensively analyzes available vision-driven tasks and datasets. Compared to existing surveys that focus on isolated domains, our taxonomy categorizes attention-worthy traffic entities into two main groups that are anomalies and normal but critical entities, integrating ten categories and twenty subclasses. It establishes connections between inherently related fields and provides a unified analytical framework. Our survey highlights the analysis of 35 vision-driven tasks and comprehensive examinations and visualizations of 73 available datasets based on the proposed taxonomy. The cross-domain investigation covers the pros and cons of each benchmark with the aim of providing information on standards unification and resource optimization. Our article concludes with a systematic discussion of the existing weaknesses, underlining the potential effects and promising solutions from various perspectives. The integrated taxonomy, comprehensive analysis, and recapitulatory tables serve as valuable contributions to this rapidly evolving field by providing researchers with a holistic overview, guiding strategic resource selection, and highlighting critical research gaps.
>
---
#### [replaced 106] Efficient All-Pairs Correlation Volume Sampling for Optical Flow Estimation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.16942](https://arxiv.org/pdf/2505.16942)**

> **作者:** Karlis Martins Briedis; Markus Gross; Christopher Schroers
>
> **备注:** CVPR 2026
>
> **摘要:** Recent optical flow estimation methods often employ local cost sampling from a dense all-pairs correlation volume. This results in quadratic computational and memory complexity in the number of pixels. Although an alternative memory-efficient implementation with on-demand cost computation exists, this is significantly slower in practice and therefore many prior methods process images at downsampled resolutions, missing fine-grained details. To address this, we propose an algorithm for both memory and compute-efficient implementation of the all-pairs correlation volume sampling, still matching the exact mathematical operator as defined by RAFT. Our approach outperforms on-demand sampling by up to 92% while maintaining equally low memory usage, and performs at least on par with the default implementation with up to 99% lower memory usage. As cost sampling makes up a significant portion of the overall runtime, this can translate to up to 63% savings for the total end-to-end model inference on high-resolution inputs. Our evaluation of existing methods includes an 8K ultra-high-resolution dataset and an inference-time extension of the SEA-RAFT method. With this, we achieve state-of-the-art results at high resolutions both in accuracy and runtime.
>
---
#### [replaced 107] Prototyping an End-to-End Multi-Modal Tiny-CNN for Cardiovascular Sensor Patches
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18668](https://arxiv.org/pdf/2510.18668)**

> **作者:** Mustafa Fuad Rifet Ibrahim; Tunc Alkanat; Felix Manthey; Maurice Meijer; Alexander Schlaefer; Peer Stelldinger
>
> **备注:** 11 pages, 2 figures. Extended version of our 2024 IEEE PerCom paper, with direct on-device energy measurements, a BLE communication benchmark, architecture comparisons, and an extended evaluation. Submitted to Biomedical Signal Processing and Control
>
> **摘要:** The vast majority of cardiovascular diseases may be preventable if early signs and risk factors are detected. Cardiovascular monitoring with body-worn sensor devices like sensor patches allows for the detection of such signs while preserving the freedom and comfort of patients. However, the analysis of the sensor data must be robust, reliable, efficient, and highly accurate. Deep learning methods can automate data interpretation, reducing the workload of clinicians. In this work, we analyze the feasibility of applying deep learning models to the classification of synchronized electrocardiogram (ECG) and phonocardiogram (PCG) recordings on resource-constrained medical edge devices. We propose a convolutional neural network with early fusion of data to solve a binary classification problem. The model is trained and validated on the synchronized ECG and PCG recordings from the Physionet Challenge 2016 dataset. Our approach reduces memory footprint and compute cost by approximately three orders of magnitude compared with the state-of-the-art while maintaining competitive accuracy. We further demonstrate the applicability of the proposed model on medical edge devices by measuring its energy consumption on a microcontroller equipped with a neural processing unit (NPU) and benchmarking the energy of Bluetooth Low Energy (BLE) communication on a representative BLE evaluation kit across a range of payload sizes. The comparison confirms that on-device inference can be more energy efficient than continuous data streaming.
>
---
#### [replaced 108] From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15572](https://arxiv.org/pdf/2511.15572)**

> **作者:** Huiyuan Tian; Bonan Xu; Shijian Li
>
> **备注:** 22 pages, 22 figures. Accepted at the ICML 2026
>
> **摘要:** Feature-map knowledge distillation (KD) transfers internal representations well between comparably sized Vision Transformers (ViTs), but it often fails in compression. We revisit this failure and uncover a paradox. Sample-wise SVD shows that each image is highly compressible, which seems to suggest that a narrow student with a linear projector should match the teacher "in principle". However, a dataset-level view contradicts this intuition: PCA shows that the teacher is a union of low-rank subspaces with significant subspace rotation across inputs. We further introduce token-level Spectral Energy Patterns (SEP) and find an architecture-invariant encoding law: tokens spread energy broadly across channel modes even when they live in low-rank subspace, creating a bandwidth mismatch. We refer to this combined phenomenon as an encoding mismatch. We propose two minimal remedies, Lift or WideLast: (i) Lift retains a lightweight lifting projector at inference to provide wider channel, or (ii) WideLast widens only the student's last block, enabling an input-dependent expansion. On ImageNet-1K, these fixes revive feature KD for ViT compression, improving DeiT-Tiny distilled from CaiT-S24 from 74.86% to 77.53%/78.23% top-1 accuracy, and they also strengthen students trained without distillation. Our analyses clarify when and why feature-map KD fails and then how to fix it. Code and raw data are provided in this https URL.
>
---
#### [replaced 109] MiVE: Multiscale Vision-language features for reference-guided video Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.14664](https://arxiv.org/pdf/2605.14664)**

> **作者:** Tong Wang; Meng Zou; Chengjing Wu; Xiaochao Qu; Luoqi Liu; Xiaolin Hu; Ting Liu
>
> **备注:** ICML 2026
>
> **摘要:** Reference-guided video editing takes a source video, a text instruction, and a reference image as inputs, requiring the model to faithfully apply the instructed edits while preserving original motion and unedited content. Existing methods fall into two paradigms, each with inherent limitations: decoupled encoders suffer from modality gaps when processing instructions and visual content independently, while unified vision-language encoders lose fine-grained spatial details by relying solely on final-layer representations. We observe that VLM layers encode complementary information hierarchically -- early layers capture localized spatial details essential for precise editing, while deeper layers encode global semantics for instruction comprehension. Building on this insight, we present MiVE (Multiscale Vision-language features for reference-guided video Editing), a framework that repurposes VLMs as multiscale feature extractors. MiVE extracts hierarchical features from Qwen3-VL and integrates them into a unified self-attention Diffusion Transformer, eliminating the modality mismatch inherent in cross-attention designs. Experiments demonstrate that MiVE achieves state-of-the-art performance by ranking highest in human preference, outperforming both academic methods and commercial systems.
>
---
#### [replaced 110] PRBench: A Standardized Probabilistic Robustness Benchmark
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.01724](https://arxiv.org/pdf/2511.01724)**

> **作者:** Yi Zhang; Zheng Wang; Zhen Chen; Wenjie Ruan; Qing Guo; Siddartha Khastgir; Carsten Maple; Xingyu Zhao
>
> **摘要:** Deep learning models are notoriously vulnerable to imperceptible perturbations. Most existing research centers on adversarial robustness (AR), which evaluates models under worst-case scenarios by examining the existence of deterministic adversarial examples (AEs). In contrast, probabilistic robustness (PR) adopts a statistical perspective, measuring the probability that predictions remain correct under stochastic perturbations. While PR is widely regarded as a practical complement to AR, dedicated training methods for improving PR are still relatively underexplored, albeit with emerging progress. Among the few PR-targeted training methods, we identify three limitations: i non-comparable evaluation protocols; ii limited comparisons to strong AT baselines despite anecdotal PR gains from AT; and iii no unified framework to compare the generalization of these methods. Thus, we introduce PRBench, the first benchmark dedicated to evaluating improvements in PR achieved by different robustness training methods. PRBench empirically compares most common AT and PR-targeted training methods using a comprehensive set of metrics, including clean accuracy, PR and AR performance, training efficiency, and generalization error (GE). We also provide theoretical analysis on the GE of PR performance across different training methods. Main findings revealed by PRBench include: AT methods are more versatile than PR-targeted training methods in terms of improving both AR and PR performance across diverse hyperparameter settings, while PR-targeted training methods consistently yield lower GE and higher clean accuracy. A leaderboard comprising 229 trained models across 7 datasets and 10 model architectures is publicly available at this https URL.
>
---
#### [replaced 111] UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出UltraCUA，解决计算机使用代理依赖低级GUI操作的问题。通过融合GUI操作与高级工具调用，提升执行效率和稳定性。属于人工智能任务中的智能代理领域。**

- **链接: [https://arxiv.org/pdf/2510.17790](https://arxiv.org/pdf/2510.17790)**

> **作者:** Yuhao Yang; Zhen Yang; Zi-Yi Dou; Anh Nguyen; Keen You; Omar Attia; Andrew Szot; Michael Feng; Ram Ramrakhya; Alexander Toshev; Chao Huang; Yinfei Yang; Zhe Gan
>
> **摘要:** Computer-use agents face a fundamental limitation. They rely exclusively on primitive GUI actions (click, type, scroll), creating brittle execution chains prone to cascading failures. While API-driven agents harness rich capabilities through structured interfaces and tools, computer-use agents remain constrained to low-level visual interactions. We present UltraCUA, a foundation model that transcends this limitation through hybrid action-seamlessly unifying primitive GUI operations with high-level tool execution. Our innovation rests on four critical advances. First, an automated pipeline extracts and scales tool capabilities from software documentation and code repositories. Second, a synthetic data engine produces 17,000+ verifiable tasks capturing real-world computer-use complexity. Third, comprehensive hybrid action trajectory collection incorporates both GUI primitives and strategic tool calls. Fourth, a two-stage training methodology combines supervised fine-tuning with online reinforcement learning, enabling intelligent action selection between GUI and API. Evaluation with our 7B and 32B UltraCUA models reveals transformative performance gains. On OSWorld, UltraCUA achieves 22% relative improvement while executing 11% faster than existing approaches, averagely. Cross-domain validation on WindowsAgentArena demonstrates robust generalization with 21.7% success rate, surpassing Windows-trained baselines. The hybrid action paradigm proves essential, reducing error propagation while improving execution efficiency. This work establishes a scalable paradigm bridging primitive GUI interactions and high-level tool intelligence, enabling more resilient and adaptable computer use agents for diverse environments and complex real-world tasks.
>
---
#### [replaced 112] LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.15283](https://arxiv.org/pdf/2601.15283)**

> **作者:** Ruofan Liang; Norman Müller; Ethan Weber; Duncan Zauss; Nandita Vijaykumar; Peter Kontschieder; Christian Richardt
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** We present a novel approach for interactive light editing in indoor scenes from a single multi-view scene capture. Our method leverages a generative image-based light decomposition model that factorizes complex indoor scene illumination into its constituent light sources. This factorization enables independent manipulation of individual light sources, specifically allowing control over their state (on/off), chromaticity, and intensity. We further introduce multi-view lighting harmonization to ensure consistent propagation of the lighting decomposition across all scene views. This is integrated into a relightable 3D Gaussian splatting representation, providing real-time interactive control over the individual light sources. Our results demonstrate highly photorealistic lighting decomposition and relighting outcomes across diverse indoor scenes. We evaluate our method on both synthetic and real-world datasets and provide a quantitative and qualitative comparison to state-of-the-art techniques. For video results and interactive demos, see this https URL.
>
---
#### [replaced 113] Radar-Camera BEV Multi-Task Learning with Cross-Task Attention Bridge for Joint 3D Detection and Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.12918](https://arxiv.org/pdf/2604.12918)**

> **作者:** Ahmet İnanç; Özgür Erkent
>
> **备注:** 8 pages, 5 figures, 3 Tables, Accepted at Radar in Robotics: New Frontiers workshop, at IEEE International Conference on Robotics & Automation (ICRA), 2026
>
> **摘要:** Bird's-eye-view (BEV) representations are the dominant paradigm for 3D perception in autonomous driving, providing a unified spatial canvas where detection and segmentation features are geometrically registered to the same physical coordinate system. However, existing radar-camera fusion methods treat these tasks in isolation, missing the opportunity for cross-task feature sharing: object-level geometric cues from detection can sharpen segmentation, while dense road-layout context from segmentation can anchor detection. We propose \textbf{CTAB} (Cross-Task Attention Bridge), a bidirectional module that exchanges features between detection and segmentation branches via multi-scale deformable attention in shared BEV space. CTAB is integrated into a multi-task framework with an Instance Normalization-based segmentation decoder and learnable BEV upsampling to provide a more detailed BEV representation. On nuScenes, CTAB improves segmentation on 7 classes over the joint multi-task baseline at essentially neutral detection. On a 4-class subset (drivable area, pedestrian crossing, walkway, vehicle), our joint multi-task model achieves 51.0 mIoU-4 while simultaneously providing competitive 3D detection.
>
---
#### [replaced 114] An Empirical Study of Machine Learning Robustness and Scalability for Imbalanced Tabular Clinical Data in Emergency and Critical Care
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21602](https://arxiv.org/pdf/2512.21602)**

> **作者:** Yusuf Brima; Marcellin Atemkeng
>
> **摘要:** Every year, millions of patients pass through emergency departments and intensive care units, where clinicians must make high-stakes decisions under time pressure and uncertainty. Machine learning could support prediction of deterioration, triage, and rare critical outcomes, but clinical data are often severely imbalanced, biasing models toward majority classes and reducing predictive performance. Developing robust and efficient models for imbalanced clinical tabular data therefore remains an important challenge. We evaluated six model families on imbalanced tabular data from the MIMIC-IV-ED and eICU databases: Decision Tree, Random Forest, XGBoost, TabNet, TabICL, and TabPFN v2.6. Trainable models were optimized using Bayesian hyperparameter tuning, while foundation models were evaluated in their pretrained inference regime without task-specific reweighting. Models were assessed using Macro F1-score, robustness to increasing imbalance, and computational scalability across seven clinical prediction tasks. Results differed across datasets. On MIMIC-IV-ED, TabPFN v2.6 and TabICL achieved the strongest average Macro F1 ranks, with XGBoost remaining competitive. On eICU, XGBoost consistently performed best, followed by other tree-based methods, while foundation models achieved intermediate performance. Across both datasets, TabNet showed the largest degradation under increasing imbalance and the highest computational cost. Training-time analysis showed that tree-based methods scaled most favorably with dataset size, while foundation models offered low per-task adaptation cost. These findings suggest that no single model family dominates across all clinical settings. However, tabular foundation models are narrowing the performance gap with strong classical baselines while offering a distinct efficiency-performance trade-off that may benefit resource-constrained clinical environments.
>
---
#### [replaced 115] Chat2Workflow: A Benchmark for Generating Executable Visual Workflows with Natural Language
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于自然语言到可视化工作流生成任务，旨在解决手动构建工作流成本高、易出错的问题。通过构建基准和提出基线方法，推动工业级自动化发展。**

- **链接: [https://arxiv.org/pdf/2604.19667](https://arxiv.org/pdf/2604.19667)**

> **作者:** Yi Zhong; Buqiang Xu; Yijun Wang; Zifei Shan; Shuofei Qiao; Guozhou Zheng; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** At present, executable visual workflows have emerged as a mainstream paradigm in real-world industrial deployments, offering strong reliability and controllability. However, in current practice, such workflows are almost entirely constructed through manual engineering: developers must carefully design workflows, write prompts for each step, and repeatedly revise the logic as requirements evolve -- making development costly, time-consuming, and error-prone. To study whether large language models can automate this multi-round interaction process, we introduce Chat2Workflow, a benchmark for generating executable visual workflows directly from natural language, and propose a robust agentic baseline to improve performance. The benchmark is built from a large collection of real-world business workflows, with each instance designed so that the generated workflow can be transformed and directly deployed to practical workflow platforms such as Dify and Coze. Experimental results show that while state-of-the-art language models can often capture high-level intent, they struggle to generate correct, stable, and executable workflows, especially given complex and evolving requirements. Although our agentic baseline yields up to 6.05% resolve rate gains, the remaining real-world gap positions Chat2Workflow as a foundation for advancing industrial-grade automation. Code is available at this https URL.
>
---
#### [replaced 116] Pusa V1.0: Unlocking Temporal Control in Pretrained Video Diffusion Models via Vectorized Timestep Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.16116](https://arxiv.org/pdf/2507.16116)**

> **作者:** Yaofang Liu; Yumeng Ren; Aitor Artola; Yuxuan Hu; Xiaodong Cun; Xiaotong Zhao; Alan Zhao; Raymond H. Chan; Suiyun Zhang; Rui Liu; Dandan Tu; Jean-Michel Morel
>
> **备注:** Code is open-sourced at this https URL
>
> **摘要:** The rapid advancement of video diffusion models has been hindered by fundamental limitations in temporal modeling, particularly the rigid synchronization of frame evolution imposed by conventional scalar timestep variables. While task-specific adaptations and autoregressive models have sought to address these challenges, they remain constrained by computational inefficiency, catastrophic forgetting, or narrow applicability. In this work, we present \textbf{Pusa} V1.0, a versatile model that leverages \textbf{vectorized timestep adaptation (VTA)} to enable fine-grained temporal control within a unified video diffusion framework. Note that VTA is a non-destructive adaptation, which means that it fully preserves the capabilities of the base model. Unlike conventional methods like Wan-I2V, which finetune a base text-to-video (T2V) model with abundant resources to do image-to-video (I2V), we achieve comparable results in a zero-shot manner after an ultra-efficient finetuning process based on VTA. Moreover, this method also unlocks many other zero-shot capabilities simultaneously, such as start-end frames and video extension -- all without task-specific training. Meanwhile, it keeps the T2V capability from the base model. Mechanistic analyses also reveal that our approach preserves the foundation model's generative priors while surgically injecting temporal dynamics, avoiding the combinatorial explosion inherent to the vectorized timestep. This work establishes a scalable, efficient, and versatile paradigm for next-generation video synthesis, democratizing high-fidelity video generation for research and industry alike.
>
---
#### [replaced 117] MVISTA-4D: View-Consistent 4D World Model with Test-Time Action Inference for Robotic Manipulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.09878](https://arxiv.org/pdf/2602.09878)**

> **作者:** Jiaxu Wang; Yicheng Jiang; Tianlun He; Jingkai Sun; Qiang Zhang; Junhao He; Jiahang Cao; Zesen Gan; Mingyuan Sun; Qiming Shao; Xiangyu Yue
>
> **摘要:** World-model-based imagine-then-act becomes a promising paradigm for robotic manipulation, yet existing approaches typically support either purely image-based forecasting or reasoning over partial 3D geometry, limiting their ability to predict complete 4D scene dynamics. This work proposes a novel embodied 4D world model that enables geometrically consistent, arbitrary-view RGBD generation: given only a single-view RGBD observation as input, the model imagines the remaining viewpoints, which can then be back-projected and fused to assemble a more complete 3D structure across time. To efficiently learn the multi-view, cross-modality generation, we explicitly design cross-view and cross-modality feature fusion that jointly encourage consistency between RGB and depth and enforce geometric alignment across views. Beyond prediction, converting generated futures into actions is often handled by inverse dynamics, which is ill-posed because multiple actions can explain the same transition. We address this with a test-time action optimization strategy that backpropagates through the generative model to infer a trajectory-level latent best matching the predicted future, and a residual inverse dynamics model that turns this trajectory prior into accurate executable actions. Experiments on three datasets demonstrate strong performance on both 4D scene generation and downstream manipulation, and ablations provide practical insights into the key design choices.
>
---
#### [replaced 118] UniPCB: A Generation-Assisted Detection Framework for PCB Defect Inspection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.04635](https://arxiv.org/pdf/2605.04635)**

> **作者:** Huan Zhang; Lianghong Tan; Yichu Xu; Zishan Su; Jiangzhong Cao; Huanqi Wu; Linwei Zhu; Xu Zhang
>
> **摘要:** In the Industrial Internet of Things (IIoT), enabling intelligent, real-time Printed Circuit Board (PCB) defect inspection is critical for ensuring product reliability. However, existing IIoT-based visual inspection systems face two compounding challenges: scarce and imbalanced defect samples that limit model training, and insufficient feature representation under complex circuit backgrounds. Existing generation methods rely on single-modality conditions with coarse structural control, while detection methods improve architectures without addressing the data bottleneck. To resolve both challenges jointly, we propose a generation-assisted PCB defect inspection framework that integrates controlled defect synthesis with task-specific defect detection within an IIoT-enabled pipeline. On the generation side, a Multi-modal Condition Generator extracts complementary edge, depth, and text conditions in parallel. A ScaleEncoder then embeds these conditions into the diffusion U-Net at four resolutions, and a Condition Modulation applies FiLM-style spatially-adaptive modulation at each scale, enabling structurally aligned and defect-aware sample synthesis to augment the scarce IIoT dataset. On the detection side, an Inverted Residual Shift Attention couples self-attention with shift-wise convolution to jointly capture global context and local texture, and a Cross-level Complementary Fusion Block generates pixel-level gates for selective cross-level feature fusion. The synthesized samples directly enrich the detection training set, so that improvements in generation compound with improvements in detection. Extensive experiments on DsPCBSD+ demonstrate that UniPCB achieves mAP@0.5 of 98.0% and mAP@0.5:0.95 of 61.8% on defect detection, surpassing all compared methods, while the generation branch attains an FID of 129.61 and SSIM of 0.619, outperforming existing conditional generation approaches.
>
---
#### [replaced 119] EgoProx: Evaluating MLLMs on Egocentric 3D Proximity Reasoning Across a Cognitive Hierarchy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.24456](https://arxiv.org/pdf/2605.24456)**

> **作者:** Jinzhao Li; Yinuo Chen; Dongxu Piao; Panwang Pan; Yifan Yu; Dong Wang; Honglei Yan; Liang Yue; Shaofei Wang; Yixin Chen; Siyuan Huang; Miao Liu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Humans constantly reason about 3D proximity, the relations between their body and surrounding objects, to guide perception and action in daily life. Whether multimodal large language models (MLLMs) can perform such embodied 3D reasoning remains unclear. To this end, we introduce EgoProx, a benchmark for egocentric 3D proximity reasoning. We organize our tasks along a cognitive chain, covering intention, exploration, exploitation, and chain-of-actions reasoning. We also design an agent based data engine that produces diverse and consistent QA pairs at scale. We benchmark prevailing MLLMs on EgoProx and conduct additional analyses with dataset specific and task specific instruction tuning. We observe large cross-domain gains, indicating that current MLLMs contain some spatial knowledge; however, they still struggle to effectively leverage it for spatial reasoning VQA.
>
---
#### [replaced 120] EgoExo-WM: Unlocking Exo Video for Ego World Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.15477](https://arxiv.org/pdf/2605.15477)**

> **作者:** Danny Tran; Roberto Martín-Martín; Kristen Grauman
>
> **备注:** Project Page: this https URL
>
> **摘要:** Egocentric world models present a promising direction for enabling agents to predict and plan, but their performance is constrained by the limited availability of egocentric training data and its inherent partial observability of humans' physical actions. In contrast, exocentric video is abundant and reveals body poses well, but lacks direct alignment with an agent's action space -- and is not egocentric. We propose a method to bridge this gap by extracting structured body pose from exocentric video as a representation of action and transforming the exocentric video to egocentric video, informed by a human kinematics prior. This process unlocks the integration of in-the-wild exocentric data for egocentric world model training. We show that training whole-body action-conditioned egocentric world models with our converted data significantly improves both prediction quality and downstream planning performance, where we infer the sequence of body poses needed to achieve a visual goal state. Our approach paves the way to enlist arbitrary in-the-wild videos for building powerful egocentric world models, furthering applications in robot planning and augmented-reality guidance.
>
---
