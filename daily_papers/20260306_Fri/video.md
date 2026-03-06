# 计算机视觉 cs.CV

- **最新发布 113 篇**

- **更新 98 篇**

## 最新发布

#### [new 001] Transformer-Based Inpainting for Real-Time 3D Streaming in Sparse Multi-Camera Setups
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，解决多相机实时流中缺失纹理的问题。提出一种基于Transformer的修复方法，提升图像一致性与细节，实现高效实时性能。**

- **链接: [https://arxiv.org/pdf/2603.05507](https://arxiv.org/pdf/2603.05507)**

> **作者:** Leif Van Holland; Domenic Zingsheim; Mana Takhsha; Hannah Dröge; Patrick Stotko; Markus Plack; Reinhard Klein
>
> **备注:** You can find the project page this https URL
>
> **摘要:** High-quality 3D streaming from multiple cameras is crucial for immersive experiences in many AR/VR applications. The limited number of views - often due to real-time constraints - leads to missing information and incomplete surfaces in the rendered images. Existing approaches typically rely on simple heuristics for the hole filling, which can result in inconsistencies or visual artifacts. We propose to complete the missing textures using a novel, application-targeted inpainting method independent of the underlying representation as an image-based post-processing step after the novel view rendering. The method is designed as a standalone module compatible with any calibrated multi-camera system. For this we introduce a multi-view aware, transformer-based network architecture using spatio-temporal embeddings to ensure consistency across frames while preserving fine details. Additionally, our resolution-independent design allows adaptation to different camera setups, while an adaptive patch selection strategy balances inference speed and quality, allowing real-time performance. We evaluate our approach against state-of-the-art inpainting techniques under the same real-time constraints and demonstrate that our model achieves the best trade-off between quality and speed, outperforming competitors in both image and video-based metrics.
>
---
#### [new 002] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark
- **分类: cs.CV**

- **简介: 该论文提出UniM基准，用于评估多模态模型在任意交织输入输出下的理解与生成能力，解决统一多模态学习任务中的挑战。**

- **链接: [https://arxiv.org/pdf/2603.05075](https://arxiv.org/pdf/2603.05075)**

> **作者:** Yanlin Li; Minghui Guo; Kaiwen Zhang; Shize Zhang; Yiran Zhao; Haodong Li; Congyue Zhou; Weijie Zheng; Yushen Yan; Shengqiong Wu; Wei Ji; Lei Cui; Furu Wei; Hao Fei; Mong-Li Lee; Wynne Hsu
>
> **备注:** 70 pages, 63 figures, 30 tables, CVPR
>
> **摘要:** In real-world multimodal applications, systems usually need to comprehend arbitrarily combined and interleaved multimodal inputs from users, while also generating outputs in any interleaved multimedia form. This capability defines the goal of any-to-any interleaved multimodal learning under a unified paradigm of understanding and generation, posing new challenges and opportunities for advancing Multimodal Large Language Models (MLLMs). To foster and benchmark this capability, this paper introduces the UniM benchmark, the first Unified Any-to-Any Interleaved Multimodal dataset. UniM contains 31K high-quality instances across 30 domains and 7 representative modalities: text, image, audio, video, document, code, and 3D, each requiring multiple intertwined reasoning and generation capabilities. We further introduce the UniM Evaluation Suite, which assesses models along three dimensions: Semantic Correctness & Generation Quality, Response Structure Integrity, and Interleaved Coherence. In addition, we propose UniMA, an agentic baseline model equipped with traceable reasoning for structured interleaved generation. Comprehensive experiments demonstrate the difficulty of UniM and highlight key challenges and directions for advancing unified any-to-any multimodal intelligence. The project page is this https URL.
>
---
#### [new 003] Location-Aware Pretraining for Medical Difference Visual Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学差异视觉问答任务，旨在解决传统模型难以捕捉细微视觉差异的问题。通过引入位置感知预训练方法，提升模型对医学图像变化的检测与推理能力。**

- **链接: [https://arxiv.org/pdf/2603.04950](https://arxiv.org/pdf/2603.04950)**

> **作者:** Denis Musinguzi; Caren Han; Prasenjit Mitra
>
> **备注:** 11 pages
>
> **摘要:** Unlike conventional single-image models, differential medical VQA frameworks process multiple images to identify differences, mirroring the comparative diagnostic workflow of radiologists. However, standard vision encoders trained on contrastive or classification objectives often fail to capture the subtle visual variations necessary for distinguishing disease progression from acquisition differences. To address this limitation, we introduce a pretraining framework that incorporates location-aware tasks, including automatic referring expressions (AREF), grounded captioning (GCAP), and conditional automatic referring expressions (CAREF). These specific tasks enable the vision encoder to learn fine-grained, spatially grounded visual representations that are often overlooked by traditional pre-training methods. We subsequently integrate this enhanced vision encoder with a language model to perform medical difference VQA. Experimental results demonstrate that our approach achieves state-of-the-art performance in detecting and reasoning about clinically relevant changes in chest X-ray images.
>
---
#### [new 004] A Simple Baseline for Unifying Understanding, Generation, and Editing via Vanilla Next-token Prediction
- **分类: cs.CV**

- **简介: 该论文提出Wallaroo，一个基于单一步预测的统一模型，解决多模态理解、生成与编辑任务。通过分阶段训练和多分辨率支持，实现高效多模态处理。**

- **链接: [https://arxiv.org/pdf/2603.04980](https://arxiv.org/pdf/2603.04980)**

> **作者:** Jie Zhu; Hanghang Ma; Jia Wang; Yayong Guan; Yanbing Zeng; Lishuai Gao; Junqiang Wu; Jie Hu; Leye Wang
>
> **备注:** Technical report. This work serves as a straightforward autoregressive baseline for unifying understanding, generation, and editing
>
> **摘要:** In this work, we introduce Wallaroo, a simple autoregressive baseline that leverages next-token prediction to unify multi-modal understanding, image generation, and editing at the same time. Moreover, Wallaroo supports multi-resolution image input and output, as well as bilingual support for both Chinese and English. We decouple the visual encoding into separate pathways and apply a four-stage training strategy to reshape the model's capabilities. Experiments are conducted on various benchmarks where Wallaroo produces competitive performance or exceeds other unified models, suggesting the great potential of autoregressive models in unifying multi-modality understanding and generation. Our code is available at this https URL.
>
---
#### [new 005] Dark3R: Learning Structure from Motion in the Dark
- **分类: cs.CV**

- **简介: 该论文提出Dark3R，解决低光下结构从运动问题。通过教师-学生蒸馏，利用噪声-清晰图像对训练，实现鲁棒特征匹配与相机位姿估计。**

- **链接: [https://arxiv.org/pdf/2603.05330](https://arxiv.org/pdf/2603.05330)**

> **作者:** Andrew Y Guo; Anagh Malik; SaiKiran Tedla; Yutong Dai; Yiqian Qin; Zach Salehe; Benjamin Attal; Sotiris Nousias; Kyros Kutulakos; David B. Lindell
>
> **备注:** CVPR 2026, Project Page: this https URL
>
> **摘要:** We introduce Dark3R, a framework for structure from motion in the dark that operates directly on raw images with signal-to-noise ratios (SNRs) below $-4$ dB -- a regime where conventional feature- and learning-based methods break down. Our key insight is to adapt large-scale 3D foundation models to extreme low-light conditions through a teacher--student distillation process, enabling robust feature matching and camera pose estimation in low light. Dark3R requires no 3D supervision; it is trained solely on noisy--clean raw image pairs, which can be either captured directly or synthesized using a simple Poisson--Gaussian noise model applied to well-exposed raw images. To train and evaluate our approach, we introduce a new, exposure-bracketed dataset that includes $\sim$42,000 multi-view raw images with ground-truth 3D annotations, and we demonstrate that Dark3R achieves state-of-the-art structure from motion in the low-SNR regime. Further, we demonstrate state-of-the-art novel view synthesis in the dark using Dark3R's predicted poses and a coarse-to-fine radiance field optimization procedure.
>
---
#### [new 006] How far have we gone in Generative Image Restoration? A study on its capability, limitations and evaluation practices
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在评估生成式图像修复的进展与局限。通过多维评价体系分析不同模型性能，揭示其在细节质量和语义控制上的挑战，并提出新的质量评估模型。**

- **链接: [https://arxiv.org/pdf/2603.05010](https://arxiv.org/pdf/2603.05010)**

> **作者:** Xiang Yin; Jinfan Hu; Zhiyuan You; Kainan Yan; Yu Tang; Chao Dong; Jinjin Gu
>
> **摘要:** Generative Image Restoration (GIR) has achieved impressive perceptual realism, but how far have its practical capabilities truly advanced compared with previous methods? To answer this, we present a large-scale study grounded in a new multi-dimensional evaluation pipeline that assesses models on detail, sharpness, semantic correctness, and overall quality. Our analysis covers diverse architectures, including diffusion-based, GAN-based, PSNR-oriented, and general-purpose generation models, revealing critical performance disparities. Furthermore, our analysis uncovers a key evolution in failure modes that signifies a paradigm shift for the perception-oriented low-level vision field. The central challenge is evolving from the previous problem of detail scarcity (under-generation) to the new frontier of detail quality and semantic control (preventing over-generation). We also leverage our benchmark to train a new IQA model that better aligns with human perceptual judgments. Ultimately, this work provides a systematic study of modern generative image restoration models, offering crucial insights that redefine our understanding of their true state and chart a course for future development.
>
---
#### [new 007] VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VisionPangu，一个1.7B参数的紧凑多模态模型，解决图像细粒度描述生成问题。通过高效对齐和高质量监督，提升图像标题的结构化与细节表现。**

- **链接: [https://arxiv.org/pdf/2603.04957](https://arxiv.org/pdf/2603.04957)**

> **作者:** Jiaxin Fan; Wenpo Song
>
> **摘要:** Large Multimodal Models (LMMs) have achieved strong performance in vision-language understanding, yet many existing approaches rely on large-scale architectures and coarse supervision, which limits their ability to generate detailed image captions. In this work, we present VisionPangu, a compact 1.7B-parameter multimodal model designed to improve detailed image captioning through efficient multimodal alignment and high-quality supervision. Our model combines an InternVL-derived vision encoder with the OpenPangu-Embedded language backbone via a lightweight MLP projector and adopts an instruction-tuning pipeline inspired by LLaVA. By incorporating dense human-authored descriptions from the DOCCI dataset, VisionPangu improves semantic coherence and descriptive richness without relying on aggressive model scaling. Experimental results demonstrate that compact multimodal models can achieve competitive performance while producing more structured and detailed captions. The code and model weights will be publicly available at this https URL.
>
---
#### [new 008] Privacy-Aware Camera 2.0 Technical Report
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决视觉监控中的隐私与安全矛盾。通过边缘计算和AI流框架，将原始图像转换为不可逆的抽象特征，实现隐私与感知的平衡。**

- **链接: [https://arxiv.org/pdf/2603.04775](https://arxiv.org/pdf/2603.04775)**

> **作者:** Huan Song; Shuyu Tian; Ting Long; Jiang Liu; Cheng Yuan; Zhenyu Jia; Jiawei Shao; Xuelong Li
>
> **摘要:** With the increasing deployment of intelligent sensing technologies in highly sensitive environments such as restrooms and locker rooms, visual surveillance systems face a profound privacy-security paradox. Existing privacy-preserving approaches, including physical desensitization, encryption, and obfuscation, often compromise semantic understanding or fail to ensure mathematically provable irreversibility. Although Privacy Camera 1.0 eliminated visual data at the source to prevent leakage, it provided only textual judgments, leading to evidentiary blind spots in disputes. To address these limitations, this paper proposes a novel privacy-preserving perception framework based on the AI Flow paradigm and a collaborative edge-cloud architecture. By deploying a visual desensitizer at the edge, raw images are transformed in real time into abstract feature vectors through nonlinear mapping and stochastic noise injection under the Information Bottleneck principle, ensuring identity-sensitive information is stripped and original images are mathematically unreconstructable. The abstract representations are transmitted to the cloud for behavior recognition and semantic reconstruction via a "dynamic contour" visual language, achieving a critical balance between perception and privacy while enabling illustrative visual reference without exposing raw images.
>
---
#### [new 009] Fusion and Grouping Strategies in Deep Learning for Local Climate Zone Classification of Multimodal Remote Sensing Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究多模态遥感数据的本地气候区分类任务，解决数据融合与分组策略对分类精度的影响问题。通过分析不同融合与分组方法，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2603.04562](https://arxiv.org/pdf/2603.04562)**

> **作者:** Ancymol Thomas; Jaya Sreevalsan-Nair
>
> **备注:** 25 pages, 12 figures
>
> **摘要:** Local Climate Zones (LCZs) give a zoning map to study urban structures and land use and analyze the impact of urbanization on local climate. Multimodal remote sensing enables LCZ classification, for which data fusion is significant for improving accuracy owing to the data complexity. However, there is a gap in a comprehensive analysis of the fusion mechanisms used in their deep learning (DL) classifier architectures. This study analyzes different fusion strategies in the multi-class LCZ classification models for multimodal data and grouping strategies based on inherent data characteristics. The different models involving Convolutional Neural Networks (CNNs) include: (i) baseline hybrid fusion (FM1), (ii) with self- and cross-attention mechanisms (FM2), (iii) with the multi-scale Gaussian filtered images (FM3), and (iv) weighted decision-level fusion (FM4). Ablation experiments are conducted to study the pixel-, feature-, and decision-level fusion effects in the model performance. Grouping strategies include band grouping (BG) within the data modalities and label merging (LM) in the ground truth. Our analysis is exclusively done on the So2Sat LCZ42 dataset, which consists of Synthetic Aperture Radar (SAR) and Multispectral Imaging (MSI) image pairs. Our results show that FM1 consistently outperforms simple fusion methods. FM1 with BG and LM is found to be the most effective approach among all fusion strategies, giving an overall accuracy of 76.6\%. Importantly, our study highlights the effect of these strategies in improving prediction accuracy for the underrepresented classes. Our code and processed datasets are available at this https URL
>
---
#### [new 010] LAW & ORDER: Adaptive Spatial Weighting for Medical Diffusion and Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学图像分割与生成中的空间不平衡问题，提出LAW和ORDER方法，提升扩散模型生成质量与分割效率。**

- **链接: [https://arxiv.org/pdf/2603.04795](https://arxiv.org/pdf/2603.04795)**

> **作者:** Anugunj Naman; Ayushman Singh; Gaibo Zhang; Yaguang Zhang
>
> **摘要:** Medical image analysis relies on accurate segmentation, and benefits from controllable synthesis (of new training images). Yet both tasks of the cyclical pipeline face spatial imbalance: lesions occupy small regions against vast backgrounds. In particular, diffusion models have been shown to drift from prescribed lesion layouts, while efficient segmenters struggle on spatially uncertain regions. Adaptive spatial weighting addresses this by learning where to allocate computational resources. This paper introduces a pair of network adapters: 1) Learnable Adaptive Weighter (LAW) which predicts per-pixel loss modulation from features and masks for diffusion training, stabilized via a mix of normalization, clamping, and regularization to prevent degenerate solutions; and 2) Optimal Region Detection with Efficient Resolution (ORDER) which applies selective bidirectional skip attention at late decoder stages for efficient segmentation. Experiments on polyp and kidney tumor datasets demonstrate that LAW achieves 20% FID generative improvement over a uniform baseline (52.28 vs. 65.60), with synthetic data then improving downstream segmentation by 4.9% Dice coefficient (83.2% vs. 78.3%). ORDER reaches 6.0% Dice improvement on MK-UNet (81.3% vs. 75.3%) with 0.56 GFLOPs and just 42K parameters, remaining 730x smaller than the standard nnUNet.
>
---
#### [new 011] Towards Multimodal Lifelong Understanding: A Dataset and Agentic Baseline
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长期多模态学习中的记忆瓶颈和定位崩溃问题。提出MM-Lifelong数据集和ReMA模型，提升长期行为理解性能。**

- **链接: [https://arxiv.org/pdf/2603.05484](https://arxiv.org/pdf/2603.05484)**

> **作者:** Guo Chen; Lidong Lu; Yicheng Liu; Liangrui Dong; Lidong Zou; Jixin Lv; Zhenquan Li; Xinyi Mao; Baoqi Pei; Shihao Wang; Zhiqi Li; Karan Sapra; Fuxiao Liu; Yin-Dong Zheng; Yifei Huang; Limin Wang; Zhiding Yu; Andrew Tao; Guilin Liu; Tong Lu
>
> **摘要:** While datasets for video understanding have scaled to hour-long durations, they typically consist of densely concatenated clips that differ from natural, unscripted daily life. To bridge this gap, we introduce MM-Lifelong, a dataset designed for Multimodal Lifelong Understanding. Comprising 181.1 hours of footage, it is structured across Day, Week, and Month scales to capture varying temporal densities. Extensive evaluations reveal two critical failure modes in current paradigms: end-to-end MLLMs suffer from a Working Memory Bottleneck due to context saturation, while representative agentic baselines experience Global Localization Collapse when navigating sparse, month-long timelines. To address this, we propose the Recursive Multimodal Agent (ReMA), which employs dynamic memory management to iteratively update a recursive belief state, significantly outperforming existing methods. Finally, we establish dataset splits designed to isolate temporal and domain biases, providing a rigorous foundation for future research in supervised learning and out-of-distribution generalization.
>
---
#### [new 012] Lost in Translation: How Language Re-Aligns Vision for Cross-Species Pathology
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于跨物种病理学任务，旨在解决基础模型在跨癌症和跨物种转移中的性能问题。通过微调CPath-CLIP并引入语义锚定，提升癌症检测效果。**

- **链接: [https://arxiv.org/pdf/2603.04405](https://arxiv.org/pdf/2603.04405)**

> **作者:** Ekansh Arora
>
> **备注:** 27 pages, 6 figures, 7 tables. Code and data available at this https URL
>
> **摘要:** Foundation models are increasingly applied to computational pathology, yet their behavior under cross-cancer and cross-species transfer remains unspecified. This study investigated how fine-tuning CPath-CLIP affects cancer detection under same-cancer, cross-cancer, and cross-species conditions using whole-slide image patches from canine and human histopathology. Performance was measured using area under the receiver operating characteristic curve (AUC). Few-shot fine-tuning improved same-cancer (64.9% to 72.6% AUC) and cross-cancer performance (56.84% to 66.31% AUC). Cross-species evaluation revealed that while tissue matching enables meaningful transfer, performance remains below state-of-the-art benchmarks (H-optimus-0: 84.97% AUC), indicating that standard vision-language alignment is suboptimal for cross-species generalization. Embedding space analysis revealed extremely high cosine similarity (greater than 0.99) between tumor and normal prototypes. Grad-CAM shows prototype-based models remain domain-locked, while language-guided models attend to conserved tumor morphology. To address this, we introduce Semantic Anchoring, which uses language to provide a stable coordinate system for visual features. Ablation studies reveal that benefits stem from the text-alignment mechanism itself, regardless of text encoder complexity. Benchmarking against H-optimus-0 shows that CPath-CLIP's failure stems from intrinsic embedding collapse, which text alignment effectively circumvents. Additional gains were observed in same-cancer (8.52%) and cross-cancer classification (5.67%). We identified a previously uncharacterized failure mode: semantic collapse driven by species-dominated alignment rather than missing visual information. These results demonstrate that language acts as a control mechanism, enabling semantic re-interpretation without retraining.
>
---
#### [new 013] RelaxFlow: Text-Driven Amodal 3D Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本驱动的3D生成任务，解决遮挡下语义模糊问题。提出RelaxFlow框架，通过分离控制粒度，实现对未见区域的精准生成。**

- **链接: [https://arxiv.org/pdf/2603.05425](https://arxiv.org/pdf/2603.05425)**

> **作者:** Jiayin Zhu; Guoji Fu; Xiaolu Liu; Qiyuan He; Yicong Li; Angela Yao
>
> **备注:** Code: this https URL
>
> **摘要:** Image-to-3D generation faces inherent semantic ambiguity under occlusion, where partial observation alone is often insufficient to determine object category. In this work, we formalize text-driven amodal 3D generation, where text prompts steer the completion of unseen regions while strictly preserving input observation. Crucially, we identify that these objectives demand distinct control granularities: rigid control for the observation versus relaxed structural control for the prompt. To this end, we propose RelaxFlow, a training-free dual-branch framework that decouples control granularity via a Multi-Prior Consensus Module and a Relaxation Mechanism. Theoretically, we prove that our relaxation is equivalent to applying a low-pass filter on the generative vector field, which suppresses high-frequency instance details to isolate geometric structure that accommodates the observation. To facilitate evaluation, we introduce two diagnostic benchmarks, ExtremeOcc-3D and AmbiSem-3D. Extensive experiments demonstrate that RelaxFlow successfully steers the generation of unseen regions to match the prompt intent without compromising visual fidelity.
>
---
#### [new 014] GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization through EM-Guided Decomposition and Temporal Refinement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频伪造定位任务，解决弱监督与全监督之间的差异问题。提出GEM-TFL框架，通过优化标签、时间一致性修正和图结构建模提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.05095](https://arxiv.org/pdf/2603.05095)**

> **作者:** Xiaodong Zhu; Yuanming Zheng; Suting Wang; Junqi Yang; Yuhong Yang; Weiping Tu; Zhongyuan Wang
>
> **备注:** 10 pages, 4 figures, accepted by CVPR 2026
>
> **摘要:** Temporal Forgery Localization (TFL) aims to precisely identify manipulated segments within videos or audio streams, providing interpretable evidence for multimedia forensics and security. While most existing TFL methods rely on dense frame-level labels in a fully supervised manner, Weakly Supervised TFL (WS-TFL) reduces labeling cost by learning only from binary video-level labels. However, current WS-TFL approaches suffer from mismatched training and inference objectives, limited supervision from binary labels, gradient blockage caused by non-differentiable top-k aggregation, and the absence of explicit modeling of inter-proposal relationships. To address these issues, we propose GEM-TFL (Graph-based EM-powered Temporal Forgery Localization), a two-phase classification-regression framework that effectively bridges the supervision gap between training and inference. Built upon this foundation, (1) we enhance weak supervision by reformulating binary labels into multi-dimensional latent attributes through an EM-based optimization process; (2) we introduce a training-free temporal consistency refinement that realigns frame-level predictions for smoother temporal dynamics; and (3) we design a graph-based proposal refinement module that models temporal-semantic relationships among proposals for globally consistent confidence estimation. Extensive experiments on benchmark datasets demonstrate that GEM-TFL achieves more accurate and robust temporal forgery localization, substantially narrowing the gap with fully supervised methods.
>
---
#### [new 015] Layer by layer, module by module: Choose both for optimal OOD probing of ViT
- **分类: cs.CV; cs.LG; stat.ML**

- **简介: 该论文研究视觉Transformer中间层表征，解决OOD（分布外）探测问题。通过分析不同模块表现，提出按层按模块选择最优探测方式。**

- **链接: [https://arxiv.org/pdf/2603.05280](https://arxiv.org/pdf/2603.05280)**

> **作者:** Ambroise Odonnat; Vasilii Feofanov; Laetitia Chapel; Romain Tavenard; Ievgen Redko
>
> **备注:** Accepted at ICLR 2026 CAO Workshop
>
> **摘要:** Recent studies have observed that intermediate layers of foundation models often yield more discriminative representations than the final layer. While initially attributed to autoregressive pretraining, this phenomenon has also been identified in models trained via supervised and discriminative self-supervised objectives. In this paper, we conduct a comprehensive study to analyze the behavior of intermediate layers in pretrained vision transformers. Through extensive linear probing experiments across a diverse set of image classification benchmarks, we find that distribution shift between pretraining and downstream data is the primary cause of performance degradation in deeper layers. Furthermore, we perform a fine-grained analysis at the module level. Our findings reveal that standard probing of transformer block outputs is suboptimal; instead, probing the activation within the feedforward network yields the best performance under significant distribution shift, whereas the normalized output of the multi-head self-attention module is optimal when the shift is weak.
>
---
#### [new 016] Diff-ES: Stage-wise Structural Diffusion Pruning via Evolutionary Search
- **分类: cs.CV**

- **简介: 该论文属于扩散模型压缩任务，旨在解决结构化剪枝中难以平衡加速与生成质量的问题。提出Diff-ES框架，通过进化搜索优化阶段稀疏性策略，实现高效且高质量的模型剪枝。**

- **链接: [https://arxiv.org/pdf/2603.05105](https://arxiv.org/pdf/2603.05105)**

> **作者:** Zongfang Liu; Shengkun Tang; Zongliang Wu; Xin Yuan; Zhiqiang Shen
>
> **摘要:** Diffusion models have achieved remarkable success in high-fidelity image generation but remain computationally demanding due to their multi-step denoising process and large model sizes. Although prior work improves efficiency either by reducing sampling steps or by compressing model parameters, existing structured pruning approaches still struggle to balance real acceleration and image quality preservation. In particular, prior methods such as MosaicDiff rely on heuristic, manually tuned stage-wise sparsity schedules and stitch multiple independently pruned models during inference, which increases memory overhead. However, the importance of diffusion steps is highly non-uniform and model-dependent. As a result, schedules derived from simple heuristics or empirical observations often fail to generalize and may lead to suboptimal performance. To this end, we introduce \textbf{Diff-ES}, a stage-wise structural \textbf{Diff}usion pruning framework via \textbf{E}volutionary \textbf{S}earch, which optimizes the stage-wise sparsity schedule and executes it through memory-efficient weight routing without model duplication. Diff-ES divides the diffusion trajectory into multiple stages, automatically discovers an optimal stage-wise sparsity schedule via evolutionary search, and activates stage-conditioned weights dynamically without duplicating model parameters. Our framework naturally integrates with existing structured pruning methods for diffusion models including depth and width pruning. Extensive experiments on DiT and SDXL demonstrate that Diff-ES consistently achieves wall-clock speedups while incurring minimal degradation in generation quality, establishing state-of-the-art performance for structured diffusion model pruning.
>
---
#### [new 017] ORMOT: A Dataset and Framework for Omnidirectional Referring Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文提出ORMOT任务，解决传统多目标跟踪在语言描述和视野限制下的不足。构建了ORSet数据集，并提出ORTrack框架以提升长时序语言理解与跟踪效果。**

- **链接: [https://arxiv.org/pdf/2603.05384](https://arxiv.org/pdf/2603.05384)**

> **作者:** Sijia Chen; Zihan Zhou; Yanqiu Yu; En Yu; Wenbing Tao
>
> **备注:** this https URL
>
> **摘要:** Multi-Object Tracking (MOT) is a fundamental task in computer vision, aiming to track targets across video frames. Existing MOT methods perform well in general visual scenes, but face significant challenges and limitations when extended to visual-language settings. To bridge this gap, the task of Referring Multi-Object Tracking (RMOT) has recently been proposed, which aims to track objects that correspond to language descriptions. However, current RMOT methods are primarily developed on datasets captured by conventional cameras, which suffer from limited field of view. This constraint often causes targets to move out of the frame, leading to fragmented tracking and loss of contextual information. In this work, we propose a novel task, called Omnidirectional Referring Multi-Object Tracking (ORMOT), which extends RMOT to omnidirectional imagery, aiming to overcome the field-of-view (FoV) limitation of conventional datasets and improve the model's ability to understand long-horizon language descriptions. To advance the ORMOT task, we construct ORSet, an Omnidirectional Referring Multi-Object Tracking dataset, which contains 27 diverse omnidirectional scenes, 848 language descriptions, and 3,401 annotated objects, providing rich visual, temporal, and language information. Furthermore, we propose ORTrack, a Large Vision-Language Model (LVLM)-driven framework tailored for Omnidirectional Referring Multi-Object Tracking. Extensive experiments on the ORSet dataset demonstrate the effectiveness of our ORTrack framework. The dataset and code will be open-sourced at this https URL.
>
---
#### [new 018] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于长视频理解任务，解决语义漂移和计算成本高的问题。提出VideoHV-Agent框架，通过假设验证流程提升准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.04977](https://arxiv.org/pdf/2603.04977)**

> **作者:** Zheng Wang; Haoran Chen; Haoxuan Qin; Zhipeng Wei; Tianwen Qian; Cong Bai
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Long video understanding is challenging due to dense visual redundancy, long-range temporal dependencies, and the tendency of chain-of-thought and retrieval-based agents to accumulate semantic drift and correlation-driven errors. We argue that long-video reasoning should begin not with reactive retrieval, but with deliberate task formulation: the model must first articulate what must be true in the video for each candidate answer to hold. This thinking-before-finding principle motivates VideoHV-Agent, a framework that reformulates video question answering as a structured hypothesis-verification process. Based on video summaries, a Thinker rewrites answer candidates into testable hypotheses, a Judge derives a discriminative clue specifying what evidence must be checked, a Verifier grounds and tests the clue using localized, fine-grained video content, and an Answer agent integrates validated evidence to produce the final answer. Experiments on three long-video understanding benchmarks show that VideoHV-Agent achieves state-of-the-art accuracy while providing enhanced interpretability, improved logical soundness, and lower computational cost. We make our code publicly available at: this https URL.
>
---
#### [new 019] EdgeDAM: Real-time Object Tracking for Mobile Devices
- **分类: cs.CV**

- **简介: 该论文属于目标跟踪任务，旨在解决边缘设备上实时单目标跟踪的问题。提出EdgeDAM框架，通过双缓冲记忆和置信度切换策略，在保持实时性的同时提升跟踪鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.05463](https://arxiv.org/pdf/2603.05463)**

> **作者:** Syed Muhammad Raza; Syed Murtaza Hussain Abidi; Khawar Islam; Muhammad Ibrahim; Ajmal Saeed Mian
>
> **备注:** 10 pages
>
> **摘要:** Single-object tracking (SOT) on edge devices is a critical computer vision task, requiring accurate and continuous target localization across video frames under occlusion, distractor interference, and fast motion. However, recent state-of-the-art distractor-aware memory mechanisms are largely built on segmentation-based trackers and rely on mask prediction and attention-driven memory updates, which introduce substantial computational overhead and limit real-time deployment on resource-constrained hardware; meanwhile, lightweight trackers sustain high throughput but are prone to drift when visually similar distractors appear. To address these challenges, we propose EdgeDAM, a lightweight detection-guided tracking framework that reformulates distractor-aware memory for bounding-box tracking under strict edge constraints. EdgeDAM introduces two key strategies: (1) Dual-Buffer Distractor-Aware Memory (DAM), which integrates a Recent-Aware Memory to preserve temporally consistent target hypotheses and a Distractor-Resolving Memory to explicitly store hard negative candidates and penalize their re-selection during recovery; and (2) Confidence-Driven Switching with Held-Box Stabilization, where tracker reliability and temporal consistency criteria adaptively activate detection and memory-guided re-identification during occlusion, while a held-box mechanism temporarily freezes and expands the estimate to suppress distractor contamination. Extensive experiments on five benchmarks, including the distractor-focused DiDi dataset, demonstrate improved robustness under occlusion and fast motion while maintaining real-time performance on mobile devices, achieving 88.2% accuracy on DiDi and 25 FPS on an iPhone 15. Code will be released.
>
---
#### [new 020] BLINK: Behavioral Latent Modeling of NK Cell Cytotoxicity
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出BLINK模型，用于建模NK细胞毒性行为。任务是分析细胞相互作用动态，解决 cytotoxic outcome 无法从单帧准确推断的问题。工作包括构建轨迹递归状态空间模型，预测细胞凋亡并实现行为模式解析。**

- **链接: [https://arxiv.org/pdf/2603.05110](https://arxiv.org/pdf/2603.05110)**

> **作者:** Iman Nematollahi; Jose Francisco Villena-Ossa; Alina Moter; Kiana Farhadyar; Gabriel Kalweit; Abhinav Valada; Toni Cathomen; Evelyn Ullrich; Maria Kalweit
>
> **摘要:** Machine learning models of cellular interaction dynamics hold promise for understanding cell behavior. Natural killer (NK) cell cytotoxicity is a prominent example of such interaction dynamics and is commonly studied using time-resolved multi-channel fluorescence microscopy. Although tumor cell death events can be annotated at single frames, NK cytotoxic outcome emerges over time from cellular interactions and cannot be reliably inferred from frame-wise classification alone. We introduce BLINK, a trajectory-based recurrent state-space model that serves as a cell world model for NK-tumor interactions. BLINK learns latent interaction dynamics from partially observed NK-tumor interaction sequences and predicts apoptosis increments that accumulate into cytotoxic outcomes. Experiments on long-term time-lapse NK-tumor recordings show improved cytotoxic outcome detection and enable forecasting of future outcomes, together with an interpretable latent representation that organizes NK trajectories into coherent behavioral modes and temporally structured interaction phases. BLINK provides a unified framework for quantitative evaluation and structured modeling of NK cytotoxic behavior at the single-cell level.
>
---
#### [new 021] sFRC for assessing hallucinations in medical image restoration
- **分类: cs.CV; physics.med-ph; stat.ML**

- **简介: 该论文属于医学图像恢复任务，旨在解决深度学习输出中的幻觉问题。提出sFRC方法，通过傅里叶环相关分析检测幻觉，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.04673](https://arxiv.org/pdf/2603.04673)**

> **作者:** Prabhat Kc; Rongping Zeng; Nirmal Soni; Aldo Badano
>
> **备注:** 16 pages; 14 figures; 1 Supplemental document. TechRxiv Preprints, 2025
>
> **摘要:** Deep learning (DL) methods are currently being explored to restore images from sparse-view-, limited-data-, and undersampled-based acquisitions in medical applications. Although outputs from DL may appear visually appealing based on likability/subjective criteria (such as less noise, smooth features), they may also suffer from hallucinations. This issue is further exacerbated by a lack of easy-to-use techniques and robust metrics for the identification of hallucinations in DL outputs. In this work, we propose performing Fourier Ring Correlation (FRC) analysis over small patches and concomitantly (s)canning across DL outputs and their reference counterparts to detect hallucinations (termed as sFRC). We describe the rationale behind sFRC and provide its mathematical formulation. The parameters essential to sFRC may be set using predefined hallucinated features annotated by subject matter experts or using imaging theory-based hallucination maps. We use sFRC to detect hallucinations for three undersampled medical imaging problems: CT super-resolution, CT sparse view, and MRI subsampled restoration. In the testing phase, we demonstrate sFRC's effectiveness in detecting hallucinated features for the CT problem and sFRC's agreement with imaging theory-based outputs on hallucinated feature maps for the MR problem. Finally, we quantify the hallucination rates of DL methods on in-distribution versus out-of-distribution data and under increasing subsampling rates to characterize the robustness of DL methods. Beyond DL-based methods, sFRC's effectiveness in detecting hallucinations for a conventional regularization-based restoration method and a state-of-the-art unrolled method is also shown.
>
---
#### [new 022] Recognition of Daily Activities through Multi-Modal Deep Learning: A Video, Pose, and Object-Aware Approach for Ambient Assisted Living
- **分类: cs.CV**

- **简介: 该论文属于活动识别任务，旨在解决AAL系统中日常活动识别的挑战。通过融合视频、姿态和物体信息，提升识别准确性。**

- **链接: [https://arxiv.org/pdf/2603.04509](https://arxiv.org/pdf/2603.04509)**

> **作者:** Kooshan Hashemifard; Pau Climent-Pérez; Francisco Florez-Revuelta
>
> **摘要:** Recognition of daily activities is a critical element for effective Ambient Assisted Living (AAL) systems, particularly to monitor the well-being and support the independence of older adults in indoor environments. However, developing robust activity recognition systems faces significant challenges, including intra-class variability, inter-class similarity, environmental variability, camera perspectives, and scene complexity. This paper presents a multi-modal approach for the recognition of activities of daily living tailored for older adults within AAL settings. The proposed system integrates visual information processed by a 3D Convolutional Neural Network (CNN) with 3D human pose data analyzed by a Graph Convolutional Network. Contextual information, derived from an object detection module, is fused with the 3D CNN features using a cross-attention mechanism to enhance recognition accuracy. This method is evaluated using the Toyota SmartHome dataset, which consists of real-world indoor activities. The results indicate that the proposed system achieves competitive classification accuracy for a range of daily activities, highlighting its potential as an essential component for advanced AAL monitoring solutions. This advancement supports the broader goal of developing intelligent systems that promote safety and autonomy among older adults.
>
---
#### [new 023] DSA-SRGS: Super-Resolution Gaussian Splatting for Dynamic Sparse-View DSA Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动态DSA重建任务，解决稀疏视角下重建分辨率不足的问题。提出DSA-SRGS框架，结合多保真纹理学习和亚像素密度优化，提升血管细节恢复能力。**

- **链接: [https://arxiv.org/pdf/2603.04770](https://arxiv.org/pdf/2603.04770)**

> **作者:** Shiyu Zhang; Zhicong Wu; Huangxuan Zhao; Zhentao Liu; Lei Chen; Yong Luo; Lefei Zhang; Zhiming Cui; Ziwen Ke; Bo Du
>
> **备注:** 11 pages, 3 figures, 3 tables
>
> **摘要:** Digital subtraction angiography (DSA) is a key imaging technique for the auxiliary diagnosis and treatment of cerebrovascular diseases. Recent advancements in gaussian splatting and dynamic neural representations have enabled robust 3D vessel reconstruction from sparse dynamic inputs. However, these methods are fundamentally constrained by the resolution of input projections, where performing naive upsampling to enhance rendering resolution inevitably results in severe blurring and aliasing artifacts. Such lack of super-resolution capability prevents the reconstructed 4D models from recovering fine-grained vascular details and intricate branching structures, which restricts their application in precision diagnosis and treatment. To solve this problem, this paper proposes DSA-SRGS, the first super-resolution gaussian splatting framework for dynamic sparse-view DSA reconstruction. Specifically, we introduce a Multi-Fidelity Texture Learning Module that integrates high-quality priors from a fine-tuned DSA-specific super-resolution model, into the 4D reconstruction optimization. To mitigate potential hallucination artifacts from pseudo-labels, this module employs a Confidence-Aware Strategy to adaptively weight supervision signals between the original low-resolution projections and the generated high-resolution pseudo-labels. Furthermore, we develop Radiative Sub-Pixel Densification, an adaptive strategy that leverages gradient accumulation from high-resolution sub-pixel sampling to refine the 4D radiative gaussian kernels. Extensive experiments on two clinical DSA datasets demonstrate that DSA-SRGS significantly outperforms state-of-the-art methods in both quantitative metrics and qualitative visual fidelity.
>
---
#### [new 024] Evaluating and Correcting Human Annotation Bias in Dynamic Micro-Expression Recognition
- **分类: cs.CV; cs.CY**

- **简介: 该论文属于微表情识别任务，旨在解决人工标注偏差问题。通过提出GAMDSS方法，重新选择关键帧以提升模型性能，并验证了跨文化数据中的标注不确定性。**

- **链接: [https://arxiv.org/pdf/2603.04766](https://arxiv.org/pdf/2603.04766)**

> **作者:** Feng Liu; Bingyu Nan; Xuezhong Qian; Xiaolan Fu
>
> **备注:** 15 pages, 8 figures, 7 tables
>
> **摘要:** Existing manual labeling of micro-expressions is subject to errors in accuracy, especially in cross-cultural scenarios where deviation in labeling of key frames is more prominent. To address this issue, this paper presents a novel Global Anti-Monotonic Differential Selection Strategy (GAMDSS) architecture for enhancing the effectiveness of spatio-temporal modeling of micro-expressions through keyframe re-selection. Specifically, the method identifies Onset and Apex frames, which are characterized by significant micro-expression variation, from complete micro-expression action sequences via a dynamic frame reselection mechanism. It then uses these to determine Offset frames and construct a rich spatio-temporal dynamic representation. A two-branch structure with shared parameters is then used to efficiently extract spatio-temporal features. Extensive experiments are conducted on seven widely recognized micro-expression datasets. The results demonstrate that GAMDSS effectively reduces subjective errors caused by human factors in multicultural datasets such as SAMM and 4DME. Furthermore, quantitative analyses confirm that offset-frame annotations in multicultural datasets are more uncertain, providing theoretical justification for standardizing micro-expression annotations. These findings directly support our argument for reconsidering the validity and generalizability of dataset annotation paradigms. Notably, this design can be integrated into existing models without increasing the number of parameters, offering a new approach to enhancing micro-expression recognition performance. The source code is available on GitHub[this https URL].
>
---
#### [new 025] Towards 3D Scene Understanding of Gas Plumes in LWIR Hyperspectral Images Using Neural Radiance Fields
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决从LWIR高光谱图像中重建三维场景并检测气体羽流的问题。通过改进NeRF方法，减少训练图像需求并提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.05473](https://arxiv.org/pdf/2603.05473)**

> **作者:** Scout Jarman; Zigfried Hampel-Arias; Adra Carr; Kevin R. Moon
>
> **备注:** This manuscript was submitted to SPIE JARS and is under review. Code and Data can be found at this https URL and this https URL respectively. Video 1 and Video 2 can be found at this https URL and this https URL respectively
>
> **摘要:** Hyperspectral images (HSI) have many applications, ranging from environmental monitoring to national security, and can be used for material detection and identification. Longwave infrared (LWIR) HSI can be used for gas plume detection and analysis. Oftentimes, only a few images of a scene of interest are available and are analyzed individually. The ability to combine information from multiple images into a single, cohesive representation could enhance analysis by providing more context on the scene's geometry and spectral properties. Neural radiance fields (NeRFs) create a latent neural representation of volumetric scene properties that enable novel-view rendering and geometry reconstruction, offering a promising avenue for hyperspectral 3D scene reconstruction. We explore the possibility of using NeRFs to create 3D scene reconstructions from LWIR HSI and demonstrate that the model can be used for the basic downstream analysis task of gas plume detection. The physics-based DIRSIG software suite was used to generate a synthetic multi-view LWIR HSI dataset of a simple facility with a strong sulfur hexafluoride gas plume. Our method, built on the standard Mip-NeRF architecture, combines state-of-the-art methods for hyperspectral NeRFs and sparse-view NeRFs, along with a novel adaptive weighted MSE loss. Our final NeRF method requires around 50% fewer training images than the standard Mip-NeRF and achieves an average PSNR of 39.8 dB with as few as 30 training images. Gas plume detection applied to NeRF-rendered test images using the adaptive coherence estimator achieves an average AUC of 0.821 when compared with detection masks generated from ground-truth test images.
>
---
#### [new 026] MADCrowner: Margin Aware Dental Crown Design with Template Deformation and Refinement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于牙冠设计任务，解决自动化设计中精度不足与过延伸问题，提出基于模板变形与边缘分割的框架，提升几何准确性和临床可行性。**

- **链接: [https://arxiv.org/pdf/2603.04771](https://arxiv.org/pdf/2603.04771)**

> **作者:** Linda Wei; Chang Liu; Wenran Zhang; Yuxuan Hu; Ruiyang Li; Feng Qi; Changyao Tian; Ke Wang; Yuanyuan Wang; Shaoting Zhang; Dimitris Metaxas; Hongsheng Li
>
> **摘要:** Dental crown restoration is one of the most common treatment modalities for tooth defect, where personalized dental crown design is critical. While computer-aided design (CAD) systems have notably enhanced the efficiency of dental crown design, extensive manual adjustments are still required in the clinic workflow. Recent studies have explored the application of learning-based methods for the automated generation of restorative dental crowns. Nevertheless, these approaches were challenged by inadequate spatial resolution, noisy outputs, and overextension of surface reconstruction. To address these limitations, we propose \totalframework, a margin-aware mesh generation framework comprising CrownDeformR and CrownSegger. Inspired by the clinic manual workflow of dental crown design, we designed CrownDeformR to deform an initial template to the target crown based on anatomical context, which is extracted by a multi-scale intraoral scan encoder. Additionally, we introduced \marginseg, a novel margin segmentation network, to extract the cervical margin of the target tooth. The performance of CrownDeformR improved with the cervical margin as an extra constraint. And it was also utilized as the boundary condition for the tailored postprocessing method, which removed the overextended area of the reconstructed surface. We constructed a large-scale intraoral scan dataset and performed extensive experiments. The proposed method significantly outperformed existing approaches in both geometric accuracy and clinical feasibility.
>
---
#### [new 027] Evaluating GPT-5 as a Multimodal Clinical Reasoner: A Landscape Commentary
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文评估GPT-5在临床医学中的多模态推理能力，解决通用模型与专业系统在医疗任务中的性能差异问题，通过对比实验验证其在不同医学领域的表现。**

- **链接: [https://arxiv.org/pdf/2603.04763](https://arxiv.org/pdf/2603.04763)**

> **作者:** Alexandru Florea; Shansong Wang; Mingzhe Hu; Qiang Li; Zach Eidex; Luke del Balzo; Mojtaba Safari; Xiaofeng Yang
>
> **摘要:** The transition from task-specific artificial intelligence toward general-purpose foundation models raises fundamental questions about their capacity to support the integrated reasoning required in clinical medicine, where diagnosis demands synthesis of ambiguous patient narratives, laboratory data, and multimodal imaging. This landscape commentary provides the first controlled, cross-sectional evaluation of the GPT-5 family (GPT-5, GPT-5 Mini, GPT-5 Nano) against its predecessor GPT-4o across a diverse spectrum of clinically grounded tasks, including medical education examinations, text-based reasoning benchmarks, and visual question-answering in neuroradiology, digital pathology, and mammography using a standardized zero-shot chain-of-thought protocol. GPT-5 demonstrated substantial gains in expert-level textual reasoning, with absolute improvements exceeding 25 percentage-points on MedXpertQA. When tasked with multimodal synthesis, GPT-5 effectively leveraged this enhanced reasoning capacity to ground uncertain clinical narratives in concrete imaging evidence, achieving state-of-the-art or competitive performance across most VQA benchmarks and outperforming GPT-4o by margins of 10-40% in mammography tasks requiring fine-grained lesion characterization. However, performance remained moderate in neuroradiology (44% macro-average accuracy) and lagged behind domain-specific models in mammography, where specialized systems exceed 80% accuracy compared to GPT-5's 52-64%. These findings indicate that while GPT-5 represents a meaningful advance toward integrated multimodal clinical reasoning, mirroring the clinician's cognitive process of biasing uncertain information with objective findings, generalist models are not yet substitutes for purpose-built systems in highly specialized, perception-critical tasks.
>
---
#### [new 028] Person Detection and Tracking from an Overhead Crane LiDAR
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测与跟踪任务，解决工业场景下从吊车LiDAR数据中检测和跟踪人员的问题。通过构建数据集并优化检测模型，提升检测精度与实时性。**

- **链接: [https://arxiv.org/pdf/2603.04938](https://arxiv.org/pdf/2603.04938)**

> **作者:** Nilusha Jayawickrama; Henrik Toikka; Risto Ojala
>
> **备注:** 8 pages, 7 figures, 4 tables. Submitted to Ubiquitous Robots (UR) 2026. Code: this https URL
>
> **摘要:** This paper investigates person detection and tracking in an industrial indoor workspace using a LiDAR mounted on an overhead crane. The overhead viewpoint introduces a strong domain shift from common vehicle-centric LiDAR benchmarks, and limited availability of suitable public training data. Henceforth, we curate a site-specific overhead LiDAR dataset with 3D human bounding-box annotations and adapt selected candidate 3D detectors under a unified training and evaluation protocol. We further integrate lightweight tracking-by-detection using AB3DMOT and SimpleTrack to maintain person identities over time. Detection performance is reported with distance-sliced evaluation to quantify the practical operating envelope of the sensing setup. The best adapted detector configurations achieve average precision (AP) up to 0.84 within a 5.0 m horizontal radius, increasing to 0.97 at 1.0 m, with VoxelNeXt and SECOND emerging as the most reliable backbones across this range. The acquired results contribute in bridging the domain gap between standard driving datasets and overhead sensing for person detection and tracking. We also report latency measurements, highlighting practical real-time feasibility. Finally, we release our dataset and implementations in GitHub to support further research
>
---
#### [new 029] BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决事件相机与图像融合中的噪声耦合问题。通过提出BiEvLight框架，实现事件去噪与增强的协同优化。**

- **链接: [https://arxiv.org/pdf/2603.04975](https://arxiv.org/pdf/2603.04975)**

> **作者:** Zishu Yao; Xiang-Xiang Su; Shengning Zhou; Guang-Yong Chen; Guodong Fan; Xing Chen
>
> **摘要:** Event cameras, with their high dynamic range, show great promise for Low-light Image Enhancement (LLIE). Existing works primarily focus on designing effective modal fusion strategies. However, a key challenge is the dual degradation from intrinsic background activity (BA) noise in events and low signal-to-noise ratio (SNR) in images, which causes severe noise coupling during modal fusion, creating a critical performance bottleneck. We therefore posit that precise event denoising is the prerequisite to unlocking the full potential of event-based fusion. To this end, we propose BiEvLight, a hierarchical and task-aware framework that collaboratively optimizes enhancement and denoising by exploiting their intrinsic interdependence. Specifically, BiEvLight exploits the strong gradient correlation between images and events to build a gradient-guided event denoising prior that alleviates insufficient denoising in heavily noisy regions. Moreover, instead of treating event denoising as a static pre-processing stage-which inevitably incurs a trade-off between over- and under-denoising and cannot adapt to the requirements of a specific enhancement objective-we recast it as a bilevel optimization problem constrained by the enhancement task. Through cross-task interaction, the upper-level denoising problem learns event representations tailored to the lower-level enhancement objective, thereby substantially improving overall enhancement quality. Extensive experiments on the Real-world noise Dataset SDE demonstrate that our method significantly outperforms state-of-the-art (SOTA) approaches, with average improvements of 1.30dB in PSNR, 2.03dB in PSNR* and 0.047 in SSIM, respectively. The code will be publicly available at this https URL.
>
---
#### [new 030] Spinverse: Differentiable Physics for Permeability-Aware Microstructure Reconstruction from Diffusion MRI
- **分类: cs.CV; cs.LG; q-bio.QM**

- **简介: 该论文提出Spinverse，用于从扩散MRI重建微结构。解决的是如何恢复显式界面而非仅估计参数的问题，通过可微物理模拟实现渗透性感知的重建。**

- **链接: [https://arxiv.org/pdf/2603.04638](https://arxiv.org/pdf/2603.04638)**

> **作者:** Prathamesh Pradeep Khole; Mario M. Brenes; Zahra Kais Petiwala; Ehsan Mirafzali; Utkarsh Gupta; Jing-Rebecca Li; Andrada Ianus; Razvan Marinescu
>
> **备注:** 10 Pages, 5 Figures, 2 Tables
>
> **摘要:** Diffusion MRI (dMRI) is sensitive to microstructural barriers, yet most existing methods either assume impermeable boundaries or estimate voxel-level parameters without recovering explicit interfaces. We present Spinverse, a permeability-aware reconstruction method that inverts dMRI measurements through a fully differentiable Bloch-Torrey simulator. Spinverse represents tissue on a fixed tetrahedral grid and treats each interior face permeability as a learnable parameter; low-permeability faces act as diffusion barriers, so microstructural boundaries whose topology is not fixed a priori (up to the resolution of the ambient mesh) emerge without changing mesh connectivity or vertex positions. Given a target signal, we optimize face permeabilities by backpropagating a signal-matching loss through the PDE forward model, and recover an interface by thresholding the learned permeability field. To mitigate the ill-posedness of permeability inversion, we use mesh-based geometric priors; to avoid local minima, we use a staged multi-sequence optimization curriculum. Across a collection of synthetic voxel meshes, Spinverse reconstructs diverse geometries and demonstrates that sequence scheduling and regularization are critical to avoid outline-only solutions while improving both boundary accuracy and structural validity.
>
---
#### [new 031] Generic Camera Calibration using Blurry Images
- **分类: cs.CV**

- **简介: 该论文属于相机标定任务，旨在解决使用模糊图像进行通用相机标定的问题。通过几何约束和局部参数化光照模型，同时估计特征位置和点扩散函数。**

- **链接: [https://arxiv.org/pdf/2603.05159](https://arxiv.org/pdf/2603.05159)**

> **作者:** Zezhun Shi
>
> **摘要:** Camera calibration is the foundation of 3D vision. Generic camera calibration can yield more accurate results than parametric cam era calibration. However, calibrating a generic camera model using printed calibration boards requires far more images than parametric calibration, making motion blur practically unavoidable for individual users. As a f irst attempt to address this problem, we draw on geometric constraints and a local parametric illumination model to simultaneously estimate feature locations and spatially varying point spread functions, while re solving the translational ambiguity that need not be considered in con ventional image deblurring tasks. Experimental results validate the ef fectiveness of our approach.
>
---
#### [new 032] Revisiting an Old Perspective Projection for Monocular 3D Morphable Models Regression
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D人脸重建任务，旨在解决单目视频中近距拍摄时的透视失真问题。通过改进投影模型，提升3DMM在头戴设备视频中的拟合效果。**

- **链接: [https://arxiv.org/pdf/2603.04958](https://arxiv.org/pdf/2603.04958)**

> **作者:** Toby Chong; Ryota Nakajima
>
> **备注:** WACV 2026, this https URL
>
> **摘要:** We introduce a novel camera model for monocular 3D Morphable Model (3DMM) regression methods that effectively captures the perspective distortion effect commonly seen in close-up facial images. Fitting 3D morphable models to video is a key technique in content creation. In particular, regression-based approaches have produced fast and accurate results by matching the rendered output of the morphable model to the target image. These methods typically achieve stable performance with orthographic projection, which eliminates the ambiguity between focal length and object distance. However, this simplification makes them unsuitable for close-up footage, such as that captured with head-mounted cameras. We extend orthographic projection with a new shrinkage parameter, incorporating a pseudo-perspective effect while preserving the stability of the original projection. We present several techniques that allow finetuning of existing models, and demonstrate the effectiveness of our modification through both quantitative and qualitative comparisons using a custom dataset recorded with head-mounted cameras.
>
---
#### [new 033] Orthogonal Spatial-temporal Distributional Transfer for 4D Generation
- **分类: cs.CV**

- **简介: 该论文属于4D生成任务，解决缺乏大规模4D数据的问题。通过迁移3D和视频模型的先验，提出STD-4D扩散模型与Orster机制，提升4D合成质量。**

- **链接: [https://arxiv.org/pdf/2603.05081](https://arxiv.org/pdf/2603.05081)**

> **作者:** Wei Liu; Shengqiong Wu; Bobo Li; Haoyu Zhao; Hao Fei; Mong-Li Lee; Wynne Hsu
>
> **备注:** 9 pages, 6 figures, 3 tables, AAAI
>
> **摘要:** In the AIGC era, generating high-quality 4D content has garnered increasing research attention. Unfortunately, current 4D synthesis research is severely constrained by the lack of large-scale 4D datasets, preventing models from adequately learning the critical spatial-temporal features necessary for high-quality 4D generation, thus hindering progress in this domain. To combat this, we propose a novel framework that transfers rich spatial priors from existing 3D diffusion models and temporal priors from video diffusion models to enhance 4D synthesis. We develop a spatial-temporal-disentangled 4D (STD-4D) Diffusion model, which synthesizes 4D-aware videos through disentangled spatial and temporal latents. To facilitate the best feature transfer, we design a novel Orthogonal Spatial-temporal Distributional Transfer (Orster) mechanism, where the spatiotemporal feature distributions are carefully modeled and injected into the STD-4D Diffusion. Furthermore, during the 4D construction, we devise a spatial-temporal-aware HexPlane (ST-HexPlane) to integrate the transferred spatiotemporal features, thereby improving 4D deformation and 4D Gaussian feature modeling. Experiments demonstrate that our method significantly outperforms existing approaches, achieving superior spatial-temporal consistency and higher-quality 4D synthesis.
>
---
#### [new 034] FC-VFI: Faithful and Consistent Video Frame Interpolation for High-FPS Slow Motion Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频帧插值任务，旨在解决高帧率慢动作视频生成中细节丢失和运动不一致的问题。提出FC-VFI方法，提升帧率并保持视觉质量和运动一致性。**

- **链接: [https://arxiv.org/pdf/2603.04899](https://arxiv.org/pdf/2603.04899)**

> **作者:** Ganggui Ding; Hao Chen; Xiaogang Xu
>
> **备注:** ICASSP2026
>
> **摘要:** Large pre-trained video diffusion models excel in video frame interpolation but struggle to generate high fidelity frames due to reliance on intrinsic generative priors, limiting detail preservation from start and end frames. Existing methods often depend on motion control for temporal consistency, yet dense optical flow is error-prone, and sparse points lack structural context. In this paper, we propose FC-VFI for faithful and consistent video frame interpolation, supporting \(4\times\)x and \(8\times\) interpolation, boosting frame rates from 30 FPS to 120 and 240 FPS at \(2560\times 1440\)resolution while preserving visual fidelity and motion consistency. We introduce a temporal modeling strategy on the latent sequences to inherit fidelity cues from start and end frames and leverage semantic matching lines for structure-aware motion guidance, improving motion consistency. Furthermore, we propose a temporal difference loss to mitigate temporal inconsistencies. Extensive experiments show FC-VFI achieves high performance and structural integrity across diverse scenarios.
>
---
#### [new 035] DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于视频音频篡改定位任务，解决模糊边界和长距离建模问题。提出DeformTrace，结合可变形机制与中继令牌，提升定位精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.04882](https://arxiv.org/pdf/2603.04882)**

> **作者:** Xiaodong Zhu; Suting Wang; Yuanming Zheng; Junqi Yang; Yangxu Liao; Yuhong Yang; Weiping Tu; Zhongyuan Wang
>
> **备注:** 9 pages, 4 figures, accepted by AAAI 2026
>
> **摘要:** Temporal Forgery Localization (TFL) aims to precisely identify manipulated segments in video and audio, offering strong interpretability for security and forensics. While recent State Space Models (SSMs) show promise in precise temporal reasoning, their use in TFL is hindered by ambiguous boundaries, sparse forgeries, and limited long-range modeling. We propose DeformTrace, which enhances SSMs with deformable dynamics and relay mechanisms to address these challenges. Specifically, Deformable Self-SSM (DS-SSM) introduces dynamic receptive fields into SSMs for precise temporal localization. To further enhance its capacity for temporal reasoning and mitigate long-range decay, a Relay Token Mechanism is integrated into DS-SSM. Besides, Deformable Cross-SSM (DC-SSM) partitions the global state space into query-specific subspaces, reducing non-forgery information accumulation and boosting sensitivity to sparse forgeries. These components are integrated into a hybrid architecture that combines the global modeling of Transformers with the efficiency of SSMs. Extensive experiments show that DeformTrace achieves state-of-the-art performance with fewer parameters, faster inference, and stronger robustness.
>
---
#### [new 036] Digital Twin Driven Textile Classification and Foreign Object Recognition in Automated Sorting Systems
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于纺织品分类与异物检测任务，解决自动化分拣中的变形衣物识别和复杂环境下的异物检测问题。工作包括构建数字孪生系统，融合视觉语言模型与感知技术实现高效分拣。**

- **链接: [https://arxiv.org/pdf/2603.05230](https://arxiv.org/pdf/2603.05230)**

> **作者:** Serkan Ergun; Tobias Mitterer; Hubert Zangl
>
> **备注:** 10 pages,single column, 5 figures, preprint for Photomet Edumet 2026 (Klagenfurt, Austria)
>
> **摘要:** The increasing demand for sustainable textile recycling requires robust automation solutions capable of handling deformable garments and detecting foreign objects in cluttered environments. This work presents a digital twin driven robotic sorting system that integrates grasp prediction, multi modal perception, and semantic reasoning for real world textile classification. A dual arm robotic cell equipped with RGBD sensing, capacitive tactile feedback, and collision-aware motion planning autonomously separates garments from an unsorted basket, transfers them to an inspection zone, and classifies them using state of the art Visual Language Models (VLMs). We benchmark nine VLM s from five model families on a dataset of 223 inspection scenarios comprising shirts, socks, trousers, underwear, foreign objects (including garments outside of the aforementioned classes), and empty scenes. The evaluation assesses per class accuracy, hallucination behavior, and computational performance under practical hardware constraints. Results show that the Qwen model family achieves the highest overall accuracy (up to 87.9 %), with strong foreign object detection performance, while lighter models such as Gemma3 offer competitive speed accuracy trade offs for edge deployment. A digital twin combined with MoveIt enables collision aware path planning and integrates segmented 3D point clouds of inspected garments into the virtual environment for improved manipulation reliability. The presented system demonstrates the feasibility of combining semantic VLM reasoning with conventional grasp detection and digital twin technology for scalable, autonomous textile sorting in realistic industrial settings.
>
---
#### [new 037] Structure-Guided Histopathology Synthesis via Dual-LoRA Diffusion
- **分类: cs.CV**

- **简介: 该论文属于病理图像合成任务，解决结构一致性与细胞组织真实性的难题。提出Dual-LoRA扩散框架，统一完成局部补全与全局合成，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.04565](https://arxiv.org/pdf/2603.04565)**

> **作者:** Xuan Xu; Prateek Prasanna
>
> **摘要:** Histopathology image synthesis plays an important role in tissue restoration, data augmentation, and modeling of tumor microenvironments. However, existing generative methods typically address restoration and generation as separate tasks, although both share the same objective of structure-consistent tissue synthesis under varying degrees of missingness, and often rely on weak or inconsistent structural priors that limit realistic cellular organization. We propose Dual-LoRA Controllable Diffusion, a unified centroid-guided diffusion framework that jointly supports Local Structure Completion and Global Structure Synthesis within a single model. Multi-class nuclei centroids serve as lightweight and annotation-efficient spatial priors, providing biologically meaningful guidance under both partial and complete image absence. Two task-specific LoRA adapters specialize the shared backbone for local and global objectives without retraining separate diffusion models. Extensive experiments demonstrate consistent improvements over state-of-the-art GAN and diffusion baselines across restoration and synthesis tasks. For local completion, LPIPS computed within the masked region improves from 0.1797 (HARP) to 0.1524, and for global synthesis, FID improves from 225.15 (CoSys) to 76.04, indicating improved structural fidelity and realism. Our approach achieves more faithful structural recovery in masked regions and substantially improved realism and morphology consistency in full synthesis, supporting scalable pan-cancer histopathology modeling.
>
---
#### [new 038] Mask-aware inference with State-Space Models
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉领域，解决无效数据处理问题。针对SSM在推理时无法处理缺失区域的缺陷，提出Partial Vision Mamba，提升深度补全、图像修复等任务性能。**

- **链接: [https://arxiv.org/pdf/2603.04568](https://arxiv.org/pdf/2603.04568)**

> **作者:** Ignasi Mas; Ramon Morros; Javier-Ruiz Hidalgo; Ivan Huerta
>
> **摘要:** Many real-world computer vision tasks, such as depth completion, must handle inputs with arbitrarily shaped regions of missing or invalid data. For Convolutional Neural Networks (CNNs), Partial Convolutions solved this by a mask-aware re-normalization conditioned only on valid pixels. Recently, State Space Models (SSMs) like Mamba have emerged, offering high performance with linear complexity. However, these architectures lack an inherent mechanism for handling such arbitrarily shaped invalid data at inference time. To bridge this gap, we introduce Partial Vision Mamba (PVM), a novel architectural component that ports the principles of partial operations to the Mamba backbone. We also define a series of rules to design architectures using PVM. We show the efficacy and generalizability of our approach in the tasks of depth completion, image inpainting, and classification with invalid data.
>
---
#### [new 039] Generalizable Multiscale Segmentation of Heterogeneous Map Collections
- **分类: cs.CV**

- **简介: 该论文属于地图语义分割任务，旨在解决历史地图多样性带来的识别难题。提出Semap数据集和多尺度分割框架，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.05037](https://arxiv.org/pdf/2603.05037)**

> **作者:** Remi Petitpierre
>
> **备注:** 30 pages, 15 figures
>
> **摘要:** Historical map collections are highly diverse in style, scale, and geographic focus, often consisting of many single-sheet documents. Yet most work in map recognition focuses on specialist models tailored to homogeneous map series. In contrast, this article aims to develop generalizable semantic segmentation models and ontology. First, we introduce Semap, a new open benchmark dataset comprising 1,439 manually annotated patches designed to reflect the variety of historical map documents. Second, we present a segmentation framework that combines procedural data synthesis with multiscale integration to improve robustness and transferability. This framework achieves state-of-the-art performance on both the HCMSSD and Semap datasets, showing that a diversity-driven approach to map recognition is not only viable but also beneficial. The results indicate that segmentation performance remains largely stable across map collections, scales, geographic regions, and publication contexts. By proposing benchmark datasets and methods for the generic segmentation of historical maps, this work opens the way to integrating the long tail of cartographic archives to historical geographic studies.
>
---
#### [new 040] MultiGO++: Monocular 3D Clothed Human Reconstruction via Geometry-Texture Collaboration
- **分类: cs.CV**

- **简介: 该论文属于单目3D着装人体重建任务，旨在解决纹理和几何信息不足及单一模态监督的问题。提出MultiGO++框架，通过多源纹理合成、区域感知形状提取和双通道U-Net提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.04993](https://arxiv.org/pdf/2603.04993)**

> **作者:** Nanjie Yao; Gangjian Zhang; Wenhao Shen; Jian Shu; Yu Feng; Hao Wang
>
> **摘要:** Monocular 3D clothed human reconstruction aims to generate a complete and realistic textured 3D avatar from a single image. Existing methods are commonly trained under multi-view supervision with annotated geometric priors, and during inference, these priors are estimated by the pre-trained network from the monocular input. These methods are constrained by three key limitations: texturally by unavailability of training data, geometrically by inaccurate external priors, and systematically by biased single-modality supervision, all leading to suboptimal reconstruction. To address these issues, we propose a novel reconstruction framework, named MultiGO++, which achieves effective systematic geometry-texture collaboration. It consists of three core parts: (1) A multi-source texture synthesis strategy that constructs 15,000+ 3D textured human scans to improve the performance on texture quality estimation in challenge scenarios; (2) A region-aware shape extraction module that extracts and interacts features of each body region to obtain geometry information and a Fourier geometry encoder that mitigates the modality gap to achieve effective geometry learning; (3) A dual reconstruction U-Net that leverages geometry-texture collaborative features to refine and generate high-fidelity textured 3D human meshes. Extensive experiments on two benchmarks and many in-the-wild cases show the superiority of our method over state-of-the-art approaches.
>
---
#### [new 041] Accelerating Text-to-Video Generation with Calibrated Sparse Attention
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决扩散模型运行慢的问题。通过引入CalibAtt方法，利用稀疏注意力加速生成过程，提升效率同时保持质量。**

- **链接: [https://arxiv.org/pdf/2603.05503](https://arxiv.org/pdf/2603.05503)**

> **作者:** Shai Yehezkel; Shahar Yadin; Noam Elata; Yaron Ostrovsky-Berman; Bahjat Kawar
>
> **摘要:** Recent diffusion models enable high-quality video generation, but suffer from slow runtimes. The large transformer-based backbones used in these models are bottlenecked by spatiotemporal attention. In this paper, we identify that a significant fraction of token-to-token connections consistently yield negligible scores across various inputs, and their patterns often repeat across queries. Thus, the attention computation in these cases can be skipped with little to no effect on the result. This observation continues to hold for connections among local token blocks. Motivated by this, we introduce CalibAtt, a training-free method that accelerates video generation via calibrated sparse attention. CalibAtt performs an offline calibration pass that identifies block-level sparsity and repetition patterns that are stable across inputs, and compiles these patterns into optimized attention operations for each layer, head, and diffusion timestep. At inference time, we compute the selected input-dependent connections densely, and skip the unselected ones in a hardware-efficient manner. Extensive experiments on Wan 2.1 14B, Mochi 1, and few-step distilled models at various resolutions show that CalibAtt achieves up to 1.58x end-to-end speedup, outperforming existing training-free methods while maintaining video generation quality and text-video alignment.
>
---
#### [new 042] RealWonder: Real-Time Physical Action-Conditioned Video Generation
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文提出RealWonder，解决视频生成中缺乏物理交互的问题。通过物理模拟生成视频，实现实时动作条件视频生成。**

- **链接: [https://arxiv.org/pdf/2603.05449](https://arxiv.org/pdf/2603.05449)**

> **作者:** Wei Liu; Ziyu Chen; Zizhang Li; Yue Wang; Hong-Xing Yu; Jiajun Wu
>
> **备注:** The first two authors contributed equally. The last two authors advised equally. Project website: this https URL
>
> **摘要:** Current video generation models cannot simulate physical consequences of 3D actions like forces and robotic manipulations, as they lack structural understanding of how actions affect 3D scenes. We present RealWonder, the first real-time system for action-conditioned video generation from a single image. Our key insight is using physics simulation as an intermediate bridge: instead of directly encoding continuous actions, we translate them through physics simulation into visual representations (optical flow and RGB) that video models can process. RealWonder integrates three components: 3D reconstruction from single images, physics simulation, and a distilled video generator requiring only 4 diffusion steps. Our system achieves 13.2 FPS at 480x832 resolution, enabling interactive exploration of forces, robot actions, and camera controls on rigid objects, deformable bodies, fluids, and granular materials. We envision RealWonder opens new opportunities to apply video models in immersive experiences, AR/VR, and robot learning. Our code and model weights are publicly available in our project website: this https URL
>
---
#### [new 043] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多模态脑肿瘤分割任务，解决联邦学习中的模态异质性和个性化需求问题。提出FedMEPD框架，采用联邦模态编码器和部分个性化融合解码器，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.04887](https://arxiv.org/pdf/2603.04887)**

> **作者:** Hong Liu; Dong Wei; Qian Dai; Xian Wu; Yefeng Zheng; Liansheng Wang
>
> **备注:** Medical Image Analysis 2025. arXiv admin note: substantial text overlap with arXiv:2403.11803
>
> **摘要:** Most existing federated learning (FL) methods for medical image analysis only considered intramodal heterogeneity, limiting their applicability to multimodal imaging applications. In practice, some FL participants may possess only a subset of the complete imaging modalities, posing intermodal heterogeneity as a challenge to effectively training a global model on all participants' data. Meanwhile, each participant expects a personalized model tailored to its local data characteristics in FL. This work proposes a new FL framework with federated modality-specific encoders and partially personalized multimodal fusion decoders (FedMEPD) to address the two concurrent issues. Specifically, FedMEPD employs an exclusive encoder for each modality to account for the intermodal heterogeneity. While these encoders are fully federated, the decoders are partially personalized to meet individual needs -- using the discrepancy between global and local parameter updates to dynamically determine which decoder filters are personalized. Implementation-wise, a server with full-modal data employs a fusion decoder to fuse representations from all modality-specific encoders, thus bridging the modalities to optimize the encoders via backpropagation. Moreover, multiple anchors are extracted from the fused multimodal representations and distributed to the clients in addition to the model parameters. Conversely, the clients with incomplete modalities calibrate their missing-modal representations toward the global full-modal anchors via scaled dot-product cross-attention, making up for the information loss due to absent modalities. FedMEPD is validated on the BraTS 2018 and 2020 multimodal brain tumor segmentation benchmarks. Results show that it outperforms various up-to-date methods for multimodal and personalized FL, and its novel designs are effective.
>
---
#### [new 044] Mario: Multimodal Graph Reasoning with Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出Mario框架，解决多模态图推理问题。针对跨模态一致性弱和模态偏好差异，通过图条件VLM和自适应指令调优实现有效推理。**

- **链接: [https://arxiv.org/pdf/2603.05181](https://arxiv.org/pdf/2603.05181)**

> **作者:** Yuanfu Sun; Kang Li; Pengkang Guo; Jiajin Liu; Qiaoyu Tan
>
> **备注:** CVPR 2026
>
> **摘要:** Recent advances in large language models (LLMs) have opened new avenues for multimodal reasoning. Yet, most existing methods still rely on pretrained vision-language models (VLMs) to encode image-text pairs in isolation, ignoring the relational structure that real-world multimodal data naturally form. This motivates reasoning on multimodal graphs (MMGs), where each node has textual and visual attributes and edges provide structural cues. Enabling LLM-based reasoning on such heterogeneous multimodal signals while preserving graph topology introduces two key challenges: resolving weak cross-modal consistency and handling heterogeneous modality preference. To address this, we propose Mario, a unified framework that simultaneously resolves the two above challenges and enables effective LLM-based reasoning over MMGs. Mario consists of two innovative stages. Firstly, a graph-conditioned VLM design that jointly refines textual and visual features through fine-grained cross-modal contrastive learning guided by graph topology. Secondly, a modality-adaptive graph instruction tuning mechanism that organizes aligned multimodal features into graph-aware instruction views and employs a learnable router to surface, for each node and its neighborhood, the most informative modality configuration to the LLM. Extensive experiments across diverse MMG benchmarks demonstrate that Mario consistently outperforms state-of-the-art graph models in both supervised and zero-shot scenarios for node classification and link prediction. The code will be made available at this https URL.
>
---
#### [new 045] Fusion4CA: Boosting 3D Object Detection via Comprehensive Image Exploitation
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决现有方法过度依赖LiDAR数据、忽视RGB信息的问题。通过引入对比对齐模块和视觉辅助分支，提升RGB信息利用效率。**

- **链接: [https://arxiv.org/pdf/2603.05305](https://arxiv.org/pdf/2603.05305)**

> **作者:** Kang Luo; Xin Chen; Yangyi Xiao; Hesheng Wang
>
> **摘要:** Nowadays, an increasing number of works fuse LiDAR and RGB data in the bird's-eye view (BEV) space for 3D object detection in autonomous driving systems. However, existing methods suffer from over-reliance on the LiDAR branch, with insufficient exploration of RGB information. To tackle this issue, we propose Fusion4CA, which is built upon the classic BEVFusion framework and dedicated to fully exploiting visual input with plug-and-play components. Specifically, a contrastive alignment module is designed to calibrate image features with 3D geometry, and a camera auxiliary branch is introduced to mine RGB information sufficiently during training. For further performance enhancement, we leverage an off-the-shelf cognitive adapter to make the most of pretrained image weights, and integrate a standard coordinate attention module into the fusion stage as a supplementary boost. Experiments on the nuScenes dataset demonstrate that our method achieves 69.7% mAP with only 6 training epochs and a mere 3.48% increase in inference parameters, yielding a 1.2% improvement over the baseline which is fully trained for 20 epochs. Extensive experiments in a simulated lunar environment further validate the effectiveness and generalization of our method. Our code will be released through Fusion4CA.
>
---
#### [new 046] Meta-D: Metadata-Aware Architectures for Brain Tumor Analysis and Missing-Modality Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在提升脑肿瘤分割性能。通过引入元数据引导特征提取，解决数据缺失时的分割难题。**

- **链接: [https://arxiv.org/pdf/2603.04811](https://arxiv.org/pdf/2603.04811)**

> **作者:** SangHyuk Kim; Daniel Haehn; Sumientra Rampersad
>
> **备注:** 9 pages, 2 figures, 3 tables
>
> **摘要:** We present Meta-D, an architecture that explicitly leverages categorical scanner metadata such as MRI sequence and plane orientation to guide feature extraction for brain tumor analysis. We aim to improve the performance of medical image deep learning pipelines by integrating explicit metadata to stabilize feature representations. We first evaluate this in 2D tumor detection, where injecting sequence (e.g., T1, T2) and plane (e.g., axial) metadata dynamically modulates convolutional features, yielding an absolute increase of up to 2.62% in F1-score over image-only baselines. Because metadata grounds feature extraction when data are available, we hypothesize it can serve as a robust anchor when data are missing. We apply this to 3D missing-modality tumor segmentation. Our Transformer Maximizer utilizes metadata-based cross-attention to isolate and route available modalities, ensuring the network focuses on valid slices. This targeted attention improves brain tumor segmentation Dice scores by up to 5.12% under extreme modality scarcity while reducing model parameters by 24.1%.
>
---
#### [new 047] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning
- **分类: cs.CV**

- **简介: 该论文提出FaceCam，解决单目人像视频的相机控制问题。通过尺度感知表示和数据生成策略，提升视频质量与控制能力。**

- **链接: [https://arxiv.org/pdf/2603.05506](https://arxiv.org/pdf/2603.05506)**

> **作者:** Weijie Lyu; Ming-Hsuan Yang; Zhixin Shu
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** We introduce FaceCam, a system that generates video under customizable camera trajectories for monocular human portrait video input. Recent camera control approaches based on large video-generation models have shown promising progress but often exhibit geometric distortions and visual artifacts on portrait videos due to scale-ambiguous camera representations or 3D reconstruction errors. To overcome these limitations, we propose a face-tailored scale-aware representation for camera transformations that provides deterministic conditioning without relying on 3D priors. We train a video generation model on both multi-view studio captures and in-the-wild monocular videos, and introduce two camera-control data generation strategies: synthetic camera motion and multi-shot stitching, to exploit stationary training cameras while generalizing to dynamic, continuous camera trajectories at inference time. Experiments on Ava-256 dataset and diverse in-the-wild videos demonstrate that FaceCam achieves superior performance in camera controllability, visual quality, identity and motion preservation.
>
---
#### [new 048] Structure Observation Driven Image-Text Contrastive Learning for Computed Tomography Report Generation
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决CT图像与报告之间的语义对齐问题。提出两阶段框架，通过结构级图文对比学习提升报告生成效果。**

- **链接: [https://arxiv.org/pdf/2603.04878](https://arxiv.org/pdf/2603.04878)**

> **作者:** Hong Liu; Dong Wei; Qiong Peng; Yawen Huang; Xian Wu; Yefeng Zheng; Liansheng Wang
>
> **备注:** Accept to IPMI 2025
>
> **摘要:** Computed Tomography Report Generation (CTRG) aims to automate the clinical radiology reporting process, thereby reducing the workload of report writing and facilitating patient care. While deep learning approaches have achieved remarkable advances in X-ray report generation, their effectiveness may be limited in CTRG due to larger data volumes of CT images and more intricate details required to describe them. This work introduces a novel two-stage (structure- and report-learning) framework tailored for CTRG featuring effective structure-wise image-text contrasting. In the first stage, a set of learnable structure-specific visual queries observe corresponding structures in a CT image. The resulting observation tokens are contrasted with structure-specific textual features extracted from the accompanying radiology report with a structure-wise image-text contrastive loss. In addition, text-text similarity-based soft pseudo targets are proposed to mitigate the impact of false negatives, i.e., semantically identical image structures and texts from non-paired images and reports. Thus, the model learns structure-level semantic correspondences between CT images and reports. Further, a dynamic, diversity-enhanced negative queue is proposed to guide the network in learning to discriminate various abnormalities. In the second stage, the visual structure queries are frozen and used to select the critical image patch embeddings depicting each anatomical structure, minimizing distractions from irrelevant areas while reducing memory consumption. Also, a text decoder is added and trained for report this http URL extensive experiments on two public datasets demonstrate that our framework establishes new state-of-the-art performance for CTRG in clinical efficiency, and its components are effective.
>
---
#### [new 049] Act, Think or Abstain: Complexity-Aware Adaptive Inference for Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决计算复杂度高和推理效率低的问题。通过自适应框架动态选择执行策略，提升效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.05147](https://arxiv.org/pdf/2603.05147)**

> **作者:** Riccardo Andrea Izzo; Gianluca Bardaro; Matteo Matteucci
>
> **摘要:** Current research on Vision-Language-Action (VLA) models predominantly focuses on enhancing generalization through established reasoning techniques. While effective, these improvements invariably increase computational complexity and inference latency. Furthermore, these mechanisms are typically applied indiscriminately, resulting in the inefficient allocation of resources for trivial tasks while simultaneously failing to provide the uncertainty estimation necessary to prevent catastrophic failure on out-of-distribution tasks. Inspired by human cognition, we propose an adaptive framework that dynamically routes VLA execution based on the complexity of the perceived state. Our approach transforms the VLA's vision-language backbone into an active detection tool by projecting latent embeddings into an ensemble of parametric and non-parametric estimators. This allows the system to execute known tasks immediately (Act), reason about ambiguous scenarios (Think), and preemptively halt execution when encountering significant physical or semantic anomalies (Abstain). In our empirical analysis, we observe a phenomenon where visual embeddings alone are superior for inferring task complexity due to the semantic invariance of language. Evaluated on the LIBERO and LIBERO-PRO benchmarks as well as on a real robot, our vision-only configuration achieves 80% F1-Score using as little as 5% of training data, establishing itself as a reliable and efficient task complexity detector.
>
---
#### [new 050] SRasP: Self-Reorientation Adversarial Style Perturbation for Cross-Domain Few-Shot Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于跨域小样本学习任务，旨在解决领域迁移中的分布偏移问题。提出SRasP方法，通过全局语义引导的风格扰动提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.05135](https://arxiv.org/pdf/2603.05135)**

> **作者:** Wenqian Li; Pengfei Fang; Hui Xue
>
> **摘要:** Cross-Domain Few-Shot Learning (CD-FSL) aims to transfer knowledge from a seen source domain to unseen target domains, serving as a key benchmark for evaluating the robustness and transferability of models. Existing style-based perturbation methods mitigate domain shift but often suffer from gradient instability and convergence to sharp this http URL address these limitations, we propose a novel crop-global style perturbation network, termed Self-Reorientation Adversarial \underline{S}tyle \underline{P}erturbation (SRasP). Specifically, SRasP leverages global semantic guidance to identify incoherent crops, followed by reorienting and aggregating the style gradients of these crops with the global style gradients within one image. Furthermore, we propose a novel multi-objective optimization function to maximize visual discrepancy while enforcing semantic consistency among global, crop, and adversarial features. Applying the stabilized perturbations during training encourages convergence toward flatter and more transferable solutions, improving generalization to unseen domains. Extensive experiments are conducted on multiple CD-FSL benchmarks, demonstrating consistent improvements over state-of-the-art methods.
>
---
#### [new 051] Wiki-R1: Incentivizing Multimodal Reasoning for Knowledge-based VQA via Data and Sampling Curriculum
- **分类: cs.CV**

- **简介: 该论文针对知识增强的视觉问答任务（KB-VQA），解决模型在整合外部知识时的推理与适应问题。提出Wiki-R1框架，通过数据生成和采样策略提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.05256](https://arxiv.org/pdf/2603.05256)**

> **作者:** Shan Ning; Longtian Qiu; Xuming He
>
> **备注:** Accepted by ICLR 26, code and weights are publicly available
>
> **摘要:** Knowledge-Based Visual Question Answering (KB-VQA) requires models to answer questions about an image by integrating external knowledge, posing significant challenges due to noisy retrieval and the structured, encyclopedic nature of the knowledge base. These characteristics create a distributional gap from pretrained multimodal large language models (MLLMs), making effective reasoning and domain adaptation difficult in the post-training stage. In this work, we propose \textit{Wiki-R1}, a data-generation-based curriculum reinforcement learning framework that systematically incentivizes reasoning in MLLMs for KB-VQA. Wiki-R1 constructs a sequence of training distributions aligned with the model's evolving capability, bridging the gap from pretraining to the KB-VQA target distribution. We introduce \textit{controllable curriculum data generation}, which manipulates the retriever to produce samples at desired difficulty levels, and a \textit{curriculum sampling strategy} that selects informative samples likely to yield non-zero advantages during RL updates. Sample difficulty is estimated using observed rewards and propagated to unobserved samples to guide learning. Experiments on two KB-VQA benchmarks, Encyclopedic VQA and InfoSeek, demonstrate that Wiki-R1 achieves new state-of-the-art results, improving accuracy from 35.5\% to 37.1\% on Encyclopedic VQA and from 40.1\% to 44.1\% on InfoSeek. The project page is available at this https URL.
>
---
#### [new 052] MobileFetalCLIP: Selective Repulsive Knowledge Distillation for Mobile Fetal Ultrasound Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于胎儿超声分析任务，解决大模型无法在移动设备部署的问题。通过引入选择性排斥知识蒸馏方法，提升小模型性能并实现实时应用。**

- **链接: [https://arxiv.org/pdf/2603.05421](https://arxiv.org/pdf/2603.05421)**

> **作者:** Numan Saeed; Fadillah Adamsyah Maani; Mohammad Yaqub
>
> **备注:** Project website: this http URL
>
> **摘要:** Fetal ultrasound AI could transform prenatal care in low-resource settings, yet current foundation models exceed 300M visual parameters, precluding deployment on point-of-care devices. Standard knowledge distillation fails under such extreme capacity gaps (~26x), as compact students waste capacity mimicking architectural artifacts of oversized teachers. We introduce Selective Repulsive Knowledge Distillation, which decomposes contrastive KD into diagonal and off-diagonal components: matched pair alignment is preserved while the off-diagonal weight decays into negative values, repelling the student from the teacher's inter-class confusions and forcing discovery of architecturally native features. Our 11.4M parameter student surpasses the 304M-parameter FetalCLIP teacher on zero-shot HC18 biometry validity (88.6% vs. 83.5%) and brain sub-plane F1 (0.784 vs. 0.702), while running at 1.6 ms on iPhone 16 Pro, enabling real-time assistive AI on handheld ultrasound devices. Our code, models, and app are publicly available at this https URL.
>
---
#### [new 053] CATNet: Collaborative Alignment and Transformation Network for Cooperative Perception
- **分类: cs.CV**

- **简介: 该论文属于多智能体协同感知任务，旨在解决时间延迟和多源噪声问题。提出CATNet框架，通过同步、去噪和特征选择提升感知性能。**

- **链接: [https://arxiv.org/pdf/2603.05255](https://arxiv.org/pdf/2603.05255)**

> **作者:** Gong Chen; Chaokun Zhang; Tao Tang; Pengcheng Lv; Feng Li; Xin Xie
>
> **备注:** Accepted by CVPR26
>
> **摘要:** Cooperative perception significantly enhances scene understanding by integrating complementary information from diverse agents. However, existing research often overlooks critical challenges inherent in real-world multi-source data integration, specifically high temporal latency and multi-source noise. To address these practical limitations, we propose Collaborative Alignment and Transformation Network (CATNet), an adaptive compensation framework that resolves temporal latency and noise interference in multi-agent systems. Our key innovations can be summarized in three aspects. First, we introduce a Spatio-Temporal Recurrent Synchronization (STSync) that aligns asynchronous feature streams via adjacent-frame differential modeling, establishing a temporal-spatially unified representation space. Second, we design a Dual-Branch Wavelet Enhanced Denoiser (WTDen) that suppresses global noise and reconstructs localized feature distortions within aligned representations. Third, we construct an Adaptive Feature Selector (AdpSel) that dynamically focuses on critical perceptual features for robust fusion. Extensive experiments on multiple datasets demonstrate that CATNet consistently outperforms existing methods under complex traffic conditions, proving its superior robustness and adaptability.
>
---
#### [new 054] NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，旨在根据密集意图描述和颜色查询准确检索美甲设计图像。针对描述复杂、颜色细微的问题，提出NaiLIA方法，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2603.05446](https://arxiv.org/pdf/2603.05446)**

> **作者:** Kanon Amemiya; Daichi Yashima; Kei Katsumata; Takumi Komatsu; Ryosuke Korekata; Seitaro Otsuki; Komei Sugiura
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** We focus on the task of retrieving nail design images based on dense intent descriptions, which represent multi-layered user intent for nail designs. This is challenging because such descriptions specify unconstrained painted elements and pre-manufactured embellishments as well as visual characteristics, themes, and overall impressions. In addition to these descriptions, we assume that users provide palette queries by specifying zero or more colors via a color picker, enabling the expression of subtle and continuous color nuances. Existing vision-language foundation models often struggle to incorporate such descriptions and palettes. To address this, we propose NaiLIA, a multimodal retrieval method for nail design images, which comprehensively aligns with dense intent descriptions and palette queries during retrieval. Our approach introduces a relaxed loss based on confidence scores for unlabeled images that can align with the descriptions. To evaluate NaiLIA, we constructed a benchmark consisting of 10,625 images collected from people with diverse cultural backgrounds. The images were annotated with long and dense intent descriptions given by over 200 annotators. Experimental results demonstrate that NaiLIA outperforms standard methods.
>
---
#### [new 055] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于模型量化任务，针对多模态大语言模型的训练后量化问题，提出MASQuant框架解决平滑不对齐和跨模态计算不变性问题。**

- **链接: [https://arxiv.org/pdf/2603.04800](https://arxiv.org/pdf/2603.04800)**

> **作者:** Lulu Hu; Wenhu Xiao; Xin Chen; Xinhua Xu; Bowen Xu; Kun Li; Yongliang Tao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Post-training quantization (PTQ) with computational invariance for Large Language Models~(LLMs) have demonstrated remarkable advances, however, their application to Multimodal Large Language Models~(MLLMs) presents substantial challenges. In this paper, we analyze SmoothQuant as a case study and identify two critical issues: Smoothing Misalignment and Cross-Modal Computational Invariance. To address these issues, we propose Modality-Aware Smoothing Quantization (MASQuant), a novel framework that introduces (1) Modality-Aware Smoothing (MAS), which learns separate, modality-specific smoothing factors to prevent Smoothing Misalignment, and (2) Cross-Modal Compensation (CMC), which addresses Cross-modal Computational Invariance by using SVD whitening to transform multi-modal activation differences into low-rank forms, enabling unified quantization across modalities. MASQuant demonstrates stable quantization performance across both dual-modal and tri-modal MLLMs. Experimental results show that MASQuant is competitive among the state-of-the-art PTQ algorithms. Source code: this https URL.
>
---
#### [new 056] Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于图像去噪任务，解决真实噪声生成难题。通过无需元数据的提示驱动方法，生成多样且真实的噪声图像，提升去噪效果。**

- **链接: [https://arxiv.org/pdf/2603.04870](https://arxiv.org/pdf/2603.04870)**

> **作者:** Jaekyun Ko; Dongjin Kim; Soomin Lee; Guanghui Wang; Tae Hyun Kim
>
> **备注:** CVPR 2026
>
> **摘要:** Denoising in the sRGB image space is challenging due to noise variability. Although end-to-end methods perform well, their effectiveness in real-world scenarios is limited by the scarcity of real noisy-clean image pairs, which are expensive and difficult to collect. To address this limitation, several generative methods have been developed to synthesize realistic noisy images from limited data. These generative approaches often rely on camera metadata during both training and testing to synthesize real-world noise. However, the lack of metadata or inconsistencies between devices restricts their usability. Therefore, we propose a novel framework called Prompt-Driven Noise Generation (PNG). This model is capable of acquiring high-dimensional prompt features that capture the characteristics of real-world input noise and creating a variety of realistic noisy images consistent with the distribution of the input noise. By eliminating the dependency on explicit camera metadata, our approach significantly enhances the generalizability and applicability of noise synthesis. Comprehensive experiments reveal that our model effectively produces realistic noisy images and show the successful application of these generated images in removing real-world noise across various benchmark datasets.
>
---
#### [new 057] GloSplat: Joint Pose-Appearance Optimization for Faster and More Accurate 3D Reconstruction
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，解决传统方法分离优化的问题。提出GloSplat框架，通过联合姿态与外观优化，提升重建速度和精度。**

- **链接: [https://arxiv.org/pdf/2603.04847](https://arxiv.org/pdf/2603.04847)**

> **作者:** Tianyu Xiong; Rui Li; Linjie Li; Jiaqi Yang
>
> **摘要:** Feature extraction, matching, structure from motion (SfM), and novel view synthesis (NVS) have traditionally been treated as separate problems with independent optimization objectives. We present GloSplat, a framework that performs \emph{joint pose-appearance optimization} during 3D Gaussian Splatting training. Unlike prior joint optimization methods (BARF, NeRF--, 3RGS) that rely purely on photometric gradients for pose refinement, GloSplat preserves \emph{explicit SfM feature tracks} as first-class entities throughout training: track 3D points are maintained as separate optimizable parameters from Gaussian primitives, providing persistent geometric anchors via a reprojection loss that operates alongside photometric supervision. This architectural choice prevents early-stage pose drift while enabling fine-grained refinement -- a capability absent in photometric-only approaches. We introduce two pipeline variants: (1) \textbf{GloSplat-F}, a COLMAP-free variant using retrieval-based pair selection for efficient reconstruction, and (2) \textbf{GloSplat-A}, an exhaustive matching variant for maximum quality. Both employ global SfM initialization followed by joint photometric-geometric optimization during 3DGS training. Experiments demonstrate that GloSplat-F achieves state-of-the-art among COLMAP-free methods while GloSplat-A surpasses all COLMAP-based baselines.
>
---
#### [new 058] RMK RetinaNet: Rotated Multi-Kernel RetinaNet for Robust Oriented Object Detection in Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文属于遥感图像中的旋转目标检测任务，解决检测精度低、角度回归不稳定等问题，提出RMK RetinaNet模型提升多尺度和多方向检测性能。**

- **链接: [https://arxiv.org/pdf/2603.04793](https://arxiv.org/pdf/2603.04793)**

> **作者:** Huiran Sun
>
> **摘要:** Rotated object detection in remote sensing imagery is hindered by three major bottlenecks: non-adaptive receptive field utilization, inadequate long-range multi-scale feature fusion, and discontinuities in angle regression. To address these issues, we propose Rotated Multi-Kernel RetinaNet (RMK RetinaNet). First, we design a Multi-Scale Kernel (MSK) Block to strengthen adaptive multi-scale feature extraction. Second, we incorporate a Multi-Directional Contextual Anchor Attention (MDCAA) mechanism into the feature pyramid to enhance contextual modeling across scales and orientations. Third, we introduce a Bottom-up Path to preserve fine-grained spatial details that are often degraded during downsampling. Finally, we develop an Euler Angle Encoding Module (EAEM) to enable continuous and stable angle regression. Extensive experiments on DOTA-v1.0, HRSC2016, and UCAS-AOD show that RMK RetinaNet achieves performance comparable to state-of-the-art rotated object detectors while improving robustness in multi-scale and multi-orientation scenarios.
>
---
#### [new 059] Fusion-CAM: Integrating Gradient and Region-Based Class Activation Maps for Robust Visual Explanations
- **分类: cs.CV**

- **简介: 该论文属于可解释AI任务，旨在解决深度网络决策可视化的问题。提出Fusion-CAM融合梯度与区域方法，提升解释的准确性和完整性。**

- **链接: [https://arxiv.org/pdf/2603.05386](https://arxiv.org/pdf/2603.05386)**

> **作者:** Hajar Dekdegue; Moncef Garouani; Josiane Mothe; Jordan Bernigaud
>
> **摘要:** Interpreting the decision-making process of deep convolutional neural networks remains a central challenge in achieving trustworthy and transparent artificial intelligence. Explainable AI (XAI) techniques, particularly Class Activation Map (CAM) methods, are widely adopted to visualize the input regions influencing model predictions. Gradient-based approaches (e.g. Grad-CAM) provide highly discriminative, fine-grained details by computing gradients of class activations but often yield noisy and incomplete maps that emphasize only the most salient regions rather than the complete objects. Region-based approaches (e.g. Score-CAM) aggregate information over larger areas, capturing broader object coverage at the cost of over-smoothing and reduced sensitivity to subtle features. We introduce Fusion-CAM, a novel framework that bridges this explanatory gap by unifying both paradigms through a dedicated fusion mechanism to produce robust and highly discriminative visual explanations. Our method first denoises gradient-based maps, yielding cleaner and more focused activations. It then combines the refined gradient map with region-based maps using contribution weights to enhance class coverage. Finally, we propose an adaptive similarity-based pixel-level fusion that evaluates the agreement between both paradigms and dynamically adjusts the fusion strength. This adaptive mechanism reinforces consistent activations while softly blending conflicting regions, resulting in richer, context-aware, and input-adaptive visual explanations. Extensive experiments on standard benchmarks show that Fusion-CAM consistently outperforms existing CAM variants in both qualitative visualization and quantitative evaluation, providing a robust and flexible tool for interpreting deep neural networks.
>
---
#### [new 060] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决类别不平衡导致的少数结构分割困难问题。提出SCDL框架，通过学习类条件特征分布来减轻偏差，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.05202](https://arxiv.org/pdf/2603.05202)**

> **作者:** Yingxue Su; Yiheng Zhong; Keying Zhu; Zimu Zhang; Zhuoru Zhang; Yifang Wang; Yuxin Zhang; Jingxin Liu
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Medical image segmentation is critical for computer-aided diagnosis. However, dense pixel-level annotation is time-consuming and expensive, and medical datasets often exhibit severe class imbalance. Such imbalance causes minority structures to be overwhelmed by dominant classes in feature representations, hindering the learning of discriminative features and making reliable segmentation particularly challenging. To address this, we propose the Semantic Class Distribution Learning (SCDL) framework, a plug-and-play module that mitigates supervision and representation biases by learning structured class-conditional feature distributions. SCDL integrates Class Distribution Bidirectional Alignment (CDBA) to align embeddings with learnable class proxies and leverages Semantic Anchor Constraints (SAC) to guide proxies using labeled data. Experiments on the Synapse and AMOS datasets demonstrate that SCDL significantly improves segmentation performance across both overall and class-level metrics, with particularly strong gains on minority classes, achieving state-of-the-art results. Our code is released at this https URL.
>
---
#### [new 061] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于弱监督密集视频字幕任务，解决现有方法生成的掩码语义不明确、依赖稀疏标注的问题。提出SAIL模型，通过跨模态对齐生成语义感知掩码，并引入LLM生成合成字幕增强训练。**

- **链接: [https://arxiv.org/pdf/2603.05437](https://arxiv.org/pdf/2603.05437)**

> **作者:** Ye-Chan Kim; SeungJu Cha; Si-Woo Kim; Minju Jeon; Hyungee Kim; Dong-Jin Kim
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Weakly-Supervised Dense Video Captioning aims to localize and describe events in videos trained only on caption annotations, without temporal boundaries. Prior work introduced an implicit supervision paradigm based on Gaussian masking and complementary captioning. However, existing method focuses merely on generating non-overlapping masks without considering their semantic relationship to corresponding events, resulting in simplistic, uniformly distributed masks that fail to capture semantically meaningful regions. Moreover, relying solely on ground-truth captions leads to sub-optimal performance due to the inherent sparsity of existing datasets. In this work, we propose SAIL, which constructs semantically-aware masks through cross-modal alignment. Our similarity aware training objective guides masks to emphasize video regions with high similarity to their corresponding event captions. Furthermore, to guide more accurate mask generation under sparse annotation settings, we introduce an LLM-based augmentation strategy that generates synthetic captions to provide additional alignment signals. These synthetic captions are incorporated through an inter-mask mechanism, providing auxiliary guidance for precise temporal localization without degrading the main objective. Experiments on ActivityNet Captions and YouCook2 demonstrate state-of-the-art performance on both captioning and localization metrics.
>
---
#### [new 062] AdaIAT: Adaptively Increasing Attention to Generated Text to Alleviate Hallucinations in LVLM
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型 hallucination 问题。通过增强生成文本的注意力机制，提出 AdaIAT 方法有效降低幻觉率并保持语言连贯性。**

- **链接: [https://arxiv.org/pdf/2603.04908](https://arxiv.org/pdf/2603.04908)**

> **作者:** Li'an Zhong; Ziqiang He; Jibin Zheng; Jin Li; Z. Jane Wang; Xiangui Kang
>
> **摘要:** Hallucination has been a significant impediment to the development and application of current Large Vision-Language Models (LVLMs). To mitigate hallucinations, one intuitive and effective way is to directly increase attention weights to image tokens during inference. Although this effectively reduces the hallucination rate, it often induces repetitive descriptions. To address this, we first conduct an analysis of attention patterns and reveal that real object tokens tend to assign higher attention to the generated text than hallucinated ones. This inspires us to leverage the generated text, which contains instruction-related visual information and contextual knowledge, to alleviate hallucinations while maintaining linguistic coherence. We therefore propose Attention to Generated Text (IAT) and demonstrate that it significantly reduces the hallucination rate while avoiding repetitive descriptions. To prevent naive amplification from impairing the inherent prediction capabilities of LVLMs, we further explore Adaptive IAT (AdaIAT) that employs a layer-wise threshold to control intervention time and fine-grained amplification magnitude tailored to the characteristics of each attention head. Both analysis and experiments demonstrate the effectiveness of AdaIAT. Results of several LVLMs show that AdaIAT effectively alleviates hallucination (reducing hallucination rates $C_S$ and $C_I$ on LLaVA-1.5 by 35.8% and 37.1%, respectively) while preserving linguistic performance and prediction capability, achieving an attractive trade-off.
>
---
#### [new 063] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于弱监督分类任务，针对实例依赖的部分标签学习（ID-PLL）中实例纠缠问题提出CAD框架，通过类内和类间调节提升分类性能。**

- **链接: [https://arxiv.org/pdf/2603.04825](https://arxiv.org/pdf/2603.04825)**

> **作者:** Rui Zhao; Bin Shi; Kai Sun; Bo Dong
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** Partial label learning is a prominent weakly supervised classification task, where each training instance is ambiguously labeled with a set of candidate labels. In real-world scenarios, candidate labels are often influenced by instance features, leading to the emergence of instance-dependent PLL (ID-PLL), a setting that more accurately reflects this relationship. A significant challenge in ID-PLL is instance entanglement, where instances from similar classes share overlapping features and candidate labels, resulting in increased class confusion. To address this issue, we propose a novel Class-specific Augmentation based Disentanglement (CAD) framework, which tackles instance entanglement by both intra- and inter-class regulations. For intra-class regulation, CAD amplifies class-specific features to generate class-wise augmentations and aligns same-class augmentations across instances. For inter-class regulation, CAD introduces a weighted penalty loss function that applies stronger penalties to more ambiguous labels, encouraging larger inter-class distances. By jointly applying intra- and inter-class regulations, CAD improves the clarity of class boundaries and reduces class confusion caused by entanglement. Extensive experimental results demonstrate the effectiveness of CAD in mitigating the entanglement problem and enhancing ID-PLL performance. The code is available at this https URL.
>
---
#### [new 064] Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes
- **分类: cs.CV**

- **简介: 该论文属于自然语言处理任务，针对扩散语言模型推理速度慢的问题，提出LSP调度器，通过连续前缀提交提升效率。**

- **链接: [https://arxiv.org/pdf/2603.05454](https://arxiv.org/pdf/2603.05454)**

> **作者:** Pengxiang Li; Joey Tsai; Hongwei Xue; Kunyu Shi; Shilin Yan
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Diffusion Language Models (DLMs) promise highly parallel text generation, yet their practical inference speed is often bottlenecked by suboptimal decoding schedulers. Standard approaches rely on 'scattered acceptance'-committing high confidence tokens at disjoint positions throughout the sequence. This approach inadvertently fractures the Key-Value (KV) cache, destroys memory locality, and forces the model into costly, repeated repairs across unstable token boundaries. To resolve this, we present the Longest Stable Prefix (LSP) scheduler, a training-free and model-agnostic inference paradigm based on monolithic prefix absorption. In each denoising step, LSP evaluates token stability via a single forward pass, dynamically identifies a contiguous left-aligned block of stable predictions, and snaps its boundary to natural linguistic or structural delimiters before an atomic commitment. This prefix-first topology yields dual benefits: systemically, it converts fragmented KV cache updates into efficient, contiguous appends; algorithmically, it preserves bidirectional lookahead over a geometrically shrinking active suffix, drastically reducing token flip rates and denoiser calls. Extensive evaluations on LLaDA-8B and Dream-7B demonstrate that LSP accelerates inference by up to 3.4x across rigorous benchmarks including mathematical reasoning, code generation, multilingual (CJK) tasks, and creative writing while matching or slightly improving output quality. By fundamentally restructuring the commitment topology, LSP bridges the gap between the theoretical parallelism of DLMs and practical hardware efficiency.
>
---
#### [new 065] Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉表示学习任务，旨在解决CLIP视觉编码器的表征局限问题。通过将对比信号融入扩散重建过程，提升模型的判别能力和细节感知能力。**

- **链接: [https://arxiv.org/pdf/2603.04803](https://arxiv.org/pdf/2603.04803)**

> **作者:** Boyu Han; Qianqian Xu; Shilong Bao; Zhiyong Yang; Ruochen Cui; Xilin Zhao; Qingming Huang
>
> **摘要:** The limited understanding capacity of the visual encoder in Contrastive Language-Image Pre-training (CLIP) has become a key bottleneck for downstream performance. This capacity includes both Discriminative Ability (D-Ability), which reflects class separability, and Detail Perceptual Ability (P-Ability), which focuses on fine-grained visual cues. Recent solutions use diffusion models to enhance representations by conditioning image reconstruction on CLIP visual tokens. We argue that such paradigms may compromise D-Ability and therefore fail to effectively address CLIP's representation limitations. To address this, we integrate contrastive signals into diffusion-based reconstruction to pursue more comprehensive visual representations. We begin with a straightforward design that augments the diffusion process with contrastive learning on input images. However, empirical results show that the naive combination suffers from gradient conflict and yields suboptimal performance. To balance the optimization, we introduce the Diffusion Contrastive Reconstruction (DCR), which unifies the learning objective. The key idea is to inject contrastive signals derived from each reconstructed image, rather than from the original input, into the diffusion process. Our theoretical analysis shows that the DCR loss can jointly optimize D-Ability and P-Ability. Extensive experiments across various benchmarks and multi-modal large language models validate the effectiveness of our method. The code is available at this https URL.
>
---
#### [new 066] Adaptive Prototype-based Interpretable Grading of Prostate Cancer
- **分类: cs.CV**

- **简介: 该论文属于前列腺癌分级任务，旨在解决病理诊断主观且耗时的问题。提出一种基于原型的可解释框架，提升模型可信度与性能。**

- **链接: [https://arxiv.org/pdf/2603.04947](https://arxiv.org/pdf/2603.04947)**

> **作者:** Riddhasree Bhattacharyya; Pallabi Dutta; Sushmita Mitra
>
> **摘要:** Prostate cancer being one of the frequently diagnosed malignancy in men, the rising demand for biopsies places a severe workload on pathologists. The grading procedure is tedious and subjective, motivating the development of automated systems. Although deep learning has made inroads in terms of performance, its limited interpretability poses challenges for widespread adoption in high-stake applications like medicine. Existing interpretability techniques for prostate cancer classifiers provide a coarse explanation but do not reveal why the highlighted regions matter. In this scenario, we propose a novel prototype-based weakly-supervised framework for an interpretable grading of prostate cancer from histopathology images. These networks can prove to be more trustworthy since their explicit reasoning procedure mirrors the workflow of a pathologist in comparing suspicious regions with clinically validated examples. The network is initially pre-trained at patch-level to learn robust prototypical features associated with each grade. In order to adapt it to a weakly-supervised setup for prostate cancer grading, the network is fine-tuned with a new prototype-aware loss function. Finally, a new attention-based dynamic pruning mechanism is introduced to handle inter-sample heterogeneity, while selectively emphasizing relevant prototypes for optimal performance. Extensive validation on the benchmark PANDA and SICAP datasets confirms that the framework can serve as a reliable assistive tool for pathologists in their routine diagnostic workflows.
>
---
#### [new 067] Locality-Attending Vision Transformer
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在提升视觉Transformer在分割中的表现，同时保持其分类能力。通过引入可学习的高斯核和优化特征表示，增强局部注意力机制。**

- **链接: [https://arxiv.org/pdf/2603.04892](https://arxiv.org/pdf/2603.04892)**

> **作者:** Sina Hajimiri; Farzad Beizaee; Fereshteh Shakeri; Christian Desrosiers; Ismail Ben Ayed; Jose Dolz
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Vision transformers have demonstrated remarkable success in classification by leveraging global self-attention to capture long-range dependencies. However, this same mechanism can obscure fine-grained spatial details crucial for tasks such as segmentation. In this work, we seek to enhance segmentation performance of vision transformers after standard image-level classification training. More specifically, we present a simple yet effective add-on that improves performance on segmentation tasks while retaining vision transformers' image-level recognition capabilities. In our approach, we modulate the self-attention with a learnable Gaussian kernel that biases the attention toward neighboring patches. We further refine the patch representations to learn better embeddings at patch positions. These modifications encourage tokens to focus on local surroundings and ensure meaningful representations at spatial positions, while still preserving the model's ability to incorporate global information. Experiments demonstrate the effectiveness of our modifications, evidenced by substantial segmentation gains on three benchmarks (e.g., over 6% and 4% on ADE20K for ViT Tiny and Base), without changing the training regime or sacrificing classification performance. The code is available at this https URL.
>
---
#### [new 068] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于测试时自适应（TTA）任务，旨在解决模型在分布偏移下的适应问题。提出FOZO方法，无需反向传播，通过优化提示实现高效稳定适应。**

- **链接: [https://arxiv.org/pdf/2603.04733](https://arxiv.org/pdf/2603.04733)**

> **作者:** Xingyu Wang; Tao Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Test-Time Adaptation (TTA) is essential for enabling deep learning models to handle real-world data distribution shifts. However, current approaches face significant limitations: backpropagation-based methods are not suitable for low-end deployment devices, due to their high computation and memory requirements, as well as their tendency to modify model weights during adaptation; while traditional backpropagation-free techniques exhibit constrained adaptation capabilities. In this work, we propose Forward-Only Zeroth-Order Optimization (FOZO), a novel and practical backpropagation-free paradigm for TTA. FOZO leverages a memory-efficient zeroth-order prompt optimization, which is led by objectives optimizing both intermediate feature statistics and prediction entropy. To ensure efficient and stable adaptation over the out-of-distribution data stream, we introduce a dynamically decaying perturbation scale during zeroth-order gradient estimation and theoretically prove its convergence under the TTA data stream assumption. Extensive continual adaptation experiments on ImageNet-C, ImageNet-R, and ImageNet-Sketch demonstrate FOZO's superior performance, achieving 59.52% Top-1 accuracy on ImageNet-C (5K, level 5) and outperforming main gradient-based methods and SOTA forward-only FOA (58.13%). Furthermore, FOZO exhibits strong generalization on quantized (INT8) models. These findings demonstrate that FOZO is a highly competitive solution for TTA deployment in resource-limited scenarios.
>
---
#### [new 069] MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决复杂背景下小目标易被遮挡的问题。提出MI-DETR模型，通过生物启发的运动与外观融合方法提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.05071](https://arxiv.org/pdf/2603.05071)**

> **作者:** Nian Liu; Jin Gao; Shubo Lin; Yutong Kou; Sikui Zhang; Fudong Ge; Zhiqiang Pu; Liang Li; Gang Wang; Yizheng Wang; Weiming Hu
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Infrared small target detection (ISTD) is challenging because tiny, low-contrast targets are easily obscured by complex and dynamic backgrounds. Conventional multi-frame approaches typically learn motion implicitly through deep neural networks, often requiring additional motion supervision or explicit alignment modules. We propose Motion Integration DETR (MI-DETR), a bio-inspired dual-pathway detector that processes one infrared frame per time step while explicitly modeling motion. First, a retina-inspired cellular automaton (RCA) converts raw frame sequences into a motion map defined on the same pixel grid as the appearance image, enabling parvocellular-like appearance and magnocellular-like motion pathways to be supervised by a single set of bounding boxes without extra motion labels or alignment operations. Second, a Parvocellular-Magnocellular Interconnection (PMI) Block facilitates bidirectional feature interaction between the two pathways, providing a biologically motivated intermediate interconnection mechanism. Finally, a RT-DETR decoder operates on features from the two pathways to produce detection results. Surprisingly, our proposed simple yet effective approach yields strong performance on three commonly used ISTD benchmarks. MI-DETR achieves 70.3% mAP@50 and 72.7% F1 on IRDST-H (+26.35 mAP@50 over the best multi-frame baseline), 98.0% mAP@50 on DAUB-R, and 88.3% mAP@50 on ITSDT-15K, demonstrating the effectiveness of biologically inspired motion-appearance integration. Code is available at this https URL.
>
---
#### [new 070] A 360-degree Multi-camera System for Blue Emergency Light Detection Using Color Attention RT-DETR and the ABLDataset
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于目标检测任务，旨在解决应急车辆蓝光识别问题。通过多相机系统与改进的RT-DETR模型，实现高效准确的蓝光检测，提升ADAS安全性。**

- **链接: [https://arxiv.org/pdf/2603.05058](https://arxiv.org/pdf/2603.05058)**

> **作者:** Francisco Vacalebri-Lloret; Lucas Banchero; Jose J. Lopez; Jose M. Mossi
>
> **备注:** 16 pages, 17 figures. Submitted to IEEE Transactions on Intelligent Vehicles
>
> **摘要:** This study presents an advanced system for detecting blue lights on emergency vehicles, developed using ABLDataset, a curated dataset that includes images of European emergency vehicles under various climatic and geographic conditions. The system employs a configuration of four fisheye cameras, each with a 180-degree horizontal field of view, mounted on the sides of the vehicle. A calibration process enables the azimuthal localization of the detections. Additionally, a comparative analysis of major deep neural network algorithms was conducted, including YOLO (v5, v8, and v10), RetinaNet, Faster R-CNN, and RT-DETR. RT-DETR was selected as the base model and enhanced through the incorporation of a color attention block, achieving an accuracy of 94.7 percent and a recall of 94.1 percent on the test set, with field test detections reaching up to 70 meters. Furthermore, the system estimates the approach angle of the emergency vehicle relative to the center of the car using geometric transformations. Designed for integration into a multimodal system that combines visual and acoustic data, this system has demonstrated high efficiency, offering a promising approach to enhancing Advanced Driver Assistance Systems (ADAS) and road safety.
>
---
#### [new 071] CLIP-driven Zero-shot Learning with Ambiguous Labels
- **分类: cs.CV**

- **简介: 该论文属于零样本学习任务，旨在解决标签模糊问题。提出CLIP-PZSL框架，利用CLIP提取特征并优化标签嵌入，提升模型在模糊标签下的性能。**

- **链接: [https://arxiv.org/pdf/2603.05053](https://arxiv.org/pdf/2603.05053)**

> **作者:** Jinfu Fan; Jiangnan Li; Xiaowen Yan; Xiaohui Zhong; Wenpeng Lu; Linqing Huang
>
> **备注:** Accepted by ICASSP 2026 (IEEE International Conference on Acoustics, Speech, and Signal Processing)
>
> **摘要:** Zero-shot learning (ZSL) aims to recognize unseen classes by leveraging semantic information from seen classes, but most existing methods assume accurate class labels for training instances. However, in real-world scenarios, noise and ambiguous labels can significantly reduce the performance of ZSL. To address this, we propose a new CLIP-driven partial label zero-shot learning (CLIP-PZSL) framework to handle label ambiguity. First, we use CLIP to extract instance and label features. Then, a semantic mining block fuses these features to extract discriminative label embeddings. We also introduce a partial zero-shot loss, which assigns weights to candidate labels based on their relevance to the instance and aligns instance and label embeddings to minimize semantic mismatch. As the training goes on, the ground-truth labels are progressively identified, and the refined labels and label embeddings in turn help improve the semantic alignment of instance and label features. Comprehensive experiments on several datasets demonstrate the advantage of CLIP-PZSL.
>
---
#### [new 072] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于强化学习中的决策规划任务，旨在解决世界模型在实时控制中计算成本过高的问题。通过提出一种将观测压缩为8个离散标记的紧凑分词器CompACT，提升规划效率。**

- **链接: [https://arxiv.org/pdf/2603.05438](https://arxiv.org/pdf/2603.05438)**

> **作者:** Dongwon Kim; Gawon Seo; Jinsung Lee; Minsu Cho; Suha Kwak
>
> **备注:** CVPR 2026
>
> **摘要:** World models provide a powerful framework for simulating environment dynamics conditioned on actions or instructions, enabling downstream tasks such as action planning or policy learning. Recent approaches leverage world models as learned simulators, but its application to decision-time planning remains computationally prohibitive for real-time control. A key bottleneck lies in latent representations: conventional tokenizers encode each observation into hundreds of tokens, making planning both slow and resource-intensive. To address this, we propose CompACT, a discrete tokenizer that compresses each observation into as few as 8 tokens, drastically reducing computational cost while preserving essential information for planning. An action-conditioned world model that occupies CompACT tokenizer achieves competitive planning performance with orders-of-magnitude faster planning, offering a practical step toward real-world deployment of world models.
>
---
#### [new 073] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于对抗攻击任务，旨在解决多模态大语言模型的可迁移性漏洞问题。提出MPCAttack框架，通过多范式协同优化提升对抗样本的攻击效果。**

- **链接: [https://arxiv.org/pdf/2603.04846](https://arxiv.org/pdf/2603.04846)**

> **作者:** Yuanbo Li; Tianyang Xu; Cong Hu; Tao Zhou; Xiao-Jun Wu; Josef Kittler
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** The rapid progress of Multi-Modal Large Language Models (MLLMs) has significantly advanced downstream applications. However, this progress also exposes serious transferable adversarial vulnerabilities. In general, existing adversarial attacks against MLLMs typically rely on surrogate models trained within a single learning paradigm and perform independent optimisation in their respective feature spaces. This straightforward setting naturally restricts the richness of feature representations, delivering limits on the search space and thus impeding the diversity of adversarial perturbations. To address this, we propose a novel Multi-Paradigm Collaborative Attack (MPCAttack) framework to boost the transferability of adversarial examples against MLLMs. In principle, MPCAttack aggregates semantic representations, from both visual images and language texts, to facilitate joint adversarial optimisation on the aggregated features through a Multi-Paradigm Collaborative Optimisation (MPCO) strategy. By performing contrastive matching on multi-paradigm features, MPCO adaptively balances the importance of different paradigm representations and guides the global perturbation optimisation, effectively alleviating the representation bias. Extensive experimental results on multiple benchmarks demonstrate the superiority of MPCAttack, indicating that our solution consistently outperforms state-of-the-art methods in both targeted and untargeted attacks on open-source and closed-source MLLMs. The code is released at this https URL.
>
---
#### [new 074] Scalable Injury-Risk Screening in Baseball Pitching From Broadcast Video
- **分类: cs.CV**

- **简介: 该论文属于运动损伤预测任务，解决专业设备难以普及的问题，通过单目视频提取18项生物力学指标，实现可扩展的投球损伤风险筛查。**

- **链接: [https://arxiv.org/pdf/2603.04864](https://arxiv.org/pdf/2603.04864)**

> **作者:** Jerrin Bright; Justin Mende; John Zelek
>
> **备注:** Submitted to CVPRW'26
>
> **摘要:** Injury prediction in pitching depends on precise biomechanical signals, yet gold-standard measurements come from expensive, stadium-installed multi-camera systems that are unavailable outside professional venues. We present a monocular video pipeline that recovers 18 clinically relevant biomechanics metrics from broadcast footage, positioning pose-derived kinematics as a scalable source for injury-risk modeling. Built on DreamPose3D, our approach introduces a drift-controlled global lifting module that recovers pelvis trajectory via velocity-based parameterization and sliding-window inference, lifting pelvis-rooted poses into global space. To address motion blur, compression artifacts, and extreme pitching poses, we incorporate a kinematics refinement pipeline with bone-length constraints, joint-limited inverse kinematics, smoothing, and symmetry constraints to ensure temporally stable and physically plausible kinematics. On 13 professional pitchers (156 paired pitches), 16/18 metrics achieve sub-degree agreement (MAE $< 1^{\circ}$). Using these metrics for injury prediction, an automated screening model achieves AUC 0.811 for Tommy John surgery and 0.825 for significant arm injuries on 7,348 pitchers. The resulting pose-derived metrics support scalable injury-risk screening, establishing monocular broadcast video as a viable alternative to stadium-scale motion capture for biomechanics.
>
---
#### [new 075] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多摄像头3D目标检测任务，解决模型在不同摄像头配置下泛化能力差的问题。提出CoIn3D框架，通过空间感知特征调制和相机感知数据增强提升跨配置性能。**

- **链接: [https://arxiv.org/pdf/2603.05042](https://arxiv.org/pdf/2603.05042)**

> **作者:** Zhaonian Kuang; Rui Ding; Haotian Wang; Xinhu Zheng; Meng Yang; Gang Hua
>
> **备注:** Accepted to CVPR 2026 main track
>
> **摘要:** Multi-camera 3D object detection (MC3D) has attracted increasing attention with the growing deployment of multi-sensor physical agents, such as robots and autonomous vehicles. However, MC3D models still struggle to generalize to unseen platforms with new multi-camera configurations. Current solutions simply employ a meta-camera for unified representation but lack comprehensive consideration. In this paper, we revisit this issue and identify that the devil lies in spatial prior discrepancies across source and target configurations, including different intrinsics, extrinsics, and array layouts. To address this, we propose CoIn3D, a generalizable MC3D framework that enables strong transferability from source configurations to unseen target ones. CoIn3D explicitly incorporates all identified spatial priors into both feature embedding and image observation through spatial-aware feature modulation (SFM) and camera-aware data augmentation (CDA), respectively. SFM enriches feature space by integrating four spatial representations, such as focal length, ground depth, ground gradient, and Plücker coordinate. CDA improves observation diversity under various configurations via a training-free dynamic novel-view image synthesis scheme. Extensive experiments demonstrate that CoIn3D achieves strong cross-configuration performance on landmark datasets such as NuScenes, Waymo, and Lyft, under three dominant MC3D paradigms represented by BEVDepth, BEVFormer, and PETR.
>
---
#### [new 076] InverseNet: Benchmarking Operator Mismatch and Calibration Across Compressive Imaging Modalities
- **分类: cs.CV**

- **简介: 该论文属于压缩成像任务，解决操作符不匹配问题。提出InverseNet基准，评估不同方法在多种场景下的性能，揭示深度学习与传统方法的差异及校准效果。**

- **链接: [https://arxiv.org/pdf/2603.04538](https://arxiv.org/pdf/2603.04538)**

> **作者:** Chengshuai Yang; Xin Yuan
>
> **备注:** Benchmarking Operator Mismatch and Calibration Across Compressive Imaging Modalities
>
> **摘要:** State-of-the-art EfficientSCI loses 20.58 dB when its assumed forward operator deviates from physical reality in just eight parameters, yet no existing benchmark quantifies operator mismatch, the default condition in deployed compressive imaging systems. We introduce InverseNet, the first cross-modality benchmark for operator mismatch, spanning CASSI, CACTI, and single-pixel cameras. Evaluating 12 methods under a four-scenario protocol (ideal, mismatched, oracle-corrected, blind calibration) across 27 simulated scenes and 9 real hardware captures, we find: (1) deep learning methods lose 10-21 dB under mismatch, eliminating their advantage over classical baselines; (2) performance and robustness are inversely correlated across modalities (Spearman r_s = -0.71, p < 0.01); (3) mask-oblivious architectures recover 0% of mismatch losses regardless of calibration quality, while operator-conditioned methods recover 41-90%; (4) blind grid-search calibration recovers 85-100% of the oracle bound without ground truth. Real hardware experiments confirm that simulation trends transfer to physical data. Code will be released upon acceptance.
>
---
#### [new 077] Decoding the Pulse of Reasoning VLMs in Multi-Image Understanding Tasks
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多图像理解任务中视觉语言模型的推理问题，针对注意力分散现象提出PulseFocus方法，提升模型对关键图像的关注度。**

- **链接: [https://arxiv.org/pdf/2603.04676](https://arxiv.org/pdf/2603.04676)**

> **作者:** Chenjun Li
>
> **备注:** 9 pages, 5 figures, 3 tables
>
> **摘要:** Multi-image reasoning remains a significant challenge for vision-language models (VLMs). We investigate a previously overlooked phenomenon: during chain-of-thought (CoT) generation, the text-to-image (T2I) attention of reasoning VLMs exhibits diffuse "pulses": sporadic and unfocused attention patterns that fail to concentrate on task-relevant images. We further reveal a systematic positional bias in attention allocation across images. Motivated by these observations, we propose PulseFocus, a training-free, inference-time method that structures CoT reasoning into interleaved plan/focus blocks with soft attention gating. By forcing the model to explicitly plan which image to examine and then gating decode-time attention to the referenced image, PulseFocus sharpens attention focus and yields consistent improvements on multi-image benchmarks like BLINK benchmark (+3.7%) and MuirBench (+1.07%).
>
---
#### [new 078] SURE: Semi-dense Uncertainty-REfined Feature Matching
- **分类: cs.CV**

- **简介: 该论文属于图像匹配任务，解决特征匹配可靠性问题。针对现有方法在复杂场景中误匹配仍获高相似度的问题，提出SURE框架，联合预测匹配与置信度，提升匹配精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04869](https://arxiv.org/pdf/2603.04869)**

> **作者:** Sicheng Li; Zaiwang Gu; Jie Zhang; Qing Guo; Xudong Jiang; Jun Cheng
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Establishing reliable image correspondences is essential for many robotic vision problems. However, existing methods often struggle in challenging scenarios with large viewpoint changes or textureless regions, where incorrect cor- respondences may still receive high similarity scores. This is mainly because conventional models rely solely on fea- ture similarity, lacking an explicit mechanism to estimate the reliability of predicted matches, leading to overconfident errors. To address this issue, we propose SURE, a Semi- dense Uncertainty-REfined matching framework that jointly predicts correspondences and their confidence by modeling both aleatoric and epistemic uncertainties. Our approach in- troduces a novel evidential head for trustworthy coordinate regression, along with a lightweight spatial fusion module that enhances local feature precision with minimal overhead. We evaluated our method on multiple standard benchmarks, where it consistently outperforms existing state-of-the-art semi-dense matching models in both accuracy and efficiency. our code will be available on this https URL.
>
---
#### [new 079] Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的对抗攻击任务，旨在提升攻击的可迁移性。针对现有方法依赖静态交互、效果有限的问题，提出SADCA通过动态对比学习和语义增强提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2603.04839](https://arxiv.org/pdf/2603.04839)**

> **作者:** Yuanbo Li; Tianyang Xu; Cong Hu; Tao Zhou; Xiao-Jun Wu; Josef Kittler
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** With the rapid advancement and widespread application of vision-language pre-training (VLP) models, their vulnerability to adversarial attacks has become a critical concern. In general, the adversarial examples can typically be designed to exhibit transferable power, attacking not only different models but also across diverse tasks. However, existing attacks on language-vision models mainly rely on static cross-modal interactions and focus solely on disrupting positive image-text pairs, resulting in limited cross-modal disruption and poor transferability. To address this issue, we propose a Semantic-Augmented Dynamic Contrastive Attack (SADCA) that enhances adversarial transferability through progressive and semantically guided perturbation. SADCA progressively disrupts cross-modal alignment through dynamic interactions between adversarial images and texts. This is accomplished by SADCA establishing a contrastive learning mechanism involving adversarial, positive and negative samples, to reinforce the semantic inconsistency of the obtained perturbations. Moreover, we empirically find that input transformations commonly used in traditional transfer-based attacks also benefit VLPs, which motivates a semantic augmentation module that increases the diversity and generalization of adversarial examples. Extensive experiments on multiple datasets and models demonstrate that SADCA significantly improves adversarial transferability and consistently surpasses state-of-the-art methods. The code is released at this https URL.
>
---
#### [new 080] SSR-GS: Separating Specular Reflection in Gaussian Splatting for Glossy Surface Reconstruction
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文属于3D重建任务，旨在解决复杂光照下高光表面的重建问题。提出SSR-GS框架，通过模型直接和间接镜面反射，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.05152](https://arxiv.org/pdf/2603.05152)**

> **作者:** Ningjing Fan; Yiqun Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** In recent years, 3D Gaussian splatting (3DGS) has achieved remarkable progress in novel view synthesis. However, accurately reconstructing glossy surfaces under complex illumination remains challenging, particularly in scenes with strong specular reflections and multi-surface interreflections. To address this issue, we propose SSR-GS, a specular reflection modeling framework for glossy surface reconstruction. Specifically, we introduce a prefiltered Mip-Cubemap to model direct specular reflections efficiently, and propose an IndiASG module to capture indirect specular reflections. Furthermore, we design Visual Geometry Priors (VGP) that couple a reflection-aware visual prior via a reflection score (RS) to downweight the photometric loss contribution of reflection-dominated regions, with geometry priors derived from VGGT, including progressively decayed depth supervision and transformed normal constraints. Extensive experiments on both synthetic and real-world datasets demonstrate that SSR-GS achieves state-of-the-art performance in glossy surface reconstruction.
>
---
#### [new 081] Frequency-Aware Error-Bounded Caching for Accelerating Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文针对扩散Transformer（DiT）推理速度慢的问题，提出SpectralCache框架，通过考虑时间、深度和特征的非均匀性，实现高效缓存，提升速度并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2603.05315](https://arxiv.org/pdf/2603.05315)**

> **作者:** Guandong Li
>
> **摘要:** Diffusion Transformers (DiTs) have emerged as the dominant architecture for high-quality image and video generation, yet their iterative denoising process incurs substantial computational cost during inference. Existing caching methods accelerate DiTs by reusing intermediate computations across timesteps, but they share a common limitation: treating the denoising process as uniform across time,depth, and feature dimensions. In this work, we identify three orthogonal axes of non-uniformity in DiT denoising: (1) temporal -- sensitivity to caching errors varies dramatically across the denoising trajectory; (2) depth -- consecutive caching decisions lead to cascading approximation errors; and (3) feature -- different components of the hidden state exhibit heterogeneous temporal dynamics. Based on these observations, we propose SpectralCache, a unified caching framework comprising Timestep-Aware Dynamic Scheduling (TADS), Cumulative Error Budgets (CEB), and Frequency-Decomposed Caching (FDC). On FLUX.1-schnell at 512x512 resolution, SpectralCache achieves 2.46x speedup with LPIPS 0.217 and SSIM 0.727, outperforming TeaCache (2.12x, LPIPS 0.215, SSIM 0.734) by 16% in speed while maintaining comparable quality (LPIPS difference < 1%). Our approach is training-free, plug-and-play, and compatible with existing DiT architectures.
>
---
#### [new 082] Are Multimodal LLMs Ready for Surveillance? A Reality Check on Zero-Shot Anomaly Detection in the Wild
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频异常检测任务，探讨多模态大语言模型在真实场景下的可靠性，通过实验发现其存在召回率低的问题，并尝试通过指令调整提升性能。**

- **链接: [https://arxiv.org/pdf/2603.04727](https://arxiv.org/pdf/2603.04727)**

> **作者:** Shanle Yao; Armin Danesh Pazho; Narges Rashvand; Hamed Tabkhi
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated impressive general competence in video understanding, yet their reliability for real-world Video Anomaly Detection (VAD) remains largely unexplored. Unlike conventional pipelines relying on reconstruction or pose-based cues, MLLMs enable a paradigm shift: treating anomaly detection as a language-guided reasoning task. In this work, we systematically evaluate state-of-the-art MLLMs on the ShanghaiTech and CHAD benchmarks by reformulating VAD as a binary classification task under weak temporal supervision. We investigate how prompt specificity and temporal window lengths (1s--3s) influence performance, focusing on the precision--recall trade-off. Our findings reveal a pronounced conservative bias in zero-shot settings; while models exhibit high confidence, they disproportionately favor the 'normal' class, resulting in high precision but a recall collapse that limits practical utility. We demonstrate that class-specific instructions can significantly shift this decision boundary, improving the peak F1-score on ShanghaiTech from 0.09 to 0.64, yet recall remains a critical bottleneck. These results highlight a significant performance gap for MLLMs in noisy environments and provide a foundation for future work in recall-oriented prompting and model calibration for open-world surveillance, which demands complex video understanding and reasoning.
>
---
#### [new 083] 3D-RFT: Reinforcement Fine-Tuning for Video-based 3D Scene Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出3D-RFT框架，解决视频驱动的3D场景理解问题，通过强化学习直接优化评估指标，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.04976](https://arxiv.org/pdf/2603.04976)**

> **作者:** Xiongkun Linghu; Jiangyong Huang; Baoxiong Jia; Siyuan Huang
>
> **备注:** Project page: this https URL
>
> **摘要:** Reinforcement Learning with Verifiable Rewards ( RLVR ) has emerged as a transformative paradigm for enhancing the reasoning capabilities of Large Language Models ( LLMs), yet its potential in 3D scene understanding remains under-explored. Existing approaches largely rely on Supervised Fine-Tuning ( SFT), where the token-level cross-entropy loss acts as an indirect proxy for optimization, leading to a misalignment between training objectives and task performances. To bridge this gap, we present Reinforcement Fine-Tuning for Video-based 3D Scene Understanding (3D-RFT ), the first framework to extend RLVR to video-based 3D perception and reasoning. 3D-RFT shifts the paradigm by directly optimizing the model towards evaluation metrics. 3D-RFT first activates 3D-aware Multi-modal Large Language Models ( MLLM s) via SFT, followed by reinforcement fine-tuning using Group Relative Policy Optimization ( GRPO) with strictly verifiable reward functions. We design task-specific reward functions directly from metrics like 3D IoU and F1-Score to provide more effective signals to guide model training. Extensive experiments demonstrate that 3D-RFT-4B achieves state-of-the-art performance on various video-based 3D scene understanding tasks. Notably, 3D-RFT-4B significantly outperforms larger models (e.g., VG LLM-8B) on 3D video detection, 3D visual grounding, and spatial reasoning benchmarks. We further reveal good properties of 3D-RFT such as robust efficacy, and valuable insights into training strategies and data impact. We hope 3D-RFT can serve as a robust and promising paradigm for future development of 3D scene understanding.
>
---
#### [new 084] Revisiting Shape from Polarization in the Era of Vision Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于表面法线估计任务，旨在解决传统SfP方法性能不足的问题。通过构建高质量数据集和引入数据增强，使用极化信息提升模型效果，实现更高效的小样本学习。**

- **链接: [https://arxiv.org/pdf/2603.04817](https://arxiv.org/pdf/2603.04817)**

> **作者:** Chenhao Li; Taishi Ono; Takeshi Uemori; Yusuke Moriuchi
>
> **摘要:** We show that, with polarization cues, a lightweight model trained on a small dataset can outperform RGB-only vision foundation models (VFMs) in single-shot object-level surface normal estimation. Shape from polarization (SfP) has long been studied due to the strong physical relationship between polarization and surface geometry. Meanwhile, driven by scaling laws, RGB-only VFMs trained on large datasets have recently achieved impressive performance and surpassed existing SfP methods. This situation raises questions about the necessity of polarization cues, which require specialized hardware and have limited training data. We argue that the weaker performance of prior SfP methods does not come from the polarization modality itself, but from domain gaps. These domain gaps mainly arise from two sources. First, existing synthetic datasets use limited and unrealistic 3D objects, with simple geometry and random texture maps that do not match the underlying shapes. Second, real-world polarization signals are often affected by sensor noise, which is not well modeled during training. To address the first issue, we render a high-quality polarization dataset using 1,954 3D-scanned real-world objects. We further incorporate pretrained DINOv3 priors to improve generalization to unseen objects. To address the second issue, we introduce polarization sensor-aware data augmentation that better reflects real-world conditions. With only 40K training scenes, our method significantly outperforms both state-of-the-art SfP approaches and RGB-only VFMs. Extensive experiments show that polarization cues enable a 33x reduction in training data or an 8x reduction in model parameters, while still achieving better performance than RGB-only counterparts.
>
---
#### [new 085] Interpretable Pre-Release Baseball Pitch Type Anticipation from Broadcast 3D Kinematics
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于动作识别任务，旨在通过球员身体运动预测棒球投球类型。研究利用3D姿态数据，分析上半身和手腕等关键部位，提升预测准确率。**

- **链接: [https://arxiv.org/pdf/2603.04874](https://arxiv.org/pdf/2603.04874)**

> **作者:** Jerrin Bright; Michelle Lu; John Zelek
>
> **备注:** Submitted to CVPRW'26
>
> **摘要:** How much can a pitcher's body reveal about the upcoming pitch? We study this question at scale by classifying eight pitch types from monocular 3D pose sequences, without access to ball-flight data. Our pipeline chains a diffusion-based 3D pose backbone with automatic pitching-event detection, groundtruth-validated biomechanical feature extraction, and gradient-boosted classification over 229 kinematic features. Evaluated on 119,561 professional pitches, the largest such benchmark to date, we achieve 80.4\% accuracy using body kinematics alone. A systematic importance analysis reveals that upper-body mechanics contribute 64.9\% of the predictive signal versus 35.1\% for the lower body, with wrist position (14.8\%) and trunk lateral tilt emerging as the most informative joint group and biomechanical feature, respectively. We further show that grip-defined variants (four-seam vs.\ two-seam fastball) are not separable from pose, establishing an empirical ceiling near 80\% and delineating where kinematic information ends and ball-flight information begins.
>
---
#### [new 086] Physics-consistent deep learning for blind aberration recovery in mobile optics
- **分类: cs.CV**

- **简介: 该论文属于光学图像恢复任务，旨在解决移动设备中因镜头像差导致的成像质量问题。通过引入物理一致性深度学习框架，同时恢复光学参数并提升去模糊效果。**

- **链接: [https://arxiv.org/pdf/2603.04999](https://arxiv.org/pdf/2603.04999)**

> **作者:** Kartik Jhawar; Tamo Sancho Miguel Tandoc; Khoo Jun Xuan; Wang Lipo
>
> **备注:** 4 pages, 3 figures
>
> **摘要:** Mobile photography is often limited by complex, lens-specific optical aberrations. While recent deep learning methods approach this as an end-to-end deblurring task, these "black-box" models lack explicit optical modeling and can hallucinate details. Conversely, classical blind deconvolution remains highly unstable. To bridge this gap, we present Lens2Zernike, a deep learning framework that blindly recovers physical optical parameters from a single blurred image. To the best of our knowledge, no prior work has simultaneously integrated supervision across three distinct optical domains. We introduce a novel physics-consistent strategy that explicitly minimizes errors via direct Zernike coefficient regression (z), differentiable physics constraints encompassing both wavefront and point spread function derivations (p), and auxiliary multi-task spatial map predictions (m). Through an ablation study on a ResNet-18 backbone, we demonstrate that our full multi-task framework (z+p+m) yields a 35% improvement over coefficient-only baselines. Crucially, comparative analysis reveals that our approach outperforms two established deep learning methods from previous literature, achieving significantly lower regression errors. Ultimately, we demonstrate that these recovered physical parameters enable stable non-blind deconvolution, providing substantial in-domain improvement on the patented Institute for Digital Molecular Analytics and Science (IDMxS) Mobile Camera Lens Database for restoring diffraction-limited details from severely aberrated mobile captures.
>
---
#### [new 087] A Benchmark Study of Neural Network Compression Methods for Hyperspectral Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像分类任务，旨在解决深度学习模型在资源受限平台部署的问题。通过评估剪枝、量化和知识蒸馏等压缩方法，验证其在保持分类性能的同时降低模型复杂度。**

- **链接: [https://arxiv.org/pdf/2603.04720](https://arxiv.org/pdf/2603.04720)**

> **作者:** Sai Shi
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** Deep neural networks have achieved strong performance in image classification tasks due to their ability to learn complex patterns from high-dimensional data. However, their large computational and memory requirements often limit deployment on resource-constrained platforms such as remote sensing devices and edge systems. Network compression techniques have therefore been proposed to reduce model size and computational cost while maintaining predictive performance. In this study, we conduct a systematic evaluation of neural network compression methods for a remote sensing application, namely hyperspectral land cover classification. Specifically, we examine three widely used compression strategies for convolutional neural networks: pruning, quantization, and knowledge distillation. Experiments are conducted on two benchmark hyperspectral datasets, considering classification accuracy, memory consumption, and inference efficiency. Our results demonstrate that compressed models can significantly reduce model size and computational cost while maintaining competitive classification performance. These findings provide insights into the trade-offs between compression ratio, efficiency, and accuracy, and highlight the potential of compression techniques for enabling efficient deep learning deployment in remote sensing applications.
>
---
#### [new 088] UniPAR: A Unified Framework for Pedestrian Attribute Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于行人属性识别任务，解决跨域数据处理难题。提出UniPAR框架，实现多模态数据统一处理，提升模型泛化能力与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.05114](https://arxiv.org/pdf/2603.05114)**

> **作者:** Minghe Xu; Rouying Wu; Jiarui Xu; Minhao Sun; Zikang Yan; Xiao Wang; ChiaWei Chu; Yu Li
>
> **摘要:** Pedestrian Attribute Recognition is a foundational computer vision task that provides essential support for downstream applications, including person retrieval in video surveillance and intelligent retail analytics. However, existing research is frequently constrained by the ``one-model-per-dataset" paradigm and struggles to handle significant discrepancies across domains in terms of modalities, attribute definitions, and environmental scenarios. To address these challenges, we propose UniPAR, a unified Transformer-based framework for PAR. By incorporating a unified data scheduling strategy and a dynamic classification head, UniPAR enables a single model to simultaneously process diverse datasets from heterogeneous modalities, including RGB images, video sequences, and event streams. We also introduce an innovative phased fusion encoder that explicitly aligns visual features with textual attribute queries through a late deep fusion strategy. Experimental results on the widely used benchmark datasets, including MSP60K, DukeMTMC, and EventPAR, demonstrate that UniPAR achieves performance comparable to specialized SOTA methods. Furthermore, multi-dataset joint training significantly enhances the model's cross-domain generalization and recognition robustness in extreme environments characterized by low light and motion blur. The source code of this paper will be released on this https URL
>
---
#### [new 089] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset
- **分类: cs.CV**

- **简介: 该论文聚焦真实场景下的红外图像超分辨率任务，解决真实红外图像因光学和传感退化导致的结构模糊与热 fidelity 下降问题。提出 Real-IISR 框架，通过热结构引导实现逐级重建，并构建 FLIR-IISR 数据集。**

- **链接: [https://arxiv.org/pdf/2603.04745](https://arxiv.org/pdf/2603.04745)**

> **作者:** Yang Zou; Jun Ma; Zhidong Jiao; Xingyuan Li; Zhiying Jiang; Jinyuan Liu
>
> **备注:** This paper was accepted by CVPR 2026
>
> **摘要:** Infrared image super-resolution (IISR) under real-world conditions is a practically significant yet rarely addressed task. Pioneering works are often trained and evaluated on simulated datasets or neglect the intrinsic differences between infrared and visible imaging. In practice, however, real infrared images are affected by coupled optical and sensing degradations that jointly deteriorate both structural sharpness and thermal fidelity. To address these challenges, we propose Real-IISR, a unified autoregressive framework for real-world IISR that progressively reconstructs fine-grained thermal structures and clear backgrounds in a scale-by-scale manner via thermal-structural guided visual autoregression. Specifically, a Thermal-Structural Guidance module encodes thermal priors to mitigate the mismatch between thermal radiation and structural edges. Since non-uniform degradations typically induce quantization bias, Real-IISR adopts a Condition-Adaptive Codebook that dynamically modulates discrete representations based on degradation-aware thermal priors. Also, a Thermal Order Consistency Loss enforces a monotonic relation between temperature and pixel intensity, ensuring relative brightness order rather than absolute values to maintain physical consistency under spatial misalignment and thermal drift. We build FLIR-IISR, a real-world IISR dataset with paired LR-HR infrared images acquired via automated focus variation and motion-induced blur. Extensive experiments demonstrate the promising performance of Real-IISR, providing a unified foundation for real-world IISR and benchmarking. The dataset and code are available at: this https URL.
>
---
#### [new 090] SGR3 Model: Scene Graph Retrieval-Reasoning Model in 3D
- **分类: cs.CV**

- **简介: 该论文提出SGR3模型，解决3D场景图生成任务。针对传统方法依赖多模态数据和启发式构造的问题，采用检索增强生成方法，提升关系推理能力。**

- **链接: [https://arxiv.org/pdf/2603.04614](https://arxiv.org/pdf/2603.04614)**

> **作者:** Zirui Wang; Ruiping Liu; Yufan Chen; Junwei Zheng; Weijia Fan; Kunyu Peng; Di Wen; Jiale Wei; Jiaming Zhang; Rainer Stiefelhagen
>
> **摘要:** 3D scene graphs provide a structured representation of object entities and their relationships, enabling high-level interpretation and reasoning for robots while remaining intuitively understandable to humans. Existing approaches for 3D scene graph generation typically combine scene reconstruction with graph neural networks (GNNs). However, such pipelines require multi-modal data that may not always be available, and their reliance on heuristic graph construction can constrain the prediction of relationship triplets. In this work, we introduce a Scene Graph Retrieval-Reasoning Model in 3D (SGR3 Model), a training-free framework that leverages multi-modal large language models (MLLMs) with retrieval-augmented generation (RAG) for semantic scene graph generation. SGR3 Model bypasses the need for explicit 3D reconstruction. Instead, it enhances relational reasoning by incorporating semantically aligned scene graphs retrieved via a ColPali-style cross-modal framework. To improve retrieval robustness, we further introduce a weighted patch-level similarity selection mechanism that mitigates the negative impact of blurry or semantically uninformative regions. Experiments demonstrate that SGR3 Model achieves competitive performance compared to training-free baselines and on par with GNN-based expert models. Moreover, an ablation study on the retrieval module and knowledge base scale reveals that retrieved external information is explicitly integrated into the token generation process, rather than being implicitly internalized through abstraction.
>
---
#### [new 091] Video-based Locomotion Analysis for Fish Health Monitoring
- **分类: cs.CV**

- **简介: 该论文属于鱼群运动分析任务，旨在通过视频监测鱼类健康状况。解决如何准确提取鱼的运动信息以评估健康问题。工作包括使用YOLOv11和多目标跟踪方法进行运动分析。**

- **链接: [https://arxiv.org/pdf/2603.05407](https://arxiv.org/pdf/2603.05407)**

> **作者:** Timon Palm; Clemens Seibold; Anna Hilsmann; Peter Eisert
>
> **备注:** Accepted at VISAPP 2026
>
> **摘要:** Monitoring the health conditions of fish is essential, as it enables the early detection of disease, safeguards animal welfare, and contributes to sustainable aquaculture practices. Physiological and pathological conditions of cultivated fish can be inferred by analyzing locomotion activities. In this paper, we present a system that estimates the locomotion activities from videos using multi object tracking. The core of our approach is a YOLOv11 detector embedded in a tracking-by-detection framework. We investigate various configurations of the YOLOv11-architecture as well as extensions that incorporate multiple frames to improve detection accuracy. Our system is evaluated on a manually annotated dataset of Sulawesi ricefish recorded in a home-aquarium-like setup, demonstrating its ability to reliably measure swimming direction and speed for fish health monitoring. The dataset will be made publicly available upon publication.
>
---
#### [new 092] PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing
- **分类: cs.CV**

- **简介: 该论文提出PinPoint基准，用于评估组合图像检索（CIR）任务中的多答案、硬负样本、多图查询等挑战，解决现有基准不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.04598](https://arxiv.org/pdf/2603.04598)**

> **作者:** Rohan Mahadev; Joyce Yuan; Patrick Poirson; David Xue; Hao-Yu Wu; Dmitry Kislyuk
>
> **备注:** Accepted for CVPR 2026
>
> **摘要:** Composed Image Retrieval (CIR) has made significant progress, yet current benchmarks are limited to single ground-truth answers and lack the annotations needed to evaluate false positive avoidance, robustness and multi-image reasoning. We present PinPoint, a comprehensive real world benchmark with 7,635 queries and 329K relevance judgments across 23 query categories. PinPoint advances the field by providing: (1) multiple correct answers (averaging 9.1 per query) (2) explicit hard negatives, (3) six instruction paraphrases per query for robustness testing, (4) multi-image composition support (13.4% of queries), and (5) demographic metadata for fairness evaluation. Based on our analysis of 20+ methods across 4 different major paradigms, we uncover three significant drawbacks: The best methods while achieving mAP@10 of 28.5%, still retrieves irrelevant results (hard negatives) 9% of the time. The best models also exhibit 25.1% performance variation across paraphrases, indicating significant potential for enhancing current CIR techniques. Multi-image queries performs 40 to 70% worse across different methods. To overcome these new issues uncovered by our evaluation framework, we propose a training-free reranking method based on an off-the-shelf MLLM that can be applied to any existing system to bridge the gap. We release the complete dataset, including all images, queries, annotations, retrieval index, and benchmarking code.
>
---
#### [new 093] TAPFormer: Robust Arbitrary Point Tracking via Transient Asynchronous Fusion of Frames and Events
- **分类: cs.CV**

- **简介: 该论文属于目标跟踪任务，解决RGB帧与事件流异步融合导致的时序错位问题。提出TAPFormer框架，通过TAF机制实现鲁棒的高频率任意点跟踪。**

- **链接: [https://arxiv.org/pdf/2603.04989](https://arxiv.org/pdf/2603.04989)**

> **作者:** Jiaxiong Liu; Zhen Tan; Jinpu Zhang; Yi Zhou; Hui Shen; Xieyuanli Chen; Dewen Hu
>
> **摘要:** Tracking any point (TAP) is a fundamental yet challenging task in computer vision, requiring high precision and long-term motion reasoning. Recent attempts to combine RGB frames and event streams have shown promise, yet they typically rely on synchronous or non-adaptive fusion, leading to temporal misalignment and severe degradation when one modality fails. We introduce TAPFormer, a transformer-based framework that performs asynchronous temporal-consistent fusion of frames and events for robust and high-frequency arbitrary point tracking. Our key innovation is a Transient Asynchronous Fusion (TAF) mechanism, which explicitly models the temporal evolution between discrete frames through continuous event updates, bridging the gap between low-rate frames and high-rate events. In addition, a Cross-modal Locally Weighted Fusion (CLWF) module adaptively adjusts spatial attention according to modality reliability, yielding stable and discriminative features even under blur or low light. To evaluate our approach under realistic conditions, we construct a novel real-world frame-event TAP dataset under diverse illumination and motion conditions. Our method outperforms existing point trackers, achieving a 28.2% improvement in average pixel error within threshold. Moreover, on standard point tracking benchmarks, our tracker consistently achieves the best performance. Project website: this http URL
>
---
#### [new 094] The Impact of Preprocessing Methods on Racial Encoding and Model Robustness in CXR Diagnosis
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于医疗AI任务，旨在解决深度学习模型中的种族偏差问题。通过图像预处理方法减少种族快捷学习，提升模型公平性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.05157](https://arxiv.org/pdf/2603.05157)**

> **作者:** Dishantkumar Sutariya; Eike Petersen
>
> **备注:** Preprint accepted for publication at BVM 2026 (this https URL)
>
> **摘要:** Deep learning models can identify racial identity with high accuracy from chest X-ray (CXR) recordings. Thus, there is widespread concern about the potential for racial shortcut learning, where a model inadvertently learns to systematically bias its diagnostic predictions as a function of racial identity. Such racial biases threaten healthcare equity and model reliability, as models may systematically misdiagnose certain demographic groups. Since racial shortcuts are diffuse - non-localized and distributed throughout the whole CXR recording - image preprocessing methods may influence racial shortcut learning, yet the potential of such methods for reducing biases remains underexplored. Here, we investigate the effects of image preprocessing methods including lung masking, lung cropping, and Contrast Limited Adaptive Histogram Equalization (CLAHE). These approaches aim to suppress spurious cues encoding racial information while preserving diagnostic accuracy. Our experiments reveal that simple bounding box-based lung cropping can be an effective strategy for reducing racial shortcut learning while maintaining diagnostic model performance, bypassing frequently postulated fairness-accuracy trade-offs.
>
---
#### [new 095] HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的幻觉检测任务，旨在在生成文本前预测幻觉风险。通过分析模型内部表示，无需解码即可实现高效检测。**

- **链接: [https://arxiv.org/pdf/2603.05465](https://arxiv.org/pdf/2603.05465)**

> **作者:** Sai Akhil Kogilathota; Sripadha Vallabha E G; Luzhe Sun; Jiawei Zhou
>
> **摘要:** Hallucinations remain a persistent challenge for vision-language models (VLMs), which often describe nonexistent objects or fabricate facts. Existing detection methods typically operate after text generation, making intervention both costly and untimely. We investigate whether hallucination risk can instead be predicted before any token is generated by probing a model's internal representations in a single forward pass. Across a diverse set of vision-language tasks and eight modern VLMs, including Llama-3.2-Vision, Gemma-3, Phi-4-VL, and Qwen2.5-VL, we examine three families of internal representations: (i) visual-only features without multimodal fusion, (ii) vision-token representations within the text decoder, and (iii) query-token representations that integrate visual and textual information before generation. Probes trained on these representations achieve strong hallucination-detection performance without decoding, reaching up to 0.93 AUROC on Gemma-3-12B, Phi-4-VL 5.6B, and Molmo 7B. Late query-token states are the most predictive for most models, while visual or mid-layer features dominate in a few architectures (e.g., ~0.79 AUROC for Qwen2.5-VL-7B using visual-only features). These results demonstrate that (1) hallucination risk is detectable pre-generation, (2) the most informative layer and modality vary across architectures, and (3) lightweight probes have the potential to enable early abstention, selective routing, and adaptive decoding to improve both safety and efficiency.
>
---
#### [new 096] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model
- **分类: cs.CV**

- **简介: 该论文属于无监督域适应任务，解决医疗图像分割中跨域泛化问题。提出Tell2Adapt框架，利用视觉基础模型生成高质量伪标签并提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2603.05012](https://arxiv.org/pdf/2603.05012)**

> **作者:** Yulong Shi; Shijie Li; Ziyi Li; Lin Qi
>
> **备注:** Accepted by IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)
>
> **摘要:** Source Free Unsupervised Domain Adaptation (SFUDA) is critical for deploying deep learning models across diverse clinical settings. However, existing methods are typically designed for low-gap, specific domain shifts and cannot generalize into a unified, multi-modalities, and multi-target framework, which presents a major barrier to real-world application. To overcome this issue, we introduce Tell2Adapt, a novel SFUDA framework that harnesses the vast, generalizable knowledge of the Vision Foundation Model (VFM). Our approach ensures high-fidelity VFM prompts through Context-Aware Prompts Regularization (CAPR), which robustly translates varied text prompts into canonical instructions. This enables the generation of high-quality pseudo-labels for efficiently adapting the lightweight student model to target domain. To guarantee clinical reliability, the framework incorporates Visual Plausibility Refinement (VPR), which leverages the VFM's anatomical knowledge to re-ground the adapted model's predictions in target image's low-level visual features, effectively removing noise and false positives. We conduct one of the most extensive SFUDA evaluations to date, validating our framework across 10 domain adaptation directions and 22 anatomical targets, including brain, cardiac, polyp, and abdominal targets. Our results demonstrate that Tell2Adapt consistently outperforms existing approaches, achieving SOTA for a unified SFUDA framework in medical image segmentation. Code are avaliable at this https URL.
>
---
#### [new 097] Exploiting Intermediate Reconstructions in Optical Coherence Tomography for Test-Time Adaption of Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决低质量成像设备下分割性能不足的问题。通过利用光学相干断层扫描的中间重建结果，提升分割精度并提供不确定性估计。**

- **链接: [https://arxiv.org/pdf/2603.05041](https://arxiv.org/pdf/2603.05041)**

> **作者:** Thomas Pinetz; Veit Hucke; Hrvoje Bogunovic
>
> **备注:** Accepted at MIDL 2026
>
> **摘要:** Primary health care frequently relies on low-cost imaging devices, which are commonly used for screening purposes. To ensure accurate diagnosis, these systems depend on advanced reconstruction algorithms designed to approximate the performance of high-quality counterparts. Such algorithms typically employ iterative reconstruction methods that incorporate domain-specific prior knowledge. However, downstream task performance is generally assessed using only the final reconstructed image, thereby disregarding the informative intermediate representations generated throughout the reconstruction process. In this work, we propose IRTTA to exploit these intermediate representations at test-time by adapting the normalization-layer parameters of a frozen downstream network via a modulator network that conditions on the current reconstruction timescale. The modulator network is learned during test-time using an averaged entropy loss across all individual timesteps. Variation among the timestep-wise segmentations additionally provides uncertainty estimates at no extra cost. This approach enhances segmentation performance and enables semantically meaningful uncertainty estimation, all without modifying either the reconstruction process or the downstream model.
>
---
#### [new 098] Logi-PAR: Logic-Infused Patient Activity Recognition via Differentiable Rule
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于患者活动识别任务，旨在解决现有模型仅能识别活动而无法解释原因的问题。提出Logi-PAR框架，融合逻辑规则与视觉线索，实现可解释的活动识别与风险推理。**

- **链接: [https://arxiv.org/pdf/2603.05184](https://arxiv.org/pdf/2603.05184)**

> **作者:** Muhammad Zarar; MingZheng Zhang; Xiaowang Zhang; Zhiyong Feng; Sofonias Yitagesu; Kawsar Farooq
>
> **摘要:** Patient Activity Recognition (PAR) in clinical settings uses activity data to improve safety and quality of care. Although significant progress has been made, current models mainly identify which activity is occurring. They often spatially compose sub-sparse visual cues using global and local attention mechanisms, yet only learn logically implicit patterns due to their neural-pipeline. Advancing clinical safety requires methods that can infer why a set of visual cues implies a risk, and how these can be compositionally reasoned through explicit logic beyond mere classification. To address this, we proposed Logi-PAR, the first Logic-Infused Patient Activity Recognition Framework that integrates contextual fact fusion as a multi-view primitive extractor and injects neural-guided differentiable rules. Our method automatically learns rules from visual cues, optimizing them end-to-end while enabling the implicit emergence patterns to be explicitly labelled during training. To the best of our knowledge, Logi-PAR is the first framework to recognize patient activity by applying learnable logic rules to symbolic mappings. It produces auditable why explanations as rule traces and supports counterfactual interventions (e.g., risk would decrease by 65% if assistance were present). Extensive evaluation on clinical benchmarks (VAST and OmniFall) demonstrates state-of-the-art performance, significantly outperforming Vision-Language Models and transformer baselines. The code is available via: this https URL}
>
---
#### [new 099] Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging. Review Paper
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割与分类任务，旨在解决脑胶质瘤精准分割与分类问题，比较传统方法与深度学习的效果，指出CNN优于传统方法。**

- **链接: [https://arxiv.org/pdf/2603.04796](https://arxiv.org/pdf/2603.04796)**

> **作者:** Kiranmayee Janardhan; Vinay Martin DSa Prabhu; T. Christy Bobby
>
> **备注:** 22 pages, 4 Figures
>
> **摘要:** Segmentation is crucial for brain gliomas as it delineates the glioma s extent and location, aiding in precise treatment planning and monitoring, thus improving patient outcomes. Accurate segmentation ensures proper identification of the glioma s size and position, transforming images into applicable data for analysis. Classification of brain gliomas is also essential because different types require different treatment approaches. Accurately classifying brain gliomas by size, location, and aggressiveness is essential for personalized prognosis prediction, follow-up care, and monitoring disease progression, ensuring effective diagnosis, treatment, and management. In glioma research, irregular tissues are often observable, but error free and reproducible segmentation is challenging. Many researchers have surveyed brain glioma segmentation, proposing both fully automatic and semi-automatic techniques. The adoption of these methods by radiologists depends on ease of use and supervision, with semi-automatic techniques preferred due to the need for accurate evaluations. This review evaluates effective segmentation and classification techniques post magnetic resonance imaging acquisition, highlighting that convolutional neural network architectures outperform traditional techniques in these tasks.
>
---
#### [new 100] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer
- **分类: cs.CV**

- **简介: 该论文属于4D重建任务，解决动态场景中运动物体影响相机姿态估计的问题。提出MoRe网络，通过注意力机制分离动态运动与静态结构，实现高效准确的4D重建。**

- **链接: [https://arxiv.org/pdf/2603.05078](https://arxiv.org/pdf/2603.05078)**

> **作者:** Juntong Fang; Zequn Chen; Weiqi Zhang; Donglin Di; Xuancheng Zhang; Chengmin Yang; Yu-Shen Liu
>
> **备注:** Accepted by CVPR 2025. Project page:this https URL
>
> **摘要:** Reconstructing dynamic 4D scenes remains challenging due to the presence of moving objects that corrupt camera pose estimation. Existing optimization methods alleviate this issue with additional supervision, but they are mostly computationally expensive and impractical in real-time applications. To address these limitations, we propose MoRe, a feedforward 4D reconstruction network that efficiently recovers dynamic 3D scenes from monocular videos. Built upon a strong static reconstruction backbone, MoRe employs an attention-forcing strategy to disentangle dynamic motion from static structure. To further enhance robustness, we fine-tune the model on large-scale, diverse datasets encompassing both dynamic and static scenes. Moreover, our grouped causal attention captures temporal dependencies and adapts to varying token lengths across frames, ensuring temporally coherent geometry reconstruction. Extensive experiments on multiple benchmarks demonstrate that MoRe achieves high-quality dynamic reconstructions with exceptional efficiency.
>
---
#### [new 101] SPyCer: Semi-Supervised Physics-Guided Contextual Attention for Near-Surface Air Temperature Estimation from Satellite Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SPyCer模型，用于从卫星图像中估计近地表空气温度。解决传感器稀疏分布导致的空间连续性不足问题，通过物理引导的注意力机制提升估计精度与一致性。**

- **链接: [https://arxiv.org/pdf/2603.05219](https://arxiv.org/pdf/2603.05219)**

> **作者:** Sofiane Bouaziz; Adel Hafiane; Raphael Canals; Rachid Nedjai
>
> **摘要:** Modern Earth observation relies on satellites to capture detailed surface properties. Yet, many phenomena that affect humans and ecosystems unfold in the atmosphere close to the surface. Near-ground sensors provide accurate measurements of certain environmental characteristics, such as near-surface air temperature (NSAT). However, they remain sparse and unevenly distributed, limiting their ability to provide continuous spatial measurements. To bridge this gap, we introduce SPyCer, a semi-supervised physics-guided network that can leverage pixel information and physical modeling to guide the learning process through meaningful physical properties. It is designed for continuous estimation of NSAT by proxy using satellite imagery. SPyCer frames NSAT prediction as a pixel-wise vision problem, where each near-ground sensor is projected onto satellite image coordinates and positioned at the center of a local image patch. The corresponding sensor pixel is supervised using both observed NSAT and physics-based constraints, while surrounding pixels contribute through physics-guided regularization derived from the surface energy balance and advection-diffusion-reaction partial differential equations. To capture the physical influence of neighboring pixels, SPyCer employs a multi-head attention guided by land cover characteristics and modulated with Gaussian distance weighting. Experiments on real-world datasets demonstrate that SPyCer produces spatially coherent and physically consistent NSAT estimates, outperforming existing baselines in terms of accuracy, generalization, and alignment with underlying physical processes.
>
---
#### [new 102] On Multi-Step Theorem Prediction via Non-Parametric Structural Priors
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于自动化推理任务，旨在解决多步定理预测中的泛化问题。通过引入结构先验，提升模型在复杂推理中的表现。**

- **链接: [https://arxiv.org/pdf/2603.04852](https://arxiv.org/pdf/2603.04852)**

> **作者:** Junbo Zhao; Ting Zhang; Can Li; Wei He; Jingdong Wang; Hua Huang
>
> **摘要:** Multi-step theorem prediction is a central challenge in automated reasoning. Existing neural-symbolic approaches rely heavily on supervised parametric models, which exhibit limited generalization to evolving theorem libraries. In this work, we explore training-free theorem prediction through the lens of in-context learning (ICL). We identify a critical scalability bottleneck, termed Structural Drift: as reasoning depth increases, the performance of vanilla ICL degrades sharply, often collapsing to near zero. We attribute this failure to the LLM's inability to recover latent topological dependencies, leading to unstructured exploration. To address this issue, we propose Theorem Precedence Graphs, which encode temporal dependencies from historical solution traces as directed graphs, and impose explicit topological constraints that effectively prune the search space during inference. Coupled with retrieval-augmented graph construction and a stepwise symbolic executor, our approach enables LLMs to act as structured planners without any gradient-based optimization. Experiments on the FormalGeo7k benchmark show that our method achieves 89.29% accuracy, substantially outperforming ICL baselines and matching state-of-the-art supervised models. These results indicate that explicit structural priors offer a promising direction for scaling LLM-based symbolic reasoning.
>
---
#### [new 103] Loop Closure via Maximal Cliques in 3D LiDAR-Based SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D LiDAR SLAM任务，解决 loop closure 检测问题。提出 CliReg 算法，通过最大团搜索替代 RANSAC，提升鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2603.05397](https://arxiv.org/pdf/2603.05397)**

> **作者:** Javier Laserna; Saurabh Gupta; Oscar Martinez Mozos; Cyrill Stachniss; Pablo San Segundo
>
> **备注:** Accepted in the 2025 European Conference on Mobile Robots (ECMR). This is the author's version of the work
>
> **摘要:** Reliable loop closure detection remains a critical challenge in 3D LiDAR-based SLAM, especially under sensor noise, environmental ambiguity, and viewpoint variation conditions. RANSAC is often used in the context of loop closures for geometric model fitting in the presence of outliers. However, this approach may fail, leading to map inconsistency. We introduce a novel deterministic algorithm, CliReg, for loop closure validation that replaces RANSAC verification with a maximal clique search over a compatibility graph of feature correspondences. This formulation avoids random sampling and increases robustness in the presence of noise and outliers. We integrated our approach into a real- time pipeline employing binary 3D descriptors and a Hamming distance embedding binary search tree-based matching. We evaluated it on multiple real-world datasets featuring diverse LiDAR sensors. The results demonstrate that our proposed technique consistently achieves a lower pose error and more reliable loop closures than RANSAC, especially in sparse or ambiguous conditions. Additional experiments on 2D projection-based maps confirm its generality across spatial domains, making our approach a robust and efficient alternative for loop closure detection.
>
---
#### [new 104] SkillNet: Create, Evaluate, and Connect AI Skills
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出SkillNet，解决AI技能难以积累与迁移的问题。构建统一技能框架，支持创建、评估与连接，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2603.04448](https://arxiv.org/pdf/2603.04448)**

> **作者:** Yuan Liang; Ruobin Zhong; Haoming Xu; Chen Jiang; Yi Zhong; Runnan Fang; Jia-Chen Gu; Shumin Deng; Yunzhi Yao; Mengru Wang; Shuofei Qiao; Xin Xu; Tongtong Wu; Kun Wang; Yang Liu; Zhen Bi; Jungang Lou; Yuchen Eleanor Jiang; Hangcheng Zhu; Gang Yu; Haiwen Hong; Longtao Huang; Hui Xue; Chenxi Wang; Yijun Wang; Zifei Shan; Xi Chen; Zhaopeng Tu; Feiyu Xiong; Xin Xie; Peng Zhang; Zhengke Gui; Lei Liang; Jun Zhou; Chiyu Wu; Jin Shang; Yu Gong; Junyu Lin; Changliang Xu; Hongjie Deng; Wen Zhang; Keyan Ding; Qiang Zhang; Fei Huang; Ningyu Zhang; Jeff Z. Pan; Guilin Qi; Haofen Wang; Huajun Chen
>
> **备注:** this http URL
>
> **摘要:** Current AI agents can flexibly invoke tools and execute complex tasks, yet their long-term advancement is hindered by the lack of systematic accumulation and transfer of skills. Without a unified mechanism for skill consolidation, agents frequently ``reinvent the wheel'', rediscovering solutions in isolated contexts without leveraging prior strategies. To overcome this limitation, we introduce SkillNet, an open infrastructure designed to create, evaluate, and organize AI skills at scale. SkillNet structures skills within a unified ontology that supports creating skills from heterogeneous sources, establishing rich relational connections, and performing multi-dimensional evaluation across Safety, Completeness, Executability, Maintainability, and Cost-awareness. Our infrastructure integrates a repository of over 200,000 skills, an interactive platform, and a versatile Python toolkit. Experimental evaluations on ALFWorld, WebShop, and ScienceWorld demonstrate that SkillNet significantly enhances agent performance, improving average rewards by 40% and reducing execution steps by 30% across multiple backbone models. By formalizing skills as evolving, composable assets, SkillNet provides a robust foundation for agents to move from transient experience to durable mastery.
>
---
#### [new 105] ICHOR: A Robust Representation Learning Approach for ASL CBF Maps with Self-Supervised Masked Autoencoders
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文提出ICHOR方法，用于ASL CBF图像的表示学习，解决图像质量不一、数据标注不足等问题，通过自监督预训练提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.05247](https://arxiv.org/pdf/2603.05247)**

> **作者:** Xavier Beltran-Urbano; Yiran Li; Xinglin Zeng; Katie R. Jobson; Manuel Taso; Christopher A. Brown; David A. Wolk; Corey T. McMillan; Ilya M. Nashrallah; Paul A. Yushkevich; Ze Wang; John A. Detre; Sudipto Dolui
>
> **摘要:** Arterial spin labeling (ASL) perfusion MRI allows direct quantification of regional cerebral blood flow (CBF) without exogenous contrast, enabling noninvasive measurements that can be repeated without constraints imposed by contrast injection. ASL is increasingly acquired in research studies and clinical MRI protocols. Building on successes in structural imaging, recent efforts have implemented deep learning based methods to improve image quality, enable automated quality control, and derive robust quantitative and predictive biomarkers with ASL derived CBF. However, progress has been limited by variable image quality, substantial inter-site, vendor and protocol differences, and limited availability of labeled datasets needed to train models that generalize across cohorts. To address these challenges, we introduce ICHOR, a self supervised pre-training approach for ASL CBF maps that learns transferable representations using 3D masked autoencoders. ICHOR is pretrained via masked image modeling using a Vision Transformer backbone and can be used as a general-purpose encoder for downstream ASL tasks. For pre-training, we curated one of the largest ASL datasets to date, comprising 11,405 ASL CBF scans from 14 studies spanning multiple sites and acquisition protocols. We evaluated the pre-trained ICHOR encoder on three downstream diagnostic classification tasks and one ASL CBF map quality prediction regression task. Across all evaluations, ICHOR outperformed existing neuroimaging self-supervised pre-training methods adapted to ASL. Pre-trained weights and code will be made publicly available.
>
---
#### [new 106] OpenFrontier: General Navigation with Visual-Language Grounded Frontiers
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决复杂环境中高效导航问题。提出OpenFrontier框架，无需训练和精细调优，利用视觉语言先验实现高效导航。**

- **链接: [https://arxiv.org/pdf/2603.05377](https://arxiv.org/pdf/2603.05377)**

> **作者:** Esteban Padilla; Boyang Sun; Marc Pollefeys; Hermann Blum
>
> **摘要:** Open-world navigation requires robots to make decisions in complex everyday environments while adapting to flexible task requirements. Conventional navigation approaches often rely on dense 3D reconstruction and hand-crafted goal metrics, which limits their generalization across tasks and environments. Recent advances in vision--language navigation (VLN) and vision--language--action (VLA) models enable end-to-end policies conditioned on natural language, but typically require interactive training, large-scale data collection, or task-specific fine-tuning with a mobile agent. We formulate navigation as a sparse subgoal identification and reaching problem and observe that providing visual anchoring targets for high-level semantic priors enables highly efficient goal-conditioned navigation. Based on this insight, we select navigation frontiers as semantic anchors and propose OpenFrontier, a training-free navigation framework that seamlessly integrates diverse vision--language prior models. OpenFrontier enables efficient navigation with a lightweight system design, without dense 3D mapping, policy training, or model fine-tuning. We evaluate OpenFrontier across multiple navigation benchmarks and demonstrate strong zero-shot performance, as well as effective real-world deployment on a mobile robot.
>
---
#### [new 107] WebChain: A Large-Scale Human-Annotated Dataset of Real-World Web Interaction Traces
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出WebChain数据集，用于加速网页代理的可复现研究。解决真实网页交互轨迹标注不足的问题，通过多模态数据和新训练方法提升网页代理性能。**

- **链接: [https://arxiv.org/pdf/2603.05295](https://arxiv.org/pdf/2603.05295)**

> **作者:** Sicheng Fan; Rui Wan; Yifei Leng; Gaoning Liang; Li Ling; Yanyi Shang; Dehan Kong
>
> **摘要:** We introduce WebChain, the largest open-source dataset of human-annotated trajectories on real-world websites, designed to accelerate reproducible research in web agents. It contains 31,725 trajectories and 318k steps, featuring a core Triple Alignment of visual, structural, and action data to provide rich, multi-modal supervision. The data is collected via a scalable pipeline that ensures coverage of complex, high-value tasks often missed by synthetic methods. Leveraging this dataset, we propose a Dual Mid-Training recipe that decouples spatial grounding from planning, achieving state-of-the-art performance on our proposed WebChainBench and other public GUI benchmarks. Our work provides the data and insights necessary to build and rigorously evaluate the next generation of scalable web agents.
>
---
#### [new 108] The Thinking Boundary: Quantifying Reasoning Suitability of Multimodal Tasks via Dual Tuning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态任务研究，旨在解决何时推理训练有效的问题。通过双调优框架评估推理效果，提出“思考边界”以指导数据和训练策略选择。**

- **链接: [https://arxiv.org/pdf/2603.04415](https://arxiv.org/pdf/2603.04415)**

> **作者:** Ruobing Zheng; Tianqi Li; Jianing Li; Qingpei Guo; Yi Yuan; Jingdong Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** While reasoning-enhanced Large Language Models (LLMs) have demonstrated remarkable advances in complex tasks such as mathematics and coding, their effectiveness across universal multimodal scenarios remains uncertain. The trend of releasing parallel "Instruct" and "Thinking" models by leading developers serves merely as a resource-intensive workaround, stemming from the lack of a criterion for determining when reasoning is truly beneficial. In this paper, we propose Dual Tuning, a framework designed to assess whether reasoning yields positive gains for target tasks under given base models and datasets. By jointly fine-tuning on paired Chain-of-Thought (CoT) and Direct-Answer (DA) data under controlled prompts, we systematically quantify and compare the gains of both training modes using the proposed metrics, and establish the "Thinking Boundary" to evaluate the suitability of reasoning training across diverse multimodal tasks, including spatial, mathematical, and multi-disciplinary domains. We further explore the impact of reinforcement training and thinking patterns on reasoning suitability, and validate whether the "Thinking Boundary" can guide data refinement. Our findings challenge the "reasoning-for-all" paradigm, providing practical guidance for identifying appropriate data and training strategies, and motivating the development of resource-efficient, adaptive auto-think systems.
>
---
#### [new 109] Using Vision + Language Models to Predict Item Difficulty
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于心理测量与自动化题库开发任务，旨在预测数据可视化测试题的难度。通过分析文本和图像特征，使用大语言模型评估题目难度，提升测试效率。**

- **链接: [https://arxiv.org/pdf/2603.04670](https://arxiv.org/pdf/2603.04670)**

> **作者:** Samin Khan
>
> **摘要:** This project investigates the capabilities of large language models (LLMs) to determine the difficulty of data visualization literacy test items. We explore whether features derived from item text (question and answer options), the visualization image, or a combination of both can predict item difficulty (proportion of correct responses) for U.S. adults. We use GPT-4.1-nano to analyze items and generate predictions based on these distinct feature sets. The multimodal approach, using both visual and text features, yields the lowest mean absolute error (MAE) (0.224), outperforming the unimodal vision-only (0.282) and text-only (0.338) approaches. The best-performing multimodal model was applied to a held-out test set for external evaluation and achieved a mean squared error of 0.10805, demonstrating the potential of LLMs for psychometric analysis and automated item development.
>
---
#### [new 110] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多模态联邦学习任务，旨在解决客户端性能个性化、模态差异和模型异构问题。提出FedAFD框架，通过对抗对齐和知识蒸馏提升协作效果。**

- **链接: [https://arxiv.org/pdf/2603.04890](https://arxiv.org/pdf/2603.04890)**

> **作者:** Min Tan; Junchao Ma; Yinfu Feng; Jiajun Ding; Wenwen Pan; Tingting Han; Qian Zheng; Zhenzhong Kuang; Zhou Yu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Multimodal Federated Learning (MFL) enables clients with heterogeneous data modalities to collaboratively train models without sharing raw data, offering a privacy-preserving framework that leverages complementary cross-modal information. However, existing methods often overlook personalized client performance and struggle with modality/task discrepancies, as well as model heterogeneity. To address these challenges, we propose FedAFD, a unified MFL framework that enhances client and server learning. On the client side, we introduce a bi-level adversarial alignment strategy to align local and global representations within and across modalities, mitigating modality and task gaps. We further design a granularity-aware fusion module to integrate global knowledge into the personalized features adaptively. On the server side, to handle model heterogeneity, we propose a similarity-guided ensemble distillation mechanism that aggregates client representations on shared public data based on feature similarity and distills the fused knowledge into the global model. Extensive experiments conducted under both IID and non-IID settings demonstrate that FedAFD achieves superior performance and efficiency for both the client and the server.
>
---
#### [new 111] Axiomatic On-Manifold Shapley via Optimal Generative Flows
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于可解释AI任务，解决Shapley attribution的离流形伪影问题，提出基于最优生成流的在流形上Aumann-Shapley归因方法。**

- **链接: [https://arxiv.org/pdf/2603.05093](https://arxiv.org/pdf/2603.05093)**

> **作者:** Cenwei Zhang; Lin Zhu; Manxi Lin; Lei You
>
> **备注:** 11 figures, 22 pages
>
> **摘要:** Shapley-based attribution is critical for post-hoc XAI but suffers from off-manifold artifacts due to heuristic baselines. While generative methods attempt to address this, they often introduce geometric inefficiency and discretization drift. We propose a formal theory of on-manifold Aumann-Shapley attributions driven by optimal generative flows. We prove a representation theorem establishing the gradient line integral as the unique functional satisfying efficiency and geometric axioms, notably reparameterization invariance. To resolve path ambiguity, we select the kinetic-energy-minimizing Wasserstein-2 geodesic transporting a prior to the data distribution. This yields a canonical attribution family that recovers classical Shapley for additive models and admits provable stability bounds against flow approximation errors. By reframing baseline selection as a variational problem, our method experimentally outperforms baselines, achieving strict manifold adherence via vanishing Flow Consistency Error and superior semantic alignment characterized by Structure-Aware Total Variation. Our code is on this https URL.
>
---
#### [new 112] TimeWarp: Evaluating Web Agents by Revisiting the Past
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于Web代理评估任务，旨在解决代理在网页变化下的适应性问题。通过构建TimeWarp基准和提出TimeTraj算法，提升代理的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04949](https://arxiv.org/pdf/2603.04949)**

> **作者:** Md Farhan Ishmam; Kenneth Marino
>
> **摘要:** The improvement of web agents on current benchmarks raises the question: Do today's agents perform just as well when the web changes? We introduce TimeWarp, a benchmark that emulates the evolving web using containerized environments that vary in UI, design, and layout. TimeWarp consists of three web environments, each with six UI versions spanning different eras of the internet, paired with a set of complex, realistic tasks requiring different forms of web navigation. Our experiments reveal web agents' vulnerability to changes and the limitations of behavior cloning (BC) on single-version trajectories. To address this, we propose TimeTraj, a simple yet effective algorithm that uses plan distillation to collect trajectories across multiple versions. By training agents on teacher rollouts using our BC-variant, we achieve substantial performance gains: $20.4\%\rightarrow37.7\%$ for Qwen-3 4B and $0\%\rightarrow27.0\%$ for Llama-3.1 8B models. We hope our work helps researchers study generalization across web designs and unlock a new paradigm for collecting plans rather than trajectories, thereby improving the robustness of web agents.
>
---
#### [new 113] Beyond the Patch: Exploring Vulnerabilities of Visuomotor Policies via Viewpoint-Consistent 3D Adversarial Object
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉控制领域，旨在解决动态视角下2D对抗补丁失效的问题。通过3D对抗纹理优化方法，提升对机器人策略的攻击效果。**

- **链接: [https://arxiv.org/pdf/2603.04913](https://arxiv.org/pdf/2603.04913)**

> **作者:** Chanmi Lee; Minsung Yoon; Woojae Kim; Sebin Lee; Sung-eui Yoon
>
> **备注:** 8 pages, 10 figures, Accepted to ICRA 2026. Project page: this https URL
>
> **摘要:** Neural network-based visuomotor policies enable robots to perform manipulation tasks but remain susceptible to perceptual attacks. For example, conventional 2D adversarial patches are effective under fixed-camera setups, where appearance is relatively consistent; however, their efficacy often diminishes under dynamic viewpoints from moving cameras, such as wrist-mounted setups, due to perspective distortions. To proactively investigate potential vulnerabilities beyond 2D patches, this work proposes a viewpoint-consistent adversarial texture optimization method for 3D objects through differentiable rendering. As optimization strategies, we employ Expectation over Transformation (EOT) with a Coarse-to-Fine (C2F) curriculum, exploiting distance-dependent frequency characteristics to induce textures effective across varying camera-object distances. We further integrate saliency-guided perturbations to redirect policy attention and design a targeted loss that persistently drives robots toward adversarial objects. Our comprehensive experiments show that the proposed method is effective under various environmental conditions, while confirming its black-box transferability and real-world applicability.
>
---
## 更新

#### [replaced 001] Flatness Guided Test-Time Adaptation for Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.18864](https://arxiv.org/pdf/2501.18864)**

> **作者:** Aodi Li; Liansheng Zhuang; Xiao Long; Houqiang Li; Shafei Wang
>
> **摘要:** Test-time adaptation (TTA) of Vision-Language Models (VLMs) has emerged as a technique for tackling distribution shifts during the test time. Recent research indicates that the test-time adaptation is intrinsically linked to the model's training history. However, existing TTA methods, such as Test-time Prompt Tuning, often design adaptation strategies in isolation from the models' training characteristics, which degrade their performance. This paper argues that the flatness acquired via sharpness-aware training is an efficient clue for the test-time adaptation of VLMs. Built on this insight, this paper proposes a novel Flatness-Guided Adaptation framework (FGA) for VLMs to cohesively unify training and test-time procedures. Its core idea is to leverage the alignment between the training minimum and test loss flat regions to guide the adaptation process. Specifically, our FGA consists of a prompt-tuning stage and a test-time adaptation stage. In the tuning stage, a Sharpness-Aware Prompt Tuning method is utilized to identify the training flat minimum, offering a geometric clue of flatness for subsequent adaptation. In the test stage, a Sharpness-based Test Sample Selection approach is proposed to ensure the alignment of flat minima between the training and each augmented test sample's loss landscape. In comparison to existing TTA methods, our FGA avoids the expensive prompt parameter updates during test time, and substantially reduces the computation overhead. Extensive experiments on both domain generalization and cross-dataset benchmarks demonstrate that our FGA achieves superior performance over prevalent TTA methods. Notably, when employing a ViT-B/16 image encoder, FGA even outperforms TPT+CoOp by an average of 4.88% across all four ImageNet out-of-domain variants.
>
---
#### [replaced 002] DDP-WM: Disentangled Dynamics Prediction for Efficient World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DDP-WM，解决世界模型计算效率低的问题。通过分解动态预测，提升实时性能，适用于导航与操作任务。**

- **链接: [https://arxiv.org/pdf/2602.01780](https://arxiv.org/pdf/2602.01780)**

> **作者:** Shicheng Yin; Kaixuan Yin; Weixing Chen; Yang Liu; Guanbin Li; Liang Lin
>
> **备注:** Efficient and high-fidelity world model. Code is available at this https URL
>
> **摘要:** World models are essential for autonomous robotic planning. However, the substantial computational overhead of existing dense Transformerbased models significantly hinders real-time deployment. To address this efficiency-performance bottleneck, we introduce DDP-WM, a novel world model centered on the principle of Disentangled Dynamics Prediction (DDP). We hypothesize that latent state evolution in observed scenes is heterogeneous and can be decomposed into sparse primary dynamics driven by physical interactions and secondary context-driven background updates. DDP-WM realizes this decomposition through an architecture that integrates efficient historical processing with dynamic localization to isolate primary dynamics. By employing a crossattention mechanism for background updates, the framework optimizes resource allocation and provides a smooth optimization landscape for planners. Extensive experiments demonstrate that DDP-WM achieves significant efficiency and performance across diverse tasks, including navigation, precise tabletop manipulation, and complex deformable or multi-body interactions. Specifically, on the challenging Push-T task, DDP-WM achieves an approximately 9 times inference speedup and improves the MPC success rate from 90% to98% compared to state-of-the-art dense models. The results establish a promising path for developing efficient, high-fidelity world models. Codes is available at this https URL.
>
---
#### [replaced 003] MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-k Activations
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01219](https://arxiv.org/pdf/2602.01219)**

> **作者:** Qishuai Wen; Zhiyuan Huang; Xianghan Meng; Wei He; Chun-Guang Li
>
> **备注:** Code is available at this https URL
>
> **摘要:** The attention operator in Transformers can be viewed as a two-layer fast-weight MLP, whose weights are dynamically instantiated from input tokens and whose width equals sequence length N. As the context extends, the expressive capacity of such an N-width MLP increases, but scaling its fast weights becomes prohibitively expensive for extremely long sequences. Recently, this fast-weight scaling perspective has motivated the Mixture-of-Experts (MoE) attention, which partitions the sequence into fast-weight experts and sparsely routes the tokens to them. In this paper, we elevate this perspective to a unifying framework for a wide range of efficient attention methods by interpreting them as scaling fast weights through either routing or compression. Then we propose a compress-and-route strategy, which compresses the N-width MLP into a narrower one using a small set of landmark queries and constructs deformable experts by gathering top-k activated key-value pairs for each landmark query. We call this strategy a Mixture of Top-k Activations (MiTA), and refer to the resulting efficient mechanism as MiTA attention. Preliminary experiments on vision tasks demonstrate the promise of our MiTA attention and motivate further investigation on its optimization and broader applications in more challenging settings.
>
---
#### [replaced 004] Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.02175](https://arxiv.org/pdf/2603.02175)**

> **作者:** Yiqi Lin; Guoqiang Liang; Ziyun Zeng; Zechen Bai; Yanzhe Chen; Mike Zheng Shou
>
> **备注:** Project page: this https URL Huggingface Demo: this https URL
>
> **摘要:** Instruction-based video editing has witnessed rapid progress, yet current methods often struggle with precise visual control, as natural language is inherently limited in describing complex visual nuances. Although reference-guided editing offers a robust solution, its potential is currently bottlenecked by the scarcity of high-quality paired training data. To bridge this gap, we introduce a scalable data generation pipeline that transforms existing video editing pairs into high-fidelity training quadruplets, leveraging image generative models to create synthesized reference scaffolds. Using this pipeline, we construct RefVIE, a large-scale dataset tailored for instruction-reference-following tasks, and establish RefVIE-Bench for comprehensive evaluation. Furthermore, we propose a unified editing architecture, Kiwi-Edit, that synergizes learnable queries and latent visual features for reference semantic guidance. Our model achieves significant gains in instruction following and reference fidelity via a progressive multi-stage training curriculum. Extensive experiments demonstrate that our data and architecture establish a new state-of-the-art in controllable video editing. All datasets, models, and code is released at this https URL.
>
---
#### [replaced 005] Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16786](https://arxiv.org/pdf/2511.16786)**

> **作者:** Yaoxin Yang; Peng Ye; Xudong Tan; Chongjun Tu; Maosen Zhao; Jia Hao; Tao Chen
>
> **备注:** CVPR2026
>
> **摘要:** Multimodal large language models suffer from substantial inference overhead since multimodal KV Cache grows proportionally with the visual input length. Existing multimodal KV Cache compression methods mostly rely on attention score to reduce cache size, which makes them are incompatible with established efficient attention kernels (e.g., FlashAttention) and ignores the contribution of value vectors to the attention output. In this work, we revisit multimodal KV Cache compression from the perspective of the KV matrices' distribution. First, we observe that frequency-domain energy of multimodal KV matrices is predominantly concentrated in low-frequency and extract this principal energy via a low-pass filter. Further, we find that removing KV pairs that deviate substantially from this principal energy leads to a pronounced performance drop, which we define as Outlier KVs. Considering Outlier KVs are more likely to encode features critical for inference, we propose FlashCache, a frequency-domain-guided, Outlier-KV-aware KV Cache compression framework. First, we introduce an Outlier KV Recognition Module that models the principal component of multimodal KV matrices in the frequency domain and preferentially retains KV pairs that significantly deviate from it. Furthermore, Dynamic Budget Allocation Module is designed to adaptively determine the per-layer KV Cache size to retain more Outlier KVs. Experiments on multiple MLLMs and benchmarks demonstrate that FlashCache outperforms state-of-the-art multimoal KV compression methods, achieving up to 1.69 times faster decoding with 80% lower KV memory usage while maintaining task performance.
>
---
#### [replaced 006] DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.24096](https://arxiv.org/pdf/2602.24096)**

> **作者:** Yuxuan Zhang; Katarína Tóthová; Zian Wang; Kangxue Yin; Haithem Turki; Riccardo de Lutio; Yen-Yu Chang; Or Litany; Sanja Fidler; Zan Gojcic
>
> **备注:** For more details and updates, please visit our project website: this https URL
>
> **摘要:** Simulation is essential to the development and evaluation of autonomous robots such as self-driving vehicles. Neural reconstruction is emerging as a promising solution as it enables simulating a wide variety of scenarios from real-world data alone in an automated and scalable way. However, while methods such as NeRF and 3D Gaussian Splatting can produce visually compelling results, they often exhibit artifacts particularly when rendering novel views, and fail to realistically integrate inserted dynamic objects, especially when they were captured from different scenes. To overcome these limitations, we introduce DiffusionHarmonizer, an online generative enhancement framework that transforms renderings from such imperfect scenes into temporally consistent outputs while improving their realism. At its core is a single-step temporally-conditioned enhancer that is converted from a pretrained multi-step image diffusion model, capable of running in online simulators on a single GPU. The key to training it effectively is a custom data curation pipeline that constructs synthetic-real pairs emphasizing appearance harmonization, artifact correction, and lighting realism. The result is a scalable system that significantly elevates simulation fidelity in both research and production environments.
>
---
#### [replaced 007] TerraCodec: Compressing Optical Earth Observation Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.12670](https://arxiv.org/pdf/2510.12670)**

> **作者:** Julen Costa-Watanabe; Isabelle Wittmann; Benedikt Blumenstiel; Konrad Schindler
>
> **摘要:** Earth observation (EO) satellites produce massive streams of multispectral image time series, posing pressing challenges for storage and transmission. Yet, learned EO compression remains fragmented and lacks publicly available, large-scale pretrained codecs. Moreover, prior work has largely focused on image compression, leaving temporal redundancy and EO video codecs underexplored. To address these gaps, we introduce TerraCodec (TEC), a family of learned codecs pretrained on Sentinel-2 EO data. TEC includes efficient multispectral image variants and a Temporal Transformer model (TEC-TT) that leverages dependencies across time. To overcome the fixed-rate setting of today's neural codecs, we present Latent Repacking, a novel method for training flexible-rate transformer models that operate on varying rate-distortion settings. TerraCodec outperforms classical codecs, achieving 3-10x higher compression at equivalent image quality. Beyond compression, TEC-TT enables zero-shot cloud inpainting, surpassing state-of-the-art methods on the AllClear benchmark. Our results establish neural codecs as a promising direction for Earth observation. Our code and models are publically available at this https URL.
>
---
#### [replaced 008] BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.23589](https://arxiv.org/pdf/2509.23589)**

> **作者:** Shu Liu; Wenlin Chen; Weihao Li; Zheng Wang; Lijin Yang; Jianing Huang; Yipin Zhang; Zhongzhan Huang; Ze Cheng; Hao Yang
>
> **备注:** Accepted for publication at ICLR 2026
>
> **摘要:** Diffusion-based planners have shown strong potential for autonomous driving by capturing multi-modal driving behaviors. A key challenge is how to effectively guide these models for safe and reactive planning in closed-loop settings, where the ego vehicle's actions influence future states. Recent work leverages typical expert driving behaviors (i.e., anchors) to guide diffusion planners but relies on a truncated diffusion schedule that introduces an asymmetry between the forward and denoising processes, diverging from the core principles of diffusion models. To address this, we introduce BridgeDrive, a novel anchor-guided diffusion bridge policy for closed-loop trajectory planning. Our approach formulates planning as a diffusion bridge that directly transforms coarse anchor trajectories into refined, context-aware plans, ensuring theoretical consistency between the forward and reverse processes. BridgeDrive is compatible with efficient ODE solvers, enabling real-time deployment. We achieve state-of-the-art performance on the Bench2Drive closed-loop evaluation benchmark, improving the success rate by 7.72% and 2.45% over prior arts with PDM-Lite and LEAD datasets, respectively. Project page: this https URL.
>
---
#### [replaced 009] EA-Swin: An Embedding-Agnostic Swin Transformer for AI-Generated Video Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17260](https://arxiv.org/pdf/2602.17260)**

> **作者:** Hung Mai; Loi Dinh; Duc Hai Nguyen; Dat Do; Luong Doan; Khanh Nguyen Quoc; Huan Vu; Naeem Ul Islam; Tuan Do
>
> **备注:** 2nd preprint version
>
> **摘要:** Recent advances in foundation video generators such as Sora2, Veo3, and other commercial systems have produced highly realistic synthetic videos, exposing the limitations of existing detection methods that rely on shallow embedding trajectories, image-based adaptation, or computationally heavy MLLMs. We propose EA-Swin, an Embedding-Agnostic Swin Transformer that models spatiotemporal dependencies directly on pretrained video embeddings via a factorized windowed attention design, making it compatible with generic ViT-style patch-based encoders. Moreover, we construct the EA-Video dataset, a benchmark dataset comprising 130K videos that integrates newly collected samples with curated existing datasets, covering diverse commercial and open-source generators and including unseen-generator splits for rigorous cross-distribution evaluation. Extensive experiments show that EA-Swin achieves 0.97-0.99 accuracy across major generators, outperforming prior SoTA methods (typically 0.8-0.9) by a margin of 5-20\%, while maintaining strong generalization to unseen distributions, establishing a scalable and robust solution for modern AI-generated video detection.
>
---
#### [replaced 010] PhyGDPO: Physics-Aware Groupwise Direct Preference Optimization for Physically Consistent Text-to-Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.24551](https://arxiv.org/pdf/2512.24551)**

> **作者:** Yuanhao Cai; Kunpeng Li; Menglin Jia; Jialiang Wang; Junzhe Sun; Feng Liang; Weifeng Chen; Felix Juefei-Xu; Chu Wang; Ali Thabet; Xiaoliang Dai; Xuan Ju; Alan Yuille; Ji Hou
>
> **摘要:** Recent advances in text-to-video (T2V) generation have achieved good visual quality, yet synthesizing videos that faithfully follow physical laws remains an open challenge. Existing methods mainly based on graphics or prompt extension struggle to generalize beyond simple simulated environments or learn implicit physical reasoning. The scarcity of training data with rich physics interactions and phenomena is also a problem. In this paper, we first introduce a Physics-Augmented video data construction Pipeline, PhyAugPipe, that leverages a vision-language model (VLM) with chain-of-thought reasoning to collect a large-scale training dataset, PhyVidGen-135K. Then we formulate a principled Physics-aware Groupwise Direct Preference Optimization, PhyGDPO, framework that uses real-world video as winning case to guarantee correct physics learning and builds upon the groupwise Plackett-Luce probabilistic model to capture holistic preferences beyond pairwise comparisons. In PhyGDPO, we design a Physics-Guided Rewarding (PGR) scheme that leverages VLM-based physical rewards to direct the optimization to focus on challenging physics cases. In addition, we propose a LoRA-Switch Reference (LoRA-SR) scheme that avoids full-model duplication as reference for efficient DPO training. Experiments show that our method significantly outperforms state-of-the-art open-source methods on PhyGenBench and VideoPhy2. Please check our project page at this https URL for more video results. Our code, models, and data will be released at this https URL
>
---
#### [replaced 011] MambaTAD: When State-Space Models Meet Long-Range Temporal Action Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.17929](https://arxiv.org/pdf/2511.17929)**

> **作者:** Hui Lu; Yi Yu; Shijian Lu; Deepu Rajan; Boon Poh Ng; Alex C. Kot; Xudong Jiang
>
> **摘要:** Temporal Action Detection (TAD) aims to identify and localize actions by determining their starting and ending frames within untrimmed videos. Recent Structured State-Space Models such as Mamba have demonstrated potential in TAD due to their long-range modeling capability and linear computational complexity. On the other hand, structured state-space models often face two key challenges in TAD, namely, decay of temporal context due to recursive processing and self-element conflict during global visual context modeling, which become more severe while handling long-span action instances. Additionally, traditional methods for TAD struggle with detecting long-span action instances due to a lack of global awareness and inefficient detection heads. This paper presents MambaTAD, a new state-space TAD model that introduces long-range modeling and global feature detection capabilities for accurate temporal action detection. MambaTAD comprises two novel designs that complement each other with superior TAD performance. First, it introduces a Diagonal-Masked Bidirectional State-Space (DMBSS) module which effectively facilitates global feature fusion and temporal action detection. Second, it introduces a global feature fusion head that refines the detection progressively with multi-granularity features and global awareness. In addition, MambaTAD tackles TAD in an end-to-end one-stage manner using a new state-space temporal adapter(SSTA) which reduces network parameters and computation cost with linear complexity. Extensive experiments show that MambaTAD achieves superior TAD performance consistently across multiple public benchmarks.
>
---
#### [replaced 012] MultiShadow: Multi-Object Shadow Generation for Image Compositing via Diffusion Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02743](https://arxiv.org/pdf/2603.02743)**

> **作者:** Waqas Ahmed; Dean Diepeveen; Ferdous Sohel
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Realistic shadow generation is crucial for achieving seamless image compositing, yet existing methods primarily focus on single-object insertion and often fail to generalize when multiple foreground objects are composited into a background scene. In practice, however, modern compositing pipelines and real-world applications often insert multiple objects simultaneously, necessitating shadows that are jointly consistent in terms of geometry, attachment, and location. In this paper, we address the under-explored problem of multi-object shadow generation, aiming to synthesize physically plausible shadows for multiple inserted objects. Our approach exploits the multimodal capabilities of a pre-trained text-to-image diffusion model. An image pathway injects dense, multi-scale features to provide fine-grained spatial guidance, while a text-based pathway encodes per-object shadow bounding boxes as learned positional tokens and fuses them via cross-attention. An attention-alignment loss further grounds these tokens to their corresponding shadow regions. To support this task, we augment the DESOBAv2 dataset by constructing composite scenes with multiple inserted objects and automatically derive prompts combining object category and shadow positioning information. Experimental results demonstrate that our method achieves state-of-the-art performance in both single and multi-object shadow generation settings.
>
---
#### [replaced 013] Track Anything Behind Everything: Zero-Shot Amodal Video Object Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.19210](https://arxiv.org/pdf/2411.19210)**

> **作者:** Finlay G. C. Hudson; William A. P. Smith
>
> **摘要:** We present Track Anything Behind Everything (TABE), a novel pipeline for zero-shot amodal video object segmentation. Unlike existing methods that require pretrained class labels, our approach uses a single query mask from the first frame where the object is visible, enabling flexible, zero-shot inference. We pose amodal segmentation as generative outpainting from modal (visible) masks using a pretrained video diffusion model. We do not need to re-train the diffusion model to accommodate additional input channels but instead use a pretrained model that we fine-tune at test-time to allow specialisation towards the tracked object. Our TABE pipeline is specifically designed to handle amodal completion, even in scenarios where objects are completely occluded. Our model and code will all be released.
>
---
#### [replaced 014] Observer-Actor: Active Vision Imitation Learning with Sparse-View Gaussian Splatting
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出ObAct框架，用于主动视觉模仿学习，解决机器人视角受限问题。通过动态调整观察者与执行者角色，提升视觉清晰度，增强策略鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.18140](https://arxiv.org/pdf/2511.18140)**

> **作者:** Yilong Wang; Cheng Qian; Ruomeng Fan; Edward Johns
>
> **备注:** Accepted at ICRA 2026. Project Webpage: this https URL
>
> **摘要:** We propose Observer Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wrist-mounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observer's observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policy's observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods -- trajectory transfer and behavior cloning -- and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at this https URL.
>
---
#### [replaced 015] PhysLLM: Harnessing Large Language Models for Cross-Modal Remote Physiological Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.03621](https://arxiv.org/pdf/2505.03621)**

> **作者:** Yiping Xie; Bo Zhao; Mingtong Dai; Jian-Ping Zhou; Yue Sun; Tao Tan; Weicheng Xie; Linlin Shen; Zitong Yu
>
> **备注:** Accepted by International Conference on Learning Representations (ICLR) 2026
>
> **摘要:** Remote photoplethysmography (rPPG) enables non-contact physiological measurement but remains highly susceptible to illumination changes, motion artifacts, and limited temporal modeling. Large Language Models (LLMs) excel at capturing long-range dependencies, offering a potential solution but struggle with the continuous, noise-sensitive nature of rPPG signals due to their text-centric design. To bridge this gap, we introduce the PhysLLM, a collaborative optimization framework that synergizes LLMs with domain-specific rPPG components. Specifically, the Text Prototype Guidance (TPG) strategy is proposed to establish cross-modal alignment by projecting hemodynamic features into LLM-interpretable semantic space, effectively bridging the representational gap between physiological signals and linguistic tokens. Besides, a novel Dual-Domain Stationary (DDS) Algorithm is proposed for resolving signal instability through adaptive time-frequency domain feature re-weighting. Finally, rPPG task-specific cues systematically inject physiological priors through physiological statistics, environmental contextual answering, and task description, leveraging cross-modal learning to integrate both visual and textual information, enabling dynamic adaptation to challenging scenarios like variable illumination and subject movements. Evaluation on four benchmark datasets, PhysLLM achieves state-of-the-art accuracy and robustness, demonstrating superior generalization across lighting variations and motion scenarios. The source code is available at this https URL.
>
---
#### [replaced 016] PowerCLIP: Powerset Alignment for Contrastive Pre-Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23170](https://arxiv.org/pdf/2511.23170)**

> **作者:** Masaki Kawamura; Nakamasa Inoue; Rintaro Yanagi; Hirokatsu Kataoka; Rio Yokota
>
> **摘要:** Contrastive vision-language pre-training frameworks such as CLIP have demonstrated impressive zero-shot performance across a range of vision-language tasks. Recent studies have shown that aligning individual text tokens with specific image patches or regions enhances fine-grained compositional understanding. However, it remains challenging to capture compositional semantics that span multiple image regions. To address this limitation, we propose PowerCLIP, a novel contrastive pre-training framework enhanced by powerset alignment, which exhaustively optimizes region-to-phrase alignments by minimizing the loss defined between powersets of image regions and textual parse trees. Since the naive powerset construction incurs exponential computational cost due to the combinatorial explosion in the number of region subsets, we introduce efficient non-linear aggregators (NLAs) that reduce complexity from O(2^M) to O(M) with respect to the number of regions M, while approximating the exact loss value with arbitrary precision. Our extensive experiments demonstrate that PowerCLIP outperforms state-of-the-art methods in zero-shot classification and retrieval tasks, underscoring the compositionality and robustness of our approach. Our code will be made publicly available.
>
---
#### [replaced 017] AutoV: Loss-Oriented Ranking for Visual Prompt Retrieval in LVLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.16112](https://arxiv.org/pdf/2506.16112)**

> **作者:** Yuan Zhang; Chun-Kai Fan; Sicheng Yu; Junwen Pan; Tao Huang; Ming Lu; Kuan Cheng; Qi She; Shanghang Zhang
>
> **摘要:** Inspired by text prompts in large language models, visual prompts have been explored to enhance the perceptual capabilities of large vision-language models (LVLMs). However, performance tends to saturate under single visual prompt designs, making further prompt engineering increasingly ineffective. To address this limitation, we shift from prompt engineering to prompt retrieval and propose AutoV, a lightweight framework for instance-adaptive visual prompt identification. Given an input image and a textual query, AutoV automatically locates the most suitable visual prompt from a diverse candidate pool. Training such a retrieval framework requires prompt-level supervision, yet prompt quality is inherently ambiguous and difficult to assess reliably, even for humans. To enable automatic supervision, we evaluate visual prompts using a pre-trained LVLM and label them according to their prediction losses. Using the loss-oriented ranking as a robust training signal, AutoV learns to retrieve the query-aware optimal prompt for each instance without manual annotation. Experiments indicate that AutoV enhances the performance of various LVLMs on image understanding, captioning, grounding, and classification tasks. For example, AutoV improves LLaVA-OV by $\textbf{10.2}\%$ on VizWiz and boosts Qwen2.5-VL by $\textbf{3.8}\%$ on MMMU, respectively.
>
---
#### [replaced 018] CCSD: Cross-Modal Compositional Self-Distillation for Robust Brain Tumor Segmentation with Missing Modalities
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14599](https://arxiv.org/pdf/2511.14599)**

> **作者:** Dongqing Xie; Yonghuang Wu; Zisheng Ai; Jun Min; Zhencun Jiang; Shaojin Geng; Lei Wang
>
> **备注:** 29 pages, 5 figures, 6 tables
>
> **摘要:** The accurate segmentation of brain tumors from multi-modal MRI is critical for clinical diagnosis and treatment planning. While integrating complementary information from various MRI sequences is a common practice, the frequent absence of one or more modalities in real-world clinical settings poses a significant challenge, severely compromising the performance and generalizability of deep learning-based segmentation models. To address this challenge, we propose a novel Cross-Modal Compositional Self-Distillation (CCSD) framework that can flexibly handle arbitrary combinations of input modalities. CCSD adopts a shared-specific encoder-decoder architecture and incorporates two self-distillation strategies: (i) a hierarchical modality self-distillation mechanism that transfers knowledge across modality hierarchies to reduce semantic discrepancies, and (ii) a progressive modality combination distillation approach that enhances robustness to missing modalities by simulating gradual modality dropout during training. Extensive experiments on public brain tumor segmentation benchmarks demonstrate that CCSD achieves state-of-the-art performance across various missing-modality scenarios, with strong generalization and stability.
>
---
#### [replaced 019] OSPO: Object-Centric Self-Improving Preference Optimization for Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.02015](https://arxiv.org/pdf/2506.02015)**

> **作者:** Yoonjin Oh; Yongjin Kim; Hyomin Kim; Donghwan Chi; Sungwoong Kim
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have enabled unified multimodal understanding and generation. However, they still struggle with fine-grained text-image alignment, often failing to faithfully depict objects with correct attributes such as color, shape, and spatial relations. To mitigate this issue, previous studies have explored preference optimization methods such as DPO and GRPO, but these approaches incur substantial computational cost, both in constructing preference data and in performing optimization. This has motivated self-improving preference optimization approaches, in which the MLLM autonomously generates its own training data, self-estimates preference feedback, and self-optimizes using the resulting self-constructed preference pairs. However, existing self-improving methods still overlook fine-grained, object-level semantics, allowing object hallucination to persist. To tackle this problem, we propose Object-centric Self-improving Preference Optimization (OSPO), a self-improving framework designed to enhance object-level text-image alignment. OSPO explicitly constructs object-centric preference data without relying on any external data and external models. We also introduce a new approach that leverages attention-based object masks together with an object-weighted SimPO loss to enhance object-specific fidelity. Extensive experiments on three compositional image generation benchmarks demonstrate that OSPO significantly improves fine-grained alignment and reduces object hallucination, outperforming prior self-improving methods and even specialized diffusion-based text-to-image models.
>
---
#### [replaced 020] SASG-DA: Sparse-Aware Semantic-Guided Diffusion Augmentation For Myoelectric Gesture Recognition
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.08344](https://arxiv.org/pdf/2511.08344)**

> **作者:** Chen Liu; Can Han; Weishi Xu; Yaqi Wang; Dahong Qian
>
> **备注:** Accepted by IEEE Journal of Biomedical and Health Informatics (JBHI), 2026
>
> **摘要:** Surface electromyography (sEMG)-based gesture recognition plays a critical role in human-machine interaction (HMI), particularly for rehabilitation and prosthetic control. However, sEMG-based systems often suffer from the scarcity of informative training data, leading to overfitting and poor generalization in deep learning models. Data augmentation offers a promising approach to increasing the size and diversity of training data, where faithfulness and diversity are two critical factors to effectiveness. However, promoting untargeted diversity can result in redundant samples with limited utility. To address these challenges, we propose a novel diffusion-based data augmentation approach, Sparse-Aware Semantic-Guided Diffusion Augmentation (SASG-DA). To enhance generation faithfulness, we introduce the Semantic Representation Guidance (SRG) mechanism by leveraging fine-grained, task-aware semantic representations as generation conditions. To enable flexible and diverse sample generation, we propose a Gaussian Modeling Semantic Sampling (GMSS) strategy, which models the semantic representation distribution and allows stochastic sampling to produce both faithful and diverse samples. To enhance targeted diversity, we further introduce a Sparse-Aware Semantic Sampling strategy to explicitly explore underrepresented regions, improving distribution coverage and sample utility. Extensive experiments on benchmark sEMG datasets, Ninapro DB2, DB4, and DB7, demonstrate that SASG-DA significantly outperforms existing augmentation methods. Overall, our proposed data augmentation approach effectively mitigates overfitting and improves recognition performance and generalization by offering both faithful and diverse samples.
>
---
#### [replaced 021] Motion-Aware Animatable Gaussian Avatars Deblurring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16758](https://arxiv.org/pdf/2411.16758)**

> **作者:** Muyao Niu; Yifan Zhan; Qingtian Zhu; Zhuoxiao Li; Wei Wang; Zhihang Zhong; Xiao Sun; Yinqiang Zheng
>
> **备注:** Accepted at CVPR 2026, Codes: this https URL
>
> **摘要:** The creation of 3D human avatars from multi-view videos is a significant yet challenging task in computer vision. However, existing techniques rely on high-quality, sharp images as input, which are often impractical to obtain in real-world scenarios due to variations in human motion speed and intensity. This paper introduces a novel method for directly reconstructing sharp 3D human Gaussian avatars from blurry videos. The proposed approach incorporates a 3D-aware, physics-based model of blur formation caused by human motion, together with a 3D human motion model designed to resolve ambiguities in motion-induced blur. This framework enables the joint optimization of the avatar representation and motion parameters from a coarse initialization. Comprehensive benchmarks are established using both a synthetic dataset and a real-world dataset captured with a 360-degree synchronous hybrid-exposure camera system. Extensive evaluations demonstrate the effectiveness of the model across diverse conditions. Codes Available: this https URL
>
---
#### [replaced 022] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.24290](https://arxiv.org/pdf/2602.24290)**

> **作者:** Junhwa Hur; Charles Herrmann; Songyou Peng; Philipp Henzler; Zeyu Ma; Todd Zickler; Deqing Sun
>
> **备注:** ICLR 2026, Project page: this https URL
>
> **摘要:** Dense 4D reconstruction from unposed images remains a critical challenge, with current methods relying on slow test-time optimization or fragmented, task-specific feedforward models. We introduce UFO-4D, a unified feedforward framework to reconstruct a dense, explicit 4D representation from just a pair of unposed images. UFO-4D directly estimates dynamic 3D Gaussian Splats, enabling the joint and consistent estimation of 3D geometry, 3D motion, and camera pose in a feedforward manner. Our core insight is that differentiably rendering multiple signals from a single Dynamic 3D Gaussian representation offers major training advantages. This approach enables a self-supervised image synthesis loss while tightly coupling appearance, depth, and motion. Since all modalities share the same geometric primitives, supervising one inherently regularizes and improves the others. This synergy overcomes data scarcity, allowing UFO-4D to outperform prior work by up to 3 times in joint geometry, motion, and camera pose estimation. Our representation also enables high-fidelity 4D interpolation across novel views and time. Please visit our project page for visual results: this https URL
>
---
#### [replaced 023] DAP: A Discrete-token Autoregressive Planner for Autonomous Driving
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13306](https://arxiv.org/pdf/2511.13306)**

> **作者:** Bowen Ye; Bin Zhang; Hang Zhao
>
> **摘要:** Gaining sustainable performance improvement with scaling data and model budget remains a pivotal yet unresolved challenge in autonomous driving. While autoregressive models exhibited promising data-scaling efficiency in planning tasks, predicting ego trajectories alone suffers sparse supervision and weakly constrains how scene evolution should shape ego motion. Therefore, we introduce DAP, a discrete-token autoregressive planner that jointly forecasts BEV semantics and ego trajectories, thereby enforcing comprehensive representation learning and allowing predicted dynamics to directly condition ego motion. In addition, we incorporate a reinforcement-learning-based fine-tuning, which preserves supervised behavior cloning priors while injecting reward-guided improvements. Despite a compact 160M parameter budget, DAP achieves state-of-the-art performance on open-loop metrics and delivers competitive closed-loop results on the NAVSIM benchmark. Overall, the fully discrete-token autoregressive formulation operating on both rasterized BEV and ego actions provides a compact yet scalable planning paradigm for autonomous driving.
>
---
#### [replaced 024] Bidirectional Temporal Dynamics Modeling for EEG-based Driving Fatigue Recognition
- **分类: cs.OH; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.14071](https://arxiv.org/pdf/2602.14071)**

> **作者:** Yip Tin Po; Jianming Wang; Yutao Miao; Jiayan Zhang; Yunxu Zhao; Xiaomin Ouyang; Zhihong Li; Nevin L. Zhang
>
> **摘要:** Driving fatigue is a major contributor to traffic accidents and poses a serious threat to road safety. Electroencephalography (EEG) provides a direct measurement of neural activity, yet EEG-based fatigue recognition is hindered by strong non-stationarity and asymmetric neural dynamics. To address these challenges, we propose DeltaGateNet, a novel framework that explicitly captures Bidirectional temporal dynamics for EEG-based driving fatigue recognition. Our key idea is to introduce a Bidirectional Delta module that decomposes first-order temporal differences into positive and negative components, enabling explicit modeling of asymmetric neural activation and suppression patterns. Furthermore, we design a Gated Temporal Convolution module to capture long-term temporal dependencies for each EEG channel using depthwise temporal convolutions and residual learning, preserving channel-wise specificity while enhancing temporal representation robustness. Extensive experiments conducted under both intra-subject and inter-subject evaluation settings on the public SEED-VIG and SADT driving fatigue datasets demonstrate that DeltaGateNet consistently outperforms existing methods. On SEED-VIG, DeltaGateNet achieves an intra-subject accuracy of 81.89% and an inter-subject accuracy of 55.55%. On the balanced SADT 2022 dataset, it attains intra-subject and inter-subject accuracies of 96.81% and 83.21%, respectively, while on the unbalanced SADT 2952 dataset, it achieves 96.84% intra-subject and 84.49% inter-subject accuracy. These results indicate that explicitly modeling Bidirectional temporal dynamics yields robust and generalizable performance under varying subject and class-distribution conditions.
>
---
#### [replaced 025] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.02282](https://arxiv.org/pdf/2510.02282)**

> **作者:** Kyoungjun Park; Yifan Yang; Juheon Yi; Shicheng Zheng; Yifei Shen; Dongqi Han; Caihua Shan; Muhammad Muaz; Lili Qiu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** The rapid proliferation of AI-generated video necessitates robust detection tools that offer both high accuracy and human-interpretable explanations. While existing MLLM-based detectors rely on supervised fine-tuning (SFT) or direct preference optimization (DPO), these methods are often bottlenecked by static, pre-labeled datasets that fail to capture the evolving, multi-step physical inconsistencies of modern generative models. To bridge this gap, we introduce VidGuard-R1, the first video authenticity detector to utilize group relative policy optimization (GRPO). Moving beyond passive preference matching, VidGuard-R1 employs a reinforcement learning framework that encourages the model to explore and rank multiple reasoning paths. By introducing specialized reward models for temporal stability and diffusion-aware complexity, we incentivize the model to discover 'physics-grounded' artifacts. Our contributions include: (1) a curated dataset of 140,000 challenging real/fake video pairs; (2) a GRPO-based training paradigm that achieves state-of-the-art zero-shot performance; and (3) a reasoning-first architecture that provides precise, verifiable rationales for its forensic judgments. Project website: this https URL.
>
---
#### [replaced 026] True Self-Supervised Novel View Synthesis is Transferable
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.13063](https://arxiv.org/pdf/2510.13063)**

> **作者:** Thomas W. Mitchel; Hyunwoo Ryu; Vincent Sitzmann
>
> **摘要:** In this paper, we identify that the key criterion for determining whether a model is truly capable of novel view synthesis (NVS) is transferability: Whether any pose representation extracted from one video sequence can be used to re-render the same camera trajectory in another. We analyze prior work on self-supervised NVS and find that their predicted poses do not transfer: The same set of poses lead to different camera trajectories in different 3D scenes. Here, we present XFactor, the first geometry-free self-supervised model capable of true NVS. XFactor combines pair-wise pose estimation with a simple augmentation scheme of the inputs and outputs that jointly enables disentangling camera pose from scene content and facilitates geometric reasoning. Remarkably, we show that XFactor achieves transferability with unconstrained latent pose variables, without any 3D inductive biases or concepts from multi-view geometry -- such as an explicit parameterization of poses as elements of SE(3). We introduce a new metric to quantify transferability, and through large-scale experiments, we demonstrate that XFactor significantly outperforms prior pose-free NVS transformers, and show that latent poses are highly correlated with real-world poses through probing experiments.
>
---
#### [replaced 027] 3D Dynamics-Aware Manipulation: Endowing Manipulation Policies with 3D Foresight
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决深度运动下2D建模不足的问题。通过引入3D动态建模与策略学习，提升操作策略的3D预见能力。**

- **链接: [https://arxiv.org/pdf/2502.10028](https://arxiv.org/pdf/2502.10028)**

> **作者:** Yuxin He; Ruihao Zhang; Xianzu Wu; Zhiyuan Zhang; Cheng Ding; Qiang Nie
>
> **备注:** ICRA 2026
>
> **摘要:** The incorporation of world modeling into manipulation policy learning has pushed the boundary of manipulation performance. However, existing efforts simply model the 2D visual dynamics, which is insufficient for robust manipulation when target tasks involve prominent depth-wise movement. To address this, we present a 3D dynamics-aware manipulation framework that seamlessly integrates 3D world modeling and policy learning. Three self-supervised learning tasks (current depth estimation, future RGB-D prediction, 3D flow prediction) are introduced within our framework, which complement each other and endow the policy model with 3D foresight. Extensive experiments on simulation and the real world show that 3D foresight can greatly boost the performance of manipulation policies without sacrificing inference speed. Code is available at this https URL.
>
---
#### [replaced 028] Fully Automatic Data Labeling for Ultrasound Screen Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13197](https://arxiv.org/pdf/2511.13197)**

> **作者:** Alberto Gomez; Jorge Oliveira; Ramon Casero; Agis Chartsias
>
> **备注:** Submitted to ISBI AI-POCUS workshop 2026
>
> **摘要:** Ultrasound (US) machines display images on a built-in monitor, but routine transfer to hospital systems relies on DICOM. We propose a fully automatic method to generate labeled data that can be used to train a screen detector model, and a pipeline to use that model to extract and rectify the US image from a photograph of the monitor, without any need for human annotation. This removes the DICOM bottleneck and enables rapid testing and prototyping of new algorithms. In a proof-of-concept study, the rectified images retained enough visual fidelity to classify cardiac views with a balanced accuracy of 0.79 with respect to the native DICOMs., the rectified images retained enough visual fidelity to classify cardiac views with a balanced accuracy of 0.79 with respect to the native DICOMs.
>
---
#### [replaced 029] Diffusion Probe: Generated Image Result Prediction Using CNN Probes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23783](https://arxiv.org/pdf/2602.23783)**

> **作者:** Benlei Cui; Bukun Huang; Zhizeng Ye; Xuemei Dong; Tuo Chen; Hui Xue; Dingkang Yang; Longtao Huang; Jingqun Tang; Haiwen Hong
>
> **备注:** CVPR 2026
>
> **摘要:** Text-to-image (T2I) diffusion models lack an efficient mechanism for early quality assessment, leading to costly trial-and-error in multi-generation scenarios such as prompt iteration, agent-based generation, and flow-grpo. We reveal a strong correlation between early diffusion cross-attention distributions and final image quality. Based on this finding, we introduce Diffusion Probe, a framework that leverages internal cross-attention maps as predictive signals. We design a lightweight predictor that maps statistical properties of early-stage cross-attention extracted from initial denoising steps to the final image's overall quality. This enables accurate forecasting of image quality across diverse evaluation metrics long before full synthesis is complete. We validate Diffusion Probe across a wide range of settings. On multiple T2I models, across early denoising windows, resolutions, and quality metrics, it achieves strong correlation (PCC > 0.7) and high classification performance (AUC-ROC > 0.9). Its reliability translates into practical gains. By enabling early quality-aware decisions in workflows such as prompt optimization, seed selection, and accelerated RL training, the probe supports more targeted sampling and avoids computation on low-potential generations. This reduces computational overhead while improving final output this http URL Probe is model-agnostic, efficient, and broadly applicable, offering a practical solution for improving T2I generation efficiency through early quality prediction.
>
---
#### [replaced 030] Agentic Very Long Video Understanding
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.18157](https://arxiv.org/pdf/2601.18157)**

> **作者:** Aniket Rege; Arka Sadhu; Yuliang Li; Kejie Li; Ramya Korlakai Vinayak; Yuning Chai; Yong Jae Lee; Hyo Jin Kim
>
> **备注:** 27 pages, 7 figures, 8 tables
>
> **摘要:** The advent of always-on personal AI assistants, enabled by all-day wearable devices such as smart glasses, demands a new level of contextual understanding, one that goes beyond short, isolated events to encompass the continuous, longitudinal stream of egocentric video. Achieving this vision requires advances in long-horizon video understanding, where systems must interpret and recall visual and audio information spanning days or even weeks. Existing methods, including large language models and retrieval-augmented generation, are constrained by limited context windows and lack the ability to perform compositional, multi-hop reasoning over very long video streams. In this work, we address these challenges through EGAgent, an enhanced agentic framework centered on entity scene graphs, which represent people, places, objects, and their relationships over time. Our system equips a planning agent with tools for structured search and reasoning over these graphs, as well as hybrid visual and audio search capabilities, enabling detailed, cross-modal, and temporally coherent reasoning. Experiments on the EgoLifeQA and Video-MME (Long) datasets show that our method achieves state-of-the-art performance on EgoLifeQA (57.5%) and competitive performance on Video-MME (Long) (74.1%) for complex longitudinal video understanding tasks. Code is available at this https URL.
>
---
#### [replaced 031] Hyperspherical Latents Improve Continuous-Token Autoregressive Generation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.24335](https://arxiv.org/pdf/2509.24335)**

> **作者:** Guolin Ke; Hui Xue
>
> **备注:** ICLR version
>
> **摘要:** Autoregressive (AR) models are promising for image generation, yet continuous-token AR variants often trail latent diffusion and masked-generation models. The core issue is heterogeneous variance in VAE latents, which is amplified during AR decoding, especially under classifier-free guidance (CFG), and can cause variance collapse. We propose SphereAR to address this issue. Its core design is to constrain all AR inputs and outputs -- including after CFG -- to lie on a fixed-radius hypersphere (constant $\ell_2$ norm), leveraging hyperspherical VAEs. Our theoretical analysis shows that hyperspherical constraint removes the scale component (the primary cause of variance collapse), thereby stabilizing AR decoding. Empirically, on ImageNet generation, SphereAR-H (943M) sets a new state of the art for AR models, achieving FID 1.34. Even at smaller scales, SphereAR-L (479M) reaches FID 1.54 and SphereAR-B (208M) reaches 1.92, matching or surpassing much larger baselines such as MAR-H (943M, 1.55) and VAR-d30 (2B, 1.92). To our knowledge, this is the first time a pure next-token AR image generator with raster order surpasses diffusion and masked-generation models at comparable parameter scales.
>
---
#### [replaced 032] DRBD-Mamba for Robust and Efficient Brain Tumor Segmentation with Analytical Insights
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.14383](https://arxiv.org/pdf/2510.14383)**

> **作者:** Danish Ali; Ajmal Mian; Naveed Akhtar; Ghulam Mubashar Hassan
>
> **摘要:** Accurate brain tumor segmentation is significant for clinical diagnosis and treatment but remains challenging due to tumor heterogeneity. Mamba-based State Space Models have demonstrated promising performance. However, despite their computational efficiency over other neural architectures, they incur considerable overhead for this task due to their sequential feature computation across multiple spatial axes. Moreover, their robustness across diverse BraTS data partitions remains largely unexplored, leaving a critical gap in reliable evaluation. To address this, we first propose a dual-resolution bi-directional Mamba (DRBD-Mamba), an efficient 3D segmentation model that captures multi-scale long-range dependencies with minimal computational overhead. We leverage a space-filling curve to preserve spatial locality during 3D-to-1D feature mapping, thereby reducing reliance on computationally expensive multi-axial feature scans. To enrich feature representation, we propose a gated fusion module that adaptively integrates forward and reverse contexts, along with a quantization block that improves robustness. We further propose five systematic folds on BraTS2023 for rigorous evaluation of segmentation techniques under diverse conditions and present analysis of common failure scenarios. On the 20% test set used by recent methods, our model achieves Dice improvements of 0.10% for whole tumor, 1.75% for tumor core, and 0.93% for enhancing tumor. Evaluations on the proposed systematic folds demonstrate that our model maintains competitive whole tumor accuracy while achieving clear average Dice gains of 1.16% for tumor core and 1.68% for enhancing tumor over existing state-of-the-art. Furthermore, our model achieves a 15x efficiency improvement while maintaining high segmentation accuracy, highlighting its robustness and computational advantage over existing methods.
>
---
#### [replaced 033] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.04764](https://arxiv.org/pdf/2506.04764)**

> **作者:** Suhan Woo; Seongwon Lee; Jinwoo Jang; Euntai Kim
>
> **备注:** CVPR 2026
>
> **摘要:** Visual environments are inherently hierarchical, as a panoramic view naturally encompasses and organizes multiple perspective views within its field. Capturing this hierarchy is crucial for effective perspective-to-equirectangular (P2E) visual place recognition. In this work, we introduce HypeVPR, a hierarchical embedding framework in hyperbolic space specifically designed to address the challenges of P2E matching. HypeVPR leverages the intrinsic ability of hyperbolic space to represent hierarchical structures, allowing panoramic descriptors to encode both broad contextual information and fine-grained local details. To this end, we propose a hierarchical feature aggregation mechanism that organizes local-to-global feature representations within hyperbolic space. Furthermore, HypeVPR's hierarchical organization naturally enables flexible control over the accuracy-efficiency trade-off without additional training, while maintaining robust matching across different image types. This approach enables HypeVPR to achieve competitive performance while significantly accelerating retrieval and reducing database storage requirements. Project page: this https URL
>
---
#### [replaced 034] Pailitao-VL: Unified Embedding and Reranker for Real-Time Multi-Modal Industrial Search
- **分类: cs.IR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.13704](https://arxiv.org/pdf/2602.13704)**

> **作者:** Lei Chen; Chen Ju; Xu Chen; Zhicheng Wang; Yuheng Jiao; Hongfeng Zhan; Zhaoyang Li; Shihao Xu; Zhixiang Zhao; Tong Jia; Lin Li; Yuan Gao; Jun Song; Jinsong Lan; Xiaoyong Zhu; Bo Zheng
>
> **摘要:** In this work, we presented Pailitao-VL, a comprehensive multi-modal retrieval system engineered for high-precision, real-time industrial search. We here address three critical challenges in the current SOTA solution: insufficient retrieval granularity, vulnerability to environmental noise, and prohibitive efficiency-performance gap. Our primary contribution lies in two fundamental paradigm shifts. First, we transitioned the embedding paradigm from traditional contrastive learning to an absolute ID-recognition task. Through anchoring instances to a globally consistent latent space defined by billions of semantic prototypes, we successfully overcome the stochasticity and granularity bottlenecks inherent in existing embedding solutions. Second, we evolved the generative reranker from isolated pointwise evaluation to the compare-and-calibrate listwise policy. By synergizing chunk-based comparative reasoning with calibrated absolute relevance scoring, the system achieves nuanced discriminative resolution while circumventing the prohibitive latency typically associated with conventional reranking methods. Extensive offline benchmarks and online A/B tests on Alibaba e-commerce platform confirm that Pailitao-VL achieves state-of-the-art performance and delivers substantial business impact. This work demonstrates a robust and scalable path for deploying advanced MLLM-based retrieval architectures in demanding, large-scale production environments.
>
---
#### [replaced 035] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.18534](https://arxiv.org/pdf/2507.18534)**

> **作者:** Xingyu Qiu; Mengying Yang; Xinghua Ma; Dong Liang; Fanding Li; Gongning Luo; Wei Wang; Kuanquan Wang; Shuo Li
>
> **备注:** 16 pages, 4 figures, accepted by CVPR 2026
>
> **摘要:** Although EDM aims to unify the design space of diffusion models, its reliance on fixed Gaussian noise prevents it from explaining emerging flow-based methods that diffuse arbitrary noise. Moreover, our study reveals that EDM's forcible injection of Gaussian noise has adverse effects on image restoration task, as it corrupts the degraded images, overextends the restoration distance, and increases the task's complexity. To interpret diverse methods for handling distinct noise patterns within a unified theoretical framework and to minimize the restoration distance, we propose EDA, which Elucidates the Design space of Arbitrary-noise diffusion models. Theoretically, EDA expands noise pattern flexibility while preserving EDM's modularity, with rigorous proof that increased noise complexity introduces no additional computational overhead during restoration. EDA is validated on three representative medical image denoising and natural image restoration tasks: MRI bias field correction (global smooth noise), CT metal artifact removal (global sharp noise) and natural image shadow removal (local boundary-aware noise). With only 5 sampling steps, competitive results against specialized methods across medical and natural tasks demonstrate EDA's strong generalization capability for image restoration. Code is available at: this https URL.
>
---
#### [replaced 036] UniComp: Rethinking Video Compression Through Informational Uniqueness
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03575](https://arxiv.org/pdf/2512.03575)**

> **作者:** Chao Yuan; Shimin Chen; Minliang Lin; Limeng Qiao; Guanglu Wan; Lin Ma
>
> **摘要:** Distinct from attention-based compression methods, this paper presents an information uniqueness driven video compression framework, termed UniComp, which aims to maximize the information fidelity of video representations under constrained computational budgets. Starting from the information-theoretic perspective, we formulate the vision compression as an optimization problem that minimizes conditional entropy (reconstruction error) between retained and full tokens. To achieve this, we introduce the notion of information uniqueness to measure intrinsic redundancy among tokens to link with reconstruction error. Based on uniqueness, we design three modules-Frame Group Fusion, Token Allocation, and Spatial Dynamic Compression-that progressively perform semantic frame grouping, adaptive resource allocation, and fine-grained spatial compression. Extensive experiments demonstrate that UniComp consistently outperforms existing compression methods in preserving essential visual tokens under limited computational budgets, highlighting the pivotal role of information uniqueness in token compression efficacy.
>
---
#### [replaced 037] Graph-Based Multi-Modal Light-weight Network for Adaptive Brain Tumor Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.09995](https://arxiv.org/pdf/2507.09995)**

> **作者:** Guohao Huo; Ruiting Dai; Zitong Wang; Junxin Kong; Hao Tang
>
> **摘要:** Multi-modal brain tumor segmentation remains challenging for practical deployment due to the high computational costs of mainstream models. In this work, we propose GMLN-BTS, a Graph-based Multi-modal interaction Lightweight Network for brain tumor segmentation. Our architecture achieves high-precision, resource-efficient segmentation through three key components. First, a Modality-Aware Adaptive Encoder (M2AE) facilitates efficient multi-scale semantic extraction. Second, a Graph-based Multi-Modal Collaborative Interaction Module (G2MCIM) leverages graph structures to model complementary cross-modal relationships. Finally, a Voxel Refinement UpSampling Module (VRUM) integrates linear interpolation with multi-scale transposed convolutions to suppress artifacts and preserve boundary details. Experimental results on BraTS 2017, 2019, and 2021 benchmarks demonstrate that GMLN-BTS achieves state-of-the-art performance among lightweight models. With only 4.58M parameters, our method reduces parameter count by 98% compared to mainstream 3D Transformers while significantly outperforming existing compact approaches.
>
---
#### [replaced 038] EmboTeam: Grounding LLM Reasoning into Reactive Behavior Trees via PDDL for Embodied Multi-Robot Collaboration
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出EmboTeam框架，解决多机器人协作中的长周期任务规划问题。通过LLM与经典规划器结合，生成行为树实现动态协调。**

- **链接: [https://arxiv.org/pdf/2601.11063](https://arxiv.org/pdf/2601.11063)**

> **作者:** Haishan Zeng; Mengna Wang; Peng Li
>
> **摘要:** In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose EmboTeam, a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experiments show EmboTeam improves the task success rate from 12% to 55% and goal condition recall from 32% to 72% over the LaMMA-P baseline.
>
---
#### [replaced 039] STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19854](https://arxiv.org/pdf/2511.19854)**

> **作者:** Jiankuo Zhao; Xiangyu Zhu; Zidu Wang; Zhen Lei
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Reconstructing high-fidelity and animatable 3D head avatars from monocular videos remains a challenging yet essential task. Existing methods based on 3D Gaussian Splatting typically bind Gaussians to mesh triangles and model deformations solely via Linear Blend Skinning, which results in rigid motion and limited expressiveness. Moreover, they lack specialized strategies to handle frequently occluded regions (e.g., mouth interiors, eyelids). To address these limitations, we propose STAvatar, which consists of two key components: (1) a UV-Adaptive Soft Binding framework that leverages both image-based and geometric priors to learn per-Gaussian feature offsets within the UV space. This UV representation supports dynamic resampling, ensuring full compatibility with Adaptive Density Control (ADC) and enhanced adaptability to shape and textural variations. (2) a Temporal ADC strategy, which first clusters structurally similar frames to facilitate more targeted computation of the densification criterion. It further introduces a novel fused perceptual error as clone criterion to jointly capture geometric and textural discrepancies, encouraging densification in regions requiring finer details. Extensive experiments on four benchmark datasets demonstrate that STAvatar achieves state-of-the-art reconstruction performance, especially in capturing fine-grained details and reconstructing frequently occluded regions.
>
---
#### [replaced 040] SAMPO-Path: Segmentation Intent-Aligned Preference Optimization for Pathology Foundation Model Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02464](https://arxiv.org/pdf/2508.02464)**

> **作者:** Yonghuang Wu; Wenwen Zeng; Xuan Xie; Chengqian Zhao; Guoqing Wu; Jinhua Yu
>
> **备注:** 15 pages, 9 tables, 8 figures
>
> **摘要:** Foundation models have shown strong performance in multi-object segmentation with visual prompts, yet histopathology images remain challenging due to high cellular density, heterogeneity, and the gap between pixel-level supervision and clinical segmentation intent (e.g., selectively segmenting nuclei of a specific type). In practice, such intents are expressed through diverse and noisy prompts, causing prompt-intent misalignment and inconsistent predictions. We introduce SAMPO (Segmentation Anything Model with Preference Optimization), a preference-aligned fine-tuning framework that explicitly aligns pathology foundation models with clinical segmentation intent. SAMPO is the first to adapt Direct Preference Optimization (DPO) to pure vision foundation models, enabling accurate segmentation from minimal and imperfect prompts. The framework features three key components: (1) online prompt-centric preference mining to synthesize preference pairs across prompt qualities; (2) multi-mask preference learning to leverage output ambiguity for fine-grained ranking supervision; and (3) a hybrid loss combining preference optimization with pixel-level supervision for stable training. Trained on two datasets covering four tasks and evaluated on corresponding test sets and 12 external validation datasets, SAMPO consistently improves segmentation accuracy, robustness to prompt variations, and clinical intent adherence in dense histopathology images.
>
---
#### [replaced 041] FluenceFormer: Transformer-Driven Multi-Beam Fluence Map Regression for Radiotherapy Planning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.22425](https://arxiv.org/pdf/2512.22425)**

> **作者:** Ujunwa Mgboh; Rafi Ibn Sultan; Joshua Kim; Kundan Thind; Dongxiao Zhu
>
> **备注:** Accepted at Medical Imaging with Deep Learning (MIDL-2026)
>
> **摘要:** Fluence map prediction is central to automated radiotherapy planning but remains an ill-posed inverse problem due to the complex relationship between volumetric anatomy and beam-intensity modulation. Convolutional methods in prior work often struggle to capture long-range dependencies, which can lead to structurally inconsistent or physically unrealizable plans. We introduce \textbf{FluenceFormer}, a backbone-agnostic transformer framework for direct, geometry-aware fluence regression. The model uses a unified two-stage design: Stage~1 predicts a global dose prior from anatomical inputs, and Stage~2 conditions this prior on explicit beam geometry to regress physically calibrated fluence maps. Central to the approach is the \textbf{Fluence-Aware Regression (FAR)} loss, a physics-informed objective that integrates voxel-level fidelity, gradient smoothness, structural consistency, and beam-wise energy conservation. We evaluate the generality of the framework across multiple transformer backbones, including Swin UNETR, UNETR, nnFormer, and MedFormer, using a prostate IMRT dataset. FluenceFormer with Swin UNETR achieves the strongest performance among the evaluated models and improves over existing benchmark CNN and single-stage methods, reducing Energy Error to $\mathbf{4.5\%}$ and yielding statistically significant gains in structural fidelity ($p < 0.05$).
>
---
#### [replaced 042] TumorFlow: Physics-Guided Longitudinal MRI Synthesis of Glioblastoma Growth
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.04058](https://arxiv.org/pdf/2603.04058)**

> **作者:** Valentin Biller; Niklas Bubeck; Lucas Zimmer; Ayhan Can Erdur; Sandeep Nagar; Anke Meyer-Baese; Daniel Rückert; Benedikt Wiestler; Jonas Weidner
>
> **摘要:** Glioblastoma exhibits diverse, infiltrative, and patient-specific growth patterns that are only partially visible on routine MRI, making it difficult to reliably assess true tumor extent and personalize treatment planning and follow-up. We present a biophysically-conditioned generative framework that synthesizes biologically realistic 3D brain MRI volumes from estimated, spatially continuous tumor-concentration fields. Our approach combines a generative model with tumor-infiltration maps that can be propagated through time using a biophysical growth model, enabling fine-grained control over tumor shape and growth while preserving patient anatomy. This enables us to synthesize consistent tumor growth trajectories directly in the space of real patients, providing interpretable, controllable estimation of tumor infiltration and progression beyond what is explicitly observed in imaging. We evaluate the framework on longitudinal glioblastoma cases and demonstrate that it can generate temporally coherent sequences with realistic changes in tumor appearance and surrounding tissue response. These results suggest that integrating mechanistic tumor growth priors with modern generative modeling can provide a practical tool for patient-specific progression visualization and for generating controlled synthetic data to support downstream neuro-oncology workflows. In longitudinal extrapolation, we achieve a consistent 75% Dice overlap with the biophysical model while maintaining a constant PSNR of 25 in the surrounding tissue. Our code is available at: this https URL
>
---
#### [replaced 043] Revolutionizing Mixed Precision Quantization: Towards Training-free Automatic Proxy Discovery via Large Language Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07419](https://arxiv.org/pdf/2512.07419)**

> **作者:** Haidong Kang; Jun Du; Lihong Lin
>
> **摘要:** Mixed-Precision Quantization (MPQ) liberates Deep Neural Networks (DNNs) from the Out-Of-Memory (OOM) bottleneck and has garnered increasing research attention. However, conventional methods either rely on costly differentiable optimization search, which is neither efficient nor flexible, or learn a quantized DNN from a proxy (e.g., HAWQ) manually designed by human experts, which is labor-intensive and requires extensive expert knowledge. Can we design a proxy without involving any human experts or training? In this paper, we provide an affirmative answer by proposing a novel Large Language Model (LLM)-driven Training-free Automatic Proxy (dubbed TAP) discovery framework. It reforms the design paradigm of MPQ by utilizing LLMs and evolutionary search strategies to automatically find superior TAP tailored for MPQ. In addition, to bridge the gap between black-box LLMs and the challenging MPQ task, we introduce a lightweight Direct Preference Optimization (DPO)-based strategy controller that dynamically reweights the selection probabilities of the three prompt templates for evolutionary search strategies according to fitness signals, without fine-tuning the LLM. This forms a task-aware feedback loop that improves proxy generation across evolutions. Extensive experiments on mainstream benchmarks demonstrate that TAP achieves state-of-the-art performance. Finally, we believe that our TAP will significantly contribute to the MPQ community by providing a new perspective on LLM-driven design algorithms.
>
---
#### [replaced 044] Rolling Sink: Bridging Limited-Horizon Training and Open-Ended Testing in Autoregressive Video Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07775](https://arxiv.org/pdf/2602.07775)**

> **作者:** Haodong Li; Shaoteng Liu; Zhe Lin; Manmohan Chandraker
>
> **备注:** Figure PDFs were compressed to 150 dpi to comply with arXiv's submission size limit. Project page: this https URL
>
> **摘要:** Recently, autoregressive (AR) video diffusion models has achieved remarkable performance. However, due to their limited training durations, a train-test gap emerges when testing at longer horizons, leading to rapid visual degradations. Following Self Forcing, which studies the train-test gap within the training duration, this work studies the train-test gap beyond the training duration, i.e., the gap between the limited horizons during training and open-ended horizons during testing. Since open-ended testing can extend beyond any finite training window, and long-video training is computationally expensive, we pursue a training-free solution to bridge this gap. To explore a training-free solution, we conduct a systematic analysis of AR cache maintenance. These insights lead to Rolling Sink. Built on Self Forcing (trained on only 5s clips), Rolling Sink effectively scales the AR video synthesis to ultra-long durations (e.g., 5-30 minutes at 16 FPS) at test time, with consistent subjects, stable colors, coherent structures, and smooth motions. As demonstrated by extensive experiments, Rolling Sink achieves superior long-horizon visual fidelity and temporal consistency compared to SOTA baselines. Project page: this https URL
>
---
#### [replaced 045] Dr. Seg: Revisiting GRPO Training for Visual Large Language Models through Perception-Oriented Design
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.00152](https://arxiv.org/pdf/2603.00152)**

> **作者:** Haoxiang Sun; Tao Wang; Chenwei Tang; Li Yuan; Jiancheng Lv
>
> **摘要:** Following the success of Group Relative Policy Optimization (GRPO) in foundation LLMs, an increasing number of works have sought to adapt GRPO to Visual Large Language Models (VLLMs) for visual perception tasks (e.g., detection and segmentation). However, much of this line of research rests on a long-standing yet unexamined assumption: training paradigms developed for language reasoning can be transferred seamlessly to visual perception. Our experiments show that this assumption is not valid, revealing intrinsic differences between reasoning-oriented and perception-oriented settings. Using reasoning segmentation as a representative case, we surface two overlooked factors: (i) the need for a broader output space, and (ii) the importance of fine-grained, stable rewards. Building on these observations, we propose Dr.~Seg, a simple, plug-and-play GRPO-based framework consisting of a Look-to-Confirm mechanism and a Distribution-Ranked Reward module, requiring no architectural modifications and integrating seamlessly with existing GRPO-based VLLMs. Extensive experiments demonstrate that Dr.~Seg improves performance in complex visual scenarios while maintaining strong generalization. Code, models, and datasets are available at this https URL.
>
---
#### [replaced 046] ReactDance: Hierarchical Representation for High-Fidelity and Coherent Long-Form Reactive Dance Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.05589](https://arxiv.org/pdf/2505.05589)**

> **作者:** Jingzhong Lin; Xinru Li; Yuanyuan Qi; Bohao Zhang; Wenxiang Liu; Kecheng Tang; Wenxuan Huang; Xiangfeng Xu; Bangyan Li; Changbo Wang; Gaoqi He
>
> **摘要:** Reactive dance generation (RDG), the task of generating a dance conditioned on a lead dancer's motion, holds significant promise for enhancing human-robot interaction and immersive digital entertainment. Despite progress in duet synchronization and motion-music alignment, two key challenges remain: generating fine-grained spatial interactions and ensuring long-term temporal coherence. In this work, we introduce \textbf{ReactDance}, a diffusion framework that operates on a novel hierarchical latent space to address these spatiotemporal challenges in RDG. First, for high-fidelity spatial expression and fine-grained control, we propose Hierarchical Finite Scalar Quantization (\textbf{HFSQ}). This multi-scale motion representation effectively disentangles coarse body posture from subtle limb dynamics, enabling independent and detailed control over both aspects through a layered guidance mechanism. Second, to efficiently generate long sequences with high temporal coherence, we propose Blockwise Local Context (\textbf{BLC}), a non-autoregressive sampling strategy. Departing from slow, frame-by-frame generation, BLC partitions the sequence into blocks and synthesizes them in parallel via periodic causal masking and positional encodings. Coherence across these blocks is ensured by a dense sliding-window training approach that enriches the representation with local temporal context. Extensive experiments show that ReactDance substantially outperforms state-of-the-art methods in motion quality, long-term coherence, and sampling efficiency. Project page: this https URL.
>
---
#### [replaced 047] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21637](https://arxiv.org/pdf/2602.21637)**

> **作者:** Di Zhang; Zhangpeng Gong; Xiaobo Pang; Jiashuai Liu; Junbo Lu; Hao Cui; Jiusong Ge; Zhi Zeng; Kai Yi; Yinghua Li; Si Liu; Tingsong Yu; Haoran Wang; Mireia Crispin-Ortuzar; Weimiao Yu; Chen Li; Zeyu Gao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Foundation models have recently achieved impressive success in computational pathology, demonstrating strong generalization across diverse histopathology tasks. However, existing models overlook the heterogeneous and non-uniform organization of pathological regions of interest (ROIs) because they rely on natural image backbones not tailored for tissue morphology. Consequently, they often fail to capture the coherent tissue architecture beyond isolated patches, limiting interpretability and clinical relevance. To address these challenges, we present Cross-modal Adaptive Region Encoder (CARE), a foundation model for pathology that automatically partitions WSIs into several morphologically relevant regions. Specifically, CARE employs a two-stage pretraining strategy: (1) a self-supervised unimodal pretraining stage that learns morphological representations from 34,277 whole-slide images (WSIs) without segmentation annotations, and (2) a cross-modal alignment stage that leverages RNA and protein profiles to refine the construction and representation of adaptive regions. This molecular guidance enables CARE to identify biologically relevant patterns and generate irregular yet coherent tissue regions, selecting the most representative area as ROI. CARE supports a broad range of pathology-related tasks, using either the ROI feature or the slide-level feature obtained by aggregating adaptive regions. Based on only one-tenth of the pretraining data typically used by mainstream foundation models, CARE achieves superior average performance across 33 downstream benchmarks, including morphological classification, molecular prediction, and survival analysis, and outperforms other foundation model baselines overall.
>
---
#### [replaced 048] FLAIR-HUB: Large-scale Multimodal Dataset for Land Cover and Crop Mapping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.07080](https://arxiv.org/pdf/2506.07080)**

> **作者:** Anatol Garioud; Sébastien Giordano; Nicolas David; Nicolas Gonthier
>
> **摘要:** The growing availability of high-quality Earth Observation (EO) data enables accurate global land cover and crop type monitoring. However, the volume and heterogeneity of these datasets pose major processing and annotation challenges. To address this, the French National Institute of Geographical and Forest Information (IGN) is actively exploring innovative strategies to exploit diverse EO data, which require large annotated datasets. IGN introduces FLAIR-HUB, the largest multi-sensor land cover dataset with very-high-resolution (20 cm) annotations, covering 2528 km2 of France. It combines six aligned modalities: aerial imagery, Sentinel-1/2 time series, SPOT imagery, topographic data, and historical aerial images. Extensive benchmarks evaluate multimodal fusion and deep learning models (CNNs, transformers) for land cover or crop mapping and also explore multi-task learning. Results underscore the complexity of multimodal fusion and fine-grained classification, with best land cover performance (78.2% accuracy, 65.8% mIoU) achieved using nearly all modalities. FLAIR-HUB supports supervised and multimodal pretraining, with data and code available at this https URL.
>
---
#### [replaced 049] Motion Illusions Generated Using Predictive Neural Networks Also Fool Humans
- **分类: cs.NE; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2112.13243](https://arxiv.org/pdf/2112.13243)**

> **作者:** Lana Sinapayen; Eiji Watanabe
>
> **摘要:** Why do we sometimes perceive static images as if they were moving? Visual motion illusions enjoy a sustained popularity, yet there is no definitive answer to the question of why they work. Here we present evidence in favor of the hypothesis that illusory motion is a side effect of the predictive abilities of the brain. We present a generative model, the Evolutionary Illusion GENerator (EIGen), that creates new visual motion illusions based on a video predictive neural network. We confirm that the constructed illusions are effective on human participants through a psychometric survey. Our results support the hypothesis that illusory motion might be the consequence of perceiving the brain's own predictions rather than perceiving raw visual input from the eyes. The philosophical motivation of this paper is to call attention to the untapped potential of "motivated failures", ways for artificial systems to fail as biological systems fail, as a worthy outlet for Artificial Intelligence and Artificial Life research.
>
---
#### [replaced 050] FreeAct: Freeing Activations for LLM Quantization
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于模型量化任务，旨在解决传统量化方法无法适应动态激活分布的问题。提出FreeAct框架，通过动态分配变换矩阵提升量化效果。**

- **链接: [https://arxiv.org/pdf/2603.01776](https://arxiv.org/pdf/2603.01776)**

> **作者:** Xiaohao Liu; Xiaobo Xia; Manyi Zhang; Ji-Fu Li; Xianzhi Yu; Fei Shen; Xiu Su; See-Kiong Ng; Tat-Seng Chua
>
> **备注:** 26 pages, 18 figures, 2 tables
>
> **摘要:** Quantization is pivotal for mitigating the significant memory and computational overhead of Large Language Models (LLMs). While emerging transformation-based methods have successfully enhanced quantization by projecting feature spaces onto smoother manifolds using orthogonal matrices, they typically enforce a rigid one-to-one transformation constraint. This static approach fails to account for the dynamic patterns inherent in input activations, particularly within diffusion LLMs (dLLMs) and Multimodal LLMs (MLLMs), where varying token types exhibit distinct distributions. To advance this, we propose FreeAct, a novel quantization framework that relaxes the static one-to-one constraint to accommodate dynamic activation disparities. Theoretically, we leverage the rank-deficient nature of activations to derive a solution space that extends beyond simple inverse matrices, enabling the decoupling of activation transformations from weights. Methodologically, FreeAct identifies token-specific dynamics (i.e., vision v.s. text, or masked tokens) and allocates distinct transformation matrices to the activation side, while maintaining a unified, static transformation for the weights. Extensive experiments across dLLMs and MLLMs demonstrate that FreeAct significantly outperforms baselines, up to 5.3% performance improvement, with in-depth analyses. Our code will be publicly released.
>
---
#### [replaced 051] Collaborative Learning of Local 3D Occupancy Prediction and Versatile Global Occupancy Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，旨在提升自动驾驶中的环境感知。通过融合全局先验信息，增强局部预测的鲁棒性，并构建大规模全局占用地图。**

- **链接: [https://arxiv.org/pdf/2504.13596](https://arxiv.org/pdf/2504.13596)**

> **作者:** Shanshuai Yuan; Julong Wei; Muer Tie; Xiangyun Ren; Zhongxue Gan; Wenchao Ding
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Vision-based 3D semantic occupancy prediction is vital for autonomous driving, enabling unified modeling of static infrastructure and dynamic agents. Global occupancy maps serve as long-term memory priors, providing valuable historical context that enhances local perception. This is particularly important in challenging scenarios such as occlusion or poor illumination, where current and nearby observations may be unreliable or incomplete. Priors aggregated from previous traversals under better conditions help fill gaps and enhance the robustness of local 3D occupancy prediction. In this paper, we propose Long-term Memory Prior Occupancy (LMPOcc), a plug-and-play framework that incorporates global occupancy priors to boost local prediction and simultaneously updates global maps with new observations. To realize the information gain from global priors, we design an efficient and lightweight Current-Prior Fusion module that adaptively integrates prior and current features. Meanwhile, we introduce a model-agnostic prior format to enable continual updating of global occupancy and ensure compatibility across diverse prediction baselines. LMPOcc achieves state-of-the-art local occupancy prediction performance validated on the Occ3D-nuScenes benchmark, especially on static semantic categories. Furthermore, we verify LMPOcc's capability to build large-scale global occupancy maps through multi-vehicle crowdsourcing, and utilize occupancy-derived dense depth to support the construction of 3D open-vocabulary maps. Our method opens up a new paradigm for continuous global information updating and storage, paving the way towards more comprehensive and scalable scene understanding in large outdoor environments.
>
---
#### [replaced 052] RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22013](https://arxiv.org/pdf/2602.22013)**

> **作者:** I-Hsiang Chen; Yu-Wei Liu; Tse-Yu Wu; Yu-Chien Chiang; Jen-Chien Yang; Wei-Ting Chen
>
> **备注:** Accepted by CVPR2026; Project Page: this https URL
>
> **摘要:** Vision-based Retrieval-Augmented Generation (VisRAG) leverages vision-language models (VLMs) to jointly retrieve relevant visual documents and generate grounded answers based on multimodal evidence. However, existing VisRAG models degrade in performance when visual inputs suffer from distortions such as blur, noise, low light, or shadow, where semantic and degradation factors become entangled within pretrained visual encoders, leading to errors in both retrieval and generation stages. To address this limitation, we introduce RobustVisRAG, a causality-guided dual-path framework that improves VisRAG robustness while preserving efficiency and zero-shot generalization. RobustVisRAG uses a non-causal path to capture degradation signals through unidirectional attention and a causal path to learn purified semantics guided by these signals. Together with the proposed Non-Causal Distortion Modeling and Causal Semantic Alignment objectives, the framework enforces a clear separation between semantics and degradations, enabling stable retrieval and generation under challenging visual conditions. To evaluate robustness under realistic conditions, we introduce the Distortion-VisRAG dataset, a large-scale benchmark containing both synthetic and real-world degraded documents across seven domains, with 12 synthetic and 5 real distortion types that comprehensively reflect practical visual degradations. Experimental results show that RobustVisRAG improves retrieval, generation, and end-to-end performance by 7.35%, 6.35%, and 12.40%, respectively, on real-world degradations, while maintaining comparable accuracy on clean inputs.
>
---
#### [replaced 053] Distant Object Localisation from Noisy Image Segmentation Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机火灾监测任务，解决远距离目标定位问题。通过多视角三角测量和粒子滤波方法，在计算资源有限情况下实现可靠定位与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2509.20906](https://arxiv.org/pdf/2509.20906)**

> **作者:** Julius Pesonen; Arno Solin; Eija Honkavaara
>
> **摘要:** 3D object localisation based on a sequence of camera measurements is essential for safety-critical surveillance tasks, such as drone-based wildfire monitoring. Localisation of objects detected with a camera can typically be solved with specialised sensor configurations or 3D scene reconstruction. However, in the context of distant objects or tasks limited by the amount of available computational resources, neither solution is feasible. In this paper, we show that the task can be solved with either multi-view triangulation or particle filters, with the latter also providing shape and uncertainty estimates. We studied the solutions using 3D simulation and drone-based image segmentation sequences with global navigation satellite system (GNSS) based camera pose estimates. The results suggest that combining the proposed methods with pre-existing image segmentation models and drone-carried computational resources yields a reliable system for drone-based wildfire monitoring. The proposed solutions are independent of the detection method, also enabling quick adaptation to similar tasks.
>
---
#### [replaced 054] Gated Differential Linear Attention: A Linear-Time Decoder for High-Fidelity Medical Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02727](https://arxiv.org/pdf/2603.02727)**

> **作者:** Hongbo Zheng; Afshin Bozorgpour; Dorit Merhof; Minjia Zhang
>
> **摘要:** Medical image segmentation requires models that preserve fine anatomical boundaries while remaining efficient for clinical deployment. While transformers capture long-range dependencies, they suffer from quadratic attention cost and large data requirements, whereas CNNs are compute-friendly yet struggle with global reasoning. Linear attention offers $\mathcal{O}(N)$ scaling, but often exhibits training instability and attention dilution, yielding diffuse maps. We introduce PVT-GDLA, a decoder-centric Transformer that restores sharp, long-range dependencies at linear time. Its core, Gated Differential Linear Attention (GDLA), computes two kernelized attention paths on complementary query/key subspaces and subtracts them with a learnable, channel-wise scale to cancel common-mode noise and amplify relevant context. A lightweight, head-specific gate injects nonlinearity and input-adaptive sparsity, mitigating attention sink, and a parallel local token-mixing branch with depthwise convolution strengthens neighboring-token interactions, improving boundary fidelity, all while retaining $\mathcal{O}(N)$ complexity and low parameter overhead. Coupled with a pretrained Pyramid Vision Transformer (PVT) encoder, PVT-GDLA achieves state-of-the-art accuracy across CT, MRI, ultrasound, and dermoscopy benchmarks under equal training budgets, with comparable parameters but lower FLOPs than CNN-, Transformer-, hybrid-, and linear-attention baselines. PVT-GDLA provides a practical path to fast, scalable, high-fidelity medical segmentation in clinical environments and other resource-constrained settings.
>
---
#### [replaced 055] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01007](https://arxiv.org/pdf/2603.01007)**

> **作者:** Xubo Zhu; Haoyang Zhang; Fei He; Rui Wu; Yanhu Shan; Wen Yang; Huai Yu
>
> **备注:** 10 pages, 6 figures. Accepted at CVPR 2026
>
> **摘要:** 3D semantic occupancy prediction is crucial for autonomous driving perception, offering comprehensive geometric scene understanding and semantic recognition. However, existing methods struggle with geometric misalignment in view transformation due to the lack of pixel-level accurate depth estimation, and severe spatial class imbalance where semantic categories exhibit strong spatial anisotropy. To address these challenges, we propose Dr. Occ, a depth- and region-guided occupancy prediction framework. Specifically, we introduce a depth-guided 2D-to-3D View Transformer (D$^2$-VFormer) that effectively leverages high-quality dense depth cues from MoGe-2 to construct reliable geometric priors, thereby enabling precise geometric alignment of voxel features. Moreover, inspired by the Mixture-of-Experts (MoE) framework, we propose a region-guided Expert Transformer (R/R$^2$-EFormer) that adaptively allocates region-specific experts to focus on different spatial regions, effectively addressing spatial semantic variations. Thus, the two components make complementary contributions: depth guidance ensures geometric alignment, while region experts enhance semantic learning. Experiments on the Occ3D--nuScenes benchmark demonstrate that Dr. Occ improves the strong baseline BEVDet4D by 7.43% mIoU and 3.09% IoU under the full vision-only setting.
>
---
#### [replaced 056] EgoCampus: Egocentric Pedestrian Eye Gaze Model and Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07668](https://arxiv.org/pdf/2512.07668)**

> **作者:** Ronan John; Aditya Kesari; Vincenzo DiMatteo; Kristin Dana
>
> **摘要:** We address the challenge of predicting human visual attention during real-world navigation by measuring and modeling egocentric pedestrian eye gaze in an outdoor campus setting. We introduce the EgoCampus dataset, which spans 25 unique outdoor paths over 6 km across a university campus with recordings from more than 80 distinct human pedestrians, resulting in a diverse set of gaze-annotated videos. The system used for collection, Meta's Project Aria glasses, integrates eye tracking, front-facing RGB cameras, inertial sensors, and GPS to provide rich data from the human perspective. Unlike many prior egocentric datasets that focus on indoor tasks or exclude eye gaze information, our work emphasizes visual attention while subjects walk in outdoor campus paths. Using this data, we develop EgoCampusNet, a novel method to predict eye gaze of navigating pedestrians as they move through outdoor environments. Our contributions provide both a new resource for studying real-world attention and a resource for future work in gaze prediction models for navigation. Dataset and code will be made publicly available at a later date at this https URL .
>
---
#### [replaced 057] RESAR-BEV: An Explainable Progressive Residual Autoregressive Approach for Camera-Radar Fusion in BEV Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.06515](https://arxiv.org/pdf/2505.06515)**

> **作者:** Zhiwen Zeng; Yunfei Yin; Zheng Yuan; Argho Dey; Xianjian Bao
>
> **备注:** This work was submitted to IEEE Transactions on Intelligent Transportation Systems (T-ITS) on 09-May-2025; revised 5 October 2025 and 26 January 2026; accepted 1 March 2026
>
> **摘要:** Bird's-Eye-View (BEV) semantic segmentation provides comprehensive environmental perception for autonomous driving but suffers multi-modal misalignment and sensor noise. We propose RESAR-BEV, a progressive refinement framework that advances beyond single-step end-to-end approaches: (1) progressive refinement through residual autoregressive learning that decomposes BEV segmentation into interpretable coarse-to-fine stages via our Drive-Transformer and Modifier-Transformer residual prediction cascaded architecture, (2) robust BEV representation combining ground-proximity voxels with adaptive height offsets and dual-path voxel feature encoding (max+attention pooling) for efficient feature extraction, and (3) decoupled supervision with offline Ground Truth decomposition and online joint optimization to prevent overfitting while ensuring structural coherence. Experiments on nuScenes demonstrate RESAR-BEV achieves state-of-the-art performance with 54.0% mIoU across 7 essential driving-scene categories while maintaining real-time capability at 14.6 FPS. The framework exhibits robustness in challenging scenarios of long-range perception and adverse weather conditions.
>
---
#### [replaced 058] MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.00204](https://arxiv.org/pdf/2601.00204)**

> **作者:** Xiaokun Sun; Zeyu Cai; Hao Tang; Ying Tai; Jian Yang; Zhenyu Zhang
>
> **备注:** Accepted by CVPR 2026; Project page: this https URL
>
> **摘要:** 3D morphing remains challenging due to the difficulty of generating semantically consistent and temporally smooth deformations, especially across categories. We present MorphAny3D, a training-free framework that leverages Structured Latent (SLAT) representations for high-quality 3D morphing. Our key insight is that intelligently blending source and target SLAT features within the attention mechanisms of 3D generators naturally produces plausible morphing sequences. To this end, we introduce Morphing Cross-Attention (MCA), which fuses source and target information for structural coherence, and Temporal-Fused Self-Attention (TFSA), which enhances temporal consistency by incorporating features from preceding frames. An orientation correction strategy further mitigates the pose ambiguity within the morphing steps. Extensive experiments show that our method generates state-of-the-art morphing sequences, even for challenging cross-category cases. MorphAny3D further supports advanced applications such as decoupled morphing and 3D style transfer, and can be generalized to other SLAT-based generative models. Project page: this https URL.
>
---
#### [replaced 059] Continuous Space-Time Video Super-Resolution with 3D Fourier Fields
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.26325](https://arxiv.org/pdf/2509.26325)**

> **作者:** Alexander Becker; Julius Erbach; Dominik Narnhofer; Konrad Schindler
>
> **摘要:** We introduce a novel formulation for continuous space-time video super-resolution. Instead of decoupling the representation of a video sequence into separate spatial and temporal components and relying on brittle, explicit frame warping for motion compensation, we encode video as a continuous, spatio-temporally coherent 3D Video Fourier Field (VFF). That representation offers three key advantages: (1) it enables cheap, flexible sampling at arbitrary locations in space and time; (2) it is able to simultaneously capture fine spatial detail and smooth temporal dynamics; and (3) it offers the possibility to include an analytical, Gaussian point spread function in the sampling to ensure aliasing-free reconstruction at arbitrary scale. The coefficients of the proposed, Fourier-like sinusoidal basis are predicted with a neural encoder with a large spatio-temporal receptive field, conditioned on the low-resolution input video. Through extensive experiments, we show that our joint modeling substantially improves both spatial and temporal super-resolution and sets a new state of the art for multiple benchmarks: across a wide range of upscaling factors, it delivers sharper and temporally more consistent reconstructions than existing baselines, while being computationally more efficient. Project page: this https URL.
>
---
#### [replaced 060] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理任务，旨在评估和提升模型的视觉接地推理能力。针对现有缺乏基准的问题，提出TreeBench基准和TreeVGR方法，提升模型的可解释性和准确性。**

- **链接: [https://arxiv.org/pdf/2507.07999](https://arxiv.org/pdf/2507.07999)**

> **作者:** Haochen Wang; Xiangtai Li; Zilong Huang; Anran Wang; Jiacong Wang; Tao Zhang; Jiani Zheng; Sule Bai; Zijian Kang; Jiashi Feng; Zhuochen Wang; Zhaoxiang Zhang
>
> **备注:** ICLR 2026 Camera Ready Version
>
> **摘要:** Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically referencing visual regions, just like human "thinking with images". However, no benchmark exists to evaluate these capabilities holistically. To bridge this gap, we propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark built on three principles: (1) focused visual perception of subtle targets in complex scenes, (2) traceable evidence via bounding box evaluation, and (3) second-order reasoning to test object interactions and spatial hierarchies beyond simple object localization. Prioritizing images with dense objects, we initially sample 1K high-quality images from SA-1B, and incorporate eight LMM experts to manually annotate questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, even the most advanced models struggle with this benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only 54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual Grounded Reasoning), a training paradigm to supervise localization and reasoning jointly with reinforcement learning, enabling accurate localizations and explainable reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench (+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is key to advancing vision-grounded reasoning. The code is available at this https URL.
>
---
#### [replaced 061] MotionStream: Real-Time Video Generation with Interactive Motion Controls
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.01266](https://arxiv.org/pdf/2511.01266)**

> **作者:** Joonghyuk Shin; Zhengqi Li; Richard Zhang; Jun-Yan Zhu; Jaesik Park; Eli Shechtman; Xun Huang
>
> **备注:** ICLR 2026, Project webpage: this https URL
>
> **摘要:** Current motion-conditioned video generation methods suffer from prohibitive latency (minutes per video) and non-causal processing that prevents real-time interaction. We present MotionStream, enabling sub-second latency with up to 29 FPS streaming generation on a single GPU. Our approach begins by augmenting a text-to-video model with motion control, which generates high-quality videos that adhere to the global text prompt and local motion guidance, but does not perform inference on the fly. As such, we distill this bidirectional teacher into a causal student through Self Forcing with Distribution Matching Distillation, enabling real-time streaming inference. Several key challenges arise when generating videos of long, potentially infinite time-horizons -- (1) bridging the domain gap from training on finite length and extrapolating to infinite horizons, (2) sustaining high quality by preventing error accumulation, and (3) maintaining fast inference, without incurring growth in computational cost due to increasing context windows. A key to our approach is introducing carefully designed sliding-window causal attention, combined with attention sinks. By incorporating self-rollout with attention sinks and KV cache rolling during training, we properly simulate inference-time extrapolations with a fixed context window, enabling constant-speed generation of arbitrarily long videos. Our models achieve state-of-the-art results in motion following and video quality while being two orders of magnitude faster, uniquely enabling infinite-length streaming. With MotionStream, users can paint trajectories, control cameras, or transfer motion, and see results unfold in real-time, delivering a truly interactive experience.
>
---
#### [replaced 062] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22091](https://arxiv.org/pdf/2602.22091)**

> **作者:** Matthew Strong; Wei-Jer Chang; Quentin Herau; Jiezhi Yang; Yihan Hu; Chensheng Peng; Wei Zhan
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Ego-centric driving videos available online provide an abundant source of visual data for autonomous driving, yet their lack of annotations makes it difficult to learn representations that capture both semantic structure and 3D geometry. Recent advances in large feedforward spatial models demonstrate that point maps and ego-motion can be inferred in a single forward pass, suggesting a promising direction for scalable driving perception. We therefore propose a label-free, teacher-guided framework for learning autonomous driving representations directly from unposed videos. Unlike prior self-supervised approaches that focus primarily on frame-to-frame consistency, we posit that safe and reactive driving depends critically on temporal context. To this end, we leverage a feedforward architecture equipped with a lightweight autoregressive module, trained using multi-modal supervisory signals that guide the model to jointly predict current and future point maps, camera poses, semantic segmentation, and motion masks. Multi-modal teachers provide sequence-level pseudo-supervision, enabling LFG to learn a unified pseudo-4D representation from raw YouTube videos without poses, labels, or LiDAR. The resulting encoder not only transfers effectively to downstream autonomous driving planning on the NAVSIM benchmark, surpassing multi-camera and LiDAR baselines with only a single monocular camera, but also yields strong performance when evaluated on a range of semantic, geometric, and qualitative motion prediction tasks. These geometry and motion-aware features position LFG as a compelling video-centric foundation model for autonomous driving.
>
---
#### [replaced 063] Learning to Select Like Humans: Explainable Active Learning for Medical Imaging
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.13308](https://arxiv.org/pdf/2602.13308)**

> **作者:** Ifrat Ikhtear Uddin; Longwei Wang; Xiao Qin; Yang Zhou; KC Santosh
>
> **备注:** Accepted for publication IEEE Conference on Artificial Intelligence 2026, Granada, Spain
>
> **摘要:** Medical image analysis requires substantial labeled data for model training, yet expert annotation is expensive and time-consuming. Active learning (AL) addresses this challenge by strategically selecting the most informative samples for the annotation purpose, but traditional methods solely rely on predictive uncertainty while ignoring whether models learn from clinically meaningful features a critical requirement for clinical deployment. We propose an explainability-guided active learning framework that integrates spatial attention alignment into a sample acquisition process. Our approach advocates for a dual-criterion selection strategy combining: (i) classification uncertainty to identify informative examples, and (ii) attention misalignment with radiologist-defined regions-of-interest (ROIs) to target samples where the model focuses on incorrect features. By measuring misalignment between Grad-CAM attention maps and expert annotations using Dice similarity, our acquisition function judiciously identifies samples that enhance both predictive performance and spatial interpretability. We evaluate the framework using three expert-annotated medical imaging datasets, namely, BraTS (MRI brain tumors), VinDr-CXR (chest X-rays), and SIIM-COVID-19 (chest X-rays). Using only 570 strategically selected samples, our explainability-guided approach consistently outperforms random sampling across all the datasets, achieving 77.22% accuracy on BraTS, 52.37% on VinDr-CXR, and 52.66% on SIIM-COVID. Grad-CAM visualizations confirm that the models trained by our dual-criterion selection focus on diagnostically relevant regions, demonstrating that incorporating explanation guidance into sample acquisition yields superior data efficiency while maintaining clinical interpretability.
>
---
#### [replaced 064] IoUCert: Robustness Verification for Anchor-based Object Detectors
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03043](https://arxiv.org/pdf/2603.03043)**

> **作者:** Benedikt Brückner; Alejandro J. Mercado; Yanghao Zhang; Panagiotis Kouvaros; Alessio Lomuscio
>
> **摘要:** While formal robustness verification has seen significant success in image classification, scaling these guarantees to object detection remains notoriously difficult due to complex non-linear coordinate transformations and Intersection-over-Union (IoU) metrics. We introduce IoUCert, a novel formal verification framework designed specifically to overcome these bottlenecks in foundational anchor-based object detection architectures. Focusing on the object localisation component in single-object settings, we propose a coordinate transformation that enables our algorithm to circumvent precision-degrading relaxations of non-linear box prediction functions. This allows us to optimise bounds directly with respect to the anchor box offsets which enables a novel Interval Bound Propagation method that derives optimal IoU bounds. We demonstrate that our method enables, for the first time, the robustness verification of realistic, anchor-based models including SSD, YOLOv2, and YOLOv3 variants against various input perturbations.
>
---
#### [replaced 065] DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14266](https://arxiv.org/pdf/2512.14266)**

> **作者:** Shreedhar Govil; Didier Stricker; Jason Rambach
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Predicting driver attention is a critical problem for developing explainable autonomous driving systems and understanding driver behavior in mixed human-autonomous vehicle traffic scenarios. Although significant progress has been made through large-scale driver attention datasets and deep learning architectures, existing works are constrained by narrow frontal field-of-view and limited driving diversity. Consequently, they fail to capture the full spatial context of driving environments, especially during lane changes, turns, and interactions involving peripheral objects such as pedestrians or cyclists. In this paper, we introduce DriverGaze360, a large-scale 360$^\circ$ field of view driver attention dataset, containing $\sim$1 million gaze-labeled frames collected from 19 human drivers, enabling comprehensive omnidirectional modeling of driver gaze behavior. Moreover, our panoramic attention prediction approach, DriverGaze360-Net, jointly learns attention maps and attended objects by employing an auxiliary semantic segmentation head. This improves spatial awareness and attention prediction across wide panoramic inputs. Extensive experiments demonstrate that DriverGaze360-Net achieves state-of-the-art attention prediction performance on multiple metrics on panoramic driving images. Dataset and method available at this https URL.
>
---
#### [replaced 066] CityGuard: Graph-Aware Private Descriptors for Bias-Resilient Identity Search Across Urban Cameras
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.18047](https://arxiv.org/pdf/2602.18047)**

> **作者:** Rong Fu; Yibo Meng; Jia Yee Tan; Jiaxuan Lu; Rui Lu; Jiekai Wu; Zhaolu Kang; Simon Fong
>
> **备注:** 36 pages, 12 figures
>
> **摘要:** City-scale person re-identification across distributed cameras must handle severe appearance changes from viewpoint, occlusion, and domain shift while complying with data protection rules that prevent sharing raw imagery. We introduce CityGuard, a topology-aware transformer for privacy-preserving identity retrieval in decentralized surveillance. The framework integrates three components. A dispersion-adaptive metric learner adjusts instance-level margins according to feature spread, increasing intra-class compactness. Spatially conditioned attention injects coarse geometry, such as GPS or deployment floor plans, into graph-based self-attention to enable projectively consistent cross-view alignment using only coarse geometric priors without requiring survey-grade calibration. Differentially private embedding maps are coupled with compact approximate indexes to support secure and cost-efficient deployment. Together these designs produce descriptors robust to viewpoint variation, occlusion, and domain shifts, and they enable a tunable balance between privacy and utility under rigorous differential-privacy accounting. Experiments on Market-1501 and additional public benchmarks, complemented by database-scale retrieval studies, show consistent gains in retrieval precision and query throughput over strong baselines, confirming the practicality of the framework for privacy-critical urban identity matching.
>
---
#### [replaced 067] SpineBench: A Clinically Salient, Level-Aware Benchmark Powered by the SpineMed-450k Corpus
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.03160](https://arxiv.org/pdf/2510.03160)**

> **作者:** Ming Zhao; Wenhui Dong; Yang Zhang; Xiang Zheng; Zhonghao Zhang; Zian Zhou; Yunzhi Guan; Liukun Xu; Wei Peng; Zhaoyang Gong; Zhicheng Zhang; Dachuan Li; Xiaosheng Ma; Yuli Ma; Jianing Ni; Changjiang Jiang; Lixia Tian; Qixin Chen; Kaishun Xia; Pingping Liu; Tongshun Zhang; Zhiqiang Liu; Zhongyan Bi; Chenyang Si; Tiansheng Sun; Caifeng Shan
>
> **摘要:** Spine disorders affect 619 million people globally and are a leading cause of disability, yet AI-assisted diagnosis remains limited by the lack of level-aware, multimodal datasets. Clinical decision-making for spine disorders requires sophisticated reasoning across X-ray, CT, and MRI at specific vertebral levels. However, progress has been constrained by the absence of traceable, clinically-grounded instruction data and standardized, spine-specific benchmarks. To address this, we introduce SpineMed, an ecosystem co-designed with practicing spine surgeons. It features SpineMed-450k, the first large-scale dataset explicitly designed for vertebral-level reasoning across imaging modalities with over 450,000 instruction instances, and SpineBench, a clinically-grounded evaluation framework. SpineMed-450k is curated from diverse sources, including textbooks, guidelines, open datasets, and ~1,000 de-identified hospital cases, using a clinician-in-the-loop pipeline with a two-stage LLM generation method (draft and revision) to ensure high-quality, traceable data for question-answering, multi-turn consultations, and report generation. SpineBench evaluates models on clinically salient axes, including level identification, pathology assessment, and surgical planning. Our comprehensive evaluation of several recently advanced large vision-language models (LVLMs) on SpineBench reveals systematic weaknesses in fine-grained, level-specific reasoning. In contrast, our model fine-tuned on SpineMed-450k demonstrates consistent and significant improvements across all tasks. Clinician assessments confirm the diagnostic clarity and practical utility of our model's outputs.
>
---
#### [replaced 068] EDITOR: Effective and Interpretable Prompt Inversion for Text-to-Image Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03067](https://arxiv.org/pdf/2506.03067)**

> **作者:** Mingzhe Li; Kejing Xia; Gehao Zhang; Zhenting Wang; Guanhong Tao; Siqi Pan; Juan Zhai; Shiqing Ma
>
> **摘要:** Text-to-image generation models~(e.g., Stable Diffusion) have achieved significant advancements, enabling the creation of high-quality and realistic images based on textual descriptions. Prompt inversion, the task of identifying the textual prompt used to generate a specific artifact, holds significant potential for applications including data attribution, model provenance, and watermarking validation. Recent studies introduced a delayed projection scheme to optimize for prompts representative of the vocabulary space, though challenges in semantic fluency and efficiency remain. Advanced image captioning models or visual large language models can generate highly interpretable prompts, but they often lack in image similarity. In this paper, we propose a prompt inversion technique called \sys for text-to-image diffusion models, which includes initializing embeddings using a pre-trained image captioning model, refining them through reverse-engineering in the latent space, and converting them to texts using an embedding-to-text model. Our experiments on the widely-used datasets, such as MS COCO, LAION, Flickr and DiffusionDB, show that our method outperforms existing methods in terms of image similarity, textual alignment, prompt interpretability and generalizability. We further illustrate the application of our generated prompts in tasks such as cross-concept image synthesis, concept manipulation, evolutionary multi-concept generation and unsupervised segmentation.
>
---
#### [replaced 069] Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02573](https://arxiv.org/pdf/2603.02573)**

> **作者:** Jiahao Lu; Jiayi Xu; Wenbo Hu; Ruijie Zhu; Chengfeng Zhao; Sai-Kit Yeung; Ying Shan; Yuan Liu
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** Estimating the 3D trajectory of every pixel from a monocular video is crucial and promising for a comprehensive understanding of the 3D dynamics of videos. Recent monocular 3D tracking works demonstrate impressive performance, but are limited to either tracking sparse points on the first frame or a slow optimization-based framework for dense tracking. In this paper, we propose a feedforward model, called Track4World, enabling an efficient holistic 3D tracking of every pixel in the world-centric coordinate system. Built on the global 3D scene representation encoded by a VGGT-style ViT, Track4World applies a novel 3D correlation scheme to simultaneously estimate the pixel-wise 2D and 3D dense flow between arbitrary frame pairs. The estimated scene flow, along with the reconstructed 3D geometry, enables subsequent efficient 3D tracking of every pixel of this video. Extensive experiments on multiple benchmarks demonstrate that our approach consistently outperforms existing methods in 2D/3D flow estimation and 3D tracking, highlighting its robustness and scalability for real-world 4D reconstruction tasks.
>
---
#### [replaced 070] RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.21692](https://arxiv.org/pdf/2503.21692)**

> **作者:** Daniel Bermuth; Alexander Poeppel; Wolfgang Reif
>
> **摘要:** The integration of multi-view imaging and pose estimation represents a significant advance in computer vision applications, offering new possibilities for understanding human movement and interactions. This work presents a new algorithm that improves multi-view multi-person pose estimation, focusing on fast triangulation speeds and good generalization capabilities. The approach extends to whole-body pose estimation, capturing details from facial expressions to finger movements across multiple individuals and viewpoints. Adaptability to different settings is demonstrated through strong performance across unseen datasets and configurations. To support further progress in this field, all of this work is publicly accessible.
>
---
#### [replaced 071] A Unified Framework for Joint Detection of Lacunes and Enlarged Perivascular Spaces
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.04243](https://arxiv.org/pdf/2603.04243)**

> **作者:** Lucas He; Krinos Li; Hanyuan Zhang; Runlong He; Silvia Ingala; Luigi Lorenzini; Marleen de Bruijne; Frederik Barkhof; Rhodri Davies; Carole Sudre
>
> **摘要:** Cerebral small vessel disease (CSVD) markers, specifically enlarged perivascular spaces (EPVS) and lacunae, present a unique challenge in medical image analysis due to their radiological mimicry. Standard segmentation networks struggle with feature interference and extreme class imbalance when handling these divergent targets simultaneously. To address these issues, we propose a morphology-decoupled framework where Zero-Initialized Gated Cross-Task Attention exploits dense EPVS context to guide sparse lacune detection. Furthermore, biological and topological consistency are enforced via a mixed-supervision strategy integrating Mutual Exclusion and Centerline Dice losses. Finally, we introduce an Anatomically-Informed Inference Calibration mechanism to dynamically suppress false positives based on tissue semantics. Extensive 5-folds cross-validation on the VALDO 2021 dataset (N=40) demonstrates state-of-the-art performance, notably surpassing task winners in lacunae detection precision (71.1%, p=0.01) and F1-score (62.6%, p=0.03). Furthermore, evaluation on the external EPAD cohort (N=1762) confirms the model's robustness for large-scale population studies. Code will be released upon acceptance.
>
---
#### [replaced 072] InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文属于多模态人类动画生成任务，旨在解决多主体互动场景下条件控制不精确的问题。提出新框架实现区域化条件绑定与布局对齐，提升多人对话视频生成质量。**

- **链接: [https://arxiv.org/pdf/2506.09984](https://arxiv.org/pdf/2506.09984)**

> **作者:** Zhenzhi Wang; Jiaqi Yang; Jianwen Jiang; Chao Liang; Gaojie Lin; Zerong Zheng; Ceyuan Yang; Yuan Zhang; Mingyuan Gao; Dahua Lin
>
> **备注:** ICLR 2026 Camera Ready Version. TL;DR: The first multi-person dialogue video generation method from pairs of reference image and audio via explicit layout-aligned condition injection. Project page this https URL
>
> **摘要:** End-to-end human animation with rich multi-modal conditions, e.g., text, image and audio has achieved remarkable advancements in recent years. However, most existing methods could only animate a single subject and inject conditions in a global manner, ignoring scenarios where multiple concepts could appear in the same video with rich human-human interactions and human-object interactions. Such a global assumption prevents precise and per-identity control of multiple concepts including humans and objects, therefore hinders applications. In this work, we discard the single-entity assumption and introduce a novel framework that enforces strong, region-specific binding of conditions from modalities to each identity's spatiotemporal footprint. Given reference images of multiple concepts, our method could automatically infer layout information by leveraging a mask predictor to match appearance cues between the denoised video and each reference appearance. Furthermore, we inject local audio condition into its corresponding region to ensure layout-aligned modality matching in an iterative manner. This design enables the high-quality generation of human dialogue videos between two to three people or video customization from multiple reference images. Empirical results and ablation studies validate the effectiveness of our explicit layout control for multi-modal conditions compared to implicit counterparts and other existing methods. Video demos are available at this https URL
>
---
#### [replaced 073] Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Contextually Important Moments
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究基础模型识别视频中关键事件的能力，针对足球比赛视频，构建数据集评估模型表现，发现其性能接近随机，需改进多模态融合方法。**

- **链接: [https://arxiv.org/pdf/2601.16333](https://arxiv.org/pdf/2601.16333)**

> **作者:** Aditya K Surikuchi; Raquel Fernández; Sandro Pezzelle
>
> **摘要:** Foundation models are used for many real-world applications involving language generation from temporally-ordered multimodal events. In this work, we study the ability of models to identify the most important sub-events in a video, which is a fundamental prerequisite for narrating or summarizing multimodal events. Specifically, we focus on football games and evaluate models on their ability to distinguish between important and non-important sub-events in a game. To this end, we construct a new dataset by leveraging human preferences for importance implicit in football game highlight reels, without any additional annotation costs. Using our dataset, we compare several state-of-the-art multimodal models and show that they are not far from chance level performance. Analyses of models beyond standard evaluation metrics reveal their tendency to rely on a single dominant modality and their ineffectiveness in synthesizing necessary information from multiple sources. Our findings underline the importance of modular architectures that can handle sample-level heterogeneity in multimodal data and the need for complementary training procedures that can maximize cross-modal synergy.
>
---
#### [replaced 074] EasyAnimate: High-Performance Video Generation Framework with Hybrid Windows Attention and Reward Backpropagation
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于视频生成任务，旨在解决生成速度慢和质量不高的问题。提出EasyAnimate框架，采用混合窗口注意力和奖励反向传播优化模型性能。**

- **链接: [https://arxiv.org/pdf/2405.18991](https://arxiv.org/pdf/2405.18991)**

> **作者:** Jiaqi Xu; Kunzhe Huang; Xinyi Zou; Yunkuo Chen; Bo Liu; MengLi Cheng; Jun Huang; Xing Shi
>
> **备注:** 10 pages, 8 figures, ACM MM 2025
>
> **摘要:** This paper introduces EasyAnimate, an efficient and high quality video generation framework that leverages diffusion transformers to achieve high-quality video production, encompassing data processing, model training, and end-to-end inference. Despite substantial advancements achieved by video diffusion models, existing video generation models still struggles with slow generation speeds and less-than-ideal video quality. To improve training and inference efficiency without compromising performance, we propose Hybrid Window Attention. We design the multidirectional sliding window attention in Hybrid Window Attention, which provides stronger receptive capabilities in 3D dimensions compared to naive one, while reducing the model's computational complexity as the video sequence length increases. To enhance video generation quality, we optimize EasyAnimate using reward backpropagation to better align with human preferences. As a post-training method, it greatly enhances the model's performance while ensuring efficiency. In addition to the aforementioned improvements, EasyAnimate integrates a series of further refinements that significantly improve both computational efficiency and model performance. We introduce a new training strategy called Training with Token Length to resolve uneven GPU utilization in training videos of varying resolutions and lengths, thereby enhancing efficiency. Additionally, we use a multimodal large language model as the text encoder to improve text comprehension of the model. Experiments demonstrate significant enhancements resulting from the above improvements. The EasyAnimate achieves state-of-the-art performance on both the VBench leaderboard and human evaluation. Code and pre-trained models are available at this https URL.
>
---
#### [replaced 075] Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13454](https://arxiv.org/pdf/2510.13454)**

> **作者:** Hyojun Go; Dominik Narnhofer; Goutam Bhat; Prune Truong; Federico Tombari; Konrad Schindler
>
> **备注:** ICLR 2026 (Oral), Project page: this https URL
>
> **摘要:** The rapid progress of large, pretrained models for both visual content generation and 3D reconstruction opens up new possibilities for text-to-3D generation. Intuitively, one could obtain a formidable 3D scene generator if one were able to combine the power of a modern latent text-to-video model as "generator" with the geometric abilities of a recent (feedforward) 3D reconstruction system as "decoder". We introduce VIST3A, a general framework that does just that, addressing two main challenges. First, the two components must be joined in a way that preserves the rich knowledge encoded in their weights. We revisit model stitching, i.e., we identify the layer in the 3D decoder that best matches the latent representation produced by the text-to-video generator and stitch the two parts together. That operation requires only a small dataset and no labels. Second, the text-to-video generator must be aligned with the stitched 3D decoder, to ensure that the generated latents are decodable into consistent, perceptually convincing 3D scene geometry. To that end, we adapt direct reward finetuning, a popular technique for human preference alignment. We evaluate the proposed VIST3A approach with different video generators and 3D reconstruction models. All tested pairings markedly improve over prior text-to-3D models that output Gaussian splats. Moreover, by choosing a suitable 3D base model, VIST3A also enables high-quality text-to-pointmap generation.
>
---
#### [replaced 076] MedFuncta: A Unified Framework for Learning Efficient Medical Neural Fields
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2502.14401](https://arxiv.org/pdf/2502.14401)**

> **作者:** Paul Friedrich; Florentin Bieder; Julian McGinnis; Julia Wolleb; Daniel Rueckert; Philippe C. Cattin
>
> **备注:** Accepted at MIDL 2026 (Oral) Project page: this https URL Code: this https URL Dataset: this https URL
>
> **摘要:** Research in medical imaging primarily focuses on discrete data representations that poorly scale with grid resolution and fail to capture the often continuous nature of the underlying signal. Neural Fields (NFs) offer a powerful alternative by modeling data as continuous functions. While single-instance NFs have successfully been applied in medical contexts, extending them to large-scale medical datasets remains an open challenge. We therefore introduce MedFuncta, a unified framework for large-scale NF training on diverse medical signals. Building on Functa, our approach encodes data into a unified representation, namely a 1D latent vector, that modulates a shared, meta-learned NF, enabling generalization across a dataset. We revisit common design choices, introducing a non-constant frequency parameter $\omega$ in widely used SIREN activations, and establish a connection between this $\omega$-schedule and layer-wise learning rates, relating our findings to recent work in theoretical learning dynamics. We additionally introduce a scalable meta-learning strategy for shared network learning that employs sparse supervision during training, thereby reducing memory consumption and computational overhead while maintaining competitive performance. Finally, we evaluate MedFuncta across a diverse range of medical datasets and show how to solve relevant downstream tasks on our neural data representation. To promote further research in this direction, we release our code, model weights and the first large-scale dataset - MedNF - containing > 500 k latent vectors for multi-instance medical NFs.
>
---
#### [replaced 077] Pursuing Minimal Sufficiency in Spatial Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.16688](https://arxiv.org/pdf/2510.16688)**

> **作者:** Yejie Guo; Yunzhong Hou; Wufei Ma; Meng Tang; Ming-Hsuan Yang
>
> **摘要:** Spatial reasoning, the ability to ground language in 3D understanding, remains a persistent challenge for Vision-Language Models (VLMs). We identify two fundamental bottlenecks: inadequate 3D understanding capabilities stemming from 2D-centric pre-training, and reasoning failures induced by redundant 3D information. To address these, we first construct a Minimal Sufficient Set (MSS) of information before answering a given question: a compact selection of 3D perception results from \textit{expert models}. We introduce MSSR (Minimal Sufficient Spatial Reasoner), a dual-agent framework that implements this principle. A Perception Agent programmatically queries 3D scenes using a versatile perception toolbox to extract sufficient information, including a novel SOG (Situated Orientation Grounding) module that robustly extracts language-grounded directions. A Reasoning Agent then iteratively refines this information to pursue minimality, pruning redundant details and requesting missing ones in a closed loop until the MSS is curated. Extensive experiments demonstrate that our method, by explicitly pursuing both sufficiency and minimality, significantly improves accuracy and achieves state-of-the-art performance across two challenging benchmarks. Furthermore, our framework produces interpretable reasoning paths, offering a promising source of high-quality training data for future models. Source code is available at this https URL.
>
---
#### [replaced 078] DPAC: Distribution-Preserving Adversarial Control for Diffusion Sampling
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.01153](https://arxiv.org/pdf/2512.01153)**

> **作者:** Han-Jin Lee; Han-Ju Lee; Jin-Seong Kim; Seok-Hwan Choi
>
> **摘要:** Adversarially guided diffusion sampling often achieves the target class, but sample quality degrades as deviations between the adversarially controlled and nominal trajectories accumulate. We formalize this degradation as a path-space Kullback-Leibler divergence(path-KL) between controlled and nominal (uncontrolled) diffusion processes, thereby showing via Girsanov's theorem that it exactly equals the control energy. Building on this stochastic optimal control (SOC) view, we theoretically establish that minimizing this path-KL simultaneously tightens upper bounds on both the 2-Wasserstein distance and Fréchet Inception Distance (FID), revealing a principled connection between adversarial control energy and perceptual fidelity. From a variational perspective, we derive a first-order optimality condition for the control: among all directions that yield the same classification gain, the component tangent to iso-(log-)density surfaces (i.e., orthogonal to the score) minimizes path-KL, whereas the normal component directly increases distributional drift. This leads to DPAC (Distribution-Preserving Adversarial Control), a diffusion guidance rule that projects adversarial gradients onto the tangent space defined by the generative score geometry. We further show that in discrete solvers, the tangent projection cancels the O({\Delta}t) leading error term in the Wasserstein distance, achieving an O({\Delta}t^2) quality gap; moreover, it remains second-order robust to score or metric approximation. Empirical studies on ImageNet-100 validate the theoretical predictions, confirming that DPAC achieves lower FID and estimated path-KL at matched attack success rates.
>
---
#### [replaced 079] Parallel Diffusion Solver via Residual Dirichlet Policy Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.22796](https://arxiv.org/pdf/2512.22796)**

> **作者:** Ruoyu Wang; Ziyu Li; Beier Zhu; Liangyu Yuan; Hanwang Zhang; Xun Yang; Xiaojun Chang; Chi Zhang
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2507.14797
>
> **摘要:** Diffusion models (DMs) have achieved state-of-the-art generative performance but suffer from high sampling latency due to their sequential denoising nature. Existing solver-based acceleration methods often face significant image quality degradation under a low-latency budget, primarily due to accumulated truncation errors arising from the inability to capture high-curvature trajectory segments. In this paper, we propose the Ensemble Parallel Direction solver (dubbed as EPD-Solver), a novel ODE solver that mitigates these errors by incorporating multiple parallel gradient evaluations in each step. Motivated by the geometric insight that sampling trajectories are largely confined to a low-dimensional manifold, EPD-Solver leverages the Mean Value Theorem for vector-valued functions to approximate the integral solution more accurately. Importantly, since the additional gradient computations are independent, they can be fully parallelized, preserving low-latency sampling nature. We introduce a two-stage optimization framework. Initially, EPD-Solver optimizes a small set of learnable parameters via a distillation-based approach. We further propose a parameter-efficient Reinforcement Learning (RL) fine-tuning scheme that reformulates the solver as a stochastic Dirichlet policy. Unlike traditional methods that fine-tune the massive backbone, our RL approach operates strictly within the low-dimensional solver space, effectively mitigating reward hacking while enhancing performance in complex text-to-image (T2I) generation tasks. In addition, our method is flexible and can serve as a plugin (EPD-Plugin) to improve existing ODE samplers.
>
---
#### [replaced 080] Optimizing Multi-Modality Trackers via Significance-Regularized Tuning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.17488](https://arxiv.org/pdf/2508.17488)**

> **作者:** Zhiwen Chen; Jinjian Wu; Zhiyu Zhu; Yifan Zhang; Guangming Shi; Junhui Hou
>
> **摘要:** This paper tackles the critical challenge of optimizing multi-modality trackers by effectively adapting pre-trained models for RGB data. Existing fine-tuning paradigms oscillate between excessive flexibility and over-restriction, both leading to suboptimal plasticity-stability trade-offs. To mitigate this dilemma, we propose a novel significance-regularized fine-tuning framework, which delicately refines the learning process by incorporating intrinsic parameter significance. Through a comprehensive investigation of the transition from pre-trained to multi-modality contexts, we identify that parameters crucial to preserving foundational patterns and managing cross-domain shifts are the primary drivers of this issue. Specifically, we first probe the tangent space of pre-trained weights to measure and orient prior significance, dedicated to preserving generalization. Subsequently, we characterize transfer significance during the fine-tuning phase, emphasizing adaptability and stability. By incorporating these parameter significance terms as unified regularization, our method markedly enhances transferability across modalities. Extensive experiments showcase the superior performance of our method, surpassing current state-of-the-art techniques across various multi-modal tracking benchmarks. The source code and models are publicly available at this https URL.
>
---
#### [replaced 081] EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于轨迹预测任务，解决ego视角下噪声观测导致的预测不鲁棒问题。提出EgoTraj-Bench基准和BiFlow模型，提升预测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.00405](https://arxiv.org/pdf/2510.00405)**

> **作者:** Jiayi Liu; Jiaming Zhou; Ke Ye; Kun-Yu Lin; Allan Wang; Junwei Liang
>
> **摘要:** Reliable trajectory prediction from an ego-centric perspective is crucial for robotic navigation in human-centric environments. However, existing methods typically assume noiseless observation histories, failing to account for the perceptual artifacts inherent in first-person vision, such as occlusions, ID switches, and tracking drift. This discrepancy between training assumptions and deployment reality severely limits model robustness. To bridge this gap, we introduce EgoTraj-Bench, built upon TBD dataset, which is the first real-world benchmark that aligns noisy, first-person visual histories with clean, bird's-eye-view future trajectories, enabling robust learning under realistic perceptual constraints. Building on this benchmark, we propose BiFlow, a dual-stream flow matching model that concurrently denoises historical observations and forecasts future motion. To better model agent intent, BiFlow incorporates our EgoAnchor mechanism, which conditions the prediction decoder on distilled historical features via feature modulation. Extensive experiments show that BiFlow achieves state-of-the-art performance, reducing minADE and minFDE by 10-15% on average and demonstrating superior robustness. We anticipate that our benchmark and model will provide a critical foundation for robust real-world ego-centric trajectory prediction. The benchmark library is available at: this https URL.
>
---
#### [replaced 082] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.16714](https://arxiv.org/pdf/2510.16714)**

> **作者:** Xiongkun Linghu; Jiangyong Huang; Ziyu Zhu; Baoxiong Jia; Siyuan Huang
>
> **备注:** Accepted by ICLR 2026. Project page: this https URL
>
> **摘要:** Existing research on 3D Large Language Models (LLMs) still struggles to achieve grounded question-answering, primarily due to the under-exploration of the mechanism of human-like scene-object grounded reasoning. This paper bridges the gap by presenting a novel framework. We first introduce a grounded Chain-of-Thought reasoning method in 3D scenes (SCENECOT), decoupling a complex reasoning task into simpler and manageable problems, and building corresponding visual clues based on multimodal expert modules. To enable such a method, we develop SCENECOT-185K, the first large-scale grounded CoT reasoning dataset, consisting of 185K high-quality instances. Extensive experiments across various complex 3D scene reasoning benchmarks demonstrate that our new framework achieves strong performance with high grounding-QA coherence. To the best of our knowledge, this is the first successful application of CoT reasoning to 3D scene understanding, enabling step-by-step human-like reasoning and showing potential for extension to broader 3D scene understanding scenarios.
>
---
#### [replaced 083] When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21977](https://arxiv.org/pdf/2602.21977)**

> **作者:** Liangwei Lyu; Jiaqi Xu; Jianwei Ding; Qiyao Deng
>
> **摘要:** Low-Rank Adaptation (LoRA) has emerged as a leading technique for efficiently fine-tuning text-to-image diffusion models, and its widespread adoption on open-source platforms has fostered a vibrant culture of model sharing and customization. However, the same modular and plug-and-play flexibility that makes LoRA appealing also introduces a broader attack surface. To highlight this risk, we propose Masquerade-LoRA (MasqLoRA), the first systematic attack framework that leverages an independent LoRA module as the attack vehicle to stealthily inject malicious behavior into text-to-image diffusion models. MasqLoRA operates by freezing the base model parameters and updating only the low-rank adapter weights using a small number of "trigger word-target image" pairs. This enables the attacker to train a standalone backdoor LoRA module that embeds a hidden cross-modal mapping: when the module is loaded and a specific textual trigger is provided, the model produces a predefined visual output; otherwise, it behaves indistinguishably from the benign model, ensuring the stealthiness of the attack. Experimental results demonstrate that MasqLoRA can be trained with minimal resource overhead and achieves a high attack success rate of 99.8%. MasqLoRA reveals a severe and unique threat in the AI supply chain, underscoring the urgent need for dedicated defense mechanisms for the LoRA-centric sharing ecosystem.
>
---
#### [replaced 084] Noise2Ghost: Self-supervised deep convolutional reconstruction for ghost imaging
- **分类: cs.CV; cs.LG; physics.data-an**

- **链接: [https://arxiv.org/pdf/2504.10288](https://arxiv.org/pdf/2504.10288)**

> **作者:** Mathieu Manni; Dmitry Karpov; K. Joost Batenburg; Sharon Shwartz; Nicola Viganò
>
> **摘要:** We present a new self-supervised deep-learning-based Ghost Imaging (GI) reconstruction method, which provides unparalleled reconstruction quality for noisy acquisitions among unsupervised methods. We present the supporting mathematical framework and results from theoretical and real data use cases. Self-supervision removes the need for clean reference data while offering strong noise reduction. This provides the necessary tools for addressing signal-to-noise ratio concerns for GI acquisitions in emerging and cutting-edge low-light GI scenarios. Notable examples include micro- and nano-scale x-ray emission imaging, e.g., x-ray fluorescence imaging of dose-sensitive samples. Their applications include in-vivo and in-operando case studies for biological samples and batteries.
>
---
#### [replaced 085] ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14654](https://arxiv.org/pdf/2512.14654)**

> **作者:** Lihong Wang; Liangqi Li; Weiwei Feng; Jiamin Wu; Changtao Miao; Tieru Wu; Rui Ma; Bo Zhang; Zhe Li
>
> **备注:** Accepted to CVPR 2026 (Main Track)
>
> **摘要:** CoT has significantly enhanced the reasoning ability of LLMs while it faces challenges when extended to multimodal domains, particularly in mathematical tasks. Existing MLLMs typically perform textual reasoning solely from a single static mathematical image, overlooking dynamic visual acquisition during reasoning. In contrast, humans repeatedly examine visual image and employ step-by-step reasoning to prove intermediate propositions. This strategy of decomposing the problem-solving process into key logical nodes adheres to Miller's Law in cognitive science. Inspired by this insight, we propose a ViRC framework for multimodal mathematical tasks, introducing a Reason Chunking mechanism that structures multimodal mathematical CoT into consecutive Critical Reasoning Units (CRUs) to simulate human expert problem-solving patterns. CRUs ensure intra-unit textual coherence for intermediate proposition verification while integrating visual information across units to generate subsequent propositions and support structured reasoning. To this end, we present CRUX dataset by using three visual tools and four reasoning patterns to provide explicitly annotated CRUs across multiple reasoning paths for each mathematical problem. Leveraging the CRUX dataset, we propose a progressive training strategy inspired by human cognitive learning, which includes Instructional SFT, Practice SFT, and Strategic RL, aimed at further strengthening the Reason Chunking ability of the model. The resulting ViRC-7B model achieves a 18.8% average improvement over baselines across multiple mathematical benchmarks. Code is available at this https URL.
>
---
#### [replaced 086] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出GAR模型，解决多模态大语言模型在区域级细粒度理解与跨区域关系建模上的不足，通过全局上下文感知和多提示交互，提升对任意区域的精准理解和复杂推理能力。**

- **链接: [https://arxiv.org/pdf/2510.18876](https://arxiv.org/pdf/2510.18876)**

> **作者:** Haochen Wang; Yuhao Wang; Tao Zhang; Yikang Zhou; Yanwei Li; Jiacong Wang; Jiani Zheng; Ye Tian; Jiahao Meng; Zilong Huang; Guangcan Mai; Anran Wang; Yunhai Tong; Zhuochen Wang; Xiangtai Li; Zhaoxiang Zhang
>
> **备注:** ICLR 2026 Camera Ready Version
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at holistic understanding, they struggle in capturing the dense world with complex scenes, requiring fine-grained analysis of intricate details and object inter-relationships. Region-level MLLMs have been a promising step. However, previous attempts are generally optimized to understand given regions in isolation, neglecting crucial global contexts. To address this, we introduce Grasp Any Region (GAR) for comprehen- sive region-level visual understanding. Empowered by an effective RoI-aligned feature replay technique, GAR supports (1) precise perception by leveraging necessary global contexts, and (2) modeling interactions between multiple prompts. Together, it then naturally achieves (3) advanced compositional reasoning to answer specific free-form questions about any region, shifting the paradigm from passive description to active dialogue. Moreover, we construct GAR-Bench, which not only provides a more accurate evaluation of single-region comprehension, but also, more importantly, measures interactions and complex reasoning across multiple regions. Extensive experiments have demonstrated that GAR-1B not only maintains the state-of-the-art captioning capabilities, e.g., outperforming DAM-3B +4.5 on DLC-Bench, but also excels at modeling relationships between multiple prompts with advanced comprehension capabilities, even surpassing InternVL3-78B on GAR-Bench-VQA. More importantly, our zero-shot GAR-8B even outperforms in-domain VideoRefer-7B on VideoRefer-BenchQ, indicating its strong capabilities can be easily transferred to videos.
>
---
#### [replaced 087] DMD-augmented Unpaired Neural Schrödinger Bridge for Ultra-Low Field MRI Enhancement
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.03769](https://arxiv.org/pdf/2603.03769)**

> **作者:** Youngmin Kim; Jaeyun Shin; Jeongchan Kim; Taehoon Lee; Jaemin Kim; Peter Hsu; Jelle Veraart; Jong Chul Ye
>
> **摘要:** Ultra Low Field (64 mT) brain MRI improves accessibility but suffers from reduced image quality compared to 3 T. As paired 64 mT - 3 T scans are scarce, we propose an unpaired 64 mT $\rightarrow$ 3 T translation framework that enhances realism while preserving anatomy. Our method builds upon the Unpaired Neural Schrödinge Bridge (UNSB) with multi-step refinement. To strengthen target distribution alignment, we augment the adversarial objective with DMD2-style diffusion-guided distribution matching using a frozen 3T diffusion teacher. To explicitly constrain global structure beyond patch-level correspondence, we combine PatchNCE with an Anatomical Structure Preservation (ASP) regularizer that enforces soft foreground background consistency and boundary aware constraints. Evaluated on two disjoint cohorts, the proposed framework achieves an improved realism structure trade-off, enhancing distribution level realism on unpaired benchmarks while increasing structural fidelity on the paired cohort compared to unpaired baselines.
>
---
#### [replaced 088] Quadrotor Navigation using Reinforcement Learning with Privileged Information
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于无人机导航任务，解决大障碍物绕行问题。通过强化学习结合特权信息，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2509.08177](https://arxiv.org/pdf/2509.08177)**

> **作者:** Jonathan Lee; Abhishek Rathod; Kshitij Goel; John Stecklein; Wennie Tabib
>
> **摘要:** This paper presents a reinforcement learning-based quadrotor navigation method that leverages efficient differentiable simulation, novel loss functions, and privileged information to navigate around large obstacles. Prior learning-based methods perform well in scenes that exhibit narrow obstacles, but struggle when the goal location is blocked by large walls or terrain. In contrast, the proposed method utilizes time-of-arrival (ToA) maps as privileged information and a yaw alignment loss to guide the robot around large obstacles. The policy is evaluated in photo-realistic simulation environments containing large obstacles, sharp corners, and dead-ends. Our approach achieves an 86% success rate and outperforms baseline strategies by 34%. We deploy the policy onboard a custom quadrotor in outdoor cluttered environments both during the day and night. The policy is validated across 20 flights, covering 589 meters without collisions at speeds up to 4 m/s.
>
---
#### [replaced 089] AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.00589](https://arxiv.org/pdf/2603.00589)**

> **作者:** Cencen Liu; Dongyang Zhang; Wen Yin; Jielei Wang; Tianyu Li; Ji Guo; Wenbo Jiang; Guoqing Wang; Guoming Lu
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** Visual autoregressive (VAR) models have recently emerged as a promising alternative for image generation, offering stable training, non-iterative inference, and high-fidelity synthesis through next-scale prediction. This encourages the exploration of VAR for image super-resolution (ISR), yet its application remains underexplored and faces two critical challenges: locality-biased attention, which fragments spatial structures, and residual-only supervision, which accumulates errors across scales, severely compromises global consistency of reconstructed images. To address these issues, we propose AlignVAR, a globally consistent visual autoregressive framework tailored for ISR, featuring two key components: (1) Spatial Consistency Autoregression (SCA), which applies an adaptive mask to reweight attention toward structurally correlated regions, thereby mitigating excessive locality and enhancing long-range dependencies; and (2) Hierarchical Consistency Constraint (HCC), which augments residual learning with full reconstruction supervision at each scale, exposing accumulated deviations early and stabilizing the coarse-to-fine refinement process. Extensive experiments demonstrate that AlignVAR consistently enhances structural coherence and perceptual fidelity over existing generative methods, while delivering over 10x faster inference with nearly 50% fewer parameters than leading diffusion-based approaches, establishing a new paradigm for efficient ISR.
>
---
#### [replaced 090] NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出一种相位保持的扩散模型（φ-PD），解决图像生成中结构不一致问题，适用于需要几何一致性的任务，如重渲染和图像到图像翻译。**

- **链接: [https://arxiv.org/pdf/2512.05106](https://arxiv.org/pdf/2512.05106)**

> **作者:** Yu Zeng; Charles Ochoa; Mingyuan Zhou; Vishal M. Patel; Vitor Guizilini; Rowan McAllister
>
> **摘要:** Standard diffusion corrupts data using Gaussian noise whose Fourier coefficients have random magnitudes and random phases. While effective for unconditional or text-to-image generation, corrupting phase components destroys spatial structure, making it ill-suited for tasks requiring geometric consistency, such as re-rendering, simulation enhancement, and image-to-image translation. We introduce Phase-Preserving Diffusion (\phi-PD), a model-agnostic reformulation of the diffusion process that preserves input phase while randomizing magnitude, enabling structure-aligned generation without architectural changes or additional parameters. We further propose Frequency-Selective Structured (FSS) noise, which provides continuous control over structural rigidity via a single frequency-cutoff parameter. \phi-PD adds no inference-time cost and is compatible with any diffusion model for images or videos. Across photorealistic and stylized re-rendering, as well as sim-to-real enhancement for driving planners, \phi-PD produces controllable, spatially aligned results. When applied to the CARLA simulator, \phi-PD significantly improves sim-to-real planner transfer performance. The method is complementary to existing conditioning approaches and broadly applicable to image-to-image and video-to-video generation. Videos, additional examples, and code are available on our \href{this https URL}{project page}.
>
---
#### [replaced 091] Learnable Sparsity for Vision Generative Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.02852](https://arxiv.org/pdf/2412.02852)**

> **作者:** Yang Zhang; Er Jin; Wenzhong Liang; Yanfei Dong; Ashkan Khakzar; Philip Torr; Johannes Stegmaier; Kenji Kawaguchi
>
> **备注:** Project page: this https URL
>
> **摘要:** Diffusion models have achieved impressive advancements in various vision tasks. However, these gains often rely on increasing model size, which escalates computational complexity and memory demands, complicating deployment, raising inference costs, and causing environmental impact. While some studies have explored pruning techniques to improve the memory efficiency of diffusion models, most existing methods require extensive retraining to retain the model performance. Retraining a modern large diffusion model is extremely costly and resource-intensive, which limits the practicality of these methods. In this work, we achieve low-cost diffusion pruning without retraining by proposing a model-agnostic structural pruning framework for diffusion models that learns a differentiable mask to sparsify the model. To ensure effective pruning that preserves the quality of the final denoised latent, we design a novel end-to-end pruning objective that spans the entire diffusion process. As end-to-end pruning is memory-intensive, we further propose time step gradient checkpointing, a technique that significantly reduces memory usage during optimization, enabling end-to-end pruning within a limited memory budget. Results on state-of-the-art U-Net diffusion models SDXL and diffusion transformers (FLUX) demonstrate that our method can effectively prune up to 20% parameters with minimal perceptible performance degradation, and notably, without the need for model retraining. We also showcase that our method can still prune on top of time step distilled diffusion models.
>
---
#### [replaced 092] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.00141](https://arxiv.org/pdf/2511.00141)**

> **作者:** Janghoon Cho; Jungsoo Lee; Munawar Hayat; Kyuwoong Hwang; Fatih Porikli; Sungha Choi
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Recent studies in long video understanding have harnessed the advanced visual-language reasoning capabilities of Large Multimodal Models (LMMs), driving the evolution of video-LMMs specialized for processing extended video sequences. However, the scalability of these models is severely limited by the overwhelming volume of visual tokens generated from extended video sequences. To address this challenge, we propose FLoC, an efficient visual token compression framework based on the facility location function, a principled approach that swiftly selects a compact yet highly representative and diverse subset of visual tokens within a predefined budget on the number of visual tokens. By integrating the lazy greedy algorithm, our method achieves remarkable efficiency gains by swiftly selecting a compact subset of tokens, drastically reducing the number of visual tokens while guaranteeing near-optimal performance. Notably, our approach is training-free, model-agnostic, and query-agnostic, providing a versatile solution that seamlessly integrates with diverse video-LLMs and existing workflows. Extensive evaluations on large-scale benchmarks, such as Video-MME, MLVU, LongVideoBench, and EgoSchema, show that our framework consistently surpasses recent compression techniques, highlighting its effectiveness and robustness in addressing the challenges of long video understanding as well as its processing efficiency.
>
---
#### [replaced 093] Seeing Through Uncertainty: A Free-Energy Approach for Real-Time Perceptual Adaptation in Robust Visual Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉导航任务，解决传感器噪声下的鲁棒性问题。提出FEP-Nav框架，通过分解自由能实现实时感知适应。**

- **链接: [https://arxiv.org/pdf/2403.01977](https://arxiv.org/pdf/2403.01977)**

> **作者:** Maytus Piriyajitakonkij; Rishabh Dev Yadav; Mingfei Sun; Mengmi Zhang; Wei Pan
>
> **摘要:** Navigation in the natural world is a feat of adaptive inference, where biological organisms maintain goal-directed behaviour despite noisy and incomplete sensory streams. Central to this ability is the Free Energy Principle (FEP), which posits that perception is a generative process where the brain minimises Variational Free Energy (VFE) to maintain accurate internal models of the world. While Deep Neural Networks (DNNs) have served as powerful analogues for biological brains, they typically lack the real-time plasticity required to handle abrupt sensory shifts. We introduce FEP-Nav, a biologically-inspired framework that implements real-time perceptual adaptation for robust visual navigation. By decomposing VFE into its constituent components--prediction error and Bayesian surprise--we propose a dual-mechanism architecture: a Top-down Decoder that provides an internal expectation of uncorrupted sensory input, and Adaptive Normalisation that dynamically aligns shifted feature distributions with prior beliefs. Theoretically, we demonstrate that this integration of reconstruction and normalisation provides a formal mechanism for minimising VFE during inference without the need for gradient-based updates. Evaluations across a diverse suite of simulated and real-world visual corruptions demonstrate that FEP-Nav facilitates a substantial recovery of navigation performance, consistently exceeding the capabilities of both non-adaptive baselines and strong adaptive methods. We show that bridging machine learning with the brain's variational principles offers a robust strategy for autonomous behaviour, enabling robots to remain functional under sensory conditions that typically degrade the performance of standard adaptive models.
>
---
#### [replaced 094] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.00918](https://arxiv.org/pdf/2603.00918)**

> **作者:** Seungwook Kim; Minsu Cho
>
> **备注:** 19 pages, accepted to CVPR 2026. Project page this https URL
>
> **摘要:** Text-to-image generation powers content creation across design, media, and data augmentation. Post-training of text-to-image generative models is a promising path to better match human preferences, factuality, and improved aesthetics. We introduce SOLACE (Adaptive Rewarding by self-Confidence), a post-training framework that replaces external reward supervision with an internal self-confidence signal, obtained by evaluating how accurately the model recovers injected noise under self-denoising probes. SOLACE converts this intrinsic signal into scalar rewards, enabling fully unsupervised optimization without additional datasets, annotators, or reward models. Empirically, by reinforcing high-confidence generations, SOLACE delivers consistent gains in compositional generation, text rendering and text-image alignment over the baseline. We also find that integrating SOLACE with external rewards results in a complementary improvement, with alleviated reward hacking.
>
---
#### [replaced 095] RadarVLM: A Vision-Language Model Approach for Radar Scene Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21105](https://arxiv.org/pdf/2511.21105)**

> **作者:** Pushkal Mishra; Kshitiz Bansal; Dinesh Bharadia
>
> **摘要:** Radar sensors provide reliable perception across adverse weather, lighting, and long-range conditions, yet existing machine learning approaches remain fragmented and task-specific, with each downstream task employing distinct architectures and training objectives. We present RadarVLM, a vision-language framework that learns unified scene-level representations through structured spatial language supervision. Leveraging the CARLA simulator with a realistic radar model, we collect over 800k radar-caption pairs across 110+ hours of simulated driving in diverse scenarios. We make two key contributions: (1) a structured caption framework encoding vehicle distributions in the radar's native coordinate system, and (2) Spatially-Grounded CLIP (SG-CLIP) objective that replaces binary matching with continuous scene similarity, enabling fine-grained spatial reasoning. We further propose localization-aware evaluation metrics that directly assess spatial accuracy beyond traditional linguistic similarity measures. Validated on generative captioning and vehicle segmentation, SG-CLIP achieves up to 50\% relative F1-score improvement over vanilla CLIP and a 21\% AP gain on segmentation, demonstrating that language grounding produces spatially structured representations.
>
---
#### [replaced 096] Gaussian Wardrobe: Compositional 3D Gaussian Avatars for Free-Form Virtual Try-On
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2603.04290](https://arxiv.org/pdf/2603.04290)**

> **作者:** Zhiyi Chen; Hsuan-I Ho; Tianjian Jiang; Jie Song; Manuel Kaufmann; Chen Guo
>
> **备注:** 3DV 2026, 16 pages, 12 figures
>
> **摘要:** We introduce Gaussian Wardrobe, a novel framework to digitalize compositional 3D neural avatars from multi-view videos. Existing methods for 3D neural avatars typically treat the human body and clothing as an inseparable entity. However, this paradigm fails to capture the dynamics of complex free-form garments and limits the reuse of clothing across different individuals. To overcome these problems, we develop a novel, compositional 3D Gaussian representation to build avatars from multiple layers of free-form garments. The core of our method is decomposing neural avatars into bodies and layers of shape-agnostic neural garments. To achieve this, our framework learns to disentangle each garment layer from multi-view videos and canonicalizes it into a shape-independent space. In experiments, our method models photorealistic avatars with high-fidelity dynamics, achieving new state-of-the-art performance on novel pose synthesis benchmarks. In addition, we demonstrate that the learned compositional garments contribute to a versatile digital wardrobe, enabling a practical virtual try-on application where clothing can be freely transferred to new subjects. Project page: this https URL
>
---
#### [replaced 097] NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.04179](https://arxiv.org/pdf/2603.04179)**

> **作者:** Weirong Chen; Chuanxia Zheng; Ganlin Zhang; Andrea Vedaldi; Daniel Cremers
>
> **备注:** Accepted to ICLR 2026. Project Page: this https URL
>
> **摘要:** We present NOVA3R, an effective approach for non-pixel-aligned 3D reconstruction from a set of unposed images in a feed-forward manner. Unlike pixel-aligned methods that tie geometry to per-ray predictions, our formulation learns a global, view-agnostic scene representation that decouples reconstruction from pixel alignment. This addresses two key limitations in pixel-aligned 3D: (1) it recovers both visible and invisible points with a complete scene representation, and (2) it produces physically plausible geometry with fewer duplicated structures in overlapping regions. To achieve this, we introduce a scene-token mechanism that aggregates information across unposed images and a diffusion-based 3D decoder that reconstructs complete, non-pixel-aligned point clouds. Extensive experiments on both scene-level and object-level datasets demonstrate that NOVA3R outperforms state-of-the-art methods in terms of reconstruction accuracy and completeness.
>
---
#### [replaced 098] HSG-12M: A Large-Scale Benchmark of Spatial Multigraphs from the Energy Spectra of Non-Hermitian Crystals
- **分类: cs.LG; cond-mat.mes-hall; cond-mat.other; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08618](https://arxiv.org/pdf/2506.08618)**

> **作者:** Xianquan Yan; Hakan Akgün; Kenji Kawaguchi; N. Duane Loh; Ching Hua Lee
>
> **备注:** 49 pages, 13 figures, 14 tables. Code & pipeline: [this https URL] Dataset: [this https URL] Dataset released under CC BY 4.0. Benchmark scripts and data loaders included
>
> **摘要:** AI is transforming scientific research by revealing new ways to understand complex physical systems, but its impact remains constrained by the lack of large, high-quality domain-specific datasets. A rich, largely untapped resource lies in non-Hermitian quantum physics, where the energy spectra of crystals form intricate geometries on the complex plane -- termed as Hamiltonian spectral graphs. Despite their significance as fingerprints for electronic behavior, their systematic study has been intractable due to the reliance on manual extraction. To unlock this potential, we introduce Poly2Graph: a high-performance, open-source pipeline that automates the mapping of 1-D crystal Hamiltonians to spectral graphs. Using this tool, we present HSG-12M: a dataset containing 11.6 million static and 5.1 million dynamic Hamiltonian spectral graphs across 1401 characteristic-polynomial classes, distilled from 177 TB of spectral potential data. Crucially, HSG-12M is the first large-scale dataset of spatial multigraphs -- graphs embedded in a metric space where multiple geometrically distinct trajectories between two nodes are retained as separate edges. This simultaneously addresses a critical gap, as existing graph benchmarks overwhelmingly assume simple, non-spatial edges, discarding vital geometric information. Benchmarks with popular GNNs expose new challenges in learning spatial multi-edges at scale. Beyond its practical utility, we show that spectral graphs serve as universal topological fingerprints of polynomials, vectors, and matrices, forging a new algebra-to-graph link. HSG-12M lays the groundwork for data-driven scientific discovery in condensed matter physics, new opportunities in geometry-aware graph learning and beyond.
>
---
