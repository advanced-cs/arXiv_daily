# 计算机视觉 cs.CV

- **最新发布 138 篇**

- **更新 108 篇**

## 最新发布

#### [new 001] Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于模型融合任务，解决LoRA模块合并中的子空间覆盖不足和方向各向异性问题。通过TARA-Merging方法，提升模型对多任务的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.26299](https://arxiv.org/pdf/2603.26299)**

> **作者:** Wooseong Jeong; Wonyoung Lee; Kuk-Jin Yoon
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Merging multiple Low-Rank Adaptation (LoRA) modules is promising for constructing general-purpose systems, yet challenging because LoRA update directions span different subspaces and contribute unevenly. When merged naively, such mismatches can weaken the directions most critical to certain task losses while overemphasizing relatively less important ones, ultimately reducing the model's ability to represent all tasks faithfully. We revisit this problem through two perspectives: subspace coverage, which captures how broadly LoRA directions cover diverse representational directions, and anisotropy, which reflects the imbalance of influence across those directions. We propose TARA-Merging (Task-Rank Anisotropy Alignment), which aligns merging weights using a preference-weighted cross-entropy pseudo-loss while preserving task-relevant LoRA subspaces. This ensures broad subspace coverage and mitigates anisotropy via direction-wise reweighting. Across eight vision and six NLI benchmarks, TARA-Merging consistently outperforms vanilla and LoRA-aware baselines, demonstrating strong robustness and generalization, and highlighting the importance of addressing both subspace coverage and anisotropy in LoRA merging.
>
---
#### [new 002] MemCam: Memory-Augmented Camera Control for Consistent Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于交互式视频生成任务，旨在解决动态摄像机控制下场景一致性不足的问题。通过引入记忆增强机制，提升长视频生成的场景一致性。**

- **链接: [https://arxiv.org/pdf/2603.26193](https://arxiv.org/pdf/2603.26193)**

> **作者:** Xinhang Gao; Junlin Guan; Shuhan Luo; Wenzhuo Li; Guanghuan Tan; Jiacheng Wang
>
> **备注:** 6 pages, 3 figures, 3 tables, accepted by IJCNN 2026
>
> **摘要:** Interactive video generation has significant potential for scene simulation and video creation. However, existing methods often struggle with maintaining scene consistency during long video generation under dynamic camera control due to limited contextual information. To address this challenge, we propose MemCam, a memory-augmented interactive video generation approach that treats previously generated frames as external memory and leverages them as contextual conditioning to achieve controllable camera viewpoints with high scene consistency. To enable longer and more relevant context, we design a context compression module that encodes memory frames into compact representations and employs co-visibility-based selection to dynamically retrieve the most relevant historical frames, thereby reducing computational overhead while enriching contextual information. Experiments on interactive video generation tasks show that MemCam significantly outperforms existing baseline methods as well as open-source state-of-the-art approaches in terms of scene consistency, particularly in long video scenarios with large camera rotations.
>
---
#### [new 003] Tunable Soft Equivariance with Guarantees
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉领域，解决模型严格等变性不足的问题。通过投影权重到特定子空间，构建可调软等变模型，提升性能并降低等变误差。**

- **链接: [https://arxiv.org/pdf/2603.26657](https://arxiv.org/pdf/2603.26657)**

> **作者:** Md Ashiqur Rahman; Lim Jun Hao; Jeremiah Jiang; Teck-Yian Lim; Raymond A. Yeh
>
> **摘要:** Equivariance is a fundamental property in computer vision models, yet strict equivariance is rarely satisfied in real-world data, which can limit a model's performance. Controlling the degree of equivariance is therefore desirable. We propose a general framework for constructing soft equivariant models by projecting the model weights into a designed subspace. The method applies to any pre-trained architecture and provides theoretical bounds on the induced equivariance error. Empirically, we demonstrate the effectiveness of our method on multiple pre-trained backbones, including ViT and ResNet, across image classification, semantic segmentation, and human-trajectory prediction tasks. Notably, our approach improves the performance while simultaneously reducing equivariance error on the competitive ImageNet benchmark.
>
---
#### [new 004] Progressive Learning with Anatomical Priors for Reliable Left Atrial Scar Segmentation from Late Gadolinium Enhancement MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于左心房瘢痕分割任务，旨在解决LGE-MRI图像中瘢痕分割准确性低的问题。通过引入解剖先验和分阶段学习策略，提升分割精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.26186](https://arxiv.org/pdf/2603.26186)**

> **作者:** Jing Zhang; Bastien Bergere; Emilie Bollache; Jonas Leite; Mikaël Laredo; Alban Redheuil; Nadjia Kachenoura
>
> **备注:** 16 pages, 3 figures, 3 tables
>
> **摘要:** Cardiac MRI late gadolinium enhancement (LGE) enables non-invasive identification of left atrial (LA) scar, whose spatial distribution is strongly associated with atrial fibrillation (AF) severity and recurrence. However, automatic LA scar segmentation remains challenging due to low contrast, annotation variability, and the lack of anatomical constraints, often leading to non-reliable predictions. Accordingly, our aim was to propose a progressive learning strategy to segment LA scar from LGE images inspired from a clinical workflow. A 3-stage framework based on SwinUNETR was implemented, comprising: 1) a first LA cavity pre-learning model, 2) dual-task model which further learns spatial relationship between LA geometry and scar patterns, and 3) fine-tuning on precise segmentation of the scar. Furthermore, we introduced an anatomy-aware spatially weighted loss that incorporates prior clinical knowledge by constraining scar predictions to anatomically plausible LA wall regions while mitigating annotation bias. Our preliminary results obtained on validation LGE volumes from LASCARQS public dataset after 5-fold cross validation, LA segmentation had Dice score of 0.94, LA scar segmentation achieved Dice score of 0.50, Hausdorff Distance of 11.84 mm, Average Surface Distance of 1.80 mm, outperforming only a one-stage scar segmentation with 0.49, 13.02 mm, 1.96 mm, repectively. By explicitly embedding clinical anatomical priors and diagnostic reasoning into deep learning, the proposed approach improved the accuracy and reliability of LA scar segmentation from LGE, revealing the importance of clinically informed model design.
>
---
#### [new 005] Zero-Shot Depth from Defocus
- **分类: cs.CV**

- **简介: 本文研究深度从失焦（DfD）任务，解决零样本泛化问题。提出新基准ZEDD和网络FOSSA，提升深度估计性能。**

- **链接: [https://arxiv.org/pdf/2603.26658](https://arxiv.org/pdf/2603.26658)**

> **作者:** Yiming Zuo; Hongyu Wen; Venkat Subramanian; Patrick Chen; Karhan Kayan; Mario Bijelic; Felix Heide; Jia Deng
>
> **摘要:** Depth from Defocus (DfD) is the task of estimating a dense metric depth map from a focus stack. Unlike previous works overfitting to a certain dataset, this paper focuses on the challenging and practical setting of zero-shot generalization. We first propose a new real-world DfD benchmark ZEDD, which contains 8.3x more scenes and significantly higher quality images and ground-truth depth maps compared to previous benchmarks. We also design a novel network architecture named FOSSA. FOSSA is a Transformer-based architecture with novel designs tailored to the DfD task. The key contribution is a stack attention layer with a focus distance embedding, allowing efficient information exchange across the focus stack. Finally, we develop a new training data pipeline allowing us to utilize existing large-scale RGBD datasets to generate synthetic focus stacks. Experiment results on ZEDD and other benchmarks show a significant improvement over the baselines, reducing errors by up to 55.7%. The ZEDD benchmark is released at this https URL. The code and checkpoints are released at this https URL.
>
---
#### [new 006] SALMUBench: A Benchmark for Sensitive Association-Level Multimodal Unlearning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态模型的敏感信息删除任务，旨在解决对比训练模型中细粒度关联遗忘问题。构建了SALMUBench基准，提出新评估协议以衡量删除效果与副作用。**

- **链接: [https://arxiv.org/pdf/2603.26316](https://arxiv.org/pdf/2603.26316)**

> **作者:** Cai Selvas-Sala; Lei Kang; Lluis Gomez
>
> **备注:** Accepted to CVPR 2026. Project page: this http URL
>
> **摘要:** As multimodal models like CLIP become integral to downstream systems, the need to remove sensitive information is critical. However, machine unlearning for contrastively-trained encoders remains underexplored, and existing evaluations fail to diagnose fine-grained, association-level forgetting. We introduce SALMUBench (Sensitive Association-Level Multimodal Unlearning), a benchmark built upon a synthetic dataset of 60K persona-attribute associations and two foundational models: a Compromised model polluted with this data, and a Clean model without it. To isolate unlearning effects, both are trained from scratch on the same 400M-pair retain base, with the Compromised model additionally trained on the sensitive set. We propose a novel evaluation protocol with structured holdout sets (holdout identity, holdout association) to precisely measure unlearning efficacy and collateral damage. Our benchmark reveals that while utility-efficient deletion is feasible, current methods exhibit distinct failure modes: they either fail to forget effectively or over-generalize by erasing more than intended. SALMUBench sets a new standard for comprehensive unlearning evaluation, and we publicly release our dataset, models, evaluation scripts, and leaderboards to foster future research.
>
---
#### [new 007] CREval: An Automated Interpretable Evaluation for Creative Image Manipulation under Complex Instructions
- **分类: cs.CV**

- **简介: 该论文聚焦于复杂指令下的创意图像编辑任务，提出CREval评估框架和基准，解决现有评估方法不系统、不可解释的问题。**

- **链接: [https://arxiv.org/pdf/2603.26174](https://arxiv.org/pdf/2603.26174)**

> **作者:** Chonghuinan Wang; Zihan Chen; Yuxiang Wei; Tianyi Jiang; Xiaohe Wu; Fan Li; Wangmeng Zuo; Hongxun Yao
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Instruction-based multimodal image manipulation has recently made rapid progress. However, existing evaluation methods lack a systematic and human-aligned framework for assessing model performance on complex and creative editing tasks. To address this gap, we propose CREval, a fully automated question-answer (QA)-based evaluation pipeline that overcomes the incompleteness and poor interpretability of opaque Multimodal Large Language Models (MLLMs) scoring. Simultaneously, we introduce CREval-Bench, a comprehensive benchmark specifically designed for creative image manipulation under complex instructions. CREval-Bench covers three categories and nine creative dimensions, comprising over 800 editing samples and 13K evaluation queries. Leveraging this pipeline and benchmark, we systematically evaluate a diverse set of state-of-the-art open and closed-source models. The results reveal that while closed-source models generally outperform open-source ones on complex and creative tasks, all models still struggle to complete such edits effectively. In addition, user studies demonstrate strong consistency between CREval's automated metrics and human judgments. Therefore, CREval provides a reliable foundation for evaluating image editing models on complex and creative image manipulation tasks, and highlights key challenges and opportunities for future research.
>
---
#### [new 008] LEMON: a foundation model for nuclear morphology in Computational Pathology
- **分类: cs.CV**

- **简介: 该论文提出LEMON模型，解决计算病理学中单细胞图像表示学习问题。通过自监督学习，提取细胞形态特征，支持癌症研究与精准医学。**

- **链接: [https://arxiv.org/pdf/2603.25802](https://arxiv.org/pdf/2603.25802)**

> **作者:** Loïc Chadoutaud; Alice Blondel; Hana Feki; Jacqueline Fontugne; Emmanuel Barillot; Thomas Walter
>
> **摘要:** Computational pathology relies on effective representation learning to support cancer research and precision medicine. Although self-supervised learning has driven major progress at the patch and whole-slide image levels, representation learning at the single-cell level remains comparatively underexplored, despite its importance for characterizing cell types and cellular phenotypes. We introduce LEMON (Learning Embeddings from Morphology Of Nuclei), a self-supervised foundation model for scalable single-cell image representation learning. Trained on millions of cell images from diverse tissues and cancer types, LEMON learns robust and versatile morphological representations that support large-scale single-cell analyses in pathology. We evaluate LEMON on five benchmark datasets across a range of prediction tasks and show that it provides strong performance, highlighting its potential as a new paradigm for cell-level computational pathology. Model weights are available at this https URL.
>
---
#### [new 009] The Limits of Learning from Pictures and Text: Vision-Language Models and Embodied Scene Understanding
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型在场景理解中的局限性，旨在解决其无法有效捕捉物体用途等问题。通过实验和分析，发现模型在 affordance 任务上表现不足，表明单纯依赖图像和文本不足以实现完整场景理解。**

- **链接: [https://arxiv.org/pdf/2603.26589](https://arxiv.org/pdf/2603.26589)**

> **作者:** Gillian Rosenberg; Skylar Stadhard; Bruce C. Hansen; Michelle R. Greene
>
> **备注:** 7 figures, 5 tables
>
> **摘要:** What information is sufficient to learn the full richness of human scene understanding? The distributional hypothesis holds that the statistical co-occurrence of language and images captures the conceptual knowledge underlying visual cognition. Vision-language models (VLMs) are trained on massive paired text-image corpora but lack embodied experience, making them an ideal test of the distributional hypothesis. We report two experiments comparing descriptions generated by 18 VLMs to those of over 2000 human observers across 15 high-level scene understanding tasks, spanning general knowledge, affordances, sensory experiences, affective responses, and future prediction. Because many tasks lack ground truth answers, we developed a Human-Calibrated Cosine Distance (HCD) metric that measures VLM output similarity to the distribution of human responses, scaled by within-human variability. In Experiment 1, VLMs approached human-level performance on general knowledge tasks, but showed a robust deficit for affordance tasks that resisted prompt engineering and did not improve with newer model releases. In Experiment 2, we tested six mechanistic hypotheses for explaining this affordance gap, finding that the deficit was structural rather than stylistic and was not resolved by providing explicit spatial information. Corpus analyses revealed that image captioning datasets contain sparse agent-addressed affordance language, consistent with Gricean accounts of why embodied knowledge may be systematically underrepresented in language. Together, these findings suggest that distributional learning from images and text is insufficient for affordance-based scene understanding, implying that some dimensions of human visual cognition may require the kind of agent-centered, three-dimensional experience that no photograph or caption can encode.
>
---
#### [new 010] MA-Bench: Towards Fine-grained Micro-Action Understanding
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决微动作理解不足的问题。构建了MA-Bench基准和训练数据，提升模型对细微动作的识别与解释能力。**

- **链接: [https://arxiv.org/pdf/2603.26586](https://arxiv.org/pdf/2603.26586)**

> **作者:** Kun Li; Jihao Gu; Fei Wang; Zhiliang Wu; Hehe Fan; Dan Guo
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** With the rapid development of Multimodal Large Language Models (MLLMs), their potential in Micro-Action understanding, a vital role in human emotion analysis, remains unexplored due to the absence of specialized benchmarks. To tackle this issue, we present MA-Bench, a benchmark comprising 1,000 videos and a three-tier evaluation architecture that progressively examines micro-action perception, relational comprehension, and interpretive reasoning. MA-Bench contains 12,000 structured question-answer pairs, enabling systematic assessment of both recognition accuracy and action interpretation. The results of 23 representative MLLMs reveal that there are significant challenges in capturing motion granularity and fine-grained body-part dynamics. To address these challenges, we further construct MA-Bench-Train, a large-scale training corpus with 20.5K videos annotated with structured micro-action captions for fine-tuning MLLMs. The results of Qwen3-VL-8B fine-tuned on MA-Bench-Train show clear performance improvements across micro-action reasoning and explanation tasks. Our work aims to establish a foundation benchmark for advancing MLLMs in understanding subtle micro-action and human-related behaviors. Project Page: this https URL
>
---
#### [new 011] Geo$^\textbf{2}$: Geometry-Guided Cross-view Geo-Localization and Image Synthesis
- **分类: cs.CV**

- **简介: 该论文提出Geo²，解决跨视图地理定位与图像合成问题，利用几何先验提升任务性能。**

- **链接: [https://arxiv.org/pdf/2603.25819](https://arxiv.org/pdf/2603.25819)**

> **作者:** Yancheng Zhang; Xiaohan Zhang; Guangyu Sun; Zonglin Lyu; Safwan Wshah; Chen Chen
>
> **摘要:** Cross-view geo-spatial learning consists of two important tasks: Cross-View Geo-Localization (CVGL) and Cross-View Image Synthesis (CVIS), both of which rely on establishing geometric correspondences between ground and aerial views. Recent Geometric Foundation Models (GFMs) have demonstrated strong capabilities in extracting generalizable 3D geometric features from images, but their potential in cross-view geo-spatial tasks remains underexplored. In this work, we present Geo^2, a unified framework that leverages Geometric priors from GFMs (e.g., VGGT) to jointly perform geo-spatial tasks, CVGL and bidirectional CVIS. Despite the 3D reconstruction ability of GFMs, directly applying them to CVGL and CVIS remains challenging due to the large viewpoint gap between ground and aerial imagery. We propose GeoMap, which embeds ground and aerial features into a shared 3D-aware latent space, effectively reducing cross-view discrepancies for localization. This shared latent space naturally bridges cross-view image synthesis in both directions. To exploit this, we propose GeoFlow, a flow-matching model conditioned on geometry-aware latent embeddings. We further introduce a consistency loss to enforce latent alignment between the two synthesis directions, ensuring bidirectional coherence. Extensive experiments on standard benchmarks, including CVUSA, CVACT, and VIGOR, demonstrate that Geo^2 achieves state-of-the-art performance in both localization and synthesis, highlighting the effectiveness of 3D geometric priors for cross-view geo-spatial learning.
>
---
#### [new 012] DUGAE: Unified Geometry and Attribute Enhancement via Spatiotemporal Correlations for G-PCC Compressed Dynamic Point Clouds
- **分类: cs.CV**

- **简介: 该论文提出DUGAE框架，用于提升G-PCC压缩动态点云的几何与属性质量，解决静态处理方法无法利用时空相关性的问题。**

- **链接: [https://arxiv.org/pdf/2603.26183](https://arxiv.org/pdf/2603.26183)**

> **作者:** Pan Zhao; Hui Yuan; Chang Sun; Chongzhen Tian; Raouf Hamzaoui; Sam Kwong
>
> **摘要:** Existing post-decoding quality enhancement methods for point clouds are designed for static data and typically process each frame independently. As a result, they cannot effectively exploit the spatiotemporal correlations present in point cloud this http URL propose a unified geometry and attribute enhancement framework (DUGAE) for G-PCC compressed dynamic point clouds that explicitly exploits inter-frame spatiotemporal correlations in both geometry and attributes. First, a dynamic geometry enhancement network (DGE-Net) based on sparse convolution (SPConv) and feature-domain geometry motion compensation (GMC) aligns and aggregates spatiotemporal information. Then, a detail-aware k-nearest neighbors (DA-KNN) recoloring module maps the original attributes onto the enhanced geometry at the encoder side, improving mapping completeness and preserving attribute details. Finally, a dynamic attribute enhancement network (DAE-Net) with dedicated temporal feature extraction and feature-domain attribute motion compensation (AMC) refines attributes by modeling complex spatiotemporal correlations. On seven dynamic point clouds from the 8iVFB v2, Owlii, and MVUB datasets, DUGAE significantly enhanced the performance of the latest G-PCC geometry-based solid content test model (GeS-TM v10). For geometry (D1), it achieved an average BD-PSNR gain of 11.03 dB and a 93.95% BD-bitrate reduction. For the luma component, it achieved a 4.23 dB BD-PSNR gain with a 66.61% BD-bitrate reduction. DUGAE also improved perceptual quality (as measured by PCQM) and outperformed V-PCC. Our source code will be released on GitHub at: this https URL
>
---
#### [new 013] VLAgeBench: Benchmarking Large Vision-Language Models for Zero-Shot Human Age Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于零样本人脸年龄估计任务，旨在评估大型视觉语言模型在无需微调情况下的表现，解决传统方法依赖标注数据的问题。**

- **链接: [https://arxiv.org/pdf/2603.26015](https://arxiv.org/pdf/2603.26015)**

> **作者:** Rakib Hossain Sajib; Md Kishor Morol; Rajan Das Gupta; Mohammad Sakib Mahmood; Shuvra Smaran Das
>
> **摘要:** Human age estimation from facial images represents a challenging computer vision task with significant applications in biometrics, healthcare, and human-computer interaction. While traditional deep learning approaches require extensive labeled datasets and domain-specific training, recent advances in large vision-language models (LVLMs) offer the potential for zero-shot age estimation. This study presents a comprehensive zero-shot evaluation of state-of-the-art Large Vision-Language Models (LVLMs) for facial age estimation, a task traditionally dominated by domain-specific convolutional networks and supervised learning. We assess the performance of GPT-4o, Claude 3.5 Sonnet, and LLaMA 3.2 Vision on two benchmark datasets, UTKFace and FG-NET, without any fine-tuning or task-specific adaptation. Using eight evaluation metrics, including MAE, MSE, RMSE, MAPE, MBE, $R^2$, CCC, and $\pm$5-year accuracy, we demonstrate that general-purpose LVLMs can deliver competitive performance in zero-shot settings. Our findings highlight the emergent capabilities of LVLMs for accurate biometric age estimation and position these models as promising tools for real-world applications. Additionally, we highlight performance disparities linked to image quality and demographic subgroups, underscoring the need for fairness-aware multimodal inference. This work introduces a reproducible benchmark and positions LVLMs as promising tools for real-world applications in forensic science, healthcare monitoring, and human-computer interaction. The benchmark focuses on strict zero-shot inference without fine-tuning and highlights remaining challenges related to prompt sensitivity, interpretability, computational cost, and demographic fairness.
>
---
#### [new 014] R-PGA: Robust Physical Adversarial Camouflage Generation via Relightable 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于物理对抗伪装生成任务，旨在解决动态场景下对抗样本泛化能力差的问题。通过引入3DGS和HPCM模块提升模拟精度与优化鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26067](https://arxiv.org/pdf/2603.26067)**

> **作者:** Tianrui Lou; Siyuan Liang; Jiawei Liang; Yuze Gao; Xiaochun Cao
>
> **备注:** Under review
>
> **摘要:** Physical adversarial camouflage poses a severe security threat to autonomous driving systems by mapping adversarial textures onto 3D objects. Nevertheless, current methods remain brittle in complex dynamic scenarios, failing to generalize across diverse geometric (e.g., viewing configurations) and radiometric (e.g., dynamic illumination, atmospheric scattering) variations. We attribute this deficiency to two fundamental limitations in simulation and optimization. First, the reliance on coarse, oversimplified simulations (e.g., via CARLA) induces a significant domain gap, confining optimization to a biased feature space. Second, standard strategies targeting average performance result in a rugged loss landscape, leaving the camouflage vulnerable to configuration this http URL bridge these gaps, we propose the Relightable Physical 3D Gaussian Splatting (3DGS) based Attack framework (R-PGA). Technically, to address the simulation fidelity issue, we leverage 3DGS to ensure photo-realistic reconstruction and augment it with physically disentangled attributes to decouple intrinsic material from lighting. Furthermore, we design a hybrid rendering pipeline that leverages precise Relightable 3DGS for foreground rendering, while employing a pre-trained image translation model to synthesize plausible relighted backgrounds that align with the relighted this http URL address the optimization robustness issue, we propose the Hard Physical Configuration Mining (HPCM) module, designed to actively mine worst-case physical configurations and suppress their corresponding loss peaks. This strategy not only diminishes the overall loss magnitude but also effectively flattens the rugged loss landscape, ensuring consistent adversarial effectiveness and robustness across varying physical configurations.
>
---
#### [new 015] Reflect to Inform: Boosting Multimodal Reasoning via Information-Gain-Driven Verification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态推理任务，旨在解决长文本生成中模型偏离图像证据的问题。通过引入VRE框架，提升模型的视觉验证能力，减少幻觉。**

- **链接: [https://arxiv.org/pdf/2603.26348](https://arxiv.org/pdf/2603.26348)**

> **作者:** Shuai Lv; Chang Liu; Feng Tang; Yujie Yuan; Aojun Zhou; Kui Zhang; Xi Yang; Yangqiu Song
>
> **摘要:** Multimodal Large Language Models (MLLMs) achieve strong multimodal reasoning performance, yet we identify a recurring failure mode in long-form generation: as outputs grow longer, models progressively drift away from image evidence and fall back on textual priors, resulting in ungrounded reasoning and hallucinations. Interestingly, Based on attention analysis, we find that MLLMs have a latent capability for late-stage visual verification that is present but not consistently activated. Motivated by this observation, we propose Visual Re-Examination (VRE), a self-evolving training framework that enables MLLMs to autonomously perform visual introspection during reasoning without additional visual inputs. Rather than distilling visual capabilities from a stronger teacher, VRE promotes iterative self-improvement by leveraging the model itself to generate reflection traces, making visual information actionable through information gain. Extensive experiments across diverse multimodal benchmarks demonstrate that VRE consistently improves reasoning accuracy and perceptual reliability, while substantially reducing hallucinations, especially in long-chain settings. Code is available at this https URL.
>
---
#### [new 016] Knowledge is Power: Advancing Few-shot Action Recognition with Multimodal Semantics from MLLMs
- **分类: cs.CV**

- **简介: 该论文属于少样本动作识别任务，旨在解决传统方法依赖单一视觉空间和低效特征提取的问题。通过引入多模态大语言模型，构建端到端框架，提升动作识别性能。**

- **链接: [https://arxiv.org/pdf/2603.26033](https://arxiv.org/pdf/2603.26033)**

> **作者:** Jiazheng Xing; Chao Xu; Hangjie Yuan; Mengmeng Wang; Jun Dan; Hangwei Qian; Yong Liu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have propelled the field of few-shot action recognition (FSAR). However, preliminary explorations in this area primarily focus on generating captions to form a suboptimal feature->caption->feature pipeline and adopt metric learning solely within the visual space. In this paper, we propose FSAR-LLaVA, the first end-to-end method to leverage MLLMs (such as Video-LLaVA) as a multimodal knowledge base for directly enhancing FSAR. First, at the feature level, we leverage the MLLM's multimodal decoder to extract spatiotemporally and semantically enriched representations, which are then decoupled and enhanced by our Multimodal Feature-Enhanced Module into distinct visual and textual features that fully exploit their semantic knowledge for FSAR. Next, we leverage the versatility of MLLMs to craft input prompts that flexibly adapt to diverse scenarios, and use their aligned outputs to drive our designed Composite Task-Oriented Prototype Construction, effectively bridging the distribution gap between meta-train and meta-test sets. Finally, to enable multimodal features to guide metric learning jointly, we introduce a training-free Multimodal Prototype Matching Metric that adaptively selects the most decisive cues and efficiently leverages the decoupled feature representations produced by MLLMs. Extensive experiments demonstrate superior performance across various tasks with minimal trainable parameters.
>
---
#### [new 017] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning
- **分类: cs.CV**

- **简介: 该论文属于视频表示学习任务，解决图像到视频迁移中的时序一致性和语义分离性矛盾。通过轻量投影层和优化目标，提升视频任务性能。**

- **链接: [https://arxiv.org/pdf/2603.26597](https://arxiv.org/pdf/2603.26597)**

> **作者:** Yang Liu; Qianqian Xu; Peisong Wen; Siran Dai; Xilin Zhao; Qingming Huang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Recent studies have made notable progress in video representation learning by transferring image-pretrained models to video tasks, typically with complex temporal modules and video fine-tuning. However, fine-tuning heavy modules may compromise inter-video semantic separability, i.e., the essential ability to distinguish objects across videos. While reducing the tunable parameters hinders their intra-video temporal consistency, which is required for stable representations of the same object within a video. This dilemma indicates a potential trade-off between the intra-video temporal consistency and inter-video semantic separability during image-to-video transfer. To this end, we propose the Consistency-Separability Trade-off Transfer Learning (Co-Settle) framework, which applies a lightweight projection layer on top of the frozen image-pretrained encoder to adjust representation space with a temporal cycle consistency objective and a semantic separability constraint. We further provide a theoretical support showing that the optimized projection yields a better trade-off between the two properties under appropriate conditions. Experiments on eight image-pretrained models demonstrate consistent improvements across multiple levels of video tasks with only five epochs of self-supervised training. The code is available at this https URL.
>
---
#### [new 018] Only Whats Necessary: Pareto Optimal Data Minimization for Privacy Preserving Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，解决隐私保护与检测性能之间的平衡问题。提出一种基于帕累托最优的数据最小化框架，抑制PII同时保留关键异常检测信息。**

- **链接: [https://arxiv.org/pdf/2603.26354](https://arxiv.org/pdf/2603.26354)**

> **作者:** Nazia Aslam; Abhisek Ray; Thomas B. Moeslund; Kamal Nasrollahi
>
> **备注:** 10 pages, CVPR conference
>
> **摘要:** Video anomaly detection (VAD) systems are increasingly deployed in safety critical environments and require a large amount of data for accurate detection. However, such data may contain personally identifiable information (PII), including facial cues and sensitive demographic attributes, creating compliance challenges under the EU General Data Protection Regulation (GDPR). In particular, GDPR requires that personal data be limited to what is strictly necessary for a specified processing purpose. To address this, we introduce Only What's Necessary, a privacy-by-design framework for VAD that explicitly controls the amount and type of visual information exposed to the detection pipeline. The framework combines breadth based and depth based data minimization mechanisms to suppress PII while preserving cues relevant to anomaly detection. We evaluate a range of minimization configurations by feeding the minimized videos to both a VAD model and a privacy inference model. We employ two ranking based methods, along with Pareto analysis, to characterize the resulting trade off between privacy and utility. From the non-dominated frontier, we identify sweet spot operating points that minimize personal data exposure with limited degradation in detection performance. Extensive experiments on publicly available datasets demonstrate the effectiveness of the proposed framework.
>
---
#### [new 019] Diffusion MRI Transformer with a Diffusion Space Rotary Positional Embedding (D-RoPE)
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决dMRI特征学习的挑战。提出D-RoPE模型，以捕捉扩散数据的空间与方向特性，提升模型在不同采集条件下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.25977](https://arxiv.org/pdf/2603.25977)**

> **作者:** Gustavo Chau Loo Kung; Mohammad Abbasi; Camila Blank; Juze Zhang; Alan Q. Wang; Sophie Ostmeier; Akshay Chaudhari; Kilian Pohl; Ehsan Adeli
>
> **摘要:** Diffusion Magnetic Resonance Imaging (dMRI) plays a critical role in studying microstructural changes in the brain. It is, therefore, widely used in clinical practice; yet progress in learning general-purpose representations from dMRI has been limited. A key challenge is that existing deep learning approaches are not well-suited to capture the unique properties of diffusion signals. Brain dMRI is normally composed of several brain volumes, each with different attenuation characteristics dependent on the direction and strength of the diffusion-sensitized gradients. Thus, there is a need to jointly model spatial, diffusion-weighting, and directional dependencies in dMRI. Furthermore, varying acquisition protocols (e.g., differing numbers of directions) further limit traditional models. To address these gaps, we introduce a diffusion space rotatory positional embedding (D-RoPE) plugged into our dMRI transformer to capture both the spatial structure and directional characteristics of diffusion data, enabling robust and transferable representations across diverse acquisition settings and an arbitrary number of diffusion directions. After self-supervised masked autoencoding pretraining, tests on several downstream tasks show that the learned representations and the pretrained model can provide competitive or superior performance compared to several baselines in these downstream tasks (even compared to a fully trained baseline); the finetuned features from our pretrained encoder resulted in a 6% higher accuracy in classifying mild cognitive impairment and a 0.05 increase in the correlation coefficient when predicting cognitive scores. Code is available at: this http URL.
>
---
#### [new 020] Beyond Where to Look: Trajectory-Guided Reinforcement Learning for Multimodal RLVR
- **分类: cs.CV**

- **简介: 该论文属于多模态强化学习任务，旨在解决视觉证据与推理脱节的问题。提出TGRL方法，通过专家轨迹引导模型有效整合视觉信息进行细粒度推理。**

- **链接: [https://arxiv.org/pdf/2603.26126](https://arxiv.org/pdf/2603.26126)**

> **作者:** Jinda Lu; Junkang Wu; Jinghan Li; Kexin Huang; Shuo Yang; Mingzhu Chen; Jiancan Wu; Kuien Liu; Xiang Wang
>
> **摘要:** Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) for multimodal large language models (MLLMs) have mainly focused on improving final answer correctness and strengthening visual grounding. However, a critical bottleneck remains: although models can attend to relevant visual regions, they often fail to effectively incorporate visual evidence into subsequent reasoning, leading to reasoning chains that are weakly grounded in visual facts. To address this issue, we propose Trajectory-Guided Reinforcement Learning (TGRL), which guides the policy model to integrate visual evidence into fine-grained reasoning processes using expert reasoning trajectories from stronger models. We further introduce token-level reweighting and trajectory filtering to ensure stable and effective policy optimization. Extensive experiments on multiple multimodal reasoning benchmarks demonstrate that TGRL consistently improves reasoning performance and effectively bridges the gap between visual perception and logical reasoning.
>
---
#### [new 021] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions
- **分类: cs.CV**

- **简介: 该论文属于4D重建任务，解决单目视频中人体-可动物体交互的重建问题。提出ArtHOI框架，融合多模型先验并优化物体尺度与姿态，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.25791](https://arxiv.org/pdf/2603.25791)**

> **作者:** Zikai Wang; Zhilu Zhang; Yiqing Wang; Hui Li; Wangmeng Zuo
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Existing hand-object interactions (HOI) methods are largely limited to rigid objects, while 4D reconstruction methods of articulated objects generally require pre-scanning the object or even multi-view videos. It remains an unexplored but significant challenge to reconstruct 4D human-articulated-object interactions from a single monocular RGB video. Fortunately, recent advancements in foundation models present a new opportunity to address this highly ill-posed problem. To this end, we introduce ArtHOI, an optimization-based framework that integrates and refines priors from multiple foundation models. Our key contribution is a suite of novel methodologies designed to resolve the inherent inaccuracies and physical unreality of these priors. In particular, we introduce an Adaptive Sampling Refinement (ASR) method to optimize object's metric scale and pose for grounding its normalized mesh in world space. Furthermore, we propose a Multimodal Large Language Model (MLLM) guided hand-object alignment method, utilizing contact reasoning information as constraints of hand-object mesh composition optimization. To facilitate a comprehensive evaluation, we also contribute two new datasets, ArtHOI-RGBD and ArtHOI-Wild. Extensive experiments validate the robustness and effectiveness of our ArtHOI across diverse objects and interactions. Project: this https URL.
>
---
#### [new 022] Label-Free Cross-Task LoRA Merging with Null-Space Compression
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出一种无需标签的LoRA合并方法NSC，解决多任务模型融合中跨任务性能不均衡问题，通过压缩空域提升模型效果。**

- **链接: [https://arxiv.org/pdf/2603.26317](https://arxiv.org/pdf/2603.26317)**

> **作者:** Wonyoung Lee; Wooseong Jeong; Kuk-Jin Yoon
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Model merging combines independently fine-tuned checkpoints without joint multi-task training. In the era of foundation-model, fine-tuning with Low-Rank Adaptation (LoRA) is prevalent, making LoRA merging a promising target. Existing approaches can work in homogeneous settings where all target tasks are classification but often fail when tasks span classification and regression. Approaches using entropy-based surrogates do not apply to regression and are costly for large language models due to long token sequences. We introduce Null-Space Compression (NSC) Merging, a label-free, output-agnostic method that sets merge weights from adapter geometry. Our key observation is that during LoRA finetuning the down-projection factor $A$ in $\Delta W = BA$ compresses its null space, and the compression correlates with performance. NSC uses this as an optimization signal for merging that can generalize across classification, regression, and sequence generation. NSC achieves state-of-the-art performance across twenty heterogeneous vision tasks with balanced gains where prior methods overfit subsets of tasks. It also outperforms baselines on six NLI benchmarks and on vision-language evaluations for VQA and image captioning, demonstrating scalability and effectiveness.
>
---
#### [new 023] HINT: Composed Image Retrieval with Dual-path Compositional Contextualized Network
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，解决CIR中忽略上下文信息的问题。提出HINT模型，通过双路径结构增强上下文编码和相似性差异放大，提升检索性能。**

- **链接: [https://arxiv.org/pdf/2603.26341](https://arxiv.org/pdf/2603.26341)**

> **作者:** Mingyu Zhang; Zixu Li; Zhiwei Chen; Zhiheng Fu; Xiaowei Zhu; Jiajia Nie; Yinwei Wei; Yupeng Hu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Composed Image Retrieval (CIR) is a challenging image retrieval paradigm. It aims to retrieve target images from large-scale image databases that are consistent with the modification semantics, based on a multimodal query composed of a reference image and modification text. Although existing methods have made significant progress in cross-modal alignment and feature fusion, a key flaw remains: the neglect of contextual information in discriminating matching samples. However, addressing this limitation is not an easy task due to two challenges: 1) implicit dependencies and 2) the lack of a differential amplification mechanism. To address these challenges, we propose a dual-patH composItional coNtextualized neTwork (HINT), which can perform contextualized encoding and amplify the similarity differences between matching and non-matching samples, thus improving the upper performance of CIR models in complex scenarios. Our HINT model achieves optimal performance on all metrics across two CIR benchmark datasets, demonstrating the superiority of our HINT model. Codes are available at this https URL.
>
---
#### [new 024] Learnable Instance Attention Filtering for Adaptive Detector Distillation
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决知识蒸馏中实例差异性忽略的问题。提出LIAF-KD框架，通过可学习的实例选择器动态调整实例权重，提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2603.26088](https://arxiv.org/pdf/2603.26088)**

> **作者:** Chen Liu; Qizhen Lan; Zhicheng Ding; Xinyu Chu; Qing Tian
>
> **摘要:** As deep vision models grow increasingly complex to achieve higher performance, deployment efficiency has become a critical concern. Knowledge distillation (KD) mitigates this issue by transferring knowledge from large teacher models to compact student models. While many feature-based KD methods rely on spatial filtering to guide distillation, they typically treat all object instances uniformly, ignoring instance-level variability. Moreover, existing attention filtering mechanisms are typically heuristic or teacher-driven, rather than learned with the student. To address these limitations, we propose Learnable Instance Attention Filtering for Adaptive Detector Distillation (LIAF-KD), a novel framework that introduces learnable instance selectors to dynamically evaluate and reweight instance importance during distillation. Notably, the student contributes to this process based on its evolving learning state. Experiments on the KITTI and COCO datasets demonstrate consistent improvements, with a 2% gain on a GFL ResNet-50 student without added complexity, outperforming state-of-the-art methods.
>
---
#### [new 025] Rethinking Token Pruning for Historical Screenshots in GUI Visual Agents: Semantic, Spatial, and Temporal Perspectives
- **分类: cs.CV**

- **简介: 该论文研究GUI视觉代理中的历史截图令牌剪枝问题，旨在提升计算效率。通过分析语义、空间和时间特性，提出有效剪枝策略。**

- **链接: [https://arxiv.org/pdf/2603.26041](https://arxiv.org/pdf/2603.26041)**

> **作者:** Daiqiang Li; Zihao Pan; Zeyu Zhang; Ronghao Chen; Huacan Wang; Honggang Chen; Haiyun Jiang
>
> **摘要:** In recent years, GUI visual agents built upon Multimodal Large Language Models (MLLMs) have demonstrated strong potential in navigation tasks. However, high-resolution GUI screenshots produce a large number of visual tokens, making the direct preservation of complete historical information computationally expensive. In this paper, we conduct an empirical study on token pruning for historical screenshots in GUI scenarios and distill three practical insights that are crucial for designing effective pruning strategies. First, we observe that GUI screenshots exhibit a distinctive foreground-background semantic composition. To probe this property, we apply a simple edge-based separation to partition screenshots into foreground and background regions. Surprisingly, we find that, contrary to the common assumption that background areas have little semantic value, they effectively capture interface-state transitions, thereby providing auxiliary cues for GUI reasoning. Second, compared with carefully designed pruning strategies, random pruning possesses an inherent advantage in preserving spatial structure, enabling better performance under the same computational budget. Finally, we observe that GUI Agents exhibit a recency effect similar to human cognition: by allocating larger token budgets to more recent screenshots and heavily compressing distant ones, we can significantly reduce computational cost while maintaining nearly unchanged performance. These findings offer new insights and practical guidance for the design of efficient GUI visual agents.
>
---
#### [new 026] BEVMAPMATCH: Multimodal BEV Neural Map Matching for Robust Re-Localization of Autonomous Vehicles
- **分类: cs.CV**

- **简介: 该论文属于自主车辆重定位任务，解决GNSS缺失或弱化的环境下的定位问题。通过多模态BEV分割与注意力机制实现精准地图匹配。**

- **链接: [https://arxiv.org/pdf/2603.25963](https://arxiv.org/pdf/2603.25963)**

> **作者:** Shounak Sural; Ragunathan Rajkumar
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Localization in GNSS-denied and GNSS-degraded environments is a challenge for the safe widespread deployment of autonomous vehicles. Such GNSS-challenged environments require alternative methods for robust localization. In this work, we propose BEVMapMatch, a framework for robust vehicle re-localization on a known map without the need for GNSS priors. BEVMapMatch uses a context-aware lidar+camera fusion method to generate multimodal Bird's Eye View (BEV) segmentations around the ego vehicle in both good and adverse weather conditions. Leveraging a search mechanism based on cross-attention, the generated BEV segmentation maps are then used for the retrieval of candidate map patches for map-matching purposes. Finally, BEVMapMatch uses the top retrieved candidate for finer alignment against the generated BEV segmentation, achieving accurate global localization without the need for GNSS. Multiple frames of generated BEV segmentation further improve localization accuracy. Extensive evaluations show that BEVMapMatch outperforms existing methods for re-localization in GNSS-denied and adverse environments, with a Recall@1m of 39.8%, being nearly twice as much as the best performing re-localization baseline. Our code and data will be made available at this https URL.
>
---
#### [new 027] End-to-end Feature Alignment: A Simple CNN with Intrinsic Class Attribution
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决模型可解释性问题。提出FA-CNN，通过端到端特征对齐增强模型的可解释性，使特征图直接体现类别归属。**

- **链接: [https://arxiv.org/pdf/2603.25798](https://arxiv.org/pdf/2603.25798)**

> **作者:** Parniyan Farvardin; David Chapman
>
> **摘要:** We present Feature-Align CNN (FA-CNN), a prototype CNN architecture with intrinsic class attribution through end-to-end feature alignment. Our intuition is that the use of unordered operations such as Linear and Conv2D layers cause unnecessary shuffling and mixing of semantic concepts, thereby making raw feature maps difficult to understand. We introduce two new order preserving layers, the dampened skip connection, and the global average pooling classifier head. These layers force the model to maintain an end-to-end feature alignment from the raw input pixels all the way to final class logits. This end-to-end alignment enhances the interpretability of the model by allowing the raw feature maps to intrinsically exhibit class attribution. We prove theoretically that FA-CNN penultimate feature maps are identical to Grad-CAM saliency maps. Moreover, we prove that these feature maps slowly morph layer-by-layer over network depth, showing the evolution of features through network depth toward penultimate class activations. FA-CNN performs well on benchmark image classification datasets. Moreover, we compare the averaged FA-CNN raw feature maps against Grad-CAM and permutation methods in a percent pixels removed interpretability task. We conclude this work with a discussion and future, including limitations and extensions toward hybrid models.
>
---
#### [new 028] Good Scores, Bad Data: A Metric for Multimodal Coherence
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出多模态一致性评分（MCS），用于评估多模态数据融合质量，解决高任务准确率下数据不一致的问题。通过四个维度衡量，无需人工标注，提升检测敏感性。**

- **链接: [https://arxiv.org/pdf/2603.25924](https://arxiv.org/pdf/2603.25924)**

> **作者:** Vasundra Srinivasan
>
> **备注:** 9 pages, 6 figures, NeurIPS 2024 format
>
> **摘要:** Multimodal AI systems are evaluated by downstream task accuracy, but high accuracy does not mean the underlying data is coherent. A model can score well on Visual Question Answering (VQA) while its inputs contradict each other. We introduce the Multimodal Coherence Score (MCS), a metric that evaluates fusion quality independent of any downstream model. MCS decomposes coherence into four dimensions, identity, spatial, semantic, and decision, with weights learned via Nelder-Mead optimization. We evaluate on 1,000 Visual Genome images using DETR, CLIP, and ViLT, and validate on 150 COCO images with no retraining. Across three fusion architectures, MCS discriminates quality with higher sensitivity than task accuracy alone (Spearman rho = 0.093 vs. 0.071). Perturbation experiments confirm each dimension responds independently to its failure mode with zero cross-talk. MCS is lightweight, requires no human annotation, and tells you not just that something broke, but what broke.
>
---
#### [new 029] IP-Bench: Benchmark for Image Protection Methods in Image-to-Video Generation Scenarios
- **分类: cs.CV**

- **简介: 该论文属于图像保护任务，旨在解决I2V生成场景中保护方法评估缺乏统一基准的问题。提出IP-Bench基准，评估6种方法和5个模型的鲁棒性与迁移能力。**

- **链接: [https://arxiv.org/pdf/2603.26154](https://arxiv.org/pdf/2603.26154)**

> **作者:** Xiaofeng Li; Leyi Sheng; Zhen Sun; Zongmin Zhang; Jiaheng Wei; Xinlei He
>
> **摘要:** With the rapid advancement of image-to-video (I2V) generation models, their potential for misuse in creating malicious content has become a significant concern. For instance, a single image can be exploited to generate a fake video, which can be used to attract attention and gain benefits. This phenomenon is referred to as an I2V generation misuse. Existing image protection methods suffer from the absence of a unified benchmark, leading to an incomplete evaluation framework. Furthermore, these methods have not been systematically assessed in I2V generation scenarios and against preprocessing attacks, which complicates the evaluation of their effectiveness in real-world deployment this http URL address this challenge, we propose IP-Bench (Image Protection Bench), the first systematic benchmark designed to evaluate protection methods in I2V generation scenarios. This benchmark examines 6 representative protection methods and 5 state-of-the-art I2V models. Furthermore, our work systematically evaluates protection methods' robustness with two robustness attack strategies under practical scenarios and analyzes their cross-model & cross-modality transferability. Overall, IP-Bench establishes a systematic, reproducible, and extensible evaluation framework for image protection methods in I2V generation scenarios.
>
---
#### [new 030] OSA: Echocardiography Video Segmentation via Orthogonalized State Update and Anatomical Prior-aware Feature Enhancement
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决超声视频中左心室准确且稳定分割的问题。提出OSA框架，通过约束状态更新和增强解剖先验特征，提升分割精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.26188](https://arxiv.org/pdf/2603.26188)**

> **作者:** Rui Wang; Huisi Wu; Jing Qin
>
> **摘要:** Accurate and temporally consistent segmentation of the left ventricle from echocardiography videos is essential for estimating the ejection fraction and assessing cardiac function. However, modeling spatiotemporal dynamics remains difficult due to severe speckle noise and rapid non-rigid deformations. Existing linear recurrent models offer efficient in-context associative recall for temporal tracking, but rely on unconstrained state updates, which cause progressive singular value decay in the state matrix, a phenomenon known as rank collapse, resulting in anatomical details being overwhelmed by noise. To address this, we propose OSA, a framework that constrains the state evolution on the Stiefel manifold. We introduce the Orthogonalized State Update (OSU) mechanism, which formulates the memory evolution as Euclidean projected gradient descent on the Stiefel manifold to prevent rank collapse and maintain stable temporal transitions. Furthermore, an Anatomical Prior-aware Feature Enhancement module explicitly separates anatomical structures from speckle noise through a physics-driven process, providing the temporal tracker with noise-resilient structural cues. Comprehensive experiments on the CAMUS and EchoNet-Dynamic datasets show that OSA achieves state-of-the-art segmentation accuracy and temporal stability, while maintaining real-time inference efficiency for clinical deployment. Codes are available at this https URL.
>
---
#### [new 031] TaxaAdapter: Vision Taxonomy Models are Key to Fine-grained Image Generation over the Tree of Life
- **分类: cs.CV**

- **简介: 该论文属于细粒度图像生成任务，旨在解决物种识别困难的问题。通过引入视觉分类模型，提升生成图像的物种准确性与形态一致性。**

- **链接: [https://arxiv.org/pdf/2603.26128](https://arxiv.org/pdf/2603.26128)**

> **作者:** Mridul Khurana; Amin Karimi Monsefi; Justin Lee; Medha Sawhney; David Carlyn; Julia Chae; Jianyang Gu; Rajiv Ramnath; Sara Beery; Wei-Lun Chao; Anuj Karpatne; Cheng Zhang
>
> **摘要:** Accurately generating images across the Tree of Life is difficult: there are over 10M distinct species on Earth, many of which differ only by subtle visual traits. Despite the remarkable progress in text-to-image synthesis, existing models often fail to capture the fine-grained visual cues that define species identity, even when their outputs appear photo-realistic. To this end, we propose TaxaAdapter, a simple and lightweight approach that incorporates Vision Taxonomy Models (VTMs) such as BioCLIP to guide fine-grained species generation. Our method injects VTM embeddings into a frozen text-to-image diffusion model, improving species-level fidelity while preserving flexible text control over attributes such as pose, style, and background. Extensive experiments demonstrate that TaxaAdapter consistently improves morphology fidelity and species-identity accuracy over strong baselines, with a cleaner architecture and training recipe. To better evaluate these improvements, we also introduce a multimodal Large Language Model-based metric that summarizes trait-level descriptions from generated and real images, providing a more interpretable measure of morphological consistency. Beyond this, we observe that TaxaAdapter exhibits strong generalization capabilities, enabling species synthesis in challenging regimes such as few-shot species with only a handful of training images and even species unseen during training. Overall, our results highlight that VTMs are a key ingredient for scalable, fine-grained species generation.
>
---
#### [new 032] Make Geometry Matter for Spatial Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型的 spatial reasoning 任务，旨在解决模型过度依赖2D视觉线索而忽视几何信息的问题。提出GeoSR框架，通过几何引导机制提升空间推理性能。**

- **链接: [https://arxiv.org/pdf/2603.26639](https://arxiv.org/pdf/2603.26639)**

> **作者:** Shihua Zhang; Qiuhong Shen; Shizun Wang; Tianbo Pan; Xinchao Wang
>
> **摘要:** Empowered by large-scale training, vision-language models (VLMs) achieve strong image and video understanding, yet their ability to perform spatial reasoning in both static scenes and dynamic videos remains limited. Recent advances try to handle this limitation by injecting geometry tokens from pretrained 3D foundation models into VLMs. Nevertheless, we observe that naive token fusion followed by standard fine-tuning in this line of work often leaves such geometric cues underutilized for spatial reasoning, as VLMs tend to rely heavily on 2D visual cues. In this paper, we propose GeoSR, a framework designed to make geometry matter by encouraging VLMs to actively reason with geometry tokens. GeoSR introduces two key components: (1) Geometry-Unleashing Masking, which strategically masks portions of 2D vision tokens during training to weaken non-geometric shortcuts and force the model to consult geometry tokens for spatial reasoning; and (2) Geometry-Guided Fusion, a gated routing mechanism that adaptively amplifies geometry token contributions in regions where geometric evidence is critical. Together, these designs unleash the potential of geometry tokens for spatial reasoning tasks. Extensive experiments on both static and dynamic spatial reasoning benchmarks demonstrate that GeoSR consistently outperforms prior methods and establishes new state-of-the-art performance by effectively leveraging geometric information. The project page is available at this https URL.
>
---
#### [new 033] Collision-Aware Vision-Language Learning for End-to-End Driving with Multimodal Infraction Datasets
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于自动驾驶任务，旨在解决高违规率问题。通过构建多模态数据集并提出VLAAD模型，提升碰撞预测能力，增强驾驶表现。**

- **链接: [https://arxiv.org/pdf/2603.25946](https://arxiv.org/pdf/2603.25946)**

> **作者:** Alex Koran; Dimitrios Sinodinos; Hadi Hojjati; Takuya Nanri; Fangge Chen; Narges Armanfard
>
> **备注:** 33 pages, 11 figures
>
> **摘要:** High infraction rates remain the primary bottleneck for end-to-end (E2E) autonomous driving, as evidenced by the low driving scores on the CARLA Leaderboard. Despite collision-related infractions being the dominant failure mode in closed-loop evaluations, collision-aware representation learning has received limited attention. To address this gap, we first develop a Video-Language-Augmented Anomaly Detector (VLAAD), leveraging a Multiple Instance Learning (MIL) formulation to obtain stable, temporally localized collision signals for proactive prediction. To transition these capabilities into closed-loop simulations, we must overcome the limitations of existing simulator datasets, which lack multimodality and are frequently restricted to simple intersection scenarios. Therefore, we introduce CARLA-Collide, a large-scale multimodal dataset capturing realistic collision events across highly diverse road networks. Trained on this diverse simulator data, VLAAD serves as a collision-aware plug-in module that can be seamlessly integrated into existing E2E driving models. By integrating our module into a pretrained TransFuser++ agent, we demonstrate a 14.12% relative increase in driving score with minimal fine-tuning. Beyond closed-loop evaluation, we further assess the generalization capability of VLAAD in an open-loop setting using real-world driving data. To support this analysis, we introduce Real-Collide, a multimodal dataset of diverse dashcam videos paired with semantically rich annotations for collision detection and prediction. On this benchmark, despite containing only 0.6B parameters, VLAAD outperforms a multi-billion-parameter vision-language model, achieving a 23.3% improvement in AUC.
>
---
#### [new 034] GLASS: Geometry-aware Local Alignment and Structure Synchronization Network for 2D-3D Registration
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于2D-3D图像配准任务，旨在解决重复场景中匹配错误和结构不一致问题。提出LGE和GDC模块，提升几何一致性与匹配精度。**

- **链接: [https://arxiv.org/pdf/2603.26262](https://arxiv.org/pdf/2603.26262)**

> **作者:** Zhixin Cheng; Jiacheng Deng; Xinjun Li; Bohao Liao; Li Liu; Xiaotian Yin; Baoqun Yin; Tianzhu Zhang
>
> **备注:** Accepted by IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** Image-to-point cloud registration methods typically follow a coarse-to-fine pipeline, extracting patch-level correspondences and refining them into dense pixel-to-point matches. However, in scenes with repetitive patterns, images often lack sufficient 3D structural cues and alignment with point clouds, leading to incorrect matches. Moreover, prior methods usually overlook structural consistency, limiting the full exploitation of correspondences. To address these issues, we propose two novel modules: the Local Geometry Enhancement (LGE) module and the Graph Distribution Consistency (GDC) module. LGE enhances both image and point cloud features with normal vectors, injecting geometric structure into image features to reduce mismatches. GDC constructs a graph from matched points to update features and explicitly constrain similarity distributions. Extensive experiments and ablations on two benchmarks, RGB-D Scenes v2 and 7-Scenes, demonstrate that our approach achieves state-of-the-art performance in image-to-point cloud registration.
>
---
#### [new 035] Scene Grounding In the Wild
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决无重叠图像下场景重建不一致的问题。通过将局部重建与参考模型对齐，实现全局一致的场景重构。**

- **链接: [https://arxiv.org/pdf/2603.26584](https://arxiv.org/pdf/2603.26584)**

> **作者:** Tamir Cohen; Leo Segre; Shay Shomer-Chai; Shai Avidan; Hadar Averbuch-Elor
>
> **备注:** Project page at this https URL
>
> **摘要:** Reconstructing accurate 3D models of large-scale real-world scenes from unstructured, in-the-wild imagery remains a core challenge in computer vision, especially when the input views have little or no overlap. In such cases, existing reconstruction pipelines often produce multiple disconnected partial reconstructions or erroneously merge non-overlapping regions into overlapping geometry. In this work, we propose a framework that grounds each partial reconstruction to a complete reference model of the scene, enabling globally consistent alignment even in the absence of visual overlap. We obtain reference models from dense, geospatially accurate pseudo-synthetic renderings derived from Google Earth Studio. These renderings provide full scene coverage but differ substantially in appearance from real-world photographs. Our key insight is that, despite this significant domain gap, both domains share the same underlying scene semantics. We represent the reference model using 3D Gaussian Splatting, augmenting each Gaussian with semantic features, and formulate alignment as an inverse feature-based optimization scheme that estimates a global 6DoF pose and scale while keeping the reference model fixed. Furthermore, we introduce the WikiEarth dataset, which registers existing partial 3D reconstructions with pseudo-synthetic reference models. We demonstrate that our approach consistently improves global alignment when initialized with various classical and learning-based pipelines, while mitigating failure modes of state-of-the-art end-to-end models. All code and data will be released.
>
---
#### [new 036] GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，解决透明物体（如玻璃）的建模问题。通过分解高斯表示，分离反射与透射辐射，实现场景级透明度建模。**

- **链接: [https://arxiv.org/pdf/2603.26181](https://arxiv.org/pdf/2603.26181)**

> **作者:** Youngju Na; Jaeseong Yun; Soohyun Ryu; Hyunsu Kim; Sung-Eui Yoon; Suyong Yeon
>
> **备注:** CVPR 2026, Project page: this https URL
>
> **摘要:** While 3D Gaussian splatting has emerged as a powerful paradigm, it fundamentally fails to model transparency such as glass panels. The core challenge lies in decoupling the intertwined radiance contributions from transparent interfaces and the transmitted geometry observed through the glass. We present GLINT, a framework that models scene-scale transparency through explicit decomposed Gaussian representation. GLINT reconstructs the primary interface and models reflected and transmitted radiance separately, enabling consistent radiance transport. During optimization, GLINT bootstraps transparency localization from geometry-separation cues induced by the decomposition, together with geometry and material priors from a pre-trained video relighting model. Extensive experiments demonstrate consistent improvements over prior methods for reconstructing complex transparent scenes.
>
---
#### [new 037] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Assistants
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态模型公平性任务，旨在解决MLLM在不同群体间表现不均的问题。通过参数高效微调方法FairLLaVA，减少视觉指令中的群体差异，提升公平性和临床性能。**

- **链接: [https://arxiv.org/pdf/2603.26008](https://arxiv.org/pdf/2603.26008)**

> **作者:** Mahesh Bhosale; Abdul Wasi; Shantam Srivastava; Shifa Latif; Tianyu Luan; Mingchen Gao; David Doermann; Xuan Gong
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** While powerful in image-conditioned generation, multimodal large language models (MLLMs) can display uneven performance across demographic groups, highlighting fairness risks. In safety-critical clinical settings, such disparities risk producing unequal diagnostic narratives and eroding trust in AI-assisted decision-making. While fairness has been studied extensively in vision-only and language-only models, its impact on MLLMs remains largely underexplored. To address these biases, we introduce FairLLaVA, a parameter-efficient fine-tuning method that mitigates group disparities in visual instruction tuning without compromising overall performance. By minimizing the mutual information between target attributes, FairLLaVA regularizes the model's representations to be demographic-invariant. The method can be incorporated as a lightweight plug-in, maintaining efficiency with low-rank adapter fine-tuning, and provides an architecture-agnostic approach to fair visual instruction following. Extensive experiments on large-scale chest radiology report generation and dermoscopy visual question answering benchmarks show that FairLLaVA consistently reduces inter-group disparities while improving both equity-scaled clinical performance and natural language generation quality across diverse medical imaging modalities. Code can be accessed at this https URL.
>
---
#### [new 038] Beyond MACs: Hardware Efficient Architecture Design for Vision Backbones
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在提升视觉主干网络的效率。针对传统MACs指标的不足，提出LowFormer架构，优化执行效率并提升性能。**

- **链接: [https://arxiv.org/pdf/2603.26551](https://arxiv.org/pdf/2603.26551)**

> **作者:** Moritz Nottebaum; Matteo Dunnhofer; Christian Micheloni
>
> **备注:** Submitted to International Journal of Computer Vision (IJCV); currently under minor revision
>
> **摘要:** Vision backbone networks play a central role in modern computer vision. Enhancing their efficiency directly benefits a wide range of downstream applications. To measure efficiency, many publications rely on MACs (Multiply Accumulate operations) as a predictor of execution time. In this paper, we experimentally demonstrate the shortcomings of such a metric, especially in the context of edge devices. By contrasting the MAC count and execution time of common architectural design elements, we identify key factors for efficient execution and provide insights to optimize backbone design. Based on these insights, we present LowFormer, a novel vision backbone family. LowFormer features a streamlined macro and micro design that includes Lowtention, a lightweight alternative to Multi-Head Self-Attention. Lowtention not only proves more efficient, but also enables superior results on ImageNet. Additionally, we present an edge GPU version of LowFormer, that can further improve upon its baseline's speed on edge GPU and desktop GPU. We demonstrate LowFormer's wide applicability by evaluating it on smaller image classification datasets, as well as adapting it to several downstream tasks, such as object detection, semantic segmentation, image retrieval, and visual object tracking. LowFormer models consistently achieve remarkable speed-ups across various hardware platforms compared to recent state-of-the-art backbones. Code and models are available at this https URL.
>
---
#### [new 039] Do All Vision Transformers Need Registers? A Cross-Architectural Reassessment
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉Transformer研究任务，旨在解决注意力图中的伪影问题。通过复现与扩展实验，验证并修正了前人关于“寄存器”必要性的结论。**

- **链接: [https://arxiv.org/pdf/2603.25803](https://arxiv.org/pdf/2603.25803)**

> **作者:** Spiros Baxevanakis; Platon Karageorgis; Ioannis Dravilas; Konrad Szewczyk
>
> **备注:** Preprint. Submitted to Transactions on Machine Learning Research (TMLR). 26 pages, 17 figures
>
> **摘要:** Training Vision Transformers (ViTs) presents significant challenges, one of which is the emergence of artifacts in attention maps, hindering their interpretability. Darcet et al. (2024) investigated this phenomenon and attributed it to the need of ViTs to store global information beyond the [CLS] token. They proposed a novel solution involving the addition of empty input tokens, named registers, which successfully eliminate artifacts and improve the clarity of attention maps. In this work, we reproduce the findings of Darcet et al. (2024) and evaluate the generalizability of their claims across multiple models, including DINO, DINOv2, OpenCLIP, and DeiT3. While we confirm the validity of several of their key claims, our results reveal that some claims do not extend universally to other models. Additionally, we explore the impact of model size, extending their findings to smaller models. Finally, we untie terminology inconsistencies found in the original paper and explain their impact when generalizing to a wider range of models.
>
---
#### [new 040] From Pixels to Privacy: Temporally Consistent Video Anonymization via Token Pruning for Privacy Preserving Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于视频隐私保护任务，解决视频中敏感信息泄露问题。通过注意力机制分离动作与隐私特征，选择性删除隐私内容，保障视频分析隐私安全。**

- **链接: [https://arxiv.org/pdf/2603.26336](https://arxiv.org/pdf/2603.26336)**

> **作者:** Nazia Aslam; Abhisek Ray; Joakim Bruslund Haurum; Lukas Esterle; Kamal Nasrollahi
>
> **备注:** 10 pages, CVPR paper
>
> **摘要:** Recent advances in large-scale video models have significantly improved video understanding across domains such as surveillance, healthcare, and entertainment. However, these models also amplify privacy risks by encoding sensitive attributes, including facial identity, race, and gender. While image anonymization has been extensively studied, video anonymization remains relatively underexplored, even though modern video models can leverage spatiotemporal motion patterns as biometric identifiers. To address this challenge, we propose a novel attention-driven spatiotemporal video anonymization framework based on systematic disentanglement of utility and privacy features. Our key insight is that attention mechanisms in Vision Transformers (ViTs) can be explicitly structured to separate action-relevant information from privacy-sensitive content. Building on this insight, we introduce two task-specific classification tokens, an action CLS token and a privacy CLS token, that learn complementary representations within a shared Transformer backbone. We contrast their attention distributions to compute a utility-privacy score for each spatiotemporal tubelet, and keep the top-k tubelets with the highest scores. This selectively prunes tubelets dominated by privacy cues while preserving those most critical for action recognition. Extensive experiments demonstrate that our approach maintains action recognition performance comparable to models trained on raw videos, while substantially reducing privacy leakage. These results indicate that attention-driven spatiotemporal pruning offers an effective and principled solution for privacy-preserving video analytics.
>
---
#### [new 041] SHANDS: A Multi-View Dataset and Benchmark for Surgical Hand-Gesture and Error Recognition Toward Medical Training
- **分类: cs.CV**

- **简介: 该论文提出SHands数据集，用于手术手势与错误识别，解决医疗培训中评估成本高、难扩展的问题。通过多视角视频和详细标注，支持AI模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2603.26400](https://arxiv.org/pdf/2603.26400)**

> **作者:** Le Ma; Thiago Freitas dos Santos; Nadia Magnenat-Thalmann; Katarzyna Wac
>
> **摘要:** In surgical training for medical students, proficiency development relies on expert-led skill assessment, which is costly, time-limited, difficult to scale, and its expertise remains confined to institutions with available specialists. Automated AI-based assessment offers a viable alternative, but progress is constrained by the lack of datasets containing realistic trainee errors and the multi-view variability needed to train robust computer vision approaches. To address this gap, we present Surgical-Hands (SHands), a large-scale multi-view video dataset for surgical hand-gesture and error recognition for medical training. \textsc{SHands} captures linear incision and suturing using five RGB cameras from complementary viewpoints, performed by 52 participants (20 experts and 32 trainees), each completing three standardized trials per procedure. The videos are annotated at the frame level with 15 gesture primitives and include a validated taxonomy of 8 trainee error types, enabling both gesture recognition and error detection. We further define standardized evaluation protocols for single-view, multi-view, and cross-view generalization, and benchmark state-of-the-art deep learning models on the dataset. SHands is publicly released to support the development of robust and scalable AI systems for surgical training grounded in clinically curated domain knowledge.
>
---
#### [new 042] Seeing Through Smoke: Surgical Desmoking for Improved Visual Perception
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决手术中烟雾干扰视觉的问题。通过构建深度学习模型和数据集，实现烟雾去除，提升手术视觉效果。**

- **链接: [https://arxiv.org/pdf/2603.25867](https://arxiv.org/pdf/2603.25867)**

> **作者:** Jingpei Lu; Fengyi Jiang; Xiaorui Zhang; Lingbo Jin; Omid Mohareri
>
> **备注:** 8 pages, 4 figures, 3 tables
>
> **摘要:** Minimally invasive and robot-assisted surgery relies heavily on endoscopic imaging, yet surgical smoke produced by electrocautery and vessel-sealing instruments can severely degrade visual perception and hinder vision-based functionalities. We present a transformer-based surgical desmoking model with a physics-inspired desmoking head that jointly predicts smoke-free image and corresponding smoke map. To address the scarcity of paired smoky-to-smoke-free training data, we develop a synthetic data generation pipeline that blends artificial smoke patterns with real endoscopic images, yielding over 80,000 paired samples for supervised training. We further curate, to our knowledge, the largest paired surgical smoke dataset to date, comprising 5,817 image pairs captured with the da Vinci robotic surgical system, enabling benchmarking on high-resolution endoscopic images. Extensive experiments on both a public benchmark and our dataset demonstrate state-of-the-art performance in image reconstruction compared to existing dehazing and desmoking approaches. We also assess the impact of desmoking on downstream stereo depth estimation and instrument segmentation, highlighting both the potential benefits and current limitations of digital smoke removal methods.
>
---
#### [new 043] Real-Time Branch-to-Tool Distance Estimation for Autonomous UAV Pruning: Benchmarking Five DEFOM-Stereo Variants from Simulation to Jetson Deployment
- **分类: cs.CV**

- **简介: 该论文研究自主无人机修剪树木中的实时分支到工具距离估计问题，通过优化DEFOM-Stereo模型实现安全控制。**

- **链接: [https://arxiv.org/pdf/2603.26250](https://arxiv.org/pdf/2603.26250)**

> **作者:** Yida Lin; Bing Xue; Mengjie Zhang; Sam Schofield; Richard Green
>
> **摘要:** Autonomous tree pruning with unmanned aerial vehicles (UAVs) is a safety-critical real-world task: the onboard perception system must estimate the metric distance from a cutting tool to thin tree branches in real time so that the UAV can approach, align, and actuate the pruner without collision. We address this problem by training five variants of DEFOM-Stereo - a recent foundation-model-based stereo matcher - on a task-specific synthetic dataset and deploying the checkpoints on an NVIDIA Jetson Orin Super 16 GB. The training corpus is built in Unreal Engine 5 with a simulated ZED Mini stereo camera capturing 5,520 stereo pairs across 115 tree instances from three viewpoints at 2m distance; dense EXR depth maps provide exact, spatially complete supervision for thin branches. On the synthetic test set, DEFOM-Stereo ViT-S achieves the best depth-domain accuracy (EPE 1.74 px, D1-all 5.81%, delta-1 95.90%, depth MAE 23.40 cm) but its Jetson inference speed of ~2.2 FPS (~450 ms per frame) remains too slow for responsive closed-loop tool control. A newly introduced balanced variant, DEFOM-PrunePlus (~21M backbone, ~3.3 FPS on Jetson), offers the best deployable accuracy-speed trade-off (EPE 5.87 px, depth MAE 64.26 cm, delta-1 87.59%): its frame rate is sufficient for real-time guidance and its depth accuracy supports safe branch approach planning at the 2m operating range. The lightweight DEFOM-PruneStereo (~6.9 FPS) and DEFOM-PruneNano (~8.5 FPS) run fast but sacrifice substantial accuracy (depth MAE > 57 cm), making estimates too unreliable for safe actuation. Zero-shot inference on real photographs confirms that full-capacity models preserve branch geometry, validating the sim-to-real transfer. We conclude that DEFOM-PrunePlus provides the most practical accuracy-latency balance for onboard distance estimation, while ViT-S serves as the reference for future hardware.
>
---
#### [new 044] MPDiT: Multi-Patch Global-to-Local Transformer Architecture For Efficient Flow Matching and Diffusion Model
- **分类: cs.CV**

- **简介: 该论文提出MPDiT，一种多块全局到局部Transformer架构，用于高效扩散模型。解决DiT计算冗余问题，通过分层块设计降低计算量，提升训练效率。**

- **链接: [https://arxiv.org/pdf/2603.26357](https://arxiv.org/pdf/2603.26357)**

> **作者:** Quan Dao; Dimitris Metaxas
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Transformer architectures, particularly Diffusion Transformers (DiTs), have become widely used in diffusion and flow-matching models due to their strong performance compared to convolutional UNets. However, the isotropic design of DiTs processes the same number of patchified tokens in every block, leading to relatively heavy computation during training process. In this work, we introduce a multi-patch transformer design in which early blocks operate on larger patches to capture coarse global context, while later blocks use smaller patches to refine local details. This hierarchical design could reduces computational cost by up to 50\% in GFLOPs while achieving good generative performance. In addition, we also propose improved designs for time and class embeddings that accelerate training convergence. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our architectural choices. Code is released at \url{this https URL}
>
---
#### [new 045] Fus3D: Decoding Consolidated 3D Geometry from Feed-forward Geometry Transformer Latents
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决从无标定图像中快速生成完整SDF的问题。通过直接从几何Transformer特征提取3D信息，提升重建精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.25827](https://arxiv.org/pdf/2603.25827)**

> **作者:** Laura Fink; Linus Franke; George Kopanas; Marc Stamminger; Peter Hedman
>
> **摘要:** We propose a feed-forward method for dense Signed Distance Field (SDF) regression from unstructured image collections in less than three seconds, without camera calibration or post-hoc fusion. Our key insight is that the intermediate feature space of pretrained multi-view feed-forward geometry transformers already encodes a powerful joint world representation; yet, existing pipelines discard it, routing features through per-view prediction heads before assembling 3D geometry post-hoc, which discards valuable completeness information and accumulates inaccuracies. We instead perform 3D extraction directly from geometry transformer features via learned volumetric extraction: voxelized canonical embeddings that progressively absorb multi-view geometry information through interleaved cross- and self-attention into a structured volumetric latent grid. A simple convolutional decoder then maps this grid to a dense SDF. We additionally propose a scalable, validity-aware supervision scheme directly using SDFs derived from depth maps or 3D assets, tackling practical issues like non-watertight meshes. Our approach yields complete and well-defined distance values across sparse- and dense-view settings and demonstrates geometrically plausible completions. Code and further material can be found at this https URL.
>
---
#### [new 046] GazeQwen: Lightweight Gaze-Conditioned LLM Modulation for Streaming Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GazeQwen，解决视频理解中未能有效利用眼动信息的问题。通过参数高效方法，将眼动信息注入大语言模型，提升视频理解性能。**

- **链接: [https://arxiv.org/pdf/2603.25841](https://arxiv.org/pdf/2603.25841)**

> **作者:** Trong Thang Pham; Hien Nguyen; Ngan Le
>
> **摘要:** Current multimodal large language models (MLLMs) cannot effectively utilize eye-gaze information for video understanding, even when gaze cues are supplied via visual overlays or text descriptions. We introduce GazeQwen, a parameter efficient approach that equips an open-source MLLM with gaze awareness through hidden-state modulation. At its core is a compact gaze resampler (~1-5 M trainable parameters) that encodes V-JEPA 2.1 video features together with fixation-derived positional encodings and produces additive residuals injected into selected LLM decoder layers via forward hooks. An optional second training stage adds low-rank adapters (LoRA) to the LLM for tighter integration. Evaluated on all 10 tasks of the StreamGaze benchmark, GazeQwen reaches 63.9% accuracy, a +16.1 point gain over the same Qwen2.5-VL-7B backbone with gaze as visual prompts and +10.5 points over GPT-4o, the highest score among all open-source and proprietary models tested. These results suggest that learning where to inject gaze within an LLM is more effective than scaling model size or engineering better prompts. All code and checkpoints are available at this https URL .
>
---
#### [new 047] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D语义分割任务，解决开放词汇下的分割问题。通过引入几何-语义一致性机制，提升分割精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26260](https://arxiv.org/pdf/2603.26260)**

> **作者:** Xujing Tao; Chuxin Wang; Yubo Ai; Zhixin Cheng; Zhuoyuan Li; Liangsheng Liu; Yujia Chen; Xinjun Li; Qiao Li; Wenfei Yang; Tianzhu Zhang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Open-vocabulary 3D semantic segmentation aims to segment arbitrary categories beyond the training set. Existing methods predominantly rely on distilling knowledge from 2D open-vocabulary models. However, aligning 3D features to the 2D representation space restricts intrinsic 3D geometric learning and inherits errors from 2D predictions. To address these limitations, we propose GeoGuide, a novel framework that leverages pretrained 3D models to integrate hierarchical geometry-semantic consistency for open-vocabulary 3D segmentation. Specifically, we introduce an Uncertainty-based Superpoint Distillation module to fuse geometric and semantic features for estimating per-point uncertainty, adaptively weighting 2D features within superpoints to suppress noise while preserving discriminative information to enhance local semantic consistency. Furthermore, our Instance-level Mask Reconstruction module leverages geometric priors to enforce semantic consistency within instances by reconstructing complete instance masks. Additionally, our Inter-Instance Relation Consistency module aligns geometric and semantic similarity matrices to calibrate cross-instance consistency for same-category objects, mitigating viewpoint-induced semantic drift. Extensive experiments on ScanNet v2, Matterport3D, and nuScenes demonstrate the superior performance of GeoGuide.
>
---
#### [new 048] Conditional Diffusion for 3D CT Volume Reconstruction from 2D X-rays
- **分类: cs.CV**

- **简介: 该论文属于3D CT重建任务，旨在从2D X-ray生成高质量3D CT体积，解决临床通用性差的问题。提出AXON框架，结合扩散模型与ControlNet，提升重建精度与分辨率。**

- **链接: [https://arxiv.org/pdf/2603.26509](https://arxiv.org/pdf/2603.26509)**

> **作者:** Martin Rath; Morteza Ghahremani; Yitong Li; Ashkan Taghipour; Marcus Makowski; Christian Wachinger
>
> **摘要:** Computed tomography (CT) provides rich 3D anatomical details but is often constrained by high radiation exposure, substantial costs, and limited availability. While standard chest X-rays are cost-effective and widely accessible, they only provide 2D projections with limited pathological information. Reconstructing 3D CT volumes from 2D X-rays offers a transformative solution to increase diagnostic accessibility, yet existing methods predominantly rely on synthetic X-ray projections, limiting clinical generalization. In this work, we propose AXON, a multi-stage diffusion-based framework that reconstructs high-fidelity 3D CT volumes directly from real X-rays. AXON employs a coarse-to-fine strategy, with a Brownian Bridge diffusion model-based initial stage for global structural synthesis, followed by a ControlNet-based refinement stage for local intensity optimization. It also supports bi-planar X-ray input to mitigate depth ambiguities inherent in 2D-to-3D reconstruction. A super-resolution network is integrated to upscale the generated volumes to achieve diagnostic-grade resolution. Evaluations on both public and external datasets demonstrate that AXON significantly outperforms state-of-the-art baselines, achieving a 11.9% improvement in PSNR and a 11.0% increase in SSIM with robust generalizability across disparate clinical distributions. Our code is available at this https URL.
>
---
#### [new 049] A Survey of OCR Evaluation Methods and Metrics and the Invisibility of Historical Documents
- **分类: cs.CV; cs.DL**

- **简介: 该论文属于OCR评估任务，探讨现有评估方法对历史文档的忽视问题，指出黑人报纸等文献在训练数据和基准中缺失，导致结构性不透明和代表性损害。**

- **链接: [https://arxiv.org/pdf/2603.25761](https://arxiv.org/pdf/2603.25761)**

> **作者:** Fitsum Sileshi Beyene; Christopher L. Dancy
>
> **备注:** This manuscript is the author's submitted version to the ACM Conference on Fairness, Accountability, and Transparency (FAccT 2026). Please cite the final published version via ACM Digital Library when available
>
> **摘要:** Optical character recognition (OCR) and document understanding systems increasingly rely on large vision and vision-language models, yet evaluation remains centered on modern, Western, and institutional documents. This emphasis masks system behavior in historical and marginalized archives, where layout, typography, and material degradation shape interpretation. This study examines how OCR and document understanding systems are evaluated, with particular attention to Black historical newspapers. We review OCR and document understanding papers, as well as benchmark datasets, which are published between 2006 and 2025 using the PRISMA framework. We look into how the studies report training data, benchmark design, and evaluation metrics for vision transformer and multimodal OCR systems. During the review, we found that Black newspapers and other community-produced historical documents rarely appear in reported training data or evaluation benchmarks. Most evaluations emphasize character accuracy and task success on modern layouts. They rarely capture structural failures common in historical newspapers, including column collapse, typographic errors, and hallucinated text. To put these findings into perspective, we use previous empirical studies and archival statistics from significant Black press collections to show how evaluation gaps lead to structural invisibility and representational harm. We propose that these gaps occur due to organizational (meso) and institutional (macro) behaviors and structure, shaped by benchmark incentives and data governance decisions.
>
---
#### [new 050] HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出HandVQA，用于评估和提升视觉语言模型在手部精细空间推理方面的能力。针对当前模型在手部结构理解上的不足，通过构建大规模基准测试，验证并改进模型表现。**

- **链接: [https://arxiv.org/pdf/2603.26362](https://arxiv.org/pdf/2603.26362)**

> **作者:** MD Khalequzzaman Chowdhury Sayem; Mubarrat Tajoar Chowdhury; Yihalem Yimolal Tiruneh; Muneeb A. Khan; Muhammad Salman Ali; Binod Bhattarai; Seungryul Baek
>
> **备注:** Accepted in CVPR 2026; Project page, code, and dataset: this https URL
>
> **摘要:** Understanding the fine-grained articulation of human hands is critical in high-stakes settings such as robot-assisted surgery, chip manufacturing, and AR/VR-based human-AI interaction. Despite achieving near-human performance on general vision-language benchmarks, current vision-language models (VLMs) struggle with fine-grained spatial reasoning, especially in interpreting complex and articulated hand poses. We introduce HandVQA, a large-scale diagnostic benchmark designed to evaluate VLMs' understanding of detailed hand anatomy through visual question answering. Built upon high-quality 3D hand datasets (FreiHAND, InterHand2.6M, FPHA), our benchmark includes over 1.6M controlled multiple-choice questions that probe spatial relationships between hand joints, such as angles, distances, and relative positions. We evaluate several state-of-the-art VLMs (LLaVA, DeepSeek and Qwen-VL) in both base and fine-tuned settings, using lightweight fine-tuning via LoRA. Our findings reveal systematic limitations in current models, including hallucinated finger parts, incorrect geometric interpretations, and poor generalization. HandVQA not only exposes these critical reasoning gaps but provides a validated path to improvement. We demonstrate that the 3D-grounded spatial knowledge learned from our benchmark transfers in a zero-shot setting, significantly improving accuracy of model on novel downstream tasks like hand gesture recognition (+10.33%) and hand-object interaction (+2.63%).
>
---
#### [new 051] Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization
- **分类: cs.CV**

- **简介: 该论文属于文本到图像模型验证任务，旨在解决第三方平台虚假宣称使用官方模型的问题。通过分析模型在语义边界上的输出稳定性，提出BPO方法实现高效验证。**

- **链接: [https://arxiv.org/pdf/2603.26328](https://arxiv.org/pdf/2603.26328)**

> **作者:** Zidong Zhao; Yihao Huang; Qing Guo; Tianlin Li; Anran Li; Kailong Wang; Jin Song Dong; Geguang Pu
>
> **备注:** Accepted to CVPR 2026 (Findings)
>
> **摘要:** As Text-to-Image (T2I) generation becomes widespread, third-party platforms increasingly integrate multiple model APIs for convenient image creation. However, false claims of using official models can mislead users and harm model owners' reputations, making model verification essential to confirm whether an API's underlying model matches its claim. Existing methods address this by using verification prompts generated by official model owners, but the generation relies on multiple reference models for optimization, leading to high computational cost and sensitivity to model selection. To address this problem, we propose a reference-free T2I model verification method called Boundary-aware Prompt Optimization (BPO). It directly explores the intrinsic characteristics of the target model. The key insight is that although different T2I models produce similar outputs for normal prompts, their semantic boundaries in the embedding space (transition zones between two concepts such as "corgi" and "bagel") are distinct. Prompts near these boundaries generate unstable outputs (e.g., sometimes a corgi and sometimes a bagel) on the target model but remain stable on other models. By identifying such boundary-adjacent prompts, BPO captures model-specific behaviors that serve as reliable verification cues for distinguishing T2I models. Experiments on five T2I models and four baselines demonstrate that BPO achieves superior verification accuracy.
>
---
#### [new 052] Learnable Quantum Efficiency Filters for Urban Hyperspectral Segmentation
- **分类: cs.CV**

- **简介: 该论文属于城市高光谱分割任务，旨在解决高维数据的高效学习与解释问题。提出LQE方法，通过物理约束实现可学习的光谱降维，提升模型性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.26528](https://arxiv.org/pdf/2603.26528)**

> **作者:** Imad Ali Shah; Jiarong Li; Ethan Delaney; Enda Ward; Martin Glavin; Edward Jones; Brian Deegan
>
> **摘要:** Hyperspectral sensing provides rich spectral information for scene understanding in urban driving, but its high dimensionality poses challenges for interpretation and efficient learning. We introduce Learnable Quantum Efficiency (LQE), a physics-inspired, interpretable dimensionality reduction (DR) method that parameterizes smooth high-order spectral response functions that emulate plausible sensor quantum efficiency curves. Unlike conventional methods or unconstrained learnable layers, LQE enforces physically motivated constraints, including a single dominant peak, smooth responses, and bounded bandwidth. This formulation yields a compact spectral representation that preserves discriminative information while remaining fully differentiable and end-to-end trainable within semantic segmentation models (SSMs). We conduct systematic evaluations across three publicly available multi-class hyperspectral urban driving datasets, comparing LQE against six conventional and seven learnable baseline DR methods across six SSMs. Averaged across all SSMs and configurations, LQE achieves the highest average mIoU, improving over conventional methods by 2.45\%, 0.45\%, and 1.04\%, and over learnable methods by 1.18\%, 1.56\%, and 0.81\% on HyKo, HSI-Drive, and Hyperspectral City, respectively. LQE maintains strong parameter efficiency (12--36 parameters compared to 51--22K for competing learnable approaches) and competitive inference latency. Ablation studies show that low-order configurations are optimal, while the learned spectral filters converge to dataset-intrinsic wavelength patterns. These results demonstrate that physics-informed spectral learning can improve both performance and interpretability, providing a principled bridge between hyperspectral perception and data-driven multispectral sensor design for automotive vision systems.
>
---
#### [new 053] Provably Contractive and High-Quality Denoisers for Convergent Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决现有网络稳定性不足的问题。通过设计具有严格Lipschitz约束的去噪网络，提升模型鲁棒性与收敛性，同时保持高质量输出。**

- **链接: [https://arxiv.org/pdf/2603.26168](https://arxiv.org/pdf/2603.26168)**

> **作者:** Shubhi Shukla; Pravin Nair
>
> **摘要:** Image restoration, the recovery of clean images from degraded measurements, has applications in various domains like surveillance, defense, and medical imaging. Despite achieving state-of-the-art (SOTA) restoration performance, existing convolutional and attention-based networks lack stability guarantees under minor shifts in input, exposing a robustness accuracy trade-off. We develop provably contractive (global Lipschitz $< 1$) denoiser networks that considerably reduce this gap. Our design composes proximal layers obtained from unfolding techniques, with Lipschitz-controlled convolutional refinements. By contractivity, our denoiser guarantees that input perturbations of strength $\|\delta\|\le\varepsilon$ induce at most $\varepsilon$ change at the output, while strong baselines such as DnCNN and Restormer can exhibit larger deviations under the same perturbations. On image denoising, the proposed model is competitive with unconstrained SOTA denoisers, reporting the tightest gap for a provably 1-Lipschitz model and establishing that such gaps are indeed achievable by contractive denoisers. Moreover, the proposed denoisers act as strong regularizers for image restoration that provably effect convergence in Plug-and-Play algorithms. Our results show that enforcing strict Lipschitz control does not inherently degrade output quality, challenging a common assumption in the literature and moving the field toward verifiable and stable vision models. Codes and pretrained models are available at this https URL
>
---
#### [new 054] SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras
- **分类: cs.CV**

- **简介: 该论文属于4D重建任务，解决动态场景高成本多相机依赖问题。提出SparseCam4D框架，利用稀疏相机实现时空一致的高质量重建。**

- **链接: [https://arxiv.org/pdf/2603.26481](https://arxiv.org/pdf/2603.26481)**

> **作者:** Weihong Pan; Xiaoyu Zhang; Zhuang Zhang; Zhichao Ye; Nan Wang; Haomin Liu; Guofeng Zhang
>
> **备注:** CVPR 2026
>
> **摘要:** High-quality 4D reconstruction enables photorealistic and immersive rendering of the dynamic real world. However, unlike static scenes that can be fully captured with a single camera, high-quality dynamic scenes typically require dense arrays of tens or even hundreds of synchronized cameras. Dependence on such costly lab setups severely limits practical scalability. The reliance on such costly lab setups severely limits practical scalability. To this end, we propose a sparse-camera dynamic reconstruction framework that exploits abundant yet inconsistent generative observations. Our key innovation is the Spatio-Temporal Distortion Field, which provides a unified mechanism for modeling inconsistencies in generative observations across both spatial and temporal dimensions. Building on this, we develop a complete pipeline that enables 4D reconstruction from sparse and uncalibrated camera inputs. We evaluate our method on multi-camera dynamic scene benchmarks, achieving spatio-temporally consistent high-fidelity renderings and significantly outperforming existing approaches.
>
---
#### [new 055] JRM: Joint Reconstruction Model for Multiple Objects without Alignment
- **分类: cs.CV**

- **简介: 该论文提出JRM模型，解决多对象无对齐重建问题。通过隐式聚合不一致观测，提升重建质量与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.25985](https://arxiv.org/pdf/2603.25985)**

> **作者:** Qirui Wu; Yawar Siddiqui; Duncan Frost; Samir Aroudj; Armen Avetisyan; Richard Newcombe; Angel X. Chang; Jakob Engel; Henry Howard-Jenkins
>
> **摘要:** Object-centric reconstruction seeks to recover the 3D structure of a scene through composition of independent objects. While this independence can simplify modeling, it discards strong signals that could improve reconstruction, notably repetition where the same object model is seen multiple times in a scene, or across scans. We propose the Joint Reconstruction Model (JRM) to leverage repetition by framing object reconstruction as one of personalized generation: multiple observations share a common subject that should be consistent for all observations, while still adhering to the specific pose and state from each. Prior methods in this direction rely on explicit matching and rigid alignment across observations, making them sensitive to errors and difficult to extend to non-rigid transformations. In contrast, JRM is a 3D flow-matching generative model that implicitly aggregates unaligned observations in its latent space, learning to produce consistent and faithful reconstructions in a data-driven manner without explicit constraints. Evaluations on synthetic and real-world data show that JRM's implicit aggregation removes the need for explicit alignment, improves robustness to incorrect associations, and naturally handles non-rigid changes such as articulation. Overall, JRM outperforms both independent and alignment-based baselines in reconstruction quality.
>
---
#### [new 056] Speech-Synchronized Whiteboard Generation via VLM-Driven Structured Drawing Representations
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于教育视频生成任务，解决语音与白板绘图同步的问题。通过构建数据集并使用视觉语言模型，实现语音驱动的结构化绘图生成。**

- **链接: [https://arxiv.org/pdf/2603.25870](https://arxiv.org/pdf/2603.25870)**

> **作者:** Suraj Prasad; Pinak Mahapatra
>
> **摘要:** Creating whiteboard-style educational videos demands precise coordination between freehand illustrations and spoken narration, yet no existing method addresses this multimodal synchronization problem with structured, reproducible drawing representations. We present the first dataset of 24 paired Excalidraw demonstrations with narrated audio, where every drawing element carries millisecond-precision creation timestamps spanning 8 STEM domains. Using this data, we study whether a vision-language model (Qwen2-VL-7B), fine-tuned via LoRA, can predict full stroke sequences synchronized to speech from only 24 demonstrations. Our topic-stratified five-fold evaluation reveals that timestamp conditioning significantly improves temporal alignment over ablated baselines, while the model generalizes across unseen STEM topics. We discuss transferability to real classroom settings and release our dataset and code to support future research in automated educational content generation.
>
---
#### [new 057] CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型优化任务，针对CPU设备设计高效骨干网络。解决CPU并行能力弱导致的效率问题，通过改进卷积结构提升硬件效率，提出CPUBone模型实现良好速度与精度平衡。**

- **链接: [https://arxiv.org/pdf/2603.26425](https://arxiv.org/pdf/2603.26425)**

> **作者:** Moritz Nottebaum; Matteo Dunnhofer; Christian Micheloni
>
> **备注:** Accepted at CVPR Findings 2026
>
> **摘要:** Recent research on vision backbone architectures has predominantly focused on optimizing efficiency for hardware platforms with high parallel processing capabilities. This category increasingly includes embedded systems such as mobile phones and embedded AI accelerator modules. In contrast, CPUs do not have the possibility to parallelize operations in the same manner, wherefore models benefit from a specific design philosophy that balances amount of operations (MACs) and hardware-efficient execution by having high MACs per second (MACpS). In pursuit of this, we investigate two modifications to standard convolutions, aimed at reducing computational cost: grouping convolutions and reducing kernel sizes. While both adaptations substantially decrease the total number of MACs required for inference, sustaining low latency necessitates preserving hardware-efficiency. Our experiments across diverse CPU devices confirm that these adaptations successfully retain high hardware-efficiency on CPUs. Based on these insights, we introduce CPUBone, a new family of vision backbone models optimized for CPU-based inference. CPUBone achieves state-of-the-art Speed-Accuracy Trade-offs (SATs) across a wide range of CPU devices and effectively transfers its efficiency to downstream tasks such as object detection and semantic segmentation. Models and code are available at this https URL.
>
---
#### [new 058] Restore, Assess, Repeat: A Unified Framework for Iterative Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决复杂退化下的恢复效率与泛化问题。提出RAR框架，融合质量评估与恢复过程，实现高效迭代恢复。**

- **链接: [https://arxiv.org/pdf/2603.26385](https://arxiv.org/pdf/2603.26385)**

> **作者:** I-Hsiang Chen; Isma Hadji; Enrique Sanchez; Adrian Bulat; Sy-Yen Kuo; Radu Timofte; Georgios Tzimiropoulos; Brais Martinez
>
> **备注:** Accepted by CVPR2026; Project Page: this https URL
>
> **摘要:** Image restoration aims to recover high quality images from inputs degraded by various factors, such as adverse weather, blur, or low light. While recent studies have shown remarkable progress across individual or unified restoration tasks, they still suffer from limited generalization and inefficiency when handling unknown or composite degradations. To address these limitations, we propose RAR, a Restore, Assess and Repeat process, that integrates Image Quality Assessment (IQA) and Image Restoration (IR) into a unified framework to iteratively and efficiently achieve high quality image restoration. Specifically, we introduce a restoration process that operates entirely in the latent domain to jointly perform degradation identification, image restoration, and quality verification. The resulting model is fully trainable end to end and allows for an all-in-one assess and restore approach that dynamically adapts the restoration process. Also, the tight integration of IQA and IR into a unified model minimizes the latency and information loss that typically arises from keeping the two modules disjoint, (e.g. during image and/or text decoding). Extensive experiments show that our approach consistent improvements under single, unknown and composite degradations, thereby establishing a new state-of-the-art.
>
---
#### [new 059] AutoWeather4D: Autonomous Driving Video Weather Conversion via G-Buffer Dual-Pass Editing
- **分类: cs.CV**

- **简介: 该论文属于视频天气转换任务，旨在解决生成模型依赖大量数据及几何光照纠缠的问题。提出AutoWeather4D框架，通过G-buffer双通道编辑实现几何与光照解耦，提升控制精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.26546](https://arxiv.org/pdf/2603.26546)**

> **作者:** Tianyu Liu; Weitao Xiong; Kunming Luo; Manyuan Zhang; Peng Liu; Yuan Liu; Ping Tan
>
> **摘要:** Generative video models have significantly advanced the photorealistic synthesis of adverse weather for autonomous driving; however, they consistently demand massive datasets to learn rare weather scenarios. While 3D-aware editing methods alleviate these data constraints by augmenting existing video footage, they are fundamentally bottlenecked by costly per-scene optimization and suffer from inherent geometric and illumination entanglement. In this work, we introduce AutoWeather4D, a feed-forward 3D-aware weather editing framework designed to explicitly decouple geometry and illumination. At the core of our approach is a G-buffer Dual-pass Editing mechanism. The Geometry Pass leverages explicit structural foundations to enable surface-anchored physical interactions, while the Light Pass analytically resolves light transport, accumulating the contributions of local illuminants into the global illumination to enable dynamic 3D local relighting. Extensive experiments demonstrate that AutoWeather4D achieves comparable photorealism and structural consistency to generative baselines while enabling fine-grained parametric physical control, serving as a practical data engine for autonomous driving.
>
---
#### [new 060] DiReCT: Disentangled Regularization of Contrastive Trajectories for Physics-Refined Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，解决物理一致性问题。针对流匹配模型违反物理规律的问题，提出DiReCT框架，通过对比学习分离语义与物理轨迹，提升视频物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.25931](https://arxiv.org/pdf/2603.25931)**

> **作者:** Abolfazl Meyarian; Amin Karimi Monsefi; Rajiv Ramnath; Ser-Nam Lim
>
> **摘要:** Flow-matching video generators produce temporally coherent, high-fidelity outputs yet routinely violate elementary physics because their reconstruction objectives penalize per-frame deviations without distinguishing physically consistent dynamics from impossible ones. Contrastive flow matching offers a principled remedy by pushing apart velocity-field trajectories of differing conditions, but we identify a fundamental obstacle in the text-conditioned video setting: semantic-physics entanglement. Because natural-language prompts couple scene content with physical behavior, naive negative sampling draws conditions whose velocity fields largely overlap with the positive sample's, causing the contrastive gradient to directly oppose the flow-matching objective. We formalize this gradient conflict, deriving a precise alignment condition that reveals when contrastive learning helps versus harms training. Guided by this analysis, we introduce DiReCT (Disentangled Regularization of Contrastive Trajectories), a lightweight post-training framework that decomposes the contrastive signal into two complementary scales: a macro-contrastive term that draws partition-exclusive negatives from semantically distant regions for interference-free global trajectory separation, and a micro-contrastive term that constructs hard negatives sharing full scene semantics with the positive sample but differing along a single, LLM-perturbed axis of physical behavior; spanning kinematics, forces, materials, interactions, and magnitudes. A velocity-space distributional regularizer helps to prevent catastrophic forgetting of pretrained visual quality. When applied to Wan 2.1-1.3B, our method improves the physical commonsense score on VideoPhy by 16.7% and 11.3% compared to the baseline and SFT, respectively, without increasing training time.
>
---
#### [new 061] Face2Parts: Exploring Coarse-to-Fine Inter-Regional Facial Dependencies for Generalized Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决多样化的深度伪造识别问题。通过提取多区域特征并利用注意力机制，提出Face2Parts方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.26036](https://arxiv.org/pdf/2603.26036)**

> **作者:** Kutub Uddin; Nusrat Tasnim; Byung Tae Oh
>
> **摘要:** Multimedia data, particularly images and videos, is integral to various applications, including surveillance, visual interaction, biometrics, evidence gathering, and advertising. However, amateur or skilled counterfeiters can simulate them to create deepfakes, often for slanderous motives. To address this challenge, several forensic methods have been developed to ensure the authenticity of the content. The effectiveness of these methods depends on their focus, with challenges arising from the diverse nature of manipulations. In this article, we analyze existing forensic methods and observe that each method has unique strengths in detecting deepfake traces by focusing on specific facial regions, such as the frame, face, lips, eyes, or nose. Considering these insights, we propose a novel hybrid approach called Face2Parts based on hierarchical feature representation ($HFR$) that takes advantage of coarse-to-fine information to improve deepfake detection. The proposed method involves extracting features from the frame, face, and key facial regions (i.e., lips, eyes, and nose) separately to explore the coarse-to-fine relationships. This approach enables us to capture inter-dependencies among facial regions using a channel-attention mechanism and deep triplet learning. We evaluated the proposed method on benchmark deepfake datasets in both intra-, inter-dataset, and inter-manipulation settings. The proposed method achieves an average AUC of 98.42\% on FF++, 79.80\% on CDF1, 85.34\% on CDF2, 89.41\% on DFD, 84.07\% on DFDC, 95.62\% on DTIM, 80.76\% on PDD, and 100\% on WLDR, respectively. The results demonstrate that our approach generalizes effectively and achieves promising performance to outperform the existing methods.
>
---
#### [new 062] VGGRPO: Towards World-Consistent Video Generation with 4D Latent Reward
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频几何一致性问题。通过引入潜空间几何引导框架VGGRPO，提升视频的几何一致性和稳定性，避免依赖昂贵的VAE解码。**

- **链接: [https://arxiv.org/pdf/2603.26599](https://arxiv.org/pdf/2603.26599)**

> **作者:** Zhaochong An; Orest Kupyn; Théo Uscidda; Andrea Colaco; Karan Ahuja; Serge Belongie; Mar Gonzalez-Franco; Marta Tintore Gazulla
>
> **备注:** Project Page: this https URL
>
> **摘要:** Large-scale video diffusion models achieve impressive visual quality, yet often fail to preserve geometric consistency. Prior approaches improve consistency either by augmenting the generator with additional modules or applying geometry-aware alignment. However, architectural modifications can compromise the generalization of internet-scale pretrained models, while existing alignment methods are limited to static scenes and rely on RGB-space rewards that require repeated VAE decoding, incurring substantial compute overhead and failing to generalize to highly dynamic real-world scenes. To preserve the pretrained capacity while improving geometric consistency, we propose VGGRPO (Visual Geometry GRPO), a latent geometry-guided framework for geometry-aware video post-training. VGGRPO introduces a Latent Geometry Model (LGM) that stitches video diffusion latents to geometry foundation models, enabling direct decoding of scene geometry from the latent space. By constructing LGM from a geometry model with 4D reconstruction capability, VGGRPO naturally extends to dynamic scenes, overcoming the static-scene limitations of prior methods. Building on this, we perform latent-space Group Relative Policy Optimization with two complementary rewards: a camera motion smoothness reward that penalizes jittery trajectories, and a geometry reprojection consistency reward that enforces cross-view geometric coherence. Experiments on both static and dynamic benchmarks show that VGGRPO improves camera stability, geometry consistency, and overall quality while eliminating costly VAE decoding, making latent-space geometry-guided reinforcement an efficient and flexible approach to world-consistent video generation.
>
---
#### [new 063] DRUM: Diffusion-based Raydrop-aware Unpaired Mapping for Sim2Real LiDAR Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于Sim2Real任务，解决合成LiDAR数据到真实数据的域适应问题。提出DRUM框架，利用扩散模型生成真实特征，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.26263](https://arxiv.org/pdf/2603.26263)**

> **作者:** Tomoya Miyawaki; Kazuto Nakashima; Yumi Iwashita; Ryo Kurazume
>
> **备注:** ICRA 2026
>
> **摘要:** LiDAR-based semantic segmentation is a key component for autonomous mobile robots, yet large-scale annotation of LiDAR point clouds is prohibitively expensive and time-consuming. Although simulators can provide labeled synthetic data, models trained on synthetic data often underperform on real-world data due to a data-level domain gap. To address this issue, we propose DRUM, a novel Sim2Real translation framework. We leverage a diffusion model pre-trained on unlabeled real-world data as a generative prior and translate synthetic data by reproducing two key measurement characteristics: reflectance intensity and raydrop noise. To improve sample fidelity, we introduce a raydrop-aware masked guidance mechanism that selectively enforces consistency with the input synthetic data while preserving realistic raydrop noise induced by the diffusion prior. Experimental results demonstrate that DRUM consistently improves Sim2Real performance across multiple representations of LiDAR data. The project page is available at this https URL.
>
---
#### [new 064] Neuro-Cognitive Reward Modeling for Human-Centered Autonomous Vehicle Control
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶领域，旨在解决机器驾驶与人类意图对齐的问题。通过EEG信号提取认知信息，融入强化学习奖励机制，提升车辆碰撞规避能力。**

- **链接: [https://arxiv.org/pdf/2603.25968](https://arxiv.org/pdf/2603.25968)**

> **作者:** Zhuoli Zhuang; Yu-Cheng Chang; Yu-Kai Wang; Thomas Do; Chin-Teng Lin
>
> **摘要:** Recent advancements in computer vision have accelerated the development of autonomous driving. Despite these advancements, training machines to drive in a way that aligns with human expectations remains a significant challenge. Human factors are still essential, as humans possess a sophisticated cognitive system capable of rapidly interpreting scene information and making accurate decisions. Aligning machine with human intent has been explored with Reinforcement Learning with Human Feedback (RLHF). Conventional RLHF methods rely on collecting human preference data by manually ranking generated outputs, which is time-consuming and indirect. In this work, we propose an electroencephalography (EEG)-guided decision-making framework to incorporate human cognitive insights without behaviour response interruption into reinforcement learning (RL) for autonomous driving. We collected EEG signals from 20 participants in a realistic driving simulator and analyzed event-related potentials (ERP) in response to sudden environmental changes. Our proposed framework employs a neural network to predict the strength of ERP based on the cognitive information from visual scene information. Moreover, we explore the integration of such cognitive information into the reward signal of the RL algorithm. Experimental results show that our framework can improve the collision avoidance ability of the RL algorithm, highlighting the potential of neuro-cognitive feedback in enhancing autonomous driving systems. Our project page is: this https URL.
>
---
#### [new 065] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，解决恶劣天气下测试时适应（TTA）问题。提出CD-Buffer框架，结合减法与加法策略，自动平衡不同域偏移强度下的特征处理。**

- **链接: [https://arxiv.org/pdf/2603.26092](https://arxiv.org/pdf/2603.26092)**

> **作者:** Youngjun Song; Hyeongyu Kim; Dosik Hwang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Test-Time Adaptation (TTA) enables real-time adaptation to domain shifts without off-line retraining. Recent TTA methods have predominantly explored additive approaches that introduce lightweight modules for feature refinement. Recently, a subtractive approach that removes domain-sensitive channels has emerged as an alternative direction. We observe that these paradigms exhibit complementary effectiveness patterns: subtractive methods excel under severe shifts by removing corrupted features, while additive methods are effective under moderate shifts requiring refinement. However, each paradigm operates effectively only within limited shift severity ranges, failing to generalize across diverse corruption levels. This leads to the following question: can we adaptively balance both strategies based on measured feature-level domain shift? We propose CD-Buffer, a novel complementary dual-buffer framework where subtractive and additive mechanisms operate in opposite yet coordinated directions driven by a unified discrepancy metric. Our key innovation lies in the discrepancy-driven coupling: Our framework couples removal and refinement through a unified discrepancy metric, automatically balancing both strategies based on feature-level shift severity. This establishes automatic channel-wise balancing that adapts differentiated treatment to heterogeneous shift magnitudes without manual tuning. Extensive experiments on KITTI, Cityscapes, and ACDC datasets demonstrate state-of-the-art performance, consistently achieving superior results across diverse weather conditions and severity levels.
>
---
#### [new 066] ClipTTT: CLIP-Guided Test-Time Training Helps LVLMs See Better
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决LVLM在测试时视觉输入受损导致的幻觉问题。通过ClipTTT方法，利用CLIP引导实时训练，提升模型在退化条件下的表现。**

- **链接: [https://arxiv.org/pdf/2603.26486](https://arxiv.org/pdf/2603.26486)**

> **作者:** Mriganka Nath; Anurag Das; Jiahao Xie; Bernt Schiele
>
> **备注:** 30 pages, 12 figures
>
> **摘要:** Large vision-language models (LVLMs) tend to hallucinate, especially when visual inputs are corrupted at test time. We show that such corruptions act as additional distribution shifts, significantly amplifying hallucination rates in real-world applications. To address this, we propose CLIP-guided Test-Time Training (ClipTTT), a method to adapt LVLMs under degraded conditions on the fly with a single test sample. Specifically, we leverage the image-text alignment strength of a pre-trained CLIP model as a stable guidance signal to identify reliable self-supervision targets, enabling rapid adaptation without altering the base LVLMs. Extensive experiments on standard hallucination benchmarks, with 15 common corruptions, demonstrate that ClipTTT effectively mitigates hallucinations and improves descriptive faithfulness under visual corruptions.
>
---
#### [new 067] PerceptionComp: A Video Benchmark for Complex Perception-Centric Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PerceptionComp，一个复杂视频感知推理基准，用于评估多步骤、长时序的视觉推理能力，解决视频理解中的复杂感知问题。**

- **链接: [https://arxiv.org/pdf/2603.26653](https://arxiv.org/pdf/2603.26653)**

> **作者:** Shaoxuan Li; Zhixuan Zhao; Hanze Deng; Zirun Ma; Shulin Tian; Zuyan Liu; Yushi Hu; Haoning Wu; Yuhao Dong; Benlin Liu; Ziwei Liu; Ranjay Krishna
>
> **备注:** Project Page: this https URL
>
> **摘要:** We introduce PerceptionComp, a manually annotated benchmark for complex, long-horizon, perception-centric video reasoning. PerceptionComp is designed so that no single moment is sufficient: answering each question requires multiple temporally separated pieces of visual evidence and compositional constraints under conjunctive and sequential logic, spanning perceptual subtasks such as objects, attributes, relations, locations, actions, and events, and requiring skills including semantic recognition, visual correspondence, temporal reasoning, and spatial reasoning. The benchmark contains 1,114 highly complex questions on 279 videos from diverse domains including city walk tours, indoor villa tours, video games, and extreme outdoor sports, with 100% manual annotation. Human studies show that PerceptionComp requires substantial test-time thinking and repeated perception steps: participants take much longer than on prior benchmarks, and accuracy drops to near chance (18.97%) when rewatching is disallowed. State-of-the-art MLLMs also perform substantially worse on PerceptionComp than on existing benchmarks: the best model in our evaluation, Gemini-3-Flash, reaches only 45.96% accuracy in the five-choice setting, while open-source models remain below 40%. These results suggest that perception-centric long-horizon video reasoning remains a major bottleneck, and we hope PerceptionComp will help drive progress in perceptual reasoning.
>
---
#### [new 068] DuSCN-FusionNet: An Interpretable Dual-Channel Structural Covariance Fusion Framework for ADHD Classification Using Structural MRI
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于ADHD分类任务，旨在解决缺乏可靠影像生物标志物的问题。提出DuSCN-FusionNet框架，结合结构协方差网络与特征融合，提升分类性能并增强可解释性。**

- **链接: [https://arxiv.org/pdf/2603.26351](https://arxiv.org/pdf/2603.26351)**

> **作者:** Qurat Ul Ain; Alptekin Temizel; Soyiba Jawed
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Attention Deficit Hyperactivity Disorder (ADHD) is a highly prevalent neurodevelopmental condition; however, its neurobiological diagnosis remains challenging due to the lack of reliable imaging-based biomarkers, particularly anatomical markers. Structural MRI (sMRI) provides a non-invasive modality for investigating brain alterations associated with ADHD; nevertheless, most deep learning approaches function as black-box systems, limiting clinical trust and interpretability. In this work, we propose DuSCN-FusionNet, an interpretable sMRI-based framework for ADHD classification that leverages dual-channel Structural Covariance Networks (SCNs) to capture inter-regional morphological relationships. ROI-wise mean intensity and intra-regional variability descriptors are used to construct intensity-based and heterogeneity-based SCNs, which are processed through an SCN-CNN encoder. In parallel, auxiliary ROI-wise variability features and global statistical descriptors are integrated via late-stage fusion to enhance performance. The model is evaluated using stratified 10-fold cross-validation with a 5-seed ensemble strategy, achieving a mean balanced accuracy of 80.59% and an AUC of 0.778 on the Peking University site of the ADHD-200 dataset. DuSCN-FusionNet further achieves precision, recall, and F1-scores of 81.66%, 80.59%, and 80.27%, respectively. Moreover, Grad-CAM is adapted to the SCN domain to derive ROI-level importance scores, enabling the identification of structurally relevant brain regions as potential biomarkers.
>
---
#### [new 069] HyVIC: A Metric-Driven Spatio-Spectral Hyperspectral Image Compression Architecture Based on Variational Autoencoders
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像压缩任务，旨在解决传统方法未能有效处理高光谱数据时空冗余的问题。提出HyVIC架构，通过平衡空间与光谱特征学习，提升压缩效率和重建质量。**

- **链接: [https://arxiv.org/pdf/2603.26468](https://arxiv.org/pdf/2603.26468)**

> **作者:** Martin Hermann Paul Fuchs; Behnood Rasti; Begüm Demir
>
> **摘要:** The rapid growth of hyperspectral data archives in remote sensing (RS) necessitates effective compression methods for storage and transmission. Recent advances in learning-based hyperspectral image (HSI) compression have significantly enhanced both reconstruction fidelity and compression efficiency. However, existing methods typically adapt variational image compression models designed for natural images, without adequately accounting for the distinct spatio-spectral redundancies inherent in HSIs. In particular, they lack explicit architectural designs to balance spatial and spectral feature learning, limiting their ability to effectively leverage the unique characteristics of hyperspectral data. To address this issue, we introduce spatio-spectral variational hyperspectral image compression architecture (HyVIC). The proposed model comprises four main components: 1) adjustable spatio-spectral encoder; 2) spatio-spectral hyperencoder; 3) spatio-spectral hyperdecoder; and 4) adjustable spatio-spectral decoder. We demonstrate that the trade-off between spatial and spectral feature learning is crucial for the reconstruction fidelity, and therefore present a metric-driven strategy to systematically select the hyperparameters of the proposed model. Extensive experiments on two benchmark datasets demonstrate the effectiveness of the proposed model, achieving high spatial and spectral reconstruction fidelity across a wide range of compression ratios (CRs) and improving the state of the art by up to 4.66dB in terms of BD-PSNR. Based on our results, we offer insights and derive practical guidelines to guide future research directions in learning-based variational HSI compression. Our code and pre-trained model weights are publicly available at this https URL .
>
---
#### [new 070] ARTA: Adaptive Mixed-Resolution Token Allocation for Efficient Dense Feature Extraction
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ARTA，一种高效的密集特征提取方法。针对视觉Transformer计算效率低的问题，通过混合分辨率令牌分配，提升性能并减少计算量。**

- **链接: [https://arxiv.org/pdf/2603.26258](https://arxiv.org/pdf/2603.26258)**

> **作者:** David Hagerman; Roman Naeem; Erik Brorsson; Fredrik Kahl; Lennart Svensson
>
> **摘要:** We present ARTA, a mixed-resolution coarse-to-fine vision transformer for efficient dense feature extraction. Unlike models that begin with dense high-resolution (fine) tokens, ARTA starts with low-resolution (coarse) tokens and uses a lightweight allocator to predict which regions require more fine tokens. The allocator iteratively predicts a semantic (class) boundary score and allocates additional tokens to patches above a low threshold, concentrating token density near boundaries while maintaining high sensitivity to weak boundary evidence. This targeted allocation encourages tokens to represent a single semantic class rather than a mixture of classes. Mixed-resolution attention enables interaction between coarse and fine tokens, focusing computation on semantically complex areas while avoiding redundant processing in homogeneous regions. Experiments demonstrate that ARTA achieves state-of-the-art results on ADE20K and COCO-Stuff with substantially fewer FLOPs, and delivers competitive performance on Cityscapes at markedly lower compute. For example, ARTA-Base attains 54.6 mIoU on ADE20K in the ~100M-parameter class while using fewer FLOPs and less memory than comparable backbones.
>
---
#### [new 071] HAD: Heterogeneity-Aware Distillation for Lifelong Heterogeneous Learning
- **分类: cs.CV**

- **简介: 该论文属于 lifelong heterogeneous learning 任务，解决跨异构任务学习中知识保留问题。提出 HAD 方法，通过自蒸馏保留异构知识，提升密集预测性能。**

- **链接: [https://arxiv.org/pdf/2603.26192](https://arxiv.org/pdf/2603.26192)**

> **作者:** Xuerui Zhang; Xuehao Wang; Zhan Zhuang; Linglan Zhao; Ziyue Li; Xinmin Zhang; Zhihuan Song; Yu Zhang
>
> **摘要:** Lifelong learning aims to preserve knowledge acquired from previous tasks while incorporating knowledge from a sequence of new tasks. However, most prior work explores only streams of homogeneous tasks (\textit{e.g.}, only classification tasks) and neglects the scenario of learning across heterogeneous tasks that possess different structures of outputs. In this work, we formalize this broader setting as lifelong heterogeneous learning (LHL). Departing from conventional lifelong learning, the task sequence of LHL spans different task types, and the learner needs to retain heterogeneous knowledge for different output space structures. To instantiate the LHL, we focus on LHL in the context of dense prediction (LHL4DP), a realistic and challenging scenario. To this end, we propose the Heterogeneity-Aware Distillation (HAD) method, an exemplar-free approach that preserves previously gained heterogeneous knowledge by self-distillation in each training phase. The proposed HAD comprises two complementary components, including a distribution-balanced heterogeneity-aware distillation loss to alleviate the global imbalance of prediction distribution and a salience-guided heterogeneity-aware distillation loss that concentrates learning on informative edge pixels extracted with the Sobel operator. Extensive experiments demonstrate that the proposed HAD method significantly outperforms existing methods in this new scenario.
>
---
#### [new 072] GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation
- **分类: cs.CV**

- **简介: 该论文提出GaussianGPT，属于3D场景生成任务，旨在通过自回归方式直接生成3D高斯分布，解决传统方法依赖扩散模型的问题。**

- **链接: [https://arxiv.org/pdf/2603.26661](https://arxiv.org/pdf/2603.26661)**

> **作者:** Nicolas von Lützow; Barbara Rössle; Katharina Schmid; Matthias Nießner
>
> **备注:** Project page: this https URL - Project video: this https URL
>
> **摘要:** Most recent advances in 3D generative modeling rely on diffusion or flow-matching formulations. We instead explore a fully autoregressive alternative and introduce GaussianGPT, a transformer-based model that directly generates 3D Gaussians via next-token prediction, thus facilitating full 3D scene generation. We first compress Gaussian primitives into a discrete latent grid using a sparse 3D convolutional autoencoder with vector quantization. The resulting tokens are serialized and modeled using a causal transformer with 3D rotary positional embedding, enabling sequential generation of spatial structure and appearance. Unlike diffusion-based methods that refine scenes holistically, our formulation constructs scenes step-by-step, naturally supporting completion, outpainting, controllable sampling via temperature, and flexible generation horizons. This formulation leverages the compositional inductive biases and scalability of autoregressive modeling while operating on explicit representations compatible with modern neural rendering pipelines, positioning autoregressive transformers as a complementary paradigm for controllable and context-aware 3D generation.
>
---
#### [new 073] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于GUI接地任务，旨在探索离散扩散视觉语言模型在GUI理解中的可行性。通过改进掩码策略，提升模型性能，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.26211](https://arxiv.org/pdf/2603.26211)**

> **作者:** Shrinidhi Kumbhar; Haofu Liao; Srikar Appalaraju; Kunwar Yashraj Singh
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Autoregressive (AR) vision-language models (VLMs) have long dominated multimodal understanding, reasoning, and graphical user interface (GUI) grounding. Recently, discrete diffusion vision-language models (DVLMs) have shown strong performance in multimodal reasoning, offering bidirectional attention, parallel token generation, and iterative refinement. However, their potential for GUI grounding remains unexplored. In this work, we evaluate whether discrete DVLMs can serve as a viable alternative to AR models for GUI grounding. We adapt LLaDA-V for single-turn action and bounding-box prediction, framing the task as text generation from multimodal input. To better capture the hierarchical structure of bounding-box geometry, we propose a hybrid masking schedule that combines linear and deterministic masking, improving grounding accuracy by up to 6.1 points in Step Success Rate (SSR) over the GUI-adapted LLaDA-V trained with linear masking. Evaluations on four datasets spanning web, desktop, and mobile interfaces show that the adapted diffusion model with hybrid masking consistently outperforms the linear-masked variant and performs competitively with autoregressive counterparts despite limited pretraining. Systematic ablations reveal that increasing diffusion steps, generation length, and block length improves accuracy but also increases latency, with accuracy plateauing beyond a certain number of diffusion steps. Expanding the training data with diverse GUI domains further reduces latency by about 1.3 seconds and improves grounding accuracy by an average of 20 points across benchmarks. These results demonstrate that discrete DVLMs are a promising modeling framework for GUI grounding and represent an important step toward diffusion-based GUI agents.
>
---
#### [new 074] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于文本到图像扩散模型中的概念擦除任务，旨在去除特定概念同时保留相关概念。提出NLCE框架，通过三阶段方法实现精准擦除并维护邻近概念结构。**

- **链接: [https://arxiv.org/pdf/2603.25994](https://arxiv.org/pdf/2603.25994)**

> **作者:** Zhuan Shi; Alireza Dehghanpour Farashah; Rik de Vries; Golnoosh Farnadi
>
> **备注:** Accepted by CVPR 2026 main
>
> **摘要:** Concept erasure in text-to-image diffusion models seeks to remove undesired concepts while preserving overall generative capability. Localized erasure methods aim to restrict edits to the spatial region occupied by the target concept. However, we observe that suppressing a concept can unintentionally weaken semantically related neighbor concepts, reducing fidelity in fine-grained domains. We propose Neighbor-Aware Localized Concept Erasure (NLCE), a training-free framework designed to better preserve neighboring concepts while removing target concepts. It operates in three stages: (1) a spectrally-weighted embedding modulation that attenuates target concept directions while stabilizing neighbor concept representations, (2) an attention-guided spatial gate that identifies regions exhibiting residual concept activation, and (3) a spatially-gated hard erasure that eliminates remaining traces only where necessary. This neighbor-aware pipeline enables localized concept removal while maintaining the surrounding concept neighborhood structure. Experiments on fine-grained datasets (Oxford Flowers, Stanford Dogs) show that our method effectively removes target concepts while better preserving closely related categories. Additional results on celebrity identity, explicit content and artistic style demonstrate robustness and generalization to broader erasure scenarios.
>
---
#### [new 075] Dynamic LIBRAS Gesture Recognition via CNN over Spatiotemporal Matrix Representation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于手势识别任务，旨在通过CNN对动态手部动作进行分类，解决LIBRAS手势在家庭自动化中的识别问题。**

- **链接: [https://arxiv.org/pdf/2603.25863](https://arxiv.org/pdf/2603.25863)**

> **作者:** Jasmine Moreira
>
> **备注:** 6 pages, 10 figures, 1 table
>
> **摘要:** This paper proposes a method for dynamic hand gesture recognition based on the composition of two models: the MediaPipe Hand Landmarker, responsible for extracting 21 skeletal keypoints of the hand, and a convolutional neural network (CNN) trained to classify gestures from a spatiotemporal matrix representation of dimensions 90 by 21 of those keypoints. The method is applied to the recognition of LIBRAS (Brazilian Sign Language) gestures for device control in a home automation system, covering 11 classes of static and dynamic gestures. For real-time inference, a sliding window with temporal frame triplication is used, enabling continuous recognition without recurrent networks. Tests achieved 95\% accuracy under low-light conditions and 92\% under normal lighting. The results indicate that the approach is effective, although systematic experiments with greater user diversity are needed for a more thorough evaluation of generalization.
>
---
#### [new 076] From Synthetic Data to Real Restorations: Diffusion Model for Patient-specific Dental Crown Completion
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于牙齿修复任务，解决缺损牙冠生成问题。通过扩散模型，基于局部解剖信息自动完成牙冠重建，提升修复精度与安全性。**

- **链接: [https://arxiv.org/pdf/2603.26588](https://arxiv.org/pdf/2603.26588)**

> **作者:** Dávid Pukanec; Tibor Kubík; Michal Španěl
>
> **备注:** VISAPP 2026 Conference
>
> **摘要:** We present ToothCraft, a diffusion-based model for the contextual generation of tooth crowns, trained on artificially created incomplete teeth. Building upon recent advancements in conditioned diffusion models for 3D shapes, we developed a model capable of an automated tooth crown completion conditioned on local anatomical context. To address the lack of training data for this task, we designed an augmentation pipeline that generates incomplete tooth geometries from a publicly available dataset of complete dental arches (3DS, ODD). By synthesising a diverse set of training examples, our approach enables robust learning across a wide spectrum of tooth defects. Experimental results demonstrate the strong capability of our model to reconstruct complete tooth crowns, achieving an intersection over union (IoU) of 81.8% and a Chamfer Distance (CD) of 0.00034 on synthetically damaged testing restorations. Our experiments demonstrate that the model can be applied directly to real-world cases, effectively filling in incomplete teeth, while generated crowns show minimal intersection with the opposing dentition, thus reducing the risk of occlusal interference. Access to the code, model weights, and dataset information will be available at: this https URL
>
---
#### [new 077] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline
- **分类: cs.CV**

- **简介: 该论文提出视频流畅性评估（VFA）任务，解决现有视频质量评估中对流畅性关注不足的问题。构建了FluVid数据集，开发基准测试并提出FluNet模型以提升流畅性评估性能。**

- **链接: [https://arxiv.org/pdf/2603.26055](https://arxiv.org/pdf/2603.26055)**

> **作者:** Qizhi Xie; Kun Yuan; Yunpeng Qu; Ming Sun; Chao Zhou; Jihong Zhu
>
> **备注:** 14 pages, 6 figures. Accepted by CVPR 2026 findings track
>
> **摘要:** Accurately estimating humans' subjective feedback on video fluency, e.g., motion consistency and frame continuity, is crucial for various applications like streaming and gaming. Yet, it has long been overlooked, as prior arts have focused on solving it in the video quality assessment (VQA) task, merely as a sub-dimension of overall quality. In this work, we conduct pilot experiments and reveal that current VQA predictions largely underrepresent fluency, thereby limiting their applicability. To this end, we pioneer Video Fluency Assessment (VFA) as a standalone perceptual task focused on the temporal dimension. To advance VFA research, 1) we construct a fluency-oriented dataset, FluVid, comprising 4,606 in-the-wild videos with balanced fluency distribution, featuring the first-ever scoring criteria and human study for VFA. 2) We develop a large-scale benchmark of 23 methods, the most comprehensive one thus far on FluVid, gathering insights for VFA-tailored model designs. 3) We propose a baseline model called FluNet, which deploys temporal permuted self-attention (T-PSA) to enrich input fluency information and enhance long-range inter-frame interactions. Our work not only achieves state-of-the-art performance but, more importantly, offers the community a roadmap to explore solutions for VFA.
>
---
#### [new 078] Dynamic Token Compression for Efficient Video Understanding through Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，解决视频中视觉token冗余导致的计算成本高和性能下降问题。提出SCORE框架，通过强化学习实现动态token压缩，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.26365](https://arxiv.org/pdf/2603.26365)**

> **作者:** Shida Wang; YongXiang Hua; Zhou Tao; Haoyu Cao; Linli Xu
>
> **摘要:** Multimodal Large Language Models have demonstrated remarkable capabilities in video understanding, yet face prohibitive computational costs and performance degradation from ''context rot'' due to massive visual token redundancy. Existing compression strategies typically rely on heuristics or fixed transformations that are often decoupled from the downstream task objectives, limiting their adaptability and effectiveness. To address this, we propose SCORE (Surprise-augmented token COmpression via REinforcement learning), a unified framework that learns an adaptive token compression policy. SCORE introduces a lightweight policy network conditioned on a surprise-augmented state representation that incorporates inter-frame residuals to explicitly capture temporal dynamics and motion saliency. We optimize this policy using a group-wise reinforcement learning scheme with a split-advantage estimator, stabilized by a two-stage curriculum transferring from static pseudo-videos to real dynamic videos. Extensive experiments on diverse video understanding benchmarks demonstrate that SCORE significantly outperforms state-of-the-art baselines. Notably, SCORE achieves a 16x prefill speedup while preserving 99.5% of original performance at a 10% retention ratio, offering a scalable solution for efficient long-form video understanding.
>
---
#### [new 079] A-SelecT: Automatic Timestep Selection for Diffusion Transformer Representation Learning
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于生成模型的表示学习任务，旨在提升扩散Transformer（DiT）的训练效率和表征能力。针对当前 timestep 选择不足的问题，提出A-SelecT方法，动态选择信息丰富的timestep，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.25758](https://arxiv.org/pdf/2603.25758)**

> **作者:** Changyu Liu; James Chenhao Liang; Wenhao Yang; Yiming Cui; Jinghao Yang; Tianyang Wang; Qifan Wang; Dongfang Liu; Cheng Han
>
> **摘要:** Diffusion models have significantly reshaped the field of generative artificial intelligence and are now increasingly explored for their capacity in discriminative representation learning. Diffusion Transformer (DiT) has recently gained attention as a promising alternative to conventional U-Net-based diffusion models, demonstrating a promising avenue for downstream discriminative tasks via generative pre-training. However, its current training efficiency and representational capacity remain largely constrained due to the inadequate timestep searching and insufficient exploitation of DiT-specific feature representations. In light of this view, we introduce Automatically Selected Timestep (A-SelecT) that dynamically pinpoints DiT's most information-rich timestep from the selected transformer feature in a single run, eliminating the need for both computationally intensive exhaustive timestep searching and suboptimal discriminative feature selection. Extensive experiments on classification and segmentation benchmarks demonstrate that DiT, empowered by A-SelecT, surpasses all prior diffusion-based attempts efficiently and effectively.
>
---
#### [new 080] From Pen to Pixel: Translating Hand-Drawn Plots into Graphical APIs via a Novel Benchmark and Efficient Adapter
- **分类: cs.CV**

- **简介: 该论文属于数据可视化任务，旨在解决手绘图表到图形API的转换问题。通过构建HDpy-13数据集和提出Plot-Adapter模型，提升非专家用户生成图表的能力。**

- **链接: [https://arxiv.org/pdf/2603.26356](https://arxiv.org/pdf/2603.26356)**

> **作者:** Zhenghao Xu; Mengning Yang
>
> **摘要:** As plots play a critical role in modern data visualization and analysis, Plot2API is launched to help non-experts and beginners create their desired plots by directly recommending graphical APIs from reference plot images by neural networks. However, previous works on Plot2API have primarily focused on the recommendation for standard plot images, while overlooking the hand-drawn plot images that are more accessible to non-experts and beginners. To make matters worse, both Plot2API models trained on standard plot images and powerful multi-modal large language models struggle to effectively recommend APIs for hand-drawn plot images due to the domain gap and lack of expertise. To facilitate non-experts and beginners, we introduce a hand-drawn plot dataset named HDpy-13 to improve the performance of graphical API recommendations for hand-drawn plot images. Additionally, to alleviate the considerable strain of parameter growth and computational resource costs arising from multi-domain and multi-language challenges in Plot2API, we propose Plot-Adapter that allows for the training and storage of separate adapters rather than requiring an entire model for each language and domain. In particular, Plot-Adapter incorporates a lightweight CNN block to improve the ability to capture local features and implements projection matrix sharing to reduce the number of fine-tuning parameters further. Experimental results demonstrate both the effectiveness of HDpy-13 and the efficiency of Plot-Adapter.
>
---
#### [new 081] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生存预测任务，解决多模态数据缺失问题。提出MUST框架，分离模态特异性信息并生成缺失模态表示，提升预测准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26071](https://arxiv.org/pdf/2603.26071)**

> **作者:** Kyungwon Kim; Dosik Hwang
>
> **备注:** Accepted to CVPR 2026. 10 pages, 5 figures, supplementary included
>
> **摘要:** Accurate survival prediction from multimodal medical data is essential for precision oncology, yet clinical deployment faces a persistent challenge: modalities are frequently incomplete due to cost constraints, technical limitations, or retrospective data availability. While recent methods attempt to address missing modalities through feature alignment or joint distribution learning, they fundamentally lack explicit modeling of the unique contributions of each modality as opposed to the information derivable from other modalities. We propose MUST (Modality-Specific representation-aware Transformer), a novel framework that explicitly decomposes each modality's representation into modality-specific and cross-modal contextualized components through algebraic constraints in a learned low-rank shared subspace. This decomposition enables precise identification of what information is lost when a modality is absent. For the truly modality-specific information that cannot be inferred from available modalities, we employ conditional latent diffusion models to generate high-quality representations conditioned on recovered shared information and learned structural priors. Extensive experiments on five TCGA cancer datasets demonstrate that MUST achieves state-of-the-art performance with complete data while maintaining robust predictions in both missing pathology and missing genomics conditions, with clinically acceptable inference latency.
>
---
#### [new 082] HolisticSemGes: Semantic Grounding of Holistic Co-Speech Gesture Generation with Contrastive Flow-Matching
- **分类: cs.CV**

- **简介: 该论文属于语音同步手势生成任务，旨在解决现有方法依赖预定义规则、缺乏语义一致性及跨模态不一致的问题。提出一种基于对比流匹配的模型，提升手势的语义相关性和跨模态一致性。**

- **链接: [https://arxiv.org/pdf/2603.26553](https://arxiv.org/pdf/2603.26553)**

> **作者:** Lanmiao Liu; Esam Ghaleb; Aslı Özyürek; Zerrin Yumak
>
> **摘要:** While the field of co-speech gesture generation has seen significant advances, producing holistic, semantically grounded gestures remains a challenge. Existing approaches rely on external semantic retrieval methods, which limit their generalisation capability due to dependency on predefined linguistic rules. Flow-matching-based methods produce promising results; however, the network is optimised using only semantically congruent samples without exposure to negative examples, leading to learning rhythmic gestures rather than sparse motion, such as iconic and metaphoric gestures. Furthermore, by modelling body parts in isolation, the majority of methods fail to maintain crossmodal consistency. We introduce a Contrastive Flow Matching-based co-speech gesture generation model that uses mismatched audio-text conditions as negatives, training the velocity field to follow the correct motion trajectory while repelling semantically incongruent trajectories. Our model ensures cross-modal coherence by embedding text, audio, and holistic motion into a composite latent space via cosine and contrastive objectives. Extensive experiments and a user study demonstrate that our proposed approach outperforms state-of-the-art methods on two datasets, BEAT2 and SHOW.
>
---
#### [new 083] Learning to Trim: End-to-End Causal Graph Pruning with Dynamic Anatomical Feature Banks for Medical VQA
- **分类: cs.CV**

- **简介: 该论文属于医学视觉问答任务，旨在解决模型依赖数据集特定关联而非真实诊断证据的问题。提出LCT框架，结合动态解剖特征库与可微剪枝模块，提升模型的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.26028](https://arxiv.org/pdf/2603.26028)**

> **作者:** Zibo Xu; Qiang Li; Weizhi Nie; Yuting Su
>
> **摘要:** Medical Visual Question Answering (MedVQA) models often exhibit limited generalization due to reliance on dataset-specific correlations, such as recurring anatomical patterns or question-type regularities, rather than genuine diagnostic evidence. Existing causal approaches are typically implemented as static adjustments or post-hoc corrections. To address this issue, we propose a Learnable Causal Trimming (LCT) framework that integrates causal pruning into end-to-end optimization. We introduce a Dynamic Anatomical Feature Bank (DAFB), updated via a momentum mechanism, to capture global prototypes of frequent anatomical and linguistic patterns, serving as an approximation of dataset-level regularities. We further design a differentiable trimming module that estimates the dependency between instance-level representations and the global feature bank. Features highly correlated with global prototypes are softly suppressed, while instance-specific evidence is emphasized. This learnable mechanism encourages the model to prioritize causal signals over spurious correlations adaptively. Experiments on VQA-RAD, SLAKE, SLAKE-CP and PathVQA demonstrate that LCT consistently improves robustness and generalization over existing debiasing strategies.
>
---
#### [new 084] Seeing Like Radiologists: Context- and Gaze-Guided Vision-Language Pretraining for Chest X-rays
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉-语言预训练任务，旨在解决模型无法模拟放射科医生诊断流程的问题。通过引入上下文和注视引导机制，提升疾病模式建模与跨模态对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.26049](https://arxiv.org/pdf/2603.26049)**

> **作者:** Kang Liu; Zhuoqi Ma; Siyu Liang; Yunan Li; Xiyue Gao; Chao Liang; Kun Xie; Qiguang Miao
>
> **备注:** Code: this https URL
>
> **摘要:** Despite recent advances in medical vision-language pretraining, existing models still struggle to capture the diagnostic workflow: radiographs are typically treated as context-agnostic images, while radiologists' gaze -- a crucial cue for visual reasoning -- remains largely underexplored by existing methods. These limitations hinder the modeling of disease-specific patterns and weaken cross-modal alignment. To bridge this gap, we introduce CoGaze, a Context- and Gaze-guided vision-language pretraining framework for chest X-rays. We first propose a context-infused vision encoder that models how radiologists integrate clinical context -- including patient history, symptoms, and diagnostic intent -- to guide diagnostic reasoning. We then present a multi-level supervision paradigm that (1) enforces intra- and inter-modal semantic alignment through hybrid-positive contrastive learning, (2) injects diagnostic priors via disease-aware cross-modal representation learning, and (3) leverages radiologists' gaze as probabilistic priors to guide attention toward diagnostically salient regions. Extensive experiments demonstrate that CoGaze consistently outperforms state-of-the-art methods across diverse tasks, achieving up to +2.0% CheXbertF1 and +1.2% BLEU2 for free-text and structured report generation, +23.2% AUROC for zero-shot classification, and +12.2% Precision@1 for image-text retrieval. Code is available at this https URL.
>
---
#### [new 085] 4DRaL: Bridging 4D Radar with LiDAR for Place Recognition using Knowledge Distillation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于 place recognition 任务，旨在提升4D雷达在恶劣天气下的定位性能。通过知识蒸馏，将LiDAR模型的知识迁移至4D雷达模型，解决其数据稀疏和噪声问题。**

- **链接: [https://arxiv.org/pdf/2603.26206](https://arxiv.org/pdf/2603.26206)**

> **作者:** Ningyuan Huang; Zhiheng Li; Zheng Fang
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Place recognition is crucial for loop closure detection and global localization in robotics. Although mainstream algorithms typically rely on cameras and LiDAR, these sensors are susceptible to adverse weather conditions. Fortunately, the recently developed 4D millimeter-wave radar (4D radar) offers a promising solution for all-weather place recognition. However, the inherent noise and sparsity in 4D radar data significantly limit its performance. Thus, in this paper, we propose a novel framework called 4DRaL that leverages knowledge distillation (KD) to enhance the place recognition performance of 4D radar. Its core is to adopt a high-performance LiDAR-to-LiDAR (L2L) place recognition model as a teacher to guide the training of a 4D radar-to-4D radar (R2R) place recognition model. 4DRaL comprises three key KD modules: a local image enhancement module to handle the sparsity of raw 4D radar points, a feature distribution distillation module that ensures the student model generates more discriminative features, and a response distillation module to maintain consistency in feature space between the teacher and student models. More importantly, 4DRaL can also be trained for 4D radar-to-LiDAR (R2L) place recognition through different module configurations. Experimental results prove that 4DRaL achieves state-of-the-art performance in both R2R and R2L tasks regardless of normal or adverse weather.
>
---
#### [new 086] Automated Quality Assessment of Blind Sweep Obstetric Ultrasound for Improved Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决BSOU图像质量对AI诊断的影响问题。通过模拟采集偏差，评估质量并提升诊断可靠性。**

- **链接: [https://arxiv.org/pdf/2603.25886](https://arxiv.org/pdf/2603.25886)**

> **作者:** Prasiddha Bhandari; Kanchan Poudel; Nishant Luitel; Bishram Acharya; Angelina Ghimire; Tyler Wellman; Kilian Koepsell; Pradeep Raj Regmi; Bishesh Khanal
>
> **摘要:** Blind Sweep Obstetric Ultrasound (BSOU) enables scalable fetal imaging in low-resource settings by allowing minimally trained operators to acquire standardized sweep videos for automated Artificial Intelligence(AI) interpretation. However, the reliability of such AI systems depends critically on the quality of the acquired sweeps, and little is known about how deviations from the intended protocol affect downstream predictions. In this work, we present a systematic evaluation of BSOU quality and its impact on three key AI tasks: sweep-tag classification, fetal presentation classification, and placenta-location classification. We simulate plausible acquisition deviations, including reversed sweep direction, probe inversion, and incomplete sweeps, to quantify model robustness, and we develop automated quality-assessment models capable of detecting these perturbations. To approximate real-world deployment, we simulate a feedback loop in which flagged sweeps are re-acquired, showing that such correction improves downstream task performance. Our findings highlight the sensitivity of BSOU-based AI models to acquisition variability and demonstrate that automated quality assessment can play a central role in building reliable, scalable AI-assisted prenatal ultrasound workflows, particularly in low-resource environments.
>
---
#### [new 087] Low-Rank-Modulated Functa: Exploring the Latent Space of Implicit Neural Representations for Interpretable Ultrasound Video Analysis
- **分类: cs.CV**

- **简介: 该论文提出LRM-Functa，用于可解释的超声视频分析，解决隐空间结构不清晰和缺乏可解释性的问题，通过低秩调制提升视频帧的生成与理解能力。**

- **链接: [https://arxiv.org/pdf/2603.25951](https://arxiv.org/pdf/2603.25951)**

> **作者:** Julia Wolleb; Cristiana Baloescu; Alicia Durrer; Hemant D. Tagare; Xenophon Papademetris
>
> **摘要:** Implicit neural representations (INRs) have emerged as a powerful framework for continuous image representation learning. In Functa-based approaches, each image is encoded as a latent modulation vector that conditions a shared INR, enabling strong reconstruction performance. However, the structure and interpretability of the corresponding latent spaces remain largely unexplored. In this work, we investigate the latent space of Functa-based models for ultrasound videos and propose Low-Rank-Modulated Functa (LRM-Functa), a novel architecture that enforces a low-rank adaptation of modulation vectors in the time-resolved latent space. When applied to cardiac ultrasound, the resulting latent space exhibits clearly structured periodic trajectories, facilitating visualization and interpretability of temporal patterns. The latent space can be traversed to sample novel frames, revealing smooth transitions along the cardiac cycle, and enabling direct readout of end-diastolic (ED) and end-systolic (ES) frames without additional model training. We show that LRM-Functa outperforms prior methods in unsupervised ED and ES frame detection, while compressing each video frame to as low as rank k=2 without sacrificing competitive downstream performance on ejection fraction prediction. Evaluations on out-of-distribution frame selection in a cardiac point-of-care dataset, as well as on lung ultrasound for B-line classification, demonstrate the generalizability of our approach. Overall, LRM-Functa provides a compact, interpretable, and generalizable framework for ultrasound video analysis. The code is available at this https URL.
>
---
#### [new 088] Consistency Beyond Contrast: Enhancing Open-Vocabulary Object Detection Robustness via Contextual Consistency Learning
- **分类: cs.CV**

- **简介: 该论文属于开放词汇目标检测任务，旨在解决模型在不同场景中检测同一物体时的鲁棒性问题。通过引入上下文一致性学习框架，提升模型在变化环境下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.26179](https://arxiv.org/pdf/2603.26179)**

> **作者:** Bozhao Li; Shaocong Wu; Tong Shao; Senqiao Yang; Qiben Shan; Zhuotao Tian; Jingyong Su
>
> **摘要:** Recent advances in open-vocabulary object detection focus primarily on two aspects: scaling up datasets and leveraging contrastive learning to align language and vision modalities. However, these approaches often neglect internal consistency within a single modality, particularly when background or environmental changes occur. This lack of consistency leads to a performance drop because the model struggles to detect the same object in different scenes, which reveals a robustness gap. To address this issue, we introduce Contextual Consistency Learning (CCL), a novel framework that integrates two key strategies: Contextual Bootstrapped Data Generation (CBDG) and Contextual Consistency Loss (CCLoss). CBDG functions as a data generation mechanism, producing images that contain the same objects across diverse backgrounds. This is essential because existing datasets alone do not support our CCL framework. The CCLoss further enforces the invariance of object features despite environmental changes, thereby improving the model's robustness in different scenes. These strategies collectively form a unified framework for ensuring contextual consistency within the same modality. Our method achieves state-of-the-art performance, surpassing previous approaches by +16.3 AP on OmniLabel and +14.9 AP on D3. These results demonstrate the importance of enforcing intra-modal consistency, significantly enhancing model generalization in diverse environments. Our code is publicly available at: this https URL.
>
---
#### [new 089] Generation Is Compression: Zero-Shot Video Coding via Stochastic Rectified Flow
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频压缩任务，旨在将预训练生成模型直接作为编解码器使用，解决传统编码与生成模型分离的问题。通过转换确定性流为随机过程，实现高效且灵活的视频压缩。**

- **链接: [https://arxiv.org/pdf/2603.26571](https://arxiv.org/pdf/2603.26571)**

> **作者:** Ziyue Zeng; Xun Su; Haoyuan Liu; Bingyu Lu; Yui Tatsumi; Hiroshi Watanabe
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Existing generative video compression methods use generative models only as post-hoc reconstruction modules atop conventional codecs. We propose \emph{Generative Video Codec} (GVC), a zero-shot framework that turns a pretrained video generative model into the codec itself: the transmitted bitstream directly specifies the generative decoding trajectory, with no retraining required. To enable this, we convert the deterministic rectified-flow ODE of modern video foundation models into an equivalent SDE at inference time, unlocking per-step stochastic injection points for codebook-driven compression. Building on this unified backbone, we instantiate three complementary conditioning strategies -- \emph{Image-to-Video} (I2V) with adaptive tail-frame atom allocation, \emph{Text-to-Video} (T2V) operating at near-zero side information as a pure generative prior, and \emph{First-Last-Frame-to-Video} (FLF2V) with boundary-sharing GOP chaining for dual-anchor temporal control. Together, these variants span a principled trade-off space between spatial fidelity, temporal coherence, and compression efficiency. Experiments on standard benchmarks show that GVC achieves high-quality reconstruction below 0.002\,bpp while supporting flexible bitrate control through a single hyperparameter.
>
---
#### [new 090] Efficient Few-Shot Learning for Edge AI via Knowledge Distillation on MobileViT
- **分类: cs.CV**

- **简介: 该论文属于边缘AI的少样本学习任务，旨在解决低数据环境下模型效率与性能问题。通过知识蒸馏优化MobileViT模型，在保持高准确率的同时显著降低参数和计算量。**

- **链接: [https://arxiv.org/pdf/2603.26145](https://arxiv.org/pdf/2603.26145)**

> **作者:** Shuhei Tsuyuki; Reda Bensaid; Jérémy Morlier; Mathieu Léonardon; Naoya Onizawa; Vincent Gripon; Takahiro Hanyu
>
> **摘要:** Efficient and adaptable deep learning models are an important area of deep learning research, driven by the need for highly efficient models on edge devices. Few-shot learning enables the use of deep learning models in low-data regimes, a capability that is highly sought after in real-world applications where collecting large annotated datasets is costly or impractical. This challenge is particularly relevant in edge scenarios, where connectivity may be limited, low-latency responses are required, or energy consumption constraints are critical. We propose and evaluate a pre-training method for the MobileViT backbone designed for edge computing. Specifically, we employ knowledge distillation, which transfers the generalization ability of a large-scale teacher model to a lightweight student model. This method achieves accuracy improvements of 14% and 6.7% for one-shot and five-shot classification, respectively, on the MiniImageNet benchmark, compared to the ResNet12 baseline, while reducing by 69% the number of parameters and by 88% the computational complexity of the model, in FLOPs. Furthermore, we deployed the proposed models on a Jetson Orin Nano platform and measured power consumption directly at the power supply, showing that the dynamic energy consumption is reduced by 37% with a latency of 2.6 ms. These results demonstrate that the proposed method is a promising and practical solution for deploying few-shot learning models on edge AI hardware.
>
---
#### [new 091] Detailed Geometry and Appearance from Opportunistic Motion
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决稀疏视角下几何与外观恢复的难题。通过利用物体运动提供额外视角，提出联合优化方法和新外观模型，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.26665](https://arxiv.org/pdf/2603.26665)**

> **作者:** Ryosuke Hirai; Kohei Yamashita; Antoine Guédon; Ryo Kawahara; Vincent Lepetit; Ko Nishino
>
> **摘要:** Reconstructing 3D geometry and appearance from a sparse set of fixed cameras is a foundational task with broad applications, yet it remains fundamentally constrained by the limited viewpoints. We show that this bound can be broken by exploiting opportunistic object motion: as a person manipulates an object~(e.g., moving a chair or lifting a mug), the static cameras effectively ``orbit'' the object in its local coordinate frame, providing additional virtual viewpoints. Harnessing this object motion, however, poses two challenges: the tight coupling of object pose and geometry estimation and the complex appearance variations of a moving object under static illumination. We address these by formulating a joint pose and shape optimization using 2D Gaussian splatting with alternating minimization of 6DoF trajectories and primitive parameters, and by introducing a novel appearance model that factorizes diffuse and specular components with reflected directional probing within the spherical harmonics space. Extensive experiments on synthetic and real-world datasets with extremely sparse viewpoints demonstrate that our method recovers significantly more accurate geometry and appearance than state-of-the-art baselines.
>
---
#### [new 092] Evaluating Synthetic Images as Effective Substitutes for Experimental Data in Surface Roughness Classification
- **分类: cs.CV; cond-mat.mtrl-sci**

- **简介: 该论文属于表面粗糙度分类任务，旨在解决实验数据不足和成本高的问题。通过使用生成的合成图像补充真实数据，提升分类效果与数据效率。**

- **链接: [https://arxiv.org/pdf/2603.25765](https://arxiv.org/pdf/2603.25765)**

> **作者:** Binwei Chen; Huachao Leng; Chi Yeung Mang; Tsz Wai Cheung; Yanhua Chen; Wai Keung Anthony Loh; Chi Ho Wong; Chak Yin Tang
>
> **摘要:** Hard coatings play a critical role in industry, with ceramic materials offering outstanding hardness and thermal stability for applications that demand superior mechanical performance. However, deploying artificial intelligence (AI) for surface roughness classification is often constrained by the need for large labeled datasets and costly high-resolution imaging equipment. In this study, we explore the use of synthetic images, generated with Stable Diffusion XL, as an efficient alternative or supplement to experimentally acquired data for classifying ceramic surface roughness. We show that augmenting authentic datasets with generative images yields test accuracies comparable to those obtained using exclusively experimental images, demonstrating that synthetic images effectively reproduce the structural features necessary for classification. We further assess method robustness by systematically varying key training hyperparameters (epoch count, batch size, and learning rate), and identify configurations that preserve performance while reducing data requirements. Our results indicate that generative AI can substantially improve data efficiency and reliability in materials-image classification workflows, offering a practical route to lower experimental cost, accelerate model development, and expand AI applicability in materials engineering.
>
---
#### [new 093] Mitigating the Reasoning Tax in Vision-Language Fine-Tuning with Input-Adaptive Depth Aggregation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决微调中推理能力下降的问题。通过引入输入自适应的跨深度聚合机制，提升模型推理与感知性能。**

- **链接: [https://arxiv.org/pdf/2603.26330](https://arxiv.org/pdf/2603.26330)**

> **作者:** Yiming Ren; Yujiu Yang; Junjie Wang
>
> **摘要:** Supervised fine-tuning (SFT) on visual instruction data often improves perceptual capabilities in vision-language models (VLMs) while degrading reasoning performance, creating a persistent reasoning tax during post-training. We investigate whether this degradation is related to disrupted access to depth-wise representations, and find that even fixed cross-depth aggregation substantially restores reasoning, suggesting that preserved cross-depth access is an important missing factor in VLM fine-tuning. Building on this observation, we propose Input-Adaptive Depth Aggregation (IADA), a lightweight mechanism that makes cross-depth retrieval input-adaptive, modality-aware, and efficiently parameterized through a low-rank bottleneck. On Qwen3-VL-2B, IADA improves the average reasoning score by 9.5 points and the average perception score by $3.3$ points over LoRA-only fine-tuning with only 0.14M additional parameters, with the strongest gains appearing in parameter-efficient low-rank settings.
>
---
#### [new 094] Think over Trajectories: Leveraging Video Generation to Reconstruct GPS Trajectories from Cellular Signaling
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究Sig2GPS任务，解决从蜂窝信号重建高精度GPS轨迹的问题。通过将问题转化为图像到视频生成，利用地图视觉域直接生成连续路径。**

- **链接: [https://arxiv.org/pdf/2603.26610](https://arxiv.org/pdf/2603.26610)**

> **作者:** Ruixing Zhang; Hanzhang Jiang; Leilei Sun; Liangzhe Han; Jibin Wang; Weifeng Lv
>
> **摘要:** Mobile devices continuously interact with cellular base stations, generating massive volumes of signaling records that provide broad coverage for understanding human mobility. However, such records offer only coarse location cues (e.g., serving-cell identifiers) and therefore limit their direct use in applications that require high-precision GPS trajectories. This paper studies the Sig2GPS problem: reconstructing GPS trajectories from cellular signaling. Inspired by domain experts often lay the signaling trace on the map and sketch the corresponding GPS route, unlike conventional solutions that rely on complex multi-stage engineering pipelines or regress coordinates, Sig2GPS is reframed as an image-to-video generation task that directly operates in the map-visual domain: signaling traces are rendered on a map, and a video generation model is trained to draw a continuous GPS path. To support this paradigm, a paired signaling-to-trajectory video dataset is constructed to fine-tune an open-source video model, and a trajectory-aware reinforcement learning-based optimization method is introduced to improve generation fidelity via rewards. Experiments on large-scale real-world datasets show substantial improvements over strong engineered and learning-based baselines, while additional results on next GPS prediction indicate scalability and cross-city transferability. Overall, these results suggest that map-visual video generation provides a practical interface for trajectory data mining by enabling direct generation and refinement of continuous paths under map constraints.
>
---
#### [new 095] Few Shots Text to Image Retrieval: New Benchmarking Dataset and Optimization Methods
- **分类: cs.CV**

- **简介: 该论文提出FSIR任务及基准数据集FSIR-BD，解决图像检索中组合查询和分布外样本的问题。通过少量参考样例优化检索性能。**

- **链接: [https://arxiv.org/pdf/2603.25891](https://arxiv.org/pdf/2603.25891)**

> **作者:** Ofer Idan; Vladi Vexler; Gil Lederman; Dima Sivov; Aviad Cohen Zada; Shir Niego Komforti
>
> **摘要:** Pre-trained vision-language models (VLMs) excel in multimodal tasks, commonly encoding images as embedding vectors for storage in databases and retrieval via approximate nearest neighbor search (ANNS). However, these models struggle with compositional queries and out-of-distribution (OOD) image-text pairs. Inspired by human cognition's ability to learn from minimal examples, we address this performance gap through few-shot learning approaches specifically designed for image retrieval. We introduce the Few-Shot Text-to-Image Retrieval (FSIR) task and its accompanying benchmark dataset, FSIR-BD - the first to explicitly target image retrieval by text accompanied by reference examples, focusing on the challenging compositional and OOD queries. The compositional part is divided to urban scenes and nature species, both in specific situations or with distinctive features. FSIR-BD contains 38,353 images and 303 queries, with 82% comprising the test corpus (averaging per query 37 positives, ground truth matches, and significant number of hard negatives) and 18% forming the few-shot reference corpus (FSR) of exemplar positive and hard negative images. Additionally, we propose two novel retrieval optimization methods leveraging single shot or few shot reference examples in the FSR to improve performance. Both methods are compatible with any pre-trained image encoder, making them applicable to existing large-scale environments. Our experiments demonstrate that: (1) FSIR-BD provides a challenging benchmark for image retrieval; and (2) our optimization methods outperform existing baselines as measured by mean Average Precision (mAP). Further research into FSIR optimization methods will help narrow the gap between machine and human-level understanding, particularly for compositional reasoning from limited examples.
>
---
#### [new 096] Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态媒体验证任务，旨在解决复杂虚假信息检测中特征稀释问题。提出MaLSF框架，通过双向跨模态验证和语义聚合，有效定位局部语义冲突。**

- **链接: [https://arxiv.org/pdf/2603.26052](https://arxiv.org/pdf/2603.26052)**

> **作者:** Zizhao Chen; Ping Wei; Ziyang Ren; Huan Li; Xiangru Yin
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** As multimodal misinformation becomes more sophisticated, its detection and grounding are crucial. However, current multimodal verification methods, relying on passive holistic fusion, struggle with sophisticated misinformation. Due to 'feature dilution,' global alignments tend to average out subtle local semantic inconsistencies, effectively masking the very conflicts they are designed to find. We introduce MaLSF (Mask-aware Local Semantic Fusion), a novel framework that shifts the paradigm to active, bidirectional verification, mimicking human cognitive cross-referencing. MaLSF utilizes mask-label pairs as semantic anchors to bridge pixels and words. Its core mechanism features two innovations: 1) a Bidirectional Cross-modal Verification (BCV) module that acts as an interrogator, using parallel query streams (Text-as-Query and Image-as-Query) to explicitly pinpoint conflicts; and 2) a Hierarchical Semantic Aggregation (HSA) module that intelligently aggregates these multi-granularity conflict signals for task-specific reasoning. In addition, to extract fine-grained mask-label pairs, we introduce a set of diverse mask-label pair extraction parsers. MaLSF achieves state-of-the-art performance on both the DGM4 and multimodal fake news detection tasks. Extensive ablation studies and visualization results further verify its effectiveness and interpretability.
>
---
#### [new 097] DenseSwinV2: Channel Attentive Dual Branch CNN Transformer Learning for Cassava Leaf Disease Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类任务，旨在解决木薯叶片疾病识别问题。提出DenseSwinV2框架，结合密集卷积和定制SwinV2模型，提升分类精度。**

- **链接: [https://arxiv.org/pdf/2603.25935](https://arxiv.org/pdf/2603.25935)**

> **作者:** Shah Saood; Saddam Hussain Khan
>
> **备注:** 30 Pages, 12 Figures, 3 Tables
>
> **摘要:** This work presents a new Hybrid Dense SwinV2, a two-branch framework that jointly leverages densely connected convolutional features and hierarchical customized Swin Transformer V2 (SwinV2) representations for cassava disease classification. The proposed framework captures high resolution local features through its DenseNet branch, preserving the fine structural cues and also allowing for effective gradient flow. Concurrently, the customized SwinV2 models global contextual dependencies through the idea of shifted-window self attention, which enables the capture of long range interactions critical in distinguishing between visually similar lesions. Moreover, an attention channel-squeeze module is employed for each CNN Transformer stream independently to emphasize discriminative disease related responses and suppress redundant or background driven activations. Finally, these discriminative channels are fused to achieve refined representations from the dense local and SwinV2 global correlated strengthened feature maps, respectively. The proposed Dense SwinV2 utilized a public cassava leaf disease dataset of 31000 images, comprised of five diseases, including brown streak, mosaic, green mottle, bacterial blight, and normal leaf conditions. The proposed Dense SwinV2 demonstrates a significant classification accuracy of 98.02 percent with an F1 score of 97.81 percent, outperforming well-established convolutional and transformer models. These results underline the fact that Hybrid Dense SwinV2 offers robustness and practicality in the field level diagnosis of cassava disease and real world challenges related to occlusion, noise, and complex backgrounds.
>
---
#### [new 098] THFM: A Unified Video Foundation Model for 4D Human Perception and Beyond
- **分类: cs.CV**

- **简介: 该论文提出THFM，一个统一的视频基础模型，用于4D人体感知。解决多任务感知问题，通过单一架构处理密集和稀疏任务，无需真实数据训练。**

- **链接: [https://arxiv.org/pdf/2603.25892](https://arxiv.org/pdf/2603.25892)**

> **作者:** Letian Wang; Andrei Zanfir; Eduard Gabriel Bazavan; Misha Andriluka; Cristian Sminchisescu
>
> **摘要:** We present THFM, a unified video foundation model for human-centric perception that jointly addresses dense tasks (depth, normals, segmentation, dense pose) and sparse tasks (2d/3d keypoint estimation) within a single architecture. THFM is derived from a pretrained text-to-video diffusion model, repurposed as a single-forward-pass perception model and augmented with learnable tokens for sparse predictions. Modulated by the text prompt, our single unified model is capable of performing various perception tasks. Crucially, our model is on-par or surpassing state-of-the-art specialized models on a variety of benchmarks despite being trained exclusively on synthetic data (i.e.~without training on real-world or benchmark specific data). We further highlight intriguing emergent properties of our model, which we attribute to the underlying diffusion-based video representation. For example, our model trained on videos with a single human in the scene generalizes to multiple humans and other object classes such as anthropomorphic characters and animals -- a capability that hasn't been demonstrated in the past.
>
---
#### [new 099] Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于AI内容水印任务，旨在解决现有方法无法精确恢复水印的问题。提出Gaussian Shannon框架，通过通信信道理论实现精准水印嵌入与恢复。**

- **链接: [https://arxiv.org/pdf/2603.26167](https://arxiv.org/pdf/2603.26167)**

> **作者:** Yi Zhang; Hongbo Huang; Liang-Jie Zhang
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** Diffusion models generate high-quality images but pose serious risks like copyright violation and disinformation. Watermarking is a key defense for tracing and authenticating AI-generated content. However, existing methods rely on threshold-based detection, which only supports fuzzy matching and cannot recover structured watermark data bit-exactly, making them unsuitable for offline verification or applications requiring lossless metadata (e.g., licensing instructions). To address this problem, in this paper, we propose Gaussian Shannon, a watermarking framework that treats the diffusion process as a noisy communication channel and enables both robust tracing and exact bit recovery. Our method embeds watermarks in the initial Gaussian noise without fine-tuning or quality loss. We identify two types of channel interference, namely local bit flips and global stochastic distortions, and design a cascaded defense combining error-correcting codes and majority voting. This ensures reliable end-to-end transmission of semantic payloads. Experiments across three Stable Diffusion variants and seven perturbation types show that Gaussian Shannon achieves state-of-the-art bit-level accuracy while maintaining a high true positive rate, enabling trustworthy rights attribution in real-world deployment. The source code have been made available at: this https URL
>
---
#### [new 100] OVI-MAP:Open-Vocabulary Instance-Semantic Mapping
- **分类: cs.CV**

- **简介: 该论文属于开放词汇实例语义建图任务，旨在解决复杂环境中实时、稳定且灵活的语义映射问题。通过解耦实例重建与语义推理，实现零样本语义标注与高效处理。**

- **链接: [https://arxiv.org/pdf/2603.26541](https://arxiv.org/pdf/2603.26541)**

> **作者:** Zilong Deng; Federico Tombari; Marc Pollefeys; Johanna Wald; Daniel Barath
>
> **摘要:** Incremental open-vocabulary 3D instance-semantic mapping is essential for autonomous agents operating in complex everyday environments. However, it remains challenging due to the need for robust instance segmentation, real-time processing, and flexible open-set reasoning. Existing methods often rely on the closed-set assumption or dense per-pixel language fusion, which limits scalability and temporal consistency. We introduce OVI-MAP that decouples instance reconstruction from semantic inference. We propose to build a class-agnostic 3D instance map that is incrementally constructed from RGB-D input, while semantic features are extracted only from a small set of automatically selected views using vision-language models. This design enables stable instance tracking and zero-shot semantic labeling throughout online exploration. Our system operates in real time and outperforms state-of-the-art open-vocabulary mapping baselines on standard benchmarks.
>
---
#### [new 101] Drive-Through 3D Vehicle Exterior Reconstruction via Dynamic-Scene SfM and Distortion-Aware Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决复杂场景下车辆外表面高保真重建问题，通过动态场景SfM和畸变感知高斯喷射方法实现高质量模型生成。**

- **链接: [https://arxiv.org/pdf/2603.26638](https://arxiv.org/pdf/2603.26638)**

> **作者:** Nitin Kulkarni; Akhil Devarashetti; Charlie Cluss; Livio Forte; Philip Schneider; Chunming Qiao; Alina Vereshchaka
>
> **备注:** 8 pages, 7 figures, Submitted to IEEE IROS 2026 (under review)
>
> **摘要:** High-fidelity 3D reconstruction of vehicle exteriors improves buyer confidence in online automotive marketplaces, but generating these models in cluttered dealership drive-throughs presents severe technical challenges. Unlike static-scene photogrammetry, this setting features a dynamic vehicle moving against heavily cluttered, static backgrounds. This problem is further compounded by wide-angle lens distortion, specular automotive paint, and non-rigid wheel rotations that violate classical epipolar constraints. We propose an end-to-end pipeline utilizing a two-pillar camera rig. First, we resolve dynamic-scene ambiguities by coupling SAM 3 for instance segmentation with motion-gating to cleanly isolate the moving vehicle, explicitly masking out non-rigid wheels to enforce strict epipolar geometry. Second, we extract robust correspondences directly on raw, distorted 4K imagery using the RoMa v2 learned matcher guided by semantic confidence masks. Third, these matches are integrated into a rig-aware SfM optimization that utilizes CAD-derived relative pose priors to eliminate scale drift. Finally, we use a distortion-aware 3D Gaussian Splatting framework (3DGUT) coupled with a stochastic Markov Chain Monte Carlo (MCMC) densification strategy to render reflective surfaces. Evaluations on 25 real-world vehicles across 10 dealerships demonstrate that our full pipeline achieves a PSNR of 28.66 dB, an SSIM of 0.89, and an LPIPS of 0.21 on held-out views, representing a 3.85 dB improvement over standard 3D-GS, delivering inspection-grade interactive 3D models without controlled studio infrastructure.
>
---
#### [new 102] PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery
- **分类: cs.CV**

- **简介: 该论文属于手部运动恢复任务，解决现有方法缺乏物理一致性的问题。提出物理感知的扩散框架，通过动态残差和方差估计提升运动的物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.26068](https://arxiv.org/pdf/2603.26068)**

> **作者:** Elkhan Ismayilzada; Yufei Zhang; Zijun Cui
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Significant advancements made in reconstructing hands from images have delivered accurate single-frame estimates, yet they often lack physics consistency and provide no notion of how confidently the motion satisfies physics. In this paper, we propose a novel physics-aware conditional diffusion framework that refines noisy pose sequences into physically plausible hand motion while estimating the physics variance in motion estimates. Building on a MeshCNN-Transformer backbone, we formulate Euler-Lagrange dynamics for articulated hands. Unlike prior works that enforce zero residuals, we treat the resulting dynamic residuals as virtual observables to more effectively integrate physics. Through a last-layer Laplace approximation, our method produces per-joint, per-time variances that measure physics consistency and offers interpretable variance maps indicating where physical consistency weakens. Experiments on two well-known hand datasets show consistent gains over strong image-based initializations and competitive video-based methods. Qualitative results confirm that our variance estimations are aligned with the physical plausibility of the motion in image-based estimates.
>
---
#### [new 103] Image-based Quantification of Postural Deviations on Patients with Cervical Dystonia: A Machine Learning Approach Using Synthetic Training Data
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决颈椎 dystonia 患者姿势偏差客观评估问题。通过机器学习结合合成数据，开发了自动化的头部姿态评估工具。**

- **链接: [https://arxiv.org/pdf/2603.26444](https://arxiv.org/pdf/2603.26444)**

> **作者:** Roland Stenger; Sebastian Löns; Nele Brügge; Feline Hamami; Alexander Münchau; Theresa Paulus; Anne Weissbach; Tatiana Usnich; Max Borsche; Martje G. Pauly; Lara M. Lange; Markus A. Hobert; Rebecca Herzog; Ana Luísa de Almeida Marcelino; Tina Mainka; Friederike Schumann; Lukas L. Goede; Johanna Reimer; Julienne Haas; Jos Becktepe; Alexander Baumann; Robin Wolke; Chi Wang Ip; Thorsten Odorfer; Daniel Zeller; Lisa Harder-Rauschenberger; John-Ih Lee; Philipp Albrecht; Tristan Kölsche; Joachim K. Krauss; Johanna M. Nagel; Joachim Runge; Johanna Doll-Lee; Simone Zittel; Kai Grimm; Pawel Tacik; André Lee; Tobias Bäumer; Sebastian Fudickar
>
> **摘要:** Cervical dystonia (CD) is the most common form of dystonia, yet current assessment relies on subjective clinical rating scales, such as the Toronto Western Spasmodic Torticollis Rating Scale (TWSTRS), which requires expertise, is subjective and faces low inter-rater reliability some items of the score. To address the lack of established objective tools for monitoring disease severity and treatment response, this study validates an automated image-based head pose and shift estimation system for patients with CD. We developed an assessment tool that combines a pretrained head-pose estimation algorithm for rotational symptoms with a deep learning model trained exclusively on ~16,000 synthetic avatar images to evaluate rare translational symptoms, specifically lateral shift. This synthetic data approach overcomes the scarcity of clinical training examples. The system's performance was validated in a multicenter study by comparing its predicted scores against the consensus ratings of 20 clinical experts using a dataset of 100 real patient images and 100 labeled synthetic avatars. The automated system demonstrated strong agreement with expert clinical ratings for rotational symptoms, achieving high correlations for torticollis (r=0.91), laterocollis (r=0.81), and anteroretrocollis (r=0.78). For lateral shift, the tool achieved a moderate correlation (r=0.55) with clinical ratings and demonstrated higher accuracy than human raters in controlled benchmark tests on avatars. By leveraging synthetic training data to bridge the clinical data gap, this model successfully generalizes to real-world patients, providing a validated, objective tool for CD postural assessment that can enable standardized clinical decision-making and trial evaluation.
>
---
#### [new 104] Polarization-Based Eye Tracking with Personalized Siamese Architectures
- **分类: cs.CV**

- **简介: 该论文属于眼动追踪任务，解决个体差异导致的校准问题。通过Siamese架构实现个性化，减少校准样本并提升精度。**

- **链接: [https://arxiv.org/pdf/2603.25889](https://arxiv.org/pdf/2603.25889)**

> **作者:** Beyza Kalkanli; Tom Bu; Mahsa Shakeri; Alexander Fix; Dave Stronks; Dmitri Model; Mantas Žurauskas
>
> **备注:** Accepted to ETRA 2026 as full paper
>
> **摘要:** Head-mounted devices integrated with eye tracking promise a solution for natural human-computer interaction. However, they typically require per-user calibration for optimal performance due to inter-person variability. A differential personalization approach using Siamese architectures learns relative gaze displacements and reconstructs absolute gaze from a small set of calibration frames. In this paper, we benchmark Siamese personalization on polarization-enabled eye tracking. For benchmarking, we use a 338-subject dataset captured with a polarization-sensitive camera and 850 nm illumination. We achieve performance comparable to linear calibration with 10-fold fewer samples. Using polarization inputs for Siamese personalization reduces gaze error by up to 12% compared to near-infrared (NIR)-based inputs. Combining Siamese personalization with linear calibration yields further improvements of up to 13% over a linearly calibrated baseline. These results establish Siamese personalization as a practical approach enabling accurate eye tracking.
>
---
#### [new 105] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文属于开放词汇目标检测任务，旨在解决伪装物体检测中视觉特征与背景相似导致的识别难题。通过设计动态聚焦方法和文本融合策略提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.26109](https://arxiv.org/pdf/2603.26109)**

> **作者:** Jiaming Liang; Yifeng Zhan; Chunlin Liu; Weihua Zheng; Bingye Peng; Qiwei Liang; Boyang Cai; Xiaochun Mai; Qiang Nie
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Open-vocabulary object detection (OVOD) aims to detect known and unknown objects in the open world by leveraging text prompts. Benefiting from the emergence of large-scale vision--language pre-trained models, OVOD has demonstrated strong zero-shot generalization capabilities. However, when dealing with camouflaged objects, the detector often fails to distinguish and localize objects because the visual features of the objects and the background are highly similar. To bridge this gap, we construct a benchmark named OVCOD-D by augmenting carefully selected camouflaged object images with fine-grained textual descriptions. Due to the limited scale of available camouflaged object datasets, we adopt detectors pre-trained on large-scale object detection datasets as our baseline methods, as they possess stronger zero-shot generalization ability. In the specificity-aware sub-descriptions generated by multimodal large models, there still exist confusing and overly decorative modifiers. To mitigate such interference, we design a sub-description principal component contrastive fusion strategy that reduces noisy textual components. Furthermore, to address the challenge that the visual features of camouflaged objects are highly similar to those of their surrounding environment, we propose a specificity-guided regional weak alignment and dynamic focusing method, which aims to strengthen the detector's ability to discriminate camouflaged objects from background. Under the open-set evaluation setting, the proposed method achieves an AP of 56.4 on the OVCOD-D benchmark.
>
---
#### [new 106] InstaVSR: Taming Diffusion for Efficient and Temporally Consistent Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于视频超分辨率任务，解决扩散模型在视频中的时间不稳定和计算成本高的问题。提出InstaVSR框架，结合简化扩散、时序正则化和对抗学习，实现高效且稳定的效果。**

- **链接: [https://arxiv.org/pdf/2603.26134](https://arxiv.org/pdf/2603.26134)**

> **作者:** Jintong Hu; Bin Chen; Zhenyu Hu; Jiayue Liu; Guo Wang; Lu Qi
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Video super-resolution (VSR) seeks to reconstruct high-resolution frames from low-resolution inputs. While diffusion-based methods have substantially improved perceptual quality, extending them to video remains challenging for two reasons: strong generative priors can introduce temporal instability, and multi-frame diffusion pipelines are often too expensive for practical deployment. To address both challenges simultaneously, we propose InstaVSR, a lightweight diffusion framework for efficient video super-resolution. InstaVSR combines three ingredients: (1) a pruned one-step diffusion backbone that removes several costly components from conventional diffusion-based VSR pipelines, (2) recurrent training with flow-guided temporal regularization to improve frame-to-frame stability, and (3) dual-space adversarial learning in latent and pixel spaces to preserve perceptual quality after backbone simplification. On an NVIDIA RTX 4090, InstaVSR processes a 30-frame video at 2K$\times$2K resolution in under one minute with only 7 GB of memory usage, substantially reducing the computational cost compared to existing diffusion-based methods while maintaining favorable perceptual quality with significantly smoother temporal transitions.
>
---
#### [new 107] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文提出GUIDE基准，用于评估AI在开放性GUI任务中理解用户意图和提供帮助的能力。旨在解决用户意图识别与协作辅助问题。**

- **链接: [https://arxiv.org/pdf/2603.25864](https://arxiv.org/pdf/2603.25864)**

> **作者:** Saelyne Yang; Jaesang Yu; Yi-Hao Peng; Kevin Qinghong Lin; Jae Won Cho; Yale Song; Juho Kim
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Graphical User Interface (GUI) agents have the potential to assist users in interacting with complex software (e.g., PowerPoint, Photoshop). While prior research has primarily focused on automating user actions through clicks and keystrokes, this paradigm overlooks human intention, where users value the ability to explore, iterate, and refine their ideas while maintaining agency. To move beyond automation and toward collaboration, GUI agents must understand what users are doing and why. We introduce GUIDE (GUI User Intent Detection Evaluation), a benchmark that evaluates AI models on their ability to perceive user behavior, infer intent, and provide assistance in open-ended GUI tasks. GUIDE consists of 67.5 hours of screen recordings from 120 novice user demonstrations with think-aloud narrations, across 10 software. GUIDE defines three tasks - (i) Behavior State Detection, (ii) Intent Prediction, and (iii) Help Prediction that test a model's ability to recognize behavior state, reason about goals, and decide when and how to help. Evaluations across eight state-of-the-art multimodal models reveal that all models struggled, achieving only 44.6% and 55.0% accuracy on behavior state and help prediction. However, providing user context significantly improved the performance, raising help prediction by up to 50.2pp, highlighting the critical role of structured user understanding in effective assistance. Our dataset is available at this https URL.
>
---
#### [new 108] PhysVid: Physics Aware Local Conditioning for Generative Video Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生成视频任务，旨在解决生成视频违背物理规律的问题。通过引入物理感知的局部条件引导，提升视频的物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.26285](https://arxiv.org/pdf/2603.26285)**

> **作者:** Saurabh; Pathak; Elahe Arani; Mykola Pechenizkiy; Bahram Zonooz
>
> **备注:** Accepted for CVPR 2026
>
> **摘要:** Generative video models achieve high visual fidelity but often violate basic physical principles, limiting reliability in real-world settings. Prior attempts to inject physics rely on conditioning: frame-level signals are domain-specific and short-horizon, while global text prompts are coarse and noisy, missing fine-grained dynamics. We present PhysVid, a physics-aware local conditioning scheme that operates over temporally contiguous chunks of frames. Each chunk is annotated with physics-grounded descriptions of states, interactions, and constraints, which are fused with the global prompt via chunk-aware cross-attention during training. At inference, we introduce negative physics prompts (descriptions of locally relevant law violations) to steer generation away from implausible trajectories. On VideoPhy, PhysVid improves physical commonsense scores by $\approx 33\%$ over baseline video generators, and by up to $\approx 8\%$ on VideoPhy2. These results show that local, physics-aware guidance substantially increases physical plausibility in generative video and marks a step toward physics-grounded video models.
>
---
#### [new 109] MuDD: A Multimodal Deception Detection Dataset and GSR-Guided Progressive Distillation for Non-Contact Deception Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于欺骗检测任务，旨在解决非接触模态中欺骗线索不稳定的问题。通过构建MuDD数据集并提出GPD框架，利用GSR引导跨模态知识蒸馏，提升欺骗检测性能。**

- **链接: [https://arxiv.org/pdf/2603.26064](https://arxiv.org/pdf/2603.26064)**

> **作者:** Peiyuan Jiang; Yao Liu; Yanglei Gan; Jiaye Yang; Lu Liu; Daibing Yao; Qiao Liu
>
> **摘要:** Non-contact automatic deception detection remains challenging because visual and auditory deception cues often lack stable cross-subject patterns. In contrast, galvanic skin response (GSR) provides more reliable physiological cues and has been widely used in contact-based deception detection. In this work, we leverage stable deception-related knowledge in GSR to guide representation learning in non-contact modalities through cross-modal knowledge distillation. A key obstacle, however, is the lack of a suitable dataset for this setting. To address this, we introduce MuDD, a large-scale Multimodal Deception Detection dataset containing recordings from 130 participants over 690 minutes. In addition to video, audio, and GSR, MuDD also provides Photoplethysmography, heart rate, and personality traits, supporting broader scientific studies of deception. Based on this dataset, we propose GSR-guided Progressive Distillation (GPD), a cross-modal distillation framework for mitigating the negative transfer caused by the large modality mismatch between GSR and non-contact signals. The core innovation of GPD is the integration of progressive feature-level and digit-level distillation with dynamic routing, which allows the model to adaptively determine how teacher knowledge should be transferred during training, leading to more stable cross-modal knowledge transfer. Extensive experiments and visualizations show that GPD outperforms existing methods and achieves state-of-the-art performance on both deception detection and concealed-digit identification.
>
---
#### [new 110] FAST3DIS: Feed-forward Anchored Scene Transformer for 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文提出FAST3DIS，解决3D实例分割问题，通过端到端Transformer架构避免后处理聚类，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.25993](https://arxiv.org/pdf/2603.25993)**

> **作者:** Changyang Li; Xueqing Huang; Shin-Fang Chng; Huangying Zhan; Qingan Yan; Yi Xu
>
> **摘要:** While recent feed-forward 3D reconstruction models provide a strong geometric foundation for scene understanding, extending them to 3D instance segmentation typically relies on a disjointed "lift-and-cluster" paradigm. Grouping dense pixel-wise embeddings via non-differentiable clustering scales poorly with the number of views and disconnects representation learning from the final segmentation objective. In this paper, we present a Feed-forward Anchored Scene Transformer for 3D Instance Segmentation (FAST3DIS), an end-to-end approach that effectively bypasses post-hoc clustering. We introduce a 3D-anchored, query-based Transformer architecture built upon a foundational depth backbone, adapted efficiently to learn instance-specific semantics while retaining its zero-shot geometric priors. We formulate a learned 3D anchor generator coupled with an anchor-sampling cross-attention mechanism for view-consistent 3D instance segmentation. By projecting 3D object queries directly into multi-view feature maps, our method samples context efficiently. Furthermore, we introduce a dual-level regularization strategy, that couples multi-view contrastive learning with a dynamically scheduled spatial overlap penalty to explicitly prevent query collisions and ensure precise instance boundaries. Experiments on complex indoor 3D datasets demonstrate that our approach achieves competitive segmentation accuracy with significantly improved memory scalability and inference speed over state-of-the-art clustering-based methods.
>
---
#### [new 111] Meta-Learned Adaptive Optimization for Robust Human Mesh Recovery with Uncertainty-Aware Parameter Updates
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人体网格重建任务，解决单图重建中的深度模糊和泛化问题。通过元学习和自适应优化，提升初始化质量并减少计算开销，实现更准确和鲁棒的 mesh 恢复。**

- **链接: [https://arxiv.org/pdf/2603.26447](https://arxiv.org/pdf/2603.26447)**

> **作者:** Shaurjya Mandal; Nutan Sharma; John Galeotti
>
> **摘要:** Human mesh recovery from single images remains challenging due to inherent depth ambiguity and limited generalization across domains. While recent methods combine regression and optimization approaches, they struggle with poor initialization for test-time refinement and inefficient parameter updates during optimization. We propose a novel meta-learning framework that trains models to produce optimization-friendly initializations while incorporating uncertainty-aware adaptive updates during test-time refinement. Our approach introduces three key innovations: (1) a meta-learning strategy that simulates test-time optimization during training to learn better parameter initializations, (2) a selective parameter caching mechanism that identifies and freezes converged joints to reduce computational overhead, and (3) distribution-based adaptive updates that sample parameter changes from learned distributions, enabling robust exploration while quantifying uncertainty. Additionally, we employ stochastic approximation techniques to handle intractable gradients in complex loss landscapes. Extensive experiments on standard benchmarks demonstrate that our method achieves state-of-the-art performance, reducing MPJPE by 10.3 on 3DPW and 8.0 on Human3.6M compared to strong baselines. Our approach shows superior domain adaptation capabilities with minimal performance degradation across different environmental conditions, while providing meaningful uncertainty estimates that correlate with actual prediction errors. Combining meta-learning and adaptive optimization enables accurate mesh recovery and robust generalization to challenging scenarios.
>
---
#### [new 112] Unlabeled Cross-Center Automatic Analysis for TAAD: An Integrated Framework from Segmentation to Clinical Features
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，解决跨机构TAAD临床特征提取问题。提出无监督域适应框架，无需目标域标注即可实现准确分割与特征提取。**

- **链接: [https://arxiv.org/pdf/2603.26019](https://arxiv.org/pdf/2603.26019)**

> **作者:** Mengdi Liu; Qiang Li; Weizhi Nie; Shaopeng Zhang; Yuting Su
>
> **摘要:** Type A Aortic Dissection (TAAD) is a life-threatening cardiovascular emergency that demands rapid and precise preoperative evaluation. While key anatomical and pathological features are decisive for surgical planning, current research focuses predominantly on improving segmentation accuracy, leaving the reliable, quantitative extraction of clinically actionable features largely under-explored. Furthermore, constructing comprehensive TAAD datasets requires labor-intensive, expert level pixel-wise annotations, which is impractical for most clinical institutions. Due to significant domain shift, models trained on a single center dataset also suffer from severe performance degradation during cross-institutional deployment. This study addresses a clinically critical challenge: the accurate extraction of key TAAD clinical features during cross-institutional deployment in the total absence of target-domain annotations. To this end, we propose an unsupervised domain adaptation (UDA)-driven framework for the automated extraction of TAAD clinical features. The framework leverages limited source-domain labels while effectively adapting to unlabeled data from target domains. Tailored for real-world emergency workflows, our framework aims to achieve stable cross-institutional multi-class segmentation, reliable and quantifiable clinical feature extraction, and practical deployability independent of high-cost annotations. Extensive experiments demonstrate that our method significantly improves cross-domain segmentation performance compared to existing state-of-the-art approaches. More importantly, a reader study involving multiple cardiovascular surgeons confirms that the automatically extracted clinical features provide meaningful assistance for preoperative assessment, highlighting the practical utility of the proposed end-to-end segmentation-to-feature pipeline.
>
---
#### [new 113] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis
- **分类: cs.CV**

- **简介: 该论文针对内镜视频分析任务，解决标注数据不足与运动偏差问题，提出FPRL框架，通过静态与上下文语义分层学习提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.25778](https://arxiv.org/pdf/2603.25778)**

> **作者:** Yuan Zhang; Sihao Dou; Kai Hu; Shuhua Deng; Chunhong Cao; Fen Xiao; Xieping Gao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Endoscopic video analysis is essential for early gastrointestinal screening but remains hindered by limited high-quality annotations. While self-supervised video pre-training shows promise, existing methods developed for natural videos prioritize dense spatio-temporal modeling and exhibit motion bias, overlooking the static, structured semantics critical to clinical decision-making. To address this challenge, we propose Focus-to-Perceive Representation Learning (FPRL), a cognition-inspired hierarchical framework that emulates clinical examination. FPRL first focuses on intra-frame lesion-centric regions to learn static semantics, and then perceives their evolution across frames to model contextual semantics. To achieve this, FPRL employs a hierarchical semantic modeling mechanism that explicitly distinguishes and collaboratively learns both types of semantics. Specifically, it begins by capturing static semantics via teacher-prior adaptive masking (TPAM) combined with multi-view sparse sampling. This approach mitigates redundant temporal dependencies and enables the model to concentrate on lesion-related local semantics. Following this, contextual semantics are derived through cross-view masked feature completion (CVMFC) and attention-guided temporal prediction (AGTP). These processes establish cross-view correspondences and effectively model structured inter-frame evolution, thereby reinforcing temporal semantic continuity while preserving global contextual integrity. Extensive experiments on 11 endoscopic video datasets show that FPRL achieves superior performance across diverse downstream tasks, demonstrating its effectiveness in endoscopic video representation learning. The code is available at this https URL.
>
---
#### [new 114] World Reasoning Arena
- **分类: cs.CV**

- **简介: 该论文提出WR-Arena基准，用于评估世界模型的模拟能力，解决现有基准过于狭窄的问题。工作包括构建多维评估体系和多样化数据集，推动更智能的环境模拟发展。**

- **链接: [https://arxiv.org/pdf/2603.25887](https://arxiv.org/pdf/2603.25887)**

> **作者:** PAN Team Institute of Foundation Models; Qiyue Gao; Kun Zhou; Jiannan Xiang; Zihan Liu; Dequan Yang; Junrong Chen; Arif Ahmad; Cong Zeng; Ganesh Bannur; Xinqi Huang; Zheqi Liu; Yi Gu; Yichi Yang; Guangyi Liu; Zhiting Hu; Zhengzhong Liu; Eric Xing
>
> **摘要:** World models (WMs) are intended to serve as internal simulators of the real world that enable agents to understand, anticipate, and act upon complex environments. Existing WM benchmarks remain narrowly focused on next-state prediction and visual fidelity, overlooking the richer simulation capabilities required for intelligent behavior. To address this gap, we introduce WR-Arena, a comprehensive benchmark for evaluating WMs along three fundamental dimensions of next world simulation: (i) Action Simulation Fidelity, the ability to interpret and follow semantically meaningful, multi-step instructions and generate diverse counterfactual rollouts; (ii) Long-horizon Forecast, the ability to sustain accurate, coherent, and physically plausible simulations across extended interactions; and (iii) Simulative Reasoning and Planning, the ability to support goal-directed reasoning by simulating, comparing, and selecting among alternative futures in both structured and open-ended environments. We build a task taxonomy and curate diverse datasets designed to probe these capabilities, moving beyond single-turn and perceptual evaluations. Through extensive experiments with state-of-the-art WMs, our results expose a substantial gap between current models and human-level hypothetical reasoning, and establish WR-Arena as both a diagnostic tool and a guideline for advancing next-generation world models capable of robust understanding, forecasting, and purposeful action. The code is available at this https URL.
>
---
#### [new 115] SkinGPT-X: A Self-Evolving Collaborative Multi-Agent System for Transparent and Trustworthy Dermatological Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SkinGPT-X，解决皮肤科诊断中多类别、罕见病及可解释性问题，通过多智能体系统实现透明可靠诊断。**

- **链接: [https://arxiv.org/pdf/2603.26122](https://arxiv.org/pdf/2603.26122)**

> **作者:** Zhangtianyi Chen; Yuhao Shen; Florensia Widjaja; Yan Xu; Liyuan Sun; Zijian Wang; Hongyi Chen; Wufei Dai; Juexiao Zhou
>
> **摘要:** While recent advancements in Large Language Models have significantly advanced dermatological diagnosis, monolithic LLMs frequently struggle with fine-grained, large-scale multi-class diagnostic tasks and rare skin disease diagnosis owing to training data sparsity, while also lacking the interpretability and traceability essential for clinical reasoning. Although multi-agent systems can offer more transparent and explainable diagnostics, existing frameworks are primarily concentrated on Visual Question Answering and conversational tasks, and their heavy reliance on static knowledge bases restricts adaptability in complex real-world clinical settings. Here, we present SkinGPT-X, a multimodal collaborative multi-agent system for dermatological diagnosis integrated with a self-evolving dermatological memory mechanism. By simulating the diagnostic workflow of dermatologists and enabling continuous memory evolution, SkinGPT-X delivers transparent and trustworthy diagnostics for the management of complex and rare dermatological cases. To validate the robustness of SkinGPT-X, we design a three-tier comparative experiment. First, we benchmark SkinGPT-X against four state-of-the-art LLMs across four public datasets, demonstrating its state-of-the-art performance with a +9.6% accuracy improvement on DDI31 and +13% weighted F1 gain on Dermnet over the state-of-the-art model. Second, we construct a large-scale multi-class dataset covering 498 distinct dermatological categories to evaluate its fine-grained classification capabilities. Finally, we curate the rare skin disease dataset, the first benchmark to address the scarcity of clinical rare skin diseases which contains 564 clinical samples with eight rare dermatological diseases. On this dataset, SkinGPT-X achieves a +9.8% accuracy improvement, a +7.1% weighted F1 improvement, a +10% Cohen's Kappa improvement.
>
---
#### [new 116] Reinforcing Structured Chain-of-Thought for Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，旨在解决多模态大语言模型在推理中出现的思维偏差和时间理解不足问题。提出SDRL框架，无需监督微调，通过结构化思考链提升性能。**

- **链接: [https://arxiv.org/pdf/2603.25942](https://arxiv.org/pdf/2603.25942)**

> **作者:** Peiyao Wang; Haotian Xu; Noranart Vesdapunt; Rui Hou; Jingyi Zhang; Haibin Ling; Oleksandr Obiednikov; Ning Zhou; Kah Kuen Fu
>
> **备注:** Accepted to CVPR 2026 (Main Conference)
>
> **摘要:** Multi-modal Large Language Models (MLLMs) show promise in video understanding. However, their reasoning often suffers from thinking drift and weak temporal comprehension, even when enhanced by Reinforcement Learning (RL) techniques like Group Relative Policy Optimization (GRPO). Moreover, existing RL methods usually depend on Supervised Fine-Tuning (SFT), which requires costly Chain-of-Thought (CoT) annotation and multi-stage training, and enforces fixed reasoning paths, limiting MLLMs' ability to generalize and potentially inducing bias. To overcome these limitations, we introduce Summary-Driven Reinforcement Learning (SDRL), a novel single-stage RL framework that obviates the need for SFT by utilizing a Structured CoT format: Summarize -> Think -> Answer. SDRL introduces two self-supervised mechanisms integrated into the GRPO objective: 1) Consistency of Vision Knowledge (CVK) enforces factual grounding by reducing KL divergence among generated summaries; and 2) Dynamic Variety of Reasoning (DVR) promotes exploration by dynamically modulating thinking diversity based on group accuracy. This novel integration effectively balances alignment and exploration, supervising both the final answer and the reasoning process. Our method achieves state-of-the-art performance on seven public VideoQA datasets.
>
---
#### [new 117] Shared Representation for 3D Pose Estimation, Action Classification, and Progress Prediction from Tactile Signals
- **分类: cs.CV**

- **简介: 该论文属于人体姿态估计、动作分类和进度预测任务，旨在解决触觉信号下多任务学习问题。提出SCOTTI模型，实现三任务联合优化，提升性能。**

- **链接: [https://arxiv.org/pdf/2603.25906](https://arxiv.org/pdf/2603.25906)**

> **作者:** Isaac Han; Seoyoung Lee; Sangyeon Park; Ecehan Akan; Yiyue Luo; Joseph DelPreto; Kyung-Joong Kim
>
> **摘要:** Estimating human pose, classifying actions, and predicting movement progress are essential for human-robot interaction. While vision-based methods suffer from occlusion and privacy concerns in realistic environments, tactile sensing avoids these issues. However, prior tactile-based approaches handle each task separately, leading to suboptimal performance. In this study, we propose a Shared COnvolutional Transformer for Tactile Inference (SCOTTI) that learns a shared representation to simultaneously address three separate prediction tasks: 3D human pose estimation, action class categorization, and action completion progress estimation. To the best of our knowledge, this is the first work to explore action progress prediction using foot tactile signals from custom wireless insole sensors. This unified approach leverages the mutual benefits of multi-task learning, enabling the model to achieve improved performance across all three tasks compared to learning them independently. Experimental results demonstrate that SCOTTI outperforms existing approaches across all three tasks. Additionally, we introduce a novel dataset collected from 15 participants performing various activities and exercises, with 7 hours of total duration, across eight different activities.
>
---
#### [new 118] When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多主体个性化生成任务，旨在解决模型在复杂场景中身份混淆的问题。通过构建基准测试和提出新评估指标，揭示现有模型在多主体交互中的性能下降。**

- **链接: [https://arxiv.org/pdf/2603.26078](https://arxiv.org/pdf/2603.26078)**

> **作者:** Zhihan Chen; Yuhuan Zhao; Yijie Zhu; Xinyu Yao
>
> **备注:** 10 pages, 7 figures, accepted by CVPR 2026 Workshop P13N
>
> **摘要:** Subject-driven text-to-image diffusion models have achieved remarkable success in preserving single identities, yet their ability to compose multiple interacting subjects remains largely unexplored and highly challenging. Existing evaluation protocols typically rely on global CLIP metrics, which are insensitive to local identity collapse and fail to capture the severity of multi-subject entanglement. In this paper, we identify a pervasive "Illusion of Scalability" in current models: while they excel at synthesizing 2-4 subjects in simple layouts, they suffer from catastrophic identity collapse when scaled to 6-10 subjects or tasked with complex physical interactions. To systematically expose this failure mode, we construct a rigorous stress-test benchmark comprising 75 prompts distributed across varying subject counts and interaction difficulties (Neutral, Occlusion, Interaction). Furthermore, we demonstrate that standard CLIP-based metrics are fundamentally flawed for this task, as they often assign high scores to semantically correct but identity-collapsed images (e.g., generating generic clones). To address this, we introduce the Subject Collapse Rate (SCR), a novel evaluation metric grounded in DINOv2's structural priors, which strictly penalizes local attention leakage and homogenization. Our extensive evaluation of state-of-the-art models (MOSAIC, XVerse, PSR) reveals a precipitous drop in identity fidelity as scene complexity grows, with SCR approaching 100% at 10 subjects. We trace this collapse to the semantic shortcuts inherent in global attention routing, underscoring the urgent need for explicit physical disentanglement in future generative architectures.
>
---
#### [new 119] GeoReFormer: Geometry-Aware Refinement for Lane Segment Detection and Topology Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦于自动驾驶中的车道线检测与拓扑推理任务，解决传统方法未能有效编码几何与关系结构的问题。提出GeoReFormer模型，通过几何感知的查询初始化和拓扑传播提升检测精度与一致性。**

- **链接: [https://arxiv.org/pdf/2603.26018](https://arxiv.org/pdf/2603.26018)**

> **作者:** Danny Abraham; Nikhil Kamalkumar Advani; Arun Das; Nikil Dutt
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Accurate 3D lane segment detection and topology reasoning are critical for structured online map construction in autonomous driving. Recent transformer-based approaches formulate this task as query-based set prediction, yet largely inherit decoder designs originally developed for compact object detection. However, lane segments are continuous polylines embedded in directed graphs, and generic query initialization and unconstrained refinement do not explicitly encode this geometric and relational structure. We propose GeoReFormer (Geometry-aware Refinement Transformer), a unified query-based architecture that embeds geometry- and topology-aware inductive biases directly within the transformer decoder. GeoReFormer introduces data-driven geometric priors for structured query initialization, bounded coordinate-space refinement for stable polyline deformation, and per-query gated topology propagation to selectively integrate relational context. On the OpenLane-V2 benchmark, GeoReFormer achieves state-of-the-art performance with 34.5% mAP while improving topology consistency over strong transformer baselines, demonstrating the utility of explicit geometric and relational structure encoding.
>
---
#### [new 120] Finding Distributed Object-Centric Properties in Self-Supervised Transformers
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文属于视觉模型研究任务，旨在解决自监督ViT中对象定位不准确的问题。通过分析注意力机制，提出Object-DINO方法提取分布式对象中心信息，提升无监督目标发现和视觉定位效果。**

- **链接: [https://arxiv.org/pdf/2603.26127](https://arxiv.org/pdf/2603.26127)**

> **作者:** Samyak Rawlekar; Amitabh Swain; Yujun Cai; Yiwei Wang; Ming-Hsuan Yang; Narendra Ahuja
>
> **备注:** Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Self-supervised Vision Transformers (ViTs) like DINO show an emergent ability to discover objects, typically observed in [CLS] token attention maps of the final layer. However, these maps often contain spurious activations resulting in poor localization of objects. This is because the [CLS] token, trained on an image-level objective, summarizes the entire image instead of focusing on objects. This aggregation dilutes the object-centric information existing in the local, patch-level interactions. We analyze this by computing inter-patch similarity using patch-level attention components (query, key, and value) across all layers. We find that: (1) Object-centric properties are encoded in the similarity maps derived from all three components ($q, k, v$), unlike prior work that uses only key features or the [CLS] token. (2) This object-centric information is distributed across the network, not just confined to the final layer. Based on these insights, we introduce Object-DINO, a training-free method that extracts this distributed object-centric information. Object-DINO clusters attention heads across all layers based on the similarities of their patches and automatically identifies the object-centric cluster corresponding to all objects. We demonstrate Object-DINO's effectiveness on two applications: enhancing unsupervised object discovery (+3.6 to +12.4 CorLoc gains) and mitigating object hallucination in Multimodal Large Language Models by providing visual grounding. Our results demonstrate that using this distributed object-centric information improves downstream tasks without additional training.
>
---
#### [new 121] Beyond Language: Grounding Referring Expressions with Hand Pointing in Egocentric Vision
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，旨在解决语言模糊和忽略非语言指代线索的问题。通过构建多模态数据集并提出SV-CoT框架，提升基于手势和语言的指代理解能力。**

- **链接: [https://arxiv.org/pdf/2603.26646](https://arxiv.org/pdf/2603.26646)**

> **作者:** Ling Li; Bowen Liu; Zinuo Zhan; Peng Jie; Jianhui Zhong; Kenglun Chang; Zhidong Deng
>
> **摘要:** Traditional Visual Grounding (VG) predominantly relies on textual descriptions to localize objects, a paradigm that inherently struggles with linguistic ambiguity and often ignores non-verbal deictic cues prevalent in real-world interactions. In natural egocentric engagements, hand-pointing combined with speech forms the most intuitive referring mechanism. To bridge this gap, we introduce EgoPoint-Ground, the first large-scale multimodal dataset dedicated to egocentric deictic visual grounding. Comprising over \textbf{15k} interactive samples in complex scenes, the dataset provides rich, multi-grained annotations including hand-target bounding box pairs and dense semantic captions. We establish a comprehensive benchmark for hand-pointing referring expression resolution, evaluating a wide spectrum of mainstream Multimodal Large Language Models (MLLMs) and state-of-the-art VG architectures. Furthermore, we propose SV-CoT, a novel baseline framework that reformulates grounding as a structured inference process, synergizing gestural and linguistic cues through a Visual Chain-of-Thought paradigm. Extensive experiments demonstrate that SV-CoT achieves an $\textbf{11.7\%}$ absolute improvement over existing methods, effectively mitigating semantic ambiguity and advancing the capability of agents to comprehend multimodal physical intents. The dataset and code will be made publicly available.
>
---
#### [new 122] ViGoR-Bench: How Far Are Visual Generative Models From Zero-Shot Visual Reasoners?
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ViGoR-Bench，用于评估视觉生成模型的零样本推理能力。针对现有评估方法不足，设计统一基准，解决生成模型逻辑推理缺陷问题。**

- **链接: [https://arxiv.org/pdf/2603.25823](https://arxiv.org/pdf/2603.25823)**

> **作者:** Haonan Han; Jiancheng Huang; Xiaopeng Sun; Junyan He; Rui Yang; Jie Hu; Xiaojiang Peng; Lin Ma; Xiaoming Wei; Xiu Li
>
> **摘要:** Beneath the stunning visual fidelity of modern AIGC models lies a "logical desert", where systems fail tasks that require physical, causal, or complex spatial reasoning. Current evaluations largely rely on superficial metrics or fragmented benchmarks, creating a ``performance mirage'' that overlooks the generative process. To address this, we introduce ViGoR Vision-G}nerative Reasoning-centric Benchmark), a unified framework designed to dismantle this mirage. ViGoR distinguishes itself through four key innovations: 1) holistic cross-modal coverage bridging Image-to-Image and Video tasks; 2) a dual-track mechanism evaluating both intermediate processes and final results; 3) an evidence-grounded automated judge ensuring high human alignment; and 4) granular diagnostic analysis that decomposes performance into fine-grained cognitive dimensions. Experiments on over 20 leading models reveal that even state-of-the-art systems harbor significant reasoning deficits, establishing ViGoR as a critical ``stress test'' for the next generation of intelligent vision models. The demo have been available at this https URL
>
---
#### [new 123] Dual-Stage Invariant Continual Learning under Extreme Visual Sparsity
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，解决极端视觉稀疏条件下的持续学习问题。针对背景干扰导致的特征漂移，提出双阶段不变持续学习框架，提升模型稳定性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.26190](https://arxiv.org/pdf/2603.26190)**

> **作者:** Rangya Zhang; Jiaping Xiao; Lu Bai; Yuhang Zhang; Mir Feroskhan
>
> **摘要:** Continual learning seeks to maintain stable adaptation under non-stationary environments, yet this problem becomes particularly challenging in object detection, where most existing methods implicitly assume relatively balanced visual conditions. In extreme-sparsity regimes, such as those observed in space-based resident space object (RSO) detection scenarios, foreground signals are overwhelmingly dominated by background observations. Under such conditions, we analytically demonstrate that background-driven gradients destabilize the feature backbone during sequential domain shifts, causing progressive representation drift. This exposes a structural limitation of continual learning approaches relying solely on output-level distillation, as they fail to preserve intermediate representation stability. To address this, we propose a dual-stage invariant continual learning framework via joint distillation, enforcing structural and semantic consistency on both backbone representations and detection predictions, respectively, thereby suppressing error propagation at its source while maintaining adaptability. Furthermore, to regulate gradient statistics under severe imbalance, we introduce a sparsity-aware data conditioning strategy combining patch-based sampling and distribution-aware augmentation. Experiments on a high-resolution space-based RSO detection dataset show consistent improvement over established continual object detection methods, achieving an absolute gain of +4.0 mAP under sequential domain shifts.
>
---
#### [new 124] Experimental study on surveillance video-based indoor occupancy measurement with occupant-centric control
- **分类: eess.SY; cs.CV**

- **简介: 该论文属于智能建筑中的 occupancy measurement 任务，旨在提升基于视频的人员检测准确性，以优化 HVAC 控制。通过对比不同方法，提出 LLM 增强的解决方案，实现17.94%的节能效果。**

- **链接: [https://arxiv.org/pdf/2603.26081](https://arxiv.org/pdf/2603.26081)**

> **作者:** Irfan Qaisar; Kailai Sun; Qingshan Jia; Qianchuan Zhao
>
> **摘要:** Accurate occupancy information is essential for closed-loop occupant-centric control (OCC) in smart buildings. However, existing vision-based occupancy measurement methods often struggle to provide stable and accurate measurements in real indoor environments, and their implications for downstream HVAC control remain insufficiently studied. To achieve Net Zero emissions by 2050, this paper presents an experimental study of large language models (LLMs)-enhanced vision-based indoor occupancy measurement and its impact on OCC-enabled HVAC operation. Detection-only, tracking-based, and LLM-based refinement pipelines are compared under identical conditions using real surveillance data collected from a research laboratory in China, with frame-level manual ground-truth annotations. Results show that tracking-based methods improve temporal stability over detection-only measurement, while LLM-based refinement further improves occupancy measurement performance and reduces false unoccupied prediction. The best-performing pipeline, YOLOv8+DeepSeek, achieves an accuracy of 0.8824 and an F1-score of 0.9320. This pipeline is then integrated into an HVAC supervisory model predictive control framework in OpenStudio-EnergyPlus. Experimental results demonstrate that the proposed framework can support more efficient OCC operation, achieving a substantial HVAC energy-saving potential of 17.94%. These findings provide an effective methodology and practical foundation for future research in AI-enhanced smart building operations.
>
---
#### [new 125] Cone-Beam CT Image Quality Enhancement Using A Latent Diffusion Model Trained with Simulated CBCT Artifacts
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决CBCT图像质量差的问题。通过构建无过矫正的扩散模型，提升图像质量并保持解剖结构。**

- **链接: [https://arxiv.org/pdf/2603.26014](https://arxiv.org/pdf/2603.26014)**

> **作者:** Naruki Murahashi; Mitsuhiro Nakamura; Megumi Nakao
>
> **摘要:** Cone-beam computed tomography (CBCT) images are problematic in clinical medicine because of their low contrast and high artifact content compared with conventional CT images. Although there are some studies to improve image quality, in regions subject to organ deformation, the anatomical structure may change after such image quality improvement. In this study, we propose an overcorrection-free CBCT image quality enhancement method based on a conditional latent diffusion model using pseudo-CBCT images. Pseudo-CBCT images are created from CT images using a simple method that simulates CBCT artifacts and are spatially consistent with the CT images. By performing self-supervised learning with these spatially consistent paired images, we can improve image quality while maintaining anatomical structures. Furthermore, extending the framework of the conditional diffusion model to latent space improves the efficiency of image processing. Our model was trained on pelvic CT-pseudo-CBCT paired data and was applied to both pseudo-CBCT and real CBCT data. The experimental results using data of 75 cases show that with our proposed method, the structural changes were less than 1/1000th (in terms of the number of pixels) of those of a conventional method involving learning with real images, and the correlation coefficient between the CT value distributions of the generated and reference images was 0.916, approaching the same level as conventional methods. We also confirmed that the proposed framework achieves faster processing and superior improvement performance compared with the framework of a conditional diffusion model, even under constrained training settings.
>
---
#### [new 126] Learning to Recorrupt: Noise Distribution Agnostic Self-Supervised Image Denoising
- **分类: eess.IV; cs.CV; stat.ML**

- **简介: 该论文属于图像去噪任务，解决传统方法依赖噪声分布的问题。提出L2R方法，无需先验噪声知识，通过可学习网络实现有效去噪。**

- **链接: [https://arxiv.org/pdf/2603.25869](https://arxiv.org/pdf/2603.25869)**

> **作者:** Brayan Monroy; Jorge Bacca; Julián Tachella
>
> **摘要:** Self-supervised image denoising methods have traditionally relied on either architectural constraints or specialized loss functions that require prior knowledge of the noise distribution to avoid the trivial identity mapping. Among these, approaches such as Noisier2Noise or Recorrupted2Recorrupted, create training pairs by adding synthetic noise to the noisy images. While effective, these recorruption-based approaches require precise knowledge of the noise distribution, which is often not available. We present Learning to Recorrupt (L2R), a noise distribution-agnostic denoising technique that eliminates the need for knowledge of the noise distribution. Our method introduces a learnable monotonic neural network that learns the recorruption process through a min-max saddle-point objective. The proposed method achieves state-of-the-art performance across unconventional and heavy-tailed noise distributions, such as log-gamma, Laplace, and spatially correlated noise, as well as signal-dependent noise models such as Poisson-Gaussian noise.
>
---
#### [new 127] Longitudinal Boundary Sharpness Coefficient Slopes Predict Time to Alzheimer's Disease Conversion in Mild Cognitive Impairment: A Survival Analysis Using the ADNI Cohort
- **分类: q-bio.NC; cs.AI; cs.CV**

- **简介: 该论文属于预测任务，旨在解决MCI向AD转化时间的预测问题。通过分析BSC变化率，使用随机生存森林模型提高预测准确性。**

- **链接: [https://arxiv.org/pdf/2603.26007](https://arxiv.org/pdf/2603.26007)**

> **作者:** Ishaan Cherukuri
>
> **摘要:** Predicting whether someone with mild cognitive impairment (MCI) will progress to Alzheimer's disease (AD) is crucial in the early stages of neurodegeneration. This uncertainty limits enrollment in clinical trials and delays urgent treatment. The Boundary Sharpness Coefficient (BSC) measures how well-defined the gray-white matter boundary looks on structural MRI. This study measures how BSC changes over time, namely, how fast the boundary degrades each year works much better than looking at a single baseline scan for predicting MCI-to-AD conversion. This study analyzed 1,824 T1-weighted MRI scans from 450 ADNI subjects (95 converters, 355 stable; mean follow-up: 4.84 years). BSC voxel-wise maps were computed using tissue segmentation at the gray-white matter cortical ribbon. Previous studies have used CNN and RNN models that reached 96.0% accuracy for AD classification and 84.2% for MCI conversion, but those approaches disregard specific regions within the brain. This study focused specifically on the gray-white matter interface. The approach uses temporal slope features capturing boundary degradation rates, feeding them into Random Survival Forest, a non-parametric ensemble method for right-censored survival data. The Random Survival Forest trained on BSC slopes achieved a test C-index of 0.63, a 163% improvement over baseline parametric models (test C-index: 0.24). Structural MRI costs a fraction of PET imaging ($800--$1,500 vs. $5,000--$7,000) and does not require CSF collection. These temporal biomarkers could help with patient-centered safety screening as well as risk assessment.
>
---
#### [new 128] Accurate Precipitation Forecast by Efficiently Learning from Massive Atmospheric Variables and Unbalanced Distribution
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于短时降水预测任务，旨在解决降水样本不平衡和模型效率低的问题。提出新模型与WMCE损失函数，有效利用大气数据，提升预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.26108](https://arxiv.org/pdf/2603.26108)**

> **作者:** Shuangliang Li; Siwei Li; Li Li; Weijie Zou; Jie Yang; Maolin Zhang
>
> **摘要:** Short-term (0-24 hours) precipitation forecasting is highly valuable to socioeconomic activities and public safety. However, the highly complex evolution patterns of precipitation events, the extreme imbalance between precipitation and non-precipitation samples, and the inability of existing models to efficiently and effectively utilize large volumes of multi-source atmospheric observation data hinder improvements in precipitation forecasting accuracy and computational efficiency. To address the above challenges, this study developed a novel forecasting model capable of effectively and efficiently utilizing massive atmospheric observations by automatically extracting and iteratively predicting the latent features strongly associated with precipitation evolution. Furthermore, this study introduces a 'WMCE' loss function, designed to accurately discriminate extremely scarce precipitation events while precisely predicting their intensity values. Extensive experiments on two datasets demonstrate that our proposed model substantially and consistently outperforms all prevalent baselines in both accuracy and efficiency. Moreover, the proposed forecasting model substantially lowers the computational cost required to obtain valuable predictions compared to existing approaches, thereby positioning it as a milestone for efficient and practical precipitation forecasting.
>
---
#### [new 129] Decoding Defensive Coverage Responsibilities in American Football Using Factorized Attention Based Transformer Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于体育分析任务，旨在预测美式足球防守覆盖责任。通过因子化注意力Transformer模型，解决个体防守者分配与匹配问题，提升战术理解与策略制定。**

- **链接: [https://arxiv.org/pdf/2603.25901](https://arxiv.org/pdf/2603.25901)**

> **作者:** Kevin Song; Evan Diewald; Ornob Siddiquee; Chris Boomhower; Keegan Abdoo; Mike Band; Amy Lee
>
> **备注:** 19 pages, 8 figures, ISACE 2026
>
> **摘要:** Defensive coverage schemes in the National Football League (NFL) represent complex tactical patterns requiring coordinated assignments among defenders who must react dynamically to the offense's passing concept. This paper presents a factorized attention-based transformer model applied to NFL multi-agent play tracking data to predict individual coverage assignments, receiver-defender matchups, and the targeted defender on every pass play. Unlike previous approaches that focus on post-hoc coverage classification at the team level, our model enables predictive modeling of individual player assignments and matchup dynamics throughout the play. The factorized attention mechanism separates temporal and agent dimensions, allowing independent modeling of player movement patterns and inter-player relationships. Trained on randomly truncated trajectories, the model generates frame-by-frame predictions that capture how defensive responsibilities evolve from pre-snap through pass arrival. Our models achieve approximately 89\%+ accuracy for all tasks, with true accuracy potentially higher given annotation ambiguity in the ground truth labels. These outputs also enable novel derivative metrics, including disguise rate and double coverage rate, which enable enhanced storytelling in TV broadcasts as well as provide actionable insights for team strategy development and player evaluation.
>
---
#### [new 130] GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于GUI代理领域，解决其因训练数据不足导致的领域偏差问题。通过视频检索和自动标注，提升代理对特定应用的理解与操作能力。**

- **链接: [https://arxiv.org/pdf/2603.26266](https://arxiv.org/pdf/2603.26266)**

> **作者:** Rui Xie; Zhi Gao; Chenrui Shi; Zirui Shang; Lu Chen; Qing Li
>
> **备注:** 28 pages, 8 figures, 7 tables
>
> **摘要:** Large vision-language models have endowed GUI agents with strong general capabilities for interface understanding and interaction. However, due to insufficient exposure to domain-specific software operation data during training, these agents exhibit significant domain bias - they lack familiarity with the specific operation workflows (planning) and UI element layouts (grounding) of particular applications, limiting their real-world task performance. In this paper, we present GUIDE (GUI Unbiasing via Instructional-Video Driven Expertise), a training-free, plug-and-play framework that resolves GUI agent domain bias by autonomously acquiring domain-specific expertise from web tutorial videos through a retrieval-augmented automated annotation pipeline. GUIDE introduces two key innovations. First, a subtitle-driven Video-RAG pipeline unlocks video semantics through subtitle analysis, performing progressive three-stage retrieval - domain classification, topic extraction, and relevance matching - to identify task-relevant tutorial videos. Second, a fully automated annotation pipeline built on an inverse dynamics paradigm feeds consecutive keyframes enhanced with UI element detection into VLMs, inferring the required planning and grounding knowledge that are injected into the agent's corresponding modules to address both manifestations of domain bias. Extensive experiments on OSWorld demonstrate GUIDE's generality as a plug-and-play component for both multi-agent systems and single-model agents. It consistently yields over 5% improvements and reduces execution steps - without modifying any model parameters or architecture - validating GUIDE as an architecture-agnostic enhancement to bridge GUI agent domain bias.
>
---
#### [new 131] FINDER: Zero-Shot Field-Integrated Network for Distortion-free EPI Reconstruction in Diffusion MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出FINDER框架，解决扩散MRI中EPI序列的几何失真问题。通过联合优化图像和B0场图，实现无失真重建。**

- **链接: [https://arxiv.org/pdf/2603.26117](https://arxiv.org/pdf/2603.26117)**

> **作者:** Namgyu Han; Seong Dae Yun; Chaeeun Lim; Sunghyun Seok; Sunju Kim; Yoonhwan Kim; Yohan Jun; Tae Hyung Kim; Berkin Bilgic; Jaejin Cho
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Echo-planar imaging (EPI) remains the cornerstone of diffusion MRI, but it is prone to severe geometric distortions due to its rapid sampling scheme that renders the sequence highly sensitive to $B_{0}$ field inhomogeneities. While deep learning has helped improve MRI reconstruction, integrating robust geometric distortion correction into a self-supervised framework remains an unmet need. To address this, we present FINDER (Field-Integrated Network for Distortion-free EPI Reconstruction), a novel zero-shot, scan-specific framework that reformulates reconstruction as a joint optimization of the underlying image and the $B_{0}$ field map. Specifically, we employ a physics-guided unrolled network that integrates dual-domain denoisers and virtual coil extensions to enforce robust data consistency. This is coupled with an Implicit Neural Representation (INR) conditioned on spatial coordinates and latent image features to model the off-resonance field as a continuous, differentiable function. Employing an alternating minimization strategy, FINDER synergistically updates the reconstruction network and the field map, effectively disentangling susceptibility-induced geometric distortions from anatomical structures. Experimental results demonstrate that FINDER achieves superior geometric fidelity and image quality compared to state-of-the-art baselines, offering a robust solution for high-quality diffusion imaging.
>
---
#### [new 132] SAFT: Sensitivity-Aware Filtering and Transmission for Adaptive 3D Point Cloud Communication over Wireless Channels
- **分类: cs.IT; cs.CV**

- **简介: 该论文提出SAFT框架，解决无线信道下3D点云传输问题，通过敏感性感知过滤和自适应重建提升传输可靠性与几何保真度。**

- **链接: [https://arxiv.org/pdf/2603.26197](https://arxiv.org/pdf/2603.26197)**

> **作者:** Huda Adam Sirag Mekki; Hui Yuan; Mohanad M. G. Hassan; Zejia Chen; Guanghui Zhang
>
> **摘要:** Reliable transmission of 3D point clouds over wireless channels is challenging due to time-varying signal-to-noise ratio (SNR) and limited bandwidth. This paper introduces sensitivity-aware filtering and transmission (SAFT), a learned transmission framework that integrates a Point-BERT-inspired encoder, a sensitivity-guided token filtering (STF) unit, a quantization block, and an SNR-aware decoder for adaptive reconstruction. Specifically, the STF module assigns token-wise importance scores based on the reconstruction sensitivity of each token under channel perturbation. We further employ a training-only symbol-usage penalty to stabilize the discrete representation, without affecting the transmitted payload. Experiments on ShapeNet, ModelNet40, and 8iVFB show that SAFT improves geometric fidelity (D1/D2 PSNR) compared with a separate source--channel coding pipeline (G-PCC combined with LDPC and QAM) and existing learned baselines, with the largest gains observed in low-SNR regimes, highlighting improved robustness under limited bandwidth.
>
---
#### [new 133] DFM-VLA: Iterative Action Refinement for Robot Manipulation via Discrete Flow Matching
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DFM-VLA，解决机器人操作中动作序列生成的误差修正问题，通过离散流匹配实现迭代优化，提升操作性能与成功率。**

- **链接: [https://arxiv.org/pdf/2603.26320](https://arxiv.org/pdf/2603.26320)**

> **作者:** Jiayi Chen; Wenxuan Song; Shuai Chen; Jingbo Wang; Zhijun Li; Haoang Li
>
> **摘要:** Vision--Language--Action (VLA) models that encode actions using a discrete tokenization scheme are increasingly adopted for robotic manipulation, but existing decoding paradigms remain fundamentally limited. Whether actions are decoded sequentially by autoregressive VLAs or in parallel by discrete diffusion VLAs, once a token is generated, it is typically fixed and cannot be revised in subsequent iterations, so early token errors cannot be effectively corrected later. We propose DFM-VLA, a discrete flow matching VLA for iterative refinement of action tokens. DFM-VLA~models a token-level probability velocity field that dynamically updates the full action sequence across refinement iterations. We investigate two ways to construct the velocity field: an auxiliary velocity-head formulation and an action-embedding-guided formulation. Our framework further adopts a two-stage decoding strategy with an iterative refinement stage followed by deterministic validation for stable convergence. Extensive experiments on CALVIN, LIBERO, and real-world manipulation tasks show that DFM-VLA consistently outperforms strong autoregressive, discrete diffusion, and continuous diffusion baselines in manipulation performance while retaining high inference efficiency. In particular, DFM-VLA achieves an average success length of 4.44 on CALVIN and an average success rate of 95.7\% on LIBERO, highlighting the value of action refinement via discrete flow matching for robotic manipulation. Our project is available \url{this https URL}
>
---
#### [new 134] PruneFuse: Efficient Data Selection via Weight Pruning and Network Fusion
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出PruneFuse，用于高效数据选择任务，解决传统方法计算成本高的问题。通过结构化剪枝和网络融合，提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.26138](https://arxiv.org/pdf/2603.26138)**

> **作者:** Humaira Kousar; Hasnain Irshad Bhatti; Jaekyun Moon
>
> **备注:** Published in TMLR (Featured Certification). arXiv admin note: substantial text overlap with arXiv:2501.01118
>
> **摘要:** Efficient data selection is crucial for enhancing the training efficiency of deep neural networks and minimizing annotation requirements. Traditional methods often face high computational costs, limiting their scalability and practical use. We introduce PruneFuse, a novel strategy that leverages pruned networks for data selection and later fuses them with the original network to optimize training. PruneFuse operates in two stages: First, it applies structured pruning to create a smaller pruned network that, due to its structural coherence with the original network, is well-suited for the data selection task. This small network is then trained and selects the most informative samples from the dataset. Second, the trained pruned network is seamlessly fused with the original network. This integration leverages the insights gained during the training of the pruned network to facilitate the learning process of the fused network while leaving room for the network to discover more robust solutions. Extensive experimentation on various datasets demonstrates that PruneFuse significantly reduces computational costs for data selection, achieves better performance than baselines, and accelerates the overall training process.
>
---
#### [new 135] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于测试时适应任务，解决分布偏移下的性能下降问题。提出AcTTA框架，通过动态调整激活函数实现模型自适应，无需修改权重或源数据。**

- **链接: [https://arxiv.org/pdf/2603.26096](https://arxiv.org/pdf/2603.26096)**

> **作者:** Hyeongyu Kim; Geonhui Han; Dosik Hwang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Test-time adaptation (TTA) aims to mitigate performance degradation under distribution shifts by updating model parameters during inference. Existing approaches have primarily framed adaptation around affine modulation, focusing on recalibrating normalization layers. This perspective, while effective, overlooks another influential component in representation dynamics: the activation function. We revisit this overlooked space and propose AcTTA, an activation-aware framework that reinterprets conventional activation functions from a learnable perspective and updates them adaptively at test time. AcTTA reformulates conventional activation functions (e.g., ReLU, GELU) into parameterized forms that shift their response threshold and modulate gradient sensitivity, enabling the network to adjust activation behavior under domain shifts. This functional reparameterization enables continuous adjustment of activation behavior without modifying network weights or requiring source data. Despite its simplicity, AcTTA achieves robust and stable adaptation across diverse corruptions. Across CIFAR10-C, CIFAR100-C, and ImageNet-C, AcTTA consistently surpasses normalization-based TTA methods. Our findings highlight activation adaptation as a compact and effective route toward domain-shift-robust test-time learning, broadening the prevailing affine-centric view of adaptation.
>
---
#### [new 136] ComVi: Context-Aware Optimized Comment Display in Video Playback
- **分类: cs.MM; cs.CV; cs.GR; cs.HC**

- **简介: 该论文提出ComVi系统，解决视频播放时评论与内容不同步的问题。通过音频视频相关性分析，将评论同步到合适时间点，提升观看体验。属于视频评论优化任务。**

- **链接: [https://arxiv.org/pdf/2603.26173](https://arxiv.org/pdf/2603.26173)**

> **作者:** Minsun Kim; Dawon Lee; Junyong Noh
>
> **备注:** To appear in Proceedings of the ACM CHI Conference on Human Factors in Computing Systems (CHI 2026)
>
> **摘要:** On general video-sharing platforms like YouTube, comments are displayed independently of video playback. As viewers often read comments while watching a video, they may encounter ones referring to moments unrelated to the current scene, which can reveal spoilers and disrupt immersion. To address this problem, we present ComVi, a novel system that displays comments at contextually relevant moments, enabling viewers to see time-synchronized comments and video content together. We first map all comments to relevant video timestamps by computing audio-visual correlation, then construct the comment sequence through an optimization that considers temporal relevance, popularity (number of likes), and display duration for comfortable reading. In a user study, ComVi provided a significantly more engaging experience than conventional video interfaces (i.e., YouTube and Danmaku), with 71.9% of participants selecting ComVi as their most preferred interface.
>
---
#### [new 137] Adapting Segment Anything Model 3 for Concept-Driven Lesion Segmentation in Medical Images: An Experimental Study
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决通用病变分割难题。通过评估SAM3模型，结合概念提示和多模态数据，提升分割的泛化能力和准确性。**

- **链接: [https://arxiv.org/pdf/2603.25945](https://arxiv.org/pdf/2603.25945)**

> **作者:** Guoping Xu; Jayaram K. Udupa; Yubing Tong; Xin Long; Ying Zhang; Jie Deng; Weiguo Lu; You Zhang
>
> **备注:** 31 pages, 8 figures
>
> **摘要:** Accurate lesion segmentation is essential in medical image analysis, yet most existing methods are designed for specific anatomical sites or imaging modalities, limiting their generalizability. Recent vision-language foundation models enable concept-driven segmentation in natural images, offering a promising direction for more flexible medical image analysis. However, concept-prompt-based lesion segmentation, particularly with the latest Segment Anything Model 3 (SAM3), remains underexplored. In this work, we present a systematic evaluation of SAM3 for lesion segmentation. We assess its performance using geometric bounding boxes and concept-based text and image prompts across multiple modalities, including multiparametric MRI, CT, ultrasound, dermoscopy, and endoscopy. To improve robustness, we incorporate additional prior knowledge, such as adjacent-slice predictions, multiparametric information, and prior annotations. We further compare different fine-tuning strategies, including partial module tuning, adapter-based methods, and full-model optimization. Experiments on 13 datasets covering 11 lesion types demonstrate that SAM3 achieves strong cross-modality generalization, reliable concept-driven segmentation, and accurate lesion delineation. These results highlight the potential of concept-based foundation models for scalable and practical medical image segmentation. Code and trained models will be released at: this https URL
>
---
#### [new 138] Adapting Frozen Mono-modal Backbones for Multi-modal Registration via Contrast-Agnostic Instance Optimization
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像配准任务，解决多模态下模型泛化能力不足的问题。通过冻结单模态模型并引入轻量适配模块，提升多模态和域外场景的配准性能。**

- **链接: [https://arxiv.org/pdf/2603.26393](https://arxiv.org/pdf/2603.26393)**

> **作者:** Yi Zhang; Yidong Zhao; Qian Tao
>
> **备注:** MICCAI Learn2Reg Challenge
>
> **摘要:** Deformable image registration remains a central challenge in medical image analysis, particularly under multi-modal scenarios where intensity distributions vary significantly across scans. While deep learning methods provide efficient feed-forward predictions, they often fail to generalize robustly under distribution shifts at test time. A straightforward remedy is full network fine-tuning, yet for modern architectures such as Transformers or deep U-Nets, this adaptation is prohibitively expensive in both memory and runtime when operating in 3D. Meanwhile, the naive fine-tuning struggles more with potential degradation in performance in the existence of drastic domain shifts. In this work, we propose a registration framework that integrates a frozen pretrained \textbf{mono-modal} registration model with a lightweight adaptation pipeline for \textbf{multi-modal} image registration. Specifically, we employ style transfer based on contrast-agnostic representation generation and refinement modules to bridge modality and domain gaps with instance optimization at test time. This design is orthogonal to the choice of backbone mono-modal model, thus avoids the computational burden of full fine-tuning while retaining the flexibility to adapt to unseen domains. We evaluate our approach on the Learn2Reg 2025 LUMIR validation set and observe consistent improvements over the pretrained state-of-the-art mono-modal backbone. In particular, the method ranks second on the multi-modal subset, third on the out-of-domain subset, and achieves fourth place overall in Dice score. These results demonstrate that combining frozen mono-modal models with modality adaptation and lightweight instance optimization offers an effective and practical pathway toward robust multi-modal registration.
>
---
## 更新

#### [replaced 001] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.22687](https://arxiv.org/pdf/2603.22687)**

> **作者:** Jiayin Sun; Caixia Sun; Boyu Yang; Hailin Li; Xiao Chen; Yi Zhang; Errui Ding; Liang Li; Chao Deng; Junlan Feng
>
> **备注:** accepted by CVPR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently demonstrated remarkable perceptual and reasoning abilities. However, they struggle to perceive fine-grained geometric structures, constraining their ability of geometric understanding and visual reasoning. To address this, we propose GeoTikzBridge, a framework that enhances local geometric perception and visual reasoning through tikz-based code generation. Within this framework, we build two models supported by two complementary datasets. The GeoTikzBridge-Base model is trained on GeoTikz-Base dataset, the largest image-to-tikz dataset to date with 2.5M pairs (16 $\times$ larger than existing open-sourced datasets). This process is achieved via iterative data expansion and a localized geometric transformation strategy. Subsequently, GeoTikzBridge-Instruct is fine-tuned on GeoTikz-Instruct dataset which is the first instruction-augmented tikz dataset supporting visual reasoning. Extensive experimental results demonstrate that our models achieve state-of-the-art performance among open-sourced MLLMs. Furthermore, GeoTikzBridge models can serve as plug-and-play reasoning modules for any MLLM(LLM), enhancing reasoning performance in geometric problem-solving. Datasets and codes are publicly available at: this https URL.
>
---
#### [replaced 002] PedaCo-Gen: Scaffolding Pedagogical Agency in Human-AI Collaborative Video Authoring
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [https://arxiv.org/pdf/2602.19623](https://arxiv.org/pdf/2602.19623)**

> **作者:** Injun Baek; Yearim Kim; Nojun Kwak
>
> **摘要:** While advancements in Text-to-Video (T2V) generative AI offer a promising path toward democratizing content creation, current models are often optimized for visual fidelity rather than instructional efficacy. This study introduces PedaCo-Gen, a pedagogically-informed human-AI collaborative video generating system for authoring instructional videos based on Mayer's Cognitive Theory of Multimedia Learning (CTML). Moving away from traditional "one-shot" generation, PedaCo-Gen introduces an Intermediate Representation (IR) phase, enabling educators to interactively review and refine video blueprints-comprising scripts and visual descriptions-with an AI reviewer. Our study with 23 education experts demonstrates that PedaCo-Gen significantly enhances video quality across various topics and CTML principles compared to baselines. Participants perceived the AI-driven guidance not merely as a set of instructions but as a metacognitive scaffold that augmented their instructional design expertise, reporting high production efficiency (M=4.26) and guide validity (M=4.04). These findings highlight the importance of reclaiming pedagogical agency through principled co-creation, providing a foundation for future AI authoring tools that harmonize generative power with human professional expertise.
>
---
#### [replaced 003] EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.18739](https://arxiv.org/pdf/2603.18739)**

> **作者:** Longfei Liu; Yongjie Hou; Yang Li; Qirui Wang; Youyang Sha; Yongjun Yu; Yinzhi Wang; Peizhe Ru; Xuanlong Yu; Xi Shen
>
> **备注:** Code is available at: this https URL
>
> **摘要:** Deploying high-performance dense prediction models on resource-constrained edge devices remains challenging due to strict limits on computation and memory. In practice, lightweight systems for object detection, instance segmentation, and pose estimation are still dominated by CNN-based architectures such as YOLO, while compact Vision Transformers (ViTs) often struggle to achieve similarly strong accuracy efficiency tradeoff, even with large scale pretraining. We argue that this gap is largely due to insufficient task specific representation learning in small scale ViTs, rather than an inherent mismatch between ViTs and edge dense prediction. To address this issue, we introduce EdgeCrafter, a unified compact ViT framework for edge dense prediction centered on ECDet, a detection model built from a distilled compact backbone and an edge-friendly encoder decoder design. On the COCO dataset, ECDet-S achieves 51.7 AP with fewer than 10M parameters using only COCO annotations. For instance segmentation, ECInsSeg achieves performance comparable to RF-DETR while using substantially fewer parameters. For pose estimation, ECPose-X reaches 74.8 AP, significantly outperforming YOLO26Pose-X (71.6 AP). These results show that compact ViTs, when paired with task-specialized distillation and edge-aware design, can be a practical and competitive option for edge dense prediction. Code is available at: this https URL
>
---
#### [replaced 004] CheXGenBench: A Unified Benchmark For Fidelity, Privacy and Utility of Synthetic Chest Radiographs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.10496](https://arxiv.org/pdf/2505.10496)**

> **作者:** Raman Dutt; Pedro Sanchez; Yongchen Yao; Steven McDonagh; Sotirios A. Tsaftaris; Timothy Hospedales
>
> **摘要:** We introduce CheXGenBench, a rigorous and multifaceted evaluation framework for synthetic chest radiograph generation that simultaneously assesses fidelity, privacy risks, and clinical utility across state-of-the-art text-to-image generative models. Despite rapid advancements in generative AI for real-world imagery, medical domain evaluations have been hindered by methodological inconsistencies, outdated architectural comparisons, and disconnected assessment criteria that rarely address the practical clinical value of synthetic samples. CheXGenBench overcomes these limitations through standardised data partitioning and a unified evaluation protocol comprising over 20 quantitative metrics that systematically analyse generation quality, potential privacy vulnerabilities, and downstream clinical applicability across 11 leading text-to-image architectures. Our results reveal critical inefficiencies in the existing evaluation protocols, particularly in assessing generative fidelity, leading to inconsistent and uninformative comparisons. Our framework establishes a standardised benchmark for the medical AI community, enabling objective and reproducible comparisons while facilitating seamless integration of both existing and future generative models. Additionally, we release a high-quality, synthetic dataset, SynthCheX-75K, comprising 75K radiographs generated by the top-performing model (Sana 0.6B) in our benchmark to support further research in this critical domain. Through CheXGenBench, we establish a new state-of-the-art and release our framework, models, and SynthCheX-75K dataset at this https URL
>
---
#### [replaced 005] Compositional Image Synthesis with Inference-Time Scaling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.24133](https://arxiv.org/pdf/2510.24133)**

> **作者:** Minsuk Ji; Sanghyeok Lee; Namhyuk Ahn
>
> **备注:** projcet page: this https URL
>
> **摘要:** Despite their impressive realism, modern text-to-image models still struggle with compositionality, often failing to render accurate object counts, attributes, and spatial relations. To address this challenge, we present a training-free framework that combines an object-centric approach with self-refinement to improve layout faithfulness while preserving aesthetic quality. Specifically, we leverage large language models (LLMs) to synthesize explicit layouts from input prompts, and we inject these layouts into the image generation process, where a object-centric vision-language model (VLM) judge reranks multiple candidates to select the most prompt-aligned outcome iteratively. By unifying explicit layout-grounding with self-refine-based inference-time scaling, our framework achieves stronger scene alignment with prompts compared to recent text-to-image models. The code are available at this https URL.
>
---
#### [replaced 006] Zero-Shot Personalized Camera Motion Control for Image-to-Video Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.09472](https://arxiv.org/pdf/2504.09472)**

> **作者:** Pooja Guhan; Divya Kothandaraman; Geonsun Lee; Tsung-Wei Huang; Guan-Ming Su; Dinesh Manocha
>
> **摘要:** Specifying nuanced and compelling camera motion remains a significant hurdle for non-expert creators using generative tools, creating an "expressive gap" where generic text prompts fail to capture cinematic vision. This barrier limits individual creativity and restricts the accessibility of cinematic production for small-scale industries and educational content creators. To address this, we present a zero-shot diffusion-based framework for personalized camera motion control, enabling the transfer of cinematic movements from a single reference video onto a user-provided static image without requiring 3D data, predefined trajectories, or complex graphical interfaces. Our technical contribution involves an inference-time optimization strategy using dual Low-Rank Adaptation (LoRA) networks, with an orthogonality regularizer that encourages separation between spatial appearance and temporal motion updates, alongside a homography-based refinement strategy that provides weak geometric guidance. We evaluate our approach using a new metric, CameraScore, and two distinct user studies. A 72-participant perceptual study demonstrates that our method significantly outperforms existing baselines in motion accuracy (90.45% preference) and scene preservation (70.31% preference). Furthermore, a 12-participant task-based interaction study confirms that our workflow significantly improves usability and creative control (p < 0.001) compared to standard text- or preset-based prompts. We hope this work lays a foundation for future advancements in camera motion transfer across diverse scenes.
>
---
#### [replaced 007] MS-ISSM: Objective Quality Assessment of Point Clouds Using Multi-scale Implicit Structural Similarity
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2601.01200](https://arxiv.org/pdf/2601.01200)**

> **作者:** Zhang Chen; Shuai Wan; Yuezhe Zhang; Siyu Ren; Fuzheng Yang; Junhui Hou
>
> **摘要:** The unstructured and irregular nature of points poses a significant challenge for accurate point cloud quality assessment (PCQA), particularly in establishing accurate perceptual feature correspondence. To tackle this, we propose the Multi-scale Implicit Structural Similarity Measurement (MS-ISSM). Unlike traditional point-to-point matching, MS-ISSM utilizes radial basis function (RBF) to represent local features continuously, transforming distortion measurement into a comparison of implicit function coefficients. This approach effectively circumvents matching errors inherent in irregular data. Additionally, we propose a ResGrouped-MLP quality assessment network, which robustly maps multi-scale feature differences to perceptual scores. The network architecture departs from traditional flat multi-layer perceptron (MLP) by adopting a grouped encoding strategy integrated with residual blocks and channel-wise attention mechanisms. This hierarchical design allows the model to preserve the distinct physical semantics of luma, chroma, and geometry while adaptively focusing on the most salient distortion features across High, Medium, and Low scales. Experimental results on multiple benchmarks demonstrate that MS-ISSM outperforms state-of-the-art metrics in both reliability and generalization. The source code is available at: this https URL.
>
---
#### [replaced 008] UE5-Forest: A Photorealistic Synthetic Stereo Dataset for UAV Forestry Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.15304](https://arxiv.org/pdf/2603.15304)**

> **作者:** Yida Lin; Bing Xue; Mengjie Zhang; Sam Schofield; Richard Green
>
> **摘要:** Dense ground-truth disparity maps are practically unobtainable in forestry environments, where thin overlapping branches and complex canopy geometry defeat conventional depth sensors -- a critical bottleneck for training supervised stereo matching networks for autonomous UAV-based pruning. We present UE5-Forest, a photorealistic synthetic stereo dataset built entirely in Unreal Engine 5 (UE5). One hundred and fifteen photogrammetry-scanned trees from the Quixel Megascans library are placed in virtual scenes and captured by a simulated stereo rig whose intrinsics -- 63 mm baseline, 2.8 mm focal length, 3.84 mm sensor width -- replicate the ZED Mini camera mounted on our drone. Orbiting each tree at up to 2 m across three elevation bands (horizontal, +45 degrees, -45 degrees) yields 5,520 rectified 1920 x 1080 stereo pairs with pixel-perfect disparity labels. We provide a statistical characterisation of the dataset -- covering disparity distributions, scene diversity, and visual fidelity -- and a qualitative comparison with real-world Canterbury Tree Branches imagery that confirms the photorealistic quality and geometric plausibility of the rendered data. The dataset will be publicly released to provide the community with a ready-to-use benchmark and training resource for stereo-based forestry depth estimation.
>
---
#### [replaced 009] EDU-CIRCUIT-HW: Evaluating Multimodal Large Language Models on Real-World University-Level STEM Student Handwritten Solutions
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [https://arxiv.org/pdf/2602.00095](https://arxiv.org/pdf/2602.00095)**

> **作者:** Weiyu Sun; Liangliang Chen; Yongnuo Cai; Huiru Xie; Yi Zeng; Ying Zhang
>
> **摘要:** Multimodal Large Language Models (MLLMs) hold significant promise for revolutionizing traditional education and reducing teachers' workload. However, accurately interpreting unconstrained STEM student handwritten solutions with intertwined mathematical formulas, diagrams, and textual reasoning poses a significant challenge due to the lack of authentic and domain-specific benchmarks. Additionally, current evaluation paradigms predominantly rely on the outcomes of downstream tasks (e.g., auto-grading), which often probe only a subset of the recognized content, thereby failing to capture the MLLMs' understanding of complex handwritten logic as a whole. To bridge this gap, we release EDU-CIRCUIT-HW, a dataset consisting of 1,300+ authentic student handwritten solutions from a university-level STEM course. Utilizing the expert-verified verbatim transcriptions and grading reports of student solutions, we simultaneously evaluate various MLLMs' upstream recognition fidelity and downstream auto-grading performance. Our evaluation uncovers an astonishing scale of latent failures within MLLM-recognized student handwritten content, highlighting the models' insufficient reliability for auto-grading and other understanding-oriented applications in high-stakes educational settings. In solution, we present a case study demonstrating that leveraging identified error patterns to preemptively detect and rectify recognition errors, with only minimal human intervention (e.g., with 3.3% assignments routed to human graders while the rest to GPT-5.1 grader), can effectively enhance the robustness of the deployed AI-enabled grading system on unseen student solutions.
>
---
#### [replaced 010] From Pixels to Patches: Pooling Strategies for Earth Embeddings
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.02080](https://arxiv.org/pdf/2603.02080)**

> **作者:** Isaac Corley; Caleb Robinson; Inbal Becker-Reshef; Juan M. Lavista Ferres
>
> **备注:** ICLR 2026 ML4RS Workshop
>
> **摘要:** Geospatial foundation models increasingly expose pixel-level embedding products that can be downloaded and reused without access to the underlying encoder. In this setting, downstream tasks with patch- or region-level labels require a post-hoc aggregation step that maps dense pixel embeddings to a single representation. The default choice, mean pooling, discards within-patch variability and can underperform under spatial distribution shift. To study this setting, we introduce EuroSAT-Embed: 81,000 embedding GeoTIFFs derived from three foundation models: AlphaEarth, OlmoEarth, and Tessera. Using these fixed embedding products, we benchmark 11 training-free pooling methods and 2 train-set-fitted baselines under both random and geographically disjoint test splits. Richer pooling schemes reduce the geographic generalization gap by over 50% relative to mean pooling and improve accuracy by up to 6% on spatial splits. We recommend a three-tier strategy: (1) mean as a baseline, (2) stats pooling (min/max/mean/std) as the default at 4x the embedding dimension, and (3) covariance pooling for peak accuracy. Across all three embedding products, simple distributional statistics improve spatial-split performance over mean pooling.
>
---
#### [replaced 011] QPT V2: Masked Image Modeling Advances Visual Scoring
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2407.16541](https://arxiv.org/pdf/2407.16541)**

> **作者:** Qizhi Xie; Kun Yuan; Yunpeng Qu; Mingda Wu; Ming Sun; Chao Zhou; Jihong Zhu
>
> **备注:** 8 pages, 6 figures. Accepted by ACM MM 24
>
> **摘要:** Quality assessment and aesthetics assessment aim to evaluate the perceived quality and aesthetics of visual content. Current learning-based methods suffer greatly from the scarcity of labeled data and usually perform sub-optimally in terms of generalization. Although masked image modeling (MIM) has achieved noteworthy advancements across various high-level tasks (e.g., classification, detection etc.). In this work, we take on a novel perspective to investigate its capabilities in terms of quality- and aesthetics-awareness. To this end, we propose Quality- and aesthetics-aware pretraining (QPT V2), the first pretraining framework based on MIM that offers a unified solution to quality and aesthetics assessment. To perceive the high-level semantics and fine-grained details, pretraining data is curated. To comprehensively encompass quality- and aesthetics-related factors, degradation is introduced. To capture multi-scale quality and aesthetic information, model structure is modified. Extensive experimental results on 11 downstream benchmarks clearly show the superior performance of QPT V2 in comparison with current state-of-the-art approaches and other pretraining paradigms.
>
---
#### [replaced 012] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出StreamGaze，解决流视频中利用眼动信号进行时间推理和主动理解的问题，通过构建相关数据集评估MLLMs的表现。**

- **链接: [https://arxiv.org/pdf/2512.01707](https://arxiv.org/pdf/2512.01707)**

> **作者:** Daeun Lee; Subhojyoti Mukherjee; Branislav Kveton; Ryan A. Rossi; Viet Dac Lai; Seunghyun Yoon; Trung Bui; Franck Dernoncourt; Mohit Bansal
>
> **备注:** Accepted to CVPR 2026, Project page: this https URL
>
> **摘要:** Streaming video understanding requires models not only to process temporally incoming frames, but also to anticipate user intention for realistic applications such as Augmented Reality (AR) glasses. While prior streaming benchmarks evaluate temporal reasoning, none measure whether Multimodal Large Language Models (MLLMs) can interpret or leverage human gaze signals within a streaming setting. To fill this gap, we introduce StreamGaze, the first benchmark designed to evaluate how effectively MLLMs utilize gaze for temporal and proactive reasoning in streaming videos. StreamGaze introduces gaze-guided past, present, and proactive tasks that comprehensively assess streaming video understanding. These tasks evaluate whether models can use real-time gaze signals to follow shifting attention and infer user intentions based only on past and currently observed frames. To build StreamGaze, we develop a gaze-video Question Answering (QA) generation pipeline that aligns egocentric videos with raw gaze trajectories through fixation extraction, region-specific visual prompting, and scanpath construction. This pipeline produces spatio-temporally grounded QA pairs that reflect human perceptual dynamics. Across all StreamGaze tasks, we observe substantial performance gaps between state-of-the-art MLLMs and human performance, highlighting key limitations in gaze-based temporal reasoning, intention modeling, and proactive prediction. We further provide detailed analyses of gaze prompting strategies, reasoning behaviors, and task-specific failure modes, offering insights into current limitations and directions for future research. All data and code are publicly available to support continued research in gaze-guided streaming video understanding.
>
---
#### [replaced 013] Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GRIP方法，用于从稀疏IMU和足底压力传感器数据中重建物理合理的运动，解决人体运动捕捉任务中的动态与地面交互问题。**

- **链接: [https://arxiv.org/pdf/2603.16233](https://arxiv.org/pdf/2603.16233)**

> **作者:** Ryosuke Hori; Jyun-Ting Song; Zhengyi Luo; Jinkun Cao; Soyong Shin; Hideo Saito; Kris Kitani
>
> **摘要:** We propose Ground Reaction Inertial Poser (GRIP), a method that reconstructs physically plausible human motion using four wearable devices. Unlike conventional IMU-only approaches, GRIP combines IMU signals with foot pressure data to capture both body dynamics and ground interactions. Furthermore, rather than relying solely on kinematic estimation, GRIP uses a digital twin of a person, in the form of a synthetic humanoid in a physics simulator, to reconstruct realistic and physically plausible motion. At its core, GRIP consists of two modules: KinematicsNet, which estimates body poses and velocities from sensor data, and DynamicsNet, which controls the humanoid in the simulator using the residual between the KinematicsNet prediction and the simulated humanoid state. To enable robust training and fair evaluation, we introduce a large-scale dataset, Pressure and Inertial Sensing for Human Motion and Interaction (PRISM), that captures diverse human motions with synchronized IMUs and insole pressure sensors. Experimental results show that GRIP outperforms existing IMU-only and IMU-pressure fusion methods across all evaluated datasets, achieving higher global pose accuracy and improved physical consistency.
>
---
#### [replaced 014] CLARITY: Medical World Model for Guiding Treatment Decisions by Modeling Context-Aware Disease Trajectories in Latent Space
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08029](https://arxiv.org/pdf/2512.08029)**

> **作者:** Tianxingjian Ding; Yuanhao Zou; Chen Chen; Mubarak Shah; Yu Tian
>
> **摘要:** Clinical decision-making in oncology requires predicting dynamic disease evolution, a task current static AI predictors cannot perform. While world models (WMs) offer a paradigm for generative prediction, existing medical applications remain limited. Existing methods often rely on stochastic diffusion models, focusing on visual reconstruction rather than causal, physiological transitions. Furthermore, in medical domain, models like MeWM typically ignore patient-specific temporal and clinical contexts and lack a feedback mechanism to link predictions to treatment decisions. To address these gaps, we introduce CLARITY, a medical world model that forecasts disease evolution directly within a structured latent space. It explicitly integrates time intervals (temporal context) and patient-specific data (clinical context) to model treatment-conditioned progression as a smooth, interpretable trajectory, and thus generate physiologically faithful, individualized treatment plans. Finally, CLARITY introduces a novel prediction-to-decision framework, translating latent rollouts into transparent, actionable recommendations. CLARITY demonstrates state-of-the-art performance in treatment planning. On the MU-Glioma-Post dataset, our approach outperforms recent MeWM by 12\%, and significantly surpasses all other medical-specific large language models.
>
---
#### [replaced 015] INSIGHT: Enhancing Autonomous Driving Safety through Vision-Language Models on Context-Aware Hazard Detection and Edge Case Evaluation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.00262](https://arxiv.org/pdf/2502.00262)**

> **作者:** Dianwei Chen; Zifan Zhang; Lei Cheng; Yuchen Liu; Xianfeng Terry Yang
>
> **摘要:** Autonomous driving systems face significant challenges in handling unpredictable edge-case scenarios, such as adversarial pedestrian movements, dangerous vehicle maneuvers, and sudden environmental changes. Current end-to-end driving models struggle with generalization to these rare events due to limitations in traditional detection and prediction approaches. To address this, we propose INSIGHT (Integration of Semantic and Visual Inputs for Generalized Hazard Tracking), a hierarchical vision-language model (VLM) framework designed to enhance hazard detection and edge-case evaluation. By using multimodal data fusion, our approach integrates semantic and visual representations, enabling precise interpretation of driving scenarios and accurate forecasting of potential dangers. Through supervised fine-tuning of VLMs, we optimize spatial hazard localization using attention-based mechanisms and coordinate regression techniques. Experimental results on the BDD100K dataset demonstrate a substantial improvement in hazard prediction straightforwardness and accuracy over existing models, achieving a notable increase in generalization performance. This advancement enhances the robustness and safety of autonomous driving systems, ensuring improved situational awareness and potential decision-making in complex real-world scenarios.
>
---
#### [replaced 016] ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视频生成中物理不真实的问题。提出ABot-PhysWorld模型，提升物理合理性与动作控制能力，并引入EZSbench评估基准。**

- **链接: [https://arxiv.org/pdf/2603.23376](https://arxiv.org/pdf/2603.23376)**

> **作者:** Yuzhi Chen; Ronghan Chen; Dongjie Huo; Yandan Yang; Dekang Qi; Haoyun Liu; Tong Lin; Shuang Zeng; Junjin Xiao; Xinyuan Chang; Feng Xiong; Xing Wei; Zhiheng Ma; Mu Xu
>
> **备注:** Code: this https URL
>
> **摘要:** Video-based world models offer a powerful paradigm for embodied simulation and planning, yet state-of-the-art models often generate physically implausible manipulations - such as object penetration and anti-gravity motion - due to training on generic visual data and likelihood-based objectives that ignore physical laws. We present ABot-PhysWorld, a 14B Diffusion Transformer model that generates visually realistic, physically plausible, and action-controllable videos. Built on a curated dataset of three million manipulation clips with physics-aware annotation, it uses a novel DPO-based post-training framework with decoupled discriminators to suppress unphysical behaviors while preserving visual quality. A parallel context block enables precise spatial action injection for cross-embodiment control. To better evaluate generalization, we introduce EZSbench, the first training-independent embodied zero-shot benchmark combining real and synthetic unseen robot-task-scene combinations. It employs a decoupled protocol to separately assess physical realism and action alignment. ABot-PhysWorld achieves new state-of-the-art performance on PBench and EZSbench, surpassing Veo 3.1 and Sora v2 Pro in physical plausibility and trajectory consistency. We will release EZSbench to promote standardized evaluation in embodied video generation.
>
---
#### [replaced 017] Rethinking Diffusion Model-Based Video Super-Resolution: Leveraging Dense Guidance from Aligned Features
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16928](https://arxiv.org/pdf/2511.16928)**

> **作者:** Jingyi Xu; Meisong Zheng; Ying Chen; Minglang Qiao; Xin Deng; Mai Xu
>
> **备注:** Accepted by CVPR 2026,20pages
>
> **摘要:** Diffusion model (DM) based Video Super-Resolution (VSR) approaches achieve impressive perceptual quality. However, they suffer from error accumulation, spatial artifacts, and a trade-off between perceptual quality and fidelity, primarily caused by inaccurate alignment and insufficient compensation between video frames. In this paper, within the DM-based VSR pipeline, we revisit the role of alignment and compensation between adjacent video frames and reveal two crucial observations: (a) the feature domain is better suited than the pixel domain for information compensation due to its stronger spatial and temporal correlations, and (b) warping at an upscaled resolution better preserves high-frequency information, but this benefit is not necessarily monotonic. Therefore, we propose a novel Densely Guided diffusion model with Aligned Features for Video Super-Resolution (DGAF-VSR), with an Optical Guided Warping Module (OGWM) to maintain high-frequency details in the aligned features and a Feature-wise Temporal Condition Module (FTCM) to deliver dense guidance in the feature domain. Extensive experiments on synthetic and real-world datasets demonstrate that DGAF-VSR surpasses state-of-the-art methods in key aspects of VSR, including perceptual quality (35.82\% DISTS reduction), fidelity (0.20 dB PSNR gain), and temporal consistency (30.37\% tLPIPS reduction).
>
---
#### [replaced 018] UniPart: Part-Level 3D Generation with Unified 3D Geom-Seg Latents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.09435](https://arxiv.org/pdf/2512.09435)**

> **作者:** Xufan He; Yushuang Wu; Xiaoyang Guo; Chongjie Ye; Jiaqing Zhou; Tianlei Hu; Xiaoguang Han; Dong Du
>
> **备注:** Project page: this https URL
>
> **摘要:** Part-level 3D generation is essential for applications requiring decomposable and structured 3D synthesis. However, existing methods either rely on implicit part segmentation with limited granularity control or depend on strong external segmenters trained on large annotated datasets. In this work, we observe that part awareness emerges naturally during whole-object geometry learning and propose Geom-Seg VecSet, a unified geometry-segmentation latent representation that jointly encodes object geometry and part-level structure. Building on this representation, we introduce UniPart, a two-stage latent diffusion framework for image-guided part-level 3D generation. The first stage performs joint geometry generation and latent part segmentation, while the second stage conditions part-level diffusion on both whole-object and part-specific latents. A dual-space generation scheme further enhances geometric fidelity by predicting part latents in both global and canonical spaces. Extensive experiments demonstrate that UniPart achieves superior segmentation controllability and part-level geometric quality compared with existing approaches.
>
---
#### [replaced 019] SSeg: Active Sparse Point-Label Augmentation for Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.10163](https://arxiv.org/pdf/2510.10163)**

> **作者:** Cesar Borja; Carlos Plou; Ruben Martinez-Cantin; Ana C. Murillo
>
> **摘要:** Semantic segmentation is essential for automating remote sensing analysis in fields like ecology. However, fine-grained analysis of complex aerial or underwater imagery remains an open challenge, even for state-of-the-art models. Progress is frequently hindered by the high cost of obtaining the dense, expert-annotated labels required for model supervision. While sparse point-labels are easier to obtain, they introduce challenges regarding which points to annotate and how to propagate the sparse information. We present SSeg, a novel framework that addresses both issues. SSeg first employs an active sampling strategy to guide annotators, maximizing the value of their point labels. Then, it propagates these sparse labels with a hybrid approach leveraging both the best of SAM2 and superpixel-based methods. Experiments on two diverse monitoring datasets demonstrate SSeg's benefits over state-of-the-art approaches. Our main contribution is a simple but effective interactive annotation tool integrating our algorithms. It enables ecology researchers to leverage foundation models and computer vision to efficiently generate high-quality segmentation masks to process their data.
>
---
#### [replaced 020] LagerNVS: Latent Geometry for Fully Neural Real-time Novel View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20176](https://arxiv.org/pdf/2603.20176)**

> **作者:** Stanislaw Szymanowicz; Minghao Chen; Jianyuan Wang; Christian Rupprecht; Andrea Vedaldi
>
> **备注:** IEEE CVF Conference on Computer Vision and Pattern Recognition 2026. Project page with code, models and examples: this http URL
>
> **摘要:** Recent work has shown that neural networks can perform 3D tasks such as Novel View Synthesis (NVS) without explicit 3D reconstruction. Even so, we argue that strong 3D inductive biases are still helpful in the design of such networks. We show this point by introducing LagerNVS, an encoder-decoder neural network for NVS that builds on `3D-aware' latent features. The encoder is initialized from a 3D reconstruction network pre-trained using explicit 3D supervision. This is paired with a lightweight decoder, and trained end-to-end with photometric losses. LagerNVS achieves state-of-the-art deterministic feed-forward Novel View Synthesis (including 31.4 PSNR on Re10k), with and without known cameras, renders in real time, generalizes to in-the-wild data, and can be paired with a diffusion decoder for generative extrapolation.
>
---
#### [replaced 021] Gaussian Mapping for Evolving Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.06909](https://arxiv.org/pdf/2506.06909)**

> **作者:** Vladimir Yugay; Thies Kersten; Luca Carlone; Theo Gevers; Martin R. Oswald; Lukas Schmid
>
> **摘要:** Mapping systems with novel view synthesis (NVS) capabilities, most notably 3D Gaussian Splatting (3DGS), are widely used in computer vision, as well as in various applications, including augmented reality, robotics, and autonomous driving. However, many current approaches are limited to static scenes. While recent works have begun addressing short-term dynamics (motion within the camera's view), long-term dynamics (the scene evolving through changes out of view) remain less explored. To overcome this limitation, we introduce a dynamic scene adaptation mechanism to continuously update 3DGS to reflect the latest changes. Since maintaining consistency remains challenging due to stale observations disrupting the reconstruction process, we further propose a novel keyframe management mechanism that discards outdated observations while preserving as much information as possible. We thoroughly evaluate Gaussian Mapping for Evolving Scenes (GaME) on both synthetic and real-world datasets, achieving a 29.7% improvement in PSNR and a 3 times improvement in L1 depth error over the most competitive baseline.
>
---
#### [replaced 022] EPOFusion: Exposure aware Progressive Optimization Method for Infrared and Visible Image Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16130](https://arxiv.org/pdf/2603.16130)**

> **作者:** Zhiwei Wang; Yayu Zheng; Defeng He; Li Zhao; Xiaoqin Zhang; Yuxing Li; Edmund Y. Lam
>
> **摘要:** Overexposure frequently occurs in practical scenarios, causing the loss of critical visual information. However, existing infrared and visible fusion methods still exhibit unsatisfactory performance in highly bright regions. To address this, we propose EPOFusion, an exposure-aware fusion model. Specifically, a guidance module is introduced to facilitate the encoder in extracting fine-grained infrared features from overexposed regions. Meanwhile, an iterative decoder incorporating a multiscale context fusion module is designed to progressively enhance the fused image, ensuring consistent details and superior visual quality. Finally, an adaptive loss function dynamically constrains the fusion process, enabling an effective balance between the modalities under varying exposure conditions. To achieve better exposure awareness, we construct the first infrared and visible overexposure dataset (IVOE) with high quality infrared guided annotations for overexposed regions. Extensive experiments show that EPOFusion outperforms existing methods. It maintains infrared cues in overexposed regions while achieving visually faithful fusion in non-overexposed areas, thereby enhancing both visual fidelity and downstream task performance. Code, fusion results and IVOE dataset will be made available at this https URL.
>
---
#### [replaced 023] When to Think and When to Look: Uncertainty-Guided Lookback
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决测试时思考效果不佳的问题。通过分析不同思考长度对视觉推理的影响，提出基于不确定性的回溯策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.15613](https://arxiv.org/pdf/2511.15613)**

> **作者:** Jing Bi; Filippos Bellos; Junjia Guo; Yayuan Li; Chao Huang; Yolo Y. Tang; Luchuan Song; Susan Liang; Zhongfei Mark Zhang; Jason J. Corso; Chenliang Xu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets.
>
---
#### [replaced 024] Revisiting Diffusion Model Predictions Through Dimensionality
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.21419](https://arxiv.org/pdf/2601.21419)**

> **作者:** Qing Jin; Chaoyang Wang
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Recent advances in diffusion and flow matching models have highlighted a shift in the preferred prediction target -- moving from noise ($\varepsilon$) and velocity (v) to direct data (x) prediction -- particularly in high-dimensional settings. However, a formal explanation of why the optimal target depends on the specific properties of the data remains elusive. In this work, we provide a theoretical framework based on a generalized prediction formulation that accommodates arbitrary output targets, of which $\varepsilon$-, v-, and x-prediction are special cases. We derive the analytical relationship between data's geometry and the optimal prediction target, offering a rigorous justification for why x-prediction becomes superior when the ambient dimension significantly exceeds the data's intrinsic dimension. Furthermore, while our theory identifies dimensionality as the governing factor for the optimal prediction target, the intrinsic dimension of manifold-bound data is typically intractable to estimate in practice. To bridge this gap, we propose k-Diff, a framework that employs a data-driven approach to learn the optimal prediction parameter k directly from data, bypassing the need for explicit dimension estimation. Extensive experiments in both latent-space and pixel-space image generation demonstrate that k-Diff consistently outperforms fixed-target baselines across varying architectures and data scales, providing a principled and automated approach to enhancing generative performance.
>
---
#### [replaced 025] IVEBench: Modern Benchmark Suite for Instruction-Guided Video Editing Assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.11647](https://arxiv.org/pdf/2510.11647)**

> **作者:** Yinan Chen; Jiangning Zhang; Teng Hu; Yuxiang Zeng; Zhucun Xue; Qingdong He; Chengjie Wang; Yong Liu; Xiaobin Hu; Shuicheng Yan
>
> **备注:** Accepted by ICLR 2026. Equal contributions from first two authors. Project page: this https URL Code: this https URL Dataset: this https URL
>
> **摘要:** Instruction-guided video editing has emerged as a rapidly advancing research direction, offering new opportunities for intuitive content transformation while also posing significant challenges for systematic evaluation. Existing video editing benchmarks fail to support the evaluation of instruction-guided video editing adequately and further suffer from limited source diversity, narrow task coverage and incomplete evaluation metrics. To address the above limitations, we introduce IVEBench, a modern benchmark suite specifically designed for instruction-guided video editing assessment. IVEBench comprises a diverse database of 600 high-quality source videos, spanning seven semantic dimensions, and covering video lengths ranging from 32 to 1,024 frames. It further includes 8 categories of editing tasks with 35 subcategories, whose prompts are generated and refined through large language models and expert review. Crucially, IVEBench establishes a three-dimensional evaluation protocol encompassing video quality, instruction compliance and video fidelity, integrating both traditional metrics and multimodal large language model-based assessments. Extensive experiments demonstrate the effectiveness of IVEBench in benchmarking state-of-the-art instruction-guided video editing methods, showing its ability to provide comprehensive and human-aligned evaluation outcomes.
>
---
#### [replaced 026] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.17468](https://arxiv.org/pdf/2601.17468)**

> **作者:** Chia-Ming Lee; Yu-Fan Lin; Jin-Hui Jiang; Yu-Jou Hsiao; Chih-Chung Hsu; Yu-Lun Liu
>
> **备注:** CVPR 2026 Camera Ready; Project page: this https URL
>
> **摘要:** Single Image Reflection Separation (SIRS) disentangles mixed images into transmission and reflection layers. Existing methods suffer from transmission-reflection confusion under nonlinear mixing, particularly in deep decoder layers, due to implicit fusion mechanisms and inadequate multi-scale coordination. We propose ReflexSplit, a dual-stream framework with three key innovations. (1) Cross-scale Gated Fusion (CrGF) adaptively aggregates semantic priors, texture details, and decoder context across hierarchical depths, stabilizing gradient flow and maintaining feature consistency. (2) Layer Fusion-Separation Blocks (LFSB) alternate between fusion for shared structure extraction and differential separation for layer-specific disentanglement. Inspired by Differential Transformer, we extend attention cancellation to dual-stream separation via cross-stream subtraction. (3) Curriculum training progressively strengthens differential separation through depth-dependent initialization and epoch-wise warmup. Extensive experiments on synthetic and real-world benchmarks demonstrate state-of-the-art performance with superior perceptual quality and robust generalization. Our code is available at this https URL.
>
---
#### [replaced 027] Enhancing Neural Video Compression of Static Scenes with Positive-Incentive Noise
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.06095](https://arxiv.org/pdf/2603.06095)**

> **作者:** Cheng Yuan; Zhenyu Jia; Jiawei Shao; Xuelong Li
>
> **摘要:** Static scene videos, such as surveillance feeds and videotelephony streams, constitute a dominant share of storage consumption and network traffic. However, both traditional standardized codecs and neural video compression (NVC) methods struggle to encode these videos efficiently due to inadequate usage of temporal redundancy and severe distribution gaps between training and test data, respectively. While recent generative compression methods improve perceptual quality, they introduce hallucinated details that are unacceptable in authenticity-critical applications. To overcome these limitations, we propose a positive-incentive camera (PIC) framework for static scene videos, where short-term temporal changes are reinterpreted as positive-incentive noise to facilitate NVC model finetuning. By disentangling transient variations from the persistent background, structured prior information is internalized in the compression model. During inference, the invariant component requires minimal signaling, thus reducing data transmission while maintaining pixel-level fidelity. Experiment results show that PIC achieves visually lossless reconstruction for static scenes at an extremely low compression rate of 0.009%, while the DCVC-FM baseline requires 20.5% higher Bjøntegaard delta (BD) rate. Our method provides an effective solution to trade computation for bandwidth, enabling robust video transmission under adverse network conditions and economic long-term retention of surveillance footage.
>
---
#### [replaced 028] Towards Knowledge Guided Pretraining Approaches for Multimodal Foundation Models: Applications in Remote Sensing
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2407.19660](https://arxiv.org/pdf/2407.19660)**

> **作者:** Praveen Ravirathinam; Ajitesh Parthasarathy; Ankush Khandelwal; Rahul Ghosh; Vipin Kumar
>
> **备注:** 33 pages with appendix
>
> **摘要:** Self-supervised learning has emerged as a powerful paradigm for pretraining foundation models using large-scale data. Existing pretraining approaches predominantly rely on masked reconstruction or next-token prediction strategies, demonstrating strong performance across various downstream tasks, including geoscience applications. However, these approaches do not fully capture the knowledge of causal interplay between different geospatial and environmental variables. To address this limitation, we propose Knowledge Guided Variable-Step Forecasting (KG-VSF), a novel pretraining task that models forecasting as a conditional generation task, where driver variables (e.g., weather) inform the prediction of response variables (e.g., satellite imagery). We demonstrate that pretraining in such a fashion leads to strong embeddings which give enhanced performance when finetuned on downstream tasks where capturing this causality matters such as pixel wise crop type mapping, soil moisture estimation and forecasting, missing image prediction, and future image forecasting when compared to finetuning embeddings from other standard pretraining approaches.
>
---
#### [replaced 029] Leveraging Arbitrary Data Sources for AI-Generated Image Detection Without Sacrificing Generalization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00717](https://arxiv.org/pdf/2603.00717)**

> **作者:** Qinghui He; Haifeng Zhang; Xiuli Bi; Bo Liu; Chi-Man Pun; Bin Xiao
>
> **备注:** Accepted to CVPR Findings 2026
>
> **摘要:** The accelerating advancement of generative models has introduced new challenges for detecting AI-generated images, especially in real-world scenarios where novel generation techniques emerge rapidly. Existing learning paradigms are likely to make classifiers data-dependent, resulting in narrow decision margins and, consequently, limited generalization ability to unseen generative models. We observe that both real and generated images intend to form clustered low-dimensional manifolds within high-level feature spaces extracted by pre-trained visual encoders. Building on this observation, we propose a single-class attribution modeling framework that first amplifies the intrinsic differences between real and generated images by constructing a compact attribution space from any single-class training set, either composed of real images or generated ones, and then establishes a more stable decision boundary upon the enlarged separation. This process enhances class distinction and mitigates the reliance on generator-specific artifacts, thereby improving cross-model generalization. Extensive experiments show that our method generalizes well across various unseen generative models, outperforming existing detectors by as much as 7.21% in accuracy and 7.20% in cross-model generalization.
>
---
#### [replaced 030] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22949](https://arxiv.org/pdf/2602.22949)**

> **作者:** Junuk Cha; Jihyeon Kim; Han-Mu Park
>
> **备注:** Accepted to CVPR 2026, camera-ready version
>
> **摘要:** Fingerspelling is a component of sign languages in which words are spelled out letter by letter using specific hand poses. Automatic fingerspelling recognition plays a crucial role in bridging the communication gap between Deaf and hearing communities, yet it remains challenging due to the signing-hand ambiguity issue, the lack of appropriate training losses, and the out-of-vocabulary (OOV) problem. Prior fingerspelling recognition methods rely on explicit signing-hand detection, which often leads to recognition failures, and on a connectionist temporal classification (CTC) loss, which exhibits the peaky behavior problem. To address these issues, we develop OpenFS, an open-source approach for fingerspelling recognition and synthesis. We propose a multi-hand-capable fingerspelling recognizer that supports both single- and multi-hand inputs and performs implicit signing-hand detection by incorporating a dual-level positional encoding and a signing-hand focus (SF) loss. The SF loss encourages cross-attention to focus on the signing hand, enabling implicit signing-hand detection during recognition. Furthermore, without relying on the CTC loss, we introduce a monotonic alignment (MA) loss that enforces the output letter sequence to follow the temporal order of the input pose sequence through cross-attention regularization. In addition, we propose a frame-wise letter-conditioned generator that synthesizes realistic fingerspelling pose sequences for OOV words. This generator enables the construction of a new synthetic benchmark, called FSNeo. Through comprehensive experiments, we demonstrate that our approach achieves state-of-the-art performance in recognition and validate the effectiveness of the proposed recognizer and generator. Codes and data are available in: this https URL.
>
---
#### [replaced 031] Binary Verification for Zero-Shot Vision
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10983](https://arxiv.org/pdf/2511.10983)**

> **作者:** Rongbin Hu; Jeffrey Liu
>
> **摘要:** We propose a training-free, binary verification workflow for zero-shot vision with off-the-shelf VLMs. It comprises two steps: (i) quantization, which turns the open-ended query into a multiple-choice question (MCQ) with a small, explicit list of unambiguous candidates; and (ii) binarization, which asks one True/False question per candidate and resolves deterministically: if exactly one is True, select it; otherwise, revert to an MCQ over the remaining plausible candidates. We evaluate the workflow on referring expression grounding (REC), spatial reasoning (Spatial-Map, Spatial-Grid, Spatial-Maze), and BLINK-Jigsaw. Relative to answering open-ended queries directly, quantization to MCQ yields large gains, and True/False binarization provides a consistent additional boost. Across all tasks, the same workflow produces significant improvements, indicating generality. We further integrate the proposed REC workflow into a real-world video processing and editing system, and present the system architecture and end-to-end pipeline in the paper. Together, these components yield a simple and unified workflow that emphasizes inference-time design over task-specific training. It offers a practical, drop-in path to stronger zero-shot vision with today's VLMs.
>
---
#### [replaced 032] Smol-GS: Compact Representations for Abstract 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00850](https://arxiv.org/pdf/2512.00850)**

> **作者:** Haishan Wang; Mohammad Hassan Vali; Arno Solin
>
> **摘要:** We present Smol-GS, a novel method for learning compact representations for 3D Gaussian Splatting (3DGS). Our approach learns highly efficient splat-wise features to model 3D space which capture abstracted cues, including color, opacity, transformation, and material properties. We propose octree-derived positional encoding, which explicitly models spatial locality and enhances representation efficiency. We further apply entropy-based compression to exploit feature redundancy, and compress splat coordinates using a recursive voxel hierarchy. This design enables orders-of-magnitude storage reduction while preserving representation flexibility. Smol-GS achieves state-of-the-art compression performance on standard benchmarks with high-level rendering quality.
>
---
#### [replaced 033] Hear What Matters! Text-conditioned Selective Video-to-Audio Generation
- **分类: cs.CV; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出文本条件的视频到音频生成任务，旨在从多对象视频中提取用户指定的声音。工作包括模型SELVA设计及自监督视频混合方案。**

- **链接: [https://arxiv.org/pdf/2512.02650](https://arxiv.org/pdf/2512.02650)**

> **作者:** Junwon Lee; Juhan Nam; Jiyoung Lee
>
> **备注:** accepted to CVPR 2026
>
> **摘要:** This work introduces a new task, text-conditioned selective video-to-audio (V2A) generation, which produces only the user-intended sound from a multi-object video. This capability is especially crucial in multimedia production, where audio tracks are handled individually for each sound source for precise editing, mixing, and creative control. We propose SELVA, a novel text-conditioned V2A model that treats the text prompt as an explicit selector to distinctly extract prompt-relevant sound-source visual features from the video encoder. To suppress text-irrelevant activations with efficient video encoder finetuning, the proposed supplementary tokens promote cross-attention to yield robust semantic and temporal grounding. SELVA further employs an autonomous video-mixing scheme in a self-supervised manner to overcome the lack of mono audio track supervision. We evaluate SELVA on VGG-MONOAUDIO, a curated benchmark of clean single-source videos for such a task. Extensive experiments and ablations consistently verify its effectiveness across audio quality, semantic alignment, and temporal synchronization.
>
---
#### [replaced 034] CARPE: Context-Aware Image Representation Prioritization via Ensemble for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.13622](https://arxiv.org/pdf/2601.13622)**

> **作者:** Donghee Lee; Rui Cai; Zhe Zhao
>
> **摘要:** Large vision-language models (LVLMs) are typically trained using autoregressive language modeling objectives, which align visual representations with linguistic space. While effective for multimodal reasoning, this alignment can weaken vision-centric capabilities, causing LVLMs to underperform their base vision encoders on tasks such as image classification. To address this limitation, we propose Context-Aware Image Representation Prioritization via Ensemble (CARPE), a lightweight framework that integrates raw vision features with aligned LLM representations through vision-integration layers and a context-aware ensemble mechanism. This design enhances the model's ability to adaptively weight visual and textual modalities and enables the model to capture various aspects of image representations. Extensive experiments demonstrate that CARPE improves performance on both image classification and diverse vision-language benchmarks. Our results suggest that modality balancing plays a critical role in multimodal generalization by improving representation utilization within autoregressive LVLMs.
>
---
#### [replaced 035] RoAD Benchmark: How LiDAR Models Fail under Coupled Domain Shifts and Label Evolution
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.07855](https://arxiv.org/pdf/2601.07855)**

> **作者:** Subeen Lee; Siyeong Lee; Namil Kim; Jaesik Choi
>
> **摘要:** For 3D perception systems to operate reliably in real-world environments, they must remain robust to evolving sensor characteristics and changes in object taxonomies. However, existing adaptive learning paradigms struggle in LiDAR settings where domain shifts and label-space evolution occur simultaneously. We introduce \textbf{Robust Autonomous Driving under Dataset shifts (RoAD)}, a benchmark for evaluating model robustness in LiDAR-based object classification under intertwined domain shifts and label evolution, including subclass refinement, unseen-class insertion, and label expansion. RoAD evaluates three learning scenarios with increasing adaptation, from fixed representations (zero-shot transfer and linear probing) to sequential updates (continual learning). Experiments span large-scale autonomous driving datasets, including Waymo, nuScenes, and Argoverse2. Our analysis identifies central failure modes: (i) \textit{limited transferability} under subclass refinement and unseen-class insertion, and on non-vehicle class; and (ii) \textit{accelerated forgetting during continual adaptation}, driven by feature collapse and self-supervised learning objectives.
>
---
#### [replaced 036] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22025](https://arxiv.org/pdf/2602.22025)**

> **作者:** Shuang Song; Debao Huang; Deyan Deng; Haolin Xiong; Yang Tang; Yajie Zhao; Rongjun Qin
>
> **备注:** CVPR 2026
>
> **摘要:** Intrinsic image decomposition (IID) of outdoor scenes is crucial for relighting, editing, and understanding large-scale environments, but progress has been limited by the lack of real-world datasets with reliable albedo and shading supervision. We introduce Olbedo, a large-scale aerial dataset for outdoor albedo--shading decomposition in the wild. Olbedo contains 5,664 UAV images captured across four landscape types, multiple years, and diverse illumination conditions. Each view is accompanied by multi-view consistent albedo and shading maps, metric depth, surface normals, sun and sky shading components, camera poses, and, for recent flights, measured HDR sky domes. These annotations are derived from an inverse-rendering refinement pipeline over multi-view stereo reconstructions and calibrated sky illumination, together with per-pixel confidence masks. We demonstrate that Olbedo enables state-of-the-art diffusion-based IID models, originally trained on synthetic indoor data, to generalize to real outdoor imagery: fine-tuning on Olbedo significantly improves single-view outdoor albedo prediction on the MatrixCity benchmark. We further illustrate applications of Olbedo-trained models to multi-view consistent relighting of 3D assets, material editing, and scene change analysis for urban digital twins. We release the dataset, baseline models, and an evaluation protocol to support future research in outdoor intrinsic decomposition and illumination-aware aerial vision.
>
---
#### [replaced 037] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CoMo方法，用于从互联网视频中学习连续潜在运动，解决机器人学习中的信息丢失和动态捕捉不足问题。**

- **链接: [https://arxiv.org/pdf/2505.17006](https://arxiv.org/pdf/2505.17006)**

> **作者:** Jiange Yang; Yansong Shi; Haoyi Zhu; Mingyu Liu; Kaijing Ma; Yating Wang; Gangshan Wu; Tong He; Limin Wang
>
> **备注:** CVPR 2026
>
> **摘要:** Unsupervised learning of latent motion from Internet videos is crucial for robot learning. Existing discrete methods generally mitigate the shortcut learning caused by extracting excessive static backgrounds through vector quantization with a small codebook size. However, they suffer from information loss and struggle to capture more complex and fine-grained dynamics. Moreover, there is an inherent gap between the distribution of discrete latent motion and continuous robot action, which hinders the joint learning of a unified policy. We propose CoMo, which aims to learn more precise continuous latent motion from internet-scale videos. CoMo employs an early temporal difference (Td) mechanism to increase the shortcut learning difficulty and explicitly enhance motion cues. Additionally, to ensure latent motion better captures meaningful foregrounds, we further propose a temporal contrastive learning (Tcl) scheme. Specifically, positive pairs are constructed with a small future frame temporal offset, while negative pairs are formed by directly reversing the temporal direction. The proposed Td and Tcl work synergistically and effectively ensure that the latent motion focuses better on the foreground and reinforces motion cues. Critically, CoMo exhibits strong zeroshot generalization, enabling it to generate effective pseudo action labels for unseen videos. Extensive simulated and real-world experiments show that policies co-trained with CoMo pseudo action labels achieve superior performance with both diffusion and auto-regressive architectures.
>
---
#### [replaced 038] Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2405.00181](https://arxiv.org/pdf/2405.00181)**

> **作者:** Hang Du; Sicheng Zhang; Binzhu Xie; Guoshun Nan; Jiayang Zhang; Junrui Xu; Hangyu Liu; Sicong Leng; Jiangming Liu; Hehe Fan; Dajiu Huang; Jing Feng; Linli Chen; Can Zhang; Xuhuan Li; Hao Zhang; Jianhang Chen; Qimei Cui; Xiaofeng Tao
>
> **备注:** Accepted in CVPR2024, Codebase: this https URL
>
> **摘要:** Video anomaly understanding (VAU) aims to automatically comprehend unusual occurrences in videos, thereby enabling various applications such as traffic surveillance and industrial manufacturing. While existing VAU benchmarks primarily concentrate on anomaly detection and localization, our focus is on more practicality, prompting us to raise the following crucial questions: "what anomaly occurred?", "why did it happen?", and "how severe is this abnormal event?". In pursuit of these answers, we present a comprehensive benchmark for Causation Understanding of Video Anomaly (CUVA). Specifically, each instance of the proposed benchmark involves three sets of human annotations to indicate the "what", "why" and "how" of an anomaly, including 1) anomaly type, start and end times, and event descriptions, 2) natural language explanations for the cause of an anomaly, and 3) free text reflecting the effect of the abnormality. In addition, we also introduce MMEval, a novel evaluation metric designed to better align with human preferences for CUVA, facilitating the measurement of existing LLMs in comprehending the underlying cause and corresponding effect of video anomalies. Finally, we propose a novel prompt-based method that can serve as a baseline approach for the challenging CUVA. We conduct extensive experiments to show the superiority of our evaluation metric and the prompt-based approach. Our code and dataset are available at this https URL.
>
---
#### [replaced 039] EVA: Efficient Reinforcement Learning for End-to-End Video Agent
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EVA，一种高效的端到端视频智能体框架，解决长视频理解中的效率与适应性问题。通过强化学习实现规划优先的推理，提升视频理解性能。**

- **链接: [https://arxiv.org/pdf/2603.22918](https://arxiv.org/pdf/2603.22918)**

> **作者:** Yaolun Zhang; Ruohui Wang; Jiahao Wang; Yepeng Tang; Xuanyu Zheng; Haonan Duan; Hao Lu; Hanming Deng; Lewei Lu
>
> **备注:** CVPR2026
>
> **摘要:** Video understanding with multimodal large language models (MLLMs) remains challenging due to the long token sequences of videos, which contain extensive temporal dependencies and redundant frames. Existing approaches typically treat MLLMs as passive recognizers, processing entire videos or uniformly sampled frames without adaptive reasoning. Recent agent-based methods introduce external tools, yet still depend on manually designed workflows and perception-first strategies, resulting in inefficiency on long videos. We present EVA, an Efficient Reinforcement Learning framework for End-to-End Video Agent, which enables planning-before-perception through iterative summary-plan-action-reflection reasoning. EVA autonomously decides what to watch, when to watch, and how to watch, achieving query-driven and efficient video understanding. To train such agents, we design a simple yet effective three-stage learning pipeline - comprising supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Group Relative Policy Optimization (GRPO) - that bridges supervised imitation and reinforcement learning. We further construct high-quality datasets for each stage, supporting stable and reproducible training. We evaluate EVA on six video understanding benchmarks, demonstrating its comprehensive capabilities. Compared with existing baselines, EVA achieves a substantial improvement of 6-12% over general MLLM baselines and a further 1-3% gain over prior adaptive agent methods.
>
---
#### [replaced 040] FastCache: Fast Caching for Diffusion Transformer Through Learnable Linear Approximation
- **分类: cs.LG; cs.AI; cs.CV; cs.MM; cs.PF**

- **链接: [https://arxiv.org/pdf/2505.20353](https://arxiv.org/pdf/2505.20353)**

> **作者:** Dong Liu; Yanxuan Yu; Jiayi Zhang; Yifan Li; Ben Lengerich; Ying Nian Wu
>
> **摘要:** Diffusion Transformers (DiT) are powerful generative models but remain computationally intensive due to their iterative structure and deep transformer stacks. To alleviate this inefficiency, we propose \textbf{FastCache}, a hidden-state-level caching and compression framework that accelerates DiT inference by exploiting redundancy within the model's internal representations. FastCache introduces a dual strategy: (1) a spatial-aware token selection mechanism that adaptively filters redundant tokens based on hidden-state saliency, and (2) a transformer-level cache that reuses latent activations across timesteps when changes fall below a predefined threshold. These modules work jointly to reduce unnecessary computation while preserving generation fidelity through learnable linear approximation. Theoretical analysis shows that FastCache maintains bounded approximation error under a hypothesis-testing-based decision rule. Empirical evaluations across multiple DiT variants demonstrate substantial reductions in latency and memory usage, achieving the best generation quality among existing cache methods, as measured by FID and t-FID. To further improve the speedup of FastCache, we also introduce a token merging module that merges redundant tokens based on k-NN density. Code is available at \href{this https URL}{this https URL}.
>
---
#### [replaced 041] Fast-dVLA: Accelerating Discrete Diffusion VLA to Real-Time Performance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决预训练VLA模型在微调中性能提升有限且成本高的问题。通过解耦辅助任务目标，提升模型能力并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2603.25661](https://arxiv.org/pdf/2603.25661)**

> **作者:** Wenxuan Song; Jiayi Chen; Shuai Chen; Jingbo Wang; Pengxiang Ding; Han Zhao; Yikai Qin; Xinhu Zheng; Donglin Wang; Yan Wang; Haoang Li
>
> **摘要:** This paper proposes a novel approach to address the challenge that pretrained VLA models often fail to effectively improve performance and reduce adaptation costs during standard supervised finetuning (SFT). Some advanced finetuning methods with auxiliary training objectives can improve performance and reduce the number of convergence steps. However, they typically incur significant computational overhead due to the additional losses from auxiliary tasks. To simultaneously achieve the enhanced capabilities of auxiliary training with the simplicity of standard SFT, we decouple the two objectives of auxiliary task training within the parameter space, namely, enhancing general capabilities and fitting task-specific action distributions. To deliver this goal, we only need to train the model to converge on a small-scale task set using two distinct training strategies. The difference between the resulting model parameters can then be interpreted as capability vectors provided by auxiliary tasks. These vectors are then merged with pretrained parameters to form a capability-enhanced meta model. Moreover, when standard SFT is augmented with a lightweight orthogonal regularization loss, the merged model attains performance comparable to auxiliary finetuned baselines with reduced computational overhead. Experimental results demonstrate that this approach is highly effective across diverse robot tasks. Project page: this https URL
>
---
#### [replaced 042] BeetleFlow: An Integrative Deep Learning Pipeline for Beetle Image Processing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00255](https://arxiv.org/pdf/2511.00255)**

> **作者:** Fangxun Liu; S M Rayeed; Samuel Stevens; Alyson East; Cheng Hsuan Chiang; Colin Lee; Daniel Yi; Junke Yang; Tejas Naik; Ziyi Wang; Connor Kilrain; Elijah H Buckwalter; Jiacheng Hou; Saul Ibaven Bueno; Shuheng Wang; Xinyue Ma; Yifan Liu; Zhiyuan Tao; Ziheng Zhang; Eric Sokol; Michael Belitz; Sydne Record; Charles V. Stewart; Wei-Lun Chao
>
> **备注:** 4 pages, NeurIPS 2025 Workshop Imageomics
>
> **摘要:** In entomology and ecology research, biologists often need to collect a large number of insects, among which beetles are the most common species. A common practice for biologists to organize beetles is to place them on trays and take a picture of each tray. Given the images of thousands of such trays, it is important to have an automated pipeline to process the large-scale data for further research. Therefore, we develop a 3-stage pipeline to detect all the beetles on each tray, sort and crop the image of each beetle, and do morphological segmentation on the cropped beetles. For detection, we design an iterative process utilizing a transformer-based open-vocabulary object detector and a vision-language model. For segmentation, we manually labeled 670 beetle images and fine-tuned two variants of a transformer-based segmentation model to achieve fine-grained segmentation of beetles with relatively high accuracy. The pipeline integrates multiple deep learning methods and is specialized for beetle image processing, which can greatly improve the efficiency to process large-scale beetle data and accelerate biological research.
>
---
#### [replaced 043] Any4D: Open-Prompt 4D Generation from Natural Language and Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18746](https://arxiv.org/pdf/2511.18746)**

> **作者:** Hao Li; Qiao Sun
>
> **备注:** The authors identified issues in the 4D generation pipeline and evaluation that affect result validity. To ensure scientific accuracy, we will revise the methodology and experiments thoroughly before resubmitting. This version should not be cited or relied upon
>
> **摘要:** While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a \textit{"GPT moment"} in the embodied domain. There is a naive observation: \textit{the diversity of embodied data far exceeds the relatively small space of possible primitive motions}. Based on this insight, we propose \textbf{Primitive Embodied World Models} (PEWM), which restricts video generation to fixed shorter horizons, our approach \textit{1) enables} fine-grained alignment between linguistic concepts and visual representations of robotic actions, \textit{2) reduces} learning complexity, \textit{3) improves} data efficiency in embodied data collection, and \textit{4) decreases} inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence.
>
---
#### [replaced 044] ORION: ORthonormal Text Encoding for Universal VLM AdaptatION
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19530](https://arxiv.org/pdf/2602.19530)**

> **作者:** Omprakash Chakraborty; Jose Dolz; Ismail Ben Ayed
>
> **摘要:** Vision language models (VLMs) have demonstrated remarkable generalization across diverse tasks, yet their performance remains constrained by the quality and geometry of the textual prototypes used to represent classes. Standard zero shot classifiers, derived from frozen text encoders and handcrafted prompts, may yield correlated or weakly separated embeddings that limit task specific discriminability. We introduce ORION, a text encoder fine tuning framework that improves pretrained VLMs using only class names. Our method optimizes, via low rank adaptation, a novel loss integrating two terms, one promoting pairwise orthogonality between the textual representations of the classes of a given task and the other penalizing deviations from the initial class prototypes. Furthermore, we provide a probabilistic interpretation of our orthogonality penalty, connecting it to the general maximum likelihood estimation (MLE) principle via Huygens theorem. We report extensive experiments on 11 benchmarks and three large VLM backbones, showing that the refined textual embeddings yield powerful replacements for the standard CLIP prototypes. Added as plug and play module on top of various state of the art methods, and across different prediction settings (zero shot, few shot and test time adaptation), ORION improves the performance consistently and significantly.
>
---
#### [replaced 045] AMFD: Distillation via Adaptive Multimodal Fusion for Multispectral Pedestrian Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.12944](https://arxiv.org/pdf/2405.12944)**

> **作者:** Zizhao Chen; Yeqiang Qian; Xiaoxiao Yang; Chunxiang Wang; Ming Yang
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** Multispectral pedestrian detection has been shown to be effective in improving performance within complex illumination scenarios. However, prevalent double-stream networks in multispectral detection employ two separate feature extraction branches for multi-modal data, leading to nearly double the inference time compared to single-stream networks utilizing only one feature extraction branch. This increased inference time has hindered the widespread employment of multispectral pedestrian detection in embedded devices for autonomous systems. To address this limitation, various knowledge distillation methods have been proposed. However, traditional distillation methods focus only on the fusion features and ignore the large amount of information in the original multi-modal features, thereby restricting the student network's performance. To tackle the challenge, we introduce the Adaptive Modal Fusion Distillation (AMFD) framework, which can fully utilize the original modal features of the teacher network. Specifically, a Modal Extraction Alignment (MEA) module is utilized to derive learning weights for student networks, integrating focal and global attention mechanisms. This methodology enables the student network to acquire optimal fusion strategies independent from that of teacher network without necessitating an additional feature fusion module. Furthermore, we present the SMOD dataset, a well-aligned challenging multispectral dataset for detection. Extensive experiments on the challenging KAIST, LLVIP and SMOD datasets are conducted to validate the effectiveness of AMFD. The results demonstrate that our method outperforms existing state-of-the-art methods in both reducing log-average Miss Rate and improving mean Average Precision. The code is available at this https URL.
>
---
#### [replaced 046] MM-OVSeg:Multimodal Optical-SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.17528](https://arxiv.org/pdf/2603.17528)**

> **作者:** Yimin Wei; Aoran Xiao; Hongruixuan Chen; Junshi Xia; Naoto Yokoya
>
> **备注:** CVPR2026
>
> **摘要:** Open-vocabulary segmentation enables pixel-level recognition from an open set of textual categories, allowing generalization beyond fixed classes. Despite great potential in remote sensing, progress in this area remains largely limited to clear-sky optical data and struggles under cloudy or haze-contaminated conditions. We present MM-OVSeg, a multimodal Optical-SAR fusion framework for resilient open-vocabulary segmentation under adverse weather conditions. MM-OVSeg leverages the complementary strengths of the two modalities--optical imagery provides rich spectral semantics, while synthetic aperture radar (SAR) offers cloud-penetrating structural cues. To address the cross-modal domain gap and the limited dense prediction capability of current vision-language models, we propose two key designs: a cross-modal unification process for multi-sensor representation alignment, and a dual-encoder fusion module that integrates hierarchical features from multiple vision foundation models for text-aligned multimodal segmentation. Extensive experiments demonstrate that MM-OVSeg achieves superior robustness and generalization across diverse cloud conditions. The source dataset and code are available at this https URL.
>
---
#### [replaced 047] PISCO: Precise Video Instance Insertion with Sparse Control
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.08277](https://arxiv.org/pdf/2602.08277)**

> **作者:** Xiangbo Gao; Renjie Li; Xinghao Chen; Yuheng Wu; Suofei Feng; Qing Yin; Zhengzhong Tu
>
> **摘要:** The landscape of AI video generation is undergoing a pivotal shift: moving beyond general generation - which relies on exhaustive prompt-engineering and "cherry-picking" - towards fine-grained, controllable generation and high-fidelity post-processing. In professional AI-assisted filmmaking, it is crucial to perform precise, targeted modifications. A cornerstone of this transition is video instance insertion, which requires inserting a specific instance into existing footage while maintaining scene integrity. Unlike traditional video editing, this task demands several requirements: precise spatial-temporal placement, physically consistent scene interaction, and the faithful preservation of original dynamics - all achieved under minimal user effort. In this paper, we propose PISCO, a video diffusion model for precise video instance insertion with arbitrary sparse keyframe control. PISCO allows users to specify a single keyframe, start-and-end keyframes, or sparse keyframes at arbitrary timestamps, and automatically propagates object appearance, motion, and interaction. To address the severe distribution shift induced by sparse conditioning in pretrained video diffusion models, we introduce Variable-Information Guidance for robust conditioning and Distribution-Preserving Temporal Masking to stabilize temporal generation, together with geometry-aware conditioning for realistic scene adaptation. We further construct PISCO-Bench, a benchmark with verified instance annotations and paired clean background videos, and evaluate performance using both reference-based and reference-free perceptual metrics. Experiments demonstrate that PISCO consistently outperforms strong inpainting and video editing baselines under sparse control, and exhibits clear, monotonic performance improvements as additional control signals are provided. Project page: this http URL.
>
---
#### [replaced 048] Clinical Metadata Guided Limited-Angle CT Image Reconstruction
- **分类: cs.CV; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2509.01752](https://arxiv.org/pdf/2509.01752)**

> **作者:** Yu Shi; Shuyi Fan; Changsheng Fang; Shuo Han; Haodong Li; Li Zhou; Bahareh Morovati; Dayang Wang; Hengyong Yu
>
> **备注:** IEEE Transactions on Medical Imaging, 2026
>
> **摘要:** Limited-angle computed tomography (LACT) offers improved temporal resolution and reduced radiation dose for cardiac imaging, but suffers from severe artifacts due to truncated projections. To address the ill-posedness of LACT reconstruction, we propose a two-stage diffusion framework guided by structured clinical metadata. In the first stage, a transformer-based diffusion model conditioned exclusively on metadata, including acquisition parameters, patient demographics, and diagnostic impressions, generates coarse anatomical priors from noise. The second stage further refines the images by integrating both the coarse prior and metadata to produce high-fidelity results. Physics-based data consistency is enforced at each sampling step in both stages using an Alternating Direction Method of Multipliers module, ensuring alignment with the measured projections. Extensive experiments on both synthetic and real cardiac CT datasets demonstrate that incorporating metadata significantly improves reconstruction fidelity, particularly under severe angular truncation. Compared to existing metadata-free baselines, our method achieves superior performance in SSIM, PSNR, nMI, and PCC. Ablation studies confirm that different types of metadata contribute complementary benefits, particularly diagnostic and demographic priors under limited-angle conditions. These findings highlight the dual role of clinical metadata in improving both reconstruction quality and efficiency, supporting their integration into future metadata-guided medical imaging frameworks.
>
---
#### [replaced 049] GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.01448](https://arxiv.org/pdf/2510.01448)**

> **作者:** Angel Daruna; Nicholas Meegan; Han-Pang Chiu; Supun Samarasekera; Rakesh Kumar
>
> **备注:** Accepted to CVPR 2026 main track
>
> **摘要:** Worldwide visual geo-localization aims to determine the geographic location of an image anywhere on Earth using only its visual content. Despite recent progress, learning expressive representations of geographic space remains challenging due to the inherently low-dimensional nature of geographic coordinates. We formulate global geo-localization as aligning the visual representation of a query image with a learned geographic representation. Our approach explicitly models the world as a hierarchy of learned geographic embeddings, enabling a distributed and multi-scale representation of geographic space. In addition, we introduce a semantic fusion module that efficiently integrates appearance features with semantic segmentation through latent cross-attention, producing a more robust visual representation for localization. Experiments on five widely used geo-localization benchmarks demonstrate that our method achieves new state-of-the-art results on 22 of 25 reported metrics. Ablation studies show that these improvements are primarily driven by the proposed geographic representation and semantic fusion mechanism.
>
---
#### [replaced 050] PriVi: Towards A General-Purpose Video Model For Primate Behavior In The Wild
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.09675](https://arxiv.org/pdf/2511.09675)**

> **作者:** Felix B. Mueller; Jan F. Meier; Timo Lueddecke; Richard Vogg; Roger L. Freixanet; Valentin Hassler; Tiffany Bosshard; Elif Karakoc; William J. O'Hearn; Sofia M. Pereira; Sandro Sehner; Kaja Wierucka; Judith Burkart; Claudia Fichtel; Julia Fischer; Alexander Gail; Catherine Hobaiter; Julia Ostner; Liran Samuni; Oliver Schülke; Neda Shahidi; Erin G. Wessling; Alexander S. Ecker
>
> **备注:** 9 pages, 5 figures, CVPR 2026
>
> **摘要:** Non-human primates are our closest living relatives, and analyzing their behavior is central to research in cognition, evolution, and conservation. Computer vision could greatly aid this research, but existing methods often rely on human-centric pretrained models and focus on single datasets, which limits generalization. We address this limitation by shifting from a model-centric to a data-centric approach and introduce PriVi, a large-scale primate-centric video pretraining dataset. PriVi contains 424 hours of curated video, combining 174 hours from behavioral research across 11 settings with 250 hours of diverse web-sourced footage, assembled through a scalable data curation pipeline. We continue pretraining V-JEPA, a large-scale video model, on PriVi to learn primate-specific representations and evaluate it using a lightweight frozen classifier. Across four benchmark datasets, ChimpACT, PanAf500, BaboonLand, and ChimpBehave, our approach consistently outperforms prior work, including fully finetuned baselines, and scales favorably with fewer labels. These results demonstrate for the first time that domain-level pretraining, where pretraining is conducted on similar data but not the target dataset itself, works for video models. Our primate-centric pretraining substantially improves data efficiency and generalization, making it a promising approach for low-label applications. Dataset, code, and models are available: this https URL
>
---
#### [replaced 051] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.22225](https://arxiv.org/pdf/2509.22225)**

> **作者:** Jiayu Ding; Xinpeng Liu; Zhiyi Pan; Shiqiang Long; Ge Li
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Lifting 2D open-vocabulary understanding into 3D Gaussian Splatting (3DGS) scenes is a critical challenge. Mainstream methods, built on an embedding paradigm, suffer from three key flaws: (i) geometry-semantic inconsistency, where points, rather than objects, serve as the semantic basis, limiting semantic fidelity; (ii) semantic bloat from injecting gigabytes of feature data into the geometry; and (iii) semantic rigidity, as one feature per Gaussian struggles to capture rich polysemy. To overcome these limitations, we introduce ExtrinSplat, a framework built on the extrinsic paradigm that decouples geometry from semantics. Instead of embedding features, ExtrinSplat clusters Gaussians into multi-granularity, overlapping 3D object groups. A Vision-Language Model (VLM) then interprets these groups to generate lightweight textual hypotheses, creating an extrinsic index layer that natively supports complex polysemy. By replacing costly feature embedding with lightweight indices, ExtrinSplat reduces scene adaptation time from hours to minutes and lowers storage overhead by several orders of magnitude. On benchmark tasks for open-vocabulary 3D object selection and semantic segmentation, ExtrinSplat outperforms established embedding-based frameworks, validating the efficacy and efficiency of the proposed extrinsic paradigm.
>
---
#### [replaced 052] GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于GUI接地任务，解决如何高效将自然语言指令映射到屏幕操作区域的问题。提出GUI-AIMA框架，通过注意力对齐实现精准定位，提升数据效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.00810](https://arxiv.org/pdf/2511.00810)**

> **作者:** Shijie Zhou; Viet Dac Lai; Hao Tan; Jihyung Kil; Wanrong Zhu; Changyou Chen; Ruiyi Zhang
>
> **摘要:** Graphical user interface (GUI) grounding is a key capability for computer-use agents, mapping natural-language instructions to actionable regions on the screen. Existing Multimodal Large Language Model (MLLM) approaches typically formulate GUI grounding as a text-based coordinate generation task. However, directly generating precise coordinates from visual inputs is challenging and often data-intensive. A more intuitive strategy is to first identify instruction-relevant visual patches and then determine the exact click location within them. Motivated by recent observations that general MLLMs exhibit native grounding ability embedded in their attention maps, we propose GUI-AIMA, an attention-based and coordinate-free supervised fine-tuning framework for efficient GUI grounding. GUI-AIMA aligns the intrinsic multimodal attention of MLLMs with patch-wise grounding signals. These signals are calculated adaptively for diverse user instructions by multi-head aggregation on simplified query-visual attention matrices. Besides, its coordinate-free manner can easily integrate a plug-and-play zoom-in stage. GUI-AIMA-3B was trained with only 509k samples (around 101k screenshots), demonstrating exceptional data efficiency and verifying that light training can trigger the native grounding capability of MLLMs. It achieves state-of-the-art performance among 3B models, attaining an average accuracy of 61.5% on ScreenSpot-Pro, 92.1% on ScreenSpot-v2, 68.1% on OSWorld-G, 79.1% on MMBench-GUI-L2, and 60.0% on UI-Vision. Project page: this https URL
>
---
#### [replaced 053] Few TensoRF: Enhance the Few-shot on Tensorial Radiance Fields
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.25008](https://arxiv.org/pdf/2603.25008)**

> **作者:** Thanh-Hai Le; Hoang-Hau Tran; Trong-Nghia Vu
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** This paper presents Few TensoRF, a 3D reconstruction framework that combines TensorRF's efficient tensor based representation with FreeNeRF's frequency driven few shot regularization. Using TensorRF to significantly accelerate rendering speed and introducing frequency and occlusion masks, the method improves stability and reconstruction quality under sparse input views. Experiments on the Synthesis NeRF benchmark show that Few TensoRF method improves the average PSNR from 21.45 dB (TensorRF) to 23.70 dB, with the fine tuned version reaching 24.52 dB, while maintaining TensorRF's fast \(\approx10-15\) minute training time. Experiments on the THuman 2.0 dataset further demonstrate competitive performance in human body reconstruction, achieving 27.37 - 34.00 dB with only eight input images. These results highlight Few TensoRF as an efficient and data effective solution for real-time 3D reconstruction across diverse scenes.
>
---
#### [replaced 054] GeoNDC: A Queryable Neural Data Cube for Planetary-Scale Earth Observation
- **分类: cs.CV; physics.geo-ph**

- **链接: [https://arxiv.org/pdf/2603.25037](https://arxiv.org/pdf/2603.25037)**

> **作者:** Jianbo Qi; Mengyao Li; Baogui Jiang; Yidan Chen; Xihan Mu; Qiao Wang
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** Satellite Earth observation has accumulated massive spatiotemporal archives essential for monitoring environmental change, yet these remain organized as discrete raster files, making them costly to store, transmit, and query. We present GeoNDC, a queryable neural data cube that encodes planetary-scale Earth observation data as a continuous spatiotemporal implicit neural field, enabling on-demand queries and continuous-time reconstruction without full decompression. Experiments on a 20-year global MODIS MCD43A4 reflectance record ($8016 \times 4008$ pixels, 7 bands, 915 temporal frames) show that the learned representation supports direct spatiotemporal queries on consumer hardware. On Sentinel-2 imagery (10 m), continuous temporal parameterization recovers cloud-free dynamics with high fidelity ($R^2 > 0.85$) under simulated 2-km cloud occlusion. On HiGLASS biophysical products (LAI and FPAR), GeoNDC attains near-perfect accuracy ($R^2 > 0.98$). The representation compresses the 20-year MODIS archive to 0.44\,GB -- approximately 95:1 relative to an optimized Int16 baseline -- with high spectral fidelity (mean $R^2 > 0.98$, mean RMSE $= 0.021$). These results suggest GeoNDC offers a unified AI-native representation for planetary-scale Earth observation, complementing raw archives with a compact, analysis-ready data layer integrating query, reconstruction, and compression in a single framework.
>
---
#### [replaced 055] Versatile Recompression-Aware Perceptual Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18090](https://arxiv.org/pdf/2511.18090)**

> **作者:** Mingwei He; Tongda Xu; Xingtong Ge; Ming Sun; Chao Zhou; Yan Wang
>
> **摘要:** Perceptual image super-resolution (SR) methods restore degraded images and produce sharp outputs. In practice, those outputs are usually recompressed for storage and transmission. Ignoring recompression is suboptimal as the downstream codec might add additional artifacts to restored images. However, jointly optimizing SR and recompression is challenging, as the codecs are not differentiable and vary in configuration. In this paper, we present \textbf{Versatile Recompression-Aware Perceptual Super-Resolution (VRPSR)}, which makes existing perceptual SR aware of versatile compression. First, we formulate compression as conditional text-to-image generation and utilize a pre-trained diffusion model to build a generalizable codec simulator. Next, we propose a set of training techniques tailored for perceptual SR, including optimizing the simulator using perceptual targets and adopting slightly compressed images as the training target. Empirically, our VRPSR achieves 10% - 40% bitrate savings based on Real-ESRGAN and S3Diff under H.264/H.265/H.266 single-picture (intra) compression. Besides, our VRPSR facilitates joint optimization of SR and the post-processing model after recompression.
>
---
#### [replaced 056] Attention Misses Visual Risk: Risk-Adaptive Steering for Multimodal Safety Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13698](https://arxiv.org/pdf/2510.13698)**

> **作者:** Jonghyun Park; Minhyuk Seo; Chaewon Yeo; Jonghyun Choi
>
> **摘要:** Even modern AI models often remain vulnerable to multimodal queries in which harmful intent is embedded in images. A widely used approach for safety alignment is training with extensive multimodal safety datasets, but the costs of data curation and training are often prohibitive. To mitigate these costs, inference-time alignment has recently been explored, but they often lack generalizability across diverse multimodal jailbreaks and still incur notable overhead due to extra forward passes for response refinement or heavy pre-deployment calibration procedures. Here, we identify insufficient visual attention to safety-critical image regions as one of the key causes of multimodal safety failures. Building on this insight, we propose Multimodal Risk-Adaptive Steering (MoRAS), which enhances safety-critical visual attention via concise visual contexts for accurate multimodal risk assessment. This risk signal enables risk-adaptive steering for direct refusals, reducing inference overhead while remaining generalizable across diverse multimodal jailbreaks. Notably, MoRAS requires only a small calibration set to estimate multimodal risk, substantially reducing pre-deployment overhead. We conduct various empirical validations across multiple benchmarks and MLLM backbones, and observe that the proposed MoRAS consistently mitigates jailbreaks, preserves utility, and reduces computational overhead compared to state-of-the-art inference-time defenses.
>
---
#### [replaced 057] PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20778](https://arxiv.org/pdf/2603.20778)**

> **作者:** Xiaoya Cheng; Long Wang; Yan Liu; Xinyi Liu; Hanlin Tan; Yu Liu; Maojun Zhang; Shen Yan
>
> **摘要:** We present PiLoT, a unified framework that tackles UAV-based ego and target geo-localization. Conventional approaches rely on decoupled pipelines that fuse GNSS and Visual-Inertial Odometry (VIO) for ego-pose estimation, and active sensors like laser rangefinders for target localization. However, these methods are susceptible to failure in GNSS-denied environments and incur substantial hardware costs and complexity. PiLoT breaks this paradigm by directly registering live video stream against a geo-referenced 3D map. To achieve robust, accurate, and real-time performance, we introduce three key contributions: 1) a Dual-Thread Engine that decouples map rendering from core localization thread, ensuring both low latency while maintaining drift-free accuracy; 2) a large-scale synthetic dataset with precise geometric annotations (camera pose, depth maps). This dataset enables the training of a lightweight network that generalizes in a zero-shot manner from simulation to real data; and 3) a Joint Neural-Guided Stochastic-Gradient Optimizer (JNGO) that achieves robust convergence even under aggressive motion. Evaluations on a comprehensive set of public and newly collected benchmarks show that PiLoT outperforms state-of-the-art methods while running over 25 FPS on NVIDIA Jetson Orin platform. Our code and dataset is available at: this https URL.
>
---
#### [replaced 058] DUET-VLM: Dual stage Unified Efficient Token reduction for VLM Training and Inference
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.18846](https://arxiv.org/pdf/2602.18846)**

> **作者:** Aditya Kumar Singh; Hitesh Kandala; Pratik Prabhanjan Brahma; Zicheng Liu; Emad Barsoum
>
> **备注:** 15 Pages, 8 figures, 15 tables, CVPR 2026; Code: this https URL
>
> **摘要:** Vision-language models (VLMs) have achieved remarkable multimodal understanding and reasoning capabilities, yet remain computationally expensive due to dense visual tokenization. Existing efficiency approaches either merge redundant visual tokens or drop them progressively in language backbone, often trading accuracy for speed. In this work, we propose DUET-VLM, a versatile plug-and-play dual compression framework that consists of (a) vision-only redundancy aware compression of vision encoder's output into information-preserving tokens, followed by (b) layer-wise, salient text-guided dropping of visual tokens within the language backbone to progressively prune less informative tokens. This coordinated token management enables aggressive compression while retaining critical semantics. On LLaVA-1.5-7B, our approach maintains over 99% of baseline accuracy with 67% fewer tokens, and still retains >97% even at 89% reduction. With this dual-stage compression during training, it achieves 99.7% accuracy at 67% and 97.6% at 89%, surpassing prior SoTA visual token reduction methods across multiple benchmarks. When integrated into Video-LLaVA-7B, it even surpasses the baseline -- achieving >100% accuracy with a substantial 53.1% token reduction and retaining 97.6% accuracy under an extreme 93.4% setting. These results highlight end-to-end training with DUET-VLM, enabling robust adaptation to reduced visual (image/video) input without sacrificing accuracy, producing compact yet semantically rich representations within the same computational budget. Our code is available at this https URL.
>
---
#### [replaced 059] Skullptor: High Fidelity 3D Head Reconstruction in Seconds with Multi-View Normal Prediction
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2602.21100](https://arxiv.org/pdf/2602.21100)**

> **作者:** Noé Artru; Rukhshanda Hussain; Emeline Got; Alexandre Messier; David B. Lindell; Abdallah Dib
>
> **备注:** For our project page, see this https URL
>
> **摘要:** Reconstructing high-fidelity 3D head geometry from images is critical for a wide range of applications, yet existing methods face fundamental limitations. Traditional photogrammetry achieves exceptional detail but requires extensive camera arrays (25-200+ views), substantial computation, and manual cleanup in challenging areas like facial hair. Recent alternatives present a fundamental trade-off: foundation models enable efficient single-image reconstruction but lack fine geometric detail, while optimization-based methods achieve higher fidelity but require dense views and expensive computation. We bridge this gap with a hybrid approach that combines the strengths of both paradigms. Our method introduces a multi-view surface normal prediction model that extends monocular foundation models with cross-view attention to produce geometrically consistent normals in a feed-forward pass. We then leverage these predictions as strong geometric priors within an inverse rendering optimization framework to recover high-frequency surface details. Our approach outperforms state-of-the-art single-image and multi-view methods, achieving high-fidelity reconstruction on par with dense-view photogrammetry while reducing camera requirements and computational cost.
>
---
#### [replaced 060] ORIC: Benchmarking Object Recognition under Contextual Incongruity in Large Vision-Language Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.15695](https://arxiv.org/pdf/2509.15695)**

> **作者:** Zhaoyang Li; Zhan Ling; Yuchen Zhou; Litian Gong; Erdem Bıyık; Hao Su
>
> **备注:** We request withdrawal of this paper because one of the listed institutional affiliations was included without proper authorization. This issue cannot be resolved through a simple revision, and we therefore request withdrawal to prevent dissemination of incorrect or unauthorized affiliation information
>
> **摘要:** Large Vision-Language Models (LVLMs) excel at captioning, visual question answering, and robotics by combining vision and language, yet they often miss obvious objects or hallucinate nonexistent ones in atypical scenes. We examine these failures through the lens of uncertainty, focusing on contextual incongruity, where objects appear unexpectedly or fail to appear in expected contexts, and show that such cases increase recognition difficulty for state-of-the- art LVLMs. To study this regime, we introduce the Object Recognition in Incongruous Context (ORIC) framework, which constructs incongruous object-context pairs through two complementary strategies: (1) LLM-guided sampling to identify hard-to-recognize objects present in the image and (2) CLIP-guided sampling to mine plausible but absent ones. Applied to MSCOCO, ORIC creates ORIC-Bench and ORIC-style training data. Evaluating 18 LVLMs and 2 open-vocabulary detectors reveals significant degradation and bias under incongruous contexts. Visual Reinforcement Fine-Tuning of Qwen3-VL-8B-Instruct on 600 ORIC samples improves performance on ORIC-Bench, AMBER, and HallusionBench. Overall, we show that contextual incongruity is a key source of uncertainty and provide tools for more reliable LVLMs. The dataset and code are publicly available at this https URL.
>
---
#### [replaced 061] UniSER: A Foundation Model for Unified Soft Effects Removal
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14183](https://arxiv.org/pdf/2511.14183)**

> **作者:** Jingdong Zhang; Lingzhi Zhang; Qing Liu; Mang Tik Chiu; Connelly Barnes; Yizhou Wang; Haoran You; Xiaoyang Liu; Yuqian Zhou; Zhe Lin; Eli Shechtman; Sohrab Amirghodsi; Xin Li; Wenping Wang; Xiaohang Zhan
>
> **摘要:** Digital images are often degraded by soft effects such as lens flare, haze, shadows, and reflections, which reduce aesthetics even though the underlying pixels remain partially visible. The prevailing works address these degradations in isolation, developing highly specialized, specialist models that lack scalability and fail to exploit the shared underlying essences of these restoration problems. Meanwhile, although recent large-scale generalist models (e.g., GPT-4o, Flux Kontext, Nano Banana) offer powerful text-driven editing capabilities, they heavily rely on detailed prompts and often fail to achieve robust removal on such fine-grained tasks while preserving the scene's identity. Leveraging the common essence of soft effects, i.e., semi-transparent occlusions, we introduce a foundational versatile model UniSER, capable of addressing diverse degradations caused by soft effects within a single framework. Our methodology centers on curating a massive 3.8M-pair dataset to ensure robustness and generalization, which includes novel, physically-plausible data to fill critical gaps in public benchmarks, and a tailored training pipeline that fine-tunes a Diffusion Transformer to learn robust restoration priors from this diverse data, integrating fine-grained mask and strength controls. This synergistic approach allows UniSER to significantly outperform both specialist and generalist models, achieving robust, high-fidelity restoration in the wild.
>
---
#### [replaced 062] The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.14375](https://arxiv.org/pdf/2603.14375)**

> **作者:** Xiangbo Gao; Mingyang Wu; Siyuan Yang; Jiongze Yu; Pardis Taghavi; Fangzhou Lin; Zhengzhong Tu
>
> **摘要:** While recent generative video models have achieved remarkable visual realism and are being explored as world models, true physical simulation requires mastering both space and time. Current models can produce visually smooth kinematics, yet they lack a reliable internal motion pulse to ground these motions in a consistent, real-world time scale. This temporal ambiguity stems from the common practice of indiscriminately training on videos with vastly different real-world speeds, forcing them into standardized frame rates. This leads to what we term chronometric hallucination: generated sequences exhibit ambiguous, unstable, and uncontrollable physical motion speeds. To address this, we propose Visual Chronometer, a predictor that recovers the Physical Frames Per Second (PhyFPS) directly from the visual dynamics of an input video. Trained via controlled temporal resampling, our method estimates the true temporal scale implied by the motion itself, bypassing unreliable metadata. To systematically quantify this issue, we establish two benchmarks, PhyFPS-Bench-Real and PhyFPS-Bench-Gen. Our evaluations reveal a harsh reality: state-of-the-art video generators suffer from severe PhyFPS misalignment and temporal instability. Finally, we demonstrate that applying PhyFPS corrections significantly improves the human-perceived naturalness of AI-generated videos. Our project page is this https URL.
>
---
#### [replaced 063] Wid3R: Wide Field-of-View 3D Reconstruction via Camera Model Conditioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05321](https://arxiv.org/pdf/2602.05321)**

> **作者:** Dongki Jung; Jaehoon Choi; Adil Qureshi; Somi Jeong; Dinesh Manocha; Suyong Yeon
>
> **摘要:** We present Wid3R, a feed-forward neural network for multi-view visual geometry reconstruction that supports wide field-of-view camera models. Unlike existing methods that assume rectified or pinhole inputs, Wid3R directly models wide-angle imagery without explicit calibration or undistortion. Our approach leverages a ray-based representation with spherical harmonics and introduces a novel camera model token to enable distortion-aware reconstruction. To the best of our knowledge, Wid3R is the first multi-frame feed-forward 3D reconstruction method that supports 360 imagery. Moreover, we show that conditioning on diverse camera types improves generalization to 360 scenes and alleviates data sparsity issues. Wid3R achieves significant performance gains, improving AUC@30 by up to +33.67 on Zip-NeRF (fisheye) and +77.33 on Stanford2D3D (360).
>
---
#### [replaced 064] CLEAR: Causal Learning Framework For Robust Histopathology Tumor Detection Under Out-Of-Distribution Shifts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.14273](https://arxiv.org/pdf/2510.14273)**

> **作者:** Kieu-Anh Truong Thi; Huy-Hieu Pham; Duc-Trong Le
>
> **摘要:** Domain shift in histopathology, often caused by differences in acquisition processes or data sources, poses a major challenge to the generalization ability of deep learning models. Existing methods primarily rely on modeling statistical correlations by aligning feature distributions or introducing statistical variation, yet they often overlook causal relationships. In this work, we propose a novel causal-inference-based framework that leverages semantic features while mitigating the impact of confounders. Our method implements the front-door principle by designing transformation strategies that explicitly incorporate mediators and observed tissue slides. We validate our method on the CAMELYON17 dataset and a private histopathology dataset, demonstrating consistent performance gains across unseen domains. As a result, our approach achieved up to a 7% improvement in both the CAMELYON17 dataset and the private histopathology dataset, outperforming existing baselines. These results highlight the potential of causal inference as a powerful tool for addressing domain shift in histopathology image analysis.
>
---
#### [replaced 065] CrisiSense-RAG: Crisis Sensing Multimodal Retrieval-Augmented Generation for Rapid Disaster Impact Assessment
- **分类: cs.CY; cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2602.13239](https://arxiv.org/pdf/2602.13239)**

> **作者:** Yiming Xiao; Kai Yin; Ali Mostafavi
>
> **备注:** 27 pages, 4 figures
>
> **摘要:** Timely and spatially resolved disaster impact assessment is essential for effective emergency response. However, automated methods typically struggle with temporal asynchrony. Real-time human reports capture peak hazard conditions while high-resolution satellite imagery is frequently acquired after peak conditions. This often reflects flood recession rather than maximum extent. Naive fusion of these misaligned streams can yield dangerous underestimates when post-event imagery overrides documented peak flooding. We present CrisiSense-RAG, which is a multimodal retrieval-augmented generation framework that reframes impact assessment as evidence synthesis over heterogeneous data sources without disaster-specific fine-tuning. The system employs hybrid dense-sparse retrieval for text sources and CLIP-based retrieval for aerial imagery. A split-pipeline architecture feeds into asynchronous fusion logic that prioritizes real-time social evidence for peak flood extent while treating imagery as persistent evidence of structural damage. Evaluated on Hurricane Harvey across 207 ZIP-code queries, the framework achieves a flood extent MAE of 10.94% to 28.40% and damage severity MAE of 16.47% to 21.65% in zero-shot settings. Prompt-level alignment proves critical for quantitative validity because metric grounding improves damage estimates by up to 4.75 percentage points. These results demonstrate a practical and deployable approach to rapid resilience intelligence under real-world data constraints.
>
---
#### [replaced 066] Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12766](https://arxiv.org/pdf/2506.12766)**

> **作者:** Ruojing Li; Wei An; Yingqian Wang; Xinyi Ying; Yimian Dai; Longguang Wang; Miao Li; Yulan Guo; Li Liu
>
> **摘要:** Infrared small target (IRST) detection is challenging in simultaneously achieving precise, robust, and efficient performance due to extremely dim targets and strong interference. Current learning-based methods attempt to leverage ``more" information from both the spatial and the short-term temporal domains, but suffer from unreliable performance under complex conditions while incurring computational redundancy. In this paper, we explore the ``more essential" information from a more crucial domain for the detection. Through theoretical analysis, we reveal that the global temporal saliency and correlation information in the temporal profile demonstrate significant superiority in distinguishing target signals from other signals. To investigate whether such superiority is preferentially leveraged by well-trained networks, we built the first prediction attribution tool in this field and verified the importance of the temporal profile information. Inspired by the above conclusions, we remodel the IRST detection task as a one-dimensional signal anomaly detection task, and propose an efficient deep temporal probe network (DeepPro) that only performs calculations in the time dimension for IRST detection. We conducted extensive experiments to fully validate the effectiveness of our method. The experimental results are exciting, as our DeepPro outperforms existing state-of-the-art IRST detection methods on widely-used benchmarks with extremely high efficiency, and achieves a significant improvement on dim targets and in complex scenarios. We provide a new modeling domain, a new insight, a new method, and a new performance, which can promote the development of IRST detection. Codes are available at this https URL.
>
---
#### [replaced 067] Evidence-based diagnostic reasoning with multi-agent copilot for human pathology
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.20964](https://arxiv.org/pdf/2506.20964)**

> **作者:** Luca L. Weishaupt; Chengkuan Chen; Drew F. K. Williamson; Richard J. Chen; Guillaume Jaume; Tong Ding; Bowen Chen; Anurag Vaidya; Long Phi Le; Guillaume Jaume; Ming Y. Lu; Faisal Mahmood
>
> **摘要:** Pathology is experiencing rapid digital transformation driven by whole-slide imaging and artificial intelligence (AI). While deep learning-based computational pathology has achieved notable success, traditional models primarily focus on image analysis without integrating natural language instruction or rich, text-based context. Current multimodal large language models (MLLMs) in computational pathology face limitations, including insufficient training data, inadequate support and evaluation for multi-image understanding, and a lack of autonomous, diagnostic reasoning capabilities. To address these limitations, we introduce PathChat+, a new MLLM specifically designed for human pathology, trained on over 1 million diverse, pathology-specific instruction samples and nearly 5.5 million question answer turns. Extensive evaluations across diverse pathology benchmarks demonstrated that PathChat+ substantially outperforms the prior PathChat copilot, as well as both state-of-the-art (SOTA) general-purpose and other pathology-specific models. Furthermore, we present SlideSeek, a reasoning-enabled multi-agent AI system leveraging PathChat+ to autonomously evaluate gigapixel whole-slide images (WSIs) through iterative, hierarchical diagnostic reasoning, reaching high accuracy on DDxBench, a challenging open-ended differential diagnosis benchmark, while also capable of generating visually grounded, humanly-interpretable summary reports.
>
---
#### [replaced 068] Interact2Ar: Full-Body Human-Human Interaction Generation via Autoregressive Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19692](https://arxiv.org/pdf/2512.19692)**

> **作者:** Pablo Ruiz-Ponce; Sergio Escalera; José García-Rodríguez; Jiankang Deng; Rolandos Alexandros Potamias
>
> **备注:** Project Page: this https URL
>
> **摘要:** Generating realistic human-human interactions is a challenging task that requires not only high-quality individual body and hand motions, but also coherent coordination among all interactants. Due to limitations in available data and increased learning complexity, previous methods tend to ignore hand motions, limiting the realism and expressivity of the interactions. Additionally, current diffusion-based approaches generate entire motion sequences simultaneously, limiting their ability to capture the reactive and adaptive nature of human interactions. To address these limitations, we introduce Interact2Ar, the first end-to-end text-conditioned autoregressive diffusion model for generating full-body, human-human interactions. Interact2Ar incorporates detailed hand kinematics through dedicated parallel branches, enabling high-fidelity full-body generation. Furthermore, we introduce an autoregressive pipeline coupled with a novel memory technique that facilitates adaptation to the inherent variability of human interactions using efficient large context windows. The adaptability of our model enables a series of downstream applications, including temporal motion composition, real-time adaptation to disturbances, and extension beyond dyadic to multi-person scenarios. To validate the generated motions, we introduce a set of robust evaluators and extended metrics designed specifically for assessing full-body interactions. Through quantitative and qualitative experiments, we demonstrate the state-of-the-art performance of Interact2Ar.
>
---
#### [replaced 069] EOGS++: Earth Observation Gaussian Splatting with Internal Camera Refinement and Direct Panchromatic Rendering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16542](https://arxiv.org/pdf/2511.16542)**

> **作者:** Pierrick Bournez; Luca Savant Aira; Thibaud Ehret; Gabriele Facciolo
>
> **备注:** 8 pages, ISPRS
>
> **摘要:** Recently, 3D Gaussian Splatting has been introduced as a compelling alternative to NeRF for Earth observation, offering competitive reconstruction quality with significantly reduced training times. In this work, we extend the Earth Observation Gaussian Splatting (EOGS) framework to propose EOGS++, a novel method tailored for satellite imagery that directly operates on raw high-resolution panchromatic data without requiring external preprocessing. Furthermore, leveraging optical flow techniques we embed bundle adjustment directly within the training process, avoiding reliance on external optimization tools while improving camera pose estimation. We also introduce several improvements to the original implementation, including early stopping and TSDF post-processing, all contributing to sharper reconstructions and better geometric accuracy. Experiments on the IARPA 2016 and DFC2019 datasets demonstrate that EOGS++ achieves state-of-the-art performance in terms of reconstruction quality and efficiency, outperforming the original EOGS method and other NeRF-based methods while maintaining the computational advantages of Gaussian Splatting. Our model demonstrates an improvement from 1.33 to 1.19 mean MAE errors on buildings compared to the original EOGS models
>
---
#### [replaced 070] WildDepth: A Multimodal Dataset for 3D Wildlife Perception and Depth Estimation
- **分类: cs.CV; cs.DL**

- **链接: [https://arxiv.org/pdf/2603.16816](https://arxiv.org/pdf/2603.16816)**

> **作者:** Muhammad Aamir; Naoya Muramatsu; Sangyun Shin; Matthew Wijers; Jia-Xing Zhong; Xinyu Hou; Amir Patel; Andrew Loveridge; Andrew Markham
>
> **摘要:** Depth estimation and 3D reconstruction have been extensively studied as core topics in computer vision. Starting from rigid objects with relatively simple geometric shapes, such as vehicles, the research has expanded to address general objects, including challenging deformable objects, such as humans and animals. However, for the animal, in particular, the majority of existing models are trained based on datasets without metric scale, which can help validate image-only models. To address this limitation, we present WildDepth, a multimodal dataset and benchmark suite for depth estimation, behavior detection, and 3D reconstruction from diverse categories of animals ranging from domestic to wild environments with synchronized RGB and LiDAR. Experimental results show that the use of multi-modal data improves depth reliability by up to 10% RMSE, while RGB-LiDAR fusion enhances 3D reconstruction fidelity by 12% in Chamfer distance. By releasing WildDepth and its benchmarks, we aim to foster robust multimodal perception systems that generalize across domains.
>
---
#### [replaced 071] Relaxed Rigidity with Ray-based Grouping for Dynamic Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24994](https://arxiv.org/pdf/2603.24994)**

> **作者:** Junoh Lee; Junmyeong Lee; Yeon-Ji Song; Inhwan Bae; Jisu Shin; Hae-Gon Jeon; Jin-Hwa Kim
>
> **备注:** 24 pages, 7 figures
>
> **摘要:** The reconstruction of dynamic 3D scenes using 3D Gaussian Splatting has shown significant promise. A key challenge, however, remains in modeling realistic motion, as most methods fail to align the motion of Gaussians with real-world physical dynamics. This misalignment is particularly problematic for monocular video datasets, where failing to maintain coherent motion undermines local geometric structure, ultimately leading to degraded reconstruction quality. Consequently, many state-of-the-art approaches rely heavily on external priors, such as optical flow or 2D tracks, to enforce temporal coherence. In this work, we propose a novel method to explicitly preserve the local geometric structure of Gaussians across time in 4D scenes. Our core idea is to introduce a view-space ray grouping strategy that clusters Gaussians intersected by the same ray, considering only those whose $\alpha$-blending weights exceed a threshold. We then apply constraints to these groups to maintain a consistent spatial distribution, effectively preserving their local geometry. This approach enforces a more physically plausible motion model by ensuring that local geometry remains stable over time, eliminating the reliance on external guidance. We demonstrate the efficacy of our method by integrating it into two distinct baseline models. Extensive experiments on challenging monocular datasets show that our approach significantly outperforms existing methods, achieving superior temporal consistency and reconstruction quality.
>
---
#### [replaced 072] Editable-DeepSC: Reliable Cross-Modal Semantic Communications for Facial Editing
- **分类: cs.IT; cs.CV; cs.NI**

- **链接: [https://arxiv.org/pdf/2411.15702](https://arxiv.org/pdf/2411.15702)**

> **作者:** Bin Chen; Wenbo Yu; Qinshan Zhang; Tianqu Zhuang; Hao Wu; Yong Jiang; Shu-Tao Xia
>
> **摘要:** Interactive computer vision (CV) plays a crucial role in various real-world applications, whose performance is highly dependent on communication networks. Nonetheless, the data-oriented characteristics of conventional communications often do not align with the special needs of interactive CV tasks. To alleviate this issue, the recently emerged semantic communications only transmit task-related semantic information and exhibit a promising landscape to address this problem. However, the communication challenges associated with Semantic Facial Editing, one of the most important interactive CV applications on social media, still remain largely unexplored. In this paper, we fill this gap by proposing Editable-DeepSC, a novel cross-modal semantic communication approach for facial editing. Firstly, we theoretically discuss different transmission schemes that separately handle communications and editings, and emphasize the necessity of Joint Editing-Channel Coding (JECC) via iterative attributes matching, which integrates editings into the communication chain to preserve more semantic mutual information. To compactly represent the high-dimensional data, we leverage inversion methods via pre-trained StyleGAN priors for semantic coding. To tackle the dynamic channel noise conditions, we propose SNR-aware channel coding via model fine-tuning. Extensive experiments indicate that Editable-DeepSC can achieve superior editings while significantly saving the transmission bandwidth, even under high-resolution and out-of-distribution (OOD) settings.
>
---
#### [replaced 073] The Effective Depth Paradox: Evaluating the Relationship between Architectural Topology and Trainability in Deep CNNs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.13298](https://arxiv.org/pdf/2602.13298)**

> **作者:** Manfred M. Fischer; Joshua Pitts
>
> **摘要:** This paper investigates the relationship between convolutional neural network (CNN) and image recognition performance through a comparative study of the VGG, ResNet and GoogLeNet architectural families. By evaluating these models under a unified experimental framework on upscaled CIFAR-10 data, we isolate the effects of depth from confounding implementation variables. We introduce a formal distinction between nominal depth ($D_{\mathrm{nom}}$), the total count of weight-bearing layers, and effective depth ($D_{\mathrm{eff}}$), an operational metric representing the expected number of sequential transformations encountered along all feasible forward paths. As derived in Section 3, $D_{\mathrm{eff}}$ is computed through topology-specific proxies: as the total sequential count for plain networks, the arithmetic mean of minimum and maximum path lengths for residual structures, and the sum of average branch depths for multi-branch modules. Our empirical results demonstrate that while sequential architectures such as VGG suffer from diminishing returns and severe gradient attenuation as $D_{\mathrm{nom}}$ increases, architectures with identity shortcuts or branching modules maintain optimization stability. This stability is achieved by decoupling $D_{\mathrm{eff}}$ from $D_{\mathrm{nom}}$, thus ensuring a manageable functional depth for gradient propagation. We conclude that effective depth serves as a superior predictor of a network's scaling potential and practical trainability compared to traditional layer counts, providing a principled framework for future architectural innovation.
>
---
#### [replaced 074] Towards single-shot coherent imaging via overlap-free ptychography
- **分类: physics.optics; cs.AI; cs.CV; cs.LG; physics.comp-ph**

- **链接: [https://arxiv.org/pdf/2602.21361](https://arxiv.org/pdf/2602.21361)**

> **作者:** Oliver Hoidn; Aashwin Mishra; Steven Henke; Albert Vong; Matthew Seaberg
>
> **摘要:** Ptychographic imaging at synchrotron and XFEL sources requires dense overlapping scans, limiting throughput and increasing dose. Extending coherent diffractive imaging to overlap-free operation on extended samples remains an open problem. Here, we extend PtychoPINN (O. Hoidn \emph{et al.}, \emph{Scientific Reports} \textbf{13}, 22789, 2023) to deliver \emph{overlap-free, single-shot} reconstructions in a Fresnel coherent diffraction imaging (CDI) geometry while also accelerating conventional multi-shot ptychography. The framework couples a differentiable forward model of coherent scattering with a Poisson photon-counting likelihood; real-space overlap enters as a tunable parameter via coordinate-based grouping rather than a hard requirement. On synthetic benchmarks, reconstructions remain accurate at low counts ($\sim\!10^4$ photons/frame), and overlap-free single-shot reconstruction with an experimental probe reaches amplitude structural similarity (SSIM) 0.904, compared with 0.968 for overlap-constrained reconstruction. Against a data-saturated supervised model with the same backbone (16,384 training images), PtychoPINN achieves higher SSIM with only 1,024 images and generalizes to unseen illumination profiles. Per-graphics processing unit (GPU) throughput is approximately $40\times$ that of least-squares maximum-likelihood (LSQ-ML) reconstruction at matched $128\times128$ resolution. These results, validated on experimental data from the Advanced Photon Source and the Linac Coherent Light Source, unify single-exposure Fresnel CDI and overlapped ptychography within one framework, supporting dose-efficient, high-throughput imaging at modern light sources.
>
---
#### [replaced 075] Beyond Deepfake vs Real: Facial Deepfake Detection in the Open-Set Paradigm
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08055](https://arxiv.org/pdf/2503.08055)**

> **作者:** Nadarasar Bahavan; Sachith Seneviratne; Sanjay Saha; Ken Chen; Sanka Rasnayaka; Saman Halgamuge
>
> **备注:** IEEE/CVF Conference on Computer Vision and Pattern Recognition - Workshop
>
> **摘要:** Facial forgery methods such as deepfakes can be misused for identity manipulation and spreading misinformation. They have evolved alongside advancements in generative AI, leading to new and more sophisticated forgery techniques that diverge from existing ``known" methods. Conventional deepfake detection methods use the closed-set paradigm, thus limiting their applicability to detecting forgeries created using methods that are not part of the training dataset. In this paper, we propose a shift from the closed-set paradigm for deepfake detection. In the open-set paradigm, models are designed not only to identify images created by known facial forgery methods but also to identify and flag those produced by previously unknown methods as `unknown' and not as unforged or real or nmanipulated. In this paper, we propose an open-set deepfake classification algorithm based on supervised contrastive learning. The open-set paradigm used in our model allows it to function as a more robust tool capable of handling emerging and unseen deepfake techniques, enhancing reliability and confidence, and complementing forensic analysis. In the open-set paradigm, we identify three groups, including the `unknown' group that is neither considered a known deepfake nor real. We investigate deepfake open-set classification across three scenarios: classifying deepfakes from unknown methods not as real, distinguishing real images from deepfakes, and classifying deepfakes from known methods, using the FaceForensics++ dataset as a benchmark. Our method achieves state-of-the-art results in the first two tasks and competitive results in the third task.
>
---
#### [replaced 076] CLIP-RD: Relational Distillation for Efficient CLIP Knowledge Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25383](https://arxiv.org/pdf/2603.25383)**

> **作者:** Jeannie Chung; Hanna Jang; Ingyeong Yang; Uiwon Hwang; Jaehyeong Sim
>
> **摘要:** CLIP aligns image and text embeddings via contrastive learning and demonstrates strong zero-shot generalization. Its large-scale architecture requires substantial computational and memory resources, motivating the distillation of its capabilities into lightweight student models. However, existing CLIP distillation methods do not explicitly model multi-directional relational dependencies between teacher and student embeddings, limiting the student's ability to preserve the structural relationships encoded by the teacher. To address this, we propose a relational knowledge distillation framework that introduces two novel methods, Vertical Relational Distillation (VRD) and Cross Relational Distillation (XRD). VRD enforces consistency of teacher-student distillation strength across modalities at the distribution level, while XRD imposes bidirectional symmetry on cross-modal teacher-student similarity distributions. By jointly modeling multi-directional relational structures, CLIP-RD promotes faithful alignment of the student embedding geometry with that of the teacher, outperforming existing methods by 0.8%p.
>
---
#### [replaced 077] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10971](https://arxiv.org/pdf/2511.10971)**

> **作者:** Anzhe Cheng; Shukai Duan; Shixuan Li; Chenzhong Yin; Mingxi Cheng; Heng Ping; Tamoghna Chattopadhyay; Sophia I Thomopoulos; Shahin Nazarian; Paul Thompson; Paul Bogdan
>
> **备注:** Accepted in CVPR2026 Main Track
>
> **摘要:** Mixture-of-Experts (MoE) architectures expand model capacity by sparsely activating experts but face two core challenges: misalignment between router logits and each expert's internal structure leads to unstable routing and expert underutilization, and load imbalances create straggler bottlenecks. Standard solutions, such as auxiliary load-balancing losses, can reduce load disparities but often weaken expert specialization and hurt downstream performance. To address these issues, we propose ERMoE, a sparse MoE transformer that reparameterizes each expert in a learned orthonormal eigenbasis and replaces learned gating logits with an "Eigenbasis Score", defined as the cosine similarity between input features and an expert's basis. This content-aware routing ties token assignments directly to experts' representation spaces, stabilizing utilization and promoting interpretable specialization without sacrificing sparsity. Crucially, ERMoE removes the need for explicit balancing losses and avoids the interfering gradients they introduce. We show that ERMoE achieves state-of-the-art accuracy on ImageNet classification and cross-modal image-text retrieval benchmarks (e.g., COCO, Flickr30K), while naturally producing flatter expert load distributions. Moreover, a 3D MRI variant (ERMoE-ba) improves brain age prediction accuracy by more than 7\% and yields anatomically interpretable expert specializations. ERMoE thus introduces a new architectural principle for sparse expert models that directly addresses routing instabilities and enables improved performance with scalable, interpretable specialization.
>
---
#### [replaced 078] Local Precise Refinement: A Dual-Gated Mixture-of-Experts for Enhancing Foundation Model Generalization against Spectral Shifts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.13352](https://arxiv.org/pdf/2603.13352)**

> **作者:** Xi Chen; Maojun Zhang; Yu Liu; Shen Yan
>
> **摘要:** Domain Generalization Semantic Segmentation (DGSS) in spectral remote sensing is severely challenged by spectral shifts across diverse acquisition conditions, which cause significant performance degradation for models deployed in unseen domains. While fine-tuning foundation models is a promising direction, existing methods employ global, homogeneous adjustments. This "one-size-fits-all" tuning struggles with the spatial heterogeneity of land cover, causing semantic confusion. We argue that the key to robust DGSS lies not in a single global adaptation, but in performing fine-grained, spatially-adaptive refinement of a foundation model's features. To achieve this, we propose SpectralMoE, a novel fine-tuning framework for DGSS. It operationalizes this principle by utilizing a Mixture-of-Experts (MoE) architecture to perform \textbf{local precise refinement} on the foundation model's features, incorporating depth features estimated from selected RGB bands of the spectral remote sensing imagery to guide the fine-tuning process. Specifically, SpectralMoE employs a dual-gated MoE architecture that independently routes visual and depth features to top-k selected experts for specialized refinement, enabling modality-specific adjustments. A subsequent cross-attention mechanism then judiciously fuses the refined structural cues into the visual stream, mitigating semantic ambiguities caused by spectral variations. Extensive experiments show that SpectralMoE sets a new state-of-the-art on multiple DGSS benchmarks across hyperspectral, multispectral, and RGB remote sensing imagery.
>
---
#### [replaced 079] Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多智能体驾驶仿真任务，旨在提升行为模型的效率与鲁棒性。通过优化场景表示和交互建模，实现更高效的训练与推理。**

- **链接: [https://arxiv.org/pdf/2512.05812](https://arxiv.org/pdf/2512.05812)**

> **作者:** Fabian Konstantinidis; Moritz Sackmann; Ulrich Hofmann; Christoph Stiller
>
> **备注:** This is the author's accepted version of a paper to appear in the IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.
>
---
#### [replaced 080] TimeSenCLIP: A Time Series Vision-Language Model for Remote Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11919](https://arxiv.org/pdf/2508.11919)**

> **作者:** Pallavi Jain; Diego Marcos; Dino Ienco; Roberto Interdonato; Tristan Berchoux
>
> **备注:** Accepted (ISPRS Journal of Photogrammetry and Remote Sensing)
>
> **摘要:** Vision-language models (VLMs) have shown significant promise in remote sensing applications, particularly for land-use and land-cover (LULC) mapping via zero-shot classification and retrieval. However, current approaches face several key challenges, such as the dependence on caption-based supervision, which is often not available or very limited in terms of the covered semantics, and the fact of being adapted from generic VLM architectures that are suitable for very high resolution images. Consequently, these models tend to prioritize spatial context over spectral and temporal information, limiting their effectiveness for medium-resolution remote sensing imagery. In this work, we present TimeSenCLIP, a lightweight VLM for remote sensing time series, using a cross-view temporal contrastive framework to align multispectral Sentinel-2 time series with geo-tagged ground-level imagery, without requiring textual annotations. Unlike prior VLMs, TimeSenCLIP emphasizes temporal and spectral signals over spatial context, investigating whether single-pixel time series contain sufficient information for solving a variety of tasks.
>
---
#### [replaced 081] Masked Training for Robust Arrhythmia Detection from Digitalized Multiple Layout ECG Images
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09165](https://arxiv.org/pdf/2508.09165)**

> **作者:** Shanwei Zhang; Deyun Zhang; Yirao Tao; Kexin Wang; Shijia Geng; Jun Li; Qinghao Zhao; Xingpeng Liu; Xingliang Wu; Shengyong Chen; Yuxi Zhou; Shenda Hong
>
> **备注:** 28 pages, 9 figures
>
> **摘要:** Background: Electrocardiograms are indispensable for diagnosing cardiovascular diseases, yet in many settings they exist only as paper printouts stored in multiple recording layouts. Converting these images into digital signals introduces two key challenges: temporal asynchrony among leads and partial blackout missing, where contiguous signal segments become entirely unavailable. Existing models cannot adequately handle these concurrent problems while maintaining interpretability. Methods: We propose PatchECG, combining an adaptive variable block count missing learning mechanism with a masked training strategy. The model segments each lead into fixed-length patches, discards entirely missing patches, and encodes the remainder via a pluggable patch encoder. A disordered patch attention mechanism with patch-level temporal and lead embeddings captures cross-lead and temporal dependencies without interpolation. PatchECG was trained on PTB-XL and evaluated under seven simulated layout conditions, with external validation on 400 real ECG images from Chaoyang Hospital across three clinical layouts. Results: PatchECG achieves an average AUROC of approximately 0.835 across all simulated layouts. On the Chaoyang cohort, the model attains an overall AUROC of 0.778 for atrial fibrillation detection, rising to 0.893 on the 12x1 subset -- surpassing the pre-trained baseline by 0.111 and 0.190, respectively. Model attention aligns with cardiologist annotations at a rate approaching inter-clinician agreement. Conclusions: PatchECG provides a robust, interpolation-free, and interpretable solution for arrhythmia detection from digitized ECG images across diverse layouts. Its direct modeling of asynchronous and partially missing signals, combined with clinically aligned attention, positions it as a practical tool for cardiac diagnostics from legacy ECG archives in real-world clinical environments.
>
---
#### [replaced 082] Out-of-Sight Embodied Agents: Multimodal Tracking, Sensor Fusion, and Trajectory Forecasting
- **分类: cs.CV; cs.LG; cs.MA; cs.MM; cs.RO**

- **简介: 该论文属于轨迹预测任务，解决出视线目标的轨迹预测与去噪问题。通过视觉-定位对齐模块，提升自动驾驶等场景下的感知鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.15219](https://arxiv.org/pdf/2509.15219)**

> **作者:** Haichao Zhang; Yi Xu; Yun Fu
>
> **备注:** Published in IEEE Transactions on Pattern Analysis and Machine Intelligence (Early Access), pp. 1-14, March 23, 2026
>
> **摘要:** Trajectory prediction is a fundamental problem in computer vision, vision-language-action models, world models, and autonomous systems, with broad impact on autonomous driving, robotics, and surveillance. However, most existing methods assume complete and clean observations, and therefore do not adequately handle out-of-sight agents or noisy sensing signals caused by limited camera coverage, occlusions, and the absence of ground-truth denoised trajectories. These challenges raise safety concerns and reduce robustness in real-world deployment. In this extended study, we introduce major improvements to Out-of-Sight Trajectory (OST), a task for predicting noise-free visual trajectories of out-of-sight objects from noisy sensor observations. Building on our prior work, we expand Out-of-Sight Trajectory Prediction (OOSTraj) from pedestrians to both pedestrians and vehicles, increasing its relevance to autonomous driving, robotics, and surveillance. Our improved Vision-Positioning Denoising Module exploits camera calibration to establish vision-position correspondence, mitigating the lack of direct visual cues and enabling effective unsupervised denoising of noisy sensor signals. Extensive experiments on the Vi-Fi and JRDB datasets show that our method achieves state-of-the-art results for both trajectory denoising and trajectory prediction, with clear gains over prior baselines. We also compare with classical denoising methods, including Kalman filtering, and adapt recent trajectory prediction models to this setting, establishing a stronger benchmark. To the best of our knowledge, this is the first work to use vision-positioning projection to denoise noisy sensor trajectories of out-of-sight agents, opening new directions for future research.
>
---
#### [replaced 083] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于长视频推理任务，旨在解决长视频理解中上下文有限和视觉细节丢失的问题。提出WorldMM模型，结合多模态记忆实现高效信息检索与推理。**

- **链接: [https://arxiv.org/pdf/2512.02425](https://arxiv.org/pdf/2512.02425)**

> **作者:** Woongyeong Yeo; Kangsan Kim; Jaehong Yoon; Sung Ju Hwang
>
> **备注:** CVPR 2026. Project page : this https URL
>
> **摘要:** Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average 8.4% performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning.
>
---
#### [replaced 084] A.I.R.: Enabling Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04428](https://arxiv.org/pdf/2510.04428)**

> **作者:** Yuanhao Zou; Shengji Jin; Andong Deng; Youpeng Zhao; Jun Wang; Chen Chen
>
> **备注:** ICLR 2026 Paper
>
> **摘要:** Effectively applying Vision-Language Models (VLMs) to Video Question Answering (VideoQA) hinges on selecting a concise yet comprehensive set of frames, as processing entire videos is computationally infeasible. However, current frame selection methods face a critical trade-off: approaches relying on lightweight similarity models, such as CLIP, often fail to capture the nuances of complex queries, resulting in inaccurate similarity scores that cannot reflect the authentic query-frame relevance, which further undermines frame selection. Meanwhile, methods that leverage a VLM for deeper analysis achieve higher accuracy but incur prohibitive computational costs. To address these limitations, we propose A.I.R., a training-free approach for Adaptive, Iterative, and Reasoning-based frame selection. We leverage a powerful VLM to perform deep, semantic analysis on complex queries, and this analysis is deployed within a cost-effective iterative loop that processes only a small batch of the most high-potential frames at a time. Extensive experiments on various VideoQA benchmarks demonstrate that our approach outperforms existing frame selection methods, significantly boosts the performance of the foundation VLM, and achieves substantial gains in computational efficiency over other VLM-based techniques.
>
---
#### [replaced 085] Particulate: Feed-Forward 3D Object Articulation
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [https://arxiv.org/pdf/2512.11798](https://arxiv.org/pdf/2512.11798)**

> **作者:** Ruining Li; Yuxin Yao; Chuanxia Zheng; Christian Rupprecht; Joan Lasenby; Shangzhe Wu; Andrea Vedaldi
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** We introduce Particulate, a feed-forward model that, given a 3D mesh of an object, infers its articulations, including its 3D parts, their kinematic structure, and the motion constraints. The model is based on a transformer network, the Part Articulation Transformer, which predicts all these parameters for all joints. We train the network end-to-end on a diverse collection of articulated 3D assets from public datasets. During inference, Particulate maps the output of the network back to the input mesh, yielding a fully articulated 3D model in seconds, much faster than prior approaches that require per-object optimization. Particulate also works on AI-generated 3D assets, enabling the generation of articulated 3D objects from a single (real or synthetic) image when combined with an off-the-shelf image-to-3D model. We further introduce a new challenging benchmark for 3D articulation estimation curated from high-quality public 3D assets, and redesign the evaluation protocol to be more consistent with human preferences. Empirically, Particulate significantly outperforms state-of-the-art approaches.
>
---
#### [replaced 086] Learning Neural Parametric 3D Breast Shape Models for Metrical Surface Reconstruction From Monocular RGB Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13540](https://arxiv.org/pdf/2510.13540)**

> **作者:** Maximilian Weiherer; Antonia von Riedheim; Vanessa Brébant; Bernhard Egger; Christoph Palm
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL
>
> **摘要:** We present a neural parametric 3D breast shape model and, based on this model, introduce a low-cost and accessible 3D surface reconstruction pipeline capable of recovering accurate breast geometry from a monocular RGB video. In contrast to widely used, commercially available yet prohibitively expensive 3D breast scanning solutions and existing low-cost alternatives, our method requires neither specialized hardware nor proprietary software and can be used with any device that is able to record RGB videos. The key building blocks of our pipeline are a state-of-the-art, off-the-shelf Structure-from-motion pipeline, paired with a parametric breast model for robust and metrically correct surface reconstruction. Our model, similarly to the recently proposed implicit Regensburg Breast Shape Model (iRBSM), leverages implicit neural representations to model breast shapes. However, unlike the iRBSM, which employs a single global neural signed distance function (SDF), our approach -- inspired by recent state-of-the-art face models -- decomposes the implicit breast domain into multiple smaller regions, each represented by a local neural SDF anchored at anatomical landmark positions. When incorporated into our surface reconstruction pipeline, the proposed model, dubbed liRBSM (short for localized iRBSM), significantly outperforms the iRBSM in terms of reconstruction quality, yielding more detailed surface reconstruction than its global counterpart. Overall, we find that the introduced pipeline is able to recover high-quality 3D breast geometry within an error margin of less than 2 mm. Our method is fast (requires less than six minutes), fully transparent and open-source, and -- together with the model -- publicly available at this https URL.
>
---
#### [replaced 087] IMAGHarmony: Controllable Image Editing with Consistent Object Quantity and Layout
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.01949](https://arxiv.org/pdf/2506.01949)**

> **作者:** Fei Shen; Yutong Gao; Jian Yu; Xiaoyu Du; Jinhui Tang
>
> **摘要:** Despite advances in diffusion-based image editing, manipulating multi-object scenes remains challenging. Existing approaches often achieve semantic changes at the expense of structural consistency, failing to preserve exact object counts and spatial layouts without introducing unintended relocations or background modifications. To address this limitation, we introduce quantity-and-layout-consistent image editing (QL-Edit) to modify object semantics while maintaining the original instance cardinality and spatial layout. We propose IMAGHarmony, a parameter-efficient framework featuring a harmony-aware (HA) module that incorporates perception cues from the reference image into the diffusion process. This enables the model to jointly reason about object semantics, counts, and spatial positions for improved structural consistency. Furthermore, we introduce a preference-guided noise selection (PNS) strategy that identifies favorable initialization conditions, substantially improving generation stability in challenging multi-object scenarios. To support systematic evaluation, we construct HarmonyBench, a benchmark designed to measure semantic editing accuracy and structural consistency under quantity and layout constraints. Extensive experiments demonstrate that IMAGHarmony consistently outperforms existing methods in both structural preservation and semantic accuracy. Notably, our framework is highly efficient, requiring only 200 training images and 10.6M trainable parameters. Code, models, and data are available at \url{this https URL}.
>
---
#### [replaced 088] MLLM-based Textual Explanations for Face Comparison
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.16629](https://arxiv.org/pdf/2603.16629)**

> **作者:** Redwan Sony; Anil K Jain; Arun Ross
>
> **备注:** Accepted at 14th International Workshop on Biometrics and Forensics (IWBF)
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently been proposed as a means to generate natural-language explanations for face recognition decisions. While such explanations facilitate human interpretability, their reliability on unconstrained face images remains underexplored. In this work, we systematically analyze MLLM-generated explanations for the unconstrained face verification task on the challenging IJB-S dataset, with a particular focus on extreme pose variation and surveillance imagery. Our results show that even when MLLMs produce correct verification decisions, the accompanying explanations frequently rely on non-verifiable or hallucinated facial attributes that are not supported by visual evidence. We further study the effect of incorporating information from traditional face recognition systems, viz., scores and decisions, alongside the input images. Although such information improves categorical verification performance, it does not consistently lead to faithful explanations. To evaluate the explanations beyond decision accuracy, we introduce a likelihood-ratio-based framework that measures the evidential strength of textual explanations. Our findings highlight fundamental limitations of current MLLMs for explainable face recognition and underscore the need for a principled evaluation of reliable and trustworthy explanations in biometric applications. Code is available at this https URL.
>
---
#### [replaced 089] StreamDiT: Real-Time Streaming Text-to-Video Generation
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.03745](https://arxiv.org/pdf/2507.03745)**

> **作者:** Akio Kodaira; Tingbo Hou; Ji Hou; Markos Georgopoulos; Felix Juefei-Xu; Masayoshi Tomizuka; Yue Zhao
>
> **备注:** CVPR 2026
>
> **摘要:** Recently, great progress has been achieved in text-to-video (T2V) generation by scaling transformer-based diffusion models to billions of parameters, which can generate high-quality videos. However, existing models typically produce only short clips offline, restricting their use cases in interactive and real-time applications. This paper addresses these challenges by proposing StreamDiT, a streaming video generation model. StreamDiT training is based on flow matching by adding a moving buffer. We design mixed training with different partitioning schemes of buffered frames to boost both content consistency and visual quality. StreamDiT modeling is based on adaLN DiT with varying time embedding and window attention. To practice the proposed method, we train a StreamDiT model with 4B parameters. In addition, we propose a multistep distillation method tailored for StreamDiT. Sampling distillation is performed in each segment of a chosen partitioning scheme. After distillation, the total number of function evaluations (NFEs) is reduced to the number of chunks in a buffer. Finally, our distilled model reaches real-time performance at 16 FPS on one GPU, which can generate video streams at 512p resolution. We evaluate our method through both quantitative metrics and human evaluation. Our model enables real-time applications, e.g. streaming generation, interactive generation, and video-to-video. We provide video results and more examples in our project website: this https URL
>
---
#### [replaced 090] ModTrack: Sensor-Agnostic Multi-View Tracking via Identity-Informed PHD Filtering with Covariance Propagation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.15812](https://arxiv.org/pdf/2603.15812)**

> **作者:** Aditya Iyer; Jack Roberts; Nora Ayanian
>
> **摘要:** Multi-View Multi-Object Tracking (MV-MOT) aims to localize and maintain consistent identities of objects observed by multiple sensors. This task is challenging, as viewpoint changes and occlusion disrupt identity consistency across views and time. Recent end-to-end approaches address this by jointly learning 2D Bird's Eye View (BEV) representations and identity associations, achieving high tracking accuracy. However, these methods offer no principled uncertainty accounting and remain tightly coupled to their training configuration, limiting generalization across sensor layouts, modalities, or datasets without retraining. We propose ModTrack, a modular MV-MOT system that matches end-to-end performance while providing cross-modal, sensor-agnostic generalization and traceable uncertainty. ModTrack confines learning methods to just the \textit{Detection and Feature Extraction} stage of the MV-MOT pipeline, performing all fusion, association, and tracking with closed-form analytical methods. Our design reduces each sensor's output to calibrated position-covariance pairs $(\mathbf{z}, R)$; cross-view clustering and precision-weighted fusion then yield unified estimates $(\hat{\mathbf{z}}, \hat{R})$ for identity assignment and temporal tracking. A feedback-coupled, identity-informed Gaussian Mixture Probability Hypothesis Density (GM-PHD) filter with HMM motion modes uses these fused estimates to maintain identities under missed detections and heavy occlusion. ModTrack achieves 95.5 IDF1 and 91.4 MOTA on \textit{WildTrack}, surpassing all prior modular methods by over 21 points and rivaling the state-of-the-art end-to-end methods while providing deployment flexibility they cannot. Specifically, the same tracker core transfers unchanged to \textit{MultiviewX} and \textit{RadarScenes}, with only perception-module replacement required to extend to new domains and sensor modalities.
>
---
#### [replaced 091] MIBURI: Towards Expressive Interactive Gesture Synthesis
- **分类: cs.CV; cs.GR; cs.HC**

- **链接: [https://arxiv.org/pdf/2603.03282](https://arxiv.org/pdf/2603.03282)**

> **作者:** M. Hamza Mughal; Rishabh Dabral; Vera Demberg; Christian Theobalt
>
> **备注:** CVPR 2026 (Main). Project page: this https URL
>
> **摘要:** Embodied Conversational Agents (ECAs) aim to emulate human face-to-face interaction through speech, gestures, and facial expressions. Current large language model (LLM)-based conversational agents lack embodiment and the expressive gestures essential for natural interaction. Existing solutions for ECAs often produce rigid, low-diversity motions, that are unsuitable for human-like interaction. Alternatively, generative methods for co-speech gesture synthesis yield natural body gestures but depend on future speech context and require long run-times. To bridge this gap, we present MIBURI, the first online, causal framework for generating expressive full-body gestures and facial expressions synchronized with real-time spoken dialogue. We employ body-part aware gesture codecs that encode hierarchical motion details into multi-level discrete tokens. These tokens are then autoregressively generated by a two-dimensional causal framework conditioned on LLM-based speech-text embeddings, modeling both temporal dynamics and part-level motion hierarchy in real time. Further, we introduce auxiliary objectives to encourage expressive and diverse gestures while preventing convergence to static poses. Comparative evaluations demonstrate that our causal and real-time approach produces natural and contextually aligned gestures against recent baselines. We urge the reader to explore demo videos on this https URL.
>
---
#### [replaced 092] From Imitation to Intuition: Intrinsic Reasoning for Open-Instance Video Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10300](https://arxiv.org/pdf/2603.10300)**

> **作者:** Ke Zhang; Xiangchen Zhao; Yunjie Tian; Jiayu Zheng; Vishal M. Patel; Di Fu
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Conventional video classification models, acting as effective imitators, excel in scenarios with homogeneous data distributions. However, real-world applications often present an open-instance challenge, where intra-class variations are vast and complex, beyond existing benchmarks. While traditional video encoder models struggle to fit these diverse distributions, vision-language models (VLMs) offer superior generalization but have not fully leveraged their reasoning capabilities (intuition) for such tasks. In this paper, we bridge this gap with an intrinsic reasoning framework that evolves open-instance video classification from imitation to intuition. Our approach, namely DeepIntuit, begins with a cold-start supervised alignment to initialize reasoning capability, followed by refinement using Group Relative Policy Optimization (GRPO) to enhance reasoning coherence through reinforcement learning. Crucially, to translate this reasoning into accurate classification, DeepIntuit then introduces an intuitive calibration stage. In this stage, a classifier is trained on this intrinsic reasoning traces generated by the refined VLM, ensuring stable knowledge transfer without distribution mismatch. Extensive experiments demonstrate that for open-instance video classification, DeepIntuit benefits significantly from transcending simple feature imitation and evolving toward intrinsic reasoning. Our project is available at this https URL.
>
---
#### [replaced 093] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Wanderland框架，解决开放世界具身AI的可重复评估问题。通过高保真模拟和几何精准重建，提升导航与视图合成性能，推动相关研究。**

- **链接: [https://arxiv.org/pdf/2511.20620](https://arxiv.org/pdf/2511.20620)**

> **作者:** Xinhao Liu; Jiaqi Li; Youming Deng; Ruxin Chen; Yingjia Zhang; Yifei Ma; Li Guo; Yiming Li; Jing Zhang; Chen Feng
>
> **备注:** CVPR 2026
>
> **摘要:** Reproducible closed-loop evaluation remains a major bottleneck in Embodied AI such as visual navigation. A promising path forward is high-fidelity simulation that combines photorealistic sensor rendering with geometrically grounded interaction in complex, open-world urban environments. Although recent video-3DGS methods ease open-world scene capturing, they are still unsuitable for benchmarking due to large visual and geometric sim-to-real gaps. To address these challenges, we introduce Wanderland, a real-to-sim framework that features multi-sensor capture, reliable reconstruction, accurate geometry, and robust view synthesis. Using this pipeline, we curate a diverse dataset of indoor-outdoor urban scenes and systematically demonstrate how image-only pipelines scale poorly, how geometry quality impacts novel view synthesis, and how all of these adversely affect navigation policy learning and evaluation reliability. Beyond serving as a trusted testbed for embodied navigation, Wanderland's rich raw sensor data further allows benchmarking of 3D reconstruction and novel view synthesis models. Our work establishes a new foundation for reproducible research in open-world embodied AI. Project website is at this https URL.
>
---
#### [replaced 094] Fourier Decomposition for Explicit Representation of 3D Point Cloud Attributes
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2503.10055](https://arxiv.org/pdf/2503.10055)**

> **作者:** Donghyun Kim; Chanyoung Kim; Hyunah Ko; Seong Jae Hwang
>
> **摘要:** While 3D point clouds are widely used in vision applications, their irregular and sparse nature make them challenging to handle. In response, numerous encoding approaches have been proposed to capture the rich semantic information of point clouds. Yet, a critical limitation persists: a lack of consideration for colored point clouds, which serve as more expressive 3D representations encompassing both color and geometry. While existing methods handle color and geometry separately on a per-point basis, this leads to a limited receptive field and restricted ability to capture relationships across multiple points. To address this, we pioneer a colored point cloud encoding methodology that leverages 3D Fourier decomposition to disentangle color and geometric features while extending the receptive field through spectral-domain operations. Our analysis confirms that our approach effectively separates feature components, where the amplitude uniquely captures color attributes and the phase encodes geometric structure, thereby enabling independent learning and utilization of both attributes. We validate our colored point cloud encoding approach on classification, segmentation, and style transfer tasks, achieving state-of-the-art results on the DensePoint dataset.
>
---
#### [replaced 095] Score2Instruct: Scaling Up Video Quality-Centric Instructions via Automated Dimension Scoring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21011](https://arxiv.org/pdf/2506.21011)**

> **作者:** Qizhi Xie; Kun Yuan; Yunpeng Qu; Jiachao Gong; Mingda Wu; Ming Sun; Chao Zhou; Jihong Zhu
>
> **备注:** 16 pages, 5 figures. Accepted by CVPR 2026 main conference
>
> **摘要:** Classical video quality assessment methods generate a numerical score to judge a video's perceived visual fidelity and clarity. Yet, a score fails to describe the video's complex quality dimensions, restricting its applicability. Benefiting from the human-friendly linguistic output, adapting video large multimodal models to VQA via instruction tuning has the potential to address this issue. The core of the approach lies in the video quality-centric instruction data. Previous explorations mainly focus on the image domain, and their data generation processes heavily rely on human quality annotations and proprietary systems, limiting data scalability and effectiveness. To address these challenges, we propose the Score-based Instruction Generation pipeline. Specifically, SIG first scores multiple quality dimensions of an unlabeled video and maps scores to text-defined levels. It then explicitly incorporates a hierarchical Chain-of-Thought to model the correlation between specific dimensions and overall quality, mimicking the human visual system's reasoning process. The automated pipeline eliminates the reliance on expert-written quality descriptions and proprietary systems, ensuring data scalability and generation efficiency. To this end, the resulting Score2Instruct dataset contains over 320K diverse instruction-response pairs, laying the basis for instruction tuning. Moreover, to advance video LMMs' quality scoring and justification abilities simultaneously, we devise a progressive tuning strategy to fully unleash the power of S2I. Built upon SIG, we further curate a benchmark termed S2I-Bench with 400 open-ended questions to better evaluate the quality justification capacity of video LMMs. Experimental results on the S2I-Bench and existing benchmarks indicate that our method consistently improves quality scoring and justification capabilities across multiple video LMMs.
>
---
#### [replaced 096] Adaptive Multi-Scale Channel-Spatial Attention Aggregation Framework for 3D Indoor Semantic Scene Completion Toward Assisting Visually Impaired
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.16385](https://arxiv.org/pdf/2602.16385)**

> **作者:** Qi He; XiangXiang Wang; Jingtao Zhang; Yongbin Yu; Hongxiang Chu; Manping Fan; JingYe Cai; Zhenglin Yang
>
> **备注:** 17 pages, 9 figures, 5 tables
>
> **摘要:** Independent indoor mobility remains a critical challenge for individuals with visual impairments, largely due to the limited capability of existing assistive systems in detecting fine-grained hazardous objects such as chairs, tables, and small obstacles. These perceptual blind zones substantially increase the risk of collision in unfamiliar environments. To bridge the gap between monocular 3D vision research and practical assistive deployment, this paper proposes an Adaptive Multi-scale Attention Aggregation (AMAA) framework for monocular 3D semantic scene completion using only a wearable RGB camera. The proposed framework addresses two major limitations in 2D-to-3D feature lifting: noise diffusion during back-projection and structural instability in multi-scale fusion. A parallel channel--spatial attention mechanism is introduced to recalibrate lifted features along semantic and geometric dimensions, while a hierarchical adaptive gating strategy regulates cross-scale information flow to preserve fine-grained structural details. Experiments on the NYUv2 benchmark demonstrate that AMAA achieves an overall mIoU of 27.88%. Crucially, it yields significant relative improvements of 16.9% for small objects and 10.4% for tables over the MonoScene baseline. Furthermore, a wearable prototype based on an NVIDIA Jetson Orin NX and a ZED~2i camera validates stable real-time performance in indoor environments, demonstrating the feasibility of deploying monocular 3D scene completion for assistive navigation.
>
---
#### [replaced 097] HIFICL: High-Fidelity In-Context Learning for Multimodal Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12760](https://arxiv.org/pdf/2603.12760)**

> **作者:** Xiaoyu Li; Yuhang Liu; Xuanshuo Kang; Zheng Luo; Fangqi Lou; Xiaohua Wu; Zihan Xiong
>
> **备注:** Accepted to CVPR 2026. Code available at this https URL
>
> **摘要:** In-Context Learning (ICL) is a significant paradigm for Large Multimodal Models (LMMs), using a few in-context demonstrations (ICDs) for new task adaptation. However, its performance is sensitive to demonstration configurations and computationally expensive. Mathematically, the influence of these demonstrations can be decomposed into a dynamic mixture of the standard attention output and the context values. Current approximation methods simplify this process by learning a "shift vector". Inspired by the exact decomposition, we introduce High-Fidelity In-Context Learning (HIFICL) to more faithfully model the ICL mechanism. HIFICL consists of three key components: 1) a set of "virtual key-value pairs" to act as a learnable context, 2) a low-rank factorization for stable and regularized training, and 3) a simple end-to-end training objective. From another perspective, this mechanism constitutes a form of context-aware Parameter-Efficient Fine-Tuning (PEFT). Extensive experiments show that HiFICL consistently outperforms existing approximation methods on several multimodal benchmarks. The code is available at this https URL.
>
---
#### [replaced 098] One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.02898](https://arxiv.org/pdf/2510.02898)**

> **作者:** Lorenzo Bianchi; Giacomo Pacini; Fabio Carrara; Nicola Messina; Giuseppe Amato; Fabrizio Falchi
>
> **备注:** IEEE CVF Conference on Computer Vision and Pattern Recognition 2026. Project page with code, models and examples: this https URL
>
> **摘要:** Zero-shot captioners are recently proposed models that utilize common-space vision-language representations to caption images without relying on paired image-text data. To caption an image, they proceed by textually decoding a text-aligned image feature, but they limit their scope to global representations and whole-image captions. We present a unified framework for zero-shot captioning that shifts from an image-centric to a patch-centric paradigm, enabling the captioning of arbitrary regions without the need of region-level supervision. Instead of relying on global image representations, we treat individual patches as atomic captioning units and aggregate them to describe arbitrary regions, from single patches to non-contiguous areas and entire images. We analyze the key ingredients that enable current latent captioners to work in our novel proposed framework. Experiments demonstrate that backbones producing meaningful, dense visual features, such as DINO, are key to achieving state-of-the-art performance in multiple region-based captioning tasks. Compared to other baselines and state-of-the-art competitors, our models achieve better performance on zero-shot dense captioning and region-set captioning. We also introduce a new trace captioning task that further demonstrates the effectiveness of patch-wise semantic representations for flexible caption generation. Project page at this https URL .
>
---
#### [replaced 099] PokeFusion Attention: A Lightweight Cross-Attention Mechanism for Style-Conditioned Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03220](https://arxiv.org/pdf/2602.03220)**

> **作者:** Jingbang Tang
>
> **备注:** 12 pages, 5 figures. Revised version with improved method description and corrected references
>
> **摘要:** Style-conditioned text-to-image (T2I) generation with diffusion models requires both stable character structure and consistent, fine-grained style expression across diverse prompts. Existing approaches either rely on text-only prompting, which is often insufficient to specify visual style, or introduce reference-based adapters that depend on external images at inference time, increasing system complexity and limiting deployment flexibility. We propose PokeFusion Attention, a lightweight decoder-level cross-attention mechanism that models style as a learned distributional prior rather than instance-level conditioning. The method integrates textual semantics with learned style embeddings directly within the diffusion decoder, enabling effective stylized generation without requiring reference images at inference time. Only the cross-attention layers and a compact style projection module are trained, while the pretrained diffusion backbone remains frozen, resulting in a parameter-efficient and plug-and-play design. Experiments on a stylized character generation benchmark demonstrate that the proposed method improves style fidelity, semantic alignment, and structural consistency compared with representative adapter-based baselines, while maintaining low parameter overhead and simple inference.
>
---
#### [replaced 100] Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12887](https://arxiv.org/pdf/2512.12887)**

> **作者:** Han Liu; Bogdan Georgescu; Yanbo Zhang; Youngjin Yoo; Michael Baumgartner; Riqiang Gao; Jianing Wang; Gengyan Zhao; Eli Gibson; Dorin Comaniciu; Sasa Grbic
>
> **备注:** 1st Place in VLM3D Challenge
>
> **摘要:** 3D medical image classification is essential for modern clinical workflows. Medical foundation models (FMs) have emerged as a promising approach for scaling to new tasks, yet current research suffers from three critical pitfalls: data-regime bias, suboptimal adaptation, and insufficient task coverage. In this paper, we address these pitfalls and introduce AnyMC3D, a scalable 3D classifier adapted from 2D FMs. Our method scales efficiently to new tasks by adding only lightweight plugins (about 1M parameters per task) on top of a single frozen backbone. This versatile framework also supports multi-view inputs, auxiliary pixel-level supervision, and interpretable heatmap generation. We establish a comprehensive benchmark of 12 tasks covering diverse pathologies, anatomies, and modalities, and systematically analyze state-of-the-art 3D classification techniques. Our analysis reveals key insights: (1) effective adaptation is essential to unlock FM potential, (2) general-purpose FMs can match medical-specific FMs if properly adapted, and (3) 2D-based methods surpass 3D architectures for 3D classification. For the first time, we demonstrate the feasibility of achieving state-of-the-art performance across diverse applications using a single scalable framework (including 1st place in the VLM3D challenge), eliminating the need for separate task-specific models.
>
---
#### [replaced 101] High-Fidelity Human Avatars from Laptop Webcams using Edge Compute
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.02468](https://arxiv.org/pdf/2502.02468)**

> **作者:** Akash Haridas; Imran N. Junejo
>
> **备注:** 6 pages, 6 figures, 1 table
>
> **摘要:** Photo-realistic human avatars have broad applications, yet high-fidelity avatar generation has traditionally required expensive professional camera rigs and extensive artistic labor. Recent research has enabled constructing them automatically from smartphones with RGB and IR sensors, however, these new methods still rely on high-resolution cameras on modern smartphones and often require offloading the processing to powerful servers with GPUs. Modern applications such as video conferencing call for the ability to generate these avatars from consumer-grade laptop webcams using limited compute available on-device. In this work, we develop a novel method based on 3D morphable models, landmark detection, photorealistic texture GANs, and differentiable rendering to tackle the problem of low webcam image quality and edge computation. We build an automatic system to generate high-fidelity animatable avatars under these limitations, leveraging the compute capabilities of AMD mobile processors.
>
---
#### [replaced 102] Making Training-Free Diffusion Segmentors Scale with the Generative Power
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.06178](https://arxiv.org/pdf/2603.06178)**

> **作者:** Benyuan Meng; Qianqian Xu; Zitai Wang; Xiaochun Cao; Longtao Huang; Qingming Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** As powerful generative models, text-to-image diffusion models have recently been explored for discriminative tasks. A line of research focuses on adapting a pre-trained diffusion model to semantic segmentation without any further training, leading to training-free diffusion segmentors. These methods typically rely on cross-attention maps from the model's attention layers, which are assumed to capture semantic relationships between image pixels and text tokens. Ideally, such approaches should benefit from more powerful diffusion models, i.e., stronger generative capability should lead to better segmentation. However, we observe that existing methods often fail to scale accordingly. To understand this issue, we identify two underlying gaps: (i) cross-attention is computed across multiple heads and layers, but there exists a discrepancy between these individual attention maps and a unified global representation. (ii) Even when a global map is available, it does not directly translate to accurate semantic correlation for segmentation, due to score imbalances among different text tokens. To bridge these gaps, we propose two techniques: auto aggregation and per-pixel rescaling, which together enable training-free segmentation to better leverage generative capability. We evaluate our approach on standard semantic segmentation benchmarks and further integrate it into a generative technique, demonstrating both improved performance broad applicability. Codes are at this https URL.
>
---
#### [replaced 103] Hierarchical and Multimodal Data for Daily Activity Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.17696](https://arxiv.org/pdf/2504.17696)**

> **作者:** Ghazal Kaviani; Yavuz Yarici; Seulgi Kim; Mohit Prabhushankar; Ghassan AlRegib; Mashhour Solh; Ameya Patil
>
> **备注:** Accepted for publication in DMLR
>
> **摘要:** Daily Activity Recordings for Artificial Intelligence (DARai, pronounced "Dahr-ree") is a multimodal, hierarchically annotated dataset constructed to understand human activities in real-world settings. DARai consists of continuous scripted and unscripted recordings of 50 participants in 10 different environments, totaling over 200 hours of data from 20 sensors including multiple camera views, depth and radar sensors, wearable inertial measurement units (IMUs), electromyography (EMG), insole pressure sensors, biomonitor sensors, and gaze tracker. To capture the complexity in human activities, DARai is annotated at three levels of hierarchy: (i) high-level activities (L1) that are independent tasks, (ii) lower-level actions (L2) that are patterns shared between activities, and (iii) fine-grained procedures (L3) that detail the exact execution steps for actions. The dataset annotations and recordings are designed so that 22.7% of L2 actions are shared between L1 activities and 14.2% of L3 procedures are shared between L2 actions. The overlap and unscripted nature of DARai allows counterfactual activities in the dataset. Experiments with various machine learning models showcase the value of DARai in uncovering important challenges in human-centered applications. Specifically, we conduct unimodal and multimodal sensor fusion experiments for recognition, temporal localization, and future action anticipation across all hierarchical annotation levels. To highlight the limitations of individual sensors, we also conduct domain-variant experiments that are enabled by DARai's multi-sensor and counterfactual activity design setup. The code, documentation, and dataset are available at the dedicated DARai website: this https URL
>
---
#### [replaced 104] LoGSAM: Parameter-Efficient Cross-Modal Grounding for MRI Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.17576](https://arxiv.org/pdf/2603.17576)**

> **作者:** Mohammad Robaitul Islam Bhuiyan; Sheethal Bhat; Melika Qahqaie; Tri-Thien Nguyen; Paula Andrea Perez-Toro; Tomas Arias-Vergara; Andreas Maier
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Precise localization and delineation of brain tumors using Magnetic Resonance Imaging (MRI) are essential for planning therapy and guiding surgical decisions. However, most existing approaches rely on task-specific supervised models and are constrained by the limited availability of annotated data. To address this, we propose LoGSAM, a parameter-efficient, detection-driven framework that transforms radiologist dictation into text prompts for foundation-model-based localization and segmentation. Radiologist speech is first transcribed and translated using a pretrained Whisper ASR model, followed by negation-aware clinical NLP to extract tumor-specific textual prompts. These prompts guide text-conditioned tumor localization via a LoRA-adapted vision-language detection model, Grounding DINO (GDINO). The LoRA adaptation updates using 5% of the model parameters, thereby enabling computationally efficient domain adaptation while preserving pretrained cross-modal knowledge. The predicted bounding boxes are used as prompts for MedSAM to generate pixel-level tumor masks without any additional fine-tuning. Conditioning the frozen MedSAM on LoGSAM-derived priors yields a state-of-the-art dice score of 80.32% on BRISC 2025. In addition, we evaluate the full pipeline using German dictations from a board-certified radiologist on 12 unseen MRI scans, achieving 91.7% case-level accuracy. These results highlight the feasibility of constructing a modular, speech-to-segmentation pipeline by intelligently leveraging pretrained foundation models with minimal parameter updates.
>
---
#### [replaced 105] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21077](https://arxiv.org/pdf/2603.21077)**

> **作者:** Nan Zhou; Huiqun Wang; Yaoyan Zheng; Di Huang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Multimodal large language models (MLLMs) achieve remarkable progress in cross-modal perception and reasoning, yet a fundamental question remains unresolved: should the vision encoder be fine-tuned or frozen? Despite the success of models such as LLaVA and Qwen-VL, inconsistent design choices and heterogeneous training setups hinder a unified understanding of visual fine-tuning (VFT) in MLLMs. Through a configuration-aligned benchmark, we find that existing VFT methods fail to consistently outperform the frozen baseline across multimodal tasks. Our analysis suggests that this instability arises from visual preference conflicts, where the context-agnostic nature of vision encoders induces divergent parameter updates under diverse multimodal context. To address this issue, we propose the Context-aware Visual Fine-tuning (CoVFT) framework, which explicitly incorporates multimodal context into visual adaptation. By integrating a Context Vector Extraction (CVE) and a Contextual Mixture-of-Experts (CoMoE) module, CoVFT decomposes conflicting optimization signals and enables stable, context-sensitive visual updates. Extensive experiments on 12 multimodal benchmarks demonstrate that CoVFT achieves state-of-the-art performance with superior stability. Notably, fine-tuning a 7B MLLM with CoVFT surpasses the average performance of its 13B counterpart, revealing substantial untapped potential in visual encoder optimization within MLLMs.
>
---
#### [replaced 106] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23885](https://arxiv.org/pdf/2603.23885)**

> **作者:** Gengluo Li; Pengyuan Lyu; Chengquan Zhang; Huawen Shen; Liang Wu; Xingyu Wan; Gangyan Zeng; Han Hu; Can Ma; Yu Zhou
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Document parsing has recently advanced with multimodal large language models (MLLMs) that directly map document images to structured outputs. Traditional cascaded pipelines depend on precise layout analysis and often fail under casually captured or non-standard conditions. Although end-to-end approaches mitigate this dependency, they still exhibit repetitive, hallucinated, and structurally inconsistent predictions - primarily due to the scarcity of large-scale, high-quality full-page (document-level) end-to-end parsing data and the lack of structure-aware training strategies. To address these challenges, we propose a data-training co-design framework for robust end-to-end document parsing. A Realistic Scene Synthesis strategy constructs large-scale, structurally diverse full-page end-to-end supervision by composing layout templates with rich document elements, while a Document-Aware Training Recipe introduces progressive learning and structure-token optimization to enhance structural fidelity and decoding stability. We further build Wild-OmniDocBench, a benchmark derived from real-world captured documents for robustness evaluation. Integrated into a 1B-parameter MLLM, our method achieves superior accuracy and robustness across both scanned/digital and real-world captured scenarios. All models, data synthesis pipelines, and benchmarks will be publicly released to advance future research in document understanding.
>
---
#### [replaced 107] IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IRIS-SLAM，解决语义定位与建图中的几何-实例统一表示问题，提升地图一致性与回环检测可靠性。**

- **链接: [https://arxiv.org/pdf/2602.18709](https://arxiv.org/pdf/2602.18709)**

> **作者:** Tingyang Xiao; Liu Liu; Wei Feng; Zhengyu Zou; Xiaolin Zhou; Wei Sui; Hao Li; Dingwen Zhang; Zhizhong Su
>
> **摘要:** Geometry foundation models have significantly advanced dense geometric SLAM, yet existing systems often lack deep semantic understanding and robust loop closure capabilities. Meanwhile, contemporary semantic mapping approaches are frequently hindered by decoupled architectures and fragile data association. We propose IRIS-SLAM, a novel RGB semantic SLAM system that leverages unified geometric-instance representations derived from an instance-extended foundation model. By extending a geometry foundation model to concurrently predict dense geometry and cross-view consistent instance embeddings, we enable a semantic-synergized association mechanism and instance-guided loop closure detection. Our approach effectively utilizes viewpoint-agnostic semantic anchors to bridge the gap between geometric reconstruction and open-vocabulary mapping. Experimental results demonstrate that IRIS-SLAM significantly outperforms state-of-the-art methods, particularly in map consistency and wide-baseline loop closure reliability.
>
---
#### [replaced 108] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文提出DiFlowDubber，解决视频配音中的语音与唇形同步及语音表现力问题，通过跨模态对齐和离散流匹配实现更精准的自动配音。**

- **链接: [https://arxiv.org/pdf/2603.14267](https://arxiv.org/pdf/2603.14267)**

> **作者:** Ngoc-Son Nguyen; Thanh V. T. Tran; Jeongsoo Choi; Hieu-Nghia Huynh-Nguyen; Truong-Son Hy; Van Nguyen
>
> **备注:** Accepted at CVPR 2026 Findings
>
> **摘要:** Video dubbing has broad applications in filmmaking, multimedia creation, and assistive speech technology. Existing approaches either train directly on limited dubbing datasets or adopt a two-stage pipeline that adapts pre-trained text-to-speech (TTS) models, which often struggle to produce expressive prosody, rich acoustic characteristics, and precise synchronization. To address these issues, we propose DiFlowDubber with a novel two-stage training framework that effectively transfers knowledge from a pre-trained TTS model to video-driven dubbing, with a discrete flow matching generative backbone. Specifically, we design a FaPro module that captures global prosody and stylistic cues from facial expressions and leverages this information to guide the modeling of subsequent speech attributes. To ensure precise speech-lip synchronization, we introduce a Synchronizer module that bridges the modality gap among text, video, and speech, thereby improving cross-modal alignment and generating speech that is temporally synchronized with lip movements. Experiments on two primary benchmark datasets demonstrate that DiFlowDubber outperforms previous methods across multiple metrics.
>
---
