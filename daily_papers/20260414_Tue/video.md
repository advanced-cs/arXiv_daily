# 计算机视觉 cs.CV

- **最新发布 343 篇**

- **更新 181 篇**

## 最新发布

#### [new 001] LDEPrompt: Layer-importance guided Dual Expandable Prompt Pool for Pre-trained Model-based Class-Incremental Learning
- **分类: cs.CV**

- **简介: 该论文属于类增量学习任务，旨在解决固定提示池、人工选择和依赖预训练模型的问题。提出LDEPrompt方法，实现动态提示池扩展与自适应层选择。**

- **链接: [https://arxiv.org/pdf/2604.11091](https://arxiv.org/pdf/2604.11091)**

> **作者:** Linjie Li; Zhenyu Wu; Huiyu Xiao; Yang Ji
>
> **备注:** Accepted to ICASSP2026
>
> **摘要:** Prompt-based class-incremental learning methods typically construct a prompt pool consisting of multiple trainable key-prompts and perform instance-level matching to select the most suitable prompt embeddings, which has shown promising results. However, existing approaches face several limitations, including fixed prompt pools, manual selection of prompt embeddings, and strong reliance on the pretrained backbone for prompt selection. To address these issues, we propose a \textbf{L}ayer-importance guided \textbf{D}ual \textbf{E}xpandable \textbf{P}rompt Pool (\textbf{LDEPrompt}), which enables adaptive layer selection as well as dynamic freezing and expansion of the prompt pool. Extensive experiments on widely used class-incremental learning benchmarks demonstrate that LDEPrompt achieves state-of-the-art performance, validating its effectiveness and scalability.
>
---
#### [new 002] Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images
- **分类: cs.CV**

- **简介: 该论文属于3D表示学习任务，旨在解决从无姿态多视角图像中学习鲁棒3D表示的问题。通过提出双掩码策略、粗到细的高斯点云策略和位姿校准机制，提升几何与语义的一致性。**

- **链接: [https://arxiv.org/pdf/2604.10573](https://arxiv.org/pdf/2604.10573)**

> **作者:** Bo Zhou; Qiuxia Lai; Zeren Sun; Xiangbo Shu; Yazhou Yao; Wenguan Wang
>
> **备注:** CVPR 2026
>
> **摘要:** Robust 3D representation learning forms the perceptual foundation of spatial intelligence, enabling downstream tasks in scene understanding and embodied AI. However, learning such representations directly from unposed multi-view images remains challenging. Recent self-supervised methods attempt to unify geometry, appearance, and semantics in a feed-forward manner, but they often suffer from weak geometry induction, limited appearance detail, and inconsistencies between geometry and semantics. We introduce UniSplat, a feed-forward framework designed to address these limitations through three complementary components. First, we propose a dual-masking strategy that strengthens geometry induction in the encoder. By masking both encoder and decoder tokens, and targeting decoder masks toward geometry-rich regions, the model is forced to infer structural information from incomplete visual cues, yielding geometry-aware representations even under unposed inputs. Second, we develop a coarse-to-fine Gaussian splatting strategy that reduces appearance-semantics inconsistencies by progressively refining the radiance field. Finally, to enforce geometric-semantic consistency, we introduce a pose-conditioned recalibration mechanism that interrelates the outputs of multiple heads by re-projecting predicted 3D point and semantic maps into the image plane using estimated camera parameters, and aligning them with corresponding RGB and semantic predictions to ensure cross-task consistency, thereby resolving geometry-semantic mismatches. Together, these components yield unified 3D representations that are robust to unposed, sparse-view inputs and generalize across diverse tasks, laying a perceptual foundation for spatial intelligence.
>
---
#### [new 003] LARY: A Latent Action Representation Yielding Benchmark for Generalizable Vision-to-Action Alignment
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LARY基准，解决视觉到动作对齐问题，通过大规模视频数据评估潜在动作表示，验证通用视觉模型在物理控制中的有效性。**

- **链接: [https://arxiv.org/pdf/2604.11689](https://arxiv.org/pdf/2604.11689)**

> **作者:** Dujun Nie; Fengjiao Chen; Qi Lv; Jun Kuang; Xiaoyu Li; Xuezhi Cao; Xunliang Cai
>
> **备注:** Project: this https URL Code: this https URL Dataset: this https URL
>
> **摘要:** While the shortage of explicit action data limits Vision-Language-Action (VLA) models, human action videos offer a scalable yet unlabeled data source. A critical challenge in utilizing large-scale human video datasets lies in transforming visual signals into ontology-independent representations, known as latent actions. However, the capacity of latent action representation to derive robust control from visual observations has yet to be rigorously evaluated. We introduce the Latent Action Representation Yielding (LARY) Benchmark, a unified framework for evaluating latent action representations on both high-level semantic actions (what to do) and low-level robotic control (how to do). The comprehensively curated dataset encompasses over one million videos (1,000 hours) spanning 151 action categories, alongside 620K image pairs and 595K motion trajectories across diverse embodiments and environments. Our experiments reveal two crucial insights: (i) General visual foundation models, trained without any action supervision, consistently outperform specialized embodied latent action models. (ii) Latent-based visual space is fundamentally better aligned to physical action space than pixel-based space. These results suggest that general visual representations inherently encode action-relevant knowledge for physical control, and that semantic-level abstraction serves as a fundamentally more effective pathway from vision to action than pixel-level reconstruction.
>
---
#### [new 004] Architecture-Agnostic Modality-Isolated Gated Fusion for Robust Multi-Modal Prostate MRI Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态医学图像分割任务，解决MRI序列缺失或退化时的分割鲁棒性问题。提出MIGF模块，通过分离模态编码与门控机制提升模型在不完整输入下的性能。**

- **链接: [https://arxiv.org/pdf/2604.10702](https://arxiv.org/pdf/2604.10702)**

> **作者:** Yongbo Shu; Wenzhao Xie; Shanhu Yao; Zirui Xin; Luo Lei; Kewen Chen; Aijing Luo
>
> **备注:** 36 pages, 4 figures, 5 tables
>
> **摘要:** Multi-parametric prostate MRI -- combining T2-weighted, apparent diffusion coefficient, and high b-value diffusion-weighted sequences -- is central to non-invasive detection of clinically significant prostate cancer, yet in routine practice individual sequences may be missing or degraded by motion, artifacts, or abbreviated protocols. Existing multi-modal fusion strategies typically assume complete inputs and entangle modality-specific information at early layers, offering limited resilience when one channel is corrupted or absent. We propose Modality-Isolated Gated Fusion (MIGF), an architecture-agnostic module that maintains separate modality-specific encoding streams before a learned gating stage, combined with modality dropout training to enforce compensation behavior under incomplete inputs. We benchmark six bare backbones and assess MIGF-equipped models under seven missing-modality and artifact scenarios on the PI-CAI dataset (1,500 studies, fold-0 split, five random seeds). Among bare backbones, nnUNet provided the strongest balance of performance and stability. MIGF improved ideal-scenario Ranking Score for UNet, nnUNet, and Mamba by 2.8%, 4.6%, and 13.4%, respectively; the best model, MIGFNet-nnUNet (gating + ModDrop, no deep supervision), achieved 0.7304 +/- 0.056. Mechanistic analysis reveals that robustness gains arise from strict modality isolation and dropout-driven compensation rather than adaptive per-sample quality routing: the gate converged to a stable modality prior, and deep supervision was beneficial only for the largest backbone while degrading lighter models. These findings support a simpler design principle for robust multi-modal segmentation: structurally contain corrupted inputs first, then train explicitly for incomplete-input compensation.
>
---
#### [new 005] Decoupled Similarity for Task-Aware Token Pruning in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决token pruning中因注意力偏差导致的次优决策问题。提出DeSAP方法，通过解耦相似性与视觉显著性实现精准任务感知的token剪枝。**

- **链接: [https://arxiv.org/pdf/2604.11240](https://arxiv.org/pdf/2604.11240)**

> **作者:** Kexin Ma; Jing Xiao; Chaofeng Chen; Geyong Min; Guibo Zhu; Jinqiao Wang; Liang Liao
>
> **摘要:** Token pruning has emerged as an effective approach to reduce the substantial computational overhead of Large Vision-Language Models (LVLMs) by discarding less informative visual tokens while preserving performance. However, existing methods typically rely on individual attention sources from different LVLM components, resulting in incomplete and suboptimal pruning decisions due to biased attention distributions. To address this problem, we propose DeSAP, a novel Decoupled Similarity-Aware Pruning method for precise, task-aware token pruning within the visual encoder. Specifically, DeSAP introduces a decoupled similarity to capture fine-grained cross-modal relevance between visual features and text tokens, providing explicit task-related guidance for pruning. By integrating decoupled similarity with visual saliency signals derived from visual attention, DeSAP performs token pruning under the guidance of both task-related and visual cues, enabling robust pruning even under aggressive pruning ratios. Extensive experiments across diverse benchmarks and architectures show that DeSAP consistently outperforms SOTA methods in both accuracy and efficiency. On LLaVA-1.5-7B, DeSAP achieves a 10 times FLOPs reduction and a 2.3 times prefill speedup by retaining only 11.1% of visual tokens, while maintaining 98.1% of the original performance.
>
---
#### [new 006] Parameter Efficient Fine-tuning for Domain-specific Gastrointestinal Disease Recognition
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决跨源数据分布差异问题。通过引入低秩适配模块（LoRA），提升参数效率并优化下游任务性能。**

- **链接: [https://arxiv.org/pdf/2604.10451](https://arxiv.org/pdf/2604.10451)**

> **作者:** Sanjaya Poudel; Nikita Kunwor; Raj Simkhada; Mustafa Munir; Manish Dhakal; Khem Poudel
>
> **备注:** 6 pages, 3 figures, CVPR conference
>
> **摘要:** Despite recent advancements in the field of medical image analysis with the use of pretrained foundation models, the issue of distribution shifts between cross-source images largely remains adamant. To circumvent that issue, investigators generally train a separate model for each source. However, this method becomes expensive when we fully fine-tune pretrained large models for a single dataset, as we must store multiple copies of those models. Thus, in this work, we propose using a low-rank adaptation (LoRA) module for fine-tuning downstream classification tasks. LoRAs learn lightweight task-specific low-rank matrices that perturb pretrained weights to optimize those downstream tasks. For gastrointestinal tract diseases, they exhibit significantly better results than end-to-end finetuning with improved parameter efficiency. Code is available at: this http URL.
>
---
#### [new 007] Turning Generators into Retrievers: Unlocking MLLMs for Natural Language-Guided Geo-Localization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自然语言引导的地理定位任务，解决跨模态检索效率低的问题。通过微调MLLM提升其检索能力，实现更优的跨模态对齐与性能。**

- **链接: [https://arxiv.org/pdf/2604.10721](https://arxiv.org/pdf/2604.10721)**

> **作者:** Yuqi Chen; Xiaohan Zhang; Ahmad Arrabi; Waqas Sultani; Chen Chen; Safwan Wshah
>
> **备注:** CVPRF
>
> **摘要:** Natural-language Guided Cross-view Geo-localization (NGCG) aims to retrieve geo-tagged satellite imagery using textual descriptions of ground scenes. While recent NGCG methods commonly rely on CLIP-style dual-encoder architectures, they often suffer from weak cross-modal generalization and require complex architectural designs. In contrast, Multimodal Large Language Models (MLLMs) offer powerful semantic reasoning capabilities but are not directly optimized for retrieval tasks. In this work, we present a simple yet effective framework to adapt MLLMs for NGCG via parameter-efficient finetuning. Our approach optimizes latent representations within the MLLM while preserving its pretrained multimodal knowledge, enabling strong cross-modal alignment without redesigning model architectures. Through systematic analysis of diverse variables, from model backbone to feature aggregation, we provide practical and generalizable insights for leveraging MLLMs in NGCG. Our method achieves SOTA on GeoText-1652 with a 12.2% improvement in Text-to-Image Recall@1 and secures top performance in 5 out of 12 subtasks on CVG-Text, all while surpassing baselines with far fewer trainable parameters. These results position MLLMs as a robust foundation for semantic cross-view retrieval and pave the way for MLLM-based NGCG to be adopted as a scalable, powerful alternative to traditional dual-encoder designs. Project page and code are available at this https URL.
>
---
#### [new 008] Text-Guided 6D Object Pose Rearrangement via Closed-Loop VLM Agents
- **分类: cs.CV**

- **简介: 该论文属于6D物体位姿重排任务，旨在解决文本引导下目标物体在3D场景中的位姿预测问题。通过闭环交互与推理技术，提升VLM的3D理解能力，实现精准位姿调整。**

- **链接: [https://arxiv.org/pdf/2604.09781](https://arxiv.org/pdf/2604.09781)**

> **作者:** Sangwon Baik; Gunhee Kim; Mingi Choi; Hanbyul Joo
>
> **摘要:** Vision-Language Models (VLMs) exhibit strong visual reasoning capabilities, yet they still struggle with 3D understanding. In particular, VLMs often fail to infer a text-consistent goal 6D pose of a target object in a 3D scene. However, we find that with some inference-time techniques and iterative reasoning, VLMs can achieve dramatic performance gains. Concretely, given a 3D scene represented by an RGB-D image (or a compositional scene of 3D meshes) and a text instruction specifying a desired state change, we repeat the following loop: observe the current scene; evaluate whether it is faithful to the instruction; propose a pose update for the target object; apply the update; and render the updated scene. Through this closed-loop interaction, the VLM effectively acts as an agent. We further introduce three inference-time techniques that are essential to this closed-loop process: (i) multi-view reasoning with supporting view selection, (ii) object-centered coordinate system visualization, and (iii) single-axis rotation prediction. Without any additional fine-tuning or new modules, our approach surpasses prior methods at predicting the text-guided goal 6D pose of the target object. It works consistently across both closed-source and open-source VLMs. Moreover, when combining our 6D pose prediction with simple robot motion planning, it enables more successful robot manipulation than existing methods. Finally, we conduct an ablation study to demonstrate the necessity of each proposed technique.
>
---
#### [new 009] NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在提升模型在真实场景下的鲁棒性。通过构建包含多种变换的数据集，评估模型在不同图像处理下的检测能力。**

- **链接: [https://arxiv.org/pdf/2604.11487](https://arxiv.org/pdf/2604.11487)**

> **作者:** Aleksandr Gushchin; Khaled Abud; Ekaterina Shumitskaya; Artem Filippov; Georgii Bychkov; Sergey Lavrushkin; Mikhail Erofeev; Anastasia Antsiferova; Changsheng Chen; Shunquan Tan; Radu Timofte; Dmitry Vatolin; Chuanbiao Song; Zijian Yu; Hao Tan; Jun Lan; Zhiqiang Yang; Yongwei Tang; Zhiqiang Wu; Jia Wen Seow; Hong Vin Koay; Haodong Ren; Feng Xu; Shuai Chen; Ruiyang Xia; Qi Zhang; Yaowen Xu; Zhaofan Zou; Hao Sun; Dagong Lu; Mufeng Yao; Xinlei Xu; Fei Wu; Fengjun Guo; Cong Luo; Hardik Sharma; Aashish Negi; Prateek Shaily; Jayant Kumar; Sachin Chaudhary; Akshay Dudhane; Praful Hambarde; Amit Shukla; Zhilin Tu; Fengpeng Li; Jiamin Zhang; Jianwei Fei; Kemou Li; Haiwei Wu; Bilel Benjdira; Anas M. Ali; Wadii Boulila; Chenfan Qu; Junchi Li
>
> **备注:** CVPR 2026 NTIRE Workshop Paper, Robust AI-Generated Image Detection Technical Report
>
> **摘要:** This paper presents an overview of the NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild, held in conjunction with the NTIRE workshop at CVPR 2026. The goal of this challenge was to develop detection models capable of distinguishing real images from generated ones in realistic scenarios: the images are often transformed (cropped, resized, compressed, blurred) for practical usage, and therefore, the detection models should be robust to such transformations. The challenge is based on a novel dataset consisting of 108,750 real and 185,750 AI-generated images from 42 generators comprising a large variety of open-source and closed-source models of various architectures, augmented with 36 image transformations. Methods were evaluated using ROC AUC on the full test set, including both transformed and untransformed images. A total of 511 participants registered, with 20 teams submitting valid final solutions. This report provides a comprehensive overview of the challenge, describes the proposed solutions, and can be used as a valuable reference for researchers and practitioners in increasing the robustness of the detection models to real-world transformations.
>
---
#### [new 010] ACCIDENT: A Benchmark Dataset for Vehicle Accident Detection from Traffic Surveillance Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ACCIDENT数据集，用于车辆事故检测任务，解决监督和零样本场景下的事故定位与分类问题。工作包括数据收集、标注及基准测试。**

- **链接: [https://arxiv.org/pdf/2604.09819](https://arxiv.org/pdf/2604.09819)**

> **作者:** Lukas Picek; Michal Čermák; Marek Hanzl; Vojtěch Čermák
>
> **摘要:** We introduce ACCIDENT, a benchmark dataset for traffic accident detection in CCTV footage, designed to evaluate models in supervised (IID and OOD) and zero-shot settings, reflecting both data-rich and data-scarce scenarios. The benchmark consists of a curated set of 2,027 real and 2,211 synthetic clips annotated with the accident time, spatial location, and high-level collision type. We define three core tasks: (i) temporal localization of the accident, (ii) its spatial localization, and (iii) collision type classification. Each task is evaluated using custom metrics that account for the uncertainty and ambiguity inherent in CCTV footage. In addition to the benchmark, we provide a diverse set of baselines, including heuristic, motion-aware, and vision-language approaches, and show that ACCIDENT is challenging. You can access the ACCIDENT at: this https URL
>
---
#### [new 011] Using Deep Learning Models Pretrained by Self-Supervised Learning for Protein Localization
- **分类: cs.CV**

- **简介: 该论文研究使用自监督学习预训练的深度学习模型进行蛋白质定位任务，解决小样本数据训练难题，通过迁移学习提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.10970](https://arxiv.org/pdf/2604.10970)**

> **作者:** Ben Isselmann; Dilara Göksu; Heinz Neumann; Andreas Weinmann
>
> **备注:** 29 pages, 8 figures, submitted to BMC Bioinformatics
>
> **摘要:** Background: Task-specific microscopy datasets are often small, making it difficult to train deep learning models that learn robust features. While self-supervised learning (SSL) has shown promise through pretraining on large, domain-specific datasets, generalizability across datasets with differing staining protocols and channel configurations remains underexplored. We investigated the generalizability of SSL models pretrained on ImageNet-1k and HPA FOV, evaluating their embeddings on OpenCell with and without fine-tuning, two channel-mismatch strategies, and varying fine-tuning data fractions. We additionally analyzed single-cell embeddings on a labeled OpenCell subset. Result: DINO-based ViT backbones pretrained on HPA FOV or ImageNet-1k transfer well to OpenCell even without fine-tuning. The HPA FOV-pretrained model achieved the highest zero-shot performance (macro $F_1$ 0.822 $\pm$ 0.007). Fine-tuning further improved performance to 0.860 $\pm$ 0.013. At the single-cell level, the HPA single-cell-pretrained model achieved the highest k-nearest neighbor performance across all neighborhood sizes (macro $F_1$ $\geq$ 0.796). Conclusion: SSL methods like DINO, pretrained on large domain-relevant datasets, enable effective use of deep learning features for fine-tuning on small, task-specific microscopy datasets.
>
---
#### [new 012] Toward Accountable AI-Generated Content on Social Platforms: Steganographic Attribution and Multimodal Harm Detection
- **分类: cs.CV; cs.AI; cs.CR; cs.ET**

- **简介: 该论文属于AI内容审计任务，旨在解决AI生成内容的溯源与有害信息检测问题。通过隐写术嵌入标识符，并结合多模态检测实现可靠追踪。**

- **链接: [https://arxiv.org/pdf/2604.10460](https://arxiv.org/pdf/2604.10460)**

> **作者:** Xinlei Guan; David Arosemena; Tejaswi Dhandu; Kuan Huang; Meng Xu; Miles Q. Li; Bingyu Shen; Ruiyang Qin; Umamaheswara Rao Tida; Boyang Li
>
> **备注:** 12 pages, 31 figures
>
> **摘要:** The rapid growth of generative AI has introduced new challenges in content moderation and digital forensics. In particular, benign AI-generated images can be paired with harmful or misleading text, creating difficult-to-detect misuse. This contextual misuse undermines the traditional moderation framework and complicates attribution, as synthetic images typically lack persistent metadata or device signatures. We introduce a steganography enabled attribution framework that embeds cryptographically signed identifiers into images at creation time and uses multimodal harmful content detection as a trigger for attribution verification. Our system evaluates five watermarking methods across spatial, frequency, and wavelet domains. It also integrates a CLIP-based fusion model for multimodal harmful-content detection. Experiments demonstrate that spread-spectrum watermarking, especially in the wavelet domain, provides strong robustness under blur distortions, and our multimodal fusion detector achieves an AUC-ROC of 0.99, enabling reliable cross-modal attribution verification. These components form an end-to-end forensic pipeline that enables reliable tracing of harmful deployments of AI-generated imagery, supporting accountability in modern synthetic media environments. Our code is available at GitHub: this https URL
>
---
#### [new 013] Are We Recognizing the Jaguar or Its Background? A Diagnostic Framework for Jaguar Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于野生动物重识别任务，旨在解决模型依赖错误特征（如背景而非毛色）的问题。通过构建基准和诊断框架，评估并改进模型的识别依据。**

- **链接: [https://arxiv.org/pdf/2604.09690](https://arxiv.org/pdf/2604.09690)**

> **作者:** Antonio Rueda-Toicen; Abigail Allen Martin; Daniil Morozov; Matin Mahmood; Alexandra Schild; Shahabeddin Dayani; Davide Panza; Gerard de Melo
>
> **备注:** 33 pages, 11 figures
>
> **摘要:** Jaguar re-identification (re-ID) from citizen-science imagery can look strong on standard retrieval metrics while still relying on the wrong evidence, such as background context or silhouette shape, instead of the coat pattern that defines identity. We introduce a diagnostic framework for wildlife re-ID with two axes: a leakage-controlled context ratio, background/foreground, computed from inpainted background-only versus foreground-only images, and a laterality diagnostic based on cross-flank retrieval and mirror self-similarity. To make these diagnostics measurable, we curate a Pantanal jaguar benchmark with per-pixel segmentation masks and an identity-balanced evaluation protocol. We then use representative mitigation families, ArcFace fine-tuning, anti-symmetry regularization, and Lorentz hyperbolic embeddings, as case studies under the same evaluation lens. The goal is not only to ask which model ranks best, but also what visual evidence it uses to do so.
>
---
#### [new 014] LAST: Leveraging Tools as Hints to Enhance Spatial Reasoning for Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于空间推理任务，旨在解决MLLM在复杂几何布局解析中的幻觉和不精确问题。提出LAST框架，通过整合视觉工具提升空间推理能力。**

- **链接: [https://arxiv.org/pdf/2604.09712](https://arxiv.org/pdf/2604.09712)**

> **作者:** Shi-Yu Tian; Zhi Zhou; Kun-Yang Yu; Ming Yang; Yang Chen; Ziqiao Shang; Lan-Zhe Guo; Yu-Feng Li
>
> **备注:** 13 pages
>
> **摘要:** Spatial reasoning is a cornerstone capability for intelligent systems to perceive and interact with the physical world. However, multimodal large language models (MLLMs) frequently suffer from hallucinations and imprecision when parsing complex geometric layouts. As data-driven scaling struggles to internalize structured geometric priors and spatial constraints, integrating mature, specialized vision models presents a compelling alternative. Despite its promise, applying this paradigm to spatial reasoning is hindered by two key challenges: The difficulty of invoking heterogeneous, parameter-rich tools, as well as the challenge of understanding and effectively leveraging their diverse low-level outputs (e.g., segmentation masks, depth maps) in high-level reasoning. To address these challenges, we propose LAST, a unified framework for tool-augmented spatial reasoning. LAST features an extensible interactive sandbox, termed LAST-Box, which abstracts heterogeneous tool invocations into atomic instructions and reusable spatial skills, returning multimodal hints (e.g., annotated images and textual descriptions) that can be directly consumed by LLMs. We further design a three-stage progressive training strategy that guides models from understanding tool outputs to proficient and adaptive tool invocation. Experiments on four datasets show that LAST-7B achieves around 20\% performance gains over its backbone and outperforms strong proprietary closed-source LLMs, substantially enhancing reasoning on complex spatial tasks.
>
---
#### [new 015] Beyond Model Design: Data-Centric Training and Self-Ensemble for Gaussian Color Image Denoising
- **分类: cs.CV**

- **简介: 该论文针对高斯噪声彩色图像去噪任务，通过数据增强和两阶段优化提升模型性能，采用自集成策略进一步释放模型能力。**

- **链接: [https://arxiv.org/pdf/2604.11468](https://arxiv.org/pdf/2604.11468)**

> **作者:** Gengjia Chang; Xining Ge; Weijun Yuan; Zhan Li; Qiurong Song; Luen Zhu; Shuhong Liu
>
> **摘要:** This paper presents our solution to the NTIRE 2026 Image Denoising Challenge (Gaussian color image denoising at fixed noise level $\sigma = 50$). Rather than proposing a new restoration backbone, we revisit the performance boundary of the mature Restormer architecture from two complementary directions: stronger data-centric training and more complete Test-Time capability release. Starting from the public Restormer $\sigma\!=\!50$ baseline, we expand the standard multi-dataset training recipe with larger and more diverse public image corpora and organize optimization into two stages. At inference, we apply $\times 8$ geometric self-ensemble to further release model capacity. A TLC-style local inference wrapper is retained for implementation consistency; however, systematic ablation reveals its quantitative contribution to be negligible in this setting. On the challenge validation set of 100 images, our final submission achieves 30.762 dB PSNR and 0.861 SSIM, improving over the public Restormer $\sigma\!=\!50$ pretrained baseline by up to 3.366 dB PSNR. Ablation studies show that the dominant gain originates from the expanded training corpus and the two-stage optimization schedule, and self-ensemble provides marginal but consistent improvement.
>
---
#### [new 016] Head-wise Modality Specialization within MLLMs for Robust Fake News Detection under Missing Modality
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态虚假新闻检测任务，解决缺失模态下的鲁棒性问题。通过引入头级模态专精机制和单模态知识保留策略，提升模型在模态缺失时的验证能力。**

- **链接: [https://arxiv.org/pdf/2604.09711](https://arxiv.org/pdf/2604.09711)**

> **作者:** Kai Qian; Weijie Shi; Jiaqi Wang; Mengze Li; Hao Chen; Yue Cui; Hanghui Guo; Ziyi Liu; Jia Zhu; Jiajie Xu
>
> **摘要:** Multimodal fake news detection (MFND) aims to verify news credibility by jointly exploiting textual and visual evidence. However, real-world news dissemination frequently suffers from missing modality due to deleted images, corrupted screenshots, and similar issues. Thus, robust detection in this scenario requires preserving strong verification ability for each modality, which is challenging in MFND due to insufficient learning of the low-contribution modality and scarce unimodal annotations. To address this issue, we propose Head-wise Modality Specialization within Multimodal Large Language Models (MLLMs) for robust MFND under missing modality. Specifically, we first systematically study attention heads in MLLMs and their relationship with performance under missing modality, showing that modality-critical heads serve as key carriers of unimodal verification ability through their modality specialization. Based on this observation, to better preserve verification ability for the low-contribution modality, we introduce a head-wise specialization mechanism that explicitly allocates these heads to different modalities and preserves their specialization through lower-bound attention constraints. Furthermore, to better exploit scarce unimodal annotations, we propose a Unimodal Knowledge Retention strategy that prevents these heads from drifting away from the unimodal knowledge learned from limited supervision. Experiments show that our method improves robustness under missing modality while preserving performance with full multimodal input.
>
---
#### [new 017] LIDARLearn: A Unified Deep Learning Library for 3D Point Cloud Classification, Segmentation, and Self-Supervised Representation Learning
- **分类: cs.CV**

- **简介: 该论文提出\lib{}，一个统一的深度学习库，用于3D点云分类、分割和自监督表示学习，解决模型实现分散、难以公平比较的问题。**

- **链接: [https://arxiv.org/pdf/2604.10780](https://arxiv.org/pdf/2604.10780)**

> **作者:** Said Ohamouddou; Hanaa El Afia; Abdellatif El Afia; Raddouane Chiheb
>
> **摘要:** Three-dimensional (3D) point cloud analysis has become central to applications ranging from autonomous driving and robotics to forestry and ecological monitoring. Although numerous deep learning methods have been proposed for point cloud understanding, including supervised backbones, self-supervised pre-training (SSL), and parameter-efficient fine-tuning (PEFT), their implementations are scattered across incompatible codebases with differing data pipelines, evaluation protocols, and configuration formats, making fair comparisons difficult. We introduce \lib{}, a unified, extensible PyTorch library that integrates over 55 model configurations covering 29 supervised architectures, seven SSL pre-training methods, and five PEFT strategies, all within a single registry-based framework supporting classification, semantic segmentation, part segmentation, and few-shot learning. \lib{} provides standardised training runners, cross-validation with stratified $K$-fold splitting, automated LaTeX/CSV table generation, built-in Friedman/Nemenyi statistical testing with critical-difference diagrams for rigorous multi-model comparison, and a comprehensive test suite with 2\,200+ automated tests validating every configuration end-to-end. The code is available at this https URL under the MIT licence.
>
---
#### [new 018] Visual Enhanced Depth Scaling for Multimodal Latent Reasoning
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决视觉信息优化不足和复杂token梯度不稳定问题。提出视觉重放模块和深度缩放机制，提升视觉感知与深层推理能力。**

- **链接: [https://arxiv.org/pdf/2604.10500](https://arxiv.org/pdf/2604.10500)**

> **作者:** Yudong Han; Yong Wang; Zaiquan Yang; Zhen Qu; Liyuan Pan; Xiangxiang Chu
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Multimodal latent reasoning has emerged as a promising paradigm that replaces explicit Chain-of-Thought (CoT) decoding with implicit feature propagation, simultaneously enhancing representation informativeness and reducing inference latency. By analyzing token-level gradient dynamics during latent training, we reveal two critical observations: (1) visual tokens exhibit significantly higher and more volatile gradient norms than their textual counterparts due to inherent language bias, resulting in systematic visual under-optimization; and (2) semantically simple tokens converge rapidly, whereas complex tokens exhibit persistent gradient instability constrained by fixed architectural depths. To address these limitations, we propose a visual replay module and routing depth scaling to collaboratively enhance visual perception and refine complicated latents for deeper contextual reasoning. The former module leverages causal self-attention to estimate token saliency, reinforcing fine-grained grounding through spatially-coherent constraints. Complementarily, the latter mechanism adaptively allocates additional reasoning steps to complex tokens, enabling deeper contextual refinement. Guided by a curriculum strategy that progressively internalizes explicit CoT into compact latent representations, our framework achieves state-of-the-art performance across diverse benchmarks while delivering substantial inference speedups over explicit CoT baselines.
>
---
#### [new 019] MLLM-as-a-Judge Exhibits Model Preference Bias
- **分类: cs.CV**

- **简介: 该论文属于模型评估任务，旨在解决MLLM-as-a-Judge方法中的模型偏好偏差问题。通过提出Philautia-Eval分析偏差来源，并引入Pomms缓解偏差。**

- **链接: [https://arxiv.org/pdf/2604.11589](https://arxiv.org/pdf/2604.11589)**

> **作者:** Shuitsu Koyama; Yuiga Wada; Daichi Yashima; Komei Sugiura
>
> **摘要:** Automatic evaluation using multimodal large language models (MLLMs), commonly referred to as MLLM-as-a-Judge, has been widely used to measure model performance. If such MLLM-as-a-Judge methods were biased, they could distort model comparisons and benchmark-driven scientific progress. However, it remains unclear to what extent MLLM-as-a-Judge methods favor or disfavor text generated by specific MLLMs. In this study, we propose Philautia-Eval to investigate such model-specific preference bias. Philautia-Eval quantifies the degree of the bias by disentangling preference tendencies from differences in generation quality. Using 1.29M caption-score pairs collected from 12 MLLMs, we found that representative MLLMs tend to exhibit self-preference bias. Moreover, experimental results indicate mutual preference bias within particular model families, which is potentially driven by reused connectors and overlapping instruction-tuning resources. Finally, we introduce a simple ensemble of MLLMs, Pomms. Our results demonstrated that Pomms effectively mitigated the model-specific preference bias while maintaining performance.
>
---
#### [new 020] BLPR: Robust License Plate Recognition under Viewpoint and Illumination Variations via Confidence-Driven VLM Fallback
- **分类: cs.CV**

- **简介: 该论文属于车牌识别任务，针对光照和视角变化下的识别难题，提出BLPR框架，结合合成数据与真实数据训练，并引入视觉-语言模型提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.09927](https://arxiv.org/pdf/2604.09927)**

> **作者:** Guillermo Auza Banegas; Diego Calvimontes Vera; Sergio Castro Sandoval; Natalia Condori Peredo; Edwin Salcedo
>
> **摘要:** Robust license plate recognition in unconstrained environments remains a significant challenge, particularly in underrepresented regions with limited data availability and unique visual characteristics, such as Bolivia. Recognition accuracy in real-world conditions is often degraded by factors such as illumination changes and viewpoint distortion. To address these challenges, we introduce BLPR, a novel deep learning-based License Plate Detection and Recognition (LPDR) framework specifically designed for Bolivian license plates. The proposed system follows a two-stage pipeline where a YOLO-based detector is pretrained on synthetic data generated in Blender to simulate extreme perspectives and lighting conditions, and subsequently fine-tuned on street-level data collected in La Paz, Bolivia. Detected plates are geometrically rectified and passed to a character recognition model. To improve robustness under ambiguous scenarios, a lightweight vision-language model (Gemma3 4B) is selectively triggered as a confidence-based fallback mechanism. The proposed framework further leverages synthetic-to-real domain adaptation to improve robustness under diverse real-world conditions. We also introduce the first publicly available Bolivian LPDR dataset, enabling evaluation under diverse viewpoint and illumination conditions. The system achieves a character-level recognition accuracy of 89.6% on real-world data, demonstrating its effectiveness for deployment in challenging urban environments. Our project is publicly available at this https URL.
>
---
#### [new 021] Self-supervised Pretraining of Cell Segmentation Models
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于细胞实例分割任务，解决标注数据稀缺问题。通过自监督预训练和领域适应，提出DINOCell框架，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.10609](https://arxiv.org/pdf/2604.10609)**

> **作者:** Kaden Stillwagon; Alexandra Dunnum VandeLoo; Benjamin Magondu; Craig R. Forest
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Instance segmentation enables the analysis of spatial and temporal properties of cells in microscopy images by identifying the pixels belonging to each cell. However, progress is constrained by the scarcity of high-quality labeled microscopy datasets. Many recent approaches address this challenge by initializing models with segmentation-pretrained weights from large-scale natural-image models such as Segment Anything Model (SAM). However, representations learned from natural images often encode objectness and texture priors that are poorly aligned with microscopy data, leading to degraded performance under domain shift. We propose DINOCell, a self-supervised framework for cell instance segmentation that leverages representations from DINOv2 and adapts them to microscopy through continued self-supervised training on unlabeled cell images prior to supervised fine-tuning. On the LIVECell benchmark, DINOCell achieves a SEG score of 0.784, improving by 10.42% over leading SAM-based models, and demonstrates strong zero-shot performance on three out-of-distribution microscopy datasets. These results highlight the benefits of domain-adapted self-supervised pretraining for robust cell segmentation.
>
---
#### [new 022] MatRes: Zero-Shot Test-Time Model Adaptation for Simultaneous Matching and Restoration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MatRes，解决图像恢复与几何匹配的相互干扰问题。通过单对低质与高质图像，联合提升恢复质量和对应估计，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2604.10081](https://arxiv.org/pdf/2604.10081)**

> **作者:** Kanggeon Lee; Soochahn Lee; Kyoung Mu Lee
>
> **摘要:** Real-world image pairs often exhibit both severe degradations and large viewpoint changes, making image restoration and geometric matching mutually interfering tasks when treated independently. In this work, we propose MatRes, a zero-shot test-time adaptation framework that jointly improves restoration quality and correspondence estimation using only a single low-quality and high-quality image pair. By enforcing conditional similarity at corresponding locations, MatRes updates only lightweight modules while keeping all pretrained components frozen, requiring no offline training or additional supervision. Extensive experiments across diverse combinations show that MatRes yields significant gains in both restoration and geometric alignment compared to using either restoration or matching models alone. MatRes offers a practical and widely applicable solution for real-world scenarios where users commonly capture multiple images of a scene with varying viewpoints and quality, effectively addressing the often-overlooked mutual interference between matching and restoration.
>
---
#### [new 023] Development and evaluation of CADe systems in low-prevalence setting: The RARE25 challenge for early detection of Barrett's neoplasia
- **分类: cs.CV**

- **简介: 该论文属于早期Barrett食管癌变的CADe系统研究，旨在解决低患病率下的检测难题。通过构建真实场景数据集，评估不同方法的性能，推动更稳健的检测系统发展。**

- **链接: [https://arxiv.org/pdf/2604.11171](https://arxiv.org/pdf/2604.11171)**

> **作者:** Tim J.M. Jaspers; Francisco Caetano; Cris H.B. Claessens; Carolus H.J. Kusters; Rixta A.H. van Eijck van Heslinga; Floor Slooter; Jacques J. Bergman; Peter H.N. De With; Martijn R. Jong; Albert J. de Groof; Fons van der Sommen
>
> **备注:** The final author list is currently being finalized and will be updated in subsequent versions
>
> **摘要:** Computer-aided detection (CADe) of early neoplasia in Barrett's esophagus is a low-prevalence surveillance problem in which clinically relevant findings are rare. Although many CADe systems report strong performance on balanced or enriched datasets, their behavior under realistic prevalence remains insufficiently characterized. The RARE25 challenge addresses this gap by introducing a large-scale, prevalence-aware benchmark for neoplasia detection. It includes a public training set and a hidden test set reflecting real-world incidence. Methods were evaluated using operating-point-specific metrics emphasizing high sensitivity and accounting for prevalence. Eleven teams from seven countries submitted approaches using diverse architectures, pretraining, ensembling, and calibration strategies. While several methods achieved strong discriminative performance, positive predictive values remained low, highlighting the difficulty of low-prevalence detection and the risk of overestimating clinical utility when prevalence is ignored. All methods relied on fully supervised classification despite the dominance of normal findings, indicating a lack of prevalence-agnostic approaches such as anomaly detection or one-class learning. By releasing a public dataset and a reproducible evaluation framework, RARE25 aims to support the development of CADe systems robust to prevalence shift and suitable for clinical surveillance workflows.
>
---
#### [new 024] How to Design a Compact High-Throughput Video Camera?
- **分类: cs.CV**

- **简介: 论文属于视频成像任务，旨在解决高吞吐量视频采集中的读取与传输瓶颈。通过设计低比特梯度相机和多尺度重建CNN，提升系统效率与图像质量。**

- **链接: [https://arxiv.org/pdf/2604.10619](https://arxiv.org/pdf/2604.10619)**

> **作者:** Chenxi Qiu; Tao Yue; Xuemei Hu
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** High throughput video acquisition is a challenging problem and has been drawing increasing attention. Existing high throughput imaging systems splice hundreds of sub-images/videos into high throughput videos, suffering from extremely high system complexity. Alternatively, with pixel sizes reducing to sub-micrometer levels, integrating ultra-high throughput on a single chip is becoming feasible. Nevertheless, the readout and output transmission speed cannot keep pace with the increasing pixel numbers. To this end, this paper analyzes the strength of gradient cameras in fast readout and efficient representation, and proposes a low-bit gradient camera scheme based on existing technologies that can resolve the readout and transmission bottlenecks for high throughput video imaging. A multi-scale reconstruction CNN is proposed to reconstruct high-resolution images. Extensive experiments on both simulated and real data are conducted to demonstrate the promising quality and feasibility of the proposed method.
>
---
#### [new 025] PA-SFM: Tracker-free differentiable acoustic radiation for freehand 3D photoacoustic imaging
- **分类: cs.CV**

- **简介: 该论文属于3D成像任务，旨在解决手持光声成像中运动伪影问题。通过无需追踪器的框架PA-SFM，利用单模态数据实现高精度定位与重建。**

- **链接: [https://arxiv.org/pdf/2604.09643](https://arxiv.org/pdf/2604.09643)**

> **作者:** Shuang Li; Jian Gao; Chulhong Kim; Seongwook Choi; Qian Chen; Yibing Wang; Shuang Wu; Yu Zhang; Tingting Huang; Yucheng Zhou; Boxin Yao; Yao Yao; Changhui Li
>
> **摘要:** Three-dimensional (3D) handheld photoacoustic tomography typically relies on bulky and expensive external positioning sensors to correct motion artifacts, which severely limits its clinical flexibility and accessibility. To address this challenge, we present PA-SFM, a tracker-free framework that leverages exclusively single-modality photoacoustic data for both sensor pose recovery and high-fidelity 3D reconstruction via differentiable acoustic radiation modeling. Unlike traditional structure-from-motion (SFM) methods based on visual features, PA-SFM integrates the acoustic wave equation into a differentiable programming pipeline. By leveraging a high-performance, GPU-accelerated acoustic radiation kernel, the framework simultaneously optimizes the 3D photoacoustic source distribution and the sensor array pose via gradient descent. To ensure robust convergence in freehand scenarios, we introduce a coarse-to-fine optimization strategy that incorporates geometric consistency checks and rigid-body constraints to eliminate motion outliers. We validated the proposed method through both numerical simulations and in-vivo rat experiments. The results demonstrate that PA-SFM achieves sub-millimeter positioning accuracy and restores high-resolution 3D vascular structures comparable to ground-truth benchmarks, offering a low-cost, software-defined solution for clinical freehand photoacoustic imaging. The source code is publicly available at \href{this https URL}{this https URL}.
>
---
#### [new 026] FlowCoMotion: Text-to-Motion Generation via Token-Latent Flow Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到动作生成任务，旨在解决连续与离散运动表示的不足。提出FlowCoMotion框架，结合语义与细节，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.11083](https://arxiv.org/pdf/2604.11083)**

> **作者:** Dawei Guan; Di Yang; Chengjie Jin; Jiangtao Wang
>
> **备注:** 23 pages, 14 figures
>
> **摘要:** Text-to-motion generation is driven by learning motion representations for semantic alignment with language. Existing methods rely on either continuous or discrete motion representations. However, continuous representations entangle semantics with dynamics, while discrete representations lose fine-grained motion details. In this context, we propose FlowCoMotion, a novel motion generation framework that unifies both treatments from a modeling perspective. Specifically, FlowCoMotion employs token-latent coupling to capture both semantic content and high-fidelity motion details. In the latent branch, we apply multi-view distillation to regularize the continuous latent space, while in the token branch we use discrete temporal resolution quantization to extract high-level semantic cues. The motion latent is then obtained by combining the representations from the two branches through a token-latent coupling network. Subsequently, a velocity field is predicted based on the textual conditions. An ODE solver integrates this velocity field from a simple prior, thereby guiding the sample to the potential state of the target motion. Extensive experiments show that FlowCoMotion achieves competitive performance on text-to-motion benchmarks, including HumanML3D and SnapMoGen.
>
---
#### [new 027] Budget-Aware Uncertainty for Radiotherapy Segmentation QA Using nnU-Net
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于放射治疗分割质量保证任务，旨在解决自动分割模型可靠性问题。通过结合不确定性量化与校准，生成 voxel 级别不确定性图，指导人工审核。**

- **链接: [https://arxiv.org/pdf/2604.11798](https://arxiv.org/pdf/2604.11798)**

> **作者:** Ricardo Coimbra Brioso; Lorenzo Mondo; Damiano Dei; Nicola Lambri; Pietro Mancosu; Marta Scorsetti; Daniele Loiacono
>
> **摘要:** Accurate delineation of the Clinical Target Volume (CTV) is essential for radiotherapy planning, yet remains time-consuming and difficult to assess, especially for complex treatments such as Total Marrow and Lymph Node Irradiation (TMLI). While deep learning-based auto-segmentation can reduce workload, safe clinical deployment requires reliable cues indicating where models may be wrong. In this work, we propose a budget-aware uncertainty-driven quality assurance (QA) framework built on nnU-Net, combining uncertainty quantification and post-hoc calibration to produce voxel-wise uncertainty maps (based on predictive entropy) that can guide targeted manual review. We compare temperature scaling (TS), deep ensembles (DE), checkpoint ensembles (CE), and test-time augmentation (TTA), evaluated both individually and in combination on TMLI as a representative use case. Reliability is assessed through ROI-masked calibration metrics and uncertainty--error alignment under realistic revision constraints, summarized as AUC over the top 0-5% most uncertain voxels. Across configurations, segmentation accuracy remains stable, whereas TS substantially improves calibration. Uncertainty-error alignment improves most with calibrated checkpoint-based inference, leading to uncertainty maps that highlight more consistently regions requiring manual edits. Overall, integrating calibration with efficient ensembling seems a promising strategy to implement a budget-aware QA workflow for radiotherapy segmentation.
>
---
#### [new 028] Neural Stochastic Processes for Satellite Precipitation Refinement
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于降水估算任务，解决卫星数据偏差与地面观测稀疏的问题。提出Neural Stochastic Process模型，融合两者以提高精度。**

- **链接: [https://arxiv.org/pdf/2604.10414](https://arxiv.org/pdf/2604.10414)**

> **作者:** Shunya Nagashima; Takumi Bannai; Shuitsu Koyama; Tomoya Mitsui; Shuntaro Suzuki
>
> **摘要:** Accurate precipitation estimation is critical for flood forecasting, water resource management, and disaster preparedness. Satellite products provide global hourly coverage but contain systematic biases; ground-based gauges are accurate at point locations but too sparse for direct gridded correction. Existing methods fuse these sources by interpolating gauge observations onto the satellite grid, but treat each time step independently and therefore discard temporal structure in precipitation fields. We propose Neural Stochastic Process (NSP), a model that pairs a Neural Process encoder conditioning on arbitrary sets of gauge observations with a latent Neural SDE on a 2D spatial representation. NSP is trained under a single variational objective with simulation-free cost. We also introduce QPEBench, a benchmark of 43{,}756 hourly samples over the Contiguous United States (2021--2025) with four aligned data sources and six evaluation metrics. On QPEBench, NSP outperforms 13 baselines across all six metrics and surpasses JAXA's operational gauge-calibrated product. An additional experiment on Kyushu, Japan confirms generalization to a different region with independent data sources.
>
---
#### [new 029] Unfolding 3D Gaussian Splatting via Iterative Gaussian Synopsis
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决3DGS存储大、结构不规则的问题。通过迭代生成多级LOD，提升效率并减少冗余，实现高效渲染。**

- **链接: [https://arxiv.org/pdf/2604.11685](https://arxiv.org/pdf/2604.11685)**

> **作者:** Yuqin Lu; Yang Zhou; Yihua Dai; Guiqing Li; Shengfeng He
>
> **摘要:** 3D Gaussian Splatting (3DGS) has become a state-of-the-art framework for real-time, high-fidelity novel view synthesis. However, its substantial storage requirements and inherently unstructured representation pose challenges for deployment in streaming and resource-constrained environments. Existing Level-of-Detail (LOD) strategies, particularly those based on bottom-up construction, often introduce redundancy or lead to fidelity degradation. To overcome these limitations, we propose Iterative Gaussian Synopsis, a novel framework for compact and progressive rendering through a top-down "unfolding" scheme. Our approach begins with a full-resolution 3DGS model and iteratively derives coarser LODs using an adaptive, learnable mask-based pruning mechanism. This process constructs a multi-level hierarchy that preserves visual quality while improving efficiency. We integrate hierarchical spatial grids, which capture the global scene structure, with a shared Anchor Codebook that models localized details. This combination produces a compact yet expressive feature representation, designed to minimize redundancy and support efficient, level-specific adaptation. The unfolding mechanism promotes inter-layer reusability and requires only minimal data overhead for progressive refinement. Experiments show that our method maintains high rendering quality across all LODs while achieving substantial storage reduction. These results demonstrate the practicality and scalability of our approach for real-time 3DGS rendering in bandwidth- and memory-constrained scenarios.
>
---
#### [new 030] Revisiting the Scale Loss Function and Gaussian-Shape Convolution for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，解决训练不稳定和空间注意力不足的问题。提出基于差异的尺度损失和高斯卷积，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.09991](https://arxiv.org/pdf/2604.09991)**

> **作者:** Hao Li; Man Fung Zhuo
>
> **摘要:** Infrared small target detection still faces two persistent challenges: training instability from non-monotonic scale loss functions, and inadequate spatial attention due to generic convolution kernels that ignore the physical imaging characteristics of small targets. In this paper, we revisit both aspects. For the loss side, we propose a \emph{diff-based scale loss} that weights predictions according to the signed area difference between the predicted mask and the ground truth, yielding strictly monotonic gradients and stable convergence. We further analyze a family of four scale loss variants to understand how their geometric properties affect detection behavior. For the spatial side, we introduce \emph{Gaussian-shaped convolution} with a learnable scale parameter to match the center-concentrated intensity profile of infrared small targets, and augment it with a \emph{rotated pinwheel mask} that adaptively aligns the kernel with target orientation via a straight-through estimator. Extensive experiments on IRSTD-1k, NUDT-SIRST, and SIRST-UAVB demonstrate consistent improvements in $mIoU$, $P_d$, and $F_a$ over state-of-the-art methods. We release our anonymous code and pretrained models.
>
---
#### [new 031] Panoptic Pairwise Distortion Graph
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出一种新的图像对比评估方法，通过构建结构化区域图来表示图像对的退化信息。任务是解决细粒度的图像配对评估问题，工作包括数据集、基准测试和模型设计。**

- **链接: [https://arxiv.org/pdf/2604.11004](https://arxiv.org/pdf/2604.11004)**

> **作者:** Muhammad Kamran Janjua; Abdul Wahab; Bahador Rashidi
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** In this work, we introduce a new perspective on comparative image assessment by representing an image pair as a structured composition of its regions. In contrast, existing methods focus on whole image analysis, while implicitly relying on region-level understanding. We extend the intra-image notion of a scene graph to inter-image, and propose a novel task of Distortion Graph (DG). DG treats paired images as a structured topology grounded in regions, and represents dense degradation information such as distortion type, severity, comparison and quality score in a compact interpretable graph structure. To realize the task of learning a distortion graph, we contribute (i) a region-level dataset, PandaSet, (ii) a benchmark suite, PandaBench, with varying region-level difficulty, and (iii) an efficient architecture, Panda, to generate distortion graphs. We demonstrate that PandaBench poses a significant challenge for state-of-the-art multimodal large language models (MLLMs) as they fail to understand region-level degradations even when fed with explicit region cues. We show that training on PandaSet or prompting with DG elicits region-wise distortion understanding, opening a new direction for fine-grained, structured pairwise image assessment.
>
---
#### [new 032] MedP-CLIP: Medical CLIP with Region-Aware Prompt Integration
- **分类: cs.CV**

- **简介: 该论文提出MedP-CLIP，解决医学图像中细粒度区域理解问题。通过整合医学先验知识和区域提示机制，提升模型在医学任务中的表现。**

- **链接: [https://arxiv.org/pdf/2604.11197](https://arxiv.org/pdf/2604.11197)**

> **作者:** Jiahui Peng; He Yao; Jingwen Li; Yanzhou Su; Sibo Ju; Yujie Lu; Jin Ye; Hongchun Lu; Xue Li; Lincheng Jiang; Min Zhu; Junlong Cheng
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) has demonstrated outstanding performance in global image understanding and zero-shot transfer through large-scale text-image alignment. However, the core of medical image analysis often lies in the fine-grained understanding of specific anatomical structures or lesion regions. Therefore, precisely comprehending region-of-interest (RoI) information provided by medical professionals or perception models becomes crucial. To address this need, we propose MedP-CLIP, a region-aware medical vision-language model (VLM). MedP-CLIP innovatively integrates medical prior knowledge and designs a feature-level region prompt integration mechanism, enabling it to flexibly respond to various prompt forms (e.g., points, bounding boxes, masks) while maintaining global contextual awareness when focusing on local regions. We pre-train the model on a meticulously constructed large-scale dataset (containing over 6.4 million medical images and 97.3 million region-level annotations), equipping it with cross-disease and cross-modality fine-grained spatial semantic understanding capabilities. Experiments demonstrate that MedP-CLIP significantly outperforms baseline methods in various medical tasks, including zero-shot recognition, interactive segmentation, and empowering multimodal large language models. This model provides a scalable, plug-and-play visual backbone for medical AI, combining holistic image understanding with precise regional analysis.
>
---
#### [new 033] The Deployment Gap in AI Media Detection: Platform-Aware and Visually Constrained Adversarial Evaluation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI媒体检测任务，解决实验室评估与实际部署间的性能差距问题。通过构建平台感知的对抗评估框架，模拟真实部署中的图像变换，揭示检测器在真实环境下的脆弱性。**

- **链接: [https://arxiv.org/pdf/2604.09706](https://arxiv.org/pdf/2604.09706)**

> **作者:** Aishwarya Budhkar; Trishita Dhara; Siddhesh Sheth
>
> **备注:** Accepted at CVPR AIMS 2026
>
> **摘要:** Recent AI media detectors report near-perfect performance under clean laboratory evaluation, yet their robustness under realistic deployment conditions remains underexplored. In practice, AI-generated images are resized, compressed, re-encoded, and visually modified before being shared on online platforms. We argue that this creates a deployment gap between laboratory robustness and real-world reliability. In this work, we introduce a platform-aware adversarial evaluation framework for AI media detection that explicitly models deployment transforms (e.g., resizing, compression, screenshot-style distortions) and constrains perturbations to visually plausible meme-style bands rather than full-image noise. Under this threat model, detectors achieving AUC $\approx$ 0{.}99 in clean settings experience substantial degradation. Per-image platform-aware attacks reduce AUC to significantly lower levels and achieve high fake-to-real misclassification rates, despite strict visual constraints. We further demonstrate that universal perturbations exist even under localized band constraints, revealing shared vulnerability directions across inputs. Beyond accuracy degradation, we observe pronounced calibration collapse under attack, where detectors become confidently incorrect. Our findings highlight that robustness measured under clean conditions substantially overestimates deployment robustness. We advocate for platform-aware evaluation as a necessary component of future AI media security benchmarks and release our evaluation framework to facilitate standardized robustness assessment.
>
---
#### [new 034] Variational Latent Entropy Estimation Disentanglement: Controlled Attribute Leakage for Face Recognition
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在解决嵌入向量中非身份信息（如性别、种族）的泄露问题。提出VLEED方法，通过变分自编码器实现属性解耦，提升隐私与公平性。**

- **链接: [https://arxiv.org/pdf/2604.11250](https://arxiv.org/pdf/2604.11250)**

> **作者:** Ünsal Öztürk; Vedrana Krivokuća Hahn; Sushil Bhattacharjee; Sébastien Marcel
>
> **备注:** Submitted to IEEE Transactions on Information Forensics and Security (TIFS). 13 pages, 5 figures, 4 tables
>
> **摘要:** Face recognition embeddings encode identity, but they also encode other factors such as gender and ethnicity. Depending on how these factors are used by a downstream system, separating them from the information needed for verification is important for both privacy and fairness. We propose Variational Latent Entropy Estimation Disentanglement (VLEED), a post-hoc method that transforms pretrained embeddings with a variational autoencoder and encourages a distilled representation where the categorical variable of interest is separated from identity-relevant information. VLEED uses a mutual information-based objective realised through the estimation of the entropy of the categorical attribute in the latent space, and provides stable training with fine-grained control over information removal. We evaluate our method on IJB-C, RFW, and VGGFace2 for gender and ethnicity disentanglement, and compare it to various state-of-the-art methods. We report verification utility, predictability of the disentangled variable under linear and nonlinear classifiers, and group disparity metrics based on false match rates. Our results show that VLEED offers a wide range of privacy-utility tradeoffs over existing methods and can also reduce recognition bias across demographic groups.
>
---
#### [new 035] CDPR: Cross-modal Diffusion with Polarization for Reliable Monocular Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决复杂场景下深度估计可靠性问题。通过引入偏振信息与扩散模型结合，提升估计鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.11097](https://arxiv.org/pdf/2604.11097)**

> **作者:** Rongjia Yu; Tong Jia; Hao Wang; Xiaofang Li; Xiao Yang; Zinuo Zhang; Cuiwei Liu
>
> **备注:** preprint version of IEEE TMM 2026 Regular Paper
>
> **摘要:** Monocular depth estimation is a fundamental yet challenging task in computer vision, especially under complex conditions such as textureless surfaces, transparency, and specular reflections. Recent diffusion-based approaches have significantly advanced performance by reformulating depth prediction as a denoising process in the latent space. However, existing methods rely solely on RGB inputs, which often lack sufficient cues in challenging regions. In this work, we present CDPR - Cross-modal Diffusion with Polarization for Reliable Monocular Depth Estimation - a novel diffusion-based framework that integrates physically grounded polarization priors to enhance estimation robustness. Specifically, we encode both RGB and polarization (AoLP/DoLP) images into a shared latent space via a pre-trained Variational Autoencoder (VAE), and dynamically fuse multi-modal information through a learnable confidence-aware gating mechanism. This fusion module adaptively suppresses noisy signals in polarization inputs while preserving informative cues, particularly around reflective or transparent surfaces, and provides the integrated latent representation for subsequent monocular depth estimation. Beyond depth estimation, we further verify that our framework can be easily generalized to surface normal prediction with minimal modification, showcasing its scalability to general polarization-guided dense prediction tasks. Experiments on both synthetic and real-world datasets validate that CDPR significantly outperforms RGB-only baselines in challenging regions while maintaining competitive performance in standard scenes.
>
---
#### [new 036] FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FashionMV，解决产品级图像检索问题，通过多视角数据和模型机制提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.10297](https://arxiv.org/pdf/2604.10297)**

> **作者:** Peng Yuan; Bingyin Mei; Hui Zhang
>
> **摘要:** Composed Image Retrieval (CIR) retrieves target images using a reference image paired with modification text. Despite rapid advances, all existing methods and datasets operate at the image level -- a single reference image plus modification text in, a single target image out -- while real e-commerce users reason about products shown from multiple viewpoints. We term this mismatch View Incompleteness and formally define a new Multi-View CIR task that generalizes standard CIR from image-level to product-level retrieval. To support this task, we construct FashionMV, the first large-scale multi-view fashion dataset for product-level CIR, comprising 127K products, 472K multi-view images, and over 220K CIR triplets, built through a fully automated pipeline leveraging large multimodal models. We further propose ProCIR (Product-level Composed Image Retrieval), a modeling framework built upon a multimodal large language model that employs three complementary mechanisms -- two-stage dialogue, caption-based alignment, and chain-of-thought guidance -- together with an optional supervised fine-tuning (SFT) stage that injects structured product knowledge prior to contrastive training. Systematic ablation across 16 configurations on three fashion benchmarks reveals that: (1) alignment is the single most critical mechanism; (2) the two-stage dialogue architecture is a prerequisite for effective alignment; and (3) SFT and chain-of-thought serve as partially redundant knowledge injection paths. Our best 0.8B-parameter model outperforms all baselines, including general-purpose embedding models 10x its size. The dataset, model, and code are publicly available at this https URL.
>
---
#### [new 037] Global monitoring of methane point sources using deep learning on hyperspectral radiance measurements from EMIT
- **分类: cs.CV; cs.LG; physics.ao-ph**

- **简介: 该论文属于 methane point source monitoring 任务，旨在解决人工识别效率低的问题。通过深度学习模型 MAPL-EMIT，实现高效、自动化的甲烷泄漏检测与定位。**

- **链接: [https://arxiv.org/pdf/2604.10094](https://arxiv.org/pdf/2604.10094)**

> **作者:** Vishal V. Batchu; Michelangelo Conserva; Alex Wilson; Anna M. Michalak; Varun Gulshan; Philip G. Brodrick; Andrew K. Thorpe; Christopher V. Arsdale
>
> **备注:** 43 pages, 27 figures, 4 tables
>
> **摘要:** Anthropogenic methane (CH4) point sources drive near-term climate forcing, safety hazards, and system inefficiencies. Space-based imaging spectroscopy is emerging as a tool for identifying emissions globally, but existing approaches largely rely on manual plume identification. Here we present the Methane Analysis and Plume Localization with EMIT (MAPL-EMIT) model, an end-to-end vision transformer framework that leverages the complete radiance spectrum from the Earth Surface Mineral Dust Source Investigation (EMIT) instrument to jointly retrieve methane enhancements across all pixels within a scene. This approach brings together spectral and spatial context to significantly lower detection limits. MAPL-EMIT simultaneously supports enhancement quantification, plume delineation, and source localization, even for multiple overlapping plumes. The model was trained on 3.6 million physics-based synthetic plumes injected into global EMIT radiance data. Synthetic evaluation confirms the model's ability to identify plumes with high recall and precision and to capture weaker plumes relative to existing matched-filter approaches. On real-world benchmarks, MAPL-EMIT captures 79% of known hand-annotated NASA L2B plume complexes across a test set of 1084 EMIT granules, while capturing twice as many plausible plumes than identified by human analysts. Further validation against coincident airborne data, top-emitting landfills, and controlled release experiments confirms the model's ability to identify previously uncaptured sources. By incorporating model-generated metrics such as spectral fit scores and estimated noise levels, the framework can further limit false-positive rates. Overall, MAPL-EMIT enables high-throughput implementation on the full EMIT catalog, shifting methane monitoring from labor-intensive workflows to a rapid, scalable paradigm for global plume mapping at the facility scale.
>
---
#### [new 038] Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions
- **分类: cs.CV**

- **简介: 该论文属于视觉-触觉定位任务，旨在通过触觉输入定位图像中相同材质的区域。针对现有方法无法捕捉局部对应关系的问题，提出一种基于密集跨模态特征交互的模型，并构建新数据集提升性能。**

- **链接: [https://arxiv.org/pdf/2604.11579](https://arxiv.org/pdf/2604.11579)**

> **作者:** Seongyu Kim; Seungwoo Lee; Hyeonggon Ryu; Joon Son Chung; Arda Senocak
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** We address the problem of tactile localization, where the goal is to identify image regions that share the same material properties as a tactile input. Existing visuo-tactile methods rely on global alignment and thus fail to capture the fine-grained local correspondences required for this task. The challenge is amplified by existing datasets, which predominantly contain close-up, low-diversity images. We propose a model that learns local visuo-tactile alignment via dense cross-modal feature interactions, producing tactile saliency maps for touch-conditioned material segmentation. To overcome dataset constraints, we introduce: (i) in-the-wild multi-material scene images that expand visual diversity, and (ii) a material-diversity pairing strategy that aligns each tactile sample with visually varied yet tactilely consistent images, improving contextual localization and robustness to weak signals. We also construct two new tactile-grounded material segmentation datasets for quantitative evaluation. Experiments on both new and existing benchmarks show that our approach substantially outperforms prior visuo-tactile methods in tactile localization.
>
---
#### [new 039] LogitDynamics: Reliable ViT Error Detection from Layerwise Logit Trajectories
- **分类: cs.CV**

- **简介: 该论文属于视觉模型错误检测任务，旨在通过层间logit轨迹预测分类错误。工作包括设计轻量级头部提取特征，并用线性探针预测错误。**

- **链接: [https://arxiv.org/pdf/2604.10643](https://arxiv.org/pdf/2604.10643)**

> **作者:** Ido Beigelman; Moti Freiman
>
> **备注:** Accepted to the HOW 2026 workshop at CVPR 2026; 7 pages, 3 figures
>
> **摘要:** Reliable confidence estimation is critical when deploying vision models. We study error prediction: determining whether an image classifier's output is correct using only signals from a single forward pass. Motivated by internal-signal hallucination detection in large language models, we investigate whether similar depth-wise signals exist in Vision Transformers (ViTs). We propose a simple method that models how class evidence evolves across layers. By attaching lightweight linear heads to intermediate layers, we extract features from the last L layers that capture both the logits of the predicted class and its top-K competitors, as well as statistics describing instability of top-ranked classes across depth. A linear probe trained on these features predicts the error indicator. Across datasets, our method improves or matches AUCPR over baselines and shows stronger cross-dataset generalization while requiring minimal additional computation.
>
---
#### [new 040] ReplicateAnyScene: Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment
- **分类: cs.CV**

- **简介: 该论文提出ReplicateAnyScene，解决视频到3D场景的零样本生成问题，通过文本-视觉-空间对齐实现自动化3D场景构建。**

- **链接: [https://arxiv.org/pdf/2604.10789](https://arxiv.org/pdf/2604.10789)**

> **作者:** Mingyu Dong; Chong Xia; Mingyuan Jia; Weichen Lyu; Long Xu; Zheng Zhu; Yueqi Duan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Humans exhibit an innate capacity to rapidly perceive and segment objects from video observations, and even mentally assemble them into structured 3D scenes. Replicating such capability, termed compositional 3D reconstruction, is pivotal for the advancement of Spatial Intelligence and Embodied AI. However, existing methods struggle to achieve practical deployment due to the insufficient integration of cross-modal information, leaving them dependent on manual object prompting, reliant on auxiliary visual inputs, and restricted to overly simplistic scenes by training biases. To address these limitations, we propose ReplicateAnyScene, a framework capable of fully automated and zero-shot transformation of casually captured videos into compositional 3D scenes. Specifically, our pipeline incorporates a five-stage cascade to extract and structurally align generic priors from vision foundation models across textual, visual, and spatial dimensions, grounding them into structured 3D representations and ensuring semantic coherence and physical plausibility of the constructed scenes. To facilitate a more comprehensive evaluation of this task, we further introduce the C3DR benchmark to assess reconstruction quality from diverse aspects. Extensive experiments demonstrate the superiority of our method over existing baselines in generating high-quality compositional 3D scenes.
>
---
#### [new 041] Pseudo-Unification: Entropy Probing Reveals Divergent Information Patterns in Unified Multimodal Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态模型研究任务，旨在解决统一模型中信息不一致的问题。通过信息理论方法分析模型内部机制，揭示伪统一现象的根源，并提出改进方案。**

- **链接: [https://arxiv.org/pdf/2604.10949](https://arxiv.org/pdf/2604.10949)**

> **作者:** Songlin Yang; Xianghao Kong; Anyi Rao
>
> **摘要:** Unified multimodal models (UMMs) were designed to combine the reasoning ability of large language models (LLMs) with the generation capability of vision models. In practice, however, this synergy remains elusive: UMMs fail to transfer LLM-like reasoning to image synthesis and exhibit divergent response behaviors. We term this phenomenon pseudo-unification. Diagnosing its internal causes is important, but existing probing methods either lack model-internal insight or ignore prompt-response dependencies. To address these limitations, we propose an information-theoretic probing framework that jointly analyzes how UMMs encode inputs and generate outputs. Applied to ten representative UMMs, our framework reveals that pseudo-unification stems from a dual divergence: (i) Modality-Asymmetric Encoding, where vision and language follow different entropy trajectories, and (ii) Pattern-Split Response, where text generation exhibits high-entropy creativity while image synthesis enforces low-entropy fidelity. Only models that unify both sides (e.g., via contextual prediction) achieve more genuine unification, enabling stronger reasoning-based text-to-image generation even with fewer parameters. Our work provides the first model-internal probing of unification, demonstrating that real multimodal synergy requires consistency in information flow, not just shared parameters.
>
---
#### [new 042] BareBones: Benchmarking Zero-Shot Geometric Comprehension in VLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的几何理解任务，旨在解决模型是否真正理解几何结构的问题。通过构建BareBones基准，测试模型在无RGB信息下的几何识别能力。**

- **链接: [https://arxiv.org/pdf/2604.10528](https://arxiv.org/pdf/2604.10528)**

> **作者:** Aaditya Baranwal; Vishal Yadav; Abhishek Rajora
>
> **备注:** Accepted at CVPR (13th FGVC Workshop) 2026
>
> **摘要:** While Vision-Language Models (VLMs) demonstrate remarkable zero-shot recognition capabilities across a diverse spectrum of multimodal tasks, it yet remains an open question whether these architectures genuinely comprehend geometric structure or merely exploit RGB textures and contextual priors as statistical shortcuts. Existing evaluations fail to isolate this mechanism, conflating semantic reasoning with texture mapping and relying on imprecise annotations that inadvertently leak environmental cues. To address this gap, we introduce \textbf{BareBones}, a zero-shot benchmark designed to stress-test pure geometric shape comprehension. We curate pixel-level silhouettes of geometrically distinct classes across six datasets: five established segmentation sources (ImageNet-S, DIS5K, ThinObject5K, PASCAL VOC, CUB-200) and our novel flagship collection, WTP-Bench, establishing a noise-free geometric taxonomy. WTP-Bench is an extreme, fine-grained visual puzzle that forces models to identify inter-class geometric concepts from boundary contours alone. Our evaluation of 26 state-of-the-art proprietary and open-weight VLMs (\eg, GPT-4.1, Gemini, Claude Sonnet 4.5, LLaVA) reveals a consistent, severe performance collapse under RGB deprivation, a phenomenon we term the \textit{Texture Bias Cliff}. By documenting universal structural blindspots, BareBones establishes a rigorous yardstick for genuine geometric grounding.
>
---
#### [new 043] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation
- **分类: cs.CV**

- **简介: 该论文属于生成任务，解决无法生成矢量动画的问题。提出LottieGPT框架，通过tokenize和自回归生成，实现从文本或图像生成可编辑的矢量动画。**

- **链接: [https://arxiv.org/pdf/2604.11792](https://arxiv.org/pdf/2604.11792)**

> **作者:** Junhao Chen; Kejun Gao; Yuehan Cui; Mingze Sun; Mingjin Chen; Shaohui Wang; Xiaoxiao Long; Fei Ma; Qi Tian; Ruqi Huang; Hao Zhao
>
> **备注:** Accepted by CVPR 2026. Project Page: this https URL
>
> **摘要:** Despite rapid progress in video generation, existing models are incapable of producing vector animation, a dominant and highly expressive form of multimedia on the Internet. Vector animations offer resolution-independence, compactness, semantic structure, and editable parametric motion representations, yet current generative models operate exclusively in raster space and thus cannot synthesize them. Meanwhile, recent advances in large multimodal models demonstrate strong capabilities in generating structured data such as slides, 3D meshes, LEGO sequences, and indoor layouts, suggesting that native vector animation generation may be achievable. In this work, we present the first framework for tokenizing and autoregressively generating vector animations. We adopt Lottie, a widely deployed JSON-based animation standard, and design a tailored Lottie Tokenizer that encodes layered geometric primitives, transforms, and keyframe-based motion into a compact and semantically aligned token sequence. To support large-scale training, we also construct LottieAnimation-660K, the largest and most diverse vector animation dataset to date, consisting of 660k real-world Lottie animation and 15M static Lottie image files curated from broad Internet sources. Building upon these components, we finetune Qwen-VL to create LottieGPT, a native multimodal model capable of generating coherent, editable vector animations directly from natural language or visual prompts. Experiments show that our tokenizer dramatically reduces sequence length while preserving structural fidelity, enabling effective autoregressive learning of dynamic vector content. LottieGPT exhibits strong generalization across diverse animation styles and outperforms previous state-of-the-art models on SVG generation (a special case of single-frame vector animation).
>
---
#### [new 044] Gait Recognition with Temporal Kolmogorov-Arnold Networks
- **分类: cs.CV**

- **简介: 该论文属于行为识别任务，旨在解决步态识别中序列敏感、噪声干扰和长期依赖建模困难的问题。提出TKAN模型，结合记忆机制提升识别性能。**

- **链接: [https://arxiv.org/pdf/2604.09990](https://arxiv.org/pdf/2604.09990)**

> **作者:** Mohammed Asad; Dinesh Kumar Vishwakarma
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Gait recognition is a biometric modality that identifies individuals from their characteristic walking patterns. Unlike conventional biometric traits, gait can be acquired at a distance and without active subject cooperation, making it suitable for surveillance and public safety applications. Nevertheless, silhouette-based temporal models remain sensitive to long sequences, observation noise, and appearance-related covariates. Recurrent architectures often struggle to preserve information from earlier frames and are inherently sequential to optimize, whereas transformer-based models typically require greater computational resources and larger training sets and may be sensitive to irregular sequence lengths and noisy inputs. These limitations reduce robustness under clothing variation, carrying conditions, and view changes, while also hindering the joint modeling of local gait cycles and longer-term motion trends. To address these challenges, we introduce a Temporal Kolmogorov-Arnold Network (TKAN) for gait recognition. The proposed model replaces fixed edge weights with learnable one-dimensional functions and incorporates a two-level memory mechanism consisting of short-term RKAN sublayers and a gated long-term pathway. This design enables efficient modeling of both cycle-level dynamics and broader temporal context while maintaining a compact backbone. Experiments on the CASIA-B dataset indicate that the proposed CNN+TKAN framework achieves strong recognition performance under the reported evaluation setting.
>
---
#### [new 045] Revisiting Compositionality in Dual-Encoder Vision-Language Models: The Role of Inference
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决双编码器模型在组合性任务上的性能瓶颈。通过引入局部对齐机制，提升模型的组合泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11496](https://arxiv.org/pdf/2604.11496)**

> **作者:** Imanol Miranda; Ander Salaberria; Eneko Agirre; Gorka Azkune
>
> **摘要:** Dual-encoder Vision-Language Models (VLMs) such as CLIP are often characterized as bag-of-words systems due to their poor performance on compositional benchmarks. We argue that this limitation may stem less from deficient representations than from the standard inference protocol based on global cosine similarity. First, through controlled diagnostic experiments, we show that explicitly enforcing fine-grained region-segment alignment at inference dramatically improves compositional performance without updating pretrained encoders. We then introduce a lightweight transformer that learns such alignments directly from frozen patch and token embeddings. Comparing against full fine-tuning and prior end-to-end compositional training methods, we find that although these approaches improve in-domain retrieval, their gains do not consistently transfer under distribution shift. In contrast, learning localized alignment over frozen representations matches full fine-tuning on in-domain retrieval while yielding substantial improvements on controlled out-of-domain compositional benchmarks. These results identify global embedding matching as a key bottleneck in dual-encoder VLMs and highlight the importance of alignment mechanisms for robust compositional generalization.
>
---
#### [new 046] SVD-Prune: Training-Free Token Pruning For Efficient Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决高计算成本下的token剪枝问题。提出SVD-Prune方法，通过奇异值分解选择关键token，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.11530](https://arxiv.org/pdf/2604.11530)**

> **作者:** Yvon Apedo; Martyna Poreba; Michal Szczepanski; Samia Bouchafa
>
> **摘要:** Vision-Language Models (VLM) have revolutionized multimodal learning by jointly processing visual and textual information. Yet, they face significant challenges due to the high computational and memory demands of processing long sequences of vision tokens. Many existing methods rely on local heuristics, such as attention scores or token norms. However, these criteria suffer from positional bias and information dispersion, limiting their ability to preserve essential content at high pruning ratios and leading to performance degradation on visually detailed images. To address these issues, we propose SVD-Prune, a trainingfree, plug-and-play token pruning method based on Singular Value Decomposition. It decomposes the vision token feature matrix and selects the top-K tokens using statistical leverage scores, ensuring only tokens contributing most to the dominant global variance are preserved. Experiments show that SVD-Prune consistently outperforms prior pruning methods under extreme vision token budgets, maintaining strong performance even with 32 and 16 vision tokens.
>
---
#### [new 047] MMRareBench: A Rare-Disease Multimodal and Multi-Image Medical Benchmark
- **分类: cs.CV**

- **简介: 该论文提出MMRareBench，首个针对罕见病的多模态多图像医学基准，解决罕见病临床任务评估不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.10755](https://arxiv.org/pdf/2604.10755)**

> **作者:** Junzhi Ning; Jiashi Lin; Yingying Fang; Wei Li; Jiyao Liu; Cheng Tang; Chenglong Ma; Wenhao Tang; Tianbin Li; Ziyan Huang; Guang Yang; Junjun He
>
> **摘要:** Multimodal large language models (MLLMs) have advanced clinical tasks for common conditions, but their performance on rare diseases remains largely untested. In rare-disease scenarios, clinicians often lack prior clinical knowledge, forcing them to rely strictly on case-level evidence for clinical judgments. Existing benchmarks predominantly evaluate common-condition, single-image settings, leaving multimodal and multi-image evidence integration under rare-disease data scarcity systematically unevaluated. We introduce MMRareBench, to our knowledge the first rare-disease benchmark jointly evaluating multimodal and multi-image clinical capability across four workflow-aligned tracks: diagnosis, treatment planning, cross-image evidence alignment, and examination suggestion. The benchmark comprises 1,756 question-answer pairs with 7,958 associated medical images curated from PMC case reports, with Orphanet-anchored ontology alignment, track-specific leakage control, evidence-grounded annotations, and a two-level evaluation protocol. A systematic evaluation of 23 MLLMs reveals fragmented capability profiles and universally low treatment-planning performance, with medical-domain models trailing general-purpose MLLMs substantially on multi-image tracks despite competitive diagnostic scores. These patterns are consistent with a capacity dilution effect: medical fine-tuning can narrow the diagnostic gap but may erode the compositional multi-image capability that rare-disease evidence integration demands.
>
---
#### [new 048] Observe Less, Understand More: Cost-aware Cross-scale Observation for Remote Sensing Understanding
- **分类: cs.CV**

- **简介: 该论文属于遥感理解任务，解决多尺度观测成本与效果的平衡问题。通过联合HR采样与跨块表示预测，提升任务性能同时降低成本。**

- **链接: [https://arxiv.org/pdf/2604.11415](https://arxiv.org/pdf/2604.11415)**

> **作者:** Zhenghao Xie; Jing Xiao; Zhenqi Wang; Kexin Ma; Liang Liao; Gui-Song Xia; Mi Wang
>
> **摘要:** Remote sensing understanding inherently requires multi-resolution observation, since different targets and application tasks demand different levels of spatial detail. While low-resolution (LR) imagery enables efficient global observation, high-resolution (HR) imagery provides critical local details at much higher acquisition cost and limited coverage. This motivates a cross-scale sensing strategy that selectively acquires HR imagery from LR-based global perception to improve task performance under constrained cost. Existing methods for HR sampling methods typically make selection decisions from isolated LR patches, which ignore fine-grained intra-patch importance and cross-patch contextual interactions, leading to fragmented feature representation and suboptimal scene reasoning under sparse HR observations. To address this issue, we formulate cross-scale remote sensing understanding as a unified cost-aware problem that couples fine-grained HR sampling with cross-patch representation prediction, enabling more effective task reasoning with fewer HR observations. Furthermore, we present GL-10M, a large-scale benchmark of 10 million spatially aligned multi-resolution images, enabling systematic evaluation of budget-constrained cross-scale reasoning in remote sensing. Extensive experiments on recognition and retrieval tasks show that our method consistently achieves a superior performance-cost trade-off.
>
---
#### [new 049] FF3R: Feedforward Feature 3D Reconstruction from Unconstrained views
- **分类: cs.CV**

- **简介: 该论文提出FF3R，解决多视角3D重建中的几何与语义统一问题，无需标注数据，通过创新模块提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.09862](https://arxiv.org/pdf/2604.09862)**

> **作者:** Chaoyi Zhou; Run Wang; Feng Luo; Mert D. Pesé; Zhiwen Fan; Yiqi Zhong; Siyu Huang
>
> **备注:** CVPR 2026 Findings. Project Page: this https URL
>
> **摘要:** Recent advances in vision foundation models have revolutionized geometry reconstruction and semantic understanding. Yet, most of the existing approaches treat these capabilities in isolation, leading to redundant pipelines and compounded errors. This paper introduces FF3R, a fully annotation-free feed-forward framework that unifies geometric and semantic reasoning from unconstrained multi-view image sequences. Unlike previous methods, FF3R does not require camera poses, depth maps, or semantic labels, relying solely on rendering supervision for RGB and feature maps, establishing a scalable paradigm for unified 3D reasoning. In addition, we address two critical challenges in feedforward feature reconstruction pipelines, namely global semantic inconsistency and local structural inconsistency, through two key innovations: (i) a Token-wise Fusion Module that enriches geometry tokens with semantic context via cross-attention, and (ii) a Semantic-Geometry Mutual Boosting mechanism combining geometry-guided feature warping for global consistency with semantic-aware voxelization for local coherence. Extensive experiments on ScanNet and DL3DV-10K demonstrate FF3R's superior performance in novel-view synthesis, open-vocabulary semantic segmentation, and depth estimation, with strong generalization to in-the-wild scenarios, paving the way for embodied intelligence systems that demand both spatial and semantic understanding.
>
---
#### [new 050] MedVeriSeg: Teaching MLLM-Based Medical Segmentation Models to Verify Query Validity Without Extra Training
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决MLLM模型对无效查询无法识别的问题。提出MedVeriSeg框架，通过分析特征相似性判断查询有效性，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2604.10242](https://arxiv.org/pdf/2604.10242)**

> **作者:** Ziqian Lu; Qinyue Tong; Jun Liu; Yunlong Yu
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Despite recent advances in MLLM-based medical image segmentation, existing LISA-like methods cannot reliably reject false queries and often produce hallucinated segmentation masks for absent targets. This limitation reduces practical reliability in both medical education and clinical use. In this work, we propose MedVeriSeg, a training-free verification framework that equips LISA-like medical segmentation models with the ability to identify and reject false queries which contain non-existent targets. Our key observation is that the similarity map between the [SEG] token feature and MLLM image features exhibits markedly different distribution patterns for true and false queries. Based on this, we introduce a Similarity Response Quality Scoring Module that characterizes the similarity map from three aspects: strength, compactness, and purity, producing an initial target-existence prediction. We further incorporate qualitative visual evidence by using GPT-4o to jointly assess the similarity heatmap and the results of Similarity Response Quality Scoring Module for final verification. Experiments on a small-scale benchmark constructed from SA-Med2D-20M show that MedVeriSeg effectively rejects false-query segmentation requests while maintaining reliable recognition of true queries.
>
---
#### [new 051] Long-Horizon Streaming Video Generation via Hybrid Attention with Decoupled Distillation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决长视频生成中注意力机制丢失远距离信息和计算开销大的问题。提出Hybrid Forcing方法，结合线性时间和块稀疏注意力，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2604.10103](https://arxiv.org/pdf/2604.10103)**

> **作者:** Ruibin Li; Tao Yang; Fangzhou Ai; Tianhe Wu; Shilei Wen; Bingyue Peng; Lei Zhang
>
> **摘要:** Streaming video generation (SVG) distills a pretrained bidirectional video diffusion model into an autoregressive model equipped with sliding window attention (SWA). However, SWA inevitably loses distant history during long video generation, and its computational overhead remains a critical challenge to real-time deployment. In this work, we propose Hybrid Forcing, which jointly optimizes temporal information retention and computational efficiency through a hybrid attention design. First, we introduce lightweight linear temporal attention to preserve long-range dependencies beyond the sliding window. In particular, we maintain a compact key-value state to incrementally absorb evicted tokens, retaining temporal context with negligible memory and computational overhead. Second, we incorporate block-sparse attention into the local sliding window to reduce redundant computation within short-range modeling, reallocating computational capacity toward more critical dependencies. Finally, we introduce a decoupled distillation strategy tailored to the hybrid attention design. A few-step initial distillation is performed under dense attention, then the distillation of our proposed linear temporal and block-sparse attention is activated for streaming modeling, ensuring stable optimization. Extensive experiments on both short- and long-form video generation benchmarks demonstrate that Hybrid Forcing consistently achieves state-of-the-art performance. Notably, our model achieves real-time, unbounded 832x480 video generation at 29.5 FPS on a single NVIDIA H100 GPU without quantization or model compression. The source code and trained models are available at this https URL.
>
---
#### [new 052] BoxTuning: Directly Injecting the Object Box for Multimodal Model Fine-Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出BoxTuning，解决视频问答中对象信息表达效率低的问题。通过视觉提示直接注入目标框信息，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.11136](https://arxiv.org/pdf/2604.11136)**

> **作者:** Zekun Qian; Ruize Han; Wei Feng
>
> **摘要:** Object-level spatial-temporal understanding is essential for video question answering, yet existing multimodal large language models (MLLMs) encode frames holistically and lack explicit mechanisms for fine-grained object grounding. Recent work addresses this by serializing bounding box coordinates as text tokens, but this text-coordinate paradigm suffers from a fundamental modality mismatch: object information is inherently visual, yet encoding it as text incurs a high token cost that forces aggressive temporal downsampling. We propose BoxTuning, which resolves this mismatch by injecting object spatial-temporal information directly into the visual modality. Colored bounding boxes and trajectory trails are rendered onto video frames as visual prompts, with only a concise color-to-object legend retained as text. This reduces the token cost significantly, achieving 87-93% text token reduction in practice. It also preserves full temporal resolution, where the trajectory trails further encode inter-frame motion direction and speed within each keyframe, recovering fine-grained dynamics that text-coordinate methods are forced to discard. Experimental results on five video QA benchmarks (CLEVRER, Perception Test, STAR, NExT-QA, IntentQA) show that BoxTuning surpasses text-coordinate baselines on spatially oriented tasks and nearly eliminates the accuracy degradation observed on reasoning-centric tasks, establishing visual prompting as a more natural and efficient paradigm for conveying object information to video MLLMs.
>
---
#### [new 053] Zero-Shot Synthetic-to-Real Handwritten Text Recognition via Task Analogies
- **分类: cs.CV**

- **简介: 该论文属于手写文本识别任务，解决合成数据到真实数据的零样本泛化问题。通过学习源语言中参数变化模式，并将其迁移至目标语言，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.09713](https://arxiv.org/pdf/2604.09713)**

> **作者:** Carlos Garrido-Munoz; Aniello Panariello; Silvia Cascianelli; Angelo Porrello; Simone Calderara; Jorge Calvo-Zaragoza; Rita Cucchiara
>
> **摘要:** Handwritten Text Recognition (HTR) models trained on synthetic handwriting often struggle to generalize to real text, and existing adaptation methods still require real samples from the target domain. In this work, we tackle the fully zero-shot synthetic-to-real generalization setting, where no real data from the target language is available. Our approach learns how model parameters change when moving from synthetic to real handwriting in one or more source languages and transfers this learned correction to new target languages. When using multiple sources, we rely on linguistic similarity to weigh their contrubition when combining them. Experiments across five languages and six architectures show consistent improvements over synthetic-only baselines and reveal that the transferred corrections benefit even languages unrelated to the sources.
>
---
#### [new 054] Do Thought Streams Matter? Evaluating Reasoning in Gemini Vision-Language Models for Video Scene Understanding
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型在视频场景理解中的推理过程，评估内部思考流对输出的影响，旨在提升模型的推理质量和效率。**

- **链接: [https://arxiv.org/pdf/2604.11177](https://arxiv.org/pdf/2604.11177)**

> **作者:** Shivam Sharma; Sankalp Nagaonkar; Ashish Choithani; Ashutosh Trivedi
>
> **摘要:** We benchmark how internal reasoning traces, which we call thought streams, affect video scene understanding in vision-language models. Using four configurations of Google's Gemini 2.5 Flash and Flash Lite across scenes extracted from 100 hours of video, we ask three questions: does more thinking lead to better outputs, where do the gains stop, and what do these models actually think about? We introduce three evaluation metrics. Contentfulness measures how much of the thought stream is useful scene content versus meta-commentary. Thought-Final Coverage measures how faithfully the thought stream translates into the final output. Dominant Entity Analysis identifies which subjects, actions, and settings the model focuses on. GPT-5 serves as an independent judge. We find that quality gains from additional thinking plateau quickly, with most improvement occurring in the first few hundred tokens. Flash Lite offers the best balance between quality and token usage. Tight reasoning budgets cause the model to add content in the final output that it never reasoned about, a form of compression-step hallucination. Despite being different model tiers, Flash and Flash Lite produce similar thought streams, though they differ in style: Flash discusses its reasoning process, while Lite focuses on describing the scene.
>
---
#### [new 055] U$^{2}$Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation
- **分类: cs.CV**

- **简介: 该论文属于光学流估计任务，旨在解决无监督方法中缺乏可靠不确定性估计的问题。提出U$^{2}$Flow框架，联合估计光流与像素级不确定性，提升模型鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.10056](https://arxiv.org/pdf/2604.10056)**

> **作者:** Xunpei Sun; Wenwei Lin; Yi Chang; Gang Chen
>
> **备注:** Accepted as an oral presentation at CVPR 2026
>
> **摘要:** Unsupervised optical flow methods typically lack reliable uncertainty estimation, limiting their robustness and interpretability. We propose U$^{2}$Flow, the first recurrent unsupervised framework that jointly estimates optical flow and per-pixel uncertainty. The core innovation is a decoupled learning strategy that derives uncertainty supervision from augmentation consistency via a Laplace-based maximum likelihood objective, enabling stable training without ground truth. The predicted uncertainty is further integrated into the network to guide adaptive flow refinement and dynamically modulate the regional smoothness loss. Furthermore, we introduce an uncertainty-guided bidirectional flow fusion mechanism that enhances robustness in challenging regions. Extensive experiments on KITTI and Sintel demonstrate that U$^{2}$Flow achieves state-of-the-art performance among unsupervised methods while producing highly reliable uncertainty maps, validating the effectiveness of our joint estimation paradigm. The code is available at this https URL.
>
---
#### [new 056] HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出HOG-Layout，用于3D场景的生成、优化与编辑。解决手动创建耗时、数据驱动缺乏多样性的问题，通过大模型实现文本驱动的高效场景生成与实时编辑。**

- **链接: [https://arxiv.org/pdf/2604.10772](https://arxiv.org/pdf/2604.10772)**

> **作者:** Haiyan Jiang; Deyu Zhang; Dongdong Weng; Weitao Song; Henry Been-Lirn Duh
>
> **备注:** CVPR 2026
>
> **摘要:** 3D layout generation and editing play a crucial role in Embodied AI and immersive VR interaction. However, manual creation requires tedious labor, while data-driven generation often lacks diversity. The emergence of large models introduces new possibilities for 3D scene synthesis. We present HOG-Layout that enables text-driven hierarchical scene generation, optimization and real-time scene editing with large language models (LLMs) and vision-language models (VLMs). HOG-Layout improves scene semantic consistency and plausibility through retrieval-augmented generation (RAG) technology, incorporates an optimization module to enhance physical consistency, and adopts a hierarchical representation to enhance inference and optimization, achieving real-time editing. Experimental results demonstrate that HOG-Layout produces more reasonable environments compared with existing baselines, while supporting fast and intuitive scene editing.
>
---
#### [new 057] Bootstrapping Video Semantic Segmentation Model via Distillation-assisted Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视频语义分割任务，旨在解决依赖密集标注数据的问题。通过DiTTA框架，将图像分割模型转化为时序感知的视频分割模型，无需标注视频数据。**

- **链接: [https://arxiv.org/pdf/2604.10950](https://arxiv.org/pdf/2604.10950)**

> **作者:** Jihun Kim; Hoyong Kwon; Hyeokjun Kweon; Kuk-Jin Yoon
>
> **摘要:** Fully supervised Video Semantic Segmentation (VSS) relies heavily on densely annotated video data, limiting practical applicability. Alternatively, applying pre-trained Image Semantic Segmentation (ISS) models frame-by-frame avoids annotation costs but ignores crucial temporal coherence. Recent foundation models such as SAM2 enable high-quality mask propagation yet remain impractical for direct VSS due to limited semantic understanding and computational overhead. In this paper, we propose DiTTA (Distillation-assisted Test-Time Adaptation), a novel framework that converts an ISS model into a temporally-aware VSS model through efficient test-time adaptation (TTA), without annotated videos. DiTTA distills SAM2's temporal segmentation knowledge into the ISS model during a brief, single-pass initialization phase, complemented by a lightweight temporal fusion module to aggregate cross-frame context. Crucially, DiTTA achieves robust generalization even when adapting with highly limited partial video snippets (e.g., initial 10%), significantly outperforming zero-shot refinement approaches that repeatedly invoke SAM2 during inference. Extensive experiments on VSPW and Cityscapes demonstrate DiTTA's effectiveness, achieving competitive or superior performance relative to fully-supervised VSS methods, thus providing a practical and annotation-free solution for real-world VSS tasks.
>
---
#### [new 058] MapATM: Enhancing HD Map Construction through Actor Trajectory Modeling
- **分类: cs.CV**

- **简介: 该论文属于高精度地图构建任务，旨在提升车道检测准确性。通过引入车辆轨迹信息，提出MapATM模型，有效改善复杂环境下的地图可靠性。**

- **链接: [https://arxiv.org/pdf/2604.11081](https://arxiv.org/pdf/2604.11081)**

> **作者:** Mingyang Li; Brian Lee; Rui Zuo; Brent Bacchus; Priyantha Mudalige; Qinru Qiu
>
> **备注:** 6 pages, 4 figures, 5 tables
>
> **摘要:** High-definition (HD) mapping tasks, which perform lane detections and predictions, are extremely challenging due to non-ideal conditions such as view occlusions, distant lane visibility, and adverse weather conditions. Those conditions often result in compromised lane detection accuracy and reduced reliability within autonomous driving systems. To address these challenges, we introduce MapATM, a novel deep neural network that effectively leverages historical actor trajectory information to improve lane detection accuracy, where actors refer to moving vehicles. By utilizing actor trajectories as structural priors for road geometry, MapATM achieves substantial performance enhancements, notably increasing AP by 4.6 for lane dividers and mAP by 2.6 on the challenging NuScenes dataset, representing relative improvements of 10.1% and 6.1%, respectively, compared to strong baseline methods. Extensive qualitative evaluations further demonstrate MapATM's capability to consistently maintain stable and robust map reconstruction across diverse and complex driving scenarios, underscoring its practical value for autonomous driving applications.
>
---
#### [new 059] ArtiCAD: Articulated CAD Assembly Design via Multi-Agent Code Generation
- **分类: cs.CV**

- **简介: 该论文提出ArtiCAD，解决从文本或图像生成可编辑的关节式CAD装配的问题。通过多智能体协作，实现高效、高质量的装配设计。**

- **链接: [https://arxiv.org/pdf/2604.10992](https://arxiv.org/pdf/2604.10992)**

> **作者:** Yuan Shui; Yandong Guan; Zhanwei Zhang; Juncheng Hu; Jing Zhang; Dong Xu; Qian Yu
>
> **摘要:** Parametric Computer-Aided Design (CAD) of articulated assemblies is essential for product development, yet generating these multi-part, movable models from high-level descriptions remains unexplored. To address this, we propose ArtiCAD, the first training-free multi-agent system capable of generating editable, articulated CAD assemblies directly from text or images. Our system divides this complex task among four specialized agents: Design, Generation, Assembly, and Review. One of our key insights is to predict assembly relationships during the initial design stage rather than the assembly stage. By utilizing a Connector that explicitly defines attachment points and joint parameters, ArtiCAD determines these relationships before geometry generation, effectively bypassing the limited spatial reasoning capabilities of current LLMs and VLMs. To further ensure high-quality outputs, we introduce validation steps in the generation and assembly stages, accompanied by a cross-stage rollback mechanism that accurately isolates and corrects design- and code-level errors. Additionally, a self-evolving experience store accumulates design knowledge to continuously improve performance on future tasks. Extensive evaluations on three datasets (ArtiCAD-Bench, CADPrompt, and ACD) validate the effectiveness of our approach. We further demonstrate the applicability of ArtiCAD in requirement-driven conceptual design, physical prototyping, and the generation of embodied AI training assets through URDF export.
>
---
#### [new 060] Seg2Change: Adapting Open-Vocabulary Semantic Segmentation Model for Remote Sensing Change Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，旨在解决传统方法受限于预定义类别的问题。通过构建通用数据集和设计适配器，将开放词汇语义分割模型应用于任意类别变化检测。**

- **链接: [https://arxiv.org/pdf/2604.11231](https://arxiv.org/pdf/2604.11231)**

> **作者:** You Su; Yonghong Song; Jingqi Chen; Zehan Wen
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Change detection is a fundamental task in remote sensing, aiming to quantify the impacts of human activities and ecological dynamics on land-cover changes. Existing change detection methods are limited to predefined classes in training datasets, which constrains their scalability in real-world scenarios. In recent years, numerous advanced open-vocabulary semantic segmentation models have emerged for remote sensing imagery. However, there is still a lack of an effective framework for directly applying these models to open-vocabulary change detection (OVCD), a novel task that integrates vision and language to detect changes across arbitrary categories. To address these challenges, we first construct a category-agnostic change detection dataset, termed CA-CDD. Further, we design a category-agnostic change head to detect the transitions of arbitrary categories and index them to specific classes. Based on them, we propose Seg2Change, an adapter designed to adapt open-vocabulary semantic segmentation models to change detection task. Without bells and whistles, this simple yet effective framework achieves state-of-the-art OVCD performance (+9.52 IoU on WHU-CD and +5.50 mIoU on SECOND). Our code is released at this https URL.
>
---
#### [new 061] HDR Video Generation via Latent Alignment with Logarithmic Encoding
- **分类: cs.CV**

- **简介: 该论文属于HDR视频生成任务，解决生成模型与HDR数据不匹配的问题。通过 logarithmic 编码对齐预训练模型的潜在空间，实现高效HDR生成。**

- **链接: [https://arxiv.org/pdf/2604.11788](https://arxiv.org/pdf/2604.11788)**

> **作者:** Naomi Ken Korem; Mohamed Oumoumad; Harel Cain; Matan Ben Yosef; Urska Jelercic; Ofir Bibi; Yaron Inger; Or Patashnik; Daniel Cohen-Or
>
> **备注:** this https URL
>
> **摘要:** High dynamic range (HDR) imagery offers a rich and faithful representation of scene radiance, but remains challenging for generative models due to its mismatch with the bounded, perceptually compressed data on which these models are trained. A natural solution is to learn new representations for HDR, which introduces additional complexity and data requirements. In this work, we show that HDR generation can be achieved in a much simpler way by leveraging the strong visual priors already captured by pretrained generative models. We observe that a logarithmic encoding widely used in cinematic pipelines maps HDR imagery into a distribution that is naturally aligned with the latent space of these models, enabling direct adaptation via lightweight fine-tuning without retraining an encoder. To recover details that are not directly observable in the input, we further introduce a training strategy based on camera-mimicking degradations that encourages the model to infer missing high dynamic range content from its learned priors. Combining these insights, we demonstrate high-quality HDR video generation using a pretrained video model with minimal adaptation, achieving strong results across diverse scenes and challenging lighting conditions. Our results indicate that HDR, despite representing a fundamentally different image formation regime, can be handled effectively without redesigning generative models, provided that the representation is chosen to align with their learned priors.
>
---
#### [new 062] Product Review Based on Optimized Facial Expression Detection
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文属于产品评价任务，旨在通过分析顾客面部表情评估产品接受度。采用改进的Harris算法提取特征点，提升检测速度与准确性。**

- **链接: [https://arxiv.org/pdf/2604.10885](https://arxiv.org/pdf/2604.10885)**

> **作者:** Vikrant Chaugule; Abhishek D; Aadheeshwar Vijayakumar; Pravin Bhaskar Ramteke; Shashidhar G. Koolagudi
>
> **备注:** 9 pages, 11 figures, Published in the 2016 Ninth International Conference on Contemporary Computing (IC3), August 11-13, 2016, Noida, India. This is a pre-print version of the paper
>
> **摘要:** This paper proposes a method to review public acceptance of products based on their brand by analyzing the facial expression of the customer intending to buy the product from a supermarket or hypermarket. In such cases, facial expression recognition plays a significant role in product review. Here, facial expression detection is performed by extracting feature points using a modified Harris algorithm. The modified Harris algorithm reduced the time complexity of the existing feature extraction Harris Algorithm. A comparison of time complexities of existing algorithms is done with proposed algorithm. The algorithm proved to be significantly faster and nearly accurate for the needed application by reducing the time complexity for corner points detection.
>
---
#### [new 063] Representations Before Pixels: Semantics-Guided Hierarchical Video Prediction
- **分类: cs.CV**

- **简介: 该论文属于视频预测任务，旨在提升未来视频的视觉质量和语义一致性。提出Re2Pix框架，先预测语义表示再生成图像，解决训练与推理不匹配问题。**

- **链接: [https://arxiv.org/pdf/2604.11707](https://arxiv.org/pdf/2604.11707)**

> **作者:** Efstathios Karypidis; Spyros Gidaris; Nikos Komodakis
>
> **摘要:** Accurate future video prediction requires both high visual fidelity and consistent scene semantics, particularly in complex dynamic environments such as autonomous driving. We present Re2Pix, a hierarchical video prediction framework that decomposes forecasting into two stages: semantic representation prediction and representation-guided visual synthesis. Instead of directly predicting future RGB frames, our approach first forecasts future scene structure in the feature space of a frozen vision foundation model, and then conditions a latent diffusion model on these predicted representations to render photorealistic frames. This decomposition enables the model to focus first on scene dynamics and then on appearance generation. A key challenge arises from the train-test mismatch between ground-truth representations available during training and predicted ones used at inference. To address this, we introduce two conditioning strategies, nested dropout and mixed supervision, that improve robustness to imperfect autoregressive predictions. Experiments on challenging driving benchmarks demonstrate that the proposed semantics-first design significantly improves temporal semantic consistency, perceptual quality, and training efficiency compared to strong diffusion baselines. We provide the implementation code at this https URL
>
---
#### [new 064] Data-Efficient Semantic Segmentation of 3D Point Clouds via Open-Vocabulary Image Segmentation-based Pseudo-Labeling
- **分类: cs.CV**

- **简介: 该论文属于3D点云语义分割任务，解决数据不足问题。提出PLOVIS方法，利用开放词汇图像分割生成伪标签，提升模型在少量标注数据下的性能。**

- **链接: [https://arxiv.org/pdf/2604.11007](https://arxiv.org/pdf/2604.11007)**

> **作者:** Takahiko Furuya
>
> **摘要:** Semantic segmentation of 3D point cloud scenes is a crucial task for various applications. In real-world scenarios, training segmentation models often faces three concurrent forms of data insufficiency: scarcity of training scenes, scarcity of point-level annotations, and absence of 2D image sequences from which point clouds were reconstructed. Existing data-efficient algorithms typically address only one or two of these challenges, leaving the joint treatment of all three unexplored. This paper proposes a data-efficient training framework specifically designed to address the three forms of data insufficiency. Our proposed algorithm, called Point pseudo-Labeling via Open-Vocabulary Image Segmentation (PLOVIS), leverages an Open-Vocabulary Image Segmentation (OVIS) model as a pseudo label generator to compensate for the lack of training data. PLOVIS creates 2D images for pseudo-labeling directly from training 3D point clouds, eliminating the need for 2D image sequences. To mitigate the inherent noise and class imbalance in pseudo labels, we introduce a two-stage filtering of pseudo labels combined with a class-balanced memory bank for effective training. The two-stage filtering mechanism first removes low-confidence pseudo labels, then discards likely incorrect pseudo labels, thereby enhancing the quality of pseudo labels. Experiments on four benchmark datasets, i.e., ScanNet, S3DIS, Toronto3D, and Semantic3D, under realistic data-scarce conditions (a few tens of training 3D scenes, each annotated with only <100 3D points) demonstrate that PLOVIS consistently outperforms existing methods including standard fine-tuning strategies and state-of-the-art weakly supervised learning algorithms. Code will be made publicly available.
>
---
#### [new 065] Robust Fair Disease Diagnosis in CT Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像诊断任务，旨在解决CT影像中因数据偏差导致的诊断不公平问题。通过引入双层级损失函数，提升模型在不同性别群体中的公平性与准确率。**

- **链接: [https://arxiv.org/pdf/2604.09710](https://arxiv.org/pdf/2604.09710)**

> **作者:** Justin Li; Daniel Ding; Asmita Yuki Pritha; Aryana Hou; Xin Wang; Shu Hu
>
> **备注:** 8 pages, 3 figures, 2 tables. Accepted at the 3rd Workshop on New Trends in AI-Generated Media and Security (AIMS) @ CVPR 2026
>
> **摘要:** Automated diagnosis from chest CT has improved considerably with deep learning, but models trained on skewed datasets tend to perform unevenly across patient demographics. However, the situation is worse than simple demographic bias. In clinical data, class imbalance and group underrepresentation often coincide, creating compound failure modes that neither standard rebalancing nor fairness corrections can fix alone. We introduce a two-level objective that targets both axes of this problem. Logit-adjusted cross-entropy loss operates at the sample level, shifting decision margins by class frequency with provable consistency guarantees. Conditional Value at Risk aggregation operates at the group level, directing optimization pressure toward whichever demographic group currently has the higher loss. We evaluate on the Fair Disease Diagnosis benchmark using a 3D ResNet-18 pretrained on Kinetics-400, classifying CT volumes into Adenocarcinoma, Squamous Cell Carcinoma, COVID-19, and Normal groups with patient sex annotations. The training set illustrates the compound problem concretely: squamous cell carcinoma has 84 samples total, 5 of them female. The combined loss reaches a gender-averaged macro F1 of 0.8403 with a fairness gap of 0.0239, a 13.3% improvement in score and 78% reduction in demographic disparity over the baseline. Ablations show that each component alone falls short. The code is publicly available at this https URL.
>
---
#### [new 066] Training-Free Object-Background Compositional T2I via Dynamic Spatial Guidance and Multi-Path Pruning
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决背景与前景交互不足的问题。提出动态空间引导和多路径剪枝方法，提升背景一致性与构图控制。**

- **链接: [https://arxiv.org/pdf/2604.09850](https://arxiv.org/pdf/2604.09850)**

> **作者:** Yang Deng; David Mould; Paul L. Rosin; Yu-Kun Lai
>
> **摘要:** Existing text-to-image diffusion models, while excelling at subject synthesis, exhibit a persistent foreground bias that treats the background as a passive and under-optimized byproduct. This imbalance compromises global scene coherence and constrains compositional control. To address the limitation, we propose a training-free framework that restructures diffusion sampling to explicitly account for foreground-background interactions. Our approach consists of two key components. First, Dynamic Spatial Guidance introduces a soft, time step dependent gating mechanism that modulates foreground and background attention during the diffusion process, enabling spatially balanced generation. Second, Multi-Path Pruning performs multi-path latent exploration and dynamically filters candidate trajectories using both internal attention statistics and external semantic alignment signals, retaining trajectories that better satisfy object-background constraints. We further develop a benchmark specifically designed to evaluate object-background compositionality. Extensive evaluations across multiple diffusion backbones demonstrate consistent improvements in background coherence and object-background compositional alignment.
>
---
#### [new 067] SMFormer: Empowering Self-supervised Stereo Matching via Foundation Models and Data Augmentation
- **分类: cs.CV**

- **简介: 该论文属于立体匹配任务，旨在解决自监督方法因光照干扰导致的精度不足问题。通过引入视觉基础模型和数据增强，提升特征鲁棒性和预测一致性。**

- **链接: [https://arxiv.org/pdf/2604.10218](https://arxiv.org/pdf/2604.10218)**

> **作者:** Yun Wang; Zhengjie Yang; Jiahao Zheng; Zhanjie Zhang; Dapeng Oliver Wu; Yulan Guo
>
> **摘要:** Recent self-supervised stereo matching methods have made significant progress. They typically rely on the photometric consistency assumption, which presumes corresponding points across views share the same appearance. However, this assumption could be compromised by real-world disturbances, resulting in invalid supervisory signals and a significant accuracy gap compared to supervised methods. To address this issue, we propose SMFormer, a framework integrating more reliable self-supervision guided by the Vision Foundation Model (VFM) and data augmentation. We first incorporate the VFM with the Feature Pyramid Network (FPN), providing a discriminative and robust feature representation against disturbance in various scenarios. We then devise an effective data augmentation mechanism that ensures robustness to various transformations. The data augmentation mechanism explicitly enforces consistency between learned features and those influenced by illumination variations. Additionally, it regularizes the output consistency between disparity predictions of strong augmented samples and those generated from standard samples. Experiments on multiple mainstream benchmarks demonstrate that our SMFormer achieves state-of-the-art (SOTA) performance among self-supervised methods and even competes on par with supervised ones. Remarkably, in the challenging Booster benchmark, SMFormer even outperforms some SOTA supervised methods, such as CFNet.
>
---
#### [new 068] ABot-Claw: A Foundation for Persistent, Cooperative, and Self-Evolving Robotic Agents
- **分类: cs.CV**

- **简介: 该论文提出ABot-Claw，解决开放环境中机器人长期协作与自我进化问题，通过整合感知、记忆和反馈机制，实现从语言到物理动作的闭环控制。**

- **链接: [https://arxiv.org/pdf/2604.10096](https://arxiv.org/pdf/2604.10096)**

> **作者:** Dongjie Huo; Haoyun Liu; Guoqing Liu; Dekang Qi; Zhiming Sun; Maoguo Gao; Jianxin He; Yandan Yang; Xinyuan Chang; Feng Xiong; Xing Wei; Zhiheng Ma; Mu Xu
>
> **摘要:** Current embodied intelligent systems still face a substantial gap between high-level reasoning and low-level physical execution in open-world environments. Although Vision-Language-Action (VLA) models provide strong perception and intuitive responses, their open-loop nature limits long-horizon performance. Agents incorporating System 2 cognitive mechanisms improve planning, but usually operate in closed sandboxes with predefined toolkits and limited real-system control. OpenClaw provides a localized runtime with full system privileges, but lacks the embodied control architecture required for long-duration, multi-robot execution. We therefore propose ABot-Claw, an embodied extension of OpenClaw that integrates: 1) a unified embodiment interface with capability-driven scheduling for heterogeneous robot coordination; 2) a visual-centric cross-embodiment multimodal memory for persistent context retention and grounded retrieval; and 3) a critic-based closed-loop feedback mechanism with a generalist reward model for online progress evaluation, local correction, and replanning. With a decoupled architecture spanning the OpenClaw layer, shared service layer, and robot embodiment layer, ABot-Claw enables real-world interaction, closes the loop from natural language intent to physical action, and supports progressively self-evolving robotic agents in open, dynamic environments.
>
---
#### [new 069] Geoparsing: Diagram Parsing for Plane and Solid Geometry with a Unified Formal Language
- **分类: cs.CV**

- **简介: 该论文属于几何解析任务，旨在解决MLLM在几何推理中的感知瓶颈。通过设计统一形式语言和构建大规模数据集，提升平面与立体几何的解析能力。**

- **链接: [https://arxiv.org/pdf/2604.11600](https://arxiv.org/pdf/2604.11600)**

> **作者:** Peijie Wang; Ming-Liang Zhang; Jun Cao; Chao Deng; Dekang Ran; Hongda Sun; Pi Bu; Xuan Zhang; Yingyao Wang; Jun Song; Bo Zheng; Fei Yin; Cheng-Lin Liu
>
> **备注:** Accepted to ACL2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress but continue to struggle with geometric reasoning, primarily due to the perception bottleneck regarding fine-grained visual elements. While formal languages have aided plane geometry understanding, solid geometry which requires spatial understanding remains largely unexplored. In this paper, we address this challenge by designing a unified formal language that integrates plane and solid geometry, comprehensively covering geometric structures and semantic relations. We construct GDP-29K, a large-scale dataset comprising 20k plane and 9k solid geometry samples collected from diverse real-world sources, each paired with its ground-truth formal description. To ensure syntactic correctness and geometric consistency, we propose a training paradigm that combines Supervised Fine-Tuning with Reinforcement Learning via Verifiable Rewards. Experiments show that our approach achieves state-of-the-art parsing performance. Furthermore, we demonstrate that our parsed formal descriptions serve as a critical cognitive scaffold, significantly boosting MLLMs' capabilities for downstream geometry reasoning tasks. Our data and code are available at Geoparsing.
>
---
#### [new 070] 3D Multi-View Stylization with Pose-Free Correspondences Matching for Robust 3D Geometry Preservation
- **分类: cs.CV**

- **简介: 该论文属于多视角风格化任务，解决风格化导致几何信息破坏的问题。通过引入一致性损失和深度保持机制，提升3D重建的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2604.09639](https://arxiv.org/pdf/2604.09639)**

> **作者:** Shirsha Bose
>
> **摘要:** Artistic style transfer is well studied for images and videos, but extending it to multi-view 3D scenes remains difficult because stylization can disrupt correspondences needed by geometry-aware pipelines. Independent per-view stylization often causes texture drift, warped edges, and inconsistent shading, degrading SLAM, depth prediction, and multi-view reconstruction. This thesis addresses multi-view stylization that remains usable for downstream 3D tasks without assuming camera poses or an explicit 3D representation during training. We introduce a feed-forward stylization network trained with per-scene test-time optimization under a composite objective coupling appearance transfer with geometry preservation. Stylization is driven by an AdaIN-inspired loss from a frozen VGG-19 encoder, matching channel-wise moments to a style image. To stabilize structure across viewpoints, we propose a correspondence-based consistency loss using SuperPoint and SuperGlue, constraining descriptors from a stylized anchor view to remain consistent with matched descriptors from the original multi-view set. We also impose a depth-preservation loss using MiDaS/DPT and use global color alignment to reduce depth-model domain shift. A staged weight schedule introduces geometry and depth constraints. We evaluate on Tanks and Temples and Mip-NeRF 360 using image and reconstruction metrics. Style adherence and structure retention are measured by Color Histogram Distance (CHD) and Structure Distance (DSD). For 3D consistency, we use monocular DROID-SLAM trajectories and symmetric Chamfer distance on back-projected point clouds. Across ablations, correspondence and depth regularization reduce structural distortion and improve SLAM stability and reconstructed geometry; on scenes with MuVieCAST baselines, our method yields stronger trajectory and point-cloud consistency while maintaining competitive stylization.
>
---
#### [new 071] On The Application of Linear Attention in Multimodal Transformers
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决Transformer中注意力机制计算复杂度高的问题。通过引入线性注意力机制，降低计算开销至线性，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2604.10064](https://arxiv.org/pdf/2604.10064)**

> **作者:** Armin Gerami; Seyedehanita Madani; Ramani Duraiswami
>
> **备注:** Workshop on Any-to-Any Multimodal Learning (Any2Any), CVPR 2026
>
> **摘要:** Multimodal Transformers serve as the backbone for state-of-the-art vision-language models, yet their quadratic attention complexity remains a critical barrier to scalability. In this work, we investigate the viability of Linear Attention (LA) as a high-efficiency alternative within multimodal frameworks. By integrating LA, we reduce the computational overhead from quadratic to linear relative to sequence length while preserving competitive performance. We evaluate our approach across ViT-S/16, ViT-B/16, and ViT-L/16 architectures trained on the LAION-400M dataset, with validation focused on ImageNet-21K zero-shot accuracy. Our systematic evaluation demonstrates that Linear Attention not only yields significant computational savings but also adheres to the same scaling laws as standard softmax attention. These findings position Linear Attention as a robust, scalable solution for next-generation multimodal Transformers tasked with processing increasingly large and complex datasets.
>
---
#### [new 072] PASTA: Vision Transformer Patch Aggregation for Weakly Supervised Target and Anomaly Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出PASTA方法，用于工业和农业中的目标与异常分割任务。针对弱监督下像素级分割精度不足的问题，利用视觉Transformer和文本提示实现高效准确的分割。**

- **链接: [https://arxiv.org/pdf/2604.09701](https://arxiv.org/pdf/2604.09701)**

> **作者:** Melanie Neubauer; Elmar Rueckert; Christian Rauch
>
> **摘要:** Detecting unseen anomalies in unstructured environments presents a critical challenge for industrial and agricultural applications such as material recycling and weeding. Existing perception systems frequently fail to satisfy the strict operational requirements of these domains, specifically real-time processing, pixel-level segmentation precision, and robust accuracy, due to their reliance on exhaustively annotated datasets. To address these limitations, we propose a weakly supervised pipeline for object segmentation and classification using weak image-level supervision called 'Patch Aggregation for Segmentation of Targets and Anomalies' (PASTA). By comparing an observed scene with a nominal reference, PASTA identifies Target and Anomaly objects through distribution analysis in self-supervised Vision Transformer (ViT) feature spaces. Our pipeline utilizes semantic text-prompts via the Segment Anything Model 3 to guide zero-shot object segmentation. Evaluations on a custom steel scrap recycling dataset and a plant dataset demonstrate a 75.8% training time reduction of our approach to domain-specific baselines. While being domain-agnostic, our method achieves superior Target (up to 88.3% IoU) and Anomaly (up to 63.5% IoU) segmentation performance in the industrial and agricultural domain.
>
---
#### [new 073] Data-Efficient Surgical Phase Segmentation in Small-Incision Cataract Surgery: A Controlled Study of Vision Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于手术阶段分割任务，旨在解决小切口白内障手术中标签数据稀缺的问题。通过对比视觉模型，验证了基础模型在手术流程理解中的有效性。**

- **链接: [https://arxiv.org/pdf/2604.10514](https://arxiv.org/pdf/2604.10514)**

> **作者:** Lincoln Spencer; Song Wang; Chen Chen
>
> **摘要:** Surgical phase segmentation is central to computer-assisted surgery, yet robust models remain difficult to develop when labeled surgical videos are scarce. We study data-efficient phase segmentation for manual small-incision cataract surgery (SICS) through a controlled comparison of visual representations. To isolate representation quality, we pair each visual encoder with the same temporal model (MS-TCN++) under identical training and evaluation settings on SICS-155 (19 phases). We compare supervised encoders (ResNet-50, I3D) against large self-supervised foundation models (DINOv3, V-JEPA2), and use a cached-feature pipeline that decouples expensive visual encoding from lightweight temporal learning. Foundation-model features improve segmentation performance in this setup, with DINOv3 ViT-7B achieving the best overall results (83.4% accuracy, 87.0 edit score). We further examine cataract-domain transfer using unlabeled videos and lightweight adaptation, and analyze when it helps or hurts. Overall, the study indicates strong transferability of modern vision foundation models to surgical workflow understanding and provides practical guidance for low-label medical video settings. The project website is available at: this https URL
>
---
#### [new 074] EviRCOD: Evidence-Guided Probabilistic Decoding for Referring Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文属于Referring Camouflaged Object Detection任务，解决语义对齐、不确定性建模和边界保持问题，提出EviRCOD框架提升检测性能与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2604.10894](https://arxiv.org/pdf/2604.10894)**

> **作者:** Ye Wang; Kai Huang; Sumin Shen; Chenyang Ma
>
> **摘要:** Referring Camouflaged Object Detection (Ref-COD) focuses on segmenting specific camouflaged targets in a query image using category-aligned references. Despite recent advances, existing methods struggle with reference-target semantic alignment, explicit uncertainty modeling, and robust boundary preservation. To address these issues, we propose EviRCOD, an integrated framework consisting of three core components: (1) a Reference-Guided Deformable Encoder (RGDE) that employs hierarchical reference-driven modulation and multi-scale deformable aggregation to inject semantic priors and align cross-scale representations; (2) an Uncertainty-Aware Evidential Decoder (UAED) that incorporates Dirichlet evidence estimation into hierarchical decoding to model uncertainty and propagate confidence across scales; and (3) a Boundary-Aware Refinement Module (BARM) that selectively enhances ambiguous boundaries by exploiting low-level edge cues and prediction confidence. Experiments on the Ref-COD benchmark demonstrate that EviRCOD achieves state-of-the-art detection performance while providing well-calibrated uncertainty estimates. Code is available at: this https URL.
>
---
#### [new 075] Progressively Texture-Aware Diffusion for Contrast-Enhanced Sparse-View CT
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于稀疏视角CT重建任务，旨在提升图像内容恢复和纹理一致性。提出PTD模型，通过分阶段学习实现高效高质重建。**

- **链接: [https://arxiv.org/pdf/2604.11559](https://arxiv.org/pdf/2604.11559)**

> **作者:** Tianqi Wang; Wenchao Du; Hongyu Yang
>
> **备注:** ICASSP2026
>
> **摘要:** Diffusion-based sparse-view CT (SVCT) imaging has achieved remarkable advancements in recent years, thanks to its more stable generative capability. However, recovering reliable image content and visually consistent textures is still a crucial challenge. In this paper, we present a Progressively Texture-aware Diffusion (PTD) model, a coarse-to-fine learning framework tailored for SVCT. Specifically, PTD comprises a basic reconstructive module PTD$_{\textit{rec}}$ and a conditional diffusion module PTD$_{\textit{diff}}$. PTD$_{\textit{rec}}$ first learns a deterministic mapping to recover the majority of the underlying low-frequency signals (i.e., coarse content with smoothed textures), which serves as the initial estimation to enable fidelity. Moreover, PTD$_{\textit{diff}}$ aims to reconstruct high-fidelity details for coarse prediction, which explores a dual-domain guided conditional diffusion to generate reliable and consistent textures. Extensive experiments on sparse-view CT reconstruction demonstrate that our PTD achieves superior performance in terms of structure similarity and visual appeal with only a few sampling steps, which mitigates the randomness inherent in general diffusion models and enables a better trade-off between visual quality and fidelity of high-frequency details.
>
---
#### [new 076] Cross-Cultural Value Awareness in Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于AI公平性研究任务，旨在解决LVLMs中文化偏见问题。通过分析不同文化背景下模型的价值判断，评估其对文化差异的敏感度。**

- **链接: [https://arxiv.org/pdf/2604.09945](https://arxiv.org/pdf/2604.09945)**

> **作者:** Phillip Howard; Xin Su; Kathleen C. Fraser
>
> **摘要:** The rapid adoption of large vision-language models (LVLMs) in recent years has been accompanied by growing fairness concerns due to their propensity to reinforce harmful societal stereotypes. While significant attention has been paid to such fairness concerns in the context of social biases, relatively little prior work has examined the presence of stereotypes in LVLMs related to cultural contexts such as religion, nationality, and socioeconomic status. In this work, we aim to narrow this gap by investigating how cultural contexts depicted in images influence the judgments LVLMs make about a person's moral, ethical, and political values. We conduct a multi-dimensional analysis of such value judgments in five popular LVLMs using counterfactual image sets, which depict the same person across different cultural contexts. Our evaluation framework diagnoses LVLM awareness of cultural value differences through the use of Moral Foundations Theory, lexical analyses, and the sensitivity of generated values to depicted cultural contexts.
>
---
#### [new 077] STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation
- **分类: cs.CV**

- **简介: 该论文属于视频表示任务，解决视频中静态与动态成分混淆的问题。通过引入时空哈希编码，提升视频表示质量与运动模式学习效果。**

- **链接: [https://arxiv.org/pdf/2604.10910](https://arxiv.org/pdf/2604.10910)**

> **作者:** Jierun Lin; Jiacong Chen; Qingyu Mao; Shuai Liu; Xiandong Meng; Fanyang Meng; Yongsheng Liang
>
> **摘要:** 2D Gaussian Splatting (2DGS) has recently become a promising paradigm for high-quality video representation. However, existing methods employ content-agnostic or spatio-temporal feature overlapping embeddings to predict canonical Gaussian primitive deformations, which entangles static and dynamic components in videos and prevents modeling their distinct properties effectively. These result in inaccurate predictions for spatio-temporal deformations and unsatisfactory representation quality. To address these problems, this paper proposes a Spatio-Temporal hash encoding framework for Gaussian-based Video representation (STGV). By decomposing video features into learnable 2D spatial and 3D temporal hash encodings, STGV effectively facilitates the learning of motion patterns for dynamic components while maintaining background details for static this http URL addition, we construct a more stable and consistent initial canonical Gaussian representation through a key frame canonical initialization strategy, preventing from feature overlapping and a structurally incoherent geometry representation. Experimental results demonstrate that our method attains better video representation quality (+0.98 PSNR) against other Gaussian-based methods and achieves competitive performance in downstream video tasks.
>
---
#### [new 078] LRD-Net: A Lightweight Real-Centered Detection Network for Cross-Domain Face Forgery Detection
- **分类: cs.CV**

- **简介: 该论文属于跨域人脸伪造检测任务，解决现有方法泛化能力差和计算开销大的问题。提出LRD-Net，通过轻量级架构和真实中心学习策略，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.10862](https://arxiv.org/pdf/2604.10862)**

> **作者:** Xuecen Zhang; Vipin Chaudhary
>
> **摘要:** The rapid advancement of diffusion-based generative models has made face forgery detection a critical challenge in digital forensics. Current detection methods face two fundamental limitations: poor cross-domain generalization when encountering unseen forgery types, and substantial computational overhead that hinders deployment on resource-constrained devices. We propose LRD-Net (Lightweight Real-centered Detection Network), a novel framework that addresses both challenges simultaneously. Unlike existing dual-branch approaches that process spatial and frequency information independently, LRD-Net adopts a sequential frequency-guided architecture where a lightweight Multi-Scale Wavelet Guidance Module generates attention signals that condition a MobileNetV3-based spatial backbone. This design enables effective exploitation of frequency-domain cues while avoiding the redundancy of parallel feature extraction. Furthermore, LRD-Net employs a real-centered learning strategy with exponential moving average prototype updates and drift regularization, anchoring representations around authentic facial images rather than modeling diverse forgery patterns. Extensive experiments on the DiFF benchmark demonstrate that LRD-Net achieves state-of-the-art cross-domain detection accuracy, consistently outperforming existing methods. Critically, LRD-Net accomplishes this with only 2.63M parameters - approximately 9x fewer than conventional approaches - while achieving over 8x faster training and nearly 10x faster inference. These results demonstrate that robust cross-domain face forgery detection can be achieved without sacrificing computational efficiency, making LRD-Net suitable for real-time deployment in mobile authentication systems and resource-constrained environments.
>
---
#### [new 079] OmniScript: Towards Audio-Visual Script Generation for Long-Form Cinematic Video
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出OmniScript，解决长视频到详细脚本的生成任务，通过多模态模型和时间感知评估框架，提升脚本的时空准确性与语义完整性。**

- **链接: [https://arxiv.org/pdf/2604.11102](https://arxiv.org/pdf/2604.11102)**

> **作者:** Junfu Pu; Yuxin Chen; Teng Wang; Ying Shan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Current multimodal large language models (MLLMs) have demonstrated remarkable capabilities in short-form video understanding, yet translating long-form cinematic videos into detailed, temporally grounded scripts remains a significant challenge. This paper introduces the novel video-to-script (V2S) task, aiming to generate hierarchical, scene-by-scene scripts encompassing character actions, dialogues, expressions, and audio cues. To facilitate this, we construct a first-of-its-kind human-annotated benchmark and propose a temporally-aware hierarchical evaluation framework. Furthermore, we present OmniScript, an 8B-parameter omni-modal (audio-visual) language model tailored for long-form narrative comprehension. OmniScript is trained via a progressive pipeline that leverages chain-of-thought supervised fine-tuning for plot and character reasoning, followed by reinforcement learning using temporally segmented rewards. Extensive experiments demonstrate that despite its parameter efficiency, OmniScript significantly outperforms larger open-source models and achieves performance comparable to state-of-the-art proprietary models, including Gemini 3-Pro, in both temporal localization and multi-field semantic accuracy.
>
---
#### [new 080] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training
- **分类: cs.CV**

- **简介: 该论文属于rPPG任务，解决无监督训练中视频质量影响模型性能的问题。提出rPPG-VQA框架，结合信号与场景分析评估视频质量，并设计采样策略优化训练数据。**

- **链接: [https://arxiv.org/pdf/2604.11156](https://arxiv.org/pdf/2604.11156)**

> **作者:** Tianyang Dai; Ming Chang; Yan Chen; Yang Hu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Unsupervised remote photoplethysmography (rPPG) promises to leverage unlabeled video data, but its potential is hindered by a critical challenge: training on low-quality "in-the-wild" videos severely degrades model performance. An essential step missing here is to assess the suitability of the videos for rPPG model learning before using them for the task. Existing video quality assessment (VQA) methods are mainly designed for human perception and not directly applicable to the above purpose. In this work, we propose rPPG-VQA, a novel framework for assessing video suitability for rPPG. We integrate signal-level and scene-level analyses and design a dual-branch assessment architecture. The signal-level branch evaluates the physiological signal quality of the videos via robust signal-to-noise ratio (SNR) estimation with a multi-method consensus mechanism, and the scene-level branch uses a multimodal large language model (MLLM) to identify interferences like motion and unstable lighting. Furthermore, we propose a two-stage adaptive sampling (TAS) strategy that utilizes the quality score to curate optimal training datasets. Experiments show that by training on large-scale, "in-the-wild" videos filtered by our framework, we can develop unsupervised rPPG models that achieve a substantial improvement in accuracy on standard benchmarks. Our code is available at this https URL.
>
---
#### [new 081] Structured State-Space Regularization for Compact and Generation-Friendly Image Tokenization
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决 latent space 既紧凑又适合生成的问题。通过引入结构化状态空间正则化，提升生成模型效果。**

- **链接: [https://arxiv.org/pdf/2604.11089](https://arxiv.org/pdf/2604.11089)**

> **作者:** Jinsung Lee; Jaemin Oh; Namhun Kim; Dongwon Kim; Byung-Jun Yoon; Suha Kwak
>
> **备注:** Related blog posts in this https URL : Towards 2-Dimensional State-Space Models series
>
> **摘要:** Image tokenizers are central to modern vision models as they often operate in latent spaces. An ideal latent space must be simultaneously compact and generation-friendly: it should capture image's essential content compactly while remaining easy to model with generative approaches. In this work, we introduce a novel regularizer to align latent spaces with these two objectives. The key idea is to guide tokenizers to mimic the hidden state dynamics of state-space models (SSMs), thereby transferring their critical property, frequency awareness, to latent features. Grounded in a theoretical analysis of SSMs, our regularizer enforces encoding of fine spatial structures and frequency-domain cues into compact latent features; leading to more effective use of representation capacity and improved generative modelability. Experiments demonstrate that our method improves generation quality in diffusion models while incurring only minimal loss in reconstruction fidelity.
>
---
#### [new 082] MuPPet: Multi-person 2D-to-3D Pose Lifting
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于多人体态估计任务，旨在解决多人群体中个体间关系建模不足的问题。提出MuPPet框架，通过编码、增强和注意力机制提升3D姿态估计效果。**

- **链接: [https://arxiv.org/pdf/2604.09715](https://arxiv.org/pdf/2604.09715)**

> **作者:** Thomas Markhorst; Zhi-Yi Lin; Jouh Yeong Chew; Jan van Gemert; Xucong Zhang
>
> **备注:** Accepted at CVPRw 2026
>
> **摘要:** Multi-person social interactions are inherently built on coherence and relationships among all individuals within the group, making multi-person localization and body pose estimation essential to understanding these social dynamics. One promising approach is 2D-to-3D pose lifting which provides a 3D human pose consisting of rich spatial details by building on the significant advances in 2D pose estimation. However, the existing 2D-to-3D pose lifting methods often neglect inter-person relationships or cannot handle varying group sizes, limiting their effectiveness in multi-person settings. We propose MuPPet, a novel multi-person 2D-to-3D pose lifting framework that explicitly models inter-person correlations. To leverage these inter-person dependencies, our approach introduces Person Encoding to structure individual representations, Permutation Augmentation to enhance training diversity, and Dynamic Multi-Person Attention to adaptively model correlations between individuals. Extensive experiments on group interaction datasets demonstrate MuPPet significantly outperforms state-of-the-art single- and multi-person 2D-to-3D pose lifting methods, and improves robustness in occlusion scenarios. Our findings highlight the importance of modeling inter-person correlations, paving the way for accurate and socially-aware 3D pose estimation. Our code is available at: this https URL
>
---
#### [new 083] Is There Knowledge Left to Extract? Evidence of Fragility in Medically Fine-Tuned Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉语言模型研究，探讨其在复杂任务中的表现与局限性。通过实验验证医疗微调效果有限，模型性能脆弱且依赖提示。**

- **链接: [https://arxiv.org/pdf/2604.09841](https://arxiv.org/pdf/2604.09841)**

> **作者:** Oliver McLaughlin; Daniel Shubin; Carsten Eickhoff; Ritambhara Singh; William Rudman; Michal Golovanevsky
>
> **摘要:** Vision-language models (VLMs) are increasingly adapted through domain-specific fine-tuning, yet it remains unclear whether this improves reasoning beyond superficial visual cues, particularly in high-stakes domains like medicine. We evaluate four paired open-source VLMs (LLaVA vs. LLaVA-Med; Gemma vs. MedGemma) across four medical imaging tasks of increasing difficulty: brain tumor, pneumonia, skin cancer, and histopathology classification. We find that performance degrades toward near-random levels as task difficulty increases, indicating limited clinical reasoning. Medical fine-tuning provides no consistent advantage, and models are highly sensitive to prompt formulation, with minor changes causing large swings in accuracy and refusal rates. To test whether closed-form VQA suppresses latent knowledge, we introduce a description-based pipeline where models generate image descriptions that a text-only model (GPT-5.1) uses for diagnosis. This recovers a limited additional signal but remains bounded by task difficulty. Analysis of vision encoder embeddings further shows that failures stem from both weak visual representations and downstream reasoning. Overall, medical VLM performance is fragile, prompt-dependent, and not reliably improved by domain-specific fine-tuning.
>
---
#### [new 084] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于安全攻击任务，旨在破解大视觉语言模型的安全机制。通过操纵注意力模式，提高攻击成功率并减少迭代次数。**

- **链接: [https://arxiv.org/pdf/2604.10299](https://arxiv.org/pdf/2604.10299)**

> **作者:** Jingru Li; Wei Ren; Tianqing Zhu
>
> **备注:** Accepted to ACL 2026. Code: this https URL
>
> **摘要:** Large Vision-Language Models (LVLMs) rely on attention-based retrieval of safety instructions to maintain alignment during generation. Existing attacks typically optimize image perturbations to maximize harmful output likelihood, but suffer from slow convergence due to gradient conflict between adversarial objectives and the model's safety-retrieval mechanism. We propose Attention-Guided Visual Jailbreaking, which circumvents rather than overpowers safety alignment by directly manipulating attention patterns. Our method introduces two simple auxiliary objectives: (1) suppressing attention to alignment-relevant prefix tokens and (2) anchoring generation on adversarial image features. This simple yet effective push-pull formulation reduces gradient conflict by 45% and achieves 94.4% attack success rate on Qwen-VL (vs. 68.8% baseline) with 40% fewer iterations. At tighter perturbation budgets ($\epsilon=8/255$), we maintain 59.0% ASR compared to 45.7% for standard methods. Mechanistic analysis reveals a failure mode we term safety blindness: successful attacks suppress system-prompt attention by 80%, causing models to generate harmful content not by overriding safety rules, but by failing to retrieve them.
>
---
#### [new 085] OmniShow: Unifying Multimodal Conditions for Human-Object Interaction Video Generation
- **分类: cs.CV**

- **简介: 该论文属于人-物交互视频生成任务，旨在解决多模态条件下的视频生成问题。提出OmniShow框架，融合文本、图像、音频和姿态信息，提升生成质量和可控性。**

- **链接: [https://arxiv.org/pdf/2604.11804](https://arxiv.org/pdf/2604.11804)**

> **作者:** Donghao Zhou; Guisheng Liu; Hao Yang; Jiatong Li; Jingyu Lin; Xiaohu Huang; Yichen Liu; Xin Gao; Cunjian Chen; Shilei Wen; Chi-Wing Fu; Pheng-Ann Heng
>
> **备注:** Project page: this https URL
>
> **摘要:** In this work, we study Human-Object Interaction Video Generation (HOIVG), which aims to synthesize high-quality human-object interaction videos conditioned on text, reference images, audio, and pose. This task holds significant practical value for automating content creation in real-world applications, such as e-commerce demonstrations, short video production, and interactive entertainment. However, existing approaches fail to accommodate all these requisite conditions. We present OmniShow, an end-to-end framework tailored for this practical yet challenging task, capable of harmonizing multimodal conditions and delivering industry-grade performance. To overcome the trade-off between controllability and quality, we introduce Unified Channel-wise Conditioning for efficient image and pose injection, and Gated Local-Context Attention to ensure precise audio-visual synchronization. To effectively address data scarcity, we develop a Decoupled-Then-Joint Training strategy that leverages a multi-stage training process with model merging to efficiently harness heterogeneous sub-task datasets. Furthermore, to fill the evaluation gap in this field, we establish HOIVG-Bench, a dedicated and comprehensive benchmark for HOIVG. Extensive experiments demonstrate that OmniShow achieves overall state-of-the-art performance across various multimodal conditioning settings, setting a solid standard for the emerging HOIVG task.
>
---
#### [new 086] Defending against Patch-Based and Texture-Based Adversarial Attacks with Spectral Decomposition
- **分类: cs.CV**

- **简介: 该论文属于对抗防御任务，解决物理世界中基于补丁和纹理的对抗攻击问题。通过频谱分解与对抗训练结合，提出ASD防御方法。**

- **链接: [https://arxiv.org/pdf/2604.10715](https://arxiv.org/pdf/2604.10715)**

> **作者:** Wei Zhang; Xinyu Chang; Xiao Li; Yiming Zhu; Xiaolin Hu
>
> **备注:** Accepted by IEEE TIFS
>
> **摘要:** Adversarial examples present significant challenges to the security of Deep Neural Network (DNN) applications. Specifically, there are patch-based and texture-based attacks that are usually used to craft physical-world adversarial examples, posing real threats to security-critical applications such as person detection in surveillance and autonomous systems, because those attacks are physically realizable. Existing defense mechanisms face challenges in the adaptive attack setting, i.e., the attacks are specifically designed against them. In this paper, we propose Adversarial Spectrum Defense (ASD), a defense mechanism that leverages spectral decomposition via Discrete Wavelet Transform (DWT) to analyze adversarial patterns across multiple frequency scales. The multi-resolution and localization capability of DWT enables ASD to capture both high-frequency (fine-grained) and low-frequency (spatially pervasive) perturbations. By integrating this spectral analysis with the off-the-shelf Adversarial Training (AT) model, ASD provides a comprehensive defense strategy against both patch-based and texture-based adversarial attacks. Extensive experiments demonstrate that ASD+AT achieved state-of-the-art (SOTA) performance against various attacks, outperforming the APs of previous defense methods by 21.73%, in the face of strong adaptive adversaries specifically designed against ASD. Code available at this https URL .
>
---
#### [new 087] CoFusion: Multispectral and Hyperspectral Image Fusion via Spectral Coordinate Attention
- **分类: cs.CV**

- **简介: 该论文属于多光谱与高光谱图像融合任务，旨在解决空间细节与光谱保真度的平衡问题。提出CoFusion框架，通过跨尺度和跨模态协作提升融合效果。**

- **链接: [https://arxiv.org/pdf/2604.10584](https://arxiv.org/pdf/2604.10584)**

> **作者:** Baisong Li
>
> **摘要:** Multispectral and Hyperspectral Image Fusion (MHIF) aims to reconstruct high-resolution images by integrating low-resolution hyperspectral images (LRHSI) and high-resolution multispectral images (HRMSI). However, existing methods face limitations in modeling cross-scale interactions and spatial-spectral collaboration, making it difficult to achieve an optimal trade-off between spatial detail enhancement and spectral fidelity. To address this challenge, we propose CoFusion: a unified spatial-spectral collaborative fusion framework that explicitly models cross-scale and cross-modal dependencies. Specifically, a Multi-Scale Generator (MSG) is designed to construct a three-level pyramidal architecture, enabling the effective integration of global semantics and local details. Within each scale, a dual-branch strategy is employed: the Spatial Coordinate-Aware Mixing module (SpaCAM) is utilized to capture multi-scale spatial contexts, while the Spectral Coordinate-Aware Mixing module (SpeCAM) enhances spectral representations through frequency decomposition and coordinate mixing. Furthermore, we introduce the Spatial-Spectral Cross-Fusion Module (SSCFM) to perform dynamic cross-modal alignment and complementary feature fusion. Extensive experiments on multiple benchmark datasets demonstrate that CoFusion consistently outperforms state-of-the-art methods, achieving superior performance in both spatial reconstruction and spectral consistency.
>
---
#### [new 088] ExpertEdit: Learning Skill-Aware Motion Editing from Expert Videos
- **分类: cs.CV**

- **简介: 该论文属于运动编辑任务，旨在通过专家视频生成个性化反馈，提升初学者动作技能。工作包括提出ExpertEdit框架，利用无配对数据学习专家动作先验，实现无需监督的局部技能优化。**

- **链接: [https://arxiv.org/pdf/2604.10466](https://arxiv.org/pdf/2604.10466)**

> **作者:** Arjun Somayazulu; Kristen Grauman
>
> **摘要:** Visual feedback is critical for motor skill acquisition in sports and rehabilitation, and psychological studies show that observing near-perfect versions of one's own performance accelerates learning more effectively than watching expert demonstrations alone. We propose to enable such personalized feedback by automatically editing a person's motion to reflect higher skill. Existing motion editing approaches are poorly suited for this setting because they assume paired input-output data -- rare and expensive to curate for skill-driven tasks -- and explicit edit guidance at inference. We introduce ExpertEdit, a framework for skill-driven motion editing trained exclusively on unpaired expert video demonstrations. ExpertEdit learns an expert motion prior with a masked language modeling objective that infills masked motion spans with expert-level refinements. At inference, novice motion is masked at skill-critical moments and projected into the learned expert manifold, producing localized skill improvements without paired supervision or manual edit guidance. Across eight diverse techniques and three sports from Ego-Exo4D and Karate Kyokushin, ExpertEdit outperforms state-of-the-art supervised motion editing methods on multiple metrics of motion realism and expert quality. Project page: this https URL .
>
---
#### [new 089] LoGo-MR: Screening Breast MRI for Cancer Risk Prediction by Efficient Omni-Slice Modeling
- **分类: cs.CV**

- **简介: 该论文提出LoGo-MR框架，用于乳腺MRI的癌症风险预测，解决传统方法在计算效率和跨切片建模上的不足。**

- **链接: [https://arxiv.org/pdf/2604.11348](https://arxiv.org/pdf/2604.11348)**

> **作者:** Xin Wang; Yuan Gao; George Yiasemis; Antonio Portaluri; Zahra Aghdam; Muzhen He; Luyi Han; Yaofei Duan; Chunyao Lu; Xinglong Liang; Tianyu Zhang; Vivien van Veldhuizen; Yue Sun; Tao Tan; Ritse Mann; Jonas Teuwen
>
> **摘要:** Efficient and explainable breast cancer (BC) risk prediction is critical for large-scale population-based screening. Breast MRI provides functional information for personalized risk assessment. Yet effective modeling remains challenging as fully 3D CNNs capture volumetric context at high computational cost, whereas lightweight 2D CNNs fail to model inter-slice continuity. Importantly, breast MRI modeling for shor- and long-term BC risk stratification remains underexplored. In this study, we propose LoGo-MR, a 2.5D local-global structural modeling framework for five-year BC risk prediction. Aligned with clinical interpretation, our framework first employs neighbor-slice encoding to capture subtle local cues linked to short-term risk. It then integrates transformer-enhanced multiple-instance learning (MIL) to model distributed global patterns related to long-term risk and provide interpretable slice importance. We further apply this framework across axial, sagittal, and coronal planes as LoGo3-MR to capture complementary volumetric information. This multi-plane formulation enables voxel-level risk saliency mapping, which may assist radiologists in localizing risk-relevant regions during breast MRI interpretation. Evaluated on a large breast MRI screening cohort (~7.5K), our method outperforms 2D/3D baselines and existing SOTA MIL methods, achieving AUCs of 0.77-0.69 for 1- to 5-year prediction and improving C-index by ~6% over 3D CNNs. LoGo3-MR further improves overall performance with interpretable localization across three planes, and validation across seven backbones shows consistent gains. These results highlight the clinical potential of efficient MRI-based BC risk stratification for large-scale screening. Code will be released publicly.
>
---
#### [new 090] NTIRE 2026 The 3rd Restore Any Image Model (RAIM) Challenge: AI Flash Portrait (Track 3)
- **分类: cs.CV**

- **简介: 该论文属于低光人像修复任务，旨在解决噪声抑制、细节保留与色彩还原的平衡问题。通过构建数据集和评估体系，推动相关技术发展。**

- **链接: [https://arxiv.org/pdf/2604.11230](https://arxiv.org/pdf/2604.11230)**

> **作者:** Ya-nan Guan; Shaonan Zhang; Hang Guo; Yawen Wang; Xinying Fan; Tianqu Zhuang; Jie Liang; Hui Zeng; Guanyi Qin; Lishen Qu; Tao Dai; Shu-Tao Xia; Lei Zhang; Radu Timofte; Bin Chen; Yuanbo Zhou; Hongwei Wang; Qinquan Gao; Tong Tong; Yanxin Qian; Lizhao You; Jingru Cong; Lei Xiong; Shuyuan Zhu; Zhi-Qiang Zhong; Kan Lv; Yang Yang; Kailing Tang; Minjian Zhang; Zhipei Lei; Zhe Xu; Liwen Zhang; Dingyong Gou; Yanlin Wu; Cong Li; Xiaohui Cui; Jiajia Liu; Guoyi Xu; Yaoxin Jiang; Yaokun Shi; Jiachen Tu; Liqing Wang; Shihang Li; Bo Zhang; Biao Wang; Haiming Xu; Xiang Long; Xurui Liao; Yanqiao Zhai; Haozhe Li; Shijun Shi; Jiangning Zhang; Yong Liu; Kai Hu; Jing Xu; Xianfang Zeng; Yuyang Liu; Minchen Wei
>
> **备注:** Accepted to CVPR 2026 Workshop. Includes supplementary material as ancillary file
>
> **摘要:** In this paper, we present a comprehensive overview of the NTIRE 2026 3rd Restore Any Image Model (RAIM) challenge, with a specific focus on Track 3: AI Flash Portrait. Despite significant advancements in deep learning for image restoration, existing models still encounter substantial challenges in real-world low-light portrait scenarios. Specifically, they struggle to achieve an optimal balance among noise suppression, detail preservation, and faithful illumination and color reproduction. To bridge this gap, this challenge aims to establish a novel benchmark for real-world low-light portrait restoration. We comprehensively evaluate the proposed algorithms utilizing a hybrid evaluation system that integrates objective quantitative metrics with rigorous subjective assessment protocols. For this competition, we provide a dataset containing 800 groups of real-captured low-light portrait data. Each group consists of a 1K-resolution low-light input image, a 1K ground truth (GT), and a 1K person mask. This challenge has garnered widespread attention from both academia and industry, attracting over 100 participating teams and receiving more than 3,000 valid submissions. This report details the motivation behind the challenge, the dataset construction process, the evaluation metrics, and the various phases of the competition. The released dataset and baseline code for this track are publicly available from the same \href{this https URL}{GitHub repository}, and the official challenge webpage is hosted on \href{this https URL}{CodaBench}.
>
---
#### [new 091] LMMs Meet Object-Centric Vision: Understanding, Segmentation, Editing and Generation
- **分类: cs.CV**

- **简介: 该论文属于多模态视觉任务，旨在解决LMMs在对象级理解、分割、编辑和生成中的不足。工作包括综述相关进展，总结关键方法与评估标准，并探讨未来方向。**

- **链接: [https://arxiv.org/pdf/2604.11789](https://arxiv.org/pdf/2604.11789)**

> **作者:** Yuqian Yuan; Wenqiao Zhang; Juekai Lin; Yu Zhong; Mingjian Gao; Binhe Yu; Yunqi Cao; Wentong Li; Yueting Zhuang; Beng Chin Ooi
>
> **备注:** 38 pages, 6 figures
>
> **摘要:** Large Multimodal Models (LMMs) have achieved remarkable progress in general-purpose vision--language understanding, yet they remain limited in tasks requiring precise object-level grounding, fine-grained spatial reasoning, and controllable visual manipulation. In particular, existing systems often struggle to identify the correct instance, preserve object identity across interactions, and localize or modify designated regions with high precision. Object-centric vision provides a principled framework for addressing these challenges by promoting explicit representations and operations over visual entities, thereby extending multimodal systems from global scene understanding to object-level understanding, segmentation, editing, and generation. This paper presents a comprehensive review of recent advances at the convergence of LMMs and object-centric vision. We organize the literature into four major themes: object-centric visual understanding, object-centric referring segmentation, object-centric visual editing, and object-centric visual generation. We further summarize the key modeling paradigms, learning strategies, and evaluation protocols that support these capabilities. Finally, we discuss open challenges and future directions, including robust instance permanence, fine-grained spatial control, consistent multi-step interaction, unified cross-task modeling, and reliable benchmarking under distribution shift. We hope this paper provides a structured perspective on the development of scalable, precise, and trustworthy object-centric multimodal systems.
>
---
#### [new 092] The Impact of Federated Learning on Distributed Remote Sensing Archives
- **分类: cs.CV**

- **简介: 该论文研究联邦学习在分布式遥感数据中的应用，解决数据分散与非独立同分布带来的模型训练问题，通过实验比较不同算法和模型效果。**

- **链接: [https://arxiv.org/pdf/2604.11562](https://arxiv.org/pdf/2604.11562)**

> **作者:** Anand Umashankar; Karam Tomotaki-Dawoud; Nicolai Schneider
>
> **备注:** This work was completed in 2021. It is posted as a historical record and reference baseline
>
> **摘要:** Remote sensing archives are inherently distributed: Earth observation missions such as Sentinel-1, Sentinel-2, and Sentinel-3 have collectively accumulated more than 5 petabytes of imagery, stored and processed across many geographically dispersed platforms. Training machine learning models on such data in a centralized fashion is impractical due to data volume, sovereignty constraints, and geographic distribution. Federated learning (FL) addresses this by keeping data local and exchanging only model updates. A central challenge for remote sensing is the non-IID nature of Earth observation data: label distributions vary strongly by geographic region, degrading the convergence of standard FL algorithms. In this paper, we conduct a systematic empirical study of three FL strategies -- FedAvg, FedProx, and bulk synchronous parallel (BSP) -- applied to multi-label remote sensing image classification under controlled non-IID label-skew conditions. We evaluate three convolutional neural network (CNN) architectures of increasing depth (LeNet, AlexNet, and ResNet-34) and analyze the joint effect of algorithm choice, model capacity, client fraction, client count, batch size, and communication cost. Experiments on the UC Merced multi-label dataset show that FedProx outperforms FedAvg for deeper architectures under data heterogeneity, that BSP approaches centralized accuracy at the cost of high sequential communication, and that LeNet provides the best accuracy-communication trade-off for the dataset scale considered.
>
---
#### [new 093] Degradation-Consistent Paired Training for Robust AI-Generated Image Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI生成图像检测任务，旨在解决真实场景下图像退化导致的性能下降问题。提出DCPT方法，通过配对一致性约束提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10102](https://arxiv.org/pdf/2604.10102)**

> **作者:** Zongyou Yang; Yinghan Hou; Xiaokun Yang
>
> **备注:** 6 pages, 5 figures, 2 tables
>
> **摘要:** AI-generated image detectors suffer significant performance degradation under real-world image corruptions such as JPEG compression, Gaussian blur, and resolution downsampling. We observe that state-of-the-art methods, including B-Free, treat degradation robustness as a byproduct of data augmentation rather than an explicit training objective. In this work, we propose Degradation-Consistent Paired Training (DCPT), a simple yet effective training strategy that explicitly enforces robustness through paired consistency constraints. For each training image, we construct a clean view and a degraded view, then impose two constraints: a feature consistency loss that minimizes the cosine distance between clean and degraded representations, and a prediction consistency loss based on symmetric KL divergence that aligns output distributions across views. DCPT adds zero additional parameters and zero inference overhead. Experiments on the Synthbuster benchmark (9 generators, 8 degradation conditions) demonstrate that DCPT improves the degraded-condition average accuracy by 9.1 percentage points compared to an identical baseline without paired training, while sacrificing only 0.9% clean accuracy. The improvement is most pronounced under JPEG compression (+15.7% to +17.9%). Ablation further reveals that adding architectural components leads to overfitting on limited training data, confirming that training objective improvement is more effective than architectural augmentation for degradation robustness.
>
---
#### [new 094] TaFall: Balance-Informed Fall Detection via Passive Thermal Sensing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跌倒检测任务，旨在解决隐私与可靠性之间的平衡问题。通过热成像传感，提出TaFall系统，利用姿态和平衡动态进行跌倒检测。**

- **链接: [https://arxiv.org/pdf/2604.09693](https://arxiv.org/pdf/2604.09693)**

> **作者:** Chengxiao Li; Xie Zhang; Wei Zhu; Yan Jiang; Chenshu Wu
>
> **摘要:** Falls are a major cause of injury and mortality among older adults, yet most incidents occur in private indoor environments where monitoring must balance effectiveness with privacy. Existing privacy-preserving fall detection approaches, particularly those based on radio frequency sensing, often rely on coarse motion cues, which limits reliability in real-world deployments. We introduce TaFall, a balance-informed fall detection system based on low-cost, privacy-preserving thermal array sensing. The key insight is that TaFall models a fall as a process of balance degradation and detects falls by estimating pose-driven biomechanical balance dynamics. To enable this capability from low-resolution thermal array maps, we propose (i) an appearance-motion fusion model for robust pose reconstruction, (ii) physically grounded balance-aware learning, and (iii) pose-bridged pretraining to improve robustness. TaFall achieves a detection rate of 98.26% with a false alarm rate of 0.65% on our dataset with over 3,000 fall instances from 35 participants across diverse indoor environments. In 27 day deployments across four homes, TaFall attains an ultra-low false alarm rate of 0.00126% and a pilot bathroom study confirms robustness under moisture and thermal interference. Together, these results establish TaFall as a reliable and privacy-preserving approach to fall detection in everyday living environments.
>
---
#### [new 095] Identity-Aware U-Net: Fine-grained Cell Segmentation via Identity-Aware Representation Learning
- **分类: cs.CV; cs.AI; q-bio.QM**

- **简介: 该论文属于细粒度目标分割任务，解决相似形状物体难以区分的问题。提出IAU-Net框架，结合空间定位与实例鉴别，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.09702](https://arxiv.org/pdf/2604.09702)**

> **作者:** Rui Xiao
>
> **摘要:** Precise segmentation of objects with highly similar shapes remains a challenging problem in dense prediction, especially in scenarios with ambiguous boundaries, overlapping instances, and weak inter-instance visual differences. While conventional segmentation models are effective at localizing object regions, they often lack the discriminative capacity required to reliably distinguish a target object from morphologically similar distractors. In this work, we study fine-grained object segmentation from an identity-aware perspective and propose Identity-Aware U-Net (IAU-Net), a unified framework that jointly models spatial localization and instance discrimination. Built upon a U-Net-style encoder-decoder architecture, our method augments the segmentation backbone with an auxiliary embedding branch that learns discriminative identity representations from high-level features, while the main branch predicts pixel-accurate masks. To enhance robustness in distinguishing objects with near-identical contours or textures, we further incorporate triplet-based metric learning, which pulls target-consistent embeddings together and separates them from hard negatives with similar morphology. This design enables the model to move beyond category-level segmentation and acquire a stronger capability for precise discrimination among visually similar objects. Experiments on benchmarks including cell segmentation demonstrate promising results, particularly in challenging cases involving similar contours, dense layouts, and ambiguous boundaries.
>
---
#### [new 096] GazeVaLM: A Multi-Observer Eye-Tracking Benchmark for Evaluating Clinical Realism in AI-Generated X-Rays
- **分类: cs.CV**

- **简介: 该论文提出GazeVaLM，一个用于评估AI生成X光片临床真实性的多观察者眼动基准。任务是研究临床感知，解决AI生成图像真实性评估问题，通过收集专家眼动数据及模型预测进行对比分析。**

- **链接: [https://arxiv.org/pdf/2604.11653](https://arxiv.org/pdf/2604.11653)**

> **作者:** David Wong; Zeynep Isik; Bin Wang; Marouane Tliba; Gorkem Durak; Elif Keles; Halil Ertugrul Aktas; Aladine Chetouani; Cagdas Topel; Nicolo Gennaro; Camila Lopes Vendrami; Tugce Agirlar Trabzonlu; Amir Ali Rahsepar; Laetitia Perronne; Matthew Antalek; Onural Ozturk; Gokcan Okur; Andrew C. Gordon; Ayis Pyrros; Frank H. Miller; Amir Borhani; Hatice Savas; Eric Hart; Elizabeth Krupinski; Ulas Bagci
>
> **备注:** This work appears in ACM ETRA 2026
>
> **摘要:** We introduce GazeVaLM, a public eye-tracking dataset for studying clinical perception during chest radiograph authenticity assessment. The dataset comprises 960 gaze recordings from 16 expert radiologists interpreting 30 real and 30 synthetic chest X-rays (generated by diffusion based generative AI) under two conditions: diagnostic assessment and real-fake classification (Visual Turing test). For each image-observer pair, we provide raw gaze samples, fixation maps, scanpaths, saliency density maps, structured diagnostic labels, and authenticity judgments. We extend the protocol to 6 state-of-the-art multimodal LLMs, releasing their predicted diagnoses, authenticity labels, and confidence scores under matched conditions - enabling direct human-AI comparison at both decision and uncertainty levels. We further provide analyses of gaze agreement, inter-observer consistency, and benchmarking of radiologists versus LLMs in diagnostic accuracy and authenticity detection. GazeVaLM supports research in gaze modeling, clinical decision-making, human-AI comparison, generative image realism assessment, and uncertainty quantification. By jointly releasing visual attention data, clinical labels, and model predictions, we aim to facilitate reproducible research on how experts and AI systems perceive, interpret, and evaluate medical images. The dataset is available at this https URL.
>
---
#### [new 097] Towards Realistic 3D Emission Materials: Dataset, Baseline, and Evaluation for Emission Texture Generation
- **分类: cs.CV**

- **简介: 该论文属于3D纹理生成任务，旨在解决现有方法无法生成发光材质的问题。通过构建数据集、提出基线模型和定义评估指标，实现真实发光纹理的生成。**

- **链接: [https://arxiv.org/pdf/2604.11006](https://arxiv.org/pdf/2604.11006)**

> **作者:** Zhiyuan Zhang; Zijian Zhou; Linjun Li; Long Chen; Hao Tang; Yichen Gong
>
> **备注:** Dataset will be available at this https URL
>
> **摘要:** 3D texture generation is receiving increasing attention, as it enables the creation of realistic and aesthetic texture materials for untextured 3D meshes. However, existing 3D texture generation methods are limited to producing only a few types of non-emissive PBR materials (e.g., albedo, metallic maps and roughness maps), making them difficult to replicate highly popular styles, such as cyberpunk, failing to achieve effects like realistic LED emissions. To address this limitation, we propose a novel task, emission texture generation, which enables the synthesized 3D objects to faithfully reproduce the emission materials from input reference images. Our key contributions include: first, We construct the Objaverse-Emission dataset, the first dataset that contains 40k 3D assets with high-quality emission materials. Second, we propose EmissionGen, a novel baseline for the emission texture generation task. Third, we define detailed evaluation metrics for the emission texture generation task. Our results demonstrate significant potential for future industrial applications. Dataset will be available at this https URL.
>
---
#### [new 098] Multinex: Lightweight Low-light Image Enhancement via Multi-prior Retinex
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于低光图像增强任务，旨在解决现有方法模型庞大、训练复杂及色彩不稳定的问题。提出Multinex框架，通过多先验Retinex结构实现轻量化高效增强。**

- **链接: [https://arxiv.org/pdf/2604.10359](https://arxiv.org/pdf/2604.10359)**

> **作者:** Alexandru Brateanu; Tingting Mu; Codruta Ancuti; Cosmin Ancuti
>
> **摘要:** Low-light image enhancement (LLIE) aims to restore natural visibility, color fidelity, and structural detail under severe illumination degradation. State-of-the-art (SOTA) LLIE techniques often rely on large models and multi-stage training, limiting practicality for edge deployment. Moreover, their dependence on a single color space introduces instability and visible exposure or color artifacts. To address these, we propose Multinex, an ultra-lightweight structured framework that integrates multiple fine-grained representations within a principled Retinex residual formulation. It decomposes an image into illumination and color prior stacks derived from distinct analytic representations, and learns to fuse these representations into luminance and reflectance adjustments required to correct exposure. By prioritizing enhancement over reconstruction and exploiting lightweight neural operations, Multinex significantly reduces computational cost, exemplified by its lightweight (45K parameters) and nano (0.7K parameters) versions. Extensive benchmarks show that all lightweight variants significantly outperform their corresponding lightweight SOTA models, and reach comparable performance to heavy models. Paper page available at this https URL.
>
---
#### [new 099] Mining Attribute Subspaces for Efficient Fine-tuning of 3D Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于3D模型微调任务，解决LoRA子空间的关联性与解耦问题。通过合成数据生成和实验，提取并验证了不同变化类型的LoRA子空间，提升微调效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.10095](https://arxiv.org/pdf/2604.10095)**

> **作者:** Yu Jiang; Hanwen Jiang; Ahmed Abdelkader; Wen-Sheng Chu; Brandon Y. Feng; Zhangyang Wang; Qixing Huang
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** With the emergence of 3D foundation models, there is growing interest in fine-tuning them for downstream tasks, where LoRA is the dominant fine-tuning paradigm. As 3D datasets exhibit distinct variations in texture, geometry, camera motion, and lighting, there are interesting fundamental questions: 1) Are there LoRA subspaces associated with each type of variation? 2) Are these subspaces disentangled (i.e., orthogonal to each other)? 3) How do we compute them effectively? This paper provides answers to all these questions. We introduce a robust approach that generates synthetic datasets with controlled variations, fine-tunes a LoRA adapter on each dataset, and extracts a LoRA sub-space associated with each type of variation. We show that these subspaces are approximately disentangled. Integrating them leads to a reduced LoRA subspace that enables efficient LoRA fine-tuning with improved prediction accuracy for downstream tasks. In particular, we show that such a reduced LoRA subspace, despite being derived entirely from synthetic data, generalizes to real datasets. An ablation study validates the effectiveness of the choices in our approach.
>
---
#### [new 100] Energy-oriented Diffusion Bridge for Image Restoration with Foundational Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决扩散桥模型采样效率低和恢复质量受限的问题。提出E-Bridge框架，通过优化轨迹和单步映射提升性能。**

- **链接: [https://arxiv.org/pdf/2604.10983](https://arxiv.org/pdf/2604.10983)**

> **作者:** Jinhui Hou; Zhiyu Zhu; Junhui Hou
>
> **备注:** Accepted to ICLR26
>
> **摘要:** Diffusion bridge models have shown great promise in image restoration by explicitly connecting clean and degraded image distributions. However, they often rely on complex and high-cost trajectories, which limit both sampling efficiency and final restoration quality. To address this, we propose an Energy-oriented diffusion Bridge (E-Bridge) framework to approximate a set of low-cost manifold geodesic trajectories to boost the performance of the proposed method. We achieve this by designing a novel bridge process that evolves over a shorter time horizon and makes the reverse process start from an entropy-regularized point that mixes the degraded image and Gaussian noise, which theoretically reduces the required trajectory energy. To solve this process efficiently, we draw inspiration from consistency models to learn a single-step mapping function, optimized via a continuous-time consistency objective tailored for our trajectory, so as to analytically map any state on the trajectory to the target image. Notably, the trajectory length in our framework becomes a tunable task-adaptive knob, allowing the model to adaptively balance information preservation against generative power for tasks of varying degradation, such as denoising versus super-resolution. Extensive experiments demonstrate that our E-Bridge achieves state-of-the-art performance across various image restoration tasks while enabling high-quality recovery with a single or fewer sampling steps. Our project page is this https URL.
>
---
#### [new 101] Beyond Reconstruction: Reconstruction-to-Vector Diffusion for Hyperspectral Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于高光谱异常检测任务，解决传统方法在子像素异常消失和训练偏差问题。提出R2VD框架，通过四阶段流程提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.11390](https://arxiv.org/pdf/2604.11390)**

> **作者:** Jijun Xiang; Jiayi Wang; Pengxiang Wang; Cheng Chen; Nian Wang; Tao Wang
>
> **摘要:** While Hyperspectral Anomaly Detection (HAD) excels at identifying sparse targets in complex scenes, existing models remain trapped in a scalar "reconstruction-as-endpoint" paradigm. This reliance on ambiguous scalar residuals consistently triggers sub-pixel anomaly vanishing during spatial downsampling, alongside severe confirmation bias when unpurified anomalies corrupt training weights. In this paper, we propose Reconstruction-to-Vector Diffusion (R2VD), which fundamentally redefines reconstruction as a manifold purification origin to establish a novel residual-guided generative dynamics paradigm. Our framework introduces a four-stage pipeline: (1) a Physical Prior Extraction (PPE) stage that mitigates early confirmation bias via dual-stream statistical guidance; (2) a Guided Manifold Purification (GMP) stage utilizing an OmniContext Autoencoder (OCA) to extract purified residual maps while preserving fragile sub-pixel topologies; (3) a Residual Score Modeling (RSM) stage where a Diffusion Transformer (DiT), guarded by a Physical Spectral Firewall (PSF), effectively isolates cross-spectral leakage; and (4) a Vector Dynamics Inference (VDI) stage that robustly decouples targets from backgrounds by evaluating high-dimensional vector interference patterns instead of conventional scalar errors. Comprehensive evaluations on eight datasets confirm that R2VD establishes a new state-of-the-art, delivering exceptional target detectability and background suppression.
>
---
#### [new 102] LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization
- **分类: cs.CV**

- **简介: 该论文属于LiDAR重定位任务，解决复杂场景下定位精度低的问题。提出LEADER框架，通过几何编码器和截断相对可靠性损失提升定位鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.11355](https://arxiv.org/pdf/2604.11355)**

> **作者:** Jianshi Wu; Minghang Zhu; Dunqiang Liu; Wen Li; Sheng Ao; Siqi Shen; Chenglu Wen; Cheng Wang
>
> **备注:** Accepted to CVPR 2026 (Highlight)
>
> **摘要:** LiDAR relocalization has attracted increasing attention as it can deliver accurate 6-DoF pose estimation in complex 3D environments. Recent learning-based regression methods offer efficient solutions by directly predicting global poses without the need for explicit map storage. However, these methods often struggle in challenging scenes due to their equal treatment of all predicted points, which is vulnerable to noise and outliers. In this paper, we propose LEADER, a robust LiDAR-based relocalization framework enhanced by a simple, yet effective geometric encoder. Specifically, a Robust Projection-based Geometric Encoder architecture which captures multi-scale geometric features is first presented to enhance descriptiveness in geometric representation. A Truncated Relative Reliability loss is then formulated to model point-wise ambiguity and mitigate the influence of unreliable predictions. Extensive experiments on the Oxford RobotCar and NCLT datasets demonstrate that LEADER outperforms state-of-the-art methods, achieving 24.1% and 73.9% relative reductions in position error over existing techniques, respectively. The source code is released on this https URL.
>
---
#### [new 103] VGGT-HPE: Reframing Head Pose Estimation as Relative Pose Prediction
- **分类: cs.CV**

- **简介: 该论文属于头姿估计任务，解决传统绝对姿态回归的局限性。提出VGGT-HPE，通过相对姿态预测提升精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10106](https://arxiv.org/pdf/2604.10106)**

> **作者:** Vasiliki Vasileiou; Panagiotis P. Filntisis; Petros Maragos; Kostas Daniilidis
>
> **备注:** CVPRW 2026
>
> **摘要:** Monocular head pose estimation is traditionally formulated as direct regression from a single image to an absolute pose. This paradigm forces the network to implicitly internalize a dataset-specific canonical reference frame. In this work, we argue that predicting the relative rigid transformation between two observed head configurations is a fundamentally easier and more robust formulation. We introduce VGGT-HPE, a relative head pose estimator built upon a general-purpose geometry foundation model. Finetuned exclusively on synthetic facial renderings, our method sidesteps the need for an implicit anchor by reducing the problem to estimating a geometric displacement from an explicitly provided anchor with a known pose. As a practical benefit, the relative formulation also allows the anchor to be chosen at test time - for instance, a near-neutral frame or a temporally adjacent one - so that the prediction difficulty can be controlled by the application. Despite zero real-world training data, VGGT-HPE achieves state-of-the-art results on the BIWI benchmark, outperforming established absolute regression methods trained on mixed and real datasets. Through controlled easy- and hard-pair benchmarks, we also systematically validate our core hypothesis: relative prediction is intrinsically more accurate than absolute regression, with the advantage scaling alongside the difficulty of the target pose. Project page and code: this https URL
>
---
#### [new 104] TRACE: Thermal Recognition Attentive-Framework for CO2 Emissions from Livestock
- **分类: cs.CV**

- **简介: 该论文提出TRACE框架，用于从热成像视频中检测牛的CO2排放，解决非接触式、连续监测问题。任务为CO2排放量化，工作包括气体感知注意力机制和时间融合模块。**

- **链接: [https://arxiv.org/pdf/2604.09648](https://arxiv.org/pdf/2604.09648)**

> **作者:** Taminul Islam; Abdellah Lakhssassi; Toqi Tahamid Sarker; Mohamed Embaby; Khaled R Ahmed; Amer AbuGhazaleh
>
> **摘要:** Quantifying exhaled CO2 from free-roaming cattle is both a direct indicator of rumen metabolic state and a prerequisite for farm-scale carbon accounting, yet no existing system can deliver continuous, spatially resolved measurements without physical confinement or contact. We present TRACE (Thermal Recognition Attentive-Framework for CO2 Emissions from Livestock), the first unified framework to jointly address per-frame CO2 plume segmentation and clip-level emission flux classification from mid-wave infrared (MWIR) thermal video. TRACE contributes three domain-specific advances: a Thermal Gas-Aware Attention (TGAA) encoder that incorporates per-pixel gas intensity as a spatial supervisory signal to direct self-attention toward high-emission regions at each encoder stage; an Attention-based Temporal Fusion (ATF) module that captures breath-cycle dynamics through structured cross-frame attention for sequence-level flux classification; and a four-stage progressive training curriculum that couples both objectives while preventing gradient interference. Benchmarked against fifteen state-of-the-art models on the CO2 Farm Thermal Gas Dataset, TRACE achieves an mIoU of 0.998 and the best result on every segmentation and classification metric simultaneously, outperforming domain-specific gas segmenters with several times more parameters and surpassing all baselines in flux classification. Ablation studies confirm that each component is individually essential: gas-conditioned attention alone determines precise plume boundary localization, and temporal reasoning is indispensable for flux-level discrimination. TRACE establishes a practical path toward non-invasive, continuous, per-animal CO2 monitoring from overhead thermal cameras at commercial scale. Codes are available at this https URL.
>
---
#### [new 105] Learning Robustness at Test-Time from a Non-Robust Teacher
- **分类: cs.CV**

- **简介: 该论文属于测试时适应任务，旨在提升非鲁棒预训练模型的对抗鲁棒性。通过提出无标签框架，增强优化稳定性与鲁棒性-精度平衡。**

- **链接: [https://arxiv.org/pdf/2604.11590](https://arxiv.org/pdf/2604.11590)**

> **作者:** Stefano Bianchettin; Giulio Rossolini; Giorgio Buttazzo
>
> **摘要:** Nowadays, pretrained models are increasingly used as general-purpose backbones and adapted at test-time to downstream environments where target data are scarce and unlabeled. While this paradigm has proven effective for improving clean accuracy on the target domain, adversarial robustness has received far less attention, especially when the original pretrained model is not explicitly designed to be robust. This raises a practical question: \emph{can a pretrained, non-robust model be adapted at test-time to improve adversarial robustness on a target distribution?} To face this question, this work studies how adversarial training strategies behave when integrated into adaptation schemes for the unsupervised test-time setting, where only a small set of unlabeled target samples is available. It first analyzes how classical adversarial training formulations can be extended to this scenario, showing that straightforward distillation-based adaptations remain unstable and highly sensitive to hyperparameter tuning, particularly when the teacher itself is non-robust. To address these limitations, the work proposes a label-free framework that uses the predictions of a non-robust teacher model as a semantic anchor for both the clean and adversarial objectives during adaptation. We further provide theoretical insights showing that our formulation yields a more stable alternative to the self-consistency-based regularization commonly used in classical adversarial training. Experiments evaluate the proposed approach on CIFAR-10 and ImageNet under induced photometric transformations. The results support the theoretical insights by showing that the proposed approach achieves improved optimization stability, lower sensitivity to parameter choices, and a better robustness-accuracy trade-off than existing baselines in this post-deployment test-time setting.
>
---
#### [new 106] A Deep Equilibrium Network for Hyperspectral Unmixing
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像解混任务，旨在解决传统方法和深度学习方法在建模复杂特征与物理可解释性上的不足。提出DEQ-Unmix，通过深度均衡模型实现高效、常内存的解混。**

- **链接: [https://arxiv.org/pdf/2604.11279](https://arxiv.org/pdf/2604.11279)**

> **作者:** Chentong Wang; Jincheng Gao; Fei Zhu; Jie Chen
>
> **摘要:** Hyperspectral unmixing (HU) is crucial for analyzing hyperspectral imagery, yet achieving accurate unmixing remains challenging. While traditional methods struggle to effectively model complex spectral-spatial features, deep learning approaches often lack physical interpretability. Unrolling-based methods, despite offering network interpretability, inadequately exploit spectral-spatial information and incur high memory costs and numerical precision issues during backpropagation. To address these limitations, we propose DEQ-Unmix, which reformulates abundance estimation as a deep equilibrium model, enabling efficient constant-memory training via implicit differentiation. It replaces the gradient operator of the data reconstruction term with a trainable convolutional network to capture spectral-spatial information. By leveraging implicit differentiation, DEQ-Unmix enables efficient and constant-memory backpropagation. Experiments on synthetic and two real-world datasets demonstrate that DEQ-Unmix achieves superior unmixing performance while maintaining constant memory cost.
>
---
#### [new 107] AC-MIL: Weakly Supervised Atrial LGE-MRI Quality Assessment via Adversarial Concept Disentanglement
- **分类: cs.CV**

- **简介: 该论文属于医学图像质量评估任务，解决弱监督下LGE-MRI扫描质量问题。提出AC-MIL框架，分解图像质量为临床概念，实现可解释的诊断分析。**

- **链接: [https://arxiv.org/pdf/2604.10303](https://arxiv.org/pdf/2604.10303)**

> **作者:** K M Arefeen Sultan; Kaysen Hansen; Benjamin Orkild; Alan Morris; Eugene Kholmovski; Erik Bieging; Eugene Kwan; Ravi Ranjan; Ed DiBella; Shireen Elhabian
>
> **摘要:** High-quality Late Gadolinium Enhancement (LGE) MRI can be helpful for atrial fibrillation management, yet scan quality is frequently compromised by patient motion, irregular breathing, and suboptimal image acquisition timing. While Multiple Instance Learning (MIL) has emerged as a powerful tool for automated quality assessment under weak supervision, current state-of-the-art methods map localized visual evidence to a single, opaque global feature vector. This black box approach fails to provide actionable feedback on specific failure modes, obscuring whether a scan degrades due to motion blur, inadequate contrast, or a lack of anatomical context. In this paper, we propose Adversarial Concept-MIL (AC-MIL), a weakly supervised framework that decomposes global image quality into clinically defined radiological concepts using only volume-level supervision. To capture latent quality variations without entangling predefined concepts, our framework incorporates an unsupervised residual branch guided by an adversarial erasure mechanism to strictly prevent information leakage. Furthermore, we introduce a spatial diversity constraint that penalizes overlap between distinct concept attention maps, ensuring localized and interpretable feature extraction. Extensive experiments on a clinical dataset of atrial LGE-MRI volumes demonstrate that AC-MIL successfully opens the MIL black box, providing highly localized spatial concept maps that allow clinicians to pinpoint the specific causes of non-diagnostic scans. Crucially, our framework achieves this deep clinical transparency while maintaining highly competitive ordinal grading performance against existing baselines. Code to be released on acceptance.
>
---
#### [new 108] Active Diffusion Matching: Score-based Iterative Alignment of Cross-Modal Retinal Images
- **分类: cs.CV**

- **简介: 该论文属于跨模态图像对齐任务，旨在解决标准眼底图像与超广角眼底图像的对齐难题。提出ADM方法，通过扩散模型实现全局与局部对齐优化。**

- **链接: [https://arxiv.org/pdf/2604.10084](https://arxiv.org/pdf/2604.10084)**

> **作者:** Kanggeon Lee; Su Jeong Song; Soochahn Lee; Kyoung Mu Lee
>
> **摘要:** Objective: The study aims to address the challenge of aligning Standard Fundus Images (SFIs) and Ultra-Widefield Fundus Images (UWFIs), which is difficult due to their substantial differences in viewing range and the amorphous appearance of the retina. Currently, no specialized method exists for this task, and existing image alignment techniques lack accuracy. Methods: We propose Active Diffusion Matching (ADM), a novel cross-modal alignment method. ADM integrates two interdependent score-based diffusion models to jointly estimate global transformations and local deformations via an iterative Langevin Markov chain. This approach facilitates a stochastic, progressive search for optimal alignment. Additionally, custom sampling strategies are introduced to enhance the adaptability of ADM to given input image pairs. Results: Comparative experimental evaluations demonstrate that ADM achieves state-of-the-art alignment accuracy. This was validated on two datasets: a private dataset of SFI-UWFI pairs and a public dataset of SFI-SFI pairs, with mAUC improvements of 5.2 and 0.4 points on the private and public datasets, respectively, compared to existing state-of-the-art methods. Conclusion: ADM effectively bridges the gap in aligning SFIs and UWFIs, providing an innovative solution to a previously unaddressed challenge. The method's ability to jointly optimize global and local alignment makes it highly effective for cross-modal image alignment tasks. Significance: ADM has the potential to transform the integrated analysis of SFIs and UWFIs, enabling better clinical utility and supporting learning-based image enhancements. This advancement could significantly improve diagnostic accuracy and patient outcomes in ophthalmology.
>
---
#### [new 109] Boxes2Pixels: Learning Defect Segmentation from Noisy SAM Masks
- **分类: cs.CV**

- **简介: 该论文属于缺陷分割任务，旨在解决工业检测中缺乏精确标注的问题。通过构建Boxes2Pixels框架，利用噪声SAM掩码进行模型训练，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.11162](https://arxiv.org/pdf/2604.11162)**

> **作者:** Camile Lendering; Erkut Akdag; Egor Bondarev
>
> **备注:** Accepted for presentation at the AI4RWC Workshop at CVPR 2026
>
> **摘要:** Accurate defect segmentation is critical for industrial inspection, yet dense pixel-level annotations are rarely available. A common workaround is to convert inexpensive bounding boxes into pseudo-masks using foundation segmentation models such as the Segment Anything Model (SAM). However, these pseudo-labels are systematically noisy on industrial surfaces, often hallucinating background structure while missing sparse defects. To address this limitation, a noise-robust box-to-pixel distillation framework, Boxes2Pixels, is proposed that treats SAM as a noisy teacher rather than a source of ground-truth supervision. Bounding boxes are converted into pseudo-masks offline by SAM, and a compact student is trained with (i) a hierarchical decoder over frozen DINOv2 features for semantic stability, (ii) an auxiliary binary localization head to decouple sparse foreground discovery from class prediction, and (iii) a one-sided online self-correction mechanism that relaxes background supervision when the student is confident, targeting teacher false negatives. On a manually annotated wind turbine inspection benchmark, the proposed Boxes2Pixels improves anomaly mIoU by +6.97 and binary IoU by +9.71 over the strongest baseline trained under identical weak supervision. Moreover, online self-correction increases the binary recall by +18.56, while the model employs 80\% fewer trainable parameters. Code is available at this https URL.
>
---
#### [new 110] Prints in the Magnetic Dust: Robust Similarity Search in Legacy Media Images Using Checksum Count Vectors
- **分类: cs.CV; cs.HC; eess.IV**

- **简介: 该论文属于历史数字文物处理任务，旨在通过Checksum Count Vectors实现磁性介质图像的相似性搜索，解决重复与变体检测问题。**

- **链接: [https://arxiv.org/pdf/2604.09657](https://arxiv.org/pdf/2604.09657)**

> **作者:** Maciej Grzeszczuk; Kinga Skorupska; Grzegorz M. Wójcik
>
> **备注:** 10 pages, 6 figures. Peer-reviewed, presented on Machine Intelligence and Digital Interaction (MIDI) Conference on 11 december 2025 in Warsaw, POLAND. To be included in the proceedings (print in progress)
>
> **摘要:** Digitizing magnetic media containing computer data is only the first step towards the preservation of early home computing era artifacts. The audio tape images must be decoded, verified, repaired if necessary, tested, and documented. If parts of this process could be effectively automated, volunteers could focus on contributing contextual and historical knowledge rather than struggling with technical tools. We therefore propose a feature representation based on Checksum Count Vectors and evaluate its applicability to detecting duplicates and variants of recordings within a large data store. The approach was tested on a collection of decoded tape images (n=4902), achieving 58\% accuracy in detecting variants and 97% accuracy in identifying alternative copies, for damaged recordings with up to 75% of records missing. These results represent an important step towards fully automated pipelines for restoration, de-duplication, and semantic integration of historical digital artifacts through sequence matching, automatic repair and knowledge discovery.
>
---
#### [new 111] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于异常检测任务，旨在解决通用异常检测问题。针对多模态大语言模型在该任务中的不足，提出MMR-AD数据集和Anomaly-R1基线模型，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.10971](https://arxiv.org/pdf/2604.10971)**

> **作者:** Xincheng Yao; Zefeng Qian; Chao Shi; Jiayang Song; Chongyang Zhang
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** In the progress of industrial anomaly detection, general anomaly detection (GAD) is an emerging trend and also the ultimate goal. Unlike the conventional single- and multi-class AD, general AD aims to train a general AD model that can directly detect anomalies in diverse novel classes without any retraining or fine-tuning on the target data. Recently, Multimodal Large Language Models (MLLMs) have shown great promise in achieving general anomaly detection due to their revolutionary visual understanding and language reasoning capabilities. However, MLLM's general AD ability remains underexplored due to: (1) MLLMs are pretrained on amounts of data sourced from the Web, these data still have significant gaps with the data in AD scenarios. Moreover, the image-text pairs during pretraining are also not specifically for AD tasks. (2) The current mainstream AD datasets are image-based and not yet suitable for post-training MLLMs. To facilitate MLLM-based general AD research, we present MMR-AD, which is a comprehensive benchmark for both training and evaluating MLLM-based AD models. With MMR-AD, we reveal that the AD performance of current SOTA generalist MLLMs still falls far behind the industrial requirements. Based on MMR-AD, we also propose a baseline model, Anomaly-R1, which is a reasoning-based AD model that learns from the CoT data in MMR-AD and is further enhanced by reinforcement learning. Extensive experiments show that our Anomaly-R1 achieves remarkable improvements over generalist MLLMs in both anomaly detection and localization.
>
---
#### [new 112] Bridging the RGB-IR Gap: Consensus and Discrepancy Modeling for Text-Guided Multispectral Detection
- **分类: cs.CV**

- **简介: 该论文属于多光谱目标检测任务，旨在解决RGB与IR图像在语义对齐和特征融合中的差异问题。通过构建语义桥梁和双向对齐模块，提升跨模态检测性能。**

- **链接: [https://arxiv.org/pdf/2604.11234](https://arxiv.org/pdf/2604.11234)**

> **作者:** Jiaqi Wu; Zhen Wang; Enhao Huang; Kangqing Shen; Yulin Wang; Yang Yue; Yifan Pu; Gao Huang
>
> **备注:** 17 pages ,Under review
>
> **摘要:** Text-guided multispectral object detection uses text semantics to guide semantic-aware cross-modal interaction between RGB and IR for more robust perception. However, notable limitations remain: (1) existing methods often use text only as an auxiliary semantic enhancement signal, without exploiting its guiding role to bridge the inherent granularity asymmetry between RGB and IR; and (2) conventional data-driven attention-based fusion tends to emphasize stable consensus while overlooking potentially valuable cross-modal discrepancies. To address these issues, we propose a semantic bridge fusion framework with bi-support modeling for multispectral object detection. Specifically, text is used as a shared semantic bridge to align RGB and IR responses under a unified category condition, while the recalibrated thermal semantic prior is projected onto the RGB branch for semantic-level mapping fusion. We further formulate RGB-IR interaction evidence into the regular consensus support and the complementary discrepancy support that contains potentially discriminative cues, and introduce them into fusion via dynamic recalibration as a structured inductive bias. In addition, we design a bidirectional semantic alignment module for closed-loop vision-text guidance enhancement. Extensive experiments demonstrate the effectiveness of the proposed fusion framework and its superior detection performance on multispectral benchmarks. Code is available at this https URL.
>
---
#### [new 113] NTIRE 2026 Challenge on Single Image Reflection Removal in the Wild: Datasets, Results, and Methods
- **分类: cs.CV**

- **简介: 该论文聚焦单图像反射去除任务，旨在解决真实场景下图像中反射问题。通过构建OpenRR-5k数据集，推动实际应用中的反射去除技术发展。**

- **链接: [https://arxiv.org/pdf/2604.10321](https://arxiv.org/pdf/2604.10321)**

> **作者:** Jie Cai; Kangning Yang; Zhiyuan Li; Florin-Alexandru Vasluianu; Radu Timofte; Jinlong Li; Jinglin Shen; Zibo Meng; Junyan Cao; Lu Zhao; Pengwei Liu; Yuyi Zhang; Fengjun Guo; Jiagao Hu; Zepeng Wang; Fei Wang; Daiguo Zhou; Yi'ang Chen; Honghui Zhu; Mengru Yang; Yan Luo; Kui Jiang; Jin Guo; Jonghyuk Park; Jae-Young Sim; Wei Zhou; Hongyu Huang; Linfeng Li; Lindong Kong; Saiprasad Meesiyawar; Misbha Falak Khanpagadi; Nikhil Akalwadi; Ramesh Ashok Tabib; Uma Mudenagudi; Bilel Benjdira; Anas M. Ali; Wadii Boulila; Kosuke Shigematsu; Hiroto Shirono; Asuka Shin; Guoyi Xu; Yaoxin Jiang; Jiajia Liu; Yaokun Shi; Jiachen Tu; Shreeniketh Joshi; Jin-Hui Jiang; Yu-Fan Lin; Yu-Jou Hsiao; Chia-Ming Lee; Fu-En Yang; Yu-Chiang Frank Wang; Chih-Chung Hsu
>
> **摘要:** In this paper, we review the NTIRE 2026 challenge on single-image reflection removal (SIRR) in the Wild. SIRR is a fundamental task in image restoration. Despite progress in academic research, most methods are tested on synthetic images or limited real-world images, creating a gap in real-world applications. In this challenge, we provide participants with the OpenRR-5k dataset, which requires them to process real-world images that cover a range of reflection scenarios and intensities, with the goal of generating clean images without reflections. The challenge attracted more than 100 registrations, with 11 of them participating in the final testing phase. The top-ranked methods advanced the state-of-the-art reflection removal performance and earned unanimous recognition from the five experts in the field. The proposed OpenRR-5k dataset is available at this https URL, and the homepage of this challenge is at this https URL. Due to page limitations, this article only presents partial content; the full report and detailed analyses are available in the extended arXiv version.
>
---
#### [new 114] Language Prompt vs. Image Enhancement: Boosting Object Detection With CLIP in Hazy Environments
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决雾霾环境下物体检测困难的问题。通过引入语言提示增强语义，提出CLIP-CE和FAME方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.10637](https://arxiv.org/pdf/2604.10637)**

> **作者:** Jian Pang; Bingfeng Zhang; Jin Wang; Baodi Liu; Dapeng Tao; Weifeng Liu
>
> **摘要:** Object detection in hazy environments is challenging because degraded objects are nearly invisible and their semantics are weakened by environmental noise, making it difficult for detectors to identify. Common approaches involve image enhancement to boost weakened semantics, but these methods are limited by the instability of enhanced modules. This paper proposes a novel solution by employing language prompts to enhance weakened semantics without image enhancement. Specifically, we design Approximation of Mutual Exclusion (AME) to provide credible weights for Cross-Entropy Loss, resulting in CLIP-guided Cross-Entropy Loss (CLIP-CE). The provided weights assess the semantic weakening of objects. Through the backpropagation of CLIP-CE, weakened semantics are enhanced, making degraded objects easier to detect. In addition, we present Fine-tuned AME (FAME) which adaptively fine-tunes the weight of AME based on the predicted confidence. The proposed FAME compensates for the imbalanced optimization in AME. Furthermore, we present HazyCOCO, a large-scale synthetic hazy dataset comprising 61258 images. Experimental results demonstrate that our method achieves state-of-the-art performance. The code and dataset will be released.
>
---
#### [new 115] SignReasoner: Compositional Reasoning for Complex Traffic Sign Understanding via Functional Structure Units
- **分类: cs.CV**

- **简介: 该论文提出SignReasoner，解决复杂交通标志理解中的组合泛化问题。通过功能结构单元（FSU）分解标志，提升模型在新配置下的表现。**

- **链接: [https://arxiv.org/pdf/2604.10436](https://arxiv.org/pdf/2604.10436)**

> **作者:** Ruibin Wang; Zhenyu Lin; Xinhai Zhao
>
> **备注:** CVPRF 2026
>
> **摘要:** Accurate semantic understanding of complex traffic signs-including those with intricate layouts, multi-lingual text, and composite symbols-is critical for autonomous driving safety. Current models, both specialized small ones and large Vision Language Models (VLMs), suffer from a significant bottleneck: a lack of compositional generalization, leading to failure when encountering novel sign configurations. To overcome this, we propose SignReasoner, a novel paradigm that transforms general VLMs into expert traffic sign reasoners. Our core innovation is Functional Structure Unit (FSU), which shifts from common instance-based modeling to flexible function-based decomposition. By breaking down complex signs into minimal, core functional blocks (e.g., Direction, Notice, Lane), our model learns the underlying structural grammar, enabling robust generalization to unseen compositions. We define this decomposition as the FSU-Reasoning task and introduce a two-stage VLM post-training pipeline to maximize performance: Iterative Caption-FSU Distillation that enhances the model's accuracy in both FSU-reasoning and caption generation; FSU-GRPO that uses Tree Edit Distance (TED) to compute FSU differences as the rewards in GRPO algorithm, boosting reasoning abilities. Experiments on the newly proposed FSU-Reasoning benchmark, TrafficSignEval, show that SignReasoner achieves new SOTA with remarkable data efficiency and no architectural modification, significantly improving the traffic sign understanding in various VLMs.
>
---
#### [new 116] Empowering Video Translation using Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频翻译任务，旨在解决传统方法的局限性，通过多模态大语言模型提升翻译质量与鲁棒性。工作包括构建三角色分类框架并分析相关技术挑战。**

- **链接: [https://arxiv.org/pdf/2604.11283](https://arxiv.org/pdf/2604.11283)**

> **作者:** Bingzheng QU; Kehai Chen; Xuefeng Bai; Min Zhang
>
> **摘要:** Recent developments in video translation have further enhanced cross-lingual access to video content, with multimodal large language models (MLLMs) playing an increasingly important supporting role. With strong multimodal understanding, reasoning, and generation capabilities, MLLMs-based video translation systems are overcoming the limitations of traditional cascaded pipelines that separately handle automatic speech recognition, machine translation, text-to-speech and lip synchronization. These MLLM-powered approaches not only achieve competitive or superior translation quality, but also demonstrate stronger robustness in zero-shot settings and multi-speaker scenarios, while jointly modeling semantic fidelity, timing, speaker identity, and emotional consistency. However, despite the rapid progress of MLLMs and extensive surveys on general video-language understanding, a focused and systematic review of how MLLMs empower video translation tasks is still lacking. To fill this gap, we provide the first comprehensive overview of MLLMs-based video translation, organized around a three-role taxonomy: 1) Semantic Reasoner, which characterizes how MLLMs perform video understanding, temporal reasoning, and multimodal fusion; 2) Expressive Performer, which analyzes LLM-driven and LLM-augmented techniques for expressive, controllable speech generation; and 3) Visual Synthesizer, which examines different types of video generators for high-fidelity lip-sync and visual alignment. Finally, we discuss open challenges in video understanding, temporal modeling, and multimodal alignment, and outline promising future research directions for MLLMs-powered video translation.
>
---
#### [new 117] PERCEPT-Net: A Perceptual Loss Driven Framework for Reducing MRI Artifact Tissue Confusion
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决MRI中运动伪影与组织混淆的问题。通过引入PERCEPT-Net框架，利用感知损失提升伪影抑制效果，同时保持解剖结构完整性。**

- **链接: [https://arxiv.org/pdf/2604.10439](https://arxiv.org/pdf/2604.10439)**

> **作者:** Ziheng Guo; Danqun Zheng; Chengwei Chen; Boyang Pan; Shuai Li; Ziqin Yu; Xiaoxiao Chen; Langdi Zhong; Yun Bian; Nan-Jie Gong
>
> **备注:** 18 pages, 7 figures, 6 tables. Submitted to Medical Physics. Code available upon request
>
> **摘要:** Purpose: Existing deep learning-based MRI artifact correction models exhibit poor clinical generalization due to inherent artifact-tissue confusion, failing to discriminate artifacts from anatomical structures. To resolve this, we introduce PERCEPT-Net, a framework leveraging dedicated perceptual supervision for structure-preserving artifact suppression. Method: PERCEPT-Net utilizes a residual U-Net backbone integrated with a multi-scale recovery module and dual attention mechanisms to preserve anatomical context and salient features. The core mechanism, Motion Perceptual Loss (MPL), provides artifact-aware supervision by learning generalizable motion artifact representations. This logic directly guides the network to suppress artifacts while maintaining anatomical fidelity. Training utilized a hybrid dataset of real and simulated sequences, followed by prospective validation via objective metrics and expert radiologist assessments. Result: PERCEPT-Net outperformed state-of-the-art methods on clinical data. Ablation analysis established a direct causal link between MPL and performance; its omission caused a significant deterioration in structural consistency (p < 0.001) and tissue contrast (p < 0.001). Radiologist evaluations corroborated these objective metrics, scoring PERCEPT-Net significantly higher in global image quality (median 3 vs. 2, p < 0.001) and verifying the preservation of critical diagnostic structures. Conclusion: By integrating task-specific, artifact-aware perceptual learning, PERCEPT-Net suppresses motion artifacts in clinical MRI without compromising anatomical integrity. This framework improves clinical robustness and provides a verifiable mechanism to mitigate over-smoothing and structural degradation in medical image reconstruction.
>
---
#### [new 118] Towards Automated Solar Panel Integrity: Hybrid Deep Feature Extraction for Advanced Surface Defect Identification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于太阳能板缺陷检测任务，旨在解决人工检测效率低、成本高的问题。通过融合手工特征与深度学习特征，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.10969](https://arxiv.org/pdf/2604.10969)**

> **作者:** Muhammad Junaid Asif; Muhammad Saad Rafaqat; Usman Nazakat; Uzair Khan; Rana Fayyaz Ahmad
>
> **摘要:** To ensure energy efficiency and reliable operations, it is essential to monitor solar panels in generation plants to detect defects. It is quite labor-intensive, time consuming and costly to manually monitor large-scale solar plants and those installed in remote areas. Manual inspection may also be susceptible to human errors. Consequently, it is necessary to create an automated, intelligent defect-detection system, that ensures continuous monitoring, early fault detection, and maximum power generation. We proposed a novel hybrid method for defect detection in SOLAR plates by combining both handcrafted and deep learning features. Local Binary Pattern (LBP), Histogram of Gradients (HoG) and Gabor Filters were used for the extraction of handcrafted features. Deep features extracted by leveraging the use of DenseNet-169. Both handcrafted and deep features were concatenated and then fed to three distinct types of classifiers, including Support Vector Machines (SVM), Extreme Gradient Boost (XGBoost) and Light Gradient-Boosting Machine (LGBM). Experimental results evaluated on the augmented dataset show the superior performance, especially DenseNet-169 + Gabor (SVM), had the highest scores with 99.17% accuracy which was higher than all the other systems. In general, the proposed hybrid framework offers better defect-detection accuracy, resistance, and flexibility that has a solid basis on the real-life use of the automated PV panels monitoring system.
>
---
#### [new 119] Lung Cancer Detection Using Deep Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于肺癌检测任务，旨在通过深度学习算法提高早期诊断准确性。研究比较了多种CNN模型，并提出一种新型16层CNN模型以提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.10765](https://arxiv.org/pdf/2604.10765)**

> **作者:** Imama Ajmi; Abhishek Das
>
> **备注:** 8 pages
>
> **摘要:** Lung cancer, the second leading cause of cancer-related deaths, is primarily linked to long-term tobacco smoking (85% of cases). Surprisingly, 10-15% of cases occur in non-smokers. In 2020, approximately 2 million people were affected globally, resulting in 1.5 million deaths. The survival rate, at around 20%, lags behind other cancers, partly due to late-stage symptom manifestation. Necessitates early and accurate detection for effective treatment. Performance metrics such as accuracy, precision, recall (sensitivity), and F1-score are computed to provide a comprehensive evaluation of each model's capabilities. By comparing these metrics, this study offers insights into the strengths and limitations of each approach, contributing to the advancement of lung cancer detection techniques. In this paper, we are going to discuss the methodologies of lung cancer detection using different deep learning algorithms - InceptionV3, MobileNetV2, VGG16, ResNet152 - are explored for their efficacy in classifying lung cancer cases. Our Proposed Model algorithm based is a 16 layers architecture based on CNN model. Our Proposed model exhibits several key highlights that contribute to its novelty. By integrating multiple layer types such as convolutional, pooling, flatten, dropout, fully connected and dense layers, the model leverages the strengths of each layer to enhance its predictive capabilities. Novelty of our proposed model is that its accuracy is increasing consistently with the increasing no of epochs. We have tested the model performance up to epoch no 30. Our proposed model also overcome the overfitting problem.
>
---
#### [new 120] Analytical Modeling and Correction of Distance Error in Homography-Based Ground-Plane Mapping
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，解决单目相机距离估计中的误差问题。通过分析同源变换扰动与距离误差的关系，提出两种校正方法，提升地面平面映射精度。**

- **链接: [https://arxiv.org/pdf/2604.10805](https://arxiv.org/pdf/2604.10805)**

> **作者:** Mateusz Szulc; Marcin Iwanowski
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Accurate distance estimation from monocular cameras is essential for intelligent monitoring systems. In many deployments, image coordinates are mapped to ground positions using planar homographies initialized by manual selection of corresponding regions. Small inaccuracies in this initialization propagate into systematic distance distortions. This paper derives an explicit relationship between homography perturbations and the resulting distance error, showing that the error grows approximately quadratically with the true distance from the camera. Based on this model, two simple correction strategies are evaluated: regression-based estimation of the quadratic error function and direct optimization of the homography via coordinate-based gradient descent. A large-scale simulation study with more than 19 million test samples demonstrates that regression achieves higher peak accuracy when the model is reliably fitted, whereas gradient descent provides greater robustness against poor initial calibration. This suggests that improving geometric calibration may yield greater performance gains than increasing model complexity in many practical systems.
>
---
#### [new 121] Precision Synthesis of Multi-Tracer PET via VLM-Modulated Rectified Flow for Stratifying Mild Cognitive Impairment
- **分类: cs.CV**

- **简介: 该论文属于医学影像合成任务，旨在解决PET成本高、辐射大的问题。通过DIReCT$++$模型，从MRI和临床信息生成多示踪剂PET，提升AD早期诊断效率。**

- **链接: [https://arxiv.org/pdf/2604.11176](https://arxiv.org/pdf/2604.11176)**

> **作者:** Tuo Liu; Shuijin Lin; Shaozhen Yan; Haifeng Wang; Jie Lu; Jianhua Ma; Chunfeng Lian
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** The biological definition of Alzheimer's disease (AD) relies on multi-modal neuroimaging, yet the clinical utility of positron emission tomography (PET) is limited by cost and radiation exposure, hindering early screening at preclinical or prodromal stages. While generative models offer a promising alternative by synthesizing PET from magnetic resonance imaging (MRI), achieving subject-specific precision remains a primary challenge. Here, we introduce DIReCT$++$, a Domain-Informed ReCTified flow model for synthesizing multi-tracer PET from MRI combined with fundamental clinical information. Our approach integrates a 3D rectified flow architecture to capture complex cross-modal and cross-tracer relationships with a domain-adapted vision-language model (BiomedCLIP) that provides text-guided, personalized generation using clinical scores and imaging knowledge. Extensive evaluations on multi-center datasets demonstrate that DIReCT$++$ not only produces synthetic PET images ($^{18}$F-AV-45 and $^{18}$F-FDG) of superior fidelity and generalizability but also accurately recapitulates disease-specific patterns. Crucially, combining these synthesized PET images with MRI enables precise personalized stratification of mild cognitive impairment (MCI), advancing a scalable, data-efficient tool for the early diagnosis and prognostic prediction of AD. The source code will be released on this https URL.
>
---
#### [new 122] Context Matters: Vision-Based Depression Detection Comparing Classical and Deep Approaches
- **分类: cs.CV**

- **简介: 该论文属于抑郁检测任务，比较经典与深度学习方法在视觉模态中的表现，探讨其准确性、公平性和跨情境泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.10344](https://arxiv.org/pdf/2604.10344)**

> **作者:** Maneesh Bilalpur; Saurabh Hinduja; Sonish Sivarajkumar; Nicholas Allen; Yanshan Wang; Itir Onal Ertugrul; Jeffrey F. Cohn
>
> **摘要:** The classical approach to detecting depression from vision emphasizes interpretable features, such as facial expression, and classifiers such as the Support Vector Machine (SVM). With the advent of deep learning, there has been a shift in feature representations and classification approaches. Contemporary approaches use learnt features from general-purpose vision models such as VGGNet to train machine learning models. Little is known about how classical and deep approaches compare in depression detection with respect to accuracy, fairness, and generalizability, especially across contexts. To address these questions, we compared classical and deep approaches to the detection of depression in the visual modality in two different contexts: Mother-child interactions in the TPOT database and patient-clinician interviews in the Pitt database. In the former, depression was operationalized as a history of depression per the DSM and current or recent clinically significant symptoms. In the latter, all participants met initial criteria for depression per DSM, and depression was reassessed over the course of treatment. The classical approach included handcrafted features with SVM classifiers. Learnt features were turn-level embeddings from the FMAE-IAT that were combined with Multi-Layer Perceptron classifiers. The classical approach achieved higher accuracy in both contexts. It was also significantly fairer than the deep approach in the patient-clinician context. Cross-context generalizability was modest at best for both approaches, which suggests that depression may be context-specific.
>
---
#### [new 123] FGML-DG: Feynman-Inspired Cognitive Science Paradigm for Cross-Domain Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于跨域医学图像分割任务，旨在解决领域泛化问题。通过引入认知科学启发的元学习框架，提升模型在不同数据源上的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.10524](https://arxiv.org/pdf/2604.10524)**

> **作者:** Yucheng Song; Chenxi Li; Haokang Ding; Zhining Liao; Zhifang Liao
>
> **摘要:** In medical image segmentation across multiple modalities (e.g., MRI, CT, etc.) and heterogeneous data sources (e.g., different hospitals and devices), Domain Generalization (DG) remains a critical challenge in AI-driven healthcare. This challenge primarily arises from domain shifts, imaging variations, and patient diversity, which often lead to degraded model performance in unseen domains. To address these limitations, we identify key issues in existing methods, including insufficient simplification of complex style features, inadequate reuse of domain knowledge, and a lack of feedback-driven optimization. To tackle these problems, inspired by Feynman's learning techniques in educational psychology, this paper introduces a cognitive science-inspired meta-learning paradigm for medical image domain generalization segmentation. We propose, for the first time, a cognitive-inspired Feynman-Guided Meta-Learning framework for medical image domain generalization segmentation (FGML-DG), which mimics human cognitive learning processes to enhance model learning and knowledge transfer. Specifically, we first leverage the 'concept understanding' principle from Feynman's learning method to simplify complex features across domains into style information statistics, achieving precise style feature alignment. Second, we design a meta-style memory and recall method (MetaStyle) to emulate the human memory system's utilization of past knowledge. Finally, we incorporate a Feedback-Driven Re-Training strategy (FDRT), which mimics Feynman's emphasis on targeted relearning, enabling the model to dynamically adjust learning focus based on prediction errors. Experimental results demonstrate that our method outperforms other existing domain generalization approaches on two challenging medical image domain generalization tasks.
>
---
#### [new 124] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于个性化图像美学评估任务，旨在解决VLM是否编码多层级美学属性的问题。通过分析VLM内部表示，发现其可有效支持个性化评估，并验证简单模型的可行性。**

- **链接: [https://arxiv.org/pdf/2604.11374](https://arxiv.org/pdf/2604.11374)**

> **作者:** Koki Ryu; Hitomi Yanaka
>
> **备注:** To appear at ACL 2026 findings
>
> **摘要:** Personalized image aesthetics assessment (PIAA) is an important research problem with practical real-world applications. While methods based on vision-language models (VLMs) are promising candidates for PIAA, it remains unclear whether they internally encode rich, multi-level aesthetic attributes required for effective personalization. In this paper, we first analyze the internal representations of VLMs to examine the presence and distribution of such aesthetic attributes, and then leverage them for lightweight, individual-level personalization without model fine-tuning. Our analysis reveals that VLMs encode diverse aesthetic attributes that propagate into the language decoder layers. Building on these representations, we demonstrate that simple linear models can perform PIAA effectively. We further analyze how aesthetic information is transferred across layers in different VLM architectures and across image domains. Our findings provide insights into how VLMs can be utilized for modeling subjective, individual aesthetic preferences. Our code is available at this https URL.
>
---
#### [new 125] EditCrafter: Tuning-free High-Resolution Image Editing via Pretrained Diffusion Model
- **分类: cs.CV**

- **简介: 论文提出EditCrafter，用于高分辨率图像编辑任务，解决现有方法无法处理任意比例和更高分辨率图像的问题。通过无调优的扩散模型实现高质量编辑。**

- **链接: [https://arxiv.org/pdf/2604.10268](https://arxiv.org/pdf/2604.10268)**

> **作者:** Kunho Kim; Sumin Seo; Yongjun Cho; Hyungjin Chung
>
> **备注:** Accepted to CVPRW 2026 Proceeding Track. Project page: this https URL
>
> **摘要:** We propose EditCrafter, a high-resolution image editing method that operates without tuning, leveraging pretrained text-to-image (T2I) diffusion models to process images at resolutions significantly exceeding those used during training. Leveraging the generative priors of large-scale T2I diffusion models enables the development of a wide array of novel generation and editing applications. Although numerous image editing methods have been proposed based on diffusion models and exhibit high-quality editing results, they are difficult to apply to images with arbitrary aspect ratios or higher resolutions since they only work at the training resolutions (512x512 or 1024x1024). Naively applying patch-wise editing fails with unrealistic object structures and repetition. To address these challenges, we introduce EditCrafter, a simple yet effective editing pipeline. EditCrafter operates by first performing tiled inversion, which preserves the original identity of the input high-resolution image. We further propose a noise-damped manifold-constrained classifier-free guidance (NDCFG++) that is tailored for high resolution image editing from the inverted latent. Our experiments show that the our EditCrafter can achieve impressive editing results across various resolutions without fine-tuning and optimization.
>
---
#### [new 126] Intra-finger Variability of Diffusion-based Latent Fingerprint Generation
- **分类: cs.CV**

- **简介: 该论文属于指纹生成任务，旨在解决合成指纹的多样性与身份一致性问题。通过构建风格库和分析生成指纹的细节，发现模型存在局部和全局不一致现象。**

- **链接: [https://arxiv.org/pdf/2604.10040](https://arxiv.org/pdf/2604.10040)**

> **作者:** Noor Hussein; Anil K. Jain; Karthik Nandakumar
>
> **备注:** Accepted at the 2nd Workshop on Foundation and Generative Models in Biometrics (FoundGen-Bio), held in conjunction with CVPR 2026
>
> **摘要:** The primary goal of this work is to systematically evaluate the intra-finger variability of synthetic fingerprints (particularly latent prints) generated using a state-of-the-art diffusion model. Specifically, we focus on enhancing the latent style diversity of the generative model by constructing a comprehensive \textit{latent style bank} curated from seven diverse datasets, which enables the precise synthesis of latent prints with over 40 distinct styles encapsulating different surfaces and processing techniques. We also implement a semi-automated framework to understand the integrity of fingerprint ridges and minutiae in the generated impressions. Our analysis indicates that though the generation process largely preserves the identity, a small number of local inconsistencies (addition and removal of minutiae) are introduced, especially when there are poor quality regions in the reference image. Furthermore, mismatch between the reference image and the chosen style embedding that guides the generation process introduces global inconsistencies in the form of hallucinated ridge patterns. These insights highlight the limitations of existing synthetic fingerprint generators and the need to further improve these models to simultaneously enhance both diversity and identity consistency.
>
---
#### [new 127] On the Robustness of Watermarking for Autoregressive Image Generation
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文研究AR图像生成的水印鲁棒性，旨在检测和归属合成图像。工作揭示现有水印方案易受攻击，无法可靠用于数据集过滤，并提出三种新攻击方法。**

- **链接: [https://arxiv.org/pdf/2604.11720](https://arxiv.org/pdf/2604.11720)**

> **作者:** Andreas Müller; Denis Lukovnikov; Shingo Kodama; Minh Pham; Anubhav Jain; Jonathan Petit; Niv Cohen; Asja Fischer
>
> **摘要:** The proliferation of autoregressive (AR) image generators demands reliable detection and attribution of their outputs to mitigate misinformation, and to filter synthetic images from training data to prevent model collapse. To address this need, watermarking techniques, specifically designed for AR models, embed a subtle signal at generation time, enabling downstream verification through a corresponding watermark detector. In this work, we study these schemes and demonstrate their vulnerability to both watermark removal and forgery attacks. We assess existing attacks and further introduce three new attacks: (i) a vector-quantized regeneration removal attack, (ii) adversarial optimization-based attack, and (iii) a frequency injection attack. Our evaluation reveals that removal and forgery attacks can be effective with access to a single watermarked reference image and without access to original model parameters or watermarking secrets. Our findings indicate that existing watermarking schemes for AR image generation do not reliably support synthetic content detection for dataset filtering. Moreover, they enable Watermark Mimicry, whereby authentic images can be manipulated to imitate a generator's watermark and trigger false detection to prevent their inclusion in future model training.
>
---
#### [new 128] Point2Pose: Occlusion-Recovering 6D Pose Tracking and 3D Reconstruction for Multiple Unknown Objects Via 2D Point Trackers
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Point2Pose，用于多未知物体的6D位姿跟踪与3D重建，解决单目RGB-D视频中遮挡恢复问题。**

- **链接: [https://arxiv.org/pdf/2604.10415](https://arxiv.org/pdf/2604.10415)**

> **作者:** Tzu-Yuan Lin; Ho Jae Lee; Kevin Doherty; Yonghyeon Lee; Sangbae Kim
>
> **摘要:** We present Point2Pose, a model-free method for causal 6D pose tracking of multiple rigid objects from monocular RGB-D video. Initialized only from sparse image points on the objects to be tracked, our approach tracks multiple unseen objects without requiring object CAD models or category priors. Point2Pose leverages a 2D point tracker to obtain long-range correspondences, enabling instant recovery after complete occlusion. Simultaneously, the system incrementally reconstructs an online Truncated Signed Distance Function (TSDF) representation of the tracked targets. Alongside the method, we introduce a new multi-object tracking dataset comprising both simulation and real-world sequences, with motion-capture ground truth for evaluation. Experiments show that Point2Pose achieves performance comparable to the state-of-the-art methods on a severe-occlusion benchmark, while additionally supporting multi-object tracking and recovery from complete occlusion, capabilities that are not supported by previous model-free tracking approaches.
>
---
#### [new 129] Adapting 2D Multi-Modal Large Language Model for 3D CT Image Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决3D医疗图像预训练不足和特征提取不适应的问题。通过迁移2D模型并引入TGH-MoE框架，提升医学报告生成和视觉问答性能。**

- **链接: [https://arxiv.org/pdf/2604.10233](https://arxiv.org/pdf/2604.10233)**

> **作者:** Yang Yu; Dunyuan Xu; Yaoqian Li; Xiaomeng Li; Jinpeng Li; Pheng-Ann Heng
>
> **摘要:** 3D medical image analysis is of great importance in disease diagnosis and treatment. Recently, multimodal large language models (MLLMs) have exhibited robust perceptual capacity, strong cross-modal alignment, and promising generalizability. Therefore, they have great potential to improve the performance of medical report generation (MRG) and medical visual question answering (MVQA), which serve as two important tasks in clinical scenarios. However, due to the scarcity of 3D medical images, existing 3D medical MLLMs suffer from insufficiently pretrained vision encoder and inability to extract customized image features for different kinds of tasks. In this paper, we propose to first transfer a 2D MLLM, which is well trained with 2D natural images, to support 3D medical volumetric inputs while reusing all of its pre-trained parameters. To enable the vision encoder to extract tailored image features for various tasks, we then design a Text-Guided Hierarchical MoE (TGH-MoE) framework, which can distinguish tasks under the guidance of the text prompt. Furthermore, we propose a two-stage training strategy to learn both task-shared and task-specific image features. As demonstrated empirically, our method outperforms existing 3D medical MLLMs in both MRG and MVQA tasks. Our code will be released once this paper is accepted.
>
---
#### [new 130] NTIRE 2026 Challenge on Short-form UGC Video Restoration in the Wild with Generative Models: Datasets, Methods and Results
- **分类: cs.CV**

- **简介: 该论文介绍NTIRE 2026挑战，聚焦于生成模型下的短格式UGC视频修复任务，提出KwaiVIR数据集，包含合成与真实视频，旨在提升复杂退化下视频修复效果。**

- **链接: [https://arxiv.org/pdf/2604.10551](https://arxiv.org/pdf/2604.10551)**

> **作者:** Xin Li; Jiachao Gong; Xijun Wang; Shiyao Xiong; Bingchen Li; Suhang Yao; Chao Zhou; Zhibo Chen; Radu Timofte; Yuxiang Chen; Shibo Yin; Yilian Zhong; Yushun Fang; Xilei Zhu; Yahui Wang; Chen Lu; Meisong Zheng; Xiaoxu Chen; Jing Yang; Zhaokun Hu; Jiahui Liu; Ying Chen; Haoran Bai; Sibin Deng; Shengxi Li; Mai Xu; Junyang Chen; Hao Chen; Xinzhe Zhu; Fengkai Zhang; Long Sun; Yixing Yang; Xindong Zhang; Jiangxin Dong; Jinshan Pan; Jiyuan Zhang; Shuai Liu; Yibin Huang; Xiaotao Wang; Lei Lei; Zhirui Liu; Shinan Chen; Shang-Quan Sun; Wenqi Ren; Jingyi Xu; Zihong Chen; Zhuoya Zou; Xiuhao Qiu; Jingyu Ma; Huiyuan Fu; Kun Liu; Huadong Ma; Dehao Feng; Zhijie Ma; Boqi Zhang; Jiawei Shi; Hao Kang; Yixin Yang; Yeying Jin; Xu Cheng; Yuxuan Jiang; Chengxi Zeng; Tianhao Peng; Fan Zhang; David Bull; Yanan Xing; Jiachen Tu; Guoyi Xu; Yaoxin Jiang; Jiajia Liu; Yaokun Shi; Wei Zhou; Linfeng Li; Hang Song; Qi Xu; Kun Yuan; Yizhen Shao; Yulin Ren
>
> **备注:** Accepted by CVPR 2026 workshop; NTIRE 2026
>
> **摘要:** This paper presents an overview of the NTIRE 2026 Challenge on Short-form UGC Video Restoration in the Wild with Generative Models. This challenge utilizes a new short-form UGC (S-UGC) video restoration benchmark, termed KwaiVIR, which is contributed by USTC and Kuaishou Technology. It contains both synthetically distorted videos and real-world short-form UGC videos in the wild. For this edition, the released data include 200 synthetic training videos, 48 wild training videos, 11 validation videos, and 20 testing videos. The primary goal of this challenge is to establish a strong and practical benchmark for restoring short-form UGC videos under complex real-world degradations, especially in the emerging paradigm of generative-model-based S-UGC video restoration. This challenge has two tracks: (i) the primary track is a subjective track, where the evaluation is based on a user study; (ii) the second track is an objective track. These two tracks enable a comprehensive assessment of restoration quality. In total, 95 teams have registered for this competition. And 12 teams submitted valid final solutions and fact sheets for the testing phase. The submitted methods achieved strong performance on the KwaiVIR benchmark, demonstrating encouraging progress in short-form UGC video restoration in the wild.
>
---
#### [new 131] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?
- **分类: cs.CV**

- **简介: 该论文研究跨模态SAR-光学卫星配准任务，评估预训练匹配器在无微调情况下的性能，探讨模型迁移效果与部署协议的影响。**

- **链接: [https://arxiv.org/pdf/2604.10217](https://arxiv.org/pdf/2604.10217)**

> **作者:** Isaac Corley; Alex Stoken; Gabriele Berton
>
> **摘要:** Cross-modal optical-SAR (Synthetic Aperture Radar) registration is a bottleneck for disaster-response via remote sensing, yet modern image matchers are developed and benchmarked almost exclusively on natural-image domains. We evaluate twenty-four pretrained matcher families--in a zero-shot setting with no fine-tuning or domain adaptation on satellite or SAR data--on SpaceNet9 and two additional cross-modal benchmarks under a deterministic protocol with tiled large-image inference, robust geometric filtering, and tie-point-grounded metrics. Our results reveal asymmetric transfer--matchers with explicit cross-modal training do not uniformly outperform those without it. While XoFTR (trained for visible-thermal matching) and RoMa achieve the lowest reported mean error at $3.0$ px on the labeled SpaceNet9 training scenes, RoMa achieves this without any cross-modal training, and MatchAnything-ELoFTR ($3.4$ px)--trained on synthetic cross-modal pairs--matches closely, suggesting (as a working hypothesis) that foundation-model features (DINOv2) may contribute to modality invariance that partially substitutes for explicit cross-modal supervision. 3D-reconstruction matchers (MASt3R, DUSt3R), which are not designed for traditional 2D image matching, are highly protocol-sensitive and remain fragile under default settings. Deployment protocol choices (geometry model, tile size, inlier gating) shift accuracy by up to $33\times$ for a single matcher, sometimes exceeding the effect of swapping matchers entirely within the evaluated sweep--affine geometry alone reduces mean error from $12.34$ to $9.74$ px. These findings inform both practical deployment of existing matchers and future matcher design for cross-modal satellite registration.
>
---
#### [new 132] LumiMotion: Improving Gaussian Relighting with Scene Dynamics
- **分类: cs.CV**

- **简介: 该论文属于3D重建中的逆渲染任务，旨在解决动态场景下光照与材质分离的问题。通过利用动态区域作为监督信号，提出LumiMotion方法，提升光照重渲染效果。**

- **链接: [https://arxiv.org/pdf/2604.10994](https://arxiv.org/pdf/2604.10994)**

> **作者:** Joanna Kaleta; Piotr Wójcik; Kacper Marzol; Tomasz Trzciński; Kacper Kania; Marek Kowalski
>
> **备注:** CVPR2026
>
> **摘要:** In 3D reconstruction, the problem of inverse rendering, namely recovering the illumination of the scene and the material properties, is fundamental. Existing Gaussian Splatting-based methods primarily target static scenes and often assume simplified or moderate lighting to avoid entangling shadows with surface appearance. This limits their ability to accurately separate lighting effects from material properties, particularly in real-world conditions. We address this limitation by leveraging dynamic elements - regions of the scene that undergo motion - as a supervisory signal for inverse rendering. Motion reveals the same surfaces under varying lighting conditions, providing stronger cues for disentangling material and illumination. This thesis is supported by our experimental results which show we improve LPIPS by 23% for albedo estimation and by 15% for scene relighting relative to next-best baseline. To this end, we introduce LumiMotion, the first Gaussian-based approach that leverages dynamics for inverse rendering and operates in arbitrary dynamic scenes. Our method learns a dynamic 2D Gaussian Splatting representation that employs a set of novel constraints which encourage the dynamic regions of the scene to deform, while keeping static regions stable. As we demonstrate, this separation is crucial for correct optimization of the albedo. Finally, we release a new synthetic benchmark comprising five scenes under four lighting conditions, each in both static and dynamic variants, for the first time enabling systematic evaluation of inverse rendering methods in dynamic environments and challenging lighting. Link to project page: this https URL
>
---
#### [new 133] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification
- **分类: cs.CV**

- **简介: 该论文属于音频-视觉问答任务，解决缺失模态下的性能下降问题。提出R²ScP框架，通过检索恢复和语义净化提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10695](https://arxiv.org/pdf/2604.10695)**

> **作者:** Jiayu Zhang; Shuo Ye; Qilang Ye; Zihan Song; Jiajian Huang; Zitong Yu
>
> **摘要:** Recent Audio-Visual Question Answering (AVQA) methods have advanced significantly. However, most AVQA methods lack effective mechanisms for handling missing modalities, suffering from severe performance degradation in real-world scenarios with data interruptions. Furthermore, prevailing methods for handling missing modalities predominantly rely on generative imputation to synthesize missing features. While partially effective, these methods tend to capture inter-modal commonalities but struggle to acquire unique, modality-specific knowledge within the missing data, leading to hallucinations and compromised reasoning accuracy. To tackle these challenges, we propose R$^{2}$ScP, a novel framework that shifts the paradigm of missing modality handling from traditional generative imputation to retrieval-based recovery. Specifically, we leverage cross-modal retrieval via unified semantic embeddings to acquire missing domain-specific knowledge. To maximize semantic restoration, we introduce a context-aware adaptive purification mechanism that eliminates latent semantic noise within the retrieved data. Additionally, we employ a two-stage training strategy to explicitly model the semantic relationships between knowledge from different sources. Extensive experiments demonstrate that R$^{2}$ScP significantly improves AVQA and enhances robustness in modal-incomplete scenarios.
>
---
#### [new 134] From Redaction to Restoration: Deep Learning for Medical Image Anonymization and Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像匿名化任务，旨在解决去标识化过程中影响后续分析的问题。通过深度学习框架实现图像去标识与重建，提升数据共享安全性与可用性。**

- **链接: [https://arxiv.org/pdf/2604.11376](https://arxiv.org/pdf/2604.11376)**

> **作者:** Adrienne Kline; Abhijit Gaonkar; Daniel Pittman; Chris Kuehn; Nils Forkert
>
> **摘要:** Removing patient-specific information from medical images is crucial to enable sharing and open science without compromising patient identities. However, many methods currently used for deidentification have negative effects on downstream image analysis tasks because of removal of relevant but non-identifiable information. This work presents an end-to-end deep learning framework for transforming raw clinical image volumes into de-identified, analysis-ready datasets without compromising downstream utility. The methodology developed and tested in this work first detects and redacts regions likely to contain protected health information (PHI), such as burned-in text and metadata, and then uses a generative deep learning model to inpaint the redacted areas with anatomically and imaging plausible content. The proposed pipeline leverages a lightweight hybrid architecture, combining CRNN-based redaction with a latent-diffusion inpainting restoration module (Stable Diffusion 2). We evaluate the approach using both privacy-oriented metrics, which quantify residual PHI and success of redaction, and image-quality and task-based metrics, which assess the fidelity of restored volumes for representative deep learning applications. Our results suggest that the proposed method yields de-identified medical images that are visually coherent, maintaining fidelity for downstream models, while substantially reducing the risk of patient re-identification. By automating anonymization and image reconstruction within a single workflow, and dissemination of large-scale medical imaging collections, thereby lowering a key barrier to data sharing and multi-institutional collaboration in medical imaging AI.
>
---
#### [new 135] SwinTextUNet: Integrating CLIP-Based Text Guidance into Swin Transformer U-Nets for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决传统模型在低对比度或模糊区域表现不佳的问题。通过融合CLIP文本嵌入与Swin Transformer U-Net，提升分割精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10000](https://arxiv.org/pdf/2604.10000)**

> **作者:** Ashfak Yeafi; Parthaw Goswami; Md Khairul Islam; Ashifa Islam Shamme
>
> **摘要:** Precise medical image segmentation is fundamental for enabling computer aided diagnosis and effective treatment planning. Traditional models that rely solely on visual features often struggle when confronted with ambiguous or low contrast patterns. To overcome these limitations, we introduce SwinTextUNet, a multimodal segmentation framework that incorporates Contrastive Language Image Pretraining (CLIP), derived textual embeddings into a Swin Transformer UNet backbone. By integrating cross attention and convolutional fusion, the model effectively aligns semantic text guidance with hierarchical visual representations, enhancing robustness and accuracy. We evaluate our approach on the QaTaCOV19 dataset, where the proposed four stage variant achieves an optimal balance between performance and complexity, yielding Dice and IoU scores of 86.47% and 78.2%, respectively. Ablation studies further validate the importance of text guidance and multimodal fusion. These findings underscore the promise of vision language integration in advancing medical image segmentation and supporting clinically meaningful diagnostic tools.
>
---
#### [new 136] Improving Deep Learning-Based Target Volume Auto-Delineation for Adaptive MR-Guided Radiotherapy in Head and Neck Cancer: Impact of a Volume-Aware Dice Loss
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升头颈癌自适应MRI引导放疗中的靶区自动勾画。通过引入体积感知的Dice损失函数，优化小淋巴结的检测，同时平衡主要肿瘤的分割精度。**

- **链接: [https://arxiv.org/pdf/2604.10130](https://arxiv.org/pdf/2604.10130)**

> **作者:** Sogand Beirami; Zahra Esmaeilzadeh; Ahmed Gomaa; Pluvio Stephan; Ishita Sheth; Thomas Weissmann; Juliane Szkitsak; Philipp Schubert; Yixing Huang; Annette Schwarz; Stefanie Corradini; Florian Putz
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Background: Manual delineation of target volumes in head and neck cancer (HNC) remains a significant bottleneck in radiotherapy planning, characterized by high inter-observer variability and time consumption. This study evaluates the integration of a Volume-Aware (VA) Dice loss function into a self-configuring deep learning framework to enhance the auto-segmentation of primary tumors (PT) and metastatic lymph nodes (LN) for adaptive MR-guided radiotherapy. We investigate how volume-sensitive weighting affects the detection of small, anatomically complex nodal metastases compared to conventional loss functions. Methods: Utilizing the HNTS-MRG 2024 dataset, we implemented an nnU-Net ResEnc M architecture. We conducted a multi-label segmentation task, comparing a standard Dice loss baseline against two Volume-Aware configurations: a "Dual Mask" setup (VA loss on both PT and LN) and a "Selective LN Mask" setup (VA loss on LN only). Evaluation metrics included volumetric Dice scores, surface-based metrics (SDS, MSD, HD95), and lesion-wise binary detection sensitivity and precision. Results: The Selective LN Mask configuration achieved the highest LN Volumetric Dice Score (0.758 vs. 0.734 baseline) and significantly improved LN Lesion-Wise Detection Sensitivity (84.93% vs. 81.80%). However, a critical trade-off was observed; PT detection precision declined significantly in the selective setup (63.65% vs. 81.27%). The Dual Mask configuration provided the most balanced performance across both targets, maintaining primary tumor precision at 82.04% while improving LN sensitivity to 83.46%. Conclusions: A volume-sensitive loss function mitigated the under-representation of small metastatic lesions in HNC. While selective weighting yielded the best nodal detection, a dual-mask approach is required in multi-label tasks to maintain segmentation accuracy for larger primary tumor volumes.
>
---
#### [new 137] From UAV Imagery to Agronomic Reasoning: A Multimodal LLM Benchmark for Plant Phenotyping
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于植物表型分析任务，旨在解决农业领域中多模态模型应用的挑战。工作包括构建PlantXpert基准，评估11个视觉语言模型在作物表型中的表现。**

- **链接: [https://arxiv.org/pdf/2604.09907](https://arxiv.org/pdf/2604.09907)**

> **作者:** Yu Wu; Guangzeng Han; Ibra Niang Niang; Francia Ravelombola; Maiara Oliveira; Jason Davis; Dong Chen; Feng Lin; Xiaolei Huang
>
> **备注:** In review
>
> **摘要:** To improve crop genetics, high-throughput, effective and comprehensive phenotyping is a critical prerequisite. While such tasks were traditionally performed manually, recent advances in multimodal foundation models, especially in vision-language models (VLMs), have enabled more automated and robust phenotypic analysis. However, plant science remains a particularly challenging domain for foundation models because it requires domain-specific knowledge, fine-grained visual interpretation, and complex biological and agronomic reasoning. To address this gap, we develop PlantXpert, an evidence-grounded multimodal reasoning benchmark for soybean and cotton phenotyping. Our benchmark provides a structured and reproducible framework for agronomic adaptation of VLMs, and enables controlled comparison between base models and their domain-adapted counterparts. We constructed a dataset comprising 385 digital images and more than 3,000 benchmark samples spanning key plant science domains including disease, pest control, weed management, and yield. The benchmark can assess diverse capabilities including visual expertise, quantitative reasoning, and multi-step agronomic reasoning. A total of 11 state-of-the-art VLMs were evaluated. The results indicate that task-specific fine-tuning leads to substantial improvement in accuracy, with models such as Qwen3-VL-4B and Qwen3-VL-30B achieving up to 78%. At the same time, gains from model scaling diminish beyond a certain capacity, generalization across soybean and cotton remains uneven, and quantitative as well as biologically grounded reasoning continue to pose substantial challenges. These findings suggest that PlantXpert can serve as a foundation for assessing evidence-grounded agronomic reasoning and for advancing multimodal model development in plant science.
>
---
#### [new 138] SyncFix: Fixing 3D Reconstructions via Multi-View Synchronization
- **分类: cs.CV**

- **简介: 该论文提出SyncFix，用于修复3D重建中的语义和几何不一致问题。属于3D重建任务，通过多视角同步优化提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.11797](https://arxiv.org/pdf/2604.11797)**

> **作者:** Deming Li; Abhay Yadav; Cheng Peng; Rama Chellappa; Anand Bhattad
>
> **摘要:** We present SyncFix, a framework that enforces cross-view consistency during the diffusion-based refinement of reconstructed scenes. SyncFix formulates refinement as a joint latent bridge matching problem, synchronizing distorted and clean representations across multiple views to fix the semantic and geometric inconsistencies. This means SyncFix learns a joint conditional over multiple views to enforce consistency throughout the denoising trajectory. Our training is done only on image pairs, but it generalizes naturally to an arbitrary number of views during inference. Moreover, reconstruction quality improves with additional views, with diminishing returns at higher view counts. Qualitative and quantitative results demonstrate that SyncFix consistently generates high-quality reconstructions and surpasses current state-of-the-art baselines, even in the absence of clean reference images. SyncFix achieves even higher fidelity when sparse references are available.
>
---
#### [new 139] ReXSonoVQA: A Video QA Benchmark for Procedure-Centric Ultrasound Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ReXSonoVQA，一个针对超声操作视频的问答基准，解决动态过程理解问题，旨在提升视觉语言模型在医疗影像中的应用。**

- **链接: [https://arxiv.org/pdf/2604.10916](https://arxiv.org/pdf/2604.10916)**

> **作者:** Xucheng Wang; Xiaoman Zhang; Sung Eun Kim; Ankit Pal; Pranav Rajpurkar
>
> **摘要:** Ultrasound acquisition requires skilled probe manipulation and real-time adjustments. Vision-language models (VLMs) could enable autonomous ultrasound systems, but existing benchmarks evaluate only static images, not dynamic procedural understanding. We introduce ReXSonoVQA, a video QA benchmark with 514 video clips and 514 questions (249 MCQ, 265 free-response) targeting three competencies: Action-Goal Reasoning, Artifact Resolution & Optimization, and Procedure Context & Planning. Zero-shot evaluation of Gemini 3 Pro, Qwen3.5-397B, LLaVA-Video-72B, and Seed 2.0 Pro shows VLMs can extract some procedural information, but troubleshooting questions remain challenging with minimal gains over text-only baselines, exposing limitations in causal reasoning. ReXSonoVQA enables developing perception systems for ultrasound training, guidance, and robotic automation.
>
---
#### [new 140] Rethinking Video Human-Object Interaction: Set Prediction over Time for Unified Detection and Anticipation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频人-物交互理解任务，解决检测与预测分离及标注不准确的问题。提出HOI-DA框架和DETAnt-HOI基准，实现联合检测与未来预测。**

- **链接: [https://arxiv.org/pdf/2604.10397](https://arxiv.org/pdf/2604.10397)**

> **作者:** Yuanhao Luo; Di Wen; Kunyu Peng; Ruiping Liu; Junwei Zheng; Yufan Chen; Jiale Wei; Rainer Stiefelhage
>
> **备注:** 17 pages, 8 figures, code will be publicly available
>
> **摘要:** Video-based human-object interaction (HOI) understanding requires both detecting ongoing interactions and anticipating their future evolution. However, existing methods usually treat anticipation as a downstream forecasting task built on externally constructed human-object pairs, limiting joint reasoning between detection and prediction. In addition, sparse keyframe annotations in current benchmarks can temporally misalign nominal future labels from actual future dynamics, reducing the reliability of anticipation evaluation. To address these issues, we introduce DETAnt-HOI, a temporally corrected benchmark derived from VidHOI and Action Genome for more faithful multi-horizon evaluation, and HOI-DA, a pair-centric framework that jointly performs subject-object localization, present HOI detection, and future anticipation by modeling future interactions as residual transitions from current pair states. Experiments show consistent improvements in both detection and anticipation, with larger gains at longer horizons. Our results highlight that anticipation is most effective when learned jointly with detection as a structural constraint on pair-level video representation learning. Benchmark and code will be publicly available.
>
---
#### [new 141] AIM-Bench: Benchmarking and Improving Affective Image Manipulation via Fine-Grained Hierarchical Control
- **分类: cs.CV**

- **简介: 该论文属于情感图像编辑任务，旨在解决现有基准缺乏细粒度情感控制的问题。构建了AIM-Bench基准，并提出AIM-40k数据集以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.10454](https://arxiv.org/pdf/2604.10454)**

> **作者:** Shi Chen; Xuecheng Wu; Heli Sun; Yunyun Shi; Xinyi Yin; Fengjian Xue; Jinheng Xie; Dingkang Yang; Hao Wang; Junxiao Xue; Liang He
>
> **摘要:** Affective Image Manipulation (AIM) aims to evoke specific emotions through targeted editing. Current image editing benchmarks primarily focus on object-level modifications in general scenarios, lacking the fine-grained granularity to capture affective dimensions. To bridge this gap, we introduce the first benchmark designed for AIM termed AIM-Bench. This benchmark is built upon a dual-path affective modeling scheme that integrates the Mikels emotion taxonomy with the Valence-Arousal-Dominance framework, enabling high-level semantic and fine-grained continuous manipulation. Through a hierarchical human-in-the-loop workflow, we finally curate 800 high-quality samples covering 8 emotional categories and 5 editing types. To effectively assess performance, we also design a composite evaluation suite combining rule-based and model-based metrics to holistically assess instruction consistency, aesthetics, and emotional expressiveness. Extensive evaluations reveal that current editing models face significant challenges, most notably a prevalent positivity bias, which stemming from inherent imbalances in training data distribution. To tackle this, we propose a scalable data engine utilizing an inverse repainting strategy to construct AIM-40k, a balanced instruction-tuning dataset comprising 40k samples. Concretely, we enhance raw affective images via generative redrawing to establish high-fidelity ground truths, and synthesize input images with divergent emotions and paired precise instructions. Fine-tuning a baseline model on AIM-40k yields a 9.15% relative improvement in overall performance, demonstrating the effectiveness of our AIM-40k. Our data and related code will be made open soon.
>
---
#### [new 142] Rein3D: Reinforced 3D Indoor Scene Generation with Panoramic Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于3D室内场景生成任务，解决从稀疏输入生成高质量、全局一致的3D场景问题。通过结合3DGS与视频扩散模型，提出Rein3D框架，提升场景重建质量与长距离探索能力。**

- **链接: [https://arxiv.org/pdf/2604.10578](https://arxiv.org/pdf/2604.10578)**

> **作者:** Dehui Wang; Congsheng Xu; Rong Wei; Yue Shi; Shoufa Chen; Dingxiang Luo; Tianshuo Yang; Xiaokang Yang; Yusen Qin; Rui Tang; Yao Mu
>
> **摘要:** The growing demand for Embodied AI and VR applications has highlighted the need for synthesizing high-quality 3D indoor scenes from sparse inputs. However, existing approaches struggle to infer massive amounts of missing geometry in large unseen areas while maintaining global consistency, often producing locally plausible but globally inconsistent reconstructions. We present Rein3D, a framework that reconstructs full 360-degree indoor environments by coupling explicit 3D Gaussian Splatting (3DGS) with temporally coherent priors from video diffusion models. Our approach follows a "restore-and-refine" paradigm: we employ a radial exploration strategy to render imperfect panoramic videos along trajectories starting from the origin, effectively uncovering occluded regions from a coarse 3DGS initialization. These sequences are restored by a panoramic video-to-video diffusion model and further enhanced via video super-resolution to synthesize high-fidelity geometry and textures. Finally, these refined videos serve as pseudo-ground truths to update the global 3D Gaussian field. To support this task, we construct PanoV2V-15K, a dataset of over 15K paired clean and degraded panoramic videos for diffusion-based scene restoration. Experiments demonstrate that Rein3D produces photorealistic and globally consistent 3D scenes and significantly improves long-range camera exploration compared with existing baselines.
>
---
#### [new 143] SIMPLER: H&E-Informed Representation Learning for Structured Illumination Microscopy
- **分类: cs.CV**

- **简介: 该论文提出SIMPLER，解决SIM与H&E模态不匹配问题，通过联合预训练提升SIM性能，同时保持H&E表示。**

- **链接: [https://arxiv.org/pdf/2604.10334](https://arxiv.org/pdf/2604.10334)**

> **作者:** Abu Zahid Bin Aziz; Syed Fahim Ahmed; Gnanesh Rasineni; Mei Wang; Olcaytu Hatipoglu; Marisa Ricci; Malaiyah Shaw; Guang Li; J. Quincy Brown; Valerio Pascucci; Shireen Elhabian
>
> **摘要:** Structured Illumination Microscopy (SIM) enables rapid, high-contrast optical sectioning of fresh tissue without staining or physical sectioning, making it promising for intraoperative and point-of-care diagnostics. Recent foundation and large-scale self-supervised models in digital pathology have demonstrated strong performance on section-based modalities such as Hematoxylin and Eosin (H&E) and immunohistochemistry (IHC). However, these approaches are predominantly trained on thin tissue sections and do not explicitly address thick-tissue fluorescence modalities such as SIM. When transferred directly to SIM, performance is constrained by substantial modality shift, and naive fine-tuning often overfits to modality-specific appearance rather than underlying histological structure. We introduce SIMPLER (Structured Illumination Microscopy-Powered Learning for Embedding Representations), a cross-modality self-supervised pretraining framework that leverages H&E as a semantic anchor to learn reusable SIM representations. H&E encodes rich cellular and glandular structure aligned with established clinical annotations, while SIM provides rapid, nondestructive imaging of fresh tissue. During pretraining, SIM and H&E are progressively aligned through adversarial, contrastive, and reconstruction-based objectives, encouraging SIM embeddings to internalize histological structure from H&E without collapsing modality-specific characteristics. A single pretrained SIMPLER encoder transfers across multiple downstream tasks, including multiple instance learning and morphological clustering, consistently outperforming SIM models trained from scratch or H&E-only pretraining. Importantly, joint alignment enhances SIM performance without degrading H&E representations, demonstrating asymmetric enrichment rather
>
---
#### [new 144] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决手术中遮挡对模型性能的影响问题。构建基准测试框架，评估不同模型在遮挡下的表现，提出新的评价方法。**

- **链接: [https://arxiv.org/pdf/2604.11711](https://arxiv.org/pdf/2604.11711)**

> **作者:** Nhan Ho; Luu Le; Thanh-Huy Nguyen; Thien Nguyen; Xiaofeng Liu; Ulas Bagci
>
> **备注:** Accepted at CV4Clinic, CVPR 2026. 10 pages, 4 figures
>
> **摘要:** Occlusion, where target structures are partially hidden by surgical instruments or overlapping tissues, remains a critical yet underexplored challenge for foundation segmentation models in clinical endoscopy. We introduce OccSAM-Bench, a benchmark designed to systematically evaluate SAM-family models under controlled, synthesized surgical occlusion. Our framework simulates two occlusion types (i.e., surgical tool overlay and cutout) across three calibrated severity levels on three public polyp datasets. We propose a novel three-region evaluation protocol that decomposes segmentation performance into full, visible-only, and invisible targets. This metric exposes behaviors that standard amodal evaluation obscures, revealing two distinct model archetypes: Occluder-Aware models (SAM, SAM 2, SAM 3, MedSAM3), which prioritize visible tissue delineation and reject instruments, and Occluder-Agnostic models (MedSAM, MedSAM2), which confidently predict into occluded regions. SAM-Med2D aligns with neither and underperforms across all conditions. Ultimately, our results demonstrate that occlusion robustness is not uniform across architectures, and model selection must be driven by specific clinical intent-whether prioritizing conservative visible-tissue segmentation or the amodal inference of hidden anatomy.
>
---
#### [new 145] EDFNet: Early Fusion of Edge and Depth for Thin-Obstacle Segmentation in UAV Navigation
- **分类: cs.CV**

- **简介: 该论文属于无人机避障任务，解决细小障碍物（如电线、树枝）分割问题。提出EDFNet框架，融合RGB、深度和边缘信息，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.09694](https://arxiv.org/pdf/2604.09694)**

> **作者:** Negar Fathi
>
> **摘要:** Autonomous Unmanned Aerial Vehicles (UAVs) must reliably detect thin obstacles such as wires, poles, and branches to navigate safely in real-world environments. These structures remain difficult to perceive because they occupy few pixels, often exhibit weak visual contrast, and are strongly affected by class imbalance. Existing segmentation methods primarily target coarser obstacles and do not fully exploit the complementary multimodal cues needed for thin-structure perception. We present EDFNet, a modular early-fusion segmentation framework that integrates RGB, depth, and edge information for thin-obstacle perception in cluttered aerial scenes. We evaluate EDFNet on the Drone Depth and Obstacle Segmentation (DDOS) dataset across sixteen modality-backbone configurations using U-Net and DeepLabV3 in pretrained and non-pretrained settings. The results show that early RGB-Depth-Edge fusion provides a competitive and well-balanced baseline, with the most consistent gains appearing in boundary-sensitive and recall-oriented metrics. The pretrained RGBDE U-Net achieves the best overall performance, with the highest Thin-Structure Evaluation Score (0.244), mean IoU (0.219), and boundary IoU (0.234), while maintaining competitive runtime performance (19.62 FPS) on our evaluation hardware. However, performance on the rarest ultra-thin categories remains low across all models, indicating that reliable ultra-thin segmentation is still an open challenge. Overall, these findings position early RGB-Depth-Edge fusion as a practical and modular baseline for thin-obstacle segmentation in UAV navigation.
>
---
#### [new 146] Bidirectional Learning of Facial Action Units and Expressions via Structured Semantic Mapping across Heterogeneous Datasets
- **分类: cs.CV**

- **简介: 该论文属于面部动作单元检测与表情识别任务，解决跨数据集的双向学习问题。提出SSM框架，通过语义映射实现两者联合优化。**

- **链接: [https://arxiv.org/pdf/2604.10541](https://arxiv.org/pdf/2604.10541)**

> **作者:** Jia Li; Yu Zhang; Yin Chen; Zhenzhen Hu; Yong Li; Richang Hong; Shiguang Shan; Meng Wang
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** Facial action unit (AU) detection and facial expression (FE) recognition can be jointly viewed as affective facial behavior tasks, representing fine-grained muscular activations and coarse-grained holistic affective states, respectively. Despite their inherent semantic correlation, existing studies predominantly focus on knowledge transfer from AUs to FEs, while bidirectional learning remains insufficiently explored. In practice, this challenge is further compounded by heterogeneous data conditions, where AU and FE datasets differ in annotation paradigms (frame-level vs.\ clip-level), label granularity, and data availability and diversity, hindering effective joint learning. To address these issues, we propose a Structured Semantic Mapping (SSM) framework for bidirectional AU--FE learning under different data domains and heterogeneous supervision. SSM consists of three key components: (1) a shared visual backbone that learns unified facial representations from dynamic AU and FE videos; (2) semantic mediation via a Textual Semantic Prototype (TSP) module, which constructs structured semantic prototypes from fixed textual descriptions augmented with learnable context prompts, serving as supervision signals and cross-task alignment anchors in a shared semantic space; and (3) a Dynamic Prior Mapping (DPM) module that incorporates prior knowledge derived from the Facial Action Coding System and learns a data-driven association matrix in a high-level feature space, enabling explicit and bidirectional knowledge transfer. Extensive experiments on popular AU detection and FE recognition benchmarks show that SSM achieves state-of-the-art performance on both tasks simultaneously, and demonstrate that holistic expression semantics can in turn enhance fine-grained AU learning even across heterogeneous datasets.
>
---
#### [new 147] A Modular Zero-Shot Pipeline for Accident Detection, Localization, and Classification in Traffic Surveillance Video
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对交通监控视频中的事故检测任务，解决无标注数据下事故时间、位置和类型预测问题。通过三个模块实现定位与分类，使用预训练模型无需领域微调。**

- **链接: [https://arxiv.org/pdf/2604.09685](https://arxiv.org/pdf/2604.09685)**

> **作者:** Amey Thakur; Sarvesh Talele
>
> **备注:** 9 pages, 7 figures, 2 tables. Submitted to the ACCIDENT @ CVPR 2026 Workshop. Source code and notebook available at this https URL
>
> **摘要:** We describe a zero-shot pipeline developed for the ACCIDENT @ CVPR 2026 challenge. The challenge requires predicting when, where, and what type of traffic accident occurs in surveillance video, without labeled real-world training data. Our method separates the problem into three independent modules. The first module localizes the collision in time by running peak detection on z-score normalized frame-difference signals. The second module finds the impact location by computing the weighted centroid of cumulative dense optical flow magnitude maps using the Farneback algorithm. The third module classifies collision type by measuring cosine similarity between CLIP image embeddings of frames near the detected peak and text embeddings built from multi-prompt natural language descriptions of each collision category. No domain-specific fine-tuning is involved; the pipeline processes each video using only pre-trained model weights. Our implementation is publicly available as a Kaggle notebook.
>
---
#### [new 148] MosaicMRI: A Diverse Dataset and Benchmark for Raw Musculoskeletal MRI
- **分类: cs.CV; cs.LG; eess.SP; physics.med-ph; stat.ML**

- **简介: 该论文提出MosaicMRI数据集，用于解决肌骨MRI中深度学习模型训练与评估的多样性不足问题，通过大规模多模态数据提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11762](https://arxiv.org/pdf/2604.11762)**

> **作者:** Paula Arguello; Berk Tinaz; Mohammad Shahab Sepehri; Maryam Soltanolkotabi; Mahdi Soltanolkotabi
>
> **备注:** 15 pages, 6 figures, preliminary version
>
> **摘要:** Deep learning underpins a wide range of applications in MRI, including reconstruction, artifact removal, and segmentation. However, progress has been driven largely by public datasets focused on brain and knee imaging, shaping how models are trained and evaluated. As a result, careful studies of the reliability of these models across diverse anatomical settings remain limited. In this work, we introduce MosaicMRI, a large and diverse collection of fully sampled raw musculoskeletal (MSK) MR measurements designed for training and evaluating machine-learning-based methods. MosaicMRI is the largest open-source raw MSK MRI dataset to date, comprising 2,671 volumes and 80,156 slices. The dataset offers substantial diversity in volume orientation (e.g., axial, sagittal), imaging contrasts (e.g., PD, T1, T2), anatomies (e.g., spine, knee, hip, ankle, and others), and numbers of acquisition coils. Using VarNet as a baseline for accelerated reconstruction task, we perform a comprehensive set of experiments to study scaling behavior with respect to both model capacity and dataset size. Interestingly, models trained on the combined anatomies significantly outperform anatomy-specific models in low-sample regimes, highlighting the benefits of anatomical diversity and the presence of exploitable cross-anatomical correlations. We further evaluate robustness and cross-anatomy generalization by training models on one anatomy (e.g., spine) and testing them on another (e.g., knee). Notably, we identify groups of body parts (e.g., foot and elbow) that generalize well with each other, and highlight that performance under domain shifts depends on both training set size, anatomy, and protocol-specific factors.
>
---
#### [new 149] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching Vision-Language-Action Models
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于安全领域，针对流匹配视觉-语言-动作模型的后门漏洞提出FlowHijack攻击方法，解决连续动作生成机制的安全问题。**

- **链接: [https://arxiv.org/pdf/2604.09651](https://arxiv.org/pdf/2604.09651)**

> **作者:** Xinyuan An; Tao Luo; Gengyun Peng; Yaobing Wang; Kui Ren; Dongxia Wang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Vision-Language-Action (VLA) models are emerging as a cornerstone for robotics, with flow-matching policies like $\pi_0$ showing great promise in generating smooth, continuous actions. As these models advance, their unique action generation mechanism - the vector field dynamics - presents a critical yet unexplored security vulnerability, particularly backdoor vulnerabilities. Existing backdoor attacks designed for autoregressive discretization VLAs cannot be directly applied to this new continuous dynamics. We introduce FlowHijack, the first backdoor attack framework to systematically target the underlying vector-field dynamics of flow-matching VLAs. Our method combines a novel $\tau$-conditioned injection strategy, which manipulates the initial phase of the action generation, with a dynamics mimicry regularizer. Experiments demonstrate that FlowHijack achieves high attack success rates using stealthy, context-aware triggers where prior works failed. Crucially, it preserves benign task performance and, by enforcing kinematic similarity, generates malicious actions that are behaviorally indistinguishable from normal actions. Our findings reveal a significant vulnerability in continuous embodied models, highlighting the urgent need for defenses targeting the model's internal generative dynamics.
>
---
#### [new 150] Radiology Report Generation for Low-Quality X-Ray Images
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决低质量X光图像导致的报告生成性能下降问题。通过引入质量评估和双循环训练策略提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10188](https://arxiv.org/pdf/2604.10188)**

> **作者:** Hongze Zhu; Chen Hu; Jiaxuan Jiang; Hong Liu; Yawen Huang; Ming Hu; Tianyu Wang; Zhijian Wu; Yefeng Zheng
>
> **摘要:** Vision-Language Models (VLMs) have significantly advanced automated Radiology Report Generation (RRG). However, existing methods implicitly assume high-quality inputs, overlooking the noise and artifacts prevalent in real-world clinical environments. Consequently, current models exhibit severe performance degradation when processing suboptimal images. To bridge this gap, we propose a robust report generation framework explicitly designed for image quality variations. We first introduce an Automated Quality Assessment Agent (AQAA) to identify low-quality samples within the MIMIC-CXR dataset and establish the Low-quality Radiology Report Generation (LRRG) benchmark. To tackle degradation-induced shifts, we propose a novel Dual-loop Training Strategy leveraging bi-level optimization and gradient consistency. This approach ensures the model learns quality-agnostic diagnostic features by aligning gradient directions across varying quality regimes. Extensive experiments demonstrate that our approach effectively mitigates model performance degradation caused by image quality deterioration. The code and data will be released upon acceptance.
>
---
#### [new 151] Vector Field Synthesis with Sparse Streamlines Using Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于向量场生成任务，旨在从稀疏流线中合成符合物理规律的2D向量场。工作采用扩散模型实现高效、物理一致的重建。**

- **链接: [https://arxiv.org/pdf/2604.09838](https://arxiv.org/pdf/2604.09838)**

> **作者:** Nguyen K. Phan; Ricardo Morales; Sebastian D. Espriella; Guoning Chen
>
> **备注:** 5 pages, 4 figures; published at IEEE VIS 2025
>
> **摘要:** We present a novel diffusion-based framework for synthesizing 2D vector fields from sparse, coherent inputs (i.e., streamlines) while maintaining physical plausibility. Our method employs a conditional denoising diffusion probabilistic model with classifier-free guidance, enabling progressive reconstruction that preserves both geometric and physical constraints. Experimental results demonstrate our method's ability to synthesize plausible vector fields that adhere to physical laws while maintaining fidelity to sparse input observations, outperforming traditional optimization-based approaches in terms of flexibility and physical consistency.
>
---
#### [new 152] Pair2Scene: Learning Local Object Relations for Procedural Scene Generation
- **分类: cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决数据稀缺和空间关系建模困难的问题。提出Pair2Scene框架，通过学习局部对象关系生成物理和语义合理的场景。**

- **链接: [https://arxiv.org/pdf/2604.11808](https://arxiv.org/pdf/2604.11808)**

> **作者:** Xingjian Ran; Shujie Zhang; Weipeng Zhong; Li Luo; Bo Dai
>
> **摘要:** Generating high-fidelity 3D indoor scenes remains a significant challenge due to data scarcity and the complexity of modeling intricate spatial relations. Current methods often struggle to scale beyond training distribution to dense scenes or rely on LLMs/VLMs that lack the ability for precise spatial reasoning. Building on top of the observation that object placement relies mainly on local dependencies instead of information-redundant global distributions, in this paper, we propose Pair2Scene, a novel procedural generation framework that integrates learned local rules with scene hierarchies and physics-based algorithms. These rules mainly capture two types of inter-object relations, namely support relations that follow physical hierarchies, and functional relations that reflect semantic links. We model these rules through a network, which estimates spatial position distributions of dependent objects conditioned on position and geometry of the anchor ones. Accordingly, we curate a dataset 3D-Pairs from existing scene data to train the model. During inference, our framework can generate scenes by recursively applying our model within a hierarchical structure, leveraging collision-aware rejection sampling to align local rules into coherent global layouts. Extensive experiments demonstrate that our framework outperforms existing methods in generating complex environments that go beyond training data while maintaining physical and semantic plausibility.
>
---
#### [new 153] Dual-Branch Remote Sensing Infrared Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于红外图像超分辨率任务，旨在提升低分辨率红外图像的清晰度，同时保持目标轮廓和辐射稳定性。工作包括提出双分支系统，结合局部Transformer与全局状态空间模型，提升重建效果。**

- **链接: [https://arxiv.org/pdf/2604.10112](https://arxiv.org/pdf/2604.10112)**

> **作者:** Xining Ge; Gengjia Chang; Weijun Yuan; Zhan Li; Zhanglu Chen; Boyang Yao; Yihang Chen; Yifan Deng; Shuhong Liu
>
> **摘要:** Remote sensing infrared image super-resolution aims to recover sharper thermal observations from low-resolution inputs while preserving target contours, scene layout, and radiometric stability. Unlike visible-image super-resolution, thermal imagery is weakly textured and more sensitive to unstable local sharpening, which makes complementary local and global modeling especially important. This paper presents our solution to the NTIRE 2026 Infrared Image Super-Resolution Challenge, a dual-branch system that combines a HAT-L branch and a MambaIRv2-L branch. The inference pipeline applies test-time local conversion on HAT, eight-way self-ensemble on MambaIRv2, and fixed equal-weight image-space fusion. We report both the official challenge score and a reproducible evaluation on 12 synthetic times-four thermal samples derived from Caltech Aerial RGB-Thermal, on which the fused output outperforms either single branch in PSNR, SSIM, and the overall Score. The results suggest that infrared super-resolution benefits from explicit complementarity between locally strong transformer restoration and globally stable state-space modeling.
>
---
#### [new 154] What and Where to Adapt: Structure-Semantics Co-Tuning for Machine Vision Compression via Synergistic Adapters
- **分类: cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决预训练编码器在适应不同任务时的效率问题。通过引入协同适配器，提升压缩性能并减少参数量。**

- **链接: [https://arxiv.org/pdf/2604.10017](https://arxiv.org/pdf/2604.10017)**

> **作者:** Shaobo Liu; Haobo Xiong; Kai Liu; Yuna Lin
>
> **备注:** Accepted by the IEEE/CVF Conference on Computer Vision and Pattern Recognition Findings, 2026
>
> **摘要:** Parameter-efficient fine-tuning of pre-trained codecs is a promising direction in image compression for human and machine vision. While most existing works have primarily focused on tuning the feature structure within the encoder-decoder backbones, the adaptation of the statistical semantics within the entropy model has received limited attention despite its function of predicting the probability distribution of latent features. Our analysis reveals that naive adapter insertion into the entropy model can lead to suboptimal outcomes, underscoring that the effectiveness of adapter-based tuning depends critically on the coordination between adapter type and placement across the compression pipeline. Therefore, we introduce Structure-Semantics Co-Tuning (S2-CoT), a novel framework that achieves this coordination via two specialized, synergistic adapters: the Structural Fidelity Adapter (SFA) and the Semantic Context Adapter (SCA). SFA is integrated into the encoder-decoder to preserve high-fidelity representations by dynamically fusing spatial and frequency information; meanwhile, the SCA adapts the entropy model to align with SFA-tuned features by refining the channel context for more efficient statistical coding. Through joint optimization, S2-CoT turns potential performance degradation into synergistic gains, achieving state-of-the-art results across four diverse base codecs with only a small fraction of trainable parameters, closely matching full fine-tuning performance. Code is available at this https URL.
>
---
#### [new 155] A Comparison of Multi-View Stereo Methods for Photogrammetric 3D Reconstruction: From Traditional to Learning-Based Approaches
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，比较传统与学习型多视图立体方法，解决速度与精度问题，通过实验评估不同方法的准确性、覆盖率和运行时间。**

- **链接: [https://arxiv.org/pdf/2604.10246](https://arxiv.org/pdf/2604.10246)**

> **作者:** Yawen Li; George Vosselman; Francesco Nex
>
> **摘要:** Photogrammetric 3D reconstruction has long relied on traditional Structure-from-Motion (SfM) and Multi-View Stereo (MVS) methods, which provide high accuracy but face challenges in speed and scalability. Recently, learning-based MVS methods have emerged, aiming for faster and more efficient reconstruction. This work presents a comparative evaluation between a representative traditional MVS pipeline (COLMAP) and state-of-the-art learning-based approaches, including geometry-guided methods (MVSNet, PatchmatchNet, MVSAnywhere, MVSFormer++) and end-to-end frameworks (Stereo4D, FoundationStereo, DUSt3R, MASt3R, Fast3R, VGGT). Two experiments were conducted on different aerial scenarios. The first experiment used the MARS-LVIG dataset, where ground-truth 3D reconstruction was provided by LiDAR point clouds. The second experiment used a public scene from the Pix4D official website, with ground truth generated by Pix4Dmapper. We evaluated accuracy, coverage, and runtime across all methods. Experimental results show that although COLMAP can provide reliable and geometrically consistent reconstruction results, it requires more computation time. In cases where traditional methods fail in image registration, learning-based approaches exhibit stronger feature-matching capability and greater robustness. Geometry-guided methods usually require careful dataset preparation and often depend on camera pose or depth priors generated by COLMAP. End-to-end methods such as DUSt3R and VGGT achieve competitive accuracy and reasonable coverage while offering substantially faster reconstruction. However, they exhibit relatively large residuals in 3D reconstruction, particularly in challenging scenarios.
>
---
#### [new 156] LiveGesture Streamable Co-Speech Gesture Generation Model
- **分类: cs.CV**

- **简介: 该论文提出LiveGesture，一种实时、流式驱动的全身手势生成框架，解决在线手势生成问题。通过模块化设计实现区域协调运动生成，提升实时性和多样性。**

- **链接: [https://arxiv.org/pdf/2604.10927](https://arxiv.org/pdf/2604.10927)**

> **作者:** Muhammad Usama Saleem; Mayur Jagdishbhai Patel; Ekkasit Pinyoanuntapong; Zhongxing Qin; Li Yang; Hongfei Xue; Ahmed Helmy; Chen Chen; Pu Wang
>
> **摘要:** We propose LiveGesture, the first fully streamable, speech-driven full-body gesture generation framework that operates with zero look-ahead and supports arbitrary sequence length. Unlike existing co-speech gesture methods, which are designed for offline generation and either treat body regions independently or entangle all joints within a single model, LiveGesture is built from the ground up for causal, region-coordinated motion generation. LiveGesture consists of two main modules: the Streamable Vector Quantized Motion Tokenizer (SVQ) and the Hierarchical Autoregressive Transformer (HAR). The SVQ tokenizer converts the motion sequence of each body region into causal, discrete motion tokens, enabling real-time, streamable token decoding. On top of SVQ, HAR employs region-expert autoregressive (xAR) transformers to model expressive, fine-grained motion dynamics for each body region. A causal spatio-temporal fusion module (xAR Fusion) then captures and integrates correlated motion dynamics across regions. Both xAR and xAR Fusion are conditioned on live, continuously arriving audio signals encoded by a streamable causal audio encoder. To enhance robustness under streaming noise and prediction errors, we introduce autoregressive masking training, which leverages uncertainty-guided token masking and random region masking to expose the model to imperfect, partially erroneous histories during training. Experiments on the BEAT2 dataset demonstrate that LiveGesture produces coherent, diverse, and beat-synchronous full-body gestures in real time, matching or surpassing state-of-the-art offline methods under true zero look-ahead conditions.
>
---
#### [new 157] Multi-Granularity Reasoning for Image Quality Assessment via Attribute-Aware Reinforcement Learning to Rank
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决单一粒度评估与多维感知属性忽略的问题。提出MG-IQA框架，实现整体质量与细粒度属性的联合评估。**

- **链接: [https://arxiv.org/pdf/2604.09704](https://arxiv.org/pdf/2604.09704)**

> **作者:** Xiangyong Chen; Xiaochuan Lin; Haoran Liu; Xuan Li; Yichen Su; Xiangwei Guo
>
> **摘要:** Recent advances in reasoning-induced image quality assessment (IQA) have demonstrated the power of reinforcement learning to rank (RL2R) for training vision-language models (VLMs) to assess perceptual quality. However, existing approaches operate at a single granularity, predicting only an overall quality score, while overlooking the multi-dimensional nature of human quality perception, which encompasses attributes such as sharpness, color fidelity, noise level, and compositional aesthetics. In this paper, we propose MG-IQA (Multi-Granularity IQA), a multi-granularity reasoning framework that extends RL2R to jointly assess overall image quality and fine-grained quality attributes within a single inference pass. Our approach introduces three key innovations: (1) an attribute-aware prompting strategy that elicits structured multi-attribute reasoning from VLMs; (2) a multi-dimensional Thurstone reward model that computes attribute-specific fidelity rewards for group relative policy optimization; and (3) a cross-domain alignment mechanism that enables stable joint training across synthetic distortion, authentic distortion, and AI-generated image datasets without perceptual scale re-alignment. Extensive experiments on eight IQA benchmarks demonstrate that MG-IQA consistently outperforms state-of-the-art methods in both overall quality prediction (average SRCC improvement of 2.1\%) and attribute-level assessment, while generating interpretable, human-aligned quality descriptions.
>
---
#### [new 158] Video-based Heart Rate Estimation with Angle-guided ROI Optimization and Graph Signal Denoising
- **分类: cs.CV**

- **简介: 该论文属于rPPG任务，旨在提升视频心率估计在面部运动下的准确性。通过引入角度引导的ROI优化和图信号去噪模块，有效抑制运动伪影。**

- **链接: [https://arxiv.org/pdf/2604.11395](https://arxiv.org/pdf/2604.11395)**

> **作者:** Gan Pei; Junhao Ning; Boqiu Shen; Yan Zhu; Menghan Hu
>
> **备注:** This paper has been accepted by ICASSP 2026
>
> **摘要:** Remote photoplethysmography (rPPG) enables non-contact heart rate measurement from facial videos, but its performance is significantly degraded by facial motions such as speaking and head shaking. To address this issue, we propose two plug-and-play modules. The Angle-guided ROI Adaptive Optimization module quantifies ROI-Camera angles to refine motion-affected signals and capture global motion, while the Multi-region Joint Graph Signal Denoising module jointly models intra- and inter-regional ROI signals using graph signal processing to suppress motion artifacts. The modules are compatible with reflection model-based rPPG methods and validated on three public datasets. Results show that jointly use markedly reduces MAE, with an average decrease of 20.38\% over the baseline, while ablation studies confirm the effectiveness of each module. The work demonstrates the potential of angle-guided optimization and graph-based denoising to enhance rPPG performance in motion scenarios.
>
---
#### [new 159] FineEdit: Fine-Grained Image Edit with Bounding Box Guidance
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决传统模型依赖语言提示导致定位不准确的问题。通过引入边界框指导，提出FineEdit方法及数据集，提升编辑精度与背景一致性。**

- **链接: [https://arxiv.org/pdf/2604.10954](https://arxiv.org/pdf/2604.10954)**

> **作者:** Haohang Xu; Lin Liu; Zhibo Zhang; Rong Cong; Xiaopeng Zhang; Qi Tian
>
> **摘要:** Diffusion-based image editing models have achieved significant progress in real world applications. However, conventional models typically rely on natural language prompts, which often lack the precision required to localize target objects. Consequently, these models struggle to maintain background consistency due to their global image regeneration paradigm. Recognizing that visual cues provide an intuitive means for users to highlight specific areas of interest, we utilize bounding boxes as guidance to explicitly define the editing target. This approach ensures that the diffusion model can accurately localize the target while preserving background consistency. To achieve this, we propose FineEdit, a multi-level bounding box injection method that enables the model to utilize spatial conditions more effectively. To support this high precision guidance, we present FineEdit-1.2M, a large scale, fine-grained dataset comprising 1.2 million image editing pairs with precise bounding box annotations. Furthermore, we construct a comprehensive benchmark, termed FineEdit-Bench, which includes 1,000 images across 10 subjects to effectively evaluate region based editing capabilities. Evaluations on FineEdit-Bench demonstrate that our model significantly outperforms state-of-the-art open-source models (e.g., Qwen-Image-Edit and LongCat-Image-Edit) in instruction compliance and background preservation. Further assessments on open benchmarks (GEdit and ImgEdit Bench) confirm its superior generalization and robustness.
>
---
#### [new 160] A Benchmark and Multi-Agent System for Instruction-driven Cinematic Video Compilation
- **分类: cs.CV**

- **简介: 该论文属于视频编纂任务，旨在解决长视频自动剪辑问题。提出CineBench基准和CineAgents系统，提升编纂的叙事连贯性。**

- **链接: [https://arxiv.org/pdf/2604.10456](https://arxiv.org/pdf/2604.10456)**

> **作者:** Peixuan Zhang; Chang Zhou; Ziyuan Zhang; Hualuo Liu; Chunjie Zhang; Jingqi Liu; Xiaohui Zhou; Xi Chen; Shuchen Weng; Si Li; Boxin Shi
>
> **摘要:** The surging demand for adapting long-form cinematic content into short videos has motivated the need for versatile automatic video compilation systems. However, existing compilation methods are limited to predefined tasks, and the community lacks a comprehensive benchmark to evaluate the cinematic compilation. To address this, we introduce CineBench, the first benchmark for instruction-driven cinematic video compilation, featuring diverse user instructions and high-quality ground-truth compilations annotated by professional editors. To overcome contextual collapse and temporal fragmentation, we present CineAgents, a multi-agent system that reformulates cinematic video compilation into ``design-and-compose'' paradigm. CineAgents performs script reverse-engineering to construct a hierarchical narrative memory to provide multi-level context and employs an iterative narrative planning process that refines a creative blueprint into a final compiled script. Extensive experiments demonstrate that CineAgents significantly outperforms existing methods, generating compilations with superior narrative coherence and logical coherence.
>
---
#### [new 161] GTASA: Ground Truth Annotations for Spatiotemporal Analysis, Evaluation and Training of Video Models
- **分类: cs.CV**

- **简介: 该论文提出GTASA数据集，解决视频模型评估与训练中缺乏真实标注的问题，通过构建时空关系图和事件映射，提升视频理解性能。**

- **链接: [https://arxiv.org/pdf/2604.10385](https://arxiv.org/pdf/2604.10385)**

> **作者:** Nicolae Cudlenco; Mihai Masala; Marius Leordeanu
>
> **摘要:** Generating complex multi-actor scenario videos remains difficult even for state-of-the-art neural generators, while evaluating them is hard due to the lack of ground truth for physical plausibility and semantic faithfulness. We introduce GTASA, a corpus of multi-actor videos with per-frame spatial relation graphs and event-level temporal mappings, and the system that produced it based on Graphs of Events in Space and Time (GEST): GEST-Engine. We compare our method with both open and closed source neural generators and prove both qualitatively (human evaluation of physical validity and semantic alignment) and quantitatively (via training video captioning models) the clear advantages of our method. Probing four frozen video encoders across 11 spatiotemporal reasoning tasks enabled by GTASA's exact 3D ground truth reveals that self-supervised encoders encode spatial structure significantly better than VLM visual encoders.
>
---
#### [new 162] GIF: A Conditional Multimodal Generative Framework for IR Drop Imaging in Chip Layouts
- **分类: cs.CV**

- **简介: 该论文提出GIF框架，用于芯片布局中的IR降压成像。解决传统EDA工具效率低、成本高的问题，通过融合几何和拓扑信息生成高质量IR降压图像。**

- **链接: [https://arxiv.org/pdf/2604.09999](https://arxiv.org/pdf/2604.09999)**

> **作者:** Kiran Thorat; Nicole Meng; Mostafa Karami; Caiwen Ding; Yingjie Lao; Zhijie Jerry Shi
>
> **摘要:** IR drop analysis is essential in physical chip design to ensure the power integrity of on-chip power delivery networks. Traditional Electronic Design Automation (EDA) tools have become slow and expensive as transistor density scales. Recent works have introduced machine learning (ML)-based methods that formulate IR drop analysis as an image prediction problem. These existing ML approaches fail to capture both local and long-range dependencies and ignore crucial geometrical and topological information from physical layouts and logical connectivity. To address these limitations, we propose GIF, a Generative IR drop Framework that uses both geometrical and topological information to generate IR drop images. GIF fuses image and graph features to guide a conditional diffusion process, producing high-quality IR drop images. For instance, On the CircuitNet-N28 dataset, GIF achieves 0.78 SSIM, 0.95 Pearson correlation, 21.77 PSNR, and 0.026 NMAE, outperforming prior methods. These results demonstrate that our framework, using diffusion based multimodal conditioning, reliably generates high quality IR drop images. This shows that IR drop analysis can effectively leverage recent advances in generative modeling when geometric layout features and logical circuit topology are jointly modeled. By combining geometry aware spatial features with logical graph representations, GIF enables IR drop analysis to benefit from recent advances in generative modeling for structured image generation.
>
---
#### [new 163] DocRevive: A Unified Pipeline for Document Text Restoration
- **分类: cs.CV**

- **简介: 该论文属于文档文本修复任务，旨在解决损坏或缺失文本的重建问题。提出统一管道结合OCR、图像分析和扩散模型，实现语义连贯的文本恢复。**

- **链接: [https://arxiv.org/pdf/2604.10077](https://arxiv.org/pdf/2604.10077)**

> **作者:** Kunal Purkayastha; Ayan Banerjee; Josep Llados; Umapada Pal
>
> **摘要:** In Document Understanding, the challenge of reconstructing damaged, occluded, or incomplete text remains a critical yet unexplored problem. Subsequent document understanding tasks can benefit from a document reconstruction process. In response, this paper presents a novel unified pipeline combining state-of-the-art Optical Character Recognition (OCR), advanced image analysis, masked language modeling, and diffusion-based models to restore and reconstruct text while preserving visual integrity. We create a synthetic dataset of 30{,}078 degraded document images that simulates diverse document degradation scenarios, setting a benchmark for restoration tasks. Our pipeline detects and recognizes text, identifies degradation with an occlusion detector, and uses an inpainting model for semantically coherent reconstruction. A diffusion-based module seamlessly reintegrates text, matching font, size, and alignment. To evaluate restoration quality, we propose a Unified Context Similarity Metric (UCSM), incorporating edit, semantic, and length similarities with a contextual predictability measure that penalizes deviations when the correct text is contextually obvious. Our work advances document restoration, benefiting archival research and digital preservation while setting a new standard for text reconstruction. The OPRB dataset and code are available at \href{this https URL}{Hugging Face} and \href{this https URL}{Github} respectively.
>
---
#### [new 164] EgoFun3D: Modeling Interactive Objects from Egocentric Videos using Function Templates
- **分类: cs.CV**

- **简介: 该论文提出EgoFun3D任务，旨在从第一视角视频中建模可交互的3D物体。通过功能模板捕捉物体间功能关系，构建数据集并设计四阶段处理流程，解决真实场景下交互对象建模难题。**

- **链接: [https://arxiv.org/pdf/2604.11038](https://arxiv.org/pdf/2604.11038)**

> **作者:** Weikun Peng; Denys Iliash; Manolis Savva
>
> **备注:** Project website: this https URL
>
> **摘要:** We present EgoFun3D, a coordinated task formulation, dataset, and benchmark for modeling interactive 3D objects from egocentric videos. Interactive objects are of high interest for embodied AI but scarce, making modeling from readily available real-world videos valuable. Our task focuses on obtaining simulation-ready interactive 3D objects from egocentric video input. While prior work largely focuses on articulations, we capture general cross-part functional mappings (e.g., rotation of stove knob controls stove burner temperature) through function templates, a structured computational representation. Function templates enable precise evaluation and direct compilation into executable code across simulation platforms. To enable comprehensive benchmarking, we introduce a dataset of 271 egocentric videos featuring challenging real-world interactions with paired 3D geometry, segmentation over 2D and 3D, articulation and function template annotations. To tackle the task, we propose a 4-stage pipeline consisting of: 2D part segmentation, reconstruction, articulation estimation, and function template inference. Comprehensive benchmarking shows that the task is challenging for off-the-shelf methods, highlighting avenues for future work.
>
---
#### [new 165] Sparse Hypergraph-Enhanced Frame-Event Object Detection with Fine-Grained MoE
- **分类: cs.CV**

- **简介: 该论文属于多模态目标检测任务，解决RGB与事件流数据融合效率低的问题。提出Hyper-FEOD框架，通过稀疏超图和细粒度MoE模块提升检测性能与效率。**

- **链接: [https://arxiv.org/pdf/2604.11140](https://arxiv.org/pdf/2604.11140)**

> **作者:** Wei Bao; Yuehan Wang; Tianhang Zhou; Siqi Li; Yue Gao
>
> **摘要:** Integrating frame-based RGB cameras with event streams offers a promising solution for robust object detection under challenging dynamic conditions. However, the inherent heterogeneity and data redundancy of these modalities often lead to prohibitive computational overhead or suboptimal feature fusion. In this paper, we propose Hyper-FEOD, a high-performance and efficient detection framework, which synergistically optimizes multi-modal interaction through two core components. First, we introduce Sparse Hypergraph-enhanced Cross-Modal Fusion (S-HCF), which leverages the inherent sparsity of event streams to construct an event-guided activity map. By performing high-order hypergraph modeling exclusively on selected motion-critical sparse tokens, S-HCF captures complex non-local dependencies between RGB and event data while overcoming the traditional complexity bottlenecks of hypergraph computation. Second, we design a Fine-Grained Mixture of Experts (FG-MoE) Enhancement module to address the diverse semantic requirements of different image regions. This module employs specialized hypergraph experts tailored for object boundaries, internal textures, and backgrounds, utilizing a pixel-level spatial gating mechanism to adaptively route and enhance features. Combined with a load-balancing loss and zero-initialization strategy, FG-MoE ensures stable training and precise feature refinement without disrupting the pre-trained backbone's distribution. Experimental results on mainstream RGB-Event benchmarks demonstrate that Hyper-FEOD achieves a superior accuracy-efficiency trade-off, outperforming state-of-the-art methods while maintaining a lightweight footprint suitable for real-time edge deployment.
>
---
#### [new 166] The Second Challenge on Real-World Face Restoration at NTIRE 2026: Methods and Results
- **分类: cs.CV**

- **简介: 该论文属于真实世界人脸修复任务，旨在提升修复结果的自然度和身份一致性。工作包括提出解决方案、评估模型性能，并分析最新趋势。**

- **链接: [https://arxiv.org/pdf/2604.10532](https://arxiv.org/pdf/2604.10532)**

> **作者:** Jingkai Wang; Jue Gong; Zheng Chen; Kai Liu; Jiatong Li; Yulun Zhang; Radu Timofte; Jiachen Tu; Yaokun Shi; Guoyi Xu; Yaoxin Jiang; Jiajia Liu; Yingsi Chen; Yijiao Liu; Hui Li; Yu Wang; Congchao Zhu; Alexandru-Gabriel Lefterache; Anamaria Radoi; Chuanyue Yan; Tao Lu; Yanduo Zhang; Kanghui Zhao; Jiaming Wang; Yuqi Li; WenBo Xiong; Yifei Chen; Xian Hu; Wei Deng; Daiguo Zhou; Sujith Roy V; Claudia Jesuraj; Vikas B; Spoorthi LC; Nikhil Akalwadi; Ramesh Ashok Tabib; Uma Mudenagudi; Yuxuan Jiang; Chengxi Zeng; Tianhao Peng; Fan Zhang; David Bull Wei Zhou; Linfeng Li; Hongyu Huang; Hoyoung Lee; SangYun Oh; ChangYoung Jeong; Axi Niu; Jinyang Zhang; Zhenguo Wu; Senyan Qing; Jinqiu Sun; Yanning Zhang
>
> **备注:** NTIRE 26: this https URL . NTIRE Real-World Face Restoration: this https URL . CVPR 2026 Workshop
>
> **摘要:** This paper provides a review of the NTIRE 2026 challenge on real-world face restoration, highlighting the proposed solutions and the resulting outcomes. The challenge focuses on generating natural and realistic outputs while maintaining identity consistency. Its goal is to advance state-of-the-art solutions for perceptual quality and realism, without imposing constraints on computational resources or training data. Performance is evaluated using a weighted image quality assessment (IQA) score and employs the AdaFace model as an identity checker. The competition attracted 96 registrants, with 10 teams submitting valid models; ultimately, 9 teams achieved valid scores in the final ranking. This collaborative effort advances the performance of real-world face restoration while offering an in-depth overview of the latest trends in the field.
>
---
#### [new 167] A Dual Cross-Attention Graph Learning Framework For Multimodal MRI-Based Major Depressive Disorder Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态MRI的抑郁症检测任务，旨在解决单模态无法全面反映脑变化的问题。提出双交叉注意力框架，有效融合结构和功能MRI数据，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.10116](https://arxiv.org/pdf/2604.10116)**

> **作者:** Nojod M. Alotaibi; Areej M. Alhothali
>
> **备注:** 19 pages, 1 figure
>
> **摘要:** Major depressive disorder (MDD) is a prevalent mental disorder associated with complex neurobiological changes that cannot be fully captured using a single imaging modality. The use of multimodal magnetic resonance imaging (MRI) provides a more comprehensive understanding of brain changes by combining structural and functional data. Despite this, the effective integration of these modalities remains challenging. In this study, we propose a dual cross-attention-based multimodal fusion framework that explicitly models bidirectional interactions between structural MRI (sMRI) and resting-state functional MRI (rs-fMRI) representations. The proposed approach is tested on the large-scale REST-meta-MDD dataset using both structural and functional brain atlas configurations. Numerous experiments conducted under a 10-fold stratified cross-validation demonstrated that the proposed fusion algorithm achieves robust and competitive performance across all atlas types. The proposed method consistently outperforms conventional feature-level concatenation for functional atlases, while maintaining comparable performance for structural atlases. The most effective dual cross-attention multimodal model obtained 84.71% accuracy, 86.42% sensitivity, 82.89% specificity, 84.34% precision, and 85.37% F1-score. These findings emphasize the importance of explicitly modeling cross-modal interactions for multimodal neuroimaging-based MDD classification.
>
---
#### [new 168] Data-Driven Automated Identification of Optimal Feature-Representative Images in Infrared Thermography Using Statistical and Morphological Metrics
- **分类: cs.CV; physics.app-ph; physics.data-an**

- **简介: 该论文属于红外热成像中的缺陷检测任务，旨在解决自动识别最具代表性的缺陷图像问题。通过统计和形态学指标，实现无需先验信息的缺陷图像选择。**

- **链接: [https://arxiv.org/pdf/2604.09728](https://arxiv.org/pdf/2604.09728)**

> **作者:** Harutyun Yagdjian; Martin Gurka
>
> **备注:** 21 pages + 4 Appendix, 13 figures
>
> **摘要:** Infrared thermography (IRT) is a widely used non-destructive testing technique for detecting structural features such as subsurface defects. However, most IRT post-processing methods generate image sequences in which defect visibility varies strongly across time, frequency, or coefficient/index domains, making the identification of defect-representative images a critical challenge. Conventional evaluation metrics, such as the signal-to-noise ratio (SNR) or the Tanimoto criterion, often require prior knowledge of defect locations or defect-free reference regions, limiting their suitability for automated and unsupervised analysis. In this work, a data-driven methodology is proposed to identify images within IRT datasets that are most likely to contain and represent structural features, particularly anomalies and defects, without requiring prior spatial information. The approach is based on three complementary metrics: the Homogeneity Index of Mixture (HI), which quantifies statistical heterogeneity via deviations of local intensity distributions from a global reference distribution; a Representative Elementary Area (REA), derived from a Minkowski-functional adaptation of the Representative Elementary Volume concept to two-dimensional images; and a geometrical-topological Total Variation Energy (TVE) index, also based on two-dimensional Minkowski functionals, designed to improve sensitivity to localized anomalies. The framework is validated experimentally using pulse-heated IRT data from a carbon fiber-reinforced polymer (CFRP) plate containing six artificial defects at depths between 0.135 mm and 0.810 mm, and is further supported by one-dimensional N-layer thermal model simulations. The results demonstrate robust and unbiased ranking of image sequences and provide a reliable basis for automated defect-oriented image selection in IRT.
>
---
#### [new 169] SinkTrack: Attention Sink based Context Anchoring for Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于自然语言处理任务，解决LLM的幻觉和上下文遗忘问题。提出SinkTrack方法，通过注意力锚定保持对初始输入的聚焦，提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2604.10027](https://arxiv.org/pdf/2604.10027)**

> **作者:** Xu Liu; Guikun Chen; Wenguan Wang
>
> **备注:** ICLR 2026. Code: this https URL
>
> **摘要:** Large language models (LLMs) suffer from hallucination and context forgetting. Prior studies suggest that attention drift is a primary cause of these problems, where LLMs' focus shifts towards newly generated tokens and away from the initial input context. To counteract this, we make use of a related, intrinsic characteristic of LLMs: attention sink -- the tendency to consistently allocate high attention to the very first token (i.e., <BOS>) of a sequence. Concretely, we propose an advanced context anchoring method, SinkTrack, which treats <BOS> as an information anchor and injects key contextual features (such as those derived from the input image or instruction) into its representation. As such, LLM remains anchored to the initial input context throughout the entire generation process. SinkTrack is training-free, plug-and-play, and introduces negligible inference overhead. Experiments demonstrate that SinkTrack mitigates hallucination and context forgetting across both textual (e.g., +21.6% on SQuAD2.0 with Llama3.1-8B-Instruct) and multi-modal (e.g., +22.8% on M3CoT with Qwen2.5-VL-7B-Instruct) tasks. Its consistent gains across different architectures and scales underscore the robustness and generalizability. We also analyze its underlying working mechanism from the perspective of information delivery. Our source code is available at this https URL.
>
---
#### [new 170] Semantic Manipulation Localization
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Semantic Manipulation Localization（SML）任务，解决传统图像篡改定位方法在面对语义细微修改时效果不佳的问题。通过构建基准和提出TRACE框架，提升对语义篡改的定位能力。**

- **链接: [https://arxiv.org/pdf/2604.10132](https://arxiv.org/pdf/2604.10132)**

> **作者:** Zhenshan Tan; Chenhan Lu; Yuxiang Huang; Ziwen He; Xiang Zhang; Yuzhe Sha; Xianyi Chen; Tianrun Chen; Zhangjie Fu
>
> **摘要:** Image Manipulation Localization (IML) aims to identify edited regions in an image. However, with the increasing use of modern image editing and generative models, many manipulations no longer exhibit obvious low-level artifacts. Instead, they often involve subtle but meaning-altering edits to an object's attributes, state, or relationships while remaining highly consistent with the surrounding content. This makes conventional IML methods less effective because they mainly rely on artifact detection rather than semantic sensitivity. To address this issue, we introduce Semantic Manipulation Localization (SML), a new task that focuses on localizing subtle semantic edits that significantly change image interpretation. We further construct a dedicated fine-grained benchmark for SML using a semantics-driven manipulation pipeline with pixel-level annotations. Based on this task, we propose TRACE (Targeted Reasoning of Attributed Cognitive Edits), an end-to-end framework that models semantic sensitivity through three progressively coupled components: semantic anchoring, semantic perturbation sensing, and semantic-constrained reasoning. Specifically, TRACE first identifies semantically meaningful regions that support image understanding, then injects perturbation-sensitive frequency cues to capture subtle edits under strong visual consistency, and finally verifies candidate regions through joint reasoning over semantic content and semantic scope. Extensive experiments show that TRACE consistently outperforms existing IML methods on our benchmark and produces more complete, compact, and semantically coherent localization results. These results demonstrate the necessity of moving beyond artifact-based localization and provide a new direction for image forensics in complex semantic editing scenarios.
>
---
#### [new 171] GS4City: Hierarchical Semantic Gaussian Splatting via City-Model Priors
- **分类: cs.CV**

- **简介: 该论文提出GS4City，解决城市场景中语义3D高斯点云的构建问题，通过结合城市模型先验提升语义分割精度。**

- **链接: [https://arxiv.org/pdf/2604.11401](https://arxiv.org/pdf/2604.11401)**

> **作者:** Qilin Zhang; Jinyu Zhu; Olaf Wysocki; Benjamin Busam; Boris Jutzi
>
> **摘要:** Recent semantic 3D Gaussian Splatting (3DGS) methods primarily rely on 2D foundation models, often yielding ambiguous boundaries and limited support for structured urban semantics. While city models such as CityGML encode hierarchically organized semantics together with building geometry, these labels cannot be directly mapped to Gaussian primitives. We present GS4City, a hierarchical semantic Gaussian Splatting method that incorporates city-model priors for urban scene understanding. GS4City derives reliable image-aligned masks from Level of Detail (LoD) 3 CityGML models via two-pass raycasting, explicitly using parent-child relations to validate and recover fine-grained facade elements. It then fuses these geometry-grounded masks with foundation-model predictions to establish scene-consistent instance correspondences, and learns a compact identity encoding for each Gaussian under joint 2D identity supervision and 3D spatial regularization. Experiments on the TUM2TWIN and Gold Coast datasets show that GS4City effectively incorporates structured building semantics into Gaussian scene representations, outperforming existing 2D-driven semantic 3DGS baselines, including LangSplat and Gaga, by up to 15.8 IoU points in coarse building segmentation and 14.2 mIoU points in fine-grained semantic segmentation. By bridging structured city models and photorealistic Gaussian scene representations, GS4City enables semantically queryable and structure-aware urban reconstruction. Code is available at this https URL.
>
---
#### [new 172] Topo-ADV: Generating Topology-Driven Imperceptible Adversarial Point Clouds
- **分类: cs.CV; cs.CG**

- **简介: 该论文属于3D点云安全任务，旨在攻击深度学习模型。提出Topo-ADV方法，通过改变拓扑结构实现高效且不可察觉的对抗样本生成。**

- **链接: [https://arxiv.org/pdf/2604.09879](https://arxiv.org/pdf/2604.09879)**

> **作者:** Gayathry Chandramana Krishnan Nampoothiry; Raghuram Venkatapuram; Anirban Ghosh; Ayan Dutta
>
> **备注:** Under review
>
> **摘要:** Deep neural networks for 3D point cloud understanding have achieved remarkable success in object classification and recognition, yet recent work shows that these models remain highly vulnerable to adversarial perturbations. Existing 3D attacks predominantly manipulate geometric properties such as point locations, curvature, or surface structure, implicitly assuming that preserving global shape fidelity preserves semantic content. In this work, we challenge this assumption and introduce the first topology-driven adversarial attack for point cloud deep learning. Our key insight is that the homological structure of a 3D object constitutes a previously unexplored vulnerability surface. We propose Topo-ADV, an end-to-end differentiable framework that incorporates persistent homology as an explicit optimization objective, enabling gradient-based manipulation of topological features during adversarial example generation. By embedding persistence diagrams through differentiable topological representations, our method jointly optimizes (i) a topology divergence loss that alters persistence, (ii) a misclassification objective, and (iii) geometric imperceptibility constraints that preserve visual plausibility. Experiments demonstrate that subtle topology-driven perturbations consistently achieve up to 100% attack success rates on benchmark datasets such as ModelNet40, ShapeNet Part, and ScanObjectNN using PointNet and DGCNN classifiers, while remaining geometrically indistinguishable from the original point clouds, beating state-of-the-art methods on various perceptibility metrics.
>
---
#### [new 173] Grid2Matrix: Revealing Digital Agnosia in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型研究，旨在解决模型在处理细节丰富图像时的失败问题。通过构建Grid2Matrix基准，发现并分析了视觉-语言模型的“数字失认”现象。**

- **链接: [https://arxiv.org/pdf/2604.09687](https://arxiv.org/pdf/2604.09687)**

> **作者:** Yunkai Zhang; Linda Li; Yingxin Cui; Xiyuan Ruan; Zeyu Zheng; Kezhen Chen; Yi Zhang; Diji Yang
>
> **摘要:** Vision-Language Models (VLMs) excel on many multimodal reasoning benchmarks, but these evaluations often do not require an exhaustive readout of the image and can therefore obscure failures in faithfully capturing all visual details. We introduce Grid2Matrix (G2M), a controlled benchmark in which a model is shown a color grid and a color-to-number mapping, and must output the corresponding matrix. By varying grid size and the number of colors, G2M provides a simple way to increase visual complexity while minimizing semantic confounds. We find that VLMs exhibit a sharp early collapse in zero-shot end-to-end evaluation, failing on surprisingly small grids rather than degrading gradually as the task becomes denser. We probe the visual encoders of VLMs from two representative families and find that they preserve substantially more of the grid information than the corresponding end-to-end outputs. This suggests that the failure is not explained by visual encoding alone, but also reflects a gap between what remains recoverable from visual features and what is ultimately expressed in language. We term this gap \textit{Digital Agnosia}. Further analyses show that these errors are highly structured and depend strongly on how grid cells overlap with visual patch boundaries. We also find that common strategies such as model scaling and multimodal alignment do not fully eliminate this failure mode. We expect G2M to serve as a useful testbed for understanding where and how VLMs lose fine visual details, and for evaluating tasks where missing even small visual details can matter, such as tables, charts, forms, and GUIs.
>
---
#### [new 174] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对RGB-D语义分割中深度缺失或退化的问题，提出GeomPrompt和GeomPrompt-Recovery模块，通过几何提示提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.11585](https://arxiv.org/pdf/2604.11585)**

> **作者:** Krishna Jaganathan; Patricio Vela
>
> **备注:** Accepted to the CVPR 2026 URVIS Workshop. Project page: this https URL
>
> **摘要:** Multimodal perception systems for robotics and embodied AI often assume reliable RGB-D sensing, but in practice, depth is frequently missing, noisy, or corrupted. We thus present GeomPrompt, a lightweight cross-modal adaptation module that synthesizes a task-driven geometric prompt from RGB alone for the fourth channel of a frozen RGB-D semantic segmentation model, without depth supervision. We further introduce GeomPrompt-Recovery, an adaptation module that compensates for degraded depth by predicting the fourth channel correction relevant for the frozen segmenter. Both modules are trained solely with downstream segmentation supervision, enabling recovery of the geometric prior useful for segmentation, rather than estimating depth signals. On SUN RGB-D, GeomPrompt improves over RGB-only inference by +6.1 mIoU on DFormer and +3.0 mIoU on GeminiFusion, while remaining competitive with strong monocular depth estimators. For degraded depth, GeomPrompt-Recovery consistently improves robustness, yielding gains up to +3.6 mIoU under severe depth corruptions. GeomPrompt is also substantially more efficient than monocular depth baselines, reaching 7.8 ms latency versus 38.3 ms and 71.9 ms. These results suggest that task-driven geometric prompting is an efficient mechanism for cross-modal compensation under missing and degraded depth inputs in RGB-D perception.
>
---
#### [new 175] GeoMeld: Toward Semantically Grounded Foundation Models for Remote Sensing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GeoMeld数据集和GeoMeld-FM框架，解决遥感中多模态基础模型的语义对齐问题，提升跨传感器泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.10591](https://arxiv.org/pdf/2604.10591)**

> **作者:** Maram Hasan; Md Aminur Hossain; Savitra Roy; Souparna Bhowmik; Ayush V. Patel; Mainak Singha; Subhasis Chaudhuri; Muhammad Haris Khan; Biplab Banerjee
>
> **备注:** Accepted at CVPR Workshop 2026; 8 pages, 6 figures
>
> **摘要:** Effective foundation modeling in remote sensing requires spatially aligned heterogeneous modalities coupled with semantically grounded supervision, yet such resources remain limited at scale. We present GeoMeld, a large-scale multimodal dataset with approximately 2.5 million spatially aligned samples. The dataset spans diverse modalities and resolutions and is constructed under a unified alignment protocol for modality-aware representation learning. GeoMeld provides semantically grounded language supervision through an agentic captioning framework that synthesizes and verifies annotations from spectral signals, terrain statistics, and structured geographic metadata, encoding measurable cross-modality relationships within textual descriptions. To leverage this dataset, we introduce GeoMeld-FM, a pretraining framework that combines multi-pretext masked autoencoding over aligned modalities, JEPA representation learning, and caption-vision contrastive alignment. This joint objective enables the learned representation space to capture both reliable cross-sensor physical consistency and grounded semantics. Experiments demonstrate consistent gains in downstream transfer and cross-sensor robustness. Together, GeoMeld and GeoMeld-FM establish a scalable reference framework for semantically grounded multi-modal foundation modeling in remote sensing.
>
---
#### [new 176] Ambivalence/Hesitancy Recognition in Videos for Personalized Digital Health Interventions
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 论文属于情感识别任务，旨在解决数字健康干预中自动识别个体的矛盾与犹豫情绪问题。通过深度学习模型分析视频多模态数据，探索监督学习、领域自适应和零样本推理等方法。**

- **链接: [https://arxiv.org/pdf/2604.11730](https://arxiv.org/pdf/2604.11730)**

> **作者:** Manuela González-González; Soufiane Belharbi; Muhammad Osama Zeeshan; Masoumeh Sharafi; Muhammad Haseeb Aslam; Lorenzo Sia; Nicolas Richet; Marco Pedersoli; Alessandro Lameiras Koerich; Simon L Bacon; Eric Granger
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Using behavioural science, health interventions focus on behaviour change by providing a framework to help patients acquire and maintain healthy habits that improve medical outcomes. In-person interventions are costly and difficult to scale, especially in resource-limited regions. Digital health interventions offer a cost-effective approach, potentially supporting independent living and self-management. Automating such interventions, especially through machine learning, has gained considerable attention recently. Ambivalence and hesitancy (A/H) play a primary role for individuals to delay, avoid, or abandon health interventions. A/H are subtle and conflicting emotions that place a person in a state between positive and negative evaluations of a behaviour, or between acceptance and refusal to engage in it. They manifest as affective inconsistency across modalities or within a modality, such as language, facial, vocal expressions, and body language. While experts can be trained to recognize A/H, integrating them into digital health interventions is costly and less effective. Automatic A/H recognition is therefore critical for the personalization and cost-effectiveness of digital health interventions. Here, we explore the application of deep learning models for A/H recognition in videos, a multi-modal task by nature. In particular, this paper covers three learning setups: supervised learning, unsupervised domain adaptation for personalization, and zero-shot inference via large language models (LLMs). Our experiments are conducted on the unique and recently published BAH video dataset for A/H recognition. Our results show limited performance, suggesting that more adapted multi-modal models are required for accurate A/H recognition. Better methods for modeling spatio-temporal and multimodal fusion are necessary to leverage conflicts within/across modalities.
>
---
#### [new 177] F3G-Avatar : Face Focused Full-body Gaussian Avatar
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人体全身体积重建任务，旨在解决现有方法难以保留面部细节的问题。提出F3G-Avatar，通过双分支结构提升面部几何与表情的重建质量。**

- **链接: [https://arxiv.org/pdf/2604.09835](https://arxiv.org/pdf/2604.09835)**

> **作者:** Willem Menu; Erkut Akdag; Pedro Quesado; Yasaman Kashefbahrami; Egor Bondarev
>
> **备注:** CVPRW 3DMV, 10 pages
>
> **摘要:** Existing full-body Gaussian avatar methods primarily optimize global reconstruction quality and often fail to preserve fine-grained facial geometry and expression details. This challenge arises from limited facial representational capacity that causes difficulties in modeling high-frequency pose-dependent deformations. To address this, we propose F3G-Avatar, a full-body, face-aware avatar synthesis method that reconstructs animatable human representations from multi-view RGB video and regressed pose/shape parameters. Starting from a clothed Momentum Human Rig (MHR) template, front/back positional maps are rendered and decoded into 3D Gaussians through a two-branch architecture: a body branch that captures pose-dependent non-rigid deformations and a face-focused deformation branch that refines head geometry and appearance. The predicted Gaussians are fused, posed with linear blend skinning (LBS), and rendered with differentiable Gaussian splatting. Training combines reconstruction and perceptual objectives with a face-specific adversarial loss to enhance realism in close-up views. Experiments demonstrate strong rendering quality, with face-view performance reaching PSNR/SSIM/LPIPS of 26.243/0.964/0.084 on the AvatarReX dataset. Ablations further highlight contributions of the MHR template and the face-focused deformation. F3G-Avatar provides a practical, high-quality pipeline for realistic, animatable full-body avatar synthesis.
>
---
#### [new 178] ConvFormer3D-TAP: Phase/Uncertainty-Aware Front-End Fusion for Cine CMR View Classification Pipelines
- **分类: cs.CV**

- **简介: 该论文属于心脏MRI视图分类任务，旨在解决临床条件下视图识别不准导致的分析误差问题。提出ConvFormer3D-TAP模型，结合3D卷积与注意力机制提升分类准确性。**

- **链接: [https://arxiv.org/pdf/2604.11389](https://arxiv.org/pdf/2604.11389)**

> **作者:** Nafiseh Ghaffar Nia; Vinesh Appadurai; Suchithra V.; Chinmay Rane; Daniel Pittman; James Carr; Adrienne Kline
>
> **摘要:** Reliable recognition of standard cine cardiac MRI views is essential because each view determines which cardiac anatomy is visualized and which quantitative analyses can be performed. Incorrect view identification, whether by a human reader or an automated deep learning system, can propagate errors into segmentation, volumetric assessment, strain analysis, and valve evaluation. However, accurate view classification remains challenging under routine clinical variability in scanner vendor, acquisition protocol, motion artifacts, and plane prescription. We present ConvFormer3D-TAP, a cine-specific spatiotemporal architecture that integrates 3D convolutional tokenization with multiscale self-attention. The model is trained using masked spatiotemporal reconstruction and uncertainty-weighted multi-clip fusion to enhance robustness across cardiac phases and ambiguous temporal segments. The design captures complementary cues: local anatomical structure through convolutional priors and long-range cardiac-cycle dynamics through hierarchical attention. On a cohort of 150,974 clinically acquired cine sequences spanning six standard cine cardiac MRI views, ConvFormer3D-TAP achieved 96% validation accuracy with per-class F1-scores >= 0.94 and strong calibration (ECE = 0.025; Brier = 0.040). Error analysis shows that residual confusions are concentrated in anatomically adjacent long-axis and LVOT/AV view pairs, consistent with intrinsic prescription overlap. These results support ConvFormer3D-TAP as a scalable front-end for view routing, filtering and quality control in end-to-end cMRI workflows.
>
---
#### [new 179] TraversalBench: Challenging Paths to Follow for Vision Language Models
- **分类: cs.CV**

- **简介: 该论文提出TraversalBench，用于评估视觉语言模型在复杂视觉路径追踪任务中的表现，解决模型在持续视觉推理中的局限性。**

- **链接: [https://arxiv.org/pdf/2604.10999](https://arxiv.org/pdf/2604.10999)**

> **作者:** Clara Petrova; Zhuo Chen; Marin Soljačić
>
> **摘要:** Vision-language models (VLMs) perform strongly on many multimodal benchmarks. However, the ability to follow complex visual paths -- a task that human observers typically find straightforward -- remains under-tested. We introduce TraversalBench, a controlled benchmark for exact visual path traversal. Each instance contains a single continuous polyline, a unique start marker, and markers placed at path vertices; the task is to recover the exact ordered sequence encountered when traversing the path from start to finish. The benchmark explicitly balances key path-structural factors including self-intersection count, tortuosity, vertex count, and nearby confounding lines, while minimizing reliance on OCR, world knowledge, and open-ended planning. We find that self-intersections are the dominant source of difficulty. A first-crossing analysis shows that errors are sharply localized: performance is relatively stable immediately before the first crossing, then drops steeply when the model must resolve the correct continuation. By contrast, nearby confounding lines produce a weaker persistent degradation that compounds with repeated exposure. These analyses make TraversalBench a useful diagnostic for identifying whether models suffer from human-like failures or other breakdowns in sustained visual processing. An auxiliary reading-order benchmark further reveals a consistent preference for layouts compatible with left-to-right serialization, while not explaining away the main effects of path complexity. Together, these results position TraversalBench as a controlled diagnostic of path-faithful visual reasoning and as a useful testbed for studying multimodal spatial reasoning under ambiguity, clutter, and distractor structure. More broadly, we position TraversalBench as a contribution to the still-limited area of sustained visual grounding benchmarks for VLMs.
>
---
#### [new 180] VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VGA-Bench，用于视频生成质量与美学评估的统一基准和多模型框架，解决现有评估体系缺乏全面性的问题。**

- **链接: [https://arxiv.org/pdf/2604.10127](https://arxiv.org/pdf/2604.10127)**

> **作者:** Longteng Jiang; DanDan Zheng; Qianqian Qiao; Heng Huang; Huaye Wang; Yihang Bo; Bao Peng; Jingdong Chen; Jun Zhou; Xin Jin
>
> **备注:** CVPR 2026
>
> **摘要:** The rapid advancement of AIGC-based video generation has underscored the critical need for comprehensive evaluation frameworks that go beyond traditional generation quality metrics to encompass aesthetic appeal. However, existing benchmarks remain largely focused on technical fidelity, leaving a significant gap in holistic assessment-particularly with respect to perceptual and artistic qualities. To address this limitation, we introduce VGA-Bench, a unified benchmark for joint evaluation of video generation quality and aesthetic quality. VGA-Bench is built upon a principled three-tier taxonomy: Aesthetic Quality, Aesthetic Tagging, and Generation Quality, each decomposed into multiple fine-grained sub-dimensions to enable systematic assessment. Guided by this taxonomy, we design 1,016 diverse prompts and generate a large-scale dataset of over 60,000 videos using 12 video generation models, ensuring broad coverage across content, style, and artifacts. To enable scalable and automated evaluation, we annotate a subset of the dataset via human labeling and develop three dedicated multi-task neural assessors: VAQA-Net for aesthetic quality prediction, VTag-Net for automatic aesthetic tagging, and VGQA-Net for generation and basic quality attributes. Extensive experiments demonstrate that our models achieve reliable alignment with human judgments, offering both accuracy and efficiency. We release VGA-Bench as a public benchmark to foster research in AIGC evaluation, with applications in content moderation, model debugging, and generative model optimization.
>
---
#### [new 181] SatReg: Regression-based Neural Architecture Search for Lightweight Satellite Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于轻量化卫星图像分割任务，解决边缘设备上的实时性与能耗问题。通过回归方法优化模型结构，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2604.10306](https://arxiv.org/pdf/2604.10306)**

> **作者:** Edward Humes; Tinoosh Mohsenin
>
> **摘要:** As Earth-observation workloads move toward onboard and edge processing, remote-sensing segmentation models must operate under tight latency and energy constraints. We present SatReg, a regression-based hardware-aware tuning framework for lightweight remote-sensing segmentation on edge platforms. Using CM-UNet as the teacher architecture, we reduce the search space to two dominant width-related variables, profile a small set of student models on an NVIDIA Jetson Orin Nano, and fit low-order surrogate models for mIoU, latency, and power. Knowledge distillation is used to efficiently train the sampled students. The learned surrogates enable fast selection of near-optimal architecture settings for deployment targets without exhaustive search. Results show that the selected variables affect task accuracy and hardware cost differently, making reduced-space regression a practical strategy for adapting hybrid CNN-Mamba segmentation models to future space-edge systems.
>
---
#### [new 182] H-SPAM: Hierarchical Superpixel Anything Model
- **分类: cs.CV**

- **简介: 该论文提出H-SPAM，解决超像素分割中精度不足与多尺度表示受限的问题，通过层次化区域合并生成准确且嵌套的超像素。**

- **链接: [https://arxiv.org/pdf/2604.11218](https://arxiv.org/pdf/2604.11218)**

> **作者:** Julien Walther; Rémi Giraud; Michaël Clément
>
> **摘要:** Superpixels offer a compact image representation by grouping pixels into coherent regions. Recent methods have reached a plateau in terms of segmentation accuracy by generating noisy superpixel shapes. Moreover, most existing approaches produce a single fixed-scale partition that limits their use in vision pipelines that would benefit multi-scale representations. In this work, we introduce H-SPAM (Hierarchical Superpixel Anything Model), a unified framework for generating accurate, regular, and perfectly nested hierarchical superpixels. Starting from a fine partition, guided by deep features and external object priors, H-SPAM constructs the hierarchy through a two-phase region merging process that first preserves object consistency and then allows controlled inter-object grouping. The hierarchy can also be modulated using visual attention maps or user input to preserve important regions longer in the hierarchy. Experiments on standard benchmarks show that H-SPAM strongly outperforms existing hierarchical methods in both accuracy and regularity, while performing on par with most recent state-of-the-art non-hierarchical methods. Code and pretrained models are available: this https URL.
>
---
#### [new 183] FlowPalm: Optical Flow Driven Non-Rigid Deformation for Geometrically Diverse Palmprint Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决合成掌纹数据几何多样性不足的问题。提出FlowPalm框架，利用光流模拟真实掌纹的非刚性变形，提升生成数据的真实性。**

- **链接: [https://arxiv.org/pdf/2604.09989](https://arxiv.org/pdf/2604.09989)**

> **作者:** Yuchen Zou; Huikai Shao; Lihuang Fang; Zhipeng Xiong; Dexing Zhong
>
> **摘要:** Recently, synthetic palmprints have been increasingly used as substitutes for real data to train recognition models. To be effective, such synthetic data must reflect the diversity of real palmprints, including both style variation and geometric variation. However, existing palmprint generation methods mainly focus on style translation, while geometric variation is either ignored or approximated by simple handcrafted augmentations. In this work, we propose FlowPalm, an optical-flow-driven palmprint generation framework capable of simulating the complex non-rigid deformations observed in real palms. Specifically, FlowPalm estimates optical flows between real palmprint pairs to capture the statistical patterns of geometric deformations. Building on these priors, we design a progressive sampling process that gradually introduces the geometric deformations during diffusion while maintaining identity consistency. Extensive experiments on six benchmark datasets demonstrate that FlowPalm significantly outperforms state-of-the-art palmprint generation approaches in downstream recognition tasks. Project page: this https URL
>
---
#### [new 184] Progressive Deep Learning for Automated Spheno-Occipital Synchondrosis Maturation Assessment
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决SOS评估的高变异性和低可靠性问题。通过渐进式深度学习框架，提升CBCT图像中SOS成熟度的准确识别。**

- **链接: [https://arxiv.org/pdf/2604.10945](https://arxiv.org/pdf/2604.10945)**

> **作者:** Omid Halimi Milani; Amanda Nikho; Marouane Tliba; Lauren Mills; Emadeldeen Hamdan; Ahmet Enis Cetin; Mohammed H. Elnagar
>
> **摘要:** Accurate assessment of spheno-occipital synchondrosis (SOS) maturation is a key indicator of craniofacial growth and a critical determinant for orthodontic and surgical timing. However, SOS staging from cone-beam CT (CBCT) relies on subtle, continuously evolving morphological cues, leading to high inter-observer variability and poor reproducibility, especially at transitional fusion stages. We frame SOS assessment as a fine-grained visual recognition problem and propose a progressive representation-learning framework that explicitly mirrors how expert clinicians reason about synchondral fusion: from coarse anatomical structure to increasingly subtle patterns of closure. Rather than training a full-capacity network end-to-end, we sequentially grow the model by activating deeper blocks over time, allowing early layers to first encode stable cranial base morphology before higher-level layers specialize in discriminating adjacent maturation stages. This yields a curriculum over network depth that aligns deep feature learning with the biological continuum of SOS fusion. Extensive experiments across convolutional and transformer-based architectures show that this expert-inspired training strategy produces more stable optimization and consistently higher accuracy than standard training, particularly for ambiguous intermediate stages. Importantly, these gains are achieved without changing network architectures or loss functions, demonstrating that training dynamics alone can substantially improve anatomical representation learning. The proposed framework establishes a principled link between expert dental intuition and deep visual representations, enabling robust, data-efficient SOS staging from CBCT and offering a general strategy for modeling other continuous biological processes in medical imaging.
>
---
#### [new 185] At FullTilt: Real-Time Open-Set 3D Macromolecule Detection Directly from Tilted 2D Projections
- **分类: cs.CV**

- **简介: 该论文属于3D分子检测任务，解决cryo-ET中因VRAM限制导致的处理效率低问题。提出FullTilt框架，直接处理2D倾斜序列，提升检测速度与精度。**

- **链接: [https://arxiv.org/pdf/2604.10766](https://arxiv.org/pdf/2604.10766)**

> **作者:** Ming-Yang Ho; Alberto Bartesaghi
>
> **摘要:** Open-set 3D macromolecule detection in cryogenic electron tomography eliminates the need for target-specific model retraining. However, strict VRAM constraints prohibit processing an entire 3D tomogram, forcing current methods to rely on slow sliding-window inference over extracted subvolumes. To overcome this, we propose FullTilt, an end-to-end framework that redefines 3D detection by operating directly on aligned 2D tilt-series. Because a tilt-series contains significantly fewer images than slices in a reconstructed tomogram, FullTilt eliminates redundant volumetric computation, accelerating inference by orders of magnitude. To process the entire tilt-series simultaneously, we introduce a tilt-series encoder to efficiently fuse cross-view information. We further propose a multiclass visual prompt encoder for flexible prompting, a tilt-aware query initializer to effectively anchor 3D queries, and an auxiliary geometric primitives module to enhance the model's understanding of multi-view geometry while improving robustness to adverse imaging artifacts. Extensive evaluations on three real-world datasets demonstrate that FullTilt achieves state-of-the-art zero-shot performance while drastically reducing runtime and VRAM requirements, paving the way for rapid, large-scale visual proteomics analysis. All code and data will be publicly available upon publication.
>
---
#### [new 186] Training Deep Visual Networks Beyond Loss and Accuracy Through a Dynamical Systems Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉识别任务，旨在通过动态系统方法研究深度网络训练过程。提出三种度量指标，分析模型内部表示的变化，以补充传统损失和准确率的不足。**

- **链接: [https://arxiv.org/pdf/2604.09716](https://arxiv.org/pdf/2604.09716)**

> **作者:** Hai La Quang; Hassan Ugail; Newton Howard; Cong Tran Tien; Nam Vu Hoai; Hung Nguyen Viet
>
> **摘要:** Deep visual recognition models are usually trained and evaluated using metrics such as loss and accuracy. While these measures show whether a model is improving, they reveal very little about how its internal representations change during training. This paper introduces a complementary way to study that process by examining training through the lens of dynamical systems. Drawing on ideas from signal analysis originally used to study biological neural activity, we define three measures from layer activations collected across training epochs: an integration score that reflects long-range coordination across layers, a metastability score that captures how flexibly the network shifts between more and less synchronised states, and a combined dynamical stability index. We apply this framework to nine combinations of model architecture and dataset, including several ResNet variants, DenseNet-121, MobileNetV2, VGG-16, and a pretrained Vision Transformer on CIFAR-10 and CIFAR-100. The results suggest three main patterns. First, the integration measure consistently distinguishes the easier CIFAR-10 setting from the more difficult CIFAR-100 setting. Second, changes in the volatility of the stability index may provide an early sign of convergence before accuracy fully plateaus. Third, the relationship between integration and metastability appears to reflect different styles of training behaviour. Overall, this study offers an exploratory but promising new way to understand deep visual training beyond loss and accuracy.
>
---
#### [new 187] I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，探讨TTA在测试阶段是否提升准确率。研究发现TTA在多数情况下反而降低精度，分析其原因并提出验证建议。**

- **链接: [https://arxiv.org/pdf/2604.09697](https://arxiv.org/pdf/2604.09697)**

> **作者:** Daniel Nobrega Medeiros
>
> **备注:** 9 pages, 7 figures, 2 tables
>
> **摘要:** Test-time augmentation (TTA)--aggregating predictions over multiple augmented copies of a test input--is widely assumed to improve classification accuracy, particularly in medical imaging where it is routinely deployed in production systems and competition solutions. We present a systematic empirical study challenging this assumption across three MedMNIST v2 benchmarks and four architectures spanning three orders of magnitude in parameter count (21K to 11M). Our principal finding is that TTA with standard augmentation pipelines consistently degrades accuracy relative to single-pass inference, with drops as severe as 31.6 percentage points for ResNet-18 on pathology images. This degradation affects all architectures, including convolutional models, and worsens with more augmented views. The sole exception is ResNet-18 on dermatology images, which gains a modest +1.6%. We identify the distribution shift between augmented and training-time inputs--amplified by batch normalization statistics mismatch--as the primary mechanism. Our ablation studies show that augmentation strategy matters critically: intensity-only augmentations preserve more performance than geometric transforms, and including the original unaugmented image partially mitigates but does not eliminate the accuracy drop. These findings serve as a cautionary note for practitioners: TTA should not be applied as a default post-hoc improvement but must be validated on the specific model-dataset combination.
>
---
#### [new 188] STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于4D点云视频理解任务，旨在解决现有方法难以捕捉几何特征的问题。通过引入频谱分析，设计STS-Mixer框架，融合时空与频谱信息，提升4D点云视频的表示与理解能力。**

- **链接: [https://arxiv.org/pdf/2604.11637](https://arxiv.org/pdf/2604.11637)**

> **作者:** Wenhao Li; Xueying Jiang; Gongjie Zhang; Xiaoqin Zhang; Ling Shao; Shijian Lu
>
> **备注:** Accepted by CVPR 2026, Open Sourced
>
> **摘要:** 4D point cloud videos capture rich spatial and temporal dynamics of scenes which possess unique values in various 4D understanding tasks. However, most existing methods work in the spatiotemporal domain where the underlying geometric characteristics of 4D point cloud videos are hard to capture, leading to degraded representation learning and understanding of 4D point cloud videos. We address the above challenge from a complementary spectral perspective. By transforming 4D point cloud videos into graph spectral signals, we can decompose them into multiple frequency bands each of which captures distinct geometric structures of point cloud videos. Our spectral analysis reveals that the decomposed low-frequency signals capture more coarse shapes while high-frequency signals encode more fine-grained geometry details. Building on these observations, we design Spatio-Temporal-Spectral Mixer (STS-Mixer), a unified framework that mixes spatial, temporal, and spectral representations of point cloud videos. STS-Mixer integrates multi-band delineated spectral signals with spatiotemporal information to capture rich geometries and temporal dynamics, while enabling fine-grained and holistic understanding of 4D point cloud videos. Extensive experiments show that STS-Mixer achieves superior performance consistently across multiple widely adopted benchmarks on both 3D action recognition and 4D semantic segmentation tasks. Code and models are available at this https URL.
>
---
#### [new 189] 3DTV: A Feedforward Interpolation Network for Real-Time View Synthesis
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 该论文提出3DTV，解决实时视图合成问题。通过轻量几何与学习结合，实现高效稀疏视角插值，无需场景优化，适用于AR/VR等交互应用。**

- **链接: [https://arxiv.org/pdf/2604.11211](https://arxiv.org/pdf/2604.11211)**

> **作者:** Stefan Schulz; Fernando Edelstein; Hannah Dröge; Matthias B. Hullin; Markus Plack
>
> **摘要:** Real-time free-viewpoint rendering requires balancing multi-camera redundancy with the latency constraints of interactive applications. We address this challenge by combining lightweight geometry with learning and propose 3DTV, a feedforward network for real-time sparse-view interpolation. A Delaunay-based triplet selection ensures angular coverage for each target view. Building on this, we introduce a pose-aware depth module that estimates a coarse-to-fine depth pyramid, enabling efficient feature reprojection and occlusion-aware blending. Unlike methods that require scene-specific optimization, 3DTV runs feedforward without retraining, making it practical for AR/VR, telepresence, and interactive applications. Our experiments on challenging multi-view video datasets demonstrate that 3DTV consistently achieves a strong balance of quality and efficiency, outperforming recent real-time novel-view baselines. Crucially, 3DTV avoids explicit proxies, enabling robust rendering across diverse scenes. This makes it a practical solution for low-latency multi-view streaming and interactive rendering. Project Page: this https URL
>
---
#### [new 190] TAG-Head: Time-Aligned Graph Head for Plug-and-Play Fine-grained Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于细粒度人体动作识别任务，解决视觉相似动作识别困难的问题。提出TAG-Head，通过轻量图结构提升3D主干网络性能，仅使用RGB视频即可达到先进效果。**

- **链接: [https://arxiv.org/pdf/2604.11498](https://arxiv.org/pdf/2604.11498)**

> **作者:** Imtiaz Ul Hassan; Nik Bessis; Ardhendu Behera
>
> **备注:** 15 pages, 3 figures, to appear in ICPR 2026
>
> **摘要:** Fine-grained human action recognition (FHAR) is challenging because visually similar actions differ by subtle spatio-temporal cues. Many recent systems enhance discriminability with extra modalities (e.g., pose, text, optical flow), but this increases annotation burden and computational cost. We introduce TAG-Head, a lightweight spatio-temporal graph head that upgrades standard 3D backbones (SlowFast, R(2+1)D-34, I3D, etc.) for FHAR using RGB only. Our pipeline first applies a Transformer encoder with learnable 3D positional encodings to the backbone tokens, capturing long-range dependencies across space and time. The resulting features are then refined by a graph in which (i) fully-connected intra-frame edges to resolve subtle appearance differences within frames, and (ii) time-aligned temporal edges that connect features at the same spatial location across frames to stabilise motion cues without over-smoothing. The head is compact (little parameter/FLOP overhead), plug-and-play across backbones, and trained end-to-end with the backbone. Extensive evaluations on FineGym (Gym99 and Gym288) and HAA500 show that TAG-Head sets a new state-of-the-art among RGB-only models and surpasses many recent multimodal approaches (video + pose + text) that rely on privileged information. Ablations disentangle the contributions of the Transformer and the graph topology, and complexity analyses confirm low latency. TAG-Head advances FHAR by explicitly coupling global context with high-resolution spatial interactions and low-variance temporal continuity inside a slim, composable graph head. The simplicity of the design enables straightforward adoption in practical systems that favour RGB-only sensors, while delivering performance gains typically associated with heavier or multimodal models. Code will be released on GitHub.
>
---
#### [new 191] LOLGORITHM: Funny Comment Generation Agent For Short Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于短视频评论生成任务，旨在解决现有方法无法生成符合平台文化规范的评论问题。提出LOLGORITHM框架，实现多种风格评论生成，并通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.09729](https://arxiv.org/pdf/2604.09729)**

> **作者:** Xuan Ouyang; Senan Wang; Bouzhou Wang; Siyuan Xiahou; Jinrong Zhou; Yuekang Li
>
> **摘要:** Short-form video platforms have become central to multimedia information dissemination, where comments play a critical role in driving engagement, propagation, and algorithmic feedback. However, existing approaches -- including video summarization and live-streaming danmaku generation -- fail to produce authentic comments that conform to platform-specific cultural and linguistic norms. In this paper, we present LOLGORITHM, a novel modular multi-agent framework for stylized short-form video comment generation. LOLGORITHM supports six controllable comment styles and comprises three core modules: video content summarization, video classification, and comment generation with semantic retrieval and hot meme augmentation. We further construct a bilingual dataset of 3,267 videos and 16,335 comments spanning five high-engagement categories across YouTube and Douyin. Evaluation combining automatic scoring and large-scale human preference analysis demonstrates that LOLGORITHM consistently outperforms baseline methods, achieving human preference selection rates of 80.46\% on YouTube and 84.29\% on Douyin across 107 respondents. Ablation studies confirm that these gains are attributable to the framework architecture rather than the choice of backbone LLM, underscoring the robustness and generalizability of our approach.
>
---
#### [new 192] Attention-Guided Flow-Matching for Sparse 3D Geological Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于地质建模任务，解决从稀疏数据生成高分辨率3D地质模型的问题。提出3D-GeoFlow框架，结合注意力机制与流匹配，提升结构一致性与真实性。**

- **链接: [https://arxiv.org/pdf/2604.09700](https://arxiv.org/pdf/2604.09700)**

> **作者:** Zhixiang Lu; Mengqi Han; Peixin Guo; Tianming Bai; Jionglong Su; Fei Fang; Sifan Song
>
> **摘要:** Constructing high-resolution 3D geological models from sparse 1D borehole and 2D surface data is a highly ill-posed inverse problem. Traditional heuristic and implicit modeling methods fundamentally fail to capture non-linear topological discontinuities under extreme sparsity, often yielding unrealistic artifacts. Furthermore, while deep generative architectures like Diffusion Models have revolutionized continuous domains, they suffer from severe representation collapse when conditioned on sparse categorical grids. To bridge this gap, we propose 3D-GeoFlow, the first Attention-Guided Continuous Flow Matching framework tailored for sparse multimodal geological modeling. By reformulating discrete categorical generation as a simulation-free, continuous vector field regression optimized via Mean Squared Error, our model establishes stable, deterministic optimal transport paths. Crucially, we integrate 3D Attention Gates to dynamically propagate localized borehole features across the volumetric latent space, ensuring macroscopic structural coherence. To validate our framework, we curated a large-scale multimodal dataset comprising 2,200 procedurally generated 3D geological cases. Extensive out-of-distribution (OOD) evaluations demonstrate that 3D-GeoFlow achieves a paradigm shift, significantly outperforming heuristic interpolations and standard diffusion baselines.
>
---
#### [new 193] UHD-GPGNet: UHD Video Denoising via Gaussian-Process-Guided Local Spatio-Temporal Modeling
- **分类: cs.CV**

- **简介: 该论文属于视频去噪任务，解决UHD视频中复杂时空退化问题，提出UHD-GPGNet框架，通过高斯过程引导的局部时空建模实现高效去噪与细节保留。**

- **链接: [https://arxiv.org/pdf/2604.11014](https://arxiv.org/pdf/2604.11014)**

> **作者:** Weiyuan He; Chen Wu; Pengwen Dai; Wei Wang; Dianjie Lu; Guijuan Zhang; Linwei Fan; Yongzhen Wang; Zhuoran Zheng
>
> **摘要:** Ultra-high-definition (UHD) video denoising requires simultaneously suppressing complex spatio-temporal degradations, preserving fine textures and chromatic stability, and maintaining efficient full-resolution 4K deployment. In this paper, we propose UHD-GPGNet, a Gaussian-process-guided local spatio-temporal denoising framework that addresses these requirements jointly. Rather than relying on implicit feature learning alone, the method estimates sparse GP posterior statistics over compact spatio-temporal descriptors to explicitly characterize local degradation response and uncertainty, which then guide adaptive temporal-detail fusion. A structure-color collaborative reconstruction head decouples luminance, chroma, and high-frequency correction, while a heteroscedastic objective and overlap-tiled inference further stabilize optimization and enable memory-bounded 4K deployment. Experiments on UVG and RealisVideo-4K show that UHD-GPGNet achieves competitive restoration fidelity with substantially fewer parameters than existing methods, enables real-time full-resolution 4K inference with significant speedup over the closest quality competitor, and maintains robust performance across a multi-level mixed-degradation schedule.A real-world study on phone-captured 4K video further confirms that the model, trained entirely on synthetic degradation, generalizes to unseen real sensor noise and improves downstream object detection under challenging conditions.
>
---
#### [new 194] Attention-Guided Dual-Stream Learning for Group Engagement Recognition: Fusing Transformer-Encoded Motion Dynamics with Scene Context via Adaptive Gating
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于群体参与度识别任务，解决传统方法多针对个体或在线课堂的问题。提出DualEngage框架，融合运动动态与场景信息，提升群体参与度识别效果。**

- **链接: [https://arxiv.org/pdf/2604.10078](https://arxiv.org/pdf/2604.10078)**

> **作者:** Saniah Kayenat Chowdhury; Muhammad E.H. Chowdhury
>
> **摘要:** Student engagement is crucial for improving learning outcomes in group activities. Highly engaged students perform better both individually and contribute to overall group success. However, most existing automated engagement recognition methods are designed for online classrooms or estimate engagement at the individual level. Addressing this gap, we propose DualEngage, a novel two-stream framework for group-level engagement recognition from in-classroom videos. It models engagement as a joint function of both individual and group-level behaviors. The primary stream models person-level motion dynamics by detecting and tracking students, extracting dense optical flow with the Recurrent All-Pairs Field Transforms network, encoding temporal motion patterns using a transformer encoder, and finally aggregating per-student representations through attention pooling into a unified representation. The secondary stream captures scene-level spatiotemporal information from the full video clip, leveraging a pretrained three-dimensional Residual Network. The two-stream representations are combined via softmax-gated fusion, which dynamically weights each stream's contribution based on the joint context of both features. DualEngage learns a joint representation of individual actions with overarching group dynamics. We evaluate the proposed approach using fivefold cross-validation on the Classroom Group Engagement Dataset developed by Ocean University of China, achieving an average classification accuracy of 0.9621+/-0.0161 with a macro-averaged F1 of 0.9530+/-0.0204. To understand the contribution of each branch, we further conduct an ablation study comparing single-stream variants against the two-stream model. This work is among the first in classroom engagement recognition to adopt a dual-stream design that explicitly leverages motion cues as an estimator.
>
---
#### [new 195] Finetune Like You Pretrain: Boosting Zero-shot Adversarial Robustness in Vision-language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的对抗鲁棒性增强任务，旨在提升CLIP模型在零样本场景下的对抗攻击防御能力。通过模仿预训练过程进行对抗微调，并引入正则化策略，有效提升了模型的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11576](https://arxiv.org/pdf/2604.11576)**

> **作者:** Songlong Xing; Weijie Wang; Zhengyu Zhao; Jindong Gu; Philip Torr; Nicu Sebe
>
> **备注:** Accepted to CVPR Findings Track 2026
>
> **摘要:** Despite their impressive zero-shot abilities, vision-language models such as CLIP have been shown to be susceptible to adversarial attacks. To enhance its adversarial robustness, recent studies finetune the pretrained vision encoder of CLIP with adversarial examples on a proxy dataset such as ImageNet by aligning adversarial images with correct class labels. However, these methods overlook the important roles of training data distributions and learning objectives, resulting in reduced zero-shot capabilities and limited transferability of robustness across domains and datasets. In this work, we propose a simple yet effective paradigm AdvFLYP, which follows the training recipe of CLIP's pretraining process when performing adversarial finetuning to the model. Specifically, AdvFLYP finetunes CLIP with adversarial images created based on image-text pairs collected from the web, and match them with their corresponding texts via a contrastive loss. To alleviate distortion of adversarial image embeddings of noisy web images, we further propose to regularise AdvFLYP by penalising deviation of adversarial image features. We show that logit- and feature-level regularisation terms benefit robustness and clean accuracy, respectively. Extensive experiments on 14 downstream datasets spanning various domains show the superiority of our paradigm over mainstream practices. Our code and model weights are released at this https URL.
>
---
#### [new 196] A3-FPN: Asymptotic Content-Aware Pyramid Attention Network for Dense Visual Prediction
- **分类: cs.CV**

- **简介: 该论文属于密集视觉预测任务，旨在解决对象尺度变化和小目标识别问题。提出A3-FPN网络，通过多尺度特征增强和注意力机制提升性能。**

- **链接: [https://arxiv.org/pdf/2604.10210](https://arxiv.org/pdf/2604.10210)**

> **作者:** Meng'en Qin; Yu Song; Quanling Zhao; Xiaodong Yang; Yingtao Che; Xiaohui Yang
>
> **摘要:** Learning multi-scale representations is the common strategy to tackle object scale variation in dense prediction tasks. Although existing feature pyramid networks have greatly advanced visual recognition, inherent design defects inhibit them from capturing discriminative features and recognizing small objects. In this work, we propose Asymptotic Content-Aware Pyramid Attention Network (A3-FPN), to augment multi-scale feature representation via the asymptotically disentangled framework and content-aware attention modules. Specifically, A3-FPN employs a horizontally-spread column network that enables asymptotically global feature interaction and disentangles each level from all hierarchical representations. In feature fusion, it collects supplementary content from the adjacent level to generate position-wise offsets and weights for context-aware resampling, and learns deep context reweights to improve intra-category similarity. In feature reassembly, it further strengthens intra-scale discriminative feature learning and reassembles redundant features based on information content and spatial variation of feature maps. Extensive experiments on MS COCO, VisDrone2019-DET and Cityscapes demonstrate that A3-FPN can be easily integrated into state-of-the-art CNN and Transformer-based architectures, yielding remarkable performance gains. Notably, when paired with OneFormer and Swin-L backbone, A3-FPN achieves 49.6 mask AP on MS COCO and 85.6 mIoU on Cityscapes. Codes are available at this https URL.
>
---
#### [new 197] UNIGEOCLIP: Unified Geospatial Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于多模态地理信息学习任务，旨在解决不同地理数据对齐问题。提出UNIGEOCLIP框架，实现五种模态的统一嵌入空间对齐，提升地理信息检索与推理性能。**

- **链接: [https://arxiv.org/pdf/2604.11668](https://arxiv.org/pdf/2604.11668)**

> **作者:** Guillaume Astruc; Eduard Trulls; Jan Hosang; Loic Landrieu; Paul-Edouard Sarlin
>
> **摘要:** The growing availability of co-located geospatial data spanning aerial imagery, street-level views, elevation models, text, and geographic coordinates offers a unique opportunity for multimodal representation learning. We introduce UNIGEOCLIP, a massively multimodal contrastive framework to jointly align five complementary geospatial modalities in a single unified embedding space. Unlike prior approaches that fuse modalities or rely on a central pivot representation, our method performs all-to-all contrastive alignment, enabling seamless comparison, retrieval, and reasoning across arbitrary combinations of modalities. We further propose a scaled latitude-longitude encoder that improves spatial representation by capturing multi-scale geographic structure. Extensive experiments across downstream geospatial tasks demonstrate that UNIGEOCLIP consistently outperforms single-modality contrastive models and coordinate-only baselines, highlighting the benefits of holistic multimodal geospatial alignment. A reference implementation is available at this https URL.
>
---
#### [new 198] MorphoFlow: Sparse-Supervised Generative Shape Modeling with Adaptive Latent Relevance
- **分类: cs.CV**

- **简介: 该论文提出MorphoFlow，用于稀疏监督下的生成式形状建模，解决传统方法依赖密集标注和固定潜在表示的问题，提升模型的可扩展性和灵活性。**

- **链接: [https://arxiv.org/pdf/2604.11636](https://arxiv.org/pdf/2604.11636)**

> **作者:** Mokshagna Sai Teja Karanam; Tushar Kataria; Shireen Elhabian
>
> **摘要:** Statistical shape modeling (SSM) is central to population level analysis of anatomical variability, yet most existing approaches rely on densely annotated segmentations and fixed latent representations. These requirements limit scalability and reduce flexibility when modeling complex anatomical variation. We introduce MorphoFlow, a sparse supervised generative shape modeling framework that learns compact probabilistic shape representations directly from sparse surface annotations. MorphoFlow integrates neural implicit shape representations with an autodecoder formulation and autoregressive normalizing flows to learn an expressive probabilistic density over the latent shape space. The neural implicit representation enables resolution-agnostic modeling of 3D anatomy, while the autodecoder formulation supports direct optimization of per-instance latent codes under sparse supervision. The autoregressive flow captures the distribution of latent anatomical variability providing a tractable, likelihood-based generative model of shapes. To promote compact and structured latent representations, we incorporate adaptive latent relevance weighting through sparsity-inducing priors, enabling the model to regulate the contribution of individual latent dimensions according to their relevance to the underlying anatomical variation while preserving generative expressivity. The resulting latent space supports uncertainty quantification and anatomically plausible shape synthesis without manual latent dimensionality tuning. Evaluation on publicly available lumbar vertebrae and femur datasets demonstrates accurate high-resolution reconstruction from sparse inputs and recovery of structured modes of anatomical variation consistent with population level trends.
>
---
#### [new 199] PAS: Estimating the target accuracy before domain adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于领域自适应任务，旨在解决如何在无目标域标签的情况下选择合适的源域和预训练模型。提出PAS评分，用于估计迁移效果，提升目标域准确率。**

- **链接: [https://arxiv.org/pdf/2604.09863](https://arxiv.org/pdf/2604.09863)**

> **作者:** Raphaella Diniz; Jackson de Faria; Martin Ester
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** The goal of domain adaptation is to make predictions for unlabeled samples from a target domain with the help of labeled samples from a different but related source domain. The performance of domain adaptation methods is highly influenced by the choice of source domain and pre-trained feature extractor. However, the selection of source data and pre-trained model is not trivial due to the absence of a labeled validation set for the target domain and the large number of available pre-trained models. In this work, we propose PAS, a novel score designed to estimate the transferability of a source domain set and a pre-trained feature extractor to a target classification task before actually performing domain adaptation. PAS leverages the generalization power of pre-trained models and assesses source-target compatibility based on the pre-trained feature embeddings. We integrate PAS into a framework that indicates the most relevant pre-trained model and source domain among multiple candidates, thus improving target accuracy while reducing the computational overhead. Extensive experiments on image classification benchmarks demonstrate that PAS correlates strongly with actual target accuracy and consistently guides the selection of the best-performing pre-trained model and source domain for adaptation.
>
---
#### [new 200] Does Your VFM Speak Plant? The Botanical Grammar of Vision Foundation Models for Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决零样本视觉基础模型在农业场景中的性能问题，通过优化提示词提升检测效果，并验证其跨域迁移能力。**

- **链接: [https://arxiv.org/pdf/2604.09920](https://arxiv.org/pdf/2604.09920)**

> **作者:** Lars Lundqvist; Earl Ranario; Hamid Kamangir; Heesup Yun; Christine Diepenbrock; Brian N. Bailey; J. Mason Earles
>
> **摘要:** Vision foundation models (VFMs) offer the promise of zero-shot object detection without task-specific training data, yet their performance in complex agricultural scenes remains highly sensitive to text prompt construction. We present a systematic prompt optimization framework evaluating four open-vocabulary detectors -- YOLO World, SAM3, Grounding DINO, and OWLv2 -- for cowpea flower and pod detection across synthetic and real field imagery. We decompose prompts into eight axes and conduct one-factor-at-a-time analysis followed by combinatorial optimization, revealing that models respond divergently to prompt structure: conditions that optimize one architecture can collapse another. Applying model-specific combinatorial prompts yields substantial gains over a naive species-name baseline, including +0.357 mAP@0.5 for YOLO World and +0.362 mAP@0.5 for OWLv2 on synthetic cowpea flower data. To evaluate cross-task generalization, we use an LLM to translate the discovered axis structure to a morphologically distinct target -- cowpea pods -- and compare against prompting using the discovered optimal structures from synthetic flower data. Crucially, prompt structures optimized exclusively on synthetic data transfer effectively to real-world fields: synthetic-pipeline prompts match or exceed those discovered on labeled real data for the majority of model-object combinations (flower: 0.374 vs. 0.353 for YOLO World; pod: 0.429 vs. 0.371 for SAM3). Our findings demonstrate that prompt engineering can substantially close the gap between zero-shot VFMs and supervised detectors without requiring manual annotation, and that optimal prompts are model-specific, non-obvious, and transferable across domains.
>
---
#### [new 201] UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人体姿态估计任务，解决低光环境下标注数据不足和视觉信息丢失的问题。提出UDAPose框架，通过图像合成和动态注意力机制提升姿态估计性能。**

- **链接: [https://arxiv.org/pdf/2604.10485](https://arxiv.org/pdf/2604.10485)**

> **作者:** Haopeng Chen; Yihao Ai; Kabeen Kim; Robby T. Tan; Yixin Chen; Bo Wang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Low-visibility scenarios, such as low-light conditions, pose significant challenges to human pose estimation due to the scarcity of annotated low-light datasets and the loss of visual information under poor illumination. Recent domain adaptation techniques attempt to utilize well-lit labels by augmenting well-lit images to mimic low-light conditions. But handcrafted augmentations oversimplify noise patterns, while learning-based methods often fail to preserve high-frequency low-light characteristics, producing unrealistic images that lead pose models to generalize poorly to real low-light scenes. Moreover, recent pose estimators rely on image cues through image-to-keypoint cross-attention, but these cues become unreliable under low-light conditions. To address these issues, we propose Unsupervised Domain Adaptation for Pose Estimation (UDAPose), a novel framework that synthesizes low-light images and dynamically fuses visual cues with pose priors for improved pose estimation. Specifically, our synthesis method incorporates a Direct-Current-based High-Pass Filter (DHF) and a Low-light Characteristics Injection Module (LCIM) to inject high-frequency details from input low-light images, overcoming rigidity or the detail loss in existing approaches. Furthermore, we introduce a Dynamic Control of Attention (DCA) module that adaptively balances image cues with learned pose priors in the Transformer architecture. Experiments show that UDAPose outperforms state-of-the-art methods, with notable AP gains of 10.1 (56.4%) on the ExLPose-test hard set (LL-H) and 7.4 (31.4%) in cross-dataset validation on EHPT-XC. Code: this https URL
>
---
#### [new 202] Do Instance Priors Help Weakly Supervised Semantic Segmentation?
- **分类: cs.CV**

- **简介: 该论文属于弱监督语义分割任务，旨在降低标注成本。通过改进SAM模型，利用弱标签进行有效分割，提升分割质量。**

- **链接: [https://arxiv.org/pdf/2604.11170](https://arxiv.org/pdf/2604.11170)**

> **作者:** Anurag Das; Anna Kukleva; Xinting Hu; Yuki M. Asano; Bernt Schiele
>
> **备注:** 23 pages, 15 figures
>
> **摘要:** Semantic segmentation requires dense pixel-level annotations, which are costly and time-consuming to acquire. To address this, we present SeSAM, a framework that uses a foundational segmentation model, i.e. Segment Anything Model (SAM), with weak labels, including coarse masks, scribbles, and points. SAM, originally designed for instance-based segmentation, cannot be directly used for semantic segmentation tasks. In this work, we identify specific challenges faced by SAM and determine appropriate components to adapt it for class-based segmentation using weak labels. Specifically, SeSAM decomposes class masks into connected components, samples point prompts along object skeletons, selects SAM masks using weak-label coverage, and iteratively refines labels using pseudo-labels, enabling SAM-generated masks to be effectively used for semantic segmentation. Integrated with a semi-supervised learning framework, SeSAM balances ground-truth labels, SAM-based pseudo-labels, and high-confidence pseudo-labels, significantly improving segmentation quality. Extensive experiments across multiple benchmarks and weak annotation types show that SeSAM consistently outperforms weakly supervised baselines while substantially reducing annotation cost relative to fine supervision.
>
---
#### [new 203] Script-a-Video: Deep Structured Audio-visual Captions via Factorized Streams and Relational Grounding
- **分类: cs.CV**

- **简介: 该论文提出MTSS框架，解决视频描述中信息耦合导致的表达不准确和扩展性差问题，通过分层流和关系锚定提升视频理解与生成效果。**

- **链接: [https://arxiv.org/pdf/2604.11244](https://arxiv.org/pdf/2604.11244)**

> **作者:** Tencent Hunyuan Team
>
> **摘要:** Advances in Multimodal Large Language Models (MLLMs) are transforming video captioning from a descriptive endpoint into a semantic interface for both video understanding and generation. However, the dominant paradigm still casts videos as monolithic narrative paragraphs that entangle visual, auditory, and identity information. This dense coupling not only compromises representational fidelity but also limits scalability, since even local edits can trigger global rewrites. To address this structural bottleneck, we propose Multi-Stream Scene Script (MTSS), a novel paradigm that replaces monolithic text with factorized and explicitly grounded scene descriptions. MTSS is built on two core principles: Stream Factorization, which decouples a video into complementary streams (Reference, Shot, Event, and Global), and Relational Grounding, which reconnects these isolated streams through explicit identity and temporal links to maintain holistic video consistency. Extensive experiments demonstrate that MTSS consistently enhances video understanding across various models, achieving an average reduction of 25% in the total error rate on Video-SALMONN-2 and an average performance gain of 67% on the Daily-Omni reasoning benchmark. It also narrows the performance gap between smaller and larger MLLMs, indicating a substantially more learnable caption interface. Finally, even without architectural adaptation, replacing monolithic prompts with MTSS in multi-shot video generation yields substantial human-rated improvements: a 45% boost in cross-shot identity consistency, a 56% boost in audio-visual alignment, and a 71% boost in temporal controllability.
>
---
#### [new 204] Hierarchical Textual Knowledge for Enhanced Image Clustering
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于图像聚类任务，旨在解决视觉相似但语义不同的类别难以区分的问题。通过构建层次化文本知识增强特征，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2604.11144](https://arxiv.org/pdf/2604.11144)**

> **作者:** Yijie Zhong; Yunfan Gao; Weipeng Jiang; Haofen Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Image clustering aims to group images in an unsupervised fashion. Traditional methods focus on knowledge from visual space, making it difficult to distinguish between visually similar but semantically different classes. Recent advances in vision-language models enable the use of textual knowledge to enhance image clustering. However, most existing methods rely on coarse class labels or simple nouns, overlooking the rich conceptual and attribute-level semantics embedded in textual space. In this paper, we propose a knowledge-enhanced clustering (KEC) method that constructs a hierarchical concept-attribute structured knowledge with the help of large language models (LLMs) to guide clustering. Specifically, we first condense redundant textual labels into abstract concepts and then automatically extract discriminative attributes for each single concept and similar concept pairs, via structured prompts to LLMs. This knowledge is instantiated for each input image to achieve the knowledge-enhanced features. The knowledge-enhanced features with original visual features are adapted to various downstream clustering algorithms. We evaluate KEC on 20 diverse datasets, showing consistent improvements across existing methods using additional textual knowledge. KEC without training outperforms zero-shot CLIP on 14 out of 20 datasets. Furthermore, the naive use of textual knowledge may harm clustering performance, while KEC provides both accuracy and robustness.
>
---
#### [new 205] Naka-GS: A Bionics-inspired Dual-Branch Naka Correction and Progressive Point Pruning for Low-Light 3DGS
- **分类: cs.CV**

- **简介: 该论文属于低光环境下3D重建任务，旨在解决低光导致的图像质量差、几何误差等问题。提出NAKA-GS框架，结合色彩校正与点云优化，提升重建效果。**

- **链接: [https://arxiv.org/pdf/2604.11142](https://arxiv.org/pdf/2604.11142)**

> **作者:** Runyu Zhu; SiXun Dong; Zhiqiang Zhang; Qingxia Ye; Zhihua Xu
>
> **摘要:** Low-light conditions severely hinder 3D restoration and reconstruction by degrading image visibility, introducing color distortions, and contaminating geometric priors for downstream optimization. We present NAKA-GS, a bionics-inspired framework for low-light 3D Gaussian Splatting that jointly improves photometric restoration and geometric initialization. Our method starts with a Naka-guided chroma-correction network, which combines physics-prior low-light enhancement, dual-branch input modeling, frequency-decoupled correction, and mask-guided optimization to suppress bright-region chromatic artifacts and edge-structure errors. The enhanced images are then fed into a feed-forward multi-view reconstruction model to produce dense scene priors. To further improve Gaussian initialization, we introduce a lightweight Point Preprocessing Module (PPM) that performs coordinate alignment, voxel pooling, and distance-adaptive progressive pruning to remove noisy and redundant points while preserving representative structures. Without introducing heavy inference overhead, NAKA-GS improves restoration quality, training stability, and optimization efficiency for low-light 3D reconstruction. The proposed method was presented in the NTIRE 3D Restoration and Reconstruction (3DRR) Challenge, and outperformed the baseline methods by a large margin. The code is available at this https URL
>
---
#### [new 206] ReContraster: Making Your Posters Stand Out with Regional Contrast
- **分类: cs.CV**

- **简介: 该论文提出ReContraster，用于提升海报视觉吸引力。属于海报设计任务，解决如何通过区域对比使海报更突出的问题，采用多智能体系统和混合去噪策略生成优质海报。**

- **链接: [https://arxiv.org/pdf/2604.10442](https://arxiv.org/pdf/2604.10442)**

> **作者:** Peixuan Zhang; Zijian Jia; Ziqi Cai; Shuchen Weng; Si Li; Boxin Shi
>
> **摘要:** Effective poster design requires rapidly capturing attention and clearly conveying messages. Inspired by the ``contrast effects'' principle, we propose ReContraster, the first training-free model to leverage regional contrast to make posters stand out. By emulating the cognitive behaviors of a poster designer, ReContraster introduces the compositional multi-agent system to identify elements, organize layout, and evaluate generated poster candidates. To further ensure harmonious transitions across region boundaries, ReContraster integrates the hybrid denoising strategy during the diffusion process. We additionally contribute a new benchmark dataset for comprehensive evaluation. Seven quantitative metrics and four user studies confirm its superiority over relevant state-of-the-art methods, producing visually striking and aesthetically appealing posters.
>
---
#### [new 207] HiddenObjects: Scalable Diffusion-Distilled Spatial Priors for Object Placement
- **分类: cs.CV**

- **简介: 该论文提出一种方法，通过蒸馏扩散模型中的隐式位置知识，学习显式的物体放置先验，用于自然场景中的物体布局。解决传统方法依赖标注数据或存在伪影的问题，构建了大规模数据集并提升了物体放置效果。**

- **链接: [https://arxiv.org/pdf/2604.10675](https://arxiv.org/pdf/2604.10675)**

> **作者:** Marco Schouten; Ioannis Siglidis; Serge Belongie; Dim P. Papadopoulos
>
> **摘要:** We propose a method to learn explicit, class-conditioned spatial priors for object placement in natural scenes by distilling the implicit placement knowledge encoded in text-conditioned diffusion models. Prior work relies either on manually annotated data, which is inherently limited in scale, or on inpainting-based object-removal pipelines, whose artifacts promote shortcut learning. To address these limitations, we introduce a fully automated and scalable framework that evaluates dense object placements on high-quality real backgrounds using a diffusion-based inpainting pipeline. With this pipeline, we construct HiddenObjects, a large-scale dataset comprising 27M placement annotations, evaluated across 27k distinct scenes, with ranked bounding box insertions for different images and object categories. Experimental results show that our spatial priors outperform sparse human annotations on a downstream image editing task (3.90 vs. 2.68 VLM-Judge), and significantly surpass existing placement baselines and zero-shot Vision-Language Models for object placement. Furthermore, we distill these priors into a lightweight model for fast practical inference (230,000x faster).
>
---
#### [new 208] Dual-Exposure Imaging with Events
- **分类: cs.CV**

- **简介: 该论文属于图像增强任务，旨在解决低光环境下双曝光成像的伪影问题。通过引入事件相机信息，提出E-DEI算法提升图像质量。**

- **链接: [https://arxiv.org/pdf/2604.10273](https://arxiv.org/pdf/2604.10273)**

> **作者:** Mingyuan Lin; Hongyi Liu; Chu He; Wen Yang; Gui-Song Xia; Lei Yu
>
> **摘要:** By combining complementary benefits of short- and long-exposure images, Dual-Exposure Imaging (DEI) enhances image quality in low-light scenarios. However, existing DEI approaches inevitably suffer from producing artifacts due to spatial displacement from scene motion and image feature discrepancies from different exposure times. To tackle this problem, we propose a novel Event-based DEI (E-DEI) algorithm, which reconstructs high-quality images from dual-exposure image pairs and events, leveraging high temporal resolution of event cameras to provide accurate inter-/intra-frame dynamic information. Specifically, we decompose this complex task into an integration of two sub-tasks, i.e., event-based motion deblurring and low-light image enhancement tasks, which guides us to design E-DEI network as a dual-path parallel feature propagation architecture. We propose a Dual-path Feature Alignment and Fusion (DFAF) module to effectively align and fuse features extracted from dual-exposure images with assistance of events. Furthermore, we build a real-world Dataset containing Paired low-/normal-light Images and Events (PIED). Experiments on multiple datasets show the superiority of our method. The code and dataset are available at github.
>
---
#### [new 209] Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression
- **分类: cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决低比特率下结构保真与率失真优化的问题。提出RDVQ框架，通过可微分向量量化实现端到端优化，提升压缩效率和质量。**

- **链接: [https://arxiv.org/pdf/2604.10546](https://arxiv.org/pdf/2604.10546)**

> **作者:** Shiyin Jiang; Wei Long; Minghao Han; Zhenghao Chen; Ce Zhu; Shuhang Gu
>
> **备注:** Accepted for publication at CVPR 2026 as an Oral presentation
>
> **摘要:** The rapid growth of visual data under stringent storage and bandwidth constraints makes extremely low-bitrate image compression increasingly important. While Vector Quantization (VQ) offers strong structural fidelity, existing methods lack a principled mechanism for joint rate-distortion (RD) optimization due to the disconnect between representation learning and entropy modeling. We propose RDVQ, a unified framework that enables end-to-end RD optimization for VQ-based compression via a differentiable relaxation of the codebook distribution, allowing the entropy loss to directly shape the latent prior. We further develop an autoregressive entropy model that supports accurate entropy modeling and test-time rate control. Extensive experiments demonstrate that RDVQ achieves strong performance at extremely low bitrates with a lightweight architecture, attaining competitive or superior perceptual quality with significantly fewer parameters. Compared with RDEIC, RDVQ reduces bitrate by up to 75.71% on DISTS and 37.63% on LPIPS on DIV2K-val. Beyond empirical gains, RDVQ introduces an entropy-constrained formulation of VQ, highlighting the potential for a more unified view of image tokenization and compression. The code will be available at this https URL.
>
---
#### [new 210] FastSHADE: Fast Self-augmented Hierarchical Asymmetric Denoising for Efficient inference on mobile devices
- **分类: cs.CV**

- **简介: 该论文属于图像去噪任务，解决移动设备实时高保真去噪问题。提出FastSHADE网络，采用分层结构和自增强策略，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.10275](https://arxiv.org/pdf/2604.10275)**

> **作者:** Nikolay Falaleev
>
> **摘要:** Real-time image denoising is essential for modern mobile photography but remains challenging due to the strict latency and power constraints of edge devices. This paper presents FastSHADE (Fast Self-augmented Hierarchical Asymmetric Denoising), a lightweight U-Net-style network tailored for real-time, high-fidelity restoration on mobile GPUs. Our method features a multi-stage architecture incorporating a novel Asymmetric Frequency Denoising Block (AFDB) that decouples spatial structure extraction from high-frequency noise suppression to maximize efficiency, and a Spatially Gated Upsampler (SGU) that optimizes high-resolution skip connection fusion. To address generalization, we introduce an efficient Noise Shifting Self-Augmentation strategy that enhances data diversity without inducing domain shifts. Evaluations on the MAI2021 benchmark demonstrate that our scalable model family establishes a highly efficient speed-fidelity trade-off. Our base FastSHADE-M variant maintains real-time latency (<50 ms on a modern mobile GPU) while preserving structural integrity, and our scaled-up FastSHADE-XL establishes a new state-of-the-art for overall image quality. Ultimately, FastSHADE successfully bridges the gap between theoretical network efficiency and practical deployment for real-world mobile ISP pipelines.
>
---
#### [new 211] POINTS-Long: Adaptive Dual-Mode Visual Reasoning in MLLMs
- **分类: cs.CV**

- **简介: 该论文提出POINTS-Long，解决MLLM在长视频场景下的可扩展性问题。通过双模式机制实现视觉token动态调整，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.11627](https://arxiv.org/pdf/2604.11627)**

> **作者:** Haicheng Wang; Yuan Liu; Yikun Liu; Zhemeng Yu; Zhongyin Zhao; Yangxiu You; Zilin Yu; Le Tian; Xiao Zhou; Jie Zhou; Weidi Xie; Yanfeng Wang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently demonstrated remarkable capabilities in cross-modal understanding and generation. However, the rapid growth of visual token sequences--especially in long-video and streaming scenarios--poses a major challenge to their scalability and real-world deployment. Thus, we introduce POINTS-Long, a native dual-mode MLLM featuring dynamic visual token scaling inspired by the human visual system. The model supports two complementary perception modes: focus mode and standby mode, enabling users to dynamically trade off efficiency and accuracy during inference. On fine-grained visual tasks, the focus mode retains the optimal performance, while on long-form general visual understanding, the standby mode retains 97.7-99.7% of the original accuracy using only 1/40-1/10th of the visual tokens. Moreover, POINTS-Long natively supports streaming visual understanding via a dynamically detachable KV-cache design, allowing efficient maintenance of ultra-long visual memory. Our work provides new insights into the design of future MLLMs and lays the foundation for adaptive and efficient long-form visual understanding.
>
---
#### [new 212] A Comparative Study of Modern Object Detectors for Robust Apple Detection in Orchard Imagery
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决果园中苹果检测的困难问题。通过对比多种检测器，评估其在复杂环境下的性能，为实际应用提供选择依据。**

- **链接: [https://arxiv.org/pdf/2604.09996](https://arxiv.org/pdf/2604.09996)**

> **作者:** Mohammed Asad; Ajai Kumar Gautam; Priyanshu Dhiman; Rishi Raj Prajapati
>
> **备注:** Accepted at ICICV 2026; 8 pages, 4 figures
>
> **摘要:** Accurate apple detection in orchard images is important for yield prediction, fruit counting, robotic harvesting, and crop monitoring. However, changing illumination, leaf clutter, dense fruit clusters, and partial occlusion make detection difficult. To provide a fair and reproducible comparison, this study establishes a controlled benchmark for single-class apple detection on the public AppleBBCH81 dataset using one deterministic train, validation, and test split and a unified evaluation protocol across six representative detectors: YOLOv10n, YOLO11n, RT-DETR-L, Faster R-CNN (ResNet50-FPN), FCOS (ResNet50-FPN), and SSDLite320 (MobileNetV3-Large). Performance is evaluated primarily using COCO-style mAP@0.5 and mAP@0.5:0.95, and threshold-dependent behavior is further analyzed using precision-recall curves and fixed-threshold precision, recall, and F1-score at IoU = 0.5. On the validation split, YOLO11n achieves the best strict localization performance with mAP@0.5:0.95 = 0.6065 and mAP@0.5 = 0.9620, followed closely by RT-DETR-L and YOLOv10n. At a fixed operating point with confidence >= 0.05, YOLOv10n attains the highest F1-score, whereas RT-DETR-L achieves very high recall but low precision because of many false positives at low confidence. These findings show that detector selection for orchard deployment should be guided not only by localization-aware accuracy but also by threshold robustness and the requirements of the downstream task.
>
---
#### [new 213] STORM: End-to-End Referring Multi-Object Tracking in Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出STORM，解决视频中多目标指代跟踪（RMOT）问题。通过端到端框架联合处理目标定位与跟踪，提升性能并构建新数据集STORM-Bench。**

- **链接: [https://arxiv.org/pdf/2604.10527](https://arxiv.org/pdf/2604.10527)**

> **作者:** Zijia Lu; Jingru Yi; Jue Wang; Yuxiao Chen; Junwen Chen; Xinyu Li; Davide Modolo
>
> **备注:** CVPR 2026 Findings
>
> **摘要:** Referring multi-object tracking (RMOT) is a task of associating all the objects in a video that semantically match with given textual queries or referring expressions. Existing RMOT approaches decompose object grounding and tracking into separated modules and exhibit limited performance due to the scarcity of training videos, ambiguous annotations, and restricted domains. In this work, we introduce STORM, an end-to-end MLLM that jointly performs grounding and tracking within a unified framework, eliminating external detectors and enabling coherent reasoning over appearance, motion, and language. To improve data efficiency, we propose a task-composition learning (TCL) strategy that decomposes RMOT into image grounding and object tracking, allowing STORM to leverage data-rich sub-tasks and learn structured spatial--temporal reasoning. We further construct STORM-Bench, a new RMOT dataset with accurate trajectories and diverse, unambiguous referring expressions generated through a bottom-up annotation pipeline. Extensive experiments show that STORM achieves state-of-the-art performance on image grounding, single-object tracking, and RMOT benchmarks, demonstrating strong generalization and robust spatial--temporal grounding in complex real-world scenarios. STORM-Bench is released at this https URL.
>
---
#### [new 214] Warm-Started Reinforcement Learning for Iterative 3D/2D Liver Registration
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像配准任务，解决术前CT与术中视频的配准问题。提出一种基于强化学习的框架，实现快速、自动的迭代配准。**

- **链接: [https://arxiv.org/pdf/2604.10245](https://arxiv.org/pdf/2604.10245)**

> **作者:** Hanyuan Zhang; Lucas He; Zijie Cheng; Abdolrahim Kadkhodamohammadi; Danail Stoyanov; Brian R. Davidson; Evangeles B. Mazomenos; Matthew.J Clarkson
>
> **备注:** Laparoscopic Liver Surgery, Augmented Reality, Image Registration, Reinforcement Learning
>
> **摘要:** Registration between preoperative CT and intraoperative laparoscopic video plays a crucial role in augmented reality (AR) guidance for minimally invasive surgery. Learning-based methods have recently achieved registration errors comparable to optimization-based approaches while offering faster inference. However, many supervised methods produce coarse alignments that rely on additional optimization-based refinement, thereby increasing inference time. We present a discrete-action reinforcement learning (RL) framework that formulates CT-to-video registration as a sequential decision-making process. A shared feature encoder, warm-started from a supervised pose estimation network to provide stable geometric features and faster convergence, extracts representations from CT renderings and laparoscopic frames, while an RL policy head learns to choose rigid transformations along six degrees of freedom and to decide when to stop the iteration. Experiments on a public laparoscopic dataset demonstrated that our method achieved an average target registration error (TRE) of 15.70 mm, comparable to supervised approaches with optimization, while achieving faster convergence. The proposed RL-based formulation enables automated, efficient iterative registration without manually tuned step sizes or stopping criteria. This discrete framework provides a practical foundation for future continuous-action and deformable registration models in surgical AR applications.
>
---
#### [new 215] Semantic-Geometric Dual Compression: Training-Free Visual Token Reduction for Ultra-High-Resolution Remote Sensing Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像理解任务，解决UHR影像处理中的计算效率问题。提出DualComp框架，通过语义和几何双流压缩，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.11122](https://arxiv.org/pdf/2604.11122)**

> **作者:** Yueying Li; Fengxiang Wang; Yan Li; Mingshuo Chen; Mengying Zhao; Long Lan
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated immense potential in Earth observation. However, the massive visual tokens generated when processing Ultra-High-Resolution (UHR) imagery introduce prohibitive computational overhead, severely bottlenecking their inference efficiency. Existing visual token compression methods predominantly adopt static and uniform compression strategies, neglecting the inherent "Semantic-Geometric Duality" in remote sensing interpretation tasks. Specifically, object semantic tasks focus on the abstract semantics of objects and benefit from aggressive background pruning, whereas scene geometric tasks critically rely on the integrity of spatial topology. To address this challenge, we propose DualComp, a task-adaptive dual-stream token compression framework. Dynamically guided by a lightweight pre-trained router, DualComp decouples feature processing into two dedicated pathways. In the object semantic stream, the Spatially-Contiguous Semantic Aggregator (SCSA) utilizes size-adaptive clustering to aggregates redundant background while protecting small object. In the scene geometric stream, the Instruction-Guided Structure Recoverer (IGSR) introduces a greedy path-tracing topology completion mechanism to reconstruct spatial skeletons. Experiments on the UHR remote sensing benchmark XLRS-Bench demonstrate that DualComp accomplishes high-fidelity remote sensing interpretation at an exceptionally low computational cost, achieving simultaneous improvements in both efficiency and accuracy.
>
---
#### [new 216] Not Your Stereo-Typical Estimator: Combining Vision and Language for Volume Perception
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; eess.IV**

- **简介: 该论文属于视觉体积估计任务，旨在解决单视角图像模糊性和复杂3D重建的问题。通过融合立体视觉和自然语言文本的先验知识，提升体积估计精度。**

- **链接: [https://arxiv.org/pdf/2604.09886](https://arxiv.org/pdf/2604.09886)**

> **作者:** Gautham Vinod; Bruce Coburn; Siddeshwar Raghavan; Fengqing Zhu
>
> **摘要:** Accurate volume estimation of objects from visual data is a long-standing challenge in computer vision with significant applications in robotics, logistics, and smart health. Existing methods often rely on complex 3D reconstruction pipelines or struggle with the ambiguity inherent in single-view images. To address these limitations, we introduce a new method that fuses implicit 3D cues from stereo vision with explicit prior knowledge from natural language text. Our approach extracts deep features from a stereo image pair and a descriptive text prompt that contains the object's class and an approximate volume, then integrates them using a simple yet effective projection layer into a unified, multi-modal representation for regression. We conduct extensive experiments on public datasets demonstrating that our text-guided approach significantly outperforms vision-only baselines. Our findings show that leveraging even simple textual priors can effectively guide the volume estimation task, paving the way for more context-aware visual measurement systems. Code: this https URL.
>
---
#### [new 217] Omnimodal Dataset Distillation via High-order Proxy Alignment
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于数据蒸馏任务，解决多模态数据压缩问题。提出HoPA方法，通过高阶对齐实现跨模态高效蒸馏。**

- **链接: [https://arxiv.org/pdf/2604.10666](https://arxiv.org/pdf/2604.10666)**

> **作者:** Yuxuan Gao; Xiaohao Liu; Xiaobo Xia; Tongliang Liu
>
> **摘要:** Dataset distillation compresses large-scale datasets into compact synthetic sets while preserving training performance, but existing methods are largely restricted to single-modal or bimodal settings. Extending dataset distillation to scenarios involving more than two modalities, i.e., Omnimodal Dataset Distillation, remains underexplored and challenging due to increased heterogeneity and complex cross-modal interactions. In this work, we identify the key determinant that bounds the endpoint discrepancy in the omnimodal setting, which is exacerbated with an increasing number of modalities. To this end, we propose HoPA, a unified method that captures high-order cross-modal alignments via a compact proxy, which is compatible with trajectory matching as well. By abstracting omnimodal alignment with a shared similarity structure, our method avoids the combinatorial complexity of pairwise modality modeling and enables scalable joint distillation across heterogeneous modalities. Theoretical analysis from the spectral perspective reveals the rationality of our proposed method against bimodal dataset distillation techniques. Extensive experiments on various benchmarks demonstrate that the proposed method achieves superior compression-performance trade-offs compared to existing competitors. The source code will be publicly released.
>
---
#### [new 218] TAPNext++: What's Next for Tracking Any Point (TAP)?
- **分类: cs.CV**

- **简介: 该论文属于视频中任意点跟踪任务，解决长序列跟踪和重检测问题。提出TAPNext++模型，提升跟踪性能并引入新评估指标。**

- **链接: [https://arxiv.org/pdf/2604.10582](https://arxiv.org/pdf/2604.10582)**

> **作者:** Sebastian Jung; Artem Zholus; Martin Sundermeyer; Carl Doersch; Ross Goroshin; David Joseph Tan; Sarath Chandar; Rudolph Triebel; Federico Tombari
>
> **备注:** 8 pages, will be publised at CVPR Findings 2026, Website this https URL
>
> **摘要:** Tracking-Any-Point (TAP) models aim to track any point through a video which is a crucial task in AR/XR and robotics applications. The recently introduced TAPNext approach proposes an end-to-end, recurrent transformer architecture to track points frame-by-frame in a purely online fashion -- demonstrating competitive performance at minimal latency. However, we show that TAPNext struggles with longer video sequences and also frequently fails to re-detect query points that reappear after being occluded or leaving the frame. In this work, we present TAPNext++, a model that tracks points in sequences that are orders of magnitude longer while preserving the low memory and compute footprint of the architecture. We train the recurrent video transformer using several data-driven solutions, including training on long 1024-frame sequences enabled by sequence parallelism techniques. We highlight that re-detection performance is a blind spot in the current literature and introduce a new metric, Re-Detection Average Jaccard ($AJ_{RD}$), to explicitly evaluate tracking on re-appearing points. To improve re-detection of points, we introduce tailored geometric augmentations, such as periodic roll that simulates point re-entries, and supervising occluded points. We demonstrate that recurrent transformers can be substantially improved for point tracking and set a new state-of-the-art on multiple benchmarks. Model and code can be found at this https URL.
>
---
#### [new 219] Demographic and Linguistic Bias Evaluation in Omnimodal Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态语言模型的公平性评估任务，旨在检测和分析模型在不同人口统计和语言群体中的偏差。工作包括对四类模型在多个任务上的性能比较。**

- **链接: [https://arxiv.org/pdf/2604.10014](https://arxiv.org/pdf/2604.10014)**

> **作者:** Alaa Elobaid
>
> **备注:** Accepted at ICPR 2026. Full paper with complete appendix (31 pages total)
>
> **摘要:** This paper provides a comprehensive evaluation of demographic and linguistic biases in omnimodal language models that process text, images, audio, and video within a single framework. Although these models are being widely deployed, their performance across different demographic groups and modalities is not well studied. Four omnimodal models are evaluated on tasks that include demographic attribute estimation, identity verification, activity recognition, multilingual speech transcription, and language identification. Accuracy differences are measured across age, gender, skin tone, language, and country of origin. The results show that image and video understanding tasks generally exhibit better performance with smaller demographic disparities. In contrast, audio understanding tasks exhibit significantly lower performance and substantial bias, including large accuracy differences across age groups, genders, and languages, and frequent prediction collapse toward narrow categories. These findings highlight the importance of evaluating fairness across all supported modalities as omnimodal language models are increasingly used in real-world applications.
>
---
#### [new 220] PointSplat: Efficient Geometry-Driven Pruning and Transformer Refinement for 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决高斯分布数量多导致的存储与效率问题。提出PointSplat框架，通过几何驱动的剪枝和Transformer优化，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.09903](https://arxiv.org/pdf/2604.09903)**

> **作者:** Anh Thuan Tran; Jana Kosecka
>
> **备注:** Accepted to CVPRW 2026 (3DMV)
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently unlocked real-time, high-fidelity novel view synthesis by representing scenes using explicit 3D primitives. However, traditional methods often require millions of Gaussians to capture complex scenes, leading to significant memory and storage demands. Recent approaches have addressed this issue through pruning and per-scene fine-tuning of Gaussian parameters, thereby reducing the model size while maintaining visual quality. These strategies typically rely on 2D images to compute important scores followed by scene-specific optimization. In this work, we introduce PointSplat, 3D geometry-driven prune-and-refine framework that bridges previously disjoint directions of gaussian pruning and transformer refinement. Our method includes two key components: (1) an efficient geometry-driven strategy that ranks Gaussians based solely on their 3D attributes, removing reliance on 2D images during pruning stage, and (2) a dual-branch encoder that separates, re-weights geometric and appearance to avoid feature imbalance. Extensive experiments on ScanNet++ and Replica across varying sparsity levels demonstrate that PointSplat consistently achieves competitive rendering quality and superior efficiency without additional per-scene optimization.
>
---
#### [new 221] Anatomy-Informed Deep Learning for Abdominal Aortic Aneurysm Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决腹主动脉瘤分割中的误检和边界不准确问题。通过引入解剖先验的排除掩码，提升模型的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10312](https://arxiv.org/pdf/2604.10312)**

> **作者:** Osamah Sufyan; Martin Brückmann; Ralph Wickenhöfer; Babette Dellen; Uwe Jaekel
>
> **备注:** International Conference on Computational Science
>
> **摘要:** In CT angiography, the accurate segmentation of abdominal aortic aneurysms (AAAs) is difficult due to large anatomical variability, low-contrast vessel boundaries, and the close proximity of organs whose intensities resemble vascular structures, often leading to false positives. To address these challenges, we propose an anatomy-aware segmentation framework that integrates organ exclusion masks derived from TotalSegmentator into the training process. These masks encode explicit anatomical priors by identifying non-vascular organsand penalizing aneurysm predictions within these regions, thereby guiding the U-Net to focus on the aorta and its pathological dilation while suppressing anatomically implausible predictions. Despite being trained on a relatively small dataset, the anatomy-aware model achieves high accuracy, substantially reduces false positives, and improves boundary consistency compared to a standard U-Net baseline. The results demonstrate that incorporating anatomical knowledge through exclusion masks provides an efficient mechanism to enhance robustness and generalization, enabling reliable AAA segmentation even with limited training data.
>
---
#### [new 222] PhyMix: Towards Physically Consistent Single-Image 3D Indoor Scene Generation with Implicit--Explicit Optimization
- **分类: cs.CV**

- **简介: 该论文属于3D室内场景生成任务，旨在解决现有方法生成的场景缺乏物理一致性的问题。通过引入物理评估器和优化框架，提升生成场景的物理合理性。**

- **链接: [https://arxiv.org/pdf/2604.10125](https://arxiv.org/pdf/2604.10125)**

> **作者:** Dongli Wu; Jingyu Hu; Ka-Hei Hui; Xiaobao Wei; Chengwen Luo; Jianqiang Li; Zhengzhe Liu
>
> **摘要:** Existing single-image 3D indoor scene generators often produce results that look visually plausible but fail to obey real-world physics, limiting their reliability in robotics, embodied AI, and design. To examine this gap, we introduce a unified Physics Evaluator that measures four main aspects: geometric priors, contact, stability, and deployability, which are further decomposed into nine sub-constraints, establishing the first benchmark to measure physical consistency. Based on this evaluator, our analysis shows that state-of-the-art methods remain largely physics-unaware. To overcome this limitation, we further propose a framework that integrates feedback from the Physics Evaluator into both training and inference, enhancing the physical plausibility of generated scenes. Specifically, we propose PhyMix, which is composed of two complementary components: (i) implicit alignment via Scene-GRPO, a critic-free group-relative policy optimization that leverages the Physics Evaluator as a preference signal and biases sampling towards physically feasible layouts, and (ii) explicit refinement via a plug-and-play Test-Time Optimizer (TTO) that uses differentiable evaluator signals to correct residual violations during generation. Overall, our method unifies evaluation, reward shaping, and inference-time correction, producing 3D indoor scenes that are visually faithful and physically plausible. Extensive synthetic evaluations confirm state-of-the-art performance in both visual fidelity and physical plausibility, and extensive qualitative examples in stylized and real-world images further showcase the robustness of the method. We will release codes and models upon publication.
>
---
#### [new 223] Retinal Cyst Detection from Optical Coherence Tomography Images
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **简介: 该论文属于医学图像分割任务，旨在准确检测视网膜囊肿。针对现有方法精度低、受图像质量影响大的问题，采用ResNet进行分块分类，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2604.10843](https://arxiv.org/pdf/2604.10843)**

> **作者:** Abhishek Dharmaratnakar; Aadheeshwar Vijayakumar; Suchand Dayanand
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Retinal Cysts are formed by leakage and accumulation of fluid in the retina due to the incompetence of retinal vasculature. These cystic spaces have significance in several ocular diseases such as age-related macular degeneration, diabetic macular edema, etc. Optical coherence tomography is one of the predominant diagnosing techniques for imaging retinal pathologies. Segmenting and quantification of intraretinal cysts plays the vital role in predicting visual acuity. In literature, several methods have been proposed for automatic segmentation of intraretinal cysts. As cystoid macular edema becomes a major problem to humankind, we need to quantify it accurately and operate it out, else it might cause many problems later on. Though research is being carried out in this area, not much of progress has been made and accuracy achieved so far is 68\% which is very less. Also, the methods depend on the quality of the image and give very low results for high noise images like topcon. This work uses ResNet CNN (Convolutional Neural Network) approach of segmentation by the way of patchwise classification for training on image set from cyst segmentation challenge dataset and testing on test data set given by 2 different graders for all 4 vendors in the challenge. It also compares these methods using first publicly available novel cyst segmentation challenge dataset. The methods were evaluated using quantitative measures to assess their robustness against the challenges of intraretinal cyst segmentation. The results are found to be better than the previous state of the art approaches giving more than 70\% dice coefficient on all vendors irrespective of their quality.
>
---
#### [new 224] See Fair, Speak Truth: Equitable Attention Improves Grounding and Reduces Hallucination in Vision-Language Alignment
- **分类: cs.CV**

- **简介: 该论文属于视觉语言对齐任务，旨在解决多模态大模型中的对象幻觉问题。通过引入DOP-OBC策略，实现更公平的注意力分配，提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2604.09749](https://arxiv.org/pdf/2604.09749)**

> **作者:** Mohammad Anas Azeez; Ankan Deria; Zohaib Hasan Siddiqui; Adinath Madhavrao Dukre; Rafiq Ali; Sara Atito; Yutong Xie; Imran Razzak
>
> **摘要:** Multimodal large language models (MLLMs) frequently hallucinate objects that are absent from the visual input, often because attention during decoding is disproportionately drawn to visually dominant or frequently occurring content. We observe that this inequity in attention allocation is a root cause of object hallucination: when rare, small, or contextually peripheral objects receive insufficient attention, the model fails to ground its generation in the full visual scene. We argue that every object in an image, regardless of its size, frequency or visual salience, deserves equal representational opportunity during decoding. To this end, we propose DOP-OBC, a training-free and architecture-agnostic decoding strategy built on the principle of equitable attention. Two complementary object-aware signals work in tandem: a Dominant Object Penalty (DOP) that softly suppresses attention over-concentration on visually dominant regions, and an Outlier Boost Coefficient (OBC) that amplifies attention toward rare yet confidently detected objects. These signals are injected as per-row logit modulations within the causal attention mask, requiring no weight updates and preserving autoregressive decoding properties. Extensive experiments across image and video MLLMs demonstrate consistent reductions in object hallucination on CHAIR and POPE benchmarks, alongside improvements in GPT-4o assessed captioning quality across correctness, consistency, detail, context and temporal dimensions. DOP-OBC establishes that fairness in attention allocation is not merely a design principle but a practical and effective path toward more faithful multimodal generation.
>
---
#### [new 225] IMPACT: A Dataset for Multi-Granularity Human Procedural Action Understanding in Industrial Assembly
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出IMPACT数据集，用于工业装配中的多粒度人类操作理解，解决真实场景下动作识别与异常检测问题，包含多视角RGB-D数据、精细标注及异常分类。**

- **链接: [https://arxiv.org/pdf/2604.10409](https://arxiv.org/pdf/2604.10409)**

> **作者:** Di Wen; Zeyun Zhong; David Schneider; Manuel Zaremski; Linus Kunzmann; Yitian Shi; Ruiping Liu; Yufan Chen; Junwei Zheng; Jiahang Li; Jonas Hemmerich; Qiyi Tong; Patric Grauberger; Arash Ajoudani; Danda Pani Paudel; Sven Matthiesen; Barbara Deml; Jürgen Beyerer; Luc Van Gool; Rainer Stiefelhagen; Kunyu Peng
>
> **备注:** 9 pages, 2 figures, benchmark and dataset are available at this https URL
>
> **摘要:** We introduce IMPACT, a synchronized five-view RGB-D dataset for deployment-oriented industrial procedural understanding, built around real assembly and disassembly of a commercial angle grinder with professional-grade tools. To our knowledge, IMPACT is the first real industrial assembly benchmark that jointly provides synchronized ego-exo RGB-D capture, decoupled bimanual annotation, compliance-aware state tracking, and explicit anomaly--recovery supervision within a single real industrial workflow. It comprises 112 trials from 13 participants totaling 39.5 hours, with multi-route execution governed by a partial-order prerequisite graph, a six-category anomaly taxonomy, and operator cognitive load measured via NASA-TLX. The annotation hierarchy links hand-specific atomic actions to coarse procedural steps, component assembly states, and per-hand compliance phases, with synchronized null spans across views to decouple perceptual limitations from algorithmic failure. Systematic baselines reveal fundamental limitations that remain invisible to single-task benchmarks, particularly under realistic deployment conditions that involve incomplete observations, flexible execution paths, and corrective behavior. The full dataset, annotations, and evaluation code are available at this https URL.
>
---
#### [new 226] YUV20K: A Complexity-Driven Benchmark and Trajectory-Aware Alignment Model for Video Camouflaged Object Detection
- **分类: cs.CV; cs.DB**

- **简介: 该论文聚焦视频伪装目标检测任务，解决数据稀缺与模型鲁棒性不足的问题。提出YUV20K基准和轨迹感知对齐模型，提升复杂运动场景下的检测性能。**

- **链接: [https://arxiv.org/pdf/2604.09985](https://arxiv.org/pdf/2604.09985)**

> **作者:** Yiyu Liu; Shuo Ye; Chao Hao; Zitong Yu
>
> **摘要:** Video Camouflaged Object Detection (VCOD) is currently constrained by the scarcity of challenging benchmarks and the limited robustness of models against erratic motion dynamics. Existing methods often struggle with Motion-Induced Appearance Instability and Temporal Feature Misalignment caused by complex motion scenarios. To address the data bottleneck, we present YUV20K, a pixel-level annoated complexity-driven VCOD benchmark. Comprising 24,295 annotated frames across 91 scenes and 47 kinds of species, it specifically targets challenging scenarios like large-displacement motion, camera motion and other 4 types scenarios. On the methodological front, we propose a novel framework featuring two key modules: Motion Feature Stabilization (MFS) and Trajectory-Aware Alignment (TAA). The MFS module utilizes frame-agnostic Semantic Basis Primitives to stablize features, while the TAA module leverages trajectory-guided deformable sampling to ensure precise temporal alignment. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art competitors on existing datasets and establishes a new baseline on the challenging YUV20K. Notably, our framework exhibits superior cross-domain generalization and robustness when confronting complex spatiotemporal scenarios. Our code and dataset will be available at this https URL
>
---
#### [new 227] Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，针对真实退化复杂的问题，提出一种融合退化信息和结构保持的扩散框架，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.11470](https://arxiv.org/pdf/2604.11470)**

> **作者:** Yang Ji; Zonghao Chen; Zhihao Xue; Junqin Hu
>
> **摘要:** Real-world image super-resolution is particularly challenging for diffusion models because real degradations are complex, heterogeneous, and rarely modeled explicitly. We propose a degradation-aware and structure-preserving diffusion framework for real-world SR. Specifically, we introduce Degradation-aware Token Injection, which encodes lightweight degradation statistics from low-resolution inputs and fuses them with semantic conditioning features, enabling explicit degradation-aware restoration. We further propose Spatially Asymmetric Noise Injection, which modulates diffusion noise with local edge strength to better preserve structural regions during training. Both modules are lightweight add-ons to the adopted diffusion SR framework, requiring only minor modifications to the conditioning pipeline. Experiments on DIV2K and RealSR show that our method delivers competitive no-reference perceptual quality and visually more realistic restoration results than recent baselines, while maintaining a favorable perception--distortion trade-off. Ablations confirm the effectiveness of each module and their complementary gains when combined. The code and model are publicly available at this https URL.
>
---
#### [new 228] Multi-Head Attention based interaction-aware architecture for Bangla Handwritten Character Recognition: Introducing a Primary Dataset
- **分类: cs.CV**

- **简介: 该论文属于手写Bangla字符识别任务，旨在解决数据不平衡和字符相似度高的问题。构建了一个平衡数据集，并提出一种融合多注意力机制的深度学习模型，提升了识别准确率。**

- **链接: [https://arxiv.org/pdf/2604.09717](https://arxiv.org/pdf/2604.09717)**

> **作者:** Mirza Raquib; Asif Pervez Polok; Kedar Nath Biswas; Farida Siddiqi Prity; Saydul Akbar Murad; Nick Rahimi
>
> **摘要:** Character recognition is the fundamental part of an optical character recognition (OCR) system. Word recognition, sentence transcription, document digitization, and language processing are some of the higher-order activities that can be done accurately through character recognition. Nonetheless, recognizing handwritten Bangla characters is not an easy task because they are written in different styles with inconsistent stroke patterns and a high degree of visual character resemblance. The datasets available are usually limited in intra-class and inequitable in class distribution. We have constructed a new balanced dataset of Bangla written characters to overcome those problems. This consists of 78 classes and each class has approximately 650 samples. It contains the basic characters, composite (Juktobarno) characters and numerals. The samples were a diverse group comprising a large age range and socioeconomic groups. Elementary and high school students, university students, and professionals are the contributing factors. The sample also has right and left-handed writers. We have further proposed an interaction-aware hybrid deep learning architecture that integrates EfficientNetB3, Vision Transformer, and Conformer modules in parallel. A multi-head cross-attention fusion mechanism enables effective feature interaction across these components. The proposed model achieves 98.84% accuracy on the constructed dataset and 96.49% on the external CHBCR benchmark, demonstrating strong generalization capability. Grad-CAM visualizations further provide interpretability by highlighting discriminative regions. The dataset and source code of this research is publicly available at: this https URL.
>
---
#### [new 229] DINO_4D: Semantic-Aware 4D Reconstruction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于4D重建任务，解决动态场景中语义漂移问题。通过引入语义先验，提升重建精度与完整性。**

- **链接: [https://arxiv.org/pdf/2604.09877](https://arxiv.org/pdf/2604.09877)**

> **作者:** Yiru Yang; Zhuojie Wu; Quentin Marguet; Nishant Kumar Singh; Max Schulthess
>
> **摘要:** In the intersection of computer vision and robotic perception, 4D reconstruction of dynamic scenes serve as the critical bridge connecting low-level geometric sensing with high-level semantic understanding. We present DINO\_4D, introducing frozen DINOv3 features as structural priors, injecting semantic awareness into the reconstruction process to effectively suppress semantic drift during dynamic tracking. Experiments on the Point Odyssey and TUM-Dynamics benchmarks demonstrate that our method maintains the linear time complexity $O(T)$ of its predecessors while significantly improving Tracking Accuracy (APD) and Reconstruction Completeness. DINO\_4D establishes a new paradigm for constructing 4D World Models that possess both geometric precision and semantic understanding.
>
---
#### [new 230] The Devil is in the Details -- From OCR for Old Church Slavonic to Purely Visual Stemma Reconstruction
- **分类: cs.CV**

- **简介: 论文探讨OCR技术在古教会斯拉夫语文本处理中的应用，并提出纯视觉的谱系重建方法，解决文本识别与文献谱系分析问题。**

- **链接: [https://arxiv.org/pdf/2604.11724](https://arxiv.org/pdf/2604.11724)**

> **作者:** Armin Hoenen
>
> **备注:** International conference at Valamo monastery, Finnland, 2026
>
> **摘要:** The age of artificial intelligence has brought many new possibilities and pitfalls in many fields and tasks. The devil is in the details, and those come to the fore when building new pipelines and executing small practical experiments. OCR and stemmatology are no exception. The current investigation starts comparing a range of OCR-systems, from classical over machine learning to LLMs, for roughly 6,000 characters of late handwritten church slavonic manuscripts from the 18th century. Focussing on basic letter correctness, more than 10 CS OCR-systems among which 2 LLMs (GPT5 and Gemini3-flash) are being compared. Then, post-processing via LLMs is assessed and finally, different agentic OCR architectures (specialized post-processing agents, an agentic pipeline and RAG) are tested. With new technology elaborated, experiments suggest, church slavonic CER for basic letters may reach as low as 2-3% but elaborated diacritics could still present a problem. How well OCR can prime stemmatology as a downstream task is the entry point to the second part of the article which introduces a new stemmatic method based solely on image processing. Here, a pipeline of automated visual glyph extraction, clustering and pairwise statistical comparison leading to a distance matrix and ultimately a stemma, is being presented and applied to two small corpora, one for the church slavonic Gospel of Mark from the 14th to 16th centuries, one for the Roman de la Rose in French from the 14th and 15th centuries. Basic functioning of the method can be demonstrated.
>
---
#### [new 231] WBCBench 2026: A Challenge for Robust White Blood Cell Classification Under Class Imbalance
- **分类: cs.CV**

- **简介: 该论文属于白细胞分类任务，旨在解决类别不平衡和域移位问题。通过构建WBCBench 2026基准，提供不同难度的图像数据集和评估标准，以测试算法的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10797](https://arxiv.org/pdf/2604.10797)**

> **作者:** Xin Tian; Xudong Ma; Tianqi Yang; Alin Achim; Bartłomiej W Papież; Phandee Watanaboonyongcharoen; Nantheera Anantrasirichai
>
> **备注:** IEEE International Symposium on Biomedical Imaging (ISBI)
>
> **摘要:** We present WBCBench 2026, an ISBI challenge and benchmark for automated WBC classification designed to stress-test algorithms under three key difficulties: (i) severe class imbalance across 13 morphologically fine-grained WBC classes, (ii) strict patient-level separation between training, validation and test sets, and (iii) synthetic scanner- and setting-induced domain shift via controlled noise, blur and illumination perturbations. All images are single-site microscopic blood smear acquisitions with standardised staining and expert hematopathologist annotations. This paper reviews the challenge and summarises the proposed solutions and final outcomes. The benchmark is organised into two phases. Phase 1 provides a pristine training set. Phase 2 introduces degraded images with split-specific severity distributions for train, validation and test, emulating a realistic shift between development and deployment conditions. We specify a standardised submission schema, open-source evaluator, and macro-averaged F1 score as the primary ranking metric.
>
---
#### [new 232] Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决多事件视频中时间顺序控制和语义干扰问题。提出Prompt Relay方法，在推理阶段实现精细的时间控制，提升视频与文本的一致性。**

- **链接: [https://arxiv.org/pdf/2604.10030](https://arxiv.org/pdf/2604.10030)**

> **作者:** Gordon Chen; Ziqi Huang; Ziwei Liu
>
> **摘要:** Video diffusion models have achieved remarkable progress in generating high-quality videos. However, these models struggle to represent the temporal succession of multiple events in real-world videos and lack explicit mechanisms to control when semantic concepts appear, how long they persist, and the order in which multiple events occur. Such control is especially important for movie-grade video synthesis, where coherent storytelling depends on precise timing, duration, and transitions between events. When using a single paragraph-style prompt to describe a sequence of complex events, models often exhibit semantic entanglement, where concepts intended for different moments in the video bleed into one another, resulting in poor text-video alignment. To address these limitations, we propose Prompt Relay, an inference-time, plug-and-play method to enable fine-grained temporal control in multi-event video generation, requiring no architectural modifications and no additional computational overhead. Prompt Relay introduces a penalty into the cross-attention mechanism, so that each temporal segment attends only to its assigned prompt, allowing the model to represent one semantic concept at a time and thereby improving temporal prompt alignment, reducing semantic interference, and enhancing visual quality.
>
---
#### [new 233] Unmixing-Guided Spatial-Spectral Mamba with Clustering Tokens for Hyperspectral Image Classification
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像分类任务，旨在解决光谱混合效应和细节丢失问题。通过设计解混引导的Mamba模块和聚类令牌策略，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.09948](https://arxiv.org/pdf/2604.09948)**

> **作者:** Yimin Zhu; Lincoln Linlin Xu
>
> **摘要:** Although hyperspectral image (HSI) classification is critical for supporting various environmental applications, it is a challenging task due to the spectral-mixture effect, the spatial-spectral heterogeneity and the difficulty to preserve class boundaries and details. This letter presents a novel unmixing-guided spatial-spectral Mamba with clustering tokens for improved HSI classification, with the following contributions. First, to disentangle the spectral mixture effect in HSI for improved pattern discovery, we design a novel spectral unmixing network that not only automatically learns endmembers and abundance maps from HSI but also accounts for endmember variabilities. Second, to generate Mamba token sequences, based on the clusters defined by abundance maps, we design an efficient Top-\textit{K} token selection strategy to adaptively sequence the tokens for improved representational capability. Third, to improve spatial-spectral feature learning and detail preservation, based on the Top-\textit{K} token sequences, we design a novel unmixing-guided spatial-spectral Mamba module that greatly improves traditional Mamba models in terms of token learning and sequencing. Fourth, to learn simultaneously the endmember-abundance patterns and classification labels, a multi-task scheme is designed for model supervision, leading to a new unmixing-classification framework that outputs not only accurate classification maps but also a comprehensive spectral-library and abundance maps. Comparative experiments on four HSI datasets demonstrate that our model can greatly outperform the other state-of-the-art approaches. Code is available at this https URL
>
---
#### [new 234] Real-Time Human Reconstruction and Animation using Feed-Forward Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于人体3D重建与实时动画任务，解决传统方法依赖深度监督或多次推理的问题。通过单次前向传播生成可动画化的3D人体模型，提升效率与实时性。**

- **链接: [https://arxiv.org/pdf/2604.10259](https://arxiv.org/pdf/2604.10259)**

> **作者:** Devdoot Chatterjee; Zakaria Laskar; C.V. Jawahar
>
> **摘要:** We present a generalizable feed-forward Gaussian splatting framework for human 3D reconstruction and real-time animation that operates directly on multi-view RGB images and their associated SMPL-X poses. Unlike prior methods that rely on depth supervision, fixed input views, UV map, or repeated feed-forward inference for each target view or pose, our approach predicts, in a canonical pose, a set of 3D Gaussian primitives associated with each SMPL-X vertex. One Gaussian is regularized to remain close to the SMPL-X surface, providing a strong geometric prior and stable correspondence to the parametric body model, while an additional small set of unconstrained Gaussians per vertex allows the representation to capture geometric structures that deviate from the parametric surface, such as clothing and hair. In contrast to recent approaches such as HumanRAM, which require repeated network inference to synthesize novel poses, our method produces an animatable human representation from a single forward pass; by explicitly associating Gaussian primitives with SMPL-X vertices, the reconstructed model can be efficiently animated via linear blend skinning without further network evaluation. We evaluate our method on the THuman 2.1, AvatarReX and THuman 4.0 datasets, where it achieves reconstruction quality comparable to state-of-the-art methods while uniquely supporting real-time animation and interactive applications. Code and pre-trained models are available at this https URL .
>
---
#### [new 235] Reasoning Resides in Layers: Restoring Temporal Reasoning in Video-Language Models with Layer-Selective Merging
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频语言模型任务，旨在解决视觉对齐导致的时间推理能力下降问题。提出MERIT框架，通过分层融合恢复时间推理，无需重新训练。**

- **链接: [https://arxiv.org/pdf/2604.11399](https://arxiv.org/pdf/2604.11399)**

> **作者:** Zihang Fu; Haonan Wang; Jian Kang; Kenji Kawaguchi; Jiaying Wu
>
> **摘要:** Multimodal adaptation equips large language models (LLMs) with perceptual capabilities, but often weakens the reasoning ability inherited from language-only pretraining. This trade-off is especially pronounced in video-language models (VLMs), where visual alignment can impair temporal reasoning (TR) over sequential events. We propose MERIT, a training-free, task-driven model merging framework for restoring TR in VLMs. MERIT searches over layer-wise self-attention merging recipes between a VLM and its paired text-only backbone using an objective that improves TR while penalizing degradation in temporal perception (TP). Across three representative VLMs and multiple challenging video benchmarks, MERIT consistently improves TR, preserves or improves TP, and generalizes beyond the search set to four distinct benchmarks. It also outperforms uniform full-model merging and random layer selection, showing that effective recovery depends on selecting the right layers. Interventional masking and frame-level attribution further show that the selected layers are disproportionately important for reasoning and shift model decisions toward temporally and causally relevant evidence. These results show that targeted, perception-aware model merging can effectively restore TR in VLMs without retraining.
>
---
#### [new 236] Visual Late Chunking: An Empirical Study of Contextual Chunking for Efficient Visual Document Retrieval
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于视觉文档检索任务，解决多向量模型存储与计算成本高的问题。提出ColChunk框架，通过多模态晚期分块实现高效上下文表示。**

- **链接: [https://arxiv.org/pdf/2604.10167](https://arxiv.org/pdf/2604.10167)**

> **作者:** Yibo Yan; Mingdong Ou; Yi Cao; Jiahao Huo; Xin Zou; Shuliang Liu; James Kwok; Xuming Hu
>
> **备注:** Preprint
>
> **摘要:** Multi-vector models dominate Visual Document Retrieval (VDR) due to their fine-grained matching capabilities, but their high storage and computational costs present a major barrier to practical deployment. In this paper, we propose ColChunk, a plug-and-play framework that introduces multimodal late chunking to construct efficient, contextualized multi-vectors. Unlike existing pruning or fixed-token approaches, ColChunk employs hierarchical clustering on patch-level embeddings, fused with a 2D position prior to ensure spatial-semantic coherence. This adaptive grouping allows for a content-aware representation that preserves global context while drastically reducing the vector count. Evaluations across 24 VDR datasets demonstrate ColChunk achieves over a 90% reduction in storage requirements while simultaneously delivering a 9-point average improvement in nDCG@5 across representative single-vector models. ColChunk provides a practical solution for balancing retrieval accuracy and efficiency in visual document systems.
>
---
#### [new 237] ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型量化任务，解决激活异常值问题。提出ReSpinQuant框架，通过子空间残差旋转实现高效层间量化，兼顾精度与推理效率。**

- **链接: [https://arxiv.org/pdf/2604.11080](https://arxiv.org/pdf/2604.11080)**

> **作者:** Suyoung Kim; Sunghyun Wee; Hyeonjin Kim; Kyomin Hwang; Hyunho Lee; Nojun Kwak
>
> **摘要:** Rotation-based Post-Training Quantization (PTQ) has emerged as a promising solution for mitigating activation outliers in the quantization of Large Language Models (LLMs). Global rotation methods achieve inference efficiency by fusing activation rotations into attention and FFN blocks, but suffer from limited expressivity as they are constrained to use a single learnable rotation matrix across all layers. To tackle this, layer-wise transformation methods emerged, achieving superior accuracy through localized adaptation. However, layer-wise methods cannot fuse activation rotation matrices into weights, requiring online computations and causing significant overhead. In this paper, we propose ReSpinQuant, a quantization framework that resolves such overhead by leveraging offline activation rotation fusion and matching basis using efficient residual subspace rotation. This design reconciles the high expressivity of layer-wise adaptation with only negligible inference overhead. Extensive experiments on W4A4 and W3A3 quantization demonstrate that ReSpinQuant achieves state-of-the-art performance, outperforming global rotation methods and matching the accuracy of computationally expensive layer-wise methods with minimal overhead.
>
---
#### [new 238] Counting to Four is still a Chore for VLMs
- **分类: cs.CV**

- **简介: 该论文研究视觉-语言模型在计数任务中的表现，揭示其失败原因并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2604.10039](https://arxiv.org/pdf/2604.10039)**

> **作者:** Duy Le Dinh Anh; Patrick Amadeus Irawan; Tuan Van Vo
>
> **摘要:** Vision--language models (VLMs) have achieved impressive performance on complex multimodal reasoning tasks, yet they still fail on simple grounding skills such as object counting. Existing evaluations mostly assess only final outputs, offering limited insight into where these failures arise inside the model. In this work, we present an empirical study of VLM counting behavior through both behavioral and mechanistic analysis. We introduce COUNTINGTRICKS, a controlled evaluation suite of simple shape-based counting cases designed to expose vulnerabilities under different patchification layouts and adversarial prompting conditions. Using attention analysis and component-wise probing, we show that count-relevant visual evidence is strongest in the modality projection stage but degrades substantially in later language layers, where models become more susceptible to text priors. Motivated by this finding, we further evaluate Modality Attention Share (MAS), a lightweight intervention that encourages a minimum budget of visual attention during answer generation. Our results suggest that counting failures in VLMs stem not only from visual perception limits, but also from the underuse of visual evidence during language-stage reasoning. Code and dataset will be released at this https URL.
>
---
#### [new 239] Sign Language Recognition in the Age of LLMs
- **分类: cs.CV; cs.CL**

- **简介: 论文研究了在零样本条件下，视觉语言模型（VLMs）是否能有效识别手语。任务属于手语识别，旨在评估VLMs在无需特定训练下的表现。工作包括实验对比不同VLMs在WLASL300数据集上的效果。**

- **链接: [https://arxiv.org/pdf/2604.11225](https://arxiv.org/pdf/2604.11225)**

> **作者:** Vaclav Javorek; Jakub Honzik; Ivan Gruber; Tomas Zelezny; Marek Hruz
>
> **备注:** Accepted at the CVPR 2026 Workshop on Multimodal Sign Language Research (MSLR), 8 pages, 3 figures
>
> **摘要:** Recent Vision Language Models (VLMs) have demonstrated strong performance across a wide range of multimodal reasoning tasks. This raises the question of whether such general-purpose models can also address specialized visual recognition problems such as isolated sign language recognition (ISLR) without task-specific training. In this work, we investigate the capability of modern VLMs to perform ISLR in a zero-shot setting. We evaluate several open-source and proprietary VLMs on the WLASL300 benchmark. Our experiments show that, under prompt-only zero-shot inference, current open-source VLMs remain far behind classic supervised ISLR classifiers by a wide margin. However, follow-up experiments reveal that these models capture partial visual-semantic alignment between signs and text descriptions. Larger proprietary models achieve substantially higher accuracy, highlighting the importance of model scale and training data diversity. All our code is publicly available on GitHub.
>
---
#### [new 240] Any 3D Scene is Worth 1K Tokens: 3D-Grounded Representation for Scene Generation at Scale
- **分类: cs.CV; cs.CG**

- **简介: 该论文属于3D场景生成任务，旨在解决2D方法在3D表示和空间一致性上的局限。提出3DRAE和3DDiT，在3D潜空间中直接生成场景，提升效率与一致性。**

- **链接: [https://arxiv.org/pdf/2604.11331](https://arxiv.org/pdf/2604.11331)**

> **作者:** Dongxu Wei; Qi Xu; Zhiqi Li; Hangning Zhou; Cong Qiu; Hailong Qin; Mu Yang; Zhaopeng Cui; Peidong Liu
>
> **备注:** Under Review. Project Page: this https URL
>
> **摘要:** 3D scene generation has long been dominated by 2D multi-view or video diffusion models. This is due not only to the lack of scene-level 3D latent representation, but also to the fact that most scene-level 3D visual data exists in the form of multi-view images or videos, which are naturally compatible with 2D diffusion architectures. Typically, these 2D-based approaches degrade 3D spatial extrapolation to 2D temporal extension, which introduces two fundamental issues: (i) representing 3D scenes via 2D views leads to significant representation redundancy, and (ii) latent space rooted in 2D inherently limits the spatial consistency of the generated 3D scenes. In this paper, we propose, for the first time, to perform 3D scene generation directly within an implicit 3D latent space to address these limitations. First, we repurpose frozen 2D representation encoders to construct our 3D Representation Autoencoder (3DRAE), which grounds view-coupled 2D semantic representations into a view-decoupled 3D latent representation. This enables representing 3D scenes observed from arbitrary numbers of views--at any resolution and aspect ratio--with fixed complexity and rich semantics. Then we introduce 3D Diffusion Transformer (3DDiT), which performs diffusion modeling in this 3D latent space, achieving remarkably efficient and spatially consistent 3D scene generation while supporting diverse conditioning configurations. Moreover, since our approach directly generates a 3D scene representation, it can be decoded to images and optional point maps along arbitrary camera trajectories without requiring per-trajectory diffusion sampling pass, which is common in 2D-based approaches.
>
---
#### [new 241] RESP: Reference-guided Sequential Prompting for Visual Glitch Detection in Video Games
- **分类: cs.CV**

- **简介: 该论文属于视频游戏视觉故障检测任务，解决传统方法在视频级检测中的不足。提出RESP框架，通过参考帧引导提示提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.11082](https://arxiv.org/pdf/2604.11082)**

> **作者:** Yakun Yu; Ashley Wiens; Adrián Barahona-Ríos; Benedict Wilkins; Saman Zadtootaghaj; Nabajeet Barman; Cor-Paul Bezemer
>
> **摘要:** Visual glitches in video games degrade player experience and perceived quality, yet manual quality assurance cannot scale to the growing test surface of modern game development. Prior automation efforts, particularly those using vision-language models (VLMs), largely operate on single frames or rely on limited video-level baselines that struggle under realistic scene variation, making robust video-level glitch detection challenging. We present RESP, a practical multi-frame framework for gameplay glitch detection with VLMs. Our key idea is reference-guided prompting: for each test frame, we select a reference frame from earlier in the same video, establishing a visual baseline and reframing detection as within-video comparison rather than isolated classification. RESP sequentially prompts the VLM with reference/test pairs and aggregates noisy frame predictions into a stable video-level decision without fine-tuning the VLM. To enable controlled analysis of reference effects, we introduce RefGlitch, a synthetic dataset of manually labeled reference/test frame pairs with balanced coverage across five glitch types. Experiments across five VLMs and three datasets (one synthetic, two real-world) show that reference guidance consistently strengthens frame-level detection and that the improved frame-level evidence reliably transfers to stronger video-level triage under realistic QA conditions. Code and data are available at: \href{this https URL}{this https URL}.
>
---
#### [new 242] RADA: Region-Aware Dual-encoder Auxiliary learning for Barely-supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决少量标注数据下的分割问题。提出RADA框架，结合视觉与语义信息，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.11164](https://arxiv.org/pdf/2604.11164)**

> **作者:** Shuang Zeng; Boxu Xie; Lei Zhu; Xinliang Zhang; Jiakui Hu; Zhengjian Yao; Yuanwei Li; Yuxing Lu; Yanye Lu
>
> **摘要:** Deep learning has greatly advanced medical image segmentation, but its success relies heavily on fully supervised learning, which requires dense annotations that are costly and time-consuming for 3D volumetric scans. Barely-supervised learning reduces annotation burden by using only a few labeled slices per volume. Existing methods typically propagate sparse annotations to unlabeled slices through geometric continuity to generate pseudo-labels, but this strategy lacks semantic understanding, often resulting in low-quality pseudo-labels. Furthermore, medical image segmentation is inherently a pixel-level visual understanding task, where accuracy fundamentally depends on the quality of local, fine-grained visual features. Inspired by this, we propose RADA, a novel Region-Aware Dual-encoder Auxiliary learning pipeline which introduces a dual-encoder framework pre-trained on Alpha-CLIP to extract fine-grained, region-specific visual features from the original images and limited annotations. The framework combines image-level fine-grained visual features with text-level semantic guidance, providing region-aware semantic supervision that bridges image-level semantics and pixel-level segmentation. Integrated into a triple-view training framework, RADA achieves SOTA performance under extremely sparse annotation settings on LA2018, KiTS19 and LiTS, demonstrating robust generalization across diverse datasets.
>
---
#### [new 243] LoViF 2026 Challenge on Human-oriented Semantic Image Quality Assessment: Methods and Results
- **分类: cs.CV**

- **简介: 该论文介绍LoViF 2026挑战，聚焦于从人类视角评估图像语义质量。旨在建立新的基准，解决语义信息损失评价问题，构建SeIQA数据集并评估参赛方案性能。**

- **链接: [https://arxiv.org/pdf/2604.11207](https://arxiv.org/pdf/2604.11207)**

> **作者:** Xin Li; Daoli Xu; Wei Luo; Guoqiang Xiang; Haoran Li; Chengyu Zhuang; Zhibo Chen; Jian Guan; Weping Li; Weixia Zhang; Wei Sun; Zhihua Wang; Dandan Zhu; Chengguang Zhu; Ayush Gupta; Rachit Agarwal; Shouvik Das; Biplab Ch Das; Amartya Ghosh; Kanglong Fan; Wen Wen; Shuyan Zhai; Tianwu Zhi; Aoxiang Zhang; Jianzhao Liu; Yabin Zhang; Jiajun Wang; Yipeng Sun; Kaiwei Lian; Banghao Yin
>
> **备注:** Accepted by CVPR2026 Workshop; LoViF Challenge
>
> **摘要:** This paper reviews the LoViF 2026 Challenge on Human-oriented Semantic Image Quality Assessment. This challenge aims to raise a new direction, i.e., how to evaluate the loss of semantic information from the human perspective, intending to promote the development of some new directions, like semantic coding, processing, and semantic-oriented optimization, etc. Unlike existing datasets of quality assessment, we form a dataset of human-oriented semantic quality assessment, termed the SeIQA dataset. This dataset is divided into three parts for this competition: (i) training data: 510 pairs of degraded images and their corresponding ground truth references; (ii) validation data: 80 pairs of degraded images and their corresponding ground-truth references; (iii) testing data: 160 pairs of degraded images and their corresponding ground-truth references. The primary objective of this challenge is to establish a new and powerful benchmark for human-oriented semantic image quality assessment. There are a total of 58 teams registered in this competition, and 6 teams submitted valid solutions and fact sheets for the final testing phase. These submissions achieved state-of-the-art (SOTA) performance on the SeIQA dataset.
>
---
#### [new 244] COREY: A Prototype Study of Entropy-Guided Operator Fusion with Hadamard Reparameterization for Selective State Space Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度学习优化任务，旨在解决SSM在部署中的内存带宽限制问题。通过操作融合和Hadamard重参数化提升效率，降低延迟和内存流量。**

- **链接: [https://arxiv.org/pdf/2604.10597](https://arxiv.org/pdf/2604.10597)**

> **作者:** Bo Ma; Jinsong Wu; Hongjiang Wei; Weiqi Yan
>
> **摘要:** State Space Models (SSMs), represented by the Mamba family, provide linear-time sequence modeling and are attractive for long-context inference. Yet practical deployments remain memory-bandwidth limited because selective state updates are often decomposed into fragmented kernels with repeated intermediate tensor materialization. We present COREY, a prototype framework that combines memory-aware operator fusion with Hadamard-based feature reparameterization. Activation entropy, estimated with fixed-width histograms, is used as a runtime scheduling statistic to place fusion boundaries and choose tile sizes. To regularize heavy-tailed activations, we absorb normalized Hadamard transforms into linear projections, preserving functional equivalence while reducing peak-coordinate concentration. In a controlled prototype study over heavy-tailed SSM activations, COREY consistently reduces proxy latency, improves throughput, and lowers DRAM traffic relative to unfused and fixed-depth baselines. Low-bit results are reported only through a hand-crafted stability proxy and are intended as diagnostic evidence rather than checkpoint-level quality claims. Code repository: this https URL.
>
---
#### [new 245] AmodalSVG: Amodal Image Vectorization via Semantic Layer Peeling
- **分类: cs.CV**

- **简介: 该论文提出AmodalSVG，解决图像矢量化任务中语义纠缠和几何不完整的问题。通过语义分层和自适应矢量化，生成可编辑的SVG结构。**

- **链接: [https://arxiv.org/pdf/2604.10940](https://arxiv.org/pdf/2604.10940)**

> **作者:** Juncheng Hu; Ziteng Xue; Guotao Liang; Anran Qi; Buyu Li; Sheng Wang; Dong Xu; Qian Yu
>
> **摘要:** We introduce AmodalSVG, a new framework for amodal image vectorization that produces semantically organized and geometrically complete SVG representations from natural images. Existing vectorization methods operate under a modal paradigm: tracing only visible pixels and disregarding occlusion. Consequently, the resulting SVGs are semantically entangled and geometrically incomplete, limiting SVG's structural editability. In contrast, AmodalSVG reconstructs full object geometries, including occluded regions, into independent, editable vector layers. To achieve this, AmodalSVG reformulates image vectorization as a two-stage framework, performing semantic decoupling and completion in the raster domain to produce amodally complete semantic layers, which are then independently vectorized. In the first stage, we introduce Semantic Layer Peeling (SLP), a VLM-guided strategy that progressively decomposes an image into semantically coherent layers. By hybrid inpainting, SLP recovers complete object appearances under occlusions, enabling explicit semantic decoupling. To vectorize these layers efficiently, we propose Adaptive Layered Vectorization (ALV), which dynamically modulates the primitive budget via an error-budget-driven adjustment mechanism. Extensive experiments demonstrate that AmodalSVG significantly outperforms prior methods in visual fidelity. Moreover, the resulting amodal layers enable object-level editing directly in the vector domain, capabilities not supported by existing vectorization approaches. Code will be released upon acceptance.
>
---
#### [new 246] FishRoPE: Projective Rotary Position Embeddings for Omnidirectional Visual Perception
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉感知任务，解决鱼眼相机几何不一致问题。通过FishRoPE改进位置编码，使模型适应鱼眼图像，提升检测与分割性能。**

- **链接: [https://arxiv.org/pdf/2604.10391](https://arxiv.org/pdf/2604.10391)**

> **作者:** Rahul Ahuja; Mudit Jain; Bala Murali Manoghar Sai Sudhakar; Venkatraman Narayanan; Pratik Likhar; Varun Ravi Kumar; Senthil Yogamani
>
> **摘要:** Vision foundation models (VFMs) and Bird's Eye View (BEV) representation have advanced visual perception substantially, yet their internal spatial representations assume the rectilinear geometry of pinhole cameras. Fisheye cameras, widely deployed on production autonomous vehicles for their surround-view coverage, exhibit severe radial distortion that renders these representations geometrically inconsistent. At the same time, the scarcity of large-scale fisheye annotations makes retraining foundation models from scratch impractical. We present \ours, a lightweight framework that adapts frozen VFMs to fisheye geometry through two components: a frozen DINOv2 backbone with Low-Rank Adaptation (LoRA) that transfers rich self-supervised features to fisheye without task-specific pretraining, and Fisheye Rotary Position Embedding (FishRoPE), which reparameterizes the attention mechanism in the spherical coordinates of the fisheye projection so that both self-attention and cross-attention operate on angular separation rather than pixel distance. FishRoPE is architecture-agnostic, introduces negligible computational overhead, and naturally reduces to the standard formulation under pinhole geometry. We evaluate \ours on WoodScape 2D detection (54.3 mAP) and SynWoodScapes BEV segmentation (65.1 mIoU), where it achieves state-of-the-art results on both benchmarks.
>
---
#### [new 247] Evaluating the Impact of Medical Image Reconstruction on Downstream AI Fairness and Performance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像重建任务，旨在评估重建对下游AI公平性和性能的影响。研究发现传统指标与任务表现不一致，且重建可能加剧性别偏见。**

- **链接: [https://arxiv.org/pdf/2604.10904](https://arxiv.org/pdf/2604.10904)**

> **作者:** Matteo Wohlrapp; Niklas Bubeck; Daniel Rueckert; William Lotter
>
> **备注:** Proceedings of the Medical Imaging with Deep Learning (MIDL) Conference 2026
>
> **摘要:** AI-based image reconstruction models are increasingly deployed in clinical workflows to improve image quality from noisy data, such as low-dose X-rays or accelerated MRI scans. However, these models are typically evaluated using pixel-level metrics like PSNR, leaving their impact on downstream diagnostic performance and fairness unclear. We introduce a scalable evaluation framework that applies reconstruction and diagnostic AI models in tandem, which we apply to two tasks (classification, segmentation), three reconstruction approaches (U-Net, GAN, diffusion), and two data types (X-ray, MRI) to assess the potential downstream implications of reconstruction. We find that conventional reconstruction metrics poorly track task performance, where diagnostic accuracy remains largely stable even as reconstruction PSNR declines with increasing image noise. Fairness metrics exhibit greater variability, with reconstruction sometimes amplifying demographic biases, particularly regarding patient sex. However, the overall magnitude of this additional bias is modest compared to the inherent biases already present in diagnostic models. To explore potential bias mitigation, we adapt two strategies from classification literature to the reconstruction setting, but observe limited efficacy. Overall, our findings emphasize the importance of holistic performance and fairness assessments throughout the entire medical imaging workflow, especially as generative reconstruction models are increasingly deployed.
>
---
#### [new 248] Online Reasoning Video Object Segmentation
- **分类: cs.CV**

- **简介: 该论文研究在线推理视频对象分割任务，解决实时、因果决策与指代变化问题。提出ORVOSB基准和基线模型，支持长期时序推理。**

- **链接: [https://arxiv.org/pdf/2604.11411](https://arxiv.org/pdf/2604.11411)**

> **作者:** Jinyuan Liu; Yang Wang; Zeyu Zhao; Weixin Li; Song Wang; Ruize Han
>
> **摘要:** Reasoning video object segmentation predicts pixel-level masks in videos from natural-language queries that may involve implicit and temporally grounded references. However, existing methods are developed and evaluated in an offline regime, where the entire video is available at inference time and future frames can be exploited for retrospective disambiguation, deviating from real-world deployments that require strictly causal, frame-by-frame decisions. We study Online Reasoning Video Object Segmentation (ORVOS), where models must incrementally interpret queries using only past and current frames without revisiting previous predictions, while handling referent shifts as events unfold. To support evaluation, we introduce ORVOSB, a benchmark with frame-level causal annotations and referent-shift labels, comprising 210 videos, 12,907 annotated frames, and 512 queries across five reasoning categories. We further propose a baseline with continually-updated segmentation prompts and a structured temporal token reservoir for long-horizon reasoning under bounded computation. Experiments show that existing methods struggle under strict causality and referent shifts, while our baseline establishes a strong foundation for future research.
>
---
#### [new 249] TAMISeg: Text-Aligned Multi-scale Medical Image Segmentation with Semantic Encoder Distillation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决标注不足和结构复杂的问题。提出TAMISeg框架，结合文本提示和语义蒸馏，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2604.10912](https://arxiv.org/pdf/2604.10912)**

> **作者:** Qiang Gao; Yi Wang; Yong Zhang; Yong Li; Yongbing Deng; Lan Du; Cunjian Chen
>
> **备注:** Accepted by IEEE International Conference on Multimedia and Expo (ICME), 2026
>
> **摘要:** Medical image segmentation remains challenging due to limited fine-grained annotations, complex anatomical structures, and image degradation from noise, low contrast, or illumination variation. We propose TAMISeg, a text-guided segmentation framework that incorporates clinical language prompts and semantic distillation as auxiliary semantic cues to enhance visual understanding and reduce reliance on pixel-level fine-grained annotations. TAMISeg integrates three core components: a consistency-aware encoder pretrained with strong perturbations for robust feature extraction, a semantic encoder distillation module with supervision from a frozen DINOv3 teacher to enhance semantic discriminability, and a scale-adaptive decoder that segments anatomical structures across different spatial scales. Experiments on the Kvasir-SEG, MosMedData+, and QaTa-COV19 datasets demonstrate that TAMISeg consistently outperforms existing uni-modal and multi-modal methods in both qualitative and quantitative evaluations. Code will be made publicly available at this https URL.
>
---
#### [new 250] Learning Long-term Motion Embeddings for Efficient Kinematics Generation
- **分类: cs.CV**

- **简介: 该论文属于动作生成任务，旨在高效生成符合文本或空间指令的长期真实运动。通过学习长时运动嵌入并训练条件流匹配模型，提升生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.11737](https://arxiv.org/pdf/2604.11737)**

> **作者:** Nick Stracke; Kolja Bauer; Stefan Andreas Baumann; Miguel Angel Bautista; Josh Susskind; Björn Ommer
>
> **备注:** for the project page and code, view this https URL
>
> **摘要:** Understanding and predicting motion is a fundamental component of visual intelligence. Although modern video models exhibit strong comprehension of scene dynamics, exploring multiple possible futures through full video synthesis remains prohibitively inefficient. We model scene dynamics orders of magnitude more efficiently by directly operating on a long-term motion embedding that is learned from large-scale trajectories obtained from tracker models. This enables efficient generation of long, realistic motions that fulfill goals specified via text prompts or spatial pokes. To achieve this, we first learn a highly compressed motion embedding with a temporal compression factor of 64x. In this space, we train a conditional flow-matching model to generate motion latents conditioned on task descriptions. The resulting motion distributions outperform those of both state-of-the-art video models and specialized task-specific approaches.
>
---
#### [new 251] Efficient Transceiver Design for Aerial Image Transmission and Large-scale Scene Reconstruction
- **分类: cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于无线图像传输任务，解决低空智能网络中3D场景重建的高效传输问题。通过结合深度学习与3D高斯泼溅，设计端到端收发器，提升重建质量并降低开销。**

- **链接: [https://arxiv.org/pdf/2604.11098](https://arxiv.org/pdf/2604.11098)**

> **作者:** Zeyi Ren; Jialin Dong; Wei Zuo; Yikun Wang; Bingyang Cheng; Sheng Zhou; Zhisheng Niu
>
> **备注:** 6 pages, 6 figures, submitted to IEEE ISIT-w
>
> **摘要:** Large-scale three-dimensional (3D) scene reconstruction in low-altitude intelligent networks (LAIN) demands highly efficient wireless image transmission. However, existing schemes struggle to balance severe pilot overhead with the transmission accuracy required to maintain reconstruction fidelity. To strike a balance between efficiency and reliability, this paper proposes a novel deep learning-based end-to-end (E2E) transceiver design that integrates 3D Gaussian Splatting (3DGS) directly into the training process. By jointly optimizing the communication modules via the combined 3DGS rendering loss, our approach explicitly improves scene recovery quality. Furthermore, this task-driven framework enables the use of a sparse pilot scheme, significantly reducing transmission overhead while maintaining robust image recovery under low-altitude channel conditions. Extensive experiments on real-world aerial image datasets demonstrate that the proposed E2E design significantly outperforms existing baselines, delivering superior transmission performance and accurate 3D scene reconstructions.
>
---
#### [new 252] Enhancing Fine-Grained Spatial Grounding in 3D CT Report Generation via Discriminative Guidance
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决3D CT报告中细粒度空间定位不足的问题。通过引入DCP-PD框架提升报告生成的准确性与空间一致性。**

- **链接: [https://arxiv.org/pdf/2604.10437](https://arxiv.org/pdf/2604.10437)**

> **作者:** Chenyu Wang; Weicheng Dai; Han Liu; Wenchao Li; Kayhan Batmanghelich
>
> **摘要:** Vision--language models (VLMs) for radiology report generation (RRG) can produce long-form chest CT reports from volumetric scans and show strong potential to improve radiology workflow efficiency and consistency. However, existing methods face two key limitations: (i) training supervision is often coarse, aligning a whole CT volume with a full free-text report without explicit alignment for fine-grained attributes or pathology locations; and (ii) evaluation is typically holistic (lexical overlap, entity matching, or LLM-as-a-judge scores) and not diagnostic for spatial grounding. We propose \emph{Discriminative Cue-Prompting with Prompt Dropout (DCP-PD)}, a plug-and-play framework that distills fine-grained cues from free-text reports and uses them to guide report generation while mitigating shortcut reliance via prompt dropout. DCP-PD achieves state-of-the-art performance on CT-RATE, improving macro F1 from $=0.501$ to $0.603$ (20% relative), and substantially boosts out-of-distribution performance on Rad-ChestCT from F1 $=0.266$ to $0.503$ (89% relative). Finally, we introduce a hierarchical, location-aware question-set protocol (presence $\rightarrow$ laterality $\rightarrow$ lobe) to directly assess pathology-location grounding, showing that fine-grained spatial localization remains challenging even for models that score highly on current benchmarks.
>
---
#### [new 253] Spatio-Temporal Difference Guided Motion Deblurring with the Complementary Vision Sensor
- **分类: cs.CV**

- **简介: 该论文属于图像去模糊任务，解决极端运动下RGB图像模糊问题。通过引入互补视觉传感器数据，提出STGDNet网络，融合时空差异信息提升去模糊效果。**

- **链接: [https://arxiv.org/pdf/2604.10554](https://arxiv.org/pdf/2604.10554)**

> **作者:** Yapeng Meng; Lin Yang; Yuguo Chen; Xiangru Chen; Taoyi Wang; Lijian Wang; Zheyu Yang; Yihan Lin; Rong Zhao
>
> **摘要:** Motion blur arises when rapid scene changes occur during the exposure period, collapsing rich intra-exposure motion into a single RGB frame. Without explicit structural or temporal cues, RGB-only deblurring is highly ill-posed and often fails under extreme motion. Inspired by the human visual system, brain-inspired vision sensors introduce temporally dense information to alleviate this problem. However, event cameras still suffer from event rate saturation under rapid motion, while the event modality entangles edge features and motion cues, which limits their effectiveness. As a recent breakthrough, the complementary vision sensor (CVS), Tianmouc, captures synchronized RGB frames together with high-frame-rate, multi-bit spatial difference (SD, encoding structural edges) and temporal difference (TD, encoding motion cues) data within a single RGB exposure, offering a promising solution for RGB deblurring under extreme dynamic scenes. To fully leverage these complementary modalities, we propose Spatio-Temporal Difference Guided Deblur Net (STGDNet), which adopts a recurrent multi-branch architecture that iteratively encodes and fuses SD and TD sequences to restore structure and color details lost in blurry RGB inputs. Our method outperforms current RGB or event-based approaches in both synthetic CVS dataset and real-world evaluations. Moreover, STGDNet exhibits strong generalization capability across over 100 extreme real-world scenarios. Project page: this https URL
>
---
#### [new 254] Lightweight Low-Light Image Enhancement via Distribution-Normalizing Preprocessing and Depthwise U-Net
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于低光图像增强任务，旨在提升低光图像质量。提出一种轻量级两阶段框架，结合分布归一化预处理和深度可分离U-Net，减少参数量并提高效率。**

- **链接: [https://arxiv.org/pdf/2604.11071](https://arxiv.org/pdf/2604.11071)**

> **作者:** Shimon Murai; Teppei Kurita; Ryuta Satoh; Yusuke Moriuchi
>
> **备注:** Technical report for the NTIRE 2026 Efficient Low-Light Image Enhancement Challenge (CVPR 2026 Workshops), 4th place solution
>
> **摘要:** We present a lightweight two-stage framework for low-light image enhancement (LLIE) that achieves competitive perceptual quality with significantly fewer parameters than existing methods. Our approach combines frozen algorithm-based preprocessing with a compact U-Net built entirely from depthwise-separable convolutions. The preprocessing normalizes the input distribution by providing complementary brightness-corrected views, enabling the trainable network to focus on residual color correction. Our method achieved 4th place in the CVPR 2026 NTIRE Efficient Low-Light Image Enhancement Challenge. We further provide extended benchmarks and ablations to demonstrate the general effectiveness of our methods.
>
---
#### [new 255] Immune2V: Image Immunization Against Dual-Stream Image-to-Video Generation
- **分类: cs.CV**

- **简介: 该论文属于图像到视频生成任务，旨在防御深度伪造。针对现有方法对I2V攻击无效的问题，提出Immune2V框架，通过控制潜在空间和文本引导提升免疫效果。**

- **链接: [https://arxiv.org/pdf/2604.10837](https://arxiv.org/pdf/2604.10837)**

> **作者:** Zeqian Long; Ozgur Kara; Haotian Xue; Yongxin Chen; James M. Rehg
>
> **摘要:** Image-to-video (I2V) generation has the potential for societal harm because it enables the unauthorized animation of static images to create realistic deepfakes. While existing defenses effectively protect against static image manipulation, extending these to I2V generation remains underexplored and non-trivial. In this paper, we systematically analyze why modern I2V models are highly robust against naive image-level adversarial attacks (i.e., immunization). We observe that the video encoding process rapidly dilutes the adversarial noise across future frames, and the continuous text-conditioned guidance actively overrides the intended disruptive effect of the immunization. Building on these findings, we propose the Immune2V framework which enforces temporally balanced latent divergence at the encoder level to prevent signal dilution, and aligns intermediate generative representations with a precomputed collapse-inducing trajectory to counteract the text-guidance override. Extensive experiments demonstrate that Immune2V produces substantially stronger and more persistent degradation than adapted image-level baselines under the same imperceptibility budget.
>
---
#### [new 256] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言检索任务，旨在解决传统系统无法适应用户条件的问题。提出CLAY方法，在不额外训练的情况下，将预训练模型的嵌入空间转化为文本条件相似性空间，实现高效多条件检索。**

- **链接: [https://arxiv.org/pdf/2604.11539](https://arxiv.org/pdf/2604.11539)**

> **作者:** Sohwi Lim; Lee Hyoseok; Jungjoon Park; Tae-Hyun Oh
>
> **备注:** CVPR 2026, Project page: this https URL
>
> **摘要:** Human perception of visual similarity is inherently adaptive and subjective, depending on the users' interests and focus. However, most image retrieval systems fail to reflect this flexibility, relying on a fixed, monolithic metric that cannot incorporate multiple conditions simultaneously. To address this, we propose CLAY, an adaptive similarity computation method that reframes the embedding space of pretrained Vision-Language Models (VLMs) as a text-conditional similarity space without additional training. This design separates the textual conditioning process and visual feature extraction, allowing highly efficient and multi-conditioned retrieval with fixed visual embeddings. We also construct a synthetic evaluation dataset CLAY-EVAL, for comprehensive assessment under diverse conditioned retrieval settings. Experiments on standard datasets and our proposed dataset show that CLAY achieves high retrieval accuracy and notable computational efficiency compared to previous works.
>
---
#### [new 257] Do vision models perceive illusory motion in static images like humans?
- **分类: cs.CV**

- **简介: 该论文属于视觉运动感知任务，旨在探究视觉模型是否能像人类一样在静态图像中感知错觉运动。研究对比了多种光学流模型，发现多数无法准确模拟人类感知，仅人类启发模型表现良好。**

- **链接: [https://arxiv.org/pdf/2604.09853](https://arxiv.org/pdf/2604.09853)**

> **作者:** Isabella Elaine Rosario; Fan L. Cheng; Zitang Sun; Nikolaus Kriegeskorte
>
> **备注:** Accepted to CVPR 2026 Workshops (Findings). * Equal contribution
>
> **摘要:** Understanding human motion processing is essential for building reliable, human-centered computer vision systems. Although deep neural networks (DNNs) achieve strong performance in optical flow estimation, they remain less robust than humans and rely on fundamentally different computational strategies. Visual motion illusions provide a powerful probe into these mechanisms, revealing how human and machine vision align or diverge. While recent DNN-based motion models can reproduce dynamic illusions such as reverse-phi, it remains unclear whether they can perceive illusory motion in static images, exemplified by the Rotating Snakes illusion. We evaluate several representative optical flow models on Rotating Snakes and show that most fail to generate flow fields consistent with human perception. Under simulated conditions mimicking saccadic eye movements, only the human-inspired Dual-Channel model exhibits the expected rotational motion, with the closest correspondence emerging during the saccade simulation. Ablation analyses further reveal that both luminance-based and higher-order color--feature--based motion signals contribute to this behavior and that a recurrent attention mechanism is critical for integrating local cues. Our results highlight a substantial gap between current optical-flow models and human visual motion processing, and offer insights for developing future motion-estimation systems with improved correspondence to human perception and human-centric AI.
>
---
#### [new 258] Investigating Bias and Fairness in Appearance-based Gaze Estimation
- **分类: cs.CV**

- **简介: 该论文属于视觉任务，研究外观基 gaze 估计中的偏差与公平性问题。旨在评估不同人群的性能差异，提出公平性基准，并探讨缓解策略的有效性。**

- **链接: [https://arxiv.org/pdf/2604.10707](https://arxiv.org/pdf/2604.10707)**

> **作者:** Burak Akgül; Erol Şahin; Sinan Kalkan
>
> **摘要:** While appearance-based gaze estimation has achieved significant improvements in accuracy and domain adaptation, the fairness of these systems across different demographic groups remains largely unexplored. To date, there is no comprehensive benchmark quantifying algorithmic bias in gaze estimation. This paper presents the first extensive evaluation of fairness in appearance-based gaze estimation, focusing on ethnicity and gender attributes. We establish a fairness baseline by analyzing state-of-the-art models using standard fairness metrics, revealing significant performance disparities. Furthermore, we evaluate the effectiveness of existing bias mitigation strategies when applied to the gaze domain and show that their fairness contributions are limited. We summarize key insights and open issues. Overall, our work calls for research into developing robust, equitable gaze estimators. To support future research and reproducibility, we publicly release our annotations, code, and trained models at: this http URL
>
---
#### [new 259] A Compact and Efficient 1.251 Million Parameter Machine Learning CNN Model PD36-C for Plant Disease Detection: A Case Study
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于植物病害分类任务，旨在开发一个高效的小型CNN模型PD36-C，用于准确检测植物疾病。**

- **链接: [https://arxiv.org/pdf/2604.11332](https://arxiv.org/pdf/2604.11332)**

> **作者:** Shkelqim Sherifi
>
> **备注:** 17 pages, 24 figures
>
> **摘要:** Deep learning has markedly advanced image based plant disease diagnosis as improved hardware and dataset quality have enabled increasingly accurate neural network models. This paper presents PD36 C, a compact convolutional neural network (1,250,694 parameters and 4.77 MB) for plant disease classification. Trained with TensorFlow Keras on the New Plant Diseases Dataset (87k images, 38 classes), PD36 C is designed for robustness and edge deployability, complemented by a Qt for Python desktop application that offers an intuitive GUI and offline inference on commodity hardware. Across experiments, training accuracy reached 0.99697 by epoch 30, and average test accuracy was 0.9953 across 38 classes. Per class performance is uniformly high; on the lower end, Corn (maize) Cercospora leaf spot achieved precision around 0.9777 and recall around 0.9634, indicating occasional confusion with visually similar categories, while on the upper end numerous classes including Apple Black rot, Cedar apple rust, Blueberry healthy, Cherry Powdery mildew, Cherry healthy, and all four grape categories achieved perfect precision 1.00 and recall of 1.00, indicating no false positives and strong coverage. These results show that with a well curated dataset and careful architectural design, small CNNs can achieve competitive accuracy compared with recent baselines while remaining practical for edge scenarios. We also note typical constraints such as adverse weather, low quality imagery, and leaves exhibiting multiple concurrent diseases that can degrade performance and warrant future work on domain robustness. Overall, PD36 C and its application pipeline contribute a field ready, efficient solution for AI assisted plant disease detection in smart agriculture.
>
---
#### [new 260] Byte-level generative predictions for forensics multimedia carving
- **分类: cs.CV**

- **简介: 该论文属于数字取证任务，旨在解决碎片化多媒体文件恢复问题。通过生成模型预测缺失字节，提升碎片匹配效果。**

- **链接: [https://arxiv.org/pdf/2604.11010](https://arxiv.org/pdf/2604.11010)**

> **作者:** Jaewon Lee; Md Eimran Hossain Eimon; Avinash Srinivasan; Hari Kalva
>
> **备注:** Accepted for publication at the "SPIE Defense + Security" Conference
>
> **摘要:** Digital forensic investigations often face significant challenges when recovering fragmented multimedia files that lack file system metadata. While traditional file carving relies on signatures and discriminative deep learning models for fragment classification, these methods cannot reconstruct or predict missing data. We propose a generative approach to multimedia carving using bGPT, a byte-level transformer designed for next-byte prediction. By feeding partial BMP image data into the model, we simulate the generation of likely fragment continuations. We evaluate the fidelity of these predictions using different metrics, namely, cosine similarity, structural similarity index (SSIM), chi-square distance, and Jensen-Shannon divergence (JSD). Our findings demonstrate that generative models can effectively predict byte-level patterns to support fragment matching in unallocated disk space.
>
---
#### [new 261] CAGE: Bridging the Accuracy-Aesthetics Gap in Educational Diagrams via Code-Anchored Generative Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于教育图像生成任务，旨在解决准确性和美观性之间的矛盾。通过结合代码生成与扩散模型，提出CAGE方法提升教育图表质量。**

- **链接: [https://arxiv.org/pdf/2604.09691](https://arxiv.org/pdf/2604.09691)**

> **作者:** Dikshant Kukreja; Kshitij Sah; Karan Goyal; Mukesh Mohania; Vikram Goyal
>
> **摘要:** Educational diagrams -- labeled illustrations of biological processes, chemical structures, physical systems, and mathematical concepts -- are essential cognitive tools in K-12 instruction. Yet no existing method can generate them both accurately and engagingly. Open-source diffusion models produce visually rich images but catastrophically garble text labels. Code-based generation via LLMs guarantees label correctness but yields visually flat outputs. Closed-source APIs partially bridge this gap but remain unreliable and prohibitively expensive at educational scale. We quantify this accuracy-aesthetics dilemma across all three paradigms on 400 K-12 diagram prompts, measuring both label fidelity and visual quality through complementary automated and human evaluation protocols. To resolve it, we propose CAGE (Code-Anchored Generative Enhancement): an LLM synthesizes executable code producing a structurally correct diagram, then a diffusion model conditioned on the programmatic output via ControlNet refines it into a visually polished graphic while preserving label fidelity. We also introduce EduDiagram-2K, a collection of 2,000 paired programmatic-stylized diagrams enabling this pipeline, and present proof-of-concept results and a research agenda for the multimedia community.
>
---
#### [new 262] FREE-Switch: Frequency-based Dynamic LoRA Switch for Style Transfer
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，解决多适配器融合中的内容漂移和细节损失问题。提出FREE-Switch方法，通过频率域动态切换LoRA适配器，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.10023](https://arxiv.org/pdf/2604.10023)**

> **作者:** Shenghe Zheng; Minyu Zhang; Tianhao Liu; Hongzhi Wang
>
> **备注:** CVPR Findings 2026
>
> **摘要:** With the growing availability of open-sourced adapters trained on the same diffusion backbone for diverse scenes and objects, combining these pretrained weights enables low-cost customized generation. However, most existing model merging methods are designed for classification or text generation, and when applied to image generation, they suffer from content drift due to error accumulation across multiple diffusion steps. For image-oriented methods, training-based approaches are computationally expensive and unsuitable for edge deployment, while training-free ones use uniform fusion strategies that ignore inter-adapter differences, leading to detail degradation. We find that since different adapters are specialized for generating different types of content, the contribution of each diffusion step carries different significance for each adapter. Accordingly, we propose a frequency-domain importance-driven dynamic LoRA switch method. Furthermore, we observe that maintaining semantic consistency across adapters effectively mitigates detail loss; thus, we design an automatic Generation Alignment mechanism to align generation intents at the semantic level. Experiments demonstrate that our FREE-Switch (Frequency-based Efficient and Dynamic LoRA Switch) framework efficiently combines adapters for different objects and styles, substantially reducing the training cost of high-quality customized generation.
>
---
#### [new 263] NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images: Methods and Results
- **分类: cs.CV**

- **简介: 本文属于雨滴去除任务，旨在解决昼夜双焦点图像中的雨滴去除问题。通过构建数据集和举办挑战赛，评估并推动相关技术的发展。**

- **链接: [https://arxiv.org/pdf/2604.10634](https://arxiv.org/pdf/2604.10634)**

> **作者:** Xin Li; Yeying Jin; Suhang Yao; Beibei Lin; Zhaoxin Fan; Wending Yan; Xin Jin; Zongwei Wu; Bingchen Li; Peishu Shi; Yufei Yang; Yu Li; Zhibo Chen; Bihan Wen; Robby T. Tan; Radu Timofte; Runzhe Li; Kui Jiang; Zhaocheng Yu; Yiang Chen; Junjun Jiang; Xianming Liu; Hongde Gu; Zeliang Li; Mache You; Jiangxin Dong; Jinshan Pan; Qiyu Rong; Bowen Shao; Hongyuan Jing; Mengmeng Zhang; Bo Ding; Hui Zhang; Yi Ren; Mohab Kishawy; Jun Chen; Anh-Kiet Duong; Petra Gomez-Kramer; Jean-Michel Carozza; Wangzhi Xing; Xin Lu; Enxuan Gu; Jingxi Zhang; Diqi Chen; Qiaosi Yi; Bingcai Wei; Wenjie Li; Bowen Tie; Heng Guo; Zhanyu Ma; Jiachen Tu; Guoyi Xu; Yaoxin Jiang; Cici Liu; Yaokun Shi; Paula Garrido Mellado; Daniel Feijoo; Alvaro Garcia Lara; Marcos V. Conde; Zhidong Zhu; Bangshu Xiong; Qiaofeng Ou; Zhibo Rao; Wei Li; Zida Zhang; Hui Geng; Qisheng Xu; Xuyao Deng; Changjian Wang; Kele Xu; Guanglu Dong; Qiyao Zhao; Tianheng Zheng; Chunlei Li; Lichao Mou; Chao Ren; Chang-De Peng; Chieh-Yu Tsai; Guan-Cheng Liu; Li-Wei Kang; Abhishek Rajak; Milan Kumar Singh; Ankit Kumar; Dimple Sonone; Kishor Upla; Kiran Raja; Huilin Zhao; Xing Xu; Chuan Chen; Yeming Lao; Wenjing Xun; Li Yang; Bilel Benjdira; Anas M. Ali; Wadii Boulila; Hao Yang; Ruikun Zhang; Liyuan Pan
>
> **备注:** Accepted by CVPR2026 Workshop; NTIRE 2026 Challenge Report
>
> **摘要:** This paper presents an overview of the NTIRE 2026 Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images. Building upon the success of the first edition, this challenge attracted a wide range of impressive solutions, all developed and evaluated on our real-world Raindrop Clarity dataset~\cite{jin2024raindrop}. For this edition, we adjust the dataset with 14,139 images for training, 407 images for validation, and 593 images for testing. The primary goal of this challenge is to establish a strong and practical benchmark for the removal of raindrops under various illumination and focus conditions. In total, 168 teams have registered for the competition, and 17 teams submitted valid final solutions and fact sheets for the testing phase. The submitted methods achieved strong performance on the Raindrop Clarity dataset, demonstrating the growing progress in this challenging task.
>
---
#### [new 264] Efficient KernelSHAP Explanations for Patch-based 3D Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D医学图像分割中的解释性问题，提出高效KernelSHAP方法，通过区域限制和缓存加速，提升解释效率与临床可解释性。**

- **链接: [https://arxiv.org/pdf/2604.11775](https://arxiv.org/pdf/2604.11775)**

> **作者:** Ricardo Coimbra Brioso; Giulio Sichili; Damiano Dei; Nicola Lambri; Pietro Mancosu; Marta Scorsetti; Daniele Loiacono
>
> **摘要:** Perturbation-based explainability methods such as KernelSHAP provide model-agnostic attributions but are typically impractical for patch-based 3D medical image segmentation due to the large number of coalition evaluations and the high cost of sliding-window inference. We present an efficient KernelSHAP framework for volumetric CT segmentation that restricts computation to a user-defined region of interest and its receptive-field support, and accelerates inference via patch logit caching, reusing baseline predictions for unaffected patches while preserving nnU-Net's fusion scheme. To enable clinically meaningful attributions, we compare three automatically generated feature abstractions within the receptive-field crop: whole-organ units, regular FCC supervoxels, and hybrid organ-aware supervoxels, and we study multiple aggregation/value functions targeting stabilizing evidence (TP/Dice/Soft Dice) or false-positive behavior. Experiments on whole-body CT segmentations show that caching substantially reduces redundant computation (with computational savings ranging from 15% to 30%) and that faithfulness and interpretability exhibit clear trade-offs: regular supervoxels often maximize perturbation-based metrics but lack anatomical alignment, whereas organ-aware units yield more clinically interpretable explanations and are particularly effective for highlighting false-positive drivers under normalized metrics.
>
---
#### [new 265] Towards Adaptive Open-Set Object Detection via Category-Level Collaboration Knowledge Mining
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自适应开放集目标检测任务，解决域间泛化和新类别识别问题。提出类别级协作知识挖掘方法，提升跨域表示和新类别分类性能。**

- **链接: [https://arxiv.org/pdf/2604.11195](https://arxiv.org/pdf/2604.11195)**

> **作者:** Yuqi Ji; Junjie Ke; Lihuo He; Lizhi Wang; Xinbo Gao
>
> **备注:** 15 pages,9 figures,accepted by IEEE Transactions on Image Processing
>
> **摘要:** Existing object detectors often struggle to generalize across domains while adapting to emerging novel categories. Adaptive open-set object detection (AOOD) addresses this challenge by training on base categories in the source domain and adapting to both base and novel categories in the target domain without target annotations. However, current AOOD methods remain limited by weak cross-domain representations, ambiguity among novel categories, and source-domain feature bias. To address these issues, we propose a category-level collaboration knowledge mining strategy that exploits both inter-class and intra-class relationships across domains. Specifically, we construct a clustering-based memory bank to encode class prototypes, auxiliary features, and intra-class disparity information, and iteratively update it via unsupervised clustering to enhance category-level knowledge representation. We further design a base-to-novel selection metric to discover source-domain features related to novel categories and use them to initialize novel-category classifiers. In addition, an adaptive feature assignment strategy transfers the learned category-level knowledge to the target domain and asynchronously updates the memory bank to alleviate source-domain bias. Extensive experiments on multiple benchmarks show that our method consistently surpasses state-of-the-art AOOD methods by 1.1-5.5 mAP.
>
---
#### [new 266] Training-Free Model Ensemble for Single-Image Super-Resolution via Strong-Branch Compensation
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在无需额外训练的情况下有效融合多个预训练模型。通过构建双分支结构，提升重建质量，实现高效输出级集成。**

- **链接: [https://arxiv.org/pdf/2604.11564](https://arxiv.org/pdf/2604.11564)**

> **作者:** Gengjia Chang; Xining Ge; Weijun Yuan; Zhan Li; Qiurong Song; Luen Zhu; Shuhong Liu
>
> **摘要:** Single-image super-resolution has progressed from deep convolutional baselines to stronger Transformer and state-space architectures, yet the corresponding performance gains typically come with higher training cost, longer engineering iteration, and heavier deployment burden. In many practical settings, multiple pretrained models with partially complementary behaviors are already available, and the binding constraint is no longer architectural capacity but how effectively their outputs can be combined without additional training. Rather than pursuing further architectural redesign, this paper proposes a training-free output-level ensemble framework. A dual-branch pipeline is constructed in which a Hybrid attention network with TLC inference provides stable main reconstruction, while a MambaIRv2 branch with geometric self-ensemble supplies strong compensation for high-frequency detail recovery. The two branches process the same low-resolution input independently and are fused in the image space via a lightweight weighted combination, without updating any model parameters or introducing an additional trainable module. As our solution to the NTIRE 2026 Image Super-Resolution ($\times 4$) Challenge, the proposed design consistently improves over the base branch and slightly exceeds the pure strong branch in PSNR at the best operating point under a unified DIV2K bicubic $\times 4$ evaluation protocol. Ablation studies confirm that output-level compensation provides a low-overhead and practically accessible upgrade path for existing super-resolution systems.
>
---
#### [new 267] Learnable Motion-Focused Tokenization for Effective and Efficient Video Unsupervised Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视频无监督域适应任务，旨在提升动作识别模型在不同域间的适应能力。针对背景冗余和计算效率低的问题，提出LMFT方法，通过学习过滤低运动区域，提高模型效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.09955](https://arxiv.org/pdf/2604.09955)**

> **作者:** Tzu Ling Liu; Ian Stavness; Mrigank Rochan
>
> **备注:** Accepted to IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Video Unsupervised Domain Adaptation (VUDA) poses a significant challenge in action recognition, requiring the adaptation of a model from a labeled source domain to an unlabeled target domain. Despite recent advances, existing VUDA methods often fall short of fully supervised performance, a key reason being the prevalence of static and uninformative backgrounds that exacerbate domain shifts. Additionally, prior approaches largely overlook computational efficiency, limiting real-world adoption. To address these issues, we propose Learnable Motion-Focused Tokenization (LMFT) for VUDA. LMFT tokenizes video frames into patch tokens and learns to discard low-motion, redundant tokens, primarily corresponding to background regions, while retaining motion-rich, action-relevant tokens for adaptation. Extensive experiments on three standard VUDA benchmarks across 21 domain adaptation settings show that our VUDA framework with LMFT achieves state-of-the-art performance while significantly reducing computational overhead. LMFT thus enables VUDA that is both effective and computationally efficient.
>
---
#### [new 268] DeepShapeMatchingKit: Accelerated Functional Map Solver and Shape Matching Pipelines Revisited
- **分类: cs.CV**

- **简介: 该论文属于3D形状匹配任务，解决功能映射计算效率低的问题，提出向量化方法提升速度，并改进特征提取与评估指标。**

- **链接: [https://arxiv.org/pdf/2604.10377](https://arxiv.org/pdf/2604.10377)**

> **作者:** Yizheng Xie; Lennart Bastian; Congyue Deng; Thomas W. Mitchel; Maolin Gao; Daniel Cremers
>
> **备注:** 10 pages, 8 figures, CVPR 2026 Image Matching Workshop (IEEE proceedings)
>
> **摘要:** Deep functional maps, leveraging learned feature extractors and spectral correspondence solvers, are fundamental to non-rigid 3D shape matching. Based on an analysis of open-source implementations, we find that standard functional map implementations solve k independent linear systems serially, which is a computational bottleneck at higher spectral resolution. We thus propose a vectorized reformulation that solves all systems in a single kernel call, achieving up to a 33x speedup while preserving the exact solution. Furthermore, we identify and document a previously unnoticed implementation divergence in the spatial gradient features of the mainstay DiffusionNet: two variants that parameterize distinct families of tangent-plane transformations, and present experiments analyzing their respective behaviors across diverse benchmarks. We additionally revisit overlap prediction evaluation for partial-to-partial matching and show that balanced accuracy provides a useful complementary metric under varying overlap ratios. To share these advancements with the wider community, we present an open-source codebase, DeepShapeMatchingKit, that incorporates these improvements and standardizes training, evaluation, and data pipelines for common deep shape matching methods. The codebase is available at: this https URL
>
---
#### [new 269] Improving Layout Representation Learning Across Inconsistently Annotated Datasets via Agentic Harmonization
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决不一致标注数据带来的模型性能下降问题。通过引入代理标签对齐流程，提升跨数据集的布局表示学习效果。**

- **链接: [https://arxiv.org/pdf/2604.11042](https://arxiv.org/pdf/2604.11042)**

> **作者:** Renyu Li; Vladimir Kirilenko; Yao You; Crag Wolfe
>
> **备注:** 12 pages, 6 figures, 5 tables
>
> **摘要:** Fine-tuning object detection (OD) models on combined datasets assumes annotation compatibility, yet datasets often encode conflicting spatial definitions for semantically equivalent categories. We propose an agentic label harmonization workflow that uses a vision-language model to reconcile both category semantics and bounding box granularity across heterogeneous sources before training. We evaluate on document layout detection as a challenging case study, where annotation standards vary widely across corpora. Without harmonization, naïve mixed-dataset fine-tuning degrades a pretrained RT-DETRv2 detector: on SCORE-Bench, which measures how accurately the full document conversion pipeline reproduces ground-truth structure, table TEDS drops from 0.800 to 0.750. Applied to two corpora whose 16 and 10 category taxonomies share only 8 direct correspondences, harmonization yields consistent gains across content fidelity, table structure, and spatial consistency: detection F-score improves from 0.860 to 0.883, table TEDS improves to 0.814, and mean bounding box overlap drops from 0.043 to 0.016. Representation analysis further shows that harmonized training produces more compact and separable post-decoder embeddings, confirming that annotation inconsistency distorts the learned feature space and that resolving it before training restores representation structure.
>
---
#### [new 270] Multi-Frequency Local Plasticity for Visual Representation Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉表征学习任务，旨在探索结构化架构偏差是否能弥补端到端梯度学习的不足。通过多频Gabor分解、竞争学习和反馈机制等方法，在不依赖反向传播的情况下提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.09734](https://arxiv.org/pdf/2604.09734)**

> **作者:** Mehdi Fatan Serj; C. Alejandro Parraga; Xavier Otazu
>
> **摘要:** We study how far structured architectural bias can compensate for the absence of end-to-end gradient-based representation learning in visual recognition. Building on the VisNet tradition, we introduce a modular hierarchical framework combining: (i) fixed multi-frequency Gabor decomposition into F=7 parallel streams; (ii) within-stream competitive learning with Hebbian and Oja updates and anti-Hebbian decorrelation; (iii) an associative memory module inspired by modern Hopfield retrieval; and (iv) iterative top-down modulation using local prediction and reconstruction signals. Representational layers are trained without end-to-end backpropagation through the full hierarchy; only the final linear readout and top-down projection matrices are optimized by gradient descent. We therefore interpret the model as a hybrid system that is predominantly locally trained but includes a small number of gradient-trained parameters. On CIFAR-10, the full model reaches 80.1% +/- 0.3% top-1 accuracy, linear probe), compared with 71.0% for a Hebbian-only baseline and 83.4% for a gradient-trained model on the same fixed Gabor basis. On CIFAR-100, performance is 54.8%. Factorial analysis indicates that multi-frequency streams, associative memory, and top-down feedback contribute largely additively, with a significant Streams x TopDown interaction (p=0.02). These results suggest that carefully chosen architectural priors can recover a substantial fraction of the performance typically associated with global gradient training, while leaving a measurable residual gap. Experiments are limited to CIFAR-10/100.
>
---
#### [new 271] Uncertainty-Guided Attention and Entropy-Weighted Loss for Precise Plant Seedling Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于植物幼苗分割任务，解决复杂背景和细小结构分割难题。提出UGDA-Net，结合不确定性引导注意力和熵加权损失，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.10823](https://arxiv.org/pdf/2604.10823)**

> **作者:** Mohamed Ehab; Ali Hamdi
>
> **摘要:** Plant seedling segmentation supports automated phenotyping in precision agriculture. Standard segmentation models face difficulties due to intricate background images and fine structures in leaves. We introduce UGDA-Net (Uncertainty-Guided Dual Attention Network with Entropy-Weighted Loss and Deep Supervision). Three novel components make up UGDA-Net. The first component is Uncertainty-Guided Dual Attention (UGDA). UGDA uses channel variance to modulate feature maps. The second component is an entropy-weighted hybrid loss function. This loss function focuses on high-uncertainty boundary pixels. The third component employs deep supervision for intermediate encoder layers. We performed a comprehensive systematic ablation study. This study focuses on two widely-used architectures, U-Net and LinkNet. It analyzes five incremental configurations: Baseline, Loss-only, Attention-only, Deep Supervision, and UGDA-Net. We trained UGDA-net using a high-resolution plant seedling image dataset containing 432 images. We demonstrate improved segmentation performance and accuracy. With an increase in Dice coefficient of 9.3% above baseline. LinkNet's variance is 13.2% above baseline. Overlays that are qualitative in nature show the reduced false positives at the leaf boundary. Uncertainty heatmaps are consistent with the complex morphology. UGDA-Net aids in the segmentation of delicate structures in plants and provides a high-def solution. The results showed that uncertainty-guided attention and uncertainty-weighted loss are two complementing systems.
>
---
#### [new 272] I Walk the Line: Examining the Role of Gestalt Continuity in Object Binding for Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉Transformer中物体绑定机制，探讨是否依赖格式塔连续性原则。任务属于计算机视觉中的对象识别与特征整合，解决模型如何将图像部分组合成整体对象的问题。工作包括实验验证、注意力头分析及消融实验。**

- **链接: [https://arxiv.org/pdf/2604.09942](https://arxiv.org/pdf/2604.09942)**

> **作者:** Alexa R. Tartaglini; Michael A. Lepori
>
> **摘要:** Object binding is a foundational process in visual cognition, during which low-level perceptual features are joined into object representations. Binding has been considered a fundamental challenge for neural networks, and a major milestone on the way to artificial models with flexible visual intelligence. Recently, several investigations have demonstrated evidence that binding mechanisms emerge in pretrained vision models, enabling them to associate portions of an image that contain an object. The question remains: how are these models binding objects together? In this work, we investigate whether vision models rely on the principle of Gestalt continuity to perform object binding, over and above other principles like similarity and proximity. Using synthetic datasets, we demonstrate that binding probes are sensitive to continuity across a wide range of pretrained vision transformers. Next, we uncover particular attention heads that track continuity, and show that these heads generalize across datasets. Finally, we ablate these attention heads, and show that they often contribute to producing representations that encode object binding.
>
---
#### [new 273] MedLVR: Latent Visual Reasoning for Reliable Medical Visual Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗视觉问答任务，旨在解决医学VQA中视觉证据保留不足的问题。提出MedLVR框架，通过潜在视觉推理提升答案可靠性。**

- **链接: [https://arxiv.org/pdf/2604.09757](https://arxiv.org/pdf/2604.09757)**

> **作者:** Suyang Xi; Songtao Hu; Yuxiang Lai; Wangyun Dan; Yaqi Liu; Shansong Wang; Xiaofeng Yang
>
> **摘要:** Medical vision--language models (VLMs) have shown strong potential for medical visual question answering (VQA), yet their reasoning remains largely text-centric: images are encoded once as static context, and subsequent inference is dominated by language. This paradigm is fundamentally limited in clinical scenarios, where accurate answers often depend on subtle, localized visual evidence that cannot be reliably preserved in static embeddings. We propose \textsc{MedLVR}, a latent visual reasoning framework that introduces an explicit visual evidence state into autoregressive decoding. Instead of relying solely on text-based intermediate reasoning, \textsc{MedLVR} interleaves a short latent reasoning segment within the decoder by reusing hidden states as continuous latent steps, enabling iterative preservation and refinement of query-relevant visual evidence before answer generation. To support effective visual supervision, we adopt a two-stage training strategy: region of interest (ROI)-supervised fine-tuning aligns latent states with clinically relevant image evidence, and Visual-Latent Policy Optimization (VLPO) further optimizes latent reasoning and answer generation under outcome-level rewards. Experiments on OmniMedVQA and five external medical VQA benchmarks show that \textsc{MedLVR} consistently outperforms recent reasoning baselines and improves the average score over the Qwen2.5-VL-7B backbone from 48.3\% to 53.4\%. These results show that latent visual reasoning provides an effective mechanism for preserving diagnostically relevant visual evidence and improving the reliability of medical VQA.
>
---
#### [new 274] Who Handles Orientation? Investigating Invariance in Feature Matching
- **分类: cs.CV**

- **简介: 该论文属于图像匹配任务，解决大平面旋转下的特征匹配问题。通过实验研究旋转不变性在描述子和匹配器中的应用，发现早期引入可提升效率且不影响性能。**

- **链接: [https://arxiv.org/pdf/2604.11809](https://arxiv.org/pdf/2604.11809)**

> **作者:** David Nordström; Johan Edstedt; Fredrik Kahl; Georg Bökman
>
> **摘要:** Finding matching keypoints between images is a core problem in 3D computer vision. However, modern matchers struggle with large in-plane rotations. A straightforward mitigation is to learn rotation invariance via data augmentation. However, it remains unclear at which stage rotation invariance should be incorporated. In this paper, we study this in the context of a modern sparse matching pipeline. We perform extensive experiments by training on a large collection of 3D vision datasets and evaluating on popular image matching benchmarks. Surprisingly, we find that incorporating rotation invariance already in the descriptor yields similar performance to handling it in the matcher. However, rotation invariance is achieved earlier in the matcher when it is learned in the descriptor, allowing for a faster rotation-invariant matcher. Further, we find that enforcing rotation invariance does not hurt upright performance when trained at scale. Finally, we study the emergence of rotation invariance through scale and find that increasing the training data size substantially improves generalization to rotated images. We release two matchers robust to in-plane rotations that achieve state-of-the-art performance on e.g. multi-modal (WxBS), extreme (HardMatch), and satellite image matching (SatAst). Code is available at this https URL.
>
---
#### [new 275] RobustMedSAM: Degradation-Resilient Medical Image Segmentation via Robust Foundation Model Adaptation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决模型在图像退化下的可靠性问题。通过融合不同预训练模块，提升分割鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.09814](https://arxiv.org/pdf/2604.09814)**

> **作者:** Jieru Li; Matthew Chen; Micky C. Nnamdi; J. Ben Tamo; Benoit L. Marteau; May D. Wang
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Medical image segmentation models built on Segment Anything Model (SAM) achieve strong performance on clean benchmarks, yet their reliability often degrades under realistic image corruptions such as noise, blur, motion artifacts, and modality-specific distortions. Existing approaches address either medical-domain adaptation or corruption robustness, but not both jointly. In SAM, we find that these capabilities are concentrated in complementary modules: the image encoder preserves medical priors, while the mask decoder governs corruption robustness. Motivated by this observation, we propose RobustMedSAM, which adopts module-wise checkpoint fusion by initializing the image encoder from MedSAM and the mask decoder from RobustSAM under a shared ViT-B architecture. We then fine-tune only the mask decoder on 35 medical datasets from MedSegBench, spanning six imaging modalities and 12 corruption types, while freezing the remaining components to preserve pretrained medical representations. We additionally investigate an SVD-based parameter-efficient variant for limited encoder adaptation. Experiments on both in-distribution and out-of-distribution benchmarks show that RobustMedSAM improves degraded-image Dice from 0.613 to 0.719 (+0.106) over SAM, demonstrating that structured fusion of complementary pretrained models is an effective and practical approach for robust medical image segmentation.
>
---
#### [new 276] FreeScale: Scaling 3D Scenes via Certainty-Aware Free-View Generation
- **分类: cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决NVS模型训练数据不足的问题。通过改进的自由视角采样策略，提升重建场景质量，增强模型性能。**

- **链接: [https://arxiv.org/pdf/2604.10512](https://arxiv.org/pdf/2604.10512)**

> **作者:** Chenhan Jiang; Yu Chen; Qingwen Zhang; Jifei Song; Songcen Xu; Dit-Yan Yeung; Jiankang Deng
>
> **备注:** CVPR2026
>
> **摘要:** The development of generalizable Novel View Synthesis (NVS) models is critically limited by the scarcity of large-scale training data featuring diverse and precise camera trajectories. While real-world captures are photorealistic, they are typically sparse and discrete. Conversely, synthetic data scales but suffers from a domain gap and often lacks realistic semantics. We introduce FreeScale, a novel framework that leverages the power of scene reconstruction to transform limited real-world image sequences into a scalable source of high-quality training data. Our key insight is that an imperfect reconstructed scene serves as a rich geometric proxy, but naively sampling from it amplifies artifacts. To this end, we propose a certainty-aware free-view sampling strategy identifying novel viewpoints that are both semantically meaningful and minimally affected by reconstruction errors. We demonstrate FreeScale's effectiveness by scaling up the training of feedforward NVS models, achieving a notable gain of 2.7 dB in PSNR on challenging out-of-distribution benchmarks. Furthermore, we show that the generated data can actively enhance per-scene 3D Gaussian Splatting optimization, leading to consistent improvements across multiple datasets. Our work provides a practical and powerful data generation engine to overcome a fundamental bottleneck in 3D vision. Project page: this https URL.
>
---
#### [new 277] Face Density as a Proxy for Data Complexity: Quantifying the Hardness of Instance Count
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究目标检测任务，探讨实例密度对数据复杂性的影响。通过控制实验，发现密度增加导致模型性能下降，且低密度训练的模型无法泛化到高密度场景。**

- **链接: [https://arxiv.org/pdf/2604.09689](https://arxiv.org/pdf/2604.09689)**

> **作者:** Abolfazl Mohammadi-Seif; Ricardo Baeza-Yates
>
> **备注:** Accepted for publication at IEEE CAI 2026
>
> **摘要:** Machine learning progress has historically prioritized model-centric innovations, yet achievable performance is frequently capped by the intrinsic complexity of the data itself. In this work, we isolate and quantify the impact of instance density (measured by face count) as a primary driver of data complexity. Rather than simply observing that ``crowded scenes are harder,'' we rigorously control for class imbalance to measure the precise degradation caused by density alone. Controlled experiments on the WIDER FACE and Open Images datasets, restricted to exactly 1 to 18 faces per image with perfectly balanced sampling, reveal that model performance degrades monotonically with increasing face count. This trend holds across classification, regression, and detection paradigms, even when models are fully exposed to the entire density range. Furthermore, we demonstrate that models trained on low-density regimes fail to generalize to higher densities, exhibiting a systematic under-counting bias, with error rates increasing by up to 4.6x, which suggests density acts as a domain shift. These findings establish instance density as an intrinsic, quantifiable dimension of data hardness and motivate specific interventions in curriculum learning and density-stratified evaluation.
>
---
#### [new 278] Test-time Scaling over Perception: Resolving the Grounding Paradox in Thinking with Images
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，解决视觉推理中的“定位悖论”问题。提出TTSP框架，通过扩展感知过程提升推理鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.11025](https://arxiv.org/pdf/2604.11025)**

> **作者:** Zheng Jiang; Yiming Chen; Nan He; Jiahui Chen; Chaoyang Li; Houde Qian; Lifeng Sun
>
> **摘要:** Recent multimodal large language models (MLLMs) have begun to support Thinking with Images by invoking visual tools such as zooming and cropping during inference. Yet these systems remain brittle in fine-grained visual reasoning because they must decide where to look before they have access to the evidence needed to make that decision correctly. We identify this circular dependency as the Grounding Paradox. To address it, we propose Test-Time Scaling over Perception (TTSP), a framework that treats perception itself as a scalable inference process. TTSP generates multiple exploratory perception traces, filters unreliable traces using entropy-based confidence estimation, distills validated observations into structured knowledge, and iteratively refines subsequent exploration toward unresolved uncertainty. Extensive experiments on high-resolution and general multimodal reasoning benchmarks show that TTSP consistently outperforms strong baselines across backbone sizes, while also exhibiting favorable scalability and token efficiency. Our results suggest that scaling perception at test time is a promising direction for robust multimodal reasoning under perceptual uncertainty.
>
---
#### [new 279] LoViF 2026 The First Challenge on Weather Removal in Videos
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文介绍LoViF 2026视频去雾挑战，旨在解决恶劣天气下视频恢复问题，通过构建数据集推动视觉真实且时间一致的视频修复技术。**

- **链接: [https://arxiv.org/pdf/2604.10655](https://arxiv.org/pdf/2604.10655)**

> **作者:** Chenghao Qian
>
> **备注:** CVPR Workshop Challenge Report
>
> **摘要:** This paper presents a review of the LoViF 2026 Challenge on Weather Removal in Videos. The challenge encourages the development of methods for restoring clean videos from inputs degraded by adverse weather conditions such as rain and snow, with an emphasis on achieving visually plausible and temporally consistent results while preserving scene structure and motion dynamics. To support this task, we introduce a new short-form WRV dataset tailored for video weather removal. It consists of 18 videos 1,216 synthesized frames paired with 1,216 real-world ground-truth frames at a resolution of 832 x 480, and is split into training, validation, and test sets with a ratio of 1:1:1. The goal of this challenge is to advance robust and realistic video restoration under real-world weather conditions, with evaluation protocols that jointly consider fidelity and perceptual quality. The challenge attracted 37 participants and received 5 valid final submissions with corresponding fact sheets, contributing to progress in weather removal for videos. The project is publicly available at this https URL.
>
---
#### [new 280] Assessing Privacy Preservation and Utility in Online Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于隐私保护任务，旨在解决在线视觉语言模型中个人身份信息泄露问题，通过提出方法在保护隐私的同时保持图像的可用性。**

- **链接: [https://arxiv.org/pdf/2604.09695](https://arxiv.org/pdf/2604.09695)**

> **作者:** Karmesh Siddharam Chaudhari; Youxiang Zhu; Amy Feng; Xiaohui Liang; Honggang Zhang
>
> **备注:** Accepted for publication in IEEE ICC 2026. \c{opyright} IEEE. Personal use of this material is permitted. The final version will appear in IEEE Xplore
>
> **摘要:** The increasing use of Online Vision Language Models (OVLMs) for processing images has introduced significant privacy risks, as individuals frequently upload images for various utilities, unaware of the potential for privacy violations. Images contain relationships that relate to Personally Identifiable Information (PII), where even seemingly harmless details can indirectly reveal sensitive information through surrounding clues. This paper explores the critical issue of PII disclosure in images uploaded to OVLMs and its implications for user privacy. We investigate how the extraction of contextual relationships from images can lead to direct (explicit) or indirect (implicit) exposure of PII, significantly compromising personal privacy. Furthermore, we propose methods to protect privacy while preserving the intended utility of the images in Vision Language Model (VLM)-based applications. Our evaluation demonstrates the efficacy of these techniques, highlighting the delicate balance between maintaining utility and protecting privacy in online image processing environments. Index Terms-Personally Identifiable Information (PII), Privacy, Utility, privacy concerns, sensitive information
>
---
#### [new 281] Immunizing 3D Gaussian Generative Models Against Unauthorized Fine-Tuning via Attribute-Space Traps
- **分类: cs.CV**

- **简介: 该论文属于3D生成模型安全防护任务，旨在防止未经授权的微调攻击。通过引入属性空间陷阱，破坏模型结构完整性，同时保持授权任务性能。**

- **链接: [https://arxiv.org/pdf/2604.09688](https://arxiv.org/pdf/2604.09688)**

> **作者:** Jianwei Zhang; Sihan Cao; Chaoning Zhang; Ziming Hong; Jiaxin Huang; Pengcheng Zheng; Caiyan Qin; Wei Dong; Yang Yang; Tongliang Liu
>
> **摘要:** Recent large-scale generative models enable high-quality 3D synthesis. However, the public accessibility of pre-trained weights introduces a critical vulnerability. Adversaries can fine-tune these models to steal specialized knowledge acquired during pre-training, leading to intellectual property infringement. Unlike defenses for 2D images and language models, 3D generators require specialized protection due to their explicit Gaussian representations, which expose fundamental structural parameters directly to gradient-based optimization. We propose GaussLock, the first approach designed to defend 3D generative models against fine-tuning attacks. GaussLock is a lightweight parameter-space immunization framework that integrates authorized distillation with attribute-aware trap losses targeting position, scale, rotation, opacity, and color. Specifically, these traps systematically collapse spatial distributions, distort geometric shapes, align rotational axes, and suppress primitive visibility to fundamentally destroy structural integrity. By jointly optimizing these dual objectives, the distillation process preserves fidelity on authorized tasks while the embedded traps actively disrupt unauthorized reconstructions. Experiments on large-scale Gaussian models demonstrate that GaussLock effectively neutralizes unauthorized fine-tuning attacks. It substantially degrades the quality of unauthorized reconstructions, evidenced by significantly higher LPIPS and lower PSNR, while effectively maintaining performance on authorized fine-tuning.
>
---
#### [new 282] PACO: Proxy-Task Alignment and Online Calibration for On-the-Fly Category Discovery
- **分类: cs.CV**

- **简介: 该论文提出PACO，解决在线流式类别发现问题，通过动态决策和阈值校准，提升模型在未知类别识别与新类别创建上的性能。**

- **链接: [https://arxiv.org/pdf/2604.11484](https://arxiv.org/pdf/2604.11484)**

> **作者:** Weidong Tang; Bohan Zhang; Zhixiang Chi; ZiZhang Wu; Yang Wang; Yanan Wu
>
> **备注:** 16 pages, 6 figures, 7 tables, 1 algorithm
>
> **摘要:** On-the-Fly Category Discovery (OCD) requires a model, trained on an offline support set, to recognize known classes while discovering new ones from an online streaming sequence. Existing methods focus heavily on offline training. They aim to learn discriminative representations on the support set so that novel classes can be separated at test time. However, their discovery mechanism at inference is typically reduced to a single threshold. We argue that this paradigm is fundamentally flawed as OCD is not a static classification problem, but a dynamic process. The model must continuously decide 1) whether a sample belongs to a known class, 2) matches an existing novel category, or 3) should initiate a new one. Moreover, prior methods treat the support set as fixed knowledge. They do not update their decision boundaries as new evidence arrives during inference. This leads to unstable and inconsistent category formation. Our experiments confirm these issues. With properly calibrated and adaptive thresholds, substantial improvements can be achieved, even without changing the representation. Motivated by this, we propose PACO, a support-set-calibrated, tree-structured online decision framework. The framework models inference as a sequence of hierarchical decisions, including known-class routing, birth-aware novel assignment, and attach-versus-create operations over a dynamic prototype memory. Furthermore, we simulate the proxy discovery process to initialize the thresholds during offline training to align with inference. Thresholds are continuously updated during inference using mature novel prototypes. Importantly, PACO requires no heavy training and no dataset-specific tuning. It can be directly integrated into existing OCD pipelines as an inference-time module. Extensive experiments show significant improvements over SOTA baselines across seven benchmarks.
>
---
#### [new 283] Class-Adaptive Cooperative Perception for Multi-Class LiDAR-based 3D Object Detection in V2X Systems
- **分类: cs.CV; cs.AI; cs.ET**

- **简介: 该论文属于多类LiDAR 3D目标检测任务，旨在解决协同感知中统一融合策略导致的多类目标检测不平衡问题。通过引入类别自适应的融合机制和优化损失权重，提升多类目标检测性能。**

- **链接: [https://arxiv.org/pdf/2604.10305](https://arxiv.org/pdf/2604.10305)**

> **作者:** Blessing Agyei Kyem; Joshua Kofi Asamoah; Armstrong Aboah
>
> **备注:** 16 pages, 7 figures, 4 tables
>
> **摘要:** Cooperative perception allows connected vehicles and roadside infrastructure to share sensor observations, creating a fused scene representation beyond the capability of any single platform. However, most cooperative 3D object detectors use a uniform fusion strategy for all object classes, which limits their ability to handle the different geometric structures and point-sampling patterns of small and large objects. This problem is further reinforced by narrow evaluation protocols that often emphasize a single dominant class or only a few cooperation settings, leaving robust multi-class detection across diverse vehicle-to-everything interactions insufficiently explored. To address this gap, we propose a class-adaptive cooperative perception architecture for multi-class 3D object detection from LiDAR data. The model integrates four components: multi-scale window attention with learned scale routing for spatially adaptive feature extraction, a class-specific fusion module that separates small and large objects into attentive fusion pathways, bird's-eye-view enhancement through parallel dilated convolution and channel recalibration for richer contextual representation, and class-balanced objective weighting to reduce bias toward frequent categories. Experiments on the V2X-Real benchmark cover vehicle-centric, infrastructure-centric, vehicle-to-vehicle, infrastructure-to-infrastructure, and vehicle-to-infrastructure settings under identical backbone and training configurations. The proposed method consistently improves mean detection performance over strong intermediate-fusion baselines, with the largest gains on trucks, clear improvements on pedestrians, and competitive results on cars. These results show that aligning feature extraction and fusion with class-dependent geometry and point density leads to more balanced cooperative perception in realistic vehicle-to-everything deployments.
>
---
#### [new 284] HO-Flow: Generalizable Hand-Object Interaction Generation with Latent Flow Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于手-物体交互生成任务，旨在解决3D手-物体运动序列生成中的物理合理性与时间一致性问题。工作包括引入交互感知的变分自编码器和掩码流匹配模型，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2604.10836](https://arxiv.org/pdf/2604.10836)**

> **作者:** Zerui Chen; Rolandos Alexandros Potamias; Shizhe Chen; Jiankang Deng; Cordelia Schmid; Stefanos Zafeiriou
>
> **备注:** Project Page: this https URL
>
> **摘要:** Generating realistic 3D hand-object interactions (HOI) is a fundamental challenge in computer vision and robotics, requiring both temporal coherence and high-fidelity physical plausibility. Existing methods remain limited in their ability to learn expressive motion representations for generation and perform temporal reasoning. In this paper, we present HO-Flow, a framework for synthesizing realistic hand-object motion sequences from texts and canoncial 3D objects. HO-Flow first employs an interaction-aware variational autoencoder to encode sequences of hand and object motions into a unified latent manifold by incorporating hand and object kinematics, enabling the representation to capture rich interaction dynamics. It then leverages a masked flow matching model that combines auto-regressive temporal reasoning with continuous latent generation, improving temporal coherence. To further enhance generalization, HO-Flow predicts object motions relative to the initial frame, enabling effective pre-training on large-scale synthetic data. Experiments on the GRAB, OakInk, and DexYCB benchmarks demonstrate that HO-Flow achieves state-of-the-art performance in both physical plausibility and motion diversity for interaction motion synthesis.
>
---
#### [new 285] Scene Change Detection with Vision-Language Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于场景变化检测任务，旨在解决城市环境中因光照、季节等因素导致的检测困难。提出LangSCD框架，结合视觉与语言信息提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.11402](https://arxiv.org/pdf/2604.11402)**

> **作者:** Diwei Sheng; Vijayraj Gohil; Satyam Gaba; Zihan Liu; Giles Hamilton-Fletcher; John-Ross Rizzo; Yongqing Liang; Chen Feng
>
> **摘要:** Scene change detection (SCD) is crucial for urban monitoring and navigation but remains challenging in real-world environments due to lighting variations, seasonal shifts, viewpoint differences, and complex urban layouts. Existing methods rely primarily on low-level visual features, limiting their ability to accurately identify changed objects amid the visual complexity of urban scenes. In this paper, we propose LangSCD, a vision-language framework for scene change detection that overcomes this single-modal limitation by incorporating semantic reasoning through language. Our approach introduces a modular language component that leverages vision-language models (VLMs) to generate textual descriptions of scene changes, which are fused with visual features through a cross-modal feature enhancer. We further introduce a geometric-semantic matching module that refines the predicted masks by enforcing semantic consistency and spatial completeness. Existing real-world scene change detection benchmarks provide only binary change annotations, which are insufficient for downstream applications requiring fine-grained understanding of scene dynamics. To address this limitation, we introduce NYC-CD, a large-scale dataset of 8,122 real-world image pairs collected in New York City with multiclass change annotations generated through a semi-automatic pipeline. Extensive experiments across multiple street-view benchmarks demonstrate that our language and matching modules consistently improve existing change-detection architectures, achieving state-of-the-art performance and highlighting the value of integrating linguistic reasoning with visual representations for robust scene change detection.
>
---
#### [new 286] Agentic Video Generation: From Text to Executable Event Graphs via Tool-Constrained LLM Planning
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决现有系统生成结果语义不可靠的问题。通过构建可执行的事件图结构，结合自然语言规划与程序化约束，提升视频生成的物理有效性和语义一致性。**

- **链接: [https://arxiv.org/pdf/2604.10383](https://arxiv.org/pdf/2604.10383)**

> **作者:** Nicolae Cudlenco; Mihai Masala; Marius Leordeanu
>
> **摘要:** Existing multi-agent video generation systems use LLM agents to orchestrate neural video generators, producing visually impressive but semantically unreliable outputs with no ground truth annotations. We present an agentic system that inverts this paradigm: instead of generating pixels, the LLM constructs a formal Graph of Events in Space and Time (GEST) -- a structured specification of actors, actions, objects, and temporal constraints -- which is then executed deterministically in a 3D game engine. A staged LLM refinement pipeline fails entirely at this task (0 of 50 attempts produce an executable specification), motivating a fundamentally different architecture based on a separation of concerns: the LLM handles narrative planning through natural language reasoning, while a programmatic state backend enforces all simulator constraints through validated tool calls, guaranteeing that every generated specification is executable by construction. The system uses a hierarchical two-agent architecture -- a Director that plans the story and a Scene Builder that constructs individual scenes through a round-based state machine -- with dedicated Relation Subagents that populate the logical and semantic edge types of the GEST formalism that procedural generation leaves empty, making this the first approach to exercise the full expressive capacity of the representation. We evaluate in two stages: autonomous generation against procedural baselines via a 3-model LLM jury, where agentic narratives win 79% of text and 74% of video comparisons; and seeded generation where the same text is given to our system, VEO 3.1, and WAN 2.2, with human annotations showing engine-generated videos substantially outperform neural generators on physical validity (58% vs 25% and 20%) and semantic alignment (3.75/5 vs 2.33 and 1.50).
>
---
#### [new 287] BEM: Training-Free Background Embedding Memory for False-Positive Suppression in Real-Time Fixed-Background Camera
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决真实场景中因分布差异导致的误检问题。提出BEM模块，在不训练的情况下利用背景信息减少误报，保持实时性。**

- **链接: [https://arxiv.org/pdf/2604.11714](https://arxiv.org/pdf/2604.11714)**

> **作者:** Junwoo Park; Jangho Lee; Sunho Lim
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Pretrained detectors perform well on benchmarks but often suffer performance degradation in real-world deployments due to distribution gaps between training data and target environments. COCO-like benchmarks emphasize category diversity rather than instance density, causing detectors trained under per-class sparsity to struggle in dense, single- or few-class scenes such as surveillance and traffic monitoring. In fixed-camera environments, the quasi-static background provides a stable, label-free prior that can be exploited at inference to suppress spurious detections. To address the issue, we propose Background Embedding Memory (BEM), a lightweight, training-free, weight-frozen module that can be attached to pretrained detectors during inference. BEM estimates clean background embeddings, maintains a prototype memory, and re-scores detection logits with an inverse-similarity, rank-weighted penalty, effectively reducing false positives while maintaining recall. Empirically, background-frame cosine similarity correlates negatively with object count and positively with Precision-Confidence AUC (P-AUC), motivating its use as a training-free control signal. Across YOLO and RT-DETR families on LLVIP and simulated surveillance streams, BEM consistently reduces false positives while preserving real-time performance. Our code is available at this https URL
>
---
#### [new 288] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain
- **分类: cs.CV**

- **简介: 该论文提出DiningBench，一个用于食品领域视觉语言模型的多视角基准，解决细粒度分类、营养估计和视觉问答任务中的性能不足问题。**

- **链接: [https://arxiv.org/pdf/2604.10425](https://arxiv.org/pdf/2604.10425)**

> **作者:** Song Jin; Juntian Zhang; Xun Zhang; Zeying Tian; Fei Jiang; Guojun Yin; Wei Lin; Yong Liu; Rui Yan
>
> **备注:** ACL 2026 Main
>
> **摘要:** Recent advancements in Vision-Language Models (VLMs) have revolutionized general visual understanding. However, their application in the food domain remains constrained by benchmarks that rely on coarse-grained categories, single-view imagery, and inaccurate metadata. To bridge this gap, we introduce DiningBench, a hierarchical, multi-view benchmark designed to evaluate VLMs across three levels of cognitive complexity: Fine-Grained Classification, Nutrition Estimation, and Visual Question Answering. Unlike previous datasets, DiningBench comprises 3,021 distinct dishes with an average of 5.27 images per entry, incorporating fine-grained "hard" negatives from identical menus and rigorous, verification-based nutritional data. We conduct an extensive evaluation of 29 state-of-the-art open-source and proprietary models. Our experiments reveal that while current VLMs excel at general reasoning, they struggle significantly with fine-grained visual discrimination and precise nutritional reasoning. Furthermore, we systematically investigate the impact of multi-view inputs and Chain-of-Thought reasoning, identifying five primary failure modes. DiningBench serves as a challenging testbed to drive the next generation of food-centric VLM research. All codes are released in this https URL.
>
---
#### [new 289] Uncertainty-quantified Pulse Signal Recovery from Facial Video using Regularized Stochastic Interpolants
- **分类: cs.CV**

- **简介: 该论文属于iPPG任务，解决BVP信号恢复中的不确定性问题。通过构建随机插值模型，实现测试时的后验采样与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2604.10777](https://arxiv.org/pdf/2604.10777)**

> **作者:** Vineet R. Shenoy; Cheng Peng; Rama Chellappa; Yu Sun
>
> **摘要:** Imaging Photoplethysmography (iPPG), an optical procedure which recovers a human's blood volume pulse (BVP) waveform using pixel readout from a camera, is an exciting research field with many researchers performing clinical studies of iPPG algorithms. While current algorithms to solve the iPPG task have shown outstanding performance on benchmark datasets, no state-of-the art algorithms, to the best of our knowledge, performs test-time sampling of solution space, precluding an uncertainty analysis that is critical for clinical applications. We address this deficiency though a new paradigm named Regularized Interpolants with Stochasticity for iPPG (RIS-iPPG). Modeling iPPG recovery as an inverse problem, we build probability paths that evolve the camera pixel distribution to the ground-truth signal distribution by predicting the instantaneous flow and score vectors of a time-dependent stochastic process; and at test-time, we sample the posterior distribution of the correct BVP waveform given the camera pixel intensity measurements by solving a stochastic differential equation. Given that physiological changes are slowly varying, we show that iPPG recovery can be improved through regularization that maximizes the correlation between the residual flow vector predictions of two adjacent time windows. Experimental results on three datasets show that RIS-iPPG provides superior reconstruction quality and uncertainty estimates of the reconstruction, a critical tool for the widespread adoption of iPPG algorithms in clinical and consumer settings.
>
---
#### [new 290] You Only Judge Once: Multi-response Reward Modeling in a Single Forward Pass
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种多响应奖励模型，用于高效比较多个生成回复，解决传统模型逐个评估效率低的问题。通过单次前向传播实现多响应评分，提升速度与性能。**

- **链接: [https://arxiv.org/pdf/2604.10966](https://arxiv.org/pdf/2604.10966)**

> **作者:** Yinuo Yang; Zixian Ma; Manasi Ganti; Jieyu Zhang; Ranjay Krishna
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** We present a discriminative multimodal reward model that scores all candidate responses in a single forward pass. Conventional discriminative reward models evaluate each response independently, requiring multiple forward passes, one for each potential response. Our approach concatenates multiple responses with separator tokens and applies cross-entropy over their scalar scores, enabling direct comparative reasoning and efficient $N$-way preference learning. The multi-response design also yields up to $N\times$ wall-clock speedup and FLOPs reduction over conventional single-response scoring. To enable $N$-way reward evaluation beyond existing pairwise benchmarks, we construct two new benchmarks: (1) MR$^2$Bench-Image contains human-annotated rankings over responses from 8 diverse models; (2) MR$^2$Bench-Video is a large-scale video-based reward benchmark derived from 94K crowdsourced pairwise human judgments over video question-answering spanning 19 models, denoised via preference graph ensemble. Both benchmarks provide 4-response evaluation variants sampled from the full rankings. Built on a 4B vision-language backbone with LoRA fine-tuning and a lightweight MLP value head, our model achieves state-of-the-art results on six multimodal reward benchmarks, including MR$^2$Bench-Image, MR$^2$Bench-Video, and four other existing benchmarks. Our model outperforms existing larger generative and discriminative reward models. We further demonstrate that our reward model, when used in reinforcement learning with GRPO, produces improved policy models that maintain performance across standard multimodal benchmarks while substantially improving open-ended generation quality, outperforming a single-response discriminative reward model (RM) baseline by a large margin in both training stability and open-ended generation quality.
>
---
#### [new 291] Orthogonal Quadratic Complements for Vision Transformer Feed-Forward Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉Transformer任务，旨在提升模型精度。针对现有方法混淆二阶交互与冗余的问题，提出OQC方法，通过正交投影增强特征表达，有效提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.09709](https://arxiv.org/pdf/2604.09709)**

> **作者:** Wang Zixian
>
> **摘要:** Recent bilinear feed-forward replacements for vision transformers can substantially improve accuracy, but they often conflate two effects: stronger second-order interactions and increased redundancy relative to the main branch. We study a complementary design principle in which auxiliary quadratic features contribute only information not already captured by the dominant hidden representation. To this end, we propose Orthogonal Quadratic Complements (OQC), which construct a low-rank quadratic auxiliary branch and explicitly project it onto the orthogonal complement of the main branch before injection. We further study an efficient low-rank realization (OQC-LR) and gated extensions (OQC-static and OQC-dynamic). Under a parameter-matched Deep-ViT and CIFAR-100 protocol with a fixed penultimate residual readout, full OQC improves an AFBO baseline from 64.25 +/- 0.22 to 65.59 +/- 0.22, while OQC-LR reaches 65.52 +/- 0.25 with a substantially better speed-accuracy tradeoff. On TinyImageNet, the gated extension OQC-dynamic achieves 51.88 +/- 0.32, improving the baseline (50.45 +/- 0.21) by 1.43 points and outperforming all ungated variants. Mechanism analyses show near-zero post-projection auxiliary-main overlap together with improved representation geometry and class separation. The full family, including both ungated and gated variants, generalizes consistently across both datasets.
>
---
#### [new 292] HuiYanEarth-SAR: A Foundation Model for High-Fidelity and Low-Cost Global Remote Sensing Imagery Generation
- **分类: cs.CV**

- **简介: 该论文属于SAR图像生成任务，旨在解决高保真与低成本全球遥感图像生成问题。提出HuiYanEarth-SAR模型，结合地理先验与散射机制，实现基于坐标生成高质量SAR图像。**

- **链接: [https://arxiv.org/pdf/2604.11444](https://arxiv.org/pdf/2604.11444)**

> **作者:** Yongxiang Liu; Jie Zhou; Yafei Song; Tianpeng Liu; Li Liu
>
> **摘要:** Synthetic Aperture Radar (SAR) imagery generation is essential for deepening the study of scattering mechanisms, establishing trustworthy electromagnetic scene models, and fundamentally alleviating the data scarcity bottleneck that constrains development in this field. However, existing methods find it difficult to simultaneously ensure high fidelity in both global geospatial semantics and microscopic scattering mechanisms, resulting in severe challenges for global generation. To address this, we propose \textbf{HuiYanEarth-SAR}, the first foundational SAR imagery generation model based on AlphaEarth and integrated scattering mechanisms. By injecting geospatial priors to control macroscopic structures and utilizing implicit scattering characteristic modeling to ensure the authenticity of microscopic textures, we achieve the capability of generating high-fidelity SAR images for global locations solely based on geographic coordinates. This study not only constructs an efficient SAR scene simulator but also establishes a bridge connecting geography, scatter mechanism, and artificial intelligence from a methodological standpoint. It advances SAR research by expanding the paradigm from perception and understanding to simulation and creation, providing key technical support for constructing a high-confidence digital twin of the Earth.
>
---
#### [new 293] Multi-modal, multi-scale representation learning for satellite imagery analysis just needs a good ALiBi
- **分类: cs.CV**

- **简介: 该论文属于卫星图像分析任务，解决多尺度、多模态表示学习问题。提出Scale-ALiBi机制，提升模型性能并发布新数据集。**

- **链接: [https://arxiv.org/pdf/2604.10347](https://arxiv.org/pdf/2604.10347)**

> **作者:** Patrick Kage; Pavlos Andreadis
>
> **备注:** Originally appeared at the 4th Space Imaging Workshop at the Georgia Institute of Technology, October 7-9, 2024
>
> **摘要:** Vision foundation models have been shown to be effective at processing satellite imagery into representations fit for downstream tasks, however, creating models which operate over multiple spatial resolutions and modes is challenging. This paper presents Scale-ALiBi, a linear bias transformer attention mechanism with a spatial encoding bias to relationships between image patches at different ground sample distance scales. We provide an implementation of Scale-ALiBi over a dataset of aligned high- and low-resolution optical and low-resolution SAR satellite imagery data using a triple-contrastive and reconstructive architecture, show an improvement on the GEO-Bench benchmark, and release the newly curated dataset publicly.
>
---
#### [new 294] LVSum: A Benchmark for Timestamp-Aware Long Video Summarization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于长视频摘要任务，旨在解决模型在长时间序列中保持时间准确性的难题。提出LVSum基准，评估模型的时间理解能力。**

- **链接: [https://arxiv.org/pdf/2604.10024](https://arxiv.org/pdf/2604.10024)**

> **作者:** Alkesh Patel; Melis Ozyildirim; Ying-Chang Cheng; Ganesh Nagarajan
>
> **备注:** 25 pages, 5 tables, 3 figures
>
> **摘要:** Long video summarization presents significant challenges for current multimodal large language models (MLLMs), particularly in maintaining temporal fidelity over extended durations and producing summaries that are both semantically and temporally grounded. In this work, we present LVSum, a human-annotated benchmark designed specifically for evaluating long video summarization with fine-grained temporal alignment. LVSum comprises diverse long-form videos across 13 domains, each paired with human-generated summaries containing precise temporal references. We conduct a comprehensive evaluation of both proprietary and open-source MLLMs on LVSum, assessing performance using newly introduced LLM-based metrics for content relevance and modality coherence, alongside standard evaluation metrics. Our experiments reveal systematic gaps in temporal understanding among existing MLLMs and offer insights that establish a new foundation for advancing temporal reasoning in long video summarization.
>
---
#### [new 295] Towards Brain MRI Foundation Models for the Clinic: Findings from the FOMO25 Challenge
- **分类: cs.CV**

- **简介: 该论文属于脑部MRI分析任务，旨在解决临床数据异质性和标注成本高的问题。通过自监督学习构建基础模型，并在真实临床数据上验证其泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11679](https://arxiv.org/pdf/2604.11679)**

> **作者:** Asbjørn Munk; Stefano Cerri; Vardan Nersesjan; Christian Hedeager Krag; Jakob Ambsdorf; Pablo Rocamora García; Julia Machnio; Peirong Liu; Suhyun Ahn; Nasrin Akbari; Yasmina Al Khalil; Kimberly Amador; Sina Amirrajab; Tal Arbel; Meritxell Bach Cuadra; Ujjwal Baid; Bhakti Baheti; Jaume Banus; Kamil Barbierik; Christoph Brune; Yansong Bu; Baptiste Callard; Yuhan Chen; Cornelius Crijnen; Corentin Dancette; Peter Drotar; Prasad Dutande; Nils D. Forkert; Saurabh Garg; Jakub Gazda; Matej Gazda; Benoît Gérin; Partha Ghosh; Weikang Gong; Pedro M. Gordaliza; Sam Hashemi; Tobias Heimann; Fucang Jia; Jiexin Jiang; Emily Kaczmarek; Chris Kang; Seung Kwan Kang; Mohammad Khazaei; Julien Khlaut; Petros Koutsouvelis; Jae Sung Lee; Yuchong Li; Mengye Lyu; Mingchen Ma; Anant Madabhushi; Klaus H. Maier-Hein; Pierre Manceron; Andrés Martínez Mora; Moona Mazher; Felix Meister; Nataliia Molchanova; Steven A. Niederer; Leonard Nürnberg; Jinah Park; Abdul Qayyum; Jonas Richiardi; Antoine Saporta; Branislav Setlak; Ning Shen; Justin Szeto; Constantin Ulrich; Puru Vaish; Vibujithan Vigneshwaran; Leroy Volmer; Zihao Wang; Siqi Wei; Anthony Winder; Jelmer M. Wolterink; Maxence Wynen; Chang Yang; Si Young Yie; Mostafa Mehdipour Ghazi; Akshay Pai; Espen Jimenez Solem; Sebastian Nørgaard Llambias; Mikael Boesen; Michael Eriksen Benros; Juan Eugenio Iglesias; Mads Nielsen
>
> **摘要:** Clinical deployment of automated brain MRI analysis faces a fundamental challenge: clinical data is heterogeneous and noisy, and high-quality labels are prohibitively costly to obtain. Self-supervised learning (SSL) can address this by leveraging the vast amounts of unlabeled data produced in clinical workflows to train robust \textit{foundation models} that adapt out-of-domain with minimal supervision. However, the development of foundation models for brain MRI has been limited by small pretraining datasets and in-domain benchmarking focused on high-quality, research-grade data. To address this gap, we organized the FOMO25 challenge as a satellite event at MICCAI 2025. FOMO25 provided participants with a large pretraining dataset, FOMO60K, and evaluated models on data sourced directly from clinical workflows in few-shot and out-of-domain settings. Tasks covered infarct classification, meningioma segmentation, and brain age regression, and considered both models trained on FOMO60K (method track) and any data (open track). Nineteen foundation models from sixteen teams were evaluated using a standardized containerized pipeline. Results show that (a) self-supervised pretraining improves generalization on clinical data under domain shift, with the strongest models trained \textit{out-of-domain} surpassing supervised baselines trained \textit{in-domain}. (b) No single pretraining objective benefits all tasks: MAE favors segmentation, hybrid reconstruction-contrastive objectives favor classification, and (c) strong performance was achieved by small pretrained models, and improvements from scaling model size and training duration did not yield reliable benefits.
>
---
#### [new 296] Biomarker-Based Pretraining for Chagas Disease Screening in Electrocardiograms
- **分类: cs.CV**

- **简介: 该论文属于心电图（ECG）中克鲁斯病筛查任务，解决标签稀缺和噪声问题。通过基于生物标志物的预训练方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.09782](https://arxiv.org/pdf/2604.09782)**

> **作者:** Elias Stenhede; Arian Ranjbar
>
> **摘要:** Chagas disease screening via ECGs is limited by scarce and noisy labels in existing datasets. We propose a biomarker-based pretraining approach, where an ECG feature extractor is first trained to predict percentile-binned blood biomarkers from the MIMIC-IV-ECG dataset. The pretrained model is then fine-tuned on Brazilian datasets for Chagas detection. Our 5-model ensemble, developed by the Ahus AIM team, achieved a challenge score of 0.269 on the hidden test set, ranking 5th in Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025. Source code and the model are shared on GitHub: this http URL
>
---
#### [new 297] Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型任务，旨在解决模型生成内容与视觉信息矛盾的幻觉问题。提出DaID方法，通过双锚点机制提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2604.10071](https://arxiv.org/pdf/2604.10071)**

> **作者:** Yebo Wu; Han Jin; Zhijiang Guo; Li Li
>
> **备注:** Accepted for Findings of ACL 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable reasoning capabilities yet continue to suffer from hallucination, where generated text contradicts visual content. In this paper, we introduce Dual-Anchor Introspective Decoding (DaID), a novel contrastive decoding framework that dynamically calibrates each token generation by mining the model's internal perceptual discrepancies. Specifically, DaID identifies a Spotlight layer to amplify visual factual signals and a Shadow layer to suppress textual inertia. By leveraging visual attention distributions to guide this dual-anchor selection process, our method ensures precise, token-specific adaptation. Experimental results across multiple benchmarks and MLLMs demonstrate that DaID significantly mitigates hallucination while enhancing general reasoning capabilities.
>
---
#### [new 298] Particle Diffusion Matching: Random Walk Correspondence Search for the Alignment of Standard and Ultra-Widefield Fundus Images
- **分类: cs.CV**

- **简介: 该论文属于图像配准任务，解决标准眼底图像与超广角眼底图像对齐问题。提出粒子扩散匹配方法，通过随机游走对应搜索实现精准配准。**

- **链接: [https://arxiv.org/pdf/2604.10085](https://arxiv.org/pdf/2604.10085)**

> **作者:** Kanggeon Lee; Soochahn Lee; Kyoung Mu Lee
>
> **摘要:** We propose a robust alignment technique for Standard Fundus Images (SFIs) and Ultra-Widefield Fundus Images (UWFIs), which are challenging to align due to differences in scale, appearance, and the scarcity of distinctive features. Our method, termed Particle Diffusion Matching (PDM), performs alignment through an iterative Random Walk Correspondence Search (RWCS) guided by a diffusion model. At each iteration, the model estimates displacement vectors for particle points by considering local appearance, the structural distribution of particles, and an estimated global transformation, enabling progressive refinement of correspondences even under difficult conditions. PDM achieves state-of-the-art performance across multiple retinal image alignment benchmarks, showing substantial improvement on a primary dataset of SFI-UWFI pairs and demonstrating its effectiveness in real-world clinical scenarios. By providing accurate and scalable correspondence estimation, PDM overcomes the limitations of existing methods and facilitates the integration of complementary retinal image modalities. This diffusion-guided search strategy offers a new direction for improving downstream supervised learning, disease diagnosis, and multi-modal image analysis in ophthalmology.
>
---
#### [new 299] Camyla: Scaling Autonomous Research in Medical Image Segmentation
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出Camyla系统，解决医学图像分割中的自主研究问题，通过自动化生成模型和论文，提升研究效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.10696](https://arxiv.org/pdf/2604.10696)**

> **作者:** Yifan Gao; Haoyue Li; Feng Yuan; Xin Gao; Weiran Huang; Xiaosong Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** We present Camyla, a system for fully autonomous research within the scientific domain of medical image segmentation. Camyla transforms raw datasets into literature-grounded research proposals, executable experiments, and complete manuscripts without human intervention. Autonomous experimentation over long horizons poses three interrelated challenges: search effort drifts toward unpromising directions, knowledge from earlier trials degrades as context accumulates, and recovery from failures collapses into repetitive incremental fixes. To address these challenges, the system combines three coupled mechanisms: Quality-Weighted Branch Exploration for allocating effort across competing proposals, Layered Reflective Memory for retaining and compressing cross-trial knowledge at multiple granularities, and Divergent Diagnostic Feedback for diversifying recovery after underperforming trials. The system is evaluated on CamylaBench, a contamination-free benchmark of 31 datasets constructed exclusively from 2025 publications, under a strict zero-intervention protocol across two independent runs within a total of 28 days on an 8-GPU cluster. Across the two runs, Camyla generates more than 2,700 novel model implementations and 40 complete manuscripts, and surpasses the strongest per-dataset baseline selected from 14 established architectures, including nnU-Net, on 22 and 18 of 31 datasets under identical training budgets, respectively (union: 24/31). Senior human reviewers score the generated manuscripts at the T1/T2 boundary of contemporary medical imaging journals. Relative to automated baselines, Camyla outperforms AutoML and NAS systems on aggregate segmentation performance and exceeds six open-ended research agents on both task completion and baseline-surpassing frequency. These results suggest that domain-scale autonomous research is achievable in medical image segmentation.
>
---
#### [new 300] GLEaN: A Text-to-image Bias Detection Approach for Public Comprehension
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于文本到图像模型偏见检测任务，旨在解决公众理解模型偏见的问题。提出GLEaN方法，通过生成和分析图像来可视化模型偏见。**

- **链接: [https://arxiv.org/pdf/2604.09923](https://arxiv.org/pdf/2604.09923)**

> **作者:** Bochu Ding; Brinnae Bent; Augustus Wendell
>
> **摘要:** Text-to-image (T2I) models, and their encoded biases, increasingly shape the visual media the public encounters. While researchers have produced a rich body of work on bias measurement, auditing, and mitigation in T2I systems, those methods largely target technical stakeholders, leaving a gap in public legibility. We introduce GLEaN (Generative Likeness Evaluation at N-Scale), a portrait-based explainability pipeline designed to make T2I model biases visually understandable to a broad audience. GLEaN comprises three stages: automated large-scale image generation from identity prompts, facial landmark-based filtering and spatial alignment, and median-pixel composition that distills a model's central tendency into a single representative portrait. The resulting composites require no statistical background to interpret; a viewer can see, at a glance, who a model 'imagines' when prompted with 'a doctor' versus a 'felon.' We demonstrate GLEaN on Stable Diffusion XL across 40 social and occupational identity prompts, producing composites that reproduce documented biases and surface new associations between skin tone and predicted emotion. We find in a between-subjects user study (N = 291) that GLEaN portraits communicate biases as effectively as conventional data tables, but require significantly less viewing time. Because the method relies solely on generated outputs, it can also be replicated on any black-box and closed-weight systems without access to model internals. GLEaN offers a scalable, model-agnostic approach to bias explainability, purpose-built for public comprehension, and is publicly available at this https URL.
>
---
#### [new 301] Audio-Omni: Extending Multi-modal Understanding to Versatile Audio Generation and Editing
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文提出Audio-Omni，解决音频生成与编辑的统一框架问题，整合多模态理解，提升音频处理能力。**

- **链接: [https://arxiv.org/pdf/2604.10708](https://arxiv.org/pdf/2604.10708)**

> **作者:** Zeyue Tian; Binxin Yang; Zhaoyang Liu; Jiexuan Zhang; Ruibin Yuan; Hubery Yin; Qifeng Chen; Chen Li; Jing Lv; Wei Xue; Yike Guo
>
> **摘要:** Recent progress in multimodal models has spurred rapid advances in audio understanding, generation, and editing. However, these capabilities are typically addressed by specialized models, leaving the development of a truly unified framework that can seamlessly integrate all three tasks underexplored. While some pioneering works have explored unifying audio understanding and generation, they often remain confined to specific domains. To address this, we introduce Audio-Omni, the first end-to-end framework to unify generation and editing across general sound, music, and speech domains, with integrated multi-modal understanding capabilities. Our architecture synergizes a frozen Multimodal Large Language Model for high-level reasoning with a trainable Diffusion Transformer for high-fidelity synthesis. To overcome the critical data scarcity in audio editing, we construct AudioEdit, a new large-scale dataset comprising over one million meticulously curated editing pairs. Extensive experiments demonstrate that Audio-Omni achieves state-of-the-art performance across a suite of benchmarks, outperforming prior unified approaches while achieving performance on par with or superior to specialized expert models. Beyond its core capabilities, Audio-Omni exhibits remarkable inherited capabilities, including knowledge-augmented reasoning generation, in-context generation, and zero-shot cross-lingual control for audio generation, highlighting a promising direction toward universal generative audio intelligence. The code, model, and dataset will be publicly released on this https URL.
>
---
#### [new 302] WebForge: Breaking the Realism-Reproducibility-Scalability Trilemma in Browser Agent Benchmark
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出WebForge框架，解决浏览器代理基准中的真实性、可复现性与可扩展性难题。通过自动化流程生成交互式网页环境，构建多维任务基准，用于更全面评估模型能力。**

- **链接: [https://arxiv.org/pdf/2604.10988](https://arxiv.org/pdf/2604.10988)**

> **作者:** Peng Yuan; Yuyang Yin; Yuxuan Cai; Zheng Wei
>
> **备注:** 14 pages, 6 figures, 6 tables, plus 29-page supplementary. Code: this https URL Dataset: this https URL
>
> **摘要:** Existing browser agent benchmarks face a fundamental trilemma: real-website benchmarks lack reproducibility due to content drift, controlled environments sacrifice realism by omitting real-web noise, and both require costly manual curation that limits scalability. We present WebForge, the first fully automated framework that resolves this trilemma through a four-agent pipeline -- Plan, Generate, Refine, and Validate -- that produces interactive, self-contained web environments end-to-end without human annotation. A seven-dimensional difficulty control framework structures task design along navigation depth, visual complexity, reasoning difficulty, and more, enabling systematic capability profiling beyond single aggregate scores. Using WebForge, we construct WebForge-Bench, a benchmark of 934 tasks spanning 7 domains and 3 difficulty levels. Multi-model experiments show that difficulty stratification effectively differentiates model capabilities, while cross-domain analysis exposes capability biases invisible to aggregate metrics. Together, these results confirm that multi-dimensional evaluation reveals distinct capability profiles that a single aggregate score cannot capture. Code and benchmark are publicly available at this https URL.
>
---
#### [new 303] Physics-Informed Synthetic Dataset and Denoising TIE-Reconstructed Phase Maps in Transient Flows Using Deep Learning
- **分类: physics.optics; cs.CV; physics.comp-ph**

- **简介: 该论文属于图像去噪任务，旨在解决TIE相位图中的低频伪影问题。通过生成物理合成数据训练网络，有效提升真实相位图质量。**

- **链接: [https://arxiv.org/pdf/2604.10610](https://arxiv.org/pdf/2604.10610)**

> **作者:** Krishna Rajput; Vipul Gupta; Sudheesh K. Rajput; Yasuhiro Awatsuji
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** High-speed quantitative phase imaging enables non-intrusive visualization of transient compressible gas flows and energetic phenomena. However, phase maps reconstructed via the transport of intensity equation (TIE) suffer from spatially correlated low-frequency artifacts introduced by the inverse Laplacian solver, which obscure meaningful flow structures such as jet plumes, shockwave fronts, and density gradients. Conventional filtering approaches fail because signal and noise occupy overlapping spatial frequency bands, and no paired ground truth exists since every frame represents a physically unique, non-repeatable flow state. We address this by developing a physics-informed synthetic training dataset where clean targets are procedurally generated using physically plausible gas flow morphologies, including compressible jet plumes, turbulent eddy fields, density fronts, periodic air pockets, and expansion fans, and passed through a forward TIE simulation followed by inverse Laplacian reconstruction to produce realistic noisy phase maps. A U-Net-based convolutional denoising network trained solely on this synthetic data is evaluated on real phase maps acquired at 25,000 fps, demonstrating zero-shot generalization to real parallel TIE recordings, with a 13,260% improvement in signal-to-background ratio and 100.8% improvement in jet-region structural sharpness across 20 evaluated frames.
>
---
#### [new 304] Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval
- **分类: cs.IR; cs.CV**

- **简介: 该论文属于古文字 decipherment 任务，旨在解决Oracle Bone Script 解码难题。通过生成合成字典并基于视觉相似性检索，提升未见字符的识别准确率。**

- **链接: [https://arxiv.org/pdf/2604.09668](https://arxiv.org/pdf/2604.09668)**

> **作者:** Yin Wu; Gangjian Zhang; Jiayu Chen; Chang Xu; Yuyu Luo; Nan Tang; Hui Xiong
>
> **备注:** 19 pages, 4 figures. Under review at Nature Machine Intelligence
>
> **摘要:** Understanding humanity's earliest writing systems is crucial for reconstructing civilization's origins, yet many ancient scripts remain undeciphered. Oracle Bone Script (OBS) from China's Shang dynasty exemplifies this challenge: only approximately 1,500 of roughly 4,600 characters have been decoded, and a substantial portion of these 3,000-year-old inscriptions remains only partially understood. Limited by extreme data scarcity, existing computational methods achieve under 3% accuracy on unseen characters -- the core palaeographic challenge. We overcome this by reframing decipherment from classification to dictionary-based retrieval. Using deep learning guided by character evolution principles, we generate a comprehensive synthetic dictionary of plausible OBS variants for modern Chinese characters. Scholars query unknown inscriptions to retrieve visually similar candidates with transparent evidence, replacing algorithmic black boxes with interpretable hypotheses. Our approach achieves 54.3% Top-10 and 86.6% Top-50 accuracy for unseen characters. This scalable, transparent framework accelerates decipherment of a pivotal undeciphered script and establishes a generalizable methodology for AI-assisted archaeological discovery.
>
---
#### [new 305] Evaluating Visual Prompts with Eye-Tracking Data for MLLM-Based Human Activity Recognition
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文属于人活动识别任务，解决高频率眼动数据在MLLM中应用的效率问题，通过视觉提示将数据转化为图像进行有效处理。**

- **链接: [https://arxiv.org/pdf/2604.09585](https://arxiv.org/pdf/2604.09585)**

> **作者:** Jae Young Choi; Seon Gyeom Kim; Hyungjun Yoon; Taeckyung Lee; Donggun Lee; Jaeryung Chung; Jihyung Kil; Ryan Rossi; Sung-Ju Lee; Tak Yeon Lee
>
> **备注:** 6 pages. Conditionally accepted to IEEE PacificVis 2026 (VisNotes track)
>
> **摘要:** Large Language Models (LLMs) have emerged as foundation models for IoT applications such as human activity recognition (HAR). However, directly applying high-frequency and multi-dimensional sensor data, such as eye-tracking data, leads to information loss and high token costs. To mitigate this, we investigate a visual prompting strategy that transforms sensor signals into data visualization images as an input to multimodal LLMs (MLLMs) using eye-tracking data. We conducted a systematic evaluation of MLLM-based HAR across three public eye-tracking datasets using three visualization types of timeline, heatmap, and scanpath, under varying temporal window sizes. Our findings suggest that visual prompting provides a token-efficient and scalable representation for eye-tracking data, highlighting its potential to enable MLLMs to effectively reason over high-frequency sensor signals in IoT contexts.
>
---
#### [new 306] TinyGaze: Lightweight Gaze-Gesture Recognition on Commodity Mobile Devices
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于轻量级眼动手势识别任务，旨在解决移动设备上高效实现无手输入的问题。提出TinyHAR模型，在小样本下实现高精度识别。**

- **链接: [https://arxiv.org/pdf/2604.09658](https://arxiv.org/pdf/2604.09658)**

> **作者:** Yaxiong Lei; Hyochan Cho; Fergus Buchanan; Shijing He; Xinya Gong; Yuheng Wang; Juan Ye
>
> **备注:** 6 pages, 3 figures. Extended Abstracts of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26), April 13-17, 2026, Barcelona, Spain
>
> **摘要:** Gaze gestures can provide hands free input on mobile devices, but practical use requires (i) gestures users can learn and recall and (ii) recognition models that are efficient enough for on-device deployment. We present an end-to-end pipeline using commodity ARKit head/eye transforms and a scaffolded guidance-to-recall protocol grounded in learning theory. In a pilot feasibility study (N=4 participants; 240 trials; controlled single-session setting), we benchmark a compact time-series model (TinyHAR) against deeper baselines (DeepConvLSTM, SA-HAR) on 5-way gesture recognition and 4-way user identification. TinyHAR achieves strong performance in this pilot benchmark (Macro F1 = 0.960 for gesture recognition; Macro F1 = 0.997 for user identification) while using only 46k parameters. A modality analysis further indicates that head pose dynamics are highly informative for mobile gaze gestures, highlighting embodied head--eye coordination as a key design consideration. Although the small sample size and controlled setting limit generalizability, these results indicate a potential direction for further investigation into on-device gaze gesture recognition.
>
---
#### [new 307] The Salami Slicing Threat: Exploiting Cumulative Risks in LLM Systems
- **分类: cs.CR; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于安全防护任务，针对LLM的多轮越狱攻击问题，提出“萨拉米切片风险”概念及自动攻击框架Salami Attack，有效提升攻击成功率并探讨防御策略。**

- **链接: [https://arxiv.org/pdf/2604.11309](https://arxiv.org/pdf/2604.11309)**

> **作者:** Yihao Zhang; Kai Wang; Jiangrong Wu; Haolin Wu; Yuxuan Zhou; Zeming Wei; Dongxian Wu; Xun Chen; Jun Sun; Meng Sun
>
> **摘要:** Large Language Models (LLMs) face prominent security risks from jailbreaking, a practice that manipulates models to bypass built-in security constraints and generate unethical or unsafe content. Among various jailbreak techniques, multi-turn jailbreak attacks are more covert and persistent than single-turn counterparts, exposing critical vulnerabilities of LLMs. However, existing multi-turn jailbreak methods suffer from two fundamental limitations that affect the actual impact in real-world scenarios: (a) As models become more context-aware, any explicit harmful trigger is increasingly likely to be flagged and blocked; (b) Successful final-step triggers often require finely tuned, model-specific contexts, making such attacks highly context-dependent. To fill this gap, we propose \textit{Salami Slicing Risk}, which operates by chaining numerous low-risk inputs that individually evade alignment thresholds but cumulatively accumulate harmful intent to ultimately trigger high-risk behaviors, without heavy reliance on pre-designed contextual structures. Building on this risk, we develop Salami Attack, an automatic framework universally applicable to multiple model types and modalities. Rigorous experiments demonstrate its state-of-the-art performance across diverse models and modalities, achieving over 90\% Attack Success Rate on GPT-4o and Gemini, as well as robustness against real-world alignment defenses. We also proposed a defense strategy to constrain the Salami Attack by at least 44.8\% while achieving a maximum blocking rate of 64.8\% against other multi-turn jailbreak attacks. Our findings provide critical insights into the pervasive risks of multi-turn jailbreaking and offer actionable mitigation strategies to enhance LLM security.
>
---
#### [new 308] Search-MIND: Training-Free Multi-Modal Medical Image Registration
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决多模态图像配准中的非线性强度关系和局部最优问题。提出Search-MIND框架，通过优化策略实现无需训练的精准配准。**

- **链接: [https://arxiv.org/pdf/2604.09743](https://arxiv.org/pdf/2604.09743)**

> **作者:** Boya Wang; Ruizhe Li; Chao Chen; Xin Chen
>
> **摘要:** Multi-modal image registration plays a critical role in precision medicine but faces challenges from non-linear intensity relationships and local optima. While deep learning models enable rapid inference, they often suffer from generalization collapse on unseen modalities. To address this, we propose Search-MIND, a training-free, iterative optimization framework for instance-specific registration. Our pipeline utilizes a coarse-to-fine strategy: a hierarchical coarse alignment stage followed by deformable refinement. We introduce two novel loss functions: Variance-Weighted Mutual Information (VWMI), which prioritizes informative tissue regions to shield global alignment from background noise and uniform regions, and Search-MIND (S-MIND), which broadens the convergence basin of structural descriptors by considering larger local search range. Evaluations on CARE Liver 2025 and CHAOS Challenge datasets show that Search-MIND consistently outperforms classical baselines like ANTs and foundation model-based approaches like DINO-reg, offering superior stability across diverse modalities.
>
---
#### [new 309] Preventing Latent Rehearsal Decay in Online Continual SSL with SOLAR
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究在线持续自监督学习（OCSSL）任务，解决模型在连续数据流中快速收敛与稳定性之间的平衡问题。提出SOLAR方法，通过管理记忆缓冲区和引入损失函数，提升性能与速度。**

- **链接: [https://arxiv.org/pdf/2604.10586](https://arxiv.org/pdf/2604.10586)**

> **作者:** Giacomo Cignoni; Simone Magistri; Andrew D. Bagdanov; Antonio Carta
>
> **摘要:** This paper explores Online Continual Self-Supervised Learning (OCSSL), a scenario in which models learn from continuous streams of unlabeled, non-stationary data, where methods typically employ replay and fast convergence is a central desideratum. We find that OCSSL requires particular attention to the stability-plasticity trade-off: stable methods (e.g. replay with Reservoir sampling) are able to converge faster compared to plastic ones (e.g. FIFO buffer), but incur in performance drops under certain conditions. We explain this collapse phenomenon with the Latent Rehearsal Decay hypothesis, which attributes it to latent space degradation under excessive stability of replay. We introduce two metrics (Overlap and Deviation) that diagnose latent degradation and correlate with accuracy declines. Building on these insights, we propose SOLAR, which leverages efficient online proxies of Deviation to guide buffer management and incorporates an explicit Overlap loss, allowing SOLAR to adaptively managing plasticity. Experiments demonstrate that SOLAR achieves state-of-the-art performance on OCSSL vision benchmarks, with both high convergence speed and final performance.
>
---
#### [new 310] Solving Physics Olympiad via Reinforcement Learning on Physics Simulators
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于物理推理任务，旨在解决物理领域缺乏大规模QA数据的问题。通过物理模拟器生成合成数据，利用强化学习训练LLM，提升其物理推理能力。**

- **链接: [https://arxiv.org/pdf/2604.11805](https://arxiv.org/pdf/2604.11805)**

> **作者:** Mihir Prabhudesai; Aryan Satpathy; Yangmin Li; Zheyang Qin; Nikash Bhardwaj; Amir Zadeh; Chuan Li; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project Webpage - this https URL
>
> **摘要:** We have witnessed remarkable advances in LLM reasoning capabilities with the advent of DeepSeek-R1. However, much of this progress has been fueled by the abundance of internet question-answer (QA) pairs, a major bottleneck going forward, since such data is limited in scale and concentrated mainly in domains like mathematics. In contrast, other sciences such as physics lack large-scale QA datasets to effectively train reasoning-capable models. In this work, we show that physics simulators can serve as a powerful alternative source of supervision for training LLMs for physical reasoning. We generate random scenes in physics engines, create synthetic question-answer pairs from simulated interactions, and train LLMs using reinforcement learning on this synthetic data. Our models exhibit zero-shot sim-to-real transfer to real-world physics benchmarks: for example, training solely on synthetic simulated data improves performance on IPhO (International Physics Olympiad) problems by 5-10 percentage points across model sizes. These results demonstrate that physics simulators can act as scalable data generators, enabling LLMs to acquire deep physical reasoning skills beyond the limitations of internet-scale QA data. Code available at: this https URL.
>
---
#### [new 311] Edu-MMBias: A Three-Tier Multimodal Benchmark for Auditing Social Bias in Vision-Language Models under Educational Contexts
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言模型公平性审计任务，旨在检测教育场景下的社会偏见。通过构建多模态基准Edu-MMBias，分析模型在认知、情感和行为层面的偏见问题。**

- **链接: [https://arxiv.org/pdf/2604.10200](https://arxiv.org/pdf/2604.10200)**

> **作者:** Ruijia Li; Mingzi Zhang; Zengyi Yu; Yuang Wei; Bo Jiang
>
> **摘要:** As Vision-Language Models (VLMs) become integral to educational decision-making, ensuring their fairness is paramount. However, current text-centric evaluations neglect the visual modality, leaving an unregulated channel for latent social biases. To bridge this gap, we present Edu-MMBias, a systematic auditing framework grounded in the tri-component model of attitudes from social psychology. This framework diagnoses bias across three hierarchical dimensions: cognitive, affective, and behavioral. Utilizing a specialized generative pipeline that incorporates a self-correct mechanism and human-in-the-loop verification, we synthesize contamination-resistant student profiles to conduct a holistic stress test on state-of-the-art VLMs. Our extensive audit reveals critical, counter-intuitive patterns: models exhibit a compensatory class bias favoring lower-status narratives while simultaneously harboring deep-seated health and racial stereotypes. Crucially, we find that visual inputs act as a safety backdoor, triggering a resurgence of biases that bypass text-based alignment safeguards and revealing a systematic misalignment between latent cognition and final decision-making. The contributions of this paper are available at: this https URL.
>
---
#### [new 312] NeuVolEx: Implicit Neural Features for Volume Exploration
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出NeuVolEx，用于体积探索任务，解决ROI分类与聚类问题。通过隐式神经特征提升体积数据的分析效果。**

- **链接: [https://arxiv.org/pdf/2604.11172](https://arxiv.org/pdf/2604.11172)**

> **作者:** Haill An; Suhyeon Kim; Donghyuk Choo; Younhyun Jung
>
> **备注:** 11 pages, 9 figures. Under review
>
> **摘要:** Direct volume rendering (DVR) aims to help users identify and examine regions of interest (ROIs) within volumetric data, and feature representations that support effective ROI classification and clustering play a fundamental role in volume exploration. Existing approaches typically rely on either explicit local feature representations or implicit convolutional feature representations learned from raw volumes. However, explicit local feature representations are limited in capturing broader geometric patterns and spatial correlations, while implicit convolutional feature representations do not necessarily ensure robust performance in practice, where user supervision is typically limited. Meanwhile, implicit neural representations (INRs) have recently shown strong promise in DVR for volume compression, owing to their ability to compactly parameterize continuous volumetric fields. In this work, we propose NeuVolEx, a neural volume exploration approach that extends the role of INRs beyond volume compression. Unlike prior compression methods that focus on INR outputs, NeuVolEx leverages feature representations learned during INR training as a robust basis for volume exploration. To better adapt these feature representations to exploration tasks, we augment a base INR with a structural encoder and a multi-task learning scheme that improve spatial coherence for ROI characterization. We validate NeuVolEx on two fundamental volume exploration tasks: image-based transfer function (TF) design and viewpoint recommendation. NeuVolEx enables accurate ROI classification under sparse user supervision for image-based TF design and supports unsupervised clustering to identify compact complementary viewpoints that reveal different ROI clusters. Experiments on diverse volume datasets with varying modalities and ROI complexities demonstrate NeuVolEx improves both effectiveness and usability over prior methods
>
---
#### [new 313] A Faster Path to Continual Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决传统方法训练成本高的问题。提出C-Flat Turbo优化器，减少梯度计算，提升训练效率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2604.11064](https://arxiv.org/pdf/2604.11064)**

> **作者:** Wei Li; Hangjie Yuan; Zixiang Zhao; Borui Kang; Ziwei Liu; Tao Feng
>
> **摘要:** Continual Learning (CL) aims to train neural networks on a dynamic stream of tasks without forgetting previously learned knowledge. Among optimization-based approaches, C-Flat has emerged as a promising solution due to its plug-and-play nature and its ability to encourage uniformly low-loss regions for both new and old tasks. However, C-Flat requires three additional gradient computations per iteration, imposing substantial overhead on the optimization process. In this work, we propose C-Flat Turbo, a faster yet stronger optimizer that significantly reduces the training cost. We show that the gradients associated with first-order flatness contain direction-invariant components relative to the proxy-model gradients, enabling us to skip redundant gradient computations in the perturbed ascent steps. Moreover, we observe that these flatness-promoting gradients progressively stabilize across tasks, which motivates a linear scheduling strategy with an adaptive trigger to allocate larger turbo steps for later tasks. Experiments show that C-Flat Turbo is 1.0$\times$ to 1.25$\times$ faster than C-Flat across a wide range of CL methods, while achieving comparable or even improved accuracy.
>
---
#### [new 314] Continuous Adversarial Flow Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出连续对抗流模型，用于生成任务，解决传统流匹配样本对齐不足的问题，通过引入判别器提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.11521](https://arxiv.org/pdf/2604.11521)**

> **作者:** Shanchuan Lin; Ceyuan Yang; Zhijie Lin; Hao Chen; Haoqi Fan
>
> **摘要:** We propose continuous adversarial flow models, a type of continuous-time flow model trained with an adversarial objective. Unlike flow matching, which uses a fixed mean-squared-error criterion, our approach introduces a learned discriminator to guide training. This change in objective induces a different generalized distribution, which empirically produces samples that are better aligned with the target data distribution. Our method is primarily proposed for post-training existing flow-matching models, although it can also train models from scratch. On the ImageNet 256px generation task, our post-training substantially improves the guidance-free FID of latent-space SiT from 8.26 to 3.63 and of pixel-space JiT from 7.17 to 3.57. It also improves guided generation, reducing FID from 2.06 to 1.53 for SiT and from 1.86 to 1.80 for JiT. We further evaluate our approach on text-to-image generation, where it achieves improved results on both the GenEval and DPG benchmarks.
>
---
#### [new 315] Sharpness-Aware Surrogate Training for On-Sensor Spiking Neural Networks
- **分类: cs.NE; cs.CV; cs.LG**

- **简介: 该论文属于边缘计算任务，解决SNN在部署时精度下降问题，提出SAST方法提升模型稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2604.09696](https://arxiv.org/pdf/2604.09696)**

> **作者:** Maximilian Nicholson
>
> **备注:** Currently under review at a conference workshop
>
> **摘要:** Spiking neural networks (SNNs) are a natural computational model for on-sensor and near-sensor vision, where event driven processors must operate under strict power budgets with hard binary spikes. However, models trained with surrogate gradients often degrade sharply when the smooth surrogate nonlinearity is replaced by a hard threshold at deployment; a surrogate-to-hard transfer gap that directly limits on-sensor accuracy. We study Sharpness-Aware Surrogate Training (SAST), which applies Sharpness-Aware Minimization (SAM) to a surrogate-forward SNN so that the training objective is smooth and the gradient is exact, and position it as one gap-reduction strategy under the tested settings rather than the only viable mechanism. Under explicit contraction assumptions we provide state-stability, input-Lipschitz, and smoothness bounds, together with a corresponding nonconvex convergence result. On two event-camera benchmarks, swap-only hard-spike accuracy improves from 65.7\% to 94.7\% on N-MNIST and from 31.8\% to 63.3\% on DVS Gesture. Under a hardware-aware inference simulation (INT8/INT4 weight quantization, fixed-point membrane potentials, discrete leak factors), SAST remains strong: on N-MNIST, hard-spike accuracy improves from 47.6\% to 96.9\% (INT8) and from 43.2\% to 81.0\% (INT4), while on DVS Gesture it improves from 25.3\% to 47.6\% (INT8) and from 26.0\% to 43.8\% (INT4). SynOps also decrease under the same hardware-aware setting, including 1734k$\rightarrow$1315k (N-MNIST, INT8) and 86221k$\rightarrow$4323k (DVS Gesture, INT8). These results suggest that SAST is a promising component in a broader toolbox for on-sensor spiking inference under the tested settings.
>
---
#### [new 316] Efficient Matrix Implementation for Rotary Position Embedding
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于自然语言处理中的位置编码优化任务，旨在解决RoPE实现中计算效率低的问题。通过提出RoME，用矩阵变换替代向量操作，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2604.09742](https://arxiv.org/pdf/2604.09742)**

> **作者:** Chen Minqi; Zhongqi Yue; Shihao Zhang; Yun Xu; Peng Wu; kaixiang Xu; Zeyi Huang; Hanwang Zhang
>
> **摘要:** Rotary Position Embedding (RoPE) has become a core component of modern Transformer architectures across language, vision, and 3D domains. However, existing implementations rely on vector-level split and merge operations that introduce non-negligible computational overhead, often overlooked in attention optimization. The problem is further amplified in multi-dimensional settings (e.g., 2D and 3D RoPE), where additional vector operations and uneven feature partitions degrade hardware utilization. To overcome these limitations, we propose RoME (Rotary Matrix position Embedding), a mathematically equivalent yet computationally efficient reformulation of RoPE that replaces vector operations with unified matrix transformations. RoME eliminates dimension-specific operations, simplifies implementation, and enables fused parallel execution across Cube and Vector units on modern NPUs. Experiments show that RoME delivers substantial acceleration at both the operator and full-model levels. The implementation is available at this https URL.
>
---
#### [new 317] Back to the Barn with LLAMAs: Evolving Pretrained LLM Backbones in Finetuning Vision Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究预训练大语言模型对视觉语言模型性能的影响，探讨如何有效更新VLM以利用更先进的LLM。任务属于多模态学习，解决LLM进化对VLM效果的不确定性问题，通过对比不同LLM版本验证其影响。**

- **链接: [https://arxiv.org/pdf/2604.10985](https://arxiv.org/pdf/2604.10985)**

> **作者:** Sameera Horawalavithana; Lauren Phillips; Ian Stewart; Sai Munikoti; Karl Pazdernik
>
> **备注:** Preprint and under review
>
> **摘要:** Vision-Language Models (VLMs) have rapidly advanced by leveraging powerful pre-trained Large Language Models (LLMs) as core reasoning backbones. As new and more capable LLMs emerge with improved reasoning, instruction-following, and generalization, there is a pressing need to efficiently update existing VLMs to incorporate these advancements. However, the integration of new LLMs into VLMs, particularly how the evolving LLMs contribute to multimodal reasoning, alignment, and task-specific performance remains underexplored. Addressing this gap is important for VLM development, given the rapid evolution of pretrained LLM backbones. This study presents a controlled and systematic investigation of how changes in the pretrained LLM backbone affect downstream VLM task performance. By having the vision encoder, training data, and post-training algorithm remain same across LLAMA-1, LLAMA-2, and LLAMA-3 based VLMs, we find that newer LLM backbones do not always lead to better VLMs, but the performance depends on the downstream VLM task. For example, in visual question and answering tasks, newer LLM backbones tend to solve different questions rather than just more questions, and our analysis shows this is driven by differences in how the models process information, including better calibrated confidence and more stable internal representations. We also find that some VLM capabilities appear only in the newest LLM generation, while tasks that depend mainly on visual understanding see little benefit from a newer LLM backbone.
>
---
#### [new 318] Quantum-Gated Task-interaction Knowledge Distillation for Pre-trained Model-based Class-Incremental Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于类增量学习任务，解决多任务间知识干扰和遗忘问题。提出QKD框架，通过量子门控机制实现任务间知识迁移，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.11112](https://arxiv.org/pdf/2604.11112)**

> **作者:** Linjie Li; Huiyu Xiao; Jiarui Cao; Zhenyu Wu; Yang Ji
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** Class-incremental learning (CIL) aims to continuously accumulate knowledge from a stream of tasks and construct a unified classifier over all seen classes. Although pretrained models (PTMs) have shown promising performance in CIL, they still struggle with the entanglement of multi-task subspaces, leading to catastrophic forgetting when task routing parameters are poorly calibrated or task-level representations are rigidly fixed. To address this issue, we propose a novel Quantum-Gated Task-interaction Knowledge Distillation (QKD) framework that leverages quantum gating to guide inter-task knowledge transfer. Specifically, we introduce a quantum-gated task modulation gating mechanism to model the relational dependencies among task embedding, dynamically capturing the sample-to-task relevance for both joint training and inference across streaming tasks. Guided by the quantum gating outputs, we perform task-interaction knowledge distillation guided by these task-embedding-level correlation weights from old to new adapters, enabling the model to bridge the representation gaps between independent task subspaces. Extensive experiments demonstrate that QKD effectively mitigates forgetting and achieves state-of-the-art performance.
>
---
#### [new 319] Rethinking the Diffusion Model from a Langevin Perspective
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于生成模型领域，旨在从Langevin视角重新解释扩散模型，解决其原理理解与统一框架构建问题，提供更直观的理论分析。**

- **链接: [https://arxiv.org/pdf/2604.10465](https://arxiv.org/pdf/2604.10465)**

> **作者:** Candi Zheng; Yuan Lan
>
> **备注:** 20 pages, 7 figures
>
> **摘要:** Diffusion models are often introduced from multiple perspectives, such as VAEs, score matching, or flow matching, accompanied by dense and technically demanding mathematics that can be difficult for beginners to grasp. One classic question is: how does the reverse process invert the forward process to generate data from pure noise? This article systematically organizes the diffusion model from a fresh Langevin perspective, offering a simpler, clearer, and more intuitive answer. We also address the following questions: how can ODE-based and SDE-based diffusion models be unified under a single framework? Why are diffusion models theoretically superior to ordinary VAEs? Why is flow matching not fundamentally simpler than denoising or score matching, but equivalent under maximum-likelihood? We demonstrate that the Langevin perspective offers clear and straightforward answers to these questions, bridging existing interpretations of diffusion models, showing how different formulations can be converted into one another within a common framework, and offering pedagogical value for both learners and experienced researchers seeking deeper intuition.
>
---
#### [new 320] Belief-Aware VLM Model for Human-like Reasoning
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于意图推理任务，旨在解决传统模型难以捕捉长期人类意图的问题。提出一种融合记忆与强化学习的信念感知VLM框架，提升模型的类人推理能力。**

- **链接: [https://arxiv.org/pdf/2604.09686](https://arxiv.org/pdf/2604.09686)**

> **作者:** Anshul Nayak; Shahil Shaik; Yue Wang
>
> **备注:** 6 Pages, 3 figures, 1 Table
>
> **摘要:** Traditional neural network models for intent inference rely heavily on observable states and struggle to generalize across diverse tasks and dynamic environments. Recent advances in Vision Language Models (VLMs) and Vision Language Action (VLA) models introduce common-sense reasoning through large-scale multimodal pretraining, enabling zero-shot performance across tasks. However, these models still lack explicit mechanisms to represent and update belief, limiting their ability to reason like humans or capture the evolving human intent over long-horizon. To address this, we propose a belief-aware VLM framework that integrates retrieval-based memory and reinforcement learning. Instead of learning an explicit belief model, we approximate belief using a vector-based memory that retrieves relevant multimodal context, which is incorporated into the VLM for reasoning. We further refine decision-making using a reinforcement learning policy over the VLM latent space. We evaluate our approach on publicly available VQA datasets such as HD-EPIC and demonstrate consistent improvements over zero-shot baselines, highlighting the importance of belief-aware reasoning.
>
---
#### [new 321] QShield: Securing Neural Networks Against Adversarial Attacks using Quantum Circuits
- **分类: cs.CR; cs.AI; cs.CV; cs.LG; quant-ph**

- **简介: 论文提出QShield，一种结合量子电路的混合神经网络架构，用于增强深度学习模型对对抗攻击的鲁棒性。该工作旨在提升安全关键应用中的模型可靠性。**

- **链接: [https://arxiv.org/pdf/2604.10933](https://arxiv.org/pdf/2604.10933)**

> **作者:** Navid Azimi; Aditya Prakash; Yao Wang; Li Xiong
>
> **摘要:** Deep neural networks remain highly vulnerable to adversarial perturbations, limiting their reliability in security- and safety-critical applications. To address this challenge, we introduce QShield, a modular hybrid quantum-classical neural network (HQCNN) architecture designed to enhance the adversarial robustness of classical deep learning models. QShield integrates a conventional convolutional neural network (CNN) backbone for feature extraction with a quantum processing module that encodes the extracted features into quantum states, applies structured entanglement operations under realistic noise models, and outputs a hybrid prediction through a dynamically weighted fusion mechanism implemented via a lightweight multilayer perceptron (MLP). We systematically evaluate both classical and hybrid quantum-classical models on the MNIST, OrganAMNIST, and CIFAR-10 datasets, using a comprehensive set of robustness, efficiency, and computational performance metrics. Our results demonstrate that classical models are highly vulnerable to adversarial attacks, whereas the proposed hybrid models with entanglement patterns maintain high predictive accuracy while substantially reducing attack success rates across a wide range of adversarial attacks. Furthermore, the proposed hybrid architecture significantly increased the computational cost required to generate adversarial examples, thereby introducing an additional layer of defense. These findings indicate that the proposed modular hybrid architecture achieves a practical balance between predictive accuracy and adversarial robustness, positioning it as a promising approach for secure and reliable machine learning in sensitive and safety-critical applications.
>
---
#### [new 322] ComSim: Building Scalable Real-World Robot Data Generation via Compositional Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决真实世界训练数据获取困难的问题。通过组合仿真方法生成高质量、多样化的训练数据，缩小仿真与现实的差距。**

- **链接: [https://arxiv.org/pdf/2604.11386](https://arxiv.org/pdf/2604.11386)**

> **作者:** Yiran Qin; Jiahua Ma; Li Kang; Wenzhan Li; Yihang Jiao; Xin Wen; Xiufeng Song; Heng Zhou; Jiwen Yu; Zhenfei Yin; Xihui Liu; Philip Torr; Yilun Du; Ruimao Zhang
>
> **备注:** 14 pages, 8 figures, 4 tables; supplementary material included; Project page: this https URL
>
> **摘要:** Recent advancements in foundational models, such as large language models and world models, have greatly enhanced the capabilities of robotics, enabling robots to autonomously perform complex tasks. However, acquiring large-scale, high-quality training data for robotics remains a challenge, as it often requires substantial manual effort and is limited in its coverage of diverse real-world environments. To address this, we propose a novel hybrid approach called Compositional Simulation, which combines classical simulation and neural simulation to generate accurate action-video pairs while maintaining real-world consistency. Our approach utilizes a closed-loop real-sim-real data augmentation pipeline, leveraging a small amount of real-world data to generate diverse, large-scale training datasets that cover a broader spectrum of real-world scenarios. We train a neural simulator to transform classical simulation videos into real-world representations, improving the accuracy of policy models trained in real-world environments. Through extensive experiments, we demonstrate that our method significantly reduces the sim2real domain gap, resulting in higher success rates in real-world policy model training. Our approach offers a scalable solution for generating robust training data and bridging the gap between simulated and real-world robotics.
>
---
#### [new 323] K-STEMIT: Knowledge-Informed Spatio-Temporal Efficient Multi-Branch Graph Neural Network for Subsurface Stratigraphy Thickness Estimation from Radar Data
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出K-STEMIT模型，用于从雷达数据中估计冰层厚度，解决传统方法对噪声敏感和缺乏物理知识的问题。**

- **链接: [https://arxiv.org/pdf/2604.09922](https://arxiv.org/pdf/2604.09922)**

> **作者:** Zesheng Liu; Maryam Rahnemoonfar
>
> **摘要:** Subsurface stratigraphy contains important spatio-temporal information about accumulation, deformation, and layer formation in polar ice sheets. In particular, variations in internal ice layer thickness provide valuable constraints for snow mass balance estimation and projections of ice sheet change. Although radar sensors can capture these layered structures as depth-resolved radargrams, convolutional neural networks applied directly to radar images are often sensitive to speckle noise and acquisition artifacts. In addition, purely data-driven methods may underuse physical knowledge, leading to unrealistic thickness estimates under spatial or temporal extrapolation. To address these challenges, we develop K-STEMIT, a novel knowledge-informed, efficient, multi-branch spatio-temporal graph neural network that combines a geometric framework for spatial learning with temporal convolution to capture temporal dynamics, and incorporates physical data synchronized from the Model Atmospheric Regional physical weather model. An adaptive feature fusion strategy is employed to dynamically combine features learned from different branches. Extensive experiments have been conducted to compare K-STEMIT against current state-of-the-art methods in both knowledge-informed and non-knowledge-informed settings, as well as other existing methods. Results show that K-STEMIT consistently achieves the highest accuracy while maintaining near-optimal efficiency. Most notably, incorporating adaptive feature fusion and physical priors reduces the root mean-squared error by 21.01% with negligible additional cost compared to its conventional multi-branch variants. Additionally, our proposed K-STEMIT achieves consistently lower per-year relative MAE, enabling reliable, continuous spatiotemporal assessment of snow accumulation variability across large spatial regions.
>
---
#### [new 324] ReaLiTy and LADS: A Unified Framework and Dataset Suite for LiDAR Adaptation Across Sensors and Adverse Weather Conditions
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于LiDAR感知任务，旨在解决传感器和天气变化下的适应问题。提出ReaLiTy框架和LADS数据集，生成物理一致的LiDAR数据，提升跨域一致性与真实天气效果。**

- **链接: [https://arxiv.org/pdf/2604.10213](https://arxiv.org/pdf/2604.10213)**

> **作者:** Vivek Anand; Bharat Lohani; Rakesh Mishra; Gaurav Pandey
>
> **摘要:** Reliable LiDAR perception requires robustness across sensors, environments, and adverse weather. However, existing datasets rarely provide physically consistent observations of the same scene under varying sensor configurations and weather conditions, limiting systematic analysis of domain shifts. This work presents ReaLiTy, a unified physics-informed framework that transforms LiDAR data to match target sensor specifications and weather conditions. The framework integrates physically grounded cues with a learning-based module to generate realistic intensity patterns, while a physics-based weather model introduces consistent geometric and radiometric degradations. Building on this framework, we introduce the LiDAR Adaptation Dataset Suite (LADS), a collection of physically consistent, transformation-ready point clouds with one-to-one correspondence to original datasets. Experiments demonstrate improved cross-domain consistency and realistic weather effects. ReaLiTy and LADS provide a reproducible foundation for studying LiDAR adaptation and simulation-driven perception in intelligent transportation systems.
>
---
#### [new 325] R2E-VID: Two-Stage Robust Routing via Temporal Gating for Elastic Edge-Cloud Video Inference
- **分类: cs.NI; cs.CV; cs.DC**

- **简介: 该论文属于边缘-云视频推理任务，旨在解决动态资源分配与路由效率问题。提出R2E-VID框架，通过时间门控实现弹性负载划分和优化，降低延迟与成本。**

- **链接: [https://arxiv.org/pdf/2604.09681](https://arxiv.org/pdf/2604.09681)**

> **作者:** Zheming Yang; Lulu Zuo; Shun Lu; Yangyu Zhang; Zhicheng Li; Xiangyang Li; Yang You
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** With the rapid growth of large-scale video analytics applications, edge-cloud collaborative systems have become the dominant paradigm for real-time inference. However, existing approaches often fail to dynamically adapt to heterogeneous video content and fluctuating resource conditions, resulting in suboptimal routing efficiency and high computational costs. In this paper, we propose R2E-VID, a two-stage robust routing framework via temporal gating for elastic edge-cloud video inference. In the first stage, R2E-VID introduces a temporal gating mechanism that models the temporal consistency and motion dynamics of incoming video streams to predict the optimal routing pattern for each segment. This enables adaptive partitioning of inference workloads between edge and cloud nodes, achieving fine-grained spatiotemporal elasticity. In the second stage, a robust routing optimization module refines the allocation through multi-model adaptation, jointly minimizing inference delay and resource consumption under dynamic network and workload variations. Extensive experiments on public datasets demonstrate that R2E-VID achieves up to 60% reduction in overall cost compared to cloud-centric baselines, and delivers 35-45% lower delay while improving inference accuracy by 2-7% over state-of-the-art edge-cloud solutions.
>
---
#### [new 326] EagleVision: A Multi-Task Benchmark for Cross-Domain Perception in High-Speed Autonomous Racing
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EagleVision，一个用于高速自动驾驶赛车的多任务基准，解决跨域感知问题。通过构建标注数据集和评估协议，研究模型在不同场景下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11400](https://arxiv.org/pdf/2604.11400)**

> **作者:** Zakhar Yagudin; Murad Mebrahtu; Ren Jin; Jiaqi Huang; Yujia Yue; Dzmitry Tsetserukou; Jorge Dias; Majid Khonji
>
> **摘要:** High-speed autonomous racing presents extreme perception challenges, including large relative velocities and substantial domain shifts from conventional urban-driving datasets. Existing benchmarks do not adequately capture these high-dynamic conditions. We introduce EagleVision, a unified LiDAR-based multi-task benchmark for 3D detection and trajectory prediction in high-speed racing, providing newly annotated 3D bounding boxes for the Indy Autonomous Challenge dataset (14,893 frames) and the A2RL Real competition dataset (1,163 frames), together with 12,000 simulator-generated annotated frames, all standardized under a common evaluation protocol. Using a dataset-centric transfer framework, we quantify cross-domain generalization across urban, simulator, and real racing domains. Urban pretraining improves detection over scratch training (NDS 0.72 vs. 0.69), while intermediate pretraining on real racing data achieves the best transfer to A2RL (NDS 0.726), outperforming simulator-only adaptation. For trajectory prediction, Indy-trained models surpass in-domain A2RL training on A2RL test sequences (FDE 0.947 vs. 1.250), highlighting the role of motion-distribution coverage in cross-domain forecasting. EagleVision enables systematic study of perception generalization under extreme high-speed dynamics. The dataset and benchmark are publicly available at this https URL
>
---
#### [new 327] Anthropogenic Regional Adaptation in Multimodal Vision-Language Model
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决区域文化对齐问题。提出Anthropogenic Regional Adaptation和GG-EZ方法，提升模型在特定地区的文化相关性，同时保持全局性能。**

- **链接: [https://arxiv.org/pdf/2604.11490](https://arxiv.org/pdf/2604.11490)**

> **作者:** Samuel Cahyawijaya; Peerat Limkonchotiwat; Tack Hwa Wong; Hitesh Laxmichand Patel; Amit Agarwal; Manuel Antonio Rufino; Carlos Rafael Catalan; Muhammad Reza Qorib; Vicky Feliren; Holy Lovenia; Aye Hninn Khine; Frederikus Hudi; David Anugraha; Alham Fikri Aji; Romrawin Chumpu; Viet-Thanh Pham; Minghan Wang; Mohamed Fazli Imam; Ruochen Zhang; Joseph Marvin Imperial; Do Xuan Long; Musa Izzanardi Wijanarko; Joel Ruben Antony Moniz; Patrick Amadeus Irawan; Hanif Muhammad Zhafran; Isaiah Flores; Ira Salsabila; Jun Kevin; Jostin Jerico Rosal; Patricia Nicole Monderin; Kun Kerdthaisong; Ahmad Mustafid; My Chiffon Nguyen; Natchapon Jongwiriyanurak; Siva Worajitwannakul; Haochen Li; Adrian Xuan Wei Lim; Bin Wang; Muhammad Ravi Shulthan Habibi; Lynnette Hui Xian Ng; Mithil Bangera; Yeshil Bangera; Priyaranjan Pattnayak; Dun Li Chan; Sherissa Caren Djuniwar; Hee Ming Shan
>
> **摘要:** While the field of vision-language (VL) has achieved remarkable success in integrating visual and textual information across multiple languages and domains, there is still no dedicated framework for assessing human-centric alignment in vision-language systems. We offer two contributions to address this gap. First, we introduce Anthropogenic Regional Adaptation: a novel paradigm that aims to optimize model relevance to specific regional contexts while ensuring the retention of global generalization capabilities. Second, we present a simple, but effective adaptation method named Geographical-generalization-made-easy (GG-EZ), which utilizes regional data filtering and model merging. Through comprehensive experiments on 3 VL architectures: large vision-language models, text-to-image diffusion models, and vision-language embedding models, and a case study in Southeast Asia (SEA) regional adaptation, we demonstrate the importance of Anthropogenic Regional Adaptation and the effectiveness of GG-EZ, showing 5-15% gains in cultural relevance metrics across SEA while maintaining over 98% of global performance and even occasionally surpassing it. Our findings establish Anthropogenic Regional Alignment as a foundational paradigm towards applicability of multimodal vision-language models in diverse regions and demonstrate a simple-yet-effective baseline method that optimizes regional value alignment while preserving global generalization.
>
---
#### [new 328] Towards Multi-Source Domain Generalization for Sleep Staging with Noisy Labels
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于睡眠分期任务，解决多源域泛化中的噪声标签问题。提出FF-TRUST框架，通过时频一致性与置信度正则化提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10009](https://arxiv.org/pdf/2604.10009)**

> **作者:** Kening Wang; Di Wen; Yufan Chen; Ruiping Liu; Junwei Zheng; Jiale Wei; Kailun Yang; Rainer Stiefelhagen; Kunyu Peng
>
> **备注:** The benchmark and code will be made publicly available at this https URL
>
> **摘要:** Automatic sleep staging is a multimodal learning problem involving heterogeneous physiological signals such as EEG and EOG, which often suffer from domain shifts across institutions, devices, and populations. In practice, these data are also affected by noisy annotations, yet label-noise-robust multi-source domain generalization remains underexplored. We present the first benchmark for Noisy Labels in Multi-Source Domain-Generalized Sleep Staging (NL-DGSS) and show that existing noisy-label learning methods degrade substantially when domain shifts and label noise coexist. To address this challenge, we propose FF-TRUST, a domain-invariant multimodal sleep staging framework with Joint Time-Frequency Early Learning Regularization (JTF-ELR). By jointly exploiting temporal and spectral consistency together with confidence-diversity regularization, FF-TRUST improves robustness under noisy supervision. Experiments on five public datasets demonstrate consistent state-of-the-art performance under diverse symmetric and asymmetric noise settings. The benchmark and code will be made publicly available at this https URL.
>
---
#### [new 329] Compact single-shot ranging and near-far imaging using metasurfaces
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于成像任务，旨在解决单次拍摄中远近景同时成像与测距的问题。通过超表面技术实现紧凑系统，可在同一传感器上获取不同距离的图像，并实现高精度被动测距。**

- **链接: [https://arxiv.org/pdf/2604.10037](https://arxiv.org/pdf/2604.10037)**

> **作者:** Junjie Luo; Yuxuan Liu; Wei Ting Chen; Qing Wang; Qi Guo
>
> **摘要:** We present a metasurface imaging system capable of simultaneously capturing two images at close range (1-2~cm) and an additional image at long range (about 40~cm) on a shared photosensor. The close-range image pair focuses at 1.4~cm and 2.0~cm, respectively, which forms a focal stack, enabling passive ranging with an accuracy of $\pm$1~mm from 12~mm to 20~mm through a computationally efficient depth-from-defocus algorithm for a simplified scenario. The entire system is compact, with a total track length of 15~mm, making it suitable for seamless integration into edge platforms for defense and other resource-constrained applications.
>
---
#### [new 330] ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出ProGAL-VLA模型，解决VLA模型语言忽视和指令不敏感问题，通过构建3D图、符号子目标和对比损失提升鲁棒性和实体检索效果。**

- **链接: [https://arxiv.org/pdf/2604.09824](https://arxiv.org/pdf/2604.09824)**

> **作者:** Nastaran Darabi; Amit Ranjan Trivedi
>
> **摘要:** Vision language action (VLA) models enable generalist robotic agents but often exhibit language ignorance, relying on visual shortcuts and remaining insensitive to instruction changes. We present Prospective Grounding and Alignment VLA (ProGAL-VLA), which constructs a 3D entity-centric graph (GSM), uses a slow planner to produce symbolic sub-goals, and aligns them with grounded entities via a Grounding Alignment Contrastive (GAC) loss. All actions are conditioned on a verified goal embedding $g_t$, whose attention entropy provides an intrinsic ambiguity signal. On LIBERO-Plus, ProGAL-VLA increases robustness under robot perturbations from 30.3 to 71.5 percent, reduces language ignorance by 3x-4x, and improves entity retrieval from 0.41 to 0.71 Recall@1. On the Custom Ambiguity Benchmark, it reaches AUROC 0.81 (vs. 0.52), AUPR 0.79, and raises clarification on ambiguous inputs from 0.09 to 0.81 without harming unambiguous success. The verification bottleneck increases mutual information of language-actions, the GAC loss imposes an entity-level InfoNCE bound, and attention entropy yields calibrated selective prediction, indicating that explicit verified grounding is an effective path toward instruction-sensitive, ambiguity-aware agents.
>
---
#### [new 331] Device-Conditioned Neural Architecture Search for Efficient Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决异构硬件部署效率低的问题。提出DC-QFA框架，通过设备条件量化训练和架构搜索，实现高效、通用的模型部署。**

- **链接: [https://arxiv.org/pdf/2604.10170](https://arxiv.org/pdf/2604.10170)**

> **作者:** Yiming Wu; Huan Wang; Zhenghao Chen; Ge Yuan; Dong Xu
>
> **备注:** 17 pages, 4 figures
>
> **摘要:** The growing complexity of visuomotor policies poses significant challenges for deployment with heterogeneous robotic hardware constraints. However, most existing model-efficient approaches for robotic manipulation are device- and model-specific, lack generalizability, and require time-consuming per-device optimization during the adaptation process. In this work, we propose a unified framework named \textbf{D}evice-\textbf{C}onditioned \textbf{Q}uantization-\textbf{F}or-\textbf{A}ll (DC-QFA) which amortizes deployment effort with the device-conditioned quantization-aware training and hardware-constrained architecture search. Specifically, we introduce a single supernet that spans a rich design space over network architectures and mixed-precision bit-widths. It is optimized with latency- and memory-aware regularization, guided by per-device lookup tables. With this supernet, for each target platform, we can perform a once-for-all lightweight search to select an optimal subnet without any per-device re-optimization, which enables more generalizable deployment across heterogeneous hardware, and substantially reduces deployment time. To improve long-horizon stability under low precision, we further introduce multi-step on-policy distillation to mitigate error accumulation during closed-loop execution. Extensive experiments on three representative policy backbones, such as DiffusionPolicy-T, MDT-V, and OpenVLA-OFT, demonstrate that our DC-QFA achieves $2\text{-}3\times$ acceleration on edge devices, consumer-grade GPUs, and cloud platforms, with negligible performance drop in task success. Real-world evaluations on an Inovo robot equipped with a force/torque sensor further validates that our low-bit DC-QFA policies maintain stable, contact-rich manipulation even under severe quantization.
>
---
#### [new 332] Efficient Personalization of Generative User Interfaces
- **分类: cs.LG; cs.AI; cs.CV; cs.HC**

- **简介: 该论文属于个性化任务，解决生成式用户界面的个性化难题。通过分析设计师偏好差异，提出一种高效偏好建模方法，提升界面生成的个性化效果。**

- **链接: [https://arxiv.org/pdf/2604.09876](https://arxiv.org/pdf/2604.09876)**

> **作者:** Yi-Hao Peng; Samarth Das; Jeffrey P. Bigham; Jason Wu
>
> **摘要:** Generative user interfaces (UIs) create new opportunities to adapt interfaces to individual users on demand, but personalization remains difficult because desirable UI properties are subjective, hard to articulate, and costly to infer from sparse feedback. We study this problem through a new dataset in which 20 trained designers each provide pairwise judgments over the same 600 generated UIs, enabling direct analysis of preference divergence. We find substantial disagreement across designers (average kappa = 0.25), and written rationales reveal that even when designers appeal to similar concepts such as hierarchy or cleanliness, designers differ in how they define, prioritize, and apply those concepts. Motivated by these findings, we develop a sample-efficient personalization method that represents a new user in terms of prior designers rather than a fixed rubric of design concepts. In a technical evaluation, our preference model outperforms both a pretrained UI evaluator and a larger multimodal model, and scales better with additional feedback. When used to personalize generation, it also produces interfaces preferred by 12 new designers over baseline approaches, including direct user prompting. Our findings suggest that lightweight preference elicitation can serve as a practical foundation for personalized generative UI systems.
>
---
#### [new 333] Zero-shot World Models Are Developmentally Efficient Learners
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出一种零样本视觉世界模型（ZWM），用于模拟儿童高效学习物理世界的能力。任务是理解物理场景，解决数据效率和泛化问题。工作包括构建基于因果推理的模型，实现多任务快速学习。**

- **链接: [https://arxiv.org/pdf/2604.10333](https://arxiv.org/pdf/2604.10333)**

> **作者:** Khai Loong Aw; Klemen Kotar; Wanhee Lee; Seungwoo Kim; Khaled Jedoui; Rahul Venkatesh; Lilian Naing Chen; Michael C. Frank; Daniel L.K. Yamins
>
> **摘要:** Young children demonstrate early abilities to understand their physical world, estimating depth, motion, object coherence, interactions, and many other aspects of physical scene understanding. Children are both data-efficient and flexible cognitive systems, creating competence despite extremely limited training data, while generalizing to myriad untrained tasks -- a major challenge even for today's best AI systems. Here we introduce a novel computational hypothesis for these abilities, the Zero-shot Visual World Model (ZWM). ZWM is based on three principles: a sparse temporally-factored predictor that decouples appearance from dynamics; zero-shot estimation through approximate causal inference; and composition of inferences to build more complex abilities. We show that ZWM can be learned from the first-person experience of a single child, rapidly generating competence across multiple physical understanding benchmarks. It also broadly recapitulates behavioral signatures of child development and builds brain-like internal representations. Our work presents a blueprint for efficient and flexible learning from human-scale data, advancing both a computational account for children's early physical understanding and a path toward data-efficient AI systems.
>
---
#### [new 334] ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出ClawGUI框架，解决GUI代理训练、评估与部署的问题，整合RL、标准化评估及跨平台部署，提升代理性能与实用性。**

- **链接: [https://arxiv.org/pdf/2604.11784](https://arxiv.org/pdf/2604.11784)**

> **作者:** Fei Tang; Zhiqiong Lu; Boxuan Zhang; Weiming Lu; Jun Xiao; Yueting Zhuang; Yongliang Shen
>
> **摘要:** GUI agents drive applications through their visual interfaces instead of programmatic APIs, interacting with arbitrary software via taps, swipes, and keystrokes, reaching a long tail of applications that CLI-based agents cannot. Yet progress in this area is bottlenecked less by modeling capacity than by the absence of a coherent full-stack infrastructure: online RL training suffers from environment instability and closed pipelines, evaluation protocols drift silently across works, and trained agents rarely reach real users on real devices. We present \textbf{ClawGUI}, an open-source framework addressing these three gaps within a single harness. \textbf{ClawGUI-RL} provides the first open-source GUI agent RL infrastructure with validated support for both parallel virtual environments and real physical devices, integrating GiGPO with a Process Reward Model for dense step-level supervision. \textbf{ClawGUI-Eval} enforces a fully standardized evaluation pipeline across 6 benchmarks and 11+ models, achieving 95.8\% reproduction against official baselines. \textbf{ClawGUI-Agent} brings trained agents to Android, HarmonyOS, and iOS through 12+ chat platforms with hybrid CLI-GUI control and persistent personalized memory. Trained end to end within this pipeline, \textbf{ClawGUI-2B} achieves 17.1\% Success Rate on MobileWorld GUI-Only, outperforming the same-scale MAI-UI-2B baseline by 6.0\%.
>
---
#### [new 335] Agentic Exploration of PDE Spaces using Latent Foundation Models for Parameterized Simulations
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于科学发现任务，旨在解决PDE驱动系统自动化探索难题。通过结合多智能体LLM与潜在基础模型，实现对流场参数的高效、连续探索与规律发现。**

- **链接: [https://arxiv.org/pdf/2604.09584](https://arxiv.org/pdf/2604.09584)**

> **作者:** Abhijeet Vishwasrao; Francisco Giral; Mahmoud Golestanian; Federica Tonti; Andrea Arroyo Ramo; Adrian Lozano-Duran; Steven L. Brunton; Sergio Hoyas; Soledad Le Clainche; Hector Gomez; Ricardo Vinuesa
>
> **摘要:** Flow physics and more broadly physical phenomena governed by partial differential equations (PDEs), are inherently continuous, high-dimensional and often chaotic in nature. Traditionally, researchers have explored these rich spatiotemporal PDE solution spaces using laboratory experiments and/or computationally expensive numerical simulations. This severely limits automated and large-scale exploration, unlike domains such as drug discovery or materials science, where discrete, tokenizable representations naturally interface with large language models. We address this by coupling multi-agent LLMs with latent foundation models (LFMs), a generative model over parametrised simulations, that learns explicit, compact and disentangled latent representations of flow fields, enabling continuous exploration across governing PDE parameters and boundary conditions. The LFM serves as an on-demand surrogate simulator, allowing agents to query arbitrary parameter configurations at negligible cost. A hierarchical agent architecture orchestrates exploration through a closed loop of hypothesis, experimentation, analysis and verification, with a tool-modular interface requiring no user support. Applied to flow past tandem cylinders at Re = 500, the framework autonomously evaluates over 1,600 parameter-location pairs and discovers divergent scaling laws: a regime-dependent two-mode structure for minimum displacement thickness and a robust linear scaling for maximum momentum thickness, with both landscapes exhibiting a dual-extrema structure that emerges at the near-wake to co-shedding regime transition. The coupling of the learned physical representations with agentic reasoning establishes a general paradigm for automated scientific discovery in PDE-governed systems.
>
---
#### [new 336] EvoDiagram: Agentic Editable Diagram Creation via Design Expertise Evolution
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文提出EvoDiagram，解决自动创建可编辑图表的任务，通过多智能体系统和设计知识演化机制，提升图表的结构一致性和美观性。**

- **链接: [https://arxiv.org/pdf/2604.09568](https://arxiv.org/pdf/2604.09568)**

> **作者:** Tianfu Wang; Leilei Ding; Ziyang Tao; Yi Zhan; Zhiyuan Ma; Wei Wu; Yuxuan Lei; Yuan Feng; Junyang Wang; Yin Wu; Yizhao Xu; Hongyuan Zhu; Qi Liu; Nicholas Jing Yuan; Yanyong Zhang; Hui Xiong
>
> **摘要:** High-fidelity diagram creation requires the complex orchestration of semantic topology, visual styling, and spatial layout, posing a significant challenge for automated systems. Existing methods also suffer from a representation gap: pixel-based models often lack precise control, while code-based synthesis limits intuitive flexibility. To bridge this gap, we introduce EvoDiagram, an agentic framework that generates object-level editable diagrams via an intermediate canvas schema. EvoDiagram employs a coordinated multi-agent system to decouple semantic intent from rendering logic, resolving conflicts across heterogeneous design layers. Additionally, we propose a design knowledge evolution mechanism that distills execution traces into a hierarchical memory of domain guidelines, enabling agents to retrieve context-aware expertise adaptively. We further release CanvasBench, a benchmark consisting of both data and metrics for canvas-based diagramming. Extensive experiments demonstrate that EvoDiagram exhibits excellent performance and balance against baselines in generating editable, structurally consistent, and aesthetically coherent diagrams. Our code is available at this https URL.
>
---
#### [new 337] VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文属于视觉与语言导航任务，解决虚假前提指令下的导航问题。通过构建VLN-NF基准和提出ROAM方法，提升代理在目标不存在时的探索与决策能力。**

- **链接: [https://arxiv.org/pdf/2604.10533](https://arxiv.org/pdf/2604.10533)**

> **作者:** Hung-Ting Su; Ting-Jun Wang; Jia-Fong Yeh; Min Sun; Winston H. Hsu
>
> **备注:** Accepted at ACL 2026. The first two authors contributed equally to the technical work
>
> **摘要:** Conventional Vision-and-Language Navigation (VLN) benchmarks assume instructions are feasible and the referenced target exists, leaving agents ill-equipped to handle false-premise goals. We introduce VLN-NF, a benchmark with false-premise instructions where the target is absent from the specified room and agents must navigate, gather evidence through in-room exploration, and explicitly output NOT-FOUND. VLN-NF is constructed via a scalable pipeline that rewrites VLN instructions using an LLM and verifies target absence with a VLM, producing plausible yet factually incorrect goals. We further propose REV-SPL to jointly evaluate room reaching, exploration coverage, and decision correctness. To address this challenge, we present ROAM, a two-stage hybrid that combines supervised room-level navigation with LLM/VLM-driven in-room exploration guided by a free-space clearance prior. ROAM achieves the best REV-SPL among compared methods, while baselines often under-explore and terminate prematurely under unreliable instructions. VLN-NF project page can be found at this https URL.
>
---
#### [new 338] Tipiano: Cascaded Piano Hand Motion Synthesis via Fingertip Priors
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于钢琴手部运动合成任务，解决物理方法僵硬、数据方法不准确的问题，通过四阶段框架提升运动精度与自然度。**

- **链接: [https://arxiv.org/pdf/2604.09692](https://arxiv.org/pdf/2604.09692)**

> **作者:** Joonhyung Bae; Kirak Kim; Hyeyoon Cho; Sein Lee; Yoon-Seok Choi; Hyeon Hur; Gyubin Lee; Akira Maezawa; Satoshi Obata; Jonghwa Park; Jaebum Park; Juhan Nam
>
> **摘要:** Synthesizing realistic piano hand motions requires both precision and naturalness. Physics-based methods achieve precision but produce stiff motions; data-driven models learn natural dynamics but struggle with positional accuracy. Piano motion exhibits a natural hierarchy: fingertip positions are nearly deterministic given piano geometry and fingering, while wrist and intermediate joints offer stylistic freedom. We present [OURS], a four-stage framework exploiting this hierarchy: (1) statistics-based fingertip positioning, (2) FiLM-conditioned trajectory refinement, (3) wrist estimation, and (4) STGCN-based pose synthesis. We contribute expert-annotated fingerings for the FürElise dataset (153 pieces, ~10 hours). Experiments demonstrate F1 = 0.910, substantially outperforming diffusion baselines (F1 = 0.121), with user study (N=41) confirming quality approaching motion capture. Expert evaluation by professional pianists (N=5) identified anticipatory motion as the key remaining gap, providing concrete directions for future improvement.
>
---
#### [new 339] Autonomous Diffractometry Enabled by Visual Reinforcement Learning
- **分类: cs.LG; cond-mat.mtrl-sci; cs.CV**

- **简介: 该论文属于材料科学中的晶体对准任务，解决人工依赖问题。通过视觉强化学习，实现无需晶体学知识的自动对准，提升实验效率。**

- **链接: [https://arxiv.org/pdf/2604.11773](https://arxiv.org/pdf/2604.11773)**

> **作者:** J. Oppliger; M. Stifter; A. Rüegg; I. Biało; L. Martinelli; P. G. Freeman; D. Prabhakaran; J. Zhao; Q. Wang; J. Chang
>
> **备注:** 20 pages, 16 figures
>
> **摘要:** Automation underpins progress across scientific and industrial disciplines. Yet, automating tasks requiring interpretation of abstract visual information remain challenging. For example, crystal alignment strongly relies on humans with the ability to comprehend diffraction patterns. Here we introduce an autonomous system that aligns single crystals without access to crystallography and diffraction theory. Using a model-free reinforcement learning framework, an agent learns to identify and navigate towards high-symmetry orientations directly from Laue diffraction patterns. Despite the absence of human supervision, the agent develops human-like strategies to achieve time-efficient alignment across different crystal symmetry classes. With this, we provide a computational framework for intelligent diffractometers. As such, our approach advances the development of automated experimental workflows in materials science.
>
---
#### [new 340] StarVLA-$α$: Reducing Complexity in Vision-Language-Action Systems
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言-动作系统任务，旨在简化VLA模型设计。通过构建简单基线StarVLA-α，研究关键设计因素，提升性能并减少复杂性。**

- **链接: [https://arxiv.org/pdf/2604.11757](https://arxiv.org/pdf/2604.11757)**

> **作者:** Jinhui Ye; Ning Gao; Senqiao Yang; Jinliang Zheng; Zixuan Wang; Yuxin Chen; Pengguang Chen; Yilun Chen; Shu Liu; Jiaya Jia
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as a promising paradigm for building general-purpose robotic agents. However, the VLA landscape remains highly fragmented and complex: as existing approaches vary substantially in architectures, training data, embodiment configurations, and benchmark-specific engineering. In this work, we introduce StarVLA-$\alpha$, a simple yet strong baseline designed to study VLA design choices under controlled conditions. StarVLA-$\alpha$ deliberately minimizes architectural and pipeline complexity to reduce experimental confounders and enable systematic analysis. Specifically, we re-evaluate several key design axes, including action modeling strategies, robot-specific pretraining, and interface engineering. Across unified multi-benchmark training on LIBERO, SimplerEnv, RoboTwin, and RoboCasa, the same simple baseline remains highly competitive, indicating that a strong VLM backbone combined with minimal design is already sufficient to achieve strong performance without relying on additional architectural complexity or engineering tricks. Notably, our single generalist model outperforms $\pi_{0.5}$ by 20\% on the public real-world RoboChallenge benchmark. We expect StarVLA-$\alpha$ to serve as a solid starting point for future research in the VLA regime. Code will be released at this https URL.
>
---
#### [new 341] ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机械臂物体操控任务，解决单目RGB下物体姿态估计与操控问题。通过3D高斯点云和强化学习，实现高效、鲁棒的物体翻转。**

- **链接: [https://arxiv.org/pdf/2604.11138](https://arxiv.org/pdf/2604.11138)**

> **作者:** Arjun Bhardwaj; Maximum Wilder-Smith; Mayank Mittal; Vaishakh Patil; Marco Hutter
>
> **摘要:** In-hand object reorientation requires precise estimation of the object pose to handle complex task dynamics. While RGB sensing offers rich semantic cues for pose tracking, existing solutions rely on multi-camera setups or costly ray tracing. We present a sim-to-real framework for monocular RGB in-hand reorientation that integrates 3D Gaussian Splatting (3DGS) to bridge the visual sim-to-real gap. Our key insight is performing domain randomization in the Gaussian representation space: by applying physically consistent, pre-rendering augmentations to 3D Gaussians, we generate photorealistic, randomized visual data for object pose estimation. The manipulation policy is trained using curriculum-based reinforcement learning with teacher-student distillation, enabling efficient learning of complex behaviors. Importantly, both perception and control models can be trained independently on consumer-grade hardware, eliminating the need for large compute clusters. Experiments show that the pose estimator trained with 3DGS data outperforms those trained using conventional rendering data in challenging visual environments. We validate the system on a physical multi-fingered hand equipped with an RGB camera, demonstrating robust reorientation of five diverse objects even under challenging lighting conditions. Our results highlight Gaussian splatting as a practical path for RGB-only dexterous manipulation. For videos of the hardware deployments and additional supplementary materials, please refer to the project website: this https URL
>
---
#### [new 342] LIDEA: Human-to-Robot Imitation Learning via Implicit Feature Distillation and Explicit Geometry Alignment
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于模仿学习任务，旨在解决人类与机器人之间身体差异导致的泛化问题。通过隐式特征蒸馏和显式几何对齐，提升从人类视频中学习机器人操作的效果。**

- **链接: [https://arxiv.org/pdf/2604.10677](https://arxiv.org/pdf/2604.10677)**

> **作者:** Yifu Xu; Bokai Lin; Xinyu Zhan; Hongjie Fang; Yong-Lu Li; Cewu Lu; Lixin Yang
>
> **摘要:** Scaling up robot learning is hindered by the scarcity of robotic demonstrations, whereas human videos offer a vast, untapped source of interaction data. However, bridging the embodiment gap between human hands and robot arms remains a critical challenge. Existing cross-embodiment transfer strategies typically rely on visual editing, but they often introduce visual artifacts due to intrinsic discrepancies in visual appearance and 3D geometry. To address these limitations, we introduce LIDEA (Implicit Feature Distillation and Explicit Geometric Alignment), an imitation learning framework in which policy learning benefits from human demonstrations. In the 2D visual domain, LIDEA employs a dual-stage transitive distillation pipeline that aligns human and robot representations in a shared latent space. In the 3D geometric domain, we propose an embodiment-agnostic alignment strategy that explicitly decouples embodiment from interaction geometry, ensuring consistent 3D-aware perception. Extensive experiments empirically validate LIDEA from two perspectives: data efficiency and OOD robustness. Results show that human data substitutes up to 80% of costly robot demonstrations, and the framework successfully transfers unseen patterns from human videos for out-of-distribution generalization.
>
---
#### [new 343] Brain-Grasp: Graph-based Saliency Priors for Improved fMRI-based Visual Brain Decoding
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于脑机接口中的视觉解码任务，旨在提升fMRI图像重建的结构和语义一致性。通过图结构引导的显著性先验，结合扩散模型，实现更准确的图像生成。**

- **链接: [https://arxiv.org/pdf/2604.10617](https://arxiv.org/pdf/2604.10617)**

> **作者:** Mohammad Moradi; Morteza Moradi; Marco Grassia; Giuseppe Mangioni
>
> **摘要:** Recent progress in brain-guided image generation has improved the quality of fMRI-based reconstructions; however, fundamental challenges remain in preserving object-level structure and semantic fidelity. Many existing approaches overlook the spatial arrangement of salient objects, leading to conceptually inconsistent outputs. We propose a saliency-driven decoding framework that employs graph-informed saliency priors to translate structural cues from brain signals into spatial masks. These masks, together with semantic information extracted from embeddings, condition a diffusion model to guide image regeneration, helping preserve object conformity while maintaining natural scene composition. In contrast to pipelines that invoke multiple diffusion stages, our approach relies on a single frozen model, offering a more lightweight yet effective design. Experiments show that this strategy improves both conceptual alignment and structural similarity to the original stimuli, while also introducing a new direction for efficient, interpretable, and structurally grounded brain decoding.
>
---
## 更新

#### [replaced 001] DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.11892](https://arxiv.org/pdf/2503.11892)**

> **作者:** Chengxuan Qian; Shuo Xing; Shawn Li; Yue Zhao; Zhengzhong Tu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Multimodal representation learning aims to capture both shared and complementary semantic information across multiple modalities. However, the intrinsic heterogeneity of diverse modalities presents substantial challenges to achieve effective cross-modal collaboration and integration. To address this, we introduce DecAlign, a novel hierarchical cross-modal alignment framework designed to decouple multimodal representations into modality-unique (heterogeneous) and modality-common (homogeneous) features. For handling heterogeneity, we employ a prototype-guided optimal transport alignment strategy leveraging gaussian mixture modeling and multi-marginal transport plans, thus mitigating distribution discrepancies while preserving modality-unique characteristics. To reinforce homogeneity, we ensure semantic consistency across modalities by aligning latent distribution matching with Maximum Mean Discrepancy regularization. Furthermore, we incorporate a multimodal transformer to enhance high-level semantic feature fusion, thereby further reducing cross-modal inconsistencies. Our extensive experiments on four widely used multimodal benchmarks demonstrate that DecAlign consistently outperforms existing state-of-the-art methods across five metrics. These results highlight the efficacy of DecAlign in enhancing superior cross-modal alignment and semantic consistency while preserving modality-unique features, marking a significant advancement in multimodal representation learning scenarios. Our project page is at this https URL.
>
---
#### [replaced 002] WiFlow: A Lightweight WiFi-based Continuous Human Pose Estimation Network with Spatio-Temporal Feature Decoupling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.08661](https://arxiv.org/pdf/2602.08661)**

> **作者:** Yi Dao; Lankai Zhang; Hao Liu; Haiwei Zhang; Wenbo Wang
>
> **摘要:** Human pose estimation is fundamental to intelligent perception in the Internet of Things (IoT), enabling applications ranging from smart healthcare to human-computer interaction. While WiFi-based methods have gained traction, they often struggle with continuous motion and high computational overhead. This work presents WiFlow, a novel framework for continuous human pose estimation using WiFi signals. Unlike vision-based approaches such as two-dimensional deep residual networks that treat Channel State Information (CSI) as images, WiFlow employs an encoder-decoder architecture. The encoder captures spatio-temporal features of CSI using temporal and asymmetric convolutions, preserving the original sequential structure of signals. It then refines keypoint features of human bodies to be tracked and capture their structural dependencies via axial attention. The decoder subsequently maps the encoded high-dimensional features into keypoint coordinates. Trained on a self-collected dataset of 360,000 synchronized CSI-pose samples from 5 subjects performing continuous sequences of 8 daily activities, WiFlow achieves a Percentage of Correct Keypoints (PCK) of 97.25% at a threshold of 20% (PCK@20) and 99.48% at PCK@50, with a mean per-joint position error of 0.007 m. With only 2.23M parameters, WiFlow significantly reduces model complexity and computational cost, establishing a new performance baseline for practical WiFi-based human pose estimation. Our code and datasets are available at this https URL.
>
---
#### [replaced 003] Delta Rectified Flow Sampling for Text-to-Image Editing
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.05342](https://arxiv.org/pdf/2509.05342)**

> **作者:** Gaspard Beaudouin; Minghan Li; Jaeyeon Kim; Sung-Hoon Yoon; Mengyu Wang
>
> **摘要:** We propose Delta Rectified Flow Sampling (DRFS), a novel inversion-free, path-aware editing framework within rectified flow models for text-to-image editing. DRFS is a distillation-based method that explicitly models the discrepancy between the source and target velocity fields in order to mitigate over-smoothing artifacts rampant in prior distillation sampling approaches. We further introduce a time-dependent shift term to push noisy latents closer to the target trajectory, enhancing the alignment with the target distribution. We theoretically demonstrate that disabling this shift recovers Delta Denoising Score (DDS), bridging score-based diffusion optimization and velocity-based rectified-flow optimization. Moreover, under rectified-flow dynamics, a linear shift schedule recovers the inversion-free method FlowEdit as a strict special case, yielding a unifying view of optimization and ODE editing. We conduct an analysis to guide the design of our shift term, and experimental results on the widely used PIE Benchmark indicate that DRFS achieves superior editing quality, fidelity, and controllability while requiring no architectural modifications. Code is available at this https URL.
>
---
#### [replaced 004] Integrating Semi-Supervised and Active Learning for Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2501.19227](https://arxiv.org/pdf/2501.19227)**

> **作者:** Wanli Ma; Oktay Karakus; Paul L. Rosin
>
> **摘要:** In this paper, we propose a novel active learning approach integrated with an improved semi-supervised learning framework to reduce the cost of manual annotation and enhance model performance. Our proposed approach effectively leverages both the labelled data selected through active learning and the unlabelled data excluded from the selection process. The proposed active learning approach pinpoints areas where the pseudo-labels are likely to be inaccurate. Then, an automatic and efficient pseudo-label auto-refinement (PLAR) module is proposed to correct pixels with potentially erroneous pseudo-labels by comparing their feature representations with those of labelled regions. This approach operates without increasing the labelling budget and is based on the cluster assumption, which states that pixels belonging to the same class should exhibit similar representations in feature space. Furthermore, manual labelling is only applied to the most difficult and uncertain areas in unlabelled data, where insufficient information prevents the PLAR module from making a decision. We evaluated the proposed hybrid semi-supervised active learning framework on two benchmark datasets, one from natural and the other from remote sensing imagery domains. In both cases, it outperformed state-of-the-art methods in the semantic segmentation task.
>
---
#### [replaced 005] Variational Visual Question Answering for Uncertainty-Aware Selective Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.09591](https://arxiv.org/pdf/2505.09591)**

> **作者:** Tobias Jan Wieczorek; Nathalie Daun; Mohammad Emtiyaz Khan; Marcus Rohrbach
>
> **备注:** TMLR April 2026 version. 13 pages main paper, 31 pages with appendix. Updated bibliography
>
> **摘要:** Despite remarkable progress in recent years, Vision Language Models (VLMs) remain prone to overconfidence and hallucinations on tasks such as Visual Question Answering (VQA) and Visual Reasoning. Bayesian methods can potentially improve reliability by helping models predict selectively, that is, models respond only when they are sufficiently confident. Unfortunately, such approaches can be costly and ineffective for large models, and there exists little evidence to show otherwise for multimodal applications. Here, we show for the first time the effectiveness and competitive edge of variational Bayes for selective prediction in VQA. We build on recent advances in variational methods for deep learning and propose an extension called "Variational VQA". This method improves calibration and yields significant gains for selective prediction on VQA and Visual Reasoning, particularly when the error tolerance is low ($\leq 1\%$). Often, just one posterior sample yields more reliable answers than those given by models trained with AdamW. In addition, we propose a new risk-averse selector that outperforms standard sample averaging by considering the variance of predictions. Overall, we present compelling evidence that variational learning is a viable option to make large VLMs safer and more trustworthy.
>
---
#### [replaced 006] LookBench: A Live and Holistic Open Benchmark for Fashion Image Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.14706](https://arxiv.org/pdf/2601.14706)**

> **作者:** Gensmo.ai; Chao Gao; Siqiao Xue; Jiwen Fu; Tingyi Gu; Shanshan Li; Fan Zhou
>
> **备注:** The first two authors contributed equally to this work. Project site: this https URL
>
> **摘要:** In this paper, we present LookBench (We use the term "look" to reflect retrieval that mirrors how people shop -- finding the exact item, a close substitute, or a visually consistent alternative.), a live, holistic and challenging benchmark for fashion image retrieval in real e-commerce settings. LookBench includes both recent product images sourced from live websites and AI-generated fashion images, reflecting contemporary trends and use cases. Each test sample is time-stamped and we intend to update the benchmark periodically, enabling contamination-aware evaluation aligned with declared training cutoffs. Grounded in our fine-grained attribute taxonomy, LookBench covers single-item and outfit-level retrieval across. Our experiments reveal that LookBench poses a significant challenge on strong baselines, with many models achieving below $60\%$ Recall@1. Our proprietary model achieves the best performance on LookBench, and we release an open-source counterpart that ranks second, with both models attaining state-of-the-art results on legacy Fashion200K evaluations. LookBench is designed to be updated semi-annually with new test samples and progressively harder task variants, providing a durable measure of progress. We publicly release our leaderboard, dataset, evaluation code, and trained models.
>
---
#### [replaced 007] HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10128](https://arxiv.org/pdf/2603.10128)**

> **作者:** Daichao Zhao; Qiupu Chen; Feng He; Xin Ning; Qiankun Li
>
> **备注:** Accepted by CVPR 2026 (HighLight)
>
> **摘要:** Lane detection is a crucial task in autonomous driving, as it helps ensure the safe operation of vehicles. However, existing datasets such as CULane and TuSimple contain relatively limited data under extreme weather conditions, including rain, snow, and fog. As a result, detection models trained on these datasets often become unreliable in such environments, which may lead to serious safety-critical failures on the road. To address this issue, we propose HG-Lane, a High-fidelity Generation framework for Lane Scenes under adverse weather and lighting conditions without requiring re-annotation. Based on this framework, we further construct a benchmark that includes adverse weather and lighting scenarios, containing 30,000 images. Experimental results demonstrate that our method consistently and significantly improves the performance of existing lane detection networks. For example, using the state-of-the-art CLRNet, the overall mF1 score on our benchmark increases by 20.87 percent. The F1@50 score for the overall, normal, snow, rain, fog, night, and dusk categories increases by 19.75 percent, 8.63 percent, 38.8 percent, 14.96 percent, 26.84 percent, 21.5 percent, and 12.04 percent, respectively. The code and dataset are available at: this https URL.
>
---
#### [replaced 008] RealSR-R1: Reinforcement Learning for Real-World Image Super-Resolution with Vision-Language Chain-of-Thought
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.16796](https://arxiv.org/pdf/2506.16796)**

> **作者:** Junbo Qiao; Miaomiao Cai; Wei Li; Xudong Huang; Jie Hu; Xinghao Chen; Shaohui Lin; Hongkai Xiong
>
> **摘要:** Real-World Image Super-Resolution is one of the most challenging task in image restoration. However, existing methods struggle with an accurate understanding of degraded image content, leading to reconstructed results that are both low-fidelity and unnatural. We present RealSR-R1 in this work, which empowers the RealSR models with understanding and reasoning capabilities. Inspired by the success of Chain of Thought (CoT) in large language models (LLMs), we simulate the human process of handling degraded images and propose the VLCoT framework, which integrates vision and language reasoning. The framework aims to precisely restore image details by progressively generating more comprehensive text and higher-resolution images. To overcome the challenge of traditional supervised learning CoT failing to generalize to real-world scenarios, we introduce, for the first time, Group Relative Policy Optimization (GRPO) into the Real-World Image Super-Resolution task. We propose VLCoT-GRPO as a solution, which designs four reward functions: (1) Format reward, used to standardize the CoT process; (2) Degradation reward, to incentivize accurate degradation estimation; (3) Understanding reward, to ensure the accuracy of the generated content; and (4) Generation reward, where we propose using a visual expert model to evaluate the quality of generated images, encouraging the model to generate more realistic images. Extensive experiments demonstrate that our proposed RealSR-R1 can generate realistic details and accurately understand image content, particularly in semantically rich scenes or images with severe degradation.
>
---
#### [replaced 009] S4M: 4-points to Segment Anything
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.05534](https://arxiv.org/pdf/2503.05534)**

> **作者:** Adrien Meyer; Lorenzo Arboit; Giuseppe Massimiani; Shih-Min Yin; Didier Mutter; Nicolas Padoy
>
> **摘要:** Purpose: The Segment Anything Model (SAM) promises to ease the annotation bottleneck in medical segmentation, but overlapping anatomy and blurred boundaries make its point prompts ambiguous, leading to cycles of manual refinement to achieve precise masks. Better prompting strategies are needed. Methods: We propose a structured prompting strategy using 4 points as a compact instance-level shape description. We study two 4-point variants: extreme points and the proposed major/minor axis endpoints, inspired by ultrasound measurement practice. SAM cannot fully exploit such structured prompts because it treats all points identically and lacks geometry-aware reasoning. To address this, we introduce S4M (4-points to Segment Anything), which augments SAM to interpret 4 points as relational cues rather than isolated clicks. S4M expands the prompt space with role-specific embeddings and adds an auxiliary "Canvas" pretext task that sketches coarse masks directly from prompts, fostering geometry-aware reasoning. Results: Across eight datasets in ultrasound and surgical endoscopy, S4M improves segmentation by +3.42 mIoU over a strong SAM baseline at equal prompt budget. An annotation study with three clinicians further shows that major/minor prompts enable faster annotation. Conclusion: S4M increases performance, reduces annotation effort, and aligns prompting with clinical practice, enabling more scalable dataset development in medical imaging. We release our code and pretrained models at this https URL.
>
---
#### [replaced 010] Bidirectional Cross-Attention Fusion of High-Res RGB and Low-Res HSI for Multimodal Automated Waste Sorting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.13941](https://arxiv.org/pdf/2603.13941)**

> **作者:** Jonas V. Funk; Lukas Roming; Andreas Michel; Paul Bäcker; Georg Maier; Thomas Längle; Markus Klute
>
> **备注:** Submitted to Information Fusion (Elsevier). 23 pages, 10 figures, 7 tables
>
> **摘要:** Growing waste streams and the transition to a circular economy require efficient automated waste sorting. In industrial settings, materials move on fast conveyor belts, where reliable identification and ejection demand pixel-accurate segmentation. RGB imaging delivers high-resolution spatial detail, which is essential for accurate segmentation, but it confuses materials that look similar in the visible spectrum. Hyperspectral imaging (HSI) provides spectral signatures that separate such materials, yet its lower spatial resolution limits detail. Effective waste sorting therefore needs methods that fuse both modalities to exploit their complementary strengths. We present Bidirectional Cross-Attention Fusion (BCAF), which aligns high-resolution RGB with low-resolution HSI at their native grids via localized, bidirectional cross-attention, avoiding pre-upsampling or early spectral collapse. BCAF uses two independent backbones: a standard Swin Transformer for RGB and an HSI-adapted Swin backbone that preserves spectral structure through 3D tokenization with spectral self-attention. We also analyze trade-offs between RGB input resolution and the number of HSI spectral slices. Although our evaluation targets RGB-HSI fusion, BCAF is modality-agnostic and applies to co-registered RGB with lower-resolution, high-channel auxiliary sensors. On the benchmark SpectralWaste dataset, BCAF achieves state-of-the-art performance of 76.4% mIoU at 31 images/s and 75.4% mIoU at 55 images/s. We further evaluate a novel industrial dataset: K3I-Cycling (first RGB subset already released on Fordatis). On this dataset, BCAF reaches 62.3% mIoU for material segmentation (paper, metal, plastic, etc.) and 66.2% mIoU for plastic-type segmentation (PET, PP, HDPE, LDPE, PS, etc.). Code and model checkpoints are publicly available at this https URL .
>
---
#### [replaced 011] Enhancing Geo-localization for Crowdsourced Flood Imagery via LLM-Guided Attention
- **分类: cs.CL; cs.AI; cs.CV; cs.CY**

- **简介: 该论文属于图像地理定位任务，旨在解决社交媒体洪水图像缺乏可靠地理信息的问题。通过引入LLM引导的注意力机制，提升VPR模型的定位性能。**

- **链接: [https://arxiv.org/pdf/2512.11811](https://arxiv.org/pdf/2512.11811)**

> **作者:** Fengyi Xu; Jun Ma; Waishan Qiu; Cui Guo; Jack C.P. Cheng
>
> **备注:** Updated author list to include additional contributor. Revised title and improved methodology section based on collaborative feedback
>
> **摘要:** Crowdsourced social media imagery provides real-time visual evidence of urban flooding but often lacks reliable geographic metadata for emergency response. Existing Visual Place Recognition (VPR) models struggle to geo-localize these images due to cross-source domain shifts and visual distortions. We present VPR-AttLLM, a model-agnostic framework integrating the semantic reasoning and geospatial knowledge of Large Language Models (LLMs) into VPR pipelines via attention-guided descriptor enhancement. VPR-AttLLM uses LLMs to isolate location-informative regions and suppress transient noise, improving retrieval without model retraining or new data. We evaluate this framework across San Francisco and Hong Kong using established queries, synthetic flooding scenarios, and real social media flood images. Integrating VPR-AttLLM with state-of-the-art models (CosPlace, EigenPlaces, SALAD) consistently improves recall, yielding 1-3% relative gains and up to 8% on challenging real flood imagery. By embedding urban perception principles into attention mechanisms, VPR-AttLLM bridges human-like spatial reasoning with modern VPR architectures. Its plug-and-play design and cross-source robustness offer a scalable solution for rapid geo-localization of crowdsourced crisis imagery, advancing cognitive urban resilience.
>
---
#### [replaced 012] THOM: Generating Physically Plausible Hand-Object Meshes From Text
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.02736](https://arxiv.org/pdf/2604.02736)**

> **作者:** Uyoung Jeong; Yihalem Yimolal Tiruneh; Hyung Jin Chang; Seungryul Baek; Kwang In Kim
>
> **备注:** accepted to CVPR Findings 2026
>
> **摘要:** Generating photorealistic 3D hand-object interactions (HOIs) from text is important for applications like robotic grasping and AR/VR content creation. In practice, however, achieving both visual fidelity and physical plausibility remains difficult, as mesh extraction from text-generated Gaussians is inherently ill-posed and the resulting meshes are often unreliable for physics-based optimization. We present THOM, a training-free framework that generates physically plausible 3D HOI meshes directly from text prompts, without requiring template object meshes. THOM follows a two-stage pipeline: it first generates hand and object Gaussians guided by text, and then refines their interaction using physics-based optimization. To enable reliable interaction modeling, we introduce a mesh extraction method with an explicit vertex-to-Gaussian mapping, which enables topology-aware regularization. We further improve physical plausibility through contact-aware optimization and vision-language model (VLM)-guided translation refinement. Extensive experiments show that THOM produces high-quality HOIs with strong text alignment, visual realism, and interaction plausibility.
>
---
#### [replaced 013] Affostruction: 3D Affordance Grounding with Generative Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.09211](https://arxiv.org/pdf/2601.09211)**

> **作者:** Chunghyun Park; Seunghyeon Lee; Minsu Cho
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** This paper addresses the problem of affordance grounding from RGBD images of an object, which aims to localize surface regions corresponding to a text query that describes an action on the object. While existing methods predict affordance regions only on visible surfaces, we propose Affostruction, a generative framework that reconstructs complete object geometry from partial RGBD observations and grounds affordances on the full shape including unobserved regions. Our approach introduces sparse voxel fusion of multi-view features for constant-complexity generative reconstruction, a flow-based formulation that captures the inherent ambiguity of affordance distributions, and an active view selection strategy guided by predicted affordances. Affostruction outperforms existing methods by large margins on challenging benchmarks, achieving 19.1 aIoU on affordance grounding and 32.67 IoU for 3D reconstruction.
>
---
#### [replaced 014] Dark-EvGS: Event Camera as an Eye for Radiance Field in the Dark
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.11931](https://arxiv.org/pdf/2507.11931)**

> **作者:** Jingqian Wu; Peiqi Duan; Zongqiang Wang; Changwei Wang; Boxin Shi; Edmund Y. Lam
>
> **摘要:** In low-light environments, conventional cameras often struggle to capture clear multi-view images of objects due to dynamic range limitations and motion blur caused by long exposure. Event cameras, with their high-dynamic range and high-speed properties, have the potential to mitigate these issues. Additionally, 3D Gaussian Splatting (GS) enables radiance field reconstruction, facilitating bright frame synthesis from multiple viewpoints in low-light conditions. However, naively using an event-assisted 3D GS approach still faced challenges because, in low light, events are noisy, frames lack quality, and the color tone may be inconsistent. To address these issues, we propose Dark-EvGS, the first event-assisted 3D GS framework that enables the reconstruction of bright frames from arbitrary viewpoints along the camera trajectory. Triplet-level supervision is proposed to gain holistic knowledge, granular details, and sharp scene rendering. The color tone matching block is proposed to guarantee the color consistency of the rendered frames. Furthermore, we introduce the first real-captured dataset for the event-guided bright frame synthesis task via 3D GS-based radiance field reconstruction. Experiments demonstrate that our method achieves better results than existing methods, conquering radiance field reconstruction under challenging low-light conditions. The code and sample data are included in the supplementary material.
>
---
#### [replaced 015] MerNav: A Highly Generalizable Memory-Execute-Review Framework for Zero-Shot Object Goal Navigation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在提升零样本目标导航的成功率和泛化能力。提出MerNav框架，结合记忆、执行与回顾模块，有效提升导航性能。**

- **链接: [https://arxiv.org/pdf/2602.05467](https://arxiv.org/pdf/2602.05467)**

> **作者:** Dekang Qi; Shuang Zeng; Xinyuan Chang; Feng Xiong; Shichao Xie; Xiaolong Wu; Mu Xu
>
> **备注:** 9 pages, 2 figures, 5 tables, conference
>
> **摘要:** Visual Language Navigation (VLN) is one of the fundamental capabilities for embodied intelligence and a critical challenge that urgently needs to be addressed. However, existing methods are still unsatisfactory in terms of both success rate (SR) and generalization: Supervised Fine-Tuning (SFT) approaches typically achieve higher SR, while Training-Free (TF) approaches often generalize better, but it is difficult to obtain both simultaneously. To this end, we propose a Memory-Execute-Review framework. It consists of three parts: a hierarchical memory module for providing information support, an execute module for routine decision-making and actions, and a review module for handling abnormal situations and correcting behavior. We validated the effectiveness of this framework on the Object Goal Navigation task. Across 4 datasets, our average SR achieved absolute improvements of 7% and 5% compared to all baseline methods under TF and Zero-Shot (ZS) settings, respectively. On the most commonly used HM3D_v0.1 and the more challenging open vocabulary dataset HM3D_OVON, the SR improved by 8% and 6%, under ZS settings. Furthermore, on the MP3D and HM3D_OVON datasets, our method not only outperformed all TF methods but also surpassed all SFT methods, achieving comprehensive leadership in both SR (5% and 2%) and generalization. Additionally, we deployed the MerNav model on the humanoid robot and conducted experiments in the real world. The project address is: this https URL
>
---
#### [replaced 016] Flow Gym: A framework for the development, benchmarking, training, and deployment of flow-field quantification methods
- **分类: physics.flu-dyn; cs.CV; cs.SE; physics.comp-ph**

- **链接: [https://arxiv.org/pdf/2512.20642](https://arxiv.org/pdf/2512.20642)**

> **作者:** Francesco Banelli; Antonio Terpin; Alan Bonomi; Raffaello D'Andrea
>
> **备注:** Code: this https URL. Published in SoftwareX
>
> **摘要:** Particle image velocimetry (PIV) and related optical-flow methods are widely used to quantify fluid motion, but their development and evaluation are often hindered by fragmented software, inconsistent interfaces, and limited reproducibility. To address these challenges, we present Flow Gym, a framework for developing, benchmarking, training, and deploying flow-field quantification methods, with a primary focus on PIV. Its core contribution is a standardized interface that allows classical and learning-based algorithms to be integrated, compared, and deployed within a common pipeline. The framework includes JAX implementations and wrappers for existing methods, modular pre-processing and post-processing components, and utilities for training and benchmarking. By leveraging JAX, Flow Gym supports hardware-accelerated execution while remaining interoperable with external implementations from libraries such as OpenCV and PyTorch. It can operate on both synthetic and experimental data and supports the same workflow for offline benchmarking and real-time deployment. Flow Gym is designed to improve reproducibility, reduce barriers to method development, and facilitate the translation of flow-field quantification algorithms from research to experimental settings.
>
---
#### [replaced 017] CraftGraffiti: Exploring Human Identity with Custom Graffiti Art via Facial-Preserving Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.20640](https://arxiv.org/pdf/2508.20640)**

> **作者:** Ayan Banerjee; Fernando Vilariño; Josep Lladós
>
> **摘要:** Preserving facial identity under extreme stylistic transformation remains a major challenge in generative art. In graffiti, a high-contrast, abstract medium, subtle distortions to the eyes, nose, or mouth can erase the subject's recognizability, undermining both personal and cultural authenticity. We present CraftGraffiti, an end-to-end text-guided graffiti generation framework designed with facial feature preservation as a primary objective. Given an input image and a style and pose descriptive prompt, CraftGraffiti first applies graffiti style transfer via LoRA-fine-tuned pretrained diffusion transformer, then enforces identity fidelity through a face-consistent self-attention mechanism that augments attention layers with explicit identity embeddings. Pose customization is achieved without keypoints, using CLIP-guided prompt extension to enable dynamic re-posing while retaining facial coherence. We formally justify and empirically validate the "style-first, identity-after" paradigm, showing it reduces attribute drift compared to the reverse order. Quantitative results demonstrate competitive facial feature consistency and state-of-the-art aesthetic and human preference scores, while qualitative analyses and a live deployment at the Cruilla Festival highlight the system's real-world creative impact. CraftGraffiti advances the goal of identity-respectful AI-assisted artistry, offering a principled approach for blending stylistic freedom with recognizability in creative AI applications.
>
---
#### [replaced 018] VisText-Mosquito: A Unified Multimodal Dataset for Visual Detection, Segmentation, and Textual Explanation on Mosquito Breeding Sites
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VisText-Mosquito数据集，用于蚊虫滋生地的视觉检测、分割和文本解释，解决蚊媒疾病预防问题。**

- **链接: [https://arxiv.org/pdf/2506.14629](https://arxiv.org/pdf/2506.14629)**

> **作者:** Md. Adnanul Islam; Md. Faiyaz Abdullah Sayeedi; Md. Asaduzzaman Shuvo; Shahanur Rahman Bappy; Md Asiful Islam; Swakkhar Shatabda
>
> **备注:** Accepted at CVPRW 2026
>
> **摘要:** Mosquito-borne diseases pose a major global health risk, requiring early detection and proactive control of breeding sites to prevent outbreaks. In this paper, we present VisText-Mosquito, a multimodal dataset that integrates visual and textual data to support automated detection, segmentation, and explanation for mosquito breeding site analysis. The dataset includes 1,828 annotated images for object detection, 142 images for water surface segmentation, and natural language explanation texts linked to each image. The YOLOv9s model achieves the highest precision of 0.92926 and mAP@50 of 0.92891 for object detection, while YOLOv11n-Seg reaches a segmentation precision of 0.91587 and mAP@50 of 0.79795. For textual explanation generation, we tested a range of large vision-language models (LVLMs) in both zero-shot and few-shot settings. Our fine-tuned Mosquito-LLaMA3-8B model achieved the best results, with a final loss of 0.0028, a BLEU score of 54.7, BERTScore of 0.91, and ROUGE-L of 0.85. This dataset and model framework emphasize the theme "Prevention is Better than Cure", showcasing how AI-based detection can proactively address mosquito-borne disease risks. The dataset and implementation code are publicly available at GitHub: this https URL
>
---
#### [replaced 019] Eevee: Towards Close-up High-resolution Video-based Virtual Try-on
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18957](https://arxiv.org/pdf/2511.18957)**

> **作者:** Jianhao Zeng; Yancheng Bai; Ruidong Chen; Xuanpu Zhang; Lei Sun; Dongyang Jin; Ryan Xu; Nannan Zhang; Dan Song; Xiangxiang Chu
>
> **摘要:** Video virtual try-on technology provides a cost-effective solution for creating marketing videos in fashion e-commerce. However, its practical adoption is hindered by two critical limitations. First, the reliance on a single garment image as input in current virtual try-on datasets limits the accurate capture of realistic texture details. Second, most existing methods focus solely on generating full-shot virtual try-on videos, neglecting the business's demand for videos that also provide detailed close-ups. To address these challenges, we introduce a high-resolution dataset for video-based virtual try-on. This dataset offers two key features. First, it provides more detailed information on the garments, which includes high-fidelity images with detailed close-ups and textual descriptions; Second, it uniquely includes full-shot and close-up try-on videos of real human models. Furthermore, accurately assessing consistency becomes significantly more critical for the close-up videos, which demand high-fidelity preservation of garment details. To facilitate such fine-grained evaluation, we propose a new garment consistency metric VGID (Video Garment Inception Distance) that quantifies the preservation of both texture and structure. Our experiments validate these contributions. We demonstrate that by utilizing the detailed images from our dataset, existing video generation models can extract and incorporate texture features, significantly enhancing the realism and detail fidelity of virtual try-on results. Furthermore, we conduct a comprehensive benchmark of recent models. The benchmark effectively identifies the texture and structural preservation problems among current methods.
>
---
#### [replaced 020] ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ActDistill，解决VLA模型计算开销大、推理延迟高的问题，通过知识蒸馏将强模型能力迁移到轻量模型，提升效率。**

- **链接: [https://arxiv.org/pdf/2511.18082](https://arxiv.org/pdf/2511.18082)**

> **作者:** Wencheng Ye; Tianshi Wang; Lei Zhu; Fengling Li; Guoli Yang; Hengtao Shen
>
> **摘要:** Recent Vision-Language-Action (VLA) models have shown impressive flexibility and generalization, yet their deployment in robotic manipulation remains limited by heavy computational overhead and inference latency. In this work, we present ActDistill, a general action-guided self-derived distillation framework that transfers the action prediction capability of any existing VLA model to a lightweight counterpart. Unlike previous efficiency strategies that primarily emphasize vision-language correlations, ActDistill leverages action priors to guide knowledge transfer and model compression, achieving action-oriented efficiency for VLA models. Specifically, we employ a well-trained VLA model as the teacher and introduce a graph-structured encapsulation strategy to explicitly model the hierarchical evolution of action prediction. The student model, derived from the graph-encapsulated teacher, is further equipped with a dynamic router that adaptively selects computation paths based on action prediction demands, guided by hierarchical graph-informed supervision to ensure smooth and efficient evolution. During inference, graph-related auxiliary components are removed, allowing the student to execute only dynamically routed layers and predict high-precision actions with minimal computation and latency. Experiments on embodied benchmarks demonstrate that ActDistill achieves comparable or superior performance to full-scale VLA models while reducing computation by over 50% with up to 1.67 times speedup, thereby establishing a general paradigm toward efficient embodied intelligence.
>
---
#### [replaced 021] KiseKloset for Fashion Retrieval and Recommendation
- **分类: cs.IR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.23471](https://arxiv.org/pdf/2506.23471)**

> **作者:** Thanh-Tung Phan-Nguyen; Khoi-Nguyen Nguyen-Ngoc; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** The global fashion e-commerce industry has become integral to people's daily lives, leveraging technological advancements to offer personalized shopping experiences, primarily through recommendation systems that enhance customer engagement through personalized suggestions. To improve customers' experience in online shopping, we propose a novel comprehensive KiseKloset system for outfit retrieval and recommendation. We explore two approaches for outfit retrieval: similar item retrieval and text feedback-guided item retrieval. Notably, we introduce a novel transformer architecture designed to recommend complementary items from diverse categories. Furthermore, we enhance the overall performance of the search pipeline by integrating approximate algorithms to optimize the search process. Additionally, addressing the crucial needs of online shoppers, we employ a lightweight yet efficient virtual try-on framework capable of real-time operation, memory efficiency, and maintaining realistic outputs compared to its predecessors. This virtual try-on module empowers users to visualize specific garments on themselves, enhancing the customers' experience and reducing costs associated with damaged items for retailers. We deployed our end-to-end system for online users to test and provide feedback, enabling us to measure their satisfaction levels. The results of our user study revealed that 84% of participants found our comprehensive system highly useful, significantly improving their online shopping experience.
>
---
#### [replaced 022] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出LaMI方法，通过晚融合多图像提升大语言模型的常识推理能力，解决视觉与文本结合任务中的性能不足问题。**

- **链接: [https://arxiv.org/pdf/2406.13621](https://arxiv.org/pdf/2406.13621)**

> **作者:** Guy Yariv; Idan Schwartz; Yossi Adi; Sagie Benaim
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Commonsense reasoning often requires both textual and visual knowledge, yet Large Language Models (LLMs) trained solely on text lack visual grounding (e.g., "what color is an emperor penguin's belly?"). Visual Language Models (VLMs) perform better on visually grounded tasks but face two limitations: (i) often reduced performance on text-only commonsense reasoning compared to text-trained LLMs, and (ii) adapting newly released LLMs to vision input typically requires costly multimodal training. An alternative augments LLMs with test-time visual signals, improving visual commonsense without harming textual reasoning, but prior designs often rely on early fusion and a single image, which can be suboptimal. We propose a late multi-image fusion method: multiple images are generated from the text prompt with a lightweight parallel sampling, and their prediction probabilities are combined with those of a text-only LLM through a late-fusion layer that integrates projected visual features just before the final prediction. Across visual commonsense and NLP benchmarks, our method significantly outperforms augmented LLMs on visual reasoning, matches VLMs on vision-based tasks, and, when applied to strong LLMs such as LLaMA 3, also improves NLP performance while adding only modest test-time overhead. Project page is available at: this https URL.
>
---
#### [replaced 023] ProPhy: Progressive Physical Alignment for Dynamic World Simulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05564](https://arxiv.org/pdf/2512.05564)**

> **作者:** Zijun Wang; Panwen Hu; Jing Wang; Terry Jingchen Zhang; Yuhao Cheng; Long Chen; Yiqiang Yan; Zutao Jiang; Hanhui Li; Xiaodan Liang
>
> **摘要:** Recent advances in video generation have shown remarkable potential for constructing world simulators. However, current models still struggle to produce physically consistent results, particularly when handling large-scale or complex dynamics. This limitation arises primarily because existing approaches respond isotropically to physical prompts and neglect the fine-grained alignment between generated content and localized physical cues. To address these challenges, we propose ProPhy, a Progressive Physical Alignment Framework that enables explicit physics-aware conditioning and anisotropic generation. ProPhy employs a two-stage Mixture-of-Physics-Experts mechanism for discriminative physical prior extraction, where Semantic Experts infer semantic-level physical principles from textual descriptions, and Refinement Experts capture token-level physical dynamics. This mechanism allows the model to learn fine-grained, physics-aware video representations that better reflect underlying physical laws. Furthermore, we introduce a physical alignment strategy that transfers the physical reasoning capabilities of vision-language models into the Refinement Experts, facilitating a more accurate representation of dynamic physical phenomena. Extensive experiments on physics-aware video generation benchmarks demonstrate that ProPhy produces more realistic, dynamic, and physically coherent results than existing state-of-the-art methods.
>
---
#### [replaced 024] Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于软体连续机器人动力学建模任务，旨在解决视频数据中模型不可解释的问题。提出ABCD和VONs，实现可视觉解释的动力学建模。**

- **链接: [https://arxiv.org/pdf/2511.18322](https://arxiv.org/pdf/2511.18322)**

> **作者:** Henrik Krauss; Johann Licher; Naoya Takeishi; Annika Raatz; Takehisa Yairi
>
> **备注:** Dataset available at: this https URL
>
> **摘要:** Learning soft continuum robot (SCR) dynamics from video offers flexibility but existing methods lack interpretability or rely on prior assumptions. Model-based approaches require prior knowledge and manual design. We bridge this gap by introducing: (1) The Attention Broadcast Decoder (ABCD), a plug-and-play module for autoencoder-based latent dynamics learning that generates pixel-accurate attention maps localizing each latent dimension's contribution while filtering static backgrounds, enabling visual interpretability via spatially grounded latents and on-image overlays. (2) Visual Oscillator Networks (VONs), a 2D latent oscillator network coupled to ABCD attention maps for on-image visualization of learned masses, coupling stiffness, and forces, enabling mechanical interpretability. We validate our approach on single- and double-segment SCRs, demonstrating that ABCD-based models significantly improve multi-step prediction accuracy with 5.8x error reduction for Koopman operators and 3.5x for oscillator networks on a two-segment robot. VONs autonomously discover a chain structure of oscillators. This fully data-driven approach yields compact, mechanically interpretable models with potential relevance for future control applications.
>
---
#### [replaced 025] DiffClean: Diffusion-based Makeup Removal for Accurate Age Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.13292](https://arxiv.org/pdf/2507.13292)**

> **作者:** Ekta Gavas; Sudipta Banerjee; Chinmay Hegde; Nasir Memon
>
> **备注:** Revised version with minor changes and code release
>
> **摘要:** Accurate age verification can protect underage users from unauthorized access to online platforms and e-commerce sites that provide age-restricted services. However, accurate age estimation can be confounded by several factors, including facial makeup that can induce changes to alter perceived identity and age to fool both humans and machines. In this work, we propose DiffClean which erases makeup traces using a text-guided diffusion model to defend against makeup attacks. DiffClean improves age estimation (minor vs. adult accuracy by 5.8%) and face verification (TMR by 5.1% at FMR=0.01%) compared to images with makeup. Our method is robust across digitally simulated and real-world makeup styles, and outperforms multiple baselines in terms of biometric and perceptual quality. Our codes are available at this https URL.
>
---
#### [replaced 026] Fusion Complexity Inversion: Why Simpler Cross View Modules Outperform SSMs and Cross View Attention Transformers for Pasture Biomass Regression
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.07819](https://arxiv.org/pdf/2603.07819)**

> **作者:** Mridankan Mandal
>
> **备注:** Accepted to CVPR: Vision for Agriculture Workshop 2026 and published at IEEE Xplore Workshop proceedings
>
> **摘要:** Accurate estimation of pasture biomass from agricultural imagery is critical for sustainable livestock management, yet existing methods are limited by the small, imbalanced, and sparsely annotated datasets typical of real world monitoring. In this study, adaptation of vision foundation models to agricultural regression is systematically evaluated on the CSIRO Pasture Biomass benchmark, a 357 image dual view dataset with laboratory validated, component wise ground truth for five biomass targets, through 17 configurations spanning four backbones (EfficientNet-B3 to DINOv3-ViT-L), five cross view fusion mechanisms, and a 4x2 metadata factorial. A counterintuitive principle, termed "fusion complexity inversion", is uncovered: on scarce agricultural data, a two layer gated depthwise convolution (R^2 = 0.903) outperforms cross view attention transformers (0.833), bidirectional SSMs (0.819), and full Mamba (0.793, below the no fusion baseline). Backbone pretraining scale is found to monotonically dominate all architectural choices, with the DINOv2 -> DINOv3 upgrade alone yielding +5.0 R^2 points. Training only metadata (species, state, and NDVI) is shown to create a universal ceiling at R^2 ~ 0.829, collapsing an 8.4 point fusion spread to 0.1 points. Actionable guidelines for sparse agricultural benchmarks are established: backbone quality should be prioritized over fusion complexity, local modules preferred over global alternatives, and features unavailable at inference excluded.
>
---
#### [replaced 027] PSF-Med: Measuring and Explaining Paraphrase Sensitivity in Medical Vision Language Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.21428](https://arxiv.org/pdf/2602.21428)**

> **作者:** Binesh Sadanandan; Vahid Behzadan
>
> **摘要:** Medical Vision Language Models (VLMs) can change their answers when clinicians rephrase the same question, a failure mode that threatens deployment safety. We introduce PSF-Med, a benchmark of 26,850 chest X-ray questions paired with 92,856 meaning-preserving paraphrases across MIMIC-CXR, PadChest, and VinDr-CXR, spanning clinical populations in the US, Spain, and Vietnam. Every paraphrase is validated by an LLM judge using a bidirectional clinical entailment rubric, with 91.6% cross-family agreement. Across nine VLMs, including general-purpose models, we find flip rates from 3% to 37%. However, low flip rate does not imply visual grounding: text-only baselines show that some models stay consistent even when the image is removed, suggesting they rely on language priors. To study mechanisms in one model, we apply GemmaScope 2 Sparse Autoencoders (SAEs) to MedGemma 4B and analyze FlipBank, a curated set of 158 flip cases. We identify a sparse feature at layer 17 that correlates with prompt framing and predicts decision margin shifts. In causal patching, removing this feature's contribution recovers 45% of the yes-minus-no logit margin on average and fully reverses 15% of flips. Acting on this finding, we show that clamping the identified feature at inference reduces flip rates by 31% relative with only a 1.3 percentage-point accuracy cost, while also decreasing text-prior reliance. These results suggest that flip rate alone is not enough; robustness evaluations should test both paraphrase stability and image reliance.
>
---
#### [replaced 028] Zero-Shot Quantization via Weight-Space Arithmetic
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.03420](https://arxiv.org/pdf/2604.03420)**

> **作者:** Daniele Solombrino; Antonio Andrea Gargiulo; Adrian Robert Minut; Luca Zhou; Alessandro Zirilli; Emanuele Rodolà
>
> **摘要:** We show that robustness to post-training quantization (PTQ) is a transferable direction in weight space. We call this direction the quantization vector: extracted from a donor task by simple weight-space arithmetic, it can be used to patch a receiver model and improve post-PTQ Top-1 accuracy by up to 60 points in a 3-bit setting, without receiver-side quantization-aware training (QAT). Because the method requires no receiver training data, it provides a zero-shot, low-cost alternative to QAT for extremely low-bit deployment. Across four ViT scales and 22 image classification tasks, donor quantization vectors often yield substantial gains even when donor and receiver tasks differ markedly. We further prove rigorously that quantization vectors are well-defined and do not suffer from reparameterization symmetries, and provide a local geometric account of their effect. Together, these results suggest that quantization robustness can be partially isolated, reused, and transferred through simple weight-space algebra.
>
---
#### [replaced 029] Unified Removal of Raindrops and Reflections: A New Benchmark and A Novel Pipeline
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16446](https://arxiv.org/pdf/2603.16446)**

> **作者:** Xingyu Liu; Zewei He; Yu Chen; Chunyu Zhu; Zixuan Chen; Xing Luo; Zhe-Ming Lu
>
> **备注:** 17 pages, 12 figures, 3 tables
>
> **摘要:** When capturing images through glass surfaces or windshields on rainy days, raindrops and reflections frequently co-occur to significantly reduce the visibility of captured images. This practical problem lacks attention and needs to be resolved urgently. Prior de-raindrop, de-reflection, and all-in-one models have failed to address this composite degradation. To this end, we first formally define the unified removal of raindrops and reflections (UR$^3$) task for the first time and construct a real-shot dataset, namely RainDrop and ReFlection (RDRF), which provides a new benchmark with substantial, high-quality, diverse image pairs. Then, we propose a novel diffusion-based framework (i.e., DiffUR$^3$) with several target designs to address this challenging task. By leveraging the powerful generative prior, DiffUR$^3$ successfully removes both types of degradations. Extensive experiments demonstrate that our method achieves state-of-the-art performance on our benchmark and on challenging in-the-wild images. The RDRF dataset and the codes will be made public upon acceptance.
>
---
#### [replaced 030] MM-LIMA: Less Is More for Alignment in Multi-Modal Datasets
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型对齐任务，旨在用少量高质量数据提升模型性能。工作包括提出数据质量评估指标和自动筛选方法，使模型在200条数据上表现优于基准。**

- **链接: [https://arxiv.org/pdf/2308.12067](https://arxiv.org/pdf/2308.12067)**

> **作者:** Lai Wei; Xiaozhe Li; Zihao Jiang; Weiran Huang; Lichao Sun
>
> **备注:** Published at Artificial Intelligence for Engineering
>
> **摘要:** Multimodal large language models are typically trained in two stages: first pre-training on image-text pairs, and then fine-tuning using supervised vision-language instruction data. Recent studies have shown that large language models can achieve satisfactory results even with a limited amount of high-quality instruction-following data. In this paper, we introduce MM-LIMA, which is fine-tuned on a small dataset comprising only 200 examples, amounting to approximately 6% of the instruction-following data used in the alignment dataset for MiniGPT-4. To achieve this, we first propose several metrics to access the quality of multimodal instruction data. Based on these metrics, we present an effective and trainable data selector to automatically identify and filter low-quality vision-language data. By employing this method, MM-LIMA outperforms the original MiniGPT-4 on various evaluations. Overall, our findings demonstrate that less but high-quality instruction tuning data is efficient in enabling multimodal large language models to generate better output. Our code is available at this https URL.
>
---
#### [replaced 031] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.05418](https://arxiv.org/pdf/2604.05418)**

> **作者:** Honghao Fu; Miao Xu; Yiwei Wang; Dailing Zhang; Liu Jun; Yujun Cai
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Scaling multimodal large language models (MLLMs) to long videos is constrained by limited context windows. While retrieval-augmented generation (RAG) is a promising remedy by organizing query-relevant visual evidence into a compact context, most existing methods (i) flatten videos into independent segments, breaking their inherent spatio-temporal structure, and (ii) depend on explicit semantic matching, which can miss cues that are implicitly relevant to the query's intent. To overcome these limitations, we propose VideoStir, a structured and intent-aware long-video RAG framework. It firstly structures a video as a spatio-temporal graph at clip level, and then performs multi-hop retrieval to aggregate evidence across distant yet contextually related events. Furthermore, it introduces an MLLM-backed intent-relevance scorer that retrieves frames based on their alignment with the query's reasoning intent. To support this capability, we curate IR-600K, a large-scale dataset tailored for learning frame-query intent alignment. Experiments show that VideoStir is competitive with state-of-the-art baselines without relying on auxiliary information, highlighting the promise of shifting long-video RAG from flattened semantic matching to structured, intent-aware reasoning. Codes and checkpoints are available at Github.
>
---
#### [replaced 032] Semantic Segmentation Algorithm Based on Light Field and LiDAR Fusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.06687](https://arxiv.org/pdf/2510.06687)**

> **作者:** Jie Luo; Yuxuan Jiang; Xin Jin; Mingyu Liu; Yihui Fan
>
> **摘要:** Semantic segmentation serves as a cornerstone of scene understanding in autonomous driving but continues to face significant challenges under complex conditions such as occlusion. Light field and LiDAR modalities provide complementary visual and spatial cues that are beneficial for robust perception; however, their effective integration is hindered by limited viewpoint diversity and inherent modality discrepancies. To address these challenges, the first multimodal semantic segmentation dataset integrating light field data and point cloud data is proposed. Based on this dataset, we proposed a multi-modal light field point-cloud fusion segmentation network(Mlpfseg), incorporating feature completion and depth perception to segment both camera images and LiDAR point clouds simultaneously. The feature completion module addresses the density mismatch between point clouds and image pixels by performing differential reconstruction of point-cloud feature maps, enhancing the fusion of these modalities. The depth perception module improves the segmentation of occluded objects by reinforcing attention scores for better occlusion awareness. Our method outperforms image-only segmentation by 1.71 Mean Intersection over Union(mIoU) and point cloud-only segmentation by 2.38 mIoU, demonstrating its effectiveness.
>
---
#### [replaced 033] Switch-JustDance: Benchmarking Whole Body Motion Tracking Controllers Using a Commercial Console Game
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在解决缺乏真实场景下人机对比基准的问题。通过游戏平台实现机器人全身控制评估，验证了其有效性并 benchmark 了三种控制器。**

- **链接: [https://arxiv.org/pdf/2511.17925](https://arxiv.org/pdf/2511.17925)**

> **作者:** Jeonghwan Kim; Wontaek Kim; Yidan Lu; Jin Cheng; Fatemeh Zargarbashi; Zicheng Zeng; Zekun Qi; Zhiyang Dou; Nitish Sontakke; Donghoon Baek; Sehoon Ha; Tianyu Li
>
> **摘要:** Recent advances in whole-body robot control have enabled humanoid and legged robots to perform increasingly agile and coordinated motions. However, standardized benchmarks for evaluating these capabilities in real-world settings, and in direct comparison to humans, remain scarce. Existing evaluations often rely on pre-collected human motion datasets or simulation-based experiments, which limit reproducibility, overlook hardware factors, and hinder fair human-robot comparisons. We present Switch-JustDance, a low-cost and reproducible benchmarking pipeline that leverages motion-sensing console games, Just Dance on the Nintendo Switch, to evaluate robot whole-body control. Using Just Dance on the Nintendo Switch as a representative platform, Switch-JustDance converts in-game choreography into robot-executable motions through streaming, motion reconstruction, and motion retargeting modules and enables users to evaluate controller performance through the game's built-in scoring system. We first validate the evaluation properties of Just Dance, analyzing its reliability, validity, sensitivity, and potential sources of bias. Our results show that the platform provides consistent and interpretable performance measures, making it a suitable tool for benchmarking embodied AI. Building on this foundation, we benchmark three state-of-the-art humanoid whole-body controllers on hardware and provide insights into their relative strengths and limitations.
>
---
#### [replaced 034] Masked Training for Robust Arrhythmia Detection from Digitalized Multiple Layout ECG Images
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09165](https://arxiv.org/pdf/2508.09165)**

> **作者:** Shanwei Zhang; Deyun Zhang; Yirao Tao; Kexin Wang; Shijia Geng; Jun Li; Qinghao Zhao; Xingpeng Liu; Xingliang Wu; Shengyong Chen; Yuxi Zhou; Shenda Hong
>
> **备注:** 28 pages, 9 figures
>
> **摘要:** Background: Electrocardiograms are indispensable for diagnosing cardiovascular diseases, yet in many settings they exist only as paper printouts stored in multiple recording layouts. Converting these images into digital signals introduces two key challenges: temporal asynchrony among leads and partial blackout missing, where contiguous signal segments become entirely unavailable. Existing models cannot adequately handle these concurrent problems while maintaining interpretability. Methods: We propose PatchECG, combining an adaptive variable block count missing learning mechanism with a masked training strategy. The model segments each lead into fixed-length patches, discards entirely missing patches, and encodes the remainder via a pluggable patch encoder. A disordered patch attention mechanism with patch-level temporal and lead embeddings captures cross-lead and temporal dependencies without interpolation. PatchECG was trained on PTB-XL and evaluated under seven simulated layout conditions, with external validation on 400 real ECG images from Chaoyang Hospital across three clinical layouts. Results: PatchECG achieves an average AUROC of approximately 0.835 across all simulated layouts. On the Chaoyang cohort, the model attains an overall AUROC of 0.778 for atrial fibrillation detection, rising to 0.893 on the 12x1 subset -- surpassing the pre-trained baseline by 0.111 and 0.190, respectively. Model attention aligns with cardiologist annotations at a rate approaching inter-clinician agreement. Conclusions: PatchECG provides a robust, interpolation-free, and interpretable solution for arrhythmia detection from digitized ECG images across diverse layouts. Its direct modeling of asynchronous and partially missing signals, combined with clinically aligned attention, positions it as a practical tool for cardiac diagnostics from legacy ECG archives in real-world clinical environments.
>
---
#### [replaced 035] GeoFormer: A Lightweight Swin Transformer for Joint Building Height and Footprint Estimation from Sentinel Imagery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.09932](https://arxiv.org/pdf/2602.09932)**

> **作者:** Han Jinzhen; JinByeong Lee; JiSung Kim; MinKyung Cho; DaHee Kim; HongSik Yun
>
> **摘要:** Building height (BH) and footprint (BF) are fundamental urban morphological parameters required by climate modelling, disaster-risk assessment, and population mapping, yet globally consistent data remain scarce. In this work, we develop GeoFormer, a lightweight Swin Transformer-based multi-task learning framework that jointly estimates BH and BF on a 100 m grid using only open-access Sentinel-1 SAR, Sentinel-2 multispectral, and DEM data. A geo-blocked data-splitting strategy enforces strict spatial independence between training and evaluation regions across 54 morphologically diverse cities. We set representative CNN baselines (ResNet, UNet, SENet) as benchmarks and thoroughly evaluate GeoFormer's prediction accuracy, computational efficiency, and spatial transferability. Results show that GeoFormer achieves a BH RMSE of 3.19 m with only 0.32 M parameters -- outperforming the best CNN baseline (UNet) by 7.5% -- indicating that windowed local attention is more effective than convolution for scene-level building-parameter retrieval. Systematic ablation on context window size, model capacity, and input modality further reveals that a 5x5 (500 m) receptive field is optimal, DEM is indispensable for height estimation, and multispectral reflectance carries the dominant predictive signal. Cross-continent transfer tests confirm BH RMSE below 3.5 m without region-specific fine-tuning. All code, model weights, and the resulting global product are publicly released.
>
---
#### [replaced 036] 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17012](https://arxiv.org/pdf/2512.17012)**

> **作者:** Chiao-An Yang; Ryo Hachiuma; Sifei Liu; Subhashree Radhakrishnan; Raymond A. Yeh; Yu-Chiang Frank Wang; Min-Hung Chen
>
> **备注:** CVPR 2026 (Highlight). Project page: this https URL. GitHub: this https URL. Dataset: this https URL
>
> **摘要:** Despite advances in Multimodal LLMs (MLLMs), their ability to reason over 3D structures and temporal dynamics remains limited, constrained by weak 4D perception and temporal understanding. Existing 3D and 4D Video Question Answering (VQA) benchmarks also emphasize static scenes and lack region-level prompting. We tackle these issues by introducing: (a) 4D-RGPT, a specialized MLLM designed to capture 4D representations from video inputs with enhanced temporal perception; (b) Perceptual 4D Distillation (P4D), a training framework that transfers 4D representations from a frozen expert model into 4D-RGPT for comprehensive 4D perception; and (c) R4D-Bench, a benchmark for depth-aware dynamic scenes with region-level prompting, built via a hybrid automated and human-verified pipeline. Our 4D-RGPT achieves notable improvements on both existing 4D VQA benchmarks and the proposed R4D-Bench benchmark.
>
---
#### [replaced 037] ConceptPose: Training-Free Zero-Shot Object Pose Estimation using Concept Vectors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.09056](https://arxiv.org/pdf/2512.09056)**

> **作者:** Liming Kuang; Yordanka Velikova; Mahdi Saleh; Jan-Nico Zaech; Danda Pani Paudel; Benjamin Busam
>
> **摘要:** Object pose estimation is a fundamental task in computer vision and robotics, yet most methods require extensive, dataset-specific training. Concurrently, large-scale vision language models show remarkable zero-shot capabilities. In this work, we bridge these two worlds by introducing ConceptPose, a framework for object pose estimation that is both training-free and model-free. ConceptPose leverages a vision-language-model (VLM) to create open-vocabulary 3D concept maps, where each point is tagged with a concept vector derived from saliency maps. By establishing robust 3D-3D correspondences across concept maps, our approach allows precise estimation of 6DoF relative pose. Without any object or dataset-specific training, our approach achieves state-of-the-art results on common zero shot relative pose estimation benchmarks, outperforming the strongest baseline by a relative 62\% in average ADD(-S) score, including methods that utilize extensive dataset-specific training.
>
---
#### [replaced 038] A Data-driven Loss Weighting Scheme across Heterogeneous Tasks for Image Denoising
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2301.06081](https://arxiv.org/pdf/2301.06081)**

> **作者:** Xiangyu Rui; Xiangyong Cao; Xile Zhao; Deyu Meng; Michael K. NG
>
> **摘要:** In a variational denoising model, weight in the data fidelity term plays the role of enhancing the noise-removal capability. It is profoundly correlated with noise information, while also balancing the data fidelity and regularization terms. However, the difficulty of assigning weight is expected to be substantial when the noise pattern is beyond independent identical Gaussian distribution, e.g., impulse noise, stripe noise, or a mixture of several patterns, etc. Furthermore, how to leverage weight to balance the data fidelity and regularization terms is even less evident. In this work, we propose a data-driven loss weighting (DLW) scheme to address these issues. Specifically, DLW trains a parameterized weight function (i.e., a neural network) that maps the noisy image to the weight. The training is achieved by a bilevel optimization framework, where the lower level problem is solving several denoising models with the same weight predicted by the weight function and the upper level problem minimizes the distance between the restored image and the clean image. In this way, information from both the noise and the regularization can be efficiently extracted to determine the weight function. DLW also facilitates the easy implementation of a trained weight function on denoising models. Numerical results verify the remarkable performance of DLW on improving the ability of various variational denoising models to handle different complex noise. This implies that DLW has the ability to transfer the noise knowledge at the model level to heterogeneous tasks beyond the training ones and the generalization theory underlying DLW is studied, validating its intrinsic transferability.
>
---
#### [replaced 039] FedKLPR: KL-Guided Pruning-Aware Federated Learning for Person Re-Identification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.17431](https://arxiv.org/pdf/2508.17431)**

> **作者:** Po-Hsien Yu; Yu-Syuan Tseng; Shao-Yi Chien
>
> **备注:** 14 pages, 5 figures, 6 tables, submitted to IEEE Transactions on Multimedia
>
> **摘要:** Person re-identification (re-ID) is a fundamental task in intelligent surveillance and public safety. Federated learning (FL) provides a privacy-preserving paradigm by enabling collaborative model training without centralized data collection. However, applying FL to real-world re-ID systems remains challenging due to statistical heterogeneity across clients caused by non-IID data distributions and substantial communication overhead resulting from the frequent transmission of large-scale models. To address these challenges, we propose FedKLPR, a lightweight and communication-efficient federated learning framework for person re-ID. FedKLPR consists of three key components. First, KL-Divergence-Guided training, including the KL-Divergence Regularization Loss (KLL) and the KL-Divergence-aggregation Weight (KLAW), is designed to alleviate statistical heterogeneity and improve convergence stability under non-IID settings. Second, an unstructured pruning strategy is incorporated to reduce communication overhead, and the Pruning-ratio-aggregation Weight (PRAW) is introduced to reflect the relative importance of client parameters. Together with KLAW, PRAW forms a novel aggregation method, namely KL-Divergence-Prune Weighted Aggregation (KLPWA), which enables more effective aggregation of pruned local models under non-IID data distributions and enhances global model robustness. Third, Cross-Round Recovery (CRR) employs a dynamic pruning control mechanism to prevent excessive pruning and preserve model accuracy during iterative compression. Experimental results on eight benchmark datasets demonstrate that FedKLPR achieves substantial communication savings while maintaining competitive accuracy. Compared with state-of-the-art methods, FedKLPR reduces communication cost by 40\%--42\% on ResNet-50 while achieving superior overall performance.
>
---
#### [replaced 040] Automatic Uncertainty-Aware Synthetic Data Bootstrapping for Historical Map Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15875](https://arxiv.org/pdf/2511.15875)**

> **作者:** Lukas Arzoumanidis; Julius Knechtel; Jan-Henrik Haunert; Youness Dehbi
>
> **摘要:** The automated analysis of historical documents, particularly maps, has drastically benefited from advances in deep learning and its success across various computer vision applications. However, most deep learning-based methods heavily rely on large amounts of annotated training data, which are typically unavailable for historical maps, especially for those belonging to specific, homogeneous cartographic domains, also known as corpora. Creating high-quality training data suitable for machine learning often takes a significant amount of time and involves extensive manual effort. While synthetic training data can alleviate the scarcity of real-world samples, it often lacks the affinity (realism) and diversity (variation) necessary for effective learning. By transferring the cartographic style of a historical map corpus onto modern vector data, we bootstrap an effectively unlimited number of synthetic historical maps suitable for tasks such as land-cover interpretation of a homogeneous historical map corpus. We propose an automatic deep generative approach and an alternative manual stochastic degradation technique to emulate the visual uncertainty and noise, also known as aleatoric uncertainty, commonly observed in historical map scans. To quantitatively evaluate the effectiveness and applicability of our approach, the bootstrapped training datasets were employed for domain-adaptive semantic segmentation on a homogeneous map corpus using a Self-Constructing Graph Convolutional Network, enabling a comprehensive assessment of the impact of our data bootstrapping methods.
>
---
#### [replaced 041] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出GameplayQA，用于评估3D环境中智能体的感知与推理能力，解决多智能体行为理解问题，通过密集标注视频生成诊断问答对。**

- **链接: [https://arxiv.org/pdf/2603.24329](https://arxiv.org/pdf/2603.24329)**

> **作者:** Yunzhe Wang; Runhui Xu; Kexin Zheng; Tianyi Zhang; Jayavibhav Niranjan Kogundi; Soham Hans; Volkan Ustun
>
> **备注:** Accepted to the Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Multimodal LLMs are increasingly deployed as perceptual backbones for autonomous agents in 3D environments, from robotics to virtual worlds. These applications require agents to perceive rapid state changes, attribute actions to the correct entities, and reason about concurrent multi-agent behaviors from a first-person perspective, capabilities that existing benchmarks do not adequately evaluate. We introduce GameplayQA, a framework for evaluating agentic-centric perception and reasoning through video understanding. Specifically, we densely annotate multiplayer 3D gameplay videos at 1.22 labels/second, with time-synced, concurrent captions of states, actions, and events structured around a triadic system of Self, Other Agents, and the World, a natural decomposition for multi-agent environments. From these annotations, we refined 2.4K diagnostic QA pairs organized into three levels of cognitive complexity, accompanied by a structured distractor taxonomy that enables fine-grained analysis of where models hallucinate. Evaluation of frontier MLLMs reveals a substantial gap from human performance, with common failures in temporal and cross-video grounding, agent-role attribution, and handling the decision density of the game. We hope GameplayQA stimulates future research at the intersection of embodied AI, agentic perception, and world modeling.
>
---
#### [replaced 042] TARAC: Mitigating Hallucination in LVLMs via Temporal Attention Real-time Accumulative Connection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.04099](https://arxiv.org/pdf/2504.04099)**

> **作者:** Lei Jiang; Chunzhao Xie; Tongxuan Liu; Yuting Zeng; jinrong Guo; Yunheng Shen; Weizhe Huang; Jing Li; Xiaohua Xu
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Large Vision-Language Models have demonstrated remarkable capabilities, yet they suffer from hallucinations that limit practical deployment. While various mitigation strategies exist, they often incur high computational overhead or require extensive retraining. In this paper, we address the issue of visual attention decay during generation, a key factor contributing to hallucinations. We propose Temporal Attention Real-time Accumulative Connection (TARAC), a novel training-free framework that dynamically accumulates and re-injects historical attention to sustain visual grounding. Inspired by cognitive reinforcement mechanisms, TARAC operates as a lightweight, plug-and-play module. Extensive experiments across diverse models (e.g., LLaVA, Qwen2-VL) and benchmarks demonstrate that TARAC significantly outperforms state-of-the-art methods. Remarkably, it achieves these gains with negligible inference overhead ($\sim$4\% TPOT increase), compared to the substantial costs of existing training-free baselines. Specifically, TARAC reduces hallucinated sentences by 25.2\% on CHAIR and improves Perception score by +10.65 on MME, validating its effectiveness and efficiency.
>
---
#### [replaced 043] Temporal-Aware Spiking Transformer Hashing Based on 3D-DWT
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.06786](https://arxiv.org/pdf/2501.06786)**

> **作者:** Zihao Mei; Jianhao Li; Bolin Zhang; Chong Wang; Lijun Guo; Guoqi Li; Jiangbo Qian
>
> **备注:** TCYB under review. This work has been submitted to the lEEE for possible publication
>
> **摘要:** With the rapid growth of dynamic vision sensor (DVS) data, constructing a low-energy, efficient data retrieval system has become an urgent task. Hash learning is one of the most important retrieval technologies which can keep the distance between hash codes consistent with the distance between DVS data. As spiking neural networks (SNNs) can encode information through spikes, they demonstrate great potential in promoting energy efficiency. Based on the binary characteristics of SNNs, we first propose a novel supervised hashing method named Spikinghash with a hierarchical lightweight structure. Spiking WaveMixer (SWM) is deployed in shallow layers, utilizing a multilevel 3D discrete wavelet transform (3D-DWT) to decouple spatiotemporal features into various low-frequency and high frequency components, and then employing efficient spectral feature fusion. SWM can effectively capture the temporal dependencies and local spatial features. Spiking Self-Attention (SSA) is deployed in deeper layers to further extract global spatiotemporal information. We also design a hash layer utilizing binary characteristic of SNNs, which integrates information over multiple time steps to generate final hash codes. Furthermore, we propose a new dynamic soft similarity loss for SNNs, which utilizes membrane potentials to construct a learnable similarity matrix as soft labels to fully capture the similarity differences between classes and compensate information loss in SNNs, thereby improving retrieval performance. Experiments on multiple datasets demonstrate that Spikinghash can achieve state-of-the-art results with low energy consumption and fewer parameters.
>
---
#### [replaced 044] Catalyst: Out-of-Distribution Detection via Elastic Scaling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02409](https://arxiv.org/pdf/2602.02409)**

> **作者:** Abid Hassan; Tuan Ngo; Saad Shafiq; Nenad Medvidovic
>
> **备注:** Accepted at Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Out-of-distribution (OOD) detection is critical for the safe deployment of deep neural networks. State-of-the-art post-hoc methods typically derive OOD scores from the output logits or penultimate feature vector obtained via global average pooling (GAP). We contend that this exclusive reliance on the logit or feature vector discards a rich, complementary signal: the raw channel-wise statistics of the pre-pooling feature map lost in GAP. In this paper, we introduce Catalyst, a post-hoc framework that exploits these under-explored signals. Catalyst computes an input-dependent scaling factor ($\gamma$) on-the-fly from these raw statistics (e.g., mean, standard deviation, and maximum activation). This $\gamma$ is then fused with the existing baseline score, multiplicatively modulating it -- an $\textit{elastic scaling}$ -- to push the ID and OOD distributions further apart. We demonstrate Catalyst is a generalizable framework: it seamlessly integrates with logit-based methods (e.g., Energy, ReAct, SCALE) and also provides a significant boost to distance-based detectors like KNN. As a result, Catalyst achieves substantial and consistent performance gains, reducing the average False Positive Rate by 32.87 on CIFAR-10 (ResNet-18), 27.94% on CIFAR-100 (ResNet-18), and 22.25% on ImageNet (ResNet-50). Our results highlight the untapped potential of pre-pooling statistics and demonstrate that Catalyst is complementary to existing OOD detection approaches. Our code is available here: this https URL
>
---
#### [replaced 045] VAGNet: Vision-based Accident Anticipation with Global Features
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.09305](https://arxiv.org/pdf/2604.09305)**

> **作者:** Vipooshan Vipulananthan; Charith D. Chitraranjan
>
> **摘要:** Traffic accidents are a leading cause of fatalities and injuries across the globe. Therefore, the ability to anticipate hazardous situations in advance is essential. Automated accident anticipation enables timely intervention through driver alerts and collision avoidance maneuvers, forming a key component of advanced driver assistance systems. In autonomous driving, such predictive capabilities support proactive safety behaviors, such as initiating defensive driving and human takeover when required. Using dashcam video as input offers a cost-effective solution, but it is challenging due to the complexity of real-world driving scenes. Accident anticipation systems need to operate in real-time. However, current methods involve extracting features from each detected object, which is computationally intensive. We propose VAGNet, a deep neural network that learns to predict accidents from dash-cam video using global features of traffic scenes without requiring explicit object-level features. The network consists of transformer and graph modules, and we use the vision foundation model VideoMAE-V2 for global feature extraction. Experiments on four benchmark datasets (DAD, DoTA, DADA, and Nexar) show that our method anticipates accidents with higher average precision and mean time-to-accident while being computationally more efficient compared to existing methods.
>
---
#### [replaced 046] FM-SIREN & FM-FINER: Implicit Neural Representation Using Nyquist-based Orthogonality
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23438](https://arxiv.org/pdf/2509.23438)**

> **作者:** Mohammed Alsakabi; Wael Mobeirek; John M. Dolan; Ozan K. Tonguz
>
> **摘要:** Existing periodic activation-based implicit neural representation (INR) networks, such as SIREN and FINER, suffer from hidden feature redundancy, where neurons within a layer capture overlapping frequency components due to the use of a fixed frequency multiplier. This redundancy limits the expressive capacity of multilayer perceptrons (MLPs). Drawing inspiration from classical signal processing methods such as the Discrete Sine Transform (DST), in this paper, we propose FM-SIREN and FM-FINER, which assign Nyquist-informed, neuron-specific frequency multipliers to periodic activations. Contrary to existing approaches, our design introduces frequency diversity without requiring hyperparameter tuning or additional network depth. This simple yet principled approach reduces the redundancy of features by nearly 50% and consistently improves signal reconstruction across diverse INR tasks, such as fitting 1D audio, 2D image and 3D shape, and video, outperforming their baseline counterparts while maintaining efficiency.
>
---
#### [replaced 047] Progressive Multimodal Interaction Network for Reliable Quantification of Fish Feeding Intensity in Aquaculture
- **分类: cs.CV; cs.AI; cs.ET**

- **链接: [https://arxiv.org/pdf/2506.14170](https://arxiv.org/pdf/2506.14170)**

> **作者:** Shulong Zhang; Mingyuan Yao; Jiayin Zhao; Daoliang Li; Yingyi Chen; Haihua Wang
>
> **摘要:** Accurate quantification of fish feeding intensity is crucial for precision feeding in aquaculture, as it directly affects feed utilization and farming efficiency. Although multimodal fusion has proven to be an effective solution, existing methods often overlook the inconsistencies in responses and decision conflicts between different modalities, thus limiting the reliability of the quantification results. To address this issue, this paper proposes a Progressive Multimodal Interaction Network (PMIN) that integrates image, audio, and water-wave data for fish feeding intensity quantification. Specifically, a unified feature extraction framework is first constructed to map inputs from different modalities into a structurally consistent feature space, thereby reducing representational discrepancies across modalities. Then, an auxiliary-modality reinforcement primary-modality mechanism is designed to facilitate the fusion of cross-modal information, which is achieved through channel aware recalibration and dual-stage attention interaction. Furthermore, a decision fusion strategy based on adaptive evidence reasoning is introduced to jointly model the confidence, reliability, and conflicts of modality-specific outputs, so as to improve the stability and robustness of the final judgment. Experiments are conducted on a multimodal fish feeding intensity dataset containing 7089 samples. The results show that PMIN has an accuracy of 96.76%, while maintaining relatively low parameter count and computational cost, and its overall performance outperforms both homogeneous and heterogeneous comparison models. Ablation studies, comparative experiments, and real-world application results further validate the effectiveness and superiority of the proposed method. It can provide reliable support for automated feeding monitoring and precise feeding decisions in smart aquaculture.
>
---
#### [replaced 048] Beyond Matching to Tiles: Bridging Unaligned Aerial and Satellite Views for Vision-Only UAV Navigation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.22153](https://arxiv.org/pdf/2603.22153)**

> **作者:** Kejia Liu; Haoyang Zhou; Ruoyu Xu; Peicheng Wang; Mingli Song; Haofei Zhang
>
> **备注:** Accepted as a conference paper by CVPR2026
>
> **摘要:** Recent advances in cross-view geo-localization (CVGL) methods have shown strong potential for supporting unmanned aerial vehicle (UAV) navigation in GNSS-denied environments. However, existing work predominantly focuses on matching UAV views to onboard map tiles, which introduces an inherent trade-off between accuracy and storage overhead, and overlooks the importance of the UAV's heading during navigation. Moreover, the substantial discrepancies and varying overlaps in cross-view scenarios have been insufficiently considered, limiting their generalization to real-world scenarios. In this paper, we present Bearing-UAV, a purely vision-driven cross-view navigation method that jointly predicts UAV absolute location and heading from neighboring features, enabling accurate, lightweight, and robust navigation in the wild. Our method leverages global and local structural features and explicitly encodes relative spatial relationships, making it robust to cross-view variations, misalignment, and feature-sparse conditions. We also present Bearing-UAV-90k, a multi-city benchmark for evaluating cross-view localization and navigation. Extensive experiments show encouraging results that Bearing-UAV yields lower localization error than previous matching/retrieval paradigm across diverse terrains. Our code and dataset will be made publicly available.
>
---
#### [replaced 049] Specificity-aware reinforcement learning for fine-grained open-world classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03197](https://arxiv.org/pdf/2603.03197)**

> **作者:** Samuele Angheben; Davide Berasi; Alessandro Conti; Elisa Ricci; Yiming Wang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Classifying fine-grained visual concepts under open-world settings, i.e., without a predefined label set, demands models to be both accurate and specific. Recent reasoning Large Multimodal Models (LMMs) exhibit strong visual understanding capability but tend to produce overly generic predictions when performing fine-grained image classification. Our preliminary analysis reveals that models do possess the intrinsic fine-grained domain knowledge. However, promoting more specific predictions (specificity) without compromising correct ones (correctness) remains a non-trivial and understudied challenge. In this work, we investigate how to steer reasoning LMMs toward predictions that are both correct and specific. We propose a novel specificity-aware reinforcement learning framework, SpeciaRL, to fine-tune reasoning LMMs on fine-grained image classification under the open-world setting. SpeciaRL introduces a dynamic, verifier-based reward signal anchored to the best predictions within online rollouts, promoting specificity while respecting the model's capabilities to prevent incorrect predictions. Our out-of-domain experiments show that SpeciaRL delivers the best trade-off between correctness and specificity across extensive fine-grained benchmarks, surpassing existing methods and advancing open-world fine-grained image classification. Code and model are publicly available at this https URL.
>
---
#### [replaced 050] DoReMi: Bridging 3D Domains via Topology-Aware Domain-Representation Mixture of Experts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11232](https://arxiv.org/pdf/2511.11232)**

> **作者:** Mingwei Xing; Xinliang Wang; Yifeng Shi
>
> **备注:** The first two authors contributed equally to this paper
>
> **摘要:** Constructing a unified 3D scene understanding model has long been hindered by the significant topological discrepancies across different sensor modalities. While applying the Mixture-of-Experts (MoE) architecture is an effective approach to achieving universal understanding, we observe that existing 3D MoE networks often suffer from semantics-driven routing bias. This makes it challenging to address cross-domain data characterized by "semantic consistency yet topological heterogeneity." To overcome this challenge, we propose DoReMi (Topology-Aware Domain-Representation Mixture of Experts). Specifically, we introduce a self-supervised pre-training branch based on multi attributes, such as topological and texture variations, to anchor cross-domain structural priors. Building upon this, we design a domain-aware expert branch comprising two core mechanisms: Domain Spatial-Guided Routing (DSR), which achieves an acute perception of local topological variations by extracting spatial contexts, and Entropy-controlled Dynamic Allocation (EDA), which dynamically adjusts the number of activated experts by quantifying routing uncertainty to ensure training stability. Through the synergy of these dual branches, DoReMi achieves a deep integration of universal feature extraction and highly adaptive expert allocation. Extensive experiments across various tasks, encompassing both indoor and outdoor scenes, validate the superiority of DoReMi. It achieves 80.1% mIoU on the ScanNet validation set and 77.2% mIoU on S3DIS, comprehensively outperforming existing state-of-the-art methods. The code will be released soon.
>
---
#### [replaced 051] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.04334](https://arxiv.org/pdf/2509.04334)**

> **作者:** Pengyue Jia; Yingyi Zhang; Xiangyu Zhao; Sharon Li
>
> **备注:** ACL 2026 Main
>
> **摘要:** Geographic reasoning is a fundamental cognitive capability that requires models to infer plausible locations by synthesizing visual evidence with spatial world knowledge. Despite recent advances in large vision-language models (LVLMs), existing evaluation paradigms remain largely outcome-centric, relying on static datasets and predefined labels that are conceptually misaligned with open-world geographic inference. Such outcome-centric evaluations often focus exclusively on label matching, leaving the underlying linguistic reasoning chains as unexamined black boxes. In this work, we introduce GeoArena, a dynamic, human-preference-based evaluation framework for benchmarking open-world geographic reasoning. GeoArena reframes evaluation as a pairwise reasoning alignment task on in-the-wild images, where human judges compare model-generated explanations based on reasoning quality, evidence synthesis, and plausibility. We deploy GeoArena as a public platform and benchmark 17 frontier LVLMs using thousands of human judgments, which complements existing benchmarks and supports the development of geographically grounded, human-aligned AI systems. We further provide detailed analyses of model behavior, including reliability of human preferences and factors influencing judgments of geographic reasoning quality.
>
---
#### [replaced 052] What Users Leave Unsaid: Under-Specified Queries Limit Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.06165](https://arxiv.org/pdf/2601.06165)**

> **作者:** Dasol Choi; Guijin Son; Hanwool Lee; Minhyuk Kim; Hyunwoo Ko; Teabin Lim; Ahn Eungyeol; Jungwhan Kim; Seunghyeok Hong; Youngsook Song
>
> **摘要:** Current vision-language benchmarks predominantly feature well-structured questions with clear, explicit prompts. However, real user queries are often informal and underspecified. Users naturally leave much unsaid, relying on images to convey context. We introduce HAERAE-Vision, a benchmark of 653 real-world visual questions from Korean online communities (0.76% survival from 86K candidates), each paired with an explicit rewrite, yielding 1,306 query variants in total. Evaluating 39 VLMs, we find that even state-of-the-art models (GPT-5, Gemini 2.5 Pro) achieve under 50% on the original queries. Crucially, query explicitation alone yields 8 to 22 point improvements, with smaller models benefiting most. We further show that even with web search, under-specified queries underperform explicit queries without search, revealing that current retrieval cannot compensate for what users leave unsaid. Our findings demonstrate that a substantial portion of VLM difficulty stem from natural query under-specification instead of model capability, highlighting a critical gap between benchmark evaluation and real-world deployment.
>
---
#### [replaced 053] GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文属于视觉生成任务，解决复杂文本提示下物体空间关系和属性精确生成的问题。通过强化学习增强模型的语义空间推理能力，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2505.17022](https://arxiv.org/pdf/2505.17022)**

> **作者:** Chengqi Duan; Rongyao Fang; Yuqing Wang; Kun Wang; Linjiang Huang; Xingyu Zeng; Hongsheng Li; Xihui Liu
>
> **备注:** Github page refer to: this https URL. Published as a conference paper at ICLR 2026
>
> **摘要:** Visual generation models have made remarkable progress in creating realistic images from text prompts, yet struggle with complex prompts that specify multiple objects with precise spatial relationships and attributes. Effective handling of such prompts requires explicit reasoning about the semantic content and spatial layout. We present GoT-R1, a framework that applies reinforcement learning to enhance semantic-spatial reasoning in visual generation. Building upon the Generation Chain-of-Thought approach, GoT-R1 enables models to autonomously discover effective reasoning strategies beyond predefined templates through carefully designed reinforcement learning. To achieve this, we propose a dual-stage multi-dimensional reward framework that leverages MLLMs to evaluate both the reasoning process and final output, enabling effective supervision across the entire generation pipeline. The reward system assesses semantic alignment, spatial accuracy, and visual quality in a unified approach. Experimental results demonstrate significant improvements on T2I-CompBench benchmark, particularly in compositional tasks involving precise spatial relationships and attribute binding. GoT-R1 advances the state-of-the-art in image generation by successfully transferring sophisticated reasoning capabilities to the visual generation domain. To facilitate future research, we make our code and pretrained models publicly available at this https URL.
>
---
#### [replaced 054] CARINOX: Inference-time Scaling with Category-Aware Reward-based Initial Noise Optimization and Exploration
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂语义对齐问题。提出CARINOX框架，结合噪声优化与探索，提升图像与文本的匹配度。**

- **链接: [https://arxiv.org/pdf/2509.17458](https://arxiv.org/pdf/2509.17458)**

> **作者:** Seyed Amir Kasaei; Ali Aghayari; Arash Marioriyad; Niki Sepasian; Shayan Baghayi Nejad; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **备注:** Accepted at TMLR (2026)
>
> **摘要:** Text-to-image diffusion models, such as Stable Diffusion, can produce high-quality and diverse images but often fail to achieve compositional alignment, particularly when prompts describe complex object relationships, attributes, or spatial arrangements. Recent inference-time approaches address this by optimizing or exploring the initial noise under the guidance of reward functions that score text-image alignment without requiring model fine-tuning. While promising, each strategy has intrinsic limitations when used alone: optimization can stall due to poor initialization or unfavorable search trajectories, whereas exploration may require a prohibitively large number of samples to locate a satisfactory output. Our analysis further shows that neither single reward metrics nor ad-hoc combinations reliably capture all aspects of compositionality, leading to weak or inconsistent guidance. To overcome these challenges, we present Category-Aware Reward-based Initial Noise Optimization and Exploration (CARINOX), a unified framework that combines noise optimization and exploration with a principled reward selection procedure grounded in correlation with human judgments. Evaluations on two complementary benchmarks covering diverse compositional challenges show that CARINOX raises average alignment scores by +16% on T2I-CompBench++ and +11% on the HRS benchmark, consistently outperforming state-of-the-art optimization and exploration-based methods across all major categories, while preserving image quality and diversity. The project page is available at this https URL.
>
---
#### [replaced 055] Fake-HR1: Rethinking Reasoning of Vision Language Model for Synthetic Image Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.10042](https://arxiv.org/pdf/2602.10042)**

> **作者:** Changjiang Jiang; Xinkuan Sha; Fengchang Yu; Jingjing Liu; Jian Liu; Mingqi Fang; Chenfeng Zhang; Wei Lu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Recent studies have demonstrated that incorporating Chain-of-Thought (CoT) reasoning into the detection process can enhance a model's ability to detect synthetic images. However, excessively lengthy reasoning incurs substantial resource overhead, including token consumption and latency, which is particularly redundant when handling obviously generated forgeries. To address this issue, we propose Fake-HR1, a large-scale hybrid-reasoning model that, to the best of our knowledge, is the first to adaptively determine whether reasoning is necessary based on the characteristics of the generative detection task. To achieve this, we design a two-stage training framework: we first perform Hybrid Fine-Tuning (HFT) for cold-start initialization, followed by online reinforcement learning with Hybrid-Reasoning Grouped Policy Optimization (HGRPO) to implicitly learn when to select an appropriate reasoning mode. Experimental results show that Fake-HR1 adaptively performs reasoning across different types of queries, surpassing existing LLMs in both reasoning ability and generative detection performance, while significantly improving response efficiency.
>
---
#### [replaced 056] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: 该论文属于语言模型控制任务，旨在统一理解不同控制方法的效果。通过分析偏好与效用的权衡，提出新方法SPLIT以提升控制效果并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2602.02343](https://arxiv.org/pdf/2602.02343)**

> **作者:** Ziwen Xu; Chenyan Wu; Hengyu Sun; Haiwen Hong; Mengru Wang; Yunzhi Yao; Longtao Huang; Hui Xue; Shumin Deng; Zhixuan Chu; Huajun Chen; Ningyu Zhang
>
> **备注:** ACL 2026
>
> **摘要:** Methods for controlling large language models (LLMs), including local weight fine-tuning, LoRA-based adaptation, and activation-based interventions, are often studied in isolation, obscuring their connections and making comparison difficult. In this work, we present a unified view that frames these interventions as dynamic weight updates induced by a control signal, placing them within a single conceptual framework. Building on this view, we propose a unified preference-utility analysis that separates control effects into preference, defined as the tendency toward a target concept, and utility, defined as coherent and task-valid generation, and measures both on a shared log-odds scale using polarity-paired contrastive examples. Across methods, we observe a consistent trade-off between preference and utility: stronger control increases preference while predictably reducing utility. We further explain this behavior through an activation manifold perspective, in which control shifts representations along target-concept directions to enhance preference, while utility declines primarily when interventions push representations off the model's valid-generation manifold. Finally, we introduce a new steering approach SPLIT guided by this analysis that improves preference while better preserving utility. Code is available at this https URL.
>
---
#### [replaced 057] Dual-Margin Embedding for Fine-Grained Long-Tailed Plant Taxonomy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18994](https://arxiv.org/pdf/2512.18994)**

> **作者:** Cheng Yaw Low; Heejoon Koo; Jaewoo Park; Meeyoung Cha
>
> **备注:** 4 figures, 5 tables, and 17 pages
>
> **摘要:** Taxonomic classification of ecological families, genera, and species underpins biodiversity monitoring and conservation. Existing computer vision methods typically address fine-grained recognition and long-tailed learning in isolation. However, additional challenges such as spatiotemporal domain shift, hierarchical taxonomic structure, and previously unseen taxa often co-occur in real-world deployment, leading to brittle performance under open-world conditions. We propose TaxoNet, an embedding learning framework with a theoretically grounded dual-margin objective that reshapes class decision boundaries under class imbalance to improve fine-grained discrimination while strengthening rare-class representation geometry. We evaluate TaxoNet in open-world settings that capture co-occurring recognition challenges. Leveraging diverse plant datasets, including Google Auto-Arborist (urban tree imagery), iNaturalist (Plantae observations across heterogeneous ecosystems), and NAFlora-Mini (herbarium collections), we demonstrate that TaxoNet consistently outperforms strong baselines, including multimodal foundation models.
>
---
#### [replaced 058] CylinderDepth: Cylindrical Spatial Attention for Multi-View Consistent Self-Supervised Surround Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16428](https://arxiv.org/pdf/2511.16428)**

> **作者:** Samer Abualhanud; Christian Grannemann; Max Mehltretter
>
> **备注:** Accepted at 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
>
> **摘要:** Self-supervised surround-view depth estimation enables dense, low-cost 3D perception with a 360° field of view from multiple minimally overlapping images. Yet, most existing methods suffer from depth estimates that are inconsistent across overlapping images. To address this limitation, we propose a novel geometry-guided method for calibrated, time-synchronized multi-camera rigs that predicts dense metric depth. Our approach targets two main sources of inconsistency: the limited receptive field in border regions of single-image depth estimation, and the difficulty of correspondence matching. We mitigate these two issues by extending the receptive field across views and restricting cross-view attention to a small neighborhood. To this end, we establish the neighborhood relationships between images by mapping the image-specific feature positions onto a shared cylinder. Based on the cylindrical positions, we apply an explicit spatial attention mechanism, with non-learned weighting, that aggregates features across images according to their distances on the cylinder. The modulated features are then decoded into a depth map for each view. Evaluated on the DDAD and nuScenes datasets, our method improves both cross-view depth consistency and overall depth accuracy compared with state-of-the-art approaches. Code is available at this https URL.
>
---
#### [replaced 059] Dual-R-DETR: Resolving Query Competition with Pairwise Routing in Transformer Decoders
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13876](https://arxiv.org/pdf/2512.13876)**

> **作者:** Ye Zhang; Qi Chen; Wenyou Huang; Rui Liu; Zhengjian Kang
>
> **备注:** 6 pages, 2 figures, Accepted at ICME2026
>
> **摘要:** Detection Transformers (DETR) formulate object detection as a set prediction problem and enable end-to-end training without post-processing. However, object queries in DETR interact through symmetric self-attention, which enforces uniform competition among all query pairs. This often leads to inefficient query dynamics, where multiple queries converge to the same object while others fail to explore alternative regions. We propose Dual-R-DETR, a competition-aware DETR framework that explicitly regulates query interactions via pairwise routing in transformer decoders. Dual-R-DETR distinguishes query-to-query relations as either competitive or cooperative based on appearance similarity, prediction confidence, and spatial geometry. It introduces two complementary routing behaviors: suppressor routing to attenuate interactions among queries targeting the same object, and delegator routing to encourage diversification across distinct regions. These behaviors are realized through lightweight, learnable low-rank biases injected into decoder self-attention, enabling asymmetric query interactions while preserving the standard attention formulation. To ensure inference efficiency, routing biases are applied only during training using a dual-branch strategy, and inference reverts to vanilla self-attention with no additional computational cost. Extensive experiments on COCO and Cityscapes demonstrate that Dual-R-DETR consistently improves multiple DETR variants, outperforming DINO by 1.7% mAP with a ResNet-50 backbone and achieving 57.6% mAP with Swin-L under comparable settings. Code is available at this https URL.
>
---
#### [replaced 060] Accelerating Transformer-Based Monocular SLAM via Geometric Utility Scoring
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于单目SLAM任务，旨在解决GFM部署中的计算冗余问题。提出LeanGate网络，在特征提取前预测几何效用分数，有效减少冗余帧处理，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2604.08718](https://arxiv.org/pdf/2604.08718)**

> **作者:** Xinmiao Xiong; Bangya Liu; Hao Wang; Dayou Li; Nuo Chen; Andrew Feng; Mingyu Ding; Suman Banerjee; Yang Zhou; Zhiwen Fan
>
> **摘要:** Geometric Foundation Models (GFMs) have recently advanced monocular SLAM by providing robust, calibration-free 3D priors. However, deploying these models on dense video streams introduces significant computational redundancy. Current GFM-based SLAM systems typically rely on post hoc keyframe selection. Because of this, they must perform expensive dense geometric decoding simply to determine whether a frame contains novel geometry, resulting in late rejection and wasted computation. To mitigate this inefficiency, we propose LeanGate, a lightweight feed-forward frame-gating network. LeanGate predicts a geometric utility score to assess a frame's mapping value prior to the heavy GFM feature extraction and matching stages. As a predictive plug-and-play module, our approach bypasses over 90% of redundant frames. Evaluations on standard SLAM benchmarks demonstrate that LeanGate reduces tracking FLOPs by more than 85% and achieves a 5x end-to-end throughput speedup. Furthermore, it maintains the tracking and mapping accuracy of dense baselines. Project page: this https URL
>
---
#### [replaced 061] Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20725](https://arxiv.org/pdf/2603.20725)**

> **作者:** Zihao Wang; Yuxiang Wei; Xinpeng Zhou; Tianyu Zhang; Tao Liang; Yalong Bai; Hongzhi Zhang; Wangmeng Zuo
>
> **摘要:** Text-to-image generation has advanced rapidly, yet it still struggles to capture the nuanced user preferences. Existing approaches typically rely on multimodal large language models to infer user preferences, but the derived prompts or latent codes rarely reflect them faithfully, leading to suboptimal personalization. We present Premier, a novel preference modulation framework for personalized image generation. Premier represents each user's preference as a learnable embedding and introduces a preference adapter that fuses the user embedding with the text prompt. To enable accurate and fine-grained preference control, the fused preference embedding is further used to modulate the generative process. To enhance the distinctness of individual preference and improve alignment between outputs and user-specific styles, we incorporate a dispersion loss that enforces separation among user embeddings. When user data are scarce, new users are represented as linear combinations of existing preference embeddings learned during training, enabling effective generalization. Experiments show that Premier outperforms prior methods under the same history length, achieving stronger preference alignment and superior performance on text consistency, ViPer proxy metrics, and expert evaluations.
>
---
#### [replaced 062] Ninja Codes: Neurally Generated Fiducial Markers for Stealthy 6-DoF Tracking
- **分类: cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2510.18976](https://arxiv.org/pdf/2510.18976)**

> **作者:** Yuichiro Takeuchi; Yusuke Imoto; Shunya Kato
>
> **备注:** CVPR 2026 Findings; Project page: this https URL
>
> **摘要:** In this paper we describe Ninja Codes, neurally generated fiducial markers that can be made to naturally blend into various real-world environments. An encoder network converts arbitrary images into Ninja Codes by applying visually modest alterations; the resulting codes, printed and pasted onto surfaces, can provide stealthy 6-DoF location tracking for a wide range of applications including robotics and augmented reality. Ninja Codes can be printed using standard color printers on regular printing paper, and can be detected using any device equipped with a modern RGB camera and capable of running inference. Through experiments, we demonstrate Ninja Codes' ability to provide reliable location tracking under common indoor lighting conditions, while successfully concealing themselves within diverse environmental textures. We expect Ninja Codes to offer particular value in scenarios where the conspicuous appearance of conventional fiducial markers makes them undesirable for aesthetic and other reasons.
>
---
#### [replaced 063] HFI: A unified framework for training-free detection and implicit watermarking of latent diffusion model generated images
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2412.20704](https://arxiv.org/pdf/2412.20704)**

> **作者:** Sungik Choi; Hankook Lee; Jaehoon Lee; Seunghyun Kim; Stanley Jungkyu Choi; Moontae Lee
>
> **摘要:** Dramatic advances in the quality of the latent diffusion models (LDMs) also led to the malicious use of AI-generated images. While current AI-generated image detection methods assume the availability of real/AI-generated images for training, this is practically limited given the vast expressibility of LDMs. This motivates the training-free detection setup where no related data are available in advance. The existing LDM-generated image detection method assumes that images generated by LDM are easier to reconstruct using an autoencoder than real images. However, we observe that this reconstruction distance is overfitted to background information, leading the current method to underperform in detecting images with simple backgrounds. To address this, we propose a novel method called HFI. Specifically, by viewing the autoencoder of LDM as a downsampling-upsampling kernel, HFI measures the extent of aliasing, a distortion of high-frequency information that appears in the reconstructed image. HFI is training-free, efficient, and consistently outperforms other training-free methods in detecting challenging images generated by various generative models. We also show that HFI can successfully detect the images generated from the specified LDM as a means of implicit watermarking. HFI outperforms the best baseline method while achieving magnitudes of
>
---
#### [replaced 064] TerraSky3D: Multi-View Reconstructions of European Landmarks in 4K
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.28287](https://arxiv.org/pdf/2603.28287)**

> **作者:** Mattia D'Urso; Yuxi Hu; Christian Sormann; Mattia Rossi; Friedrich Fraundorfer
>
> **备注:** Accepted at 3DMV (CVPR Workshop 2026)
>
> **摘要:** Despite the growing need for data of more and more sophisticated 3D reconstruction pipelines, we can still observe a scarcity of suitable public datasets. Existing 3D datasets are either low resolution, limited to a small amount of scenes, based on images of varying quality because retrieved from the internet, or limited to specific capturing scenarios. Motivated by this lack of suitable 3D datasets, we captured TerraSky3D, a high-resolution large-scale 3D reconstruction dataset comprising 50,000 images divided into 150 ground, aerial, and mixed scenes. The dataset focuses on European landmarks and comes with curated calibration data, camera poses, and depth maps. TerraSky3D tries to answer the need for challenging dataset that can be used to train and evaluate 3D reconstruction-related pipelines.
>
---
#### [replaced 065] FPBench: A Comprehensive Benchmark of Multimodal Large Language Models for Fingerprint Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18073](https://arxiv.org/pdf/2512.18073)**

> **作者:** Ekta Gavas; Sudipta Banerjee; Chinmay Hegde; Nasir Memon
>
> **备注:** Revised version with additional experiments and code release
>
> **摘要:** Multimodal LLMs (MLLMs) are capable of performing complex data analysis, visual question answering, generation, and reasoning tasks. However, their ability to analyze biometric data is relatively underexplored. In this work, we investigate the effectiveness of MLLMs in understanding fine structural and textural details present in fingerprint images. To this end, we design a comprehensive benchmark, FPBench, to evaluate 20 MLLMs (open-source and proprietary models) across 7 real and synthetic datasets on a suite of 8 biometric and forensic tasks (e.g., pattern analysis, fingerprint verification, real versus synthetic classification, etc.) using zero-shot and chain-of-thought prompting strategies. We further fine-tune vision and language encoders on a subset of open-source MLLMs to demonstrate domain adaptation. FPBench is a novel benchmark designed as a first step towards developing foundation models in fingerprints. Our findings indicate fine-tuning of vision and language encoders improves the performance by 7%-39%. Our codes are available at this https URL.
>
---
#### [replaced 066] BiCLIP: Domain Canonicalization via Structured Geometric Transformation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于跨模态对齐任务，旨在解决视觉语言模型在不同领域中的适应问题。通过引入BiCLIP框架，利用少量样本估计几何变换，提升跨模态对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.08942](https://arxiv.org/pdf/2603.08942)**

> **作者:** Pranav Mantini; Shishir K. Shah
>
> **备注:** Accepted at Domain Generalization: Evolution, Breakthroughs, and Future Horizons Workshop at CVPR 2026
>
> **摘要:** Recent advances in vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities, yet adapting these models to specialized domains remains a significant challenge. Building on recent theoretical insights suggesting that independently trained VLMs are related by a canonical transformation, we extend this understanding to the concept of domains. We hypothesize that image features across disparate domains are related by a canonicalized geometric transformation that can be recovered using a small set of anchors. Few-shot classification provides a natural setting for this alignment, as the limited labeled samples serve as the anchors required to estimate this transformation. Motivated by this hypothesis, we introduce BiCLIP, a framework that applies a targeted transformation to multimodal features to enhance cross-modal alignment. Our approach is characterized by its extreme simplicity and low parameter footprint. Extensive evaluations across 11 standard benchmarks, including EuroSAT, DTD, and FGVCAircraft, demonstrate that BiCLIP consistently achieves state-of-the-art results. Furthermore, we provide empirical verification of existing geometric findings by analyzing the orthogonality and angular distribution of the learned transformations, confirming that structured alignment is the key to robust domain adaptation. Code is available at this https URL
>
---
#### [replaced 067] GaNI: Global and Near Field Illumination Aware Neural Inverse Rendering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.15651](https://arxiv.org/pdf/2403.15651)**

> **作者:** Jiaye Wu; Saeed Hadadan; Geng Lin; Matthias Zwicker; David Jacobs; Roni Sengupta
>
> **摘要:** In this paper, we present GaNI, a Global and Near-field Illumination-aware neural inverse rendering technique that can reconstruct geometry, albedo, and roughness parameters from images of a scene captured with co-located light and camera. Existing inverse rendering techniques with co-located light-camera focus on single objects only, without modeling global illumination and near-field lighting more prominent in scenes with multiple objects. We introduce a system that solves this problem in two stages; we first reconstruct the geometry powered by neural volumetric rendering NeuS, followed by inverse neural radiosity that uses the previously predicted geometry to estimate albedo and roughness. However, such a naive combination fails and we propose multiple technical contributions that enable this two-stage approach. We observe that NeuS fails to handle near-field illumination and strong specular reflections from the flashlight in a scene. We propose to implicitly model the effects of near-field illumination and introduce a surface angle loss function to handle specular reflections. Similarly, we observe that invNeRad assumes constant illumination throughout the capture and cannot handle moving flashlights during capture. We propose a light position-aware radiance cache network and additional smoothness priors on roughness to reconstruct reflectance. Experimental evaluation on synthetic and real data shows that our method outperforms the existing co-located light-camera-based inverse rendering techniques. Our approach produces significantly better reflectance and slightly better geometry than capture strategies that do not require a dark room.
>
---
#### [replaced 068] Near OOD Detection for Vision-Language Prompt Learning with Contrastive Logit Score
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.16091](https://arxiv.org/pdf/2405.16091)**

> **作者:** Myong Chol Jung; Joanna Dipnall; Belinda Gabbe; He Zhao
>
> **备注:** Published at International Journal of Computer Vision (IJCV)
>
> **摘要:** Prompt learning has emerged as an efficient and effective method for fine-tuning vision-language models such as CLIP. While many studies have explored generalisation abilities of these models in few-shot classification tasks and a few studies have addressed far out-of-distribution (OOD) of the models, their potential for addressing near OOD detection remains underexplored. Existing methods either require training from scratch, need fine-tuning, or are not designed for vision-language prompt learning. To address this, we introduce the Contrastive Logit Score (CLS), a novel post-hoc, plug-and-play scoring function. CLS significantly improves near OOD detection of pre-trained vision-language prompt learning methods without modifying their model architectures or requiring retraining. Our method achieves up to an 11.67% improvement in AUROC for near OOD detection with minimal computational overhead. Extensive evaluations validate the effectiveness, efficiency, and generalisability of our approach. Our code is available at this https URL.
>
---
#### [replaced 069] LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于端到端自动驾驶任务，旨在解决专家与学习者之间的信息不对称问题，通过改进模仿学习提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2512.20563](https://arxiv.org/pdf/2512.20563)**

> **作者:** Long Nguyen; Micha Fauth; Bernhard Jaeger; Daniel Dauner; Maximilian Igl; Andreas Geiger; Kashyap Chitta
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Simulators can generate virtually unlimited driving data, yet imitation learning policies in simulation still struggle to achieve robust closed-loop performance. Motivated by this gap, we empirically study how misalignment between privileged expert demonstrations and sensor-based student observations can limit the effectiveness of imitation learning. More precisely, experts have significantly higher visibility (e.g., ignoring occlusions) and far lower uncertainty (e.g., knowing other vehicles' actions), making them difficult to imitate reliably. Furthermore, navigational intent (i.e., the route to follow) is under-specified in student models at test time via only a single target point. We demonstrate that these asymmetries can measurably limit driving performance in CARLA and offer practical interventions to address them. After careful modifications to narrow the gaps between expert and student, our TransFuser v6 (TFv6) student policy achieves a new state of the art on all major publicly available CARLA closed-loop benchmarks, reaching 95 DS on Bench2Drive and more than doubling prior performances on Longest6~v2 and Town13. Additionally, by integrating perception supervision from our dataset into a shared sim-to-real pipeline, we show consistent gains on the NAVSIM and Waymo Vision-Based End-to-End driving benchmarks. Our code, data, and models are publicly available at this https URL.
>
---
#### [replaced 070] ITIScore: An Image-to-Text-to-Image Rating Framework for the Image Captioning Ability of MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.03765](https://arxiv.org/pdf/2604.03765)**

> **作者:** Zitong Xu; Huiyu Duan; Shengyao Qin; Guangyu Yang; Guangji Ma; Xiongkuo Min; Ke Gu; Guangtao Zhai; Patrick Le Callet
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have greatly improved image understanding and captioning capabilities. However, existing image captioning benchmarks typically suffer from limited diversity in caption length, the absence of recent advanced MLLMs, and insufficient human annotations, which potentially introduces bias and limits the ability to comprehensively assess the performance of modern MLLMs. To address these limitations, we present a new large-scale image captioning benchmark, termed, ICBench, which covers 12 content categories and consists of both short and long captions generated by 10 advanced MLLMs on 2K images, resulting in 40K captions in total. We conduct extensive human subjective studies to obtain mean opinion scores (MOSs) across fine-grained evaluation dimensions, where short captions are assessed in terms of fluency, relevance, and conciseness, while long captions are evaluated based on fluency, relevance, and completeness. Furthermore, we propose an automated evaluation metric, \textbf{ITIScore}, based on an image-to-text-to-image framework, which measures caption quality through reconstruction consistency. Experimental results demonstrate strong alignment between our automatic metric and human judgments, as well as robust zero-shot generalization ability on other public captioning datasets. Both the dataset and model will be released upon publication.
>
---
#### [replaced 071] CityGuard: Graph-Aware Private Descriptors for Bias-Resilient Identity Search Across Urban Cameras
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.18047](https://arxiv.org/pdf/2602.18047)**

> **作者:** Rong Fu; Yibo Meng; Jia Yee Tan; Jiaxuan Lu; Rui Lu; Jiekai Wu; Zhaolu Kang; Simon Fong
>
> **备注:** 36 pages, 12 figures
>
> **摘要:** City-scale person re-identification across distributed cameras must handle severe appearance changes from viewpoint, occlusion, and domain shift while complying with data protection rules that prevent sharing raw imagery. We introduce CityGuard, a topology-aware transformer for privacy-preserving identity retrieval in decentralized surveillance. The framework integrates three components. A dispersion-adaptive metric learner adjusts instance-level margins according to feature spread, increasing intra-class compactness. Spatially conditioned attention injects coarse geometry, such as GPS or deployment floor plans, into graph-based self-attention to enable projectively consistent cross-view alignment using only coarse geometric priors without requiring survey-grade calibration. Differentially private embedding maps are coupled with compact approximate indexes to support secure and cost-efficient deployment. Together these designs produce descriptors robust to viewpoint variation, occlusion, and domain shifts, and they enable a tunable balance between privacy and utility under rigorous differential-privacy accounting. Experiments on Market-1501 and additional public benchmarks, complemented by database-scale retrieval studies, show consistent gains in retrieval precision and query throughput over strong baselines, confirming the practicality of the framework for privacy-critical urban identity matching.
>
---
#### [replaced 072] Post-Processing Methods for Improving Accuracy in MRI Inpainting
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.15282](https://arxiv.org/pdf/2510.15282)**

> **作者:** Nishad Kulkarni; Krithika Iyer; Austin Tapp; Abhijeet Parida; Daniel Capellán-Martín; Zhifan Jiang; María J. Ledesma-Carbayo; Syed Muhammad Anwar; Marius George Linguraru
>
> **摘要:** Magnetic Resonance Imaging (MRI) is the primary imaging modality used in the diagnosis, assessment, and treatment planning for brain pathologies. However, most automated MRI analysis tools, such as segmentation and registration pipelines, are optimized for healthy anatomies and often fail when confronted with large lesions such as tumors. To overcome this, image inpainting techniques aim to locally synthesize healthy brain tissues in tumor regions, enabling the reliable application of general-purpose tools. In this work, we systematically evaluate state-of-the-art inpainting models and observe a saturation in their standalone performance. In response, we introduce a methodology combining model ensembling with efficient post-processing strategies such as median filtering, histogram matching, and pixel averaging. Further anatomical refinement is achieved via a lightweight U-Net enhancement stage. Comprehensive evaluation demonstrates that our proposed pipeline improves the anatomical plausibility and visual fidelity of inpainted regions, yielding higher accuracy and more robust outcomes than individual baseline models. By combining established models with targeted post-processing, we achieve improved and more accessible inpainting outcomes, supporting broader clinical deployment and sustainable, resource-conscious research. Our 2025 BraTS inpainting docker is available at this https URL.
>
---
#### [replaced 073] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.03723](https://arxiv.org/pdf/2604.03723)**

> **作者:** Guiyu Zhang; Yabo Chen; Xunzhi Xiang; Junchao Huang; Zhongyu Wang; Li Jiang
>
> **备注:** CVPR 2026
>
> **摘要:** Controlling both camera motion and object dynamics is essential for coherent and expressive video generation, yet current methods typically handle only one motion type or rely on ambiguous 2D cues that entangle camera-induced parallax with true object movement. We present SymphoMotion, a unified motion-control framework that jointly governs camera trajectories and object dynamics within a single model. SymphoMotion features a Camera Trajectory Control mechanism that integrates explicit camera paths with geometry-aware cues to ensure stable, structurally consistent viewpoint transitions, and an Object Dynamics Control mechanism that combines 2D visual guidance with 3D trajectory embeddings to enable depth-aware, spatially coherent object manipulation. To support large-scale training and evaluation, we further construct RealCOD-25K, a comprehensive real-world dataset containing paired camera poses and object-level 3D trajectories across diverse indoor and outdoor scenes, addressing a key data gap in unified motion control. Extensive experiments and user studies show that SymphoMotion significantly outperforms existing methods in visual fidelity, camera controllability, and object-motion accuracy, establishing a new benchmark for unified motion control in video generation. Codes and data are publicly available at this https URL.
>
---
#### [replaced 074] Towards Mitigating Modality Bias in Vision-Language Models for Temporal Action Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.21078](https://arxiv.org/pdf/2601.21078)**

> **作者:** Jiaqi Li; Guangming Wang; Shuntian Zheng; Minzhe Ni; Xiaoman Lu; Guanghui Ye; Yu Guan
>
> **摘要:** Temporal Action Localization (TAL) requires identifying both the boundaries and categories of actions in untrimmed videos. While vision-language models (VLMs) offer rich semantics to complement visual evidence, existing approaches tend to overemphasize linguistic priors at the expense of visual performance, leading to a pronounced modality bias. We propose ActionVLM, a vision-language aggregation framework that systematically mitigates modality bias in TAL. Our key insight is to preserve vision as the dominant signal while adaptively exploiting language only when beneficial. To this end, we introduce (i) a debiasing reweighting module that estimates the language advantage-the incremental benefit of language over vision-only predictions-and dynamically reweights language modality accordingly, and (ii) a residual aggregation strategy that treats language as a complementary refinement rather than the primary driver. This combination alleviates modality bias, reduces overconfidence from linguistic priors, and strengthens temporal reasoning. Experiments on THUMOS14 show that our model outperforms state-of-the-art by up to 3.2% mAP. Our code is available at this https URL
>
---
#### [replaced 075] Hide-and-Seek Attribution: Weakly Supervised Segmentation of Vertebral Metastases in CT
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.06849](https://arxiv.org/pdf/2512.06849)**

> **作者:** Matan Atad; Alexander W. Marka; Lisa Steinhelfer; Anna Curto-Vilalta; Yannik Leonhardt; Sarah C. Foreman; Anna-Sophia Walburga Dietrich; Robert Graf; Alexandra S. Gersing; Bjoern Menze; Daniel Rueckert; Jan S. Kirschke; Hendrik Möller
>
> **备注:** Accepted to MIDL 2026
>
> **摘要:** Accurate segmentation of vertebral metastasis in CT is clinically important yet difficult to scale, as voxel-level annotations are scarce and both lytic and blastic lesions often resemble benign degenerative changes. We introduce a 2D weakly supervised method trained solely on vertebra-level healthy/malignant labels, without any lesion masks. The method combines a Diffusion Autoencoder (DAE) that produces a classifier-guided healthy edit of each vertebra with pixel-wise difference maps that propose suspect candidate lesions. To determine which regions truly reflect malignancy, we introduce Hide-and-Seek Attribution: each candidate is revealed in turn while all others are hidden, the edited image is projected back to the data manifold by the DAE, and a latent-space classifier quantifies the isolated malignant contribution of that component. High-scoring regions form the final lytic or blastic segmentation. On held-out radiologist annotations, we achieve strong blastic/lytic performance despite no mask supervision (F1: 0.91/0.85; Dice: 0.87/0.78), exceeding baselines (F1: 0.79/0.67; Dice: 0.74/0.55). These results show that vertebra-level labels can be transformed into reliable lesion masks, demonstrating that generative editing combined with selective occlusion supports accurate weakly supervised segmentation in CT.
>
---
#### [replaced 076] Linear Attention Based Deep Nonlocal Means Filtering for Multiplicative Noise Removal
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2407.05087](https://arxiv.org/pdf/2407.05087)**

> **作者:** Xiao Siyao; Huang Libing; Zhang Shunsheng
>
> **摘要:** Multiplicative noise widely exists in radar images, medical images and other important fields' images. Compared to normal noises, multiplicative noise has a generally stronger effect on the visual expression of images. Aiming at the denoising problem of multiplicative noise, we linearize the nonlocal means algorithm with deep learning and propose a linear attention mechanism based deep nonlocal means filtering (LDNLM). Starting from the traditional nonlocal means filtering, we employ deep channel convolution neural networks to extract the information of the neighborhood matrix and obtain representation vectors of every pixel. Then we replace the similarity calculation and weighted averaging processes with the inner operations of the attention mechanism. To reduce the computational overhead, through the formula of similarity calculation and weighted averaging, we derive a nonlocal filter with linear complexity. Experiments on both simulated and real multiplicative noise demonstrate that the LDNLM is more competitive compared with the state-of-the-art methods. Additionally, we prove that the LDNLM possesses interpretability close to traditional NLM. The source code and pre-trained model are available at this https URL.
>
---
#### [replaced 077] A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09977](https://arxiv.org/pdf/2508.09977)**

> **作者:** Shuting He; Peilin Ji; Yitong Yang; Changshuo Wang; Jiayi Ji; Yinglin Wang; Henghui Ding
>
> **备注:** GitHub Repo: this https URL
>
> **摘要:** In the context of novel view synthesis, 3D Gaussian Splatting (3DGS) has recently emerged as an efficient and competitive counterpart to Neural Radiance Field (NeRF), enabling high-fidelity photorealistic rendering in real time. Beyond novel view synthesis, the explicit and compact nature of 3DGS enables a wide range of downstream applications that require geometric and semantic understanding. This survey provides a comprehensive overview of recent progress in 3DGS applications. It first reviews the reconstruction preliminaries of 3DGS, followed by the problem formulation, 2D foundation models, and related NeRF-based research areas that inform downstream 3DGS applications. We then categorize 3DGS applications into three foundational tasks: segmentation, editing, and generation, alongside additional functional applications built upon or tightly coupled with these foundational capabilities. For each, we summarize representative methods, supervision strategies, and learning paradigms, highlighting shared design principles and emerging trends. Commonly used datasets and evaluation protocols are also summarized, along with comparative analyses of recent methods across public benchmarks. To support ongoing research and development, a continually updated repository of papers, code, and resources is maintained at this https URL.
>
---
#### [replaced 078] Generalizable Deepfake Detection Based on Forgery-aware Layer Masking and Multi-artifact Subspace Decomposition
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2601.01041](https://arxiv.org/pdf/2601.01041)**

> **作者:** Xiang Zhang; Wenliang Weng; Daoyong Fu; Beijing Chen; Ziqiang Li; Ziwen He; Zhangjie Fu
>
> **摘要:** Deepfake detection remains highly challenging, particularly in cross-dataset scenarios and complex real-world settings. This challenge mainly arises because artifact patterns vary substantially across different forgery methods, whereas adapting pretrained models to such artifacts often overemphasizes forgery-specific cues and disturbs semantic representations, thereby weakening generalization. Existing approaches typically rely on full-parameter fine-tuning or auxiliary supervision to improve discrimination. However, they often struggle to model diverse forgery artifacts without compromising pretrained representations. To address these limitations, we propose FMSD, a deepfake detection framework built upon Forgery-aware Layer Masking and Multi-Artifact Subspace Decomposition. Specifically, Forgery-aware Layer Masking evaluates the bias-variance characteristics of layer-wise gradients to identify forgery-sensitive layers, thereby selectively updating them while reducing unnecessary disturbance to pretrained representations. Building upon this, Multi-Artifact Subspace Decomposition further decomposes the selected layer weights via Singular Value Decomposition (SVD) into a semantic subspace and multiple learnable artifact subspaces. These subspaces are optimized to capture heterogeneous and complementary forgery artifacts, enabling effective modeling of diverse forgery patterns while preserving pretrained semantic representations. Furthermore, orthogonality and spectral consistency constraints are imposed to regularize the artifact subspaces, reducing redundancy across them while preserving the overall spectral structure of pretrained weights.
>
---
#### [replaced 079] ParseBench: A Document Parsing Benchmark for AI Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08538](https://arxiv.org/pdf/2604.08538)**

> **作者:** Boyang Zhang; Sebastián G. Acosta; Preston Carlson; Sacha Bron; Pierre-Loïc Doulcet; Daniel B. Ospina; Simon Suo
>
> **摘要:** AI agents are changing the requirements for document parsing. What matters is semantic correctness: parsed output must preserve the structure and meaning needed for autonomous decisions, including correct table structure, precise chart data, semantically meaningful formatting, and visual grounding. Existing benchmarks do not fully capture this setting for enterprise automation, relying on narrow document distributions and text-similarity metrics that miss agent-critical failures. We introduce ParseBench, a benchmark of ${\sim}2{,}000$ human-verified pages from enterprise documents spanning insurance, finance, and government, organized around five capability dimensions: tables, charts, content faithfulness, semantic formatting, and visual grounding. Across 14 methods spanning vision-language models, specialized document parsers, and LlamaParse, the benchmark reveals a fragmented capability landscape: no method is consistently strong across all five dimensions. LlamaParse Agentic achieves the highest overall score at 84.9%, and the benchmark highlights the remaining capability gaps across current systems. Dataset and evaluation code are available on this https URL and this https URL.
>
---
#### [replaced 080] Exploring the best way for UAV visual localization under Low-altitude Multi-view Observation Condition: a Benchmark
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于UAV视觉定位任务，旨在解决低空多视角下的定位问题。构建了大规模数据集，提出统一框架并分析关键因素，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2503.10692](https://arxiv.org/pdf/2503.10692)**

> **作者:** Yibin Ye; Xichao Teng; Shuo Chen; Leqi Liu; Kun Wang; Xiaokai Song; Zhang Li
>
> **备注:** Accepted by CVPRF 2026 (Findings of the Conference on Computer Vision and Pattern Recognition 2026)
>
> **摘要:** Absolute Visual Localization (AVL) enables an Unmanned Aerial Vehicle (UAV) to determine its position in GNSS-denied environments by establishing geometric relationships between UAV images and geo-tagged reference maps. While many previous works have achieved AVL with image retrieval and matching techniques, research in low-altitude multi-view scenarios still remains limited. Low-altitude multi-view conditions present greater challenges due to extreme viewpoint changes. To investigate effective UAV AVL approaches under such conditions, we present this benchmark. Firstly, a large-scale low-altitude multi-view dataset called AnyVisLoc was constructed. This dataset includes 18,000 images captured at multiple scenes and altitudes, along with 2.5D reference maps containing aerial photogrammetry maps and historical satellite maps. Secondly, a unified framework was proposed to integrate the state-of-the-art AVL approaches and comprehensively test their performance. The best combined method was chosen as the baseline, and the key factors influencing localization accuracy are thoroughly analyzed based on it. This baseline achieved a 74.1% localization accuracy within 5 m under low-altitude, multi-view conditions. In addition, a novel retrieval metric called PDM@K was introduced to better align with the characteristics of the UAV AVL task. Overall, this benchmark revealed the challenges of low-altitude, multi-view UAV AVL and provided valuable guidance for future research. The dataset and code are available at this https URL
>
---
#### [replaced 081] Can LLMs Reason About Attention? Towards Zero-Shot Analysis of Multimodal Classroom Behavior
- **分类: cs.HC; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.03401](https://arxiv.org/pdf/2604.03401)**

> **作者:** Nolan Platt; Sehrish Nizamani; Alp Tural; Elif Tural; Saad Nizamani; Andrew Katz; Yoonje Lee; Nada Basit
>
> **备注:** 8 pages, 2 figures. Preprint
>
> **摘要:** Understanding student engagement usually requires time-consuming manual observation or invasive recording that raises privacy concerns. We present a privacy-preserving pipeline that analyzes classroom videos to extract insights about student attention, without storing any identifiable footage. Our system runs on a single GPU, using OpenPose for skeletal extraction and Gaze-LLE for visual attention estimation. Original video frames are deleted immediately after pose extraction, thus only geometric coordinates (stored as JSON) are retained, ensuring compliance with FERPA. The extracted pose and gaze data is processed by QwQ-32B-Reasoning, which performs zero-shot analysis of student behavior across lecture segments. Instructors access results through a web dashboard featuring attention heatmaps and behavioral summaries. Our preliminary findings suggest that LLMs may show promise for multimodal behavior understanding, although they still struggle with spatial reasoning about classroom layouts. We discuss these limitations and outline directions for improving LLM spatial comprehension in educational analytics contexts.
>
---
#### [replaced 082] Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08995](https://arxiv.org/pdf/2604.08995)**

> **作者:** Zile Wang; Zexiang Liu; Jiaxing Li; Kaichen Huang; Baixin Xu; Fei Kang; Mengyin An; Peiyu Wang; Biao Jiang; Yichen Wei; Yidan Xietian; Jiangbo Pei; Liang Hu; Boyi Jiang; Hua Xue; Zidong Wang; Haofeng Sun; Wei Li; Wanli Ouyang; Xianglong He; Yang Liu; Yangguang Li; Yahui Zhou
>
> **备注:** Project page: this https URL
>
> **摘要:** With the advancement of interactive video generation, diffusion models have increasingly demonstrated their potential as world models. However, existing approaches still struggle to simultaneously achieve memory-enabled long-term temporal consistency and high-resolution real-time generation, limiting their applicability in real-world scenarios. To address this, we present Matrix-Game 3.0, a memory-augmented interactive world model designed for 720p real-time longform video generation. Building upon Matrix-Game 2.0, we introduce systematic improvements across data, model, and inference. First, we develop an upgraded industrial-scale infinite data engine that integrates Unreal Engine-based synthetic data, large-scale automated collection from AAA games, and real-world video augmentation to produce high-quality Video-Pose-Action-Prompt quadruplet data at scale. Second, we propose a training framework for long-horizon consistency: by modeling prediction residuals and re-injecting imperfect generated frames during training, the base model learns self-correction; meanwhile, camera-aware memory retrieval and injection enable the base model to achieve long horizon spatiotemporal consistency. Third, we design a multi-segment autoregressive distillation strategy based on Distribution Matching Distillation (DMD), combined with model quantization and VAE decoder pruning, to achieve efficient real-time inference. Experimental results show that Matrix-Game 3.0 achieves up to 40 FPS real-time generation at 720p resolution with a 5B model, while maintaining stable memory consistency over minute-long sequences. Scaling up to a 2x14B model further improves generation quality, dynamics, and generalization. Our approach provides a practical pathway toward industrial-scale deployable world models.
>
---
#### [replaced 083] On the Effectiveness of Textual Prompting with Lightweight Fine-Tuning for SAM3 Remote Sensing Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15564](https://arxiv.org/pdf/2512.15564)**

> **作者:** Roni Blushtein-Livnon; Osher Rafaeli; David Ioffe; Amir Boger; Karen Sandberg Esquenazi; Tal Svoray
>
> **摘要:** Remote sensing (RS) image segmentation is constrained by the limited availability of annotated data and a gap between overhead imagery and natural images used to train foundational models. This motivates effective adaptation under limited supervision. SAM3 concept-driven framework generates masks from textual prompts without requiring task-specific modifications, which may enable this adaptation. We evaluate SAM3 for RS imagery across four target types, comparing textual, geometric, and hybrid prompting strategies, under lightweight fine-tuning scales with increasing supervision, alongside zero-shot inference. Results show that combining semantic and geometric cues yields the highest performance across targets and metrics. Text-only prompting exhibits the lowest performance, with marked score gaps for irregularly shaped targets, reflecting limited semantic alignment between SAM3 textual representations and their overhead appearances. Nevertheless, textual prompting with light fine-tuning offers a practical performance-effort trade-off for geometrically regular and visually salient targets. Across targets, performance improves between zero-shot inference and fine-tuning, followed by diminishing returns as the supervision scale increases. Namely, a modest geometric annotation effort is sufficient for effective adaptation. A persistent gap between Precision and IoU further indicates that under-segmentation and boundary inaccuracies remain prevalent error patterns in RS tasks, particularly for irregular and less prevalent targets.
>
---
#### [replaced 084] XD-MAP: Cross-Modal Domain Adaptation via Semantic Parametric Maps for Scalable Training Data Generation
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [https://arxiv.org/pdf/2601.14477](https://arxiv.org/pdf/2601.14477)**

> **作者:** Frank Bieder; Hendrik Königshof; Haohao Hu; Fabian Immel; Yinzhe Shen; Jan-Hendrik Pauls; Christoph Stiller
>
> **备注:** 10 pages, 7 figures, 3 tables, accepted at CVPRW
>
> **摘要:** Until open-world foundation models match the performance of specialized approaches, deep learning systems remain dependent on task- and sensor-specific data availability. To bridge the gap between available datasets and deployment domains, domain adaptation strategies are widely used. In this work, we propose XD-MAP, a novel approach to transfer sensor-specific knowledge from an image dataset to LiDAR, an entirely different sensing domain. Our method leverages detections on camera images to create a semantic parametric map. The map elements are modeled to produce pseudo labels in the target domain without any manual annotation effort. Unlike previous domain transfer approaches, our method does not require direct overlap between sensors and enables extending the angular perception range from a front-view camera to a full 360° view. On our large-scale road feature dataset, XD-MAP outperforms single shot baseline approaches by +19.5 mIoU for 2D semantic segmentation, +19.5 PQth for 2D panoptic segmentation, and +32.3 mIoU in 3D semantic segmentation. The results demonstrate the effectiveness of our approach achieving strong performance on LiDAR data without any manual labeling.
>
---
#### [replaced 085] TaleDiffusion: Multi-Character Story Generation with Dialogue Rendering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.04123](https://arxiv.org/pdf/2509.04123)**

> **作者:** Ayan Banerjee; Josep Llados; Umapada Pal; Anjan Dutta
>
> **摘要:** Text-to-story visualization is challenging due to the need for consistent interaction among multiple characters across frames. Existing methods struggle with character consistency, leading to artifact generation and inaccurate dialogue rendering, which results in disjointed storytelling. In response, we introduce TaleDiffusion, a novel framework for generating multi-character stories with an iterative process, maintaining character consistency, and accurate dialogue assignment via postprocessing. Given a story, we use a pre-trained LLM to generate per-frame descriptions, character details, and dialogues via in-context learning, followed by a bounded attention-based per-box mask technique to control character interactions and minimize artifacts. We then apply an identity-consistent self-attention mechanism to ensure character consistency across frames and region-aware cross-attention for precise object placement. Dialogues are also rendered as bubbles and assigned to characters via CLIPSeg. Experimental results demonstrate that TaleDiffusion outperforms existing methods in consistency, noise reduction, and dialogue rendering.
>
---
#### [replaced 086] Optimization-Guided Diffusion for Interactive Scene Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07661](https://arxiv.org/pdf/2512.07661)**

> **作者:** Shihao Li; Naisheng Ye; Tianyu Li; Kashyap Chitta; Tuo An; Peng Su; Boyang Wang; Haiou Liu; Chen Lv; Hongyang Li
>
> **摘要:** Realistic and diverse multi-agent driving scenes are crucial for evaluating autonomous vehicles, but safety-critical events which are essential for this task are rare and underrepresented in driving datasets. Data-driven scene generation offers a low-cost alternative by synthesizing complex traffic behaviors from existing driving logs. However, existing models often lack controllability or yield samples that violate physical or social constraints, limiting their usability. We present OMEGA, an optimization-guided, training-free framework that enforces structural consistency and interaction awareness during diffusion-based sampling from a scene generation model. OMEGA re-anchors each reverse diffusion step via constrained optimization, steering the generation towards physically plausible and behaviorally coherent trajectories. Building on this framework, we formulate ego-attacker interactions as a game-theoretic optimization in the distribution space, approximating Nash equilibria to generate realistic, safety-critical adversarial scenarios. Experiments on nuPlan and Waymo show that OMEGA improves generation realism, consistency, and controllability, increasing the ratio of physically and behaviorally valid scenes from 32.35% to 72.27% for free exploration capabilities, and from 11% to 80% for controllability-focused generation. Our approach can also generate $5\times$ more near-collision frames with a time-to-collision under three seconds while maintaining the overall scene realism.
>
---
#### [replaced 087] Improving the Reasoning of Multi-Image Grounding in MLLMs via Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.00748](https://arxiv.org/pdf/2507.00748)**

> **作者:** Bob Zhang; Haoran Li; Tao Zhang; Jianan Li; Cilin Yan; Xikai Liu; Jiayin Cai; Yanbin Hao
>
> **备注:** 5 pages
>
> **摘要:** Multimodal Large Language Models (MLLMs) perform well in single-image visual grounding but struggle with real-world tasks that demand cross-image reasoning and multi-modal instructions. To address this, we adopt a reinforcement learning (RL) based post-training strategy for MLLMs in multi-image grounding tasks. We first synthesize high-quality chain-of-thought (CoT) data for cold-start initialization, followed by supervised fine-tuning (SFT) using low-rank adaptation (LoRA). Subsequently, we apply rejection sampling with the merged SFT model to curate reliable RL data and use rule-based RL to guide the model toward optimal reasoning paths. Extensive experiments demonstrate the effectiveness of our approach, achieving +9.04% on MIG-Bench and +4.41% on average across seven out-of-domain benchmarks.
>
---
#### [replaced 088] IMPLICITSTAINER: Resolution Agnostic Data-Efficient Virtual Staining Using Neural Implicit Functions
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.09831](https://arxiv.org/pdf/2505.09831)**

> **作者:** Tushar Kataria; Beatrice Knudsen; Shireen Y. Elhabian
>
> **摘要:** Hematoxylin and eosin (H&E)-stained slides are central to cancer diagnosis and monitoring, visualizing tissue architecture and cellular morphology. However, H&E lacks the molecular specificity needed to distinguish cell states and functional activation. Antibody-based stains, such as immunohistochemistry (IHC), are therefore required to identify specific phenotypes (e.g., CD3$^+$ T cells or HER2-positive tumor cells) but are costly, time-consuming, and not universally available. Deep learning-based image translation methods, often termed virtual staining, offer a complementary alternative by generating virtual immunostains directly from H&E images. Most existing virtual staining methods are patch-based and operate at fixed resolutions, often requiring large datasets and additional post-hoc super-resolution models to generate high-resolution images. Furthermore, GAN- and diffusion-based approaches introduce stochasticity into generated stains which, although beneficial for visual realism in natural images, can lead to hallucinations and structural distortions that affect the accuracy and reliability required for clinical use. We propose IMPLICITSTAINER, a deterministic framework that reformulates virtual staining as a continuous pixel-level translation problem. In contrast to existing patch-based approaches, IMPLICITSTAINER formulates image translation as a continuous spatial mapping using neural implicit deep learning models. Each target-domain (IHC) pixel is predicted from a high-dimensional embedding of the corresponding source-domain H&E pixel, its local spatial neighborhood, and explicit coordinate information. IMPLICITSTAINER enables resolution-agnostic inference, improves robustness in low-data regimes, and yields deterministic, reproducible outputs. Across more than twenty baselines, IMPLICITSTAINER achieves SOTA performance on virtual staining tasks, including IHC and mIF.
>
---
#### [replaced 089] Inferring Dynamic Physical Properties from Video Foundation Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.02311](https://arxiv.org/pdf/2510.02311)**

> **作者:** Guanqi Zhan; Xianzheng Ma; Weidi Xie; Andrew Zisserman
>
> **摘要:** We study the task of predicting dynamic physical properties from videos. More specifically, we consider physical properties that require temporal information to be inferred: elasticity of a bouncing object, viscosity of a flowing liquid, and dynamic friction of an object sliding on a surface. To this end, we make the following contributions: (i) We collect a new video dataset for each physical property, consisting of synthetic training and testing splits, as well as a real split for real world evaluation. (ii) We explore three ways to infer the physical property from videos: (a) an oracle method where we supply the visual cues that intrinsically reflect the property using classical computer vision techniques; (b) a simple read out mechanism using a visual prompt and trainable prompt vector for cross-attention on pre-trained video generative and self-supervised models; and (c) prompt strategies for Multi-modal Large Language Models (MLLMs). (iii) We show that a video foundation model trained in a generative (DynamiCrafter) or trained in a self-supervised manner (V-JEPA-2) achieve a generally similar performance, though behind that of the oracle, and that MLLMs are currently inferior to the other models, though their performance can be improved through suitable prompting. The dataset, model, and code are available at this https URL.
>
---
#### [replaced 090] A Survey on Deep Learning Techniques for Action Anticipation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2309.17257](https://arxiv.org/pdf/2309.17257)**

> **作者:** Zeyun Zhong; Manuel Martin; Michael Voit; Juergen Gall; Jürgen Beyerer
>
> **备注:** If any relevant references are missing, please contact the authors for future inclusion
>
> **摘要:** The ability to anticipate possible future human actions is essential for a wide range of applications, including autonomous driving and human-robot interaction. Consequently, numerous methods have been introduced for action anticipation in recent years, with deep learning-based approaches being particularly popular. In this work, we review the recent advances of action anticipation algorithms with a particular focus on daily-living scenarios. Additionally, we classify these methods according to their primary contributions and summarize them in tabular form, allowing readers to grasp the details at a glance. Furthermore, we delve into the common evaluation metrics and datasets used for action anticipation and provide future directions with systematical discussions.
>
---
#### [replaced 091] Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12982](https://arxiv.org/pdf/2512.12982)**

> **作者:** Ziheng Qin; Yuheng Ji; Renshuai Tao; Yuxuan Tian; Yuyang Liu; Yipu Wang; Xiaolong Zheng
>
> **摘要:** The pursuit of a universal AI-generated image (AIGI) detector often relies on aggregating data from numerous generators to improve generalization. However, this paper identifies a paradoxical phenomenon we term the Benefit then Conflict dilemma, where detector performance stagnates and eventually degrades as source diversity expands. Our systematic analysis, diagnoses this failure by identifying two core issues: severe data-level heterogeneity, which causes the feature distributions of real and synthetic images to increasingly overlap, and a critical model-level bottleneck from fixed, pretrained encoders that cannot adapt to the rising complexity. To address these challenges, we propose Generator-Aware Prototype Learning (GAPL), a framework that constrain representation with a structured learning paradigm. GAPL learns a compact set of canonical forgery prototypes to create a unified, low-variance feature space, effectively countering data this http URL resolve the model bottleneck, it employs a two-stage training scheme with Low-Rank Adaptation, enhancing its discriminative power while preserving valuable pretrained knowledge. This approach establishes a more robust and generalizable decision boundary. Through extensive experiments, we demonstrate that GAPL achieves state-of-the-art performance, showing superior detection accuracy across a wide variety of GAN and diffusion-based generators. Code is available at this https URL
>
---
#### [replaced 092] Intelligent bear deterrence system based on computer vision: Reducing human bear conflicts in remote areas
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.23178](https://arxiv.org/pdf/2503.23178)**

> **作者:** Pengyu Chen; Teng Fei; Yunyan Du; Jiawei Yi; Yi Li; John A. Kupfer
>
> **摘要:** Conflicts between humans and bears on the Tibetan Plateau present substantial threats to local communities and hinder wildlife preservation initiatives. This research introduces a novel strategy that incorporates computer vision alongside Internet of Things (IoT) technologies to alleviate these issues. Tailored specifically for the harsh environment of the Tibetan Plateau, the approach utilizes the K210 development board paired with the YOLO object detection framework along with a tailored bear-deterrent mechanism, offering minimal energy usage and real-time efficiency in bear identification and deterrence. The model's performance was evaluated experimentally, achieving a mean Average Precision (mAP) of 91.4%, demonstrating excellent precision and dependability. By integrating energy-efficient components, the proposed system effectively surpasses the challenges of remote and off-grid environments, ensuring uninterrupted operation in secluded locations. This study provides a viable, eco-friendly, and expandable solution to mitigate human-bear conflicts, thereby improving human safety and promoting bear conservation in isolated areas like Yushu, China.
>
---
#### [replaced 093] Exploring Cross-Modal Flows for Few-Shot Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.14543](https://arxiv.org/pdf/2510.14543)**

> **作者:** Ziqi Jiang; Yanghao Wang; Long Chen
>
> **备注:** Accepted by ICLR2026
>
> **摘要:** Aligning features from different modalities, is one of the most fundamental challenges for cross-modal tasks. Although pre-trained vision-language models can achieve a general alignment between image and text, they often require parameter-efficient fine-tuning (PEFT) for further adjustment. Today's PEFT methods (e.g., prompt tuning, LoRA-based, or adapter-based) always selectively fine-tune a subset of parameters, which can slightly adjust either visual or textual features, and avoid overfitting. In this paper, we are the first to highlight that all existing PEFT methods perform one-step adjustment. It is insufficient for complex (or difficult) datasets, where features of different modalities are highly entangled. To this end, we propose the first model-agnostic multi-step adjustment approach by learning a cross-modal velocity field: Flow Matching Alignment (FMA). Specifically, to ensure the correspondence between categories during training, we first utilize a fixed coupling strategy. Then, we propose a noise augmentation strategy to alleviate the data scarcity issue. Finally, we design an early-stopping solver, which terminates the transformation process earlier, improving both efficiency and accuracy. Compared with one-step PEFT methods, FMA has the multi-step rectification ability to achieve more precise and robust alignment. Extensive results have demonstrated that FMA can consistently yield significant performance gains across various benchmarks and backbones, particularly on challenging datasets.
>
---
#### [replaced 094] RL makes MLLMs see better than SFT
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.16333](https://arxiv.org/pdf/2510.16333)**

> **作者:** Junha Song; Sangdoo Yun; Dongyoon Han; Jaegul Choo; Byeongho Heo
>
> **摘要:** A dominant assumption in Multimodal Language Model (MLLM) research is that its performance is largely inherited from the LLM backbone, given its immense parameter scale and remarkable capabilities. This has created a void in the understanding of the vision encoder, which determines how MLLMs perceive images. The recent shift in MLLM training paradigms, from Supervised Finetuning (SFT) to Reinforcement Learning (RL), magnifies this oversight-namely, the significant lack of analysis on how such training reshapes the vision encoder as well as the MLLM. To address this, we first investigate the impact of training strategies on MLLMs, where RL shows a clear advantage over SFT in strongly vision-related VQA benchmarks. Motivated by this, we conduct a critical yet under-explored analysis of the vision encoder of MLLMs through diverse and in-depth experiments, ranging from ImageNet classification and segmentation to gradient visualization. Our results demonstrate that MLLM's post-training strategy (i.e., SFT or RL) not only leads to distinct outcomes on MLLM downstream tasks, but also fundamentally reshapes MLLM's underlying visual representations. Specifically, the key finding of our study is that RL produces stronger and precisely localized visual representations compared to SFT, boosting the ability of the vision encoder for MLLM. We then reframe our findings into a simple recipe for building strong vision encoders for MLLMs, Preference-Instructed Vision OpTimization (PIVOT). When integrated into MLLMs, a PIVOT-trained vision encoder outperforms even larger and more heavily-trained counterparts, despite requiring less than 1% of the computational cost of standard vision pretraining. This result opens an effective and efficient path for advancing the vision backbones of MLLMs. Project page available at this https URL
>
---
#### [replaced 095] ELT: Elastic Looped Transformers for Visual Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.09168](https://arxiv.org/pdf/2604.09168)**

> **作者:** Sahil Goyal; Swayam Agrawal; Gautham Govind Anil; Prateek Jain; Sujoy Paul; Aditya Kusupati
>
> **摘要:** We introduce Elastic Looped Transformers (ELT), a highly parameter-efficient class of visual generative models based on a recurrent transformer architecture. While conventional generative models rely on deep stacks of unique transformer layers, our approach employs iterative, weight-shared transformer blocks to drastically reduce parameter counts while maintaining high synthesis quality. To effectively train these models for image and video generation, we propose the idea of Intra-Loop Self Distillation (ILSD), where student configurations (intermediate loops) are distilled from the teacher configuration (maximum training loops) to ensure consistency across the model's depth in a single training step. Our framework yields a family of elastic models from a single training run, enabling Any-Time inference capability with dynamic trade-offs between computational cost and generation quality, with the same parameter count. ELT significantly shifts the efficiency frontier for visual synthesis. With $4\times$ reduction in parameter count under iso-inference-compute settings, ELT achieves a competitive FID of $2.0$ on class-conditional ImageNet $256 \times 256$ and FVD of $72.8$ on class-conditional UCF-101.
>
---
#### [replaced 096] Perceptual Inductive Bias Is What You Need Before Contrastive Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.01201](https://arxiv.org/pdf/2506.01201)**

> **作者:** Tianqin Li; Junru Zhao; Dunhan Jiang; Shenghao Wu; Alan Ramirez; Tai Sing Lee
>
> **备注:** CVPR 2025. Tianqin Li and Junru Zhao contributed equally to this work. Due to a formatting error during the CVPR submission, the equal contribution note was omitted in the official proceedings. This arXiv version corrects that oversight. The author order follows alphabetical order by last name. Code: this https URL
>
> **摘要:** David Marr's seminal theory of human perception stipulates that visual processing is a multi-stage process, prioritizing the derivation of boundary and surface properties before forming semantic object representations. In contrast, contrastive representation learning frameworks typically bypass this explicit multi-stage approach, defining their objective as the direct learning of a semantic representation space for objects. While effective in general contexts, this approach sacrifices the inductive biases of vision, leading to slower convergence speed and learning shortcut resulting in texture bias. In this work, we demonstrate that leveraging Marr's multi-stage theory-by first constructing boundary and surface-level representations using perceptual constructs from early visual processing stages and subsequently training for object semantics-leads to 2x faster convergence on ResNet18, improved final representations on semantic segmentation, depth estimation, and object recognition, and enhanced robustness and out-of-distribution capability. Together, we propose a pretraining stage before the general contrastive representation pretraining to further enhance the final representation quality and reduce the overall convergence time via inductive bias from human vision systems.
>
---
#### [replaced 097] StableTTA: Training-Free Test-Time Adaptation that Improves Model Accuracy on ImageNet1K to 96%
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.04552](https://arxiv.org/pdf/2604.04552)**

> **作者:** Zheng Li; Jerry Cheng; Huanying Helen Gu
>
> **备注:** 21 pages, 8 figures, 6 tables
>
> **摘要:** Ensemble methods are widely used to improve predictive performance, but their effectiveness often comes at the cost of increased memory usage and computational complexity. In this paper, we identify a conflict in aggregation strategies that negatively impacts prediction stability. We propose test-time adaptation (StableTTA), a training-free method employs novel image and logit processing. Empirical results on ImageNet-1K show gains of 10.93\%-32.82\% in top-1 accuracy, with 33 models achieving over 95\% accuracy and several surpassing 96\%. Notably, StableTTA allows lightweight architectures to outperform ViT by 11.75\% in top-1 accuracy while reducing parameter count and computational cost by 97.1\% and 89.1\%, respectively, enabling high-accuracy inference on resource-constrained devices. Code is available at: this https URL, including a 3-minute reproduction demo.
>
---
#### [replaced 098] Iterative Inference-time Scaling with Adaptive Frequency Steering for Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23532](https://arxiv.org/pdf/2512.23532)**

> **作者:** Hexin Zhang; Dong Li; Jie Huang; Bingzhou Wang; Xueyang Fu; Zhengjun Zha
>
> **摘要:** Diffusion models have become a leading paradigm for image super-resolution (SR), but existing methods struggle to guarantee both the high-frequency perceptual quality and the low-frequency structural fidelity of generated images. Although inference-time scaling can theoretically improve this trade-off by allocating more computation, existing strategies remain suboptimal: reward-driven particle optimization often causes perceptual over-smoothing, while optimal-path search tends to lose structural consistency. To overcome these difficulties, we propose Iterative Diffusion Inference-Time Scaling with Adaptive Frequency Steering (IAFS), a training-free framework that jointly leverages iterative refinement and frequency-aware particle fusion. IAFS addresses the challenge of balancing perceptual quality and structural fidelity by progressively refining the generated image through iterative correction of structural deviations. Simultaneously, it ensures effective frequency fusion by adaptively integrating high-frequency perceptual cues with low-frequency structural information, allowing for a more accurate and balanced reconstruction across different image details. Extensive experiments across multiple diffusion-based SR models show that IAFS effectively resolves the perception-fidelity conflict, yielding consistently improved perceptual detail and structural accuracy, and outperforming existing inference-time scaling methods.
>
---
#### [replaced 099] MVAD: A Benchmark Dataset for Multimodal AI-Generated Video-Audio Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00336](https://arxiv.org/pdf/2512.00336)**

> **作者:** Mengxue Hu; Yunfeng Diao; Changtao Miao; Zhiqing Guo; Jianshu Li; Zhe Li; Joey Tianyi Zhou
>
> **备注:** 7 pages,2 figures
>
> **摘要:** The rapid advancement of AI-generated multimodal video-audio content has raised significant concerns regarding information security and content authenticity. Existing synthetic video datasets predominantly focus on the visual modality alone, while the few incorporating audio are largely confined to facial deepfakes--a limitation that fails to address the expanding landscape of general multimodal AI-generated content and substantially impedes the development of trustworthy detection systems. To bridge this critical gap, we introduce the Multimodal Video-Audio Dataset (MVAD), the first comprehensive dataset specifically designed for detecting AI-generated multimodal video-audio content. Our dataset exhibits three key characteristics: (1) genuine multimodality with samples generated according to three realistic video-audio forgery patterns; (2) high perceptual quality achieved through diverse state-of-the-art generative models; and (3) comprehensive diversity spanning realistic and anime visual styles, four content categories (humans, animals, objects, and scenes), and four video-audio multimodal data types. Our dataset will be available at this https URL.
>
---
#### [replaced 100] Learning to Focus and Precise Cropping: A Reinforcement Learning Framework with Information Gaps and Grounding Loss for MLLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.27494](https://arxiv.org/pdf/2603.27494)**

> **作者:** Xuanpu Zhao; Zhentao Tan; Dianmo Sheng; Tianxiang Chen; Yao Liu; Yue Wu; Tao Gong; Qi Chu; Nenghai Yu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** To enhance the perception and reasoning capabilities of multimodal large language models in complex visual scenes, recent research has introduced agent-based workflows. In these works, MLLMs autonomously utilize image cropping tool to analyze regions of interest for question answering. While existing training strategies, such as those employing supervised fine-tuning and reinforcement learning, have made significant progress, our empirical analysis reveals a key limitation. We demonstrate the model's strong reliance on global input and its weak dependence on the details within the cropped region. To address this issue, we propose a novel two-stage reinforcement learning framework that does not require trajectory supervision. In the first stage, we introduce the ``Information Gap" mechanism by adjusting the granularity of the global image. This mechanism trains the model to answer questions by focusing on cropped key regions, driven by the information gain these regions provide. The second stage further enhances cropping precision by incorporating a grounding loss, using a small number of bounding box annotations. Experiments show that our method significantly enhances the model's attention to cropped regions, enabling it to achieve state-of-the-art performance on high-resolution visual question-answering benchmarks. Our method provides a more efficient approach for perceiving and reasoning fine-grained details in MLLMs. Code is available at: this https URL.
>
---
#### [replaced 101] Beyond the Beep: Scalable Collision Anticipation and Real-Time Explainability with BADAS-2.0
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出BADAS-2.0，用于车辆碰撞预警，解决长尾场景识别与实时解释问题，通过数据增强、模型压缩和可解释性分析提升性能。**

- **链接: [https://arxiv.org/pdf/2604.05767](https://arxiv.org/pdf/2604.05767)**

> **作者:** Roni Goldshmidt; Hamish Scott; Lorenzo Niccolini; Hernan Matzner
>
> **摘要:** We present BADAS-2.0, the second generation of our collision anticipation system, building on BADAS-1.0, which showed that fine-tuning V-JEPA2 on large-scale ego-centric dashcam data outperforms both academic baselines and production ADAS systems. BADAS-2.0 advances the state of the art along three axes. (i) Long-tail benchmark and accuracy: We introduce a 10-group long-tail benchmark targeting rare and safety-critical scenarios. To construct it, BADAS-1.0 is used as an active oracle to score millions of unlabeled drives and surface high-risk candidates for annotation. Combined with Nexar's Atlas platform for targeted data collection, this expands the dataset from 40k to 178,500 labeled videos (~2M clips), yielding consistent gains across all subgroups, with the largest improvements on the hardest long-tail cases. (ii) Knowledge distillation to edge: Domain-specific self-supervised pre-training on 2.25M unlabeled driving videos enables distillation into compact models, BADAS-2.0-Flash (86M) and BADAS-2.0-Flash-Lite (22M), achieving 7-12x speedup with near-parity accuracy, enabling real-time edge deployment. (iii) Explainability: BADAS-2.0 produces real-time object-centric attention heatmaps that localize the evidence behind predictions. BADAS-Reason extends this with a vision-language model that consumes the last frame and heatmap to generate driver actions and structured textual reasoning. Inference code and evaluation benchmarks are publicly available.
>
---
#### [replaced 102] SpotFormer: Multi-Scale Spatio-Temporal Transformer for Facial Expression Spotting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.20799](https://arxiv.org/pdf/2407.20799)**

> **作者:** Yicheng Deng; Hideaki Hayashi; Hajime Nagahara
>
> **备注:** Accepted by IEEE Transactions on Affective Computing
>
> **摘要:** Facial expression spotting, identifying periods where facial expressions occur in a video, is a significant yet challenging task in facial expression analysis. The issues of irrelevant facial movements and the challenge of detecting subtle motions in micro-expressions remain unresolved, hindering accurate expression spotting. In this paper, we propose an efficient framework for facial expression spotting. First, we propose a Compact Sliding-Window-based Multi-temporal-Resolution Optical flow (CSW-MRO) feature, which calculates multi-temporal-resolution optical flow of the input image sequence within compact sliding windows. The window length is tailored to perceive complete micro-expressions and distinguish between general macro- and micro-expressions. CSW-MRO can effectively reveal subtle motions while avoiding the optical flow being dominated by head movements. Second, we propose SpotFormer, a multi-scale spatio-temporal Transformer that simultaneously encodes spatio-temporal relationships of the CSW-MRO features for accurate frame-level probability estimation. In SpotFormer, we use the proposed Facial Local Graph Pooling (FLGP) operation and convolutional layers to extract multi-scale spatio-temporal features. We show the validity of the architecture of SpotFormer by comparing it with several model variants. Third, we introduce supervised contrastive learning into SpotFormer to enhance the discriminability between different types of expressions. Extensive experiments on SAMM-LV, CAS(ME)^2, and CAS(ME)^3 show that our method outperforms state-of-the-art models, particularly in micro-expression spotting. Code is available at this https URL.
>
---
#### [replaced 103] Tango: Taming Visual Signals for Efficient Video Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.09547](https://arxiv.org/pdf/2604.09547)**

> **作者:** Shukang Yin; Sirui Zhao; Hanchao Wang; Baozhi Jia; Xianquan Wang; Chaoyou Fu; Enhong Chen
>
> **备注:** Code: this https URL
>
> **摘要:** Token pruning has emerged as a mainstream approach for developing efficient Video Large Language Models (Video LLMs). This work revisits and advances the two predominant token-pruning paradigms: attention-based selection and similarity-based clustering. Our study reveals two critical limitations in existing methods: (1) conventional top-k selection strategies fail to fully account for the attention distribution, which is often spatially multi-modal and long-tailed in magnitude; and (2) direct similarity-based clustering frequently generates fragmented clusters, resulting in distorted representations after pooling. To address these bottlenecks, we propose Tango, a novel framework designed to optimize the utilization of visual signals. Tango integrates a diversity-driven strategy to enhance attention-based token selection, and introduces Spatio-temporal Rotary Position Embedding (ST-RoPE) to preserve geometric structure via locality priors. Comprehensive experiments across various Video LLMs and video understanding benchmarks demonstrate the effectiveness and generalizability of our approach. Notably, when retaining only 10% of the video tokens, Tango preserves 98.9% of the original performance on LLaVA-OV while delivering a 1.88$\times$ inference speedup.
>
---
#### [replaced 104] INSPATIO-WORLD: A Real-Time 4D World Simulator via Spatiotemporal Autoregressive Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07209](https://arxiv.org/pdf/2604.07209)**

> **作者:** InSpatio Team; Donghui Shen; Guofeng Zhang; Haomin Liu; Haoyu Ji; Hujun Bao; Hongjia Zhai; Jialin Liu; Jing Guo; Nan Wang; Siji Pan; Weihong Pan; Weijian Xie; Xianbin Liu; Xiaojun Xiang; Xiaoyu Zhang; Xinyu Chen; Yifu Wang; Yipeng Chen; Zhenzhou Fan; Zhewen Le; Zhichao Ye; Ziqiang Zhao
>
> **摘要:** Building world models with spatial consistency and real-time interactivity remains a fundamental challenge in computer vision. Current video generation paradigms often struggle with a lack of spatial persistence and insufficient visual realism, making it difficult to support seamless navigation in complex environments. To address these challenges, we propose INSPATIO-WORLD, a novel real-time framework capable of recovering and generating high-fidelity, dynamic interactive scenes from a single reference video. At the core of our approach is a Spatiotemporal Autoregressive (STAR) architecture, which enables consistent and controllable scene evolution through two tightly coupled components: Implicit Spatiotemporal Cache aggregates reference and historical observations into a latent world representation, ensuring global consistency during long-horizon navigation; Explicit Spatial Constraint Module enforces geometric structure and translates user interactions into precise and physically plausible camera trajectories. Furthermore, we introduce Joint Distribution Matching Distillation (JDMD). By using real-world data distributions as a regularizing guide, JDMD effectively overcomes the fidelity degradation typically caused by over-reliance on synthetic data. Extensive experiments demonstrate that INSPATIO-WORLD significantly outperforms existing state-of-the-art (SOTA) models in spatial consistency and interaction precision, ranking first among real-time interactive methods on the WorldScore-Dynamic benchmark, and establishing a practical pipeline for navigating 4D environments reconstructed from monocular videos.
>
---
#### [replaced 105] Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02123](https://arxiv.org/pdf/2603.02123)**

> **作者:** Jiahao Huang; Fengyan Lin; Xuechao Yang; Chen Feng; Kexin Zhu; Xu Yang; Zhide Chen
>
> **备注:** This paper has been accepted by CVPR 2026
>
> **摘要:** The development of affective multimodal language models (MLMs) has long been constrained by a gap between low-level perception and high-level interaction, leading to fragmented affective capabilities and limited generalization. To bridge this gap, we propose a cognitively inspired three-level hierarchy that organizes affective tasks according to their cognitive depth-perception, understanding, and interaction-and provides a unified conceptual foundation for advancing affective modeling. Guided by this hierarchy, we introduce Nano-EmoX, a small-scale multitask MLM, and P2E (Perception-to-Empathy), a curriculum-based training framework. Nano-EmoX integrates a suite of omni-modal encoders, including an enhanced facial encoder and a fusion encoder, to capture key multimodal affective cues and improve cross-task transferability. The outputs are projected into a unified language space via heterogeneous adapters, empowering a lightweight language model to tackle diverse affective tasks. Concurrently, P2E progressively cultivates emotional intelligence by aligning rapid perception with chain-of-thought-driven empathy. To the best of our knowledge, Nano-EmoX is the first compact MLM (2.2B) to unify six core affective tasks across all three hierarchy levels, achieving state-of-the-art or highly competitive performance across multiple benchmarks, demonstrating excellent efficiency and generalization. The code is available at this https URL.
>
---
#### [replaced 106] FashionStylist: An Expert Knowledge-enhanced Multimodal Dataset for Fashion Understanding
- **分类: cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2604.09249](https://arxiv.org/pdf/2604.09249)**

> **作者:** Kaidong Feng; Zhuoxuan Huang; Huizhong Guo; Yuting Jin; Xinyu Chen; Yue Liang; Yifei Gai; Li Zhou; Yunshan Ma; Zhu Sun
>
> **摘要:** Fashion understanding requires both visual perception and expert-level reasoning about style, occasion, compatibility, and outfit rationale. However, existing fashion datasets remain fragmented and task-specific, often focusing on item attributes, outfit co-occurrence, or weak textual supervision, and thus provide limited support for holistic outfit understanding. In this paper, we introduce FashionStylist, an expert-annotated benchmark for holistic and expert-level fashion understanding. Constructed through a dedicated fashion-expert annotation pipeline, FashionStylist provides professionally grounded annotations at both the item and outfit levels. It supports three representative tasks: outfit-to-item grounding, outfit completion, and outfit evaluation. These tasks cover realistic item recovery from complex outfits with layering and accessories, compatibility-aware composition beyond co-occurrence matching, and expert-level assessment of style, season, occasion, and overall coherence. Experimental results show that FashionStylist serves not only as a unified benchmark for multiple fashion tasks, but also as an effective training resource for improving grounding, completion, and outfit-level semantic evaluation in MLLM-based fashion systems.
>
---
#### [replaced 107] HDR 3D Gaussian Splatting via Luminance-Chromaticity Decomposition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12895](https://arxiv.org/pdf/2511.12895)**

> **作者:** Kaixuan Zhang; Minxian Li; Mingwu Ren; Jiankang Deng; Xiatian Zhu
>
> **摘要:** High Dynamic Range (HDR) 3D reconstruction is pivotal for professional content creation in filmmaking and virtual production. Existing methods typically rely on multi-exposure Low Dynamic Range (LDR) supervision to constrain the learning process within vast brightness spaces, resulting in complex, dual-branch architectures. This work explores the feasibility of learning HDR 3D models exclusively in the HDR data space to simplify model design. By analyzing 3D Gaussian Splatting (3DGS) for HDR imagery, we reveal that its failure stems from the limited capacity of Spherical Harmonics (SH) to capture extreme radiance variations across views, often biasing towards high-radiance observations. While increasing SH orders improves training fitting, it leads to severe overfitting and excessive parameter overhead. To address this, we propose \textit{Luminance-Chromaticity Decomposition 3DGS} (LCD-GS). By decoupling luminance and chromaticity into independent parameters, LCD-GS significantly enhances learning flexibility with minimal parameter increase (\textit{e.g.}, one extra scalar per primitive). Notably, LCD-GS maintains the original training and inference pipeline, requiring only a change in color representation. Extensive experiments on synthetic and real datasets demonstrate that LCD-GS consistently outperforms state-of-the-art methods in reconstruction fidelity and dynamic-range preservation even with a simpler, more efficient architecture, providing an elegant paradigm for professional-grade HDR 3D modeling. Code and datasets will be released.
>
---
#### [replaced 108] StaMo: Unsupervised Learning of Generalizable Robot Motion from Compact State Representation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人运动学习任务，旨在解决状态表示冗余或信息不足的问题。通过无监督方法学习紧凑状态表示，提升机器人任务成功率和可解释性。**

- **链接: [https://arxiv.org/pdf/2510.05057](https://arxiv.org/pdf/2510.05057)**

> **作者:** Mingyu Liu; Jiuhe Shu; Hui Chen; Zeju Li; Canyu Zhao; Jiange Yang; Shenyuan Gao; Hao Chen; Chunhua Shen
>
> **摘要:** A fundamental challenge in embodied intelligence is developing expressive and compact state representations for efficient world modeling and decision making. However, existing methods often fail to achieve this balance, yielding representations that are either overly redundant or lacking in task-critical information. We propose an unsupervised approach that learns a highly compressed two-token state representation using a lightweight encoder and a pre-trained Diffusion Transformer (DiT) decoder, capitalizing on its strong generative prior. Our representation is efficient, interpretable, and integrates seamlessly into existing VLA-based models, improving performance by 14.3% on LIBERO and 30% in real-world task success with minimal inference overhead. More importantly, we find that the difference between these tokens, obtained via latent interpolation, naturally serves as a highly effective latent action, which can be further decoded into executable robot actions. This emergent capability reveals that our representation captures structured dynamics without explicit supervision. We name our method StaMo for its ability to learn generalizable robotic Motion from compact State representation, which is encoded from static images, challenging the prevalent dependence to learning latent action on complex architectures and video data. The resulting latent actions also enhance policy co-training, outperforming prior methods by 10.4% with improved interpretability. Moreover, our approach scales effectively across diverse data sources, including real-world robot data, simulation, and human egocentric video.
>
---
#### [replaced 109] Unified Multimodal Uncertain Inference
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.08701](https://arxiv.org/pdf/2604.08701)**

> **作者:** Dengjia Zhang; Alexander Martin; William Jurayj; Kenton Murray; Benjamin Van Durme; Reno Kriz
>
> **备注:** Update citations
>
> **摘要:** We introduce Unified Multimodal Uncertain Inference (UMUI), a multimodal inference task spanning text, audio, and video, where models must produce calibrated probability estimates of hypotheses conditioned on a premise in any modality or combination. While uncertain inference has been explored in text, extension to other modalities has been limited to single-modality binary entailment judgments, leaving no framework for fine-grained probabilistic reasoning in or across other modalities. To address this, we curate a human-annotated evaluation set with scalar probability judgments across audio, visual, and audiovisual settings, and additionally evaluate on existing text and audio benchmarks. We introduce CLUE (Calibrated Latent Uncertainty Estimation), which combines self-consistent teacher calibration and distribution-based confidence probing to produce calibrated predictions. We demonstrate that our 3B-parameter model achieves equivalent or stronger performance than baselines up to 32B parameters across all modalities.
>
---
#### [replaced 110] DisCo-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉楼图定位任务，解决 minimalist floorplan 中因重复结构导致的定位模糊问题。提出 DisCo-FLoc 方法，通过双层视觉几何对比，无需语义标签提升定位准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01822](https://arxiv.org/pdf/2601.01822)**

> **作者:** Shiyong Meng; Tao Zou; Bolei Chen; Chaoxu Mu; Jianxin Wang
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Since floorplan data is readily available, long-term persistent, and robust to changes in visual appearance, visual Floorplan Localization (FLoc) has garnered significant attention. Existing methods either ingeniously match geometric priors or utilize sparse semantics to reduce FLoc uncertainty. However, they still suffer from ambiguous FLoc caused by repetitive structures within minimalist floorplans. Moreover, expensive but limited semantic annotations restrict their applicability. To address these issues, we propose DisCo-FLoc, which utilizes dual-level visual-geometric Contrasts to Disambiguate depth-aware visual Floc, without requiring additional semantic labels. Our solution begins with a ray regression predictor tailored for ray-casting-based FLoc, predicting a series of FLoc candidates using depth estimation expertise. In addition, a novel contrastive learning method with position-level and orientation-level constraints is proposed to strictly match depth-aware visual features with the corresponding geometric structures in the floorplan. Such matches can effectively eliminate FLoc ambiguity and select the optimal imaging pose from FLoc candidates. Exhaustive comparative studies on two standard visual Floc benchmarks demonstrate that our method outperforms the state-of-the-art semantic-based method, achieving significant improvements in both robustness and accuracy.
>
---
#### [replaced 111] LPNSR: Optimal Noise-Guided Diffusion Image Super-Resolution Via Learnable Noise Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.21045](https://arxiv.org/pdf/2603.21045)**

> **作者:** Shuwei Huang; Shizhuo Liu; Zijun Wei
>
> **摘要:** Diffusion-based image super-resolution (SR) aims to reconstruct high-resolution (HR) images from low-resolution (LR) observations, yet faces a fundamental trade-off between inference efficiency and reconstruction quality in limited-step sampling scenarios. A critical yet underexplored question is: what is the optimal noise to inject at each intermediate diffusion step? In this paper, we establish a theoretical framework that derives the closed-form analytical solution for optimal intermediate noise in diffusion models from a maximum likelihood estimation perspective, revealing a consistent conditional dependence structure that generalizes across diffusion paradigms. We instantiate this framework under the residual-shifting diffusion paradigm and accordingly design an LR-guided multi-input-aware noise predictor to replace random Gaussian this http URL further mitigate initialization bias with a high-quality pre-upsampling network. The compact 4-step trajectory uniquely enables end-to-end optimization of the entire reverse chain, which is computationally prohibitive for conventional long-trajectory diffusion models. Extensive experiments demonstrate that LPNSR achieves state-of-the-art perceptual performance on both synthetic and real-world datasets, without relying on any large-scale text-to-image priors. The source code of our method can be found at this https URL.
>
---
#### [replaced 112] Auto-regressive transformation for image alignment
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.04864](https://arxiv.org/pdf/2505.04864)**

> **作者:** Kanggeon Lee; Soochahn Lee; Kyoung Mu Lee
>
> **摘要:** Existing methods for image alignment struggle in cases involving feature-sparse regions, extreme scale and field-of-view differences, and large deformations, often resulting in suboptimal accuracy. Robustness to these challenges can be improved through iterative refinement of the transform field while focusing on critical regions in multi-scale image representations. We thus propose Auto-Regressive Transformation (ART), a novel method that iteratively estimates the coarse-to-fine transformations through an auto-regressive pipeline. Leveraging hierarchical multi-scale features, our network refines the transform field parameters using randomly sampled points at each scale. By incorporating guidance from the cross-attention layer, the model focuses on critical regions, ensuring accurate alignment even in challenging, feature-limited conditions. Extensive experiments demonstrate that ART significantly outperforms state-of-the-art methods on planar images and achieves comparable performance on 3D scene images, establishing it as a powerful and versatile solution for precise image alignment.
>
---
#### [replaced 113] Decoupled Generative Modeling for Human-Object Interaction Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19049](https://arxiv.org/pdf/2512.19049)**

> **作者:** Hwanhee Jung; Seunggwan Lee; Jeongyoon Yoon; SeungHyeon Kim; Giljoo Nam; Qixing Huang; Sangpil Kim
>
> **摘要:** Synthesizing realistic human-object interaction (HOI) is essential for 3D computer vision and robotics, underpinning animation and embodied control. Existing approaches often require manually specified intermediate waypoints and place all optimization objectives on a single network, which increases complexity, reduces flexibility, and leads to errors such as unsynchronized human and object motion or penetration. To address these issues, we propose Decoupled Generative Modeling for Human-Object Interaction Synthesis (DecHOI), which separates path planning and action synthesis. A trajectory generator first produces human and object trajectories without prescribed waypoints, and an action generator conditions on these paths to synthesize detailed motions. To further improve contact realism, we employ adversarial training with a discriminator that focuses on the dynamics of distal joints. The framework also models a moving counterpart and supports responsive, long-sequence planning in dynamic scenes, while preserving plan consistency. Across two benchmarks, FullBodyManipulation and 3D-FUTURE, DecHOI surpasses prior methods on most quantitative metrics and qualitative evaluations, and perceptual studies likewise prefer our results.
>
---
#### [replaced 114] Interactive Interface For Semantic Segmentation Dataset Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.23470](https://arxiv.org/pdf/2506.23470)**

> **作者:** Ngoc-Do Tran; Minh-Tuan Huynh; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** The rapid advancement of AI and computer vision has significantly increased the demand for high-quality annotated datasets, particularly for semantic segmentation. However, creating such datasets is resource-intensive, requiring substantial time, labor, and financial investment, and often raises privacy concerns due to the use of real-world data. To mitigate these challenges, we present SynthLab, consisting of a modular platform for visual data synthesis and a user-friendly interface. The modular architecture of SynthLab enables easy maintenance, scalability with centralized updates, and seamless integration of new features. Each module handles distinct aspects of computer vision tasks, enhancing flexibility and adaptability. Meanwhile, its interactive, user-friendly interface allows users to quickly customize their data pipelines through drag-and-drop actions. Extensive user studies involving a diverse range of users across different ages, professions, and expertise levels, have demonstrated flexible usage, and high accessibility of SynthLab, enabling users without deep technical expertise to harness AI for real-world applications.
>
---
#### [replaced 115] Bag of Bags: Adaptive Visual Vocabularies for Genizah Join Image Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08138](https://arxiv.org/pdf/2604.08138)**

> **作者:** Sharva Gogawale; Gal Grudka; Daria Vasyutinsky-Shapira; Omer Ventura; Berat Kurar-Barakat; Nachum Dershowitz
>
> **摘要:** A join is a set of manuscript fragments identified as originally emanating from the same manuscript. We study manuscript join retrieval: Given a query image of a fragment, retrieve other fragments originating from the same physical manuscript. We propose Bag of Bags (BoB), an image-level representation that replaces the global-level visual codebook of classical Bag of Words (BoW) with a fragment-specific vocabulary of local visual words. Our pipeline trains a sparse convolutional autoencoder on binarized fragment patches, encodes connected components from each page, clusters the resulting embeddings with per-image k-means, and compares images using set-to-set distances between their local vocabularies. Evaluated on fragments from the Cairo Genizah, the best BoB variant (viz. Chamfer) achieves Hit@1 of 0.78 and MRR of 0.84, compared to 0.74 and 0.80, respectively, for the strongest BoW baseline (BoW-RawPatches-$\chi^2$), a 6.1% relative improvement in top-1 accuracy. We furthermore study a mass-weighted BoB-OT variant that incorporates cluster population into prototype matching and present a formal approximation guarantee bounding its deviation from full component-level optimal transport. A two-stage pipeline using a BoW shortlist followed by BoB-OT reranking provides a practical compromise between retrieval strength and computational cost, supporting applicability to larger manuscript collections. The code and dataset are available at this https URL.
>
---
#### [replaced 116] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2412.17574](https://arxiv.org/pdf/2412.17574)**

> **作者:** Ting Zhou; Daoyuan Chen; Qirui Jiao; Bolin Ding; Yaliang Li; Ying Shen
>
> **备注:** Accepted as a conference paper at CVPR 2026
>
> **摘要:** Evaluating the nuanced human-centric video understanding capabilities of Multimodal Large Language Models (MLLMs) remains a great challenge, as existing benchmarks often overlook the intricacies of emotion, behavior, and cross-modal alignment. We introduce HumanVBench, a comprehensive video benchmark designed to rigorously probe these capabilities across 16 fine-grained tasks. A cornerstone of our work is a novel and scalable benchmark construction methodology, featuring two automated pipelines that synthesize high-quality video annotations and challenging multiple-choice questions with minimal human labor. By leveraging state-of-the-art models for annotation and systematically converting model-induced errors into plausible distractors, our framework provides a generalizable ``machine'' for creating nuanced evaluation suites. Our extensive evaluation of 30 leading MLLMs on HumanVBench reveals critical deficiencies, particularly in perceiving subtle emotions and aligning speech with visual cues, with even top proprietary models falling short of human performance. We open-source HumanVBench and our synthesis pipelines to catalyze the development of more socially intelligent and capable video MLLMs.
>
---
#### [replaced 117] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于人机协作控制任务，旨在解决机器人在辅助场景中跟踪人类互动动作的问题。通过多智能体强化学习，训练出能感知并适应人类动态的控制策略。**

- **链接: [https://arxiv.org/pdf/2603.11346](https://arxiv.org/pdf/2603.11346)**

> **作者:** Yuto Shibata; Kashu Yamazaki; Lalit Jayanti; Yoshimitsu Aoki; Mariko Isogawa; Katerina Fragkiadaki
>
> **备注:** Accepted at CVPR 2026 (main). Project page: this https URL
>
> **摘要:** Humanoid robotics has strong potential to transform daily service and caregiving applications. Although recent advances in general motion tracking within physics engines (GMT) have enabled virtual characters and humanoid robots to reproduce a broad range of human motions, these behaviors are primarily limited to contact-less social interactions or isolated movements. Assistive scenarios, by contrast, require continuous awareness of a human partner and rapid adaptation to their evolving posture and dynamics. In this paper, we formulate the imitation of closely interacting, force-exchanging human-human motion sequences as a multi-agent reinforcement learning problem. We jointly train partner-aware policies for both the supporter (assistant) agent and the recipient agent in a physics simulator to track assistive motion references. To make this problem tractable, we introduce a partner policies initialization scheme that transfers priors from single-human motion-tracking controllers, greatly improving exploration. We further propose dynamic reference retargeting and contact-promoting reward, which adapt the assistant's reference motion to the recipient's real-time pose and encourage physically meaningful support. We show that AssistMimic is the first method capable of successfully tracking assistive interaction motions on established benchmarks, demonstrating the benefits of a multi-agent RL formulation for physically grounded and socially aware humanoid control.
>
---
#### [replaced 118] Seeing Through Deception: Uncovering Misleading Creator Intent in Multimodal News with Vision-Language Models
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于多模态虚假信息检测任务，旨在解决创作者意图误导的问题。工作包括构建DeceptionDecoded数据集，并评估VLMs在意图推理上的不足。**

- **链接: [https://arxiv.org/pdf/2505.15489](https://arxiv.org/pdf/2505.15489)**

> **作者:** Jiaying Wu; Fanxiao Li; Zihang Fu; Min-Yen Kan; Bryan Hooi
>
> **备注:** ICLR 2026
>
> **摘要:** The impact of multimodal misinformation arises not only from factual inaccuracies but also from the misleading narratives that creators deliberately embed. Interpreting such creator intent is therefore essential for multimodal misinformation detection (MMD) and effective information governance. To this end, we introduce DeceptionDecoded, a large-scale benchmark of 12,000 image-caption pairs grounded in trustworthy reference articles, created using an intent-guided simulation framework that models both the desired influence and the execution plan of news creators. The dataset captures both misleading and non-misleading cases, spanning manipulations across visual and textual modalities, and supports three intent-centric tasks: (1) misleading intent detection, (2) misleading source attribution, and (3) creator desire inference. We evaluate 14 state-of-the-art vision-language models (VLMs) and find that they struggle with intent reasoning, often relying on shallow cues such as surface-level alignment, stylistic polish, or heuristic authenticity signals. To bridge this, our framework systematically synthesizes data that enables models to learn implication-level intent reasoning. Models trained on DeceptionDecoded demonstrate strong transferability to real-world MMD, validating our framework as both a benchmark to diagnose VLM fragility and a data synthesis engine that provides high-quality, intent-focused resources for enhancing robustness in real-world multimodal misinformation governance.
>
---
#### [replaced 119] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03447](https://arxiv.org/pdf/2508.03447)**

> **作者:** Qiyu Chen; Zhen Qu; Wei Luo; Haiming Yao; Yunkang Cao; Yuxin Jiang; Yinan Duan; Huiyuan Luo; Chengkan Lv; Zhengtao Zhang
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** Recently, large pre-trained vision-language models have shown remarkable performance in zero-shot anomaly detection (ZSAD). With fine-tuning on a single auxiliary dataset, the model enables cross-category anomaly detection on diverse datasets covering industrial defects and medical lesions. Compared to manually designed prompts, prompt learning eliminates the need for expert knowledge and trial-and-error. However, it still faces the following challenges: (i) static learnable tokens struggle to capture the continuous and diverse patterns of normal and anomalous states, limiting generalization to unseen categories; (ii) fixed textual labels provide overly sparse category information, making the model prone to overfitting to a specific semantic subspace. To address these issues, we propose Conditional Prompt Synthesis (CoPS), a novel framework that synthesizes dynamic prompts conditioned on visual features to enhance ZSAD performance. Specifically, we extract representative normal and anomaly prototypes from fine-grained patch features and explicitly inject them into prompts, enabling adaptive state modeling. Given the sparsity of class labels, we leverage a variational autoencoder to model semantic image features and implicitly fuse varied class tokens into prompts. Additionally, integrated with our spatially-aware alignment mechanism, extensive experiments demonstrate that CoPS surpasses state-of-the-art methods by 1.4% in classification AUROC and 1.9% in segmentation AUROC across 13 industrial and medical datasets. The code is available at this https URL.
>
---
#### [replaced 120] Context-Aware Semantic Segmentation via Stage-Wise Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11310](https://arxiv.org/pdf/2601.11310)**

> **作者:** Antoine Carreaud; Elias Naha; Arthur Chansel; Nina Lahellec; Jan Skaloud; Adrien Gressin
>
> **摘要:** Semantic ultra-high-resolution (UHR) image segmentation is essential in remote sensing applications such as aerial mapping and environmental monitoring. Transformer-based models remain challenging in this setting because memory grows quadratically with the number of tokens, limiting either spatial resolution or contextual scope. We introduce CASWiT (Context-Aware Stage-Wise Transformer), a dual-branch Swin-based architecture that injects low-resolution contextual information into fine-grained high-resolution features through lightweight stage-wise cross-attention. To strengthen cross-scale learning, we also propose a SimMIM-style pretraining strategy based on masked reconstruction of the high-resolution image. Extensive experiments on the large-scale FLAIR-HUB aerial dataset demonstrate the effectiveness of CASWiT. Under our RGB-only UHR protocol, CASWiT reaches 66.37% mIoU with a SegFormer decoder, improving over strong RGB baselines while also improving boundary quality. On the URUR benchmark, CASWiT reaches 49.2% mIoU under the official evaluation protocol, and it also transfers effectively to medical UHR segmentation benchmarks. Code and pretrained models are available at this https URL
>
---
#### [replaced 121] PanoSAMic: Panoramic Image Segmentation from SAM Feature Encoding and Dual View Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.07447](https://arxiv.org/pdf/2601.07447)**

> **作者:** Mahdi Chamseddine; Didier Stricker; Jason Rambach
>
> **备注:** Accepted in ICPR 2026
>
> **摘要:** Existing image foundation models are not optimized for spherical images having been trained primarily on perspective images. PanoSAMic integrates the pre-trained Segment Anything (SAM) encoder to make use of its extensive training and integrate it into a semantic segmentation model for panoramic images using multiple modalities. We modify the SAM encoder to output multi-stage features and introduce a novel spatio-modal fusion module that allows the model to select the relevant modalities and best features from each modality for different areas of the input. Furthermore, our semantic decoder uses spherical attention and dual view fusion to overcome the distortions and edge discontinuity often associated with panoramic images. PanoSAMic achieves state-of-the-art (SotA) results on Stanford2D3DS for RGB, RGB-D, and RGB-D-N modalities and on Matterport3D for RGB and RGB-D modalities. this https URL
>
---
#### [replaced 122] How to Spin an Object: First, Get the Shape Right
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2412.10273](https://arxiv.org/pdf/2412.10273)**

> **作者:** Rishabh Kabra; Drew A. Hudson; Sjoerd van Steenkiste; Joao Carreira; Niloy J. Mitra
>
> **摘要:** Image-to-3D models increasingly rely on hierarchical generation to disentangle geometry and texture. However, the design choices underlying these two-stage models--particularly the optimal choice of intermediate geometric representations--remain largely understudied. To investigate this, we introduce unPIC (undo-a-Picture), a modular framework for empirical analysis of image-to-3D pipelines. By factorizing the generation process into a multiview-geometry prior followed by an appearance decoder, unPIC enables a rigorous comparison of intermediate geometry representations. Through this framework, we identify that a specific representation, Camera-Relative Object Coordinates (CROCS), significantly outperforms alternatives such as depth maps, pretrained visual features, and other pointmap-based representations. We demonstrate that CROCS is not only easier for the first-stage geometry prior to predict, but also serves as an effective conditioning signal for ensuring 360-degree consistency during appearance decoding. Another advantage is that CROCS enables fully feedforward, direct 3D point cloud generation without requiring a separate post-hoc reconstruction step. Our unPIC formulation utilizing CROCS achieves superior novel-view quality, geometric accuracy, and multiview consistency; it outperforms leading baselines, including InstantMesh, Direct3D, CAT3D, Free3D, and EscherNet, on datasets of real-world 3D captures like Google Scanned Objects and the Digital Twin Catalog.
>
---
#### [replaced 123] RemoteAgent: Bridging Vague Human Intents and Earth Observation with RL-based Agentic MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07765](https://arxiv.org/pdf/2604.07765)**

> **作者:** Liang Yao; Shengxiang Xu; Fan Liu; Chuanyi Zhang; Bishun Yao; Rui Min; Yongjun Li; Chaoqian Ouyang; Shimin Di; Min-Ling Zhang
>
> **摘要:** Earth Observation (EO) systems are essentially designed to support domain experts who often express their requirements through vague natural language rather than precise, machine-friendly instructions. Depending on the specific application scenario, these vague queries can demand vastly different levels of visual precision. Consequently, a practical EO AI system must bridge the gap between ambiguous human queries and the appropriate multi-granularity visual analysis tasks, ranging from holistic image interpretation to fine-grained pixel-wise predictions. While Multi-modal Large Language Models (MLLMs) demonstrate strong semantic understanding, their text-based output format is inherently ill-suited for dense, precision-critical spatial predictions. Existing agentic frameworks address this limitation by delegating tasks to external tools, but indiscriminate tool invocation is computationally inefficient and underutilizes the MLLM's native capabilities. To this end, we propose RemoteAgent, an agentic framework that strategically respects the intrinsic capability boundaries of MLLMs. To empower this framework to understand real user intents, we construct VagueEO, a human-centric instruction dataset pairing EO tasks with simulated vague natural-language queries. By leveraging VagueEO for reinforcement fine-tuning, we align an MLLM into a robust cognitive core that directly resolves image- and sparse region-level tasks. Consequently, RemoteAgent processes suitable tasks internally while intelligently orchestrating specialized tools via the Model Context Protocol exclusively for dense predictions. Extensive experiments demonstrate that RemoteAgent achieves robust intent recognition capabilities while delivering highly competitive performance across diverse EO tasks.
>
---
#### [replaced 124] MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19172](https://arxiv.org/pdf/2511.19172)**

> **作者:** Kehua Chen; Tianlu Mao; Xinzhu Ma; Hao Jiang; Zehao Li; Zihan Liu; Shuqin Gao; Honglong Zhao; Feng Dai; Yucheng Zhang; Zhaoqi Wang
>
> **备注:** Accepted by CVPR26; Project page: this https URL
>
> **摘要:** Recently, 3D Gaussian Splatting and its derivatives have achieved significant breakthroughs in large-scale scene reconstruction. However, how to efficiently and stably achieve high-quality geometric fidelity remains a core challenge. To address this issue, we introduce MetroGS, a novel Gaussian Splatting framework for efficient and robust reconstruction in complex urban environments. Our method is built upon a distributed 2D Gaussian Splatting representation as the core foundation, serving as a unified backbone for subsequent modules. To handle potential sparse regions in complex scenes, we propose a structured dense enhancement scheme that utilizes SfM priors and a pointmap model to achieve a denser initialization, while incorporating a sparsity compensation mechanism to improve reconstruction completeness. Furthermore, we design a progressive hybrid geometric optimization strategy that organically integrates monocular and multi-view optimization to achieve efficient and accurate geometric refinement. Finally, to address the appearance inconsistency commonly observed in large-scale scenes, we introduce a depth-guided appearance modeling approach that learns spatial features with 3D consistency, facilitating effective decoupling between geometry and appearance and further enhancing reconstruction stability. Experiments on large-scale urban datasets demonstrate that MetroGS achieves superior geometric accuracy, rendering quality, offering a unified solution for high-fidelity large-scale scene reconstruction.
>
---
#### [replaced 125] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21998](https://arxiv.org/pdf/2511.21998)**

> **作者:** Apratim Bhattacharyya; Bicheng Xu; Sanjay Haresh; Reza Pourreza; Litian Liu; Sunny Panchal; Pulkit Madan; Leonid Sigal; Roland Memisevic
>
> **备注:** Accepted to NeurIPS 2025 (Project page: this https URL)
>
> **摘要:** Multi-modal Large Language Models (LLM) have advanced conversational abilities but struggle with providing live, interactive step-by-step guidance, a key capability for future AI assistants. Effective guidance requires not only delivering instructions but also detecting their successful execution, as well as identifying and alerting users to mistakes, all of which has to happen in real-time. This requires models that are not turn-based, but that can react asynchronously to a video stream, as well as video data showing users performing tasks including mistakes and their corrections. To this end, we introduce Qualcomm Interactive Cooking, a new benchmark and dataset built upon CaptainCook4D, which contains user mistakes during task execution. Our dataset and benchmark features densely annotated, timed instructions and feedback messages, specifically including mistake alerts precisely timestamped to their visual occurrence in the video. We evaluate state-of-the-art multi-modal LLMs on the Qualcomm Interactive Cooking benchmark and introduce LiveMamba, a streaming multi-modal LLM designed for interactive instructional guidance. This work provides the first dedicated benchmark and a strong baseline for developing and evaluating on live, situated coaching.
>
---
#### [replaced 126] Towards Efficient Large Vision-Language Models: A Comprehensive Survey on Inference Strategies
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 本文综述了提升大视觉语言模型推理效率的策略，针对其计算量大、扩展性差的问题，从四个维度分析优化方法，旨在推动高效多模态系统的发展。**

- **链接: [https://arxiv.org/pdf/2603.27960](https://arxiv.org/pdf/2603.27960)**

> **作者:** Surendra Pathak; Bo Han
>
> **备注:** 12 pages
>
> **摘要:** Although Large Vision Language Models (LVLMs) have demonstrated impressive multimodal reasoning capabilities, their scalability and deployment are constrained by massive computational requirements. In particular, the massive amount of visual tokens from high-resolution input data aggravates the situation due to the quadratic complexity of attention mechanisms. To address these issues, the research community has developed several optimization frameworks. This paper presents a comprehensive survey of the current state-of-the-art techniques for accelerating LVLM inference. We introduce a systematic taxonomy that categorizes existing optimization frameworks into four primary dimensions: visual token compression, memory management and serving, efficient architectural design, and advanced decoding strategies. Furthermore, we critically examine the limitations of these current methodologies and identify critical open problems to inspire future research directions in efficient multimodal systems.
>
---
#### [replaced 127] PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决端到端模型依赖昂贵传感器和计算资源的问题。提出PRIX架构，仅使用摄像头数据，通过视觉特征提取和生成式规划实现高效路径预测。**

- **链接: [https://arxiv.org/pdf/2507.17596](https://arxiv.org/pdf/2507.17596)**

> **作者:** Maciej K. Wozniak; Lianhang Liu; Yixi Cai; Patric Jensfelt
>
> **备注:** Accepted for Robotics and Automation Letters (RA-L) and will be presented at iROS 2026
>
> **摘要:** While end-to-end autonomous driving models show promising results, their practical deployment is often hindered by large model sizes, a reliance on expensive LiDAR sensors and computationally intensive BEV feature representations. This limits their scalability, especially for mass-market vehicles equipped only with cameras. To address these challenges, we propose PRIX (Plan from Raw Pixels). Our novel and efficient end-to-end driving architecture operates using only camera data, without explicit BEV representation and forgoing the need for LiDAR. PRIX leverages a visual feature extractor coupled with a generative planning head to predict safe trajectories from raw pixel inputs directly. A core component of our architecture is the Context-aware Recalibration Transformer (CaRT), a novel module designed to effectively enhance multi-level visual features for more robust planning. We demonstrate through comprehensive experiments that PRIX achieves state-of-the-art performance on the NavSim and nuScenes benchmarks, matching the capabilities of larger, multimodal diffusion planners while being significantly more efficient in terms of inference speed and model size, making it a practical solution for real-world deployment. Our work is open-source and the code will be at this https URL.
>
---
#### [replaced 128] Interpretable Alzheimer's Diagnosis via Multimodal Fusion of Regional Brain Experts
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.10966](https://arxiv.org/pdf/2512.10966)**

> **作者:** Farica Zhuang; Shu Yang; Dinara Aliyeva; Zixuan Wen; Duy Duong-Tran; Christos Davatzikos; Tianlong Chen; Song Wang; Li Shen
>
> **备注:** Published at IEEE ICHI 2026
>
> **摘要:** Accurate and early diagnosis of Alzheimer's disease (AD) is critical for effective intervention and requires integrating complementary information from multimodal neuroimaging data. However, conventional fusion approaches often rely on simple concatenation of features, which cannot adaptively balance the contributions of biomarkers such as amyloid PET and MRI across brain regions. In this work, we propose MREF-AD, a Multimodal Regional Expert Fusion model for AD diagnosis. It is a Mixture-of-Experts (MoE) framework that models mesoscopic brain regions within each modality as independent experts and employs a gating network to learn subject-specific fusion weights. Utilizing tabular neuroimaging and demographic information from the Alzheimer's Disease Neuroimaging Initiative (ADNI), MREF-AD achieves competitive performance over strong classic and deep baselines while providing interpretable, modality- and region-level insight into how structural and molecular imaging jointly contribute to AD diagnosis.
>
---
#### [replaced 129] Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23606](https://arxiv.org/pdf/2505.23606)**

> **作者:** Qingyu Shi; Jinbin Bai; Zhuoran Zhao; Wenhao Chai; Kaidong Yu; Jianzong Wu; Shuangyong Song; Yunhai Tong; Xiangtai Li; Xuelong Li; Shuicheng Yan
>
> **备注:** Accepted to ICLR 2026. Codes and Supplementary Material: this https URL
>
> **摘要:** Unified generation models aim to handle diverse tasks across modalities -- such as text generation, image generation, and vision-language reasoning -- within a single architecture and decoding paradigm. Autoregressive unified models suffer from slow inference due to sequential decoding, and non-autoregressive unified models suffer from weak generalization due to limited pretrained backbones. We introduce the second-generation Meissonic: Muddit, a unified discrete diffusion transformer that enables fast and parallel generation across both text and image modalities. Unlike prior unified diffusion models trained from scratch, Muddit integrates strong visual priors from a pretrained text-to-image backbone with a lightweight text decoder, enabling flexible and high-quality multimodal generation under a unified architecture. Empirical results show that Muddit achieves competitive or superior performance compared to significantly larger autoregressive models in both quality and efficiency. The work highlights the potential of purely discrete diffusion, when equipped with strong visual priors, as a scalable and effective backbone for unified generation.
>
---
#### [replaced 130] ForestPrune: High-ratio Visual Token Compression for Video Multimodal Large Language Models via Spatial-Temporal Forest Modeling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.22911](https://arxiv.org/pdf/2603.22911)**

> **作者:** Shaobo Ju; Baiyang Song; Tao Chen; Jiapeng Zhang; Qiong Wu; Chao Chang; HuaiXi Wang; Yiyi Zhou; Rongrong Ji
>
> **摘要:** Due to the great saving of computation and memory overhead, token compression has become a research hot-spot for MLLMs and achieved remarkable progress in image-language tasks. However, for the video, existing methods still fall short of high-ratio token compression. We attribute this shortcoming to the insufficient modeling of temporal and continual video content, and propose a novel and training-free token pruning method for video MLLMs, termed ForestPrune, which achieves effective and high-ratio pruning via Spatial-temporal Forest Modeling. In practice, ForestPrune construct token forests across video frames based on the semantic, spatial and temporal constraints, making an overall comprehension of videos. Afterwards, ForestPrune evaluates the importance of token trees and nodes based on tree depth and node roles, thereby obtaining a globally optimal pruning decision. To validate ForestPrune, we apply it to two representative video MLLMs, namely LLaVA-Video and LLaVA-OneVision, and conduct extensive experiments on a bunch of video benchmarks. The experimental results not only show the great effectiveness for video MLLMs, e.g., retaining 95.8% average accuracy while reducing 90% tokens for LLaVA-OneVision, but also show its superior performance and efficiency than the compared token compression methods, e.g., +10.1% accuracy on MLVU and -81.4% pruning time than FrameFusion on LLaVA-Video.
>
---
#### [replaced 131] Unified Unsupervised and Sparsely-Supervised 3D Object Detection by Semantic Pseudo-Labeling and Prototype Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21484](https://arxiv.org/pdf/2602.21484)**

> **作者:** Yushen He; Lei Zhao; Weidong Chen
>
> **摘要:** 3D object detection is essential for autonomous driving and robotic perception, yet its reliance on large-scale manually annotated data limits scalability and adaptability. To reduce annotation dependency, unsupervised and sparsely-supervised paradigms have emerged. However, they face intertwined challenges: low-quality pseudo-labels, unstable feature mining, and a lack of a unified training framework. This paper proposes SPL, a unified training framework for both unsupervised and sparsely-supervised 3D object detection via \underline{S}emantic \underline{P}seudo-labeling and prototype \underline{L}earning. SPL first generates high-quality pseudo-labels by integrating image semantics, point cloud geometry, and temporal cues, producing both 3D bounding boxes for dense objects and 3D point labels for sparse ones. These pseudo-labels are not used directly but as probabilistic priors within a novel, multi-stage prototype learning strategy. This strategy stabilizes feature representation learning through memory-based initialization and momentum-based prototype updating, effectively mining features from both labeled and unlabeled data. Extensive experiments on KITTI and nuScenes datasets demonstrate that SPL significantly outperforms state-of-the-art methods in both settings. Our work provides a robust and generalizable solution for learning 3D object detectors with minimal or no manual annotations. Our code is available at this https URL.
>
---
#### [replaced 132] VERTIGO: Visual Preference Optimization for Cinematic Camera Trajectory Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.02467](https://arxiv.org/pdf/2604.02467)**

> **作者:** Mengtian Li; Yuwei Lu; Feifei Li; Chenqi Gan; Zhifeng Xie; Xi Wang
>
> **备注:** 28 pages, 10 figures, ECCV 2026
>
> **摘要:** Cinematic camera control relies on a tight feedback loop between director and cinematographer, where camera motion and framing are continuously reviewed and refined. Recent generative camera systems can produce diverse, text-conditioned trajectories, but they lack this "director in the loop" and have no explicit supervision of whether a shot is visually desirable. This results in in-distribution camera motion but poor framing, off-screen characters, and undesirable visual aesthetics. In this paper, we introduce VERTIGO, the first framework for visual preference optimization of camera trajectory generators. Our framework leverages a real-time graphics engine (Unity) to render 2D visual previews from generated camera motion. A cinematically fine-tuned vision-language model then scores these previews using our proposed cyclic semantic similarity mechanism, which aligns renders with text prompts. This process provides the visual preference signals for Direct Preference Optimization (DPO) post-training. Both quantitative evaluations and user studies on Unity renders and diffusion-based Camera-to-Video pipelines show consistent gains in condition adherence, framing quality, and perceptual realism. Notably, VERTIGO reduces the character off-screen rate from 38% to nearly 0% while preserving the geometric fidelity of camera motion. User study participants further prefer VERTIGO over baselines across composition, consistency, prompt adherence, and aesthetic quality, confirming the perceptual benefits of our visual preference post-training.
>
---
#### [replaced 133] Mirai: Autoregressive Visual Generation Needs Foresight
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.14671](https://arxiv.org/pdf/2601.14671)**

> **作者:** Yonghao Yu; Lang Huang; Zerun Wang; Runyi Li; Toshihiko Yamasaki
>
> **摘要:** Autoregressive (AR) visual generators model images as sequences of discrete tokens and are trained with a next-token likelihood objective. This strict causal supervision optimizes each step based only on the immediate next token, which can weaken global coherence and slow convergence. We investigate whether foresight, training signals that originate from later tokens, can improve autoregressive visual generation. We conduct a series of controlled diagnostics along the injection level, foresight layout, and foresight source axes, revealing a key insight: aligning foresight with AR models' internal representations on the 2D image grid improves causal modeling. We formulate this insight with Mirai (meaning "future" in Japanese), a general framework that injects future information into AR training with no architecture change and no extra inference overhead: Mirai-E uses explicit foresight from multiple future positions of unidirectional representations, whereas Mirai-I leverages implicit foresight from matched bidirectional representations. Extensive experiments show that Mirai significantly accelerates convergence and improves generation quality. For instance, Mirai can speed up LlamaGen-B's convergence by up to 10$\times$ and reduce the generation FID from 5.34 to 4.34 on the ImageNet class-condition image generation benchmark. Our study highlights that visual autoregressive models need foresight.
>
---
#### [replaced 134] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出ZARA，解决运动时间序列的零样本识别问题。通过知识增强的代理框架，实现无需训练的活动推理，提升跨数据集和场景的泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.04038](https://arxiv.org/pdf/2508.04038)**

> **作者:** Zechen Li; Baiyu Chen; Hao Xue; Flora D. Salim
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Motion sensor time-series are central to Human Activity Recognition (HAR), yet conventional approaches are constrained to fixed activity sets and typically require costly parameter retraining to adapt to new behaviors. While Large Language Models (LLMs) offer promising open-set reasoning capabilities, applying them directly to numerical time-series often leads to hallucinations and weak grounding. To address this challenge, we propose ZARA (Zero-training Activity Reasoning Agents), a knowledge- and retrieval-augmented agentic framework for motion time-series reasoning in a training-free inference setting. Rather than relying on black-box projections, ZARA distills reference data into a statistically grounded textual knowledge base that transforms implicit signal patterns into verifiable natural-language priors. Guided by retrieved evidence, ZARA iteratively selects discriminative cues and performs grounded reasoning over candidate activities. Extensive experiments on eight benchmarks show that ZARA generalizes robustly to unseen subjects and across datasets, demonstrating strong transferability across heterogeneous sensor domains. These results mark a step toward trustworthy, plug-and-play motion understanding beyond dataset-specific artifacts. Our code is available at this https URL.
>
---
#### [replaced 135] RoboStereo: Dual-Tower 4D Embodied World Models for Unified Policy Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12639](https://arxiv.org/pdf/2603.12639)**

> **作者:** Ruicheng Zhang; Guangyu Chen; Zunnan Xu; Zihao Liu; Zhizhou Zhong; Mingyang Zhang; Jun Zhou; Xiu Li
>
> **摘要:** Scalable Embodied AI faces fundamental constraints due to prohibitive costs and safety risks of real-world interaction. While Embodied World Models (EWMs) offer promise through imagined rollouts, existing approaches suffer from geometric hallucinations and lack unified optimization frameworks for practical policy improvement. We introduce RoboStereo, a symmetric dual-tower 4D world model that employs bidirectional cross-modal enhancement to ensure spatiotemporal geometric consistency and alleviate physics hallucinations. Building upon this high-fidelity 4D simulator, we present the first unified framework for world-model-based policy optimization: (1) Test-Time Policy Augmentation (TTPA) for pre-execution verification, (2) Imitative-Evolutionary Policy Learning (IEPL) leveraging visual perceptual rewards to learn from expert demonstrations, and (3) Open-Exploration Policy Learning (OEPL) enabling autonomous skill discovery and self-correction. Comprehensive experiments demonstrate RoboStereo achieves state-of-the-art generation quality, with our unified framework delivering >97% average relative improvement on fine-grained manipulation tasks.
>
---
#### [replaced 136] KSDiff: Keyframe-Augmented Speech-Aware Dual-Path Diffusion for Facial Animation
- **分类: cs.GR; cs.AI; cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2509.20128](https://arxiv.org/pdf/2509.20128)**

> **作者:** Tianle Lyu; Junchuan Zhao; Ye Wang
>
> **备注:** Paper accepted at ICASSP 2026, 5 pages, 3 figures, 3 tables
>
> **摘要:** Audio-driven facial animation has made significant progress in multimedia applications, with diffusion models showing strong potential for talking-face synthesis. However, most existing works treat speech features as a monolithic representation and fail to capture their fine-grained roles in driving different facial motions, while also overlooking the importance of modeling keyframes with intense dynamics. To address these limitations, we propose KSDiff, a Keyframe-Augmented Speech-Aware Dual-Path Diffusion framework. Specifically, the raw audio and transcript are processed by a Dual-Path Speech Encoder (DPSE) to disentangle expression-related and head-pose-related features, while an autoregressive Keyframe Establishment Learning (KEL) module predicts the most salient motion frames. These components are integrated into a Dual-path Motion generator to synthesize coherent and realistic facial motions. Extensive experiments on HDTF and VoxCeleb demonstrate that KSDiff achieves state-of-the-art performance, with improvements in both lip synchronization accuracy and head-pose naturalness. Our results highlight the effectiveness of combining speech disentanglement with keyframe-aware diffusion for talking-head generation. The demo page is available at: this https URL.
>
---
#### [replaced 137] Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.05510](https://arxiv.org/pdf/2604.05510)**

> **作者:** Yanming Xiu; Zhengyuan Jiang; Neil Zhenqiang Gong; Maria Gorlatova
>
> **备注:** CVPR 2026 Findings
>
> **摘要:** Augmented reality (AR) has rapidly expanded over the past decade. As AR becomes increasingly integrated into daily life, its security and reliability emerge as critical challenges. Among various threats, contradictory virtual content attacks, where malicious or inconsistent virtual elements are introduced into the user's view, pose a unique risk by misleading users, creating semantic confusion, or delivering harmful information. In this work, we systematically model such attacks and present ContrAR, a novel benchmark for evaluating the robustness of vision-language models (VLMs) against virtual content manipulation and contradiction in AR. ContrAR contains 312 real-world AR videos validated by 10 human participants. We further benchmark 11 VLMs, including both commercial and open-source models. Experimental results reveal that while current VLMs exhibit reasonable understanding of contradictory virtual content, room still remains for improvement in detecting and reasoning about adversarial content manipulations in AR environments. Moreover, balancing detection accuracy and latency remains challenging.
>
---
#### [replaced 138] Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型在视觉-语言冲突中的决策机制，探讨其是否因感知缺陷或决策错误导致错误回答。通过分析模型内部信号竞争，发现错误源于仲裁而非感知，提出干预方法提升视觉接地效果。**

- **链接: [https://arxiv.org/pdf/2604.09364](https://arxiv.org/pdf/2604.09364)**

> **作者:** Farhad Nooralahzadeh; Omid Rohanian; Yi Zhang; Jonathan Fürst; Kurt Stockinger
>
> **摘要:** When a Vision-Language Model (VLM) sees a blue banana and answers "yellow", is the problem of perception or arbitration? We explore the question in ten VLMs with various sizes and reveal an Encoding-Grounding Dissociation: models that fail to report what they see (and thus provide a wrong answer) still encode the visual evidence as strongly as models that provide the correct answer. Using Multimodal Arbitration Crossover (MAC) analysis with layer-by-layer Logit Lens probing, we track the competition between visual and prior signals across every layer of each model. We show that visual attributes can be linearly decodable from early layers (AUC > 0.86). The accuracy remains nearly identical for both successful and failed samples. However, the gap in the final-layer logit - not the strength of encoding - better predicts grounding outcomes with a correlation of $\rho=$ 0.847. After having studied when VLMs base their answers on image clues rather than prior knowledge, we want to understand the causal relationships. We establish causality through full-sequence activation patching. The standard last-token interventions in LLM interpretability do not affect VLMs. In contrast, replacing the full token sequence at layers identified by MAC alters 60 to 84% of outputs. Partial-token decomposition shows that image tokens carry almost all of the causal impact, while text tokens have none. Scaling addresses the remaining architectural differences to achieve perfect retention. Moving from diagnosis to intervention, we show that training-free activation steering - both linear and sparse autoencoder-guided - in early layers can improve visual grounding by up to +3.8% with degrading performance in some setups. Overall, these findings lead to a clear conclusion: VLMs already see well, but the challenge is acting on what they see. Targeted interventions can help to bridge this gap.
>
---
#### [replaced 139] GrOCE:Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12968](https://arxiv.org/pdf/2511.12968)**

> **作者:** Ning Han; Zhenyu Ge; Feng Han; Yuhua Sun; Chengqing Li; Jingjing Chen
>
> **备注:** Accepted to CVPR 2026 Highlight
>
> **摘要:** Concept erasure aims to remove harmful, inappropriate, or copyrighted content from text-to-image diffusion models while preserving non-target semantics. However, existing methods either rely on costly fine-tuning or apply coarse semantic separation, often degrading unrelated concepts and lacking adaptability to evolving concept sets. In this paper, we propose Graph-Guided Online Concept Erasure (GrOCE), a training-free framework that performs precise and context-aware online removal of target concepts. GrOCE constructs dynamic semantic graphs to identify clusters of target concepts and selectively suppress their influence within text prompts. It consists of three synergistic components: (1) dynamic semantic graph construction (Construct) incrementally builds a weighted graph over vocabulary concepts to capture semantic affinities; (2) adaptive cluster identification (Identify) extracts a target concept cluster through multi-hop traversal and diffusion-based scoring to quantify semantic influence; and (3) selective severing (Sever) removes semantic components associated with the target cluster from the text prompt while retaining non-target semantics and the global sentence structure. Extensive experiments demonstrate that GrOCE achieves state-of-the-art performance on the Concept Similarity (CS) and Fréchet Inception Distance (FID) metrics, offering efficient, accurate, and stable concept erasure.
>
---
#### [replaced 140] Learning World Models for Interactive Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.21996](https://arxiv.org/pdf/2505.21996)**

> **作者:** Taiye Chen; Xun Hu; Zihan Ding; Chi Jin
>
> **备注:** Project page: this https URL
>
> **摘要:** Foundational world models must be both interactive and preserve spatiotemporal coherence for effective future planning with action choices. However, present models for long video generation have limited inherent world modeling capabilities due to two main challenges: compounding errors and insufficient memory mechanisms. We enhance image-to-video models with interactive capabilities through additional action conditioning and autoregressive framework, and reveal that compounding error is inherently irreducible in autoregressive video generation, while insufficient memory mechanism leads to incoherence of world models. We propose video retrieval augmented generation (VRAG) with explicit global state conditioning, which significantly reduces long-term compounding errors and increases spatiotemporal consistency of world models. In contrast, naive autoregressive generation with extended context windows and retrieval-augmented generation prove less effective for video generation, primarily due to the limited in-context learning capabilities of current video models. Our work illuminates the fundamental challenges in video world models and establishes a comprehensive benchmark for improving video generation models with internal world modeling capabilities.
>
---
#### [replaced 141] PrefPaint: Enhancing Medical Image Inpainting through Expert Human Feedback
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21834](https://arxiv.org/pdf/2506.21834)**

> **作者:** Duy-Bao Bui; Hoang-Khang Nguyen; Thao Thi Phuong Dao; Kim Anh Phung; Tam V. Nguyen; Justin Zhan; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** Inpainting, the process of filling missing or corrupted image parts, has broad applications in medical imaging. However, generating anatomically accurate synthetic polyp images for clinical AI is a largely underexplored problem. In specialized fields like gastroenterology, inaccuracies in generated images can lead to false patterns and significant errors in downstream diagnosis. To ensure reliability, models require direct feedback from domain experts like oncologists. We propose PrefPaint, an interactive system that incorporates expert human feedback into Stable Diffusion Inpainting. By using D3PO instead of full RLHF, our approach bypasses the need for computationally expensive reward models, making it a highly practical choice for resource-constrained clinical settings. Furthermore, we introduce a streamlined web-based interface to facilitate this expert-in-the-loop training. Central to this platform is the Model Tree versioning interface, a novel HCI concept that visualizes the evolutionary progression of fine-tuned models. This interactive interface provides a smooth and intuitive user experience, making it easier to offer feedback and manage the fine-tuning process. User studies show that PrefPaint outperforms existing methods, reducing visual inconsistencies and generating highly realistic, anatomically accurate polyp images suitable for clinical AI applications.
>
---
#### [replaced 142] Contour Refinement using Discrete Diffusion in Low Data Regime
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05880](https://arxiv.org/pdf/2602.05880)**

> **作者:** Fei Yu Guan; Ian Keefe; Sophie Wilkinson; Daniel D.B. Perrakis; Steven Waslander
>
> **备注:** CRV 2026, 8 pages, 6 figures
>
> **摘要:** Boundary detection of irregular and translucent objects is an important problem with applications in medical imaging, environmental monitoring and manufacturing, where many of these applications are plagued with scarce labeled data and low in situ computational resources. While recent image segmentation studies focus on segmentation mask alignment with ground-truth, the task of boundary detection remains understudied, especially in the low data regime. In this work, we present a lightweight discrete diffusion contour refinement pipeline for robust boundary detection in the low data regime. We use a Convolutional Neural Network(CNN) architecture with self-attention layers as the core of our pipeline, and condition on a segmentation mask, iteratively denoising a sparse contour representation. We introduce multiple novel adaptations for improved low-data efficacy and inference efficiency, including using a simplified diffusion process, a customized model architecture, and minimal post processing to produce a dense, isolated contour given a dataset of size <500 training images. Our method outperforms several SOTA baselines on the medical imaging dataset KVASIR, is competitive on HAM10K and our custom wildfire dataset, Smoke, while improving inference framerate by 3.5X.
>
---
#### [replaced 143] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.17012](https://arxiv.org/pdf/2505.17012)**

> **作者:** Haoning Wu; Xiao Huang; Yaohui Chen; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **备注:** Accepted by CVPR 2026 (Highlight); Project Page: this https URL
>
> **摘要:** Existing evaluations of multimodal large language models (MLLMs) on spatial intelligence are typically fragmented and limited in scope. In this work, we aim to conduct a holistic assessment of the spatial understanding capabilities of modern MLLMs and propose complementary data-driven and agent-based solutions. Specifically, we make the following contributions: (i) we introduce SpatialScore, to our knowledge, the most comprehensive and diverse benchmark for multimodal spatial intelligence to date. It covers multiple visual data types, input modalities, and question-answering formats, and contains approximately 5K manually verified samples spanning 30 distinct tasks; (ii) using SpatialScore, we extensively evaluate 49 representative MLLMs, revealing persistent challenges and a substantial gap between current models and human-level spatial intelligence; (iii) to advance model capabilities, we construct SpatialCorpus, a large-scale training resource with 331K multimodal QA samples that supports fine-tuning on spatial reasoning tasks and significantly improves the performance of existing models (e.g., Qwen3-VL); (iv) to complement this data-driven route with a training-free paradigm, we develop SpatialAgent, a multi-agent system equipped with 12 specialized spatial perception tools that supports both Plan-Execute and ReAct reasoning, enabling substantial gains in spatial reasoning without additional model training. Extensive experiments and in-depth analyses demonstrate the effectiveness of our benchmark, corpus, and agent framework. We expect these resources to serve as a solid foundation for advancing MLLMs toward human-level spatial intelligence. All data, code, and models will be released to the research community.
>
---
#### [replaced 144] TCSA-UDA: Text-Driven Cross-Semantic Alignment for Unsupervised Domain Adaptation in Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05782](https://arxiv.org/pdf/2511.05782)**

> **作者:** Lalit Maurya; Honghai Liu; Reyer Zwiggelaar
>
> **摘要:** Unsupervised domain adaptation for medical image segmentation remains a significant challenge due to substantial domain shifts across imaging modalities, such as CT and MRI. While recent vision-language representation learning methods have shown promise, their potential in UDA segmentation tasks remains underexplored. To address this gap, we propose TCSA-UDA, a Text-driven Cross-Semantic Alignment framework that leverages domain-invariant textual class descriptions to guide visual representation learning. Our approach introduces a vision-language covariance cosine loss to directly align image encoder features with inter-class textual semantic relations, encouraging semantically meaningful and modality-invariant feature representations. Additionally, we incorporate a prototype alignment module that aligns class-wise pixel-level feature distributions across domains using high-level semantic prototypes. This mitigates residual category-level discrepancies and enhances cross-modal consistency. Extensive experiments on challenging cross-modality cardiac, abdominal, and brain tumor segmentation benchmarks demonstrate that our TCSA-UDA framework significantly reduces domain shift and consistently outperforms state-of-the-art UDA methods, establishing a new paradigm for integrating language-driven semantics into domain-adaptive medical image analysis.
>
---
#### [replaced 145] LOOPE: Learnable Optimal Patch Order in Positional Embeddings for Vision Transformers
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.14386](https://arxiv.org/pdf/2504.14386)**

> **作者:** Md Abtahi Majeed Chowdhury; Md Rifat Ur Rahman; Akil Ahmad Taki
>
> **摘要:** Positional embeddings (PE) play a crucial role in Vision Transformers (ViTs) by providing spatial information otherwise lost due to the permutation invariant nature of self attention. While absolute positional embeddings (APE) have shown theoretical advantages over relative positional embeddings (RPE), particularly due to the ability of sinusoidal functions to preserve spatial inductive biases like monotonicity and shift invariance, a fundamental challenge arises when mapping a 2D grid to a 1D sequence. Existing methods have mostly overlooked or never explored the impact of patch ordering in positional embeddings. To address this, we propose LOOPE, a learnable patch-ordering method that optimizes spatial representation for a given set of frequencies, providing a principled approach to patch order optimization. Empirical results show that our PE significantly improves classification accuracy across various ViT architectures. To rigorously evaluate the effectiveness of positional embeddings, we introduce the "Three Cell Experiment", a novel benchmarking framework that assesses the ability of PEs to retain relative and absolute positional information across different ViT architectures. Unlike standard evaluations, which typically report a performance gap of 4 to 6% between models with and without PE, our method reveals a striking 30 to 35% difference, offering a more sensitive diagnostic tool to measure the efficacy of PEs. Our experimental analysis confirms that the proposed LOOPE demonstrates enhanced effectiveness in retaining both relative and absolute positional information.
>
---
#### [replaced 146] Decompose, Mix, Adapt: A Unified Framework for Parameter-Efficient Neural Network Recombination and Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27383](https://arxiv.org/pdf/2603.27383)**

> **作者:** Nazia Tasnim; Shrimai Prabhumoye; Bryan A. Plummer
>
> **备注:** Accepted in CVPR, 2026 (Main Track)
>
> **摘要:** Parameter Recombination (PR) methods aim to efficiently compose the weights of a neural network for applications like Parameter-Efficient FineTuning (PEFT) and Model Compression (MC), among others. Most methods typically focus on one application of PR, which can make composing them challenging. For example, when deploying a large model you may wish to compress the model and also quickly adapt to new settings. However, PEFT methods often can still contain millions of parameters. This may be small compared to the original model size, but can be problematic in resource constrained deployments like edge devices, where they take a larger portion of the compressed model's parameters. To address this, we present Coefficient-gated weight Recombination by Interpolated Shared basis Projections (CRISP), a general approach that seamlessly integrates multiple PR tasks within the same framework. CRISP accomplishes this by factorizing pretrained weights into basis matrices and their component mixing projections. Sharing basis matrices across layers and adjusting its size enables us to perform MC, whereas the mixer weight's small size (fewer than 200 in some experiments) enables CRISP to support PEFT. Experiments show CRISP outperforms methods from prior work capable of dual-task applications by 4-5\% while also outperforming the state-of-the-art in PEFT by 1.5\% and PEFT+MC combinations by 1\%. Our code is available on the repository: this https URL.
>
---
#### [replaced 147] Neural Surface Reconstruction from Sparse Views Using Epipolar Geometry
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.04301](https://arxiv.org/pdf/2406.04301)**

> **作者:** Xinhai Chang; Kaichen Zhou
>
> **摘要:** Reconstructing accurate surfaces from sparse multi-view images remains challenging due to severe geometric ambiguity and occlusions. Existing generalizable neural surface reconstruction methods primarily rely on cost volumes that summarize multi-view features using simple statistics (e.g., mean and variance), which discard critical view-dependent geometric structure and often lead to over-smoothed reconstructions. We propose EpiS, a generalizable neural surface reconstruction framework that explicitly leverages epipolar geometry for sparse-view inputs. Instead of directly regressing geometry from cost-volume statistics, EpiS uses coarse cost-volume features to guide the aggregation of fine-grained epipolar features sampled along corresponding epipolar lines across source views. An epipolar transformer fuses multi-view information, followed by ray-wise aggregation to produce SDF-aware features for surface estimation. To further mitigate information loss under sparse views, we introduce a geometry regularization strategy that leverages a pretrained monocular depth model through scale-invariant global and local constraints. Extensive experiments on DTU and BlendedMVS demonstrate that EpiS significantly outperforms state-of-the-art generalizable surface reconstruction methods under sparse-view settings, while maintaining strong generalization without per-scene optimization.
>
---
#### [replaced 148] Uncertainty-Based Ensemble Learning in CMR Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.09269](https://arxiv.org/pdf/2502.09269)**

> **作者:** Yiwei Liu; Liang Zhong; Lingyi Wen; Yuankai Wu
>
> **摘要:** Existing methods derive clinical functional metrics from ventricular semantic segmentation in cardiac cine sequences. While performing well on overall segmentation, they struggle with the end slices. To address this, we extract global uncertainty from segmentation variance and use it in our ensemble learning method, Streaming, for classifier weighting, balancing overall and end-slice performance. We introduce the End Coefficient (EC) to quantify end-slice accuracy. Experiments on ACDC and M\&Ms datasets show that our framework achieves near state-of-the-art Dice Similarity Coefficient (DSC) and outperforms all models on end-slice performance, improving patient-specific segmentation accuracy. We open-sourced our code on this https URL.
>
---
#### [replaced 149] A Two-Stage Dual-Modality Model for Facial Emotional Expression Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12221](https://arxiv.org/pdf/2603.12221)**

> **作者:** Jiajun Sun; Zhe Gao
>
> **备注:** Camera-ready version. 14 pages, 5 figures in total: 8 pages main text with 4 figures, 3 pages references, and 3 pages appendix with 1 figure. Accepted at the 10th ABAW Workshop, CVPR 2026
>
> **摘要:** This paper addresses the expression (EXPR) recognition challenge in the 10th Affective Behavior Analysis in-the-Wild (ABAW) workshop and competition, which requires frame-level classification of eight facial emotional expressions from unconstrained videos. This task is challenging due to inaccurate face localization, large pose and scale variations, motion blur, temporal instability, and other confounding factors across adjacent frames. We propose a two-stage dual-modal (audio-visual) model to address these difficulties. Stage I focuses on robust visual feature extraction with a pretrained DINOv2-based encoder. Specifically, DINOv2 ViT-L/14 is used as the backbone, a padding-aware augmentation (PadAug) strategy is employed for image padding and data preprocessing from raw videos, and a mixture-of-experts (MoE) training head is introduced to enhance classifier diversity. Stage II addresses modality fusion and temporal consistency. For the visual modality, faces are re-cropped from raw videos at multiple scales, and the extracted visual features are averaged to form a robust frame-level representation. Concurrently, frame-aligned Wav2Vec 2.0 audio features are derived from short audio windows to provide complementary acoustic cues. These dual-modal features are integrated via a lightweight gated fusion module, followed by inference-time temporal smoothing. Experiments on the ABAW dataset demonstrate the effectiveness of the proposed method. The two-stage model achieves a Macro-F1 score of 0.5368 on the official validation set and 0.5122 +/- 0.0277 under 5-fold cross-validation, outperforming the official baselines.
>
---
#### [replaced 150] EdgeDAM: Real-time Object Tracking for Mobile Devices
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05463](https://arxiv.org/pdf/2603.05463)**

> **作者:** Syed Muhammad Raza; Syed Murtaza Hussain Abidi; Khawar Islam; Muhammad Ibrahim; Ajmal Saeed Mian
>
> **备注:** The paper is not accepted in any conference. We are revising our framework completely and update more authors for this work in the future
>
> **摘要:** Single-object tracking (SOT) on edge devices is a critical computer vision task, requiring accurate and continuous target localization across video frames under occlusion, distractor interference, and fast motion. However, recent state-of-the-art distractor-aware memory mechanisms are largely built on segmentation-based trackers and rely on mask prediction and attention-driven memory updates, which introduce substantial computational overhead and limit real-time deployment on resource-constrained hardware; meanwhile, lightweight trackers sustain high throughput but are prone to drift when visually similar distractors appear. To address these challenges, we propose EdgeDAM, a lightweight detection-guided tracking framework that reformulates distractor-aware memory for bounding-box tracking under strict edge constraints. EdgeDAM introduces two key strategies: (1) Dual-Buffer Distractor-Aware Memory (DAM), which integrates a Recent-Aware Memory to preserve temporally consistent target hypotheses and a Distractor-Resolving Memory to explicitly store hard negative candidates and penalize their re-selection during recovery; and (2) Confidence-Driven Switching with Held-Box Stabilization, where tracker reliability and temporal consistency criteria adaptively activate detection and memory-guided re-identification during occlusion, while a held-box mechanism temporarily freezes and expands the estimate to suppress distractor contamination. Extensive experiments on five benchmarks, including the distractor-focused DiDi dataset, demonstrate improved robustness under occlusion and fast motion while maintaining real-time performance on mobile devices, achieving 88.2% accuracy on DiDi and 25 FPS on an iPhone 15. Code will be released.
>
---
#### [replaced 151] Adversarial Video Promotion Against Text-to-Video Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06964](https://arxiv.org/pdf/2508.06964)**

> **作者:** Qiwei Tian; Chenhao Lin; Zhengyu Zhao; Qian Li; Shuai Liu; Chao Shen
>
> **备注:** This paper has been accepted by TIFS
>
> **摘要:** Thanks to the development of cross-modal models, text-to-video retrieval (T2VR) is advancing rapidly, but its robustness remains largely unexamined. Existing attacks against T2VR are designed to push videos away from queries, i.e., suppressing the ranks of videos, while the attacks that pull videos towards selected queries, i.e., promoting the ranks of videos, remain largely unexplored. These attacks can be more impactful as attackers may gain more views/clicks for financial benefits and widespread (mis)information. To this end, we pioneer the first attack against T2VR to promote videos adversarially, dubbed the Video Promotion attack (ViPro). We further propose Modal Refinement (MoRe) to capture the finer-grained, intricate interaction between visual and textual modalities to enhance black-box transferability. Comprehensive experiments cover 2 existing baselines, 3 leading T2VR models, 3 prevailing datasets with over 10k videos, evaluated under 3 scenarios. All experiments are conducted in a multi-target setting to reflect realistic scenarios where attackers seek to promote the video regarding multiple queries simultaneously. We also evaluated our attacks for defences and imperceptibility. Overall, ViPro surpasses other baselines by over $30/10/4\%$ for white/grey/black-box settings on average. Our work highlights an overlooked vulnerability, provides a qualitative analysis on the upper/lower bound of our attacks, and offers insights into potential counterplays. Code will be publicly available at this https URL.
>
---
#### [replaced 152] Benchmarking Large Vision-Language Models on Fine-Grained Image Tasks: A Comprehensive Evaluation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.14988](https://arxiv.org/pdf/2504.14988)**

> **作者:** Hong-Tao Yu; Yuxin Peng; Serge Belongie; Xiu-Shen Wei
>
> **摘要:** Recent advancements in Large Vision-Language Models (LVLMs) have demonstrated remarkable multimodal perception capabilities, garnering significant attention. While numerous evaluation studies have emerged, assessing LVLMs both holistically and on specialized tasks, fine-grained image tasks-fundamental to computer vision-remain largely unexplored. To fill this gap, we introduce a comprehensive fine-grained evaluation benchmark, i.e., FG-BMK, comprising 1.01 million questions and 0.33 million images. Our evaluation systematically examines LVLMs from both human-oriented and machine-oriented perspectives, focusing on their semantic recognition and fine-grained feature representation capabilities. Through extensive experiments on twelve representative LVLMs/VLMs, we uncover key findings regarding the influence of training paradigms, modality alignment, perturbation susceptibility, and fine-grained category reasoning on task performance. This work provides critical insights into the limitations of current LVLMs and offers guidance for future data construction and model design in the development of more advanced LVLMs. Our code is open-source and available at this https URL.
>
---
#### [replaced 153] Agri-R1: Agricultural Reasoning for Disease Diagnosis via Automated-Synthesis and Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于农业疾病诊断任务，解决传统方法依赖大量标注数据、可解释性差的问题。通过自动化合成数据和强化学习优化模型，提升诊断准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.04672](https://arxiv.org/pdf/2601.04672)**

> **作者:** Wentao Zhang; Mingkun Xu; Qi Zhang; Shangyang Li; Derek F. Wong; Lifei Wang; Yanchao Yang; Lina Lu; Tao Fang
>
> **备注:** This paper is submitted for review to the 2026 ACM MM Conference. The corresponding authors are Tao Fang and Lina Lu, where Tao Fang is the senior Corresponding Author (Last Author) and the principal supervisor of this work, having led the research design, guided the methodology, and overseen the entire project
>
> **摘要:** Agricultural disease diagnosis challenges VLMs, as conventional fine-tuning requires extensive labels, lacks interpretability, and generalizes poorly. While reasoning improves model robustness, existing methods rely on costly expert annotations and rarely address the open-ended, diverse nature of agricultural queries. To address these limitations, we propose \textbf{Agri-R1}, a reasoning-enhanced large model for agriculture. Our framework automates high-quality reasoning data generation via vision-language synthesis and LLM-based filtering, using only 19\% of available samples. Training employs Group Relative Policy Optimization (GRPO) with a novel reward function that integrates domain-specific lexicons and fuzzy matching to assess both correctness and linguistic flexibility in open-ended responses. Evaluated on CDDMBench, our resulting 3B-parameter model achieves performance competitive with 7B- to 13B-parameter baselines, showing a +27.9\% relative gain in disease recognition accuracy, +33.3\% in agricultural knowledge QA, and a +26.10-point improvement in cross-domain generalization over standard fine-tuning. These results suggest that automated reasoning synthesis paired with domain-aware reward design may provide a broadly applicable paradigm for RL-based VLM adaptation in data-scarce specialized domains. Our code and data are publicly available at: this https URL.
>
---
#### [replaced 154] DoSReMC: Domain Shift Resilient Mammography Classification using Batch Normalization Adaptation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.15452](https://arxiv.org/pdf/2508.15452)**

> **作者:** Uğurcan Akyüz; Deniz Katircioglu-Öztürk; Emre K. Süslü; Burhan Keleş; Mete C. Kaya; Gamze Durhan; Meltem G. Akpınar; Figen B. Demirkazık; Gözde B. Akar
>
> **摘要:** Numerous deep learning-based solutions have been developed for the automatic recognition of breast cancer using mammography images. However, their performance often declines when applied to data from different domains, primarily due to domain shift - the variation in data distributions between source and target domains. This performance drop limits the safe and equitable deployment of AI in real-world clinical settings. In this study, we present DoSReMC (Domain Shift Resilient Mammography Classification), a batch normalization (BN) adaptation framework designed to enhance cross-domain generalization without retraining the entire model. Using three large-scale full-field digital mammography (FFDM) datasets - including HCTP, a newly introduced, pathologically confirmed in-house dataset - we conduct a systematic cross-domain evaluation with convolutional neural networks (CNNs). Our results demonstrate that BN layers are a primary source of domain dependence: they perform effectively when training and testing occur within the same domain, and they significantly impair model generalization under domain shift. DoSReMC addresses this limitation by fine-tuning only the BN and fully connected (FC) layers, while preserving pretrained convolutional filters. We further integrate this targeted adaptation with an adversarial training scheme, yielding additional improvements in cross-domain generalizability while reducing the computational cost of model training. DoSReMC can be readily incorporated into existing AI pipelines and applied across diverse clinical environments, providing a practical pathway toward more robust and generalizable mammography classification systems.
>
---
#### [replaced 155] Text-to-Image Models and Their Representation of People from Different Nationalities Engaging in Activities
- **分类: cs.CV; cs.CY**

- **链接: [https://arxiv.org/pdf/2504.06313](https://arxiv.org/pdf/2504.06313)**

> **作者:** Abdulkareem Alsudais
>
> **摘要:** This paper investigates how popular text-to-image (T2I) models, DALL-E 3 and Gemini 3 Pro Preview, depict people from 206 nationalities when prompted to generate images of individuals engaging in common everyday activities. Five scenarios were developed, and 2,060 images were generated using input prompts that specified nationalities across five activities. When aggregating across activities and models, results showed that 28.4% of the images depicted individuals wearing traditional attire, including attire that is impractical for the specified activities in several cases. This pattern was statistically significantly associated with regions, with the Middle East & North Africa and Sub-Saharan Africa disproportionately affected, and was also associated with World Bank income groups. Similar region- and income-linked patterns were observed for images labeled as depicting impractical attire in two athletics-related activities. To assess image-text alignment, CLIP, ALIGN, and GPT-4.1 mini were used to score 9,270 image-prompt pairs. Images labeled as featuring traditional attire received statistically significantly higher alignment scores when prompts included country names, and this pattern weakened or reversed when country names were removed. Revised prompt analysis showed that one model frequently inserted the word "traditional" (50.3% for traditional-labeled images vs. 16.6% otherwise). These results indicate that these representational patterns can be shaped by several components of the pipeline, including image generator, evaluation models, and prompt revision.
>
---
#### [replaced 156] MASS: Motion-Aware Spatial-Temporal Grounding for Physics Reasoning and Comprehension in Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18373](https://arxiv.org/pdf/2511.18373)**

> **作者:** Xiyang Wu; Zongxia Li; Jihui Jin; Guangyao Shi; Gouthaman KV; Vishnu Raj; Nilotpal Sinha; Jingxi Chen; Fan Du; Dinesh Manocha
>
> **摘要:** Vision Language Models (VLMs) perform well on standard video tasks but struggle with physics-related reasoning involving motion dynamics and spatial interactions. We present a novel approach to address this gap by translating physical-world context cues into interpretable representations aligned with VLM perception, comprehension, and reasoning. We introduce MASS, a model-agnostic approach that injects spatiotemporal signals into the VLM language space via depth-based 3D encoding and visual grounding, coupled with a motion tracker for object dynamics. We also contribute a comprehensive benchmark, MASS-Bench, consisting of 4,350 real-world and AIGC videos and 8,361 free-form video question-answering pairs focused on physics-related comprehension tasks, with detailed annotations including visual detections and grounding over sub-segments, as well as full-sequence 3D motion tracking of entities. To strengthen cross-modal alignment and reasoning, we apply reinforcement fine-tuning to MASS. Experiments and ablations show that our refined VLMs outperform comparable baselines, larger models, and prior state-of-the-art models, achieving performance comparable to closed-source state-of-the-art VLMs, with only a 2\% gap to Gemini-2.5-Flash on physics reasoning and comprehension.
>
---
#### [replaced 157] M$^{2}$SNet: Multi-scale in Multi-scale Subtraction Network for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2303.10894](https://arxiv.org/pdf/2303.10894)**

> **作者:** Xiaoqi Zhao; Hongpeng Jia; Youwei Pang; Long Lv; Feng Tian; Lihe Zhang; Weibing Sun; Huchuan Lu
>
> **备注:** Machine Intelligence Research 2026
>
> **摘要:** Accurate medical image segmentation is critical for early medical diagnosis. Most existing methods are based on U-shape structure and use element-wise addition or concatenation to fuse different level features progressively in decoder. However, both the two operations easily generate plenty of redundant information, which will weaken the complementarity between different level features, resulting in inaccurate localization and blurred edges of lesions. To address this challenge, we propose a general multi-scale in multi-scale subtraction network (M$^{2}$SNet) to finish diverse segmentation from medical image. Specifically, we first design a basic subtraction unit (SU) to produce the difference features between adjacent levels in encoder. Next, we expand the single-scale SU to the intra-layer multi-scale SU, which can provide the decoder with both pixel-level and structure-level difference information. Then, we pyramidally equip the multi-scale SUs at different levels with varying receptive fields, thereby achieving the inter-layer multi-scale feature aggregation and obtaining rich multi-scale difference information. In addition, we build a training-free network ``LossNet'' to comprehensively supervise the task-aware features from bottom layer to top layer, which drives our multi-scale subtraction network to capture the detailed and structural cues simultaneously. Without bells and whistles, our method performs favorably against most state-of-the-art methods under different evaluation metrics on eleven datasets of four different medical image segmentation tasks of diverse image modalities, including color colonoscopy imaging, ultrasound imaging, computed tomography (CT), and optical coherence tomography (OCT). The source code can be available at this https URL.
>
---
#### [replaced 158] Quantization Robustness to Input Degradations for Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.19600](https://arxiv.org/pdf/2508.19600)**

> **作者:** Toghrul Karimov; Hassan Imani; Allan Kazakov
>
> **摘要:** Post-training quantization (PTQ) is crucial for deploying efficient object detection models, like YOLO, on resource-constrained devices. However, the impact of reduced precision on model robustness to real-world input degradations such as noise, blur, and compression artifacts is a significant concern. This paper presents a comprehensive empirical study evaluating the robustness of YOLO models (nano to extra-large scales) across multiple precision formats: FP32, FP16 (TensorRT), Dynamic UINT8 (ONNX), and Static INT8 (TensorRT). We introduce and evaluate a degradation-aware calibration strategy for Static INT8 PTQ, where the TensorRT calibration process is exposed to a mix of clean and synthetically degraded images. Models were benchmarked on the COCO dataset under seven distinct degradation conditions (including various types and levels of noise, blur, low contrast, and JPEG compression) and a mixed-degradation scenario. Results indicate that while Static INT8 TensorRT engines offer substantial speedups (~1.5-3.3x) with a moderate accuracy drop (~3-7% mAP50-95) on clean data, the proposed degradation-aware calibration did not yield consistent, broad improvements in robustness over standard clean-data calibration across most models and degradations. A notable exception was observed for larger model scales under specific noise conditions, suggesting model capacity may influence the efficacy of this calibration approach. These findings highlight the challenges in enhancing PTQ robustness and provide insights for deploying quantized detectors in uncontrolled environments. All code and evaluation tables are available at this https URL.
>
---
#### [replaced 159] RobustSpring: Benchmarking Robustness to Image Corruptions for Optical Flow, Scene Flow and Stereo
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.09368](https://arxiv.org/pdf/2505.09368)**

> **作者:** Victor Oei; Jenny Schmalfuss; Lukas Mehl; Madlen Bartsch; Shashank Agnihotri; Margret Keuper; Andreas Bulling; Andrés Bruhn
>
> **摘要:** Standard benchmarks for optical flow, scene flow, and stereo vision algorithms generally focus on model accuracy rather than robustness to image corruptions like noise or rain. Hence, the resilience of models to such real-world perturbations is largely unquantified. To address this, we present RobustSpring, a comprehensive dataset and benchmark for evaluating robustness to image corruptions for optical flow, scene flow, and stereo models. RobustSpring applies 20 different image corruptions, including noise, blur, color changes, quality degradations, and weather distortions, in a time-, stereo-, and depth-consistent manner to the high-resolution Spring dataset, creating a suite of 20,000 corrupted images that reflect challenging conditions. RobustSpring enables comparisons of model robustness via a new corruption robustness metric. Integration with the Spring benchmark enables two-axis evaluations of both accuracy and robustness. We benchmark a curated selection of initial models, observing that robustness varies widely by corruption type, and experimentally show that evaluations on RobustSpring indicate real-world robustness. RobustSpring is a new computer vision benchmark to treat robustness as a first-class citizen, fostering models that are accurate and resilient. It is available at this https URL.
>
---
#### [replaced 160] COXNet: Cross-Layer Fusion with Adaptive Alignment and Scale Integration for RGBT Tiny Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.09533](https://arxiv.org/pdf/2508.09533)**

> **作者:** Peiran Peng; Tingfa Xu; Liqiang Song; Mengqi Zhu; Yuqiang Fang; Jianan Li
>
> **摘要:** Detecting tiny objects in multimodal Red-Green-Blue-Thermal (RGBT) imagery is a critical challenge in computer vision, particularly in surveillance, search and rescue, and autonomous navigation. Drone-based scenarios exacerbate these challenges due to spatial misalignment, low-light conditions, occlusion, and cluttered backgrounds. Current methods struggle to leverage the complementary information between visible and thermal modalities effectively. We propose COXNet, a novel framework for RGBT tiny object detection, addressing these issues through three core innovations: i) the Cross-Layer Fusion Module, fusing high-level visible and low-level thermal features for enhanced semantic and spatial accuracy; ii) the Dynamic Alignment and Scale Refinement module, correcting cross-modal spatial misalignments and preserving multi-scale features; and iii) an optimized label assignment strategy using the GeoShape Similarity Measure for better localization. COXNet achieves a 3.32\% mAP$_{50}$ improvement on the RGBTDronePerson dataset over state-of-the-art methods, demonstrating its effectiveness for robust detection in complex environments.
>
---
#### [replaced 161] DecepGPT: Schema-Driven Deception Detection with Multicultural Datasets and Robust Multimodal Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.23916](https://arxiv.org/pdf/2603.23916)**

> **作者:** Jiajian Huang; Dongliang Zhu; Zitong YU; Hui Ma; Jiayu Zhang; Chunmei Zhu; Xiaochun Cao
>
> **备注:** 17 pages, 11 figures, 12 tables
>
> **摘要:** Multimodal deception detection aims to identify deceptive behavior by analyzing audiovisual cues for forensics and security. In these high-stakes settings, investigators need verifiable evidence connecting audiovisual cues to final decisions, along with reliable generalization across domains and cultural contexts. However, existing benchmarks provide only binary labels without intermediate reasoning cues. Datasets are also small with limited scenario coverage, leading to shortcut learning. We address these issues through three contributions. First, we construct reasoning datasets by augmenting existing benchmarks with structured cue-level descriptions and reasoning chains, enabling model output auditable reports. Second, we release T4-Deception, a multicultural dataset based on the unified ``To Tell The Truth'' television format implemented across four countries. With 1695 samples, it is the largest non-laboratory deception detection dataset. Third, we propose two modules for robust learning under small-data conditions. Stabilized Individuality-Commonality Synergy (SICS) refines multimodal representations by synergizing learnable global priors with sample-adaptive residuals, followed by a polarity-aware adjustment that bi-directionally recalibrates representations. Distilled Modality Consistency (DMC) aligns modality-specific predictions with the fused multimodal predictions via knowledge distillation to prevent unimodal shortcut learning. Experiments on three established benchmarks and our novel dataset demonstrate that our method achieves state-of-the-art performance in both in-domain and cross-domain scenarios, while exhibiting superior transferability across diverse cultural contexts. The datasets and codes will be released.
>
---
#### [replaced 162] CPAM: Context-Preserving Adaptive Manipulation for Zero-Shot Real Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.18438](https://arxiv.org/pdf/2506.18438)**

> **作者:** Dinh-Khoi Vo; Thanh-Toan Do; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** Editing natural images using textual descriptions in text-to-image diffusion models remains a significant challenge, particularly in achieving consistent generation and handling complex, non-rigid objects. Existing methods often struggle to preserve textures and identity, require extensive fine-tuning, and exhibit limitations in editing specific spatial regions or objects while retaining background details. This paper proposes Context-Preserving Adaptive Manipulation (CPAM), a novel zero-shot framework for complicated, non-rigid real image editing. Specifically, we propose a preservation adaptation module that adjusts self-attention mechanisms to preserve and independently control the object and background effectively. This ensures that the objects' shapes, textures, and identities are maintained while keeping the background undistorted during the editing process using the mask guidance technique. Additionally, we develop a localized extraction module to mitigate the interference with the non-desired modified regions during conditioning in cross-attention mechanisms. We also introduce various mask-guidance strategies to facilitate diverse image manipulation tasks in a simple manner. CPAM can be seamlessly integrated with multiple diffusion backbones, including SD1.5, SD2.1, and SDXL, demonstrating strong generalization across different model architectures. Extensive experiments on our newly constructed Image Manipulation BenchmArk (IMBA), a robust benchmark dataset specifically designed for real image editing, demonstrate that our proposed method is the preferred choice among human raters, outperforming existing state-of-the-art editing techniques. The source code and data will be publicly released at the project page: this https URL
>
---
#### [replaced 163] Grounded Forcing: Bridging Time-Independent Semantics and Proximal Dynamics in Autoregressive Video Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.06939](https://arxiv.org/pdf/2604.06939)**

> **作者:** Jintao Chen; Chengyu Bai; Junjun Hu; Xinda Xue; Mu Xu
>
> **摘要:** Autoregressive video synthesis offers a promising pathway for infinite-horizon generation but is fundamentally hindered by three intertwined challenges: semantic forgetting from context limitations, visual drift due to positional extrapolation, and controllability loss during interactive instruction switching. Current methods often tackle these issues in isolation, limiting long-term coherence. We introduce Grounded Forcing, a novel framework that bridges time-independent semantics and proximal dynamics through three interlocking mechanisms. First, to address semantic forgetting, we propose a Dual Memory KV Cache that decouples local temporal dynamics from global semantic anchors, ensuring long-term semantic coherence and identity stability. Second, to suppress visual drift, we design Dual-Reference RoPE Injection, which confines positional embeddings within the training manifold while rendering global semantics time-invariant. Third, to resolve controllability issues, we develop Asymmetric Proximity Recache, which facilitates smooth semantic inheritance during prompt transitions via proximity-weighted cache updates. These components operate synergistically to tether the generative process to stable semantic cores while accommodating flexible local dynamics. Extensive experiments demonstrate that Grounded Forcing significantly enhances long-range consistency and visual stability, establishing a robust foundation for interactive long-form video synthesis.
>
---
#### [replaced 164] FORGE: Fine-grained Multimodal Evaluation for Manufacturing Scenarios
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.07413](https://arxiv.org/pdf/2604.07413)**

> **作者:** Xiangru Jian; Hao Xu; Wei Pang; Xinjian Zhao; Chengyu Tao; Qixin Zhang; Xikun Zhang; Chao Zhang; Guanzhi Deng; Alex Xue; Juan Du; Tianshu Yu; Garth Tarr; Linqi Song; Qiuzhuang Sun; Dacheng Tao
>
> **备注:** Project Page:this https URL
>
> **摘要:** The manufacturing sector is increasingly adopting Multimodal Large Language Models (MLLMs) to transition from simple perception to autonomous execution, yet current evaluations fail to reflect the rigorous demands of real-world manufacturing environments. Progress is hindered by data scarcity and a lack of fine-grained domain semantics in existing datasets. To bridge this gap, we introduce FORGE. Wefirst construct a high-quality multimodal dataset that combines real-world 2D images and 3D point clouds, annotated with fine-grained domain semantics (e.g., exact model numbers). We then evaluate 18 state-of-the-art MLLMs across three manufacturing tasks, namely workpiece verification, structural surface inspection, and assembly verification, revealing significant performance gaps. Counter to conventional understanding, the bottleneck analysis shows that visual grounding is not the primary limiting factor. Instead, insufficient domain-specific knowledge is the key bottleneck, setting a clear direction for future research. Beyond evaluation, we show that our structured annotations can serve as an actionable training resource: supervised fine-tuning of a compact 3B-parameter model on our data yields up to 90.8% relative improvement in accuracy on held-out manufacturing scenarios, providing preliminary evidence for a practical pathway toward domain-adapted manufacturing MLLMs. The code and datasets are available at this https URL.
>
---
#### [replaced 165] AccidentSim: Generating Vehicle Collision Videos with Physically Realistic Collision Trajectories from Real-World Accident Reports
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.20654](https://arxiv.org/pdf/2503.20654)**

> **作者:** Xiangwen Zhang; Qian Zhang; Longfei Han; Qiang Qu; Xiaoming Chen; Weidong Cai
>
> **摘要:** Collecting real-world vehicle accident videos for autonomous driving research is challenging due to their rarity and complexity. While existing driving video generation methods may produce visually realistic videos, they often fail to deliver physically realistic simulations because they lack the capability to generate accurate post-collision trajectories. In this paper, we introduce AccidentSim, a novel framework that generates physically realistic vehicle collision videos by extracting and utilizing the physical clues and contextual information available in real-world vehicle accident reports. Specifically, AccidentSim leverages a reliable physical simulator to replicate post-collision vehicle trajectories from the physical and contextual information in the accident reports and to build a vehicle collision trajectory dataset. This dataset is then used to fine-tune a language model, enabling it to respond to user prompts and predict physically consistent post-collision trajectories across various driving scenarios based on user descriptions. Finally, we employ Neural Radiance Fields (NeRF) to render high-quality backgrounds, merging them with the foreground vehicles that exhibit physically realistic trajectories to generate vehicle collision videos. Experimental results demonstrate that the videos produced by AccidentSim excel in both visual and physical authenticity.
>
---
#### [replaced 166] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01390](https://arxiv.org/pdf/2512.01390)**

> **作者:** Seungho Choi; Jeahun Sung; Jihyong Oh
>
> **备注:** CVPR 2026 (camera ready ver.). Please visit our project page at this https URL
>
> **摘要:** Real-image super-resolution (Real-ISR) seeks to recover HR images from LR inputs with mixed, unknown degradations. While diffusion models surpass GANs in perceptual quality, they under-reconstruct high-frequency (HF) details due to a low-frequency (LF) bias and a depth-wise "low-first, high-later" hierarchy. We introduce FRAMER, a plug-and-play training scheme that exploits diffusion priors without changing the backbone or inference. At each denoising step, the final-layer feature map teaches all intermediate layers. Teacher and student feature maps are decomposed into LF/HF bands via FFT masks to align supervision with the model's internal frequency hierarchy. For LF, an Intra Contrastive Loss (IntraCL) stabilizes globally shared structure. For HF, an Inter Contrastive Loss (InterCL) sharpens instance-specific details using random-layer and in-batch negatives. Two adaptive modulators, Frequency-based Adaptive Weight (FAW) and Frequency-based Alignment Modulation (FAM), reweight per-layer LF/HF signals and gate distillation by current similarity. Across U-Net and DiT backbones (e.g., Stable Diffusion 2, 3), FRAMER consistently improves PSNR/SSIM and perceptual metrics (LPIPS, NIQE, MANIQA, MUSIQ). Ablations validate the final-layer teacher and random-layer negatives.
>
---
#### [replaced 167] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25866](https://arxiv.org/pdf/2509.25866)**

> **作者:** Chi Zhang; Haibo Qiu; Qiming Zhang; Zhixiong Zeng; Lin Ma; Jing Zhang
>
> **备注:** CVPR2026 FINDINGS
>
> **摘要:** The "thinking with images" paradigm represents a pivotal shift in the reasoning of Vision Language Models (VLMs), moving from text-dominant chain-of-thought to image-interactive reasoning. By invoking visual tools or generating intermediate visual representations, VLMs can iteratively attend to fine-grained regions, enabling deeper image understanding and more faithful multimodal reasoning. As an emerging paradigm, however, it still leaves substantial room for exploration in data construction accuracy, structural design, and broader application scenarios, which offer rich opportunities for advancing multimodal reasoning. To further advance this line of work, we present DeepSketcher, a comprehensive suite comprising both an image-text interleaved dataset and a self-contained model. The dataset contains 31k chain-of-thought (CoT) reasoning trajectories with diverse tool calls and resulting edited images, covering a wide range of data types and manipulation instructions with high annotation accuracy. Building on this resource, we design a model that performs interleaved image-text reasoning and natively generates "visual thoughts" by operating directly in the visual embedding space, rather than invoking external tools and repeatedly re-encoding generated images. This design enables tool-free and more flexible "thinking with images". Extensive experiments on multimodal reasoning benchmarks demonstrate strong performance, validating both the utility of the dataset and the effectiveness of the model design.
>
---
#### [replaced 168] AdvDINO: Domain-Adversarial Self-Supervised Representation Learning for Spatial Proteomics
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.04955](https://arxiv.org/pdf/2508.04955)**

> **作者:** Stella Su; Marc Harary; Scott J. Rodig; William Lotter
>
> **备注:** Proceedings of the Medical Imaging with Deep Learning (MIDL) Conference 2026
>
> **摘要:** Self-supervised learning (SSL) has emerged as a powerful approach for learning visual representations without manual annotations. However, the robustness of standard SSL methods to domain shift -- systematic differences across data sources -- remains uncertain, posing an especially critical challenge in biomedical imaging where batch effects can obscure true biological signals. We present AdvDINO, a domain-adversarial SSL framework that integrates a gradient reversal layer into the DINOv2 architecture to promote domain-invariant feature learning. Applied to a real-world cohort of six-channel multiplex immunofluorescence (mIF) whole slide images from lung cancer patients, AdvDINO mitigates slide-specific biases to learn more robust and biologically meaningful representations than non-adversarial baselines. Across more than 5.46 million mIF image tiles, the model uncovers phenotype clusters with differing proteomic profiles and prognostic significance, and enables strong survival prediction performance via attention-based multiple instance learning. The improved robustness also extends to a breast cancer cohort. While demonstrated on mIF data, AdvDINO is broadly applicable to other medical imaging domains, where domain shift is a common challenge.
>
---
#### [replaced 169] CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.16644](https://arxiv.org/pdf/2508.16644)**

> **作者:** Anindya Mondal; Ayan Banerjee; Sauradip Nag; Josep Llados; Xiatian Zhu; Anjan Dutta
>
> **摘要:** Diffusion models excel at photorealistic synthesis but struggle with precise object counts, especially in high-density settings. We introduce COUNTLOOP, a training-free framework that achieves precise instance control through iterative, structured feedback. Our method alternates between synthesis and evaluation: a VLM-based planner generates structured scene layouts, while a VLM-based critic provides explicit feedback on object counts, spatial arrangements, and visual quality to refine the layout iteratively. Instance-driven attention masking and cumulative attention composition further prevent semantic leakage, ensuring clear object separation even in densely occluded scenes. Evaluations on COCO-Count, T2I-CompBench, and two newly introduced high instance benchmarks show that COUNTLOOP reduces counting error by up to 57% and achieves the highest or comparable spatial quality scores across all benchmarks, while maintaining photorealism.
>
---
#### [replaced 170] PnP-CM: Consistency Models as Plug-and-Play Priors for Inverse Problems
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; physics.med-ph; stat.ML**

- **链接: [https://arxiv.org/pdf/2509.22736](https://arxiv.org/pdf/2509.22736)**

> **作者:** Merve Gülle; Junno Yun; Yaşar Utku Alçalar; Mehmet Akçakaya
>
> **备注:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026
>
> **摘要:** Diffusion models have found extensive use in solving inverse problems, by sampling from an approximate posterior distribution of data given the measurements. Recently, consistency models (CMs) have been proposed to directly predict the final output from any point on the diffusion ODE trajectory, enabling high-quality sampling in just a few neural function evaluations (NFEs). CMs have also been utilized for inverse problems, but existing CM-based solvers either require additional task-specific training or utilize data fidelity operations with slow convergence, limiting their applicability to large-scale problems and making them difficult to extend to nonlinear settings. In this work, we reinterpret CMs as proximal operators of a prior, enabling their integration into plug-and-play (PnP) frameworks. Specifically, we propose PnP-CM, an ADMM-based PnP solver that provides a unified framework for solving a wide range of inverse problems, and incorporates noise perturbations and momentum-based updates to improve performance in the low-NFE regime. We evaluate our approach on a diverse set of linear and nonlinear inverse problems. We also train and apply CMs to MRI data for the first time. Our results show that PnP-CM achieves high-quality reconstructions in as few as 4 NFEs, and produces meaningful results in 2 steps, highlighting its effectiveness in real-world inverse problems while outperforming existing CM-based approaches.
>
---
#### [replaced 171] Sat2Sound: A Unified Framework for Zero-Shot Soundscape Mapping
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出Sat2Sound，解决跨模态声音地图构建问题。通过融合多模态数据，提升声音分布预测的准确性和多样性，实现精准、可解释的声音景观映射。**

- **链接: [https://arxiv.org/pdf/2505.13777](https://arxiv.org/pdf/2505.13777)**

> **作者:** Subash Khanal; Srikumar Sastry; Aayush Dhakal; Adeel Ahmad; Abby Stylianou; Nathan Jacobs
>
> **备注:** Accepted to EarthVision 2026
>
> **摘要:** We present Sat2Sound, a unified multimodal framework for geospatial soundscape understanding, designed to predict and map the distribution of sounds across the Earth's surface. Existing methods for this task rely on paired satellite images and geotagged audio samples, which often fail to capture the full diversity of sound at a location. Sat2Sound overcomes this limitation by augmenting datasets with semantically rich, vision-language model-generated soundscape descriptions, which broaden the range of possible ambient sounds represented at each location. Our framework jointly learns from audio, text descriptions of audio, satellite images, and synthetic image captions through contrastive and codebook-aligned learning, discovering a set of "soundscape concepts" shared across modalities, enabling hyper-localized, explainable soundscape mapping. Sat2Sound achieves state-of-the-art performance in cross-modal retrieval between satellite image and audio on the GeoSound and SoundingEarth benchmarks. Finally, by retrieving detailed soundscape captions that can be rendered through text-to-audio models, Sat2Sound enables location-conditioned soundscape synthesis for immersive and educational applications, even with limited computational resources. Our code and models are available at this https URL.
>
---
#### [replaced 172] DRIFT: Deep Restoration, ISP Fusion, and Tone-mapping
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.03402](https://arxiv.org/pdf/2604.03402)**

> **作者:** Soumendu Majee; Joshua Peter Ebenezer; Abhinau K. Venkataramanan; Weidi Liu; Thilo Balke; Zeeshan Nadir; Sreenithy Chandran; Seok-Jun Lee; Hamid Rahim Sheikh
>
> **备注:** Proceedings of CVPR 2026
>
> **摘要:** Smartphone cameras have gained immense popularity with the adoption of high-resolution and high-dynamic range imaging. As a result, high-performance camera Image Signal Processors (ISPs) are crucial in generating high-quality images for the end user while keeping computational costs low. In this paper, we propose DRIFT (Deep Restoration, ISP Fusion, and Tone-mapping): an efficient AI mobile camera pipeline that generates high quality RGB images from hand-held raw captures. The first stage of DRIFT is a Multi-Frame Processing (MFP) network that is trained using a adversarial perceptual loss to perform multi-frame alignment, denoising, demosaicing, and super-resolution. Then, the output of DRIFT-MFP is processed by a novel deep-learning based tone-mapping (DRIFT-TM) solution that allows for tone tunability, ensures tone-consistency with a reference pipeline, and can be run efficiently for high-resolution images on a mobile device. We show qualitative and quantitative comparisons against state-of-the-art MFP and tone-mapping methods to demonstrate the effectiveness of our approach.
>
---
#### [replaced 173] Can Textual Reasoning Improve the Performance of MLLMs on Fine-grained Visual Classification?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.06993](https://arxiv.org/pdf/2601.06993)**

> **作者:** Jie Zhu; Yiyang Su; Xiaoming Liu
>
> **备注:** CVPR Finding, 2026
>
> **摘要:** Multi-modal large language models (MLLMs) exhibit strong general-purpose capabilities, yet still struggle on Fine-Grained Visual Classification (FGVC), a core perception task that requires subtle visual discrimination and is crucial for many real-world applications. A widely adopted strategy for boosting performance on challenging tasks such as math and coding is Chain-of-Thought (CoT) reasoning. However, several prior works have reported that CoT can actually harm performance on visual perception tasks. These studies, though, examine the issue from relatively narrow angles and leave open why CoT degrades perception-heavy performance. We systematically re-examine the role of CoT in FGVC through the lenses of zero-shot evaluation and multiple training paradigms. Across these settings, we uncover a central paradox: the degradation induced by CoT is largely driven by the reasoning length, in which longer textual reasoning consistently lowers classification accuracy. We term this phenomenon the ``Cost of Thinking''. Building on this finding, we make two key contributions: (1) MRN, a simple and general plug-and-play normalization method for multi-reward optimization that balances heterogeneous reward signals, and (2) ReFine-RFT, a framework that combines ensemble rewards with MRN to constrain reasoning length while providing dense accuracy-oriented feedback. Extensive experiments demonstrate the effectiveness of our findings and the proposed ReFine-RFT, achieving state-of-the-art performance across FGVC benchmarks. Project page: \href{this https URL}{ReFine-RFT}.
>
---
#### [replaced 174] What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.08101](https://arxiv.org/pdf/2407.08101)**

> **作者:** Sunny Panchal; Apratim Bhattacharyya; Guillaume Berger; Antoine Mercier; Cornelius Bohm; Florian Dietrichkeit; Reza Pourreza; Xuanlin Li; Pulkit Madan; Mingu Lee; Mark Todorovich; Ingo Bax; Roland Memisevic
>
> **备注:** Accepted to the 2024 NeurIPS Datasets and Benchmarks track; Data: this https URL Dataset quick start guide: this https URL and Stream-VLM code: this https URL
>
> **摘要:** Vision-language models have shown impressive progress in recent years. However, existing models are largely limited to turn-based interactions, where each turn must be stepped (i.e., prompted) by the user. Open-ended, asynchronous interactions, where an AI model may proactively deliver timely responses or feedback based on the unfolding situation in real-time, are an open challenge. In this work, we present the QEVD benchmark and dataset, which explores human-AI interaction in the challenging, yet controlled, real-world domain of fitness coaching -- a task which intrinsically requires monitoring live user activity and providing immediate feedback. The benchmark requires vision-language models to recognize complex human actions, identify possible mistakes, and provide appropriate feedback in real-time. Our experiments reveal the limitations of existing state-of-the-art vision-language models for such asynchronous situated interactions. Motivated by this, we propose a simple end-to-end streaming baseline that can respond asynchronously to human actions with appropriate feedback at the appropriate time.
>
---
#### [replaced 175] ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.07927](https://arxiv.org/pdf/2510.07927)**

> **作者:** Qunyi Zhang; Songan Zhang; Jiaqi Liu; Jinbao Wang; Xiaoning Lei; Guoyang Xie; Guannan Jiang; Zhichao Lu
>
> **备注:** accpted by IEEE Transactions on Artificial Intelligence
>
> **摘要:** Anomaly detection plays a pivotal role in manufacturing quality control, yet its application is constrained by limited abnormal samples and high manual annotation costs. While anomaly synthesis offers a promising solution, existing studies predominantly treat anomaly synthesis as an auxiliary component within anomaly detection frameworks, lacking systematic evaluation of anomaly synthesis algorithms. Current research also overlook crucial factors specific to anomaly synthesis, such as decoupling its impact from detection, quantitative analysis of synthetic data and adaptability across different scenarios. To address these limitations, we propose ASBench, the first comprehensive benchmarking framework dedicated to evaluating anomaly synthesis methods. Our framework introduces four critical evaluation dimensions: (i) the generalization performance across different datasets and pipelines (ii) the ratio of synthetic to real data (iii) the correlation between intrinsic metrics of synthesis images and anomaly detection performance metrics , and (iv) strategies for hybrid anomaly synthesis methods. Through extensive experiments, ASBench not only reveals limitations in current anomaly synthesis methods but also provides actionable insights for future research directions in anomaly synthesis
>
---
#### [replaced 176] Scone: Bridging Composition and Distinction in Subject-Driven Image Generation via Unified Understanding-Generation Modeling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.12675](https://arxiv.org/pdf/2512.12675)**

> **作者:** Yuran Wang; Bohan Zeng; Chengzhuo Tong; Wenxuan Liu; Yang Shi; Xiaochen Ma; Hao Liang; Yuanxing Zhang; Wentao Zhang
>
> **备注:** Code: this https URL
>
> **摘要:** Subject-driven image generation has advanced from single- to multi-subject composition, while neglecting distinction, the ability to distinguish and generate the correct subject when inputs contain multiple candidates. This limitation restricts effectiveness in complex, realistic visual settings. We propose Scone, a unified understanding-generation method that integrates composition and distinction. Scone enables the understanding expert to act as a semantic bridge, conveying semantic information and guiding the generation expert to preserve subject identity while minimizing interference. A two-stage training scheme first learns composition, then enhances distinction through semantic alignment and attention-based masking. We also introduce SconeEval, a benchmark for evaluating both composition and distinction across diverse scenarios. Experiments demonstrate that Scone outperforms existing open-source models in composition and distinction tasks on two benchmarks. Our model, benchmark, and training data are available at: this https URL.
>
---
#### [replaced 177] ProDiG: Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.02003](https://arxiv.org/pdf/2604.02003)**

> **作者:** Sirshapan Mitra; Yogesh S. Rawat
>
> **备注:** CVPR Findings 2026
>
> **摘要:** Generating ground-level views and coherent 3D site models from aerial-only imagery is challenging due to extreme viewpoint changes, missing intermediate observations, and large scale variations. Existing methods either refine renderings post-hoc, often producing geometrically inconsistent results, or rely on multi-altitude ground-truth, which is rarely available. Gaussian Splatting and diffusion-based refinements improve fidelity under small variations but fail under wide aerial-toground gaps. To address these limitations, we introduce ProDiG (Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction), a diffusionguided framework that progressively transforms aerial 3D representations toward ground-level fidelity. ProDiG synthesizes intermediate-altitude views and refines the Gaussian representation at each stage using a geometry-aware causal attention module that injects epipolar structure into reference-view diffusion. A distance-adaptive Gaussian module dynamically adjusts Gaussian scale and opacity based on camera distance, ensuring stable reconstruction across large viewpoint gaps. Together, these components enable progressive, geometrically grounded refinement without requiring additional ground-truth viewpoints. Extensive experiments on synthetic and real-world datasets demonstrate that ProDiG produces visually realistic ground-level renderings and coherent 3D geometry, significantly outperforming existing approaches in terms of visual quality, geometric consistency, and robustness to extreme viewpoint changes. Project Page: this https URL
>
---
#### [replaced 178] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.03416](https://arxiv.org/pdf/2601.03416)**

> **作者:** Xiangdong Hu; Yangyang Jiang; Qin Hu; Xiaojun Jia
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026), Main Conference
>
> **摘要:** Multimodal Large Language Models (MLLMs) have become widely deployed, yet their safety alignment remains fragile under adversarial inputs. Previous work has shown that increasing inference steps can disrupt safety mechanisms and lead MLLMs to generate attacker-desired harmful content. However, most existing attacks focus on increasing the complexity of the modified visual task itself and do not explicitly leverage the model's own reasoning incentives. This leads to them underperforming on reasoning models (Models with Chain-of-Thoughts) compared to non-reasoning ones (Models without Chain-of-Thoughts). If a model can think like a human, can we influence its cognitive-stage decisions so that it proactively completes a jailbreak? To validate this idea, we propose GAMBI} (Gamified Adversarial Multimodal Breakout via Instructional Traps), a novel multimodal jailbreak framework that decomposes and reassembles harmful visual semantics, then constructs a gamified scene that drives the model to explore, reconstruct intent, and answer as part of winning the game. The resulting structured reasoning chain increases task complexity in both vision and text, positioning the model as a participant whose goal pursuit reduces safety attention and induces it to answer the reconstructed malicious query. Extensive experiments on popular reasoning and non-reasoning MLLMs demonstrate that GAMBIT achieves high Attack Success Rates (ASR), reaching 92.13% on Gemini 2.5 Flash, 91.20% on QvQ-MAX, and 85.87% on GPT-4o, significantly outperforming baselines.
>
---
#### [replaced 179] SciPostLayoutTree: A Dataset for Structural Analysis of Scientific Posters
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18329](https://arxiv.org/pdf/2511.18329)**

> **作者:** Shohei Tanaka; Atsushi Hashimoto; Yoshitaka Ushiku
>
> **备注:** CVPR Findings 2026
>
> **摘要:** Scientific posters play a vital role in academic communication by presenting ideas through visual summaries. Analyzing reading order and parent-child relations of posters is essential for building structure-aware interfaces that facilitate clear and accurate understanding of research content. Despite their prevalence in academic communication, posters remain underexplored in structural analysis research, which has primarily focused on papers. To address this gap, we constructed SciPostLayoutTree, a dataset of approximately 8,000 posters annotated with reading order and parent-child relations. Compared to an existing structural analysis dataset, SciPostLayoutTree contains more instances of spatially challenging relations, including upward, horizontal, and long-distance relations. As a solution to these challenges, we develop Layout Tree Decoder, which incorporates visual features as well as bounding box features including position and category information. The model also uses beam search to predict relations while capturing sequence-level plausibility. Experimental results demonstrate that our model improves the prediction accuracy for spatially challenging relations and establishes a solid baseline for poster structure analysis. The dataset is publicly available at this https URL. The code is also publicly available at this https URL.
>
---
#### [replaced 180] MVOS_HSI: A Python Library for Preprocessing Agricultural Crop Hyperspectral Data
- **分类: cs.SE; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07656](https://arxiv.org/pdf/2604.07656)**

> **作者:** Rishik Aggarwal; Krisha Joshi; Pappu Kumar Yadav; Jianwei Qin; Thomas F. Burks; Moon S. Kim
>
> **备注:** 11 pages
>
> **摘要:** Hyperspectral imaging (HSI) allows researchers to study plant traits non-destructively. By capturing hundreds of narrow spectral bands per pixel, it reveals details about plant biochemistry and stress that standard cameras miss. However, processing this data is often challenging. Many labs still rely on loosely organized collections of lab-specific MATLAB or Python scripts, which makes workflows difficult to share and results difficult to reproduce. MVOS_HSI is an open-source Python library that provides an end-to-end workflow for processing leaf-level HSI data. The software handles everything from calibrating raw ENVI files to detecting and clipping individual leaves based on multiple vegetation indices (NDVI, CIRedEdge and GCI). It also includes tools for data augmentation to create training-time variations for machine learning and utilities to visualize spectral profiles. MVOS_HSI can be used as an importable Python library or run directly from the command line. The code and documentation are available on GitHub. By consolidating these common tasks into a single package, MVOS_HSI helps researchers produce consistent and reproducible results in plant phenotyping
>
---
#### [replaced 181] SCITUNE: Aligning Large Language Models with Human-Curated Scientific Multimodal Instructions
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于科学任务中的指令微调研究，旨在提升大语言模型对科学多模态指令的遵循能力。通过引入人类标注的科学多模态数据，改进模型在科学问答等任务上的表现。**

- **链接: [https://arxiv.org/pdf/2307.01139](https://arxiv.org/pdf/2307.01139)**

> **作者:** Sameera Horawalavithana; Sai Munikoti; Ian Stewart; Henry Kvinge; Karl Pazdernik
>
> **备注:** In Proceedings of the 1st Workshop on NLP for Science, Association for Computational Linguistics
>
> **摘要:** Instruction finetuning is a popular paradigm to align large language models (LLM) with human intent. Despite its popularity, this idea is less explored in improving LLMs to align existing foundation models with scientific disciplines, concepts and goals. In this work, we present \textit{SciTune} as a tuning framework to improve the ability of LLMs to follow multimodal instructions generated from scientific publications. To test our methodology, we train a large multimodal model LLaMA-SciTune that connects a vision encoder and LLM for science-focused visual and language understanding. LLaMA-SciTune significantly outperforms the state-of-the-art models in the generated figure types and captions in SciCap and VisText benchmarks. In comparison to the models that are finetuned with synthetic data only, LLaMA-SciTune surpasses human performance on average and in many sub-categories on the ScienceQA benchmark. Our results demonstrate that human-generated scientific multimodal instructions remain highly valuable in tuning LLMs to perform well on science tasks, despite their lower volume and relative scarcity compared to synthetic data. We publicly release the SciTune codebase this https URL.
>
---
