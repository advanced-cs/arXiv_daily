# 计算机视觉 cs.CV

- **最新发布 106 篇**

- **更新 65 篇**

## 最新发布

#### [new 001] SGAP-Gaze: Scene Grid Attention Based Point-of-Gaze Estimation Network for Driver Gaze
- **分类: cs.CV**

- **简介: 该论文属于驾驶员注视点估计任务，旨在提升驾驶场景中驾驶员注意力的准确识别。通过融合面部与场景信息，提出SGAP-Gaze模型，有效降低误差。**

- **链接: [https://arxiv.org/pdf/2604.19888](https://arxiv.org/pdf/2604.19888)**

> **作者:** Pavan Kumar Sharma; Pranamesh Chakraborty
>
> **摘要:** Driver gaze estimation is essential for understanding the driver's situational awareness of surrounding traffic. Existing gaze estimation models use driver facial information to predict the Point-of-Gaze (PoG) or the 3D gaze direction vector. We propose a benchmark dataset, Urban Driving-Face Scene Gaze (UD-FSG), comprising synchronized driver-face and traffic-scene images. The scene images provide cues about surrounding traffic, which can help improve the gaze estimation model, along with the face images. We propose SGAP-Gaze, Scene-Grid Attention based Point-of-Gaze estimation network, trained and tested on our UD-FSG dataset, which explicitly incorporates the scene images into the gaze estimation modelling. The gaze estimation network integrates driver face, eye, iris, and scene contextual information. First, the extracted features from facial modalities are fused to form a gaze intent vector. Then, attention scores are computed over the spatial scene grid using a Transformer-based attention mechanism fusing face and scene image features to obtain the PoG. The proposed SGAP-Gaze model achieves a mean pixel error of 104.73 on the UD-FSG dataset and 63.48 on LBW dataset, achieving a 23.5% reduction in mean pixel error compared to state-of-the-art driver gaze estimation models. The spatial pixel distribution analysis shows that SGAP-Gaze consistently achieves lower mean pixel error than existing methods across all spatial ranges, including the outer regions of the scene, which are rare but critical for understanding driver attention. These results highlight the effectiveness of integrating multi-modal gaze cues with scene-aware attention for a robust driver PoG estimation model in real-world driving environments.
>
---
#### [new 002] Video-ToC: Video Tree-of-Cue Reasoning
- **分类: cs.CV**

- **简介: 该论文提出Video-ToC框架，解决视频理解中推理能力不足和幻觉问题。通过视觉线索定位、推理奖励机制和自动标注数据集，提升视频分析效果。**

- **链接: [https://arxiv.org/pdf/2604.20473](https://arxiv.org/pdf/2604.20473)**

> **作者:** Qizhong Tan; Zhuotao Tian; Guangming Lu; Jun Yu; Wenjie Pei
>
> **摘要:** Existing Video Large Language Models (Video LLMs) struggle with complex video understanding, exhibiting limited reasoning capabilities and potential hallucinations. In particular, these methods tend to perform reasoning solely relying on the pretrained inherent reasoning rationales whilst lacking perception-aware adaptation to the input video content. To address this, we propose \textbf{Video-ToC}, a novel video reasoning framework that enhances video understanding through tree-of-cue reasoning. Specifically, our approach introduces three key innovations: (1) A tree-guided visual cue localization mechanism, which endows the model with enhanced fine-grained perceptual capabilities through structured reasoning patterns; (2) A reasoning-demand reward mechanism, which dynamically adjusts the reward value for reinforcement learning (RL) based on the estimation of reasoning demands, enabling on-demand incentives for more effective reasoning strategies; and (3) An automated annotation pipeline that constructs the Video-ToC-SFT-1k and Video-ToC-RL-2k datasets for supervised fine-tuning (SFT) and RL training, respectively. Extensive evaluations on six video understanding benchmarks and a video hallucination benchmark demonstrate the superiority of Video-ToC over baselines and recent methods. Code is available at this https URL.
>
---
#### [new 003] Semantic-Fast-SAM: Efficient Semantic Segmenter
- **分类: cs.CV**

- **简介: 该论文提出Semantic-Fast-SAM，解决实时语义分割问题。结合FastSAM与语义标注策略，提升效率并保持精度，适用于机器人等实时场景。**

- **链接: [https://arxiv.org/pdf/2604.20169](https://arxiv.org/pdf/2604.20169)**

> **作者:** Byunghyun Kim
>
> **备注:** APSIPA ASC 2025
>
> **摘要:** We propose Semantic-Fast-SAM (SFS), a semantic segmentation framework that combines the Fast Segment Anything model with a semantic labeling pipeline to achieve real-time performance without sacrificing accuracy. FastSAM is an efficient CNN-based re-implementation of the Segment Anything Model (SAM) that runs much faster than the original transformer-based SAM. Building upon FastSAM's rapid mask generation, we integrate a Semantic-Segment-Anything (SSA) labeling strategy to assign meaningful categories to each mask. The resulting SFS model produces high-quality semantic segmentation maps at a fraction of the computational cost and memory footprint of the original SAM-based approach. Experiments on Cityscapes and ADE20K benchmarks demonstrate that SFS matches the accuracy of prior SAM-based methods (mIoU ~ 70.33 on Cityscapes and 48.01 on ADE20K) while achieving approximately 20x faster inference than SSA in the closed-set setting. We also show that SFS effectively handles open-vocabulary segmentation by leveraging CLIP-based semantic heads, outperforming recent open-vocabulary models on broad class labeling. This work enables practical real-time semantic segmentation with the "segment-anything" capability, broadening the applicability of foundation segmentation models in robotics scenarios. The implementation is available at this https URL.
>
---
#### [new 004] Gaussians on a Diet: High-Quality Memory-Bounded 3D Gaussian Splatting Training
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决3DGS训练中内存占用过高的问题。通过动态优化高斯分布，实现低内存消耗的高质量训练。**

- **链接: [https://arxiv.org/pdf/2604.20046](https://arxiv.org/pdf/2604.20046)**

> **作者:** Yangming Zhang; Jian Xu; Kunxiong Zhu; Wei Niu; Miao Yin
>
> **摘要:** 3D Gaussian Splatting (3DGS) has revolutionized novel view synthesis with high-quality rendering through continuous aggregations of millions of 3D Gaussian primitives. However, it suffers from a substantial memory footprint, particularly during training due to uncontrolled densification, posing a critical bottleneck for deployment on memory-constrained edge devices. While existing methods prune redundant Gaussians post-training, they fail to address the peak memory spikes caused by the abrupt growth of Gaussians early in the training process. To solve the training memory consumption problem, we propose a systematic memory-bounded training framework that dynamically optimizes Gaussians through iterative growth and pruning. In other words, the proposed framework alternates between incremental pruning of low-impact Gaussians and strategic growing of new primitives with an adaptive Gaussian compensation, maintaining a near-constant low memory usage while progressively refining rendering fidelity. We comprehensively evaluate the proposed training framework on various real-world datasets under strict memory constraints, showing significant improvements over existing state-of-the-art methods. Particularly, our proposed method practically enables memory-efficient 3DGS training on NVIDIA Jetson AGX Xavier, achieving similar visual quality with up to 80% lower peak training memory consumption than the original 3DGS.
>
---
#### [new 005] RSRCC: A Remote Sensing Regional Change Comprehension Benchmark Constructed via Retrieval-Augmented Best-of-N Ranking
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RSRCC基准，解决遥感变化问答任务中的细粒度语义推理问题。通过构建包含12.6万条数据的基准，提升变化描述的准确性与解释性。**

- **链接: [https://arxiv.org/pdf/2604.20623](https://arxiv.org/pdf/2604.20623)**

> **作者:** Roie Kazoom; Yotam Gigi; George Leifman; Tomer Shekel; Genady Beryozkin
>
> **摘要:** Traditional change detection identifies where changes occur, but does not explain what changed in natural language. Existing remote sensing change captioning datasets typically describe overall image-level differences, leaving fine-grained localized semantic reasoning largely unexplored. To close this gap, we present RSRCC, a new benchmark for remote sensing change question-answering containing 126k questions, split into 87k training, 17.1k validation, and 22k test instances. Unlike prior datasets, RSRCC is built around localized, change-specific questions that require reasoning about a particular semantic change. To the best of our knowledge, this is the first remote sensing change question-answering benchmark designed explicitly for such fine-grained reasoning-based supervision. To construct RSRCC, we introduce a hierarchical semi-supervised curation pipeline that uses Best-of-N ranking as a critical final ambiguity-resolution stage. First, candidate change regions are extracted from semantic segmentation masks, then initially screened using an image-text embedding model, and finally validated through retrieval-augmented vision-language curation with Best-of-N ranking. This process enables scalable filtering of noisy and ambiguous candidates while preserving semantically meaningful changes. The dataset is available at this https URL.
>
---
#### [new 006] Evian: Towards Explainable Visual Instruction-tuning Data Auditing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型数据审计任务，旨在解决训练数据质量评估问题。通过构建基准数据集和提出分解评估框架EVIAN，提升数据质量筛选的精确性。**

- **链接: [https://arxiv.org/pdf/2604.20544](https://arxiv.org/pdf/2604.20544)**

> **作者:** Zimu Jia; Mingjie Xu; Andrew Estornell; Jiaheng Wei
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** The efficacy of Large Vision-Language Models (LVLMs) is critically dependent on the quality of their training data, requiring a precise balance between visual fidelity and instruction-following capability. Existing datasets, however, are plagued by inconsistent quality, and current data filtering methods rely on coarse-grained scores that lack the granularity to identify nuanced semantic flaws like logical fallacies or factual errors. This creates a fundamental bottleneck in developing more reliable models. To address this, we make three core contributions. First, we construct a large-scale, 300K-sample benchmark by systematically injecting diverse, subtle defects to provide a challenging testbed for data auditing. Second, we introduce a novel "Decomposition-then-Evaluation" paradigm that breaks model responses into constituent cognitive components: visual description, subjective inference, and factual claim, enabling targeted analysis. Third, we instantiate this paradigm via EVIAN (Explainable Visual Instruction-tuning Data AuditiNg), an automated framework that evaluates these components along the orthogonal axes of Image-Text Consistency, Logical Coherence, and Factual Accuracy. Our empirical findings challenge the prevailing scale-centric paradigm: a model fine-tuned on a compact, high-quality subset curated by EVIAN consistently surpassed models trained on orders-of-magnitude larger datasets. We also reveal that dividing complex auditing into verifiable subtasks enables robust curation, and that Logical Coherence is the most critical factor in data quality evaluation.
>
---
#### [new 007] SSL-R1: Self-Supervised Visual Reinforcement Post-Training for Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出SSL-R1，一种基于视觉自监督的强化学习框架，解决多模态大语言模型视觉理解不足的问题。通过图像生成可验证奖励，提升模型多模态推理能力。**

- **链接: [https://arxiv.org/pdf/2604.20705](https://arxiv.org/pdf/2604.20705)**

> **作者:** Jiahao Xie; Alessio Tonioni; Nathalie Rauschmayr; Federico Tombari; Bernt Schiele
>
> **摘要:** Reinforcement learning (RL) with verifiable rewards (RLVR) has demonstrated the great potential of enhancing the reasoning abilities in multimodal large language models (MLLMs). However, the reliance on language-centric priors and expensive manual annotations prevents MLLMs' intrinsic visual understanding and scalable reward designs. In this work, we introduce SSL-R1, a generic self-supervised RL framework that derives verifiable rewards directly from images. To this end, we revisit self-supervised learning (SSL) in visual domains and reformulate widely-used SSL tasks into a set of verifiable visual puzzles for RL post-training, requiring neither human nor external model supervision. Training MLLMs on these tasks substantially improves their performance on multimodal understanding and reasoning benchmarks, highlighting the potential of leveraging vision-centric self-supervised tasks for MLLM post-training. We think this work will provide useful experience in devising effective self-supervised verifiable rewards to enable RL at scale. Project page: this https URL.
>
---
#### [new 008] Investigation of cardinality classification for bacterial colony counting using explainable artificial intelligence
- **分类: cs.CV**

- **简介: 该论文属于细菌菌落计数任务，旨在解决MicrobiaNet模型在三及以上菌落分类上的困难。通过XAI分析，发现高视觉相似性是性能瓶颈，并建议改进模型或采用密度估计方法。**

- **链接: [https://arxiv.org/pdf/2604.20026](https://arxiv.org/pdf/2604.20026)**

> **作者:** Minghua Zheng; Na Helian; Peter C. R. Lane; Yi Sun; Allen Donald
>
> **备注:** 54 pages, 48 figures
>
> **摘要:** Automatic bacterial colony counting is a highly sought-after technology in modern biological laboratories because it eliminates manual counting effort. Previous work has observed that MicrobiaNet, currently the best-performing cardinality classification model for colony counting, has difficulty distinguishing colonies of three or more individuals. However, it is unclear if this is due to properties of the data together with inherent characteristics of the MicrobiaNet model. By analysing MicrobiaNet with explainable artificial intelligence (XAI), we demonstrate that XAI can provide insights into how data properties constrain cardinality classification performance in colony counting. Our results show that high visual similarity across classes is the key issue hindering further performance improvement, revising prior assertions about MicrobiaNet. These findings suggest future work should focus on models that explicitly incorporate visual similarity or explore density estimation approaches, with broader implications for neural network classifiers trained on imbalanced datasets.
>
---
#### [new 009] Learning to count small and clustered objects with application to bacterial colonies
- **分类: cs.CV**

- **简介: 该论文属于目标计数任务，解决细菌菌落小而密集的计数问题。提出ACFamNet和ACFamNet Pro模型，提升计数精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.20030](https://arxiv.org/pdf/2604.20030)**

> **作者:** Minghua Zheng; Na Helian; Peter C. R. Lane; Yi Sun; Allen Donald
>
> **备注:** 59 pages, 26 figures
>
> **摘要:** Automated bacterial colony counting from images is an important technique to obtain data required for the development of vaccines and antibiotics. However, bacterial colonies present unique machine vision challenges that affect counting, including (1) small physical size, (2) object clustering, (3) high data annotation cost, and (4) limited cross-species generalisation. While FamNet is an established object counting technique effective for clustered objects and costly data annotation, its effectiveness for small colony sizes and cross-species generalisation remains unknown. To address the first three challenges, we propose ACFamNet, an extension of FamNet that handles small and clustered objects using a novel region of interest pooling with alignment and optimised feature engineering. To address all four challenges above, we introduce ACFamNet Pro, which augments ACFamNet with multi-head attention and residual connections, enabling dynamic weighting of objects and improved gradient flow. Experiments show that ACFamNet Pro achieves a mean normalised absolute error (MNAE) of 9.64% under 5-fold cross-validation, outperforming ACFamNet and FamNet by 2.23% and 12.71%, respectively.
>
---
#### [new 010] ProMMSearchAgent: A Generalizable Multimodal Search Agent Trained with Process-Oriented Rewards
- **分类: cs.CV**

- **简介: 该论文提出ProMMSearchAgent，解决多模态搜索中的监督稀疏和环境不可控问题。通过过程导向奖励和模拟训练，提升搜索性能，实现零样本迁移。**

- **链接: [https://arxiv.org/pdf/2604.20486](https://arxiv.org/pdf/2604.20486)**

> **作者:** Wentao Yan; Shengqin Wang; Huichi Zhou; Yihang Chen; Kun Shao; Yuan Xie; Zhizhong Zhang
>
> **摘要:** Training multimodal agents via reinforcement learning for knowledge-intensive visual reasoning is fundamentally hindered by the extreme sparsity of outcome-based supervision and the unpredictability of live web environments. To resolve these algorithmic and environmental bottlenecks, we introduce ProMMSearchAgent, establishing a novel Sim-to-Real training paradigm for multimodal search. We decouple policy learning into a deterministic, local static sandbox. Crucially, to learn effectively within this constrained environment, we propose an introspective process-oriented reward. By probing the agent's own parametric knowledge boundaries, we generate dense behavioral metadata that explicitly rewards the correct cognitive decision, initiating a multimodal or text search only when visually or factually uncertain. Extensive experiments demonstrate that our locally-trained policy transfers zero-shot to the live Google Search API. ProMMSearchAgent achieves new SOTA performance, outperforming MMSearch-R1 by +5.1% on FVQA-test, +6.3% on InfoSeek, and +11.3% on MMSearch.
>
---
#### [new 011] Semi-Supervised Flow Matching for Mosaiced and Panchromatic Fusion Imaging
- **分类: cs.CV**

- **简介: 该论文属于图像融合任务，旨在解决低分辨率马赛克HSI与高分辨率PAN图像融合问题。提出半监督流匹配框架，提升融合效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.20128](https://arxiv.org/pdf/2604.20128)**

> **作者:** Peiming Luo; Nan Wang; Litong Liu; Jiahan Huang; Chenxu Wu; Renwei Dian; Junming Hou
>
> **摘要:** Fusing a low resolution (LR) mosaiced hyperspectral image (HSI) with a high resolution (HR) panchromatic (PAN) image offers a promising avenue for video-rate HR-HSI imaging via single-shot acquisition, yet its severely ill-posed nature remains a significant challenge. In this work, we propose a novel semi-supervised flow matching framework for mosaiced and PAN image fusion. Unlike previous diffusion-based approaches constrained by specific protocols or handcrafted assumptions, our method seamlessly integrates an unsupervised scheme with flow matching, resulting in a generalizable and efficient generative framework. Specifically, our method follows a two-stage training pipeline. First, we pretrain an unsupervised prior network to produce an initial pseudo HR-HSI. Building on this, we then train a conditional flow matching model to generate the target HR-HSI, introducing a random voting mechanism that iteratively refines the initial HR-HSI estimate, enabling robust and effective fusion. During inference, we employ a conflict-free gradient guidance strategy that ensures spectrally and spatially consistent HR-HSI reconstruction. Experiments on multiple benchmark datasets demonstrate that our method achieves superior quantitative and qualitative performance by a significant margin compared to representative baselines. Beyond mosaiced and PAN fusion, our approach provides a flexible generative framework that can be readily extended to other image fusion tasks and integrated with unsupervised or blind image restoration algorithms.
>
---
#### [new 012] Exploring Spatial Intelligence from a Generative Perspective
- **分类: cs.CV**

- **简介: 该论文属于多模态模型任务，旨在解决空间智能评估与提升问题。提出GSI-Bench基准，通过图像编辑评估生成模型的空间约束能力，并验证生成训练可增强空间推理。**

- **链接: [https://arxiv.org/pdf/2604.20570](https://arxiv.org/pdf/2604.20570)**

> **作者:** Muzhi Zhu; Shunyao Jiang; Huanyi Zheng; Zekai Luo; Hao Zhong; Anzhou Li; Kaijun Wang; Jintao Rong; Yang Liu; Hao Chen; Tao Lin; Chunhua Shen
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** Spatial intelligence is essential for multimodal large language models, yet current benchmarks largely assess it only from an understanding perspective. We ask whether modern generative or unified multimodal models also possess generative spatial intelligence (GSI), the ability to respect and manipulate 3D spatial constraints during image generation, and whether such capability can be measured or improved. We introduce GSI-Bench, the first benchmark designed to quantify GSI through spatially grounded image editing. It consists of two complementary components: GSI-Real, a high-quality real-world dataset built via a 3D-prior-guided generation and filtering pipeline, and GSI-Syn, a large-scale synthetic benchmark with controllable spatial operations and fully automated labeling. Together with a unified evaluation protocol, GSI-Bench enables scalable, model-agnostic assessment of spatial compliance and editing fidelity. Experiments show that fine-tuning unified multimodal models on GSI-Syn yields substantial gains on both synthetic and real tasks and, strikingly, also improves downstream spatial understanding. This provides the first clear evidence that generative training can tangibly strengthen spatial reasoning, establishing a new pathway for advancing spatial intelligence in multimodal models.
>
---
#### [new 013] WildFireVQA: A Large-Scale Radiometric Thermal VQA Benchmark for Aerial Wildfire Monitoring
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出WildFireVQA，一个用于航空火灾监测的多模态视觉问答基准，整合RGB与热辐射数据，解决火灾识别与分析问题。**

- **链接: [https://arxiv.org/pdf/2604.20190](https://arxiv.org/pdf/2604.20190)**

> **作者:** Mobin Habibpour; Niloufar Alipour Talemi; John Spodnik; Camren J. Khoury; Fatemeh Afghah
>
> **摘要:** Wildfire monitoring requires timely, actionable situational awareness from airborne platforms, yet existing aerial visual question answering (VQA) benchmarks do not evaluate wildfire-specific multimodal reasoning grounded in thermal measurements. We introduce WildFireVQA, a large-scale VQA benchmark for aerial wildfire monitoring that integrates RGB imagery with radiometric thermal data. WildFireVQA contains 6,097 RGB-thermal samples, where each sample includes an RGB image, a color-mapped thermal visualization, and a radiometric thermal TIFF, and is paired with 34 questions, yielding a total of 207,298 multiple-choice questions spanning presence and detection, classification, distribution and segmentation, localization and direction, cross-modal reasoning, and flight planning for operational wildfire intelligence. To improve annotation reliability, we combine multimodal large language model (MLLM)-based answer generation with sensor-driven deterministic labeling, manual verification, and intra-frame and inter-frame consistency checks. We further establish a comprehensive evaluation protocol for representative MLLMs under RGB, Thermal, and retrieval-augmented settings using radiometric thermal statistics. Experiments show that across task categories, RGB remains the strongest modality for current models, while retrieved thermal context yields gains for stronger MLLMs, highlighting both the value of temperature-grounded reasoning and the limitations of existing MLLMs in safety-critical wildfire scenarios. The dataset and benchmark code are open-source at this https URL.
>
---
#### [new 014] Stability-Driven Motion Generation for Object-Guided Human-Human Co-Manipulation
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于人机协同操作任务，旨在解决多人协作操作物体时的运动生成问题。通过引入流匹配框架，结合策略生成、对抗交互先验和稳定性驱动模拟，提升运动自然性和操作有效性。**

- **链接: [https://arxiv.org/pdf/2604.20336](https://arxiv.org/pdf/2604.20336)**

> **作者:** Jiahao Xu; Xiaohan Yuan; Xingchen Wu; Chongyang Xu; Kun Li; Buzhen Huang
>
> **备注:** CVPR 2026
>
> **摘要:** Co-manipulation requires multiple humans to synchronize their motions with a shared object while ensuring reasonable interactions, maintaining natural poses, and preserving stable states. However, most existing motion generation approaches are designed for single-character scenarios or fail to account for payload-induced dynamics. In this work, we propose a flow-matching framework that ensures the generated co-manipulation motions align with the intended goals while maintaining naturalness and effectiveness. Specifically, we first introduce a generative model that derives explicit manipulation strategies from the object's affordance and spatial configuration, which guide the motion flow toward successful manipulation. To improve motion quality, we then design an adversarial interaction prior that promotes natural individual poses and realistic inter-person interactions during co-manipulation. In addition, we also incorporate a stability-driven simulation into the flow matching process, which refines unstable interaction states through sampling-based optimization and directly adjusts the vector field regression to promote more effective manipulation. The experimental results demonstrate that our method achieves higher contact accuracy, lower penetration, and better distributional fidelity compared to state-of-the-art human-object interaction baselines. The code is available at this https URL.
>
---
#### [new 015] LEXIS: LatEnt ProXimal Interaction Signatures for 3D HOI from an Image
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D人体-物体交互重建任务，旨在解决从单张图像中准确恢复人体与物体的物理耦合问题。通过引入LEXIS-Flow框架，提升重建的精确性和真实性。**

- **链接: [https://arxiv.org/pdf/2604.20800](https://arxiv.org/pdf/2604.20800)**

> **作者:** Dimitrije Antić; Alvaro Budria; George Paschalidis; Sai Kumar Dwivedi; Dimitrios Tzionas
>
> **备注:** 26 pages, 11 figures, 4 tables. Project page: this https URL
>
> **摘要:** Reconstructing 3D Human-Object Interaction from an RGB image is essential for perceptive systems. Yet, this remains challenging as it requires capturing the subtle physical coupling between the body and objects. While current methods rely on sparse, binary contact cues, these fail to model the continuous proximity and dense spatial relationships that characterize natural interactions. We address this limitation via InterFields, a representation that encodes dense, continuous proximity across the entire body and object surfaces. However, inferring these fields from single images is inherently ill-posed. To tackle this, our intuition is that interaction patterns are characteristically structured by the action and object geometry. We capture this structure in LEXIS, a novel discrete manifold of interaction signatures learned via a VQ-VAE. We then develop LEXIS-Flow, a diffusion framework that leverages LEXIS signatures to estimate human and object meshes alongside their InterFields. Notably, these InterFields help in a guided refinement that ensures physically-plausible, proximity-aware reconstructions without requiring post-hoc optimization. Evaluation on Open3DHOI and BEHAVE shows that LEXIS-Flow significantly outperforms existing SotA baselines in reconstruction, contact, and proximity quality. Our approach not only improves generalization but also yields reconstructions perceived as more realistic, moving us closer to holistic 3D scene understanding. Code & models will be public at this https URL.
>
---
#### [new 016] ConeSep: Cone-based Robust Noise-Unlearning Compositional Network for Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，解决噪声三元组对应问题。提出ConeSep网络，通过几何量化、负边界学习和边界定向去噪提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.20358](https://arxiv.org/pdf/2604.20358)**

> **作者:** Zixu Li; Yupeng Hu; Zhiwei Chen; Mingyu Zhang; Zhiheng Fu; Liqiang Nie
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** The Composed Image Retrieval (CIR) task provides a flexible retrieval paradigm via a reference image and modification text, but it heavily relies on expensive and error-prone triplet annotations. This paper systematically investigates the Noisy Triplet Correspondence (NTC) problem introduced by annotations. We find that NTC noise, particularly ``hard noise'' (i.e., the reference and target images are highly similar but the modification text is incorrect), poses a unique challenge to existing Noise Correspondence Learning (NCL) methods because it breaks the traditional ``small loss hypothesis''. We identify and elucidate three key, yet overlooked, challenges in the NTC task, namely (C1) Modality Suppression, (C2) Negative Anchor Deficiency, and (C3) Unlearning Backlash. To address these challenges, we propose a Cone-based robuSt noisE-unlearning comPositional network (ConeSep). Specifically, we first propose Geometric Fidelity Quantization, theoretically establishing and practically estimating a noise boundary to precisely locate noisy correspondence. Next, we introduce Negative Boundary Learning, which learns a ``diagonal negative combination'' for each query as its explicit semantic opposite-anchor in the embedding space. Finally, we design Boundary-based Targeted Unlearning, which models the noisy correction process as an optimal transport problem, elegantly avoiding Unlearning Backlash. Extensive experiments on benchmark datasets (FashionIQ and CIRR) demonstrate that ConeSep significantly outperforms current state-of-the-art methods, which fully demonstrates the effectiveness and robustness of our method.
>
---
#### [new 017] UniCon3R: Contact-aware 3D Human-Scene Reconstruction from Monocular Video
- **分类: cs.CV**

- **简介: 该论文提出UniCon3R，解决单目视频中人体与场景的物理合理三维重建问题。通过建模人体与环境的接触关系，提升重建的物理真实性和运动估计精度。**

- **链接: [https://arxiv.org/pdf/2604.19923](https://arxiv.org/pdf/2604.19923)**

> **作者:** Tanuj Sur; Shashank Tripathi; Nikos Athanasiou; Ha Linh Nguyen; Kai Xu; Michael J. Black; Angela Yao
>
> **备注:** Project page: this https URL
>
> **摘要:** We introduce UniCon3R (Unified Contact-aware 3D Reconstruction), a unified feed-forward framework for online human-scene 4D reconstruction from monocular videos. Recent feed-forward methods enable real-time world-coordinate human motion and scene reconstruction, but they often produce physically implausible artifacts such as bodies floating above the ground or penetrating parts of the scene. The key reason is that existing approaches fail to model physical interactions between the human and the environment. A natural next step is to predict human-scene contact as an auxiliary output -- yet we find this alone is not sufficient: contact must actively correct the reconstruction. To address this, we explicitly model interaction by inferring 3D contact from the human pose and scene geometry and use the contact as a corrective cue for generating the final pose. This enables UniCon3R to jointly recover high-fidelity scene geometry and spatially aligned 3D humans within the scene. Experiments on standard human-centric video benchmarks such as RICH, EMDB, 3DPW and SLOPER4D show that UniCon3R outperforms state-of-the-art baselines on physical plausibility and global human motion estimation while achieving real-time online inference. We experimentally demonstrate that contact serves as a powerful internal prior rather than just an external metric, thus establishing a new paradigm for physically grounded joint human-scene reconstruction. Project page is available at this https URL .
>
---
#### [new 018] Infection-Reasoner: A Compact Vision-Language Model for Wound Infection Classification with Evidence-Grounded Clinical Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决慢性伤口感染分类及解释生成问题。通过构建一个紧凑的视觉-语言模型，结合推理蒸馏和强化学习，提升分类准确性和解释质量。**

- **链接: [https://arxiv.org/pdf/2604.19937](https://arxiv.org/pdf/2604.19937)**

> **作者:** Palawat Busaranuvong; Reza Saadati Fard; Emmanuel Agu; Deepak Kumar; Shefalika Gautam; Bengisu Tulu; Diane Strong
>
> **摘要:** Assessing chronic wound infection from photographs is challenging because visual appearance varies across wound etiologies, anatomical locations, and imaging conditions. Prior image-based deep learning methods have mainly focused on classification with limited interpretability, despite the need for evidence-grounded explanations to support point-of-care decision making. We present Infection-Reasoner, a compact 4B-parameter reasoning vision-language model for chronic wound infection classification and rationale generation. To address the scarcity of expert-labeled wound images with reasoning annotations, Infection-Reasoner is trained using a two-stage pipeline: (1) reasoning distillation, in which GPT-5.1 generates chain-of-thought rationales for unlabeled wound images to initialize wound-specific reasoning in a smaller student model (Qwen3-VL-4B-Thinking), and (2) reinforcement learning post-training with Group Relative Policy Optimization on a small labeled infection dataset to refine classification reasoning. On a held-out heterogeneous wound dataset, Infection-Reasoner achieved 86.8\% accuracy, 86.4\% sensitivity, and 87.1\% specificity, outperforming several strong baselines, including GPT-5.1. Rationale quality was further evaluated using both multimodal large language model (MLLM) judges and wound expert review. Across four MLLM judges, visual-support agreement scores ranged from 0.722 to 0.903, while expert review rated 61.8\% of rationales as Correct and 32.4\% as Partially Correct.
>
---
#### [new 019] MambaLiteUNet: Cross-Gated Adaptive Feature Fusion for Robust Skin Lesion Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于皮肤病变分割任务，旨在提升分割精度与效率。针对现有模型边界和纹理识别不足的问题，提出MambaLiteUNet框架，融合多种模块以增强特征交互和细节保留。**

- **链接: [https://arxiv.org/pdf/2604.20286](https://arxiv.org/pdf/2604.20286)**

> **作者:** Md Maklachur Rahman; Soon Ki Jung; Tracy Hammond
>
> **备注:** Accepted at CVPR 2026 Main
>
> **摘要:** Recent segmentation models have demonstrated promising efficiency by aggressively reducing parameter counts and computational complexity. However, these models often struggle to accurately delineate fine lesion boundaries and texture patterns essential for early skin cancer diagnosis and treatment planning. In this paper, we propose MambaLiteUNet, a compact yet robust segmentation framework that integrates Mamba state space modeling into a U-Net architecture, along with three key modules: Adaptive Multi-Branch Mamba Feature Fusion (AMF), Local-Global Feature Mixing (LGFM), and Cross-Gated Attention (CGA). These modules are designed to enhance local-global feature interaction, preserve spatial details, and improve the quality of skip connections. MambaLiteUNet achieves an average IoU of 87.12% and average Dice score of 93.09% across ISIC2017, ISIC2018, HAM10000, and PH2 benchmarks, outperforming state-of-the-art models. Compared to U-Net, our model improves average IoU and Dice by 7.72 and 4.61 points, respectively, while reducing parameters by 93.6% and GFLOPs by 97.6%. Additionally, in domain generalization with six unseen lesion categories, MambaLiteUNet achieves 77.61% IoU and 87.23% Dice, performing best among all evaluated models. Our extensive experiments demonstrate that MambaLiteUNet achieves a strong balance between accuracy and efficiency, making it a competitive and practical solution for dermatological image segmentation. Our code is publicly available at: this https URL.
>
---
#### [new 020] Exploring High-Order Self-Similarity for Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频动态建模问题。通过研究高阶时空自相似性，提出MOSS模块，提升运动建模能力，且计算成本低。**

- **链接: [https://arxiv.org/pdf/2604.20760](https://arxiv.org/pdf/2604.20760)**

> **作者:** Manjin Kim; Heeseung Kwon; Karteek Alahari; Minsu Cho
>
> **摘要:** Space-time self-similarity (STSS), which captures visual correspondences across frames, provides an effective way to represent temporal dynamics for video understanding. In this work, we explore higher-order STSS and demonstrate how STSSs at different orders reveal distinct aspects of these dynamics. We then introduce the Multi-Order Self-Similarity (MOSS) module, a lightweight neural module designed to learn and integrate multi-order STSS features. It can be applied to diverse video tasks to enhance motion modeling capabilities while consuming only marginal computational cost and memory usage. Extensive experiments on video action recognition, motion-centric video VQA, and real-world robotic tasks consistently demonstrate substantial improvements, validating the broad applicability of MOSS as a general temporal modeling module. The source code and checkpoints will be publicly available.
>
---
#### [new 021] PASTA: A Patch-Agnostic Twofold-Stealthy Backdoor Attack on Vision Transformers
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于后门攻击任务，旨在解决ViT模型在视觉和注意力域的隐蔽性问题。提出PASTA方法，实现高成功率且隐蔽的后门攻击。**

- **链接: [https://arxiv.org/pdf/2604.20047](https://arxiv.org/pdf/2604.20047)**

> **作者:** Dazhuang Liu; Yanqi Qiao; Rui Wang; Kaitai Liang; Georgios Smaragdakis
>
> **摘要:** Vision Transformers (ViTs) have achieved remarkable success across vision tasks, yet recent studies show they remain vulnerable to backdoor attacks. Existing patch-wise attacks typically assume a single fixed trigger location during inference to maximize trigger attention. However, they overlook the self-attention mechanism in ViTs, which captures long-range dependencies across patches. In this work, we observe that a patch-wise trigger can achieve high attack effectiveness when activating backdoors across neighboring patches, a phenomenon we term the Trigger Radiating Effect (TRE). We further find that inter-patch trigger insertion during training can synergistically enhance TRE compared to single-patch insertion. Prior ViT-specific attacks that maximize trigger attention often sacrifice visual and attention stealthiness, making them detectable. Based on these insights, we propose PASTA, a twofold stealthy patch-wise backdoor attack in both pixel and attention domains. PASTA enables backdoor activation when the trigger is placed at arbitrary patches during inference. To achieve this, we introduce a multi-location trigger insertion strategy to enhance TRE. However, preserving stealthiness while maintaining strong TRE is challenging, as TRE is weakened under stealthy constraints. We therefore formulate a bi-level optimization problem and propose an adaptive backdoor learning framework, where the model and trigger iteratively adapt to each other to avoid local optima. Extensive experiments show that PASTA achieves 99.13% attack success rate across arbitrary patches on average, while significantly improving visual and attention stealthiness (144.43x and 18.68x) and robustness (2.79x) against state-of-the-art ViT defenses across four datasets, outperforming CNN- and ViT-based baselines.
>
---
#### [new 022] A Computational Model of Message Sensation Value in Short Video Multimodal Features that Predicts Sensory and Behavioral Engagement
- **分类: cs.CV**

- **简介: 该论文属于短视频内容分析任务，旨在解决多模态特征对观众参与度影响的问题。通过构建MSV计算模型，分析感官与行为参与度的关系。**

- **链接: [https://arxiv.org/pdf/2604.19995](https://arxiv.org/pdf/2604.19995)**

> **作者:** Haoning Xue; Jingwen Zhang; Xiaohui Wang; Diane Dagyong Kim; Yunya Song
>
> **摘要:** The contemporary media landscape is characterized by sensational short videos. While prior research examines the effects of individual multimodal features, the collective impact of multimodal features on viewer engagement with short videos remains unknown. Grounded in the theoretical framework of Message Sensation Value (MSV), this study develops and tests a computational model of MSV with multimodal feature analysis and human evaluation of 1,200 short videos. This model that predicts sensory and behavioral engagement was further validated across two unseen datasets from three short video platforms (combined N = 14,492). While MSV is positively associated with sensory engagement, it shows an inverted U-shaped relationship with behavioral engagement: Higher MSV elicits stronger sensory stimulation, but moderate MSV optimizes behavioral engagement. This research advances the theoretical understanding of short video engagement and introduces a robust computational tool for short video research.
>
---
#### [new 023] Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，解决低比特部署下的高效单图像超分辨率问题。通过量化感知训练和教师指导优化，提升模型精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.20291](https://arxiv.org/pdf/2604.20291)**

> **作者:** Pham Phuong Nam Nguyen; Nam Tien Le; Thi Kim Trang Vo; Nhu Tinh Anh Nguyen
>
> **备注:** 10 pages, 4 figures. Accepted at the Mobile AI (MAI) 2026 Workshop at CVPR 2026
>
> **摘要:** Efficient single-image super-resolution (SISR) requires balancing reconstruction fidelity, model compactness, and robustness under low-bit deployment, which is especially challenging for x3 SR. We present a deployment-oriented quantized SISR framework based on an extract-refine-upsample design. The student performs most computation in the low-resolution space and uses a lightweight re-parameterizable backbone with PixelShuffle reconstruction, yielding a compact inference graph. To improve quality without significantly increasing complexity, we adopt a three-stage training pipeline: Stage 1 learns a basic reconstruction mapping with spatial supervision; Stage 2 refines fidelity using Charbonnier loss, DCT-domain supervision, and confidence-weighted output-level distillation from a Mamba-based teacher; and Stage 3 applies quantization-aware training directly on the fused deploy graph. We further use weight clipping and BatchNorm recalibration to improve quantization stability. On the MAI 2026 Quantized 4K Image Super-Resolution Challenge test set, our final AIO MAI submission achieves 29.79 dB PSNR and 0.8634 SSIM, obtaining a final score of 1.8 under the target mobile INT8 deployment setting. Ablation on Stage 3 optimization shows that teacher-guided supervision improves the dynamic INT8 TFLite reconstruction from 29.91 dB/0.853 to 30.0003 dB/0.856, while the fixed-shape deployable INT8 TFLite artifact attains 30.006 dB/0.857.
>
---
#### [new 024] LaplacianFormer:Rethinking Linear Attention with Laplacian Kernel
- **分类: cs.CV; cs.AI**

- **简介: 论文提出LaplacianFormer，解决Transformer在高分辨率视觉任务中的计算复杂度问题。通过引入拉普拉斯核替代softmax，提升注意力表达能力并优化计算效率。**

- **链接: [https://arxiv.org/pdf/2604.20368](https://arxiv.org/pdf/2604.20368)**

> **作者:** Zhe Feng; Sen Lian; Changwei Wang; Muyang Zhang; Tianlong Tan; Rongtao Xu; Weiliang Meng; Xiaopeng Zhang
>
> **摘要:** The quadratic complexity of softmax attention presents a major obstacle for scaling Transformers to high-resolution vision tasks. Existing linear attention variants often replace the softmax with Gaussian kernels to reduce complexity, but such approximations lack theoretical grounding and tend to oversuppress mid-range token interactions. We propose LaplacianFormer, a Transformer variant that employs a Laplacian kernel as a principled alternative to softmax, motivated by empirical observations and theoretical analysis. To address expressiveness degradation under low-rank approximations, we introduce a provably injective feature map that retains fine-grained token information. For efficient computation, we adopt a Nyström approximation of the kernel matrix and solve the resulting system using Newton--Schulz iteration, avoiding costly matrix inversion and SVD. We further develop custom CUDA implementations for both the kernel and solver, enabling high-throughput forward and backward passes suitable for edge deployment. Experiments on ImageNet show that LaplacianFormer achieves strong performance-efficiency trade-offs while improving attention expressiveness.
>
---
#### [new 025] Dual Causal Inference: Integrating Backdoor Adjustment and Instrumental Variable Learning for Medical VQA
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉问答任务，旨在解决模型过拟合和跨模态混淆问题。提出Dual Causal Inference框架，结合后门调整和工具变量学习，提升模型的可信诊断推理能力。**

- **链接: [https://arxiv.org/pdf/2604.20306](https://arxiv.org/pdf/2604.20306)**

> **作者:** Zibo Xu; Qiang Li; Ke Lu; Jin Wang; Weizhi Nie; Yuting Su
>
> **摘要:** Medical Visual Question Answering (MedVQA) aims to generate clinically reliable answers conditioned on complex medical images and questions. However, existing methods often overfit to superficial cross-modal correlations, neglecting the intrinsic biases embedded in multimodal medical data. Consequently, models become vulnerable to cross-modal confounding effects, severely hindering their ability to provide trustworthy diagnostic reasoning. To address this limitation, we propose a novel Dual Causal Inference (DCI) framework for MedVQA. To the best of our knowledge, DCI is the first unified architecture that integrates Backdoor Adjustment (BDA) and Instrumental Variable (IV) learning to jointly tackle both observable and unobserved confounders. Specifically, we formulate a Structural Causal Model (SCM) where observable cross-modal biases (e.g., frequent visual and textual co-occurrences) are mitigated via BDA, while unobserved confounders are compensated using an IV learned from a shared latent space. To guarantee the validity of the IV, we design mutual information constraints that maximize its dependence on the fused multimodal representations while minimizing its associations with the unobserved confounders and target answers. Through this dual mechanism, DCI extracts deconfounded representations that capture genuine causal relationships. Extensive experiments on four benchmark datasets, SLAKE, SLAKE-CP, VQA-RAD, and PathVQA, demonstrate that our method consistently outperforms existing approaches, particularly in out-of-distribution (OOD) generalization. Furthermore, qualitative analyses confirm that DCI significantly enhances the interpretability and robustness of cross-modal reasoning by explicitly disentangling true causal effects from spurious cross-modal shortcuts.
>
---
#### [new 026] From Scene to Object: Text-Guided Dual-Gaze Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言认知建模任务，解决现有数据缺乏细粒度对象级注意力标注的问题。通过构建新数据集和提出DualGaze-VLM模型，实现精准的对象级注意力预测。**

- **链接: [https://arxiv.org/pdf/2604.20191](https://arxiv.org/pdf/2604.20191)**

> **作者:** Zehong Ke; Yanbo Jiang; Jinhao Li; Zhiyuan Liu; Yiqian Tu; Qingwen Meng; Heye Huang; Jianqiang Wang
>
> **摘要:** Interpretable driver attention prediction is crucial for human-like autonomous driving. However, existing datasets provide only scene-level global gaze rather than fine-grained object-level annotations, inherently failing to support text-grounded cognitive modeling. Consequently, while Vision-Language Models (VLMs) hold great potential for semantic reasoning, this critical data limitations leads to severe text-vision decoupling and visual-bias hallucinations. To break this bottleneck and achieve precise object-level attention prediction, this paper proposes a novel dual-branch gaze prediction framework, establishing a complete paradigm from data construction to model architecture. First, we construct G-W3DA, a object-level driver attention dataset. By integrating a multimodal large language model with the Segment Anything Model 3 (SAM3), we decouple macroscopic heatmaps into object-level masks under rigorous cross-validation, fundamentally eliminating annotation hallucinations. Building upon this high-quality data foundation, we propose the DualGaze-VLM architecture. This architecture extracts the hidden states of semantic queries and dynamically modulates visual features via a Condition-Aware SE-Gate, achieving intent-driven precise spatial anchoring. Extensive experiments on the W3DA benchmark demonstrate that DualGaze-VLM consistently surpasses existing state-of-the-art (SOTA) models in spatial alignment metrics, notably achieving up to a 17.8% improvement in Similarity (SIM) under safety-critical scenarios. Furthermore, a visual Turing test reveals that the attention heatmaps generated by DualGaze-VLM are perceived as authentic by 88.22% of human evaluators, proving its capability to generate rational cognitive priors.
>
---
#### [new 027] MLG-Stereo: ViT Based Stereo Matching with Multi-Stage Local-Global Enhancement
- **分类: cs.CV**

- **简介: 该论文属于立体匹配任务，旨在解决ViT方法在细节预测和任意分辨率处理上的不足。提出MLG-Stereo，通过多粒度特征、局部-全局成本体积和引导递归单元提升性能。**

- **链接: [https://arxiv.org/pdf/2604.20393](https://arxiv.org/pdf/2604.20393)**

> **作者:** Haoyu Zhang; Jingyi Zhou; Peng Ye; Jiakang Yuan; Lin Zhang; Feng Xu; Tao Chen
>
> **摘要:** With the development of deep learning, ViT-based stereo matching methods have made significant progress due to their remarkable robustness and zero-shot ability. However, due to the limitations of ViTs in handling resolution sensitivity and their relative neglect of local information, the ability of ViT-based methods to predict details and handle arbitrary-resolution images is still weaker than that of CNN-based methods. To address these shortcomings, we propose MLG-Stereo, a systematic pipeline-level design that extends global modeling beyond the encoder stage. First, we propose a Multi-Granularity Feature Network to effectively balance global context and local geometric information, enabling comprehensive feature extraction from images of arbitrary resolution and bridging the gap between training and inference scales. Then, a Local-Global Cost Volume is constructed to capture both locally-correlated and global-aware matching information. Finally, a Local-Global Guided Recurrent Unit is introduced to iteratively optimize the disparity locally under the guidance of global information. Extensive experiments are conducted on multiple benchmark datasets, demonstrating that our MLG-Stereo exhibits highly competitive performance on the Middlebury and KITTI-2015 benchmarks compared to contemporaneous leading methods, and achieves outstanding results in the KITTI-2012 dataset.
>
---
#### [new 028] Weighted Knowledge Distillation for Semi-Supervised Segmentation of Maxillary Sinus in Panoramic X-ray Images
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决颌窦在全景X光图像中分割困难的问题。通过半监督框架和加权知识蒸馏方法，提升分割精度与边界准确性。**

- **链接: [https://arxiv.org/pdf/2604.20213](https://arxiv.org/pdf/2604.20213)**

> **作者:** Juha Park; Jiho Choi; Jong Pil Yun; Yong Chan Park; Han-Gyeol Yeom; Byung Do Lee; Sang Jun Lee
>
> **备注:** 14 pages, 6 figures. Under review
>
> **摘要:** Accurate segmentation of maxillary sinus in panoramic X-ray images is essential for dental diagnosis and surgical planning; however, this task remains relatively underexplored in dental imaging research. Structural overlap, ambiguous anatomical boundaries inherent to two-dimensional panoramic projections, and the limited availability of large scale clinical datasets with reliable pixel-level annotations make the development and evaluation of segmentation models challenging. To address these challenges, we propose a semi-supervised segmentation framework that effectively leverages both labeled and unlabeled panoramic radiographs, where knowledge distillation is utilized to train a student model with reliable structural information distilled from a teacher model. Specifically, we introduce a weighted knowledge distillation loss to suppress unreliable distillation signals caused by structural discrepancies between teacher and student predictions. To further enhance the quality of pseudo labels generated by the teacher network, we introduce SinusCycle-GAN which is a refinement network based on unpaired image-to-image translation. This refinement process improves the precision of boundaries and reduces noise propagation when learning from unlabeled data during semi-supervised training. To evaluate the proposed method, we collected clinical panoramic X-ray images from 2,511 patients, and experimental results demonstrate that the proposed method outperforms state-of-the-art segmentation models, achieving the Dice score of 96.35\% while reducing boundary error. The results indicate that the proposed semi-supervised framework provides robust and anatomically consistent segmentation performance under limited labeled data conditions, highlighting its potential for broader dental image analysis applications.
>
---
#### [new 029] FluSplat: Sparse-View 3D Editing without Test-Time Optimization
- **分类: cs.CV**

- **简介: 该论文属于3D场景编辑任务，解决跨视角一致性问题。提出一种无需测试时优化的框架，通过训练阶段的跨视角正则化实现高效、一致的3D编辑。**

- **链接: [https://arxiv.org/pdf/2604.20038](https://arxiv.org/pdf/2604.20038)**

> **作者:** Haitao Huang; Shin-Fang Chng; Huangying Zhan; Qingan Yan; Yi Xu
>
> **摘要:** Recent advances in text-guided image editing and 3D Gaussian Splatting (3DGS) have enabled high-quality 3D scene manipulation. However, existing pipelines rely on iterative edit-and-fit optimization at test time, alternating between 2D diffusion editing and 3D reconstruction. This process is computationally expensive, scene-specific, and prone to cross-view inconsistencies. We propose a feed-forward framework for cross-view consistent 3D scene editing from sparse views. Instead of enforcing consistency through iterative 3D refinement, we introduce a cross-view regularization scheme in the image domain during training. By jointly supervising multi-view edits with geometric alignment constraints, our model produces view-consistent results without per-scene optimization at inference. The edited views are then lifted into 3D via a feedforward 3DGS model, yielding a coherent 3DGS representation in a single forward pass. Experiments demonstrate competitive editing fidelity and substantially improved cross-view consistency compared to optimization-based methods, while reducing inference time by orders of magnitude.
>
---
#### [new 030] EmbodiedMidtrain: Bridging the Gap between Vision-Language Models and Vision-Language-Action Models via Mid-training
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EmbodiedMidtrain，解决VLM到VLA迁移性能不足的问题，通过中段训练提升模型在机器人操作任务中的表现。**

- **链接: [https://arxiv.org/pdf/2604.20012](https://arxiv.org/pdf/2604.20012)**

> **作者:** Yiyang Du; Zhanqiu Guo; Xin Ye; Liu Ren; Chenyan Xiong
>
> **摘要:** Vision-Language-Action Models (VLAs) inherit their visual and linguistic capabilities from Vision-Language Models (VLMs), yet most VLAs are built from off-the-shelf VLMs that are not adapted to the embodied domain, limiting their downstream performance. In this work, we propose EmbodiedMidtrain to bridge the gap between VLMs and VLAs. We first characterize the data distribution gap between them, showing that VLA data occupy compact regions that are largely separated from the broader VLM distribution, while the degree of alignment varies substantially both across and within VLM data sources. Then, we build a mid-training data engine that leverages a lightweight learnable proximity estimator to select the most VLA-aligned candidates from a large VLM pool, and mid-trains the VLM on this curated mixture before downstream VLA fine-tuning. Experiments on three robot manipulation benchmarks show that mid-training consistently improves performance across different VLM backbones, achieving results competitive with expert VLAs and off-the-shelf VLMs trained with larger model scale and training budgets. Further analysis reveals that mid-training provides a stronger initialization for VLA fine-tuning, with gains emerging from the earliest steps and widening throughout training. Moreover, the data engine captures both dataset-level and sample-level alignment signals, favoring spatial reasoning over text-centric tasks while preserving the diversity of the VLM data. We will release all code, data and models for future research.
>
---
#### [new 031] Cognitive Alignment At No Cost: Inducing Human Attention Biases For Interpretable Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型可解释性研究，旨在提升Vision Transformer的可解释性。通过调整自注意力权重，使模型更贴近人类注意力特征，同时不损害分类性能。**

- **链接: [https://arxiv.org/pdf/2604.20027](https://arxiv.org/pdf/2604.20027)**

> **作者:** Ethan Knights
>
> **摘要:** For state-of-the-art image understanding, Vision Transformers (ViTs) have become the standard architecture but their processing diverges substantially from human attentional characteristics. We investigate whether this cognitive gap can be shrunk by fine-tuning the self-attention weights of Google's ViT-B/16 on human saliency fixation maps. To isolate the effects of semantically relevant signals from generic human supervision, the tuned model is compared against a shuffled control. Fine-tuning significantly improved alignment across five saliency metrics and induced three hallmark human-like biases: tuning reversed the baseline's anti-human large-object bias toward small-objects, amplified the animacy preference and diminished extreme attention entropy. Bayesian parity analysis provides decisive to very-strong evidence that this cognitive alignment comes at no cost to the model's original classification performance on in- (ImageNet), corrupted (ImageNet-C) and out-of-distribution (ObjectNet) benchmarks. An equivalent procedure applied to a ResNet-50 Convolutional Neural Network (CNN) instead degraded both alignment and accuracy, suggesting that the ViT's modular self-attention mechanism is uniquely suited for dissociating spatial priority from representational logic. These findings demonstrate that biologically grounded priors can be instilled as a free emergent property of human-aligned attention, to improve transformer interpretability.
>
---
#### [new 032] Fast-then-Fine: A Two-Stage Framework with Multi-Granular Representation for Cross-Modal Retrieval in Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于跨模态检索任务，旨在解决遥感图像与文本匹配中的精度与效率问题。提出两阶段框架FTF，先快速召回后精细排序，提升检索效率与对齐精度。**

- **链接: [https://arxiv.org/pdf/2604.20429](https://arxiv.org/pdf/2604.20429)**

> **作者:** Xi Chen; Xu Chen; Xiangyang Jia; Xu Zhang; Shuquan Wei; Wei Wang
>
> **摘要:** Remote sensing (RS) image-text retrieval plays a critical role in understanding massive RS imagery. However, the dense multi-object distribution and complex backgrounds in RS imagery make it difficult to simultaneously achieve fine-grained cross-modal alignment and efficient retrieval. Existing methods either rely on complex cross-modal interactions that lead to low retrieval efficiency, or depend on large-scale vision-language model pre-training, which requires massive data and computational resources. To address these issues, we propose a fast-then-fine (FTF) two-stage retrieval framework that decomposes retrieval into a text-agnostic recall stage for efficient candidate selection and a text-guided rerank stage for fine-grained alignment. Specifically, in the recall stage, text-agnostic coarse-grained representations are employed for efficient candidate selection; in the rerank stage, a parameter-free balanced text-guided interaction block enhances fine-grained alignment without introducing additional learnable parameters. Furthermore, an inter- and intra-modal loss is designed to jointly optimize cross-modal alignment across multi-granular representations. Extensive experiments on public benchmarks demonstrate that the FTF achieves competitive retrieval accuracy while significantly improving retrieval efficiency compared with existing methods.
>
---
#### [new 033] KD-Judge: A Knowledge-Driven Automated Judge Framework for Functional Fitness Movements on Edge Devices
- **分类: cs.CV**

- **简介: 该论文提出KD-Judge，解决功能性健身动作重复标准自动化判断问题。通过知识驱动框架实现规则结构化与高效推理，提升判断透明度与效率。**

- **链接: [https://arxiv.org/pdf/2604.19834](https://arxiv.org/pdf/2604.19834)**

> **作者:** Shaibal Saha; Fan Li; Yunge Li; Arun Iyengar; Lucas Alves; Lanyu Xu
>
> **备注:** Accepted at IEEE/ACM CHASE 2026
>
> **摘要:** Functional fitness movements are widely used in training, competition, and health-oriented exercise programs, yet consistently enforcing repetition (rep) standards remains challenging due to subjective human judgment, time constraints, and evolving rules. Existing AI-based approaches mainly rely on learned scoring or reference-based comparisons and lack explicit rule-based, limiting transparency and deterministic rep-level validation. To address these limitations, we propose KD-Judge, a novel knowledge-driven automated judging framework for functional fitness movements. It converts unstructured rulebook standards into executable, machine-readable representations using an LLM-based retrieval-augmented generation and chain-of-thought rule-structuring pipeline. The structured rules are then incorporated by a deterministic rule-based judging system with pose-guided kinematic reasoning to assess rep validity and temporal boundaries. To improve efficiency on edge devices, including a high-performance desktop and the resource-constrained Jetson AGX Xavier, we introduce a dual strategy caching mechanism that can be selectively applied to reduce redundant and unnecessary computation. Experiments demonstrate reliable rule-structuring performance and accurate rep-level assessment, with judgment evaluation conducted on the CFRep dataset, achieving faster-than-real-time execution (real-time factor (RTF) < 1). When the proposed caching strategy is enabled, the system achieves up to 3.36x and 15.91x speedups on resource-constrained edge device compared to the non-caching baseline for pre-recorded and live-streaming scenarios, respectively. These results show that KD-Judge enables transparent, efficient, and scalable rule-grounded rep-level analysis that can complement human judging in practice.
>
---
#### [new 034] Hybrid Latent Reasoning with Decoupled Policy Optimization
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，解决视觉与文本融合中的语义丢失问题。提出HyLaR框架，结合离散文本与连续视觉表征，提升细粒度感知与多模态理解能力。**

- **链接: [https://arxiv.org/pdf/2604.20328](https://arxiv.org/pdf/2604.20328)**

> **作者:** Tao Cheng; Shi-Zhe Chen; Hao Zhang; Yixin Qin; Jinwen Luo; Zheng Wei
>
> **备注:** Tech report
>
> **摘要:** Chain-of-Thought (CoT) reasoning significantly elevates the complex problem-solving capabilities of multimodal large language models (MLLMs). However, adapting CoT to vision typically discretizes signals to fit LLM inputs, causing early semantic collapse and discarding fine-grained details. While external tools can mitigate this, they introduce a rigid bottleneck, confining reasoning to predefined operations. Although recent latent reasoning paradigms internalize visual states to overcome these limitations, optimizing the resulting hybrid discrete-continuous action space remains challenging. In this work, we propose HyLaR (Hybrid Latent Reasoning), a framework that seamlessly interleaves discrete text generation with continuous visual latent representations. Specifically, following an initial cold-start supervised fine-tuning (SFT), we introduce DePO (Decoupled Policy Optimization) to enable effective reinforcement learning within this hybrid space. DePO decomposes the policy gradient objective, applying independent trust-region constraints to the textual and latent components, alongside an exact closed-form von Mises-Fisher (vMF) KL regularizer. Extensive experiments demonstrate that HyLaR outperforms standard MLLMs and state-of-the-art latent reasoning approaches across fine-grained perception and general multimodal understanding benchmarks. Code is available at this https URL.
>
---
#### [new 035] Render-in-the-Loop: Vector Graphics Generation via Visual Self-Feedback
- **分类: cs.CV**

- **简介: 该论文属于文本或图像到矢量图形生成任务，旨在解决传统方法无法有效利用视觉上下文的问题。通过引入视觉反馈机制，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2604.20730](https://arxiv.org/pdf/2604.20730)**

> **作者:** Guotao Liang; Zhangcheng Wang; Juncheng Hu; Haitao Zhou; Ziteng Xue; Jing Zhang; Dong Xu; Qian Yu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown promising capabilities in generating Scalable Vector Graphics (SVG) via direct code synthesis. However, existing paradigms typically adopt an open-loop "blind drawing" approach, where models generate symbolic code sequences without perceiving intermediate visual outcomes. This methodology severely underutilizes the powerful visual priors embedded in MLLMs vision encoders, treating SVG generation as a disjointed textual sequence modeling task rather than an integrated visuo-spatial one. Consequently, models struggle to reason about partial canvas states and implicit occlusion relationships, which are visually explicit but textually ambiguous. To bridge this gap, we propose Render-in-the-Loop, a novel generation paradigm that reformulates SVG synthesis as a step-wise, visual-context-aware process. By rendering intermediate code states into a cumulative canvas, the model explicitly observes the evolving visual context at each step, leveraging on-the-fly feedback to guide subsequent generation. However, we demonstrate that applying this visual loop naively to off-the-shelf models is suboptimal due to their inability to leverage incremental visual-code mappings. To address this, we first utilize fine-grained path decomposition to construct dense multi-step visual trajectories, and then introduce a Visual Self-Feedback (VSF) training strategy to condition the next primitive generation on intermediate visual states. Furthermore, a Render-and-Verify (RaV) inference mechanism is proposed to effectively filter degenerate and redundant primitives. Our framework, instantiated on a multimodal foundation model, outperforms strong open-weight baselines on the standard MMSVGBench. This result highlights the remarkable data efficiency and generalization capability of our Render-in-the-Loop paradigm for both Text-to-SVG and Image-to-SVG tasks.
>
---
#### [new 036] Rabies diagnosis in low-data settings: A comparative study on the impact of data augmentation and transfer learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于疾病诊断任务，旨在解决低数据环境下狂犬病自动诊断问题。通过深度学习与数据增强技术提升模型性能，实现快速可靠检测。**

- **链接: [https://arxiv.org/pdf/2604.19823](https://arxiv.org/pdf/2604.19823)**

> **作者:** Khalil Akremi; Mariem Handous; Zied Bouslama; Farah Bassalah; Maryem Jebali; Mariem Hanachi; Ines Abdeljaoued-Tej
>
> **备注:** This work has been accepted for publication in ICMI IEEE Conference (04/2026)
>
> **摘要:** Rabies remains a major public health concern across many African and Asian countries, where accurate diagnosis is critical for effective epidemiological surveillance. The gold standard diagnostic methods rely heavily on fluorescence microscopy, necessitating skilled laboratory personnel for the accurate interpretation of results. Such expertise is often scarce, particularly in regions with low annual sample volumes. This paper presents an automated, AI-driven diagnostic system designed to address these challenges. We developed a robust pipeline utilizing fluorescent image analysis through transfer learning with four deep learning architectures: EfficientNetB0, EfficientNetB2, VGG16, and Vision Transformer (ViTB16). Three distinct data augmentation strategies were evaluated to enhance model generalization on a dataset of 155 microscopic images (123 positive and 32 negative). Our results demonstrate that TrivialAugmentWide was the most effective augmentation technique, as it preserved critical fluorescent patterns while improving model robustness. The EfficientNetB0 model, utilizing Geometric & Color augmentation and selected through stratified 3fold cross-validation, achieved optimal classification performance on cropped images. Despite constraints posed by class imbalance and a limited dataset size, this work confirms the viability of deep learning for automating rabies diagnosis. The proposed method enables fast and reliable detection with significant potential for further optimization. An online tool was deployed to facilitate practical access, establishing a framework for future medical imaging applications. This research underscores the potential of optimized deep learning models to transform rabies diagnostics and improve public health outcomes.
>
---
#### [new 037] Adapting TrOCR for Printed Tigrinya Text Recognition: Word-Aware Loss Weighting for Cross-Script Transfer Learning
- **分类: cs.CV**

- **简介: 该论文属于光学字符识别（OCR）任务，解决非洲语言Tigrinya文字识别问题。通过改进TrOCR模型，引入词 aware 损失加权，提升跨脚本迁移效果。**

- **链接: [https://arxiv.org/pdf/2604.20813](https://arxiv.org/pdf/2604.20813)**

> **作者:** Yonatan Haile Medhanie; Yuanhua Ni
>
> **备注:** Code and models available at this https URL Pre-trained models: this https URL, this https URL
>
> **摘要:** Transformer-based OCR models have shown strong performance on Latin and CJK scripts, but their application to African syllabic writing systems remains limited. We present the first adaptation of TrOCR for printed Tigrinya using the Ge'ez script. Starting from a pre-trained model, we extend the byte-level BPE tokenizer to cover 230 Ge'ez characters and introduce Word-Aware Loss Weighting to resolve systematic word-boundary failures that arise when applying Latin-centric BPE conventions to a new script. The unmodified model produces no usable output on Ge'ez text. After adaptation, the TrOCR-Printed variant achieves 0.22% Character Error Rate and 97.20% exact match accuracy on a held-out test set of 5,000 synthetic images from the GLOCR dataset. An ablation study confirms that Word-Aware Loss Weighting is the critical component, reducing CER by two orders of magnitude compared to vocabulary extension alone. The full pipeline trains in under three hours on a single 8 GB consumer GPU. All code, model weights, and evaluation scripts are publicly released.
>
---
#### [new 038] Structure-Augmented Standard Plane Detection with Temporal Aggregation in Blind-Sweep Fetal Ultrasound
- **分类: cs.CV**

- **简介: 该论文属于胎儿超声图像分析任务，旨在解决盲扫超声中标准平面检测不稳定的问题。通过结构增强和时间聚合方法提升检测准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.20591](https://arxiv.org/pdf/2604.20591)**

> **作者:** Keli Niu; He Zhao; Qianhui Men
>
> **摘要:** In low-resource settings, blind-sweep ultrasound provides a practical and accessible method for identifying fetal growth restriction. However, unlike freehand ultrasound which is subjectively controlled, detection of biometry plane in blind-sweep ultrasound is more challenging due to the uncontrolled fetal structure to be observed and the variaties of oblique planes in the scan. In this work, we propose a structure-augmented system to detect fetal abdomen plane, where the abdominal structure is highlighted using a segmentation prior. Since standard planes are emerging gradually, the decision boundary of the keyframes is unstable to predict. We thus aggregated the structure-augmented planes with a temporal sliding window to help stabilise keyframe localisation. Extensive results indicate that the structure-augmented temporal sliding strategy significantly improves and stabilises the detection of anatomically meaningful planes, which enables more reliable biometric measurements in blind-sweep ultrasound.
>
---
#### [new 039] Object Referring-Guided Scanpath Prediction with Perception-Enhanced Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出ScanVLA模型，解决对象指称引导的注视路径预测任务，通过融合视觉与语言信息并增强位置感知，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.20361](https://arxiv.org/pdf/2604.20361)**

> **作者:** Rong Quan; Yantao Lai; Dong Liang; Jie Qin
>
> **备注:** ICMR 2026
>
> **摘要:** Object Referring-guided Scanpath Prediction (ORSP) aims to predict the human attention scanpath when they search for a specific target object in a visual scene according to a linguistic description describing the object. Multimodal information fusion is a key point of ORSP. Therefore, we propose a novel model, ScanVLA, to first exploit a Vision-Language Model (VLM) to extract and fuse inherently aligned visual and linguistic feature representations from the input image and referring expression. Next, to enhance the ScanVLA's perception of fine-grained positional information, we not only propose a novel History Enhanced Scanpath Decoder (HESD) that directly takes historical fixations' position information as input to help predict a more reasonable position for the current fixation, but also adopt a frozen Segmentation LoRA as an auxiliary component to help localize the referred object more precisely, which improves the scanpath prediction task without incurring additional large computational and time costs. Extensive experimental results demonstrate that ScanVLA can significantly outperform existing scanpath prediction methods under object referring.
>
---
#### [new 040] Hallucination Early Detection in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决多物体生成中的幻觉问题。提出HEaD+方法，通过早期检测减少无效生成，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.20354](https://arxiv.org/pdf/2604.20354)**

> **作者:** Federico Betti; Lorenzo Baraldi; Lorenzo Baraldi; Rita Cucchiara; Nicu Sebe
>
> **备注:** 21 pages, 6 figures, 4 tables. Published in International Journal of Computer Vision (IJCV)
>
> **摘要:** Text-to-Image generation has seen significant advancements in output realism with the advent of diffusion models. However, diffusion models encounter difficulties when tasked with generating multiple objects, frequently resulting in hallucinations where certain entities are omitted. While existing solutions typically focus on optimizing latent representations within diffusion models, the relevance of the initial generation seed is typically underestimated. While using various seeds in multiple iterations can improve results, this method also significantly increases time and energy costs. To address this challenge, we introduce HEaD+ (Hallucination Early Detection +), a novel approach designed to identify incorrect generations early in the diffusion process. The HEaD+ framework integrates cross-attention maps and textual information with a novel input, the Predicted Final Image. The objective is to assess whether to proceed with the current generation or restart it with a different seed, thereby exploring multiple-generation seeds while conserving time. HEaD+ is trained on the newly created InsideGen dataset of 45,000 generated images, each containing prompts with up to seven objects. Our findings demonstrate a 6-8% increase in the likelihood of achieving a complete generation (i.e., an image accurately representing all specified subjects) with four objects when applying HEaD+ alongside existing models. Additionally, HEaD+ reduces generation times by up to 32% when aiming for a complete image, enhancing the efficiency of generating complete and accurate object representations relative to leading models. Moreover, we propose an integrated localization module that predicts object centroid positions and verifies pairwise spatial relations (if requested by the users) at an intermediate timestep, gating generation together with object presence to further improve relation-consistent outcomes.
>
---
#### [new 041] Wan-Image: Pushing the Boundaries of Generative Visual Intelligence
- **分类: cs.CV**

- **简介: 该论文提出Wan-Image，解决图像生成中的可控性、复杂文本和身份保持问题，通过多模态架构提升专业视觉生成能力。**

- **链接: [https://arxiv.org/pdf/2604.19858](https://arxiv.org/pdf/2604.19858)**

> **作者:** Chaojie Mao; Chen-Wei Xie; Chongyang Zhong; Haoyou Deng; Jiaxing Zhao; Jie Xiao; Jinbo Xing; Jingfeng Zhang; Jingren Zhou; Jingyi Zhang; Jun Dan; Kai Zhu; Kang Zhao; Keyu Yan; Minghui Chen; Pandeng Li; Shuangle Chen; Tong Shen; Yu Liu; Yue Jiang; Yulin Pan; Yuxiang Tuo; Zeyinzi Jiang; Zhen Han; Ang Wang; Bang Zhang; Baole Ai; Bin Wen; Boang Feng; Feiwu Yu; Gang Wang; Haiming Zhao; He Kang; Jianjing Xiang; Jianyuan Zeng; Jinkai Wang; Ke Sun; Linqian Wu; Pei Gong; Pingyu Wu; Ruiwen Wu; Tongtong Su; Wenmeng Zhou; Wenting Shen; Wenyuan Yu; Xianjun Xu; Xiaoming Huang; Xiejie Shen; Xin Xu; Yan Kou; Yangyu Lv; Yifan Zhai; Yitong Huang; Yun Zheng; Yuntao Hong; Zhicheng Zhang
>
> **摘要:** We present Wan-Image, a unified visual generation system explicitly engineered to paradigm-shift image generation models from casual synthesizers into professional-grade productivity tools. While contemporary diffusion models excel at aesthetic generation, they frequently encounter critical bottlenecks in rigorous design workflows that demand absolute controllability, complex typography rendering, and strict identity preservation. To address these challenges, Wan-Image features a natively unified multi-modal architecture by synergizing the cognitive capabilities of large language models with the high-fidelity pixel synthesis of diffusion transformers, which seamlessly translates highly nuanced user intents into precise visual outputs. It is fundamentally powered by large-scale multi-modal data scaling, a systematic fine-grained annotation engine, and curated reinforcement learning data to surpass basic instruction following and unlock expert-level professional capabilities. These include ultra-long complex text rendering, hyper-diverse portrait generation, palette-guided generation, multi-subject identity preservation, coherent sequential visual generation, precise multi-modal interactive editing, native alpha-channel generation, and high-efficiency 4K synthesis. Across diverse human evaluations, Wan-Image exceeds Seedream 5.0 Lite and GPT Image 1.5 in overall performance, reaching parity with Nano Banana Pro in challenging tasks. Ultimately, Wan-Image revolutionizes visual content creation across e-commerce, entertainment, education, and personal productivity, redefining the boundaries of professional visual synthesis.
>
---
#### [new 042] Learning Spatial-Temporal Coherent Correlations for Speech-Preserving Facial Expression Manipulation
- **分类: cs.CV**

- **简介: 该论文属于语音保持的面部表情操控任务，解决缺乏配对数据导致的应用限制。提出STCCL算法，通过空间-时间相关性学习，提升表情修改时语音相关面部动画的保留效果。**

- **链接: [https://arxiv.org/pdf/2604.20226](https://arxiv.org/pdf/2604.20226)**

> **作者:** Tianshui Chen; Jianman Lin; Zhijing Yang; Chunmei Qing; Guangrun Wang; Liang Lin
>
> **摘要:** Speech-preserving facial expression manipulation (SPFEM) aims to modify facial emotions while meticulously maintaining the mouth animation associated with spoken content. Current works depend on inaccessible paired training samples for the person, where two aligned frames exhibit the same speech content yet differ in emotional expression, limiting the SPFEM applications in real-world scenarios. In this work, we discover that speakers who convey the same content with different emotions exhibit highly correlated local facial animations in both spatial and temporal spaces, providing valuable supervision for SPFEM. To capitalize on this insight, we propose a novel spatial-temporal coherent correlation learning (STCCL) algorithm, which models the aforementioned correlations as explicit metrics and integrates the metrics to supervise manipulating facial expression and meanwhile better preserving the facial animation of spoken content. To this end, it first learns a spatial coherent correlation metric, ensuring that the visual correlations of adjacent local regions within an image linked to a specific emotion closely resemble those of corresponding regions in an image linked to a different emotion. Simultaneously, it develops a temporal coherent correlation metric, ensuring that the visual correlations of specific regions across adjacent image frames associated with one emotion are similar to those in the corresponding regions of frames associated with another emotion. Recognizing that visual correlations are not uniform across all regions, we have also crafted a correlation-aware adaptive strategy that prioritizes regions that present greater challenges. During SPFEM model training, we construct the spatial-temporal coherent correlation metric between corresponding local regions of the input and output image frames as an additional loss to supervise the generation process.
>
---
#### [new 043] Optimizing Data Augmentation for Real-Time Small UAV Detection: A Lightweight Context-Aware Approach
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在提升轻量级模型在实时小无人机检测中的性能。通过设计一种结合Mosaic和HSV增强的数据增强方法，解决模型泛化能力和稳定性问题。**

- **链接: [https://arxiv.org/pdf/2604.19999](https://arxiv.org/pdf/2604.19999)**

> **作者:** Amir Zamani; Zeinab Abedini
>
> **备注:** Accepted for presentation at the 34th International Conference on Electrical Engineering (ICEE 2026)
>
> **摘要:** Visual detection of Unmanned Aerial Vehicles (UAVs) is a critical task in surveillance systems due to their small physical size and environmental challenges. Although deep learning models have achieved significant progress, deploying them on edge devices necessitates the use of lightweight models, such as YOLOv11 Nano, which possess limited learning capacity. In this research, an efficient and context-aware data augmentation pipeline, combining Mosaic strategies and HSV color-space adaptation, is proposed to enhance the performance of these models. Experimental results on four standard datasets demonstrate that the proposed approach, compared to heavy and instance-level methods like Copy-Paste, not only prevents the generation of synthetic artifacts and overfitting but also significantly improves mean Average Precision (mAP) across all scenarios. Furthermore, the evaluation of generalization capability under foggy conditions revealed that the proposed method offers the optimal balance between Precision and stability for real-time systems, whereas alternative methods, such as MixUp, are effective only in specific applications.
>
---
#### [new 044] Rethinking Where to Edit: Task-Aware Localization for Instruction-Based Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决IIE中过度编辑导致无关区域变化的问题。提出一种无需训练的任务感知定位框架，提升编辑区域的准确性。**

- **链接: [https://arxiv.org/pdf/2604.20258](https://arxiv.org/pdf/2604.20258)**

> **作者:** Jingxuan He; Xiyu Wang; Mengyu Zheng; Xiangyu Zeng; Yunke Wang; Chang Xu
>
> **摘要:** Instruction-based image editing (IIE) aims to modify images according to textual instructions while preserving irrelevant content. Despite recent advances in diffusion transformers, existing methods often suffer from over-editing, introducing unintended changes to regions unrelated to the desired edit. We identify that this limitation arises from the lack of an explicit mechanism for edit localization. In particular, different editing operations (e.g., addition, removal and replacement) induce distinct spatial patterns, yet current IIE models typically treat localization in a task-agnostic manner. To address this limitation, we propose a training-free, task-aware edit localization framework that exploits the intrinsic source and target image streams within IIE models. For each image stream, We first obtain attention-based edit cues, and then construct feature centroids based on these attentive cues to partition tokens into edit and non-edit regions. Based on the observation that optimal localization is inherently task-dependent, we further introduce a unified mask construction strategy that selectively leverages source and target image streams for different editing tasks. We provide a systematic analysis for our proposed insights and approaches. Extensive experiments on EdiVal-Bench demonstrate our framework consistently improves non-edit region consistency while maintaining strong instruction-following performance on top of powerful recent image editing backbones, including Step1X-Edit and Qwen-Image-Edit.
>
---
#### [new 045] Environmental Understanding Vision-Language Model for Embodied Agent
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于机器人环境理解任务，旨在提升视觉-语言模型在具身代理中的环境理解能力。通过优化四个核心技能，提高任务执行的可靠性与成功率。**

- **链接: [https://arxiv.org/pdf/2604.19839](https://arxiv.org/pdf/2604.19839)**

> **作者:** Jinsik Bang; Jaeyeon Bae; Donggyu Lee; Siyeol Jung; Taehwan Kim
>
> **备注:** CVPR Findings 2026, Project Page: this https URL
>
> **摘要:** Vision-language models (VLMs) have shown strong perception and reasoning abilities for instruction-following embodied agents. However, despite these abilities and their generalization performance, they still face limitations in environmental understanding, often failing on interactions or relying on environment metadata during execution. To address this challenge, we propose a novel framework named Environmental Understanding Embodied Agent (EUEA), which fine-tunes four core skills: 1) object perception for identifying relevant objects, 2) task planning for generating interaction subgoals, 3) action understanding for judging success likelihood, and 4) goal recognition for determining goal completion. By fine-tuning VLMs with EUEA skills, our framework enables more reliable task execution for instruction-following. We further introduce a recovery step that leverages these core skills and a group relative policy optimization (GRPO) stage that refines inconsistent skill predictions. The recovery step samples alternative actions to correct failure cases, and the GRPO stage refines inconsistent skill predictions. Across ALFRED tasks, our VLM significantly outperforms a behavior-cloning baseline, achieving an 8.86% improvement in average success rate. The recovery and GRPO stages provide an additional 3.03% gain, further enhancing overall performance. Finally, our skill-level analyses reveal key limitations in the environmental understanding of closed- and open-source VLMs and identify the capabilities necessary for effective agent-environment interaction.
>
---
#### [new 046] HumanScore: Benchmarking Human Motions in Generated Videos
- **分类: cs.CV**

- **简介: 该论文属于视频生成质量评估任务，旨在解决AI生成视频中人体动作真实性评估问题。提出HumanScore框架，通过六个指标评估动作合理性与生物力学一致性。**

- **链接: [https://arxiv.org/pdf/2604.20157](https://arxiv.org/pdf/2604.20157)**

> **作者:** Yusu Fang; Tiange Xiang; Tian Tan; Narayan Schuetz; Scott Delp; Li Fei-Fei; Ehsan Adeli
>
> **摘要:** Recent advances in model architectures, compute, and data scale have driven rapid progress in video generation, producing increasingly realistic content. Yet, no prior method systematically measures how faithfully these systems render human bodies and motion dynamics. In this paper, we present HumanScore, a systematic framework to evaluate the quality of human motions in AI-generated videos. HumanScore defines six interpretable metrics spanning kinematic plausibility, temporal stability, and biomechanical consistency, enabling fine-grained diagnosis beyond visual realism alone. Through carefully designed prompts, we elicit a diverse set of movements at varying intensities and evaluate videos generated by thirteen state-of-the-art models. Our analysis reveals consistent gaps between perceptual plausibility and motion biomechanical fidelity, identifies recurrent failure modes (e.g., temporal jitter, anatomically implausible poses, and motion drift), and produces robust model rankings from quantitative and physically meaningful criteria.
>
---
#### [new 047] DeVI: Physics-based Dexterous Human-Object Interaction via Synthetic Video Imitation
- **分类: cs.CV**

- **简介: 该论文提出DeVI框架，解决物理逼真的人机交互控制问题。通过合成视频实现无需3D数据的零样本泛化，提升操作精度与多样性。**

- **链接: [https://arxiv.org/pdf/2604.20841](https://arxiv.org/pdf/2604.20841)**

> **作者:** Hyeonwoo Kim; Jeonghwan Kim; Kyungwon Cho; Hanbyul Joo
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recent advances in video generative models enable the synthesis of realistic human-object interaction videos across a wide range of scenarios and object categories, including complex dexterous manipulations that are difficult to capture with motion capture systems. While the rich interaction knowledge embedded in these synthetic videos holds strong potential for motion planning in dexterous robotic manipulation, their limited physical fidelity and purely 2D nature make them difficult to use directly as imitation targets in physics-based character control. We present DeVI (Dexterous Video Imitation), a novel framework that leverages text-conditioned synthetic videos to enable physically plausible dexterous agent control for interacting with unseen target objects. To overcome the imprecision of generative 2D cues, we introduce a hybrid tracking reward that integrates 3D human tracking with robust 2D object tracking. Unlike methods relying on high-quality 3D kinematic demonstrations, DeVI requires only the generated video, enabling zero-shot generalization across diverse objects and interaction types. Extensive experiments demonstrate that DeVI outperforms existing approaches that imitate 3D human-object interaction demonstrations, particularly in modeling dexterous hand-object interactions. We further validate the effectiveness of DeVI in multi-object scenes and text-driven action diversity, showcasing the advantage of using video as an HOI-aware motion planner.
>
---
#### [new 048] CCTVBench: Contrastive Consistency Traffic VideoQA Benchmark for Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文提出CCTVBench，用于评估多模态大模型在交通视频问答中的对比一致性，解决模型在真实与假设场景中可靠识别危险的问题。**

- **链接: [https://arxiv.org/pdf/2604.20460](https://arxiv.org/pdf/2604.20460)**

> **作者:** Xingcheng Zhou; Hao Guo; Rui Song; Walter Zimmer; Mingyu Liu; André Schamschurko; Hu Cao; Alois Knoll
>
> **摘要:** Safety-critical traffic reasoning requires contrastive consistency: models must detect true hazards when an accident occurs, and reliably reject plausible-but-false hypotheses under near-identical counterfactual scenes. We present CCTVBench, a Contrastive Consistency Traffic VideoQA Benchmark built on paired real accident videos and world-model-generated counterfactual counterparts, together with minimally different, mutually exclusive hypothesis questions. CCTVBench enforces a single structured decision pattern over each video question quadruple and provides actionable diagnostics that decompose failures into positive omission, positive swap, negative hallucination, and mutual-exclusivity violation, while separating video versus question consistency. Experiments across open-source and proprietary video LLMs reveal a large and persistent gap between standard per-instance QA metrics and quadruple-level contrastive consistency, with unreliable none-of-the-above rejection as a key bottleneck. Finally, we introduce C-TCD, a contrastive decoding approach leveraging a semantically exclusive counterpart video as the contrast input at inference time, improving both instance-level QA and contrastive consistency.
>
---
#### [new 049] Beyond ZOH: Advanced Discretization Strategies for Vision Mamba
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型任务，旨在解决Vision Mamba中离散化方法导致的精度问题。通过对比多种离散化策略，提出更优方案提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.20606](https://arxiv.org/pdf/2604.20606)**

> **作者:** Fady Ibrahim; Guangjun Liu; Guanghui Wang
>
> **摘要:** Vision Mamba, as a state space model (SSM), employs a zero-order hold (ZOH) discretization, which assumes that input signals remain constant between sampling instants. This assumption degrades temporal fidelity in dynamic visual environments and constrains the attainable accuracy of modern SSM-based vision models. In this paper, we present a systematic and controlled comparison of six discretization schemes instantiated within the Vision Mamba framework: ZOH, first-order hold (FOH), bilinear/Tustin transform (BIL), polynomial interpolation (POL), higher-order hold (HOH), and the fourth-order Runge-Kutta method (RK4). We evaluate each method on standard visual benchmarks to quantify its influence in image classification, semantic segmentation, and object detection. Our results demonstrate that POL and HOH yield the largest gains in accuracy at the cost of higher training-time computation. In contrast, the BIL provides consistent improvements over ZOH with modest additional overhead, offering the most favorable trade-off between precision and efficiency. These findings elucidate the pivotal role of discretization in SSM-based vision architectures and furnish empirically grounded justification for adopting BIL as the default discretization baseline for state-of-the-art SSM models.
>
---
#### [new 050] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型生成中出现的幻觉问题。通过提出MPD框架，在不降低性能的前提下有效减少幻觉。**

- **链接: [https://arxiv.org/pdf/2604.20366](https://arxiv.org/pdf/2604.20366)**

> **作者:** Xingyu Zhu; Junfeng Fang; Shuo Wang; Beier Zhu; Zhicai Wang; Yonghui Yang; Xiangnan He
>
> **备注:** ACL 2026 (Oral)
>
> **摘要:** Large Vision-Language Models (LVLMs) exhibit powerful generative capabilities but frequently produce hallucinations that compromise output reliability. Fine-tuning on annotated data devoid of hallucinations offers the most direct solution, while its high computational cost motivates recent representation-based methods, which focus on mitigating hallucinatory components within hidden representations. Though efficient, we empirically observe that these methods degrade general generation capacity due to incomplete extraction of hallucination components and non-selective parameter updates. To address these limitations, we propose MPD, a dual-stage framework for mitigating hallucinations without performance degradation. Specifically, our MPD relies on two essential factors: (1) semantic-aware component disentanglement to extract pure hallucination components, and (2) interpretable parameter updates that selectively modify parameters most relevant to hallucination. Extensive experiments demonstrate that MPD achieves state-of-the-art performance, reducing hallucinations by 23.4\% while maintaining 97.4\% of general generative capability as evaluated on LLaVA-Bench and MME, with no additional computational cost.
>
---
#### [new 051] Global Offshore Wind Infrastructure: Deployment and Operational Dynamics from Dense Sentinel-1 Time Series
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感监测任务，解决 offshore 风电设施动态监测问题，通过 Sentinel-1 数据构建时间序列分析库，支持部署与运营分析。**

- **链接: [https://arxiv.org/pdf/2604.20822](https://arxiv.org/pdf/2604.20822)**

> **作者:** Thorsten Hoeser; Felix Bachofer; Claudia Kuenzer
>
> **备注:** 25 pages, 16 figures
>
> **摘要:** The offshore wind energy sector is expanding rapidly, increasing the need for independent, high-temporal-resolution monitoring of infrastructure deployment and operation at global scale. While Earth Observation based offshore wind infrastructure mapping has matured for spatial localization, existing open datasets lack temporally dense and semantically fine-grained information on construction and operational dynamics. We introduce a global Sentinel-1 synthetic aperture radar (SAR) time series data corpus that resolves deployment and operational phases of offshore wind infrastructure from 2016Q1 to 2025Q1. Building on an updated object detection workflow, we compile 15,606 time series at detected infrastructure locations, with overall 14,840,637 events as analysis-ready 1D SAR backscatter profiles, one profile per Sentinel-1 acquisition and location. To enable direct use and benchmarking, we release (i) the analysis ready 1D SAR profiles, (ii) event-level baseline semantic labels generated by a rule-based classifier, and (iii) an expert-annotated benchmark dataset of 553 time series with 328,657 event labels. The baseline classifier achieves a macro F1 score of 0.84 in event-wise evaluation and an area under the collapsed edit similarity-quality threshold curve (AUC) of 0.785, indicating temporal coherence. We demonstrate that the resulting corpus supports global-scale analyses of deployment dynamics, the identification of differences in regional deployment patterns, vessel interactions, and operational events, and provides a reference for developing and comparing time series classification methods for offshore wind infrastructure monitoring.
>
---
#### [new 052] Lucky High Dynamic Range Smartphone Imaging
- **分类: cs.CV**

- **简介: 该论文属于高动态范围成像任务，旨在提升智能手机相机的动态范围。通过轻量网络和间接处理原始像素，解决传统方法产生的伪影问题，并实现跨设备的零样本泛化。**

- **链接: [https://arxiv.org/pdf/2604.19976](https://arxiv.org/pdf/2604.19976)**

> **作者:** Baiang Li; Ruyu Yan; Ethan Tseng; Zhoutong Zhang; Adam Finkelstein; Jiawen Chen; Felix Heide
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** While the human eye can perceive an impressive twenty stops of dynamic range, smartphone camera sensors remain limited to about twelve stops despite decades of research. A variety of high dynamic range (HDR) image capture and processing techniques have been proposed, and, in practice, they can extend the dynamic range by 3-5 stops for handheld photography. This paper proposes an approach that robustly captures dynamic range using a handheld smartphone camera and lightweight networks suitable for running on mobile devices. Our method operates indirectly on linear raw pixels in bracketed exposures. Every pixel in the final HDR image is a convex combination of input pixels in the neighborhood, adjusted for exposure, and thus avoids hallucination artifacts typical of recent deep image synthesis networks. We validate our system on both synthetic imagery and unseen real bracketed images -- we confirm zero-shot generalization of the method to smartphone camera captures. Our iterative inference architecture is capable of processing an arbitrary number of bracketed input photos, and we show examples from capture stacks containing 3--9 images. Our training process relies only on synthetic captures yet generalizes to unseen real photos from several cameras. Moreover, we show that this training scheme improves other SOTA methods over their pretrained counterparts.
>
---
#### [new 053] GeoRelight: Learning Joint Geometrical Relighting and Reconstruction with Flexible Multi-Modal Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文提出GeoRelight，解决单图重光照与3D重建联合问题。通过多模态扩散Transformer，结合合成与真实数据，提升物理一致性与性能。**

- **链接: [https://arxiv.org/pdf/2604.20715](https://arxiv.org/pdf/2604.20715)**

> **作者:** Yuxuan Xue; Ruofan Liang; Egor Zakharov; Timur Bagautdinov; Chen Cao; Giljoo Nam; Shunsuke Saito; Gerard Pons-Moll; Javier Romero
>
> **备注:** CVPR 2026 Highlight; Project page: this https URL
>
> **摘要:** Relighting a person from a single photo is an attractive but ill-posed task, as a 2D image ambiguously entangles 3D geometry, intrinsic appearance, and illumination. Current methods either use sequential pipelines that suffer from error accumulation, or they do not explicitly leverage 3D geometry during relighting, which limits physical consistency. Since relighting and estimation of 3D geometry are mutually beneficial tasks, we propose a unified Multi-Modal Diffusion Transformer (DiT) that jointly solves for both: GeoRelight. We make this possible through two key technical contributions: isotropic NDC-Orthographic Depth (iNOD), a distortion-free 3D representation compatible with latent diffusion models; and a strategic mixed-data training method that combines synthetic and auto-labeled real data. By solving geometry and relighting jointly, GeoRelight achieves better performance than both sequential models and previous systems that ignored geometry.
>
---
#### [new 054] Image Generators are Generalist Vision Learners
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决视觉理解与生成问题。通过图像生成预训练，模型学习到通用视觉表征，实现多种视觉任务的最优性能。**

- **链接: [https://arxiv.org/pdf/2604.20329](https://arxiv.org/pdf/2604.20329)**

> **作者:** Valentin Gabeur; Shangbang Long; Songyou Peng; Paul Voigtlaender; Shuyang Sun; Yanan Bao; Karen Truong; Zhicheng Wang; Wenlei Zhou; Jonathan T. Barron; Kyle Genova; Nithish Kannen; Sherry Ben; Yandong Li; Mandy Guo; Suhas Yogin; Yiming Gu; Huizhong Chen; Oliver Wang; Saining Xie; Howard Zhou; Kaiming He; Thomas Funkhouser; Jean-Baptiste Alayrac; Radu Soricut
>
> **备注:** Project Page: this http URL
>
> **摘要:** Recent works show that image and video generators exhibit zero-shot visual understanding behaviors, in a way reminiscent of how LLMs develop emergent capabilities of language understanding and reasoning from generative pretraining. While it has long been conjectured that the ability to create visual content implies an ability to understand it, there has been limited evidence that generative vision models have developed strong understanding capabilities. In this work, we demonstrate that image generation training serves a role similar to LLM pretraining, and lets models learn powerful and general visual representations that enable SOTA performance on various vision tasks. We introduce Vision Banana, a generalist model built by instruction-tuning Nano Banana Pro (NBP) on a mixture of its original training data alongside a small amount of vision task data. By parameterizing the output space of vision tasks as RGB images, we seamlessly reframe perception as image generation. Our generalist model, Vision Banana, achieves SOTA results on a variety of vision tasks involving both 2D and 3D understanding, beating or rivaling zero-shot domain-specialists, including Segment Anything Model 3 on segmentation tasks, and the Depth Anything series on metric depth estimation. We show that these results can be achieved with lightweight instruction-tuning without sacrificing the base model's image generation capabilities. The superior results suggest that image generation pretraining is a generalist vision learner. It also shows that image generation serves as a unified and universal interface for vision tasks, similar to text generation's role in language understanding and reasoning. We could be witnessing a major paradigm shift for computer vision, where generative vision pretraining takes a central role in building Foundational Vision Models for both generation and understanding.
>
---
#### [new 055] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出OMIBench，用于评估大视觉语言模型在多图像推理任务中的表现。针对现有基准仅关注单图分析的问题，OMIBench涵盖多学科奥赛题目，旨在提升模型跨图像推理能力。**

- **链接: [https://arxiv.org/pdf/2604.20806](https://arxiv.org/pdf/2604.20806)**

> **作者:** Qiguang Chen; Chengyu Luan; Jiajun Wu; Qiming Yu; Yi Yang; Yizhuo Li; Jingqi Tong; Xiachong Feng; Libo Qin; Wanxiang Che
>
> **备注:** ACL 2026 Camera Ready
>
> **摘要:** Large vision-language models (LVLMs) have made substantial advances in reasoning tasks at the Olympiad level. Nevertheless, current Olympiad-level multimodal reasoning benchmarks for these models often emphasize single-image analysis and fail to exploit contextual information across multiple images. We present OMIBench, a benchmark designed to evaluate Olympiad-level reasoning when the required evidence is distributed over multiple images. It contains problems from biology, chemistry, mathematics, and physics Olympiads, together with manually annotated rationales and evaluation protocols for both exact and semantic answer matching. Across extensive experiments on OMIBench, we observe meaningful performance gaps in existing models. Even the strongest LVLMs, such as Gemini-3-Pro, attain only about 50% on the benchmark. These results position OMIBench as a focused resources for studying and improving multi-image reasoning in LVLMs.
>
---
#### [new 056] MMCORE: MultiModal COnnection with Representation Aligned Latent Embeddings
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出MMCORE，用于多模态图像生成与编辑任务。解决如何高效融合视觉与语言模型的问题，通过预训练VLM生成嵌入作为扩散模型的条件信号，提升生成质量并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2604.19902](https://arxiv.org/pdf/2604.19902)**

> **作者:** Zijie Li; Yichun Shi; Jingxiang Sun; Ye Wang; Yixuan Huang; Zhiyao Guo; Xiaochen Lian; Peihao Zhu; Yu Tian; Zhonghua Zhai; Peng Wang
>
> **摘要:** We present MMCORE, a unified framework designed for multimodal image generation and editing. MMCORE leverages a pre-trained Vision-Language Model (VLM) to predict semantic visual embeddings via learnable query tokens, which subsequently serve as conditioning signals for a diffusion model. This streamlined design effectively transfers the rich understanding and reasoning capabilities of VLMs into the visual generation process. By obviating the need for deep fusion between autoregressive and diffusion models or training from scratch, MMCORE significantly reduces computational overhead while maintaining high-fidelity synthesis. MMCORE seamlessly integrates text-to-image synthesis with interleaved image generation, demonstrating robust multimodal comprehension in complex scenarios such as spatial reasoning and visual grounding. Comprehensive evaluations indicate that MMCORE consistently outperforms state-of-the-art baselines across a broad spectrum of text-to-image and single/multi-image editing benchmarks.
>
---
#### [new 057] LLaDA2.0-Uni: Unifying Multimodal Understanding and Generation with Diffusion Large Language Model
- **分类: cs.CV**

- **简介: 该论文提出LLaDA2.0-Uni，一种统一的多模态大语言模型，解决多模态理解和生成任务，通过扩散机制实现高效图像生成与编辑。**

- **链接: [https://arxiv.org/pdf/2604.20796](https://arxiv.org/pdf/2604.20796)**

> **作者:** Inclusion AI; Tiwei Bie; Haoxing Chen; Tieyuan Chen; Zhenglin Cheng; Long Cui; Kai Gan; Zhicheng Huang; Zhenzhong Lan; Haoquan Li; Jianguo Li; Tao Lin; Qi Qin; Hongjun Wang; Xiaomei Wang; Haoyuan Wu; Yi Xin; Junbo Zhao
>
> **备注:** LLaDA2.0-Uni Technical Report
>
> **摘要:** We present LLaDA2.0-Uni, a unified discrete diffusion large language model (dLLM) that supports multimodal understanding and generation within a natively integrated framework. Its architecture combines a fully semantic discrete tokenizer, a MoE-based dLLM backbone, and a diffusion decoder. By discretizing continuous visual inputs via SigLIP-VQ, the model enables block-level masked diffusion for both text and vision inputs within the backbone, while the decoder reconstructs visual tokens into high-fidelity images. Inference efficiency is enhanced beyond parallel decoding through prefix-aware optimizations in the backbone and few-step distillation in the decoder. Supported by carefully curated large-scale data and a tailored multi-stage training pipeline, LLaDA2.0-Uni matches specialized VLMs in multimodal understanding while delivering strong performance in image generation and editing. Its native support for interleaved generation and reasoning establishes a promising and scalable paradigm for next-generation unified foundation models. Codes and models are available at this https URL.
>
---
#### [new 058] Opportunistic Bone-Loss Screening from Routine Knee Radiographs Using a Multi-Task Deep Learning Framework with Sensitivity-Constrained Threshold Optimization
- **分类: cs.CV**

- **简介: 该论文提出一种多任务深度学习框架，用于从常规膝关节X光片中筛查骨量丢失，解决 osteoporosis 和 osteopenia 早期诊断问题。**

- **链接: [https://arxiv.org/pdf/2604.20268](https://arxiv.org/pdf/2604.20268)**

> **作者:** Zhaochen Li; Xinghao Yan; Runni Zhou; Xiaoyang Li; Chenjie Zhu; Gege Wang; Yu Shi; Lixin Zhang; Rongrong Fu; Liehao Yan; Yuan Chai
>
> **摘要:** Background: Osteoporosis and osteopenia are often undiagnosed until fragility fractures occur. Dual-energy X-ray absorptiometry (DXA) is the reference standard for bone mineral density (BMD) assessment, but access remains limited. Knee radiographs are obtained at high volume for osteoarthritis evaluation and may offer an opportunity for opportunistic bone-loss screening. Objective: To develop and evaluate a multi-task deep learning system for opportunistic bone-loss screening from routine knee radiographs without additional imaging or patient visits. Methods: We developed STR-Net, a multi-task framework for single-channel grayscale knee radiographs. The model includes a shared backbone, global average pooling feature aggregation, a shared neck, and a task-aware representation routing module connected to three task-specific heads: binary screening (Normal vs. Bone Loss), severity sub-classification (Osteopenia vs. Osteoporosis), and weakly coupled T-score regression with optional clinical variables. A sensitivity-constrained threshold optimization strategy (minimum sensitivity >= 0.86) was applied. The dataset included 1,570 knee radiographs, split at the patient level into training (n=1,120), validation (n=226), and test (n=224) sets. Results: On the held-out test set, STR-Net achieved an AUROC of 0.933, sensitivity of 0.904, specificity of 0.773, and AUPRC of 0.956 for binary screening. Severity sub-classification achieved an AUROC of 0.898. The T-score regression branch showed a Pearson correlation of 0.801 with DXA-measured T-scores in a pilot subset (n=31), with MAE of 0.279 and RMSE of 0.347. Conclusions: STR-Net enables single-pass bone-loss screening, severity stratification, and quantitative T-score estimation from routine knee radiographs. Prospective clinical validation is needed before deployment.
>
---
#### [new 059] SignDATA: Data Pipeline for Sign Language Translation
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SignDATA，解决手语数据预处理不一致的问题，通过标准化流程生成可学习的姿势或视频数据，支持多种后端和配置管理。**

- **链接: [https://arxiv.org/pdf/2604.20357](https://arxiv.org/pdf/2604.20357)**

> **作者:** Kuanwei Chen; Tingyi Lin
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** Sign-language datasets are difficult to preprocess consistently because they vary in annotation schema, clip timing, signer framing, and privacy constraints. Existing work usually reports downstream models, while the preprocessing pipeline that converts raw video into training-ready pose or video artifacts remains fragmented, backend-specific, and weakly documented. We present SignDATA, a config-driven preprocessing toolkit that standardizes heterogeneous sign-language corpora into comparable outputs for learning. The system supports two end-to-end recipes: a pose recipe that performs acquisition, manifesting, person localization, clipping, cropping, landmark extraction, normalization, and WebDataset export, and a video recipe that replaces pose extraction with signer-cropped video packaging. SignDATA exposes interchangeable MediaPipe and MMPose backends behind a common interface, typed job schemas, experiment-level overrides, and per-stage checkpointing with config- and manifest-aware hashes. We validate the toolkit through a research-oriented evaluation design centered on backend comparison, preprocessing ablations, and privacy-aware video generation on datasets. Our contribution is a reproducible preprocessing layer for sign-language research that makes extractor choice, normalization policy, and privacy tradeoffs explicit, configurable, and empirically this http URL is available at this https URL.
>
---
#### [new 060] DistortBench: Benchmarking Vision Language Models on Image Distortion Identification
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出DistortBench，用于评估视觉语言模型在图像失真识别上的能力，解决低层感知理解不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.19966](https://arxiv.org/pdf/2604.19966)**

> **作者:** Divyanshu Goyal; Akhil Eppa; Vanya Bannihatti Kumar
>
> **摘要:** Vision-language models (VLMs) are increasingly used in settings where sensitivity to low-level image degradations matters, including content moderation, image restoration, and quality monitoring. Yet their ability to recognize distortion type and severity remains poorly understood. We present DistortBench, a diagnostic benchmark for no-reference distortion perception in VLMs. DistortBench contains 13,500 four-choice questions covering 27 distortion types, six perceptual categories, and five severity levels: 25 distortions inherit KADID-10k calibrations, while two added rotation distortions use monotonic angle-based levels. We evaluate 18 VLMs, including 17 open-weight models from five families and one proprietary model. Despite strong performance on high-level vision-language tasks, the best model reaches only 61.9% accuracy, just below the human majority-vote baseline of 65.7% (average individual: 60.2%), indicating that low-level perceptual understanding remains a major weakness of current VLMs. Our analysis further reveals weak and non-monotonic scaling with model size, performance drops in most base--thinking pairs, and distinct severity-response patterns across model families. We hope DistortBench will serve as a useful benchmark for measuring and improving low-level visual perception in VLMs.
>
---
#### [new 061] SurgCoT: Advancing Spatiotemporal Reasoning in Surgical Videos through a Chain-of-Thought Benchmark
- **分类: cs.CV**

- **简介: 该论文属于医学视频分析任务，旨在解决多模态大语言模型在手术视频中细粒度时空推理能力不足的问题。提出SurgCoT基准，评估五种核心推理维度，以提升模型的临床推理能力。**

- **链接: [https://arxiv.org/pdf/2604.20319](https://arxiv.org/pdf/2604.20319)**

> **作者:** Gui Wang; YongSong Zhou; Kaijun Deng; Wooi Ping Cheah; Rong Qu; Jianfeng Ren; Linlin Shen
>
> **备注:** Accept by CVPR2026
>
> **摘要:** Fine-grained spatiotemporal reasoning on surgical videos is critical, yet the capabilities of Multi-modal Large Language Models (MLLMs) in this domain remain largely unexplored. To bridge this gap, we introduce SurgCoT, a unified benchmark for evaluating chain-of-thought (CoT) reasoning in MLLMs across 7 surgical specialties and 35 diverse procedures. SurgCoT assesses five core reasoning dimensions: Causal Action Ordering, Cue-Action Alignment, Affordance Mapping, Micro-Transition Localization, and Anomaly Onset Tracking, through a structured CoT framework with an intensive annotation protocol (Question-Option-Knowledge-Clue-Answer), where the Knowledge field provides essential background context and Clue provides definitive spatiotemporal evidence. Evaluation of 10 leading MLLMs shows: 1) commercial models outperform open-source and medical-specialized variants; 2) significant gaps exist in surgical CoT reasoning; 3) SurgCoT enables effective evaluation and enhances progressive spatiotemporal reasoning. SurgCoT provides a reproducible testbed to narrow the gap between MLLM capabilities and clinical reasoning demands. Code: this https URL.
>
---
#### [new 062] Camera Control for Text-to-Image Generation via Learning Viewpoint Tokens
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决自然语言中相机控制不精确的问题。通过学习视角标记，实现更准确的全局场景控制。**

- **链接: [https://arxiv.org/pdf/2604.19954](https://arxiv.org/pdf/2604.19954)**

> **作者:** Xinxuan Lu; Charless Fowlkes; Alexander C. Berg
>
> **摘要:** Current text-to-image models struggle to provide precise camera control using natural language alone. In this work, we present a framework for precise camera control with global scene understanding in text-to-image generation by learning parametric camera tokens. We fine-tune image generation models for viewpoint-conditioned text-to-image generation on a curated dataset that combines 3D-rendered images for geometric supervision and photorealistic augmentations for appearance and background diversity. Qualitative and quantitative experiments demonstrate that our method achieves state-of-the-art accuracy while preserving image quality and prompt fidelity. Unlike prior methods that overfit to object-specific appearance correlations, our viewpoint tokens learn factorized geometric representations that transfer to unseen object categories. Our work shows that text-vision latent spaces can be endowed with explicit 3D camera structure, offering a pathway toward geometrically-aware prompts for text-to-image generation. Project page: this https URL
>
---
#### [new 063] Normalizing Flows with Iterative Denoising
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生成模型任务，旨在提升Normalizing Flows的性能。通过引入迭代TARFlow（iTARFlow），解决传统方法在图像建模中的不足，实现更高质量的生成效果。**

- **链接: [https://arxiv.org/pdf/2604.20041](https://arxiv.org/pdf/2604.20041)**

> **作者:** Tianrong Chen; Jiatao Gu; David Berthelot; Joshua Susskind; Shuangfei Zhai
>
> **摘要:** Normalizing Flows (NFs) are a classical family of likelihood-based methods that have received revived attention. Recent efforts such as TARFlow have shown that NFs are capable of achieving promising performance on image modeling tasks, making them viable alternatives to other methods such as diffusion models. In this work, we further advance the state of Normalizing Flow generative models by introducing iterative TARFlow (iTARFlow). Unlike diffusion models, iTARFlow maintains a fully end-to-end, likelihood-based objective during training. During sampling, it performs autoregressive generation followed by an iterative denoising procedure inspired by diffusion-style methods. Through extensive experiments, we show that iTARFlow achieves competitive performance across ImageNet resolutions of 64, 128, and 256 pixels, demonstrating its potential as a strong generative model and advancing the frontier of Normalizing Flows. In addition, we analyze the characteristic artifacts produced by iTARFlow, offering insights that may shed light on future improvements. Code is available at this https URL.
>
---
#### [new 064] Online CS-based SAR Edge-Mapping
- **分类: cs.CV**

- **简介: 该论文属于SAR目标识别任务，旨在解决传统ATR算法计算量大、存储负担重的问题。提出在线边缘映射方法，直接分类场景，减少计算和存储需求。**

- **链接: [https://arxiv.org/pdf/2604.19989](https://arxiv.org/pdf/2604.19989)**

> **作者:** Conor Flynn; Radoslav Ivanov; Birsen Yazici
>
> **备注:** SPIE Defense and Commercial Sensing 2026, Algorithms for Synthetic Aperture Radar Imagery XXXIII
>
> **摘要:** With modern defense applications increasingly relying on inexpensive, small Unmanned Aerial Vehicles (UAVs), a major challenge lies in designing intelligent and computationally efficient onboard Automatic Target Recognition (ATR) algorithms to carry out operational objectives. This is especially critical in Synthetic Aperture Radar (SAR), where processing techniques such as ATR are often carried out post data collection, requiring onboard systems to bear the memory burden of storing the back-scattered signals. To alleviate this high cost, we propose an online, direct, edge-mapping technique which bypasses the image reconstruction step to classify scenes and targets. Furthermore, by reconstructing the scene as an edge-map we inherently promote sparsity, requiring fewer measurements and computational power than classic SAR reconstruction algorithms such as backprojection.
>
---
#### [new 065] X-Cache: Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference
- **分类: cs.CV**

- **简介: 该论文提出X-Cache，用于加速少步自回归世界模型的推理。针对实时模拟中的高推理成本问题，通过跨块缓存提升效率，实现71%的块跳过率和2.6倍速度提升。**

- **链接: [https://arxiv.org/pdf/2604.20289](https://arxiv.org/pdf/2604.20289)**

> **作者:** Yixiao Zeng; Jianlei Zheng; Chaoda Zheng; Shijia Chen; Mingdian Liu; Tongping Liu; Tengwei Luo; Yu Zhang; Boyang Wang; Linkun Xu; Siyuan Lu; Bo Tian; Xianming Liu
>
> **备注:** Technical Report
>
> **摘要:** Real-time world simulation is becoming a key infrastructure for scalable evaluation and online reinforcement learning of autonomous driving systems. Recent driving world models built on autoregressive video diffusion achieve high-fidelity, controllable multi-camera generation, but their inference cost remains a bottleneck for interactive deployment. However, existing diffusion caching methods are designed for offline video generation with multiple denoising steps, and do not transfer to this scenario. Few-step distilled models have no inter-step redundancy left for these methods to reuse, and sequence-level parallelization techniques require future conditioning that closed-loop interactive generation does not provide. We present X-Cache, a training-free acceleration method that caches along a different axis: across consecutive generation chunks rather than across denoising steps. X-Cache maintains per-block residual caches that persist across chunks, and applies a dual-metric gating mechanism over a structure- and action-aware block-input fingerprint to independently decide whether each block should recompute or reuse its cached residual. To prevent approximation errors from permanently contaminating the autoregressive KV cache, X-Cache identifies KV update chunks (the forward passes that write clean keys and values into the persistent cache) and unconditionally forces full computation on these chunks, cutting off error propagation. We implement X-Cache on X-world, a production multi-camera action-conditioned driving world model built on multi-block causal DiT with few-step denoising and rolling KV cache. X-Cache achieves 71% block skip rate with 2.6x wall-clock speedup while maintaining minimum degradation.
>
---
#### [new 066] Improving Facial Emotion Recognition through Dataset Merging and Balanced Training Strategies
- **分类: cs.CV**

- **简介: 该论文属于面部情绪识别任务，旨在解决数据不平衡问题。通过合并数据集并采用增强和加权采样策略，提升模型性能，实现82%的识别准确率。**

- **链接: [https://arxiv.org/pdf/2604.20307](https://arxiv.org/pdf/2604.20307)**

> **作者:** Serap Kırbız
>
> **摘要:** In this paper, a deep learning framework is proposed for automatic facial emotion based on deep convolutional networks. In order to increase the generalization ability and the robustness of the method, the dataset size is increased by merging three publicly available facial emotion datasets: CK+, FER+ and KDEF. Despite the increase in dataset size, the minority classes still suffer from insufficient number of training samples, leading to data imbalance. The data imbalance problem is minimized by online and offline augmentation techniques and random weighted sampling. Experimental results demonstrate that the proposed method can recognize the seven basic emotions with 82% accuracy. The results demonstrate the effectiveness of the proposed approach in tackling the challenges of data imbalance and improving classification performance in facial emotion recognition.
>
---
#### [new 067] RareSpot+: A Benchmark, Model, and Active Learning Framework for Small and Rare Wildlife in Aerial Imagery
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决小而稀有野生动物在航拍图像中的检测难题。提出RareSpot+框架，结合多尺度一致性学习和主动学习，提升检测效果并降低标注成本。**

- **链接: [https://arxiv.org/pdf/2604.20000](https://arxiv.org/pdf/2604.20000)**

> **作者:** Bowen Zhang; Jesse T. Boulerice; Charvi Mendiratta; Nikhil Kuniyil; Satish Kumar; Hila Shamon; B. S. Manjunath
>
> **摘要:** Automated wildlife monitoring from aerial imagery is vital for conservation but remains limited by two persistent challenges: the difficulty of detecting small, rare species and the high cost of large-scale expert annotation. Prairie dogs exemplify this problem -- they are ecologically important yet appear tiny, sparsely distributed, and visually indistinct from their surroundings, posing a severe challenge for conventional detection models. To overcome these limitations, we present RareSpot+, a detection framework that integrates multi-scale consistency learning, context-aware augmentation, and geospatially guided active learning to address these issues. A novel multi-scale consistency loss aligns intermediate feature maps across detection heads, enhancing localization of small (approx. 30 pixels wide) objects without architectural changes, while context-aware augmentation improves robustness by synthesizing hard, ecologically plausible examples. A geospatial active learning module exploits domain-specific spatial priors linking prairie dogs and burrows, together with test-time augmentation and a meta-uncertainty model, to reduce redundant labeling. On a 2 km^2 aerial dataset, RareSpot+ improves detection over the baseline mAP@50 by +35.2% (absolute +0.13). Cross-dataset tests on HerdNet, AED, and several other wildlife benchmarks demonstrate robust detector-level transferability. The active learning module further boosts prairie dog AP by 14.5% using an annotation budget of just 1.7% of the unlabeled tiles. Beyond detection, RareSpot+ enables spatial ecological analyses such as clustering and co-occurrence, linking vision-based detection with quantitative ecology.
>
---
#### [new 068] SpaCeFormer: Fast Proposal-Free Open-Vocabulary 3D Instance Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SpaCeFormer，解决开放词汇3D实例分割任务，通过无提议方法实现快速准确分割。**

- **链接: [https://arxiv.org/pdf/2604.20395](https://arxiv.org/pdf/2604.20395)**

> **作者:** Chris Choy; Junha Lee; Chunghyun Park; Minsu Cho; Jan Kautz
>
> **备注:** Project page: this https URL
>
> **摘要:** Open-vocabulary 3D instance segmentation is a core capability for robotics and AR/VR, but prior methods trade one bottleneck for another: multi-stage 2D+3D pipelines aggregate foundation-model outputs at hundreds of seconds per scene, while pseudo-labeled end-to-end approaches rely on fragmented masks and external region proposals. We present SpaCeFormer, a proposal-free space-curve transformer that runs at 0.14 seconds per scene, 2-3 orders of magnitude faster than multi-stage 2D+3D pipelines. We pair it with SpaCeFormer-3M, the largest open-vocabulary 3D instance segmentation dataset (3.0M multi-view-consistent captions over 604K instances from 7.4K scenes) built through multi-view mask clustering and multi-view VLM captioning; it reaches 21x higher mask recall than prior single-view pipelines (54.3% vs 2.5% at IoU > 0.5). SpaCeFormer combines spatial window attention with Morton-curve serialization for spatially coherent features, and uses a RoPE-enhanced decoder to predict instance masks directly from learned queries without external proposals. On ScanNet200 we achieve 11.1 zero-shot mAP, a 2.8x improvement over the prior best proposal-free method; on ScanNet++ and Replica, we reach 22.9 and 24.1 mAP, surpassing all prior methods including those using multi-view 2D inputs.
>
---
#### [new 069] Physics-Informed Conditional Diffusion for Motion-Robust Retinal Temporal Laser Speckle Contrast Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决retinal LSCI在运动干扰下因帧数少导致的重建不稳定问题。提出RetinaDiff框架，结合物理先验与扩散模型，提升重建效果。**

- **链接: [https://arxiv.org/pdf/2604.20594](https://arxiv.org/pdf/2604.20594)**

> **作者:** Qian Chen; Yuehao Chen; Qiang Wang; Lei Zhu; Yanye Lu; Qiushi Ren
>
> **摘要:** Retinal laser speckle contrast imaging (LSCI) is a noninvasive optical modality for monitoring retinal blood flow dynamics. However, conventional temporal LSCI (tLSCI) reconstruction relies on sufficiently long speckle sequences to obtain stable temporal statistics, which makes it vulnerable to acquisition disturbances and limits effective temporal resolution. A physically informed reconstruction framework, termed RetinaDiff (Retinal Diffusion Model), is proposed for retinal tLSCI that is robust to motion and requires only a few frames. In RetinaDiff, registration based on phase correlation is first applied to stabilize the raw speckle sequence before contrast computation, reducing interframe misalignment so that fluctuations at each pixel primarily reflect true flow dynamics. This step provides a physics prior corrected for motion and a high quality multiframe tLSCI reference. Next, guided by the physics prior, a conditional diffusion model performs inverse reconstruction by jointly conditioning on the registered speckle sequence and the corrected prior. Experiments on data acquired with a retinal LSCI system developed in house show improved structural continuity and statistical stability compared with direct reconstruction from few frames and representative baselines. The framework also remains effective in a small number of extremely challenging cases, where both the direct 5-frame input and the conventional multiframe reconstruction are severely degraded. Overall, this work provides a practical and physically grounded route for reliable retinal tLSCI reconstruction from extremely limited frames. The source code and model weights will be publicly available at this https URL.
>
---
#### [new 070] Self-supervised pretraining for an iterative image size agnostic vision transformer
- **分类: cs.CV**

- **简介: 该论文属于视觉Transformer的自监督预训练任务，旨在解决模型对图像尺寸依赖及计算效率低的问题。通过迭代处理多尺度块，实现图像尺寸无关的高效预训练。**

- **链接: [https://arxiv.org/pdf/2604.20392](https://arxiv.org/pdf/2604.20392)**

> **作者:** Nedyalko Prisadnikov; Danda Pani Paudel; Yuqian Fu; Luc Van Gool
>
> **摘要:** Vision Transformers (ViTs) dominate self-supervised learning (SSL). While they have proven highly effective for large-scale pretraining, they are computationally inefficient and scale poorly with image size. Consequently, foundational models like DINO are constrained to low-resolution processing. A recent foveal-inspired transformer achieves resolution agnosticism by iteratively processing a fixed-size context of multi-zoom patches. This model demonstrated promising results via supervised learning, utilizing a sequential, recurrent-like process without backpropagation through time. To unlock its potential as a foundational backbone, we introduce a novel sequential-to-global SSL framework based on DINO's self-distillation objective. Supported by an efficient integral-image patch extraction method, our approach enables large-scale pretraining for image-size agnostic vision encoders. We achieve competitive performance on ImageNet-1K and downstream classification tasks, maintaining a constant computational budget regardless of input resolution.
>
---
#### [new 071] GeoRect4D: Geometry-Compatible Generative Rectification for Dynamic Sparse-View 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于动态3D重建任务，解决稀疏视角视频重建中的几何坍塌和时序不一致问题。提出GeoRect4D框架，结合生成细化与3D一致性优化，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.20784](https://arxiv.org/pdf/2604.20784)**

> **作者:** Zhenlong Wu; Zihan Zheng; Xuanxuan Wang; Qianhe Wang; Hua Yang; Xiaoyun Zhang; Qiang Hu; Wenjun Zhang
>
> **摘要:** Reconstructing dynamic 3D scenes from sparse multi-view videos is highly ill-posed, often leading to geometric collapse, trajectory drift, and floating artifacts. Recent attempts introduce generative priors to hallucinate missing content, yet naive integration frequently causes structural drift and temporal inconsistency due to the mismatch between stochastic 2D generation and deterministic 3D geometry. In this paper, we propose GeoRect4D, a novel unified framework for sparse-view dynamic reconstruction that couples explicit 3D consistency with generative refinement via a closed-loop optimization process. Specifically, GeoRect4D introduces a degradation-aware feedback mechanism that incorporates a robust anchor-based dynamic 3DGS substrate with a single-step diffusion rectifier to hallucinate high-fidelity details. This rectifier utilizes a structural locking mechanism and spatiotemporal coordinated attention, effectively preserving physical plausibility while restoring missing content. Furthermore, we present a progressive optimization strategy that employs stochastic geometric purification to eliminate floaters and generative distillation to infuse texture details into the explicit representation. Extensive experiments demonstrate that GeoRect4D achieves state-of-the-art performance in reconstruction fidelity, perceptual quality, and spatiotemporal consistency across multiple datasets.
>
---
#### [new 072] Visual Reasoning through Tool-supervised Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文研究多模态大语言模型的工具使用问题，旨在提升其解决复杂视觉推理任务的能力。提出ToolsRL框架，通过监督强化学习实现有效工具调用。**

- **链接: [https://arxiv.org/pdf/2604.19945](https://arxiv.org/pdf/2604.19945)**

> **作者:** Qihua Dong; Gozde Sahin; Pei Wang; Zhaowei Cai; Robik Shrestha; Hao Yang; Davide Modolo
>
> **备注:** Accepted to CVPR 2026 Findings. 17 pages
>
> **摘要:** In this paper, we investigate the problem of how to effectively master tool-use to solve complex visual reasoning tasks for Multimodal Large Language Models. To achieve that, we propose a novel Tool-supervised Reinforcement Learning (ToolsRL) framework, with direct tool supervision for more effective tool-use learning. We focus on a series of simple, native, and interpretable visual tools, including zoom-in, rotate, flip, and draw point/line, whose tool supervision is easy to collect. A reinforcement learning curriculum is developed, where the first stage is solely optimized by a set of well motivated tool-specific rewards, and the second stage is trained with the accuracy targeted rewards while allowing calling tools. In this way, tool calling capability is mastered before using tools to complete visual reasoning tasks, avoiding the potential optimization conflict among those heterogeneous tasks. Our experiments have shown that the tool-supervised curriculum training is efficient and ToolsRL can achieve strong tool-use capabilities for complex visual reasoning tasks.
>
---
#### [new 073] Where are they looking in the operating room?
- **分类: cs.CV**

- **简介: 该论文将 gaze-following 引入手术室场景，解决临床角色识别、手术阶段划分和团队沟通检测问题，提出新方法并取得优异效果。**

- **链接: [https://arxiv.org/pdf/2604.20574](https://arxiv.org/pdf/2604.20574)**

> **作者:** Keqi Chen; Séraphin Baributsa; Lilien Schewski; Vinkle Srivastav; Didier Mutter; Guido Beldi; Sandra Keller; Nicolas Padoy
>
> **摘要:** Purpose: Gaze-following, the task of inferring where individuals are looking, has been widely studied in computer vision, advancing research in visual attention modeling, social scene understanding, and human-robot interaction. However, gaze-following has never been explored in the operating room (OR), a complex, high-stakes environment where visual attention plays an important role in surgical workflow analysis. In this work, we introduce the concept of gaze-following to the surgical domain, and demonstrate its great potential for understanding clinical roles, surgical phases, and team communications in the OR. Methods: We extend the 4D-OR dataset with gaze-following annotations, and extend the Team-OR dataset with gaze-following and a new team communication activity annotations. Then, we propose novel approaches to address clinical role prediction, surgical phase recognition, and team communication detection using a gaze-following model. For role and phase recognition, we propose a gaze heatmap-based approach that uses gaze predictions solely; for team communication detection, we train a spatial-temporal model in a self-supervised way that encodes gaze-based clip features, and then feed the features into a temporal activity detection model. Results: Experimental results on the 4D-OR and Team-OR datasets demonstrate that our approach achieves state-of-the-art performance on all downstream tasks. Quantitatively, our approach obtains F1 scores of 0.92 for clinical role prediction and 0.95 for surgical phase recognition. Furthermore, it significantly outperforms existing baselines in team communication detection, improving previous best performances by over 30%. Conclusion: We introduce gaze-following in the OR as a novel research direction in surgical data science, highlighting its great potential to advance surgical workflow analysis in computer-assisted interventions.
>
---
#### [new 074] CrackForward: Context-Aware Severity Stage Crack Synthesis for Data Augmentation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决裂纹数据不足的问题。通过生成真实裂纹样本，提升分割模型性能。**

- **链接: [https://arxiv.org/pdf/2604.19941](https://arxiv.org/pdf/2604.19941)**

> **作者:** Nassim Sadallah; Mohand Saïd Allili
>
> **备注:** 6
>
> **摘要:** Reliable crack detection and segmentation are vital for structural health monitoring, yet the scarcity of well-annotated data constitutes a major challenge. To address this limitation, we propose a novel context-aware generative framework designed to synthesize realistic crack growth patterns for data augmentation. Unlike existing methods that primarily manipulate textures or background content, CrackForward explicitly models crack morphology by combining directional crack elongation with learned thickening and branching. Our framework integrates two key innovations: (i) a contextually guided crack expansion module, which uses local directional cues and adaptive random walk to simulate realistic propagation paths; and (ii) a two-stage U-Net-style generator that learns to reproduce spatially varying crack characteristics such as thickness, branching, and growth. Experimental results show that the generated samples preserve target-stage saturation and thickness characteristics and improve the performance of several crack segmentation architectures. These results indicate that structure-aware synthetic crack generation can provide more informative training data than conventional augmentation alone.
>
---
#### [new 075] If you're waiting for a sign... that might not be it! Mitigating Trust Boundary Confusion from Visual Injections on Vision-Language Agentic Systems
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI安全任务，解决视觉注入导致的信任边界混淆问题。通过设计数据集和防御框架，提升视觉语言代理系统对误导性信号的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.19844](https://arxiv.org/pdf/2604.19844)**

> **作者:** Jiamin Chang; Minhui Xue; Ruoxi Sun; Shuchao Pang; Salil S. Kanhere; Hammond Pearce
>
> **摘要:** Recent advances in embodied Vision-Language Agentic Systems (VLAS), powered by large vision-language models (LVLMs), enable AI systems to perceive and reason over real-world scenes. Within this context, environmental signals such as traffic lights are essential in-band signals that can and should influence agent behavior. However, similar signals could also be crafted to operate as misleading visual injections, overriding user intent and posing security risks. This duality creates a fundamental challenge: agents must respond to legitimate environmental cues while remaining robust to misleading ones. We refer to this tension as trust boundary confusion. To study this behavior, we design a dual-intent dataset and evaluation framework, through which we show that current LVLM-based agents fail to reliably balance this trade-off, either ignoring useful signals or following harmful ones. We systematically evaluate 7 LVLM agents across multiple embodied settings under both structure-based and noise-based visual injections. To address these vulnerabilities, we propose a multi-agent defense framework that separates perception from decision-making to dynamically assess the reliability of visual inputs. Our approach significantly reduces misleading behaviors while preserving correct responses and provides robustness guarantees under adversarial perturbations. The code of the evaluation framework and artifacts are made available at this https URL.
>
---
#### [new 076] Amodal SAM: A Unified Amodal Segmentation Framework with Generalization
- **分类: cs.CV**

- **简介: 该论文提出Amodal SAM，解决物体完整形状分割问题，提升模型在新类别和场景中的泛化能力。通过改进模块和学习目标实现高效分割。**

- **链接: [https://arxiv.org/pdf/2604.20748](https://arxiv.org/pdf/2604.20748)**

> **作者:** Bo Zhang; Zhuotao Tian; Xin Tao; Songlin Tang; Jun Yu; Wenjie Pei
>
> **摘要:** Amodal segmentation is a challenging task that aims to predict the complete geometric shape of objects, including their occluded regions. Although existing methods primarily focus on amodal segmentation within the training domain, these approaches often lack the generalization capacity to extend effectively to novel object categories and unseen contexts. This paper introduces Amodal SAM, a unified framework that leverages SAM (Segment Anything Model) for both amodal image and amodal video segmentation. Amodal SAM preserves the powerful generalization ability of SAM while extending its inherent capabilities to the amodal segmentation task. The improvements lie in three aspects: (1) a lightweight Spatial Completion Adapter that enables occluded region reconstruction, (2) a Target-Aware Occlusion Synthesis (TAOS) pipeline that addresses the scarcity of amodal annotations by generating diverse synthetic training data, and (3) novel learning objectives that enforce regional consistency and topological regularization. Extensive experiments demonstrate that Amodal SAM achieves state-of-the-art performance on standard benchmarks, while simultaneously exhibiting robust generalization to novel scenarios. We anticipate that this research will advance the field toward practical amodal segmentation systems capable of operating effectively in unconstrained real-world environments.
>
---
#### [new 077] Fourier Series Coder: A Novel Perspective on Angle Boundary Discontinuity Problem for Oriented Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决角度边界不连续问题。提出FSC编码器，通过傅里叶基实现稳定的角度编码与解码，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.20281](https://arxiv.org/pdf/2604.20281)**

> **作者:** Minghong Wei; Pu Cao; Zhihao Chen; Zhiyuan Zang; Lu Yang; Qing Song
>
> **摘要:** With the rapid advancement of intelligent driving and remote sensing, oriented object detection has gained widespread attention. However, achieving high-precision performance is fundamentally constrained by the Angle Boundary Discontinuity (ABD) and Cyclic Ambiguity (CA) problems, which typically cause significant angle fluctuations near periodic boundaries. Although recent studies propose continuous angle coders to alleviate these issues, our theoretical and empirical analyses reveal that state-of-the-art methods still suffer from substantial cyclic errors. We attribute this instability to the structural noise amplification within their non-orthogonal decoding mechanisms. This mathematical vulnerability significantly exacerbates angular deviations, particularly for square-like objects. To resolve this fundamentally, we propose the Fourier Series Coder (FSC), a lightweight plug-and-play component that establishes a continuous, reversible, and mathematically robust angle encoding-decoding paradigm. By rigorously mapping angles onto a minimal orthogonal Fourier basis and explicitly enforcing a geometric manifold constraint, FSC effectively prevents feature modulus collapse. This structurally stabilized representation ensures highly robust phase unwrapping, intrinsically eliminating the need for heuristic truncations while achieving strict boundary continuity and superior noise immunity. Extensive experiments across three large-scale datasets demonstrate that FSC achieves highly competitive overall performance, yielding substantial improvements in high-precision detection. The code will be available at this https URL.
>
---
#### [new 078] X-PCR: A Benchmark for Cross-modality Progressive Clinical Reasoning in Ophthalmic Diagnosis
- **分类: cs.CV**

- **简介: 该论文提出X-PCR基准，解决多模态临床推理评估问题，通过六阶段推理和跨模态任务评估眼科诊断能力。**

- **链接: [https://arxiv.org/pdf/2604.20350](https://arxiv.org/pdf/2604.20350)**

> **作者:** Gui Wang; Zehao Zhong; YongSong Zhou; Yudong Li; Ende Wu; Wooi Ping Cheah; Rong Qu; Jianfeng Ren; Linlin Shen
>
> **备注:** Accept by CVPR2026
>
> **摘要:** Despite significant progress in Multi-modal Large Language Models (MLLMs), their clinical reasoning capacity for multi-modal diagnosis remains largely unexamined. Current benchmarks, mostly single-modality data, can't evaluate progressive reasoning and cross-modal integration essential for clinical practice. We introduce the Cross-Modality Progressive Clinical Reasoning (X-PCR) benchmark, the first comprehensive evaluation of MLLMs through a complete ophthalmology diagnostic workflow, with two reasoning tasks: 1) a six-stage progressive reasoning chain spanning image quality assessment to clinical decision-making, and 2) a cross-modality reasoning task integrating six imaging modalities. The benchmark comprises 26,415 images and 177,868 expert-verified VQA pairs curated from 51 public datasets, covering 52 ophthalmic diseases. Evaluation of 21 MLLMs reveals critical gaps in progressive reasoning and cross-modal integration. Dataset and code: this https URL.
>
---
#### [new 079] Topology-Aware Skeleton Detection via Lighthouse-Guided Structured Inference
- **分类: cs.CV**

- **简介: 该论文属于目标骨架检测任务，解决骨架结构不连续问题。提出Lighthouse-Skel方法，通过结构推理提升骨架连通性与完整性。**

- **链接: [https://arxiv.org/pdf/2604.20123](https://arxiv.org/pdf/2604.20123)**

> **作者:** Daoyong Fu; Xiang Zhang; Zhaohuan Zhan; Fan Yang; Ke Yang
>
> **摘要:** In natural images, object skeletons are used to represent geometric shapes. However, even slight variations in pose or movement can cause noticeable changes in skeleton structure, increasing the difficulty of detecting the skeleton and often resulting in discontinuous skeletons. Existing methods primarily focus on point-level skeleton point detection and overlook the importance of structural continuity in recovering complete skeletons. To address this issue, we propose Lighthouse-Skel, a topology-aware skeleton detection method via lighthouse-guided structured inference. Specifically, we introduce a dual-branch collaborative detection framework that jointly learns skeleton confidence field and structural anchors, including endpoints and junction points. The spatial distributions learned by the point branch guide the network to focus on topologically vulnerable regions, which improves the accuracy of skeleton detection. Based on the learned skeleton confidence field, we further propose a lighthouse-guided topology completion strategy, which uses detected junction points and breakpoints as lighthouses to reconnect discontinuous skeleton segments along low-cost paths, thereby improving skeleton continuity and structural integrity. Experimental results on four public datasets demonstrate that the proposed method achieves competitive detection accuracy while substantially improving skeleton connectivity and structural integrity.
>
---
#### [new 080] GSCompleter: A Distillation-Free Plugin for Metric-Aware 3D Gaussian Splatting Completion in Seconds
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决稀疏视角下3D高斯点云补全问题。提出GSCompleter，通过生成-注册流程替代传统迭代方法，提升补全效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.20155](https://arxiv.org/pdf/2604.20155)**

> **作者:** Ao Gao; Jingyu Gong; Xin Tan; Zhizhong Zhang; Yuan Xie
>
> **摘要:** While 3D Gaussian Splatting (3DGS) has revolutionized real-time rendering, its performance degrades significantly under sparse-view extrapolation, manifesting as severe geometric voids and artifacts. Existing solutions primarily rely on an iterative "Repair-then-Distill" paradigm, which is inherently unstable and prone to overfitting. In this work, we propose GSCompleter, a distillation-free plugin that shifts scene completion to a stable "Generate-then-Register" workflow. Our approach first synthesizes plausible 2D reference images and explicitly lifts them into metric-scale 3D primitives via a robust Stereo-Anchor mechanism. These primitives are then seamlessly integrated into the global context through a novel Ray-Constrained Registration strategy. This shift to a rapid registration paradigm delivers superior 3DGS completion performance across three distinct benchmarks, enhancing the quality and efficiency of various baselines and achieving new SOTA results.
>
---
#### [new 081] The Expense of Seeing: Attaining Trustworthy Multimodal Reasoning Within the Monolithic Paradigm
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态AI任务，旨在解决视觉语言模型信任度问题。提出新评估协议与指标，揭示模型视觉理解缺陷，挑战现有评估方法。**

- **链接: [https://arxiv.org/pdf/2604.20665](https://arxiv.org/pdf/2604.20665)**

> **作者:** Karan Goyal; Dikshant Kukreja
>
> **摘要:** The rapid proliferation of Vision-Language Models (VLMs) is widely celebrated as the dawn of unified multimodal knowledge discovery but its foundation operates on a dangerous, unquestioned axiom: that current VLMs faithfully synthesise multimodal data. We argue they do not. Instead, a profound crisis of trustworthiness underlies the dominant Vision Encoder-Projector-LLM paradigm. Rather than extracting grounded knowledge from visual inputs, state-of-the-art models frequently exhibit functional blindness, i.e., exploiting strong language priors to bypass severe visual representation bottlenecks. In this work, we challenge the conventional methodology of multimodal evaluation, which relies on data ablation or new dataset creation and therefore fatally conflates dataset biases with architectural incapacity. We propose a radical, information-theoretic departure: the Modality Translation Protocol, designed to quantifiably unmask the Expense of Seeing. By translating semantic payloads rather than ablating them, we formulate three novel metrics -- the Toll (ToS), Curse (CoS), and Fallacy (FoS) of Seeing -- culminating in the Semantic Sufficiency Criterion (SSC). Furthermore, we posit a provocative Divergence Law of Multimodal Scaling, hypothesising that as the underlying language engines scale to unprecedented reasoning capabilities, the mathematical penalty of the visual knowledge bottleneck paradoxically increases. We challenge the KDD community to abandon the illusory pursuit of "multimodal gain". By elevating the SSC from a passive diagnostic constraint to an active architectural blueprint, we provide the rigorous, trustworthy foundation required to force the next generation of AI systems to truly see the data, achieving true multimodal reasoning.
>
---
#### [new 082] MD-Face: MoE-Enhanced Label-Free Disentangled Representation for Interactive Facial Attribute Editing
- **分类: cs.CV**

- **简介: 该论文属于面部属性编辑任务，旨在解决属性纠缠问题。提出MD-Face框架，通过MoE和几何感知损失实现无标签的解耦表示学习，提升编辑效果。**

- **链接: [https://arxiv.org/pdf/2604.20317](https://arxiv.org/pdf/2604.20317)**

> **作者:** Xuan Cui; Yunfei Zhao; Bo Liu; Wei Duan; Xingrong Fan
>
> **摘要:** GAN-based facial attribute editing is widely used in virtual avatars and social media but often suffers from attribute entanglement, where modifying one face attribute unintentionally alters others. While supervised disentangled representation learning can address this, it relies heavily on labeled data, incurring high annotation costs. To address these challenges, we propose MD-Face, a label-free disentangled representation learning framework based on Mixture of Experts (MoE). MD-Face utilizes a MoE backbone with a gating mechanism that dynamically allocates experts, enabling the model to learn semantic vectors with greater independence. To further enhance attribute entanglement, we introduce a geometry-aware loss, which aligns each semantic vector with its corresponding Semantic Boundary Vector (SBV) through a Jacobian-based pushforward method. Experiments with ProGAN and StyleGAN show that MD-Face outperforms unsupervised baselines and competes with supervised ones. Compared to diffusion-based methods, it offers better image quality and lower inference latency, making it ideal for interactive editing.
>
---
#### [new 083] DynamicRad: Content-Adaptive Sparse Attention for Long Video Diffusion
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决长视频扩散模型效率与质量平衡问题。提出DynamicRad，通过自适应稀疏注意力机制提升推理速度并保持高质量。**

- **链接: [https://arxiv.org/pdf/2604.20470](https://arxiv.org/pdf/2604.20470)**

> **作者:** Yongji Long; Shijun Liang; Jintao Li; Yun Li
>
> **摘要:** Leveraging the natural spatiotemporal energy decay in video diffusion offers a path to efficiency, yet relying solely on rigid static masks risks losing critical long-range information in complex dynamics. To address this issue, we propose \textbf{DynamicRad}, a unified sparse-attention paradigm that grounds adaptive selection within a radial locality prior. DynamicRad introduces a \textbf{dual-mode} strategy: \textit{static-ratio} for speed-optimized execution and \textit{dynamic-threshold} for quality-first filtering. To ensure robustness without online search overhead, we integrate an offline Bayesian Optimization (BO) pipeline coupled with a \textbf{semantic motion router}. This lightweight projection module maps prompt embeddings to optimal sparsity regimes with \textbf{minimal runtime overhead}. Unlike online profiling methods, our offline BO optimizes attention reconstruction error (MSE) on a physics-based proxy task, ensuring rapid convergence. Experiments on HunyuanVideo and Wan2.1-14B demonstrate that DynamicRad pushes the efficiency--quality Pareto frontier, achieving \textbf{1.7$\times$--2.5$\times$ inference speedups} with \textbf{over 80\% effective sparsity}. In some long-sequence settings, the dynamic mode even matches or exceeds the dense baseline, while mask-aware LoRA further improves long-horizon coherence. Code is available at this https URL.
>
---
#### [new 084] Random Walk on Point Clouds for Feature Detection
- **分类: cs.CV**

- **简介: 该论文属于点云特征点检测任务，旨在提取能完整描述模型形状的特征点。通过提出DSN和随机游走方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.20474](https://arxiv.org/pdf/2604.20474)**

> **作者:** Yuhe Zhang; Zhikun Tu; Zhi Li; Jian Gao; Bao Guo; Shunli Zhang
>
> **备注:** 20 pages, 11 figures. Published in Information Sciences
>
> **摘要:** The points on the point clouds that can entirely outline the shape of the model are of critical importance, as they serve as the foundation for numerous point cloud processing tasks and are widely utilized in computer graphics and computer-aided design. This study introduces a novel method, RWoDSN, for extracting such feature points, incorporating considerations of sharp-to-smooth transitions, large-to-small scales, and textural-to-detailed features. We approach feature extraction as a two-stage context-dependent analysis problem. In the first stage, we propose a novel neighborhood descriptor, termed the Disk Sampling Neighborhood (DSN), which, unlike traditional spatially and geometrically invariant approaches, preserves a matrix structure while maintaining normal neighborhood relationships. In the second stage, a random walk is performed on the DSN (RWoDSN), yielding a graph-based DSN that simultaneously accounts for the spatial distribution, topological properties, and geometric characteristics of the local surface surrounding each point. This enables the effective extraction of feature points. Experimental results demonstrate that the proposed RWoDSN method achieves a recall of 0.769-22% higher than the current state-of-the-art-alongside a precision of 0.784. Furthermore, it significantly outperforms several traditional and deep-learning techniques across eight evaluation metrics.
>
---
#### [new 085] UniCVR: From Alignment to Reranking for Unified Zero-Shot Composed Visual Retrieval
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出UniCVR，解决跨任务的零样本组合视觉检索问题。融合多模态大语言模型与视觉语言预训练模型，通过两阶段方法提升检索精度。**

- **链接: [https://arxiv.org/pdf/2604.20318](https://arxiv.org/pdf/2604.20318)**

> **作者:** Haokun Wen; Xuemeng Song; Haoyu Zhang; Xiangyu Zhao; Weili Guan; Liqiang Nie
>
> **摘要:** Composed image retrieval, multi-turn composed image retrieval, and composed video retrieval all share a common paradigm: composing the reference visual with modification text to retrieve the desired target. Despite this shared structure, the three tasks have been studied in isolation, with no prior work proposing a unified framework, let alone a zero-shot solution. In this paper, we propose UniCVR, the first unified zero-shot composed visual retrieval framework that jointly addresses all three tasks without any task-specific human-annotated data. UniCVR strategically combines two complementary strengths: Multimodal Large Language Models (MLLMs) for compositional query understanding and Vision-Language Pre-trained (VLP) models for structured visual retrieval. Concretely, UniCVR operates in two stages. In Stage I, we train the MLLM as a compositional query embedder via contrastive learning on a curated multi-source dataset of approximately 3.5M samples, bridging the heterogeneous embedding spaces between the MLLM and the frozen VLP gallery encoder. A cluster-based hard negative sampling strategy is proposed to strengthen contrastive supervision. In Stage II, we introduce an MLLM-guided dual-level reranking mechanism that applies adaptive budgeted subset scoring to a small number of top-ranked candidates, and then exploits the resulting relevance signals through a dual-level re-scoring scheme, producing more accurate final rankings with minimal computational overhead. Extensive experiments across five benchmarks covering all three tasks demonstrate that UniCVR achieves cutting-edge performance, validating its effectiveness and generalizability. Our data and code will be released upon acceptance.
>
---
#### [new 086] R-CoV: Region-Aware Chain-of-Verification for Alleviating Object Hallucinations in LVLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决对象幻觉问题。通过提出R-CoV方法，利用区域验证链减少模型中不存在对象的错误描述。**

- **链接: [https://arxiv.org/pdf/2604.20696](https://arxiv.org/pdf/2604.20696)**

> **作者:** Jiahao Xie; Alessio Tonioni; Nathalie Rauschmayr; Federico Tombari; Bernt Schiele
>
> **摘要:** Large vision-language models (LVLMs) have demonstrated impressive performance in various multimodal understanding and reasoning tasks. However, they still struggle with object hallucinations, i.e., the claim of nonexistent objects in the visual input. To address this challenge, we propose Region-aware Chain-of-Verification (R-CoV), a visual chain-of-verification method to alleviate object hallucinations in LVLMs in a post-hoc manner. Motivated by how humans comprehend intricate visual information -- often focusing on specific image regions or details within a given sample -- we elicit such region-level processing from LVLMs themselves and use it as a chaining cue to detect and alleviate their own object hallucinations. Specifically, our R-CoV consists of six steps: initial response generation, entity extraction, coordinate generation, region description, verification execution, and final response generation. As a simple yet effective method, R-CoV can be seamlessly integrated into various LVLMs in a training-free manner and without relying on external detection models. Extensive experiments on several widely used hallucination benchmarks across multiple LVLMs demonstrate that R-CoV can significantly alleviate object hallucinations in LVLMs. Project page: this https URL.
>
---
#### [new 087] RefAerial: A Benchmark and Approach for Referring Detection in Aerial Images
- **分类: cs.CV**

- **简介: 该论文属于遥感图像指代检测任务，解决现有数据集不适用于航空图像的问题。构建了RefAerial数据集，并提出SCS框架提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.20543](https://arxiv.org/pdf/2604.20543)**

> **作者:** Guyue Hu; Hao Song; Yuxing Tong; Duzhi Yuan; Dengdi Sun; Aihua Zheng; Chenglong Li; Jin Tang
>
> **摘要:** Referring detection refers to locate the target referred by natural languages, which has recently attracted growing research interests. However, existing datasets are limited to ground images with large object centered in relative small scenes. This paper introduces a large-scale challenging dataset for referring detection in aerial images, termed as RefAerial. It distinguishes from conventional ground referring detection datasets by 4 characteristics: (1) low but diverse object-to-scene ratios, (2) numerous targets and distractors, (3)complex and fine-grained referring descriptions, (4) diverse and broad scenes in the aerial view. We also develop a human-in-the-loop referring expansion and annotation engine (REA-Engine) for efficient semi-automated referring pair annotation. Besides, we observe that existing ground referring detection approaches exhibiting serious performance degradation on our aerial dataset since the intrinsic scale variety issue within or across aerial images. Therefore, we further propose a novel scale-comprehensive and sensitive (SCS) framework for referring detection in aerial images. It consists of a mixture-of-granularity (MoG) attention and a two-stage comprehensive-to-sensitive (CtS) decoding strategy. Specifically, the mixture-of-granularity attention is developed for scale-comprehensive target understanding. In addition, the two-stage comprehensive-to-sensitive decoding strategy is designed for coarse-to-fine referring target decoding. Eventually, the proposed SCS framework achieves remarkable performance on our aerial referring detection dataset and even promising performance boost on conventional ground referring detection datasets.
>
---
#### [new 088] SceneOrchestra: Efficient Agentic 3D Scene Synthesis via Full Tool-Call Trajectory Generation
- **分类: cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决传统框架效率低、质量差的问题。提出SceneOrchestra，通过优化工具调用流程提升生成效率和质量。**

- **链接: [https://arxiv.org/pdf/2604.19907](https://arxiv.org/pdf/2604.19907)**

> **作者:** Yun He; Kelin Yu; Matthias Zwicker
>
> **摘要:** Recent agentic frameworks for 3D scene synthesis have advanced realism and diversity by integrating heterogeneous generation and editing tools. These tools are organized into workflows orchestrated by an off-the-shelf LLM. Current approaches typically adopt an execute-review-reflect loop: at each step, the orchestrator executes a tool, renders intermediate results for review, and then decides on the tool and its parameters for the next step. However, this design has two key limitations. First, next-step tool selection and parameter configuration are driven by heuristic rules, which can lead to suboptimal execution flows, unnecessary tool invocations, degraded output quality, and increased runtime. Second, rendering and reviewing intermediate results after each step introduces additional latency. To address these issues, we propose SceneOrchestra, a trainable orchestration framework that optimizes the tool-call execution flow and eliminates the step-by-step review loop, improving both efficiency and output quality. SceneOrchestra consists of an orchestrator and a discriminator, which we fine-tune with a two-phase training strategy. In the first phase, the orchestrator learns context-aware tool selection and complete tool-call trajectory generation, while the discriminator is trained to assess the quality of full trajectories, enabling it to select the best trajectory from multiple candidates. In the second phase, we perform interleaved training, where the discriminator adapts to the orchestrator's evolving trajectory distribution and distills its discriminative capability back into the orchestrator. At inference, we only use the orchestrator to generate and execute full tool-call trajectories from instructions, without requiring the discriminator. Extensive experiments show that our method achieves state-of-the-art scene quality while reducing runtime compared to previous work.
>
---
#### [new 089] FurnSet: Exploiting Repeats for 3D Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决单视角下物体几何与布局的重建问题。通过识别和利用重复物体实例，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.20093](https://arxiv.org/pdf/2604.20093)**

> **作者:** Paul Dobre; Xin Wang; Hongzhou Yang
>
> **摘要:** Single-view 3D scene reconstruction involves inferring both object geometry and spatial layout. Existing methods typically reconstruct objects independently or rely on implicit scene context, failing to exploit the repeated instances commonly present in realworld scenes. We propose FurnSet, a framework that explicitly identifies and leverages repeated object instances to improve reconstruction. Our method introduces per-object CLS tokens and a set-aware self-attention mechanism that groups identical instances and aggregates complementary observations across them, enabling joint reconstruction. We further combine scene-level and object-level conditioning to guide object reconstruction, followed by layout optimization using object point clouds with 3D and 2D projection losses for scene alignment. Experiments on 3D-Future and 3D-Front demonstrate improved scene reconstruction quality, highlighting the effectiveness of exploiting repetition for robust 3D scene reconstruction.
>
---
#### [new 090] On the Impact of Face Segmentation-Based Background Removal on Recognition and Morphing Attack Detection
- **分类: cs.CV**

- **简介: 该论文属于生物特征识别任务，研究背景分割对人脸识别和攻击检测的影响，旨在解决实际场景下背景不可控带来的性能问题，通过实验分析不同分割方法的效果。**

- **链接: [https://arxiv.org/pdf/2604.20585](https://arxiv.org/pdf/2604.20585)**

> **作者:** Eduarda Caldeira; Guray Ozgur; Fadi Boutros; Naser Damer
>
> **备注:** Accepted at FG 2026
>
> **摘要:** This study investigates the impact of face image background correction through segmentation on face recognition and morphing attack detection performance in realistic, unconstrained image capture scenarios. The motivation is driven by operational biometric systems such as the European Entry/Exit System (EES), which require facial enrolment at airports and other border crossing points where controlled backgrounds usually required for such captures cannot always be guaranteed, as well as by accessibility needs that may necessitate image capture outside traditional office environments. By analyzing how such preprocessing steps influence both recognition accuracy and security mechanisms, this work addresses a critical gap between usability-driven image normalization and the reliability requirements of large-scale biometric identification systems. Our study evaluates a comprehensive range of segmentation techniques, three families of morphing attack detection methods, and four distinct face recognition models, using databases that include both controlled and in-the-wild image captures. The results reveal consistent patterns linking segmentation to both recognition performance and face image quality. Additionally, segmentation is shown to systematically influence morphing attack detection performance. These findings highlight the need for careful consideration when deploying such preprocessing techniques in operational biometric systems.
>
---
#### [new 091] IMPACT-CYCLE: A Contract-Based Multi-Agent System for Claim-Level Supervisory Correction of Long-Video Semantic Memory
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出IMPACT-CYCLE系统，解决长视频理解中错误修正成本高的问题。通过多智能体协作维护语义记忆，提升修正效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.20136](https://arxiv.org/pdf/2604.20136)**

> **作者:** Weitong Kong; Di Wen; Kunyu Peng; David Schneider; Zeyun Zhong; Alexander Jaus; Zdravko Marinov; Jiale Wei; Ruiping Liu; Junwei Zheng; Yufan Chen; Lei Qi; Rainer Stiefelhagen
>
> **备注:** 7 pages, 2 figures, code are available at this https URL
>
> **摘要:** Correcting errors in long-video understanding is disproportionately costly: existing multimodal pipelines produce opaque, end-to-end outputs that expose no intermediate state for inspection, forcing annotators to revisit raw video and reconstruct temporal logic from scratch. The core bottleneck is not generation quality alone, but the absence of a supervisory interface through which human effort can be proportional to the scope of each error. We present IMPACT-CYCLE, a supervisory multi-agent system that reformulates long-video understanding as iterative claim-level maintenance of a shared semantic memory -- a structured, versioned state encoding typed claims, a claim dependency graph, and a provenance log. Role-specialized agents operating under explicit authority contracts decompose verification into local object-relation correctness, cross-temporal consistency, and global semantic coherence, with corrections confined to structurally dependent claims. When automated evidence is insufficient, the system escalates to human arbitration as the supervisory authority with final override rights; dependency-closure re-verification then ensures correction cost remains proportional to error scope. Experiments on VidOR show substantially improved downstream reasoning (VQA: 0.71 to 0.79) and a 4.8x reduction in human arbitration cost, with workload significantly lower than manual annotation. Code will be released at this https URL.
>
---
#### [new 092] MAPRPose: Mask-Aware Proposal and Amodal Refinement for Multi-Object 6D Pose Estimation
- **分类: cs.CV**

- **简介: 该论文提出MAPRPose，解决杂乱场景中多目标6D位姿估计问题。通过两阶段框架提升位姿预测的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.20650](https://arxiv.org/pdf/2604.20650)**

> **作者:** Yang Luo; Yan Gong; Yongsheng Gao; Xiaoying Sun; Jie Zhao
>
> **摘要:** 6D object pose estimation in cluttered scenes remains challenging due to severe occlusion and sensor noise. We propose MAPRPose, a two-stage framework that leverages mask-aware correspondences for pose proposal and amodal-driven Region-of-Interest (ROI) prediction for robust refinement. In the Mask-Aware Pose Proposal (MAPP) stage, we lift 2D correspondences into 3D space to establish reliable keypoint matches and generate geometrically consistent pose hypotheses based on correspondence-level scoring, from which the top-$K$ candidates are selected. In the refinement stage, we introduce a tensorized render-and-compare pipeline integrated with an Amodal Mask Prediction and ROI Re-Alignment (AMPR) module. By reconstructing complete object geometry and dynamically adjusting the ROI, AMPR mitigates localization errors and spatial misalignment under heavy occlusion. Furthermore, our GPU-accelerated RGB-XYZ reprojection enables simultaneous refinement of all $N \times B$ pose hypotheses in a single forward pass. Evaluated on the BOP benchmark, MAPRPose achieves a state-of-the-art Average Recall (AR) of 76.5%, outperforming FoundationPose by 3.1% AR while delivering a 43x speedup in multi-object inference.
>
---
#### [new 093] Bio-inspired Color Constancy: From Gray Anchoring Theory to Gray Pixel Methods
- **分类: cs.CV**

- **简介: 该论文属于颜色恒常性任务，旨在解决生物启发方法在颜色恒常性中的应用问题。通过分析灰锚理论和灰像素方法，提出一种基于学习的高效算法。**

- **链接: [https://arxiv.org/pdf/2604.20243](https://arxiv.org/pdf/2604.20243)**

> **作者:** Kai-Fu Yang; Fu-Ya Luo; Yong-Jie Li
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Color constancy is a fundamental ability of many biological visual systems and a crucial step in computer imaging systems. Bio-inspired modeling offers a promising way to elucidate the computational principles underlying color constancy and to develop efficient computational methods. However, bio-inspired methods for color constancy remain underexplored and lack a comprehensive analysis. This paper presents a comprehensive technical framework that integrates biological mechanisms, computational theory, and algorithmic implementation for bio-inspired color constancy. Specifically, we systematically revisit the computational theory of biological color constancy, which shows that illuminant estimation can be reduced to the task of gray-anchor (pixel or surface) detection in early vision. Subsequently, typical gray-pixel detection methods, including Gray-Pixel and Grayness-Index, are reinterpreted within a unified theoretical framework with the Lambertian reflection model and biological color-opponent mechanisms. Finally, we propose a simple learning-based method that couples reflection-model constraints with feature learning to explore the potential of bio-inspired color constancy based on gray-pixel detection. Extensive experiments confirm the effectiveness of gray-pixel detection for color constancy and demonstrate the potential of bio-inspired methods.
>
---
#### [new 094] TactileEval: A Step Towards Automated Fine-Grained Evaluation and Editing of Tactile Graphics
- **分类: cs.CV**

- **简介: 该论文提出TactileEval，用于自动化评估和编辑触觉图形。解决BVI学习者触觉图形质量评估问题，通过建立质量分类体系并实现自动化修复。**

- **链接: [https://arxiv.org/pdf/2604.19829](https://arxiv.org/pdf/2604.19829)**

> **作者:** Adnan Khan; Abbas Akkasi; Majid Komeili
>
> **备注:** Code, data, and models are available at this https URL
>
> **摘要:** Tactile graphics require careful expert validation before reaching blind and visually impaired (BVI) learners, yet existing datasets provide only coarse holistic quality ratings that offer no actionable repair signal. We present TactileEval, a three-stage pipeline that takes a first step toward automating this process. Drawing on expert free-text comments from the TactileNet dataset, we establish a five-category quality taxonomy; encompassing view angle, part completeness, background clutter, texture separation, and line quality aligned with BANA standards. We subsequently gathered 14,095 structured annotations via Amazon Mechanical Turk, spanning 66 object classes organized into six distinct families. A reproducible ViT-L/14 feature probe trained on this data achieves 85.70% overall test accuracy across 30 different tasks, with consistent difficulty ordering suggesting the taxonomy suggesting the taxonomy captures meaningful perceptual structure. Building on these evaluations, we present a ViT-guided automated editing pipeline that routes classifier scores through family-specific prompt templates to produce targeted corrections via gpt-image-1 image editing. Code, data, and models are available at this https URL
>
---
#### [new 095] CHASM: Unveiling Covert Advertisements on Chinese Social Media
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.CY**

- **简介: 该论文属于社会媒体内容检测任务，旨在解决 covert advertisements 的识别问题。作者构建了 CHASM 数据集，并评估 MLLMs 的检测能力，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2604.20511](https://arxiv.org/pdf/2604.20511)**

> **作者:** Jingyi Zheng; Tianyi Hu; Yule Liu; Zhen Sun; Zongmin Zhang; Zifan Peng; Wenhan Dong; Xinlei He
>
> **备注:** NeuIPS 2025 (Datasets and Benchmarks Track)
>
> **摘要:** Current benchmarks for evaluating large language models (LLMs) in social media moderation completely overlook a serious threat: covert advertisements, which disguise themselves as regular posts to deceive and mislead consumers into making purchases, leading to significant ethical and legal concerns. In this paper, we present the CHASM, a first-of-its-kind dataset designed to evaluate the capability of Multimodal Large Language Models (MLLMs) in detecting covert advertisements on social media. CHASM is a high-quality, anonymized, manually curated dataset consisting of 4,992 instances, based on real-world scenarios from the Chinese social media platform Rednote. The dataset was collected and annotated under strict privacy protection and quality control protocols. It includes many product experience sharing posts that closely resemble covert advertisements, making the dataset particularly this http URL results show that under both zero-shot and in-context learning settings, none of the current MLLMs are sufficiently reliable for detecting covert this http URL further experiments revealed that fine-tuning open-source MLLMs on our dataset yielded noticeable performance gains. However, significant challenges persist, such as detecting subtle cues in comments and differences in visual and textual this http URL provide in-depth error analysis and outline future research directions. We hope our study can serve as a call for the research community and platform moderators to develop more precise defenses against this emerging threat.
>
---
#### [new 096] FedSIR: Spectral Client Identification and Relabeling for Federated Learning with Noisy Labels
- **分类: cs.LG; cs.AI; cs.CV; cs.DC; eess.SP**

- **简介: 该论文属于联邦学习任务，旨在解决噪声标签影响模型性能的问题。通过谱结构分析识别并修正噪声，提升联邦学习鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.20825](https://arxiv.org/pdf/2604.20825)**

> **作者:** Sina Gholami; Abdulmoneam Ali; Tania Haghighi; Ahmed Arafa; Minhaj Nur Alam
>
> **备注:** Accepted at the 5th Workshop on Federated Learning for Computer Vision (FedVision), CVPR 2026. Sina Gholami and Abdulmoneam Ali contributed equally
>
> **摘要:** Federated learning (FL) enables collaborative model training without sharing raw data; however, the presence of noisy labels across distributed clients can severely degrade the learning performance. In this paper, we propose FedSIR, a multi-stage framework for robust FL under noisy labels. Different from existing approaches that mainly rely on designing noise-tolerant loss functions or exploiting loss dynamics during training, our method leverages the spectral structure of client feature representations to identify and mitigate label noise. Our framework consists of three key components. First, we identify clean and noisy clients by analyzing the spectral consistency of class-wise feature subspaces with minimal communication overhead. Second, clean clients provide spectral references that enable noisy clients to relabel potentially corrupted samples using both dominant class directions and residual subspaces. Third, we employ a noise-aware training strategy that integrates logit-adjusted loss, knowledge distillation, and distance-aware aggregation to further stabilize federated optimization. Extensive experiments on standard FL benchmarks demonstrate that FedSIR consistently outperforms state-of-the-art methods for FL with noisy labels. The code is available at this https URL.
>
---
#### [new 097] From Image to Music Language: A Two-Stage Structure Decoding Approach for Complex Polyphonic OMR
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于光学音乐识别任务，解决复杂钢琴乐谱的结构解码问题。通过拓扑识别与概率搜索方法，将符号和事件解码为结构化乐谱。**

- **链接: [https://arxiv.org/pdf/2604.20522](https://arxiv.org/pdf/2604.20522)**

> **作者:** Nan Xu; Shiheng Li; Shengchao Hou
>
> **备注:** 49 pages, 16 figures, 16 tables
>
> **摘要:** We propose a new approach for the second stage of a practical two-stage Optical Music Recognition (OMR) pipeline. Given symbol and event candidates from the visual pipeline, we decode them into an editable, verifiable, and exportable score structure. We focus on complex polyphonic staff notation, especially piano scores, where voice separation and intra-measure timing are the main bottlenecks. Our approach formulates second-stage decoding as a structure decoding problem and uses topology recognition with probability-guided search (BeadSolver) as its core method. We also describe a data strategy that combines procedural generation with recognition-feedback annotations. The result is a practical decoding component for real OMR systems and a path to accumulate structured score data for future end-to-end, multimodal, and RL-style methods.
>
---
#### [new 098] Pairing Regularization for Mitigating Many-to-One Collapse in GANs
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成对抗网络（GAN）训练任务，旨在解决内在模式崩溃问题。通过引入配对正则化，增强潜在变量与生成样本的一致性，提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2604.20130](https://arxiv.org/pdf/2604.20130)**

> **作者:** Kuan-Yu Lin; Yu-Chih Huang; Tie Liu
>
> **摘要:** Mode collapse remains a fundamental challenge in training generative adversarial networks (GANs). While existing works have primarily focused on inter-mode collapse, such as mode dropping, intra-mode collapse-where many latent variables map to the same or highly similar outputs-has received significantly less attention. In this work, we propose a pairing regularizer jointly optimized with the generator to mitigate the many-to-one collapse by enforcing local consistency between latent variables and generated samples. We show that the effect of pairing regularization depends on the dominant failure mode of training. In collapse-prone regimes with limited exploration, pairing encourages structured local exploration, leading to improved coverage and higher recall. In contrast, under stabilized training with sufficient exploration, pairing refines the generator's induced data density by discouraging redundant mappings, thereby improving precision without sacrificing recall. Extensive experiments on both toy distributions and real-image benchmarks demonstrate that the proposed regularizer effectively complements existing stabilization techniques by directly addressing intra-mode collapse.
>
---
#### [new 099] Maximum Likelihood Reconstruction for Multi-Look Digital Holography with Markov-Modeled Speckle Correlation
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于数字全息重建任务，旨在解决多视角测量中因硬件限制导致的散斑相关问题。通过建立一阶马尔可夫模型，提出一种最大似然估计方法，提升重建性能。**

- **链接: [https://arxiv.org/pdf/2604.20154](https://arxiv.org/pdf/2604.20154)**

> **作者:** Xi Chen; Arian Maleki; Shirin Jalali
>
> **摘要:** Multi-look acquisition is a widely used strategy for reducing speckle noise in coherent imaging systems such as digital holography. By acquiring multiple measurements, speckle can be suppressed through averaging or joint reconstruction, typically under the assumption that speckle realizations across looks are statistically independent. In practice, however, hardware constraints limit measurement diversity, leading to inter-look correlation that degrades the performance of conventional methods. In this work, we study the reconstruction of speckle-free reflectivity from complex-valued multi-look measurements in the presence of correlated speckle. We model the inter-look dependence using a first-order Markov process and derive the corresponding likelihood under a first-order Markov approximation, resulting in a constrained maximum likelihood estimation problem. To solve this problem, we develop an efficient projected gradient descent framework that combines gradient-based updates with implicit regularization via deep image priors, and leverages Monte Carlo approximation and matrix-free operators for scalable computation. Simulation results demonstrate that the proposed approach remains robust under strong inter-look correlation, achieving performance close to the ideal independent-look scenario and consistently outperforming methods that ignore such dependencies. These results highlight the importance of explicitly modeling inter-look correlation and provide a practical framework for multi-look holographic reconstruction under realistic acquisition conditions. Our code is available at: this https URL.
>
---
#### [new 100] Energy-Based Open-Set Active Learning for Object Classification
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于目标分类任务，解决开放集主动学习中的样本选择问题。提出一种基于能量的双阶段框架，区分已知与未知类别，提升标注效率和分类性能。**

- **链接: [https://arxiv.org/pdf/2604.20083](https://arxiv.org/pdf/2604.20083)**

> **作者:** Zongyao Lyu; William J. Beksi
>
> **备注:** To be published in the 2026 International Conference on Pattern Recognition (ICPR)
>
> **摘要:** Active learning (AL) has emerged as a crucial methodology for minimizing labeling costs in deep learning by selecting the most valuable samples from a pool of unlabeled data for annotation. Traditional AL operates under a closed-set assumption, where all classes in the dataset are known and consistent. However, real-world scenarios often present open-set conditions in which unlabeled data contains both known and unknown classes. In such environments, standard AL techniques struggle. They can mistakenly query samples from unknown categories, leading to inefficient use of annotation budgets. In this paper, we propose a novel dual-stage energy-based framework for open-set AL. Our method employs two specialized energy-based models (EBMs). The first, an energy-based known/unknown separator, filters out samples likely to belong to unknown classes. The second, an energy-based sample scorer, assesses the informativeness of the filtered known samples. Using the energy landscape, our models distinguish between data points from known and unknown classes in the unlabeled pool by assigning lower energy to known samples and higher energy to unknown samples, ensuring that only samples from classes of interest are selected for labeling. By integrating these components, our approach ensures efficient and targeted sample selection, maximizing learning impact in each iteration. Experiments on 2D (CIFAR-10, CIFAR-100, TinyImageNet) and 3D (ModelNet40) object classification benchmarks demonstrates that our framework outperforms existing approaches, achieving superior annotation efficiency and classification performance in open-set environments.
>
---
#### [new 101] Hybrid Multi-Phase Page Matching and Multi-Layer Diff Detection for Japanese Building Permit Document Review
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于文档比对任务，解决日本建筑许可文件人工比对效率低、易出错的问题。提出混合多阶段页面匹配算法与多层差异检测引擎，实现高效准确的文档对比。**

- **链接: [https://arxiv.org/pdf/2604.19770](https://arxiv.org/pdf/2604.19770)**

> **作者:** Mitsumasa Wada
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** We present a hybrid multi-phase page matching algorithm for automated comparison of Japanese building permit document sets. Building permit review in Japan requires cross-referencing large PDF document sets across revision cycles, a process that is labor-intensive and error-prone when performed manually. The algorithm combines longest common subsequence (LCS) structural alignment, a seven-phase consensus matching pipeline, and a dynamic programming optimal alignment stage to robustly pair pages across revisions even when page order, numbering, or content changes substantially. A subsequent multi-layer diff engine -- comprising text-level, table-level, and pixel-level visual differencing -- produces highlighted difference reports. Evaluation on real-world permit document sets achieves F1=0.80 and precision=1.00 on a manually annotated ground-truth benchmark, with zero false-positive matched pairs.
>
---
#### [new 102] Fast Amortized Fitting of Scientific Signals Across Time and Ensembles via Transferable Neural Fields
- **分类: cs.LG; cs.CE; cs.CV**

- **简介: 该论文属于科学信号建模任务，旨在解决INR在高维科学场景中收敛慢、扩展性差的问题。通过迁移学习提升信号重建效率与精度。**

- **链接: [https://arxiv.org/pdf/2604.19979](https://arxiv.org/pdf/2604.19979)**

> **作者:** Sophia Zorek; Kushal Vyas; Yuhao Liu; David Lenz; Tom Peterka; Guha Balakrishnan
>
> **摘要:** Neural fields, also known as implicit neural representations (INRs), offer a powerful framework for modeling continuous geometry, but their effectiveness in high-dimensional scientific settings is limited by slow convergence and scaling challenges. In this study, we extend INR models to handle spatiotemporal and multivariate signals and show how INR features can be transferred across scientific signals to enable efficient and scalable representation across time and ensemble runs in an amortized fashion. Across controlled transformation regimes (e.g., geometric transformations and localized perturbations of synthetic fields) and high-fidelity scientific domains-including turbulent flows, fluid-material impact dynamics, and astrophysical systems-we show that transferable features improve not only signal fidelity but also the accuracy of derived geometric and physical quantities, including density gradients and vorticity. In particular, transferable features reduce iterations to reach target reconstruction quality by up to an order of magnitude, increase early-stage reconstruction quality by multiple dB (with gains exceeding 10 dB in some cases), and consistently improve gradient-based physical accuracy.
>
---
#### [new 103] ParetoSlider: Diffusion Models Post-Training for Continuous Reward Control
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于多目标强化学习任务，旨在解决生成模型在多个冲突目标间缺乏灵活控制的问题。通过训练模型逼近帕累托前沿，实现推理时的连续奖励控制。**

- **链接: [https://arxiv.org/pdf/2604.20816](https://arxiv.org/pdf/2604.20816)**

> **作者:** Shelly Golan; Michael Finkelson; Ariel Bereslavsky; Yotam Nitzan; Or Patashnik
>
> **备注:** Project page: this https URL
>
> **摘要:** Reinforcement Learning (RL) post-training has become the standard for aligning generative models with human preferences, yet most methods rely on a single scalar reward. When multiple criteria matter, the prevailing practice of ``early scalarization'' collapses rewards into a fixed weighted sum. This commits the model to a single trade-off point at training time, providing no inference-time control over inherently conflicting goals -- such as prompt adherence versus source fidelity in image editing. We introduce ParetoSlider, a multi-objective RL (MORL) framework that trains a single diffusion model to approximate the entire Pareto front. By training the model with continuously varying preference weights as a conditioning signal, we enable users to navigate optimal trade-offs at inference time without retraining or maintaining multiple checkpoints. We evaluate ParetoSlider across three state-of-the-art flow-matching backbones: SD3.5, FluxKontext, and LTX-2. Our single preference-conditioned model matches or exceeds the performance of baselines trained separately for fixed reward trade-offs, while uniquely providing fine-grained control over competing generative goals.
>
---
#### [new 104] Lifecycle-Aware Federated Continual Learning in Mobile Autonomous Systems
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于联邦持续学习任务，解决分布式自主系统在长期任务中遗忘和模型退化问题。提出双时间尺度框架，结合预防遗忘与恢复机制，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.20745](https://arxiv.org/pdf/2604.20745)**

> **作者:** Beining Wu; Jun Huang
>
> **备注:** Submitted to IEEE
>
> **摘要:** Federated continual learning (FCL) allows distributed autonomous fleets to adapt collaboratively to evolving terrain types across extended mission lifecycles. However, current approaches face several key challenges: 1) they use uniform protection strategies that do not account for the varying sensitivities to forgetting on different network layers; 2) they focus primarily on preventing forgetting during training, without addressing the long-term effects of cumulative drift; and 3) they often depend on idealized simulations that fail to capture the real-world heterogeneity present in distributed fleets. In this paper, we propose a lifecycle-aware dual-timescale FCL framework that incorporates training-time (pre-forgetting) prevention and (post-forgetting) recovery. Under this framework, we design a layer-selective rehearsal strategy that mitigates immediate forgetting during local training, and a rapid knowledge recovery strategy that restores degraded models after long-term cumulative drift. We present a theoretical analysis that characterizes heterogeneous forgetting dynamics and establishes the inevitability of long-term degradation. Our experimental results show that this framework achieves up to 8.3\% mIoU improvement over the strongest federated baseline and up to 31.7\% over conventional fine-tuning. We also deploy the FCL framework on a real-world rover testbed to assess system-level robustness under realistic constraints; the testing results further confirm the effectiveness of our FCL design.
>
---
#### [new 105] Secure Rate-Distortion-Perception: A Randomized Distributed Function Computation Approach for Realism
- **分类: cs.IT; cs.CR; cs.CV; eess.IV**

- **简介: 该论文研究安全率失真感知问题，解决在公开信道传输时保持感知质量与信息安全的矛盾。通过随机分布式函数计算方法，分析不同信道下的安全率失真区域，并验证共同随机性对通信速率的影响。**

- **链接: [https://arxiv.org/pdf/2604.20245](https://arxiv.org/pdf/2604.20245)**

> **作者:** Gustaf Åhlgren; Onur Günlü
>
> **备注:** 20 pages, 6 figures, (submitted) journal version
>
> **摘要:** Fundamental rate-distortion-perception (RDP) trade-offs arise in applications requiring maintained perceptual quality of reconstructed data, such as neural image compression. When compressed data is transmitted over public communication channels, security risks emerge. We therefore study secure RDP under negligible information leakage over both noiseless channels and broadcast channels, BCs, with correlated noise components. For noiseless channels, the exact secure RDP region is characterized. For BCs, an inner bound is derived and shown to be tight for a class of more-capable BCs. Separate source-channel coding is further shown to be optimal for this exact secure RDP region with unlimited common randomness available. Moreover, when both encoder and decoder have access to side information correlated with the source and the channel is noiseless, the exact RDP region is established. If only the decoder has correlated side information in the noiseless setting, an inner bound is derived along with a special case where the region is exact. Binary and Gaussian examples demonstrate that common randomness can significantly reduce the communication rate in secure RDP settings, unlike in standard rate-distortion settings. Thus, our results illustrate that random binning-based coding achieves strong secrecy, low distortion, and high perceptual quality simultaneously.
>
---
#### [new 106] Diagnosing Urban Street Vitality via a Visual-Semantic and Spatiotemporal Framework for Street-Level Economics
- **分类: cs.CY; cs.CV; econ.EM**

- **简介: 该论文属于城市街道经济评估任务，旨在解决传统方法语义浅层和忽视品牌层次与结构衰退的问题。通过构建视觉-语义时空框架SEVI，融合街景与位置数据，实现街道活力的精准诊断。**

- **链接: [https://arxiv.org/pdf/2604.19798](https://arxiv.org/pdf/2604.19798)**

> **作者:** Xinxin Zhuo; Mengyuan Niu; Ruizhe Wang; Junyan Yang; Qiao Wang
>
> **备注:** Submitted to ACM Transactions on Spatial Computing. This paper is currently under review
>
> **摘要:** Micro-scale street-level economic assessment is fundamental for precision spatial resource allocation. While Street View Imagery (SVI) advances urban sensing, existing approaches remain semantically superficial and overlook brand hierarchy heterogeneity and structural recession. To address this, we propose a visual-semantic and field-based spatiotemporal framework, operationalized via the Street Economic Vitality Index (SEVI). Our approach integrates physical and semantic streetscape parsing through instance segmentation of signboards, glass interfaces, and storefront closures. A dual-stage VLM-LLM pipeline standardizes signage into global hierarchies to quantify a spatially smoothed brand premium index. To overcome static SVI limitations, we introduce a temporal lag design using Location-Based Services (LBS) data to capture realized demand. Combined with a category-weighted Gaussian spillover model, we construct a three-dimensional diagnostic system covering Commercial Activity, Spatial Utilization, and Physical Environment. Experiments based on time-lagged geographically weighted regression across eight tidal periods in Nanjing reveal quasi-causal spatiotemporal heterogeneity. Street vibrancy arises from interactions between hierarchical brand clustering and mall-induced externalities. High-quality interfaces show peak attraction during midday and evening, while structural recession produces a lagged nighttime repulsion effect. The framework offers evidence-based support for precision spatial governance.
>
---
## 更新

#### [replaced 001] Towards Reliable Human Evaluations in Gesture Generation: Insights from a Community-Driven State-of-the-Art Benchmark
- **分类: cs.CV; cs.GR; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.01233](https://arxiv.org/pdf/2511.01233)**

> **作者:** Rajmund Nagy; Hendric Voss; Thanh Hoang-Minh; Mihail Tsakov; Teodor Nikolov; Zeyi Zhang; Tenglong Ao; Sicheng Yang; Shaoli Huang; Yongkang Cheng; M. Hamza Mughal; Rishabh Dabral; Kiran Chhatre; Christian Theobalt; Libin Liu; Stefan Kopp; Rachel McDonnell; Michael Neff; Taras Kucherenko; Youngwoo Yoon; Gustav Eje Henter
>
> **备注:** Accepted to CVPR 2026, Findings Track. 23 pages, 10 figures. The last two authors made equal contributions
>
> **摘要:** We review human evaluation practices in automatic, speech-driven 3D gesture generation and find a lack of standardisation and frequent use of flawed experimental setups. This leads to a situation where it is impossible to know how different methods compare, or what the state of the art is. In order to address common shortcomings of evaluation design, and to standardise future user studies in gesture-generation works, we introduce a detailed human evaluation protocol for the widely-used BEAT2 motion-capture dataset. Using this protocol, we conduct large-scale crowdsourced evaluation to rank six recent gesture-generation models -- each trained by its original authors -- across two key evaluation dimensions: motion realism and speech-gesture alignment. Our results show that 1) motion realism has become a saturated evaluation measure on the BEAT2 dataset, with older models performing on par with more recent approaches; 2) previous findings of high speech-gesture alignment do not hold up under rigorous evaluation, even for specialised models; and 3) the field must adopt disentangled assessments of motion quality and multimodal alignment for accurate benchmarking in order to make progress. To drive standardisation and enable new evaluation research, we release five hours of synthetic motion from the benchmarked models; over 750 rendered video stimuli from the user studies -- enabling new evaluations without requiring model reimplementation -- alongside our open-source rendering script, and 16,000 pairwise human preference votes collected for our benchmark.
>
---
#### [replaced 002] PromptEcho: Annotation-Free Reward from Vision-Language Models for Text-to-Image Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.12652](https://arxiv.org/pdf/2604.12652)**

> **作者:** Jinlong Liu; Wanggui He; Peng Zhang; Mushui Liu; Hao Jiang; Pipei Huang
>
> **摘要:** Reinforcement learning (RL) can improve the prompt following capability of text-to-image (T2I) models, yet obtaining high-quality reward signals remains challenging: CLIP Score is too coarse-grained, while VLM-based reward models (e.g., RewardDance) require costly human-annotated preference data and additional fine-tuning. We propose PromptEcho, a reward construction method that requires \emph{no} annotation and \emph{no} reward model training. Given a generated image and a guiding query, PromptEcho computes the token-level cross-entropy loss of a frozen VLM with the original prompt as the label, directly extracting the image-text alignment knowledge encoded during VLM pretraining. The reward is deterministic, computationally efficient, and improves automatically as stronger open-source VLMs become available. For evaluation, we develop DenseAlignBench, a benchmark of concept-rich dense captions for rigorously testing prompt following capability. Experimental results on two state-of-the-art T2I models (Z-Image and QwenImage-2512) demonstrate that PromptEcho achieves substantial improvements on DenseAlignBench (+26.8pp / +16.2pp net win rate), along with consistent gains on GenEval, DPG-Bench, and TIIFBench without any task-specific training. Ablation studies confirm that PromptEcho comprehensively outperforms inference-based scoring with the same VLM, and that reward quality scales with VLM size. We will open-source the trained models and the DenseAlignBench.
>
---
#### [replaced 003] Structure-Semantic Decoupled Modulation of Global Geospatial Embeddings for High-Resolution Remote Sensing Mapping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.19591](https://arxiv.org/pdf/2604.19591)**

> **作者:** Jienan Lyu; Miao Yang; Jinchen Cai; Yiwen Hu; Guanyi Lu; Junhao Qiu; Runmin Dong
>
> **摘要:** Fine-grained high-resolution remote sensing mapping typically relies on localized visual features, which restricts cross-domain generalizability and often leads to fragmented predictions of large-scale land covers. While global geospatial foundation models offer powerful, generalizable representations, directly fusing their high-dimensional implicit embeddings with high-resolution visual features frequently triggers feature interference and spatial structure degradation due to a severe semantic-spatial gap. To overcome these limitations, we propose a Structure-Semantic Decoupled Modulation (SSDM) framework, which decouples global geospatial representations into two complementary cross-modal injection pathways. First, the structural prior modulation branch introduces the macroscopic receptive field priors from global representations into the self-attention modules of the high-resolution encoder. By guiding local feature extraction with holistic structural constraints, it effectively suppresses prediction fragmentation caused by high-frequency detail noise and excessive intra-class variance. Second, the global semantic injection branch explicitly aligns holistic context with the deep high-resolution feature space and directly supplements global semantics via cross-modal integration, thereby significantly enhancing the semantic consistency and category-level discrimination of complex land covers. Extensive experiments demonstrate that our method achieves state-of-the-art performance compared to existing cross-modal fusion approaches. By unleashing the potential of global embeddings, SSDM consistently improves high-resolution mapping accuracy across diverse scenarios, providing a universal and effective paradigm for integrating geospatial foundation models into high-resolution vision tasks.
>
---
#### [replaced 004] i-WiViG: Interpretable Window Vision GNN
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08321](https://arxiv.org/pdf/2503.08321)**

> **作者:** Ivica Obadic; Dmitry Kangin; Adrian Höhl; Dario Oliveira; Plamen P Angelov; Xiao Xiang Zhu
>
> **摘要:** Vision graph neural networks have emerged as a popular approach for modeling the global and spatial context for image recognition. However, a significant drawback of these methods is that they do not offer an inherent interpretation of the relevant spatial interactions for their prediction. We address this problem by introducing i-WiViG, an approach that enables interpretable model reasoning based on a sparse subgraph in the image. i-WiViG is based on two key postulates: 1) constraining the graph nodes' receptive field to disjoint local windows in the image, and 2) an inherently interpretable graph bottleneck with learnable sparse attention that identifies the relevant interactions among the local image windows. We evaluate our approach on both scene classification and regression tasks using natural and remote sensing imagery. Our results, supported by quantitative and qualitative evidence, demonstrate that the method delivers semantic, intuitive, and faithful explanations through the identified subgraphs. Furthermore, extensive experiments confirm that it achieves competitive performance to its black-box counterparts, even on datasets exhibiting strong texture bias. The implementation is available on this https URL.
>
---
#### [replaced 005] Towards reconstructing experimental sparse-view X-ray CT data with diffusion models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.12755](https://arxiv.org/pdf/2602.12755)**

> **作者:** Nelas J. Thomsen; Xinyuan Wang; Felix Lucka; Ezgi Demircan-Tureyen
>
> **备注:** 5 pages + references, 4 figures, 2 tables, conference paper
>
> **摘要:** Diffusion-based image generators are promising priors for ill-posed inverse problems like sparse-view X-ray Computed Tomography (CT). As most studies consider synthetic data, it is not clear whether training data mismatch (``domain shift'') or forward model mismatch complicate their successful application to experimental data. We measured CT data from a physical phantom resembling the synthetic Shepp-Logan phantom and trained diffusion priors on synthetic image data sets with different degrees of domain shift towards it. Then, we employed the priors in a Decomposed Diffusion Sampling scheme on sparse-view CT data sets with increasing difficulty leading to the experimental data. Our results reveal that domain shift plays a nuanced role: while severe mismatch causes model collapse and hallucinations, diverse priors outperform well-matched but narrow priors. Forward model mismatch pulls the image samples away from the prior manifold, which causes artifacts but can be mitigated with annealed likelihood schedules that also increase computational efficiency. Overall, we demonstrate that performance gains do not immediately translate from synthetic to experimental data, and future development must validate against real-world benchmarks.
>
---
#### [replaced 006] CLIP-RD: Relative Distillation for Efficient CLIP Knowledge Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25383](https://arxiv.org/pdf/2603.25383)**

> **作者:** Jeannie Chung; Hanna Jang; Ingyeong Yang; Uiwon Hwang; Jaehyeong Sim
>
> **摘要:** CLIP aligns image and text embeddings via contrastive learning and demonstrates strong zero-shot generalization. Its large-scale architecture requires substantial computational and memory resources, motivating the distillation of its capabilities into lightweight student models. However, existing CLIP distillation methods do not explicitly model multi-directional relational dependencies between teacher and student embeddings, limiting the student's ability to preserve the structural relationships encoded by the teacher. To address this, we propose a relational knowledge distillation framework that introduces two novel methods, Vertical Relational Distillation (VRD) and Cross Relational Distillation (XRD). VRD enforces consistency of teacher-student distillation strength across modalities at the distribution level, while XRD imposes bidirectional symmetry on cross-modal teacher-student similarity distributions. By jointly modeling multi-directional relational structures, CLIP-RD promotes faithful alignment of the student embedding geometry with that of the teacher, outperforming existing methods by 0.8%p.
>
---
#### [replaced 007] Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.09429](https://arxiv.org/pdf/2604.09429)**

> **作者:** Wonbong Jang; Shikun Liu; Soubhik Sanyal; Juan Camilo Perez; Kam Woh Ng; Sanskar Agrawal; Juan-Manuel Perez-Rua; Yiannis Douratsos; Tao Xiang
>
> **备注:** 9 pages, 6 figures, 4 tables. Project page: this https URL
>
> **摘要:** Recovering camera parameters from images and rendering scenes from novel viewpoints have been treated as separate tasks in computer vision and graphics. This separation breaks down when image coverage is sparse or poses are ambiguous, since each task depends on what the other produces. We propose Rays as Pixels, a Video Diffusion Model (VDM) that learns a joint distribution over videos and camera trajectories. To our knowledge, this is the first model to predict camera poses and do camera-controlled video generation within a single framework. We represent each camera as dense ray pixels (raxels), a pixel-aligned encoding that lives in the same latent space as video frames, and denoise the two jointly through a Decoupled Self-Cross Attention mechanism. A single trained model handles three tasks: predicting camera trajectories from video, generating video from input images along a pre-defined trajectory, and jointly synthesizing video and trajectory from input images. We evaluate on pose estimation and camera-controlled video generation, and introduce a closed-loop self-consistency test showing that the model's predicted poses and its renderings conditioned on those poses agree. Ablations against Plücker embeddings confirm that representing cameras in a shared latent space with video is subtantially more effective.
>
---
#### [replaced 008] CLIP-SVD: Efficient and Interpretable Vision-Language Adaptation via Singular Values
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出CLIP-SVD，解决视觉-语言模型在新领域适应的问题。通过奇异值微调实现高效、可解释的参数适应，仅调整少量参数即可提升性能。**

- **链接: [https://arxiv.org/pdf/2509.03740](https://arxiv.org/pdf/2509.03740)**

> **作者:** Taha Koleilat; Hassan Rivaz; Yiming Xiao
>
> **备注:** TMLR 2026
>
> **摘要:** Vision-language models (VLMs) like CLIP have shown impressive zero-shot and few-shot learning capabilities across diverse applications. However, adapting these models to new fine-grained domains remains difficult due to reliance on prompt engineering and the high cost of full model fine-tuning. Existing adaptation approaches rely on augmented components, such as prompt tokens and adapter modules, which could limit adaptation quality, destabilize the model, and compromise the rich knowledge learned during pretraining. In this work, we present CLIP-SVD, a multi-modal and parameter-efficient adaptation framework that applies Singular Value Fine-tuning (SVF) to CLIP, leveraging Singular Value Decomposition (SVD) to modify the internal parameter space of CLIP without injecting additional modules. Specifically, we fine-tune only the singular values of the CLIP parameter matrices to rescale the basis vectors for domain adaptation while retaining the pretrained model. This design enables enhanced adaptation performance using only 0.04% of the model's total parameters and better preservation of its generalization ability. CLIP-SVD achieves state-of-the-art classification results on 11 natural and 10 biomedical datasets, outperforming previous methods in both accuracy and generalization under few-shot settings. Additionally, we leverage a natural language-based approach to analyze the effectiveness and dynamics of the CLIP adaptation to allow interpretability of CLIP-SVD. Overall, this work provides the first extensive empirical evaluation of SVD-based finetuning in the vision-language model setting. The code and biomedical corpus are publicly available at this https URL.
>
---
#### [replaced 009] CARLA-Air: Fly Drones Inside a CARLA World -- A Unified Infrastructure for Air-Ground Embodied Intelligence
- **分类: cs.RO; cs.AI; cs.CV; cs.HC**

- **简介: 该论文提出CARLA-Air，融合空中与地面模拟，解决多模态智能体协同仿真问题。它统一了高保真驾驶与飞行物理，支持多种任务的开发与测试。**

- **链接: [https://arxiv.org/pdf/2603.28032](https://arxiv.org/pdf/2603.28032)**

> **作者:** Tianle Zeng; Yanci Wen; Hong Zhang
>
> **备注:** Prebuilt binaries, project page, full source code, and community discussion group are all available at: this https URL
>
> **摘要:** The convergence of low-altitude economies, embodied intelligence, and air-ground cooperative systems creates growing demand for simulation infrastructure capable of jointly modeling aerial and ground agents within a single physically coherent environment. Existing open-source platforms remain domain-segregated: driving simulators lack aerial dynamics, while multirotor simulators lack realistic ground scenes. Bridge-based co-simulation introduces synchronization overhead and cannot guarantee strict spatial-temporal consistency. We present CARLA-Air, an open-source infrastructure that unifies high-fidelity urban driving and physics-accurate multirotor flight within a single Unreal Engine process. The platform preserves both CARLA and AirSim native Python APIs and ROS 2 interfaces, enabling zero-modification code reuse. Within a shared physics tick and rendering pipeline, CARLA-Air delivers photorealistic environments with rule-compliant traffic, socially-aware pedestrians, and aerodynamically consistent UAV dynamics, synchronously capturing up to 18 sensor modalities across all platforms at each tick. The platform supports representative air-ground embodied intelligence workloads spanning cooperation, embodied navigation and vision-language action, multi-modal perception and dataset construction, and reinforcement-learning-based policy training. An extensible asset pipeline allows integration of custom robot platforms into the shared world. By inheriting AirSim's aerial capabilities -- whose upstream development has been archived -- CARLA-Air ensures this widely adopted flight stack continues to evolve within a modern infrastructure. Released with prebuilt binaries and full source: this https URL
>
---
#### [replaced 010] Generative Prior-Guided Neural Interface Reconstruction for 3D Electrical Impedance Tomography
- **分类: math.NA; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.16487](https://arxiv.org/pdf/2505.16487)**

> **作者:** Haibo Liu; Junqing Chen; Guang Lin
>
> **摘要:** Reconstructing complex 3D interfaces from indirect measurements remains a grand challenge in scientific computing, particularly for ill-posed inverse problems like Electrical Impedance Tomography (EIT). Traditional shape optimization struggles with topological changes and regularization tuning, while emerging deep learning approaches often compromise physical fidelity or require prohibitive amounts of paired training data. We present a transformative ``solver-in-the-loop'' framework that bridges this divide by coupling a pre-trained 3D generative prior with a rigorous boundary integral equation (BIE) solver. Unlike Physics-Informed Neural Networks (PINNs) that treat physics as soft constraints, our architecture enforces the governing elliptic PDE as a hard constraint at every optimization step, ensuring strict physical consistency. Simultaneously, we navigate a compact latent manifold of plausible geometries learned by a differentiable neural shape representation, effectively regularizing the ill-posed problem through data-driven priors rather than heuristic smoothing. By propagating adjoint shape derivatives directly through the neural decoder, we achieve fast, stable convergence with dramatically reduced degrees of freedom. Extensive experiments on 3D high-contrast EIT demonstrate that this principled hybrid approach yields superior geometric accuracy and data efficiency which is difficult to achieve using traditional methods, establishing a robust new paradigm for physics-constrained geometric discovery.
>
---
#### [replaced 011] Physical Knot Classification Beyond Accuracy: A Benchmark and Diagnostic Study
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23286](https://arxiv.org/pdf/2603.23286)**

> **作者:** Shiheng Nie; Yunguang Yue
>
> **备注:** 20 pages, 2 figures, supplementary material included
>
> **摘要:** Physical knot classification is a challenging fine-grained recognition task in which the intended discriminative cue is rope crossing structure; however, high closed-set accuracy may still arise from low-level appearance shortcuts rather than genuine topological understanding. In this work, we introduce dataset (1,440 images, 10 classes), which trains models on loosely tied knots and evaluates them on tightly dressed configurations to probe whether structure-guided training yields topology-specific gains. We demonstrate that topological distance successfully predicts residual inter-class confusion across multiple backbone architectures, validating the utility of our topology-aware evaluation framework. Furthermore, we propose topology-aware centroid alignment (TACA) and an auxiliary crossing-number prediction objective as two complementary forms of structural supervision. Notably, Swin-T with TACA achieves a consistent positive specificity gain (Delta_spec = +1.18 pp) across all random seeds under the canonical protocol, and auxiliary crossing-number prediction exhibits robust performance across data regimes without the real-versus-random reversal observed for centroid alignment. Causal probes reveal that background changes alone flip 17-32% of predictions and phone-photo accuracy drops by 58-69 percentage points, underscoring that appearance bias remains the principal obstacle to deployment. These results collectively demonstrate that our diagnostic workflow provides a principled and practical tool for evaluating whether a hand-crafted structural prior delivers genuine task-relevant benefit beyond generic regularization.
>
---
#### [replaced 012] Efficient Transceiver Design for Aerial Image Transmission and Large-scale Scene Reconstruction
- **分类: cs.CV; cs.LG; eess.SP**

- **链接: [https://arxiv.org/pdf/2604.11098](https://arxiv.org/pdf/2604.11098)**

> **作者:** Zeyi Ren; Jialin Dong; Wei Zuo; Yikun Wang; Bingyang Cheng; Sheng Zhou; Zhisheng Niu
>
> **备注:** 6 pages, 6 figures, Accepted in ISIT 2026 IEEE International Symposium on Information Theory-w
>
> **摘要:** Large-scale three-dimensional (3D) scene reconstruction in low-altitude intelligent networks (LAIN) demands highly efficient wireless image transmission. However, existing schemes struggle to balance severe pilot overhead with the transmission accuracy required to maintain reconstruction fidelity. To strike a balance between efficiency and reliability, this paper proposes a novel deep learning-based end-to-end (E2E) transceiver design that integrates 3D Gaussian Splatting (3DGS) directly into the training process. By jointly optimizing the communication modules via the combined 3DGS rendering loss, our approach explicitly improves scene recovery quality. Furthermore, this task-driven framework enables the use of a sparse pilot scheme, significantly reducing transmission overhead while maintaining robust image recovery under low-altitude channel conditions. Extensive experiments on real-world aerial image datasets demonstrate that the proposed E2E design significantly outperforms existing baselines, delivering superior transmission performance and accurate 3D scene reconstructions.
>
---
#### [replaced 013] FA-Seg: A Fast and Accurate Diffusion-Based Method for Open-Vocabulary Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.23323](https://arxiv.org/pdf/2506.23323)**

> **作者:** Huy Che; Vinh-Tiep Nguyen
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) aims to segment objects from arbitrary text categories without requiring densely annotated datasets. Although contrastive learning based models enable zero-shot segmentation, they often lose fine spatial precision at pixel level, due to global representation bias. In contrast, diffusion-based models naturally encode fine-grained spatial features via attention mechanisms that capture both global context and local details. However, they often face challenges in balancing the computation costs and the quality of the segmentation mask. In this work, we present FA-Seg, a Fast and Accurate training-free framework for open-vocabulary segmentation based on diffusion models. FA-Seg performs segmentation using only a (1+1)-step from a pretrained diffusion model. Moreover, instead of running multiple times for different classes, FA-Seg performs segmentation for all classes at once. To further enhance the segmentation quality, FA-Seg introduces three key components: (i) a dual-prompt mechanism for discriminative, class-aware attention extraction, (ii) a Hierarchical Attention Refinement Method (HARD) that enhances semantic precision via multi-resolution attention fusion, and (iii) a Test-Time Flipping (TTF) scheme designed to improve spatial consistency. Extensive experiments show that FA-Seg achieves state-of-the-art training-free performance, obtaining 43.8% average mIoU across PASCAL VOC, PASCAL Context, and COCO Object benchmarks while maintaining superior inference efficiency. Our results demonstrate that FA-Seg provides a strong foundation for extendability, bridging the gap between segmentation quality and inference efficiency. The source code is available at this https URL.
>
---
#### [replaced 014] Locate-Then-Examine: Grounded Region Reasoning Improves Detection of AI-Generated Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像伪造检测任务，旨在解决高质合成图像难以识别的问题。提出LTE框架，通过定位与再检策略提升检测准确性和解释性。**

- **链接: [https://arxiv.org/pdf/2510.04225](https://arxiv.org/pdf/2510.04225)**

> **作者:** Yikun Ji; Yan Hong; Bowen Deng; Jun Lan; Huijia Zhu; Weiqiang Wang; Liqing Zhang; Jianfu Zhang
>
> **备注:** 18 pages, 11 figures (including supplementary material)
>
> **摘要:** The rapid growth of AI-generated imagery has blurred the boundary between real and synthetic content, raising practical concerns for digital integrity. Vision-language models (VLMs) can provide natural language explanations, but standard one-pass classifiers often miss subtle artifacts in high-quality synthetic images and offer limited grounding in the pixels. We propose Locate-Then-Examine (LTE), a two-stage VLM-based forensic framework that first localizes suspicious regions and then re-examines these crops together with the full image to refine the real vs. AI-generated verdict and its explanation. LTE explicitly links each decision to localized visual evidence through region proposals and region-aware reasoning. To support training and evaluation, we introduce TRACE, a dataset of 20,000 real and high-quality synthetic images with region-level annotations and automatically generated forensic explanations, constructed by a VLM-based pipeline with additional consistency checks and quality control. Across TRACE and multiple external benchmarks, LTE achieves competitive accuracy and improved robustness while providing human-understandable, region-grounded explanations suitable for forensic deployment.
>
---
#### [replaced 015] A novel attention mechanism for noise-adaptive and robust segmentation of microtubules in microscopy images
- **分类: q-bio.QM; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.07800](https://arxiv.org/pdf/2507.07800)**

> **作者:** Achraf Ait Laydi; Louis Cueff; Mewen Crespo; Yousef El Mourabit; Hélène Bouvrais
>
> **摘要:** Segmenting cytoskeletal filaments in microscopy images is essential for studying their roles in cellular processes. However, this task is highly challenging due to the fine, densely packed, and intertwined nature of these structures. Imaging limitations further complicate analysis. While deep learning has advanced segmentation of large, well-defined biological structures, its performance often degrades under such adverse conditions. Additional challenges include obtaining precise annotations for curvilinear structures and managing severe class imbalance during training. We introduce a novel noise-adaptive attention mechanism that extends the Squeeze-and-Excitation (SE) module to dynamically adjust to varying noise levels. Integrated into a U-Net decoder with residual encoder blocks, this yields ASE_Res_UNet, a lightweight yet high-performance model. We also developed a synthetic dataset generation strategy that ensures accurate annotations of fine filaments in noisy images. We systematically evaluated loss functions and metrics to mitigate class imbalance, ensuring robust performance assessment. ASE_Res_UNet effectively segmented microtubules in noisy synthetic images, outperforming its ablated variants. It also demonstrated superior segmentation compared to models with alternative attention mechanisms or distinct architectures, while requiring fewer parameters, making it efficient for resource-constrained environments. Evaluation on a newly curated real microscopy dataset and a recently reannotated dataset highlighted ASE_Res_UNet's effectiveness in segmenting microtubules beyond synthetic images. For these datasets, ASE_Res_UNet was competitive with a recent synthetic data-driven approach that shares two cytoskeleton pretrained models. Importantly, ASE_Res_UNet showed strong transferability to other curvilinear structures (blood vessels and nerves) across diverse imaging conditions.
>
---
#### [replaced 016] Weak-to-Strong Knowledge Distillation Accelerates Visual Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.15451](https://arxiv.org/pdf/2604.15451)**

> **作者:** Baiang Li; Wenhao Chai; Felix Heide
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Large-scale visual learning is increasingly limited by training cost. Existing knowledge distillation methods transfer from a stronger teacher to a weaker student for compression or final-accuracy improvement. We instead investigate distillation to accelerate the training of strong students. We propose a generalizable plug-and-play recipe that freezes a weaker teacher, applies distillation only in early training, and turns it off once the student reaches and surpasses teacher-level performance. For ImageNet and CIFAR classification, this strategy reaches target thresholds much earlier, with up to 4.8 times speedup measured by epochs. We confirm that the method generalizes to other tasks and report 1.7 times epoch speedup for object detection on the COCO dataset, and 2.5 times earlier target-FID crossing for diffusion generation on the CIFAR-10 dataset, measured in steps. These findings validate our method as a universal speedup mechanism for visual learning.
>
---
#### [replaced 017] REVNET: Rotation-Equivariant Point Cloud Completion via Vector Neuron Anchor Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.08558](https://arxiv.org/pdf/2601.08558)**

> **作者:** Zhifan Ni; Eckehard Steinbach
>
> **备注:** ICPR 2026
>
> **摘要:** Incomplete point clouds captured by 3D sensors often result in the loss of both geometric and semantic information. Most existing point cloud completion methods are built on rotation-variant frameworks trained with data in canonical poses, limiting their applicability in real-world scenarios. While data augmentation with random rotations can partially mitigate this issue, it significantly increases the learning burden and still fails to guarantee robust performance under arbitrary poses. To address this challenge, we propose the Rotation-Equivariant Anchor Transformer (REVNET), a novel framework built upon the Vector Neuron (VN) network for robust point cloud completion under arbitrary rotations. To preserve local details, we represent partial point clouds as sets of equivariant anchors and design a VN Missing Anchor Transformer to predict the positions and features of missing anchors. Furthermore, we extend VN networks with a rotation-equivariant bias formulation and a ZCA-based layer normalization to improve feature expressiveness. Leveraging the flexible conversion between equivariant and invariant VN features, our model can generate point coordinates with greater stability. Experimental results show that our method outperforms state-of-the-art approaches on the synthetic MVP dataset in the equivariant setting. On the real-world KITTI dataset, REVNET delivers competitive results compared to non-equivariant networks, without requiring input pose alignment. The source code will be released on GitHub under URL: this https URL.
>
---
#### [replaced 018] OnSiteVRU: A High-Resolution Trajectory Dataset for High-Density Vulnerable Road Users
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于交通场景数据集构建任务，旨在解决VRU行为数据不足的问题。通过采集多场景轨迹数据，提升自动驾驶系统对复杂交通环境的感知能力。**

- **链接: [https://arxiv.org/pdf/2503.23365](https://arxiv.org/pdf/2503.23365)**

> **作者:** Zhangcun Yan; Jianqing Li; Peng Hang; Jian Sun
>
> **摘要:** With the acceleration of urbanization and the growth of transportation demands, the safety of vulnerable road users (VRUs, such as pedestrians and cyclists) in mixed traffic flows has become increasingly prominent, necessitating high-precision and diverse trajectory data to support the development and optimization of autonomous driving systems. However, existing datasets fall short in capturing the diversity and dynamics of VRU behaviors, making it difficult to meet the research demands of complex traffic environments. To address this gap, this study developed the OnSiteVRU datasets, which cover a variety of scenarios, including intersections, road segments, and urban villages. These datasets provide trajectory data for motor vehicles, electric bicycles, and human-powered bicycles, totaling approximately 17,429 trajectories with a precision of 0.04 seconds. The datasets integrate both aerial-view natural driving data and onboard real-time dynamic detection data, along with environmental information such as traffic signals, obstacles, and real-time maps, enabling a comprehensive reconstruction of interaction events. The results demonstrate that VRU\_Data outperforms traditional datasets in terms of VRU density and scene coverage, offering a more comprehensive representation of VRU behavioral characteristics. This provides critical support for traffic flow modeling, trajectory prediction, and autonomous driving virtual testing. The dataset is publicly available for download at: this https URL.
>
---
#### [replaced 019] MMControl: Unified Multi-Modal Control for Joint Audio-Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.19679](https://arxiv.org/pdf/2604.19679)**

> **作者:** Liyang Li; Wen Wang; Canyu Zhao; Tianjian Feng; Zhiyue Zhao; Hao Chen; Chunhua Shen
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advances in Diffusion Transformers (DiTs) have enabled high-quality joint audio-video generation, producing videos with synchronized audio within a single model. However, existing controllable generation frameworks are typically restricted to video-only control. This restricts comprehensive controllability and often leads to suboptimal cross-modal alignment. To bridge this gap, we present MMControl, which enables users to perform Multi-Modal Control in joint audio-video generation. MMControl introduces a dual-stream conditional injection mechanism. It incorporates both visual and acoustic control signals, including reference images, reference audio, depth maps, and pose sequences, into a joint generation process. These conditions are injected through bypass branches into a joint audio-video Diffusion Transformer, enabling the model to simultaneously generate identity-consistent video and timbre-consistent audio under structural constraints. Furthermore, we introduce modality-specific guidance scaling, which allows users to independently and dynamically adjust the influence strength of each visual and acoustic condition at inference time. Extensive experiments demonstrate that MMControl achieves fine-grained, composable control over character identity, voice timbre, body pose, and scene layout in joint audio-video generation.
>
---
#### [replaced 020] EchoTorrent: Towards Swift, Sustained, and Streaming Multi-Modal Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.13669](https://arxiv.org/pdf/2602.13669)**

> **作者:** Rang Meng; Weipeng Wu; Yuming Li; Chenguang Ma
>
> **摘要:** Recent multi-modal video generation models have achieved high visual quality, but their prohibitive latency and limited temporal stability hinder real-time deployment. Streaming inference exacerbates these issues, leading to pronounced multimodal degradation, such as spatial blurring, temporal drift, and lip desynchronization, which creates an unresolved efficiency-performance trade-off. To this end, we propose EchoTorrent, a novel schema with a fourfold design: (1) Multi-Teacher Training fine-tunes a pre-trained model on distinct preference domains to obtain specialized domain experts, which sequentially transfer domain-specific knowledge to a student model; (2) Adaptive CFG Calibration (ACC-DMD), which calibrates the audio CFG augmentation errors in DMD via a phased spatiotemporal schedule, eliminating redundant CFG computations and enabling single-pass inference per step; (3) Hybrid Long Tail Forcing, which enforces alignment exclusively on tail frames during long-horizon self-rollout training via a causal-bidirectional hybrid architecture, effectively mitigates spatiotemporal degradation in streaming mode while enhancing fidelity to reference frames; and (4) VAE Decoder Refiner through pixel-domain optimization of the VAE decoder to recover high-frequency details while circumventing latent-space ambiguities. Extensive experiments and analysis demonstrate that EchoTorrent achieves few-pass autoregressive generation with substantially extended temporal consistency, identity preservation, and audio-lip synchronization.
>
---
#### [replaced 021] Automated Description Generation of Cytologic Findings for Lung Cytological Images Using a Pretrained Vision Model and Dual Text Decoders: Preliminary Study
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2403.18151](https://arxiv.org/pdf/2403.18151)**

> **作者:** Atsushi Teramoto; Ayano Michiba; Yuka Kiriyama; Tetsuya Tsukamoto; Kazuyoshi Imaizumi; Hiroshi Fujita
>
> **备注:** This paper has been published in Cytopathology (2025)
>
> **摘要:** Objective: Cytology plays a crucial role in lung cancer diagnosis. Pulmonary cytology involves cell morphological characterization in the specimen and reporting the corresponding findings, which are extremely burdensome tasks. In this study, we propose a technique to generate cytologic findings from for cytologic images to assist in the reporting of pulmonary cytology. Methods: For this study, 801 patch images were retrieved using cytology specimens collected from 206 patients; the findings were assigned to each image as a dataset for generating cytologic findings. The proposed method consists of a vision model and dual text decoders. In the former, a convolutional neural network (CNN) is used to classify a given image as benign or malignant, and the features related to the image are extracted from the intermediate layer. Independent text decoders for benign and malignant cells are prepared for text generation, and the text decoder switches according to the CNN classification results. The text decoder is configured using a Transformer that uses the features obtained from the CNN for generating findings. Results: The sensitivity and specificity were 100% and 96.4%, respectively, for automated benign and malignant case classification, and the saliency map indicated characteristic benign and malignant areas. The grammar and style of the generated texts were confirmed correct, achieving a BLEU-4 score of 0.828, reflecting high degree of agreement with the gold standard, outperforming existing LLM-based image-captioning methods and single-text-decoder ablation model. Conclusion: Experimental results indicate that the proposed method is useful for pulmonary cytology classification and generation of cytologic findings.
>
---
#### [replaced 022] Tstars-Tryon 1.0: Robust and Realistic Virtual Try-On for Diverse Fashion Items
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.19748](https://arxiv.org/pdf/2604.19748)**

> **作者:** Mengting Chen; Zhengrui Chen; Yongchao Du; Zuan Gao; Taihang Hu; Jinsong Lan; Chao Lin; Yefeng Shen; Xingjian Wang; Zhao Wang; Zhengtao Wu; Xiaoli Xu; Zhengze Xu; Hao Yan; Mingzhou Zhang; Jun Zheng; Qinye Zhou; Xiaoyong Zhu; Bo Zheng
>
> **备注:** 24 pages, model evaluation report
>
> **摘要:** Recent advances in image generation and editing have opened new opportunities for virtual try-on. However, existing methods still struggle to meet complex real-world demands. We present Tstars-Tryon 1.0, a commercial-scale virtual try-on system that is robust, realistic, versatile, and highly efficient. First, our system maintains a high success rate across challenging cases like extreme poses, severe illumination variations, motion blur, and other in-the-wild conditions. Second, it delivers highly photorealistic results with fine-grained details, faithfully preserving garment texture, material properties, and structural characteristics, while largely avoiding common AI-generated artifacts. Third, beyond apparel try-on, our model supports flexible multi-image composition (up to 6 reference images) across 8 fashion categories, with coordinated control over person identity and background. Fourth, to overcome the latency bottlenecks of commercial deployment, our system is heavily optimized for inference speed, delivering near real-time generation for a seamless user experience. These capabilities are enabled by an integrated system design spanning end-to-end model architecture, a scalable data engine, robust infrastructure, and a multi-stage training paradigm. Extensive evaluation and large-scale product deployment demonstrate that Tstars-Tryon1.0 achieves leading overall performance. To support future research, we also release a comprehensive benchmark. The model has been deployed at an industrial scale on the Taobao App, serving millions of users with tens of millions of requests.
>
---
#### [replaced 023] Scaling In-Context Segmentation with Hierarchical Supervision
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.12752](https://arxiv.org/pdf/2604.12752)**

> **作者:** T. Camaret Ndir; Marco Reisert; Robin T. Schirrmeister
>
> **摘要:** In-context learning (ICL) enables medical image segmentation models to adapt to new anatomical structures from limited examples, reducing the clinical annotation burden. However, standard ICL methods typically rely on dense, global cross-attention, which scales poorly with image resolution. While recent approaches have introduced localized attention mechanisms, they often lack explicit supervision on the selection process, leading to redundant computation in non-informative regions. We propose PatchICL, a hierarchical framework that combines selective image patching with multi-level supervision. Our approach learns to actively identify and attend only to the most informative anatomical regions. Compared to UniverSeg, a strong global-attention baseline, PatchICL achieves competitive in-domain CT segmentation accuracy while reducing compute by 44\% at $512\times512$ resolution. On 35 out-of-domain datasets spanning diverse imaging modalities, PatchICL outperforms the baseline on 6 of 13 modality categories, with particular strength on modalities dominated by localized pathology such as OCT and dermoscopy. Training and evaluation code are available at this https URL
>
---
#### [replaced 024] Human-like Content Analysis for Generative AI with Language-Grounded Sparse Encoders
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.18236](https://arxiv.org/pdf/2508.18236)**

> **作者:** Yiming Tang; Arash Lagzian; Srinivas Anumasa; Qiran Zou; Yingtao Zhu; Ye Zhang; Trang Nguyen; Yih-Chung Tham; Ehsan Adeli; Ching-Yu Cheng; Yilun Du; Dianbo Liu
>
> **摘要:** The rapid development of generative AI has transformed content creation, communication, and human development. However, this technology raises profound concerns in high-stakes domains, demanding rigorous methods to analyze and evaluate AI-generated content. While existing analytic methods often treat images as indivisible wholes, real-world AI failures generally manifest as specific visual patterns that can evade holistic detection and suit more granular and decomposed analysis. Here we introduce a content analysis tool, Language-Grounded Sparse Encoders (LanSE), which decompose images into interpretable visual patterns with natural language descriptions. Utilizing interpretability modules and large multimodal models, LanSE can automatically identify visual patterns within data modalities. Our method discovers more than 5,000 visual patterns with 93\% human agreement, provides decomposed evaluation outperforming existing methods, establishes the first systematic evaluation of physical plausibility, and extends to medical imaging settings. Our method's capability to extract language-grounded patterns can be naturally adapted to numerous fields, including biology and geography, as well as other data modalities such as protein structures and time series, thereby advancing content analysis for generative AI.
>
---
#### [replaced 025] retinalysis-vascx: An explainable software toolbox for the extraction of retinal vascular biomarkers
- **分类: q-bio.TO; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.08580](https://arxiv.org/pdf/2602.08580)**

> **作者:** Jose D. Vargas Quiros; Michael J. Beyeler; Sofia Ortin Vela; EyeNED Reading Center; Sven Bergmann; Caroline C.W. Klave; Bart Liefers; VascX Research Consortium
>
> **摘要:** Automatic extraction of retinal vascular biomarkers from color fundus images (CFI) is crucial for large-scale studies of the retinal vasculature. We present VascX, an open-source Python toolbox that extracts biomarkers from CFI artery-vein segmentations. VascX starts from vessel segmentation masks, extracts their skeletons, builds undirected and directed vessel graphs, and resolves vessel segments into longer vessels. A comprehensive set of biomarkers is derived, including vascular density, central retinal equivalents (CREs), and tortuosity. Spatially localized biomarkers may be calculated over grids placed relative to the fovea and optic disc. VascX is released via GitHub and PyPI with comprehensive documentation and examples. Our test-retest reproducibility analysis on repeat imaging of the same eye by different devices shows that most VascX biomarkers have moderate to excellent agreement (ICC > 0.5), with important differences in the level of robustness of different biomarkers. Our analyses of biomarker sensitivity to image perturbations and heuristic parameter values support these differences and further characterize VascX biomarkers. Ultimately, VascX provides an explainable and easily modifiable feature-extraction toolbox that complements segmentation to produce reliable retinal vascular biomarkers. Our graph-based biomarker computation stages support reproducible, region-aware measurements suited for large-scale clinical and epidemiological research. By enabling easy extraction of existing biomarkers and rapid experimentation with new ones, VascX supports oculomics research. Its robustness and computational efficiency facilitate scalable deployment in large databases, while open-source distribution lowers barriers to adoption for ophthalmic researchers and clinicians.
>
---
#### [replaced 026] Combo-Gait: Unified Transformer Framework for Multi-Modal Gait Recognition and Attribute Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.10417](https://arxiv.org/pdf/2510.10417)**

> **作者:** Zhao-Yang Wang; Zhimin Shao; Anirudh Nanduri; Basudha Pal; Laura McDaniel; Jieneng Chen; Rama Chellappa
>
> **摘要:** Gait recognition is an important biometric for human identification at a distance, particularly under low-resolution or unconstrained environments. Current works typically focus on either 2D representations (e.g., silhouettes and skeletons) or 3D representations (e.g., meshes and SMPLs), but relying on a single modality often fails to capture the full geometric and dynamic complexity of human walking patterns. In this paper, we propose a multi-modal and multi-task framework that combines 2D temporal silhouettes with 3D SMPL features for robust gait analysis. Beyond identification, we introduce a multitask learning strategy that jointly performs gait recognition and human attribute estimation, including age, body mass index (BMI), and gender. A unified transformer is employed to effectively fuse multi-modal gait features and better learn attribute-related representations, while preserving discriminative identity cues. Extensive experiments on the large-scale BRIAR datasets, collected under challenging conditions such as long-range distances (up to 1 km) and extreme pitch angles (up to 50°), demonstrate that our approach outperforms state-of-the-art methods in gait recognition and provides accurate human attribute estimation. These results highlight the promise of multi-modal and multitask learning for advancing gait-based human understanding in real-world scenarios.
>
---
#### [replaced 027] Adaptive Forensic Feature Refinement via Intrinsic Importance Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.16879](https://arxiv.org/pdf/2604.16879)**

> **作者:** Jiazhen Yang; Junjun Zheng; Kejia Chen; Xiangheng Kong; Jie Lei; Zunlei Feng; Bingde Hu; Yang Gao
>
> **摘要:** With the rapid development of generative models and multimodal content editing technologies, the key challenge faced by synthetic image detection (SID) lies in cross-distribution generalization to unknown generation sources. In recent years, visual foundation models (VFM), which acquire rich visual priors through large scale image-text alignment pretraining, have become a promising technical route for improving the generalization ability of SID. However, existing VFM-based methods remain relatively coarse-grained in their adaptation strategies. They typically either directly use the final layer representations of VFM or simply fuse multi layer features, lacking explicit modeling of the optimal representational hierarchy for transferable forgery cues. Meanwhile, although directly fine-tuning VFM can enhance task adaptation, it may also damage the cross-modal pretrained structure that supports open-set generalization. To address this task specific tension, we reformulate VFM adaptation for SID as a joint optimization problem: it is necessary both to identify the critical representational layer that is more suitable for carrying forgery discriminative information and to constrain the disturbance caused by task knowledge injection to the pretrained structure. Based on this, we propose I2P, an SID framework centered on intrinsic importance perception. I2P first adaptively identifies the critical layer representations that are most discriminative for SID, and then constrains task-driven parameter updates within a low sensitivity parameter subspace, thereby improving task specificity while preserving the transferable structure of pretrained representations as much as possible.
>
---
#### [replaced 028] Retinex Meets Language: A Physics-Semantics-Guided Underwater Image Enhancement Network
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07076](https://arxiv.org/pdf/2603.07076)**

> **作者:** Shixuan Xu; Yabo Liu; Chao Huang; Junyu Dong; Xinghui Dong
>
> **摘要:** Underwater images often suffer from severe degradation caused by light absorption and scattering, leading to color distortion, low contrast and reduced visibility. Existing Underwater Image Enhancement (UIE) methods can be divided into two categories, i.e., prior-based and learning-based methods. The former rely on rigid physical assumptions that limit the adaptability, while the latter often face data scarcity and weak generalization. To address these issues, we propose a Physics-Semantics-Guided Underwater Image Enhancement Network (PSG-UIENet), which couples the Retinex-grounded illumination correction with the language-informed guidance. This network comprises a Prior-Free Illumination Estimator and a Semantics-Guided Image Restorer. In particular, the restorer leverages the textual descriptions generated by the Contrastive Language-Image Pre-training (CLIP) model to inject high-level semantics for perceptually meaningful guidance. Since multimodal UIE data sets are not publicly available, we also construct a large-scale image-text UIE data set, namely, LUIQD-TD, which contains 6,418 image-reference-text triplets. To explicitly measure and optimize semantic consistency between textual descriptions and images, we further design an Image-Text Semantic Similarity (ITSS) loss function. To our knowledge, this study makes the first effort to introduce both textual guidance and the multimodal data set into UIE tasks. Extensive experiments on our data set and four publicly available data sets demonstrate that the proposed PSG-UIENet achieves superior or comparable performance against fifteen state-of-the-art methods.
>
---
#### [replaced 029] Rodrigues Network for Learning Robot Actions
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决传统神经网络缺乏运动学结构先验的问题。提出RodriNet架构，融入运动学先验以提升动作学习效果。**

- **链接: [https://arxiv.org/pdf/2506.02618](https://arxiv.org/pdf/2506.02618)**

> **作者:** Jialiang Zhang; Haoran Geng; Yang You; Congyue Deng; Pieter Abbeel; Jitendra Malik; Leonidas Guibas
>
> **备注:** ICLR 2026
>
> **摘要:** Understanding and predicting articulated actions is important in robot learning. However, common architectures such as MLPs and Transformers lack inductive biases that reflect the underlying kinematic structure of articulated systems. To this end, we propose the Neural Rodrigues Operator, a learnable generalization of the classical forward kinematics operation, designed to inject kinematics-aware inductive bias into neural computation. Building on this operator, we design the Rodrigues Network (RodriNet), a novel neural architecture specialized for processing actions. We evaluate the expressivity of our network on two synthetic tasks on kinematic and motion prediction, showing significant improvements compared to standard backbones. We further demonstrate its effectiveness in two realistic applications: (i) imitation learning on robotic benchmarks with the Diffusion Policy, and (ii) single-image 3D hand reconstruction. Our results suggest that integrating structured kinematic priors into the network architecture improves action learning in various domains.
>
---
#### [replaced 030] Survival of the Cheapest: Cost-Aware Hardware Adaptation for Adversarial Robustness
- **分类: cs.CR; cs.CV; cs.LG; stat.AP**

- **链接: [https://arxiv.org/pdf/2409.07609](https://arxiv.org/pdf/2409.07609)**

> **作者:** Charles Meyers; Mohammad Reza Saleh Sedghpour; Tommy Löfstedt; Erik Elmroth
>
> **摘要:** Deploying adversarially robust machine learning systems requires continuous trade-offs between robustness, cost, and latency. We present an autonomic decision-support framework providing a quantitative foundation for adaptive hardware selection and hyper-parameter tuning in cloud-native deep learning. The framework applies accelerated failure time (AFT) models to quantify the effect of hardware choice, batch size, epochs, and validation accuracy on model survival time. This framework can be naturally integrated into an autonomic control loop (monitor--analyse--plan--execute, MAPE-K), where system metrics such as cost, robustness, and latency are continuously evaluated and used to adapt model configurations and hardware selection. Experiments across three GPU architectures confirm the framework is both sound and cost-effective: the Nvidia L4 yields a 20% increase in adversarial survival time while costing 75% less than the V100, demonstrating that expensive hardware does not necessarily improve robustness. The analysis further reveals that model inference latency is a stronger predictor of adversarial robustness than training time or hardware configuration.
>
---
#### [replaced 031] Air-Know: Arbiter-Calibrated Knowledge-Internalizing Robust Network for Composed Image Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.19386](https://arxiv.org/pdf/2604.19386)**

> **作者:** Zhiheng Fu; Yupeng Hu; Qianyun Yang; Shiqi Zhang; Zhiwei Chen; Zixu Li
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Composed Image Retrieval (CIR) has attracted significant attention due to its flexible multimodal query method, yet its development is severely constrained by the Noisy Triplet Correspondence (NTC) problem. Most existing robust learning methods rely on the "small loss hypothesis", but the unique semantic ambiguity in NTC, such as "partial matching", invalidates this assumption, leading to unreliable noise identification. This entraps the model in a self dependent vicious cycle where the learner is intertwined with the arbiter, ultimately causing catastrophic "representation pollution". To address this critical challenge, we propose a novel "Expert-Proxy-Diversion" decoupling paradigm, named Air-Know (ArbIteR calibrated Knowledge iNternalizing rObust netWork). Air-Know incorporates three core modules: (1) External Prior Arbitration (EPA), which utilizes Multimodal Large Language Models (MLLMs) as an offline expert to construct a high precision anchor dataset; (2) Expert Knowledge Internalization (EKI), which efficiently guides a lightweight proxy "arbiter" to internalize the expert's discriminative logic; (3) Dual Stream Reconciliation (DSR), which leverages the EKI's matching confidence to divert the training data, achieving a clean alignment stream and a representation feedback reconciliation stream. Extensive experiments on multiple CIR benchmark datasets demonstrate that Air-Know significantly outperforms existing SOTA methods under the NTC setting, while also showing strong competitiveness in traditional CIR.
>
---
#### [replaced 032] VAN-AD: Visual Masked Autoencoder with Normalizing Flow For Time Series Anomaly Detection
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.26842](https://arxiv.org/pdf/2603.26842)**

> **作者:** PengYu Chen; Shang Wan; Xiaohou Shi; Yuan Chang; Yan Sun; Sajal K. Das
>
> **备注:** 13 pages, 20 figures
>
> **摘要:** Time series anomaly detection (TSAD) is essential for maintaining the reliability and security of IoT-enabled service systems. Existing methods require training one specific model for each dataset, which exhibits limited generalization capability across different target datasets, hindering anomaly detection performance in various scenarios with scarce training data. To address this limitation, foundation models have emerged as a promising direction. However, existing approaches either repurpose large language models (LLMs) or construct largescale time series datasets to develop general anomaly detection foundation models, and still face challenges caused by severe cross-modal gaps or in-domain heterogeneity. In this paper, we investigate the applicability of large-scale vision models to TSAD. Specifically, we adapt a visual Masked Autoencoder (MAE) pretrained on ImageNet to the TSAD task. However, directly transferring MAE to TSAD introduces two key challenges: overgeneralization and limited local perception. To address these challenges, we propose VAN-AD, a novel MAE-based framework for TSAD. To alleviate the over-generalization issue, we design an Adaptive Distribution Mapping Module (ADMM), which maps the reconstruction results before and after MAE into a unified statistical space to amplify discrepancies caused by abnormal patterns. To overcome the limitation of local perception, we further develop a Normalizing Flow Module (NFM), which combines MAE with normalizing flow to estimate the probability density of the current window under the global distribution. Extensive experiments on nine real-world datasets demonstrate that VAN-AD consistently outperforms existing state-of-the-art methods across multiple evaluation this http URL make our code and datasets available at this https URL.
>
---
#### [replaced 033] Physics-informed Active Polarimetric 3D Imaging for Specular Surfaces
- **分类: cs.CV; physics.optics**

- **链接: [https://arxiv.org/pdf/2602.19470](https://arxiv.org/pdf/2602.19470)**

> **作者:** Jiazhang Wang; Hyelim Yang; Tianyi Wang; Florian Willomitzer
>
> **摘要:** 3D imaging of specular surfaces remains challenging in real-world scenarios, such as in-line inspection or hand-held scanning, requiring fast and accurate measurement of complex geometries. Optical metrology techniques such as deflectometry achieve high accuracy but typically rely on multi-shot acquisition, making them unsuitable for dynamic environments. Fourier-based single-shot approaches alleviate this constraint, yet their performance deteriorates when measuring surfaces with high spatial frequency structure or large curvature. Alternatively, polarimetric 3D imaging in computer vision operates in a single-shot fashion and exhibits robustness to geometric complexity. However, its accuracy is fundamentally limited by the orthographic imaging assumption. In this paper, we propose a physics-informed deep learning framework for single-shot 3D imaging of complex specular surfaces. Polarization cues provide orientation priors that assist in interpreting geometric information encoded by structured illumination. These complementary cues are processed through a dual-encoder architecture with mutual feature modulation, allowing the network to resolve their nonlinear coupling and directly infer surface normals. The proposed method achieves accurate and robust normal estimation in single-shot with fast inference, enabling practical 3D imaging of complex specular surfaces.
>
---
#### [replaced 034] Robust Principal Component Completion
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.25132](https://arxiv.org/pdf/2603.25132)**

> **作者:** Yinjian Wang; Wei Li; Yuanyuan Gui; James E. Fowler; Gemine Vivone
>
> **摘要:** Robust principal component analysis (RPCA) seeks a low-rank component and a sparse component from their summation. Yet, in many applications of interest, the sparse foreground actually replaces, or occludes, elements from the low-rank background. To address this mismatch, a new framework is proposed in which the sparse component is identified indirectly through determining its support. This approach, called robust principal component completion (RPCC), is solved via variational Bayesian inference applied to a fully probabilistic Bayesian sparse tensor factorization. Convergence to a hard classifier for the support is shown, thereby eliminating the post-hoc thresholding required of most prior RPCA-driven approaches. Experimental results reveal that the proposed approach delivers near-optimal estimates on synthetic data as well as robust foreground-extraction and anomaly-detection performance on real color video and hyperspectral datasets, respectively. Source implementation and Appendices are available at this https URL.
>
---
#### [replaced 035] AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.18562](https://arxiv.org/pdf/2604.18562)**

> **作者:** Rui Qian; Chuanhang Deng; Qiang Huang; Jian Xiong; Mingxuan Li; Yingbo Zhou; Wei Zhai; Jintao Chen; Dejing Dou
>
> **备注:** This work has been accepted to ACL 2026, please refer to this https URL
>
> **摘要:** Reasoning segmentation requires models to ground complex, implicit textual queries into precise pixel-level masks. Existing approaches rely on a single segmentation token $\texttt{<SEG>}$, whose hidden state implicitly encodes both semantic reasoning and spatial localization, limiting the model's ability to explicitly disentangle what to segment from where to segment. We introduce AnchorSeg, which reformulates reasoning segmentation as a structured conditional generation process over image tokens, conditioned on language grounded query banks. Instead of compressing all semantic reasoning and spatial localization into a single embedding, AnchorSeg constructs an ordered sequence of query banks: latent reasoning tokens that capture intermediate semantic states, and a segmentation anchor token that provides explicit spatial grounding. We model spatial conditioning as a factorized distribution over image tokens, where the anchor query determines localization signals while contextual queries provide semantic modulation. To bridge token-level predictions and pixel-level supervision, we propose Token--Mask Cycle Consistency (TMCC), a bidirectional training objective that enforces alignment across resolutions. By explicitly decoupling spatial grounding from semantic reasoning through structured language grounded query banks, AnchorSeg achieves state-of-the-art results on ReasonSeg test set (67.7\% gIoU and 68.1\% cIoU). All code and models are publicly available at this https URL.
>
---
#### [replaced 036] From Ideal to Real: Stable Video Object Removal under Imperfect Conditions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09283](https://arxiv.org/pdf/2603.09283)**

> **作者:** Jiagao Hu; Yuxuan Chen; Fuhao Li; Zepeng Wang; Fei Wang; Daiguo Zhou; Jian Luan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Removing objects from videos remains difficult in the presence of real-world imperfections such as shadows, abrupt motion, and defective masks. Existing diffusion-based video inpainting models often struggle to maintain temporal stability and visual consistency under these challenges. We propose Stable Video Object Removal (SVOR), a robust framework that achieves shadow-free, flicker-free, and mask-defect-tolerant removal through three key designs: (1) Mask Union for Stable Erasure (MUSE), a windowed union strategy applied during temporal mask downsampling to preserve all target regions observed within each window, effectively handling abrupt motion and reducing missed removals; (2) Denoising-Aware Segmentation (DA-Seg), a lightweight segmentation head on a decoupled side branch equipped with Denoising-Aware AdaLN and trained with mask degradation to provide an internal diffusion-aware localization prior without affecting content generation; and (3) Curriculum Two-Stage Training: where Stage I performs self-supervised pretraining on unpaired real-background videos with online random masks to learn realistic background and temporal priors, and Stage II refines on synthetic pairs using mask degradation and side-effect-weighted losses, jointly removing objects and their associated shadows/reflections while improving cross-domain robustness. Extensive experiments show that SVOR attains new state-of-the-art results across multiple datasets and degraded-mask benchmarks, advancing video object removal from ideal settings toward real-world applications. Project page: this https URL.
>
---
#### [replaced 037] Benchmarking ResNet for Short-Term Hypoglycemia Classification with DiaData
- **分类: eess.SP; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2511.02849](https://arxiv.org/pdf/2511.02849)**

> **作者:** Beyza Cinar; Maria Maleshkova
>
> **备注:** 11 pages, 5 Tables, 4 Figures, BHI 2025 conference (JBHI special issue). References were corrected
>
> **摘要:** Individualized therapy is driven forward by medical data analysis, which provides insight into the patient's context. In particular, for Type 1 Diabetes (T1D), which is an autoimmune disease, relationships between demographics, sensor data, and context can be analyzed. However, outliers, noisy data, and small data volumes cannot provide a reliable analysis. Hence, the research domain requires large volumes of high-quality data. Moreover, missing values can lead to information loss. To address this limitation, this study improves the data quality of DiaData, an integration of 15 separate datasets containing glucose values from 2510 subjects with T1D. Notably, we make the following contributions: 1) Outliers are identified with the interquartile range (IQR) approach and treated by replacing them with missing values. 2) Small gaps ($\le$ 25 min) are imputed with linear interpolation and larger gaps ($\ge$ 30 and $<$ 120 min) with Stineman interpolation. Based on a visual comparison, Stineman interpolation provides more realistic glucose estimates than linear interpolation for larger gaps. 3) After data cleaning, the correlation between glucose and heart rate is analyzed, yielding a moderate relation between 15 and 60 minutes before hypoglycemia ($\le$ 70 mg/dL). 4) Finally, a benchmark for hypoglycemia classification is provided with a state-of-the-art ResNet model. The model is trained with the Maindatabase and Subdatabase II of DiaData to classify hypoglycemia onset up to 2 hours in advance. Training with more data improves performance by 7% while using quality-refined data yields a 2-3% gain compared to raw data.
>
---
#### [replaced 038] AnatomicalNets: A Multi-Structure Segmentation and Contour-Based Distance Estimation Pipeline for Clinically Grounded Lung Cancer T-Staging
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.19367](https://arxiv.org/pdf/2511.19367)**

> **作者:** Saniah Kayenat Chowdhury; Rusab Sarmun; Muhammad E. H. Chowdhury; Sohaib Bassam Zoghoul; Israa Al-Hashimi; Adam Mushtak; Amith Khandakar
>
> **摘要:** Accurate tumor staging in lung cancer is crucial for prognosis and treatment planning and is governed by explicit anatomical criteria under fixed guidelines. However, most existing deep learning approaches treat this spatially structured clinical decision as an uninterpretable image classification problem. Tumor stage depends on predetermined quantitative criteria, including the tumor's dimensions and its proximity to adjacent anatomical structures, and small variations can alter the staging outcome. To address this gap, we propose AnatomicalNets, a medically grounded, multi-stage pipeline that reformulates tumor staging as a measurement and rule-based inference problem rather than a learned mapping. We employ three dedicated encoder-decoder networks to precisely segment the lung parenchyma, tumor, and mediastinum. The diaphragm boundary is estimated via a lung-contour heuristic, while the tumor's largest dimension and its proximity to adjacent structures are computed through a contour-based distance estimation method. These features are passed through a deterministic decision module following the international association for the study of lung cancer guidelines. Evaluated on the Lung-PET-CT-Dx dataset, AnatomicalNets achieves an overall classification accuracy of 91.36%. We report the per-stage F1-scores of 0.93 (T1), 0.89 (T2), 0.96 (T3), and 0.90 (T4), a critical evaluation aspect often omitted in prior literature. We highlight that the representational bottleneck in prior work lies in feature design rather than classifier capacity. This work establishes a transparent and reliable staging paradigm that bridges the gap between deep learning performance and clinical interpretability.
>
---
#### [replaced 039] Excretion Detection in Pigsties Using Convolutional and Transformerbased Deep Neural Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.00256](https://arxiv.org/pdf/2412.00256)**

> **作者:** Simon Mielke; Anthony Stein
>
> **备注:** Keywords: Artificial Intelligence, Objected detection, Pig, Urine puddle, Thermal IR data, CNN vs Transformer, Precision Livestock Farming; Stats: 53 pages, 13 figures
>
> **摘要:** Animal excretions in form of urine puddles and feces are a significant source of emissions in livestock farming. Automated detection of soiled floor in barns can contribute to improved management processes but also the derived information can be used to model emission dynamics. Previous research approaches to determine the puddle area require manual detection of the puddle in the barn. While humans can detect animal excretions on thermal images of a livestock barn, automated approaches using thresholds fail due to other objects of the same temperature, such as the animals themselves. In addition, various parameters such as the type of housing, animal species, age, sex, weather and unknown factors can influence the type and shape of excretions. Due to this heterogeneity, a method for automated detection of excretions must therefore be not only be accurate but also robust to varying conditions. These requirements can be met by using contemporary deep learning models from the field of artificial intelligence. This work is the first to investigate the suitability of different deep learning models for the detection of excretions in pigsties, thereby comparing established convolutional architectures with recent transformer-based approaches. The detection models Faster R-CNN, YOLOv8, DETR and DAB-DETR are compared and statistically assessed on two created training datasets representing two pig houses. We apply a method derived from nested cross-validation and report on the results in terms of eight common detection metrics. Our work demonstrates that all investigated deep learning models are generally suitable for reliably detecting excretions with an average precision of over 90%. The models also show robustness on out of distribution data that possesses differences from the conditions in the training data, however, with expected slight decreases in the overall detection performance.
>
---
#### [replaced 040] PipeMFL-240K: A Large-scale Dataset and Benchmark for Object Detection in Pipeline Magnetic Flux Leakage Imaging
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.07044](https://arxiv.org/pdf/2602.07044)**

> **作者:** Tianyi Qu; Songxiao Yang; Haolin Wang; Huadong Song; Xiaoting Guo; Wenguang Hu; Guanlin Liu; Honghe Chen; Yafei Ou
>
> **备注:** A dataset contains 249,320 pipeline MFL pseudo-color images and 200,020 bounding-box annotations, collected from 12 pipelines spanning approximately 1,530 km
>
> **摘要:** Pipeline integrity is critical to industrial safety and environmental protection, with Magnetic Flux Leakage (MFL) detection being a primary non-destructive testing technology. Despite the promise of deep learning for automating MFL interpretation, progress toward reliable models has been constrained by the absence of a large-scale public dataset and benchmark, making fair comparison and reproducible evaluation difficult. We introduce \textbf{PipeMFL-240K}, a large-scale, meticulously annotated dataset and benchmark for complex object detection in pipeline MFL pseudo-color images. PipeMFL-240K reflects real-world inspection complexity and poses several unique challenges: (i) an extremely long-tailed distribution over \textbf{12} categories, (ii) a high prevalence of tiny objects that often comprise only a handful of pixels and (iii) substantial intra-class variability. The dataset contains \textbf{249,320} images and \textbf{200,020} high-quality bounding-box annotations, collected from 12 pipelines spanning approximately \textbf{1,530} km. Extensive experiments are conducted with state-of-the-art object detectors to establish baselines. Results show that modern detectors still struggle with the intrinsic properties of MFL data, highlighting considerable headroom for improvement, while PipeMFL-240K provides a reliable and challenging testbed to drive future research. As the first public dataset and the first benchmark of this scale and scope for pipeline MFL inspection, it provides a critical foundation for efficient pipeline diagnostics as well as maintenance planning and is expected to accelerate algorithmic innovation and reproducible research in MFL-based pipeline integrity assessment.
>
---
#### [replaced 041] Unsupervised Local Plasticity in a Multi-Frequency VisNet Hierarchy
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.09734](https://arxiv.org/pdf/2604.09734)**

> **作者:** Mehdi Fatan Serj; C. Alejandro Parraga; Xavier Otazu
>
> **摘要:** We introduce an unsupervised visual representation learning system based entirely on local plasticity rules, without labels, backpropagation, or global error signals. The model is a VisNet-inspired hierarchical architecture combining opponent color inputs, multi-frequency Gabor and wavelet feature streams, competitive normalization with lateral inhibition, saliency modulation, associative memory, and a feedback loop. All representation learning occurs through continuous local plasticity applied to unlabeled image streams over 300 epochs. Performance is evaluated using a fixed linear probe trained only at readout time. The system achieves 80.1 percent accuracy on CIFAR-10 and 47.6 percent on CIFAR-100, improving over a Hebbian-only baseline. Ablation studies show that anti-Hebbian decorrelation, free-energy inspired plasticity, and associative memory are the main contributors, with strong synergistic effects. Even without learning, the fixed architecture alone reaches 61.4 percent on CIFAR-10, indicating that plasticity, not only inductive bias, drives most of the performance. Control analyses show that independently trained probes match co-trained ones within 0.3 percentage points, and a nearest-class-mean classifier achieves 78.3 percent without gradient-based training, confirming the intrinsic structure of the learned features. Overall, the system narrows but does not eliminate the performance gap to backpropagation-trained CNNs (5.7 percentage points on CIFAR-10, 7.5 percentage points on CIFAR-100), demonstrating that structured local plasticity alone can learn strong visual representations from raw unlabeled data.
>
---
#### [replaced 042] Unified Ultrasound Intelligence Toward an End-to-End Agentic System
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2604.16914](https://arxiv.org/pdf/2604.16914)**

> **作者:** Chen Ma; Yunshu Li; Junhu Fu; Shuyu Liang; Yuanyuan Wang; Yi Guo
>
> **备注:** Accepted by ISBI2026. 5 pages, 2 figures
>
> **摘要:** Clinical ultrasound analysis demands models that generalize across heterogeneous organs, views, and devices, while supporting interpretable workflow-level analysis. Existing methods often rely on task-wise adaptation, and joint learning may be unstable due to cross-task interference, making it hard to deliver workflow-level outputs in practice. To address these challenges, we present USTri, a tri-stage ultrasound intelligence pipeline for unified multi-organ, multi-task analysis. Stage I trains a universal generalist USGen on different domains to learn broad, transferable priors that are robust to device and protocol variability. To better handle domain shifts and reach task-aligned performance while preserving ultrasound shared knowledge, Stage II builds USpec by keeping USGen frozen and finetuning dataset-specific heads. Stage III introduces USAgent, which mimics clinician workflows by orchestrating USpec specialists for multi-step inference and deterministic structured reports. On the FMC\_UIA validation set, our model achieves the best overall performance across 4 task types and 27 datasets, outperforming state-of-the-art methods. Moreover, qualitative results show that USAgent produces clinically structured reports with high accuracy and interpretability. Our study suggests a scalable path to ultrasound intelligence that generalizes across heterogeneous ultrasound tasks and supports consistent end-to-end clinical workflows. The code is publicly available at: this https URL.
>
---
#### [replaced 043] EgoSelf: From Memory to Personalized Egocentric Assistant
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.19564](https://arxiv.org/pdf/2604.19564)**

> **作者:** Yanshuo Wang; Yuan Xu; Xuesong Li; Jie Hong; Yizhou Wang; Chang Wen Chen; Wentao Zhu
>
> **摘要:** Egocentric assistants often rely on first-person view data to capture user behavior and context for personalized services. Since different users exhibit distinct habits, preferences, and routines, such personalization is essential for truly effective assistance. However, effectively integrating long-term user data for personalization remains a key challenge. To address this, we introduce EgoSelf, a system that includes a graph-based interaction memory constructed from past observations and a dedicated learning task for personalization. The memory captures temporal and semantic relationships among interaction events and entities, from which user-specific profiles are derived. The personalized learning task is formulated as a prediction problem where the model predicts possible future interactions from individual user's historical behavior recorded in the graph. Extensive experiments demonstrate the effectiveness of EgoSelf as a personalized egocentric assistant. Code is available at this https URL.
>
---
#### [replaced 044] LLM-as-Judge Framework for Evaluating Tone-Induced Hallucination in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.18803](https://arxiv.org/pdf/2604.18803)**

> **作者:** Zhiyuan Jiang; Weihao Hong; Xinlei Guan; Tejaswi Dhandu; Miles Q. Li; Meng Xu; Kuan Huang; Umamaheswara Rao Tida; Bingyu Shen; Daehan Kwak; Boyang Li
>
> **备注:** 23 pages, 12 figures
>
> **摘要:** Vision-Language Models (VLMs) are increasingly deployed in settings where reliable visual grounding carries operational consequences, yet their behavior under progressively coercive prompt phrasing remains undercharacterized. Existing hallucination benchmarks predominantly rely on neutral prompts and binary detection, leaving open how both the incidence and the intensity of fabrication respond to graded linguistic pressure across structurally distinct task types. We present Ghost-100, a procedurally constructed benchmark of 800 synthetically generated images spanning eight categories across three task families: text-illegibility, time-reading, and object-absence, each designed under a negative-ground-truth principle that guarantees the queried target is absent, illegible, or indeterminate by construction. Every image is paired with five prompts drawn from a structured 5-Level Prompt Intensity Framework, holding the image and task identity fixed while varying only directive force, so that tone is isolated as the sole independent variable. We adopt a dual-track evaluation protocol: a rule-based H-Rate measuring the proportion of responses in which a model crosses from grounded refusal into unsupported positive commitment, and a GPT-4o-mini-judged H-Score on a 1-5 scale characterizing the confidence and specificity of fabrication once it occurs. We additionally release a three-stage automated validation workflow, which retrospectively confirms 717 of 800 images as strictly compliant. Evaluating nine open-weight VLMs, we find that H-Rate and H-Score dissociate substantially across model families, reading-style and presence-detection subsets respond to prompt pressure in qualitatively different ways, and several models exhibit non-monotonic sensitivity peaking at intermediate tone levels: patterns that aggregate metrics obscure.
>
---
#### [replaced 045] Confidence-Based Mesh Extraction from 3D Gaussians
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2603.24725](https://arxiv.org/pdf/2603.24725)**

> **作者:** Lukas Radl; Felix Windisch; Andreas Kurz; Thomas Köhler; Michael Steiner; Markus Steinberger
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS) greatly accelerated mesh extraction from posed images due to its explicit representation and fast software rasterization. While the addition of geometric losses and other priors has improved the accuracy of extracted surfaces, mesh extraction remains difficult in scenes with abundant view-dependent effects. To resolve the resulting ambiguities, prior works rely on multi-view techniques, iterative mesh extraction, or large pre-trained models, sacrificing the inherent efficiency of 3DGS. In this work, we present a simple and efficient alternative by introducing a self-supervised confidence framework to 3DGS: within this framework, learnable confidence values dynamically balance photometric and geometric supervision. Extending our confidence-driven formulation, we introduce losses which penalize per-primitive color and normal variance and demonstrate their benefits to surface extraction. Finally, we complement the above with an improved appearance model, by decoupling the individual terms of the D-SSIM loss. Our final approach delivers state-of-the-art results for unbounded meshes while remaining highly efficient.
>
---
#### [replaced 046] Evaluation of Winning Solutions of 2025 Low Power Computer Vision Challenge
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.19054](https://arxiv.org/pdf/2604.19054)**

> **作者:** Zihao Ye; Yung-Hsiang Lu; Xiao Hu; Shuai Zhang; Taotao Jing; Xin Li; Zhen Yao; Bo Lang; Zhihao Zheng; Seungmin Oh; Hankyul Kang; Seunghun Kang; Jongbin Ryu; Kexin Chen; Yuan Qi; George K Thiruvathukal; Mooi Choo Chuah
>
> **备注:** 11 pages, 8 figures, 4 tables
>
> **摘要:** The IEEE Low-Power Computer Vision Challenge (LPCVC) aims to promote the development of efficient vision models for edge devices, balancing accuracy with constraints such as latency, memory capacity, and energy use. The 2025 challenge featured three tracks: (1) Image classification under various lighting conditions and styles, (2) Open-Vocabulary Segmentation with Text Prompt, and (3) Monocular Depth Estimation. This paper presents the design of LPCVC 2025, including its competition structure and evaluation framework, which integrates the Qualcomm AI Hub for consistent and reproducible benchmarking. The paper also introduces the top-performing solutions from each track and outlines key trends and observations. The paper concludes with suggestions for future computer vision competitions.
>
---
#### [replaced 047] 3D Smoke Scene Reconstruction Guided by Vision Priors from Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.05687](https://arxiv.org/pdf/2604.05687)**

> **作者:** Xinye Zheng; Fei Wang; Yiqi Nie; Kun Li; Junjie Chen; Jiaqi Zhao; Yanyan Wei; Zhiliang Wu
>
> **摘要:** Reconstructing 3D scenes from smoke-degraded multi-view images is particularly difficult because smoke introduces strong scattering effects, view-dependent appearance changes, and severe degradation of cross-view consistency. To address these issues, we propose a framework that integrates visual priors with efficient 3D scene modeling. We employ Nano-Banana-Pro to enhance smoke-degraded images and provide clearer visual observations for reconstruction and develop Smoke-GS, a medium-aware 3D Gaussian Splatting framework for smoke scene reconstruction and restoration-oriented novel view synthesis. Smoke-GS models the scene using explicit 3D Gaussians and introduces a lightweight view-dependent medium branch to capture direction-dependent appearance variations caused by smoke. Our method preserves the rendering efficiency of 3D Gaussian Splatting while improving robustness to smoke-induced degradation. Results demonstrate the effectiveness of our method for generating consistent and visually clear novel views in challenging smoke environments.
>
---
#### [replaced 048] Location-Aware Pretraining for Medical Difference Visual Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.04950](https://arxiv.org/pdf/2603.04950)**

> **作者:** Denis Musinguzi; Caren Han; Prasenjit Mitra
>
> **备注:** 11 pages
>
> **摘要:** Differential medical VQA models compare multiple images to identify clinically meaningful changes and rely on vision encoders to capture fine-grained visual differences that reflect radiologists' comparative diagnostic workflows. However, vision encoders trained using standard contrastive or classification objectives often fail to capture the subtle variations needed to distinguish true disease progression from acquisition-related variability. To address this limitation, we introduce a location-aware pretraining framework that incorporates automatic referring expressions (AREF), grounded captioning (GCAP), and conditional automatic referring expressions (CAREF). These tasks promote the learning of fine-grained, spatially grounded visual representations. When integrated with a language model, our approach achieves state-of-the-art performance on medical difference VQA by accurately identifying and reasoning about clinically relevant changes in chest X-ray images.
>
---
#### [replaced 049] Sampling-Aware Quantization for Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.02242](https://arxiv.org/pdf/2505.02242)**

> **作者:** Qian Zeng; Jie Song; Yuanyu Wan; Huiqiong Wang; Mingli Song
>
> **备注:** 17 pages, 12 figures, CVPR2026 accepted
>
> **摘要:** Diffusion models have recently emerged as the dominant approach in visual generation tasks. However, the lengthy denoising chains and the computationally intensive noise estimation networks hinder their applicability in low-latency and resource-limited environments. Previous research has endeavored to address these limitations in a decoupled manner, utilizing either advanced samplers or efficient model quantization techniques. In this study, we uncover that quantization-induced noise disrupts directional estimation at each sampling step, further distorting the precise directional estimations of higher-order samplers when solving the sampling equations through discretized numerical methods, thereby altering the optimal sampling trajectory. To attain dual acceleration with high fidelity, we propose a sampling-aware quantization strategy, wherein a Mixed-Order Trajectory Alignment technique is devised to impose a more stringent constraint on the error bounds at each sampling step, facilitating a more linear probability flow. Extensive experiments on sparse-step fast sampling across multiple datasets demonstrate that our approach preserves the rapid convergence characteristics of high-speed samplers while maintaining superior generation quality. Code is publicly available at: this https URL.
>
---
#### [replaced 050] MSLAU-Net: A Hybrid CNN-Transformer Network for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.18823](https://arxiv.org/pdf/2505.18823)**

> **作者:** Libin Lan; Yanxin Li; Xiaojuan Liu; Juan Zhou; Jianxun Zhang; Nannan Huang; Yudong Zhang
>
> **备注:** 15 pages, 7 figures, 9 tables
>
> **摘要:** Accurate medical image segmentation allows for the precise delineation of anatomical structures and pathological regions, which is essential for treatment planning, surgical navigation, and disease monitoring. Both CNN-based and Transformer-based methods have achieved remarkable success in medical image segmentation tasks. However, CNN-based methods struggle to effectively capture global contextual information due to the inherent limitations of convolution operations. Meanwhile, Transformer-based methods suffer from insufficient local feature modeling and face challenges related to the high computational complexity caused by the self-attention mechanism. To address these limitations, we propose a novel hybrid CNN-Transformer architecture, named MSLAU-Net, which integrates the strengths of both paradigms. The proposed MSLAU-Net incorporates two key ideas. First, it introduces Multi-Scale Linear Attention, designed to efficiently extract multi-scale features from medical images while modeling long-range dependencies with low computational complexity. Second, it adopts a top-down feature aggregation mechanism, which performs multi-level feature aggregation and restores spatial resolution using a lightweight structure. Extensive experiments conducted on benchmark datasets covering three imaging modalities demonstrate that the proposed MSLAU-Net outperforms other state-of-the-art methods on nearly all evaluation metrics, validating the superiority, effectiveness, and robustness of our this http URL code is available at this https URL.
>
---
#### [replaced 051] A Synchronized Audio-Visual Multi-View Capture System
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23089](https://arxiv.org/pdf/2603.23089)**

> **作者:** Xiangwei Shi; Gara Dorta; Ruud de Jong; Ojas Shirekar; Chirag Raman
>
> **摘要:** Multi-view capture systems have been an important tool in research for recording human motion under controlling conditions. Most existing systems are specified around video streams and provide little or no support for audio acquisition and rigorous audio-video alignment, despite both being essential for studying conversational interaction where timing at the level of turn-taking, overlap, and prosody matters. In this technical report, we describe an audio-visual multi-view capture system that addresses this gap by treating synchronized audio and synchronized video as first-class signals. The system combines a multi-camera pipeline with multi-channel microphone recording under a unified timing architecture and provides a practical workflow for calibration, acquisition, and quality control that supports repeatable recordings at scale. We quantify synchronization performance in deployment and show that the resulting recordings are temporally consistent enough to support fine-grained analysis and data-driven modeling of conversation behavior.
>
---
#### [replaced 052] Integrated AI Nodule Detection and Diagnosis for Lung Cancer Screening Beyond Size and Growth-Based Standards Compared with Radiologists and Leading Models
- **分类: cs.CV; q-bio.NC**

- **链接: [https://arxiv.org/pdf/2512.00281](https://arxiv.org/pdf/2512.00281)**

> **作者:** Sylvain Bodard; Pierre Baudot; Benjamin Renoust; Charles Voyton; Gwendoline De Bie; Ezequiel Geremia; Van-Khoa Le; Danny Francis; Pierre-Henri Siot; Yousra Haddou; Vincent Bobin; Jean-Christophe Brisset; Carey C. Thomson; Valerie Bourdes; Benoit Huet
>
> **备注:** 25 pages, 8 figures, with supplementary information containing 11 figures
>
> **摘要:** Early detection of malignant lung nodules remains limited by reliance on size- and growth-based screening criteria, which can delay diagnosis. We present an integrated AI system that - unlike conventional CADe or CADx approaches - jointly performs nodule detection and malignancy assessment directly at the nodule level from low-dose CT scans within a unified aided decision framework. To address limitations in dataset scale and explainability, we designed an ensemble of shallow deep learning and feature-based specialized models, trained and evaluated on 25,709 scans with 69,449 annotated nodules, with external validation on an independent cohort. The system achieves an area under the receiver operating characteristic curve (AUC) of 0.98 internally and 0.945 on an independent cohort, outperforming radiologists and leading AI models (Sybil, Brock, Google, Kaggle). With a sensitivity of 99.3 percent at 0.5 false positives per scan, it addresses key barriers to AI adoption and demonstrates improved performance relative to both Lung-RADS size-based triage and European volume- and VDT-based screening criteria. The model outperforms radiologists across all nodule sizes and cancer stages - excelling in stage I cancers - and across all growth-based metrics, including volume-doubling time. It also surpasses radiologists by up to one year in diagnosing indeterminate and slow-growing nodules.
>
---
#### [replaced 053] From Diffusion to Flow: Efficient Motion Generation in MotionGPT3
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.26747](https://arxiv.org/pdf/2603.26747)**

> **作者:** Jaymin Ban; JiHong Jeon; SangYeop Jeong
>
> **备注:** ReALM-GEN Workshop ICLR 2026
>
> **摘要:** Recent text-driven motion generation methods span both discrete token-based approaches and continuous-latent formulations. MotionGPT3 exemplifies the latter paradigm, combining a learned continuous motion latent space with a diffusion-based prior for text-conditioned synthesis. While rectified flow objectives have recently demonstrated favorable convergence and inference-time properties relative to diffusion in image and audio generation, it remains unclear whether these advantages transfer cleanly to the motion generation setting. In this work, we conduct a controlled empirical study comparing diffusion and rectified flow objectives within the MotionGPT3 framework. By holding the model architecture, training protocol, and evaluation setup fixed, we isolate the effect of the generative objective on training dynamics, final performance, and inference efficiency. Experiments on the HumanML3D dataset show that rectified flow converges in fewer training epochs, reaches strong test performance earlier, and matches or exceeds diffusion-based motion quality under identical conditions. Moreover, flow-based priors exhibit stable behavior across a wide range of inference step counts and achieve competitive quality with fewer sampling steps, yielding improved efficiency-quality trade-offs. Overall, our results suggest that several known benefits of rectified flow objectives do extend to continuous-latent text-to-motion generation, highlighting the importance of the training objective choice in motion priors.
>
---
#### [replaced 054] Learn2Synth: Learning Optimal Data Synthesis Using Hypergradients for Brain Image Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.16719](https://arxiv.org/pdf/2411.16719)**

> **作者:** Xiaoling Hu; Xiangrui Zeng; Oula Puonti; Juan Eugenio Iglesias; Bruce Fischl; Yael Balbastre
>
> **备注:** 16 pages, 5 figures. Accepted by ICCV'25. Bruce Fischl and Yael Balbastre are co-senior authors
>
> **摘要:** Domain randomization through synthesis is a powerful strategy to train networks that are unbiased with respect to the domain of the input images. Randomization allows networks to see a virtually infinite range of intensities and artifacts during training, thereby minimizing overfitting to appearance and maximizing generalization to unseen data. Although powerful, this approach relies on the accurate tuning of a large set of hyperparameters that govern the probabilistic distribution of the synthesized images. Instead of manually tuning these parameters, we introduce Learn2Synth, a novel procedure in which synthesis parameters are learned using a small set of real labeled data. Unlike methods that impose constraints to align synthetic data with real data (e.g., contrastive or adversarial techniques), which risk misaligning the image and its label map, we tune an augmentation engine such that a segmentation network trained on synthetic data has optimal accuracy when applied to real data. This approach allows the training procedure to benefit from real labeled examples, without ever using these real examples to train the segmentation network, which avoids biasing the network towards the properties of the training set. Specifically, we develop parametric and nonparametric strategies to enhance synthetic images in a way that improves the performance of the segmentation network. We demonstrate the effectiveness of this learning strategy on synthetic and real-world brain scans. Code is available at: this https URL.
>
---
#### [replaced 055] Evolvable Embodied Agent for Robotic Manipulation via Long Short-Term Reflection and Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EEAgent框架，解决机器人任务适应与进化问题，利用大模型和LSTRO机制提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.13533](https://arxiv.org/pdf/2604.13533)**

> **作者:** Jianzong Wang; Botao Zhao; Yayun He; Junqing Peng; Xulong Zhang
>
> **备注:** This work has been accepted for publication in the Proceedings of the 2026 International Joint Conference on Neural Networks (IJCNN 2026)
>
> **摘要:** Achieving general-purpose robotics requires empowering robots to adapt and evolve based on their environment and feedback. Traditional methods face limitations such as extensive training requirements, difficulties in cross-task generalization, and lack of interpretability. Prompt learning offers new opportunities for self-evolving robots without extensive training, but simply reflecting on past experiences. However, extracting meaningful insights from task successes and failures remains a challenge. To this end, we propose the evolvable embodied agent (EEAgent) framework, which leverages large vision-language models (VLMs) for better environmental interpretation and policy planning. To enhance reflection on past experiences, we propose a long short-term reflective optimization (LSTRO) mechanism that dynamically refines prompts based on both past experiences and newly learned lessons, facilitating continuous self-evolution, thereby enhancing overall task success rates. Evaluations on six VIMA-Bench tasks reveal that our approach sets a new state-of-the-art, notably outperforming baselines in complex scenarios.
>
---
#### [replaced 056] IVY-FAKE: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.00979](https://arxiv.org/pdf/2506.00979)**

> **作者:** Changjiang Jiang; Wenhui Dong; Zhonghao Zhang; Fengchang Yu; Wei Peng; Xinbin Yuan; Yifei Bi; Ming Zhao; Zian Zhou; Chenyang Si; Caifeng Shan
>
> **备注:** 30 pages
>
> **摘要:** The rapid development of Artificial Intelligence Generated Content (AIGC) techniques has enabled the creation of high-quality synthetic content, but it also raises significant security concerns. Current detection methods face two major limitations: (1) the lack of multidimensional explainable datasets for generated images and videos. Existing open-source datasets (e.g., WildFake, GenVideo) rely on oversimplified binary annotations, which restrict the explainability and trustworthiness of trained detectors. (2) Prior MLLM-based forgery detectors (e.g., FakeVLM) exhibit insufficiently fine-grained interpretability in their step-by-step reasoning, which hinders reliable localization and explanation. To address these challenges, we introduce Ivy-Fake, the first large-scale multimodal benchmark for explainable AIGC detection. It consists of over 106K richly annotated training samples (images and videos) and 5,000 manually verified evaluation examples, sourced from multiple generative models and real world datasets through a carefully designed pipeline to ensure both diversity and quality. Furthermore, we propose Ivy-xDetector, a reinforcement learning model based on Group Relative Policy Optimization (GRPO), capable of producing explainable reasoning chains and achieving robust performance across multiple synthetic content detection benchmarks. Extensive experiments demonstrate the superiority of our dataset and confirm the effectiveness of our approach. Notably, our method improves performance on GenImage from 86.88% to 96.32%, surpassing prior state-of-the-art methods by a clear margin.
>
---
#### [replaced 057] Foundation Models in Biomedical Imaging: Turning Hype into Reality
- **分类: q-bio.QM; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.15808](https://arxiv.org/pdf/2512.15808)**

> **作者:** Amgad Muneer; Kai Zhang; Ibraheem Hamdi; Rizwan Qureshi; Muhammad Waqas; Shereen Fouad; Hazrat Ali; Syed Muhammad Anwar; Jia Wu
>
> **备注:** 9 figures and 3 tables
>
> **摘要:** Foundation models (FMs) are driving a prominent shift in biomedical imaging from task-specific models to unified backbone models for diverse tasks. This opens an avenue to integrate imaging, pathology, clinical records, and genomics data into a composite system. However, this vision contrasts sharply with modern medicine's trajectory toward more granular sub-specialization. This tension, coupled with data scarcity, domain heterogeneity, and limited interpretability, creates a gap between benchmark success and real-world clinical value. We argue that the immediate role of FMs lies in augmenting, not replacing, clinical expertise. To separate hype from reality, we introduce REAL-FM (Real-world Evaluation and Assessment of Foundation Models), a multi-dimensional framework for assessing data, technical readiness, clinical value, workflow integration, and responsible AI. Using REAL-FM, we find that while FMs excel in pattern recognition, they fall short in causal reasoning, domain robustness, and safety. Clinical translation is hindered by scarce representative data for model training, unverified generalization beyond oversimplified benchmark settings, and a lack of prospective outcome-based validation. We further examine FM reasoning paradigms, including sequential logic, spatial understanding, and symbolic domain knowledge. We envision that the path forward lies not in a monolithic medical oracle, but in coordinated subspecialist AI systems that are transparent, safe, and clinically grounded.
>
---
#### [replaced 058] CoRe: Joint Optimization with Contrastive Learning for Medical Image Registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23694](https://arxiv.org/pdf/2603.23694)**

> **作者:** Eytan Kats; Christoph Grossbroehmer; Ziad Al-Haj Hemidi; Fenja Falta; Wiebke Heyer; Mattias P. Heinrich
>
> **备注:** Preprint
>
> **摘要:** Medical image registration is a fundamental task in medical image analysis, enabling the alignment of images from different modalities or time points. However, intensity inconsistencies and nonlinear tissue deformations pose significant challenges to the robustness of registration methods. Recent approaches leveraging self-supervised representation learning show promise by pre-training feature extractors to generate robust anatomical embeddings, that farther used for the registration. In this work, we propose a novel framework that integrates equivariant contrastive learning directly into the registration model. Our approach leverages the power of contrastive learning to learn robust feature representations that are invariant to tissue deformations. By jointly optimizing the contrastive and registration objectives, we ensure that the learned representations are not only informative but also suitable for the registration task. We evaluate our method on abdominal and thoracic image registration tasks, including both intra-patient and inter-patient scenarios. Experimental results demonstrate that the integration of contrastive learning directly into the registration framework significantly improves performance, surpassing strong baseline methods.
>
---
#### [replaced 059] SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08730](https://arxiv.org/pdf/2512.08730)**

> **作者:** Kaiyu Li; Shengqi Zhang; Yujie Wang; Yupeng Deng; Zhi Wang; Deyu Meng; Xiangyong Cao
>
> **摘要:** Most existing methods for training-free open-vocabulary semantic segmentation are based on CLIP. While these approaches have made progress, they often face challenges in precise localization or require complex pipelines to combine separate modules, especially in remote sensing scenarios where numerous dense and small targets are present. Recently, Segment Anything Model 3 (SAM 3) was proposed, unifying segmentation and recognition in a promptable framework. In this paper, we present a comprehensive exploration of applying SAM 3 to the remote sensing open-vocabulary tasks (i.e., 2D semantic segmentation, change detection, and 3D semantic segmentation) without any training. First, we implement a mask fusion strategy that combines the outputs from SAM 3's semantic segmentation head and the Transformer decoder (instance head). This allows us to leverage the strengths of both heads for better land coverage. Second, we utilize the presence score from the presence head to filter out categories that do not exist in the scene, reducing false positives caused by the vast vocabulary sizes and patch-level processing in geospatial scenes. Furthermore, we extend our method to open-vocabulary change detection by a joint instance- and pixel-level verification strategy built directly upon our fused logits. We evaluate our method on extensive remote sensing datasets and tasks, including 20 segmentation datasets, 3 change detection datasets, and a 3D segmentation dataset. Experiments show that our method achieves promising performance, demonstrating the potential of SAM 3 for remote sensing open-vocabulary tasks. Our code is released at this https URL.
>
---
#### [replaced 060] From Competition to Synergy: Unlocking Reinforcement Learning for Subject-Driven Image Generation
- **分类: cs.LG; cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2510.18263](https://arxiv.org/pdf/2510.18263)**

> **作者:** Ziwei Huang; Ying Shu; Hao Fang; Quanyu Long; Wenya Wang; Qiushi Guo; Tiezheng Ge; Leilei Gan
>
> **摘要:** Subject-driven image generation models face a fundamental trade-off between identity preservation (fidelity) and prompt adherence (editability). While online reinforcement learning (RL), specifically GPRO, offers a promising solution, we find that a naive application of GRPO leads to competitive degradation, as the simple linear aggregation of rewards with static weights causes conflicting gradient signals and a misalignment with the temporal dynamics of the diffusion process. To overcome these limitations, we propose Customized-GRPO, a novel framework featuring two key innovations: (i) Synergy-Aware Reward Shaping (SARS), a non-linear mechanism that explicitly penalizes conflicted reward signals and amplifies synergistic ones, providing a sharper and more decisive gradient. (ii) Time-Aware Dynamic Weighting (TDW), which aligns the optimization pressure with the model's temporal dynamics by prioritizing prompt-following in the early, identity preservation in the later. Extensive experiments demonstrate that our method significantly outperforms naive GRPO baselines, successfully mitigating competitive degradation. Our model achieves a superior balance, generating images that both preserve key identity features and accurately adhere to complex textual prompts.
>
---
#### [replaced 061] PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20537](https://arxiv.org/pdf/2602.20537)**

> **作者:** Xinyong Cai; Changbin Sun; Yong Wang; Hongyu Yang; Yuankai Wu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Spatiotemporal predictive learning (STPL) aims to forecast future frames from past observations and is essential across a wide range of applications. Compared with recurrent or hybrid architectures, pure convolutional models offer superior efficiency and full parallelism, yet their fixed receptive fields limit their ability to adaptively capture spatially varying motion patterns. Inspired by biological center-surround organization and frequency-selective signal processing, we propose PFGNet, a fully convolutional framework that dynamically modulates receptive fields through pixel-wise frequency-guided gating. The core Peripheral Frequency Gating (PFG) block extracts localized spectral cues and adaptively fuses multi-scale large-kernel peripheral responses with learnable center suppression, effectively forming spatially adaptive band-pass filters. To maintain efficiency, all large kernels are decomposed into separable 1D convolutions ($1 \times k$ followed by $k \times 1$), reducing per-channel computational cost from $O(k^2)$ to $O(2k)$. PFGNet enables structure-aware spatiotemporal modeling without recurrence or attention. Experiments on Moving MNIST, TaxiBJ, Human3.6M, and KTH show that PFGNet delivers SOTA or near-SOTA forecasting performance with substantially fewer parameters and FLOPs. Our code is available at this https URL.
>
---
#### [replaced 062] CXR-LanIC: Language-Grounded Interpretable Classifier for Chest X-Ray Diagnosis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21464](https://arxiv.org/pdf/2510.21464)**

> **作者:** Yiming Tang; Wenjia Zhong; Rushi Shah; Dianbo Liu
>
> **摘要:** Deep learning models have achieved remarkable accuracy in chest X-ray diagnosis, yet their widespread clinical adoption remains limited by the black-box nature of their predictions. Clinicians require transparent, verifiable explanations to trust automated diagnoses and identify potential failure modes. We introduce CXR-LanIC (Language-Grounded Interpretable Classifier for Chest X-rays), a novel framework that addresses this interpretability challenge through task-aligned pattern discovery. Our approach trains transcoder-based sparse autoencoders on a BiomedCLIP diagnostic classifier to decompose medical image representations into interpretable visual patterns. By training an ensemble of 100 transcoders on multimodal embeddings from the MIMIC-CXR dataset, we discover approximately 5,000 monosemantic patterns spanning cardiac, pulmonary, pleural, structural, device, and artifact categories. Each pattern exhibits consistent activation behavior across images sharing specific radiological features, enabling transparent attribution where predictions decompose into 20-50 interpretable patterns with verifiable activation galleries. CXR-LanIC achieves competitive diagnostic accuracy on five key findings while providing the foundation for natural language explanations through planned large multimodal model annotation. Our key innovation lies in extracting interpretable features from a classifier trained on specific diagnostic objectives rather than general-purpose embeddings, ensuring discovered patterns are directly relevant to clinical decision-making, demonstrating that medical AI systems can be both accurate and interpretable, supporting safer clinical deployment through transparent, clinically grounded explanations.
>
---
#### [replaced 063] BARD: Bridging AutoRegressive and Diffusion Vision-Language Models Via Highly Efficient Progressive Block Merging and Stage-Wise Distillation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.16514](https://arxiv.org/pdf/2604.16514)**

> **作者:** Baoyou Chen; Hanchen Xia; Peng Tu; Haojun Shi; Shan Mu; Weihao Yuan; Siyu Zhu
>
> **摘要:** Autoregressive vision-language models (VLMs) deliver strong multimodal capability, but their token-by-token decoding imposes a fundamental inference bottleneck. Diffusion VLMs offer a more parallel decoding paradigm, yet directly converting a pretrained autoregressive VLM into a large-block diffusion VLM (dVLM) often leads to substantial quality degradation. In this work, we present BARD, a simple and effective bridging framework that converts a pretrained autoregressive VLM into a same-architecture, decoding-efficient dVLM. Our approach combines progressive supervised block merging, which gradually enlarges the decoding block size, with stage-wise intra-dVLM distillation from a fixed small-block diffusion anchor to recover performance lost at larger blocks. We further incorporate a mixed noise scheduler to improve robustness and token revision during denoising, and memory-friendly training to enable efficient training on long multimodal sequences. A key empirical finding is that direct autoregressive-to-diffusion distillation is poorly aligned and can even hurt performance, whereas distillation within the diffusion regime is consistently effective. Experimental results show that, with $\leq$ 4.4M data, BARD-VL transfers strong multimodal capability from Qwen3-VL to a large-block dVLM. Remarkably, BARD-VL establishes a new SOTA among comparable-scale open dVLMs on our evaluation suite at both 4B and 8B scales. At the same time, BARD-VL achieves up to 3$\times$ decoding throughput speedup compared to the source model. Code is available at: $\href{this https URL}{this~https~URL}$.
>
---
#### [replaced 064] Semantic-guided Gaussian Splatting for High-Fidelity Underwater Scene Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.00800](https://arxiv.org/pdf/2509.00800)**

> **作者:** Zhuodong Jiang; Haoran Wang; Guoxi Huang; Brett Seymour; Nantheera Anantrasirichai
>
> **摘要:** Accurate 3D reconstruction in degraded imaging conditions remains a key challenge in photogrammetry and neural rendering. In underwater environments, spatially varying visibility caused by scattering, attenuation, and sparse observations leads to highly non-uniform information quality. Existing 3D Gaussian Splatting (3DGS) methods typically optimize primitives based on photometric signals alone, resulting in imbalanced representation, with overfitting in well-observed regions and insufficient reconstruction in degraded areas. In this paper, we propose SWAGSplatting (Semantic-guided Water-scene Augmented Gaussian Splatting), a multimodal framework that integrates semantic priors into 3DGS for robust, high-fidelity underwater reconstruction. Each Gaussian primitive is augmented with a learnable semantic feature, supervised by CLIP-based embeddings derived from region-level cues. A semantic consistency loss is introduced to align geometric reconstruction with high-level semantics, improving structural coherence and preserving salient object boundaries under challenging conditions. Furthermore, we propose an adaptive Gaussian primitive reallocation strategy that redistributes representation capacity based on both primitive importance and reconstruction error, mitigating the imbalance introduced by conventional densification. This enables more effective modeling of low-visibility regions without increasing computational cost. Extensive experiments on real-world datasets, including SeaThru-NeRF, Submerged3D, and S-UW, demonstrate that the proposed method consistently outperforms state-of-the-art approaches in terms of average PSNR, SSIM, and LPIPS. The results validate the effectiveness of integrating semantic priors for high-fidelity underwater scene reconstruction. Code is available at this https URL.
>
---
#### [replaced 065] The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20714](https://arxiv.org/pdf/2603.20714)**

> **作者:** Ivan Desiatov; Torsten Sattler
>
> **备注:** Sources are available at this https URL . Changes in this version: fixed wrong graphs being used in Fig. 6 (b), Fig. 10 (a,c,d) due to compilation issue; results with EDGS* are now using splat scale increase when reducing init. size (previously reported results without scale increase, but conclusions remain unchanged)
>
> **摘要:** 3D Gaussian Splatting (3DGS) has become the method of choice for photo-realistic 3D reconstruction of scenes, due to being able to efficiently and accurately recover the scene appearance and geometry from images. 3DGS represents the scene through a set of 3D Gaussians, parameterized by their position, spatial extent, and view-dependent color. Starting from an initial point cloud, 3DGS refines the Gaussians' parameters as to reconstruct a set of training images as accurately as possible. Typically, a sparse Structure-from-Motion point cloud is used as initialization. In order to obtain dense Gaussian clouds, 3DGS methods thus rely on a densification stage. In this paper, we systematically study the relation between densification and initialization. Proposing a new benchmark, we study combinations of different types of initializations (dense laser scans, dense (multi-view) stereo point clouds, dense monocular depth estimates, sparse SfM point clouds) and different densification schemes. We show that current densification approaches are not able to take full advantage of dense initialization as they are often unable to (significantly) improve over sparse SfM-based initialization. We will make our benchmark publicly available.
>
---
