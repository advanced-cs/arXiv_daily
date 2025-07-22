# 计算机视觉 cs.CV

- **最新发布 189 篇**

- **更新 133 篇**

## 最新发布

#### [new 001] Hierarchical Cross-modal Prompt Learning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决预训练模型在下游任务中因模态隔离和语义衰减导致的泛化能力不足问题。论文提出HiCroPL框架，通过层次化跨模态提示学习，实现文本与视觉信息的双向交互与语义互补，提升了模型表现，取得11项基准的最优结果。**

- **链接: [http://arxiv.org/pdf/2507.14976v1](http://arxiv.org/pdf/2507.14976v1)**

> **作者:** Hao Zheng; Shunzhi Yang; Zhuoxin He; Jinfeng Yang; Zhenhua Huang
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Pre-trained Vision-Language Models (VLMs) such as CLIP have shown excellent generalization abilities. However, adapting these large-scale models to downstream tasks while preserving their generalization capabilities remains challenging. Although prompt learning methods have shown promise, they suffer from two fundamental bottlenecks that limit generalization: (a) modality isolation, and (b) hierarchical semantic decay. To address these limitations, we propose HiCroPL, a Hierarchical Cross-modal Prompt Learning framework that establishes bidirectional knowledge flow between text and vision modalities, enabling them to refine their semantics mutually. HiCroPL routes knowledge flows by leveraging the complementary strengths of text and vision. In early layers, text prompts inject relatively clear semantics into visual prompts through a hierarchical knowledge mapper, enhancing the representation of low-level visual semantics. In later layers, visual prompts encoding specific task-relevant objects flow back to refine text prompts, enabling deeper alignment. Crucially, our hierarchical knowledge mapper allows representations at multi-scales to be fused, ensuring that deeper representations retain transferable shallow semantics thereby enhancing generalization. We further introduce a lightweight layer-specific knowledge proxy to enable efficient cross-modal interactions. Extensive evaluations across four tasks demonstrate HiCroPL's superior performance, achieving state-of-the-art results on 11 benchmarks with significant improvements. Code is available at: https://github.com/zzeoZheng/HiCroPL.
>
---
#### [new 002] A Practical Investigation of Spatially-Controlled Image Generation with Transformers
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在提升基于Transformer的图像生成模型的空间控制能力。论文通过对比不同生成范式，提出控制标记预填充方法，并探索采样策略对生成一致性的影响，澄清适配器方法的作用与局限。**

- **链接: [http://arxiv.org/pdf/2507.15724v1](http://arxiv.org/pdf/2507.15724v1)**

> **作者:** Guoxuan Xia; Harleen Hanspal; Petru-Daniel Tudosiu; Shifeng Zhang; Sarah Parisot
>
> **备注:** preprint
>
> **摘要:** Enabling image generation models to be spatially controlled is an important area of research, empowering users to better generate images according to their own fine-grained specifications via e.g. edge maps, poses. Although this task has seen impressive improvements in recent times, a focus on rapidly producing stronger models has come at the cost of detailed and fair scientific comparison. Differing training data, model architectures and generation paradigms make it difficult to disentangle the factors contributing to performance. Meanwhile, the motivations and nuances of certain approaches become lost in the literature. In this work, we aim to provide clear takeaways across generation paradigms for practitioners wishing to develop transformer-based systems for spatially-controlled generation, clarifying the literature and addressing knowledge gaps. We perform controlled experiments on ImageNet across diffusion-based/flow-based and autoregressive (AR) models. First, we establish control token prefilling as a simple, general and performant baseline approach for transformers. We then investigate previously underexplored sampling time enhancements, showing that extending classifier-free guidance to control, as well as softmax truncation, have a strong impact on control-generation consistency. Finally, we re-clarify the motivation of adapter-based approaches, demonstrating that they mitigate "forgetting" and maintain generation quality when trained on limited downstream data, but underperform full training in terms of generation-control consistency. Code will be released upon publication.
>
---
#### [new 003] Language Integration in Fine-Tuning Multimodal Large Language Models for Image-Based Regression
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像回归任务，旨在解决多模态大语言模型（MLLMs）在图像评估中未能有效利用文本语义的问题。作者提出RvTC方法，通过基于分类的回归框架和数据特定提示，提升模型性能，并在多个数据集上取得最优结果。**

- **链接: [http://arxiv.org/pdf/2507.14997v1](http://arxiv.org/pdf/2507.14997v1)**

> **作者:** Roy H. Jennings; Genady Paikin; Roy Shaul; Evgeny Soloveichik
>
> **摘要:** Multimodal Large Language Models (MLLMs) show promise for image-based regression tasks, but current approaches face key limitations. Recent methods fine-tune MLLMs using preset output vocabularies and generic task-level prompts (e.g., "How would you rate this image?"), assuming this mimics human rating behavior. Our analysis reveals these approaches provide no benefit over image-only training. Models using preset vocabularies and generic prompts perform equivalently to image-only models, failing to leverage semantic understanding from textual input. We propose Regression via Transformer-Based Classification (RvTC), which replaces vocabulary-constrained classification with a flexible bin-based approach. Unlike approaches that address discretization errors through complex distributional modeling, RvTC eliminates manual vocabulary crafting through straightforward bin increase, achieving state-of-the-art performance on four image assessment datasets using only images. More importantly, we demonstrate that data-specific prompts dramatically improve performance. Unlike generic task descriptions, prompts containing semantic information about specific images enable MLLMs to leverage cross-modal understanding. On the AVA dataset, adding challenge titles to prompts improves correlations from 0.83 to 0.90, a new state-of-the-art. We demonstrate through empirical evidence from the AVA and AGIQA-3k datasets that MLLMs benefit from semantic prompt information surpassing mere statistical biases. This underscores the importance of incorporating meaningful textual context in multimodal regression tasks.
>
---
#### [new 004] Light Future: Multimodal Action Frame Prediction via InstructPix2Pix
- **分类: cs.CV; cs.MM; cs.RO; I.2.10; I.4.8**

- **简介: 该论文属于机器人动作预测任务，旨在解决未来视觉帧预测的高效性与轻量化问题。论文提出基于InstructPix2Pix的多模态预测框架，仅需单张图像和文本指令即可预测未来100帧的视觉变化。通过微调InstructPix2Pix模型实现图文联合输入，在RoboTWin数据集上取得了优于现有方法的性能，具备低计算成本和快速推理优势。**

- **链接: [http://arxiv.org/pdf/2507.14809v1](http://arxiv.org/pdf/2507.14809v1)**

> **作者:** Zesen Zhong; Duomin Zhang; Yijia Li
>
> **备注:** 9 pages including appendix, 5 tables, 8 figures, to be submitted to WACV 2026
>
> **摘要:** Predicting future motion trajectories is a critical capability across domains such as robotics, autonomous systems, and human activity forecasting, enabling safer and more intelligent decision-making. This paper proposes a novel, efficient, and lightweight approach for robot action prediction, offering significantly reduced computational cost and inference latency compared to conventional video prediction models. Importantly, it pioneers the adaptation of the InstructPix2Pix model for forecasting future visual frames in robotic tasks, extending its utility beyond static image editing. We implement a deep learning-based visual prediction framework that forecasts what a robot will observe 100 frames (10 seconds) into the future, given a current image and a textual instruction. We repurpose and fine-tune the InstructPix2Pix model to accept both visual and textual inputs, enabling multimodal future frame prediction. Experiments on the RoboTWin dataset (generated based on real-world scenarios) demonstrate that our method achieves superior SSIM and PSNR compared to state-of-the-art baselines in robot action prediction tasks. Unlike conventional video prediction models that require multiple input frames, heavy computation, and slow inference latency, our approach only needs a single image and a text prompt as input. This lightweight design enables faster inference, reduced GPU demands, and flexible multimodal control, particularly valuable for applications like robotics and sports motion trajectory analytics, where motion trajectory precision is prioritized over visual fidelity.
>
---
#### [new 005] Compress-Align-Detect: onboard change detection from unregistered images
- **分类: cs.CV; eess.IV**

- **简介: 论文提出“Compress-Align-Detect”框架，用于卫星上实时变化检测。该任务属于遥感图像处理，旨在解决因图像下传和地面处理延迟导致的检测滞后问题。工作包括：设计轻量级图像压缩、多时相图像配准和高效变化检测子模块，集成于单一流程中，并在低功耗硬件上验证性能。**

- **链接: [http://arxiv.org/pdf/2507.15578v1](http://arxiv.org/pdf/2507.15578v1)**

> **作者:** Gabriele Inzerillo; Diego Valsesia; Aniello Fiengo; Enrico Magli
>
> **摘要:** Change detection from satellite images typically incurs a delay ranging from several hours up to days because of latency in downlinking the acquired images and generating orthorectified image products at the ground stations; this may preclude real- or near real-time applications. To overcome this limitation, we propose shifting the entire change detection workflow onboard satellites. This requires to simultaneously solve challenges in data storage, image registration and change detection with a strict complexity constraint. In this paper, we present a novel and efficient framework for onboard change detection that addresses the aforementioned challenges in an end-to-end fashion with a deep neural network composed of three interlinked submodules: (1) image compression, tailored to minimize onboard data storage resources; (2) lightweight co-registration of non-orthorectified multi-temporal image pairs; and (3) a novel temporally-invariant and computationally efficient change detection model. This is the first approach in the literature combining all these tasks in a single end-to-end framework with the constraints dictated by onboard processing. Experimental results compare each submodule with the current state-of-the-art, and evaluate the performance of the overall integrated system in realistic setting on low-power hardware. Compelling change detection results are obtained in terms of F1 score as a function of compression rate, sustaining a throughput of 0.7 Mpixel/s on a 15W accelerator.
>
---
#### [new 006] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决多模态大模型中时空信息融合不足的问题。作者提出DynImg方法，利用非关键帧作为视觉提示，引导模型关注快速运动区域，并通过4D旋转位置嵌入保持时空顺序，提升了视频理解效果。**

- **链接: [http://arxiv.org/pdf/2507.15569v1](http://arxiv.org/pdf/2507.15569v1)**

> **作者:** Xiaoyi Bao; Chenwei Xie; Hao Tang; Tingyu Weng; Xiaofeng Wang; Yun Zheng; Xingang Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** In recent years, the introduction of Multi-modal Large Language Models (MLLMs) into video understanding tasks has become increasingly prevalent. However, how to effectively integrate temporal information remains a critical research focus. Traditional approaches treat spatial and temporal information separately. Due to issues like motion blur, it is challenging to accurately represent the spatial information of rapidly moving objects. This can lead to temporally important regions being underemphasized during spatial feature extraction, which in turn hinders accurate spatio-temporal interaction and video understanding. To address this limitation, we propose an innovative video representation method called Dynamic-Image (DynImg). Specifically, we introduce a set of non-key frames as temporal prompts to highlight the spatial areas containing fast-moving objects. During the process of visual feature extraction, these prompts guide the model to pay additional attention to the fine-grained spatial features corresponding to these regions. Moreover, to maintain the correct sequence for DynImg, we employ a corresponding 4D video Rotary Position Embedding. This retains both the temporal and spatial adjacency of DynImg, helping MLLM understand the spatio-temporal order within this combined format. Experimental evaluations reveal that DynImg surpasses the state-of-the-art methods by approximately 2% across multiple video understanding benchmarks, proving the effectiveness of our temporal prompts in enhancing video comprehension.
>
---
#### [new 007] One Last Attention for Your Vision-Language Model
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型适配任务，旨在解决现有方法忽视融合表示的问题。作者提出RA达方法，通过注意力机制动态调整融合矩阵，提升模型性能，适用于多种适配场景。**

- **链接: [http://arxiv.org/pdf/2507.15480v1](http://arxiv.org/pdf/2507.15480v1)**

> **作者:** Liang Chen; Ghazi Shazan Ahmad; Tianjun Yao; Lingqiao Liu; Zhiqiang Shen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Pretrained vision-language models (VLMs), such as CLIP, achieve remarkable zero-shot performance, yet their downstream potential hinges on effective fine-tuning. Most adaptation methods typically focus on refining representation from separate modalities (text or vision) but neglect the critical role of their fused representations in the decision-making process, \emph{\ie} rational matrix that drives the final prediction. To bridge the gap, we propose a simple yet effective \textbf{R}ational \textbf{Ada}ptaion ({RAda}) to explicitly exploit the final fused representation during fine-tuning. RAda employs a learned mask, obtained from a lightweight attention layer attached at the end of a VLM, to dynamically calibrate the contribution of each element in the rational matrix, enabling targeted adjustments to the final cross-modal interactions without incurring costly modifications to intermediate features. Experiments in different settings (i.e., updating, or freezing pretrained encoders in adaptation, and test-time training that can only access the unlabeled test data) show that RAda serves as a versatile fine-tuning technique, improving the baseline with minimal code and performing comparably against current arts in most settings. Code is available at \href{https://github.com/khufia/RAda/tree/main}{github.com/khufia/RAda}.
>
---
#### [new 008] 3-Dimensional CryoEM Pose Estimation and Shift Correction Pipeline
- **分类: cs.CV**

- **简介: 该论文属于冷冻电镜（cryo-EM）图像处理任务，旨在解决低信噪比下三维重构中的姿态估计与平移校正问题。作者提出了一种基于多维缩放和鲁棒优化的管道，改进旋转矩阵估计与平移校正，提升低信噪比下的重构精度。**

- **链接: [http://arxiv.org/pdf/2507.14924v1](http://arxiv.org/pdf/2507.14924v1)**

> **作者:** Kaishva Chintan Shah; Virajith Boddapati; Karthik S. Gurumoorthy; Sandip Kaledhonkar; Ajit Rajwade
>
> **摘要:** Accurate pose estimation and shift correction are key challenges in cryo-EM due to the very low SNR, which directly impacts the fidelity of 3D reconstructions. We present an approach for pose estimation in cryo-EM that leverages multi-dimensional scaling (MDS) techniques in a robust manner to estimate the 3D rotation matrix of each particle from pairs of dihedral angles. We express the rotation matrix in the form of an axis of rotation and a unit vector in the plane perpendicular to the axis. The technique leverages the concept of common lines in 3D reconstruction from projections. However, common line estimation is ridden with large errors due to the very low SNR of cryo-EM projection images. To address this challenge, we introduce two complementary components: (i) a robust joint optimization framework for pose estimation based on an $\ell_1$-norm objective or a similar robust norm, which simultaneously estimates rotation axes and in-plane vectors while exactly enforcing unit norm and orthogonality constraints via projected coordinate descent; and (ii) an iterative shift correction algorithm that estimates consistent in-plane translations through a global least-squares formulation. While prior approaches have leveraged such embeddings and common-line geometry for orientation recovery, existing formulations typically rely on $\ell_2$-based objectives that are sensitive to noise, and enforce geometric constraints only approximately. These choices, combined with a sequential pipeline structure, can lead to compounding errors and suboptimal reconstructions in low-SNR regimes. Our pipeline consistently outperforms prior methods in both Euler angle accuracy and reconstruction fidelity, as measured by the Fourier Shell Correlation (FSC).
>
---
#### [new 009] Latent Denoising Makes Good Visual Tokenizers
- **分类: cs.CV**

- **简介: 该论文属于视觉生成模型中的 tokenizer 设计任务，旨在提升视觉 tokenizer 在生成建模中的有效性。作者提出 Latent Denoising Tokenizer (l-DeTok)，通过在潜空间中引入去噪目标，使 token embeddings 更鲁棒、更易重建。实验表明该方法在多个生成模型中表现优于标准 tokenizer。**

- **链接: [http://arxiv.org/pdf/2507.15856v1](http://arxiv.org/pdf/2507.15856v1)**

> **作者:** Jiawei Yang; Tianhong Li; Lijie Fan; Yonglong Tian; Yue Wang
>
> **备注:** Code is available at: https://github.com/Jiawei-Yang/DeTok
>
> **摘要:** Despite their fundamental role, it remains unclear what properties could make visual tokenizers more effective for generative modeling. We observe that modern generative models share a conceptually similar training objective -- reconstructing clean signals from corrupted inputs such as Gaussian noise or masking -- a process we term denoising. Motivated by this insight, we propose aligning tokenizer embeddings directly with the downstream denoising objective, encouraging latent embeddings to be more easily reconstructed even when heavily corrupted. To achieve this, we introduce the Latent Denoising Tokenizer (l-DeTok), a simple yet effective tokenizer trained to reconstruct clean images from latent embeddings corrupted by interpolative noise and random masking. Extensive experiments on ImageNet 256x256 demonstrate that our tokenizer consistently outperforms standard tokenizers across six representative generative models. Our findings highlight denoising as a fundamental design principle for tokenizer development, and we hope it could motivate new perspectives for future tokenizer design.
>
---
#### [new 010] Artificial Intelligence in the Food Industry: Food Waste Estimation based on Computer Vision, a Brief Case Study in a University Dining Hall
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分割任务，旨在解决大学食堂食物浪费量化问题。通过计算机视觉，使用U-Net等模型对餐前餐后食物图像进行语义分割，实现餐盘级食物浪费估计，并提出DPA指标评估效果。**

- **链接: [http://arxiv.org/pdf/2507.14662v1](http://arxiv.org/pdf/2507.14662v1)**

> **作者:** Shayan Rokhva; Babak Teimourpour
>
> **备注:** Questions & Recommendations: shayanrokhva1999@gmail.com; shayan1999rokh@yahoo.com
>
> **摘要:** Quantifying post-consumer food waste in institutional dining settings is essential for supporting data-driven sustainability strategies. This study presents a cost-effective computer vision framework that estimates plate-level food waste by utilizing semantic segmentation of RGB images taken before and after meal consumption across five Iranian dishes. Four fully supervised models (U-Net, U-Net++, and their lightweight variants) were trained using a capped dynamic inverse-frequency loss and AdamW optimizer, then evaluated through a comprehensive set of metrics, including Pixel Accuracy, Dice, IoU, and a custom-defined Distributional Pixel Agreement (DPA) metric tailored to the task. All models achieved satisfying performance, and for each food type, at least one model approached or surpassed 90% DPA, demonstrating strong alignment in pixel-wise proportion estimates. Lighter models with reduced parameter counts offered faster inference, achieving real-time throughput on an NVIDIA T4 GPU. Further analysis showed superior segmentation performance for dry and more rigid components (e.g., rice and fries), while more complex, fragmented, or viscous dishes, such as stews, showed reduced performance, specifically post-consumption. Despite limitations such as reliance on 2D imaging, constrained food variety, and manual data collection, the proposed framework is pioneering and represents a scalable, contactless solution for continuous monitoring of food consumption. This research lays foundational groundwork for automated, real-time waste tracking systems in large-scale food service environments and offers actionable insights and outlines feasible future directions for dining hall management and policymakers aiming to reduce institutional food waste.
>
---
#### [new 011] Cross-Domain Few-Shot Learning with Coalescent Projections and Latent Space Reservation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于跨域少样本学习任务，旨在解决因更新过多参数导致的过拟合问题。作者提出了一种新的方法，包括合并投影和基于自监督变换的伪类别生成，以提高模型在不同域上的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.15243v1](http://arxiv.org/pdf/2507.15243v1)**

> **作者:** Naeem Paeedeh; Mahardhika Pratama; Wolfgang Mayer; Jimmy Cao; Ryszard Kowlczyk
>
> **摘要:** Despite the progress in Cross-Domain Few-Shot Learning (CD-FSL), a model pre-trained with DINO combined with a prototypical classifier outperforms the latest SOTA methods. A crucial limitation that needs to be overcome is that updating too many parameters of the transformers leads to overfitting due to the scarcity of labeled samples. To address this challenge, we propose a new concept, Coalescent Projection (CP), as an effective successor to soft prompts. Additionally, we propose a novel pseudo-class generation method combined with Self-Supervised Transformations (SSTs) that relies solely on the base domain to prepare the network for encountering unseen samples from different domains. The proposed method exhibits its effectiveness in comprehensive experiments on the extreme domain shift scenario of the BSCD-FSL benchmark. Our code is published at https://github.com/Naeem-Paeedeh/CPLSR.
>
---
#### [new 012] Descrip3D: Enhancing Large Language Model-based 3D Scene Understanding with Object-Level Text Descriptions
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决现有模型在理解物体间空间与语义关系上的不足。论文提出Descrip3D框架，通过引入物体的文本描述增强关系理解，并采用双级融合方法整合语言信息，提升多任务性能。**

- **链接: [http://arxiv.org/pdf/2507.14555v1](http://arxiv.org/pdf/2507.14555v1)**

> **作者:** Jintang Xue; Ganning Zhao; Jie-En Yao; Hong-En Chen; Yue Hu; Meida Chen; Suya You; C. -C. Jay Kuo
>
> **摘要:** Understanding 3D scenes goes beyond simply recognizing objects; it requires reasoning about the spatial and semantic relationships between them. Current 3D scene-language models often struggle with this relational understanding, particularly when visual embeddings alone do not adequately convey the roles and interactions of objects. In this paper, we introduce Descrip3D, a novel and powerful framework that explicitly encodes the relationships between objects using natural language. Unlike previous methods that rely only on 2D and 3D embeddings, Descrip3D enhances each object with a textual description that captures both its intrinsic attributes and contextual relationships. These relational cues are incorporated into the model through a dual-level integration: embedding fusion and prompt-level injection. This allows for unified reasoning across various tasks such as grounding, captioning, and question answering, all without the need for task-specific heads or additional supervision. When evaluated on five benchmark datasets, including ScanRefer, Multi3DRefer, ScanQA, SQA3D, and Scan2Cap, Descrip3D consistently outperforms strong baseline models, demonstrating the effectiveness of language-guided relational representation for understanding complex indoor scenes.
>
---
#### [new 013] Hallucination Score: Towards Mitigating Hallucinations in Generative Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决生成模型中出现的“幻觉”问题，即生成细节与低分辨率或真实图像不匹配。论文提出了一种基于多模态大语言模型的“幻觉评分”（HS），用于量化此类问题，并探索通过特征对齐缓解幻觉的方法。**

- **链接: [http://arxiv.org/pdf/2507.14367v1](http://arxiv.org/pdf/2507.14367v1)**

> **作者:** Weiming Ren; Raghav Goyal; Zhiming Hu; Tristan Ty Aumentado-Armstrong; Iqbal Mohomed; Alex Levinshtein
>
> **备注:** 12 pages, 17 figures and 7 tables
>
> **摘要:** Generative super-resolution (GSR) currently sets the state-of-the-art in terms of perceptual image quality, overcoming the "regression-to-the-mean" blur of prior non-generative models. However, from a human perspective, such models do not fully conform to the optimal balance between quality and fidelity. Instead, a different class of artifacts, in which generated details fail to perceptually match the low resolution image (LRI) or ground-truth image (GTI), is a critical but under studied issue in GSR, limiting its practical deployments. In this work, we focus on measuring, analyzing, and mitigating these artifacts (i.e., "hallucinations"). We observe that hallucinations are not well-characterized with existing image metrics or quality models, as they are orthogonal to both exact fidelity and no-reference quality. Instead, we take advantage of a multimodal large language model (MLLM) by constructing a prompt that assesses hallucinatory visual elements and generates a "Hallucination Score" (HS). We find that our HS is closely aligned with human evaluations, and also provides complementary insights to prior image metrics used for super-resolution (SR) models. In addition, we find certain deep feature distances have strong correlations with HS. We therefore propose to align the GSR models by using such features as differentiable reward functions to mitigate hallucinations.
>
---
#### [new 014] A Survey on Efficiency Optimization Techniques for DNN-based Video Analytics: Process Systems, Algorithms, and Applications
- **分类: cs.CV**

- **简介: 该论文属于综述任务，旨在系统梳理提升基于深度神经网络（DNN）的视频分析效率的优化技术。论文从硬件支持、数据处理、操作部署等多个角度总结了现有方法，并分析了当前性能优化中存在的问题与挑战。**

- **链接: [http://arxiv.org/pdf/2507.15628v1](http://arxiv.org/pdf/2507.15628v1)**

> **作者:** Shanjiang Tang; Rui Huang; Hsinyu Luo; Chunjiang Wang; Ce Yu; Yusen Li; Hao Fu; Chao Sun; and Jian Xiao
>
> **摘要:** The explosive growth of video data in recent years has brought higher demands for video analytics, where accuracy and efficiency remain the two primary concerns. Deep neural networks (DNNs) have been widely adopted to ensure accuracy; however, improving their efficiency in video analytics remains an open challenge. Different from existing surveys that make summaries of DNN-based video mainly from the accuracy optimization aspect, in this survey, we aim to provide a thorough review of optimization techniques focusing on the improvement of the efficiency of DNNs in video analytics. We organize existing methods in a bottom-up manner, covering multiple perspectives such as hardware support, data processing, operational deployment, etc. Finally, based on the optimization framework and existing works, we analyze and discuss the problems and challenges in the performance optimization of DNN-based video analytics.
>
---
#### [new 015] DCHM: Depth-Consistent Human Modeling for Multiview Detection
- **分类: cs.CV**

- **简介: 该论文属于多视角行人检测任务，旨在解决现有方法在噪声、精度及泛化能力方面的问题。作者提出DCHM框架，通过超像素级高斯点云实现深度一致性估计，提升多视角融合与行人定位精度，无需依赖人工标注。实验表明其在复杂场景下优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.14505v1](http://arxiv.org/pdf/2507.14505v1)**

> **作者:** Jiahao Ma; Tianyu Wang; Miaomiao Liu; David Ahmedt-Aristizabal; Chuong Nguyen
>
> **备注:** multi-view detection, sparse-view reconstruction
>
> **摘要:** Multiview pedestrian detection typically involves two stages: human modeling and pedestrian localization. Human modeling represents pedestrians in 3D space by fusing multiview information, making its quality crucial for detection accuracy. However, existing methods often introduce noise and have low precision. While some approaches reduce noise by fitting on costly multiview 3D annotations, they often struggle to generalize across diverse scenes. To eliminate reliance on human-labeled annotations and accurately model humans, we propose Depth-Consistent Human Modeling (DCHM), a framework designed for consistent depth estimation and multiview fusion in global coordinates. Specifically, our proposed pipeline with superpixel-wise Gaussian Splatting achieves multiview depth consistency in sparse-view, large-scaled, and crowded scenarios, producing precise point clouds for pedestrian localization. Extensive validations demonstrate that our method significantly reduces noise during human modeling, outperforming previous state-of-the-art baselines. Additionally, to our knowledge, DCHM is the first to reconstruct pedestrians and perform multiview segmentation in such a challenging setting. Code is available on the \href{https://jiahao-ma.github.io/DCHM/}{project page}.
>
---
#### [new 016] Multispectral State-Space Feature Fusion: Bridging Shared and Cross-Parametric Interactions for Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多光谱目标检测任务，旨在解决现有方法对局部互补特征偏好过强及感受野与计算复杂度间权衡的问题。作者提出MS2Fusion框架，基于状态空间模型，通过双路径参数交互机制实现高效融合。实验表明其在多个基准上优于现有方法，并具有良好的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.14643v1](http://arxiv.org/pdf/2507.14643v1)**

> **作者:** Jifeng Shen; Haibo Zhan; Shaohua Dong; Xin Zuo; Wankou Yang; Haibin Ling
>
> **备注:** submitted on 30/4/2025, Under Major Revision
>
> **摘要:** Modern multispectral feature fusion for object detection faces two critical limitations: (1) Excessive preference for local complementary features over cross-modal shared semantics adversely affects generalization performance; and (2) The trade-off between the receptive field size and computational complexity present critical bottlenecks for scalable feature modeling. Addressing these issues, a novel Multispectral State-Space Feature Fusion framework, dubbed MS2Fusion, is proposed based on the state space model (SSM), achieving efficient and effective fusion through a dual-path parametric interaction mechanism. More specifically, the first cross-parameter interaction branch inherits the advantage of cross-attention in mining complementary information with cross-modal hidden state decoding in SSM. The second shared-parameter branch explores cross-modal alignment with joint embedding to obtain cross-modal similar semantic features and structures through parameter sharing in SSM. Finally, these two paths are jointly optimized with SSM for fusing multispectral features in a unified framework, allowing our MS2Fusion to enjoy both functional complementarity and shared semantic space. In our extensive experiments on mainstream benchmarks including FLIR, M3FD and LLVIP, our MS2Fusion significantly outperforms other state-of-the-art multispectral object detection methods, evidencing its superiority. Moreover, MS2Fusion is general and applicable to other multispectral perception tasks. We show that, even without specific design, MS2Fusion achieves state-of-the-art results on RGB-T semantic segmentation and RGBT salient object detection, showing its generality. The source code will be available at https://github.com/61s61min/MS2Fusion.git.
>
---
#### [new 017] Visual Place Recognition for Large-Scale UAV Applications
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉地点识别（vPR）任务，旨在解决无人机在大范围、高海拔环境下定位难的问题。作者提出了LASED大规模数据集，并引入可转向卷积神经网络，以提升模型对旋转变化的鲁棒性，显著提高了识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.15089v1](http://arxiv.org/pdf/2507.15089v1)**

> **作者:** Ioannis Tsampikos Papapetros; Ioannis Kansizoglou; Antonios Gasteratos
>
> **摘要:** Visual Place Recognition (vPR) plays a crucial role in Unmanned Aerial Vehicle (UAV) navigation, enabling robust localization across diverse environments. Despite significant advancements, aerial vPR faces unique challenges due to the limited availability of large-scale, high-altitude datasets, which limits model generalization, along with the inherent rotational ambiguity in UAV imagery. To address these challenges, we introduce LASED, a large-scale aerial dataset with approximately one million images, systematically sampled from 170,000 unique locations throughout Estonia over a decade, offering extensive geographic and temporal diversity. Its structured design ensures clear place separation significantly enhancing model training for aerial scenarios. Furthermore, we propose the integration of steerable Convolutional Neural Networks (CNNs) to explicitly handle rotational variance, leveraging their inherent rotational equivariance to produce robust, orientation-invariant feature representations. Our extensive benchmarking demonstrates that models trained on LASED achieve significantly higher recall compared to those trained on smaller, less diverse datasets, highlighting the benefits of extensive geographic coverage and temporal diversity. Moreover, steerable CNNs effectively address rotational ambiguity inherent in aerial imagery, consistently outperforming conventional convolutional architectures, achieving on average 12\% recall improvement over the best-performing non-steerable network. By combining structured, large-scale datasets with rotation-equivariant neural networks, our approach significantly enhances model robustness and generalization for aerial vPR.
>
---
#### [new 018] Extracting Visual Facts from Intermediate Layers for Mitigating Hallucinations in Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLM）任务，旨在解决模型生成内容时出现的物体幻觉问题。作者提出了一种无需训练的方法EVA，通过提取中间层中的视觉事实信息，动态选择关键层并结合原始与纯文本输入的输出对比，纠正最终输出。方法有效降低了幻觉率，适用于多种模型和解码策略。**

- **链接: [http://arxiv.org/pdf/2507.15652v1](http://arxiv.org/pdf/2507.15652v1)**

> **作者:** Haoran Zhou; Zihan Zhang; Hao Chen
>
> **摘要:** Multimodal Large Language Models (MLLMs) have made significant strides by combining visual recognition and language understanding to generate content that is both coherent and contextually accurate. However, MLLMs continue to struggle with object hallucinations, where models produce seemingly plausible but factually incorrect outputs, including objects that do not exist in the image. Recent work has revealed that the prior knowledge in MLLMs significantly suppresses visual information in deep layers, causing hallucinatory outputs. However, how these priors suppress visual information at the intermediate layer stage in MLLMs remains unclear. We observe that visual factual knowledge and the differences between intermediate-layer prior/original probability distributions show similar evolutionary trends in intermediate layers. Motivated by this, we introduce Decoding by Extracting Visual Facts (EVA), a simple, training-free method that dynamically selects intermediate layers with the most significant visual factual information. By contrasting the output distributions of the selected layer derived from the original input and pure-text input, EVA extracts visual factual knowledge and proportionally incorporates it into the final layer to correct the output logits. Importantly, EVA is model-agnostic, seamlessly integrates with various classic decoding strategies, and is applicable across different MLLMs. We validate EVA on widely-used benchmarks, and the results show that it significantly reduces hallucination rates compared to baseline methods, underscoring its effectiveness in mitigating hallucinations.
>
---
#### [new 019] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有主体驱动生成方法依赖训练的问题。作者提出了FreeCus，一种无需训练的框架，通过注意力共享机制、改进的特征提取和多模态语义表示，激活扩散变换器的零样本生成能力，实现高质量的主体定制化合成。**

- **链接: [http://arxiv.org/pdf/2507.15249v1](http://arxiv.org/pdf/2507.15249v1)**

> **作者:** Yanbing Zhang; Zhe Wang; Qin Zhou; Mengping Yang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** In light of recent breakthroughs in text-to-image (T2I) generation, particularly with diffusion transformers (DiT), subject-driven technologies are increasingly being employed for high-fidelity customized production that preserves subject identity from reference inputs, enabling thrilling design workflows and engaging entertainment. Existing alternatives typically require either per-subject optimization via trainable text embeddings or training specialized encoders for subject feature extraction on large-scale datasets. Such dependencies on training procedures fundamentally constrain their practical applications. More importantly, current methodologies fail to fully leverage the inherent zero-shot potential of modern diffusion transformers (e.g., the Flux series) for authentic subject-driven synthesis. To bridge this gap, we propose FreeCus, a genuinely training-free framework that activates DiT's capabilities through three key innovations: 1) We introduce a pivotal attention sharing mechanism that captures the subject's layout integrity while preserving crucial editing flexibility. 2) Through a straightforward analysis of DiT's dynamic shifting, we propose an upgraded variant that significantly improves fine-grained feature extraction. 3) We further integrate advanced Multimodal Large Language Models (MLLMs) to enrich cross-modal semantic representations. Extensive experiments reflect that our method successfully unlocks DiT's zero-shot ability for consistent subject synthesis across diverse contexts, achieving state-of-the-art or comparable results compared to approaches that require additional training. Notably, our framework demonstrates seamless compatibility with existing inpainting pipelines and control modules, facilitating more compelling experiences. Our code is available at: https://github.com/Monalissaa/FreeCus.
>
---
#### [new 020] Probabilistic smooth attention for deep multiple instance learning in medical imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决标签稀缺下的多实例学习问题。现有方法通过注意力机制聚合实例特征，但忽视了注意力值的不确定性。论文提出一种概率注意力框架，建模注意力分布，同时考虑全局与局部交互，提升了分类性能并提供可解释的不确定性图。**

- **链接: [http://arxiv.org/pdf/2507.14932v1](http://arxiv.org/pdf/2507.14932v1)**

> **作者:** Francisco M. Castro-Macías; Pablo Morales-Álvarez; Yunan Wu; Rafael Molina; Aggelos K. Katsaggelos
>
> **摘要:** The Multiple Instance Learning (MIL) paradigm is attracting plenty of attention in medical imaging classification, where labeled data is scarce. MIL methods cast medical images as bags of instances (e.g. patches in whole slide images, or slices in CT scans), and only bag labels are required for training. Deep MIL approaches have obtained promising results by aggregating instance-level representations via an attention mechanism to compute the bag-level prediction. These methods typically capture both local interactions among adjacent instances and global, long-range dependencies through various mechanisms. However, they treat attention values deterministically, potentially overlooking uncertainty in the contribution of individual instances. In this work we propose a novel probabilistic framework that estimates a probability distribution over the attention values, and accounts for both global and local interactions. In a comprehensive evaluation involving {\color{review} eleven} state-of-the-art baselines and three medical datasets, we show that our approach achieves top predictive performance in different metrics. Moreover, the probabilistic treatment of the attention provides uncertainty maps that are interpretable in terms of illness localization.
>
---
#### [new 021] TokensGen: Harnessing Condensed Tokens for Long Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决长视频生成中的内存瓶颈和长期不一致问题。论文提出TokensGen框架，通过压缩视频块为语义令牌，分阶段生成长视频，确保内容连贯与平滑过渡，提升生成效率与质量。**

- **链接: [http://arxiv.org/pdf/2507.15728v1](http://arxiv.org/pdf/2507.15728v1)**

> **作者:** Wenqi Ouyang; Zeqi Xiao; Danni Yang; Yifan Zhou; Shuai Yang; Lei Yang; Jianlou Si; Xingang Pan
>
> **备注:** Project page: https://vicky0522.github.io/tokensgen-webpage/
>
> **摘要:** Generating consistent long videos is a complex challenge: while diffusion-based generative models generate visually impressive short clips, extending them to longer durations often leads to memory bottlenecks and long-term inconsistency. In this paper, we propose TokensGen, a novel two-stage framework that leverages condensed tokens to address these issues. Our method decomposes long video generation into three core tasks: (1) inner-clip semantic control, (2) long-term consistency control, and (3) inter-clip smooth transition. First, we train To2V (Token-to-Video), a short video diffusion model guided by text and video tokens, with a Video Tokenizer that condenses short clips into semantically rich tokens. Second, we introduce T2To (Text-to-Token), a video token diffusion transformer that generates all tokens at once, ensuring global consistency across clips. Finally, during inference, an adaptive FIFO-Diffusion strategy seamlessly connects adjacent clips, reducing boundary artifacts and enhancing smooth transitions. Experimental results demonstrate that our approach significantly enhances long-term temporal and content coherence without incurring prohibitive computational overhead. By leveraging condensed tokens and pre-trained short video models, our method provides a scalable, modular solution for long video generation, opening new possibilities for storytelling, cinematic production, and immersive simulations. Please see our project page at https://vicky0522.github.io/tokensgen-webpage/ .
>
---
#### [new 022] True Multimodal In-Context Learning Needs Attention to the Visual Context
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态语言模型任务，旨在解决当前模型在多模态上下文学习（MICL）中忽视视觉信息的问题。作者提出DARA微调策略，增强模型对视觉内容的关注，并构建TrueMICL数据集，专门用于评估真正需要多模态理解的上下文学习能力。**

- **链接: [http://arxiv.org/pdf/2507.15807v1](http://arxiv.org/pdf/2507.15807v1)**

> **作者:** Shuo Chen; Jianzhe Liu; Zhen Han; Yan Xia; Daniel Cremers; Philip Torr; Volker Tresp; Jindong Gu
>
> **备注:** accepted to COLM 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs), built on powerful language backbones, have enabled Multimodal In-Context Learning (MICL)-adapting to new tasks from a few multimodal demonstrations consisting of images, questions, and answers. Despite showing noticeable improvement on standard vision-language datasets, current MLLMs struggle to leverage visual information in the demonstrations. Specifically, they tend to neglect visual cues and over-rely on textual patterns, leading to mere text imitation rather than genuine multimodal adaptation. This behavior makes MICL still unimodal and largely restricts its practical utility. More importantly, this limitation is often concealed by the improved performance on tasks that do not require understanding the visual context. As a result, how to effectively enhance MICL ability and reliably evaluate the MICL performance remains underexplored. To address these issues, we first introduce Dynamic Attention Reallocation (DARA), an efficient fine-tuning strategy that encourages models to attend to the visual context by rebalancing attention across visual and textual tokens. In addition, we present TrueMICL, an MICL-dedicated dataset with both support and test sets that explicitly requires the integration of multimodal information-particularly visual content-for correct task completion. Extensive experiments demonstrate the effectiveness of our holistic solution, showcasing substantial improvements in the true multimodal in-context learning capabilities. Code and datasets are available at https://chenxshuo.github.io/true-micl-colm .
>
---
#### [new 023] Enhancing Visual Planning with Auxiliary Tasks and Multi-token Prediction
- **分类: cs.CV**

- **简介: 论文研究视觉规划任务，旨在解决视频中长期动作预测的数据稀缺与动作结构建模问题。作者提出VideoPlan，通过辅助任务增强训练，并采用多token预测提升动作序列预测效果，在COIN和CrossTask数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.15130v1](http://arxiv.org/pdf/2507.15130v1)**

> **作者:** Ce Zhang; Yale Song; Ruta Desai; Michael Louis Iuzzolino; Joseph Tighe; Gedas Bertasius; Satwik Kottur
>
> **摘要:** Visual Planning for Assistance (VPA) aims to predict a sequence of user actions required to achieve a specified goal based on a video showing the user's progress. Although recent advances in multimodal large language models (MLLMs) have shown promising results in video understanding, long-horizon visual planning remains a challenging problem. We identify two challenges in training large MLLMs for video-based planning tasks: (1) scarcity of procedural annotations, limiting the model's ability to learn procedural task dynamics effectively, and (2) inefficiency of next-token prediction objective to explicitly capture the structured action space for visual planning when compared to free-form, natural language. To tackle data scarcity, we introduce Auxiliary Task Augmentation. We design and train our model on auxiliary tasks relevant to long-horizon video-based planning (e.g., goal prediction) to augment the model's planning ability. To more explicitly model the structured action space unique to visual planning tasks, we leverage Multi-token Prediction, extending traditional next-token prediction by using multiple heads to predict multiple future tokens during training. Our approach, VideoPlan, achieves state-of-the-art VPA performance on the COIN and CrossTask datasets, surpassing prior methods by 7.3% and 3.4%, respectively, when predicting 3 future actions. We further extend our method to the challenging Ego4D Long-term Action Anticipation task, and show that it is on par with the state-of-the-art approaches despite not using specialized egocentric features. Code will be made available.
>
---
#### [new 024] IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决真实场景红外图像因缺乏对齐文本数据和领域特性带来的理解难题。作者构建了包含26万对真实红外图像与文本的数据集IR-TD，并提出IRGPT模型与双跨模态课程学习策略，显著提升红外图像理解性能。**

- **链接: [http://arxiv.org/pdf/2507.14449v1](http://arxiv.org/pdf/2507.14449v1)**

> **作者:** Zhe Cao; Jin Zhang; Ruiheng Zhang
>
> **备注:** 11 pages, 7 figures. This paper is accepted by ICCV 2025
>
> **摘要:** Real-world infrared imagery presents unique challenges for vision-language models due to the scarcity of aligned text data and domain-specific characteristics. Although existing methods have advanced the field, their reliance on synthetic infrared images generated through style transfer from visible images, which limits their ability to capture the unique characteristics of the infrared modality. To address this, we propose IRGPT, the first multi-modal large language model for real-world infrared images, built upon a large-scale InfraRed-Text Dataset (IR-TD) comprising over 260K authentic image-text pairs. The proposed IR-TD dataset contains real infrared images paired with meticulously handcrafted texts, where the initial drafts originated from two complementary processes: (1) LLM-generated descriptions of visible images, and (2) rule-based descriptions of annotations. Furthermore, we introduce a bi-cross-modal curriculum transfer learning strategy that systematically transfers knowledge from visible to infrared domains by considering the difficulty scores of both infrared-visible and infrared-text. Evaluated on a benchmark of 9 tasks (e.g., recognition, grounding), IRGPT achieves state-of-the-art performance even compared with larger-scale models.
>
---
#### [new 025] Rethinking Pan-sharpening: Principled Design, Unified Training, and a Universal Loss Surpass Brute-Force Scaling
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务中的全色锐化（pan-sharpening）研究。针对当前模型复杂、计算开销大且泛化能力差的问题，论文提出了轻量级模型PanTiny，采用统一训练策略和通用损失函数，提升了性能与效率，验证了设计优化优于盲目扩大模型。**

- **链接: [http://arxiv.org/pdf/2507.15059v1](http://arxiv.org/pdf/2507.15059v1)**

> **作者:** Ran Zhang; Xuanhua He; Li Xueheng; Ke Cao; Liu Liu; Wenbo Xu; Fang Jiabin; Yang Qize; Jie Zhang
>
> **摘要:** The field of pan-sharpening has recently seen a trend towards increasingly large and complex models, often trained on single, specific satellite datasets. This approach, however, leads to high computational overhead and poor generalization on full resolution data, a paradigm we challenge in this paper. In response to this issue, we propose PanTiny, a lightweight, single-step pan-sharpening framework designed for both efficiency and robust performance. More critically, we introduce multiple-in-one training paradigm, where a single, compact model is trained simultaneously on three distinct satellite datasets (WV2, WV3, and GF2) with different resolution and spectral information. Our experiments show that this unified training strategy not only simplifies deployment but also significantly boosts generalization on full-resolution data. Further, we introduce a universally powerful composite loss function that elevates the performance of almost all of models for pan-sharpening, pushing state-of-the-art metrics into a new era. Our PanTiny model, benefiting from these innovations, achieves a superior performance-to-efficiency balance, outperforming most larger, specialized models. Through extensive ablation studies, we validate that principled engineering in model design, training paradigms, and loss functions can surpass brute-force scaling. Our work advocates for a community-wide shift towards creating efficient, generalizable, and data-conscious models for pan-sharpening. The code is available at https://github.com/Zirconium233/PanTiny .
>
---
#### [new 026] Motion Segmentation and Egomotion Estimation from Event-Based Normal Flow
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于计算机视觉与机器人领域，旨在解决基于事件的运动分割与自运动估计问题。利用神经形态视觉传感器的稀疏事件数据，结合几何约束和惯性测量，提出一种优化框架，实现无需光流计算的准确运动分割与运动估计，适用于实时机器人应用。**

- **链接: [http://arxiv.org/pdf/2507.14500v1](http://arxiv.org/pdf/2507.14500v1)**

> **作者:** Zhiyuan Hua; Dehao Yuan; Cornelia Fermüller
>
> **摘要:** This paper introduces a robust framework for motion segmentation and egomotion estimation using event-based normal flow, tailored specifically for neuromorphic vision sensors. In contrast to traditional methods that rely heavily on optical flow or explicit depth estimation, our approach exploits the sparse, high-temporal-resolution event data and incorporates geometric constraints between normal flow, scene structure, and inertial measurements. The proposed optimization-based pipeline iteratively performs event over-segmentation, isolates independently moving objects via residual analysis, and refines segmentations using hierarchical clustering informed by motion similarity and temporal consistency. Experimental results on the EVIMO2v2 dataset validate that our method achieves accurate segmentation and translational motion estimation without requiring full optical flow computation. This approach demonstrates significant advantages at object boundaries and offers considerable potential for scalable, real-time robotic and navigation applications.
>
---
#### [new 027] SegDT: A Diffusion Transformer-Based Segmentation Model for Medical Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决皮肤病变区域的自动精准分割问题。作者提出了SegDT模型，基于扩散变压器并引入修正流，提升了生成质量与推理速度。实验表明其在多个数据集上达到先进水平，适用于低成本硬件，有助于医疗诊断。**

- **链接: [http://arxiv.org/pdf/2507.15595v1](http://arxiv.org/pdf/2507.15595v1)**

> **作者:** Salah Eddine Bekhouche; Gaby Maroun; Fadi Dornaika; Abdenour Hadid
>
> **摘要:** Medical image segmentation is crucial for many healthcare tasks, including disease diagnosis and treatment planning. One key area is the segmentation of skin lesions, which is vital for diagnosing skin cancer and monitoring patients. In this context, this paper introduces SegDT, a new segmentation model based on diffusion transformer (DiT). SegDT is designed to work on low-cost hardware and incorporates Rectified Flow, which improves the generation quality at reduced inference steps and maintains the flexibility of standard diffusion models. Our method is evaluated on three benchmarking datasets and compared against several existing works, achieving state-of-the-art results while maintaining fast inference speeds. This makes the proposed model appealing for real-world medical applications. This work advances the performance and capabilities of deep learning models in medical image analysis, enabling faster, more accurate diagnostic tools for healthcare professionals. The code is made publicly available at \href{https://github.com/Bekhouche/SegDT}{GitHub}.
>
---
#### [new 028] SAIGFormer: A Spatially-Adaptive Illumination-Guided Network for Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决非均匀光照场景（如逆光、阴影）下亮度恢复不均的问题。作者提出SAIGFormer框架，通过动态积分图像表示和空间自适应积分光照估计器（SAI²E）建模光照变化，并引入光照引导的多头自注意力机制（IG-MSA）优化亮度特征，实现更准确的光照增强。**

- **链接: [http://arxiv.org/pdf/2507.15520v1](http://arxiv.org/pdf/2507.15520v1)**

> **作者:** Hanting Li; Fei Zhou; Xin Sun; Yang Hua; Jungong Han; Liang-Jie Zhang
>
> **备注:** 11 pages, 10 figures, 6 tables
>
> **摘要:** Recent Transformer-based low-light enhancement methods have made promising progress in recovering global illumination. However, they still struggle with non-uniform lighting scenarios, such as backlit and shadow, appearing as over-exposure or inadequate brightness restoration. To address this challenge, we present a Spatially-Adaptive Illumination-Guided Transformer (SAIGFormer) framework that enables accurate illumination restoration. Specifically, we propose a dynamic integral image representation to model the spatially-varying illumination, and further construct a novel Spatially-Adaptive Integral Illumination Estimator ($\text{SAI}^2\text{E}$). Moreover, we introduce an Illumination-Guided Multi-head Self-Attention (IG-MSA) mechanism, which leverages the illumination to calibrate the lightness-relevant features toward visual-pleased illumination enhancement. Extensive experiments on five standard low-light datasets and a cross-domain benchmark (LOL-Blur) demonstrate that our SAIGFormer significantly outperforms state-of-the-art methods in both quantitative and qualitative metrics. In particular, our method achieves superior performance in non-uniform illumination enhancement while exhibiting strong generalization capabilities across multiple datasets. Code is available at https://github.com/LHTcode/SAIGFormer.git.
>
---
#### [new 029] Exploring Superposition and Interference in State-of-the-Art Low-Parameter Vision Models
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决低参数神经网络中特征干扰问题。通过研究瓶颈架构与超线性激活函数的关系，提出减少干扰的设计方法，并构建了NoDepth Bottleneck架构，在ImageNet数据集上验证其高效性和扩展性。**

- **链接: [http://arxiv.org/pdf/2507.15798v1](http://arxiv.org/pdf/2507.15798v1)**

> **作者:** Lilian Hollard; Lucas Mohimont; Nathalie Gaveau; Luiz-Angelo Steffenel
>
> **摘要:** The paper investigates the performance of state-of-the-art low-parameter deep neural networks for computer vision, focusing on bottleneck architectures and their behavior using superlinear activation functions. We address interference in feature maps, a phenomenon associated with superposition, where neurons simultaneously encode multiple characteristics. Our research suggests that limiting interference can enhance scaling and accuracy in very low-scaled networks (under 1.5M parameters). We identify key design elements that reduce interference by examining various bottleneck architectures, leading to a more efficient neural network. Consequently, we propose a proof-of-concept architecture named NoDepth Bottleneck built on mechanistic insights from our experiments, demonstrating robust scaling accuracy on the ImageNet dataset. These findings contribute to more efficient and scalable neural networks for the low-parameter range and advance the understanding of bottlenecks in computer vision. https://caiac.pubpub.org/pub/3dh6rsel
>
---
#### [new 030] GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 论文提出GEMINUS，用于端到端自动驾驶的双感知全局与场景自适应专家混合框架。该工作旨在解决复杂交通环境中单一策略难以适应多样场景的问题，通过结合全局专家与场景专家，并设计双感知路由机制，实现更优的驾驶性能。**

- **链接: [http://arxiv.org/pdf/2507.14456v1](http://arxiv.org/pdf/2507.14456v1)**

> **作者:** Chi Wan; Yixin Cui; Jiatong Du; Shuo Yang; Yulong Bai; Yanjun Huang
>
> **摘要:** End-to-end autonomous driving requires adaptive and robust handling of complex and diverse traffic environments. However, prevalent single-mode planning methods attempt to learn an overall policy while struggling to acquire diversified driving skills to handle diverse scenarios. Therefore, this paper proposes GEMINUS, a Mixture-of-Experts end-to-end autonomous driving framework featuring a Global Expert, a Scene-Adaptive Experts Group, and equipped with a Dual-aware Router. Specifically, the Global Expert is trained on the overall dataset, possessing robust performance. The Scene-Adaptive Experts are trained on corresponding scene subsets, achieving adaptive performance. The Dual-aware Router simultaneously considers scenario-level features and routing uncertainty to dynamically activate expert modules. Through the effective coupling of the Global Expert and the Scene-Adaptive Experts Group via the Dual-aware Router, GEMINUS achieves adaptive and robust performance in diverse scenarios. GEMINUS outperforms existing methods in the Bench2Drive closed-loop benchmark and achieves state-of-the-art performance in Driving Score and Success Rate, even with only monocular vision input. Furthermore, ablation studies demonstrate significant improvements over the original single-expert baseline: 7.67% in Driving Score, 22.06% in Success Rate, and 19.41% in MultiAbility-Mean. The code will be available at https://github.com/newbrains1/GEMINUS.
>
---
#### [new 031] VisGuard: Securing Visualization Dissemination through Tamper-Resistant Data Retrieval
- **分类: cs.CV**

- **简介: 该论文属于可视化数据检索任务，旨在解决图像在传播过程中因篡改导致的元数据丢失问题。作者提出了VisGuard框架，通过嵌入抗篡改的数据链接，确保即使图像被裁剪或编辑，元数据仍可恢复，从而实现可视化信息的安全传播与交互重建。**

- **链接: [http://arxiv.org/pdf/2507.14459v1](http://arxiv.org/pdf/2507.14459v1)**

> **作者:** Huayuan Ye; Juntong Chen; Shenzhuo Zhang; Yipeng Zhang; Changbo Wang; Chenhui Li
>
> **备注:** 9 pages, IEEE VIS 2025
>
> **摘要:** The dissemination of visualizations is primarily in the form of raster images, which often results in the loss of critical information such as source code, interactive features, and metadata. While previous methods have proposed embedding metadata into images to facilitate Visualization Image Data Retrieval (VIDR), most existing methods lack practicability since they are fragile to common image tampering during online distribution such as cropping and editing. To address this issue, we propose VisGuard, a tamper-resistant VIDR framework that reliably embeds metadata link into visualization images. The embedded data link remains recoverable even after substantial tampering upon images. We propose several techniques to enhance robustness, including repetitive data tiling, invertible information broadcasting, and an anchor-based scheme for crop localization. VisGuard enables various applications, including interactive chart reconstruction, tampering detection, and copyright protection. We conduct comprehensive experiments on VisGuard's superior performance in data retrieval accuracy, embedding capacity, and security against tampering and steganalysis, demonstrating VisGuard's competence in facilitating and safeguarding visualization dissemination and information conveyance.
>
---
#### [new 032] InterAct-Video: Reasoning-Rich Video QA for Urban Traffic
- **分类: cs.CV**

- **简介: 该论文属于视频问答（VideoQA）任务，旨在解决现有模型在复杂交通场景中推理能力不足的问题。作者构建了一个名为InterAct-VideoQA的数据集，包含真实交通视频和大量问答对，用于评估和提升模型对交通场景的理解与推理能力。**

- **链接: [http://arxiv.org/pdf/2507.14743v1](http://arxiv.org/pdf/2507.14743v1)**

> **作者:** Joseph Raj Vishal; Rutuja Patil; Manas Srinivas Gowda; Katha Naik; Yezhou Yang; Bharatesh Chakravarthi
>
> **摘要:** Traffic monitoring is crucial for urban mobility, road safety, and intelligent transportation systems (ITS). Deep learning has advanced video-based traffic monitoring through video question answering (VideoQA) models, enabling structured insight extraction from traffic videos. However, existing VideoQA models struggle with the complexity of real-world traffic scenes, where multiple concurrent events unfold across spatiotemporal dimensions. To address these challenges, this paper introduces \textbf{InterAct VideoQA}, a curated dataset designed to benchmark and enhance VideoQA models for traffic monitoring tasks. The InterAct VideoQA dataset comprises 8 hours of real-world traffic footage collected from diverse intersections, segmented into 10-second video clips, with over 25,000 question-answer (QA) pairs covering spatiotemporal dynamics, vehicle interactions, incident detection, and other critical traffic attributes. State-of-the-art VideoQA models are evaluated on InterAct VideoQA, exposing challenges in reasoning over fine-grained spatiotemporal dependencies within complex traffic scenarios. Additionally, fine-tuning these models on InterAct VideoQA yields notable performance improvements, demonstrating the necessity of domain-specific datasets for VideoQA. InterAct VideoQA is publicly available as a benchmark dataset to facilitate future research in real-world deployable VideoQA models for intelligent transportation systems. GitHub Repo: https://github.com/joe-rabbit/InterAct_VideoQA
>
---
#### [new 033] Efficient Face Image Quality Assessment via Self-training and Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决人脸识别应用中图像质量评估模型计算复杂的问题。作者提出了一种基于自训练和知识蒸馏的轻量化方法，先训练强教师模型并用其生成伪标签，再利用伪标签数据训练轻量学生模型。最终学生模型在保持低计算成本的同时达到与教师模型相当的性能，并在国际比赛中取得第一名。**

- **链接: [http://arxiv.org/pdf/2507.15709v1](http://arxiv.org/pdf/2507.15709v1)**

> **作者:** Wei Sun; Weixia Zhang; Linhan Cao; Jun Jia; Xiangyang Zhu; Dandan Zhu; Xiongkuo Min; Guangtao Zhai
>
> **备注:** Efficient-FIQA achieved first place in the ICCV VQualA 2025 Face Image Quality Assessment Challenge
>
> **摘要:** Face image quality assessment (FIQA) is essential for various face-related applications. Although FIQA has been extensively studied and achieved significant progress, the computational complexity of FIQA algorithms remains a key concern for ensuring scalability and practical deployment in real-world systems. In this paper, we aim to develop a computationally efficient FIQA method that can be easily deployed in real-world applications. Specifically, our method consists of two stages: training a powerful teacher model and distilling a lightweight student model from it. To build a strong teacher model, we adopt a self-training strategy to improve its capacity. We first train the teacher model using labeled face images, then use it to generate pseudo-labels for a set of unlabeled images. These pseudo-labeled samples are used in two ways: (1) to distill knowledge into the student model, and (2) to combine with the original labeled images to further enhance the teacher model through self-training. The enhanced teacher model is used to further pseudo-label another set of unlabeled images for distilling the student models. The student model is trained using a combination of labeled images, pseudo-labeled images from the original teacher model, and pseudo-labeled images from the enhanced teacher model. Experimental results demonstrate that our student model achieves comparable performance to the teacher model with an extremely low computational overhead. Moreover, our method achieved first place in the ICCV 2025 VQualA FIQA Challenge. The code is available at https://github.com/sunwei925/Efficient-FIQA.git.
>
---
#### [new 034] Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization
- **分类: cs.CV; 68T45; I.2.10; H.3.3**

- **简介: 该论文属于交互式文本到视频检索（TVR）任务，旨在解决文本查询模糊、映射不清和视频帧质量低等不确定性问题。作者提出了UMIVR框架，通过量化三种不确定性并自适应生成澄清问题，逐步优化用户查询，提升检索效果。实验表明其在MSR-VTT-1k数据集上显著提高了召回率。**

- **链接: [http://arxiv.org/pdf/2507.15504v1](http://arxiv.org/pdf/2507.15504v1)**

> **作者:** Bingqing Zhang; Zhuo Cao; Heming Du; Yang Li; Xue Li; Jiajun Liu; Sen Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Despite recent advances, Text-to-video retrieval (TVR) is still hindered by multiple inherent uncertainties, such as ambiguous textual queries, indistinct text-video mappings, and low-quality video frames. Although interactive systems have emerged to address these challenges by refining user intent through clarifying questions, current methods typically rely on heuristic or ad-hoc strategies without explicitly quantifying these uncertainties, limiting their effectiveness. Motivated by this gap, we propose UMIVR, an Uncertainty-Minimizing Interactive Text-to-Video Retrieval framework that explicitly quantifies three critical uncertainties-text ambiguity, mapping uncertainty, and frame uncertainty-via principled, training-free metrics: semantic entropy-based Text Ambiguity Score (TAS), Jensen-Shannon divergence-based Mapping Uncertainty Score (MUS), and a Temporal Quality-based Frame Sampler (TQFS). By adaptively generating targeted clarifying questions guided by these uncertainty measures, UMIVR iteratively refines user queries, significantly reducing retrieval ambiguity. Extensive experiments on multiple benchmarks validate UMIVR's effectiveness, achieving notable gains in Recall@1 (69.2\% after 10 interactive rounds) on the MSR-VTT-1k dataset, thereby establishing an uncertainty-minimizing foundation for interactive TVR.
>
---
#### [new 035] Open-set Cross Modal Generalization via Multimodal Unified Representation
- **分类: cs.CV**

- **简介: 该论文提出开放集跨模态泛化（OSCMG）任务，解决现有跨模态统一表示在开放环境下的泛化能力不足问题。通过设计MICU模型，结合多模态对比学习与自监督拼图机制，提升跨模态对齐与未知类别的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.14935v1](http://arxiv.org/pdf/2507.14935v1)**

> **作者:** Hai Huang; Yan Xia; Shulei Wang; Hanting Wang; Minghui Fang; Shengpeng Ji; Sashuai Zhou; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** This paper extends Cross Modal Generalization (CMG) to open-set environments by proposing the more challenging Open-set Cross Modal Generalization (OSCMG) task. This task evaluates multimodal unified representations in open-set conditions, addressing the limitations of prior closed-set cross-modal evaluations. OSCMG requires not only cross-modal knowledge transfer but also robust generalization to unseen classes within new modalities, a scenario frequently encountered in real-world applications. Existing multimodal unified representation work lacks consideration for open-set environments. To tackle this, we propose MICU, comprising two key components: Fine-Coarse Masked multimodal InfoNCE (FCMI) and Cross modal Unified Jigsaw Puzzles (CUJP). FCMI enhances multimodal alignment by applying contrastive learning at both holistic semantic and temporal levels, incorporating masking to enhance generalization. CUJP enhances feature diversity and model uncertainty by integrating modality-agnostic feature selection with self-supervised learning, thereby strengthening the model's ability to handle unknown categories in open-set tasks. Extensive experiments on CMG and the newly proposed OSCMG validate the effectiveness of our approach. The code is available at https://github.com/haihuangcode/CMG.
>
---
#### [new 036] OpenBreastUS: Benchmarking Neural Operators for Wave Imaging Using Breast Ultrasound Computed Tomography
- **分类: cs.CV; cs.LG; 35Q92, 68U10; I.4.5; J.2; J.3**

- **简介: 该论文属于医学成像与偏微分方程求解任务，旨在解决传统数值方法在超声波层析成像中计算昂贵且不稳定的问题。作者构建了OpenBreastUS数据集，包含8000个真实乳腺模型和1600万次波模拟，用于评估神经算子在正向模拟与逆成像任务中的性能，推动其在实际医疗成像中的应用。**

- **链接: [http://arxiv.org/pdf/2507.15035v1](http://arxiv.org/pdf/2507.15035v1)**

> **作者:** Zhijun Zeng; Youjia Zheng; Hao Hu; Zeyuan Dong; Yihang Zheng; Xinliang Liu; Jinzhuo Wang; Zuoqiang Shi; Linfeng Zhang; Yubing Li; He Sun
>
> **摘要:** Accurate and efficient simulation of wave equations is crucial in computational wave imaging applications, such as ultrasound computed tomography (USCT), which reconstructs tissue material properties from observed scattered waves. Traditional numerical solvers for wave equations are computationally intensive and often unstable, limiting their practical applications for quasi-real-time image reconstruction. Neural operators offer an innovative approach by accelerating PDE solving using neural networks; however, their effectiveness in realistic imaging is limited because existing datasets oversimplify real-world complexity. In this paper, we present OpenBreastUS, a large-scale wave equation dataset designed to bridge the gap between theoretical equations and practical imaging applications. OpenBreastUS includes 8,000 anatomically realistic human breast phantoms and over 16 million frequency-domain wave simulations using real USCT configurations. It enables a comprehensive benchmarking of popular neural operators for both forward simulation and inverse imaging tasks, allowing analysis of their performance, scalability, and generalization capabilities. By offering a realistic and extensive dataset, OpenBreastUS not only serves as a platform for developing innovative neural PDE solvers but also facilitates their deployment in real-world medical imaging problems. For the first time, we demonstrate efficient in vivo imaging of the human breast using neural operator solvers.
>
---
#### [new 037] GPI-Net: Gestalt-Guided Parallel Interaction Network via Orthogonal Geometric Consistency for Robust Point Cloud Registration
- **分类: cs.CV**

- **简介: 该论文属于点云配准任务，旨在解决特征融合中局部与全局信息难以协调的问题。作者提出GPI-Net，结合Gestalt原则与正交几何一致性，通过新设计的GFA模块和DMG模块增强特征交互，提升配准鲁棒性与精度。**

- **链接: [http://arxiv.org/pdf/2507.14452v1](http://arxiv.org/pdf/2507.14452v1)**

> **作者:** Weikang Gu; Mingyue Han; Li Xue; Heng Dong; Changcai Yang; Riqing Chen; Lifang Wei
>
> **备注:** 9 pages, 4 figures. Accepted to IJCAI 2025
>
> **摘要:** The accurate identification of high-quality correspondences is a prerequisite task in feature-based point cloud registration. However, it is extremely challenging to handle the fusion of local and global features due to feature redundancy and complex spatial relationships. Given that Gestalt principles provide key advantages in analyzing local and global relationships, we propose a novel Gestalt-guided Parallel Interaction Network via orthogonal geometric consistency (GPI-Net) in this paper. It utilizes Gestalt principles to facilitate complementary communication between local and global information. Specifically, we introduce an orthogonal integration strategy to optimally reduce redundant information and generate a more compact global structure for high-quality correspondences. To capture geometric features in correspondences, we leverage a Gestalt Feature Attention (GFA) block through a hybrid utilization of self-attention and cross-attention mechanisms. Furthermore, to facilitate the integration of local detail information into the global structure, we design an innovative Dual-path Multi-Granularity parallel interaction aggregation (DMG) block to promote information exchange across different granularities. Extensive experiments on various challenging tasks demonstrate the superior performance of our proposed GPI-Net in comparison to existing methods. The code will be released at https://github.com/gwk/GPI-Net.
>
---
#### [new 038] StableAnimator++: Overcoming Pose Misalignment and Face Distortion for Human Image Animation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人体图像动画生成任务，旨在解决参考图像与驱动视频在体型、位置差异大时的身份一致性问题。论文提出StableAnimator++框架，通过可学习姿态对齐、身份保持模块及面部优化方法，实现高质量视频生成，无需后处理即可保持身份一致性。**

- **链接: [http://arxiv.org/pdf/2507.15064v1](http://arxiv.org/pdf/2507.15064v1)**

> **作者:** Shuyuan Tu; Zhen Xing; Xintong Han; Zhi-Qi Cheng; Qi Dai; Chong Luo; Zuxuan Wu; Yu-Gang Jiang
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2411.17697
>
> **摘要:** Current diffusion models for human image animation often struggle to maintain identity (ID) consistency, especially when the reference image and driving video differ significantly in body size or position. We introduce StableAnimator++, the first ID-preserving video diffusion framework with learnable pose alignment, capable of generating high-quality videos conditioned on a reference image and a pose sequence without any post-processing. Building upon a video diffusion model, StableAnimator++ contains carefully designed modules for both training and inference, striving for identity consistency. In particular, StableAnimator++ first uses learnable layers to predict the similarity transformation matrices between the reference image and the driven poses via injecting guidance from Singular Value Decomposition (SVD). These matrices align the driven poses with the reference image, mitigating misalignment to a great extent. StableAnimator++ then computes image and face embeddings using off-the-shelf encoders, refining the face embeddings via a global content-aware Face Encoder. To further maintain ID, we introduce a distribution-aware ID Adapter that counteracts interference caused by temporal layers while preserving ID via distribution alignment. During the inference stage, we propose a novel Hamilton-Jacobi-Bellman (HJB) based face optimization integrated into the denoising process, guiding the diffusion trajectory for enhanced facial fidelity. Experiments on benchmarks show the effectiveness of StableAnimator++ both qualitatively and quantitatively.
>
---
#### [new 039] RoadFusion: Latent Diffusion Model for Pavement Defect Detection
- **分类: cs.CV**

- **简介: 该论文属于图像检测任务，旨在解决路面缺陷检测中数据不足、域间差异和缺陷多样性问题。论文提出RoadFusion框架，通过潜扩散模型生成缺陷图像，并设计双路径特征适配器和轻量判别器，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.15346v1](http://arxiv.org/pdf/2507.15346v1)**

> **作者:** Muhammad Aqeel; Kidus Dagnaw Bellete; Francesco Setti
>
> **备注:** Accepted to ICIAP 2025
>
> **摘要:** Pavement defect detection faces critical challenges including limited annotated data, domain shift between training and deployment environments, and high variability in defect appearances across different road conditions. We propose RoadFusion, a framework that addresses these limitations through synthetic anomaly generation with dual-path feature adaptation. A latent diffusion model synthesizes diverse, realistic defects using text prompts and spatial masks, enabling effective training under data scarcity. Two separate feature adaptors specialize representations for normal and anomalous inputs, improving robustness to domain shift and defect variability. A lightweight discriminator learns to distinguish fine-grained defect patterns at the patch level. Evaluated on six benchmark datasets, RoadFusion achieves consistently strong performance across both classification and localization tasks, setting new state-of-the-art in multiple metrics relevant to real-world road inspection.
>
---
#### [new 040] MultiRetNet: A Multimodal Vision Model and Deferral System for Staging Diabetic Retinopathy
- **分类: cs.CV**

- **简介: 该论文属于医疗图像分析与辅助诊断任务，旨在解决糖尿病视网膜病变（DR）早期检测不足、诊断延迟问题。论文提出MultiRetNet模型，融合视网膜图像、社会经济因素与合并症数据，提升DR分期准确性，并引入临床延迟系统，识别需医生复核的异常样本，以提高基层筛查效率与公平性。**

- **链接: [http://arxiv.org/pdf/2507.14738v1](http://arxiv.org/pdf/2507.14738v1)**

> **作者:** Jeannie She; Katie Spivakovsky
>
> **摘要:** Diabetic retinopathy (DR) is a leading cause of preventable blindness, affecting over 100 million people worldwide. In the United States, individuals from lower-income communities face a higher risk of progressing to advanced stages before diagnosis, largely due to limited access to screening. Comorbid conditions further accelerate disease progression. We propose MultiRetNet, a novel pipeline combining retinal imaging, socioeconomic factors, and comorbidity profiles to improve DR staging accuracy, integrated with a clinical deferral system for a clinical human-in-the-loop implementation. We experiment with three multimodal fusion methods and identify fusion through a fully connected layer as the most versatile methodology. We synthesize adversarial, low-quality images and use contrastive learning to train the deferral system, guiding the model to identify out-of-distribution samples that warrant clinician review. By maintaining diagnostic accuracy on suboptimal images and integrating critical health data, our system can improve early detection, particularly in underserved populations where advanced DR is often first identified. This approach may reduce healthcare costs, increase early detection rates, and address disparities in access to care, promoting healthcare equity.
>
---
#### [new 041] Conditional Video Generation for High-Efficiency Video Compression
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频压缩任务，旨在解决高压缩比下视觉质量下降的问题。作者提出一种基于条件扩散模型的框架，通过多粒度条件输入、紧凑表示和多条件训练，实现感知优化的视频重建，显著提升压缩视频的视觉质量。**

- **链接: [http://arxiv.org/pdf/2507.15269v1](http://arxiv.org/pdf/2507.15269v1)**

> **作者:** Fangqiu Yi; Jingyu Xu; Jiawei Shao; Chi Zhang; Xuelong Li
>
> **摘要:** Perceptual studies demonstrate that conditional diffusion models excel at reconstructing video content aligned with human visual perception. Building on this insight, we propose a video compression framework that leverages conditional diffusion models for perceptually optimized reconstruction. Specifically, we reframe video compression as a conditional generation task, where a generative model synthesizes video from sparse, yet informative signals. Our approach introduces three key modules: (1) Multi-granular conditioning that captures both static scene structure and dynamic spatio-temporal cues; (2) Compact representations designed for efficient transmission without sacrificing semantic richness; (3) Multi-condition training with modality dropout and role-aware embeddings, which prevent over-reliance on any single modality and enhance robustness. Extensive experiments show that our method significantly outperforms both traditional and neural codecs on perceptual quality metrics such as Fr\'echet Video Distance (FVD) and LPIPS, especially under high compression ratios.
>
---
#### [new 042] Adaptive 3D Gaussian Splatting Video Streaming: Visual Saliency-Aware Tiling and Meta-Learning-Based Bitrate Adaptation
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 该论文研究3D高斯点绘视频流，旨在解决其在分块、质量评估和码率自适应方面的挑战。提出了基于视觉显著性的自适应分块方法、结合空间域退化和2D渲染图像质量的评估框架，以及基于元学习的码率自适应算法。实验表明方法优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.14454v1](http://arxiv.org/pdf/2507.14454v1)**

> **作者:** Han Gong; Qiyue Li; Jie Li; Zhi Liu
>
> **摘要:** 3D Gaussian splatting video (3DGS) streaming has recently emerged as a research hotspot in both academia and industry, owing to its impressive ability to deliver immersive 3D video experiences. However, research in this area is still in its early stages, and several fundamental challenges, such as tiling, quality assessment, and bitrate adaptation, require further investigation. In this paper, we tackle these challenges by proposing a comprehensive set of solutions. Specifically, we propose an adaptive 3DGS tiling technique guided by saliency analysis, which integrates both spatial and temporal features. Each tile is encoded into versions possessing dedicated deformation fields and multiple quality levels for adaptive selection. We also introduce a novel quality assessment framework for 3DGS video that jointly evaluates spatial-domain degradation in 3DGS representations during streaming and the quality of the resulting 2D rendered images. Additionally, we develop a meta-learning-based adaptive bitrate algorithm specifically tailored for 3DGS video streaming, achieving optimal performance across varying network conditions. Extensive experiments demonstrate that our proposed approaches significantly outperform state-of-the-art methods.
>
---
#### [new 043] Paired Image Generation with Diffusion-Guided Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像生成与数据增强任务，旨在解决乳腺断层扫描图像中肿块病灶标注数据不足的问题。通过提出一种无需外部条件的配对图像生成方法，同时生成图像及其标注，提升模型训练效果。**

- **链接: [http://arxiv.org/pdf/2507.14833v1](http://arxiv.org/pdf/2507.14833v1)**

> **作者:** Haoxuan Zhang; Wenju Cui; Yuzhu Cao; Tao Tan; Jie Liu; Yunsong Peng; Jian Zheng
>
> **摘要:** The segmentation of mass lesions in digital breast tomosynthesis (DBT) images is very significant for the early screening of breast cancer. However, the high-density breast tissue often leads to high concealment of the mass lesions, which makes manual annotation difficult and time-consuming. As a result, there is a lack of annotated data for model training. Diffusion models are commonly used for data augmentation, but the existing methods face two challenges. First, due to the high concealment of lesions, it is difficult for the model to learn the features of the lesion area. This leads to the low generation quality of the lesion areas, thus limiting the quality of the generated images. Second, existing methods can only generate images and cannot generate corresponding annotations, which restricts the usability of the generated images in supervised training. In this work, we propose a paired image generation method. The method does not require external conditions and can achieve the generation of paired images by training an extra diffusion guider for the conditional diffusion model. During the experimental phase, we generated paired DBT slices and mass lesion masks. Then, we incorporated them into the supervised training process of the mass lesion segmentation task. The experimental results show that our method can improve the generation quality without external conditions. Moreover, it contributes to alleviating the shortage of annotated data, thus enhancing the performance of downstream tasks.
>
---
#### [new 044] Polymorph: Energy-Efficient Multi-Label Classification for Video Streams on Embedded Devices
- **分类: cs.CV; cs.PF**

- **简介: 该论文属于视频流多标签分类任务，旨在解决嵌入式设备上实时推理的能耗与计算限制问题。论文提出Polymorph框架，利用标签稀疏性、时间连续性和共现模式，动态选择轻量级LoRA适配器进行推理，降低能耗并提升性能。**

- **链接: [http://arxiv.org/pdf/2507.14959v1](http://arxiv.org/pdf/2507.14959v1)**

> **作者:** Saeid Ghafouri; Mohsen Fayyaz; Xiangchen Li; Deepu John; Bo Ji; Dimitrios Nikolopoulos; Hans Vandierendonck
>
> **摘要:** Real-time multi-label video classification on embedded devices is constrained by limited compute and energy budgets. Yet, video streams exhibit structural properties such as label sparsity, temporal continuity, and label co-occurrence that can be leveraged for more efficient inference. We introduce Polymorph, a context-aware framework that activates a minimal set of lightweight Low Rank Adapters (LoRA) per frame. Each adapter specializes in a subset of classes derived from co-occurrence patterns and is implemented as a LoRA weight over a shared backbone. At runtime, Polymorph dynamically selects and composes only the adapters needed to cover the active labels, avoiding full-model switching and weight merging. This modular strategy improves scalability while reducing latency and energy overhead. Polymorph achieves 40% lower energy consumption and improves mAP by 9 points over strong baselines on the TAO dataset. Polymorph is open source at https://github.com/inference-serving/polymorph/.
>
---
#### [new 045] Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与图形学任务，旨在解决传统3D重建与视角合成方法计算复杂、效率低的问题。论文系统综述了基于深度学习的前馈方法，涵盖点云、3D高斯泼溅、NeRF等表示架构，分析关键任务与数据集，并探讨未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.14501v1](http://arxiv.org/pdf/2507.14501v1)**

> **作者:** Jiahui Zhang; Yuelei Li; Anpei Chen; Muyu Xu; Kunhao Liu; Jianyuan Wang; Xiao-Xiao Long; Hanxue Liang; Zexiang Xu; Hao Su; Christian Theobalt; Christian Rupprecht; Andrea Vedaldi; Hanspeter Pfister; Shijian Lu; Fangneng Zhan
>
> **备注:** A project page associated with this survey is available at https://fnzhan.com/projects/Feed-Forward-3D
>
> **摘要:** 3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.
>
---
#### [new 046] HOLa: Zero-Shot HOI Detection with Low-Rank Decomposed VLM Feature Adaptation
- **分类: cs.CV**

- **简介: 该论文属于零样本人体-物体交互（HOI）检测任务，旨在解决现有方法在未见动作类别上泛化能力差、动作区分度低的问题。作者提出HOLa方法，通过低秩分解VLM文本特征，构建共享基特征与可适应权重，并引入人类-物体标记和LLM引导的正则化，提升未见类别的检测性能。实验表明该方法在HICO-DET数据集上表现最优。**

- **链接: [http://arxiv.org/pdf/2507.15542v1](http://arxiv.org/pdf/2507.15542v1)**

> **作者:** Qinqian Lei; Bo Wang; Robby T. Tan
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Zero-shot human-object interaction (HOI) detection remains a challenging task, particularly in generalizing to unseen actions. Existing methods address this challenge by tapping Vision-Language Models (VLMs) to access knowledge beyond the training data. However, they either struggle to distinguish actions involving the same object or demonstrate limited generalization to unseen classes. In this paper, we introduce HOLa (Zero-Shot HOI Detection with Low-Rank Decomposed VLM Feature Adaptation), a novel approach that both enhances generalization to unseen classes and improves action distinction. In training, HOLa decomposes VLM text features for given HOI classes via low-rank factorization, producing class-shared basis features and adaptable weights. These features and weights form a compact HOI representation that preserves shared information across classes, enhancing generalization to unseen classes. Subsequently, we refine action distinction by adapting weights for each HOI class and introducing human-object tokens to enrich visual interaction representations. To further distinguish unseen actions, we guide the weight adaptation with LLM-derived action regularization. Experimental results show that our method sets a new state-of-the-art across zero-shot HOI settings on HICO-DET, achieving an unseen-class mAP of 27.91 in the unseen-verb setting. Our code is available at https://github.com/ChelsieLei/HOLa.
>
---
#### [new 047] LoopNet: A Multitasking Few-Shot Learning Approach for Loop Closure in Large Scale SLAM
- **分类: cs.CV; cs.LG**

- **简介: 论文提出LoopNet，一种基于多任务ResNet架构的少样本学习方法，用于大规模SLAM中的回环检测。任务是提高回环检测精度并满足嵌入式设备的实时计算需求。工作包括设计可在线重训练的网络结构、融合DISK描述子提升性能，并发布新数据集LoopDB。**

- **链接: [http://arxiv.org/pdf/2507.15109v1](http://arxiv.org/pdf/2507.15109v1)**

> **作者:** Mohammad-Maher Nakshbandi; Ziad Sharawy; Sorin Grigorescu
>
> **摘要:** One of the main challenges in the Simultaneous Localization and Mapping (SLAM) loop closure problem is the recognition of previously visited places. In this work, we tackle the two main problems of real-time SLAM systems: 1) loop closure detection accuracy and 2) real-time computation constraints on the embedded hardware. Our LoopNet method is based on a multitasking variant of the classical ResNet architecture, adapted for online retraining on a dynamic visual dataset and optimized for embedded devices. The online retraining is designed using a few-shot learning approach. The architecture provides both an index into the queried visual dataset, and a measurement of the prediction quality. Moreover, by leveraging DISK (DIStinctive Keypoints) descriptors, LoopNet surpasses the limitations of handcrafted features and traditional deep learning methods, offering better performance under varying conditions. Code is available at https://github.com/RovisLab/LoopNet. Additinally, we introduce a new loop closure benchmarking dataset, coined LoopDB, which is available at https://github.com/RovisLab/LoopDB.
>
---
#### [new 048] Procedure Learning via Regularized Gromov-Wasserstein Optimal Transport
- **分类: cs.CV**

- **简介: 该论文属于自监督程序学习任务，旨在从无标签的程序视频中发现关键步骤并确定其顺序。现有方法在处理顺序变化、冗余帧和重复动作时表现不佳。论文提出一种融合正则化Gromov-Wasserstein最优传输框架，结合结构先验与对比正则化，有效解决帧映射中的退化解问题，提升了程序学习的准确性。**

- **链接: [http://arxiv.org/pdf/2507.15540v1](http://arxiv.org/pdf/2507.15540v1)**

> **作者:** Syed Ahmed Mahmood; Ali Shah Ali; Umer Ahmed; Fawad Javed Fateh; M. Zeeshan Zia; Quoc-Huy Tran
>
> **摘要:** We study the problem of self-supervised procedure learning, which discovers key steps and establishes their order from a set of unlabeled procedural videos. Previous procedure learning methods typically learn frame-to-frame correspondences between videos before determining key steps and their order. However, their performance often suffers from order variations, background/redundant frames, and repeated actions. To overcome these challenges, we propose a self-supervised procedure learning framework, which utilizes a fused Gromov-Wasserstein optimal transport formulation with a structural prior for computing frame-to-frame mapping between videos. However, optimizing exclusively for the above temporal alignment term may lead to degenerate solutions, where all frames are mapped to a small cluster in the embedding space and hence every video is associated with only one key step. To address that limitation, we further integrate a contrastive regularization term, which maps different frames to different points in the embedding space, avoiding the collapse to trivial solutions. Finally, we conduct extensive experiments on large-scale egocentric (i.e., EgoProceL) and third-person (i.e., ProceL and CrossTask) benchmarks to demonstrate superior performance by our approach against previous methods, including OPEL which relies on a traditional Kantorovich optimal transport formulation with an optimality prior.
>
---
#### [new 049] Semantic Segmentation based Scene Understanding in Autonomous Vehicles
- **分类: cs.CV; I.4.8**

- **简介: 该论文属于自动驾驶中的场景理解任务，旨在解决通过语义分割提升自动驾驶对周围环境的理解能力。论文使用BDD100k数据集，探索了多种高效模型，并采用不同Backbone作为编码器。结果表明，选择合适的Backbone能显著提升语义分割性能，从而提高场景理解的准确性。**

- **链接: [http://arxiv.org/pdf/2507.14303v1](http://arxiv.org/pdf/2507.14303v1)**

> **作者:** Ehsan Rassekh
>
> **备注:** 74 pages, 35 figures, Master's Thesis, Institute for Advanced Studies in Basic Sciences (IASBS), Zanjan, Iran, 2023
>
> **摘要:** In recent years, the concept of artificial intelligence (AI) has become a prominent keyword because it is promising in solving complex tasks. The need for human expertise in specific areas may no longer be needed because machines have achieved successful results using artificial intelligence and can make the right decisions in critical situations. This process is possible with the help of deep learning (DL), one of the most popular artificial intelligence technologies. One of the areas in which the use of DL is used is in the development of self-driving cars, which is very effective and important. In this work, we propose several efficient models to investigate scene understanding through semantic segmentation. We use the BDD100k dataset to investigate these models. Another contribution of this work is the usage of several Backbones as encoders for models. The obtained results show that choosing the appropriate backbone has a great effect on the performance of the model for semantic segmentation. Better performance in semantic segmentation allows us to understand better the scene and the environment around the agent. In the end, we analyze and evaluate the proposed models in terms of accuracy, mean IoU, and loss function, and the results show that these metrics are improved.
>
---
#### [new 050] Uncovering Critical Features for Deepfake Detection through the Lottery Ticket Hypothesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度伪造检测任务，旨在解决检测模型复杂、机制不清晰的问题。通过彩票假设理论，寻找关键子网络，在保持高检测准确率的同时大幅压缩模型规模，并验证了子网络在不同数据集上的可迁移性。**

- **链接: [http://arxiv.org/pdf/2507.15636v1](http://arxiv.org/pdf/2507.15636v1)**

> **作者:** Lisan Al Amin; Md. Ismail Hossain; Thanh Thi Nguyen; Tasnim Jahan; Mahbubul Islam; Faisal Quader
>
> **备注:** Accepted for publication at the 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC)
>
> **摘要:** Recent advances in deepfake technology have created increasingly convincing synthetic media that poses significant challenges to information integrity and social trust. While current detection methods show promise, their underlying mechanisms remain poorly understood, and the large sizes of their models make them challenging to deploy in resource-limited environments. This study investigates the application of the Lottery Ticket Hypothesis (LTH) to deepfake detection, aiming to identify the key features crucial for recognizing deepfakes. We examine how neural networks can be efficiently pruned while maintaining high detection accuracy. Through extensive experiments with MesoNet, CNN-5, and ResNet-18 architectures on the OpenForensic and FaceForensics++ datasets, we find that deepfake detection networks contain winning tickets, i.e., subnetworks, that preserve performance even at substantial sparsity levels. Our results indicate that MesoNet retains 56.2% accuracy at 80% sparsity on the OpenForensic dataset, with only 3,000 parameters, which is about 90% of its baseline accuracy (62.6%). The results also show that our proposed LTH-based iterative magnitude pruning approach consistently outperforms one-shot pruning methods. Using Grad-CAM visualization, we analyze how pruned networks maintain their focus on critical facial regions for deepfake detection. Additionally, we demonstrate the transferability of winning tickets across datasets, suggesting potential for efficient, deployable deepfake detection systems.
>
---
#### [new 051] Hybrid-supervised Hypergraph-enhanced Transformer for Micro-gesture Based Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文属于情感识别任务，旨在通过微手势建模识别情绪状态。论文提出了一种基于超图增强Transformer的混合监督框架，结合自监督重建与监督学习，提升微手势中细微动作的建模能力，从而更准确识别情绪。**

- **链接: [http://arxiv.org/pdf/2507.14867v1](http://arxiv.org/pdf/2507.14867v1)**

> **作者:** Zhaoqiang Xia; Hexiang Huang; Haoyu Chen; Xiaoyi Feng; Guoying Zhao
>
> **摘要:** Micro-gestures are unconsciously performed body gestures that can convey the emotion states of humans and start to attract more research attention in the fields of human behavior understanding and affective computing as an emerging topic. However, the modeling of human emotion based on micro-gestures has not been explored sufficiently. In this work, we propose to recognize the emotion states based on the micro-gestures by reconstructing the behavior patterns with a hypergraph-enhanced Transformer in a hybrid-supervised framework. In the framework, hypergraph Transformer based encoder and decoder are separately designed by stacking the hypergraph-enhanced self-attention and multiscale temporal convolution modules. Especially, to better capture the subtle motion of micro-gestures, we construct a decoder with additional upsampling operations for a reconstruction task in a self-supervised learning manner. We further propose a hypergraph-enhanced self-attention module where the hyperedges between skeleton joints are gradually updated to present the relationships of body joints for modeling the subtle local motion. Lastly, for exploiting the relationship between the emotion states and local motion of micro-gestures, an emotion recognition head from the output of encoder is designed with a shallow architecture and learned in a supervised way. The end-to-end framework is jointly trained in a one-stage way by comprehensively utilizing self-reconstruction and supervision information. The proposed method is evaluated on two publicly available datasets, namely iMiGUE and SMG, and achieves the best performance under multiple metrics, which is superior to the existing methods.
>
---
#### [new 052] An Uncertainty-aware DETR Enhancement Framework for Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在提升检测框定位精度并建模预测不确定性。针对传统检测器忽略预测不确定性的缺陷，作者提出一种基于DETR的不确定性感知增强框架，使用多维高斯分布建模检测框，引入Gromov-Wasserstein距离和贝叶斯风险机制优化检测可靠性，并通过置信区间量化定位不确定性。实验表明该方法在COCO及白细胞检测数据集上均有效提升性能。**

- **链接: [http://arxiv.org/pdf/2507.14855v1](http://arxiv.org/pdf/2507.14855v1)**

> **作者:** Xingshu Chen; Sicheng Yu; Chong Cheng; Hao Wang; Ting Tian
>
> **摘要:** This paper investigates the problem of object detection with a focus on improving both the localization accuracy of bounding boxes and explicitly modeling prediction uncertainty. Conventional detectors rely on deterministic bounding box regression, ignoring uncertainty in predictions and limiting model robustness. In this paper, we propose an uncertainty-aware enhancement framework for DETR-based object detectors. We model bounding boxes as multivariate Gaussian distributions and incorporate the Gromov-Wasserstein distance into the loss function to better align the predicted and ground-truth distributions. Building on this, we derive a Bayes Risk formulation to filter high-risk information and improve detection reliability. We also propose a simple algorithm to quantify localization uncertainty via confidence intervals. Experiments on the COCO benchmark show that our method can be effectively integrated into existing DETR variants, enhancing their performance. We further extend our framework to leukocyte detection tasks, achieving state-of-the-art results on the LISC and WBCDD datasets. These results confirm the scalability of our framework across both general and domain-specific detection tasks. Code page: https://github.com/ParadiseforAndaChen/An-Uncertainty-aware-DETR-Enhancement-Framework-for-Object-Detection.
>
---
#### [new 053] DUSTrack: Semi-automated point tracking in ultrasound videos
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于医学图像处理任务，旨在解决超声视频中组织运动跟踪不准确的问题。作者提出了DUSTrack，结合深度学习与光流技术，实现对B型超声视频中任意点的半自动跟踪，提升了跟踪精度与鲁棒性，并通过多个应用场景验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.14368v1](http://arxiv.org/pdf/2507.14368v1)**

> **作者:** Praneeth Namburi; Roger Pallarès-López; Jessica Rosendorf; Duarte Folgado; Brian W. Anthony
>
> **摘要:** Ultrasound technology enables safe, non-invasive imaging of dynamic tissue behavior, making it a valuable tool in medicine, biomechanics, and sports science. However, accurately tracking tissue motion in B-mode ultrasound remains challenging due to speckle noise, low edge contrast, and out-of-plane movement. These challenges complicate the task of tracking anatomical landmarks over time, which is essential for quantifying tissue dynamics in many clinical and research applications. This manuscript introduces DUSTrack (Deep learning and optical flow-based toolkit for UltraSound Tracking), a semi-automated framework for tracking arbitrary points in B-mode ultrasound videos. We combine deep learning with optical flow to deliver high-quality and robust tracking across diverse anatomical structures and motion patterns. The toolkit includes a graphical user interface that streamlines the generation of high-quality training data and supports iterative model refinement. It also implements a novel optical-flow-based filtering technique that reduces high-frequency frame-to-frame noise while preserving rapid tissue motion. DUSTrack demonstrates superior accuracy compared to contemporary zero-shot point trackers and performs on par with specialized methods, establishing its potential as a general and foundational tool for clinical and biomechanical research. We demonstrate DUSTrack's versatility through three use cases: cardiac wall motion tracking in echocardiograms, muscle deformation analysis during reaching tasks, and fascicle tracking during ankle plantarflexion. As an open-source solution, DUSTrack offers a powerful, flexible framework for point tracking to quantify tissue motion from ultrasound videos. DUSTrack is available at https://github.com/praneethnamburi/DUSTrack.
>
---
#### [new 054] CLIPTTA: Robust Contrastive Vision-Language Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的测试时自适应任务，旨在解决模型在分布偏移下的泛化问题。作者提出了CLIPTTA方法，采用软对比损失以对齐CLIP预训练目标，并设计了面向开放集的OCE损失用于异常样本检测，提升了自适应性能与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.14312v1](http://arxiv.org/pdf/2507.14312v1)**

> **作者:** Marc Lafon; Gustavo Adolfo Vargas Hakim; Clément Rambour; Christian Desrosier; Nicolas Thome
>
> **摘要:** Vision-language models (VLMs) like CLIP exhibit strong zero-shot capabilities but often fail to generalize under distribution shifts. Test-time adaptation (TTA) allows models to update at inference time without labeled data, typically via entropy minimization. However, this objective is fundamentally misaligned with the contrastive image-text training of VLMs, limiting adaptation performance and introducing failure modes such as pseudo-label drift and class collapse. We propose CLIPTTA, a new gradient-based TTA method for vision-language models that leverages a soft contrastive loss aligned with CLIP's pre-training objective. We provide a theoretical analysis of CLIPTTA's gradients, showing how its batch-aware design mitigates the risk of collapse. We further extend CLIPTTA to the open-set setting, where both in-distribution (ID) and out-of-distribution (OOD) samples are encountered, using an Outlier Contrastive Exposure (OCE) loss to improve OOD detection. Evaluated on 75 datasets spanning diverse distribution shifts, CLIPTTA consistently outperforms entropy-based objectives and is highly competitive with state-of-the-art TTA methods, outperforming them on a large number of datasets and exhibiting more stable performance across diverse shifts.
>
---
#### [new 055] BenchDepth: Are We on the Right Way to Evaluate Depth Foundation Models?
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决深度估计模型评估不一致的问题。作者提出了BenchDepth，通过五个实际应用任务评估深度模型性能，避免传统对齐方法的偏差，推动深度估计模型的公平比较与未来发展。**

- **链接: [http://arxiv.org/pdf/2507.15321v1](http://arxiv.org/pdf/2507.15321v1)**

> **作者:** Zhenyu Li; Haotong Lin; Jiashi Feng; Peter Wonka; Bingyi Kang
>
> **备注:** Webpage: https://zhyever.github.io/benchdepth
>
> **摘要:** Depth estimation is a fundamental task in computer vision with diverse applications. Recent advancements in deep learning have led to powerful depth foundation models (DFMs), yet their evaluation remains challenging due to inconsistencies in existing protocols. Traditional benchmarks rely on alignment-based metrics that introduce biases, favor certain depth representations, and complicate fair comparisons. In this work, we propose BenchDepth, a new benchmark that evaluates DFMs through five carefully selected downstream proxy tasks: depth completion, stereo matching, monocular feed-forward 3D scene reconstruction, SLAM, and vision-language spatial understanding. Unlike conventional evaluation protocols, our approach assesses DFMs based on their practical utility in real-world applications, bypassing problematic alignment procedures. We benchmark eight state-of-the-art DFMs and provide an in-depth analysis of key findings and observations. We hope our work sparks further discussion in the community on best practices for depth model evaluation and paves the way for future research and advancements in depth estimation.
>
---
#### [new 056] Hi^2-GSLoc: Dual-Hierarchical Gaussian-Specific Visual Relocalization for Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于视觉重定位任务，旨在解决遥感场景中相机6自由度姿态估计的精度与效率问题。现有方法在精度或计算复杂度上存在不足，为此，作者提出了Hi²-GSLoc，利用3D高斯点阵作为场景表示，结合稀疏到稠密、由粗到精的双层次定位框架，提升了定位精度与效率，并通过分区训练、并行匹配和动态内存管理应对大规模遥感场景挑战。**

- **链接: [http://arxiv.org/pdf/2507.15683v1](http://arxiv.org/pdf/2507.15683v1)**

> **作者:** Boni Hu; Zhenyu Xia; Lin Chen; Pengcheng Han; Shuhui Bu
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Visual relocalization, which estimates the 6-degree-of-freedom (6-DoF) camera pose from query images, is fundamental to remote sensing and UAV applications. Existing methods face inherent trade-offs: image-based retrieval and pose regression approaches lack precision, while structure-based methods that register queries to Structure-from-Motion (SfM) models suffer from computational complexity and limited scalability. These challenges are particularly pronounced in remote sensing scenarios due to large-scale scenes, high altitude variations, and domain gaps of existing visual priors. To overcome these limitations, we leverage 3D Gaussian Splatting (3DGS) as a novel scene representation that compactly encodes both 3D geometry and appearance. We introduce $\mathrm{Hi}^2$-GSLoc, a dual-hierarchical relocalization framework that follows a sparse-to-dense and coarse-to-fine paradigm, fully exploiting the rich semantic information and geometric constraints inherent in Gaussian primitives. To handle large-scale remote sensing scenarios, we incorporate partitioned Gaussian training, GPU-accelerated parallel matching, and dynamic memory management strategies. Our approach consists of two stages: (1) a sparse stage featuring a Gaussian-specific consistent render-aware sampling strategy and landmark-guided detector for robust and accurate initial pose estimation, and (2) a dense stage that iteratively refines poses through coarse-to-fine dense rasterization matching while incorporating reliability verification. Through comprehensive evaluation on simulation data, public datasets, and real flight experiments, we demonstrate that our method delivers competitive localization accuracy, recall rate, and computational efficiency while effectively filtering unreliable pose estimates. The results confirm the effectiveness of our approach for practical remote sensing applications.
>
---
#### [new 057] Label tree semantic losses for rich multi-class medical image segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决现有方法忽略标签间语义关系的问题。作者提出了两种基于标签树结构的语义损失函数，并结合稀疏标注训练方法，提升了分割性能，在脑部MRI和神经外科高光谱图像数据上均取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2507.15777v1](http://arxiv.org/pdf/2507.15777v1)**

> **作者:** Junwen Wang; Oscar MacCormac; William Rochford; Aaron Kujawa; Jonathan Shapey; Tom Vercauteren
>
> **备注:** arXiv admin note: text overlap with arXiv:2506.21150
>
> **摘要:** Rich and accurate medical image segmentation is poised to underpin the next generation of AI-defined clinical practice by delineating critical anatomy for pre-operative planning, guiding real-time intra-operative navigation, and supporting precise post-operative assessment. However, commonly used learning methods for medical and surgical imaging segmentation tasks penalise all errors equivalently and thus fail to exploit any inter-class semantics in the labels space. This becomes particularly problematic as the cardinality and richness of labels increases to include subtly different classes. In this work, we propose two tree-based semantic loss functions which take advantage of a hierarchical organisation of the labels. We further incorporate our losses in a recently proposed approach for training with sparse, background-free annotations to extend the applicability of our proposed losses. Extensive experiments are reported on two medical and surgical image segmentation tasks, namely head MRI for whole brain parcellation (WBP) with full supervision and neurosurgical hyperspectral imaging (HSI) for scene understanding with sparse annotations. Results demonstrate that our proposed method reaches state-of-the-art performance in both cases.
>
---
#### [new 058] Minutiae-Anchored Local Dense Representation for Fingerprint Matching
- **分类: cs.CV**

- **简介: 该论文属于生物特征识别任务，旨在解决不同采集条件下指纹匹配的鲁棒性与准确性问题。论文提出DMD方法，通过细节点锚定的局部密集表示，融合细粒度纹理和判别特征，并利用前景分割提升匹配效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.15297v1](http://arxiv.org/pdf/2507.15297v1)**

> **作者:** Zhiyu Pan; Xiongjun Guan; Yongjie Duan; Jianjiang Feng; Jie Zhou
>
> **备注:** Under review
>
> **摘要:** Fingerprint matching under diverse capture conditions remains a fundamental challenge in biometric recognition. To achieve robust and accurate performance in such scenarios, we propose DMD, a minutiae-anchored local dense representation which captures both fine-grained ridge textures and discriminative minutiae features in a spatially structured manner. Specifically, descriptors are extracted from local patches centered and oriented on each detected minutia, forming a three-dimensional tensor, where two dimensions represent spatial locations on the fingerprint plane and the third encodes semantic features. This representation explicitly captures abstract features of local image patches, enabling a multi-level, fine-grained description that aggregates information from multiple minutiae and their surrounding ridge structures. Furthermore, thanks to its strong spatial correspondence with the patch image, DMD allows for the use of foreground segmentation masks to identify valid descriptor regions. During matching, comparisons are then restricted to overlapping foreground areas, improving efficiency and robustness. Extensive experiments on rolled, plain, parital, contactless, and latent fingerprint datasets demonstrate the effectiveness and generalizability of the proposed method. It achieves state-of-the-art accuracy across multiple benchmarks while maintaining high computational efficiency, showing strong potential for large-scale fingerprint recognition. Corresponding code is available at https://github.com/Yu-Yy/DMD.
>
---
#### [new 059] MinCD-PnP: Learning 2D-3D Correspondences with Approximate Blind PnP
- **分类: cs.CV**

- **简介: 该论文属于图像与点云配准任务，旨在解决2D-3D对应关系学习中对噪声和异常值敏感的问题。作者提出了一种基于近似盲PnP的对应学习方法MinCD-PnP，并设计了相应的轻量级网络MinCD-Net，以提高配准的鲁棒性和效率。**

- **链接: [http://arxiv.org/pdf/2507.15257v1](http://arxiv.org/pdf/2507.15257v1)**

> **作者:** Pei An; Jiaqi Yang; Muyao Peng; You Yang; Qiong Liu; Xiaolin Wu; Liangliang Nan
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Image-to-point-cloud (I2P) registration is a fundamental problem in computer vision, focusing on establishing 2D-3D correspondences between an image and a point cloud. The differential perspective-n-point (PnP) has been widely used to supervise I2P registration networks by enforcing the projective constraints on 2D-3D correspondences. However, differential PnP is highly sensitive to noise and outliers in the predicted correspondences. This issue hinders the effectiveness of correspondence learning. Inspired by the robustness of blind PnP against noise and outliers in correspondences, we propose an approximated blind PnP based correspondence learning approach. To mitigate the high computational cost of blind PnP, we simplify blind PnP to an amenable task of minimizing Chamfer distance between learned 2D and 3D keypoints, called MinCD-PnP. To effectively solve MinCD-PnP, we design a lightweight multi-task learning module, named as MinCD-Net, which can be easily integrated into the existing I2P registration architectures. Extensive experiments on 7-Scenes, RGBD-V2, ScanNet, and self-collected datasets demonstrate that MinCD-Net outperforms state-of-the-art methods and achieves a higher inlier ratio (IR) and registration recall (RR) in both cross-scene and cross-dataset settings.
>
---
#### [new 060] Training Self-Supervised Depth Completion Using Sparse Measurements and a Single Image
- **分类: cs.CV**

- **简介: 该论文属于深度补全任务，旨在解决从稀疏深度测量中恢复密集深度图的问题。现有方法依赖密集标注或多视角图像，而本文提出一种仅需单张图像和稀疏深度的自监督方法，设计新损失函数并结合分割图提升深度估计效果。**

- **链接: [http://arxiv.org/pdf/2507.14845v1](http://arxiv.org/pdf/2507.14845v1)**

> **作者:** Rizhao Fan; Zhigen Li; Heping Li; Ning An
>
> **摘要:** Depth completion is an important vision task, and many efforts have been made to enhance the quality of depth maps from sparse depth measurements. Despite significant advances, training these models to recover dense depth from sparse measurements remains a challenging problem. Supervised learning methods rely on dense depth labels to predict unobserved regions, while self-supervised approaches require image sequences to enforce geometric constraints and photometric consistency between frames. However, acquiring dense annotations is costly, and multi-frame dependencies limit the applicability of self-supervised methods in static or single-frame scenarios. To address these challenges, we propose a novel self-supervised depth completion paradigm that requires only sparse depth measurements and their corresponding image for training. Unlike existing methods, our approach eliminates the need for dense depth labels or additional images captured from neighboring viewpoints. By leveraging the characteristics of depth distribution, we design novel loss functions that effectively propagate depth information from observed points to unobserved regions. Additionally, we incorporate segmentation maps generated by vision foundation models to further enhance depth estimation. Extensive experiments demonstrate the effectiveness of our proposed method.
>
---
#### [new 061] Region-aware Depth Scale Adaptation with Sparse Measurements
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决深度预测模型输出为相对尺度而非实际度量尺度的问题。现有方法需额外训练，成本高且影响模型泛化能力。作者提出一种无需训练的非学习方法，利用稀疏深度测量将相对深度转换为度量深度，保留模型原有性能的同时实现准确尺度转换。**

- **链接: [http://arxiv.org/pdf/2507.14879v1](http://arxiv.org/pdf/2507.14879v1)**

> **作者:** Rizhao Fan; Tianfang Ma; Zhigen Li; Ning An; Jian Cheng
>
> **摘要:** In recent years, the emergence of foundation models for depth prediction has led to remarkable progress, particularly in zero-shot monocular depth estimation. These models generate impressive depth predictions; however, their outputs are often in relative scale rather than metric scale. This limitation poses challenges for direct deployment in real-world applications. To address this, several scale adaptation methods have been proposed to enable foundation models to produce metric depth. However, these methods are typically costly, as they require additional training on new domains and datasets. Moreover, fine-tuning these models often compromises their original generalization capabilities, limiting their adaptability across diverse scenes. In this paper, we introduce a non-learning-based approach that leverages sparse depth measurements to adapt the relative-scale predictions of foundation models into metric-scale depth. Our method requires neither retraining nor fine-tuning, thereby preserving the strong generalization ability of the original foundation models while enabling them to produce metric depth. Experimental results demonstrate the effectiveness of our approach, high-lighting its potential to bridge the gap between relative and metric depth without incurring additional computational costs or sacrificing generalization ability.
>
---
#### [new 062] Performance comparison of medical image classification systems using TensorFlow Keras, PyTorch, and JAX
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在比较TensorFlow Keras、PyTorch和JAX在血细胞图像分类中的性能差异。论文通过使用BloodMNIST数据集，评估不同框架在分类准确率和推理时间上的表现，探讨图像分辨率和框架优化对结果的影响，以辅助医学诊断系统选择合适工具。**

- **链接: [http://arxiv.org/pdf/2507.14587v1](http://arxiv.org/pdf/2507.14587v1)**

> **作者:** Merjem Bećirović; Amina Kurtović; Nordin Smajlović; Medina Kapo; Amila Akagić
>
> **摘要:** Medical imaging plays a vital role in early disease diagnosis and monitoring. Specifically, blood microscopy offers valuable insights into blood cell morphology and the detection of hematological disorders. In recent years, deep learning-based automated classification systems have demonstrated high potential in enhancing the accuracy and efficiency of blood image analysis. However, a detailed performance analysis of specific deep learning frameworks appears to be lacking. This paper compares the performance of three popular deep learning frameworks, TensorFlow with Keras, PyTorch, and JAX, in classifying blood cell images from the publicly available BloodMNIST dataset. The study primarily focuses on inference time differences, but also classification performance for different image sizes. The results reveal variations in performance across frameworks, influenced by factors such as image resolution and framework-specific optimizations. Classification accuracy for JAX and PyTorch was comparable to current benchmarks, showcasing the efficiency of these frameworks for medical image classification.
>
---
#### [new 063] EBA-AI: Ethics-Guided Bias-Aware AI for Efficient Underwater Image Enhancement and Coral Reef Monitoring
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于水下图像增强任务，旨在解决AI模型在珊瑚礁监测中面临的数据集偏差、计算成本高和可解释性差的问题。作者提出了EBA-AI框架，利用CLIP嵌入减轻数据偏差，引入自适应处理降低能耗，并结合不确定性估计与可解释性技术，提升模型公平性、效率与可信度。**

- **链接: [http://arxiv.org/pdf/2507.15036v1](http://arxiv.org/pdf/2507.15036v1)**

> **作者:** Lyes Saad Saoud; Irfan Hussain
>
> **摘要:** Underwater image enhancement is vital for marine conservation, particularly coral reef monitoring. However, AI-based enhancement models often face dataset bias, high computational costs, and lack of transparency, leading to potential misinterpretations. This paper introduces EBA-AI, an ethics-guided bias-aware AI framework to address these challenges. EBA-AI leverages CLIP embeddings to detect and mitigate dataset bias, ensuring balanced representation across varied underwater environments. It also integrates adaptive processing to optimize energy efficiency, significantly reducing GPU usage while maintaining competitive enhancement quality. Experiments on LSUI400, Oceanex, and UIEB100 show that while PSNR drops by a controlled 1.0 dB, computational savings enable real-time feasibility for large-scale marine monitoring. Additionally, uncertainty estimation and explainability techniques enhance trust in AI-driven environmental decisions. Comparisons with CycleGAN, FunIEGAN, RAUNENet, WaterNet, UGAN, PUGAN, and UTUIE validate EBA-AI's effectiveness in balancing efficiency, fairness, and interpretability in underwater image processing. By addressing key limitations of AI-driven enhancement, this work contributes to sustainable, bias-aware, and computationally efficient marine conservation efforts. For interactive visualizations, animations, source code, and access to the preprint, visit: https://lyessaadsaoud.github.io/EBA-AI/
>
---
#### [new 064] Event-based Graph Representation with Spatial and Motion Vectors for Asynchronous Object Detection
- **分类: cs.CV**

- **简介: 该论文属于异步目标检测任务，旨在解决事件相机数据因转换为稠密张量而丧失优势的问题。论文提出一种结合空间结构与运动向量的多图表示方法，构建空间与时间解耦的图模型，提升检测精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.15150v1](http://arxiv.org/pdf/2507.15150v1)**

> **作者:** Aayush Atul Verma; Arpitsinh Vaghela; Bharatesh Chakravarthi; Kaustav Chanda; Yezhou Yang
>
> **摘要:** Event-based sensors offer high temporal resolution and low latency by generating sparse, asynchronous data. However, converting this irregular data into dense tensors for use in standard neural networks diminishes these inherent advantages, motivating research into graph representations. While such methods preserve sparsity and support asynchronous inference, their performance on downstream tasks remains limited due to suboptimal modeling of spatiotemporal dynamics. In this work, we propose a novel spatiotemporal multigraph representation to better capture spatial structure and temporal changes. Our approach constructs two decoupled graphs: a spatial graph leveraging B-spline basis functions to model global structure, and a temporal graph utilizing motion vector-based attention for local dynamic changes. This design enables the use of efficient 2D kernels in place of computationally expensive 3D kernels. We evaluate our method on the Gen1 automotive and eTraM datasets for event-based object detection, achieving over a 6% improvement in detection accuracy compared to previous graph-based works, with a 5x speedup, reduced parameter count, and no increase in computational cost. These results highlight the effectiveness of structured graph modeling for asynchronous vision. Project page: eventbasedvision.github.io/eGSMV.
>
---
#### [new 065] Distilling Parallel Gradients for Fast ODE Solvers of Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型（DMs）采样延迟高、质量下降的问题。作者提出了一种并行ODE求解器EPD，通过并行梯度评估减少截断误差，并以蒸馏方式优化参数。该方法在低延迟下显著提升生成质量，适用于多种图像合成场景。**

- **链接: [http://arxiv.org/pdf/2507.14797v1](http://arxiv.org/pdf/2507.14797v1)**

> **作者:** Beier Zhu; Ruoyu Wang; Tong Zhao; Hanwang Zhang; Chi Zhang
>
> **备注:** To appear in ICCV 2025
>
> **摘要:** Diffusion models (DMs) have achieved state-of-the-art generative performance but suffer from high sampling latency due to their sequential denoising nature. Existing solver-based acceleration methods often face image quality degradation under a low-latency budget. In this paper, we propose the Ensemble Parallel Direction solver (dubbed as \ours), a novel ODE solver that mitigates truncation errors by incorporating multiple parallel gradient evaluations in each ODE step. Importantly, since the additional gradient computations are independent, they can be fully parallelized, preserving low-latency sampling. Our method optimizes a small set of learnable parameters in a distillation fashion, ensuring minimal training overhead. In addition, our method can serve as a plugin to improve existing ODE samplers. Extensive experiments on various image synthesis benchmarks demonstrate the effectiveness of our \ours~in achieving high-quality and low-latency sampling. For example, at the same latency level of 5 NFE, EPD achieves an FID of 4.47 on CIFAR-10, 7.97 on FFHQ, 8.17 on ImageNet, and 8.26 on LSUN Bedroom, surpassing existing learning-based solvers by a significant margin. Codes are available in https://github.com/BeierZhu/EPD.
>
---
#### [new 066] BleedOrigin: Dynamic Bleeding Source Localization in Endoscopic Submucosal Dissection via Dual-Stage Detection and Tracking
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗图像分析任务，旨在解决ESD手术中出血源精确定位与跟踪问题。现有AI方法侧重出血区域分割，缺乏对出血源动态检测与跟踪的研究。作者构建了首个出血源数据集BleedOrigin-Bench，并提出BleedOrigin-Net双阶段检测-跟踪框架，实现了出血起始检测与持续定位，提升了术中止血效率。**

- **链接: [http://arxiv.org/pdf/2507.15094v1](http://arxiv.org/pdf/2507.15094v1)**

> **作者:** Mengya Xu; Rulin Zhou; An Wang; Chaoyang Lyu; Zhen Li; Ning Zhong; Hongliang Ren
>
> **备注:** 27 pages, 14 figures
>
> **摘要:** Intraoperative bleeding during Endoscopic Submucosal Dissection (ESD) poses significant risks, demanding precise, real-time localization and continuous monitoring of the bleeding source for effective hemostatic intervention. In particular, endoscopists have to repeatedly flush to clear blood, allowing only milliseconds to identify bleeding sources, an inefficient process that prolongs operations and elevates patient risks. However, current Artificial Intelligence (AI) methods primarily focus on bleeding region segmentation, overlooking the critical need for accurate bleeding source detection and temporal tracking in the challenging ESD environment, which is marked by frequent visual obstructions and dynamic scene changes. This gap is widened by the lack of specialized datasets, hindering the development of robust AI-assisted guidance systems. To address these challenges, we introduce BleedOrigin-Bench, the first comprehensive ESD bleeding source dataset, featuring 1,771 expert-annotated bleeding sources across 106,222 frames from 44 procedures, supplemented with 39,755 pseudo-labeled frames. This benchmark covers 8 anatomical sites and 6 challenging clinical scenarios. We also present BleedOrigin-Net, a novel dual-stage detection-tracking framework for the bleeding source localization in ESD procedures, addressing the complete workflow from bleeding onset detection to continuous spatial tracking. We compare with widely-used object detection models (YOLOv11/v12), multimodal large language models, and point tracking methods. Extensive evaluation demonstrates state-of-the-art performance, achieving 96.85% frame-level accuracy ($\pm\leq8$ frames) for bleeding onset detection, 70.24% pixel-level accuracy ($\leq100$ px) for initial source detection, and 96.11% pixel-level accuracy ($\leq100$ px) for point tracking.
>
---
#### [new 067] Exploring Scalable Unified Modeling for General Low-Level Vision
- **分类: cs.CV**

- **简介: 该论文属于低层视觉任务，旨在解决多样化任务下的统一建模问题。作者提出了VPIP框架，通过可视提示引导模型处理多种低层视觉任务，并构建了统一模型GenLV。实验表明其在多种任务上表现优异，具备良好泛化能力和适应性。**

- **链接: [http://arxiv.org/pdf/2507.14801v1](http://arxiv.org/pdf/2507.14801v1)**

> **作者:** Xiangyu Chen; Kaiwen Zhu; Yuandong Pu; Shuo Cao; Xiaohui Li; Wenlong Zhang; Yihao Liu; Yu Qiao; Jiantao Zhou; Chao Dong
>
> **摘要:** Low-level vision involves a wide spectrum of tasks, including image restoration, enhancement, stylization, and feature extraction, which differ significantly in both task formulation and output domains. To address the challenge of unified modeling across such diverse tasks, we propose a Visual task Prompt-based Image Processing (VPIP) framework that leverages input-target image pairs as visual prompts to guide the model in performing a variety of low-level vision tasks. The framework comprises an end-to-end image processing backbone, a prompt encoder, and a prompt interaction module, enabling flexible integration with various architectures and effective utilization of task-specific visual representations. Based on this design, we develop a unified low-level vision model, GenLV, and evaluate its performance across multiple representative tasks. To explore the scalability of this approach, we extend the framework along two dimensions: model capacity and task diversity. We construct a large-scale benchmark consisting of over 100 low-level vision tasks and train multiple versions of the model with varying scales. Experimental results show that the proposed method achieves considerable performance across a wide range of tasks. Notably, increasing the number of training tasks enhances generalization, particularly for tasks with limited data, indicating the model's ability to learn transferable representations through joint training. Further evaluations in zero-shot generalization, few-shot transfer, and task-specific fine-tuning scenarios demonstrate the model's strong adaptability, confirming the effectiveness, scalability, and potential of the proposed framework as a unified foundation for general low-level vision modeling.
>
---
#### [new 068] CylinderPlane: Nested Cylinder Representation for 3D-aware Image Generation
- **分类: cs.CV; 68T45; I.4.5**

- **简介: 该论文属于3D-aware图像生成任务，旨在解决Tri-plane表示在360°视图生成中的多面伪影和特征模糊问题。论文提出CylinderPlane，利用圆柱坐标系统分离不同角度特征，并引入嵌套圆柱结构以适应复杂几何和多尺度细节，提升生成质量和视角一致性。**

- **链接: [http://arxiv.org/pdf/2507.15606v1](http://arxiv.org/pdf/2507.15606v1)**

> **作者:** Ru Jia; Xiaozhuang Ma; Jianji Wang; Nanning Zheng
>
> **备注:** 5 pages, 4 figures, to be published
>
> **摘要:** While the proposal of the Tri-plane representation has advanced the development of the 3D-aware image generative models, problems rooted in its inherent structure, such as multi-face artifacts caused by sharing the same features in symmetric regions, limit its ability to generate 360$^\circ$ view images. In this paper, we propose CylinderPlane, a novel implicit representation based on Cylindrical Coordinate System, to eliminate the feature ambiguity issue and ensure multi-view consistency in 360$^\circ$. Different from the inevitable feature entanglement in Cartesian coordinate-based Tri-plane representation, the cylindrical coordinate system explicitly separates features at different angles, allowing our cylindrical representation possible to achieve high-quality, artifacts-free 360$^\circ$ image synthesis. We further introduce the nested cylinder representation that composites multiple cylinders at different scales, thereby enabling the model more adaptable to complex geometry and varying resolutions. The combination of cylinders with different resolutions can effectively capture more critical locations and multi-scale features, greatly facilitates fine detail learning and robustness to different resolutions. Moreover, our representation is agnostic to implicit rendering methods and can be easily integrated into any neural rendering pipeline. Extensive experiments on both synthetic dataset and unstructured in-the-wild images demonstrate that our proposed representation achieves superior performance over previous methods.
>
---
#### [new 069] ExDD: Explicit Dual Distribution Learning for Surface Defect Detection via Diffusion Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 论文属于工业表面缺陷检测任务，旨在解决单类异常检测方法在分布假设和数据不足上的局限。作者提出ExDD框架，通过双分布建模和扩散合成生成带缺陷的样本，并引入邻域感知评分机制提升检测效果，实验证明其性能优越。**

- **链接: [http://arxiv.org/pdf/2507.15335v1](http://arxiv.org/pdf/2507.15335v1)**

> **作者:** Muhammad Aqeel; Federico Leonardi; Francesco Setti
>
> **备注:** Accepted to ICIAP 2025
>
> **摘要:** Industrial defect detection systems face critical limitations when confined to one-class anomaly detection paradigms, which assume uniform outlier distributions and struggle with data scarcity in realworld manufacturing environments. We present ExDD (Explicit Dual Distribution), a novel framework that transcends these limitations by explicitly modeling dual feature distributions. Our approach leverages parallel memory banks that capture the distinct statistical properties of both normality and anomalous patterns, addressing the fundamental flaw of uniform outlier assumptions. To overcome data scarcity, we employ latent diffusion models with domain-specific textual conditioning, generating in-distribution synthetic defects that preserve industrial context. Our neighborhood-aware ratio scoring mechanism elegantly fuses complementary distance metrics, amplifying signals in regions exhibiting both deviation from normality and similarity to known defect patterns. Experimental validation on KSDD2 demonstrates superior performance (94.2% I-AUROC, 97.7% P-AUROC), with optimal augmentation at 100 synthetic samples.
>
---
#### [new 070] Appearance Harmonization via Bilateral Grid Prediction with Transformers for 3DGS
- **分类: cs.CV**

- **简介: 该论文属于图像重建任务，旨在解决多视角图像中因相机处理导致的光度不一致问题。通过引入基于Transformer的双边网格预测方法，实现跨场景泛化，提升3D高斯泼溅重建质量，同时保持训练高效性。**

- **链接: [http://arxiv.org/pdf/2507.15748v1](http://arxiv.org/pdf/2507.15748v1)**

> **作者:** Jisu Shin; Richard Shaw; Seunghyun Shin; Anton Pelykh; Zhensong Zhang; Hae-Gon Jeon; Eduardo Perez-Pellitero
>
> **备注:** 10 pages, 3 figures, NeurIPS 2025 under review
>
> **摘要:** Modern camera pipelines apply extensive on-device processing, such as exposure adjustment, white balance, and color correction, which, while beneficial individually, often introduce photometric inconsistencies across views. These appearance variations violate multi-view consistency and degrade the quality of novel view synthesis. Joint optimization of scene representations and per-image appearance embeddings has been proposed to address this issue, but at the cost of increased computational complexity and slower training. In this work, we propose a transformer-based method that predicts spatially adaptive bilateral grids to correct photometric variations in a multi-view consistent manner, enabling robust cross-scene generalization without the need for scene-specific retraining. By incorporating the learned grids into the 3D Gaussian Splatting pipeline, we improve reconstruction quality while maintaining high training efficiency. Extensive experiments show that our approach outperforms or matches existing scene-specific optimization methods in reconstruction fidelity and convergence speed.
>
---
#### [new 071] Regularized Low-Rank Adaptation for Few-Shot Organ Segmentation
- **分类: cs.CV**

- **简介: 论文提出一种用于医学图像分割的参数高效微调方法，解决预设低秩适应中固定秩选择困难的问题。通过引入l_1稀疏正则化动态调整秩，自动适应任务需求，在少样本设置下显著提升了性能。**

- **链接: [http://arxiv.org/pdf/2507.15793v1](http://arxiv.org/pdf/2507.15793v1)**

> **作者:** Ghassen Baklouti; Julio Silva-Rodríguez; Jose Dolz; Houda Bahig; Ismail Ben Ayed
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) of pre-trained foundation models is increasingly attracting interest in medical imaging due to its effectiveness and computational efficiency. Among these methods, Low-Rank Adaptation (LoRA) is a notable approach based on the assumption that the adaptation inherently occurs in a low-dimensional subspace. While it has shown good performance, its implementation requires a fixed and unalterable rank, which might be challenging to select given the unique complexities and requirements of each medical imaging downstream task. Inspired by advancements in natural image processing, we introduce a novel approach for medical image segmentation that dynamically adjusts the intrinsic rank during adaptation. Viewing the low-rank representation of the trainable weight matrices as a singular value decomposition, we introduce an l_1 sparsity regularizer to the loss function, and tackle it with a proximal optimizer. The regularizer could be viewed as a penalty on the decomposition rank. Hence, its minimization enables to find task-adapted ranks automatically. Our method is evaluated in a realistic few-shot fine-tuning setting, where we compare it first to the standard LoRA and then to several other PEFT methods across two distinguishable tasks: base organs and novel organs. Our extensive experiments demonstrate the significant performance improvements driven by our method, highlighting its efficiency and robustness against suboptimal rank initialization. Our code is publicly available: https://github.com/ghassenbaklouti/ARENA
>
---
#### [new 072] LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于点云几何压缩任务，旨在解决现有AI方法依赖训练数据分布和有损压缩的问题。论文提出LINR-PCGC，首次实现基于隐式神经表示的无损点云压缩，通过分组编码框架和轻量网络设计，显著提升编码速度并减小解码器规模，取得了优于传统和AI方法的压缩性能。**

- **链接: [http://arxiv.org/pdf/2507.15686v1](http://arxiv.org/pdf/2507.15686v1)**

> **作者:** Wenjie Huang; Qi Yang; Shuting Xia; He Huang; Zhu Li; Yiling Xu
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Existing AI-based point cloud compression methods struggle with dependence on specific training data distributions, which limits their real-world deployment. Implicit Neural Representation (INR) methods solve the above problem by encoding overfitted network parameters to the bitstream, resulting in more distribution-agnostic results. However, due to the limitation of encoding time and decoder size, current INR based methods only consider lossy geometry compression. In this paper, we propose the first INR based lossless point cloud geometry compression method called Lossless Implicit Neural Representations for Point Cloud Geometry Compression (LINR-PCGC). To accelerate encoding speed, we design a group of point clouds level coding framework with an effective network initialization strategy, which can reduce around 60% encoding time. A lightweight coding network based on multiscale SparseConv, consisting of scale context extraction, child node prediction, and model compression modules, is proposed to realize fast inference and compact decoder size. Experimental results show that our method consistently outperforms traditional and AI-based methods: for example, with the convergence time in the MVUB dataset, our method reduces the bitstream by approximately 21.21% compared to G-PCC TMC13v23 and 21.95% compared to SparsePCGC. Our project can be seen on https://huangwenjie2023.github.io/LINR-PCGC/.
>
---
#### [new 073] Axis-Aligned Document Dewarping
- **分类: cs.CV**

- **简介: 该论文属于文档图像处理任务，旨在解决文档去扭曲问题。现有方法依赖监督回归，未利用文档几何特性。作者提出利用轴对齐特性，在训练中引入几何约束，推理时使用轴对齐预处理，并设计新评价指标AAD，显著提升去扭曲效果，取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2507.15000v1](http://arxiv.org/pdf/2507.15000v1)**

> **作者:** Chaoyun Wang; I-Chao Shen; Takeo Igarashi; Nanning Zheng; Caigui Jiang
>
> **摘要:** Document dewarping is crucial for many applications. However, existing learning-based methods primarily rely on supervised regression with annotated data without leveraging the inherent geometric properties in physical documents to the dewarping process. Our key insight is that a well-dewarped document is characterized by transforming distorted feature lines into axis-aligned ones. This property aligns with the inherent axis-aligned nature of the discrete grid geometry in planar documents. In the training phase, we propose an axis-aligned geometric constraint to enhance document dewarping. In the inference phase, we propose an axis alignment preprocessing strategy to reduce the dewarping difficulty. In the evaluation phase, we introduce a new metric, Axis-Aligned Distortion (AAD), that not only incorporates geometric meaning and aligns with human visual perception but also demonstrates greater robustness. As a result, our method achieves SOTA results on multiple existing benchmarks and achieves 18.2%~34.5% improvements on the AAD metric.
>
---
#### [new 074] Clutter Detection and Removal by Multi-Objective Analysis for Photographic Guidance
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于图像处理与摄影指导任务，旨在解决照片中杂乱物体干扰美观的问题。作者提出了一种基于多目标分析的杂乱检测与去除系统，通过美学评估算法和生成对抗网络实现杂乱物体识别与修复，帮助用户提升摄影质量。**

- **链接: [http://arxiv.org/pdf/2507.14553v1](http://arxiv.org/pdf/2507.14553v1)**

> **作者:** Xiaoran Wu
>
> **摘要:** Clutter in photos is a distraction preventing photographers from conveying the intended emotions or stories to the audience. Photography amateurs frequently include clutter in their photos due to unconscious negligence or the lack of experience in creating a decluttered, aesthetically appealing scene for shooting. We are thus motivated to develop a camera guidance system that provides solutions and guidance for clutter identification and removal. We estimate and visualize the contribution of objects to the overall aesthetics and content of a photo, based on which users can interactively identify clutter. Suggestions on getting rid of clutter, as well as a tool that removes cluttered objects computationally, are provided to guide users to deal with different kinds of clutter and improve their photographic work. Two technical novelties underpin interactions in our system: a clutter distinguishment algorithm with aesthetics evaluations for objects and an iterative image inpainting algorithm based on generative adversarial nets that reconstructs missing regions of removed objects for high-resolution images. User studies demonstrate that our system provides flexible interfaces and accurate algorithms that allow users to better identify distractions and take higher quality images within less time.
>
---
#### [new 075] Multimodal AI for Gastrointestinal Diagnostics: Tackling VQA in MEDVQA-GI 2025
- **分类: cs.CV; cs.AI; 68T45 (Machine vision and scene understanding); I.2.10; I.4.8; H.3.1**

- **简介: 该论文属于医学图像分析任务，旨在解决胃肠内镜图像的视觉问答（VQA）问题。作者基于Florence多模态模型构建VQA系统，并结合医学特征增强的数据增强策略，提升模型泛化能力，在KASVIR数据集上表现良好，为医学VQA提供了有效基线。**

- **链接: [http://arxiv.org/pdf/2507.14544v1](http://arxiv.org/pdf/2507.14544v1)**

> **作者:** Sujata Gaihre; Amir Thapa Magar; Prasuna Pokharel; Laxmi Tiwari
>
> **备注:** accepted to ImageCLEF 2025, to be published in the lab proceedings
>
> **摘要:** This paper describes our approach to Subtask 1 of the ImageCLEFmed MEDVQA 2025 Challenge, which targets visual question answering (VQA) for gastrointestinal endoscopy. We adopt the Florence model-a large-scale multimodal foundation model-as the backbone of our VQA pipeline, pairing a powerful vision encoder with a text encoder to interpret endoscopic images and produce clinically relevant answers. To improve generalization, we apply domain-specific augmentations that preserve medical features while increasing training diversity. Experiments on the KASVIR dataset show that fine-tuning Florence yields accurate responses on the official challenge metrics. Our results highlight the potential of large multimodal models in medical VQA and provide a strong baseline for future work on explainability, robustness, and clinical integration. The code is publicly available at: https://github.com/TiwariLaxuu/VQA-Florence.git
>
---
#### [new 076] Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决现有3D高斯点绘重建方法依赖大量数据和计算资源、重建速度慢的问题。作者提出Stereo-GS，通过立体视觉模型提取局部图像对特征，结合全局注意力机制进行融合，分别预测几何和外观信息，并通过优化网络提升重建质量，实现了无需相机参数的高效3D重建。**

- **链接: [http://arxiv.org/pdf/2507.14921v1](http://arxiv.org/pdf/2507.14921v1)**

> **作者:** Xiufeng Huang; Ka Chun Cheung; Runmin Cong; Simon See; Renjie Wan
>
> **备注:** ACMMM2025. Non-camera-ready version
>
> **摘要:** Generalizable 3D Gaussian Splatting reconstruction showcases advanced Image-to-3D content creation but requires substantial computational resources and large datasets, posing challenges to training models from scratch. Current methods usually entangle the prediction of 3D Gaussian geometry and appearance, which rely heavily on data-driven priors and result in slow regression speeds. To address this, we propose \method, a disentangled framework for efficient 3D Gaussian prediction. Our method extracts features from local image pairs using a stereo vision backbone and fuses them via global attention blocks. Dedicated point and Gaussian prediction heads generate multi-view point-maps for geometry and Gaussian features for appearance, combined as GS-maps to represent the 3DGS object. A refinement network enhances these GS-maps for high-quality reconstruction. Unlike existing methods that depend on camera parameters, our approach achieves pose-free 3D reconstruction, improving robustness and practicality. By reducing resource demands while maintaining high-quality outputs, \method provides an efficient, scalable solution for real-world 3D content generation.
>
---
#### [new 077] A Novel Downsampling Strategy Based on Information Complementarity for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决传统下采样方法导致关键空间信息丢失的问题。作者提出了一种基于信息互补性的下采样方法——混合池化下采样（HPD），通过使用MinMaxPooling保留图像的明暗对比和细节特征，提升了分割性能，在ACDC和Synapse数据集上平均DSC系数提高了0.5%。**

- **链接: [http://arxiv.org/pdf/2507.14790v1](http://arxiv.org/pdf/2507.14790v1)**

> **作者:** Wenbo Yue; Chang Li; Guoping Xu
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** In convolutional neural networks (CNNs), downsampling operations are crucial to model performance. Although traditional downsampling methods (such as maximum pooling and cross-row convolution) perform well in feature aggregation, receptive field expansion, and computational reduction, they may lead to the loss of key spatial information in semantic segmentation tasks, thereby affecting the pixel-by-pixel prediction accuracy.To this end, this study proposes a downsampling method based on information complementarity - Hybrid Pooling Downsampling (HPD). The core is to replace the traditional method with MinMaxPooling, and effectively retain the light and dark contrast and detail features of the image by extracting the maximum value information of the local area.Experiment on various CNN architectures on the ACDC and Synapse datasets show that HPD outperforms traditional methods in segmentation performance, and increases the DSC coefficient by 0.5% on average. The results show that the HPD module provides an efficient solution for semantic segmentation tasks.
>
---
#### [new 078] Aesthetics is Cheap, Show me the Text: An Empirical Evaluation of State-of-the-Art Generative Models for OCR
- **分类: cs.CV**

- **简介: 该论文属于图像生成与OCR任务，旨在评估当前生成模型在文本图像生成与编辑中的表现。论文提出OCR生成任务概念，涵盖五大类33项任务，综合测试6个先进模型，揭示其在文本图像生成中的不足，主张将该能力集成至通用生成模型中。**

- **链接: [http://arxiv.org/pdf/2507.15085v1](http://arxiv.org/pdf/2507.15085v1)**

> **作者:** Peirong Zhang; Haowei Xu; Jiaxin Zhang; Guitao Xu; Xuhan Zheng; Zhenhua Yang; Junle Liu; Yuyi Zhang; Lianwen Jin
>
> **摘要:** Text image is a unique and crucial information medium that integrates visual aesthetics and linguistic semantics in modern e-society. Due to their subtlety and complexity, the generation of text images represents a challenging and evolving frontier in the image generation field. The recent surge of specialized image generators (\emph{e.g.}, Flux-series) and unified generative models (\emph{e.g.}, GPT-4o), which demonstrate exceptional fidelity, raises a natural question: can they master the intricacies of text image generation and editing? Motivated by this, we assess current state-of-the-art generative models' capabilities in terms of text image generation and editing. We incorporate various typical optical character recognition (OCR) tasks into our evaluation and broaden the concept of text-based generation tasks into OCR generative tasks. We select 33 representative tasks and categorize them into five categories: document, handwritten text, scene text, artistic text, and complex \& layout-rich text. For comprehensive evaluation, we examine six models across both closed-source and open-source domains, using tailored, high-quality image inputs and prompts. Through this evaluation, we draw crucial observations and identify the weaknesses of current generative models for OCR tasks. We argue that photorealistic text image generation and editing should be internalized as foundational skills into general-domain generative models, rather than being delegated to specialized solutions, and we hope this empirical analysis can provide valuable insights for the community to achieve this goal. This evaluation is online and will be continuously updated at our GitHub repository.
>
---
#### [new 079] GeMix: Conditional GAN-Based Mixup for Improved Medical Image Augmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决传统Mixup方法生成图像不真实的问题。作者提出GeMix方法，使用基于条件GAN的混合策略，生成更逼真、标签感知的医学图像，提升了分类性能，尤其降低了新冠肺炎检测的假阴性率。**

- **链接: [http://arxiv.org/pdf/2507.15577v1](http://arxiv.org/pdf/2507.15577v1)**

> **作者:** Hugo Carlesso; Maria Eliza Patulea; Moncef Garouani; Radu Tudor Ionescu; Josiane Mothe
>
> **摘要:** Mixup has become a popular augmentation strategy for image classification, yet its naive pixel-wise interpolation often produces unrealistic images that can hinder learning, particularly in high-stakes medical applications. We propose GeMix, a two-stage framework that replaces heuristic blending with a learned, label-aware interpolation powered by class-conditional GANs. First, a StyleGAN2-ADA generator is trained on the target dataset. During augmentation, we sample two label vectors from Dirichlet priors biased toward different classes and blend them via a Beta-distributed coefficient. Then, we condition the generator on this soft label to synthesize visually coherent images that lie along a continuous class manifold. We benchmark GeMix on the large-scale COVIDx-CT-3 dataset using three backbones (ResNet-50, ResNet-101, EfficientNet-B0). When combined with real data, our method increases macro-F1 over traditional mixup for all backbones, reducing the false negative rate for COVID-19 detection. GeMix is thus a drop-in replacement for pixel-space mixup, delivering stronger regularization and greater semantic fidelity, without disrupting existing training pipelines. We publicly release our code at https://github.com/hugocarlesso/GeMix to foster reproducibility and further research.
>
---
#### [new 080] Mammo-SAE: Interpreting Breast Cancer Concept Learning with Sparse Autoencoders
- **分类: cs.CV**

- **简介: 该论文属于医学图像解释任务，旨在提升乳腺癌概念学习的可解释性。通过在乳腺X线图像-报告数据上预训练的Mammo-CLIP模型中引入稀疏自编码器（Mammo-SAE），识别与临床相关特征（如肿块、可疑钙化）相关的潜在特征，揭示模型决策机制并分析微调过程中的关键神经元。**

- **链接: [http://arxiv.org/pdf/2507.15227v1](http://arxiv.org/pdf/2507.15227v1)**

> **作者:** Krishna Kanth Nakka
>
> **备注:** Preprint. Under review
>
> **摘要:** Interpretability is critical in high-stakes domains such as medical imaging, where understanding model decisions is essential for clinical adoption. In this work, we introduce Sparse Autoencoder (SAE)-based interpretability to breast imaging by analyzing {Mammo-CLIP}, a vision--language foundation model pretrained on large-scale mammogram image--report pairs. We train a patch-level \texttt{Mammo-SAE} on Mammo-CLIP to identify and probe latent features associated with clinically relevant breast concepts such as \textit{mass} and \textit{suspicious calcification}. Our findings reveal that top activated class level latent neurons in the SAE latent space often tend to align with ground truth regions, and also uncover several confounding factors influencing the model's decision-making process. Additionally, we analyze which latent neurons the model relies on during downstream finetuning for improving the breast concept prediction. This study highlights the promise of interpretable SAE latent representations in providing deeper insight into the internal workings of foundation models at every layer for breast imaging.
>
---
#### [new 081] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决扩散模型在资源受限环境下部署困难的问题。作者提出了SegQuant框架，结合图量化策略与双尺度量化方法，提升模型压缩效果与通用性，无需重新训练即可兼容主流部署工具。**

- **链接: [http://arxiv.org/pdf/2507.14811v1](http://arxiv.org/pdf/2507.14811v1)**

> **作者:** Jiaji Zhang; Ruichao Sun; Hailiang Zhao; Jiaju Wu; Peng Chen; Hao Li; Xinkui Zhao; Kingsum Chow; Gang Xiong; Lin Ye; Shuiguang Deng
>
> **摘要:** Diffusion models have demonstrated exceptional generative capabilities but are computationally intensive, posing significant challenges for deployment in resource-constrained or latency-sensitive environments. Quantization offers an effective means to reduce model size and computational cost, with post-training quantization (PTQ) being particularly appealing due to its compatibility with pre-trained models without requiring retraining or training data. However, existing PTQ methods for diffusion models often rely on architecture-specific heuristics that limit their generalizability and hinder integration with industrial deployment pipelines. To address these limitations, we propose SegQuant, a unified quantization framework that adaptively combines complementary techniques to enhance cross-model versatility. SegQuant consists of a segment-aware, graph-based quantization strategy (SegLinear) that captures structural semantics and spatial heterogeneity, along with a dual-scale quantization scheme (DualScale) that preserves polarity-asymmetric activations, which is crucial for maintaining visual fidelity in generated outputs. SegQuant is broadly applicable beyond Transformer-based diffusion models, achieving strong performance while ensuring seamless compatibility with mainstream deployment tools.
>
---
#### [new 082] Exp-Graph: How Connections Learn Facial Attributes in Graph-based Expression Recognition
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于面部表情识别任务，旨在解决如何有效结合面部属性结构信息以提升识别准确率的问题。作者提出了Exp-Graph框架，利用图卷积网络和视觉Transformer建模面部属性间的局部与全局依赖关系，从而学习更具表达力的语义表示，实验证明其在多种场景下具有良好的识别性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.14608v1](http://arxiv.org/pdf/2507.14608v1)**

> **作者:** Nandani Sharma; Dinesh Singh
>
> **摘要:** Facial expression recognition is crucial for human-computer interaction applications such as face animation, video surveillance, affective computing, medical analysis, etc. Since the structure of facial attributes varies with facial expressions, incorporating structural information into facial attributes is essential for facial expression recognition. In this paper, we propose Exp-Graph, a novel framework designed to represent the structural relationships among facial attributes using graph-based modeling for facial expression recognition. For facial attributes graph representation, facial landmarks are used as the graph's vertices. At the same time, the edges are determined based on the proximity of the facial landmark and the similarity of the local appearance of the facial attributes encoded using the vision transformer. Additionally, graph convolutional networks are utilized to capture and integrate these structural dependencies into the encoding of facial attributes, thereby enhancing the accuracy of expression recognition. Thus, Exp-Graph learns from the facial attribute graphs highly expressive semantic representations. On the other hand, the vision transformer and graph convolutional blocks help the framework exploit the local and global dependencies among the facial attributes that are essential for the recognition of facial expressions. We conducted comprehensive evaluations of the proposed Exp-Graph model on three benchmark datasets: Oulu-CASIA, eNTERFACE05, and AFEW. The model achieved recognition accuracies of 98.09\%, 79.01\%, and 56.39\%, respectively. These results indicate that Exp-Graph maintains strong generalization capabilities across both controlled laboratory settings and real-world, unconstrained environments, underscoring its effectiveness for practical facial expression recognition applications.
>
---
#### [new 083] Comparative Analysis of Algorithms for the Fitting of Tessellations to 3D Image Data
- **分类: cs.CV; cond-mat.mtrl-sci; math.OC**

- **简介: 该论文属于图像建模与优化任务，旨在解决如何准确拟合3D图像数据（如多晶体和泡沫材料）到特定的镶嵌模型（如Voronoi、Laguerre及广义平衡幂图）。作者比较了多种优化算法（如线性/非线性规划、交叉熵方法、梯度下降）在模型拟合中的表现，评估模型复杂度、优化复杂度与拟合质量之间的权衡。**

- **链接: [http://arxiv.org/pdf/2507.14268v1](http://arxiv.org/pdf/2507.14268v1)**

> **作者:** Andreas Alpers; Orkun Furat; Christian Jung; Matthias Neumann; Claudia Redenbach; Aigerim Saken; Volker Schmidt
>
> **备注:** 31 pages, 16 figures, 8 tables
>
> **摘要:** This paper presents a comparative analysis of algorithmic strategies for fitting tessellation models to 3D image data of materials such as polycrystals and foams. In this steadily advancing field, we review and assess optimization-based methods -- including linear and nonlinear programming, stochastic optimization via the cross-entropy method, and gradient descent -- for generating Voronoi, Laguerre, and generalized balanced power diagrams (GBPDs) that approximate voxelbased grain structures. The quality of fit is evaluated on real-world datasets using discrepancy measures that quantify differences in grain volume, surface area, and topology. Our results highlight trade-offs between model complexity, the complexity of the optimization routines involved, and the quality of approximation, providing guidance for selecting appropriate methods based on data characteristics and application needs.
>
---
#### [new 084] OptiCorNet: Optimizing Sequence-Based Context Correlation for Visual Place Recognition
- **分类: cs.CV**

- **简介: 该论文属于视觉地点识别（VPR）任务，旨在解决动态环境和外观变化下的长期定位难题。现有方法忽略图像序列的时间一致性，而该文提出OptiCorNet，结合空间特征提取与时间差分建模，通过轻量级编码器与可学习时间差分模块（DSD），提升识别鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.14477v1](http://arxiv.org/pdf/2507.14477v1)**

> **作者:** Zhenyu Li; Tianyi Shang; Pengjie Xu; Ruirui Zhang; Fanchen Kong
>
> **备注:** 5 figures
>
> **摘要:** Visual Place Recognition (VPR) in dynamic and perceptually aliased environments remains a fundamental challenge for long-term localization. Existing deep learning-based solutions predominantly focus on single-frame embeddings, neglecting the temporal coherence present in image sequences. This paper presents OptiCorNet, a novel sequence modeling framework that unifies spatial feature extraction and temporal differencing into a differentiable, end-to-end trainable module. Central to our approach is a lightweight 1D convolutional encoder combined with a learnable differential temporal operator, termed Differentiable Sequence Delta (DSD), which jointly captures short-term spatial context and long-range temporal transitions. The DSD module models directional differences across sequences via a fixed-weight differencing kernel, followed by an LSTM-based refinement and optional residual projection, yielding compact, discriminative descriptors robust to viewpoint and appearance shifts. To further enhance inter-class separability, we incorporate a quadruplet loss that optimizes both positive alignment and multi-negative divergence within each batch. Unlike prior VPR methods that treat temporal aggregation as post-processing, OptiCorNet learns sequence-level embeddings directly, enabling more effective end-to-end place recognition. Comprehensive evaluations on multiple public benchmarks demonstrate that our approach outperforms state-of-the-art baselines under challenging seasonal and viewpoint variations.
>
---
#### [new 085] Hierarchical Part-based Generative Model for Realistic 3D Blood Vessel
- **分类: cs.CV**

- **简介: 该论文属于3D血管建模任务，旨在解决复杂血管几何与拓扑结构难以准确表示的问题。作者提出了一种基于层次化部件的生成框架，分阶段构建血管全局结构与局部几何细节，并通过真实数据验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.15223v1](http://arxiv.org/pdf/2507.15223v1)**

> **作者:** Siqi Chen; Guoqing Zhang; Jiahao Lai; Bingzhi Shen; Sihong Zhang; Caixia Dong; Xuejin Chen; Yang Li
>
> **摘要:** Advancements in 3D vision have increased the impact of blood vessel modeling on medical applications. However, accurately representing the complex geometry and topology of blood vessels remains a challenge due to their intricate branching patterns, curvatures, and irregular shapes. In this study, we propose a hierarchical part-based frame work for 3D vessel generation that separates the global binary tree-like topology from local geometric details. Our approach proceeds in three stages: (1) key graph generation to model the overall hierarchical struc ture, (2) vessel segment generation conditioned on geometric properties, and (3) hierarchical vessel assembly by integrating the local segments according to the global key graph. We validate our framework on real world datasets, demonstrating superior performance over existing methods in modeling complex vascular networks. This work marks the first successful application of a part-based generative approach for 3D vessel modeling, setting a new benchmark for vascular data generation. The code is available at: https://github.com/CybercatChen/PartVessel.git.
>
---
#### [new 086] Diffusion models for multivariate subsurface generation and efficient probabilistic inversion
- **分类: cs.CV; cs.LG; physics.geo-ph; stat.AP**

- **简介: 该论文属于生成建模与概率反演任务，旨在解决多变量地下建模与高效概率反演问题。作者提出改进的扩散模型方法，增强多变量地质建模能力，引入似然近似以处理噪声污染，并在多变量地质场景中验证性能，结果显示其在统计稳健性、后验采样效率和计算成本方面优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.15809v1](http://arxiv.org/pdf/2507.15809v1)**

> **作者:** Roberto Miele; Niklas Linde
>
> **摘要:** Diffusion models offer stable training and state-of-the-art performance for deep generative modeling tasks. Here, we consider their use in the context of multivariate subsurface modeling and probabilistic inversion. We first demonstrate that diffusion models enhance multivariate modeling capabilities compared to variational autoencoders and generative adversarial networks. In diffusion modeling, the generative process involves a comparatively large number of time steps with update rules that can be modified to account for conditioning data. We propose different corrections to the popular Diffusion Posterior Sampling approach by Chung et al. (2023). In particular, we introduce a likelihood approximation accounting for the noise-contamination that is inherent in diffusion modeling. We assess performance in a multivariate geological scenario involving facies and correlated acoustic impedance. Conditional modeling is demonstrated using both local hard data (well logs) and nonlinear geophysics (fullstack seismic data). Our tests show significantly improved statistical robustness, enhanced sampling of the posterior probability density function and reduced computational costs, compared to the original approach. The method can be used with both hard and indirect conditioning data, individually or simultaneously. As the inversion is included within the diffusion process, it is faster than other methods requiring an outer-loop around the generative model, such as Markov chain Monte Carlo.
>
---
#### [new 087] BeatFormer: Efficient motion-robust remote heart rate estimation through unsupervised spectral zoomed attention filters
- **分类: cs.CV**

- **简介: 论文提出BeatFormer，用于远程心率估计（rPPG）任务，旨在解决运动干扰下心率检测的鲁棒性与效率问题。通过结合注意力机制与频域分析，提升复杂场景下的性能，并引入无监督学习方法SCL，实现无需标签训练。**

- **链接: [http://arxiv.org/pdf/2507.14885v1](http://arxiv.org/pdf/2507.14885v1)**

> **作者:** Joaquim Comas; Federico Sukno
>
> **摘要:** Remote photoplethysmography (rPPG) captures cardiac signals from facial videos and is gaining attention for its diverse applications. While deep learning has advanced rPPG estimation, it relies on large, diverse datasets for effective generalization. In contrast, handcrafted methods utilize physiological priors for better generalization in unseen scenarios like motion while maintaining computational efficiency. However, their linear assumptions limit performance in complex conditions, where deep learning provides superior pulsatile information extraction. This highlights the need for hybrid approaches that combine the strengths of both methods. To address this, we present BeatFormer, a lightweight spectral attention model for rPPG estimation, which integrates zoomed orthonormal complex attention and frequency-domain energy measurement, enabling a highly efficient model. Additionally, we introduce Spectral Contrastive Learning (SCL), which allows BeatFormer to be trained without any PPG or HR labels. We validate BeatFormer on the PURE, UBFC-rPPG, and MMPD datasets, demonstrating its robustness and performance, particularly in cross-dataset evaluations under motion scenarios.
>
---
#### [new 088] Semantic-Aware Representation Learning for Multi-label Image Classification
- **分类: cs.CV**

- **简介: 该论文属于多标签图像分类任务，旨在解决现有方法中图像表示存在噪声、定位不准确的问题。论文提出了一种语义感知表示学习方法（SARL），通过语义相关特征提取、基于最优传输的注意力机制和区域得分聚合策略，提升了多标签图像分类的性能。**

- **链接: [http://arxiv.org/pdf/2507.14918v1](http://arxiv.org/pdf/2507.14918v1)**

> **作者:** Ren-Dong Xie; Zhi-Fen He; Bo Li; Bin Liu; Jin-Yan Hu
>
> **摘要:** Multi-label image classification, an important research area in computer vision, focuses on identifying multiple labels or concepts within an image. Existing approaches often employ attention mechanisms or graph convolutional networks (GCNs) to learn image representation. However, this representation may contain noise and may not locate objects precisely. Therefore, this paper proposes a Semantic-Aware Representation Learning (SARL) for multi-label image classification. First, a label semantic-related feature learning module is utilized to extract semantic-related features. Then, an optimal transport-based attention mechanism is designed to obtain semantically aligned image representation. Finally, a regional score aggregation strategy is used for multi-label prediction. Experimental results on two benchmark datasets, PASCAL VOC 2007 and MS-COCO, demonstrate the superiority of SARL over existing methods.
>
---
#### [new 089] EgoPrune: Efficient Token Pruning for Egomotion Video Reasoning in Embodied Agent
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型中的视频推理任务，旨在解决具身智能体在第一视角运动视频（egomotion video）中进行高效推理的问题。针对现有方法无法有效利用第一视角视频的时空连续性与运动约束，作者提出了无需训练的token剪枝方法EgoPrune，包含关键帧选择、视角感知冗余过滤和基于MMR的token选择。实验表明该方法在多个基准上优于已有方法，并在边缘设备上展示了良好的部署效果。**

- **链接: [http://arxiv.org/pdf/2507.15428v1](http://arxiv.org/pdf/2507.15428v1)**

> **作者:** Jiaao Li; Kaiyuan Li; Chen Gao; Yong Li; Xinlei Chen
>
> **摘要:** Egomotion videos are first-person recordings where the view changes continuously due to the agent's movement. As they serve as the primary visual input for embodied AI agents, making egomotion video reasoning more efficient is therefore essential for real-world deployment. Recent advances in vision-language models have enabled strong multimodal reasoning capabilities, but their computational cost remains prohibitive for long, redundant video inputs. Existing token pruning methods, typically designed for third-person videos, fail to leverage the spatiotemporal continuity and motion constraints inherent in egomotion settings. To address this, we propose EgoPrune, a training-free token pruning method tailored for egomotion video reasoning. EgoPrune comprises three components: a keyframe selector adapted from EmbodiedR for temporally efficient sampling; Perspective-Aware Redundancy Filtering (PARF), which aligns visual tokens using perspective transformations and removes redundant tokens; and a Maximal Marginal Relevance (MMR)-based token selector that jointly considers visual-text relevance and intra-frame diversity. Experiments on two egomotion video benchmarks show that EgoPrune consistently outperforms prior training-free methods across various pruning ratios while significantly reducing FLOPs, memory usage, and latency. Moreover, we deploy EgoPrune on an embodied agent equipped with a Jetson Orin NX 16GB edge device, demonstrating its real-world efficiency and suitability for on-device egomotion video reasoning.
>
---
#### [new 090] Rethinking Occlusion in FER: A Semantic-Aware Perspective and Go Beyond
- **分类: cs.CV**

- **简介: 该论文属于面部表情识别（FER）任务，旨在解决面部遮挡和数据集偏差导致的识别不准问题。论文提出了ORSANet模型，引入多模态语义引导、多尺度交互模块和动态对抗损失，提升了遮挡情况下的识别效果，并构建了新数据集Occlu-FER。**

- **链接: [http://arxiv.org/pdf/2507.15401v1](http://arxiv.org/pdf/2507.15401v1)**

> **作者:** Huiyu Zhai; Xingxing Yang; Yalan Ye; Chenyang Li; Bin Fan; Changze Li
>
> **摘要:** Facial expression recognition (FER) is a challenging task due to pervasive occlusion and dataset biases. Especially when facial information is partially occluded, existing FER models struggle to extract effective facial features, leading to inaccurate classifications. In response, we present ORSANet, which introduces the following three key contributions: First, we introduce auxiliary multi-modal semantic guidance to disambiguate facial occlusion and learn high-level semantic knowledge, which is two-fold: 1) we introduce semantic segmentation maps as dense semantics prior to generate semantics-enhanced facial representations; 2) we introduce facial landmarks as sparse geometric prior to mitigate intrinsic noises in FER, such as identity and gender biases. Second, to facilitate the effective incorporation of these two multi-modal priors, we customize a Multi-scale Cross-interaction Module (MCM) to adaptively fuse the landmark feature and semantics-enhanced representations within different scales. Third, we design a Dynamic Adversarial Repulsion Enhancement Loss (DARELoss) that dynamically adjusts the margins of ambiguous classes, further enhancing the model's ability to distinguish similar expressions. We further construct the first occlusion-oriented FER dataset to facilitate specialized robustness analysis on various real-world occlusion conditions, dubbed Occlu-FER. Extensive experiments on both public benchmarks and Occlu-FER demonstrate that our proposed ORSANet achieves SOTA recognition performance. Code is publicly available at https://github.com/Wenyuzhy/ORSANet-master.
>
---
#### [new 091] HW-MLVQA: Elucidating Multilingual Handwritten Document Understanding with a Comprehensive VQA Benchmark
- **分类: cs.CV**

- **简介: 该论文属于多语言手写文档理解任务，旨在解决现有MLVQA模型在处理手写文档时能力不足的问题。作者构建了HW-MLVQA基准，包含1,600页手写文档和2,400个问答对，支持文本、图像及图文多模态评估，并测试OCR模型在无真实文本转录场景下的表现，以推动该领域研究进展。**

- **链接: [http://arxiv.org/pdf/2507.15655v1](http://arxiv.org/pdf/2507.15655v1)**

> **作者:** Aniket Pal; Ajoy Mondal; Minesh Mathew; C. V. Jawahar
>
> **备注:** This is a minor revision of the original paper submitted to IJDAR
>
> **摘要:** The proliferation of MultiLingual Visual Question Answering (MLVQA) benchmarks augments the capabilities of large language models (LLMs) and multi-modal LLMs, thereby enabling them to adeptly capture the intricate linguistic subtleties and visual complexities inherent across diverse languages. Despite its potential, the current MLVQA model struggles to fully utilize its capabilities when dealing with the extensive variety of handwritten documents. This article delineates HW-MLVQA, an avant-garde VQA benchmark meticulously crafted to mitigate the dearth of authentic Multilingual Handwritten document comprehension. HW-MLVQA encompasses an extensive collection of 1,600 handwritten Pages complemented by 2,400 question-answers. Furthermore, it provides a robust benchmark evaluation framework spanning three distinct modalities: text, image, and an integrated image & text modality. To simulate authentic real-world contexts devoid of ground truth textual transcriptions, we facilitates a rigorous assessment of proprietary and open-source OCR models. The benchmark aspires to facilitate pivotal advancements in multilingual handwritten document interpretation, fostering innovation and scholarly inquiry within this specialized domain.
>
---
#### [new 092] Decision PCR: Decision version of the Point Cloud Registration task
- **分类: cs.CV**

- **简介: 本文属于点云配准（PCR）任务，旨在解决低重叠情况下传统评估指标失效的问题。作者提出了Decision PCR任务，构建了相应数据集，并设计了深度学习分类器以评估配准质量。该方法提升了现有配准算法的性能，在3DLoMatch和ETH数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.14965v1](http://arxiv.org/pdf/2507.14965v1)**

> **作者:** Yaojie Zhang; Tianlun Huang; Weijun Wang; Wei Feng
>
> **摘要:** Low-overlap point cloud registration (PCR) remains a significant challenge in 3D vision. Traditional evaluation metrics, such as Maximum Inlier Count, become ineffective under extremely low inlier ratios. In this paper, we revisit the registration result evaluation problem and identify the Decision version of the PCR task as the fundamental problem. To address this Decision PCR task, we propose a data-driven approach. First, we construct a corresponding dataset based on the 3DMatch dataset. Then, a deep learning-based classifier is trained to reliably assess registration quality, overcoming the limitations of traditional metrics. To our knowledge, this is the first comprehensive study to address this task through a deep learning framework. We incorporate this classifier into standard PCR pipelines. When integrated with our approach, existing state-of-the-art PCR methods exhibit significantly enhanced registration performance. For example, combining our framework with GeoTransformer achieves a new SOTA registration recall of 86.97\% on the challenging 3DLoMatch benchmark. Our method also demonstrates strong generalization capabilities on the unseen outdoor ETH dataset.
>
---
#### [new 093] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决现有视频大模型在正确性和鲁棒性上与人类智能存在差距的问题。作者构建了Video-TT基准，包含1,000个YouTube Shorts视频及问答，用于评估模型对复杂视觉叙事的理解和对抗性问题的鲁棒性，揭示了模型与人类表现的显著差距。**

- **链接: [http://arxiv.org/pdf/2507.15028v1](http://arxiv.org/pdf/2507.15028v1)**

> **作者:** Yuanhan Zhang; Yunice Chew; Yuhao Dong; Aria Leo; Bo Hu; Ziwei Liu
>
> **备注:** ICCV 2025; Project page: https://zhangyuanhan-ai.github.io/video-tt/
>
> **摘要:** Human intelligence requires correctness and robustness, with the former being foundational for the latter. In video understanding, correctness ensures the accurate interpretation of visual content, and robustness maintains consistent performance in challenging conditions. Despite advances in video large language models (video LLMs), existing benchmarks inadequately reflect the gap between these models and human intelligence in maintaining correctness and robustness in video interpretation. We introduce the Video Thinking Test (Video-TT), to assess if video LLMs can interpret real-world videos as effectively as humans. Video-TT reflects genuine gaps in understanding complex visual narratives, and evaluates robustness against natural adversarial questions. Video-TT comprises 1,000 YouTube Shorts videos, each with one open-ended question and four adversarial questions that probe visual and narrative complexity. Our evaluation shows a significant gap between video LLMs and human performance.
>
---
#### [new 094] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention
- **分类: cs.CV**

- **简介: 论文属于通用类别发现（GCD）任务，旨在解决模型在处理无标签数据时因关注无关背景导致的注意力分散问题。作者提出了一种轻量级模块AF，包含TIME和TAP两个组件，用于自适应地剪枝非信息性特征，提升GCD性能。**

- **链接: [http://arxiv.org/pdf/2507.14315v1](http://arxiv.org/pdf/2507.14315v1)**

> **作者:** Qiyu Xu; Zhanxuan Hu; Yu Duan; Ercheng Pei; Yonghang Tai
>
> **摘要:** Generalized Category Discovery (GCD) aims to classify unlabeled data from both known and unknown categories by leveraging knowledge from labeled known categories. While existing methods have made notable progress, they often overlook a hidden stumbling block in GCD: distracted attention. Specifically, when processing unlabeled data, models tend to focus not only on key objects in the image but also on task-irrelevant background regions, leading to suboptimal feature extraction. To remove this stumbling block, we propose Attention Focusing (AF), an adaptive mechanism designed to sharpen the model's focus by pruning non-informative tokens. AF consists of two simple yet effective components: Token Importance Measurement (TIME) and Token Adaptive Pruning (TAP), working in a cascade. TIME quantifies token importance across multiple scales, while TAP prunes non-informative tokens by utilizing the multi-scale importance scores provided by TIME. AF is a lightweight, plug-and-play module that integrates seamlessly into existing GCD methods with minimal computational overhead. When incorporated into one prominent GCD method, SimGCD, AF achieves up to 15.4% performance improvement over the baseline with minimal computational overhead. The implementation code is provided in https://github.com/Afleve/AFGCD.
>
---
#### [new 095] AI-Powered Precision in Sport Taekwondo: Enhancing Fairness, Speed, and Trust in Competition (FST.ai)
- **分类: cs.CV; cs.AI; 68T45; I.2.10**

- **简介: 该论文属于体育科技任务，旨在解决竞技跆拳道中人工判分存在的主观性、延迟和不一致问题。作者提出了FST.ai框架，结合计算机视觉与深度学习，实现头部踢击动作的实时识别与评分，提升裁决的公正性、速度与透明度，并展示其跨运动项目的应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.14657v1](http://arxiv.org/pdf/2507.14657v1)**

> **作者:** Keivan Shariatmadar; Ahmad Osman
>
> **备注:** 24 pages, 9 figures
>
> **摘要:** The integration of Artificial Intelligence (AI) into sports officiating represents a paradigm shift in how decisions are made in competitive environments. Traditional manual systems, even when supported by Instant Video Replay (IVR), often suffer from latency, subjectivity, and inconsistent enforcement, undermining fairness and athlete trust. This paper introduces FST.ai, a novel AI-powered framework designed to enhance officiating in Sport Taekwondo, particularly focusing on the complex task of real-time head kick detection and scoring. Leveraging computer vision, deep learning, and edge inference, the system automates the identification and classification of key actions, significantly reducing decision time from minutes to seconds while improving consistency and transparency. Importantly, the methodology is not limited to Taekwondo. The underlying framework -- based on pose estimation, motion classification, and impact analysis -- can be adapted to a wide range of sports requiring action detection, such as judo, karate, fencing, or even team sports like football and basketball, where foul recognition or performance tracking is critical. By addressing one of Taekwondo's most challenging scenarios -- head kick scoring -- we demonstrate the robustness, scalability, and sport-agnostic potential of FST.ai to transform officiating standards across multiple disciplines.
>
---
#### [new 096] Gene-DML: Dual-Pathway Multi-Level Discrimination for Gene Expression Prediction from Histopathology Images
- **分类: cs.CV**

- **简介: 该论文属于基因表达预测任务，旨在解决从组织病理图像准确预测基因表达的问题。现有方法未能充分利用多层级跨模态表示对齐，限制了性能。论文提出Gene-DML框架，通过双路径多级判别结构，同时建模细粒度和结构级跨模态关系，提升预测准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.14670v1](http://arxiv.org/pdf/2507.14670v1)**

> **作者:** Yaxuan Song; Jianan Fan; Hang Chang; Weidong Cai
>
> **备注:** 16 pages, 15 tables, 8 figures
>
> **摘要:** Accurately predicting gene expression from histopathology images offers a scalable and non-invasive approach to molecular profiling, with significant implications for precision medicine and computational pathology. However, existing methods often underutilize the cross-modal representation alignment between histopathology images and gene expression profiles across multiple representational levels, thereby limiting their prediction performance. To address this, we propose Gene-DML, a unified framework that structures latent space through Dual-pathway Multi-Level discrimination to enhance correspondence between morphological and transcriptional modalities. The multi-scale instance-level discrimination pathway aligns hierarchical histopathology representations extracted at local, neighbor, and global levels with gene expression profiles, capturing scale-aware morphological-transcriptional relationships. In parallel, the cross-level instance-group discrimination pathway enforces structural consistency between individual (image/gene) instances and modality-crossed (gene/image, respectively) groups, strengthening the alignment across modalities. By jointly modelling fine-grained and structural-level discrimination, Gene-DML is able to learn robust cross-modal representations, enhancing both predictive accuracy and generalization across diverse biological contexts. Extensive experiments on public spatial transcriptomics datasets demonstrate that Gene-DML achieves state-of-the-art performance in gene expression prediction. The code and checkpoints will be released soon.
>
---
#### [new 097] Depthwise-Dilated Convolutional Adapters for Medical Object Tracking and Segmentation Using the Segment Anything Model 2
- **分类: cs.CV**

- **简介: 该论文属于医学视频目标跟踪与分割任务，旨在解决现有方法适应性差、需大量数据重训练的问题。作者提出DD-SAM2框架，在SAM2模型中引入深度空洞适配器（DD-Adapter），提升多尺度特征提取能力，实现小数据下高效微调，无需大量数据与高计算成本。**

- **链接: [http://arxiv.org/pdf/2507.14613v1](http://arxiv.org/pdf/2507.14613v1)**

> **作者:** Guoping Xu; Christopher Kabat; You Zhang
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** Recent advances in medical image segmentation have been driven by deep learning; however, most existing methods remain limited by modality-specific designs and exhibit poor adaptability to dynamic medical imaging scenarios. The Segment Anything Model 2 (SAM2) and its related variants, which introduce a streaming memory mechanism for real-time video segmentation, present new opportunities for prompt-based, generalizable solutions. Nevertheless, adapting these models to medical video scenarios typically requires large-scale datasets for retraining or transfer learning, leading to high computational costs and the risk of catastrophic forgetting. To address these challenges, we propose DD-SAM2, an efficient adaptation framework for SAM2 that incorporates a Depthwise-Dilated Adapter (DD-Adapter) to enhance multi-scale feature extraction with minimal parameter overhead. This design enables effective fine-tuning of SAM2 on medical videos with limited training data. Unlike existing adapter-based methods focused solely on static images, DD-SAM2 fully exploits SAM2's streaming memory for medical video object tracking and segmentation. Comprehensive evaluations on TrackRad2025 (tumor segmentation) and EchoNet-Dynamic (left ventricle tracking) datasets demonstrate superior performance, achieving Dice scores of 0.93 and 0.97, respectively. To the best of our knowledge, this work provides an initial attempt at systematically exploring adapter-based SAM2 fine-tuning for medical video segmentation and tracking. Code, datasets, and models will be publicly available at https://github.com/apple1986/DD-SAM2.
>
---
#### [new 098] Benchmarking GANs, Diffusion Models, and Flow Matching for T1w-to-T2w MRI Translation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像到图像翻译任务，旨在解决通过计算方法合成缺失的MRI对比图像以减少扫描时间的问题。论文工作包括对生成对抗网络、扩散模型和流匹配技术在三个MRI数据集上的性能进行基准测试与分析。**

- **链接: [http://arxiv.org/pdf/2507.14575v1](http://arxiv.org/pdf/2507.14575v1)**

> **作者:** Andrea Moschetto; Lemuel Puglisi; Alec Sargood; Pierluigi Dell'Acqua; Francesco Guarnera; Sebastiano Battiato; Daniele Ravì
>
> **摘要:** Magnetic Resonance Imaging (MRI) enables the acquisition of multiple image contrasts, such as T1-weighted (T1w) and T2-weighted (T2w) scans, each offering distinct diagnostic insights. However, acquiring all desired modalities increases scan time and cost, motivating research into computational methods for cross-modal synthesis. To address this, recent approaches aim to synthesize missing MRI contrasts from those already acquired, reducing acquisition time while preserving diagnostic quality. Image-to-image (I2I) translation provides a promising framework for this task. In this paper, we present a comprehensive benchmark of generative models$\unicode{x2013}$specifically, Generative Adversarial Networks (GANs), diffusion models, and flow matching (FM) techniques$\unicode{x2013}$for T1w-to-T2w 2D MRI I2I translation. All frameworks are implemented with comparable settings and evaluated on three publicly available MRI datasets of healthy adults. Our quantitative and qualitative analyses show that the GAN-based Pix2Pix model outperforms diffusion and FM-based methods in terms of structural fidelity, image quality, and computational efficiency. Consistent with existing literature, these results suggest that flow-based models are prone to overfitting on small datasets and simpler tasks, and may require more data to match or surpass GAN performance. These findings offer practical guidance for deploying I2I translation techniques in real-world MRI workflows and highlight promising directions for future research in cross-modal medical image synthesis. Code and models are publicly available at https://github.com/AndreaMoschetto/medical-I2I-benchmark.
>
---
#### [new 099] Efficient Whole Slide Pathology VQA via Token Compression
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于病理图像视觉问答（VQA）任务，旨在解决全切片图像（WSI）因高分辨率导致的计算资源消耗大、现有方法缺乏生成能力的问题。论文提出TCP-LLaVA模型，通过引入可训练压缩token，聚合多模态信息并减少输入长度，从而提升VQA准确率并降低资源消耗。**

- **链接: [http://arxiv.org/pdf/2507.14497v1](http://arxiv.org/pdf/2507.14497v1)**

> **作者:** Weimin Lyu; Qingqiao Hu; Kehan Qi; Zhan Shi; Wentao Huang; Saumya Gupta; Chao Chen
>
> **摘要:** Whole-slide images (WSIs) in pathology can reach up to 10,000 x 10,000 pixels, posing significant challenges for multimodal large language model (MLLM) due to long context length and high computational demands. Previous methods typically focus on patch-level analysis or slide-level classification using CLIP-based models with multi-instance learning, but they lack the generative capabilities needed for visual question answering (VQA). More recent MLLM-based approaches address VQA by feeding thousands of patch tokens directly into the language model, which leads to excessive resource consumption. To address these limitations, we propose Token Compression Pathology LLaVA (TCP-LLaVA), the first MLLM architecture to perform WSI VQA via token compression. TCP-LLaVA introduces a set of trainable compression tokens that aggregate visual and textual information through a modality compression module, inspired by the [CLS] token mechanism in BERT. Only the compressed tokens are forwarded to the LLM for answer generation, significantly reducing input length and computational cost. Experiments on ten TCGA tumor subtypes show that TCP-LLaVA outperforms existing MLLM baselines in VQA accuracy while reducing training resource consumption by a substantial margin.
>
---
#### [new 100] ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding
- **分类: cs.CV**

- **简介: 该论文属于图像美学评估任务，旨在解决现有方法缺乏细粒度属性分析和专家级理解的问题。作者提出了ArtiMuse模型和ArtiMuse-10K数据集，实现图像美学的精细评估与专业解析。**

- **链接: [http://arxiv.org/pdf/2507.14533v1](http://arxiv.org/pdf/2507.14533v1)**

> **作者:** Shuo Cao; Nan Ma; Jiayang Li; Xiaohui Li; Lihao Shao; Kaiwen Zhu; Yu Zhou; Yuandong Pu; Jiarui Wu; Jiaquan Wang; Bo Qu; Wenhai Wang; Yu Qiao; Dajuin Yao; Yihao Liu
>
> **备注:** 43 pages, 31 figures, 13 tables
>
> **摘要:** The rapid advancement of educational applications, artistic creation, and AI-generated content (AIGC) technologies has substantially increased practical requirements for comprehensive Image Aesthetics Assessment (IAA), particularly demanding methods capable of delivering both quantitative scoring and professional understanding. Multimodal Large Language Model (MLLM)-based IAA methods demonstrate stronger perceptual and generalization capabilities compared to traditional approaches, yet they suffer from modality bias (score-only or text-only) and lack fine-grained attribute decomposition, thereby failing to support further aesthetic assessment. In this paper, we present:(1) ArtiMuse, an innovative MLLM-based IAA model with Joint Scoring and Expert-Level Understanding capabilities; (2) ArtiMuse-10K, the first expert-curated image aesthetic dataset comprising 10,000 images spanning 5 main categories and 15 subcategories, each annotated by professional experts with 8-dimensional attributes analysis and a holistic score. Both the model and dataset will be made public to advance the field.
>
---
#### [new 101] TriCLIP-3D: A Unified Parameter-Efficient Framework for Tri-Modal 3D Visual Grounding based on CLIP
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D视觉定位任务，旨在解决现有方法依赖复杂多模型、训练效率低的问题。作者提出TriCLIP-3D，利用2D预训练模型CLIP统一处理RGB图像、文本和点云数据，通过适配器微调和GARF模块实现多模态特征融合，显著减少参数并提升性能。**

- **链接: [http://arxiv.org/pdf/2507.14904v1](http://arxiv.org/pdf/2507.14904v1)**

> **作者:** Fan Li; Zanyi Wang; Zeyi Huang; Guang Dai; Jingdong Wang; Mengmeng Wang
>
> **摘要:** 3D visual grounding allows an embodied agent to understand visual information in real-world 3D environments based on human instructions, which is crucial for embodied intelligence. Existing 3D visual grounding methods typically rely on separate encoders for different modalities (e.g., RGB images, text, and 3D point clouds), resulting in large and complex models that are inefficient to train. While some approaches use pre-trained 2D multi-modal models like CLIP for 3D tasks, they still struggle with aligning point cloud data to 2D encoders. As a result, these methods continue to depend on 3D encoders for feature extraction, further increasing model complexity and training inefficiency. In this paper, we propose a unified 2D pre-trained multi-modal network to process all three modalities (RGB images, text, and point clouds), significantly simplifying the architecture. By leveraging a 2D CLIP bi-modal model with adapter-based fine-tuning, this framework effectively adapts to the tri-modal setting, improving both adaptability and performance across modalities. Our Geometric-Aware 2D-3D Feature Recovery and Fusion (GARF) module is designed to fuse geometric multi-scale features from point clouds and images. We then integrate textual features for final modality fusion and introduce a multi-modal decoder to facilitate deep cross-modal understanding. Together, our method achieves unified feature extraction and fusion across the three modalities, enabling an end-to-end 3D visual grounding model. Compared to the baseline, our method reduces the number of trainable parameters by approximately 58\%, while achieving a 6.52\% improvement in the 3D detection task and a 6.25\% improvement in the 3D visual grounding task.
>
---
#### [new 102] Towards Holistic Surgical Scene Graph
- **分类: cs.CV**

- **简介: 该论文属于医疗图像理解任务，旨在提升手术场景图的表达能力。针对现有方法未能充分建模工具-动作-目标组合及操作手身份的问题，作者提出了Endoscapes-SG201数据集与SSG-Com方法，以更全面地表示手术场景中的关键要素。**

- **链接: [http://arxiv.org/pdf/2507.15541v1](http://arxiv.org/pdf/2507.15541v1)**

> **作者:** Jongmin Shin; Enki Cho; Ka Yong Kim; Jung Yong Kim; Seong Tae Kim; Namkee Oh
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Surgical scene understanding is crucial for computer-assisted intervention systems, requiring visual comprehension of surgical scenes that involves diverse elements such as surgical tools, anatomical structures, and their interactions. To effectively represent the complex information in surgical scenes, graph-based approaches have been explored to structurally model surgical entities and their relationships. Previous surgical scene graph studies have demonstrated the feasibility of representing surgical scenes using graphs. However, certain aspects of surgical scenes-such as diverse combinations of tool-action-target and the identity of the hand operating the tool-remain underexplored in graph-based representations, despite their importance. To incorporate these aspects into graph representations, we propose Endoscapes-SG201 dataset, which includes annotations for tool-action-target combinations and hand identity. We also introduce SSG-Com, a graph-based method designed to learn and represent these critical elements. Through experiments on downstream tasks such as critical view of safety assessment and action triplet recognition, we demonstrated the importance of integrating these essential scene graph components, highlighting their significant contribution to surgical scene understanding. The code and dataset are available at https://github.com/ailab-kyunghee/SSG-Com
>
---
#### [new 103] BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM
- **分类: cs.CV**

- **简介: 该论文属于多模态AI生成内容检测任务，旨在解决当前检测方法受限于单模态、无法有效识别跨模态合成内容的问题。作者提出了BusterX++框架，结合强化学习与多阶段训练，实现跨模态检测与解释，并构建了高质量基准GenBuster++用于评估。**

- **链接: [http://arxiv.org/pdf/2507.14632v1](http://arxiv.org/pdf/2507.14632v1)**

> **作者:** Haiquan Wen; Tianxiao Li; Zhenglin Huang; Yiwei He; Guangliang Cheng
>
> **摘要:** Recent advances in generative AI have dramatically improved image and video synthesis capabilities, significantly increasing the risk of misinformation through sophisticated fake content. In response, detection methods have evolved from traditional approaches to multimodal large language models (MLLMs), offering enhanced transparency and interpretability in identifying synthetic media. However, current detection systems remain fundamentally limited by their single-modality design. These approaches analyze images or videos separately, making them ineffective against synthetic content that combines multiple media formats. To address these challenges, we introduce \textbf{BusterX++}, a novel framework designed specifically for cross-modal detection and explanation of synthetic media. Our approach incorporates an advanced reinforcement learning (RL) post-training strategy that eliminates cold-start. Through Multi-stage Training, Thinking Reward, and Hybrid Reasoning, BusterX++ achieves stable and substantial performance improvements. To enable comprehensive evaluation, we also present \textbf{GenBuster++}, a cross-modal benchmark leveraging state-of-the-art image and video generation techniques. This benchmark comprises 4,000 images and video clips, meticulously curated by human experts using a novel filtering methodology to ensure high quality, diversity, and real-world applicability. Extensive experiments demonstrate the effectiveness and generalizability of our approach.
>
---
#### [new 104] LeAdQA: LLM-Driven Context-Aware Temporal Grounding for Video Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频问答（VideoQA）任务，旨在解决现有方法在处理长视频时难以精准捕捉关键事件及因果关系的问题。作者提出了LeAdQA，结合因果感知的查询优化与细粒度视觉定位，提升复杂问题的理解与回答准确性。**

- **链接: [http://arxiv.org/pdf/2507.14784v1](http://arxiv.org/pdf/2507.14784v1)**

> **作者:** Xinxin Dong; Baoyun Peng; Haokai Ma; Yufei Wang; Zixuan Dong; Fei Hu; Xiaodong Wang
>
> **摘要:** Video Question Answering (VideoQA) requires identifying sparse critical moments in long videos and reasoning about their causal relationships to answer semantically complex questions. While recent advances in multimodal learning have improved alignment and fusion, current approaches remain limited by two prevalent but fundamentally flawed strategies: (1) task-agnostic sampling indiscriminately processes all frames, overwhelming key events with irrelevant content; and (2) heuristic retrieval captures superficial patterns but misses causal-temporal structures needed for complex reasoning. To address these challenges, we introduce LeAdQA, an innovative approach that bridges these gaps through synergizing causal-aware query refinement with fine-grained visual grounding. Our method first leverages LLMs to reformulate question-option pairs, resolving causal ambiguities and sharpening temporal focus. These refined queries subsequently direct a temporal grounding model to precisely retrieve the most salient segments, complemented by an adaptive fusion mechanism dynamically integrating the evidence to maximize relevance. The integrated visual-textual cues are then processed by an MLLM to generate accurate, contextually-grounded answers. Experiments on NExT-QA, IntentQA, and NExT-GQA demonstrate that our method's precise visual grounding substantially enhances the understanding of video-question relationships, achieving state-of-the-art (SOTA) performance on complex reasoning tasks while maintaining computational efficiency.
>
---
#### [new 105] Dense-depth map guided deep Lidar-Visual Odometry with Sparse Point Clouds and Images
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于定位与导航任务，旨在解决自主系统中姿态估计不准确的问题。通过融合激光雷达点云和图像，引入稠密深度图与注意力机制，优化光流估计与姿态预测，提升了定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.15496v1](http://arxiv.org/pdf/2507.15496v1)**

> **作者:** JunYing Huang; Ao Xu; DongSun Yong; KeRen Li; YuanFeng Wang; Qi Qin
>
> **摘要:** Odometry is a critical task for autonomous systems for self-localization and navigation. We propose a novel LiDAR-Visual odometry framework that integrates LiDAR point clouds and images for accurate and robust pose estimation. Our method utilizes a dense-depth map estimated from point clouds and images through depth completion, and incorporates a multi-scale feature extraction network with attention mechanisms, enabling adaptive depth-aware representations. Furthermore, we leverage dense depth information to refine flow estimation and mitigate errors in occlusion-prone regions. Our hierarchical pose refinement module optimizes motion estimation progressively, ensuring robust predictions against dynamic environments and scale ambiguities. Comprehensive experiments on the KITTI odometry benchmark demonstrate that our approach achieves similar or superior accuracy and robustness compared to state-of-the-art visual and LiDAR odometry methods.
>
---
#### [new 106] FOCUS: Fused Observation of Channels for Unveiling Spectra
- **分类: cs.CV; cs.AI**

- **简介: 论文提出FOCUS框架，解决高光谱成像（HSI）中视觉Transformer（ViTs）的可解释性问题。通过引入类别特定的光谱提示和可学习[SINK]标记，实现高效的空间-光谱解释，生成稳定且可解释的显著性图与光谱重要性曲线，提升带级交并比15%，减少注意力坍塌超40%。**

- **链接: [http://arxiv.org/pdf/2507.14787v1](http://arxiv.org/pdf/2507.14787v1)**

> **作者:** Xi Xiao; Aristeidis Tsaris; Anika Tabassum; John Lagergren; Larry M. York; Tianyang Wang; Xiao Wang
>
> **摘要:** Hyperspectral imaging (HSI) captures hundreds of narrow, contiguous wavelength bands, making it a powerful tool in biology, agriculture, and environmental monitoring. However, interpreting Vision Transformers (ViTs) in this setting remains largely unexplored due to two key challenges: (1) existing saliency methods struggle to capture meaningful spectral cues, often collapsing attention onto the class token, and (2) full-spectrum ViTs are computationally prohibitive for interpretability, given the high-dimensional nature of HSI data. We present FOCUS, the first framework that enables reliable and efficient spatial-spectral interpretability for frozen ViTs. FOCUS introduces two core components: class-specific spectral prompts that guide attention toward semantically meaningful wavelength groups, and a learnable [SINK] token trained with an attraction loss to absorb noisy or redundant attention. Together, these designs make it possible to generate stable and interpretable 3D saliency maps and spectral importance curves in a single forward pass, without any gradient backpropagation or backbone modification. FOCUS improves band-level IoU by 15 percent, reduces attention collapse by over 40 percent, and produces saliency results that align closely with expert annotations. With less than 1 percent parameter overhead, our method makes high-resolution ViT interpretability practical for real-world hyperspectral applications, bridging a long-standing gap between black-box modeling and trustworthy HSI decision-making.
>
---
#### [new 107] ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于半监督语义分割任务，旨在解决标注数据稀缺问题。论文提出ConformalSAM框架，利用基础分割模型SEEM生成未标注数据的预测掩码，并通过共形预测校准模型不确定性，筛选高置信度标签用于监督，结合自训练策略提升模型性能。实验表明该方法优于现有半监督方法。**

- **链接: [http://arxiv.org/pdf/2507.15803v1](http://arxiv.org/pdf/2507.15803v1)**

> **作者:** Danhui Chen; Ziquan Liu; Chuxi Yang; Dan Wang; Yan Yan; Yi Xu; Xiangyang Ji
>
> **备注:** ICCV 2025
>
> **摘要:** Pixel-level vision tasks, such as semantic segmentation, require extensive and high-quality annotated data, which is costly to obtain. Semi-supervised semantic segmentation (SSSS) has emerged as a solution to alleviate the labeling burden by leveraging both labeled and unlabeled data through self-training techniques. Meanwhile, the advent of foundational segmentation models pre-trained on massive data, has shown the potential to generalize across domains effectively. This work explores whether a foundational segmentation model can address label scarcity in the pixel-level vision task as an annotator for unlabeled images. Specifically, we investigate the efficacy of using SEEM, a Segment Anything Model (SAM) variant fine-tuned for textual input, to generate predictive masks for unlabeled data. To address the shortcomings of using SEEM-generated masks as supervision, we propose ConformalSAM, a novel SSSS framework which first calibrates the foundation model using the target domain's labeled data and then filters out unreliable pixel labels of unlabeled data so that only high-confidence labels are used as supervision. By leveraging conformal prediction (CP) to adapt foundation models to target data through uncertainty calibration, ConformalSAM exploits the strong capability of the foundational segmentation model reliably which benefits the early-stage learning, while a subsequent self-reliance training strategy mitigates overfitting to SEEM-generated masks in the later training stage. Our experiment demonstrates that, on three standard benchmarks of SSSS, ConformalSAM achieves superior performance compared to recent SSSS methods and helps boost the performance of those methods as a plug-in.
>
---
#### [new 108] In-context Learning of Vision Language Models for Detection of Physical and Digital Attacks against Face Recognition Systems
- **分类: cs.CV**

- **简介: 该论文属于生物识别安全任务，旨在检测人脸识别系统的物理和数字攻击。现有方法依赖大量训练数据且泛化能力不足。论文提出基于视觉语言模型（VLM）的上下文学习框架，无需大量训练数据即可有效检测多种攻击类型。实验表明其性能优于传统CNN方法，提升了攻击检测的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.15285v1](http://arxiv.org/pdf/2507.15285v1)**

> **作者:** Lazaro Janier Gonzalez-Soler; Maciej Salwowski; Christoph Busch
>
> **备注:** Submitted to IEEE-TIFS
>
> **摘要:** Recent advances in biometric systems have significantly improved the detection and prevention of fraudulent activities. However, as detection methods improve, attack techniques become increasingly sophisticated. Attacks on face recognition systems can be broadly divided into physical and digital approaches. Traditionally, deep learning models have been the primary defence against such attacks. While these models perform exceptionally well in scenarios for which they have been trained, they often struggle to adapt to different types of attacks or varying environmental conditions. These subsystems require substantial amounts of training data to achieve reliable performance, yet biometric data collection faces significant challenges, including privacy concerns and the logistical difficulties of capturing diverse attack scenarios under controlled conditions. This work investigates the application of Vision Language Models (VLM) and proposes an in-context learning framework for detecting physical presentation attacks and digital morphing attacks in biometric systems. Focusing on open-source models, the first systematic framework for the quantitative evaluation of VLMs in security-critical scenarios through in-context learning techniques is established. The experimental evaluation conducted on freely available databases demonstrates that the proposed subsystem achieves competitive performance for physical and digital attack detection, outperforming some of the traditional CNNs without resource-intensive training. The experimental results validate the proposed framework as a promising tool for improving generalisation in attack detection.
>
---
#### [new 109] DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting
- **分类: cs.CV; eess.IV; eess.SP**

- **简介: 本文属于3D重建任务，旨在解决稀疏视角下高斯点绘（3DGS）因高频过拟合导致的新视角生成质量差的问题。作者提出DWTGS方法，通过小波空间中的低频监督与高频稀疏约束进行正则化，提升泛化能力并减少高频伪影。**

- **链接: [http://arxiv.org/pdf/2507.15690v1](http://arxiv.org/pdf/2507.15690v1)**

> **作者:** Hung Nguyen; Runfa Li; An Le; Truong Nguyen
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Sparse-view 3D Gaussian Splatting (3DGS) presents significant challenges in reconstructing high-quality novel views, as it often overfits to the widely-varying high-frequency (HF) details of the sparse training views. While frequency regularization can be a promising approach, its typical reliance on Fourier transforms causes difficult parameter tuning and biases towards detrimental HF learning. We propose DWTGS, a framework that rethinks frequency regularization by leveraging wavelet-space losses that provide additional spatial supervision. Specifically, we supervise only the low-frequency (LF) LL subbands at multiple DWT levels, while enforcing sparsity on the HF HH subband in a self-supervised manner. Experiments across benchmarks show that DWTGS consistently outperforms Fourier-based counterparts, as this LF-centric strategy improves generalization and reduces HF hallucinations.
>
---
#### [new 110] PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，旨在解决真实世界图像在未见域中去雾效果差的问题。作者提出了PHATNet，通过物理引导的雾气迁移网络，将目标域的雾气模式迁移到源域无雾图像上，生成特定域的微调数据集，以提升模型的域适应能力。**

- **链接: [http://arxiv.org/pdf/2507.14826v1](http://arxiv.org/pdf/2507.14826v1)**

> **作者:** Fu-Jen Tsai; Yan-Tsung Peng; Yen-Yu Lin; Chia-Wen Lin
>
> **备注:** ICCV 2025
>
> **摘要:** Image dehazing aims to remove unwanted hazy artifacts in images. Although previous research has collected paired real-world hazy and haze-free images to improve dehazing models' performance in real-world scenarios, these models often experience significant performance drops when handling unseen real-world hazy images due to limited training data. This issue motivates us to develop a flexible domain adaptation method to enhance dehazing performance during testing. Observing that predicting haze patterns is generally easier than recovering clean content, we propose the Physics-guided Haze Transfer Network (PHATNet) which transfers haze patterns from unseen target domains to source-domain haze-free images, creating domain-specific fine-tuning sets to update dehazing models for effective domain adaptation. Additionally, we introduce a Haze-Transfer-Consistency loss and a Content-Leakage Loss to enhance PHATNet's disentanglement ability. Experimental results demonstrate that PHATNet significantly boosts state-of-the-art dehazing models on benchmark real-world image dehazing datasets.
>
---
#### [new 111] FinChart-Bench: Benchmarking Financial Chart Comprehension in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决金融图表理解的不足。作者构建了FinChart-Bench，包含1,200张真实金融图表及7,016个相关问题，用于评估25个先进模型。结果显示当前模型在指令跟随、空间推理等方面存在局限，尚无法可靠地自动评估金融图表。**

- **链接: [http://arxiv.org/pdf/2507.14823v1](http://arxiv.org/pdf/2507.14823v1)**

> **作者:** Dong Shu; Haoyang Yuan; Yuchen Wang; Yanguang Liu; Huopu Zhang; Haiyan Zhao; Mengnan Du
>
> **备注:** 20 Pages, 18 Figures
>
> **摘要:** Large vision-language models (LVLMs) have made significant progress in chart understanding. However, financial charts, characterized by complex temporal structures and domain-specific terminology, remain notably underexplored. We introduce FinChart-Bench, the first benchmark specifically focused on real-world financial charts. FinChart-Bench comprises 1,200 financial chart images collected from 2015 to 2024, each annotated with True/False (TF), Multiple Choice (MC), and Question Answering (QA) questions, totaling 7,016 questions. We conduct a comprehensive evaluation of 25 state-of-the-art LVLMs on FinChart-Bench. Our evaluation reveals critical insights: (1) the performance gap between open-source and closed-source models is narrowing, (2) performance degradation occurs in upgraded models within families, (3) many models struggle with instruction following, (4) both advanced models show significant limitations in spatial reasoning abilities, and (5) current LVLMs are not reliable enough to serve as automated evaluators. These findings highlight important limitations in current LVLM capabilities for financial chart understanding. The FinChart-Bench dataset is available at https://huggingface.co/datasets/Tizzzzy/FinChart-Bench.
>
---
#### [new 112] LEAD: Exploring Logit Space Evolution for Model Selection
- **分类: cs.CV**

- **简介: 该论文属于模型选择任务，旨在解决如何高效选择适合下游任务的预训练视觉模型。现有方法未能准确建模微调过程中的非线性优化动态。论文提出LEAD，基于logit空间建模，通过微分方程描述非线性演化过程，并设计类别感知分解方法，有效预测模型可迁移性，从而跳过耗时的微调步骤。**

- **链接: [http://arxiv.org/pdf/2507.14559v1](http://arxiv.org/pdf/2507.14559v1)**

> **作者:** Zixuan Hu; Xiaotong Li; Shixiang Tang; Jun Liu; Yichun Hu; Ling-Yu Duan
>
> **备注:** Accepted by CVPR 2024
>
> **摘要:** The remarkable success of pretrain-then-finetune paradigm has led to a proliferation of available pre-trained models for vision tasks. This surge presents a significant challenge in efficiently choosing the most suitable pre-trained models for downstream tasks. The critical aspect of this challenge lies in effectively predicting the model transferability by considering the underlying fine-tuning dynamics. Existing methods often model fine-tuning dynamics in feature space with linear transformations, which do not precisely align with the fine-tuning objective and fail to grasp the essential nonlinearity from optimization. To this end, we present LEAD, a finetuning-aligned approach based on the network output of logits. LEAD proposes a theoretical framework to model the optimization process and derives an ordinary differential equation (ODE) to depict the nonlinear evolution toward the final logit state. Additionally, we design a class-aware decomposition method to consider the varying evolution dynamics across classes and further ensure practical applicability. Integrating the closely aligned optimization objective and nonlinear modeling capabilities derived from the differential equation, our method offers a concise solution to effectively bridge the optimization gap in a single step, bypassing the lengthy fine-tuning process. The comprehensive experiments on 24 supervised and self-supervised pre-trained models across 10 downstream datasets demonstrate impressive performances and showcase its broad adaptability even in low-data scenarios.
>
---
#### [new 113] DiSCO-3D : Discovering and segmenting Sub-Concepts from Open-vocabulary queries in NeRF
- **分类: cs.CV**

- **简介: 该论文属于3D语义分割任务，旨在解决开放词汇查询下的场景自适应语义分割问题。提出了DiSCO-3D方法，结合神经场表示、无监督分割与弱开放词汇引导，实现对3D场景中子概念的发现与分割，兼顾场景内容与用户意图。**

- **链接: [http://arxiv.org/pdf/2507.14596v1](http://arxiv.org/pdf/2507.14596v1)**

> **作者:** Doriand Petit; Steve Bourgeois; Vincent Gay-Bellile; Florian Chabot; Loïc Barthe
>
> **备注:** Published at ICCV'25
>
> **摘要:** 3D semantic segmentation provides high-level scene understanding for applications in robotics, autonomous systems, \textit{etc}. Traditional methods adapt exclusively to either task-specific goals (open-vocabulary segmentation) or scene content (unsupervised semantic segmentation). We propose DiSCO-3D, the first method addressing the broader problem of 3D Open-Vocabulary Sub-concepts Discovery, which aims to provide a 3D semantic segmentation that adapts to both the scene and user queries. We build DiSCO-3D on Neural Fields representations, combining unsupervised segmentation with weak open-vocabulary guidance. Our evaluations demonstrate that DiSCO-3D achieves effective performance in Open-Vocabulary Sub-concepts Discovery and exhibits state-of-the-art results in the edge cases of both open-vocabulary and unsupervised segmentation.
>
---
#### [new 114] From Semantics, Scene to Instance-awareness: Distilling Foundation Model for Open-vocabulary Situation Recognition
- **分类: cs.CV**

- **简介: 该论文属于开放词汇的场景感知任务，旨在解决现有模型在复杂场景识别中泛化能力差、难以识别罕见和未见情境的问题。论文提出了一种多模态交互提示蒸馏框架（MIPD），通过从大模型中提取知识，提升小模型对未见情境的识别能力。**

- **链接: [http://arxiv.org/pdf/2507.14686v1](http://arxiv.org/pdf/2507.14686v1)**

> **作者:** Chen Cai; Tianyi Liu; Jianjun Gao; Wenyang Liu; Kejun Wu; Ruoyu Wang; Yi Wang; Soo Chin Liew
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) exhibit strong zero-shot abilities but struggle with complex Grounded Situation Recognition (GSR) and are resource-intensive for edge device deployment. Meanwhile, conventional GSR models often lack generalization ability, falling short in recognizing unseen and rare situations. In this paper, we exploit transferring knowledge from a teacher MLLM to a small GSR model to enhance its generalization and zero-shot abilities, thereby introducing the task of Open-vocabulary Grounded Situation Recognition (Ov-GSR). To achieve this, we propose Multimodal Interactive Prompt Distillation (MIPD), a novel framework that distills enriched multimodal knowledge from the foundation model, enabling the student Ov-GSR model to recognize unseen situations and be better aware of rare situations. Specifically, the MIPD framework first leverages the LLM-based Judgmental Rationales Generator (JRG) to construct positive and negative glimpse and gaze rationales enriched with contextual semantic information. The proposed scene-aware and instance-perception prompts are then introduced to align rationales with visual information from the MLLM teacher via the Negative-Guided Multimodal Prompting Alignment (NMPA) module, effectively capturing holistic and perceptual multimodal knowledge. Finally, the aligned multimodal knowledge is distilled into the student Ov-GSR model, providing a stronger foundation for generalization that enhances situation understanding, bridges the gap between seen and unseen scenarios, and mitigates prediction bias in rare cases. We evaluate MIPD on the refined Ov-SWiG dataset, achieving superior performance on seen, rare, and unseen situations, and further demonstrate improved unseen detection on the HICO-DET dataset.
>
---
#### [new 115] GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset
- **分类: cs.CV; I.4.6; I.2.10**

- **简介: 该论文属于遥感图像处理任务，旨在解决复杂梯田区域精细化农业地块提取缺乏高质量数据的问题。作者构建了首个全球精细标注的梯田地块数据集GTPBD，包含20多万个样本，支持语义分割、边缘检测、地块提取和无监督域适应等任务，并进行了多方法基准测试，填补了梯田遥感研究的关键空白。**

- **链接: [http://arxiv.org/pdf/2507.14697v1](http://arxiv.org/pdf/2507.14697v1)**

> **作者:** Zhiwei Zhang; Zi Ye; Yibin Wen; Shuai Yuan; Haohuan Fu; Jianxi Huang; Juepeng Zheng
>
> **备注:** 38 pages, 18 figures, submitted to NeurIPS 2025
>
> **摘要:** Agricultural parcels serve as basic units for conducting agricultural practices and applications, which is vital for land ownership registration, food security assessment, soil erosion monitoring, etc. However, existing agriculture parcel extraction studies only focus on mid-resolution mapping or regular plain farmlands while lacking representation of complex terraced terrains due to the demands of precision agriculture.In this paper, we introduce a more fine-grained terraced parcel dataset named GTPBD (Global Terraced Parcel and Boundary Dataset), which is the first fine-grained dataset covering major worldwide terraced regions with more than 200,000 complex terraced parcels with manual annotation. GTPBD comprises 47,537 high-resolution images with three-level labels, including pixel-level boundary labels, mask labels, and parcel labels. It covers seven major geographic zones in China and transcontinental climatic regions around the world.Compared to the existing datasets, the GTPBD dataset brings considerable challenges due to the: (1) terrain diversity; (2) complex and irregular parcel objects; and (3) multiple domain styles. Our proposed GTPBD dataset is suitable for four different tasks, including semantic segmentation, edge detection, terraced parcel extraction, and unsupervised domain adaptation (UDA) tasks.Accordingly, we benchmark the GTPBD dataset on eight semantic segmentation methods, four edge extraction methods, three parcel extraction methods, and five UDA methods, along with a multi-dimensional evaluation framework integrating pixel-level and object-level metrics. GTPBD fills a critical gap in terraced remote sensing research, providing a basic infrastructure for fine-grained agricultural terrain analysis and cross-scenario knowledge transfer.
>
---
#### [new 116] OmniVTON: Training-Free Universal Virtual Try-On
- **分类: cs.CV**

- **简介: 该论文属于图像虚拟试穿任务，旨在解决现有方法在跨域泛化和数据偏差上的限制。论文提出OmniVTON，首个无需训练的通用虚拟试穿框架，通过解耦服装和姿态条件，实现高保真纹理和姿态一致性。方法包括服装先验生成、边界缝合和DDIM反转技术，有效处理多条件约束，支持多人虚拟试穿。**

- **链接: [http://arxiv.org/pdf/2507.15037v1](http://arxiv.org/pdf/2507.15037v1)**

> **作者:** Zhaotong Yang; Yuhui Li; Shengfeng He; Xinzhe Li; Yangyang Xu; Junyu Dong; Yong Du
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Image-based Virtual Try-On (VTON) techniques rely on either supervised in-shop approaches, which ensure high fidelity but struggle with cross-domain generalization, or unsupervised in-the-wild methods, which improve adaptability but remain constrained by data biases and limited universality. A unified, training-free solution that works across both scenarios remains an open challenge. We propose OmniVTON, the first training-free universal VTON framework that decouples garment and pose conditioning to achieve both texture fidelity and pose consistency across diverse settings. To preserve garment details, we introduce a garment prior generation mechanism that aligns clothing with the body, followed by continuous boundary stitching technique to achieve fine-grained texture retention. For precise pose alignment, we utilize DDIM inversion to capture structural cues while suppressing texture interference, ensuring accurate body alignment independent of the original image textures. By disentangling garment and pose constraints, OmniVTON eliminates the bias inherent in diffusion models when handling multiple conditions simultaneously. Experimental results demonstrate that OmniVTON achieves superior performance across diverse datasets, garment types, and application scenarios. Notably, it is the first framework capable of multi-human VTON, enabling realistic garment transfer across multiple individuals in a single scene. Code is available at https://github.com/Jerome-Young/OmniVTON
>
---
#### [new 117] FastSmoothSAM: A Fast Smooth Method For Segment Anything Model
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决FastSAM模型分割边缘锯齿化、不精确的问题。论文提出了一种基于B-Spline曲线拟合的四阶段边缘优化方法，提升分割边缘的平滑度与准确性，同时保持FastSAM的实时处理能力，增强其在工业、医疗等场景的应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.15008v1](http://arxiv.org/pdf/2507.15008v1)**

> **作者:** Jiasheng Xu; Yewang Chen
>
> **摘要:** Accurately identifying and representing object edges is a challenging task in computer vision and image processing. The Segment Anything Model (SAM) has significantly influenced the field of image segmentation, but suffers from high memory consumption and long inference times, limiting its efficiency in real-time applications. To address these limitations, Fast Segment Anything (FastSAM) was proposed, achieving real-time segmentation. However, FastSAM often generates jagged edges that deviate from the true object shapes. Therefore, this paper introduces a novel refinement approach using B-Spline curve fitting techniques to enhance the edge quality in FastSAM. Leveraging the robust shape control and flexible geometric construction of B-Splines, a four-stage refining process involving two rounds of curve fitting is employed to effectively smooth jagged edges. This approach significantly improves the visual quality and analytical accuracy of object edges without compromising critical geometric information. The proposed method improves the practical utility of FastSAM by improving segmentation accuracy while maintaining real-time processing capabilities. This advancement unlocks greater potential for FastSAM technology in various real-world scenarios, such as industrial automation, medical imaging, and autonomous systems, where precise and efficient edge recognition is crucial.
>
---
#### [new 118] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出MeshMamba，基于Mamba状态空间模型生成与重建3D人体网格。任务为3D人体建模，解决高顶点数量下的生成与重建效率问题，实现全身（含手、脸）网格处理。方法包括顶点序列化策略及两个应用模型：MambaDiff3D用于生成，Mamba-HMR用于图像到网格的重建。**

- **链接: [http://arxiv.org/pdf/2507.15212v1](http://arxiv.org/pdf/2507.15212v1)**

> **作者:** Yusuke Yoshiyasu; Leyuan Sun; Ryusuke Sagawa
>
> **备注:** Accepted at ICCV2025
>
> **摘要:** In this paper, we introduce MeshMamba, a neural network model for learning 3D articulated mesh models by employing the recently proposed Mamba State Space Models (Mamba-SSMs). MeshMamba is efficient and scalable in handling a large number of input tokens, enabling the generation and reconstruction of body mesh models with more than 10,000 vertices, capturing clothing and hand geometries. The key to effectively learning MeshMamba is the serialization technique of mesh vertices into orderings that are easily processed by Mamba. This is achieved by sorting the vertices based on body part annotations or the 3D vertex locations of a template mesh, such that the ordering respects the structure of articulated shapes. Based on MeshMamba, we design 1) MambaDiff3D, a denoising diffusion model for generating 3D articulated meshes and 2) Mamba-HMR, a 3D human mesh recovery model that reconstructs a human body shape and pose from a single image. Experimental results showed that MambaDiff3D can generate dense 3D human meshes in clothes, with grasping hands, etc., and outperforms previous approaches in the 3D human shape generation task. Additionally, Mamba-HMR extends the capabilities of previous non-parametric human mesh recovery approaches, which were limited to handling body-only poses using around 500 vertex tokens, to the whole-body setting with face and hands, while achieving competitive performance in (near) real-time.
>
---
#### [new 119] CRAFT: A Neuro-Symbolic Framework for Visual Functional Affordance Grounding
- **分类: cs.CV**

- **简介: 该论文提出CRAFT，一个神经符号框架，用于可解释的视觉功能可操作性推理。任务是识别场景中支持特定动作的物体。通过结合ConceptNet、语言模型与CLIP视觉特征，使用能量推理循环提升预测，实现透明、可解释的场景理解。**

- **链接: [http://arxiv.org/pdf/2507.14426v1](http://arxiv.org/pdf/2507.14426v1)**

> **作者:** Zhou Chen; Joe Lin; Sathyanarayanan N. Aakur
>
> **备注:** Accepted to NeSy 2025
>
> **摘要:** We introduce CRAFT, a neuro-symbolic framework for interpretable affordance grounding, which identifies the objects in a scene that enable a given action (e.g., "cut"). CRAFT integrates structured commonsense priors from ConceptNet and language models with visual evidence from CLIP, using an energy-based reasoning loop to refine predictions iteratively. This process yields transparent, goal-driven decisions to ground symbolic and perceptual structures. Experiments in multi-object, label-free settings demonstrate that CRAFT enhances accuracy while improving interpretability, providing a step toward robust and trustworthy scene understanding.
>
---
#### [new 120] Seeing Through Deepfakes: A Human-Inspired Framework for Multi-Face Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多面孔深度伪造检测任务，旨在解决多脸场景下深度伪造视频检测难题。通过分析人类识别伪造面孔的线索，提出HICOM框架，结合场景运动一致性、面孔外观兼容性等关键特征，提升检测准确率，并引入大语言模型增强解释性。实验表明其在多数据集上表现优异，具有较强泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.14807v1](http://arxiv.org/pdf/2507.14807v1)**

> **作者:** Juan Hu; Shaojing Fan; Terence Sim
>
> **摘要:** Multi-face deepfake videos are becoming increasingly prevalent, often appearing in natural social settings that challenge existing detection methods. Most current approaches excel at single-face detection but struggle in multi-face scenarios, due to a lack of awareness of crucial contextual cues. In this work, we develop a novel approach that leverages human cognition to analyze and defend against multi-face deepfake videos. Through a series of human studies, we systematically examine how people detect deepfake faces in social settings. Our quantitative analysis reveals four key cues humans rely on: scene-motion coherence, inter-face appearance compatibility, interpersonal gaze alignment, and face-body consistency. Guided by these insights, we introduce \textsf{HICOM}, a novel framework designed to detect every fake face in multi-face scenarios. Extensive experiments on benchmark datasets show that \textsf{HICOM} improves average accuracy by 3.3\% in in-dataset detection and 2.8\% under real-world perturbations. Moreover, it outperforms existing methods by 5.8\% on unseen datasets, demonstrating the generalization of human-inspired cues. \textsf{HICOM} further enhances interpretability by incorporating an LLM to provide human-readable explanations, making detection results more transparent and convincing. Our work sheds light on involving human factors to enhance defense against deepfakes.
>
---
#### [new 121] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决大规模真实数据收集与标注成本高的问题。通过使用小规模高保真合成数据训练高效视觉模型，在保持准确性的同时显著降低训练与推理成本，并提升数据多样性与公平性控制。**

- **链接: [http://arxiv.org/pdf/2507.15365v1](http://arxiv.org/pdf/2507.15365v1)**

> **作者:** Fatemeh Saleh; Sadegh Aliakbarian; Charlie Hewitt; Lohit Petikam; Xiao-Xian; Antonio Criminisi; Thomas J. Cashman; Tadas Baltrušaitis
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** The state of the art in human-centric computer vision achieves high accuracy and robustness across a diverse range of tasks. The most effective models in this domain have billions of parameters, thus requiring extremely large datasets, expensive training regimes, and compute-intensive inference. In this paper, we demonstrate that it is possible to train models on much smaller but high-fidelity synthetic datasets, with no loss in accuracy and higher efficiency. Using synthetic training data provides us with excellent levels of detail and perfect labels, while providing strong guarantees for data provenance, usage rights, and user consent. Procedural data synthesis also provides us with explicit control on data diversity, that we can use to address unfairness in the models we train. Extensive quantitative assessment on real input images demonstrates accuracy of our models on three dense prediction tasks: depth estimation, surface normal estimation, and soft foreground segmentation. Our models require only a fraction of the cost of training and inference when compared with foundational models of similar accuracy. Our human-centric synthetic dataset and trained models are available at https://aka.ms/DAViD.
>
---
#### [new 122] Improving Joint Embedding Predictive Architecture with Diffusion Noise
- **分类: cs.CV**

- **简介: 该论文属于自监督学习与生成模型结合的任务，旨在提升表征学习能力。作者提出N-JEPA方法，将扩散噪声引入掩码图像建模，通过位置嵌入和多级噪声调度增强模型鲁棒性，改善下游任务的分类性能。**

- **链接: [http://arxiv.org/pdf/2507.15216v1](http://arxiv.org/pdf/2507.15216v1)**

> **作者:** Yuping Qiu; Rui Zhu; Ying-cong Chen
>
> **摘要:** Self-supervised learning has become an incredibly successful method for feature learning, widely applied to many downstream tasks. It has proven especially effective for discriminative tasks, surpassing the trending generative models. However, generative models perform better in image generation and detail enhancement. Thus, it is natural for us to find a connection between SSL and generative models to further enhance the representation capacity of SSL. As generative models can create new samples by approximating the data distribution, such modeling should also lead to a semantic understanding of the raw visual data, which is necessary for recognition tasks. This enlightens us to combine the core principle of the diffusion model: diffusion noise, with SSL to learn a competitive recognition model. Specifically, diffusion noise can be viewed as a particular state of mask that reveals a close relationship between masked image modeling (MIM) and diffusion models. In this paper, we propose N-JEPA (Noise-based JEPA) to incorporate diffusion noise into MIM by the position embedding of masked tokens. The multi-level noise schedule is a series of feature augmentations to further enhance the robustness of our model. We perform a comprehensive study to confirm its effectiveness in the classification of downstream tasks. Codes will be released soon in public.
>
---
#### [new 123] An aerial color image anomaly dataset for search missions in complex forested terrain
- **分类: cs.CV**

- **简介: 该论文属于图像处理与计算机视觉任务，旨在解决复杂森林环境中难以检测的异常目标问题。作者构建了一个包含隐蔽异常的高分辨率航拍图像数据集，并提供在线交互平台以支持标注与扩展，用于改进搜索与营救任务中的异常检测方法。**

- **链接: [http://arxiv.org/pdf/2507.15492v1](http://arxiv.org/pdf/2507.15492v1)**

> **作者:** Rakesh John Amala Arokia Nathan; Matthias Gessner; Nurullah Özkan; Marius Bock; Mohamed Youssef; Maximilian Mews; Björn Piltz; Ralf Berger; Oliver Bimber
>
> **备注:** 17 pages
>
> **摘要:** After a family murder in rural Germany, authorities failed to locate the suspect in a vast forest despite a massive search. To aid the search, a research aircraft captured high-resolution aerial imagery. Due to dense vegetation obscuring small clues, automated analysis was ineffective, prompting a crowd-search initiative. This effort produced a unique dataset of labeled, hard-to-detect anomalies under occluded, real-world conditions. It can serve as a benchmark for improving anomaly detection approaches in complex forest environments, supporting manhunts and rescue operations. Initial benchmark tests showed existing methods performed poorly, highlighting the need for context-aware approaches. The dataset is openly accessible for offline processing. An additional interactive web interface supports online viewing and dynamic growth by allowing users to annotate and submit new findings.
>
---
#### [new 124] SurgX: Neuron-Concept Association for Explainable Surgical Phase Recognition
- **分类: cs.CV**

- **简介: 该论文属于手术阶段识别任务，旨在提升模型的可解释性。当前深度学习模型在手术阶段识别中表现良好，但缺乏透明度，难以理解其决策过程。为此，作者提出了SurgX框架，通过将神经元与手术相关概念关联，增强模型的可解释性。论文工作包括选择代表性示例、构建概念集、神经元-概念关联及识别关键神经元，并在两个手术数据集上进行了验证。**

- **链接: [http://arxiv.org/pdf/2507.15418v1](http://arxiv.org/pdf/2507.15418v1)**

> **作者:** Ka Young Kim; Hyeon Bae Kim; Seong Tae Kim
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Surgical phase recognition plays a crucial role in surgical workflow analysis, enabling various applications such as surgical monitoring, skill assessment, and workflow optimization. Despite significant advancements in deep learning-based surgical phase recognition, these models remain inherently opaque, making it difficult to understand how they make decisions. This lack of interpretability hinders trust and makes it challenging to debug the model. To address this challenge, we propose SurgX, a novel concept-based explanation framework that enhances the interpretability of surgical phase recognition models by associating neurons with relevant concepts. In this paper, we introduce the process of selecting representative example sequences for neurons, constructing a concept set tailored to the surgical video dataset, associating neurons with concepts and identifying neurons crucial for predictions. Through extensive experiments on two surgical phase recognition models, we validate our method and analyze the explanation for prediction. This highlights the potential of our method in explaining surgical phase recognition. The code is available at https://github.com/ailab-kyunghee/SurgX
>
---
#### [new 125] Can Your Model Separate Yolks with a Water Bottle? Benchmarking Physical Commonsense Understanding in Video Generation Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决当前文本到视频生成模型在物理常识理解上的不足。作者构建了PhysVidBench基准，包含383个强调工具使用、材料属性和物理交互的提示，并提出三阶段评估流程，通过物理问题生成、视频描述和问答推理，系统评估模型的物理常识理解能力。**

- **链接: [http://arxiv.org/pdf/2507.15824v1](http://arxiv.org/pdf/2507.15824v1)**

> **作者:** Enes Sanli; Baris Sarper Tezcan; Aykut Erdem; Erkut Erdem
>
> **摘要:** Recent progress in text-to-video (T2V) generation has enabled the synthesis of visually compelling and temporally coherent videos from natural language. However, these models often fall short in basic physical commonsense, producing outputs that violate intuitive expectations around causality, object behavior, and tool use. Addressing this gap, we present PhysVidBench, a benchmark designed to evaluate the physical reasoning capabilities of T2V systems. The benchmark includes 383 carefully curated prompts, emphasizing tool use, material properties, and procedural interactions, and domains where physical plausibility is crucial. For each prompt, we generate videos using diverse state-of-the-art models and adopt a three-stage evaluation pipeline: (1) formulate grounded physics questions from the prompt, (2) caption the generated video with a vision-language model, and (3) task a language model to answer several physics-involved questions using only the caption. This indirect strategy circumvents common hallucination issues in direct video-based evaluation. By highlighting affordances and tool-mediated actions, areas overlooked in current T2V evaluations, PhysVidBench provides a structured, interpretable framework for assessing physical commonsense in generative video models.
>
---
#### [new 126] WSI-Agents: A Collaborative Multi-Agent System for Multi-Modal Whole Slide Image Analysis
- **分类: cs.CV; cs.AI; 68T07, 92C55; I.2.7; I.4.8; J.3**

- **简介: 该论文属于医学图像分析任务，旨在解决多模态全切片图像（WSI）分析中任务专用模型与多任务模型之间的性能差距问题。论文提出了WSI-Agents，一种结合专业功能代理、任务分配和验证机制的协作多代理系统，以提升分析的准确性与多任务处理能力。**

- **链接: [http://arxiv.org/pdf/2507.14680v1](http://arxiv.org/pdf/2507.14680v1)**

> **作者:** Xinheng Lyu; Yuci Liang; Wenting Chen; Meidan Ding; Jiaqi Yang; Guolin Huang; Daokun Zhang; Xiangjian He; Linlin Shen
>
> **摘要:** Whole slide images (WSIs) are vital in digital pathology, enabling gigapixel tissue analysis across various pathological tasks. While recent advancements in multi-modal large language models (MLLMs) allow multi-task WSI analysis through natural language, they often underperform compared to task-specific models. Collaborative multi-agent systems have emerged as a promising solution to balance versatility and accuracy in healthcare, yet their potential remains underexplored in pathology-specific domains. To address these issues, we propose WSI-Agents, a novel collaborative multi-agent system for multi-modal WSI analysis. WSI-Agents integrates specialized functional agents with robust task allocation and verification mechanisms to enhance both task-specific accuracy and multi-task versatility through three components: (1) a task allocation module assigning tasks to expert agents using a model zoo of patch and WSI level MLLMs, (2) a verification mechanism ensuring accuracy through internal consistency checks and external validation using pathology knowledge bases and domain-specific models, and (3) a summary module synthesizing the final summary with visual interpretation maps. Extensive experiments on multi-modal WSI benchmarks show WSI-Agents's superiority to current WSI MLLMs and medical agent frameworks across diverse tasks.
>
---
#### [new 127] DFQ-ViT: Data-Free Quantization for Vision Transformers without Fine-tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决视觉Transformer在无数据情况下的量化问题。现有方法因合成数据质量差和量化模型激活分布差异导致性能下降。论文提出DFQ-ViT，通过逐步合成高质量样本和引入激活校正矩阵来对齐分布，无需微调即可提升量化模型性能，达到与使用真实数据量化相当的效果。**

- **链接: [http://arxiv.org/pdf/2507.14481v1](http://arxiv.org/pdf/2507.14481v1)**

> **作者:** Yujia Tong; Jingling Yuan; Tian Zhang; Jianquan Liu; Chuang Hu
>
> **摘要:** Data-Free Quantization (DFQ) enables the quantization of Vision Transformers (ViTs) without requiring access to data, allowing for the deployment of ViTs on devices with limited resources. In DFQ, the quantization model must be calibrated using synthetic samples, making the quality of these synthetic samples crucial. Existing methods fail to fully capture and balance the global and local features within the samples, resulting in limited synthetic data quality. Moreover, we have found that during inference, there is a significant difference in the distributions of intermediate layer activations between the quantized and full-precision models. These issues lead to a severe performance degradation of the quantized model. To address these problems, we propose a pipeline for Data-Free Quantization for Vision Transformers (DFQ-ViT). Specifically, we synthesize samples in order of increasing difficulty, effectively enhancing the quality of synthetic data. During the calibration and inference stage, we introduce the activation correction matrix for the quantized model to align the intermediate layer activations with those of the full-precision model. Extensive experiments demonstrate that DFQ-ViT achieves remarkable superiority over existing DFQ methods and its performance is on par with models quantized through real data. For example, the performance of DeiT-T with 3-bit weights quantization is 4.29% higher than the state-of-the-art. Our method eliminates the need for fine-tuning, which not only reduces computational overhead but also lowers the deployment barriers for edge devices. This characteristic aligns with the principles of Green Learning by improving energy efficiency and facilitating real-world applications in resource-constrained environments.
>
---
#### [new 128] Docopilot: Improving Multimodal Models for Document-Level Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态文档理解任务，旨在解决现有模型在复杂多页文档理解上的不足。作者构建了一个高质量文档级数据集Doc-750K，并基于此开发了多模态模型Docopilot，有效处理文档级依赖，提升了理解效果。**

- **链接: [http://arxiv.org/pdf/2507.14675v1](http://arxiv.org/pdf/2507.14675v1)**

> **作者:** Yuchen Duan; Zhe Chen; Yusong Hu; Weiyun Wang; Shenglong Ye; Botian Shi; Lewei Lu; Qibin Hou; Tong Lu; Hongsheng Li; Jifeng Dai; Wenhai Wang
>
> **摘要:** Despite significant progress in multimodal large language models (MLLMs), their performance on complex, multi-page document comprehension remains inadequate, largely due to the lack of high-quality, document-level datasets. While current retrieval-augmented generation (RAG) methods offer partial solutions, they suffer from issues, such as fragmented retrieval contexts, multi-stage error accumulation, and extra time costs of retrieval. In this work, we present a high-quality document-level dataset, Doc-750K, designed to support in-depth understanding of multimodal documents. This dataset includes diverse document structures, extensive cross-page dependencies, and real question-answer pairs derived from the original documents. Building on the dataset, we develop a native multimodal model, Docopilot, which can accurately handle document-level dependencies without relying on RAG. Experiments demonstrate that Docopilot achieves superior coherence, accuracy, and efficiency in document understanding tasks and multi-turn interactions, setting a new baseline for document-level multimodal understanding. Data, code, and models are released at https://github.com/OpenGVLab/Docopilot
>
---
#### [new 129] Experimenting active and sequential learning in a medieval music manuscript
- **分类: cs.CV; I.2.10; I.4.8; H.3.3**

- **简介: 该论文属于光学音乐识别（OMR）任务，旨在解决历史音乐手稿数字化中标注数据稀缺的问题。作者基于YOLOv8模型，采用主动学习与顺序学习方法，通过选择高不确定性样例进行迭代标注与训练，以提升识别性能。实验表明，在少量标注数据下可实现接近全监督训练的准确率。**

- **链接: [http://arxiv.org/pdf/2507.15633v1](http://arxiv.org/pdf/2507.15633v1)**

> **作者:** Sachin Sharma; Federico Simonetta; Michele Flammini
>
> **备注:** 6 pages, 4 figures, accepted at IEEE MLSP 2025 (IEEE International Workshop on Machine Learning for Signal Processing). Special Session: Applications of AI in Cultural and Artistic Heritage
>
> **摘要:** Optical Music Recognition (OMR) is a cornerstone of music digitization initiatives in cultural heritage, yet it remains limited by the scarcity of annotated data and the complexity of historical manuscripts. In this paper, we present a preliminary study of Active Learning (AL) and Sequential Learning (SL) tailored for object detection and layout recognition in an old medieval music manuscript. Leveraging YOLOv8, our system selects samples with the highest uncertainty (lowest prediction confidence) for iterative labeling and retraining. Our approach starts with a single annotated image and successfully boosts performance while minimizing manual labeling. Experimental results indicate that comparable accuracy to fully supervised training can be achieved with significantly fewer labeled examples. We test the methodology as a preliminary investigation on a novel dataset offered to the community by the Anonymous project, which studies laude, a poetical-musical genre spread across Italy during the 12th-16th Century. We show that in the manuscript at-hand, uncertainty-based AL is not effective and advocates for more usable methods in data-scarcity scenarios.
>
---
#### [new 130] Benefit from Reference: Retrieval-Augmented Cross-modal Point Cloud Completion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D点云补全任务，旨在解决不完整点云缺乏结构特征导致的补全困难问题。通过引入跨模态检索，结合参考样本的结构先验信息，设计结构共享特征编码器和渐进式检索增强生成器，提升补全效果与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.14485v1](http://arxiv.org/pdf/2507.14485v1)**

> **作者:** Hongye Hou; Liu Zhan; Yang Yang
>
> **摘要:** Completing the whole 3D structure based on an incomplete point cloud is a challenging task, particularly when the residual point cloud lacks typical structural characteristics. Recent methods based on cross-modal learning attempt to introduce instance images to aid the structure feature learning. However, they still focus on each particular input class, limiting their generation abilities. In this work, we propose a novel retrieval-augmented point cloud completion framework. The core idea is to incorporate cross-modal retrieval into completion task to learn structural prior information from similar reference samples. Specifically, we design a Structural Shared Feature Encoder (SSFE) to jointly extract cross-modal features and reconstruct reference features as priors. Benefiting from a dual-channel control gate in the encoder, relevant structural features in the reference sample are enhanced and irrelevant information interference is suppressed. In addition, we propose a Progressive Retrieval-Augmented Generator (PRAG) that employs a hierarchical feature fusion mechanism to integrate reference prior information with input features from global to local. Through extensive evaluations on multiple datasets and real-world scenes, our method shows its effectiveness in generating fine-grained point clouds, as well as its generalization capability in handling sparse data and unseen categories.
>
---
#### [new 131] SeC: Advancing Complex Video Object Segmentation via Progressive Concept Construction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频目标分割（VOS）任务，旨在解决现有方法在复杂场景中因依赖外观匹配而表现不佳的问题。作者提出SeC框架，通过逐步构建高层次、以对象为中心的概念表示，结合视觉语言模型提升语义理解，实现更鲁棒的视频目标分割。论文还引入了新基准SeCVOS用于评估复杂场景下的模型表现。**

- **链接: [http://arxiv.org/pdf/2507.15852v1](http://arxiv.org/pdf/2507.15852v1)**

> **作者:** Zhixiong Zhang; Shuangrui Ding; Xiaoyi Dong; Songxin He; Jianfan Lin; Junsong Tang; Yuhang Zang; Yuhang Cao; Dahua Lin; Jiaqi Wang
>
> **备注:** project page: https://rookiexiong7.github.io/projects/SeC/; code: https://github.com/OpenIXCLab/SeC; dataset: https://huggingface.co/datasets/OpenIXCLab/SeCVOS
>
> **摘要:** Video Object Segmentation (VOS) is a core task in computer vision, requiring models to track and segment target objects across video frames. Despite notable advances with recent efforts, current techniques still lag behind human capabilities in handling drastic visual variations, occlusions, and complex scene changes. This limitation arises from their reliance on appearance matching, neglecting the human-like conceptual understanding of objects that enables robust identification across temporal dynamics. Motivated by this gap, we propose Segment Concept (SeC), a concept-driven segmentation framework that shifts from conventional feature matching to the progressive construction and utilization of high-level, object-centric representations. SeC employs Large Vision-Language Models (LVLMs) to integrate visual cues across diverse frames, constructing robust conceptual priors. During inference, SeC forms a comprehensive semantic representation of the target based on processed frames, realizing robust segmentation of follow-up frames. Furthermore, SeC adaptively balances LVLM-based semantic reasoning with enhanced feature matching, dynamically adjusting computational efforts based on scene complexity. To rigorously assess VOS methods in scenarios demanding high-level conceptual reasoning and robust semantic understanding, we introduce the Semantic Complex Scenarios Video Object Segmentation benchmark (SeCVOS). SeCVOS comprises 160 manually annotated multi-scenario videos designed to challenge models with substantial appearance variations and dynamic scene transformations. In particular, SeC achieves an 11.8-point improvement over SAM 2.1 on SeCVOS, establishing a new state-of-the-art in concept-aware video object segmentation.
>
---
#### [new 132] Adaptive 3D Gaussian Splatting Video Streaming
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于3D高斯点阵视频流任务，旨在解决其数据量大、压缩传输复杂的问题。工作包括设计基于高斯变形场的视频构建方法，采用混合显著性分块和差异化质量建模，实现高效压缩与带宽适应，提升传输性能。**

- **链接: [http://arxiv.org/pdf/2507.14432v1](http://arxiv.org/pdf/2507.14432v1)**

> **作者:** Han Gong; Qiyue Li; Zhi Liu; Hao Zhou; Peng Yuan Zhou; Zhu Li; Jie Li
>
> **摘要:** The advent of 3D Gaussian splatting (3DGS) has significantly enhanced the quality of volumetric video representation. Meanwhile, in contrast to conventional volumetric video, 3DGS video poses significant challenges for streaming due to its substantially larger data volume and the heightened complexity involved in compression and transmission. To address these issues, we introduce an innovative framework for 3DGS volumetric video streaming. Specifically, we design a 3DGS video construction method based on the Gaussian deformation field. By employing hybrid saliency tiling and differentiated quality modeling of 3DGS video, we achieve efficient data compression and adaptation to bandwidth fluctuations while ensuring high transmission quality. Then we build a complete 3DGS video streaming system and validate the transmission performance. Through experimental evaluation, our method demonstrated superiority over existing approaches in various aspects, including video quality, compression effectiveness, and transmission rate.
>
---
#### [new 133] Few-Shot Object Detection via Spatial-Channel State Space Model
- **分类: cs.CV**

- **简介: 该论文属于小样本目标检测（FSOD）任务，旨在解决因训练样本有限导致的特征提取不准确问题。论文提出了一种基于Mamba的“空间-通道状态空间建模”（SCSM）模块，通过建模通道间相关性，提升特征表示质量，从而改善检测性能。**

- **链接: [http://arxiv.org/pdf/2507.15308v1](http://arxiv.org/pdf/2507.15308v1)**

> **作者:** Zhimeng Xin; Tianxu Wu; Yixiong Zou; Shiming Chen; Dingjie Fu; Xinge You
>
> **摘要:** Due to the limited training samples in few-shot object detection (FSOD), we observe that current methods may struggle to accurately extract effective features from each channel. Specifically, this issue manifests in two aspects: i) channels with high weights may not necessarily be effective, and ii) channels with low weights may still hold significant value. To handle this problem, we consider utilizing the inter-channel correlation to facilitate the novel model's adaptation process to novel conditions, ensuring the model can correctly highlight effective channels and rectify those incorrect ones. Since the channel sequence is also 1-dimensional, its similarity with the temporal sequence inspires us to take Mamba for modeling the correlation in the channel sequence. Based on this concept, we propose a Spatial-Channel State Space Modeling (SCSM) module for spatial-channel state modeling, which highlights the effective patterns and rectifies those ineffective ones in feature channels. In SCSM, we design the Spatial Feature Modeling (SFM) module to balance the learning of spatial relationships and channel relationships, and then introduce the Channel State Modeling (CSM) module based on Mamba to learn correlation in channels. Extensive experiments on the VOC and COCO datasets show that the SCSM module enables the novel detector to improve the quality of focused feature representation in channels and achieve state-of-the-art performance.
>
---
#### [new 134] Learning from Heterogeneity: Generalizing Dynamic Facial Expression Recognition via Distributionally Robust Optimization
- **分类: cs.CV**

- **简介: 该论文属于动态面部表情识别（DFER）任务，旨在解决多源数据和个体差异导致的性能下降问题。作者提出了HDF框架，包含时间-频率分布注意力模块（DAM）和分布感知优化模块（DSM），以增强时频建模并平衡损失优化，从而提升识别准确率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.15765v1](http://arxiv.org/pdf/2507.15765v1)**

> **作者:** Feng-Qi Cui; Anyang Tong; Jinyang Huang; Jie Zhang; Dan Guo; Zhi Liu; Meng Wang
>
> **备注:** Accepted by ACM MM'25
>
> **摘要:** Dynamic Facial Expression Recognition (DFER) plays a critical role in affective computing and human-computer interaction. Although existing methods achieve comparable performance, they inevitably suffer from performance degradation under sample heterogeneity caused by multi-source data and individual expression variability. To address these challenges, we propose a novel framework, called Heterogeneity-aware Distributional Framework (HDF), and design two plug-and-play modules to enhance time-frequency modeling and mitigate optimization imbalance caused by hard samples. Specifically, the Time-Frequency Distributional Attention Module (DAM) captures both temporal consistency and frequency robustness through a dual-branch attention design, improving tolerance to sequence inconsistency and visual style shifts. Then, based on gradient sensitivity and information bottleneck principles, an adaptive optimization module Distribution-aware Scaling Module (DSM) is introduced to dynamically balance classification and contrastive losses, enabling more stable and discriminative representation learning. Extensive experiments on two widely used datasets, DFEW and FERV39k, demonstrate that HDF significantly improves both recognition accuracy and robustness. Our method achieves superior weighted average recall (WAR) and unweighted average recall (UAR) while maintaining strong generalization across diverse and imbalanced scenarios. Codes are released at https://github.com/QIcita/HDF_DFER.
>
---
#### [new 135] Grounding Degradations in Natural Language for All-In-One Video Restoration
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于视频恢复任务，旨在解决多种自然语言描述退化类型的视频恢复问题。作者提出了一种一体化的框架，利用基础模型将视频帧的退化感知语义上下文与自然语言结合，实现了无需退化知识的训练和推理。此外，作者提出了新的多退化基准测试，推动了该领域的标准化评估。**

- **链接: [http://arxiv.org/pdf/2507.14851v1](http://arxiv.org/pdf/2507.14851v1)**

> **作者:** Muhammad Kamran Janjua; Amirhosein Ghasemabadi; Kunlin Zhang; Mohammad Salameh; Chao Gao; Di Niu
>
> **备注:** 17 pages
>
> **摘要:** In this work, we propose an all-in-one video restoration framework that grounds degradation-aware semantic context of video frames in natural language via foundation models, offering interpretable and flexible guidance. Unlike prior art, our method assumes no degradation knowledge in train or test time and learns an approximation to the grounded knowledge such that the foundation model can be safely disentangled during inference adding no extra cost. Further, we call for standardization of benchmarks in all-in-one video restoration, and propose two benchmarks in multi-degradation setting, three-task (3D) and four-task (4D), and two time-varying composite degradation benchmarks; one of the latter being our proposed dataset with varying snow intensity, simulating how weather degradations affect videos naturally. We compare our method with prior works and report state-of-the-art performance on all benchmarks.
>
---
#### [new 136] An Evaluation of DUSt3R/MASt3R/VGGT 3D Reconstruction on Photogrammetric Aerial Blocks
- **分类: cs.CV**

- **简介: 论文评估了DUSt3R、MASt3R和VGGT三种3D重建模型在航拍图像块上的表现，旨在解决低重叠、无序图像的密集3D重建问题。研究验证了这些模型在稀疏图像集上的有效性，对比传统方法COLMAP，在点云完整性和姿态估计方面有显著提升，尤其适用于低分辨率和稀疏场景。**

- **链接: [http://arxiv.org/pdf/2507.14798v1](http://arxiv.org/pdf/2507.14798v1)**

> **作者:** Xinyi Wu; Steven Landgraf; Markus Ulrich; Rongjun Qin
>
> **备注:** 23 pages, 6 figures, this manuscript has been submitted to Geo-spatial Information Science for consideration
>
> **摘要:** State-of-the-art 3D computer vision algorithms continue to advance in handling sparse, unordered image sets. Recently developed foundational models for 3D reconstruction, such as Dense and Unconstrained Stereo 3D Reconstruction (DUSt3R), Matching and Stereo 3D Reconstruction (MASt3R), and Visual Geometry Grounded Transformer (VGGT), have attracted attention due to their ability to handle very sparse image overlaps. Evaluating DUSt3R/MASt3R/VGGT on typical aerial images matters, as these models may handle extremely low image overlaps, stereo occlusions, and textureless regions. For redundant collections, they can accelerate 3D reconstruction by using extremely sparsified image sets. Despite tests on various computer vision benchmarks, their potential on photogrammetric aerial blocks remains unexplored. This paper conducts a comprehensive evaluation of the pre-trained DUSt3R/MASt3R/VGGT models on the aerial blocks of the UseGeo dataset for pose estimation and dense 3D reconstruction. Results show these methods can accurately reconstruct dense point clouds from very sparse image sets (fewer than 10 images, up to 518 pixels resolution), with completeness gains up to +50% over COLMAP. VGGT also demonstrates higher computational efficiency, scalability, and more reliable camera pose estimation. However, all exhibit limitations with high-resolution images and large sets, as pose reliability declines with more images and geometric complexity. These findings suggest transformer-based methods cannot fully replace traditional SfM and MVS, but offer promise as complementary approaches, especially in challenging, low-resolution, and sparse scenarios.
>
---
#### [new 137] Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering Human Perceptual Variability on Facial Expressions
- **分类: cs.CV; cs.CY**

- **简介: 该论文属于情感认知科学任务，旨在解决个体间情绪感知差异建模问题。论文提出通过ANN决策边界生成模糊面部表情图像，构建诱发人类感知不确定性的数据集varEmotion，并通过行为实验验证ANN与人类感知的一致性，最终实现ANN预测与群体及个体情绪感知模式的对齐。**

- **链接: [http://arxiv.org/pdf/2507.14549v1](http://arxiv.org/pdf/2507.14549v1)**

> **作者:** Haotian Deng; Chi Zhang; Chen Wei; Quanying Liu
>
> **备注:** Accepted by IJCNN 2025
>
> **摘要:** A fundamental challenge in affective cognitive science is to develop models that accurately capture the relationship between external emotional stimuli and human internal experiences. While ANNs have demonstrated remarkable accuracy in facial expression recognition, their ability to model inter-individual differences in human perception remains underexplored. This study investigates the phenomenon of high perceptual variability-where individuals exhibit significant differences in emotion categorization even when viewing the same stimulus. Inspired by the similarity between ANNs and human perception, we hypothesize that facial expression samples that are ambiguous for ANN classifiers also elicit divergent perceptual judgments among human observers. To examine this hypothesis, we introduce a novel perceptual boundary sampling method to generate facial expression stimuli that lie along ANN decision boundaries. These ambiguous samples form the basis of the varEmotion dataset, constructed through large-scale human behavioral experiments. Our analysis reveals that these ANN-confusing stimuli also provoke heightened perceptual uncertainty in human participants, highlighting shared computational principles in emotion perception. Finally, by fine-tuning ANN representations using behavioral data, we achieve alignment between ANN predictions and both group-level and individual-level human perceptual patterns. Our findings establish a systematic link between ANN decision boundaries and human perceptual variability, offering new insights into personalized modeling of emotional interpretation.
>
---
#### [new 138] Visual-Language Model Knowledge Distillation Method for Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决CLIP模型参数过多和局部失真特征识别不足的问题。通过设计质量评分提示模板、微调CLIP，并提出模态自适应的知识蒸馏策略，指导学生模型学习，实验证明该方法在降低复杂度的同时优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.15680v1](http://arxiv.org/pdf/2507.15680v1)**

> **作者:** Yongkang Hou; Jiarun Song
>
> **摘要:** Image Quality Assessment (IQA) is a core task in computer vision. Multimodal methods based on vision-language models, such as CLIP, have demonstrated exceptional generalization capabilities in IQA tasks. To address the issues of excessive parameter burden and insufficient ability to identify local distorted features in CLIP for IQA, this study proposes a visual-language model knowledge distillation method aimed at guiding the training of models with architectural advantages using CLIP's IQA knowledge. First, quality-graded prompt templates were designed to guide CLIP to output quality scores. Then, CLIP is fine-tuned to enhance its capabilities in IQA tasks. Finally, a modality-adaptive knowledge distillation strategy is proposed to achieve guidance from the CLIP teacher model to the student model. Our experiments were conducted on multiple IQA datasets, and the results show that the proposed method significantly reduces model complexity while outperforming existing IQA methods, demonstrating strong potential for practical deployment.
>
---
#### [new 139] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于三维重建与新视角渲染任务，旨在解决稀疏视角图像下的表面重建与视图生成问题。现有方法在细节或全局几何上存在不足，本文提出SurfaceSplat，结合SDF与3D高斯点绘方法，相互优化几何与渲染效果，提升重建与渲染质量。**

- **链接: [http://arxiv.org/pdf/2507.15602v1](http://arxiv.org/pdf/2507.15602v1)**

> **作者:** Zihui Gao; Jia-Wang Bian; Guosheng Lin; Hao Chen; Chunhua Shen
>
> **摘要:** Surface reconstruction and novel view rendering from sparse-view images are challenging. Signed Distance Function (SDF)-based methods struggle with fine details, while 3D Gaussian Splatting (3DGS)-based approaches lack global geometry coherence. We propose a novel hybrid method that combines the strengths of both approaches: SDF captures coarse geometry to enhance 3DGS-based rendering, while newly rendered images from 3DGS refine the details of SDF for accurate surface reconstruction. As a result, our method surpasses state-of-the-art approaches in surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets. Code will be released at https://github.com/Gaozihui/SurfaceSplat.
>
---
#### [new 140] Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）模型任务，旨在解决现有模型在复杂操作任务中灵活性差、泛化能力弱的问题。作者提出了Being-H0，通过大规模人类视频进行预训练，引入物理指令调优和部分动作分层编码，提升手部动作生成与指令跟随能力，并支持机器人实际操作。**

- **链接: [http://arxiv.org/pdf/2507.15597v1](http://arxiv.org/pdf/2507.15597v1)**

> **作者:** Hao Luo; Yicheng Feng; Wanpeng Zhang; Sipeng Zheng; Ye Wang; Haoqi Yuan; Jiazheng Liu; Chaoyi Xu; Qin Jin; Zongqing Lu
>
> **备注:** 37 pages
>
> **摘要:** We introduce Being-H0, a dexterous Vision-Language-Action model (VLA) trained on large-scale human videos. Existing VLAs struggle with complex manipulation tasks requiring high dexterity and generalize poorly to novel scenarios and tasks, primarily due to their reliance on synthetic data with significant sim-to-real gaps or teleoperated demonstrations lacking scale and diversity. To address this data bottleneck, we propose leveraging human hands as a foundation manipulator, capitalizing on the rich dexterity and scalability present in web data. Our approach centers on physical instruction tuning, a novel training paradigm that combines large-scale VLA pretraining from human videos, physical space alignment for 3D reasoning, and post-training adaptation for robotic tasks. Additionally, we introduce a part-level motion tokenization method which achieves millimeter-level reconstruction accuracy to model precise hand trajectories for action learning. To support our proposed paradigm, we further develop a comprehensive data curation pipeline that integrates heterogeneous sources -- including motion capture, VR, and RGB-only videos -- into a large-scale dataset with millions of motion-based instructional instances. We empirically show the excellence of Being-H0 in hand motion generation and instruction following, and it also scales well with model and data sizes. Importantly, we observe the expected gains of Being-H0 in real-world robotic manipulation as physical instruction tuning is applied. More details are available at https://beingbeyond.github.io/Being-H0.
>
---
#### [new 141] Real Time Captioning of Sign Language Gestures in Video Meetings
- **分类: cs.CV; cs.CY; cs.HC; cs.LG; I.4.6**

- **简介: 该论文属于计算机视觉与自然语言处理任务，旨在解决听障人士在视频会议中沟通困难的问题。作者提出了一种浏览器扩展工具，可实时将美国手语（ASL）手势识别并翻译为字幕，促进听障人士与普通人的无障碍交流。**

- **链接: [http://arxiv.org/pdf/2507.14543v1](http://arxiv.org/pdf/2507.14543v1)**

> **作者:** Sharanya Mukherjee; Md Hishaam Akhtar; Kannadasan R
>
> **备注:** 7 pages, 2 figures, 1 table, Presented at ICCMDE 2021
>
> **摘要:** It has always been a rather tough task to communicate with someone possessing a hearing impairment. One of the most tested ways to establish such a communication is through the use of sign based languages. However, not many people are aware of the smaller intricacies involved with sign language. Sign language recognition using computer vision aims at eliminating the communication barrier between deaf-mute and ordinary people so that they can properly communicate with others. Recently the pandemic has left the whole world shaken up and has transformed the way we communicate. Video meetings have become essential for everyone, even people with a hearing disability. In recent studies, it has been found that people with hearing disabilities prefer to sign over typing during these video calls. In this paper, we are proposing a browser extension that will automatically translate sign language to subtitles for everyone else in the video call. The Large-scale dataset which contains more than 2000 Word-Level ASL videos, which were performed by over 100 signers will be used.
>
---
#### [new 142] Smart Eyes for Silent Threats: VLMs and In-Context Learning for THz Imaging
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决太赫兹（THz）成像中因标注数据少、分辨率低和视觉模糊导致的分类难题。作者采用无需微调的视觉-语言模型（VLMs）结合上下文学习（ICL），通过模态对齐提示框架，在零样本和一样本设置下提升了分类效果。这是首次将ICL增强的VLMs应用于THz成像领域。**

- **链接: [http://arxiv.org/pdf/2507.15576v1](http://arxiv.org/pdf/2507.15576v1)**

> **作者:** Nicolas Poggi; Shashank Agnihotri; Margret Keuper
>
> **摘要:** Terahertz (THz) imaging enables non-invasive analysis for applications such as security screening and material classification, but effective image classification remains challenging due to limited annotations, low resolution, and visual ambiguity. We introduce In-Context Learning (ICL) with Vision-Language Models (VLMs) as a flexible, interpretable alternative that requires no fine-tuning. Using a modality-aligned prompting framework, we adapt two open-weight VLMs to the THz domain and evaluate them under zero-shot and one-shot settings. Our results show that ICL improves classification and interpretability in low-data regimes. This is the first application of ICL-enhanced VLMs to THz imaging, offering a promising direction for resource-constrained scientific domains. Code: \href{https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main}{GitHub repository}.
>
---
#### [new 143] QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于图像逆问题任务，旨在解决深度学习模型在科学和医学成像中可能生成虚假细节的问题。作者提出QUTCC方法，通过非线性和非均匀量化训练与校准，实现更精确的不确定性估计，有效识别图像重建中的幻觉并提供更紧致的置信区间。**

- **链接: [http://arxiv.org/pdf/2507.14760v1](http://arxiv.org/pdf/2507.14760v1)**

> **作者:** Cassandra Tong Ye; Shamus Li; Tyler King; Kristina Monakhova
>
> **摘要:** Deep learning models often hallucinate, producing realistic artifacts that are not truly present in the sample. This can have dire consequences for scientific and medical inverse problems, such as MRI and microscopy denoising, where accuracy is more important than perceptual quality. Uncertainty quantification techniques, such as conformal prediction, can pinpoint outliers and provide guarantees for image regression tasks, improving reliability. However, existing methods utilize a linear constant scaling factor to calibrate uncertainty bounds, resulting in larger, less informative bounds. We propose QUTCC, a quantile uncertainty training and calibration technique that enables nonlinear, non-uniform scaling of quantile predictions to enable tighter uncertainty estimates. Using a U-Net architecture with a quantile embedding, QUTCC enables the prediction of the full conditional distribution of quantiles for the imaging task. During calibration, QUTCC generates uncertainty bounds by iteratively querying the network for upper and lower quantiles, progressively refining the bounds to obtain a tighter interval that captures the desired coverage. We evaluate our method on several denoising tasks as well as compressive MRI reconstruction. Our method successfully pinpoints hallucinations in image estimates and consistently achieves tighter uncertainty intervals than prior methods while maintaining the same statistical coverage.
>
---
#### [new 144] On Splitting Lightweight Semantic Image Segmentation for Wireless Communications
- **分类: cs.NI; cs.CV; eess.IV**

- **简介: 该论文属于语义通信任务，旨在解决资源受限环境下图像语义分割的通信成本与计算负担问题。论文提出将语义分割过程拆分，由发送端和接收端协同完成，从而降低传输带宽和发送端计算压力，同时保持分割精度。实验表明该方法可减少72%的传输比特率和19%以上的计算负载。**

- **链接: [http://arxiv.org/pdf/2507.14199v1](http://arxiv.org/pdf/2507.14199v1)**

> **作者:** Ebrahim Abu-Helalah; Jordi Serra; Jordi Perez-Romero
>
> **备注:** IEEE International Mediterranean Conference on Communications and Networking
>
> **摘要:** Semantic communication represents a promising technique towards reducing communication costs, especially when dealing with image segmentation, but it still lacks a balance between computational efficiency and bandwidth requirements while maintaining high image segmentation accuracy, particularly in resource-limited environments and changing channel conditions. On the other hand, the more complex and larger semantic image segmentation models become, the more stressed the devices are when processing data. This paper proposes a novel approach to implementing semantic communication based on splitting the semantic image segmentation process between a resource constrained transmitter and the receiver. This allows saving bandwidth by reducing the transmitted data while maintaining the accuracy of the semantic image segmentation. Additionally, it reduces the computational requirements at the resource constrained transmitter compared to doing all the semantic image segmentation in the transmitter. The proposed approach is evaluated by means of simulation-based experiments in terms of different metrics such as computational resource usage, required bit rate and segmentation accuracy. The results when comparing the proposal with the full semantic image segmentation in the transmitter show that up to 72% of the bit rate was reduced in the transmission process. In addition, the computational load of the transmitter is reduced by more than 19%. This reflects the interest of this technique for its application in communication systems, particularly in the upcoming 6G systems.
>
---
#### [new 145] APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **简介: 论文提出了一种统一的可训练神经元结构APTx Neuron，将线性变换与非线性激活整合到单一表达式中，旨在提升神经网络的计算效率与表达能力。该结构在MNIST数据集上验证，表现出较高的准确率与参数效率。论文属于神经网络架构设计任务，旨在简化传统神经元结构并提升性能。**

- **链接: [http://arxiv.org/pdf/2507.14270v1](http://arxiv.org/pdf/2507.14270v1)**

> **作者:** Ravin Kumar
>
> **备注:** 10 pages, 2 figures, 1 table, and GitHub repository for the source code
>
> **摘要:** We propose the APTx Neuron, a novel, unified neural computation unit that integrates non-linear activation and linear transformation into a single trainable expression. The APTx Neuron is derived from the APTx activation function, thereby eliminating the need for separate activation layers and making the architecture both computationally efficient and elegant. The proposed neuron follows the functional form $y = \sum_{i=1}^{n} ((\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i) + \delta$, where all parameters $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ are trainable. We validate our APTx Neuron-based architecture on the MNIST dataset, achieving up to 96.69\% test accuracy in just 20 epochs using approximately 332K trainable parameters. The results highlight the superior expressiveness and computational efficiency of the APTx Neuron compared to traditional neurons, pointing toward a new paradigm in unified neuron design and the architectures built upon it.
>
---
#### [new 146] Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态图表推理任务，旨在解决现有方法在复杂图表推理上的不足。作者提出了Chart-R1模型，结合思维链监督与强化学习微调，并通过程序化数据合成生成高质量推理数据。实验表明，Chart-R1在图表推理表现上优于现有方法，甚至媲美GPT-4o、Claude-3.5等大模型。**

- **链接: [http://arxiv.org/pdf/2507.15509v1](http://arxiv.org/pdf/2507.15509v1)**

> **作者:** Lei Chen; Xuanle Zhao; Zhixiong Zeng; Jing Huang; Yufeng Zhong; Lin Ma
>
> **备注:** technical report
>
> **摘要:** Recently, inspired by OpenAI-o1/o3 and Deepseek-R1, the R1-Style method based on reinforcement learning fine-tuning has received widespread attention from the community. Previous R1-Style methods mainly focus on mathematical reasoning and code intelligence. It is of great research significance to verify their advantages on more general multimodal data. Chart is an important multimodal data type with rich information, which brings important research challenges in complex reasoning. In this work, we introduce Chart-R1, a chart-domain vision-language model with reinforcement learning fine-tuning to enable complex chart reasoning. To support Chart-R1, we first propose a novel programmatic data synthesis technology to generate high-quality step-by-step chart reasoning data covering single- and multi-subcharts, which makes up for the lack of reasoning data in the chart domain. Then we develop a two-stage training strategy: Chart-COT with step-by-step chain-of-thought supervision, and Chart-RFT with numerically sensitive reinforcement fine-tuning. Chart-COT aims to decompose complex chart reasoning tasks into fine-grained, understandable subtasks through step-by-step supervision, which lays a good foundation for improving the reasoning level of reinforcement learning. Chart-RFT utilize the typical group relative policy optimization strategy, in which a relatively soft reward is adopted for numerical response to emphasize the numerical sensitivity in the chart domain. We conduct extensive experiments on open-source benchmarks and self-built chart reasoning dataset (\emph{i.e., ChartRQA}). Experimental results show that Chart-R1 has significant advantages compared to chart-domain methods, even comparable to open/closed source large-scale models (\emph{e.g., GPT-4o, Claude-3.5}).
>
---
#### [new 147] RARE-UNet: Resolution-Aligned Routing Entry for Adaptive Medical Image Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决现有模型在低分辨率输入下性能下降的问题。作者提出了RARE-UNet，一种分辨率感知的多尺度分割架构，通过动态调整推理路径，实现对不同分辨率输入的自适应分割，提升了模型在低分辨率下的性能与推理速度。**

- **链接: [http://arxiv.org/pdf/2507.15524v1](http://arxiv.org/pdf/2507.15524v1)**

> **作者:** Simon Winther Albertsen; Hjalte Svaneborg Bjørnstrup; Mostafa Mehdipour Ghazi
>
> **备注:** EMA4MICCAI 2025
>
> **摘要:** Accurate segmentation is crucial for clinical applications, but existing models often assume fixed, high-resolution inputs and degrade significantly when faced with lower-resolution data in real-world scenarios. To address this limitation, we propose RARE-UNet, a resolution-aware multi-scale segmentation architecture that dynamically adapts its inference path to the spatial resolution of the input. Central to our design are multi-scale blocks integrated at multiple encoder depths, a resolution-aware routing mechanism, and consistency-driven training that aligns multi-resolution features with full-resolution representations. We evaluate RARE-UNet on two benchmark brain imaging tasks for hippocampus and tumor segmentation. Compared to standard UNet, its multi-resolution augmented variant, and nnUNet, our model achieves the highest average Dice scores of 0.84 and 0.65 across resolution, while maintaining consistent performance and significantly reduced inference time at lower resolutions. These results highlight the effectiveness and scalability of our architecture in achieving resolution-robust segmentation. The codes are available at: https://github.com/simonsejse/RARE-UNet.
>
---
#### [new 148] Self-Supervised Joint Reconstruction and Denoising of T2-Weighted PROPELLER MRI of the Lungs at 0.55T
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像重建任务，旨在提升低场强T2加权肺部MRI图像质量。作者提出一种自监督联合重建与去噪模型，利用K空间数据的内在冗余结构，在无需干净标签的情况下实现图像优化。实验表明该方法优于传统MPPCA方法，提升图像清晰度并减少扫描时间。**

- **链接: [http://arxiv.org/pdf/2507.14308v1](http://arxiv.org/pdf/2507.14308v1)**

> **作者:** Jingjia Chen; Haoyang Pei; Christoph Maier; Mary Bruno; Qiuting Wen; Seon-Hi Shin; William Moore; Hersh Chandarana; Li Feng
>
> **摘要:** Purpose: This study aims to improve 0.55T T2-weighted PROPELLER lung MRI through a self-supervised joint reconstruction and denoising model. Methods: T2-weighted 0.55T lung MRI dataset including 44 patients with previous covid infection were used. A self-supervised learning framework was developed, where each blade of the PROPELLER acquisition was split along the readout direction into two partitions. One subset trains the unrolled reconstruction network, while the other subset is used for loss calculation, enabling self-supervised training without clean targets and leveraging matched noise statistics for denoising. For comparison, Marchenko-Pastur Principal Component Analysis (MPPCA) was performed along the coil dimension, followed by conventional parallel imaging reconstruction. The quality of the reconstructed lung MRI was assessed visually by two experienced radiologists independently. Results: The proposed self-supervised model improved the clarity and structural integrity of the lung images. For cases with available CT scans, the reconstructed images demonstrated strong alignment with corresponding CT images. Additionally, the proposed model enables further scan time reduction by requiring only half the number of blades. Reader evaluations confirmed that the proposed method outperformed MPPCA-denoised images across all categories (Wilcoxon signed-rank test, p<0.001), with moderate inter-reader agreement (weighted Cohen's kappa=0.55; percentage of exact and within +/-1 point agreement=91%). Conclusion: By leveraging intrinsic structural redundancies between two disjoint splits of k-space subsets, the proposed self-supervised learning model effectively reconstructs the image while suppressing the noise for 0.55T T2-weighted lung MRI with PROPELLER sampling.
>
---
#### [new 149] GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于GUI交互任务，旨在解决自然语言到界面位置映射中奖励信号稀疏的问题。作者提出GUI-G²框架，用高斯分布建模界面元素，通过点奖励与覆盖奖励机制，实现连续优化，显著提升交互准确性。**

- **链接: [http://arxiv.org/pdf/2507.15846v1](http://arxiv.org/pdf/2507.15846v1)**

> **作者:** Fei Tang; Zhangxuan Gu; Zhengxi Lu; Xuyang Liu; Shuheng Shen; Changhua Meng; Wen Wang; Wenqi Zhang; Yongliang Shen; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **摘要:** Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current reinforcement learning approaches use binary rewards that treat elements as hit-or-miss targets, creating sparse signals that ignore the continuous nature of spatial interactions. Motivated by human clicking behavior that naturally forms Gaussian distributions centered on target elements, we introduce GUI Gaussian Grounding Rewards (GUI-G$^2$), a principled reward framework that models GUI elements as continuous Gaussian distributions across the interface plane. GUI-G$^2$ incorporates two synergistic mechanisms: Gaussian point rewards model precise localization through exponentially decaying distributions centered on element centroids, while coverage rewards assess spatial alignment by measuring the overlap between predicted Gaussian distributions and target regions. To handle diverse element scales, we develop an adaptive variance mechanism that calibrates reward distributions based on element dimensions. This framework transforms GUI grounding from sparse binary classification to dense continuous optimization, where Gaussian distributions generate rich gradient signals that guide models toward optimal interaction positions. Extensive experiments across ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro benchmarks demonstrate that GUI-G$^2$, substantially outperforms state-of-the-art method UI-TARS-72B, with the most significant improvement of 24.7% on ScreenSpot-Pro. Our analysis reveals that continuous modeling provides superior robustness to interface variations and enhanced generalization to unseen layouts, establishing a new paradigm for spatial reasoning in GUI interaction tasks.
>
---
#### [new 150] Low-Latency Event-Based Velocimetry for Quadrotor Control in a Narrow Pipe
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于无人机控制任务，旨在解决四旋翼飞行器在狭窄管道内悬停时受气流扰动不稳定的问题。工作包括：提出低延迟烟雾测速方法，结合神经网络估计气流扰动，并通过强化学习控制器实现实时反馈控制，有效应对管道内复杂气流，首次实现闭环控制飞行。**

- **链接: [http://arxiv.org/pdf/2507.15444v1](http://arxiv.org/pdf/2507.15444v1)**

> **作者:** Leonard Bauersfeld; Davide Scaramuzza
>
> **备注:** 17 pages
>
> **摘要:** Autonomous quadrotor flight in confined spaces such as pipes and tunnels presents significant challenges due to unsteady, self-induced aerodynamic disturbances. Very recent advances have enabled flight in such conditions, but they either rely on constant motion through the pipe to mitigate airflow recirculation effects or suffer from limited stability during hovering. In this work, we present the first closed-loop control system for quadrotors for hovering in narrow pipes that leverages real-time flow field measurements. We develop a low-latency, event-based smoke velocimetry method that estimates local airflow at high temporal resolution. This flow information is used by a disturbance estimator based on a recurrent convolutional neural network, which infers force and torque disturbances in real time. The estimated disturbances are integrated into a learning-based controller trained via reinforcement learning. The flow-feedback control proves particularly effective during lateral translation maneuvers in the pipe cross-section. There, the real-time disturbance information enables the controller to effectively counteract transient aerodynamic effects, thereby preventing collisions with the pipe wall. To the best of our knowledge, this work represents the first demonstration of an aerial robot with closed-loop control informed by real-time flow field measurements. This opens new directions for research on flight in aerodynamically complex environments. In addition, our work also sheds light on the characteristic flow structures that emerge during flight in narrow, circular pipes, providing new insights at the intersection of robotics and fluid dynamics.
>
---
#### [new 151] Latent Space Synergy: Text-Guided Data Augmentation for Direct Diffusion Biomedical Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决数据稀缺导致的息肉检测难题。作者提出SynDiff框架，结合文本引导的合成数据生成与扩散模型，实现单步推理分割。通过生成多样化的合成息肉数据增强训练，提升了模型鲁棒性，在CVC-ClinicDB数据集上表现出色，适用于资源有限的临床场景。**

- **链接: [http://arxiv.org/pdf/2507.15361v1](http://arxiv.org/pdf/2507.15361v1)**

> **作者:** Muhammad Aqeel; Maham Nazir; Zanxi Ruan; Francesco Setti
>
> **备注:** Accepted to CVGMMI Workshop at ICIAP 2025
>
> **摘要:** Medical image segmentation suffers from data scarcity, particularly in polyp detection where annotation requires specialized expertise. We present SynDiff, a framework combining text-guided synthetic data generation with efficient diffusion-based segmentation. Our approach employs latent diffusion models to generate clinically realistic synthetic polyps through text-conditioned inpainting, augmenting limited training data with semantically diverse samples. Unlike traditional diffusion methods requiring iterative denoising, we introduce direct latent estimation enabling single-step inference with T x computational speedup. On CVC-ClinicDB, SynDiff achieves 96.0% Dice and 92.9% IoU while maintaining real-time capability suitable for clinical deployment. The framework demonstrates that controlled synthetic augmentation improves segmentation robustness without distribution shift. SynDiff bridges the gap between data-hungry deep learning models and clinical constraints, offering an efficient solution for deployment in resourcelimited medical settings.
>
---
#### [new 152] The Origin of Self-Attention: From Pairwise Affinity Matrices to Transformers
- **分类: cs.LG; cs.CV; 68T07, 05C50, 15A18; I.2.6; I.2.7; I.5.1**

- **简介: 该论文属于机器学习模型机制分析任务，旨在揭示自注意力机制的起源与本质。论文通过追溯自注意力在计算机视觉、自然语言处理和图学习中的发展，提出其核心是基于成对亲和矩阵（affinity matrix）的信息流动控制。工作重点在于将自注意力与Inf-FS方法关联，表明其为Inf-FS的一种特例，并统一不同模型的数学基础。**

- **链接: [http://arxiv.org/pdf/2507.14560v1](http://arxiv.org/pdf/2507.14560v1)**

> **作者:** Giorgio Roffo
>
> **备注:** 24 pages, 10 figures, submitted for review. Companion code and reproducibility materials available
>
> **摘要:** The self-attention mechanism, now central to deep learning architectures such as Transformers, is a modern instance of a more general computational principle: learning and using pairwise affinity matrices to control how information flows through a model. This paper traces the conceptual origins of self-attention across multiple domains, including computer vision, natural language processing, and graph learning, through their shared reliance on an affinity matrix, denoted as A. We highlight Infinite Feature Selection (Inf-FS) as a foundational approach that generalizes the idea of affinity-based weighting. Unlike the fixed dot-product structure used in Transformers, Inf-FS defines A either through domain knowledge or by learning, and computes feature relevance through multi-hop propagation over the affinity graph. From this perspective, self-attention can be seen as a special case of Inf-FS: it uses a single-hop affinity computation where A is dynamically built from token similarities. We argue that the underlying structure, reasoning over pairwise relationships, is preserved across both approaches, and the key differences lie in how the affinity matrix is defined and applied. By situating self-attention within the broader paradigm of affinity-based computation, we unify several strands of machine learning research and highlight a common mathematical foundation that underpins diverse models and tasks.
>
---
#### [new 153] MiDeSeC: A Dataset for Mitosis Detection and Segmentation in Breast Cancer Histopathology Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出了MiDeSeC数据集，用于乳腺癌组织切片中分裂细胞的检测与分割任务。为解决因有丝分裂形态多样而需大量数据覆盖的问题，作者采集了25名患者的50个高分辨率区域，包含500多个有丝分裂实例，并划分训练与测试集以支持模型开发。**

- **链接: [http://arxiv.org/pdf/2507.14271v1](http://arxiv.org/pdf/2507.14271v1)**

> **作者:** Refik Samet; Nooshin Nemati; Emrah Hancer; Serpil Sak; Bilge Ayca Kirmizi; Zeynep Yildirim
>
> **摘要:** The MiDeSeC dataset is created through H&E stained invasive breast carcinoma, no special type (NST) slides of 25 different patients captured at 40x magnification from the Department of Medical Pathology at Ankara University. The slides have been scanned by 3D Histech Panoramic p250 Flash-3 scanner and Olympus BX50 microscope. As several possible mitosis shapes exist, it is crucial to have a large dataset to cover all the cases. Accordingly, a total of 50 regions is selected from glass slides for 25 patients, each of regions with a size of 1024*1024 pixels. There are more than 500 mitoses in total in these 50 regions. Two-thirds of the regions are reserved for training, the other third for testing.
>
---
#### [new 154] U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning with MLLMs
- **分类: cs.IR; cs.CV**

- **简介: 该论文属于通用多模态检索（UMR）任务，旨在解决跨模态复杂检索问题。通过分析基于MLLM的嵌入学习方法，探索影响检索性能的关键因素，并提出U-MARVEL框架，改进训练策略和嵌入生成方法，以提升模型表现及泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.14902v1](http://arxiv.org/pdf/2507.14902v1)**

> **作者:** Xiaojie Li; Chu Li; Shi-Zhe Chen; Xi Chen
>
> **备注:** Technical Report (in progress)
>
> **摘要:** Universal multimodal retrieval (UMR), which aims to address complex retrieval tasks where both queries and candidates span diverse modalities, has been significantly advanced by the emergence of MLLMs. While state-of-the-art MLLM-based methods in the literature predominantly adopt contrastive learning principles, they often differ in their specific training recipes. Despite their success, the mechanisms underlying their retrieval capabilities remain largely unexplored, potentially resulting in suboptimal performance and limited generalization ability. To address these issues, we present a comprehensive study aimed at uncovering the key factors that drive effective embedding learning for UMR using MLLMs. We begin by implementing a general MLLM-based embedding learning pipeline, and systematically analyze the primary contributors to high-performing universal retrieval systems. Based on this, we explore various aspects of the details in embedding generation and training strategies, including progressive transition, hard negative mining and re-ranker distillation. Notably, our findings reveal that often-overlooked factors can have a substantial impact on model performance. Building on these discoveries, we introduce a unified framework termed U-MARVEL (\textbf{U}niversal \textbf{M}ultimod\textbf{A}l \textbf{R}etrie\textbf{V}al via \textbf{E}mbedding \textbf{L}earning), which outperforms state-of-the-art competitors on the M-BEIR benchmark by a large margin in supervised settings, and also exihibits strong zero-shot performance on several tasks such as composed image retrieval and text-to-video retrieval. These results underscore the generalization potential of our framework across various embedding-based retrieval tasks. Code is available at https://github.com/chaxjli/U-MARVEL
>
---
#### [new 155] EndoControlMag: Robust Endoscopic Vascular Motion Magnification with Periodic Reference Resetting and Hierarchical Tissue-aware Dual-Mask Contro
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决内镜手术中血管微小运动可视化难题。作者提出EndoControlMag框架，通过周期参考重置和分层组织感知双掩码策略，实现无需训练的血管运动放大，有效应对遮挡、器械干扰等复杂场景，提升手术精度与决策能力。**

- **链接: [http://arxiv.org/pdf/2507.15292v1](http://arxiv.org/pdf/2507.15292v1)**

> **作者:** An Wanga; Rulin Zhou; Mengya Xu; Yiru Ye; Longfei Gou; Yiting Chang; Hao Chen; Chwee Ming Lim; Jiankun Wang; Hongliang Ren
>
> **摘要:** Visualizing subtle vascular motions in endoscopic surgery is crucial for surgical precision and decision-making, yet remains challenging due to the complex and dynamic nature of surgical scenes. To address this, we introduce EndoControlMag, a training-free, Lagrangian-based framework with mask-conditioned vascular motion magnification tailored to endoscopic environments. Our approach features two key modules: a Periodic Reference Resetting (PRR) scheme that divides videos into short overlapping clips with dynamically updated reference frames to prevent error accumulation while maintaining temporal coherence, and a Hierarchical Tissue-aware Magnification (HTM) framework with dual-mode mask dilation. HTM first tracks vessel cores using a pretrained visual tracking model to maintain accurate localization despite occlusions and view changes. It then applies one of two adaptive softening strategies to surrounding tissues: motion-based softening that modulates magnification strength proportional to observed tissue displacement, or distance-based exponential decay that simulates biomechanical force attenuation. This dual-mode approach accommodates diverse surgical scenarios-motion-based softening excels with complex tissue deformations while distance-based softening provides stability during unreliable optical flow conditions. We evaluate EndoControlMag on our EndoVMM24 dataset spanning four different surgery types and various challenging scenarios, including occlusions, instrument disturbance, view changes, and vessel deformations. Quantitative metrics, visual assessments, and expert surgeon evaluations demonstrate that EndoControlMag significantly outperforms existing methods in both magnification accuracy and visual quality while maintaining robustness across challenging surgical conditions. The code, dataset, and video results are available at https://szupc.github.io/EndoControlMag/.
>
---
#### [new 156] Prompt-aware of Frame Sampling for Efficient Text-Video Retrieval
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于文本-视频检索任务，旨在解决现有方法在准确性和计算效率间的权衡问题。论文提出ProCLIP框架，通过提示感知的帧采样策略和两阶段候选剪枝方法，在保持高准确率的同时显著提升检索效率。**

- **链接: [http://arxiv.org/pdf/2507.15491v1](http://arxiv.org/pdf/2507.15491v1)**

> **作者:** Deyu Zhang; Tingting Long; Jinrui Zhang; Ligeng Chen; Ju Ren; Yaoxue Zhang
>
> **摘要:** Enabling efficient text-video retrieval on edge-end devices is critical for real-world applications. Yet, existing methods face a critical challenge in balancing accuracy and computational efficiency: uniform frame sampling methods ensure content coverage but incur prohibitive computational costs, while salient-frame sampling methods reduce overhead but suffer from query-agnostic frame selection that biases retrieval results. To address this, we propose ProCLIP, a user-centric framework that achieves state-of-the-art accuracy with significantly improved efficiency. We design a prompt-aware frame sampling strategy that dynamically guides lightweight feature extractors using textual prompts to select semantically relevant frames, overcoming the limitations of existing salient-frame sampling methods which rely on static, query-agnostic selection criteria. Moreover, we adopt a two-stage candidate pruning strategy that combines rapid coarse filtering via a lightweight module with CLIP-powered fine-grained re-ranking, enhancing retrieval efficiency while preserving accuracy. Experiments across benchmarks show ProCLIP achieves 75.3% latency reduction versus baselines while maintaining competitive accuracy, i.e., R@1=49.0 in MSR-VTT dataset. Code is available at https://github.com/tiffylong/ProCLIP.
>
---
#### [new 157] Exposing and Mitigating Calibration Biases and Demographic Unfairness in MLLM Few-Shot In-Context Learning for Medical Image Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文研究多模态大语言模型（MLLMs）在医疗图像分类中的小样本上下文学习任务，旨在解决模型预测置信度在校准和人口统计公平性方面存在的偏差问题。作者提出了CALIN方法，通过分层估计校准矩阵，在推理时对置信度进行校准，以提升准确性和公平性。实验表明该方法在多个医疗图像数据集上有效。**

- **链接: [http://arxiv.org/pdf/2506.23298v3](http://arxiv.org/pdf/2506.23298v3)**

> **作者:** Xing Shen; Justin Szeto; Mingyang Li; Hengguan Huang; Tal Arbel
>
> **备注:** Preprint version. The peer-reviewed version of this paper has been accepted to MICCAI 2025 main conference
>
> **摘要:** Multimodal large language models (MLLMs) have enormous potential to perform few-shot in-context learning in the context of medical image analysis. However, safe deployment of these models into real-world clinical practice requires an in-depth analysis of the accuracies of their predictions, and their associated calibration errors, particularly across different demographic subgroups. In this work, we present the first investigation into the calibration biases and demographic unfairness of MLLMs' predictions and confidence scores in few-shot in-context learning for medical image classification. We introduce CALIN, an inference-time calibration method designed to mitigate the associated biases. Specifically, CALIN estimates the amount of calibration needed, represented by calibration matrices, using a bi-level procedure: progressing from the population level to the subgroup level prior to inference. It then applies this estimation to calibrate the predicted confidence scores during inference. Experimental results on three medical imaging datasets: PAPILA for fundus image classification, HAM10000 for skin cancer classification, and MIMIC-CXR for chest X-ray classification demonstrate CALIN's effectiveness at ensuring fair confidence calibration in its prediction, while improving its overall prediction accuracies and exhibiting minimum fairness-utility trade-off. Our codebase can be found at https://github.com/xingbpshen/medical-calibration-fairness-mllm.
>
---
#### [new 158] CXR-TFT: Multi-Modal Temporal Fusion Transformer for Predicting Chest X-ray Trajectories
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出CXR-TFT模型，旨在通过融合胸部X光（CXR）图像、放射报告及高频临床数据（如生命体征），预测重症患者CXR变化轨迹。任务属医疗预测，解决CXR检查频率低、传统方法无法捕捉时间动态的问题。模型采用Transformer结构，结合图像嵌入与临床数据插值，实现CXR异常的早期预测，提升危重病症如ARDS的及时干预能力。**

- **链接: [http://arxiv.org/pdf/2507.14766v1](http://arxiv.org/pdf/2507.14766v1)**

> **作者:** Mehak Arora; Ayman Ali; Kaiyuan Wu; Carolyn Davis; Takashi Shimazui; Mahmoud Alwakeel; Victor Moas; Philip Yang; Annette Esper; Rishikesan Kamaleswaran
>
> **备注:** In Review for MICCAI 2025
>
> **摘要:** In intensive care units (ICUs), patients with complex clinical conditions require vigilant monitoring and prompt interventions. Chest X-rays (CXRs) are a vital diagnostic tool, providing insights into clinical trajectories, but their irregular acquisition limits their utility. Existing tools for CXR interpretation are constrained by cross-sectional analysis, failing to capture temporal dynamics. To address this, we introduce CXR-TFT, a novel multi-modal framework that integrates temporally sparse CXR imaging and radiology reports with high-frequency clinical data, such as vital signs, laboratory values, and respiratory flow sheets, to predict the trajectory of CXR findings in critically ill patients. CXR-TFT leverages latent embeddings from a vision encoder that are temporally aligned with hourly clinical data through interpolation. A transformer model is then trained to predict CXR embeddings at each hour, conditioned on previous embeddings and clinical measurements. In a retrospective study of 20,000 ICU patients, CXR-TFT demonstrated high accuracy in forecasting abnormal CXR findings up to 12 hours before they became radiographically evident. This predictive capability in clinical data holds significant potential for enhancing the management of time-sensitive conditions like acute respiratory distress syndrome, where early intervention is crucial and diagnoses are often delayed. By providing distinctive temporal resolution in prognostic CXR analysis, CXR-TFT offers actionable 'whole patient' insights that can directly improve clinical outcomes.
>
---
#### [new 159] Breaking the Illusion of Security via Interpretation: Interpretable Vision Transformer Systems under Attack
- **分类: cs.CR; cs.AI; cs.CV; cs.LG; I.2.10; I.2.6; I.5.1; D.4.6; K.6.5**

- **简介: 该论文研究视觉Transformer（ViT）模型在结合解释模型时的安全性问题，旨在解决对抗攻击下模型与解释同时被欺骗的漏洞。作者提出了名为AdViT的攻击方法，成功误导模型预测与解释，实验证明其在白盒和黑盒场景下攻击成功率均为100%，并保持高置信度与解释准确性，增加攻击隐蔽性。**

- **链接: [http://arxiv.org/pdf/2507.14248v1](http://arxiv.org/pdf/2507.14248v1)**

> **作者:** Eldor Abdukhamidov; Mohammed Abuhamad; Simon S. Woo; Hyoungshick Kim; Tamer Abuhmed
>
> **摘要:** Vision transformer (ViT) models, when coupled with interpretation models, are regarded as secure and challenging to deceive, making them well-suited for security-critical domains such as medical applications, autonomous vehicles, drones, and robotics. However, successful attacks on these systems can lead to severe consequences. Recent research on threats targeting ViT models primarily focuses on generating the smallest adversarial perturbations that can deceive the models with high confidence, without considering their impact on model interpretations. Nevertheless, the use of interpretation models can effectively assist in detecting adversarial examples. This study investigates the vulnerability of transformer models to adversarial attacks, even when combined with interpretation models. We propose an attack called "AdViT" that generates adversarial examples capable of misleading both a given transformer model and its coupled interpretation model. Through extensive experiments on various transformer models and two transformer-based interpreters, we demonstrate that AdViT achieves a 100% attack success rate in both white-box and black-box scenarios. In white-box scenarios, it reaches up to 98% misclassification confidence, while in black-box scenarios, it reaches up to 76% misclassification confidence. Remarkably, AdViT consistently generates accurate interpretations in both scenarios, making the adversarial examples more difficult to detect.
>
---
#### [new 160] Performance Analysis of Post-Training Quantization for CNN-based Conjunctival Pallor Anemia Detection
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在通过结膜苍白检测儿童贫血。论文使用CP-AnemiC数据集和MobileNet模型，评估不同量化方案对模型性能的影响，以优化边缘设备上的部署效果。**

- **链接: [http://arxiv.org/pdf/2507.15151v1](http://arxiv.org/pdf/2507.15151v1)**

> **作者:** Sebastian A. Cruz Romero; Wilfredo E. Lugo Beauchamp
>
> **备注:** Accepted at International Symposium on Intelligent Computing & Networks 2025
>
> **摘要:** Anemia is a widespread global health issue, particularly among young children in low-resource settings. Traditional methods for anemia detection often require expensive equipment and expert knowledge, creating barriers to early and accurate diagnosis. To address these challenges, we explore the use of deep learning models for detecting anemia through conjunctival pallor, focusing on the CP-AnemiC dataset, which includes 710 images from children aged 6-59 months. The dataset is annotated with hemoglobin levels, gender, age and other demographic data, enabling the development of machine learning models for accurate anemia detection. We use the MobileNet architecture as a backbone, known for its efficiency in mobile and embedded vision applications, and fine-tune our model end-to-end using data augmentation techniques and a cross-validation strategy. Our model implementation achieved an accuracy of 0.9313, a precision of 0.9374, and an F1 score of 0.9773 demonstrating strong performance on the dataset. To optimize the model for deployment on edge devices, we performed post-training quantization, evaluating the impact of different bit-widths (FP32, FP16, INT8, and INT4) on model performance. Preliminary results suggest that while FP16 quantization maintains high accuracy (0.9250), precision (0.9370), and F1 Score (0.9377), more aggressive quantization (INT8 and INT4) leads to significant performance degradation. Overall, our study supports further exploration of quantization schemes and hardware optimizations to assess trade-offs between model size, inference time, and diagnostic accuracy in mobile healthcare applications.
>
---
#### [new 161] Personalized 4D Whole Heart Geometry Reconstruction from Cine MRI for Cardiac Digital Twins
- **分类: eess.IV; cs.CV**

- **简介: 该论文旨在通过弱监督学习模型，从多视角2D心脏电影MRI重建个性化4D全心几何模型，用于心脏数字孪生。它解决了现有心脏模型无法完整模拟四腔室电生理和机械活动的问题，实现了高时间分辨率的心脏参数自动提取。**

- **链接: [http://arxiv.org/pdf/2507.15203v1](http://arxiv.org/pdf/2507.15203v1)**

> **作者:** Xiaoyue Liu; Xicheng Sheng; Xiahai Zhuang; Vicente Grau; Mark YY Chan; Ching-Hui Sia; Lei Li
>
> **摘要:** Cardiac digital twins (CDTs) provide personalized in-silico cardiac representations and hold great potential for precision medicine in cardiology. However, whole-heart CDT models that simulate the full organ-scale electromechanics of all four heart chambers remain limited. In this work, we propose a weakly supervised learning model to reconstruct 4D (3D+t) heart mesh directly from multi-view 2D cardiac cine MRIs. This is achieved by learning a self-supervised mapping between cine MRIs and 4D cardiac meshes, enabling the generation of personalized heart models that closely correspond to input cine MRIs. The resulting 4D heart meshes can facilitate the automatic extraction of key cardiac variables, including ejection fraction and dynamic chamber volume changes with high temporal resolution. It demonstrates the feasibility of inferring personalized 4D heart models from cardiac MRIs, paving the way for an efficient CDT platform for precision medicine. The code will be publicly released once the manuscript is accepted.
>
---
#### [new 162] To Label or Not to Label: PALM -- A Predictive Model for Evaluating Sample Efficiency in Active Learning Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于主动学习评估任务，旨在解决传统评估方法忽略学习过程动态的问题。作者提出了PALM模型，通过四个参数描述主动学习性能，实现对学习曲线的预测与策略比较，验证其在多个数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.15381v1](http://arxiv.org/pdf/2507.15381v1)**

> **作者:** Julia Machnio; Mads Nielsen; Mostafa Mehdipour Ghazi
>
> **备注:** ICCV 2025
>
> **摘要:** Active learning (AL) seeks to reduce annotation costs by selecting the most informative samples for labeling, making it particularly valuable in resource-constrained settings. However, traditional evaluation methods, which focus solely on final accuracy, fail to capture the full dynamics of the learning process. To address this gap, we propose PALM (Performance Analysis of Active Learning Models), a unified and interpretable mathematical model that characterizes AL trajectories through four key parameters: achievable accuracy, coverage efficiency, early-stage performance, and scalability. PALM provides a predictive description of AL behavior from partial observations, enabling the estimation of future performance and facilitating principled comparisons across different strategies. We validate PALM through extensive experiments on CIFAR-10/100 and ImageNet-50/100/200, covering a wide range of AL methods and self-supervised embeddings. Our results demonstrate that PALM generalizes effectively across datasets, budgets, and strategies, accurately predicting full learning curves from limited labeled data. Importantly, PALM reveals crucial insights into learning efficiency, data space coverage, and the scalability of AL methods. By enabling the selection of cost-effective strategies and predicting performance under tight budget constraints, PALM lays the basis for more systematic, reproducible, and data-efficient evaluation of AL in both research and real-world applications. The code is available at: https://github.com/juliamachnio/PALM.
>
---
#### [new 163] GR-3 Technical Report
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出GR-3，一种大规模视觉-语言-动作模型，旨在构建通用机器人策略。它通过多阶段训练实现对新任务的快速适应，具备处理复杂操作和移动任务的能力。实验表明其性能优于现有方法，结合ByteMini机器人可实现多样化任务执行。**

- **链接: [http://arxiv.org/pdf/2507.15493v1](http://arxiv.org/pdf/2507.15493v1)**

> **作者:** Chilam Cheang; Sijin Chen; Zhongren Cui; Yingdong Hu; Liqun Huang; Tao Kong; Hang Li; Yifeng Li; Yuxiao Liu; Xiao Ma; Hao Niu; Wenxuan Ou; Wanli Peng; Zeyu Ren; Haixin Shi; Jiawen Tian; Hongtao Wu; Xin Xiao; Yuyang Xiao; Jiafeng Xu; Yichu Yang
>
> **备注:** Tech report. Authors are listed in alphabetical order. Project page: https://seed.bytedance.com/GR3/
>
> **摘要:** We report our recent progress towards building generalist robot policies, the development of GR-3. GR-3 is a large-scale vision-language-action (VLA) model. It showcases exceptional capabilities in generalizing to novel objects, environments, and instructions involving abstract concepts. Furthermore, it can be efficiently fine-tuned with minimal human trajectory data, enabling rapid and cost-effective adaptation to new settings. GR-3 also excels in handling long-horizon and dexterous tasks, including those requiring bi-manual manipulation and mobile movement, showcasing robust and reliable performance. These capabilities are achieved through a multi-faceted training recipe that includes co-training with web-scale vision-language data, efficient fine-tuning from human trajectory data collected via VR devices, and effective imitation learning with robot trajectory data. In addition, we introduce ByteMini, a versatile bi-manual mobile robot designed with exceptional flexibility and reliability, capable of accomplishing a wide range of tasks when integrated with GR-3. Through extensive real-world experiments, we show GR-3 surpasses the state-of-the-art baseline method, $\pi_0$, on a wide variety of challenging tasks. We hope GR-3 can serve as a step towards building generalist robots capable of assisting humans in daily life.
>
---
#### [new 164] MedSR-Impact: Transformer-Based Super-Resolution for Lung CT Segmentation, Radiomics, Classification, and Prognosis
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决低剂量CT分辨率不足影响诊断的问题。作者提出了TVSRN-V2模型，基于Transformer进行三维超分辨率重建，提升肺部CT图像质量，并验证其在分割、放射组学和预后预测中的效果，增强了模型在不同设备上的适应性和临床实用性。**

- **链接: [http://arxiv.org/pdf/2507.15340v1](http://arxiv.org/pdf/2507.15340v1)**

> **作者:** Marc Boubnovski Martell; Kristofer Linton-Reid; Mitchell Chen; Sumeet Hindocha; Benjamin Hunter; Marco A. Calzado; Richard Lee; Joram M. Posma; Eric O. Aboagye
>
> **摘要:** High-resolution volumetric computed tomography (CT) is essential for accurate diagnosis and treatment planning in thoracic diseases; however, it is limited by radiation dose and hardware costs. We present the Transformer Volumetric Super-Resolution Network (\textbf{TVSRN-V2}), a transformer-based super-resolution (SR) framework designed for practical deployment in clinical lung CT analysis. Built from scalable components, including Through-Plane Attention Blocks (TAB) and Swin Transformer V2 -- our model effectively reconstructs fine anatomical details in low-dose CT volumes and integrates seamlessly with downstream analysis pipelines. We evaluate its effectiveness on three critical lung cancer tasks -- lobe segmentation, radiomics, and prognosis -- across multiple clinical cohorts. To enhance robustness across variable acquisition protocols, we introduce pseudo-low-resolution augmentation, simulating scanner diversity without requiring private data. TVSRN-V2 demonstrates a significant improvement in segmentation accuracy (+4\% Dice), higher radiomic feature reproducibility, and enhanced predictive performance (+0.06 C-index and AUC). These results indicate that SR-driven recovery of structural detail significantly enhances clinical decision support, positioning TVSRN-V2 as a well-engineered, clinically viable system for dose-efficient imaging and quantitative analysis in real-world CT workflows.
>
---
#### [new 165] WebGuard: Building a Generalizable Guardrail for Web Agents
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于安全评估任务，旨在解决自主网页代理（Web Agents）可能执行有害操作的风险问题。作者构建了名为WebGuard的数据集，包含4,939个标注动作，涵盖22个领域，提出三级风险分类体系。通过评估发现当前大模型表现不足，进而尝试微调专用模型提升预测性能，取得显著改进，但仍未能达到高风险场景的可靠性要求。**

- **链接: [http://arxiv.org/pdf/2507.14293v1](http://arxiv.org/pdf/2507.14293v1)**

> **作者:** Boyuan Zheng; Zeyi Liao; Scott Salisbury; Zeyuan Liu; Michael Lin; Qinyuan Zheng; Zifan Wang; Xiang Deng; Dawn Song; Huan Sun; Yu Su
>
> **备注:** We publicly release WebGuard, along with its annotation tools and fine-tuned models, to facilitate open-source research on monitoring and safeguarding web agents. All resources are available at https://github.com/OSU-NLP-Group/WebGuard
>
> **摘要:** The rapid development of autonomous web agents powered by Large Language Models (LLMs), while greatly elevating efficiency, exposes the frontier risk of taking unintended or harmful actions. This situation underscores an urgent need for effective safety measures, akin to access controls for human users. To address this critical challenge, we introduce WebGuard, the first comprehensive dataset designed to support the assessment of web agent action risks and facilitate the development of guardrails for real-world online environments. In doing so, WebGuard specifically focuses on predicting the outcome of state-changing actions and contains 4,939 human-annotated actions from 193 websites across 22 diverse domains, including often-overlooked long-tail websites. These actions are categorized using a novel three-tier risk schema: SAFE, LOW, and HIGH. The dataset includes designated training and test splits to support evaluation under diverse generalization settings. Our initial evaluations reveal a concerning deficiency: even frontier LLMs achieve less than 60% accuracy in predicting action outcomes and less than 60% recall in lagging HIGH-risk actions, highlighting the risks of deploying current-generation agents without dedicated safeguards. We therefore investigate fine-tuning specialized guardrail models using WebGuard. We conduct comprehensive evaluations across multiple generalization settings and find that a fine-tuned Qwen2.5VL-7B model yields a substantial improvement in performance, boosting accuracy from 37% to 80% and HIGH-risk action recall from 20% to 76%. Despite these improvements, the performance still falls short of the reliability required for high-stakes deployment, where guardrails must approach near-perfect accuracy and recall.
>
---
#### [new 166] LOVO: Efficient Complex Object Query in Large-Scale Video Datasets
- **分类: cs.IR; cs.CV; cs.DB**

- **简介: 论文提出LOVO系统，旨在高效处理大规模视频数据中的复杂对象查询任务。该系统通过一次特征提取构建紧凑索引，利用向量数据库支持任意对象查询，并通过近似最近邻搜索和跨模态重排序提升检索效率与准确率。解决了现有方法在适应性与查询延迟上的不足。**

- **链接: [http://arxiv.org/pdf/2507.14301v1](http://arxiv.org/pdf/2507.14301v1)**

> **作者:** Yuxin Liu; Yuezhang Peng; Hefeng Zhou; Hongze Liu; Xinyu Lu; Jiong Lou; Chentao Wu; Wei Zhao; Jie Li
>
> **备注:** @inproceedings{liu2025lovo,title={LOVO: Efficient Complex Object Query in Large-Scale Video Datasets},author={Liu, Yuxin and Peng, Yuezhang and Zhou, Hefeng and Liu, Hongze and Lu, Xinyu and Lou, Jiong and Wu, Chentao and Zhao, Wei and Li, Jie},booktitle={2025 IEEE 41st International Conference on Data Engineering (ICDE)},pages={1938--1951},year={2025},organization={IEEE Computer Society}}
>
> **摘要:** The widespread deployment of cameras has led to an exponential increase in video data, creating vast opportunities for applications such as traffic management and crime surveillance. However, querying specific objects from large-scale video datasets presents challenges, including (1) processing massive and continuously growing data volumes, (2) supporting complex query requirements, and (3) ensuring low-latency execution. Existing video analysis methods struggle with either limited adaptability to unseen object classes or suffer from high query latency. In this paper, we present LOVO, a novel system designed to efficiently handle comp$\underline{L}$ex $\underline{O}$bject queries in large-scale $\underline{V}$ide$\underline{O}$ datasets. Agnostic to user queries, LOVO performs one-time feature extraction using pre-trained visual encoders, generating compact visual embeddings for key frames to build an efficient index. These visual embeddings, along with associated bounding boxes, are organized in an inverted multi-index structure within a vector database, which supports queries for any objects. During the query phase, LOVO transforms object queries to query embeddings and conducts fast approximate nearest-neighbor searches on the visual embeddings. Finally, a cross-modal rerank is performed to refine the results by fusing visual features with detailed textual features. Evaluation on real-world video datasets demonstrates that LOVO outperforms existing methods in handling complex queries, with near-optimal query accuracy and up to 85x lower search latency, while significantly reducing index construction costs. This system redefines the state-of-the-art object query approaches in video analysis, setting a new benchmark for complex object queries with a novel, scalable, and efficient approach that excels in dynamic environments.
>
---
#### [new 167] In-Depth and In-Breadth: Pre-training Multimodal Language Models Customized for Comprehensive Chart Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决现有图表理解模型泛化能力弱、数据对齐不足的问题。作者提出了ChartScope，通过合成多样化图表数据和采用Dual-Path训练策略，提升模型对图表及底层数据的理解。实验表明其在多种图表类型上表现优异，并发布了ChartDQA新基准。**

- **链接: [http://arxiv.org/pdf/2507.14298v1](http://arxiv.org/pdf/2507.14298v1)**

> **作者:** Wan-Cyuan Fan; Yen-Chun Chen; Mengchen Liu; Alexander Jacobson; Lu Yuan; Leonid Sigal
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2407.14506
>
> **摘要:** Recent methods for customizing Large Vision Language Models (LVLMs) for domain-specific tasks have shown promising results in scientific chart comprehension. However, existing approaches face two major limitations: First, they rely on paired data from only a few chart types, limiting generalization to wide range of chart types. Secondly, they lack targeted pre-training for chart-data alignment, which hampers the model's understanding of underlying data. In this paper, we introduce ChartScope, an LVLM optimized for in-depth chart comprehension across diverse chart types. We propose an efficient data generation pipeline that synthesizes paired data for a wide range of chart types, along with a novel Dual-Path training strategy that enabling the model to succinctly capture essential data details while preserving robust reasoning capabilities by incorporating reasoning over the underlying data. Lastly, we establish ChartDQA, a new benchmark for evaluating not only question-answering at different levels but also underlying data understanding. Experimental results demonstrate that ChartScope significantly enhances comprehension on a wide range of chart types. The code and data are available at https://davidhalladay.github.io/chartscope_demo.
>
---
#### [new 168] DeSamba: Decoupled Spectral Adaptive Framework for 3D Multi-Sequence MRI Lesion Classification
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决多序列MRI数据的3D病灶分类问题。作者提出DeSamba框架，通过解耦表示学习和频谱自适应融合，提升分类性能。实验表明其在两个数据集上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.15487v1](http://arxiv.org/pdf/2507.15487v1)**

> **作者:** Dezhen Wang; Sheng Miao; Rongxin Chai; Jiufa Cui
>
> **备注:** 7 figures, 3 tables, submitted to AAAI2026
>
> **摘要:** Magnetic Resonance Imaging (MRI) sequences provide rich spatial and frequency domain information, which is crucial for accurate lesion classification in medical imaging. However, effectively integrating multi-sequence MRI data for robust 3D lesion classification remains a challenge. In this paper, we propose DeSamba (Decoupled Spectral Adaptive Network and Mamba-Based Model), a novel framework designed to extract decoupled representations and adaptively fuse spatial and spectral features for lesion classification. DeSamba introduces a Decoupled Representation Learning Module (DRLM) that decouples features from different MRI sequences through self-reconstruction and cross-reconstruction, and a Spectral Adaptive Modulation Block (SAMB) within the proposed SAMNet, enabling dynamic fusion of spectral and spatial information based on lesion characteristics. We evaluate DeSamba on two clinically relevant 3D datasets. On a six-class spinal metastasis dataset (n=1,448), DeSamba achieves 62.10% Top-1 accuracy, 63.62% F1-score, 87.71% AUC, and 93.55% Top-3 accuracy on an external validation set (n=372), outperforming all state-of-the-art (SOTA) baselines. On a spondylitis dataset (n=251) involving a challenging binary classification task, DeSamba achieves 70.00%/64.52% accuracy and 74.75/73.88 AUC on internal and external validation sets, respectively. Ablation studies demonstrate that both DRLM and SAMB significantly contribute to overall performance, with over 10% relative improvement compared to the baseline. Our results highlight the potential of DeSamba as a generalizable and effective solution for 3D lesion classification in multi-sequence medical imaging.
>
---
#### [new 169] Diffusion Beats Autoregressive in Data-Constrained Settings
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 论文研究在数据受限场景下扩散模型与自回归（AR）模型的性能对比，发现扩散模型在计算资源充足但数据稀缺时显著优于AR模型。扩散模型通过隐式数据增强，更好地利用重复数据，实现更低验证损失和更优下游任务表现，并提出了扩散模型的新扩展规律和计算阈值公式。**

- **链接: [http://arxiv.org/pdf/2507.15857v1](http://arxiv.org/pdf/2507.15857v1)**

> **作者:** Mihir Prabhudesai; Menging Wu; Amir Zadeh; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project Webpage: https://diffusion-scaling.github.io
>
> **摘要:** Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: https://diffusion-scaling.github.io.
>
---
#### [new 170] A Study of Anatomical Priors for Deep Learning-Based Segmentation of Pheochromocytoma in Abdominal CT
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升腹部CT中嗜铬细胞瘤（PCC）的自动分割精度。为解决分割不准确影响病情评估的问题，研究引入基于解剖先验的多类标注策略，采用nnU-Net框架进行实验，最终发现结合肿瘤、肾脏和主动脉的标注策略效果最佳，提升了分割性能并支持临床评估。**

- **链接: [http://arxiv.org/pdf/2507.15193v1](http://arxiv.org/pdf/2507.15193v1)**

> **作者:** Tanjin Taher Toma; Tejas Sudharshan Mathai; Bikash Santra; Pritam Mukherjee; Jianfei Liu; Wesley Jong; Darwish Alabyad; Vivek Batheja; Abhishek Jha; Mayank Patel; Darko Pucar; Jayadira del Rivero; Karel Pacak; Ronald M. Summers
>
> **摘要:** Accurate segmentation of pheochromocytoma (PCC) in abdominal CT scans is essential for tumor burden estimation, prognosis, and treatment planning. It may also help infer genetic clusters, reducing reliance on expensive testing. This study systematically evaluates anatomical priors to identify configurations that improve deep learning-based PCC segmentation. We employed the nnU-Net framework to evaluate eleven annotation strategies for accurate 3D segmentation of pheochromocytoma, introducing a set of novel multi-class schemes based on organ-specific anatomical priors. These priors were derived from adjacent organs commonly surrounding adrenal tumors (e.g., liver, spleen, kidney, aorta, adrenal gland, and pancreas), and were compared against a broad body-region prior used in previous work. The framework was trained and tested on 105 contrast-enhanced CT scans from 91 patients at the NIH Clinical Center. Performance was measured using Dice Similarity Coefficient (DSC), Normalized Surface Distance (NSD), and instance-wise F1 score. Among all strategies, the Tumor + Kidney + Aorta (TKA) annotation achieved the highest segmentation accuracy, significantly outperforming the previously used Tumor + Body (TB) annotation across DSC (p = 0.0097), NSD (p = 0.0110), and F1 score (25.84% improvement at an IoU threshold of 0.5), measured on a 70-30 train-test split. The TKA model also showed superior tumor burden quantification (R^2 = 0.968) and strong segmentation across all genetic subtypes. In five-fold cross-validation, TKA consistently outperformed TB across IoU thresholds (0.1 to 0.5), reinforcing its robustness and generalizability. These findings highlight the value of incorporating relevant anatomical context in deep learning models to achieve precise PCC segmentation, supporting clinical assessment and longitudinal monitoring.
>
---
#### [new 171] Flow Equivariant Recurrent Neural Networks
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于序列建模任务，旨在解决现有循环神经网络（RNN）缺乏时间参数化对称性（流等变性）导致的泛化能力不足问题。作者提出流等变RNN，使隐藏状态随输入动态结构化变换，提升了训练速度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.14793v1](http://arxiv.org/pdf/2507.14793v1)**

> **作者:** T. Anderson Keller
>
> **摘要:** Data arrives at our senses as a continuous stream, smoothly transforming from one instant to the next. These smooth transformations can be viewed as continuous symmetries of the environment that we inhabit, defining equivalence relations between stimuli over time. In machine learning, neural network architectures that respect symmetries of their data are called equivariant and have provable benefits in terms of generalization ability and sample efficiency. To date, however, equivariance has been considered only for static transformations and feed-forward networks, limiting its applicability to sequence models, such as recurrent neural networks (RNNs), and corresponding time-parameterized sequence transformations. In this work, we extend equivariant network theory to this regime of `flows' -- one-parameter Lie subgroups capturing natural transformations over time, such as visual motion. We begin by showing that standard RNNs are generally not flow equivariant: their hidden states fail to transform in a geometrically structured manner for moving stimuli. We then show how flow equivariance can be introduced, and demonstrate that these models significantly outperform their non-equivariant counterparts in terms of training speed, length generalization, and velocity generalization, on both next step prediction and sequence classification. We present this work as a first step towards building sequence models that respect the time-parameterized symmetries which govern the world around us.
>
---
#### [new 172] Towards a Proactive Autoscaling Framework for Data Stream Processing at the Edge using GRU and Transfer Learning
- **分类: cs.DC; cs.CV; cs.LG; cs.PF**

- **简介: 该论文属于边缘计算中的自动扩展任务，旨在解决边缘流处理中资源分配不均的问题。通过使用GRU进行负载预测，并结合迁移学习和动态水平扩展模块，实现对资源的高效管理，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.14597v1](http://arxiv.org/pdf/2507.14597v1)**

> **作者:** Eugene Armah; Linda Amoako Bannning
>
> **摘要:** Processing data at high speeds is becoming increasingly critical as digital economies generate enormous data. The current paradigms for timely data processing are edge computing and data stream processing (DSP). Edge computing places resources closer to where data is generated, while stream processing analyzes the unbounded high-speed data in motion. However, edge stream processing faces rapid workload fluctuations, complicating resource provisioning. Inadequate resource allocation leads to bottlenecks, whereas excess allocation results in wastage. Existing reactive methods, such as threshold-based policies and queuing theory scale only after performance degrades, potentially violating SLAs. Although reinforcement learning (RL) offers a proactive approach through agents that learn optimal runtime adaptation policies, it requires extensive simulation. Furthermore, predictive machine learning models face online distribution and concept drift that minimize their accuracy. We propose a three-step solution to the proactive edge stream processing autoscaling problem. Firstly, a GRU neural network forecasts the upstream load using real-world and synthetic DSP datasets. Secondly, a transfer learning framework integrates the predictive model into an online stream processing system using the DTW algorithm and joint distribution adaptation to handle the disparities between offline and online domains. Finally, a horizontal autoscaling module dynamically adjusts the degree of operator parallelism, based on predicted load while considering edge resource constraints. The lightweight GRU model for load predictions recorded up to 1.3\% SMAPE value on a real-world data set. It outperformed CNN, ARIMA, and Prophet on the SMAPE and RMSE evaluation metrics, with lower training time than the computationally intensive RL models.
>
---
#### [new 173] PET Image Reconstruction Using Deep Diffusion Image Prior
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像重建任务，旨在解决PET图像重建中 tracer-specific对比度差异和计算成本高的问题。论文提出了一种基于扩散模型和解剖先验的PET图像重建方法，通过交替扩散采样和模型微调，并引入HQS算法提升效率，实现了跨tracer和扫描仪的高质量低剂量PET图像重建。**

- **链接: [http://arxiv.org/pdf/2507.15078v1](http://arxiv.org/pdf/2507.15078v1)**

> **作者:** Fumio Hashimoto; Kuang Gong
>
> **备注:** 11 pages, 11 figures
>
> **摘要:** Diffusion models have shown great promise in medical image denoising and reconstruction, but their application to Positron Emission Tomography (PET) imaging remains limited by tracer-specific contrast variability and high computational demands. In this work, we proposed an anatomical prior-guided PET image reconstruction method based on diffusion models, inspired by the deep diffusion image prior (DDIP) framework. The proposed method alternated between diffusion sampling and model fine-tuning guided by the PET sinogram, enabling the reconstruction of high-quality images from various PET tracers using a score function pretrained on a dataset of another tracer. To improve computational efficiency, the half-quadratic splitting (HQS) algorithm was adopted to decouple network optimization from iterative PET reconstruction. The proposed method was evaluated using one simulation and two clinical datasets. For the simulation study, a model pretrained on [$^{18}$F]FDG data was tested on amyloid-negative PET data to assess out-of-distribution (OOD) performance. For the clinical-data validation, ten low-dose [$^{18}$F]FDG datasets and one [$^{18}$F]Florbetapir dataset were tested on a model pretrained on data from another tracer. Experiment results show that the proposed PET reconstruction method can generalize robustly across tracer distributions and scanner types, providing an efficient and versatile reconstruction framework for low-dose PET imaging.
>
---
#### [new 174] Real-Time Scene Reconstruction using Light Field Probes
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于计算机图形学任务，旨在解决大规模场景的实时真实感重建问题。现有方法在场景规模、质量与渲染速度间难以平衡。论文提出一种基于光场探针的隐式场景表示方法，无需显式几何数据，通过多尺度隐式表示和探针数据结构实现高效复杂场景重建，兼顾渲染质量与效率，适用于VR/AR应用。**

- **链接: [http://arxiv.org/pdf/2507.14624v1](http://arxiv.org/pdf/2507.14624v1)**

> **作者:** Yaru Liu; Derek Nowrouzezahri; Morgan Mcguire
>
> **摘要:** Reconstructing photo-realistic large-scale scenes from images, for example at city scale, is a long-standing problem in computer graphics. Neural rendering is an emerging technique that enables photo-realistic image synthesis from previously unobserved viewpoints; however, state-of-the-art neural rendering methods have difficulty efficiently rendering a high complex large-scale scene because these methods typically trade scene size, fidelity, and rendering speed for quality. The other stream of techniques utilizes scene geometries for reconstruction. But the cost of building and maintaining a large set of geometry data increases as scene size grows. Our work explores novel view synthesis methods that efficiently reconstruct complex scenes without explicit use of scene geometries. Specifically, given sparse images of the scene (captured from the real world), we reconstruct intermediate, multi-scale, implicit representations of scene geometries. In this way, our method avoids explicitly relying on scene geometry, significantly reducing the computational cost of maintaining large 3D data. Unlike current methods, we reconstruct the scene using a probe data structure. Probe data hold highly accurate depth information of dense data points, enabling the reconstruction of highly complex scenes. By reconstructing the scene using probe data, the rendering cost is independent of the complexity of the scene. As such, our approach combines geometry reconstruction and novel view synthesis. Moreover, when rendering large-scale scenes, compressing and streaming probe data is more efficient than using explicit scene geometry. Therefore, our neural representation approach can potentially be applied to virtual reality (VR) and augmented reality (AR) applications.
>
---
#### [new 175] Look, Focus, Act: Efficient and Robust Robot Learning via Human Gaze and Foveated Vision Transformers
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人视觉任务，旨在解决机器人视觉处理效率低、注意力不集中问题。受人类主动视觉启发，作者提出利用注视信息与渐进式视觉Transformer，构建主动视觉机器人系统，提升处理效率与精度。**

- **链接: [http://arxiv.org/pdf/2507.15833v1](http://arxiv.org/pdf/2507.15833v1)**

> **作者:** Ian Chuang; Andrew Lee; Dechen Gao; Jinyu Zou; Iman Soltani
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** Human vision is a highly active process driven by gaze, which directs attention and fixation to task-relevant regions and dramatically reduces visual processing. In contrast, robot learning systems typically rely on passive, uniform processing of raw camera images. In this work, we explore how incorporating human-like active gaze into robotic policies can enhance both efficiency and performance. We build on recent advances in foveated image processing and apply them to an Active Vision robot system that emulates both human head movement and eye tracking. Extending prior work on the AV-ALOHA robot simulation platform, we introduce a framework for simultaneously collecting eye-tracking data and robot demonstrations from a human operator as well as a simulation benchmark and dataset for training robot policies that incorporate human gaze. Given the widespread use of Vision Transformers (ViTs) in robot learning, we integrate gaze information into ViTs using a foveated patch tokenization scheme inspired by recent work in image segmentation. Compared to uniform patch tokenization, this significantly reduces the number of tokens-and thus computation-without sacrificing visual fidelity near regions of interest. We also explore two approaches to gaze imitation and prediction from human data. The first is a two-stage model that predicts gaze to guide foveation and action; the second integrates gaze into the action space, allowing the policy to jointly predict gaze and actions end-to-end. Our results show that our method for foveated robot vision not only drastically reduces computational overhead, but also improves performance for high precision tasks and robustness to unseen distractors. Together, these findings suggest that human-inspired visual processing offers a useful inductive bias for robotic vision systems. https://ian-chuang.github.io/gaze-av-aloha/
>
---
#### [new 176] Blended Point Cloud Diffusion for Localized Text-guided Shape Editing
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D形状编辑任务，旨在解决局部文本引导编辑时全局一致性难以保持的问题。作者提出了一种基于点云扩散模型的编辑框架，结合局部结构引导和推理阶段坐标混合算法，实现细粒度编辑并保持形状整体一致性，无需复杂逆过程。**

- **链接: [http://arxiv.org/pdf/2507.15399v1](http://arxiv.org/pdf/2507.15399v1)**

> **作者:** Etai Sella; Noam Atia; Ron Mokady; Hadar Averbuch-Elor
>
> **备注:** Accepted to ICCV 2025. Project Page: https://tau-vailab.github.io/BlendedPC/
>
> **摘要:** Natural language offers a highly intuitive interface for enabling localized fine-grained edits of 3D shapes. However, prior works face challenges in preserving global coherence while locally modifying the input 3D shape. In this work, we introduce an inpainting-based framework for editing shapes represented as point clouds. Our approach leverages foundation 3D diffusion models for achieving localized shape edits, adding structural guidance in the form of a partial conditional shape, ensuring that other regions correctly preserve the shape's identity. Furthermore, to encourage identity preservation also within the local edited region, we propose an inference-time coordinate blending algorithm which balances reconstruction of the full shape with inpainting at a progression of noise levels during the inference process. Our coordinate blending algorithm seamlessly blends the original shape with its edited version, enabling a fine-grained editing of 3D shapes, all while circumventing the need for computationally expensive and often inaccurate inversion. Extensive experiments show that our method outperforms alternative techniques across a wide range of metrics that evaluate both fidelity to the original shape and also adherence to the textual description.
>
---
#### [new 177] Design of an Edge-based Portable EHR System for Anemia Screening in Remote Health Applications
- **分类: cs.ET; cs.AI; cs.CV; cs.CY; cs.LG; cs.SE**

- **简介: 该论文设计了一种基于边缘计算的便携式电子健康记录系统，用于偏远地区贫血筛查。任务是解决医疗系统在离线、资源有限环境中互操作性差、依赖昂贵基础设施的问题。工作包括开发支持离线操作、数据加密、模块化诊断集成的系统，并集成基于甲床分析的非侵入性贫血检测模型，优化模型性能与隐私合规，实现低成本、可扩展的医疗支持方案。**

- **链接: [http://arxiv.org/pdf/2507.15146v1](http://arxiv.org/pdf/2507.15146v1)**

> **作者:** Sebastian A. Cruz Romero; Misael J. Mercado Hernandez; Samir Y. Ali Rivera; Jorge A. Santiago Fernandez; Wilfredo E. Lugo Beauchamp
>
> **备注:** Accepted at IEEE Global Humanitarian Technology Conference 2025
>
> **摘要:** The design of medical systems for remote, resource-limited environments faces persistent challenges due to poor interoperability, lack of offline support, and dependency on costly infrastructure. Many existing digital health solutions neglect these constraints, limiting their effectiveness for frontline health workers in underserved regions. This paper presents a portable, edge-enabled Electronic Health Record platform optimized for offline-first operation, secure patient data management, and modular diagnostic integration. Running on small-form factor embedded devices, it provides AES-256 encrypted local storage with optional cloud synchronization for interoperability. As a use case, we integrated a non-invasive anemia screening module leveraging fingernail pallor analysis. Trained on 250 patient cases (27\% anemia prevalence) with KDE-balanced data, the Random Forest model achieved a test RMSE of 1.969 g/dL and MAE of 1.490 g/dL. A severity-based model reached 79.2\% sensitivity. To optimize performance, a YOLOv8n-based nail bed detector was quantized to INT8, reducing inference latency from 46.96 ms to 21.50 ms while maintaining mAP@0.5 at 0.995. The system emphasizes low-cost deployment, modularity, and data privacy compliance (HIPAA/GDPR), addressing critical barriers to digital health adoption in disconnected settings. Our work demonstrates a scalable approach to enhance portable health information systems and support frontline healthcare in underserved regions.
>
---
#### [new 178] Generative Distribution Distillation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于知识蒸馏任务，旨在解决高维优化和缺乏语义监督的问题。论文提出Generative Distribution Distillation（GenDD）框架，引入Split Tokenization和Distribution Contraction技术，实现稳定有效的无监督及有监督知识蒸馏，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2507.14503v1](http://arxiv.org/pdf/2507.14503v1)**

> **作者:** Jiequan Cui; Beier Zhu; Qingshan Xu; Xiaogang Xu; Pengguang Chen; Xiaojuan Qi; Bei Yu; Hanwang Zhang; Richang Hong
>
> **备注:** Technique report
>
> **摘要:** In this paper, we formulate the knowledge distillation (KD) as a conditional generative problem and propose the \textit{Generative Distribution Distillation (GenDD)} framework. A naive \textit{GenDD} baseline encounters two major challenges: the curse of high-dimensional optimization and the lack of semantic supervision from labels. To address these issues, we introduce a \textit{Split Tokenization} strategy, achieving stable and effective unsupervised KD. Additionally, we develop the \textit{Distribution Contraction} technique to integrate label supervision into the reconstruction objective. Our theoretical proof demonstrates that \textit{GenDD} with \textit{Distribution Contraction} serves as a gradient-level surrogate for multi-task learning, realizing efficient supervised training without explicit classification loss on multi-step sampling image representations. To evaluate the effectiveness of our method, we conduct experiments on balanced, imbalanced, and unlabeled data. Experimental results show that \textit{GenDD} performs competitively in the unsupervised setting, significantly surpassing KL baseline by \textbf{16.29\%} on ImageNet validation set. With label supervision, our ResNet-50 achieves \textbf{82.28\%} top-1 accuracy on ImageNet in 600 epochs training, establishing a new state-of-the-art.
>
---
#### [new 179] Uncertainty-aware Probabilistic 3D Human Motion Forecasting via Invertible Networks
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D人体运动预测任务，旨在解决预测中的不确定性建模问题。现有方法难以量化预测置信度，影响安全关键场景应用。论文提出ProbHMI，利用可逆网络在解耦隐空间中建模姿态，显式预测未来隐分布，实现有效不确定性估计，提升风险感知决策能力。**

- **链接: [http://arxiv.org/pdf/2507.14694v1](http://arxiv.org/pdf/2507.14694v1)**

> **作者:** Yue Ma; Kanglei Zhou; Fuyang Yu; Frederick W. B. Li; Xiaohui Liang
>
> **摘要:** 3D human motion forecasting aims to enable autonomous applications. Estimating uncertainty for each prediction (i.e., confidence based on probability density or quantile) is essential for safety-critical contexts like human-robot collaboration to minimize risks. However, existing diverse motion forecasting approaches struggle with uncertainty quantification due to implicit probabilistic representations hindering uncertainty modeling. We propose ProbHMI, which introduces invertible networks to parameterize poses in a disentangled latent space, enabling probabilistic dynamics modeling. A forecasting module then explicitly predicts future latent distributions, allowing effective uncertainty quantification. Evaluated on benchmarks, ProbHMI achieves strong performance for both deterministic and diverse prediction while validating uncertainty calibration, critical for risk-aware decision making.
>
---
#### [new 180] Self-Supervised Distillation of Legacy Rule-Based Methods for Enhanced EEG-Based Decision-Making
- **分类: cs.CE; cs.CV**

- **简介: 该论文属于医学信号处理任务，旨在提升癫痫治疗中颅内脑电图（iEEG）高频振荡（HFOs）的自动检测精度。传统基于规则的方法精度低，人工标注困难。作者提出SS2LD框架，结合变分自编码器和弱监督学习，利用已有检测结果进行自监督训练，优化HFO分类，减少误报，实现高效精准的病理HFO识别。**

- **链接: [http://arxiv.org/pdf/2507.14542v1](http://arxiv.org/pdf/2507.14542v1)**

> **作者:** Yipeng Zhang; Yuanyi Ding; Chenda Duan; Atsuro Daida; Hiroki Nariai; Vwani Roychowdhury
>
> **摘要:** High-frequency oscillations (HFOs) in intracranial Electroencephalography (iEEG) are critical biomarkers for localizing the epileptogenic zone in epilepsy treatment. However, traditional rule-based detectors for HFOs suffer from unsatisfactory precision, producing false positives that require time-consuming manual review. Supervised machine learning approaches have been used to classify the detection results, yet they typically depend on labeled datasets, which are difficult to acquire due to the need for specialized expertise. Moreover, accurate labeling of HFOs is challenging due to low inter-rater reliability and inconsistent annotation practices across institutions. The lack of a clear consensus on what constitutes a pathological HFO further challenges supervised refinement approaches. To address this, we leverage the insight that legacy detectors reliably capture clinically relevant signals despite their relatively high false positive rates. We thus propose the Self-Supervised to Label Discovery (SS2LD) framework to refine the large set of candidate events generated by legacy detectors into a precise set of pathological HFOs. SS2LD employs a variational autoencoder (VAE) for morphological pre-training to learn meaningful latent representation of the detected events. These representations are clustered to derive weak supervision for pathological events. A classifier then uses this supervision to refine detection boundaries, trained on real and VAE-augmented data. Evaluated on large multi-institutional interictal iEEG datasets, SS2LD outperforms state-of-the-art methods. SS2LD offers a scalable, label-efficient, and clinically effective strategy to identify pathological HFOs using legacy detectors.
>
---
#### [new 181] Personalized 3D Myocardial Infarct Geometry Reconstruction from Cine MRI with Explicit Cardiac Motion Modeling
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决无需对比剂的心肌梗死（MI）三维几何重建问题。现有方法依赖对比增强MRI，存在副作用且分辨率低。作者提出新框架，通过4D双心室网格重建和运动模式建模，实现基于心脏运动的梗死区域自动识别，验证了无对比剂、高精度3D梗死重建的可行性。**

- **链接: [http://arxiv.org/pdf/2507.15194v1](http://arxiv.org/pdf/2507.15194v1)**

> **作者:** Yilin Lyu; Fan Yang; Xiaoyue Liu; Zichen Jiang; Joshua Dillon; Debbie Zhao; Martyn Nash; Charlene Mauger; Alistair Young; Ching-Hui Sia; Mark YY Chan; Lei Li
>
> **备注:** 11 pages
>
> **摘要:** Accurate representation of myocardial infarct geometry is crucial for patient-specific cardiac modeling in MI patients. While Late gadolinium enhancement (LGE) MRI is the clinical gold standard for infarct detection, it requires contrast agents, introducing side effects and patient discomfort. Moreover, infarct reconstruction from LGE often relies on sparsely sampled 2D slices, limiting spatial resolution and accuracy. In this work, we propose a novel framework for automatically reconstructing high-fidelity 3D myocardial infarct geometry from 2D clinically standard cine MRI, eliminating the need for contrast agents. Specifically, we first reconstruct the 4D biventricular mesh from multi-view cine MRIs via an automatic deep shape fitting model, biv-me. Then, we design a infarction reconstruction model, CMotion2Infarct-Net, to explicitly utilize the motion patterns within this dynamic geometry to localize infarct regions. Evaluated on 205 cine MRI scans from 126 MI patients, our method shows reasonable agreement with manual delineation. This study demonstrates the feasibility of contrast-free, cardiac motion-driven 3D infarct reconstruction, paving the way for efficient digital twin of MI.
>
---
#### [new 182] Classification of Histopathology Slides with Persistence Homology Convolutions
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决传统卷积神经网络在分析组织病理学图像时丢失拓扑信息的问题。作者提出了一种新的方法——持久同调卷积，用于提取局部拓扑特征。实验表明，该方法优于传统模型，且对超参数更鲁棒，能更好地捕捉图像的几何结构信息。**

- **链接: [http://arxiv.org/pdf/2507.14378v1](http://arxiv.org/pdf/2507.14378v1)**

> **作者:** Shrunal Pothagoni; Benjamin Schweinhart
>
> **摘要:** Convolutional neural networks (CNNs) are a standard tool for computer vision tasks such as image classification. However, typical model architectures may result in the loss of topological information. In specific domains such as histopathology, topology is an important descriptor that can be used to distinguish between disease-indicating tissue by analyzing the shape characteristics of cells. Current literature suggests that reintroducing topological information using persistent homology can improve medical diagnostics; however, previous methods utilize global topological summaries which do not contain information about the locality of topological features. To address this gap, we present a novel method that generates local persistent homology-based data using a modified version of the convolution operator called Persistent Homology Convolutions. This method captures information about the locality and translation invariance of topological features. We perform a comparative study using various representations of histopathology slides and find that models trained with persistent homology convolutions outperform conventionally trained models and are less sensitive to hyperparameters. These results indicate that persistent homology convolutions extract meaningful geometric information from the histopathology slides.
>
---
#### [new 183] A Steel Surface Defect Detection Method Based on Lightweight Convolution Optimization
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于工业检测任务，旨在解决钢铁表面多尺度缺陷识别精度低的问题。通过改进YOLOv9s模型，引入C3Ghost、SCConv模块及CARAFE上采样算子，优化特征提取与还原，提升小目标缺陷检测效果。**

- **链接: [http://arxiv.org/pdf/2507.15476v1](http://arxiv.org/pdf/2507.15476v1)**

> **作者:** Cong Chen; Ming Chen; Hoileong Lee; Yan Li; Jiyang Yu
>
> **摘要:** Surface defect detection of steel, especially the recognition of multi-scale defects, has always been a major challenge in industrial manufacturing. Steel surfaces not only have defects of various sizes and shapes, which limit the accuracy of traditional image processing and detection methods in complex environments. However, traditional defect detection methods face issues of insufficient accuracy and high miss-detection rates when dealing with small target defects. To address this issue, this study proposes a detection framework based on deep learning, specifically YOLOv9s, combined with the C3Ghost module, SCConv module, and CARAFE upsampling operator, to improve detection accuracy and model performance. First, the SCConv module is used to reduce feature redundancy and optimize feature representation by reconstructing the spatial and channel dimensions. Second, the C3Ghost module is introduced to enhance the model's feature extraction ability by reducing redundant computations and parameter volume, thereby improving model efficiency. Finally, the CARAFE upsampling operator, which can more finely reorganize feature maps in a content-aware manner, optimizes the upsampling process and ensures detailed restoration of high-resolution defect regions. Experimental results demonstrate that the proposed model achieves higher accuracy and robustness in steel surface defect detection tasks compared to other methods, effectively addressing defect detection problems.
>
---
#### [new 184] NuSeC: A Dataset for Nuclei Segmentation in Breast Cancer Histopathology Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决乳腺癌组织切片中细胞核分割问题。作者构建了NuSeC数据集，包含100张图像，分为训练集和测试集，用于评估分割方法性能。**

- **链接: [http://arxiv.org/pdf/2507.14272v1](http://arxiv.org/pdf/2507.14272v1)**

> **作者:** Refik Samet; Nooshin Nemati; Emrah Hancer; Serpil Sak; Bilge Ayca Kirmizi
>
> **摘要:** The NuSeC dataset is created by selecting 4 images with the size of 1024*1024 pixels from the slides of each patient among 25 patients. Therefore, there are a total of 100 images in the NuSeC dataset. To carry out a consistent comparative analysis between the methods that will be developed using the NuSeC dataset by the researchers in the future, we divide the NuSeC dataset 75% as the training set and 25% as the testing set. In detail, an image is randomly selected from 4 images of each patient among 25 patients to build the testing set, and then the remaining images are reserved for the training set. While the training set includes 75 images with around 30000 nuclei structures, the testing set includes 25 images with around 6000 nuclei structures.
>
---
#### [new 185] ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting
- **分类: cs.GR; cs.AI; cs.CV; cs.HC**

- **简介: 该论文属于三维场景重建与语义理解任务，旨在解决现有高斯泼溅技术缺乏语义感知的问题。作者提出了ObjectGS，通过建模每个物体为带有语义ID的局部锚点，实现物体级别的重建与语义分割。方法在保持高质量重建的同时，提升了物体级理解和应用兼容性。**

- **链接: [http://arxiv.org/pdf/2507.15454v1](http://arxiv.org/pdf/2507.15454v1)**

> **作者:** Ruijie Zhu; Mulin Yu; Linning Xu; Lihan Jiang; Yixuan Li; Tianzhu Zhang; Jiangmiao Pang; Bo Dai
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** 3D Gaussian Splatting is renowned for its high-fidelity reconstructions and real-time novel view synthesis, yet its lack of semantic understanding limits object-level perception. In this work, we propose ObjectGS, an object-aware framework that unifies 3D scene reconstruction with semantic understanding. Instead of treating the scene as a unified whole, ObjectGS models individual objects as local anchors that generate neural Gaussians and share object IDs, enabling precise object-level reconstruction. During training, we dynamically grow or prune these anchors and optimize their features, while a one-hot ID encoding with a classification loss enforces clear semantic constraints. We show through extensive experiments that ObjectGS not only outperforms state-of-the-art methods on open-vocabulary and panoptic segmentation tasks, but also integrates seamlessly with applications like mesh extraction and scene editing. Project page: https://ruijiezhu94.github.io/ObjectGS_page
>
---
#### [new 186] InsightX Agent: An LMM-based Agentic Framework with Integrated Tools for Reliable X-ray NDT Analysis
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于工业无损检测任务，旨在解决现有X射线检测方法缺乏可解释性、交互性与自我评估能力的问题。作者提出InsightX Agent，基于大多媒体模型的智能框架，结合SDMSD检测器与EGR工具，实现高效、可靠且可解释的X射线缺陷分析。**

- **链接: [http://arxiv.org/pdf/2507.14899v1](http://arxiv.org/pdf/2507.14899v1)**

> **作者:** Jiale Liu; Huan Wang; Yue Zhang; Xiaoyu Luo; Jiaxiang Hu; Zhiliang Liu; Min Xie
>
> **摘要:** Non-destructive testing (NDT), particularly X-ray inspection, is vital for industrial quality assurance, yet existing deep-learning-based approaches often lack interactivity, interpretability, and the capacity for critical self-assessment, limiting their reliability and operator trust. To address these shortcomings, this paper proposes InsightX Agent, a novel LMM-based agentic framework designed to deliver reliable, interpretable, and interactive X-ray NDT analysis. Unlike typical sequential pipelines, InsightX Agent positions a Large Multimodal Model (LMM) as a central orchestrator, coordinating between the Sparse Deformable Multi-Scale Detector (SDMSD) and the Evidence-Grounded Reflection (EGR) tool. The SDMSD generates dense defect region proposals for multi-scale feature maps and sparsifies them through Non-Maximum Suppression (NMS), optimizing detection of small, dense targets in X-ray images while maintaining computational efficiency. The EGR tool guides the LMM agent through a chain-of-thought-inspired review process, incorporating context assessment, individual defect analysis, false positive elimination, confidence recalibration and quality assurance to validate and refine the SDMSD's initial proposals. By strategically employing and intelligently using tools, InsightX Agent moves beyond passive data processing to active reasoning, enhancing diagnostic reliability and providing interpretations that integrate diverse information sources. Experimental evaluations on the GDXray+ dataset demonstrate that InsightX Agent not only achieves a high object detection F1-score of 96.35% but also offers significantly improved interpretability and trustworthiness in its analyses, highlighting the transformative potential of agentic LLM frameworks for industrial inspection tasks.
>
---
#### [new 187] Hyper-spectral Unmixing algorithms for remote compositional surface mapping: a review of the state of the art
- **分类: astro-ph.IM; astro-ph.EP; cs.CV**

- **简介: 该论文属于遥感图像分析任务，旨在解决通过高光谱图像推断地表覆盖材料类型及其丰度和空间分布的问题。论文综述了当前主流的高光谱解混算法，比较了其性能，分析了常用公开数据集，并指出了当前存在的挑战和未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.14260v1](http://arxiv.org/pdf/2507.14260v1)**

> **作者:** Alfredo Gimenez Zapiola; Andrea Boselli; Alessandra Menafoglio; Simone Vantini
>
> **摘要:** This work concerns a detailed review of data analysis methods used for remotely sensed images of large areas of the Earth and of other solid astronomical objects. In detail, it focuses on the problem of inferring the materials that cover the surfaces captured by hyper-spectral images and estimating their abundances and spatial distributions within the region. The most successful and relevant hyper-spectral unmixing methods are reported as well as compared, as an addition to analysing the most recent methodologies. The most important public data-sets in this setting, which are vastly used in the testing and validation of the former, are also systematically explored. Finally, open problems are spotlighted and concrete recommendations for future research are provided.
>
---
#### [new 188] Gaussian Splatting with Discretized SDF for Relightable Assets
- **分类: cs.GR; cs.CV**

- **简介: 论文属于逆渲染任务，旨在提升高斯点阵的几何表达与光照分离能力。现有方法用连续SDF正则化几何，但增加内存和训练复杂度。本文提出离散SDF，编码于各高斯点内，并通过一致性损失确保几何一致性。该方法在不增加内存的情况下提升重光照质量，优于现有高斯逆渲染方法。**

- **链接: [http://arxiv.org/pdf/2507.15629v1](http://arxiv.org/pdf/2507.15629v1)**

> **作者:** Zuo-Liang Zhu; Jian Yang; Beibei Wang
>
> **摘要:** 3D Gaussian splatting (3DGS) has shown its detailed expressive ability and highly efficient rendering speed in the novel view synthesis (NVS) task. The application to inverse rendering still faces several challenges, as the discrete nature of Gaussian primitives makes it difficult to apply geometry constraints. Recent works introduce the signed distance field (SDF) as an extra continuous representation to regularize the geometry defined by Gaussian primitives. It improves the decomposition quality, at the cost of increasing memory usage and complicating training. Unlike these works, we introduce a discretized SDF to represent the continuous SDF in a discrete manner by encoding it within each Gaussian using a sampled value. This approach allows us to link the SDF with the Gaussian opacity through an SDF-to-opacity transformation, enabling rendering the SDF via splatting and avoiding the computational cost of ray marching.The key challenge is to regularize the discrete samples to be consistent with the underlying SDF, as the discrete representation can hardly apply the gradient-based constraints (\eg Eikonal loss). For this, we project Gaussians onto the zero-level set of SDF and enforce alignment with the surface from splatting, namely a projection-based consistency loss. Thanks to the discretized SDF, our method achieves higher relighting quality, while requiring no extra memory beyond GS and avoiding complex manually designed optimization. The experiments reveal that our method outperforms existing Gaussian-based inverse rendering methods. Our code is available at https://github.com/NK-CS-ZZL/DiscretizedSDF.
>
---
#### [new 189] Towards Geometric and Textural Consistency 3D Scene Generation via Single Image-guided Model Generation and Layout Optimization
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决单张RGB图像生成几何与纹理一致的3D场景问题。论文提出一个三阶段框架，通过图像实例分割、伪立体视角构建与模型选择、以及布局优化，实现高质量、场景一致性的3D重建。**

- **链接: [http://arxiv.org/pdf/2507.14841v1](http://arxiv.org/pdf/2507.14841v1)**

> **作者:** Xiang Tang; Ruotong Li; Xiaopeng Fan
>
> **备注:** 15 pages, 8 figures, Project page: https://xdlbw.github.io/sing3d/
>
> **摘要:** In recent years, 3D generation has made great strides in both academia and industry. However, generating 3D scenes from a single RGB image remains a significant challenge, as current approaches often struggle to ensure both object generation quality and scene coherence in multi-object scenarios. To overcome these limitations, we propose a novel three-stage framework for 3D scene generation with explicit geometric representations and high-quality textural details via single image-guided model generation and spatial layout optimization. Our method begins with an image instance segmentation and inpainting phase, which recovers missing details of occluded objects in the input images, thereby achieving complete generation of foreground 3D assets. Subsequently, our approach captures the spatial geometry of reference image by constructing pseudo-stereo viewpoint for camera parameter estimation and scene depth inference, while employing a model selection strategy to ensure optimal alignment between the 3D assets generated in the previous step and the input. Finally, through model parameterization and minimization of the Chamfer distance between point clouds in 3D and 2D space, our approach optimizes layout parameters to produce an explicit 3D scene representation that maintains precise alignment with input guidance image. Extensive experiments on multi-object scene image sets have demonstrated that our approach not only outperforms state-of-the-art methods in terms of geometric accuracy and texture fidelity of individual generated 3D models, but also has significant advantages in scene layout synthesis.
>
---
## 更新

#### [replaced 001] Video-based Exercise Classification and Activated Muscle Group Prediction with Hybrid X3D-SlowFast Network
- **分类: cs.CV; cs.LG; I.2.10; I.4.8**

- **链接: [http://arxiv.org/pdf/2406.06703v2](http://arxiv.org/pdf/2406.06703v2)**

> **作者:** Manvik Pasula; Pramit Saha
>
> **备注:** 13 pages, 1 figure, submitted to Nature Scientific Reports
>
> **摘要:** This paper introduces a simple yet effective strategy for exercise classification and muscle group activation prediction (MGAP). These tasks have significant implications for personal fitness, facilitating more affordable, accessible, safer, and simpler exercise routines. This is particularly relevant for novices and individuals with disabilities. Previous research in the field is mostly dominated by the reliance on mounted sensors and a limited scope of exercises, reducing practicality for everyday use. Furthermore, existing MGAP methodologies suffer from a similar dependency on sensors and a restricted range of muscle groups, often excluding strength training exercises, which are pivotal for a comprehensive fitness regimen. Addressing these limitations, our research employs a video-based deep learning framework that encompasses a broad spectrum of exercises and muscle groups, including those vital for strength training. Utilizing the "Workout/Exercises Video" dataset, our approach integrates the X3D and SlowFast video activity recognition models in an effective way to enhance exercise classification and MGAP performance. Our findings demonstrate that this hybrid method, obtained via weighted ensemble, outperforms existing baseline models in accuracy. Pretrained models play a crucial role in enhancing overall performance, with optimal channel reduction values for the SlowFast model identified near 10. Through an ablation study that explores fine-tuning, we further elucidate the interrelation between the two tasks. Our composite model, a weighted-average ensemble of X3D and SlowFast, sets a new benchmark in both exercise classification and MGAP across all evaluated categories, offering a robust solution to the limitations of previous approaches.
>
---
#### [replaced 002] RACR-MIL: Rank-aware contextual reasoning for weakly supervised grading of squamous cell carcinoma using whole slide images
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2308.15618v2](http://arxiv.org/pdf/2308.15618v2)**

> **作者:** Anirudh Choudhary; Mosbah Aouad; Krishnakant Saboo; Angelina Hwang; Jacob Kechter; Blake Bordeaux; Puneet Bhullar; David DiCaudo; Steven Nelson; Nneka Comfere; Emma Johnson; Olayemi Sokumbi; Jason Sluzevich; Leah Swanson; Dennis Murphree; Aaron Mangold; Ravishankar Iyer
>
> **备注:** 17 pages main text, 2 page references, 2 page appendix; under submission
>
> **摘要:** Squamous cell carcinoma (SCC) is the most common cancer subtype, with an increasing incidence and a significant impact on cancer-related mortality. SCC grading using whole slide images is inherently challenging due to the lack of a reliable protocol and substantial tissue heterogeneity. We propose RACR-MIL, the first weakly-supervised SCC grading approach achieving robust generalization across multiple anatomies (skin, head and neck, lung). RACR-MIL is an attention-based multiple-instance learning framework that enhances grade-relevant contextual representation learning and addresses tumor heterogeneity through two key innovations: (1) a hybrid WSI graph that captures both local tissue context and non-local phenotypical dependencies between tumor regions, and (2) a rank-ordering constraint in the attention mechanism that consistently prioritizes higher-grade tumor regions, aligning with pathologists diagnostic process. Our model achieves state-of-the-art performance across multiple SCC datasets, achieving 3-9% higher grading accuracy, resilience to class imbalance, and up to 16% improved tumor localization. In a pilot study, pathologists reported that RACR-MIL improved grading efficiency in 60% of cases, underscoring its potential as a clinically viable cancer diagnosis and grading assistant.
>
---
#### [replaced 003] Beyond RGB: Adaptive Parallel Processing for RAW Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13163v2](http://arxiv.org/pdf/2503.13163v2)**

> **作者:** Shani Gamrian; Hila Barel; Feiran Li; Masakazu Yoshimura; Daisuke Iso
>
> **摘要:** Object detection models are typically applied to standard RGB images processed through Image Signal Processing (ISP) pipelines, which are designed to enhance sensor-captured RAW images for human vision. However, these ISP functions can lead to a loss of critical information that may be essential in optimizing for computer vision tasks, such as object detection. In this work, we introduce Raw Adaptation Module (RAM), a module designed to replace the traditional ISP, with parameters optimized specifically for RAW object detection. Inspired by the parallel processing mechanisms of the human visual system, RAM departs from existing learned ISP methods by applying multiple ISP functions in parallel rather than sequentially, allowing for a more comprehensive capture of image features. These processed representations are then fused in a specialized module, which dynamically integrates and optimizes the information for the target task. This novel approach not only leverages the full potential of RAW sensor data but also enables task-specific pre-processing, resulting in superior object detection performance. Our approach outperforms RGB-based methods and achieves state-of-the-art results across diverse RAW image datasets under varying lighting conditions and dynamic ranges.
>
---
#### [replaced 004] Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.12251v2](http://arxiv.org/pdf/2506.12251v2)**

> **作者:** Boris Ivanovic; Cristiano Saltori; Yurong You; Yan Wang; Wenjie Luo; Marco Pavone
>
> **备注:** 12 pages, 10 figures, 5 tables
>
> **摘要:** Autoregressive Transformers are increasingly being deployed as end-to-end robot and autonomous vehicle (AV) policy architectures, owing to their scalability and potential to leverage internet-scale pretraining for generalization. Accordingly, tokenizing sensor data efficiently is paramount to ensuring the real-time feasibility of such architectures on embedded hardware. To this end, we present an efficient triplane-based multi-camera tokenization strategy that leverages recent advances in 3D neural reconstruction and rendering to produce sensor tokens that are agnostic to the number of input cameras and their resolution, while explicitly accounting for their geometry around an AV. Experiments on a large-scale AV dataset and state-of-the-art neural simulator demonstrate that our approach yields significant savings over current image patch-based tokenization strategies, producing up to 72% fewer tokens, resulting in up to 50% faster policy inference while achieving the same open-loop motion planning accuracy and improved offroad rates in closed-loop driving simulations.
>
---
#### [replaced 005] Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.20090v5](http://arxiv.org/pdf/2405.20090v5)**

> **作者:** Hao Cheng; Erjia Xiao; Jiayan Yang; Jinhao Duan; Yichi Wang; Jiahang Cao; Qiang Zhang; Le Yang; Kaidi Xu; Jindong Gu; Renjing Xu
>
> **备注:** This paper is accepted by ACM MM 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.
>
---
#### [replaced 006] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.01184v3](http://arxiv.org/pdf/2501.01184v3)**

> **作者:** Dat Nguyen; Marcella Astrid; Anis Kacem; Enjie Ghorbel; Djamila Aouada
>
> **备注:** Accepted at ICCV2025. Project Page: https://datdaigia.github.io/FakeSTormer/
>
> **摘要:** Detecting deepfake videos is highly challenging given the complexity of characterizing spatio-temporal artifacts. Most existing methods rely on binary classifiers trained using real and fake image sequences, therefore hindering their generalization capabilities to unseen generation methods. Moreover, with the constant progress in generative Artificial Intelligence (AI), deepfake artifacts are becoming imperceptible at both the spatial and the temporal levels, making them extremely difficult to capture. To address these issues, we propose a fine-grained deepfake video detection approach called FakeSTormer that enforces the modeling of subtle spatio-temporal inconsistencies while avoiding overfitting. Specifically, we introduce a multi-task learning framework that incorporates two auxiliary branches for explicitly attending artifact-prone spatial and temporal regions. Additionally, we propose a video-level data synthesis strategy that generates pseudo-fake videos with subtle spatio-temporal artifacts, providing high-quality samples and hand-free annotations for our additional branches. Extensive experiments on several challenging benchmarks demonstrate the superiority of our approach compared to recent state-of-the-art methods. The code is available at https://github.com/10Ring/FakeSTormer.
>
---
#### [replaced 007] Once-for-All: Controllable Generative Image Compression with Dynamic Granularity Adaptation
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2406.00758v4](http://arxiv.org/pdf/2406.00758v4)**

> **作者:** Anqi Li; Feng Li; Yuxi Liu; Runmin Cong; Yao Zhao; Huihui Bai
>
> **备注:** Accepted by ICLR 2025. Code is available at https://github.com/lianqi1008/Control-GIC
>
> **摘要:** Although recent generative image compression methods have demonstrated impressive potential in optimizing the rate-distortion-perception trade-off, they still face the critical challenge of flexible rate adaption to diverse compression necessities and scenarios. To overcome this challenge, this paper proposes a Controllable Generative Image Compression framework, termed Control-GIC, the first capable of fine-grained bitrate adaption across a broad spectrum while ensuring high-fidelity and generality compression. Control-GIC is grounded in a VQGAN framework that encodes an image as a sequence of variable-length codes (i.e. VQ-indices), which can be losslessly compressed and exhibits a direct positive correlation with the bitrates. Drawing inspiration from the classical coding principle, we correlate the information density of local image patches with their granular representations. Hence, we can flexibly determine a proper allocation of granularity for the patches to achieve dynamic adjustment for VQ-indices, resulting in desirable compression rates. We further develop a probabilistic conditional decoder capable of retrieving historic encoded multi-granularity representations according to transmitted codes, and then reconstruct hierarchical granular features in the formalization of conditional probability, enabling more informative aggregation to improve reconstruction realism. Our experiments show that Control-GIC allows highly flexible and controllable bitrate adaption where the results demonstrate its superior performance over recent state-of-the-art methods. Code is available at https://github.com/lianqi1008/Control-GIC.
>
---
#### [replaced 008] RoundaboutHD: High-Resolution Real-World Urban Environment Benchmark for Multi-Camera Vehicle Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08729v2](http://arxiv.org/pdf/2507.08729v2)**

> **作者:** Yuqiang Lin; Sam Lockyer; Mingxuan Sui; Li Gan; Florian Stanek; Markus Zarbock; Wenbin Li; Adrian Evans; Nic Zhang
>
> **摘要:** The multi-camera vehicle tracking (MCVT) framework holds significant potential for smart city applications, including anomaly detection, traffic density estimation, and suspect vehicle tracking. However, current publicly available datasets exhibit limitations, such as overly simplistic scenarios, low-resolution footage, and insufficiently diverse conditions, creating a considerable gap between academic research and real-world scenario. To fill this gap, we introduce RoundaboutHD, a comprehensive, high-resolution multi-camera vehicle tracking benchmark dataset specifically designed to represent real-world roundabout scenarios. RoundaboutHD provides a total of 40 minutes of labelled video footage captured by four non-overlapping, high-resolution (4K resolution, 15 fps) cameras. In total, 512 unique vehicle identities are annotated across different camera views, offering rich cross-camera association data. RoundaboutHD offers temporal consistency video footage and enhanced challenges, including increased occlusions and nonlinear movement inside the roundabout. In addition to the full MCVT dataset, several subsets are also available for object detection, single camera tracking, and image-based vehicle re-identification (ReID) tasks. Vehicle model information and camera modelling/ geometry information are also included to support further analysis. We provide baseline results for vehicle detection, single-camera tracking, image-based vehicle re-identification, and multi-camera tracking. The dataset and the evaluation code are publicly available at: https://github.com/siri-rouser/RoundaboutHD.git
>
---
#### [replaced 009] AnyTSR: Any-Scale Thermal Super-Resolution for UAV
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13682v2](http://arxiv.org/pdf/2504.13682v2)**

> **作者:** Mengyuan Li; Changhong Fu; Ziyu Lu; Zijie Zhang; Haobo Zuo; Liangliang Yao
>
> **摘要:** Thermal imaging can greatly enhance the application of intelligent unmanned aerial vehicles (UAV) in challenging environments. However, the inherent low resolution of thermal sensors leads to insufficient details and blurred boundaries. Super-resolution (SR) offers a promising solution to address this issue, while most existing SR methods are designed for fixed-scale SR. They are computationally expensive and inflexible in practical applications. To address above issues, this work proposes a novel any-scale thermal SR method (AnyTSR) for UAV within a single model. Specifically, a new image encoder is proposed to explicitly assign specific feature code to enable more accurate and flexible representation. Additionally, by effectively embedding coordinate offset information into the local feature ensemble, an innovative any-scale upsampler is proposed to better understand spatial relationships and reduce artifacts. Moreover, a novel dataset (UAV-TSR), covering both land and water scenes, is constructed for thermal SR tasks. Experimental results demonstrate that the proposed method consistently outperforms state-of-the-art methods across all scaling factors as well as generates more accurate and detailed high-resolution images. The code is located at https://github.com/vision4robotics/AnyTSR.
>
---
#### [replaced 010] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.10789v4](http://arxiv.org/pdf/2408.10789v4)**

> **作者:** Zhirui Gao; Renjiao Yi; Yuhang Huang; Wei Chen; Chenyang Zhu; Kai Xu
>
> **备注:** Accepted by ICCV 2025. Code: https://github.com/zhirui-gao/PartGS
>
> **摘要:** Low-level 3D representations, such as point clouds, meshes, NeRFs and 3D Gaussians, are commonly used for modeling 3D objects and scenes. However, cognitive studies indicate that human perception operates at higher levels and interprets 3D environments by decomposing them into meaningful structural parts, rather than low-level elements like points or voxels. Structured geometric decomposition enhances scene interpretability and facilitates downstream tasks requiring component-level manipulation. In this work, we introduce PartGS, a self-supervised part-aware reconstruction framework that integrates 2D Gaussians and superquadrics to parse objects and scenes into an interpretable decomposition, leveraging multi-view image inputs to uncover 3D structural information. Our method jointly optimizes superquadric meshes and Gaussians by coupling their parameters within a hybrid representation. On one hand, superquadrics enable the representation of a wide range of shape primitives, facilitating flexible and meaningful decompositions. On the other hand, 2D Gaussians capture detailed texture and geometric details, ensuring high-fidelity appearance and geometry reconstruction. Operating in a self-supervised manner, our approach demonstrates superior performance compared to state-of-the-art methods across extensive experiments on the DTU, ShapeNet, and real-world datasets.
>
---
#### [replaced 011] Composed Multi-modal Retrieval: A Survey of Approaches and Applications
- **分类: cs.IR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01334v2](http://arxiv.org/pdf/2503.01334v2)**

> **作者:** Kun Zhang; Jingyu Li; Zhe Li; Jingjing Zhang; Fan Li; Yandong Liu; Rui Yan; Zihang Jiang; Nan Chen; Lei Zhang; Yongdong Zhang; Zhendong Mao; S. Kevin Zhou
>
> **摘要:** The burgeoning volume of multi-modal data necessitates advanced retrieval paradigms beyond unimodal and cross-modal approaches. Composed Multi-modal Retrieval (CMR) emerges as a pivotal next-generation technology, enabling users to query images or videos by integrating a reference visual input with textual modifications, thereby achieving unprecedented flexibility and precision. This paper provides a comprehensive survey of CMR, covering its fundamental challenges, technical advancements, and applications. CMR is categorized into supervised, zero-shot, and semi-supervised learning paradigms. We discuss key research directions, including data construction, model architecture, and loss optimization in supervised CMR, as well as transformation frameworks and linear integration in zero-shot CMR, and semi-supervised CMR that leverages generated pseudo-triplets while addressing data noise/uncertainty. Additionally, we extensively survey the diverse application landscape of CMR, highlighting its transformative potential in e-commerce, social media, search engines, public security, etc. Seven high impact application scenarios are explored in detail with benchmark data sets and performance analysis. Finally, we further provide new potential research directions with the hope of inspiring exploration in other yet-to-be-explored fields. A curated list of works is available at: https://github.com/kkzhang95/Awesome-Composed-Multi-modal-Retrieval
>
---
#### [replaced 012] Sports Re-ID: Improving Re-Identification Of Players In Broadcast Videos Of Team Sports
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2206.02373v2](http://arxiv.org/pdf/2206.02373v2)**

> **作者:** Bharath Comandur
>
> **摘要:** This work focuses on player re-identification in broadcast videos of team sports. Specifically, we focus on identifying the same player in images captured from different camera viewpoints during any given moment of a match. This task differs from traditional applications of person re-id in a few important ways. Firstly, players from the same team wear highly similar clothes, thereby making it harder to tell them apart. Secondly, there are only a few number of samples for each identity, which makes it harder to train a re-id system. Thirdly, the resolutions of the images are often quite low and vary a lot. This combined with heavy occlusions and fast movements of players greatly increase the challenges for re-id. In this paper, we propose a simple but effective hierarchical data sampling procedure and a centroid loss function that, when used together, increase the mean average precision (mAP) by 7 - 11.5 and the rank-1 (R1) by 8.8 - 14.9 without any change in the network or hyper-parameters used. Our data sampling procedure improves the similarity of the training and test distributions, and thereby aids in creating better estimates of the centroids of the embeddings (or feature vectors). Surprisingly, our study shows that in the presence of severely limited data, as is the case for our application, a simple centroid loss function based on euclidean distances significantly outperforms the popular triplet-centroid loss function. We show comparable improvements for both convolutional networks and vision transformers. Our approach is among the top ranked methods in the SoccerNet Re-Identification Challenge 2022 leaderboard (test-split) with a mAP of 86.0 and a R1 of 81.5. On the sequestered challenge split, we achieve an mAP of 84.9 and a R1 of 80.1. Research on re-id for sports-related applications is very limited and our work presents one of the first discussions in the literature on this.
>
---
#### [replaced 013] R-Genie: Reasoning-Guided Generative Image Editing
- **分类: cs.CV; F.2.2, I.2.7; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.17768v2](http://arxiv.org/pdf/2505.17768v2)**

> **作者:** Dong Zhang; Lingfeng He; Rui Yan; Fei Shen; Jinhui Tang
>
> **备注:** Code: https://dongzhang89.github.io/RGenie.github.io/
>
> **摘要:** While recent advances in image editing have enabled impressive visual synthesis capabilities, current methods remain constrained by explicit textual instructions and limited editing operations, lacking deep comprehension of implicit user intentions and contextual reasoning. In this work, we introduce a new image editing paradigm: reasoning-guided generative editing, which synthesizes images based on complex, multi-faceted textual queries accepting world knowledge and intention inference. To facilitate this task, we first construct a comprehensive dataset featuring over 1,000 image-instruction-edit triples that incorporate rich reasoning contexts and real-world knowledge. We then propose R-Genie: a reasoning-guided generative image editor, which synergizes the generation power of diffusion models with advanced reasoning capabilities of multimodal large language models. R-Genie incorporates a reasoning-attention mechanism to bridge linguistic understanding with visual synthesis, enabling it to handle intricate editing requests involving abstract user intentions and contextual reasoning relations. Extensive experimental results validate that R-Genie can equip diffusion models with advanced reasoning-based editing capabilities, unlocking new potentials for intelligent image synthesis.
>
---
#### [replaced 014] AutoPartGen: Autogressive 3D Part Generation and Discovery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13346v2](http://arxiv.org/pdf/2507.13346v2)**

> **作者:** Minghao Chen; Jianyuan Wang; Roman Shapovalov; Tom Monnier; Hyunyoung Jung; Dilin Wang; Rakesh Ranjan; Iro Laina; Andrea Vedaldi
>
> **备注:** Project page: https://silent-chen.github.io/AutoPartGen/
>
> **摘要:** We introduce AutoPartGen, a model that generates objects composed of 3D parts in an autoregressive manner. This model can take as input an image of an object, 2D masks of the object's parts, or an existing 3D object, and generate a corresponding compositional 3D reconstruction. Our approach builds upon 3DShape2VecSet, a recent latent 3D representation with powerful geometric expressiveness. We observe that this latent space exhibits strong compositional properties, making it particularly well-suited for part-based generation tasks. Specifically, AutoPartGen generates object parts autoregressively, predicting one part at a time while conditioning on previously generated parts and additional inputs, such as 2D images, masks, or 3D objects. This process continues until the model decides that all parts have been generated, thus determining automatically the type and number of parts. The resulting parts can be seamlessly assembled into coherent objects or scenes without requiring additional optimization. We evaluate both the overall 3D generation capabilities and the part-level generation quality of AutoPartGen, demonstrating that it achieves state-of-the-art performance in 3D part generation.
>
---
#### [replaced 015] DGSSA: Domain generalization with structural and stylistic augmentation for retinal vessel segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.03466v2](http://arxiv.org/pdf/2501.03466v2)**

> **作者:** Bo Liu; Yudong Zhang; Shuihua Wang; Siyue Li; Jin Hong
>
> **摘要:** Retinal vascular morphology is crucial for diagnosing diseases such as diabetes, glaucoma, and hypertension, making accurate segmentation of retinal vessels essential for early intervention. Traditional segmentation methods assume that training and testing data share similar distributions, which can lead to poor performance on unseen domains due to domain shifts caused by variations in imaging devices and patient demographics. This paper presents a novel approach, DGSSA, for retinal vessel image segmentation that enhances model generalization by combining structural and style augmentation strategies. We utilize a space colonization algorithm to generate diverse vascular-like structures that closely mimic actual retinal vessels, which are then used to generate pseudo-retinal images with an improved Pix2Pix model, allowing the segmentation model to learn a broader range of structure distributions. Additionally, we utilize PixMix to implement random photometric augmentations and introduce uncertainty perturbations, thereby enriching stylistic diversity and significantly enhancing the model's adaptability to varying imaging conditions. Our framework has been rigorously evaluated on four challenging datasets-DRIVE, CHASEDB, HRF, and STARE-demonstrating state-of-the-art performance that surpasses existing methods. This validates the effectiveness of our proposed approach, highlighting its potential for clinical application in automated retinal vessel analysis.
>
---
#### [replaced 016] SpatialTrackerV2: 3D Point Tracking Made Easy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12462v2](http://arxiv.org/pdf/2507.12462v2)**

> **作者:** Yuxi Xiao; Jianyuan Wang; Nan Xue; Nikita Karaev; Yuri Makarov; Bingyi Kang; Xing Zhu; Hujun Bao; Yujun Shen; Xiaowei Zhou
>
> **备注:** International Conference on Computer Vision, ICCV 2025. Huggingface Demo: https://huggingface.co/spaces/Yuxihenry/SpatialTrackerV2, Code: https://github.com/henry123-boy/SpaTrackerV2
>
> **摘要:** We present SpatialTrackerV2, a feed-forward 3D point tracking method for monocular videos. Going beyond modular pipelines built on off-the-shelf components for 3D tracking, our approach unifies the intrinsic connections between point tracking, monocular depth, and camera pose estimation into a high-performing and feedforward 3D point tracker. It decomposes world-space 3D motion into scene geometry, camera ego-motion, and pixel-wise object motion, with a fully differentiable and end-to-end architecture, allowing scalable training across a wide range of datasets, including synthetic sequences, posed RGB-D videos, and unlabeled in-the-wild footage. By learning geometry and motion jointly from such heterogeneous data, SpatialTrackerV2 outperforms existing 3D tracking methods by 30%, and matches the accuracy of leading dynamic 3D reconstruction approaches while running 50$\times$ faster.
>
---
#### [replaced 017] MoViAD: A Modular Library for Visual Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12049v2](http://arxiv.org/pdf/2507.12049v2)**

> **作者:** Manuel Barusco; Francesco Borsatti; Arianna Stropeni; Davide Dalle Pezze; Gian Antonio Susto
>
> **摘要:** VAD is a critical field in machine learning focused on identifying deviations from normal patterns in images, often challenged by the scarcity of anomalous data and the need for unsupervised training. To accelerate research and deployment in this domain, we introduce MoViAD, a comprehensive and highly modular library designed to provide fast and easy access to state-of-the-art VAD models, trainers, datasets, and VAD utilities. MoViAD supports a wide array of scenarios, including continual, semi-supervised, few-shots, noisy, and many more. In addition, it addresses practical deployment challenges through dedicated Edge and IoT settings, offering optimized models and backbones, along with quantization and compression utilities for efficient on-device execution and distributed inference. MoViAD integrates a selection of backbones, robust evaluation VAD metrics (pixel-level and image-level) and useful profiling tools for efficiency analysis. The library is designed for fast, effortless deployment, enabling machine learning engineers to easily use it for their specific setup with custom models, datasets, and backbones. At the same time, it offers the flexibility and extensibility researchers need to develop and experiment with new methods.
>
---
#### [replaced 018] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.18423v3](http://arxiv.org/pdf/2404.18423v3)**

> **作者:** Yeon-Ji Song; Jaein Kim; Suhyung Choi; Jin-Hwa Kim; Byoung-Tak Zhang
>
> **备注:** Accepted at ICCV2025
>
> **摘要:** Human perception involves decomposing complex multi-object scenes into time-static object appearance (i.e., size, shape, color) and time-varying object motion (i.e., position, velocity, acceleration). For machines to achieve human-like intelligence in real-world interactions, understanding these physical properties of objects is essential, forming the foundation for dynamic video prediction. While recent advancements in object-centric transformers have demonstrated potential in video prediction, they primarily focus on object appearance, often overlooking motion dynamics, which is crucial for modeling dynamic interactions and maintaining temporal consistency in complex environments. To address these limitations, we propose OCK, a dynamic video prediction model leveraging object-centric kinematics and object slots. We introduce a novel component named Object Kinematics that comprises explicit object motions, serving as an additional attribute beyond conventional appearance features to model dynamic scenes. The Object Kinematics are integrated into various OCK mechanisms, enabling spatiotemporal prediction of complex object interactions over long video sequences. Our model demonstrates superior performance in handling complex scenes with intricate object attributes and motions, highlighting its potential for applicability in vision-related dynamics learning tasks.
>
---
#### [replaced 019] Rethinking Data Protection in the (Generative) Artificial Intelligence Era
- **分类: cs.LG; cs.AI; cs.CR; cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.03034v3](http://arxiv.org/pdf/2507.03034v3)**

> **作者:** Yiming Li; Shuo Shao; Yu He; Junfeng Guo; Tianwei Zhang; Zhan Qin; Pin-Yu Chen; Michael Backes; Philip Torr; Dacheng Tao; Kui Ren
>
> **备注:** Perspective paper for a broader scientific audience. The first two authors contributed equally to this paper. 13 pages
>
> **摘要:** The (generative) artificial intelligence (AI) era has profoundly reshaped the meaning and value of data. No longer confined to static content, data now permeates every stage of the AI lifecycle from the training samples that shape model parameters to the prompts and outputs that drive real-world model deployment. This shift renders traditional notions of data protection insufficient, while the boundaries of what needs safeguarding remain poorly defined. Failing to safeguard data in AI systems can inflict societal and individual, underscoring the urgent need to clearly delineate the scope of and rigorously enforce data protection. In this perspective, we propose a four-level taxonomy, including non-usability, privacy preservation, traceability, and deletability, that captures the diverse protection needs arising in modern (generative) AI models and systems. Our framework offers a structured understanding of the trade-offs between data utility and control, spanning the entire AI pipeline, including training datasets, model weights, system prompts, and AI-generated content. We analyze representative technical approaches at each level and reveal regulatory blind spots that leave critical assets exposed. By offering a structured lens to align future AI technologies and governance with trustworthy data practices, we underscore the urgency of rethinking data protection for modern AI techniques and provide timely guidance for developers, researchers, and regulators alike.
>
---
#### [replaced 020] PositionIC: Unified Position and Identity Consistency for Image Customization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13861v2](http://arxiv.org/pdf/2507.13861v2)**

> **作者:** Junjie Hu; Tianyang Han; Kai Ma; Jialin Gao; Hao Dou; Song Yang; Xianhua He; Jianhui Zhang; Junfeng Luo; Xiaoming Wei; Wenqiang Zhang
>
> **摘要:** Recent subject-driven image customization has achieved significant advancements in fidelity, yet fine-grained entity-level spatial control remains elusive, hindering the broader real-world application. This limitation is mainly attributed to scalable datasets that bind identity with precise positional cues are absent. To this end, we introduce PositionIC, a unified framework that enforces position and identity consistency for multi-subject customization. We construct a scalable synthesis pipeline that employs a bidirectional generation paradigm to eliminate subject drift and maintain semantic coherence. On top of these data, we design a lightweight positional modulation layer that decouples spatial embeddings among subjects, enabling independent, accurate placement while preserving visual fidelity. Extensive experiments demonstrate that our approach can achieve precise spatial control while maintaining high consistency in image customization task. PositionIC paves the way for controllable, high-fidelity image customization in open-world, multi-entity scenarios and will be released to foster further research.
>
---
#### [replaced 021] Improving Multimodal Learning via Imbalanced Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.10203v2](http://arxiv.org/pdf/2507.10203v2)**

> **作者:** Shicai Wei; Chunbo Luo; Yang Luo
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Multimodal learning often encounters the under-optimized problem and may perform worse than unimodal learning. Existing approaches attribute this issue to imbalanced learning across modalities and tend to address it through gradient balancing. However, this paper argues that balanced learning is not the optimal setting for multimodal learning. With bias-variance analysis, we prove that imbalanced dependency on each modality obeying the inverse ratio of their variances contributes to optimal performance. To this end, we propose the Asymmetric Representation Learning(ARL) strategy to assist multimodal learning via imbalanced optimization. ARL introduces auxiliary regularizers for each modality encoder to calculate their prediction variance. ARL then calculates coefficients via the unimodal variance to re-weight the optimization of each modality, forcing the modality dependence ratio to be inversely proportional to the modality variance ratio. Moreover, to minimize the generalization error, ARL further introduces the prediction bias of each modality and jointly optimizes them with multimodal loss. Notably, all auxiliary regularizers share parameters with the multimodal model and rely only on the modality representation. Thus the proposed ARL strategy introduces no extra parameters and is independent of the structures and fusion methods of the multimodal model. Finally, extensive experiments on various datasets validate the effectiveness and versatility of ARL. Code is available at \href{https://github.com/shicaiwei123/ICCV2025-ARL}{https://github.com/shicaiwei123/ICCV2025-ARL}
>
---
#### [replaced 022] Controllable Weather Synthesis and Removal with Video Diffusion Models
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00704v2](http://arxiv.org/pdf/2505.00704v2)**

> **作者:** Chih-Hao Lin; Zian Wang; Ruofan Liang; Yuxuan Zhang; Sanja Fidler; Shenlong Wang; Zan Gojcic
>
> **备注:** International Conference on Computer Vision (ICCV) 2025, Project Website: https://research.nvidia.com/labs/toronto-ai/WeatherWeaver/
>
> **摘要:** Generating realistic and controllable weather effects in videos is valuable for many applications. Physics-based weather simulation requires precise reconstructions that are hard to scale to in-the-wild videos, while current video editing often lacks realism and control. In this work, we introduce WeatherWeaver, a video diffusion model that synthesizes diverse weather effects -- including rain, snow, fog, and clouds -- directly into any input video without the need for 3D modeling. Our model provides precise control over weather effect intensity and supports blending various weather types, ensuring both realism and adaptability. To overcome the scarcity of paired training data, we propose a novel data strategy combining synthetic videos, generative image editing, and auto-labeled real-world videos. Extensive evaluations show that our method outperforms state-of-the-art methods in weather simulation and removal, providing high-quality, physically plausible, and scene-identity-preserving results over various real-world videos.
>
---
#### [replaced 023] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11055v3](http://arxiv.org/pdf/2507.11055v3)**

> **作者:** Shuchang Ye; Usman Naseem; Mingyuan Meng; Jinman Kim
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Medical language-guided segmentation, integrating textual clinical reports as auxiliary guidance to enhance image segmentation, has demonstrated significant improvements over unimodal approaches. However, its inherent reliance on paired image-text input, which we refer to as ``textual reliance", presents two fundamental limitations: 1) many medical segmentation datasets lack paired reports, leaving a substantial portion of image-only data underutilized for training; and 2) inference is limited to retrospective analysis of cases with paired reports, limiting its applicability in most clinical scenarios where segmentation typically precedes reporting. To address these limitations, we propose ProLearn, the first Prototype-driven Learning framework for language-guided segmentation that fundamentally alleviates textual reliance. At its core, we introduce a novel Prototype-driven Semantic Approximation (PSA) module to enable approximation of semantic guidance from textual input. PSA initializes a discrete and compact prototype space by distilling segmentation-relevant semantics from textual reports. Once initialized, it supports a query-and-respond mechanism which approximates semantic guidance for images without textual input, thereby alleviating textual reliance. Extensive experiments on QaTa-COV19, MosMedData+ and Kvasir-SEG demonstrate that ProLearn outperforms state-of-the-art language-guided methods when limited text is available.
>
---
#### [replaced 024] EgoEvGesture: Gesture Recognition Based on Egocentric Event Camera
- **分类: cs.CV; cs.RO; eess.IV; physics.optics**

- **链接: [http://arxiv.org/pdf/2503.12419v3](http://arxiv.org/pdf/2503.12419v3)**

> **作者:** Luming Wang; Hao Shi; Xiaoting Yin; Kailun Yang; Kaiwei Wang; Jian Bai
>
> **备注:** Accepted to SMC 2025. The dataset and models are made available at https://github.com/3190105222/EgoEv_Gesture
>
> **摘要:** Egocentric gesture recognition is a pivotal technology for enhancing natural human-computer interaction, yet traditional RGB-based solutions suffer from motion blur and illumination variations in dynamic scenarios. While event cameras show distinct advantages in handling high dynamic range with ultra-low power consumption, existing RGB-based architectures face inherent limitations in processing asynchronous event streams due to their synchronous frame-based nature. Moreover, from an egocentric perspective, event cameras record data that includes events generated by both head movements and hand gestures, thereby increasing the complexity of gesture recognition. To address this, we propose a novel network architecture specifically designed for event data processing, incorporating (1) a lightweight CNN with asymmetric depthwise convolutions to reduce parameters while preserving spatiotemporal features, (2) a plug-and-play state-space model as context block that decouples head movement noise from gesture dynamics, and (3) a parameter-free Bins-Temporal Shift Module (BTSM) that shifts features along bins and temporal dimensions to fuse sparse events efficiently. We further establish the EgoEvGesture dataset, the first large-scale dataset for egocentric gesture recognition using event cameras. Experimental results demonstrate that our method achieves 62.7% accuracy tested on unseen subjects with only 7M parameters, 3.1% higher than state-of-the-art approaches. Notable misclassifications in freestyle motions stem from high inter-personal variability and unseen test patterns differing from training data. Moreover, our approach achieved a remarkable accuracy of 97.0% on the DVS128 Gesture, demonstrating the effectiveness and generalization capability of our method on public datasets. The dataset and models are made available at https://github.com/3190105222/EgoEv_Gesture.
>
---
#### [replaced 025] BGM: Background Mixup for X-ray Prohibited Items Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.00460v2](http://arxiv.org/pdf/2412.00460v2)**

> **作者:** Weizhe Liu; Renshuai Tao; Hongguang Zhu; Yunda Sun; Yao Zhao; Yunchao Wei
>
> **摘要:** Current data-driven approaches for X-ray prohibited items detection remain under-explored, particularly in the design of effective data augmentations. Existing natural image augmentations for reflected light imaging neglect the data characteristics of X-ray security images. Moreover, prior X-ray augmentation methods have predominantly focused on foreground prohibited items, overlooking informative background cues. In this paper, we propose Background Mixup (BGM), a background-based augmentation technique tailored for X-ray security imaging domain. Unlike conventional methods, BGM is founded on an in-depth analysis of physical properties including: 1) X-ray Transmission Imagery: Transmitted X-ray pixels represent composite information from multiple materials along the imaging path. 2) Material-based Pseudo-coloring: Pseudo-coloring in X-ray images correlates directly with material properties, aiding in material distinction. Building upon the above insights, BGM mixes background patches across regions on both 1) texture structure and 2) material variation, to benefit models from complicated background cues. This enhances the model's capability to handle domain-specific challenges such as occlusion-induced discriminative imbalance. Importantly, BGM is orthogonal and fully compatible with existing foreground-focused augmentation techniques, enabling joint use to further enhance detection performance. Extensive experiments on multiple X-ray security benchmarks show that BGM consistently surpasses strong baselines, without additional annotations or significant training overhead. This work pioneers the exploration of background-aware augmentation in X-ray prohibited items detection and provides a lightweight, plug-and-play solution with broad applicability.
>
---
#### [replaced 026] Contour Flow Constraint: Preserving Global Shape Similarity for Deep Learning based Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09384v2](http://arxiv.org/pdf/2504.09384v2)**

> **作者:** Shengzhe Chen; Zhaoxuan Dong; Jun Liu
>
> **备注:** Submitted to IEEE Transactions on Image Processing on Dec-14-2023. Revised on Oct-16-2024, June-4-2025. Accepted on July-8-2025
>
> **摘要:** For effective image segmentation, it is crucial to employ constraints informed by prior knowledge about the characteristics of the areas to be segmented to yield favorable segmentation outcomes. However, the existing methods have primarily focused on priors of specific properties or shapes, lacking consideration of the general global shape similarity from a Contour Flow (CF) perspective. Furthermore, naturally integrating this contour flow prior image segmentation model into the activation functions of deep convolutional networks through mathematical methods is currently unexplored. In this paper, we establish a concept of global shape similarity based on the premise that two shapes exhibit comparable contours. Furthermore, we mathematically derive a contour flow constraint that ensures the preservation of global shape similarity. We propose two implementations to integrate the constraint with deep neural networks. Firstly, the constraint is converted to a shape loss, which can be seamlessly incorporated into the training phase for any learning-based segmentation framework. Secondly, we add the constraint into a variational segmentation model and derive its iterative schemes for solution. The scheme is then unrolled to get the architecture of the proposed CFSSnet. Validation experiments on diverse datasets are conducted on classic benchmark deep network segmentation models. The results indicate a great improvement in segmentation accuracy and shape similarity for the proposed shape loss, showcasing the general adaptability of the proposed loss term regardless of specific network architectures. CFSSnet shows robustness in segmenting noise-contaminated images, and inherent capability to preserve global shape similarity.
>
---
#### [replaced 027] In-2-4D: Inbetweening from Two Single-View Images to 4D Generation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.08366v2](http://arxiv.org/pdf/2504.08366v2)**

> **作者:** Sauradip Nag; Daniel Cohen-Or; Hao Zhang; Ali Mahdavi-Amiri
>
> **备注:** Technical Report
>
> **摘要:** We propose a new problem, In-2-4D, for generative 4D (i.e., 3D + motion) inbetweening from a minimalistic input setting: two single-view images capturing an object in two distinct motion states. Given two images representing the start and end states of an object in motion, our goal is to generate and reconstruct the motion in 4D. We utilize a video interpolation model to predict the motion, but large frame-to-frame motions can lead to ambiguous interpretations. To overcome this, we employ a hierarchical approach to identify keyframes that are visually close to the input states and show significant motion, then generate smooth fragments between them. For each fragment, we construct the 3D representation of the keyframe using Gaussian Splatting. The temporal frames within the fragment guide the motion, enabling their transformation into dynamic Gaussians through a deformation field. To improve temporal consistency and refine 3D motion, we expand the self-attention of multi-view diffusion across timesteps and apply rigid transformation regularization. Finally, we merge the independently generated 3D motion segments by interpolating boundary deformation fields and optimizing them to align with the guiding video, ensuring smooth and flicker-free transitions. Through extensive qualitative and quantitiave experiments as well as a user study, we show the effectiveness of our method and its components. The project page is available at https://in-2-4d.github.io/
>
---
#### [replaced 028] MDNF: Multi-Diffusion-Nets for Neural Fields on Meshes
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.03034v2](http://arxiv.org/pdf/2409.03034v2)**

> **作者:** Avigail Cohen Rimon; Tal Shnitzer; Mirela Ben Chen
>
> **备注:** Accepted to SGP 2025 (Symposium on Geometry Processing)
>
> **摘要:** We propose a novel framework for representing neural fields on triangle meshes that is multi-resolution across both spatial and frequency domains. Inspired by the Neural Fourier Filter Bank (NFFB), our architecture decomposes the spatial and frequency domains by associating finer spatial resolution levels with higher frequency bands, while coarser resolutions are mapped to lower frequencies. To achieve geometry-aware spatial decomposition we leverage multiple DiffusionNet components, each associated with a different spatial resolution level. Subsequently, we apply a Fourier feature mapping to encourage finer resolution levels to be associated with higher frequencies. The final signal is composed in a wavelet-inspired manner using a sine-activated MLP, aggregating higher-frequency signals on top of lower-frequency ones. Our architecture attains high accuracy in learning complex neural fields and is robust to discontinuities, exponential scale variations of the target field, and mesh modification. We demonstrate the effectiveness of our approach through its application to diverse neural fields, such as synthetic RGB functions, UV texture coordinates, and vertex normals, illustrating different challenges. To validate our method, we compare its performance against two alternatives, showcasing the advantages of our multi-resolution architecture.
>
---
#### [replaced 029] Stereo Any Video: Temporally Consistent Stereo Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05549v3](http://arxiv.org/pdf/2503.05549v3)**

> **作者:** Junpeng Jing; Weixun Luo; Ye Mao; Krystian Mikolajczyk
>
> **备注:** ICCV2025
>
> **摘要:** This paper introduces Stereo Any Video, a powerful framework for video stereo matching. It can estimate spatially accurate and temporally consistent disparities without relying on auxiliary information such as camera poses or optical flow. The strong capability is driven by rich priors from monocular video depth models, which are integrated with convolutional features to produce stable representations. To further enhance performance, key architectural innovations are introduced: all-to-all-pairs correlation, which constructs smooth and robust matching cost volumes, and temporal convex upsampling, which improves temporal coherence. These components collectively ensure robustness, accuracy, and temporal consistency, setting a new standard in video stereo matching. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple datasets both qualitatively and quantitatively in zero-shot settings, as well as strong generalization to real-world indoor and outdoor scenarios.
>
---
#### [replaced 030] Comparative Evaluation of Radiomics and Deep Learning Models for Disease Detection in Chest Radiography
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.12249v3](http://arxiv.org/pdf/2504.12249v3)**

> **作者:** Zhijin He; Alan B. McMillan
>
> **备注:** one figure updated for improved visual clarity; fixed inconsistencies and redundancies
>
> **摘要:** The application of artificial intelligence (AI) in medical imaging has revolutionized diagnostic practices, enabling advanced analysis and interpretation of radiological data. This study presents a comprehensive evaluation of radiomics-based and deep learning-based approaches for disease detection in chest radiography, focusing on COVID-19, lung opacity, and viral pneumonia. While deep learning models, particularly convolutional neural networks and vision transformers, learn directly from image data, radiomics-based models extract handcrafted features, offering potential advantages in data-limited scenarios. We systematically compared the diagnostic performance of various AI models, including Decision Trees, Gradient Boosting, Random Forests, Support Vector Machines, and Multi-Layer Perceptrons for radiomics, against state-of-the-art deep learning models such as InceptionV3, EfficientNetL, and ConvNeXtXLarge. Performance was evaluated across multiple sample sizes. At 24 samples, EfficientNetL achieved an AUC of 0.839, outperforming SVM (AUC = 0.762). At 4000 samples, InceptionV3 achieved the highest AUC of 0.996, compared to 0.885 for Random Forest. A Scheirer-Ray-Hare test confirmed significant main and interaction effects of model type and sample size on all metrics. Post hoc Mann-Whitney U tests with Bonferroni correction further revealed consistent performance advantages for deep learning models across most conditions. These findings provide statistically validated, data-driven recommendations for model selection in diagnostic AI. Deep learning models demonstrated higher performance and better scalability with increasing data availability, while radiomics-based models may remain useful in low-data contexts. This study addresses a critical gap in AI-based diagnostic research by offering practical guidance for deploying AI models across diverse clinical environments.
>
---
#### [replaced 031] CABLD: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17845v3](http://arxiv.org/pdf/2411.17845v3)**

> **作者:** Soorena Salari; Arash Harirpoush; Hassan Rivaz; Yiming Xiao
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Anatomical landmark detection in medical images is essential for various clinical and research applications, including disease diagnosis and surgical planning. However, manual landmark annotation is time-consuming and requires significant expertise. Existing deep learning (DL) methods often require large amounts of well-annotated data, which are costly to acquire. In this paper, we introduce CABLD, a novel self-supervised DL framework for 3D brain landmark detection in unlabeled scans with varying contrasts by using only a single reference example. To achieve this, we employed an inter-subject landmark consistency loss with an image registration loss while introducing a 3D convolution-based contrast augmentation strategy to promote model generalization to new contrasts. Additionally, we utilize an adaptive mixed loss function to schedule the contributions of different sub-tasks for optimal outcomes. We demonstrate the proposed method with the intricate task of MRI-based 3D brain landmark detection. With comprehensive experiments on four diverse clinical and public datasets, including both T1w and T2w MRI scans at different MRI field strengths, we demonstrate that CABLD outperforms the state-of-the-art methods in terms of mean radial errors (MREs) and success detection rates (SDRs). Our framework provides a robust and accurate solution for anatomical landmark detection, reducing the need for extensively annotated datasets and generalizing well across different imaging contrasts. Our code is publicly available at https://github.com/HealthX-Lab/CABLD.
>
---
#### [replaced 032] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.09795v2](http://arxiv.org/pdf/2507.09795v2)**

> **作者:** Amirhossein Ansari; Ke Wang; Pulei Xiong
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Recent advancements in Vision-Language Models like CLIP have enabled zero-shot OOD detection by leveraging both image and textual label information. Among these, negative label-based methods such as NegLabel and CSP have shown promising results by utilizing a lexicon of words to define negative labels for distinguishing OOD samples. However, these methods suffer from detecting in-distribution samples as OOD due to negative labels that are subcategories of in-distribution labels or proper nouns. They also face limitations in handling images that match multiple in-distribution and negative labels. We propose NegRefine, a novel negative label refinement framework for zero-shot OOD detection. By introducing a filtering mechanism to exclude subcategory labels and proper nouns from the negative label set and incorporating a multi-matching-aware scoring function that dynamically adjusts the contributions of multiple labels matching an image, NegRefine ensures a more robust separation between in-distribution and OOD samples. We evaluate NegRefine on large-scale benchmarks, including ImageNet-1K. The code is available at https://github.com/ah-ansari/NegRefine.
>
---
#### [replaced 033] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12137v2](http://arxiv.org/pdf/2507.12137v2)**

> **作者:** Jiawei Xu; Kai Deng; Zexin Fan; Shenlong Wang; Jin Xie; Jian Yang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Modeling and rendering dynamic urban driving scenes is crucial for self-driving simulation. Current high-quality methods typically rely on costly manual object tracklet annotations, while self-supervised approaches fail to capture dynamic object motions accurately and decompose scenes properly, resulting in rendering artifacts. We introduce AD-GS, a novel self-supervised framework for high-quality free-viewpoint rendering of driving scenes from a single log. At its core is a novel learnable motion model that integrates locality-aware B-spline curves with global-aware trigonometric functions, enabling flexible yet precise dynamic object modeling. Rather than requiring comprehensive semantic labeling, AD-GS automatically segments scenes into objects and background with the simplified pseudo 2D segmentation, representing objects using dynamic Gaussians and bidirectional temporal visibility masks. Further, our model incorporates visibility reasoning and physically rigid regularization to enhance robustness. Extensive evaluations demonstrate that our annotation-free model significantly outperforms current state-of-the-art annotation-free methods and is competitive with annotation-dependent approaches.
>
---
#### [replaced 034] Self-Tuning Self-Supervised Image Anomaly Detection
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2306.12033v3](http://arxiv.org/pdf/2306.12033v3)**

> **作者:** Jaemin Yoo; Lingxiao Zhao; Leman Akoglu
>
> **备注:** Accepted to KDD 2025
>
> **摘要:** Self-supervised learning (SSL) has emerged as a promising paradigm that presents supervisory signals to real-world problems, bypassing the extensive cost of manual labeling. Consequently, self-supervised anomaly detection (SSAD) has seen a recent surge of interest, since SSL is especially attractive for unsupervised tasks. However, recent works have reported that the choice of a data augmentation function has significant impact on the accuracy of SSAD, posing augmentation search as an essential but nontrivial problem due to lack of labeled validation data. In this paper, we introduce ST-SSAD, the first unsupervised approach to end-to-end augmentation tuning for SSAD. To this end, our work presents two key contributions. The first is a new unsupervised validation loss that quantifies the alignment between augmented training data and unlabeled validation data. The second is new differentiable augmentation functions, allowing data augmentation hyperparameter(s) to be tuned in an end-to-end manner. Experiments on two testbeds with semantic class anomalies and subtle industrial defects show that ST-SSAD gives significant performance gains over existing works. All our code and testbeds are available at https://github.com/jaeminyoo/ST-SSAD.
>
---
#### [replaced 035] Can Optical Denoising Clean Sonar Images? A Benchmark and Fusion Approach
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.01655v2](http://arxiv.org/pdf/2503.01655v2)**

> **作者:** Ziyu Wang; Tao Xue; Jingyuan Li; Haibin Zhang; Zhiqiang Xu; Gaofei Xu; Zhen Wang; Yanbin Wang; Zhiquan Liu
>
> **摘要:** Object detection in sonar images is crucial for underwater robotics applications including autonomous navigation and resource exploration. However, complex noise patterns inherent in sonar imagery, particularly speckle, reverberation, and non-Gaussian noise, significantly degrade detection accuracy. While denoising techniques have achieved remarkable success in optical imaging, their applicability to sonar data remains underexplored. This study presents the first systematic evaluation of nine state-of-the-art deep denoising models with distinct architectures, including Neighbor2Neighbor with varying noise parameters, Blind2Unblind with different noise configurations, and DSPNet, for sonar image preprocessing. We establish a rigorous benchmark using five publicly available sonar datasets and assess their impact on four representative detection algorithms: YOLOX, Faster R-CNN, SSD300, and SSDMobileNetV2. Our evaluation addresses three unresolved questions: first, how effectively optical denoising architectures transfer to sonar data; second, which model families perform best against sonar noise; and third, whether denoising truly improves detection accuracy in practical pipelines. Extensive experiments demonstrate that while denoising generally improves detection performance, effectiveness varies across methods due to their inherent biases toward specific noise types. To leverage complementary denoising effects, we propose a mutually-supervised multi-source denoising fusion framework where outputs from different denoisers mutually supervise each other at the pixel level, creating a synergistic framework that produces cleaner images.
>
---
#### [replaced 036] A Comprehensive Library for Benchmarking Multi-class Visual Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.03262v5](http://arxiv.org/pdf/2406.03262v5)**

> **作者:** Jiangning Zhang; Haoyang He; Zhenye Gan; Qingdong He; Yuxuan Cai; Zhucun Xue; Yabiao Wang; Chengjie Wang; Lei Xie; Yong Liu
>
> **摘要:** Visual anomaly detection aims to identify anomalous regions in images through unsupervised learning paradigms, with increasing application demand and value in fields such as industrial inspection and medical lesion detection. Despite significant progress in recent years, there is a lack of comprehensive benchmarks to adequately evaluate the performance of various mainstream methods across different datasets under the practical multi-class setting. The absence of standardized experimental setups can lead to potential biases in training epochs, resolution, and metric results, resulting in erroneous conclusions. This paper addresses this issue by proposing a comprehensive visual anomaly detection benchmark, ADer, which is a modular framework that is highly extensible for new methods. The benchmark includes multiple datasets from industrial and medical domains, implementing fifteen state-of-the-art methods and nine comprehensive metrics. Additionally, we have proposed the GPU-assisted ADEval package to address the slow evaluation problem of metrics like time-consuming mAU-PRO on large-scale data, significantly reducing evaluation time by more than 1000-fold. Through extensive experimental results, we objectively reveal the strengths and weaknesses of different methods and provide insights into the challenges and future directions of multi-class visual anomaly detection. We hope that ADer will become a valuable resource for researchers and practitioners in the field, promoting the development of more robust and generalizable anomaly detection systems. Full codes are open-sourced at https://github.com/zhangzjn/ader.
>
---
#### [replaced 037] Video LLMs for Temporal Reasoning in Long Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02930v4](http://arxiv.org/pdf/2412.02930v4)**

> **作者:** Fawad Javed Fateh; Umer Ahmed; Hamza Khan; M. Zeeshan Zia; Quoc-Huy Tran
>
> **摘要:** This paper introduces TemporalVLM, a video large language model (video LLM) capable of effective temporal reasoning and fine-grained understanding in long videos. At the core, our approach includes a visual encoder for mapping a long-term input video into features which are time-aware and contain both local and global cues. In particular, it first divides the input video into short-term clips, which are jointly encoded with their timestamps and fused across overlapping temporal windows into time-sensitive local features. Next, the local features are passed through a bidirectional long short-term memory (BiLSTM) module for global feature aggregation. The extracted time-aware and multi-level features are important for accurate temporal reasoning and fine-grained understanding in long videos. Moreover, to facilitate the evaluation of TemporalVLM, we present a large-scale long video dataset of industry assembly processes, namely IndustryASM, which consists of videos recorded on factory floors with actions and timestamps annotated by industrial engineers for time and motion studies and temporal action segmentation evaluation. Finally, extensive experiments on datasets of long videos, including TimeIT and IndustryASM, show that TemporalVLM achieves superior performance than previous methods across temporal reasoning and fine-grained understanding tasks, namely dense video captioning, temporal video grounding, video highlight detection, and temporal action segmentation. To the best of our knowledge, our work is the first to incorporate LSTMs into video LLMs.
>
---
#### [replaced 038] DualSwinUnet++: An Enhanced Swin-Unet Architecture With Dual Decoders For PTMC Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.18239v3](http://arxiv.org/pdf/2410.18239v3)**

> **作者:** Maryam Dialameh; Hossein Rajabzadeh; Moslem Sadeghi-Goughari; Jung Suk Sim; Hyock Ju Kwon
>
> **摘要:** Precise segmentation of papillary thyroid microcarcinoma (PTMC) during ultrasound-guided radiofrequency ablation (RFA) is critical for effective treatment but remains challenging due to acoustic artifacts, small lesion size, and anatomical variability. In this study, we propose DualSwinUnet++, a dual-decoder transformer-based architecture designed to enhance PTMC segmentation by incorporating thyroid gland context. DualSwinUnet++ employs independent linear projection heads for each decoder and a residual information flow mechanism that passes intermediate features from the first (thyroid) decoder to the second (PTMC) decoder via concatenation and transformation. These design choices allow the model to condition tumor prediction explicitly on gland morphology without shared gradient interference. Trained on a clinical ultrasound dataset with 691 annotated RFA images and evaluated against state-of-the-art models, DualSwinUnet++ achieves superior Dice and Jaccard scores while maintaining sub-200ms inference latency. The results demonstrate the model's suitability for near real-time surgical assistance and its effectiveness in improving segmentation accuracy in challenging PTMC cases.
>
---
#### [replaced 039] Calibrated and Robust Foundation Models for Vision-Language and Medical Image Tasks Under Distribution Shift
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.09222v2](http://arxiv.org/pdf/2507.09222v2)**

> **作者:** Behraj Khan; Tahir Qasim Syed; Nouman M. Durrani; Bilal Naseem; Shabir Ahmad; Rizwan Qureshi
>
> **摘要:** Foundation models like CLIP and SAM have advanced computer vision and medical imaging via low-shot transfer learning, aiding CADD with limited data. However, their deployment faces two key challenges. \textit{distribution shift} where pre-training and post-training data distributions differ (e.g., due to inter-center image acquisition) and \textit{confidence misalignment}, which leads to overconfident errors. These issues surface differently, vision-language models (e.g., CLIP) suffer from 2D embedding shift (image-text misalignment), while medical models (e.g., SAM) encounter 3D domain shifts (e.g., scanner variation) and voxel-wise calibration need. Existing solutions are domain-specific. We propose \textbf{StaRFM}, a fusion of Fisher information penalty (FIP) and confidence misalignment penalty (CMP) tackling both challenges. It applies FIP, extended to 3D via patch-wise regularization, to reduce embedding shift, and CMP, reformulated for voxel-level predictions, to calibrate segmentation uncertainty. We derive PAC-Bayes bounds. FIP controls generalization via the Fisher-Rao norm, and CMP reduces calibration error via Brier score minimization. StaRFM surpasses baselines by \texttt{+}3.5\% accuracy and 28\% lower ECE on 19 vision datasets (e.g., ImageNet, Office-Home), achieves +4.2\% DSC over SAM-FT and 4.8mm HD95 on medical benchmarks (e.g., BraTS, ATLAS), and reduces cross-domain gaps by up to 20\%. The framework is plug-and-play, requiring minimal architectural changes. Code and models are available at: \href{https://anonymous.4open.science/r/StaRFM-C0CD/}{\textcolor{blue}{\underline{StaRFM}}}
>
---
#### [replaced 040] An Explainable Neural Radiomic Sequence Model with Spatiotemporal Continuity for Quantifying 4DCT-based Pulmonary Ventilation
- **分类: physics.med-ph; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23898v2](http://arxiv.org/pdf/2503.23898v2)**

> **作者:** Rihui Zhang; Haiming Zhu; Jingtong Zhao; Lei Zhang; Fang-Fang Yin; Chunhao Wang; Zhenyu Yang
>
> **备注:** 43 pages, 13 figures
>
> **摘要:** Accurate evaluation of regional lung ventilation is essential for the management and treatment of lung cancer patients, supporting assessments of pulmonary function, optimization of therapeutic strategies, and monitoring of treatment response. Currently, ventilation scintigraphy using nuclear medicine techniques is widely employed in clinical practice; however, it is often time-consuming, costly, and entails additional radiation exposure. In this study, we propose an explainable neural radiomic sequence model to identify regions of compromised pulmonary ventilation based on four-dimensional computed tomography (4DCT). A cohort of 45 lung cancer patients from the VAMPIRE dataset was analyzed. For each patient, lung volumes were segmented from 4DCT, and voxel-wise radiomic features (56-dimensional) were extracted across the respiratory cycle to capture local intensity and texture dynamics, forming temporal radiomic sequences. Ground truth ventilation defects were delineated voxel-wise using Galligas-PET and DTPA-SPECT. To identify compromised regions, we developed a temporal saliency-enhanced explainable long short-term memory (LSTM) network trained on the radiomic sequences. Temporal saliency maps were generated to highlight key features contributing to the model's predictions. The proposed model demonstrated robust performance, achieving average (range) Dice similarity coefficients of 0.78 (0.74-0.79) for 25 PET cases and 0.78 (0.74-0.82) for 20 SPECT cases. The temporal saliency map explained three key radiomic sequences in ventilation quantification: during lung exhalation, compromised pulmonary function region typically exhibits (1) an increasing trend of intensity and (2) a decreasing trend of homogeneity, in contrast to healthy lung tissue.
>
---
#### [replaced 041] Efficient Visual Transformer by Learnable Token Merging
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.15219v2](http://arxiv.org/pdf/2407.15219v2)**

> **作者:** Yancheng Wang; Yingzhen Yang
>
> **备注:** Accepted by IEEE TPAMI
>
> **摘要:** Self-attention and transformers have been widely used in deep learning. Recent efforts have been devoted to incorporating transformer blocks into different neural architectures, including those with convolutions, leading to various visual transformers for computer vision tasks. In this paper, we propose a novel and compact transformer block, Transformer with Learnable Token Merging (LTM), or LTM-Transformer. LTM-Transformer performs token merging in a learnable scheme. LTM-Transformer is compatible with many popular and compact transformer networks, and it reduces the FLOPs and the inference time of the visual transformers while maintaining or even improving the prediction accuracy. In the experiments, we replace all the transformer blocks in popular visual transformers, including MobileViT, EfficientViT, ViT, and Swin, with LTM-Transformer blocks, leading to LTM-Transformer networks with different backbones. The LTM-Transformer is motivated by reduction of Information Bottleneck, and a novel and separable variational upper bound for the IB loss is derived. The architecture of the mask module in our LTM blocks, which generates the token merging mask, is designed to reduce the derived upper bound for the IB loss. Extensive results on computer vision tasks evidence that LTM-Transformer renders compact and efficient visual transformers with comparable or much better prediction accuracy than the original visual transformers. The code of the LTM-Transformer is available at https://github.com/Statistical-Deep-Learning/LTM}
>
---
#### [replaced 042] TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.00709v2](http://arxiv.org/pdf/2507.00709v2)**

> **作者:** Yiming Yang; Yueru Luo; Bingkun He; Hongbin Lin; Suzhong Fu; Chao Zheng; Zhipeng Cao; Erlong Li; Chao Yan; Shuguang Cui; Zhen Li
>
> **摘要:** Lane segment topology reasoning constructs a comprehensive road network by capturing the topological relationships between lane segments and their semantic types. This enables end-to-end autonomous driving systems to perform road-dependent maneuvers such as turning and lane changing. However, the limitations in consistent positional embedding and temporal multiple attribute learning in existing methods hinder accurate roadnet reconstruction. To address these issues, we propose TopoStreamer, an end-to-end temporal perception model for lane segment topology reasoning. Specifically, TopoStreamer introduces three key improvements: streaming attribute constraints, dynamic lane boundary positional encoding, and lane segment denoising. The streaming attribute constraints enforce temporal consistency in both centerline and boundary coordinates, along with their classifications. Meanwhile, dynamic lane boundary positional encoding enhances the learning of up-to-date positional information within queries, while lane segment denoising helps capture diverse lane segment patterns, ultimately improving model performance. Additionally, we assess the accuracy of existing models using a lane boundary classification metric, which serves as a crucial measure for lane-changing scenarios in autonomous driving. On the OpenLane-V2 dataset, TopoStreamer demonstrates significant improvements over state-of-the-art methods, achieving substantial performance gains of +3.0% mAP in lane segment perception and +1.7% OLS in centerline perception tasks.
>
---
#### [replaced 043] FlexiTex: Enhancing Texture Generation via Visual Guidance
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.12431v5](http://arxiv.org/pdf/2409.12431v5)**

> **作者:** DaDong Jiang; Xianghui Yang; Zibo Zhao; Sheng Zhang; Jiaao Yu; Zeqiang Lai; Shaoxiong Yang; Chunchao Guo; Xiaobo Zhou; Zhihui Ke
>
> **备注:** Accepted by AAAI 2025, Project Page: https://patrickddj.github.io/FlexiTex/
>
> **摘要:** Recent texture generation methods achieve impressive results due to the powerful generative prior they leverage from large-scale text-to-image diffusion models. However, abstract textual prompts are limited in providing global textural or shape information, which results in the texture generation methods producing blurry or inconsistent patterns. To tackle this, we present FlexiTex, embedding rich information via visual guidance to generate a high-quality texture. The core of FlexiTex is the Visual Guidance Enhancement module, which incorporates more specific information from visual guidance to reduce ambiguity in the text prompt and preserve high-frequency details. To further enhance the visual guidance, we introduce a Direction-Aware Adaptation module that automatically designs direction prompts based on different camera poses, avoiding the Janus problem and maintaining semantically global consistency. Benefiting from the visual guidance, FlexiTex produces quantitatively and qualitatively sound results, demonstrating its potential to advance texture generation for real-world applications.
>
---
#### [replaced 044] Joint Self-Supervised Video Alignment and Action Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16832v2](http://arxiv.org/pdf/2503.16832v2)**

> **作者:** Ali Shah Ali; Syed Ahmed Mahmood; Mubin Saeed; Andrey Konin; M. Zeeshan Zia; Quoc-Huy Tran
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We introduce a novel approach for simultaneous self-supervised video alignment and action segmentation based on a unified optimal transport framework. In particular, we first tackle self-supervised video alignment by developing a fused Gromov-Wasserstein optimal transport formulation with a structural prior, which trains efficiently on GPUs and needs only a few iterations for solving the optimal transport problem. Our single-task method achieves the state-of-the-art performance on multiple video alignment benchmarks and outperforms VAVA, which relies on a traditional Kantorovich optimal transport formulation with an optimality prior. Furthermore, we extend our approach by proposing a unified optimal transport framework for joint self-supervised video alignment and action segmentation, which requires training and storing a single model and saves both time and memory consumption as compared to two different single-task models. Extensive evaluations on several video alignment and action segmentation datasets demonstrate that our multi-task method achieves comparable video alignment yet superior action segmentation results over previous methods in video alignment and action segmentation respectively. Finally, to the best of our knowledge, this is the first work to unify video alignment and action segmentation into a single model.
>
---
#### [replaced 045] YOLOv8-SMOT: An Efficient and Robust Framework for Real-Time Small Object Tracking via Slice-Assisted Training and Adaptive Association
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12087v2](http://arxiv.org/pdf/2507.12087v2)**

> **作者:** Xiang Yu; Xinyao Liu; Guang Liang
>
> **摘要:** Tracking small, agile multi-objects (SMOT), such as birds, from an Unmanned Aerial Vehicle (UAV) perspective is a highly challenging computer vision task. The difficulty stems from three main sources: the extreme scarcity of target appearance features, the complex motion entanglement caused by the combined dynamics of the camera and the targets themselves, and the frequent occlusions and identity ambiguity arising from dense flocking behavior. This paper details our championship-winning solution in the MVA 2025 "Finding Birds" Small Multi-Object Tracking Challenge (SMOT4SB), which adopts the tracking-by-detection paradigm with targeted innovations at both the detection and association levels. On the detection side, we propose a systematic training enhancement framework named \textbf{SliceTrain}. This framework, through the synergy of 'deterministic full-coverage slicing' and 'slice-level stochastic augmentation, effectively addresses the problem of insufficient learning for small objects in high-resolution image training. On the tracking side, we designed a robust tracker that is completely independent of appearance information. By integrating a \textbf{motion direction maintenance (EMA)} mechanism and an \textbf{adaptive similarity metric} combining \textbf{bounding box expansion and distance penalty} into the OC-SORT framework, our tracker can stably handle irregular motion and maintain target identities. Our method achieves state-of-the-art performance on the SMOT4SB public test set, reaching an SO-HOTA score of \textbf{55.205}, which fully validates the effectiveness and advancement of our framework in solving complex real-world SMOT problems. The source code will be made available at https://github.com/Salvatore-Love/YOLOv8-SMOT.
>
---
#### [replaced 046] GreenCrossingAI: A Camera Trap/Computer Vision Pipeline for Environmental Science Research Groups
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.09410v2](http://arxiv.org/pdf/2507.09410v2)**

> **作者:** Bernie Boscoe; Shawn Johnson; Andrea Osbon; Chandler Campbell; Karen Mager
>
> **备注:** This is the preprint version of the paper in Practice and Experience in Advanced Research Computing, PEARC25
>
> **摘要:** Camera traps have long been used by wildlife researchers to monitor and study animal behavior, population dynamics, habitat use, and species diversity in a non-invasive and efficient manner. While data collection from the field has increased with new tools and capabilities, methods to develop, process, and manage the data, especially the adoption of ML/AI tools, remain challenging. These challenges include the sheer volume of data generated, the need for accurate labeling and annotation, variability in environmental conditions affecting data quality, and the integration of ML/AI tools into existing workflows that often require domain-specific customization and computational resources. This paper provides a guide to a low-resource pipeline to process camera trap data on-premise, incorporating ML/AI capabilities tailored for small research groups with limited resources and computational expertise. By focusing on practical solutions, the pipeline offers accessible approaches for data transmission, inference, and evaluation, enabling researchers to discover meaningful insights from their ever-increasing camera trap datasets.
>
---
#### [replaced 047] Prompt2DEM: High-Resolution DEMs for Urban and Open Environments from Global Prompts Using a Monocular Foundation Model
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.09681v2](http://arxiv.org/pdf/2507.09681v2)**

> **作者:** Osher Rafaeli; Tal Svoray; Ariel Nahlieli
>
> **备注:** 18 pages
>
> **摘要:** High-resolution elevation estimations are essential to understand catchment and hillslope hydrology, study urban morphology and dynamics, and monitor the growth, decline, and mortality of terrestrial ecosystems. Various deep learning approaches (e.g., super-resolution techniques, monocular depth estimation) have been developed to create high-resolution Digital Elevation Models (DEMs). However, super-resolution techniques are limited by the upscaling factor, and monocular depth estimation lacks global elevation context, making its conversion to a seamless DEM restricted. The recently introduced technique of prompt-based monocular depth estimation has opened new opportunities to extract estimates of absolute elevation in a global context. We present here a framework for the estimation of high-resolution DEMs as a new paradigm for absolute global elevation mapping. It is exemplified using low-resolution Shuttle Radar Topography Mission (SRTM) elevation data as prompts and high-resolution RGB imagery from the National Agriculture Imagery Program (NAIP). The approach fine-tunes a vision transformer encoder with LiDAR-derived DEMs and employs a versatile prompting strategy, enabling tasks such as DEM estimation, void filling, and updating. Our framework achieves a 100x resolution gain (from 30-m to 30-cm), surpassing prior methods by an order of magnitude. Evaluations across three diverse U.S. landscapes show robust generalization, capturing urban structures and fine-scale terrain features with < 5 m MAE relative to LiDAR, improving over SRTM by up to 18%. Hydrological analysis confirms suitability for hazard and environmental studies. We demonstrate scalability by applying the framework to large regions in the U.S. and Israel. All code and pretrained models are publicly available at: https://osherr1996.github.io/prompt2dem_propage/.
>
---
#### [replaced 048] A Lightweight and Robust Framework for Real-Time Colorectal Polyp Detection Using LOF-Based Preprocessing and YOLO-v11n
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.10864v2](http://arxiv.org/pdf/2507.10864v2)**

> **作者:** Saadat Behzadi; Danial Sharifrazi; Bita Mesbahzadeh; Javad Hassannataj Joloudari; Roohallah Alizadehsani
>
> **摘要:** Objectives: Timely and accurate detection of colorectal polyps plays a crucial role in diagnosing and preventing colorectal cancer, a major cause of mortality worldwide. This study introduces a new, lightweight, and efficient framework for polyp detection that combines the Local Outlier Factor (LOF) algorithm for filtering noisy data with the YOLO-v11n deep learning model. Study design: An experimental study leveraging deep learning and outlier removal techniques across multiple public datasets. Methods: The proposed approach was tested on five diverse and publicly available datasets: CVC-ColonDB, CVC-ClinicDB, Kvasir-SEG, ETIS, and EndoScene. Since these datasets originally lacked bounding box annotations, we converted their segmentation masks into suitable detection labels. To enhance the robustness and generalizability of our model, we apply 5-fold cross-validation and remove anomalous samples using the LOF method configured with 30 neighbors and a contamination ratio of 5%. Cleaned data are then fed into YOLO-v11n, a fast and resource-efficient object detection architecture optimized for real-time applications. We train the model using a combination of modern augmentation strategies to improve detection accuracy under diverse conditions. Results: Our approach significantly improves polyp localization performance, achieving a precision of 95.83%, recall of 91.85%, F1-score of 93.48%, mAP@0.5 of 96.48%, and mAP@0.5:0.95 of 77.75%. Compared to previous YOLO-based methods, our model demonstrates enhanced accuracy and efficiency. Conclusions: These results suggest that the proposed method is well-suited for real-time colonoscopy support in clinical settings. Overall, the study underscores how crucial data preprocessing and model efficiency are when designing effective AI systems for medical imaging.
>
---
#### [replaced 049] Influence of High-Performance Image-to-Image Translation Networks on Clinical Visual Assessment and Outcome Prediction: Utilizing Ultrasound to MRI Translation in Prostate Cancer
- **分类: eess.IV; cs.CV; physics.bio-ph; 14J60 (Primary) 14F05, 14J26 (Secondary); F.2.2**

- **链接: [http://arxiv.org/pdf/2501.18109v2](http://arxiv.org/pdf/2501.18109v2)**

> **作者:** Mohammad R. Salmanpour; Amin Mousavi; Yixi Xu; William B Weeks; Ilker Hacihaliloglu
>
> **备注:** 9 pages, 4 figures and 1 table
>
> **摘要:** Purpose: This study examines the core traits of image-to-image translation (I2I) networks, focusing on their effectiveness and adaptability in everyday clinical settings. Methods: We have analyzed data from 794 patients diagnosed with prostate cancer (PCa), using ten prominent 2D/3D I2I networks to convert ultrasound (US) images into MRI scans. We also introduced a new analysis of Radiomic features (RF) via the Spearman correlation coefficient to explore whether networks with high performance (SSIM>85%) could detect subtle RFs. Our study further examined synthetic images by 7 invited physicians. As a final evaluation study, we have investigated the improvement that are achieved using the synthetic MRI data on two traditional machine learning and one deep learning method. Results: In quantitative assessment, 2D-Pix2Pix network substantially outperformed the other 7 networks, with an average SSIM~0.855. The RF analysis revealed that 76 out of 186 RFs were identified using the 2D-Pix2Pix algorithm alone, although half of the RFs were lost during the translation process. A detailed qualitative review by 7 medical doctors noted a deficiency in low-level feature recognition in I2I tasks. Furthermore, the study found that synthesized image-based classification outperformed US image-based classification with an average accuracy and AUC~0.93. Conclusion: This study showed that while 2D-Pix2Pix outperformed cutting-edge networks in low-level feature discovery and overall error and similarity metrics, it still requires improvement in low-level feature performance, as highlighted by Group 3. Further, the study found using synthetic image-based classification outperformed original US image-based methods.
>
---
#### [replaced 050] Texture or Semantics? Vision-Language Models Get Lost in Font Recognition
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23768v2](http://arxiv.org/pdf/2503.23768v2)**

> **作者:** Zhecheng Li; Guoxian Song; Yujun Cai; Zhen Xiong; Junsong Yuan; Yiwei Wang
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Modern Vision-Language Models (VLMs) exhibit remarkable visual and linguistic capabilities, achieving impressive performance in various tasks such as image recognition and object localization. However, their effectiveness in fine-grained tasks remains an open question. In everyday scenarios, individuals encountering design materials, such as magazines, typography tutorials, research papers, or branding content, may wish to identify aesthetically pleasing fonts used in the text. Given their multimodal capabilities and free accessibility, many VLMs are often considered potential tools for font recognition. This raises a fundamental question: Do VLMs truly possess the capability to recognize fonts? To investigate this, we introduce the Font Recognition Benchmark (FRB), a compact and well-structured dataset comprising 15 commonly used fonts. FRB includes two versions: (i) an easy version, where 10 sentences are rendered in different fonts, and (ii) a hard version, where each text sample consists of the names of the 15 fonts themselves, introducing a stroop effect that challenges model perception. Through extensive evaluation of various VLMs on font recognition tasks, we arrive at the following key findings: (i) Current VLMs exhibit limited font recognition capabilities, with many state-of-the-art models failing to achieve satisfactory performance and being easily affected by the stroop effect introduced by textual information. (ii) Few-shot learning and Chain-of-Thought (CoT) prompting provide minimal benefits in improving font recognition accuracy across different VLMs. (iii) Attention analysis sheds light on the inherent limitations of VLMs in capturing semantic features.
>
---
#### [replaced 051] Resource-Efficient Affordance Grounding with Complementary Depth and Semantic Prompts
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.02600v2](http://arxiv.org/pdf/2503.02600v2)**

> **作者:** Yizhou Huang; Fan Yang; Guoliang Zhu; Gen Li; Hao Shi; Yukun Zuo; Wenrui Chen; Zhiyong Li; Kailun Yang
>
> **备注:** Accepted to IROS 2025. The source code will be made publicly available at https://github.com/DAWDSE/BiT-Align
>
> **摘要:** Affordance refers to the functional properties that an agent perceives and utilizes from its environment, and is key perceptual information required for robots to perform actions. This information is rich and multimodal in nature. Existing multimodal affordance methods face limitations in extracting useful information, mainly due to simple structural designs, basic fusion methods, and large model parameters, making it difficult to meet the performance requirements for practical deployment. To address these issues, this paper proposes the BiT-Align image-depth-text affordance mapping framework. The framework includes a Bypass Prompt Module (BPM) and a Text Feature Guidance (TFG) attention selection mechanism. BPM integrates the auxiliary modality depth image directly as a prompt to the primary modality RGB image, embedding it into the primary modality encoder without introducing additional encoders. This reduces the model's parameter count and effectively improves functional region localization accuracy. The TFG mechanism guides the selection and enhancement of attention heads in the image encoder using textual features, improving the understanding of affordance characteristics. Experimental results demonstrate that the proposed method achieves significant performance improvements on public AGD20K and HICO-IIF datasets. On the AGD20K dataset, compared with the current state-of-the-art method, we achieve a 6.0% improvement in the KLD metric, while reducing model parameters by 88.8%, demonstrating practical application values. The source code will be made publicly available at https://github.com/DAWDSE/BiT-Align.
>
---
#### [replaced 052] Defective Convolutional Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1911.08432v3](http://arxiv.org/pdf/1911.08432v3)**

> **作者:** Tiange Luo; Tianle Cai; Mengxiao Zhang; Siyu Chen; Di He; Liwei Wang
>
> **摘要:** Robustness of convolutional neural networks (CNNs) has gained in importance on account of adversarial examples, i.e., inputs added as well-designed perturbations that are imperceptible to humans but can cause the model to predict incorrectly. Recent research suggests that the noises in adversarial examples break the textural structure, which eventually leads to wrong predictions. To mitigate the threat of such adversarial attacks, we propose defective convolutional networks that make predictions relying less on textural information but more on shape information by properly integrating defective convolutional layers into standard CNNs. The defective convolutional layers contain defective neurons whose activations are set to be a constant function. As defective neurons contain no information and are far different from standard neurons in its spatial neighborhood, the textural features cannot be accurately extracted, and so the model has to seek other features for classification, such as the shape. We show extensive evidence to justify our proposal and demonstrate that defective CNNs can defense against black-box attacks better than standard CNNs. In particular, they achieve state-of-the-art performance against transfer-based attacks without any adversarial training being applied.
>
---
#### [replaced 053] VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20289v2](http://arxiv.org/pdf/2505.20289v2)**

> **作者:** Zeyi Huang; Yuyang Ji; Anirudh Sundara Rajan; Zefan Cai; Wen Xiao; Haohan Wang; Junjie Hu; Yong Jae Lee
>
> **摘要:** We introduce VisTA, a new reinforcement learning framework that empowers visual agents to dynamically explore, select, and combine tools from a diverse library based on empirical performance. Existing methods for tool-augmented reasoning either rely on training-free prompting or large-scale fine-tuning; both lack active tool exploration and typically assume limited tool diversity, and fine-tuning methods additionally demand extensive human supervision. In contrast, VisTA leverages end-to-end reinforcement learning to iteratively refine sophisticated, query-specific tool selection strategies, using task outcomes as feedback signals. Through Group Relative Policy Optimization (GRPO), our framework enables an agent to autonomously discover effective tool-selection pathways without requiring explicit reasoning supervision. Experiments on the ChartQA, Geometry3K, and BlindTest benchmarks demonstrate that VisTA achieves substantial performance gains over training-free baselines, especially on out-of-distribution examples. These results highlight VisTA's ability to enhance generalization, adaptively utilize diverse tools, and pave the way for flexible, experience-driven visual reasoning systems.
>
---
#### [replaced 054] MORDA: A Synthetic Dataset to Facilitate Adaptation of Object Detectors to Unseen Real-target Domain While Preserving Performance on Real-source Domain
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.04950v3](http://arxiv.org/pdf/2501.04950v3)**

> **作者:** Hojun Lim; Heecheol Yoo; Jinwoo Lee; Seungmin Jeon; Hyeongseok Jeon
>
> **备注:** 7 pages, 6 figures, 4 tables, This work has been accepted for publication in IEEE ICRA 2025. The final published version will be available via IEEE Xplore
>
> **摘要:** Deep neural network (DNN) based perception models are indispensable in the development of autonomous vehicles (AVs). However, their reliance on large-scale, high-quality data is broadly recognized as a burdensome necessity due to the substantial cost of data acquisition and labeling. Further, the issue is not a one-time concern, as AVs might need a new dataset if they are to be deployed to another region (real-target domain) that the in-hand dataset within the real-source domain cannot incorporate. To mitigate this burden, we propose leveraging synthetic environments as an auxiliary domain where the characteristics of real domains are reproduced. This approach could enable indirect experience about the real-target domain in a time- and cost-effective manner. As a practical demonstration of our methodology, nuScenes and South Korea are employed to represent real-source and real-target domains, respectively. That means we construct digital twins for several regions of South Korea, and the data-acquisition framework of nuScenes is reproduced. Blending the aforementioned components within a simulator allows us to obtain a synthetic-fusion domain in which we forge our novel driving dataset, MORDA: Mixture Of Real-domain characteristics for synthetic-data-assisted Domain Adaptation. To verify the value of synthetic features that MORDA provides in learning about driving environments of South Korea, 2D/3D detectors are trained solely on a combination of nuScenes and MORDA. Afterward, their performance is evaluated on the unforeseen real-world dataset (AI-Hub) collected in South Korea. Our experiments present that MORDA can significantly improve mean Average Precision (mAP) on AI-Hub dataset while that on nuScenes is retained or slightly enhanced.
>
---
#### [replaced 055] Synchronizing Task Behavior: Aligning Multiple Tasks during Test-Time Training
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07778v2](http://arxiv.org/pdf/2507.07778v2)**

> **作者:** Wooseong Jeong; Jegyeong Cho; Youngho Yoon; Kuk-Jin Yoon
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Generalizing neural networks to unseen target domains is a significant challenge in real-world deployments. Test-time training (TTT) addresses this by using an auxiliary self-supervised task to reduce the domain gap caused by distribution shifts between the source and target. However, we find that when models are required to perform multiple tasks under domain shifts, conventional TTT methods suffer from unsynchronized task behavior, where the adaptation steps needed for optimal performance in one task may not align with the requirements of other tasks. To address this, we propose a novel TTT approach called Synchronizing Tasks for Test-time Training (S4T), which enables the concurrent handling of multiple tasks. The core idea behind S4T is that predicting task relations across domain shifts is key to synchronizing tasks during test time. To validate our approach, we apply S4T to conventional multi-task benchmarks, integrating it with traditional TTT protocols. Our empirical results show that S4T outperforms state-of-the-art TTT methods across various benchmarks.
>
---
#### [replaced 056] Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04790v2](http://arxiv.org/pdf/2507.04790v2)**

> **作者:** Giwon Lee; Wooseong Jeong; Daehee Park; Jaewoo Jeong; Kuk-Jin Yoon
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Motion planning is a crucial component of autonomous robot driving. While various trajectory datasets exist, effectively utilizing them for a target domain remains challenging due to differences in agent interactions and environmental characteristics. Conventional approaches, such as domain adaptation or ensemble learning, leverage multiple source datasets but suffer from domain imbalance, catastrophic forgetting, and high computational costs. To address these challenges, we propose Interaction-Merged Motion Planning (IMMP), a novel approach that leverages parameter checkpoints trained on different domains during adaptation to the target domain. IMMP follows a two-step process: pre-merging to capture agent behaviors and interactions, sufficiently extracting diverse information from the source domain, followed by merging to construct an adaptable model that efficiently transfers diverse interactions to the target domain. Our method is evaluated on various planning benchmarks and models, demonstrating superior performance compared to conventional approaches.
>
---
#### [replaced 057] A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.21517v2](http://arxiv.org/pdf/2407.21517v2)**

> **作者:** Miao Cao; Lishun Wang; Huan Wang; Xin Yuan
>
> **备注:** 18 pages, Accepted by ECCV 2024
>
> **摘要:** Video Snapshot Compressive Imaging (SCI) aims to use a low-speed 2D camera to capture high-speed scene as snapshot compressed measurements, followed by a reconstruction algorithm to reconstruct the high-speed video frames. State-of-the-art (SOTA) deep learning-based algorithms have achieved impressive performance, yet with heavy computational workload. Network quantization is a promising way to reduce computational cost. However, a direct low-bit quantization will bring large performance drop. To address this challenge, in this paper, we propose a simple low-bit quantization framework (dubbed Q-SCI) for the end-to-end deep learning-based video SCI reconstruction methods which usually consist of a feature extraction, feature enhancement, and video reconstruction module. Specifically, we first design a high-quality feature extraction module and a precise video reconstruction module to extract and propagate high-quality features in the low-bit quantized model. In addition, to alleviate the information distortion of the Transformer branch in the quantized feature enhancement module, we introduce a shift operation on the query and key distributions to further bridge the performance gap. Comprehensive experimental results manifest that our Q-SCI framework can achieve superior performance, e.g., 4-bit quantized EfficientSCI-S derived by our Q-SCI framework can theoretically accelerate the real-valued EfficientSCI-S by 7.8X with only 2.3% performance gap on the simulation testing datasets. Code is available at https://github.com/mcao92/QuantizedSCI.
>
---
#### [replaced 058] Continuous sPatial-Temporal Deformable Image Registration (CPT-DIR) for motion modelling in radiotherapy: beyond classic voxel-based methods
- **分类: physics.med-ph; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.00430v2](http://arxiv.org/pdf/2405.00430v2)**

> **作者:** Xia Li; Runzhao Yang; Muheng Li; Xiangtai Li; Antony J. Lomax; Joachim M. Buhmann; Ye Zhang
>
> **摘要:** Deformable image registration (DIR) is a crucial tool in radiotherapy for analyzing anatomical changes and motion patterns. Current DIR implementations rely on discrete volumetric motion representation, which often leads to compromised accuracy and uncertainty when handling significant anatomical changes and sliding boundaries. This limitation affects the reliability of subsequent contour propagation and dose accumulation procedures, particularly in regions with complex anatomical interfaces such as the lung-chest wall boundary. Given that organ motion is inherently a continuous process in both space and time, we aimed to develop a model that preserves these fundamental properties. Drawing inspiration from fluid mechanics, we propose a novel approach using implicit neural representation (INR) for continuous modeling of patient anatomical motion. This approach ensures spatial and temporal continuity while effectively unifying Eulerian and Lagrangian specifications to enable natural continuous motion modeling and frame interpolation. The integration of these specifications provides a more comprehensive understanding of anatomical deformation patterns. By leveraging the continuous representations, the CPT-DIR method significantly enhances registration and interpolation accuracy, automation, and speed. The method demonstrates superior performance in landmark and contour precision, particularly in challenging anatomical regions, representing a substantial advancement over conventional approaches in deformable image registration. The improved efficiency and accuracy of CPT-DIR make it particularly suitable for real-time adaptive radiotherapy applications.
>
---
#### [replaced 059] Advancing Textual Prompt Learning with Anchored Attributes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09442v4](http://arxiv.org/pdf/2412.09442v4)**

> **作者:** Zheng Li; Yibing Song; Ming-Ming Cheng; Xiang Li; Jian Yang
>
> **备注:** ICCV 2025. Code: https://github.com/zhengli97/ATPrompt. Project Page: https://zhengli97.github.io/ATPrompt/
>
> **摘要:** Textual-based prompt learning methods primarily employ multiple learnable soft prompts and hard class tokens in a cascading manner as text inputs, aiming to align image and text (category) spaces for downstream tasks. However, current training is restricted to aligning images with predefined known categories and cannot be associated with unknown categories. In this work, we propose utilizing universal attributes as a bridge to enhance the alignment between images and unknown categories. Specifically, we introduce an Attribute-anchored Textual Prompt learning method for vision-language models, named ATPrompt. This approach expands the learning space of soft prompts from the original one-dimensional category level into the multi-dimensional attribute level by incorporating multiple attribute tokens into the learnable soft prompts. Through this modification, we transform the text prompt from a category-centric form to an attribute-category hybrid form. Additionally, we introduce a straightforward differentiable attribute search method to identify representative and suitable attributes for downstream tasks. As an easy-to-use plug-in technique, ATPrompt can seamlessly replace the existing basic prompt format in textual-based methods, providing general improvements at a negligible computational cost. Extensive experiments across 11 datasets validate the effectiveness of our method. Code is publicly available at https://github.com/zhengli97/ATPrompt.
>
---
#### [replaced 060] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13891v2](http://arxiv.org/pdf/2507.13891v2)**

> **作者:** Yu Wei; Jiahui Zhang; Xiaoqin Zhang; Ling Shao; Shijian Lu
>
> **摘要:** COLMAP-free 3D Gaussian Splatting (3D-GS) has recently attracted increasing attention due to its remarkable performance in reconstructing high-quality 3D scenes from unposed images or videos. However, it often struggles to handle scenes with complex camera trajectories as featured by drastic rotation and translation across adjacent camera views, leading to degraded estimation of camera poses and further local minima in joint optimization of camera poses and 3D-GS. We propose PCR-GS, an innovative COLMAP-free 3DGS technique that achieves superior 3D scene modeling and camera pose estimation via camera pose co-regularization. PCR-GS achieves regularization from two perspectives. The first is feature reprojection regularization which extracts view-robust DINO features from adjacent camera views and aligns their semantic information for camera pose regularization. The second is wavelet-based frequency regularization which exploits discrepancy in high-frequency details to further optimize the rotation matrix in camera poses. Extensive experiments over multiple real-world scenes show that the proposed PCR-GS achieves superior pose-free 3D-GS scene modeling under dramatic changes of camera trajectories.
>
---
#### [replaced 061] Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17769v2](http://arxiv.org/pdf/2411.17769v2)**

> **作者:** Xinyu Hou; Zongsheng Yue; Xiaoming Li; Chen Change Loy
>
> **备注:** ICCV 2025 Camera Ready. Project page: https://itsmag11.github.io/Omegance/
>
> **摘要:** In this work, we show that we only need a single parameter $\omega$ to effectively control granularity in diffusion-based synthesis. This parameter is incorporated during the denoising steps of the diffusion model's reverse process. This simple approach does not require model retraining or architectural modifications and incurs negligible computational overhead, yet enables precise control over the level of details in the generated outputs. Moreover, spatial masks or denoising schedules with varying $\omega$ values can be applied to achieve region-specific or timestep-specific granularity control. External control signals or reference images can guide the creation of precise $\omega$ masks, allowing targeted granularity adjustments. Despite its simplicity, the method demonstrates impressive performance across various image and video synthesis tasks and is adaptable to advanced diffusion models. The code is available at https://github.com/itsmag11/Omegance.
>
---
#### [replaced 062] CBAGAN-RRT: Convolutional Block Attention Generative Adversarial Network for Sampling-Based Path Planning
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2305.10442v3](http://arxiv.org/pdf/2305.10442v3)**

> **作者:** Abhinav Sagar; Sai Teja Gilukara
>
> **摘要:** Sampling-based path planning algorithms play an important role in autonomous robotics. However, a common problem among these algorithms is that the initial path generated is not optimal, and the convergence is too slow for real-world applications. In this paper, we propose a novel image-based learning algorithm using a Convolutional Block Attention Generative Adversarial Network (CBAGAN-RRT) with a combination of spatial and channel attention and a novel loss function to design the heuristics, find a better optimal path, and improve the convergence of the algorithm, both concerning time and speed. The probability distribution of the paths generated from our GAN model is used to guide the sampling process for the RRT algorithm. We demonstrate that our algorithm outperforms the previous state-of-the-art algorithms using both the image quality generation metrics, like IOU Score, Dice Score, FID score, and path planning metrics like time cost and the number of nodes. Ablation studies show the effectiveness of various components in our network architecture. The advantage of our approach is that we can avoid the complicated preprocessing in the state space, our model can be generalized to complex environments like those containing turns and narrow passages without loss of accuracy, and our model can be easily integrated with other sampling-based path planning algorithms.
>
---
#### [replaced 063] InteractPro: A Unified Framework for Motion-Aware Image Composition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.10090v2](http://arxiv.org/pdf/2409.10090v2)**

> **作者:** Weijing Tao; Xiaofeng Yang; Miaomiao Cui; Guosheng Lin
>
> **摘要:** We introduce InteractPro, a comprehensive framework for dynamic motion-aware image composition. At its core is InteractPlan, an intelligent planner that leverages a Large Vision Language Model (LVLM) for scenario analysis and object placement, determining the optimal composition strategy to achieve realistic motion effects. Based on each scenario, InteractPlan selects between our two specialized modules: InteractPhys and InteractMotion. InteractPhys employs an enhanced Material Point Method (MPM)-based simulation to produce physically faithful and controllable object-scene interactions, capturing diverse and abstract events that require true physical modeling. InteractMotion, in contrast, is a training-free method based on pretrained video diffusion. Traditional composition approaches suffer from two major limitations: requiring manual planning for object placement and generating static, motionless outputs. By unifying simulation-based and diffusion-based methods under planner guidance, InteractPro overcomes these challenges, ensuring richly motion-aware compositions. Extensive quantitative and qualitative evaluations demonstrate InteractPro's effectiveness in producing controllable, and coherent compositions across varied scenarios.
>
---
#### [replaced 064] HMID-Net: An Exploration of Masked Image Modeling and Knowledge Distillation in Hyperbolic Space
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.09487v2](http://arxiv.org/pdf/2507.09487v2)**

> **作者:** Changli Wang; Fang Yin; Jiafeng Liu; Rui Wu
>
> **备注:** Modified the abstract and reformatted it using latex
>
> **摘要:** Visual and semantic concepts are often structured in a hierarchical manner. For instance, textual concept `cat' entails all images of cats. A recent study, MERU, successfully adapts multimodal learning techniques from Euclidean space to hyperbolic space, effectively capturing the visual-semantic hierarchy. However, a critical question remains: how can we more efficiently train a model to capture and leverage this hierarchy? In this paper, we propose the Hyperbolic Masked Image and Distillation Network (HMID-Net), a novel and efficient method that integrates Masked Image Modeling (MIM) and knowledge distillation techniques within hyperbolic space. To the best of our knowledge, this is the first approach to leverage MIM and knowledge distillation in hyperbolic space to train highly efficient models. In addition, we introduce a distillation loss function specifically designed to facilitate effective knowledge transfer in hyperbolic space. Our experiments demonstrate that MIM and knowledge distillation techniques in hyperbolic space can achieve the same remarkable success as in Euclidean space. Extensive evaluations show that our method excels across a wide range of downstream tasks, significantly outperforming existing models like MERU and CLIP in both image classification and retrieval.
>
---
#### [replaced 065] G-DexGrasp: Generalizable Dexterous Grasping Synthesis Via Part-Aware Prior Retrieval and Prior-Assisted Generation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.19457v2](http://arxiv.org/pdf/2503.19457v2)**

> **作者:** Juntao Jian; Xiuping Liu; Zixuan Chen; Manyi Li; Jian Liu; Ruizhen Hu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent advances in dexterous grasping synthesis have demonstrated significant progress in producing reasonable and plausible grasps for many task purposes. But it remains challenging to generalize to unseen object categories and diverse task instructions. In this paper, we propose G-DexGrasp, a retrieval-augmented generation approach that can produce high-quality dexterous hand configurations for unseen object categories and language-based task instructions. The key is to retrieve generalizable grasping priors, including the fine-grained contact part and the affordance-related distribution of relevant grasping instances, for the following synthesis pipeline. Specifically, the fine-grained contact part and affordance act as generalizable guidance to infer reasonable grasping configurations for unseen objects with a generative model, while the relevant grasping distribution plays as regularization to guarantee the plausibility of synthesized grasps during the subsequent refinement optimization. Our comparison experiments validate the effectiveness of our key designs for generalization and demonstrate the remarkable performance against the existing approaches. Project page: https://g-dexgrasp.github.io/
>
---
#### [replaced 066] WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.02247v5](http://arxiv.org/pdf/2503.02247v5)**

> **作者:** Dujun Nie; Xianda Guo; Yiqun Duan; Ruijun Zhang; Long Chen
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Object Goal Navigation-requiring an agent to locate a specific object in an unseen environment-remains a core challenge in embodied AI. Although recent progress in Vision-Language Model (VLM)-based agents has demonstrated promising perception and decision-making abilities through prompting, none has yet established a fully modular world model design that reduces risky and costly interactions with the environment by predicting the future state of the world. We introduce WMNav, a novel World Model-based Navigation framework powered by Vision-Language Models (VLMs). It predicts possible outcomes of decisions and builds memories to provide feedback to the policy module. To retain the predicted state of the environment, WMNav proposes the online maintained Curiosity Value Map as part of the world model memory to provide dynamic configuration for navigation policy. By decomposing according to a human-like thinking process, WMNav effectively alleviates the impact of model hallucination by making decisions based on the feedback difference between the world model plan and observation. To further boost efficiency, we implement a two-stage action proposer strategy: broad exploration followed by precise localization. Extensive evaluation on HM3D and MP3D validates WMNav surpasses existing zero-shot benchmarks in both success rate and exploration efficiency (absolute improvement: +3.2% SR and +3.2% SPL on HM3D, +13.5% SR and +1.1% SPL on MP3D). Project page: https://b0b8k1ng.github.io/WMNav/.
>
---
#### [replaced 067] HORT: Monocular Hand-held Objects Reconstruction with Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21313v2](http://arxiv.org/pdf/2503.21313v2)**

> **作者:** Zerui Chen; Rolandos Alexandros Potamias; Shizhe Chen; Cordelia Schmid
>
> **备注:** Accepted by ICCV 2025. Project Page: https://zerchen.github.io/projects/hort.html
>
> **摘要:** Reconstructing hand-held objects in 3D from monocular images remains a significant challenge in computer vision. Most existing approaches rely on implicit 3D representations, which produce overly smooth reconstructions and are time-consuming to generate explicit 3D shapes. While more recent methods directly reconstruct point clouds with diffusion models, the multi-step denoising makes high-resolution reconstruction inefficient. To address these limitations, we propose a transformer-based model to efficiently reconstruct dense 3D point clouds of hand-held objects. Our method follows a coarse-to-fine strategy, first generating a sparse point cloud from the image and progressively refining it into a dense representation using pixel-aligned image features. To enhance reconstruction accuracy, we integrate image features with 3D hand geometry to jointly predict the object point cloud and its pose relative to the hand. Our model is trained end-to-end for optimal performance. Experimental results on both synthetic and real datasets demonstrate that our method achieves state-of-the-art accuracy with much faster inference speed, while generalizing well to in-the-wild images.
>
---
#### [replaced 068] Dance Like a Chicken: Low-Rank Stylization for Human Motion Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19557v3](http://arxiv.org/pdf/2503.19557v3)**

> **作者:** Haim Sawdayee; Chuan Guo; Guy Tevet; Bing Zhou; Jian Wang; Amit H. Bermano
>
> **备注:** Project page at https://haimsaw.github.io/LoRA-MDM/
>
> **摘要:** Text-to-motion generative models span a wide range of 3D human actions but struggle with nuanced stylistic attributes such as a "Chicken" style. Due to the scarcity of style-specific data, existing approaches pull the generative prior towards a reference style, which often results in out-of-distribution low quality generations. In this work, we introduce LoRA-MDM, a lightweight framework for motion stylization that generalizes to complex actions while maintaining editability. Our key insight is that adapting the generative prior to include the style, while preserving its overall distribution, is more effective than modifying each individual motion during generation. Building on this idea, LoRA-MDM learns to adapt the prior to include the reference style using only a few samples. The style can then be used in the context of different textual prompts for generation. The low-rank adaptation shifts the motion manifold in a semantically meaningful way, enabling realistic style infusion even for actions not present in the reference samples. Moreover, preserving the distribution structure enables advanced operations such as style blending and motion editing. We compare LoRA-MDM to state-of-the-art stylized motion generation methods and demonstrate a favorable balance between text fidelity and style consistency.
>
---
#### [replaced 069] Studying Classifier(-Free) Guidance From a Classifier-Centric Perspective
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10638v2](http://arxiv.org/pdf/2503.10638v2)**

> **作者:** Xiaoming Zhao; Alexander G. Schwing
>
> **备注:** v2: added derivation details in Appendix A
>
> **摘要:** Classifier-free guidance has become a staple for conditional generation with denoising diffusion models. However, a comprehensive understanding of classifier-free guidance is still missing. In this work, we carry out an empirical study to provide a fresh perspective on classifier-free guidance. Concretely, instead of solely focusing on classifier-free guidance, we trace back to the root, i.e., classifier guidance, pinpoint the key assumption for the derivation, and conduct a systematic study to understand the role of the classifier. We find that both classifier guidance and classifier-free guidance achieve conditional generation by pushing the denoising diffusion trajectories away from decision boundaries, i.e., areas where conditional information is usually entangled and is hard to learn. Based on this classifier-centric understanding, we propose a generic postprocessing step built upon flow-matching to shrink the gap between the learned distribution for a pre-trained denoising diffusion model and the real data distribution, majorly around the decision boundaries. Experiments on various datasets verify the effectiveness of the proposed approach.
>
---
#### [replaced 070] BrainNetMLP: An Efficient and Effective Baseline for Functional Brain Network Classification
- **分类: q-bio.NC; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11538v2](http://arxiv.org/pdf/2505.11538v2)**

> **作者:** Jiacheng Hou; Zhenjie Song; Ercan Engin Kuruoglu
>
> **备注:** Accepted for oral presentation at the 1st MICCAI Workshop on Efficient Medical AI
>
> **摘要:** Recent studies have made great progress in functional brain network classification by modeling the brain as a network of Regions of Interest (ROIs) and leveraging their connections to understand brain functionality and diagnose mental disorders. Various deep learning architectures, including Convolutional Neural Networks, Graph Neural Networks, and the recent Transformer, have been developed. However, despite the increasing complexity of these models, the performance gain has not been as salient. This raises a question: Does increasing model complexity necessarily lead to higher classification accuracy? In this paper, we revisit the simplest deep learning architecture, the Multi-Layer Perceptron (MLP), and propose a pure MLP-based method, named BrainNetMLP, for functional brain network classification, which capitalizes on the advantages of MLP, including efficient computation and fewer parameters. Moreover, BrainNetMLP incorporates a dual-branch structure to jointly capture both spatial connectivity and spectral information, enabling precise spatiotemporal feature fusion. We evaluate our proposed BrainNetMLP on two public and popular brain network classification datasets, the Human Connectome Project (HCP) and the Autism Brain Imaging Data Exchange (ABIDE). Experimental results demonstrate pure MLP-based methods can achieve state-of-the-art performance, revealing the potential of MLP-based models as more efficient yet effective alternatives in functional brain network classification. The code will be available at https://github.com/JayceonHo/BrainNetMLP.
>
---
#### [replaced 071] Towards Cross-modal Retrieval in Chinese Cultural Heritage Documents: Dataset and Solution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10921v2](http://arxiv.org/pdf/2505.10921v2)**

> **作者:** Junyi Yuan; Jian Zhang; Fangyu Wu; Dongming Lu; Huanda Lu; Qiufeng Wang
>
> **摘要:** China has a long and rich history, encompassing a vast cultural heritage that includes diverse multimodal information, such as silk patterns, Dunhuang murals, and their associated historical narratives. Cross-modal retrieval plays a pivotal role in understanding and interpreting Chinese cultural heritage by bridging visual and textual modalities to enable accurate text-to-image and image-to-text retrieval. However, despite the growing interest in multimodal research, there is a lack of specialized datasets dedicated to Chinese cultural heritage, limiting the development and evaluation of cross-modal learning models in this domain. To address this gap, we propose a multimodal dataset named CulTi, which contains 5,726 image-text pairs extracted from two series of professional documents, respectively related to ancient Chinese silk and Dunhuang murals. Compared to existing general-domain multimodal datasets, CulTi presents a challenge for cross-modal retrieval: the difficulty of local alignment between intricate decorative motifs and specialized textual descriptions. To address this challenge, we propose LACLIP, a training-free local alignment strategy built upon a fine-tuned Chinese-CLIP. LACLIP enhances the alignment of global textual descriptions with local visual regions by computing weighted similarity scores during inference. Experimental results on CulTi demonstrate that LACLIP significantly outperforms existing models in cross-modal retrieval, particularly in handling fine-grained semantic associations within Chinese cultural heritage.
>
---
#### [replaced 072] FlexiClip: Locality-Preserving Free-Form Character Animation
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2501.08676v2](http://arxiv.org/pdf/2501.08676v2)**

> **作者:** Anant Khandelwal
>
> **备注:** 13 pages, 4 figures, 7 tables, Accepted in ICML 2025, https://openreview.net/forum?id=xtxCM4XZ82
>
> **摘要:** Animating clipart images with seamless motion while maintaining visual fidelity and temporal coherence presents significant challenges. Existing methods, such as AniClipart, effectively model spatial deformations but often fail to ensure smooth temporal transitions, resulting in artifacts like abrupt motions and geometric distortions. Similarly, text-to-video (T2V) and image-to-video (I2V) models struggle to handle clipart due to the mismatch in statistical properties between natural video and clipart styles. This paper introduces FlexiClip, a novel approach designed to overcome these limitations by addressing the intertwined challenges of temporal consistency and geometric integrity. FlexiClip extends traditional B\'ezier curve-based trajectory modeling with key innovations: temporal Jacobians to correct motion dynamics incrementally, continuous-time modeling via probability flow ODEs (pfODEs) to mitigate temporal noise, and a flow matching loss inspired by GFlowNet principles to optimize smooth motion transitions. These enhancements ensure coherent animations across complex scenarios involving rapid movements and non-rigid deformations. Extensive experiments validate the effectiveness of FlexiClip in generating animations that are not only smooth and natural but also structurally consistent across diverse clipart types, including humans and animals. By integrating spatial and temporal modeling with pre-trained video diffusion models, FlexiClip sets a new standard for high-quality clipart animation, offering robust performance across a wide range of visual content. Project Page: https://creative-gen.github.io/flexiclip.github.io/
>
---
#### [replaced 073] Leveraging Spatial Context for Positive Pair Sampling in Histopathology Image Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05170v2](http://arxiv.org/pdf/2503.05170v2)**

> **作者:** Willmer Rafell Quinones Robles; Sakonporn Noree; Young Sin Ko; Bryan Wong; Jongwoo Kim; Mun Yong Yi
>
> **摘要:** Deep learning has shown strong potential in cancer classification from whole-slide images (WSIs), but the need for extensive expert annotations often limits its success. Annotation-free approaches, such as multiple instance learning (MIL) and self-supervised learning (SSL), have emerged as promising alternatives to traditional annotation-based methods. However, conventional SSL methods typically rely on synthetic data augmentations, which may fail to capture the spatial structure critical to histopathology. In this work, we propose a spatial context-driven positive pair sampling strategy that enhances SSL by leveraging the morphological coherence of spatially adjacent patches within WSIs. Our method is modular and compatible with established joint embedding SSL frameworks, including Barlow Twins, BYOL, VICReg, and DINOv2. We evaluate its effectiveness on both slide-level classification using MIL and patch-level linear probing. Experiments across four datasets demonstrate consistent performance improvements, with accuracy gains of 5\% to 10\% compared to standard augmentation-based sampling. These findings highlight the value of spatial context in improving representation learning for computational pathology and provide a biologically meaningful enhancement for pretraining models in annotation-limited settings. The code is available at https://anonymous.4open.science/r/contextual-pairs-E72F/.
>
---
#### [replaced 074] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00721v2](http://arxiv.org/pdf/2507.00721v2)**

> **作者:** Xiao Zhang; Fei Wei; Yong Wang; Wenda Zhao; Feiyi Li; Xiangxiang Chu
>
> **备注:** ICCV2025
>
> **摘要:** Zero-shot domain adaptation (ZSDA) presents substantial challenges due to the lack of images in the target domain. Previous approaches leverage Vision-Language Models (VLMs) to tackle this challenge, exploiting their zero-shot learning capabilities. However, these methods primarily address domain distribution shifts and overlook the misalignment between the detection task and VLMs, which rely on manually crafted prompts. To overcome these limitations, we propose the unified prompt and representation enhancement (UPRE) framework, which jointly optimizes both textual prompts and visual representations. Specifically, our approach introduces a multi-view domain prompt that combines linguistic domain priors with detection-specific knowledge, and a visual representation enhancement module that produces domain style variations. Furthermore, we introduce multi-level enhancement strategies, including relative domain distance and positive-negative separation, which align multi-modal representations at the image level and capture diverse visual representations at the instance level, respectively. Extensive experiments conducted on nine benchmark datasets demonstrate the superior performance of our framework in ZSDA detection scenarios. Code is available at https://github.com/AMAP-ML/UPRE.
>
---
#### [replaced 075] GURecon: Learning Detailed 3D Geometric Uncertainties for Neural Surface Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.14939v3](http://arxiv.org/pdf/2412.14939v3)**

> **作者:** Zesong Yang; Ru Zhang; Jiale Shi; Zixiang Ai; Boming Zhao; Hujun Bao; Luwei Yang; Zhaopeng Cui
>
> **备注:** Accepted by AAAI 2025. Project page: https://zju3dv.github.io/GURecon/
>
> **摘要:** Neural surface representation has demonstrated remarkable success in the areas of novel view synthesis and 3D reconstruction. However, assessing the geometric quality of 3D reconstructions in the absence of ground truth mesh remains a significant challenge, due to its rendering-based optimization process and entangled learning of appearance and geometry with photometric losses. In this paper, we present a novel framework, i.e, GURecon, which establishes a geometric uncertainty field for the neural surface based on geometric consistency. Different from existing methods that rely on rendering-based measurement, GURecon models a continuous 3D uncertainty field for the reconstructed surface, and is learned by an online distillation approach without introducing real geometric information for supervision. Moreover, in order to mitigate the interference of illumination on geometric consistency, a decoupled field is learned and exploited to finetune the uncertainty field. Experiments on various datasets demonstrate the superiority of GURecon in modeling 3D geometric uncertainty, as well as its plug-and-play extension to various neural surface representations and improvement on downstream tasks such as incremental reconstruction. The code and supplementary material are available on the project website: https://zju3dv.github.io/GURecon/.
>
---
#### [replaced 076] Vector Quantization Prompting for Continual Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.20444v2](http://arxiv.org/pdf/2410.20444v2)**

> **作者:** Li Jiao; Qiuxia Lai; Yu Li; Qiang Xu
>
> **备注:** Accepted by NeurIPS 2024
>
> **摘要:** Continual learning requires to overcome catastrophic forgetting when training a single model on a sequence of tasks. Recent top-performing approaches are prompt-based methods that utilize a set of learnable parameters (i.e., prompts) to encode task knowledge, from which appropriate ones are selected to guide the fixed pre-trained model in generating features tailored to a certain task. However, existing methods rely on predicting prompt identities for prompt selection, where the identity prediction process cannot be optimized with task loss. This limitation leads to sub-optimal prompt selection and inadequate adaptation of pre-trained features for a specific task. Previous efforts have tried to address this by directly generating prompts from input queries instead of selecting from a set of candidates. However, these prompts are continuous, which lack sufficient abstraction for task knowledge representation, making them less effective for continual learning. To address these challenges, we propose VQ-Prompt, a prompt-based continual learning method that incorporates Vector Quantization (VQ) into end-to-end training of a set of discrete prompts. In this way, VQ-Prompt can optimize the prompt selection process with task loss and meanwhile achieve effective abstraction of task knowledge for continual learning. Extensive experiments show that VQ-Prompt outperforms state-of-the-art continual learning methods across a variety of benchmarks under the challenging class-incremental setting. The code is available at \href{https://github.com/jiaolifengmi/VQ-Prompt}{this https URL}.
>
---
#### [replaced 077] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03433v2](http://arxiv.org/pdf/2506.03433v2)**

> **作者:** Yifan Li; Xin Li; Tianqin Li; Wenbin He; Yu Kong; Liu Ren
>
> **备注:** The project is available: https://jackyfl.github.io/vitsplit.github.io/
>
> **摘要:** Vision foundation models (VFMs) have demonstrated remarkable performance across a wide range of downstream tasks. While several VFM adapters have shown promising results by leveraging the prior knowledge of VFMs, we identify two inefficiencies in these approaches. First, the interaction between convolutional neural network (CNN) and VFM backbone triggers early layer gradient backpropagation. Second, existing methods require tuning all components, adding complexity. Besides, these adapters alter VFM features, underutilizing the prior knowledge. To tackle these challenges, we propose a new approach called ViT-Split, based on a key observation: the layers of several VFMs, like DINOv2, can be divided into two distinct components: an extractor for learning low-level features and an adapter for learning task-specific features. Leveraging this insight, we eliminate the CNN branch and introduce two heads, task head and prior head, to the frozen VFM. The task head is designed to learn task-specific features, mitigating the early gradient propagation issue. The prior head is used to leverage the multi-scale prior features from the frozen VFM, reducing tuning parameters and overfitting. Extensive experiments on various tasks (e.g., segmentation, detection, depth estimation, and visual question answering) validate the effectiveness and efficiency of ViT-Split. Specifically, ViT-Split reduces training time up to $4\times$ while achieving comparable or even better results on ADE20K, compared to other VFM adapters.
>
---
#### [replaced 078] PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2306.16894v2](http://arxiv.org/pdf/2306.16894v2)**

> **作者:** Wenjing Huang; Shikui Tu; Lei Xu
>
> **备注:** Accepted by Neural Networks. Code is available at https://github.com/CMACH508/PFB-Diff
>
> **摘要:** Diffusion models have demonstrated their ability to generate diverse and high-quality images, sparking considerable interest in their potential for real image editing applications. However, existing diffusion-based approaches for local image editing often suffer from undesired artifacts due to the latent-level blending of the noised target images and diffusion latent variables, which lack the necessary semantics for maintaining image consistency. To address these issues, we propose PFB-Diff, a Progressive Feature Blending method for Diffusion-based image editing. Unlike previous methods, PFB-Diff seamlessly integrates text-guided generated content into the target image through multi-level feature blending. The rich semantics encoded in deep features and the progressive blending scheme from high to low levels ensure semantic coherence and high quality in edited images. Additionally, we introduce an attention masking mechanism in the cross-attention layers to confine the impact of specific words to desired regions, further improving the performance of background editing and multi-object replacement. PFB-Diff can effectively address various editing tasks, including object/background replacement and object attribute editing. Our method demonstrates its superior performance in terms of editing accuracy and image quality without the need for fine-tuning or training. Our implementation is available at https://github.com/CMACH508/PFB-Diff.
>
---
#### [replaced 079] Coupling AI and Citizen Science in Creation of Enhanced Training Dataset for Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.03087v2](http://arxiv.org/pdf/2409.03087v2)**

> **作者:** Amir Syahmi; Xiangrong Lu; Yinxuan Li; Haoxuan Yao; Hanjun Jiang; Ishita Acharya; Shiyi Wang; Yang Nan; Xiaodan Xing; Guang Yang
>
> **摘要:** Recent advancements in medical imaging and artificial intelligence (AI) have greatly enhanced diagnostic capabilities, but the development of effective deep learning (DL) models is still constrained by the lack of high-quality annotated datasets. The traditional manual annotation process by medical experts is time- and resource-intensive, limiting the scalability of these datasets. In this work, we introduce a robust and versatile framework that combines AI and crowdsourcing to improve both the quality and quantity of medical image datasets across different modalities. Our approach utilises a user-friendly online platform that enables a diverse group of crowd annotators to label medical images efficiently. By integrating the MedSAM segmentation AI with this platform, we accelerate the annotation process while maintaining expert-level quality through an algorithm that merges crowd-labelled images. Additionally, we employ pix2pixGAN, a generative AI model, to expand the training dataset with synthetic images that capture realistic morphological features. These methods are combined into a cohesive framework designed to produce an enhanced dataset, which can serve as a universal pre-processing pipeline to boost the training of any medical deep learning segmentation model. Our results demonstrate that this framework significantly improves model performance, especially when training data is limited.
>
---
#### [replaced 080] DAA*: Deep Angular A Star for Image-based Path Planning
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.09305v2](http://arxiv.org/pdf/2507.09305v2)**

> **作者:** Zhiwei Xu
>
> **备注:** International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Path smoothness is often overlooked in path imitation learning from expert demonstrations. In this paper, we introduce a novel learning method, termed deep angular A* (DAA*), by incorporating the proposed path angular freedom (PAF) into A* to improve path similarity through adaptive path smoothness. The PAF aims to explore the effect of move angles on path node expansion by finding the trade-off between their minimum and maximum values, allowing for high adaptiveness for imitation learning. DAA* improves path optimality by closely aligning with the reference path through joint optimization of path shortening and smoothing, which correspond to heuristic distance and PAF, respectively. Throughout comprehensive evaluations on 7 datasets, including 4 maze datasets, 2 video-game datasets, and a real-world drone-view dataset containing 2 scenarios, we demonstrate remarkable improvements of our DAA* over neural A* in path similarity between the predicted and reference paths with a shorter path length when the shortest path is plausible, improving by 9.0% SPR, 6.9% ASIM, and 3.9% PSIM. Furthermore, when jointly learning pathfinding with both path loss and path probability map loss, DAA* significantly outperforms the state-of-the-art TransPath by 6.7% SPR, 6.5% PSIM, and 3.7% ASIM. We also discuss the minor trade-off between path optimality and search efficiency where applicable. Our code and model weights are available at https://github.com/zwxu064/DAAStar.git.
>
---
#### [replaced 081] EHPE: A Segmented Architecture for Enhanced Hand Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09560v2](http://arxiv.org/pdf/2507.09560v2)**

> **作者:** Bolun Zheng; Xinjie Liu; Qianyu Zhang; Canjin Wang; Fangni Chen; Mingen Xu
>
> **摘要:** 3D hand pose estimation has garnered great attention in recent years due to its critical applications in human-computer interaction, virtual reality, and related fields. The accurate estimation of hand joints is essential for high-quality hand pose estimation. However, existing methods neglect the importance of Distal Phalanx Tip (TIP) and Wrist in predicting hand joints overall and often fail to account for the phenomenon of error accumulation for distal joints in gesture estimation, which can cause certain joints to incur larger errors, resulting in misalignments and artifacts in the pose estimation and degrading the overall reconstruction quality. To address this challenge, we propose a novel segmented architecture for enhanced hand pose estimation (EHPE). We perform local extraction of TIP and wrist, thus alleviating the effect of error accumulation on TIP prediction and further reduce the predictive errors for all joints on this basis. EHPE consists of two key stages: In the TIP and Wrist Joints Extraction stage (TW-stage), the positions of the TIP and wrist joints are estimated to provide an initial accurate joint configuration; In the Prior Guided Joints Estimation stage (PG-stage), a dual-branch interaction network is employed to refine the positions of the remaining joints. Extensive experiments on two widely used benchmarks demonstrate that EHPE achieves state-of-the-arts performance. Code is available at https://github.com/SereinNout/EHPE.
>
---
#### [replaced 082] Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.15265v2](http://arxiv.org/pdf/2411.15265v2)**

> **作者:** Won Jun Kim; Hyungjin Chung; Jaemin Kim; Sangmin Lee; Byeongsu Sim; Jong Chul Ye
>
> **备注:** CVPR 2025 (poster), 19 pages, 5 figures
>
> **摘要:** Gradient-based methods are a prototypical family of explainability techniques, especially for image-based models. Nonetheless, they have several shortcomings in that they (1) require white-box access to models, (2) are vulnerable to adversarial attacks, and (3) produce attributions that lie off the image manifold, leading to explanations that are not actually faithful to the model and do not align well with human perception. To overcome these challenges, we introduce Derivative-Free Diffusion Manifold-Constrainted Gradients (FreeMCG), a novel method that serves as an improved basis for explainability of a given neural network than the traditional gradient. Specifically, by leveraging ensemble Kalman filters and diffusion models, we derive a derivative-free approximation of the model's gradient projected onto the data manifold, requiring access only to the model's outputs. We demonstrate the effectiveness of FreeMCG by applying it to both counterfactual generation and feature attribution, which have traditionally been treated as distinct tasks. Through comprehensive evaluation on both tasks, counterfactual explanation and feature attribution, we show that our method yields state-of-the-art results while preserving the essential properties expected of XAI tools.
>
---
#### [replaced 083] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08416v2](http://arxiv.org/pdf/2507.08416v2)**

> **作者:** Zesong Yang; Bangbang Yang; Wenqi Dong; Chenxuan Cao; Liyuan Cui; Yuewen Ma; Zhaopeng Cui; Hujun Bao
>
> **备注:** Accepted by ICCV 2025. Project page: https://zju3dv.github.io/instascene/
>
> **摘要:** Humans can naturally identify and mentally complete occluded objects in cluttered environments. However, imparting similar cognitive ability to robotics remains challenging even with advanced reconstruction techniques, which models scenes as undifferentiated wholes and fails to recognize complete object from partial observations. In this paper, we propose InstaScene, a new paradigm towards holistic 3D perception of complex scenes with a primary goal: decomposing arbitrary instances while ensuring complete reconstruction. To achieve precise decomposition, we develop a novel spatial contrastive learning by tracing rasterization of each instance across views, significantly enhancing semantic supervision in cluttered scenes. To overcome incompleteness from limited observations, we introduce in-situ generation that harnesses valuable observations and geometric cues, effectively guiding 3D generative models to reconstruct complete instances that seamlessly align with the real world. Experiments on scene decomposition and object completion across complex real-world and synthetic scenes demonstrate that our method achieves superior decomposition accuracy while producing geometrically faithful and visually intact objects.
>
---
#### [replaced 084] CNeuroMod-THINGS, a densely-sampled fMRI dataset for visual neuroscience
- **分类: q-bio.NC; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09024v2](http://arxiv.org/pdf/2507.09024v2)**

> **作者:** Marie St-Laurent; Basile Pinsard; Oliver Contier; Elizabeth DuPre; Katja Seeliger; Valentina Borghesani; Julie A. Boyle; Lune Bellec; Martin N. Hebart
>
> **备注:** 16 pages manuscript, 5 figures, 9 pages supplementary material
>
> **摘要:** Data-hungry neuro-AI modelling requires ever larger neuroimaging datasets. CNeuroMod-THINGS meets this need by capturing neural representations for a wide set of semantic concepts using well-characterized images in a new densely-sampled, large-scale fMRI dataset. Importantly, CNeuroMod-THINGS exploits synergies between two existing projects: the THINGS initiative (THINGS) and the Courtois Project on Neural Modelling (CNeuroMod). THINGS has developed a common set of thoroughly annotated images broadly sampling natural and man-made objects which is used to acquire a growing collection of large-scale multimodal neural responses. Meanwhile, CNeuroMod is acquiring hundreds of hours of fMRI data from a core set of participants during controlled and naturalistic tasks, including visual tasks like movie watching and videogame playing. For CNeuroMod-THINGS, four CNeuroMod participants each completed 33-36 sessions of a continuous recognition paradigm using approximately 4000 images from the THINGS stimulus set spanning 720 categories. We report behavioural and neuroimaging metrics that showcase the quality of the data. By bridging together large existing resources, CNeuroMod-THINGS expands our capacity to model broad slices of the human visual experience.
>
---
#### [replaced 085] RealGeneral: Unifying Visual Generation via Temporal In-Context Learning with Video Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10406v2](http://arxiv.org/pdf/2503.10406v2)**

> **作者:** Yijing Lin; Mengqi Huang; Shuhan Zhuang; Zhendong Mao
>
> **摘要:** Unifying diverse image generation tasks within a single framework remains a fundamental challenge in visual generation. While large language models (LLMs) achieve unification through task-agnostic data and generation, existing visual generation models fail to meet these principles. Current approaches either rely on per-task datasets and large-scale training or adapt pre-trained image models with task-specific modifications, limiting their generalizability. In this work, we explore video models as a foundation for unified image generation, leveraging their inherent ability to model temporal correlations. We introduce RealGeneral, a novel framework that reformulates image generation as a conditional frame prediction task, analogous to in-context learning in LLMs. To bridge the gap between video models and condition-image pairs, we propose (1) a Unified Conditional Embedding module for multi-modal alignment and (2) a Unified Stream DiT Block with decoupled adaptive LayerNorm and attention mask to mitigate cross-modal interference. RealGeneral demonstrates effectiveness in multiple important visual generation tasks, e.g., it achieves a 14.5% improvement in subject similarity for customized generation and a 10% enhancement in image quality for canny-to-image task. Project page: https://lyne1.github.io/realgeneral_web/; GitHub Link: https://github.com/Lyne1/RealGeneral
>
---
#### [replaced 086] An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13764v3](http://arxiv.org/pdf/2502.13764v3)**

> **作者:** Wanke Xia; Ruoxin Peng; Haoqi Chu; Xinlei Zhu; Zhiyu Yang; Lili Yang
>
> **摘要:** Rice is one of the most widely cultivated crops globally and has been developed into numerous varieties. The quality of rice during cultivation is primarily determined by its cultivar and characteristics. Traditionally, rice classification and quality assessment rely on manual visual inspection, a process that is both time-consuming and prone to errors. However, with advancements in machine vision technology, automating rice classification and quality evaluation based on its cultivar and characteristics has become increasingly feasible, enhancing both accuracy and efficiency. This study proposes a real-time evaluation mechanism for comprehensive rice grain assessment, integrating a one-stage object detection approach, a deep convolutional neural network, and traditional machine learning techniques. The proposed framework enables rice variety identification, grain completeness grading, and grain chalkiness evaluation. The rice grain dataset used in this study comprises approximately 20,000 images from six widely cultivated rice varieties in China. Experimental results demonstrate that the proposed mechanism achieves a mean average precision (mAP) of 99.14% in the object detection task and an accuracy of 97.89% in the classification task. Furthermore, the framework attains an average accuracy of 97.56% in grain completeness grading within the same rice variety, contributing to an effective quality evaluation system.
>
---
#### [replaced 087] PhenoBench: A Comprehensive Benchmark for Cell Phenotyping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03532v5](http://arxiv.org/pdf/2507.03532v5)**

> **作者:** Fabian H. Reith; Claudia Winklmayr; Jerome Luescher; Nora Koreuber; Jannik Franzen; Elias Baumann; Christian M. Schuerch; Dagmar Kainmueller; Josef Lorenz Rumberger
>
> **备注:** accepted for presentation at MICCAI 2025
>
> **摘要:** Digital pathology has seen the advent of a wealth of foundational models (FM), yet to date their performance on cell phenotyping has not been benchmarked in a unified manner. We therefore propose PhenoBench: A comprehensive benchmark for cell phenotyping on Hematoxylin and Eosin (H&E) stained histopathology images. We provide both PhenoCell, a new H&E dataset featuring 14 granular cell types identified by using multiplexed imaging, and ready-to-use fine-tuning and benchmarking code that allows the systematic evaluation of multiple prominent pathology FMs in terms of dense cell phenotype predictions in different generalization scenarios. We perform extensive benchmarking of existing FMs, providing insights into their generalization behavior under technical vs. medical domain shifts. Furthermore, while FMs achieve macro F1 scores > 0.70 on previously established benchmarks such as Lizard and PanNuke, on PhenoCell, we observe scores as low as 0.20. This indicates a much more challenging task not captured by previous benchmarks, establishing PhenoCell as a prime asset for future benchmarking of FMs and supervised models alike. Code and data are available on GitHub.
>
---
#### [replaced 088] Fourier Domain Adaptation for Traffic Light Detection in Adverse Weather
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.07901v2](http://arxiv.org/pdf/2411.07901v2)**

> **作者:** Ishaan Gakhar; Aryesh Guha; Aryaman Gupta; Amit Agarwal; Ujjwal Verma
>
> **备注:** Accepted at the 2COOOL Workshop, ICCV 2025
>
> **摘要:** Traffic light detection under adverse weather conditions remains largely unexplored in ADAS systems, with existing approaches relying on complex deep learning methods that introduce significant computational overheads during training and deployment. This paper proposes Fourier Domain Adaptation (FDA), which requires only training data modifications without architectural changes, enabling effective adaptation to rainy and foggy conditions. FDA minimizes the domain gap between source and target domains, creating a dataset for reliable performance under adverse weather. The source domain merged LISA and S2TLD datasets, processed to address class imbalance. Established methods simulated rainy and foggy scenarios to form the target domain. Semi-Supervised Learning (SSL) techniques were explored to leverage data more effectively, addressing the shortage of comprehensive datasets and poor performance of state-of-the-art models under hostile weather. Experimental results show FDA-augmented models outperform baseline models across mAP50, mAP50-95, Precision, and Recall metrics. YOLOv8 achieved a 12.25% average increase across all metrics. Average improvements of 7.69% in Precision, 19.91% in Recall, 15.85% in mAP50, and 23.81% in mAP50-95 were observed across all models, demonstrating FDA's effectiveness in mitigating adverse weather impact. These improvements enable real-world applications requiring reliable performance in challenging environmental conditions.
>
---
#### [replaced 089] An Improved Pure Fully Connected Neural Network for Rice Grain Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.03111v2](http://arxiv.org/pdf/2503.03111v2)**

> **作者:** Wanke Xia; Ruoxin Peng; Haoqi Chu; Xinlei Zhu; Zhiyu Yang; Lili Yang; Bo Lv; Xunwen Xiang
>
> **摘要:** Rice is a staple food for a significant portion of the world's population, providing essential nutrients and serving as a versatile in-gredient in a wide range of culinary traditions. Recently, the use of deep learning has enabled automated classification of rice, im-proving accuracy and efficiency. However, classical models based on first-stage training may face difficulties in distinguishing between rice varieties with similar external characteristics, thus leading to misclassifications. Considering the transparency and feasibility of model, we selected and gradually improved pure fully connected neural network to achieve classification of rice grain. The dataset we used contains both global and domestic rice images obtained from websites and laboratories respectively. First, the training mode was changed from one-stage training to two-stage training, which significantly contributes to distinguishing two similar types of rice. Secondly, the preprocessing method was changed from random tilting to horizontal or vertical position cor-rection. After those two enhancements, the accuracy of our model increased notably from 97% to 99%. In summary, two subtle methods proposed in this study can remarkably enhance the classification ability of deep learning models in terms of the classification of rice grain.
>
---
#### [replaced 090] Surf-NeRF: Surface Regularised Neural Radiance Fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18652v2](http://arxiv.org/pdf/2411.18652v2)**

> **作者:** Jack Naylor; Viorela Ila; Donald G. Dansereau
>
> **备注:** 20 pages, 17 figures, 9 tables, project page can be found at http://roboticimaging.org/Projects/SurfNeRF
>
> **摘要:** Neural Radiance Fields (NeRFs) provide a high fidelity, continuous scene representation that can realistically represent complex behaviour of light. Despite works like Ref-NeRF improving geometry through physics-inspired models, the ability for a NeRF to overcome shape-radiance ambiguity and converge to a representation consistent with real geometry remains limited. We demonstrate how both curriculum learning of a surface light field model and using a lattice-based hash encoding helps a NeRF converge towards a more geometrically accurate scene representation. We introduce four regularisation terms to impose geometric smoothness, consistency of normals, and a separation of Lambertian and specular appearance at geometry in the scene, conforming to physical models. Our approach yields 28% more accurate normals than traditional grid-based NeRF variants with reflection parameterisation. Our approach more accurately separates view-dependent appearance, conditioning a NeRF to have a geometric representation consistent with the captured scene. We demonstrate compatibility of our method with existing NeRF variants, as a key step in enabling radiance-based representations for geometry critical applications.
>
---
#### [replaced 091] From Wardrobe to Canvas: Wardrobe Polyptych LoRA for Part-level Controllable Human Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.10217v2](http://arxiv.org/pdf/2507.10217v2)**

> **作者:** Jeongho Kim; Sunghyun Park; Hyoungwoo Park; Sungrack Yun; Jaegul Choo; Seokeon Choi
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Recent diffusion models achieve personalization by learning specific subjects, allowing learned attributes to be integrated into generated images. However, personalized human image generation remains challenging due to the need for precise and consistent attribute preservation (e.g., identity, clothing details). Existing subject-driven image generation methods often require either (1) inference-time fine-tuning with few images for each new subject or (2) large-scale dataset training for generalization. Both approaches are computationally expensive and impractical for real-time applications. To address these limitations, we present Wardrobe Polyptych LoRA, a novel part-level controllable model for personalized human image generation. By training only LoRA layers, our method removes the computational burden at inference while ensuring high-fidelity synthesis of unseen subjects. Our key idea is to condition the generation on the subject's wardrobe and leverage spatial references to reduce information loss, thereby improving fidelity and consistency. Additionally, we introduce a selective subject region loss, which encourages the model to disregard some of reference images during training. Our loss ensures that generated images better align with text prompts while maintaining subject integrity. Notably, our Wardrobe Polyptych LoRA requires no additional parameters at the inference stage and performs generation using a single model trained on a few training samples. We construct a new dataset and benchmark tailored for personalized human image generation. Extensive experiments show that our approach significantly outperforms existing techniques in fidelity and consistency, enabling realistic and identity-preserving full-body synthesis.
>
---
#### [replaced 092] SemiOccam: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03582v3](http://arxiv.org/pdf/2506.03582v3)**

> **作者:** Rui Yann; Tianshuo Zhang; Xianglei Xing
>
> **备注:** CleanSTL-10 available at https://huggingface.co/datasets/Shu1L0n9/CleanSTL-10
>
> **摘要:** We present SemiOccam, an image recognition network that leverages semi-supervised learning in a highly efficient manner. Existing works often rely on complex training techniques and architectures, requiring hundreds of GPU hours for training, while their generalization ability with extremely limited labeled data remains to be improved. To address these limitations, we construct a hierarchical mixture density classification mechanism by optimizing mutual information between feature representations and target classes, compressing redundant information while retaining crucial discriminative components. Experimental results demonstrate that our method achieves state-of-the-art performance on three commonly used datasets, with accuracy exceeding 95% on two of them using only 4 labeled samples per class, and its simple architecture keeps training time at the minute level. Notably, this paper reveals a long-overlooked data leakage issue in the STL-10 dataset for semi-supervised learning and removes duplicates to ensure reliable experimental results. We release the deduplicated CleanSTL-10 dataset to facilitate fair and reproducible research. Code available at https://github.com/Shu1L0n9/SemiOccam.
>
---
#### [replaced 093] PhysX-3D: Physical-Grounded 3D Asset Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12465v3](http://arxiv.org/pdf/2507.12465v3)**

> **作者:** Ziang Cao; Zhaoxi Chen; Liang Pan; Ziwei Liu
>
> **备注:** Project page: https://physx-3d.github.io/
>
> **摘要:** 3D modeling is moving from virtual to physical. Existing 3D generation primarily emphasizes geometries and textures while neglecting physical-grounded modeling. Consequently, despite the rapid development of 3D generative models, the synthesized 3D assets often overlook rich and important physical properties, hampering their real-world application in physical domains like simulation and embodied AI. As an initial attempt to address this challenge, we propose \textbf{PhysX-3D}, an end-to-end paradigm for physical-grounded 3D asset generation. 1) To bridge the critical gap in physics-annotated 3D datasets, we present PhysXNet - the first physics-grounded 3D dataset systematically annotated across five foundational dimensions: absolute scale, material, affordance, kinematics, and function description. In particular, we devise a scalable human-in-the-loop annotation pipeline based on vision-language models, which enables efficient creation of physics-first assets from raw 3D assets.2) Furthermore, we propose \textbf{PhysXGen}, a feed-forward framework for physics-grounded image-to-3D asset generation, injecting physical knowledge into the pre-trained 3D structural space. Specifically, PhysXGen employs a dual-branch architecture to explicitly model the latent correlations between 3D structures and physical properties, thereby producing 3D assets with plausible physical predictions while preserving the native geometry quality. Extensive experiments validate the superior performance and promising generalization capability of our framework. All the code, data, and models will be released to facilitate future research in generative physical AI.
>
---
#### [replaced 094] 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09993v2](http://arxiv.org/pdf/2507.09993v2)**

> **作者:** Yixun Zhang; Lizhi Wang; Junjun Zhao; Wending Zhao; Feng Zhou; Yonghao Dang; Jianqin Yin
>
> **备注:** Submitted to WACV 2026
>
> **摘要:** Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. Existing 2D and 3D physical attacks, due to their focus on texture optimization, often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture optimization, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module that filters outliers to preserve geometric fidelity, and a physical augmentation module that simulates complex physical scenarios to enhance attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21\% to 7.38\%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks.
>
---
#### [replaced 095] OncoReg: Medical Image Registration for Oncological Challenges
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23179v3](http://arxiv.org/pdf/2503.23179v3)**

> **作者:** Wiebke Heyer; Yannic Elser; Lennart Berkel; Xinrui Song; Xuanang Xu; Pingkun Yan; Xi Jia; Jinming Duan; Zi Li; Tony C. W. Mok; BoWen LI; Tim Hable; Christian Staackmann; Christoph Großbröhmer; Lasse Hansen; Alessa Hering; Malte M. Sieren; Mattias P. Heinrich
>
> **备注:** 19 pages, 13 figures
>
> **摘要:** In modern cancer research, the vast volume of medical data generated is often underutilised due to challenges related to patient privacy. The OncoReg Challenge addresses this issue by enabling researchers to develop and validate image registration methods through a two-phase framework that ensures patient privacy while fostering the development of more generalisable AI models. Phase one involves working with a publicly available dataset, while phase two focuses on training models on a private dataset within secure hospital networks. OncoReg builds upon the foundation established by the Learn2Reg Challenge by incorporating the registration of interventional cone-beam computed tomography (CBCT) with standard planning fan-beam CT (FBCT) images in radiotherapy. Accurate image registration is crucial in oncology, particularly for dynamic treatment adjustments in image-guided radiotherapy, where precise alignment is necessary to minimise radiation exposure to healthy tissues while effectively targeting tumours. This work details the methodology and data behind the OncoReg Challenge and provides a comprehensive analysis of the competition entries and results. Findings reveal that feature extraction plays a pivotal role in this registration task. A new method emerging from this challenge demonstrated its versatility, while established approaches continue to perform comparably to newer techniques. Both deep learning and classical approaches still play significant roles in image registration, with the combination of methods, particularly in feature extraction, proving most effective.
>
---
#### [replaced 096] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07098v2](http://arxiv.org/pdf/2503.07098v2)**

> **作者:** Ding Zhong; Xu Zheng; Chenfei Liao; Yuanhuiyi Lyu; Jialei Chen; Shengyang Wu; Linfeng Zhang; Xuming Hu
>
> **摘要:** Segment Anything Model 2 (SAM2) has emerged as a strong base model in various pinhole imaging segmentation tasks. However, when applying it to $360^\circ$ domain, the significant field-of-view (FoV) gap between pinhole ($70^\circ \times 70^\circ$) and panoramic images ($180^\circ \times 360^\circ$) poses unique challenges. Two major concerns for this application includes 1) inevitable distortion and object deformation brought by the large FoV disparity between domains; 2) the lack of pixel-level semantic understanding that the original SAM2 cannot provide. To address these issues, we propose a novel OmniSAM framework, which makes the first attempt to apply SAM2 for panoramic semantic segmentation. Specifically, to bridge the first gap, OmniSAM first divides the panorama into sequences of patches. These patches are then treated as image sequences in similar manners as in video segmentation tasks. We then leverage the SAM2's memory mechanism to extract cross-patch correspondences that embeds the cross-FoV dependencies, improving feature continuity and the prediction consistency along mask boundaries. For the second gap, OmniSAM fine-tunes the pretrained image encoder and reutilize the mask decoder for semantic prediction. An FoV-based prototypical adaptation module with dynamic pseudo label update mechanism is also introduced to facilitate the alignment of memory and backbone features, thereby improving model generalization ability across different sizes of source models. Extensive experimental results demonstrate that OmniSAM outperforms the state-of-the-art methods by large margins, e.g., 79.06% (+10.22%) on SPin8-to-SPan8, 62.46% (+6.58%) on CS13-to-DP13.
>
---
#### [replaced 097] Learning-based 3D Reconstruction in Autonomous Driving: A Comprehensive Survey
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.14537v4](http://arxiv.org/pdf/2503.14537v4)**

> **作者:** Liewen Liao; Weihao Yan; Ming Yang; Songan Zhang
>
> **摘要:** Learning-based 3D reconstruction has emerged as a transformative technique in autonomous driving, enabling precise modeling of both dynamic and static environments through advanced neural representations. Despite data augmentation, 3D reconstruction inspires pioneering solution for vital tasks in the field of autonomous driving, such as scene understanding and closed-loop simulation. We investigates the details of 3D reconstruction and conducts a multi-perspective, in-depth analysis of recent advancements. Specifically, we first provide a systematic introduction of preliminaries, including data modalities, benchmarks and technical preliminaries of learning-based 3D reconstruction, facilitating instant identification of suitable methods according to sensor suites. Then, we systematically review learning-based 3D reconstruction methods in autonomous driving, categorizing approaches by subtasks and conducting multi-dimensional analysis and summary to establish a comprehensive technical reference. The development trends and existing challenges are summarized in the context of learning-based 3D reconstruction in autonomous driving. We hope that our review will inspire future researches.
>
---
#### [replaced 098] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06505v3](http://arxiv.org/pdf/2503.06505v3)**

> **作者:** Xirui Hu; Jiahao Wang; Hao Chen; Weizhan Zhang; Benqi Wang; Yikun Li; Haishun Nan
>
> **备注:** ICCV 2025
>
> **摘要:** Recent advances in text-to-image generation have driven interest in generating personalized human images that depict specific identities from reference images. Although existing methods achieve high-fidelity identity preservation, they are generally limited to single-ID scenarios and offer insufficient facial editability. We present DynamicID, a tuning-free framework that inherently facilitates both single-ID and multi-ID personalized generation with high fidelity and flexible facial editability. Our key innovations include: 1) Semantic-Activated Attention (SAA), which employs query-level activation gating to minimize disruption to the base model when injecting ID features and achieve multi-ID personalization without requiring multi-ID samples during training. 2) Identity-Motion Reconfigurator (IMR), which applies feature-space manipulation to effectively disentangle and reconfigure facial motion and identity features, supporting flexible facial editing. 3) a task-decoupled training paradigm that reduces data dependency, together with VariFace-10k, a curated dataset of 10k unique individuals, each represented by 35 distinct facial images. Experimental results demonstrate that DynamicID outperforms state-of-the-art methods in identity fidelity, facial editability, and multi-ID personalization capability. Our code will be released at https://github.com/ByteCat-bot/DynamicID.
>
---
#### [replaced 099] Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02939v2](http://arxiv.org/pdf/2507.02939v2)**

> **作者:** Yuqi Li; Chuanguang Yang; Hansheng Zeng; Zeyu Dong; Zhulin An; Yongjun Xu; Yingli Tian; Hao Wu
>
> **备注:** Accepted by ICCV-2025, 11 pages
>
> **摘要:** Spatiotemporal forecasting tasks, such as traffic flow, combustion dynamics, and weather forecasting, often require complex models that suffer from low training efficiency and high memory consumption. This paper proposes a lightweight framework, Spectral Decoupled Knowledge Distillation (termed SDKD), which transfers the multi-scale spatiotemporal representations from a complex teacher model to a more efficient lightweight student network. The teacher model follows an encoder-latent evolution-decoder architecture, where its latent evolution module decouples high-frequency details and low-frequency trends using convolution and Transformer (global low-frequency modeler). However, the multi-layer convolution and deconvolution structures result in slow training and high memory usage. To address these issues, we propose a frequency-aligned knowledge distillation strategy, which extracts multi-scale spectral features from the teacher's latent space, including both high and low frequency components, to guide the lightweight student model in capturing both local fine-grained variations and global evolution patterns. Experimental results show that SDKD significantly improves performance, achieving reductions of up to 81.3% in MSE and in MAE 52.3% on the Navier-Stokes equation dataset. The framework effectively captures both high-frequency variations and long-term trends while reducing computational complexity. Our codes are available at https://github.com/itsnotacie/SDKD
>
---
#### [replaced 100] Reviving Cultural Heritage: A Novel Approach for Comprehensive Historical Document Restoration
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05108v2](http://arxiv.org/pdf/2507.05108v2)**

> **作者:** Yuyi Zhang; Peirong Zhang; Zhenhua Yang; Pengyu Yan; Yongxin Shi; Pengwei Liu; Fengjun Guo; Lianwen Jin
>
> **摘要:** Historical documents represent an invaluable cultural heritage, yet have undergone significant degradation over time through tears, water erosion, and oxidation. Existing Historical Document Restoration (HDR) methods primarily focus on single modality or limited-size restoration, failing to meet practical needs. To fill this gap, we present a full-page HDR dataset (FPHDR) and a novel automated HDR solution (AutoHDR). Specifically, FPHDR comprises 1,633 real and 6,543 synthetic images with character-level and line-level locations, as well as character annotations in different damage grades. AutoHDR mimics historians' restoration workflows through a three-stage approach: OCR-assisted damage localization, vision-language context text prediction, and patch autoregressive appearance restoration. The modular architecture of AutoHDR enables seamless human-machine collaboration, allowing for flexible intervention and optimization at each restoration stage. Experiments demonstrate AutoHDR's remarkable performance in HDR. When processing severely damaged documents, our method improves OCR accuracy from 46.83% to 84.05%, with further enhancement to 94.25% through human-machine collaboration. We believe this work represents a significant advancement in automated historical document restoration and contributes substantially to cultural heritage preservation. The model and dataset are available at https://github.com/SCUT-DLVCLab/AutoHDR.
>
---
#### [replaced 101] Leveraging Vision-Language Models for Visual Grounding and Analysis of Automotive UI
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.05895v2](http://arxiv.org/pdf/2505.05895v2)**

> **作者:** Benjamin Raphael Ernhofer; Daniil Prokhorov; Jannica Langner; Dominik Bollmann
>
> **摘要:** Modern automotive infotainment systems require intelligent and adaptive solutions to handle frequent User Interface (UI) updates and diverse design variations. We introduce a vision-language framework for understanding and interacting with automotive infotainment systems, enabling seamless adaptation across different UI designs. To further support research in this field, we release AutomotiveUI-Bench-4K, an open-source dataset of 998 images with 4,208 annotations. Additionally, we present a synthetic data pipeline to generate training data. We fine-tune a Molmo-7B-based model using Low-Rank Adaptation (LoRa) and incorporating reasoning generated by our pipeline, along with visual grounding and evaluation capabilities. The fine-tuned Evaluative Large Action Model (ELAM) achieves strong performance on AutomotiveUI-Bench-4K (model and dataset are available on Hugging Face) and demonstrating strong cross-domain generalization, including a +5.6% improvement on ScreenSpot over the baseline model. Notably, our approach achieves 80.8% average accuracy on ScreenSpot, closely matching or even surpassing specialized models for desktop, mobile, and web, such as ShowUI, despite being trained for the infotainment domain. This research investigates how data collection and subsequent fine-tuning can lead to AI-driven progress within automotive UI understanding and interaction. The applied method is cost-efficient and fine-tuned models can be deployed on consumer-grade GPUs.
>
---
#### [replaced 102] Brain-Inspired Online Adaptation for Remote Sensing with Spiking Neural Network
- **分类: cs.LG; cs.CV; cs.NE**

- **链接: [http://arxiv.org/pdf/2409.02146v2](http://arxiv.org/pdf/2409.02146v2)**

> **作者:** Dexin Duan; Peilin liu; Bingwei Hui; Fei Wen
>
> **备注:** IEEE Transactions on Geoscience and Remote Sensing, 2025
>
> **摘要:** On-device computing, or edge computing, is becoming increasingly important for remote sensing, particularly in applications like deep network-based perception on on-orbit satellites and unmanned aerial vehicles (UAVs). In these scenarios, two brain-like capabilities are crucial for remote sensing models: (1) high energy efficiency, allowing the model to operate on edge devices with limited computing resources, and (2) online adaptation, enabling the model to quickly adapt to environmental variations, weather changes, and sensor drift. This work addresses these needs by proposing an online adaptation framework based on spiking neural networks (SNNs) for remote sensing. Starting with a pretrained SNN model, we design an efficient, unsupervised online adaptation algorithm, which adopts an approximation of the BPTT algorithm and only involves forward-in-time computation that significantly reduces the computational complexity of SNN adaptation learning. Besides, we propose an adaptive activation scaling scheme to boost online SNN adaptation performance, particularly in low time-steps. Furthermore, for the more challenging remote sensing detection task, we propose a confidence-based instance weighting scheme, which substantially improves adaptation performance in the detection task. To our knowledge, this work is the first to address the online adaptation of SNNs. Extensive experiments on seven benchmark datasets across classification, segmentation, and detection tasks demonstrate that our proposed method significantly outperforms existing domain adaptation and domain generalization approaches under varying weather conditions. The proposed method enables energy-efficient and fast online adaptation on edge devices, and has much potential in applications such as remote perception on on-orbit satellites and UAV.
>
---
#### [replaced 103] DOGR: Towards Versatile Visual Document Grounding and Referring
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.17125v2](http://arxiv.org/pdf/2411.17125v2)**

> **作者:** Yinan Zhou; Yuxin Chen; Haokun Lin; Shuyu Yang; Zhongang Qi; Chen Ma; Li Zhu; Ying Shan
>
> **备注:** 22 pages, 16 figures
>
> **摘要:** With recent advances in Multimodal Large Language Models (MLLMs), grounding and referring capabilities have gained increasing attention for achieving detailed understanding and flexible user interaction. However, these capabilities still remain underdeveloped in visual document understanding due to the scarcity of fine-grained datasets and comprehensive benchmarks. To fill this gap, we propose the DOcument Grounding and Referring data engine (DOGR-Engine), which generates two types of high-quality fine-grained document data: (1) multi-granular parsing data to improve text localization and recognition, and (2) instruction-tuning data to activate MLLMs' grounding and referring capabilities in dialogue and reasoning. Using the DOGR-Engine, we construct DOGR-Bench, a benchmark covering seven grounding and referring tasks across three document types (chart, poster, and PDF document), offering a comprehensive evaluation of fine-grained document understanding. Leveraging the generated data, we further develop DOGR, a strong baseline model that excels in text localization and recognition, while precisely grounds and refers to key textual information during conversation and reasoning, thereby advancing document understanding to a finer granularity and enable flexible interaction paradigms.
>
---
#### [replaced 104] TruthLens: Explainable DeepFake Detection for Face Manipulated and Fully Synthetic Data
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15867v2](http://arxiv.org/pdf/2503.15867v2)**

> **作者:** Rohit Kundu; Shan Jia; Vishal Mohanty; Athula Balachandran; Amit K. Roy-Chowdhury
>
> **摘要:** Detecting DeepFakes has become a crucial research area as the widespread use of AI image generators enables the effortless creation of face-manipulated and fully synthetic content, yet existing methods are often limited to binary classification (real vs. fake) and lack interpretability. To address these challenges, we propose TruthLens, a novel and highly generalizable framework for DeepFake detection that not only determines whether an image is real or fake but also provides detailed textual reasoning for its predictions. Unlike traditional methods, TruthLens effectively handles both face-manipulated DeepFakes and fully AI-generated content while addressing fine-grained queries such as "Does the eyes/nose/mouth look real or fake?" The architecture of TruthLens combines the global contextual understanding of multimodal large language models like PaliGemma2 with the localized feature extraction capabilities of vision-only models like DINOv2. This hybrid design leverages the complementary strengths of both models, enabling robust detection of subtle manipulations while maintaining interpretability. Extensive experiments on diverse datasets demonstrate that TruthLens outperforms state-of-the-art methods in detection accuracy (by 2-14%) and explainability, in both in-domain and cross-data settings, generalizing effectively across traditional and emerging manipulation techniques.
>
---
#### [replaced 105] ZS-VCOS: Zero-Shot Video Camouflaged Object Segmentation By Optical Flow and Open Vocabulary Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01431v2](http://arxiv.org/pdf/2505.01431v2)**

> **作者:** Wenqi Guo; Mohamed Shehata; Shan Du
>
> **摘要:** Camouflaged object segmentation presents unique challenges compared to traditional segmentation tasks, primarily due to the high similarity in patterns and colors between camouflaged objects and their backgrounds. Effective solutions to this problem have significant implications in critical areas such as pest control, defect detection, and lesion segmentation in medical imaging. Prior research has predominantly emphasized supervised or unsupervised pre-training methods, leaving zero-shot approaches significantly underdeveloped. Existing zero-shot techniques commonly utilize the Segment Anything Model (SAM) in automatic mode or rely on vision-language models to generate cues for segmentation; however, their performances remain unsatisfactory, due to the similarity of the camouflaged object and the background. This work studies how to avoid training by integrating large pre-trained models like SAM-2 and Owl-v2 with temporal information into a modular pipeline. Evaluated on the MoCA-Mask dataset, our approach achieves outstanding performance improvements, significantly outperforming existing zero-shot methods by raising the F-measure ($F_\beta^w$) from 0.296 to 0.628. Our approach also surpasses supervised methods, increasing the F-measure from 0.476 to 0.628. Additionally, evaluation on the MoCA-Filter dataset demonstrates an increase in the success rate from 0.628 to 0.697 when compared with FlowSAM, a supervised transfer method. A thorough ablation study further validates the individual contributions of each component. Besides our main contributions, we also highlight inconsistencies in previous work regarding metrics and settings. Code can be found in https://github.com/weathon/vcos.
>
---
#### [replaced 106] CVPT: Cross Visual Prompt Tuning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.14961v2](http://arxiv.org/pdf/2408.14961v2)**

> **作者:** Lingyun Huang; Jianxu Mao; Junfei Yi; Ziming Tao; Yaonan Wang
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) has emerged to mitigate the computational demands of large-scale models. Within computer vision, adapter-based PEFT methods are often favored over prompt-based approaches like Visual Prompt Tuning (VPT) due to the latter's performance and efficiency limitations. Our analysis reveals that VPT's shortcomings stem from its prompt deployment strategy, which can distort the model's inherent self-attention mechanism. To address this, we propose Cross Visual Prompt Tuning (CVPT). CVPT introduces a cross-attention module to directly model interactions between prompts and image tokens. This design decouples the prompts from the input sequence, preserving the original self-attention integrity while enabling efficient feature integration. Furthermore, we employ a weight-sharing mechanism for cross-attention initialization, which enhances representative capability without a large parameter overhead. Extensive experiments across 25 datasets show that CVPT significantly outperforms VPT. For instance, on the VTAB-1K benchmark, CVPT achieves over 4% higher average accuracy, rivaling leading adapter-based methods in both performance and efficiency. Our work confirms that prompt-based methods can achieve exceptional results in visual fine-tuning. The code is available at https://github.com/Lingyun0419/CVPT
>
---
#### [replaced 107] View Selection for 3D Captioning via Diffusion Ranking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.07984v2](http://arxiv.org/pdf/2404.07984v2)**

> **作者:** Tiange Luo; Justin Johnson; Honglak Lee
>
> **备注:** ECCV2024, Dataset link: https://huggingface.co/datasets/tiange/Cap3D
>
> **摘要:** Scalable annotation approaches are crucial for constructing extensive 3D-text datasets, facilitating a broader range of applications. However, existing methods sometimes lead to the generation of hallucinated captions, compromising caption quality. This paper explores the issue of hallucination in 3D object captioning, with a focus on Cap3D method, which renders 3D objects into 2D views for captioning using pre-trained models. We pinpoint a major challenge: certain rendered views of 3D objects are atypical, deviating from the training data of standard image captioning models and causing hallucinations. To tackle this, we present DiffuRank, a method that leverages a pre-trained text-to-3D model to assess the alignment between 3D objects and their 2D rendered views, where the view with high alignment closely represent the object's characteristics. By ranking all rendered views and feeding the top-ranked ones into GPT4-Vision, we enhance the accuracy and detail of captions, enabling the correction of 200k captions in the Cap3D dataset and extending it to 1 million captions across Objaverse and Objaverse-XL datasets. Additionally, we showcase the adaptability of DiffuRank by applying it to pre-trained text-to-image models for a Visual Question Answering task, where it outperforms the CLIP model.
>
---
#### [replaced 108] PerspectiveNet: Multi-View Perception for Dynamic Scene Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.16824v2](http://arxiv.org/pdf/2410.16824v2)**

> **作者:** Vinh Nguyen
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Generating detailed descriptions from multiple cameras and viewpoints is challenging due to the complex and inconsistent nature of visual data. In this paper, we introduce PerspectiveNet, a lightweight yet efficient model for generating long descriptions across multiple camera views. Our approach utilizes a vision encoder, a compact connector module to convert visual features into a fixed-size tensor, and large language models (LLMs) to harness the strong natural language generation capabilities of LLMs. The connector module is designed with three main goals: mapping visual features onto LLM embeddings, emphasizing key information needed for description generation, and producing a fixed-size feature matrix. Additionally, we augment our solution with a secondary task, the correct frame sequence detection, enabling the model to search for the correct sequence of frames to generate descriptions. Finally, we integrate the connector module, the secondary task, the LLM, and a visual feature extraction model into a single architecture, which is trained for the Traffic Safety Description and Analysis task. This task requires generating detailed, fine-grained descriptions of events from multiple cameras and viewpoints. The resulting model is lightweight, ensuring efficient training and inference, while remaining highly effective.
>
---
#### [replaced 109] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08136v2](http://arxiv.org/pdf/2507.08136v2)**

> **作者:** Chong Cheng; Yu Hu; Sicheng Yu; Beizhen Zhao; Zijian Wang; Hao Wang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated its potential in reconstructing scenes from unposed images. However, optimization-based 3DGS methods struggle with sparse views due to limited prior knowledge. Meanwhile, feed-forward Gaussian approaches are constrained by input formats, making it challenging to incorporate more input views. To address these challenges, we propose RegGS, a 3D Gaussian registration-based framework for reconstructing unposed sparse views. RegGS aligns local 3D Gaussians generated by a feed-forward network into a globally consistent 3D Gaussian representation. Technically, we implement an entropy-regularized Sinkhorn algorithm to efficiently solve the optimal transport Mixture 2-Wasserstein $(\text{MW}_2)$ distance, which serves as an alignment metric for Gaussian mixture models (GMMs) in $\mathrm{Sim}(3)$ space. Furthermore, we design a joint 3DGS registration module that integrates the $\text{MW}_2$ distance, photometric consistency, and depth geometry. This enables a coarse-to-fine registration process while accurately estimating camera poses and aligning the scene. Experiments on the RE10K and ACID datasets demonstrate that RegGS effectively registers local Gaussians with high fidelity, achieving precise pose estimation and high-quality novel-view synthesis. Project page: https://3dagentworld.github.io/reggs/.
>
---
#### [replaced 110] Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.11061v2](http://arxiv.org/pdf/2507.11061v2)**

> **作者:** Hayeon Kim; Ji Ha Jang; Se Young Chun
>
> **摘要:** Recent advances in 3D neural representations and instance-level editing models have enabled the efficient creation of high-quality 3D content. However, achieving precise local 3D edits remains challenging, especially for Gaussian Splatting, due to inconsistent multi-view 2D part segmentations and inherently ambiguous nature of Score Distillation Sampling (SDS) loss. To address these limitations, we propose RoMaP, a novel local 3D Gaussian editing framework that enables precise and drastic part-level modifications. First, we introduce a robust 3D mask generation module with our 3D-Geometry Aware Label Prediction (3D-GALP), which uses spherical harmonics (SH) coefficients to model view-dependent label variations and soft-label property, yielding accurate and consistent part segmentations across viewpoints. Second, we propose a regularized SDS loss that combines the standard SDS loss with additional regularizers. In particular, an L1 anchor loss is introduced via our Scheduled Latent Mixing and Part (SLaMP) editing method, which generates high-quality part-edited 2D images and confines modifications only to the target region while preserving contextual coherence. Additional regularizers, such as Gaussian prior removal, further improve flexibility by allowing changes beyond the existing context, and robust 3D masking prevents unintended edits. Experimental results demonstrate that our RoMaP achieves state-of-the-art local 3D editing on both reconstructed and generated Gaussian scenes and objects qualitatively and quantitatively, making it possible for more robust and flexible part-level 3D Gaussian editing. Code is available at https://janeyeon.github.io/romap.
>
---
#### [replaced 111] DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11558v3](http://arxiv.org/pdf/2506.11558v3)**

> **作者:** Bo-Cheng Chiu; Jen-Jee Chen; Yu-Chee Tseng; Feng-Chi Chen
>
> **摘要:** Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with LLM-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
>
---
#### [replaced 112] Growing a Twig to Accelerate Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14075v2](http://arxiv.org/pdf/2503.14075v2)**

> **作者:** Zhenwei Shao; Mingyang Wang; Zhou Yu; Wenwen Pan; Yan Yang; Tao Wei; Hongyuan Zhang; Ning Mao; Wei Chen; Jun Yu
>
> **备注:** accepted at ICCV 2025
>
> **摘要:** Large vision-language models (VLMs) have demonstrated remarkable capabilities in open-world multimodal understanding, yet their high computational overheads pose great challenges for practical deployment. Some recent works have proposed methods to accelerate VLMs by pruning redundant visual tokens guided by the attention maps of VLM's early layers. Despite the success of these token pruning methods, they still suffer from two major shortcomings: (i) considerable accuracy drop due to insensitive attention signals in early layers, and (ii) limited speedup when generating long responses (e.g., 30 tokens). To address the limitations above, we present TwigVLM -- a simple and general architecture by growing a lightweight twig upon an early layer of the base VLM. Compared with most existing VLM acceleration methods purely based on visual token pruning, our TwigVLM not only achieves better accuracy retention by employing a twig-guided token pruning (TTP) strategy, but also yields higher generation speed by utilizing a self-speculative decoding (SSD) strategy. Taking LLaVA-1.5-7B as the base VLM, experimental results show that TwigVLM preserves 96% of the original performance after pruning 88.9% of visual tokens and achieves 154% speedup in generating long responses, delivering significantly better performance in terms of both accuracy and speed over the state-of-the-art VLM acceleration methods.
>
---
#### [replaced 113] Learning Granularity-Aware Affordances from Human-Object Interaction for Tool-Based Functional Dexterous Grasping
- **分类: cs.RO; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2407.00614v2](http://arxiv.org/pdf/2407.00614v2)**

> **作者:** Fan Yang; Wenrui Chen; Kailun Yang; Haoran Lin; Dongsheng Luo; Conghui Tang; Zhiyong Li; Yaonan Wang
>
> **备注:** Accepted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS). The source code and the established dataset are available at https://github.com/yangfan293/GAAF-DEX
>
> **摘要:** To enable robots to use tools, the initial step is teaching robots to employ dexterous gestures for touching specific areas precisely where tasks are performed. Affordance features of objects serve as a bridge in the functional interaction between agents and objects. However, leveraging these affordance cues to help robots achieve functional tool grasping remains unresolved. To address this, we propose a granularity-aware affordance feature extraction method for locating functional affordance areas and predicting dexterous coarse gestures. We study the intrinsic mechanisms of human tool use. On one hand, we use fine-grained affordance features of object-functional finger contact areas to locate functional affordance regions. On the other hand, we use highly activated coarse-grained affordance features in hand-object interaction regions to predict grasp gestures. Additionally, we introduce a model-based post-processing module that transforms affordance localization and gesture prediction into executable robotic actions. This forms GAAF-Dex, a complete framework that learns Granularity-Aware Affordances from human-object interaction to enable tool-based functional grasping with dexterous hands. Unlike fully-supervised methods that require extensive data annotation, we employ a weakly supervised approach to extract relevant cues from exocentric (Exo) images of hand-object interactions to supervise feature extraction in egocentric (Ego) images. To support this approach, we have constructed a small-scale dataset, Functional Affordance Hand-object Interaction Dataset (FAH), which includes nearly 6K images of functional hand-object interaction Exo images and Ego images. Extensive experiments on the dataset demonstrate that our method outperforms state-of-the-art methods. The source code and the established dataset are available at https://github.com/yangfan293/GAAF-DEX.
>
---
#### [replaced 114] AutoGen Driven Multi Agent Framework for Iterative Crime Data Analysis and Prediction
- **分类: cs.MA; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11475v2](http://arxiv.org/pdf/2506.11475v2)**

> **作者:** Syeda Kisaa Fatima; Tehreem Zubair; Noman Ahmed; Asifullah Khan
>
> **摘要:** This paper introduces LUCID-MA (Learning and Understanding Crime through Dialogue of Multiple Agents), an innovative AI powered framework where multiple AI agents collaboratively analyze and understand crime data. Our system that consists of three core components: an analysis assistant that highlights spatiotemporal crime patterns; a feedback component that reviews and refines analytical results; and a prediction component that forecasts future crime trends. With a well-designed prompt and the LLaMA-2-13B-Chat-GPTQ model, it runs completely offline and allows the agents undergo self-improvement through 100 rounds of communication with less human interaction. A scoring function is incorporated to evaluate agent performance, providing visual plots to track learning progress. This work demonstrates the potential of AutoGen-style agents for autonomous, scalable, and iterative analysis in social science domains, maintaining data privacy through offline execution. It also showcases a computational model with emergent intelligence, where the system's global behavior emerges from the interactions of its agents. This emergent behavior manifests as enhanced individual agent performance, driven by collaborative dialogue between the LLM-based agents.
>
---
#### [replaced 115] VisualSpeaker: Visually-Guided 3D Avatar Lip Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06060v2](http://arxiv.org/pdf/2507.06060v2)**

> **作者:** Alexandre Symeonidis-Herzig; Özge Mercanoğlu Sincan; Richard Bowden
>
> **备注:** Accepted in International Conference on Computer Vision (ICCV) Workshops
>
> **摘要:** Realistic, high-fidelity 3D facial animations are crucial for expressive avatar systems in human-computer interaction and accessibility. Although prior methods show promising quality, their reliance on the mesh domain limits their ability to fully leverage the rapid visual innovations seen in 2D computer vision and graphics. We propose VisualSpeaker, a novel method that bridges this gap using photorealistic differentiable rendering, supervised by visual speech recognition, for improved 3D facial animation. Our contribution is a perceptual lip-reading loss, derived by passing photorealistic 3D Gaussian Splatting avatar renders through a pre-trained Visual Automatic Speech Recognition model during training. Evaluation on the MEAD dataset demonstrates that VisualSpeaker improves both the standard Lip Vertex Error metric by 56.1% and the perceptual quality of the generated animations, while retaining the controllability of mesh-driven animation. This perceptual focus naturally supports accurate mouthings, essential cues that disambiguate similar manual signs in sign language avatars.
>
---
#### [replaced 116] Revisiting Pool-based Prompt Learning for Few-shot Class-incremental Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09183v2](http://arxiv.org/pdf/2507.09183v2)**

> **作者:** Yongwei Jiang; Yixiong Zou; Yuhua Li; Ruixuan Li
>
> **备注:** Accepted to ICCV 2025, 11 pages
>
> **摘要:** Few-Shot Class-Incremental Learning (FSCIL) faces dual challenges of data scarcity and incremental learning in real-world scenarios. While pool-based prompting methods have demonstrated success in traditional incremental learning, their effectiveness in FSCIL settings remains unexplored. This paper presents the first study of current prompt pool methods in FSCIL tasks, revealing an unanticipated performance degradation in incremental sessions. Through comprehensive analysis, we identify that this phenomenon stems from token-dimension saturation: with limited data, excessive prompts compete for task-relevant information, leading to model overfitting. Based on this finding, we propose LGSP-Prompt (Local-Global Spatial Prompting), which innovatively shifts pool-based prompt learning from the token dimension to the spatial dimension. LGSP-Prompt generates spatial prompts by synergistically combining local spatial features and global frequency-domain representations to highlight key patterns in input images. We construct two spatial prompt pools enabling dynamic prompt selection to maintain acquired knowledge while effectively learning novel sessions. Extensive experiments demonstrate that our approach achieves state-of-the-art performance across multiple FSCIL benchmarks, showing significant advantages in both base knowledge preservation and incremental learning. Our implementation is available at https://github.com/Jywsuperman/LGSP.
>
---
#### [replaced 117] EgoM2P: Egocentric Multimodal Multitask Pretraining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07886v3](http://arxiv.org/pdf/2506.07886v3)**

> **作者:** Gen Li; Yutong Chen; Yiqian Wu; Kaifeng Zhao; Marc Pollefeys; Siyu Tang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Understanding multimodal signals in egocentric vision, such as RGB video, depth, camera poses, and gaze, is essential for applications in augmented reality, robotics, and human-computer interaction, enabling systems to better interpret the camera wearer's actions, intentions, and surrounding environment. However, building large-scale egocentric multimodal and multitask models presents unique challenges. Egocentric data are inherently heterogeneous, with large variations in modality coverage across devices and settings. Generating pseudo-labels for missing modalities, such as gaze or head-mounted camera trajectories, is often infeasible, making standard supervised learning approaches difficult to scale. Furthermore, dynamic camera motion and the complex temporal and spatial structure of first-person video pose additional challenges for the direct application of existing multimodal foundation models. To address these challenges, we introduce a set of efficient temporal tokenizers and propose EgoM2P, a masked modeling framework that learns from temporally-aware multimodal tokens to train a large, general-purpose model for egocentric 4D understanding. This unified design supports multitasking across diverse egocentric perception and synthesis tasks, including gaze prediction, egocentric camera tracking, and monocular depth estimation from egocentric video, and also serves as a generative model for conditional egocentric video synthesis. Across these tasks, EgoM2P matches or outperforms specialist models while being an order of magnitude faster. We will fully open-source EgoM2P to support the community and advance egocentric vision research. Project page: https://egom2p.github.io/.
>
---
#### [replaced 118] Point'n Move: Interactive Scene Object Manipulation on Gaussian Splatting Radiance Fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.16737v2](http://arxiv.org/pdf/2311.16737v2)**

> **作者:** Jiajun Huang; Hongchuan Yu
>
> **备注:** Code: https://github.com/jhuangBU/pnm
>
> **摘要:** We propose Point'n Move, a method that achieves interactive scene object manipulation with exposed region inpainting. Interactivity here further comes from intuitive object selection and real-time editing. To achieve this, we adopt Gaussian Splatting Radiance Field as the scene representation and fully leverage its explicit nature and speed advantage. Its explicit representation formulation allows us to devise a 2D prompt points to 3D mask dual-stage self-prompting segmentation algorithm, perform mask refinement and merging, minimize change as well as provide good initialization for scene inpainting and perform editing in real-time without per-editing training, all leads to superior quality and performance. We test our method by performing editing on both forward-facing and 360 scenes. We also compare our method against existing scene object removal methods, showing superior quality despite being more capable and having a speed advantage.
>
---
#### [replaced 119] FedWSQ: Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23516v2](http://arxiv.org/pdf/2506.23516v2)**

> **作者:** Seung-Wook Kim; Seongyeol Kim; Jiah Kim; Seowon Ji; Se-Ho Lee
>
> **摘要:** Federated learning (FL) often suffers from performance degradation due to key challenges such as data heterogeneity and communication constraints. To address these limitations, we present a novel FL framework called FedWSQ, which integrates weight standardization (WS) and the proposed distribution-aware non-uniform quantization (DANUQ). WS enhances FL performance by filtering out biased components in local updates during training, thereby improving the robustness of the model against data heterogeneity and unstable client participation. In addition, DANUQ minimizes quantization errors by leveraging the statistical properties of local model updates. As a result, FedWSQ significantly reduces communication overhead while maintaining superior model accuracy. Extensive experiments on FL benchmark datasets demonstrate that FedWSQ consistently outperforms existing FL methods across various challenging FL settings, including extreme data heterogeneity and ultra-low-bit communication scenarios.
>
---
#### [replaced 120] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.01425v3](http://arxiv.org/pdf/2501.01425v3)**

> **作者:** Xincheng Shuai; Henghui Ding; Zhenyuan Qin; Hao Luo; Xingjun Ma; Dacheng Tao
>
> **备注:** ICCV 2025, Project Page: https://henghuiding.github.io/SynFMC/
>
> **摘要:** Controlling the movements of dynamic objects and the camera within generated videos is a meaningful yet challenging task. Due to the lack of datasets with comprehensive 6D pose annotations, existing text-to-video methods can not simultaneously control the motions of both camera and objects in 3D-aware manner, resulting in limited controllability over generated contents. To address this issue and facilitate the research in this field, we introduce a Synthetic Dataset for Free-Form Motion Control (SynFMC). The proposed SynFMC dataset includes diverse object and environment categories and covers various motion patterns according to specific rules, simulating common and complex real-world scenarios. The complete 6D pose information facilitates models learning to disentangle the motion effects from objects and the camera in a video.~To provide precise 3D-aware motion control, we further propose a method trained on SynFMC, Free-Form Motion Control (FMC). FMC can control the 6D poses of objects and camera independently or simultaneously, producing high-fidelity videos. Moreover, it is compatible with various personalized text-to-image (T2I) models for different content styles. Extensive experiments demonstrate that the proposed FMC outperforms previous methods across multiple scenarios.
>
---
#### [replaced 121] ChartQA-X: Generating Explanations for Visual Chart Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13275v2](http://arxiv.org/pdf/2504.13275v2)**

> **作者:** Shamanthak Hegde; Pooyan Fazli; Hasti Seifi
>
> **摘要:** The ability to explain complex information from chart images is vital for effective data-driven decision-making. In this work, we address the challenge of generating detailed explanations alongside answering questions about charts. We present ChartQA-X, a comprehensive dataset comprising 30,299 chart samples across four chart types, each paired with contextually relevant questions, answers, and explanations. Explanations are generated and selected based on metrics such as faithfulness, informativeness, coherence, and perplexity. Our human evaluation with 245 participants shows that model-generated explanations in ChartQA-X surpass human-written explanations in accuracy and logic and are comparable in terms of clarity and overall quality. Moreover, models fine-tuned on ChartQA-X show substantial improvements across various metrics, including absolute gains of up to 24.57 points in explanation quality, 18.96 percentage points in question-answering accuracy, and 14.75 percentage points on unseen benchmarks for the same task. By integrating explanatory narratives with answers, our approach enables agents to communicate complex visual information more effectively, improving comprehension and fostering greater trust in the generated responses.
>
---
#### [replaced 122] HER-Seg: Holistically Efficient Segmentation for High-Resolution Medical Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06205v2](http://arxiv.org/pdf/2504.06205v2)**

> **作者:** Qing Xu; Zhenye Lou; Chenxin Li; Yue Li; Xiangjian He; Tesema Fiseha Berhanu; Rong Qu; Wenting Duan; Zhen Chen
>
> **备注:** Under Review
>
> **摘要:** High-resolution segmentation is critical for precise disease diagnosis by extracting fine-grained morphological details. Existing hierarchical encoder-decoder frameworks have demonstrated remarkable adaptability across diverse medical segmentation tasks. While beneficial, they usually require the huge computation and memory cost when handling large-size segmentation, which limits their applications in foundation model building and real-world clinical scenarios. To address this limitation, we propose a holistically efficient framework for high-resolution medical image segmentation, called HER-Seg. Specifically, we first devise a computation-efficient image encoder (CE-Encoder) to model long-range dependencies with linear complexity while maintaining sufficient representations. In particular, we introduce the dual-gated linear attention (DLA) mechanism to perform cascaded token filtering, selectively retaining important tokens while ignoring irrelevant ones to enhance attention computation efficiency. Then, we introduce a memory-efficient mask decoder (ME-Decoder) to eliminate the demand for the hierarchical structure by leveraging cross-scale segmentation decoding. Extensive experiments reveal that HER-Seg outperforms state-of-the-arts in high-resolution medical 2D, 3D and video segmentation tasks. In particular, our HER-Seg requires only 0.59GB training GPU memory and 9.39G inference FLOPs per 1024$\times$1024 image, demonstrating superior memory and computation efficiency. The code is available at https://github.com/xq141839/HER-Seg.
>
---
#### [replaced 123] Cube: A Roblox View of 3D Intelligence
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.15475v3](http://arxiv.org/pdf/2503.15475v3)**

> **作者:** Foundation AI Team; Kiran Bhat; Nishchaie Khanna; Karun Channa; Tinghui Zhou; Yiheng Zhu; Xiaoxia Sun; Charles Shang; Anirudh Sudarshan; Maurice Chu; Daiqing Li; Kangle Deng; Jean-Philippe Fauconnier; Tijmen Verhulsdonck; Maneesh Agrawala; Kayvon Fatahalian; Alexander Weiss; Christian Reiser; Ravi Kiran Chirravuri; Ravali Kandur; Alejandro Pelaez; Akash Garg; Michael Palleschi; Jessica Wang; Skylar Litz; Leon Liu; Anying Li; David Harmon; Derek Liu; Liangjun Feng; Denis Goupil; Lukas Kuczynski; Jihyun Yoon; Naveen Marri; Peiye Zhuang; Yinan Zhang; Brian Yin; Haomiao Jiang; Marcel van Workum; Thomas Lane; Bryce Erickson; Salil Pathare; Kyle Price; Steve Han; Yiqing Wang; Anupam Singh; David Baszucki
>
> **备注:** Our code and model weights can be found at: https://github.com/Roblox/cube
>
> **摘要:** Foundation models trained on vast amounts of data have demonstrated remarkable reasoning and generation capabilities in the domains of text, images, audio and video. Our goal at Roblox is to build such a foundation model for 3D intelligence, a model that can support developers in producing all aspects of a Roblox experience, from generating 3D objects and scenes to rigging characters for animation to producing programmatic scripts describing object behaviors. We discuss three key design requirements for such a 3D foundation model and then present our first step towards building such a model. We expect that 3D geometric shapes will be a core data type and describe our solution for 3D shape tokenizer. We show how our tokenization scheme can be used in applications for text-to-shape generation, shape-to-text generation and text-to-scene generation. We demonstrate how these applications can collaborate with existing large language models (LLMs) to perform scene analysis and reasoning. We conclude with a discussion outlining our path to building a fully unified foundation model for 3D intelligence.
>
---
#### [replaced 124] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.06273v2](http://arxiv.org/pdf/2503.06273v2)**

> **作者:** Jeong Hun Yeo; Minsu Kim; Chae Won Kim; Stavros Petridis; Yong Man Ro
>
> **备注:** Accepted at ICCV 2025. Code available at: https://github.com/JeongHun0716/zero-avsr
>
> **摘要:** We explore a novel zero-shot Audio-Visual Speech Recognition (AVSR) framework, dubbed Zero-AVSR, which enables speech recognition in target languages without requiring any audio-visual speech data in those languages. Specifically, we introduce the Audio-Visual Speech Romanizer (AV-Romanizer), which learns language-agnostic speech representations by predicting Roman text. Then, by leveraging the strong multilingual modeling capabilities of Large Language Models (LLMs), we propose converting the predicted Roman text into language-specific graphemes, forming the proposed Cascaded Zero-AVSR. Taking it a step further, we explore a unified Zero-AVSR approach by directly integrating the audio-visual speech representations encoded by the AV-Romanizer into the LLM. This is achieved through finetuning the adapter and the LLM using our proposed multi-task learning scheme. To capture the wide spectrum of phonetic and linguistic diversity, we also introduce a Multilingual Audio-Visual Romanized Corpus (MARC) consisting of 2,916 hours of audio-visual speech data across 82 languages, along with transcriptions in both language-specific graphemes and Roman text. Extensive analysis and experiments confirm that the proposed Zero-AVSR framework has the potential to expand language support beyond the languages seen during the training of the AV-Romanizer.
>
---
#### [replaced 125] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07485v2](http://arxiv.org/pdf/2507.07485v2)**

> **作者:** Wooseong Jeong; Kuk-Jin Yoon
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Multi-Task Learning (MTL) enables multiple tasks to be learned within a shared network, but differences in objectives across tasks can cause negative transfer, where the learning of one task degrades another task's performance. While pre-trained transformers significantly improve MTL performance, their fixed network capacity and rigid structure limit adaptability. Previous dynamic network architectures attempt to address this but are inefficient as they directly convert shared parameters into task-specific ones. We propose Dynamic Token Modulation and Expansion (DTME-MTL), a framework applicable to any transformer-based MTL architecture. DTME-MTL enhances adaptability and reduces overfitting by identifying gradient conflicts in token space and applying adaptive solutions based on conflict type. Unlike prior methods that mitigate negative transfer by duplicating network parameters, DTME-MTL operates entirely in token space, enabling efficient adaptation without excessive parameter growth. Extensive experiments demonstrate that DTME-MTL consistently improves multi-task performance with minimal computational overhead, offering a scalable and effective solution for enhancing transformer-based MTL models.
>
---
#### [replaced 126] Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08622v2](http://arxiv.org/pdf/2505.08622v2)**

> **作者:** Donghoon Kim; Minji Bae; Kyuhong Shim; Byonghyo Shim
>
> **备注:** ICLR 2025 (Official Code: https://github.com/DonghoonKim-1938/VGD)
>
> **摘要:** Text-to-image generative models like DALL-E and Stable Diffusion have revolutionized visual content creation across various applications, including advertising, personalized media, and design prototyping. However, crafting effective textual prompts to guide these models remains challenging, often requiring extensive trial and error. Existing prompt inversion approaches, such as soft and hard prompt techniques, are not so effective due to the limited interpretability and incoherent prompt generation. To address these issues, we propose Visually Guided Decoding (VGD), a gradient-free approach that leverages large language models (LLMs) and CLIP-based guidance to generate coherent and semantically aligned prompts. In essence, VGD utilizes the robust text generation capabilities of LLMs to produce human-readable prompts. Further, by employing CLIP scores to ensure alignment with user-specified visual concepts, VGD enhances the interpretability, generalization, and flexibility of prompt generation without the need for additional training. Our experiments demonstrate that VGD outperforms existing prompt inversion techniques in generating understandable and contextually relevant prompts, facilitating more intuitive and controllable interactions with text-to-image models.
>
---
#### [replaced 127] PEMF-VTO: Point-Enhanced Video Virtual Try-on via Mask-free Paradigm
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.03021v5](http://arxiv.org/pdf/2412.03021v5)**

> **作者:** Tianyu Chang; Xiaohao Chen; Zhichao Wei; Xuanpu Zhang; Qing-Guo Chen; Weihua Luo; Peipei Song; Xun Yang
>
> **摘要:** Video Virtual Try-on aims to seamlessly transfer a reference garment onto a target person in a video while preserving both visual fidelity and temporal coherence. Existing methods typically rely on inpainting masks to define the try-on area, enabling accurate garment transfer for simple scenes (e.g., in-shop videos). However, these mask-based approaches struggle with complex real-world scenarios, as overly large and inconsistent masks often destroy spatial-temporal information, leading to distorted results. Mask-free methods alleviate this issue but face challenges in accurately determining the try-on area, especially for videos with dynamic body movements. To address these limitations, we propose PEMF-VTO, a novel Point-Enhanced Mask-Free Video Virtual Try-On framework that leverages sparse point alignments to explicitly guide garment transfer. Our key innovation is the introduction of point-enhanced guidance, which provides flexible and reliable control over both spatial-level garment transfer and temporal-level video coherence. Specifically, we design a Point-Enhanced Transformer (PET) with two core components: Point-Enhanced Spatial Attention (PSA), which uses frame-cloth point alignments to precisely guide garment transfer, and Point-Enhanced Temporal Attention (PTA), which leverages frame-frame point correspondences to enhance temporal coherence and ensure smooth transitions across frames. Extensive experiments demonstrate that our PEMF-VTO outperforms state-of-the-art methods, generating more natural, coherent, and visually appealing try-on videos, particularly for challenging in-the-wild scenarios. The link to our paper's homepage is https://pemf-vto.github.io/.
>
---
#### [replaced 128] Generalized Consistency Trajectory Models for Image Manipulation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.12510v4](http://arxiv.org/pdf/2403.12510v4)**

> **作者:** Beomsu Kim; Jaemin Kim; Jeongsol Kim; Jong Chul Ye
>
> **备注:** ICLR 2025 (poster)
>
> **摘要:** Diffusion models (DMs) excel in unconditional generation, as well as on applications such as image editing and restoration. The success of DMs lies in the iterative nature of diffusion: diffusion breaks down the complex process of mapping noise to data into a sequence of simple denoising tasks. Moreover, we are able to exert fine-grained control over the generation process by injecting guidance terms into each denoising step. However, the iterative process is also computationally intensive, often taking from tens up to thousands of function evaluations. Although consistency trajectory models (CTMs) enable traversal between any time points along the probability flow ODE (PFODE) and score inference with a single function evaluation, CTMs only allow translation from Gaussian noise to data. This work aims to unlock the full potential of CTMs by proposing generalized CTMs (GCTMs), which translate between arbitrary distributions via ODEs. We discuss the design space of GCTMs and demonstrate their efficacy in various image manipulation tasks such as image-to-image translation, restoration, and editing.
>
---
#### [replaced 129] OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16091v4](http://arxiv.org/pdf/2505.16091v4)**

> **作者:** Jinpei Guo; Yifei Ji; Zheng Chen; Kai Liu; Min Liu; Wang Rao; Wenbo Li; Yong Guo; Yulun Zhang
>
> **摘要:** Pretrained latent diffusion models have shown strong potential for lossy image compression, owing to their powerful generative priors. Most existing diffusion-based methods reconstruct images by iteratively denoising from random noise, guided by compressed latent representations. While these approaches have achieved high reconstruction quality, their multi-step sampling process incurs substantial computational overhead. Moreover, they typically require training separate models for different compression bit-rates, leading to significant training and storage costs. To address these challenges, we propose a one-step diffusion codec across multiple bit-rates. termed OSCAR. Specifically, our method views compressed latents as noisy variants of the original latents, where the level of distortion depends on the bit-rate. This perspective allows them to be modeled as intermediate states along a diffusion trajectory. By establishing a mapping from the compression bit-rate to a pseudo diffusion timestep, we condition a single generative model to support reconstructions at multiple bit-rates. Meanwhile, we argue that the compressed latents retain rich structural information, thereby making one-step denoising feasible. Thus, OSCAR replaces iterative sampling with a single denoising pass, significantly improving inference efficiency. Extensive experiments demonstrate that OSCAR achieves superior performance in both quantitative and visual quality metrics. The code and models will be released at https://github.com/jp-guo/OSCAR.
>
---
#### [replaced 130] How Cars Move: Analyzing Driving Dynamics for Safer Urban Traffic
- **分类: cs.CV; cs.PF; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.04020v3](http://arxiv.org/pdf/2412.04020v3)**

> **作者:** Kangan Qian; Jinyu Miao; Xinyu Jiao; Ziang Luo; Zheng Fu; Yining Shi; Yunlong Wang; Kun Jiang; Diange Yang
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Understanding the spatial dynamics of cars within urban systems is essential for optimizing infrastructure management and resource allocation. Recent empirical approaches for analyzing traffic patterns have gained traction due to their applicability to city-scale policy development. However, conventional methodologies often rely on fragmented grid-based techniques, which may overlook critical interdependencies among spatial elements and temporal continuity. These limitations can compromise analytical effectiveness in complex urban environments. To address these challenges, we propose PriorMotion, a data integration framework designed to systematically uncover movement patterns through driving dynamics analysis. Our approach combines multi-scale empirical observations with customized analytical tools to capture evolving spatial-temporal trends in urban traffic. Comprehensive evaluations demonstrate that PriorMotion significantly enhances analytical outcomes, including increased accuracy in traffic pattern analysis, improved adaptability to heterogeneous data environments, and reduced long-term projection errors. Validation confirms its effectiveness for urban infrastructure management applications requiring precise characterization of complex spatial-temporal interactions.
>
---
#### [replaced 131] DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.02192v2](http://arxiv.org/pdf/2505.02192v2)**

> **作者:** Wenchuan Wang; Mengqi Huang; Yijing Tu; Zhendong Mao
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Customized text-to-video generation with pre-trained large-scale models has recently garnered significant attention by focusing on identity and motion consistency. Existing works typically follow the isolated customized paradigm, where the subject identity or motion dynamics are customized exclusively. However, this paradigm completely ignores the intrinsic mutual constraints and synergistic interdependencies between identity and motion, resulting in identity-motion conflicts throughout the generation process that systematically degrade. To address this, we introduce DualReal, a novel framework that employs adaptive joint training to construct interdependencies between dimensions collaboratively. Specifically, DualReal is composed of two units: (1) Dual-aware Adaptation dynamically switches the training step (i.e., identity or motion), learns the current information guided by the frozen dimension prior, and employs a regularization strategy to avoid knowledge leakage; (2) StageBlender Controller leverages the denoising stages and Diffusion Transformer depths to guide different dimensions with adaptive granularity, avoiding conflicts at various stages and ultimately achieving lossless fusion of identity and motion patterns. We constructed a more comprehensive evaluation benchmark than existing methods. The experimental results show that DualReal improves CLIP-I and DINO-I metrics by 21.7% and 31.8% on average, and achieves top performance on nearly all motion metrics. Page: https://wenc-k.github.io/dualreal-customization
>
---
#### [replaced 132] Learning from SAM: Harnessing a Foundation Model for Sim2Real Adaptation by Regularization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2309.15562v4](http://arxiv.org/pdf/2309.15562v4)**

> **作者:** Mayara E. Bonani; Max Schwarz; Sven Behnke
>
> **备注:** Accepted for IEEE International Conference on Automation Science and Engineering (CASE) 2025. M. E. Bonani and M. Schwarz contributed equally
>
> **摘要:** Domain adaptation is especially important for robotics applications, where target domain training data is usually scarce and annotations are costly to obtain. We present a method for self-supervised domain adaptation for the scenario where annotated source domain data (e.g. from synthetic generation) is available, but the target domain data is completely unannotated. Our method targets the semantic segmentation task and leverages a segmentation foundation model (Segment Anything Model) to obtain segment information on unannotated data. We take inspiration from recent advances in unsupervised local feature learning and propose an invariance-variance loss over the detected segments for regularizing feature representations in the target domain. Crucially, this loss structure and network architecture can handle overlapping segments and oversegmentation as produced by Segment Anything. We demonstrate the advantage of our method on the challenging YCB-Video and HomebrewedDB datasets and show that it outperforms prior work and, on YCB-Video, even a network trained with real annotations. Additionally, we provide insight through model ablations and show applicability to a custom robotic application.
>
---
#### [replaced 133] MARS: a Multimodal Alignment and Ranking System for Few-Shot Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07942v2](http://arxiv.org/pdf/2504.07942v2)**

> **作者:** Nico Catalano; Stefano Samele; Paolo Pertino; Matteo Matteucci
>
> **摘要:** Few Shot Segmentation aims to segment novel object classes given only a handful of labeled examples, enabling rapid adaptation with minimal supervision. Current literature crucially lacks a selection method that goes beyond visual similarity between the query and example images, leading to suboptimal predictions. We present MARS, a plug-and-play ranking system that leverages multimodal cues to filter and merge mask proposals robustly. Starting from a set of mask predictions for a single query image, we score, filter, and merge them to improve results. Proposals are evaluated using multimodal scores computed at local and global levels. Extensive experiments on COCO-20i, Pascal-5i, LVIS-92i, and FSS-1000 demonstrate that integrating all four scoring components is crucial for robust ranking, validating our contribution. As MARS can be effortlessly integrated with various mask proposal systems, we deploy it across a wide range of top-performer methods and achieve new state-of-the-art results on multiple existing benchmarks. Code will be available upon acceptance.
>
---
