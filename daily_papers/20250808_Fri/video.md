# 计算机视觉 cs.CV

- **最新发布 127 篇**

- **更新 81 篇**

## 最新发布

#### [new 001] ReasoningTrack: Chain-of-Thought Reasoning for Long-term Vision-Language Tracking
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出基于链式思维的视觉-语言跟踪框架ReasoningTrack，解决传统方法对目标动态变化的感知与推理不足问题，通过融合预训练视觉-语言模型、强化学习优化推理流程及统一特征提取网络，构建了长时视觉-语言跟踪数据集TNLLT，并验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.05221v1](http://arxiv.org/pdf/2508.05221v1)**

> **作者:** Xiao Wang; Liye Jin; Xufeng Lou; Shiao Wang; Lan Chen; Bo Jiang; Zhipeng Zhang
>
> **摘要:** Vision-language tracking has received increasing attention in recent years, as textual information can effectively address the inflexibility and inaccuracy associated with specifying the target object to be tracked. Existing works either directly fuse the fixed language with vision features or simply modify using attention, however, their performance is still limited. Recently, some researchers have explored using text generation to adapt to the variations in the target during tracking, however, these works fail to provide insights into the model's reasoning process and do not fully leverage the advantages of large models, which further limits their overall performance. To address the aforementioned issues, this paper proposes a novel reasoning-based vision-language tracking framework, named ReasoningTrack, based on a pre-trained vision-language model Qwen2.5-VL. Both SFT (Supervised Fine-Tuning) and reinforcement learning GRPO are used for the optimization of reasoning and language generation. We embed the updated language descriptions and feed them into a unified tracking backbone network together with vision features. Then, we adopt a tracking head to predict the specific location of the target object. In addition, we propose a large-scale long-term vision-language tracking benchmark dataset, termed TNLLT, which contains 200 video sequences. 20 baseline visual trackers are re-trained and evaluated on this dataset, which builds a solid foundation for the vision-language visual tracking task. Extensive experiments on multiple vision-language tracking benchmark datasets fully validated the effectiveness of our proposed reasoning-based natural language generation strategy. The source code of this paper will be released on https://github.com/Event-AHU/Open_VLTrack
>
---
#### [new 002] Head Anchor Enhanced Detection and Association for Crowded Pedestrian Tracking
- **分类: cs.CV**

- **简介: 该论文旨在解决密集行人跟踪中的遮挡问题，提出基于头部锚点增强的检测与关联方法，融合多分支检测特征和迭代Kalman滤波，提升复杂场景下的多目标跟踪鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.05514v1](http://arxiv.org/pdf/2508.05514v1)**

> **作者:** Zewei Wu; César Teixeira; Wei Ke; Zhang Xiong
>
> **摘要:** Visual pedestrian tracking represents a promising research field, with extensive applications in intelligent surveillance, behavior analysis, and human-computer interaction. However, real-world applications face significant occlusion challenges. When multiple pedestrians interact or overlap, the loss of target features severely compromises the tracker's ability to maintain stable trajectories. Traditional tracking methods, which typically rely on full-body bounding box features extracted from {Re-ID} models and linear constant-velocity motion assumptions, often struggle in severe occlusion scenarios. To address these limitations, this work proposes an enhanced tracking framework that leverages richer feature representations and a more robust motion model. Specifically, the proposed method incorporates detection features from both the regression and classification branches of an object detector, embedding spatial and positional information directly into the feature representations. To further mitigate occlusion challenges, a head keypoint detection model is introduced, as the head is less prone to occlusion compared to the full body. In terms of motion modeling, we propose an iterative Kalman filtering approach designed to align with modern detector assumptions, integrating 3D priors to better complete motion trajectories in complex scenes. By combining these advancements in appearance and motion modeling, the proposed method offers a more robust solution for multi-object tracking in crowded environments where occlusions are prevalent.
>
---
#### [new 003] Segmenting the Complex and Irregular in Two-Phase Flows: A Real-World Empirical Study with SAM2
- **分类: cs.CV; 68T45, 94A08; I.2.10**

- **简介: 该论文旨在解决多相流中复杂非凸气泡的分割问题，通过将任务转化为视觉基础模型的转移学习问题，首次验证了SAM v2.1在低标注场景下的高效分割能力。**

- **链接: [http://arxiv.org/pdf/2508.05227v1](http://arxiv.org/pdf/2508.05227v1)**

> **作者:** Semanur Küçük; Cosimo Della Santina; Angeliki Laskari
>
> **备注:** 7 pages
>
> **摘要:** Segmenting gas bubbles in multiphase flows is a critical yet unsolved challenge in numerous industrial settings, from metallurgical processing to maritime drag reduction. Traditional approaches-and most recent learning-based methods-assume near-spherical shapes, limiting their effectiveness in regimes where bubbles undergo deformation, coalescence, or breakup. This complexity is particularly evident in air lubrication systems, where coalesced bubbles form amorphous and topologically diverse patches. In this work, we revisit the problem through the lens of modern vision foundation models. We cast the task as a transfer learning problem and demonstrate, for the first time, that a fine-tuned Segment Anything Model SAM v2.1 can accurately segment highly non-convex, irregular bubble structures using as few as 100 annotated images.
>
---
#### [new 004] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文探讨了压缩模型在对抗攻击中的挑战，提出白盒/自适应攻击，并发现高真实重建显著提升攻击难度，证明真实性对鲁棒性至关重要，表明未来需突破真实性障碍以提升安全评估。**

- **链接: [http://arxiv.org/pdf/2508.05489v1](http://arxiv.org/pdf/2508.05489v1)**

> **作者:** Samuel Räber; Till Aczel; Andreas Plesner; Roger Wattenhofer
>
> **摘要:** Previous work has suggested that preprocessing images through lossy compression can defend against adversarial perturbations, but comprehensive attack evaluations have been lacking. In this paper, we construct strong white-box and adaptive attacks against various compression models and identify a critical challenge for attackers: high realism in reconstructed images significantly increases attack difficulty. Through rigorous evaluation across multiple attack scenarios, we demonstrate that compression models capable of producing realistic, high-fidelity reconstructions are substantially more resistant to our attacks. In contrast, low-realism compression models can be broken. Our analysis reveals that this is not due to gradient masking. Rather, realistic reconstructions maintaining distributional alignment with natural images seem to offer inherent robustness. This work highlights a significant obstacle for future adversarial attacks and suggests that developing more effective techniques to overcome realism represents an essential challenge for comprehensive security evaluation.
>
---
#### [new 005] CoMAD: A Multiple-Teacher Self-Supervised Distillation Framework
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CoMAD多教师自监督蒸馏框架，解决传统自监督学习在资源受限部署下的参数爆炸和知识融合不足问题。通过融合三款ViT-Base教师的特征，采用异步掩码机制并结合KL散度训练，优化了学生网络结构与性能，使ViT-Tiny在ImageNet-1K上提升至75.4%Top-1，达到紧凑SSL蒸馏的新突破。**

- **链接: [http://arxiv.org/pdf/2508.04816v1](http://arxiv.org/pdf/2508.04816v1)**

> **作者:** Sriram Mandalika; Lalitha V
>
> **备注:** 8 Pages, 2 Figures
>
> **摘要:** Numerous self-supervised learning paradigms, such as contrastive learning and masked image modeling, learn powerful representations from unlabeled data but are typically pretrained in isolation, overlooking complementary insights and yielding large models that are impractical for resource-constrained deployment. To overcome these challenges, we introduce Consensus-oriented Masked Distillation (CoMAD), a lightweight, parameter-free framework that unifies knowledge from multiple current state-of-the-art self-supervised Vision Transformers into a compact student network. CoMAD distills from three pretrained ViT-Base teachers, MAE, MoCo v3, and iBOT, each offering distinct semantic and contextual priors. Rather than naively averaging teacher outputs, we apply asymmetric masking: the student sees only 25 percent of patches while each teacher receives a progressively lighter, unique mask, forcing the student to interpolate missing features under richer contexts. Teacher embeddings are aligned to the student's space via a linear adapter and layer normalization, then fused through our joint consensus gating, which weights each token by combining cosine affinity with inter-teacher agreement. The student is trained with dual-level KL divergence on visible tokens and reconstructed feature maps, capturing both local and global structure. On ImageNet-1K, CoMAD's ViT-Tiny achieves 75.4 percent Top-1, an increment of 0.4 percent over the previous state-of-the-art. In dense-prediction transfers, it attains 47.3 percent mIoU on ADE20K, and 44.5 percent box average precision and 40.5 percent mask average precision on MS-COCO, establishing a new state-of-the-art in compact SSL distillation.
>
---
#### [new 006] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation
- **分类: cs.CV**

- **简介: 该论文提出一个基于In-Context LoRA的框架，解决身份漂移和时间限制问题，通过注入身份特征和姿态信息，生成长时可控制的视频。**

- **链接: [http://arxiv.org/pdf/2508.05091v1](http://arxiv.org/pdf/2508.05091v1)**

> **作者:** Jingxuan He; Busheng Su; Finn Wong
>
> **摘要:** Generating long, temporally coherent videos with precise control over subject identity and motion is a formidable challenge for current diffusion models, which often suffer from identity drift and are limited to short clips. We introduce PoseGen, a novel framework that generates arbitrarily long videos of a specific subject from a single reference image and a driving pose sequence. Our core innovation is an in-context LoRA finetuning strategy that injects subject appearance at the token level for identity preservation, while simultaneously conditioning on pose information at the channel level for fine-grained motion control. To overcome duration limits, PoseGen pioneers an interleaved segment generation method that seamlessly stitches video clips together, using a shared KV cache mechanism and a specialized transition process to ensure background consistency and temporal smoothness. Trained on a remarkably small 33-hour video dataset, extensive experiments show that PoseGen significantly outperforms state-of-the-art methods in identity fidelity, pose accuracy, and its unique ability to produce coherent, artifact-free videos of unlimited duration.
>
---
#### [new 007] mKG-RAG: Multimodal Knowledge Graph-Enhanced RAG for Visual Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出了一种基于多模态知识图谱的增强型生成框架（mKG-RAG），解决传统视觉问答任务中结构化知识获取不足的问题。通过结合语义提取与视觉-文本匹配技术，构建高精度多模态知识图谱，并设计双阶段检索策略提升效率。实验表明，该方法显著优于现有模型。**

- **链接: [http://arxiv.org/pdf/2508.05318v1](http://arxiv.org/pdf/2508.05318v1)**

> **作者:** Xu Yuan; Liangbo Ning; Wenqi Fan; Qing Li
>
> **摘要:** Recently, Retrieval-Augmented Generation (RAG) has been proposed to expand internal knowledge of Multimodal Large Language Models (MLLMs) by incorporating external knowledge databases into the generation process, which is widely used for knowledge-based Visual Question Answering (VQA) tasks. Despite impressive advancements, vanilla RAG-based VQA methods that rely on unstructured documents and overlook the structural relationships among knowledge elements frequently introduce irrelevant or misleading content, reducing answer accuracy and reliability. To overcome these challenges, a promising solution is to integrate multimodal knowledge graphs (KGs) into RAG-based VQA frameworks to enhance the generation by introducing structured multimodal knowledge. Therefore, in this paper, we propose a novel multimodal knowledge-augmented generation framework (mKG-RAG) based on multimodal KGs for knowledge-intensive VQA tasks. Specifically, our approach leverages MLLM-powered keyword extraction and vision-text matching to distill semantically consistent and modality-aligned entities/relationships from multimodal documents, constructing high-quality multimodal KGs as structured knowledge representations. In addition, a dual-stage retrieval strategy equipped with a question-aware multimodal retriever is introduced to improve retrieval efficiency while refining precision. Comprehensive experiments demonstrate that our approach significantly outperforms existing methods, setting a new state-of-the-art for knowledge-based VQA.
>
---
#### [new 008] Uni-cot: Towards Unified Chain-of-Thought Reasoning Across Text and Vision
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出 Uni-CoT 框架，解决跨模态推理（文本与图像）中因视觉状态过渡复杂性导致的方法局限问题。通过宏级高阶任务规划与微级子任务执行结合的双重推理机制及结构化训练策略，实现高效多模态推理，实验表明其在推理驱动生成与编辑任务中达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.05606v1](http://arxiv.org/pdf/2508.05606v1)**

> **作者:** Luozheng Qin; Jia Gong; Yuqing Sun; Tianjiao Li; Mengping Yang; Xiaomeng Yang; Chao Qu; Zhiyu Tan; Hao Li
>
> **备注:** https://sais-fuxi.github.io/projects/uni-cot/
>
> **摘要:** Chain-of-Thought (CoT) reasoning has been widely adopted to enhance Large Language Models (LLMs) by decomposing complex tasks into simpler, sequential subtasks. However, extending CoT to vision-language reasoning tasks remains challenging, as it often requires interpreting transitions of visual states to support reasoning. Existing methods often struggle with this due to limited capacity of modeling visual state transitions or incoherent visual trajectories caused by fragmented architectures. To overcome these limitations, we propose Uni-CoT, a Unified Chain-of-Thought framework that enables coherent and grounded multimodal reasoning within a single unified model. The key idea is to leverage a model capable of both image understanding and generation to reason over visual content and model evolving visual states. However, empowering a unified model to achieve that is non-trivial, given the high computational cost and the burden of training. To address this, Uni-CoT introduces a novel two-level reasoning paradigm: A Macro-Level CoT for high-level task planning and A Micro-Level CoT for subtask execution. This design significantly reduces the computational overhead. Furthermore, we introduce a structured training paradigm that combines interleaved image-text supervision for macro-level CoT with multi-task objectives for micro-level CoT. Together, these innovations allow Uni-CoT to perform scalable and coherent multi-modal reasoning. Furthermore, thanks to our design, all experiments can be efficiently completed using only 8 A100 GPUs with 80GB VRAM each. Experimental results on reasoning-driven image generation benchmark (WISE) and editing benchmarks (RISE and KRIS) indicates that Uni-CoT demonstrates SOTA performance and strong generalization, establishing Uni-CoT as a promising solution for multi-modal reasoning. Project Page and Code: https://sais-fuxi.github.io/projects/uni-cot/
>
---
#### [new 009] AU-IQA: A Benchmark Dataset for Perceptual Quality Assessment of AI-Enhanced User-Generated Content
- **分类: cs.CV; eess.IV**

- **简介: 该论文构建了AU-IQA基准数据集，用于评估AI增强的用户生成内容（AI-UGC）的感知质量。旨在填补AI-UGC性能评估空白，通过对比传统IQA与多模态模型，分析现有方法在融合增强特征场景下的表现。**

- **链接: [http://arxiv.org/pdf/2508.05016v1](http://arxiv.org/pdf/2508.05016v1)**

> **作者:** Shushi Wang; Chunyi Li; Zicheng Zhang; Han Zhou; Wei Dong; Jun Chen; Guangtao Zhai; Xiaohong Liu
>
> **摘要:** AI-based image enhancement techniques have been widely adopted in various visual applications, significantly improving the perceptual quality of user-generated content (UGC). However, the lack of specialized quality assessment models has become a significant limiting factor in this field, limiting user experience and hindering the advancement of enhancement methods. While perceptual quality assessment methods have shown strong performance on UGC and AIGC individually, their effectiveness on AI-enhanced UGC (AI-UGC) which blends features from both, remains largely unexplored. To address this gap, we construct AU-IQA, a benchmark dataset comprising 4,800 AI-UGC images produced by three representative enhancement types which include super-resolution, low-light enhancement, and denoising. On this dataset, we further evaluate a range of existing quality assessment models, including traditional IQA methods and large multimodal models. Finally, we provide a comprehensive analysis of how well current approaches perform in assessing the perceptual quality of AI-UGC. The access link to the AU-IQA is https://github.com/WNNGGU/AU-IQA-Dataset.
>
---
#### [new 010] When Deepfake Detection Meets Graph Neural Network:a Unified and Lightweight Learning Framework
- **分类: cs.CV**

- **简介: 该论文旨在解决深度伪造检测中的泛化能力不足问题，提出基于图神经网络的轻量化框架SSTGNN，通过结构化图表示视频并融合谱滤波与时间差分建模，有效捕捉空间、时序与光谱信息，实现跨域与鲁棒性提升，参数效率达42.4×。**

- **链接: [http://arxiv.org/pdf/2508.05526v1](http://arxiv.org/pdf/2508.05526v1)**

> **作者:** Haoyu Liu; Chaoyu Gong; Mengke He; Jiate Li; Kai Han; Siqiang Luo
>
> **备注:** 11 pages
>
> **摘要:** The proliferation of generative video models has made detecting AI-generated and manipulated videos an urgent challenge. Existing detection approaches often fail to generalize across diverse manipulation types due to their reliance on isolated spatial, temporal, or spectral information, and typically require large models to perform well. This paper introduces SSTGNN, a lightweight Spatial-Spectral-Temporal Graph Neural Network framework that represents videos as structured graphs, enabling joint reasoning over spatial inconsistencies, temporal artifacts, and spectral distortions. SSTGNN incorporates learnable spectral filters and temporal differential modeling into a graph-based architecture, capturing subtle manipulation traces more effectively. Extensive experiments on diverse benchmark datasets demonstrate that SSTGNN not only achieves superior performance in both in-domain and cross-domain settings, but also offers strong robustness against unseen manipulations. Remarkably, SSTGNN accomplishes these results with up to 42.4$\times$ fewer parameters than state-of-the-art models, making it highly lightweight and scalable for real-world deployment.
>
---
#### [new 011] Sculpting Margin Penalty: Intra-Task Adapter Merging and Classifier Calibration for Few-Shot Class-Incremental Learning
- **分类: cs.CV**

- **简介: 该论文研究了Few-Shot Class-Incremental Learning（FSCIL）中的数据隐私与高成本问题，提出了SMP方法通过MIAM机制整合不同阶段的罚函数并采用MPCC策略进行分类校准，有效平衡基类和新类泛化能力及决策边界模糊性。**

- **链接: [http://arxiv.org/pdf/2508.05094v1](http://arxiv.org/pdf/2508.05094v1)**

> **作者:** Liang Bai; Hong Song; Jinfu Li; Yucong Lin; Jingfan Fan; Tianyu Fu; Danni Ai; Deqiang Xiao; Jian Yang
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Real-world applications often face data privacy constraints and high acquisition costs, making the assumption of sufficient training data in incremental tasks unrealistic and leading to significant performance degradation in class-incremental learning. Forward-compatible learning, which prospectively prepares for future tasks during base task training, has emerged as a promising solution for Few-Shot Class-Incremental Learning (FSCIL). However, existing methods still struggle to balance base-class discriminability and new-class generalization. Moreover, limited access to original data during incremental tasks often results in ambiguous inter-class decision boundaries. To address these challenges, we propose SMP (Sculpting Margin Penalty), a novel FSCIL method that strategically integrates margin penalties at different stages within the parameter-efficient fine-tuning paradigm. Specifically, we introduce the Margin-aware Intra-task Adapter Merging (MIAM) mechanism for base task learning. MIAM trains two sets of low-rank adapters with distinct classification losses: one with a margin penalty to enhance base-class discriminability, and the other without margin constraints to promote generalization to future new classes. These adapters are then adaptively merged to improve forward compatibility. For incremental tasks, we propose a Margin Penalty-based Classifier Calibration (MPCC) strategy to refine decision boundaries by fine-tuning classifiers on all seen classes' embeddings with a margin penalty. Extensive experiments on CIFAR100, ImageNet-R, and CUB200 demonstrate that SMP achieves state-of-the-art performance in FSCIL while maintaining a better balance between base and new classes.
>
---
#### [new 012] Wavelet-Guided Dual-Frequency Encoding for Remote Sensing Change Detection
- **分类: cs.CV**

- **简介: 该论文属于图像处理与变化检测任务，旨在解决传统空间域方法对细小变化感知不足的问题。研究提出Wavelet-Guided Dual-Frequency Encoding（WGDF）方法，通过DWT分解、DFFE增强边缘细节、FDID处理细小变化、Transformer捕捉全局关系及PCDM细化结构，实现高精度、鲁棒性的遥感变化检测。**

- **链接: [http://arxiv.org/pdf/2508.05271v1](http://arxiv.org/pdf/2508.05271v1)**

> **作者:** Xiaoyang Zhang; Guodong Fan; Guang-Yong Chen; Zhen Hua; Jinjiang Li; Min Gan; C. L. Philip Chen
>
> **备注:** Submitted to TAES
>
> **摘要:** Change detection in remote sensing imagery plays a vital role in various engineering applications, such as natural disaster monitoring, urban expansion tracking, and infrastructure management. Despite the remarkable progress of deep learning in recent years, most existing methods still rely on spatial-domain modeling, where the limited diversity of feature representations hinders the detection of subtle change regions. We observe that frequency-domain feature modeling particularly in the wavelet domain an amplify fine-grained differences in frequency components, enhancing the perception of edge changes that are challenging to capture in the spatial domain. Thus, we propose a method called Wavelet-Guided Dual-Frequency Encoding (WGDF). Specifically, we first apply Discrete Wavelet Transform (DWT) to decompose the input images into high-frequency and low-frequency components, which are used to model local details and global structures, respectively. In the high-frequency branch, we design a Dual-Frequency Feature Enhancement (DFFE) module to strengthen edge detail representation and introduce a Frequency-Domain Interactive Difference (FDID) module to enhance the modeling of fine-grained changes. In the low-frequency branch, we exploit Transformers to capture global semantic relationships and employ a Progressive Contextual Difference Module (PCDM) to progressively refine change regions, enabling precise structural semantic characterization. Finally, the high- and low-frequency features are synergistically fused to unify local sensitivity with global discriminability. Extensive experiments on multiple remote sensing datasets demonstrate that WGDF significantly alleviates edge ambiguity and achieves superior detection accuracy and robustness compared to state-of-the-art methods. The code will be available at https://github.com/boshizhang123/WGDF.
>
---
#### [new 013] VFlowOpt: A Token Pruning Framework for LMMs with Visual Information Flow-Guided Optimization
- **分类: cs.CV**

- **简介: 该论文属于大型多模态模型（LMM）的token剪枝研究，旨在解决视觉-语言任务中因token冗余导致的计算成本高问题。VFlowOpt通过引入视觉信息引导的优化策略，结合重要性映射与递归剪枝技术，实现了90%的视觉token剪枝并提升了KV-Cache内存和推理效率。**

- **链接: [http://arxiv.org/pdf/2508.05211v1](http://arxiv.org/pdf/2508.05211v1)**

> **作者:** Sihan Yang; Runsen Xu; Chenhang Cui; Tai Wang; Dahua Lin; Jiangmiao Pang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Large Multimodal Models (LMMs) excel in visual-language tasks by leveraging numerous visual tokens for fine-grained visual information, but this token redundancy results in significant computational costs. Previous research aimed at reducing visual tokens during inference typically leverages importance maps derived from attention scores among vision-only tokens or vision-language tokens to prune tokens across one or multiple pruning stages. Despite this progress, pruning frameworks and strategies remain simplistic and insufficiently explored, often resulting in substantial performance degradation. In this paper, we propose VFlowOpt, a token pruning framework that introduces an importance map derivation process and a progressive pruning module with a recycling mechanism. The hyperparameters of its pruning strategy are further optimized by a visual information flow-guided method. Specifically, we compute an importance map for image tokens based on their attention-derived context relevance and patch-level information entropy. We then decide which tokens to retain or prune and aggregate the pruned ones as recycled tokens to avoid potential information loss. Finally, we apply a visual information flow-guided method that regards the last token in the LMM as the most representative signal of text-visual interactions. This method minimizes the discrepancy between token representations in LMMs with and without pruning, thereby enabling superior pruning strategies tailored to different LMMs. Experiments demonstrate that VFlowOpt can prune 90% of visual tokens while maintaining comparable performance, leading to an 89% reduction in KV-Cache memory and 3.8 times faster inference.
>
---
#### [new 014] SGDFuse: SAM-Guided Diffusion for High-Fidelity Infrared and Visible Image Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SGDFuse，旨在解决红外/可见光图像融合中因缺乏深层语义理解导致的关键目标丢失及细节损失的问题，通过利用SAM生成的语义掩码作为条件，结合条件扩散模型实现高精度、语义感知的融合。**

- **链接: [http://arxiv.org/pdf/2508.05264v1](http://arxiv.org/pdf/2508.05264v1)**

> **作者:** Xiaoyang Zhang; Zhen Hua; Yakun Ju; Wei Zhou; Jun Liu; Alex C. Kot
>
> **备注:** Submitted to TCSVT
>
> **摘要:** Infrared and visible image fusion (IVIF) aims to combine the thermal radiation information from infrared images with the rich texture details from visible images to enhance perceptual capabilities for downstream visual tasks. However, existing methods often fail to preserve key targets due to a lack of deep semantic understanding of the scene, while the fusion process itself can also introduce artifacts and detail loss, severely compromising both image quality and task performance. To address these issues, this paper proposes SGDFuse, a conditional diffusion model guided by the Segment Anything Model (SAM), to achieve high-fidelity and semantically-aware image fusion. The core of our method is to utilize high-quality semantic masks generated by SAM as explicit priors to guide the optimization of the fusion process via a conditional diffusion model. Specifically, the framework operates in a two-stage process: it first performs a preliminary fusion of multi-modal features, and then utilizes the semantic masks from SAM jointly with the preliminary fused image as a condition to drive the diffusion model's coarse-to-fine denoising generation. This ensures the fusion process not only has explicit semantic directionality but also guarantees the high fidelity of the final result. Extensive experiments demonstrate that SGDFuse achieves state-of-the-art performance in both subjective and objective evaluations, as well as in its adaptability to downstream tasks, providing a powerful solution to the core challenges in image fusion. The code of SGDFuse is available at https://github.com/boshizhang123/SGDFuse.
>
---
#### [new 015] SPEX: A Vision-Language Model for Land Cover Extraction on Spectral Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文旨在解决传统视觉-语言模型对遥感图像地表覆盖提取的效率不足问题，构建了基于视觉-语言指令的SPIE数据集，提出SPEX作为多模态LLM，通过特征聚合、上下文压缩和多光谱预训练实现像素级精确提取，并在多光谱数据集上验证其性能与解释性。**

- **链接: [http://arxiv.org/pdf/2508.05202v1](http://arxiv.org/pdf/2508.05202v1)**

> **作者:** Dongchen Si; Di Wang; Erzhong Gao; Xiaolei Qin; Liu Zhao; Jing Zhang; Minqiang Xu; Jianbo Zhan; Jianshe Wang; Lin Liu; Bo Du; Liangpei Zhang
>
> **摘要:** Spectral information has long been recognized as a critical cue in remote sensing observations. Although numerous vision-language models have been developed for pixel-level interpretation, spectral information remains underutilized, resulting in suboptimal performance, particularly in multispectral scenarios. To address this limitation, we construct a vision-language instruction-following dataset named SPIE, which encodes spectral priors of land-cover objects into textual attributes recognizable by large language models (LLMs), based on classical spectral index computations. Leveraging this dataset, we propose SPEX, a multimodal LLM designed for instruction-driven land cover extraction. To this end, we introduce several carefully designed components and training strategies, including multiscale feature aggregation, token context condensation, and multispectral visual pre-training, to achieve precise and flexible pixel-level interpretation. To the best of our knowledge, SPEX is the first multimodal vision-language model dedicated to land cover extraction in spectral remote sensing imagery. Extensive experiments on five public multispectral datasets demonstrate that SPEX consistently outperforms existing state-of-the-art methods in extracting typical land cover categories such as vegetation, buildings, and water bodies. Moreover, SPEX is capable of generating textual explanations for its predictions, thereby enhancing interpretability and user-friendliness. Code will be released at: https://github.com/MiliLab/SPEX.
>
---
#### [new 016] Revealing Latent Information: A Physics-inspired Self-supervised Pre-training Framework for Noisy and Sparse Events
- **分类: cs.CV**

- **简介: 该论文提出一种物理启发的自监督预训练框架，针对事件数据（稀疏/噪声）的特征提取问题，通过差分掩码建模、特征对齐与对比学习优化，有效恢复边缘和纹理信息，显著提升在视觉任务中的性能。**

- **链接: [http://arxiv.org/pdf/2508.05507v1](http://arxiv.org/pdf/2508.05507v1)**

> **作者:** Lin Zhu; Ruonan Liu; Xiao Wang; Lizhi Wang; Hua Huang
>
> **摘要:** Event camera, a novel neuromorphic vision sensor, records data with high temporal resolution and wide dynamic range, offering new possibilities for accurate visual representation in challenging scenarios. However, event data is inherently sparse and noisy, mainly reflecting brightness changes, which complicates effective feature extraction. To address this, we propose a self-supervised pre-training framework to fully reveal latent information in event data, including edge information and texture cues. Our framework consists of three stages: Difference-guided Masked Modeling, inspired by the event physical sampling process, reconstructs temporal intensity difference maps to extract enhanced information from raw event data. Backbone-fixed Feature Transition contrasts event and image features without updating the backbone to preserve representations learned from masked modeling and stabilizing their effect on contrastive learning. Focus-aimed Contrastive Learning updates the entire model to improve semantic discrimination by focusing on high-value regions. Extensive experiments show our framework is robust and consistently outperforms state-of-the-art methods on various downstream tasks, including object recognition, semantic segmentation, and optical flow estimation. The code and dataset are available at https://github.com/BIT-Vision/EventPretrain.
>
---
#### [new 017] CF3: Compact and Fast 3D Feature Fields
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在解决传统3DGS在计算成本高的问题，提出CF3通过多视图融合与自编码器优化，将GPU密集计算降至5%以下，实现紧凑高效的3D特征场生成。**

- **链接: [http://arxiv.org/pdf/2508.05254v1](http://arxiv.org/pdf/2508.05254v1)**

> **作者:** Hyunjoon Lee; Joonkyu Min; Jaesik Park
>
> **备注:** ICCV 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has begun incorporating rich information from 2D foundation models. However, most approaches rely on a bottom-up optimization process that treats raw 2D features as ground truth, incurring increased computational costs. We propose a top-down pipeline for constructing compact and fast 3D Gaussian feature fields, namely, CF3. We first perform a fast weighted fusion of multi-view 2D features with pre-trained Gaussians. This approach enables training a per-Gaussian autoencoder directly on the lifted features, instead of training autoencoders in the 2D domain. As a result, the autoencoder better aligns with the feature distribution. More importantly, we introduce an adaptive sparsification method that optimizes the Gaussian attributes of the feature field while pruning and merging the redundant Gaussians, constructing an efficient representation with preserved geometric details. Our approach achieves a competitive 3D feature field using as little as 5% of the Gaussians compared to Feature-3DGS.
>
---
#### [new 018] EndoMatcher: Generalizable Endoscopic Image Matcher via Multi-Domain Pre-training for Robot-Assisted Surgery
- **分类: cs.CV**

- **简介: 该论文旨在解决端上手术中通用性端到端图像匹配问题，通过多领域预训练（EndoMatcher）技术提升3D重建、导航等任务的精度。其创新点在于利用Vision Transformer与双交互块优化特征匹配，构建了包含1.2M张图的Endo-Mix6数据集，并采用多目标训练策略增强跨域泛化能力，显著提升了零样本匹配性能。**

- **链接: [http://arxiv.org/pdf/2508.05205v1](http://arxiv.org/pdf/2508.05205v1)**

> **作者:** Bingyu Yang; Qingyao Tian; Yimeng Geng; Huai Liao; Xinyan Huang; Jiebo Luo; Hongbin Liu
>
> **摘要:** Generalizable dense feature matching in endoscopic images is crucial for robot-assisted tasks, including 3D reconstruction, navigation, and surgical scene understanding. Yet, it remains a challenge due to difficult visual conditions (e.g., weak textures, large viewpoint variations) and a scarcity of annotated data. To address these challenges, we propose EndoMatcher, a generalizable endoscopic image matcher via large-scale, multi-domain data pre-training. To address difficult visual conditions, EndoMatcher employs a two-branch Vision Transformer to extract multi-scale features, enhanced by dual interaction blocks for robust correspondence learning. To overcome data scarcity and improve domain diversity, we construct Endo-Mix6, the first multi-domain dataset for endoscopic matching. Endo-Mix6 consists of approximately 1.2M real and synthetic image pairs across six domains, with correspondence labels generated using Structure-from-Motion and simulated transformations. The diversity and scale of Endo-Mix6 introduce new challenges in training stability due to significant variations in dataset sizes, distribution shifts, and error imbalance. To address them, a progressive multi-objective training strategy is employed to promote balanced learning and improve representation quality across domains. This enables EndoMatcher to generalize across unseen organs and imaging conditions in a zero-shot fashion. Extensive zero-shot matching experiments demonstrate that EndoMatcher increases the number of inlier matches by 140.69% and 201.43% on the Hamlyn and Bladder datasets over state-of-the-art methods, respectively, and improves the Matching Direction Prediction Accuracy (MDPA) by 9.40% on the Gastro-Matching dataset, achieving dense and accurate matching under challenging endoscopic conditions. The code is publicly available at https://github.com/Beryl2000/EndoMatcher.
>
---
#### [new 019] F2PASeg: Feature Fusion for Pituitary Anatomy Segmentation in Endoscopic Surgery
- **分类: cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 该论文提出F2PASeg方法，解决端上手术中垂体结构分割任务，通过特征融合结合数据增强和语义嵌入提升鲁棒性，解决了因图像不一致性导致的分割精度问题，并验证了其在实时手术中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05465v1](http://arxiv.org/pdf/2508.05465v1)**

> **作者:** Lumin Chen; Zhiying Wu; Tianye Lei; Xuexue Bai; Ming Feng; Yuxi Wang; Gaofeng Meng; Zhen Lei; Hongbin Liu
>
> **摘要:** Pituitary tumors often cause deformation or encapsulation of adjacent vital structures. Anatomical structure segmentation can provide surgeons with early warnings of regions that pose surgical risks, thereby enhancing the safety of pituitary surgery. However, pixel-level annotated video stream datasets for pituitary surgeries are extremely rare. To address this challenge, we introduce a new dataset for Pituitary Anatomy Segmentation (PAS). PAS comprises 7,845 time-coherent images extracted from 120 videos. To mitigate class imbalance, we apply data augmentation techniques that simulate the presence of surgical instruments in the training data. One major challenge in pituitary anatomy segmentation is the inconsistency in feature representation due to occlusions, camera motion, and surgical bleeding. By incorporating a Feature Fusion module, F2PASeg is proposed to refine anatomical structure segmentation by leveraging both high-resolution image features and deep semantic embeddings, enhancing robustness against intraoperative variations. Experimental results demonstrate that F2PASeg consistently segments critical anatomical structures in real time, providing a reliable solution for intraoperative pituitary surgery planning. Code: https://github.com/paulili08/F2PASeg.
>
---
#### [new 020] RegionMed-CLIP: A Region-Aware Multimodal Contrastive Learning Pre-trained Model for Medical Image Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RegionMed-CLIP，解决医学图像理解中局部与细粒度信息缺失的问题，通过区域感知ROI处理器和分阶段训练，构建MedRegion-500k数据集，显著优于现有视觉语言模型，推动区域-aware多模态预训练研究。**

- **链接: [http://arxiv.org/pdf/2508.05244v1](http://arxiv.org/pdf/2508.05244v1)**

> **作者:** Tianchen Fang; Guiru Liu
>
> **摘要:** Medical image understanding plays a crucial role in enabling automated diagnosis and data-driven clinical decision support. However, its progress is impeded by two primary challenges: the limited availability of high-quality annotated medical data and an overreliance on global image features, which often miss subtle but clinically significant pathological regions. To address these issues, we introduce RegionMed-CLIP, a region-aware multimodal contrastive learning framework that explicitly incorporates localized pathological signals along with holistic semantic representations. The core of our method is an innovative region-of-interest (ROI) processor that adaptively integrates fine-grained regional features with the global context, supported by a progressive training strategy that enhances hierarchical multimodal alignment. To enable large-scale region-level representation learning, we construct MedRegion-500k, a comprehensive medical image-text corpus that features extensive regional annotations and multilevel clinical descriptions. Extensive experiments on image-text retrieval, zero-shot classification, and visual question answering tasks demonstrate that RegionMed-CLIP consistently exceeds state-of-the-art vision language models by a wide margin. Our results highlight the critical importance of region-aware contrastive pre-training and position RegionMed-CLIP as a robust foundation for advancing multimodal medical image understanding.
>
---
#### [new 021] LLaVA-RE: Binary Image-Text Relevancy Evaluation with Multimodal Large Language Model
- **分类: cs.CV**

- **简介: 该论文旨在解决图像-文本二元相关性评估问题，基于多模态大型语言模型构建有效评价框架。研究提出LLaVA-RE，采用LLaVA架构结合任务指令和多模态数据集，并创新性地设计了覆盖多种任务的二元相关性数据集，验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.05602v1](http://arxiv.org/pdf/2508.05602v1)**

> **作者:** Tao Sun; Oliver Liu; JinJin Li; Lan Ma
>
> **备注:** Published in the First Workshop of Evaluation of Multi-Modal Generation 2025
>
> **摘要:** Multimodal generative AI usually involves generating image or text responses given inputs in another modality. The evaluation of image-text relevancy is essential for measuring response quality or ranking candidate responses. In particular, binary relevancy evaluation, i.e., ``Relevant'' vs. ``Not Relevant'', is a fundamental problem. However, this is a challenging task considering that texts have diverse formats and the definition of relevancy varies in different scenarios. We find that Multimodal Large Language Models (MLLMs) are an ideal choice to build such evaluators, as they can flexibly handle complex text formats and take in additional task information. In this paper, we present LLaVA-RE, a first attempt for binary image-text relevancy evaluation with MLLM. It follows the LLaVA architecture and adopts detailed task instructions and multimodal in-context samples. In addition, we propose a novel binary relevancy data set that covers various tasks. Experimental results validate the effectiveness of our framework.
>
---
#### [new 022] FLUX-Makeup: High-Fidelity, Identity-Consistent, and Robust Makeup Transfer via Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文提出FLUX-Makeup，解决基于GAN的高保真妆面迁移与身份一致性问题，通过消除辅助模块实现高效鲁棒性提升，利用RefLoRA Injector分离特征提取路径并构建数据生成框架，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.05069v1](http://arxiv.org/pdf/2508.05069v1)**

> **作者:** Jian Zhu; Shanyuan Liu; Liuzhuozheng Li; Yue Gong; He Wang; Bo Cheng; Yuhang Ma; Liebucha Wu; Xiaoyu Wu; Dawei Leng; Yuhui Yin; Yang Xu
>
> **摘要:** Makeup transfer aims to apply the makeup style from a reference face to a target face and has been increasingly adopted in practical applications. Existing GAN-based approaches typically rely on carefully designed loss functions to balance transfer quality and facial identity consistency, while diffusion-based methods often depend on additional face-control modules or algorithms to preserve identity. However, these auxiliary components tend to introduce extra errors, leading to suboptimal transfer results. To overcome these limitations, we propose FLUX-Makeup, a high-fidelity, identity-consistent, and robust makeup transfer framework that eliminates the need for any auxiliary face-control components. Instead, our method directly leverages source-reference image pairs to achieve superior transfer performance. Specifically, we build our framework upon FLUX-Kontext, using the source image as its native conditional input. Furthermore, we introduce RefLoRAInjector, a lightweight makeup feature injector that decouples the reference pathway from the backbone, enabling efficient and comprehensive extraction of makeup-related information. In parallel, we design a robust and scalable data generation pipeline to provide more accurate supervision during training. The paired makeup datasets produced by this pipeline significantly surpass the quality of all existing datasets. Extensive experiments demonstrate that FLUX-Makeup achieves state-of-the-art performance, exhibiting strong robustness across diverse scenarios.
>
---
#### [new 023] FedGIN: Federated Learning with Dynamic Global Intensity Non-linear Augmentation for Organ Segmentation using Multi-modal Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出了一种融合联邦学习（FL）与动态全局强度非线性增强模块的新框架FedGIN，用于多模态器官分割任务，解决跨模态通用性不足及隐私保护问题，通过集成GIN模块实现轻量级数据协作训练。**

- **链接: [http://arxiv.org/pdf/2508.05137v1](http://arxiv.org/pdf/2508.05137v1)**

> **作者:** Sachin Dudda Nagaraju; Ashkan Moradi; Bendik Skarre Abrahamsen; Mattijs Elschot
>
> **备注:** Paper Accepted at MICCAI 2025 DeCaf Workshop Track
>
> **摘要:** Medical image segmentation plays a crucial role in AI-assisted diagnostics, surgical planning, and treatment monitoring. Accurate and robust segmentation models are essential for enabling reliable, data-driven clinical decision making across diverse imaging modalities. Given the inherent variability in image characteristics across modalities, developing a unified model capable of generalizing effectively to multiple modalities would be highly beneficial. This model could streamline clinical workflows and reduce the need for modality-specific training. However, real-world deployment faces major challenges, including data scarcity, domain shift between modalities (e.g., CT vs. MRI), and privacy restrictions that prevent data sharing. To address these issues, we propose FedGIN, a Federated Learning (FL) framework that enables multimodal organ segmentation without sharing raw patient data. Our method integrates a lightweight Global Intensity Non-linear (GIN) augmentation module that harmonizes modality-specific intensity distributions during local training. We evaluated FedGIN using two types of datasets: an imputed dataset and a complete dataset. In the limited dataset scenario, the model was initially trained using only MRI data, and CT data was added to assess its performance improvements. In the complete dataset scenario, both MRI and CT data were fully utilized for training on all clients. In the limited-data scenario, FedGIN achieved a 12 to 18% improvement in 3D Dice scores on MRI test cases compared to FL without GIN and consistently outperformed local baselines. In the complete dataset scenario, FedGIN demonstrated near-centralized performance, with a 30% Dice score improvement over the MRI-only baseline and a 10% improvement over the CT-only baseline, highlighting its strong cross-modality generalization under privacy constraints.
>
---
#### [new 024] Navigating the Trade-off: A Synthesis of Defensive Strategies for Zero-Shot Adversarial Robustness in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究了零样本对抗鲁棒性与模型泛化能力之间的平衡，提出通过Adversarial Fine-Tuning（AFT）和训练自由防御策略优化VLMs，结合历史方法演变（从输入启发式到隐空间净化），解决了如何同时增强防御并维持性能的问题。**

- **链接: [http://arxiv.org/pdf/2508.05237v1](http://arxiv.org/pdf/2508.05237v1)**

> **作者:** Zane Xu; Jason Sun
>
> **摘要:** This report synthesizes eight seminal papers on the zero-shot adversarial robustness of vision-language models (VLMs) like CLIP. A central challenge in this domain is the inherent trade-off between enhancing adversarial robustness and preserving the model's zero-shot generalization capabilities. We analyze two primary defense paradigms: Adversarial Fine-Tuning (AFT), which modifies model parameters, and Training-Free/Test-Time Defenses, which preserve them. We trace the evolution from alignment-preserving methods (TeCoA) to embedding space re-engineering (LAAT, TIMA), and from input heuristics (AOM, TTC) to latent-space purification (CLIPure). Finally, we identify key challenges and future directions including hybrid defense strategies and adversarial pre-training.
>
---
#### [new 025] Extending Foundational Monocular Depth Estimators to Fisheye Cameras with Calibration Tokens
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出了一种自监督方法，用于将基础单目深度估计器（FMDE）扩展到焦外相机，解决了由于焦距变化带来的深度估计误差问题。通过引入光标令牌（Calibration Tokens）进行轻量级调整并利用FMDE的已表达隐空间，有效避免了传统方法中出现的失真和计算成本，实现了无需重新训练即可在 fisheye 相机上进行深度估计。**

- **链接: [http://arxiv.org/pdf/2508.04928v1](http://arxiv.org/pdf/2508.04928v1)**

> **作者:** Suchisrit Gangopadhyay; Jung-Hee Kim; Xien Chen; Patrick Rim; Hyoungseob Park; Alex Wong
>
> **摘要:** We propose a method to extend foundational monocular depth estimators (FMDEs), trained on perspective images, to fisheye images. Despite being trained on tens of millions of images, FMDEs are susceptible to the covariate shift introduced by changes in camera calibration (intrinsic, distortion) parameters, leading to erroneous depth estimates. Our method aligns the distribution of latent embeddings encoding fisheye images to those of perspective images, enabling the reuse of FMDEs for fisheye cameras without retraining or finetuning. To this end, we introduce a set of Calibration Tokens as a light-weight adaptation mechanism that modulates the latent embeddings for alignment. By exploiting the already expressive latent space of FMDEs, we posit that modulating their embeddings avoids the negative impact of artifacts and loss introduced in conventional recalibration or map projection to a canonical reference frame in the image space. Our method is self-supervised and does not require fisheye images but leverages publicly available large-scale perspective image datasets. This is done by recalibrating perspective images to fisheye images, and enforcing consistency between their estimates during training. We evaluate our approach with several FMDEs, on both indoors and outdoors, where we consistently improve over state-of-the-art methods using a single set of tokens for both. Code available at: https://github.com/JungHeeKim29/calibration-token.
>
---
#### [new 026] LuKAN: A Kolmogorov-Arnold Network Framework for 3D Human Motion Prediction
- **分类: cs.CV**

- **简介: 该论文旨在解决3D人体运动预测中的精度与效率平衡问题，提出LuKAN框架，利用KAN网络和Lucas多项式激活函数进行时空特征建模，并通过离散波形变换与逆变换实现高效预测，验证其在基准数据集上的优越性。**

- **链接: [http://arxiv.org/pdf/2508.04847v1](http://arxiv.org/pdf/2508.04847v1)**

> **作者:** Md Zahidul Hasan; A. Ben Hamza; Nizar Bouguila
>
> **摘要:** The goal of 3D human motion prediction is to forecast future 3D poses of the human body based on historical motion data. Existing methods often face limitations in achieving a balance between prediction accuracy and computational efficiency. In this paper, we present LuKAN, an effective model based on Kolmogorov-Arnold Networks (KANs) with Lucas polynomial activations. Our model first applies the discrete wavelet transform to encode temporal information in the input motion sequence. Then, a spatial projection layer is used to capture inter-joint dependencies, ensuring structural consistency of the human body. At the core of LuKAN is the Temporal Dependency Learner, which employs a KAN layer parameterized by Lucas polynomials for efficient function approximation. These polynomials provide computational efficiency and an enhanced capability to handle oscillatory behaviors. Finally, the inverse discrete wavelet transform reconstructs motion sequences in the time domain, generating temporally coherent predictions. Extensive experiments on three benchmark datasets demonstrate the competitive performance of our model compared to strong baselines, as evidenced by both quantitative and qualitative evaluations. Moreover, its compact architecture coupled with the linear recurrence of Lucas polynomials, ensures computational efficiency.
>
---
#### [new 027] ArbiViewGen: Controllable Arbitrary Viewpoint Camera Data Generation for Autonomous Driving via Stable Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决无地面真实数据导致的高精度模型训练问题，通过引入Feature-Aware Adaptive View Stitching（FAVS）和Cross-View Consistency Self-Supervised Learning（CVC-SSL）框架，利用多摄像头数据生成任意视角相机图像，实现自主驾驶场景下的可控图像生成。**

- **链接: [http://arxiv.org/pdf/2508.05236v1](http://arxiv.org/pdf/2508.05236v1)**

> **作者:** Yatong Lan; Jingfeng Chen; Yiru Wang; Lei He
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Arbitrary viewpoint image generation holds significant potential for autonomous driving, yet remains a challenging task due to the lack of ground-truth data for extrapolated views, which hampers the training of high-fidelity generative models. In this work, we propose Arbiviewgen, a novel diffusion-based framework for the generation of controllable camera images from arbitrary points of view. To address the absence of ground-truth data in unseen views, we introduce two key components: Feature-Aware Adaptive View Stitching (FAVS) and Cross-View Consistency Self-Supervised Learning (CVC-SSL). FAVS employs a hierarchical matching strategy that first establishes coarse geometric correspondences using camera poses, then performs fine-grained alignment through improved feature matching algorithms, and identifies high-confidence matching regions via clustering analysis. Building upon this, CVC-SSL adopts a self-supervised training paradigm where the model reconstructs the original camera views from the synthesized stitched images using a diffusion model, enforcing cross-view consistency without requiring supervision from extrapolated data. Our framework requires only multi-camera images and their associated poses for training, eliminating the need for additional sensors or depth maps. To our knowledge, Arbiviewgen is the first method capable of controllable arbitrary view camera image generation in multiple vehicle configurations.
>
---
#### [new 028] Textual and Visual Guided Task Adaptation for Source-Free Cross-Domain Few-Shot Segmentation
- **分类: cs.CV; I.2.10**

- **简介: 该论文提出了一种基于文本与视觉引导的跨域Few-Shot分割方法，旨在解决源域泛化能力不足的问题。通过引入Task-Specific Attention Adapters（TSAA）和视觉-视觉/文本-视觉嵌入模块，结合多模态特征融合，实现了在无源域数据支持下的目标域任务适配，提升了4.11%的分割精度。**

- **链接: [http://arxiv.org/pdf/2508.05213v1](http://arxiv.org/pdf/2508.05213v1)**

> **作者:** Jianming Liu; Wenlong Qiu; Haitao Wei
>
> **备注:** 10 pages,Accepted at ACMMM2025
>
> **摘要:** Few-Shot Segmentation(FSS) aims to efficient segmentation of new objects with few labeled samples. However, its performance significantly degrades when domain discrepancies exist between training and deployment. Cross-Domain Few-Shot Segmentation(CD-FSS) is proposed to mitigate such performance degradation. Current CD-FSS methods primarily sought to develop segmentation models on a source domain capable of cross-domain generalization. However, driven by escalating concerns over data privacy and the imperative to minimize data transfer and training expenses, the development of source-free CD-FSS approaches has become essential. In this work, we propose a source-free CD-FSS method that leverages both textual and visual information to facilitate target domain task adaptation without requiring source domain data. Specifically, we first append Task-Specific Attention Adapters (TSAA) to the feature pyramid of a pretrained backbone, which adapt multi-level features extracted from the shared pre-trained backbone to the target task. Then, the parameters of the TSAA are trained through a Visual-Visual Embedding Alignment (VVEA) module and a Text-Visual Embedding Alignment (TVEA) module. The VVEA module utilizes global-local visual features to align image features across different views, while the TVEA module leverages textual priors from pre-aligned multi-modal features (e.g., from CLIP) to guide cross-modal adaptation. By combining the outputs of these modules through dense comparison operations and subsequent fusion via skip connections, our method produces refined prediction masks. Under both 1-shot and 5-shot settings, the proposed approach achieves average segmentation accuracy improvements of 2.18\% and 4.11\%, respectively, across four cross-domain datasets, significantly outperforming state-of-the-art CD-FSS methods. Code are available at https://github.com/ljm198134/TVGTANet.
>
---
#### [new 029] 3DGabSplat: 3D Gabor Splatting for Frequency-adaptive Radiance Field Rendering
- **分类: cs.CV**

- **简介: 该论文提出3D Gabor Splatting任务，解决传统3DGS因低频泛化性导致的高精度视图合成和效率问题，通过引入多方向3D频率响应滤波器和高效CUDA渲染机制，实现了频率自适应优化与内存节约，显著提升性能表现。**

- **链接: [http://arxiv.org/pdf/2508.05343v1](http://arxiv.org/pdf/2508.05343v1)**

> **作者:** Junyu Zhou; Yuyang Huang; Wenrui Dai; Junni Zou; Ziyang Zheng; Nuowen Kan; Chenglin Li; Hongkai Xiong
>
> **备注:** Accepted by ACM MM'25
>
> **摘要:** Recent prominence in 3D Gaussian Splatting (3DGS) has enabled real-time rendering while maintaining high-fidelity novel view synthesis. However, 3DGS resorts to the Gaussian function that is low-pass by nature and is restricted in representing high-frequency details in 3D scenes. Moreover, it causes redundant primitives with degraded training and rendering efficiency and excessive memory overhead. To overcome these limitations, we propose 3D Gabor Splatting (3DGabSplat) that leverages a novel 3D Gabor-based primitive with multiple directional 3D frequency responses for radiance field representation supervised by multi-view images. The proposed 3D Gabor-based primitive forms a filter bank incorporating multiple 3D Gabor kernels at different frequencies to enhance flexibility and efficiency in capturing fine 3D details. Furthermore, to achieve novel view rendering, an efficient CUDA-based rasterizer is developed to project the multiple directional 3D frequency components characterized by 3D Gabor-based primitives onto the 2D image plane, and a frequency-adaptive mechanism is presented for adaptive joint optimization of primitives. 3DGabSplat is scalable to be a plug-and-play kernel for seamless integration into existing 3DGS paradigms to enhance both efficiency and quality of novel view synthesis. Extensive experiments demonstrate that 3DGabSplat outperforms 3DGS and its variants using alternative primitives, and achieves state-of-the-art rendering quality across both real-world and synthetic scenes. Remarkably, we achieve up to 1.35 dB PSNR gain over 3DGS with simultaneously reduced number of primitives and memory consumption.
>
---
#### [new 030] Textual Inversion for Efficient Adaptation of Open-Vocabulary Object Detectors Without Forgetting
- **分类: cs.CV**

- **简介: 该论文提出了一种文本逆向方法，旨在通过扩展开放词汇目标检测器的词汇库，实现无需遗忘的高效适配，同时保留原模型的基准性能和零样本迁移能力，有效解决了传统方法因丢失原有能力而导致的性能下降问题。**

- **链接: [http://arxiv.org/pdf/2508.05323v1](http://arxiv.org/pdf/2508.05323v1)**

> **作者:** Frank Ruis; Gertjan Burghouts; Hugo Kuijf
>
> **摘要:** Recent progress in large pre-trained vision language models (VLMs) has reached state-of-the-art performance on several object detection benchmarks and boasts strong zero-shot capabilities, but for optimal performance on specific targets some form of finetuning is still necessary. While the initial VLM weights allow for great few-shot transfer learning, this usually involves the loss of the original natural language querying and zero-shot capabilities. Inspired by the success of Textual Inversion (TI) in personalizing text-to-image diffusion models, we propose a similar formulation for open-vocabulary object detection. TI allows extending the VLM vocabulary by learning new or improving existing tokens to accurately detect novel or fine-grained objects from as little as three examples. The learned tokens are completely compatible with the original VLM weights while keeping them frozen, retaining the original model's benchmark performance, and leveraging its existing capabilities such as zero-shot domain transfer (e.g., detecting a sketch of an object after training only on real photos). The storage and gradient calculations are limited to the token embedding dimension, requiring significantly less compute than full-model fine-tuning. We evaluated whether the method matches or outperforms the baseline methods that suffer from forgetting in a wide variety of quantitative and qualitative experiments.
>
---
#### [new 031] Latent Expression Generation for Referring Image Segmentation and Grounding
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究了视觉引导任务（如RIS/REC），旨在通过多模态生成潜意识表达捕捉丰富视觉信息并解决单一文本描述带来的信息稀疏性问题。提出通过模块化设计和对比学习策略，整合互补视觉特征提升图像分割与地平线任务性能。**

- **链接: [http://arxiv.org/pdf/2508.05123v1](http://arxiv.org/pdf/2508.05123v1)**

> **作者:** Seonghoon Yu; Joonbeom Hong; Joonseok Lee; Jeany Son
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Visual grounding tasks, such as referring image segmentation (RIS) and referring expression comprehension (REC), aim to localize a target object based on a given textual description. The target object in an image can be described in multiple ways, reflecting diverse attributes such as color, position, and more. However, most existing methods rely on a single textual input, which captures only a fraction of the rich information available in the visual domain. This mismatch between rich visual details and sparse textual cues can lead to the misidentification of similar objects. To address this, we propose a novel visual grounding framework that leverages multiple latent expressions generated from a single textual input by incorporating complementary visual details absent from the original description. Specifically, we introduce subject distributor and visual concept injector modules to embed both shared-subject and distinct-attributes concepts into the latent representations, thereby capturing unique and target-specific visual cues. We also propose a positive-margin contrastive learning strategy to align all latent expressions with the original text while preserving subtle variations. Experimental results show that our method not only outperforms state-of-the-art RIS and REC approaches on multiple benchmarks but also achieves outstanding performance on the generalized referring expression segmentation (GRES) benchmark.
>
---
#### [new 032] Optimal Brain Connection: Towards Efficient Structural Pruning
- **分类: cs.CV**

- **简介: 该论文旨在改进结构剪枝方法，解决传统方法仅关注参数独立性的问题。提出Jacobian准则评估参数交互与层次依赖，并设计Equivalent Pruning机制保留所有连接贡献，通过实验验证其性能优势。**

- **链接: [http://arxiv.org/pdf/2508.05521v1](http://arxiv.org/pdf/2508.05521v1)**

> **作者:** Shaowu Chen; Wei Ma; Binhua Huang; Qingyuan Wang; Guoxin Wang; Weize Sun; Lei Huang; Deepu John
>
> **摘要:** Structural pruning has been widely studied for its effectiveness in compressing neural networks. However, existing methods often neglect the interconnections among parameters. To address this limitation, this paper proposes a structural pruning framework termed Optimal Brain Connection. First, we introduce the Jacobian Criterion, a first-order metric for evaluating the saliency of structural parameters. Unlike existing first-order methods that assess parameters in isolation, our criterion explicitly captures both intra-component interactions and inter-layer dependencies. Second, we propose the Equivalent Pruning mechanism, which utilizes autoencoders to retain the contributions of all original connection--including pruned ones--during fine-tuning. Experimental results demonstrate that the Jacobian Criterion outperforms several popular metrics in preserving model performance, while the Equivalent Pruning mechanism effectively mitigates performance degradation after fine-tuning. Code: https://github.com/ShaowuChen/Optimal_Brain_Connection
>
---
#### [new 033] Follow-Your-Instruction: A Comprehensive MLLM Agent for World Data Synthesis
- **分类: cs.CV**

- **简介: 该论文提出了一种名为"Follow-Your-Instruction"的多模态大语言模型驱动框架，用于自动合成2D/3D/4D数据，解决大规模真实数据采集成本高、场景构建受限的问题。通过MLLM-Collector收集资产、MLLM-Generator与Optimizer进行语义优化和规划，最终生成可扩展的生成式数据，验证其在现有模型中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05580v1](http://arxiv.org/pdf/2508.05580v1)**

> **作者:** Kunyu Feng; Yue Ma; Xinhua Zhang; Boshi Liu; Yikuang Yuluo; Yinhan Zhang; Runtao Liu; Hongyu Liu; Zhiyuan Qin; Shanhui Mo; Qifeng Chen; Zeyu Wang
>
> **摘要:** With the growing demands of AI-generated content (AIGC), the need for high-quality, diverse, and scalable data has become increasingly crucial. However, collecting large-scale real-world data remains costly and time-consuming, hindering the development of downstream applications. While some works attempt to collect task-specific data via a rendering process, most approaches still rely on manual scene construction, limiting their scalability and accuracy. To address these challenges, we propose Follow-Your-Instruction, a Multimodal Large Language Model (MLLM)-driven framework for automatically synthesizing high-quality 2D, 3D, and 4D data. Our \textbf{Follow-Your-Instruction} first collects assets and their associated descriptions through multimodal inputs using the MLLM-Collector. Then it constructs 3D layouts, and leverages Vision-Language Models (VLMs) for semantic refinement through multi-view scenes with the MLLM-Generator and MLLM-Optimizer, respectively. Finally, it uses MLLM-Planner to generate temporally coherent future frames. We evaluate the quality of the generated data through comprehensive experiments on the 2D, 3D, and 4D generative tasks. The results show that our synthetic data significantly boosts the performance of existing baseline models, demonstrating Follow-Your-Instruction's potential as a scalable and effective data engine for generative intelligence.
>
---
#### [new 034] Looking into the Unknown: Exploring Action Discovery for Segmentation of Known and Unknown Actions
- **分类: cs.CV**

- **简介: 该论文研究了一种解决动作分割中未知动作和不完整标注问题的方法，旨在通过增强已知动作的时序与语义粒度来标注新动作。提出两个模块：GGSM用于识别时间间隔，UASA用于分配语义类，系统验证在早餐、50 salads 和桌面组装等数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.05529v1](http://arxiv.org/pdf/2508.05529v1)**

> **作者:** Federico Spurio; Emad Bahrami; Olga Zatsarynna; Yazan Abu Farha; Gianpiero Francesca; Juergen Gall
>
> **摘要:** We introduce Action Discovery, a novel setup within Temporal Action Segmentation that addresses the challenge of defining and annotating ambiguous actions and incomplete annotations in partially labeled datasets. In this setup, only a subset of actions - referred to as known actions - is annotated in the training data, while other unknown actions remain unlabeled. This scenario is particularly relevant in domains like neuroscience, where well-defined behaviors (e.g., walking, eating) coexist with subtle or infrequent actions that are often overlooked, as well as in applications where datasets are inherently partially annotated due to ambiguous or missing labels. To address this problem, we propose a two-step approach that leverages the known annotations to guide both the temporal and semantic granularity of unknown action segments. First, we introduce the Granularity-Guided Segmentation Module (GGSM), which identifies temporal intervals for both known and unknown actions by mimicking the granularity of annotated actions. Second, we propose the Unknown Action Segment Assignment (UASA), which identifies semantically meaningful classes within the unknown actions, based on learned embedding similarities. We systematically explore the proposed setting of Action Discovery on three challenging datasets - Breakfast, 50Salads, and Desktop Assembly - demonstrating that our method considerably improves upon existing baselines.
>
---
#### [new 035] Finding Needles in Images: Can Multimodal LLMs Locate Fine Details?
- **分类: cs.CV**

- **简介: 该论文研究多模态LLMs在复杂文档细部定位任务中的能力，旨在填补现有研究中对细粒度细节提取的空白。通过构建NiM基准并提出Spot-IT方法，解决了传统LLMs在高精度任务中的局限性，展示了其在提升精确性方面的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05053v1](http://arxiv.org/pdf/2508.05053v1)**

> **作者:** Parth Thakkar; Ankush Agarwal; Prasad Kasu; Pulkit Bansal; Chaitanya Devaguptapu
>
> **备注:** Accepted at ACL 2025 in the main track
>
> **摘要:** While Multi-modal Large Language Models (MLLMs) have shown impressive capabilities in document understanding tasks, their ability to locate and reason about fine-grained details within complex documents remains understudied. Consider searching a restaurant menu for a specific nutritional detail or identifying a disclaimer in a lengthy newspaper article tasks that demand careful attention to small but significant details within a broader narrative, akin to Finding Needles in Images (NiM). To address this gap, we introduce NiM, a carefully curated benchmark spanning diverse real-world documents including newspapers, menus, and lecture images, specifically designed to evaluate MLLMs' capability in these intricate tasks. Building on this, we further propose Spot-IT, a simple yet effective approach that enhances MLLMs capability through intelligent patch selection and Gaussian attention, motivated from how humans zoom and focus when searching documents. Our extensive experiments reveal both the capabilities and limitations of current MLLMs in handling fine-grained document understanding tasks, while demonstrating the effectiveness of our approach. Spot-IT achieves significant improvements over baseline methods, particularly in scenarios requiring precise detail extraction from complex layouts.
>
---
#### [new 036] AutoIAD: Manager-Driven Multi-Agent Collaboration for Automated Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出了一种基于管理驱动的多智能体协作框架AutoIAD，用于自动化工业视觉异常检测任务，通过整合数据准备、模型训练等子代理及领域知识库，有效提升检测效率与模型性能，解决了传统人工干预不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.05503v1](http://arxiv.org/pdf/2508.05503v1)**

> **作者:** Dongwei Ji; Bingzhang Hu; Yi Zhou
>
> **摘要:** Industrial anomaly detection (IAD) is critical for manufacturing quality control, but conventionally requires significant manual effort for various application scenarios. This paper introduces AutoIAD, a multi-agent collaboration framework, specifically designed for end-to-end automated development of industrial visual anomaly detection. AutoIAD leverages a Manager-Driven central agent to orchestrate specialized sub-agents (including Data Preparation, Data Loader, Model Designer, Trainer) and integrates a domain-specific knowledge base, which intelligently handles the entire pipeline using raw industrial image data to develop a trained anomaly detection model. We construct a comprehensive benchmark using MVTec AD datasets to evaluate AutoIAD across various LLM backends. Extensive experiments demonstrate that AutoIAD significantly outperforms existing general-purpose agentic collaboration frameworks and traditional AutoML frameworks in task completion rate and model performance (AUROC), while effectively mitigating issues like hallucination through iterative refinement. Ablation studies further confirm the crucial roles of the Manager central agent and the domain knowledge base module in producing robust and high-quality IAD solutions.
>
---
#### [new 037] Symmetry Understanding of 3D Shapes via Chirality Disentanglement
- **分类: cs.CV**

- **简介: 该论文旨在通过章arity信息分离3D形状的对称性，解决形状分析中左右对称性辨识困难，开发了基于Diff3F框架的无监督特征提取方法，验证其在左-右分离、形状匹配和分割等任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05505v1](http://arxiv.org/pdf/2508.05505v1)**

> **作者:** Weikang Wang; Tobias Weißberg; Nafie El Amrani; Florian Bernard
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Chirality information (i.e. information that allows distinguishing left from right) is ubiquitous for various data modes in computer vision, including images, videos, point clouds, and meshes. While chirality has been extensively studied in the image domain, its exploration in shape analysis (such as point clouds and meshes) remains underdeveloped. Although many shape vertex descriptors have shown appealing properties (e.g. robustness to rigid-body transformations), they are often not able to disambiguate between left and right symmetric parts. Considering the ubiquity of chirality information in different shape analysis problems and the lack of chirality-aware features within current shape descriptors, developing a chirality feature extractor becomes necessary and urgent. Based on the recent Diff3F framework, we propose an unsupervised chirality feature extraction pipeline to decorate shape vertices with chirality-aware information, extracted from 2D foundation models. We evaluated the extracted chirality features through quantitative and qualitative experiments across diverse datasets. Results from downstream tasks including left-right disentanglement, shape matching, and part segmentation demonstrate their effectiveness and practical utility. Project page: https://wei-kang-wang.github.io/chirality/
>
---
#### [new 038] Hi3DEval: Advancing 3D Generation Evaluation with Hierarchical Validity
- **分类: cs.CV**

- **简介: 该论文提出Hi3DEval框架，解决3D生成评估中的单一对象视角问题，通过结合对象与部分评估及材料属性分析，构建了Hi3DBench数据集并开发了3D-aware自动化评分系统。**

- **链接: [http://arxiv.org/pdf/2508.05609v1](http://arxiv.org/pdf/2508.05609v1)**

> **作者:** Yuhan Zhang; Long Zhuo; Ziyang Chu; Tong Wu; Zhibing Li; Liang Pan; Dahua Lin; Ziwei Liu
>
> **备注:** Page: https://zyh482.github.io/Hi3DEval/
>
> **摘要:** Despite rapid advances in 3D content generation, quality assessment for the generated 3D assets remains challenging. Existing methods mainly rely on image-based metrics and operate solely at the object level, limiting their ability to capture spatial coherence, material authenticity, and high-fidelity local details. 1) To address these challenges, we introduce Hi3DEval, a hierarchical evaluation framework tailored for 3D generative content. It combines both object-level and part-level evaluation, enabling holistic assessments across multiple dimensions as well as fine-grained quality analysis. Additionally, we extend texture evaluation beyond aesthetic appearance by explicitly assessing material realism, focusing on attributes such as albedo, saturation, and metallicness. 2) To support this framework, we construct Hi3DBench, a large-scale dataset comprising diverse 3D assets and high-quality annotations, accompanied by a reliable multi-agent annotation pipeline. We further propose a 3D-aware automated scoring system based on hybrid 3D representations. Specifically, we leverage video-based representations for object-level and material-subject evaluations to enhance modeling of spatio-temporal consistency and employ pretrained 3D features for part-level perception. Extensive experiments demonstrate that our approach outperforms existing image-based metrics in modeling 3D characteristics and achieves superior alignment with human preference, providing a scalable alternative to manual evaluations. The project page is available at https://zyh482.github.io/Hi3DEval/.
>
---
#### [new 039] AI vs. Human Moderators: A Comparative Evaluation of Multimodal LLMs in Content Moderation for Brand Safety
- **分类: cs.CV; I.2.10; I.2.7; H.3.3; H.4.3; K.4.1**

- **简介: 该论文旨在比较多模态LLM在品牌安全分类中的有效性，解决在线视频内容增长导致的安全审查需求超人类能力的问题，通过构建多语言、多模态数据集并进行性能对比，评估LLMs的准确性与成本效率，同时探讨其局限性并发布数据以推动未来研究。（99字）**

- **链接: [http://arxiv.org/pdf/2508.05527v1](http://arxiv.org/pdf/2508.05527v1)**

> **作者:** Adi Levi; Or Levi; Sardhendu Mishra; Jonathan Morra
>
> **备注:** Accepted to the Computer Vision in Advertising and Marketing (CVAM) workshop at ICCV 2025
>
> **摘要:** As the volume of video content online grows exponentially, the demand for moderation of unsafe videos has surpassed human capabilities, posing both operational and mental health challenges. While recent studies demonstrated the merits of Multimodal Large Language Models (MLLMs) in various video understanding tasks, their application to multimodal content moderation, a domain that requires nuanced understanding of both visual and textual cues, remains relatively underexplored. In this work, we benchmark the capabilities of MLLMs in brand safety classification, a critical subset of content moderation for safe-guarding advertising integrity. To this end, we introduce a novel, multimodal and multilingual dataset, meticulously labeled by professional reviewers in a multitude of risk categories. Through a detailed comparative analysis, we demonstrate the effectiveness of MLLMs such as Gemini, GPT, and Llama in multimodal brand safety, and evaluate their accuracy and cost efficiency compared to professional human reviewers. Furthermore, we present an in-depth discussion shedding light on limitations of MLLMs and failure cases. We are releasing our dataset alongside this paper to facilitate future research on effective and responsible brand safety and content moderation.
>
---
#### [new 040] Revealing Temporal Label Noise in Multimodal Hateful Video Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究了多模态Hateful视频分类中时间标签噪声的影响，旨在通过细化标注解决粗粒度视频注释带来的语义混杂问题。实验表明，时间戳噪声显著改变了模型决策边界并削弱了分类能力，揭示了时序连续性与上下文依赖性的本质特征。**

- **链接: [http://arxiv.org/pdf/2508.04900v1](http://arxiv.org/pdf/2508.04900v1)**

> **作者:** Shuonan Yang; Tailin Chen; Rahul Singh; Jiangbei Yue; Jianbo Jiao; Zeyu Fu
>
> **摘要:** The rapid proliferation of online multimedia content has intensified the spread of hate speech, presenting critical societal and regulatory challenges. While recent work has advanced multimodal hateful video detection, most approaches rely on coarse, video-level annotations that overlook the temporal granularity of hateful content. This introduces substantial label noise, as videos annotated as hateful often contain long non-hateful segments. In this paper, we investigate the impact of such label ambiguity through a fine-grained approach. Specifically, we trim hateful videos from the HateMM and MultiHateClip English datasets using annotated timestamps to isolate explicitly hateful segments. We then conduct an exploratory analysis of these trimmed segments to examine the distribution and characteristics of both hateful and non-hateful content. This analysis highlights the degree of semantic overlap and the confusion introduced by coarse, video-level annotations. Finally, controlled experiments demonstrated that time-stamp noise fundamentally alters model decision boundaries and weakens classification confidence, highlighting the inherent context dependency and temporal continuity of hate speech expression. Our findings provide new insights into the temporal dynamics of multimodal hateful videos and highlight the need for temporally aware models and benchmarks for improved robustness and interpretability. Code and data are available at https://github.com/Multimodal-Intelligence-Lab-MIL/HatefulVideoLabelNoise.
>
---
#### [new 041] HAMoBE: Hierarchical and Adaptive Mixture of Biometric Experts for Video-based Person ReID
- **分类: cs.CV**

- **简介: 该论文属于人像识别任务，旨在解决传统视频ReID方法忽视多尺度特征提取的问题。通过构建Hierarchical Adaptive Mixture of Biometric Experts框架，融合预训练大模型的多层特征与自适应决策机制，显著提升了视频场景下的匹配准确率。**

- **链接: [http://arxiv.org/pdf/2508.05038v1](http://arxiv.org/pdf/2508.05038v1)**

> **作者:** Yiyang Su; Yunping Shi; Feng Liu; Xiaoming Liu
>
> **备注:** Published at ICCV 2025
>
> **摘要:** Recently, research interest in person re-identification (ReID) has increasingly focused on video-based scenarios, which are essential for robust surveillance and security in varied and dynamic environments. However, existing video-based ReID methods often overlook the necessity of identifying and selecting the most discriminative features from both videos in a query-gallery pair for effective matching. To address this issue, we propose a novel Hierarchical and Adaptive Mixture of Biometric Experts (HAMoBE) framework, which leverages multi-layer features from a pre-trained large model (e.g., CLIP) and is designed to mimic human perceptual mechanisms by independently modeling key biometric features--appearance, static body shape, and dynamic gait--and adaptively integrating them. Specifically, HAMoBE includes two levels: the first level extracts low-level features from multi-layer representations provided by the frozen large model, while the second level consists of specialized experts focusing on long-term, short-term, and temporal features. To ensure robust matching, we introduce a new dual-input decision gating network that dynamically adjusts the contributions of each expert based on their relevance to the input scenarios. Extensive evaluations on benchmarks like MEVID demonstrate that our approach yields significant performance improvements (e.g., +13.0% Rank-1 accuracy).
>
---
#### [new 042] VER-Bench: Evaluating MLLMs on Reasoning with Fine-Grained Visual Evidence
- **分类: cs.CV**

- **简介: 该论文旨在评估基于大型语言模型在细粒度视觉证据推理中的能力，解决现有基准对细微信息提取与整合的不足，提出VER-Bench框架并设计374个问题集，以增强模型对隐性信息的理解与推理能力。**

- **链接: [http://arxiv.org/pdf/2508.04852v1](http://arxiv.org/pdf/2508.04852v1)**

> **作者:** Chenhui Qiang; Zhaoyang Wei; Xumeng Han Zipeng Wang; Siyao Li; Xiangyuan Lan; Jianbin Jiao; Zhenjun Han
>
> **备注:** Accept by ACMM2025 Dataset track
>
> **摘要:** With the rapid development of MLLMs, evaluating their visual capabilities has become increasingly crucial. Current benchmarks primarily fall into two main types: basic perception benchmarks, which focus on local details but lack deep reasoning (e.g., "what is in the image?"), and mainstream reasoning benchmarks, which concentrate on prominent image elements but may fail to assess subtle clues requiring intricate analysis. However, profound visual understanding and complex reasoning depend more on interpreting subtle, inconspicuous local details than on perceiving salient, macro-level objects. These details, though occupying minimal image area, often contain richer, more critical information for robust analysis. To bridge this gap, we introduce the VER-Bench, a novel framework to evaluate MLLMs' ability to: 1) identify fine-grained visual clues, often occupying on average just 0.25% of the image area; 2) integrate these clues with world knowledge for complex reasoning. Comprising 374 carefully designed questions across Geospatial, Temporal, Situational, Intent, System State, and Symbolic reasoning, each question in VER-Bench is accompanied by structured evidence: visual clues and question-related reasoning derived from them. VER-Bench reveals current models' limitations in extracting subtle visual evidence and constructing evidence-based arguments, highlighting the need to enhance models's capabilities in fine-grained visual evidence extraction, integration, and reasoning for genuine visual understanding and human-like analysis. Dataset and additional materials are available https://github.com/verbta/ACMMM-25-Materials.
>
---
#### [new 043] Rotation Equivariant Arbitrary-scale Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文提出了一种具有旋转等价性的任意尺度图像超分辨率方法，解决了传统ASISR因几何变形导致的伪影问题。通过重构编码器与隐神经表示模块的结构，实现了端到端的旋转等价性维护，并验证了其理论优势与实际效果。**

- **链接: [http://arxiv.org/pdf/2508.05160v1](http://arxiv.org/pdf/2508.05160v1)**

> **作者:** Qi Xie; Jiahong Fu; Zongben Xu; Deyu Meng
>
> **备注:** Accepted by IEEE TPAMI, code and supplementary material is available at https://github.com/XieQi2015/Equivariant-ASISR
>
> **摘要:** The arbitrary-scale image super-resolution (ASISR), a recent popular topic in computer vision, aims to achieve arbitrary-scale high-resolution recoveries from a low-resolution input image. This task is realized by representing the image as a continuous implicit function through two fundamental modules, a deep-network-based encoder and an implicit neural representation (INR) module. Despite achieving notable progress, a crucial challenge of such a highly ill-posed setting is that many common geometric patterns, such as repetitive textures, edges, or shapes, are seriously warped and deformed in the low-resolution images, naturally leading to unexpected artifacts appearing in their high-resolution recoveries. Embedding rotation equivariance into the ASISR network is thus necessary, as it has been widely demonstrated that this enhancement enables the recovery to faithfully maintain the original orientations and structural integrity of geometric patterns underlying the input image. Motivated by this, we make efforts to construct a rotation equivariant ASISR method in this study. Specifically, we elaborately redesign the basic architectures of INR and encoder modules, incorporating intrinsic rotation equivariance capabilities beyond those of conventional ASISR networks. Through such amelioration, the ASISR network can, for the first time, be implemented with end-to-end rotational equivariance maintained from input to output. We also provide a solid theoretical analysis to evaluate its intrinsic equivariance error, demonstrating its inherent nature of embedding such an equivariance structure. The superiority of the proposed method is substantiated by experiments conducted on both simulated and real datasets. We also validate that the proposed framework can be readily integrated into current ASISR methods in a plug \& play manner to further enhance their performance.
>
---
#### [new 044] PhysPatch: A Physically Realizable and Transferable Adversarial Patch Attack for Multimodal Large Language Models-based Autonomous Driving Systems
- **分类: cs.CV**

- **简介: 该论文旨在开发物理可行且可迁移的对抗性特征提取框架，解决基于多模态大语言模型的自动驾驶系统中对抗性攻击问题。提出PhysPatch通过优化特征位置、形状与内容，结合SVD对齐和场域细化技术，显著提升攻击有效性与实现实用性。**

- **链接: [http://arxiv.org/pdf/2508.05167v1](http://arxiv.org/pdf/2508.05167v1)**

> **作者:** Qi Guo; Xiaojun Jia; Shanmin Pang; Simeng Qin; Lin Wang; Ju Jia; Yang Liu; Qing Guo
>
> **摘要:** Multimodal Large Language Models (MLLMs) are becoming integral to autonomous driving (AD) systems due to their strong vision-language reasoning capabilities. However, MLLMs are vulnerable to adversarial attacks, particularly adversarial patch attacks, which can pose serious threats in real-world scenarios. Existing patch-based attack methods are primarily designed for object detection models and perform poorly when transferred to MLLM-based systems due to the latter's complex architectures and reasoning abilities. To address these limitations, we propose PhysPatch, a physically realizable and transferable adversarial patch framework tailored for MLLM-based AD systems. PhysPatch jointly optimizes patch location, shape, and content to enhance attack effectiveness and real-world applicability. It introduces a semantic-based mask initialization strategy for realistic placement, an SVD-based local alignment loss with patch-guided crop-resize to improve transferability, and a potential field-based mask refinement method. Extensive experiments across open-source, commercial, and reasoning-capable MLLMs demonstrate that PhysPatch significantly outperforms prior methods in steering MLLM-based AD systems toward target-aligned perception and planning outputs. Moreover, PhysPatch consistently places adversarial patches in physically feasible regions of AD scenes, ensuring strong real-world applicability and deployability.
>
---
#### [new 045] Propagating Sparse Depth via Depth Foundation Model for Out-of-Distribution Depth Completion
- **分类: cs.CV**

- **简介: 该论文旨在解决深度完成任务中的稀疏数据重建问题，通过利用深度基础模型提取环境信息并设计双空间传播方案提升鲁棒性，同时在NYUv2等真实数据集上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.04984v1](http://arxiv.org/pdf/2508.04984v1)**

> **作者:** Shenglun Chen; Xinzhu Ma; Hong Zhang; Haojie Li; Zhihui Wang
>
> **备注:** Accepted by IEEE TIP
>
> **摘要:** Depth completion is a pivotal challenge in computer vision, aiming at reconstructing the dense depth map from a sparse one, typically with a paired RGB image. Existing learning based models rely on carefully prepared but limited data, leading to significant performance degradation in out-of-distribution (OOD) scenarios. Recent foundation models have demonstrated exceptional robustness in monocular depth estimation through large-scale training, and using such models to enhance the robustness of depth completion models is a promising solution. In this work, we propose a novel depth completion framework that leverages depth foundation models to attain remarkable robustness without large-scale training. Specifically, we leverage a depth foundation model to extract environmental cues, including structural and semantic context, from RGB images to guide the propagation of sparse depth information into missing regions. We further design a dual-space propagation approach, without any learnable parameters, to effectively propagates sparse depth in both 3D and 2D spaces to maintain geometric structure and local consistency. To refine the intricate structure, we introduce a learnable correction module to progressively adjust the depth prediction towards the real depth. We train our model on the NYUv2 and KITTI datasets as in-distribution datasets and extensively evaluate the framework on 16 other datasets. Our framework performs remarkably well in the OOD scenarios and outperforms existing state-of-the-art depth completion methods. Our models are released in https://github.com/shenglunch/PSD.
>
---
#### [new 046] Leveraging AI to Accelerate Clinical Data Cleaning: A Comparative Study of AI-Assisted vs. Traditional Methods
- **分类: cs.CV**

- **简介: 该论文旨在比较AI与传统方法在临床数据清洗中的效率，提出Octozi平台并验证其效果，发现AI可显著提升处理速度（6.03倍）并减少错误率（54.67%→8.48%），同时降低假阳性，证明AI在加速临床数据流程、优化成本与合规性方面具有潜力。**

- **链接: [http://arxiv.org/pdf/2508.05519v1](http://arxiv.org/pdf/2508.05519v1)**

> **作者:** Matthew Purri; Amit Patel; Erik Deurrell
>
> **摘要:** Clinical trial data cleaning represents a critical bottleneck in drug development, with manual review processes struggling to manage exponentially increasing data volumes and complexity. This paper presents Octozi, an artificial intelligence-assisted platform that combines large language models with domain-specific heuristics to transform clinical data review. In a controlled experimental study with experienced clinical reviewers (n=10), we demonstrate that AI assistance increased data cleaning throughput by 6.03-fold while simultaneously decreasing cleaning errors from 54.67% to 8.48% (a 6.44-fold improvement). Crucially, the system reduced false positive queries by 15.48-fold, minimizing unnecessary site burden. These improvements were consistent across reviewers regardless of experience level, suggesting broad applicability. Our findings indicate that AI-assisted approaches can address fundamental inefficiencies in clinical trial operations, potentially accelerating drug development timelines and reducing costs while maintaining regulatory compliance. This work establishes a framework for integrating AI into safety-critical clinical workflows and demonstrates the transformative potential of human-AI collaboration in pharmaceutical clinical trials.
>
---
#### [new 047] Test-Time Reinforcement Learning for GUI Grounding via Region Consistency
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文旨在解决GUI地面任务中的高成本与低效率问题，提出基于区域一致性的Test-Time强化学习方法(GUI-RC)，通过构建投票网格提升模型精度，实现2-3%的性能提升，为自主GUI代理提供更高效的数据支持。**

- **链接: [http://arxiv.org/pdf/2508.05615v1](http://arxiv.org/pdf/2508.05615v1)**

> **作者:** Yong Du; Yuchen Yan; Fei Tang; Zhengxi Lu; Chang Zong; Weiming Lu; Shengpei Jiang; Yongliang Shen
>
> **备注:** Project Page: https://zju-real.github.io/gui-rcpo Code: https://github.com/zju-real/gui-rcpo
>
> **摘要:** Graphical User Interface (GUI) grounding, the task of mapping natural language instructions to precise screen coordinates, is fundamental to autonomous GUI agents. While existing methods achieve strong performance through extensive supervised training or reinforcement learning with labeled rewards, they remain constrained by the cost and availability of pixel-level annotations. We observe that when models generate multiple predictions for the same GUI element, the spatial overlap patterns reveal implicit confidence signals that can guide more accurate localization. Leveraging this insight, we propose GUI-RC (Region Consistency), a test-time scaling method that constructs spatial voting grids from multiple sampled predictions to identify consensus regions where models show highest agreement. Without any training, GUI-RC improves accuracy by 2-3% across various architectures on ScreenSpot benchmarks. We further introduce GUI-RCPO (Region Consistency Policy Optimization), which transforms these consistency patterns into rewards for test-time reinforcement learning. By computing how well each prediction aligns with the collective consensus, GUI-RCPO enables models to iteratively refine their outputs on unlabeled data during inference. Extensive experiments demonstrate the generality of our approach: GUI-RC boosts Qwen2.5-VL-3B-Instruct from 80.11% to 83.57% on ScreenSpot-v2, while GUI-RCPO further improves it to 85.14% through self-supervised optimization. Our approach reveals the untapped potential of test-time scaling and test-time reinforcement learning for GUI grounding, offering a promising path toward more robust and data-efficient GUI agents.
>
---
#### [new 048] ACM Multimedia Grand Challenge on ENT Endoscopy Analysis
- **分类: cs.CV**

- **简介: 该论文提出ENTRep任务，旨在通过整合细粒度分类与双语监督进行图像-图像和文本-图像检索，解决ENT领域中精细诊断需求不足的问题，构建公共/私有基准并评价性能。**

- **链接: [http://arxiv.org/pdf/2508.04801v1](http://arxiv.org/pdf/2508.04801v1)**

> **作者:** Trong-Thuan Nguyen; Viet-Tham Huynh; Thao Thi Phuong Dao; Ha Nguyen Thi; Tien To Vu Thuy; Uyen Hanh Tran; Tam V. Nguyen; Thanh Dinh Le; Minh-Triet Tran
>
> **摘要:** Automated analysis of endoscopic imagery is a critical yet underdeveloped component of ENT (ear, nose, and throat) care, hindered by variability in devices and operators, subtle and localized findings, and fine-grained distinctions such as laterality and vocal-fold state. In addition to classification, clinicians require reliable retrieval of similar cases, both visually and through concise textual descriptions. These capabilities are rarely supported by existing public benchmarks. To this end, we introduce ENTRep, the ACM Multimedia 2025 Grand Challenge on ENT endoscopy analysis, which integrates fine-grained anatomical classification with image-to-image and text-to-image retrieval under bilingual (Vietnamese and English) clinical supervision. Specifically, the dataset comprises expert-annotated images, labeled for anatomical region and normal or abnormal status, and accompanied by dual-language narrative descriptions. In addition, we define three benchmark tasks, standardize the submission protocol, and evaluate performance on public and private test splits using server-side scoring. Moreover, we report results from the top-performing teams and provide an insight discussion.
>
---
#### [new 049] CoCAViT: Compact Vision Transformer with Robust Global Coordination
- **分类: cs.CV**

- **简介: 该论文旨在解决小规模视觉模型在出错数据（OOD）下的泛化能力不足问题，通过改进全局协调机制（CoCAViT）提升鲁棒性，实现高效且稳定的视觉表示。**

- **链接: [http://arxiv.org/pdf/2508.05307v1](http://arxiv.org/pdf/2508.05307v1)**

> **作者:** Xuyang Wang; Lingjuan Miao; Zhiqiang Zhou
>
> **摘要:** In recent years, large-scale visual backbones have demonstrated remarkable capabilities in learning general-purpose features from images via extensive pre-training. Concurrently, many efficient architectures have emerged that have performance comparable to that of larger models on in-domain benchmarks. However, we observe that for smaller models, the performance drop on out-of-distribution (OOD) data is disproportionately larger, indicating a deficiency in the generalization performance of existing efficient models. To address this, we identify key architectural bottlenecks and inappropriate design choices that contribute to this issue, retaining robustness for smaller models. To restore the global field of pure window attention, we further introduce a Coordinator-patch Cross Attention (CoCA) mechanism, featuring dynamic, domain-aware global tokens that enhance local-global feature modeling and adaptively capture robust patterns across domains with minimal computational overhead. Integrating these advancements, we present CoCAViT, a novel visual backbone designed for robust real-time visual representation. Extensive experiments empirically validate our design. At a resolution of 224*224, CoCAViT-28M achieves 84.0% top-1 accuracy on ImageNet-1K, with significant gains on multiple OOD benchmarks, compared to competing models. It also attains 52.2 mAP on COCO object detection and 51.3 mIOU on ADE20K semantic segmentation, while maintaining low latency.
>
---
#### [new 050] Decoupling Continual Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文研究了持续语义分割（CSS）任务，旨在解决传统方法中因紧凑耦合导致的子最优平衡问题。通过引入两阶段框架DecoupleCSS，分离检测与分割任务，利用LoRA预训练模型和SAM生成分割图，实现了对旧知识的保留与新类学习的高效结合，达到优异性能。**

- **链接: [http://arxiv.org/pdf/2508.05065v1](http://arxiv.org/pdf/2508.05065v1)**

> **作者:** Yifu Guo; Yuquan Lu; Wentao Zhang; Zishan Xu; Dexia Chen; Siyu Zhang; Yizhe Zhang; Ruixuan Wang
>
> **备注:** https://github.com/euyis1019/Decoupling-Continual-Semantic-Segmentation
>
> **摘要:** Continual Semantic Segmentation (CSS) requires learning new classes without forgetting previously acquired knowledge, addressing the fundamental challenge of catastrophic forgetting in dense prediction tasks. However, existing CSS methods typically employ single-stage encoder-decoder architectures where segmentation masks and class labels are tightly coupled, leading to interference between old and new class learning and suboptimal retention-plasticity balance. We introduce DecoupleCSS, a novel two-stage framework for CSS. By decoupling class-aware detection from class-agnostic segmentation, DecoupleCSS enables more effective continual learning, preserving past knowledge while learning new classes. The first stage leverages pre-trained text and image encoders, adapted using LoRA, to encode class-specific information and generate location-aware prompts. In the second stage, the Segment Anything Model (SAM) is employed to produce precise segmentation masks, ensuring that segmentation knowledge is shared across both new and previous classes. This approach improves the balance between retention and adaptability in CSS, achieving state-of-the-art performance across a variety of challenging tasks. Our code is publicly available at: https://github.com/euyis1019/Decoupling-Continual-Semantic-Segmentation.
>
---
#### [new 051] MagicHOI: Leveraging 3D Priors for Accurate Hand-object Reconstruction from Short Monocular Video Clips
- **分类: cs.CV**

- **简介: 该论文旨在解决传统RGB方法因视角受限导致的模板自由重建不足问题，提出MagicHOI通过新颖的3D视图合成扩散模型与可见接触约束融合，实现手-物体从短单目视频中高效重建。**

- **链接: [http://arxiv.org/pdf/2508.05506v1](http://arxiv.org/pdf/2508.05506v1)**

> **作者:** Shibo Wang; Haonan He; Maria Parelli; Christoph Gebhardt; Zicong Fan; Jie Song
>
> **摘要:** Most RGB-based hand-object reconstruction methods rely on object templates, while template-free methods typically assume full object visibility. This assumption often breaks in real-world settings, where fixed camera viewpoints and static grips leave parts of the object unobserved, resulting in implausible reconstructions. To overcome this, we present MagicHOI, a method for reconstructing hands and objects from short monocular interaction videos, even under limited viewpoint variation. Our key insight is that, despite the scarcity of paired 3D hand-object data, large-scale novel view synthesis diffusion models offer rich object supervision. This supervision serves as a prior to regularize unseen object regions during hand interactions. Leveraging this insight, we integrate a novel view synthesis model into our hand-object reconstruction framework. We further align hand to object by incorporating visible contact constraints. Our results demonstrate that MagicHOI significantly outperforms existing state-of-the-art hand-object reconstruction methods. We also show that novel view synthesis diffusion priors effectively regularize unseen object regions, enhancing 3D hand-object reconstruction.
>
---
#### [new 052] Attribute Guidance With Inherent Pseudo-label For Occluded Person Re-identification
- **分类: cs.CV**

- **简介: 该论文旨在解决遮挡场景下的行人重识别问题，通过引入属性指导机制，结合预训练模型的内在伪标签和双引导策略，有效提升了细粒度属性信息的提取能力。**

- **链接: [http://arxiv.org/pdf/2508.04998v1](http://arxiv.org/pdf/2508.04998v1)**

> **作者:** Rui Zhi; Zhen Yang; Haiyang Zhang
>
> **备注:** 8 pages, 2 supplement pages, 3 figures, ECAI2025
>
> **摘要:** Person re-identification (Re-ID) aims to match person images across different camera views, with occluded Re-ID addressing scenarios where pedestrians are partially visible. While pre-trained vision-language models have shown effectiveness in Re-ID tasks, they face significant challenges in occluded scenarios by focusing on holistic image semantics while neglecting fine-grained attribute information. This limitation becomes particularly evident when dealing with partially occluded pedestrians or when distinguishing between individuals with subtle appearance differences. To address this limitation, we propose Attribute-Guide ReID (AG-ReID), a novel framework that leverages pre-trained models' inherent capabilities to extract fine-grained semantic attributes without additional data or annotations. Our framework operates through a two-stage process: first generating attribute pseudo-labels that capture subtle visual characteristics, then introducing a dual-guidance mechanism that combines holistic and fine-grained attribute information to enhance image feature extraction. Extensive experiments demonstrate that AG-ReID achieves state-of-the-art results on multiple widely-used Re-ID datasets, showing significant improvements in handling occlusions and subtle attribute differences while maintaining competitive performance on standard Re-ID scenarios.
>
---
#### [new 053] Deep Learning-based Animal Behavior Analysis: Insights from Mouse Chronic Pain Models
- **分类: cs.CV**

- **简介: 该研究旨在通过深度学习分析小鼠慢性疼痛模型的行为特征，解决传统手工标注方法难以捕捉慢性疼痛本质的问题。利用动作空间投影技术提取特征并建立数据集，验证了其在多类别疼痛分类上的优越性（48.41% vs 21.33%、73.1%），并揭示药物效果差异，为临床疼痛研究提供新思路。**

- **链接: [http://arxiv.org/pdf/2508.05138v1](http://arxiv.org/pdf/2508.05138v1)**

> **作者:** Yu-Hsi Chen; Wei-Hsin Chen; Chien-Yao Wang; Hong-Yuan Mark Liao; James C. Liao; Chien-Chang Chen
>
> **摘要:** Assessing chronic pain behavior in mice is critical for preclinical studies. However, existing methods mostly rely on manual labeling of behavioral features, and humans lack a clear understanding of which behaviors best represent chronic pain. For this reason, existing methods struggle to accurately capture the insidious and persistent behavioral changes in chronic pain. This study proposes a framework to automatically discover features related to chronic pain without relying on human-defined action labels. Our method uses universal action space projector to automatically extract mouse action features, and avoids the potential bias of human labeling by retaining the rich behavioral information in the original video. In this paper, we also collected a mouse pain behavior dataset that captures the disease progression of both neuropathic and inflammatory pain across multiple time points. Our method achieves 48.41\% accuracy in a 15-class pain classification task, significantly outperforming human experts (21.33\%) and the widely used method B-SOiD (30.52\%). Furthermore, when the classification is simplified to only three categories, i.e., neuropathic pain, inflammatory pain, and no pain, then our method achieves an accuracy of 73.1\%, which is notably higher than that of human experts (48\%) and B-SOiD (58.43\%). Finally, our method revealed differences in drug efficacy for different types of pain on zero-shot Gabapentin drug testing, and the results were consistent with past drug efficacy literature. This study demonstrates the potential clinical application of our method, which can provide new insights into pain research and related drug development.
>
---
#### [new 054] FaceAnonyMixer: Cancelable Faces via Identity Consistent Latent Space Mixing
- **分类: cs.CV**

- **简介: 该论文提出了一种基于预训练生成模型的可取消人脸生成框架FaceAnonyMixer，旨在解决隐私保护与生物识别需求之间的矛盾，通过将真实面部与合成代码混合并优化损失函数实现高精度匿名化，相比现有方法实现了11%的识别性能提升。**

- **链接: [http://arxiv.org/pdf/2508.05636v1](http://arxiv.org/pdf/2508.05636v1)**

> **作者:** Mohammed Talha Alam; Fahad Shamshad; Fakhri Karray; Karthik Nandakumar
>
> **备注:** Accepted at the International Joint Conference on Biometrics (IJCB) 2025
>
> **摘要:** Advancements in face recognition (FR) technologies have amplified privacy concerns, necessitating methods that protect identity while maintaining recognition utility. Existing face anonymization methods typically focus on obscuring identity but fail to meet the requirements of biometric template protection, including revocability, unlinkability, and irreversibility. We propose FaceAnonyMixer, a cancelable face generation framework that leverages the latent space of a pre-trained generative model to synthesize privacy-preserving face images. The core idea of FaceAnonyMixer is to irreversibly mix the latent code of a real face image with a synthetic code derived from a revocable key. The mixed latent code is further refined through a carefully designed multi-objective loss to satisfy all cancelable biometric requirements. FaceAnonyMixer is capable of generating high-quality cancelable faces that can be directly matched using existing FR systems without requiring any modifications. Extensive experiments on benchmark datasets demonstrate that FaceAnonyMixer delivers superior recognition accuracy while providing significantly stronger privacy protection, achieving over an 11% gain on commercial API compared to recent cancelable biometric methods. Code is available at: https://github.com/talha-alam/faceanonymixer.
>
---
#### [new 055] WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction
- **分类: cs.CV**

- **简介: 该论文旨在解决现有视觉分词器在压缩与重建之间的平衡问题，通过组间无量化（GQ）和生成解码器（GD）创新实现高效高精度可视化重建，验证了WeTok在ImageNet 50k上的卓越性能及超越现有模型的效果。**

- **链接: [http://arxiv.org/pdf/2508.05599v1](http://arxiv.org/pdf/2508.05599v1)**

> **作者:** Shaobin Zhuang; Yiwei Guo; Canmiao Fu; Zhipeng Huang; Zeyue Tian; Ying Zhang; Chen Li; Yali Wang
>
> **备注:** 23 pages, 10 figures, 37 tables
>
> **摘要:** Visual tokenizer is a critical component for vision generation. However, the existing tokenizers often face unsatisfactory trade-off between compression ratios and reconstruction fidelity. To fill this gap, we introduce a powerful and concise WeTok tokenizer, which surpasses the previous leading tokenizers via two core innovations. (1) Group-wise lookup-free Quantization (GQ). We partition the latent features into groups, and perform lookup-free quantization for each group. As a result, GQ can efficiently overcome memory and computation limitations of prior tokenizers, while achieving a reconstruction breakthrough with more scalable codebooks. (2) Generative Decoding (GD). Different from prior tokenizers, we introduce a generative decoder with a prior of extra noise variable. In this case, GD can probabilistically model the distribution of visual data conditioned on discrete tokens, allowing WeTok to reconstruct visual details, especially at high compression ratios. Extensive experiments on mainstream benchmarks show superior performance of our WeTok. On the ImageNet 50k validation set, WeTok achieves a record-low zero-shot rFID (WeTok: 0.12 vs. FLUX-VAE: 0.18 vs. SD-VAE 3.5: 0.19). Furthermore, our highest compression model achieves a zero-shot rFID of 3.49 with a compression ratio of 768, outperforming Cosmos (384) 4.57 which has only 50% compression rate of ours. Code and models are available: https://github.com/zhuangshaobin/WeTok.
>
---
#### [new 056] AdaFusion: Prompt-Guided Inference with Adaptive Fusion of Pathology Foundation Models
- **分类: cs.CV**

- **简介: 该论文提出AdaFusion框架，旨在解决病理基础模型（PFMs）因偏见导致的下游应用不透明问题，通过动态整合多模型特征并采用轻量注意力机制实现跨域融合与性能提升。**

- **链接: [http://arxiv.org/pdf/2508.05084v1](http://arxiv.org/pdf/2508.05084v1)**

> **作者:** Yuxiang Xiao; Yang Hu; Bin Li; Tianyang Zhang; Zexi Li; Huazhu Fu; Jens Rittscher; Kaixiang Yang
>
> **备注:** 6 Tables, 11 Figures
>
> **摘要:** Pathology foundation models (PFMs) have demonstrated strong representational capabilities through self-supervised pre-training on large-scale, unannotated histopathology image datasets. However, their diverse yet opaque pretraining contexts, shaped by both data-related and structural/training factors, introduce latent biases that hinder generalisability and transparency in downstream applications. In this paper, we propose AdaFusion, a novel prompt-guided inference framework that, to our knowledge, is among the very first to dynamically integrate complementary knowledge from multiple PFMs. Our method compresses and aligns tile-level features from diverse models and employs a lightweight attention mechanism to adaptively fuse them based on tissue phenotype context. We evaluate AdaFusion on three real-world benchmarks spanning treatment response prediction, tumour grading, and spatial gene expression inference. Our approach consistently surpasses individual PFMs across both classification and regression tasks, while offering interpretable insights into each model's biosemantic specialisation. These results highlight AdaFusion's ability to bridge heterogeneous PFMs, achieving both enhanced performance and interpretability of model-specific inductive biases.
>
---
#### [new 057] Multi-tracklet Tracking for Generic Targets with Adaptive Detection Clustering
- **分类: cs.CV**

- **简介: 该论文研究了多目标跟踪任务，针对传统方法在低置信度、弱动态和长期遮挡下的性能不足，提出了结合灵活tracklet生成与多轨迹关联的MTT框架，旨在提升通用目标追踪的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.05172v1](http://arxiv.org/pdf/2508.05172v1)**

> **作者:** Zewei Wu; Longhao Wang; Cui Wang; César Teixeira; Wei Ke; Zhang Xiong
>
> **摘要:** Tracking specific targets, such as pedestrians and vehicles, has been the focus of recent vision-based multitarget tracking studies. However, in some real-world scenarios, unseen categories often challenge existing methods due to low-confidence detections, weak motion and appearance constraints, and long-term occlusions. To address these issues, this article proposes a tracklet-enhanced tracker called Multi-Tracklet Tracking (MTT) that integrates flexible tracklet generation into a multi-tracklet association framework. This framework first adaptively clusters the detection results according to their short-term spatio-temporal correlation into robust tracklets and then estimates the best tracklet partitions using multiple clues, such as location and appearance over time to mitigate error propagation in long-term association. Finally, extensive experiments on the benchmark for generic multiple object tracking demonstrate the competitiveness of the proposed framework.
>
---
#### [new 058] MELLA: Bridging Linguistic Capability and Cultural Groundedness for Low-Resource Language MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文旨在解决低资源语言模型在多模态与文化维度上的能力不足问题，通过构建双源数据集MELLA（结合文化相关web alt-text和语言生成描述），提升模型在不同语言环境下的泛化能力与描述质量，验证了文化与语言双重增强对低资源语言模型的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05502v1](http://arxiv.org/pdf/2508.05502v1)**

> **作者:** Yufei Gao; Jiaying Fei; Nuo Chen; Ruirui Chen; Guohang Yan; Yunshi Lan; Botian Shi
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown remarkable performance in high-resource languages. However, their effectiveness diminishes significantly in the contexts of low-resource languages. Current multilingual enhancement methods are often limited to text modality or rely solely on machine translation. While such approaches help models acquire basic linguistic capabilities and produce "thin descriptions", they neglect the importance of multimodal informativeness and cultural groundedness, both of which are crucial for serving low-resource language users effectively. To bridge this gap, in this study, we identify two significant objectives for a truly effective MLLM in low-resource language settings, namely 1) linguistic capability and 2) cultural groundedness, placing special emphasis on cultural awareness. To achieve these dual objectives, we propose a dual-source strategy that guides the collection of data tailored to each goal, sourcing native web alt-text for culture and MLLM-generated captions for linguistics. As a concrete implementation, we introduce MELLA, a multimodal, multilingual dataset. Experiment results show that after fine-tuning on MELLA, there is a general performance improvement for the eight languages on various MLLM backbones, with models producing "thick descriptions". We verify that the performance gains are from both cultural knowledge enhancement and linguistic capability enhancement. Our dataset can be found at https://opendatalab.com/applyMultilingualCorpus.
>
---
#### [new 059] Cross-View Localization via Redundant Sliced Observations and A-Contrario Validation
- **分类: cs.CV**

- **简介: 该论文提出了一种基于冗余观察和a-Contrario验证的跨视图定位方法（Slice-Loc），解决了传统CVL方法仅获得单个相机姿态观测的问题，通过将图像分割为子图像并利用几何刚度公式消除错误姿态后，提升了定位精度并减少了误差比例。**

- **链接: [http://arxiv.org/pdf/2508.05369v1](http://arxiv.org/pdf/2508.05369v1)**

> **作者:** Yongjun Zhang; Mingtao Xiong; Yi Wan; Gui-Song Xia
>
> **摘要:** Cross-view localization (CVL) matches ground-level images with aerial references to determine the geo-position of a camera, enabling smart vehicles to self-localize offline in GNSS-denied environments. However, most CVL methods output only a single observation, the camera pose, and lack the redundant observations required by surveying principles, making it challenging to assess localization reliability through the mutual validation of observational data. To tackle this, we introduce Slice-Loc, a two-stage method featuring an a-contrario reliability validation for CVL. Instead of using the query image as a single input, Slice-Loc divides it into sub-images and estimates the 3-DoF pose for each slice, creating redundant and independent observations. Then, a geometric rigidity formula is proposed to filter out the erroneous 3-DoF poses, and the inliers are merged to generate the final camera pose. Furthermore, we propose a model that quantifies the meaningfulness of localization by estimating the number of false alarms (NFA), according to the distribution of the locations of the sliced images. By eliminating gross errors, Slice-Loc boosts localization accuracy and effectively detects failures. After filtering out mislocalizations, Slice-Loc reduces the proportion of errors exceeding 10 m to under 3\%. In cross-city tests on the DReSS dataset, Slice-Loc cuts the mean localization error from 4.47 m to 1.86 m and the mean orientation error from $\mathbf{3.42^{\circ}}$ to $\mathbf{1.24^{\circ}}$, outperforming state-of-the-art methods. Code and dataset will be available at: https://github.com/bnothing/Slice-Loc.
>
---
#### [new 060] UNCAGE: Contrastive Attention Guidance for Masked Generative Transformers in Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于文本到图像（T2I）生成任务，旨在解决传统方法（如Diffusion模型）在属性绑定和文本-图像对齐方面的局限性。提出了UNCAGE方法，通过利用注意力映射优化成分性生成，显著提升性能并减少推理开销。**

- **链接: [http://arxiv.org/pdf/2508.05399v1](http://arxiv.org/pdf/2508.05399v1)**

> **作者:** Wonjun Kang; Byeongkeun Ahn; Minjae Lee; Kevin Galim; Seunghyuk Oh; Hyung Il Koo; Nam Ik Cho
>
> **备注:** Code is available at https://github.com/furiosa-ai/uncage
>
> **摘要:** Text-to-image (T2I) generation has been actively studied using Diffusion Models and Autoregressive Models. Recently, Masked Generative Transformers have gained attention as an alternative to Autoregressive Models to overcome the inherent limitations of causal attention and autoregressive decoding through bidirectional attention and parallel decoding, enabling efficient and high-quality image generation. However, compositional T2I generation remains challenging, as even state-of-the-art Diffusion Models often fail to accurately bind attributes and achieve proper text-image alignment. While Diffusion Models have been extensively studied for this issue, Masked Generative Transformers exhibit similar limitations but have not been explored in this context. To address this, we propose Unmasking with Contrastive Attention Guidance (UNCAGE), a novel training-free method that improves compositional fidelity by leveraging attention maps to prioritize the unmasking of tokens that clearly represent individual objects. UNCAGE consistently improves performance in both quantitative and qualitative evaluations across multiple benchmarks and metrics, with negligible inference overhead. Our code is available at https://github.com/furiosa-ai/uncage.
>
---
#### [new 061] MOSEv2: A More Challenging Dataset for Video Object Segmentation in Complex Scenes
- **分类: cs.CV**

- **简介: 该论文旨在开发MOSEv2视频目标分割数据集，解决现有复杂场景下目标识别能力不足的问题。通过引入更多场景复杂性（如物体消失、遮挡、低光等），构建了包含5024个视频和70万高精度掩码的1074个对象的挑战性数据集，并在5种场景下评估了20个目标分割方法，表明其在真实世界任务中仍面临性能下降。**

- **链接: [http://arxiv.org/pdf/2508.05630v1](http://arxiv.org/pdf/2508.05630v1)**

> **作者:** Henghui Ding; Kaining Ying; Chang Liu; Shuting He; Xudong Jiang; Yu-Gang Jiang; Philip H. S. Torr; Song Bai
>
> **备注:** MOSEv2 Dataset Report, Project Page: https://mose.video/
>
> **摘要:** Video object segmentation (VOS) aims to segment specified target objects throughout a video. Although state-of-the-art methods have achieved impressive performance (e.g., 90+% J&F) on existing benchmarks such as DAVIS and YouTube-VOS, these datasets primarily contain salient, dominant, and isolated objects, limiting their generalization to real-world scenarios. To advance VOS toward more realistic environments, coMplex video Object SEgmentation (MOSEv1) was introduced to facilitate VOS research in complex scenes. Building on the strengths and limitations of MOSEv1, we present MOSEv2, a significantly more challenging dataset designed to further advance VOS methods under real-world conditions. MOSEv2 consists of 5,024 videos and over 701,976 high-quality masks for 10,074 objects across 200 categories. Compared to its predecessor, MOSEv2 introduces significantly greater scene complexity, including more frequent object disappearance and reappearance, severe occlusions and crowding, smaller objects, as well as a range of new challenges such as adverse weather (e.g., rain, snow, fog), low-light scenes (e.g., nighttime, underwater), multi-shot sequences, camouflaged objects, non-physical targets (e.g., shadows, reflections), scenarios requiring external knowledge, etc. We benchmark 20 representative VOS methods under 5 different settings and observe consistent performance drops. For example, SAM2 drops from 76.4% on MOSEv1 to only 50.9% on MOSEv2. We further evaluate 9 video object tracking methods and find similar declines, demonstrating that MOSEv2 presents challenges across tasks. These results highlight that despite high accuracy on existing datasets, current VOS methods still struggle under real-world complexities. MOSEv2 is publicly available at https://MOSE.video.
>
---
#### [new 062] Accelerating Conditional Prompt Learning via Masked Image Modeling for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在解决零样本与Few-shot分类中的过拟合问题。通过引入掩码图像建模（MIM），结合已有框架如CoOp/CoCoOp，优化条件提示生成流程，提升模型泛化能力并降低计算成本。**

- **链接: [http://arxiv.org/pdf/2508.04942v1](http://arxiv.org/pdf/2508.04942v1)**

> **作者:** Phuoc-Nguyen Bui; Khanh-Binh Nguyen; Hyunseung Choo
>
> **备注:** ACMMM-LAVA 2025, 10 pages, camera-ready version
>
> **摘要:** Vision-language models (VLMs) like CLIP excel in zero-shot learning but often require resource-intensive training to adapt to new tasks. Prompt learning techniques, such as CoOp and CoCoOp, offer efficient adaptation but tend to overfit to known classes, limiting generalization to unseen categories. We introduce ProMIM, a plug-and-play framework that enhances conditional prompt learning by integrating masked image modeling (MIM) into existing VLM pipelines. ProMIM leverages a simple yet effective masking strategy to generate robust, instance-conditioned prompts, seamlessly augmenting methods like CoOp and CoCoOp without altering their core architectures. By masking only visible image patches and using these representations to guide prompt generation, ProMIM improves feature robustness and mitigates overfitting, all while introducing negligible additional computational cost. Extensive experiments across zero-shot and few-shot classification tasks demonstrate that ProMIM consistently boosts generalization performance when plugged into existing approaches, providing a practical, lightweight solution for real-world vision-language applications.
>
---
#### [new 063] FS-IQA: Certified Feature Smoothing for Robust Image Quality Assessment
- **分类: cs.CV**

- **简介: 本研究提出一种基于特征空间的认证方法，解决图像质量评估模型在引入噪声时的鲁棒性问题，通过分析雅可比矩阵优化输入输出关系，并支持多种模型架构，实现高效且认证的图像质量评估。**

- **链接: [http://arxiv.org/pdf/2508.05516v1](http://arxiv.org/pdf/2508.05516v1)**

> **作者:** Ekaterina Shumitskaya; Dmitriy Vatolin; Anastasia Antsiferova
>
> **摘要:** We propose a novel certified defense method for Image Quality Assessment (IQA) models based on randomized smoothing with noise applied in the feature space rather than the input space. Unlike prior approaches that inject Gaussian noise directly into input images, often degrading visual quality, our method preserves image fidelity while providing robustness guarantees. To formally connect noise levels in the feature space with corresponding input-space perturbations, we analyze the maximum singular value of the backbone network's Jacobian. Our approach supports both full-reference (FR) and no-reference (NR) IQA models without requiring any architectural modifications, suitable for various scenarios. It is also computationally efficient, requiring a single backbone forward pass per image. Compared to previous methods, it reduces inference time by 99.5% without certification and by 20.6% when certification is applied. We validate our method with extensive experiments on two benchmark datasets, involving six widely-used FR and NR IQA models and comparisons against five state-of-the-art certified defenses. Our results demonstrate consistent improvements in correlation with subjective quality scores by up to 30.9%.
>
---
#### [new 064] CRAM: Large-scale Video Continual Learning with Bootstrapped Compression
- **分类: cs.CV; cs.LG; cs.PF**

- **简介: 该论文属于视频连续学习任务，旨在解决传统CL在长视频和高内存需求下的挑战，通过压缩视觉编码实现在线训练并防止灾难性遗忘，创新性地结合视频压缩与刷新机制，提升存储效率和系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.05001v1](http://arxiv.org/pdf/2508.05001v1)**

> **作者:** Shivani Mall; Joao F. Henriques
>
> **摘要:** Continual learning (CL) promises to allow neural networks to learn from continuous streams of inputs, instead of IID (independent and identically distributed) sampling, which requires random access to a full dataset. This would allow for much smaller storage requirements and self-sufficiency of deployed systems that cope with natural distribution shifts, similarly to biological learning. We focus on video CL employing a rehearsal-based approach, which reinforces past samples from a memory buffer. We posit that part of the reason why practical video CL is challenging is the high memory requirements of video, further exacerbated by long-videos and continual streams, which are at odds with the common rehearsal-buffer size constraints. To address this, we propose to use compressed vision, i.e. store video codes (embeddings) instead of raw inputs, and train a video classifier by IID sampling from this rolling buffer. Training a video compressor online (so not depending on any pre-trained networks) means that it is also subject to catastrophic forgetting. We propose a scheme to deal with this forgetting by refreshing video codes, which requires careful decompression with a previous version of the network and recompression with a new one. We name our method Continually Refreshed Amodal Memory (CRAM). We expand current video CL benchmarks to large-scale settings, namely EpicKitchens-100 and Kinetics-700, storing thousands of relatively long videos in under 2 GB, and demonstrate empirically that our video CL method outperforms prior art with a significantly reduced memory footprint.
>
---
#### [new 065] Smoothing Slot Attention Iterations and Recurrences
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在改进Slot Attention机制以提升图像/视频对象检测性能。解决冷启动查询信息不足及帧间共享差异的问题，通过预热模块与分步微调实现SA迭代与递归的优化。**

- **链接: [http://arxiv.org/pdf/2508.05417v1](http://arxiv.org/pdf/2508.05417v1)**

> **作者:** Rongzhen Zhao; Wenyan Yang; Juho Kannala; Joni Pajarinen
>
> **摘要:** Slot Attention (SA) and its variants lie at the heart of mainstream Object-Centric Learning (OCL). Objects in an image can be aggregated into respective slot vectors, by \textit{iteratively} refining cold-start query vectors, typically three times, via SA on image features. For video, such aggregation is \textit{recurrently} shared across frames, with queries cold-started on the first frame while transitioned from the previous frame's slots on non-first frames. However, the cold-start queries lack sample-specific cues thus hinder precise aggregation on the image or video's first frame; Also, non-first frames' queries are already sample-specific thus require transforms different from the first frame's aggregation. We address these issues for the first time with our \textit{SmoothSA}: (1) To smooth SA iterations on the image or video's first frame, we \textit{preheat} the cold-start queries with rich information of input features, via a tiny module self-distilled inside OCL; (2) To smooth SA recurrences across all video frames, we \textit{differentiate} the homogeneous transforms on the first and non-first frames, by using full and single iterations respectively. Comprehensive experiments on object discovery, recognition and downstream benchmarks validate our method's effectiveness. Further analyses intuitively illuminate how our method smooths SA iterations and recurrences. Our code is available in the supplement.
>
---
#### [new 066] DualMat: PBR Material Estimation via Coherent Dual-Path Diffusion
- **分类: cs.CV**

- **简介: 该论文提出DualMat，旨在解决复杂光照条件下的PBR材料估计问题。通过构建双路径扩散框架，融合预训练视觉知识与紧凑的latent空间设计，结合特征蒸馏和rectified flow提升效率并实现高精度金属/粗糙度估计。**

- **链接: [http://arxiv.org/pdf/2508.05060v1](http://arxiv.org/pdf/2508.05060v1)**

> **作者:** Yifeng Huang; Zhang Chen; Yi Xu; Minh Hoai; Zhong Li
>
> **摘要:** We present DualMat, a novel dual-path diffusion framework for estimating Physically Based Rendering (PBR) materials from single images under complex lighting conditions. Our approach operates in two distinct latent spaces: an albedo-optimized path leveraging pretrained visual knowledge through RGB latent space, and a material-specialized path operating in a compact latent space designed for precise metallic and roughness estimation. To ensure coherent predictions between the albedo-optimized and material-specialized paths, we introduce feature distillation during training. We employ rectified flow to enhance efficiency by reducing inference steps while maintaining quality. Our framework extends to high-resolution and multi-view inputs through patch-based estimation and cross-view attention, enabling seamless integration into image-to-3D pipelines. DualMat achieves state-of-the-art performance on both Objaverse and real-world data, significantly outperforming existing methods with up to 28% improvement in albedo estimation and 39% reduction in metallic-roughness prediction errors.
>
---
#### [new 067] B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding
- **分类: cs.CV**

- **简介: 该论文旨在设计一个4D LiDAR多模态语言模型基准，解决动态户外场景中时空推理问题。通过引入B4DL、数据生成框架及跨模态模型，解决了传统LLMs在高维LiDAR数据处理中的标注不足和架构缺失挑战，并提供统一解决方案。**

- **链接: [http://arxiv.org/pdf/2508.05269v1](http://arxiv.org/pdf/2508.05269v1)**

> **作者:** Changho Choi; Youngwoo Shin; Gyojin Han; Dong-Jae Lee; Junmo Kim
>
> **备注:** Accepted at ACM MM 2025
>
> **摘要:** Understanding dynamic outdoor environments requires capturing complex object interactions and their evolution over time. LiDAR-based 4D point clouds provide precise spatial geometry and rich temporal cues, making them ideal for representing real-world scenes. However, despite their potential, 4D LiDAR remains underexplored in the context of Multimodal Large Language Models (MLLMs) due to the absence of high-quality, modality-specific annotations and the lack of MLLM architectures capable of processing its high-dimensional composition. To address these challenges, we introduce B4DL, a new benchmark specifically designed for training and evaluating MLLMs on 4D LiDAR understanding. In addition, we propose a scalable data generation pipeline and an MLLM model that, for the first time, directly processes raw 4D LiDAR by bridging it with language understanding. Combined with our dataset and benchmark, our model offers a unified solution for spatio-temporal reasoning in dynamic outdoor environments. We provide rendered 4D LiDAR videos, generated dataset, and inference outputs on diverse scenarios at: https://mmb4dl.github.io/mmb4dl/
>
---
#### [new 068] AdvDINO: Domain-Adversarial Self-Supervised Representation Learning for Spatial Proteomics
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Domain-Adversarial Self-Supervised Learning框架，解决生物医学影像领域领域转换带来的特征鲁棒性不足问题，通过梯度反转层增强模型泛化能力并提升蛋白质组学特征学习效果。**

- **链接: [http://arxiv.org/pdf/2508.04955v1](http://arxiv.org/pdf/2508.04955v1)**

> **作者:** Stella Su; Marc Harary; Scott J. Rodig; William Lotter
>
> **摘要:** Self-supervised learning (SSL) has emerged as a powerful approach for learning visual representations without manual annotations. However, the robustness of standard SSL methods to domain shift -- systematic differences across data sources -- remains uncertain, posing an especially critical challenge in biomedical imaging where batch effects can obscure true biological signals. We present AdvDINO, a domain-adversarial self-supervised learning framework that integrates a gradient reversal layer into the DINOv2 architecture to promote domain-invariant feature learning. Applied to a real-world cohort of six-channel multiplex immunofluorescence (mIF) whole slide images from non-small cell lung cancer patients, AdvDINO mitigates slide-specific biases to learn more robust and biologically meaningful representations than non-adversarial baselines. Across $>5.46$ million mIF image tiles, the model uncovers phenotype clusters with distinct proteomic profiles and prognostic significance, and improves survival prediction in attention-based multiple instance learning. While demonstrated on mIF data, AdvDINO is broadly applicable to other imaging domains -- including radiology, remote sensing, and autonomous driving -- where domain shift and limited annotated data hinder model generalization and interpretability.
>
---
#### [new 069] SPA++: Generalized Graph Spectral Alignment for Versatile Domain Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出了一种通用图谱图谱对齐框架 SPA++，用于领域自适应任务，解决因忽略内域结构而难以迁移的知识问题，通过图谱正则化与细化邻居传播机制提升跨域适应能力并验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.05182v1](http://arxiv.org/pdf/2508.05182v1)**

> **作者:** Zhiqing Xiao; Haobo Wang; Xu Lu; Wentao Ye; Gang Chen; Junbo Zhao
>
> **备注:** The article has been accepted by Frontiers of Computer Science (FCS), with the DOI: {10.1007/s11704-025-50328-w}. arXiv admin note: text overlap with arXiv:2310.17594
>
> **摘要:** Domain Adaptation (DA) aims to transfer knowledge from a labeled source domain to an unlabeled or sparsely labeled target domain under domain shifts. Most prior works focus on capturing the inter-domain transferability but largely overlook rich intra-domain structures, which empirically results in even worse discriminability. To tackle this tradeoff, we propose a generalized graph SPectral Alignment framework, SPA++. Its core is briefly condensed as follows: (1)-by casting the DA problem to graph primitives, it composes a coarse graph alignment mechanism with a novel spectral regularizer toward aligning the domain graphs in eigenspaces; (2)-we further develop a fine-grained neighbor-aware propagation mechanism for enhanced discriminability in the target domain; (3)-by incorporating data augmentation and consistency regularization, SPA++ can adapt to complex scenarios including most DA settings and even challenging distribution scenarios. Furthermore, we also provide theoretical analysis to support our method, including the generalization bound of graph-based DA and the role of spectral alignment and smoothing consistency. Extensive experiments on benchmark datasets demonstrate that SPA++ consistently outperforms existing cutting-edge methods, achieving superior robustness and adaptability across various challenging adaptation scenarios.
>
---
#### [new 070] Multimodal Causal-Driven Representation Learning for Generalizable Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文旨在解决医学影像分割任务中领域泛化不足的问题，提出MCDRL框架整合因果推理与CLIP多模态能力，通过构建领域字典消除特定因素干扰并保留结构性信息，有效提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.05008v1](http://arxiv.org/pdf/2508.05008v1)**

> **作者:** Xusheng Liang; Lihua Zhou; Nianxin Li; Miao Xu; Ziyang Song; Dong Yi; Jinlin Wu; Hongbin Liu; Jiebo Luo; Zhen Lei
>
> **备注:** Under Review
>
> **摘要:** Vision-Language Models (VLMs), such as CLIP, have demonstrated remarkable zero-shot capabilities in various computer vision tasks. However, their application to medical imaging remains challenging due to the high variability and complexity of medical data. Specifically, medical images often exhibit significant domain shifts caused by various confounders, including equipment differences, procedure artifacts, and imaging modes, which can lead to poor generalization when models are applied to unseen domains. To address this limitation, we propose Multimodal Causal-Driven Representation Learning (MCDRL), a novel framework that integrates causal inference with the VLM to tackle domain generalization in medical image segmentation. MCDRL is implemented in two steps: first, it leverages CLIP's cross-modal capabilities to identify candidate lesion regions and construct a confounder dictionary through text prompts, specifically designed to represent domain-specific variations; second, it trains a causal intervention network that utilizes this dictionary to identify and eliminate the influence of these domain-specific variations while preserving the anatomical structural information critical for segmentation tasks. Extensive experiments demonstrate that MCDRL consistently outperforms competing methods, yielding superior segmentation accuracy and exhibiting robust generalizability.
>
---
#### [new 071] Unified modality separation: A vision-language framework for unsupervised domain adaptation
- **分类: cs.CV**

- **简介: 该论文提出了一种统一模态分离框架，用于无监督领域自适应，解决模态差异导致的性能下降问题。通过分离视觉与文本模态特征并进行测试，设计了实例级模态特征指标，提升了9%性能并实现了计算效率提升。**

- **链接: [http://arxiv.org/pdf/2508.04987v1](http://arxiv.org/pdf/2508.04987v1)**

> **作者:** Xinyao Li; Jingjing Li; Zhekai Du; Lei Zhu; Heng Tao Shen
>
> **备注:** Accepted to TPAMI
>
> **摘要:** Unsupervised domain adaptation (UDA) enables models trained on a labeled source domain to handle new unlabeled domains. Recently, pre-trained vision-language models (VLMs) have demonstrated promising zero-shot performance by leveraging semantic information to facilitate target tasks. By aligning vision and text embeddings, VLMs have shown notable success in bridging domain gaps. However, inherent differences naturally exist between modalities, which is known as modality gap. Our findings reveal that direct UDA with the presence of modality gap only transfers modality-invariant knowledge, leading to suboptimal target performance. To address this limitation, we propose a unified modality separation framework that accommodates both modality-specific and modality-invariant components. During training, different modality components are disentangled from VLM features then handled separately in a unified manner. At test time, modality-adaptive ensemble weights are automatically determined to maximize the synergy of different components. To evaluate instance-level modality characteristics, we design a modality discrepancy metric to categorize samples into modality-invariant, modality-specific, and uncertain ones. The modality-invariant samples are exploited to facilitate cross-modal alignment, while uncertain ones are annotated to enhance model capabilities. Building upon prompt tuning techniques, our methods achieve up to 9% performance gain with 9 times of computational efficiencies. Extensive experiments and analysis across various backbones, baselines, datasets and adaptation settings demonstrate the efficacy of our design.
>
---
#### [new 072] Single-Step Reconstruction-Free Anomaly Detection and Segmentation via Diffusion Models
- **分类: cs.CV; eess.IV; stat.ML; 62H35, 68T07, 62M40, 68T45; I.2.6; I.2.10; I.4.6; I.4.8; I.5.1; I.5.4**

- **简介: 该论文提出了一种基于扩散模型的无重建异常检测与分割方法（RADAR），解决了传统重建方法的计算效率低、模式鲁棒性差等问题，通过直接生成异常映射而非重建输入图像，提升了检测精度和实时性。**

- **链接: [http://arxiv.org/pdf/2508.04818v1](http://arxiv.org/pdf/2508.04818v1)**

> **作者:** Mehrdad Moradi; Marco Grasso; Bianca Maria Colosimo; Kamran Paynabar
>
> **备注:** 9 pages, 8 figures, 2 tables. Submitted to an IEEE conference
>
> **摘要:** Generative models have demonstrated significant success in anomaly detection and segmentation over the past decade. Recently, diffusion models have emerged as a powerful alternative, outperforming previous approaches such as GANs and VAEs. In typical diffusion-based anomaly detection, a model is trained on normal data, and during inference, anomalous images are perturbed to a predefined intermediate step in the forward diffusion process. The corresponding normal image is then reconstructed through iterative reverse sampling. However, reconstruction-based approaches present three major challenges: (1) the reconstruction process is computationally expensive due to multiple sampling steps, making real-time applications impractical; (2) for complex or subtle patterns, the reconstructed image may correspond to a different normal pattern rather than the original input; and (3) Choosing an appropriate intermediate noise level is challenging because it is application-dependent and often assumes prior knowledge of anomalies, an assumption that does not hold in unsupervised settings. We introduce Reconstruction-free Anomaly Detection with Attention-based diffusion models in Real-time (RADAR), which overcomes the limitations of reconstruction-based anomaly detection. Unlike current SOTA methods that reconstruct the input image, RADAR directly produces anomaly maps from the diffusion model, improving both detection accuracy and computational efficiency. We evaluate RADAR on real-world 3D-printed material and the MVTec-AD dataset. Our approach surpasses state-of-the-art diffusion-based and statistical machine learning models across all key metrics, including accuracy, precision, recall, and F1 score. Specifically, RADAR improves F1 score by 7% on MVTec-AD and 13% on the 3D-printed material dataset compared to the next best model. Code available at: https://github.com/mehrdadmoradi124/RADAR
>
---
#### [new 073] CT-GRAPH: Hierarchical Graph Attention Network for Anatomy-Guided CT Report Generation
- **分类: cs.CV**

- **简介: 该论文提出CT-GRAPH，用于生成基于解剖学的医学CT报告，通过构建图结构学习细粒度器官关系并融合预训练特征，解决了传统方法仅依赖全局特征导致的报告不准确问题，提升F1分数至7.9%。**

- **链接: [http://arxiv.org/pdf/2508.05375v1](http://arxiv.org/pdf/2508.05375v1)**

> **作者:** Hamza Kalisch; Fabian Hörst; Jens Kleesiek; Ken Herrmann; Constantin Seibold
>
> **摘要:** As medical imaging is central to diagnostic processes, automating the generation of radiology reports has become increasingly relevant to assist radiologists with their heavy workloads. Most current methods rely solely on global image features, failing to capture fine-grained organ relationships crucial for accurate reporting. To this end, we propose CT-GRAPH, a hierarchical graph attention network that explicitly models radiological knowledge by structuring anatomical regions into a graph, linking fine-grained organ features to coarser anatomical systems and a global patient context. Our method leverages pretrained 3D medical feature encoders to obtain global and organ-level features by utilizing anatomical masks. These features are further refined within the graph and then integrated into a large language model to generate detailed medical reports. We evaluate our approach for the task of report generation on the large-scale chest CT dataset CT-RATE. We provide an in-depth analysis of pretrained feature encoders for CT report generation and show that our method achieves a substantial improvement of absolute 7.9\% in F1 score over current state-of-the-art methods. The code is publicly available at https://github.com/hakal104/CT-GRAPH.
>
---
#### [new 074] From Detection to Correction: Backdoor-Resilient Face Recognition via Vision-Language Trigger Detection and Noise-Based Neutralization
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出了一种基于视觉语言模型的反后门攻击方案TrueBiometric，解决了如何准确检测和修正受污染图像的问题，通过多模型联合检测与噪声校正技术实现了100%准确率，显著提升了人脸识别系统的可靠性。**

- **链接: [http://arxiv.org/pdf/2508.05409v1](http://arxiv.org/pdf/2508.05409v1)**

> **作者:** Farah Wahida; M. A. P. Chamikara; Yashothara Shanmugarasa; Mohan Baruwal Chhetri; Thilina Ranbaduge; Ibrahim Khalil
>
> **备注:** 19 Pages, 24 Figures
>
> **摘要:** Biometric systems, such as face recognition systems powered by deep neural networks (DNNs), rely on large and highly sensitive datasets. Backdoor attacks can subvert these systems by manipulating the training process. By inserting a small trigger, such as a sticker, make-up, or patterned mask, into a few training images, an adversary can later present the same trigger during authentication to be falsely recognized as another individual, thereby gaining unauthorized access. Existing defense mechanisms against backdoor attacks still face challenges in precisely identifying and mitigating poisoned images without compromising data utility, which undermines the overall reliability of the system. We propose a novel and generalizable approach, TrueBiometric: Trustworthy Biometrics, which accurately detects poisoned images using a majority voting mechanism leveraging multiple state-of-the-art large vision language models. Once identified, poisoned samples are corrected using targeted and calibrated corrective noise. Our extensive empirical results demonstrate that TrueBiometric detects and corrects poisoned images with 100\% accuracy without compromising accuracy on clean images. Compared to existing state-of-the-art approaches, TrueBiometric offers a more practical, accurate, and effective solution for mitigating backdoor attacks in face recognition systems.
>
---
#### [new 075] SMOL-MapSeg: Show Me One Label
- **分类: cs.CV**

- **简介: 该论文提出多模态语义分割任务，解决历史地图信息提取问题，通过OND知识提示机制替代传统模型并微调，实现对历史地图中差异性概念的高效分类。**

- **链接: [http://arxiv.org/pdf/2508.05501v1](http://arxiv.org/pdf/2508.05501v1)**

> **作者:** Yunshuang Yuan; Frank Thiemann; Thorsten Dahms; Monika Sester
>
> **摘要:** Historical maps are valuable for studying changes to the Earth's surface. With the rise of deep learning, models like UNet have been used to extract information from these maps through semantic segmentation. Recently, pre-trained foundation models have shown strong performance across domains such as autonomous driving, medical imaging, and industrial inspection. However, they struggle with historical maps. These models are trained on modern or domain-specific images, where patterns can be tied to predefined concepts through common sense or expert knowledge. Historical maps lack such consistency -- similar concepts can appear in vastly different shapes and styles. To address this, we propose On-Need Declarative (OND) knowledge-based prompting, which introduces explicit prompts to guide the model on what patterns correspond to which concepts. This allows users to specify the target concept and pattern during inference (on-need inference). We implement this by replacing the prompt encoder of the foundation model SAM with our OND prompting mechanism and fine-tune it on historical maps. The resulting model is called SMOL-MapSeg (Show Me One Label). Experiments show that SMOL-MapSeg can accurately segment classes defined by OND knowledge. It can also adapt to unseen classes through few-shot fine-tuning. Additionally, it outperforms a UNet-based baseline in average segmentation performance.
>
---
#### [new 076] Toward Errorless Training ImageNet-1k
- **分类: cs.CV; cs.LG; 68T07**

- **简介: 该论文提出一种误差无损的ImageNet-1k训练方法，通过解决双标签问题（重复图像）提升模型性能，实现了98.3%的准确率。**

- **链接: [http://arxiv.org/pdf/2508.04941v1](http://arxiv.org/pdf/2508.04941v1)**

> **作者:** Bo Deng; Levi Heath
>
> **备注:** 14 pages, 2 figures
>
> **摘要:** In this paper, we describe a feedforward artificial neural network trained on the ImageNet 2012 contest dataset [7] with the new method of [5] to an accuracy rate of 98.3% with a 99.69 Top-1 rate, and an average of 285.9 labels that are perfectly classified over the 10 batch partitions of the dataset. The best performing model uses 322,430,160 parameters, with 4 decimal places precision. We conjecture that the reason our model does not achieve a 100% accuracy rate is due to a double-labeling problem, by which there are duplicate images in the dataset with different labels.
>
---
#### [new 077] Deformable Attention Graph Representation Learning for Histopathology Whole Slide Image Analysis
- **分类: cs.CV**

- **简介: 该论文提出了一种基于变形注意力的图神经网络框架，用于病理学WSI图像分类，解决了传统MIL方法对空间依赖性建模不足的问题，通过动态图结构与自适应注意力增强复杂空间结构的捕获能力。**

- **链接: [http://arxiv.org/pdf/2508.05382v1](http://arxiv.org/pdf/2508.05382v1)**

> **作者:** Mingxi Fu; Xitong Ling; Yuxuan Chen; Jiawen Li; fanglei fu; Huaitian Yuan; Tian Guan; Yonghong He; Lianghui Zhu
>
> **摘要:** Accurate classification of Whole Slide Images (WSIs) and Regions of Interest (ROIs) is a fundamental challenge in computational pathology. While mainstream approaches often adopt Multiple Instance Learning (MIL), they struggle to capture the spatial dependencies among tissue structures. Graph Neural Networks (GNNs) have emerged as a solution to model inter-instance relationships, yet most rely on static graph topologies and overlook the physical spatial positions of tissue patches. Moreover, conventional attention mechanisms lack specificity, limiting their ability to focus on structurally relevant regions. In this work, we propose a novel GNN framework with deformable attention for pathology image analysis. We construct a dynamic weighted directed graph based on patch features, where each node aggregates contextual information from its neighbors via attention-weighted edges. Specifically, we incorporate learnable spatial offsets informed by the real coordinates of each patch, enabling the model to adaptively attend to morphologically relevant regions across the slide. This design significantly enhances the contextual field while preserving spatial specificity. Our framework achieves state-of-the-art performance on four benchmark datasets (TCGA-COAD, BRACS, gastric intestinal metaplasia grading, and intestinal ROI classification), demonstrating the power of deformable attention in capturing complex spatial structures in WSIs and ROIs.
>
---
#### [new 078] Open-world Point Cloud Semantic Segmentation: A Human-in-the-loop Framework
- **分类: cs.CV; cs.GR**

- **简介: 该论文为开放世界点云语义分割任务，旨在解决传统方法依赖密集数据和大量标注的局限性。提出基于人类反馈的HOW-Seg框架，通过稀疏标注引导原型生成，结合层次化分清歧与密集CRF优化，实现高精度分割，达到85% mIoU性能。**

- **链接: [http://arxiv.org/pdf/2508.04962v1](http://arxiv.org/pdf/2508.04962v1)**

> **作者:** Peng Zhang; Songru Yang; Jinsheng Sun; Weiqing Li; Zhiyong Su
>
> **备注:** To be published in IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** Open-world point cloud semantic segmentation (OW-Seg) aims to predict point labels of both base and novel classes in real-world scenarios. However, existing methods rely on resource-intensive offline incremental learning or densely annotated support data, limiting their practicality. To address these limitations, we propose HOW-Seg, the first human-in-the-loop framework for OW-Seg. Specifically, we construct class prototypes, the fundamental segmentation units, directly on the query data, avoiding the prototype bias caused by intra-class distribution shifts between the support and query data. By leveraging sparse human annotations as guidance, HOW-Seg enables prototype-based segmentation for both base and novel classes. Considering the lack of granularity of initial prototypes, we introduce a hierarchical prototype disambiguation mechanism to refine ambiguous prototypes, which correspond to annotations of different classes. To further enrich contextual awareness, we employ a dense conditional random field (CRF) upon the refined prototypes to optimize their label assignments. Through iterative human feedback, HOW-Seg dynamically improves its predictions, achieving high-quality segmentation for both base and novel classes. Experiments demonstrate that with sparse annotations (e.g., one-novel-class-one-click), HOW-Seg matches or surpasses the state-of-the-art generalized few-shot segmentation (GFS-Seg) method under the 5-shot setting. When using advanced backbones (e.g., Stratified Transformer) and denser annotations (e.g., 10 clicks per sub-scene), HOW-Seg achieves 85.27% mIoU on S3DIS and 66.37% mIoU on ScanNetv2, significantly outperforming alternatives.
>
---
#### [new 079] A Study of Gender Classification Techniques Based on Iris Images: A Deep Survey and Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文旨在探讨基于虹膜图像的性别分类技术，研究其方法论与实际应用，解决了如何利用虹膜特征进行身份识别的问题，通过分析现有方法并指出研究空白，为性别分类技术的发展提供参考。**

- **链接: [http://arxiv.org/pdf/2508.05246v1](http://arxiv.org/pdf/2508.05246v1)**

> **作者:** Basna Mohammed Salih Hasan; Ramadhan J. Mstafa
>
> **备注:** 13 Pages, 8 Figures, 1 Table
>
> **摘要:** Gender classification is attractive in a range of applications, including surveillance and monitoring, corporate profiling, and human-computer interaction. Individuals' identities may be gleaned from information about their gender, which is a kind of soft biometric.Over the years, several methods for determining a person's gender have been devised. Some of the most well-known ones are based on physical characteristics like face, fingerprint, palmprint, DNA, ears, gait, and iris. On the other hand, facial features account for the vast majority of gender classification methods. Also, the iris is a significant biometric trait because the iris, according to research, remains basically constant during an individual's life. Besides that, the iris is externally visible and is non-invasive to the user, which is important for practical applications. Furthermore, there are already high-quality methods for segmenting and encoding iris images, and the current methods facilitate selecting and extracting attribute vectors from iris textures. This study discusses several approaches to determining gender. The previous works of literature are briefly reviewed. Additionally, there are a variety of methodologies for different steps of gender classification. This study provides researchers with knowledge and analysis of the existing gender classification approaches. Also, it will assist researchers who are interested in this specific area, as well as highlight the gaps and challenges in the field, and finally provide suggestions and future paths for improvement.
>
---
#### [new 080] Test-Time Adaptation for Video Highlight Detection Using Meta-Auxiliary Learning and Cross-Modality Hallucinations
- **分类: cs.CV**

- **简介: 该论文提出了一种基于元辅助学习和跨模态假想的视频突出检测框架，旨在解决传统固定模型在不同测试视频中的泛化性不足问题。通过动态调整模型并联合优化辅助任务与主任务，显著提升了检测性能。**

- **链接: [http://arxiv.org/pdf/2508.04924v1](http://arxiv.org/pdf/2508.04924v1)**

> **作者:** Zahidul Islam; Sujoy Paul; Mrigank Rochan
>
> **摘要:** Existing video highlight detection methods, although advanced, struggle to generalize well to all test videos. These methods typically employ a generic highlight detection model for each test video, which is suboptimal as it fails to account for the unique characteristics and variations of individual test videos. Such fixed models do not adapt to the diverse content, styles, or audio and visual qualities present in new, unseen test videos, leading to reduced highlight detection performance. In this paper, we propose Highlight-TTA, a test-time adaptation framework for video highlight detection that addresses this limitation by dynamically adapting the model during testing to better align with the specific characteristics of each test video, thereby improving generalization and highlight detection performance. Highlight-TTA is jointly optimized with an auxiliary task, cross-modality hallucinations, alongside the primary highlight detection task. We utilize a meta-auxiliary training scheme to enable effective adaptation through the auxiliary task while enhancing the primary task. During testing, we adapt the trained model using the auxiliary task on the test video to further enhance its highlight detection performance. Extensive experiments with three state-of-the-art highlight detection models and three benchmark datasets show that the introduction of Highlight-TTA to these models improves their performance, yielding superior results.
>
---
#### [new 081] Dual-Stream Attention with Multi-Modal Queries for Object Detection in Transportation Applications
- **分类: cs.CV**

- **简介: 该论文旨在解决传统Transformer目标检测中的遮挡、细粒度定位和计算效率问题，提出DAMM框架通过多模态查询和双流交叉注意力提升性能，优化了场景覆盖和定位精度。**

- **链接: [http://arxiv.org/pdf/2508.04868v1](http://arxiv.org/pdf/2508.04868v1)**

> **作者:** Noreen Anwar; Guillaume-Alexandre Bilodeau; Wassim Bouachir
>
> **备注:** 10 pages
>
> **摘要:** Transformer-based object detectors often struggle with occlusions, fine-grained localization, and computational inefficiency caused by fixed queries and dense attention. We propose DAMM, Dual-stream Attention with Multi-Modal queries, a novel framework introducing both query adaptation and structured cross-attention for improved accuracy and efficiency. DAMM capitalizes on three types of queries: appearance-based queries from vision-language models, positional queries using polygonal embeddings, and random learned queries for general scene coverage. Furthermore, a dual-stream cross-attention module separately refines semantic and spatial features, boosting localization precision in cluttered scenes. We evaluated DAMM on four challenging benchmarks, and it achieved state-of-the-art performance in average precision (AP) and recall, demonstrating the effectiveness of multi-modal query adaptation and dual-stream attention. Source code is at: \href{https://github.com/DET-LIP/DAMM}{GitHub}.
>
---
#### [new 082] Automatic Image Colorization with Convolutional Neural Networks and Generative Adversarial Networks
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文探讨了自动图像颜色化任务，旨在解决高维自由度和多模态提示语的挑战，通过CNN与GAN结合分类与对抗学习方法，优化模型参数并进行场景特征建模。**

- **链接: [http://arxiv.org/pdf/2508.05068v1](http://arxiv.org/pdf/2508.05068v1)**

> **作者:** Ruiyu Li; Changyuan Qiu; Hangrui Cao; Qihan Ren; Yuqing Qiu
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Image colorization, the task of adding colors to grayscale images, has been the focus of significant research efforts in computer vision in recent years for its various application areas such as color restoration and automatic animation colorization [15, 1]. The colorization problem is challenging as it is highly ill-posed with two out of three image dimensions lost, resulting in large degrees of freedom. However, semantics of the scene as well as the surface texture could provide important cues for colors: the sky is typically blue, the clouds are typically white and the grass is typically green, and there are huge amounts of training data available for learning such priors since any colored image could serve as a training data point [20]. Colorization is initially formulated as a regression task[5], which ignores the multi-modal nature of color prediction. In this project, we explore automatic image colorization via classification and adversarial learning. We will build our models on prior works, apply modifications for our specific scenario and make comparisons.
>
---
#### [new 083] Skin-SOAP: A Weakly Supervised Framework for Generating Structured SOAP Notes
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文旨在解决皮肤癌患者SOAP笔记生成中的低效与高成本问题，提出皮肤-SOAP框架通过弱监督学习从图像和文本输入生成结构化笔记，有效降低医疗人力负担并提升临床可及性。**

- **链接: [http://arxiv.org/pdf/2508.05019v1](http://arxiv.org/pdf/2508.05019v1)**

> **作者:** Sadia Kamal; Tim Oates; Joy Wan
>
> **备注:** Accepted to IJCAI 2025 Workshops. arXiv admin note: substantial text overlap with arXiv:2506.10328
>
> **摘要:** Skin carcinoma is the most prevalent form of cancer globally, accounting for over $8 billion in annual healthcare expenditures. Early diagnosis, accurate and timely treatment are critical to improving patient survival rates. In clinical settings, physicians document patient visits using detailed SOAP (Subjective, Objective, Assessment, and Plan) notes. However, manually generating these notes is labor-intensive and contributes to clinician burnout. In this work, we propose skin-SOAP, a weakly supervised multimodal framework to generate clinically structured SOAP notes from limited inputs, including lesion images and sparse clinical text. Our approach reduces reliance on manual annotations, enabling scalable, clinically grounded documentation while alleviating clinician burden and reducing the need for large annotated data. Our method achieves performance comparable to GPT-4o, Claude, and DeepSeek Janus Pro across key clinical relevance metrics. To evaluate this clinical relevance, we introduce two novel metrics MedConceptEval and Clinical Coherence Score (CCS) which assess semantic alignment with expert medical concepts and input features, respectively.
>
---
#### [new 084] A deep learning approach to track eye movements based on events
- **分类: cs.CV; 68T05, 68T07; I.2.10; I.5.1; I.4.8; J.4**

- **简介: 该论文旨在开发基于深度学习的精确眼运动态追踪算法，解决高速人眼移动（>300°/s）下的实时定位问题，通过CNN-LSTM模型实现81%准确率，并探索Layer-wise Relevance Propagation提升模型可解释性与预测性能。**

- **链接: [http://arxiv.org/pdf/2508.04827v1](http://arxiv.org/pdf/2508.04827v1)**

> **作者:** Chirag Seth; Divya Naiken; Keyan Lin
>
> **摘要:** This research project addresses the challenge of accurately tracking eye movements during specific events by leveraging previous research. Given the rapid movements of human eyes, which can reach speeds of 300{\deg}/s, precise eye tracking typically requires expensive and high-speed cameras. Our primary objective is to locate the eye center position (x, y) using inputs from an event camera. Eye movement analysis has extensive applications in consumer electronics, especially in VR and AR product development. Therefore, our ultimate goal is to develop an interpretable and cost-effective algorithm using deep learning methods to predict human attention, thereby improving device comfort and enhancing overall user experience. To achieve this goal, we explored various approaches, with the CNN\_LSTM model proving most effective, achieving approximately 81\% accuracy. Additionally, we propose future work focusing on Layer-wise Relevance Propagation (LRP) to further enhance the model's interpretability and predictive performance.
>
---
#### [new 085] TRKT: Weakly Supervised Dynamic Scene Graph Generation with Temporal-enhanced Relation-aware Knowledge Transferring
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TRKT方法，用于弱监督动态场景图生成（WS-DSGG），解决依赖静态对象检测器的精度不足问题，通过关系与运动增强的知识融合提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.04943v1](http://arxiv.org/pdf/2508.04943v1)**

> **作者:** Zhu Xu; Ting Lei; Zhimin Li; Guan Wang; Qingchao Chen; Yuxin Peng; Yang liu
>
> **摘要:** Dynamic Scene Graph Generation (DSGG) aims to create a scene graph for each video frame by detecting objects and predicting their relationships. Weakly Supervised DSGG (WS-DSGG) reduces annotation workload by using an unlocalized scene graph from a single frame per video for training. Existing WS-DSGG methods depend on an off-the-shelf external object detector to generate pseudo labels for subsequent DSGG training. However, detectors trained on static, object-centric images struggle in dynamic, relation-aware scenarios required for DSGG, leading to inaccurate localization and low-confidence proposals. To address the challenges posed by external object detectors in WS-DSGG, we propose a Temporal-enhanced Relation-aware Knowledge Transferring (TRKT) method, which leverages knowledge to enhance detection in relation-aware dynamic scenarios. TRKT is built on two key components:(1)Relation-aware knowledge mining: we first employ object and relation class decoders that generate category-specific attention maps to highlight both object regions and interactive areas. Then we propose an Inter-frame Attention Augmentation strategy that exploits optical flow for neighboring frames to enhance the attention maps, making them motion-aware and robust to motion blur. This step yields relation- and motion-aware knowledge mining for WS-DSGG. (2) we introduce a Dual-stream Fusion Module that integrates category-specific attention maps into external detections to refine object localization and boost confidence scores for object proposals. Extensive experiments demonstrate that TRKT achieves state-of-the-art performance on Action Genome dataset. Our code is avaliable at https://github.com/XZPKU/TRKT.git.
>
---
#### [new 086] Modeling Rapid Contextual Learning in the Visual Cortex with Fast-Weight Deep Autoencoder Networks
- **分类: cs.CV**

- **简介: 该论文探讨了视觉皮层中通过自编码器模型模拟快速上下文学习的机制，提出利用ViT和LoRA实现快速权重调整的方法，解决了基于传统深度学习架构难以捕捉全局上下文特征的问题。**

- **链接: [http://arxiv.org/pdf/2508.04988v1](http://arxiv.org/pdf/2508.04988v1)**

> **作者:** Yue Li; Weifan Wang; Tai Sing Lee
>
> **摘要:** Recent neurophysiological studies have revealed that the early visual cortex can rapidly learn global image context, as evidenced by a sparsification of population responses and a reduction in mean activity when exposed to familiar versus novel image contexts. This phenomenon has been attributed primarily to local recurrent interactions, rather than changes in feedforward or feedback pathways, supported by both empirical findings and circuit-level modeling. Recurrent neural circuits capable of simulating these effects have been shown to reshape the geometry of neural manifolds, enhancing robustness and invariance to irrelevant variations. In this study, we employ a Vision Transformer (ViT)-based autoencoder to investigate, from a functional perspective, how familiarity training can induce sensitivity to global context in the early layers of a deep neural network. We hypothesize that rapid learning operates via fast weights, which encode transient or short-term memory traces, and we explore the use of Low-Rank Adaptation (LoRA) to implement such fast weights within each Transformer layer. Our results show that (1) The proposed ViT-based autoencoder's self-attention circuit performs a manifold transform similar to a neural circuit model of the familiarity effect. (2) Familiarity training aligns latent representations in early layers with those in the top layer that contains global context information. (3) Familiarity training broadens the self-attention scope within the remembered image context. (4) These effects are significantly amplified by LoRA-based fast weights. Together, these findings suggest that familiarity training introduces global sensitivity to earlier layers in a hierarchical network, and that a hybrid fast-and-slow weight architecture may provide a viable computational model for studying rapid global context learning in the brain.
>
---
#### [new 087] UGOD: Uncertainty-Guided Differentiable Opacity and Soft Dropout for Enhanced Sparse-View 3DGS
- **分类: cs.CV; cs.AI; I.4.8; I.2.10; I.5.1**

- **简介: 该论文研究了稀疏视角下的3DGS（3D Gaussian Splatting）任务，旨在解决传统方法对Gaussians权重不均导致的过拟合问题。提出通过自适应权重调整和不确定性转换为连续概率的方法，改进了渲染质量并提升了稀疏合成效果。**

- **链接: [http://arxiv.org/pdf/2508.04968v1](http://arxiv.org/pdf/2508.04968v1)**

> **作者:** Zhihao Guo; Peng Wang; Zidong Chen; Xiangyu Kong; Yan Lyu; Guanyu Gao; Liangxiu Han
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** 3D Gaussian Splatting (3DGS) has become a competitive approach for novel view synthesis (NVS) due to its advanced rendering efficiency through 3D Gaussian projection and blending. However, Gaussians are treated equally weighted for rendering in most 3DGS methods, making them prone to overfitting, which is particularly the case in sparse-view scenarios. To address this, we investigate how adaptive weighting of Gaussians affects rendering quality, which is characterised by learned uncertainties proposed. This learned uncertainty serves two key purposes: first, it guides the differentiable update of Gaussian opacity while preserving the 3DGS pipeline integrity; second, the uncertainty undergoes soft differentiable dropout regularisation, which strategically transforms the original uncertainty into continuous drop probabilities that govern the final Gaussian projection and blending process for rendering. Extensive experimental results over widely adopted datasets demonstrate that our method outperforms rivals in sparse-view 3D synthesis, achieving higher quality reconstruction with fewer Gaussians in most datasets compared to existing sparse-view approaches, e.g., compared to DropGaussian, our method achieves 3.27\% PSNR improvements on the MipNeRF 360 dataset.
>
---
#### [new 088] X-MoGen: Unified Motion Generation across Humans and Animals
- **分类: cs.CV**

- **简介: 该论文提出了一种跨物种文本驱动运动生成框架X-MoGen，解决跨物种运动生成中的形态差异问题，通过联合建模和多阶段架构（如条件图变分自编码器与掩码运动模型）实现对人和动物的统一表示，构建了UniMo4D数据集以支持跨物种通用性训练。**

- **链接: [http://arxiv.org/pdf/2508.05162v1](http://arxiv.org/pdf/2508.05162v1)**

> **作者:** Xuan Wang; Kai Ruan; Liyang Qian; Zhizhi Guo; Chang Su; Gaoang Wang
>
> **摘要:** Text-driven motion generation has attracted increasing attention due to its broad applications in virtual reality, animation, and robotics. While existing methods typically model human and animal motion separately, a joint cross-species approach offers key advantages, such as a unified representation and improved generalization. However, morphological differences across species remain a key challenge, often compromising motion plausibility. To address this, we propose \textbf{X-MoGen}, the first unified framework for cross-species text-driven motion generation covering both humans and animals. X-MoGen adopts a two-stage architecture. First, a conditional graph variational autoencoder learns canonical T-pose priors, while an autoencoder encodes motion into a shared latent space regularized by morphological loss. In the second stage, we perform masked motion modeling to generate motion embeddings conditioned on textual descriptions. During training, a morphological consistency module is employed to promote skeletal plausibility across species. To support unified modeling, we construct \textbf{UniMo4D}, a large-scale dataset of 115 species and 119k motion sequences, which integrates human and animal motions under a shared skeletal topology for joint training. Extensive experiments on UniMo4D demonstrate that X-MoGen outperforms state-of-the-art methods on both seen and unseen species.
>
---
#### [new 089] RetinexDual: Retinex-based Dual Nature Approach for Generalized Ultra-High-Definition Image Restoration
- **分类: cs.CV**

- **简介: 本论文提出基于Retinex的双任务图像恢复方法，解决高分辨率图像中反射与光照干扰问题，通过两子网络协同处理。**

- **链接: [http://arxiv.org/pdf/2508.04797v1](http://arxiv.org/pdf/2508.04797v1)**

> **作者:** Mohab Kishawy; Ali Abdellatif Hussein; Jun Chen
>
> **摘要:** Advancements in image sensing have elevated the importance of Ultra-High-Definition Image Restoration (UHD IR). Traditional methods, such as extreme downsampling or transformation from the spatial to the frequency domain, encounter significant drawbacks: downsampling induces irreversible information loss in UHD images, while our frequency analysis reveals that pure frequency-domain approaches are ineffective for spatially confined image artifacts, primarily due to the loss of degradation locality. To overcome these limitations, we present RetinexDual, a novel Retinex theory-based framework designed for generalized UHD IR tasks. RetinexDual leverages two complementary sub-networks: the Scale-Attentive maMBA (SAMBA) and the Frequency Illumination Adaptor (FIA). SAMBA, responsible for correcting the reflectance component, utilizes a coarse-to-fine mechanism to overcome the causal modeling of mamba, which effectively reduces artifacts and restores intricate details. On the other hand, FIA ensures precise correction of color and illumination distortions by operating in the frequency domain and leveraging the global context provided by it. Evaluating RetinexDual on four UHD IR tasks, namely deraining, deblurring, dehazing, and Low-Light Image Enhancement (LLIE), shows that it outperforms recent methods qualitatively and quantitatively. Ablation studies demonstrate the importance of employing distinct designs for each branch in RetinexDual, as well as the effectiveness of its various components.
>
---
#### [new 090] GAP: Gaussianize Any Point Clouds with Text Guidance
- **分类: cs.CV**

- **简介: 该论文属于**3D Gaussians生成任务**，旨在解决从颜色点云到高斯图的转换难题。通过设计多视角优化框架、引入表面锚点约束和漫反射填充策略，结合深度感知图像扩散模型，实现了对复杂场景的高效高精度生成。**

- **链接: [http://arxiv.org/pdf/2508.05631v1](http://arxiv.org/pdf/2508.05631v1)**

> **作者:** Weiqi Zhang; Junsheng Zhou; Haotian Geng; Wenyuan Zhang; Yu-Shen Liu
>
> **备注:** ICCV 2025. Project page: https://weiqi-zhang.github.io/GAP
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated its advantages in achieving fast and high-quality rendering. As point clouds serve as a widely-used and easily accessible form of 3D representation, bridging the gap between point clouds and Gaussians becomes increasingly important. Recent studies have explored how to convert the colored points into Gaussians, but directly generating Gaussians from colorless 3D point clouds remains an unsolved challenge. In this paper, we propose GAP, a novel approach that gaussianizes raw point clouds into high-fidelity 3D Gaussians with text guidance. Our key idea is to design a multi-view optimization framework that leverages a depth-aware image diffusion model to synthesize consistent appearances across different viewpoints. To ensure geometric accuracy, we introduce a surface-anchoring mechanism that effectively constrains Gaussians to lie on the surfaces of 3D shapes during optimization. Furthermore, GAP incorporates a diffuse-based inpainting strategy that specifically targets at completing hard-to-observe regions. We evaluate GAP on the Point-to-Gaussian generation task across varying complexity levels, from synthetic point clouds to challenging real-world scans, and even large-scale scenes. Project Page: https://weiqi-zhang.github.io/GAP.
>
---
#### [new 091] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗影像报告生成任务，旨在减少放射科医生工作量并提升报告质量。通过两阶段训练：1）利用临床背景引导特征提取；2）整合患者知识实现粗-细解码，解决了传统方法忽略患者历史信息的问题。**

- **链接: [http://arxiv.org/pdf/2508.05353v1](http://arxiv.org/pdf/2508.05353v1)**

> **作者:** Kang Liu; Zhuoqi Ma; Zikang Fang; Yunan Li; Kun Xie; Qiguang Miao
>
> **摘要:** Chest X-ray report generation aims to reduce radiologists' workload by automatically producing high-quality preliminary reports. A critical yet underexplored aspect of this task is the effective use of patient-specific prior knowledge -- including clinical context (e.g., symptoms, medical history) and the most recent prior image -- which radiologists routinely rely on for diagnostic reasoning. Most existing methods generate reports from single images, neglecting this essential prior information and thus failing to capture diagnostic intent or disease progression. To bridge this gap, we propose PriorRG, a novel chest X-ray report generation framework that emulates real-world clinical workflows via a two-stage training pipeline. In Stage 1, we introduce a prior-guided contrastive pre-training scheme that leverages clinical context to guide spatiotemporal feature extraction, allowing the model to align more closely with the intrinsic spatiotemporal semantics in radiology reports. In Stage 2, we present a prior-aware coarse-to-fine decoding for report generation that progressively integrates patient-specific prior knowledge with the vision encoder's hidden states. This decoding allows the model to align with diagnostic focus and track disease progression, thereby enhancing the clinical accuracy and fluency of the generated reports. Extensive experiments on MIMIC-CXR and MIMIC-ABN datasets demonstrate that PriorRG outperforms state-of-the-art methods, achieving a 3.6% BLEU-4 and 3.8% F1 score improvement on MIMIC-CXR, and a 5.9% BLEU-1 gain on MIMIC-ABN. Code and checkpoints will be released upon acceptance.
>
---
#### [new 092] Physical Adversarial Camouflage through Gradient Calibration and Regularization
- **分类: cs.CV**

- **简介: 该论文旨在解决物理欺骗伪装技术中的梯度一致性与多角度冲突问题，通过梯度校准与正交化方法增强模型稳定性，实验验证其在不同场景下的攻击成功率提升显著。**

- **链接: [http://arxiv.org/pdf/2508.05414v1](http://arxiv.org/pdf/2508.05414v1)**

> **作者:** Jiawei Liang; Siyuan Liang; Jianjie Huang; Chenxi Si; Ming Zhang; Xiaochun Cao
>
> **备注:** Accepted to IJCAI 2025
>
> **摘要:** The advancement of deep object detectors has greatly affected safety-critical fields like autonomous driving. However, physical adversarial camouflage poses a significant security risk by altering object textures to deceive detectors. Existing techniques struggle with variable physical environments, facing two main challenges: 1) inconsistent sampling point densities across distances hinder the gradient optimization from ensuring local continuity, and 2) updating texture gradients from multiple angles causes conflicts, reducing optimization stability and attack effectiveness. To address these issues, we propose a novel adversarial camouflage framework based on gradient optimization. First, we introduce a gradient calibration strategy, which ensures consistent gradient updates across distances by propagating gradients from sparsely to unsampled texture points. Additionally, we develop a gradient decorrelation method, which prioritizes and orthogonalizes gradients based on loss values, enhancing stability and effectiveness in multi-angle optimization by eliminating redundant or conflicting updates. Extensive experimental results on various detection models, angles and distances show that our method significantly exceeds the state of the art, with an average increase in attack success rate (ASR) of 13.46% across distances and 11.03% across angles. Furthermore, empirical evaluation in real-world scenarios highlights the need for more robust system design.
>
---
#### [new 093] A Novel Image Similarity Metric for Scene Composition Structure
- **分类: cs.CV; cs.IT; math.IT**

- **简介: 该论文旨在开发一种新型图像相似度指标SCSSIM，解决传统方法在捕捉场景构建结构（SCS）完整性方面的局限性，通过统计学方法量化非对象结构关系，验证其对非建模与建模扭曲的适应性，为生成式AI的结构评估提供可靠指标。**

- **链接: [http://arxiv.org/pdf/2508.05037v1](http://arxiv.org/pdf/2508.05037v1)**

> **作者:** Md Redwanul Haque; Manzur Murshed; Manoranjan Paul; Tsz-Kwan Lee
>
> **备注:** IEEE ICIP 2025
>
> **摘要:** The rapid advancement of generative AI models necessitates novel methods for evaluating image quality that extend beyond human perception. A critical concern for these models is the preservation of an image's underlying Scene Composition Structure (SCS), which defines the geometric relationships among objects and the background, their relative positions, sizes, orientations, etc. Maintaining SCS integrity is paramount for ensuring faithful and structurally accurate GenAI outputs. Traditional image similarity metrics often fall short in assessing SCS. Pixel-level approaches are overly sensitive to minor visual noise, while perception-based metrics prioritize human aesthetic appeal, neither adequately capturing structural fidelity. Furthermore, recent neural-network-based metrics introduce training overheads and potential generalization issues. We introduce the SCS Similarity Index Measure (SCSSIM), a novel, analytical, and training-free metric that quantifies SCS preservation by exploiting statistical measures derived from the Cuboidal hierarchical partitioning of images, robustly capturing non-object-based structural relationships. Our experiments demonstrate SCSSIM's high invariance to non-compositional distortions, accurately reflecting unchanged SCS. Conversely, it shows a strong monotonic decrease for compositional distortions, precisely indicating when SCS has been altered. Compared to existing metrics, SCSSIM exhibits superior properties for structural evaluation, making it an invaluable tool for developing and evaluating generative models, ensuring the integrity of scene composition.
>
---
#### [new 094] CSRAP: Enhanced Canvas Attention Scheduling for Real-Time Mission Critical Perception
- **分类: cs.CV**

- **简介: 该论文属于实时感知系统任务，解决高分辨率目标检测在低功耗边缘计算下的资源约束问题。通过扩展画布注意力调度（variable canvas size + frame rate selection），优化了推理效率与精度平衡，验证了其在Waymo Open Dataset上的有效性。**

- **链接: [http://arxiv.org/pdf/2508.04976v1](http://arxiv.org/pdf/2508.04976v1)**

> **作者:** Md Iftekharul Islam Sakib; Yigong Hu; Tarek Abdelzaher
>
> **摘要:** Real-time perception on edge platforms faces a core challenge: executing high-resolution object detection under stringent latency constraints on limited computing resources. Canvas-based attention scheduling was proposed in earlier work as a mechanism to reduce the resource demands of perception subsystems. It consolidates areas of interest in an input data frame onto a smaller area, called a canvas frame, that can be processed at the requisite frame rate. This paper extends prior canvas-based attention scheduling literature by (i) allowing for variable-size canvas frames and (ii) employing selectable canvas frame rates that may depart from the original data frame rate. We evaluate our solution by running YOLOv11, as the perception module, on an NVIDIA Jetson Orin Nano to inspect video frames from the Waymo Open Dataset. Our results show that the additional degrees of freedom improve the attainable quality/cost trade-offs, thereby allowing for a consistently higher mean average precision (mAP) and recall with respect to the state of the art.
>
---
#### [new 095] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文旨在解释视觉语言模型中相似性，解决传统方法仅捕捉第一阶贡献的问题，提出基于游戏理论的第二阶交互解释（FIxLIP），通过扩展Saliency Maps并验证其有效性，提升跨模态模型的解释精度与实用性。**

- **链接: [http://arxiv.org/pdf/2508.05430v1](http://arxiv.org/pdf/2508.05430v1)**

> **作者:** Hubert Baniecki; Maximilian Muschalik; Fabian Fumagalli; Barbara Hammer; Eyke Hüllermeier; Przemyslaw Biecek
>
> **备注:** Preprint
>
> **摘要:** Language-image pre-training (LIP) enables the development of vision-language models capable of zero-shot classification, localization, multimodal retrieval, and semantic understanding. Various explanation methods have been proposed to visualize the importance of input image-text pairs on the model's similarity outputs. However, popular saliency maps are limited by capturing only first-order attributions, overlooking the complex cross-modal interactions intrinsic to such encoders. We introduce faithful interaction explanations of LIP models (FIxLIP) as a unified approach to decomposing the similarity in vision-language encoders. FIxLIP is rooted in game theory, where we analyze how using the weighted Banzhaf interaction index offers greater flexibility and improves computational efficiency over the Shapley interaction quantification framework. From a practical perspective, we propose how to naturally extend explanation evaluation metrics, like the pointing game and area between the insertion/deletion curves, to second-order interaction explanations. Experiments on MS COCO and ImageNet-1k benchmarks validate that second-order methods like FIxLIP outperform first-order attribution methods. Beyond delivering high-quality explanations, we demonstrate the utility of FIxLIP in comparing different models like CLIP vs. SigLIP-2 and ViT-B/32 vs. ViT-L/16.
>
---
#### [new 096] Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression
- **分类: cs.CV**

- **简介: 该论文提出了一种单步扩散图像压缩模型SODEC，旨在解决多步采样延迟和过拟合问题。通过预训练VAE生成丰富潜在图并替代迭代去噪，结合fidelity引导模块和率衰减训练策略，显著提升了压缩性能和速度。**

- **链接: [http://arxiv.org/pdf/2508.04979v1](http://arxiv.org/pdf/2508.04979v1)**

> **作者:** Zheng Chen; Mingde Zhou; Jinpei Guo; Jiale Yuan; Yifei Ji; Yulun Zhang
>
> **备注:** Code is available at: https://github.com/zhengchen1999/SODEC
>
> **摘要:** Diffusion-based image compression has demonstrated impressive perceptual performance. However, it suffers from two critical drawbacks: (1) excessive decoding latency due to multi-step sampling, and (2) poor fidelity resulting from over-reliance on generative priors. To address these issues, we propose SODEC, a novel single-step diffusion image compression model. We argue that in image compression, a sufficiently informative latent renders multi-step refinement unnecessary. Based on this insight, we leverage a pre-trained VAE-based model to produce latents with rich information, and replace the iterative denoising process with a single-step decoding. Meanwhile, to improve fidelity, we introduce the fidelity guidance module, encouraging output that is faithful to the original image. Furthermore, we design the rate annealing training strategy to enable effective training under extremely low bitrates. Extensive experiments show that SODEC significantly outperforms existing methods, achieving superior rate-distortion-perception performance. Moreover, compared to previous diffusion-based compression models, SODEC improves decoding speed by more than 20$\times$. Code is released at: https://github.com/zhengchen1999/SODEC.
>
---
#### [new 097] AHDMIL: Asymmetric Hierarchical Distillation Multi-Instance Learning for Fast and Accurate Whole-Slide Image Classification
- **分类: cs.CV**

- **简介: 该论文旨在解决多实例学习在病理学图像分类中的高推理成本问题，提出AHDMIL框架通过两步训练消除无关像素并结合CKA分类器提升效率，实现快速准确的全息图像分类。**

- **链接: [http://arxiv.org/pdf/2508.05114v1](http://arxiv.org/pdf/2508.05114v1)**

> **作者:** Jiuyang Dong; Jiahan Li; Junjun Jiang; Kui Jiang; Yongbing Zhang
>
> **摘要:** Although multi-instance learning (MIL) has succeeded in pathological image classification, it faces the challenge of high inference costs due to the need to process thousands of patches from each gigapixel whole slide image (WSI). To address this, we propose AHDMIL, an Asymmetric Hierarchical Distillation Multi-Instance Learning framework that enables fast and accurate classification by eliminating irrelevant patches through a two-step training process. AHDMIL comprises two key components: the Dynamic Multi-Instance Network (DMIN), which operates on high-resolution WSIs, and the Dual-Branch Lightweight Instance Pre-screening Network (DB-LIPN), which analyzes corresponding low-resolution counterparts. In the first step, self-distillation (SD), DMIN is trained for WSI classification while generating per-instance attention scores to identify irrelevant patches. These scores guide the second step, asymmetric distillation (AD), where DB-LIPN learns to predict the relevance of each low-resolution patch. The relevant patches predicted by DB-LIPN have spatial correspondence with patches in high-resolution WSIs, which are used for fine-tuning and efficient inference of DMIN. In addition, we design the first Chebyshev-polynomial-based Kolmogorov-Arnold (CKA) classifier in computational pathology, which improves classification performance through learnable activation layers. Extensive experiments on four public datasets demonstrate that AHDMIL consistently outperforms previous state-of-the-art methods in both classification performance and inference speed. For example, on the Camelyon16 dataset, it achieves a relative improvement of 5.3% in accuracy and accelerates inference by 1.2.times. Across all datasets, area under the curve (AUC), accuracy, f1 score, and brier score show consistent gains, with average inference speedups ranging from 1.2 to 2.1 times. The code is available.
>
---
#### [new 098] DART: Dual Adaptive Refinement Transfer for Open-Vocabulary Multi-Label Recognition
- **分类: cs.CV**

- **简介: 该论文提出DART框架，解决Open-Vocabulary Multi-Label Recognition（OV-MLR）中的VLP模型难以处理细粒度定位和结构化类间推理的问题。通过双模块融合（ARM与ATM），结合弱监督下的自适应内部分配和类间迁移，实现了新状态的突破性性能。**

- **链接: [http://arxiv.org/pdf/2508.05585v1](http://arxiv.org/pdf/2508.05585v1)**

> **作者:** Haijing Liu; Tao Pu; Hefeng Wu; Keze Wang; Liang Lin
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Open-Vocabulary Multi-Label Recognition (OV-MLR) aims to identify multiple seen and unseen object categories within an image, requiring both precise intra-class localization to pinpoint objects and effective inter-class reasoning to model complex category dependencies. While Vision-Language Pre-training (VLP) models offer a strong open-vocabulary foundation, they often struggle with fine-grained localization under weak supervision and typically fail to explicitly leverage structured relational knowledge beyond basic semantics, limiting performance especially for unseen classes. To overcome these limitations, we propose the Dual Adaptive Refinement Transfer (DART) framework. DART enhances a frozen VLP backbone via two synergistic adaptive modules. For intra-class refinement, an Adaptive Refinement Module (ARM) refines patch features adaptively, coupled with a novel Weakly Supervised Patch Selecting (WPS) loss that enables discriminative localization using only image-level labels. Concurrently, for inter-class transfer, an Adaptive Transfer Module (ATM) leverages a Class Relationship Graph (CRG), constructed using structured knowledge mined from a Large Language Model (LLM), and employs graph attention network to adaptively transfer relational information between class representations. DART is the first framework, to our knowledge, to explicitly integrate external LLM-derived relational knowledge for adaptive inter-class transfer while simultaneously performing adaptive intra-class refinement under weak supervision for OV-MLR. Extensive experiments on challenging benchmarks demonstrate that our DART achieves new state-of-the-art performance, validating its effectiveness.
>
---
#### [new 099] How and Why: Taming Flow Matching for Unsupervised Anomaly Detection and Localization
- **分类: cs.CV**

- **简介: 该论文提出了一种无监督异常检测与定位的新方法，通过Time-Reversed Flow Matching（rFM）解决传统方法的模型表达性局限，建立了非概率路径重构机制并构建了"退化潜在井"，实现了动态轨迹控制与异常样本分离，为复杂数据的高效处理提供了理论框架。**

- **链接: [http://arxiv.org/pdf/2508.05461v1](http://arxiv.org/pdf/2508.05461v1)**

> **作者:** Liangwei Li; Lin Liu; Juanxiu Liu; Jing Zhang; Ruqian Hao; Xiaohui Du
>
> **摘要:** We propose a new paradigm for unsupervised anomaly detection and localization using Flow Matching (FM), which fundamentally addresses the model expressivity limitations of conventional flow-based methods. To this end, we formalize the concept of time-reversed Flow Matching (rFM) as a vector field regression along a predefined probability path to transform unknown data distributions into standard Gaussian. We bring two core observations that reshape our understanding of FM. First, we rigorously prove that FM with linear interpolation probability paths is inherently non-invertible. Second, our analysis reveals that employing reversed Gaussian probability paths in high-dimensional spaces can lead to trivial vector fields. This issue arises due to the manifold-related constraints. Building on the second observation, we propose Worst Transport (WT) displacement interpolation to reconstruct a non-probabilistic evolution path. The proposed WT-Flow enhances dynamical control over sample trajectories, constructing ''degenerate potential wells'' for anomaly-free samples while allowing anomalous samples to escape. This novel unsupervised paradigm offers a theoretically grounded separation mechanism for anomalous samples. Notably, FM provides a computationally tractable framework that scales to complex data. We present the first successful application of FM for the unsupervised anomaly detection task, achieving state-of-the-art performance at a single scale on the MVTec dataset. The reproducible code for training will be released upon camera-ready submission.
>
---
#### [new 100] VS-LLM: Visual-Semantic Depression Assessment based on LLM for Drawing Projection Test
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出使用视觉语义方法（VS-LLM）基于LLM对绘画投影测试（DPT）中的PPAT进行抑郁症评估，旨在解决传统心理评估方法依赖经验导致的低准确性问题，创新性地开发了自动化方案并验证其效果。**

- **链接: [http://arxiv.org/pdf/2508.05299v1](http://arxiv.org/pdf/2508.05299v1)**

> **作者:** Meiqi Wu; Yaxuan Kang; Xuchen Li; Shiyu Hu; Xiaotang Chen; Yunfeng Kang; Weiqiang Wang; Kaiqi Huang
>
> **摘要:** The Drawing Projection Test (DPT) is an essential tool in art therapy, allowing psychologists to assess participants' mental states through their sketches. Specifically, through sketches with the theme of "a person picking an apple from a tree (PPAT)", it can be revealed whether the participants are in mental states such as depression. Compared with scales, the DPT can enrich psychologists' understanding of an individual's mental state. However, the interpretation of the PPAT is laborious and depends on the experience of the psychologists. To address this issue, we propose an effective identification method to support psychologists in conducting a large-scale automatic DPT. Unlike traditional sketch recognition, DPT more focus on the overall evaluation of the sketches, such as color usage and space utilization. Moreover, PPAT imposes a time limit and prohibits verbal reminders, resulting in low drawing accuracy and a lack of detailed depiction. To address these challenges, we propose the following efforts: (1) Providing an experimental environment for automated analysis of PPAT sketches for depression assessment; (2) Offering a Visual-Semantic depression assessment based on LLM (VS-LLM) method; (3) Experimental results demonstrate that our method improves by 17.6% compared to the psychologist assessment method. We anticipate that this work will contribute to the research in mental state assessment based on PPAT sketches' elements recognition. Our datasets and codes are available at https://github.com/wmeiqi/VS-LLM.
>
---
#### [new 101] Robust Tracking with Particle Filtering for Fluorescent Cardiac Imaging
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在开发一种针对荧光心脏成像的鲁棒跟踪方法，解决因心肌运动和图像特征波动导致的传统追踪方法效率不足的问题。提出基于循环一致性检查的粒子滤波追踪器，实现了高精度（5.00±0.22px）并优于现有技术（22.3±1.1px、58.1±27.1px）。**

- **链接: [http://arxiv.org/pdf/2508.05262v1](http://arxiv.org/pdf/2508.05262v1)**

> **作者:** Suresh Guttikonda; Maximilian Neidhart; Johanna Sprenger; Johannes Petersen; Christian Detter; Alexander Schlaefer
>
> **备注:** Accepted to CURAC conference 2025
>
> **摘要:** Intraoperative fluorescent cardiac imaging enables quality control following coronary bypass grafting surgery. We can estimate local quantitative indicators, such as cardiac perfusion, by tracking local feature points. However, heart motion and significant fluctuations in image characteristics caused by vessel structural enrichment limit traditional tracking methods. We propose a particle filtering tracker based on cyclicconsistency checks to robustly track particles sampled to follow target landmarks. Our method tracks 117 targets simultaneously at 25.4 fps, allowing real-time estimates during interventions. It achieves a tracking error of (5.00 +/- 0.22 px) and outperforms other deep learning trackers (22.3 +/- 1.1 px) and conventional trackers (58.1 +/- 27.1 px).
>
---
#### [new 102] X-VFL: A New Vertical Federated Learning Framework with Cross Completion and Decision Subspace Alignment
- **分类: cs.LG; cs.CV; cs.DC; math.OC**

- **简介: 该论文提出X-VFL框架，解决非对齐数据样本缺失及本地独立推理问题，通过XCom与DS-Align模块实现跨客户端特征完成和子空间对齐，证明收敛性并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.05568v1](http://arxiv.org/pdf/2508.05568v1)**

> **作者:** Qinghua Yao; Xiangrui Xu; Zhize Li
>
> **备注:** 20 pages
>
> **摘要:** Vertical Federated Learning (VFL) enables collaborative learning by integrating disjoint feature subsets from multiple clients/parties. However, VFL typically faces two key challenges: i) the requirement for perfectly aligned data samples across all clients (missing features are not allowed); ii) the requirement for joint collaborative inference/prediction involving all clients (it does not support locally independent inference on a single client). To address these challenges, we propose X-VFL, a new VFL framework designed to deal with the non-aligned data samples with (partially) missing features and to support locally independent inference of new data samples for each client. In particular, we design two novel modules in X-VFL: Cross Completion (XCom) and Decision Subspace Alignment (DS-Align). XCom can complete/reconstruct missing features for non-aligned data samples by leveraging information from other clients. DS-Align aligns local features with completed and global features across all clients within the decision subspace, thus enabling locally independent inference at each client. Moreover, we provide convergence theorems for different algorithms used in training X-VFL, showing an $O(1/\sqrt{T})$ convergence rate for SGD-type algorithms and an $O(1/T)$ rate for PAGE-type algorithms, where $T$ denotes the number of training update steps. Extensive experiments on real-world datasets demonstrate that X-VFL significantly outperforms existing methods, e.g., achieving a 15% improvement in accuracy on the image CIFAR-10 dataset and a 43% improvement on the medical MIMIC-III dataset. These results validate the practical effectiveness and superiority of X-VFL, particularly in scenarios involving partially missing features and locally independent inference.
>
---
#### [new 103] DistillDrive: End-to-End Multi-Mode Autonomous Driving Distillation by Isomorphic Hetero-Source Planning Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出了一种基于对偶源规划的端到端多模态自主驾驶知识蒸馏模型，旨在解决传统方法仅关注车辆状态而忽略规划能力的问题。通过结构化场景表示作为教师模型，利用多样化实例进行多目标学习，并结合强化学习优化决策映射，构建了生成式规划实例，有效提升了模型在nuScenes和NAVSIM等数据集上的性能表现。**

- **链接: [http://arxiv.org/pdf/2508.05402v1](http://arxiv.org/pdf/2508.05402v1)**

> **作者:** Rui Yu; Xianghang Zhang; Runkai Zhao; Huaicheng Yan; Meng Wang
>
> **摘要:** End-to-end autonomous driving has been recently seen rapid development, exerting a profound influence on both industry and academia. However, the existing work places excessive focus on ego-vehicle status as their sole learning objectives and lacks of planning-oriented understanding, which limits the robustness of the overall decision-making prcocess. In this work, we introduce DistillDrive, an end-to-end knowledge distillation-based autonomous driving model that leverages diversified instance imitation to enhance multi-mode motion feature learning. Specifically, we employ a planning model based on structured scene representations as the teacher model, leveraging its diversified planning instances as multi-objective learning targets for the end-to-end model. Moreover, we incorporate reinforcement learning to enhance the optimization of state-to-decision mappings, while utilizing generative modeling to construct planning-oriented instances, fostering intricate interactions within the latent space. We validate our model on the nuScenes and NAVSIM datasets, achieving a 50\% reduction in collision rate and a 3-point improvement in closed-loop performance compared to the baseline model. Code and model are publicly available at https://github.com/YuruiAI/DistillDrive
>
---
#### [new 104] Beyond Pixels: Medical Image Quality Assessment with Implicit Neural Representations
- **分类: eess.IV; cs.CV**

- **简介: 该论文旨在解决医疗成像中伪影影响诊断准确性的问题，通过隐式神经表示（INRs）构建紧凑模型，利用深度权重空间网络、图神经网络和关系注意力Transformer等架构实现图像质量评估，验证了其在ACDC数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2508.05168v1](http://arxiv.org/pdf/2508.05168v1)**

> **作者:** Caner Özer; Patryk Rygiel; Bram de Wilde; İlkay Öksüz; Jelmer M. Wolterink
>
> **备注:** Accepted in 16th Machine Learning in Medical Imaging (MLMI 2025) workshop
>
> **摘要:** Artifacts pose a significant challenge in medical imaging, impacting diagnostic accuracy and downstream analysis. While image-based approaches for detecting artifacts can be effective, they often rely on preprocessing methods that can lead to information loss and high-memory-demand medical images, thereby limiting the scalability of classification models. In this work, we propose the use of implicit neural representations (INRs) for image quality assessment. INRs provide a compact and continuous representation of medical images, naturally handling variations in resolution and image size while reducing memory overhead. We develop deep weight space networks, graph neural networks, and relational attention transformers that operate on INRs to achieve image quality assessment. Our method is evaluated on the ACDC dataset with synthetically generated artifact patterns, demonstrating its effectiveness in assessing image quality while achieving similar performance with fewer parameters.
>
---
#### [new 105] Don't Reach for the Stars: Rethinking Topology for Resilient Federated Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究了分布式联邦学习（FL）中的拓扑结构优化问题，旨在解决传统星型结构的局限性（如单点故障、隐私保护不足等）。通过设计P2P框架，提出基于协议得分的局部更新策略，结合正则化项增强训练稳定性，实验证明在对抗性和异构环境中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.05224v1](http://arxiv.org/pdf/2508.05224v1)**

> **作者:** Mirko Konstantin; Anirban Mukhopadhyay
>
> **摘要:** Federated learning (FL) enables collaborative model training across distributed clients while preserving data privacy by keeping data local. Traditional FL approaches rely on a centralized, star-shaped topology, where a central server aggregates model updates from clients. However, this architecture introduces several limitations, including a single point of failure, limited personalization, and poor robustness to distribution shifts or vulnerability to malfunctioning clients. Moreover, update selection in centralized FL often relies on low-level parameter differences, which can be unreliable when client data is not independent and identically distributed, and offer clients little control. In this work, we propose a decentralized, peer-to-peer (P2P) FL framework. It leverages the flexibility of the P2P topology to enable each client to identify and aggregate a personalized set of trustworthy and beneficial updates.This framework is the Local Inference Guided Aggregation for Heterogeneous Training Environments to Yield Enhancement Through Agreement and Regularization (LIGHTYEAR). Central to our method is an agreement score, computed on a local validation set, which quantifies the semantic alignment of incoming updates in the function space with respect to the clients reference model. Each client uses this score to select a tailored subset of updates and performs aggregation with a regularization term that further stabilizes the training. Our empirical evaluation across two datasets shows that the proposed approach consistently outperforms both centralized baselines and existing P2P methods in terms of client-level performance, particularly under adversarial and heterogeneous conditions.
>
---
#### [new 106] Artificial Intelligence-Based Classification of Spitz Tumors
- **分类: eess.IV; cs.CV**

- **简介: 该论文旨在通过AI模型实现Spitz肿瘤与传统黑色素瘤的分类、遗传突变预测及诊断分类，解决了诊断难度大、特征复杂的问题，开发并验证了基于UNI特征的AI模型，通过实验验证其优于经验病理学家，同时探索AI辅助诊断对流程优化的作用。**

- **链接: [http://arxiv.org/pdf/2508.05391v1](http://arxiv.org/pdf/2508.05391v1)**

> **作者:** Ruben T. Lucassen; Marjanna Romers; Chiel F. Ebbelaar; Aia N. Najem; Donal P. Hayes; Antien L. Mooyaart; Sara Roshani; Liliane C. D. Wynaendts; Nikolas Stathonikos; Gerben E. Breimer; Anne M. L. Jansen; Mitko Veta; Willeke A. M. Blokx
>
> **备注:** 19 pages, 2 figures, 6 tables, 6 supplementary tables
>
> **摘要:** Spitz tumors are diagnostically challenging due to overlap in atypical histological features with conventional melanomas. We investigated to what extent AI models, using histological and/or clinical features, can: (1) distinguish Spitz tumors from conventional melanomas; (2) predict the underlying genetic aberration of Spitz tumors; and (3) predict the diagnostic category of Spitz tumors. The AI models were developed and validated using a dataset of 393 Spitz tumors and 379 conventional melanomas. Predictive performance was measured using the AUROC and the accuracy. The performance of the AI models was compared with that of four experienced pathologists in a reader study. Moreover, a simulation experiment was conducted to investigate the impact of implementing AI-based recommendations for ancillary diagnostic testing on the workflow of the pathology department. The best AI model based on UNI features reached an AUROC of 0.95 and an accuracy of 0.86 in differentiating Spitz tumors from conventional melanomas. The genetic aberration was predicted with an accuracy of 0.55 compared to 0.25 for randomly guessing. The diagnostic category was predicted with an accuracy of 0.51, where random chance-level accuracy equaled 0.33. On all three tasks, the AI models performed better than the four pathologists, although differences were not statistically significant for most individual comparisons. Based on the simulation experiment, implementing AI-based recommendations for ancillary diagnostic testing could reduce material costs, turnaround times, and examinations. In conclusion, the AI models achieved a strong predictive performance in distinguishing between Spitz tumors and conventional melanomas. On the more challenging tasks of predicting the genetic aberration and the diagnostic category of Spitz tumors, the AI models performed better than random chance.
>
---
#### [new 107] Laplacian Analysis Meets Dynamics Modelling: Gaussian Splatting for 4D Reconstruction
- **分类: cs.GR; cs.CV; cs.MM**

- **简介: 该论文旨在解决动态3DGS中高维网格采样与低秩分解引起的谱冲突问题，提出融合哈希编码与拉普拉斯模块、补偿几何变形光度畸变及优化查询策略的新框架，实现复杂动态场景的高效4D重建。**

- **链接: [http://arxiv.org/pdf/2508.04966v1](http://arxiv.org/pdf/2508.04966v1)**

> **作者:** Yifan Zhou; Beizhen Zhao; Pengcheng Wu; Hao Wang
>
> **摘要:** While 3D Gaussian Splatting (3DGS) excels in static scene modeling, its extension to dynamic scenes introduces significant challenges. Existing dynamic 3DGS methods suffer from either over-smoothing due to low-rank decomposition or feature collision from high-dimensional grid sampling. This is because of the inherent spectral conflicts between preserving motion details and maintaining deformation consistency at different frequency. To address these challenges, we propose a novel dynamic 3DGS framework with hybrid explicit-implicit functions. Our approach contains three key innovations: a spectral-aware Laplacian encoding architecture which merges Hash encoding and Laplacian-based module for flexible frequency motion control, an enhanced Gaussian dynamics attribute that compensates for photometric distortions caused by geometric deformation, and an adaptive Gaussian split strategy guided by KDTree-based primitive control to efficiently query and optimize dynamic areas. Through extensive experiments, our method demonstrates state-of-the-art performance in reconstructing complex dynamic scenes, achieving better reconstruction fidelity.
>
---
#### [new 108] ALScope: A Unified Toolkit for Deep Active Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出ALScope平台，旨在统一深度主动学习（DAL）算法的评估，解决领域间性能差异和分布偏移等问题，整合10个CV/NNLP数据集及21种算法，支持灵活参数配置并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.04937v1](http://arxiv.org/pdf/2508.04937v1)**

> **作者:** Chenkai Wu; Yuanyuan Qi; Xiaohao Yang; Jueqing Lu; Gang Liu; Wray Buntine; Lan Du
>
> **摘要:** Deep Active Learning (DAL) reduces annotation costs by selecting the most informative unlabeled samples during training. As real-world applications become more complex, challenges stemming from distribution shifts (e.g., open-set recognition) and data imbalance have gained increasing attention, prompting the development of numerous DAL algorithms. However, the lack of a unified platform has hindered fair and systematic evaluation under diverse conditions. Therefore, we present a new DAL platform ALScope for classification tasks, integrating 10 datasets from computer vision (CV) and natural language processing (NLP), and 21 representative DAL algorithms, including both classical baselines and recent approaches designed to handle challenges such as distribution shifts and data imbalance. This platform supports flexible configuration of key experimental factors, ranging from algorithm and dataset choices to task-specific factors like out-of-distribution (OOD) sample ratio, and class imbalance ratio, enabling comprehensive and realistic evaluation. We conduct extensive experiments on this platform under various settings. Our findings show that: (1) DAL algorithms' performance varies significantly across domains and task settings; (2) in non-standard scenarios such as imbalanced and open-set settings, DAL algorithms show room for improvement and require further investigation; and (3) some algorithms achieve good performance, but require significantly longer selection time.
>
---
#### [new 109] Learning from Oblivion: Predicting Knowledge Overflowed Weights via Retrodiction of Forgetting
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文旨在解决如何从数据稀缺场景中获取更具知识性的预训练权重的问题，提出KNOW方法通过控制结构遗忘逆向推理合成权重，结合元学习实现有效预测，实验结果表明其优于传统方法，推动知识转移技术突破。**

- **链接: [http://arxiv.org/pdf/2508.05059v1](http://arxiv.org/pdf/2508.05059v1)**

> **作者:** Jinhyeok Jang; Jaehong Kim; Jung Uk Kim
>
> **摘要:** Pre-trained weights have become a cornerstone of modern deep learning, enabling efficient knowledge transfer and improving downstream task performance, especially in data-scarce scenarios. However, a fundamental question remains: how can we obtain better pre-trained weights that encapsulate more knowledge beyond the given dataset? In this work, we introduce \textbf{KNowledge Overflowed Weights (KNOW)} prediction, a novel strategy that leverages structured forgetting and its inversion to synthesize knowledge-enriched weights. Our key insight is that sequential fine-tuning on progressively downsized datasets induces a structured forgetting process, which can be modeled and reversed to recover knowledge as if trained on a larger dataset. We construct a dataset of weight transitions governed by this controlled forgetting and employ meta-learning to model weight prediction effectively. Specifically, our \textbf{KNowledge Overflowed Weights Nowcaster (KNOWN)} acts as a hyper-model that learns the general evolution of weights and predicts enhanced weights with improved generalization. Extensive experiments across diverse datasets and architectures demonstrate that KNOW prediction consistently outperforms Na\"ive fine-tuning and simple weight prediction, leading to superior downstream performance. Our work provides a new perspective on reinterpreting forgetting dynamics to push the limits of knowledge transfer in deep learning.
>
---
#### [new 110] Refining Gaussian Splatting: A Volumetric Densification Approach
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文旨在改进3D Gaussian Splatting (3DGS) 的点源管理，解决传统策略因惯性体积控制不足导致的重建质量下降问题。提出基于体积密度的精炼方法并结合结构从运动与深度图像匹配技术，通过实验验证其超越现有方法的性能。**

- **链接: [http://arxiv.org/pdf/2508.05187v1](http://arxiv.org/pdf/2508.05187v1)**

> **作者:** Mohamed Abdul Gafoor; Marius Preda; Titus Zaharia
>
> **摘要:** Achieving high-quality novel view synthesis in 3D Gaussian Splatting (3DGS) often depends on effective point primitive management. The underlying Adaptive Density Control (ADC) process addresses this issue by automating densification and pruning. Yet, the vanilla 3DGS densification strategy shows key shortcomings. To address this issue, in this paper we introduce a novel density control method, which exploits the volumes of inertia associated to each Gaussian function to guide the refinement process. Furthermore, we study the effect of both traditional Structure from Motion (SfM) and Deep Image Matching (DIM) methods for point cloud initialization. Extensive experimental evaluations on the Mip-NeRF 360 dataset demonstrate that our approach surpasses 3DGS in reconstruction quality, delivering encouraging performance across diverse scenes.
>
---
#### [new 111] Physically Controllable Relighting of Photographs
- **分类: cs.GR; cs.CV; I.4**

- **简介: 该论文提出一种自监督的图像重影方法，解决了传统3D工具（如Blender）缺乏物理控制的问题，通过融合传统物理渲染与神经渲染技术，构建基于颜色网格的场景表示并实现3D光照配置，利用路径追踪引擎和神经渲染预测最终效果，开发了可微处理以支持自监督训练。**

- **链接: [http://arxiv.org/pdf/2508.05626v1](http://arxiv.org/pdf/2508.05626v1)**

> **作者:** Chris Careaga; Yağız Aksoy
>
> **备注:** Proc. SIGGRAPH 2025, 10 pages, 9 figures
>
> **摘要:** We present a self-supervised approach to in-the-wild image relighting that enables fully controllable, physically based illumination editing. We achieve this by combining the physical accuracy of traditional rendering with the photorealistic appearance made possible by neural rendering. Our pipeline works by inferring a colored mesh representation of a given scene using monocular estimates of geometry and intrinsic components. This representation allows users to define their desired illumination configuration in 3D. The scene under the new lighting can then be rendered using a path-tracing engine. We send this approximate rendering of the scene through a feed-forward neural renderer to predict the final photorealistic relighting result. We develop a differentiable rendering process to reconstruct in-the-wild scene illumination, enabling self-supervised training of our neural renderer on raw image collections. Our method represents a significant step in bringing the explicit physical control over lights available in typical 3D computer graphics tools, such as Blender, to in-the-wild relighting.
>
---
#### [new 112] Voost: A Unified and Scalable Diffusion Transformer for Bidirectional Virtual Try-On and Try-Off
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出了一种基于扩散模型的双方向虚拟试衣/试穿框架Voost，解决了服装与人体对应关系建模的挑战，通过联合学习任务并引入自适应采样技术，在保持灵活性的同时提升鲁棒性，取得实验中的优异效果。**

- **链接: [http://arxiv.org/pdf/2508.04825v1](http://arxiv.org/pdf/2508.04825v1)**

> **作者:** Seungyong Lee; Jeong-gi Kwak
>
> **备注:** Project page: https://nxnai.github.io/Voost/
>
> **摘要:** Virtual try-on aims to synthesize a realistic image of a person wearing a target garment, but accurately modeling garment-body correspondence remains a persistent challenge, especially under pose and appearance variation. In this paper, we propose Voost - a unified and scalable framework that jointly learns virtual try-on and try-off with a single diffusion transformer. By modeling both tasks jointly, Voost enables each garment-person pair to supervise both directions and supports flexible conditioning over generation direction and garment category, enhancing garment-body relational reasoning without task-specific networks, auxiliary losses, or additional labels. In addition, we introduce two inference-time techniques: attention temperature scaling for robustness to resolution or mask variation, and self-corrective sampling that leverages bidirectional consistency between tasks. Extensive experiments demonstrate that Voost achieves state-of-the-art results on both try-on and try-off benchmarks, consistently outperforming strong baselines in alignment accuracy, visual fidelity, and generalization.
>
---
#### [new 113] Learning to See and Act: Task-Aware View Planning for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出**任务感知视图规划框架（TAVP）**，解决传统固定视角方法在多任务机器人操作中的3D感知不足及任务干扰问题，通过整合主动视图探索与任务特异性表示学习，提升视觉表示 fidelity 和任务泛化能力，并在RLBench等实验中优于传统方法。**

- **链接: [http://arxiv.org/pdf/2508.05186v1](http://arxiv.org/pdf/2508.05186v1)**

> **作者:** Yongjie Bai; Zhouxia Wang; Yang Liu; Weixing Chen; Ziliang Chen; Mingtong Dai; Yongsen Zheng; Lingbo Liu; Guanbin Li; Liang Lin
>
> **备注:** 7 pages, 9 figures, project page: https://hcplab-sysu.github.io/TAVP
>
> **摘要:** Recent vision-language-action (VLA) models for multi-task robotic manipulation commonly rely on static viewpoints and shared visual encoders, which limit 3D perception and cause task interference, hindering robustness and generalization. In this work, we propose Task-Aware View Planning (TAVP), a framework designed to overcome these challenges by integrating active view planning with task-specific representation learning. TAVP employs an efficient exploration policy, accelerated by a novel pseudo-environment, to actively acquire informative views. Furthermore, we introduce a Mixture-of-Experts (MoE) visual encoder to disentangle features across different tasks, boosting both representation fidelity and task generalization. By learning to see the world in a task-aware way, TAVP generates more complete and discriminative visual representations, demonstrating significantly enhanced action prediction across a wide array of manipulation challenges. Extensive experiments on RLBench tasks show that our proposed TAVP model achieves superior performance over state-of-the-art fixed-view approaches. Visual results and code are provided at: https://hcplab-sysu.github.io/TAVP.
>
---
#### [new 114] Coarse-to-Fine Joint Registration of MR and Ultrasound Images via Imaging Style Transfer
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文开发了基于3D CycleGAN的MRI与US图像联合注册方法，通过风格迁移增强一致性，采用平移与局部变形变换实现粗-细融合，解决多模态影像不一致问题。**

- **链接: [http://arxiv.org/pdf/2508.05240v1](http://arxiv.org/pdf/2508.05240v1)**

> **作者:** Junyi Wang; Xi Zhu; Yikun Guo; Zixi Wang; Haichuan Gao; Le Zhang; Fan Zhang
>
> **摘要:** We developed a pipeline for registering pre-surgery Magnetic Resonance (MR) images and post-resection Ultrasound (US) images. Our approach leverages unpaired style transfer using 3D CycleGAN to generate synthetic T1 images, thereby enhancing registration performance. Additionally, our registration process employs both affine and local deformable transformations for a coarse-to-fine registration. The results demonstrate that our approach improves the consistency between MR and US image pairs in most cases.
>
---
#### [new 115] A Study of the Framework and Real-World Applications of Language Embedding for 3D Scene Understanding
- **分类: cs.GR; cs.CL; cs.CV**

- **简介: 该论文研究了将语言嵌入与3D场景理解结合的方法，旨在解决计算效率低、数据标注不足等问题。通过结构化综述分析，探讨了语言引导下的3D Gaussian Splatting技术及其在文本生成、场景理解等领域的应用。**

- **链接: [http://arxiv.org/pdf/2508.05064v1](http://arxiv.org/pdf/2508.05064v1)**

> **作者:** Mahmoud Chick Zaouali; Todd Charter; Yehor Karpichev; Brandon Haworth; Homayoun Najjjaran
>
> **摘要:** Gaussian Splatting has rapidly emerged as a transformative technique for real-time 3D scene representation, offering a highly efficient and expressive alternative to Neural Radiance Fields (NeRF). Its ability to render complex scenes with high fidelity has enabled progress across domains such as scene reconstruction, robotics, and interactive content creation. More recently, the integration of Large Language Models (LLMs) and language embeddings into Gaussian Splatting pipelines has opened new possibilities for text-conditioned generation, editing, and semantic scene understanding. Despite these advances, a comprehensive overview of this emerging intersection has been lacking. This survey presents a structured review of current research efforts that combine language guidance with 3D Gaussian Splatting, detailing theoretical foundations, integration strategies, and real-world use cases. We highlight key limitations such as computational bottlenecks, generalizability, and the scarcity of semantically annotated 3D Gaussian data and outline open challenges and future directions for advancing language-guided 3D scene understanding using Gaussian Splatting.
>
---
#### [new 116] Advanced Multi-Architecture Deep Learning Framework for BIRADS-Based Mammographic Image Retrieval: Comprehensive Performance Analysis with Super-Ensemble Optimization
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文旨在解决BIRADS五类分类乳腺X线成像检索任务中的复杂性问题，通过多架构深度学习框架与超融合优化技术，有效提升了模型性能并验证了其在医学检索中的实际应用价值。**

- **链接: [http://arxiv.org/pdf/2508.04790v1](http://arxiv.org/pdf/2508.04790v1)**

> **作者:** MD Shaikh Rahman; Feiroz Humayara; Syed Maudud E Rabbi; Muhammad Mahbubur Rashid
>
> **摘要:** Content-based mammographic image retrieval systems require exact BIRADS categorical matching across five distinct classes, presenting significantly greater complexity than binary classification tasks commonly addressed in literature. Current medical image retrieval studies suffer from methodological limitations including inadequate sample sizes, improper data splitting, and insufficient statistical validation that hinder clinical translation. We developed a comprehensive evaluation framework systematically comparing CNN architectures (DenseNet121, ResNet50, VGG16) with advanced training strategies including sophisticated fine-tuning, metric learning, and super-ensemble optimization. Our evaluation employed rigorous stratified data splitting (50%/20%/30% train/validation/test), 602 test queries, and systematic validation using bootstrap confidence intervals with 1,000 samples. Advanced fine-tuning with differential learning rates achieved substantial improvements: DenseNet121 (34.79% precision@10, 19.64% improvement) and ResNet50 (34.54%, 19.58% improvement). Super-ensemble optimization combining complementary architectures achieved 36.33% precision@10 (95% CI: [34.78%, 37.88%]), representing 24.93% improvement over baseline and providing 3.6 relevant cases per query. Statistical analysis revealed significant performance differences between optimization strategies (p<0.001) with large effect sizes (Cohen's d>0.8), while maintaining practical search efficiency (2.8milliseconds). Performance significantly exceeds realistic expectations for 5-class medical retrieval tasks, where literature suggests 20-25% precision@10 represents achievable performance for exact BIRADS matching. Our framework establishes new performance benchmarks while providing evidence-based architecture selection guidelines for clinical deployment in diagnostic support and quality assurance applications.
>
---
#### [new 117] QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究了一个查询驱动的动态RAG系统，解决复杂视觉问答任务中单一源检索的局限性，通过引入领域路由和搜索路由实现多模态、多轮、多跳推理，显著提升VQA任务的准确性与知识覆盖度（5.06%-6.35%）。**

- **链接: [http://arxiv.org/pdf/2508.05197v1](http://arxiv.org/pdf/2508.05197v1)**

> **作者:** Zhuohang Jiang; Pangjing Wu; Xu Yuan; Wenqi Fan; Qing Li
>
> **备注:** The source code for our system is released in https://github.com/jzzzzh/QA-Dragon
>
> **摘要:** Retrieval-Augmented Generation (RAG) has been introduced to mitigate hallucinations in Multimodal Large Language Models (MLLMs) by incorporating external knowledge into the generation process, and it has become a widely adopted approach for knowledge-intensive Visual Question Answering (VQA). However, existing RAG methods typically retrieve from either text or images in isolation, limiting their ability to address complex queries that require multi-hop reasoning or up-to-date factual knowledge. To address this limitation, we propose QA-Dragon, a Query-Aware Dynamic RAG System for Knowledge-Intensive VQA. Specifically, QA-Dragon introduces a domain router to identify the query's subject domain for domain-specific reasoning, along with a search router that dynamically selects optimal retrieval strategies. By orchestrating both text and image search agents in a hybrid setup, our system supports multimodal, multi-turn, and multi-hop reasoning, enabling it to tackle complex VQA tasks effectively. We evaluate our QA-Dragon on the Meta CRAG-MM Challenge at KDD Cup 2025, where it significantly enhances the reasoning performance of base models under challenging scenarios. Our framework achieves substantial improvements in both answer accuracy and knowledge overlap scores, outperforming baselines by 5.06% on the single-source task, 6.35% on the multi-source task, and 5.03% on the multi-turn task.
>
---
#### [new 118] Perceive-Sample-Compress: Towards Real-Time 3D Gaussian Splatting
- **分类: cs.GR; cs.CV; cs.MM**

- **简介: 该论文旨在解决传统3DGS在大规模场景管理与存储效率上的瓶颈，提出感知-样本-压缩框架，通过感知补偿优化参数精炼、金字塔采样分级管理及混合模型压缩实现高效存储，显著提升实时渲染性能与视觉质量。**

- **链接: [http://arxiv.org/pdf/2508.04965v1](http://arxiv.org/pdf/2508.04965v1)**

> **作者:** Zijian Wang; Beizhen Zhao; Hao Wang
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated remarkable capabilities in real-time and photorealistic novel view synthesis. However, traditional 3DGS representations often struggle with large-scale scene management and efficient storage, particularly when dealing with complex environments or limited computational resources. To address these limitations, we introduce a novel perceive-sample-compress framework for 3D Gaussian Splatting. Specifically, we propose a scene perception compensation algorithm that intelligently refines Gaussian parameters at each level. This algorithm intelligently prioritizes visual importance for higher fidelity rendering in critical areas, while optimizing resource usage and improving overall visible quality. Furthermore, we propose a pyramid sampling representation to manage Gaussian primitives across hierarchical levels. Finally, to facilitate efficient storage of proposed hierarchical pyramid representations, we develop a Generalized Gaussian Mixed model compression algorithm to achieve significant compression ratios without sacrificing visual fidelity. The extensive experiments demonstrate that our method significantly improves memory efficiency and high visual quality while maintaining real-time rendering speed.
>
---
#### [new 119] Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文介绍了一款集成政策学习、评估与模拟的统一世界基础平台（Genie Envisioner），旨在解决机器人行动规划和泛化能力问题。通过融合GE-Base（指令-条件扩散模型）、GE-Act（流匹配解码器）和GE-Sim（动作-条件模拟器），解决了多模态交互中的执行精度和训练效率挑战，并建立标准化评价体系支持其应用。**

- **链接: [http://arxiv.org/pdf/2508.05635v1](http://arxiv.org/pdf/2508.05635v1)**

> **作者:** Yue Liao; Pengfei Zhou; Siyuan Huang; Donglin Yang; Shengcong Chen; Yuxin Jiang; Yue Hu; Jingbin Cai; Si Liu; Jianlan Luo; Liliang Chen; Shuicheng Yan; Maoqing Yao; Guanghui Ren
>
> **备注:** https://genie-envisioner.github.io/
>
> **摘要:** We introduce Genie Envisioner (GE), a unified world foundation platform for robotic manipulation that integrates policy learning, evaluation, and simulation within a single video-generative framework. At its core, GE-Base is a large-scale, instruction-conditioned video diffusion model that captures the spatial, temporal, and semantic dynamics of real-world robotic interactions in a structured latent space. Built upon this foundation, GE-Act maps latent representations to executable action trajectories through a lightweight, flow-matching decoder, enabling precise and generalizable policy inference across diverse embodiments with minimal supervision. To support scalable evaluation and training, GE-Sim serves as an action-conditioned neural simulator, producing high-fidelity rollouts for closed-loop policy development. The platform is further equipped with EWMBench, a standardized benchmark suite measuring visual fidelity, physical consistency, and instruction-action alignment. Together, these components establish Genie Envisioner as a scalable and practical foundation for instruction-driven, general-purpose embodied intelligence. All code, models, and benchmarks will be released publicly.
>
---
#### [new 120] CryoGS: Gaussian Splatting for Cryo-EM Homogeneous Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出了一种基于GMM的Gaussian splatting方法，解决单粒子cryo-EM中从2D投影重建分子电势的任务，通过优化正交投影、标准化项和FFT对齐等创新技术，在随机初始化下实现高效、稳定的高分辨率重构，有效克服了传统方法依赖原子模型的局限性。**

- **链接: [http://arxiv.org/pdf/2508.04929v1](http://arxiv.org/pdf/2508.04929v1)**

> **作者:** Suyi Chen; Haibin Ling
>
> **摘要:** As a critical modality for structural biology, cryogenic electron microscopy (cryo-EM) facilitates the determination of macromolecular structures at near-atomic resolution. The core computational task in single-particle cryo-EM is to reconstruct the 3D electrostatic potential of a molecule from a large collection of noisy 2D projections acquired at unknown orientations. Gaussian mixture models (GMMs) provide a continuous, compact, and physically interpretable representation for molecular density and have recently gained interest in cryo-EM reconstruction. However, existing methods rely on external consensus maps or atomic models for initialization, limiting their use in self-contained pipelines. Addressing this issue, we introduce cryoGS, a GMM-based method that integrates Gaussian splatting with the physics of cryo-EM image formation. In particular, we develop an orthogonal projection-aware Gaussian splatting, with adaptations such as a normalization term and FFT-aligned coordinate system tailored for cryo-EM imaging. All these innovations enable stable and efficient homogeneous reconstruction directly from raw cryo-EM particle images using random initialization. Experimental results on real datasets validate the effectiveness and robustness of cryoGS over representative baselines. The code will be released upon publication.
>
---
#### [new 121] Divide-and-Conquer for Enhancing Unlabeled Learning, Stability, and Plasticity in Semi-supervised Continual Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于半监督持续学习任务，旨在解决无标签数据增强、模型稳定性及学习可塑性等问题，提出基于分治框架的USP方法，通过特征空间预留、伪标签分配和类均值锚定等策略协同优化三者性能。**

- **链接: [http://arxiv.org/pdf/2508.05316v1](http://arxiv.org/pdf/2508.05316v1)**

> **作者:** Yue Duan; Taicai Chen; Lei Qi; Yinghuan Shi
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Semi-supervised continual learning (SSCL) seeks to leverage both labeled and unlabeled data in a sequential learning setup, aiming to reduce annotation costs while managing continual data arrival. SSCL introduces complex challenges, including ensuring effective unlabeled learning (UL), while balancing memory stability (MS) and learning plasticity (LP). Previous SSCL efforts have typically focused on isolated aspects of the three, while this work presents USP, a divide-and-conquer framework designed to synergistically enhance these three aspects: (1) Feature Space Reservation (FSR) strategy for LP, which constructs reserved feature locations for future classes by shaping old classes into an equiangular tight frame; (2) Divide-and-Conquer Pseudo-labeling (DCP) approach for UL, which assigns reliable pseudo-labels across both high- and low-confidence unlabeled data; and (3) Class-mean-anchored Unlabeled Distillation (CUD) for MS, which reuses DCP's outputs to anchor unlabeled data to stable class means for distillation to prevent forgetting. Comprehensive evaluations show USP outperforms prior SSCL methods, with gains up to 5.94% in the last accuracy, validating its effectiveness. The code is available at https://github.com/NJUyued/USP4SSCL.
>
---
#### [new 122] Parameter-free entropy-regularized multi-view clustering with hierarchical feature selection
- **分类: cs.LG; cs.CV; math.ST; stat.TH; 62H30, 68T05, 68T09, 62H25, 94A17**

- **简介: 该论文旨在解决多视图聚类中跨异构数据自动发现模式与高维特征信息消融的问题，通过引入熵正则化和维度降维技术，开发了AMVFCM-U等算法框架，实现了高效且鲁棒的多视角特征融合。**

- **链接: [http://arxiv.org/pdf/2508.05504v1](http://arxiv.org/pdf/2508.05504v1)**

> **作者:** Kristina P. Sinaga; Sara Colantonio; Miin-Shen Yang
>
> **备注:** 81 pages, 10 figures, 17 tables
>
> **摘要:** Multi-view clustering faces critical challenges in automatically discovering patterns across heterogeneous data while managing high-dimensional features and eliminating irrelevant information. Traditional approaches suffer from manual parameter tuning and lack principled cross-view integration mechanisms. This work introduces two complementary algorithms: AMVFCM-U and AAMVFCM-U, providing a unified parameter-free framework. Our approach replaces fuzzification parameters with entropy regularization terms that enforce adaptive cross-view consensus. The core innovation employs signal-to-noise ratio based regularization ($\delta_j^h = \frac{\bar{x}_j^h}{(\sigma_j^h)^2}$) for principled feature weighting with convergence guarantees, coupled with dual-level entropy terms that automatically balance view and feature contributions. AAMVFCM-U extends this with hierarchical dimensionality reduction operating at feature and view levels through adaptive thresholding ($\theta^{h^{(t)}} = \frac{d_h^{(t)}}{n}$). Evaluation across five diverse benchmarks demonstrates superiority over 15 state-of-the-art methods. AAMVFCM-U achieves up to 97% computational efficiency gains, reduces dimensionality to 0.45% of original size, and automatically identifies critical view combinations for optimal pattern discovery.
>
---
#### [new 123] Point cloud segmentation for 3D Clothed Human Layering
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出一种基于点云的3D服装层叠分割方法，解决服装与人体结构分离问题，通过改进模型实现对衣物区域的识别，并构建包含真实和合成数据集的实验。**

- **链接: [http://arxiv.org/pdf/2508.05531v1](http://arxiv.org/pdf/2508.05531v1)**

> **作者:** Davide Garavaso; Federico Masi; Pietro Musoni; Umberto Castellani
>
> **摘要:** 3D Cloth modeling and simulation is essential for avatars creation in several fields, such as fashion, entertainment, and animation. Achieving high-quality results is challenging due to the large variability of clothed body especially in the generation of realistic wrinkles. 3D scan acquisitions provide more accuracy in the representation of real-world objects but lack semantic information that can be inferred with a reliable semantic reconstruction pipeline. To this aim, shape segmentation plays a crucial role in identifying the semantic shape parts. However, current 3D shape segmentation methods are designed for scene understanding and interpretation and only few work is devoted to modeling. In the context of clothed body modeling the segmentation is a preliminary step for fully semantic shape parts reconstruction namely the underlying body and the involved garments. These parts represent several layers with strong overlap in contrast with standard segmentation methods that provide disjoint sets. In this work we propose a new 3D point cloud segmentation paradigm where each 3D point can be simultaneously associated to different layers. In this fashion we can estimate the underlying body parts and the unseen clothed regions, i.e., the part of a cloth occluded by the clothed-layer above. We name this segmentation paradigm clothed human layering. We create a new synthetic dataset that simulates very realistic 3D scans with the ground truth of the involved clothing layers. We propose and evaluate different neural network settings to deal with 3D clothing layering. We considered both coarse and fine grained per-layer garment identification. Our experiments demonstrates the benefit in introducing proper strategies for the segmentation on the garment domain on both the synthetic and real-world scan datasets.
>
---
#### [new 124] Adapting Vision-Language Models Without Labels: A Comprehensive Survey
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究了视觉-语言模型（VLM）的无监督适应方法，旨在解决其在下游任务中性能不佳的问题。通过构建四类无标注适应框架（数据无关/丰富数据/事件测试时间/在线流式），分析核心策略并建立统一理论体系，为高效泛化提供方法支持。**

- **链接: [http://arxiv.org/pdf/2508.05547v1](http://arxiv.org/pdf/2508.05547v1)**

> **作者:** Hao Dong; Lijun Sheng; Jian Liang; Ran He; Eleni Chatzi; Olga Fink
>
> **备注:** Discussions, comments, and questions are welcome in \url{https://github.com/tim-learn/Awesome-LabelFree-VLMs}
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable generalization capabilities across a wide range of tasks. However, their performance often remains suboptimal when directly applied to specific downstream scenarios without task-specific adaptation. To enhance their utility while preserving data efficiency, recent research has increasingly focused on unsupervised adaptation methods that do not rely on labeled data. Despite the growing interest in this area, there remains a lack of a unified, task-oriented survey dedicated to unsupervised VLM adaptation. To bridge this gap, we present a comprehensive and structured overview of the field. We propose a taxonomy based on the availability and nature of unlabeled visual data, categorizing existing approaches into four key paradigms: Data-Free Transfer (no data), Unsupervised Domain Transfer (abundant data), Episodic Test-Time Adaptation (batch data), and Online Test-Time Adaptation (streaming data). Within this framework, we analyze core methodologies and adaptation strategies associated with each paradigm, aiming to establish a systematic understanding of the field. Additionally, we review representative benchmarks across diverse applications and highlight open challenges and promising directions for future research. An actively maintained repository of relevant literature is available at https://github.com/tim-learn/Awesome-LabelFree-VLMs.
>
---
#### [new 125] Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文旨在解决在拥挤人群中移动机器人因分布外场景导致性能下降的问题。通过结合适配化模糊推理生成预测不确定性，采用约束强化学习优化行为，提升其在正常与分布外场景下的安全性和鲁棒性，有效减少碰撞与入侵轨迹。**

- **链接: [http://arxiv.org/pdf/2508.05634v1](http://arxiv.org/pdf/2508.05634v1)**

> **作者:** Jianpeng Yao; Xiaopan Zhang; Yu Xia; Zejin Wang; Amit K. Roy-Chowdhury; Jiachen Li
>
> **备注:** 9th Conference on Robot Learning (CoRL 2025); Project website: https://gen-safe-nav.github.io/. arXiv admin note: text overlap with arXiv:2407.17460
>
> **摘要:** Mobile robots navigating in crowds trained using reinforcement learning are known to suffer performance degradation when faced with out-of-distribution scenarios. We propose that by properly accounting for the uncertainties of pedestrians, a robot can learn safe navigation policies that are robust to distribution shifts. Our method augments agent observations with prediction uncertainty estimates generated by adaptive conformal inference, and it uses these estimates to guide the agent's behavior through constrained reinforcement learning. The system helps regulate the agent's actions and enables it to adapt to distribution shifts. In the in-distribution setting, our approach achieves a 96.93% success rate, which is over 8.80% higher than the previous state-of-the-art baselines with over 3.72 times fewer collisions and 2.43 times fewer intrusions into ground-truth human future trajectories. In three out-of-distribution scenarios, our method shows much stronger robustness when facing distribution shifts in velocity variations, policy changes, and transitions from individual to group dynamics. We deploy our method on a real robot, and experiments show that the robot makes safe and robust decisions when interacting with both sparse and dense crowds. Our code and videos are available on https://gen-safe-nav.github.io/.
>
---
#### [new 126] Towards Robust Evaluation of Visual Activity Recognition: Resolving Verb Ambiguity with Sense Clustering
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文旨在解决视觉活动识别中的语义歧义与视角差异问题，提出通过感觉聚类框架构建多视角识别方案，并在imSitu数据集上验证其有效性，相较于传统方法更准确地评估模型表现。**

- **链接: [http://arxiv.org/pdf/2508.04945v1](http://arxiv.org/pdf/2508.04945v1)**

> **作者:** Louie Hong Yao; Nicholas Jarvis; Tianyu Jiang
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** Evaluating visual activity recognition systems is challenging due to inherent ambiguities in verb semantics and image interpretation. When describing actions in images, synonymous verbs can refer to the same event (e.g., brushing vs. grooming), while different perspectives can lead to equally valid but distinct verb choices (e.g., piloting vs. operating). Standard exact-match evaluation, which relies on a single gold answer, fails to capture these ambiguities, resulting in an incomplete assessment of model performance. To address this, we propose a vision-language clustering framework that constructs verb sense clusters, providing a more robust evaluation. Our analysis of the imSitu dataset shows that each image maps to an average of 2.8 sense clusters, with each cluster representing a distinct perspective of the image. We evaluate multiple activity recognition models and compare our cluster-based evaluation with standard evaluation methods. Additionally, our human alignment analysis suggests that the cluster-based evaluation better aligns with human judgements, offering a more nuanced assessment of model performance.
>
---
#### [new 127] RAP: Real-time Audio-driven Portrait Animation with Video Diffusion Transformer
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **简介: 该论文旨在解决实时音频驱动人像动画（RAP）的问题，通过引入混合注意力机制和静态-动态训练策略，实现了高效且高质量的实时合成。任务为生成逼真的人像视频，目标是平衡音频控制与视觉细节保留。**

- **链接: [http://arxiv.org/pdf/2508.05115v1](http://arxiv.org/pdf/2508.05115v1)**

> **作者:** Fangyu Du; Taiqing Li; Ziwei Zhang; Qian Qiao; Tan Yu; Dingcheng Zhen; Xu Jia; Yang Yang; Shunshun Yin; Siyuan Liu
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Audio-driven portrait animation aims to synthesize realistic and natural talking head videos from an input audio signal and a single reference image. While existing methods achieve high-quality results by leveraging high-dimensional intermediate representations and explicitly modeling motion dynamics, their computational complexity renders them unsuitable for real-time deployment. Real-time inference imposes stringent latency and memory constraints, often necessitating the use of highly compressed latent representations. However, operating in such compact spaces hinders the preservation of fine-grained spatiotemporal details, thereby complicating audio-visual synchronization RAP (Real-time Audio-driven Portrait animation), a unified framework for generating high-quality talking portraits under real-time constraints. Specifically, RAP introduces a hybrid attention mechanism for fine-grained audio control, and a static-dynamic training-inference paradigm that avoids explicit motion supervision. Through these techniques, RAP achieves precise audio-driven control, mitigates long-term temporal drift, and maintains high visual fidelity. Extensive experiments demonstrate that RAP achieves state-of-the-art performance while operating under real-time constraints.
>
---
## 更新

#### [replaced 001] ReferEverything: Towards Segmenting Everything We Can Speak of in Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.23287v2](http://arxiv.org/pdf/2410.23287v2)**

> **作者:** Anurag Bagchi; Zhipeng Bao; Yu-Xiong Wang; Pavel Tokmakov; Martial Hebert
>
> **备注:** Project page at https://refereverything.github.io/
>
> **摘要:** We present REM, a framework for segmenting a wide range of concepts in video that can be described through natural language. Our method leverages the universal visual-language mapping learned by video diffusion models on Internet-scale data by fine-tuning them on small-scale Referring Object Segmentation datasets. Our key insight is to preserve the entirety of the generative model's architecture by shifting its objective from predicting noise to predicting mask latents. The resulting model can accurately segment rare and unseen objects, despite only being trained on a limited set of categories. Additionally, it can effortlessly generalize to non-object dynamic concepts, such as smoke or raindrops, as demonstrated in our new benchmark for Referring Video Process Segmentation (Ref-VPS). REM performs on par with the state-of-the-art on in-domain datasets, like Ref-DAVIS, while outperforming them by up to 12 IoU points out-of-domain, leveraging the power of generative pre-training. We also show that advancements in video generation directly improve segmentation.
>
---
#### [replaced 002] WeatherEdit: Controllable Weather Editing with 4D Gaussian Field
- **分类: cs.CV; cs.AI; cs.ET; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20471v3](http://arxiv.org/pdf/2505.20471v3)**

> **作者:** Chenghao Qian; Wenjing Li; Yuhu Guo; Gustav Markkula
>
> **摘要:** In this work, we present WeatherEdit, a novel weather editing pipeline for generating realistic weather effects with controllable types and severity in 3D scenes. Our approach is structured into two key components: weather background editing and weather particle construction. For weather background editing, we introduce an all-in-one adapter that integrates multiple weather styles into a single pretrained diffusion model, enabling the generation of diverse weather effects in 2D image backgrounds. During inference, we design a Temporal-View (TV-) attention mechanism that follows a specific order to aggregate temporal and spatial information, ensuring consistent editing across multi-frame and multi-view images. To construct the weather particles, we first reconstruct a 3D scene using the edited images and then introduce a dynamic 4D Gaussian field to generate snowflakes, raindrops and fog in the scene. The attributes and dynamics of these particles are precisely controlled through physical-based modelling and simulation, ensuring realistic weather representation and flexible severity adjustments. Finally, we integrate the 4D Gaussian field with the 3D scene to render consistent and highly realistic weather effects. Experiments on multiple driving datasets demonstrate that WeatherEdit can generate diverse weather effects with controllable condition severity, highlighting its potential for autonomous driving simulation in adverse weather. See project page: https://jumponthemoon.github.io/w-edit
>
---
#### [replaced 003] EgoPrompt: Prompt Learning for Egocentric Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03266v2](http://arxiv.org/pdf/2508.03266v2)**

> **作者:** Huaihai Lyu; Chaofan Chen; Yuheng Ji; Changsheng Xu
>
> **摘要:** Driven by the increasing demand for applications in augmented and virtual reality, egocentric action recognition has emerged as a prominent research area. It is typically divided into two subtasks: recognizing the performed behavior (i.e., verb component) and identifying the objects being acted upon (i.e., noun component) from the first-person perspective. However, most existing approaches treat these two components as independent classification tasks, focusing on extracting component-specific knowledge while overlooking their inherent semantic and contextual relationships, leading to fragmented representations and sub-optimal generalization capability. To address these challenges, we propose a prompt learning-based framework, EgoPrompt, to conduct the egocentric action recognition task. Building on the existing prompting strategy to capture the component-specific knowledge, we construct a Unified Prompt Pool space to establish interaction between the two types of component representations. Specifically, the component representations (from verbs and nouns) are first decomposed into fine-grained patterns with the prompt pair form. Then, these pattern-level representations are fused through an attention-based mechanism to facilitate cross-component interaction. To ensure the prompt pool is informative, we further introduce a novel training objective, Diverse Pool Criteria. This objective realizes our goals from two perspectives: Prompt Selection Frequency Regularization and Prompt Knowledge Orthogonalization. Extensive experiments are conducted on the Ego4D, EPIC-Kitchens, and EGTEA datasets. The results consistently show that EgoPrompt achieves state-of-the-art performance across within-dataset, cross-dataset, and base-to-novel generalization benchmarks.
>
---
#### [replaced 004] Few-Shot Vision-Language Reasoning for Satellite Imagery via Verifiable Rewards
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.21745v2](http://arxiv.org/pdf/2507.21745v2)**

> **作者:** Aybora Koksal; A. Aydin Alatan
>
> **备注:** ICCV 2025 Workshop on Curated Data for Efficient Learning (CDEL). 10 pages, 3 figures, 6 tables. Our model, training code and dataset will be at https://github.com/aybora/FewShotReasoning
>
> **摘要:** Recent advances in large language and vision-language models have enabled strong reasoning capabilities, yet they remain impractical for specialized domains like remote sensing, where annotated data is scarce and expensive. We present the first few-shot reinforcement learning with verifiable reward (RLVR) framework for satellite imagery that eliminates the need for caption supervision--relying solely on lightweight, rule-based binary or IoU-based rewards. Adapting the "1-shot RLVR" paradigm from language models to vision-language models, we employ policy-gradient optimization with as few as one curated example to align model outputs for satellite reasoning tasks. Comprehensive experiments across multiple remote sensing benchmarks--including classification, visual question answering, and grounding--show that even a single example yields substantial improvements over the base model. Scaling to 128 examples matches or exceeds models trained on thousands of annotated samples. While the extreme one-shot setting can induce mild, task-specific overfitting, our approach consistently demonstrates robust generalization and efficiency across diverse tasks. Further, we find that prompt design and loss weighting significantly influence training stability and final accuracy. Our method enables cost-effective and data-efficient development of domain-specialist vision-language reasoning models, offering a pragmatic recipe for data-scarce fields: start from a compact VLM, curate a handful of reward-checkable cases, and train via RLVR.
>
---
#### [replaced 005] Personalized Safety Alignment for Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.01151v2](http://arxiv.org/pdf/2508.01151v2)**

> **作者:** Yu Lei; Jinbin Bai; Qingyu Shi; Aosong Feng; Kaidong Yu
>
> **备注:** metadata-only revision; corrected a typo in the abstract. No changes to the PDF content
>
> **摘要:** Text-to-image diffusion models have revolutionized visual content generation, but current safety mechanisms apply uniform standards that often fail to account for individual user preferences. These models overlook the diverse safety boundaries shaped by factors like age, mental health, and personal beliefs. To address this, we propose Personalized Safety Alignment (PSA), a framework that allows user-specific control over safety behaviors in generative models. PSA integrates personalized user profiles into the diffusion process, adjusting the model's behavior to match individual safety preferences while preserving image quality. We introduce a new dataset, Sage, which captures user-specific safety preferences and incorporates these profiles through a cross-attention mechanism. Experiments show that PSA outperforms existing methods in harmful content suppression and aligns generated content better with user constraints, achieving higher Win Rate and Pass Rate scores. Our code, data, and models are publicly available at https://m-e-agi-lab.github.io/PSAlign/.
>
---
#### [replaced 006] CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03539v2](http://arxiv.org/pdf/2507.03539v2)**

> **作者:** Elena Bueno-Benito; Mariella Dimiccoli
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Unsupervised action segmentation has recently pushed its limits with ASOT, an optimal transport (OT)-based method that simultaneously learns action representations and performs clustering using pseudo-labels. Unlike other OT-based approaches, ASOT makes no assumptions about action ordering and can decode a temporally consistent segmentation from a noisy cost matrix between video frames and action labels. However, the resulting segmentation lacks segment-level supervision, limiting the effectiveness of feedback between frames and action representations. To address this limitation, we propose Closed Loop Optimal Transport (CLOT), a novel OT-based framework with a multi-level cyclic feature learning mechanism. Leveraging its encoder-decoder architecture, CLOT learns pseudo-labels alongside frame and segment embeddings by solving two separate OT problems. It then refines both frame embeddings and pseudo-labels through cross-attention between the learned frame and segment embeddings, by integrating a third OT problem. Experimental results on four benchmark datasets demonstrate the benefits of cyclical learning for unsupervised action segmentation.
>
---
#### [replaced 007] Beyond Subspace Isolation: Many-to-Many Transformer for Light Field Image Super-resolution
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2401.00740v3](http://arxiv.org/pdf/2401.00740v3)**

> **作者:** Zeke Zexi Hu; Xiaoming Chen; Vera Yuk Ying Chung; Yiran Shen
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** The effective extraction of spatial-angular features plays a crucial role in light field image super-resolution (LFSR) tasks, and the introduction of convolution and Transformers leads to significant improvement in this area. Nevertheless, due to the large 4D data volume of light field images, many existing methods opted to decompose the data into a number of lower-dimensional subspaces and perform Transformers in each sub-space individually. As a side effect, these methods inadvertently restrict the self-attention mechanisms to a One-to-One scheme accessing only a limited subset of LF data, explicitly preventing comprehensive optimization on all spatial and angular cues. In this paper, we identify this limitation as subspace isolation and introduce a novel Many-to-Many Transformer (M2MT) to address it. M2MT aggregates angular information in the spatial subspace before performing the self-attention mechanism. It enables complete access to all information across all sub-aperture images (SAIs) in a light field image. Consequently, M2MT is enabled to comprehensively capture long-range correlation dependencies. With M2MT as the foundational component, we develop a simple yet effective M2MT network for LFSR. Our experimental results demonstrate that M2MT achieves state-of-the-art performance across various public datasets, and it offers a favorable balance between model performance and efficiency, yielding higher-quality LFSR results with substantially lower demand for memory and computation. We further conduct in-depth analysis using local attribution maps (LAM) to obtain visual interpretability, and the results validate that M2MT is empowered with a truly non-local context in both spatial and angular subspaces to mitigate subspace isolation and acquire effective spatial-angular representation.
>
---
#### [replaced 008] DSOcc: Leveraging Depth Awareness and Semantic Aid to Boost Camera-Based 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20951v3](http://arxiv.org/pdf/2505.20951v3)**

> **作者:** Naiyu Fang; Zheyuan Zhou; Kang Wang; Ruibo Li; Lemiao Qiu; Shuyou Zhang; Zhe Wang; Guosheng Lin
>
> **摘要:** Camera-based 3D semantic occupancy prediction offers an efficient and cost-effective solution for perceiving surrounding scenes in autonomous driving. However, existing works rely on explicit occupancy state inference, leading to numerous incorrect feature assignments, and insufficient samples restrict the learning of occupancy class inference. To address these challenges, we propose leveraging Depth awareness and Semantic aid to boost camera-based 3D semantic Occupancy prediction (DSOcc). We jointly perform occupancy state and occupancy class inference, where soft occupancy confidence is calculated by non-learning method and multiplied with image features to make voxels aware of depth, enabling adaptive implicit occupancy state inference. Instead of enhancing feature learning, we directly utilize well-trained image semantic segmentation and fuse multiple frames with their occupancy probabilities to aid occupancy class inference, thereby enhancing robustness. Experimental results demonstrate that DSOcc achieves state-of-the-art performance on the SemanticKITTI dataset among camera-based methods.
>
---
#### [replaced 009] GPSMamba: A Global Phase and Spectral Prompt-guided Mamba for Infrared Image Super-Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18998v2](http://arxiv.org/pdf/2507.18998v2)**

> **作者:** Yongsong Huang; Tomo Miyazaki; Shinichiro Omachi
>
> **备注:** This manuscript is under review, and copyright will be transferred without notice
>
> **摘要:** Infrared Image Super-Resolution (IRSR) is challenged by the low contrast and sparse textures of infrared data, requiring robust long-range modeling to maintain global coherence. While State-Space Models like Mamba offer proficiency in modeling long-range dependencies for this task, their inherent 1D causal scanning mechanism fragments the global context of 2D images, hindering fine-detail restoration. To address this, we propose Global Phase and Spectral Prompt-guided Mamba (GPSMamba), a framework that synergizes architectural guidance with non-causal supervision. First, our Adaptive Semantic-Frequency State Space Module (ASF-SSM) injects a fused semantic-frequency prompt directly into the Mamba block, integrating non-local context to guide reconstruction. Then, a novel Thermal-Spectral Attention and Phase Consistency Loss provides explicit, non-causal supervision to enforce global structural and spectral fidelity. By combining these two innovations, our work presents a systematic strategy to mitigate the limitations of causal modeling. Extensive experiments demonstrate that GPSMamba achieves state-of-the-art performance, validating our approach as a powerful new paradigm for infrared image restoration. Code is available at https://github.com/yongsongH/GPSMamba.
>
---
#### [replaced 010] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.07940v2](http://arxiv.org/pdf/2503.07940v2)**

> **作者:** Minkyun Seo; Hyungtae Lim; Kanghee Lee; Luca Carlone; Jaesik Park
>
> **备注:** 20 pages, 14 figures. Accepted as a highlight paper at ICCV 2025
>
> **摘要:** Recent advances in deep learning-based point cloud registration have improved generalization, yet most methods still require retraining or manual parameter tuning for each new environment. In this paper, we identify three key factors limiting generalization: (a) reliance on environment-specific voxel size and search radius, (b) poor out-of-domain robustness of learning-based keypoint detectors, and (c) raw coordinate usage, which exacerbates scale discrepancies. To address these issues, we present a zero-shot registration pipeline called BUFFER-X by (a) adaptively determining voxel size/search radii, (b) using farthest point sampling to bypass learned detectors, and (c) leveraging patch-wise scale normalization for consistent coordinate bounds. In particular, we present a multi-scale patch-based descriptor generation and a hierarchical inlier search across scales to improve robustness in diverse scenes. We also propose a novel generalizability benchmark using 11 datasets that cover various indoor/outdoor scenarios and sensor modalities, demonstrating that BUFFER-X achieves substantial generalization without prior information or manual parameter tuning for the test datasets. Our code is available at https://github.com/MIT-SPARK/BUFFER-X.
>
---
#### [replaced 011] Modular Transformer Architecture for Precision Agriculture Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03751v2](http://arxiv.org/pdf/2508.03751v2)**

> **作者:** Brian Gopalan; Nathalia Nascimento; Vishal Monga
>
> **备注:** Preprint of paper submitted to IEEE-AIOT 2025
>
> **摘要:** This paper addresses the critical need for efficient and accurate weed segmentation from drone video in precision agriculture. A quality-aware modular deep-learning framework is proposed that addresses common image degradation by analyzing quality conditions-such as blur and noise-and routing inputs through specialized pre-processing and transformer models optimized for each degradation type. The system first analyzes drone images for noise and blur using Mean Absolute Deviation and the Laplacian. Data is then dynamically routed to one of three vision transformer models: a baseline for clean images, a modified transformer with Fisher Vector encoding for noise reduction, or another with an unrolled Lucy-Richardson decoder to correct blur. This novel routing strategy allows the system to outperform existing CNN-based methods in both segmentation quality and computational efficiency, demonstrating a significant advancement in deep-learning applications for agriculture.
>
---
#### [replaced 012] PerSense: Training-Free Personalized Instance Segmentation in Dense Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.13518v4](http://arxiv.org/pdf/2405.13518v4)**

> **作者:** Muhammad Ibraheem Siddiqui; Muhammad Umer Sheikh; Hassan Abid; Muhammad Haris Khan
>
> **备注:** Technical report of PerSense
>
> **摘要:** The emergence of foundational models has significantly advanced segmentation approaches. However, challenges still remain in dense scenarios, where occlusions, scale variations, and clutter impede precise instance delineation. To address this, we propose PerSense, an end-to-end, training-free, and model-agnostic one-shot framework for Personalized instance Segmentation in dense images. We start with developing a new baseline capable of automatically generating instance-level point prompts via proposing a novel Instance Detection Module (IDM) that leverages density maps (DMs), encapsulating spatial distribution of objects in an image. To reduce false positives, we design the Point Prompt Selection Module (PPSM), which refines the output of IDM based on adaptive threshold and spatial gating. Both IDM and PPSM seamlessly integrate into our model-agnostic framework. Furthermore, we introduce a feedback mechanism that enables PerSense to improve the accuracy of DMs by automating the exemplar selection process for DM generation. Finally, to advance research in this relatively underexplored area, we introduce PerSense-D, an evaluation benchmark for instance segmentation in dense images. Our extensive experiments establish PerSense's superiority over SOTA in dense settings.
>
---
#### [replaced 013] Interior Object Geometry via Fitted Frames
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.14357v2](http://arxiv.org/pdf/2407.14357v2)**

> **作者:** Stephen M. Pizer; Zhiyuan Liu; Junjie Zhao; Nicholas Tapp-Hughes; James Damon; Miaomiao Zhang; JS Marron; Mohsen Taheri; Jared Vicory
>
> **摘要:** We propose a means of computing fitted frames on the boundary and in the interior of objects and using them to provide the basis for producing geometric features from them that are not only alignment-free but most importantly can be made to correspond locally across a population of objects. We describe a representation targeted for anatomic objects which is designed to enable this strong locational correspondence within object populations and thus to provide powerful object statistics. It accomplishes this by understanding an object as the diffeomorphic deformation of the closure of the interior of an ellipsoid and by using a skeletal representation fitted throughout the deformation to produce a model of the target object, where the object is provided initially in the form of a boundary mesh. Via classification performance on hippocampi shape between individuals with a disorder vs. others, we compare our method to two state-of-theart methods for producing object representations that are intended to capture geometric correspondence across a population of objects and to yield geometric features useful for statistics, and we show notably improved classification performance by this new representation, which we call the evolutionary s-rep. The geometric features that are derived from each of the representations, especially via fitted frames, are discussed.
>
---
#### [replaced 014] SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18576v3](http://arxiv.org/pdf/2507.18576v3)**

> **作者:** Shanghai AI Lab; :; Yicheng Bao; Guanxu Chen; Mingkang Chen; Yunhao Chen; Chiyu Chen; Lingjie Chen; Sirui Chen; Xinquan Chen; Jie Cheng; Yu Cheng; Dengke Deng; Yizhuo Ding; Dan Ding; Xiaoshan Ding; Yi Ding; Zhichen Dong; Lingxiao Du; Yuyu Fan; Xinshun Feng; Yanwei Fu; Yuxuan Gao; Ruijun Ge; Tianle Gu; Lujun Gui; Jiaxuan Guo; Qianxi He; Yuenan Hou; Xuhao Hu; Hong Huang; Kaichen Huang; Shiyang Huang; Yuxian Jiang; Shanzhe Lei; Jie Li; Lijun Li; Hao Li; Juncheng Li; Xiangtian Li; Yafu Li; Lingyu Li; Xueyan Li; Haotian Liang; Dongrui Liu; Qihua Liu; Zhixuan Liu; Bangwei Liu; Huacan Liu; Yuexiao Liu; Zongkai Liu; Chaochao Lu; Yudong Lu; Xiaoya Lu; Zhenghao Lu; Qitan Lv; Caoyuan Ma; Jiachen Ma; Xiaoya Ma; Zhongtian Ma; Lingyu Meng; Ziqi Miao; Yazhe Niu; Yuezhang Peng; Yuan Pu; Han Qi; Chen Qian; Xingge Qiao; Jingjing Qu; Jiashu Qu; Wanying Qu; Wenwen Qu; Xiaoye Qu; Qihan Ren; Qingnan Ren; Qingyu Ren; Jing Shao; Wenqi Shao; Shuai Shao; Dongxing Shi; Xin Song; Xinhao Song; Yan Teng; Xuan Tong; Yingchun Wang; Xuhong Wang; Shujie Wang; Xin Wang; Yige Wang; Yixu Wang; Yuanfu Wang; Futing Wang; Ruofan Wang; Wenjie Wang; Yajie Wang; Muhao Wei; Xiaoyu Wen; Fenghua Weng; Yuqi Wu; Yingtong Xiong; Xingcheng Xu; Chao Yang; Yue Yang; Yang Yao; Yulei Ye; Zhenyun Yin; Yi Yu; Bo Zhang; Qiaosheng Zhang; Jinxuan Zhang; Yexin Zhang; Yinqiang Zheng; Hefeng Zhou; Zhanhui Zhou; Pengyu Zhu; Qingzi Zhu; Yubo Zhu; Bowen Zhou
>
> **备注:** 47 pages, 18 figures, authors are listed in alphabetical order by their last names; v3 modifies minor issues
>
> **摘要:** We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI.
>
---
#### [replaced 015] Stealthy Patch-Wise Backdoor Attack in 3D Point Cloud via Curvature Awareness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09336v2](http://arxiv.org/pdf/2503.09336v2)**

> **作者:** Yu Feng; Dingxin Zhang; Runkai Zhao; Yong Xia; Heng Huang; Weidong Cai
>
> **备注:** 11 pages, 8 figures, 8 tables
>
> **摘要:** Backdoor attacks pose a severe threat to deep neural networks (DNNs) by implanting hidden backdoors that can be activated with predefined triggers to manipulate model behaviors maliciously. Existing 3D point cloud backdoor attacks primarily rely on sample-wise global modifications, which suffer from low imperceptibility. Although optimization can improve stealthiness, optimizing sample-wise triggers significantly increases computational cost. To address these limitations, we propose the Stealthy Patch-Wise Backdoor Attack (SPBA), the first patch-wise backdoor attack framework for 3D point clouds. Specifically, SPBA decomposes point clouds into local patches and employs a curvature-based imperceptibility score to guide trigger injection into visually less sensitive patches. By optimizing a unified patch-wise trigger that perturbs spectral features of selected patches, SPBA significantly enhances optimization efficiency while maintaining high stealthiness. Extensive experiments on ModelNet40 and ShapeNetPart further demonstrate that SPBA surpasses prior state-of-the-art backdoor attacks in both attack effectiveness and resistance to defense methods.
>
---
#### [replaced 016] Can Vision Language Models Understand Mimed Actions?
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21586v2](http://arxiv.org/pdf/2506.21586v2)**

> **作者:** Hyundong Cho; Spencer Lin; Tejas Srinivasan; Michael Saxon; Deuksin Kwon; Natali T. Chavez; Jonathan May
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Nonverbal communication (NVC) plays an integral role in human language, but studying NVC in general is challenging because of its broad scope and high variance in interpretation among individuals and cultures. However, mime -- the theatrical technique of suggesting intent using only gesture, expression, and movement -- is a subset of NVC that consists of explicit and embodied actions with much lower human interpretation variance. We argue that a solid understanding of mimed actions is a crucial prerequisite for vision-language models capable of interpreting and commanding more subtle aspects of NVC. Hence, we propose Mime Identification Multimodal Evaluation (MIME), a novel video-based question answering benchmark comprising of 86 mimed actions. Constructed with motion capture data, MIME consists of variations of each action with perturbations applied to the character, background, and viewpoint for evaluating recognition robustness. We find that both open-weight and API-based vision-language models perform significantly worse than humans on MIME, motivating the need for increased research for instilling more robust understanding of human gestures.
>
---
#### [replaced 017] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.15451v3](http://arxiv.org/pdf/2503.15451v3)**

> **作者:** Lixing Xiao; Shunlin Lu; Huaijin Pi; Ke Fan; Liang Pan; Yueer Zhou; Ziyong Feng; Xiaowei Zhou; Sida Peng; Jingbo Wang
>
> **备注:** ICCV 2025. Project Page: https://zju3dv.github.io/MotionStreamer/
>
> **摘要:** This paper addresses the challenge of text-conditioned streaming motion generation, which requires us to predict the next-step human pose based on variable-length historical motions and incoming texts. Existing methods struggle to achieve streaming motion generation, e.g., diffusion models are constrained by pre-defined motion lengths, while GPT-based methods suffer from delayed response and error accumulation problem due to discretized non-causal tokenization. To solve these problems, we propose MotionStreamer, a novel framework that incorporates a continuous causal latent space into a probabilistic autoregressive model. The continuous latents mitigate information loss caused by discretization and effectively reduce error accumulation during long-term autoregressive generation. In addition, by establishing temporal causal dependencies between current and historical motion latents, our model fully utilizes the available information to achieve accurate online motion decoding. Experiments show that our method outperforms existing approaches while offering more applications, including multi-round generation, long-term generation, and dynamic motion composition. Project Page: https://zju3dv.github.io/MotionStreamer/
>
---
#### [replaced 018] CLIP Meets Diffusion: A Synergistic Approach to Anomaly Detection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.11772v2](http://arxiv.org/pdf/2506.11772v2)**

> **作者:** Byeongchan Lee; John Won; Seunghyun Lee; Jinwoo Shin
>
> **摘要:** Anomaly detection is a complex problem due to the ambiguity in defining anomalies, the diversity of anomaly types (e.g., local and global defect), and the scarcity of training data. As such, it necessitates a comprehensive model capable of capturing both low-level and high-level features, even with limited data. To address this, we propose CLIPFUSION, a method that leverages both discriminative and generative foundation models. Specifically, the CLIP-based discriminative model excels at capturing global features, while the diffusion-based generative model effectively captures local details, creating a synergistic and complementary approach. Notably, we introduce a methodology for utilizing cross-attention maps and feature maps extracted from diffusion models specifically for anomaly detection. Experimental results on benchmark datasets (MVTec-AD, VisA) demonstrate that CLIPFUSION consistently outperforms baseline methods, achieving outstanding performance in both anomaly segmentation and classification. We believe that our method underscores the effectiveness of multi-modal and multi-model fusion in tackling the multifaceted challenges of anomaly detection, providing a scalable solution for real-world applications.
>
---
#### [replaced 019] DisCoRD: Discrete Tokens to Continuous Motion via Rectified Flow Decoding
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19527v4](http://arxiv.org/pdf/2411.19527v4)**

> **作者:** Jungbin Cho; Junwan Kim; Jisoo Kim; Minseo Kim; Mingu Kang; Sungeun Hong; Tae-Hyun Oh; Youngjae Yu
>
> **备注:** 11 pages
>
> **摘要:** Human motion is inherently continuous and dynamic, posing significant challenges for generative models. While discrete generation methods are widely used, they suffer from limited expressiveness and frame-wise noise artifacts. In contrast, continuous approaches produce smoother, more natural motion but often struggle to adhere to conditioning signals due to high-dimensional complexity and limited training data. To resolve this 'discord' between discrete and continuous representations we introduce DisCoRD: Discrete Tokens to Continuous Motion via Rectified Flow Decoding, a novel method that leverages rectified flow to decode discrete motion tokens in the continuous, raw motion space. Our core idea is to frame token decoding as a conditional generation task, ensuring that DisCoRD captures fine-grained dynamics and achieves smoother, more natural motions. Compatible with any discrete-based framework, our method enhances naturalness without compromising faithfulness to the conditioning signals on diverse settings. Extensive evaluations demonstrate that DisCoRD achieves state-of-the-art performance, with FID of 0.032 on HumanML3D and 0.169 on KIT-ML. These results establish DisCoRD as a robust solution for bridging the divide between discrete efficiency and continuous realism. Project website: https://whwjdqls.github.io/discord-motion/
>
---
#### [replaced 020] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2507.12714v2](http://arxiv.org/pdf/2507.12714v2)**

> **作者:** Yang Yang; Dongni Mao; Hiroaki Santo; Yasuyuki Matsushita; Fumio Okura
>
> **备注:** IEEE/CVF International Conference on Computer Vision (ICCV 2025), Highlight, Project: https://neuraleaf-yang.github.io/
>
> **摘要:** We develop a neural parametric model for 3D leaves for plant modeling and reconstruction that are essential for agriculture and computer graphics. While neural parametric models are actively studied for humans and animals, plant leaves present unique challenges due to their diverse shapes and flexible deformation. To this problem, we introduce a neural parametric model for leaves, NeuraLeaf. Capitalizing on the fact that flattened leaf shapes can be approximated as a 2D plane, NeuraLeaf disentangles the leaves' geometry into their 2D base shapes and 3D deformations. This representation allows learning from rich sources of 2D leaf image datasets for the base shapes, and also has the advantage of simultaneously learning textures aligned with the geometry. To model the 3D deformation, we propose a novel skeleton-free skinning model and create a newly captured 3D leaf dataset called DeformLeaf. We show that NeuraLeaf successfully generates a wide range of leaf shapes with deformation, resulting in accurate model fitting to 3D observations like depth maps and point clouds. Our implementation and dataset are available at https://neuraleaf-yang.github.io/.
>
---
#### [replaced 021] RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15894v3](http://arxiv.org/pdf/2502.15894v3)**

> **作者:** Min Zhao; Guande He; Yixiao Chen; Hongzhou Zhu; Chongxuan Li; Jun Zhu
>
> **备注:** ICML 2025. Project page: https://riflex-video.github.io/
>
> **摘要:** Recent advancements in video generation have enabled models to synthesize high-quality, minute-long videos. However, generating even longer videos with temporal coherence remains a major challenge and existing length extrapolation methods lead to temporal repetition or motion deceleration. In this work, we systematically analyze the role of frequency components in positional embeddings and identify an intrinsic frequency that primarily governs extrapolation behavior. Based on this insight, we propose RIFLEx, a minimal yet effective approach that reduces the intrinsic frequency to suppress repetition while preserving motion consistency, without requiring any additional modifications. RIFLEx offers a true free lunch--achieving high-quality 2x extrapolation on state-of-the-art video diffusion transformers in a completely training-free manner. Moreover, it enhances quality and enables 3x extrapolation by minimal fine-tuning without long videos. Project page and codes: https://riflex-video.github.io/.
>
---
#### [replaced 022] Learning Only with Images: Visual Reinforcement Learning with Reasoning, Rendering, and Visual Feedback
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20766v4](http://arxiv.org/pdf/2507.20766v4)**

> **作者:** Yang Chen; Yufan Shen; Wenxuan Huang; Sheng Zhou; Qunshu Lin; Xinyu Cai; Zhi Yu; Jiajun Bu; Botian Shi; Yu Qiao
>
> **摘要:** Multimodal Large Language Models (MLLMs) exhibit impressive performance across various visual tasks. Subsequent investigations into enhancing their visual reasoning abilities have significantly expanded their performance envelope. However, a critical bottleneck in the advancement of MLLMs toward deep visual reasoning is their heavy reliance on curated image-text supervision. To solve this problem, we introduce a novel framework, ``Reasoning-Rendering-Visual-Feedback'' (RRVF), that enables MLLMs to learn complex visual reasoning from only raw images. This framework builds on the ``Asymmetry of Verification'' principle, i.e., verifying the rendered output against the source image is substantially easier than performing deep visual reasoning to generate a faithful, structured representation such as code. We demonstrate that this relative ease provides an ideal reward signal for optimization via Reinforcement Learning (RL), thereby reducing reliance on image-text supervision. RRVF implements a closed-loop iterative process encompassing reasoning, rendering, and visual feedback components, enabling the model to perform complex reasoning, including self-correction through multi-turn interactions. This process is optimized end-to-end using the GRPO algorithm. Extensive evaluations are conducted on image-to-code generation across two diverse domains: data charts and web interfaces. The RRVF-trained model not only outperforms existing similarly sized open-source MLLMs and supervised fine-tuning baselines but also exhibits superior generalization. Notably, the model outperforms the more advanced MLLM used to generate visual feedback during training. Code is available at https://github.com/L-O-I/RRVF.
>
---
#### [replaced 023] TIME: Temporal-Sensitive Multi-Dimensional Instruction Tuning and Robust Benchmarking for Video-LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09994v2](http://arxiv.org/pdf/2503.09994v2)**

> **作者:** Yunxiao Wang; Meng Liu; Wenqi Liu; Xuemeng Song; Bin Wen; Fan Yang; Tingting Gao; Di Zhang; Guorui Zhou; Liqiang Nie
>
> **摘要:** Video large language models have achieved remarkable performance in tasks such as video question answering, however, their temporal understanding remains suboptimal. To address this limitation, we curate a dedicated instruction fine-tuning dataset that focuses on enhancing temporal comprehension across five key dimensions. In order to reduce reliance on costly temporal annotations, we introduce a multi-task prompt fine-tuning approach that seamlessly integrates temporal-sensitive tasks into existing instruction datasets without requiring additional annotations. Furthermore, we develop a novel benchmark for temporal-sensitive video understanding that not only fills the gaps in dimension coverage left by existing benchmarks but also rigorously filters out potential shortcuts, ensuring a more accurate evaluation. Extensive experimental results demonstrate that our approach significantly enhances the temporal understanding of video-LLMs while avoiding reliance on shortcuts.
>
---
#### [replaced 024] Magic Fixup: Streamlining Photo Editing by Watching Dynamic Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.13044v2](http://arxiv.org/pdf/2403.13044v2)**

> **作者:** Hadi Alzayer; Zhihao Xia; Xuaner Zhang; Eli Shechtman; Jia-Bin Huang; Michael Gharbi
>
> **备注:** ACM Transactions on Graphics 2025. Project page: https://magic-fixup.github.io/
>
> **摘要:** We propose a generative model that, given a coarsely edited image, synthesizes a photorealistic output that follows the prescribed layout. Our method transfers fine details from the original image and preserve the identity of its parts. Yet, it adapts it to the lighting and context defined by the new layout. Our key insight is that videos are a powerful source of supervision for this task: objects and camera motions provide many observations of how the world changes with viewpoint, lighting, and physical interactions. We construct an image dataset in which each sample is a pair of source and target frames extracted from the same video at randomly chosen time intervals. We warp the source frame toward the target using two motion models that mimic the expected test-time user edits. We supervise our model to translate the warped image into the ground truth, starting from a pretrained diffusion model. Our model design explicitly enables fine detail transfer from the source frame to the generated image, while closely following the user-specified layout. We show that by using simple segmentations and coarse 2D manipulations, we can synthesize a photorealistic edit faithful to the user's input while addressing second-order effects like harmonizing the lighting and physical interactions between edited objects.
>
---
#### [replaced 025] Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19616v2](http://arxiv.org/pdf/2505.19616v2)**

> **作者:** Rui Cai; Bangzheng Li; Xiaofei Wen; Muhao Chen; Zhe Zhao
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across tasks, yet they often exhibit difficulty in distinguishing task-relevant from irrelevant signals, particularly in tasks like Visual Question Answering (VQA), which can lead to susceptibility to misleading or spurious inputs. We refer to this broader limitation as the Cross-Modality Competency Problem: the model's inability to fairly evaluate all modalities. This vulnerability becomes more evident in modality-specific tasks such as image classification or pure text question answering, where models are expected to rely solely on one modality. In such tasks, spurious information from irrelevant modalities often leads to significant performance degradation. We refer to this failure as Modality Interference, which serves as a concrete and measurable instance of the cross-modality competency problem. We further design a perturbation-based causal diagnostic experiment to verify and quantify this problem. To mitigate modality interference, we propose a novel framework to fine-tune MLLMs, including perturbation-based data augmentations with both heuristic perturbations and adversarial perturbations via Projected Gradient Descent (PGD), and a consistency regularization strategy applied to model outputs with original and perturbed inputs. Experiments on multiple benchmark datasets (image-heavy, text-heavy, and VQA tasks) and multiple model families with different scales demonstrate significant improvements in robustness and cross-modality competency, indicating our method's effectiveness in boosting unimodal reasoning ability while enhancing performance on multimodal tasks.
>
---
#### [replaced 026] Exploring the Feasibility of Deep Learning Techniques for Accurate Gender Classification from Eye Images
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00135v2](http://arxiv.org/pdf/2508.00135v2)**

> **作者:** Basna Mohammed Salih Hasan; Ramadhan J. Mstafa
>
> **备注:** 12 pages, 18 figures, 5 tables
>
> **摘要:** Gender classification has emerged as a crucial aspect in various fields, including security, human-machine interaction, surveillance, and advertising. Nonetheless, the accuracy of this classification can be influenced by factors such as cosmetics and disguise. Consequently, our study is dedicated to addressing this concern by concentrating on gender classification using color images of the periocular region. The periocular region refers to the area surrounding the eye, including the eyelids, eyebrows, and the region between them. It contains valuable visual cues that can be used to extract key features for gender classification. This paper introduces a sophisticated Convolutional Neural Network (CNN) model that utilizes color image databases to evaluate the effectiveness of the periocular region for gender classification. To validate the model's performance, we conducted tests on two eye datasets, namely CVBL and (Female and Male). The recommended architecture achieved an outstanding accuracy of 99% on the previously unused CVBL dataset while attaining a commendable accuracy of 96% with a small number of learnable parameters (7,235,089) on the (Female and Male) dataset. To ascertain the effectiveness of our proposed model for gender classification using the periocular region, we evaluated its performance through an extensive range of metrics and compared it with other state-of-the-art approaches. The results unequivocally demonstrate the efficacy of our model, thereby suggesting its potential for practical application in domains such as security and surveillance.
>
---
#### [replaced 027] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.15877v2](http://arxiv.org/pdf/2503.15877v2)**

> **作者:** Tiange Xiang; Kai Li; Chengjiang Long; Christian Häne; Peihong Guo; Scott Delp; Ehsan Adeli; Li Fei-Fei
>
> **备注:** ICCV 2025
>
> **摘要:** Recent advances in text-to-image diffusion models have been driven by the increasing availability of paired 2D data. However, the development of 3D diffusion models has been hindered by the scarcity of high-quality 3D data, resulting in less competitive performance compared to their 2D counterparts. To address this challenge, we propose repurposing pre-trained 2D diffusion models for 3D object generation. We introduce Gaussian Atlas, a novel representation that utilizes dense 2D grids, enabling the fine-tuning of 2D diffusion models to generate 3D Gaussians. Our approach demonstrates successful transfer learning from a pre-trained 2D diffusion model to a 2D manifold flattened from 3D structures. To support model training, we compile GaussianVerse, a large-scale dataset comprising 205K high-quality 3D Gaussian fittings of various 3D objects. Our experimental results show that text-to-image diffusion models can be effectively adapted for 3D content generation, bridging the gap between 2D and 3D modeling.
>
---
#### [replaced 028] ESVQA: Perceptual Quality Assessment of Egocentric Spatial Videos
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.20423v2](http://arxiv.org/pdf/2412.20423v2)**

> **作者:** Xilei Zhu; Huiyu Duan; Liu Yang; Yucheng Zhu; Xiongkuo Min; Guangtao Zhai; Patrick Le Callet
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** With the rapid development of eXtended Reality (XR), egocentric spatial shooting and display technologies have further enhanced immersion and engagement for users, delivering more captivating and interactive experiences. Assessing the quality of experience (QoE) of egocentric spatial videos is crucial to ensure a high-quality viewing experience. However, the corresponding research is still lacking. In this paper, we use the concept of embodied experience to highlight this more immersive experience and study the new problem, i.e., embodied perceptual quality assessment for egocentric spatial videos. Specifically, we introduce the first Egocentric Spatial Video Quality Assessment Database (ESVQAD), which comprises 600 egocentric spatial videos captured using the Apple Vision Pro and their corresponding mean opinion scores (MOSs). Furthermore, we propose a novel multi-dimensional binocular feature fusion model, termed ESVQAnet, which integrates binocular spatial, motion, and semantic features to predict the overall perceptual quality. Experimental results demonstrate the ESVQAnet significantly outperforms 16 state-of-the-art VQA models on the embodied perceptual quality assessment task, and exhibits strong generalization capability on traditional VQA tasks. The database and code are available at https://github.com/iamazxl/ESVQA.
>
---
#### [replaced 029] CM-Diff: A Single Generative Network for Bidirectional Cross-Modality Translation Diffusion Model Between Infrared and Visible Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09514v2](http://arxiv.org/pdf/2503.09514v2)**

> **作者:** Bin Hu; Chenqiang Gao; Shurui Liu; Junjie Guo; Fang Chen; Fangcen Liu; Junwei Han
>
> **摘要:** Image translation is one of the crucial approaches for mitigating information deficiencies in the infrared and visible modalities, while also facilitating the enhancement of modality-specific datasets. However, existing methods for infrared and visible image translation either achieve unidirectional modality translation or rely on cycle consistency for bidirectional modality translation, which may result in suboptimal performance. In this work, we present the bidirectional cross-modality translation diffusion model (CM-Diff) for simultaneously modeling data distributions in both the infrared and visible modalities. We address this challenge by combining translation direction labels for guidance during training with cross-modality feature control. Specifically, we view the establishment of the mapping relationship between the two modalities as the process of learning data distributions and understanding modality differences, achieved through a novel Bidirectional Diffusion Training (BDT). Additionally, we propose a Statistical Constraint Inference (SCI) to ensure the generated image closely adheres to the data distribution of the target modality. Experimental results demonstrate the superiority of our CM-Diff over state-of-the-art methods, highlighting its potential for generating dual-modality datasets.
>
---
#### [replaced 030] M$^{2}$Chat: Empowering VLM for Multimodal LLM Interleaved Text-Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.17963v3](http://arxiv.org/pdf/2311.17963v3)**

> **作者:** Xiaowei Chi; Junbo Qi; Rongyu Zhang; Shanghang Zhang; Qifeng Liu; Yike Guo
>
> **摘要:** While current LLM chatbots like GPT-4V bridge the gap between human instructions and visual representations to enable text-image generations, they still lack efficient alignment methods for high-fidelity performance on multiple downstream tasks. In this paper, we propose \textbf{$M^{2}Chat$}, a novel unified multimodal LLM framework for generating interleaved text-image conversation across various scenarios. Specifically, we propose an $M^{3}Adapter$ that efficiently integrates granular low-level visual information and high-level semantic features from multi-modality prompts. Upon the well-aligned fused feature, $M^{3}Adapter$ tailors a learnable gating strategy to balance the model creativity and consistency across various tasks adaptively. Moreover, to further enhance the effectiveness of $M^{3}Adapter$ while preserving the coherence of semantic context comprehension, we introduce a two-stage $M^{3}FT$ fine-tuning strategy. This strategy optimizes disjoint groups of parameters for image-text alignment and visual-instruction respectively. Extensive experiments demonstrate our $M^{2}Chat$ surpasses state-of-the-art counterparts across diverse benchmarks, showcasing its prowess in interleaving generation, storytelling, and multimodal dialogue systems. The demo and code are available at \red{https://mattie-e.github.io/M2Chat.github.io}.
>
---
#### [replaced 031] Towards Scalable Newborn Screening: Automated General Movement Assessment in Uncontrolled Settings
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.09821v4](http://arxiv.org/pdf/2411.09821v4)**

> **作者:** Daphné Chopard; Sonia Laguna; Kieran Chin-Cheong; Annika Dietz; Anna Badura; Sven Wellmann; Julia E. Vogt
>
> **备注:** Paper at Proceedings of Machine Learning Research 298 1 22, 2025 Machine Learning for Healthcare. Iterations of previous versions accepted as oral and best paper award at ICLR 2025 Workshop on AI for Children and at the Findings track of the Machine Learning for Health (ML4H) symposium 2024
>
> **摘要:** General movements (GMs) are spontaneous, coordinated body movements in infants that offer valuable insights into the developing nervous system. Assessed through the Prechtl GM Assessment (GMA), GMs are reliable predictors for neurodevelopmental disorders. However, GMA requires specifically trained clinicians, who are limited in number. To scale up newborn screening, there is a need for an algorithm that can automatically classify GMs from infant video recordings. This data poses challenges, including variability in recording length, device type, and setting, with each video coarsely annotated for overall movement quality. In this work, we introduce a tool for extracting features from these recordings and explore various machine learning techniques for automated GM classification.
>
---
#### [replaced 032] Viewpoint Consistency in 3D Generation via Attention and CLIP Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02287v3](http://arxiv.org/pdf/2412.02287v3)**

> **作者:** Qing Zhang; Zehao Chen; Jinguang Tong; Jing Zhang; Jie Hong; Xuesong Li
>
> **摘要:** Despite recent advances in text-to-3D generation techniques, current methods often suffer from geometric inconsistencies, commonly referred to as the Janus Problem. This paper identifies the root cause of the Janus Problem: viewpoint generation bias in diffusion models, which creates a significant gap between the actual generated viewpoint and the expected one required for optimizing the 3D model. To address this issue, we propose a tuning-free approach called the Attention and CLIP Guidance (ACG) mechanism. ACG enhances desired viewpoints by adaptively controlling cross-attention maps, employs CLIP-based view-text similarities to filter out erroneous viewpoints, and uses a coarse-to-fine optimization strategy with staged prompts to progressively refine 3D generation. Extensive experiments demonstrate that our method significantly reduces the Janus Problem without compromising generation speed, establishing ACG as an efficient, plug-and-play component for existing text-to-3D frameworks.
>
---
#### [replaced 033] DepthSync: Diffusion Guidance-Based Depth Synchronization for Scale- and Geometry-Consistent Video Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01603v2](http://arxiv.org/pdf/2507.01603v2)**

> **作者:** Yue-Jiang Dong; Wang Zhao; Jiale Xu; Ying Shan; Song-Hai Zhang
>
> **备注:** Accepted by ICCV 2025; Project Homepage: https://yuejiangdong.github.io/depthsync
>
> **摘要:** Diffusion-based video depth estimation methods have achieved remarkable success with strong generalization ability. However, predicting depth for long videos remains challenging. Existing methods typically split videos into overlapping sliding windows, leading to accumulated scale discrepancies across different windows, particularly as the number of windows increases. Additionally, these methods rely solely on 2D diffusion priors, overlooking the inherent 3D geometric structure of video depths, which results in geometrically inconsistent predictions. In this paper, we propose DepthSync, a novel, training-free framework using diffusion guidance to achieve scale- and geometry-consistent depth predictions for long videos. Specifically, we introduce scale guidance to synchronize the depth scale across windows and geometry guidance to enforce geometric alignment within windows based on the inherent 3D constraints in video depths. These two terms work synergistically, steering the denoising process toward consistent depth predictions. Experiments on various datasets validate the effectiveness of our method in producing depth estimates with improved scale and geometry consistency, particularly for long videos.
>
---
#### [replaced 034] Computation-Efficient and Recognition-Friendly 3D Point Cloud Privacy Protection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15818v3](http://arxiv.org/pdf/2503.15818v3)**

> **作者:** Haotian Ma; Lin Gu; Siyi Wu; Yingying Zhu
>
> **备注:** This paper was submitted without the knowledge or consent of all co-authors, which violates arXiv submission policies
>
> **摘要:** 3D point cloud has been widely used in applications such as self-driving cars, robotics, CAD models, etc. To the best of our knowledge, these applications raised the issue of privacy leakage in 3D point clouds, which has not been studied well. Different from the 2D image privacy, which is related to texture and 2D geometric structure, the 3D point cloud is texture-less and only relevant to 3D geometric structure. In this work, we defined the 3D point cloud privacy problem and proposed an efficient privacy-preserving framework named PointFlowGMM that can support downstream classification and segmentation tasks without seeing the original data. Using a flow-based generative model, the point cloud is projected into a latent Gaussian mixture distributed subspace. We further designed a novel angular similarity loss to obfuscate the original geometric structure and reduce the model size from 767MB to 120MB without a decrease in recognition performance. The projected point cloud in the latent space is orthogonally rotated randomly to further protect the original geometric structure, the class-to-class relationship is preserved after rotation, thus, the protected point cloud can support the recognition task. We evaluated our model on multiple datasets and achieved comparable recognition results on encrypted point clouds compared to the original point clouds.
>
---
#### [replaced 035] Follow-Your-Color: Multi-Instance Sketch Colorization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16948v2](http://arxiv.org/pdf/2503.16948v2)**

> **作者:** Yinhan Zhang; Yue Ma; Bingyuan Wang; Qifeng Chen; Zeyu Wang
>
> **摘要:** We present Follow-Your-Color, a diffusion-based framework for multi-instance sketch colorization. The production of multi-instance 2D line art colorization adheres to an industry-standard workflow, which consists of three crucial stages: the design of line art characters, the coloring of individual objects, and the refinement process. The artists are required to repeat the process of coloring each instance one by one, which is inaccurate and inefficient. Meanwhile, current generative methods fail to solve this task due to the challenge of multi-instance pair data collection. To tackle these challenges, we incorporate three technical designs to ensure precise character detail transcription and achieve multi-instance sketch colorization in a single forward pass. Specifically, we first propose the self-play training strategy to address the lack of training data. Then we introduce an instance guider to feed the color of the instance. To achieve accurate color matching, we present fine-grained color matching with edge loss to enhance visual quality. Equipped with the proposed modules, Follow-Your-Color enables automatically transforming sketches into vividly-colored images with accurate consistency and multi-instance control. Experiments on our collected datasets show that our model outperforms existing methods regarding chromatic precision. Specifically, our model critically automates the colorization process with zero manual adjustments, so novice users can produce stylistically consistent artwork by providing reference instances and the original line art. Our code and additional details are available at https://yinhan-zhang.github.io/color.
>
---
#### [replaced 036] Cross-Image Contrastive Decoding: Precise, Lossless Suppression of Language Priors in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10634v4](http://arxiv.org/pdf/2505.10634v4)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng
>
> **备注:** Under Review
>
> **摘要:** Over-reliance on language priors is a major cause of hallucinations in Large Vision-Language Models (LVLMs), often leading to outputs that are linguistically plausible but visually inconsistent. Recent studies have explored contrastive decoding as a training-free solution. However, these methods typically construct contrastive visual inputs by perturbing the original image, resulting in distorted contrastive distributions, incomplete contrastive signals, and excessive suppression of language priors. Motivated by the observation that language priors tend to remain consistent across different images, we propose Cross-Image Contrastive Decoding (CICD), a simple yet effective training-free method that uses unrelated images as contrastive visual inputs. To address the issue of over-suppressing language priors, which can negatively affect the quality of generated responses, we further introduce a dynamic selection mechanism based on the cross-image differences in model behavior. By selectively suppressing language priors, our method reduces hallucinations without compromising the model's performance. Extensive experiments across multiple benchmarks and LVLMs confirm the effectiveness and generalizability of CICD, particularly in image captioning, where language priors are especially dominant.
>
---
#### [replaced 037] Brain Network Analysis Based on Fine-tuned Self-supervised Model for Brain Disease Diagnosis
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11671v2](http://arxiv.org/pdf/2506.11671v2)**

> **作者:** Yifei Tang; Hongjie Jiang; Changhong Jing; Hieu Pham; Shuqiang Wang
>
> **备注:** 13 pages, 3 figures, International Conference on Neural Computing for Advanced Applications
>
> **摘要:** Functional brain network analysis has become an indispensable tool for brain disease analysis. It is profoundly impacted by deep learning methods, which can characterize complex connections between ROIs. However, the research on foundation models of brain network is limited and constrained to a single dimension, which restricts their extensive application in neuroscience. In this study, we propose a fine-tuned brain network model for brain disease diagnosis. It expands brain region representations across multiple dimensions based on the original brain network model, thereby enhancing its generalizability. Our model consists of two key modules: (1)an adapter module that expands brain region features across different dimensions. (2)a fine-tuned foundation brain network model, based on self-supervised learning and pre-trained on fMRI data from thousands of participants. Specifically, its transformer block is able to effectively extract brain region features and compute the inter-region associations. Moreover, we derive a compact latent representation of the brain network for brain disease diagnosis. Our downstream experiments in this study demonstrate that the proposed model achieves superior performance in brain disease diagnosis, which potentially offers a promising approach in brain network analysis research.
>
---
#### [replaced 038] Diffusion Beats Autoregressive in Data-Constrained Settings
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15857v5](http://arxiv.org/pdf/2507.15857v5)**

> **作者:** Mihir Prabhudesai; Mengning Wu; Amir Zadeh; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project Webpage: https://diffusion-scaling.github.io
>
> **摘要:** Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: https://diffusion-scaling.github.io.
>
---
#### [replaced 039] STARFormer: A Novel Spatio-Temporal Aggregation Reorganization Transformer of FMRI for Brain Disorder Diagnosis
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00378v2](http://arxiv.org/pdf/2501.00378v2)**

> **作者:** Wenhao Dong; Yueyang Li; Weiming Zeng; Lei Chen; Hongjie Yan; Wai Ting Siok; Nizhuan Wang
>
> **摘要:** Many existing methods that use functional magnetic resonance imaging (fMRI) classify brain disorders, such as autism spectrum disorder (ASD) and attention deficit hyperactivity disorder (ADHD), often overlook the integration of spatial and temporal dependencies of the blood oxygen level-dependent (BOLD) signals, which may lead to inaccurate or imprecise classification results. To solve this problem, we propose a Spatio-Temporal Aggregation eorganization ransformer (STARFormer) that effectively captures both spatial and temporal features of BOLD signals by incorporating three key modules. The region of interest (ROI) spatial structure analysis module uses eigenvector centrality (EC) to reorganize brain regions based on effective connectivity, highlighting critical spatial relationships relevant to the brain disorder. The temporal feature reorganization module systematically segments the time series into equal-dimensional window tokens and captures multiscale features through variable window and cross-window attention. The spatio-temporal feature fusion module employs a parallel transformer architecture with dedicated temporal and spatial branches to extract integrated features. The proposed STARFormer has been rigorously evaluated on two publicly available datasets for the classification of ASD and ADHD. The experimental results confirm that the STARFormer achieves state-of-the-art performance across multiple evaluation metrics, providing a more accurate and reliable tool for the diagnosis of brain disorders and biomedical research. The codes are available at: https://github.com/NZWANG/STARFormer.
>
---
#### [replaced 040] RoboTron-Drive: All-in-One Large Multimodal Model for Autonomous Driving
- **分类: cs.CV; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.07689v5](http://arxiv.org/pdf/2412.07689v5)**

> **作者:** Zhijian Huang; Chengjian Feng; Feng Yan; Baihui Xiao; Zequn Jie; Yujie Zhong; Xiaodan Liang; Lin Ma
>
> **备注:** ICCV 2025
>
> **摘要:** Large Multimodal Models (LMMs) have demonstrated exceptional comprehension and interpretation capabilities in Autonomous Driving (AD) by incorporating large language models. Despite the advancements, current data-driven AD approaches tend to concentrate on a single dataset and specific tasks, neglecting their overall capabilities and ability to generalize. To bridge these gaps, we propose RoboTron-Drive, a general large multimodal model designed to process diverse data inputs, such as images and multi-view videos, while performing a broad spectrum of AD tasks, including perception, prediction, and planning. Initially, the model undergoes curriculum pre-training to process varied visual signals and perform basic visual comprehension and perception tasks. Subsequently, we augment and standardize various AD datasets to finetune the model, resulting in an all-in-one LMM for autonomous driving. To assess the general capabilities and generalization ability, we conduct evaluations on six public benchmarks and undertake zero-shot transfer on three unseen datasets, where RoboTron-Drive achieves state-of-the-art performance across all tasks. We hope RoboTron-Drive as a promising solution for AD in the real world. Project page with code: https://github.com/zhijian11/RoboTron-Drive.
>
---
#### [replaced 041] MOGO: Residual Quantized Hierarchical Causal Transformer for High-Quality and Real-Time 3D Human Motion Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05952v2](http://arxiv.org/pdf/2506.05952v2)**

> **作者:** Dongjie Fu; Tengjiao Sun; Pengcheng Fang; Xiaohao Cai; Hansung Kim
>
> **备注:** 9 pages, 4 figures, conference
>
> **摘要:** Recent advances in transformer-based text-to-motion generation have led to impressive progress in synthesizing high-quality human motion. Nevertheless, jointly achieving high fidelity, streaming capability, real-time responsiveness, and scalability remains a fundamental challenge. In this paper, we propose MOGO (Motion Generation with One-pass), a novel autoregressive framework tailored for efficient and real-time 3D motion generation. MOGO comprises two key components: (1) MoSA-VQ, a motion scale-adaptive residual vector quantization module that hierarchically discretizes motion sequences with learnable scaling to produce compact yet expressive representations; and (2) RQHC-Transformer, a residual quantized hierarchical causal transformer that generates multi-layer motion tokens in a single forward pass, significantly reducing inference latency. To enhance semantic fidelity, we further introduce a text condition alignment mechanism that improves motion decoding under textual control. Extensive experiments on benchmark datasets including HumanML3D, KIT-ML, and CMP demonstrate that MOGO achieves competitive or superior generation quality compared to state-of-the-art transformer-based methods, while offering substantial improvements in real-time performance, streaming generation, and generalization under zero-shot settings.
>
---
#### [replaced 042] A Fast Text-Driven Approach for Generating Artistic Content
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2208.01748v2](http://arxiv.org/pdf/2208.01748v2)**

> **作者:** Marian Lupascu; Ryan Murdock; Ionut Mironica; Yijun Li
>
> **备注:** 3 pages, 2 figures
>
> **摘要:** In this work, we propose a complete framework that generates visual art. Unlike previous stylization methods that are not flexible with style parameters (i.e., they allow stylization with only one style image, a single stylization text or stylization of a content image from a certain domain), our method has no such restriction. In addition, we implement an improved version that can generate a wide range of results with varying degrees of detail, style and structure, with a boost in generation speed. To further enhance the results, we insert an artistic super-resolution module in the generative pipeline. This module will bring additional details such as patterns specific to painters, slight brush marks, and so on.
>
---
#### [replaced 043] CARE: Enhancing Safety of Visual Navigation through Collision Avoidance via Repulsive Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03834v3](http://arxiv.org/pdf/2506.03834v3)**

> **作者:** Joonkyung Kim; Joonyeol Sim; Woojun Kim; Katia Sycara; Changjoo Nam
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** We propose CARE (Collision Avoidance via Repulsive Estimation) to improve the robustness of learning-based visual navigation methods. Recently, visual navigation models, particularly foundation models, have demonstrated promising performance by generating viable trajectories using only RGB images. However, these policies can generalize poorly to environments containing out-of-distribution (OOD) scenes characterized by unseen objects or different camera setups (e.g., variations in field of view, camera pose, or focal length). Without fine-tuning, such models could produce trajectories that lead to collisions, necessitating substantial efforts in data collection and additional training. To address this limitation, we introduce CARE, an attachable module that enhances the safety of visual navigation without requiring additional range sensors or fine-tuning of pretrained models. CARE can be integrated seamlessly into any RGB-based navigation model that generates local robot trajectories. It dynamically adjusts trajectories produced by a pretrained model using repulsive force vectors computed from depth images estimated directly from RGB inputs. We evaluate CARE by integrating it with state-of-the-art visual navigation models across diverse robot platforms. Real-world experiments show that CARE significantly reduces collisions (up to 100%) without compromising navigation performance in goal-conditioned navigation, and further improves collision-free travel distance (up to 10.7x) in exploration tasks. Project page: https://airlab-sogang.github.io/CARE/
>
---
#### [replaced 044] Evaluation of Safety Cognition Capability in Vision-Language Models for Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06497v2](http://arxiv.org/pdf/2503.06497v2)**

> **作者:** Enming Zhang; Peizhe Gong; Xingyuan Dai; Min Huang; Yisheng Lv; Qinghai Miao
>
> **摘要:** Ensuring the safety of vision-language models (VLMs) in autonomous driving systems is of paramount importance, yet existing research has largely focused on conventional benchmarks rather than safety-critical evaluation. In this work, we present SCD-Bench (Safety Cognition Driving Benchmark) a novel framework specifically designed to assess the safety cognition capabilities of VLMs within interactive driving scenarios. To address the scalability challenge of data annotation, we introduce ADA (Autonomous Driving Annotation), a semi-automated labeling system, further refined through expert review by professionals with domain-specific knowledge in autonomous driving. To facilitate scalable and consistent evaluation, we also propose an automated assessment pipeline leveraging large language models, which demonstrates over 98% agreement with human expert judgments. In addressing the broader challenge of aligning VLMs with safety cognition in driving environments, we construct SCD-Training, the first large-scale dataset tailored for this task, comprising 324.35K high-quality samples. Through extensive experiments, we show that models trained on SCD-Training exhibit marked improvements not only on SCD-Bench, but also on general and domain-specific benchmarks, offering a new perspective on enhancing safety-aware interactions in vision-language systems for autonomous driving.
>
---
#### [replaced 045] Towards Reliable Audio Deepfake Attribution and Model Recognition: A Multi-Level Autoencoder-Based Framework
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02521v2](http://arxiv.org/pdf/2508.02521v2)**

> **作者:** Andrea Di Pierno; Luca Guarnera; Dario Allegra; Sebastiano Battiato
>
> **摘要:** The proliferation of audio deepfakes poses a growing threat to trust in digital communications. While detection methods have advanced, attributing audio deepfakes to their source models remains an underexplored yet crucial challenge. In this paper we introduce LAVA (Layered Architecture for Voice Attribution), a hierarchical framework for audio deepfake detection and model recognition that leverages attention-enhanced latent representations extracted by a convolutional autoencoder trained solely on fake audio. Two specialized classifiers operate on these features: Audio Deepfake Attribution (ADA), which identifies the generation technology, and Audio Deepfake Model Recognition (ADMR), which recognize the specific generative model instance. To improve robustness under open-set conditions, we incorporate confidence-based rejection thresholds. Experiments on ASVspoof2021, FakeOrReal, and CodecFake show strong performance: the ADA classifier achieves F1-scores over 95% across all datasets, and the ADMR module reaches 96.31% macro F1 across six classes. Additional tests on unseen attacks from ASVpoof2019 LA and error propagation analysis confirm LAVA's robustness and reliability. The framework advances the field by introducing a supervised approach to deepfake attribution and model recognition under open-set conditions, validated on public benchmarks and accompanied by publicly released models and code. Models and code are available at https://www.github.com/adipiz99/lava-framework.
>
---
#### [replaced 046] Learning Disease State from Noisy Ordinal Disease Progression Labels
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10440v2](http://arxiv.org/pdf/2503.10440v2)**

> **作者:** Gustav Schmidt; Holger Heidrich; Philipp Berens; Sarah Müller
>
> **备注:** corrected Table 1
>
> **摘要:** Learning from noisy ordinal labels is a key challenge in medical imaging. In this work, we ask whether ordinal disease progression labels (better, worse, or stable) can be used to learn a representation allowing to classify disease state. For neovascular age-related macular degeneration (nAMD), we cast the problem of modeling disease progression between medical visits as a classification task with ordinal ranks. To enhance generalization, we tailor our model to the problem setting by (1) independent image encoding, (2) antisymmetric logit space equivariance, and (3) ordinal scale awareness. In addition, we address label noise by learning an uncertainty estimate for loss re-weighting. Our approach learns an interpretable disease representation enabling strong few-shot performance for the related task of nAMD activity classification from single images, despite being trained only on image pairs with ordinal disease progression labels.
>
---
#### [replaced 047] Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.16861v2](http://arxiv.org/pdf/2507.16861v2)**

> **作者:** Xiang Li; Zhangchi Hu; Xiao Xu; Bin Kong
>
> **摘要:** Integrating LiDAR and camera inputs into a unified Bird's-Eye-View (BEV) representation is crucial for enhancing 3D perception capabilities of autonomous vehicles. However, existing methods suffer from spatial misalignment between LiDAR and camera features, which causes inaccurate depth supervision in camera branch and erroneous fusion during cross-modal feature aggregation. The root cause of this misalignment lies in projection errors, stemming from calibration inaccuracies and rolling shutter effect. The key insight of this work is that locations of these projection errors are not random but highly predictable, as they are concentrated at object-background boundaries which 2D detectors can reliably identify. Based on this, our main motivation is to utilize 2D object priors to pre-align cross-modal features before fusion. To address local misalignment, we propose Prior Guided Depth Calibration (PGDC), which leverages 2D priors to alleviate misalignment and preserve correct cross-modal feature pairs. To resolve global misalignment, we introduce Discontinuity Aware Geometric Fusion (DAGF) to suppress residual noise from PGDC and explicitly enhance sharp depth transitions at object-background boundaries, yielding a structurally aware representation. To effectively utilize these aligned representations, we incorporate Structural Guidance Depth Modulator (SGDM), using a gated attention mechanism to efficiently fuse aligned depth and image features. Our method achieves SOTA performance on nuScenes validation dataset, with its mAP and NDS reaching 71.5% and 73.6% respectively
>
---
#### [replaced 048] S$^2$Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04016v2](http://arxiv.org/pdf/2508.04016v2)**

> **作者:** Weilun Feng; Haotong Qin; Chuanguang Yang; Xiangqi Li; Han Yang; Yuqi Li; Zhulin An; Libo Huang; Michele Magno; Yongjun Xu
>
> **摘要:** Diffusion transformers have emerged as the mainstream paradigm for video generation models. However, the use of up to billions of parameters incurs significant computational costs. Quantization offers a promising solution by reducing memory usage and accelerating inference. Nonetheless, we observe that the joint modeling of spatial and temporal information in video diffusion models (V-DMs) leads to extremely long token sequences, which introduces high calibration variance and learning challenges. To address these issues, we propose S$^2$Q-VDiT, a post-training quantization framework for V-DMs that leverages Salient data and Sparse token distillation. During the calibration phase, we identify that quantization performance is highly sensitive to the choice of calibration data. To mitigate this, we introduce \textit{Hessian-aware Salient Data Selection}, which constructs high-quality calibration datasets by considering both diffusion and quantization characteristics unique to V-DMs. To tackle the learning challenges, we further analyze the sparse attention patterns inherent in V-DMs. Based on this observation, we propose \textit{Attention-guided Sparse Token Distillation}, which exploits token-wise attention distributions to emphasize tokens that are more influential to the model's output. Under W4A6 quantization, S$^2$Q-VDiT achieves lossless performance while delivering $3.9\times$ model compression and $1.3\times$ inference acceleration. Code will be available at https://github.com/wlfeng0509/s2q-vdit.
>
---
#### [replaced 049] Sign Spotting Disambiguation using Large Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03703v4](http://arxiv.org/pdf/2507.03703v4)**

> **作者:** JianHe Low; Ozge Mercanoglu Sincan; Richard Bowden
>
> **备注:** Accepted in the international conference on Intelligent Virtual Agents (IVA Adjunct)
>
> **摘要:** Sign spotting, the task of identifying and localizing individual signs within continuous sign language video, plays a pivotal role in scaling dataset annotations and addressing the severe data scarcity issue in sign language translation. While automatic sign spotting holds great promise for enabling frame-level supervision at scale, it grapples with challenges such as vocabulary inflexibility and ambiguity inherent in continuous sign streams. Hence, we introduce a novel, training-free framework that integrates Large Language Models (LLMs) to significantly enhance sign spotting quality. Our approach extracts global spatio-temporal and hand shape features, which are then matched against a large-scale sign dictionary using dynamic time warping and cosine similarity. This dictionary-based matching inherently offers superior vocabulary flexibility without requiring model retraining. To mitigate noise and ambiguity from the matching process, an LLM performs context-aware gloss disambiguation via beam search, notably without fine-tuning. Extensive experiments on both synthetic and real-world sign language datasets demonstrate our method's superior accuracy and sentence fluency compared to traditional approaches, highlighting the potential of LLMs in advancing sign spotting.
>
---
#### [replaced 050] A solvable generative model with a linear, one-step denoiser
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17807v3](http://arxiv.org/pdf/2411.17807v3)**

> **作者:** Indranil Halder
>
> **备注:** Published at International Conference on Machine Learning 2025
>
> **摘要:** We develop an analytically tractable single-step diffusion model based on a linear denoiser and present an explicit formula for the Kullback-Leibler divergence between the generated and sampling distribution, taken to be isotropic Gaussian, showing the effect of finite diffusion time and noise scale. Our study further reveals that the monotonic fall phase of Kullback-Leibler divergence begins when the training dataset size reaches the dimension of the data points. Finally, for large-scale practical diffusion models, we explain why a higher number of diffusion steps enhances production quality based on the theoretical arguments presented before.
>
---
#### [replaced 051] STF: Shallow-Level Temporal Feedback to Enhance Spiking Transformers
- **分类: cs.NE; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00387v2](http://arxiv.org/pdf/2508.00387v2)**

> **作者:** Zeqi Zheng; Zizheng Zhu; Yingchao Yu; Yanchen Huang; Changze Lv; Junfeng Tang; Zhaofei Yu; Yaochu Jin
>
> **备注:** 32 pages, 4 figures
>
> **摘要:** Transformer-based Spiking Neural Networks (SNNs) suffer from a great performance gap compared to floating-point \mbox{Artificial} Neural Networks (ANNs) due to the binary nature of spike trains. Recent efforts have introduced deep-level feedback loops to transmit high-level semantic information to narrow this gap. However, these designs often span \mbox{multiple} deep layers, resulting in costly feature transformations, higher parameter overhead, increased energy consumption, and longer inference latency. To address this issue, we propose Shallow-level Temporal Feedback (STF), a lightweight plug-and-play module for the encoding layer, which consists of Temporal-Spatial Position Embedding (TSPE) and Temporal Feedback (TF). Extensive experiments show that STF consistently improves performance across various Transformer-based SNN backbones on static datasets, including CIFAR-10, CIFAR-100, and ImageNet-1K, under different spike timestep settings. Further analysis reveals that STF enhances the diversity of spike patterns, which is key to performance gain. Moreover, evaluations on adversarial robustness and temporal sensitivity confirm that STF outperforms direct coding and its variants, highlighting its potential as a new spike encoding scheme for static scenarios. Our code will be released upon acceptance.
>
---
#### [replaced 052] Verbalized Representation Learning for Interpretable Few-Shot Generalization
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.18651v3](http://arxiv.org/pdf/2411.18651v3)**

> **作者:** Cheng-Fu Yang; Da Yin; Wenbo Hu; Heng Ji; Nanyun Peng; Bolei Zhou; Kai-Wei Chang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Humans recognize objects after observing only a few examples, a remarkable capability enabled by their inherent language understanding of the real-world environment. Developing verbalized and interpretable representation can significantly improve model generalization in low-data settings. In this work, we propose Verbalized Representation Learning (VRL), a novel approach for automatically extracting human-interpretable features for object recognition using few-shot data. Our method uniquely captures inter-class differences and intra-class commonalities in the form of natural language by employing a Vision-Language Model (VLM) to identify key discriminative features between different classes and shared characteristics within the same class. These verbalized features are then mapped to numeric vectors through the VLM. The resulting feature vectors can be further utilized to train and infer with downstream classifiers. Experimental results show that, at the same model scale, VRL achieves a 24% absolute improvement over prior state-of-the-art methods while using 95% less data and a smaller mode. Furthermore, compared to human-labeled attributes, the features learned by VRL exhibit a 20% absolute gain when used for downstream classification tasks. Code is available at: https://github.com/joeyy5588/VRL/tree/main.
>
---
#### [replaced 053] HydroChronos: Forecasting Decades of Surface Water Change
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14362v2](http://arxiv.org/pdf/2506.14362v2)**

> **作者:** Daniele Rege Cambrin; Eleonora Poeta; Eliana Pastor; Isaac Corley; Tania Cerquitelli; Elena Baralis; Paolo Garza
>
> **备注:** Accepted to SIGSPATIAL 2025
>
> **摘要:** Forecasting surface water dynamics is crucial for water resource management and climate change adaptation. However, the field lacks comprehensive datasets and standardized benchmarks. In this paper, we introduce HydroChronos, a large-scale, multi-modal spatiotemporal dataset for surface water dynamics forecasting designed to address this gap. We couple the dataset with three forecasting tasks. The dataset includes over three decades of aligned Landsat 5 and Sentinel-2 imagery, climate data, and Digital Elevation Models for diverse lakes and rivers across Europe, North America, and South America. We also propose AquaClimaTempo UNet, a novel spatiotemporal architecture with a dedicated climate data branch, as a strong benchmark baseline. Our model significantly outperforms a Persistence baseline for forecasting future water dynamics by +14% and +11% F1 across change detection and direction of change classification tasks, and by +0.1 MAE on the magnitude of change regression. Finally, we conduct an Explainable AI analysis to identify the key climate variables and input channels that influence surface water change, providing insights to inform and guide future modeling efforts.
>
---
#### [replaced 054] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04369v2](http://arxiv.org/pdf/2508.04369v2)**

> **作者:** Canhui Tang; Zifan Han; Hongbo Sun; Sanping Zhou; Xuchong Zhang; Xin Wei; Ye Yuan; Jinglin Xu; Hao Sun
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated significant progress in vision-language tasks, yet they still face challenges when processing long-duration video inputs. The limitation arises from MLLMs' context limit and training costs, necessitating sparse frame sampling before feeding videos into MLLMs. Existing video MLLMs adopt training-free uniform sampling or keyframe search, which may miss critical events or be constrained by the pre-trained models' event understanding capabilities. Meanwhile, building a training-based method remains challenging due to the unsupervised and non-differentiable nature of sparse frame sampling. To address these problems, we propose Temporal Sampling Policy Optimization (TSPO), advancing MLLMs' long-form video-language understanding via reinforcement learning. Specifically, we first propose a trainable event-aware temporal agent, which captures event-query correlation for performing probabilistic keyframe selection. Then, we propose the TSPO reinforcement learning paradigm, which models keyframe selection and language generation as a joint decision-making process, enabling end-to-end group relative optimization with efficient rule-based rewards. Furthermore, for the TSPO's training, we propose a long video training data construction pipeline with comprehensive temporal data and video Needle-in-a-Haystack data. Finally, we incorporate rule-based answering accuracy and temporal locating reward mechanisms to optimize the temporal sampling policy. Comprehensive experiments show that our TSPO achieves state-of-the-art performance across multiple long video understanding benchmarks, and shows transferable ability across different cutting-edge Video-MLLMs. Our code is available at https://github.com/Hui-design/TSPO
>
---
#### [replaced 055] Capsule-ConvKAN: A Hybrid Neural Approach to Medical Image Classification
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.06417v2](http://arxiv.org/pdf/2507.06417v2)**

> **作者:** Laura Pituková; Peter Sinčák; László József Kovács; Peng Wang
>
> **备注:** Preprint version. Accepted to IEEE SMC 2025
>
> **摘要:** This study conducts a comprehensive comparison of four neural network architectures: Convolutional Neural Network, Capsule Network, Convolutional Kolmogorov-Arnold Network, and the newly proposed Capsule-Convolutional Kolmogorov-Arnold Network. The proposed Capsule-ConvKAN architecture combines the dynamic routing and spatial hierarchy capabilities of Capsule Network with the flexible and interpretable function approximation of Convolutional Kolmogorov-Arnold Networks. This novel hybrid model was developed to improve feature representation and classification accuracy, particularly in challenging real-world biomedical image data. The architectures were evaluated on a histopathological image dataset, where Capsule-ConvKAN achieved the highest classification performance with an accuracy of 91.21%. The results demonstrate the potential of the newly introduced Capsule-ConvKAN in capturing spatial patterns, managing complex features, and addressing the limitations of traditional convolutional models in medical image classification.
>
---
#### [replaced 056] Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16435v2](http://arxiv.org/pdf/2502.16435v2)**

> **作者:** Jen-Tse Huang; Dasen Dai; Jen-Yuan Huang; Youliang Yuan; Xiaoyuan Liu; Wenxuan Wang; Wenxiang Jiao; Pinjia He; Zhaopeng Tu; Haodong Duan
>
> **备注:** Update: Evaluated 20 MLLMs; Added generated test cases
>
> **摘要:** Despite significant progress on popular multimodal benchmarks, state-of-the-art Multimodal Large Language Models (MLLMs) continue to struggle with basic visual reasoning tasks that are trivially solved by humans, such as recognizing spatial relationships. To systematically investigate this gap, we introduce VisFactor, a benchmark that digitizes 20 vision-centric subtests from a well-established cognitive psychology assessment. These subtests span four core domains of human visual cognition: (1) Visualization and Spatial Processing, (2) Perceptual and Closure, (3) Memory, and (4) Reasoning. We evaluate 20 frontier MLLMs from GPT, Gemini, Claude, LLaMA, Qwen, and SEED families. The best-performing model achieves a score of only 25.19 out of 100, with consistent failures on tasks such as mental rotation, spatial relation inference, and figure-ground discrimination, regardless of model size or prompting strategy. These findings suggest that current MLLM performance gains on high-level benchmarks do not reflect human-like low-level visual cognition, challenging the assumption that large-scale pretraining naturally induces gestalt-like perceptual capabilities. The dataset and evaluation toolkit are publicly available at: https://github.com/CUHK-ARISE/VisFactor.
>
---
#### [replaced 057] ANPrompt: Anti-noise Prompt Tuning for Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04677v2](http://arxiv.org/pdf/2508.04677v2)**

> **作者:** Yansheng Gao; Yufei Zheng; Jinghan Qu; Zixi Zhu; Yukuan Zhang; Shengsheng Wang
>
> **摘要:** Prompt tuning has emerged as an efficient and effective technique for adapting vision-language models (VLMs) with low computational overhead. However, existing methods often overlook the vulnerability of prompt-tuned VLMs to weak semantic perturbations-such as subtle image or text noise-that degrade their generalization to unseen classes. To address this limitation, we propose ANPrompt, a novel prompt tuning framework designed to enhance robustness under such perturbations. ANPrompt first constructs weak noise text features by fusing original and noise-perturbed text embeddings, which are then clustered to form noise prompts. These noise prompts are integrated with learnable prompt tokens to generate anti-noise prompts, which are injected into the deeper layers of both image and text encoders. To further capture the noise-aware visual semantics, ANPrompt computes the Noise-Resistant Visual Prompt Prototype (NRVPP) by averaging the output prompt tokens from the vision encoder. Finally, ANPrompt introduces alignment, robustness, and anti-noise objectives by computing a Weak semantic noise Alignment Loss (WALoss) alongside the standard cross-entropy and sim loss. Experiments across 11 benchmarks demonstrate that ANPrompt consistently outperforms existing prompt tuning approaches, achieving superior robustness to semantic noise and improved generalization to novel categories.
>
---
#### [replaced 058] PromptDresser: Improving the Quality and Controllability of Virtual Try-On via Generative Textual Prompt and Prompt-aware Mask
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.16978v2](http://arxiv.org/pdf/2412.16978v2)**

> **作者:** Jeongho Kim; Hoiyeong Jin; Sunghyun Park; Jaegul Choo
>
> **备注:** 20 pages
>
> **摘要:** Recent virtual try-on approaches have advanced by finetuning pre-trained text-to-image diffusion models to leverage their powerful generative ability. However, the use of text prompts in virtual try-on remains underexplored. This paper tackles a text-editable virtual try-on task that modifies the clothing based on the provided clothing image while editing the wearing style (e.g., tucking style, fit) according to the text descriptions. In the text-editable virtual try-on, three key aspects exist: (i) designing rich text descriptions for paired person-clothing data to train the model, (ii) addressing the conflicts where textual information of the existing person's clothing interferes the generation of the new clothing, and (iii) adaptively adjust the inpainting mask aligned with the text descriptions, ensuring proper editing areas while preserving the original person's appearance irrelevant to the new clothing. To address these aspects, we propose PromptDresser, a text-editable virtual try-on model that leverages large multimodal model (LMM) assistance to enable high-quality and versatile manipulation based on generative text prompts. Our approach utilizes LMMs via in-context learning to generate detailed text descriptions for person and clothing images independently, including pose details and editing attributes using minimal human cost. Moreover, to ensure the editing areas, we adjust the inpainting mask depending on the text prompts adaptively. Our approach enhances text editability while effectively conveying clothing details that are difficult to capture through images alone, leading to improved image quality. Experiments show that PromptDresser significantly outperforms baselines, demonstrating superior text-driven control and versatile clothing manipulation. Our code is available at https://github.com/rlawjdghek/PromptDresser.
>
---
#### [replaced 059] EarthSynth: Generating Informative Earth Observation with Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12108v2](http://arxiv.org/pdf/2505.12108v2)**

> **作者:** Jiancheng Pan; Shiye Lei; Yuqian Fu; Jiahao Li; Yanxing Liu; Yuze Sun; Xiao He; Long Peng; Xiaomeng Huang; Bo Zhao
>
> **备注:** 25 pages
>
> **摘要:** Remote sensing image (RSI) interpretation typically faces challenges due to the scarcity of labeled data, which limits the performance of RSI interpretation tasks. To tackle this challenge, we propose EarthSynth, a diffusion-based generative foundation model that enables synthesizing multi-category, cross-satellite labeled Earth observation for downstream RSI interpretation tasks. To the best of our knowledge, EarthSynth is the first to explore multi-task generation for remote sensing, tackling the challenge of limited generalization in task-oriented synthesis for RSI interpretation. EarthSynth, trained on the EarthSynth-180K dataset, employs the Counterfactual Composition training strategy with a three-dimensional batch-sample selection mechanism to improve training data diversity and enhance category control. Furthermore, a rule-based method of R-Filter is proposed to filter more informative synthetic data for downstream tasks. We evaluate our EarthSynth on scene classification, object detection, and semantic segmentation in open-world scenarios. There are significant improvements in open-vocabulary understanding tasks, offering a practical solution for advancing RSI interpretation.
>
---
#### [replaced 060] StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.01343v2](http://arxiv.org/pdf/2408.01343v2)**

> **作者:** Bingyu Li; Da Zhang; Zhiyuan Zhao; Junyu Gao; Xuelong Li
>
> **摘要:** Multimodal semantic segmentation shows significant potential for enhancing segmentation accuracy in complex scenes. However, current methods often incorporate specialized feature fusion modules tailored to specific modalities, thereby restricting input flexibility and increasing the number of training parameters. To address these challenges, we propose StitchFusion, a straightforward yet effective modal fusion framework that integrates large-scale pre-trained models directly as encoders and feature fusers. This approach facilitates comprehensive multi-modal and multi-scale feature fusion, accommodating any visual modal inputs. Specifically, Our framework achieves modal integration during encoding by sharing multi-modal visual information. To enhance information exchange across modalities, we introduce a multi-directional adapter module (MultiAdapter) to enable cross-modal information transfer during encoding. By leveraging MultiAdapter to propagate multi-scale information across pre-trained encoders during the encoding process, StitchFusion achieves multi-modal visual information integration during encoding. Extensive comparative experiments demonstrate that our model achieves state-of-the-art performance on four multi-modal segmentation datasets with minimal additional parameters. Furthermore, the experimental integration of MultiAdapter with existing Feature Fusion Modules (FFMs) highlights their complementary nature. Our code is available at StitchFusion_repo.
>
---
#### [replaced 061] VLM4D: Towards Spatiotemporal Awareness in Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.02095v2](http://arxiv.org/pdf/2508.02095v2)**

> **作者:** Shijie Zhou; Alexander Vilesov; Xuehai He; Ziyu Wan; Shuwang Zhang; Aditya Nagachandra; Di Chang; Dongdong Chen; Xin Eric Wang; Achuta Kadambi
>
> **备注:** ICCV 2025, Project Website: https://vlm4d.github.io/
>
> **摘要:** Vision language models (VLMs) have shown remarkable capabilities in integrating linguistic and visual reasoning but remain fundamentally limited in understanding dynamic spatiotemporal interactions. Humans effortlessly track and reason about object movements, rotations, and perspective shifts-abilities essential for robust dynamic real-world understanding yet notably lacking in current VLMs. In this paper, we introduce VLM4D, the first benchmark specifically designed to evaluate the spatiotemporal reasoning capabilities of VLMs. Our benchmark comprises diverse real-world and synthetic videos accompanied by carefully curated question-answer pairs emphasizing translational and rotational motions, perspective awareness, and motion continuity. Through comprehensive evaluations of state-of-the-art open and closed-source VLMs, we identify significant performance gaps compared to human baselines, highlighting fundamental deficiencies in existing models. Extensive analysis reveals that VLMs struggle particularly with integrating multiple visual cues and maintaining temporal coherence. We further explore promising directions, such as leveraging 4D feature field reconstruction and targeted spatiotemporal supervised fine-tuning, demonstrating their effectiveness in enhancing spatiotemporal comprehension. Our work aims to encourage deeper exploration into improving VLMs' spatial and temporal grounding, paving the way towards more capable and reliable visual intelligence for dynamic environments.
>
---
#### [replaced 062] CountingFruit: Language-Guided 3D Fruit Counting with Semantic Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.01109v3](http://arxiv.org/pdf/2506.01109v3)**

> **作者:** Fengze Li; Yangle Liu; Jieming Ma; Hai-Ning Liang; Yaochun Shen; Huangxiang Li; Zhijing Wu
>
> **摘要:** Accurate 3D fruit counting in orchards is challenging due to heavy occlusion, semantic ambiguity between fruits and surrounding structures, and the high computational cost of volumetric reconstruction. Existing pipelines often rely on multi-view 2D segmentation and dense volumetric sampling, which lead to accumulated fusion errors and slow inference. We introduce FruitLangGS, a language-guided 3D fruit counting framework that reconstructs orchard-scale scenes using an adaptive-density Gaussian Splatting pipeline with radius-aware pruning and tile-based rasterization, enabling scalable 3D representation. During inference, compressed CLIP-aligned semantic vectors embedded in each Gaussian are filtered via a dual-threshold cosine similarity mechanism, retrieving Gaussians relevant to target prompts while suppressing common distractors (e.g., foliage), without requiring retraining or image-space masks. The selected Gaussians are then sampled into dense point clouds and clustered geometrically to estimate fruit instances, remaining robust under severe occlusion and viewpoint variation. Experiments on nine different orchard-scale datasets demonstrate that FruitLangGS consistently outperforms existing pipelines in instance counting recall, avoiding multi-view segmentation fusion errors and achieving up to 99.7% recall on Pfuji-Size_Orch2018 orchard dataset. Ablation studies further confirm that language-conditioned semantic embedding and dual-threshold prompt filtering are essential for suppressing distractors and improving counting accuracy under heavy occlusion. Beyond fruit counting, the same framework enables prompt-driven 3D semantic retrieval without retraining, highlighting the potential of language-guided 3D perception for scalable agricultural scene understanding.
>
---
#### [replaced 063] Towards a General-Purpose Zero-Shot Synthetic Low-Light Image and Video Pipeline
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.12169v2](http://arxiv.org/pdf/2504.12169v2)**

> **作者:** Joanne Lin; Crispian Morris; Ruirui Lin; Fan Zhang; David Bull; Nantheera Anantrasirichai
>
> **摘要:** Low-light conditions pose significant challenges for both human and machine annotation. This in turn has led to a lack of research into machine understanding for low-light images and (in particular) videos. A common approach is to apply annotations obtained from high quality datasets to synthetically created low light versions. In addition, these approaches are often limited through the use of unrealistic noise models. In this paper, we propose a new Degradation Estimation Network (DEN), which synthetically generates realistic standard RGB (sRGB) noise without the requirement for camera metadata. This is achieved by estimating the parameters of physics-informed noise distributions, trained in a self-supervised manner. This zero-shot approach allows our method to generate synthetic noisy content with a diverse range of realistic noise characteristics, unlike other methods which focus on recreating the noise characteristics of the training data. We evaluate our proposed synthetic pipeline using various methods trained on its synthetic data for typical low-light tasks including synthetic noise replication, video enhancement, and object detection, showing improvements of up to 24\% KLD, 21\% LPIPS, and 62\% AP$_{50-95}$, respectively.
>
---
#### [replaced 064] MetaOcc: Spatio-Temporal Fusion of Surround-View 4D Radar and Camera for 3D Occupancy Prediction with Dual Training Strategies
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.15384v2](http://arxiv.org/pdf/2501.15384v2)**

> **作者:** Long Yang; Lianqing Zheng; Wenjin Ai; Minghao Liu; Sen Li; Qunshu Lin; Shengyu Yan; Jie Bai; Zhixiong Ma; Tao Huang; Xichan Zhu
>
> **摘要:** Robust 3D occupancy prediction is essential for autonomous driving, particularly under adverse weather conditions where traditional vision-only systems struggle. While the fusion of surround-view 4D radar and cameras offers a promising low-cost solution, effectively extracting and integrating features from these heterogeneous sensors remains challenging. This paper introduces MetaOcc, a novel multi-modal framework for omnidirectional 3D occupancy prediction that leverages both multi-view 4D radar and images. To address the limitations of directly applying LiDAR-oriented encoders to sparse radar data, we propose a Radar Height Self-Attention module that enhances vertical spatial reasoning and feature extraction. Additionally, a Hierarchical Multi-scale Multi-modal Fusion strategy is developed to perform adaptive local-global fusion across modalities and time, mitigating spatio-temporal misalignments and enriching fused feature representations. To reduce reliance on expensive point cloud annotations, we further propose a pseudo-label generation pipeline based on an open-set segmentor. This enables a semi-supervised strategy that achieves 90% of the fully supervised performance using only 50% of the ground truth labels, offering an effective trade-off between annotation cost and accuracy. Extensive experiments demonstrate that MetaOcc under full supervision achieves state-of-the-art performance, outperforming previous methods by +0.47 SC IoU and +4.02 mIoU on the OmniHD-Scenes dataset, and by +1.16 SC IoU and +1.24 mIoU on the SurroundOcc-nuScenes dataset. These results demonstrate the scalability and robustness of MetaOcc across sensor domains and training conditions, paving the way for practical deployment in real-world autonomous systems. Code and data are available at https://github.com/LucasYang567/MetaOcc.
>
---
#### [replaced 065] A dataset of primary nasopharyngeal carcinoma MRI with multi-modalities segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.03253v2](http://arxiv.org/pdf/2404.03253v2)**

> **作者:** Yin Li; Qi Chen; Kai Wang; Meige Li; Liping Si; Yingwei Guo; Yu Xiong; Qixing Wang; Yang Qin; Ling Xu; Patrick van der Smagt; Jun Tang; Nutan Chen
>
> **备注:** This preprint has been submitted to and accepted in principle for publication in Scientific Data without significant changes
>
> **摘要:** Multi-modality magnetic resonance imaging(MRI) data facilitate the early diagnosis, tumor segmentation, and disease staging in the management of nasopharyngeal carcinoma (NPC). The lack of publicly available, comprehensive datasets limits advancements in diagnosis, treatment planning, and the development of machine learning algorithms for NPC. Addressing this critical need, we introduce the first comprehensive NPC MRI dataset, encompassing MR axial imaging of 277 primary NPC patients. This dataset includes T1-weighted, T2-weighted, and contrast-enhanced T1-weighted sequences, totaling 831 scans. In addition to the corresponding clinical data, manually annotated and labeled segmentations by experienced radiologists offer high-quality data resources from untreated primary NPC.
>
---
#### [replaced 066] Learned Single-Pass Multitasking Perceptual Graphics for Immersive Displays
- **分类: cs.CV; cs.GR; eess.IV**

- **链接: [http://arxiv.org/pdf/2408.07836v2](http://arxiv.org/pdf/2408.07836v2)**

> **作者:** Doğa Yılmaz; He Wang; Towaki Takikawa; Duygu Ceylan; Kaan Akşit
>
> **摘要:** Emerging immersive display technologies efficiently utilize resources with perceptual graphics methods such as foveated rendering and denoising. Running multiple perceptual graphics methods challenges devices with limited power and computational resources. We propose a computationally-lightweight learned multitasking perceptual graphics model. Given RGB images and text-prompts, our model performs text-described perceptual tasks in a single inference step. Simply daisy-chaining multiple models or training dedicated models can lead to model management issues and exhaust computational resources. In contrast, our flexible method unlocks consistent high quality perceptual effects with reasonable compute, supporting various permutations at varied intensities using adjectives in text prompts (e.g. mildly, lightly). Text-guidance provides ease of use for dynamic requirements such as creative processes. To train our model, we propose a dataset containing source and perceptually enhanced images with corresponding text prompts. We evaluate our model on desktop and embedded platforms and validate perceptual quality through a user study.
>
---
#### [replaced 067] Unlocking the Potential of MLLMs in Referring Expression Segmentation via a Light-weight Mask Decoder
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.04107v2](http://arxiv.org/pdf/2508.04107v2)**

> **作者:** Jingchao Wang; Zhijian Wu; Dingjiang Huang; Yefeng Zheng; Hong Wang
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Reference Expression Segmentation (RES) aims to segment image regions specified by referring expressions and has become popular with the rise of multimodal large models (MLLMs). While MLLMs excel in semantic understanding, their token-generation paradigm struggles with pixel-level dense prediction. Existing RES methods either couple MLLMs with the parameter-heavy Segment Anything Model (SAM) with 632M network parameters or adopt SAM-free lightweight pipelines that sacrifice accuracy. To address the trade-off between performance and cost, we specifically propose MLLMSeg, a novel framework that fully exploits the inherent visual detail features encoded in the MLLM vision encoder without introducing an extra visual encoder. Besides, we propose a detail-enhanced and semantic-consistent feature fusion module (DSFF) that fully integrates the detail-related visual feature with the semantic-related feature output by the large language model (LLM) of MLLM. Finally, we establish a light-weight mask decoder with only 34M network parameters that optimally leverages detailed spatial features from the visual encoder and semantic features from the LLM to achieve precise mask prediction. Extensive experiments demonstrate that our method generally surpasses both SAM-based and SAM-free competitors, striking a better balance between performance and cost. Code is available at https://github.com/jcwang0602/MLLMSeg.
>
---
#### [replaced 068] FullTransNet: Full Transformer with Local-Global Attention for Video Summarization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.00882v2](http://arxiv.org/pdf/2501.00882v2)**

> **作者:** Libin Lan; Lu Jiang; Tianshu Yu; Xiaojuan Liu; Zhongshi He
>
> **备注:** 15 pages, 8 figures, 4 tables; The code is at https://github.com/ChiangLu/FullTransNet
>
> **摘要:** Video summarization aims to generate a compact, informative, and representative synopsis of raw videos, which is crucial for browsing, analyzing, and understanding video content. Dominant approaches in video summarization primarily rely on recurrent or convolutional neural networks, and more recently on encoder-only transformer architectures. However, these methods typically suffer from several limitations in parallelism, modeling long-range dependencies, and providing explicit generative capabilities. To address these issues, we propose a transformer-like architecture named FullTransNet with two-fold ideas. First, it uses a full transformer with an encoder-decoder structure as an alternative architecture for video summarization. As the full transformer is specifically designed for sequence transduction tasks, its direct application to video summarization is both intuitive and effective. Second, it replaces the standard full attention mechanism with a combination of local and global sparse attention, enabling the model to capture long-range dependencies while significantly reducing computational costs. This local-global sparse attention is applied exclusively at the encoder side, where the majority of computations occur, further enhancing efficiency. Extensive experiments on two widely used benchmark datasets, SumMe and TVSum, demonstrate that our model achieves F-scores of 54.4% and 63.9%, respectively, while maintaining relatively low computational and memory requirements. These results surpass the second-best performing methods by 0.1% and 0.3%, respectively, verifying the effectiveness and efficiency of FullTransNet.
>
---
#### [replaced 069] AudioGen-Omni: A Unified Multimodal Diffusion Transformer for Video-Synchronized Audio, Speech, and Song Generation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.00733v4](http://arxiv.org/pdf/2508.00733v4)**

> **作者:** Le Wang; Jun Wang; Chunyu Qiang; Feng Deng; Chen Zhang; Di Zhang; Kun Gai
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** We present AudioGen-Omni - a unified approach based on multimodal diffusion transformers (MMDit), capable of generating high-fidelity audio, speech, and song coherently synchronized with the input video. AudioGen-Omni introduces a novel joint training paradigm that seamlessly integrates large-scale video-text-audio corpora, enabling a model capable of generating semantically rich, acoustically diverse audio conditioned on multimodal inputs and adaptable to a wide range of audio generation tasks. AudioGen-Omni employs a unified lyrics-transcription encoder that encodes graphemes and phonemes from both song and spoken inputs into dense frame-level representations. Dense frame-level representations are fused using an AdaLN-based joint attention mechanism enhanced with phase-aligned anisotropic positional infusion (PAAPI), wherein RoPE is selectively applied to temporally structured modalities to ensure precise and robust cross-modal alignment. By unfreezing all modalities and masking missing inputs, AudioGen-Omni mitigates the semantic constraints of text-frozen paradigms, enabling effective cross-modal conditioning. This joint training approach enhances audio quality, semantic alignment, and lip-sync accuracy, while also achieving state-of-the-art results on Text-to-Audio/Speech/Song tasks. With an inference time of 1.91 seconds for 8 seconds of audio, it offers substantial improvements in both efficiency and generality.
>
---
#### [replaced 070] Part Segmentation and Motion Estimation for Articulated Objects with Dynamic 3D Gaussians
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22718v2](http://arxiv.org/pdf/2506.22718v2)**

> **作者:** Jun-Jee Chao; Qingyuan Jiang; Volkan Isler
>
> **摘要:** Part segmentation and motion estimation are two fundamental problems for articulated object motion analysis. In this paper, we present a method to solve these two problems jointly from a sequence of observed point clouds of a single articulated object. The main challenge in our problem setting is that the point clouds are not assumed to be generated by a fixed set of moving points. Instead, each point cloud in the sequence could be an arbitrary sampling of the object surface at that particular time step. Such scenarios occur when the object undergoes major occlusions, or if the dataset is collected using measurements from multiple sensors asynchronously. In these scenarios, methods that rely on tracking point correspondences are not appropriate. We present an alternative approach based on a compact but effective representation where we represent the object as a collection of simple building blocks modeled as 3D Gaussians. We parameterize the Gaussians with time-dependent rotations, translations, and scales that are shared across all time steps. With our representation, part segmentation can be achieved by building correspondences between the observed points and the Gaussians. Moreover, the transformation of each point across time can be obtained by following the poses of the assigned Gaussian (even when the point is not observed). Experiments show that our method outperforms existing methods that solely rely on finding point correspondences. Additionally, we extend existing datasets to emulate real-world scenarios by considering viewpoint occlusions. We further demonstrate that our method is more robust to missing points as compared to existing approaches on these challenging datasets, even when some parts are completely occluded in some time-steps. Notably, our part segmentation performance outperforms the state-of-the-art method by 13% on point clouds with occlusions.
>
---
#### [replaced 071] AnomalyControl: Learning Cross-modal Semantic Features for Controllable Anomaly Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.06510v4](http://arxiv.org/pdf/2412.06510v4)**

> **作者:** Shidan He; Lei Liu; Xiujun Shu; Bo Wang; Yuanhao Feng; Shen Zhao
>
> **摘要:** Anomaly synthesis is a crucial approach to augment abnormal data for advancing anomaly inspection. Based on the knowledge from the large-scale pre-training, existing text-to-image anomaly synthesis methods predominantly focus on textual information or coarse-aligned visual features to guide the entire generation process. However, these methods often lack sufficient descriptors to capture the complicated characteristics of realistic anomalies (e.g., the fine-grained visual pattern of anomalies), limiting the realism and generalization of the generation process. To this end, we propose a novel anomaly synthesis framework called AnomalyControl to learn cross-modal semantic features as guidance signals, which could encode the generalized anomaly cues from text-image reference prompts and improve the realism of synthesized abnormal samples. Specifically, AnomalyControl adopts a flexible and non-matching prompt pair (i.e., a text-image reference prompt and a targeted text prompt), where a Cross-modal Semantic Modeling (CSM) module is designed to extract cross-modal semantic features from the textual and visual descriptors. Then, an Anomaly-Semantic Enhanced Attention (ASEA) mechanism is formulated to allow CSM to focus on the specific visual patterns of the anomaly, thus enhancing the realism and contextual relevance of the generated anomaly features. Treating cross-modal semantic features as the prior, a Semantic Guided Adapter (SGA) is designed to encode effective guidance signals for the adequate and controllable synthesis process. Extensive experiments indicate that AnomalyControl can achieve state-of-the-art results in anomaly synthesis compared with existing methods while exhibiting superior performance for downstream tasks.
>
---
#### [replaced 072] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.03069v2](http://arxiv.org/pdf/2412.03069v2)**

> **作者:** Liao Qu; Huichao Zhang; Yiheng Liu; Xu Wang; Yi Jiang; Yiming Gao; Hu Ye; Daniel K. Du; Zehuan Yuan; Xinglong Wu
>
> **备注:** CVPR 2025; Code and models: https://github.com/ByteVisionLab/TokenFlow
>
> **摘要:** We present TokenFlow, a novel unified image tokenizer that bridges the long-standing gap between multimodal understanding and generation. Prior research attempt to employ a single reconstruction-targeted Vector Quantization (VQ) encoder for unifying these two tasks. We observe that understanding and generation require fundamentally different granularities of visual information. This leads to a critical trade-off, particularly compromising performance in multimodal understanding tasks. TokenFlow addresses this challenge through an innovative dual-codebook architecture that decouples semantic and pixel-level feature learning while maintaining their alignment via a shared mapping mechanism. This design enables direct access to both high-level semantic representations crucial for understanding tasks and fine-grained visual features essential for generation through shared indices. Our extensive experiments demonstrate TokenFlow's superiority across multiple dimensions. Leveraging TokenFlow, we demonstrate for the first time that discrete visual input can surpass LLaVA-1.5 13B in understanding performance, achieving a 7.2\% average improvement. For image reconstruction, we achieve a strong FID score of 0.63 at 384*384 resolution. Moreover, TokenFlow establishes state-of-the-art performance in autoregressive image generation with a GenEval score of 0.55 at 256*256 resolution, achieving comparable results to SDXL.
>
---
#### [replaced 073] Patho-AgenticRAG: Towards Multimodal Agentic Retrieval-Augmented Generation for Pathology VLMs via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02258v2](http://arxiv.org/pdf/2508.02258v2)**

> **作者:** Wenchuan Zhang; Jingru Guo; Hengzhe Zhang; Penghao Zhang; Jie Chen; Shuwan Zhang; Zhang Zhang; Yuhao Yi; Hong Bu
>
> **摘要:** Although Vision Language Models (VLMs) have shown strong generalization in medical imaging, pathology presents unique challenges due to ultra-high resolution, complex tissue structures, and nuanced clinical semantics. These factors make pathology VLMs prone to hallucinations, i.e., generating outputs inconsistent with visual evidence, which undermines clinical trust. Existing RAG approaches in this domain largely depend on text-based knowledge bases, limiting their ability to leverage diagnostic visual cues. To address this, we propose Patho-AgenticRAG, a multimodal RAG framework with a database built on page-level embeddings from authoritative pathology textbooks. Unlike traditional text-only retrieval systems, it supports joint text-image search, enabling direct retrieval of textbook pages that contain both the queried text and relevant visual cues, thus avoiding the loss of critical image-based information. Patho-AgenticRAG also supports reasoning, task decomposition, and multi-turn search interactions, improving accuracy in complex diagnostic scenarios. Experiments show that Patho-AgenticRAG significantly outperforms existing multimodal models in complex pathology tasks like multiple-choice diagnosis and visual question answering. Our project is available at the Patho-AgenticRAG repository: https://github.com/Wenchuan-Zhang/Patho-AgenticRAG.
>
---
#### [replaced 074] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2508.00649v2](http://arxiv.org/pdf/2508.00649v2)**

> **作者:** Junhao Zheng; Jiahao Sun; Chenhao Lin; Zhengyu Zhao; Chen Ma; Chong Zhang; Cong Wang; Qian Wang; Chao Shen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Developing reliable defenses against patch attacks on object detectors has attracted increasing interest. However, we identify that existing defense evaluations lack a unified and comprehensive framework, resulting in inconsistent and incomplete assessments of current methods. To address this issue, we revisit 11 representative defenses and present the first patch defense benchmark, involving 2 attack goals, 13 patch attacks, 11 object detectors, and 4 diverse metrics. This leads to the large-scale adversarial patch dataset with 94 types of patches and 94,000 images. Our comprehensive analyses reveal new insights: (1) The difficulty in defending against naturalistic patches lies in the data distribution, rather than the commonly believed high frequencies. Our new dataset with diverse patch distributions can be used to improve existing defenses by 15.09% AP@0.5. (2) The average precision of the attacked object, rather than the commonly pursued patch detection accuracy, shows high consistency with defense performance. (3) Adaptive attacks can substantially bypass existing defenses, and defenses with complex/stochastic models or universal patch properties are relatively robust. We hope that our analyses will serve as guidance on properly evaluating patch attacks/defenses and advancing their design. Code and dataset are available at https://github.com/Gandolfczjh/APDE, where we will keep integrating new attacks/defenses.
>
---
#### [replaced 075] A Differentiable Wave Optics Model for End-to-End Computational Imaging System Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09774v2](http://arxiv.org/pdf/2412.09774v2)**

> **作者:** Chi-Jui Ho; Yash Belhe; Steve Rotenberg; Ravi Ramamoorthi; Tzu-Mao Li; Nicholas Antipa
>
> **摘要:** End-to-end optimization, which simultaneously optimizes optics and algorithms, has emerged as a powerful data-driven method for computational imaging system design. This method achieves joint optimization through backpropagation by incorporating differentiable optics simulators to generate measurements and algorithms to extract information from measurements. However, due to high computational costs, it is challenging to model both aberration and diffraction in light transport for end-to-end optimization of compound optics. Therefore, most existing methods compromise physical accuracy by neglecting wave optics effects or off-axis aberrations, which raises concerns about the robustness of the resulting designs. In this paper, we propose a differentiable optics simulator that efficiently models both aberration and diffraction for compound optics. Using the simulator, we conduct end-to-end optimization on scene reconstruction and classification. Experimental results demonstrate that both lenses and algorithms adopt different configurations depending on whether wave optics is modeled. We also show that systems optimized without wave optics suffer from performance degradation when wave optics effects are introduced during testing. These findings underscore the importance of accurate wave optics modeling in optimizing imaging systems for robust, high-performance applications.
>
---
#### [replaced 076] Calibrating Deep Neural Network using Euclidean Distance
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.18321v2](http://arxiv.org/pdf/2410.18321v2)**

> **作者:** Wenhao Liang; Chang Dong; Liangwei Zheng; Wei Zhang; Weitong Chen
>
> **备注:** V2
>
> **摘要:** Uncertainty is a fundamental aspect of real-world scenarios, where perfect information is rarely available. Humans naturally develop complex internal models to navigate incomplete data and effectively respond to unforeseen or partially observed events. In machine learning, Focal Loss is commonly used to reduce misclassification rates by emphasizing hard-to-classify samples. However, it does not guarantee well-calibrated predicted probabilities and may result in models that are overconfident or underconfident. High calibration error indicates a misalignment between predicted probabilities and actual outcomes, affecting model reliability. This research introduces a novel loss function called Focal Calibration Loss (FCL), designed to improve probability calibration while retaining the advantages of Focal Loss in handling difficult samples. By minimizing the Euclidean norm through a strictly proper loss, FCL penalizes the instance-wise calibration error and constrains bounds. We provide theoretical validation for proposed method and apply it to calibrate CheXNet for potential deployment in web-based health-care systems. Extensive evaluations on various models and datasets demonstrate that our method achieves SOTA performance in both calibration and accuracy metrics.
>
---
#### [replaced 077] GTR: Improving Large 3D Reconstruction Models through Geometry and Texture Refinement
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.05649v3](http://arxiv.org/pdf/2406.05649v3)**

> **作者:** Peiye Zhuang; Songfang Han; Chaoyang Wang; Aliaksandr Siarohin; Jiaxu Zou; Michael Vasilkovsky; Vladislav Shakhrai; Sergey Korolev; Sergey Tulyakov; Hsin-Ying Lee
>
> **备注:** 19 pages, 17 figures. Project page: https://snap-research.github.io/GTR/; Code: https://github.com/snap-research/snap_gtr/
>
> **摘要:** We propose a novel approach for 3D mesh reconstruction from multi-view images. Our method takes inspiration from large reconstruction models like LRM that use a transformer-based triplane generator and a Neural Radiance Field (NeRF) model trained on multi-view images. However, in our method, we introduce several important modifications that allow us to significantly enhance 3D reconstruction quality. First of all, we examine the original LRM architecture and find several shortcomings. Subsequently, we introduce respective modifications to the LRM architecture, which lead to improved multi-view image representation and more computationally efficient training. Second, in order to improve geometry reconstruction and enable supervision at full image resolution, we extract meshes from the NeRF field in a differentiable manner and fine-tune the NeRF model through mesh rendering. These modifications allow us to achieve state-of-the-art performance on both 2D and 3D evaluation metrics, such as a PSNR of 28.67 on Google Scanned Objects (GSO) dataset. Despite these superior results, our feed-forward model still struggles to reconstruct complex textures, such as text and portraits on assets. To address this, we introduce a lightweight per-instance texture refinement procedure. This procedure fine-tunes the triplane representation and the NeRF color estimation model on the mesh surface using the input multi-view images in just 4 seconds. This refinement improves the PSNR to 29.79 and achieves faithful reconstruction of complex textures, such as text. Additionally, our approach enables various downstream applications, including text- or image-to-3D generation.
>
---
#### [replaced 078] SteerPose: Simultaneous Extrinsic Camera Calibration and Matching from Articulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01691v2](http://arxiv.org/pdf/2506.01691v2)**

> **作者:** Sang-Eun Lee; Ko Nishino; Shohei Nobuhara
>
> **备注:** Accepted to BMVC2025. Project website: https://kcvl-public.github.io/steerpose/
>
> **摘要:** Can freely moving humans or animals themselves serve as calibration targets for multi-camera systems while simultaneously estimating their correspondences across views? We humans can solve this problem by mentally rotating the observed 2D poses and aligning them with those in the target views. Inspired by this cognitive ability, we propose SteerPose, a neural network that performs this rotation of 2D poses into another view. By integrating differentiable matching, SteerPose simultaneously performs extrinsic camera calibration and correspondence search within a single unified framework. We also introduce a novel geometric consistency loss that explicitly ensures that the estimated rotation and correspondences result in a valid translation estimation. Experimental results on diverse in-the-wild datasets of humans and animals validate the effectiveness and robustness of the proposed method. Furthermore, we demonstrate that our method can reconstruct the 3D poses of novel animals in multi-camera setups by leveraging off-the-shelf 2D pose estimators and our class-agnostic model.
>
---
#### [replaced 079] GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10809v3](http://arxiv.org/pdf/2504.10809v3)**

> **作者:** Christophe Bolduc; Yannick Hold-Geoffroy; Zhixin Shu; Jean-François Lalonde
>
> **摘要:** We present GaSLight, a method that generates spatially-varying lighting from regular images. Our method proposes using HDR Gaussian Splats as light source representation, marking the first time regular images can serve as light sources in a 3D renderer. Our two-stage process first enhances the dynamic range of images plausibly and accurately by leveraging the priors embedded in diffusion models. Next, we employ Gaussian Splats to model 3D lighting, achieving spatially variant lighting. Our approach yields state-of-the-art results on HDR estimations and their applications in illuminating virtual objects and scenes. To facilitate the benchmarking of images as light sources, we introduce a novel dataset of calibrated and unsaturated HDR to evaluate images as light sources. We assess our method using a combination of this novel dataset and an existing dataset from the literature. Project page: https://lvsn.github.io/gaslight/
>
---
#### [replaced 080] EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03905v4](http://arxiv.org/pdf/2507.03905v4)**

> **作者:** Rang Meng; Yan Wang; Weipeng Wu; Ruobing Zheng; Yuming Li; Chenguang Ma
>
> **摘要:** Recent work on human animation usually incorporates large-scale video models, thereby achieving more vivid performance. However, the practical use of such methods is hindered by the slow inference speed and high computational demands. Moreover, traditional work typically employs separate models for each animation task, increasing costs in multi-task scenarios and worsening the dilemma. To address these limitations, we introduce EchoMimicV3, an efficient framework that unifies multi-task and multi-modal human animation. At the core of EchoMimicV3 lies a threefold design: a Soup-of-Tasks paradigm, a Soup-of-Modals paradigm, and a novel training and inference strategy. The Soup-of-Tasks leverages multi-task mask inputs and a counter-intuitive task allocation strategy to achieve multi-task gains without multi-model pains. Meanwhile, the Soup-of-Modals introduces a Coupled-Decoupled Multi-Modal Cross Attention module to inject multi-modal conditions, complemented by a Multi-Modal Timestep Phase-aware Dynamical Allocation mechanism to modulate multi-modal mixtures. Besides, we propose Negative Direct Preference Optimization, Phase-aware Negative Classifier-Free Guidance (CFG), and Long Video CFG, which ensure stable training and inference. Extensive experiments and analyses demonstrate that EchoMimicV3, with a minimal model size of 1.3 billion parameters, achieves competitive performance in both quantitative and qualitative evaluations.
>
---
#### [replaced 081] Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15509v2](http://arxiv.org/pdf/2507.15509v2)**

> **作者:** Lei Chen; Xuanle Zhao; Zhixiong Zeng; Jing Huang; Yufeng Zhong; Lin Ma
>
> **备注:** technical report
>
> **摘要:** Recently, inspired by OpenAI-o1/o3 and Deepseek-R1, the R1-Style method based on reinforcement learning fine-tuning has received widespread attention from the community. Previous R1-Style methods mainly focus on mathematical reasoning and code intelligence. It is of great research significance to verify their advantages on more general multimodal data. Chart is an important multimodal data type with rich information, which brings important research challenges in complex reasoning. In this work, we introduce Chart-R1, a chart-domain vision-language model with reinforcement learning fine-tuning to enable complex chart reasoning. To support Chart-R1, we first propose a novel programmatic data synthesis technology to generate high-quality step-by-step chart reasoning data covering single- and multi-subcharts, which makes up for the lack of reasoning data in the chart domain. Then we develop a two-stage training strategy: Chart-COT with step-by-step chain-of-thought supervision, and Chart-RFT with numerically sensitive reinforcement fine-tuning. Chart-COT aims to decompose complex chart reasoning tasks into fine-grained, understandable subtasks through step-by-step supervision, which lays a good foundation for improving the reasoning level of reinforcement learning. Chart-RFT utilize the typical group relative policy optimization strategy, in which a relatively soft reward is adopted for numerical response to emphasize the numerical sensitivity in the chart domain. We conduct extensive experiments on open-source benchmarks and self-built chart reasoning dataset (\emph{i.e., ChartRQA}). Experimental results show that Chart-R1 has significant advantages compared to chart-domain methods, even comparable to open/closed source large-scale models (\emph{e.g., GPT-4o, Claude-3.5}).
>
---
