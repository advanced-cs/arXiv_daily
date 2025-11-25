# 计算机视觉 cs.CV

- **最新发布 378 篇**

- **更新 159 篇**

## 最新发布

#### [new 001] VideoPerceiver: Enhancing Fine-Grained Temporal Perception in Video Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出VideoPerceiver，针对视频多模态大模型在细粒度动作和瞬时事件理解上的不足。通过两阶段训练：利用“关键信息缺失”视频增强模型对细微运动的敏感性，并设计相对奖励机制强化对完整视频的依赖，从而提升对短时动作与罕见事件的感知能力。**

- **链接: [https://arxiv.org/pdf/2511.18823v1](https://arxiv.org/pdf/2511.18823v1)**

> **作者:** Fufangchen Zhao; Liao Zhang; Daiqi Shi; Yuanjun Gao; Chen Ye; Yang Cai; Jian Gao; Danfeng Yan
>
> **摘要:** We propose VideoPerceiver, a novel video multimodal large language model (VMLLM) that enhances fine-grained perception in video understanding, addressing VMLLMs' limited ability to reason about brief actions in short clips or rare transient events in long videos. VideoPerceiver adopts a two-stage training framework. During supervised fine-tuning (SFT), we construct "key-information-missing" videos by extracting event-action keywords from captions, identifying corresponding key frames, and replacing them with adjacent frames. We jointly encode original and modified video tokens with text tokens, aligning intermediate visual representations with keywords via an auxiliary contrastive loss to enhance sensitivity to fine-grained motion cues. In reinforcement learning (RL), both video variants are fed into the model to generate descriptions, and a novel relative reward ensures responses from complete videos outperform those from degraded inputs, explicitly training the model to recover temporally precise action details. We also curate a dataset of 80,000 videos with fine-grained actions and transient events. Experiments show VideoPerceiver substantially outperforms state-of-the-art VMLLMs on fine-grained action understanding and rare event captioning benchmarks, while maintaining strong performance on standard tasks. By prioritizing task-relevant visual features, our work redefines video-language model training for fine-grained perception.
>
---
#### [new 002] UMCL: Unimodal-generated Multimodal Contrastive Learning for Cross-compression-rate Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文针对社交媒体中多压缩率下的深度伪造检测难题，提出UMCL框架。通过单视觉模态生成多模态特征，利用语义对齐与对比学习增强跨压缩率鲁棒性，有效解决特征退化与多模态数据不一致问题，提升检测性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.18983v1](https://arxiv.org/pdf/2511.18983v1)**

> **作者:** Ching-Yi Lai; Chih-Yu Jian; Pei-Cheng Chuang; Chia-Ming Lee; Chih-Chung Hsu; Chiou-Ting Hsu; Chia-Wen Lin
>
> **备注:** 24-page manuscript accepted to IJCV
>
> **摘要:** In deepfake detection, the varying degrees of compression employed by social media platforms pose significant challenges for model generalization and reliability. Although existing methods have progressed from single-modal to multimodal approaches, they face critical limitations: single-modal methods struggle with feature degradation under data compression in social media streaming, while multimodal approaches require expensive data collection and labeling and suffer from inconsistent modal quality or accessibility in real-world scenarios. To address these challenges, we propose a novel Unimodal-generated Multimodal Contrastive Learning (UMCL) framework for robust cross-compression-rate (CCR) deepfake detection. In the training stage, our approach transforms a single visual modality into three complementary features: compression-robust rPPG signals, temporal landmark dynamics, and semantic embeddings from pre-trained vision-language models. These features are explicitly aligned through an affinity-driven semantic alignment (ASA) strategy, which models inter-modal relationships through affinity matrices and optimizes their consistency through contrastive learning. Subsequently, our cross-quality similarity learning (CQSL) strategy enhances feature robustness across compression rates. Extensive experiments demonstrate that our method achieves superior performance across various compression rates and manipulation types, establishing a new benchmark for robust deepfake detection. Notably, our approach maintains high detection accuracy even when individual features degrade, while providing interpretable insights into feature relationships through explicit alignment.
>
---
#### [new 003] ReMatch: Boosting Representation through Matching for Multimodal Retrieval
- **分类: cs.CV**

- **简介: 该论文针对多模态检索任务，解决传统方法忽视大模型生成能力、导致表示弱化的问题。提出ReMatch框架，通过端到端训练使MLLM兼具生成与匹配能力，利用自回归匹配增强硬负样本梯度，并引入可学习令牌生成语义丰富的多模态嵌入，显著提升检索性能与零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19278v1](https://arxiv.org/pdf/2511.19278v1)**

> **作者:** Qianying Liu; Xiao Liang; Zhiqiang Zhang; Yibo Chen; Xu Tang; Zhongfei Qing; Fengfan Zhou; Yao Hu; Paul Henderson
>
> **摘要:** We present ReMatch, a framework that leverages the generative strength of MLLMs for multimodal retrieval. Previous approaches treated an MLLM as a simple encoder, ignoring its generative nature, and under-utilising its compositional reasoning and world knowledge. We instead train the embedding MLLM end-to-end with a chat-style generative matching stage. The matching stage uses the same MLLM to autoregressively decide relevance from multi-view inputs, including both raw data and its own projected embeddings for each query and document. It provides instance-wise discrimination supervision that complements a standard contrastive loss, offering stronger gradients on hard negatives and preserving the compositional strengths of the original MLLM. To obtain semantically richer multimodal embeddings, we use multiple learnable tokens to augment each input, generating fine-grained contextual, mutually orthogonal embeddings with low inference cost. Leveraging our established high-performance baseline,we assemble the ideas mentioned above into a powerful training recipe and achieve a new state-of-the-art on the Massive Multimodal Embedding Benchmark (MMEB). Our experiments show particularly strong zero-shot generalization results on five datasets, highlighting the robustness and transferability of ReMatch.
>
---
#### [new 004] Less Is More: An Explainable AI Framework for Lightweight Malaria Classification
- **分类: cs.CV**

- **简介: 该论文针对疟疾细胞图像的二分类任务，解决深度学习模型计算量大、不透明的问题。提出基于形态特征的轻量级EMFE框架，用少量可解释特征结合逻辑回归与随机森林，在CPU上实现97.15%准确率，显著优于深度学习模型，兼顾高效、透明与部署可行性。**

- **链接: [https://arxiv.org/pdf/2511.18083v1](https://arxiv.org/pdf/2511.18083v1)**

> **作者:** Md Abdullah Al Kafi; Raka Moni; Sumit Kumar Banshal
>
> **摘要:** Background and Objective: Deep learning models have high computational needs and lack interpretability but are often the first choice for medical image classification tasks. This study addresses whether complex neural networks are essential for the simple binary classification task of malaria. We introduce the Extracted Morphological Feature Engineered (EMFE) pipeline, a transparent, reproducible, and low compute machine learning approach tailored explicitly for simple cell morphology, designed to achieve deep learning performance levels on a simple CPU only setup with the practical aim of real world deployment. Methods: The study used the NIH Malaria Cell Images dataset, with two features extracted from each cell image: the number of non background pixels and the number of holes within the cell. Logistic Regression and Random Forest were compared against ResNet18, DenseNet121, MobileNetV2, and EfficientNet across accuracy, model size, and CPU inference time. An ensemble model was created by combining Logistic Regression and Random Forests to achieve higher accuracy while retaining efficiency. Results: The single variable Logistic Regression model achieved a test accuracy of 94.80 percent with a file size of 1.2 kB and negligible inference latency (2.3 ms). The two stage ensemble improved accuracy to 97.15 percent. In contrast, the deep learning methods require 13.6 MB to 44.7 MB of storage and show significantly higher inference times (68 ms). Conclusion: This study shows that a compact feature engineering approach can produce clinically meaningful classification performance while offering gains in transparency, reproducibility, speed, and deployment feasibility. The proposed pipeline demonstrates that simple interpretable features paired with lightweight models can serve as a practical diagnostic solution for environments with limited computational resources.
>
---
#### [new 005] SCALER: SAM-Enhanced Collaborative Learning for Label-Deficient Concealed Object Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对标签稀缺的隐匿物体分割（LDCOS）任务，提出SCALER框架。通过协同优化均值教师分割器与可学习SAM，实现双向监督：一阶段用熵与不确定性加权伪标签，二阶段利用扰动鲁棒性更新SAM。实验表明其在多种弱监督场景下显著提升性能，具备通用性。**

- **链接: [https://arxiv.org/pdf/2511.18136v1](https://arxiv.org/pdf/2511.18136v1)**

> **作者:** Chunming He; Rihan Zhang; Longxiang Tang; Ziyun Yang; Kai Li; Deng-Ping Fan; Sina Farsiu
>
> **备注:** 4 figures, 6 tables
>
> **摘要:** Existing methods for label-deficient concealed object segmentation (LDCOS) either rely on consistency constraints or Segment Anything Model (SAM)-based pseudo-labeling. However, their performance remains limited due to the intrinsic concealment of targets and the scarcity of annotations. This study investigates two key questions: (1) Can consistency constraints and SAM-based supervision be jointly integrated to better exploit complementary information and enhance the segmenter? and (2) beyond that, can the segmenter in turn guide SAM through reciprocal supervision, enabling mutual improvement? To answer these questions, we present SCALER, a unified collaborative framework toward LDCOS that jointly optimizes a mean-teacher segmenter and a learnable SAM. SCALER operates in two alternating phases. In \textbf{Phase \uppercase\expandafter{\romannumeral1}}, the segmenter is optimized under fixed SAM supervision using entropy-based image-level and uncertainty-based pixel-level weighting to select reliable pseudo-label regions and emphasize harder examples. In \textbf{Phase \uppercase\expandafter{\romannumeral2}}, SAM is updated via augmentation invariance and noise resistance losses, leveraging its inherent robustness to perturbations. Experiments demonstrate that SCALER yields consistent performance gains across eight semi- and weakly-supervised COS tasks. The results further suggest that SCALER can serve as a general training paradigm to enhance both lightweight segmenters and large foundation models under label-scarce conditions. Code will be released.
>
---
#### [new 006] UnfoldLDM: Deep Unfolding-based Blind Image Restoration with Latent Diffusion Priors
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对盲图像修复（BIR）任务，解决深度展开网络（DUNs）在未知退化下的依赖性与过平滑问题。提出UnfoldLDM，结合多粒度退化感知模块与抗退化扩散模型，实现退化估计与高频细节恢复，显著提升修复质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18152v1](https://arxiv.org/pdf/2511.18152v1)**

> **作者:** Chunming He; Rihan Zhang; Zheng Chen; Bowen Yang; CHengyu Fang; Yunlong Lin; Fengyang Xiao; Sina Farsiu
>
> **备注:** 6 figures, 11 tables
>
> **摘要:** Deep unfolding networks (DUNs) combine the interpretability of model-based methods with the learning ability of deep networks, yet remain limited for blind image restoration (BIR). Existing DUNs suffer from: (1) \textbf{Degradation-specific dependency}, as their optimization frameworks are tied to a known degradation model, making them unsuitable for BIR tasks; and (2) \textbf{Over-smoothing bias}, resulting from the direct feeding of gradient descent outputs, dominated by low-frequency content, into the proximal term, suppressing fine textures. To overcome these issues, we propose UnfoldLDM to integrate DUNs with latent diffusion model (LDM) for BIR. In each stage, UnfoldLDM employs a multi-granularity degradation-aware (MGDA) module as the gradient descent step. MGDA models BIR as an unknown degradation estimation problem and estimates both the holistic degradation matrix and its decomposed forms, enabling robust degradation removal. For the proximal step, we design a degradation-resistant LDM (DR-LDM) to extract compact degradation-invariant priors from the MGDA output. Guided by this prior, an over-smoothing correction transformer (OCFormer) explicitly recovers high-frequency components and enhances texture details. This unique combination ensures the final result is degradation-free and visually rich. Experiments show that our UnfoldLDM achieves a leading place on various BIR tasks and benefits downstream tasks. Moreover, our design is compatible with existing DUN-based methods, serving as a plug-and-play framework. Code will be released.
>
---
#### [new 007] Alternating Perception-Reasoning for Hallucination-Resistant Video Understanding
- **分类: cs.CV**

- **简介: 该论文针对视频理解任务中的幻觉问题，提出感知-推理交替框架（Video-PLR）。通过循环式感知推理机制增强证据充分性，并引入事实感知评估器（FAE）作为抗幻觉奖励，提升模型可靠性。实验表明其在3B/7B参数规模下均达领先性能，且数据效率高。**

- **链接: [https://arxiv.org/pdf/2511.18463v1](https://arxiv.org/pdf/2511.18463v1)**

> **作者:** Bowei Pu; Chuanbin Liu; Yifan Ge; Peichen Zhou; Yiwei Sun; Zhiyin Lu; Jiankang Wang; Hongtao Xie
>
> **备注:** 32 pages, 36 figures
>
> **摘要:** Sufficient visual perception is the foundation of video reasoning. Nevertheless, existing Video Reasoning LLMs suffer from perception shortcuts, relying on a flawed single-step perception paradigm. This paradigm describes the video and then conducts reasoning, which runs the risk of insufficient evidence and emergent hallucinations. To address these issues, we introduce a new framework that integrates a loop-based paradigm with an anti-hallucination reward. First, to address the insufficient evidence, we introduce the Perception Loop Reasoning (PLR) paradigm. Instead of describing the video at once, each loop requires the model to describe a video segment with precise timestamps, analyze this segment, and decide the next action. Second, for the risk of hallucinations, the Factual-Aware Evaluator (FAE) evaluates each perception result as a reliable anti-hallucination reward. This reward encourages the model to provide sufficient and precise video evidence. Our FAE, which performs comparably to GPT-4o, is tuned on our AnetHallu-117K, a large-scale hallucination judgment preference dataset. Extensive experiments show that our Video-PLR achieves the state-of-the-art in both 3B and 7B parameter scales and has the best data efficiency. Our code, models, and datasets are released on: https://github.com/BoweiPu/VideoPLR.
>
---
#### [new 008] UltraFlux: Data-Model Co-Design for High-quality Native 4K Text-to-Image Generation across Diverse Aspect Ratios
- **分类: cs.CV**

- **简介: 该论文针对高分辨率文本生成图像任务，解决4K分辨率下跨宽高比生成质量差的问题。提出UltraFlux模型，通过数据-模型联合设计，结合自适应位置编码、改进的VAE、梯度重平衡损失和分阶段美学课程学习，实现高质量、多宽高比的4K图像生成，性能超越主流开源与部分闭源模型。**

- **链接: [https://arxiv.org/pdf/2511.18050v1](https://arxiv.org/pdf/2511.18050v1)**

> **作者:** Tian Ye; Song Fei; Lei Zhu
>
> **备注:** Project Page: https://w2genai-lab.github.io/UltraFlux/
>
> **摘要:** Diffusion transformers have recently delivered strong text-to-image generation around 1K resolution, but we show that extending them to native 4K across diverse aspect ratios exposes a tightly coupled failure mode spanning positional encoding, VAE compression, and optimization. Tackling any of these factors in isolation leaves substantial quality on the table. We therefore take a data-model co-design view and introduce UltraFlux, a Flux-based DiT trained natively at 4K on MultiAspect-4K-1M, a 1M-image 4K corpus with controlled multi-AR coverage, bilingual captions, and rich VLM/IQA metadata for resolution- and AR-aware sampling. On the model side, UltraFlux couples (i) Resonance 2D RoPE with YaRN for training-window-, frequency-, and AR-aware positional encoding at 4K; (ii) a simple, non-adversarial VAE post-training scheme that improves 4K reconstruction fidelity; (iii) an SNR-Aware Huber Wavelet objective that rebalances gradients across timesteps and frequency bands; and (iv) a Stage-wise Aesthetic Curriculum Learning strategy that concentrates high-aesthetic supervision on high-noise steps governed by the model prior. Together, these components yield a stable, detail-preserving 4K DiT that generalizes across wide, square, and tall ARs. On the Aesthetic-Eval at 4096 benchmark and multi-AR 4K settings, UltraFlux consistently outperforms strong open-source baselines across fidelity, aesthetic, and alignment metrics, and-with a LLM prompt refiner-matches or surpasses the proprietary Seedream 4.0.
>
---
#### [new 009] MedVision: Dataset and Benchmark for Quantitative Medical Image Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MedVision，一个用于量化医学图像分析的大规模数据集与基准。针对现有视觉语言模型在定量任务（如肿瘤尺寸、角度测量）上表现不佳的问题，构建涵盖22个数据集的3080万图像-标注对，聚焦检测、肿瘤大小估计和角度距离测量三类任务，并通过微调显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.18676v1](https://arxiv.org/pdf/2511.18676v1)**

> **作者:** Yongcheng Yao; Yongshuo Zong; Raman Dutt; Yongxin Yang; Sotirios A Tsaftaris; Timothy Hospedales
>
> **备注:** 8 pages, 8 figures, 4 tables
>
> **摘要:** Current vision-language models (VLMs) in medicine are primarily designed for categorical question answering (e.g., "Is this normal or abnormal?") or qualitative descriptive tasks. However, clinical decision-making often relies on quantitative assessments, such as measuring the size of a tumor or the angle of a joint, from which physicians draw their own diagnostic conclusions. This quantitative reasoning capability remains underexplored and poorly supported in existing VLMs. In this work, we introduce MedVision, a large-scale dataset and benchmark specifically designed to evaluate and improve VLMs on quantitative medical image analysis. MedVision spans 22 public datasets covering diverse anatomies and modalities, with 30.8 million image-annotation pairs. We focus on three representative quantitative tasks: (1) detection of anatomical structures and abnormalities, (2) tumor/lesion (T/L) size estimation, and (3) angle/distance (A/D) measurement. Our benchmarks show that current off-the-shelf VLMs perform poorly on these tasks. However, with supervised fine-tuning on MedVision, we significantly enhance their performance across detection, T/L estimation, and A/D measurement, demonstrating reduced error rates and improved precision. This work provides a foundation for developing VLMs with robust quantitative reasoning capabilities in medical imaging. Code and data are available at https://medvision-vlm.github.io.
>
---
#### [new 010] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span
- **分类: cs.CV**

- **简介: 该论文聚焦于自指视角下3D视觉感知范围的预测任务，旨在解决人类视觉注意力在三维环境中未来焦点难以预知的问题。提出EgoSpanLift方法，将2D图像中的注视预测升级为3D场景建模，结合SLAM关键点与3D U-Net、单向变压器实现时空融合，构建了包含36.4万样本的基准数据集，显著提升3D预测性能。**

- **链接: [https://arxiv.org/pdf/2511.18470v1](https://arxiv.org/pdf/2511.18470v1)**

> **作者:** Heeseung Yun; Joonil Na; Jaeyeon Kim; Calvin Murdock; Gunhee Kim
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** People continuously perceive and interact with their surroundings based on underlying intentions that drive their exploration and behaviors. While research in egocentric user and scene understanding has focused primarily on motion and contact-based interaction, forecasting human visual perception itself remains less explored despite its fundamental role in guiding human actions and its implications for AR/VR and assistive technologies. We address the challenge of egocentric 3D visual span forecasting, predicting where a person's visual perception will focus next within their three-dimensional environment. To this end, we propose EgoSpanLift, a novel method that transforms egocentric visual span forecasting from 2D image planes to 3D scenes. EgoSpanLift converts SLAM-derived keypoints into gaze-compatible geometry and extracts volumetric visual span regions. We further combine EgoSpanLift with 3D U-Net and unidirectional transformers, enabling spatio-temporal fusion to efficiently predict future visual span in the 3D grid. In addition, we curate a comprehensive benchmark from raw egocentric multisensory data, creating a testbed with 364.6K samples for 3D visual span forecasting. Our approach outperforms competitive baselines for egocentric 2D gaze anticipation and 3D localization while achieving comparable results even when projected back onto 2D image planes without additional 2D-specific training.
>
---
#### [new 011] SineProject: Machine Unlearning for Stable Vision Language Alignment
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型在遗忘特定信息（如隐私或有害内容）时破坏视觉-语言对齐的问题。提出SineProject方法，通过正弦调制可训练参数改进冻结投影器的雅可比条件数，稳定跨模态嵌入，实现高效遗忘同时保留对良性查询的响应能力，在保持低计算开销下达成最优遗忘-保留权衡。**

- **链接: [https://arxiv.org/pdf/2511.18444v1](https://arxiv.org/pdf/2511.18444v1)**

> **作者:** Arpit Garg; Hemanth Saratchandran; Simon Lucey
>
> **备注:** In Submission
>
> **摘要:** Multimodal Large Language Models (MLLMs) increasingly need to forget specific knowledge such as unsafe or private information without requiring full retraining. However, existing unlearning methods often disrupt vision language alignment, causing models to reject both harmful and benign queries. We trace this failure to the projector network during unlearning, its Jacobian becomes severely illconditioned, leading to unstable optimization and drift in cross modal embeddings. We introduce SineProject, a simple method that augments the frozen projector with sinusoidally modulated trainable parameters, improving the Jacobian's spectral conditioning and stabilizing alignment throughout unlearning. Across standard safety and privacy unlearning benchmarks using LLaVA v1.5 7B and 13B, SineProject reduces benign query refusals while achieving complete forgetting of targeted information, yielding state of the art forget retain trade offs with negligible computational overhead.
>
---
#### [new 012] Collaborative Learning with Multiple Foundation Models for Source-Free Domain Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究源域无关的域自适应（SFDA）任务，旨在不使用源数据的情况下将预训练模型适配到未标记目标域。针对单一基础模型语义覆盖有限的问题，提出CoMA框架，协同利用CLIP与BLIP等多基础模型，通过双向对齐与知识迁移，捕捉全局与局部语义信息，并引入DMI机制增强真实依赖，提升适应稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2511.19147v1](https://arxiv.org/pdf/2511.19147v1)**

> **作者:** Huisoo Lee; Jisu Han; Hyunsouk Cho; Wonjun Hwang
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Source-Free Domain Adaptation (SFDA) aims to adapt a pre-trained source model to an unlabeled target domain without access to source data. Recent advances in Foundation Models (FMs) have introduced new opportunities for leveraging external semantic knowledge to guide SFDA. However, relying on a single FM is often insufficient, as it tends to bias adaptation toward a restricted semantic coverage, failing to capture diverse contextual cues under domain shift. To overcome this limitation, we propose a Collaborative Multi-foundation Adaptation (CoMA) framework that jointly leverages two different FMs (e.g., CLIP and BLIP) with complementary properties to capture both global semantics and local contextual cues. Specifically, we employ a bidirectional adaptation mechanism that (1) aligns different FMs with the target model for task adaptation while maintaining their semantic distinctiveness, and (2) transfers complementary knowledge from the FMs to the target model. To ensure stable adaptation under mini-batch training, we introduce Decomposed Mutual Information (DMI) that selectively enhances true dependencies while suppressing false dependencies arising from incomplete class coverage. Extensive experiments demonstrate that our method consistently outperforms existing state-of-the-art SFDA methods across four benchmarks, including Office-31, Office-Home, DomainNet-126, and VisDA, under the closed-set setting, while also achieving best results on partial-set and open-set variants.
>
---
#### [new 013] LRDUN: A Low-Rank Deep Unfolding Network for Efficient Spectral Compressive Imaging
- **分类: cs.CV**

- **简介: 该论文针对光谱压缩成像（SCI）中的重建任务，解决传统深度展开网络因高维数据处理导致的计算冗余与病态问题。提出低秩深度展开网络（LRDUN），通过低秩分解建模光谱基和子空间图像，结合广义特征展开机制，在降低计算成本的同时提升重建质量。**

- **链接: [https://arxiv.org/pdf/2511.18513v1](https://arxiv.org/pdf/2511.18513v1)**

> **作者:** He Huang; Yujun Guo; Wei He
>
> **备注:** 17 pages, 16 figures,
>
> **摘要:** Deep unfolding networks (DUNs) have achieved remarkable success and become the mainstream paradigm for spectral compressive imaging (SCI) reconstruction. Existing DUNs are derived from full-HSI imaging models, where each stage operates directly on the high-dimensional HSI, refining the entire data cube based on the single 2D coded measurement. However, this paradigm leads to computational redundancy and suffers from the ill-posed nature of mapping 2D residuals back to 3D space of HSI. In this paper, we propose two novel imaging models corresponding to the spectral basis and subspace image by explicitly integrating low-rank (LR) decomposition with the sensing model. Compared to recovering the full HSI, estimating these compact low-dimensional components significantly mitigates the ill-posedness. Building upon these novel models, we develop the Low-Rank Deep Unfolding Network (LRDUN), which jointly solves the two subproblems within an unfolded proximal gradient descent (PGD) framework. Furthermore, we introduce a Generalized Feature Unfolding Mechanism (GFUM) that decouples the physical rank in the data-fidelity term from the feature dimensionality in the prior module, enhancing the representational capacity and flexibility of the network. Extensive experiments on simulated and real datasets demonstrate that the proposed LRDUN achieves state-of-the-art (SOTA) reconstruction quality with significantly reduced computational cost.
>
---
#### [new 014] Rethinking the Encoding and Annotating of 3D Bounding Box: Corner-Aware 3D Object Detection from Point Clouds
- **分类: cs.CV**

- **简介: 该论文针对LiDAR 3D目标检测中中心回归不稳定的难题，提出角点对齐回归方法。通过将预测目标从稀疏区域的中心转移至密集区域的角点，利用几何约束实现弱监督学习，显著提升检测精度，仅用BEV角点标注即达全监督83%性能。**

- **链接: [https://arxiv.org/pdf/2511.17619v1](https://arxiv.org/pdf/2511.17619v1)**

> **作者:** Qinghao Meng; Junbo Yin; Jianbing Shen; Yunde Jia
>
> **备注:** 8 pages, 5 figures, 2 tables
>
> **摘要:** Center-aligned regression remains dominant in LiDAR-based 3D object detection, yet it suffers from fundamental instability: object centers often fall in sparse or empty regions of the bird's-eye-view (BEV) due to the front-surface-biased nature of LiDAR point clouds, leading to noisy and inaccurate bounding box predictions. To circumvent this limitation, we revisit bounding box representation and propose corner-aligned regression, which shifts the prediction target from unstable centers to geometrically informative corners that reside in dense, observable regions. Leveraging the inherent geometric constraints among corners and image 2D boxes, partial parameters of 3D bounding boxes can be recovered from corner annotations, enabling a weakly supervised paradigm without requiring complete 3D labels. We design a simple yet effective corner-aware detection head that can be plugged into existing detectors. Experiments on KITTI show our method improves performance by 3.5% AP over center-based baseline, and achieves 83% of fully supervised accuracy using only BEV corner clicks, demonstrating the effectiveness of our corner-aware regression strategy.
>
---
#### [new 015] DensifyBeforehand: LiDAR-assisted Content-aware Densification for Efficient and Quality 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对3D高斯点云渲染中因自适应密度控制导致的冗余点和浮点伪影问题，提出先验稠密化方法。通过融合稀疏LiDAR与单目深度图，实现语义与几何重要区域的感知采样，优化初始点云分布，提升视觉质量与计算效率，显著降低资源消耗与训练时间。**

- **链接: [https://arxiv.org/pdf/2511.19294v1](https://arxiv.org/pdf/2511.19294v1)**

> **作者:** Phurtivilai Patt; Leyang Huang; Yinqiang Zhang; Yang Lei
>
> **摘要:** This paper addresses the limitations of existing 3D Gaussian Splatting (3DGS) methods, particularly their reliance on adaptive density control, which can lead to floating artifacts and inefficient resource usage. We propose a novel densify beforehand approach that enhances the initialization of 3D scenes by combining sparse LiDAR data with monocular depth estimation from corresponding RGB images. Our ROI-aware sampling scheme prioritizes semantically and geometrically important regions, yielding a dense point cloud that improves visual fidelity and computational efficiency. This densify beforehand approach bypasses the adaptive density control that may introduce redundant Gaussians in the original pipeline, allowing the optimization to focus on the other attributes of 3D Gaussian primitives, reducing overlap while enhancing visual quality. Our method achieves comparable results to state-of-the-art techniques while significantly lowering resource consumption and training time. We validate our approach through extensive comparisons and ablation studies on four newly collected datasets, showcasing its effectiveness in preserving regions of interest in complex scenes.
>
---
#### [new 016] The Potential and Limitations of Vision-Language Models for Human Motion Understanding: A Case Study in Data-Driven Stroke Rehabilitation
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLMs）在中风康复中的应用，旨在自动量化康复剂量和评估功能障碍。针对视频数据，提出基于运动识别的框架，未进行特定训练即实现高阶活动分类与部分动作检测，但精细量化仍受限。结果表明VLMs具潜力但当前精度不足，需优化提示与后处理。**

- **链接: [https://arxiv.org/pdf/2511.17727v1](https://arxiv.org/pdf/2511.17727v1)**

> **作者:** Victor Li; Naveenraj Kamalakannan; Avinash Parnandi; Heidi Schambra; Carlos Fernandez-Granda
>
> **摘要:** Vision-language models (VLMs) have demonstrated remarkable performance across a wide range of computer-vision tasks, sparking interest in their potential for digital health applications. Here, we apply VLMs to two fundamental challenges in data-driven stroke rehabilitation: automatic quantification of rehabilitation dose and impairment from videos. We formulate these problems as motion-identification tasks, which can be addressed using VLMs. We evaluate our proposed framework on a cohort of 29 healthy controls and 51 stroke survivors. Our results show that current VLMs lack the fine-grained motion understanding required for precise quantification: dose estimates are comparable to a baseline that excludes visual information, and impairment scores cannot be reliably predicted. Nevertheless, several findings suggest future promise. With optimized prompting and post-processing, VLMs can classify high-level activities from a few frames, detect motion and grasp with moderate accuracy, and approximate dose counts within 25% of ground truth for mildly impaired and healthy participants, all without task-specific training or finetuning. These results highlight both the current limitations and emerging opportunities of VLMs for data-driven stroke rehabilitation and broader clinical video analysis.
>
---
#### [new 017] Latent Dirichlet Transformer VAE for Hyperspectral Unmixing with Bundled Endmembers
- **分类: cs.CV**

- **简介: 该论文针对高光谱解混任务，解决光谱混合导致纯物质特征模糊的问题。提出LDVAE-T模型，利用变压器捕捉全局上下文，结合狄利克雷先验约束丰度非负且和为1，并将材料建模为包含均值与结构化协方差的光谱包，实现对材料内在变异性的建模，提升解混精度。**

- **链接: [https://arxiv.org/pdf/2511.17757v1](https://arxiv.org/pdf/2511.17757v1)**

> **作者:** Giancarlo Giannetti; Faisal Z. Qureshi
>
> **摘要:** Hyperspectral images capture rich spectral information that enables per-pixel material identification; however, spectral mixing often obscures pure material signatures. To address this challenge, we propose the Latent Dirichlet Transformer Variational Autoencoder (LDVAE-T) for hyperspectral unmixing. Our model combines the global context modeling capabilities of transformer architectures with physically meaningful constraints imposed by a Dirichlet prior in the latent space. This prior naturally enforces the sum-to-one and non-negativity conditions essential for abundance estimation, thereby improving the quality of predicted mixing ratios. A key contribution of LDVAE-T is its treatment of materials as bundled endmembers, rather than relying on fixed ground truth spectra. In the proposed method our decoder predicts, for each endmember and each patch, a mean spectrum together with a structured (segmentwise) covariance that captures correlated spectral variability. Reconstructions are formed by mixing these learned bundles with Dirichlet-distributed abundances garnered from a transformer encoder, allowing the model to represent intrinsic material variability while preserving physical interpretability. We evaluate our approach on three benchmark datasets, Samson, Jasper Ridge, and HYDICE Urban and show that LDVAE-T consistently outperforms state-of-the-art models in abundance estimation and endmember extraction, as measured by root mean squared error and spectral angle distance, respectively.
>
---
#### [new 018] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment
- **分类: cs.CV**

- **简介: 该论文针对文本到动作生成任务中扩散模型存在的语义不一致问题，提出ReAlign方法。通过步感知奖励模型与奖励引导采样策略，提升文本与动作的对齐度，优化生成动作的质量与真实性。**

- **链接: [https://arxiv.org/pdf/2511.19217v1](https://arxiv.org/pdf/2511.19217v1)**

> **作者:** Wanjiang Weng; Xiaofeng Tan; Junbo Wang; Guo-Sen Xie; Pan Zhou; Hongsong Wang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Text-to-motion generation, which synthesizes 3D human motions from text inputs, holds immense potential for applications in gaming, film, and robotics. Recently, diffusion-based methods have been shown to generate more diversity and realistic motion. However, there exists a misalignment between text and motion distributions in diffusion models, which leads to semantically inconsistent or low-quality motions. To address this limitation, we propose Reward-guided sampling Alignment (ReAlign), comprising a step-aware reward model to assess alignment quality during the denoising sampling and a reward-guided strategy that directs the diffusion process toward an optimally aligned distribution. This reward model integrates step-aware tokens and combines a text-aligned module for semantic consistency and a motion-aligned module for realism, refining noisy motions at each timestep to balance probability density and alignment. Extensive experiments of both motion generation and retrieval tasks demonstrate that our approach significantly improves text-motion alignment and motion quality compared to existing state-of-the-art methods.
>
---
#### [new 019] Reconstruction-Driven Multimodal Representation Learning for Automated Media Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对广播媒体中多模态内容自动化理解难题，提出基于重建的多模态自编码器（MMAE），通过联合重建文本、音频、视觉数据，学习跨模态统一表征。解决了单模态模型无法捕捉复杂跨模态关系的问题，实现了端到端的元数据提取与语义聚类，显著提升聚类与对齐性能。**

- **链接: [https://arxiv.org/pdf/2511.17596v1](https://arxiv.org/pdf/2511.17596v1)**

> **作者:** Yassir Benhammou; Suman Kalyan; Sujay Kumar
>
> **备注:** 8 pages, 5 figures, 4 tables
>
> **摘要:** Broadcast and media organizations increasingly rely on artificial intelligence to automate the labor-intensive processes of content indexing, tagging, and metadata generation. However, existing AI systems typically operate on a single modality-such as video, audio, or text-limiting their understanding of complex, cross-modal relationships in broadcast material. In this work, we propose a Multimodal Autoencoder (MMAE) that learns unified representations across text, audio, and visual data, enabling end-to-end automation of metadata extraction and semantic clustering. The model is trained on the recently introduced LUMA dataset, a fully aligned benchmark of multimodal triplets representative of real-world media content. By minimizing joint reconstruction losses across modalities, the MMAE discovers modality-invariant semantic structures without relying on large paired or contrastive datasets. We demonstrate significant improvements in clustering and alignment metrics (Silhouette, ARI, NMI) compared to linear baselines, indicating that reconstruction-based multimodal embeddings can serve as a foundation for scalable metadata generation and cross-modal retrieval in broadcast archives. These results highlight the potential of reconstruction-driven multimodal learning to enhance automation, searchability, and content management efficiency in modern broadcast workflows.
>
---
#### [new 020] Perceptual-Evidence Anchored Reinforced Learning for Multimodal Reasoning
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型在多模态推理中因忽视视觉感知验证导致的幻觉与奖励滥用问题，提出PEARL方法。通过构建可验证的感知检查清单，双重分支协同强化感知与推理，确保推理基于真实视觉证据，显著提升多模态推理准确性。**

- **链接: [https://arxiv.org/pdf/2511.18437v1](https://arxiv.org/pdf/2511.18437v1)**

> **作者:** Chi Zhang; Haibo Qiu; Qiming Zhang; Yufei Xu; Zhixiong Zeng; Siqi Yang; Peng Shi; Lin Ma; Jing Zhang
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Large Language Models (LLMs) and is now being applied to Vision-Language Models (VLMs). However, vanilla RLVR for VLMs verifies only the final textual output, critically neglecting the foundational step of visual perception. This oversight leads to visual hallucinations and reward hacking, as reasoning built upon flawed perception is inherently unreliable. To address this, we propose PEARL (Perceptual-Evidence Anchored Reinforced Learning), a dual-branch, perception-reasoning synergistic that strengthens multimodal reasoning by explicitly anchoring it to verified visual evidence. For each reasoning-oriented QA instance, PEARL first derive a perception checklist -- a set of perception-oriented sub-questions with verifiable answers that probe the model's understanding of key visual evidence. During training, auxiliary rollouts on this checklist yield a perceptual reward that both directly reinforces the model's perception ability and acts as a fidelity gate for reasoning. If the model passes the perception check, its policy update is biased towards evidence-anchored reasoning. Otherwise, the process is halted to prevent reasoning from flawed premises. PEARL can be seamlessly integrated with popular RL methods like GRPO and DAPO. Comprehensive experiments show PEARL achieves substantial gains on multimodal reasoning benchmarks, e.g., a +9.7% improvement over the baseline and +6.6% over GRPO on MathVerse.
>
---
#### [new 021] Zero-Shot Video Deraining with Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出一种零样本视频去雨方法，解决现有方法依赖合成数据或固定相机、泛化性差的问题。通过利用预训练文本到视频扩散模型，在不微调的前提下，结合负向提示与注意力切换机制，实现对动态场景中真实雨水的高效去除，显著提升去雨效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18537v1](https://arxiv.org/pdf/2511.18537v1)**

> **作者:** Tuomas Varanka; Juan Luis Gonzalez; Hyeongwoo Kim; Pablo Garrido; Xu Yao
>
> **备注:** WACV 2026
>
> **摘要:** Existing video deraining methods are often trained on paired datasets, either synthetic, which limits their ability to generalize to real-world rain, or captured by static cameras, which restricts their effectiveness in dynamic scenes with background and camera motion. Furthermore, recent works in fine-tuning diffusion models have shown promising results, but the fine-tuning tends to weaken the generative prior, limiting generalization to unseen cases. In this paper, we introduce the first zero-shot video deraining method for complex dynamic scenes that does not require synthetic data nor model fine-tuning, by leveraging a pretrained text-to-video diffusion model that demonstrates strong generalization capabilities. By inverting an input video into the latent space of diffusion models, its reconstruction process can be intervened and pushed away from the model's concept of rain using negative prompting. At the core of our approach is an attention switching mechanism that we found is crucial for maintaining dynamic backgrounds as well as structural consistency between the input and the derained video, mitigating artifacts introduced by naive negative prompting. Our approach is validated through extensive experiments on real-world rain datasets, demonstrating substantial improvements over prior methods and showcasing robust generalization without the need for supervised training.
>
---
#### [new 022] Neural Texture Splatting: Expressive 3D Gaussian Splatting for View Synthesis, Geometry, and Dynamic Reconstruction
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对3D高斯点阵在视图合成、几何与动态重建中的表达能力不足问题，提出神经纹理点阵（NTS）。通过引入全局神经场预测每点的外观与几何特征，增强视图与时序依赖的表达，提升模型泛化性与效率，在多种任务和输入密度下均实现性能突破。**

- **链接: [https://arxiv.org/pdf/2511.18873v1](https://arxiv.org/pdf/2511.18873v1)**

> **作者:** Yiming Wang; Shaofei Wang; Marko Mihajlovic; Siyu Tang
>
> **备注:** SIGGRAPH Asia 2025 (conference track), Project page: https://19reborn.github.io/nts/
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a leading approach for high-quality novel view synthesis, with numerous variants extending its applicability to a broad spectrum of 3D and 4D scene reconstruction tasks. Despite its success, the representational capacity of 3DGS remains limited by the use of 3D Gaussian kernels to model local variations. Recent works have proposed to augment 3DGS with additional per-primitive capacity, such as per-splat textures, to enhance its expressiveness. However, these per-splat texture approaches primarily target dense novel view synthesis with a reduced number of Gaussian primitives, and their effectiveness tends to diminish when applied to more general reconstruction scenarios. In this paper, we aim to achieve concrete performance improvement over state-of-the-art 3DGS variants across a wide range of reconstruction tasks, including novel view synthesis, geometry and dynamic reconstruction, under both sparse and dense input settings. To this end, we introduce Neural Texture Splatting (NTS). At the core of our approach is a global neural field (represented as a hybrid of a tri-plane and a neural decoder) that predicts local appearance and geometric fields for each primitive. By leveraging this shared global representation that models local texture fields across primitives, we significantly reduce model size and facilitate efficient global information exchange, demonstrating strong generalization across tasks. Furthermore, our neural modeling of local texture fields introduces expressive view- and time-dependent effects, a critical aspect that existing methods fail to account for. Extensive experiments show that Neural Texture Splatting consistently improves models and achieves state-of-the-art results across multiple benchmarks.
>
---
#### [new 023] Extreme Model Compression for Edge Vision-Language Models: Sparse Temporal Token Fusion and Adaptive Neural Compression
- **分类: cs.CV**

- **简介: 该论文针对边缘设备上视觉-语言模型的高效部署问题，提出稀疏时空标记融合（STTF）与自适应神经压缩（ANC）技术。通过动态重用视觉标记和条件激活编码分支，显著降低参数量与计算开销，在保持高精度的同时实现低延迟推理，适用于资源受限的实时边缘场景。**

- **链接: [https://arxiv.org/pdf/2511.18504v1](https://arxiv.org/pdf/2511.18504v1)**

> **作者:** Md Tasnin Tanvir; Soumitra Das; Sk Md Abidar Rahaman; Ali Shiri Sichani
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** The demand for edge AI in vision-language tasks requires models that achieve real-time performance on resource-constrained devices with limited power and memory. This paper proposes two adaptive compression techniques -- Sparse Temporal Token Fusion (STTF) and Adaptive Neural Compression (ANC) -- that integrate algorithmic innovations with hardware-aware optimizations. Unlike previous approaches relying on static pruning or uniform scaling, STTF dynamically reuses visual tokens through event-driven change detection, while ANC conditionally activates encoder branches via a learned router, enabling fine-grained adaptation to scene complexity. Our 3B-parameter TinyGPT-STTF achieves CIDEr 131.2, BLEU-4 0.38, METEOR 0.31, and ROUGE-L 0.56 on the COCO 2017 test set, surpassing LLaVA-1.5 7B by 17.6 CIDEr points while using 2.3x fewer parameters and 62x fewer on-device FLOPs. TinyGPT-ANC reaches CIDEr 128.5. On event-based vision tasks, STTF reduces average token count by 84% (from 196 to 31 tokens) while preserving 95.6% accuracy on the DVS128 Gesture dataset, and ANC cuts FLOPs by up to 90% in low-motion scenes. Compared to strong baselines, our models improve accuracy by up to 4.4% and reduce latency by up to 13x. These results enable efficient deployment of capable vision-language models on real-world edge devices.
>
---
#### [new 024] Evaluating Dataset Watermarking for Fine-tuning Traceability of Customized Diffusion Models: A Comprehensive Benchmark and Removal Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对微调扩散模型的版权追踪问题，提出评估数据集水印的综合框架，涵盖通用性、可传递性和鲁棒性。通过实验揭示现有方法在真实威胁下的不足，并提出一种可完全移除水印但不影响微调效果的方法，凸显了水印技术在实际应用中的挑战。**

- **链接: [https://arxiv.org/pdf/2511.19316v1](https://arxiv.org/pdf/2511.19316v1)**

> **作者:** Xincheng Wang; Hanchi Sun; Wenjun Sun; Kejun Xue; Wangqiu Zhou; Jianbo Zhang; Wei Sun; Dandan Zhu; Xiongkuo Min; Jun Jia; Zhijun Fang
>
> **摘要:** Recent fine-tuning techniques for diffusion models enable them to reproduce specific image sets, such as particular faces or artistic styles, but also introduce copyright and security risks. Dataset watermarking has been proposed to ensure traceability by embedding imperceptible watermarks into training images, which remain detectable in outputs even after fine-tuning. However, current methods lack a unified evaluation framework. To address this, this paper establishes a general threat model and introduces a comprehensive evaluation framework encompassing Universality, Transmissibility, and Robustness. Experiments show that existing methods perform well in universality and transmissibility, and exhibit some robustness against common image processing operations, yet still fall short under real-world threat scenarios. To reveal these vulnerabilities, the paper further proposes a practical watermark removal method that fully eliminates dataset watermarks without affecting fine-tuning, highlighting a key challenge for future research.
>
---
#### [new 025] Functional Localization Enforced Deep Anomaly Detection Using Fundus Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对眼底图像中视网膜疾病异常检测任务，解决影像质量差异、早期病变微弱及数据集域偏移问题。通过ViT模型结合多种增强策略，在多数据集上实现高精度分类，并构建GANomaly异常检测器提供可解释性与泛化能力，支持临床决策。**

- **链接: [https://arxiv.org/pdf/2511.18627v1](https://arxiv.org/pdf/2511.18627v1)**

> **作者:** Jan Benedikt Ruhland; Thorsten Papenbrock; Jan-Peter Sowa; Ali Canbay; Nicole Eter; Bernd Freisleben; Dominik Heider
>
> **摘要:** Reliable detection of retinal diseases from fundus images is challenged by the variability in imaging quality, subtle early-stage manifestations, and domain shift across datasets. In this study, we systematically evaluated a Vision Transformer (ViT) classifier under multiple augmentation and enhancement strategies across several heterogeneous public datasets, as well as the AEyeDB dataset, a high-quality fundus dataset created in-house and made available for the research community. The ViT demonstrated consistently strong performance, with accuracies ranging from 0.789 to 0.843 across datasets and diseases. Diabetic retinopathy and age-related macular degeneration were detected reliably, whereas glaucoma remained the most frequently misclassified disease. Geometric and color augmentations provided the most stable improvements, while histogram equalization benefited datasets dominated by structural subtlety. Laplacian enhancement reduced performance across different settings. On the Papila dataset, the ViT with geometric augmentation achieved an AUC of 0.91, outperforming previously reported convolutional ensemble baselines (AUC of 0.87), underscoring the advantages of transformer architectures and multi-dataset training. To complement the classifier, we developed a GANomaly-based anomaly detector, achieving an AUC of 0.76 while providing inherent reconstruction-based explainability and robust generalization to unseen data. Probabilistic calibration using GUESS enabled threshold-independent decision support for future clinical implementation.
>
---
#### [new 026] DiffSeg30k: A Multi-Turn Diffusion Editing Benchmark for Localized AIGC Detection
- **分类: cs.CV**

- **简介: 该论文提出DiffSeg30k，一个用于局部AIGC检测的多轮扩散编辑基准数据集。针对现有检测方法无法定位编辑区域的问题，构建了包含30k张带像素级标注的图像，涵盖多种扩散模型与多轮编辑。任务为细粒度图像分割，旨在同时定位编辑区域并识别生成模型，推动高精度、可泛化的AI内容检测研究。**

- **链接: [https://arxiv.org/pdf/2511.19111v1](https://arxiv.org/pdf/2511.19111v1)**

> **作者:** Hai Ci; Ziheng Peng; Pei Yang; Yingxin Xuan; Mike Zheng Shou
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Diffusion-based editing enables realistic modification of local image regions, making AI-generated content harder to detect. Existing AIGC detection benchmarks focus on classifying entire images, overlooking the localization of diffusion-based edits. We introduce DiffSeg30k, a publicly available dataset of 30k diffusion-edited images with pixel-level annotations, designed to support fine-grained detection. DiffSeg30k features: 1) In-the-wild images--we collect images or image prompts from COCO to reflect real-world content diversity; 2) Diverse diffusion models--local edits using eight SOTA diffusion models; 3) Multi-turn editing--each image undergoes up to three sequential edits to mimic real-world sequential editing; and 4) Realistic editing scenarios--a vision-language model (VLM)-based pipeline automatically identifies meaningful regions and generates context-aware prompts covering additions, removals, and attribute changes. DiffSeg30k shifts AIGC detection from binary classification to semantic segmentation, enabling simultaneous localization of edits and identification of the editing models. We benchmark three baseline segmentation approaches, revealing significant challenges in semantic segmentation tasks, particularly concerning robustness to image distortions. Experiments also reveal that segmentation models, despite being trained for pixel-level localization, emerge as highly reliable whole-image classifiers of diffusion edits, outperforming established forgery classifiers while showing great potential in cross-generator generalization. We believe DiffSeg30k will advance research in fine-grained localization of AI-generated content by demonstrating the promise and limitations of segmentation-based methods. DiffSeg30k is released at: https://huggingface.co/datasets/Chaos2629/Diffseg30k
>
---
#### [new 027] Large-Scale Pre-training Enables Multimodal AI Differentiation of Radiation Necrosis from Brain Metastasis Progression on Routine MRI
- **分类: cs.CV**

- **简介: 该论文针对脑转移瘤治疗后放射性坏死与肿瘤进展的鉴别难题，提出一种两阶段多模态自监督学习方法。利用大规模未标注MRI数据预训练ViT模型，结合临床分割图进行微调，在公开数据集上实现高精度分类，显著优于传统方法，提供可解释的临床解决方案。**

- **链接: [https://arxiv.org/pdf/2511.18208v1](https://arxiv.org/pdf/2511.18208v1)**

> **作者:** Ahmed Gomaa; Annette Schwarz; Ludwig Singer; Arnd Dörfler; Matthias Stefan May; Pluvio Stephan; Ishita Sheth; Juliane Szkitsak; Katharina Breininger; Yixing Huang; Benjamin Frey; Oliver Schnell; Daniel Delev; Roland Coras; Daniel Höfler; Philipp Schubert; Jenny Stritzelberger; Sabine Semrau; Andreas Maier; Dieter H Heiland; Udo S. Gaipl; Andrea Wittig; Rainer Fietkau; Christoph Bert; Stefanie Corradini; Florian Putz
>
> **摘要:** Background: Differentiating radiation necrosis (RN) from tumor progression after stereotactic radiosurgery (SRS) remains a critical challenge in brain metastases. While histopathology represents the gold standard, its invasiveness limits feasibility. Conventional supervised deep learning approaches are constrained by scarce biopsy-confirmed training data. Self-supervised learning (SSL) overcomes this by leveraging the growing availability of large-scale unlabeled brain metastases imaging datasets. Methods: In a two-phase deep learning strategy inspired by the foundation model paradigm, a Vision Transformer (ViT) was pre-trained via SSL on 10,167 unlabeled multi-source T1CE MRI sub-volumes. The pre-trained ViT was then fine-tuned for RN classification using a two-channel input (T1CE MRI and segmentation masks) on the public MOLAB dataset (n=109) using 20% of datasets as same-center held-out test set. External validation was performed on a second-center test cohort (n=28). Results: The self-supervised model achieved an AUC of 0.916 on the same-center test set and 0.764 on the second center test set, surpassing the fully supervised ViT (AUC 0.624/0.496; p=0.001/0.008) and radiomics (AUC 0.807/0.691; p=0.005/0.014). Multimodal integration further improved performance (AUC 0.947/0.821; p=0.073/0.001). Attention map visualizations enabled interpretability showing the model focused on clinically relevant lesion subregions. Conclusion: Large-scale pre-training on increasingly available unlabeled brain metastases datasets substantially improves AI model performance. A two-phase multimodal deep learning strategy achieved high accuracy in differentiating radiation necrosis from tumor progression using only routine T1CE MRI and standard clinical data, providing an interpretable, clinically accessible solution that warrants further validation.
>
---
#### [new 028] HyM-UNet: Synergizing Local Texture and Global Context via Hybrid CNN-Mamba Architecture for Medical Image Segmentation
- **分类: cs.CV; cs.IR**

- **简介: 该论文针对医学图像分割任务，解决CNN局部感受野限制导致的全局结构建模不足问题。提出HyM-UNet架构，融合卷积模块与Mamba模块，利用浅层保留纹理、深层捕捉长程依赖，并设计Mamba引导的跳跃连接，动态抑制噪声，提升边界感知。在ISIC 2018上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17988v1](https://arxiv.org/pdf/2511.17988v1)**

> **作者:** Haodong Chen; Xianfei Han; Qwen
>
> **摘要:** Accurate organ and lesion segmentation is a critical prerequisite for computer-aided diagnosis. Convolutional Neural Networks (CNNs), constrained by their local receptive fields, often struggle to capture complex global anatomical structures. To tackle this challenge, this paper proposes a novel hybrid architecture, HyM-UNet, designed to synergize the local feature extraction capabilities of CNNs with the efficient global modeling capabilities of Mamba. Specifically, we design a Hierarchical Encoder that utilizes convolutional modules in the shallow stages to preserve high-frequency texture details, while introducing Visual Mamba modules in the deep stages to capture long-range semantic dependencies with linear complexity. To bridge the semantic gap between the encoder and the decoder, we propose a Mamba-Guided Fusion Skip Connection (MGF-Skip). This module leverages deep semantic features as gating signals to dynamically suppress background noise within shallow features, thereby enhancing the perception of ambiguous boundaries. We conduct extensive experiments on public benchmark dataset ISIC 2018. The results demonstrate that HyM-UNet significantly outperforms existing state-of-the-art methods in terms of Dice coefficient and IoU, while maintaining lower parameter counts and inference latency. This validates the effectiveness and robustness of the proposed method in handling medical segmentation tasks characterized by complex shapes and scale variations.
>
---
#### [new 029] Neural Geometry Image-Based Representations with Optimal Transport (OT)
- **分类: cs.CV**

- **简介: 该论文针对3D网格的高效存储与修复问题，提出基于最优传输的神经几何图像表示。通过将不规则网格转换为规则图像网格，实现无需解码器、单次前向传播的高质量重建，显著提升存储效率与处理速度，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18679v1](https://arxiv.org/pdf/2511.18679v1)**

> **作者:** Xiang Gao; Yuanpeng Liu; Xinmu Wang; Jiazhi Li; Minghao Guo; Yu Guo; Xiyun Song; Heather Yu; Zhiqiang Lao; Xianfeng David Gu
>
> **备注:** WACV2026 Rround 2 Accepted
>
> **摘要:** Neural representations for 3D meshes are emerging as an effective solution for compact storage and efficient processing. Existing methods often rely on neural overfitting, where a coarse mesh is stored and progressively refined through multiple decoder networks. While this can restore high-quality surfaces, it is computationally expensive due to successive decoding passes and the irregular structure of mesh data. In contrast, images have a regular structure that enables powerful super-resolution and restoration frameworks, but applying these advantages to meshes is difficult because their irregular connectivity demands complex encoder-decoder architectures. Our key insight is that a geometry image-based representation transforms irregular meshes into a regular image grid, making efficient image-based neural processing directly applicable. Building on this idea, we introduce our neural geometry image-based representation, which is decoder-free, storage-efficient, and naturally suited for neural processing. It stores a low-resolution geometry-image mipmap of the surface, from which high-quality meshes are restored in a single forward pass. To construct geometry images, we leverage Optimal Transport (OT), which resolves oversampling in flat regions and undersampling in feature-rich regions, and enables continuous levels of detail (LoD) through geometry-image mipmapping. Experimental results demonstrate state-of-the-art storage efficiency and restoration accuracy, measured by compression ratio (CR), Chamfer distance (CD), and Hausdorff distance (HD).
>
---
#### [new 030] Mitigating Long-Tail Bias in HOI Detection via Adaptive Diversity Cache
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦于人-物交互（HOI）检测任务，针对罕见交互类别数据稀疏导致的长尾偏差问题，提出无需训练的自适应多样性缓存（ADC）模块。通过构建类别专属缓存，动态聚合高置信度、多样化的特征表示，并引入频率感知机制增强对稀有类别的识别能力，显著提升稀有交互检测性能。**

- **链接: [https://arxiv.org/pdf/2511.18811v1](https://arxiv.org/pdf/2511.18811v1)**

> **作者:** Yuqiu Jiang; Xiaozhen Qiao; Tianyu Mei; Haojian Huang; Yifan Chen; Ye Zheng; Zhe Sun
>
> **摘要:** Human-Object Interaction (HOI) detection is a fundamental task in computer vision, empowering machines to comprehend human-object relationships in diverse real-world scenarios. Recent advances in VLMs have significantly improved HOI detection by leveraging rich cross-modal representations. However, most existing VLM-based approaches rely heavily on additional training or prompt tuning, resulting in substantial computational overhead and limited scalability, particularly in long-tailed scenarios where rare interactions are severely underrepresented. In this paper, we propose the Adaptive Diversity Cache (ADC) module, a novel training-free and plug-and-play mechanism designed to mitigate long-tail bias in HOI detection. ADC constructs class-specific caches that accumulate high-confidence and diverse feature representations during inference. The method incorporates frequency-aware cache adaptation that favors rare categories and is designed to enable robust prediction calibration without requiring additional training or fine-tuning. Extensive experiments on HICO-DET and V-COCO datasets show that ADC consistently improves existing HOI detectors, achieving up to +8.57\% mAP gain on rare categories and +4.39\% on the full dataset, demonstrating its effectiveness in mitigating long-tail bias while preserving overall performance.
>
---
#### [new 031] AdaPerceiver: Transformers with Adaptive Width, Depth, and Tokens
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出AdaPerceiver，一种可在深度、宽度和令牌数上统一自适应的Transformer架构，解决模型部署中计算资源与延迟约束不匹配的问题。通过联合训练实现多配置性能稳定，在图像分类、语义分割和深度估计任务中显著提升效率，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18105v1](https://arxiv.org/pdf/2511.18105v1)**

> **作者:** Purvish Jajal; Nick John Eliopoulos; Benjamin Shiue-Hal Chou; George K. Thiruvathukal; Yung-Hsiang Lu; James C. Davis
>
> **摘要:** Modern transformer architectures achieve remarkable performance across tasks and domains but remain rigid in how they allocate computation at inference time. Real-world deployment often requires models to adapt to diverse hardware and latency constraints, yet most approaches to dynamic computation focus on a single axis -- such as reducing the number of tokens. We present a novel capability: AdaPerceiver, the first transformer architecture with unified adaptivity across depth, width, and tokens within a single model. We propose an architecture that supports adaptivity along these axes. We couple this with an efficient joint training regime that ensures the model maintains performance across its various configurations. We evaluate AdaPerceiver on image classification, semantic segmentation, and depth estimation tasks. On image classification, AdaPerceiver expands the accuracy-throughput Pareto front. It achieves 85.4% accuracy while yielding 36% higher throughput than FlexiViT-L. On dense prediction, AdaPerceiver matches ViT-H/14 while having $\sim$26x fewer encoder FLOPs (floating-point operations) on semantic segmentation and depth estimation. Finally, we show how AdaPerceiver equipped with a policy can maintain ImageNet1K accuracy ($\pm0.1$ percentage points) while reducing FLOPs by $24-33$%.
>
---
#### [new 032] FilmSceneDesigner: Chaining Set Design for Procedural Film Scene Generation
- **分类: cs.CV**

- **简介: 该论文提出FilmSceneDesigner，针对影视场景设计依赖人工、效率低的问题，构建基于代理的链式框架与程序化生成管线，实现从自然语言描述到完整3D电影场景的自动化生成。通过结构化参数与专用资产数据集，提升场景结构合理性与影视真实感，支持虚拟预演等下游应用。**

- **链接: [https://arxiv.org/pdf/2511.19137v1](https://arxiv.org/pdf/2511.19137v1)**

> **作者:** Zhifeng Xie; Keyi Zhang; Yiye Yan; Yuling Guo; Fan Yang; Jiting Zhou; Mengtian Li
>
> **摘要:** Film set design plays a pivotal role in cinematic storytelling and shaping the visual atmosphere. However, the traditional process depends on expert-driven manual modeling, which is labor-intensive and time-consuming. To address this issue, we introduce FilmSceneDesigner, an automated scene generation system that emulates professional film set design workflow. Given a natural language description, including scene type, historical period, and style, we design an agent-based chaining framework to generate structured parameters aligned with film set design workflow, guided by prompt strategies that ensure parameter accuracy and coherence. On the other hand, we propose a procedural generation pipeline which executes a series of dedicated functions with the structured parameters for floorplan and structure generation, material assignment, door and window placement, and object retrieval and layout, ultimately constructing a complete film scene from scratch. Moreover, to enhance cinematic realism and asset diversity, we construct SetDepot-Pro, a curated dataset of 6,862 film-specific 3D assets and 733 materials. Experimental results and human evaluations demonstrate that our system produces structurally sound scenes with strong cinematic fidelity, supporting downstream tasks such as virtual previs, construction drawing and mood board creation.
>
---
#### [new 033] SwiftVGGT: A Scalable Visual Geometry Grounded Transformer for Large-Scale Scenes
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对大场景3D重建中精度与效率的权衡问题，提出SwiftVGGT。无需训练，通过自研点采样与单步SVD对齐，实现快速全局一致性建图，避免依赖外部VPR和迭代优化，显著提升速度，仅需33%推理时间即达顶尖重建质量。**

- **链接: [https://arxiv.org/pdf/2511.18290v1](https://arxiv.org/pdf/2511.18290v1)**

> **作者:** Jungho Lee; Minhyeok Lee; Sunghun Yang; Minseok Kang; Sangyoun Lee
>
> **备注:** Project Page: https://Jho-Yonsei.github.io/SwiftVGGT/
>
> **摘要:** 3D reconstruction in large-scale scenes is a fundamental task in 3D perception, but the inherent trade-off between accuracy and computational efficiency remains a significant challenge. Existing methods either prioritize speed and produce low-quality results, or achieve high-quality reconstruction at the cost of slow inference times. In this paper, we propose SwiftVGGT, a training-free method that significantly reduce inference time while preserving high-quality dense 3D reconstruction. To maintain global consistency in large-scale scenes, SwiftVGGT performs loop closure without relying on the external Visual Place Recognition (VPR) model. This removes redundant computation and enables accurate reconstruction over kilometer-scale environments. Furthermore, we propose a simple yet effective point sampling method to align neighboring chunks using a single Sim(3)-based Singular Value Decomposition (SVD) step. This eliminates the need for the Iteratively Reweighted Least Squares (IRLS) optimization commonly used in prior work, leading to substantial speed-ups. We evaluate SwiftVGGT on multiple datasets and show that it achieves state-of-the-art reconstruction quality while requiring only 33% of the inference time of recent VGGT-based large-scale reconstruction approaches.
>
---
#### [new 034] ReCoGS: Real-time ReColoring for Gaussian Splatting scenes
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对高斯点云场景的实时重着色问题，提出ReCoGS方法。通过用户友好的交互式管道，实现对预训练场景中区域的精准选择与重着色，解决现有方法在视图一致性、控制精度和计算效率上的不足，支持实时编辑与可视化。**

- **链接: [https://arxiv.org/pdf/2511.18441v1](https://arxiv.org/pdf/2511.18441v1)**

> **作者:** Lorenzo Rutayisire; Nicola Capodieci; Fabio Pellacini
>
> **备注:** Project page is available at https://github.com/loryruta/recogs
>
> **摘要:** Gaussian Splatting has emerged as a leading method for novel view synthesis, offering superior training efficiency and real-time inference compared to NeRF approaches, while still delivering high-quality reconstructions. Beyond view synthesis, this 3D representation has also been explored for editing tasks. Many existing methods leverage 2D diffusion models to generate multi-view datasets for training, but they often suffer from limitations such as view inconsistencies, lack of fine-grained control, and high computational demand. In this work, we focus specifically on the editing task of recoloring. We introduce a user-friendly pipeline that enables precise selection and recoloring of regions within a pre-trained Gaussian Splatting scene. To demonstrate the real-time performance of our method, we also present an interactive tool that allows users to experiment with the pipeline in practice. Code is available at https://github.com/loryruta/recogs.
>
---
#### [new 035] Adversarial Patch Attacks on Vision-Based Cargo Occupancy Estimation via Differentiable 3D Simulation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉货运占用率估计系统中的物理对抗补丁攻击。针对卷积分类器，利用可微3D渲染在真实场景模拟中优化补丁纹理，提升攻击成功率。实验表明，3D优化补丁在“空至满”攻击中达84.94%成功，验证了系统安全漏洞，并提出增强鲁棒性的方向。**

- **链接: [https://arxiv.org/pdf/2511.19254v1](https://arxiv.org/pdf/2511.19254v1)**

> **作者:** Mohamed Rissal Hedna; Sesugh Samuel Nder
>
> **备注:** 9 pages, 5 figures, 1 algorithm
>
> **摘要:** Computer vision systems are increasingly adopted in modern logistics operations, including the estimation of trailer occupancy for planning, routing, and billing. Although effective, such systems may be vulnerable to physical adversarial attacks, particularly adversarial patches that can be printed and placed on interior surfaces. In this work, we study the feasibility of such attacks on a convolutional cargo-occupancy classifier using fully simulated 3D environments. Using Mitsuba 3 for differentiable rendering, we optimize patch textures across variations in geometry, lighting, and viewpoint, and compare their effectiveness to a 2D compositing baseline. Our experiments demonstrate that 3D-optimized patches achieve high attack success rates, especially in a denial-of-service scenario (empty to full), where success reaches 84.94 percent. Concealment attacks (full to empty) prove more challenging but still reach 30.32 percent. We analyze the factors influencing attack success, discuss implications for the security of automated logistics pipelines, and highlight directions for strengthening physical robustness. To our knowledge, this is the first study to investigate adversarial patch attacks for cargo-occupancy estimation in physically realistic, fully simulated 3D scenes.
>
---
#### [new 036] One4D: Unified 4D Generation and Reconstruction via Decoupled LoRA Control
- **分类: cs.CV**

- **简介: 该论文提出One4D，统一的4D生成与重建框架，解决动态4D内容（RGB帧与点云图）在不同条件下的生成难题。通过统一掩码条件机制和解耦LoRA控制，实现单图生成、全视频重建及稀疏帧混合任务的无缝切换，有效保持图像与点云的一致性。**

- **链接: [https://arxiv.org/pdf/2511.18922v1](https://arxiv.org/pdf/2511.18922v1)**

> **作者:** Zhenxing Mi; Yuxin Wang; Dan Xu
>
> **备注:** Project page: https://mizhenxing.github.io/One4D
>
> **摘要:** We present One4D, a unified framework for 4D generation and reconstruction that produces dynamic 4D content as synchronized RGB frames and pointmaps. By consistently handling varying sparsities of conditioning frames through a Unified Masked Conditioning (UMC) mechanism, One4D can seamlessly transition between 4D generation from a single image, 4D reconstruction from a full video, and mixed generation and reconstruction from sparse frames. Our framework adapts a powerful video generation model for joint RGB and pointmap generation, with carefully designed network architectures. The commonly used diffusion finetuning strategies for depthmap or pointmap reconstruction often fail on joint RGB and pointmap generation, quickly degrading the base video model. To address this challenge, we introduce Decoupled LoRA Control (DLC), which employs two modality-specific LoRA adapters to form decoupled computation branches for RGB frames and pointmaps, connected by lightweight, zero-initialized control links that gradually learn mutual pixel-level consistency. Trained on a mixture of synthetic and real 4D datasets under modest computational budgets, One4D produces high-quality RGB frames and accurate pointmaps across both generation and reconstruction tasks. This work represents a step toward general, high-quality geometry-based 4D world modeling using video diffusion models. Project page: https://mizhenxing.github.io/One4D
>
---
#### [new 037] Uncertainty Quantification in HSI Reconstruction using Physics-Aware Diffusion Priors and Optics-Encoded Measurements
- **分类: cs.CV**

- **简介: 该论文针对压缩感知下的高光谱图像（HSI）重建这一病态逆问题，提出基于物理先验的扩散模型HSDiff。通过引入区域级代谢增强和光谱上采样策略，提升先验多样性与不确定性校准能力，实现多模型下高保真、可解释的重建。**

- **链接: [https://arxiv.org/pdf/2511.18473v1](https://arxiv.org/pdf/2511.18473v1)**

> **作者:** Juan Romero; Qiang Fu; Matteo Ravasi; Wolfgang Heidrich
>
> **摘要:** Hyperspectral image reconstruction from a compressed measurement is a highly ill-posed inverse problem. Current data-driven methods suffer from hallucination due to the lack of spectral diversity in existing hyperspectral image datasets, particularly when they are evaluated for the metamerism phenomenon. In this work, we formulate hyperspectral image (HSI) reconstruction as a Bayesian inference problem and propose a framework, HSDiff, that utilizes an unconditionally trained, pixel-level diffusion prior and posterior diffusion sampling to generate diverse HSI samples consistent with the measurements of various hyperspectral image formation models. We propose an enhanced metameric augmentation technique using region-based metameric black and partition-of-union spectral upsampling to expand training with physically valid metameric spectra, strengthening the prior diversity and improving uncertainty calibration. We utilize HSDiff to investigate how the studied forward models shape the posterior distribution and demonstrate that guiding with effective spectral encoding provides calibrated informative uncertainty compared to non-encoded models. Through the lens of the Bayesian framework, HSDiff offers a complete, high-performance method for uncertainty-aware HSI reconstruction. Our results also reiterate the significance of effective spectral encoding in snapshot hyperspectral imaging.
>
---
#### [new 038] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling
- **分类: cs.CV**

- **简介: 该论文针对长尾数据集上的数据蒸馏任务，解决因类别不平衡导致的模型偏差与统计估计失真问题。提出统一框架，通过增强专家模型、动态校准批量归一化统计量、多轮初始化合成图像，实现无偏恢复与软标签重标注，显著提升蒸馏性能。**

- **链接: [https://arxiv.org/pdf/2511.18858v1](https://arxiv.org/pdf/2511.18858v1)**

> **作者:** Xiao Cui; Yulei Qin; Xinyue Li; Wengang Zhou; Hongsheng Li; Houqiang Li
>
> **备注:** AAAI 2026 (Oral)
>
> **摘要:** Dataset distillation creates a small distilled set that enables efficient training by capturing key information from the full dataset. While existing dataset distillation methods perform well on balanced datasets, they struggle under long-tailed distributions, where imbalanced class frequencies induce biased model representations and corrupt statistical estimates such as Batch Normalization (BN) statistics. In this paper, we rethink long-tailed dataset distillation by revisiting the limitations of trajectory-based methods, and instead adopt the statistical alignment perspective to jointly mitigate model bias and restore fair supervision. To this end, we introduce three dedicated components that enable unbiased recovery of distilled images and soft relabeling: (1) enhancing expert models (an observer model for recovery and a teacher model for relabeling) to enable reliable statistics estimation and soft-label generation; (2) recalibrating BN statistics via a full forward pass with dynamically adjusted momentum to reduce representation skew; (3) initializing synthetic images by incrementally selecting high-confidence and diverse augmentations via a multi-round mechanism that promotes coverage and diversity. Extensive experiments on four long-tailed benchmarks show consistent improvements over state-of-the-art methods across varying degrees of class imbalance.Notably, our approach improves top-1 accuracy by 15.6% on CIFAR-100-LT and 11.8% on Tiny-ImageNet-LT under IPC=10 and IF=10.
>
---
#### [new 039] SciEducator: Scientific Video Understanding and Educating via Deming-Cycle Multi-Agent System
- **分类: cs.CV**

- **简介: 该论文提出SciEducator，一个基于德明循环的多智能体系统，用于科学视频理解与教育。针对科学视频需专业知识与严谨推理的问题，构建自进化推理机制，生成多模态教育内容。建立SciVBench基准，实验证明其显著优于主流模型，开创科学视频教育新范式。**

- **链接: [https://arxiv.org/pdf/2511.17943v1](https://arxiv.org/pdf/2511.17943v1)**

> **作者:** Zhiyu Xu; Weilong Yan; Yufei Shi; Xin Meng; Tao He; Huiping Zhuang; Ming Li; Hehe Fan
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) and video agent systems have significantly improved general video understanding. However, when applied to scientific video understanding and educating, a domain that demands external professional knowledge integration and rigorous step-wise reasoning, existing approaches often struggle. To bridge this gap, we propose SciEducator, the first iterative self-evolving multi-agent system for scientific video comprehension and education. Rooted in the classical Deming Cycle from management science, our design reformulates its Plan-Do-Study-Act philosophy into a self-evolving reasoning and feedback mechanism, which facilitates the interpretation of intricate scientific activities in videos. Moreover, SciEducator can produce multimodal educational content tailored to specific scientific processes, including textual instructions, visual guides, audio narrations, and interactive references. To support evaluation, we construct SciVBench, a benchmark consisting of 500 expert-verified and literature-grounded science QA pairs across five categories, covering physical, chemical, and everyday phenomena. Extensive experiments demonstrate that SciEducator substantially outperforms leading closed-source MLLMs (e.g., Gemini, GPT-4o) and state-of-the-art video agents on the benchmark, establishing a new paradigm for the community.
>
---
#### [new 040] Beyond Description: Cognitively Benchmarking Fine-Grained Action for Embodied Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对具身智能体在复杂物理环境中执行精细动作的能力评估问题，提出CFG-Bench基准，涵盖四类认知能力的多模态问答数据。通过系统评测发现主流多模态大模型在细粒度动作理解与高阶推理上存在显著不足，并验证了针对性微调可有效提升性能。**

- **链接: [https://arxiv.org/pdf/2511.18685v1](https://arxiv.org/pdf/2511.18685v1)**

> **作者:** Dayong Liu; Chao Xu; Weihong Chen; Suyu Zhang; Juncheng Wang; Jiankang Deng; Baigui Sun; Yang Liu
>
> **摘要:** Multimodal Large Language Models (MLLMs) show promising results as decision-making engines for embodied agents operating in complex, physical environments. However, existing benchmarks often prioritize high-level planning or spatial reasoning, leaving the fine-grained action intelligence required for embodied physical interaction underexplored. To address this gap, we introduce CFG-Bench, a new benchmark designed to systematically evaluate this crucial capability. CFG-Bench consists of 1,368 curated videos paired with 19,562 three-modalities question-answer pairs targeting four cognitive abilities: 1) Physical Interaction, 2) Temporal-Causal Relation, 3) Intentional Understanding, and 4) Evaluative Judgment. Together, these dimensions provide a systematic framework for assessing a model's ability to translate visual observations into actionable knowledge, moving beyond mere surface-level recognition. Our comprehensive evaluation on CFG-Bench reveals that leading MLLMs struggle to produce detailed instructions for physical interactions and exhibit profound limitations in the higher-order reasoning of intention and evaluation. Moreover, supervised fine-tuning (SFT) on our data demonstrates that teaching an MLLMs to articulate fine-grained actions directly translates to significant performance gains on established embodied benchmarks. Our analysis highlights these limitations and offers insights for developing more capable and grounded embodied agents.
>
---
#### [new 041] Muskie: Multi-view Masked Image Modeling for 3D Vision Pre-training
- **分类: cs.CV**

- **简介: 该论文提出Muskie，一种面向3D视觉任务的多视角预训练模型。针对现有方法多视图一致性差的问题，通过跨视图掩码重建和激进遮蔽策略，使模型学习到视图不变特征与几何理解，无需3D监督。实验表明其在多视图匹配及相机位姿估计、点云重建等下游任务中表现更优。**

- **链接: [https://arxiv.org/pdf/2511.18115v1](https://arxiv.org/pdf/2511.18115v1)**

> **作者:** Wenyu Li; Sidun Liu; Peng Qiao; Yong Dou; Tongrui Hu
>
> **摘要:** We present Muskie, a native multi-view vision backbone designed for 3D vision tasks. Unlike existing models, which are frame-wise and exhibit limited multi-view consistency, Muskie is designed to process multiple views simultaneously and introduce multi-view consistency in pre-training stage. Muskie is trained to reconstruct heavily masked content in one view by finding and utilizing geometric correspondences from other views. Through this pretext task and our proposed aggressive masking strategy, the model implicitly to learn view-invariant features and develop strong geometric understanding without any 3D supervision. Compared with state-of-the-art frame-wise backbones such as DINO, Muskie achieves higher multi-view correspondence accuracy. Furthermore, we demonstrate that using Muskie as a backbone consistently enhances performance on downstream 3D tasks, including camera pose estimation and pointmap reconstruction. Codes are publicly available at https://leo-frank.github.io/Muskie/
>
---
#### [new 042] RoadBench: Benchmarking MLLMs on Fine-Grained Spatial Understanding and Reasoning under Urban Road Scenarios
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型在城市道路场景中细粒度空间理解与推理能力不足的问题，提出RoadBench基准。通过9121个验证数据，涵盖六项任务，评估模型对道路标记等细粒度元素的识别、联合理解与推理能力，揭示现有模型显著短板，推动其在复杂城市环境下的空间认知发展。**

- **链接: [https://arxiv.org/pdf/2511.18011v1](https://arxiv.org/pdf/2511.18011v1)**

> **作者:** Jun Zhang; Jie Feng; Long Chen; Junhui Wang; Zhicheng Liu; Depeng Jin; Yong Li
>
> **备注:** The code and data are publicly available at: https://github.com/tsinghua-fib-lab/RoadBench
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated powerful capabilities in general spatial understanding and reasoning. However, their fine-grained spatial understanding and reasoning capabilities in complex urban scenarios have not received significant attention in the fields of both research and industry. To fill this gap, we focus primarily on road markings as a typical example of fine-grained spatial elements under urban scenarios, given the essential role of the integrated road traffic network they form within cities. Around road markings and urban traffic systems, we propose RoadBench, a systematic benchmark that comprehensively evaluates MLLMs' fine-grained spatial understanding and reasoning capabilities using BEV and FPV image inputs. This benchmark comprises six tasks consisting of 9,121 strictly manually verified test cases. These tasks form a systematic evaluation framework that bridges understanding at local spatial scopes to global reasoning. They not only test MLLMs' capabilities in recognition, joint understanding, and reasoning but also assess their ability to integrate image information with domain knowledge. After evaluating 14 mainstream MLLMs, we confirm that RoadBench is a challenging benchmark for MLLMs while revealing significant shortcomings in existing MLLMs' fine-grained spatial understanding and reasoning capabilities within urban scenarios. In certain tasks, their performance even falls short of simple rule-based or random selection baselines. These findings, along with RoadBench itself, will contribute to the comprehensive advancement of spatial understanding capabilities for MLLMs. The benchmark code, example datasets, and raw evaluation results are available in the supplementary material.
>
---
#### [new 043] AttenDence: Maximizing Attention Confidence for Test Time Adaptation
- **分类: cs.CV**

- **简介: 该论文针对测试时自适应（TTA）任务，解决模型在分布偏移下性能下降的问题。提出通过最小化CLS token到图像块的注意力分布熵，增强模型对相关区域的关注信心，提升鲁棒性，且仅需单张测试图像即可有效工作。**

- **链接: [https://arxiv.org/pdf/2511.18925v1](https://arxiv.org/pdf/2511.18925v1)**

> **作者:** Yash Mali
>
> **备注:** Initial submission. 5 pages, 4 figures
>
> **摘要:** Test-time adaptation (TTA) enables models to adapt to distribution shifts at inference time. While entropy minimization over the output distribution has proven effective for TTA, transformers offer an additional unsupervised learning signal through their attention mechanisms. We propose minimizing the entropy of attention distributions from the CLS token to image patches as a novel TTA objective.This approach encourages the model to attend more confidently to relevant image regions under distribution shift and is effective even when only a single test image is available. We demonstrate that attention entropy minimization improves robustness across diverse corruption types while not hurting performance on clean data on a single sample stream of images at test time.
>
---
#### [new 044] MammothModa2: A Unified AR-Diffusion Framework for Multimodal Understanding and Generation
- **分类: cs.CV**

- **简介: 该论文提出MammothModa2，一个统一的自回归-扩散框架，用于多模态理解与生成。针对语义推理与高保真图像合成难以协同的问题，通过串行设计融合离散语义规划与连续图像生成，实现端到端训练下的高效生成与编辑，兼具强生成能力与多模态理解性能。**

- **链接: [https://arxiv.org/pdf/2511.18262v1](https://arxiv.org/pdf/2511.18262v1)**

> **作者:** Tao Shen; Xin Wan; Taicai Chen; Rui Zhang; Junwen Pan; Dawei Lu; Fanding Lei; Zhilin Lu; Yunfei Yang; Chen Cheng; Qi She; Chang Liu; Zhenbang Sun
>
> **摘要:** Unified multimodal models aim to integrate understanding and generation within a single framework, yet bridging the gap between discrete semantic reasoning and high-fidelity visual synthesis remains challenging. We present MammothModa2 (Mammoth2), a unified autoregressive-diffusion (AR-Diffusion) framework designed to effectively couple autoregressive semantic planning with diffusion-based generation. Mammoth2 adopts a serial design: an AR path equipped with generation experts performs global semantic modeling over discrete tokens, while a single-stream Diffusion Transformer (DiT) decoder handles high-fidelity image synthesis. A carefully designed AR-Diffusion feature alignment module combines multi-layer feature aggregation, unified condition encoding, and in-context conditioning to stably align AR's representations with the diffusion decoder's continuous latents. Mammoth2 is trained end-to-end with joint Next-Token Prediction and Flow Matching objectives, followed by supervised fine-tuning and reinforcement learning over both generation and editing. With roughly 60M supervised generation samples and no reliance on pre-trained generators, Mammoth2 delivers strong text-to-image and instruction-based editing performance on public benchmarks, achieving 0.87 on GenEval, 87.2 on DPGBench, and 4.06 on ImgEdit, while remaining competitive with understanding-only backbones (e.g., Qwen3-VL-8B) on multimodal understanding tasks. These results suggest that a carefully coupled AR-Diffusion architecture can provide high-fidelity generation and editing while maintaining strong multimodal comprehension within a single, parameter- and data-efficient model.
>
---
#### [new 045] AEGIS: Preserving privacy of 3D Facial Avatars with Adversarial Perturbations
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D面部虚拟形象的隐私泄露问题，提出AEGIS框架，通过在3D高斯点云颜色系数上施加对抗性扰动，实现跨视角一致的身份保护。解决了动态3D avatar在生物特征认证中易被识别的问题，确保身份去标识化的同时保持外观真实性和关键属性不变。**

- **链接: [https://arxiv.org/pdf/2511.17747v1](https://arxiv.org/pdf/2511.17747v1)**

> **作者:** Dawid Wolkiewicz; Anastasiya Pechko; Przemysław Spurek; Piotr Syga
>
> **摘要:** The growing adoption of photorealistic 3D facial avatars, particularly those utilizing efficient 3D Gaussian Splatting representations, introduces new risks of online identity theft, especially in systems that rely on biometric authentication. While effective adversarial masking methods have been developed for 2D images, a significant gap remains in achieving robust, viewpoint-consistent identity protection for dynamic 3D avatars. To address this, we present AEGIS, the first privacy-preserving identity masking framework for 3D Gaussian Avatars that maintains the subject's perceived characteristics. Our method aims to conceal identity-related facial features while preserving the avatar's perceptual realism and functional integrity. AEGIS applies adversarial perturbations to the Gaussian color coefficients, guided by a pre-trained face verification network, ensuring consistent protection across multiple viewpoints without retraining or modifying the avatar's geometry. AEGIS achieves complete de-identification, reducing face retrieval and verification accuracy to 0%, while maintaining high perceptual quality (SSIM = 0.9555, PSNR = 35.52 dB). It also preserves key facial attributes such as age, race, gender, and emotion, demonstrating strong privacy protection with minimal visual distortion.
>
---
#### [new 046] A Theory-Inspired Framework for Few-Shot Cross-Modal Sketch Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对手绘素描与真实图像的跨模态行人重识别任务，解决因模态差异大、标注数据少带来的泛化难题。提出KTCAA框架，基于泛化理论设计对齐增强与知识迁移催化剂，通过元学习实现少样本下的跨模态适应，显著提升模型在数据稀缺场景下的性能。**

- **链接: [https://arxiv.org/pdf/2511.18677v1](https://arxiv.org/pdf/2511.18677v1)**

> **作者:** Yunpeng Gong; Yongjie Hou; Jiangming Shi; Kim Long Diep; Min Jiang
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Sketch based person re-identification aims to match hand-drawn sketches with RGB surveillance images, but remains challenging due to significant modality gaps and limited annotated data. To address this, we introduce KTCAA, a theoretically grounded framework for few-shot cross-modal generalization. Motivated by generalization theory, we identify two key factors influencing target domain risk: (1) domain discrepancy, which quantifies the alignment difficulty between source and target distributions; and (2) perturbation invariance, which evaluates the model's robustness to modality shifts. Based on these insights, we propose two components: (1) Alignment Augmentation (AA), which applies localized sketch-style transformations to simulate target distributions and facilitate progressive alignment; and (2) Knowledge Transfer Catalyst (KTC), which enhances invariance by introducing worst-case perturbations and enforcing consistency. These modules are jointly optimized under a meta-learning paradigm that transfers alignment knowledge from data-rich RGB domains to sketch-based scenarios. Experiments on multiple benchmarks demonstrate that KTCAA achieves state-of-the-art performance, particularly in data-scarce conditions.
>
---
#### [new 047] Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera
- **分类: cs.CV**

- **简介: 该论文针对广角相机图像处理中平面CNN不适应球面几何的问题，提出统一球面前端USF。通过射线方向映射实现任意镜头的球面表示，基于空间域操作的旋转等变卷积，避免谐波变换，提升效率与鲁棒性，在分类、检测、分割任务中实现高效高精度，支持零样本迁移。**

- **链接: [https://arxiv.org/pdf/2511.18174v1](https://arxiv.org/pdf/2511.18174v1)**

> **作者:** Mukai Yu; Mosam Dabhi; Liuyue Xie; Sebastian Scherer; László A. Jeni
>
> **摘要:** Modern perception increasingly relies on fisheye, panoramic, and other wide field-of-view (FoV) cameras, yet most pipelines still apply planar CNNs designed for pinhole imagery on 2D grids, where image-space neighborhoods misrepresent physical adjacency and models are sensitive to global rotations. Frequency-domain spherical CNNs partially address this mismatch but require costly spherical harmonic transforms that constrain resolution and efficiency. We introduce the Unified Spherical Frontend (USF), a lens-agnostic framework that transforms images from any calibrated camera into a unit-sphere representation via ray-direction correspondences, and performs spherical resampling, convolution, and pooling directly in the spatial domain. USF is modular: projection, location sampling, interpolation, and resolution control are fully decoupled. Its distance-only spherical kernels offer configurable rotation-equivariance (mirroring translation-equivariance in planar CNNs) while avoiding harmonic transforms entirely. We compare standard planar backbones with their spherical counterparts across classification, detection, and segmentation tasks on synthetic (Spherical MNIST) and real-world datasets (PANDORA, Stanford 2D-3D-S), and stress-test robustness to extreme lens distortions, varying FoV, and arbitrary rotations. USF processes high-resolution spherical imagery efficiently and maintains less than 1% performance drop under random test-time rotations, even without rotational augmentation, and even enables zero-shot generalization from one lens type to unseen wide-FoV lenses with minimal performance degradation.
>
---
#### [new 048] HABIT: Human Action Benchmark for Interactive Traffic in CARLA
- **分类: cs.CV**

- **简介: 该论文提出HABIT，一个高保真交互式交通行人行为基准，用于自动驾驶仿真。针对现有模拟中人类行为简化的问题，引入真实人体动作数据，通过物理一致的动捕数据重定向，构建4730个标准化行人轨迹。该基准集成于CARLA，支持自动化场景生成与评估，揭示了主流AD系统在复杂交互中的安全缺陷。**

- **链接: [https://arxiv.org/pdf/2511.19109v1](https://arxiv.org/pdf/2511.19109v1)**

> **作者:** Mohan Ramesh; Mark Azer; Fabian B. Flohr
>
> **备注:** Accepted to WACV 2026. This is the pre-camera-ready version
>
> **摘要:** Current autonomous driving (AD) simulations are critically limited by their inadequate representation of realistic and diverse human behavior, which is essential for ensuring safety and reliability. Existing benchmarks often simplify pedestrian interactions, failing to capture complex, dynamic intentions and varied responses critical for robust system deployment. To overcome this, we introduce HABIT (Human Action Benchmark for Interactive Traffic), a high-fidelity simulation benchmark. HABIT integrates real-world human motion, sourced from mocap and videos, into CARLA (Car Learning to Act, a full autonomous driving simulator) via a modular, extensible, and physically consistent motion retargeting pipeline. From an initial pool of approximately 30,000 retargeted motions, we curate 4,730 traffic-compatible pedestrian motions, standardized in SMPL format for physically consistent trajectories. HABIT seamlessly integrates with CARLA's Leaderboard, enabling automated scenario generation and rigorous agent evaluation. Our safety metrics, including Abbreviated Injury Scale (AIS) and False Positive Braking Rate (FPBR), reveal critical failure modes in state-of-the-art AD agents missed by prior evaluations. Evaluating three state-of-the-art autonomous driving agents, InterFuser, TransFuser, and BEVDriver, demonstrates how HABIT exposes planner weaknesses that remain hidden in scripted simulations. Despite achieving close or equal to zero collisions per kilometer on the CARLA Leaderboard, the autonomous agents perform notably worse on HABIT, with up to 7.43 collisions/km and a 12.94% AIS 3+ injury risk, and they brake unnecessarily in up to 33% of cases. All components are publicly released to support reproducible, pedestrian-aware AI research.
>
---
#### [new 049] Optimal Pose Guidance for Stereo Calibration in 3D Deformation Measurement
- **分类: cs.CV**

- **简介: 该论文针对3D变形测量中立体校准效率低、精度差的问题，提出一种基于最优姿态引导的交互式校准框架。通过联合优化内外参并以协方差矩阵迹最小化为损失函数，自动生成下一最佳拍摄姿态，显著提升校准效率与精度，验证了其在真实场景中的高可靠性与应用潜力。**

- **链接: [https://arxiv.org/pdf/2511.18317v1](https://arxiv.org/pdf/2511.18317v1)**

> **作者:** Dongcai Tan; Shunkun Liang; Bin Li; Banglei Guan; Ang Su; Yuan Lin; Dapeng Zhang; Minggang Wan; Zibin Liu; Chenglong Wang; Jiajian Zhu; Zhang Li; Yang Shang; Qifeng Yu
>
> **摘要:** Stereo optical measurement techniques, such as digital image correlation (DIC), are widely used in 3D deformation measurement as non-contact, full-field measurement methods, in which stereo calibration is a crucial step. However, current stereo calibration methods lack intuitive optimal pose guidance, leading to inefficiency and suboptimal accuracy in deformation measurements. The aim of this study is to develop an interactive calibration framework that automatically generates the next optimal pose, enabling high-accuracy stereo calibration for 3D deformation measurement. We propose a pose optimization method that introduces joint optimization of relative and absolute extrinsic parameters, with the minimization of the covariance matrix trace adopted as the loss function to solve for the next optimal pose. Integrated with this method is a user-friendly graphical interface, which guides even non-expert users to capture qualified calibration images. Our proposed method demonstrates superior efficiency (requiring fewer images) and accuracy (demonstrating lower measurement errors) compared to random pose, while maintaining robustness across varying FOVs. In the thermal deformation measurement tests on an S-shaped specimen, the results exhibit high agreement with finite element analysis (FEA) simulations in both deformation magnitude and evolutionary trends. We present a pose guidance method for high-precision stereo calibration in 3D deformation measurement. The simulation experiments, real-world experiments, and thermal deformation measurement applications all demonstrate the significant application potential of our proposed method in the field of 3D deformation measurement. Keywords: Stereo calibration, Optimal pose guidance, 3D deformation measurement, Digital image correlation
>
---
#### [new 050] Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion
- **分类: cs.CV; cs.GR**

- **简介: 该论文研究稀疏输入下的新视角合成任务，旨在从少量图像中生成连贯自然的视频序列。提出零样本、生成引导的框架，利用预训练视频扩散模型在测试时补全中间视图，通过不确定性感知机制与3D高斯点云迭代优化，实现无场景训练的高质量重建。**

- **链接: [https://arxiv.org/pdf/2511.17932v1](https://arxiv.org/pdf/2511.17932v1)**

> **作者:** Yan Xu; Yixing Wang; Stella X. Yu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Given just a few glimpses of a scene, can you imagine the movie playing out as the camera glides through it? That's the lens we take on \emph{sparse-input novel view synthesis}, not only as filling spatial gaps between widely spaced views, but also as \emph{completing a natural video} unfolding through space. We recast the task as \emph{test-time natural video completion}, using powerful priors from \emph{pretrained video diffusion models} to hallucinate plausible in-between views. Our \emph{zero-shot, generation-guided} framework produces pseudo views at novel camera poses, modulated by an \emph{uncertainty-aware mechanism} for spatial coherence. These synthesized frames densify supervision for \emph{3D Gaussian Splatting} (3D-GS) for scene reconstruction, especially in under-observed regions. An iterative feedback loop lets 3D geometry and 2D view synthesis inform each other, improving both the scene reconstruction and the generated views. The result is coherent, high-fidelity renderings from sparse inputs \emph{without any scene-specific training or fine-tuning}. On LLFF, DTU, DL3DV, and MipNeRF-360, our method significantly outperforms strong 3D-GS baselines under extreme sparsity.
>
---
#### [new 051] VDC-Agent: When Video Detailed Captioners Evolve Themselves via Agentic Self-Reflection
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文提出VDC-Agent，一种无需人工标注或大模型教师的自进化视频细粒度描述框架。通过生成、评分、提示优化与自我反思的闭环，自动构建18.886万对高质量数据，微调后模型在基准上达49.08%准确率，性能超越现有方法，且推理成本相近。**

- **链接: [https://arxiv.org/pdf/2511.19436v1](https://arxiv.org/pdf/2511.19436v1)**

> **作者:** Qiang Wang; Xinyuan Gao; SongLin Dong; Jizhou Han; Jiangyang Li; Yuhang He; Yihong Gong
>
> **摘要:** We present VDC-Agent, a self-evolving framework for Video Detailed Captioning that requires neither human annotations nor larger teacher models. The agent forms a closed loop of caption generation, principle-guided scoring (score and textual suggestions), and prompt refinement. When caption quality regresses, a self-reflection path leverages the previous chain-of-thought to amend the update. Running this process on unlabeled videos produces trajectories of (caption, score) pairs. We convert the trajectories into preference tuples and filter out samples with JSON parsing errors, resulting in VDC-Agent-19K, which contains 18,886 automatically constructed pairs. We then fine-tune the base MLLM on this dataset using an easy-to-hard curriculum direct preference optimization. Built on Qwen2.5-VL-7B-Instruct, our VDC-Agent-7B attains state-of-the-art performance on the VDC benchmark with 49.08% average accuracy and 2.50 score, surpassing specialized video captioners and improving over the base model by +5.13% accuracy and +0.27 score at similar inference cost.
>
---
#### [new 052] Consolidating Diffusion-Generated Video Detection with Unified Multimodal Forgery Learning
- **分类: cs.CV**

- **简介: 该论文针对扩散模型生成视频的检测难题，提出MM-Det++算法，融合时空特征与多模态语义推理，通过统一多模态学习提升泛化能力。研究构建了大规模DVF数据集，推动视频伪造检测发展。**

- **链接: [https://arxiv.org/pdf/2511.18104v1](https://arxiv.org/pdf/2511.18104v1)**

> **作者:** Xiaohong Liu; Xiufeng Song; Huayu Zheng; Lei Bai; Xiaoming Liu; Guangtao Zhai
>
> **备注:** Code and dataset are available at https://github.com/SparkleXFantasy/MM-Det-Plus
>
> **摘要:** The proliferation of videos generated by diffusion models has raised increasing concerns about information security, highlighting the urgent need for reliable detection of synthetic media. Existing methods primarily focus on image-level forgery detection, leaving generic video-level forgery detection largely underexplored. To advance video forensics, we propose a consolidated multimodal detection algorithm, named MM-Det++, specifically designed for detecting diffusion-generated videos. Our approach consists of two innovative branches and a Unified Multimodal Learning (UML) module. Specifically, the Spatio-Temporal (ST) branch employs a novel Frame-Centric Vision Transformer (FC-ViT) to aggregate spatio-temporal information for detecting diffusion-generated videos, where the FC-tokens enable the capture of holistic forgery traces from each video frame. In parallel, the Multimodal (MM) branch adopts a learnable reasoning paradigm to acquire Multimodal Forgery Representation (MFR) by harnessing the powerful comprehension and reasoning capabilities of Multimodal Large Language Models (MLLMs), which discerns the forgery traces from a flexible semantic perspective. To integrate multimodal representations into a coherent space, a UML module is introduced to consolidate the generalization ability of MM-Det++. In addition, we also establish a large-scale and comprehensive Diffusion Video Forensics (DVF) dataset to advance research in video forgery detection. Extensive experiments demonstrate the superiority of MM-Det++ and highlight the effectiveness of unified multimodal forgery learning in detecting diffusion-generated videos.
>
---
#### [new 053] STCDiT: Spatio-Temporally Consistent Diffusion Transformer for High-Quality Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文提出STCDiT框架，用于高保真视频超分辨率任务。针对复杂运动下视频时序不一致与结构失真问题，设计了基于运动感知的分段重建和锚帧引导机制，利用锚帧保持结构信息，提升生成质量与时间一致性。**

- **链接: [https://arxiv.org/pdf/2511.18786v1](https://arxiv.org/pdf/2511.18786v1)**

> **作者:** Junyang Chen; Jiangxin Dong; Long Sun; Yixin Yang; Jinshan Pan
>
> **备注:** Project page: https://jychen9811.github.io/STCDiT_page
>
> **摘要:** We present STCDiT, a video super-resolution framework built upon a pre-trained video diffusion model, aiming to restore structurally faithful and temporally stable videos from degraded inputs, even under complex camera motions. The main challenges lie in maintaining temporal stability during reconstruction and preserving structural fidelity during generation. To address these challenges, we first develop a motion-aware VAE reconstruction method that performs segment-wise reconstruction, with each segment clip exhibiting uniform motion characteristic, thereby effectively handling videos with complex camera motions. Moreover, we observe that the first-frame latent extracted by the VAE encoder in each clip, termed the anchor-frame latent, remains unaffected by temporal compression and retains richer spatial structural information than subsequent frame latents. We further develop an anchor-frame guidance approach that leverages structural information from anchor frames to constrain the generation process and improve structural fidelity of video features. Coupling these two designs enables the video diffusion model to achieve high-quality video super-resolution. Extensive experiments show that STCDiT outperforms state-of-the-art methods in terms of structural fidelity and temporal consistency.
>
---
#### [new 054] POUR: A Provably Optimal Method for Unlearning Representations via Neural Collapse
- **分类: cs.CV**

- **简介: 该论文针对计算机视觉中的机器遗忘任务，解决现有方法仅修改分类器而未彻底清除表示的问题。提出POUR方法，基于神经坍缩理论，通过几何投影实现表示层面的可证明最优遗忘，引入RUS评估指标，实现在保留知识的同时高效遗忘特定概念。**

- **链接: [https://arxiv.org/pdf/2511.19339v1](https://arxiv.org/pdf/2511.19339v1)**

> **作者:** Anjie Le; Can Peng; Yuyuan Liu; J. Alison Noble
>
> **摘要:** In computer vision, machine unlearning aims to remove the influence of specific visual concepts or training images without retraining from scratch. Studies show that existing approaches often modify the classifier while leaving internal representations intact, resulting in incomplete forgetting. In this work, we extend the notion of unlearning to the representation level, deriving a three-term interplay between forgetting efficacy, retention fidelity, and class separation. Building on Neural Collapse theory, we show that the orthogonal projection of a simplex Equiangular Tight Frame (ETF) remains an ETF in a lower dimensional space, yielding a provably optimal forgetting operator. We further introduce the Representation Unlearning Score (RUS) to quantify representation-level forgetting and retention fidelity. Building on this, we introduce POUR (Provably Optimal Unlearning of Representations), a geometric projection method with closed-form (POUR-P) and a feature-level unlearning variant under a distillation scheme (POUR-D). Experiments on CIFAR-10/100 and PathMNIST demonstrate that POUR achieves effective unlearning while preserving retained knowledge, outperforming state-of-the-art unlearning methods on both classification-level and representation-level metrics.
>
---
#### [new 055] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition
- **分类: cs.CV**

- **简介: 该论文针对细粒度图像识别中大视觉语言模型难以区分视觉相似类别的问题，提出DiVE-k框架。通过利用模型自身top-k预测生成多选题，采用强化学习训练模型进行细粒度差异推理，有效避免记忆训练类别，提升泛化能力，在多个基准上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18305v1](https://arxiv.org/pdf/2511.18305v1)**

> **作者:** Raja Kumar; Arka Sadhu; Ram Nevatia
>
> **摘要:** Large Vision Language Models (LVLMs) possess extensive text knowledge but struggles to utilize this knowledge for fine-grained image recognition, often failing to differentiate between visually similar categories. Existing fine-tuning methods using Reinforcement Learning (RL) with exact-match reward signals are often brittle, encourage memorization of training categories, and fail to elicit differential reasoning needed for generalization to unseen classes. To address this, we propose $\textbf{DiVE-k}$, $\textbf{Di}$fferential $\textbf{V}$isual r$\textbf{E}$asoning using top-$\textbf{k}$ generations, framework that leverages model's own top-k predictions as a training signal. For each training image, DiVE-k creates a multiple-choice question from the model's top-k outputs and uses RL to train the model to select the correct answer. This approach requires the model to perform fine-grained differential reasoning among plausible options and provides a simple, verifiable reward signal that mitigates memorization and improves generalization. Experiments on five standard fine-grained datasets show that our method significantly outperforms existing approaches. In the standard base-to-novel generalization setting, DiVE-k surpasses the QWEN2.5-VL-7B and ViRFT by 10.04% and 6.16% on the Harmonic Mean metric, respectively. Further experiments show similar gains in mixed-domain and few-shot scenarios.
>
---
#### [new 056] Multi-speaker Attention Alignment for Multimodal Social Interaction
- **分类: cs.CV**

- **简介: 该论文针对多说话人场景下视觉与语言模态对齐不足的问题，提出一种无需参数新增的注意力对齐方法。通过动态选择关键注意力头并引入社会感知偏置，增强说话人视觉与语音的关联性，显著提升多模态大模型在社交互动理解任务中的表现。**

- **链接: [https://arxiv.org/pdf/2511.17952v1](https://arxiv.org/pdf/2511.17952v1)**

> **作者:** Liangyang Ouyang; Yifei Huang; Mingfang Zhang; Caixin Kang; Ryosuke Furuta; Yoichi Sato
>
> **摘要:** Understanding social interaction in video requires reasoning over a dynamic interplay of verbal and non-verbal cues: who is speaking, to whom, and with what gaze or gestures. While Multimodal Large Language Models (MLLMs) are natural candidates, simply adding visual inputs yields surprisingly inconsistent gains on social tasks. Our quantitative analysis of cross-modal attention inside state-of-the-art MLLMs reveals a core failure mode: in multi-speaker scenes, visual and textual tokens lack speaker-consistent alignment, exhibiting substantially weaker cross-modal attention than in object-centric images. To address this, we propose a multimodal multi-speaker attention alignment method that can be integrated into existing MLLMs. First, we introduce dynamic cross-modal head selection to identify attention heads most responsible for grounding. Then, an adaptive social-aware attention bias, computed from existing attention patterns and speaker locations, is injected into the attention mechanism. This bias reinforces alignment between a speaker's visual representation and their utterances without introducing trainable parameters or architectural changes. We integrate our method into three distinct MLLMs (LLaVA-NeXT-Video, Qwen2.5-VL, and InternVL3) and evaluate on three benchmarks (TVQA+, MMSI, OnlineMMSI). Across four social tasks, results demonstrate that our approach improves the ability of MLLMs and achieves state-of-the-art results. Attention visualizations confirm our method successfully focuses the model on speaker-relevant regions, enabling more robust multi-party social reasoning. Our implementation and model will be available at https://github.com/ut-vision/SocialInteraction.
>
---
#### [new 057] SciPostLayoutTree: A Dataset for Structural Analysis of Scientific Posters
- **分类: cs.CV**

- **简介: 该论文针对科学海报结构分析任务，解决其阅读顺序与父子关系识别难题。构建了包含约8000张海报的SciPostLayoutTree数据集，涵盖复杂空间关系。提出Layout Tree Decoder模型，融合视觉与框位置信息，采用束搜索提升序列合理性，显著提升复杂关系预测准确率。**

- **链接: [https://arxiv.org/pdf/2511.18329v1](https://arxiv.org/pdf/2511.18329v1)**

> **作者:** Shohei Tanaka; Atsushi Hashimoto; Yoshitaka Ushiku
>
> **摘要:** Scientific posters play a vital role in academic communication by presenting ideas through visual summaries. Analyzing reading order and parent-child relations of posters is essential for building structure-aware interfaces that facilitate clear and accurate understanding of research content. Despite their prevalence in academic communication, posters remain underexplored in structural analysis research, which has primarily focused on papers. To address this gap, we constructed SciPostLayoutTree, a dataset of approximately 8,000 posters annotated with reading order and parent-child relations. Compared to an existing structural analysis dataset, SciPostLayoutTree contains more instances of spatially challenging relations, including upward, horizontal, and long-distance relations. As a solution to these challenges, we develop Layout Tree Decoder, which incorporates visual features as well as bounding box features including position and category information. The model also uses beam search to predict relations while capturing sequence-level plausibility. Experimental results demonstrate that our model improves the prediction accuracy for spatially challenging relations and establishes a solid baseline for poster structure analysis. The dataset is publicly available at https://huggingface.co/datasets/omron-sinicx/scipostlayouttree. The code is also publicly available at https://github.com/omron-sinicx/scipostlayouttree.
>
---
#### [new 058] IE-Critic-R1: Advancing the Explanatory Measurement of Text-Driven Image Editing for Human Perception Alignment
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对文本驱动图像编辑的评估难题，提出IE-Bench基准和IE-Critic-R1模型。旨在更准确衡量编辑质量并贴近人类感知。通过构建包含近4000样本的数据库及基于可验证奖励强化学习的可解释评估方法，显著提升主观评价一致性。**

- **链接: [https://arxiv.org/pdf/2511.18055v1](https://arxiv.org/pdf/2511.18055v1)**

> **作者:** Bowen Qu; Shangkun Sun; Xiaoyu Liang; Wei Gao
>
> **备注:** 18 pages, 10 figures, 8 tables
>
> **摘要:** Recent advances in text-driven image editing have been significant, yet the task of accurately evaluating these edited images continues to pose a considerable challenge. Different from the assessment of text-driven image generation, text-driven image editing is characterized by simultaneously conditioning on both text and a source image. The edited images often retain an intrinsic connection to the original image, which dynamically change with the semantics of the text. However, previous methods tend to solely focus on text-image alignment or have not well aligned with human perception. In this work, we introduce the Text-driven Image Editing Benchmark suite (IE-Bench) to enhance the assessment of text-driven edited images. IE-Bench includes a database contains diverse source images, various editing prompts and the corresponding edited results from different editing methods, and nearly 4,000 samples with corresponding Mean Opinion Scores (MOS) provided by 15 human subjects. Furthermore, we introduce IE-Critic-R1, which, benefiting from Reinforcement Learning from Verifiable Rewards (RLVR), provides more comprehensive and explainable quality assessment for text-driven image editing that aligns with human perception. Extensive experiments demonstrate IE-Critic-R1's superior subjective-alignments on the text-driven image editing task compared with previous metrics. Related data and codes are available to the public.
>
---
#### [new 059] SyncMV4D: Synchronized Multi-view Joint Diffusion of Appearance and Motion for Hand-Object Interaction Synthesis
- **分类: cs.CV**

- **简介: 该论文提出SyncMV4D，用于生成同步多视角手物交互视频与4D动态。针对现有方法在3D几何感知和真实运动生成上的不足，通过联合扩散模型与点对齐机制，实现视觉与动态的闭环协同，提升生成结果的视觉真实性和多视角一致性。**

- **链接: [https://arxiv.org/pdf/2511.19319v1](https://arxiv.org/pdf/2511.19319v1)**

> **作者:** Lingwei Dang; Zonghan Li; Juntong Li; Hongwen Zhang; Liang An; Yebin Liu; Qingyao Wu
>
> **备注:** Project Page: https://droliven.github.io/SyncMV4D
>
> **摘要:** Hand-Object Interaction (HOI) generation plays a critical role in advancing applications across animation and robotics. Current video-based methods are predominantly single-view, which impedes comprehensive 3D geometry perception and often results in geometric distortions or unrealistic motion patterns. While 3D HOI approaches can generate dynamically plausible motions, their dependence on high-quality 3D data captured in controlled laboratory settings severely limits their generalization to real-world scenarios. To overcome these limitations, we introduce SyncMV4D, the first model that jointly generates synchronized multi-view HOI videos and 4D motions by unifying visual prior, motion dynamics, and multi-view geometry. Our framework features two core innovations: (1) a Multi-view Joint Diffusion (MJD) model that co-generates HOI videos and intermediate motions, and (2) a Diffusion Points Aligner (DPA) that refines the coarse intermediate motion into globally aligned 4D metric point tracks. To tightly couple 2D appearance with 4D dynamics, we establish a closed-loop, mutually enhancing cycle. During the diffusion denoising process, the generated video conditions the refinement of the 4D motion, while the aligned 4D point tracks are reprojected to guide next-step joint generation. Experimentally, our method demonstrates superior performance to state-of-the-art alternatives in visual realism, motion plausibility, and multi-view consistency.
>
---
#### [new 060] From Features to Reference Points: Lightweight and Adaptive Fusion for Cooperative Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文针对协同自动驾驶中的高通信开销问题，提出轻量级的RefPtsFusion框架。通过交换紧凑的参考点（如目标位置、速度），实现异构感知模型间的高效信息融合，显著降低带宽需求，同时保持高感知精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.18757v1](https://arxiv.org/pdf/2511.18757v1)**

> **作者:** Yongqi Zhu; Morui Zhu; Qi Chen; Deyuan Qu; Song Fu; Qing Yang
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** We present RefPtsFusion, a lightweight and interpretable framework for cooperative autonomous driving. Instead of sharing large feature maps or query embeddings, vehicles exchange compact reference points, e.g., objects' positions, velocities, and size information. This approach shifts the focus from "what is seen" to "where to see", creating a sensor- and model-independent interface that works well across vehicles with heterogeneous perception models while greatly reducing communication bandwidth. To enhance the richness of shared information, we further develop a selective Top-K query fusion that selectively adds high-confidence queries from the sender. It thus achieves a strong balance between accuracy and communication cost. Experiments on the M3CAD dataset show that RefPtsFusion maintains stable perception performance while reducing communication overhead by five orders of magnitude, dropping from hundreds of MB/s to only a few KB/s at 5 FPS (frame per second), compared to traditional feature-level fusion methods. Extensive experiments also demonstrate RefPtsFusion's strong robustness and consistent transmission behavior, highlighting its potential for scalable, real-time cooperative driving systems.
>
---
#### [new 061] DE-KAN: A Kolmogorov Arnold Network with Dual Encoder for accurate 2D Teeth Segmentation
- **分类: cs.CV**

- **简介: 该论文针对全景牙片中牙齿精确分割任务，解决因解剖变异、形状不规则和重叠结构导致的分割难题。提出DE-KAN模型，采用双编码器融合全局与局部特征，通过基于Kolmogorov-Arnold定理的可学习激活函数提升表达能力与可解释性，显著提升分割精度。**

- **链接: [https://arxiv.org/pdf/2511.18533v1](https://arxiv.org/pdf/2511.18533v1)**

> **作者:** Md Mizanur Rahman Mustakim; Jianwu Li; Sumya Bhuiyan; Mohammad Mehedi Hasan; Bing Han
>
> **摘要:** Accurate segmentation of individual teeth from panoramic radiographs remains a challenging task due to anatomical variations, irregular tooth shapes, and overlapping structures. These complexities often limit the performance of conventional deep learning models. To address this, we propose DE-KAN, a novel Dual Encoder Kolmogorov Arnold Network, which enhances feature representation and segmentation precision. The framework employs a ResNet-18 encoder for augmented inputs and a customized CNN encoder for original inputs, enabling the complementary extraction of global and local spatial features. These features are fused through KAN-based bottleneck layers, incorporating nonlinear learnable activation functions derived from the Kolmogorov Arnold representation theorem to improve learning capacity and interpretability. Extensive experiments on two benchmark dental X-ray datasets demonstrate that DE-KAN outperforms state-of-the-art segmentation models, achieving mIoU of 94.5%, Dice coefficient of 97.1%, accuracy of 98.91%, and recall of 97.36%, representing up to +4.7% improvement in Dice compared to existing methods.
>
---
#### [new 062] Hybrid Event Frame Sensors: Modeling, Calibration, and Simulation
- **分类: cs.CV**

- **简介: 该论文针对混合事件帧传感器（HES）中APS与EVS联合噪声建模难题，提出首个统一分统计噪声模型，涵盖多种噪声源并建立光照与暗电流关联。基于此，构建校准流程与真实噪声仿真器HESIM，实现高保真模拟，验证了在视频插值、去模糊等任务中从仿真到实测的强迁移能力。**

- **链接: [https://arxiv.org/pdf/2511.18037v1](https://arxiv.org/pdf/2511.18037v1)**

> **作者:** Yunfan Lu; Nico Messikommer; Xiaogang Xu; Liming Chen; Yuhan Chen; Nikola Zubic; Davide Scaramuzza; Hui Xiong
>
> **摘要:** Event frame hybrid sensors integrate an Active Pixel Sensor (APS) and an Event Vision Sensor (EVS) within a single chip, combining the high dynamic range and low latency of the EVS with the rich spatial intensity information from the APS. While this tight integration offers compact, temporally precise imaging, the complex circuit architecture introduces non-trivial noise patterns that remain poorly understood and unmodeled. In this work, we present the first unified, statistics-based imaging noise model that jointly describes the noise behavior of APS and EVS pixels. Our formulation explicitly incorporates photon shot noise, dark current noise, fixed-pattern noise, and quantization noise, and links EVS noise to illumination level and dark current. Based on this formulation, we further develop a calibration pipeline to estimate noise parameters from real data and offer a detailed analysis of both APS and EVS noise behaviors. Finally, we propose HESIM, a statistically grounded simulator that generates RAW frames and events under realistic, jointly calibrated noise statistics. Experiments on two hybrid sensors validate our model across multiple imaging tasks (e.g., video frame interpolation and deblurring), demonstrating strong transfer from simulation to real data.
>
---
#### [new 063] MetaDCSeg: Robust Medical Image Segmentation via Meta Dynamic Center Weighting
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学图像分割中噪声标注与模糊边界导致的模型不稳定问题，提出MetaDCSeg框架。通过动态中心距离机制，自适应学习像素级权重，强化对边界区域的关注，有效抑制噪声影响，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2511.18894v1](https://arxiv.org/pdf/2511.18894v1)**

> **作者:** Chenyu Mu; Guihai Chen; Xun Yang; Erkun Yang; Cheng Deng
>
> **摘要:** Medical image segmentation is crucial for clinical applications, but it is frequently disrupted by noisy annotations and ambiguous anatomical boundaries, which lead to instability in model training. Existing methods typically rely on global noise assumptions or confidence-based sample selection, which inadequately mitigate the performance degradation caused by annotation noise, especially in challenging boundary regions. To address this issue, we propose MetaDCSeg, a robust framework that dynamically learns optimal pixel-wise weights to suppress the influence of noisy ground-truth labels while preserving reliable annotations. By explicitly modeling boundary uncertainty through a Dynamic Center Distance (DCD) mechanism, our approach utilizes weighted feature distances for foreground, background, and boundary centers, directing the model's attention toward hard-to-segment pixels near ambiguous boundaries. This strategy enables more precise handling of structural boundaries, which are often overlooked by existing methods, and significantly enhances segmentation performance. Extensive experiments across four benchmark datasets with varying noise levels demonstrate that MetaDCSeg consistently outperforms existing state-of-the-art methods.
>
---
#### [new 064] UISearch: Graph-Based Embeddings for Multimodal Enterprise UI Screenshots Retrieval
- **分类: cs.CV**

- **简介: 该论文针对企业级UI截图检索任务，解决视觉与语义匹配忽略结构信息的问题。提出基于图的表示方法，将UI转为编码层级与空间关系的属性图，通过对比图自编码器学习多维度相似性嵌入，实现更精准的跨模态检索。**

- **链接: [https://arxiv.org/pdf/2511.19380v1](https://arxiv.org/pdf/2511.19380v1)**

> **作者:** Maroun Ayli; Youssef Bakouny; Tushar Sharma; Nader Jalloul; Hani Seifeddine; Rima Kilany
>
> **备注:** 12 pages, 2 figures, 3 algorithms, 4 tables
>
> **摘要:** Enterprise software companies maintain thousands of user interface screens across products and versions, creating critical challenges for design consistency, pattern discovery, and compliance check. Existing approaches rely on visual similarity or text semantics, lacking explicit modeling of structural properties fundamental to user interface (UI) composition. We present a novel graph-based representation that converts UI screenshots into attributed graphs encoding hierarchical relationships and spatial arrangements, potentially generalizable to document layouts, architectural diagrams, and other structured visual domains. A contrastive graph autoencoder learns embeddings preserving multi-level similarity across visual, structural, and semantic properties. The comprehensive analysis demonstrates that our structural embeddings achieve better discriminative power than state-of-the-art Vision Encoders, representing a fundamental advance in the expressiveness of the UI representation. We implement this representation in UISearch, a multi-modal search framework that combines structural embeddings with semantic search through a composable query language. On 20,396 financial software UIs, UISearch achieves 0.92 Top-5 accuracy with 47.5ms median latency (P95: 124ms), scaling to 20,000+ screens. The hybrid indexing architecture enables complex queries and supports fine-grained UI distinction impossible with vision-only approaches.
>
---
#### [new 065] IDEAL-M3D: Instance Diversity-Enriched Active Learning for Monocular 3D Detection
- **分类: cs.CV**

- **简介: 该论文针对单目3D目标检测中的标注成本高问题，提出IDEAL-M3D主动学习框架。解决传统方法图像级采样效率低及不确定性选择导致远距离物体偏倚的问题。通过实例级采样与多样性增强的集成学习，显著提升标注效率，在仅60%标注量下达到接近全量标注的性能。**

- **链接: [https://arxiv.org/pdf/2511.19301v1](https://arxiv.org/pdf/2511.19301v1)**

> **作者:** Johannes Meier; Florian Günther; Riccardo Marin; Oussema Dhaouadi; Jacques Kaiser; Daniel Cremers
>
> **摘要:** Monocular 3D detection relies on just a single camera and is therefore easy to deploy. Yet, achieving reliable 3D understanding from monocular images requires substantial annotation, and 3D labels are especially costly. To maximize performance under constrained labeling budgets, it is essential to prioritize annotating samples expected to deliver the largest performance gains. This prioritization is the focus of active learning. Curiously, we observed two significant limitations in active learning algorithms for 3D monocular object detection. First, previous approaches select entire images, which is inefficient, as non-informative instances contained in the same image also need to be labeled. Secondly, existing methods rely on uncertainty-based selection, which in monocular 3D object detection creates a bias toward depth ambiguity. Consequently, distant objects are selected, while nearby objects are overlooked. To address these limitations, we propose IDEAL-M3D, the first instance-level pipeline for monocular 3D detection. For the first time, we demonstrate that an explicitly diverse, fast-to-train ensemble improves diversity-driven active learning for monocular 3D. We induce diversity with heterogeneous backbones and task-agnostic features, loss weight perturbation, and time-dependent bagging. IDEAL-M3D shows superior performance and significant resource savings: with just 60% of the annotations, we achieve similar or better AP3D on KITTI validation and test set results compared to training the same detector on the whole dataset.
>
---
#### [new 066] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion
- **分类: cs.CV; physics.optics**

- **简介: 该论文针对移动平台热成像分辨率低、纹理模糊的问题，提出3M-TI框架，无需相机标定即可实现多相机跨模态热图超分辨率。通过引入交叉模态自注意力模块，融合热图与可见光信息，在扩散模型中提升图像细节与结构保真度，显著改善下游任务性能。**

- **链接: [https://arxiv.org/pdf/2511.19117v1](https://arxiv.org/pdf/2511.19117v1)**

> **作者:** Minchong Chen; Xiaoyun Yuan; Junzhe Wan; Jianing Zhang; Jun Zhang
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** The miniaturization of thermal sensors for mobile platforms inherently limits their spatial resolution and textural fidelity, leading to blurry and less informative images. Existing thermal super-resolution (SR) methods can be grouped into single-image and RGB-guided approaches: the former struggles to recover fine structures from limited information, while the latter relies on accurate and laborious cross-camera calibration, which hinders practical deployment and robustness. Here, we propose 3M-TI, a calibration-free Multi-camera cross-Modality diffusion framework for Mobile Thermal Imaging. At its core, 3M-TI integrates a cross-modal self-attention module (CSM) into the diffusion UNet, replacing the original self-attention layers to adaptively align thermal and RGB features throughout the denoising process, without requiring explicit camera calibration. This design enables the diffusion network to leverage its generative prior to enhance spatial resolution, structural fidelity, and texture detail in the super-resolved thermal images. Extensive evaluations on real-world mobile thermal cameras and public benchmarks validate our superior performance, achieving state-of-the-art results in both visual quality and quantitative metrics. More importantly, the thermal images enhanced by 3M-TI lead to substantial gains in critical downstream tasks like object detection and segmentation, underscoring its practical value for robust mobile thermal perception systems. More materials: https://github.com/work-submit/3MTI.
>
---
#### [new 067] CSD: Change Semantic Detection with only Semantic Change Masks for Damage Assessment in Conflict Zones
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出改变语义检测（CSD）任务，针对冲突区损伤评估中数据少、标注难、语义变化模糊等问题。利用DINOv3与多尺度交叉注意力孪生网络，仅基于变化区域的语义掩码进行检测，提出新方法并构建Gaza-Change数据集，有效提升小范围、模糊边界损伤的快速识别能力。**

- **链接: [https://arxiv.org/pdf/2511.19035v1](https://arxiv.org/pdf/2511.19035v1)**

> **作者:** Kai Zhenga; Zhenkai Wu; Fupeng Wei; Miaolan Zhou; Kai Lie; Haitao Guo; Lei Ding; Wei Zhang; Hang-Cheng Dong
>
> **摘要:** Accurately and swiftly assessing damage from conflicts is crucial for humanitarian aid and regional stability. In conflict zones, damaged zones often share similar architectural styles, with damage typically covering small areas and exhibiting blurred boundaries. These characteristics lead to limited data, annotation difficulties, and significant recognition challenges, including high intra-class similarity and ambiguous semantic changes. To address these issues, we introduce a pre-trained DINOv3 model and propose a multi-scale cross-attention difference siamese network (MC-DiSNet). The powerful visual representation capability of the DINOv3 backbone enables robust and rich feature extraction from bi-temporal remote sensing images. We also release a new Gaza-change dataset containing high-resolution satellite image pairs from 2023-2024 with pixel-level semantic change annotations. It is worth emphasizing that our annotations only include semantic pixels of changed areas. Unlike conventional semantic change detection (SCD), our approach eliminates the need for large-scale semantic annotations of bi-temporal images, instead focusing directly on the changed regions. We term this new task change semantic detection (CSD). The CSD task represents a direct extension of binary change detection (BCD). Due to the limited spatial extent of semantic regions, it presents greater challenges than traditional SCD tasks. We evaluated our method under the CSD framework on both the Gaza-Change and SECOND datasets. Experimental results demonstrate that our proposed approach effectively addresses the CSD task, and its outstanding performance paves the way for practical applications in rapid damage assessment across conflict zones.
>
---
#### [new 068] Generating Synthetic Human Blastocyst Images for In-Vitro Fertilization Blastocyst Grading
- **分类: cs.CV**

- **简介: 该论文针对体外受精中囊胚分级数据稀缺与类别不平衡问题，提出基于扩散模型的DIA框架，生成高保真、可控的合成囊胚图像。通过真实度评估与下游任务验证，证明合成数据可有效提升AI模型性能，显著改善数据质量与评估标准化。**

- **链接: [https://arxiv.org/pdf/2511.18204v1](https://arxiv.org/pdf/2511.18204v1)**

> **作者:** Pavan Narahari; Suraj Rajendran; Lorena Bori; Jonas E. Malmsten; Qiansheng Zhan; Zev Rosenwaks; Nikica Zaninovic; Iman Hajirasouliha
>
> **备注:** The manuscript is 23 pages, with five main figures and one table. The supplemental material includes 23 pages with fourteen figures and four tables
>
> **摘要:** The success of in vitro fertilization (IVF) at many clinics relies on the accurate morphological assessment of day 5 blastocysts, a process that is often subjective and inconsistent. While artificial intelligence can help standardize this evaluation, models require large, diverse, and balanced datasets, which are often unavailable due to data scarcity, natural class imbalance, and privacy constraints. Existing generative embryo models can mitigate these issues but face several limitations, such as poor image quality, small training datasets, non-robust evaluation, and lack of clinically relevant image generation for effective data augmentation. Here, we present the Diffusion Based Imaging Model for Artificial Blastocysts (DIA) framework, a set of latent diffusion models trained to generate high-fidelity, novel day 5 blastocyst images. Our models provide granular control by conditioning on Gardner-based morphological categories and z-axis focal depth. We rigorously evaluated the models using FID, a memorization metric, an embryologist Turing test, and three downstream classification tasks. Our results show that DIA models generate realistic images that embryologists could not reliably distinguish from real images. Most importantly, we demonstrated clear clinical value. Augmenting an imbalanced dataset with synthetic images significantly improved classification accuracy (p < 0.05). Also, adding synthetic images to an already large, balanced dataset yielded statistically significant performance gains, and synthetic data could replace up to 40% of real data in some cases without a statistically significant loss in accuracy. DIA provides a robust solution for mitigating data scarcity and class imbalance in embryo datasets. By generating novel, high-fidelity, and controllable synthetic images, our models can improve the performance, fairness, and standardization of AI embryo assessment tools.
>
---
#### [new 069] Q-Save: Towards Scoring and Attribution for Generated Video Evaluation
- **分类: cs.CV**

- **简介: 该论文针对AI生成视频质量评估难题，提出Q-Save基准数据集与统一评估模型。通过近万条带多维度标注的视频，实现高质量评分与可解释性归因。模型基于SlowFast框架，结合链式思维训练策略，提升评估准确性与可解释性，推动可信生成视频评价发展。**

- **链接: [https://arxiv.org/pdf/2511.18825v1](https://arxiv.org/pdf/2511.18825v1)**

> **作者:** Xiele Wu; Zicheng Zhang; Mingtao Chen; Yixian Liu; Yiming Liu; Shushi Wang; Zhichao Hu; Yuhong Liu; Guangtao Zhai; Xiaohong Liu
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** We present Q-Save, a new benchmark dataset and model for holistic and explainable evaluation of AI-generated video (AIGV) quality. The dataset contains near 10000 videos, each annotated with a scalar mean opinion score (MOS) and fine-grained attribution labels along three core dimensions: visual quality, dynamic quality, and text-video alignment. These multi-aspect annotations enable both accurate quality assessment and interpretable reasoning behind the scores. To leverage this data, we propose a unified evaluation model that jointly performs quality scoring and attribution-based explanation. The model adopts the SlowFast framework to distinguish between fast frames and slow frames - slow frames are processed with high resolution while fast frames use low resolution, balancing evaluation accuracy and computational efficiency. For training, we use data formatted in Chain-of-Thought (COT) style and employ a multi-stage strategy: we first conduct Supervised Fine-Tuning (SFT), then further enhance the model with Grouped Relative Policy Optimization (GRPO), and finally perform SFT again to improve model stability. Experimental results demonstrate that our model achieves state-of-the-art performance in video quality prediction while also providing human-aligned, interpretable justifications. Our dataset and model establish a strong foundation for explainable evaluation in generative video research, contributing to the development of multimodal generation and trustworthy AI. Code and dataset will be released upon publication.
>
---
#### [new 070] Granular Computing-driven SAM: From Coarse-to-Fine Guidance for Prompt-Free Segmentation
- **分类: cs.CV**

- **简介: 该论文聚焦于无提示图像分割任务，针对现有模型（如SAM）在区域定位和高分辨率细节建模上的不足，提出基于粒度计算的粗到细框架Grc-SAM。通过多粒度特征提取与自适应区域定位，实现无需外部提示的精准分割，并提升细节建模能力与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.19062v1](https://arxiv.org/pdf/2511.19062v1)**

> **作者:** Qiyang Yu; Yu Fang; Tianrui Li; Xuemei Cao; Yan Chen; Jianghao Li; Fan Min; Yi Zhang
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Prompt-free image segmentation aims to generate accurate masks without manual guidance. Typical pre-trained models, notably Segmentation Anything Model (SAM), generate prompts directly at a single granularity level. However, this approach has two limitations: (1) Localizability, lacking mechanisms for autonomous region localization; (2) Scalability, limited fine-grained modeling at high resolution. To address these challenges, we introduce Granular Computing-driven SAM (Grc-SAM), a coarse-to-fine framework motivated by Granular Computing (GrC). First, the coarse stage adaptively extracts high-response regions from features to achieve precise foreground localization and reduce reliance on external prompts. Second, the fine stage applies finer patch partitioning with sparse local swin-style attention to enhance detail modeling and enable high-resolution segmentation. Third, refined masks are encoded as latent prompt embeddings for the SAM decoder, replacing handcrafted prompts with an automated reasoning process. By integrating multi-granularity attention, Grc-SAM bridges granular computing with vision transformers. Extensive experimental results demonstrate Grc-SAM outperforms baseline methods in both accuracy and scalability. It offers a unique granular computational perspective for prompt-free segmentation.
>
---
#### [new 071] Are Large Vision Language Models Truly Grounded in Medical Images? Evidence from Italian Clinical Visual Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究大视觉语言模型在医学图像问答中的视觉接地问题。针对意大利语医疗视觉问答任务，通过替换图像为空白测试模型对视觉信息的依赖性，发现模型间差异显著，揭示其真实视觉理解能力不足，强调需严格评估后方可临床应用。**

- **链接: [https://arxiv.org/pdf/2511.19220v1](https://arxiv.org/pdf/2511.19220v1)**

> **作者:** Federico Felizzi; Olivia Riccomi; Michele Ferramola; Francesco Andrea Causio; Manuel Del Medico; Vittorio De Vita; Lorenzo De Mori; Alessandra Piscitelli Pietro Eric Risuleo; Bianca Destro Castaniti; Antonio Cristiano Alessia Longo; Luigi De Angelis; Mariapia Vassalli; Marcello Di Pumpo
>
> **备注:** Accepted at the Workshop on Multimodal Representation Learning for Healthcare (MMRL4H), EurIPS 2025
>
> **摘要:** Large vision language models (VLMs) have achieved impressive performance on medical visual question answering benchmarks, yet their reliance on visual information remains unclear. We investigate whether frontier VLMs demonstrate genuine visual grounding when answering Italian medical questions by testing four state-of-the-art models: Claude Sonnet 4.5, GPT-4o, GPT-5-mini, and Gemini 2.0 flash exp. Using 60 questions from the EuropeMedQA Italian dataset that explicitly require image interpretation, we substitute correct medical images with blank placeholders to test whether models truly integrate visual and textual information. Our results reveal striking variability in visual dependency: GPT-4o shows the strongest visual grounding with a 27.9pp accuracy drop (83.2% [74.6%, 91.7%] to 55.3% [44.1%, 66.6%]), while GPT-5-mini, Gemini, and Claude maintain high accuracy with modest drops of 8.5pp, 2.4pp, and 5.6pp respectively. Analysis of model-generated reasoning reveals confident explanations for fabricated visual interpretations across all models, suggesting varying degrees of reliance on textual shortcuts versus genuine visual analysis. These findings highlight critical differences in model robustness and the need for rigorous evaluation before clinical deployment.
>
---
#### [new 072] LAST: LeArning to Think in Space and Time for Generalist Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出LAST模型，解决通用视觉语言模型在3D空间和长视频理解上的不足。通过引入时空视觉思维轨迹，使模型在仅输入2D图像时，能联合提升空间与时间理解能力，在零样本和微调场景下均显著提升多类任务性能。**

- **链接: [https://arxiv.org/pdf/2511.19261v1](https://arxiv.org/pdf/2511.19261v1)**

> **作者:** Shuai Wang; Daoan Zhang; Tianyi Bai; Shitong Shao; Jiebo Luo; Jiaheng Wei
>
> **摘要:** Humans can perceive and understand 3D space and long videos from sequential visual observations. But do vision-language models (VLMs) can? Recent work demonstrates that even state-of-the-art VLMs still struggle to understand 3D space and long videos, although they are powerful in typical vision-language tasks. Current methods often rely on specialized architectural designs to improve performance for 3D tasks and video understanding tasks separately. In contrast, we propose LAST, short for LeArn to Think in Space and Time, to jointly improve 3D spatial and long video understanding for general VLMs with only a set of 2D images as inputs. LAST makes VLMs think in space and time rather than only with text before giving the final answer, building visual thinking trajectories in 3D space and temporal dimension. We demonstrate the effectiveness of LAST in two scenarios: 1) zero-shot, where we directly prompt proprietary models; and 2) fine-tuning general VLMs with data that include thinking trajectories in 3D space and time. We show that LAST brings substantial gains in various benchmarks, including 3 spatial understanding, 4 video understanding, and 3 image understanding tasks. Notably, 15.8% gains on EgoSchema with GPT-4o in a zero-shot manner and 8.3 gains on VSI-Bench compared with Qwen2.5-VL-7B.
>
---
#### [new 073] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLM）的计数能力，针对其因训练偏差导致的计数不准问题。构建合成基准数据集，分析图像与提示变量对计数的影响，并实施注意力干预以调节模型关注区域。实验表明，特定注意力调控可小幅提升复杂场景下的计数性能。**

- **链接: [https://arxiv.org/pdf/2511.17722v1](https://arxiv.org/pdf/2511.17722v1)**

> **作者:** Saurav Sengupta; Nazanin Moradinasab; Jiebei Liu; Donald E. Brown
>
> **摘要:** Recent research suggests that Vision Language Models (VLMs) often rely on inherent biases learned during training when responding to queries about visual properties of images. These biases are exacerbated when VLMs are asked highly specific questions that require them to focus on particular areas of the image in tasks such as counting. We build upon this research by developing a synthetic benchmark dataset and evaluation framework to systematically determine how counting performance varies as image and prompt properties change. Using open-source VLMs, we then analyze how attention allocation fluctuates with varying input parameters (e.g. number of objects in the image, objects color, background color, objects texture, background texture, and prompt specificity). We further implement attention-based interventions to modulate focus on visual tokens at different layers and evaluate their impact on counting performance across a range of visual conditions. Our experiments reveal that while VLM counting performance remains challenging, especially under high visual or linguistic complexity, certain attention interventions can lead to modest gains in counting performance.
>
---
#### [new 074] Learning What to Trust: Bayesian Prior-Guided Optimization for Visual Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉生成中奖励信号不确定的问题，提出贝叶斯先验引导优化（BPGO）。针对GRPO因文本-视觉对应模糊导致的反馈不可靠问题，BPGO引入语义先验锚点，通过组间信任分配与组内重归一化，增强可靠反馈、抑制噪声，提升生成质量与收敛速度。任务为后训练视觉生成优化。**

- **链接: [https://arxiv.org/pdf/2511.18919v1](https://arxiv.org/pdf/2511.18919v1)**

> **作者:** Ruiying Liu; Yuanzhi Liang; Haibin Huang; Tianshu Yu; Chi Zhang
>
> **摘要:** Group Relative Policy Optimization (GRPO) has emerged as an effective and lightweight framework for post-training visual generative models. However, its performance is fundamentally limited by the ambiguity of textual visual correspondence: a single prompt may validly describe diverse visual outputs, and a single image or video may support multiple equally correct interpretations. This many to many relationship leads reward models to generate uncertain and weakly discriminative signals, causing GRPO to underutilize reliable feedback and overfit noisy ones. We introduce Bayesian Prior-Guided Optimization (BPGO), a novel extension of GRPO that explicitly models reward uncertainty through a semantic prior anchor. BPGO adaptively modulates optimization trust at two levels: inter-group Bayesian trust allocation emphasizes updates from groups consistent with the prior while down-weighting ambiguous ones, and intra-group prior-anchored renormalization sharpens sample distinctions by expanding confident deviations and compressing uncertain scores. Across both image and video generation tasks, BPGO delivers consistently stronger semantic alignment, enhanced perceptual fidelity, and faster convergence than standard GRPO and recent variants.
>
---
#### [new 075] IDSplat: Instance-Decomposed 3D Gaussian Splatting for Driving Scenes
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中动态场景重建任务，解决现有方法依赖人工标注或无法分离静态与动态元素的问题。提出IDSplat框架，通过自监督方式实现实例级分解与可学习运动轨迹估计，利用语言引导的3D跟踪和协同转弯平滑，实现高质量、无标注的动态场景重建。**

- **链接: [https://arxiv.org/pdf/2511.19235v1](https://arxiv.org/pdf/2511.19235v1)**

> **作者:** Carl Lindström; Mahan Rafidashti; Maryam Fatemi; Lars Hammarstrand; Martin R. Oswald; Lennart Svensson
>
> **摘要:** Reconstructing dynamic driving scenes is essential for developing autonomous systems through sensor-realistic simulation. Although recent methods achieve high-fidelity reconstructions, they either rely on costly human annotations for object trajectories or use time-varying representations without explicit object-level decomposition, leading to intertwined static and dynamic elements that hinder scene separation. We present IDSplat, a self-supervised 3D Gaussian Splatting framework that reconstructs dynamic scenes with explicit instance decomposition and learnable motion trajectories, without requiring human annotations. Our key insight is to model dynamic objects as coherent instances undergoing rigid transformations, rather than unstructured time-varying primitives. For instance decomposition, we employ zero-shot, language-grounded video tracking anchored to 3D using lidar, and estimate consistent poses via feature correspondences. We introduce a coordinated-turn smoothing scheme to obtain temporally and physically consistent motion trajectories, mitigating pose misalignments and tracking failures, followed by joint optimization of object poses and Gaussian parameters. Experiments on the Waymo Open Dataset demonstrate that our method achieves competitive reconstruction quality while maintaining instance-level decomposition and generalizes across diverse sequences and view densities without retraining, making it practical for large-scale autonomous driving applications. Code will be released.
>
---
#### [new 076] 3D Ground Truth Reconstruction from Multi-Camera Annotations Using UKF
- **分类: cs.CV**

- **简介: 该论文属于多相机3D目标重建任务，旨在解决仅依赖2D标注难以获取精确3D信息的问题。提出基于无迹卡尔曼滤波（UKF）的多视角融合方法，将多相机2D标注转化为高精度3D位置与完整形状，有效处理遮挡，实现全自动、可扩展的3D地面真值重建。**

- **链接: [https://arxiv.org/pdf/2511.17609v1](https://arxiv.org/pdf/2511.17609v1)**

> **作者:** Linh Van Ma; Unse Fatima; Tepy Sokun Chriv; Haroon Imran; Moongu Jeon
>
> **备注:** International Conference on Control, Automation and Information Sciences (ICCAIS) 2025, October 27 - 29, 2025 | Jeju, Korea
>
> **摘要:** Accurate 3D ground truth estimation is critical for applications such as autonomous navigation, surveillance, and robotics. This paper introduces a novel method that uses an Unscented Kalman Filter (UKF) to fuse 2D bounding box or pose keypoint ground truth annotations from multiple calibrated cameras into accurate 3D ground truth. By leveraging human-annotated ground-truth 2D, our proposed method, a multi-camera single-object tracking algorithm, transforms 2D image coordinates into robust 3D world coordinates through homography-based projection and UKF-based fusion. Our proposed algorithm processes multi-view data to estimate object positions and shapes while effectively handling challenges such as occlusion. We evaluate our method on the CMC, Wildtrack, and Panoptic datasets, demonstrating high accuracy in 3D localization compared to the available 3D ground truth. Unlike existing approaches that provide only ground-plane information, our method also outputs the full 3D shape of each object. Additionally, the algorithm offers a scalable and fully automatic solution for multi-camera systems using only 2D image annotations.
>
---
#### [new 077] From Pixels to Posts: Retrieval-Augmented Fashion Captioning and Hashtag Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对时尚图像自动生成描述与标签任务，解决传统端到端模型在属性准确性和领域泛化上的不足。提出检索增强生成框架，结合多服装检测、颜色聚类与CLIP-FAISS检索，构建事实证据包，引导LLM生成精准、自然的图文内容，显著提升属性覆盖率与真实性。**

- **链接: [https://arxiv.org/pdf/2511.19149v1](https://arxiv.org/pdf/2511.19149v1)**

> **作者:** Moazzam Umer Gondal; Hamad Ul Qudous; Daniya Siddiqui; Asma Ahmad Farhan
>
> **备注:** Submitted to Expert Systems with Applications
>
> **摘要:** This paper introduces the retrieval-augmented framework for automatic fashion caption and hashtag generation, combining multi-garment detection, attribute reasoning, and Large Language Model (LLM) prompting. The system aims to produce visually grounded, descriptive, and stylistically interesting text for fashion imagery, overcoming the limitations of end-to-end captioners that have problems with attribute fidelity and domain generalization. The pipeline combines a YOLO-based detector for multi-garment localization, k-means clustering for dominant color extraction, and a CLIP-FAISS retrieval module for fabric and gender attribute inference based on a structured product index. These attributes, together with retrieved style examples, create a factual evidence pack that is used to guide an LLM to generate human-like captions and contextually rich hashtags. A fine-tuned BLIP model is used as a supervised baseline model for comparison. Experimental results show that the YOLO detector is able to obtain a mean Average Precision (mAP@0.5) of 0.71 for nine categories of garments. The RAG-LLM pipeline generates expressive attribute-aligned captions and achieves mean attribute coverage of 0.80 with full coverage at the 50% threshold in hashtag generation, whereas BLIP gives higher lexical overlap and lower generalization. The retrieval-augmented approach exhibits better factual grounding, less hallucination, and great potential for scalable deployment in various clothing domains. These results demonstrate the use of retrieval-augmented generation as an effective and interpretable paradigm for automated and visually grounded fashion content generation.
>
---
#### [new 078] SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining
- **分类: cs.CV**

- **简介: 该论文针对图像去雨任务，解决复杂多尺度雨迹与场景耦合导致的去雨困难问题。提出SD-PSFNet模型，通过三阶段序列架构，融合动态点扩散函数（PSF）机制与自适应门控融合，实现雨迹物理建模与渐进式特征优化，显著提升去雨效果。**

- **链接: [https://arxiv.org/pdf/2511.17993v1](https://arxiv.org/pdf/2511.17993v1)**

> **作者:** Jiayu Wang; Haoyu Bian; Haoran Sun; Shaoning Zeng
>
> **备注:** 12 pages, 7 figures, Published in AAAI 2026
>
> **摘要:** Image deraining is crucial for vision applications but is challenged by the complex multi-scale physics of rain and its coupling with scenes. To address this challenge, a novel approach inspired by multi-stage image restoration is proposed, incorporating Point Spread Function (PSF) mechanisms to reveal the image degradation process while combining dynamic physical modeling with sequential feature fusion transfer, named SD-PSFNet. Specifically, SD-PSFNet employs a sequential restoration architecture with three cascaded stages, allowing multiple dynamic evaluations and refinements of the degradation process estimation. The network utilizes components with learned PSF mechanisms to dynamically simulate rain streak optics, enabling effective rain-background separation while progressively enhancing outputs through novel PSF components at each stage. Additionally, SD-PSFNet incorporates adaptive gated fusion for optimal cross-stage feature integration, enabling sequential refinement from coarse rain removal to fine detail restoration. Our model achieves state-of-the-art PSNR/SSIM metrics on Rain100H (33.12dB/0.9371), RealRain-1k-L (42.28dB/0.9872), and RealRain-1k-H (41.08dB/0.9838). In summary, SD-PSFNet demonstrates excellent capability in complex scenes and dense rainfall conditions, providing a new physics-aware approach to image deraining.
>
---
#### [new 079] Are Image-to-Video Models Good Zero-Shot Image Editors?
- **分类: cs.CV**

- **简介: 该论文研究视频扩散模型作为零样本图像编辑器的潜力。针对提示错位、冗余时序隐变量和后期帧模糊问题，提出IF-Edit框架，通过提示增强、时序隐变量丢弃与后处理优化，实现指令驱动的高效图像编辑，在推理任务上表现优异。**

- **链接: [https://arxiv.org/pdf/2511.19435v1](https://arxiv.org/pdf/2511.19435v1)**

> **作者:** Zechuan Zhang; Zhenyuan Chen; Zongxin Yang; Yi Yang
>
> **备注:** technical report
>
> **摘要:** Large-scale video diffusion models show strong world simulation and temporal reasoning abilities, but their use as zero-shot image editors remains underexplored. We introduce IF-Edit, a tuning-free framework that repurposes pretrained image-to-video diffusion models for instruction-driven image editing. IF-Edit addresses three key challenges: prompt misalignment, redundant temporal latents, and blurry late-stage frames. It includes (1) a chain-of-thought prompt enhancement module that transforms static editing instructions into temporally grounded reasoning prompts; (2) a temporal latent dropout strategy that compresses frame latents after the expert-switch point, accelerating denoising while preserving semantic and temporal coherence; and (3) a self-consistent post-refinement step that sharpens late-stage frames using a short still-video trajectory. Experiments on four public benchmarks, covering non-rigid editing, physical and temporal reasoning, and general instruction edits, show that IF-Edit performs strongly on reasoning-centric tasks while remaining competitive on general-purpose edits. Our study provides a systematic view of video diffusion models as image editors and highlights a simple recipe for unified video-image generative reasoning.
>
---
#### [new 080] MASS: Motion-Aware Spatial-Temporal Grounding for Physics Reasoning and Comprehension in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型在物理推理与理解中的不足，提出MASS方法，通过时空感知的深度编码与运动追踪，增强模型对视频中物体动态和空间交互的理解。构建了MASS-Bench基准，实验证明其显著提升物理推理性能，优于多个基线模型。**

- **链接: [https://arxiv.org/pdf/2511.18373v1](https://arxiv.org/pdf/2511.18373v1)**

> **作者:** Xiyang Wu; Zongxia Li; Jihui Jin; Guangyao Shi; Gouthaman KV; Vishnu Raj; Nilotpal Sinha; Jingxi Chen; Fan Du; Dinesh Manocha
>
> **摘要:** Vision Language Models (VLMs) perform well on standard video tasks but struggle with physics-driven reasoning involving motion dynamics and spatial interactions. This limitation reduces their ability to interpret real or AI-generated content (AIGC) videos and to generate physically consistent content. We present an approach that addresses this gap by translating physical-world context cues into interpretable representations aligned with VLMs' perception, comprehension, and reasoning. We introduce MASS-Bench, a comprehensive benchmark consisting of 4,350 real-world and AIGC videos and 8,361 free-form video question-answering pairs focused on physics-related comprehension tasks, with detailed annotations including visual detections, sub-segment grounding, and full-sequence 3D motion tracking of entities. We further present MASS, a model-agnostic method that injects spatial-temporal signals into the VLM language space via depth-based 3D encoding and visual grounding, coupled with a motion tracker for object dynamics. To strengthen cross-modal alignment and reasoning, we apply reinforcement fine-tuning. Experiments and ablations show that our refined VLMs outperform comparable and larger baselines, as well as prior state-of-the-art models, by 8.7% and 6.0%, achieving performance comparable to close-source SoTA VLMs such as Gemini-2.5-Flash on physics reasoning and comprehension. These results validate the effectiveness of our approach.
>
---
#### [new 081] Stro-VIGRU: Defining the Vision Recurrent-Based Baseline Model for Brain Stroke Classification
- **分类: cs.CV**

- **简介: 该论文针对脑卒中早期诊断难题，提出Stro-VIGRU模型。基于预训练ViT提取特征，结合双向GRU进行分类，通过冻结部分编码器和数据增强解决特征学习与类别不平衡问题，实现94.06%的分类准确率，提升CT影像自动分析效率与准确性。**

- **链接: [https://arxiv.org/pdf/2511.18316v1](https://arxiv.org/pdf/2511.18316v1)**

> **作者:** Subhajeet Das; Pritam Paul; Rohit Bahadur; Sohan Das
>
> **备注:** Presented at the International Conference on Computational Intelligence and Data Communication, Accepted for publication in the Taylor and Francis Conference Proceedings
>
> **摘要:** Stroke majorly causes death and disability worldwide, and early recognition is one of the key elements of successful treatment of the same. It is common to diagnose strokes using CT scanning, which is fast and readily available, however, manual analysis may take time and may result in mistakes. In this work, a pre-trained Vision Transformer-based transfer learning framework is proposed for the early identification of brain stroke. A few of the encoder blocks of the ViT model are frozen, and the rest are allowed to be fine-tuned in order to learn brain stroke-specific features. The features that have been extracted are given as input to a single-layer Bi-GRU to perform classification. Class imbalance is handled by data augmentation. The model has achieved 94.06% accuracy in classifying brain stroke from the Stroke Dataset.
>
---
#### [new 082] EventSTU: Event-Guided Efficient Spatio-Temporal Understanding for Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文针对视频大模型推理成本高的问题，提出EventSTU框架。通过事件相机的变差触发特性，设计粗到细的关键帧采样与自适应令牌剪枝算法，在时空维度实现高效理解。引入问题相关性动态分配剪枝预算，并构建EventBench基准。实验表明，该方法显著降低计算量并提升速度，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2511.18920v1](https://arxiv.org/pdf/2511.18920v1)**

> **作者:** Wenhao Xu; Xin Dong; Yue Li; Haoyuan Shi; Zhiwei Xiong
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Video large language models have demonstrated strong video understanding capabilities but suffer from high inference costs due to the massive number of tokens in long videos. Inspired by event-based vision, we propose an event-guided, training-free framework for efficient spatio-temporal understanding, named EventSTU. In the temporal domain, we design a coarse-to-fine keyframe sampling algorithm that exploits the change-triggered property of event cameras to eliminate redundant frames. In the spatial domain, we design an adaptive token pruning algorithm that leverages the visual saliency of events as a zero-cost prior to guide spatial reduction. From a holistic spatio-temporal perspective, we further integrate question relevance from keyframe sampling to adaptively allocate token pruning budgets. To facilitate evaluation, we construct EventBench, the first event-inclusive, human-annotated multimodal benchmark that covers diverse real-world scenarios. Beyond physical event cameras, EventSTU also supports general video understanding using simulated events. Comprehensive experiments show that EventSTU achieves 3.01x FLOPs reduction and 3.10x prefilling speedup over the strongest baseline while still improving performance.
>
---
#### [new 083] SupLID: Geometrical Guidance for Out-of-Distribution Detection in Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对语义分割中的分布外（OOD）检测任务，解决传统方法依赖分类器置信度导致的过自信问题。提出SupLID框架，利用线性内在维度（LID）构建几何核心集，在超像素级别计算OOD得分，提升检测精度与空间平滑性。作为后处理方法，可无缝集成现有分割模型，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.18816v1](https://arxiv.org/pdf/2511.18816v1)**

> **作者:** Nimeshika Udayangani; Sarah Erfani; Christopher Leckie
>
> **备注:** 10 pages, CIKM 2025
>
> **摘要:** Out-of-Distribution (OOD) detection in semantic segmentation aims to localize anomalous regions at the pixel level, advancing beyond traditional image-level OOD techniques to better suit real-world applications such as autonomous driving. Recent literature has successfully explored the adaptation of commonly used image-level OOD methods--primarily based on classifier-derived confidence scores (e.g., energy or entropy)--for this pixel-precise task. However, these methods inherit a set of limitations, including vulnerability to overconfidence. In this work, we introduce SupLID, a novel framework that effectively guides classifier-derived OOD scores by exploiting the geometrical structure of the underlying semantic space, particularly using Linear Intrinsic Dimensionality (LID). While LID effectively characterizes the local structure of high-dimensional data by analyzing distance distributions, its direct application at the pixel level remains challenging. To overcome this, SupLID constructs a geometrical coreset that captures the intrinsic structure of the in-distribution (ID) subspace. It then computes OOD scores at the superpixel level, enabling both efficient real-time inference and improved spatial smoothness. We demonstrate that geometrical cues derived from SupLID serve as a complementary signal to traditional classifier confidence, enhancing the model's ability to detect diverse OOD scenarios. Designed as a post-hoc scoring method, SupLID can be seamlessly integrated with any semantic segmentation classifier at deployment time. Our results demonstrate that SupLID significantly enhances existing classifier-based OOD scores, achieving state-of-the-art performance across key evaluation metrics, including AUR, FPR, and AUP. Code is available at https://github.com/hdnugit/SupLID.
>
---
#### [new 084] V2X-RECT: An Efficient V2X Trajectory Prediction Framework via Redundant Interaction Filtering and Tracking Error Correction
- **分类: cs.CV**

- **简介: 该论文针对高密度交通下V2X轨迹预测中的目标关联不稳定、冗余交互多、推理效率低问题，提出V2X-RECT框架。通过多源身份匹配、信号灯引导的交互过滤和局部时空坐标编码，提升关联一致性与预测效率，显著改善精度与实时性。**

- **链接: [https://arxiv.org/pdf/2511.17941v1](https://arxiv.org/pdf/2511.17941v1)**

> **作者:** Xiangyan Kong; Xuecheng Wu; Xiongwei Zhao; Xiaodong Li; Yunyun Shi; Gang Wang; Dingkang Yang; Yang Liu; Hong Chen; Yulong Gao
>
> **摘要:** V2X prediction can alleviate perception incompleteness caused by limited line of sight through fusing trajectory data from infrastructure and vehicles, which is crucial to traffic safety and efficiency. However, in dense traffic scenarios, frequent identity switching of targets hinders cross-view association and fusion. Meanwhile, multi-source information tends to generate redundant interactions during the encoding stage, and traditional vehicle-centric encoding leads to large amounts of repetitive historical trajectory feature encoding, degrading real-time inference performance. To address these challenges, we propose V2X-RECT, a trajectory prediction framework designed for high-density environments. It enhances data association consistency, reduces redundant interactions, and reuses historical information to enable more efficient and accurate prediction. Specifically, we design a multi-source identity matching and correction module that leverages multi-view spatiotemporal relationships to achieve stable and consistent target association, mitigating the adverse effects of mismatches on trajectory encoding and cross-view feature fusion. Then we introduce traffic signal-guided interaction module, encoding trend of traffic light changes as features and exploiting their role in constraining spatiotemporal passage rights to accurately filter key interacting vehicles, while capturing the dynamic impact of signal changes on interaction patterns. Furthermore, a local spatiotemporal coordinate encoding enables reusable features of historical trajectories and map, supporting parallel decoding and significantly improving inference efficiency. Extensive experimental results across V2X-Seq and V2X-Traj datasets demonstrate that our V2X-RECT achieves significant improvements compared to SOTA methods, while also enhancing robustness and inference efficiency across diverse traffic densities.
>
---
#### [new 085] MambaRefine-YOLO: A Dual-Modality Small Object Detector for UAV Imagery
- **分类: cs.CV**

- **简介: 该论文针对无人机影像中小目标检测难题，提出MambaRefine-YOLO模型。通过双模态融合的DGC-MFM模块与分层特征聚合的HFAN结构，实现高效跨模态交互与多尺度特征增强，在双模态和单模态数据上均显著提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.19134v1](https://arxiv.org/pdf/2511.19134v1)**

> **作者:** Shuyu Cao; Minxin Chen; Yucheng Song; Zhaozhong Chen; Xinyou Zhang
>
> **备注:** Submitted to IEEE Geoscience and Remote Sensing Letters
>
> **摘要:** Small object detection in Unmanned Aerial Vehicle (UAV) imagery is a persistent challenge, hindered by low resolution and background clutter. While fusing RGB and infrared (IR) data offers a promising solution, existing methods often struggle with the trade-off between effective cross-modal interaction and computational efficiency. In this letter, we introduce MambaRefine-YOLO. Its core contributions are a Dual-Gated Complementary Mamba fusion module (DGC-MFM) that adaptively balances RGB and IR modalities through illumination-aware and difference-aware gating mechanisms, and a Hierarchical Feature Aggregation Neck (HFAN) that uses a ``refine-then-fuse'' strategy to enhance multi-scale features. Our comprehensive experiments validate this dual-pronged approach. On the dual-modality DroneVehicle dataset, the full model achieves a state-of-the-art mAP of 83.2%, an improvement of 7.9% over the baseline. On the single-modality VisDrone dataset, a variant using only the HFAN also shows significant gains, demonstrating its general applicability. Our work presents a superior balance between accuracy and speed, making it highly suitable for real-world UAV applications.
>
---
#### [new 086] Deep Hybrid Model for Region of Interest Detection in Omnidirectional Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对360°视频中兴趣区域（ROI）检测任务，旨在提升视频流传输效率与观看体验。通过构建深度混合显著性模型，结合预处理、模型预测与后处理，实现对每帧视频的ROI精准定位，并在360RAT数据集上验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2511.18856v1](https://arxiv.org/pdf/2511.18856v1)**

> **作者:** Sana Alamgeer
>
> **摘要:** The main goal of the project is to design a new model that predicts regions of interest in 360$^{\circ}$ videos. The region of interest (ROI) plays an important role in 360$^{\circ}$ video streaming. For example, ROIs are used to predict view-ports, intelligently cut the videos for live streaming, etc so that less bandwidth is used. Detecting view-ports in advance helps reduce the movement of the head while streaming and watching a video via the head-mounted device. Whereas, intelligent cuts of the videos help improve the efficiency of streaming the video to users and enhance the quality of their viewing experience. This report illustrates the secondary task to identify ROIs, in which, we design, train, and test a hybrid saliency model. In this work, we refer to saliency regions to represent the regions of interest. The method includes the processes as follows: preprocessing the video to obtain frames, developing a hybrid saliency model for predicting the region of interest, and finally post-processing the output predictions of the hybrid saliency model to obtain the output region of interest for each frame. Then, we compare the performance of the proposed method with the subjective annotations of the 360RAT dataset.
>
---
#### [new 087] Pillar-0: A New Frontier for Radiology Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Pillar-0，一个用于放射学的基底模型，解决医学影像分析中数据效率低、评估不真实等问题。通过大规模多模态数据预训练与RATE框架，实现高精度结构化标注，显著提升多种任务性能，拓展至长期风险预测等新场景，推动临床可落地的AI辅助诊断。**

- **链接: [https://arxiv.org/pdf/2511.17803v1](https://arxiv.org/pdf/2511.17803v1)**

> **作者:** Kumar Krishna Agrawal; Longchao Liu; Long Lian; Michael Nercessian; Natalia Harguindeguy; Yufu Wu; Peter Mikhael; Gigin Lin; Lecia V. Sequist; Florian Fintelmann; Trevor Darrell; Yutong Bai; Maggie Chung; Adam Yala
>
> **摘要:** Radiology plays an integral role in modern medicine, yet rising imaging volumes have far outpaced workforce growth. Foundation models offer a path toward assisting with the full spectrum of radiology tasks, but existing medical models remain limited: they process volumetric CT and MRI as low-fidelity 2D slices, discard critical grayscale contrast information, and lack evaluation frameworks that reflect real clinical practice. We introduce Pillar-0, a radiology foundation model pretrained on 42,990 abdomen-pelvis CTs, 86,411 chest CTs, 14,348 head CTs, and 11,543 breast MRIs from a large academic center, together with RATE, a scalable framework that extracts structured labels for 366 radiologic findings with near-perfect accuracy using LLMs. Across internal test sets of 14,230 abdomen-pelvis CTs, 10,646 chest CTs, 4,906 head CTs, and 1,585 breast MRIs, Pillar-0 establishes a new performance frontier, achieving mean AUROCs of 86.4, 88.0, 90.1, and 82.9, outperforming MedGemma (Google), MedImageInsight (Microsoft), Lingshu (Alibaba), and Merlin (Stanford) by 7.8-15.8 AUROC points and ranking best in 87.2\% (319/366) tasks. Pillar-0 similarly outperforms all baselines in an external validation on the Stanford Abdominal CT dataset, including Merlin (82.2 vs 80.6 AUROC). Pillar-0 extends to tasks beyond its pretraining, such as long-horizon lung cancer risk prediction, where it improves upon the state-of-the-art Sybil by 3.0 C-index points on NLST, and generalizes with gains of 5.9 (MGH) and 1.9 (CGMH). In brain hemorrhage detection, Pillar-0 obtained a >95 AUROC when using only 1/20th of the data of the next most sample efficient baseline. Pillar-0 and RATE together provide an open, clinically rigorous foundation for building high-performance radiology systems, enabling applications that were previously infeasible due to computational, data, and evaluation constraints.
>
---
#### [new 088] Point-to-Point: Sparse Motion Guidance for Controllable Video Editing
- **分类: cs.CV**

- **简介: 该论文针对视频编辑中保持运动真实性与编辑可控性之间的矛盾，提出基于锚点令牌（anchor tokens）的稀疏运动引导方法。通过利用视频扩散模型先验，自动提取关键点轨迹，实现高效、可迁移的运动表示，显著提升编辑质量与运动保真度。**

- **链接: [https://arxiv.org/pdf/2511.18277v1](https://arxiv.org/pdf/2511.18277v1)**

> **作者:** Yeji Song; Jaehyun Lee; Mijin Koo; JunHoo Lee; Nojun Kwak
>
> **摘要:** Accurately preserving motion while editing a subject remains a core challenge in video editing tasks. Existing methods often face a trade-off between edit and motion fidelity, as they rely on motion representations that are either overfitted to the layout or only implicitly defined. To overcome this limitation, we revisit point-based motion representation. However, identifying meaningful points remains challenging without human input, especially across diverse video scenarios. To address this, we propose a novel motion representation, anchor tokens, that capture the most essential motion patterns by leveraging the rich prior of a video diffusion model. Anchor tokens encode video dynamics compactly through a small number of informative point trajectories and can be flexibly relocated to align with new subjects. This allows our method, Point-to-Point, to generalize across diverse scenarios. Extensive experiments demonstrate that anchor tokens lead to more controllable and semantically aligned video edits, achieving superior performance in terms of edit and motion fidelity.
>
---
#### [new 089] BackdoorVLM: A Benchmark for Backdoor Attacks on Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出BackdoorVLM，首个针对视觉语言模型（VLMs）的后门攻击基准。针对多模态模型在后门攻击下脆弱性研究不足的问题，系统构建了5类威胁、12种攻击方法，评估了文本、图像及双模态触发器的影响，揭示了文本模态的强影响力与极低中毒率下的高攻击成功率，凸显了当前VLMs的关键安全漏洞。**

- **链接: [https://arxiv.org/pdf/2511.18921v1](https://arxiv.org/pdf/2511.18921v1)**

> **作者:** Juncheng Li; Yige Li; Hanxun Huang; Yunhao Chen; Xin Wang; Yixu Wang; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** Backdoor attacks undermine the reliability and trustworthiness of machine learning systems by injecting hidden behaviors that can be maliciously activated at inference time. While such threats have been extensively studied in unimodal settings, their impact on multimodal foundation models, particularly vision-language models (VLMs), remains largely underexplored. In this work, we introduce \textbf{BackdoorVLM}, the first comprehensive benchmark for systematically evaluating backdoor attacks on VLMs across a broad range of settings. It adopts a unified perspective that injects and analyzes backdoors across core vision-language tasks, including image captioning and visual question answering. BackdoorVLM organizes multimodal backdoor threats into 5 representative categories: targeted refusal, malicious injection, jailbreak, concept substitution, and perceptual hijack. Each category captures a distinct pathway through which an adversary can manipulate a model's behavior. We evaluate these threats using 12 representative attack methods spanning text, image, and bimodal triggers, tested on 2 open-source VLMs and 3 multimodal datasets. Our analysis reveals that VLMs exhibit strong sensitivity to textual instructions, and in bimodal backdoors the text trigger typically overwhelms the image trigger when forming the backdoor mapping. Notably, backdoors involving the textual modality remain highly potent, with poisoning rates as low as 1\% yielding over 90\% success across most tasks. These findings highlight significant, previously underexplored vulnerabilities in current VLMs. We hope that BackdoorVLM can serve as a useful benchmark for analyzing and mitigating multimodal backdoor threats. Code is available at: https://github.com/bin015/BackdoorVLM .
>
---
#### [new 090] ReEXplore: Improving MLLMs for Embodied Exploration with Contextualized Retrospective Experience Replay
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLM）在具身探索任务中的性能瓶颈，提出ReEXplore框架。针对模型依赖过时知识、训练成本高及决策空间复杂等问题，采用推理时回溯经验重放与分层前缘选择，实现高效、可追溯的探索。显著提升成功率与导航效率，优于现有基线。**

- **链接: [https://arxiv.org/pdf/2511.19033v1](https://arxiv.org/pdf/2511.19033v1)**

> **作者:** Gengyuan Zhang; Mingcong Ding; Jingpei Wu; Ruotong Liao; Volker Tresp
>
> **备注:** 8 main pages plus 13 pages Appendix
>
> **摘要:** Embodied exploration is a target-driven process that requires embodied agents to possess fine-grained perception and knowledge-enhanced decision making. While recent attempts leverage MLLMs for exploration due to their strong perceptual and reasoning abilities, we find that MLLM-based embodied agents remain suboptimal in exploring new environments: (i) they rely on profound but stale pre-trained knowledge, (ii) training-based approaches such as imitation learning or reinforcement learning are expensive for long-horizon tasks with sparse outcome rewards, and (iii) frontier-based exploration yields a large, visually nuanced action space that is difficult for MLLMs to make reliable decisions. We address these challenges with ReEXplore, a training-free framework that performs retrospective experience replay to inject distilled, abstract experience at inference time, and hierarchical frontier selection to decompose frontier ranking into coarse-to-fine decisions. Our approach enables robust, traceable, and efficient exploration. Across multiple embodied exploration benchmarks, ReEXplore yields great improvements over strong MLLM baselines, up to 3x higher performance in both success rate and in navigation efficiency under open-source backbones.
>
---
#### [new 091] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对扩散模型在少样本图像生成中速度与质量难以兼顾的问题，提出Uni-DAD统一蒸馏与适配的单阶段方法。通过双域分布匹配与多头GAN损失，实现快速、高质量的跨域生成，显著提升少步采样下的图像质量和多样性。**

- **链接: [https://arxiv.org/pdf/2511.18281v1](https://arxiv.org/pdf/2511.18281v1)**

> **作者:** Yara Bahram; Melodie Desbos; Mohammadhadi Shateri; Eric Granger
>
> **备注:** Under review paper at CVPR 2026
>
> **摘要:** Diffusion models (DMs) produce high-quality images, yet their sampling remains costly when adapted to new domains. Distilled DMs are faster but typically remain confined within their teacher's domain. Thus, fast and high-quality generation for novel domains relies on two-stage training pipelines: Adapt-then-Distill or Distill-then-Adapt. However, both add design complexity and suffer from degraded quality or diversity. We introduce Uni-DAD, a single-stage pipeline that unifies distillation and adaptation of DMs. It couples two signals during training: (i) a dual-domain distribution-matching distillation objective that guides the student toward the distributions of the source teacher and a target teacher, and (ii) a multi-head generative adversarial network (GAN) loss that encourages target realism across multiple feature scales. The source domain distillation preserves diverse source knowledge, while the multi-head GAN stabilizes training and reduces overfitting, especially in few-shot regimes. The inclusion of a target teacher facilitates adaptation to more structurally distant domains. We perform evaluations on a variety of datasets for few-shot image generation (FSIG) and subject-driven personalization (SDP). Uni-DAD delivers higher quality than state-of-the-art (SoTA) adaptation methods even with less than 4 sampling steps, and outperforms two-stage training pipelines in both quality and diversity.
>
---
#### [new 092] EgoControl: Controllable Egocentric Video Generation via 3D Full-Body Poses
- **分类: cs.CV**

- **简介: 该论文提出EgoControl，一种基于3D全身姿态的可控第一人称视频生成方法。针对现有技术难以精细控制动作的问题，引入新型姿态表示与控制机制，实现从观测帧和目标姿态序列生成连贯、逼真的未来视频，推动可控制的具身视频模拟与理解。**

- **链接: [https://arxiv.org/pdf/2511.18173v1](https://arxiv.org/pdf/2511.18173v1)**

> **作者:** Enrico Pallotta; Sina Mokhtarzadeh Azar; Lars Doorenbos; Serdar Ozsoy; Umar Iqbal; Juergen Gall
>
> **摘要:** Egocentric video generation with fine-grained control through body motion is a key requirement towards embodied AI agents that can simulate, predict, and plan actions. In this work, we propose EgoControl, a pose-controllable video diffusion model trained on egocentric data. We train a video prediction model to condition future frame generation on explicit 3D body pose sequences. To achieve precise motion control, we introduce a novel pose representation that captures both global camera dynamics and articulated body movements, and integrate it through a dedicated control mechanism within the diffusion process. Given a short sequence of observed frames and a sequence of target poses, EgoControl generates temporally coherent and visually realistic future frames that align with the provided pose control. Experimental results demonstrate that EgoControl produces high-quality, pose-consistent egocentric videos, paving the way toward controllable embodied video simulation and understanding.
>
---
#### [new 093] Decoupled Audio-Visual Dataset Distillation
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文研究音频-视觉数据集压缩任务，旨在提升跨模态对齐与私有信息保留。针对传统方法模态映射不一致及跨模态交互破坏私有特征的问题，提出DAVDD框架，通过预训练特征解耦与共现-分布联合对齐，实现高质量跨模态数据蒸馏，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17890v1](https://arxiv.org/pdf/2511.17890v1)**

> **作者:** Wenyuan Li; Guang Li; Keisuke Maeda; Takahiro Ogawa; Miki Haseyama
>
> **摘要:** Audio-Visual Dataset Distillation aims to compress large-scale datasets into compact subsets while preserving the performance of the original data. However, conventional Distribution Matching (DM) methods struggle to capture intrinsic cross-modal alignment. Subsequent studies have attempted to introduce cross-modal matching, but two major challenges remain: (i) independently and randomly initialized encoders lead to inconsistent modality mapping spaces, increasing training difficulty; and (ii) direct interactions between modalities tend to damage modality-specific (private) information, thereby degrading the quality of the distilled data. To address these challenges, we propose DAVDD, a pretraining-based decoupled audio-visual distillation framework. DAVDD leverages a diverse pretrained bank to obtain stable modality features and uses a lightweight decoupler bank to disentangle them into common and private representations. To effectively preserve cross-modal structure, we further introduce Common Intermodal Matching together with a Sample-Distribution Joint Alignment strategy, ensuring that shared representations are aligned both at the sample level and the global distribution level. Meanwhile, private representations are entirely isolated from cross-modal interaction, safeguarding modality-specific cues throughout distillation. Extensive experiments across multiple benchmarks show that DAVDD achieves state-of-the-art results under all IPC settings, demonstrating the effectiveness of decoupled representation learning for high-quality audio-visual dataset distillation. Code will be released.
>
---
#### [new 094] SpectraNet: FFT-assisted Deep Learning Classifier for Deepfake Face Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对深度伪造人脸检测任务，解决数据不平衡导致模型性能下降的问题。提出基于EfficientNet-B6的轻量级分类框架，结合预处理、过采样与优化策略，提升准确率与泛化能力，使非专家也能有效识别深伪图像。**

- **链接: [https://arxiv.org/pdf/2511.19187v1](https://arxiv.org/pdf/2511.19187v1)**

> **作者:** Nithira Jayarathne; Naveen Basnayake; Keshawa Jayasundara; Pasindu Dodampegama; Praveen Wijesinghe; Hirushika Pelagewatta; Kavishka Abeywardana; Sandushan Ranaweera; Chamira Edussooriya
>
> **备注:** 4 pages, 3 figures
>
> **摘要:** Detecting deepfake images is crucial in combating misinformation. We present a lightweight, generalizable binary classification model based on EfficientNet-B6, fine-tuned with transformation techniques to address severe class imbalances. By leveraging robust preprocessing, oversampling, and optimization strategies, our model achieves high accuracy, stability, and generalization. While incorporating Fourier transform-based phase and amplitude features showed minimal impact, our proposed framework helps non-experts to effectively identify deepfake images, making significant strides toward accessible and reliable deepfake detection.
>
---
#### [new 095] CataractCompDetect: Intraoperative Complication Detection in Cataract Surgery
- **分类: cs.CV**

- **简介: 该论文针对白内障手术中术中并发症（如虹膜脱出、后囊破裂、玻璃体丢失）检测难题，提出CataractCompDetect框架，融合相位感知定位、SAM 2跟踪、风险评分与视觉语言推理，实现精准识别。构建首个标注并发症的手术视频数据集CataComp，验证了方法的有效性，推动智能手术辅助发展。**

- **链接: [https://arxiv.org/pdf/2511.18968v1](https://arxiv.org/pdf/2511.18968v1)**

> **作者:** Bhuvan Sachdeva; Sneha Kumari; Rudransh Agarwal; Shalaka Kumaraswamy; Niharika Singri Prasad; Simon Mueller; Raphael Lechtenboehmer; Maximilian W. M. Wintergerst; Thomas Schultz; Kaushik Murali; Mohit Jain
>
> **摘要:** Cataract surgery is one of the most commonly performed surgeries worldwide, yet intraoperative complications such as iris prolapse, posterior capsule rupture (PCR), and vitreous loss remain major causes of adverse outcomes. Automated detection of such events could enable early warning systems and objective training feedback. In this work, we propose CataractCompDetect, a complication detection framework that combines phase-aware localization, SAM 2-based tracking, complication-specific risk scoring, and vision-language reasoning for final classification. To validate CataractCompDetect, we curate CataComp, the first cataract surgery video dataset annotated for intraoperative complications, comprising 53 surgeries, including 23 with clinical complications. On CataComp, CataractCompDetect achieves an average F1 score of 70.63%, with per-complication performance of 81.8% (Iris Prolapse), 60.87% (PCR), and 69.23% (Vitreous Loss). These results highlight the value of combining structured surgical priors with vision-language reasoning for recognizing rare but high-impact intraoperative events. Our dataset and code will be publicly released upon acceptance.
>
---
#### [new 096] MambaX: Image Super-Resolution with State Predictive Control
- **分类: cs.CV**

- **简介: 该论文针对图像超分辨率任务，解决现有方法忽视中间阶段误差传播的问题。提出MambaX模型，通过动态非线性状态预测控制、跨模态融合与渐进式学习，提升重建精度与泛化能力，尤其在多模态和细粒度图像中表现优异。**

- **链接: [https://arxiv.org/pdf/2511.18028v1](https://arxiv.org/pdf/2511.18028v1)**

> **作者:** Chenyu Li; Danfeng Hong; Bing Zhang; Zhaojie Pan; Naoto Yokoya; Jocelyn Chanussot
>
> **摘要:** Image super-resolution (SR) is a critical technology for overcoming the inherent hardware limitations of sensors. However, existing approaches mainly focus on directly enhancing the final resolution, often neglecting effective control over error propagation and accumulation during intermediate stages. Recently, Mamba has emerged as a promising approach that can represent the entire reconstruction process as a state sequence with multiple nodes, allowing for intermediate intervention. Nonetheless, its fixed linear mapper is limited by a narrow receptive field and restricted flexibility, which hampers its effectiveness in fine-grained images. To address this, we created a nonlinear state predictive control model \textbf{MambaX} that maps consecutive spectral bands into a latent state space and generalizes the SR task by dynamically learning the nonlinear state parameters of control equations. Compared to existing sequence models, MambaX 1) employs dynamic state predictive control learning to approximate the nonlinear differential coefficients of state-space models; 2) introduces a novel state cross-control paradigm for multimodal SR fusion; and 3) utilizes progressive transitional learning to mitigate heterogeneity caused by domain and modality shifts. Our evaluation demonstrates the superior performance of the dynamic spectrum-state representation model in both single-image SR and multimodal fusion-based SR tasks, highlighting its substantial potential to advance spectrally generalized modeling across arbitrary dimensions and modalities.
>
---
#### [new 097] Robustness of Structured Data Extraction from Perspectively Distorted Documents
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究多模态大模型在透视扭曲文档上的结构化数据提取鲁棒性问题。针对真实文档常含旋转与透视扭曲，提出基于等腰梯形变换的简化参数化方法，将复杂畸变降维至旋转角与畸变比两个参数，并评估Gemini-1.5-pro在合成数据上的字符与结构识别准确率。发现结构识别性能受扭曲显著影响，且可通过简单旋转校正改善。**

- **链接: [https://arxiv.org/pdf/2511.17607v1](https://arxiv.org/pdf/2511.17607v1)**

> **作者:** Hyakka Nakada; Yoshiyasu Tanaka
>
> **备注:** 8 pages, 12 figures
>
> **摘要:** Optical Character Recognition (OCR) for data extraction from documents is essential to intelligent informatics, such as digitizing medical records and recognizing road signs. Multi-modal Large Language Models (LLMs) can solve this task and have shown remarkable performance. Recently, it has been noticed that the accuracy of data extraction by multi-modal LLMs can be affected when in-plane rotations are present in the documents. However, real-world document images are usually not only in-plane rotated but also perspectively distorted. This study investigates the impacts of such perturbations on the data extraction accuracy for the state-of-the-art model, Gemini-1.5-pro. Because perspective distortions have a high degree of freedom, designing experiments in the same manner as single-parametric rotations is difficult. We observed typical distortions of document images and showed that most of them approximately follow an isosceles-trapezoidal transformation, which allows us to evaluate distortions with a small number of parameters. We were able to reduce the number of independent parameters from eight to two, i.e. rotation angle and distortion ratio. Then, specific entities were extracted from synthetically generated sample documents with varying these parameters. As the performance of LLMs, we evaluated not only a character-recognition accuracy but also a structure-recognition accuracy. Whereas the former represents the classical indicators for optical character recognition, the latter is related to the correctness of reading order. In particular, the structure-recognition accuracy was found to be significantly degraded by document distortion. In addition, we found that this accuracy can be improved by a simple rotational correction. This insight will contribute to the practical use of multi-modal LLMs for OCR tasks.
>
---
#### [new 098] Facade Segmentation for Solar Photovoltaic Suitability
- **分类: cs.CV**

- **简介: 该论文针对城市建筑立面光伏（BIPV）规划中自动化识别与潜力评估不足的问题，提出一种融合建筑立面结构信息的分割流水线。基于SegFormer-B5在CMP Facades数据集上微调，生成考虑组件尺寸与间距的光伏适配掩码与布局，实现立面光伏适用性精准评估，为城市级可再生能源规划提供可靠支持。**

- **链接: [https://arxiv.org/pdf/2511.18882v1](https://arxiv.org/pdf/2511.18882v1)**

> **作者:** Ayca Duran; Christoph Waibel; Bernd Bickel; Iro Armeni; Arno Schlueter
>
> **备注:** NeurIPS 2025 Tackling Climate Change with Machine Learning Workshop version. Non-archival
>
> **摘要:** Building integrated photovoltaic (BIPV) facades represent a promising pathway towards urban decarbonization, especially where roof areas are insufficient and ground-mounted arrays are infeasible. Although machine learning-based approaches to support photovoltaic (PV) planning on rooftops are well researched, automated approaches for facades still remain scarce and oversimplified. This paper therefore presents a pipeline that integrates detailed information on the architectural composition of the facade to automatically identify suitable surfaces for PV application and estimate the solar energy potential. The pipeline fine-tunes SegFormer-B5 on the CMP Facades dataset and converts semantic predictions into facade-level PV suitability masks and PV panel layouts considering module sizes and clearances. Applied to a dataset of 373 facades with known dimensions from ten cities, the results show that installable BIPV potential is significantly lower than theoretical potential, thus providing valuable insights for reliable urban energy planning. With the growing availability of facade imagery, the proposed pipeline can be scaled to support BIPV planning in cities worldwide.
>
---
#### [new 099] Using MLIR Transform to Design Sliced Convolution Algorithm
- **分类: cs.CV; cs.LG; cs.PF**

- **简介: 该论文针对深度学习中2D卷积优化问题，提出SConvTransform，基于MLIR Transform方言实现可重用的分块与数据打包变换。通过静态分析确定最优分块策略，生成高效目标代码，在ARM和Intel架构上分别达60%和67%峰值性能，验证了结构化变换的有效性。**

- **链接: [https://arxiv.org/pdf/2511.18222v1](https://arxiv.org/pdf/2511.18222v1)**

> **作者:** Victor Ferrari; Marcio Pereira; Lucas Alvarenga; Gustavo Leite; Guido Araujo
>
> **摘要:** This paper proposes SConvTransform, a Transform dialect extension that provides operations for optimizing 2D convolutions in MLIR. Its main operation, SConvOp, lowers Linalg convolutions into tiled and packed generic operations through a fully declarative transformation pipeline. The process is guided by a Convolution Slicing Analysis that determines tile sizes and data layout strategies based on input and filter shapes, as well as target architecture parameters. SConvOp handles edge cases by splitting irregular regions and adjusting affine maps where needed. All packing and tiling operations are derived from a parametric set of affine equations, enabling reusable and analyzable transformations. Although functional correctness was the primary goal of this work, the experimental evaluation demonstrates the effectiveness of SConvTransform, achieving good enough performance across different target architectures. Future work will focus on optimizing performance and porting to other target devices. When applied to standard convolution configurations, the generated code achieves up to 60% of peak performance on ARM SME and 67% on Intel AVX512. These results validate the benefit of combining static shape analysis with structured tiling and packing strategies within the MLIR Transform dialect. Furthermore, the modular design of SConvTransform facilitates integration with future extensions, enabling continued optimization of convolution workloads through MLIR's extensible compilation infrastructure.
>
---
#### [new 100] ABM-LoRA: Activation Boundary Matching for Fast Convergence in Low-Rank Adaptation
- **分类: cs.CV**

- **简介: 该论文针对低秩适配（LoRA）因随机初始化导致梯度更新偏离最优空间、收敛慢的问题，提出激活边界匹配（ABM-LoRA）初始化方法。通过对齐预训练模型与适配器的激活边界，提升梯度投影效率，减少信息损失，加速收敛。在语言理解、对话生成和视觉识别任务中均取得显著性能提升。**

- **链接: [https://arxiv.org/pdf/2511.19145v1](https://arxiv.org/pdf/2511.19145v1)**

> **作者:** Dongha Lee; Jinhee Park; Minjun Kim; Junseok Kwon
>
> **备注:** 16 pages, 5 figures, under review
>
> **摘要:** We propose Activation Boundary Matching for Low-Rank Adaptation (ABM-LoRA), a principled initialization strategy that substantially accelerates the convergence of low-rank adapters. While LoRA offers high parameter efficiency, its random initialization restricts gradient updates to a mismatched tangent space, causing significant information loss and hindering early convergence. Our ABM-LoRA addresses this by aligning the adapter's activation boundaries with those of the pretrained model before downstream training, thereby maximizing the projection of full-parameter gradients into the adapter subspace. This alignment sharply reduces information loss at initialization, yields a lower starting loss, and accelerates convergence. We demonstrate ABM-LoRA's effectiveness across diverse architectures and tasks: language understanding (T5-Base on GLUE), dialogue generation (LLaMA2-7B on WizardLM), and vision recognition (ViT-B/16 on VTAB-1K). On VTAB-1K, it achieves the highest accuracy among all methods, with strong gains on structured reasoning tasks requiring geometric understanding.
>
---
#### [new 101] ConsistCompose: Unified Multimodal Layout Control for Image Composition
- **分类: cs.CV**

- **简介: 该论文提出ConsistCompose框架，解决多实例图像生成中布局控制不足的问题。通过将布局坐标嵌入语言提示，实现统一的布局可控生成。构建了3.4M多实例数据集，支持文本与图像引导的布局生成，提升空间精度与身份一致性，推动统一的多模态图像生成范式。**

- **链接: [https://arxiv.org/pdf/2511.18333v1](https://arxiv.org/pdf/2511.18333v1)**

> **作者:** Xuanke Shi; Boxuan Li; Xiaoyang Han; Zhongang Cai; Lei Yang; Dahua Lin; Quan Wang
>
> **备注:** 22 pages, 17 figures
>
> **摘要:** Unified multimodal models that couple visual understanding with image generation have advanced rapidly, yet most systems still focus on visual grounding-aligning language with image regions-while their generative counterpart, linguistic-embedded layout-grounded generation (LELG) for layout-controllable multi-instance generation, remains underexplored and limits precise compositional control. We present ConsistCompose, a unified multimodal framework that embeds layout coordinates directly into language prompts, enabling layout-controlled multi-instance image generation from Interleaved Image-Text within a single generative interface. We further construct ConsistCompose3M, a 3.4M multi-instance generation dataset with layout and identity annotations (2.6M text-guided and 0.8M image-guided data pairs) that provides large-scale supervision for layout-conditioned generation. Within this framework, LELG is instantiated through instance-coordinate binding prompts and coordinate-aware classifier-free guidance, which translate linguistic layout cues into precise spatial control without task-specific branches. Experiments on COCO-Position and MS-Bench show that ConsistCompose substantially improves spatial accuracy over layout-controlled baselines while preserving identity fidelity and competitive general multimodal understanding, establishing a unified paradigm for layout-controllable multimodal image generation.
>
---
#### [new 102] Person Recognition in Aerial Surveillance: A Decade Survey
- **分类: cs.CV**

- **简介: 该论文聚焦于无人机等空中平台上的人体识别任务，针对高空视角下目标小、姿态多变等挑战，系统梳理近十年150余篇文献，分析数据集与技术方法，揭示现有方案应对空中特有难题的策略，并指出研究空白与未来方向。**

- **链接: [https://arxiv.org/pdf/2511.17674v1](https://arxiv.org/pdf/2511.17674v1)**

> **作者:** Kien Nguyen; Feng Liu; Clinton Fookes; Sridha Sridharan; Xiaoming Liu; Arun Ross
>
> **备注:** Accepted at T-BIOM
>
> **摘要:** The rapid emergence of airborne platforms and imaging sensors is enabling new forms of aerial surveillance due to their unprecedented advantages in scale, mobility, deployment, and covert observation capabilities. This paper provides a comprehensive overview of 150+ papers over the last 10 years of human-centric aerial surveillance tasks from a computer vision and machine learning perspective. It aims to provide readers with an in-depth systematic review and technical analysis of the current state of aerial surveillance tasks using drones, UAVs, and other airborne platforms. The object of interest is humans, where human subjects are to be detected, identified, and re-identified. More specifically, for each of these tasks, we first identify unique challenges in performing these tasks in an aerial setting compared to the popular ground-based setting and subsequently compile and analyze aerial datasets publicly available for each task. Most importantly, we delve deep into the approaches in the aerial surveillance literature with a focus on investigating how they presently address aerial challenges and techniques for improvement. We conclude the paper by discussing the gaps and open research questions to inform future research avenues.
>
---
#### [new 103] AuViRe: Audio-visual Speech Representation Reconstruction for Deepfake Temporal Localization
- **分类: cs.CV**

- **简介: 该论文针对深度伪造视频的时序定位任务，提出AuViRe方法。通过跨模态语音表示重建（从音频重建唇动或反之），利用伪造片段中重建误差增大这一特性，实现精准定位。相比现有方法，在多个数据集上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.18993v1](https://arxiv.org/pdf/2511.18993v1)**

> **作者:** Christos Koutlis; Symeon Papadopoulos
>
> **备注:** WACV 2026
>
> **摘要:** With the rapid advancement of sophisticated synthetic audio-visual content, e.g., for subtle malicious manipulations, ensuring the integrity of digital media has become paramount. This work presents a novel approach to temporal localization of deepfakes by leveraging Audio-Visual Speech Representation Reconstruction (AuViRe). Specifically, our approach reconstructs speech representations from one modality (e.g., lip movements) based on the other (e.g., audio waveform). Cross-modal reconstruction is significantly more challenging in manipulated video segments, leading to amplified discrepancies, thereby providing robust discriminative cues for precise temporal forgery localization. AuViRe outperforms the state of the art by +8.9 AP@0.95 on LAV-DF, +9.6 AP@0.5 on AV-Deepfake1M, and +5.1 AUC on an in-the-wild experiment. Code available at https://github.com/mever-team/auvire.
>
---
#### [new 104] VideoCompressa: Data-Efficient Video Understanding via Joint Temporal Compression and Spatial Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对视频理解中数据规模大导致的存储与计算成本高的问题，提出VideoCompressa框架。通过联合优化关键帧选择与潜空间压缩，实现高效视频数据合成，显著提升数据效率，仅用极少量原始数据即达到或超越全量数据训练效果。**

- **链接: [https://arxiv.org/pdf/2511.18831v1](https://arxiv.org/pdf/2511.18831v1)**

> **作者:** Shaobo Wang; Tianle Niu; Runkang Yang; Deshan Liu; Xu He; Zichen Wen; Conghui He; Xuming Hu; Linfeng Zhang
>
> **备注:** 15 pages, 6 tables, 8 figures
>
> **摘要:** The scalability of video understanding models is increasingly limited by the prohibitive storage and computational costs of large-scale video datasets. While data synthesis has improved data efficiency in the image domain, its extension to video remains challenging due to pervasive temporal redundancy and complex spatiotemporal dynamics. In this work, we uncover a critical insight: the primary source of inefficiency in video datasets is not inter-sample redundancy, but intra-sample frame-level redundancy. To leverage this insight, we introduce VideoCompressa, a novel framework for video data synthesis that reframes the problem as dynamic latent compression. Specifically, VideoCompressa jointly optimizes a differentiable keyframe selector-implemented as a lightweight ConvNet with Gumbel-Softmax sampling-to identify the most informative frames, and a pretrained, frozen Variational Autoencoder (VAE) to compress these frames into compact, semantically rich latent codes. These latent representations are then fed into a compression network, enabling end-to-end backpropagation. Crucially, the keyframe selector and synthetic latent codes are co-optimized to maximize retention of task-relevant information. Experiments show that our method achieves unprecedented data efficiency: on UCF101 with ConvNets, VideoCompressa surpasses full-data training by 2.34\% points using only 0.13\% of the original data, with over 5800x speedup compared to traditional synthesis method. Moreover, when fine-tuning Qwen2.5-7B-VL on HMDB51, VideoCompressa matches full-data performance using just 0.41\% of the training data-outperforming zero-shot baseline by 10.61\%.
>
---
#### [new 105] Ref-SAM3D: Bridging SAM3D with Text for Reference 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对SAM3D无法根据文本描述重建特定3D物体的问题，提出Ref-SAM3D，通过引入文本作为高层先验，实现仅需单张RGB图像和自然语言描述的文本引导3D重建，提升了3D重建的灵活性与实用性。**

- **链接: [https://arxiv.org/pdf/2511.19426v1](https://arxiv.org/pdf/2511.19426v1)**

> **作者:** Yun Zhou; Yaoting Wang; Guangquan Jie; Jinyu Liu; Henghui Ding
>
> **备注:** Code: https://github.com/FudanCVL/Ref-SAM3D
>
> **摘要:** SAM3D has garnered widespread attention for its strong 3D object reconstruction capabilities. However, a key limitation remains: SAM3D cannot reconstruct specific objects referred to by textual descriptions, a capability that is essential for practical applications such as 3D editing, game development, and virtual environments. To address this gap, we introduce Ref-SAM3D, a simple yet effective extension to SAM3D that incorporates textual descriptions as a high-level prior, enabling text-guided 3D reconstruction from a single RGB image. Through extensive qualitative experiments, we show that Ref-SAM3D, guided only by natural language and a single 2D view, delivers competitive and high-fidelity zero-shot reconstruction performance. Our results demonstrate that Ref-SAM3D effectively bridges the gap between 2D visual cues and 3D geometric understanding, offering a more flexible and accessible paradigm for reference-guided 3D reconstruction. Code is available at: https://github.com/FudanCVL/Ref-SAM3D.
>
---
#### [new 106] NVGS: Neural Visibility for Occlusion Culling in 3D Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对3D高斯泼溅中因高斯分布半透明性导致无法使用遮挡剔除的问题，提出NVGS方法。通过共享的小型MLP学习视角相关的可见性函数，在渲染前预判并剔除被遮挡的高斯点，结合实例化软件光栅化与张量核心加速，有效降低显存占用并提升图像质量，实现了高效遮挡剔除。**

- **链接: [https://arxiv.org/pdf/2511.19202v1](https://arxiv.org/pdf/2511.19202v1)**

> **作者:** Brent Zoomers; Florian Hahlbohm; Joni Vanherck; Lode Jorissen; Marcus Magnor; Nick Michiels
>
> **备注:** 15 pages, 13 figures
>
> **摘要:** 3D Gaussian Splatting can exploit frustum culling and level-of-detail strategies to accelerate rendering of scenes containing a large number of primitives. However, the semi-transparent nature of Gaussians prevents the application of another highly effective technique: occlusion culling. We address this limitation by proposing a novel method to learn the viewpoint-dependent visibility function of all Gaussians in a trained model using a small, shared MLP across instances of an asset in a scene. By querying it for Gaussians within the viewing frustum prior to rasterization, our method can discard occluded primitives during rendering. Leveraging Tensor Cores for efficient computation, we integrate these neural queries directly into a novel instanced software rasterizer. Our approach outperforms the current state of the art for composed scenes in terms of VRAM usage and image quality, utilizing a combination of our instanced rasterizer and occlusion culling MLP, and exhibits complementary properties to existing LoD techniques.
>
---
#### [new 107] Compact neural networks for astronomy with optimal transport bias correction
- **分类: cs.CV**

- **简介: 该论文针对天文图像分类与红移预测中的效率-分辨率矛盾，提出WaveletMamba框架，融合小波分解、状态空间模型与多级偏差校正，实现低参数量下高精度多尺度分类，显著提升计算效率并解决数据分布偏差问题。**

- **链接: [https://arxiv.org/pdf/2511.18139v1](https://arxiv.org/pdf/2511.18139v1)**

> **作者:** Shuhuan Wang; Yuzhen Xie; Jiayi Li
>
> **备注:** 18 pages, 5 figures, 3 tables. Research article
>
> **摘要:** Astronomical imaging confronts an efficiency-resolution tradeoff that limits large-scale morphological classification and redshift prediction. We introduce WaveletMamba, a theory-driven framework integrating wavelet decomposition with state-space modeling, mathematical regularization, and multi-level bias correction. WaveletMamba achieves 81.72% +/- 0.53% classification accuracy at 64x64 resolution with only 3.54M parameters, delivering high-resolution performance (80.93% +/- 0.27% at 244x244) at low-resolution inputs with 9.7x computational efficiency gains. The framework exhibits Resolution Multistability, where models trained on low-resolution data achieve consistent accuracy across different input scales despite divergent internal representations. The framework's multi-level bias correction synergizes HK distance (distribution-level optimal transport) with Color-Aware Weighting (sample-level fine-tuning), achieving 22.96% Log-MSE improvement and 26.10% outlier reduction without explicit selection function modeling. Here, we show that mathematical rigor enables unprecedented efficiency and comprehensive bias correction in scientific AI, bridging computer vision and astrophysics to revolutionize interdisciplinary scientific discovery.
>
---
#### [new 108] Rethinking Garment Conditioning in Diffusion-based Virtual Try-On
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对扩散模型在虚拟试衣（VTON）中计算与内存开销大的问题，提出高效单UNet模型Re-CatVTON。通过分析上下文特征学习机制，优化条件注入与误差控制，提升性能并降低资源消耗，实现更优的效率-性能平衡。**

- **链接: [https://arxiv.org/pdf/2511.18775v1](https://arxiv.org/pdf/2511.18775v1)**

> **作者:** Kihyun Na; Jinyoung Choi; Injung Kim
>
> **备注:** 15 pages (including references and supplementary material), 10 figures, 7 tables. Code and pretrained models will be released
>
> **摘要:** Virtual Try-On (VTON) is the task of synthesizing an image of a person wearing a target garment, conditioned on a person image and a garment image. While diffusion-based VTON models featuring a Dual UNet architecture demonstrate superior fidelity compared to single UNet models, they incur substantial computational and memory overhead due to their heavy structure. In this study, through visualization analysis and theoretical analysis, we derived three hypotheses regarding the learning of context features to condition the denoising process. Based on these hypotheses, we developed Re-CatVTON, an efficient single UNet model that achieves high performance. We further enhance the model by introducing a modified classifier-free guidance strategy tailored for VTON's spatial concatenation conditioning, and by directly injecting the ground-truth garment latent derived from the clean garment latent to prevent the accumulation of prediction error. The proposed Re-CatVTON significantly improves performance compared to its predecessor (CatVTON) and requires less computation and memory than the high-performance Dual UNet model, Leffa. Our results demonstrate improved FID, KID, and LPIPS scores, with only a marginal decrease in SSIM, establishing a new efficiency-performance trade-off for single UNet VTON models.
>
---
#### [new 109] Sequence-Adaptive Video Prediction in Continuous Streams using Diffusion Noise Optimization
- **分类: cs.CV**

- **简介: 该论文研究连续视频流中的视频预测任务，针对大模型微调成本高的问题，提出SAVi-DNO方法，在保持模型参数冻结的前提下，通过优化推理阶段的扩散噪声实现序列自适应。在Ego4D等数据集上验证了其在长视频预测中提升性能的有效性。**

- **链接: [https://arxiv.org/pdf/2511.18255v1](https://arxiv.org/pdf/2511.18255v1)**

> **作者:** Sina Mokhtarzadeh Azar; Emad Bahrami; Enrico Pallotta; Gianpiero Francesca; Radu Timofte; Juergen Gall
>
> **摘要:** In this work, we investigate diffusion-based video prediction models, which forecast future video frames, for continuous video streams. In this context, the models observe continuously new training samples, and we aim to leverage this to improve their predictions. We thus propose an approach that continuously adapts a pre-trained diffusion model to a video stream. Since fine-tuning the parameters of a large diffusion model is too expensive, we refine the diffusion noise during inference while keeping the model parameters frozen, allowing the model to adaptively determine suitable sampling noise. We term the approach Sequence Adaptive Video Prediction with Diffusion Noise Optimization (SAVi-DNO). To validate our approach, we introduce a new evaluation setting on the Ego4D dataset, focusing on simultaneous adaptation and evaluation on long continuous videos. Empirical results demonstrate improved performance based on FVD, SSIM, and PSNR metrics on long videos of Ego4D and OpenDV-YouTube, as well as videos of UCF-101 and SkyTimelapse, showcasing SAVi-DNO's effectiveness.
>
---
#### [new 110] Eevee: Towards Close-up High-resolution Video-based Virtual Try-on
- **分类: cs.CV**

- **简介: 该论文针对视频虚拟试穿任务，解决现有方法依赖单图输入导致细节失真、缺乏近景视频的问题。提出高分辨率数据集，包含真实模特的全景与近景试穿视频及详细图文信息，并设计VGID度量指标，提升纹理与结构一致性评估精度，验证了模型在细节还原上的显著改进。**

- **链接: [https://arxiv.org/pdf/2511.18957v1](https://arxiv.org/pdf/2511.18957v1)**

> **作者:** Jianhao Zeng; Yancheng Bai; Ruidong Chen; Xuanpu Zhang; Lei Sun; Dongyang Jin; Ryan Xu; Nannan Zhang; Dan Song; Xiangxiang Chu
>
> **摘要:** Video virtual try-on technology provides a cost-effective solution for creating marketing videos in fashion e-commerce. However, its practical adoption is hindered by two critical limitations. First, the reliance on a single garment image as input in current virtual try-on datasets limits the accurate capture of realistic texture details. Second, most existing methods focus solely on generating full-shot virtual try-on videos, neglecting the business's demand for videos that also provide detailed close-ups. To address these challenges, we introduce a high-resolution dataset for video-based virtual try-on. This dataset offers two key features. First, it provides more detailed information on the garments, which includes high-fidelity images with detailed close-ups and textual descriptions; Second, it uniquely includes full-shot and close-up try-on videos of real human models. Furthermore, accurately assessing consistency becomes significantly more critical for the close-up videos, which demand high-fidelity preservation of garment details. To facilitate such fine-grained evaluation, we propose a new garment consistency metric VGID (Video Garment Inception Distance) that quantifies the preservation of both texture and structure. Our experiments validate these contributions. We demonstrate that by utilizing the detailed images from our dataset, existing video generation models can extract and incorporate texture features, significantly enhancing the realism and detail fidelity of virtual try-on results. Furthermore, we conduct a comprehensive benchmark of recent models. The benchmark effectively identifies the texture and structural preservation problems among current methods.
>
---
#### [new 111] State and Scene Enhanced Prototypes for Weakly Supervised Open-Vocabulary Object Detection
- **分类: cs.CV**

- **简介: 该论文针对弱监督开放词汇目标检测任务，解决语义原型静态、缺乏对象状态与场景上下文的问题。提出状态增强原型（SESP）和场景增强伪原型（SAPP），分别生成状态感知文本描述和融合上下文的视觉-文本对齐表示，提升原型丰富性与对齐精度，显著改善检测性能。**

- **链接: [https://arxiv.org/pdf/2511.18012v1](https://arxiv.org/pdf/2511.18012v1)**

> **作者:** Jiaying Zhou; Qingchao Chen
>
> **摘要:** Open-Vocabulary Object Detection (OVOD) aims to generalize object recognition to novel categories, while Weakly Supervised OVOD (WS-OVOD) extends this by combining box-level annotations with image-level labels. Despite recent progress, two critical challenges persist in this setting. First, existing semantic prototypes, even when enriched by LLMs, are static and limited, failing to capture the rich intra-class visual variations induced by different object states (e.g., a cat's pose). Second, the standard pseudo-box generation introduces a semantic mismatch between visual region proposals (which contain context) and object-centric text embeddings. To tackle these issues, we introduce two complementary prototype enhancement strategies. To capture intra-class variations in appearance and state, we propose the State-Enhanced Semantic Prototypes (SESP), which generates state-aware textual descriptions (e.g., "a sleeping cat") to capture diverse object appearances, yielding more discriminative prototypes. Building on this, we further introduce Scene-Augmented Pseudo Prototypes (SAPP) to address the semantic mismatch. SAPP incorporates contextual semantics (e.g., "cat lying on sofa") and utilizes a soft alignment mechanism to promote contextually consistent visual-textual representations. By integrating SESP and SAPP, our method effectively enhances both the richness of semantic prototypes and the visual-textual alignment, achieving notable improvements.
>
---
#### [new 112] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视频理解中的程序性活动建模问题，提出PL-Stitch框架。针对现有自监督学习方法忽视动作时序顺序的缺陷，利用Plackett-Luce模型设计时空排序与拼图损失，引导模型学习全局流程与细粒度关联。在手术与烹饪数据集上显著提升识别与分割性能。**

- **链接: [https://arxiv.org/pdf/2511.17805v1](https://arxiv.org/pdf/2511.17805v1)**

> **作者:** Chengan Che; Chao Wang; Xinyue Chen; Sophia Tsoka; Luis C. Garcia-Peraza-Herrera
>
> **备注:** 18 pages
>
> **摘要:** Procedural activities, ranging from routine cooking to complex surgical operations, are highly structured as a set of actions conducted in a specific temporal order. Despite their success on static images and short clips, current self-supervised learning methods often overlook the procedural nature that underpins such activities. We expose the lack of procedural awareness in current SSL methods with a motivating experiment: models pretrained on forward and time-reversed sequences produce highly similar features, confirming that their representations are blind to the underlying procedural order. To address this shortcoming, we propose PL-Stitch, a self-supervised framework that harnesses the inherent temporal order of video frames as a powerful supervisory signal. Our approach integrates two novel probabilistic objectives based on the Plackett-Luce (PL) model. The primary PL objective trains the model to sort sampled frames chronologically, compelling it to learn the global workflow progression. The secondary objective, a spatio-temporal jigsaw loss, complements the learning by capturing fine-grained, cross-frame object correlations. Our approach consistently achieves superior performance across five surgical and cooking benchmarks. Specifically, PL-Stitch yields significant gains in surgical phase recognition (e.g., +11.4 pp k-NN accuracy on Cholec80) and cooking action segmentation (e.g., +5.7 pp linear probing accuracy on Breakfast), demonstrating its effectiveness for procedural video representation learning.
>
---
#### [new 113] SatSAM2: Motion-Constrained Video Object Tracking in Satellite Imagery using Promptable SAM2 and Kalman Priors
- **分类: cs.CV**

- **简介: 该论文针对卫星视频目标跟踪中泛化能力差、易丢失目标的问题，提出SatSAM2。基于SAM2，引入卡尔曼滤波运动约束模块和运动状态机，增强时序一致性与鲁棒性。构建了大规模合成基准MVOT，实验表明其在多个数据集上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18264v1](https://arxiv.org/pdf/2511.18264v1)**

> **作者:** Ruijie Fan; Junyan Ye; Huan Chen; Zilong Huang; Xiaolei Wang; Weijia Li
>
> **摘要:** Existing satellite video tracking methods often struggle with generalization, requiring scenario-specific training to achieve satisfactory performance, and are prone to track loss in the presence of occlusion. To address these challenges, we propose SatSAM2, a zero-shot satellite video tracker built on SAM2, designed to adapt foundation models to the remote sensing domain. SatSAM2 introduces two core modules: a Kalman Filter-based Constrained Motion Module (KFCMM) to exploit temporal motion cues and suppress drift, and a Motion-Constrained State Machine (MCSM) to regulate tracking states based on motion dynamics and reliability. To support large-scale evaluation, we propose MatrixCity Video Object Tracking (MVOT), a synthetic benchmark containing 1,500+ sequences and 157K annotated frames with diverse viewpoints, illumination, and occlusion conditions. Extensive experiments on two satellite tracking benchmarks and MVOT show that SatSAM2 outperforms both traditional and foundation model-based trackers, including SAM2 and its variants. Notably, on the OOTB dataset, SatSAM2 achieves a 5.84% AUC improvement over state-of-the-art methods. Our code and dataset will be publicly released to encourage further research.
>
---
#### [new 114] Breaking the Likelihood-Quality Trade-off in Diffusion Models by Merging Pretrained Experts
- **分类: cs.CV; cs.LG; stat.ML**

- **简介: 该论文针对扩散模型中感知质量与数据似然间的权衡问题，提出一种无需重训练的采样方法：在去噪过程中，高噪声阶段使用质量专家构建全局结构，低噪声阶段切换至似然专家优化像素细节。通过专家切换，同时提升图像质量和似然得分，有效打破二者固有矛盾。**

- **链接: [https://arxiv.org/pdf/2511.19434v1](https://arxiv.org/pdf/2511.19434v1)**

> **作者:** Yasin Esfandiari; Stefan Bauer; Sebastian U. Stich; Andrea Dittadi
>
> **备注:** ICLR 2025 DeLTa workshop
>
> **摘要:** Diffusion models for image generation often exhibit a trade-off between perceptual sample quality and data likelihood: training objectives emphasizing high-noise denoising steps yield realistic images but poor likelihoods, whereas likelihood-oriented training overweights low-noise steps and harms visual fidelity. We introduce a simple plug-and-play sampling method that combines two pretrained diffusion experts by switching between them along the denoising trajectory. Specifically, we apply an image-quality expert at high noise levels to shape global structure, then switch to a likelihood expert at low noise levels to refine pixel statistics. The approach requires no retraining or fine-tuning -- only the choice of an intermediate switching step. On CIFAR-10 and ImageNet32, the merged model consistently matches or outperforms its base components, improving or preserving both likelihood and sample quality relative to each expert alone. These results demonstrate that expert switching across noise levels is an effective way to break the likelihood-quality trade-off in image diffusion models.
>
---
#### [new 115] DiP: Taming Diffusion Models in Pixel Space
- **分类: cs.CV**

- **简介: 该论文针对扩散模型在像素空间生成中效率与质量的权衡问题，提出DiP框架。通过分阶段设计：用大块斑块的DiT构建全局结构，轻量级细节头恢复局部细节，实现高效高质生成。无需VAE，推理速度提升10倍，参数仅增0.3%，FID达1.90。**

- **链接: [https://arxiv.org/pdf/2511.18822v1](https://arxiv.org/pdf/2511.18822v1)**

> **作者:** Zhennan Chen; Junwei Zhu; Xu Chen; Jiangning Zhang; Xiaobin Hu; Hanzhen Zhao; Chengjie Wang; Jian Yang; Ying Tai
>
> **摘要:** Diffusion models face a fundamental trade-off between generation quality and computational efficiency. Latent Diffusion Models (LDMs) offer an efficient solution but suffer from potential information loss and non-end-to-end training. In contrast, existing pixel space models bypass VAEs but are computationally prohibitive for high-resolution synthesis. To resolve this dilemma, we propose DiP, an efficient pixel space diffusion framework. DiP decouples generation into a global and a local stage: a Diffusion Transformer (DiT) backbone operates on large patches for efficient global structure construction, while a co-trained lightweight Patch Detailer Head leverages contextual features to restore fine-grained local details. This synergistic design achieves computational efficiency comparable to LDMs without relying on a VAE. DiP is accomplished with up to 10$\times$ faster inference speeds than previous method while increasing the total number of parameters by only 0.3%, and achieves an 1.90 FID score on ImageNet 256$\times$256.
>
---
#### [new 116] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model
- **分类: cs.CV**

- **简介: 该论文针对遥感图像中仅能获取低分辨率彩色图像与高分辨率灰度图像的问题，提出MFmamba模型，实现单张全色图像的超分辨率重建与光谱恢复。通过改进UNet++结构，引入Mamba上采样块、双池注意力机制与多尺度混合交叉块，统一完成超分辨率、光谱恢复及联合任务，显著提升图像质量。**

- **链接: [https://arxiv.org/pdf/2511.18888v1](https://arxiv.org/pdf/2511.18888v1)**

> **作者:** Qian Jiang; Qianqian Wang; Xin Jin; Michal Wozniak; Shaowen Yao; Wei Zhou
>
> **备注:** 9 pages, 9 figures. This paper has been accepted for publication in AAAI-2026
>
> **摘要:** Remote sensing images are becoming increasingly widespread in military, earth resource exploration. Because of the limitation of a single sensor, we can obtain high spatial resolution grayscale panchromatic (PAN) images and low spatial resolution color multispectral (MS) images. Therefore, an important issue is to obtain a color image with high spatial resolution when there is only a PAN image at the input. The existing methods improve spatial resolution using super-resolution (SR) technology and spectral recovery using colorization technology. However, the SR technique cannot improve the spectral resolution, and the colorization technique cannot improve the spatial resolution. Moreover, the pansharpening method needs two registered inputs and can not achieve SR. As a result, an integrated approach is expected. To solve the above problems, we designed a novel multi-function model (MFmamba) to realize the tasks of SR, spectral recovery, joint SR and spectral recovery through three different inputs. Firstly, MFmamba utilizes UNet++ as the backbone, and a Mamba Upsample Block (MUB) is combined with UNet++. Secondly, a Dual Pool Attention (DPA) is designed to replace the skip connection in UNet++. Finally, a Multi-scale Hybrid Cross Block (MHCB) is proposed for initial feature extraction. Many experiments show that MFmamba is competitive in evaluation metrics and visual results and performs well in the three tasks when only the input PAN image is used.
>
---
#### [new 117] Parallel qMRI Reconstruction from 4x Accelerated Acquisitions
- **分类: cs.CV**

- **简介: 该论文属于并行MRI重建任务，旨在解决加速采集导致的图像质量下降问题。针对4倍加速下的k空间欠采样，提出端到端深度学习框架，联合估计线圈敏感度图并重建图像，无需预计算敏感度图，实现高质量、平滑的重建结果。**

- **链接: [https://arxiv.org/pdf/2511.18232v1](https://arxiv.org/pdf/2511.18232v1)**

> **作者:** Mingi Kang
>
> **摘要:** Magnetic Resonance Imaging (MRI) acquisitions require extensive scan times, limiting patient throughput and increasing susceptibility to motion artifacts. Accelerated parallel MRI techniques reduce acquisition time by undersampling k-space data, but require robust reconstruction methods to recover high-quality images. Traditional approaches like SENSE require both undersampled k-space data and pre-computed coil sensitivity maps. We propose an end-to-end deep learning framework that jointly estimates coil sensitivity maps and reconstructs images from only undersampled k-space measurements at 4x acceleration. Our two-module architecture consists of a Coil Sensitivity Map (CSM) estimation module and a U-Net-based MRI reconstruction module. We evaluate our method on multi-coil brain MRI data from 10 subjects with 8 echoes each, using 2x SENSE reconstructions as ground truth. Our approach produces visually smoother reconstructions compared to conventional SENSE output, achieving comparable visual quality despite lower PSNR/SSIM metrics. We identify key challenges including spatial misalignment between different acceleration factors and propose future directions for improved reconstruction quality.
>
---
#### [new 118] When Better Teachers Don't Make Better Students: Revisiting Knowledge Distillation for CLIP Models in VQA
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLM）中的知识蒸馏（KD）在多模态任务中的应用，聚焦于CLIP模型。针对现有方法中强教师模型未必带来更好学生模型的问题，作者系统评估了不同规模教师模型的蒸馏效果，发现传统框架难以有效扩展，导致下游任务如视觉问答性能下降，挑战了领域内普遍认知。**

- **链接: [https://arxiv.org/pdf/2511.17886v1](https://arxiv.org/pdf/2511.17886v1)**

> **作者:** Pume Tuchinda; Parinthapat Pengpun; Romrawin Chumpu; Sarana Nutanong; Peerat Limkonchotiwat
>
> **摘要:** Vision-language models (VLMs) have achieved remarkable success across multimodal tasks, yet their substantial computational demands hinder efficient deployment. Knowledge distillation (KD) has emerged as a powerful approach for building lightweight but competitive models, with strong evidence from both language and vision domains. However, its application to VLMs, particularly CLIP-style models, remains limited, often constrained to small-scale teachers and narrow evaluation tasks such as classification or retrieval. In this work, we present the first systematic study of distillation across a range of CLIP-style teacher models, ranging from standard baselines to large-scale state-of-the-art models. Contrary to trends observed in NLP and vision, we find that stronger teachers do not consistently yield better students; in fact, existing distillation frameworks often fail to scale, leading to degraded performance in downstream multimodal tasks such as visual question answering. Our findings challenge prevailing assumptions in KD and point toward new directions for designing parameter-efficient multimodal models.
>
---
#### [new 119] Modeling Retinal Ganglion Cells with Neural Differential Equations
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉神经建模任务，旨在高效模拟虎纹蝾螈视网膜神经节细胞活动。针对数据有限、需频繁重训练的场景，提出使用液态时间常数网络（LTCs）和闭式连续时间网络（CfCs），相比卷积网络与LSTM，显著提升效率与模型紧凑性，适用于视觉假体等边缘部署应用。**

- **链接: [https://arxiv.org/pdf/2511.18014v1](https://arxiv.org/pdf/2511.18014v1)**

> **作者:** Kacper Dobek; Daniel Jankowski; Krzysztof Krawiec
>
> **备注:** Accepted to the AAAI-26 Student Abstract and Poster Program, with supplementary material
>
> **摘要:** This work explores Liquid Time-Constant Networks (LTCs) and Closed-form Continuous-time Networks (CfCs) for modeling retinal ganglion cell activity in tiger salamanders across three datasets. Compared to a convolutional baseline and an LSTM, both architectures achieved lower MAE, faster convergence, smaller model sizes, and favorable query times, though with slightly lower Pearson correlation. Their efficiency and adaptability make them well suited for scenarios with limited data and frequent retraining, such as edge deployments in vision prosthetics.
>
---
#### [new 120] VeCoR - Velocity Contrastive Regularization for Flow Matching
- **分类: cs.CV**

- **简介: 该论文针对流匹配（Flow Matching）模型在低步数、轻量级设置下轨迹误差累积、生成质量下降的问题，提出速度对比正则化（VeCoR）。通过引入正负双向监督，增强模型对数据流形的保持能力，提升生成稳定性与图像质量。**

- **链接: [https://arxiv.org/pdf/2511.18942v1](https://arxiv.org/pdf/2511.18942v1)**

> **作者:** Zong-Wei Hong; Jing-lun Li; Lin-Ze Li; Shen Zhang; Yao Tang
>
> **摘要:** Flow Matching (FM) has recently emerged as a principled and efficient alternative to diffusion models. Standard FM encourages the learned velocity field to follow a target direction; however, it may accumulate errors along the trajectory and drive samples off the data manifold, leading to perceptual degradation, especially in lightweight or low-step configurations. To enhance stability and generalization, we extend FM into a balanced attract-repel scheme that provides explicit guidance on both "where to go" and "where not to go." To be formal, we propose \textbf{Velocity Contrastive Regularization (VeCoR)}, a complementary training scheme for flow-based generative modeling that augments the standard FM objective with contrastive, two-sided supervision. VeCoR not only aligns the predicted velocity with a stable reference direction (positive supervision) but also pushes it away from inconsistent, off-manifold directions (negative supervision). This contrastive formulation transforms FM from a purely attractive, one-sided objective into a two-sided training signal, regularizing trajectory evolution and improving perceptual fidelity across datasets and backbones. On ImageNet-1K 256$\times$256, VeCoR yields 22\% and 35\% relative FID reductions on SiT-XL/2 and REPA-SiT-XL/2 backbones, respectively, and achieves further FID gains (32\% relative) on MS-COCO text-to-image generation, demonstrating consistent improvements in stability, convergence, and image quality, particularly in low-step and lightweight settings. Project page: https://p458732.github.io/VeCoR_Project_Page/
>
---
#### [new 121] PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出PhysGS，一种基于贝叶斯推理的3D高斯点阵方法，用于从视觉和语言先验中估计密集的物理属性（如摩擦、硬度等）。针对现有3D重建忽略物理属性的问题，该方法通过迭代更新材料信念并建模不确定性，实现空间连续的物理属性估计，在多个数据集上显著提升估计精度。**

- **链接: [https://arxiv.org/pdf/2511.18570v1](https://arxiv.org/pdf/2511.18570v1)**

> **作者:** Samarth Chopra; Jing Liang; Gershom Seneviratne; Dinesh Manocha
>
> **备注:** Submitted to CVPR 2026
>
> **摘要:** Understanding physical properties such as friction, stiffness, hardness, and material composition is essential for enabling robots to interact safely and effectively with their surroundings. However, existing 3D reconstruction methods focus on geometry and appearance and cannot infer these underlying physical properties. We present PhysGS, a Bayesian-inferred extension of 3D Gaussian Splatting that estimates dense, per-point physical properties from visual cues and vision--language priors. We formulate property estimation as Bayesian inference over Gaussian splats, where material and property beliefs are iteratively refined as new observations arrive. PhysGS also models aleatoric and epistemic uncertainties, enabling uncertainty-aware object and scene interpretation. Across object-scale (ABO-500), indoor, and outdoor real-world datasets, PhysGS improves accuracy of the mass estimation by up to 22.8%, reduces Shore hardness error by up to 61.2%, and lowers kinetic friction error by up to 18.1% compared to deterministic baselines. Our results demonstrate that PhysGS unifies 3D reconstruction, uncertainty modeling, and physical reasoning in a single, spatially continuous framework for dense physical property estimation. Additional results are available at https://samchopra2003.github.io/physgs.
>
---
#### [new 122] ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人操作中计算开销大、推理延迟高的问题，提出ActDistill框架。通过动作引导的自蒸馏机制，将大型VLA模型的动作预测能力迁移到轻量级学生模型，利用图结构建模动作演化并引入动态路由，实现高效推理。实验表明，该方法在保持高性能的同时，计算量减少超50%，速度提升达1.67倍。**

- **链接: [https://arxiv.org/pdf/2511.18082v1](https://arxiv.org/pdf/2511.18082v1)**

> **作者:** Wencheng Ye; Tianshi Wang; Lei Zhu; Fengling Li; Guoli Yang
>
> **摘要:** Recent Vision-Language-Action (VLA) models have shown impressive flexibility and generalization, yet their deployment in robotic manipulation remains limited by heavy computational overhead and inference latency. In this work, we present ActDistill, a general action-guided self-derived distillation framework that transfers the action prediction capability of any existing VLA model to a lightweight counterpart. Unlike previous efficiency strategies that primarily emphasize vision-language correlations, ActDistill leverages action priors to guide knowledge transfer and model compression, achieving action-oriented efficiency for VLA models. Specifically, we employ a well-trained VLA model as the teacher and introduce a graph-structured encapsulation strategy to explicitly model the hierarchical evolution of action prediction. The student model, derived from the graph-encapsulated teacher, is further equipped with a dynamic router that adaptively selects computation paths based on action prediction demands, guided by hierarchical graph-informed supervision to ensure smooth and efficient evolution. During inference, graph-related auxiliary components are removed, allowing the student to execute only dynamically routed layers and predict high-precision actions with minimal computation and latency. Experiments on embodied benchmarks demonstrate that ActDistill achieves comparable or superior performance to full-scale VLA models while reducing computation by over 50% with up to 1.67 times speedup, thereby establishing a general paradigm toward efficient embodied intelligence.
>
---
#### [new 123] VAOT: Vessel-Aware Optimal Transport for Retinal Fundus Enhancement
- **分类: cs.CV**

- **简介: 该论文针对视网膜彩照图像增强任务，解决现有无配对方法导致血管结构扭曲的问题。提出VAOT框架，结合最优传输与双结构保持正则项（骨架连通性与端点稳定），在不依赖配对数据下有效保留血管拓扑与末端完整性，提升图像质量与下游分割性能。**

- **链接: [https://arxiv.org/pdf/2511.18763v1](https://arxiv.org/pdf/2511.18763v1)**

> **作者:** Xuanzhao Dong; Wenhui Zhu; Yujian Xiong; Xiwen Chen; Hao Wang; Xin Li; Jiajun Cheng; Zhipeng Wang; Shao Tang; Oana Dumitrascu; Yalin Wang
>
> **摘要:** Color fundus photography (CFP) is central to diagnosing and monitoring retinal disease, yet its acquisition variability (e.g., illumination changes) often degrades image quality, which motivates robust enhancement methods. Unpaired enhancement pipelines are typically GAN-based, however, they can distort clinically critical vasculature, altering vessel topology and endpoint integrity. Motivated by these structural alterations, we propose Vessel-Aware Optimal Transport (\textbf{VAOT}), a framework that combines an optimal-transport objective with two structure-preserving regularizers: (i) a skeleton-based loss to maintain global vascular connectivity and (ii) an endpoint-aware loss to stabilize local termini. These constraints guide learning in the unpaired setting, reducing noise while preserving vessel structure. Experimental results on synthetic degradation benchmark and downstream evaluations in vessel and lesion segmentation demonstrate the superiority of the proposed methods against several state-of-the art baselines. The code is available at https://github.com/Retinal-Research/VAOT
>
---
#### [new 124] NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering
- **分类: cs.CV**

- **简介: 该论文针对视觉基础模型（VFM）特征图下采样导致的像素级任务难题，提出零样本特征上采样方法NAF。通过跨尺度邻域注意力与旋转位置编码学习自适应权重，无需重训练即可适配任意VFM，实现高效高精度上采样，显著优于现有方法，并在图像修复等任务中展现强泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18452v1](https://arxiv.org/pdf/2511.18452v1)**

> **作者:** Loick Chambon; Paul Couairon; Eloi Zablocki; Alexandre Boulch; Nicolas Thome; Matthieu Cord
>
> **备注:** Code: https://github.com/valeoai/NAF
>
> **摘要:** Vision Foundation Models (VFMs) extract spatially downsampled representations, posing challenges for pixel-level tasks. Existing upsampling approaches face a fundamental trade-off: classical filters are fast and broadly applicable but rely on fixed forms, while modern upsamplers achieve superior accuracy through learnable, VFM-specific forms at the cost of retraining for each VFM. We introduce Neighborhood Attention Filtering (NAF), which bridges this gap by learning adaptive spatial-and-content weights through Cross-Scale Neighborhood Attention and Rotary Position Embeddings (RoPE), guided solely by the high-resolution input image. NAF operates zero-shot: it upsamples features from any VFM without retraining, making it the first VFM-agnostic architecture to outperform VFM-specific upsamplers and achieve state-of-the-art performance across multiple downstream tasks. It maintains high efficiency, scaling to 2K feature maps and reconstructing intermediate-resolution maps at 18 FPS. Beyond feature upsampling, NAF demonstrates strong performance on image restoration, highlighting its versatility. Code and checkpoints are available at https://github.com/valeoai/NAF.
>
---
#### [new 125] Sphinx: Efficiently Serving Novel View Synthesis using Regression-Guided Selective Refinement
- **分类: cs.CV**

- **简介: 该论文针对新视角合成（NVS）任务，解决扩散模型计算昂贵与回归模型质量差的矛盾。提出Sphinx框架，通过回归快速初始化引导扩散模型，并结合选择性精炼与自适应噪声调度，实现高质量、低延迟推理，显著提升效率且保持近似扩散模型的生成质量。**

- **链接: [https://arxiv.org/pdf/2511.18672v1](https://arxiv.org/pdf/2511.18672v1)**

> **作者:** Yuchen Xia; Souvik Kundu; Mosharaf Chowdhury; Nishil Talati
>
> **摘要:** Novel View Synthesis (NVS) is the task of generating new images of a scene from viewpoints that were not part of the original input. Diffusion-based NVS can generate high-quality, temporally consistent images, however, remains computationally prohibitive. Conversely, regression-based NVS offers suboptimal generation quality despite requiring significantly lower compute; leaving the design objective of a high-quality, inference-efficient NVS framework an open challenge. To close this critical gap, we present Sphinx, a training-free hybrid inference framework that achieves diffusion-level fidelity at a significantly lower compute. Sphinx proposes to use regression-based fast initialization to guide and reduce the denoising workload for the diffusion model. Additionally, it integrates selective refinement with adaptive noise scheduling, allowing more compute to uncertain regions and frames. This enables Sphinx to provide flexible navigation of the performance-quality trade-off, allowing adaptation to latency and fidelity requirements for dynamically changing inference scenarios. Our evaluation shows that Sphinx achieves an average 1.8x speedup over diffusion model inference with negligible perceptual degradation of less than 5%, establishing a new Pareto frontier between quality and latency in NVS serving.
>
---
#### [new 126] NeuroVascU-Net: A Unified Multi-Scale and Cross-Domain Adaptive Feature Fusion U-Net for Precise 3D Segmentation of Brain Vessels in Contrast-Enhanced T1 MRI
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出NeuroVascU-Net，用于从T1CE MRI中精确分割脑血管，解决神经外科手术规划中手动分割耗时、现有方法精度与效率难兼顾的问题。通过多尺度与跨域自适应特征融合模块，实现高精度（Dice=0.8609）低参数（12.4M）的3D血管分割，适用于临床实践。**

- **链接: [https://arxiv.org/pdf/2511.18422v1](https://arxiv.org/pdf/2511.18422v1)**

> **作者:** Mohammad Jafari Vayeghan; Niloufar Delfan; Mehdi Tale Masouleh; Mansour Parvaresh Rizi; Behzad Moshiri
>
> **摘要:** Precise 3D segmentation of cerebral vasculature from T1-weighted contrast-enhanced (T1CE) MRI is crucial for safe neurosurgical planning. Manual delineation is time-consuming and prone to inter-observer variability, while current automated methods often trade accuracy for computational cost, limiting clinical use. We present NeuroVascU-Net, the first deep learning architecture specifically designed to segment cerebrovascular structures directly from clinically standard T1CE MRI in neuro-oncology patients, addressing a gap in prior work dominated by TOF-MRA-based approaches. NeuroVascU-Net builds on a dilated U-Net and integrates two specialized modules: a Multi-Scale Contextual Feature Fusion ($MSC^2F$) module at the bottleneck and a Cross-Domain Adaptive Feature Fusion ($CDA^2F$) module at deeper hierarchical layers. $MSC^2F$ captures both local and global information via multi-scale dilated convolutions, while $CDA^2F$ dynamically integrates domain-specific features, enhancing representation while keeping computation low. The model was trained and validated on a curated dataset of T1CE scans from 137 brain tumor biopsy patients, annotated by a board-certified functional neurosurgeon. NeuroVascU-Net achieved a Dice score of 0.8609 and precision of 0.8841, accurately segmenting both major and fine vascular structures. Notably, it requires only 12.4M parameters, significantly fewer than transformer-based models such as Swin U-NetR. This balance of accuracy and efficiency positions NeuroVascU-Net as a practical solution for computer-assisted neurosurgical planning.
>
---
#### [new 127] Beyond Reward Margin: Rethinking and Resolving Likelihood Displacement in Diffusion Models via Video Generation
- **分类: cs.CV**

- **简介: 该论文针对扩散模型在视频生成中因直接偏好优化（DPO）导致的似然位移问题，分析其根源为奖励间距过小或过大引发的优化冲突与次优最大化。提出PG-DPO方法，结合自适应拒绝缩放与隐式偏好正则化，有效缓解该问题，显著提升生成质量与对齐效果。**

- **链接: [https://arxiv.org/pdf/2511.19049v1](https://arxiv.org/pdf/2511.19049v1)**

> **作者:** Ruojun Xu; Yu Kai; Xuhua Ren; Jiaxiang Cheng; Bing Ma; Tianxiang Zheng; Qinhlin Lu
>
> **摘要:** Direct Preference Optimization (DPO) has shown promising results in aligning generative outputs with human preferences by distinguishing between chosen and rejected samples. However, a critical limitation of DPO is likelihood displacement, where the probabilities of chosen samples paradoxically decrease during training, undermining the quality of generation. Although this issue has been investigated in autoregressive models, its impact within diffusion-based models remains largely unexplored. This gap leads to suboptimal performance in tasks involving video generation. To address this, we conduct a formal analysis of DPO loss through updating policy within the diffusion framework, which describes how the updating of specific training samples influences the model's predictions on other samples. Using this tool, we identify two main failure modes: (1) Optimization Conflict, which arises from small reward margins between chosen and rejected samples, and (2) Suboptimal Maximization, caused by large reward margins. Informed by these insights, we introduce a novel solution named Policy-Guided DPO (PG-DPO), combining Adaptive Rejection Scaling (ARS) and Implicit Preference Regularization (IPR) to effectively mitigate likelihood displacement. Experiments show that PG-DPO outperforms existing methods in both quantitative metrics and qualitative evaluations, offering a robust solution for improving preference alignment in video generation tasks.
>
---
#### [new 128] Towards Open-Ended Visual Scientific Discovery with Sparse Autoencoders
- **分类: cs.CV**

- **简介: 该论文研究如何利用稀疏自编码器（SAEs）从科学基础模型的表示中实现开放式的模式发现。针对现有方法仅能验证预设模式、无法发现未知规律的问题，提出通过SAEs挖掘模型内部潜在结构。在生态图像上验证了其发现细粒度解剖结构的能力，证明方法可推广至其他科学领域，为科学发现提供了新工具。**

- **链接: [https://arxiv.org/pdf/2511.17735v1](https://arxiv.org/pdf/2511.17735v1)**

> **作者:** Samuel Stevens; Jacob Beattie; Tanya Berger-Wolf; Yu Su
>
> **摘要:** Scientific archives now contain hundreds of petabytes of data across genomics, ecology, climate, and molecular biology that could reveal undiscovered patterns if systematically analyzed at scale. Large-scale, weakly-supervised datasets in language and vision have driven the development of foundation models whose internal representations encode structure (patterns, co-occurrences and statistical regularities) beyond their training objectives. Most existing methods extract structure only for pre-specified targets; they excel at confirmation but do not support open-ended discovery of unknown patterns. We ask whether sparse autoencoders (SAEs) can enable open-ended feature discovery from foundation model representations. We evaluate this question in controlled rediscovery studies, where the learned SAE features are tested for alignment with semantic concepts on a standard segmentation benchmark and compared against strong label-free alternatives on concept-alignment metrics. Applied to ecological imagery, the same procedure surfaces fine-grained anatomical structure without access to segmentation or part labels, providing a scientific case study with ground-truth validation. While our experiments focus on vision with an ecology case study, the method is domain-agnostic and applicable to models in other sciences (e.g., proteins, genomics, weather). Our results indicate that sparse decomposition provides a practical instrument for exploring what scientific foundation models have learned, an important prerequisite for moving from confirmation to genuine discovery.
>
---
#### [new 129] AngioDG: Interpretable Channel-informed Feature-modulated Single-source Domain Generalization for Coronary Vessel Segmentation in X-ray Angiography
- **分类: cs.CV**

- **简介: 该论文针对X射线冠状动脉造影中冠状血管分割的域泛化问题，提出AngioDG方法。针对单源域泛化中模型易过拟合于合成数据的问题，通过通道重要性分析与重加权，增强域不变特征，提升模型在未知域上的泛化能力，实现可解释的高性能分割。**

- **链接: [https://arxiv.org/pdf/2511.17724v1](https://arxiv.org/pdf/2511.17724v1)**

> **作者:** Mohammad Atwany; Mojtaba Lashgari; Robin P. Choudhury; Vicente Grau; Abhirup Banerjee
>
> **摘要:** Cardiovascular diseases are the leading cause of death globally, with X-ray Coronary Angiography (XCA) as the gold standard during real-time cardiac interventions. Segmentation of coronary vessels from XCA can facilitate downstream quantitative assessments, such as measurement of the stenosis severity and enhancing clinical decision-making. However, developing generalizable vessel segmentation models for XCA is challenging due to variations in imaging protocols and patient demographics that cause domain shifts. These limitations are exacerbated by the lack of annotated datasets, making Single-source Domain Generalization (SDG) a necessary solution for achieving generalization. Existing SDG methods are largely augmentation-based, which may not guarantee the mitigation of overfitting to augmented or synthetic domains. We propose a novel approach, ``AngioDG", to bridge this gap by channel regularization strategy to promote generalization. Our method identifies the contributions of early feature channels to task-specific metrics for DG, facilitating interpretability, and then reweights channels to calibrate and amplify domain-invariant features while attenuating domain-specific ones. We evaluate AngioDG on 6 x-ray angiography datasets for coronary vessels segmentation, achieving the best out-of-distribution performance among the compared methods, while maintaining consistent in-domain test performance.
>
---
#### [new 130] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search
- **分类: cs.CV**

- **简介: 该论文聚焦于人本开放未来任务发现（HOTD）任务，旨在让大模型在动态多变的人类意图中识别能降低人力负担的未来任务。提出HOTD-Bench基准与CMAST搜索框架，通过多智能体协同树状搜索提升任务发现能力，显著优于现有方法，并可兼容主流大模型。**

- **链接: [https://arxiv.org/pdf/2511.18929v1](https://arxiv.org/pdf/2511.18929v1)**

> **作者:** Zijian Song; Xiaoxin Lin; Tao Pu; Zhenlong Yuan; Guangrun Wang; Liang Lin
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** Recent progress in robotics and embodied AI is largely driven by Large Multimodal Models (LMMs). However, a key challenge remains underexplored: how can we advance LMMs to discover tasks that directly assist humans in open-future scenarios, where human intentions are highly concurrent and dynamic. In this work, we formalize the problem of Human-centric Open-future Task Discovery (HOTD), focusing particularly on identifying tasks that reduce human effort across multiple plausible futures. To facilitate this study, we propose an HOTD-Bench, which features over 2K real-world videos, a semi-automated annotation pipeline, and a simulation-based protocol tailored for open-set future evaluation. Additionally, we propose the Collaborative Multi-Agent Search Tree (CMAST) framework, which decomposes the complex reasoning through a multi-agent system and structures the reasoning process through a scalable search tree module. In our experiments, CMAST achieves the best performance on the HOTD-Bench, significantly surpassing existing LMMs. It also integrates well with existing LMMs, consistently improving performance.
>
---
#### [new 131] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多场景下的视觉理解持续学习任务，针对MLLM在动态场景中易发生灾难性遗忘的问题，构建了涵盖四类场景的MSVQA数据集，并提出UNIFIER框架。通过分枝解耦与特征空间一致性约束，有效缓解跨场景遗忘，实现知识积累。**

- **链接: [https://arxiv.org/pdf/2511.18507v1](https://arxiv.org/pdf/2511.18507v1)**

> **作者:** Kai Jiang; Siqi Huang; Xiangyu Chen; Jiawei Shao; Hongyuan Zhang; Xuelong Li
>
> **备注:** 18 pages, 16 figures. This is a preprint version of a paper submitted to CVPR 2026
>
> **摘要:** Continual learning in visual understanding aims to deal with catastrophic forgetting in Multimodal Large Language Models (MLLMs). MLLMs deployed on devices have to continuously adapt to dynamic scenarios in downstream tasks, such as variations in background and perspective, to effectively perform complex visual tasks. To this end, we construct a multimodal visual understanding dataset (MSVQA) encompassing four different scenarios and perspectives including high altitude, underwater, low altitude and indoor, to investigate the catastrophic forgetting in MLLMs under the dynamics of scenario shifts in real-world data streams. Furthermore, we propose mUltimodal coNtInual learning with MLLMs From multi-scenarIo pERspectives (UNIFIER) to address visual discrepancies while learning different scenarios. Specifically, it decouples the visual information from different scenarios into distinct branches within each vision block and projects them into the same feature space. A consistency constraint is imposed on the features of each branch to maintain the stability of visual representations across scenarios. Extensive experiments on the MSVQA dataset demonstrate that UNIFIER effectively alleviates forgetting of cross-scenario tasks and achieves knowledge accumulation within the same scenario.
>
---
#### [new 132] Explainable Deep Learning for Brain Tumor Classification: Comprehensive Benchmarking with Dual Interpretability and Lightweight Deployment
- **分类: cs.CV; cs.AI; cs.CY**

- **简介: 该论文针对脑肿瘤MRI图像自动分类任务，解决模型准确性、可解释性与轻量化部署难题。通过标准化评估流程，对比六种模型，提出1.31M参数的轻量CNN，实现96.49%准确率与实时推理，结合Grad-CAM和GradientShap提升可解释性，适用于资源匮乏医疗环境。**

- **链接: [https://arxiv.org/pdf/2511.17655v1](https://arxiv.org/pdf/2511.17655v1)**

> **作者:** Md. Mohaiminul Islam; Md. Mofazzal Hossen; Maher Ali Rusho; Nahiyan Nazah Ridita; Zarin Tasnia Shanta; Md. Simanto Haider; Ahmed Faizul Haque Dhrubo; Md. Khurshid Jahan; Mohammad Abdul Qayum
>
> **备注:** This paper contains 17 pages, 4 tables, and 19 figures. This Paper is already accepted in IEEE Computational Intelligence Magazine (CIM)
>
> **摘要:** Our study provides a full deep learning system for automated classification of brain tumors from MRI images, includes six benchmarked architectures (five ImageNet-pre-trained models (VGG-16, Inception V3, ResNet-50, Inception-ResNet V2, Xception) and a custom built, compact CNN (1.31M params)). The study moves the needle forward in a number of ways, including (1) full standardization of assessment with respect to preprocessing, training sets/protocols (optimizing networks with the AdamW optimizer, CosineAnnealingLR, patiene for early stopping = 7), and metrics to assess performance were identical along all models; (2) a high level of confidence in the localizations based on prior studies as both Grad-CAM and GradientShap explanation were used to establish anatomically important and meaningful attention regions and address the black-box issue; (3) a compact 1.31 million parameter CNN was developed that achieved 96.49% testing accuracy and was 100 times smaller than Inception-ResNet V2 while permitting real-time inference (375ms) on edge devices; (4) full evaluation beyond accuracy reporting based on measures of intersection over union, Hausdorff distance, and precision-recall curves, and confusion matrices across all splits. Inception-ResNet V2 reached state-of-the-art performance, achieving a 99.53% accuracy on testing and obtaining a precision, recall, and F1-score of at least 99.50% dominant performance based on metrics of recent studies. We demonstrated a lightweight model that is suitable to deploy on devices that do not have multi-GPU infrastructure in under-resourced settings. This end-to-end solution considers accuracy, interpretability, and deployability of trustworthy AI to create the framework necessary for performance assessment and deployment within advance and low-resource healthcare systems to an extent that enabled participation at the clinical screening and triage level.
>
---
#### [new 133] ChineseVideoBench: Benchmarking Multi-modal Large Models for Chinese Video Question Answering
- **分类: cs.CV**

- **简介: 该论文针对中文视频问答任务，提出ChineseVideoBench基准。旨在解决现有评估框架缺乏文化语境理解的问题。构建包含8大类12子类的中文视频数据集，设计针对性评价指标，评估多模态大模型表现，揭示当前模型在深层语义与文化理解上的挑战。**

- **链接: [https://arxiv.org/pdf/2511.18399v1](https://arxiv.org/pdf/2511.18399v1)**

> **作者:** Yuxiang Nie; Han Wang; Yongjie Ye; Haiyang Yu; Weitao Jia; Tao Zeng; Hao Feng; Xiang Fei; Yang Li; Xiaohui Lv; Guozhi Tang; Jingqun Tang; Jinghui Lu; Zehui Dai; Jiacong Wang; Dingkang Yang; An-Lan Wang; Can Huang
>
> **摘要:** This paper introduces ChineseVideoBench, a pioneering benchmark specifically designed for evaluating Multimodal Large Language Models (MLLMs) in Chinese Video Question Answering. The growing demand for sophisticated video analysis capabilities highlights the critical need for comprehensive, culturally-aware evaluation frameworks. ChineseVideoBench addresses this gap by providing a robust dataset and tailored evaluation metrics, enabling rigorous assessment of state-of-the-art MLLMs on complex Chinese video content. Specifically, ChineseVideoBench comprises 8 main classes and 12 sub-classes, encompassing tasks that demand both deep video understanding and nuanced Chinese linguistic and cultural awareness. Our empirical evaluations reveal that ChineseVideoBench presents a significant challenge to current MLLMs. Among the models assessed, Gemini 2.5 Pro achieves the highest performance with an overall score of 77.9%, while InternVL-38B emerges as the most competitive open-source model.
>
---
#### [new 134] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixtures
- **分类: cs.CV**

- **简介: 该论文针对零样本异常检测（ZSAD）任务，解决现有方法因提示工程受限导致泛化能力差的问题。提出PromptMoE框架，通过视觉引导的专家混合机制，动态组合多组语义提示，生成更具表达力的文本表征，提升对未见异常的识别与定位能力。**

- **链接: [https://arxiv.org/pdf/2511.18116v1](https://arxiv.org/pdf/2511.18116v1)**

> **作者:** Yuheng Shao; Lizhang Wang; Changhao Li; Peixian Chen; Qinyuan Liu
>
> **备注:** 14 pages, 8 figures. Accepted to AAAI 2026
>
> **摘要:** Zero-Shot Anomaly Detection (ZSAD) aims to identify and localize anomalous regions in images of unseen object classes. While recent methods based on vision-language models like CLIP show promise, their performance is constrained by existing prompt engineering strategies. Current approaches, whether relying on single fixed, learnable, or dense dynamic prompts, suffer from a representational bottleneck and are prone to overfitting on auxiliary data, failing to generalize to the complexity and diversity of unseen anomalies. To overcome these limitations, we propose $\mathtt{PromptMoE}$. Our core insight is that robust ZSAD requires a compositional approach to prompt learning. Instead of learning monolithic prompts, $\mathtt{PromptMoE}$ learns a pool of expert prompts, which serve as a basis set of composable semantic primitives, and a visually-guided Mixture-of-Experts (MoE) mechanism to dynamically combine them for each instance. Our framework materializes this concept through a Visually-Guided Mixture of Prompt (VGMoP) that employs an image-gated sparse MoE to aggregate diverse normal and abnormal expert state prompts, generating semantically rich textual representations with strong generalization. Extensive experiments across 15 datasets in industrial and medical domains demonstrate the effectiveness and state-of-the-art performance of $\mathtt{PromptMoE}$.
>
---
#### [new 135] Assessing the alignment between infants' visual and linguistic experience using multimodal language models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究婴幼儿视觉与语言经验的时间对齐问题，旨在解决早期词汇学习中多模态信息协同不足的难题。通过引入CLIP模型自动分析婴儿视角视频中的视觉-语言对齐度，发现理想对齐事件在日常生活中罕见且存在个体差异，为儿童语言习得模型提供了新方法与重要约束。**

- **链接: [https://arxiv.org/pdf/2511.18824v1](https://arxiv.org/pdf/2511.18824v1)**

> **作者:** Alvin Wei Ming Tan; Jane Yang; Tarun Sepuri; Khai Loong Aw; Robert Z. Sparks; Zi Yin; Virginia A. Marchman; Michael C. Frank; Bria Long
>
> **摘要:** Figuring out which objects or concepts words refer to is a central language learning challenge for young children. Most models of this process posit that children learn early object labels from co-occurrences of words and their referents that occur when someone around them talks about an object in the immediate physical environment. But how aligned in time are children's visual and linguistic experiences during everyday learning? To date, answers to this question have been limited by the need for labor-intensive manual annotations of vision-language co-occurrences. Here, we evaluate the use of contrastive language-image pretraining (CLIP) models to automatically characterize vision-language alignment in egocentric videos taken from the infant perspective in home environments. After validating CLIP alignment scores using human alignment judgments, we apply this metric to a large corpus of infant-perspective videos. We show that idealized aligned moments for learning (e.g., "look at the ball" with a ball present in the child's view) are relatively rare in children's everyday experiences compared to modern machine learning datasets, and highlight variability in alignment both within and across children. These findings suggest that infrequent alignment is a constraint for models describing early word learning and offer a new method for investigating children's multimodal environment.
>
---
#### [new 136] Peregrine: One-Shot Fine-Tuning for FHE Inference of General Deep CNNs
- **分类: cs.CV**

- **简介: 该论文针对全同态加密（FHE）下通用深度卷积神经网络（CNN）推理的两大难题：非线性激活函数的低次多项式近似与密文容量限制。提出单阶段微调策略（SFT）实现高效FHE友好模型转换，并设计广义交错打包（GIP）方案支持任意分辨率特征图，首次实现基于FHE的YOLO目标检测推理。**

- **链接: [https://arxiv.org/pdf/2511.18976v1](https://arxiv.org/pdf/2511.18976v1)**

> **作者:** Huaming Ling; Ying Wang; Si Chen; Junfeng Fan
>
> **摘要:** We address two fundamental challenges in adapting general deep CNNs for FHE-based inference: approximating non-linear activations such as ReLU with low-degree polynomials while minimizing accuracy degradation, and overcoming the ciphertext capacity barrier that constrains high-resolution image processing on FHE inference. Our contributions are twofold: (1) a single-stage fine-tuning (SFT) strategy that directly converts pre-trained CNNs into FHE-friendly forms using low-degree polynomials, achieving competitive accuracy with minimal training overhead; and (2) a generalized interleaved packing (GIP) scheme that is compatible with feature maps of virtually arbitrary spatial resolutions, accompanied by a suite of carefully designed homomorphic operators that preserve the GIP-form encryption throughout computation. These advances enable efficient, end-to-end FHE inference across diverse CNN architectures. Experiments on CIFAR-10, ImageNet, and MS COCO demonstrate that the FHE-friendly CNNs obtained via our SFT strategy achieve accuracy comparable to baselines using ReLU or SiLU activations. Moreover, this work presents the first demonstration of FHE-based inference for YOLO architectures in object detection leveraging low-degree polynomial activations.
>
---
#### [new 137] A Tri-Modal Dataset and a Baseline System for Tracking Unmanned Aerial Vehicles
- **分类: cs.CV**

- **简介: 该论文针对低空无人机多目标跟踪难题，提出首个三模态（RGB、红外、事件）公开数据集MM-UAV及基线系统。通过自适应对齐与动态融合机制，结合事件信号增强关联，提升复杂环境下的跟踪鲁棒性，推动多模态无人机跟踪研究。**

- **链接: [https://arxiv.org/pdf/2511.18344v1](https://arxiv.org/pdf/2511.18344v1)**

> **作者:** Tianyang Xu; Jinjie Gu; Xuefeng Zhu; XiaoJun Wu; Josef Kittler
>
> **摘要:** With the proliferation of low altitude unmanned aerial vehicles (UAVs), visual multi-object tracking is becoming a critical security technology, demanding significant robustness even in complex environmental conditions. However, tracking UAVs using a single visual modality often fails in challenging scenarios, such as low illumination, cluttered backgrounds, and rapid motion. Although multi-modal multi-object UAV tracking is more resilient, the development of effective solutions has been hindered by the absence of dedicated public datasets. To bridge this gap, we release MM-UAV, the first large-scale benchmark for Multi-Modal UAV Tracking, integrating three key sensing modalities, e.g. RGB, infrared (IR), and event signals. The dataset spans over 30 challenging scenarios, with 1,321 synchronised multi-modal sequences, and more than 2.8 million annotated frames. Accompanying the dataset, we provide a novel multi-modal multi-UAV tracking framework, designed specifically for UAV tracking applications and serving as a baseline for future research. Our framework incorporates two key technical innovations, e.g. an offset-guided adaptive alignment module to resolve spatio mismatches across sensors, and an adaptive dynamic fusion module to balance complementary information conveyed by different modalities. Furthermore, to overcome the limitations of conventional appearance modelling in multi-object tracking, we introduce an event-enhanced association mechanism that leverages motion cues from the event modality for more reliable identity maintenance. Comprehensive experiments demonstrate that the proposed framework consistently outperforms state-of-the-art methods. To foster further research in multi-modal UAV tracking, both the dataset and source code will be made publicly available at https://xuefeng-zhu5.github.io/MM-UAV/.
>
---
#### [new 138] ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文档视觉问答（Document VQA）任务，解决现有方法在文本准确率与答案空间定位可靠性之间难以兼顾的问题。提出ARIAL框架，通过大模型规划器协调OCR、语义检索、答案生成与文本到区域对齐等模块，实现精准答案提取与可解释的空间定位，在多个基准上达到最优性能。**

- **链接: [https://arxiv.org/pdf/2511.18192v1](https://arxiv.org/pdf/2511.18192v1)**

> **作者:** Ahmad Mohammadshirazi; Pinaki Prasad Guha Neogi; Dheeraj Kulshrestha; Rajiv Ramnath
>
> **摘要:** Document Visual Question Answering (VQA) requires models to not only extract accurate textual answers but also precisely localize them within document images, a capability critical for interpretability in high-stakes applications. However, existing systems achieve strong textual accuracy while producing unreliable spatial grounding, or sacrifice performance for interpretability. We present ARIAL (Agentic Reasoning for Interpretable Answer Localization), a modular framework that orchestrates specialized tools through an LLM-based planning agent to achieve both precise answer extraction and reliable spatial grounding. ARIAL decomposes Document VQA into structured subtasks: OCR-based text extraction with TrOCR, retrieval-augmented context selection using semantic search, answer generation via a fine-tuned Gemma 3-27B model, and explicit bounding-box localization through text-to-region alignment. This modular architecture produces transparent reasoning traces, enabling tool-level auditability and independent component optimization. We evaluate ARIAL on four benchmarks (DocVQA, FUNSD, CORD, and SROIE) using both textual accuracy (ANLS) and spatial precision (mAP at IoU 0.50 to 0.95). ARIAL achieves state-of-the-art results across all datasets: 88.7 ANLS and 50.1 mAP on DocVQA, 90.0 ANLS and 50.3 mAP on FUNSD, 85.5 ANLS and 60.2 mAP on CORD, and 93.1 ANLS on SROIE, surpassing the previous best method (DLaVA) by +2.8 ANLS and +3.9 mAP on DocVQA. Our work demonstrates how agentic orchestration of specialized tools can simultaneously improve performance and interpretability, providing a pathway toward trustworthy, explainable document AI systems.
>
---
#### [new 139] JigsawComm: Joint Semantic Feature Encoding and Transmission for Communication-Efficient Cooperative Perception
- **分类: cs.CV**

- **简介: 该论文针对多智能体协同感知中的通信效率问题，提出JigsawComm框架。通过联合优化语义特征编码与传输，利用语义重要性评估实现高效、无冗余的数据交换，在保持感知精度的同时，通信量减少500倍以上，显著提升系统可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.17843v1](https://arxiv.org/pdf/2511.17843v1)**

> **作者:** Chenyi Wang; Zhaowei Li; Ming F. Li; Wujie Wen
>
> **摘要:** Multi-agent cooperative perception (CP) promises to overcome the inherent occlusion and sensing-range limitations of single-agent systems (e.g., autonomous driving). However, its practicality is severely constrained by the limited communication bandwidth. Existing approaches attempt to improve bandwidth efficiency via compression or heuristic message selection, without considering the semantic relevance or cross-agent redundancy of sensory data. We argue that a practical CP system must maximize the contribution of every transmitted bit to the final perception task, by extracting and transmitting semantically essential and non-redundant data. In this paper, we formulate a joint semantic feature encoding and transmission problem, which aims to maximize CP accuracy under limited bandwidth. To solve this problem, we introduce JigsawComm, an end-to-end trained, semantic-aware, and communication-efficient CP framework that learns to ``assemble the puzzle'' of multi-agent feature transmission. It uses a regularized encoder to extract semantically-relevant and sparse features, and a lightweight Feature Utility Estimator to predict the contribution of each agent's features to the final perception task. The resulting meta utility maps are exchanged among agents and leveraged to compute a provably optimal transmission policy, which selects features from agents with the highest utility score for each location. This policy inherently eliminates redundancy and achieves a scalable $\mathcal{O}(1)$ communication cost as the number of agents increases. On the benchmarks OPV2V and DAIR-V2X, JigsawComm reduces the total data volume by up to $>$500$\times$ while achieving matching or superior accuracy compared to state-of-the-art methods.
>
---
#### [new 140] Exploring Weak-to-Strong Generalization for CLIP-based Classification
- **分类: cs.CV**

- **简介: 该论文研究视觉-语言模型中的弱到强泛化问题，针对CLIP分类任务，提出类原型学习（CPL）方法，在弱监督下通过学习更代表性的类别原型，提升模型性能。实验表明，该方法在预训练有限时显著优于基线，提升达3.67%。**

- **链接: [https://arxiv.org/pdf/2511.18396v1](https://arxiv.org/pdf/2511.18396v1)**

> **作者:** Jinhao Li; Sarah M. Erfani; Lei Feng; James Bailey; Feng Liu
>
> **备注:** TMLR
>
> **摘要:** Aligning large-scale commercial models with user intent is crucial to preventing harmful outputs. Current methods rely on human supervision but become impractical as model complexity increases. When models surpass human knowledge, providing accurate feedback becomes challenging and inefficient. A novel solution proposed recently is using a weaker model to supervise a stronger model. This concept leverages the ability of weaker models to perform evaluations, thereby reducing the workload on human supervisors. Previous work has shown the effectiveness of weak-to-strong generalization in the context of language-only models. Extending this concept to vision-language models leverages these insights, adapting the proven benefits to a multi-modal context. In our study, we explore weak-to-strong generalization for CLIP-based classification. We propose a method, class prototype learning (CPL), which aims to enhance the classification capabilities of the CLIP model, by learning more representative prototypes for each category. Our findings indicate that, despite using a simple loss function under weak supervision, CPL yields robust improvements in targeted scenarios, particularly when pretraining is limited. Extensive experiments demonstrate that our approach is effective under these settings, achieving a 3.67% improvement over strong baseline methods.
>
---
#### [new 141] Thinking Ahead: Foresight Intelligence in MLLMs and World Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦于视觉语言模型的前瞻智能，旨在解决现有模型难以推理未来事件的问题。提出FSU-QA数据集，用于评估和提升模型对未来情境的预测能力，通过实验证明其可显著增强模型的前瞻性推理，为发展具备预见性理解能力的新一代AI模型提供基础。**

- **链接: [https://arxiv.org/pdf/2511.18735v1](https://arxiv.org/pdf/2511.18735v1)**

> **作者:** Zhantao Gong; Liaoyuan Fan; Qing Guo; Xun Xu; Xulei Yang; Shijie Li
>
> **备注:** 25 pages, 27 figures, submitted to CVPR 2026
>
> **摘要:** In this work, we define Foresight Intelligence as the capability to anticipate and interpret future events-an ability essential for applications such as autonomous driving, yet largely overlooked by existing research. To bridge this gap, we introduce FSU-QA, a new Visual Question-Answering (VQA) dataset specifically designed to elicit and evaluate Foresight Intelligence. Using FSU-QA, we conduct the first comprehensive study of state-of-the-art Vision-Language Models (VLMs) under foresight-oriented tasks, revealing that current models still struggle to reason about future situations. Beyond serving as a benchmark, FSU-QA also enables the assessment of world models by measuring the semantic coherence of their generated predictions, quantified through performance gains when VLMs are augmented with such outputs. Our experiments further demonstrate that FSU-QA can effectively enhance foresight reasoning: even small VLMs fine-tuned on FSU-QA surpass much larger, advanced models by a substantial margin. Together, these findings position FSU-QA as a principled foundation for developing next-generation models capable of truly anticipating and understanding future events.
>
---
#### [new 142] VisReason: A Large-Scale Dataset for Visual Chain-of-Thought Reasoning
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出VisReason，一个大规模多领域视觉链式思维数据集，解决多模态大模型缺乏系统性视觉推理训练数据的问题。通过构建包含48.9万例带人类级分步推理的图像标注数据，提升模型在视觉理解中的可解释性与泛化能力，推动多模态智能向类人推理发展。**

- **链接: [https://arxiv.org/pdf/2511.17731v1](https://arxiv.org/pdf/2511.17731v1)**

> **作者:** Lingxiao Li; Yifan Wang; Xinyan Gao; Chen Tang; Xiangyu Yue; Chenyu You
>
> **摘要:** Chain-of-Thought (CoT) prompting has proven remarkably effective for eliciting complex reasoning in large language models (LLMs). Yet, its potential in multimodal large language models (MLLMs) remains largely untapped, hindered by the absence of large-scale datasets that capture the rich, spatially grounded reasoning intrinsic to visual understanding. Existing visual-CoT resources are typically small, domain-specific, or lack the human-like stepwise structure necessary for compositional visual reasoning. In this paper, we introduce VisReason, a large-scale dataset designed to advance visual Chain-of-Thought reasoning. VisReason comprises 489K annotated examples spanning four diverse domains, each featuring multi-round, human-like rationales that guide MLLMs through interpretable visual reasoning steps. Building upon this, we curate VisReason-Pro, a 165K subset produced with a stronger expert-level GPT annotator, enriched with detailed reasoning traces and 3D spatial grounding via depth-informed annotations. Fine-tuning the state-of-the-art Qwen2.5-VL model on VisReason and VisReason-Pro yields substantial improvements in step-by-step visual reasoning accuracy, interpretability, and cross-benchmark generalization. These results demonstrate that VisReason equips MLLMs with more systematic and generalizable reasoning capabilities. We envision VisReason as a cornerstone for cultivating human-like visual reasoning, paving the way toward the next generation of multimodal intelligence.
>
---
#### [new 143] UniRSCD: A Unified Novel Architectural Paradigm for Remote Sensing Change Detection
- **分类: cs.CV**

- **简介: 该论文针对遥感变化检测中多任务输出粒度不统一、需定制解码器的问题，提出统一架构UniRSCD。基于状态空间模型，设计频率变化提示生成器作为统一编码器，融合高低频信息，无需专用解码器；通过共享表示空间与任务自适应映射，实现二值、语义及建筑损毁等多任务统一处理，显著提升泛化性与性能。**

- **链接: [https://arxiv.org/pdf/2511.17930v1](https://arxiv.org/pdf/2511.17930v1)**

> **作者:** Yuan Qu; Zhipeng Zhang; Chaojun Xu; Qiao Wan; Mengying Xie; Yuzeng Chen; Zhenqi Liu; Yanfei Zhong
>
> **摘要:** In recent years, remote sensing change detection has garnered significant attention due to its critical role in resource monitoring and disaster assessment. Change detection tasks exist with different output granularities such as BCD, SCD, and BDA. However, existing methods require substantial expert knowledge to design specialized decoders that compensate for information loss during encoding across different tasks. This not only introduces uncertainty into the process of selecting optimal models for abrupt change scenarios (such as disaster outbreaks) but also limits the universality of these architectures. To address these challenges, this paper proposes a unified, general change detection framework named UniRSCD. Building upon a state space model backbone, we introduce a frequency change prompt generator as a unified encoder. The encoder dynamically scans bitemporal global context information while integrating high-frequency details with low-frequency holistic information, thereby eliminating the need for specialized decoders for feature compensation. Subsequently, the unified decoder and prediction head establish a shared representation space through hierarchical feature interaction and task-adaptive output mapping. This integrating various tasks such as binary change detection and semantic change detection into a unified architecture, thereby accommodating the differing output granularity requirements of distinct change detection tasks. Experimental results demonstrate that the proposed architecture can adapt to multiple change detection tasks and achieves leading performance on five datasets, including the binary change dataset LEVIR-CD, the semantic change dataset SECOND, and the building damage assessment dataset xBD.
>
---
#### [new 144] Toward explainable AI approaches for breast imaging: adapting foundation models to diverse populations
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究乳腺影像中的乳腺密度自动分类任务，旨在解决模型在不同人群和成像设备下泛化能力差的问题。通过适配BiomedCLIP基础模型，利用多模态乳腺影像数据进行训练，结合加权对比学习缓解类别不平衡，实现高精度、可解释的分类，验证了模型在多个外部数据集上的强泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.17828v1](https://arxiv.org/pdf/2511.17828v1)**

> **作者:** Guilherme J. Cavalcante; José Gabriel A. Moreira; Gabriel A. B. do Nascimento; Vincent Dong; Alex Nguyen; Thaís G. do Rêgo; Yuri Malheiros; Telmo M. Silva Filho; Carla R. Zeballos Torrez; James C. Gee; Anne Marie McCarthy; Andrew D. A. Maidment; Bruno Barufaldi
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Foundation models hold promise for specialized medical imaging tasks, though their effectiveness in breast imaging remains underexplored. This study leverages BiomedCLIP as a foundation model to address challenges in model generalization. BiomedCLIP was adapted for automated BI-RADS breast density classification using multi-modality mammographic data (synthesized 2D images, digital mammography, and digital breast tomosynthesis). Using 96,995 images, we compared single-modality (s2D only) and multi-modality training approaches, addressing class imbalance through weighted contrastive learning. Both approaches achieved similar accuracy (multi-modality: 0.74, single-modality: 0.73), with the multi-modality model offering broader applicability across different imaging modalities and higher AUC values consistently above 0.84 across BI-RADS categories. External validation on the RSNA and EMBED datasets showed strong generalization capabilities (AUC range: 0.80-0.93). GradCAM visualizations confirmed consistent and clinically relevant attention patterns, highlighting the models interpretability and robustness. This research underscores the potential of foundation models for breast imaging applications, paving the way for future extensions for diagnostic tasks.
>
---
#### [new 145] Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Chain-of-Visual-Thought（COVT）框架，解决视觉语言模型在密集视觉感知（如空间推理、几何意识）上的不足。通过引入少量连续视觉令牌，使模型在推理中融合2D外观、3D几何等多维感知信息，提升感知精度与可解释性，在多个基准上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.19418v1](https://arxiv.org/pdf/2511.19418v1)**

> **作者:** Yiming Qin; Bomin Wei; Jiaxin Ge; Konstantinos Kallidromitis; Stephanie Fu; Trevor Darrell; Xudong Wang
>
> **备注:** Project page: https://wakalsprojectpage.github.io/comt-website/
>
> **摘要:** Vision-Language Models (VLMs) excel at reasoning in linguistic space but struggle with perceptual understanding that requires dense visual perception, e.g., spatial reasoning and geometric awareness. This limitation stems from the fact that current VLMs have limited mechanisms to capture dense visual information across spatial dimensions. We introduce Chain-of-Visual-Thought (COVT), a framework that enables VLMs to reason not only in words but also through continuous visual tokens-compact latent representations that encode rich perceptual cues. Within a small budget of roughly 20 tokens, COVT distills knowledge from lightweight vision experts, capturing complementary properties such as 2D appearance, 3D geometry, spatial layout, and edge structure. During training, the VLM with COVT autoregressively predicts these visual tokens to reconstruct dense supervision signals (e.g., depth, segmentation, edges, and DINO features). At inference, the model reasons directly in the continuous visual token space, preserving efficiency while optionally decoding dense predictions for interpretability. Evaluated across more than ten diverse perception benchmarks, including CV-Bench, MMVP, RealWorldQA, MMStar, WorldMedQA, and HRBench, integrating COVT into strong VLMs such as Qwen2.5-VL and LLaVA consistently improves performance by 3% to 16% and demonstrates that compact continuous visual thinking enables more precise, grounded, and interpretable multimodal intelligence.
>
---
#### [new 146] EVCC: Enhanced Vision Transformer-ConvNeXt-CoAtNet Fusion for Classification
- **分类: cs.CV**

- **简介: 该论文针对图像分类任务，解决现有混合视觉模型计算成本高的问题。提出EVCC多分支架构，融合Vision Transformer、ConvNeXt与CoAtNet，通过自适应令牌剪枝、门控交叉注意力等创新，提升精度并降低35% FLOPs，实现高效准确的特征融合。**

- **链接: [https://arxiv.org/pdf/2511.18691v1](https://arxiv.org/pdf/2511.18691v1)**

> **作者:** Kazi Reyazul Hasan; Md Nafiu Rahman; Wasif Jalal; Sadif Ahmed; Shahriar Raj; Mubasshira Musarrat; Muhammad Abdullah Adnan
>
> **摘要:** Hybrid vision architectures combining Transformers and CNNs have significantly advanced image classification, but they usually do so at significant computational cost. We introduce EVCC (Enhanced Vision Transformer-ConvNeXt-CoAtNet), a novel multi-branch architecture integrating the Vision Transformer, lightweight ConvNeXt, and CoAtNet through key innovations: (1) adaptive token pruning with information preservation, (2) gated bidirectional cross-attention for enhanced feature refinement, (3) auxiliary classification heads for multi-task learning, and (4) a dynamic router gate employing context-aware confidence-driven weighting. Experiments across the CIFAR-100, Tobacco3482, CelebA, and Brain Cancer datasets demonstrate EVCC's superiority over powerful models like DeiT-Base, MaxViT-Base, and CrossViT-Base by consistently achieving state-of-the-art accuracy with improvements of up to 2 percentage points, while reducing FLOPs by 25 to 35%. Our adaptive architecture adjusts computational demands to deployment needs by dynamically reducing token count, efficiently balancing the accuracy-efficiency trade-off while combining global context, local details, and hierarchical features for real-world applications. The source code of our implementation is available at https://anonymous.4open.science/r/EVCC.
>
---
#### [new 147] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DeCo框架，解决像素扩散模型训练与推理慢的问题。通过频率解耦，让DiT专注低频语义，轻量解码器生成高频细节，并引入频域感知流匹配损失，提升效率与质量。在ImageNet上达FID 1.62（256×256），GenEval得分0.86，性能领先。**

- **链接: [https://arxiv.org/pdf/2511.19365v1](https://arxiv.org/pdf/2511.19365v1)**

> **作者:** Zehong Ma; Longhui Wei; Shuai Wang; Shiliang Zhang; Qi Tian
>
> **备注:** Project Page: https://zehong-ma.github.io/DeCo. Code Repository: https://github.com/Zehong-Ma/DeCo
>
> **摘要:** Pixel diffusion aims to generate images directly in pixel space in an end-to-end fashion. This approach avoids the limitations of VAE in the two-stage latent diffusion, offering higher model capacity. Existing pixel diffusion models suffer from slow training and inference, as they usually model both high-frequency signals and low-frequency semantics within a single diffusion transformer (DiT). To pursue a more efficient pixel diffusion paradigm, we propose the frequency-DeCoupled pixel diffusion framework. With the intuition to decouple the generation of high and low frequency components, we leverage a lightweight pixel decoder to generate high-frequency details conditioned on semantic guidance from the DiT. This thus frees the DiT to specialize in modeling low-frequency semantics. In addition, we introduce a frequency-aware flow-matching loss that emphasizes visually salient frequencies while suppressing insignificant ones. Extensive experiments show that DeCo achieves superior performance among pixel diffusion models, attaining FID of 1.62 (256x256) and 2.22 (512x512) on ImageNet, closing the gap with latent diffusion methods. Furthermore, our pretrained text-to-image model achieves a leading overall score of 0.86 on GenEval in system-level comparison. Codes are publicly available at https://github.com/Zehong-Ma/DeCo.
>
---
#### [new 148] ConceptGuard: Proactive Safety in Text-and-Image-to-Video Generation through Multimodal Risk Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本与图像生成视频中的多模态安全风险问题，提出ConceptGuard框架。通过对比检测与语义抑制，实现对潜在危险内容的主动识别与规避。构建了ConceptRisk数据集和T2VSafetyBench-TI2V基准，验证其在风险检测与安全生成上的优越性。**

- **链接: [https://arxiv.org/pdf/2511.18780v1](https://arxiv.org/pdf/2511.18780v1)**

> **作者:** Ruize Ma; Minghong Cai; Yilei Jiang; Jiaming Han; Yi Feng; Yingshui Tan; Xiaoyong Zhu; Bo Zhang; Bo Zheng; Xiangyu Yue
>
> **摘要:** Recent progress in video generative models has enabled the creation of high-quality videos from multimodal prompts that combine text and images. While these systems offer enhanced controllability, they also introduce new safety risks, as harmful content can emerge from individual modalities or their interaction. Existing safety methods are often text-only, require prior knowledge of the risk category, or operate as post-generation auditors, struggling to proactively mitigate such compositional, multimodal risks. To address this challenge, we present ConceptGuard, a unified safeguard framework for proactively detecting and mitigating unsafe semantics in multimodal video generation. ConceptGuard operates in two stages: First, a contrastive detection module identifies latent safety risks by projecting fused image-text inputs into a structured concept space; Second, a semantic suppression mechanism steers the generative process away from unsafe concepts by intervening in the prompt's multimodal conditioning. To support the development and rigorous evaluation of this framework, we introduce two novel benchmarks: ConceptRisk, a large-scale dataset for training on multimodal risks, and T2VSafetyBench-TI2V, the first benchmark adapted from T2VSafetyBench for the Text-and-Image-to-Video (TI2V) safety setting. Comprehensive experiments on both benchmarks show that ConceptGuard consistently outperforms existing baselines, achieving state-of-the-art results in both risk detection and safe video generation.
>
---
#### [new 149] Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization
- **分类: cs.CV**

- **简介: 该论文针对3D人体姿态估计中的长期测试时自适应问题，解决因自监督导致的误差累积。通过运动离散化与潜在运动空间聚类生成锚定动作，结合软重置机制，实现稳定自适应，有效利用个体持续运动特征，提升长期精度。**

- **链接: [https://arxiv.org/pdf/2511.18851v1](https://arxiv.org/pdf/2511.18851v1)**

> **作者:** Yilin Wen; Kechuan Dong; Yusuke Sugano
>
> **备注:** Accepted by AAAI 2026, main track
>
> **摘要:** Online test-time adaptation addresses the train-test domain gap by adapting the model on unlabeled streaming test inputs before making the final prediction. However, online adaptation for 3D human pose estimation suffers from error accumulation when relying on self-supervision with imperfect predictions, leading to degraded performance over time. To mitigate this fundamental challenge, we propose a novel solution that highlights the use of motion discretization. Specifically, we employ unsupervised clustering in the latent motion representation space to derive a set of anchor motions, whose regularity aids in supervising the human pose estimator and enables efficient self-replay. Additionally, we introduce an effective and efficient soft-reset mechanism by reverting the pose estimator to its exponential moving average during continuous adaptation. We examine long-term online adaptation by continuously adapting to out-of-domain streaming test videos of the same individual, which allows for the capture of consistent personal shape and motion traits throughout the streaming observation. By mitigating error accumulation, our solution enables robust exploitation of these personal traits for enhanced accuracy. Experiments demonstrate that our solution outperforms previous online test-time adaptation methods and validate our design choices.
>
---
#### [new 150] Together, Then Apart: Revisiting Multimodal Survival Analysis via a Min-Max Perspective
- **分类: cs.CV**

- **简介: 该论文针对多模态生存分析任务，解决现有方法过度强调模态对齐导致特征坍缩、信息丢失的问题。提出TTA框架，通过“先联合后分离”的双阶段优化：先用共享原型对齐跨模态表示，再以锚点和对比正则化增强模态特异性，实现对齐与差异性的平衡，提升模型性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.18089v1](https://arxiv.org/pdf/2511.18089v1)**

> **作者:** Wenjing Liu; Qin Ren; Wen Zhang; Yuewei Lin; Chenyu You
>
> **摘要:** Integrating heterogeneous modalities such as histopathology and genomics is central to advancing survival analysis, yet most existing methods prioritize cross-modal alignment through attention-based fusion mechanisms, often at the expense of modality-specific characteristics. This overemphasis on alignment leads to representation collapse and reduced diversity. In this work, we revisit multi-modal survival analysis via the dual lens of alignment and distinctiveness, positing that preserving modality-specific structure is as vital as achieving semantic coherence. In this paper, we introduce Together-Then-Apart (TTA), a unified min-max optimization framework that simultaneously models shared and modality-specific representations. The Together stage minimizes semantic discrepancies by aligning embeddings via shared prototypes, guided by an unbalanced optimal transport objective that adaptively highlights informative tokens. The Apart stage maximizes representational diversity through modality anchors and a contrastive regularizer that preserve unique modality information and prevent feature collapse. Extensive experiments on five TCGA benchmarks show that TTA consistently outperforms state-of-the-art methods. Beyond empirical gains, our formulation provides a new theoretical perspective of how alignment and distinctiveness can be jointly achieved in for robust, interpretable, and biologically meaningful multi-modal survival analysis.
>
---
#### [new 151] 4D-VGGT: A General Foundation Model with SpatioTemporal Awareness for Dynamic Scene Geometry Estimation
- **分类: cs.CV**

- **简介: 该论文针对动态场景几何估计任务，解决空间与时间特征异构导致的表征失配问题。提出4D-VGGT模型，通过多设置输入、多层级表征与多任务预测，实现时空解耦建模，提升动态场景几何估计的准确性与通用性。**

- **链接: [https://arxiv.org/pdf/2511.18416v1](https://arxiv.org/pdf/2511.18416v1)**

> **作者:** Haonan Wang; Hanyu Zhou; Haoyue Liu; Luxin Yan
>
> **摘要:** We investigate a challenging task of dynamic scene geometry estimation, which requires representing both spatial and temporal features. Typically, existing methods align the two features into a unified latent space to model scene geometry. However, this unified paradigm suffers from potential mismatched representation due to the heterogeneous nature between spatial and temporal features. In this work, we propose 4D-VGGT, a general foundation model with divide-and-conquer spatiotemporal representation for dynamic scene geometry. Our model is divided into three aspects: 1) Multi-setting input. We design an adaptive visual grid that supports input sequences with arbitrary numbers of views and time steps. 2) Multi-level representation. We propose a cross-view global fusion for spatial representation and a cross-time local fusion for temporal representation. 3) Multi-task prediction. We append multiple task-specific heads to spatiotemporal representations, enabling a comprehensive visual geometry estimation for dynamic scenes. Under this unified framework, these components enhance the feature discriminability and application universality of our model for dynamic scenes. In addition, we integrate multiple geometry datasets to train our model and conduct extensive experiments to verify the effectiveness of our method across various tasks on multiple dynamic scene geometry benchmarks.
>
---
#### [new 152] CellFMCount: A Fluorescence Microscopy Dataset, Benchmark, and Methods for Cell Counting
- **分类: cs.CV**

- **简介: 该论文针对自动化细胞计数任务，解决标注数据少、模型泛化难的问题。构建了包含3023张图像、超43万细胞的大型荧光显微镜数据集，提出SAM-Counter方法，实现高精度计数，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.19351v1](https://arxiv.org/pdf/2511.19351v1)**

> **作者:** Abdurahman Ali Mohammed; Catherine Fonder; Ying Wei; Wallapak Tavanapong; Donald S Sakaguchi; Qi Li; Surya K. Mallapragada
>
> **备注:** The IEEE International Conference on Data Mining (ICDM) 2025
>
> **摘要:** Accurate cell counting is essential in various biomedical research and clinical applications, including cancer diagnosis, stem cell research, and immunology. Manual counting is labor-intensive and error-prone, motivating automation through deep learning techniques. However, training reliable deep learning models requires large amounts of high-quality annotated data, which is difficult and time-consuming to produce manually. Consequently, existing cell-counting datasets are often limited, frequently containing fewer than $500$ images. In this work, we introduce a large-scale annotated dataset comprising $3{,}023$ images from immunocytochemistry experiments related to cellular differentiation, containing over $430{,}000$ manually annotated cell locations. The dataset presents significant challenges: high cell density, overlapping and morphologically diverse cells, a long-tailed distribution of cell count per image, and variation in staining protocols. We benchmark three categories of existing methods: regression-based, crowd-counting, and cell-counting techniques on a test set with cell counts ranging from $10$ to $2{,}126$ cells per image. We also evaluate how the Segment Anything Model (SAM) can be adapted for microscopy cell counting using only dot-annotated datasets. As a case study, we implement a density-map-based adaptation of SAM (SAM-Counter) and report a mean absolute error (MAE) of $22.12$, which outperforms existing approaches (second-best MAE of $27.46$). Our results underscore the value of the dataset and the benchmarking framework for driving progress in automated cell counting and provide a robust foundation for future research and development.
>
---
#### [new 153] MagicWand: A Universal Agent for Generation and Evaluation Aligned with User Preference
- **分类: cs.CV**

- **简介: 该论文针对AIGC中用户偏好对齐难题，提出MagicWand通用生成与评估代理。基于自建的UniPrefer-100K数据集和UniPreferBench基准，通过偏好增强提示、高质量生成及一致性评估，实现内容生成与评价的用户偏好对齐。**

- **链接: [https://arxiv.org/pdf/2511.18352v1](https://arxiv.org/pdf/2511.18352v1)**

> **作者:** Zitong Xu; Dake Shen; Yaosong Du; Kexiang Hao; Jinghan Huang; Xiande Huang
>
> **摘要:** Recent advances in AIGC (Artificial Intelligence Generated Content) models have enabled significant progress in image and video generation. However, users still struggle to obtain content that aligns with their preferences due to the difficulty of crafting detailed prompts and the lack of mechanisms to retain their preferences. To address these challenges, we construct \textbf{UniPrefer-100K}, a large-scale dataset comprising images, videos, and associated text that describes the styles users tend to prefer. Based on UniPrefer-100K, we propose \textbf{MagicWand}, a universal generation and evaluation agent that enhances prompts based on user preferences, leverages advanced generation models for high-quality content, and applies preference-aligned evaluation and refinement. In addition, we introduce \textbf{UniPreferBench}, the first large-scale benchmark with over 120K annotations for assessing user preference alignment across diverse AIGC tasks. Experiments on UniPreferBench demonstrate that MagicWand consistently generates content and evaluations that are well aligned with user preferences across a wide range of scenarios.
>
---
#### [new 154] Early Lung Cancer Diagnosis from Virtual Follow-up LDCT Generation via Correlational Autoencoder and Latent Flow Matching
- **分类: cs.CV**

- **简介: 该论文属于早期肺癌诊断任务，旨在通过生成虚拟一年后随访CT，提前识别恶性结节。提出CorrFlowNet模型，利用相关性自编码器与潜在空间流匹配，生成虚拟随访图像，提升早期诊断准确性，减少对真实随访的依赖。**

- **链接: [https://arxiv.org/pdf/2511.18185v1](https://arxiv.org/pdf/2511.18185v1)**

> **作者:** Yutong Wu; Yifan Wang; Qining Zhang; Chuan Zhou; Lei Ying
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Lung cancer is one of the most commonly diagnosed cancers, and early diagnosis is critical because the survival rate declines sharply once the disease progresses to advanced stages. However, achieving an early diagnosis remains challenging, particularly in distinguishing subtle early signals of malignancy from those of benign conditions. In clinical practice, a patient with a high risk may need to undergo an initial baseline and several annual follow-up examinations (e.g., CT scans) before receiving a definitive diagnosis, which can result in missing the optimal treatment. Recently, Artificial Intelligence (AI) methods have been increasingly used for early diagnosis of lung cancer, but most existing algorithms focus on radiomic features extraction from single early-stage CT scans. Inspired by recent advances in diffusion models for image generation, this paper proposes a generative method, named CorrFlowNet, which creates a virtual, one-year follow-up CT scan after the initial baseline scan. This virtual follow-up would allow for an early detection of malignant/benign nodules, reducing the need to wait for clinical follow-ups. During training, our approach employs a correlational autoencoder to encode both early baseline and follow-up CT images into a latent space that captures the dynamics of nodule progression as well as the correlations between them, followed by a flow matching algorithm on the latent space with a neural ordinary differential equation. An auxiliary classifier is used to further enhance the diagnostic accuracy. Evaluations on a real clinical dataset show our method can significantly improve downstream lung nodule risk assessment compared with existing baseline models. Moreover, its diagnostic accuracy is comparable with real clinical CT follow-ups, highlighting its potential to improve cancer diagnosis.
>
---
#### [new 155] Life-IQA: Boosting Blind Image Quality Assessment through GCN-enhanced Layer Interaction and MoE-based Feature Decoupling
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对盲图像质量评估（BIQA）中浅层与深层特征贡献不均、解码架构效率低的问题，提出Life-IQA框架。通过图卷积网络增强的层间交互和基于专家混合（MoE）的特征解耦，有效融合多层级特征并提升对不同失真类型的建模能力，显著提升评估精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.19024v1](https://arxiv.org/pdf/2511.19024v1)**

> **作者:** Long Tang; Guoquan Zhen; Jie Hao; Jianbo Zhang; Huiyu Duan; Liang Yuan; Guangtao Zhai
>
> **摘要:** Blind image quality assessment (BIQA) plays a crucial role in evaluating and optimizing visual experience. Most existing BIQA approaches fuse shallow and deep features extracted from backbone networks, while overlooking the unequal contributions to quality prediction. Moreover, while various vision encoder backbones are widely adopted in BIQA, the effective quality decoding architectures remain underexplored. To address these limitations, this paper investigates the contributions of shallow and deep features to BIQA, and proposes a effective quality feature decoding framework via GCN-enhanced \underline{l}ayer\underline{i}nteraction and MoE-based \underline{f}eature d\underline{e}coupling, termed \textbf{(Life-IQA)}. Specifically, the GCN-enhanced layer interaction module utilizes the GCN-enhanced deepest-layer features as query and the penultimate-layer features as key, value, then performs cross-attention to achieve feature interaction. Moreover, a MoE-based feature decoupling module is proposed to decouple fused representations though different experts specialized for specific distortion types or quality dimensions. Extensive experiments demonstrate that Life-IQA shows more favorable balance between accuracy and cost than a vanilla Transformer decoder and achieves state-of-the-art performance on multiple BIQA benchmarks.The code is available at: \href{https://github.com/TANGLONG2/Life-IQA/tree/main}{\texttt{Life-IQA}}.
>
---
#### [new 156] CLASH: A Benchmark for Cross-Modal Contradiction Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出CLASH基准，针对多模态矛盾检测任务，解决现有数据集忽略跨模态不一致的问题。通过含可控对象/属性矛盾的图文对，评估模型在多选与开放问答中的矛盾识别能力，揭示主流模型的系统性偏差，并证明针对性微调可显著提升检测性能。**

- **链接: [https://arxiv.org/pdf/2511.19199v1](https://arxiv.org/pdf/2511.19199v1)**

> **作者:** Teodora Popordanoska; Jiameng Li; Matthew B. Blaschko
>
> **备注:** First two authors contributed equally
>
> **摘要:** Contradictory multimodal inputs are common in real-world settings, yet existing benchmarks typically assume input consistency and fail to evaluate cross-modal contradiction detection - a fundamental capability for preventing hallucinations and ensuring reliability. We introduce CLASH, a novel benchmark for multimodal contradiction detection, featuring COCO images paired with contradictory captions containing controlled object-level or attribute-level contradictions. The samples include targeted questions evaluated in both multiple-choice and open-ended formats. The benchmark provides an extensive fine-tuning set filtered through automated quality checks, alongside a smaller human-verified diagnostic set. Our analysis of state-of-the-art models reveals substantial limitations in recognizing cross-modal conflicts, exposing systematic modality biases and category-specific weaknesses. Furthermore, we empirically demonstrate that targeted fine-tuning on CLASH substantially enhances conflict detection capabilities.
>
---
#### [new 157] EventBench: Towards Comprehensive Benchmarking of Event-based MLLMs
- **分类: cs.CV**

- **简介: 该论文提出EventBench，一个面向事件基多模态大模型的综合性基准。针对现有评估缺乏统一标准、任务覆盖不足等问题，构建了包含八项任务、百万级数据对的开放数据集，涵盖理解、识别与三维空间推理，推动事件流处理模型的全面评测。**

- **链接: [https://arxiv.org/pdf/2511.18448v1](https://arxiv.org/pdf/2511.18448v1)**

> **作者:** Shaoyu Liu; Jianing Li; Guanghui Zhao; Yunjian Zhang; Xiangyang Ji
>
> **摘要:** Multimodal large language models (MLLMs) have made significant advancements in event-based vision, yet the comprehensive evaluation of their capabilities within a unified benchmark remains largely unexplored. In this work, we introduce EventBench, a benchmark that offers eight diverse task metrics together with a large-scale event stream dataset. EventBench differs from existing event-based benchmarks in four key aspects: (1) openness in accessibility, releasing all raw event streams and task instructions across eight evaluation metrics; (2) diversity in task coverage, spanning understanding, recognition, and spatial reasoning tasks for comprehensive capability assessment; (3) integration in spatial dimensions, pioneering the design of 3D spatial reasoning tasks for event-based MLLMs; and (4) scale in data volume, with an accompanying training set of over one million event-text pairs supporting large-scale training and evaluation. Using EventBench, we evaluate state-of-the-art closed-source models such as GPT-5 and Gemini-2.5 Pro, leading open-source models including Qwen2.5-VL and InternVL3, and event-based MLLMs such as EventGPT that directly process raw event streams. Extensive evaluation reveals that while current event-based MLLMs demonstrate strong performance in event stream understanding, they continue to struggle with fine-grained recognition and spatial reasoning.
>
---
#### [new 158] FeRA: Frequency-Energy Constrained Routing for Effective Diffusion Adaptation Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文针对扩散模型在新任务上微调效率低的问题，提出FeRA框架。通过分析去噪过程中的频率能量机制，设计频率感知的参数更新策略，包含频率指示器、软路由和一致性正则化，实现高效稳定微调，适用于多种模型与分辨率。**

- **链接: [https://arxiv.org/pdf/2511.17979v1](https://arxiv.org/pdf/2511.17979v1)**

> **作者:** Bo Yin; Xiaobin Hu; Xingyu Zhou; Peng-Tao Jiang; Yue Liao; Junwei Zhu; Jiangning Zhang; Ying Tai; Chengjie Wang; Shuicheng Yan
>
> **摘要:** Diffusion models have achieved remarkable success in generative modeling, yet how to effectively adapt large pretrained models to new tasks remains challenging. We revisit the reconstruction behavior of diffusion models during denoising to unveil the underlying frequency energy mechanism governing this process. Building upon this observation, we propose FeRA, a frequency driven fine tuning framework that aligns parameter updates with the intrinsic frequency energy progression of diffusion. FeRA establishes a comprehensive frequency energy framework for effective diffusion adaptation fine tuning, comprising three synergistic components: (i) a compact frequency energy indicator that characterizes the latent bandwise energy distribution, (ii) a soft frequency router that adaptively fuses multiple frequency specific adapter experts, and (iii) a frequency energy consistency regularization that stabilizes diffusion optimization and ensures coherent adaptation across bands. Routing operates in both training and inference, with inference time routing dynamically determined by the latent frequency energy. It integrates seamlessly with adapter based tuning schemes and generalizes well across diffusion backbones and resolutions. By aligning adaptation with the frequency energy mechanism, FeRA provides a simple, stable, and compatible paradigm for effective and robust diffusion model adaptation.
>
---
#### [new 159] Understanding Task Transfer in Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究视觉语言模型（VLMs）在视觉感知任务中的任务迁移问题。针对微调后性能不可预测的问题，提出Perfection Gap Factor（PGF）度量迁移效果，构建任务迁移图，揭示任务间正负迁移规律，识别任务群组与“人格”模式，指导高效数据选择，为提升VLMs的泛化能力提供可操作方案。**

- **链接: [https://arxiv.org/pdf/2511.18787v1](https://arxiv.org/pdf/2511.18787v1)**

> **作者:** Bhuvan Sachdeva; Karan Uppal; Abhinav Java; Vineeth N. Balasubramanian
>
> **摘要:** Vision-Language Models (VLMs) perform well on multimodal benchmarks but lag behind humans and specialized models on visual perception tasks like depth estimation or object counting. Finetuning on one task can unpredictably affect performance on others, making task-specific finetuning challenging. In this paper, we address this challenge through a systematic study of task transferability. We examine how finetuning a VLM on one perception task affects its zero-shot performance on others. To quantify these effects, we introduce Perfection Gap Factor (PGF), a metric that captures both the breadth and magnitude of transfer. Using three open-weight VLMs evaluated across 13 perception tasks, we construct a task-transfer graph that reveals previously unobserved relationships among perception tasks. Our analysis uncovers patterns of positive and negative transfer, identifies groups of tasks that mutually influence each other, organizes tasks into personas based on their transfer behavior and demonstrates how PGF can guide data selection for more efficient training. These findings highlight both opportunities for positive transfer and risks of negative interference, offering actionable guidance for advancing VLMs.
>
---
#### [new 160] ArticFlow: Generative Simulation of Articulated Mechanisms
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ArticFlow，一种用于生成可动机械结构的两阶段流匹配框架。针对动作依赖变形与数据稀缺难题，通过联合隐空间流与点流，实现动作可控的高质量生成与仿真，支持形态插值与跨动作泛化，在MuJoCo上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17883v1](https://arxiv.org/pdf/2511.17883v1)**

> **作者:** Jiong Lin; Jinchen Ruan; Hod Lipson
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Recent advances in generative models have produced strong results for static 3D shapes, whereas articulated 3D generation remains challenging due to action-dependent deformations and limited datasets. We introduce ArticFlow, a two-stage flow matching framework that learns a controllable velocity field from noise to target point sets under explicit action control. ArticFlow couples (i) a latent flow that transports noise to a shape-prior code and (ii) a point flow that transports points conditioned on the action and the shape prior, enabling a single model to represent diverse articulated categories and generalize across actions. On MuJoCo Menagerie, ArticFlow functions both as a generative model and as a neural simulator: it predicts action-conditioned kinematics from a compact prior and synthesizes novel morphologies via latent interpolation. Compared with object-specific simulators and an action-conditioned variant of static point-cloud generators, ArticFlow achieves higher kinematic accuracy and better shape quality. Results show that action-conditioned flow matching is a practical route to controllable and high-quality articulated mechanism generation.
>
---
#### [new 161] Bias Is a Subspace, Not a Coordinate: A Geometric Rethinking of Post-hoc Debiasing in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究视觉语言模型中的后验去偏问题。针对现有方法仅替换特定坐标导致去偏不彻底的问题，提出基于几何视角的子空间投影去偏（SPD）框架，识别并移除线性可解的偏见子空间，有效提升公平性且保持任务性能。**

- **链接: [https://arxiv.org/pdf/2511.18123v1](https://arxiv.org/pdf/2511.18123v1)**

> **作者:** Dachuan Zhao; Weiyue Li; Zhenda Shen; Yushu Qiu; Bowen Xu; Haoyu Chen; Yongchao Chen
>
> **摘要:** Vision-Language Models (VLMs) have become indispensable for multimodal reasoning, yet their representations often encode and amplify demographic biases, resulting in biased associations and misaligned predictions in downstream tasks. Such behavior undermines fairness and distorts the intended alignment between vision and language. Recent post-hoc approaches attempt to mitigate bias by replacing the most attribute-correlated embedding coordinates with neutral values. However, our systematic analysis reveals three critical failures of this coordinate-wise approach: feature entanglement, poor cross-dataset generalization, and incomplete bias removal. We find that bias is not localized to a few coordinates but is instead distributed across a few linear subspaces. To address these limitations, we propose $\textbf{S}$ubspace $\textbf{P}$rojection $\textbf{D}$ebiasing ($\textbf{SPD}$), a geometrically principled framework that identifies and removes the entire subspace of linearly decodable bias while reinserting a neutral mean component to preserve semantic fidelity. Extensive experiments across zero-shot classification, text-to-image retrieval, and image generation validate the effectiveness of SPD: our method achieves more robust debiasing with an average improvement of $18.5\%$ across four fairness metrics, while maintaining minimal loss in task performance compared to the best debiasing baseline.
>
---
#### [new 162] C3Po: Cross-View Cross-Modality Correspondence by Pointmap Prediction
- **分类: cs.CV**

- **简介: 该论文针对跨视图、跨模态几何对应问题，聚焦地面照片与平面图的匹配。针对现有数据集在模态多样性与对应标注上的不足，构建了包含90K图像-平面图对的C3数据集。通过在该数据集上训练，显著提升对应预测性能，推动跨模态几何理解发展。**

- **链接: [https://arxiv.org/pdf/2511.18559v1](https://arxiv.org/pdf/2511.18559v1)**

> **作者:** Kuan Wei Huang; Brandon Li; Bharath Hariharan; Noah Snavely
>
> **备注:** NeurIPS 2025
>
> **摘要:** Geometric models like DUSt3R have shown great advances in understanding the geometry of a scene from pairs of photos. However, they fail when the inputs are from vastly different viewpoints (e.g., aerial vs. ground) or modalities (e.g., photos vs. abstract drawings) compared to what was observed during training. This paper addresses a challenging version of this problem: predicting correspondences between ground-level photos and floor plans. Current datasets for joint photo--floor plan reasoning are limited, either lacking in varying modalities (VIGOR) or lacking in correspondences (WAFFLE). To address these limitations, we introduce a new dataset, C3, created by first reconstructing a number of scenes in 3D from Internet photo collections via structure-from-motion, then manually registering the reconstructions to floor plans gathered from the Internet, from which we can derive correspondence between images and floor plans. C3 contains 90K paired floor plans and photos across 597 scenes with 153M pixel-level correspondences and 85K camera poses. We find that state-of-the-art correspondence models struggle on this task. By training on our new data, we can improve on the best performing method by 34% in RMSE. We also identify open challenges in cross-modal geometric reasoning that our dataset aims to help address.
>
---
#### [new 163] MagicWorld: Interactive Geometry-driven Video World Exploration
- **分类: cs.CV**

- **简介: 该论文提出MagicWorld，一种交互式视频世界生成模型，旨在解决现有方法在视角变换下的结构不稳与多步交互中的历史信息遗忘问题。通过引入基于动作的3D几何模块（AG3D）和历史缓存检索机制（HCR），增强场景几何一致性与连续性，实现更稳定的交互式视频生成。**

- **链接: [https://arxiv.org/pdf/2511.18886v1](https://arxiv.org/pdf/2511.18886v1)**

> **作者:** Guangyuan Li; Siming Zheng; Shuolin Xu; Jinwei Chen; Bo Li; Xiaobin Hu; Lei Zhao; Peng-Tao Jiang
>
> **摘要:** Recent interactive video world model methods generate scene evolution conditioned on user instructions. Although they achieve impressive results, two key limitations remain. First, they fail to fully exploit the correspondence between instruction-driven scene motion and the underlying 3D geometry, which results in structural instability under viewpoint changes. Second, they easily forget historical information during multi-step interaction, resulting in error accumulation and progressive drift in scene semantics and structure. To address these issues, we propose MagicWorld, an interactive video world model that integrates 3D geometric priors and historical retrieval. MagicWorld starts from a single scene image, employs user actions to drive dynamic scene evolution, and autoregressively synthesizes continuous scenes. We introduce the Action-Guided 3D Geometry Module (AG3D), which constructs a point cloud from the first frame of each interaction and the corresponding action, providing explicit geometric constraints for viewpoint transitions and thereby improving structural consistency. We further propose History Cache Retrieval (HCR) mechanism, which retrieves relevant historical frames during generation and injects them as conditioning signals, helping the model utilize past scene information and mitigate error accumulation. Experimental results demonstrate that MagicWorld achieves notable improvements in scene stability and continuity across interaction iterations.
>
---
#### [new 164] Plan-X: Instruct Video Generation via Semantic Planning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Plan-X框架，解决视频生成中因缺乏高层语义规划导致的视觉幻觉与指令偏离问题。通过引入可学习的多模态语义规划器，自回归生成时序语义标记，作为视频扩散模型的结构化指导，实现精准、连贯的指令对齐视频生成。**

- **链接: [https://arxiv.org/pdf/2511.17986v1](https://arxiv.org/pdf/2511.17986v1)**

> **作者:** Lun Huang; You Xie; Hongyi Xu; Tianpei Gu; Chenxu Zhang; Guoxian Song; Zenan Li; Xiaochen Zhao; Linjie Luo; Guillermo Sapiro
>
> **备注:** The project page is at https://byteaigc.github.io/Plan-X
>
> **摘要:** Diffusion Transformers have demonstrated remarkable capabilities in visual synthesis, yet they often struggle with high-level semantic reasoning and long-horizon planning. This limitation frequently leads to visual hallucinations and mis-alignments with user instructions, especially in scenarios involving complex scene understanding, human-object interactions, multi-stage actions, and in-context motion reasoning. To address these challenges, we propose Plan-X, a framework that explicitly enforces high-level semantic planning to instruct video generation process. At its core lies a Semantic Planner, a learnable multimodal language model that reasons over the user's intent from both text prompts and visual context, and autoregressively generates a sequence of text-grounded spatio-temporal semantic tokens. These semantic tokens, complementary to high-level text prompt guidance, serve as structured "semantic sketches" over time for the video diffusion model, which has its strength at synthesizing high-fidelity visual details. Plan-X effectively integrates the strength of language models in multimodal in-context reasoning and planning, together with the strength of diffusion models in photorealistic video synthesis. Extensive experiments demonstrate that our framework substantially reduces visual hallucinations and enables fine-grained, instruction-aligned video generation consistent with multimodal context.
>
---
#### [new 165] Learning Plug-and-play Memory for Guiding Video Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视频生成中物理规律违背的问题，提出可即插即用的记忆模块DiT-Mem。通过3D CNN与滤波器提取参考视频的语义与物理特征，生成记忆令牌注入DiT模型，仅训练少量参数即可提升生成视频的物理合理性与质量，实现高效、灵活的常识引导。**

- **链接: [https://arxiv.org/pdf/2511.19229v1](https://arxiv.org/pdf/2511.19229v1)**

> **作者:** Selena Song; Ziming Xu; Zijun Zhang; Kun Zhou; Jiaxian Guo; Lianhui Qin; Biwei Huang
>
> **摘要:** Diffusion Transformer(DiT) based video generation models have recently achieved impressive visual quality and temporal coherence, but they still frequently violate basic physical laws and commonsense dynamics, revealing a lack of explicit world knowledge. In this work, we explore how to equip them with a plug-and-play memory that injects useful world knowledge. Motivated by in-context memory in Transformer-based LLMs, we conduct empirical studies to show that DiT can be steered via interventions on its hidden states, and simple low-pass and high-pass filters in the embedding space naturally disentangle low-level appearance and high-level physical/semantic cues, enabling targeted guidance. Building on these observations, we propose a learnable memory encoder DiT-Mem, composed of stacked 3D CNNs, low-/high-pass filters, and self-attention layers. The encoder maps reference videos into a compact set of memory tokens, which are concatenated as the memory within the DiT self-attention layers. During training, we keep the diffusion backbone frozen, and only optimize the memory encoder. It yields a rather efficient training process on few training parameters (150M) and 10K data samples, and enables plug-and-play usage at inference time. Extensive experiments on state-of-the-art models demonstrate the effectiveness of our method in improving physical rule following and video fidelity. Our code and data are publicly released here: https://thrcle421.github.io/DiT-Mem-Web/.
>
---
#### [new 166] Vision-Motion-Reference Alignment for Referring Multi-Object Tracking via Multi-Modal Large Language Models
- **分类: cs.CV**

- **简介: 该论文针对引用多目标跟踪（RMOT）中语言参考与动态运动不匹配的问题，提出VMRMOT框架。通过引入基于物体动态行为的运动模态，利用多模态大模型增强视觉、运动与语言三者的对齐，设计了层次化对齐模块与运动引导预测头，显著提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2511.17681v1](https://arxiv.org/pdf/2511.17681v1)**

> **作者:** Weiyi Lv; Ning Zhang; Hanyang Sun; Haoran Jiang; Kai Zhao; Jing Xiao; Dan Zeng
>
> **摘要:** Referring Multi-Object Tracking (RMOT) extends conventional multi-object tracking (MOT) by introducing natural language references for multi-modal fusion tracking. RMOT benchmarks only describe the object's appearance, relative positions, and initial motion states. This so-called static regulation fails to capture dynamic changes of the object motion, including velocity changes and motion direction shifts. This limitation not only causes a temporal discrepancy between static references and dynamic vision modality but also constrains multi-modal tracking performance. To address this limitation, we propose a novel Vision-Motion-Reference aligned RMOT framework, named VMRMOT. It integrates a motion modality extracted from object dynamics to enhance the alignment between vision modality and language references through multi-modal large language models (MLLMs). Specifically, we introduce motion-aware descriptions derived from object dynamic behaviors and, leveraging the powerful temporal-reasoning capabilities of MLLMs, extract motion features as the motion modality. We further design a Vision-Motion-Reference Alignment (VMRA) module to hierarchically align visual queries with motion and reference cues, enhancing their cross-modal consistency. In addition, a Motion-Guided Prediction Head (MGPH) is developed to explore motion modality to enhance the performance of the prediction head. To the best of our knowledge, VMRMOT is the first approach to employ MLLMs in the RMOT task for vision-reference alignment. Extensive experiments on multiple RMOT benchmarks demonstrate that VMRMOT outperforms existing state-of-the-art methods.
>
---
#### [new 167] A Self-Conditioned Representation Guided Diffusion Model for Realistic Text-to-LiDAR Scene Generation
- **分类: cs.CV**

- **简介: 该论文针对文本到LiDAR场景生成任务，解决因数据稀缺和文本质量差导致的生成场景过于平滑、细节不足问题。提出T2LDM模型，引入自条件表示引导（SCRG）增强几何结构感知，并构建T2nuScenes基准与可控性度量，支持多模态生成，显著提升生成质量和可控性。**

- **链接: [https://arxiv.org/pdf/2511.19004v1](https://arxiv.org/pdf/2511.19004v1)**

> **作者:** Wentao Qu; Guofeng Mei; Yang Wu; Yongshun Gong; Xiaoshui Huang; Liang Xiao
>
> **摘要:** Text-to-LiDAR generation can customize 3D data with rich structures and diverse scenes for downstream tasks. However, the scarcity of Text-LiDAR pairs often causes insufficient training priors, generating overly smooth 3D scenes. Moreover, low-quality text descriptions may degrade generation quality and controllability. In this paper, we propose a Text-to-LiDAR Diffusion Model for scene generation, named T2LDM, with a Self-Conditioned Representation Guidance (SCRG). Specifically, SCRG, by aligning to the real representations, provides the soft supervision with reconstruction details for the Denoising Network (DN) in training, while decoupled in inference. In this way, T2LDM can perceive rich geometric structures from data distribution, generating detailed objects in scenes. Meanwhile, we construct a content-composable Text-LiDAR benchmark, T2nuScenes, along with a controllability metric. Based on this, we analyze the effects of different text prompts for LiDAR generation quality and controllability, providing practical prompt paradigms and insights. Furthermore, a directional position prior is designed to mitigate street distortion, further improving scene fidelity. Additionally, by learning a conditional encoder via frozen DN, T2LDM can support multiple conditional tasks, including Sparse-to-Dense, Dense-to-Sparse, and Semantic-to-LiDAR generation. Extensive experiments in unconditional and conditional generation demonstrate that T2LDM outperforms existing methods, achieving state-of-the-art scene generation.
>
---
#### [new 168] A Novel Dual-Stream Framework for dMRI Tractography Streamline Classification with Joint dMRI and fMRI Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对dMRI tractography中难以区分功能相似路径的问题，提出一种融合dMRI与fMRI的双流框架。通过结合轨迹几何特征与纤维终点功能信号，实现更精准的白质束功能分区，显著提升分类性能。**

- **链接: [https://arxiv.org/pdf/2511.18781v1](https://arxiv.org/pdf/2511.18781v1)**

> **作者:** Haotian Yan; Bocheng Guo; Jianzhong He; Nir A. Sochen; Ofer Pasternak; Lauren J O'Donnell; Fan Zhang
>
> **备注:** Submitted to ISBI 2026, 7 pages, 2 figures
>
> **摘要:** Streamline classification is essential to identify anatomically meaningful white matter tracts from diffusion MRI (dMRI) tractography. However, current streamline classification methods rely primarily on the geometric features of the streamline trajectory, failing to distinguish between functionally distinct fiber tracts with similar pathways. To address this, we introduce a novel dual-stream streamline classification framework that jointly analyzes dMRI and functional MRI (fMRI) data to enhance the functional coherence of tract parcellation. We design a novel network that performs streamline classification using a pretrained backbone model for full streamline trajectories, while augmenting with an auxiliary network that processes fMRI signals from fiber endpoint regions. We demonstrate our method by parcellating the corticospinal tract (CST) into its four somatotopic subdivisions. Experimental results from ablation studies and comparisons with state-of-the-art methods demonstrate our approach's superior performance.
>
---
#### [new 169] Disc3D: Automatic Curation of High-Quality 3D Dialog Data via Discriminative Object Referring
- **分类: cs.CV**

- **简介: 该论文针对3D多模态大模型缺乏高质量对话数据的问题，提出全自动的Disc3D管道。通过消除视角与指代歧义，生成250万+高质量3D场景对话数据，涵盖多任务问答与视觉定位，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.18817v1](https://arxiv.org/pdf/2511.18817v1)**

> **作者:** Siyuan Wei; Chunjie Wang; Xiao Liu; Xiaosheng Yan; Zhishan Zhou; Rui Huang
>
> **备注:** 8 pages
>
> **摘要:** 3D Multi-modal Large Language Models (MLLMs) still lag behind their 2D peers, largely because large-scale, high-quality 3D scene-dialogue datasets remain scarce. Prior efforts hinge on expensive human annotation and leave two key ambiguities unresolved: viewpoint ambiguity, where spatial language presumes unknown camera poses, and object referring ambiguity, where non-exclusive descriptions blur the line between targets and distractors. We therefore present a fully automated pipeline that converts raw 3D scans into unambiguous, high-quality dialogue data at a fraction of the previous cost. By synergizing rule-based constraints with 2D MLLMs and LLMs, the pipeline enables controllable, scalable generation without human intervention. The pipeline comprises four stages: (1) meta-annotation collection harvesting object-, frame-, and scene-level captions, (2) scene graph construction with relation correction to capture proximal object relations, (3) discriminative object referring that generates exclusive and compact descriptions, and (4) multi-task data generation synthesizing diverse dialogues. Our pipeline systematically mitigates inherent flaws in source datasets and produces the final Disc3D dataset, over 2 million samples in 25K hybrid 3D scenes, spanning scene, view, and object captioning, visual grounding, and five object-centric QA tasks. Extensive experiments demonstrate that training with Disc3D yields consistent, significant improvements on both public benchmarks and our multifaceted Disc3D-QA tasks. Code, data, and models will be publicly available.
>
---
#### [new 170] Three-Dimensional Anatomical Data Generation Based on Artificial Neural Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对手术规划中3D anatomical数据获取难的问题，提出基于神经网络的自动化生成方法。利用生物仿生水凝胶前列腺模型与定制超声扫描，结合3D GAN生成多样化的3D模型，实现高精度图像分割与三维重建，解决真实数据获取受限及软组织成像困难问题。**

- **链接: [https://arxiv.org/pdf/2511.19198v1](https://arxiv.org/pdf/2511.19198v1)**

> **作者:** Ann-Sophia Müller; Moonkwang Jeong; Meng Zhang; Jiyuan Tian; Arkadiusz Miernik; Stefanie Speidel; Tian Qiu
>
> **备注:** 6 pages, 4 figures, 1 table, IEEE International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Surgical planning and training based on machine learning requires a large amount of 3D anatomical models reconstructed from medical imaging, which is currently one of the major bottlenecks. Obtaining these data from real patients and during surgery is very demanding, if even possible, due to legal, ethical, and technical challenges. It is especially difficult for soft tissue organs with poor imaging contrast, such as the prostate. To overcome these challenges, we present a novel workflow for automated 3D anatomical data generation using data obtained from physical organ models. We additionally use a 3D Generative Adversarial Network (GAN) to obtain a manifold of 3D models useful for other downstream machine learning tasks that rely on 3D data. We demonstrate our workflow using an artificial prostate model made of biomimetic hydrogels with imaging contrast in multiple zones. This is used to physically simulate endoscopic surgery. For evaluation and 3D data generation, we place it into a customized ultrasound scanner that records the prostate before and after the procedure. A neural network is trained to segment the recorded ultrasound images, which outperforms conventional, non-learning-based computer vision techniques in terms of intersection over union (IoU). Based on the segmentations, a 3D mesh model is reconstructed, and performance feedback is provided.
>
---
#### [new 171] SegSplat: Feed-forward Gaussian Splatting and Open-Set Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文提出SegSplat，融合3D高斯泼溅与开放词汇语义分割，实现快速3D重建与语义理解。通过单次前向传播生成几何、外观及语义索引，构建紧凑语义记忆库，无需场景优化即可支持可查询语义，解决实时语义3D场景构建难题，推动机器人与AR应用发展。**

- **链接: [https://arxiv.org/pdf/2511.18386v1](https://arxiv.org/pdf/2511.18386v1)**

> **作者:** Peter Siegel; Federico Tombari; Marc Pollefeys; Daniel Barath
>
> **摘要:** We have introduced SegSplat, a novel framework designed to bridge the gap between rapid, feed-forward 3D reconstruction and rich, open-vocabulary semantic understanding. By constructing a compact semantic memory bank from multi-view 2D foundation model features and predicting discrete semantic indices alongside geometric and appearance attributes for each 3D Gaussian in a single pass, SegSplat efficiently imbues scenes with queryable semantics. Our experiments demonstrate that SegSplat achieves geometric fidelity comparable to state-of-the-art feed-forward 3D Gaussian Splatting methods while simultaneously enabling robust open-set semantic segmentation, crucially \textit{without} requiring any per-scene optimization for semantic feature integration. This work represents a significant step towards practical, on-the-fly generation of semantically aware 3D environments, vital for advancing robotic interaction, augmented reality, and other intelligent systems.
>
---
#### [new 172] Spotlight: Identifying and Localizing Video Generation Errors Using VLMs
- **分类: cs.CV**

- **简介: 该论文提出Spotlight任务，旨在定位与解释文本生成视频中的细粒度错误。针对现有评估方法无法精准识别错误位置与类型的问题，研究构建了包含1600个标注错误的数据集，分析错误分布，并评估视觉语言模型在错误识别上的不足，提出改进策略，推动视频生成的精细评估与优化。**

- **链接: [https://arxiv.org/pdf/2511.18102v1](https://arxiv.org/pdf/2511.18102v1)**

> **作者:** Aditya Chinchure; Sahithya Ravi; Pushkar Shukla; Vered Shwartz; Leonid Sigal
>
> **摘要:** Current text-to-video models (T2V) can generate high-quality, temporally coherent, and visually realistic videos. Nonetheless, errors still often occur, and are more nuanced and local compared to the previous generation of T2V models. While current evaluation paradigms assess video models across diverse dimensions, they typically evaluate videos holistically without identifying when specific errors occur or describing their nature. We address this gap by introducing Spotlight, a novel task aimed at localizing and explaining video-generation errors. We generate 600 videos using 200 diverse textual prompts and three state-of-the-art video generators (Veo 3, Seedance, and LTX-2), and annotate over 1600 fine-grained errors across six types, including motion, physics, and prompt adherence. We observe that adherence and physics errors are predominant and persist across longer segments, whereas appearance-disappearance and body pose errors manifest in shorter segments. We then evaluate current VLMs on Spotlight and find that VLMs lag significantly behind humans in error identification and localization in videos. We propose inference-time strategies to probe the limits of current VLMs on our task, improving performance by nearly 2x. Our task paves a way forward to building fine-grained evaluation tools and more sophisticated reward models for video generators.
>
---
#### [new 173] PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态人脸反欺骗（FAS）中的可解释性与泛化能力问题，提出PA-FAS方法。通过构建扩展推理路径和答案洗牌机制，增强多模态推理深度，缓解监督信号与推理路径不匹配导致的捷径学习，提升模型在少标注场景下的性能与可信度。**

- **链接: [https://arxiv.org/pdf/2511.17927v1](https://arxiv.org/pdf/2511.17927v1)**

> **作者:** Yingjie Ma; Xun Lin; Yong Xu; Weicheng Xie; Zitong Yu
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Face anti-spoofing (FAS) has recently advanced in multimodal fusion, cross-domain generalization, and interpretability. With large language models and reinforcement learning (RL), strategy-based training offers new opportunities to jointly model these aspects. However, multimodal reasoning is more complex than unimodal reasoning, requiring accurate feature representation and cross-modal verification while facing scarce, high-quality annotations, which makes direct application of RL sub-optimal. We identify two key limitations of supervised fine-tuning plus RL (SFT+RL) for multimodal FAS: (1) limited multimodal reasoning paths restrict the use of complementary modalities and shrink the exploration space after SFT, weakening the effect of RL; and (2) mismatched single-task supervision versus diverse reasoning paths causes reasoning confusion, where models may exploit shortcuts by mapping images directly to answers and ignoring the intended reasoning. To address this, we propose PA-FAS, which enhances reasoning paths by constructing high-quality extended reasoning sequences from limited annotations, enriching paths and relaxing exploration constraints. We further introduce an answer-shuffling mechanism during SFT to force comprehensive multimodal analysis instead of using superficial cues, thereby encouraging deeper reasoning and mitigating shortcut learning. PA-FAS significantly improves multimodal reasoning accuracy and cross-domain generalization, and better unifies multimodal fusion, generalization, and interpretability for trustworthy FAS.
>
---
#### [new 174] GuideFlow: Constraint-Guided Flow Matching for Planning in End-to-End Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文针对端到端自动驾驶中的规划任务，解决模仿学习导致的轨迹模式坍塌及生成模型难以直接融入安全约束的问题。提出GuideFlow框架，通过显式约束引导的流匹配机制，在生成过程中直接融入物理与安全约束，并联合训练能量模型提升优化能力，同时支持驾驶风格可控调节，显著提升规划多样性与安全性。**

- **链接: [https://arxiv.org/pdf/2511.18729v1](https://arxiv.org/pdf/2511.18729v1)**

> **作者:** Lin Liu; Caiyan Jia; Guanyi Yu; Ziying Song; JunQiao Li; Feiyang Jia; Peiliang Wu; Xiaoshuai Hao; Yandan Luo
>
> **摘要:** Driving planning is a critical component of end-to-end (E2E) autonomous driving. However, prevailing Imitative E2E Planners often suffer from multimodal trajectory mode collapse, failing to produce diverse trajectory proposals. Meanwhile, Generative E2E Planners struggle to incorporate crucial safety and physical constraints directly into the generative process, necessitating an additional optimization stage to refine their outputs. In this paper, we propose \textit{\textbf{GuideFlow}}, a novel planning framework that leverages Constrained Flow Matching. Concretely, \textit{\textbf{GuideFlow}} explicitly models the flow matching process, which inherently mitigates mode collapse and allows for flexible guidance from various conditioning signals. Our core contribution lies in directly enforcing explicit constraints within the flow matching generation process, rather than relying on implicit constraint encoding. Crucially, \textit{\textbf{GuideFlow}} unifies the training of the flow matching with the Energy-Based Model (EBM) to enhance the model's autonomous optimization capability to robustly satisfy physical constraints. Secondly, \textit{\textbf{GuideFlow}} parameterizes driving aggressiveness as a control signal during generation, enabling precise manipulation of trajectory style. Extensive evaluations on major driving benchmarks (Bench2Drive, NuScenes, NavSim and ADV-NuScenes) validate the effectiveness of \textit{\textbf{GuideFlow}}. Notably, on the NavSim test hard split (Navhard), \textit{\textbf{GuideFlow}} achieved SOTA with an EPDMS score of 43.0. The code will be released.
>
---
#### [new 175] Attention Guided Alignment in Efficient Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对高效视觉语言模型中的多模态对齐问题，指出拼接架构难以区分语义匹配与不匹配的图文对，导致物体幻觉。为此提出AGE-VLM框架，通过交错交叉注意力与SAM空间知识，引导模型关注正确图像区域，增强视觉定位能力，有效减少幻觉，在多个基准上表现优异。**

- **链接: [https://arxiv.org/pdf/2511.17793v1](https://arxiv.org/pdf/2511.17793v1)**

> **作者:** Shweta Mahajan; Hoang Le; Hyojin Park; Farzad Farhadzadeh; Munawar Hayat; Fatih Porikli
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Efficient Reasoning
>
> **摘要:** Large Vision-Language Models (VLMs) rely on effective multimodal alignment between pre-trained vision encoders and Large Language Models (LLMs) to integrate visual and textual information. This paper presents a comprehensive analysis of attention patterns in efficient VLMs, revealing that concatenation-based architectures frequently fail to distinguish between semantically matching and non-matching image-text pairs. This is a key factor for object hallucination in these models. To address this, we introduce Attention-Guided Efficient Vision-Language Models (AGE-VLM), a novel framework that enhances visual grounding through interleaved cross-attention layers to instill vision capabilities in pretrained small language models. This enforces in VLM the ability "look" at the correct image regions by leveraging spatial knowledge distilled from the Segment Anything Model (SAM), significantly reducing hallucination. We validate our approach across different vision-centric benchmarks where our method is better or comparable to prior work on efficient VLMs. Our findings provide valuable insights for future research aimed at achieving enhanced visual and linguistic understanding in VLMs.
>
---
#### [new 176] Rethinking Plant Disease Diagnosis: Bridging the Academic-Practical Gap with Vision Transformers and Zero-Shot Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦植物病害诊断任务，旨在解决学术数据集与真实田间图像之间的泛化差距。通过对比CNN、Vision Transformer和CLIP模型，发现后者在零样本学习下可直接基于文本描述分类，显著提升对复杂场景的适应性与可解释性，为实际应用提供高效解决方案。**

- **链接: [https://arxiv.org/pdf/2511.18989v1](https://arxiv.org/pdf/2511.18989v1)**

> **作者:** Wassim Benabbas; Mohammed Brahimi; Samir Akhrouf; Bilal Fortas
>
> **摘要:** Recent advances in deep learning have enabled significant progress in plant disease classification using leaf images. Much of the existing research in this field has relied on the PlantVillage dataset, which consists of well-centered plant images captured against uniform, uncluttered backgrounds. Although models trained on this dataset achieve high accuracy, they often fail to generalize to real-world field images, such as those submitted by farmers to plant diagnostic systems. This has created a significant gap between published studies and practical application requirements, highlighting the necessity of investigating and addressing this issue. In this study, we investigate whether attention-based architectures and zero-shot learning approaches can bridge the gap between curated academic datasets and real-world agricultural conditions in plant disease classification. We evaluate three model categories: Convolutional Neural Networks (CNNs), Vision Transformers, and Contrastive Language-Image Pre-training (CLIP)-based zero-shot models. While CNNs exhibit limited robustness under domain shift, Vision Transformers demonstrate stronger generalization by capturing global contextual features. Most notably, CLIP models classify diseases directly from natural language descriptions without any task-specific training, offering strong adaptability and interpretability. These findings highlight the potential of zero-shot learning as a practical and scalable domain adaptation strategy for plant health diagnosis in diverse field environments.
>
---
#### [new 177] Leveraging Adversarial Learning for Pathological Fidelity in Virtual Staining
- **分类: cs.CV**

- **简介: 该论文属于图像到图像翻译任务，旨在解决传统免疫组化染色成本高、耗时的问题。提出CSSP2P GAN模型，通过优化对抗损失提升虚拟染色的病理保真度，并揭示现有评估指标（如SSIM、PSNR）的不足，验证模型在专家盲评中的优越性能。**

- **链接: [https://arxiv.org/pdf/2511.18946v1](https://arxiv.org/pdf/2511.18946v1)**

> **作者:** José Teixeira; Pascal Klöckner; Diana Montezuma; Melis Erdal Cesur; João Fraga; Hugo M. Horlings; Jaime S. Cardoso; Sara P. Oliveira
>
> **摘要:** In addition to evaluating tumor morphology using H&E staining, immunohistochemistry is used to assess the presence of specific proteins within the tissue. However, this is a costly and labor-intensive technique, for which virtual staining, as an image-to-image translation task, offers a promising alternative. Although recent, this is an emerging field of research with 64% of published studies just in 2024. Most studies use publicly available datasets of H&E-IHC pairs from consecutive tissue sections. Recognizing the training challenges, many authors develop complex virtual staining models based on conditional Generative Adversarial Networks, but ignore the impact of adversarial loss on the quality of virtual staining. Furthermore, overlooking the issues of model evaluation, they claim improved performance based on metrics such as SSIM and PSNR, which are not sufficiently robust to evaluate the quality of virtually stained images. In this paper, we developed CSSP2P GAN, which we demonstrate to achieve heightened pathological fidelity through a blind pathological expert evaluation. Furthermore, while iteratively developing our model, we study the impact of the adversarial loss and demonstrate its crucial role in the quality of virtually stained images. Finally, while comparing our model with reference works in the field, we underscore the limitations of the currently used evaluation metrics and demonstrate the superior performance of CSSP2P GAN.
>
---
#### [new 178] Less is More: Data-Efficient Adaptation for Controllable Text-to-Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到视频生成中新增物理相机控制的高数据需求问题，提出一种数据高效微调策略。通过稀疏低质合成数据实现可控生成，效果优于真实数据微调。建立了理论框架解释该反直觉现象，推动了轻量化可控视频生成的发展。**

- **链接: [https://arxiv.org/pdf/2511.17844v1](https://arxiv.org/pdf/2511.17844v1)**

> **作者:** Shihan Cheng; Nilesh Kulkarni; David Hyde; Dmitriy Smirnov
>
> **摘要:** Fine-tuning large-scale text-to-video diffusion models to add new generative controls, such as those over physical camera parameters (e.g., shutter speed or aperture), typically requires vast, high-fidelity datasets that are difficult to acquire. In this work, we propose a data-efficient fine-tuning strategy that learns these controls from sparse, low-quality synthetic data. We show that not only does fine-tuning on such simple data enable the desired controls, it actually yields superior results to models fine-tuned on photorealistic "real" data. Beyond demonstrating these results, we provide a framework that justifies this phenomenon both intuitively and quantitatively.
>
---
#### [new 179] FVAR: Visual Autoregressive Modeling via Next Focus Prediction
- **分类: cs.CV**

- **简介: 该论文提出FVAR，一种基于“下一焦点预测”的视觉自回归模型，旨在解决传统多尺度下采样导致的混叠伪影问题。通过物理一致的模糊核构建无混叠金字塔，并引入高频残差教师网络，提升细节保真度与文本可读性，兼容现有自回归框架。**

- **链接: [https://arxiv.org/pdf/2511.18838v1](https://arxiv.org/pdf/2511.18838v1)**

> **作者:** Xiaofan Li; Chenming Wu; Yanpeng Sun; Jiaming Zhou; Delin Qu; Yansong Qu; Weihao Bo; Haibao Yu; Dingkang Liang
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Visual autoregressive models achieve remarkable generation quality through next-scale predictions across multi-scale token pyramids. However, the conventional method uses uniform scale downsampling to build these pyramids, leading to aliasing artifacts that compromise fine details and introduce unwanted jaggies and moiré patterns. To tackle this issue, we present \textbf{FVAR}, which reframes the paradigm from \emph{next-scale prediction} to \emph{next-focus prediction}, mimicking the natural process of camera focusing from blur to clarity. Our approach introduces three key innovations: \textbf{1) Next-Focus Prediction Paradigm} that transforms multi-scale autoregression by progressively reducing blur rather than simply downsampling; \textbf{2) Progressive Refocusing Pyramid Construction} that uses physics-consistent defocus kernels to build clean, alias-free multi-scale representations; and \textbf{3) High-Frequency Residual Learning} that employs a specialized residual teacher network to effectively incorporate alias information during training while maintaining deployment simplicity. Specifically, we construct optical low-pass views using defocus point spread function (PSF) kernels with decreasing radius, creating smooth blur-to-clarity transitions that eliminate aliasing at its source. To further enhance detail generation, we introduce a High-Frequency Residual Teacher that learns from both clean structure and alias residuals, distilling this knowledge to a vanilla VAR deployment network for seamless inference. Extensive experiments on ImageNet demonstrate that FVAR substantially reduces aliasing artifacts, improves fine detail preservation, and enhances text readability, achieving superior performance with perfect compatibility to existing VAR frameworks.
>
---
#### [new 180] CUS-GS: A Compact Unified Structured Gaussian Splatting Framework for Multimodal Scene Representation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CUS-GS，一种紧凑的统一结构化高斯点阵框架，用于多模态场景表示。针对现有方法在语义理解与几何建模间的割裂问题，通过体素锚点结构融合多模态语义特征与3D几何，实现高效、一致的场景表征，显著提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.17904v1](https://arxiv.org/pdf/2511.17904v1)**

> **作者:** Yuhang Ming; Chenxin Fang; Xingyuan Yu; Fan Zhang; Weichen Dai; Wanzeng Kong; Guofeng Zhang
>
> **备注:** 15 pages, 8 figures, 4 tables
>
> **摘要:** Recent advances in Gaussian Splatting based 3D scene representation have shown two major trends: semantics-oriented approaches that focus on high-level understanding but lack explicit 3D geometry modeling, and structure-oriented approaches that capture spatial structures yet provide limited semantic abstraction. To bridge this gap, we present CUS-GS, a compact unified structured Gaussian Splatting representation, which connects multimodal semantic features with structured 3D geometry. Specifically, we design a voxelized anchor structure that constructs a spatial scaffold, while extracting multimodal semantic features from a set of foundation models (e.g., CLIP, DINOv2, SEEM). Moreover, we introduce a multimodal latent feature allocation mechanism to unify appearance, geometry, and semantics across heterogeneous feature spaces, ensuring a consistent representation across multiple foundation models. Finally, we propose a feature-aware significance evaluation strategy to dynamically guide anchor growing and pruning, effectively removing redundant or invalid anchors while maintaining semantic integrity. Extensive experiments show that CUS-GS achieves competitive performance compared to state-of-the-art methods using as few as 6M parameters - an order of magnitude smaller than the closest rival at 35M - highlighting the excellent trade off between performance and model efficiency of the proposed framework.
>
---
#### [new 181] MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes
- **分类: cs.CV**

- **简介: 该论文针对大场景3D重建中几何精度与稳定性不足的问题，提出MetroGS框架。通过分布式2D高斯表示、结构化稠密初始化、渐进式几何优化及深度引导的外观建模，实现高效稳定的高保真大场景重建。**

- **链接: [https://arxiv.org/pdf/2511.19172v1](https://arxiv.org/pdf/2511.19172v1)**

> **作者:** Kehua Chen; Tianlu Mao; Zhuxin Ma; Hao Jiang; Zehao Li; Zihan Liu; Shuqi Gao; Honglong Zhao; Feng Dai; Yucheng Zhang; Zhaoqi Wang
>
> **备注:** Project page: https://m3phist0.github.io/MetroGS
>
> **摘要:** Recently, 3D Gaussian Splatting and its derivatives have achieved significant breakthroughs in large-scale scene reconstruction. However, how to efficiently and stably achieve high-quality geometric fidelity remains a core challenge. To address this issue, we introduce MetroGS, a novel Gaussian Splatting framework for efficient and robust reconstruction in complex urban environments. Our method is built upon a distributed 2D Gaussian Splatting representation as the core foundation, serving as a unified backbone for subsequent modules. To handle potential sparse regions in complex scenes, we propose a structured dense enhancement scheme that utilizes SfM priors and a pointmap model to achieve a denser initialization, while incorporating a sparsity compensation mechanism to improve reconstruction completeness. Furthermore, we design a progressive hybrid geometric optimization strategy that organically integrates monocular and multi-view optimization to achieve efficient and accurate geometric refinement. Finally, to address the appearance inconsistency commonly observed in large-scale scenes, we introduce a depth-guided appearance modeling approach that learns spatial features with 3D consistency, facilitating effective decoupling between geometry and appearance and further enhancing reconstruction stability. Experiments on large-scale urban datasets demonstrate that MetroGS achieves superior geometric accuracy, rendering quality, offering a unified solution for high-fidelity large-scale scene reconstruction.
>
---
#### [new 182] Syn-GRPO: Self-Evolving Data Synthesis for MLLM Perception Reasoning
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLM）感知能力强化学习中数据质量低、响应单一的问题，提出Syn-GRPO方法。通过在线数据生成器合成高质量、多样化的训练数据，结合去耦异步生成与多样性奖励机制，显著提升数据多样性与模型性能，在视觉感知任务中表现优异，具备长期自演化潜力。**

- **链接: [https://arxiv.org/pdf/2511.19343v1](https://arxiv.org/pdf/2511.19343v1)**

> **作者:** Qihan Huang; Haofei Zhang; Rong Wei; Yi Wang; Rui Tang; Mingli Song; Jie Song
>
> **摘要:** RL (reinforcement learning) methods (e.g., GRPO) for MLLM (Multimodal LLM) perception ability has attracted wide research interest owing to its remarkable generalization ability. Nevertheless, existing reinforcement learning methods still face the problem of low data quality, where data samples cannot elicit diverse responses from MLLMs, thus restricting the exploration scope for MLLM reinforcement learning. Some methods attempt to mitigate this problem by imposing constraints on entropy, but none address it at its root. Therefore, to tackle this problem, this work proposes Syn-GRPO (Synthesis-GRPO), which employs an online data generator to synthesize high-quality training data with diverse responses in GRPO training. Specifically, Syn-GRPO consists of two components: (1) data server; (2) GRPO workflow. The data server synthesizes new samples from existing ones using an image generation model, featuring a decoupled and asynchronous scheme to achieve high generation efficiency. The GRPO workflow provides the data server with the new image descriptions, and it leverages a diversity reward to supervise the MLLM to predict image descriptions for synthesizing samples with diverse responses. Experiment results across three visual perception tasks demonstrate that Syn-GRPO improves the data quality by a large margin, achieving significant superior performance to existing MLLM perception methods, and Syn-GRPO presents promising potential for scaling long-term self-evolving RL. Our code is available at https://github.com/hqhQAQ/Syn-GRPO.
>
---
#### [new 183] HSMix: Hard and Soft Mixing Data Augmentation for Medical Image Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对医疗图像分割中的数据稀缺与过拟合问题，提出HSMix方法。通过硬混合（超像素拼接）与软混合（亮度调整）结合，生成多样化且语义一致的增强图像，保留局部结构信息，提升模型泛化能力。方法为模型无关、可移植的插件式方案，适用于多种医学影像模态。**

- **链接: [https://arxiv.org/pdf/2511.17614v1](https://arxiv.org/pdf/2511.17614v1)**

> **作者:** Danyang Sun; Fadi Dornaika; Nagore Barrena
>
> **摘要:** Due to the high cost of annotation or the rarity of some diseases, medical image segmentation is often limited by data scarcity and the resulting overfitting problem. Self-supervised learning and semi-supervised learning can mitigate the data scarcity challenge to some extent. However, both of these paradigms are complex and require either hand-crafted pretexts or well-defined pseudo-labels. In contrast, data augmentation represents a relatively simple and straightforward approach to addressing data scarcity issues. It has led to significant improvements in image recognition tasks. However, the effectiveness of local image editing augmentation techniques in the context of segmentation has been less explored. We propose HSMix, a novel approach to local image editing data augmentation involving hard and soft mixing for medical semantic segmentation. In our approach, a hard-augmented image is created by combining homogeneous regions (superpixels) from two source images. A soft mixing method further adjusts the brightness of these composed regions with brightness mixing based on locally aggregated pixel-wise saliency coefficients. The ground-truth segmentation masks of the two source images undergo the same mixing operations to generate the associated masks for the augmented images. Our method fully exploits both the prior contour and saliency information, thus preserving local semantic information in the augmented images while enriching the augmentation space with more diversity. Our method is a plug-and-play solution that is model agnostic and applicable to a range of medical imaging modalities. Extensive experimental evidence has demonstrated its effectiveness in a variety of medical segmentation tasks. The source code is available in https://github.com/DanielaPlusPlus/HSMix.
>
---
#### [new 184] From Healthy Scans to Annotated Tumors: A Tumor Fabrication Framework for 3D Brain MRI Synthesis
- **分类: cs.CV**

- **简介: 该论文针对3D脑肿瘤MRI数据标注稀缺问题，提出全自动的Tumor Fabrication（TF）框架。通过无配对的两阶段合成方法，仅用健康扫描和少量真实标注数据，生成大量带标签的合成数据，有效提升低数据场景下肿瘤分割性能，解决临床AI中数据不足难题。**

- **链接: [https://arxiv.org/pdf/2511.18654v1](https://arxiv.org/pdf/2511.18654v1)**

> **作者:** Nayu Dong; Townim Chowdhury; Hieu Phan; Mark Jenkinson; Johan Verjans; Zhibin Liao
>
> **摘要:** The scarcity of annotated Magnetic Resonance Imaging (MRI) tumor data presents a major obstacle to accurate and automated tumor segmentation. While existing data synthesis methods offer promising solutions, they often suffer from key limitations: manual modeling is labor intensive and requires expert knowledge. Deep generative models may be used to augment data and annotation, but they typically demand large amounts of training pairs in the first place, which is impractical in data limited clinical settings. In this work, we propose Tumor Fabrication (TF), a novel two-stage framework for unpaired 3D brain tumor synthesis. The framework comprises a coarse tumor synthesis process followed by a refinement process powered by a generative model. TF is fully automated and leverages only healthy image scans along with a limited amount of real annotated data to synthesize large volumes of paired synthetic data for enriching downstream supervised segmentation training. We demonstrate that our synthetic image-label pairs used as data enrichment can significantly improve performance on downstream tumor segmentation tasks in low-data regimes, offering a scalable and reliable solution for medical image enrichment and addressing critical challenges in data scarcity for clinical AI applications.
>
---
#### [new 185] ScriptViT: Vision Transformer-Based Personalized Handwriting Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文聚焦于个性化手写体生成任务，旨在解决现有方法难以捕捉长距离风格特征的问题。提出基于视觉Transformer的风格编码器，通过多参考图像学习全局书写风格，并结合交叉注意力机制生成更忠实于作者风格的逼真手写文本，同时引入可解释性分析工具提升模型透明度。**

- **链接: [https://arxiv.org/pdf/2511.18307v1](https://arxiv.org/pdf/2511.18307v1)**

> **作者:** Sajjan Acharya; Rajendra Baskota
>
> **摘要:** Styled handwriting generation aims to synthesize handwritten text that looks both realistic and aligned with a specific writer's style. While recent approaches involving GAN, transformer and diffusion-based models have made progress, they often struggle to capture the full spectrum of writer-specific attributes, particularly global stylistic patterns that span long-range spatial dependencies. As a result, capturing subtle writer-specific traits such as consistent slant, curvature or stroke pressure, while keeping the generated text accurate is still an open problem. In this work, we present a unified framework designed to address these limitations. We introduce a Vision Transformer-based style encoder that learns global stylistic patterns from multiple reference images, allowing the model to better represent long-range structural characteristics of handwriting. We then integrate these style cues with the target text using a cross-attention mechanism, enabling the system to produce handwritten images that more faithfully reflect the intended style. To make the process more interpretable, we utilize Salient Stroke Attention Analysis (SSAA), which reveals the stroke-level features the model focuses on during style transfer. Together, these components lead to handwriting synthesis that is not only more stylistically coherent, but also easier to understand and analyze.
>
---
#### [new 186] Growing with the Generator: Self-paced GRPO for Video Generation
- **分类: cs.CV**

- **简介: 该论文针对视频生成中奖励模型静态导致的分布偏移与优化瓶颈问题，提出自适应进度的GRPO框架。通过动态调整奖励侧重（从视觉保真度到时序连贯性与语义对齐），实现奖励与生成器协同进化，提升生成质量与训练稳定性。**

- **链接: [https://arxiv.org/pdf/2511.19356v1](https://arxiv.org/pdf/2511.19356v1)**

> **作者:** Rui Li; Yuanzhi Liang; Ziqi Ni; Haibing Huang; Chi Zhang; Xuelong Li
>
> **摘要:** Group Relative Policy Optimization (GRPO) has emerged as a powerful reinforcement learning paradigm for post-training video generation models. However, existing GRPO pipelines rely on static, fixed-capacity reward models whose evaluation behavior is frozen during training. Such rigid rewards introduce distributional bias, saturate quickly as the generator improves, and ultimately limit the stability and effectiveness of reinforcement-based alignment. We propose Self-Paced GRPO, a competence-aware GRPO framework in which reward feedback co-evolves with the generator. Our method introduces a progressive reward mechanism that automatically shifts its emphasis from coarse visual fidelity to temporal coherence and fine-grained text-video semantic alignment as generation quality increases. This self-paced curriculum alleviates reward-policy mismatch, mitigates reward exploitation, and yields more stable optimization. Experiments on VBench across multiple video generation backbones demonstrate consistent improvements in both visual quality and semantic alignment over GRPO baselines with static rewards, validating the effectiveness and generality of Self-Paced GRPO.
>
---
#### [new 187] Dual-Granularity Semantic Prompting for Language Guidance Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文针对红外小目标检测任务，解决特征表示弱与背景干扰严重的问题。提出DGSPNet框架，通过双粒度语义提示（粗粒度文本先验与细粒度个性化描述）实现无标注语言引导，并设计文本引导通道与空间注意力机制，提升模型对目标的敏感性，显著提升检测性能。**

- **链接: [https://arxiv.org/pdf/2511.19306v1](https://arxiv.org/pdf/2511.19306v1)**

> **作者:** Zixuan Wang; Haoran Sun; Jiaming Lu; Wenxuan Wang; Zhongling Huang; Dingwen Zhang; Xuelin Qian; Junwei Han
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Infrared small target detection remains challenging due to limited feature representation and severe background interference, resulting in sub-optimal performance. While recent CLIP-inspired methods attempt to leverage textual guidance for detection, they are hindered by inaccurate text descriptions and reliance on manual annotations. To overcome these limitations, we propose DGSPNet, an end-to-end language prompt-driven framework. Our approach integrates dual-granularity semantic prompts: coarse-grained textual priors (e.g., 'infrared image', 'small target') and fine-grained personalized semantic descriptions derived through visual-to-textual mapping within the image space. This design not only facilitates learning fine-grained semantic information but also can inherently leverage language prompts during inference without relying on any annotation requirements. By fully leveraging the precision and conciseness of text descriptions, we further introduce a text-guide channel attention (TGCA) mechanism and text-guide spatial attention (TGSA) mechanism that enhances the model's sensitivity to potential targets across both low- and high-level feature spaces. Extensive experiments demonstrate that our method significantly improves detection accuracy and achieves state-of-the-art performance on three benchmark datasets.
>
---
#### [new 188] Signal: Selective Interaction and Global-local Alignment for Multi-Modal Object Re-Identification
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对多模态物体重识别任务，解决背景干扰与多模态一致性对齐问题。提出Signal框架，包含选择性交互模块（SIM）增强特征判别性，以及全局-局部对齐模块（GAM/LAM）实现跨模态特征精准对齐，显著提升识别性能。**

- **链接: [https://arxiv.org/pdf/2511.17965v1](https://arxiv.org/pdf/2511.17965v1)**

> **作者:** Yangyang Liu; Yuhao Wang; Pingping Zhang
>
> **备注:** Accepted by AAAI2026. More modifications may be performed
>
> **摘要:** Multi-modal object Re-IDentification (ReID) is devoted to retrieving specific objects through the exploitation of complementary multi-modal image information. Existing methods mainly concentrate on the fusion of multi-modal features, yet neglecting the background interference. Besides, current multi-modal fusion methods often focus on aligning modality pairs but suffer from multi-modal consistency alignment. To address these issues, we propose a novel selective interaction and global-local alignment framework called Signal for multi-modal object ReID. Specifically, we first propose a Selective Interaction Module (SIM) to select important patch tokens with intra-modal and inter-modal information. These important patch tokens engage in the interaction with class tokens, thereby yielding more discriminative features. Then, we propose a Global Alignment Module (GAM) to simultaneously align multi-modal features by minimizing the volume of 3D polyhedra in the gramian space. Meanwhile, we propose a Local Alignment Module (LAM) to align local features in a shift-aware manner. With these modules, our proposed framework could extract more discriminative features for object ReID. Extensive experiments on three multi-modal object ReID benchmarks (i.e., RGBNT201, RGBNT100, MSVR310) validate the effectiveness of our method. The source code is available at https://github.com/010129/Signal.
>
---
#### [new 189] Cloud4D
- **分类: cs.CV; physics.ao-ph**

- **简介: 该论文提出Cloud4D，一种基于地面摄像头的四维云状态重建框架，解决高分辨率云模拟中观测数据不足的问题。通过2D-to-3D Transformer融合多视角图像，实现25m空间、5s时间分辨率的液态水含量三维分布及风场估计，显著提升时空分辨率与精度。**

- **链接: [https://arxiv.org/pdf/2511.19431v1](https://arxiv.org/pdf/2511.19431v1)**

> **作者:** Jacob Lin; Edward Gryspeerdt; Ronald Clark
>
> **备注:** NeurIPS 2025 Spotlight, project page: https://cloud4d.jacob-lin.com/
>
> **摘要:** There has been great progress in improving numerical weather prediction and climate models using machine learning. However, most global models act at a kilometer-scale, making it challenging to model individual clouds and factors such as extreme precipitation, wind gusts, turbulence, and surface irradiance. Therefore, there is a need to move towards higher-resolution models, which in turn require high-resolution real-world observations that current instruments struggle to obtain. We present Cloud4D, the first learning-based framework that reconstructs a physically consistent, four-dimensional cloud state using only synchronized ground-based cameras. Leveraging a homography-guided 2D-to-3D transformer, Cloud4D infers the full 3D distribution of liquid water content at 25 m spatial and 5 s temporal resolution. By tracking the 3D liquid water content retrievals over time, Cloud4D additionally estimates horizontal wind vectors. Across a two-month deployment comprising six skyward cameras, our system delivers an order-of-magnitude improvement in space-time resolution relative to state-of-the-art satellite measurements, while retaining single-digit relative error ($<10\%$) against collocated radar measurements. Code and data are available on our project page https://cloud4d.jacob-lin.com/.
>
---
#### [new 190] MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对类别无关的3D姿态迁移任务，解决跨类型角色（如人形到四足）姿态转移中结构差异导致的匹配错误问题。提出MimiCAT模型，利用语义关键点构建软对应关系，实现灵活的多对多匹配，并通过条件生成与形状约束提升迁移质量，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18370v1](https://arxiv.org/pdf/2511.18370v1)**

> **作者:** Zenghao Chai; Chen Tang; Yongkang Wong; Xulei Yang; Mohan Kankanhalli
>
> **备注:** tech report
>
> **摘要:** 3D pose transfer aims to transfer the pose-style of a source mesh to a target character while preserving both the target's geometry and the source's pose characteristic. Existing methods are largely restricted to characters with similar structures and fail to generalize to category-free settings (e.g., transferring a humanoid's pose to a quadruped). The key challenge lies in the structural and transformation diversity inherent in distinct character types, which often leads to mismatched regions and poor transfer quality. To address these issues, we first construct a million-scale pose dataset across hundreds of distinct characters. We further propose MimiCAT, a cascade-transformer model designed for category-free 3D pose transfer. Instead of relying on strict one-to-one correspondence mappings, MimiCAT leverages semantic keypoint labels to learn a novel soft correspondence that enables flexible many-to-many matching across characters. The pose transfer is then formulated as a conditional generation process, in which the source transformations are first projected onto the target through soft correspondence matching and subsequently refined using shape-conditioned representations. Extensive qualitative and quantitative experiments demonstrate that MimiCAT transfers plausible poses across different characters, significantly outperforming prior methods that are limited to narrow category transfer (e.g., humanoid-to-humanoid).
>
---
#### [new 191] Video4Edit: Viewing Image Editing as a Degenerate Temporal Process
- **分类: cs.CV**

- **简介: 该论文研究图像编辑任务，针对现有方法依赖大量高质量数据、成本高昂的问题，提出将图像编辑视为退化的时序过程，利用视频预训练中单帧演化先验，实现高效微调。实验表明，仅需1%的监督数据即可达到主流模型性能。**

- **链接: [https://arxiv.org/pdf/2511.18131v1](https://arxiv.org/pdf/2511.18131v1)**

> **作者:** Xiaofan Li; Yanpeng Sun; Chenming Wu; Fan Duan; YuAn Wang; Weihao Bo; Yumeng Zhang; Dingkang Liang
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** We observe that recent advances in multimodal foundation models have propelled instruction-driven image generation and editing into a genuinely cross-modal, cooperative regime. Nevertheless, state-of-the-art editing pipelines remain costly: beyond training large diffusion/flow models, they require curating massive high-quality triplets of \{instruction, source image, edited image\} to cover diverse user intents. Moreover, the fidelity of visual replacements hinges on how precisely the instruction references the target semantics. We revisit this challenge through the lens of temporal modeling: if video can be regarded as a full temporal process, then image editing can be seen as a degenerate temporal process. This perspective allows us to transfer single-frame evolution priors from video pre-training, enabling a highly data-efficient fine-tuning regime. Empirically, our approach matches the performance of leading open-source baselines while using only about one percent of the supervision demanded by mainstream editing models.
>
---
#### [new 192] DocPTBench: Benchmarking End-to-End Photographed Document Parsing and Translation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对真实拍摄文档的解析与翻译难题，提出DocPTBench基准。它包含1300+张高分辨率实拍文档，覆盖多领域与多语言场景，提供人工验证的标注。实验表明，现有模型在实拍文档上性能显著下降，揭示了当前模型对真实世界复杂条件的脆弱性。**

- **链接: [https://arxiv.org/pdf/2511.18434v1](https://arxiv.org/pdf/2511.18434v1)**

> **作者:** Yongkun Du; Pinxuan Chen; Xuye Ying; Zhineng Chen
>
> **摘要:** The advent of Multimodal Large Language Models (MLLMs) has unlocked the potential for end-to-end document parsing and translation. However, prevailing benchmarks such as OmniDocBench and DITrans are dominated by pristine scanned or digital-born documents, and thus fail to adequately represent the intricate challenges of real-world capture conditions, such as geometric distortions and photometric variations. To fill this gap, we introduce DocPTBench, a comprehensive benchmark specifically designed for Photographed Document Parsing and Translation. DocPTBench comprises over 1,300 high-resolution photographed documents from multiple domains, includes eight translation scenarios, and provides meticulously human-verified annotations for both parsing and translation. Our experiments demonstrate that transitioning from digital-born to photographed documents results in a substantial performance decline: popular MLLMs exhibit an average accuracy drop of 18% in end-to-end parsing and 12% in translation, while specialized document parsing models show significant average decrease of 25%. This substantial performance gap underscores the unique challenges posed by documents captured in real-world conditions and reveals the limited robustness of existing models. Dataset and code are available at https://github.com/Topdu/DocPTBench.
>
---
#### [new 193] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion
- **分类: cs.CV; cs.AI; cs.LG; eess.SP**

- **简介: 该论文提出REXO，解决多视角室内雷达目标检测中因隐式特征关联导致的匹配模糊问题。通过将2D边界框扩散扩展至3D雷达空间，实现显式跨视角特征关联，并利用人体接触地面的先验知识减少参数量，显著提升检测精度。**

- **链接: [https://arxiv.org/pdf/2511.17806v1](https://arxiv.org/pdf/2511.17806v1)**

> **作者:** Ryoma Yataka; Pu Perry Wang; Petros Boufounos; Ryuhei Takahashi
>
> **备注:** 26 pages, Accepted to AAAI 2026; Code to be released
>
> **摘要:** Multi-view indoor radar perception has drawn attention due to its cost-effectiveness and low privacy risks. Existing methods often rely on {implicit} cross-view radar feature association, such as proposal pairing in RFMask or query-to-feature cross-attention in RETR, which can lead to ambiguous feature matches and degraded detection in complex indoor scenes. To address these limitations, we propose \textbf{REXO} (multi-view Radar object dEtection with 3D bounding boX diffusiOn), which lifts the 2D bounding box (BBox) diffusion process of DiffusionDet into the 3D radar space. REXO utilizes these noisy 3D BBoxes to guide an {explicit} cross-view radar feature association, enhancing the cross-view radar-conditioned denoising process. By accounting for prior knowledge that the person is in contact with the ground, REXO reduces the number of diffusion parameters by determining them from this prior. Evaluated on two open indoor radar datasets, our approach surpasses state-of-the-art methods by a margin of +4.22 AP on the HIBER dataset and +11.02 AP on the MMVR dataset.
>
---
#### [new 194] Robust Posterior Diffusion-based Sampling via Adaptive Guidance Scale
- **分类: cs.CV**

- **简介: 该论文针对图像逆问题中的扩散模型重建质量与稳定性难题，提出自适应后验扩散采样（AdaPS）方法。通过观测依赖的梯度一致性判断动态调整似然步长，实现无需调参的鲁棒优化，在超分辨率、去模糊等任务中显著提升感知质量，且对噪声、步骤数和随机性均具强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.18471v1](https://arxiv.org/pdf/2511.18471v1)**

> **作者:** Liav Hen; Tom Tirer; Raja Giryes; Shady Abu-Hussein
>
> **摘要:** Diffusion models have recently emerged as powerful generative priors for solving inverse problems, achieving state-of-the-art results across various imaging tasks. A central challenge in this setting lies in balancing the contribution of the prior with the data fidelity term: overly aggressive likelihood updates may introduce artifacts, while conservative updates can slow convergence or yield suboptimal reconstructions. In this work, we propose an adaptive likelihood step-size strategy to guide the diffusion process for inverse-problem formulations. Specifically, we develop an observation-dependent weighting scheme based on the agreement between two different approximations of the intractable intermediate likelihood gradients, that adapts naturally to the diffusion schedule, time re-spacing, and injected stochasticity. The resulting approach, Adaptive Posterior diffusion Sampling (AdaPS), is hyperparameter-free and improves reconstruction quality across diverse imaging tasks - including super-resolution, Gaussian deblurring, and motion deblurring - on CelebA-HQ and ImageNet-256 validation sets. AdaPS consistently surpasses existing diffusion-based baselines in perceptual quality with minimal or no loss in distortion, without any task-specific tuning. Extensive ablation studies further demonstrate its robustness to the number of diffusion steps, observation noise levels, and varying stochasticity.
>
---
#### [new 195] A Lightweight, Interpretable Deep Learning System for Automated Detection of Cervical Adenocarcinoma In Situ (AIS)
- **分类: cs.CV; q-bio.TO**

- **简介: 该论文属于医学图像分类任务，旨在解决宫颈腺癌原位（AIS）早期诊断困难的问题。研究基于CAISHI数据集，构建轻量级可解释的EfficientNet-B3模型，通过染色标准化与分块处理提升特征表示，利用焦点损失和平衡采样优化训练。模型在区分AIS与正常组织上表现良好，且热力图揭示了生物学意义的病变区域，实现了自动检测与可视化辅助诊断。**

- **链接: [https://arxiv.org/pdf/2511.18063v1](https://arxiv.org/pdf/2511.18063v1)**

> **作者:** Gabriela Fernandes
>
> **摘要:** Cervical adenocarcinoma in situ (AIS) is a critical premalignant lesion whose accurate histopathological diagnosis is challenging. Early detection is essential to prevent progression to invasive cervical adenocarcinoma. In this study, we developed a deep learning-based virtual pathology assistant capable of distinguishing AIS from normal cervical gland histology using the CAISHI dataset, which contains 2240 expert-labeled H&E images (1010 normal and 1230 AIS). All images underwent Macenko stain normalization and patch-based preprocessing to enhance morphological feature representation. An EfficientNet-B3 convolutional neural network was trained using class-balanced sampling and focal loss to address dataset imbalance and emphasize difficult examples. The final model achieved an overall accuracy of 0.7323, with an F1-score of 0.75 for the Abnormal class and 0.71 for the Normal class. Grad-CAM heatmaps demonstrated biologically interpretable activation patterns, highlighting nuclear atypia and glandular crowding consistent with AIS morphology. The trained model was deployed in a Gradio-based virtual diagnostic assistant. These findings demonstrate the feasibility of lightweight, interpretable AI systems for cervical gland pathology, with potential applications in screening workflows, education, and low-resource settings.
>
---
#### [new 196] Importance-Weighted Non-IID Sampling for Flow Matching Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对流匹配模型在有限采样预算下估计函数期望时因独立采样导致高方差的问题，提出一种重要性加权的非独立同分布（non-IID）采样框架。通过联合生成多样且高质量的样本，并利用残差速度场学习重要性权重，实现无偏估计，提升对稀有高影响结果的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2511.17812v1](https://arxiv.org/pdf/2511.17812v1)**

> **作者:** Xinshuang Liu; Runfa Blark Li; Shaoxiu Wei; Truong Nguyen
>
> **摘要:** Flow-matching models effectively represent complex distributions, yet estimating expectations of functions of their outputs remains challenging under limited sampling budgets. Independent sampling often yields high-variance estimates, especially when rare but with high-impact outcomes dominate the expectation. We propose an importance-weighted non-IID sampling framework that jointly draws multiple samples to cover diverse, salient regions of a flow's distribution while maintaining unbiased estimation via estimated importance weights. To balance diversity and quality, we introduce a score-based regularization for the diversity mechanism, which uses the score function, i.e., the gradient of the log probability, to ensure samples are pushed apart within high-density regions of the data manifold, mitigating off-manifold drift. We further develop the first approach for importance weighting of non-IID flow samples by learning a residual velocity field that reproduces the marginal distribution of the non-IID samples. Empirically, our method produces diverse, high-quality samples and accurate estimates of both importance weights and expectations, advancing the reliable characterization of flow-matching model outputs. Our code will be publicly available on GitHub.
>
---
#### [new 197] X-ReID: Multi-granularity Information Interaction for Video-Based Visible-Infrared Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对视频域可见光-红外行人重识别（VVI-ReID）任务，解决模态差异与时空信息利用不足问题。提出X-ReID框架，通过跨模态原型协作（CPC）对齐特征，设计多粒度信息交互（MII）融合短时、长时及跨模态信息，提升序列级表示能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17964v1](https://arxiv.org/pdf/2511.17964v1)**

> **作者:** Chenyang Yu; Xuehu Liu; Pingping Zhang; Huchuan Lu
>
> **备注:** Accepted by AAAI2026. More modifications may be performed
>
> **摘要:** Large-scale vision-language models (e.g., CLIP) have recently achieved remarkable performance in retrieval tasks, yet their potential for Video-based Visible-Infrared Person Re-Identification (VVI-ReID) remains largely unexplored. The primary challenges are narrowing the modality gap and leveraging spatiotemporal information in video sequences. To address the above issues, in this paper, we propose a novel cross-modality feature learning framework named X-ReID for VVI-ReID. Specifically, we first propose a Cross-modality Prototype Collaboration (CPC) to align and integrate features from different modalities, guiding the network to reduce the modality discrepancy. Then, a Multi-granularity Information Interaction (MII) is designed, incorporating short-term interactions from adjacent frames, long-term cross-frame information fusion, and cross-modality feature alignment to enhance temporal modeling and further reduce modality gaps. Finally, by integrating multi-granularity information, a robust sequence-level representation is achieved. Extensive experiments on two large-scale VVI-ReID benchmarks (i.e., HITSZ-VCM and BUPTCampus) demonstrate the superiority of our method over state-of-the-art methods. The source code is released at https://github.com/AsuradaYuci/X-ReID.
>
---
#### [new 198] StereoDETR: Stereo-based Transformer for 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文提出StereoDETR，一种高效立体3D目标检测框架。针对立体方法计算开销大、速度慢的问题，设计双分支结构：单目DETR分支与立体深度分支，通过可微深度采样耦合。引入约束监督处理遮挡，实现实时推理且精度超越单目方法，在KITTI数据集上达到新纪录。**

- **链接: [https://arxiv.org/pdf/2511.18788v1](https://arxiv.org/pdf/2511.18788v1)**

> **作者:** Shiyi Mu; Zichong Gu; Zhiqi Ai; Anqi Liu; Yilin Gao; Shugong Xu
>
> **备注:** Accepted by IEEE TCSVT, 2025
>
> **摘要:** Compared to monocular 3D object detection, stereo-based 3D methods offer significantly higher accuracy but still suffer from high computational overhead and latency. The state-of-the-art stereo 3D detection method achieves twice the accuracy of monocular approaches, yet its inference speed is only half as fast. In this paper, we propose StereoDETR, an efficient stereo 3D object detection framework based on DETR. StereoDETR consists of two branches: a monocular DETR branch and a stereo branch. The DETR branch is built upon 2D DETR with additional channels for predicting object scale, orientation, and sampling points. The stereo branch leverages low-cost multi-scale disparity features to predict object-level depth maps. These two branches are coupled solely through a differentiable depth sampling strategy. To handle occlusion, we introduce a constrained supervision strategy for sampling points without requiring extra annotations. StereoDETR achieves real-time inference and is the first stereo-based method to surpass monocular approaches in speed. It also achieves competitive accuracy on the public KITTI benchmark, setting new state-of-the-art results on pedestrian and cyclist subsets. The code is available at https://github.com/shiyi-mu/StereoDETR-OPEN.
>
---
#### [new 199] Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中感知不准确、长尾场景表现差的问题，提出Percept-WAM模型。通过统一2D/3D感知任务，引入空间坐标与置信度编码的World-PV/BEV tokens，结合网格条件预测机制，提升小物体、远距离等复杂场景下的感知稳定性，并实现端到端感知与控制输出。**

- **链接: [https://arxiv.org/pdf/2511.19221v1](https://arxiv.org/pdf/2511.19221v1)**

> **作者:** Jianhua Han; Meng Tian; Jiangtong Zhu; Fan He; Huixin Zhang; Sitong Guo; Dechang Zhu; Hao Tang; Pei Xu; Yuze Guo; Minzhe Niu; Haojie Zhu; Qichao Dong; Xuechao Yan; Siyuan Dong; Lu Hou; Qingqiu Huang; Xiaosong Jia; Hang Xu
>
> **摘要:** Autonomous driving heavily relies on accurate and robust spatial perception. Many failures arise from inaccuracies and instability, especially in long-tail scenarios and complex interactions. However, current vision-language models are weak at spatial grounding and understanding, and VLA systems built on them therefore show limited perception and localization ability. To address these challenges, we introduce Percept-WAM, a perception-enhanced World-Awareness-Action Model that is the first to implicitly integrate 2D/3D scene understanding abilities within a single vision-language model (VLM). Instead of relying on QA-style spatial reasoning, Percept-WAM unifies 2D/3D perception tasks into World-PV and World-BEV tokens, which encode both spatial coordinates and confidence. We propose a grid-conditioned prediction mechanism for dense object perception, incorporating IoU-aware scoring and parallel autoregressive decoding, improving stability in long-tail, far-range, and small-object scenarios. Additionally, Percept-WAM leverages pretrained VLM parameters to retain general intelligence (e.g., logical reasoning) and can output perception results and trajectory control outputs directly. Experiments show that Percept-WAM matches or surpasses classical detectors and segmenters on downstream perception benchmarks, achieving 51.7/58.9 mAP on COCO 2D detection and nuScenes BEV 3D detection. When integrated with trajectory decoders, it further improves planning performance on nuScenes and NAVSIM, e.g., surpassing DiffusionDrive by 2.1 in PMDS on NAVSIM. Qualitative results further highlight its strong open-vocabulary and long-tail generalization.
>
---
#### [new 200] DEAP-3DSAM: Decoder Enhanced and Auto Prompt SAM for 3D Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对3D医学图像分割任务，解决SAM模型在3D应用中因伪3D处理导致的空间特征丢失及依赖人工提示的问题。提出DEAP-3DSAM，通过增强解码器融合空间信息，设计双注意力自动提示器实现无需人工干预的精准分割，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.19071v1](https://arxiv.org/pdf/2511.19071v1)**

> **作者:** Fangda Chen; Jintao Tang; Pancheng Wang; Ting Wang; Shasha Li; Ting Deng
>
> **备注:** Accepted by BIBM 2024
>
> **摘要:** The Segment Anything Model (SAM) has recently demonstrated significant potential in medical image segmentation. Although SAM is primarily trained on 2D images, attempts have been made to apply it to 3D medical image segmentation. However, the pseudo 3D processing used to adapt SAM results in spatial feature loss, limiting its performance. Additionally, most SAM-based methods still rely on manual prompts, which are challenging to implement in real-world scenarios and require extensive external expert knowledge. To address these limitations, we introduce the Decoder Enhanced and Auto Prompt SAM (DEAP-3DSAM) to tackle these limitations. Specifically, we propose a Feature Enhanced Decoder that fuses the original image features with rich and detailed spatial information to enhance spatial features. We also design a Dual Attention Prompter to automatically obtain prompt information through Spatial Attention and Channel Attention. We conduct comprehensive experiments on four public abdominal tumor segmentation datasets. The results indicate that our DEAP-3DSAM achieves state-of-the-art performance in 3D image segmentation, outperforming or matching existing manual prompt methods. Furthermore, both quantitative and qualitative ablation studies confirm the effectiveness of our proposed modules.
>
---
#### [new 201] General vs Domain-Specific CNNs: Understanding Pretraining Effects on Brain MRI Tumor Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究脑肿瘤MRI分类任务，旨在比较领域特定与通用预训练CNN在小数据下的性能。通过公平对比RadImageNet DenseNet121、EfficientNetV2S和ConvNeXt-Tiny，发现通用模型在小数据下表现更优，表明领域特定预训练未必更有效。**

- **链接: [https://arxiv.org/pdf/2511.18326v1](https://arxiv.org/pdf/2511.18326v1)**

> **作者:** Helia Abedini; Saba Rahimi; Reza Vaziri
>
> **摘要:** Brain tumor detection from MRI scans plays a crucial role in early diagnosis and treatment planning. Deep convolutional neural networks (CNNs) have demonstrated strong performance in medical imaging tasks, particularly when pretrained on large datasets. However, it remains unclear which type of pretrained model performs better when only a small dataset is available: those trained on domain-specific medical data or those pretrained on large general datasets. In this study, we systematically evaluate three pretrained CNN architectures for brain tumor classification: RadImageNet DenseNet121 with medical-domain pretraining, EfficientNetV2S, and ConvNeXt-Tiny, which are modern general-purpose CNNs. All models were trained and fine-tuned under identical conditions using a limited-size brain MRI dataset to ensure a fair comparison. Our results reveal that ConvNeXt-Tiny achieved the highest accuracy, followed by EfficientNetV2S, while RadImageNet DenseNet121, despite being pretrained on domain-specific medical data, exhibited poor generalization with lower accuracy and higher loss. These findings suggest that domain-specific pretraining may not generalize well under small-data conditions. In contrast, modern, deeper general-purpose CNNs pretrained on large-scale datasets can offer superior transfer learning performance in specialized medical imaging tasks.
>
---
#### [new 202] Robust Physical Adversarial Patches Using Dynamically Optimized Clusters
- **分类: cs.CV**

- **简介: 该论文研究物理域对抗补丁的鲁棒性问题，针对尺度变化导致的插值模糊问题，提出基于SLIC动态聚类的超像素正则化方法。通过梯度回传优化超像素边界与颜色，生成尺度不变的对抗补丁，在数字与物理场景中均提升攻击性能。**

- **链接: [https://arxiv.org/pdf/2511.18656v1](https://arxiv.org/pdf/2511.18656v1)**

> **作者:** Harrison Bagley; Will Meakin; Simon Lucey; Yee Wei Law; Tat-Jun Chin
>
> **备注:** Supplementary material available at: https://drive.google.com/drive/folders/1Yntcc9CARdbvoJJ51cyUm1DWGSvU9X4V?usp=drive_link
>
> **摘要:** Physical adversarial attacks on deep learning systems is concerning due to the ease of deploying such attacks, usually by placing an adversarial patch in a scene to manipulate the outcomes of a deep learning model. Training such patches typically requires regularization that improves physical realizability (e.g., printability, smoothness) and/or robustness to real-world variability (e.g. deformations, viewing angle, noise). One type of variability that has received little attention is scale variability. When a patch is rescaled, either digitally through downsampling/upsampling or physically through changing imaging distances, interpolation-induced color mixing occurs. This smooths out pixel values, resulting in a loss of high-frequency patterns and degrading the adversarial signal. To address this, we present a novel superpixel-based regularization method that guides patch optimization to scale-resilient structures. Our ap proach employs the Simple Linear Iterative Clustering (SLIC) algorithm to dynamically cluster pixels in an adversarial patch during optimization. The Implicit Function Theorem is used to backpropagate gradients through SLIC to update the superpixel boundaries and color. This produces patches that maintain their structure over scale and are less susceptible to interpolation losses. Our method achieves greater performance in the digital domain, and when realized physically, these performance gains are preserved, leading to improved physical performance. Real-world performance was objectively assessed using a novel physical evaluation protocol that utilizes screens and cardboard cut-outs to systematically vary real-world conditions.
>
---
#### [new 203] Frequency-Adaptive Sharpness Regularization for Improving 3D Gaussian Splatting Generalization
- **分类: cs.CV**

- **简介: 该论文针对3D高斯点阵（3DGS）在少样本下泛化能力差的问题，提出频率自适应锐度正则化（FASR）。通过动态调整正则化强度以平衡细节保留与泛化性能，有效缓解了视角外重建中的伪影和细节丢失问题，显著提升了模型在新视角下的表现。**

- **链接: [https://arxiv.org/pdf/2511.17918v1](https://arxiv.org/pdf/2511.17918v1)**

> **作者:** Youngsik Yun; Dongjun Gu; Youngjung Uh
>
> **备注:** Project page: https://bbangsik13.github.io/FASR
>
> **摘要:** Despite 3D Gaussian Splatting (3DGS) excelling in most configurations, it lacks generalization across novel viewpoints in a few-shot scenario because it overfits to the sparse observations. We revisit 3DGS optimization from a machine learning perspective, framing novel view synthesis as a generalization problem to unseen viewpoints-an underexplored direction. We propose Frequency-Adaptive Sharpness Regularization (FASR), which reformulates the 3DGS training objective, thereby guiding 3DGS to converge toward a better generalization solution. Although Sharpness-Aware Minimization (SAM) similarly reduces the sharpness of the loss landscape to improve generalization of classification models, directly employing it to 3DGS is suboptimal due to the discrepancy between the tasks. Specifically, it hinders reconstructing high-frequency details due to excessive regularization, while reducing its strength leads to under-penalizing sharpness. To address this, we reflect the local frequency of images to set the regularization weight and the neighborhood radius when estimating the local sharpness. It prevents floater artifacts in novel viewpoints and reconstructs fine details that SAM tends to oversmooth. Across datasets with various configurations, our method consistently improves a wide range of baselines. Code will be available at https://bbangsik13.github.io/FASR.
>
---
#### [new 204] TSRE: Channel-Aware Typical Set Refinement for Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 该论文针对开放世界中机器学习模型的分布外（OOD）检测任务，提出TSRE方法。针对现有激活修正方法忽略通道特性与分布偏斜的问题，通过通道感知的典型集精炼与偏度校正，提升典型集估计精度，并利用修正后的激活计算能量分数，实现更准确的OOD检测。**

- **链接: [https://arxiv.org/pdf/2511.17636v1](https://arxiv.org/pdf/2511.17636v1)**

> **作者:** Weijun Gao; Rundong He; Jinyang Dong; Yongshun Gong
>
> **摘要:** Out-of-Distribution (OOD) detection is a critical capability for ensuring the safe deployment of machine learning models in open-world environments, where unexpected or anomalous inputs can compromise model reliability and performance. Activation-based methods play a fundamental role in OOD detection by mitigating anomalous activations and enhancing the separation between in-distribution (ID) and OOD data. However, existing methods apply activation rectification while often overlooking channel's intrinsic characteristics and distributional skewness, which results in inaccurate typical set estimation. This discrepancy can lead to the improper inclusion of anomalous activations across channels. To address this limitation, we propose a typical set refinement method based on discriminability and activity, which rectifies activations into a channel-aware typical set. Furthermore, we introduce a skewness-based refinement to mitigate distributional bias in typical set estimation. Finally, we leverage the rectified activations to compute the energy score for OOD detection. Experiments on the ImageNet-1K and CIFAR-100 benchmarks demonstrate that our method achieves state-of-the-art performance and generalizes effectively across backbones and score functions.
>
---
#### [new 205] Leveraging Metaheuristic Approaches to Improve Deep Learning Systems for Anxiety Disorder Detection
- **分类: cs.CV**

- **简介: 该论文属于焦虑障碍自动检测任务，旨在解决传统主观评估耗时且不一致的问题。通过融合深度学习与群体智能优化算法（如遗传算法、粒子群优化），利用可穿戴设备多模态数据优化特征与超参数，提升检测准确率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18827v1](https://arxiv.org/pdf/2511.18827v1)**

> **作者:** Mohammadreza Amiri; Monireh Hosseini
>
> **备注:** 12 pages
>
> **摘要:** Despite being among the most common psychological disorders, anxiety-related conditions are still primarily identified through subjective assessments, such as clinical interviews and self-evaluation questionnaires. These conventional methods often require significant time and may vary depending on the evaluator. However, the emergence of advanced artificial intelligence techniques has created new opportunities for detecting anxiety in a more consistent and automated manner. To address the limitations of traditional approaches, this study introduces a comprehensive model that integrates deep learning architectures with optimization strategies inspired by swarm intelligence. Using multimodal and wearable-sensor datasets, the framework analyzes physiological, emotional, and behavioral signals. Swarm intelligence techniques including genetic algorithms and particle swarm optimization are incorporated to refine the feature space and optimize hyperparameters. Meanwhile, deep learning components are tasked with deriving layered and discriminative representations from sequential, multi-source inputs. Our evaluation shows that the fusion of these two computational paradigms significantly enhances detection performance compared with using deep networks alone. The hybrid model achieves notable improvements in accuracy and demonstrates stronger generalization across various individuals. Overall, the results highlight the potential of combining metaheuristic optimization with deep learning to develop scalable, objective, and clinically meaningful solutions for assessing anxiety disorders
>
---
#### [new 206] SAM3-Adapter: Efficient Adaptation of Segment Anything 3 for Camouflage Object Segmentation, Shadow Detection, and Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对图像分割任务中复杂场景（如伪装物体、阴影、医学图像）的细粒度分割难题，提出SAM3-Adapter框架。通过适配新一代Segment Anything 3模型，显著提升分割精度与效率，实现对多种低级视觉任务的高性能适应，超越此前所有SAM基线方法。**

- **链接: [https://arxiv.org/pdf/2511.19425v1](https://arxiv.org/pdf/2511.19425v1)**

> **作者:** Tianrun Chen; Runlong Cao; Xinda Yu; Lanyun Zhu; Chaotao Ding; Deyi Ji; Cheng Chen; Qi Zhu; Chunyan Xu; Papa Mao; Ying Zang
>
> **摘要:** The rapid rise of large-scale foundation models has reshaped the landscape of image segmentation, with models such as Segment Anything achieving unprecedented versatility across diverse vision tasks. However, previous generations-including SAM and its successor-still struggle with fine-grained, low-level segmentation challenges such as camouflaged object detection, medical image segmentation, cell image segmentation, and shadow detection. To address these limitations, we originally proposed SAM-Adapter in 2023, demonstrating substantial gains on these difficult scenarios. With the emergence of Segment Anything 3 (SAM3)-a more efficient and higher-performing evolution with a redesigned architecture and improved training pipeline-we revisit these long-standing challenges. In this work, we present SAM3-Adapter, the first adapter framework tailored for SAM3 that unlocks its full segmentation capability. SAM3-Adapter not only reduces computational overhead but also consistently surpasses both SAM and SAM2-based solutions, establishing new state-of-the-art results across multiple downstream tasks, including medical imaging, camouflaged (concealed) object segmentation, and shadow detection. Built upon the modular and composable design philosophy of the original SAM-Adapter, SAM3-Adapter provides stronger generalizability, richer task adaptability, and significantly improved segmentation precision. Extensive experiments confirm that integrating SAM3 with our adapter yields superior accuracy, robustness, and efficiency compared to all prior SAM-based adaptations. We hope SAM3-Adapter can serve as a foundation for future research and practical segmentation applications. Code, pre-trained models, and data processing pipelines are available.
>
---
#### [new 207] VITAL: Vision-Encoder-centered Pre-training for LMMs in Visual Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉质量评估（VQualA）任务，解决现有大模型泛化能力差、依赖全参数微调的问题。提出以视觉编码器为中心的生成式预训练框架VITAL，构建450万级数据集，实现多任务协同训练，并通过轻量解码器快速适配，显著提升模型的通用性与可迁移性。**

- **链接: [https://arxiv.org/pdf/2511.17962v1](https://arxiv.org/pdf/2511.17962v1)**

> **作者:** Ziheng Jia; Linhan Cao; Jinliang Han; Zicheng Zhang; Jiaying Qian; Jiarui Wang; Zijian Chen; Guangtao Zhai; Xiongkuo Min
>
> **摘要:** Developing a robust visual quality assessment (VQualA) large multi-modal model (LMM) requires achieving versatility, powerfulness, and transferability. However, existing VQualA LMMs typically focus on a single task and rely on full-parameter fine-tuning, which makes them prone to overfitting on specific modalities or task types, thereby limiting their generalization capacity and transferability. To address this, we propose a vision-encoder-centered generative pre-training pipeline and develop the VITAL-Series LMMs. (1) We adopt a machine-executed annotation-scrutiny paradigm, constructing over 4.5M vision-language (VL) pairs-the largest VQualA training dataset to date. (2) We employ a multi-task training workflow that simultaneously enhances the model's quantitative scoring precision and strengthens its capability for quality interpretation across both image and video modalities. (3) Building upon the vision encoder, we realize an efficient model zoo extension: the model zoo exhibits strong zero-shot performance, and each paired decoder requires only a swift warm-up using less than 1/1000 of the pre-training data to achieve performance comparable to the fully trained counterpart. Overall, our work lays a cornerstone for advancing toward the foundation LMM for VQualA.
>
---
#### [new 208] MGA-VQA: Secure and Interpretable Graph-Augmented Visual Question Answering with Memory-Guided Protection Against Unauthorized Knowledge Use
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文档视觉问答（DocVQA）任务，解决现有方法在空间关系建模、多跳推理和可解释性方面的不足。提出MGA-VQA框架，融合图结构推理与记忆增强机制，实现高效精准的问答与空间定位，提升模型透明度与性能。**

- **链接: [https://arxiv.org/pdf/2511.17881v1](https://arxiv.org/pdf/2511.17881v1)**

> **作者:** Ahmad Mohammadshirazi; Pinaki Prasad Guha Neogi; Dheeraj Kulshrestha; Rajiv Ramnath
>
> **摘要:** Document Visual Question Answering (DocVQA) requires models to jointly understand textual semantics, spatial layout, and visual features. Current methods struggle with explicit spatial relationship modeling, inefficiency with high-resolution documents, multi-hop reasoning, and limited interpretability. We propose MGA-VQA, a multi-modal framework that integrates token-level encoding, spatial graph reasoning, memory-augmented inference, and question-guided compression. Unlike prior black-box models, MGA-VQA introduces interpretable graph-based decision pathways and structured memory access for enhanced reasoning transparency. Evaluation across six benchmarks (FUNSD, CORD, SROIE, DocVQA, STE-VQA, and RICO) demonstrates superior accuracy and efficiency, with consistent improvements in both answer prediction and spatial localization.
>
---
#### [new 209] HiFi-MambaV2: Hierarchical Shared-Routed MoE for High-Fidelity MRI Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对磁共振成像（MRI）重建任务，解决 undersampled k-space 数据中高频细节恢复与解剖结构一致性难题。提出 HiFi-MambaV2，通过分层共享路由的 MoE 架构与频率一致的拉普拉斯金字塔，实现内容自适应计算与稳定跨深度特征融合，显著提升重建质量。**

- **链接: [https://arxiv.org/pdf/2511.18534v1](https://arxiv.org/pdf/2511.18534v1)**

> **作者:** Pengcheng Fang; Hongli Chen; Guangzhen Yao; Jian Shi; Fangfang Tang; Xiaohao Cai; Shanshan Shan; Feng Liu
>
> **摘要:** Reconstructing high-fidelity MR images from undersampled k-space data requires recovering high-frequency details while maintaining anatomical coherence. We present HiFi-MambaV2, a hierarchical shared-routed Mixture-of-Experts (MoE) Mamba architecture that couples frequency decomposition with content-adaptive computation. The model comprises two core components: (i) a separable frequency-consistent Laplacian pyramid (SF-Lap) that delivers alias-resistant, stable low- and high-frequency streams; and (ii) a hierarchical shared-routed MoE that performs per-pixel top-1 sparse dispatch to shared experts and local routers, enabling effective specialization with stable cross-depth behavior. A lightweight global context path is fused into an unrolled, data-consistency-regularized backbone to reinforce long-range reasoning and preserve anatomical coherence. Evaluated on fastMRI, CC359, ACDC, M4Raw, and Prostate158, HiFi-MambaV2 consistently outperforms CNN-, Transformer-, and prior Mamba-based baselines in PSNR, SSIM, and NMSE across single- and multi-coil settings and multiple acceleration factors, consistently surpassing consistent improvements in high-frequency detail and overall structural fidelity. These results demonstrate that HiFi-MambaV2 enables reliable and robust MRI reconstruction.
>
---
#### [new 210] TPG-INR: Target Prior-Guided Implicit 3D CT Reconstruction for Enhanced Sparse-view Imaging
- **分类: cs.CV**

- **简介: 该论文针对稀疏视图三维CT重建任务，解决现有隐式方法因忽略解剖先验导致的重建精度与效率低下问题。提出TPG-INR框架，利用目标投影数据构建先验，指导体素采样与结构编码，提升学习效率与重建质量。实验表明其在稀疏视图下显著优于现有模型。**

- **链接: [https://arxiv.org/pdf/2511.18806v1](https://arxiv.org/pdf/2511.18806v1)**

> **作者:** Qinglei Cao; Ziyao Tang; Xiaoqin Tang
>
> **备注:** Please consider this version as the latest camera-ready version
>
> **摘要:** X-ray imaging, based on penetration, enables detailed visualization of internal structures. Building on this capability, existing implicit 3D reconstruction methods have adapted the NeRF model and its variants for internal CT reconstruction. However, these approaches often neglect the significance of objects' anatomical priors for implicit learning, limiting both reconstruction precision and learning efficiency, particularly in ultra-sparse view scenarios. To address these challenges, we propose a novel 3D CT reconstruction framework that employs a 'target prior' derived from the object's projection data to enhance implicit learning. Our approach integrates positional and structural encoding to facilitate voxel-wise implicit reconstruction, utilizing the target prior to guide voxel sampling and enrich structural encoding. This dual strategy significantly boosts both learning efficiency and reconstruction quality. Additionally, we introduce a CUDA-based algorithm for rapid estimation of high-quality 3D target priors from sparse-view projections. Experiments utilizing projection data from a complex abdominal dataset demonstrate that the proposed model substantially enhances learning efficiency, outperforming the current leading model, NAF, by a factor of ten. In terms of reconstruction quality, it also exceeds the most accurate model, NeRP, achieving PSNR improvements of 3.57 dB, 5.42 dB, and 5.70 dB with 10, 20, and 30 projections, respectively. The code is available at https://github.com/qlcao171/TPG-INR.
>
---
#### [new 211] RNN as Linear Transformer: A Closer Investigation into Representational Potentials of Visual Mamba Models
- **分类: cs.CV**

- **简介: 该论文研究Mamba在视觉任务中的表征能力，旨在揭示其机制。针对Mamba在视觉领域理解不足的问题，提出理论分析、新分割评估指标，并通过自监督预训练提升可解释性。实验表明其具备建模长程依赖的能力，线性探针在ImageNet上达78.5%准确率。**

- **链接: [https://arxiv.org/pdf/2511.18380v1](https://arxiv.org/pdf/2511.18380v1)**

> **作者:** Timing Yang; Guoyizhe Wei; Alan Yuille; Feng Wang
>
> **摘要:** Mamba has recently garnered attention as an effective backbone for vision tasks. However, its underlying mechanism in visual domains remains poorly understood. In this work, we systematically investigate Mamba's representational properties and make three primary contributions. First, we theoretically analyze Mamba's relationship to Softmax and Linear Attention, confirming that it can be viewed as a low-rank approximation of Softmax Attention and thereby bridging the representational gap between Softmax and Linear forms. Second, we introduce a novel binary segmentation metric for activation map evaluation, extending qualitative assessments to a quantitative measure that demonstrates Mamba's capacity to model long-range dependencies. Third, by leveraging DINO for self-supervised pretraining, we obtain clearer activation maps than those produced by standard supervised approaches, highlighting Mamba's potential for interpretability. Notably, our model also achieves a 78.5 percent linear probing accuracy on ImageNet, underscoring its strong performance. We hope this work can provide valuable insights for future investigations of Mamba-based vision architectures.
>
---
#### [new 212] Understanding, Accelerating, and Improving MeanFlow Training
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究生成模型中的MeanFlow训练机制，旨在提升少步生成质量与训练效率。通过分析瞬时与平均速度场的交互关系，发现前者是后者学习的基础，且长间隔平均速度学习依赖于前期精确的小间隔学习。据此提出分阶段训练策略，先加速瞬时速度学习，再聚焦长间隔平均速度，显著提升收敛速度与生成效果。**

- **链接: [https://arxiv.org/pdf/2511.19065v1](https://arxiv.org/pdf/2511.19065v1)**

> **作者:** Jin-Young Kim; Hyojun Go; Lea Bogensperger; Julius Erbach; Nikolai Kalischek; Federico Tombari; Konrad Schindler; Dominik Narnhofer
>
> **摘要:** MeanFlow promises high-quality generative modeling in few steps, by jointly learning instantaneous and average velocity fields. Yet, the underlying training dynamics remain unclear. We analyze the interaction between the two velocities and find: (i) well-established instantaneous velocity is a prerequisite for learning average velocity; (ii) learning of instantaneous velocity benefits from average velocity when the temporal gap is small, but degrades as the gap increases; and (iii) task-affinity analysis indicates that smooth learning of large-gap average velocities, essential for one-step generation, depends on the prior formation of accurate instantaneous and small-gap average velocities. Guided by these observations, we design an effective training scheme that accelerates the formation of instantaneous velocity, then shifts emphasis from short- to long-interval average velocity. Our enhanced MeanFlow training yields faster convergence and significantly better few-step generation: With the same DiT-XL backbone, our method reaches an impressive FID of 2.87 on 1-NFE ImageNet 256x256, compared to 3.43 for the conventional MeanFlow baseline. Alternatively, our method matches the performance of the MeanFlow baseline with 2.5x shorter training time, or with a smaller DiT-L backbone.
>
---
#### [new 213] HEAL: Learning-Free Source Free Unsupervised Domain Adaptation for Cross-Modality Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对跨模态医学图像分割中的源域无监督域自适应问题，提出无需学习的HEAL框架。通过层次去噪、边缘引导选择、尺寸感知融合等技术，在无源数据和目标标签的情况下，有效缓解域偏移，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.17958v1](https://arxiv.org/pdf/2511.17958v1)**

> **作者:** Yulong Shi; Jiapeng Li; Lin Qi
>
> **备注:** Accepted by The 36th British Machine Vision Conference (BMVC 2025)
>
> **摘要:** Growing demands for clinical data privacy and storage constraints have spurred advances in Source Free Unsupervised Domain Adaptation (SFUDA). SFUDA addresses the domain shift by adapting models from the source domain to the unseen target domain without accessing source data, even when target-domain labels are unavailable. However, SFUDA faces significant challenges: the absence of source domain data and label supervision in the target domain due to source free and unsupervised settings. To address these issues, we propose HEAL, a novel SFUDA framework that integrates Hierarchical denoising, Edge-guided selection, size-Aware fusion, and Learning-free characteristic. Large-scale cross-modality experiments demonstrate that our method outperforms existing SFUDA approaches, achieving state-of-the-art (SOTA) performance. The source code is publicly available at: https://github.com/derekshiii/HEAL.
>
---
#### [new 214] An Anatomy Aware Hybrid Deep Learning Framework for Lung Cancer Tumor Stage Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对肺癌肿瘤分期任务，解决深度学习模型忽视解剖信息的问题。提出一种融合解剖先验的混合框架，通过精准分割肺部结构，量化肿瘤大小与邻近解剖距离，结合临床规则进行分期，实现91.36%准确率与可解释性决策支持。**

- **链接: [https://arxiv.org/pdf/2511.19367v1](https://arxiv.org/pdf/2511.19367v1)**

> **作者:** Saniah Kayenat Chowdhury; Rusab Sarmun; Muhammad E. H. Chowdhury; Sohaib Bassam Zoghoul; Israa Al-Hashimi; Adam Mushtak; Amith Khandakar
>
> **摘要:** Accurate lung cancer tumor staging is crucial for prognosis and treatment planning. However, it remains challenging for end-to-end deep learning approaches, as such approaches often overlook spatial and anatomical information that are central to the tumor-node-metastasis system. The tumor stage depends on multiple quantitative criteria, including the tumor size and its proximity to the nearest anatomical structures, and small variations can alter the staging outcome. We propose a medically grounded hybrid pipeline that performs staging by explicitly measuring the tumor's size and distance properties rather than treating it as a pure image classification task. Our method employs specialized encoder-decoder networks to precisely segment the lung and adjacent anatomy, including the lobes, tumor, mediastinum, and diaphragm. Subsequently, we extract the necessary tumor properties, i.e. measure the largest tumor dimension and calculate the distance between the tumor and neighboring anatomical structures by a quantitative analysis of the segmentation masks. Finally, we apply rule-based tumor staging aligned with the medical guidelines. This novel framework has been evaluated on the Lung-PET-CT-Dx dataset, demonstrating superior performance compared to traditional deep learning models, achieving an overall classification accuracy of 91.36%. We report the per-stage F1-scores of 0.93 (T1), 0.89 (T2), 0.96 (T3), and 0.90 (T4), a critical evaluation aspect often omitted in prior literature. To our knowledge, this is the first study that embeds explicit clinical context into tumor stage classification. Unlike standard convolutional neural networks that operate in an uninterpretable "black box" manner, our method offers both state-of-the-art performance and transparent decision support.
>
---
#### [new 215] nnActive: A Framework for Evaluation of Active Learning in 3D Biomedical Segmentation
- **分类: cs.CV**

- **简介: 该论文针对3D生物医学图像分割中的标注成本高问题，提出nnActive框架。通过大尺度实验、部分标注训练、前景感知随机采样及前景效率度量，克服了现有主动学习评估的四大缺陷，揭示了主动学习在特定条件下的有效性与局限性。**

- **链接: [https://arxiv.org/pdf/2511.19183v1](https://arxiv.org/pdf/2511.19183v1)**

> **作者:** Carsten T. Lüth; Jeremias Traub; Kim-Celine Kahl; Till J. Bungert; Lukas Klein; Lars Krämer; Paul F. Jaeger; Fabian Isensee; Klaus Maier-Hein
>
> **备注:** Accepted at TMLR
>
> **摘要:** Semantic segmentation is crucial for various biomedical applications, yet its reliance on large annotated datasets presents a bottleneck due to the high cost and specialized expertise required for manual labeling. Active Learning (AL) aims to mitigate this challenge by querying only the most informative samples, thereby reducing annotation effort. However, in the domain of 3D biomedical imaging, there is no consensus on whether AL consistently outperforms Random sampling. Four evaluation pitfalls hinder the current methodological assessment. These are (1) restriction to too few datasets and annotation budgets, (2) using 2D models on 3D images without partial annotations, (3) Random baseline not being adapted to the task, and (4) measuring annotation cost only in voxels. In this work, we introduce nnActive, an open-source AL framework that overcomes these pitfalls by (1) means of a large scale study spanning four biomedical imaging datasets and three label regimes, (2) extending nnU-Net by using partial annotations for training with 3D patch-based query selection, (3) proposing Foreground Aware Random sampling strategies tackling the foreground-background class imbalance of medical images and (4) propose the foreground efficiency metric, which captures the low annotation cost of background-regions. We reveal the following findings: (A) while all AL methods outperform standard Random sampling, none reliably surpasses an improved Foreground Aware Random sampling; (B) benefits of AL depend on task specific parameters; (C) Predictive Entropy is overall the best performing AL method, but likely requires the most annotation effort; (D) AL performance can be improved with more compute intensive design choices. As a holistic, open-source framework, nnActive can serve as a catalyst for research and application of AL in 3D biomedical imaging. Code is at: https://github.com/MIC-DKFZ/nnActive
>
---
#### [new 216] Graph-based 3D Human Pose Estimation using WiFi Signals
- **分类: cs.CV**

- **简介: 该论文针对WiFi信号进行3D人体姿态估计，解决现有方法忽略关节拓扑关系的问题。提出GraphPose-Fi框架，利用图卷积网络和自注意力机制建模骨骼结构，结合多天线时频特征提取，显著提升估计精度。**

- **链接: [https://arxiv.org/pdf/2511.19105v1](https://arxiv.org/pdf/2511.19105v1)**

> **作者:** Jichao Chen; YangYang Qu; Ruibo Tang; Dirk Slock
>
> **摘要:** WiFi-based human pose estimation (HPE) has attracted increasing attention due to its resilience to occlusion and privacy-preserving compared to camera-based methods. However, existing WiFi-based HPE approaches often employ regression networks that directly map WiFi channel state information (CSI) to 3D joint coordinates, ignoring the inherent topological relationships among human joints. In this paper, we present GraphPose-Fi, a graph-based framework that explicitly models skeletal topology for WiFi-based 3D HPE. Our framework comprises a CNN encoder shared across antennas for subcarrier-time feature extraction, a lightweight attention module that adaptively reweights features over time and across antennas, and a graph-based regression head that combines GCN layers with self-attention to capture local topology and global dependencies. Our proposed method significantly outperforms existing methods on the MM-Fi dataset in various settings. The source code is available at: https://github.com/Cirrick/GraphPose-Fi.
>
---
#### [new 217] Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Yo'City，一种用于个性化、无限扩展的3D真实城市场景生成框架。针对现有方法依赖单一扩散模型、难以实现个性化与无边界生成的问题，提出基于大模型的“自上而下”规划与迭代优化机制，通过城市-区域-网格层级结构与交互式扩展，实现高质量、空间连贯的城市生成。**

- **链接: [https://arxiv.org/pdf/2511.18734v1](https://arxiv.org/pdf/2511.18734v1)**

> **作者:** Keyang Lu; Sifan Zhou; Hongbin Xu; Gang Xu; Zhifei Yang; Yikai Wang; Zhen Xiao; Jieyi Long; Ming Li
>
> **备注:** 22 pages, 16 figures
>
> **摘要:** Realistic 3D city generation is fundamental to a wide range of applications, including virtual reality and digital twins. However, most existing methods rely on training a single diffusion model, which limits their ability to generate personalized and boundless city-scale scenes. In this paper, we present Yo'City, a novel agentic framework that enables user-customized and infinitely expandable 3D city generation by leveraging the reasoning and compositional capabilities of off-the-shelf large models. Specifically, Yo'City first conceptualize the city through a top-down planning strategy that defines a hierarchical "City-District-Grid" structure. The Global Planner determines the overall layout and potential functional districts, while the Local Designer further refines each district with detailed grid-level descriptions. Subsequently, the grid-level 3D generation is achieved through a "produce-refine-evaluate" isometric image synthesis loop, followed by image-to-3D generation. To simulate continuous city evolution, Yo'City further introduces a user-interactive, relationship-guided expansion mechanism, which performs scene graph-based distance- and semantics-aware layout optimization, ensuring spatially coherent city growth. To comprehensively evaluate our method, we construct a diverse benchmark dataset and design six multi-dimensional metrics that assess generation quality from the perspectives of semantics, geometry, texture, and layout. Extensive experiments demonstrate that Yo'City consistently outperforms existing state-of-the-art methods across all evaluation aspects.
>
---
#### [new 218] Efficient Score Pre-computation for Diffusion Models via Cross-Matrix Krylov Projection
- **分类: cs.CV**

- **简介: 该论文针对扩散模型推理效率低的问题，提出基于交叉矩阵Krylov投影的评分预计算方法。通过将模型转为福克-普朗克形式，利用共享子空间加速大规模线性系统求解，显著降低计算开销，在保持图像质量的同时实现高达115倍的加速，适用于资源受限场景。**

- **链接: [https://arxiv.org/pdf/2511.17634v1](https://arxiv.org/pdf/2511.17634v1)**

> **作者:** Kaikwan Lau; Andrew S. Na; Justin W. L. Wan
>
> **摘要:** This paper presents a novel framework to accelerate score-based diffusion models. It first converts the standard stable diffusion model into the Fokker-Planck formulation which results in solving large linear systems for each image. For training involving many images, it can lead to a high computational cost. The core innovation is a cross-matrix Krylov projection method that exploits mathematical similarities between matrices, using a shared subspace built from ``seed" matrices to rapidly solve for subsequent ``target" matrices. Our experiments show that this technique achieves a 15.8\% to 43.7\% time reduction over standard sparse solvers. Additionally, we compare our method against DDPM baselines in denoising tasks, showing a speedup of up to 115$\times$. Furthermore, under a fixed computational budget, our model is able to produce high-quality images while DDPM fails to generate recognizable content, illustrating our approach is a practical method for efficient generation in resource-limited settings.
>
---
#### [new 219] DynaMix: Generalizable Person Re-identification via Dynamic Relabeling and Mixed Data Sampling
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对通用行人重识别任务，解决跨摄像头、跨环境识别中标注数据稀缺的问题。提出DynaMix方法，通过动态重标注、高效中心点维护和混合数据采样，融合少量标注与大量伪标注数据，实现大规模高效训练，显著提升泛化性能。**

- **链接: [https://arxiv.org/pdf/2511.19067v1](https://arxiv.org/pdf/2511.19067v1)**

> **作者:** Timur Mamedov; Anton Konushin; Vadim Konushin
>
> **摘要:** Generalizable person re-identification (Re-ID) aims to recognize individuals across unseen cameras and environments. While existing methods rely heavily on limited labeled multi-camera data, we propose DynaMix, a novel method that effectively combines manually labeled multi-camera and large-scale pseudo-labeled single-camera data. Unlike prior works, DynaMix dynamically adapts to the structure and noise of the training data through three core components: (1) a Relabeling Module that refines pseudo-labels of single-camera identities on-the-fly; (2) an Efficient Centroids Module that maintains robust identity representations under a large identity space; and (3) a Data Sampling Module that carefully composes mixed data mini-batches to balance learning complexity and intra-batch diversity. All components are specifically designed to operate efficiently at scale, enabling effective training on millions of images and hundreds of thousands of identities. Extensive experiments demonstrate that DynaMix consistently outperforms state-of-the-art methods in generalizable person Re-ID.
>
---
#### [new 220] DualGazeNet: A Biologically Inspired Dual-Gaze Query Network for Salient Object Detection
- **分类: cs.CV**

- **简介: 该论文针对显著性目标检测任务，提出基于生物视觉机制的DualGazeNet。它通过模拟人类双路径视觉处理与皮层注意调节，构建轻量级纯Transformer框架，有效解决现有方法因架构复杂导致的特征冗余与性能瓶颈问题，实现高效、精准且可解释的检测。**

- **链接: [https://arxiv.org/pdf/2511.18865v1](https://arxiv.org/pdf/2511.18865v1)**

> **作者:** Yu Zhang; Haoan Ping; Yuchen Li; Zhenshan Bing; Fuchun Sun; Alois Knoll
>
> **摘要:** Recent salient object detection (SOD) methods aim to improve performance in four key directions: semantic enhancement, boundary refinement, auxiliary task supervision, and multi-modal fusion. In pursuit of continuous gains, these approaches have evolved toward increasingly sophisticated architectures with multi-stage pipelines, specialized fusion modules, edge-guided learning, and elaborate attention mechanisms. However, this complexity paradoxically introduces feature redundancy and cross-component interference that obscure salient cues, ultimately reaching performance bottlenecks. In contrast, human vision achieves efficient salient object identification without such architectural complexity. This contrast raises a fundamental question: can we design a biologically grounded yet architecturally simple SOD framework that dispenses with most of this engineering complexity, while achieving state-of-the-art accuracy, computational efficiency, and interpretability? In this work, we answer this question affirmatively by introducing DualGazeNet, a biologically inspired pure Transformer framework that models the dual biological principles of robust representation learning and magnocellular-parvocellular dual-pathway processing with cortical attention modulation in the human visual system. Extensive experiments on five RGB SOD benchmarks show that DualGazeNet consistently surpasses 25 state-of-the-art CNN- and Transformer-based methods. On average, DualGazeNet achieves about 60\% higher inference speed and 53.4\% fewer FLOPs than four Transformer-based baselines of similar capacity (VST++, MDSAM, Sam2unet, and BiRefNet). Moreover, DualGazeNet exhibits strong cross-domain generalization, achieving leading or highly competitive performance on camouflaged and underwater SOD benchmarks without relying on additional modalities.
>
---
#### [new 221] CADTrack: Learning Contextual Aggregation with Deformable Alignment for Robust RGBT Tracking
- **分类: cs.CV**

- **简介: 该论文针对RGBT跟踪任务，解决模态差异导致的特征表示不稳健问题。提出CADTrack框架，通过MFI模块实现高效特征交互，CAM模块动态融合跨层上下文信息，DAM模块缓解空间错位与定位漂移，提升复杂场景下的跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.17967v1](https://arxiv.org/pdf/2511.17967v1)**

> **作者:** Hao Li; Yuhao Wang; Xiantao Hu; Wenning Hao; Pingping Zhang; Dong Wang; Huchuan Lu
>
> **备注:** Accepted by AAAI2026. More modifications may be performed
>
> **摘要:** RGB-Thermal (RGBT) tracking aims to exploit visible and thermal infrared modalities for robust all-weather object tracking. However, existing RGBT trackers struggle to resolve modality discrepancies, which poses great challenges for robust feature representation. This limitation hinders effective cross-modal information propagation and fusion, which significantly reduces the tracking accuracy. To address this limitation, we propose a novel Contextual Aggregation with Deformable Alignment framework called CADTrack for RGBT Tracking. To be specific, we first deploy the Mamba-based Feature Interaction (MFI) that establishes efficient feature interaction via state space models. This interaction module can operate with linear complexity, reducing computational cost and improving feature discrimination. Then, we propose the Contextual Aggregation Module (CAM) that dynamically activates backbone layers through sparse gating based on the Mixture-of-Experts (MoE). This module can encode complementary contextual information from cross-layer features. Finally, we propose the Deformable Alignment Module (DAM) to integrate deformable sampling and temporal propagation, mitigating spatial misalignment and localization drift. With the above components, our CADTrack achieves robust and accurate tracking in complex scenarios. Extensive experiments on five RGBT tracking benchmarks verify the effectiveness of our proposed method. The source code is released at https://github.com/IdolLab/CADTrack.
>
---
#### [new 222] Edit2Perceive: Image Editing Diffusion Models Are Strong Dense Perceivers
- **分类: cs.CV**

- **简介: 该论文提出Edit2Perceive，一个基于图像编辑扩散模型的统一框架，用于深度、法线和抠图等密集感知任务。针对传统文本到图像生成模型在结构一致性上的不足，利用编辑模型的图像到图像一致性优势，通过全参数微调与像素空间一致性损失，实现高效、高精度的结构保持推理，在小数据下达到顶尖性能。**

- **链接: [https://arxiv.org/pdf/2511.18673v1](https://arxiv.org/pdf/2511.18673v1)**

> **作者:** Yiqing Shi; Yiren Song; Mike Zheng Shou
>
> **摘要:** Recent advances in diffusion transformers have shown remarkable generalization in visual synthesis, yet most dense perception methods still rely on text-to-image (T2I) generators designed for stochastic generation. We revisit this paradigm and show that image editing diffusion models are inherently image-to-image consistent, providing a more suitable foundation for dense perception task. We introduce Edit2Perceive, a unified diffusion framework that adapts editing models for depth, normal, and matting. Built upon the FLUX.1 Kontext architecture, our approach employs full-parameter fine-tuning and a pixel-space consistency loss to enforce structure-preserving refinement across intermediate denoising states. Moreover, our single-step deterministic inference yields up to faster runtime while training on relatively small datasets. Extensive experiments demonstrate comprehensive state-of-the-art results across all three tasks, revealing the strong potential of editing-oriented diffusion transformers for geometry-aware perception.
>
---
#### [new 223] When Generative Replay Meets Evolving Deepfakes: Domain-Aware Relative Weighting for Incremental Face Forgery Detection
- **分类: cs.CV**

- **简介: 该论文针对增量式人脸伪造检测中样本重放的多样性与隐私问题，提出基于生成重放的领域感知相对权重策略（DARW）。通过区分安全与风险生成样本，动态调整监督强度，有效缓解域重叠带来的混淆，提升模型在新伪造方法下的检测性能。**

- **链接: [https://arxiv.org/pdf/2511.18436v1](https://arxiv.org/pdf/2511.18436v1)**

> **作者:** Hao Shen; Jikang Cheng; Renye Yan; Zhongyuan Wang; Wei Peng; Baojin Huang
>
> **摘要:** The rapid advancement of face generation techniques has led to a growing variety of forgery methods. Incremental forgery detection aims to gradually update existing models with new forgery data, yet current sample replay-based methods are limited by low diversity and privacy concerns. Generative replay offers a potential solution by synthesizing past data, but its feasibility for forgery detection remains unclear. In this work, we systematically investigate generative replay and identify two scenarios: when the replay generator closely resembles the new forgery model, generated real samples blur the domain boundary, creating domain-risky samples; when the replay generator differs significantly, generated samples can be safely supervised, forming domain-safe samples. To exploit generative replay effectively, we propose a novel Domain-Aware Relative Weighting (DARW) strategy. DARW directly supervises domain-safe samples while applying a Relative Separation Loss to balance supervision and potential confusion for domain-risky samples. A Domain Confusion Score dynamically adjusts this tradeoff according to sample reliability. Extensive experiments demonstrate that DARW consistently improves incremental learning performance for forgery detection under different generative replay settings and alleviates the adverse impact of domain overlap.
>
---
#### [new 224] Foundational Question Generation for Video Question Answering via an Embedding-Integrated Approach
- **分类: cs.CV**

- **简介: 该论文针对视频问答（VQA）中因标注数据偏事件中心而缺乏场景基础信息的问题，提出FIQ框架。通过生成包含物体类别、空间布局等核心属性的问答对，增强模型对视频场景的全面理解。引入VQ-CAlign模块对齐问题与视觉特征，提升推理能力。实验表明，该方法在SUTD-TrafficQA上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2511.17618v1](https://arxiv.org/pdf/2511.17618v1)**

> **作者:** Ju-Young Oh
>
> **备注:** [Master's thesis, Korea University, 2025]
>
> **摘要:** Conventional VQA approaches primarily rely on question-answer (Q&A) pairs to learn the spatio-temporal dynamics of video content. However, most existing annotations are event-centric, which restricts the model's ability to capture the comprehensive context of a scene. The lack of fundamental information such as object categories, spatial configurations, and descriptive visual attributes prevents the model from forming a complete understanding of the environment, ultimately limiting its generalization and reasoning capability. In this paper, we introduce Foundational Question Generation for Video Question Answering via an Embedding-Integrated Approach (FIQ), a framework designed to enhance the reasoning capability of VQA models by improving their foundational comprehension of video content. FIQ generates Q&A pairs from descriptive information extracted directly from videos, thereby enriching the dataset with core scene-level attributes. These generated pairs help the model develop a more holistic understanding of the video, leading to improved generalizability and reasoning performance. In addition, we propose a VQ-CAlign module that aligns task-specific question embeddings with corresponding visual features, preserving essential contextual cues and enhancing adaptability to downstream tasks. Experimental results on the SUTD-TrafficQA dataset demonstrate that FIQ achieves state-of-the-art performance, surpassing existing baseline approaches.
>
---
#### [new 225] Stage-Specific Benchmarking of Deep Learning Models for Glioblastoma Follow-Up MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对胶质母细胞瘤放疗后随访MRI中真进展（TP）与假进展（PsP）的鉴别难题，提出首个分阶段的深度学习模型基准测试。基于180例患者数据，对比多种模型在不同时间点的表现，发现后期判别能力提升，Mamba+CNN混合模型最优，但整体区分度仍有限，强调需结合纵向数据与更大样本。**

- **链接: [https://arxiv.org/pdf/2511.18595v1](https://arxiv.org/pdf/2511.18595v1)**

> **作者:** Wenhao Guo; Golrokh Mirzaei
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Differentiating true tumor progression (TP) from treatment-related pseudoprogression (PsP) in glioblastoma remains challenging, especially at early follow-up. We present the first stage-specific, cross-sectional benchmarking of deep learning models for follow-up MRI using the Burdenko GBM Progression cohort (n = 180). We analyze different post-RT scans independently to test whether architecture performance depends on time-point. Eleven representative DL families (CNNs, LSTMs, hybrids, transformers, and selective state-space models) were trained under a unified, QC-driven pipeline with patient-level cross-validation. Across both stages, accuracies were comparable (~0.70-0.74), but discrimination improved at the second follow-up, with F1 and AUC increasing for several models, indicating richer separability later in the care pathway. A Mamba+CNN hybrid consistently offered the best accuracy-efficiency trade-off, while transformer variants delivered competitive AUCs at substantially higher computational cost and lightweight CNNs were efficient but less reliable. Performance also showed sensitivity to batch size, underscoring the need for standardized training protocols. Notably, absolute discrimination remained modest overall, reflecting the intrinsic difficulty of TP vs. PsP and the dataset's size imbalance. These results establish a stage-aware benchmark and motivate future work incorporating longitudinal modeling, multi-sequence MRI, and larger multi-center cohorts.
>
---
#### [new 226] Parallel Vision Token Scheduling for Fast and Accurate Multimodal LMMs Inference
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对多模态大模型推理延迟高的问题，提出ParVTS框架。通过并行调度视觉令牌，将非关键信息路径在推理中段丢弃，实现高效计算。无需训练，兼容多种架构，可减少88.9%视觉令牌，提升1.77倍速度，降低70%算力消耗。**

- **链接: [https://arxiv.org/pdf/2511.18875v1](https://arxiv.org/pdf/2511.18875v1)**

> **作者:** Wengyi Zhan; Mingbao Lin; Zhihang Lin; Rongrong Ji
>
> **摘要:** Multimodal large language models (MLLMs) deliver impressive vision-language reasoning but suffer steep inference latency because self-attention scales quadratically with sequence length and thousands of visual tokens contributed by high-resolution images. Naively pruning less-informative visual tokens reduces this burden, yet indiscriminate removal can strip away contextual cues essential for background or fine-grained questions, undermining accuracy. In this paper, we present ParVTS (Parallel Vision Token Scheduling), a training-free scheduling framework that partitions visual tokens into subject and non-subject groups, processes them in parallel to transfer their semantics into question tokens, and discards the non-subject path mid-inference to reduce computation. This scheduling reduces computational complexity, requires no heuristics or additional modules, and is compatible with diverse existing MLLM architectures. Experiments across multiple MLLM backbones show that ParVTS prunes up to 88.9% of visual tokens with minimal performance drop, achieving 1.77x speedup and 70% FLOPs reduction.
>
---
#### [new 227] TRANSPORTER: Transferring Visual Semantics from VLM Manifolds
- **分类: cs.CV**

- **简介: 该论文提出TRANSPORTER模型，解决视觉语言模型（VLM）决策过程不可解释的问题。通过引入logits-to-video（L2V）任务，将VLM的输出分数映射为高保真视频，揭示其预测背后的语义规则。该方法不依赖具体模型，实现对对象属性、动作和场景变化的可解释生成。**

- **链接: [https://arxiv.org/pdf/2511.18359v1](https://arxiv.org/pdf/2511.18359v1)**

> **作者:** Alexandros Stergiou
>
> **备注:** Project page: https://alexandrosstergiou.github.io/TRANSPORTER
>
> **摘要:** How do video understanding models acquire their answers? Although current Vision Language Models (VLMs) reason over complex scenes with diverse objects, action performances, and scene dynamics, understanding and controlling their internal processes remains an open challenge. Motivated by recent advancements in text-to-video (T2V) generative models, this paper introduces a logits-to-video (L2V) task alongside a model-independent approach, TRANSPORTER, to generate videos that capture the underlying rules behind VLMs' predictions. Given the high-visual-fidelity produced by T2V models, TRANSPORTER learns an optimal transport coupling to VLM's high-semantic embedding spaces. In turn, logit scores define embedding directions for conditional video generation. TRANSPORTER generates videos that reflect caption changes over diverse object attributes, action adverbs, and scene context. Quantitative and qualitative evaluations across VLMs demonstrate that L2V can provide a fidelity-rich, novel direction for model interpretability that has not been previously explored.
>
---
#### [new 228] Scale What Counts, Mask What Matters: Evaluating Foundation Models for Zero-Shot Cross-Domain Wi-Fi Sensing
- **分类: cs.CV; cs.IT**

- **简介: 该论文针对Wi-Fi传感中跨域泛化能力差的问题，提出基于大规模预训练的解决方案。通过在14个异构数据集上进行掩码自编码预训练，验证了数据规模与多样性对模型性能的关键作用，发现数据量增长可带来显著提升，而模型容量已非主要瓶颈，为构建鲁棒的通用Wi-Fi感知系统提供新方向。**

- **链接: [https://arxiv.org/pdf/2511.18792v1](https://arxiv.org/pdf/2511.18792v1)**

> **作者:** Cheng Jiang; Yihe Yan; Yanxiang Wang; Chun Tung Chou; Wen Hu
>
> **摘要:** While Wi-Fi sensing offers a compelling, privacy-preserving alternative to cameras, its practical utility has been fundamentally undermined by a lack of robustness across domains. Models trained in one setup fail to generalize to new environments, hardware, or users, a critical "domain shift" problem exacerbated by modest, fragmented public datasets. We shift from this limited paradigm and apply a foundation model approach, leveraging Masked Autoencoding (MAE) style pretraining on the largest and most heterogeneous Wi-Fi CSI datasets collection assembled to date. Our study pretrains and evaluates models on over 1.3 million samples extracted from 14 datasets, collected using 4 distinct devices across the 2.4/5/6 GHz bands and bandwidths from 20 to 160 MHz. Our large-scale evaluation is the first to systematically disentangle the impacts of data diversity versus model capacity on cross-domain performance. The results establish scaling trends on Wi-Fi CSI sensing. First, our experiments show log-linear improvements in unseen domain performance as the amount of pretraining data increases, suggesting that data scale and diversity are key to domain generalization. Second, based on the current data volume, larger model can only provide marginal gains for cross-domain performance, indicating that data, rather than model capacity, is the current bottleneck for Wi-Fi sensing generalization. Finally, we conduct a series of cross-domain evaluations on human activity recognition, human gesture recognition and user identification tasks. The results show that the large-scale pretraining improves cross-domain accuracy ranging from 2.2% to 15.7%, compared to the supervised learning baseline. Overall, our findings provide insightful direction for designing future Wi-Fi sensing systems that can eventually be robust enough for real-world deployment.
>
---
#### [new 229] Evaluating Deep Learning and Traditional Approaches Used in Source Camera Identification
- **分类: cs.CV**

- **简介: 该论文研究源相机识别任务，旨在通过PRNU、JPEG压缩分析和CNN方法识别图像来源设备。对比三种技术的分类准确率，评估其实际应用潜力，为真实场景中的技术落地提供科学依据。**

- **链接: [https://arxiv.org/pdf/2511.19180v1](https://arxiv.org/pdf/2511.19180v1)**

> **作者:** Mansur Ozaman
>
> **备注:** 4 figures
>
> **摘要:** One of the most important tasks in computer vision is identifying the device using which the image was taken, useful for facilitating further comprehensive analysis of the image. This paper presents comparative analysis of three techniques used in source camera identification (SCI): Photo Response Non-Uniformity (PRNU), JPEG compression artifact analysis, and convolutional neural networks (CNNs). It evaluates each method in terms of device classification accuracy. Furthermore, the research discusses the possible scientific development needed for the implementation of the methods in real-life scenarios.
>
---
#### [new 230] MonoMSK: Monocular 3D Musculoskeletal Dynamics Estimation
- **分类: cs.CV**

- **简介: 该论文提出MonoMSK，解决单目视频中3D人体运动的生物力学真实重建问题。针对现有方法忽略物理规律、模型不准确的问题，提出融合数据驱动与物理模拟的框架，联合估计运动学与动力学，通过物理约束的逆-正向循环提升真实性，首次实现单目视频下精确的动力学估计。**

- **链接: [https://arxiv.org/pdf/2511.19326v1](https://arxiv.org/pdf/2511.19326v1)**

> **作者:** Farnoosh Koleini; Hongfei Xue; Ahmed Helmy; Pu Wang
>
> **摘要:** Reconstructing biomechanically realistic 3D human motion - recovering both kinematics (motion) and kinetics (forces) - is a critical challenge. While marker-based systems are lab-bound and slow, popular monocular methods use oversimplified, anatomically inaccurate models (e.g., SMPL) and ignore physics, fundamentally limiting their biomechanical fidelity. In this work, we introduce MonoMSK, a hybrid framework that bridges data-driven learning and physics-based simulation for biomechanically realistic 3D human motion estimation from monocular video. MonoMSK jointly recovers both kinematics (motions) and kinetics (forces and torques) through an anatomically accurate musculoskeletal model. By integrating transformer-based inverse dynamics with differentiable forward kinematics and dynamics layers governed by ODE-based simulation, MonoMSK establishes a physics-regulated inverse-forward loop that enforces biomechanical causality and physical plausibility. A novel forward-inverse consistency loss further aligns motion reconstruction with the underlying kinetic reasoning. Experiments on BML-MoVi, BEDLAM, and OpenCap show that MonoMSK significantly outperforms state-of-the-art methods in kinematic accuracy, while for the first time enabling precise monocular kinetics estimation.
>
---
#### [new 231] ProxT2I: Efficient Reward-Guided Text-to-Image Generation via Proximal Diffusion
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对文本到图像生成任务，提出基于反向离散化与学习的近似算子的高效采样方法ProxT2I，替代传统依赖得分函数的前向采样。通过强化学习优化任务奖励，提升生成质量与人类偏好对齐，同时构建1500万级高质量人脸数据集，实现高效、轻量且高性能的人像生成。**

- **链接: [https://arxiv.org/pdf/2511.18742v1](https://arxiv.org/pdf/2511.18742v1)**

> **作者:** Zhenghan Fang; Jian Zheng; Qiaozi Gao; Xiaofeng Gao; Jeremias Sulam
>
> **摘要:** Diffusion models have emerged as a dominant paradigm for generative modeling across a wide range of domains, including prompt-conditional generation. The vast majority of samplers, however, rely on forward discretization of the reverse diffusion process and use score functions that are learned from data. Such forward and explicit discretizations can be slow and unstable, requiring a large number of sampling steps to produce good-quality samples. In this work we develop a text-to-image (T2I) diffusion model based on backward discretizations, dubbed ProxT2I, relying on learned and conditional proximal operators instead of score functions. We further leverage recent advances in reinforcement learning and policy optimization to optimize our samplers for task-specific rewards. Additionally, we develop a new large-scale and open-source dataset comprising 15 million high-quality human images with fine-grained captions, called LAION-Face-T2I-15M, for training and evaluation. Our approach consistently enhances sampling efficiency and human-preference alignment compared to score-based baselines, and achieves results on par with existing state-of-the-art and open-source text-to-image models while requiring lower compute and smaller model size, offering a lightweight yet performant solution for human text-to-image generation.
>
---
#### [new 232] Upstream Probabilistic Meta-Imputation for Multimodal Pediatric Pancreatitis Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对儿科胰腺炎多模态影像分类中样本少、数据复杂的问题，提出上游概率元插补（UPMI）方法。通过模态特异性回归生成概率向量，构建7维元特征空间，结合高斯混合模型合成虚拟特征，提升随机森林分类器性能，实现更精准的疾病分类。**

- **链接: [https://arxiv.org/pdf/2511.17635v1](https://arxiv.org/pdf/2511.17635v1)**

> **作者:** Max A. Nelson; Elif Keles; Eminenur Sen Tasci; Merve Yazol; Halil Ertugrul Aktas; Ziliang Hong; Andrea Mia Bejar; Gorkem Durak; Oznur Leman Boyunaga; Ulas Bagci
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Pediatric pancreatitis is a progressive and debilitating inflammatory condition, including acute pancreatitis and chronic pancreatitis, that presents significant clinical diagnostic challenges. Machine learning-based methods also face diagnostic challenges due to limited sample availability and multimodal imaging complexity. To address these challenges, this paper introduces Upstream Probabilistic Meta-Imputation (UPMI), a light-weight augmentation strategy that operates upstream of a meta-learner in a low-dimensional meta-feature space rather than in image space. Modality-specific logistic regressions (T1W and T2W MRI radiomics) produce probability outputs that are transformed into a 7-dimensional meta-feature vector. Class-conditional Gaussian mixture models (GMMs) are then fit within each cross-validation fold to sample synthetic meta-features that, combined with real meta-features, train a Random Forest (RF) meta-classifier. On 67 pediatric subjects with paired T1W/T2W MRIs, UPMI achieves a mean AUC of 0.908 $\pm$ 0.072, a $\sim$5% relative gain over a real-only baseline (AUC 0.864 $\pm$ 0.061).
>
---
#### [new 233] Dynamic Granularity Matters: Rethinking Vision Transformers Beyond Fixed Patch Splitting
- **分类: cs.CV**

- **简介: 该论文针对视觉Transformer在细粒度局部特征表示上的不足，提出动态粗细粒度框架Grc-ViT。通过自适应调整图像块大小，结合复杂度评估与注意力优化，提升细粒度识别能力，同时平衡精度与计算效率。属于图像分类任务中的细粒度视觉建模问题。**

- **链接: [https://arxiv.org/pdf/2511.19021v1](https://arxiv.org/pdf/2511.19021v1)**

> **作者:** Qiyang Yu; Yu Fang; Tianrui Li; Xuemei Cao; Yan Chen; Jianghao Li; Fan Min
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Vision Transformers (ViTs) have demonstrated strong capabilities in capturing global dependencies but often struggle to efficiently represent fine-grained local details. Existing multi-scale approaches alleviate this issue by integrating hierarchical or hybrid features; however, they rely on fixed patch sizes and introduce redundant computation. To address these limitations, we propose Granularity-driven Vision Transformer (Grc-ViT), a dynamic coarse-to-fine framework that adaptively adjusts visual granularity based on image complexity. It comprises two key stages: (1) Coarse Granularity Evaluation module, which assesses visual complexity using edge density, entropy, and frequency-domain cues to estimate suitable patch and window sizes; (2) Fine-grained Refinement module, which refines attention computation according to the selected granularity, enabling efficient and precise feature learning. Two learnable parameters, α and \b{eta}, are optimized end-to-end to balance global reasoning and local perception. Comprehensive evaluations demonstrate that Grc-ViT enhances fine-grained discrimination while achieving a superior trade-off between accuracy and computational efficiency.
>
---
#### [new 234] Test-Time Preference Optimization for Image Restoration
- **分类: cs.CV**

- **简介: 该论文针对图像修复中模型与人类偏好不一致的问题，提出测试时偏好优化（TTPO）框架。无需重训练和人工标注，通过扩散反演生成候选图像，利用自动或人工反馈选择偏好样本，以强化修复结果的感知质量，提升多样性与适应性。**

- **链接: [https://arxiv.org/pdf/2511.19169v1](https://arxiv.org/pdf/2511.19169v1)**

> **作者:** Bingchen Li; Xin Li; Jiaqi Xu; Jiaming Guo; Wenbo Li; Renjing Pei; Zhibo Chen
>
> **备注:** Accepted by AAAI26
>
> **摘要:** Image restoration (IR) models are typically trained to recover high-quality images using L1 or LPIPS loss. To handle diverse unknown degradations, zero-shot IR methods have also been introduced. However, existing pre-trained and zero-shot IR approaches often fail to align with human preferences, resulting in restored images that may not be favored. This highlights the critical need to enhance restoration quality and adapt flexibly to various image restoration tasks or backbones without requiring model retraining and ideally without labor-intensive preference data collection. In this paper, we propose the first Test-Time Preference Optimization (TTPO) paradigm for image restoration, which enhances perceptual quality, generates preference data on-the-fly, and is compatible with any IR model backbone. Specifically, we design a training-free, three-stage pipeline: (i) generate candidate preference images online using diffusion inversion and denoising based on the initially restored image; (ii) select preferred and dispreferred images using automated preference-aligned metrics or human feedback; and (iii) use the selected preference images as reward signals to guide the diffusion denoising process, optimizing the restored image to better align with human preferences. Extensive experiments across various image restoration tasks and models demonstrate the effectiveness and flexibility of the proposed pipeline.
>
---
#### [new 235] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection
- **分类: cs.CV**

- **简介: 该论文研究开放词汇航空目标检测任务，针对现有方法依赖文本监督导致语义偏见的问题，提出无需额外监督的VK-Det框架。通过视觉知识引导的细粒度定位与原型感知伪标签策略，增强对新类别物体的识别能力，实现更优的开放词汇泛化性能。**

- **链接: [https://arxiv.org/pdf/2511.18075v1](https://arxiv.org/pdf/2511.18075v1)**

> **作者:** Jianhang Yao; Yongbin Zheng; Siqi Lu; Wanying Xu; Peng Sun
>
> **备注:** 15 pages, 8 figures, accepted by AAAI 2026
>
> **摘要:** To identify objects beyond predefined categories, open-vocabulary aerial object detection (OVAD) leverages the zero-shot capabilities of visual-language models (VLMs) to generalize from base to novel categories. Existing approaches typically utilize self-learning mechanisms with weak text supervision to generate region-level pseudo-labels to align detectors with VLMs semantic spaces. However, text dependence induces semantic bias, restricting open-vocabulary expansion to text-specified concepts. We propose $\textbf{VK-Det}$, a $\textbf{V}$isual $\textbf{K}$nowledge-guided open-vocabulary object $\textbf{Det}$ection framework $\textit{without}$ extra supervision. First, we discover and leverage vision encoder's inherent informative region perception to attain fine-grained localization and adaptive distillation. Second, we introduce a novel prototype-aware pseudo-labeling strategy. It models inter-class decision boundaries through feature clustering and maps detection regions to latent categories via prototype matching. This enhances attention to novel objects while compensating for missing supervision. Extensive experiments show state-of-the-art performance, achieving 30.1 $\mathrm{mAP}^{N}$ on DIOR and 23.3 $\mathrm{mAP}^{N}$ on DOTA, outperforming even extra supervised methods.
>
---
#### [new 236] QAL: A Loss for Recall Precision Balance in 3D Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D重建中召回与精度难以平衡的问题，提出质量感知损失（QAL），通过分离召回与精度控制，提升模型对细结构和稀疏区域的覆盖能力。实验表明，QAL在多个数据集和架构上均显著优于传统损失函数，且增强机器人抓取性能，具备良好的泛化性与实用性。**

- **链接: [https://arxiv.org/pdf/2511.17824v1](https://arxiv.org/pdf/2511.17824v1)**

> **作者:** Pranay Meshram; Yash Turkar; Kartikeya Singh; Praveen Raj Masilamani; Charuvahan Adhivarahan; Karthik Dantu
>
> **备注:** Accepted to WACV 2026. Camera-ready version to appear
>
> **摘要:** Volumetric learning underpins many 3D vision tasks such as completion, reconstruction, and mesh generation, yet training objectives still rely on Chamfer Distance (CD) or Earth Mover's Distance (EMD), which fail to balance recall and precision. We propose Quality-Aware Loss (QAL), a drop-in replacement for CD/EMD that combines a coverage-weighted nearest-neighbor term with an uncovered-ground-truth attraction term, explicitly decoupling recall and precision into tunable components. Across diverse pipelines, QAL achieves consistent coverage gains, improving by an average of +4.3 pts over CD and +2.8 pts over the best alternatives. Though modest in percentage, these improvements reliably recover thin structures and under-represented regions that CD/EMD overlook. Extensive ablations confirm stable performance across hyperparameters and across output resolutions, while full retraining on PCN and ShapeNet demonstrates generalization across datasets and backbones. Moreover, QAL-trained completions yield higher grasp scores under GraspNet evaluation, showing that improved coverage translates directly into more reliable robotic manipulation. QAL thus offers a principled, interpretable, and practical objective for robust 3D vision and safety-critical robotics pipelines
>
---
#### [new 237] Deepfake Geography: Detecting AI-Generated Satellite Images
- **分类: cs.CV**

- **简介: 该论文针对AI生成卫星图像的伪造问题，属于深度伪造检测任务。研究比较CNN与ViT在检测合成影像中的性能，基于13万+样本数据集，发现ViT在准确率（95.11%）和鲁棒性上显著优于CNN，通过可解释性方法揭示其对结构不一致性和纹理重复性的有效识别能力。**

- **链接: [https://arxiv.org/pdf/2511.17766v1](https://arxiv.org/pdf/2511.17766v1)**

> **作者:** Mansur Yerzhanuly
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** The rapid advancement of generative models such as StyleGAN2 and Stable Diffusion poses a growing threat to the authenticity of satellite imagery, which is increasingly vital for reliable analysis and decision-making across scientific and security domains. While deepfake detection has been extensively studied in facial contexts, satellite imagery presents distinct challenges, including terrain-level inconsistencies and structural artifacts. In this study, we conduct a comprehensive comparison between Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for detecting AI-generated satellite images. Using a curated dataset of over 130,000 labeled RGB images from the DM-AER and FSI datasets, we show that ViTs significantly outperform CNNs in both accuracy (95.11 percent vs. 87.02 percent) and overall robustness, owing to their ability to model long-range dependencies and global semantic structures. We further enhance model transparency using architecture-specific interpretability methods, including Grad-CAM for CNNs and Chefer's attention attribution for ViTs, revealing distinct detection behaviors and validating model trustworthiness. Our results highlight the ViT's superior performance in detecting structural inconsistencies and repetitive textural patterns characteristic of synthetic imagery. Future work will extend this research to multispectral and SAR modalities and integrate frequency-domain analysis to further strengthen detection capabilities and safeguard satellite imagery integrity in high-stakes applications.
>
---
#### [new 238] CoD: A Diffusion Foundation Model for Image Compression
- **分类: cs.CV**

- **简介: 该论文提出CoD，首个面向图像压缩的扩散基础模型，解决现有扩散编码器依赖文本条件导致压缩效率低的问题。通过端到端训练，实现超低比特率下的高保真压缩与生成，显著优于Stable Diffusion，且训练成本大幅降低。**

- **链接: [https://arxiv.org/pdf/2511.18706v1](https://arxiv.org/pdf/2511.18706v1)**

> **作者:** Zhaoyang Jia; Zihan Zheng; Naifu Xue; Jiahao Li; Bin Li; Zongyu Guo; Xiaoyi Zhang; Houqiang Li; Yan Lu
>
> **摘要:** Existing diffusion codecs typically build on text-to-image diffusion foundation models like Stable Diffusion. However, text conditioning is suboptimal from a compression perspective, hindering the potential of downstream diffusion codecs, particularly at ultra-low bitrates. To address it, we introduce \textbf{CoD}, the first \textbf{Co}mpression-oriented \textbf{D}iffusion foundation model, trained from scratch to enable end-to-end optimization of both compression and generation. CoD is not a fixed codec but a general foundation model designed for various diffusion-based codecs. It offers several advantages: \textbf{High compression efficiency}, replacing Stable Diffusion with CoD in downstream codecs like DiffC achieves SOTA results, especially at ultra-low bitrates (e.g., 0.0039 bpp); \textbf{Low-cost and reproducible training}, 300$\times$ faster training than Stable Diffusion ($\sim$ 20 vs. $\sim$ 6,250 A100 GPU days) on entirely open image-only datasets; \textbf{Providing new insights}, e.g., We find pixel-space diffusion can achieve VTM-level PSNR with high perceptual quality and can outperform GAN-based codecs using fewer parameters. We hope CoD lays the foundation for future diffusion codec research. Codes will be released.
>
---
#### [new 239] Uncertainty-Aware Dual-Student Knowledge Distillation for Efficient Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对图像分类中的模型压缩问题，提出一种不确定性感知的双学生知识蒸馏框架。通过引入教师预测不确定性，实现更精准的知识传递，并利用异构学生网络（ResNet-18与MobileNetV2）协同学习，显著提升学生模型性能。**

- **链接: [https://arxiv.org/pdf/2511.18826v1](https://arxiv.org/pdf/2511.18826v1)**

> **作者:** Aakash Gore; Anoushka Dey; Aryan Mishra
>
> **摘要:** Knowledge distillation has emerged as a powerful technique for model compression, enabling the transfer of knowledge from large teacher networks to compact student models. However, traditional knowledge distillation methods treat all teacher predictions equally, regardless of the teacher's confidence in those predictions. This paper proposes an uncertainty-aware dual-student knowledge distillation framework that leverages teacher prediction uncertainty to selectively guide student learning. We introduce a peer-learning mechanism where two heterogeneous student architectures, specifically ResNet-18 and MobileNetV2, learn collaboratively from both the teacher network and each other. Experimental results on ImageNet-100 demonstrate that our approach achieves superior performance compared to baseline knowledge distillation methods, with ResNet-18 achieving 83.84\% top-1 accuracy and MobileNetV2 achieving 81.46\% top-1 accuracy, representing improvements of 2.04\% and 0.92\% respectively over traditional single-student distillation approaches.
>
---
#### [new 240] Breaking Forgetting: Training-Free Few-Shot Class-Incremental Learning via Conditional Diffusion
- **分类: cs.CV**

- **简介: 该论文针对少样本类增量学习（FSCIL）中的灾难性遗忘与训练成本高问题，提出无需梯度优化的训练自由框架CD-FSCIL。通过关联梯度优化与条件扩散过程，用生成式扩散替代传统更新，并结合大模型生成文本增强视觉表示，有效缓解样本稀缺，实现高效无训练增量学习。**

- **链接: [https://arxiv.org/pdf/2511.18516v1](https://arxiv.org/pdf/2511.18516v1)**

> **作者:** Haidong Kang; Ketong Qian; Yi Lu
>
> **摘要:** Efforts to overcome catastrophic forgetting in Few-Shot Class-Incremental Learning (FSCIL) have primarily focused on developing more effective gradient-based optimization strategies. In contrast, little attention has been paid to the training cost explosion that inevitably arises as the number of novel classes increases, a consequence of relying on gradient learning even under extreme data scarcity. More critically, since FSCIL typically provides only a few samples for each new class, gradient-based updates not only induce severe catastrophic forgetting on base classes but also hinder adaptation to novel ones. This paper seeks to break this long-standing limitation by asking: Can we design a training-free FSCIL paradigm that entirely removes gradient optimization? We provide an affirmative answer by uncovering an intriguing connection between gradient-based optimization and the Conditional Diffusion process. Building on this observation, we propose a Conditional Diffusion-driven FSCIL (CD-FSCIL) framework that substitutes the conventional gradient update process with a diffusion-based generative transition, enabling training-free incremental adaptation while effectively mitigating forgetting. Furthermore, to enhance representation under few-shot constraints, we introduce a multimodal learning strategy that integrates visual features with natural language descriptions automatically generated by Large Language Models (LLMs). This synergy substantially alleviates the sample scarcity issue and improves generalization across novel classes. Extensive experiments on mainstream FSCIL benchmarks demonstrate that our method not only achieves state-of-the-art performance but also drastically reduces computational and memory overhead, marking a paradigm shift toward training-free continual adaptation.
>
---
#### [new 241] Vision Token Masking Alone Cannot Prevent PHI Leakage in Medical Document OCR: A Systematic Evaluation
- **分类: cs.CV; cs.CR**

- **简介: 该论文研究医疗文档OCR中隐私信息（PHI）泄露问题，针对视觉-语言模型中的视觉标记掩码策略进行系统评估。结果表明，仅靠视觉掩码无法有效防止结构化标识符泄露，需结合语言模型后处理。研究揭示了视觉级防护的局限性，提出混合防御架构以提升隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2511.18272v1](https://arxiv.org/pdf/2511.18272v1)**

> **作者:** Richard J. Young
>
> **备注:** 24 pages, 11 figures, 2 tables
>
> **摘要:** Large vision-language models (VLMs) are increasingly deployed for optical character recognition (OCR) in healthcare settings, raising critical concerns about protected health information (PHI) exposure during document processing. This work presents the first systematic evaluation of inference-time vision token masking as a privacy-preserving mechanism for medical document OCR using DeepSeek-OCR. We introduce seven masking strategies (V3-V9) targeting different architectural layers (SAM encoder blocks, compression layers, dual vision encoders, projector fusion) and evaluate PHI reduction across HIPAA-defined categories using 100 synthetic medical billing statements (drawn from a corpus of 38,517 annotated documents) with perfect ground-truth annotations. All masking strategies converge to 42.9% PHI reduction, successfully suppressing long-form spatially-distributed identifiers (patient names, dates of birth, physical addresses at 100% effectiveness) while failing to prevent short structured identifiers (medical record numbers, social security numbers, email addresses, account numbers at 0% effectiveness). Ablation studies varying mask expansion radius (r=1,2,3) demonstrate that increased spatial coverage does not improve reduction beyond this ceiling, indicating that language model contextual inference - not insufficient visual masking - drives structured identifier leakage. A simulated hybrid architecture combining vision masking with NLP post-processing achieves 88.6% total PHI reduction (assuming 80% NLP accuracy on remaining identifiers). This negative result establishes boundaries for vision-only privacy interventions in VLMs, provides guidance distinguishing PHI types amenable to vision-level versus language-level redaction, and redirects future research toward decoder-level fine-tuning and hybrid defense-in-depth architectures for HIPAA-compliant medical document processing.
>
---
#### [new 242] CORA: Consistency-Guided Semi-Supervised Framework for Reasoning Segmentation
- **分类: cs.CV**

- **简介: 该论文针对推理分割任务，解决标注成本高、泛化能力差的问题。提出CORA框架，通过条件视觉指令、一致性伪标签过滤和令牌级对比对齐，实现少样本下的鲁棒推理分割，在Cityscapes和PanNuke上以极少标注显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17755v1](https://arxiv.org/pdf/2511.17755v1)**

> **作者:** Prantik Howlader; Hoang Nguyen-Canh; Srijan Das; Jingyi Xu; Hieu Le; Dimitris Samaras
>
> **备注:** WACV 2026 accepted
>
> **摘要:** Reasoning segmentation seeks pixel-accurate masks for targets referenced by complex, often implicit instructions, requiring context-dependent reasoning over the scene. Recent multimodal language models have advanced instruction following segmentation, yet generalization remains limited. The key bottleneck is the high cost of curating diverse, high-quality pixel annotations paired with rich linguistic supervision leading to brittle performance under distribution shift. Therefore, we present CORA, a semi-supervised reasoning segmentation framework that jointly learns from limited labeled data and a large corpus of unlabeled images. CORA introduces three main components: 1) conditional visual instructions that encode spatial and contextual relationships between objects; 2) a noisy pseudo-label filter based on the consistency of Multimodal LLM's outputs across semantically equivalent queries; and 3) a token-level contrastive alignment between labeled and pseudo-labeled samples to enhance feature consistency. These components enable CORA to perform robust reasoning segmentation with minimal supervision, outperforming existing baselines under constrained annotation settings. CORA achieves state-of-the-art results, requiring as few as 100 labeled images on Cityscapes, a benchmark dataset for urban scene understanding, surpassing the baseline by $+2.3\%$. Similarly, CORA improves performance by $+2.4\%$ with only 180 labeled images on PanNuke, a histopathology dataset.
>
---
#### [new 243] DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中视觉3D目标检测的分布外（OOD）鲁棒性问题，提出DriveFlow方法。基于预训练文本到图像流模型，通过频域分解实现无训练数据增强：高频率保留前景几何，双频优化背景，提升复杂场景下的检测性能。**

- **链接: [https://arxiv.org/pdf/2511.18713v1](https://arxiv.org/pdf/2511.18713v1)**

> **作者:** Hongbin Lin; Yiming Yang; Chaoda Zheng; Yifan Zhang; Shuaicheng Niu; Zilu Guo; Yafeng Li; Gui Gui; Shuguang Cui; Zhen Li
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** In autonomous driving, vision-centric 3D object detection recognizes and localizes 3D objects from RGB images. However, due to high annotation costs and diverse outdoor scenes, training data often fails to cover all possible test scenarios, known as the out-of-distribution (OOD) issue. Training-free image editing offers a promising solution for improving model robustness by training data enhancement without any modifications to pre-trained diffusion models. Nevertheless, inversion-based methods often suffer from limited effectiveness and inherent inaccuracies, while recent rectified-flow-based approaches struggle to preserve objects with accurate 3D geometry. In this paper, we propose DriveFlow, a Rectified Flow Adaptation method for training data enhancement in autonomous driving based on pre-trained Text-to-Image flow models. Based on frequency decomposition, DriveFlow introduces two strategies to adapt noise-free editing paths derived from text-conditioned velocities. 1) High-Frequency Foreground Preservation: DriveFlow incorporates a high-frequency alignment loss for foreground to maintain precise 3D object geometry. 2) Dual-Frequency Background Optimization: DriveFlow also conducts dual-frequency optimization for background, balancing editing flexibility and semantic consistency. Comprehensive experiments validate the effectiveness and efficiency of DriveFlow, demonstrating comprehensive performance improvements on all categories across OOD scenarios. Code is available at https://github.com/Hongbin98/DriveFlow.
>
---
#### [new 244] Enhancing Multi-Label Thoracic Disease Diagnosis with Deep Ensemble-Based Uncertainty Quantification
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对胸部疾病多标签诊断中深度学习模型缺乏不确定性量化的问题，提出基于深度集成的不确定性量化方法。通过构建9成员深度集成模型，显著提升模型校准性与可靠性，实现SOTA性能，并可分解出可解释的不确定成分，推动其在临床决策中的可信应用。**

- **链接: [https://arxiv.org/pdf/2511.18839v1](https://arxiv.org/pdf/2511.18839v1)**

> **作者:** Yasiru Laksara; Uthayasanker Thayasivam
>
> **摘要:** The utility of deep learning models, such as CheXNet, in high stakes clinical settings is fundamentally constrained by their purely deterministic nature, failing to provide reliable measures of predictive confidence. This project addresses this critical gap by integrating robust Uncertainty Quantification (UQ) into a high performance diagnostic platform for 14 common thoracic diseases on the NIH ChestX-ray14 dataset. Initial architectural development failed to stabilize performance and calibration using Monte Carlo Dropout (MCD), yielding an unacceptable Expected Calibration Error (ECE) of 0.7588. This technical failure necessitated a rigorous architectural pivot to a high diversity, 9-member Deep Ensemble (DE). This resulting DE successfully stabilized performance and delivered superior reliability, achieving a State-of-the-Art (SOTA) average Area Under the Receiver Operating Characteristic Curve (AUROC) of 0.8559 and an average F1 Score of 0.3857. Crucially, the DE demonstrated superior calibration (Mean ECE of 0.0728 and Negative Log-Likelihood (NLL) of 0.1916) and enabled the reliable decomposition of total uncertainty into its Aleatoric (irreducible data noise) and Epistemic (reducible model knowledge) components, with a mean Epistemic Uncertainty (EU) of 0.0240. These results establish the Deep Ensemble as a trustworthy and explainable platform, transforming the model from a probabilistic tool into a reliable clinical decision support system.
>
---
#### [new 245] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对长尾数据集蒸馏中软标签偏差导致性能下降的问题，提出ADSA模块以自适应对齐软标签偏差。通过分析偏差来源并构建感知不平衡的泛化界，实现轻量级、通用性强的性能提升，在ImageNet-1k-LT上尾部类别准确率提升11.8%。**

- **链接: [https://arxiv.org/pdf/2511.17914v1](https://arxiv.org/pdf/2511.17914v1)**

> **作者:** Chenyang Jiang; Hang Zhao; Xinyu Zhang; Zhengcen Li; Qiben Shan; Shaocong Wu; Jingyong Su
>
> **备注:** 10 pages, accepted by NeurIPS 2025
>
> **摘要:** Dataset distillation compresses large-scale datasets into compact, highly informative synthetic data, significantly reducing storage and training costs. However, existing research primarily focuses on balanced datasets and struggles to perform under real-world long-tailed distributions. In this work, we emphasize the critical role of soft labels in long-tailed dataset distillation and uncover the underlying mechanisms contributing to performance degradation. Specifically, we derive an imbalance-aware generalization bound for model trained on distilled dataset. We then identify two primary sources of soft-label bias, which originate from the distillation model and the distilled images, through systematic perturbation of the data imbalance levels. To address this, we propose ADSA, an Adaptive Soft-label Alignment module that calibrates the entangled biases. This lightweight module integrates seamlessly into existing distillation pipelines and consistently improves performance. On ImageNet-1k-LT with EDC and IPC=50, ADSA improves tail-class accuracy by up to 11.8% and raises overall accuracy to 41.4%. Extensive experiments demonstrate that ADSA provides a robust and generalizable solution under limited label budgets and across a range of distillation techniques. Code is available at: https://github.com/j-cyoung/ADSA_DD.git.
>
---
#### [new 246] FlowPortal: Residual-Corrected Flow for Training-Free Video Relighting and Background Replacement
- **分类: cs.CV**

- **简介: 该论文针对视频重光照与背景替换任务，解决现有方法在时序一致性、空间保真度和光照自然性间的平衡难题。提出训练-free的FlowPortal框架，通过残差校正光流实现精准编辑，结合解耦条件设计与高频细节传递，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2511.18346v1](https://arxiv.org/pdf/2511.18346v1)**

> **作者:** Wenshuo Gao; Junyi Fan; Jiangyue Zeng; Shuai Yang
>
> **备注:** Project Page: https://gaowenshuo.github.io/FlowPortalProject/
>
> **摘要:** Video relighting with background replacement is a challenging task critical for applications in film production and creative media. Existing methods struggle to balance temporal consistency, spatial fidelity, and illumination naturalness. To address these issues, we introduce FlowPortal, a novel training-free flow-based video relighting framework. Our core innovation is a Residual-Corrected Flow mechanism that transforms a standard flow-based model into an editing model, guaranteeing perfect reconstruction when input conditions are identical and enabling faithful relighting when they differ, resulting in high structural consistency. This is further enhanced by a Decoupled Condition Design for precise lighting control and a High-Frequency Transfer mechanism for detail preservation. Additionally, a masking strategy isolates foreground relighting from background pure generation process. Experiments demonstrate that FlowPortal achieves superior performance in temporal coherence, structural preservation, and lighting realism, while maintaining high efficiency. Project Page: https://gaowenshuo.github.io/FlowPortalProject/.
>
---
#### [new 247] Unsupervised Multi-View Visual Anomaly Detection via Progressive Homography-Guided Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多视角图像中的无监督视觉异常检测任务，解决视点变化导致的误报问题。提出ViewSense-AD框架，通过基于单应性的多视图对齐模块与扩散模型结合，实现渐进式特征对齐，并引入轻量级融合精炼模块提升一致性，显著降低误报率，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2511.18766v1](https://arxiv.org/pdf/2511.18766v1)**

> **作者:** Xintao Chen; Xiaohao Xu; Bozhong Zheng; Yun Liu; Yingna Wu
>
> **摘要:** Unsupervised visual anomaly detection from multi-view images presents a significant challenge: distinguishing genuine defects from benign appearance variations caused by viewpoint changes. Existing methods, often designed for single-view inputs, treat multiple views as a disconnected set of images, leading to inconsistent feature representations and a high false-positive rate. To address this, we introduce ViewSense-AD (VSAD), a novel framework that learns viewpoint-invariant representations by explicitly modeling geometric consistency across views. At its core is our Multi-View Alignment Module (MVAM), which leverages homography to project and align corresponding feature regions between neighboring views. We integrate MVAM into a View-Align Latent Diffusion Model (VALDM), enabling progressive and multi-stage alignment during the denoising process. This allows the model to build a coherent and holistic understanding of the object's surface from coarse to fine scales. Furthermore, a lightweight Fusion Refiner Module (FRM) enhances the global consistency of the aligned features, suppressing noise and improving discriminative power. Anomaly detection is performed by comparing multi-level features from the diffusion model against a learned memory bank of normal prototypes. Extensive experiments on the challenging RealIAD and MANTA datasets demonstrate that VSAD sets a new state-of-the-art, significantly outperforming existing methods in pixel, view, and sample-level visual anomaly proving its robustness to large viewpoint shifts and complex textures.
>
---
#### [new 248] CrossJEPA: Cross-Modal Joint-Embedding Predictive Architecture for Efficient 3D Representation Learning from 2D Images
- **分类: cs.CV**

- **简介: 该论文提出CrossJEPA，解决3D表示学习中因缺乏大规模3D数据导致模型庞大、训练慢的问题。通过跨模态联合嵌入预测架构，利用2D图像预训练模型知识，高效学习3D点云表示，实现高精度、低资源消耗的3D特征提取。**

- **链接: [https://arxiv.org/pdf/2511.18424v1](https://arxiv.org/pdf/2511.18424v1)**

> **作者:** Avishka Perera; Kumal Hewagamage; Saeedha Nazar; Kavishka Abeywardana; Hasitha Gallella; Ranga Rodrigo; Mohamed Afham
>
> **备注:** 24 pages, 10 figures
>
> **摘要:** Image-to-point cross-modal learning has emerged to address the scarcity of large-scale 3D datasets in 3D representation learning. However, current methods that leverage 2D data often result in large, slow-to-train models, making them computationally expensive and difficult to deploy in resource-constrained environments. The architecture design of such models is therefore critical, determining their performance, memory footprint, and compute efficiency. The Joint-embedding Predictive Architecture (JEPA) has gained wide popularity in self-supervised learning for its simplicity and efficiency, but has been under-explored in cross-modal settings, partly due to the misconception that masking is intrinsic to JEPA. In this light, we propose CrossJEPA, a simple Cross-modal Joint Embedding Predictive Architecture that harnesses the knowledge of an image foundation model and trains a predictor to infer embeddings of specific rendered 2D views from corresponding 3D point clouds, thereby introducing a JEPA-style pretraining strategy beyond masking. By conditioning the predictor on cross-domain projection information, CrossJEPA purifies the supervision signal from semantics exclusive to the target domain. We further exploit the frozen teacher design with a one-time target embedding caching mechanism, yielding amortized efficiency. CrossJEPA achieves a new state-of-the-art in linear probing on the synthetic ModelNet40 (94.2%) and the real-world ScanObjectNN (88.3%) benchmarks, using only 14.1M pretraining parameters (8.5M in the point encoder), and about 6 pretraining hours on a standard single GPU. These results position CrossJEPA as a performant, memory-efficient, and fast-to-train framework for 3D representation learning via knowledge distillation. We analyze CrossJEPA intuitively, theoretically, and empirically, and extensively ablate our design choices. Code will be made available.
>
---
#### [new 249] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?
- **分类: cs.CV**

- **简介: 该论文针对二值神经网络（BNN）中深度可分离卷积难以有效应用的问题，提出1.58位卷积与预归一化残差连接，提升表达能力并稳定训练。首次成功实现BNN中深度可分离卷积的二值化，在多个数据集上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17633v1](https://arxiv.org/pdf/2511.17633v1)**

> **作者:** DoYoung Kim; Jin-Seop Lee; Noo-ri Kim; SungJoon Lee; Jee-Hyong Lee
>
> **备注:** Paper accepted to AAAI 2026
>
> **摘要:** Recent advances in model compression have highlighted the potential of low-bit precision techniques, with Binary Neural Networks (BNNs) attracting attention for their extreme efficiency. However, extreme quantization in BNNs limits representational capacity and destabilizes training, posing significant challenges for lightweight architectures with depth-wise convolutions. To address this, we propose a 1.58-bit convolution to enhance expressiveness and a pre-BN residual connection to stabilize optimization by improving the Hessian condition number. These innovations enable, to the best of our knowledge, the first successful binarization of depth-wise convolutions in BNNs. Our method achieves 33M OPs on ImageNet with MobileNet V1, establishing a new state-of-the-art in BNNs by outperforming prior methods with comparable OPs. Moreover, it consistently outperforms existing methods across various datasets, including CIFAR-10, CIFAR-100, STL-10, Tiny ImageNet, and Oxford Flowers 102, with accuracy improvements of up to 9.3 percentage points.
>
---
#### [new 250] Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution
- **分类: cs.CV**

- **简介: 该论文提出ORS3D任务，旨在解决具身智能中任务调度的效率问题。针对现有数据集忽略运筹学知识与三维空间关联的问题，构建了60K规模的ORS3D-60K数据集，并提出GRANT模型，通过调度标记实现语言理解、三维定位与并行任务优化，提升任务执行效率。**

- **链接: [https://arxiv.org/pdf/2511.19430v1](https://arxiv.org/pdf/2511.19430v1)**

> **作者:** Dingkang Liang; Cheng Zhang; Xiaopeng Xu; Jianzhong Ju; Zhenbo Luo; Xiang Bai
>
> **备注:** Accepted to AAAI 2026 (Oral). The code is available at \url{https://github.com/H-EmbodVis/GRANT}
>
> **摘要:** Task scheduling is critical for embodied AI, enabling agents to follow natural language instructions and execute actions efficiently in 3D physical worlds. However, existing datasets often simplify task planning by ignoring operations research (OR) knowledge and 3D spatial grounding. In this work, we propose Operations Research knowledge-based 3D Grounded Task Scheduling (ORS3D), a new task that requires the synergy of language understanding, 3D grounding, and efficiency optimization. Unlike prior settings, ORS3D demands that agents minimize total completion time by leveraging parallelizable subtasks, e.g., cleaning the sink while the microwave operates. To facilitate research on ORS3D, we construct ORS3D-60K, a large-scale dataset comprising 60K composite tasks across 4K real-world scenes. Furthermore, we propose GRANT, an embodied multi-modal large language model equipped with a simple yet effective scheduling token mechanism to generate efficient task schedules and grounded actions. Extensive experiments on ORS3D-60K validate the effectiveness of GRANT across language understanding, 3D grounding, and scheduling efficiency. The code is available at https://github.com/H-EmbodVis/GRANT
>
---
#### [new 251] Zero-shot segmentation of skin tumors in whole-slide images with vision-language foundation models
- **分类: cs.CV**

- **简介: 该论文提出零样本视觉-语言分割框架ZEUS，用于全切片图像中皮肤肿瘤的精细分割。针对病理图像形态多样、良恶性难辨的问题，利用预训练视觉-语言模型和文本提示生成高分辨率肿瘤掩码，实现无需像素级标注的全自动分割，显著降低标注负担，提升诊断可解释性。**

- **链接: [https://arxiv.org/pdf/2511.18978v1](https://arxiv.org/pdf/2511.18978v1)**

> **作者:** Santiago Moreno; Pablo Meseguer; Rocío del Amor; Valery Naranjo
>
> **备注:** Conference manuscript accepted for oral presentation at CASEIB 2025
>
> **摘要:** Accurate annotation of cutaneous neoplasm biopsies represents a major challenge due to their wide morphological variability, overlapping histological patterns, and the subtle distinctions between benign and malignant lesions. Vision-language foundation models (VLMs), pre-trained on paired image-text corpora, learn joint representations that bridge visual features and diagnostic terminology, enabling zero-shot localization and classification of tissue regions without pixel-level labels. However, most existing VLM applications in histopathology remain limited to slide-level tasks or rely on coarse interactive prompts, and they struggle to produce fine-grained segmentations across gigapixel whole-slide images (WSIs). In this work, we introduce a zero-shot visual-language segmentation pipeline for whole-slide images (ZEUS), a fully automated, zero-shot segmentation framework that leverages class-specific textual prompt ensembles and frozen VLM encoders to generate high-resolution tumor masks in WSIs. By partitioning each WSI into overlapping patches, extracting visual embeddings, and computing cosine similarities against text prompts, we generate a final segmentation mask. We demonstrate competitive performance on two in-house datasets, primary spindle cell neoplasms and cutaneous metastases, highlighting the influence of prompt design, domain shifts, and institutional variability in VLMs for histopathology. ZEUS markedly reduces annotation burden while offering scalable, explainable tumor delineation for downstream diagnostic workflows.
>
---
#### [new 252] Hierarchical GraphCut Phase Unwrapping based on Invariance of Diffeomorphisms Framework
- **分类: cs.CV**

- **简介: 该论文针对3D扫描中的相位解包裹任务，解决噪声、遮挡与复杂几何导致的解包裹不准确问题。提出基于微分同胚不变性的分层GraphCut框架，通过预计算奇数个共形与最优传输映射，利用多数投票融合标签图，高效估计相位整数周期数，实现45.5倍加速与更高精度，适用于实时应用。**

- **链接: [https://arxiv.org/pdf/2511.18682v1](https://arxiv.org/pdf/2511.18682v1)**

> **作者:** Xiang Gao; Xinmu Wang; Zhou Zhao; Junqi Huang; Xianfeng David Gu
>
> **备注:** Open Journal of Signal Processing (OJSP) as journal paper for ICIP2025 Accepted
>
> **摘要:** Recent years have witnessed rapid advancements in 3D scanning technologies, with applications spanning VR/AR, digital human creation, and medical imaging. Structured-light scanning with phase-shifting techniques is preferred for its use of low-intensity visible light and high accuracy, making it well suited for capturing 4D facial dynamics. A key step is phase unwrapping, which recovers continuous phase values from measurements wrapped modulo 2pi. The goal is to estimate the unwrapped phase count k in the equation Phi = phi + 2pi k, where phi is the wrapped phase and Phi is the true phase. Noise, occlusions, and complex 3D geometry make recovering the true phase challenging because phase unwrapping is ill-posed: measurements only provide modulo 2pi values, and estimating k requires assumptions about surface continuity. Existing methods trade speed for accuracy: fast approaches lack precision, while accurate algorithms are too slow for real-time use. To overcome these limitations, this work proposes a phase unwrapping framework that reformulates GraphCut-based unwrapping as a pixel-labeling problem. This framework improves the estimation of the unwrapped phase count k through the invariance property of diffeomorphisms applied in image space via conformal and optimal transport (OT) maps. An odd number of diffeomorphisms are precomputed from the input phase data, and a hierarchical GraphCut algorithm is applied in each domain. The resulting label maps are fused via majority voting to robustly estimate k at each pixel. Experimental results demonstrate a 45.5x speedup and lower L2 error in real experiments and simulations, showing potential for real-time applications.
>
---
#### [new 253] ObjectAlign: Neuro-Symbolic Object Consistency Verification and Correction
- **分类: cs.CV; cs.AI; cs.FL; cs.LG**

- **简介: 该论文针对视频编辑中的对象不一致性问题，提出ObjectAlign框架。通过融合感知度量与符号推理，实现对象一致性检测、验证与修正。创新点包括可学习的度量阈值、神经符号验证器及自适应插值修复，显著提升视频质量。**

- **链接: [https://arxiv.org/pdf/2511.18701v1](https://arxiv.org/pdf/2511.18701v1)**

> **作者:** Mustafa Munir; Harsh Goel; Xiwen Wei; Minkyu Choi; Sahil Shah; Kartikeya Bhardwaj; Paul Whatmough; Sandeep Chinchali; Radu Marculescu
>
> **摘要:** Video editing and synthesis often introduce object inconsistencies, such as frame flicker and identity drift that degrade perceptual quality. To address these issues, we introduce ObjectAlign, a novel framework that seamlessly blends perceptual metrics with symbolic reasoning to detect, verify, and correct object-level and temporal inconsistencies in edited video sequences. The novel contributions of ObjectAlign are as follows: First, we propose learnable thresholds for metrics characterizing object consistency (i.e. CLIP-based semantic similarity, LPIPS perceptual distance, histogram correlation, and SAM-derived object-mask IoU). Second, we introduce a neuro-symbolic verifier that combines two components: (a) a formal, SMT-based check that operates on masked object embeddings to provably guarantee that object identity does not drift, and (b) a temporal fidelity check that uses a probabilistic model checker to verify the video's formal representation against a temporal logic specification. A frame transition is subsequently deemed "consistent" based on a single logical assertion that requires satisfying both the learned metric thresholds and this unified neuro-symbolic constraint, ensuring both low-level stability and high-level temporal correctness. Finally, for each contiguous block of flagged frames, we propose a neural network based interpolation for adaptive frame repair, dynamically choosing the interpolation depth based on the number of frames to be corrected. This enables reconstruction of the corrupted frames from the last valid and next valid keyframes. Our results show up to 1.4 point improvement in CLIP Score and up to 6.1 point improvement in warp error compared to SOTA baselines on the DAVIS and Pexels video datasets.
>
---
#### [new 254] Personalized Federated Segmentation with Shared Feature Aggregation and Boundary-Focused Calibration
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医疗图像中多器官肿瘤分割的非独立同分布数据问题，提出个性化联邦分割方法FedOAP。通过解耦交叉注意力捕捉跨客户端的共享特征长程依赖，并引入扰动边界损失提升边界分割精度，有效缓解数据异质性，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18847v1](https://arxiv.org/pdf/2511.18847v1)**

> **作者:** Ishmam Tashdeed; Md. Atiqur Rahman; Sabrina Islam; Md. Azam Hossain
>
> **摘要:** Personalized federated learning (PFL) possesses the unique capability of preserving data confidentiality among clients while tackling the data heterogeneity problem of non-independent and identically distributed (Non-IID) data. Its advantages have led to widespread adoption in domains such as medical image segmentation. However, the existing approaches mostly overlook the potential benefits of leveraging shared features across clients, where each client contains segmentation data of different organs. In this work, we introduce a novel personalized federated approach for organ agnostic tumor segmentation (FedOAP), that utilizes cross-attention to model long-range dependencies among the shared features of different clients and a boundary-aware loss to improve segmentation consistency. FedOAP employs a decoupled cross-attention (DCA), which enables each client to retain local queries while attending to globally shared key-value pairs aggregated from all clients, thereby capturing long-range inter-organ feature dependencies. Additionally, we introduce perturbed boundary loss (PBL) which focuses on the inconsistencies of the predicted mask's boundary for each client, forcing the model to localize the margins more precisely. We evaluate FedOAP on diverse tumor segmentation tasks spanning different organs. Extensive experiments demonstrate that FedOAP consistently outperforms existing state-of-the-art federated and personalized segmentation methods.
>
---
#### [new 255] Dendritic Convolution for Noise Image Recognition
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对图像识别中的噪声干扰问题，提出一种仿生的树突卷积（DDC），通过模拟神经元树突的邻域交互与非线性处理机制，重构特征提取数学范式。在分类与检测任务中，显著提升模型在噪声环境下的性能。**

- **链接: [https://arxiv.org/pdf/2511.18699v1](https://arxiv.org/pdf/2511.18699v1)**

> **作者:** Jiarui Xue; Dongjian Yang; Ye Sun; Gang Liu
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** In real-world scenarios of image recognition, there exists substantial noise interference. Existing works primarily focus on methods such as adjusting networks or training strategies to address noisy image recognition, and the anti-noise performance has reached a bottleneck. However, little is known about the exploration of anti-interference solutions from a neuronal perspective.This paper proposes an anti-noise neuronal convolution. This convolution mimics the dendritic structure of neurons, integrates the neighborhood interaction computation logic of dendrites into the underlying design of convolutional operations, and simulates the XOR logic preprocessing function of biological dendrites through nonlinear interactions between input features, thereby fundamentally reconstructing the mathematical paradigm of feature extraction. Unlike traditional convolution where noise directly interferes with feature extraction and exerts a significant impact, DDC mitigates the influence of noise by focusing on the interaction of neighborhood information. Experimental results demonstrate that in image classification tasks (using YOLOv11-cls, VGG16, and EfficientNet-B0) and object detection tasks (using YOLOv11, YOLOv8, and YOLOv5), after replacing traditional convolution with the dendritic convolution, the accuracy of the EfficientNet-B0 model on noisy datasets is relatively improved by 11.23%, and the mean Average Precision (mAP) of YOLOv8 is increased by 19.80%. The consistency between the computation method of this convolution and the dendrites of biological neurons enables it to perform significantly better than traditional convolution in complex noisy environments.
>
---
#### [new 256] Versatile Recompression-Aware Perceptual Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文针对感知图像超分辨率任务，解决重建图像在后续压缩中引入伪影的问题。提出VRPSR方法，通过扩散模型模拟多种压缩编码器，使超分模型感知并适应不同压缩场景，实现更高效、鲁棒的图像恢复与压缩协同优化。**

- **链接: [https://arxiv.org/pdf/2511.18090v1](https://arxiv.org/pdf/2511.18090v1)**

> **作者:** Mingwei He; Tongda Xu; Xingtong Ge; Ming Sun; Chao Zhou; Yan Wang
>
> **摘要:** Perceptual image super-resolution (SR) methods restore degraded images and produce sharp outputs. In practice, those outputs are usually recompressed for storage and transmission. Ignoring recompression is suboptimal as the downstream codec might add additional artifacts to restored images. However, jointly optimizing SR and recompression is challenging, as the codecs are not differentiable and vary in configuration. In this paper, we present Versatile Recompression-Aware Perceptual Super-Resolution (VRPSR), which makes existing perceptual SR aware of versatile compression. First, we formulate compression as conditional text-to-image generation and utilize a pre-trained diffusion model to build a generalizable codec simulator. Next, we propose a set of training techniques tailored for perceptual SR, including optimizing the simulator using perceptual targets and adopting slightly compressed images as the training target. Empirically, our VRPSR saves more than 10\% bitrate based on Real-ESRGAN and S3Diff under H.264/H.265/H.266 compression. Besides, our VRPSR facilitates joint optimization of the SR and post-processing model after recompression.
>
---
#### [new 257] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System
- **分类: cs.CV**

- **简介: 该论文针对路边感知系统缺乏自然语言交互与情境推理能力的问题，提出RoadSceneVQA数据集，涵盖复杂交通场景的视觉问答任务。通过设计CAF融合模块与AD-CoT推理框架，构建了基准模型RoadMind，显著提升多模态大模型在交通行为理解中的推理准确率与效率。**

- **链接: [https://arxiv.org/pdf/2511.18286v1](https://arxiv.org/pdf/2511.18286v1)**

> **作者:** Runwei Guan; Rongsheng Hu; Shangshu Chen; Ningyuan Xiao; Xue Xia; Jiayang Liu; Beibei Chen; Ziren Tang; Ningwei Ouyang; Shaofeng Liang; Yuxuan Fan; Wanjie Sun; Yutao Yue
>
> **备注:** 9 pages, 6 figures, accepted by AAAI 2026. The model is also called Dream, to the other me in the world forever
>
> **摘要:** Current roadside perception systems mainly focus on instance-level perception, which fall short in enabling interaction via natural language and reasoning about traffic behaviors in context. To bridge this gap, we introduce RoadSceneVQA, a large-scale and richly annotated visual question answering (VQA) dataset specifically tailored for roadside scenarios. The dataset comprises 34,736 diverse QA pairs collected under varying weather, illumination, and traffic conditions, targeting not only object attributes but also the intent, legality, and interaction patterns of traffic participants. RoadSceneVQA challenges models to perform both explicit recognition and implicit commonsense reasoning, grounded in real-world traffic rules and contextual dependencies. To fully exploit the reasoning potential of Multi-modal Large Language Models (MLLMs), we further propose CogniAnchor Fusion (CAF), a vision-language fusion module inspired by human-like scene anchoring mechanisms. Moreover, we propose the Assisted Decoupled Chain-of-Thought (AD-CoT) to enhance the reasoned thinking via CoT prompting and multi-task learning. Based on the above, we propose the baseline model RoadMind. Experiments on RoadSceneVQA and CODA-LM benchmark show that the pipeline consistently improves both reasoning accuracy and computational efficiency, allowing the MLLM to achieve state-of-the-art performance in structural traffic perception and reasoning tasks.
>
---
#### [new 258] Show Me: Unifying Instructional Image and Video Generation with Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出ShowMe，一个统一的扩散模型框架，用于指令驱动的图像与视频生成。针对图像编辑忽略时序、视频预测忽视目标的痛点，通过激活时空组件并引入一致性奖励，实现两者协同优化，显著提升生成质量与一致性。**

- **链接: [https://arxiv.org/pdf/2511.17839v1](https://arxiv.org/pdf/2511.17839v1)**

> **作者:** Yujiang Pu; Zhanbo Huang; Vishnu Boddeti; Yu Kong
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Generating visual instructions in a given context is essential for developing interactive world simulators. While prior works address this problem through either text-guided image manipulation or video prediction, these tasks are typically treated in isolation. This separation reveals a fundamental issue: image manipulation methods overlook how actions unfold over time, while video prediction models often ignore the intended outcomes. To this end, we propose ShowMe, a unified framework that enables both tasks by selectively activating the spatial and temporal components of video diffusion models. In addition, we introduce structure and motion consistency rewards to improve structural fidelity and temporal coherence. Notably, this unification brings dual benefits: the spatial knowledge gained through video pretraining enhances contextual consistency and realism in non-rigid image edits, while the instruction-guided manipulation stage equips the model with stronger goal-oriented reasoning for video prediction. Experiments on diverse benchmarks demonstrate that our method outperforms expert models in both instructional image and video generation, highlighting the strength of video diffusion models as a unified action-object state transformer.
>
---
#### [new 259] UniFlow: Towards Zero-Shot LiDAR Scene Flow for Autonomous Vehicles via Cross-Domain Generalization
- **分类: cs.CV**

- **简介: 该论文研究LiDAR场景流任务，旨在解决现有方法依赖单一传感器、泛化能力差的问题。通过跨数据集训练，提出UniFlow模型，利用多源数据学习通用运动先验，显著提升在未见传感器和数据集上的性能，实现零样本跨域泛化，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18254v1](https://arxiv.org/pdf/2511.18254v1)**

> **作者:** Siyi Li; Qingwen Zhang; Ishan Khatri; Kyle Vedder; Deva Ramanan; Neehar Peri
>
> **备注:** Project Page: https://lisiyi777.github.io/UniFlow/
>
> **摘要:** LiDAR scene flow is the task of estimating per-point 3D motion between consecutive point clouds. Recent methods achieve centimeter-level accuracy on popular autonomous vehicle (AV) datasets, but are typically only trained and evaluated on a single sensor. In this paper, we aim to learn general motion priors that transfer to diverse and unseen LiDAR sensors. However, prior work in LiDAR semantic segmentation and 3D object detection demonstrate that naively training on multiple datasets yields worse performance than single dataset models. Interestingly, we find that this conventional wisdom does not hold for motion estimation, and that state-of-the-art scene flow methods greatly benefit from cross-dataset training. We posit that low-level tasks such as motion estimation may be less sensitive to sensor configuration; indeed, our analysis shows that models trained on fast-moving objects (e.g., from highway datasets) perform well on fast-moving objects, even across different datasets. Informed by our analysis, we propose UniFlow, a family of feedforward models that unifies and trains on multiple large-scale LiDAR scene flow datasets with diverse sensor placements and point cloud densities. Our frustratingly simple solution establishes a new state-of-the-art on Waymo and nuScenes, improving over prior work by 5.1% and 35.2% respectively. Moreover, UniFlow achieves state-of-the-art accuracy on unseen datasets like TruckScenes, outperforming prior TruckScenes-specific models by 30.1%.
>
---
#### [new 260] PartDiffuser: Part-wise 3D Mesh Generation via Discrete Diffusion
- **分类: cs.CV**

- **简介: 该论文提出PartDiffuser，用于3D网格生成任务。针对现有自回归方法在全局结构与局部细节间的平衡难题及误差累积问题，提出分部式半自回归扩散框架：通过语义分割分块处理，利用跨部件自回归保证拓扑一致性，块内并行离散扩散精细重建高频几何特征，结合点云条件实现高效解耦生成。**

- **链接: [https://arxiv.org/pdf/2511.18801v1](https://arxiv.org/pdf/2511.18801v1)**

> **作者:** Yichen Yang; Hong Li; Haodong Zhu; Linin Yang; Guojun Lei; Sheng Xu; Baochang Zhang
>
> **摘要:** Existing autoregressive (AR) methods for generating artist-designed meshes struggle to balance global structural consistency with high-fidelity local details, and are susceptible to error accumulation. To address this, we propose PartDiffuser, a novel semi-autoregressive diffusion framework for point-cloud-to-mesh generation. The method first performs semantic segmentation on the mesh and then operates in a "part-wise" manner: it employs autoregression between parts to ensure global topology, while utilizing a parallel discrete diffusion process within each semantic part to precisely reconstruct high-frequency geometric features. PartDiffuser is based on the DiT architecture and introduces a part-aware cross-attention mechanism, using point clouds as hierarchical geometric conditioning to dynamically control the generation process, thereby effectively decoupling the global and local generation tasks. Experiments demonstrate that this method significantly outperforms state-of-the-art (SOTA) models in generating 3D meshes with rich detail, exhibiting exceptional detail representation suitable for real-world applications.
>
---
#### [new 261] MedPEFT-CL: Dual-Phase Parameter-Efficient Continual Learning with Medical Semantic Adapter and Bidirectional Memory Consolidation
- **分类: cs.CV**

- **简介: 该论文针对医疗视觉-语言分割模型在学习新解剖结构时的灾难性遗忘问题，提出MedPEFT-CL框架。通过双阶段设计，结合语义适配器分配与双向记忆巩固，实现高效参数更新与知识保留，显著缓解遗忘，适用于持续学习的医疗视觉-语言任务。**

- **链接: [https://arxiv.org/pdf/2511.17668v1](https://arxiv.org/pdf/2511.17668v1)**

> **作者:** Ziyuan Gao
>
> **备注:** Accepted by WACV 2026 (round 2)
>
> **摘要:** Medical vision-language segmentation models suffer from catastrophic forgetting when adapting to new anatomical structures, requiring complete retraining that limits their clinical deployment. Although continual learning approaches have been studied for various applications, targeted research on continual learning approaches specifically designed for medical vision-language tasks remains underexplored. We propose MedPEFT-CL, a parameter-efficient continual learning framework that addresses both efficient learning of new tasks and preservation of previous knowledge through a dual-phase architecture based on CLIPSeg. Our dual-phase architecture features an adaptive learning phase that employs semantic similarity-based adapter allocation and parameter-efficient fine-tuning for medical tasks through prompt similarity analysis, and a knowledge consolidation phase employing bi-directional Fisher-memory coordination. This creates a reinforcing cycle: consolidation directs replay priorities while new tasks provide challenging samples that improve retention strategies. Our key contributions are: (1) a semantic-driven adapter allocation mechanism that enables efficient learning of new medical tasks, (2) a bi-modal LoRA adaptation that significantly reduces trainable parameters while maintaining cross-modal learning, and (3) bidirectional Fisher-memory coordination that prevents catastrophic forgetting from previous medical tasks. Extensive experiments across diverse medical datasets demonstrate superior forgetting mitigation and performance retention with minimal parameter overhead, making the framework effective for continual learning in medical vision-language scenarios.
>
---
#### [new 262] BackSplit: The Importance of Sub-dividing the Background in Biomedical Lesion Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像中小病灶分割难题，提出将背景细分为多个类别（BackSplit），以利用更丰富的解剖上下文信息。通过信息论分析和实验验证，证明该方法能显著提升分割性能，且无需增加推理成本，适用于自动或交互式生成的辅助标签。**

- **链接: [https://arxiv.org/pdf/2511.19394v1](https://arxiv.org/pdf/2511.19394v1)**

> **作者:** Rachit Saluja; Asli Cihangir; Ruining Deng; Johannes C. Paetzold; Fengbei Liu; Mert R. Sabuncu
>
> **摘要:** Segmenting small lesions in medical images remains notoriously difficult. Most prior work tackles this challenge by either designing better architectures, loss functions, or data augmentation schemes; and collecting more labeled data. We take a different view, arguing that part of the problem lies in how the background is modeled. Common lesion segmentation collapses all non-lesion pixels into a single "background" class, ignoring the rich anatomical context in which lesions appear. In reality, the background is highly heterogeneous-composed of tissues, organs, and other structures that can now be labeled manually or inferred automatically using existing segmentation models. In this paper, we argue that training with fine-grained labels that sub-divide the background class, which we call BackSplit, is a simple yet powerful paradigm that can offer a significant performance boost without increasing inference costs. From an information theoretic standpoint, we prove that BackSplit increases the expected Fisher Information relative to conventional binary training, leading to tighter asymptotic bounds and more stable optimization. With extensive experiments across multiple datasets and architectures, we empirically show that BackSplit consistently boosts small-lesion segmentation performance, even when auxiliary labels are generated automatically using pretrained segmentation models. Additionally, we demonstrate that auxiliary labels derived from interactive segmentation frameworks exhibit the same beneficial effect, demonstrating its robustness, simplicity, and broad applicability.
>
---
#### [new 263] In-Video Instructions: Visual Signals as Generative Control
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出“视频内指令”（In-Video Instruction）范式，将用户指令以视觉元素（如文字、箭头）嵌入视频帧中，实现对图像到视频生成的精确控制。针对传统文本提示全局模糊的问题，该方法通过视觉信号实现空间精准、对象明确的指令引导，在多对象场景下显著提升生成可控性。**

- **链接: [https://arxiv.org/pdf/2511.19401v1](https://arxiv.org/pdf/2511.19401v1)**

> **作者:** Gongfan Fang; Xinyin Ma; Xinchao Wang
>
> **摘要:** Large-scale video generative models have recently demonstrated strong visual capabilities, enabling the prediction of future frames that adhere to the logical and physical cues in the current observation. In this work, we investigate whether such capabilities can be harnessed for controllable image-to-video generation by interpreting visual signals embedded within the frames as instructions, a paradigm we term In-Video Instruction. In contrast to prompt-based control, which provides textual descriptions that are inherently global and coarse, In-Video Instruction encodes user guidance directly into the visual domain through elements such as overlaid text, arrows, or trajectories. This enables explicit, spatial-aware, and unambiguous correspondences between visual subjects and their intended actions by assigning distinct instructions to different objects. Extensive experiments on three state-of-the-art generators, including Veo 3.1, Kling 2.5, and Wan 2.2, show that video models can reliably interpret and execute such visually embedded instructions, particularly in complex multi-object scenarios.
>
---
#### [new 264] View-Consistent Diffusion Representations for 3D-Consistent Video Generation
- **分类: cs.CV**

- **简介: 该论文针对视频生成中的3D不一致性问题，提出ViCoDR方法，通过学习多视角一致的扩散表示，提升视频在相机视角变化下的几何稳定性。解决了现有模型生成视频时物体形变、结构错乱的问题，显著增强了生成视频的3D一致性。**

- **链接: [https://arxiv.org/pdf/2511.18991v1](https://arxiv.org/pdf/2511.18991v1)**

> **作者:** Duolikun Danier; Ge Gao; Steven McDonagh; Changjian Li; Hakan Bilen; Oisin Mac Aodha
>
> **摘要:** Video generation models have made significant progress in generating realistic content, enabling applications in simulation, gaming, and film making. However, current generated videos still contain visual artifacts arising from 3D inconsistencies, e.g., objects and structures deforming under changes in camera pose, which can undermine user experience and simulation fidelity. Motivated by recent findings on representation alignment for diffusion models, we hypothesize that improving the multi-view consistency of video diffusion representations will yield more 3D-consistent video generation. Through detailed analysis on multiple recent camera-controlled video diffusion models we reveal strong correlations between 3D-consistent representations and videos. We also propose ViCoDR, a new approach for improving the 3D consistency of video models by learning multi-view consistent diffusion representations. We evaluate ViCoDR on camera controlled image-to-video, text-to-video, and multi-view generation models, demonstrating significant improvements in the 3D consistency of the generated videos. Project page: https://danier97.github.io/ViCoDR.
>
---
#### [new 265] Can a Second-View Image Be a Language? Geometric and Semantic Cross-Modal Reasoning for X-ray Prohibited Item Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦于X-ray违禁品检测任务，针对传统方法依赖单视图视觉信息、难以应对复杂威胁的问题，提出将第二视图视为“语言式”模态。构建DualXrayBench基准与GSXray数据集，设计GSR模型，联合学习跨视图几何与跨模态语义，实现多视图协同推理，显著提升检测性能。**

- **链接: [https://arxiv.org/pdf/2511.18385v1](https://arxiv.org/pdf/2511.18385v1)**

> **作者:** Chuang Peng; Renshuai Tao; Zhongwei Ren; Xianglong Liu; Yunchao Wei
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Automatic X-ray prohibited items detection is vital for security inspection and has been widely studied. Traditional methods rely on visual modality, often struggling with complex threats. While recent studies incorporate language to guide single-view images, human inspectors typically use dual-view images in practice. This raises the question: can the second view provide constraints similar to a language modality? In this work, we introduce DualXrayBench, the first comprehensive benchmark for X-ray inspection that includes multiple views and modalities. It supports eight tasks designed to test cross-view reasoning. In DualXrayBench, we introduce a caption corpus consisting of 45,613 dual-view image pairs across 12 categories with corresponding captions. Building upon these data, we propose the Geometric (cross-view)-Semantic (cross-modality) Reasoner (GSR), a multimodal model that jointly learns correspondences between cross-view geometry and cross-modal semantics, treating the second-view images as a "language-like modality". To enable this, we construct the GSXray dataset, with structured Chain-of-Thought sequences: <top>, <side>, <conclusion>. Comprehensive evaluations on DualXrayBench demonstrate that GSR achieves significant improvements across all X-ray tasks, offering a new perspective for real-world X-ray inspection.
>
---
#### [new 266] Synthetic Curriculum Reinforces Compositional Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成中的组合性难题，提出基于场景图的课程强化学习框架CompGen。通过难度感知的图采样与自适应课程设计，优化模型对复杂场景中多对象及关系的生成能力，显著提升扩散与自回归模型的组合生成性能。**

- **链接: [https://arxiv.org/pdf/2511.18378v1](https://arxiv.org/pdf/2511.18378v1)**

> **作者:** Shijian Wang; Runhao Fu; Siyi Zhao; Qingqin Zhan; Xingjian Wang; Jiarui Jin; Yuan Lu; Hanqian Wu; Cunjian Chen
>
> **摘要:** Text-to-Image (T2I) generation has long been an open problem, with compositional synthesis remaining particularly challenging. This task requires accurate rendering of complex scenes containing multiple objects that exhibit diverse attributes as well as intricate spatial and semantic relationships, demanding both precise object placement and coherent inter-object interactions. In this paper, we propose a novel compositional curriculum reinforcement learning framework named CompGen that addresses compositional weakness in existing T2I models. Specifically, we leverage scene graphs to establish a novel difficulty criterion for compositional ability and develop a corresponding adaptive Markov Chain Monte Carlo graph sampling algorithm. This difficulty-aware approach enables the synthesis of training curriculum data that progressively optimize T2I models through reinforcement learning. We integrate our curriculum learning approach into Group Relative Policy Optimization (GRPO) and investigate different curriculum scheduling strategies. Our experiments reveal that CompGen exhibits distinct scaling curves under different curriculum scheduling strategies, with easy-to-hard and Gaussian sampling strategies yielding superior scaling performance compared to random sampling. Extensive experiments demonstrate that CompGen significantly enhances compositional generation capabilities for both diffusion-based and auto-regressive T2I models, highlighting its effectiveness in improving the compositional T2I generation systems.
>
---
#### [new 267] SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SWITCH基准，针对长时序具身智能中的实体控制接口（TCI）建模与操作问题。解决现有基准缺乏对视觉证据、因果预测与结果验证的评估难题。通过351个真实设备任务，评估模型在视觉问答、语义定位、动作生成等五方面能力，推动更鲁棒的具身智能发展。**

- **链接: [https://arxiv.org/pdf/2511.17649v1](https://arxiv.org/pdf/2511.17649v1)**

> **作者:** Jieru Lin; Zhiwei Yu; Börje F. Karlsson
>
> **摘要:** Autonomous intelligence requires not only perception and reasoning, but critically, effective interaction with the existing world and its infrastructure. Everyday environments are rich in tangible control interfaces (TCIs), e.g., light switches, appliance panels, and embedded GUIs, that demand commonsense and physics reasoning, but also causal prediction and outcome verification in time and space (e.g., delayed heating, remote lights). Moreover, failures here have potential safety implications, yet current benchmarks rarely test grounding, partial observability (video), or post-hoc verification in situated settings. We introduce SWITCH (Semantic World Interface Tasks for Control and Handling), an embodied, task-driven benchmark created through iterative releases to probe these gaps. Its first iteration, SWITCH-Basic, evaluates five complementary abilities:task-aware VQA, semantic UI grounding, action generation, state-transition prediction, and result verification, under egocentric RGB video input and device diversity. Across 351 tasks spanning 98 real devices and appliances, commercial and open LMMMs exhibit inconsistent performance even on single-step interactions, often over-relying on textual cues and under-using visual or video evidence (and high aggregate scores can mask such failures). SWITCH provides data, code, and held-out splits to enable reproducible evaluation and community contributions toward more challenging future iterations of the benchmark and the creation of training datasets. Benchmark resources are available at: https://github.com/BAAI-Agents/SWITCH.
>
---
#### [new 268] HunyuanVideo 1.5 Technical Report
- **分类: cs.CV**

- **简介: 该论文提出HunyuanVideo 1.5，一个仅8.3亿参数的轻量级开源视频生成模型。针对高质视频生成成本高、难以在消费级设备部署的问题，通过优化数据、DiT架构、双语编码与超分网络，实现文本/图像到视频的高效生成，显著提升视觉质量与动作连贯性，推动开放社区的视频创作普及。**

- **链接: [https://arxiv.org/pdf/2511.18870v1](https://arxiv.org/pdf/2511.18870v1)**

> **作者:** Bing Wu; Chang Zou; Changlin Li; Duojun Huang; Fang Yang; Hao Tan; Jack Peng; Jianbing Wu; Jiangfeng Xiong; Jie Jiang; Linus; Patrol; Peizhen Zhang; Peng Chen; Penghao Zhao; Qi Tian; Songtao Liu; Weijie Kong; Weiyan Wang; Xiao He; Xin Li; Xinchi Deng; Xuefei Zhe; Yang Li; Yanxin Long; Yuanbo Peng; Yue Wu; Yuhong Liu; Zhenyu Wang; Zuozhuo Dai; Bo Peng; Coopers Li; Gu Gong; Guojian Xiao; Jiahe Tian; Jiaxin Lin; Jie Liu; Jihong Zhang; Jiesong Lian; Kaihang Pan; Lei Wang; Lin Niu; Mingtao Chen; Mingyang Chen; Mingzhe Zheng; Miles Yang; Qiangqiang Hu; Qi Yang; Qiuyong Xiao; Runzhou Wu; Ryan Xu; Rui Yuan; Shanshan Sang; Shisheng Huang; Siruis Gong; Shuo Huang; Weiting Guo; Xiang Yuan; Xiaojia Chen; Xiawei Hu; Wenzhi Sun; Xiele Wu; Xianshun Ren; Xiaoyan Yuan; Xiaoyue Mi; Yepeng Zhang; Yifu Sun; Yiting Lu; Yitong Li; You Huang; Yu Tang; Yixuan Li; Yuhang Deng; Yuan Zhou; Zhichao Hu; Zhiguang Liu; Zhihe Yang; Zilin Yang; Zhenzhi Lu; Zixiang Zhou; Zhao Zhong
>
> **摘要:** We present HunyuanVideo 1.5, a lightweight yet powerful open-source video generation model that achieves state-of-the-art visual quality and motion coherence with only 8.3 billion parameters, enabling efficient inference on consumer-grade GPUs. This achievement is built upon several key components, including meticulous data curation, an advanced DiT architecture featuring selective and sliding tile attention (SSTA), enhanced bilingual understanding through glyph-aware text encoding, progressive pre-training and post-training, and an efficient video super-resolution network. Leveraging these designs, we developed a unified framework capable of high-quality text-to-video and image-to-video generation across multiple durations and resolutions.Extensive experiments demonstrate that this compact and proficient model establishes a new state-of-the-art among open-source video generation models. By releasing the code and model weights, we provide the community with a high-performance foundation that lowers the barrier to video creation and research, making advanced video generation accessible to a broader audience. All open-source assets are publicly available at https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.
>
---
#### [new 269] SPIDER: Spatial Image CorresponDence Estimator for Robust Calibration
- **分类: cs.CV**

- **简介: 该论文针对跨域图像匹配难题，提出SPIDER框架，解决大视角变化下特征匹配不准确的问题。通过融合2D与3D空间信息，提升对细粒度几何结构的敏感性，实现更鲁棒的图像对应估计，在复杂场景中显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17750v1](https://arxiv.org/pdf/2511.17750v1)**

> **作者:** Zhimin Shao; Abhay Yadav; Rama Chellappa; Cheng Peng
>
> **摘要:** Reliable image correspondences form the foundation of vision-based spatial perception, enabling recovery of 3D structure and camera poses. However, unconstrained feature matching across domains such as aerial, indoor, and outdoor scenes remains challenging due to large variations in appearance, scale and viewpoint. Feature matching has been conventionally formulated as a 2D-to-2D problem; however, recent 3D foundation models provides spatial feature matching properties based on two-view geometry. While powerful, we observe that these spatially coherent matches often concentrate on dominant planar regions, e.g., walls or ground surfaces, while being less sensitive to fine-grained geometric details, particularly under large viewpoint changes. To better understand these trade-offs, we first perform linear probe experiments to evaluate the performance of various vision foundation models for image matching. Building on these insights, we introduce SPIDER, a universal feature matching framework that integrates a shared feature extraction backbone with two specialized network heads for estimating both 2D-based and 3D-based correspondences from coarse to fine. Finally, we introduce an image-matching evaluation benchmark that focuses on unconstrained scenarios with large baselines. SPIDER significantly outperforms SoTA methods, demonstrating its strong ability as a universal image-matching method.
>
---
#### [new 270] SteadyDancer: Harmonized and Coherent Human Image Animation with First-Frame Preservation
- **分类: cs.CV**

- **简介: 该论文针对人体图像动画中的身份保持与运动控制难题，提出SteadyDancer框架。通过条件协调机制、协同姿态调制模块及分阶段解耦训练策略，实现首帧身份精准保留与动作连贯性，显著提升生成质量并降低训练成本。**

- **链接: [https://arxiv.org/pdf/2511.19320v1](https://arxiv.org/pdf/2511.19320v1)**

> **作者:** Jiaming Zhang; Shengming Cao; Rui Li; Xiaotong Zhao; Yutao Cui; Xinglin Hou; Gangshan Wu; Haolan Chen; Yu Xu; Limin Wang; Kai Ma
>
> **备注:** 10 pages, with supp
>
> **摘要:** Preserving first-frame identity while ensuring precise motion control is a fundamental challenge in human image animation. The Image-to-Motion Binding process of the dominant Reference-to-Video (R2V) paradigm overlooks critical spatio-temporal misalignments common in real-world applications, leading to failures such as identity drift and visual artifacts. We introduce SteadyDancer, an Image-to-Video (I2V) paradigm-based framework that achieves harmonized and coherent animation and is the first to ensure first-frame preservation robustly. Firstly, we propose a Condition-Reconciliation Mechanism to harmonize the two conflicting conditions, enabling precise control without sacrificing fidelity. Secondly, we design Synergistic Pose Modulation Modules to generate an adaptive and coherent pose representation that is highly compatible with the reference image. Finally, we employ a Staged Decoupled-Objective Training Pipeline that hierarchically optimizes the model for motion fidelity, visual quality, and temporal coherence. Experiments demonstrate that SteadyDancer achieves state-of-the-art performance in both appearance fidelity and motion control, while requiring significantly fewer training resources than comparable methods.
>
---
#### [new 271] MambaTAD: When State-Space Models Meet Long-Range Temporal Action Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对长时序动作检测任务，解决传统方法在处理长跨度动作时全局感知不足与计算效率低的问题。提出MambaTAD模型，引入双向状态空间模块与全局特征融合头，实现端到端的长程建模与高效检测，显著提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.17929v1](https://arxiv.org/pdf/2511.17929v1)**

> **作者:** Hui Lu; Yi Yu; Shijian Lu; Deepu Rajan; Boon Poh Ng; Alex C. Kot; Xudong Jiang
>
> **摘要:** Temporal Action Detection (TAD) aims to identify and localize actions by determining their starting and ending frames within untrimmed videos. Recent Structured State-Space Models such as Mamba have demonstrated potential in TAD due to their long-range modeling capability and linear computational complexity. On the other hand, structured state-space models often face two key challenges in TAD, namely, decay of temporal context due to recursive processing and self-element conflict during global visual context modeling, which become more severe while handling long-span action instances. Additionally, traditional methods for TAD struggle with detecting long-span action instances due to a lack of global awareness and inefficient detection heads. This paper presents MambaTAD, a new state-space TAD model that introduces long-range modeling and global feature detection capabilities for accurate temporal action detection. MambaTAD comprises two novel designs that complement each other with superior TAD performance. First, it introduces a Diagonal-Masked Bidirectional State-Space (DMBSS) module which effectively facilitates global feature fusion and temporal action detection. Second, it introduces a global feature fusion head that refines the detection progressively with multi-granularity features and global awareness. In addition, MambaTAD tackles TAD in an end-to-end one-stage manner using a new state-space temporal adapter(SSTA) which reduces network parameters and computation cost with linear complexity. Extensive experiments show that MambaTAD achieves superior TAD performance consistently across multiple public benchmarks.
>
---
#### [new 272] SFHand: A Streaming Framework for Language-guided 3D Hand Forecasting and Embodied Manipulation
- **分类: cs.CV**

- **简介: 该论文提出SFHand，首个面向语言引导的3D手部流式预测框架，解决实时人机交互中手部状态预测与语言指令融合难题。通过流式自回归架构与区域增强记忆层，实现从视频与语言输入中连续预测手部类型、位置、姿态和轨迹。构建EgoHaFL大规模数据集，验证其在手部预测与具身操作任务中的优越性能。**

- **链接: [https://arxiv.org/pdf/2511.18127v1](https://arxiv.org/pdf/2511.18127v1)**

> **作者:** Ruicong Liu; Yifei Huang; Liangyang Ouyang; Caixin Kang; Yoichi Sato
>
> **摘要:** Real-time 3D hand forecasting is a critical component for fluid human-computer interaction in applications like AR and assistive robotics. However, existing methods are ill-suited for these scenarios, as they typically require offline access to accumulated video sequences and cannot incorporate language guidance that conveys task intent. To overcome these limitations, we introduce SFHand, the first streaming framework for language-guided 3D hand forecasting. SFHand autoregressively predicts a comprehensive set of future 3D hand states, including hand type, 2D bounding box, 3D pose, and trajectory, from a continuous stream of video and language instructions. Our framework combines a streaming autoregressive architecture with an ROI-enhanced memory layer, capturing temporal context while focusing on salient hand-centric regions. To enable this research, we also introduce EgoHaFL, the first large-scale dataset featuring synchronized 3D hand poses and language instructions. We demonstrate that SFHand achieves new state-of-the-art results in 3D hand forecasting, outperforming prior work by a significant margin of up to 35.8%. Furthermore, we show the practical utility of our learned representations by transferring them to downstream embodied manipulation tasks, improving task success rates by up to 13.4% on multiple benchmarks. Dataset page: https://huggingface.co/datasets/ut-vision/EgoHaFL, project page: https://github.com/ut-vision/SFHand.
>
---
#### [new 273] Diffusion Reconstruction-based Data Likelihood Estimation for Core-Set Selection
- **分类: cs.CV**

- **简介: 该论文针对核心集选择任务，解决现有方法依赖启发式评分、忽视数据似然的问题。提出基于扩散模型的重建偏差方法，通过马尔可夫扩散过程的证据下界建立重构误差与数据似然的理论关联，实现分布感知的优选评分，并设计高效方法确定最优重构步数，实验表明仅用50%数据即接近全量训练效果。**

- **链接: [https://arxiv.org/pdf/2511.19274v1](https://arxiv.org/pdf/2511.19274v1)**

> **作者:** Mingyang Chen; Jiawei Du; Bo Huang; Yi Wang; Xiaobo Zhang; Wei Wang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Existing core-set selection methods predominantly rely on heuristic scoring signals such as training dynamics or model uncertainty, lacking explicit modeling of data likelihood. This omission may hinder the constructed subset from capturing subtle yet critical distributional structures that underpin effective model training. In this work, we propose a novel, theoretically grounded approach that leverages diffusion models to estimate data likelihood via reconstruction deviation induced by partial reverse denoising. Specifically, we establish a formal connection between reconstruction error and data likelihood, grounded in the Evidence Lower Bound (ELBO) of Markovian diffusion processes, thereby enabling a principled, distribution-aware scoring criterion for data selection. Complementarily, we introduce an efficient information-theoretic method to identify the optimal reconstruction timestep, ensuring that the deviation provides a reliable signal indicative of underlying data likelihood. Extensive experiments on ImageNet demonstrate that reconstruction deviation offers an effective scoring criterion, consistently outperforming existing baselines across selection ratios, and closely matching full-data training using only 50% of the data. Further analysis shows that the likelihood-informed nature of our score reveals informative insights in data selection, shedding light on the interplay between data distributional characteristics and model learning preferences.
>
---
#### [new 274] Multimodal AI for Body Fat Estimation: Computer Vision and Anthropometry with DEXA Benchmarks
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对体脂率精准测量成本高、难普及的问题，提出基于计算机视觉与人体测量数据的多模态AI估测方法。利用自建数据集（图像与体测数据），构建ResNet图像模型与回归模型，实现低成本体脂率估算，图像模型达RMSE 4.44%、R² 0.807，验证了AI在健康监测中的可行性。**

- **链接: [https://arxiv.org/pdf/2511.17576v1](https://arxiv.org/pdf/2511.17576v1)**

> **作者:** Rayan Aldajani
>
> **备注:** 2 pages, 2 figures, accepted at IEEE CASCON 2025
>
> **摘要:** Tracking body fat percentage is essential for effective weight management, yet gold-standard methods such as DEXA scans remain expensive and inaccessible for most people. This study evaluates the feasibility of artificial intelligence (AI) models as low-cost alternatives using frontal body images and basic anthropometric data. The dataset consists of 535 samples: 253 cases with recorded anthropometric measurements (weight, height, neck, ankle, and wrist) and 282 images obtained via web scraping from Reddit posts with self-reported body fat percentages, including some reported as DEXA-derived by the original posters. Because no public datasets exist for computer-vision-based body fat estimation, this dataset was compiled specifically for this study. Two approaches were developed: (1) ResNet-based image models and (2) regression models using anthropometric measurements. A multimodal fusion framework is also outlined for future expansion once paired datasets become available. The image-based model achieved a Root Mean Square Error (RMSE) of 4.44% and a Coefficient of Determination (R^2) of 0.807. These findings demonstrate that AI-assisted models can offer accessible and low-cost body fat estimates, supporting future consumer applications in health and fitness.
>
---
#### [new 275] Data Augmentation Strategies for Robust Lane Marking Detection
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对自动驾驶中侧装摄像头的车道线检测泛化问题，提出基于生成式AI的数据增强方法。通过几何变换、AI修复和车辆遮挡模拟，生成符合实际部署视角的训练数据，提升SCNN和UFLDv2模型在阴影等复杂条件下的鲁棒性，有效缓解领域偏移问题。**

- **链接: [https://arxiv.org/pdf/2511.18668v1](https://arxiv.org/pdf/2511.18668v1)**

> **作者:** Flora Lian; Dinh Quang Huynh; Hector Penades; J. Stephany Berrio Perez; Mao Shan; Stewart Worrall
>
> **备注:** 8 figures, 2 tables, 10 pages, ACRA, Australasian conference on robotics and automation
>
> **摘要:** Robust lane detection is essential for advanced driver assistance and autonomous driving, yet models trained on public datasets such as CULane often fail to generalise across different camera viewpoints. This paper addresses the challenge of domain shift for side-mounted cameras used in lane-wheel monitoring by introducing a generative AI-based data enhancement pipeline. The approach combines geometric perspective transformation, AI-driven inpainting, and vehicle body overlays to simulate deployment-specific viewpoints while preserving lane continuity. We evaluated the effectiveness of the proposed augmentation in two state-of-the-art models, SCNN and UFLDv2. With the augmented data trained, both models show improved robustness to different conditions, including shadows. The experimental results demonstrate gains in precision, recall, and F1 score compared to the pre-trained model. By bridging the gap between widely available datasets and deployment-specific scenarios, our method provides a scalable and practical framework to improve the reliability of lane detection in a pilot deployment scenario.
>
---
#### [new 276] Can Modern Vision Models Understand the Difference Between an Object and a Look-alike?
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型对“真实物体”与“外观相似物”的区分能力。针对现有模型在感知差异上的不足，构建了RoLA数据集，通过对比真实与仿制品图像，探索CLIP模型在嵌入空间中区分二者的能力，并提出方向迁移方法提升跨模态检索与图像描述性能。**

- **链接: [https://arxiv.org/pdf/2511.19200v1](https://arxiv.org/pdf/2511.19200v1)**

> **作者:** Itay Cohen; Ethan Fetaya; Amir Rosenfeld
>
> **摘要:** Recent advances in computer vision have yielded models with strong performance on recognition benchmarks; however, significant gaps remain in comparison to human perception. One subtle ability is to judge whether an image looks like a given object without being an instance of that object. We study whether vision-language models such as CLIP capture this distinction. We curated a dataset named RoLA (Real or Lookalike) of real and lookalike exemplars (e.g., toys, statues, drawings, pareidolia) across multiple categories, and first evaluate a prompt-based baseline with paired "real"/"lookalike" prompts. We then estimate a direction in CLIP's embedding space that moves representations between real and lookalike. Applying this direction to image and text embeddings improves discrimination in cross-modal retrieval on Conceptual12M, and also enhances captions produced by a CLIP prefix captioner.
>
---
#### [new 277] Is Complete Labeling Necessary? Understanding Active Learning in Longitudinal Medical Imaging
- **分类: cs.CV**

- **简介: 该论文针对纵向医学影像变化检测任务，解决标注成本高的问题。提出LMI-AL框架，通过配对差分基线与随访图像，迭代选择最具信息量的图像对进行标注，显著降低标注需求。实验表明，仅需不足8%标注数据即可达到全标注模型性能。**

- **链接: [https://arxiv.org/pdf/2511.18007v1](https://arxiv.org/pdf/2511.18007v1)**

> **作者:** Siteng Ma; Honghui Du; Prateek Mathur; Brendan S. Kelly; Ronan P. Killeen; Aonghus Lawlor; Ruihai Dong
>
> **备注:** This paper has been accepted at International Joint Conference on Neural Networks (IJCNN) 2025
>
> **摘要:** Detecting changes in longitudinal medical imaging using deep learning requires a substantial amount of accurately labeled data. However, labeling these images is notably more costly and time-consuming than labeling other image types, as it requires labeling across various time points, where new lesions can be minor, and subtle changes are easily missed. Deep Active Learning (DAL) has shown promise in minimizing labeling costs by selectively querying the most informative samples, but existing studies have primarily focused on static tasks like classification and segmentation. Consequently, the conventional DAL approach cannot be directly applied to change detection tasks, which involve identifying subtle differences across multiple images. In this study, we propose a novel DAL framework, named Longitudinal Medical Imaging Active Learning (LMI-AL), tailored specifically for longitudinal medical imaging. By pairing and differencing all 2D slices from baseline and follow-up 3D images, LMI-AL iteratively selects the most informative pairs for labeling using DAL, training a deep learning model with minimal manual annotation. Experimental results demonstrate that, with less than 8% of the data labeled, LMI-AL can achieve performance comparable to models trained on fully labeled datasets. We also provide a detailed analysis of the method's performance, as guidance for future research. The code is publicly available at https://github.com/HelenMa9998/Longitudinal_AL.
>
---
#### [new 278] FineXtrol: Controllable Motion Generation via Fine-Grained Text
- **分类: cs.CV**

- **简介: 该论文提出FineXtrol框架，解决文本驱动动作生成中控制精度低、时间不一致与计算成本高的问题。通过细粒度、时序感知的文本控制信号与层级对比学习模块，提升对身体部位运动的精准控制，实现高效高可控的动作生成。**

- **链接: [https://arxiv.org/pdf/2511.18927v1](https://arxiv.org/pdf/2511.18927v1)**

> **作者:** Keming Shen; Bizhu Wu; Junliang Chen; Xiaoqin Wang; Linlin Shen
>
> **备注:** 20 pages, 14 figures, AAAI 2026
>
> **摘要:** Recent works have sought to enhance the controllability and precision of text-driven motion generation. Some approaches leverage large language models (LLMs) to produce more detailed texts, while others incorporate global 3D coordinate sequences as additional control signals. However, the former often introduces misaligned details and lacks explicit temporal cues, and the latter incurs significant computational cost when converting coordinates to standard motion representations. To address these issues, we propose FineXtrol, a novel control framework for efficient motion generation guided by temporally-aware, precise, user-friendly, and fine-grained textual control signals that describe specific body part movements over time. In support of this framework, we design a hierarchical contrastive learning module that encourages the text encoder to produce more discriminative embeddings for our novel control signals, thereby improving motion controllability. Quantitative results show that FineXtrol achieves strong performance in controllable motion generation, while qualitative analysis demonstrates its flexibility in directing specific body part movements.
>
---
#### [new 279] Any4D: Open-Prompt 4D Generation from Natural Language and Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Primitive Embodied World Models（PEWM），解决具身智能中视频生成依赖大量高维交互数据的问题。通过限制生成时长、利用视觉-语言模型与起止热图引导，实现语言与动作的细粒度对齐，提升数据效率与推理速度，支持复杂任务的组合泛化。**

- **链接: [https://arxiv.org/pdf/2511.18746v1](https://arxiv.org/pdf/2511.18746v1)**

> **作者:** Hao Li; Qiao Sun
>
> **摘要:** While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a \textit{"GPT moment"} in the embodied domain. There is a naive observation: \textit{the diversity of embodied data far exceeds the relatively small space of possible primitive motions}. Based on this insight, we propose \textbf{Primitive Embodied World Models} (PEWM), which restricts video generation to fixed shorter horizons, our approach \textit{1) enables} fine-grained alignment between linguistic concepts and visual representations of robotic actions, \textit{2) reduces} learning complexity, \textit{3) improves} data efficiency in embodied data collection, and \textit{4) decreases} inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence.
>
---
#### [new 280] FlowSteer: Guiding Few-Step Image Synthesis with Authentic Trajectories
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对流匹配图像生成中少步采样效率低的问题，聚焦被忽视的ReFlow方法。提出FlowSteer，通过在线轨迹对齐与对抗性蒸馏，引导学生模型沿教师真实生成轨迹演进，并修复了调度器缺陷，显著提升少步生成质量。**

- **链接: [https://arxiv.org/pdf/2511.18834v1](https://arxiv.org/pdf/2511.18834v1)**

> **作者:** Lei Ke; Hubery Yin; Gongye Liu; Zhengyao Lv; Jingcai Guo; Chen Li; Wenhan Luo; Yujiu Yang; Jing Lyu
>
> **备注:** Few-Step Image Synthesis
>
> **摘要:** With the success of flow matching in visual generation, sampling efficiency remains a critical bottleneck for its practical application. Among flow models' accelerating methods, ReFlow has been somehow overlooked although it has theoretical consistency with flow matching. This is primarily due to its suboptimal performance in practical scenarios compared to consistency distillation and score distillation. In this work, we investigate this issue within the ReFlow framework and propose FlowSteer, a method unlocks the potential of ReFlow-based distillation by guiding the student along teacher's authentic generation trajectories. We first identify that Piecewised ReFlow's performance is hampered by a critical distribution mismatch during the training and propose Online Trajectory Alignment(OTA) to resolve it. Then, we introduce a adversarial distillation objective applied directly on the ODE trajectory, improving the student's adherence to the teacher's generation trajectory. Furthermore, we find and fix a previously undiscovered flaw in the widely-used FlowMatchEulerDiscreteScheduler that largely degrades few-step inference quality. Our experiment result on SD3 demonstrates our method's efficacy.
>
---
#### [new 281] BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction
- **分类: cs.CV**

- **简介: 该论文针对森林火灾风险预测任务，解决长期时空建模中多源数据融合不足的问题。构建了覆盖240万公顷、25年日尺度的多因子数据集BCWildfire，包含38个变量，并基于此评估多种深度学习模型，揭示关键驱动因素与模型性能关系。**

- **链接: [https://arxiv.org/pdf/2511.17597v1](https://arxiv.org/pdf/2511.17597v1)**

> **作者:** Zhengsen Xu; Sibo Cheng; Hongjie He; Lanying Wang; Wentao Sun; Jonathan Li; Lincoln Linlin Xu
>
> **备注:** This paper has been accepted by AAAI-26
>
> **摘要:** Wildfire risk prediction remains a critical yet challenging task due to the complex interactions among fuel conditions, meteorology, topography, and human activity. Despite growing interest in data-driven approaches, publicly available benchmark datasets that support long-term temporal modeling, large-scale spatial coverage, and multimodal drivers remain scarce. To address this gap, we present a 25-year, daily-resolution wildfire dataset covering 240 million hectares across British Columbia and surrounding regions. The dataset includes 38 covariates, encompassing active fire detections, weather variables, fuel conditions, terrain features, and anthropogenic factors. Using this benchmark, we evaluate a diverse set of time-series forecasting models, including CNN-based, linear-based, Transformer-based, and Mamba-based architectures. We also investigate effectiveness of position embedding and the relative importance of different fire-driving factors. The dataset and the corresponding code can be found at https://github.com/SynUW/mmFire
>
---
#### [new 282] Alias-free 4D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对动态场景重建中4D高斯点云渲染时因焦距调整或距离变化引发的高频伪影问题，提出无混叠4D高斯点云方法。通过推导最大采样频率公式，引入自适应尺度滤波器与尺度损失，有效控制采样频率，消除伪影并减少冗余点，提升多视角视频重建质量。**

- **链接: [https://arxiv.org/pdf/2511.18367v1](https://arxiv.org/pdf/2511.18367v1)**

> **作者:** Zilong Chen; Huan-ang Gao; Delin Qu; Haohan Chi; Hao Tang; Kai Zhang; Hao Zhao
>
> **备注:** Project page: https://4d-alias-free.github.io/4D-Alias-free/
>
> **摘要:** Existing dynamic scene reconstruction methods based on Gaussian Splatting enable real-time rendering and generate realistic images. However, adjusting the camera's focal length or the distance between Gaussian primitives and the camera to modify rendering resolution often introduces strong artifacts, stemming from the frequency constraints of 4D Gaussians and Gaussian scale mismatch induced by the 2D dilated filter. To address this, we derive a maximum sampling frequency formulation for 4D Gaussian Splatting and introduce a 4D scale-adaptive filter and scale loss, which flexibly regulates the sampling frequency of 4D Gaussian Splatting. Our approach eliminates high-frequency artifacts under increased rendering frequencies while effectively reducing redundant Gaussians in multi-view video reconstruction. We validate the proposed method through monocular and multi-view video reconstruction experiments.Ours project page: https://4d-alias-free.github.io/4D-Alias-free/
>
---
#### [new 283] VCU-Bridge: Hierarchical Visual Connotation Understanding via Semantic Bridging
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大模型在视觉理解中缺乏层次化推理的问题，提出VCU-Bridge框架，构建层级化视觉语义理解任务。通过引入从感知到抽象的语义桥接机制与可诊断的HVCU-Bench基准，揭示模型性能随层级上升而下降的瓶颈，并验证低层能力提升对高层推理的显著促进作用。**

- **链接: [https://arxiv.org/pdf/2511.18121v1](https://arxiv.org/pdf/2511.18121v1)**

> **作者:** Ming Zhong; Yuanlei Wang; Liuzhou Zhang; Arctanx An; Renrui Zhang; Hao Liang; Ming Lu; Ying Shen; Wentao Zhang
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel on benchmarks, their processing paradigm differs from the human ability to integrate visual information. Unlike humans who naturally bridge details and high-level concepts, models tend to treat these elements in isolation. Prevailing evaluation protocols often decouple low-level perception from high-level reasoning, overlooking their semantic and causal dependencies, which yields non-diagnostic results and obscures performance bottlenecks. We present VCU-Bridge, a framework that operationalizes a human-like hierarchy of visual connotation understanding: multi-level reasoning that advances from foundational perception through semantic bridging to abstract connotation, with an explicit evidence-to-inference trace from concrete cues to abstract conclusions. Building on this framework, we construct HVCU-Bench, a benchmark for hierarchical visual connotation understanding with explicit, level-wise diagnostics. Comprehensive experiments demonstrate a consistent decline in performance as reasoning progresses to higher levels. We further develop a data generation pipeline for instruction tuning guided by Monte Carlo Tree Search (MCTS) and show that strengthening low-level capabilities yields measurable gains at higher levels. Interestingly, it not only improves on HVCU-Bench but also brings benefits on general benchmarks (average +2.53%), especially with substantial gains on MMStar (+7.26%), demonstrating the significance of the hierarchical thinking pattern and its effectiveness in enhancing MLLM capabilities. The project page is at https://vcu-bridge.github.io .
>
---
#### [new 284] Modality-Collaborative Low-Rank Decomposers for Few-Shot Video Domain Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究少样本视频域适应（FSVDA）任务，针对视频多模态特性下域偏移导致的特征融合失效问题，提出模态协同低秩分解框架（MC-LRD）。通过分解各模态的独有与共享特征，并引入路由机制与一致性损失，实现更优的域对齐与模态协同，显著提升跨域泛化性能。**

- **链接: [https://arxiv.org/pdf/2511.18711v1](https://arxiv.org/pdf/2511.18711v1)**

> **作者:** Yuyang Wanyan; Xiaoshan Yang; Weiming Dong; Changsheng Xu
>
> **摘要:** In this paper, we study the challenging task of Few-Shot Video Domain Adaptation (FSVDA). The multimodal nature of videos introduces unique challenges, necessitating the simultaneous consideration of both domain alignment and modality collaboration in a few-shot scenario, which is ignored in previous literature. We observe that, under the influence of domain shift, the generalization performance on the target domain of each individual modality, as well as that of fused multimodal features, is constrained. Because each modality is comprised of coupled features with multiple components that exhibit different domain shifts. This variability increases the complexity of domain adaptation, thereby reducing the effectiveness of multimodal feature integration. To address these challenges, we introduce a novel framework of Modality-Collaborative LowRank Decomposers (MC-LRD) to decompose modality-unique and modality-shared features with different domain shift levels from each modality that are more friendly for domain alignment. The MC-LRD comprises multiple decomposers for each modality and Multimodal Decomposition Routers (MDR). Each decomposer has progressively shared parameters across different modalities. The MDR is leveraged to selectively activate the decomposers to produce modality-unique and modality-shared features. To ensure efficient decomposition, we apply orthogonal decorrelation constraints separately to decomposers and subrouters, enhancing their diversity. Furthermore, we propose a cross-domain activation consistency loss to guarantee that target and source samples of the same category exhibit consistent activation preferences of the decomposers, thereby facilitating domain alignment. Extensive experimental results on three public benchmarks demonstrate that our model achieves significant improvements over existing methods.
>
---
#### [new 285] FastMMoE: Accelerating Multimodal Large Language Models through Dynamic Expert Activation and Routing-Aware Token Pruning
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多模态大模型推理慢的问题，提出FastMMoE框架。通过动态专家激活与路由感知的视觉令牌剪枝，减少冗余计算。在不训练的前提下，显著降低计算量（最多55% FLOPs），同时保持95.5%性能，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17885v1](https://arxiv.org/pdf/2511.17885v1)**

> **作者:** Guoyang Xia; Yifeng Ding; Fengfa Li; Lei Ren; Wei Chen; Fangxiang Feng; Xiaojie Wang
>
> **摘要:** Multimodal large language models (MLLMs) have achieved impressive performance, but high-resolution visual inputs result in long sequences of visual tokens and substantial inference latency. Reducing redundant visual tokens is critical to ease computational/memory burdens while preserving performance, enabling MLLM deployment in resource-constrained or latency-sensitive scenarios. Current visual token pruning methods mainly rely on attention-based redundancy analysis and are tailored to dense architectures. We propose Fast Multimodal Mixture-of-Experts (FastMMoE), a training-free acceleration framework for mixture-of-experts (MoE) based MLLMs, developed from a routing analysis perspective. FastMMoE combines two complementary strategies: (i) expert activation reduction for visual tokens to minimize unnecessary expert computation; and (ii) routing-aware token pruning that leverages similarity in routing probability distributions to identify and remove highly redundant visual tokens. Experiments on large-scale MoE-MLLMs such as DeepSeek-VL2 and InternVL3.5 demonstrate that FastMMoE can reduce FLOPs by up to 55.0% while retaining approximately 95.5% of the original performance, consistently outperforming dense-model pruning baselines including FastV and SparseVLM across multiple retention rates.
>
---
#### [new 286] Now You See It, Now You Don't - Instant Concept Erasure for Safe Text-to-Image and Video Generation
- **分类: cs.CV**

- **简介: 该论文针对文本到图像/视频生成中的安全问题，提出无需训练、零开销的即时概念擦除方法ICE。解决现有方法依赖重训练、易受攻击及擦除后产生副作用的问题。通过构建擦除与保留子空间并用闭式投影器抑制其交集，实现精确、持久且鲁棒的概念移除，适用于T2I与T2V模型。**

- **链接: [https://arxiv.org/pdf/2511.18684v1](https://arxiv.org/pdf/2511.18684v1)**

> **作者:** Shristi Das Biswas; Arani Roy; Kaushik Roy
>
> **摘要:** Robust concept removal for text-to-image (T2I) and text-to-video (T2V) models is essential for their safe deployment. Existing methods, however, suffer from costly retraining, inference overhead, or vulnerability to adversarial attacks. Crucially, they rarely model the latent semantic overlap between the target erase concept and surrounding content -- causing collateral damage post-erasure -- and even fewer methods work reliably across both T2I and T2V domains. We introduce Instant Concept Erasure (ICE), a training-free, modality-agnostic, one-shot weight modification approach that achieves precise, persistent unlearning with zero overhead. ICE defines erase and preserve subspaces using anisotropic energy-weighted scaling, then explicitly regularises against their intersection using a unique, closed-form overlap projector. We pose a convex and Lipschitz-bounded Spectral Unlearning Objective, balancing erasure fidelity and intersection preservation, that admits a stable and unique analytical solution. This solution defines a dissociation operator that is translated to the model's text-conditioning layers, making the edit permanent and runtime-free. Across targeted removals of artistic styles, objects, identities, and explicit content, ICE efficiently achieves strong erasure with improved robustness to red-teaming, all while causing only minimal degradation of original generative abilities in both T2I and T2V models.
>
---
#### [new 287] Seeing What Matters: Visual Preference Policy Optimization for Visual Generation
- **分类: cs.CV**

- **简介: 该论文针对视觉生成中偏好优化的粗粒度奖励问题，提出ViPO方法。通过引入感知结构模块，将单一奖励细化为像素级优势图，增强对图像/视频中重要区域的优化能力，提升生成质量与泛化性，且兼容现有训练流程。**

- **链接: [https://arxiv.org/pdf/2511.18719v1](https://arxiv.org/pdf/2511.18719v1)**

> **作者:** Ziqi Ni; Yuanzhi Liang; Rui Li; Yi Zhou; Haibing Huang; Chi Zhang; Xuelong Li
>
> **摘要:** Reinforcement learning (RL) has become a powerful tool for post-training visual generative models, with Group Relative Policy Optimization (GRPO) increasingly used to align generators with human preferences. However, existing GRPO pipelines rely on a single scalar reward per sample, treating each image or video as a holistic entity and ignoring the rich spatial and temporal structure of visual content. This coarse supervision hinders the correction of localized artifacts and the modeling of fine-grained perceptual cues. We introduce Visual Preference Policy Optimization (ViPO), a GRPO variant that lifts scalar feedback into structured, pixel-level advantages. ViPO employs a Perceptual Structuring Module that uses pretrained vision backbones to construct spatially and temporally aware advantage maps, redistributing optimization pressure toward perceptually important regions while preserving the stability of standard GRPO. Across both image and video benchmarks, ViPO consistently outperforms vanilla GRPO, improving in-domain alignment with human-preference rewards and enhancing generalization on out-of-domain evaluations. The method is architecture-agnostic, lightweight, and fully compatible with existing GRPO training pipelines, providing a more expressive and informative learning signal for visual generation.
>
---
#### [new 288] LAA3D: A Benchmark of Detecting and Tracking Low-Altitude Aircraft in 3D Space
- **分类: cs.CV**

- **简介: 该论文针对低空飞行器3D感知数据稀缺问题，提出LAA3D数据集，包含15,000实拍图像与600,000合成帧，涵盖eVTOL、MAV等多类飞行器。构建了支持3D检测、多目标跟踪与6-DoF姿态估计的基准，提出单目检测基线MonoLAA，实现良好仿真到现实的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19057v1](https://arxiv.org/pdf/2511.19057v1)**

> **作者:** Hai Wu; Shuai Tang; Jiale Wang; Longkun Zou; Mingyue Guo; Rongqin Liang; Ke Chen; Yaowei Wang
>
> **备注:** 25 pages
>
> **摘要:** Perception of Low-Altitude Aircraft (LAA) in 3D space enables precise 3D object localization and behavior understanding. However, datasets tailored for 3D LAA perception remain scarce. To address this gap, we present LAA3D, a large-scale dataset designed to advance 3D detection and tracking of low-altitude aerial vehicles. LAA3D contains 15,000 real images and 600,000 synthetic frames, captured across diverse scenarios, including urban and suburban environments. It covers multiple aerial object categories, including electric Vertical Take-Off and Landing (eVTOL) aircraft, Micro Aerial Vehicles (MAVs), and Helicopters. Each instance is annotated with 3D bounding box, class label, and instance identity, supporting tasks such as 3D object detection, 3D multi-object tracking (MOT), and 6-DoF pose estimation. Besides, we establish the LAA3D Benchmark, integrating multiple tasks and methods with unified evaluation protocols for comparison. Furthermore, we propose MonoLAA, a monocular 3D detection baseline, achieving robust 3D localization from zoom cameras with varying focal lengths. Models pretrained on synthetic images transfer effectively to real-world data with fine-tuning, demonstrating strong sim-to-real generalization. Our LAA3D provides a comprehensive foundation for future research in low-altitude 3D object perception.
>
---
#### [new 289] DetAny4D: Detect Anything 4D Temporally in a Streaming RGB Video
- **分类: cs.CV**

- **简介: 该论文针对开放集4D目标检测任务，解决帧间不一致与误差传播问题。提出DetAny4D框架，基于大规模DA4D数据集，通过多模态融合与时空解码器实现端到端3D边界框预测，提升检测精度与时间稳定性。**

- **链接: [https://arxiv.org/pdf/2511.18814v1](https://arxiv.org/pdf/2511.18814v1)**

> **作者:** Jiawei Hou; Shenghao Zhang; Can Wang; Zheng Gu; Yonggen Ling; Taiping Zeng; Xiangyang Xue; Jingbo Zhang
>
> **摘要:** Reliable 4D object detection, which refers to 3D object detection in streaming video, is crucial for perceiving and understanding the real world. Existing open-set 4D object detection methods typically make predictions on a frame-by-frame basis without modeling temporal consistency, or rely on complex multi-stage pipelines that are prone to error propagation across cascaded stages. Progress in this area has been hindered by the lack of large-scale datasets that capture continuous reliable 3D bounding box (b-box) annotations. To overcome these challenges, we first introduce DA4D, a large-scale 4D detection dataset containing over 280k sequences with high-quality b-box annotations collected under diverse conditions. Building on DA4D, we propose DetAny4D, an open-set end-to-end framework that predicts 3D b-boxes directly from sequential inputs. DetAny4D fuses multi-modal features from pre-trained foundational models and designs a geometry-aware spatiotemporal decoder to effectively capture both spatial and temporal dynamics. Furthermore, it adopts a multi-task learning architecture coupled with a dedicated training strategy to maintain global consistency across sequences of varying lengths. Extensive experiments show that DetAny4D achieves competitive detection accuracy and significantly improves temporal stability, effectively addressing long-standing issues of jitter and inconsistency in 4D object detection. Data and code will be released upon acceptance.
>
---
#### [new 290] Understanding Counting Mechanisms in Large Language and Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究大语言模型（LLM）与视觉-语言模型（LVLM）在计数任务中的数值表征机制。通过设计CountScope工具，结合因果中介与激活拼接分析，发现模型通过分层方式逐步构建数字表征，存在可转移的内部计数器，且依赖结构线索如分隔符。研究揭示了计数过程的系统性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.17699v1](https://arxiv.org/pdf/2511.17699v1)**

> **作者:** Hosein Hasani; Amirmohammad Izadi; Fatemeh Askari; Mobin Bagherian; Sadegh Mohammadian; Mohammad Izadi; Mahdieh Soleymani Baghshah
>
> **摘要:** This paper examines how large language models (LLMs) and large vision-language models (LVLMs) represent and compute numerical information in counting tasks. We use controlled experiments with repeated textual and visual items and analyze model behavior through causal mediation and activation patching. To this end, we design a specialized tool, CountScope, for mechanistic interpretability of numerical content. Results show that individual tokens or visual features encode latent positional count information that can be extracted and transferred across contexts. Layerwise analyses reveal a progressive emergence of numerical representations, with lower layers encoding small counts and higher layers representing larger ones. We identify an internal counter mechanism that updates with each item, stored mainly in the final token or region and transferable between contexts. In LVLMs, numerical information also appears in visual embeddings, shifting between background and foreground regions depending on spatial composition. Models rely on structural cues such as separators in text, which act as shortcuts for tracking item counts and influence the accuracy of numerical predictions. Overall, counting emerges as a structured, layerwise process in LLMs and follows the same general pattern in LVLMs, shaped by the properties of the vision encoder.
>
---
#### [new 291] RegDeepLab: A Two-Stage Decoupled Framework for Interpretable Embryo Fragmentation Grading
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对辅助生殖中胚胎碎片化程度自动分级难题，提出RegDeepLab框架。通过双阶段解耦的多任务学习，融合语义分割与多尺度回归，解决模型可解释性与精度矛盾问题，实现高精度、可解释的临床辅助分级。**

- **链接: [https://arxiv.org/pdf/2511.18454v1](https://arxiv.org/pdf/2511.18454v1)**

> **作者:** Ming-Jhe Lee
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** The degree of embryo fragmentation serves as a critical morphological indicator for assessing embryo developmental potential in In Vitro Fertilization (IVF) clinical decision-making. However, current manual grading processes are not only time-consuming but also limited by significant inter-observer variability and efficiency bottlenecks. Although deep learning has demonstrated potential in automated grading in recent years, existing solutions face a significant challenge: pure regression models lack the visual explainability required for clinical practice, while pure segmentation models struggle to directly translate pixel-level masks into precise clinical grades. This study proposes RegDeepLab, a dual-branch Multi-Task Learning (MTL) framework that integrates State-of-the-Art (SOTA) semantic segmentation (DeepLabV3+) with a multi-scale regression head. Addressing the common issues of "Gradient Conflict" and "Negative Transfer" in multi-task training, we propose a "Two-Stage Decoupled Training Strategy." Experimental results demonstrate that while standard end-to-end MTL training can minimize grading error (MAE=0.046) through our designed "Feature Injection" mechanism, it compromises the integrity of segmentation boundaries. In contrast, our decoupled strategy successfully provides robust and high-precision grading predictions while preserving SOTA-level segmentation accuracy (Dice=0.729). Furthermore, we introduce a "Range Loss" to effectively utilize large-scale discrete grading data for semi-supervised learning. This study ultimately presents a dual-module clinical auxiliary solution that combines high accuracy with visual explainability.
>
---
#### [new 292] Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation
- **分类: cs.CV**

- **简介: 该论文针对低光照与模糊图像的联合增强问题，提出一种无监督生成框架。利用视觉语言模型提供感知先验，通过自适应曲线调制、频域感知位置编码和递归相位域优化，实现对复杂光照与动态模糊的建模，显著提升图像恢复质量。**

- **链接: [https://arxiv.org/pdf/2511.18591v1](https://arxiv.org/pdf/2511.18591v1)**

> **作者:** Wei Dong; Han Zhou; Junwei Lin; Jun Chen
>
> **备注:** Accepted by AAAI 2026; First Var-based method for joint LLIE and deblurring
>
> **摘要:** Real-world dark images commonly exhibit not only low visibility and contrast but also complex noise and blur, posing significant restoration challenges. Existing methods often rely on paired data or fail to model dynamic illumination and blur characteristics, leading to poor generalization. To tackle this, we propose a generative framework based on visual autoregressive (VAR) modeling, guided by perceptual priors from the vision-language model (VLM). Specifically, to supply informative conditioning cues for VAR models, we deploy an adaptive curve estimation scheme to modulate the diverse illumination based on VLM-derived visibility scores. In addition, we integrate dynamic and spatial-frequency-aware Rotary Positional Encodings (SF-RoPE) into VAR to enhance its ability to model structures degraded by blur. Furthermore, we propose a recursive phase-domain modulation strategy that mitigates blur-induced artifacts in the phase domain via bounded iterative refinement guided by VLM-assessed blur scores. Our framework is fully unsupervised and achieves state-of-the-art performance on benchmark datasets.
>
---
#### [new 293] Benchmarking Corruption Robustness of LVLMs: A Discriminative Benchmark and Robustness Alignment Metric
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型（LVLMs）在视觉退化下的鲁棒性评估问题，提出Bench-C基准与鲁棒性对齐评分（RAS）。通过筛选高区分度样本并量化预测结构退化，揭示模型在退化下的错误自信、犹豫等行为模式，实现更精准的鲁棒性分析。**

- **链接: [https://arxiv.org/pdf/2511.19032v1](https://arxiv.org/pdf/2511.19032v1)**

> **作者:** Xiangjie Sui; Songyang Li; Hanwei Zhu; Baoliang Chen; Yuming Fang; Xin Sun
>
> **备注:** 15 pages
>
> **摘要:** Despite the remarkable reasoning abilities of large vision-language models (LVLMs), their robustness under visual corruptions remains insufficiently studied. Existing evaluation paradigms exhibit two major limitations: 1) the dominance of low-discriminative samples in current datasets masks the real robustness gap between models; and 2) conventional accuracy-based metric fail to capture the degradation of the underlying prediction structure. To bridge these gaps, we introduce Bench-C, a comprehensive benchmark emphasizing discriminative samples for assessing corruption robustness, where a selection strategy is proposed to jointly consider the prediction inconsistency under corruption and the semantic diversity. Furthermore, we propose the Robustness Alignment Score (RAS), a unified metric that measures degradation in logit-level prediction structure by considering the shifts in prediction uncertainty and calibration alignment. Comprehensive experiments and analysis reveal several interesting findings: 1) model behaviors exhibit distinguish patterns under corruptions, such as erroneous confidence and hesitation; 2) despite subtle corruption may lead to a slight accuracy gain, the overall prediction structure still degrades; 3) by decomposing corruption robustness into destructive and corrective components, the distinct failure and recovery patterns across models can be revealed.
>
---
#### [new 294] LumiTex: Towards High-Fidelity PBR Texture Generation with Illumination Context
- **分类: cs.CV**

- **简介: 该论文针对PBR纹理生成中的材料分解与视图一致性难题，提出LumiTex框架。通过多分支生成、光照感知注意力机制及几何引导修复模块，实现光照条件下高质量、无缝且一致的材质贴图生成，显著提升纹理保真度。**

- **链接: [https://arxiv.org/pdf/2511.19437v1](https://arxiv.org/pdf/2511.19437v1)**

> **作者:** Jingzhi Bao; Hongze Chen; Lingting Zhu; Chenyu Liu; Runze Zhang; Keyang Luo; Zeyu Hu; Weikai Chen; Yingda Yin; Xin Wang; Zehong Lin; Jun Zhang; Xiaoguang Han
>
> **备注:** Project page: https://lumitex.vercel.app
>
> **摘要:** Physically-based rendering (PBR) provides a principled standard for realistic material-lighting interactions in computer graphics. Despite recent advances in generating PBR textures, existing methods fail to address two fundamental challenges: 1) materials decomposition from image prompts under limited illumination cues, and 2) seamless and view-consistent texture completion. To this end, we propose LumiTex, an end-to-end framework that comprises three key components: (1) a multi-branch generation scheme that disentangles albedo and metallic-roughness under shared illumination priors for robust material understanding, (2) a lighting-aware material attention mechanism that injects illumination context into the decoding process for physically grounded generation of albedo, metallic, and roughness maps, and (3) a geometry-guided inpainting module based on a large view synthesis model that enriches texture coverage and ensures seamless, view-consistent UV completion. Extensive experiments demonstrate that LumiTex achieves state-of-the-art performance in texture quality, surpassing both existing open-source and commercial methods.
>
---
#### [new 295] Rad-GS: Radar-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Rad-GS，一种用于室外大场景的雷达-视觉融合4D SLAM系统，利用3D高斯作为可微分空间表示。通过融合雷达点云与多普勒信息，实现动态物体掩码，减少渲染伪影并提升定位精度；结合非同步图像优化全局高斯表示，增强纹理一致性和新视角合成质量；采用全局八叉树与针对性管理策略，降低噪声与内存消耗，实现了千米级真实场景重建。**

- **链接: [https://arxiv.org/pdf/2511.16091v1](https://arxiv.org/pdf/2511.16091v1)**

> **作者:** Renxiang Xiao; Wei Liu; Yuanfan Zhang; Yushuai Chen; Jinming Chen; Zilu Wang; Liang Hu
>
> **摘要:** We present Rad-GS, a 4D radar-camera SLAM system designed for kilometer-scale outdoor environments, utilizing 3D Gaussian as a differentiable spatial representation. Rad-GS combines the advantages of raw radar point cloud with Doppler information and geometrically enhanced point cloud to guide dynamic object masking in synchronized images, thereby alleviating rendering artifacts and improving localization accuracy. Additionally, unsynchronized image frames are leveraged to globally refine the 3D Gaussian representation, enhancing texture consistency and novel view synthesis fidelity. Furthermore, the global octree structure coupled with a targeted Gaussian primitive management strategy further suppresses noise and significantly reduces memory consumption in large-scale environments. Extensive experiments and ablation studies demonstrate that Rad-GS achieves performance comparable to traditional 3D Gaussian methods based on camera or LiDAR inputs, highlighting the feasibility of robust outdoor mapping using 4D mmWave radar. Real-world reconstruction at kilometer scale validates the potential of Rad-GS for large-scale scene reconstruction.
>
---
#### [new 296] BideDPO: Conditional Image Generation with Simultaneous Text and Condition Alignment
- **分类: cs.CV**

- **简介: 该论文研究条件图像生成任务，针对文本与条件图像之间的冲突问题。提出BideDPO框架，通过双向解耦的偏好优化减少梯度纠缠，并引入自适应损失平衡与迭代数据生成，提升文本匹配与条件遵循能力。在DualAlign和COCO上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2511.19268v1](https://arxiv.org/pdf/2511.19268v1)**

> **作者:** Dewei Zhou; Mingwei Li; Zongxin Yang; Yu Lu; Yunqiu Xu; Zhizhong Wang; Zeyi Huang; Yi Yang
>
> **备注:** 29 pages
>
> **摘要:** Conditional image generation enhances text-to-image synthesis with structural, spatial, or stylistic priors, but current methods face challenges in handling conflicts between sources. These include 1) input-level conflicts, where the conditioning image contradicts the text prompt, and 2) model-bias conflicts, where generative biases disrupt alignment even when conditions match the text. Addressing these conflicts requires nuanced solutions, which standard supervised fine-tuning struggles to provide. Preference-based optimization techniques like Direct Preference Optimization (DPO) show promise but are limited by gradient entanglement between text and condition signals and lack disentangled training data for multi-constraint tasks. To overcome this, we propose a bidirectionally decoupled DPO framework (BideDPO). Our method creates two disentangled preference pairs-one for the condition and one for the text-to reduce gradient entanglement. The influence of pairs is managed using an Adaptive Loss Balancing strategy for balanced optimization. We introduce an automated data pipeline to sample model outputs and generate conflict-aware data. This process is embedded in an iterative optimization strategy that refines both the model and the data. We construct a DualAlign benchmark to evaluate conflict resolution between text and condition. Experiments show BideDPO significantly improves text success rates (e.g., +35%) and condition adherence. We also validate our approach using the COCO dataset. Project Pages: https://limuloo.github.io/BideDPO/.
>
---
#### [new 297] Unified Deep Learning Platform for Dust and Fault Diagnosis in Solar Panels Using Thermal and Visual Imaging
- **分类: cs.CV**

- **简介: 该论文提出统一深度学习平台，用于太阳能板尘污与故障诊断。针对太阳能板因灰尘、裂纹等导致效率下降的问题，融合热成像与可见光图像，采用CNN、ResNet及KerNet模型，实现多任务联合检测，提升诊断精度与维护效率。**

- **链接: [https://arxiv.org/pdf/2511.18514v1](https://arxiv.org/pdf/2511.18514v1)**

> **作者:** Abishek Karthik; Sreya Mynampati; Pandiyaraju V
>
> **摘要:** Solar energy is one of the most abundant and tapped sources of renewable energies with enormous future potential. Solar panel output can vary widely with factors like intensity, temperature, dirt, debris and so on affecting it. We have implemented a model on detecting dust and fault on solar panels. These two applications are centralized as a single-platform and can be utilized for routine-maintenance and any other checks. These are checked against various parameters such as power output, sinusoidal wave (I-V component of solar cell), voltage across each solar cell and others. Firstly, we filter and preprocess the obtained images using gamma removal and Gaussian filtering methods alongside some predefined processes like normalization. The first application is to detect whether a solar cell is dusty or not based on various pre-determined metrics like shadowing, leaf, droppings, air pollution and from other human activities to extent of fine-granular solar modules. The other one is detecting faults and other such occurrences on solar panels like faults, cracks, cell malfunction using thermal imaging application. This centralized platform can be vital since solar panels have different efficiency across different geography (air and heat affect) and can also be utilized for small-scale house requirements to large-scale solar farm sustentation effectively. It incorporates CNN, ResNet models that with self-attention mechanisms-KerNet model which are used for classification and results in a fine-tuned system that detects dust or any fault occurring. Thus, this multi-application model proves to be efficient and optimized in detecting dust and faults on solar panels. We have performed various comparisons and findings that demonstrates that our model has better efficiency and accuracy results overall than existing models.
>
---
#### [new 298] Test-Time Temporal Sampling for Efficient MLLM Video Understanding
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLM）处理长视频时计算成本高的问题，提出测试时时间采样（T3S）方法。通过在推理时生成多个短而多样化的视频子序列，提升视觉覆盖并降低自注意力计算复杂度，实现高效准确的长视频理解，无需训练或模型修改。**

- **链接: [https://arxiv.org/pdf/2511.17945v1](https://arxiv.org/pdf/2511.17945v1)**

> **作者:** Kaibin Wang; Mingbao Lin
>
> **摘要:** Processing long videos with multimodal large language models (MLLMs) poses a significant computational challenge, as the model's self-attention mechanism scales quadratically with the number of video tokens, resulting in high computational demand and slow inference speed. Current solutions, such as rule-based sub-sampling, learned frame selector, or memory-based summarization, often introduce their own trade-offs: they compromise accuracy, necessitate additional training, or decrease inference speed. In this paper, we propose Test-Time Temporal Sampling (T3S), a training-free, plug-and-play inference wrapper that enables MLLMs to process long videos both efficiently and effectively. T3S exploits spatiotemporal redundancy by generating multiple short and diverse subsequences of video tokens at inference time, packing them within a single forward pass, and aggregating their predictions. This multi-subsequence formulation broadens visual coverage while reducing the computational cost of self-attention from $O(L^2)$ to $O(\sum_{i=1}^m α_i^2L^2)$, where $\sum_{i=1}^m α_i^2 < 1$. Extensive experiments on long video understanding benchmarks demonstrate that T3S improves accuracy by up to 3.1% and reduces first token delay by $2.04\times$, all with minimal integration effort. Our approach operates entirely at inference time, requires no model modifications or fine-tuning, and is compatible with a wide range of pretrained MLLMs. T3S turns video redundancy into a computational advantage, offering a scalable solution for long-video understanding. The code is available at https://github.com/kaibinwang3/T3S.
>
---
#### [new 299] When Semantics Regulate: Rethinking Patch Shuffle and Internal Bias for Generated Image Detection with CLIP
- **分类: cs.CV**

- **简介: 该论文针对AI生成图像检测任务，解决现有CLIP模型依赖语义线索导致泛化能力差的问题。通过分析发现，随机打乱图像块（Patch Shuffle）能抑制语义偏差，保留局部生成痕迹。据此提出SemAnti方法，仅微调对伪造特征敏感的层，有效提升跨域检测性能。**

- **链接: [https://arxiv.org/pdf/2511.19126v1](https://arxiv.org/pdf/2511.19126v1)**

> **作者:** Beilin Chu; Weike You; Mengtao Li; Tingting Zheng; Kehan Zhao; Xuan Xu; Zhigao Lu; Jia Song; Moxuan Xu; Linna Zhou
>
> **备注:** 14 pages, 7 figures and 7 tables
>
> **摘要:** The rapid progress of GANs and Diffusion Models poses new challenges for detecting AI-generated images. Although CLIP-based detectors exhibit promising generalization, they often rely on semantic cues rather than generator artifacts, leading to brittle performance under distribution shifts. In this work, we revisit the nature of semantic bias and uncover that Patch Shuffle provides an unusually strong benefit for CLIP, that disrupts global semantic continuity while preserving local artifact cues, which reduces semantic entropy and homogenizes feature distributions between natural and synthetic images. Through a detailed layer-wise analysis, we further show that CLIP's deep semantic structure functions as a regulator that stabilizes cross-domain representations once semantic bias is suppressed. Guided by these findings, we propose SemAnti, a semantic-antagonistic fine-tuning paradigm that freezes the semantic subspace and adapts only artifact-sensitive layers under shuffled semantics. Despite its simplicity, SemAnti achieves state-of-the-art cross-domain generalization on AIGCDetectBenchmark and GenImage, demonstrating that regulating semantics is key to unlocking CLIP's full potential for robust AI-generated image detection.
>
---
#### [new 300] MedSAM3: Delving into Segment Anything with Medical Concepts
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MedSAM-3，解决医学图像分割中泛化能力差、依赖大量人工标注的问题。通过在医学图像上微调SAM3架构，实现基于文本提示的开放词汇分割，并引入多模态大模型驱动的智能体框架，支持复杂推理与迭代优化，在多种医学影像模态上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.19046v1](https://arxiv.org/pdf/2511.19046v1)**

> **作者:** Anglin Liu; Rundong Xue; Xu R. Cao; Yifan Shen; Yi Lu; Xiang Li; Qianqian Chen; Jintai Chen
>
> **摘要:** Medical image segmentation is fundamental for biomedical discovery. Existing methods lack generalizability and demand extensive, time-consuming manual annotation for new clinical application. Here, we propose MedSAM-3, a text promptable medical segmentation model for medical image and video segmentation. By fine-tuning the Segment Anything Model (SAM) 3 architecture on medical images paired with semantic conceptual labels, our MedSAM-3 enables medical Promptable Concept Segmentation (PCS), allowing precise targeting of anatomical structures via open-vocabulary text descriptions rather than solely geometric prompts. We further introduce the MedSAM-3 Agent, a framework that integrates Multimodal Large Language Models (MLLMs) to perform complex reasoning and iterative refinement in an agent-in-the-loop workflow. Comprehensive experiments across diverse medical imaging modalities, including X-ray, MRI, Ultrasound, CT, and video, demonstrate that our approach significantly outperforms existing specialist and foundation models. We will release our code and model at https://github.com/Joey-S-Liu/MedSAM3.
>
---
#### [new 301] Target-Bench: Can World Models Achieve Mapless Path Planning with Semantic Targets?
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对世界模型在无地图路径规划中的能力不足问题，提出Target-Bench基准，评估模型在真实环境中的语义目标导航性能。通过450段视频与真实轨迹数据，量化评估生成视频的路径规划能力，验证了现有模型表现有限，并展示微调可显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.17792v1](https://arxiv.org/pdf/2511.17792v1)**

> **作者:** Dingrui Wang; Hongyuan Ye; Zhihao Liang; Zhexiao Sun; Zhaowei Lu; Yuchen Zhang; Yuyu Zhao; Yuan Gao; Marvin Seegert; Finn Schäfer; Haotong Qin; Wei Li; Luigi Palmieri; Felix Jahncke; Mattia Piccinini; Johannes Betz
>
> **备注:** 10 pages
>
> **摘要:** While recent world models generate highly realistic videos, their ability to perform robot path planning remains unclear and unquantified. We introduce Target-Bench, the first benchmark specifically designed to evaluate world models on mapless path planning toward semantic targets in real-world environments. Target-Bench provides 450 robot-collected video sequences spanning 45 semantic categories with SLAM-based ground truth trajectories. Our evaluation pipeline recovers camera motion from generated videos and measures planning performance using five complementary metrics that quantify target-reaching capability, trajectory accuracy, and directional consistency. We evaluate state-of-the-art models including Sora 2, Veo 3.1, and the Wan series. The best off-the-shelf model (Wan2.2-Flash) achieves only 0.299 overall score, revealing significant limitations in current world models for robotic planning tasks. We show that fine-tuning an open-source 5B-parameter model on only 325 scenarios from our dataset achieves 0.345 overall score -- an improvement of more than 400% over its base version (0.066) and 15% higher than the best off-the-shelf model. We will open-source the code and dataset.
>
---
#### [new 302] MINDiff: Mask-Integrated Negative Attention for Controlling Overfitting in Text-to-Image Personalization
- **分类: cs.CV**

- **简介: 该论文针对文本到图像个性化中的过拟合问题，提出MINDiff方法。通过引入推理时的负注意力机制，抑制无关区域中的主体影响，提升文本对齐度并控制主体主导性。无需重训练，可直接应用于现有模型，有效缓解过拟合。**

- **链接: [https://arxiv.org/pdf/2511.17888v1](https://arxiv.org/pdf/2511.17888v1)**

> **作者:** Seulgi Jeong; Jaeil Kim
>
> **备注:** Accepted at ICCV 2025 Personalization in Generative AI Workshop
>
> **摘要:** In the personalization process of large-scale text-to-image models, overfitting often occurs when learning specific subject from a limited number of images. Existing methods, such as DreamBooth, mitigate this issue through a class-specific prior-preservation loss, which requires increased computational cost during training and limits user control during inference time. To address these limitations, we propose Mask-Integrated Negative Attention Diffusion (MINDiff). MINDiff introduces a novel concept, negative attention, which suppresses the subject's influence in masked irrelevant regions. We achieve this by modifying the cross-attention mechanism during inference. This enables semantic control and improves text alignment by reducing subject dominance in irrelevant regions. Additionally, during the inference time, users can adjust a scale parameter lambda to balance subject fidelity and text alignment. Our qualitative and quantitative experiments on DreamBooth models demonstrate that MINDiff mitigates overfitting more effectively than class-specific prior-preservation loss. As our method operates entirely at inference time and does not alter the model architecture, it can be directly applied to existing DreamBooth models without re-training. Our code is available at https://github.com/seuleepy/MINDiff.
>
---
#### [new 303] ViMix-14M: A Curated Multi-Source Video-Text Dataset with Long-Form, High-Quality Captions and Crawl-Free Access
- **分类: cs.CV**

- **简介: 该论文针对文本到视频生成中缺乏高质量、易获取的视频-文本数据集的问题，提出ViMix-14M，一个包含约1400万对视频-文本数据的多源、去重、高质量数据集。通过整合开放源、统一过滤与精炼重标注，实现无需爬虫的下载即用，显著提升数据可用性与质量，助力开源视频基础模型训练。**

- **链接: [https://arxiv.org/pdf/2511.18382v1](https://arxiv.org/pdf/2511.18382v1)**

> **作者:** Timing Yang; Sucheng Ren; Alan Yuille; Feng Wang
>
> **摘要:** Text-to-video generation has surged in interest since Sora, yet open-source models still face a data bottleneck: there is no large, high-quality, easily obtainable video-text corpus. Existing public datasets typically require manual YouTube crawling, which yields low usable volume due to link rot and access limits, and raises licensing uncertainty. This work addresses this challenge by introducing ViMix-14M, a curated multi-source video-text dataset of around 14 million pairs that provides crawl-free, download-ready access and long-form, high-quality captions tightly aligned to video. ViMix-14M is built by merging diverse open video sources, followed by unified de-duplication and quality filtering, and a multi-granularity, ground-truth-guided re-captioning pipeline that refines descriptions to better match actions, scenes, and temporal structure. We evaluate the dataset by multimodal retrieval, text-to-video generation, and video question answering tasks, observing consistent improvements over counterpart datasets. We hope this work can help removing the key barrier to training and fine-tuning open-source video foundation models, and provide insights of building high-quality and generalizable video-text datasets.
>
---
#### [new 304] Matching-Based Few-Shot Semantic Segmentation Models Are Interpretable by Design
- **分类: cs.CV**

- **简介: 该论文针对少样本语义分割（FSS）模型缺乏可解释性的问题，提出基于匹配机制的Affinity Explainer方法。通过多层级特征匹配得分生成像素级归因图，揭示支持图像中对预测贡献最大的区域。在多个基准数据集上验证了其优越性，首次为FSS模型提供了可解释性框架。**

- **链接: [https://arxiv.org/pdf/2511.18163v1](https://arxiv.org/pdf/2511.18163v1)**

> **作者:** Pasquale De Marinis; Uzay Kaymak; Rogier Brussee; Gennaro Vessio; Giovanna Castellano
>
> **摘要:** Few-Shot Semantic Segmentation (FSS) models achieve strong performance in segmenting novel classes with minimal labeled examples, yet their decision-making processes remain largely opaque. While explainable AI has advanced significantly in standard computer vision tasks, interpretability in FSS remains virtually unexplored despite its critical importance for understanding model behavior and guiding support set selection in data-scarce scenarios. This paper introduces the first dedicated method for interpreting matching-based FSS models by leveraging their inherent structural properties. Our Affinity Explainer approach extracts attribution maps that highlight which pixels in support images contribute most to query segmentation predictions, using matching scores computed between support and query features at multiple feature levels. We extend standard interpretability evaluation metrics to the FSS domain and propose additional metrics to better capture the practical utility of explanations in few-shot scenarios. Comprehensive experiments on FSS benchmark datasets, using different models, demonstrate that our Affinity Explainer significantly outperforms adapted standard attribution methods. Qualitative analysis reveals that our explanations provide structured, coherent attention patterns that align with model architectures and and enable effective model diagnosis. This work establishes the foundation for interpretable FSS research, enabling better model understanding and diagnostic for more reliable few-shot segmentation systems. The source code is publicly available at https://github.com/pasqualedem/AffinityExplainer.
>
---
#### [new 305] MonoSR: Open-Vocabulary Spatial Reasoning from Monocular Images
- **分类: cs.CV**

- **简介: 该论文提出MonoSR，一个面向单目图像的开放词汇空间推理数据集，解决现有方法多依赖多视角、局限于室内场景的问题。工作涵盖数据构建、视觉语言模型评估及辅助信息作用分析，推动真实世界中单目空间推理的发展。**

- **链接: [https://arxiv.org/pdf/2511.19119v1](https://arxiv.org/pdf/2511.19119v1)**

> **作者:** Qirui Wang; Jingyi He; Yining Pan; Si Yong Yeo; Xulei Yang; Shijie Li
>
> **摘要:** Spatial reasoning (SR), the ability to infer 3D spatial information from 2D inputs, is essential for real-world applications such as embodied AI and autonomous driving. However, existing research primarily focuses on indoor environments and typically relies on multi-view observations, which limits their generalizability to outdoor scenarios and constrains their applicability to monocular images, the most common real-world setting. In this work, we propose MonoSR, a large-scale monocular spatial reasoning dataset that spans diverse scenarios including indoor, outdoor, and object-centric settings, and supports multiple question types. MonoSR provides a path toward open-world monocular spatial reasoning. Beyond introducing the dataset, we evaluate advanced vision-language models to reveal their limitations on this challenging task. We further analyze whether auxiliary information is crucial for monocular spatial reasoning and offer practical guidance for designing future models. These contributions collectively establish a foundation for advancing monocular spatial reasoning in real-world, open-world environments.
>
---
#### [new 306] RAISECity: A Multimodal Agent Framework for Reality-Aligned 3D World Generation at City-Scale
- **分类: cs.CV**

- **简介: 该论文提出RAISECity，一个用于城市级3D世界生成的多模态智能体框架。针对现有方法在质量、真实感与可扩展性上的不足，通过动态数据处理、自反思迭代与多模态工具调用，实现高保真、现实对齐的3D场景生成，显著提升感知质量与系统性能。**

- **链接: [https://arxiv.org/pdf/2511.18005v1](https://arxiv.org/pdf/2511.18005v1)**

> **作者:** Shengyuan Wang; Zhiheng Zheng; Yu Shang; Lixuan He; Yangcheng Yu; Fan Hangyu; Jie Feng; Qingmin Liao; Yong Li
>
> **备注:** The code will be made publicly available soon at: https://github.com/tsinghua-fib-lab/RAISECity
>
> **摘要:** City-scale 3D generation is of great importance for the development of embodied intelligence and world models. Existing methods, however, face significant challenges regarding quality, fidelity, and scalability in 3D world generation. Thus, we propose RAISECity, a \textbf{R}eality-\textbf{A}ligned \textbf{I}ntelligent \textbf{S}ynthesis \textbf{E}ngine that creates detailed, \textbf{C}ity-scale 3D worlds. We introduce an agentic framework that leverages diverse multimodal foundation tools to acquire real-world knowledge, maintain robust intermediate representations, and construct complex 3D scenes. This agentic design, featuring dynamic data processing, iterative self-reflection and refinement, and the invocation of advanced multimodal tools, minimizes cumulative errors and enhances overall performance. Extensive quantitative experiments and qualitative analyses validate the superior performance of RAISECity in real-world alignment, shape precision, texture fidelity, and aesthetics level, achieving over a 90% win-rate against existing baselines for overall perceptual quality. This combination of 3D quality, reality alignment, scalability, and seamless compatibility with computer graphics pipelines makes RAISECity a promising foundation for applications in immersive media, embodied intelligence, and world models.
>
---
#### [new 307] Plug-and-Play Multi-Concept Adaptive Blending for High-Fidelity Text-to-Image Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像生成中多概念个性化融合时出现的语义不一致与特征泄漏问题，提出PnP-MIX方法。通过引导式外观注意力、掩码引导噪声混合及背景稀释++策略，实现无需微调的高保真多概念融合，显著提升复杂场景下的生成质量与区域一致性。**

- **链接: [https://arxiv.org/pdf/2511.17615v1](https://arxiv.org/pdf/2511.17615v1)**

> **作者:** Young-Beom Woo
>
> **备注:** [Master's thesis, Korea University, 2025]
>
> **摘要:** Integrating multiple personalized concepts into a single image has recently become a significant area of focus within Text-to-Image (T2I) generation. However, existing methods often underperform on complex multi-object scenes due to unintended alterations in both personalized and non-personalized regions. This not only fails to preserve the intended prompt structure but also disrupts interactions among regions, leading to semantic inconsistencies. To address this limitation, we introduce plug-and-play multi-concept adaptive blending for high-fidelity text-to-image synthesis (PnP-MIX), an innovative, tuning-free approach designed to seamlessly embed multiple personalized concepts into a single generated image. Our method leverages guided appearance attention to faithfully reflect the intended appearance of each personalized concept. To further enhance compositional fidelity, we present a mask-guided noise mixing strategy that preserves the integrity of non-personalized regions such as the background or unrelated objects while enabling the precise integration of personalized objects. Finally, to mitigate concept leakage, i.e., the inadvertent leakage of personalized concept features into other regions, we propose background dilution++, a novel strategy that effectively reduces such leakage and promotes accurate localization of features within personalized regions. Extensive experimental results demonstrate that PnP-MIX consistently surpasses existing methodologies in both single- and multi-concept personalization scenarios, underscoring its robustness and superior performance without additional model tuning.
>
---
#### [new 308] Hierarchical Semi-Supervised Active Learning for Remote Sensing
- **分类: cs.CV**

- **简介: 该论文针对遥感图像分类中标签数据稀缺问题，提出一种分层半监督主动学习框架（HSSAL）。通过结合半监督学习与分层主动学习，在迭代中利用未标注数据提升模型性能，并高效选择最具信息量的样本进行标注，显著降低标注成本，实现高精度分类。**

- **链接: [https://arxiv.org/pdf/2511.18058v1](https://arxiv.org/pdf/2511.18058v1)**

> **作者:** Wei Huang; Zhitong Xiong; Chenying Liu; Xiao Xiang Zhu
>
> **备注:** Under review
>
> **摘要:** The performance of deep learning models in remote sensing (RS) strongly depends on the availability of high-quality labeled data. However, collecting large-scale annotations is costly and time-consuming, while vast amounts of unlabeled imagery remain underutilized. To address this challenge, we propose a Hierarchical Semi-Supervised Active Learning (HSSAL) framework that integrates semi-supervised learning (SSL) and a novel hierarchical active learning (HAL) in a closed iterative loop. In each iteration, SSL refines the model using both labeled data through supervised learning and unlabeled data via weak-to-strong self-training, improving feature representation and uncertainty estimation. Guided by the refined representations and uncertainty cues of unlabeled samples, HAL then conducts sample querying through a progressive clustering strategy, selecting the most informative instances that jointly satisfy the criteria of scalability, diversity, and uncertainty. This hierarchical process ensures both efficiency and representativeness in sample selection. Extensive experiments on three benchmark RS scene classification datasets, including UCM, AID, and NWPU-RESISC45, demonstrate that HSSAL consistently outperforms SSL- or AL-only baselines. Remarkably, with only 8%, 4%, and 2% labeled training data on UCM, AID, and NWPU-RESISC45, respectively, HSSAL achieves over 95% of fully-supervised accuracy, highlighting its superior label efficiency through informativeness exploitation of unlabeled data. Our code will be released at https://github.com/zhu-xlab/RS-SSAL.
>
---
#### [new 309] EgoVITA: Learning to Plan and Verify for Egocentric Video Reasoning
- **分类: cs.CV**

- **简介: 该论文针对第一人称视频推理任务，解决视角变化带来的局部可观测与自参照运动难题。提出EgoVITA框架，通过交替进行第一人称规划与第三人称验证，利用强化学习提升模型对未来动作的因果预测能力，显著提升推理性能。**

- **链接: [https://arxiv.org/pdf/2511.18242v1](https://arxiv.org/pdf/2511.18242v1)**

> **作者:** Yogesh Kulkarni; Pooyan Fazli
>
> **摘要:** Reasoning about intentions and actions from a first-person (egocentric) perspective remains a fundamental challenge for multimodal large language models (MLLMs). Unlike third-person (exocentric) videos that capture scenes from an outside observer, egocentric videos reflect the actor's continuously changing viewpoint, introducing partial observability, limited field of view, and self-referenced motion. We introduce $\textbf{EgoVITA}$, a reinforcement learning framework that enables MLLMs to reason through structured planning and verification. Built on Group Relative Policy Optimization (GRPO), EgoVITA alternates between two stages: (1) an $\textbf{egocentric planning phase}$, where the model reasons from a first-person viewpoint to predict a step-by-step plan of future actions, and (2) an $\textbf{exocentric verification phase}$, where it switches to a third-person perspective to check the visual and logical consistency of that plan. Through GRPO, the model learns to make plans that are causally predictive of upcoming visual observations, leading to more coherent and visually grounded reasoning. EgoVITA achieves significant gains on egocentric reasoning tasks, outperforming the baseline Qwen2.5-VL-7B by $\mathbf{+7.7}$ on EgoBlind and $\mathbf{+4.4}$ on EgoOrient, while maintaining strong generalization on exocentric video tasks.
>
---
#### [new 310] NeAR: Coupled Neural Asset-Renderer Stack
- **分类: cs.CV**

- **简介: 该论文提出NeAR，一个耦合的神经资产-渲染器架构。针对传统图形管线中资产与渲染分离导致的效率与一致性问题，通过联合设计实现端到端可学习。工作包括：构建光照一致的神经资产表示，设计光照感知实时渲染器，并在多种任务上验证其优越性。**

- **链接: [https://arxiv.org/pdf/2511.18600v1](https://arxiv.org/pdf/2511.18600v1)**

> **作者:** Hong Li; Chongjie Ye; Houyuan Chen; Weiqing Xiao; Ziyang Yan; Lixing Xiao; Zhaoxi Chen; Jianfeng Xiang; Shaocong Xu; Xuhui Liu; Yikai Wang; Baochang Zhang; Xiaoguang Han; Jiaolong Yang; Hao Zhao
>
> **备注:** 20 pages, 16 figures
>
> **摘要:** Neural asset authoring and neural rendering have emerged as fundamentally disjoint threads: one generates digital assets using neural networks for traditional graphics pipelines, while the other develops neural renderers that map conventional assets to images. However, the potential of jointly designing the asset representation and renderer remains largely unexplored. We argue that coupling them can unlock an end-to-end learnable graphics stack with benefits in fidelity, consistency, and efficiency. In this paper, we explore this possibility with NeAR: a Coupled Neural Asset-Renderer Stack. On the asset side, we build on Trellis-style Structured 3D Latents and introduce a lighting-homogenized neural asset: from a casually lit input, a rectified-flow backbone predicts a Lighting-Homogenized SLAT that encodes geometry and intrinsic material cues in a compact, view-agnostic latent. On the renderer side, we design a lighting-aware neural renderer that uses this neural asset, along with explicit view embeddings and HDR environment maps, to achieve real-time, relightable rendering. We validate NeAR on four tasks: (1) G-buffer-based forward rendering, (2) random-lit single-image reconstruction, (3) unknown-lit single-image relighting, and (4) novel-view relighting. Our coupled stack surpasses state-of-the-art baselines in both quantitative metrics and perceptual quality. We hope this coupled asset-renderer perspective inspires future graphics stacks that view neural assets and renderers as co-designed components instead of independent entities.
>
---
#### [new 311] RigAnyFace: Scaling Neural Facial Mesh Auto-Rigging with Unlabeled Data
- **分类: cs.CV**

- **简介: 该论文提出RigAnyFace（RAF），一个用于多样化面部网格的可扩展神经自动绑定框架，解决复杂拓扑（如分离组件）下高精度、泛化性强的面部动画绑定问题。通过结合少量人工标注数据与大量无标签数据的2D监督策略，提升模型泛化能力，支持眼球等分离部件的精细表情动画。**

- **链接: [https://arxiv.org/pdf/2511.18601v1](https://arxiv.org/pdf/2511.18601v1)**

> **作者:** Wenchao Ma; Dario Kneubuehler; Maurice Chu; Ian Sachs; Haomiao Jiang; Sharon Xiaolei Huang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** In this paper, we present RigAnyFace (RAF), a scalable neural auto-rigging framework for facial meshes of diverse topologies, including those with multiple disconnected components. RAF deforms a static neutral facial mesh into industry-standard FACS poses to form an expressive blendshape rig. Deformations are predicted by a triangulation-agnostic surface learning network augmented with our tailored architecture design to condition on FACS parameters and efficiently process disconnected components. For training, we curated a dataset of facial meshes, with a subset meticulously rigged by professional artists to serve as accurate 3D ground truth for deformation supervision. Due to the high cost of manual rigging, this subset is limited in size, constraining the generalization ability of models trained exclusively on it. To address this, we design a 2D supervision strategy for unlabeled neutral meshes without rigs. This strategy increases data diversity and allows for scaled training, thereby enhancing the generalization ability of models trained on this augmented data. Extensive experiments demonstrate that RAF is able to rig meshes of diverse topologies on not only our artist-crafted assets but also in-the-wild samples, outperforming previous works in accuracy and generalizability. Moreover, our method advances beyond prior work by supporting multiple disconnected components, such as eyeballs, for more detailed expression animation. Project page: https://wenchao-m.github.io/RigAnyFace.github.io
>
---
#### [new 312] Beyond Words and Pixels: A Benchmark for Implicit World Knowledge Reasoning in Generative Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像生成模型在隐含世界知识与物理因果推理上的不足，提出PicWorld基准与PW-Agent多智能体评估体系。通过1100个跨类别提示，系统评估模型对隐性知识的理解能力，揭示现有模型普遍缺乏物理合理性与逻辑一致性，推动构建更具推理能力的生成架构。**

- **链接: [https://arxiv.org/pdf/2511.18271v1](https://arxiv.org/pdf/2511.18271v1)**

> **作者:** Tianyang Han; Junhao Su; Junjie Hu; Peizhen Yang; Hengyu Shi; Junfeng Luo; Jialin Gao
>
> **摘要:** Text-to-image (T2I) models today are capable of producing photorealistic, instruction-following images, yet they still frequently fail on prompts that require implicit world knowledge. Existing evaluation protocols either emphasize compositional alignment or rely on single-round VQA-based scoring, leaving critical dimensions such as knowledge grounding, multi-physics interactions, and auditable evidence-substantially undertested. To address these limitations, we introduce PicWorld, the first comprehensive benchmark that assesses the grasp of implicit world knowledge and physical causal reasoning of T2I models. This benchmark consists of 1,100 prompts across three core categories. To facilitate fine-grained evaluation, we propose PW-Agent, an evidence-grounded multi-agent evaluator to hierarchically assess images on their physical realism and logical consistency by decomposing prompts into verifiable visual evidence. We conduct a thorough analysis of 17 mainstream T2I models on PicWorld, illustrating that they universally exhibit a fundamental limitation in their capacity for implicit world knowledge and physical causal reasoning to varying degrees. The findings highlight the need for reasoning-aware, knowledge-integrative architectures in future T2I systems.
>
---
#### [new 313] Health system learning achieves generalist neuroimaging models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出“健康系统学习”范式，解决临床神经影像数据因隐私限制难以用于训练前沿AI模型的问题。通过在524万份临床MRI/CT数据上训练NeuroVFM模型，实现对脑部解剖与病理的全面表征，显著提升诊断、报告生成等任务性能，并通过视觉-语言对齐减少错误，推动通用医疗AI发展。**

- **链接: [https://arxiv.org/pdf/2511.18640v1](https://arxiv.org/pdf/2511.18640v1)**

> **作者:** Akhil Kondepudi; Akshay Rao; Chenhui Zhao; Yiwei Lyu; Samir Harake; Soumyanil Banerjee; Rushikesh Joshi; Anna-Katharina Meissner; Renly Hou; Cheng Jiang; Asadur Chowdury; Ashok Srinivasan; Brian Athey; Vikas Gulani; Aditya Pandey; Honglak Lee; Todd Hollon
>
> **备注:** 53 pages, 4 main figures, 10 extended data figures
>
> **摘要:** Frontier artificial intelligence (AI) models, such as OpenAI's GPT-5 and Meta's DINOv3, have advanced rapidly through training on internet-scale public data, yet such systems lack access to private clinical data. Neuroimaging, in particular, is underrepresented in the public domain due to identifiable facial features within MRI and CT scans, fundamentally restricting model performance in clinical medicine. Here, we show that frontier models underperform on neuroimaging tasks and that learning directly from uncurated data generated during routine clinical care at health systems, a paradigm we call health system learning, yields high-performance, generalist neuroimaging models. We introduce NeuroVFM, a visual foundation model trained on 5.24 million clinical MRI and CT volumes using a scalable volumetric joint-embedding predictive architecture. NeuroVFM learns comprehensive representations of brain anatomy and pathology, achieving state-of-the-art performance across multiple clinical tasks, including radiologic diagnosis and report generation. The model exhibits emergent neuroanatomic understanding and interpretable visual grounding of diagnostic findings. When paired with open-source language models through lightweight visual instruction tuning, NeuroVFM generates radiology reports that surpass frontier models in accuracy, clinical triage, and expert preference. Through clinically grounded visual understanding, NeuroVFM reduces hallucinated findings and critical errors, offering safer clinical decision support. These results establish health system learning as a paradigm for building generalist medical AI and provide a scalable framework for clinical foundation models.
>
---
#### [new 314] MVS-TTA: Test-Time Adaptation for Multi-View Stereo via Meta-Auxiliary Learning
- **分类: cs.CV**

- **简介: 该论文针对学习型多视图立体（MVS）方法泛化能力差的问题，提出MVS-TTA框架。通过自监督跨视图一致性损失与元辅助学习，在测试时实现高效适应，提升模型对新场景的泛化性能。该方法无需修改主模型架构，适用于多种MVS方法，是首个将优化式测试时自适应引入学习型MVS的工作。**

- **链接: [https://arxiv.org/pdf/2511.18120v1](https://arxiv.org/pdf/2511.18120v1)**

> **作者:** Hannuo Zhang; Zhixiang Chi; Yang Wang; Xinxin Zuo
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Recent learning-based multi-view stereo (MVS) methods are data-driven and have achieved remarkable progress due to large-scale training data and advanced architectures. However, their generalization remains sub-optimal due to fixed model parameters trained on limited training data distributions. In contrast, optimization-based methods enable scene-specific adaptation but lack scalability and require costly per-scene optimization. In this paper, we propose MVS-TTA, an efficient test-time adaptation (TTA) framework that enhances the adaptability of learning-based MVS methods by bridging these two paradigms. Specifically, MVS-TTA employs a self-supervised, cross-view consistency loss as an auxiliary task to guide inference-time adaptation. We introduce a meta-auxiliary learning strategy to train the model to benefit from auxiliary-task-based updates explicitly. Our framework is model-agnostic and can be applied to a wide range of MVS methods with minimal architectural changes. Extensive experiments on standard datasets (DTU, BlendedMVS) and a challenging cross-dataset generalization setting demonstrate that MVS-TTA consistently improves performance, even when applied to state-of-the-art MVS models. To our knowledge, this is the first attempt to integrate optimization-based test-time adaptation into learning-based MVS using meta-learning. The code will be available at https://github.com/mart87987-svg/MVS-TTA.
>
---
#### [new 315] NI-Tex: Non-isometric Image-based Garment Texture Generation
- **分类: cs.CV**

- **简介: 该论文针对非保距图像驱动的服装纹理生成难题，提出NI-Tex框架。通过构建3D服装视频数据集实现跨姿态纹理学习，结合纳米香蕉进行非等距图像编辑，并采用不确定性引导的迭代烘焙方法，生成高质量、无缝的PBR纹理，提升工业级3D服装设计的纹理多样性与质量。**

- **链接: [https://arxiv.org/pdf/2511.18765v1](https://arxiv.org/pdf/2511.18765v1)**

> **作者:** Hui Shan; Ming Li; Haitao Yang; Kai Zheng; Sizhe Zheng; Yanwei Fu; Xiangru Huang
>
> **摘要:** Existing industrial 3D garment meshes already cover most real-world clothing geometries, yet their texture diversity remains limited. To acquire more realistic textures, generative methods are often used to extract Physically-based Rendering (PBR) textures and materials from large collections of wild images and project them back onto garment meshes. However, most image-conditioned texture generation approaches require strict topological consistency between the input image and the input 3D mesh, or rely on accurate mesh deformation to match to the image poses, which significantly constrains the texture generation quality and flexibility. To address the challenging problem of non-isometric image-based garment texture generation, we construct 3D Garment Videos, a physically simulated, garment-centric dataset that provides consistent geometry and material supervision across diverse deformations, enabling robust cross-pose texture learning. We further employ Nano Banana for high-quality non-isometric image editing, achieving reliable cross-topology texture generation between non-isometric image-geometry pairs. Finally, we propose an iterative baking method via uncertainty-guided view selection and reweighting that fuses multi-view predictions into seamless, production-ready PBR textures. Through extensive experiments, we demonstrate that our feedforward dual-branch architecture generates versatile and spatially aligned PBR materials suitable for industry-level 3D garment design.
>
---
#### [new 316] LungX: A Hybrid EfficientNet-Vision Transformer Architecture with Multi-Scale Attention for Accurate Pneumonia Detection
- **分类: cs.CV**

- **简介: 该论文针对肺炎精准检测任务，提出LungX混合架构，融合EfficientNet多尺度特征、CBAM注意力与Vision Transformer全局建模，提升诊断准确率。在2万张胸片数据上实现86.5%准确率、0.943 AUC，显著优于基线，且具备可解释性。**

- **链接: [https://arxiv.org/pdf/2511.18425v1](https://arxiv.org/pdf/2511.18425v1)**

> **作者:** Mansur Yerzhanuly
>
> **备注:** 13 pages, 3 figures, 1 table
>
> **摘要:** Pneumonia remains a leading global cause of mortality where timely diagnosis is critical. We introduce LungX, a novel hybrid architecture combining EfficientNet's multi-scale features, CBAM attention mechanisms, and Vision Transformer's global context modeling for enhanced pneumonia detection. Evaluated on 20,000 curated chest X-rays from RSNA and CheXpert, LungX achieves state-of-the-art performance (86.5 percent accuracy, 0.943 AUC), representing a 6.7 percent AUC improvement over EfficientNet-B0 baselines. Visual analysis demonstrates superior lesion localization through interpretable attention maps. Future directions include multi-center validation and architectural optimizations targeting 88 percent accuracy for clinical deployment as an AI diagnostic aid.
>
---
#### [new 317] InfiniBench: Infinite Benchmarking for Visual Spatial Reasoning with Customizable Scene Complexity
- **分类: cs.CV**

- **简介: 该论文针对视觉空间推理能力评估难题，提出InfiniBench——一个可定制、自动化的3D场景生成基准工具。通过自然语言生成复杂物理合理的3D场景视频，解决现有基准在场景复杂度可控性与多样性上的不足，支持对视觉语言模型的空间推理能力进行高效、精准评估。**

- **链接: [https://arxiv.org/pdf/2511.18200v1](https://arxiv.org/pdf/2511.18200v1)**

> **作者:** Haoming Wang; Qiyao Xue; Wei Gao
>
> **摘要:** Modern vision-language models (VLMs) are expected to have abilities of spatial reasoning with diverse scene complexities, but evaluating such abilities is difficult due to the lack of benchmarks that are not only diverse and scalable but also fully customizable. Existing benchmarks offer limited customizability over the scene complexity and are incapable of isolating and analyzing specific VLM failure modes under distinct spatial conditions. To address this gap, instead of individually presenting benchmarks for different scene complexities, in this paper we present InfiniBench, a fully automated, customizable and user-friendly benchmark generator that can synthesize a theoretically infinite variety of 3D scenes with parameterized control on scene complexity. InfiniBench uniquely translates scene descriptions in natural language into photo-realistic videos with complex and physically plausible 3D layouts. This is achieved through three key innovations: 1) a LLM-based agentic framework that iteratively refines procedural scene constraints from scene descriptions; 2) a flexible cluster-based layout optimizer that generates dense and cluttered scenes previously intractable for procedural methods; and 3) a task-aware camera trajectory optimization method that renders scenes into videos with full object coverage as VLM input. Experiments demonstrate that InfiniBench outperforms state-of-the-art procedural and LLM-based 3D generation methods in prompt fidelity and physical plausibility, especially in high-complexity scenarios. We further showcased the usefulness of InfiniBench, by generating benchmarks for representative spatial reasoning tasks including measurement, perspective-taking and spatiotemporal tracking.
>
---
#### [new 318] Nested Unfolding Network for Real-World Concealed Object Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对真实世界隐匿物体分割（COS）任务，解决现有深度展开网络（DUN）中图像恢复与分割目标冲突、依赖预设退化类型的问题。提出嵌套展开网络（NUN），通过“恢复-分割”双阶段设计，解耦任务并实现相互优化，结合视觉语言模型动态感知退化，利用多阶段质量评估引入自一致性损失，提升鲁棒性与性能。**

- **链接: [https://arxiv.org/pdf/2511.18164v1](https://arxiv.org/pdf/2511.18164v1)**

> **作者:** Chunming He; Rihan Zhang; Dingming Zhang; Fengyang Xiao; Deng-Ping Fan; Sina Farsiu
>
> **备注:** 6 figures, 14 tables
>
> **摘要:** Deep unfolding networks (DUNs) have recently advanced concealed object segmentation (COS) by modeling segmentation as iterative foreground-background separation. However, existing DUN-based methods (RUN) inherently couple background estimation with image restoration, leading to conflicting objectives and requiring pre-defined degradation types, which are unrealistic in real-world scenarios. To address this, we propose the nested unfolding network (NUN), a unified framework for real-world COS. NUN adopts a DUN-in-DUN design, embedding a degradation-resistant unfolding network (DeRUN) within each stage of a segmentation-oriented unfolding network (SODUN). This design decouples restoration from segmentation while allowing mutual refinement. Guided by a vision-language model (VLM), DeRUN dynamically infers degradation semantics and restores high-quality images without explicit priors, whereas SODUN performs reversible estimation to refine foreground and background. Leveraging the multi-stage nature of unfolding, NUN employs image-quality assessment to select the best DeRUN outputs for subsequent stages, naturally introducing a self-consistency loss that enhances robustness. Extensive experiments show that NUN achieves a leading place on both clean and degraded benchmarks. Code will be released.
>
---
#### [new 319] Adversarial Pseudo-replay for Exemplar-free Class-incremental Learning
- **分类: cs.CV**

- **简介: 该论文针对示例无关的类增量学习（EFCIL）任务，解决因无法存储旧数据导致的灾难性遗忘问题。提出对抗伪重放（APR）方法，通过对抗攻击生成新任务图像的伪重放样本，结合知识蒸馏与协方差校准，有效缓解语义漂移，实现稳定与可塑性的平衡，在冷启动设置下达到先进性能。**

- **链接: [https://arxiv.org/pdf/2511.17973v1](https://arxiv.org/pdf/2511.17973v1)**

> **作者:** Hiroto Honda
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Exemplar-free class-incremental learning (EFCIL) aims to retain old knowledge acquired in the previous task while learning new classes, without storing the previous images due to storage constraints or privacy concerns. In EFCIL, the plasticity-stability dilemma, learning new tasks versus catastrophic forgetting, is a significant challenge, primarily due to the unavailability of images from earlier tasks. In this paper, we introduce adversarial pseudo-replay (APR), a method that perturbs the images of the new task with adversarial attack, to synthesize the pseudo-replay images online without storing any replay samples. During the new task training, the adversarial attack is conducted on the new task images with augmented old class mean prototypes as targets, and the resulting images are used for knowledge distillation to prevent semantic drift. Moreover, we calibrate the covariance matrices to compensate for the semantic drift after each task, by learning a transfer matrix on the pseudo-replay samples. Our method reconciles stability and plasticity, achieving state-of-the-art on challenging cold-start settings of the standard EFCIL benchmarks.
>
---
#### [new 320] Exploring Surround-View Fisheye Camera 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文研究环绕式鱼眼相机的3D目标检测任务。针对传统针孔相机检测器在鱼眼图像上性能下降的问题，提出FisheyeBEVDet和FisheyePETR两种方法，引入球面空间表征以建模鱼眼几何特性。构建了Fisheye3DOD数据集，实验表明方法提升精度达6.2%。**

- **链接: [https://arxiv.org/pdf/2511.18695v1](https://arxiv.org/pdf/2511.18695v1)**

> **作者:** Changcai Li; Wenwei Lin; Zuoxun Hou; Gang Chen; Wei Zhang; Huihui Zhou; Weishi Zheng
>
> **备注:** 9 pages,6 figures, accepted at AAAI 2026
>
> **摘要:** In this work, we explore the technical feasibility of implementing end-to-end 3D object detection (3DOD) with surround-view fisheye camera system. Specifically, we first investigate the performance drop incurred when transferring classic pinhole-based 3D object detectors to fisheye imagery. To mitigate this, we then develop two methods that incorporate the unique geometry of fisheye images into mainstream detection frameworks: one based on the bird's-eye-view (BEV) paradigm, named FisheyeBEVDet, and the other on the query-based paradigm, named FisheyePETR. Both methods adopt spherical spatial representations to effectively capture fisheye geometry. In light of the lack of dedicated evaluation benchmarks, we release Fisheye3DOD, a new open dataset synthesized using CARLA and featuring both standard pinhole and fisheye camera arrays. Experiments on Fisheye3DOD show that our fisheye-compatible modeling improves accuracy by up to 6.2% over baseline methods.
>
---
#### [new 321] Unified Low-Light Traffic Image Enhancement via Multi-Stage Illumination Recovery and Adaptive Noise Suppression
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对低光照交通图像增强任务，解决夜间场景中亮度不足、噪声、过曝等问题。提出无监督多阶段框架，通过分解图像并分步优化照明、反射率与过曝区域，结合自监督损失实现端到端训练，显著提升图像质量与下游感知可靠性。**

- **链接: [https://arxiv.org/pdf/2511.17612v1](https://arxiv.org/pdf/2511.17612v1)**

> **作者:** Siddiqua Namrah
>
> **备注:** Master's thesis, Korea University, 2025
>
> **摘要:** Enhancing low-light traffic images is crucial for reliable perception in autonomous driving, intelligent transportation, and urban surveillance systems. Nighttime and dimly lit traffic scenes often suffer from poor visibility due to low illumination, noise, motion blur, non-uniform lighting, and glare from vehicle headlights or street lamps, which hinder tasks such as object detection and scene understanding. To address these challenges, we propose a fully unsupervised multi-stage deep learning framework for low-light traffic image enhancement. The model decomposes images into illumination and reflectance components, progressively refined by three specialized modules: (1) Illumination Adaptation, for global and local brightness correction; (2) Reflectance Restoration, for noise suppression and structural detail recovery using spatial-channel attention; and (3) Over-Exposure Compensation, for reconstructing saturated regions and balancing scene luminance. The network is trained using self-supervised reconstruction, reflectance smoothness, perceptual consistency, and domain-aware regularization losses, eliminating the need for paired ground-truth images. Experiments on general and traffic-specific datasets demonstrate superior performance over state-of-the-art methods in both quantitative metrics (PSNR, SSIM, LPIPS, NIQE) and qualitative visual quality. Our approach enhances visibility, preserves structure, and improves downstream perception reliability in real-world low-light traffic scenarios.
>
---
#### [new 322] Observer Actor: Active Vision Imitation Learning with Sparse View Gaussian Splatting
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出Observer Actor（ObAct）框架，用于主动视觉模仿学习。针对机器人操作中视角遮挡问题，通过动态分配观察者与执行者角色，利用3D高斯点云实现虚拟视角探索，优化观测视角，提升策略鲁棒性。实验表明，相比固定摄像头，性能显著提升。**

- **链接: [https://arxiv.org/pdf/2511.18140v1](https://arxiv.org/pdf/2511.18140v1)**

> **作者:** Yilong Wang; Cheng Qian; Ruomeng Fan; Edward Johns
>
> **备注:** Videos are available on our project webpage at https://obact.github.io
>
> **摘要:** We propose Observer Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wrist-mounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observer's observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policy's observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods -- trajectory transfer and behavior cloning -- and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at https://obact.github.io.
>
---
#### [new 323] Fluid Grey 2: How Well Does Generative Adversarial Network Learn Deeper Topology Structure in Architecture That Matches Images?
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究图像生成对抗网络（GAN）在建筑图像中自动学习空间拓扑结构的能力。针对传统方法因多步处理导致信息损失的问题，提出基于Grasshopper的检测模块，验证pix2pix模型可自主识别并应用拓扑关系。通过量化分析与可视化，揭示不同输入模式对学习效率的影响，为建筑与城市更新中的拓扑保持提供新方法。**

- **链接: [https://arxiv.org/pdf/2511.17643v1](https://arxiv.org/pdf/2511.17643v1)**

> **作者:** Yayan Qiu; Sean Hanna
>
> **摘要:** Taking into account the regional characteristics of intrinsic and extrinsic properties of space is an essential issue in architectural design and urban renewal, which is often achieved step by step using image and graph-based GANs. However, each model nesting and data conversion may cause information loss, and it is necessary to streamline the tools to facilitate architects and users to participate in the design. Therefore, this study hopes to prove that I2I GAN also has the potential to recognize topological relationships autonomously. Therefore, this research proposes a method for quickly detecting the ability of pix2pix to learn topological relationships, which is achieved by adding two Grasshopper-based detection modules before and after GAN. At the same time, quantitative data is provided and its learning process is visualized, and changes in different input modes such as greyscale and RGB affect its learning efficiency. There are two innovations in this paper: 1) It proves that pix2pix can automatically learn spatial topological relationships and apply them to architectural design. 2) It fills the gap in detecting the performance of Image-based Generation GAN from a topological perspective. Moreover, the detection method proposed in this study takes a short time and is simple to operate. The two detection modules can be widely used for customizing image datasets with the same topological structure and for batch detection of topological relationships of images. In the future, this paper may provide a theoretical foundation and data support for the application of architectural design and urban renewal that use GAN to preserve spatial topological characteristics.
>
---
#### [new 324] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对多模态情感分析中的模态竞争问题，提出PaSE框架。通过原型对齐校准与基于Shapley值的梯度调制，增强模态协作，缓解主导模态压制弱模态的问题，提升融合效果。**

- **链接: [https://arxiv.org/pdf/2511.17585v1](https://arxiv.org/pdf/2511.17585v1)**

> **作者:** Kang He; Boyu Chen; Yuzhe Ding; Fei Li; Chong Teng; Donghong Ji
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multimodal Sentiment Analysis (MSA) seeks to understand human emotions by integrating textual, acoustic, and visual signals. Although multimodal fusion is designed to leverage cross-modal complementarity, real-world scenarios often exhibit modality competition: dominant modalities tend to overshadow weaker ones, leading to suboptimal performance.In this paper, we propose PaSE, a novel Prototype-aligned Calibration and Shapley-optimized Equilibrium framework, which enhances collaboration while explicitly mitigating modality competition. PaSE first applies Prototype-guided Calibration Learning (PCL) to refine unimodal representations and align them through an Entropic Optimal Transport mechanism that ensures semantic consistency. To further stabilize optimization, we introduce a Dual-Phase Optimization strategy. A prototype-gated fusion module is first used to extract shared representations, followed by Shapley-based Gradient Modulation (SGM), which adaptively adjusts gradients according to the contribution of each modality. Extensive experiments on IEMOCAP, MOSI, and MOSEI confirm that PaSE achieves the superior performance and effectively alleviates modality competition.
>
---
#### [new 325] Categorical Equivariant Deep Learning: Category-Equivariant Neural Networks and Universal Approximation Theorems
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出范畴等变深度学习框架（CENN），统一了群、偏序集、图与层叠神经网络的等变性。通过范畴论与径向测度构建线性与非线性层，证明了有限深度CENN在连续等变变换空间中稠密，实现了广义等变万能逼近定理，拓展了等变深度学习至非几何对称性。**

- **链接: [https://arxiv.org/pdf/2511.18417v1](https://arxiv.org/pdf/2511.18417v1)**

> **作者:** Yoshihiro Maruyama
>
> **摘要:** We develop a theory of category-equivariant neural networks (CENNs) that unifies group/groupoid-equivariant networks, poset/lattice-equivariant networks, graph and sheaf neural networks. Equivariance is formulated as naturality in a topological category with Radon measures, formulating linear and nonlinear layers in the categorical setup. We prove the equivariant universal approximation theorem in the general setting: the class of finite-depth CENNs is dense in the space of continuous equivariant transformations. We instantiate the framework for groups/groupoids, posets/lattices, graphs and cellular sheaves, deriving universal approximation theorems for them in a systematic manner. Categorical equivariant deep learning thus allows us to expand the horizons of equivariant deep learning beyond group actions, encompassing not only geometric symmetries but also contextual and compositional symmetries.
>
---
#### [new 326] Enhancing UAV Search under Occlusion using Next Best View Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对无人机在高遮挡环境（如密林）中搜索时视野受限的问题，提出一种改进的“下一最佳视角”规划方法。通过引入可见性与几何双重启发式策略，优化相机视角选择，提升遮挡环境下目标检测率与覆盖效率，显著增强搜救任务的搜索性能。**

- **链接: [https://arxiv.org/pdf/2511.18353v1](https://arxiv.org/pdf/2511.18353v1)**

> **作者:** Sigrid Helene Strand; Thomas Wiedemann; Bram Burczek; Dmitriy Shutin
>
> **备注:** Submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
>
> **摘要:** Search and rescue missions are often critical following sudden natural disasters or in high-risk environmental situations. The most challenging search and rescue missions involve difficult-to-access terrains, such as dense forests with high occlusion. Deploying unmanned aerial vehicles for exploration can significantly enhance search effectiveness, facilitate access to challenging environments, and reduce search time. However, in dense forests, the effectiveness of unmanned aerial vehicles depends on their ability to capture clear views of the ground, necessitating a robust search strategy to optimize camera positioning and perspective. This work presents an optimized planning strategy and an efficient algorithm for the next best view problem in occluded environments. Two novel optimization heuristics, a geometry heuristic, and a visibility heuristic, are proposed to enhance search performance by selecting optimal camera viewpoints. Comparative evaluations in both simulated and real-world settings reveal that the visibility heuristic achieves greater performance, identifying over 90% of hidden objects in simulated forests and offering 10% better detection rates than the geometry heuristic. Additionally, real-world experiments demonstrate that the visibility heuristic provides better coverage under the canopy, highlighting its potential for improving search and rescue missions in occluded environments.
>
---
#### [new 327] pFedBBN: A Personalized Federated Test-Time Adaptation with Balanced Batch Normalization for Class-Imbalanced Data
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对联邦学习中的类别不平衡问题，提出pFedBBN框架，解决测试时适应中因数据分布偏移和少数类稀疏导致的性能下降。通过无监督的平衡批归一化与类感知聚合，实现客户端个性化适应与隐私保护下的协同优化，显著提升少数类识别能力。**

- **链接: [https://arxiv.org/pdf/2511.18066v1](https://arxiv.org/pdf/2511.18066v1)**

> **作者:** Md Akil Raihan Iftee; Syed Md. Ahnaf Hasan; Mir Sazzat Hossain; Rakibul Hasan Rajib; Amin Ahsan Ali; AKM Mahbubur Rahman; Sajib Mistry; Monowar Bhuyan
>
> **备注:** 25 pages, 7 tables, 21 figures
>
> **摘要:** Test-time adaptation (TTA) in federated learning (FL) is crucial for handling unseen data distributions across clients, particularly when faced with domain shifts and skewed class distributions. Class Imbalance (CI) remains a fundamental challenge in FL, where rare but critical classes are often severely underrepresented in individual client datasets. Although prior work has addressed CI during training through reliable aggregation and local class distribution alignment, these methods typically rely on access to labeled data or coordination among clients, and none address class unsupervised adaptation to dynamic domains or distribution shifts at inference time under federated CI constraints. Revealing the failure of state-of-the-art TTA in federated client adaptation in CI scenario, we propose pFedBBN,a personalized federated test-time adaptation framework that employs balanced batch normalization (BBN) during local client adaptation to mitigate prediction bias by treating all classes equally, while also enabling client collaboration guided by BBN similarity, ensuring that clients with similar balanced representations reinforce each other and that adaptation remains aligned with domain-specific characteristics. pFedBBN supports fully unsupervised local adaptation and introduces a class-aware model aggregation strategy that enables personalized inference without compromising privacy. It addresses both distribution shifts and class imbalance through balanced feature normalization and domain-aware collaboration, without requiring any labeled or raw data from clients. Extensive experiments across diverse baselines show that pFedBBN consistently enhances robustness and minority-class performance over state-of-the-art FL and TTA methods.
>
---
#### [new 328] Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Ultrasound-Xray Policy with Limited Labels
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对儿童髋关节发育不良（DDH）的影像筛查，提出一种以超声为主、辐射保护为原则的可选择性成像策略。通过自监督预训练与校准的置信度规则，实现有限标注下高精度测量与智能拒诊，减少不必要的X光检查，提升临床实用性。**

- **链接: [https://arxiv.org/pdf/2511.18457v1](https://arxiv.org/pdf/2511.18457v1)**

> **作者:** Duncan Stothers; Ben Stothers; Emily Schaeffer; Kishore Mulpuri
>
> **备注:** Accepted (with oral presentation) to the AAAI 2026 AIMedHealth Bridge Program
>
> **摘要:** We study an ultrasound-first, radiation-preserving policy for developmental dysplasia of the hip (DDH) that requests a radiograph only when needed. We (i) pretrain modality-specific encoders (ResNet-18) with SimSiam on a large unlabelled registry (37186 ultrasound; 19546 radiographs), (ii) freeze the backbones and fit small, measurement-faithful heads on DDH relevant landmarks and measurements (iii) calibrate a one sided conformal deferral rule on ultrasound predictions that provides finite sample coverage guarantees under exchangeability, using a held-out calibration set. Ultrasound heads predict Graf alpha, beta, and femoral head coverage; X-ray heads predict acetabular index (AI), center-edge (CE) angle and IHDI grade. On our held out labeled evaluation set, ultrasound measurement error is modest (e.g., alpha MAE ~= 9.7 degrees, coverage MAE ~= 14.0%), while radiographic probes achieve AI and CE MAEs of ~= 7.6 degrees and ~= 8.9 degrees, respectively. The calibrated US-only policy is explored across rule families (alpha-only; alpha OR coverage; alpha AND coverage), uncertainty inflation factors, and per-utility trade-offs using decision-curve analysis. Conservative settings yield high coverage with near-zero US-only rates; permissive settings (e.g., alpha OR coverage at larger deltas) achieve non-zero US-only throughput with expected coverage tradeoffs. The result is a simple, reproducible pipeline that turns limited labels into interpretable measurements and tunable selective imaging curves suitable for clinical handoff and future external validation.
>
---
#### [new 329] BOOD: Boundary-based Out-Of-Distribution Data Generation
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对出域检测（OOD）中难以生成有效异常特征的问题，提出基于边界的出域数据生成框架BOOD。通过在潜在空间中识别并扰动靠近决策边界的内域特征，利用扩散模型生成高质量的类人化异常图像，显著提升检测性能，实验表明其在CIFAR-100上大幅降低FPR95并提高AUROC。**

- **链接: [https://arxiv.org/pdf/2508.00350v1](https://arxiv.org/pdf/2508.00350v1)**

> **作者:** Qilin Liao; Shuo Yang; Bo Zhao; Ping Luo; Hengshuang Zhao
>
> **备注:** 14 pages, 8 figures, To be published in the Proceedings of the International Conference on Machine Learning (ICML) 2025
>
> **摘要:** Harnessing the power of diffusion models to synthesize auxiliary training data based on latent space features has proven effective in enhancing out-of-distribution (OOD) detection performance. However, extracting effective features outside the in-distribution (ID) boundary in latent space remains challenging due to the difficulty of identifying decision boundaries between classes. This paper proposes a novel framework called Boundary-based Out-Of-Distribution data generation (BOOD), which synthesizes high-quality OOD features and generates human-compatible outlier images using diffusion models. BOOD first learns a text-conditioned latent feature space from the ID dataset, selects ID features closest to the decision boundary, and perturbs them to cross the decision boundary to form OOD features. These synthetic OOD features are then decoded into images in pixel space by a diffusion model. Compared to previous works, BOOD provides a more training efficient strategy for synthesizing informative OOD features, facilitating clearer distinctions between ID and OOD data. Extensive experimental results on common benchmarks demonstrate that BOOD surpasses the state-of-the-art method significantly, achieving a 29.64% decrease in average FPR95 (40.31% vs. 10.67%) and a 7.27% improvement in average AUROC (90.15% vs. 97.42%) on the CIFAR-100 dataset.
>
---
#### [new 330] Inverse Rendering for High-Genus Surface Meshes from Multi-View Images
- **分类: cs.GR; cs.CV**

- **简介: 该论文针对高亏格表面网格从多视角图像中重建的逆渲染任务，解决现有方法在高亏格表面丢失拓扑特征、低亏格表面过度平滑的问题。提出自适应V-cycle重网格化与重参数化Adam优化器，结合高斯-邦内定理约束拓扑一致性，有效提升几何细节与拓扑保真度。**

- **链接: [https://arxiv.org/pdf/2511.18680v1](https://arxiv.org/pdf/2511.18680v1)**

> **作者:** Xiang Gao; Xinmu Wang; Xiaolong Wu; Jiazhi Li; Jingyu Shi; Yu Guo; Yuanpeng Liu; Xiyun Song; Heather Yu; Zongfang Lin; Xianfeng David Gu
>
> **备注:** 3DV2026 Accepted (Poster)
>
> **摘要:** We present a topology-informed inverse rendering approach for reconstructing high-genus surface meshes from multi-view images. Compared to 3D representations like voxels and point clouds, mesh-based representations are preferred as they enable the application of differential geometry theory and are optimized for modern graphics pipelines. However, existing inverse rendering methods often fail catastrophically on high-genus surfaces, leading to the loss of key topological features, and tend to oversmooth low-genus surfaces, resulting in the loss of surface details. This failure stems from their overreliance on Adam-based optimizers, which can lead to vanishing and exploding gradients. To overcome these challenges, we introduce an adaptive V-cycle remeshing scheme in conjunction with a re-parametrized Adam optimizer to enhance topological and geometric awareness. By periodically coarsening and refining the deforming mesh, our method informs mesh vertices of their current topology and geometry before optimization, mitigating gradient issues while preserving essential topological features. Additionally, we enforce topological consistency by constructing topological primitives with genus numbers that match those of ground truth using Gauss-Bonnet theorem. Experimental results demonstrate that our inverse rendering approach outperforms the current state-of-the-art method, achieving significant improvements in Chamfer Distance and Volume IoU, particularly for high-genus surfaces, while also enhancing surface details for low-genus surfaces.
>
---
#### [new 331] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对团队运动中多智能体轨迹预测任务，解决现有模型仅关注单个智能体精度而忽视群体协同性的问题。提出CausalTraj模型，基于时序因果和似然性建模，优化联合预测性能，在NBA、篮球和足球数据集上实现最优联合指标，生成更合理、连贯的多智能体未来轨迹。**

- **链接: [https://arxiv.org/pdf/2511.18248v1](https://arxiv.org/pdf/2511.18248v1)**

> **作者:** Wei Zhen Teoh
>
> **备注:** 9 pages, 3 figures, accepted to the AI4TS Workshop at AAAI 2026
>
> **摘要:** Jointly forecasting trajectories of multiple interacting agents is a core challenge in sports analytics and other domains involving complex group dynamics. Accurate prediction enables realistic simulation and strategic understanding of gameplay evolution. Most existing models are evaluated solely on per-agent accuracy metrics (minADE, minFDE), which assess each agent independently on its best-of-k prediction. However these metrics overlook whether the model learns which predicted trajectories can jointly form a plausible multi-agent future. Many state-of-the-art models are designed and optimized primarily based on these metrics. As a result, they may underperform on joint predictions and also fail to generate coherent, interpretable multi-agent scenarios in team sports. We propose CausalTraj, a temporally causal, likelihood-based model that is built to generate jointly probable multi-agent trajectory forecasts. To better assess collective modeling capability, we emphasize joint metrics (minJADE, minJFDE) that measure joint accuracy across agents within the best generated scenario sample. Evaluated on the NBA SportVU, Basketball-U, and Football-U datasets, CausalTraj achieves competitive per-agent accuracy and the best recorded results on joint metrics, while yielding qualitatively coherent and realistic gameplay evolutions.
>
---
#### [new 332] Flow Map Distillation Without Data
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究生成模型加速任务，针对现有流模型蒸馏依赖外部数据导致的教师-数据不匹配问题，提出无需数据的蒸馏方法。通过仅从先验分布采样，构建可纠正自身误差的框架，实现单步采样下FID达1.45（256x256）和1.49（512x512），显著优于已有方法。**

- **链接: [https://arxiv.org/pdf/2511.19428v1](https://arxiv.org/pdf/2511.19428v1)**

> **作者:** Shangyuan Tong; Nanye Ma; Saining Xie; Tommi Jaakkola
>
> **摘要:** State-of-the-art flow models achieve remarkable quality but require slow, iterative sampling. To accelerate this, flow maps can be distilled from pre-trained teachers, a procedure that conventionally requires sampling from an external dataset. We argue that this data-dependency introduces a fundamental risk of Teacher-Data Mismatch, as a static dataset may provide an incomplete or even misaligned representation of the teacher's full generative capabilities. This leads us to question whether this reliance on data is truly necessary for successful flow map distillation. In this work, we explore a data-free alternative that samples only from the prior distribution, a distribution the teacher is guaranteed to follow by construction, thereby circumventing the mismatch risk entirely. To demonstrate the practical viability of this philosophy, we introduce a principled framework that learns to predict the teacher's sampling path while actively correcting for its own compounding errors to ensure high fidelity. Our approach surpasses all data-based counterparts and establishes a new state-of-the-art by a significant margin. Specifically, distilling from SiT-XL/2+REPA, our method reaches an impressive FID of 1.45 on ImageNet 256x256, and 1.49 on ImageNet 512x512, both with only 1 sampling step. We hope our work establishes a more robust paradigm for accelerating generative models and motivates the broader adoption of flow map distillation without data.
>
---
#### [new 333] FedPoisonTTP: A Threat Model and Poisoning Attack for Federated Test-Time Personalization
- **分类: cs.CR; cs.CV**

- **简介: 该论文针对联邦学习中测试时个性化（Test-time Personalization）的安全风险，提出FedPoisonTTP攻击框架。针对客户端在测试时本地适应过程中可能遭受的数据投毒问题，通过构建代理模型、生成高熵或类别置信的毒化样本，实现隐蔽攻击，导致全局与本地性能显著下降。**

- **链接: [https://arxiv.org/pdf/2511.19248v1](https://arxiv.org/pdf/2511.19248v1)**

> **作者:** Md Akil Raihan Iftee; Syed Md. Ahnaf Hasan; Amin Ahsan Ali; AKM Mahbubur Rahman; Sajib Mistry; Aneesh Krishna
>
> **备注:** 13 pages, 3 figures, 2 tables
>
> **摘要:** Test-time personalization in federated learning enables models at clients to adjust online to local domain shifts, enhancing robustness and personalization in deployment. Yet, existing federated learning work largely overlooks the security risks that arise when local adaptation occurs at test time. Heterogeneous domain arrivals, diverse adaptation algorithms, and limited cross-client visibility create vulnerabilities where compromised participants can craft poisoned inputs and submit adversarial updates that undermine both global and per-client performance. To address this threat, we introduce FedPoisonTTP, a realistic grey-box attack framework that explores test-time data poisoning in the federated adaptation setting. FedPoisonTTP distills a surrogate model from adversarial queries, synthesizes in-distribution poisons using feature-consistency, and optimizes attack objectives to generate high-entropy or class-confident poisons that evade common adaptation filters. These poisons are injected during local adaptation and spread through collaborative updates, leading to broad degradation. Extensive experiments on corrupted vision benchmarks show that compromised participants can substantially diminish overall test-time performance.
>
---
#### [new 334] Robust Detection of Retinal Neovascularization in Widefield Optical Coherence Tomography
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对糖尿病视网膜病变中视网膜新生血管（RNV）的早期检测难题，提出一种基于深度学习的宽视野OCTA图像自动分析方法。不同于传统分层分割策略，将RNV识别转为直接二分类定位任务，实现了高精度诊断与分割，支持纵向监测，显著提升RNV筛查效率与临床应用价值。**

- **链接: [https://arxiv.org/pdf/2511.17744v1](https://arxiv.org/pdf/2511.17744v1)**

> **作者:** Jinyi Hao; Jie Wang; Kotaro Tsuboi; Liqin Gao; Tristan T. Hormel; Yukun Guo; An-Lun Wu; Min Gao; Christina J. Flaxel; Steven T. Bailey; Thomas S. Hwang; Yali Jia
>
> **备注:** 17 pages, 11 figures. Submitted to Optica. Corresponding author: Yali Jia. Affiliations: ((1) Casey Eye Institute, Oregon Health & Science University, USA (2) Department of Ophthalmology, Aichi Medical University, Japan (3) Department of Biomedical Engineering, Oregon Health & Science University, USA (4) Department of Ophthalmology, Mackay Memorial Hospital, Taiwan)
>
> **摘要:** Retinal neovascularization (RNV) is a vision threatening development in diabetic retinopathy (DR). Vision loss associated with RNV is preventable with timely intervention, making RNV clinical screening and monitoring a priority. Optical coherence tomography (OCT) angiography (OCTA) provides high-resolution imaging and high-sensitivity detection of RNV lesions. With recent commercial devices introducing widefield OCTA imaging to the clinic, the technology stands to improve early detection of RNV pathology. However, to meet clinical requirements these imaging capabilities must be combined with effective RNV detection and quantification, but existing algorithms for OCTA images are optimized for conventional, i.e. narrow, fields of view. Here, we present a novel approach for RNV diagnosis and staging on widefield OCT/OCTA. Unlike conventional methods dependent on multi-layer retinal segmentation, our model reframes RNV identification as a direct binary localization task. Our fully automated approach was trained and validated on 589 widefield scans (17x17-mm to 26x21-mm) collected from multiple devices at multiple clinics. Our method achieved a device-dependent area under curve (AUC) ranging from 0.96 to 0.99 for RNV diagnosis, and mean intersection over union (IOU) ranging from 0.76 to 0.88 for segmentation. We also demonstrate our method's ability to monitor lesion growth longitudinally. Our results indicate that deep learning-based analysis for widefield OCTA images could offer a valuable means for improving RNV screening and management.
>
---
#### [new 335] AutoFocus-IL: VLM-based Saliency Maps for Data-Efficient Visual Imitation Learning without Extra Human Annotations
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉模仿学习中的数据效率与泛化能力问题，提出AutoFocus-IL方法。通过利用视觉语言模型自动生成时序显著性图，引导策略关注任务相关特征，抑制干扰因素，无需额外人工标注。实验表明其性能优于标准行为克隆及依赖人类监督的先进方法。**

- **链接: [https://arxiv.org/pdf/2511.18617v1](https://arxiv.org/pdf/2511.18617v1)**

> **作者:** Litian Gong; Fatemeh Bahrani; Yutai Zhou; Amin Banayeeanzade; Jiachen Li; Erdem Biyik
>
> **备注:** 8 pages, 6 figures. Code and datasets available at http://autofocus-il.github.io/
>
> **摘要:** AutoFocus-IL is a simple yet effective method to improve data efficiency and generalization in visual imitation learning by guiding policies to attend to task-relevant features rather than distractors and spurious correlations. Although saliency regularization has emerged as a promising way to achieve this, existing approaches typically require costly supervision such as human gaze data or manual saliency annotations. In contrast, AutoFocus-IL leverages vision-language models (VLMs) to automatically identify and track key objects in demonstrations, generating temporal saliency maps that highlight causal visual signals while suppressing distractors. These maps are then used to regularize behavior cloning policies, yielding stronger alignment between visual attention and task-relevant cues. Experiments in both the CARLA simulator and real-robot manipulation tasks demonstrate that AutoFocus-IL not only outperforms standard behavior cloning but also surpasses state-of-the-art baselines that assume privileged access to human supervision, such as gaze data. Code, datasets, and trained policy videos are available at https://AutoFocus-IL.github.io/.
>
---
#### [new 336] Classification of Transient Astronomical Object Light Curves Using LSTM Neural Networks
- **分类: cs.LG; astro-ph.IM; cs.AI; cs.CV**

- **简介: 该论文针对天文瞬变源光变曲线分类任务，使用双向LSTM网络对PLAsTiCC数据集进行分类。为缓解类别不平衡，将14类合并为5类。通过预处理与模型训练，发现模型在S-Like和周期性类别表现优异，但在快速、长时标及周期/非周期区分上表现较差，且早期观测数据导致性能下降。**

- **链接: [https://arxiv.org/pdf/2511.17564v1](https://arxiv.org/pdf/2511.17564v1)**

> **作者:** Guilherme Grancho D. Fernandes; Marco A. Barroca; Mateus dos Santos; Rafael S. Oliveira
>
> **备注:** 12 pages, 11 figures, 2 tables
>
> **摘要:** This study presents a bidirectional Long Short-Term Memory (LSTM) neural network for classifying transient astronomical object light curves from the Photometric LSST Astronomical Time-series Classification Challenge (PLAsTiCC) dataset. The original fourteen object classes were reorganized into five generalized categories (S-Like, Fast, Long, Periodic, and Non-Periodic) to address class imbalance. After preprocessing with padding, temporal rescaling, and flux normalization, a bidirectional LSTM network with masking layers was trained and evaluated on a test set of 19,920 objects. The model achieved strong performance for S-Like and Periodic classes, with ROC area under the curve (AUC) values of 0.95 and 0.99, and Precision-Recall AUC values of 0.98 and 0.89, respectively. However, performance was significantly lower for Fast and Long classes (ROC AUC of 0.68 for Long class), and the model exhibited difficulty distinguishing between Periodic and Non-Periodic objects. Evaluation on partial light curve data (5, 10,and 20 days from detection) revealed substantial performance degradation, with increased misclassification toward the S-Like class. These findings indicate that class imbalance and limited temporal information are primary limitations, suggesting that class balancing strategies and preprocessing techniques focusing on detection moments could improve performance.
>
---
#### [new 337] Auxiliary Gene Learning: Spatial Gene Expression Estimation by Auxiliary Gene Selection
- **分类: cs.LG; cs.CV; q-bio.GN**

- **简介: 该论文针对空间转录组学中基因表达估计的噪声问题，提出辅助基因学习（AGL）框架。通过将低表达基因作为辅助任务，联合训练提升目标基因预测性能。设计基于先验知识的可微分顶-k基因选择方法（DkGSB），有效解决辅助基因组合优化难题，显著提升估计精度。**

- **链接: [https://arxiv.org/pdf/2511.18336v1](https://arxiv.org/pdf/2511.18336v1)**

> **作者:** Kaito Shiku; Kazuya Nishimura; Shinnosuke Matsuo; Yasuhiro Kojima; Ryoma Bise
>
> **备注:** Accepted to Association for the Advancement of Artificial Intelligence (AAAI) 2026
>
> **摘要:** Spatial transcriptomics (ST) is a novel technology that enables the observation of gene expression at the resolution of individual spots within pathological tissues. ST quantifies the expression of tens of thousands of genes in a tissue section; however, heavy observational noise is often introduced during measurement. In prior studies, to ensure meaningful assessment, both training and evaluation have been restricted to only a small subset of highly variable genes, and genes outside this subset have also been excluded from the training process. However, since there are likely co-expression relationships between genes, low-expression genes may still contribute to the estimation of the evaluation target. In this paper, we propose $Auxiliary \ Gene \ Learning$ (AGL) that utilizes the benefit of the ignored genes by reformulating their expression estimation as auxiliary tasks and training them jointly with the primary tasks. To effectively leverage auxiliary genes, we must select a subset of auxiliary genes that positively influence the prediction of the target genes. However, this is a challenging optimization problem due to the vast number of possible combinations. To overcome this challenge, we propose Prior-Knowledge-Based Differentiable Top-$k$ Gene Selection via Bi-level Optimization (DkGSB), a method that ranks genes by leveraging prior knowledge and relaxes the combinatorial selection problem into a differentiable top-$k$ selection problem. The experiments confirm the effectiveness of incorporating auxiliary genes and show that the proposed method outperforms conventional auxiliary task learning approaches.
>
---
#### [new 338] TeamPath: Building MultiModal Pathology Experts with Reasoning AI Copilots
- **分类: q-bio.QM; cs.CV**

- **简介: 该论文提出TeamPath，一种基于强化学习与路由增强的多模态病理学AI系统，旨在解决现有模型在病理诊断中缺乏严谨推理路径和跨任务适应性的问题。通过整合图像、文本与转录组数据，系统可辅助专家进行精准诊断、信息摘要与跨模态生成，提升病理分析效率与准确性。**

- **链接: [https://arxiv.org/pdf/2511.17652v1](https://arxiv.org/pdf/2511.17652v1)**

> **作者:** Tianyu Liu; Weihao Xuan; Hao Wu; Peter Humphrey; Marcello DiStasio; Heli Qi; Rui Yang; Simeng Han; Tinglin Huang; Fang Wu; Nan Liu; Irene Li; Hua Xu; Hongyu Zhao
>
> **备注:** 35 pages, 6 figures
>
> **摘要:** Advances in AI have introduced several strong models in computational pathology to usher it into the era of multi-modal diagnosis, analysis, and interpretation. However, the current pathology-specific visual language models still lack capacities in making diagnosis with rigorous reasoning paths as well as handling divergent tasks, and thus challenges of building AI Copilots for real scenarios still exist. Here we introduce TeamPath, an AI system powered by reinforcement learning and router-enhanced solutions based on large-scale histopathology multimodal datasets, to work as a virtual assistant for expert-level disease diagnosis, patch-level information summarization, and cross-modality generation to integrate transcriptomic information for the clinical usage. We also collaborate with pathologists from Yale School of Medicine to demonstrate that TeamPath can assist them in working more efficiently by identifying and correcting expert conclusions and reasoning paths. Overall, TeamPath can flexibly choose the best settings according to the needs, and serve as an innovative and reliable system for information communication across different modalities and experts.
>
---
#### [new 339] Linear Algebraic Approaches to Neuroimaging Data Compression: A Comparative Analysis of Matrix and Tensor Decomposition Methods for High-Dimensional Medical Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像压缩任务，旨在解决高维神经影像数据存储与传输中的效率问题。通过对比Tucker分解与SVD方法，发现Tucker在保持结构与时间关系上更优，适用于需高保真重建的场景。**

- **链接: [https://arxiv.org/pdf/2511.18197v1](https://arxiv.org/pdf/2511.18197v1)**

> **作者:** Jaeho Kim; Daniel David; Ana Vizitiv
>
> **摘要:** This paper evaluates Tucker decomposition and Singular Value Decomposition (SVD) for compressing neuroimaging data. Tucker decomposition preserves multi-dimensional relationships, achieving superior reconstruction fidelity and perceptual similarity. SVD excels in extreme compression but sacrifices fidelity. The results highlight Tucker decomposition's suitability for applications requiring the preservation of structural and temporal relationships.
>
---
#### [new 340] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文针对半监督学习中的类别不平衡问题，提出SC-SSL框架。通过解耦采样控制，分别处理训练与推理阶段的特征与权重不平衡，有效缓解因分布差异导致的模型偏差，提升少数类分类性能。**

- **链接: [https://arxiv.org/pdf/2511.18773v1](https://arxiv.org/pdf/2511.18773v1)**

> **作者:** Senmao Tian; Xiang Wei; Shunli Zhang
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Class imbalance remains a critical challenge in semi-supervised learning (SSL), especially when distributional mismatches between labeled and unlabeled data lead to biased classification. Although existing methods address this issue by adjusting logits based on the estimated class distribution of unlabeled data, they often handle model imbalance in a coarse-grained manner, conflating data imbalance with bias arising from varying class-specific learning difficulties. To address this issue, we propose a unified framework, SC-SSL, which suppresses model bias through decoupled sampling control. During training, we identify the key variables for sampling control under ideal conditions. By introducing a classifier with explicit expansion capability and adaptively adjusting sampling probabilities across different data distributions, SC-SSL mitigates feature-level imbalance for minority classes. In the inference phase, we further analyze the weight imbalance of the linear classifier and apply post-hoc sampling control with an optimization bias vector to directly calibrate the logits. Extensive experiments across various benchmark datasets and distribution settings validate the consistency and state-of-the-art performance of SC-SSL.
>
---
#### [new 341] GRIT-LP: Graph Transformer with Long-Range Skip Connection and Partitioned Spatial Graphs for Accurate Ice Layer Thickness Prediction
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对极地冰层厚度预测任务，解决图变压器因过度平滑和长程依赖建模弱导致的深度受限问题。提出GRIT-LP模型，通过分块空间图构建保留局部一致性，并引入长程跳跃连接增强信息流动，显著提升预测精度。**

- **链接: [https://arxiv.org/pdf/2511.18716v1](https://arxiv.org/pdf/2511.18716v1)**

> **作者:** Zesheng Liu; Maryam Rahnemoonfar
>
> **摘要:** Graph transformers have demonstrated remarkable capability on complex spatio-temporal tasks, yet their depth is often limited by oversmoothing and weak long-range dependency modeling. To address these challenges, we introduce GRIT-LP, a graph transformer explicitly designed for polar ice-layer thickness estimation from polar radar imagery. Accurately estimating ice layer thickness is critical for understanding snow accumulation, reconstructing past climate patterns and reducing uncertainties in projections of future ice sheet evolution and sea level rise. GRIT-LP combines an inductive geometric graph learning framework with self-attention mechanism, and introduces two major innovations that jointly address challenges in modeling the spatio-temporal patterns of ice layers: a partitioned spatial graph construction strategy that forms overlapping, fully connected local neighborhoods to preserve spatial coherence and suppress noise from irrelevant long-range links, and a long-range skip connection mechanism within the transformer that improves information flow and mitigates oversmoothing in deeper attention layers. We conducted extensive experiments, demonstrating that GRIT-LP outperforms current state-of-the-art methods with a 24.92\% improvement in root mean squared error. These results highlight the effectiveness of graph transformers in modeling spatiotemporal patterns by capturing both localized structural features and long-range dependencies across internal ice layers, and demonstrate their potential to advance data-driven understanding of cryospheric processes.
>
---
#### [new 342] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文针对视频到音频生成任务，解决现有方法因目标纠缠导致的多维感知平衡难题。提出PrismAudio框架，通过四类专项思维链（CoT）与对应奖励函数，实现多维度强化学习优化，并引入Fast-GRPO降低计算开销。构建AudioCanvas基准测试集，实验证明其在四项感知维度上均达领先性能。**

- **链接: [https://arxiv.org/pdf/2511.18833v1](https://arxiv.org/pdf/2511.18833v1)**

> **作者:** Huadai Liu; Kaicheng Luo; Wen Wang; Qian Chen; Peiwen Sun; Rongjie Huang; Xiangang Li; Jieping Ye; Wei Xue
>
> **备注:** Preprint
>
> **摘要:** Video-to-Audio (V2A) generation requires balancing four critical perceptual dimensions: semantic consistency, audio-visual temporal synchrony, aesthetic quality, and spatial accuracy; yet existing methods suffer from objective entanglement that conflates competing goals in single loss functions and lack human preference alignment. We introduce PrismAudio, the first framework to integrate Reinforcement Learning into V2A generation with specialized Chain-of-Thought (CoT) planning. Our approach decomposes monolithic reasoning into four specialized CoT modules (Semantic, Temporal, Aesthetic, and Spatial CoT), each paired with targeted reward functions. This CoT-reward correspondence enables multidimensional RL optimization that guides the model to jointly generate better reasoning across all perspectives, solving the objective entanglement problem while preserving interpretability. To make this optimization computationally practical, we propose Fast-GRPO, which employs hybrid ODE-SDE sampling that dramatically reduces the training overhead compared to existing GRPO implementations. We also introduce AudioCanvas, a rigorous benchmark that is more distributionally balanced and covers more realistically diverse and challenging scenarios than existing datasets, with 300 single-event classes and 501 multi-event samples. Experimental results demonstrate that PrismAudio achieves state-of-the-art performance across all four perceptual dimensions on both the in-domain VGGSound test set and out-of-domain AudioCanvas benchmark. The project page is available at https://PrismAudio-Project.github.io.
>
---
#### [new 343] Neural B-Frame Coding: Tackling Domain Shift Issues with Lightweight Online Motion Resolution Adaptation
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文针对视频编码中B帧因GOP尺寸不匹配导致的大运动估计不准问题，提出轻量级分类器动态预测下采样因子，以平衡率失真与计算成本。通过三种分类方法实现在线运动分辨率自适应，无需重训练现有编码器，显著降低复杂度并逼近最优性能。**

- **链接: [https://arxiv.org/pdf/2511.18724v1](https://arxiv.org/pdf/2511.18724v1)**

> **作者:** Sang NguyenQuang; Xiem HoangVan; Wen-Hsiao Peng
>
> **备注:** Accepted by TCAS-II: Express Briefs
>
> **摘要:** Learned B-frame codecs with hierarchical temporal prediction often encounter the domain-shift issue due to mismatches between the Group-of-Pictures (GOP) sizes for training and testing, leading to inaccurate motion estimates, particularly for large motion. A common solution is to turn large motion into small motion by downsampling video frames during motion estimation. However, determining the optimal downsampling factor typically requires costly rate-distortion optimization. This work introduces lightweight classifiers to predict downsampling factors. These classifiers leverage simple state signals from current and reference frames to balance rate-distortion performance with computational cost. Three variants are proposed: (1) a binary classifier (Bi-Class) trained with Focal Loss to choose between high and low resolutions, (2) a multi-class classifier (Mu-Class) trained with novel soft labels based on rate-distortion costs, and (3) a co-class approach (Co-Class) that combines the predictive capability of the multi-class classifier with the selective search of the binary classifier. All classifier methods can work seamlessly with existing B-frame codecs without requiring codec retraining. Experimental results show that they achieve coding performance comparable to exhaustive search methods while significantly reducing computational complexity. The code is available at: https://github.com/NYCU-MAPL/Fast-OMRA.git.
>
---
#### [new 344] Deep Learning-based Lightweight RGB Object Tracking for Augmented Reality Devices
- **分类: cs.HC; cs.CV**

- **简介: 该论文针对增强现实（AR）设备资源受限问题，提出一种轻量级RGB目标跟踪算法。通过紧凑的孪生网络结构及模型剪枝、量化、知识蒸馏等优化技术，显著降低计算与内存开销，实现在移动AR头显上30 FPS的实时跟踪，精度接近顶尖方法，解决了高精度追踪在轻量设备上难以部署的问题。**

- **链接: [https://arxiv.org/pdf/2511.17508v1](https://arxiv.org/pdf/2511.17508v1)**

> **作者:** Alice Smith; Bob Johnson; Xiaoyu Zhu; Carol Lee
>
> **摘要:** Augmented Reality (AR) applications often require robust real-time tracking of objects in the user's environment to correctly overlay virtual content. Recent advances in computer vision have produced highly accurate deep learning-based object trackers, but these models are typically too heavy in computation and memory for wearable AR devices. In this paper, we present a lightweight RGB object tracking algorithm designed specifically for resource-constrained AR platforms. The proposed tracker employs a compact Siamese neural network architecture and incorporates optimization techniques such as model pruning, quantization, and knowledge distillation to drastically reduce model size and inference cost while maintaining high tracking accuracy. We train the tracker offline on large video datasets using deep convolutional neural networks and then deploy it on-device for real-time tracking. Experimental results on standard tracking benchmarks show that our approach achieves comparable accuracy to state-of-the-art trackers, yet runs in real-time on a mobile AR headset at around 30 FPS -- more than an order of magnitude faster than prior high-performance trackers on the same hardware. This work enables practical, robust object tracking for AR use-cases, opening the door to more interactive and dynamic AR experiences on lightweight devices.
>
---
#### [new 345] Learning Straight Flows: Variational Flow Matching for Efficient Generation
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对生成模型中流匹配（Flow Matching）因依赖弯曲轨迹导致无法实现一步生成的问题，提出S-VFM方法。通过引入变分隐变量建模生成全局信息，显式约束轨迹直线性，提升训练与推理效率，在多个基准上表现优异。**

- **链接: [https://arxiv.org/pdf/2511.17583v1](https://arxiv.org/pdf/2511.17583v1)**

> **作者:** Chenrui Ma; Xi Xiao; Tianyang Wang; Xiao Wang; Yanning Shen
>
> **摘要:** Flow Matching has limited ability in achieving one-step generation due to its reliance on learned curved trajectories. Previous studies have attempted to address this limitation by either modifying the coupling distribution to prevent interpolant intersections or introducing consistency and mean-velocity modeling to promote straight trajectory learning. However, these approaches often suffer from discrete approximation errors, training instability, and convergence difficulties. To tackle these issues, in the present work, we propose \textbf{S}traight \textbf{V}ariational \textbf{F}low \textbf{M}atching (\textbf{S-VFM}), which integrates a variational latent code representing the ``generation overview'' into the Flow Matching framework. \textbf{S-VFM} explicitly enforces trajectory straightness, ideally producing linear generation paths. The proposed method achieves competitive performance across three challenge benchmarks and demonstrates advantages in both training and inference efficiency compared with existing methods.
>
---
#### [new 346] Temporal-adaptive Weight Quantization for Spiking Neural Networks
- **分类: cs.NE; cs.AI; cs.CV**

- **简介: 该论文针对脉冲神经网络（SNN）中的权重量化问题，提出时序自适应权重量化（TaWQ）方法。通过模拟星形胶质细胞的突触调节机制，使量化权重随时间动态调整，实现超低比特存储与计算，在保持高精度的同时显著降低能耗，适用于静态与类脑数据集。**

- **链接: [https://arxiv.org/pdf/2511.17567v1](https://arxiv.org/pdf/2511.17567v1)**

> **作者:** Han Zhang; Qingyan Meng; Jiaqi Wang; Baiyu Chen; Zhengyu Ma; Xiaopeng Fan
>
> **摘要:** Weight quantization in spiking neural networks (SNNs) could further reduce energy consumption. However, quantizing weights without sacrificing accuracy remains challenging. In this study, inspired by astrocyte-mediated synaptic modulation in the biological nervous systems, we propose Temporal-adaptive Weight Quantization (TaWQ), which incorporates weight quantization with temporal dynamics to adaptively allocate ultra-low-bit weights along the temporal dimension. Extensive experiments on static (e.g., ImageNet) and neuromorphic (e.g., CIFAR10-DVS) datasets demonstrate that our TaWQ maintains high energy efficiency (4.12M, 0.63mJ) while incurring a negligible quantization loss of only 0.22% on ImageNet.
>
---
#### [new 347] SloMo-Fast: Slow-Momentum and Fast-Adaptive Teachers for Source-Free Continual Test-Time Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对源无关的持续测试时自适应（CTTA）任务，解决现有方法依赖源数据、易遗忘旧域的问题。提出SloMo-Fast框架，采用慢教师（保记忆）与快教师（速适应）双教师机制，实现对新旧域的高效适应与泛化。引入Cyclic-TTA基准，验证其在循环域变化下的优越性能。**

- **链接: [https://arxiv.org/pdf/2511.18468v1](https://arxiv.org/pdf/2511.18468v1)**

> **作者:** Md Akil Raihan Iftee; Mir Sazzat Hossain; Rakibul Hasan Rajib; Tariq Iqbal; Md Mofijul Islam; M Ashraful Amin; Amin Ahsan Ali; AKM Mahbubur Rahman
>
> **备注:** 38 pages, 38 tables, 16 figures
>
> **摘要:** Continual Test-Time Adaptation (CTTA) is crucial for deploying models in real-world applications with unseen, evolving target domains. Existing CTTA methods, however, often rely on source data or prototypes, limiting their applicability in privacy-sensitive and resource-constrained settings. Additionally, these methods suffer from long-term forgetting, which degrades performance on previously encountered domains as target domains shift. To address these challenges, we propose SloMo-Fast, a source-free, dual-teacher CTTA framework designed for enhanced adaptability and generalization. It includes two complementary teachers: the Slow-Teacher, which exhibits slow forgetting and retains long-term knowledge of previously encountered domains to ensure robust generalization, and the Fast-Teacher rapidly adapts to new domains while accumulating and integrating knowledge across them. This framework preserves knowledge of past domains and adapts efficiently to new ones. We also introduce Cyclic Test-Time Adaptation (Cyclic-TTA), a novel CTTA benchmark that simulates recurring domain shifts. Our extensive experiments demonstrate that SloMo-Fast consistently outperforms state-of-the-art methods across Cyclic-TTA, as well as ten other CTTA settings, highlighting its ability to both adapt and generalize across evolving and revisited domains.
>
---
#### [new 348] Animated Territorial Data Extractor (ATDE): A Computer-Vision Method for Extracting Territorial Data from Animated Historical Maps
- **分类: cs.CY; cs.CV**

- **简介: 该论文提出ATDE，一种基于计算机视觉的工具，用于从动画历史地图视频中提取领土数据。针对手动提取效率低、依赖地理数据的问题，通过颜色分割与像素计数，将视频转为时间序列数据，实现无需形状文件的自动化处理，适用于教育与初步分析。**

- **链接: [https://arxiv.org/pdf/2511.17920v1](https://arxiv.org/pdf/2511.17920v1)**

> **作者:** Hamza Alshamy; Isaiah Woram; Advay Mishra; Zihan Xia; Pascal Wallisch
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** We present Animated Territorial Data Extractor (ATDE), a computer vision tool that extracts quantitative territorial data from animated historical map videos. ATDE employs HSV-based color segmentation, RGB channel filtering, and Direct-Neighbor Filtering to identify and count pixels representing territorial control. Combined with preprocessing for temporal alignment and cross-video scaling, the pipeline converts animated videos into structured time-series data. We demonstrate the tool on ten Chinese dynasties (200 BCE - 1912 CE), producing year-by-year pixel counts that align with expected historical patterns. While not a substitute for authoritative historical datasets, ATDE is well-suited for educational demonstrations, preliminary data exploration, and comparative analysis of territorial dynamics. The tool requires no pre-existing shapefiles and can be applied to any animated map video given seed colors and basic configuration. Code and examples are available on GitHub.
>
---
#### [new 349] Multimodal Real-Time Anomaly Detection and Industrial Applications
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文针对工业场景中的实时异常检测任务，解决多模态数据融合与高精度实时识别难题。提出两代系统：首代基于YOLOv8、ByteTrack与AST；进阶版融合多模型音频、双目标检测器及双向跨模态注意力，实现高效精准的多模态异常检测，适用于工业安全等实际场景。**

- **链接: [https://arxiv.org/pdf/2511.18698v1](https://arxiv.org/pdf/2511.18698v1)**

> **作者:** Aman Verma; Keshav Samdani; Mohd. Samiuddin Shafi
>
> **摘要:** This paper presents the design, implementation, and evolution of a comprehensive multimodal room-monitoring system that integrates synchronized video and audio processing for real-time activity recognition and anomaly detection. We describe two iterations of the system: an initial lightweight implementation using YOLOv8, ByteTrack, and the Audio Spectrogram Transformer (AST), and an advanced version that incorporates multi-model audio ensembles, hybrid object detection, bidirectional cross-modal attention, and multi-method anomaly detection. The evolution demonstrates significant improvements in accuracy, robustness, and industrial applicability. The advanced system combines three audio models (AST, Wav2Vec2, and HuBERT) for comprehensive audio understanding, dual object detectors (YOLO and DETR) for improved accuracy, and sophisticated fusion mechanisms for enhanced cross-modal learning. Experimental evaluation shows the system's effectiveness in general monitoring scenarios as well as specialized industrial safety applications, achieving real-time performance on standard hardware while maintaining high accuracy.
>
---
#### [new 350] MobileVLA-R1: Reinforcing Vision-Language-Action for Mobile Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对四足机器人视觉-语言-动作（VLA）任务中语义理解与连续控制难以对齐的问题，提出MobileVLA-R1框架。通过构建多粒度思维链数据集，采用两阶段训练提升推理一致性与控制稳定性，在仿真与真实场景均实现显著性能提升。**

- **链接: [https://arxiv.org/pdf/2511.17889v1](https://arxiv.org/pdf/2511.17889v1)**

> **作者:** Ting Huang; Dongjian Li; Rui Yang; Zeyu Zhang; Zida Yang; Hao Tang
>
> **摘要:** Grounding natural-language instructions into continuous control for quadruped robots remains a fundamental challenge in vision language action. Existing methods struggle to bridge high-level semantic reasoning and low-level actuation, leading to unstable grounding and weak generalization in the real world. To address these issues, we present MobileVLA-R1, a unified vision-language-action framework that enables explicit reasoning and continuous control for quadruped robots. We construct MobileVLA-CoT, a large-scale dataset of multi-granularity chain-of-thought (CoT) for embodied trajectories, providing structured reasoning supervision for alignment. Built upon this foundation, we introduce a two-stage training paradigm that combines supervised CoT alignment with GRPO reinforcement learning to enhance reasoning consistency, control stability, and long-horizon execution. Extensive evaluations on VLN and VLA tasks demonstrate superior performance over strong baselines, with approximately a 5% improvement. Real-world deployment on a quadruped robot validates robust performance in complex environments. Code: https://github.com/AIGeeksGroup/MobileVLA-R1. Website: https://aigeeksgroup.github.io/MobileVLA-R1.
>
---
#### [new 351] Robust and Generalizable GNN Fine-Tuning via Uncertainty-aware Adapter Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对预训练图神经网络（GNN）微调中的噪声敏感与泛化能力弱问题，提出不确定性感知适配器（UAdapterGNN）。通过引入高斯概率适配模块，使模型在面对噪声边和模糊节点属性时具备更强鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18859v1](https://arxiv.org/pdf/2511.18859v1)**

> **作者:** Bo Jiang; Weijun Zhao; Beibei Wang; Xiao Wang; Jin Tang
>
> **摘要:** Recently, fine-tuning large-scale pre-trained GNNs has yielded remarkable attention in adapting pre-trained GNN models for downstream graph learning tasks. One representative fine-tuning method is to exploit adapter (termed AdapterGNN) which aims to 'augment' the pre-trained model by inserting a lightweight module to make the 'augmented' model better adapt to the downstream tasks. However, graph data may contain various types of noise in downstream tasks, such as noisy edges and ambiguous node attributes. Existing AdapterGNNs are often prone to graph noise and exhibit limited generalizability. How to enhance the robustness and generalization ability of GNNs' fine tuning remains an open problem. In this paper, we show that the above problem can be well addressed by integrating uncertainty learning into the GNN adapter. We propose the Uncertainty-aware Adapter (UAdapterGNN) that fortifies pre-trained GNN models against noisy graph data in the fine-tuning process. Specifically, in contrast to regular AdapterGNN, our UAdapterGNN exploits Gaussian probabilistic adapter to augment the pre-trained GNN model. In this way, when the graph contains various noises,our method can automatically absorb the effects of changes in the variances of the Gaussian distribution, thereby significantly enhancing the model's robustness. Also, UAdapterGNN can further improve the generalization ability of the model on the downstream tasks. Extensive experiments on several benchmarks demonstrate the effectiveness, robustness and high generalization ability of the proposed UAdapterGNN method.
>
---
#### [new 352] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对统一多模态模型（UMM）在理解与生成间存在的表示不一致问题，提出UniGame框架。通过在共享令牌接口引入轻量扰动器，使生成分支主动挑战理解分支，实现自对抗训练。该方法提升了模型一致性、鲁棒性与跨模态连贯性，且对架构无依赖，参数增加少，效果显著。**

- **链接: [https://arxiv.org/pdf/2511.19413v1](https://arxiv.org/pdf/2511.19413v1)**

> **作者:** Zhaolong Su; Wang Lu; Hao Chen; Sharon Li; Jindong Wang
>
> **摘要:** Unified Multimodal Models (UMMs) have shown impressive performance in both understanding and generation with a single architecture. However, UMMs still exhibit a fundamental inconsistency: understanding favors compact embeddings, whereas generation favors reconstruction-rich representations. This structural trade-off produces misaligned decision boundaries, degraded cross-modal coherence, and heightened vulnerability under distributional and adversarial shifts. In this paper, we present UniGame, a self-adversarial post-training framework that directly targets the inconsistencies. By applying a lightweight perturber at the shared token interface, UniGame enables the generation branch to actively seek and challenge fragile understanding, turning the model itself into its own adversary. Experiments demonstrate that UniGame significantly improves the consistency (+4.6%). Moreover, it also achieves substantial improvements in understanding (+3.6%), generation (+0.02), out-of-distribution and adversarial robustness (+4.8% and +6.2% on NaturalBench and AdVQA). The framework is architecture-agnostic, introduces less than 1% additional parameters, and is complementary to existing post-training methods. These results position adversarial self-play as a general and effective principle for enhancing the coherence, stability, and unified competence of future multimodal foundation models. The official code is available at: https://github.com/AIFrontierLab/UniGame
>
---
#### [new 353] Saving Foundation Flow-Matching Priors for Inverse Problems
- **分类: cs.LG; cs.CV; eess.IV; eess.SP**

- **简介: 该论文针对基础流匹配模型在逆问题中性能不足的问题，提出FMPlug框架。通过实例引导的时变预热策略与高斯性正则化，增强模型对逆问题的适应性，有效提升图像恢复与科学逆问题的求解性能，推动基础流匹配模型成为可复用的通用先验。**

- **链接: [https://arxiv.org/pdf/2511.16520v1](https://arxiv.org/pdf/2511.16520v1)**

> **作者:** Yuxiang Wan; Ryan Devera; Wenjie Zhang; Ju Sun
>
> **摘要:** Foundation flow-matching (FM) models promise a universal prior for solving inverse problems (IPs), yet today they trail behind domain-specific or even untrained priors. How can we unlock their potential? We introduce FMPlug, a plug-in framework that redefines how foundation FMs are used in IPs. FMPlug combines an instance-guided, time-dependent warm-start strategy with a sharp Gaussianity regularization, adding problem-specific guidance while preserving the Gaussian structures. This leads to a significant performance boost across image restoration and scientific IPs. Our results point to a path for making foundation FM models practical, reusable priors for IP solving.
>
---
#### [new 354] Switch-JustDance: Benchmarking Whole Body Motion Tracking Policies Using a Commercial Console Game
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Switch-JustDance，一种基于任天堂Switch游戏《舞力全开》的低成本、可复现的人形机器人全身运动控制基准测试方法。针对现有评估缺乏真实场景对比与人类基准的问题，利用游戏实时评分系统量化机器人表现，验证了其可靠性与敏感性，并对三种先进控制器进行实机评测。**

- **链接: [https://arxiv.org/pdf/2511.17925v1](https://arxiv.org/pdf/2511.17925v1)**

> **作者:** Jeonghwan Kim; Wontaek Kim; Yidan Lu; Jin Cheng; Fatemeh Zargarbashi; Zicheng Zeng; Zekun Qi; Zhiyang Dou; Nitish Sontakke; Donghoon Baek; Sehoon Ha; Tianyu Li
>
> **摘要:** Recent advances in whole-body robot control have enabled humanoid and legged robots to perform increasingly agile and coordinated motions. However, standardized benchmarks for evaluating these capabilities in real-world settings, and in direct comparison to humans, remain scarce. Existing evaluations often rely on pre-collected human motion datasets or simulation-based experiments, which limit reproducibility, overlook hardware factors, and hinder fair human-robot comparisons. We present Switch-JustDance, a low-cost and reproducible benchmarking pipeline that leverages motion-sensing console games, Just Dance on the Nintendo Switch, to evaluate robot whole-body control. Using Just Dance on the Nintendo Switch as a representative platform, Switch-JustDance converts in-game choreography into robot-executable motions through streaming, motion reconstruction, and motion retargeting modules and enables users to evaluate controller performance through the game's built-in scoring system. We first validate the evaluation properties of Just Dance, analyzing its reliability, validity, sensitivity, and potential sources of bias. Our results show that the platform provides consistent and interpretable performance measures, making it a suitable tool for benchmarking embodied AI. Building on this foundation, we benchmark three state-of-the-art humanoid whole-body controllers on hardware and provide insights into their relative strengths and limitations.
>
---
#### [new 355] Deterministic Continuous Replacement: Fast and Stable Module Replacement in Pretrained Transformers
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对预训练Transformer中模块替换的稳定性问题，提出确定性连续替换（DCR）方法。旨在解决冷启动重初始化导致的模型不稳定问题，通过确定性渐变权重融合教师与学生输出，实现高效、稳定的注意力机制替换，显著提升收敛速度与对齐效果。**

- **链接: [https://arxiv.org/pdf/2511.18670v1](https://arxiv.org/pdf/2511.18670v1)**

> **作者:** Rowan Bradbury; Aniket Srinivasan Ashok; Sai Ram Kasanagottu; Gunmay Jhingran; Shuai Meng
>
> **备注:** Accepted to NeurIPS 2025 ScaleOPT Workshop; 8 pages; includes figures
>
> **摘要:** Replacing modules in pretrained models, especially swapping quadratic self-attention for efficient attention alternatives, poses a hard optimization problem: cold-start reinitialization destabilizes frozen backbones. We isolate this core stability challenge in a controlled study. Deterministic Continuous Replacement (DCR) blends teacher and student outputs with a deterministic, annealed weight. Theoretically, DCR eliminates gate-induced gradient variance inherent to stochastic replacement. In a single-seed study, DCR attains faster convergence and stronger alignment than stochastic gating and distillation baselines on controlled attention replacement, establishing a foundation for heterogeneous operator swaps.
>
---
#### [new 356] VLM in a flash: I/O-Efficient Sparsification of Vision-Language Model via Neuron Chunking
- **分类: cs.LG; cs.AI; cs.CV; cs.PF**

- **简介: 该论文针对边缘部署视觉语言模型时的I/O效率问题，提出Neuron Chunking方法。通过将神经元分块并结合重要性与访问延迟，优化权重卸载的存储访问模式，显著降低Flash存储读写开销，提升I/O效率。**

- **链接: [https://arxiv.org/pdf/2511.18692v1](https://arxiv.org/pdf/2511.18692v1)**

> **作者:** Kichang Yang; Seonjun Kim; Minjae Kim; Nairan Zhang; Chi Zhang; Youngki Lee
>
> **摘要:** Edge deployment of large Vision-Language Models (VLMs) increasingly relies on flash-based weight offloading, where activation sparsification is used to reduce I/O overhead. However, conventional sparsification remains model-centric, selecting neurons solely by activation magnitude and neglecting how access patterns influence flash performance. We present Neuron Chunking, an I/O-efficient sparsification strategy that operates on chunks (i.e., groups of contiguous neurons in memory) and couples neuron importance with storage access cost. The method models I/O latency through a lightweight abstraction of access contiguity and selects chunks with high utility, defined as neuron importance normalized by estimated latency. By aligning sparsification decisions with the underlying storage behavior, Neuron Chunking improves I/O efficiency by up to 4.65x and 5.76x on Jetson Orin Nano and Jetson AGX Orin, respectively.
>
---
#### [new 357] Self-Empowering VLMs: Achieving Hierarchical Consistency via Self-Elicited Knowledge Distillation
- **分类: cs.MM; cs.CV**

- **简介: 该论文针对视觉-语言模型在层次化理解任务中一致性差的问题，提出自诱导知识蒸馏（SEKD）方法。通过自生成多步推理监督信号，使单步学生模型获得跨层级状态感知能力，显著提升路径一致性与零样本泛化性能，无需人工标注即可扩展至新任务。**

- **链接: [https://arxiv.org/pdf/2511.18415v1](https://arxiv.org/pdf/2511.18415v1)**

> **作者:** Wei Yang; Yiran Zhu; Zilin Li; Xunjia Zhang; Hongtao Wang
>
> **备注:** 21 pages, 18 tables, 6 figures
>
> **摘要:** Vision-language models (VLMs) possess rich knowledge but often fail on hierarchical understanding tasks, where the goal is to predict a coarse-to-fine taxonomy path that remains consistent across all levels. We compare three inference paradigms for hierarchical VQA and find that stepwise reasoning, when conditioned on prior answers, significantly outperforms single-pass prompting. Further analysis indicates that the main limitation of current VLMs is their inability to maintain cross-level state, rather than a lack of taxonomic knowledge. Motivated by this diagnosis, we propose Self-Elicited Knowledge Distillation (SEKD), which requires no human labels or external tools: the same VLM is prompted to reason step by step and act as a teacher by exposing its hard labels, soft distributions, and decoder hidden states, while a single-pass student distills these signals. The student VLM remains efficient while approaching the accuracy of its multi-step teacher. It improves in-domain path consistency (HCA) by up to +29.50 percentage points, raises zero-shot HCA on an unseen taxonomy from 4.15% to 42.26%, and yields gains on challenging mathematical benchmarks. Because all supervision is self-elicited, SEKD scales to new taxonomies and datasets without annotation cost, providing a practical route to imbue compact VLMs with dependency-aware multi-step reasoning.
>
---
#### [new 358] Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文针对软连续体机器人动力学建模中数据驱动方法缺乏物理可解释性的难题，提出基于注意力广播解码器（ABCD）的自编码器框架。通过生成像素级注意力图并耦合2D振荡器网络，实现无需先验知识的动力学参数可视化与高精度多步预测，显著提升模型可解释性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18322v1](https://arxiv.org/pdf/2511.18322v1)**

> **作者:** Henrik Krauss; Johann Licher; Naoya Takeishi; Annika Raatz; Takehisa Yairi
>
> **摘要:** Data-driven learning of soft continuum robot (SCR) dynamics from high-dimensional observations offers flexibility but often lacks physical interpretability, while model-based approaches require prior knowledge and can be computationally expensive. We bridge this gap by introducing (1) the Attention Broadcast Decoder (ABCD), a plug-and-play module for autoencoder-based latent dynamics learning that generates pixel-accurate attention maps localizing each latent dimension's contribution while filtering static backgrounds. (2) By coupling these attention maps to 2D oscillator networks, we enable direct on-image visualization of learned dynamics (masses, stiffness, and forces) without prior knowledge. We validate our approach on single- and double-segment SCRs, demonstrating that ABCD-based models significantly improve multi-step prediction accuracy: 5.7x error reduction for Koopman operators and 3.5x for oscillator networks on the two-segment robot. The learned oscillator network autonomously discovers a chain structure of oscillators. Unlike standard methods, ABCD models enable smooth latent space extrapolation beyond training data. This fully data-driven approach yields compact, physically interpretable models suitable for control applications.
>
---
#### [new 359] Towards Generalizable Deepfake Detection via Forgery-aware Audio-Visual Adaptation: A Variational Bayesian Approach
- **分类: cs.MM; cs.CV**

- **简介: 该论文针对多模态深度伪造检测任务，解决现有方法泛化能力不足的问题。提出基于变分贝叶斯的伪造感知音视频自适应框架（FoVB），通过差异卷积与高通滤波提取伪造痕迹，利用变分贝叶斯估计音视频相关性的高斯潜变量，并施加正交约束分离模态内与跨模态特征，有效提升检测性能。**

- **链接: [https://arxiv.org/pdf/2511.19080v1](https://arxiv.org/pdf/2511.19080v1)**

> **作者:** Fan Nie; Jiangqun Ni; Jian Zhang; Bin Zhang; Weizhe Zhang; Bin Li
>
> **备注:** TIFS AQE
>
> **摘要:** The widespread application of AIGC contents has brought not only unprecedented opportunities, but also potential security concerns, e.g., audio-visual deepfakes. Therefore, it is of great importance to develop an effective and generalizable method for multi-modal deepfake detection. Typically, the audio-visual correlation learning could expose subtle cross-modal inconsistencies, e.g., audio-visual misalignment, which serve as crucial clues in deepfake detection. In this paper, we reformulate the correlation learning with variational Bayesian estimation, where audio-visual correlation is approximated as a Gaussian distributed latent variable, and thus develop a novel framework for deepfake detection, i.e., Forgery-aware Audio-Visual Adaptation with Variational Bayes (FoVB). Specifically, given the prior knowledge of pre-trained backbones, we adopt two core designs to estimate audio-visual correlations effectively. First, we exploit various difference convolutions and a high-pass filter to discern local and global forgery traces from both modalities. Second, with the extracted forgery-aware features, we estimate the latent Gaussian variable of audio-visual correlation via variational Bayes. Then, we factorize the variable into modality-specific and correlation-specific ones with orthogonality constraint, allowing them to better learn intra-modal and cross-modal forgery traces with less entanglement. Extensive experiments demonstrate that our FoVB outperforms other state-of-the-art methods in various benchmarks.
>
---
#### [new 360] GContextFormer: A global context-aware hybrid multi-head attention approach with scaled additive aggregation for multimodal trajectory prediction
- **分类: cs.AI; cs.CV; cs.LG; cs.MA; cs.RO; cs.SI**

- **简介: 该论文针对无图多模态轨迹预测任务，解决地图依赖模型数据成本高及无图方法缺乏全局上下文导致意图错配的问题。提出GContextFormer，通过全局感知的混合注意力与缩放加性聚合，增强模式间意图对齐，实现更鲁棒、可解释的多路径预测。**

- **链接: [https://arxiv.org/pdf/2511.18874v1](https://arxiv.org/pdf/2511.18874v1)**

> **作者:** Yuzhi Chen; Yuanchang Xie; Lei Zhao; Pan Liu; Yajie Zou; Chen Wang
>
> **摘要:** Multimodal trajectory prediction generates multiple plausible future trajectories to address vehicle motion uncertainty from intention ambiguity and execution variability. However, HD map-dependent models suffer from costly data acquisition, delayed updates, and vulnerability to corrupted inputs, causing prediction failures. Map-free approaches lack global context, with pairwise attention over-amplifying straight patterns while suppressing transitional patterns, resulting in motion-intention misalignment. This paper proposes GContextFormer, a plug-and-play encoder-decoder architecture with global context-aware hybrid attention and scaled additive aggregation achieving intention-aligned multimodal prediction without map reliance. The Motion-Aware Encoder builds scene-level intention prior via bounded scaled additive aggregation over mode-embedded trajectory tokens and refines per-mode representations under shared global context, mitigating inter-mode suppression and promoting intention alignment. The Hierarchical Interaction Decoder decomposes social reasoning into dual-pathway cross-attention: a standard pathway ensures uniform geometric coverage over agent-mode pairs while a neighbor-context-enhanced pathway emphasizes salient interactions, with gating module mediating their contributions to maintain coverage-focus balance. Experiments on eight highway-ramp scenarios from TOD-VT dataset show GContextFormer outperforms state-of-the-art baselines. Compared to existing transformer models, GContextFormer achieves greater robustness and concentrated improvements in high-curvature and transition zones via spatial distributions. Interpretability is achieved through motion mode distinctions and neighbor context modulation exposing reasoning attribution. The modular architecture supports extensibility toward cross-domain multimodal reasoning tasks. Source: https://fenghy-chen.github.io/sources/.
>
---
#### [new 361] TRIDENT: A Trimodal Cascade Generative Framework for Drug and RNA-Conditioned Cellular Morphology Synthesis
- **分类: cs.LG; cs.CV; q-bio.QM**

- **简介: 该论文提出TRIDENT框架，解决药物/RNA扰动到细胞形态的因果建模问题。针对现有方法忽略转录组到表型的映射，提出基于扰动与基因表达联合条件的生成式级联模型，并构建MorphoGene数据集。实验表明其显著提升形态合成精度，验证了RNA条件对高保真度的关键作用。**

- **链接: [https://arxiv.org/pdf/2511.18287v1](https://arxiv.org/pdf/2511.18287v1)**

> **作者:** Rui Peng; Ziru Liu; Lingyuan Ye; Yuxing Lu; Boxin Shi; Jinzhuo Wang
>
> **摘要:** Accurately modeling the relationship between perturbations, transcriptional responses, and phenotypic changes is essential for building an AI Virtual Cell (AIVC). However, existing methods typically constrained to modeling direct associations, such as Perturbation $\rightarrow$ RNA or Perturbation $\rightarrow$ Morphology, overlook the crucial causal link from RNA to morphology. To bridge this gap, we propose TRIDENT, a cascade generative framework that synthesizes realistic cellular morphology by conditioning on both the perturbation and the corresponding gene expression profile. To train and evaluate this task, we construct MorphoGene, a new dataset pairing L1000 gene expression with Cell Painting images for 98 compounds. TRIDENT significantly outperforms state-of-the-art approaches, achieving up to 7-fold improvement with strong generalization to unseen compounds. In a case study on docetaxel, we validate that RNA-guided synthesis accurately produces the corresponding phenotype. An ablation study further confirms that this RNA conditioning is essential for the model's high fidelity. By explicitly modeling transcriptome-phenome mapping, TRIDENT provides a powerful in silico tool and moves us closer to a predictive virtual cell.
>
---
#### [new 362] CubeletWorld: A New Abstraction for Scalable 3D Modeling
- **分类: cs.LG; cs.CV; cs.CY**

- **简介: 该论文提出CubeletWorld，一种基于3D立方体单元的都市环境抽象框架，用于解决多源异构城市数据集成难、隐私保护与可扩展性差的问题。通过将城市划分为离散立方体单元，实现对基础设施、人流等数据的隐私保护建模，支持规划、预测等任务，克服了传统方法对代理感知的依赖，提升了模型泛化能力与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.17664v1](https://arxiv.org/pdf/2511.17664v1)**

> **作者:** Azlaan Mustafa Samad; Hoang H. Nguyen; Lukas Berg; Henrik Müller; Yuan Xue; Daniel Kudenko; Zahra Ahmadi
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Modern cities produce vast streams of heterogeneous data, from infrastructure maps to mobility logs and satellite imagery. However, integrating these sources into coherent spatial models for planning and prediction remains a major challenge. Existing agent-centric methods often rely on direct environmental sensing, limiting scalability and raising privacy concerns. This paper introduces CubeletWorld, a novel framework for representing and analyzing urban environments through a discretized 3D grid of spatial units called cubelets. This abstraction enables privacy-preserving modeling by embedding diverse data signals, such as infrastructure, movement, or environmental indicators, into localized cubelet states. CubeletWorld supports downstream tasks such as planning, navigation, and occupancy prediction without requiring agent-driven sensing. To evaluate this paradigm, we propose the CubeletWorld State Prediction task, which involves predicting the cubelet state using a realistic dataset containing various urban elements like streets and buildings through this discretized representation. We explore a range of modified core models suitable for our setting and analyze challenges posed by increasing spatial granularity, specifically the issue of sparsity in representation and scalability of baselines. In contrast to existing 3D occupancy prediction models, our cubelet-centric approach focuses on inferring state at the spatial unit level, enabling greater generalizability across regions and improved privacy compliance. Our results demonstrate that CubeletWorld offers a flexible and extensible framework for learning from complex urban data, and it opens up new possibilities for scalable simulation and decision support in domains such as socio-demographic modeling, environmental monitoring, and emergency response. The code and datasets can be downloaded from here.
>
---
#### [new 363] AVERY: Adaptive VLM Split Computing through Embodied Self-Awareness for Efficient Disaster Response Systems
- **分类: cs.DC; cs.AR; cs.CV; cs.LG; cs.NI**

- **简介: 该论文针对灾难响应中无人机因资源受限无法部署视觉语言模型（VLM）的问题，提出AVERY框架。通过认知启发的双流分割与自适应计算，实现云端协同推理，在低带宽下提升效率与准确性，显著降低能耗，支持实时语义智能。**

- **链接: [https://arxiv.org/pdf/2511.18151v1](https://arxiv.org/pdf/2511.18151v1)**

> **作者:** Rajat Bhattacharjya; Sing-Yao Wu; Hyunwoo Oh; Chaewon Nam; Suyeon Koo; Mohsen Imani; Elaheh Bozorgzadeh; Nikil Dutt
>
> **备注:** 8 pages, 5 figures. Paper is currently under review. Authors' version posted for personal use and not for redistribution
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) in disaster response require complex, queryable intelligence that on-board CNNs cannot provide. While Vision-Language Models (VLMs) offer this semantic reasoning, their high resource demands make on-device deployment infeasible, and naive cloud offloading fails under the low-bandwidth networks common in disaster zones. We present AVERY, a framework that enables VLM deployment through adaptive split computing. We advance the split computing paradigm beyond traditional depth-wise partitioning by introducing a functional, cognitive-inspired dual-stream split that separates the VLM into a high-frequency, low-resolution "context stream" for real-time awareness and a low-frequency, high-fidelity "insight stream" for deep analysis. A lightweight, self-aware on-board controller manages this architecture, monitoring network conditions and operator intent to dynamically select from pre-trained compression models, navigating the fundamental accuracy-throughput trade-off. Evaluated using the VLM LISA-7B across an edge-cloud scenario under fluctuating network conditions, AVERY consistently outperforms static configurations, achieving 11.2% higher accuracy than raw image compression and 93.98% lower energy consumption compared to full-edge execution, thereby enhancing mission efficiency and enabling real-time, queryable intelligence on resource-constrained platforms in dynamic environments.
>
---
#### [new 364] EgoCogNav: Cognition-aware Human Egocentric Navigation
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出EgoCogNav框架，旨在解决人因感知在头戴式导航中被忽视的问题。通过融合场景与感官线索，联合预测轨迹与头部运动，并引入感知路径不确定性作为潜在状态。构建了CEN数据集，实验表明模型能有效捕捉人类导航中的扫描、犹豫等行为，在未见环境中具有良好泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.17581v1](https://arxiv.org/pdf/2511.17581v1)**

> **作者:** Zhiwen Qiu; Ziang Liu; Wenqian Niu; Tapomayukh Bhattacharjee; Saleh Kalantari
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Modeling the cognitive and experiential factors of human navigation is central to deepening our understanding of human-environment interaction and to enabling safe social navigation and effective assistive wayfinding. Most existing methods focus on forecasting motions in fully observed scenes and often neglect human factors that capture how people feel and respond to space. To address this gap, We propose EgoCogNav, a multimodal egocentric navigation framework that predicts perceived path uncertainty as a latent state and jointly forecasts trajectories and head motion by fusing scene features with sensory cues. To facilitate research in the field, we introduce the Cognition-aware Egocentric Navigation (CEN) dataset consisting 6 hours of real-world egocentric recordings capturing diverse navigation behaviors in real-world scenarios. Experiments show that EgoCogNav learns the perceived uncertainty that highly correlates with human-like behaviors such as scanning, hesitation, and backtracking while generalizing to unseen environments.
>
---
#### [new 365] CNN-Based Camera Pose Estimation and Localisation of Scan Images for Aircraft Visual Inspection
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对飞机外部视觉检测中的相机位姿估计与图像定位问题，提出一种无需基础设施、可现场部署的基于CNN的方法。通过域随机化生成合成数据并微调网络，结合飞机几何信息优化损失函数，实现高精度位姿估计（误差<0.24m，<2°），并设计完整扫描流程，适用于受限环境下的自动化检测。**

- **链接: [https://arxiv.org/pdf/2511.18702v1](https://arxiv.org/pdf/2511.18702v1)**

> **作者:** Xueyan Oh; Leonard Loh; Shaohui Foong; Zhong Bao Andy Koh; Kow Leong Ng; Poh Kang Tan; Pei Lin Pearlin Toh; U-Xuan Tan
>
> **备注:** 12 pages, 12 figures
>
> **摘要:** General Visual Inspection is a manual inspection process regularly used to detect and localise obvious damage on the exterior of commercial aircraft. There has been increasing demand to perform this process at the boarding gate to minimise the downtime of the aircraft and automating this process is desired to reduce the reliance on human labour. Automating this typically requires estimating a camera's pose with respect to the aircraft for initialisation but most existing localisation methods require infrastructure, which is very challenging in uncontrolled outdoor environments and within the limited turnover time (approximately 2 hours) on an airport tarmac. Additionally, many airlines and airports do not allow contact with the aircraft's surface or using UAVs for inspection between flights, and restrict access to commercial aircraft. Hence, this paper proposes an on-site method that is infrastructure-free and easy to deploy for estimating a pan-tilt-zoom camera's pose and localising scan images. This method initialises using the same pan-tilt-zoom camera used for the inspection task by utilising a Deep Convolutional Neural Network fine-tuned on only synthetic images to predict its own pose. We apply domain randomisation to generate the dataset for fine-tuning the network and modify its loss function by leveraging aircraft geometry to improve accuracy. We also propose a workflow for initialisation, scan path planning, and precise localisation of images captured from a pan-tilt-zoom camera. We evaluate and demonstrate our approach through experiments with real aircraft, achieving root-mean-square camera pose estimation errors of less than 0.24 m and 2 degrees for all real scenes.
>
---
#### [new 366] Compressor-VLA: Instruction-Guided Visual Token Compression for Efficient Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人操作中因冗余视觉令牌导致的计算开销问题，提出Compressor-VLA框架。通过指令引导的语义任务压缩与空间细节保留模块，实现高效、任务相关的视觉信息压缩，显著降低计算量并提升实时性，验证了其在仿真到真实场景中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.18950v1](https://arxiv.org/pdf/2511.18950v1)**

> **作者:** Juntao Gao; Feiyang Ye; Jing Zhang; Wenjing Qian
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a powerful paradigm in Embodied AI. However, the significant computational overhead of processing redundant visual tokens remains a critical bottleneck for real-time robotic deployment. While standard token pruning techniques can alleviate this, these task-agnostic methods struggle to preserve task-critical visual information. To address this challenge, simultaneously preserving both the holistic context and fine-grained details for precise action, we propose Compressor-VLA, a novel hybrid instruction-conditioned token compression framework designed for efficient, task-oriented compression of visual information in VLA models. The proposed Compressor-VLA framework consists of two token compression modules: a Semantic Task Compressor (STC) that distills holistic, task-relevant context, and a Spatial Refinement Compressor (SRC) that preserves fine-grained spatial details. This compression is dynamically modulated by the natural language instruction, allowing for the adaptive condensation of task-relevant visual information. Experimentally, extensive evaluations demonstrate that Compressor-VLA achieves a competitive success rate on the LIBERO benchmark while reducing FLOPs by 59% and the visual token count by over 3x compared to its baseline. The real-robot deployments on a dual-arm robot platform validate the model's sim-to-real transferability and practical applicability. Moreover, qualitative analyses reveal that our instruction guidance dynamically steers the model's perceptual focus toward task-relevant objects, thereby validating the effectiveness of our approach.
>
---
#### [new 367] Stable Multi-Drone GNSS Tracking System for Marine Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对海洋机器人在水下GNSS信号失效导致定位困难的问题，提出一种基于多无人机的稳定GNSS追踪系统。通过视觉检测、轻量级多目标跟踪与GNSS三角测量融合，结合信心加权EKF和跨无人机ID对齐算法，实现水面及近水面机器人实时、鲁棒的高精度定位，有效克服传统方法误差累积与依赖基础设施的缺陷。**

- **链接: [https://arxiv.org/pdf/2511.18694v1](https://arxiv.org/pdf/2511.18694v1)**

> **作者:** Shuo Wen; Edwin Meriaux; Mariana Sosa Guzmán; Zhizun Wang; Junming Shi; Gregory Dudek
>
> **摘要:** Accurate localization is essential for marine robotics, yet Global Navigation Satellite System (GNSS) signals are unreliable or unavailable even at a very short distance below the water surface. Traditional alternatives, such as inertial navigation, Doppler Velocity Loggers (DVL), SLAM, and acoustic methods, suffer from error accumulation, high computational demands, or infrastructure dependence. In this work, we present a scalable multi-drone GNSS-based tracking system for surface and near-surface marine robots. Our approach combines efficient visual detection, lightweight multi-object tracking, GNSS-based triangulation, and a confidence-weighted Extended Kalman Filter (EKF) to provide stable GNSS estimation in real time. We further introduce a cross-drone tracking ID alignment algorithm that enforces global consistency across views, enabling robust multi-robot tracking with redundant aerial coverage. We validate our system in diversified complex settings to show the scalability and robustness of the proposed algorithm.
>
---
#### [new 368] SYNAPSE: Synergizing an Adapter and Finetuning for High-Fidelity EEG Synthesis from a CLIP-Aligned Encoder
- **分类: eess.SP; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文提出SYNAPSE框架，解决高噪声、低分辨率的EEG信号生成高质量图像的问题。通过两阶段设计：先用CLIP对齐的自编码器学习语义结构化表征，再轻量适配Stable Diffusion实现高效条件生成，显著提升图像保真度与跨被试泛化能力，强调感知重建优于分类对齐。**

- **链接: [https://arxiv.org/pdf/2511.17547v1](https://arxiv.org/pdf/2511.17547v1)**

> **作者:** Jeyoung Lee; Hochul Kang
>
> **摘要:** Recent progress in diffusion-based generative models has enabled high-quality image synthesis conditioned on diverse modalities. Extending such models to brain signals could deepen our understanding of human perception and mental representations. However,electroencephalography (EEG) presents major challenges for image generation due to high noise, low spatial resolution, and strong inter-subject variability. Existing approaches,such as DreamDiffusion, BrainVis, and GWIT, primarily adapt EEG features to pre-trained Stable Diffusion models using complex alignment or classification pipelines, often resulting in large parameter counts and limited interpretability. We introduce SYNAPSE, a two-stage framework that bridges EEG signal representation learning and high-fidelity image synthesis. In Stage1, a CLIP-aligned EEG autoencoder learns a semantically structured latent representation by combining signal reconstruction and cross-modal alignment objectives. In Stage2, the pretrained encoder is frozen and integrated with a lightweight adaptation of Stable Diffusion, enabling efficient conditioning on EEG features with minimal trainable parameters. Our method achieves a semantically coherent latent space and state-of-the-art perceptual fidelity on the CVPR40 dataset, outperforming prior EEG-to-image models in both reconstruction efficiency and image quality. Quantitative and qualitative analyses demonstrate that SYNAPSE generalizes effectively across subjects, preserving visual semantics even when class-level agreement is reduced. These results suggest that reconstructing what the brain perceives, rather than what it classifies, is key to faithful EEG-based image generation.
>
---
#### [new 369] Spectral Super-Resolution Neural Operator with Atmospheric Radiative Transfer Prior
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对遥感中光谱超分辨率任务，解决数据驱动方法忽略物理规律导致光谱不真实的问题。提出SSRNO框架，融合大气辐射传输先验，通过三阶段流程实现物理一致的高光谱图像重建，具备连续谱重建与零样本外推能力，显著提升结果真实性与泛化性。**

- **链接: [https://arxiv.org/pdf/2511.17895v1](https://arxiv.org/pdf/2511.17895v1)**

> **作者:** Ziye Zhang; Bin Pan; Zhenwei Shi
>
> **摘要:** Spectral super-resolution (SSR) aims to reconstruct hyperspectral images (HSIs) from multispectral observations, with broad applications in remote sensing. Data-driven methods are widely used, but they often overlook physical principles, leading to unrealistic spectra, particularly in atmosphere-affected bands. To address this challenge, we propose the Spectral Super-Resolution Neural Operator (SSRNO), which incorporates atmospheric radiative transfer (ART) prior into the data-driven procedure, yielding more physically consistent predictions. The proposed SSRNO framework consists of three stages: upsampling, reconstruction, and refinement. In the upsampling stage, we leverage prior information to expand the input multispectral image, producing a physically plausible hyperspectral estimate. Subsequently, we utilize a neural operator in the reconstruction stage to learn a continuous mapping across the spectral domain. Finally, the refinement stage imposes a hard constraint on the output HSI to eliminate color distortion. The upsampling and refinement stages are implemented via the proposed guidance matrix projection (GMP) method, and the reconstruction neural operator adopts U-shaped spectral-aware convolution (SAC) layers to capture multi-scale features. Moreover, we theoretically demonstrate the optimality of the GMP method. With the neural operator and ART priors, SSRNO also achieves continuous spectral reconstruction and zero-shot extrapolation. Various experiments validate the effectiveness and generalization ability of the proposed approach.
>
---
#### [new 370] Shape-Adapting Gated Experts: Dynamic Expert Routing for Colonoscopic Lesion Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对病理图像中细胞形态与尺度多样带来的分割挑战，提出SAGE框架。通过动态专家路由机制，实现输入自适应的模型计算，提升对复杂视觉结构的建模能力。在多个医学图像分割数据集上达到领先性能，显著增强模型泛化性与灵活性。**

- **链接: [https://arxiv.org/pdf/2511.18493v1](https://arxiv.org/pdf/2511.18493v1)**

> **作者:** Gia Huy Thai; Hoang-Nguyen Vu; Anh-Minh Phan; Quang-Thinh Ly; Tram Dinh; Thi-Ngoc-Truc Nguyen; Nhat Ho
>
> **摘要:** The substantial diversity in cell scale and form remains a primary challenge in computer-aided cancer detection on gigapixel Whole Slide Images (WSIs), attributable to cellular heterogeneity. Existing CNN-Transformer hybrids rely on static computation graphs with fixed routing, which consequently causes redundant computation and limits their adaptability to input variability. We propose Shape-Adapting Gated Experts (SAGE), an input-adaptive framework that enables dynamic expert routing in heterogeneous visual networks. SAGE reconfigures static backbones into dynamically routed expert architectures. SAGE's dual-path design features a backbone stream that preserves representation and selectively activates an expert path through hierarchical gating. This gating mechanism operates at multiple hierarchical levels, performing a two-level, hierarchical selection between shared and specialized experts to modulate model logits for Top-K activation. Our Shape-Adapting Hub (SA-Hub) harmonizes structural and semantic representations across the CNN and the Transformer module, effectively bridging diverse modules. Embodied as SAGE-UNet, our model achieves superior segmentation on three medical benchmarks: EBHI, DigestPath, and GlaS, yielding state-of-the-art Dice Scores of 95.57%, 95.16%, and 94.17%, respectively, and robustly generalizes across domains by adaptively balancing local refinement and global context. SAGE provides a scalable foundation for dynamic expert routing, enabling flexible visual reasoning.
>
---
#### [new 371] MatMart: Material Reconstruction of 3D Objects via Diffusion
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出MatMart，一种基于扩散模型的3D物体材质重建框架。针对多视角材质估计中精度与泛化性不足的问题，采用两阶段重建与视图-材质交叉注意力机制，实现从任意数量输入图像中高保真重建材质，通过端到端优化单个扩散模型，提升稳定性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.18900v1](https://arxiv.org/pdf/2511.18900v1)**

> **作者:** Xiuchao Wu; Pengfei Zhu; Jiangjing Lyu; Xinguo Liu; Jie Guo; Yanwen Guo; Weiwei Xu; Chengfei Lyu
>
> **摘要:** Applying diffusion models to physically-based material estimation and generation has recently gained prominence. In this paper, we propose \ttt, a novel material reconstruction framework for 3D objects, offering the following advantages. First, \ttt\ adopts a two-stage reconstruction, starting with accurate material prediction from inputs and followed by prior-guided material generation for unobserved views, yielding high-fidelity results. Second, by utilizing progressive inference alongside the proposed view-material cross-attention (VMCA), \ttt\ enables reconstruction from an arbitrary number of input images, demonstrating strong scalability and flexibility. Finally, \ttt\ achieves both material prediction and generation capabilities through end-to-end optimization of a single diffusion model, without relying on additional pre-trained models, thereby exhibiting enhanced stability across various types of objects. Extensive experiments demonstrate that \ttt\ achieves superior performance in material reconstruction compared to existing methods.
>
---
#### [new 372] Mixture of Horizons in Action Chunking
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对视觉-语言-动作模型在机器人操作中因固定动作时域（horizon）导致的长期规划与精细控制权衡问题，提出混合时域（MoH）策略。通过并行处理多时域动作片段并动态融合，实现长程前瞻与短程精度兼顾，显著提升复杂任务性能与推理效率。**

- **链接: [https://arxiv.org/pdf/2511.19433v1](https://arxiv.org/pdf/2511.19433v1)**

> **作者:** Dong Jing; Gang Wang; Jiaqi Liu; Weiliang Tang; Zelong Sun; Yunchao Yao; Zhenyu Wei; Yunhui Liu; Zhiwu Lu; Mingyu Ding
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** Vision-language-action (VLA) models have shown remarkable capabilities in robotic manipulation, but their performance is sensitive to the $\textbf{action chunk length}$ used during training, termed $\textbf{horizon}$. Our empirical study reveals an inherent trade-off: longer horizons provide stronger global foresight but degrade fine-grained accuracy, while shorter ones sharpen local control yet struggle on long-term tasks, implying fixed choice of single horizons being suboptimal. To mitigate the trade-off, we propose a $\textbf{mixture of horizons (MoH)}$ strategy. MoH rearranges the action chunk into several segments with different horizons, processes them in parallel with a shared action transformer, and fuses outputs with a light linear gate. It has three appealing benefits. 1) MoH exploits long-term foresight and short-term precision jointly within a single model, improving both performance and generalizability to complex tasks. 2) MoH is plug-and-play for full-attention action modules with minimal training or inference overhead. 3) MoH enables dynamic inference with adaptive horizons, which selects stable actions through cross-horizon consensus, achieving 2.5$\times$ higher throughput than baselines while preserving superior performance. Extensive experiments over flow-based policies $π_0$, $π_{0.5}$, and one-step regression policy $π_{\text{reg}}$ demonstrate that MoH yields consistent and significant gains on both simulations and real-world tasks. Notably, under mixed-task setting, $π_{0.5}$ with MoH reaches a new state-of-the-art with 99$\%$ average success rate on LIBERO after only $30k$ training iterations. Project page: https://github.com/Timsty1/MixtureOfHorizons
>
---
#### [new 373] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在动态决策中忽视历史上下文的问题，提出AVA-VLA框架。通过引入基于信念状态的主动视觉注意力机制，利用递归状态动态聚焦关键视觉信息，将任务从MDP重构为POMDP，显著提升模型在机器人任务中的表现与真实世界迁移能力。**

- **链接: [https://arxiv.org/pdf/2511.18960v1](https://arxiv.org/pdf/2511.18960v1)**

> **作者:** Lei Xiao; Jifeng Li; Juntao Gao; Feiyang Ye; Yan Jin; Jingjing Qian; Jing Zhang; Yong Wu; Xiaoyuan Yu
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in embodied AI tasks. However, existing VLA models, often built upon Vision-Language Models (VLMs), typically process dense visual inputs independently at each timestep. This approach implicitly models the task as a Markov Decision Process (MDP). However, this history-agnostic design is suboptimal for effective visual token processing in dynamic sequential decision-making, as it fails to leverage the context of history. To address this limitation, we reformulate the problem from a Partially Observable Markov Decision Process (POMDP) perspective and propose a novel framework named AVA-VLA. Inspired by the POMDP that the action generation should be conditioned on the belief state. AVA-VLA introduces Active Visual Attention (AVA) to dynamically modulate visual processing. It achieves this by leveraging the recurrent state, which is a neural approximation of the agent's belief state derived from the previous decision step. Specifically, the AVA module uses the recurrent state to compute the soft weights to actively process task-relevant visual tokens based on its historical context. Comprehensive evaluations demonstrate that AVA-VLA achieves state-of-the-art performance across popular robotic benchmarks, including LIBERO and CALVIN. Furthermore, real-world deployments on a dual-arm robot platform validate the framework's practical applicability and robust sim-to-real transferability.
>
---
#### [new 374] ChronoGS: Disentangling Invariants and Changes in Multi-Period Scenes
- **分类: cs.GR; cs.CV**

- **简介: 该论文针对多时期场景重建任务，解决长期、非连续变化下几何与外观演化导致的重建不一致问题。提出ChronoGS模型，通过时序调制高斯表示，在统一框架内分离稳定与变化成分，实现时序一致的多期重建，并发布ChronoScene数据集推动研究。**

- **链接: [https://arxiv.org/pdf/2511.18794v1](https://arxiv.org/pdf/2511.18794v1)**

> **作者:** Zhongtao Wang; Jiaqi Dai; Qingtian Zhu; Yilong Li; Mai Su; Fei Zhu; Meng Gai; Shaorong Wang; Chengwei Pan; Yisong Chen; Guoping Wang
>
> **摘要:** Multi-period image collections are common in real-world applications. Cities are re-scanned for mapping, construction sites are revisited for progress tracking, and natural regions are monitored for environmental change. Such data form multi-period scenes, where geometry and appearance evolve. Reconstructing such scenes is an important yet underexplored problem. Existing pipelines rely on incompatible assumptions: static and in-the-wild methods enforce a single geometry, while dynamic ones assume smooth motion, both failing under long-term, discontinuous changes. To solve this problem, we introduce ChronoGS, a temporally modulated Gaussian representation that reconstructs all periods within a unified anchor scaffold. It's also designed to disentangle stable and evolving components, achieving temporally consistent reconstruction of multi-period scenes. To catalyze relevant research, we release ChronoScene dataset, a benchmark of real and synthetic multi-period scenes, capturing geometric and appearance variation. Experiments demonstrate that ChronoGS consistently outperforms baselines in reconstruction quality and temporal consistency. Our code and the ChronoScene dataset are publicly available at https://github.com/ZhongtaoWang/ChronoGS.
>
---
#### [new 375] From Tables to Signals: Revealing Spectral Adaptivity in TabPFN
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究表格式基础模型TabPFN的内在机制，揭示其在上下文学习中的频谱自适应性。通过信号重建视角，发现其频率容量优于标准MLP，且能随样本数动态调整。证明位置编码影响其频响，并实现无需训练的图像去噪，为任务无关隐式建模提供新思路。**

- **链接: [https://arxiv.org/pdf/2511.18278v1](https://arxiv.org/pdf/2511.18278v1)**

> **作者:** Jianqiao Zheng; Cameron Gordon; Yiping Ji; Hemanth Saratchandran; Simon Lucey
>
> **摘要:** Task-agnostic tabular foundation models such as TabPFN have achieved impressive performance on tabular learning tasks, yet the origins of their inductive biases remain poorly understood. In this work, we study TabPFN through the lens of signal reconstruction and provide the first frequency-based analysis of its in-context learning behavior. We show that TabPFN possesses a broader effective frequency capacity than standard ReLU-MLPs, even without hyperparameter tuning. Moreover, unlike MLPs whose spectra evolve primarily over training epochs, we find that TabPFN's spectral capacity adapts directly to the number of samples provided in-context, a phenomenon we term Spectral Adaptivity. We further demonstrate that positional encoding modulates TabPFN's frequency response, mirroring classical results in implicit neural representations. Finally, we show that these properties enable TabPFN to perform training-free and hyperparameter-free image denoising, illustrating its potential as a task-agnostic implicit model. Our analysis provides new insight into the structure and inductive biases of tabular foundation models and highlights their promise for broader signal reconstruction tasks.
>
---
#### [new 376] DeepCoT: Deep Continual Transformers for Real-Time Inference on Data Streams
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文针对流数据实时推理中计算冗余问题，提出DeepCoT——一种适用于深层Transformer的持续学习架构。通过在编码器中消除冗余计算，实现线性时间复杂度，在音频、视频、文本流上保持高性能的同时，将运行时间降低两个数量级。**

- **链接: [https://arxiv.org/pdf/2511.17693v1](https://arxiv.org/pdf/2511.17693v1)**

> **作者:** Ginés Carreto Picón; Peng Yuan Zhou; Qi Zhang; Alexandros Iosifidis
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Transformer-based models have dramatically increased their size and parameter count to tackle increasingly complex tasks. At the same time, there is a growing demand for low-latency inference on resource-constrained devices that achieves high performance. In particular, stream data inference is typically performed over a sliding temporal window, leading to highly redundant computations. The recent Continual Transformers have addressed this issue, but they can only be effectively used in shallow models, which limits their scope and generalization power. In this paper, we propose the Deep Continual Transformer (DeepCoT), a redundancy-free encoder-only model that can be applied over existing deep encoder architectures with minimal changes. In our experiments over audio, video, and text streams, we show that DeepCoTs retain comparative performance to their non-continual baselines while offering a linear computational cost for all Transformer layers, which reduces up to two orders of magnitude in the running time compared to previous efficient models.
>
---
#### [new 377] TimePre: Bridging Accuracy, Efficiency, and Stability in Probabilistic Time-Series Forecasting
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对概率时间序列预测中生成模型效率低、非采样框架训练不稳的问题，提出TimePre框架。通过引入Stabilized Instance Normalization（SIN）层，有效解决MLP骨干与MCL范式结合时的统计偏移与假设崩溃问题，实现了高精度、高速度与稳定性的统一。**

- **链接: [https://arxiv.org/pdf/2511.18539v1](https://arxiv.org/pdf/2511.18539v1)**

> **作者:** Lingyu Jiang; Lingyu Xu; Peiran Li; Qianwen Ge; Dingyi Zhuang; Shuo Xing; Wenjing Chen; Xiangbo Gao; Ting-Hsuan Chen; Xueying Zhan; Xin Zhang; Ziming Zhang; Zhengzhong Tu; Michael Zielewski; Kazunori Yamada; Fangzhou Lin
>
> **备注:** 15 pages, 5 figures, 6 tables
>
> **摘要:** Probabilistic Time-Series Forecasting (PTSF) is critical for uncertainty-aware decision making, but existing generative models, such as diffusion-based approaches, are computationally prohibitive due to expensive iterative sampling. Non-sampling frameworks like Multiple Choice Learning (MCL) offer an efficient alternative, but suffer from severe training instability and hypothesis collapse, which has historically hindered their performance. This problem is dramatically exacerbated when attempting to combine them with modern, efficient MLP-based backbones. To resolve this fundamental incompatibility, we propose TimePre, a novel framework that successfully unifies the efficiency of MLP-based models with the distributional flexibility of the MCL paradigm. The core of our solution is Stabilized Instance Normalization (SIN), a novel normalization layer that explicitly remedies this incompatibility. SIN stabilizes the hybrid architecture by correcting channel-wise statistical shifts, definitively resolving the catastrophic hypothesis collapse. Extensive experiments on six benchmark datasets demonstrate that TimePre achieves new state-of-the-art accuracy on key probabilistic metrics. Critically, TimePre achieves inference speeds orders of magnitude faster than sampling-based models and, unlike prior MCL work, demonstrates stable performance scaling. It thus bridges the long-standing gap between accuracy, efficiency, and stability in probabilistic forecasting.
>
---
#### [new 378] Real-Time Object Tracking with On-Device Deep Learning for Adaptive Beamforming in Dynamic Acoustic Environments
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文针对动态声学环境中声源定位与定向拾音难题，提出一种基于嵌入式深度学习的实时目标跟踪系统。通过单目深度估计与双目视觉实现3D目标定位，结合微型麦克风阵列实现2D波束成形动态调整，实时同步声学聚焦与目标位置，显著提升信干比，适用于智能会议、智能家居等场景。**

- **链接: [https://arxiv.org/pdf/2511.19396v1](https://arxiv.org/pdf/2511.19396v1)**

> **作者:** Jorge Ortigoso-Narro; Jose A. Belloch; Adrian Amor-Martin; Sandra Roger; Maximo Cobos
>
> **摘要:** Advances in object tracking and acoustic beamforming are driving new capabilities in surveillance, human-computer interaction, and robotics. This work presents an embedded system that integrates deep learning-based tracking with beamforming to achieve precise sound source localization and directional audio capture in dynamic environments. The approach combines single-camera depth estimation and stereo vision to enable accurate 3D localization of moving objects. A planar concentric circular microphone array constructed with MEMS microphones provides a compact, energy-efficient platform supporting 2D beam steering across azimuth and elevation. Real-time tracking outputs continuously adapt the array's focus, synchronizing the acoustic response with the target's position. By uniting learned spatial awareness with dynamic steering, the system maintains robust performance in the presence of multiple or moving sources. Experimental evaluation demonstrates significant gains in signal-to-interference ratio, making the design well-suited for teleconferencing, smart home devices, and assistive technologies.
>
---
## 更新

#### [replaced 001] Training-Free Efficient Video Generation via Dynamic Token Carving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.16864v2](https://arxiv.org/pdf/2505.16864v2)**

> **作者:** Yuechen Zhang; Jinbo Xing; Bin Xia; Shaoteng Liu; Bohao Peng; Xin Tao; Pengfei Wan; Eric Lo; Jiaya Jia
>
> **备注:** NeurIPS 2025, Project Page: https://julianjuaner.github.io/projects/jenga/
>
> **摘要:** Despite the remarkable generation quality of video Diffusion Transformer (DiT) models, their practical deployment is severely hindered by extensive computational requirements. This inefficiency stems from two key challenges: the quadratic complexity of self-attention with respect to token length and the multi-step nature of diffusion models. To address these limitations, we present Jenga, a novel inference pipeline that combines dynamic attention carving with progressive resolution generation. Our approach leverages two key insights: (1) early denoising steps do not require high-resolution latents, and (2) later steps do not require dense attention. Jenga introduces a block-wise attention mechanism that dynamically selects relevant token interactions using 3D space-filling curves, alongside a progressive resolution strategy that gradually increases latent resolution during generation. Experimental results demonstrate that Jenga achieves substantial speedups across multiple state-of-the-art video diffusion models while maintaining comparable generation quality (8.83$\times$ speedup with 0.01\% performance drop on VBench). As a plug-and-play solution, Jenga enables practical, high-quality video generation on modern hardware by reducing inference time from minutes to seconds -- without requiring model retraining. Code: https://github.com/dvlab-research/Jenga
>
---
#### [replaced 002] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations
- **分类: cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.17265v4](https://arxiv.org/pdf/2411.17265v4)**

> **作者:** Lehan He; Zeren Chen; Zhelun Shi; Tianyu Yu; Jing Shao; Lu Sheng
>
> **备注:** 34 pages, 12 figures, Accepted by NeurIPS 2025
>
> **摘要:** The success of Direct Preference Optimization (DPO) in mitigating hallucinations in Vision Language Models (VLMs) critically hinges on the true reward gaps within preference pairs. However, current methods, typically relying on ranking or rewriting strategies, often struggle to optimize these reward gaps in a systematic way during data curation. A core difficulty lies in precisely characterizing and strategically manipulating the overall reward gap configuration, that is, the deliberate design of how to shape these reward gaps within each preference pair across the data. To address this, we introduce Topic-level Preference Rewriting(TPR), a novel framework designed for the systematic optimization of reward gap configuration. Through selectively replacing semantic topics within VLM responses with model's own resampled candidates for targeted rewriting, TPR can provide topic-level control over fine-grained semantic details. This precise control enables advanced data curation strategies, such as progressively adjusting the difficulty of rejected responses, thereby sculpting an effective reward gap configuration that guides the model to overcome challenging hallucinations. Comprehensive experiments demonstrate TPR achieves state-of-the-art performance on multiple hallucination benchmarks, outperforming previous methods by an average of 20%. Notably, it significantly reduces hallucinations by up to 93% on ObjectHal-Bench, and also exhibits superior data efficiency towards robust and cost-effective VLM alignment. Code and datasets are available at https://tpr-dpo.github.io .
>
---
#### [replaced 003] When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.16203v2](https://arxiv.org/pdf/2511.16203v2)**

> **作者:** Yuping Yan; Yuhan Xie; Yixin Zhang; Lingjuan Lyu; Handing Wang; Yaochu Jin
>
> **摘要:** Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.
>
---
#### [replaced 004] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06741v4](https://arxiv.org/pdf/2511.06741v4)**

> **作者:** Wenbo Huang; Jinghui Zhang; Zhenghao Chen; Guang Li; Lei Zhang; Yang Cao; Fang Dong; Takahiro Ogawa; Miki Haseyama
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Wide-angle videos in few-shot action recognition (FSAR) effectively express actions within specific scenarios. However, without a global understanding of both subjects and background, recognizing actions in such samples remains challenging because of the background distractions. Receptance Weighted Key Value (RWKV), which learns interaction between various dimensions, shows promise for global modeling. While directly applying RWKV to wide-angle FSAR may fail to highlight subjects due to excessive background information. Additionally, temporal relation degraded by frames with similar backgrounds is difficult to reconstruct, further impacting performance. Therefore, we design the CompOund SegmenTation and Temporal REconstructing RWKV (Otter). Specifically, the Compound Segmentation Module~(CSM) is devised to segment and emphasize key patches in each frame, effectively highlighting subjects against background information. The Temporal Reconstruction Module (TRM) is incorporated into the temporal-enhanced prototype construction to enable bidirectional scanning, allowing better reconstruct temporal relation. Furthermore, a regular prototype is combined with the temporal-enhanced prototype to simultaneously enhance subject emphasis and temporal modeling, improving wide-angle FSAR performance. Extensive experiments on benchmarks such as SSv2, Kinetics, UCF101, and HMDB51 demonstrate that Otter achieves state-of-the-art performance. Extra evaluation on the VideoBadminton dataset further validates the superiority of Otter in wide-angle FSAR.
>
---
#### [replaced 005] The Geometry of Cortical Computation: Manifold Disentanglement and Predictive Dynamics in VCNet
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.02995v3](https://arxiv.org/pdf/2508.02995v3)**

> **作者:** Brennen A. Hill; Zhang Xinyu; Timothy Putra Prasetio
>
> **备注:** Published in the proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Symmetry and Geometry in Neural Representations (NeurReps). Additionally accepted for presentation in NeurIPS 2025 Workshop: Interpreting Cognition in Deep Learning Models (CogInterp)
>
> **摘要:** Despite their success, modern convolutional neural networks (CNNs) exhibit fundamental limitations, including data inefficiency, poor out-of-distribution generalization, and vulnerability to adversarial perturbations. These shortcomings can be traced to a lack of inductive biases that reflect the inherent geometric structure of the visual world. The primate visual system, in contrast, demonstrates superior efficiency and robustness, suggesting that its architectural and computational principles,which evolved to internalize these structures,may offer a blueprint for more capable artificial vision. This paper introduces Visual Cortex Network (VCNet), a novel neural network architecture whose design is informed by the macro-scale organization of the primate visual cortex. VCNet is framed as a geometric framework that emulates key biological mechanisms, including hierarchical processing across distinct cortical areas, dual-stream information segregation for learning disentangled representations, and top-down predictive feedback for representation refinement. We interpret these mechanisms through the lens of geometry and dynamical systems, positing that they guide the learning of structured, low-dimensional neural manifolds. We evaluate VCNet on two specialized benchmarks: the Spots-10 animal pattern dataset, which probes sensitivity to natural textures, and a light field image classification task, which requires processing higher-dimensional visual data. Our results show that VCNet achieves state-of-the-art accuracy of 92.1\% on Spots-10 and 74.4\% on the light field dataset, surpassing contemporary models of comparable size. This work demonstrates that integrating high-level neuroscientific principles, viewed through a geometric lens, can lead to more efficient and robust models, providing a promising direction for addressing long-standing challenges in machine learning.
>
---
#### [replaced 006] MeteorPred: A Meteorological Multimodal Large Model and Dataset for Severe Weather Event Prediction
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06859v2](https://arxiv.org/pdf/2508.06859v2)**

> **作者:** Shuo Tang; Jian Xu; Jiadong Zhang; Yi Chen; Qizhao Jin; Lingdong Shen; Chenglin Liu; Shiming Xiang
>
> **摘要:** Timely and accurate forecasts of severe weather events are essential for early warning and for constraining downstream analysis and decision-making. Since severe weather events prediction still depends on subjective, time-consuming expert interpretation, end-to-end "AI weather station" systems are emerging but face three major challenges: (1) scarcity of severe weather event samples; (2) imperfect alignment between high-dimensional meteorological data and textual warnings; (3) current multimodal language models cannot effectively process high-dimensional meteorological inputs or capture their complex spatiotemporal dependencies. To address these challenges, we introduce MP-Bench, the first large-scale multimodal dataset for severe weather events prediction, comprising 421,363 pairs of raw multi-year meteorological data and corresponding text caption, covering a wide range of severe weather scenarios. On top of this dataset, we develop a Meteorology Multimodal Large Model (MMLM) that directly ingests 4D meteorological inputs. In addition, it is designed to accommodate the unique characteristics of 4D meteorological data flow, incorporating three plug-and-play adaptive fusion modules that enable dynamic feature extraction and integration across temporal sequences, vertical pressure layers, and spatial dimensions. Extensive experiments on MP-Bench show that MMLM achieves strong performance across multiple tasks, demonstrating effective severe weather understanding and representing a key step toward automated, AI-driven severe weather events forecasting systems. Our source code and dataset will be made publicly available.
>
---
#### [replaced 007] Fairness in Multi-modal Medical Diagnosis with Demonstration Selection
- **分类: cs.CV; cs.CY; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.15986v2](https://arxiv.org/pdf/2511.15986v2)**

> **作者:** Dawei Li; Zijian Gu; Peng Wang; Chuhan Song; Zhen Tan; Mohan Zhang; Tianlong Chen; Yu Tian; Song Wang
>
> **备注:** 10 pages (including 2 pages of references), 4 figures. This work explores fairness in multi-modal medical image reasoning using in-context learning
>
> **摘要:** Multimodal large language models (MLLMs) have shown strong potential for medical image reasoning, yet fairness across demographic groups remains a major concern. Existing debiasing methods often rely on large labeled datasets or fine-tuning, which are impractical for foundation-scale models. We explore In-Context Learning (ICL) as a lightweight, tuning-free alternative for improving fairness. Through systematic analysis, we find that conventional demonstration selection (DS) strategies fail to ensure fairness due to demographic imbalance in selected exemplars. To address this, we propose Fairness-Aware Demonstration Selection (FADS), which builds demographically balanced and semantically relevant demonstrations via clustering-based sampling. Experiments on multiple medical imaging benchmarks show that FADS consistently reduces gender-, race-, and ethnicity-related disparities while maintaining strong accuracy, offering an efficient and scalable path toward fair medical image reasoning. These results highlight the potential of fairness-aware in-context learning as a scalable and data-efficient solution for equitable medical image reasoning.
>
---
#### [replaced 008] Fine-Grained GRPO for Precise Preference Alignment in Flow Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.01982v3](https://arxiv.org/pdf/2510.01982v3)**

> **作者:** Yujie Zhou; Pengyang Ling; Jiazi Bu; Yibin Wang; Yuhang Zang; Jiaqi Wang; Li Niu; Guangtao Zhai
>
> **备注:** Project Page: https://bujiazi.github.io/g2rpo.github.io/
>
> **摘要:** The incorporation of online reinforcement learning (RL) into diffusion and flow-based generative models has recently gained attention as a powerful paradigm for aligning model behavior with human preferences. By leveraging stochastic sampling via Stochastic Differential Equations (SDEs) during the denoising phase, these models can explore a variety of denoising trajectories, enhancing the exploratory capacity of RL. However, despite their ability to discover potentially high-reward samples, current approaches often struggle to effectively align with preferences due to the sparsity and narrowness of reward feedback. To overcome this limitation, we introduce a novel framework called Granular-GRPO (G$^2$RPO), which enables fine-grained and comprehensive evaluation of sampling directions in the RL training of flow models. Specifically, we propose a Singular Stochastic Sampling mechanism that supports step-wise stochastic exploration while ensuring strong correlation between injected noise and reward signals, enabling more accurate credit assignment to each SDE perturbation. Additionally, to mitigate the bias introduced by fixed-granularity denoising, we design a Multi-Granularity Advantage Integration module that aggregates advantages computed across multiple diffusion scales, resulting in a more robust and holistic assessment of sampling trajectories. Extensive experiments on various reward models, including both in-domain and out-of-domain settings, demonstrate that our G$^2$RPO outperforms existing flow-based GRPO baselines, highlighting its effectiveness and generalization capability.
>
---
#### [replaced 009] HOSIG: Full-Body Human-Object-Scene Interaction Generation with Hierarchical Scene Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.01579v2](https://arxiv.org/pdf/2506.01579v2)**

> **作者:** Wei Yao; Yunlian Sun; Hongwen Zhang; Yebin Liu; Jinhui Tang
>
> **摘要:** Generating high-fidelity full-body human interactions with dynamic objects and static scenes remains a critical challenge in computer graphics and animation. Existing methods for human-object interaction often neglect scene context, leading to implausible penetrations, while human-scene interaction approaches struggle to coordinate fine-grained manipulations with long-range navigation. To address these limitations, we propose HOSIG, a novel framework for synthesizing full-body interactions through hierarchical scene perception. Our method decouples the task into three key components: 1) a scene-aware grasp pose generator that ensures collision-free whole-body postures with precise hand-object contact by integrating local geometry constraints, 2) a heuristic navigation algorithm that autonomously plans obstacle-avoiding paths in complex indoor environments via compressed 2D floor maps and dual-component spatial reasoning, and 3) a scene-guided motion diffusion model that generates trajectory-controlled, full-body motions with finger-level accuracy by incorporating spatial anchors and dual-space classifier-free guidance. Extensive experiments on the TRUMANS dataset demonstrate superior performance over state-of-the-art methods. Notably, our framework supports unlimited motion length through autoregressive generation and requires minimal manual intervention. This work bridges the critical gap between scene-aware navigation and dexterous object manipulation, advancing the frontier of embodied interaction synthesis. Codes will be available after publication. Project page: http://yw0208.github.io/hosig
>
---
#### [replaced 010] Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2411.13093v4](https://arxiv.org/pdf/2411.13093v4)**

> **作者:** Yongdong Luo; Xiawu Zheng; Guilin Li; Shukang Yin; Haojia Lin; Chaoyou Fu; Jinfa Huang; Jiayi Ji; Fei Chao; Jiebo Luo; Rongrong Ji
>
> **备注:** Accepted at NeurIPS 2025. Camera-ready version
>
> **摘要:** Existing large video-language models (LVLMs) struggle to comprehend long videos correctly due to limited context. To address this problem, fine-tuning long-context LVLMs and employing GPT-based agents have emerged as promising solutions. However, fine-tuning LVLMs would require extensive high-quality data and substantial GPU resources, while GPT-based agents would rely on proprietary models (e.g., GPT-4o). In this paper, we propose Video Retrieval-Augmented Generation (Video-RAG), a training-free and cost-effective pipeline that employs visually-aligned auxiliary texts to help facilitate cross-modality alignment while providing additional information beyond the visual content. Specifically, we leverage open-source external tools to extract visually-aligned information from pure video data (e.g., audio, optical character, and object detection), and incorporate the extracted information into an existing LVLM as auxiliary texts, alongside video frames and queries, in a plug-and-play manner. Our Video-RAG offers several key advantages: (i) lightweight with low computing overhead due to single-turn retrieval; (ii) easy implementation and compatibility with any LVLM; and (iii) significant, consistent performance gains across long video understanding benchmarks, including Video-MME, MLVU, and LongVideoBench. Notably, our model demonstrates superior performance over proprietary models like Gemini-1.5-Pro and GPT-4o when utilized with a 72B model.
>
---
#### [replaced 011] Prototypical Contrastive Learning-based CLIP Fine-tuning for Object Re-identification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2310.17218v2](https://arxiv.org/pdf/2310.17218v2)**

> **作者:** Jiachen Li; Xiaojin Gong
>
> **摘要:** This work aims to adapt large-scale pre-trained vision-language models, such as contrastive language-image pretraining (CLIP), to enhance the performance of object reidentification (Re-ID) across various supervision settings. Although prompt learning has enabled a recent work named CLIP-ReID to achieve promising performance, the underlying mechanisms and the necessity of prompt learning remain unclear due to the absence of semantic labels in ReID tasks. In this work, we first analyze the role prompt learning in CLIP-ReID and identify its limitations. Based on our investigations, we propose a simple yet effective approach to adapt CLIP for supervised object Re-ID. Our approach directly fine-tunes the image encoder of CLIP using a prototypical contrastive learning (PCL) loss, eliminating the need for prompt learning. Experimental results on both person and vehicle Re-ID datasets demonstrate the competitiveness of our method compared to CLIP-ReID. Furthermore, we extend our PCL-based CLIP fine-tuning approach to unsupervised scenarios, where we achieve state-of-the art performance.
>
---
#### [replaced 012] Q-SAM2: Accurate Quantization for Segment Anything Model 2
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.09782v2](https://arxiv.org/pdf/2506.09782v2)**

> **作者:** Nicola Farronato; Florian Scheidegger; Mattia Rigotti; Cristiano Malossi; Michele Magno; Haotong Qin
>
> **备注:** 22 pages
>
> **摘要:** The Segment Anything Model 2 (SAM2) is a powerful foundation model for promptable segmentation. However, its high computational and memory costs are a major barrier to deployment on resource-constrained devices. In this paper, we present Q-SAM2, an accurate low-bit quantization method that achieves high compression and high fidelity. To address performance degradation arising from challenging weight and activation distributions during quantization, Q-SAM2 introduces two novel contributions: Variance-Reduced Calibration (VRC), an initialization method that reduces weight statistical variance by minimizing the Frobenius norm over a small calibration batch; and Learnable Statistical Clipping (LSC), a Quantization-Aware Training (QAT) method that learns momentum-stabilized clipping factors to manage outliers in weights and activations. Comprehensive experiments demonstrate that Q-SAM2 achieves highly accurate inference with substantial efficiency gains, significantly surpassing state-of-the-art general QAT schemes, particularly in the ultra-low 2-bit regime. Specifically, Q-SAM2 achieves an accuracy gain of up to 9.7 ppt in J&F on the video segmentation benchmark and 7.3 ppt in mIoU for instance segmentation over the best competing QAT model, all while achieving an 8x reduction in model size compared to the BF16 baseline.
>
---
#### [replaced 013] GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2406.18572v4](https://arxiv.org/pdf/2406.18572v4)**

> **作者:** Ling Li; Yu Ye; Yao Zhou; Bingchuan Jiang; Wei Zeng
>
> **摘要:** This work tackles the problem of geo-localization with a new paradigm using a large vision-language model (LVLM) augmented with human inference knowledge. A primary challenge here is the scarcity of data for training the LVLM - existing street-view datasets often contain numerous low-quality images lacking visual clues, and lack any reasoning inference. To address the data-quality issue, we devise a CLIP-based network to quantify the degree of street-view images being locatable, leading to the creation of a new dataset comprising highly locatable street views. To enhance reasoning inference, we integrate external knowledge obtained from real geo-localization games, tapping into valuable human inference capabilities. The data are utilized to train GeoReasoner, which undergoes fine-tuning through dedicated reasoning and location-tuning stages. Qualitative and quantitative evaluations illustrate that GeoReasoner outperforms counterpart LVLMs by more than 25% at country-level and 38% at city-level geo-localization tasks, and surpasses StreetCLIP performance while requiring fewer training resources. The data and code are available at https://github.com/lingli1996/GeoReasoner.
>
---
#### [replaced 014] VideoLights: Feature Refinement and Cross-Task Alignment Transformer for Joint Video Highlight Detection and Moment Retrieval
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2412.01558v2](https://arxiv.org/pdf/2412.01558v2)**

> **作者:** Dhiman Paul; Md Rizwan Parvez; Nabeel Mohammed; Shafin Rahman
>
> **摘要:** Prevailing joint prediction transformers for Video Highlight Detection and Moment Retrieval (HD/MR) exhibit deficiencies in handling cross-task dynamics, achieving robust video-text alignment, and utilizing effective attention mechanisms, with the potential of Large Language/Vision-Language Models (LLMs/LVLMs) being largely untapped. This paper introduces VideoLights, a novel HD/MR framework addressing these limitations by incorporating: (i) Convolutional Projection and Feature Refinement modules with an alignment loss for enhanced video-text feature congruity; (ii) a Bi-Directional Cross-Modal Fusion network for strongly coupled query-aware representations; (iii) a Uni-directional joint-task feedback mechanism for synergistic task improvement; (iv) hard positive/negative losses for adaptive learning; and (v) the leveraging of LVLMs (e.g., BLIP-2) for superior multimodal feature integration and intelligent pre-training with synthetic data. Comprehensive evaluations on QVHighlights, TVSum, and Charades-STA benchmarks demonstrate that VideoLights significantly surpasses existing baselines, establishing new state-of-the-art performances. Codes and model checkpoints are available at https://github.com/dpaul06/VideoLights .
>
---
#### [replaced 015] Don't Reach for the Stars: Rethinking Topology for Resilient Federated Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05224v2](https://arxiv.org/pdf/2508.05224v2)**

> **作者:** Mirko Konstantin; Anirban Mukhopadhyay
>
> **摘要:** Federated learning (FL) enables collaborative model training across distributed clients while preserving data privacy by keeping data local. Traditional FL approaches rely on a centralized, star-shaped topology, where a central server aggregates model updates from clients. However, this architecture introduces several limitations, including a single point of failure, limited personalization, and poor robustness to distribution shifts or vulnerability to malfunctioning clients. Moreover, update selection in centralized FL often relies on low-level parameter differences, which can be unreliable when client data is not independent and identically distributed, and offer clients little control. In this work, we propose a decentralized, peer-to-peer (P2P) FL framework. It leverages the flexibility of the P2P topology to enable each client to identify and aggregate a personalized set of trustworthy and beneficial updates.This framework is the Local Inference Guided Aggregation for Heterogeneous Training Environments to Yield Enhancement Through Agreement and Regularization (LIGHTYEAR). Central to our method is an agreement score, computed on a local validation set, which quantifies the semantic alignment of incoming updates in the function space with respect to the clients reference model. Each client uses this score to select a tailored subset of updates and performs aggregation with a regularization term that further stabilizes the training. Our empirical evaluation across five datasets shows that the proposed approach consistently outperforms both, centralized baselines and existing P2P methods in terms of client-level performance, particularly under adversarial and heterogeneous conditions.
>
---
#### [replaced 016] Optimization-Free Style Transfer for 3D Gaussian Splats
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05813v2](https://arxiv.org/pdf/2508.05813v2)**

> **作者:** Raphael Du Sablon; David Hart
>
> **摘要:** The task of style transfer for 3D Gaussian splats has been explored in many previous works, but these require reconstructing or fine-tuning the splat while incorporating style information or optimizing a feature extraction network on the splat representation. We propose a reconstruction- and optimization-free approach to stylizing 3D Gaussian splats, allowing for direct stylization on a .ply or .splat file without requiring the original camera views. This is done by generating a graph structure across the implicit surface of the splat representation. A feed-forward, surface-based stylization method is then used and interpolated back to the individual splats in the scene. This also allows for fast stylization of splats with no additional training, achieving speeds under 2 minutes even on CPU-based consumer hardware. We demonstrate the quality results this approach achieves and compare to other 3D Gaussian splat style transfer methods. Code is publicly available at https://github.com/davidmhart/FastSplatStyler.
>
---
#### [replaced 017] The Shape of Sight: A Homological Framework for Unifying Visual Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/1802.04723v2](https://arxiv.org/pdf/1802.04723v2)**

> **作者:** Xin Li
>
> **摘要:** Visual perception, the brain's construction of a stable world from sensory data, faces several long-standing, fundamental challenges. While often studied separately, these problems have resisted a single, unifying computational framework. In this perspective, we propose a homological framework for visual perception. We argue that the brain's latent representations are governed by their topological parity. This parity interpretation functionally separates homological structures into two distinct classes: 1) Even-dimensional homology ($H_{even}$) acts as static, integrative scaffolds. These structures bind context and content into ``wholes'' or ``what'', serving as the stable, resonant cavities for perceptual objects; 2) Odd-dimensional homology ($H_{odd}$) acts as dynamic, recurrent flows. These structures represent paths, transformations, and self-sustaining ``traces'' or ``where'' that navigate the perceptual landscape. This scaffold-and-flow model is supported by the ventral-dorsal pathway separation and provides a unified solution to three core problems in visual perception. Homological parity hypothesis recasts visual perception not as a linear computation, but as a dynamic interaction between stable, integrative structures and the recurrent, self-sustaining flows that run on them. This perspective offers a new mathematical foundation for linking neural dynamics to perception and cognition.
>
---
#### [replaced 018] Intraoperative 2D/3D Registration via Spherical Similarity Learning and Differentiable Levenberg-Marquardt Optimization
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2509.06890v3](https://arxiv.org/pdf/2509.06890v3)**

> **作者:** Minheng Chen; Youyong Kong
>
> **备注:** WACV 2026 Accepted
>
> **摘要:** Intraoperative 2D/3D registration aligns preoperative 3D volumes with real-time 2D radiographs, enabling accurate localization of instruments and implants. A recent fully differentiable similarity learning framework approximates geodesic distances on SE(3), expanding the capture range of registration and mitigating the effects of substantial disturbances, but existing Euclidean approximations distort manifold structure and slow convergence. To address these limitations, we explore similarity learning in non-Euclidean spherical feature spaces to better capture and fit complex manifold structure. We extract feature embeddings using a CNN-Transformer encoder, project them into spherical space, and approximate their geodesic distances with Riemannian distances in the bi-invariant SO(4) space. This enables a more expressive and geometrically consistent deep similarity metric, enhancing the ability to distinguish subtle pose differences. During inference, we replace gradient descent with fully differentiable Levenberg-Marquardt optimization to accelerate convergence. Experiments on real and synthetic datasets show superior accuracy in both patient-specific and patient-agnostic scenarios.
>
---
#### [replaced 019] 2D Gaussians Spatial Transport for Point-supervised Density Regression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14477v2](https://arxiv.org/pdf/2511.14477v2)**

> **作者:** Miao Shang; Xiaopeng Hong
>
> **备注:** 15 pages, 6 figures. This is the preprint version of the paper and supplemental material to appear in AAAI, 2026. Please cite the final published version. Code is available at https://github.com/infinite0522/GST
>
> **摘要:** This paper introduces Gaussian Spatial Transport (GST), a novel framework that leverages Gaussian splatting to facilitate transport from the probability measure in the image coordinate space to the annotation map. We propose a Gaussian splatting-based method to estimate pixel-annotation correspondence, which is then used to compute a transport plan derived from Bayesian probability. To integrate the resulting transport plan into standard network optimization in typical computer vision tasks, we derive a loss function that measures discrepancy after transport. Extensive experiments on representative computer vision tasks, including crowd counting and landmark detection, validate the effectiveness of our approach. Compared to conventional optimal transport schemes, GST eliminates iterative transport plan computation during training, significantly improving efficiency. Code is available at https://github.com/infinite0522/GST.
>
---
#### [replaced 020] KV-Efficient VLA: A Method to Speed up Vision Language Models with RNN-Gated Chunked KV Cache
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.21354v2](https://arxiv.org/pdf/2509.21354v2)**

> **作者:** Wanshun Xu; Long Zhuang; Lianlei Shan
>
> **摘要:** Vision-Language-Action (VLA) models offer a unified framework for robotic perception and control, but their ability to scale to real-world, long-horizon tasks is limited by the high computational cost of attention and the large memory required for storing key-value (KV) pairs during inference, particularly when retaining historical image tokens as context. Recent methods have focused on scaling backbone architectures to improve generalization, with less emphasis on addressing inference inefficiencies essential for real-time use. In this work, we present KV-Efficient VLA, a model-agnostic memory compression approach designed to address these limitations by introducing a lightweight mechanism to selectively retain high-utility context. Our method partitions the KV cache into fixed-size chunks and employs a recurrent gating module to summarize and filter the historical context according to learned utility scores. This design aims to preserve recent fine-grained detail while aggressively pruning stale, low-relevance memory. Based on experiments, our approach can yield an average of 24.6% FLOPs savings, 1.34x inference speedup, and 1.87x reduction in KV memory. Our method integrates seamlessly into recent VLA stacks, enabling scalable inference without modifying downstream control logic.
>
---
#### [replaced 021] Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.00062v3](https://arxiv.org/pdf/2509.00062v3)**

> **作者:** Justin Jung
>
> **备注:** Accepted at NeurIPS 2025 Structured Probabilistic Inference & Generative Modeling Workshop
>
> **摘要:** Generating realistic sparse multi-category 3D voxel structures is difficult due to the cubic memory scaling of voxel structures and moreover the significant class imbalance caused by sparsity. We introduce Scaffold Diffusion, a generative model designed for sparse multi-category 3D voxel structures. By treating voxels as tokens, Scaffold Diffusion uses a discrete diffusion language model to generate 3D voxel structures. We show that discrete diffusion language models can be extended beyond inherently sequential domains such as text to generate spatially coherent 3D structures. We evaluate on Minecraft house structures from the 3D-Craft dataset and demonstrate that, unlike prior baselines and an auto-regressive formulation, Scaffold Diffusion produces realistic and coherent structures even when trained on data with over 98% sparsity. We provide an interactive viewer where readers can visualize generated samples and the generation process: https://scaffold.deepexploration.org/
>
---
#### [replaced 022] LivePyxel: Accelerating image annotations with a Python-integrated webcam live streaming
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.13504v2](https://arxiv.org/pdf/2509.13504v2)**

> **作者:** Uriel Garcilazo-Cruz; Joseph O. Okeme; Rodrigo A. Vargas-Hernández
>
> **备注:** 9 pages, 10 figures, SM, 5 pages, 5 figures, 1 Table
>
> **摘要:** The lack of flexible annotation tools has hindered the deployment of AI models in some scientific areas. Most existing image annotation software requires users to upload a precollected dataset, which limits support for on-demand pipelines and introduces unnecessary steps to acquire images. This constraint is particularly problematic in laboratory environments, where on-site data acquisition from instruments such as microscopes is increasingly common. In this work, we introduce \texttt{LivePixel}, a Python-based graphical user interface that integrates with imaging systems, such as webcams, microscopes, and others, to enable on-site image annotation. LivePyxel is designed to be easy to use through a simple interface that allows users to precisely delimit areas for annotation using tools commonly found in commercial graphics editing software. Of particular interest is the availability of Bézier splines and binary masks, and the software's capacity to work with non-destructive layers that enable high-performance editing. LivePyxel also integrates a wide compatibility across video devices, and it's optimized for object detection operations via the use of OpenCV in combination with high-performance libraries designed to handle matrix and linear algebra operations via Numpy effectively. LivePyxel facilitates seamless data collection and labeling, accelerating the development of AI models in experimental workflows. LivePyxel is freely available at https://github.com/UGarCil/LivePyxel
>
---
#### [replaced 023] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.13848v2](https://arxiv.org/pdf/2509.13848v2)**

> **作者:** Jiayi Pan; Jiaming Xu; Yongkang Zhou; Guohao Dai
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Feature caching has recently emerged as a promising method for diffusion model acceleration. It effectively alleviates the inefficiency problem caused by high computational requirements by caching similar features in the inference process of the diffusion model. In this paper, we analyze existing feature caching methods from the perspective of information utilization, and point out that relying solely on historical information will lead to constrained accuracy and speed performance. And we propose a novel paradigm that introduces future information via self-speculation based on the information similarity at the same time step across different iteration times. Based on this paradigm, we present \textit{SpecDiff}, a training-free multi-level feature caching strategy including a cached feature selection algorithm and a multi-level feature classification algorithm. (1) Feature selection algorithm based on self-speculative information. \textit{SpecDiff} determines a dynamic importance score for each token based on self-speculative information and historical information, and performs cached feature selection through the importance score. (2) Multi-level feature classification algorithm based on feature importance scores. \textit{SpecDiff} classifies tokens by leveraging the differences in feature importance scores and introduces a multi-level feature calculation strategy. Extensive experiments show that \textit{SpecDiff} achieves average 2.80 \times, 2.74 \times , and 3.17\times speedup with negligible quality loss in Stable Diffusion 3, 3.5, and FLUX compared to RFlow on NVIDIA A800-80GB GPU. By merging speculative and historical information, \textit{SpecDiff} overcomes the speedup-accuracy trade-off bottleneck, pushing the Pareto frontier of speedup and accuracy in the efficient diffusion model inference.
>
---
#### [replaced 024] MedBridge: Bridging Foundation Vision-Language Models to Medical Image Diagnosis in Chest X-Ray
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.21698v2](https://arxiv.org/pdf/2505.21698v2)**

> **作者:** Yitong Li; Morteza Ghahremani; Christian Wachinger
>
> **摘要:** Recent vision-language foundation models deliver state-of-the-art results in natural image classification, but falter in medical images due to pronounced domain shifts. Training a medical foundation model also requires substantial resources, including extensive annotated data and high computational capacity. To bridge this gap with minimal overhead, we introduce MedBridge, a lightweight multimodal adaptation framework that flexibly re-purposes arbitrary pre-trained foundation VLMs for medical image diagnosis. MedBridge comprises three novel core components. First, a Focal Sampling module that subsamples and extracts high-resolution local regions to capture subtle pathological features, compensating for the limited input resolution of foundation VLMs. Second, a Query-Encoder model with a small set of learnable queries to align the feature maps of frozen VLMs with medical semantics, without requiring retraining of the backbone layers. Third, a Mixture of Experts mechanism, driven by learnable queries, harnesses the complementary strength of various VLMs to maximize diagnostic performance. We evaluate MedBridge on five chest radiograph benchmarks in three key adaptation tasks, demonstrating its superior performance in both cross-domain and in-domain adaptation settings under varying levels of training data availability. MedBridge achieved an improvement of 6-15% in AUC compared to state-of-the-art VLM adaptation methods in multi-label thoracic disease diagnosis, underscoring its effectiveness in leveraging diverse foundation models for accurate and data-efficient medical diagnosis. Our project and code are available at https://github.com/ai-med/MedBridge.
>
---
#### [replaced 025] Scene Summarization: Clustering Scene Videos into Spatially Diverse Frames
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2311.17940v3](https://arxiv.org/pdf/2311.17940v3)**

> **作者:** Chao Chen; Mingzhi Zhu; Ankush Pratap Singh; Yu Yan; Felix Juefei-Xu; Chen Feng
>
> **摘要:** Humans are remarkably efficient at forming spatial understanding from just a few visual observations. When browsing real estate or navigating unfamiliar spaces, they intuitively select a small set of views that summarize the spatial layout. Inspired by this ability, we introduce scene summarization, the task of condensing long, continuous scene videos into a compact set of spatially diverse keyframes that facilitate global spatial reasoning. Unlike conventional video summarization-which focuses on user-edited, fragmented clips and often ignores spatial continuity-our goal is to mimic how humans abstract spatial layout from sparse views. We propose SceneSum, a two-stage self-supervised pipeline that first clusters video frames using visual place recognition to promote spatial diversity, then selects representative keyframes from each cluster under resource constraints. When camera trajectories are available, a lightweight supervised loss further refines clustering and selection. Experiments on real and simulated indoor datasets show that SceneSum produces more spatially informative summaries and outperforms existing video summarization baselines.
>
---
#### [replaced 026] Fusing Biomechanical and Spatio-Temporal Features for Fall Prediction: Characterizing and Mitigating the Simulation-to-Reality Gap
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14620v2](https://arxiv.org/pdf/2511.14620v2)**

> **作者:** Md Fokhrul Islam; Sajeda Al-Hammouri; Christopher J. Arellano; Kavan Hazeli; Heman Shakeri
>
> **摘要:** Falls are a leading cause of injury and loss of independence among older adults. Vision-based fall prediction systems offer a non-invasive solution to anticipate falls seconds before impact, but their development is hindered by the scarcity of available fall data. Contributing to these efforts, this study proposes the Biomechanical Spatio-Temporal Graph Convolutional Network (BioST-GCN), a dual-stream model that combines both pose and biomechanical information using a cross-attention fusion mechanism. Our model outperforms the vanilla ST-GCN baseline by 5.32% and 2.91% F1-score on the simulated MCF-UA stunt-actor and MUVIM datasets, respectively. The spatio-temporal attention mechanisms in the ST-GCN stream also provide interpretability by identifying critical joints and temporal phases. However, a critical simulation-reality gap persists. While our model achieves an 89.0% F1-score with full supervision on simulated data, zero-shot generalization to unseen subjects drops to 35.9%. This performance decline is likely due to biases in simulated data, such as 'intent-to-fall' cues. For older adults, particularly those with diabetes or frailty, this gap is exacerbated by their unique kinematic profiles. To address this, we propose personalization strategies and advocate for privacy-preserving data pipelines to enable real-world validation. Our findings underscore the urgent need to bridge the gap between simulated and real-world data to develop effective fall prediction systems for vulnerable elderly populations.
>
---
#### [replaced 027] JointTuner: Appearance-Motion Adaptive Joint Training for Customized Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.23951v3](https://arxiv.org/pdf/2503.23951v3)**

> **作者:** Fangda Chen; Shanshan Zhao; Chuanfu Xu; Long Lan
>
> **备注:** Project Page: https://fdchen24.github.io/JointTuner-Website
>
> **摘要:** Recent advancements in customized video generation have led to significant improvements in the simultaneous adaptation of appearance and motion. Typically, decoupling the appearance and motion training, prior methods often introduce concept interference, resulting in inaccurate rendering of appearance features or motion patterns. In addition, these methods often suffer from appearance contamination, in which background and foreground elements from reference videos distort the customized video. This paper aims to alleviate these issues by proposing JointTuner. The core motivation of our JointTuner is to enable joint optimization of both appearance and motion components, upon which two key innovations are developed, i.e., Gated Low-Rank Adaptation (GLoRA) and Appearance-independent Temporal Loss (AiT Loss). Specifically, GLoRA uses a context-aware activation layer, analogous to a gating regulator, to dynamically steer LoRA modules toward learning either appearance or motion while maintaining spatio-temporal consistency. Moreover, with the finding that channel-temporal shift noise suppresses appearance-related low-frequencies while enhancing motion-related high-frequencies, we designed the AiT Loss. This loss adds the same shift to the diffusion model's predicted noise during fine-tuning, forcing the model to prioritize learning motion patterns. JointTuner's architecture-agnostic design supports both UNet (e.g., ZeroScope) and Diffusion Transformer (e.g., CogVideoX) backbones, ensuring its customization capabilities scale with the evolution of foundational video models. Furthermore, we present a systematic evaluation framework for appearance-motion combined customization, covering 90 combinations evaluated along four critical dimensions: semantic alignment, motion dynamism, temporal consistency, and perceptual quality. Our project homepage is available online.
>
---
#### [replaced 028] Matrix-game 2.0: An open-source real-time and streaming interactive world model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13009v2](https://arxiv.org/pdf/2508.13009v2)**

> **作者:** Xianglong He; Chunli Peng; Zexiang Liu; Boyang Wang; Yifan Zhang; Qi Cui; Fei Kang; Biao Jiang; Mengyin An; Yangyang Ren; Baixin Xu; Hao-Xiang Guo; Kaixiong Gong; Cyrus Wu; Wei Li; Xuchen Song; Yang Liu; Eric Li; Yahui Zhou
>
> **备注:** Project Page: https://matrix-game-v2.github.io
>
> **摘要:** Recent advances in interactive video generations have demonstrated diffusion model's potential as world models by capturing complex physical dynamics and interactive behaviors. However, existing interactive world models depend on bidirectional attention and lengthy inference steps, severely limiting real-time performance. Consequently, they are hard to simulate real-world dynamics, where outcomes must update instantaneously based on historical context and current actions. To address this, we present Matrix-Game 2.0, an interactive world model generates long videos on-the-fly via few-step auto-regressive diffusion. Our framework consists of three key components: (1) A scalable data production pipeline for Unreal Engine and GTA5 environments to effectively produce massive amounts (about 1200 hours) of video data with diverse interaction annotations; (2) An action injection module that enables frame-level mouse and keyboard inputs as interactive conditions; (3) A few-step distillation based on the casual architecture for real-time and streaming video generation. Matrix Game 2.0 can generate high-quality minute-level videos across diverse scenes at an ultra-fast speed of 25 FPS. We open-source our model weights and codebase to advance research in interactive world modeling.
>
---
#### [replaced 029] Visual Anagrams Reveal Hidden Differences in Holistic Shape Processing Across Vision Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.00493v3](https://arxiv.org/pdf/2507.00493v3)**

> **作者:** Fenil R. Doshi; Thomas Fel; Talia Konkle; George Alvarez
>
> **备注:** Project page: https://www.fenildoshi.com/configural-shape/ updated email address
>
> **摘要:** Humans are able to recognize objects based on both local texture cues and the configuration of object parts, yet contemporary vision models primarily harvest local texture cues, yielding brittle, non-compositional features. Work on shape-vs-texture bias has pitted shape and texture representations in opposition, measuring shape relative to texture, ignoring the possibility that models (and humans) can simultaneously rely on both types of cues, and obscuring the absolute quality of both types of representation. We therefore recast shape evaluation as a matter of absolute configural competence, operationalized by the Configural Shape Score (CSS), which (i) measures the ability to recognize both images in Object-Anagram pairs that preserve local texture while permuting global part arrangement to depict different object categories. Across 86 convolutional, transformer, and hybrid models, CSS (ii) uncovers a broad spectrum of configural sensitivity with fully self-supervised and language-aligned transformers -- exemplified by DINOv2, SigLIP2 and EVA-CLIP -- occupying the top end of the CSS spectrum. Mechanistic probes reveal that (iii) high-CSS networks depend on long-range interactions: radius-controlled attention masks abolish performance showing a distinctive U-shaped integration profile, and representational-similarity analyses expose a mid-depth transition from local to global coding. A BagNet control remains at chance (iv), ruling out "border-hacking" strategies. Finally, (v) we show that configural shape score also predicts other shape-dependent evals. Overall, we propose that the path toward truly robust, generalizable, and human-like vision systems may not lie in forcing an artificial choice between shape and texture, but rather in architectural and learning frameworks that seamlessly integrate both local-texture and global configural shape.
>
---
#### [replaced 030] OmniLens++: Blind Lens Aberration Correction via Large LensLib Pre-Training and Latent PSF Representation
- **分类: eess.IV; cs.CV; cs.LG; physics.optics**

- **链接: [https://arxiv.org/pdf/2511.17126v2](https://arxiv.org/pdf/2511.17126v2)**

> **作者:** Qi Jiang; Xiaolong Qian; Yao Gao; Lei Sun; Kailun Yang; Zhonghua Yi; Wenyong Li; Ming-Hsuan Yang; Luc Van Gool; Kaiwei Wang
>
> **备注:** The source code and datasets will be made publicly available at https://github.com/zju-jiangqi/OmniLens2
>
> **摘要:** Emerging deep-learning-based lens library pre-training (LensLib-PT) pipeline offers a new avenue for blind lens aberration correction by training a universal neural network, demonstrating strong capability in handling diverse unknown optical degradations. This work proposes the OmniLens++ framework, which resolves two challenges that hinder the generalization ability of existing pipelines: the difficulty of scaling data and the absence of prior guidance characterizing optical degradation. To improve data scalability, we expand the design specifications to increase the degradation diversity of the lens source, and we sample a more uniform distribution by quantifying the spatial-variation patterns and severity of optical degradation. In terms of model design, to leverage the Point Spread Functions (PSFs), which intuitively describe optical degradation, as guidance in a blind paradigm, we propose the Latent PSF Representation (LPR). The VQVAE framework is introduced to learn latent features of LensLib's PSFs, which is assisted by modeling the optical degradation process to constrain the learning of degradation priors. Experiments on diverse aberrations of real-world lenses and synthetic LensLib show that OmniLens++ exhibits state-of-the-art generalization capacity in blind aberration correction. Beyond performance, the AODLibpro is verified as a scalable foundation for more effective training across diverse aberrations, and LPR can further tap the potential of large-scale LensLib. The source code and datasets will be made publicly available at https://github.com/zju-jiangqi/OmniLens2.
>
---
#### [replaced 031] FCDM: A Physics-Guided Bidirectional Frequency Aware Convolution and Diffusion-Based Model for Sinogram Inpainting
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2409.06714v5](https://arxiv.org/pdf/2409.06714v5)**

> **作者:** Jiaze E; Srutarshi Banerjee; Tekin Bicer; Guannan Wang; Yanfu Zhang; Bin Ren
>
> **摘要:** Computed tomography (CT) is widely used in scientific imaging systems such as synchrotron and laboratory-based nano-CT, but acquiring full-view sinograms requires high radiation dose and long scan times. Sparse-view CT alleviates this burden but yields incomplete sinograms with structured signal loss, hampering accurate reconstruction. Unlike RGB images, sinograms encode overlapping features along projection paths and exhibit distinct directional spectral patterns, which make conventional RGB-oriented inpainting approaches--including diffusion models--ineffective for sinogram restoration, as they disregard the angular dependencies and physical constraints inherent to tomographic data. To overcome these limitations, we propose FCDM, a diffusion-based framework tailored for sinograms, which restores global structure through bidirectional frequency reasoning and angular-aware masking, while enforcing physical plausibility via physics-guided constraints and frequency-adaptive noise control. Experiments on real-world datasets show that FCDM consistently outperforms baselines, achieving SSIM over 0.93 and PSNR above 31 dB across diverse sparse-view scenarios.
>
---
#### [replaced 032] MoReMouse: Monocular Reconstruction of Laboratory Mouse
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.04258v2](https://arxiv.org/pdf/2507.04258v2)**

> **作者:** Yuan Zhong; Jingxiang Sun; Zhongbin Zhang; Liang An; Yebin Liu
>
> **摘要:** Laboratory mice, particularly the C57BL/6 strain, are essential animal models in biomedical research. However, accurate 3D surface motion reconstruction of mice remains a significant challenge due to their complex non-rigid deformations, textureless fur-covered surfaces, and the lack of realistic 3D mesh models. Moreover, existing visual datasets for mice reconstruction only contain sparse viewpoints without 3D geometries. To fill the gap, we introduce MoReMouse, the first monocular dense 3D reconstruction network specifically designed for C57BL/6 mice. To achieve high-fidelity 3D reconstructions, we present three key innovations. First, we create the first high-fidelity, dense-view synthetic dataset for C57BL/6 mice by rendering a realistic, anatomically accurate Gaussian mouse avatar. Second, MoReMouse leverages a transformer-based feedforward architecture combined with triplane representation, enabling high-quality 3D surface generation from a single image, optimized for the intricacies of small animal morphology. Third, we propose geodesic-based continuous correspondence embeddings on the mouse surface, which serve as strong semantic priors, improving surface consistency and reconstruction stability, especially in highly dynamic regions like limbs and tail. Through extensive quantitative and qualitative evaluations, we demonstrate that MoReMouse significantly outperforms existing open-source methods in both accuracy and robustness.
>
---
#### [replaced 033] RN-SDEs: Limited-Angle CT Reconstruction with Residual Null-Space Diffusion Stochastic Differential Equations
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2409.13930v3](https://arxiv.org/pdf/2409.13930v3)**

> **作者:** Jiaqi Guo; Santiago Lopez-Tapia; Wing Shun Li; Yunan Wu; Marcelo Carignano; Martin Kröger; Vinayak P. Dravid; Igal Szleifer; Vadim Backman; Aggelos K. Katsaggelos
>
> **摘要:** Computed tomography is a widely used imaging modality with applications ranging from medical imaging to material analysis. One major challenge arises from the lack of scanning information at certain angles, resulting in distortion or artifacts in the reconstructed images. This is referred to as the Limited Angle Computed Tomography (LACT) reconstruction problem. To address this problem, we propose the use of Residual Null-Space Diffusion Stochastic Differential Equations (RN-SDEs), which are a variant of diffusion models that characterize the diffusion process with mean-reverting (MR) stochastic differential equations. To demonstrate the generalizability of RN-SDEs, we conducted experiments with two different LACT datasets, ChromSTEM and C4KC-KiTS. Through extensive experiments, we demonstrate that by leveraging learned MR-SDEs as a prior and emphasizing data consistency using Range-Null Space Decomposition (RNSD) based rectification, we can recover high-quality images from severely degraded ones and achieve state-of-the-art performance in most LACT tasks. Additionally, we present a quantitative comparison of RN-SDE with other networks, in terms of computational complexity and runtime efficiency, highlighting the superior effectiveness of our proposed approach.
>
---
#### [replaced 034] FOCUS: Efficient Keyframe Selection for Long Video Understanding
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.27280v2](https://arxiv.org/pdf/2510.27280v2)**

> **作者:** Zirui Zhu; Hailun Xu; Yang Luo; Yong Liu; Kanchan Sarkar; Zhenheng Yang; Yang You
>
> **摘要:** Multimodal large language models (MLLMs) represent images and video frames as visual tokens. Scaling from single images to hour-long videos, however, inflates the token budget far beyond practical limits. Popular pipelines therefore either uniformly subsample or apply keyframe selection with retrieval-style scoring using smaller vision-language models. However, these keyframe selection methods still rely on pre-filtering before selection to reduce the inference cost and can miss the most informative moments. We propose FOCUS, Frame-Optimistic Confidence Upper-bound Selection, a training-free, model-agnostic keyframe selection module that selects query-relevant frames under a strict token budget. FOCUS formulates keyframe selection as a combinatorial pure-exploration (CPE) problem in multi-armed bandits: it treats short temporal clips as arms, and uses empirical means and Bernstein confidence radius to identify informative regions while preserving exploration of uncertain areas. The resulting two-stage exploration-exploitation procedure reduces from a sequential policy with theoretical guarantees, first identifying high-value temporal regions, then selecting top-scoring frames within each region. On two long-video question-answering benchmarks, FOCUS delivers substantial accuracy improvements while processing less than 2% of video frames. For videos longer than 20 minutes, it achieves an 11.9% gain in accuracy on LongVideoBench, demonstrating its effectiveness as a keyframe selection method and providing a simple and general solution for scalable long-video understanding with MLLMs. Code is available at https://github.com/NUS-HPC-AI-Lab/FOCUS.
>
---
#### [replaced 035] OmniDocLayout: Towards Diverse Document Layout Generation via Coarse-to-Fine LLM Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.26213v2](https://arxiv.org/pdf/2510.26213v2)**

> **作者:** Hengrui Kang; Zhuangcheng Gu; Zhiyuan Zhao; Zichen Wen; Bin Wang; Weijia Li; Conghui He
>
> **备注:** TL;DR: With the proposed OmniDocLayout-1M dataset and the LLM-based coarse-to-fine learning strategy, we enable diverse and complex document layout generation that achieves both strong condition consistency and adherence to fundamental aesthetic principles
>
> **摘要:** Document AI has advanced rapidly and is attracting increasing attention. Yet, while most efforts have focused on document layout analysis (DLA), its generative counterpart, layout generation, remains underexplored. Distinct from traditional graphic layout design and room layout planning, document layout generation typically involves a larger number of elements per page and exhibits greater structural diversity and complexity. Currently, a major obstacle lies in the scarcity of diverse document layouts: academic papers with Manhattan-style structures dominate existing studies, while open-world genres such as newspapers and magazines remain severely underrepresented. To address this gap, we curate OmniDocLayout-1M, the first million-scale dataset of diverse document layouts, covering six common document types and comprising contemporary layouts collected from multiple sources. Moreover, since existing methods struggle in complex domains and often fail to arrange long sequences coherently, we introduce OmniDocLayout-LLM, a 0.5B model with designed two-stage Coarse-to-Fine learning paradigm:1) learning universal layout principles from our dataset with coarse category definitions, and 2) transferring the knowledge to a specific domain with few fine-grained annotated samples. Extensive experiments demonstrate that our approach achieves strong performance on multiple domains in M$^6$Doc dataset, substantially surpassing both existing layout generation experts and several latest general-purpose LLMs. Our code, dataset, and models will be publicly released.
>
---
#### [replaced 036] The SA-FARI Dataset: Segment Anything in Footage of Animals for Recognition and Identification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.15622v2](https://arxiv.org/pdf/2511.15622v2)**

> **作者:** Dante Francisco Wasmuht; Otto Brookes; Maximillian Schall; Pablo Palencia; Chris Beirne; Tilo Burghardt; Majid Mirmehdi; Hjalmar Kühl; Mimi Arandjelovic; Sam Pottie; Peter Bermant; Brandon Asheim; Yi Jin Toh; Adam Elzinga; Jason Holmberg; Andrew Whitworth; Eleanor Flatt; Laura Gustafson; Chaitanya Ryali; Yuan-Ting Hu; Baishan Guo; Andrew Westbury; Kate Saenko; Didac Suris
>
> **摘要:** Automated video analysis is critical for wildlife conservation. A foundational task in this domain is multi-animal tracking (MAT), which underpins applications such as individual re-identification and behavior recognition. However, existing datasets are limited in scale, constrained to a few species, or lack sufficient temporal and geographical diversity - leaving no suitable benchmark for training general-purpose MAT models applicable across wild animal populations. To address this, we introduce SA-FARI, the largest open-source MAT dataset for wild animals. It comprises 11,609 camera trap videos collected over approximately 10 years (2014-2024) from 741 locations across 4 continents, spanning 99 species categories. Each video is exhaustively annotated culminating in ~46 hours of densely annotated footage containing 16,224 masklet identities and 942,702 individual bounding boxes, segmentation masks, and species labels. Alongside the task-specific annotations, we publish anonymized camera trap locations for each video. Finally, we present comprehensive benchmarks on SA-FARI using state-of-the-art vision-language models for detection and tracking, including SAM 3, evaluated with both species-specific and generic animal prompts. We also compare against vision-only methods developed specifically for wildlife analysis. SA-FARI is the first large-scale dataset to combine high species diversity, multi-region coverage, and high-quality spatio-temporal annotations, offering a new foundation for advancing generalizable multianimal tracking in the wild. The dataset is available at https://www.conservationxlabs.com/sa-fari.
>
---
#### [replaced 037] Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.00416v3](https://arxiv.org/pdf/2507.00416v3)**

> **作者:** Tao Lin; Gen Li; Yilei Zhong; Yanwen Zou; Yuxin Du; Jiting Liu; Encheng Gu; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising framework for enabling generalist robots capable of perceiving, reasoning, and acting in the real world. These models usually build upon pretrained Vision-Language Models (VLMs), which excel at semantic understanding due to large-scale image and text pretraining. However, existing VLMs typically lack precise spatial understanding capabilities, as they are primarily tuned on 2D image-text pairs without 3D supervision. To address this limitation, recent approaches have incorporated explicit 3D inputs such as point clouds or depth maps, but this necessitates additional depth sensors or pre-trained depth estimation models, which may yield defective results. In contrast, our work introduces a plug-and-play module that implicitly incorporates 3D geometry features into VLA models by leveraging an off-the-shelf visual geometry foundation model. This integration provides the model with depth-aware visual representations, improving its ability to understand the geometric structure of the scene and the spatial relationships among objects from RGB images alone. We evaluate our method on a set of spatially challenging tasks in both simulation and the real world. Extensive evaluations show that our method significantly improves the performance of state-of-the-art VLA models across diverse scenarios.
>
---
#### [replaced 038] Benchmarking Endoscopic Surgical Image Restoration and Beyond
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19161v2](https://arxiv.org/pdf/2505.19161v2)**

> **作者:** Jialun Pei; Diandian Guo; Donghui Yang; Zhixi Li; Yuxin Feng; Long Ma; Bo Du; Pheng-Ann Heng
>
> **摘要:** In endoscopic surgery, a clear and high-quality visual field is critical for surgeons to make accurate intraoperative decisions. However, persistent visual degradation, including smoke generated by energy devices, lens fogging from thermal gradients, and lens contamination due to blood or tissue fluid splashes during surgical procedures, severely impairs visual clarity. These degenerations can seriously hinder surgical workflow and pose risks to patient safety. To systematically investigate and address various forms of surgical scene degradation, we introduce a real- world open-source surgical image restoration dataset covering endoscopic environments, called SurgClean, which involves multi-type image restoration tasks from two medical sites, i.e., desmoking, defogging, and desplashing. SurgClean comprises 3,113 images with diverse degradation types and corresponding paired reference labels. Based on SurgClean, we establish a standardized evaluation benchmark and provide performance for 22 representative generic task-specific image restoration approaches, including 12 generic and 10 task-specific image restoration approaches. Experimental results reveal substantial performance gaps relative to clinical requirements, highlighting a critical opportunity for algorithm advancements in intelligent surgical restoration. Furthermore, we explore the degradation discrepancies between surgical and natural scenes from structural perception and semantic under- standing perspectives, providing fundamental insights for domain-specific image restoration research. Our work aims to empower restoration algorithms and improve the efficiency of clinical procedures.
>
---
#### [replaced 039] Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.19659v2](https://arxiv.org/pdf/2509.19659v2)**

> **作者:** Aravind Narayanan; Vahid Reza Khazaie; Shaina Raza
>
> **备注:** Accepted to NeurIPS 2025 Workshop (Evaluating the Evolving LLM Lifecycle)
>
> **摘要:** Large vision-language models (VLMs) can jointly interpret images and text, but they are also prone to absorbing and reproducing harmful social stereotypes when visual cues such as age, gender, race, clothing, or occupation are present. To investigate these risks, we introduce a news-image benchmark consisting of 1,343 image-question pairs drawn from diverse outlets, which we annotated with ground-truth answers and demographic attributes (age, gender, race, occupation, and sports). We evaluate a range of state-of-the-art VLMs and employ a large language model (LLM) as judge, with human verification. Our findings show that: (i) visual context systematically shifts model outputs in open-ended settings; (ii) bias prevalence varies across attributes and models, with particularly high risk for gender and occupation; and (iii) higher faithfulness does not necessarily correspond to lower bias. We release the benchmark prompts, evaluation rubric, and code to support reproducible and fairness-aware multimodal assessment.
>
---
#### [replaced 040] SA-Person: Text-Based Person Retrieval with Scene-aware Re-ranking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.24466v3](https://arxiv.org/pdf/2505.24466v3)**

> **作者:** Yingjia Xu; Jinlin Wu; Daming Gao; Zhen Chen; Yang Yang; Min Cao; Mang Ye; Zhen Lei
>
> **备注:** 13 pages, 8 figures. Under review
>
> **摘要:** Text-based person retrieval aims to identify a target individual from an image gallery using a natural language description. Existing methods primarily focus on appearance-driven cross-modal retrieval, yet face significant challenges due to the visual complexity of scenes and the inherent ambiguity of textual descriptions. The contextual information, such as landmarks and relational cues, provides complementary cues that can offer valuable complementary insights for retrieval, but remains underexploited in current approaches. Motivated by this limitation, we propose a novel paradigm: scene-aware text-based person retrieval, which explicitly integrates both individual appearance and global scene context to improve retrieval accuracy. To support this, we first introduce ScenePerson-13W, a large-scale benchmark dataset comprising over 100,000 real-world scenes with rich annotations encompassing both pedestrian attributes and scene context. Based on this dataset, we further present SA-Person, a two-stage retrieval framework. In the first stage, SA-Person performs discriminative appearance grounding by aligning textual descriptions with pedestrian-specific regions. In the second stage, it introduces SceneRanker, a training-free, scene-aware re-ranking module that refines retrieval results by jointly reasoning over pedestrian appearance and the global scene context. Extensive experiments on ScenePerson-13W and existing benchmarks demonstrate the effectiveness of our proposed SA-Person. Both the dataset and code will be publicly released to facilitate future research.
>
---
#### [replaced 041] Pressure2Motion: Hierarchical Human Motion Reconstruction from Ground Pressure with Text Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05038v2](https://arxiv.org/pdf/2511.05038v2)**

> **作者:** Zhengxuan Li; Qinhui Yang; Yiyu Zhuang; Chuan Guo; Xinxin Zuo; Xiaoxiao Long; Yao Yao; Xun Cao; Qiu Shen; Hao Zhu
>
> **摘要:** We present Pressure2Motion, a novel motion capture algorithm that reconstructs human motion from a ground pressure sequence and text prompt. At inference time, Pressure2Motion requires only a pressure mat, eliminating the need for specialized lighting setups, cameras, or wearable devices, making it suitable for privacy-preserving, low-light, and low-cost motion capture scenarios. Such a task is severely ill-posed due to the indeterminacy of pressure signals with respect to full-body motion. To address this issue, we introduce Pressure2Motion, a generative model that leverages pressure features as input and utilizes a text prompt as a high-level guiding constraint to resolve ambiguities. Specifically, our model adopts a dual-level feature extractor to accurately interpret pressure data, followed by a hierarchical diffusion model that discerns broad-scale movement trajectories and subtle posture adjustments. Both the physical cues gained from the pressure sequence and the semantic guidance derived from descriptive texts are leveraged to guide the motion estimation with precision. To the best of our knowledge, Pressure2Motion is a pioneering work in leveraging both pressure data and linguistic priors for motion reconstruction, and the established MPL benchmark is the first benchmark for this novel motion capture task. Experiments show that our method generates high-fidelity, physically plausible motions, establishing a new state of the art for this task. The codes and benchmarks will be publicly released upon publication.
>
---
#### [replaced 042] ControlThinker: Unveiling Latent Semantics for Controllable Image Generation through Visual Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03596v2](https://arxiv.org/pdf/2506.03596v2)**

> **作者:** Feng Han; Yang Jiao; Shaoxiang Chen; Junhao Xu; Jingjing Chen; Yu-Gang Jiang
>
> **摘要:** The field of controllable image generation has seen significant advancements, with various architectures improving generation layout consistency with control signals. However, contemporary methods still face challenges in bridging the semantic gap between input text prompts with sparse semantics and the target images, often over-relying on low-level control signals to infer regional details. To address this challenge, we propose ControlThinker, a novel framework that employs a "comprehend-then-generate" paradigm. Firstly, by incentivizing the visual reasoning capability of a MLLM, latent semantics from control images are mined to enrich text prompts. This enriched semantic understanding then seamlessly aids in image generation without the need for additional complex modifications. To further tackle the uncertainty arising from the ambiguity of control images, we encourage broader exploration of reasoning trajectories and select the optimal one using a metric-based output reward model (ORM). Extensive experimental results demonstrate that ControlThinker effectively mitigates the semantic gap between raw text prompts and target images, resulting in improved visual quality and semantic consistency across a wide range of benchmarks. The code and models are available at https://github.com/Maplebb/ControlThinker.
>
---
#### [replaced 043] HiGFA: Hierarchical Guidance for Fine-grained Data Augmentation with Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12547v2](https://arxiv.org/pdf/2511.12547v2)**

> **作者:** Zhiguang Lu; Qianqian Xu; Peisong Wen; Siran Dai; Qingming Huang
>
> **摘要:** Generative diffusion models show promise for data augmentation. However, applying them to fine-grained tasks presents a significant challenge: ensuring synthetic images accurately capture the subtle, category-defining features critical for high fidelity. Standard approaches, such as text-based Classifier-Free Guidance (CFG), often lack the required specificity, potentially generating misleading examples that degrade fine-grained classifier performance. To address this, we propose Hierarchically Guided Fine-grained Augmentation (HiGFA). HiGFA leverages the temporal dynamics of the diffusion sampling process. It employs strong text and transformed contour guidance with fixed strengths in the early-to-mid sampling stages to establish overall scene, style, and structure. In the final sampling stages, HiGFA activates a specialized fine-grained classifier guidance and dynamically modulates the strength of all guidance signals based on prediction confidence. This hierarchical, confidence-driven orchestration enables HiGFA to generate diverse yet faithful synthetic images by intelligently balancing global structure formation with precise detail refinement. Experiments on several FGVC datasets demonstrate the effectiveness of HiGFA.
>
---
#### [replaced 044] STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13186v2](https://arxiv.org/pdf/2510.13186v2)**

> **作者:** Zhen Li; Xibin Jin; Guoliang Li; Shuai Wang; Miaowen Wen; Huseyin Arslan; Derrick Wing Kwan Ng; Chengzhong Xu
>
> **摘要:** Edge Gaussian splatting (EGS), which aggregates data from distributed clients and trains a global GS model at the edge server, is an emerging paradigm for scene reconstruction. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead.Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments unveil that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. It is found that the GS-oriented objective can be accurately predicted with low sampling ratios (e.g.,10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
>
---
#### [replaced 045] PointAD+: Learning Hierarchical Representations for Zero-shot 3D Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.03277v5](https://arxiv.org/pdf/2509.03277v5)**

> **作者:** Qihang Zhou; Shibo He; Jiangtao Yan; Wenchao Meng; Jiming Chen
>
> **备注:** Submitted to TPAMI
>
> **摘要:** In this paper, we aim to transfer CLIP's robust 2D generalization capabilities to identify 3D anomalies across unseen objects of highly diverse class semantics. To this end, we propose a unified framework to comprehensively detect and segment 3D anomalies by leveraging both point- and pixel-level information. We first design PointAD, which leverages point-pixel correspondence to represent 3D anomalies through their associated rendering pixel representations. This approach is referred to as implicit 3D representation, as it focuses solely on rendering pixel anomalies but neglects the inherent spatial relationships within point clouds. Then, we propose PointAD+ to further broaden the interpretation of 3D anomalies by introducing explicit 3D representation, emphasizing spatial abnormality to uncover abnormal spatial relationships. Hence, we propose G-aggregation to involve geometry information to enable the aggregated point representations spatially aware. To simultaneously capture rendering and spatial abnormality, PointAD+ proposes hierarchical representation learning, incorporating implicit and explicit anomaly semantics into hierarchical text prompts: rendering prompts for the rendering layer and geometry prompts for the geometry layer. A cross-hierarchy contrastive alignment is further introduced to promote the interaction between the rendering and geometry layers, facilitating mutual anomaly learning. Finally, PointAD+ integrates anomaly semantics from both layers to capture the generalized anomaly semantics. During the test, PointAD+ can integrate RGB information in a plug-and-play manner and further improve its detection performance. Extensive experiments demonstrate the superiority of PointAD+ in ZS 3D anomaly detection across unseen objects with highly diverse class semantics, achieving a holistic understanding of abnormality.
>
---
#### [replaced 046] PositionIC: Unified Position and Identity Consistency for Image Customization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.13861v4](https://arxiv.org/pdf/2507.13861v4)**

> **作者:** Junjie Hu; Tianyang Han; Kai Ma; Jialin Gao; Song Yang; Xianhua He; Junfeng Luo; Xiaoming Wei; Wenqiang Zhang
>
> **摘要:** Recent subject-driven image customization excels in fidelity, yet fine-grained instance-level spatial control remains an elusive challenge, hindering real-world applications. This limitation stems from two factors: a scarcity of scalable, position-annotated datasets, and the entanglement of identity and layout by global attention mechanisms. To this end, we introduce \modelname{}, a unified framework for high-fidelity, spatially controllable multi-subject customization. First, we present BMPDS, the first automatic data-synthesis pipeline for position-annotated multi-subject datasets, effectively providing crucial spatial supervision. Second, we design a lightweight, layout-aware diffusion framework that integrates a novel visibility-aware attention mechanism. This mechanism explicitly models spatial relationships via an NeRF-inspired volumetric weight regulation to effectively decouple instance-level spatial embeddings from semantic identity features, enabling precise, occlusion-aware placement of multiple subjects. Extensive experiments demonstrate \modelname{} achieves state-of-the-art performance on public benchmarks, setting new records for spatial precision and identity consistency. Our work represents a significant step towards truly controllable, high-fidelity image customization in multi-entity scenarios. Code and data will be publicly released.
>
---
#### [replaced 047] CascadedViT: Cascaded Chunk-FeedForward and Cascaded Group Attention Vision Transformer
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14111v2](https://arxiv.org/pdf/2511.14111v2)**

> **作者:** Srivathsan Sivakumar; Faisal Z. Qureshi
>
> **摘要:** Vision Transformers (ViTs) have demonstrated remarkable performance across a range of computer vision tasks; however, their high computational, memory, and energy demands hinder deployment on resource-constrained platforms. In this paper, we propose \emph{Cascaded-ViT (CViT)}, a lightweight and compute-efficient vision transformer architecture featuring a novel feedforward network design called \emph{Cascaded-Chunk Feed Forward Network (CCFFN)}. By splitting input features, CCFFN improves parameter and FLOP efficiency without sacrificing accuracy. Experiments on ImageNet-1K show that our \emph{CViT-XL} model achieves 75.5\% Top-1 accuracy while reducing FLOPs by 15\% and energy consumption by 3.3\% compared to EfficientViT-M5. Across various model sizes, the CViT family consistently exhibits the lowest energy consumption, making it suitable for deployment on battery-constrained devices such as mobile phones and drones. Furthermore, when evaluated using a new metric called \emph{Accuracy-Per-FLOP (APF)}, which quantifies compute efficiency relative to accuracy, CViT models consistently achieve top-ranking efficiency. Particularly, CViT-L is 2.2\% more accurate than EfficientViT-M2 while having comparable APF scores.
>
---
#### [replaced 048] End-to-End Visual Autonomous Parking via Control-Aided Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.11090v2](https://arxiv.org/pdf/2509.11090v2)**

> **作者:** Chao Chen; Shunyu Yao; Yuanwu He; Feng Tao; Ruojing Song; Yuliang Guo; Xinyu Huang; Chenxu Wu; Liu Ren; Chen Feng
>
> **摘要:** Precise parking requires an end-to-end system where perception adaptively provides policy-relevant details - especially in critical areas where fine control decisions are essential. End-to-end learning offers a unified framework by directly mapping sensor inputs to control actions, but existing approaches lack effective synergy between perception and control. Instead, we propose CAA-Policy, an end-to-end imitation learning system that allows control signal to guide the learning of visual attention via a novel Control-Aided Attention (CAA) mechanism. We train such an attention module in a self-supervised manner, using backpropagated gradients from the control outputs instead of from the training loss. This strategy encourages attention to focus on visual features that induce high variance in action outputs, rather than merely minimizing the training loss - a shift we demonstrate leads to a more robust and generalizable policy. To further strengthen the framework, CAA-Policy incorporates short-horizon waypoint prediction as an auxiliary task to improve temporal consistency of control outputs, a learnable motion prediction module to robustly track target slots over time, and a modified target tokenization scheme for more effective feature fusion. Extensive experiments in the CARLA simulator show that CAA-Policy consistently surpasses both the end-to-end learning baseline and the modular BEV segmentation + hybrid A* pipeline, achieving superior accuracy, robustness, and interpretability. Code and Collected Training datasets will be released. Code is released at https://github.com/ai4ce/CAAPolicy.
>
---
#### [replaced 049] Preventing Shortcut Learning in Medical Image Analysis through Intermediate Layer Knowledge Distillation from Specialist Teachers
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.17421v2](https://arxiv.org/pdf/2511.17421v2)**

> **作者:** Christopher Boland; Sotirios Tsaftaris; Sonia Dahdouh
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) https://melba-journal.org/2025:020
>
> **摘要:** Deep learning models are prone to learning shortcut solutions to problems using spuriously correlated yet irrelevant features of their training data. In high-risk applications such as medical image analysis, this phenomenon may prevent models from using clinically meaningful features when making predictions, potentially leading to poor robustness and harm to patients. We demonstrate that different types of shortcuts (those that are diffuse and spread throughout the image, as well as those that are localized to specific areas) manifest distinctly across network layers and can, therefore, be more effectively targeted through mitigation strategies that target the intermediate layers. We propose a novel knowledge distillation framework that leverages a teacher network fine-tuned on a small subset of task-relevant data to mitigate shortcut learning in a student network trained on a large dataset corrupted with a bias feature. Through extensive experiments on CheXpert, ISIC 2017, and SimBA datasets using various architectures (ResNet-18, AlexNet, DenseNet-121, and 3D CNNs), we demonstrate consistent improvements over traditional Empirical Risk Minimization, augmentation-based bias-mitigation, and group-based bias-mitigation approaches. In many cases, we achieve comparable performance with a baseline model trained on bias-free data, even on out-of-distribution test data. Our results demonstrate the practical applicability of our approach to real-world medical imaging scenarios where bias annotations are limited and shortcut features are difficult to identify a priori.
>
---
#### [replaced 050] Neural Collapse-Inspired Multi-Label Federated Learning under Label-Distribution Skew
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12544v3](https://arxiv.org/pdf/2509.12544v3)**

> **作者:** Can Peng; Yuyuan Liu; Yingyu Yang; Pramit Saha; Qianye Yang; J. Alison Noble
>
> **摘要:** Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy, yet it remains challenging as data distributions can be highly heterogeneous. These challenges are further amplified in multi-label scenarios, where data exhibit characteristics such as label co-occurrence, inter-label dependency, and discrepancies between local and global label relationships. While most existing FL studies focus on single-label classification, real-world applications, such as in medical imaging, involve multi-label data with highly skewed label distributions across clients. To address this important yet underexplored problem, we propose FedNCA-ML, a novel FL framework that aligns feature distributions across clients and learns discriminative, well-clustered representations inspired by Neural Collapse (NC) theory. NC describes an ideal latent-space geometry where each class's features collapse to their mean, forming a maximally separated simplex. To extend this theory to multi-label settings, we introduce a feature disentanglement module that extracts class-specific representations. The clustering of these disentangled features is guided by a shared NC-inspired structure, mitigating conflicts among client models caused by heterogeneous local data. Furthermore, we design regularisation losses to encourage compact and consistent feature clustering in the latent space. Experiments on four benchmark datasets under eight FL settings demonstrate the effectiveness of the proposed method, achieving improvements of up to 3.92% in class-wise AUC and 4.93% in class-wise F1 score.
>
---
#### [replaced 051] A Target-based Multi-LiDAR Multi-Camera Extrinsic Calibration System
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.16621v2](https://arxiv.org/pdf/2507.16621v2)**

> **作者:** Lorenzo Gentilini; Pierpaolo Serio; Valentina Donzella; Lorenzo Pollini
>
> **备注:** RiTA 2025 Accepted, 13 Pages, 6 Figures and 2 Tables
>
> **摘要:** Extrinsic Calibration represents the cornerstone of autonomous driving. Its accuracy plays a crucial role in the perception pipeline, as any errors can have implications for the safety of the vehicle. Modern sensor systems collect different types of data from the environment, making it harder to align the data. To this end, we propose a target-based extrinsic calibration system tailored for a multi-LiDAR and multi-camera sensor suite. This system enables cross-calibration between LiDARs and cameras with limited prior knowledge using a custom ChArUco board and a tailored nonlinear optimization method. We test the system with real-world data gathered in a warehouse. Results demonstrated the effectiveness of the proposed method, highlighting the feasibility of a unique pipeline tailored for various types of sensors.
>
---
#### [replaced 052] Studying Classifier(-Free) Guidance From a Classifier-Centric Perspective
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.10638v3](https://arxiv.org/pdf/2503.10638v3)**

> **作者:** Xiaoming Zhao; Alexander G. Schwing
>
> **备注:** v3: AAAI 2026; v2: added derivation details in Appendix A
>
> **摘要:** Classifier-free guidance has become a staple for conditional generation with denoising diffusion models. However, a comprehensive understanding of classifier-free guidance is still missing. In this work, we carry out an empirical study to provide a fresh perspective on classifier-free guidance. Concretely, instead of solely focusing on classifier-free guidance, we trace back to the root, i.e., classifier guidance, pinpoint the key assumption for the derivation, and conduct a systematic study to understand the role of the classifier. On 1D data, we find that both classifier guidance and classifier-free guidance achieve conditional generation by pushing the denoising diffusion trajectories away from decision boundaries, i.e., areas where conditional information is usually entangled and is hard to learn. To validate this classifier-centric perspective on high-dimensional data, we assess whether a flow-matching postprocessing step that is designed to narrow the gap between a pre-trained diffusion model's learned distribution and the real data distribution, especially near decision boundaries, can improve the performance. Experiments on various datasets verify our classifier-centric understanding.
>
---
#### [replaced 053] OMGSR: You Only Need One Mid-timestep Guidance for Real-World Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.08227v2](https://arxiv.org/pdf/2508.08227v2)**

> **作者:** Zhiqiang Wu; Zhaomang Sun; Tong Zhou; Bingtao Fu; Ji Cong; Yitong Dong; Huaqi Zhang; Xuan Tang; Mingsong Chen; Xian Wei
>
> **摘要:** Denoising Diffusion Probabilistic Models (DDPMs) show promising potential in one-step Real-World Image Super-Resolution (Real-ISR). Current one-step Real-ISR methods typically inject the low-quality (LQ) image latent representation at the start or end timestep of the DDPM scheduler. Recent studies have begun to note that the LQ image latent and the pre-trained noisy latent representations are intuitively closer at a mid-timestep. However, a quantitative analysis of these latent representations remains lacking. Considering these latent representations can be decomposed into signal and noise, we propose a method based on the Signal-to-Noise Ratio (SNR) to pre-compute an average optimal mid-timestep for injection. To better approximate the pre-trained noisy latent representation, we further introduce the Latent Representation Refinement (LRR) loss via a LoRA-enhanced VAE encoder. We also fine-tune the backbone of the DDPM-based generative model using LoRA to perform one-step denoising at the average optimal mid-timestep. Based on these components, we present OMGSR, a GAN-based Real-ISR framework that employs a DDPM-based generative model as the generator and a DINOv3-ConvNeXt model with multi-level discriminator heads as the discriminator. We also propose the DINOv3-ConvNeXt DISTS (Dv3CD) loss, which is enhanced for structural perception at varying resolutions. Within the OMGSR framework, we develop OMGSR-S based on SD2.1-base. An ablation study confirms that our pre-computation strategy and LRR loss significantly improve the baseline. Comparative studies demonstrate that OMGSR-S achieves state-of-the-art performance across multiple metrics. Code is available at \hyperlink{Github}{https://github.com/wuer5/OMGSR}.
>
---
#### [replaced 054] Sim-DETR: Unlock DETR for Temporal Sentence Grounding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23867v2](https://arxiv.org/pdf/2509.23867v2)**

> **作者:** Jiajin Tang; Zhengxuan Wei; Yuchen Zhu; Cheng Shi; Guanbin Li; Liang Lin; Sibei Yang
>
> **备注:** This work is accepted by ICCV 2025
>
> **摘要:** Temporal sentence grounding aims to identify exact moments in a video that correspond to a given textual query, typically addressed with detection transformer (DETR) solutions. However, we find that typical strategies designed to enhance DETR do not improve, and may even degrade, its performance in this task. We systematically analyze and identify the root causes of this abnormal behavior: (1) conflicts between queries from similar target moments and (2) internal query conflicts due to the tension between global semantics and local localization. Building on these insights, we propose a simple yet powerful baseline, Sim-DETR, which extends the standard DETR with two minor modifications in the decoder layers: (1) constraining self-attention between queries based on their semantic and positional overlap and (2) adding query-to-frame alignment to bridge the global and local contexts. Experiments demonstrate that Sim-DETR unlocks the full potential of DETR for temporal sentence grounding, offering a strong baseline for future research.
>
---
#### [replaced 055] Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.09532v3](https://arxiv.org/pdf/2506.09532v3)**

> **作者:** Shuai Wang; Zhenhua Liu; Jiaheng Wei; Xuanwu Yin; Dong Li; Emad Barsoum
>
> **备注:** v3: fix typos, add data scaling exp
>
> **摘要:** We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks.
>
---
#### [replaced 056] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2505.19536v3](https://arxiv.org/pdf/2505.19536v3)**

> **作者:** Jintao Tong; Wenwei Jin; Pengda Qin; Anqi Li; Yixiong Zou; Yuhong Li; Yuhua Li; Ruixuan Li
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Large vision-language models (LVLMs) excel at multimodal understanding but suffer from high computational costs due to redundant vision tokens. Existing pruning methods typically rely on single-layer attention scores to rank and prune redundant visual tokens to solve this inefficiency. However, as the interaction between tokens and layers is complicated, this raises a basic question: Is such a simple single-layer criterion sufficient to identify redundancy? To answer this question, we rethink the emergence of redundant visual tokens from a fundamental perspective: information flow, which models the interaction between tokens and layers by capturing how information moves between tokens across layers. We find (1) the CLS token acts as an information relay, which can simplify the complicated flow analysis; (2) the redundancy emerges progressively and dynamically via layer-wise attention concentration; and (3) relying solely on attention scores from single layers can lead to contradictory redundancy identification. Based on this, we propose FlowCut, an information-flow-aware pruning framework, mitigating the insufficiency of the current criterion for identifying redundant tokens and better aligning with the model's inherent behaviors. Extensive experiments show that FlowCut achieves superior results, outperforming SoTA by 1.6% on LLaVA-1.5-7B with 88.9% token reduction, and by 4.3% on LLaVA-NeXT-7B with 94.4% reduction, delivering 3.2x speed-up in the prefilling stage. Our code is available at https://github.com/TungChintao/FlowCut
>
---
#### [replaced 057] Multiview point cloud registration with anisotropic and space-varying localization noise
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2201.00708v2](https://arxiv.org/pdf/2201.00708v2)**

> **作者:** Denis Fortun; Etienne Baudrier; Fabian Zwettler; Markus Sauer; Sylvain Faisan
>
> **摘要:** In this paper, we address the problem of registering multiple point clouds corrupted with high anisotropic localization noise. Our approach follows the widely used framework of Gaussian mixture model (GMM) reconstruction with an expectation-maximization (EM) algorithm. Existing methods are based on an implicit assumption of space-invariant isotropic Gaussian noise. However, this assumption is violated in practice in applications such as single molecule localization microscopy (SMLM). To address this issue, we propose to introduce an explicit localization noise model that decouples shape modeling with the GMM from noise handling. We design a stochastic EM algorithm that considers noise-free data as a latent variable, with closed-form solutions at each EM step. The first advantage of our approach is to handle space-variant and anisotropic Gaussian noise with arbitrary covariances. The second advantage is to leverage the explicit noise model to impose prior knowledge about the noise that may be available from physical sensors. We show on various simulated data that our noise handling strategy improves significantly the robustness to high levels of anisotropic noise. We also demonstrate the performance of our method on real SMLM data.
>
---
#### [replaced 058] ConMamba: Contrastive Vision Mamba for Plant Disease Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03213v2](https://arxiv.org/pdf/2506.03213v2)**

> **作者:** Abdullah Al Mamun; Miaohua Zhang; David Ahmedt-Aristizabal; Zeeshan Hayder; Mohammad Awrangjeb
>
> **摘要:** Plant Disease Detection (PDD) is a key aspect of precision agriculture. However, existing deep learning methods often rely on extensively annotated datasets, which are time-consuming and costly to generate. Self-supervised Learning (SSL) offers a promising alternative by exploiting the abundance of unlabeled data. However, most existing SSL approaches suffer from high computational costs due to convolutional neural networks or transformer-based architectures. Additionally, they struggle to capture long-range dependencies in visual representation and rely on static loss functions that fail to align local and global features effectively. To address these challenges, we propose ConMamba, a novel SSL framework specially designed for PDD. ConMamba integrates the Vision Mamba Encoder (VME), which employs a bidirectional State Space Model (SSM) to capture long-range dependencies efficiently. Furthermore, we introduce a dual-level contrastive loss with dynamic weight adjustment to optimize local-global feature alignment. Experimental results on three benchmark datasets demonstrate that ConMamba significantly outperforms state-of-the-art methods across multiple evaluation metrics. This provides an efficient and robust solution for PDD.
>
---
#### [replaced 059] MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.03931v2](https://arxiv.org/pdf/2501.03931v2)**

> **作者:** Yuechen Zhang; Yaoyang Liu; Bin Xia; Bohao Peng; Zexin Yan; Eric Lo; Jiaya Jia
>
> **备注:** ICCV 2025, It is best viewed in Acrobat. Project Page: https://julianjuaner.github.io/projects/MagicMirror/
>
> **摘要:** We present MagicMirror, a framework for generating identity-preserved videos with cinematic-level quality and dynamic motion. While recent advances in video diffusion models have shown impressive capabilities in text-to-video generation, maintaining consistent identity while producing natural motion remains challenging. Previous methods either require person-specific fine-tuning or struggle to balance identity preservation with motion diversity. Built upon Video Diffusion Transformers, our method introduces three key components: (1) a dual-branch facial feature extractor that captures both identity and structural features, (2) a lightweight cross-modal adapter with Conditioned Adaptive Normalization for efficient identity integration, and (3) a two-stage training strategy combining synthetic identity pairs with video data. Extensive experiments demonstrate that MagicMirror effectively balances identity consistency with natural motion, outperforming existing methods across multiple metrics while requiring minimal parameters added. The code and model will be made publicly available.
>
---
#### [replaced 060] DMAT: An End-to-End Framework for Joint Atmospheric Turbulence Mitigation and Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.04323v3](https://arxiv.org/pdf/2507.04323v3)**

> **作者:** Paul Hill; Zhiming Liu; Alin Achim; Dave Bull; Nantheera Anantrasirichai
>
> **备注:** Accepted to WACV2026
>
> **摘要:** Atmospheric Turbulence (AT) degrades the clarity and accuracy of surveillance imagery, posing challenges not only for visualization quality but also for object classification and scene tracking. Deep learning-based methods have been proposed to improve visual quality, but spatio-temporal distortions remain a significant issue. Although deep learning-based object detection performs well under normal conditions, it struggles to operate effectively on sequences distorted by atmospheric turbulence. In this paper, we propose a novel framework that learns to compensate for distorted features while simultaneously improving visualization and object detection. This end-to-end training strategy leverages and exchanges knowledge of low-level distorted features in the AT mitigator with semantic features extracted in the object detector. Specifically, in the AT mitigator a 3D Mamba-based structure is used to handle the spatio-temporal displacements and blurring caused by turbulence. Optimization is achieved through back-propagation in both the AT mitigator and object detector. Our proposed DMAT outperforms state-of-the-art AT mitigation and object detection systems up to a 15% improvement on datasets corrupted by generated turbulence.
>
---
#### [replaced 061] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.13837v5](https://arxiv.org/pdf/2504.13837v5)**

> **作者:** Yang Yue; Zhiqi Chen; Rui Lu; Andrew Zhao; Zhaokai Wang; Yang Yue; Shiji Song; Gao Huang
>
> **备注:** 31 pages, 27 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
>
---
#### [replaced 062] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.18716v2](https://arxiv.org/pdf/2405.18716v2)**

> **作者:** Chaitat Utintu; Pinaki Nath Chowdhury; Aneeshan Sain; Subhadeep Koley; Ayan Kumar Bhunia; Yi-Zhe Song
>
> **备注:** Project Page: \url{https://chaitron.github.io/SketchDeco/}
>
> **摘要:** We introduce SketchDeco, a training-free approach to sketch colourisation that bridges the gap between professional design needs and intuitive, region-based control. Our method empowers artists to use simple masks and colour palettes for precise spatial and chromatic specification, avoiding both the tediousness of manual assignment and the ambiguity of text-based prompts. We reformulate this task as a novel, training-free composition problem. Our core technical contribution is a guided latent-space blending process: we first leverage diffusion inversion to precisely ``paint'' user-defined colours into specified regions, and then use a custom self-attention mechanism to harmoniously blend these local edits with a globally consistent base image. This ensures both local colour fidelity and global harmony without requiring any model fine-tuning. Our system produces high-quality results in 15--20 inference steps on consumer GPUs, making professional-quality, controllable colourisation accessible.
>
---
#### [replaced 063] Teacher Encoder-Student Decoder Denoising Guided Segmentation Network for Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2501.12104v4](https://arxiv.org/pdf/2501.12104v4)**

> **作者:** Shixuan Song; Hao Chen; Shu Hu; Xin Wang; Jinrong Hu; Xi Wu
>
> **摘要:** Visual anomaly detection is a highly challenging task, often categorized as a one-class classification and segmentation problem. Recent studies have demonstrated that the student-teacher (S-T) framework effectively addresses this challenge. However, most S-T frameworks rely solely on pre-trained teacher networks to guide student networks in learning multi-scale similar features, overlooking the potential of the student networks to enhance learning through multi-scale feature fusion. In this study, we propose a novel model named PFADSeg, which integrates a pre-trained teacher network, a denoising student network with multi-scale feature fusion, and a guided anomaly segmentation network into a unified framework. By adopting a unique teacher-encoder and student-decoder denoising mode, the model improves the student network's ability to learn from teacher network features. Furthermore, an adaptive feature fusion mechanism is introduced to train a self-supervised segmentation network that synthesizes anomaly masks autonomously, significantly increasing detection performance. Rigorous evaluations on the widely-used MVTec AD dataset demonstrate that PFADSeg exhibits excellent performance, achieving an image-level AUC of 98.9%, a pixel-level mean precision of 76.4%, and an instance-level mean precision of 78.7%.
>
---
#### [replaced 064] InfoScale: Unleashing Training-free Variable-scaled Image Generation via Effective Utilization of Information
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.01421v3](https://arxiv.org/pdf/2509.01421v3)**

> **作者:** Guohui Zhang; Jiangtong Tan; Linjiang Huang; Zhonghang Yuan; Mingde Yao; Jie Huang; Feng Zhao
>
> **摘要:** Diffusion models (DMs) have become dominant in visual generation but suffer performance drop when tested on resolutions that differ from the training scale, whether lower or higher. In fact, the key challenge in generating variable-scale images lies in the differing amounts of information across resolutions, which requires information conversion procedures to be varied for generating variable-scaled images. In this paper, we investigate the issues of three critical aspects in DMs for a unified analysis in variable-scaled generation: dilated convolution, attention mechanisms, and initial noise. Specifically, 1) dilated convolution in DMs for the higher-resolution generation loses high-frequency information. 2) Attention for variable-scaled image generation struggles to adjust the information aggregation adaptively. 3) The spatial distribution of information in the initial noise is misaligned with variable-scaled image. To solve the above problems, we propose \textbf{InfoScale}, an information-centric framework for variable-scaled image generation by effectively utilizing information from three aspects correspondingly. For information loss in 1), we introduce Progressive Frequency Compensation module to compensate for high-frequency information lost by dilated convolution in higher-resolution generation. For information aggregation inflexibility in 2), we introduce Adaptive Information Aggregation module to adaptively aggregate information in lower-resolution generation and achieve an effective balance between local and global information in higher-resolution generation. For information distribution misalignment in 3), we design Noise Adaptation module to re-distribute information in initial noise for variable-scaled generation. Our method is plug-and-play for DMs and extensive experiments demonstrate the effectiveness in variable-scaled image generation.
>
---
#### [replaced 065] PriorDrive: Enhancing Online HD Mapping with Unified Vector Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.05352v4](https://arxiv.org/pdf/2409.05352v4)**

> **作者:** Shuang Zeng; Xinyuan Chang; Xinran Liu; Yujian Yuan; Shiyi Liang; Zheng Pan; Mu Xu; Xing Wei
>
> **备注:** AAAI 2026; Code: https://github.com/MIV-XJTU/PriorDrive
>
> **摘要:** High-Definition Maps (HD maps) are essential for the precise navigation and decision-making of autonomous vehicles, yet their creation and upkeep present significant cost and timeliness challenges. The online construction of HD maps using on-board sensors has emerged as a promising solution; however, these methods can be impeded by incomplete data due to occlusions and inclement weather, while their performance in distant regions remains unsatisfying. This paper proposes PriorDrive to address these limitations by directly harnessing the power of various vectorized prior maps, significantly enhancing the robustness and accuracy of online HD map construction. Our approach integrates a variety of prior maps uniformly, such as OpenStreetMap's Standard Definition Maps (SD maps), outdated HD maps from vendors, and locally constructed maps from historical vehicle data. To effectively integrate such prior information into online mapping models, we introduce a Hybrid Prior Representation (HPQuery) that standardizes the representation of diverse map elements. We further propose a Unified Vector Encoder (UVE), which employs fused prior embedding and a dual encoding mechanism to encode vector data. To improve the UVE's generalizability and performance, we propose a segment-level and point-level pre-training strategy that enables the UVE to learn the prior distribution of vector data. Through extensive testing on the nuScenes, Argoverse 2 and OpenLane-V2, we demonstrate that PriorDrive is highly compatible with various online mapping models and substantially improves map prediction capabilities. The integration of prior maps through PriorDrive offers a robust solution to the challenges of single-perception data, paving the way for more reliable autonomous vehicle navigation. Code is available at https://github.com/MIV-XJTU/PriorDrive.
>
---
#### [replaced 066] Unsupervised and Source-Free Ranking of Biomedical Segmentation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.00450v2](https://arxiv.org/pdf/2503.00450v2)**

> **作者:** Joshua Talks; Kevin Marchesini; Luca Lumetti; Federico Bolelli; Anna Kreshuk
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** Model transfer presents a solution to the challenges of segmentation in the biomedical community, where the immense cost of data annotation is a major bottleneck in the use of deep learning. At the same time, hundreds of models get trained on biomedical data, submitted to challenges, and posted in model zoos and repositories. A major hurdle to wider adoption of pre-trained models lies in the lack of methods for best model selection. While such methods have been proposed for classification models, semantic and instance segmentation model ranking remain largely unaddressed, especially in a practically important setting where no labels are available on the target dataset. Similarly, if unsupervised domain adaptation is used, practitioners are faced with the task of selecting the best adapted model without target domain labels. Building on previous work linking model generalisation and consistency under perturbation, we propose the first unsupervised and source-free transferability estimator for semantic and instance segmentation tasks. We evaluate on multiple segmentation problems across biomedical imaging, finding a strong correlation between the rankings based on our estimator and rankings based on target dataset performance.
>
---
#### [replaced 067] VLCE: A Knowledge-Enhanced Framework for Image Description in Disaster Assessment
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.21609v4](https://arxiv.org/pdf/2509.21609v4)**

> **作者:** Md. Mahfuzur Rahman; Kishor Datta Gupta; Marufa Kamal; Fahad Rahman; Sunzida Siddique; Ahmed Rafi Hasan; Mohd Ariful Haque; Roy George
>
> **备注:** 30 pages, 40 figures, 3 algorithms
>
> **摘要:** The processes of classification and segmentation utilizing artificial intelligence play a vital role in the automation of disaster assessments. However, contemporary VLMs produce details that are inadequately aligned with the objectives of disaster assessment, primarily due to their deficiency in domain knowledge and the absence of a more refined descriptive process. This research presents the Vision Language Caption Enhancer (VLCE), a dedicated multimodal framework aimed at integrating external semantic knowledge from ConceptNet and WordNet to improve the captioning process. The objective is to produce disaster-specific descriptions that effectively convert raw visual data into actionable intelligence. VLCE utilizes two separate architectures: a CNN-LSTM model that incorporates a ResNet50 backbone, pretrained on EuroSat for satellite imagery (xBD dataset), and a Vision Transformer developed for UAV imagery (RescueNet dataset). In various architectural frameworks and datasets, VLCE exhibits a consistent advantage over baseline models such as LLaVA and QwenVL. Our optimal configuration reaches an impressive 95.33\% on InfoMetIC for UAV imagery while also demonstrating strong performance across satellite imagery. The proposed framework signifies a significant transition from basic visual classification to the generation of comprehensive situational intelligence, demonstrating immediate applicability for implementation in real-time disaster assessment systems.
>
---
#### [replaced 068] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.15580v3](https://arxiv.org/pdf/2511.15580v3)**

> **作者:** Sifan Zhou; Yichao Cao; Jiahao Nie; Yuqian Fu; Ziyu Zhao; Xiaobo Lu; Shuo Wang
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** 3D single object tracking (SOT) in LiDAR point clouds is a critical task in computer vision and autonomous driving. Despite great success having been achieved, the inherent sparsity of point clouds introduces a dual-redundancy challenge that limits existing trackers: (1) vast spatial redundancy from background noise impairs accuracy, and (2) informational redundancy within the foreground hinders efficiency. To tackle these issues, we propose CompTrack, a novel end-to-end framework that systematically eliminates both forms of redundancy in point clouds. First, CompTrack incorporates a Spatial Foreground Predictor (SFP) module to filter out irrelevant background noise based on information entropy, addressing spatial redundancy. Subsequently, its core is an Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module that eliminates the informational redundancy within the foreground. Theoretically grounded in low-rank approximation, this module leverages an online SVD analysis to adaptively compress the redundant foreground into a compact and highly informative set of proxy tokens. Extensive experiments on KITTI, nuScenes and Waymo datasets demonstrate that CompTrack achieves top-performing tracking performance with superior efficiency, running at a real-time 90 FPS on a single RTX 3090 GPU.
>
---
#### [replaced 069] SGDFuse: SAM-Guided Diffusion for High-Fidelity Infrared and Visible Image Fusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.05264v4](https://arxiv.org/pdf/2508.05264v4)**

> **作者:** Xiaoyang Zhang; jinjiang Li; Guodong Fan; Yakun Ju; Linwei Fan; Jun Liu; Alex C. Kot
>
> **备注:** Submitted to Information Fusion
>
> **摘要:** Infrared and visible image fusion (IVIF) aims to combine the thermal radiation information from infrared images with the rich texture details from visible images to enhance perceptual capabilities for downstream visual tasks. However, existing methods often fail to preserve key targets due to a lack of deep semantic understanding of the scene, while the fusion process itself can also introduce artifacts and detail loss, severely compromising both image quality and task performance. To address these issues, this paper proposes SGDFuse, a conditional diffusion model guided by the Segment Anything Model (SAM), to achieve high-fidelity and semantically-aware image fusion. The core of our method is to utilize high-quality semantic masks generated by SAM as explicit priors to guide the optimization of the fusion process via a conditional diffusion model. Specifically, the framework operates in a two-stage process: it first performs a preliminary fusion of multi-modal features, and then utilizes the semantic masks from SAM jointly with the preliminary fused image as a condition to drive the diffusion model's coarse-to-fine denoising generation. This ensures the fusion process not only has explicit semantic directionality but also guarantees the high fidelity of the final result. Extensive experiments demonstrate that SGDFuse achieves state-of-the-art performance in both subjective and objective evaluations, as well as in its adaptability to downstream tasks, providing a powerful solution to the core challenges in image fusion. The code of SGDFuse is available at https://github.com/boshizhang123/SGDFuse.
>
---
#### [replaced 070] ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2510.08562v2](https://arxiv.org/pdf/2510.08562v2)**

> **作者:** Zhiyu Zheng; Shaoyu Chen; Haoran Yin; Xinbang Zhang; Jialv Zou; Xinggang Wang; Qian Zhang; Lefei Zhang
>
> **摘要:** End-to-end autonomous driving (E2EAD) systems, which learn to predict future trajectories directly from sensor data, are fundamentally challenged by the inherent spatio-temporal imbalance of trajectory data. This imbalance creates a significant optimization burden, causing models to learn spurious correlations instead of robust driving logic, while also prioritizing uncertain, distant predictions, thereby compromising immediate safety. To address these issues, we propose ResAD, a novel Normalized Residual Trajectory Modeling framework. Instead of predicting the future trajectory directly, our approach reframes and simplifies the learning task by predicting the residual deviation from a deterministic inertial reference. This inertial reference serves as a strong physical prior, compelling the model to move beyond simple pattern-matching and instead focus its capacity on learning the necessary, context-driven deviations (e.g., traffic rules, obstacles) from this default, inertially-guided path. To mitigate the optimization imbalance caused by uncertain, long-term horizons, ResAD further incorporates Point-wise Normalization of the predicted residual. This technique re-weights the optimization objective, preventing large-magnitude errors associated with distant, uncertain waypoints from dominating the learning signal. On the NAVSIM v1 and v2 benchmarks, ResAD achieves state-of-the-art results of 88.8 PDMS and 85.5 EPDMS with only two denoising steps, demonstrating that ResAD significantly simplifies the learning task and improves planning performance. The code will be released to facilitate further research.
>
---
#### [replaced 071] DiffBreak: Is Diffusion-Based Purification Robust?
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.16598v4](https://arxiv.org/pdf/2411.16598v4)**

> **作者:** Andre Kassis; Urs Hengartner; Yaoliang Yu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Diffusion-based purification (DBP) has become a cornerstone defense against adversarial examples (AEs), regarded as robust due to its use of diffusion models (DMs) that project AEs onto the natural data manifold. We refute this core claim, theoretically proving that gradient-based attacks effectively target the DM rather than the classifier, causing DBP's outputs to align with adversarial distributions. This prompts a reassessment of DBP's robustness, accrediting it two critical factors: inaccurate gradients and improper evaluation protocols that test only a single random purification of the AE. We show that when accounting for stochasticity and resubmission risk, DBP collapses. To support this, we introduce DiffBreak, the first reliable toolkit for differentiation through DBP, eliminating gradient mismatches that previously further inflated robustness estimates. We also analyze the current defense scheme used for DBP where classification relies on a single purification, pinpointing its inherent invalidity. We provide a statistically grounded majority-vote (MV) alternative that aggregates predictions across multiple purified copies, showing partial but meaningful robustness gain. We then propose a novel adaptation of an optimization method against deepfake watermarking, crafting systemic perturbations that defeat DBP even under MV, challenging DBP's viability.
>
---
#### [replaced 072] Minimax Multi-Target Conformal Prediction with Applications to Imaging Inverse Problems
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13533v2](https://arxiv.org/pdf/2511.13533v2)**

> **作者:** Jeffrey Wen; Rizwan Ahmad; Philip Schniter
>
> **摘要:** In ill-posed imaging inverse problems, uncertainty quantification remains a fundamental challenge, especially in safety-critical applications. Recently, conformal prediction has been used to quantify the uncertainty that the inverse problem contributes to downstream tasks like image classification, image quality assessment, fat mass quantification, etc. While existing works handle only a scalar estimation target, practical applications often involve multiple targets. In response, we propose an asymptotically minimax approach to multi-target conformal prediction that provides tight prediction intervals while ensuring joint marginal coverage. We then outline how our minimax approach can be applied to multi-metric blind image quality assessment, multi-task uncertainty quantification, and multi-round measurement acquisition. Finally, we numerically demonstrate the benefits of our minimax method, relative to existing multi-target conformal prediction methods, using both synthetic and magnetic resonance imaging (MRI) data. Code is available at https://github.com/jwen307/multi_target_minimax.
>
---
#### [replaced 073] Roadside Monocular 3D Detection Prompted by 2D Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2404.01064v4](https://arxiv.org/pdf/2404.01064v4)**

> **作者:** Yechi Ma; Yanan Li; Wei Hua; Shu Kong
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Roadside monocular 3D detection requires detecting objects of predefined classes in an RGB frame and predicting their 3D attributes, such as bird's-eye-view (BEV) locations. It has broad applications in traffic control, vehicle-vehicle communication, and vehicle-infrastructure cooperative perception. To address this task, we introduce Promptable 3D Detector (Pro3D), a novel detector design that leverages 2D detections as prompts. We build our Pro3D upon two key insights. First, compared to a typical 3D detector, a 2D detector is ``easier'' to train due to fewer loss terms and performs significantly better at localizing objects w.r.t 2D metrics. Second, once 2D detections precisely locate objects in the image, a 3D detector can focus on lifting these detections into 3D BEV, especially when fixed camera pose or scene geometry provide an informative prior. To encode and incorporate 2D detections, we explore three methods: (a) concatenating features from both 2D and 3D detectors, (b) attentively fusing 2D and 3D detector features, and (c) encoding properties of predicted 2D bounding boxes \{$x$, $y$, width, height, label\} and attentively fusing them with the 3D detector feature. Interestingly, the third method significantly outperforms the others, underscoring the effectiveness of 2D detections as prompts that offer precise object targets and allow the 3D detector to focus on lifting them into 3D. Pro3D is adaptable for use with a wide range of 2D and 3D detectors with minimal modifications. Comprehensive experiments demonstrate that our Pro3D significantly enhances existing methods, achieving state-of-the-art results on two contemporary benchmarks.
>
---
#### [replaced 074] ImAgent: A Unified Multimodal Agent Framework for Test-Time Scalable Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.11483v2](https://arxiv.org/pdf/2511.11483v2)**

> **作者:** Kaishen Wang; Ruibo Chen; Tong Zheng; Heng Huang
>
> **备注:** 12 pages, 5 tables, 6 figures
>
> **摘要:** Recent text-to-image (T2I) models have made remarkable progress in generating visually realistic and semantically coherent images. However, they still suffer from randomness and inconsistency with the given prompts, particularly when textual descriptions are vague or underspecified. Existing approaches, such as prompt rewriting, best-of-N sampling, and self-refinement, can mitigate these issues but usually require additional modules and operate independently, hindering test-time scaling efficiency and increasing computational overhead. In this paper, we introduce ImAgent, a training-free unified multimodal agent that integrates reasoning, generation, and self-evaluation within a single framework for efficient test-time scaling. Guided by a policy controller, multiple generation actions dynamically interact and self-organize to enhance image fidelity and semantic alignment without relying on external models. Extensive experiments on image generation and editing tasks demonstrate that ImAgent consistently improves over the backbone and even surpasses other strong baselines where the backbone model fails, highlighting the potential of unified multimodal agents for adaptive and efficient image generation under test-time scaling.
>
---
#### [replaced 075] REArtGS++: Generalizable Articulation Reconstruction with Temporal Geometry Constraint via Planar Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17059v2](https://arxiv.org/pdf/2511.17059v2)**

> **作者:** Di Wu; Liu Liu; Anran Huang; Yuyan Liu; Qiaojun Yu; Shaofan Liu; Liangtu Song; Cewu Lu
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Articulated objects are pervasive in daily environments, such as drawers and refrigerators. Towards their part-level surface reconstruction and joint parameter estimation, REArtGS introduces a category-agnostic approach using multi-view RGB images at two different states. However, we observe that REArtGS still struggles with screw-joint or multi-part objects and lacks geometric constraints for unseen states. In this paper, we propose REArtGS++, a novel method towards generalizable articulated object reconstruction with temporal geometry constraint and planar Gaussian splatting. We first model a decoupled screw motion for each joint without type prior, and jointly optimize part-aware Gaussians with joint parameters through part motion blending. To introduce time-continuous geometric constraint for articulated modeling, we encourage Gaussians to be planar and propose a temporally consistent regularization between planar normal and depth through Taylor first-order expansion. Extensive experiments on both synthetic and real-world articulated objects demonstrate our superiority in generalizable part-level surface reconstruction and joint parameter estimation, compared to existing approaches. Project Site: https://sites.google.com/view/reartgs2/home.
>
---
#### [replaced 076] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06299v3](https://arxiv.org/pdf/2511.06299v3)**

> **作者:** Haoqin Hong; Ding Fan; Fubin Dou; Zhi-Li Zhou; Haoran Sun; Congcong Zhu; Jingrun Chen
>
> **备注:** Accepted by AAAI-26
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS), an explicit scene representation technique, has shown significant promise for dynamic novel-view synthesis from monocular video input. However, purely data-driven 3DGS often struggles to capture the diverse physics-driven motion patterns in dynamic scenes. To fill this gap, we propose Physics-Informed Deformable Gaussian Splatting (PIDG), which treats each Gaussian particle as a Lagrangian material point with time-varying constitutive parameters and is supervised by 2D optical flow via motion projection. Specifically, we adopt static-dynamic decoupled 4D decomposed hash encoding to reconstruct geometry and motion efficiently. Subsequently, we impose the Cauchy momentum residual as a physics constraint, enabling independent prediction of each particle's velocity and constitutive stress via a time-evolving material field. Finally, we further supervise data fitting by matching Lagrangian particle flow to camera-compensated optical flow, which accelerates convergence and improves generalization. Experiments on a custom physics-driven dataset as well as on standard synthetic and real-world datasets demonstrate significant gains in physical consistency and monocular dynamic reconstruction quality.
>
---
#### [replaced 077] Find Them All: Unveiling MLLMs for Versatile Person Re-identification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.06908v2](https://arxiv.org/pdf/2508.06908v2)**

> **作者:** Jinhao Li; Zijian Chen; Lirong Deng; Guangtao Zhai; Changbo Wang
>
> **摘要:** Person re-identification (ReID) aims to retrieve images of a target person from the gallery set, with wide applications in medical rehabilitation and public security. However, traditional person ReID models are typically uni-modal, resulting in limited generalizability across heterogeneous data modalities. Recently, the emergence of multi-modal large language models (MLLMs) has shown a promising avenue for addressing this issue. Despite this potential, existing methods merely regard MLLMs as feature extractors or caption generators, leaving their capabilities in person ReID tasks largely unexplored. To bridge this gap, we introduce a novel benchmark for \underline{\textbf{V}}ersatile \underline{\textbf{P}}erson \underline{\textbf{Re}}-\underline{\textbf{ID}}entification, termed VP-ReID. The benchmark includes 257,310 multi-modal queries and gallery images, covering ten diverse person ReID tasks. In addition, we propose two task-oriented evaluation schemes for MLLM-based person ReID. Extensive experiments demonstrate the impressive versatility, effectiveness, and interpretability of MLLMs in various person ReID tasks. Nevertheless, they also have limitations in handling a few modalities, particularly thermal and infrared data. We hope that VP-ReID can facilitate the community in developing more robust and generalizable cross-modal foundation models for person ReID.
>
---
#### [replaced 078] Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.04726v2](https://arxiv.org/pdf/2507.04726v2)**

> **作者:** Raz Lapid; Almog Dubin
>
> **备注:** Accepted at RDS @ AAAI 2026
>
> **摘要:** Text-to-image diffusion models achieve high-fidelity image generation from natural language prompts. ControlNets extend these models by enabling conditioning on structural inputs (e.g., edge maps, depth, pose), providing fine-grained control over outputs. Yet their reliance on large, publicly scraped datasets and community fine-tuning makes them vulnerable to data poisoning. We introduce a model-poisoning attack that embeds a covert backdoor into a ControlNet, causing it to produce attacker-specified content when exposed to visual triggers, without textual prompts. Experiments show that poisoning only 1% of the fine-tuning corpus yields a 90-98% attack success rate, while 5% further strengthens the backdoor, all while preserving normal generation quality. To mitigate this risk, we propose clean fine-tuning (CFT): freezing the diffusion backbone and fine-tuning only the ControlNet on a sanitized dataset with a reduced learning rate. CFT lowers attack success rates on held-out data. These results expose a critical security weakness in open-source, ControlNet-guided diffusion pipelines and demonstrate that CFT offers a practical defense for responsible synthetic-data pipelines.
>
---
#### [replaced 079] QGait: Toward Accurate Quantization for Gait Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.13859v2](https://arxiv.org/pdf/2405.13859v2)**

> **作者:** Senmao Tian; Haoyu Gao; Gangyi Hong; Shuyun Wang; JingJie Wang; Xin Yu; Shunli Zhang
>
> **备注:** Accepted as an oral presentation at IJCB 2025
>
> **摘要:** Existing deep learning methods have made significant progress in gait recognition. Quantization can facilitate the application of gait models as a model-agnostic general compression technique. Typically, appearance-based models binarize inputs into silhouette sequences. However, mainstream quantization methods prioritize minimizing task loss over quantization error, which is detrimental to gait recognition with binarized inputs. To address this, we propose a differentiable soft quantizer, which better simulates the gradient of the round function during backpropagation. This enables the network to learn from subtle input perturbations. However, our theoretical analysis and empirical studies reveal that directly applying the soft quantizer can hinder network convergence. We addressed this issue by adopting a two-stage training strategy, introducing a soft quantizer during the fine-tuning phase. However, in the first stage of training, we observed a significant change in the output distribution of different samples in the feature space compared to the full-precision network. It is this change that led to a loss in performance. Based on this, we propose an Inter-class Distance-guided Calibration (IDC) strategy to preserve the relative distance between the embeddings of samples with different labels. Extensive experiments validate the effectiveness of our approach, demonstrating state-of-the-art accuracy across various settings and datasets.
>
---
#### [replaced 080] AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03100v3](https://arxiv.org/pdf/2508.03100v3)**

> **作者:** Yogesh Kulkarni; Pooyan Fazli
>
> **摘要:** Multimodal reasoning over long-horizon video is challenging due to the need for precise spatiotemporal fusion and alignment across modalities. While recent methods such as Group Relative Policy Optimization (GRPO) have shown promise in this domain, they suffer from three key limitations: (1) data inefficiency from their on-policy design, (2) a vanishing advantage problem, where identical or near-identical rewards within a group eliminate the learning signal by producing zero-valued advantages, and (3) uniform credit assignment that fails to emphasize critical reasoning steps. We introduce $\textbf{AVATAR}$ ($\textbf{A}$udio-$\textbf{V}$ideo $\textbf{A}$gen$\textbf{t}$ for $\textbf{A}$lignment and $\textbf{R}$easoning), a framework that addresses these limitations through two core components: (1) an off-policy training architecture that improves sample efficiency and resolves vanishing advantages by reusing past experiences with greater reward diversity, and (2) Temporal Advantage Shaping (TAS), a novel credit assignment strategy that upweights key reasoning phases during learning. $\textbf{AVATAR}$ achieves strong performance across various benchmarks, outperforming the Qwen2.5-Omni baseline by $\mathbf{+5.4}$ on MMVU, $\mathbf{+4.9}$ on OmniBench, and $\mathbf{+4.5}$ on Video-Holmes, while demonstrating $\textbf{$5$$\times$ sample efficiency}$, requiring $80\%$ fewer generated completions to reach target performance.
>
---
#### [replaced 081] Advancing Autonomous Driving: DepthSense with Radar and Spatial Attention
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2109.05265v4](https://arxiv.org/pdf/2109.05265v4)**

> **作者:** Muhamamd Ishfaq Hussain; Zubia Naz; Muhammad Aasim Rafique; Moongu Jeon
>
> **摘要:** Depth perception is crucial for spatial understanding and has traditionally been achieved through stereoscopic imaging. However, the precision of depth estimation using stereoscopic methods depends on the accurate calibration of binocular vision sensors. Monocular cameras, while more accessible, often suffer from reduced accuracy, especially under challenging imaging conditions. Optical sensors, too, face limitations in adverse environments, leading researchers to explore radar technology as a reliable alternative. Although radar provides coarse but accurate signals, its integration with fine-grained monocular camera data remains underexplored. In this research, we propose DepthSense, a novel radar-assisted monocular depth enhancement approach. DepthSense employs an encoder-decoder architecture, a Radar Residual Network, feature fusion with a spatial attention mechanism, and an ordinal regression layer to deliver precise depth estimations. We conducted extensive experiments on the nuScenes dataset to validate the effectiveness of DepthSense. Our methodology not only surpasses existing approaches in quantitative performance but also reduces parameter complexity and inference times. Our findings demonstrate that DepthSense represents a significant advancement over traditional stereo methods, offering a robust and efficient solution for depth estimation in autonomous driving. By leveraging the complementary strengths of radar and monocular camera data, DepthSense sets a new benchmark in the field, paving the way for more reliable and accurate spatial perception systems.
>
---
#### [replaced 082] MultiCrafter: High-Fidelity Multi-Subject Generation via Disentangled Attention and Identity-Aware Preference Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21953v2](https://arxiv.org/pdf/2509.21953v2)**

> **作者:** Tao Wu; Yibo Jiang; Yehao Lu; Zhizhong Wang; Zeyi Huang; Zequn Qin; Xi Li
>
> **备注:** Project Page: https://wutao-cs.github.io/MultiCrafter/
>
> **摘要:** Multi-subject image generation aims to synthesize user-provided subjects in a single image while preserving subject fidelity, ensuring prompt consistency, and aligning with human aesthetic preferences. Existing In-Context-Learning based methods are limited by their highly coupled training paradigm. These methods attempt to achieve both high subject fidelity and multi-dimensional human preference alignment within a single training stage, relying on a single, indirect reconstruction loss, which is difficult to simultaneously satisfy both these goals. To address this, we propose MultiCrafter, a framework that decouples this task into two distinct training stages. First, in a pre-training stage, we introduce an explicit positional supervision mechanism that effectively resolves attention bleeding and drastically enhances subject fidelity. Second, in a post-training stage, we propose Identity-Preserving Preference Optimization, a novel online reinforcement learning framework. We feature a scoring mechanism to accurately assess multi-subject fidelity based on the Hungarian matching algorithm, which allows the model to optimize for aesthetics and prompt alignment while ensuring subject fidelity achieved in the first stage. Experiments validate that our decoupling framework significantly improves subject fidelity while aligning with human preferences better.
>
---
#### [replaced 083] DSOcc: Leveraging Depth Awareness and Semantic Aid to Boost Camera-Based 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.20951v4](https://arxiv.org/pdf/2505.20951v4)**

> **作者:** Naiyu Fang; Zheyuan Zhou; Kang Wang; Ruibo Li; Lemiao Qiu; Shuyou Zhang; Zhe Wang; Guosheng Lin
>
> **摘要:** Camera-based 3D semantic occupancy prediction offers an efficient and cost-effective solution for perceiving surrounding scenes in autonomous driving. However, existing works rely on explicit occupancy state inference, leading to numerous incorrect feature assignments, and insufficient samples restrict the learning of occupancy class inference. To address these challenges, we propose leveraging \textbf{D}epth awareness and \textbf{S}emantic aid to boost camera-based 3D semantic \textbf{Occ}upancy prediction (\textbf{DSOcc}). We jointly perform occupancy state and occupancy class inference, where soft occupancy confidence is calculated by non-learning method and multiplied with image features to make voxels aware of depth, enabling adaptive implicit occupancy state inference. Instead of enhancing feature learning, we directly utilize well-trained image semantic segmentation and fuse multiple frames with their occupancy probabilities to aid occupancy class inference, thereby enhancing robustness. Experimental results demonstrate that DSOcc achieves state-of-the-art performance on the SemanticKITTI dataset among camera-based methods and achieves competitive performance on the SSCBench-KITTI-360 and Occ3D-nuScenes datasets. Code will be released on github.
>
---
#### [replaced 084] InstantViR: Real-Time Video Inverse Problem Solver with Distilled Diffusion Prior
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14208v2](https://arxiv.org/pdf/2511.14208v2)**

> **作者:** Weimin Bai; Suzhe Xu; Yiwei Ren; Jinhua Hao; Ming Sun; Wenzheng Chen; He Sun
>
> **摘要:** Video inverse problems are fundamental to streaming, telepresence, and AR/VR, where high perceptual quality must coexist with tight latency constraints. Diffusion-based priors currently deliver state-of-the-art reconstructions, but existing approaches either adapt image diffusion models with ad hoc temporal regularizers - leading to temporal artifacts - or rely on native video diffusion models whose iterative posterior sampling is far too slow for real-time use. We introduce InstantViR, an amortized inference framework for ultra-fast video reconstruction powered by a pre-trained video diffusion prior. We distill a powerful bidirectional video diffusion model (teacher) into a causal autoregressive student that maps a degraded video directly to its restored version in a single forward pass, inheriting the teacher's strong temporal modeling while completely removing iterative test-time optimization. The distillation is prior-driven: it only requires the teacher diffusion model and known degradation operators, and does not rely on externally paired clean/noisy video data. To further boost throughput, we replace the video-diffusion backbone VAE with a high-efficiency LeanVAE via an innovative teacher-space regularized distillation scheme, enabling low-latency latent-space processing. Across streaming random inpainting, Gaussian deblurring and super-resolution, InstantViR matches or surpasses the reconstruction quality of diffusion-based baselines while running at over 35 FPS on NVIDIA A100 GPUs, achieving up to 100 times speedups over iterative video diffusion solvers. These results show that diffusion-based video reconstruction is compatible with real-time, interactive, editable, streaming scenarios, turning high-quality video restoration into a practical component of modern vision systems.
>
---
#### [replaced 085] Prompt-guided Disentangled Representation for Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21783v4](https://arxiv.org/pdf/2509.21783v4)**

> **作者:** Tianci Wu; Guangming Zhu; Jiang Lu; Siyuan Wang; Ning Wang; Nuoye Xiong; Zhang Liang
>
> **摘要:** Action recognition is a fundamental task in video understanding. Existing methods typically extract unified features to process all actions in one video, which makes it challenging to model the interactions between different objects in multi-action scenarios. To alleviate this issue, we explore disentangling any specified actions from complex scenes as an effective solution. In this paper, we propose Prompt-guided Disentangled Representation for Action Recognition (ProDA), a novel framework that disentangles any specified actions from a multi-action scene. ProDA leverages Spatio-temporal Scene Graphs (SSGs) and introduces Dynamic Prompt Module (DPM) to guide a Graph Parsing Neural Network (GPNN) in generating action-specific representations. Furthermore, we design a video-adapted GPNN that aggregates information using dynamic weights. Experiments in video action recognition demonstrate the effectiveness of our approach when compared with the state-of-the-art methods. Our code can be found in https://github.com/iamsnaping/ProDA.git
>
---
#### [replaced 086] Algorithms Trained on Normal Chest X-rays Can Predict Health Insurance Types
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.11030v3](https://arxiv.org/pdf/2511.11030v3)**

> **作者:** Chi-Yu Chen; Rawan Abulibdeh; Arash Asgari; Leo Anthony Celi; Deirdre Goode; Hassan Hamidi; Laleh Seyyed-Kalantari; Ned McCague; Thomas Sounack; Po-Chih Kuo
>
> **备注:** Submitting to MIDL 2026
>
> **摘要:** Artificial intelligence is revealing what medicine never intended to encode. Deep vision models, trained on chest X-rays, can now detect not only disease but also invisible traces of social inequality. In this study, we show that state-of-the-art architectures (DenseNet121, SwinV2-B, MedMamba) can predict a patient's health insurance type, a strong proxy for socioeconomic status, from normal chest X-rays with significant accuracy (AUC around 0.67 on MIMIC-CXR-JPG, 0.68 on CheXpert). The signal persists even when age, race, and sex are controlled for, and remains detectable when the model is trained exclusively on a single racial group. Patch-based occlusion reveals that the signal is diffuse rather than localized, embedded in the upper and mid-thoracic regions. This suggests that deep networks may be internalizing subtle traces of clinical environments, equipment differences, or care pathways; learning socioeconomic segregation itself. These findings challenge the assumption that medical images are neutral biological data. By uncovering how models perceive and exploit these hidden social signatures, this work reframes fairness in medical AI: the goal is no longer only to balance datasets or adjust thresholds, but to interrogate and disentangle the social fingerprints embedded in clinical data itself.
>
---
#### [replaced 087] Restore Text First, Enhance Image Later: Two-Stage Scene Text Image Super-Resolution with Glyph Structure Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21590v2](https://arxiv.org/pdf/2510.21590v2)**

> **作者:** Minxing Luo; Linlong Fan; Wang Qiushi; Ge Wu; Yiyan Luo; Yuhang Yu; Jinwei Chen; Yaxing Wang; Qingnan Fan; Jian Yang
>
> **摘要:** Current image super-resolution methods show strong performance on natural images but distort text, creating a fundamental trade-off between image quality and textual readability. To address this, we introduce TIGER (Text-Image Guided supEr-Resolution), a novel two-stage framework that breaks this trade-off through a "text-first, image-later" paradigm. TIGER explicitly decouples glyph restoration from image enhancement: it first reconstructs precise text structures and uses them to guide full-image super-resolution. This ensures high fidelity and readability. To support comprehensive training and evaluation, we present the UZ-ST (UltraZoom-Scene Text) dataset, the first Chinese scene text dataset with extreme zoom. Extensive experiments show TIGER achieves state-of-the-art performance, enhancing readability and image quality.
>
---
#### [replaced 088] Unreal Robotics Lab: A High-Fidelity Robotics Simulator with Advanced Physics and Rendering
- **分类: cs.RO; cs.CV; cs.GR; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.14135v2](https://arxiv.org/pdf/2504.14135v2)**

> **作者:** Jonathan Embley-Riches; Jianwei Liu; Simon Julier; Dimitrios Kanoulas
>
> **摘要:** High-fidelity simulation is essential for robotics research, enabling safe and efficient testing of perception, control, and navigation algorithms. However, achieving both photorealistic rendering and accurate physics modeling remains a challenge. This paper presents a novel simulation framework, the Unreal Robotics Lab (URL), that integrates the advanced rendering capabilities of the Unreal Engine with MuJoCo's high-precision physics simulation. Our approach enables realistic robotic perception while maintaining accurate physical interactions, facilitating benchmarking and dataset generation for vision-based robotics applications. The system supports complex environmental effects, such as smoke, fire, and water dynamics, which are critical to evaluating robotic performance under adverse conditions. We benchmark visual navigation and SLAM methods within our framework, demonstrating its utility for testing real-world robustness in controlled yet diverse scenarios. By bridging the gap between physics accuracy and photorealistic rendering, our framework provides a powerful tool for advancing robotics research and sim-to-real transfer. Our open-source framework is available at https://unrealroboticslab.github.io/.
>
---
#### [replaced 089] Automatic Multi-View X-Ray/CT Registration Using Bone Substructure Contours
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.13292v2](https://arxiv.org/pdf/2506.13292v2)**

> **作者:** Roman Flepp; Leon Nissen; Bastian Sigrist; Arend Nieuwland; Nicola Cavalcanti; Philipp Fürnstahl; Thomas Dreher; Lilian Calvet
>
> **备注:** This paper was accepted to IPCAI 2025. The Project Webpage is: https://rflepp.github.io/BoneSubstructureContours2D3DRegistration/
>
> **摘要:** Purpose: Accurate intraoperative X-ray/CT registration is essential for surgical navigation in orthopedic procedures. However, existing methods struggle with consistently achieving sub-millimeter accuracy, robustness under broad initial pose estimates or need manual key-point annotations. This work aims to address these challenges by proposing a novel multi-view X-ray/CT registration method for intraoperative bone registration. Methods: The proposed registration method consists of a multi-view, contour-based iterative closest point (ICP) optimization. Unlike previous methods, which attempt to match bone contours across the entire silhouette in both imaging modalities, we focus on matching specific subcategories of contours corresponding to bone substructures. This leads to reduced ambiguity in the ICP matches, resulting in a more robust and accurate registration solution. This approach requires only two X-ray images and operates fully automatically. Additionally, we contribute a dataset of 5 cadaveric specimens, including real X-ray images, X-ray image poses and the corresponding CT scans. Results: The proposed registration method is evaluated on real X-ray images using mean reprojection error (mRPD). The method consistently achieves sub-millimeter accuracy with a mRPD 0.67mm compared to 5.35mm by a commercial solution requiring manual intervention. Furthermore, the method offers improved practical applicability, being fully automatic. Conclusion: Our method offers a practical, accurate, and efficient solution for multi-view X-ray/CT registration in orthopedic surgeries, which can be easily combined with tracking systems. By improving registration accuracy and minimizing manual intervention, it enhances intraoperative navigation, contributing to more accurate and effective surgical outcomes in computer-assisted surgery (CAS).
>
---
#### [replaced 090] VCE: Safe Autoregressive Image Generation via Visual Contrast Exploitation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16986v2](https://arxiv.org/pdf/2509.16986v2)**

> **作者:** Feng Han; Chao Gong; Zhipeng Wei; Jingjing Chen; Yu-Gang Jiang
>
> **摘要:** Recently, autoregressive image generation models have wowed audiences with their remarkable capability in creating surprisingly realistic images. Models such as GPT-4o and LlamaGen can not only produce images that faithfully mimic renowned artistic styles like Ghibli, Van Gogh, or Picasso, but also potentially generate Not-Safe-For-Work (NSFW) content, raising significant concerns regarding copyright infringement and ethical use. Despite these concerns, methods to safeguard autoregressive text-to-image models remain underexplored. Previous concept erasure methods, primarily designed for diffusion models that operate in denoising latent space, are not directly applicable to autoregressive models that generate images token by token. To address this critical gap, we propose Visual Contrast Exploitation (VCE), a novel framework comprising: (1) an innovative contrastive image pair construction paradigm that precisely decouples unsafe concepts from their associated content semantics, and (2) a sophisticated DPO-based training approach that enhances the model's ability to identify and leverage visual contrastive features from image pairs, enabling precise concept erasure. Our comprehensive experiments across three challenging tasks-artist style erasure, explicit content erasure, and object removal-demonstrate that our method effectively secures the model, achieving state-of-the-art results while erasing unsafe concepts and maintaining the integrity of unrelated safe concepts. The code and models are available at https://github.com/Maplebb/VCE.
>
---
#### [replaced 091] Faster and Better 3D Splatting via Group Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.07608v3](https://arxiv.org/pdf/2412.07608v3)**

> **作者:** Chengbo Wang; Guozheng Ma; Yifei Xue; Yizhen Lao
>
> **备注:** Accepted to ICCV 2025. Code is available at https://github.com/Chengbo-Wang/3DGS-with-Group-Training
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, demonstrating remarkable capability in high-fidelity scene reconstruction through its Gaussian primitive representations. However, the computational overhead induced by the massive number of primitives poses a significant bottleneck to training efficiency. To overcome this challenge, we propose Group Training, a simple yet effective strategy that organizes Gaussian primitives into manageable groups, optimizing training efficiency and improving rendering quality. This approach shows universal compatibility with existing 3DGS frameworks, including vanilla 3DGS and Mip-Splatting, consistently achieving accelerated training while maintaining superior synthesis quality. Extensive experiments reveal that our straightforward Group Training strategy achieves up to 30\% faster convergence and improved rendering quality across diverse scenarios. Project Website: https://chengbo-wang.github.io/3DGS-with-Group-Training/
>
---
#### [replaced 092] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.06659v3](https://arxiv.org/pdf/2506.06659v3)**

> **作者:** Wenhao Yao; Zhenxin Li; Shiyi Lan; Zi Wang; Xinglong Sun; Jose M. Alvarez; Zuxuan Wu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Autonomous vehicles must navigate safely in complex driving environments. Imitating a single expert trajectory, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each. However, they face optimization challenges in precisely selecting the best option from thousands of candidates and distinguishing subtle but safety-critical differences, especially in rare and challenging scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, with 83.02 Driving Score and 60.00 Success Rate on the Bench2Drive benchmark, demonstrating superior planning capabilities in various driving scenarios.
>
---
#### [replaced 093] In-Situ Tweedie Discrete Diffusion Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.01047v2](https://arxiv.org/pdf/2510.01047v2)**

> **作者:** Xiao Li; Jiaqi Zhang; Shuxiang Zhang; Tianshui Chen; Liang Lin; Guangrun Wang
>
> **摘要:** While diffusion models excel at generating continuous data such as images, adapting them to discrete tasks has relied on indirect approaches that either operate in continuous embedding spaces or use token masking mechanisms, both of which deviate from modeling the true discrete data distribution that can be theoretically guaranteed by Tweedie's formula. We propose in-situ Tweedie Discrete Diffusion (TDD), a framework that performs diffusion guaranteed by Tweedie's formula directly within the discrete one-hot space, hence "in-situ." Unlike prior methods that diffuse continuous embeddings or mask tokens, TDD directly corrupts one-hot vectors with Gaussian noise and performs iterative denoising through a timestep-conditioned cross-entropy objective rather than mean-squared-error reconstruction. At each denoising step, the model predicts class probabilities, applies argmax to obtain discrete predictions, converts them to one-hot vectors, and feeds them into the next iteration with progressively reduced noise. This process naturally unifies discriminative classification and generative modeling under a single framework. Experiments demonstrate that TDD achieves strong performance on both image classification and text generation tasks, with extensive ablation studies confirming the effectiveness of each design component. Our work establishes a principled approach to discrete diffusion that preserves the core characteristics of diffusion models while operating natively in discrete space.
>
---
#### [replaced 094] ERANet: Edge Replacement Augmentation for Semi-Supervised Meniscus Segmentation with Prototype Consistency Alignment and Conditional Self-Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.07331v2](https://arxiv.org/pdf/2502.07331v2)**

> **作者:** Siyue Li; Yongcheng Yao; Junru Zhong; Shutian Zhao; Fan Xiao; Tim-Yun Michael Ong; Ki-Wai Kevin Ho; James F. Griffith; Yudong Zhang; Shuihua Wang; Jin Hong; Weitian Chen
>
> **摘要:** Manual segmentation is labor-intensive, and automatic segmentation remains challenging due to the inherent variability in meniscal morphology, partial volume effects, and low contrast between the meniscus and surrounding tissues. To address these challenges, we propose ERANet, an innovative semi-supervised framework for meniscus segmentation that effectively leverages both labeled and unlabeled images through advanced augmentation and learning strategies. ERANet integrates three key components: edge replacement augmentation (ERA), prototype consistency alignment (PCA), and a conditional self-training (CST) strategy within a mean teacher architecture. ERA introduces anatomically relevant perturbations by simulating meniscal variations, ensuring that augmentations align with the structural context. PCA enhances segmentation performance by aligning intra-class features and promoting compact, discriminative feature representations, particularly in scenarios with limited labeled data. CST improves segmentation robustness by iteratively refining pseudo-labels and mitigating the impact of label noise during training. Together, these innovations establish ERANet as a robust and scalable solution for meniscus segmentation, effectively addressing key barriers to practical implementation. We validated ERANet comprehensively on 3D Double Echo Steady State (DESS) and 3D Fast/Turbo Spin Echo (FSE/TSE) MRI sequences. The results demonstrate the superior performance of ERANet compared to state-of-the-art methods. The proposed framework achieves reliable and accurate segmentation of meniscus structures, even when trained on minimal labeled data. Extensive ablation studies further highlight the synergistic contributions of ERA, PCA, and CST, solidifying ERANet as a transformative solution for semi-supervised meniscus segmentation in medical imaging.
>
---
#### [replaced 095] FreeInv: Free Lunch for Improving DDIM Inversion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.23035v2](https://arxiv.org/pdf/2503.23035v2)**

> **作者:** Yuxiang Bao; Huijie Liu; Xun Gao; Huan Fu; Guoliang Kang
>
> **摘要:** Naive DDIM inversion process usually suffers from a trajectory deviation issue, i.e., the latent trajectory during reconstruction deviates from the one during inversion. To alleviate this issue, previous methods either learn to mitigate the deviation or design cumbersome compensation strategy to reduce the mismatch error, exhibiting substantial time and computation cost. In this work, we present a nearly free-lunch method (named FreeInv) to address the issue more effectively and efficiently. In FreeInv, we randomly transform the latent representation and keep the transformation the same between the corresponding inversion and reconstruction time-step. It is motivated from a statistical perspective that an ensemble of DDIM inversion processes for multiple trajectories yields a smaller trajectory mismatch error on expectation. Moreover, through theoretical analysis and empirical study, we show that FreeInv performs an efficient ensemble of multiple trajectories. FreeInv can be freely integrated into existing inversion-based image and video editing techniques. Especially for inverting video sequences, it brings more significant fidelity and efficiency improvements. Comprehensive quantitative and qualitative evaluation on PIE benchmark and DAVIS dataset shows that FreeInv remarkably outperforms conventional DDIM inversion, and is competitive among previous state-of-the-art inversion methods, with superior computation efficiency.
>
---
#### [replaced 096] CUPID: Generative 3D Reconstruction via Joint Object and Pose Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20776v2](https://arxiv.org/pdf/2510.20776v2)**

> **作者:** Binbin Huang; Haobin Duan; Yiqun Zhao; Zibo Zhao; Yi Ma; Shenghua Gao
>
> **备注:** project page at https://cupid3d.github.io
>
> **摘要:** We introduce Cupid, a generative 3D reconstruction framework that jointly models the full distribution over both canonical objects and camera poses. Our two-stage flow-based model first generates a coarse 3D structure and 2D-3D correspondences to estimate the camera pose robustly. Conditioned on this pose, a refinement stage injects pixel-aligned image features directly into the generative process, marrying the rich prior of a generative model with the geometric fidelity of reconstruction. This strategy achieves exceptional faithfulness, outperforming state-of-the-art reconstruction methods by over 3 dB PSNR and 10% in Chamfer Distance. As a unified generative model that decouples the object and camera pose, Cupid naturally extends to multi-view and scene-level reconstruction tasks without requiring post-hoc optimization or fine-tuning.
>
---
#### [replaced 097] Leverage Cross-Attention for End-to-End Open-Vocabulary Panoptic Reconstruction
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2501.01119v2](https://arxiv.org/pdf/2501.01119v2)**

> **作者:** Xuan Yu; Yuxuan Xie; Yili Liu; Haojian Lu; Rong Xiong; Yiyi Liao; Yue Wang
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Open-vocabulary panoptic reconstruction offers comprehensive scene understanding, enabling advances in embodied robotics and photorealistic simulation. In this paper, we propose PanopticRecon++, an end-to-end method that formulates panoptic reconstruction through a novel cross-attention perspective. This perspective models the relationship between 3D instances (as queries) and the scene's 3D embedding field (as keys) through their attention map. Unlike existing methods that separate the optimization of queries and keys or overlook spatial proximity, PanopticRecon++ introduces learnable 3D Gaussians as instance queries. This formulation injects 3D spatial priors to preserve proximity while maintaining end-to-end optimizability. Moreover, this query formulation facilitates the alignment of 2D open-vocabulary instance IDs across frames by leveraging optimal linear assignment with instance masks rendered from the queries. Additionally, we ensure semantic-instance segmentation consistency by fusing query-based instance segmentation probabilities with semantic probabilities in a novel panoptic head supervised by a panoptic loss. During training, the number of instance query tokens dynamically adapts to match the number of objects. PanopticRecon++ shows competitive performance in terms of 3D and 2D segmentation and reconstruction performance on both simulation and real-world datasets, and demonstrates a user case as a robot simulator. Our project website is at: https://yuxuan1206.github.io/panopticrecon_pp/
>
---
#### [replaced 098] PRISM-Bench: A Benchmark of Puzzle-Based Visual Tasks with CoT Error Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23594v3](https://arxiv.org/pdf/2510.23594v3)**

> **作者:** Yusu Qian; Cheng Wan; Chao Jia; Yinfei Yang; Qingyu Zhao; Zhe Gan
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable progress on vision-language tasks, yet their reasoning processes remain sometimes unreliable. We introduce PRISM-Bench, a benchmark of puzzle-based visual challenges designed to evaluate not only whether models can solve problems, but how their reasoning unfolds. Unlike prior evaluations that measure only final-answer accuracy, PRISM-Bench introduces a diagnostic task: given a visual puzzle and a step-by-step chain-of-thought (CoT) containing exactly one error, models must identify the first incorrect step. This setting enables fine-grained assessment of logical consistency, error detection, and visual reasoning. The puzzles in PRISM-Bench require multi-step symbolic, geometric, and analogical reasoning, resisting shortcuts based on superficial pattern matching. Evaluations across state-of-the-art MLLMs reveal a persistent gap between fluent generation and faithful reasoning: models that produce plausible CoTs often fail to locate simple logical faults. By disentangling answer generation from reasoning verification, PRISM-Bench offers a sharper lens on multimodal reasoning competence and underscores the need for diagnostic evaluation protocols in the development of trustworthy MLLMs.
>
---
#### [replaced 099] A Training-Free Style-aligned Image Generation with Scale-wise Autoregressive Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.06144v2](https://arxiv.org/pdf/2504.06144v2)**

> **作者:** Jihun Park; Jongmin Gim; Kyoungmin Lee; Minseok Oh; Minwoo Choi; Jaeyeul Kim; Woo Chool Park; Sunghoon Im
>
> **备注:** 18 pages, 15 figures
>
> **摘要:** We present a training-free style-aligned image generation method that leverages a scale-wise autoregressive model. While large-scale text-to-image (T2I) models, particularly diffusion-based methods, have demonstrated impressive generation quality, they often suffer from style misalignment across generated image sets and slow inference speeds, limiting their practical usability. To address these issues, we propose three key components: initial feature replacement to ensure consistent background appearance, pivotal feature interpolation to align object placement, and dynamic style injection, which reinforces style consistency using a schedule function. Unlike previous methods requiring fine-tuning or additional training, our approach maintains fast inference while preserving individual content details. Extensive experiments show that our method achieves generation quality comparable to competing approaches, significantly improves style alignment, and delivers inference speeds over six times faster than the fastest model.
>
---
#### [replaced 100] Resilient Contrastive Pre-training under Non-Stationary Drift
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2502.07620v3](https://arxiv.org/pdf/2502.07620v3)**

> **作者:** Xiaoyu Yang; Jie Lu; En Yu; Wei Duan
>
> **备注:** 17pages, 3 figures
>
> **摘要:** The remarkable success of large-scale contrastive pre-training has been largely driven by by vast yet static datasets. However, as the scaling paradigm evolves, this paradigm encounters a fundamental challenge when applied to dynamic data streams characterized by concept drift - unpredictable changes in the underlying data distribution. This paper aims to advance robust pre-training under such non-stationary environments. We begin by revealing that conventional contrastive pre-training methods are highly susceptible to concept drift, resulting in significant substantial bias and instability within the learned feature representations. To systematically analyze these effects, we develop a structural causal model that elucidates how drift acts as a confounder, distorting the learned representations. Based on these causal insights, we propose Resilient Contrastive Pre-training (RCP), a novel method that incorporates causal intervention. RCP formulates a causally-informed objective to mitigate drift-induced biases through targeted interventions. The method is designed for simple and scalable implementation and exhibits notable adaptability, promoting robust and autonomous pre-training on non-stationary data. Comprehensive experiments across various downstream tasks consistently demonstrate that RCP effectively alleviates the detrimental impact of concept drift, yielding more resilient and generalizable representations.
>
---
#### [replaced 101] Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.22174v3](https://arxiv.org/pdf/2503.22174v3)**

> **作者:** Jialun Pei; Zhangjun Zhou; Diandian Guo; Zhixi Li; Jing Qin; Bo Du; Pheng-Ann Heng
>
> **摘要:** Intraoperative bleeding in laparoscopic surgery causes rapid obscuration of the operative field to hinder the surgical process and increases the risk of postoperative complications. Intelligent detection of bleeding areas can quantify the blood loss to assist decision-making, while locating bleeding points helps surgeons quickly identify the source of bleeding and achieve hemostasis in time to improve surgical success rates. To fill the benchmark gap, we first construct a real-world laparoscopic surgical bleeding detection dataset, named SurgBlood, comprising 5,330 frames from 95 surgical video clips with bleeding region and point annotations. Accordingly, we develop a dual-task synergistic online detector called BlooDet, enabling simultaneous detection of bleeding regions and points in laparoscopic surgery. The baseline embraces a dual-branch bidirectional guid- ance design based on Segment Anything Model 2. The mask branch detects bleeding regions through adaptive edge and point prompt embeddings, while the point branch leverages mask memory to induce bleeding point memory modeling and captures point motion direction via inter-frame optical flow. By coupled bidirectional guidance, our framework explores spatial-temporal correlations while exploiting memory modeling to infer current bleeding status. Extensive experiments indicate that our method outperforms 13 counterparts in bleeding detection.
>
---
#### [replaced 102] Prompt Guiding Multi-Scale Adaptive Sparse Representation-driven Network for Low-Dose CT MAR
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.19687v2](https://arxiv.org/pdf/2504.19687v2)**

> **作者:** Baoshun Shi; Bing Chen; Shaolei Zhang; Huazhu Fu; Zhanli Hu
>
> **摘要:** Low-dose CT (LDCT) is capable of reducing X-ray radiation exposure, but it will potentially degrade image quality, even yields metal artifacts at the case of metallic implants. For simultaneous LDCT reconstruction and metal artifact reduction (LDMAR), existing deep learning-based efforts face two main limitations: i) the network design neglects multi-scale and within-scale information; ii) training a distinct model for each dose necessitates significant storage space for multiple doses. To fill these gaps, we propose a prompt guiding multi-scale adaptive sparse representation-driven network, abbreviated as PMSRNet, for LDMAR task. Specifically, we construct PMSRNet inspired from multi-scale sparsifying frames, and it can simultaneously employ within-scale characteristics and cross-scale complementarity owing to an elaborated prompt guiding scale-adaptive threshold generator (PSATG) and a built multi-scale coefficient fusion module (MSFuM). The PSATG can adaptively capture multiple contextual information to generate more faithful thresholds, achieved by fusing features from local, regional, and global levels. Furthermore, we elaborate a model interpretable dual domain LDMAR framework called PDuMSRNet, and train single model with a prompt guiding strategy for multiple dose levels. We build a prompt guiding module, whose input contains dose level, metal mask and input instance, to provide various guiding information, allowing a single model to accommodate various CT dose settings. Extensive experiments at various dose levels demonstrate that the proposed methods outperform the state-of-the-art LDMAR methods.
>
---
#### [replaced 103] Interpretable and Testable Vision Features via Sparse Autoencoders
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.06755v2](https://arxiv.org/pdf/2502.06755v2)**

> **作者:** Samuel Stevens; Wei-Lun Chao; Tanya Berger-Wolf; Yu Su
>
> **备注:** Main text is 10 pages with 7 figures
>
> **摘要:** To truly understand vision models, we must not only interpret their learned features but also validate these interpretations through controlled experiments. While earlier work offers either rich semantics or direct control, few post-hoc tools supply both in a single, model-agnostic procedure. We use sparse autoencoders (SAEs) to bridge this gap; each sparse feature comes with real-image exemplars that reveal its meaning and a decoding vector that can be manipulated to probe its influence on downstream task behavior. By applying our method to widely-used pre-trained vision models, we reveal meaningful differences in the semantic abstractions learned by different pre-training objectives. We then show that a single SAE trained on frozen ViT activations supports patch-level causal edits across tasks (classification and segmentation) all without retraining the ViT or task heads. These qualitative, falsifiable demonstrations position SAEs as a practical bridge between concept discovery and causal probing of vision models. We provide code, demos and models on our project website: https://osu-nlp-group.github.io/saev.
>
---
#### [replaced 104] UniREditBench: A Unified Reasoning-based Image Editing Benchmark
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.01295v2](https://arxiv.org/pdf/2511.01295v2)**

> **作者:** Feng Han; Yibin Wang; Chenglin Li; Zheming Liang; Dianyi Wang; Yang Jiao; Zhipeng Wei; Chao Gong; Cheng Jin; Jingjing Chen; Jiaqi Wang
>
> **备注:** Project page: https://maplebb.github.io/UniREditBench
>
> **摘要:** Recent advances in multi-modal generative models have driven substantial improvements in image editing. However, current generative models still struggle with handling diverse and complex image editing tasks that require implicit reasoning, underscoring the need for a comprehensive benchmark to systematically assess their performance across various reasoning scenarios. Existing benchmarks primarily focus on single-object attribute transformation in realistic scenarios, which, while effective, encounter two key challenges: (1) they largely overlook multi-object interactions as well as game-world scenarios that involve human-defined rules, which are common in real-life applications; (2) they only rely on textual references to evaluate the generated images, potentially leading to systematic misjudgments, especially in complex reasoning scenarios. To this end, this work proposes UniREditBench, a unified benchmark for reasoning-based image editing evaluation. It comprises 2,700 meticulously curated samples, covering both real- and game-world scenarios across 8 primary dimensions and 18 sub-dimensions. To improve evaluation reliability, we introduce multimodal dual-reference evaluation, providing both textual and ground-truth image references for each sample assessment. Furthermore, we design an automated multi-scenario data synthesis pipeline and construct UniREdit-Data-100K, a large-scale synthetic dataset with high-quality chain-of-thought (CoT) reasoning annotations. We fine-tune Bagel on this dataset and develop UniREdit-Bagel, demonstrating substantial improvements in both in-domain and out-of-distribution settings. Through thorough benchmarking of both open-source and closed-source image editing models, we reveal their strengths and weaknesses across various aspects.
>
---
#### [replaced 105] Improvement of Spiking Neural Network with Bit Planes and Color Models
- **分类: cs.CV; cs.NE; eess.IV**

- **链接: [https://arxiv.org/pdf/2410.08229v5](https://arxiv.org/pdf/2410.08229v5)**

> **作者:** Nhan T. Luu; Duong T. Luu; Nam N. Pham; Thang C. Truong
>
> **备注:** Accepted for publication at IEEE Access
>
> **摘要:** Spiking neural network (SNN) has emerged as a promising paradigm in computational neuroscience and artificial intelligence, offering advantages such as low energy consumption and small memory footprint. However, their practical adoption is constrained by several challenges, prominently among them being performance optimization. In this study, we present a novel approach to enhance the performance of SNN for images through a new coding method that exploits bit plane representation. Our proposed technique is designed to improve the accuracy of SNN without increasing model size. Also, we investigate the impacts of color models of the proposed coding process. Through extensive experimental validation, we demonstrate the effectiveness of our coding strategy in achieving performance gain across multiple datasets. To the best of our knowledge, this is the first research that considers bit planes and color models in the context of SNN. By leveraging the unique characteristics of bit planes, we hope to unlock new potentials in SNNs performance, potentially paving the way for more efficient and effective SNNs models in future researches and applications.
>
---
#### [replaced 106] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.05662v3](https://arxiv.org/pdf/2504.05662v3)**

> **作者:** Shunsuke Sakai; Xiangteng He; Chunzhi Gu; Leonid Sigal; Tatsuhito Hasegawa
>
> **备注:** Code is available at https://github.com/SkyShunsuke/InversionAD
>
> **摘要:** Despite the remarkable success, recent reconstruction-based anomaly detection (AD) methods via diffusion modeling still involve fine-grained noise-strength tuning and computationally expensive multi-step denoising, leading to a fundamental tension between fidelity and efficiency. In this paper, we propose InvAD, a novel inversion-based anomaly detection approach ("detection via noising in latent space") that circumvents explicit reconstruction. Importantly, we contend that the limitations in prior reconstruction-based methods originate from the prevailing "detection via denoising in RGB space" paradigm. To address this, we model AD under a reconstruction-free formulation, which directly infers the final latent variable corresponding to the input image via DDIM inversion, and then measures the deviation based on the known prior distribution for anomaly scoring. Specifically, in approximating the original probability flow ODE using the Euler method, we enforce only a few inversion steps to noise the clean image to pursue inference efficiency. As the added noise is adaptively derived with the learned diffusion model, the original features for the clean testing image can still be leveraged to yield high detection accuracy. We perform extensive experiments and detailed analyses across four widely used industrial and medical AD benchmarks under the unsupervised unified setting to demonstrate the effectiveness of our model, achieving state-of-the-art AD performance and approximately 2x inference-time speedup without diffusion distillation.
>
---
#### [replaced 107] Learning to Upscale 3D Segmentations in Neuroimaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.21697v2](https://arxiv.org/pdf/2505.21697v2)**

> **作者:** Xiaoling Hu; Peirong Liu; Dina Zemlyanker; Jonathan Williams Ramirez; Oula Puonti; Juan Eugenio Iglesias
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Obtaining high-resolution (HR) segmentations from coarse annotations is a pervasive challenge in computer vision. Applications include inferring pixel-level segmentations from token-level labels in vision transformers, upsampling coarse masks to full resolution, and transferring annotations from legacy low-resolution (LR) datasets to modern HR imagery. These challenges are especially acute in 3D neuroimaging, where manual labeling is costly and resolutions continually increase. We propose a scalable framework that generalizes across resolutions and domains by regressing signed distance maps, enabling smooth, boundary-aware supervision. Crucially, our model predicts one class at a time, which substantially reduces memory usage during training and inference (critical for large 3D volumes) and naturally supports generalization to unseen classes. Generalization is further improved through training on synthetic, domain-randomized data. We validate our approach on ultra-high-resolution (UHR) human brain MRI (~100 μm), where most existing methods operate at 1 mm resolution. Our framework effectively upsamples such standard-resolution segmentations to UHR detail. Results on synthetic and real data demonstrate superior scalability and generalization compared to conventional segmentation methods. Code is available at: https://github.com/HuXiaoling/Learn2Upscale.
>
---
#### [replaced 108] Frame-wise Conditioning Adaptation for Fine-Tuning Diffusion Models in Text-to-Video Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.12953v2](https://arxiv.org/pdf/2503.12953v2)**

> **作者:** Zheyuan Liu; Junyan Wang; Zicheng Duan; Cristian Rodriguez-Opazo; Anton van den Hengel
>
> **备注:** Accepted by TMLR, 11/2025. 29 pages, 15 figures
>
> **摘要:** Text-video prediction (TVP) is a downstream video generation task that requires a model to produce subsequent video frames given a series of initial video frames and text describing the required motion. In practice TVP methods focus on a particular category of videos depicting manipulations of objects carried out by human beings or robot arms. Previous methods adapt models pre-trained on text-to-image tasks, and thus tend to generate video that lacks the required continuity. A natural progression would be to leverage more recent pre-trained text-to-video (T2V) models. This approach is rendered more challenging by the fact that the most common fine-tuning technique, low-rank adaptation (LoRA), yields undesirable results. In this work, we propose an adaptation-based strategy we label Frame-wise Conditioning Adaptation (FCA). Within the module, we devise a sub-module that produces frame-wise text embeddings from the input text, which acts as an additional text condition to aid generation. We use FCA to fine-tune the T2V model, which incorporates the initial frame(s) as an extra condition. We compare and discuss the more effective strategy for injecting such embeddings into the T2V model. We conduct extensive ablation studies on our design choices with quantitative and qualitative performance analysis. Our approach establishes a new state-of-the-art for the task of TVP. Our code is open-source at https://github.com/Cuberick-Orion/FCA .
>
---
#### [replaced 109] Spatiotemporal Graph Convolutional Recurrent Neural Network Model for Citywide Air Pollution Forecasting
- **分类: cs.CV; cs.LG; eess.SP**

- **链接: [https://arxiv.org/pdf/2304.12630v2](https://arxiv.org/pdf/2304.12630v2)**

> **作者:** Van-Duc Le; Tien-Cuong Bui; Sang-Kyun Cha
>
> **备注:** Updated metadata
>
> **摘要:** Citywide Air Pollution Forecasting tries to precisely predict the air quality multiple hours ahead for the entire city. This topic is challenged since air pollution varies in a spatiotemporal manner and depends on many complicated factors. Our previous research has solved the problem by considering the whole city as an image and leveraged a Convolutional Long Short-Term Memory (ConvLSTM) model to learn the spatiotemporal features. However, an image-based representation may not be ideal as air pollution and other impact factors have natural graph structures. In this research, we argue that a Graph Convolutional Network (GCN) can efficiently represent the spatial features of air quality readings in the whole city. Specially, we extend the ConvLSTM model to a Spatiotemporal Graph Convolutional Recurrent Neural Network (Spatiotemporal GCRNN) model by tightly integrating a GCN architecture into an RNN structure for efficient learning spatiotemporal characteristics of air quality values and their influential factors. Our extensive experiments prove the proposed model has a better performance compare to the state-of-the-art ConvLSTM model for air pollution predicting while the number of parameters is much smaller. Moreover, our approach is also superior to a hybrid GCN-based method in a real-world air pollution dataset.
>
---
#### [replaced 110] K-FACE: A Large-Scale KIST Face Database in Consideration with Unconstrained Environments
- **分类: cs.CV; cs.DB**

- **链接: [https://arxiv.org/pdf/2103.02211v2](https://arxiv.org/pdf/2103.02211v2)**

> **作者:** Yeji Choi; Hyunjung Park; Gi Pyo Nam; Haksub Kim; Heeseung Choi; Junghyun Cho; Ig-Jae Kim
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** In this paper, we introduce a new large-scale face database from KIST, denoted as K-FACE, and describe a novel capturing device specifically designed to obtain the data. The K-FACE database contains more than 1 million high-quality images of 1,000 subjects selected by considering the ratio of gender and age groups. It includes a variety of attributes, including 27 poses, 35 lighting conditions, three expressions, and occlusions by the combination of five types of accessories. As the K-FACE database is systematically constructed through a hemispherical capturing system with elaborate lighting control and multiple cameras, it is possible to accurately analyze the effects of factors that cause performance degradation, such as poses, lighting changes, and accessories. We consider not only the balance of external environmental factors, such as pose and lighting, but also the balance of personal characteristics such as gender and age group. The gender ratio is the same, while the age groups of subjects are uniformly distributed from the 20s to 50s for both genders. The K-FACE database can be extensively utilized in various vision tasks, such as face recognition, face frontalization, illumination normalization, face age estimation, and three-dimensional face model generation. We expect systematic diversity and uniformity of the K-FACE database to promote these research fields.
>
---
#### [replaced 111] Beyond Complete Shapes: A Benchmark for Quantitative Evaluation of 3D Shape Surface Matching Algorithms
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.03511v3](https://arxiv.org/pdf/2411.03511v3)**

> **作者:** Viktoria Ehm; Nafie El Amrani; Yizheng Xie; Lennart Bastian; Maolin Gao; Weikang Wang; Lu Sang; Dongliang Cao; Tobias Weißberg; Zorah Lähner; Daniel Cremers; Florian Bernard
>
> **摘要:** Finding correspondences between 3D deformable shapes is an important and long-standing problem in geometry processing, computer vision, graphics, and beyond. While various shape matching datasets exist, they are mostly static or limited in size, restricting their adaptation to different problem settings, including both full and partial shape matching. In particular the existing partial shape matching datasets are small (fewer than 100 shapes) and thus unsuitable for data-hungry machine learning approaches. Moreover, the type of partiality present in existing datasets is often artificial and far from realistic. To address these limitations, we introduce a generic and flexible framework for the procedural generation of challenging full and partial shape matching datasets. Our framework allows the propagation of custom annotations across shapes, making it useful for various applications. By utilising our framework and manually creating cross-dataset correspondences between seven existing (complete geometry) shape matching datasets, we propose a new large benchmark BeCoS with a total of 2543 shapes. Based on this, we offer several challenging benchmark settings, covering both full and partial matching, for which we evaluate respective state-of-the-art methods as baselines.
>
---
#### [replaced 112] Spatial Knowledge Graph-Guided Multimodal Synthesis
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2505.22633v3](https://arxiv.org/pdf/2505.22633v3)**

> **作者:** Yida Xue; Zhen Bi; Jinnan Yang; Jungang Lou; Kehai Chen; Min Zhang; Huajun Chen; Ningyu Zhang
>
> **备注:** IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced their capabilities; however, their spatial perception abilities remain a notable limitation. To address this challenge, multimodal data synthesis offers a promising solution. Yet, ensuring that synthesized data adhere to spatial common sense is a non-trivial task. Our approach addresses this critical gap by providing a systematic framework for generating spatially coherent data. In this work, we introduce SKG2DATA, a novel multimodal synthesis approach guided by spatial knowledge graphs, grounded in the concept of knowledge-to-data generation. SKG2DATA employs an automated pipeline for constructing Spatial Knowledge Graph (SKG) that effectively captures human-like spatial cognition, including directional and distance relationships. These structured representations then serve as precise guidance for our integrated synthesis pipeline, where a diffusion model generates spatially-consistent images while a MLLM produces corresponding textual descriptions. The automated construction of SKG enables scalable generation of diverse yet realistic spatial configurations, overcoming the limitations of manual data collection and annotation. Extensive experiments demonstrate that data synthesized from diverse types of spatial knowledge, including direction and distance, enhance the spatial perception and reasoning abilities of MLLMs markedly, albeit with a slight cost to their general capabilities. We hope that the idea of knowledge-based data synthesis can advance the development of spatial intelligence. Code is available at https://github.com/zjunlp/Knowledge2Data.
>
---
#### [replaced 113] Physics-Based Decomposition of Reflectance and Shading using a Single Visible-Thermal Image Pair
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.10388v2](https://arxiv.org/pdf/2509.10388v2)**

> **作者:** Zeqing Leo Yuan; Mani Ramanagopal; Aswin C. Sankaranarayanan; Srinivasa G. Narasimhan
>
> **摘要:** Decomposing an image into its underlying photometric factors--surface reflectance and shading--is a long-standing challenge due to the lack of extensive ground-truth data for real-world scenes. We introduce a novel physics-based approach for intrinsic image decomposition using a pair of visible and thermal images. We leverage the principle that light not reflected from an opaque surface is absorbed and detected as heat by a thermal camera. This allows us to relate the ordinalities (or relative magnitudes) between visible and thermal image intensities to the ordinalities of shading and reflectance, which enables a dense self-supervision of an optimizing neural network to recover shading and reflectance. We perform quantitative evaluations with known reflectance and shading under natural and artificial lighting, and qualitative experiments across diverse scenes. The results demonstrate superior performance over both physics-based and recent learning-based methods, providing a path toward scalable real-world data curation with supervision.
>
---
#### [replaced 114] MSCloudCAM: Multi-Scale Context Adaptation with Convolutional Cross-Attention for Multispectral Cloud Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.10802v3](https://arxiv.org/pdf/2510.10802v3)**

> **作者:** Md Abdullah Al Mazid; Liangdong Deng; Naphtali Rishe
>
> **备注:** 6 pages, 3 Figures
>
> **摘要:** Clouds remain a major obstacle in optical satellite imaging, limiting accurate environmental and climate analysis. To address the strong spectral variability and the large scale differences among cloud types, we propose MSCloudCAM, a novel multi-scale context adapter network with convolution based cross-attention tailored for multispectral and multi-sensor cloud segmentation. A key contribution of MSCloudCAM is the explicit modeling of multiple complementary multi-scale context extractors. And also, rather than simply stacking or concatenating their outputs, our formulation uses one extractor's fine-resolution features and the other extractor's global contextual representations enabling dynamic, scale-aware feature selection. Building on this idea, we design a new convolution-based cross attention adapter that effectively fuses localized, detailed information with broader multi-scale context. Integrated with a hierarchical vision backbone and refined through channel and spatial attention mechanisms, MSCloudCAM achieves strong spectral-spatial discrimination. Experiments on various multisensor datatsets e.g. CloudSEN12 (Sentinel-2) and L8Biome (Landsat-8) show that MSCloudCAM outperforms recent state-of-the-art models while maintaining competitive model complexity, highlighting the novelty and effectiveness of the proposed design for large-scale Earth observation.
>
---
#### [replaced 115] PairHuman: A High-Fidelity Photographic Dataset for Customized Dual-Person Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.16712v2](https://arxiv.org/pdf/2511.16712v2)**

> **作者:** Ting Pan; Ye Wang; Peiguang Jing; Rui Ma; Zili Yi; Yu Liu
>
> **备注:** 46 pages, 31 figures
>
> **摘要:** Personalized dual-person portrait customization has considerable potential applications, such as preserving emotional memories and facilitating wedding photography planning. However, the absence of a benchmark dataset hinders the pursuit of high-quality customization in dual-person portrait generation. In this paper, we propose the PairHuman dataset, which is the first large-scale benchmark dataset specifically designed for generating dual-person portraits that meet high photographic standards. The PairHuman dataset contains more than 100K images that capture a variety of scenes, attire, and dual-person interactions, along with rich metadata, including detailed image descriptions, person localization, human keypoints, and attribute tags. We also introduce DHumanDiff, which is a baseline specifically crafted for dual-person portrait generation that features enhanced facial consistency and simultaneously balances in personalized person generation and semantic-driven scene creation. Finally, the experimental results demonstrate that our dataset and method produce highly customized portraits with superior visual quality that are tailored to human preferences. Our dataset is publicly available at https://github.com/annaoooo/PairHuman.
>
---
#### [replaced 116] VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2411.17991v2](https://arxiv.org/pdf/2411.17991v2)**

> **作者:** Yueqian Wang; Xiaojun Meng; Yuxuan Wang; Jianxin Liang; Jiansheng Wei; Huishuai Zhang; Dongyan Zhao
>
> **备注:** 9 pages
>
> **摘要:** Recent researches on video large language models (VideoLLM) predominantly focus on model architectures and training datasets, leaving the interaction format between the user and the model under-explored. In existing works, users often interact with VideoLLMs by using the entire video and a query as input, after which the model generates a response. This interaction format constrains the application of VideoLLMs in scenarios such as live-streaming comprehension where videos do not end and responses are required in a real-time manner, and also results in unsatisfactory performance on time-sensitive tasks that requires localizing video segments. In this paper, we focus on a video-text duet interaction format. This interaction format is characterized by the continuous playback of the video, and both the user and the model can insert their text messages at any position during the video playback. When a text message ends, the video continues to play, akin to the alternative of two performers in a duet. We construct MMDuetIT, a video-text training dataset designed to adapt VideoLLMs to video-text duet interaction format. We also introduce the Multi-Answer Grounded Video Question Answering (MAGQA) task to benchmark the real-time response ability of VideoLLMs. Trained on MMDuetIT, MMDuet demonstrates that adopting the video-text duet interaction format enables the model to achieve significant improvements in various time-sensitive tasks (76% CIDEr on YouCook2 dense video captioning, 90\% mAP on QVHighlights highlight detection and 25% R@0.5 on Charades-STA temporal video grounding) with minimal training efforts, and also enable VideoLLMs to reply in a real-time manner as the video plays.
>
---
#### [replaced 117] COLI: A Hierarchical Efficient Compressor for Large Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.11443v2](https://arxiv.org/pdf/2507.11443v2)**

> **作者:** Haoran Wang; Hanyu Pei; Yang Lyu; Kai Zhang; Li Li; Feng-Lei Fan
>
> **摘要:** The escalating adoption of high-resolution, large-field-of-view imagery amplifies the need for efficient compression methodologies. Conventional techniques frequently fail to preserve critical image details, while data-driven approaches exhibit limited generalizability. Implicit Neural Representations (INRs) present a promising alternative by learning continuous mappings from spatial coordinates to pixel intensities for individual images, thereby storing network weights rather than raw pixels and avoiding the generalization problem. However, INR-based compression of large images faces challenges including slow compression speed and suboptimal compression ratios. To address these limitations, we introduce COLI (Compressor for Large Images), a novel framework leveraging Neural Representations for Videos (NeRV). First, recognizing that INR-based compression constitutes a training process, we accelerate its convergence through a pretraining-finetuning paradigm, mixed-precision training, and reformulation of the sequential loss into a parallelizable objective. Second, capitalizing on INRs' transformation of image storage constraints into weight storage, we implement Hyper-Compression, a novel post-training technique to substantially enhance compression ratios while maintaining minimal output distortion. Evaluations across two medical imaging datasets demonstrate that COLI consistently achieves competitive or superior PSNR and SSIM metrics at significantly reduced bits per pixel (bpp), while accelerating NeRV training by up to 4 times.
>
---
#### [replaced 118] RichControl: Structure- and Appearance-Rich Training-Free Spatial Control for Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.02792v4](https://arxiv.org/pdf/2507.02792v4)**

> **作者:** Liheng Zhang; Lexi Pang; Hang Ye; Xiaoxuan Ma; Yizhou Wang
>
> **摘要:** Text-to-image (T2I) diffusion models have shown remarkable success in generating high-quality images from text prompts. Recent efforts extend these models to incorporate conditional images (e.g., canny edge) for fine-grained spatial control. Among them, feature injection methods have emerged as a training-free alternative to traditional fine-tuning-based approaches. However, they often suffer from structural misalignment, condition leakage, and visual artifacts, especially when the condition image diverges significantly from natural RGB distributions. Through an empirical analysis of existing methods, we identify a key limitation: the sampling schedule of condition features, previously unexplored, fails to account for the evolving interplay between structure preservation and domain alignment throughout diffusion steps. Inspired by this observation, we propose a flexible training-free framework that decouples the sampling schedule of condition features from the denoising process, and systematically investigate the spectrum of feature injection schedules for a higher-quality structure guidance in the feature space. Specifically, we find that condition features sampled from a single timestep are sufficient, yielding a simple yet efficient schedule that balances structure alignment and appearance quality. We further enhance the sampling process by introducing a restart refinement schedule, and improve the visual quality with an appearance-rich prompting strategy. Together, these designs enable training-free generation that is both structure-rich and appearance-rich. Extensive experiments show that our approach achieves state-of-the-art results across diverse zero-shot conditioning scenarios.
>
---
#### [replaced 119] D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2507.05859v2](https://arxiv.org/pdf/2507.05859v2)**

> **作者:** Wenkang Zhang; Yan Zhao; Qiang Wang; Zhixin Xu; Li Song; Zhengxue Cheng
>
> **备注:** AAAI-26 accepted, code: https://github.com/Mr-Zwkid/D-FCGS
>
> **摘要:** Free-Viewpoint Video (FVV) enables immersive 3D experiences, but efficient compression of dynamic 3D representation remains a major challenge. Existing dynamic 3D Gaussian Splatting methods couple reconstruction with optimization-dependent compression and customized motion formats, limiting generalization and standardization. To address this, we propose D-FCGS, a novel Feedforward Compression framework for Dynamic Gaussian Splatting. Key innovations include: (1) a standardized Group-of-Frames (GoF) structure with I-P coding, leveraging sparse control points to extract inter-frame motion tensors; (2) a dual prior-aware entropy model that fuses hyperprior and spatial-temporal priors for accurate rate estimation; (3) a control-point-guided motion compensation mechanism and refinement network to enhance view-consistent fidelity. Trained on Gaussian frames derived from multi-view videos, D-FCGS generalizes across diverse scenes in a zero-shot fashion. Experiments show that it matches the rate-distortion performance of optimization-based methods, achieving over 40 times compression compared to the baseline while preserving visual quality across viewpoints. This work advances feedforward compression of dynamic 3DGS, facilitating scalable FVV transmission and storage for immersive applications.
>
---
#### [replaced 120] AsynEIO: Asynchronous Monocular Event-Inertial Odometry Using Gaussian Process Regression
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.12175v2](https://arxiv.org/pdf/2411.12175v2)**

> **作者:** Zhixiang Wang; Xudong Li; Yizhai Zhang; Fan Zhang; Panfeng Huang
>
> **备注:** 20 pages, 20 figures
>
> **摘要:** Event cameras, when combined with inertial sensors, show significant potential for motion estimation in challenging scenarios, such as high-speed maneuvers and low-light environments. There are many methods for producing such estimations, but most boil down to a synchronous discrete-time fusion problem. However, the asynchronous nature of event cameras and their unique fusion mechanism with inertial sensors remain underexplored. In this paper, we introduce a monocular event-inertial odometry method called AsynEIO, designed to fuse asynchronous event and inertial data within a unified Gaussian Process (GP) regression framework. Our approach incorporates an event-driven frontend that tracks feature trajectories directly from raw event streams at a high temporal resolution. These tracked feature trajectories, along with various inertial factors, are integrated into the same GP regression framework to enable asynchronous fusion. With deriving analytical residual Jacobians and noise models, our method constructs a factor graph that is iteratively optimized and pruned using a sliding-window optimizer. Comparative assessments highlight the performance of different inertial fusion strategies, suggesting optimal choices for varying conditions. Experimental results on both public datasets and our own event-inertial sequences indicate that AsynEIO outperforms existing methods, especially in high-speed and low-illumination scenarios.
>
---
#### [replaced 121] Alpha Divergence Losses for Biometric Verification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.13621v3](https://arxiv.org/pdf/2511.13621v3)**

> **作者:** Dimitrios Koutsianos; Ladislav Mosner; Yannis Panagakis; Themos Stafylakis
>
> **摘要:** Performance in face and speaker verification is largely driven by margin-based softmax losses such as CosFace and ArcFace. Recently introduced $α$-divergence loss functions offer a compelling alternative, particularly due to their ability to induce sparse solutions (when $α>1$). However, integrating an angular margin-crucial for verification tasks-is not straightforward. We find that this integration can be achieved in at least two distinct ways: via the reference measure (prior probabilities) or via the logits (unnormalized log-likelihoods). In this paper, we explore both pathways, deriving two novel margin-based $α$-divergence losses: Q-Margin (margin in the reference measure) and A3M (margin in the logits). We identify and address a training instability in A3M-caused by sparsity-with a simple yet effective prototype re-initialization strategy. Our methods achieve significant performance gains on the challenging IJB-B and IJB-C face verification benchmarks. We demonstrate similarly strong performance in speaker verification on VoxCeleb. Crucially, our models significantly outperform strong baselines at low false acceptance rates (FAR). This capability is critical for practical high-security applications, such as banking authentication, when minimizing false authentications is paramount. Finally, the sparsity of $α$-divergence-based posteriors enables memory-efficient training, which is crucial for datasets with millions of identities.
>
---
#### [replaced 122] GraphPilot: Grounded Scene Graph Conditioning for Language-Based Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11266v2](https://arxiv.org/pdf/2511.11266v2)**

> **作者:** Fabian Schmidt; Markus Enzweiler; Abhinav Valada
>
> **摘要:** Vision-language models have recently emerged as promising planners for autonomous driving, where success hinges on topology-aware reasoning over spatial structure and dynamic interactions from multimodal input. However, existing models are typically trained without supervision that explicitly encodes these relational dependencies, limiting their ability to infer how agents and other traffic entities influence one another from raw sensor data. In this work, we bridge this gap with a novel model-agnostic method that conditions language-based driving models on structured relational context in the form of traffic scene graphs. We serialize scene graphs at various abstraction levels and formats, and incorporate them into the models via structured prompt templates, enabling a systematic analysis of when and how relational supervision is most beneficial. Extensive evaluations on the public LangAuto benchmark show that scene graph conditioning of state-of-the-art approaches yields large and persistent improvement in driving performance. Notably, we observe up to a 15.6\% increase in driving score for LMDrive and 17.5\% for BEVDriver, indicating that models can better internalize and ground relational priors through scene graph-conditioned training, even without requiring scene graph input at test-time. Code, fine-tuned models, and our scene graph dataset are publicly available at https://github.com/iis-esslingen/GraphPilot.
>
---
#### [replaced 123] Learning to Drive Anywhere with Model-Based Reannotation
- **分类: cs.RO; cs.CV; cs.LG; eess.SY**

- **链接: [https://arxiv.org/pdf/2505.05592v3](https://arxiv.org/pdf/2505.05592v3)**

> **作者:** Noriaki Hirose; Lydia Ignatova; Kyle Stachowicz; Catherine Glossop; Sergey Levine; Dhruv Shah
>
> **备注:** 9 pages, 8 figures, 6 tables
>
> **摘要:** Developing broadly generalizable visual navigation policies for robots is a significant challenge, primarily constrained by the availability of large-scale, diverse training data. While curated datasets collected by researchers offer high quality, their limited size restricts policy generalization. To overcome this, we explore leveraging abundant, passively collected data sources, including large volumes of crowd-sourced teleoperation data and unlabeled YouTube videos, despite their potential for lower quality or missing action labels. We propose Model-Based ReAnnotation (MBRA), a framework that utilizes a learned short-horizon, model-based expert model to relabel or generate high-quality actions for these passive datasets. This relabeled data is then distilled into LogoNav, a long-horizon navigation policy conditioned on visual goals or GPS waypoints. We demonstrate that LogoNav, trained using MBRA-processed data, achieves state-of-the-art performance, enabling robust navigation over distances exceeding 300 meters in previously unseen indoor and outdoor environments. Our extensive real-world evaluations, conducted across a fleet of robots (including quadrupeds) in six cities on three continents, validate the policy's ability to generalize and navigate effectively even amidst pedestrians in crowded settings.
>
---
#### [replaced 124] Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.01632v3](https://arxiv.org/pdf/2504.01632v3)**

> **作者:** Giulia Marchiori Pietrosanti; Giulio Rossolini; Alessandro Biondi; Giorgio Buttazzo
>
> **备注:** Accepted for publication in Pattern Recognition
>
> **摘要:** The robustness of deep neural networks is a crucial factor in safety-critical applications, particularly in complex and dynamic environments (e.g., medical or driving scenarios) where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remains underexplored. This paper fills this gap by introducing novel, region-aware metrics for benchmarking the spatial robustness of segmentation models, along with an evaluation framework to assess the impact of natural localized corruptions. Furthermore, it uncovers the inherent complexity of evaluating worst-case spatial robustness using only a single localized adversarial attack. To address this, the work proposes a region-aware multi-attack adversarial analysis to systematically assess model robustness across specific image regions. The proposed metrics and analysis were exploited to evaluate 14 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones, and vice versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks.
>
---
#### [replaced 125] RefVTON: person-to-person Try on with Additional Unpaired Visual Reference
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00956v3](https://arxiv.org/pdf/2511.00956v3)**

> **作者:** Liuzhuozheng Li; Yue Gong; Shanyuan Liu; Bo Cheng; Yuhang Ma; Liebucha Wu; Dengyang Jiang; Zanyi Wang; Dawei Leng; Yuhui Yin
>
> **摘要:** We introduce RefTON, a flux-based person-to-person virtual try-on framework that enhances garment realism through unpaired visual references. Unlike conventional approaches that rely on complex auxiliary inputs such as body parsing and warped mask or require finely designed extract branches to process various input conditions, RefTON streamlines the process by directly generating try-on results from a source image and a target garment, without the need for structural guidance or auxiliary components to handle diverse inputs. Moreover, inspired by human clothing selection behavior, RefTON leverages additional reference images (the target garment worn on different individuals) to provide powerful guidance for refining texture alignment and maintaining the garment details. To enable this capability, we built a dataset containing unpaired reference images for training. Extensive experiments on public benchmarks demonstrate that RefTON achieves competitive or superior performance compared to state-of-the-art methods, while maintaining a simple and efficient person-to-person design.
>
---
#### [replaced 126] Monocular Person Localization under Camera Ego-motion
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.02916v2](https://arxiv.org/pdf/2503.02916v2)**

> **作者:** Yu Zhan; Hanjing Ye; Hong Zhang
>
> **备注:** Accepted by IROS2025. Project page: https://medlartea.github.io/rpf-quadruped/
>
> **摘要:** Localizing a person from a moving monocular camera is critical for Human-Robot Interaction (HRI). To estimate the 3D human position from a 2D image, existing methods either depend on the geometric assumption of a fixed camera or use a position regression model trained on datasets containing little camera ego-motion. These methods are vulnerable to severe camera ego-motion, resulting in inaccurate person localization. We consider person localization as a part of a pose estimation problem. By representing a human with a four-point model, our method jointly estimates the 2D camera attitude and the person's 3D location through optimization. Evaluations on both public datasets and real robot experiments demonstrate our method outperforms baselines in person localization accuracy. Our method is further implemented into a person-following system and deployed on an agile quadruped robot.
>
---
#### [replaced 127] TokenCLIP: Token-wise Prompt Learning for Zero-shot Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21171v3](https://arxiv.org/pdf/2510.21171v3)**

> **作者:** Qihang Zhou; Binbin Gao; Guansong Pang; Xin Wang; Jiming Chen; Shibo He
>
> **摘要:** Adapting CLIP for anomaly detection on unseen objects has shown strong potential in a zero-shot manner. However, existing methods typically rely on a single textual space to align with visual semantics across diverse objects and domains. The indiscriminate alignment hinders the model from accurately capturing varied anomaly semantics. We propose TokenCLIP, a token-wise adaptation framework that enables dynamic alignment between visual and learnable textual spaces for fine-grained anomaly learning. Rather than mapping all visual tokens to a single, token-agnostic textual space, TokenCLIP aligns each token with a customized textual subspace that represents its visual characteristics. Explicitly assigning a unique learnable textual space to each token is computationally intractable and prone to insufficient optimization. We instead expand the token-agnostic textual space into a set of orthogonal subspaces, and then dynamically assign each token to a subspace combination guided by semantic affinity, which jointly supports customized and efficient token-wise adaptation. To this end, we formulate dynamic alignment as an optimal transport problem, where all visual tokens in an image are transported to textual subspaces based on semantic similarity. The transport constraints of OT ensure sufficient optimization across subspaces and encourage them to focus on different semantics. Solving the problem yields a transport plan that adaptively assigns each token to semantically relevant subspaces. A top-k masking is then applied to sparsify the plan and specialize subspaces for distinct visual regions. Extensive experiments demonstrate the superiority of TokenCLIP.
>
---
#### [replaced 128] Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12609v2](https://arxiv.org/pdf/2511.12609v2)**

> **作者:** Yunxin Li; Xinyu Chen; Shenyuan Jiang; Haoyuan Shi; Zhenyu Liu; Xuanyu Zhang; Nanhao Deng; Zhenran Xu; Yicheng Ma; Meishan Zhang; Baotian Hu; Min Zhang
>
> **备注:** 47 pages,10 Figures, Project Website: https://idealistxy.github.io/Uni-MoE-v2.github.io/ Codes: https://github.com/HITsz-TMG/Uni-MoE
>
> **摘要:** We present Uni-MoE 2.0 from the Lychee family. As a fully open-source omnimodal large model (OLM), it substantially advances Lychee's Uni-MoE series in language-centric multimodal understanding, reasoning, and generating. Based on the dense LLM, we build Uni-MoE-2.0-Omni from scratch through three core contributions: dynamic-capacity Mixture-of-Experts (MoE) design, a progressive training strategy enhanced with an iterative reinforcement strategy, and a carefully curated multimodal data matching technique. It is capable of omnimodal understanding, as well as generating images, text, and speech. Architecturally, our new MoE framework balances computational efficiency and capability for 10 cross-modal inputs using shared, routed, and null experts, while our Omni-Modality 3D RoPE ensures spatio-temporal cross-modality alignment in the self-attention layer. For training, following cross-modal pretraining, we use a progressive supervised fine-tuning strategy that activates modality-specific experts and is enhanced by balanced data composition and an iterative GSPO-DPO method to stabilise RL training and improve reasoning. Data-wise, the base model, trained on approximately 75B tokens of open-source multimodal data, is equipped with special speech and image generation tokens, allowing it to learn these generative tasks by conditioning its outputs on linguistic cues. Extensive evaluation across 85 benchmarks demonstrates that our model achieves SOTA or highly competitive performance against leading OLMs, surpassing Qwen2.5-Omni (trained with 1.2T tokens) on over 50 of 76 benchmarks. Key strengths include video understanding (+7% avg. of 8), omnimodallity understanding (+7% avg. of 4), and audiovisual reasoning (+4%). It also advances long-form speech processing (reducing WER by 4.2%) and leads in low-level image processing and controllable generation across 5 metrics.
>
---
#### [replaced 129] PSA-MIL: A Probabilistic Spatial Attention-Based Multiple Instance Learning for Whole Slide Image Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.16284v2](https://arxiv.org/pdf/2503.16284v2)**

> **作者:** Sharon Peled; Yosef E. Maruvka; Moti Freiman
>
> **备注:** Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026
>
> **摘要:** Whole Slide Images (WSIs) are high-resolution digital scans widely used in medical diagnostics. WSI classification is typically approached using Multiple Instance Learning (MIL), where the slide is partitioned into tiles treated as interconnected instances. While attention-based MIL methods aim to identify the most informative tiles, they often fail to fully exploit the spatial relationships among them, potentially overlooking intricate tissue structures crucial for accurate diagnosis. To address this limitation, we propose Probabilistic Spatial Attention MIL (PSA-MIL), a novel attention-based MIL framework that integrates spatial context into the attention mechanism through learnable distance-decayed priors, formulated within a probabilistic interpretation of self-attention as a posterior distribution. This formulation enables a dynamic inference of spatial relationships during training, eliminating the need for predefined assumptions often imposed by previous approaches. Additionally, we suggest a spatial pruning strategy for the posterior, effectively reducing self-attention's quadratic complexity. To further enhance spatial modeling, we introduce a diversity loss that encourages variation among attention heads, ensuring each captures distinct spatial representations. Together, PSA-MIL enables a more data-driven and adaptive integration of spatial context, moving beyond predefined constraints. We achieve state-of-the-art performance across both contextual and non-contextual baselines, while significantly reducing computational costs.
>
---
#### [replaced 130] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05186v4](https://arxiv.org/pdf/2508.05186v4)**

> **作者:** Yongjie Bai; Zhouxia Wang; Yang Liu; Kaijun Luo; Yifan Wen; Mingtong Dai; Weixing Chen; Ziliang Chen; Lingbo Liu; Guanbin Li; Liang Lin
>
> **备注:** 24 pages, 15 figures, project page: https://hcplab-sysu.github.io/TAVP
>
> **摘要:** Recent vision-language-action (VLA) models for multi-task robotic manipulation commonly rely on static viewpoints and shared visual encoders, which limit 3D perception and cause task interference, hindering robustness and generalization. In this work, we propose Task-aware Virtual View Exploration (TVVE), a framework designed to overcome these challenges by integrating virtual view exploration with task-specific representation learning. TVVE employs an efficient exploration policy, accelerated by a novel pseudo-environment, to acquire informative views. Furthermore, we introduce a Task-aware Mixture-of-Experts (TaskMoE) visual encoder to disentangle features across different tasks, boosting both representation fidelity and task generalization. By learning to see the world in a task-aware way, TVVE generates more complete and discriminative visual representations, demonstrating significantly enhanced action prediction across a wide array of manipulation challenges. To further validate the robustness and generalization capability of TVVE under out-of-distribution (OOD) settings, we construct a challenging benchmark, RLBench-OG, covering various visual perturbations and camera pose variations. Extensive experiments on RLBench and RLBench-OG show that our TVVE achieves superior performance over state-of-the-art approaches. In real-robot experiments, TVVE demonstrates exceptional performance and generalizes robustly in multiple OOD settings, including visual disturbances and unseen instructions. Visual results and code are provided at: https://hcplab-sysu.github.io/TAVP.
>
---
#### [replaced 131] Zero-Shot Coreset Selection via Iterative Subspace Sampling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.15349v2](https://arxiv.org/pdf/2411.15349v2)**

> **作者:** Brent A. Griffin; Jacob Marks; Jason J. Corso
>
> **备注:** WACV 2026
>
> **摘要:** Deep learning increasingly relies on massive data with substantial storage, annotation, and training costs. To reduce costs, coreset selection finds a representative subset of data to train models while ideally performing on par with the full data training. To maximize performance, current state-of-the-art coreset methods select data using dataset-specific ground truth labels and training. However, these methodological requirements prevent selection at scale on real-world, unlabeled data. To that end, this paper addresses the selection of coresets that achieve state-of-the-art performance but without using any labels or training on candidate data. Instead, our solution, Zero-Shot Coreset Selection via Iterative Subspace Sampling (ZCore), uses previously-trained foundation models to generate zero-shot, high-dimensional embedding spaces to interpret unlabeled data. ZCore then iteratively quantifies the relative value of all candidate data based on coverage and redundancy in numerous subspace distributions. Finally, ZCore selects a coreset sized for any data budget to train downstream models. We evaluate ZCore on four datasets and outperform several state-of-the-art label-based methods, especially at low data rates that provide the most substantial cost reduction. On ImageNet, ZCore selections for 10% training data achieve a downstream validation accuracy of 53.99%, which outperforms prior label-based methods and removes annotation and training costs for 1.15 million images. Our paper's code is publicly available at https://github.com/voxel51/zcore.
>
---
#### [replaced 132] SplatCo: Structure-View Collaborative Gaussian Splatting for Detail-Preserving Rendering of Large-Scale Unbounded Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.17951v2](https://arxiv.org/pdf/2505.17951v2)**

> **作者:** Haihong Xiao; Jianan Zou; Yuxin Zhou; Ying He; Wenxiong Kang
>
> **摘要:** We present SplatCo, a structure-view collaborative Gaussian splatting framework for high-fidelity rendering of complex outdoor environments. SplatCo builds upon two novel components: (1) a cross-structure collaboration module that combines global tri-plane representations, which capture coarse scene layouts, with local context grid features that represent fine surface details. This fusion is achieved through a novel hierarchical compensation strategy, ensuring both global consistency and local detail preservation; and (2) a cross-view assisted training strategy that enhances multi-view consistency by synchronizing gradient updates across viewpoints, applying visibility-aware densification, and pruning overfitted or inaccurate Gaussians based on structural consistency. Through joint optimization of structural representation and multi-view coherence, SplatCo effectively reconstructs fine-grained geometric structures and complex textures in large-scale scenes. Comprehensive evaluations on 13 diverse large-scale scenes, including Mill19, MatrixCity, Tanks & Temples, WHU, and custom aerial captures, demonstrate that SplatCo consistently achieves higher reconstruction quality than state-of-the-art methods, with PSNR improvements of 1-2 dB and SSIM gains of 0.1 to 0.2. These results establish a new benchmark for high-fidelity rendering of large-scale unbounded scenes. Code and additional information are available at https://github.com/SCUT-BIP-Lab/SplatCo.
>
---
#### [replaced 133] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2508.10936v2](https://arxiv.org/pdf/2508.10936v2)**

> **作者:** Cheng Chen; Hao Huang; Saurabh Bagchi
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Collaborative perception enables connected vehicles to share information, overcoming occlusions and extending the limited sensing range inherent in single-agent (non-collaborative) systems. Existing vision-only methods for 3D semantic occupancy prediction commonly rely on dense 3D voxels, which incur high communication costs, or 2D planar features, which require accurate depth estimation or additional supervision, limiting their applicability to collaborative scenarios. To address these challenges, we propose the first approach leveraging sparse 3D semantic Gaussian splatting for collaborative 3D semantic occupancy prediction. By sharing and fusing intermediate Gaussian primitives, our method provides three benefits: a neighborhood-based cross-agent fusion that removes duplicates and suppresses noisy or inconsistent Gaussians; a joint encoding of geometry and semantics in each primitive, which reduces reliance on depth supervision and allows simple rigid alignment; and sparse, object-centric messages that preserve structural information while reducing communication volume. Extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. When further reducing the number of transmitted Gaussians, our method still achieves a +1.9 improvement in mIoU, using only 34.6% communication volume, highlighting robust performance under limited communication budgets.
>
---
#### [replaced 134] ReBrain: Brain MRI Reconstruction from Sparse CT Slice via Retrieval-Augmented Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.17068v2](https://arxiv.org/pdf/2511.17068v2)**

> **作者:** Junming Liu; Yifei Sun; Weihua Cheng; Yujin Kang; Yirong Chen; Ding Wang; Guosun Zeng
>
> **备注:** 16 pages, 12 figures, 7 tables; Accepted by WACV 2026
>
> **摘要:** Magnetic Resonance Imaging (MRI) plays a crucial role in brain disease diagnosis, but it is not always feasible for certain patients due to physical or clinical constraints. Recent studies attempt to synthesize MRI from Computed Tomography (CT) scans; however, low-dose protocols often result in highly sparse CT volumes with poor through-plane resolution, making accurate reconstruction of the full brain MRI volume particularly challenging. To address this, we propose ReBrain, a retrieval-augmented diffusion framework for brain MRI reconstruction. Given any 3D CT scan with limited slices, we first employ a Brownian Bridge Diffusion Model (BBDM) to synthesize MRI slices along the 2D dimension. Simultaneously, we retrieve structurally and pathologically similar CT slices from a comprehensive prior database via a fine-tuned retrieval model. These retrieved slices are used as references, incorporated through a ControlNet branch to guide the generation of intermediate MRI slices and ensure structural continuity. We further account for rare retrieval failures when the database lacks suitable references and apply spherical linear interpolation to provide supplementary guidance. Extensive experiments on SynthRAD2023 and BraTS demonstrate that ReBrain achieves state-of-the-art performance in cross-modal reconstruction under sparse conditions.
>
---
#### [replaced 135] Sketch-1-to-3: One Single Sketch to 3D Detailed Face Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.17852v3](https://arxiv.org/pdf/2502.17852v3)**

> **作者:** Liting Wen; Zimo Yang; Xianlin Zhang; Chi Ding; Mingdao Wang; Xueming Li
>
> **备注:** Accepted by ACM MMAsia 2025
>
> **摘要:** 3D face reconstruction from a single sketch is a critical yet underexplored task with significant practical applications. The primary challenges stem from the substantial modality gap between 2D sketches and 3D facial structures, including: (1) accurately extracting facial keypoints from 2D sketches; (2) preserving diverse facial expressions and fine-grained texture details; and (3) training a high-performing model with limited data. In this paper, we propose Sketch-1-to-3, a novel framework for realistic 3D face reconstruction from a single sketch, to address these challenges. Specifically, we first introduce the Geometric Contour and Texture Detail (GCTD) module, which enhances the extraction of geometric contours and texture details from facial sketches. Additionally, we design a deep learning architecture with a domain adaptation module and a tailored loss function to align sketches with the 3D facial space, enabling high-fidelity expression and texture reconstruction. To facilitate evaluation and further research, we construct SketchFaces, a real hand-drawn facial sketch dataset, and Syn-SketchFaces, a synthetic facial sketch dataset. Extensive experiments demonstrate that Sketch-1-to-3 achieves state-of-the-art performance in sketch-based 3D face reconstruction.
>
---
#### [replaced 136] Full-scale Representation Guided Network for Retinal Vessel Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2501.18921v2](https://arxiv.org/pdf/2501.18921v2)**

> **作者:** Sunyong Seo; Sangwook Yoo; Huisu Yoon
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** The U-Net architecture and its variants have remained state-of-the-art (SOTA) for retinal vessel segmentation over the past decade. In this study, we introduce a Full-Scale Guided Network (FSG-Net), where a novel feature representation module using modernized convolution blocks effectively captures full-scale structural information, while a guided convolution block subsequently refines this information. Specifically, we introduce an attention-guided filter within the guided convolution block, leveraging its similarity to unsharp masking to enhance fine vascular structures. Passing full-scale information to the attention block facilitates the generation of more contextually relevant attention maps, which are then passed to the attention-guided filter, providing further refinement to the segmentation performance. The structure preceding the guided convolution block can be replaced by any U-Net variant, ensuring flexibility and scalability across various segmentation tasks. For a fair comparison, we re-implemented recent studies available in public repositories to evaluate their scalability and reproducibility. Our experiments demonstrate that, despite its compact architecture, FSG-Net delivers performance competitive with SOTA methods across multiple public datasets. Ablation studies further demonstrate that each proposed component meaningfully contributes to this competitive performance. Our code is available on https://github.com/ZombaSY/FSG-Net-pytorch.
>
---
#### [replaced 137] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding
- **分类: cs.CV; cs.CL; cs.CR**

- **链接: [https://arxiv.org/pdf/2508.09456v3](https://arxiv.org/pdf/2508.09456v3)**

> **作者:** Junxian Li; Beining Xu; Simin Chen; Jiatong Li; Jingdi Lei; Haodong Zhao; Di Zhang
>
> **备注:** 20 pages, 13 Figures
>
> **摘要:** Recent advances in vision-language models (VLMs) have significantly enhanced the visual grounding task, which involves locating objects in an image based on natural language queries. Despite these advancements, the security of VLM-based grounding systems has not been thoroughly investigated. This paper reveals a novel and realistic vulnerability: the first multi-target backdoor attack on VLM-based visual grounding. Unlike prior attacks that rely on static triggers or fixed targets, we propose IAG, a method that dynamically generates input-aware, text-guided triggers conditioned on any specified target object description to execute the attack. This is achieved through a text-conditioned UNet that embeds imperceptible target semantic cues into visual inputs while preserving normal grounding performance on benign samples. We further develop a joint training objective that balances language capability with perceptual reconstruction to ensure imperceptibility, effectiveness, and stealth. Extensive experiments on multiple VLMs (e.g., LLaVA, InternVL, Ferret) and benchmarks (RefCOCO, RefCOCO+, RefCOCOg, Flickr30k Entities, and ShowUI) demonstrate that IAG achieves the best ASRs compared with other baselines on almost all settings without compromising clean accuracy, maintaining robustness against existing defenses, and exhibiting transferability across datasets and models. These findings underscore critical security risks in grounding-capable VLMs and highlight the need for further research on trustworthy multimodal understanding.
>
---
#### [replaced 138] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2412.03121v2](https://arxiv.org/pdf/2412.03121v2)**

> **作者:** Yijia Guo; Wenkai Huang; Yang Li; Gaolei Li; Hang Zhang; Liwen Hu; Jianhua Li; Tiejun Huang; Lei Ma
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** 3D Gaussian splatting (3DGS) has demonstrated impressive 3D reconstruction performance with explicit scene representations. Given the widespread application of 3DGS in 3D reconstruction and generation tasks, there is an urgent need to protect the copyright of 3DGS assets. However, existing copyright protection techniques for 3DGS overlook the usability of 3D assets, posing challenges for practical deployment. Here we describe splats in splats, the first 3DGS steganography framework that embeds 3D content in 3DGS itself without modifying any attributes. To achieve this, we take a deep insight into spherical harmonics (SH) and devise an importance-graded SH coefficient encryption strategy to embed the hidden SH coefficients. Furthermore, we employ a convolutional autoencoder to establish a mapping between the original Gaussian primitives' opacity and the hidden Gaussian primitives' opacity. Extensive experiments indicate that our method significantly outperforms existing 3D steganography techniques, with 5.31% higher scene fidelity and 3x faster rendering speed, while ensuring security, robustness, and user experience.
>
---
#### [replaced 139] U-REPA: Aligning Diffusion U-Nets to ViTs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.18414v2](https://arxiv.org/pdf/2503.18414v2)**

> **作者:** Yuchuan Tian; Hanting Chen; Mengyu Zheng; Yuchen Liang; Chao Xu; Yunhe Wang
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** Representation Alignment (REPA) that aligns Diffusion Transformer (DiT) hidden-states with ViT visual encoders has proven highly effective in DiT training, demonstrating superior convergence properties, but it has not been validated on the canonical diffusion U-Net architecture that shows faster convergence compared to DiTs. However, adapting REPA to U-Net architectures presents unique challenges: (1) different block functionalities necessitate revised alignment strategies; (2) spatial-dimension inconsistencies emerge from U-Net's spatial downsampling operations; (3) space gaps between U-Net and ViT hinder the effectiveness of tokenwise alignment. To encounter these challenges, we propose \textbf{U-REPA}, a representation alignment paradigm that bridges U-Net hidden states and ViT features as follows: Firstly, we propose via observation that due to skip connection, the middle stage of U-Net is the best alignment option. Secondly, we propose upsampling of U-Net features after passing them through MLPs. Thirdly, we observe difficulty when performing tokenwise similarity alignment, and further introduces a manifold loss that regularizes the relative similarity between samples. Experiments indicate that the resulting U-REPA could achieve excellent generation quality and greatly accelerates the convergence speed. With CFG guidance interval, U-REPA could reach $FID<1.5$ in 200 epochs or 1M iterations on ImageNet 256 $\times$ 256, and needs only half the total epochs to perform better than REPA under sd-vae-ft-ema. Codes: https://github.com/YuchuanTian/U-REPA
>
---
#### [replaced 140] Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16669v2](https://arxiv.org/pdf/2511.16669v2)**

> **作者:** Junhao Cheng; Liang Hou; Xin Tao; Jing Liao
>
> **备注:** Project page: https://video-as-answer.github.io/
>
> **摘要:** While language models have become impactful in many real-world applications, video generation remains largely confined to entertainment. Motivated by video's inherent capacity to demonstrate physical-world information that is difficult to convey through language alone (e.g., imagine teaching someone to tie a tie using only text), we identify an underutilized opportunity to extend video as a new answer modality for Next-Event Prediction (NEP), formalized as Video-Next-Event Prediction (VNEP). While the established NEP task takes a video with a procedural or predictive question as input to predict the next event in text, VNEP requires dynamic video responses. This shift from telling to showing unlocks more intuitive and customized answers for procedural learning and creative exploration. However, this task remains challenging for existing models, as it demands an understanding of multimodal input, instruction-conditioned reasoning, and the generation of video with visual and semantic consistency. To address this, we introduce VANS, a model that leverages reinforcement learning to align a Vision-Language Model (VLM) with a Video Diffusion Model (VDM) for VNEP. The core of VANS is our proposed Joint-GRPO that orchestrates the VLM and VDM to function as a unit. Driven by a shared reward on their respective output, it optimizes the VLM to produce captions that are both accurate and friendly to visualize, while guiding the VDM to generate videos that are faithful to these captions and the input visual context. To enable this learning, we craft VANS-Data-100K, a dedicated dataset for the VNEP task. Experiments on procedural and predictive benchmarks demonstrate that VANS achieves state-of-the-art performance in both video event prediction and visualization. Codes are released in https://github.com/KlingTeam/VANS.
>
---
#### [replaced 141] ImmerIris: A Large-Scale Dataset and Benchmark for Off-Axis and Unconstrained Iris Recognition in Immersive Applications
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.10113v2](https://arxiv.org/pdf/2510.10113v2)**

> **作者:** Yuxi Mi; Qiuyang Yuan; Zhizhou Zhong; Xuan Zhao; Jiaogen Zhou; Fubao Zhu; Jihong Guan; Shuigeng Zhou
>
> **摘要:** Recently, iris recognition is regaining prominence in immersive applications such as extended reality as a means of seamless user identification. This application scenario introduces unique challenges compared to traditional iris recognition under controlled setups, as the ocular images are primarily captured off-axis and less constrained, causing perspective distortion, intra-subject variation, and quality degradation in iris textures. Datasets capturing these challenges remain limited. This paper fills this gap by presenting a large-scale iris dataset collected via head-mounted displays, termed ImmerIris. It contains 499,791 ocular images from 564 subjects, and is, to our knowledge, the largest public iris dataset to date and among the first dedicated to immersive applications. It is accompanied by a comprehensive set of evaluation protocols that benchmark recognition systems under various challenging conditions. This paper also draws attention to a shared obstacle of current recognition methods, the reliance on a pre-processing, normalization stage, which is fallible in off-axis and unconstrained setups. To this end, this paper further proposes a normalization-free paradigm that directly learns from minimally adjusted ocular images. Despite its simplicity, it outperforms normalization-based prior arts, indicating a promising direction for robust iris recognition.
>
---
#### [replaced 142] MeshCone: Second-Order Cone Programming for Geometrically-Constrained Mesh Enhancement
- **分类: cs.GR; cs.CV; math.OC**

- **链接: [https://arxiv.org/pdf/2412.08484v3](https://arxiv.org/pdf/2412.08484v3)**

> **作者:** Alexander Valverde
>
> **摘要:** Modern geometric generation methods rely heavily on deep learning methods that, while powerful, often lack interpretability and require extensive training data. This work introduces MeshCone, a convex optimization framework for mesh enhancement from partially deformed meshes that requires no training data. We formulate the problem as a second-order cone program where vertex positions are optimized to align with target geometry while enforcing smoothness through convex edge-length regularization. Our convex relaxation enables deterministic, interpretable solutions with proven convergence properties via the Splitting Conic Solver (SCS). We demonstrate robust performance across 56 diverse object categories from ShapeNet and ThreeDScans, achieving superior refinement quality compared to classical baselines while maintaining sub-second inference times. This work establishes a principled baseline demonstrating what convex optimization alone can achieve, providing mathematical guarantees and interpretability that complement data-driven approaches.
>
---
#### [replaced 143] Directed-CP: Directed Collaborative Perception for Connected and Autonomous Vehicles via Proactive Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.08840v4](https://arxiv.org/pdf/2409.08840v4)**

> **作者:** Yihang Tao; Senkang Hu; Zhengru Fang; Yuguang Fang
>
> **备注:** Accepted by ICRA'25
>
> **摘要:** Collaborative perception (CP) leverages visual data from connected and autonomous vehicles (CAV) to enhance an ego vehicle's field of view (FoV). Despite recent progress, current CP methods expand the ego vehicle's 360-degree perceptual range almost equally, which faces two key challenges. Firstly, in areas with uneven traffic distribution, focusing on directions with little traffic offers limited benefits. Secondly, under limited communication budgets, allocating excessive bandwidth to less critical directions lowers the perception accuracy in more vital areas. To address these issues, we propose Direct-CP, a proactive and direction-aware CP system aiming at improving CP in specific directions. Our key idea is to enable an ego vehicle to proactively signal its interested directions and readjust its attention to enhance local directional CP performance. To achieve this, we first propose an RSU-aided direction masking mechanism that assists an ego vehicle in identifying vital directions. Additionally, we design a direction-aware selective attention module to wisely aggregate pertinent features based on ego vehicle's directional priorities, communication budget, and the positional data of CAVs. Moreover, we introduce a direction-weighted detection loss (DWLoss) to capture the divergence between directional CP outcomes and the ground truth, facilitating effective model training. Extensive experiments on the V2X-Sim 2.0 dataset demonstrate that our approach achieves 19.8\% higher local perception accuracy in interested directions and 2.5\% higher overall perception accuracy than the state-of-the-art methods in collaborative 3D object detection tasks. Codes are available at https://github.com/yihangtao/Directed-CP.git.
>
---
#### [replaced 144] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.13589v3](https://arxiv.org/pdf/2506.13589v3)**

> **作者:** Zhucun Xue; Jiangning Zhang; Xurong Xie; Yuxuan Cai; Yong Liu; Xiangtai Li; Dacheng Tao
>
> **备注:** NeurIPS 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) perform well in video understanding but degrade on long videos due to fixed-length context and weak long-term dependency modeling. Retrieval-Augmented Generation (RAG) can expand knowledge dynamically, yet existing video RAG schemes adopt fixed retrieval paradigms that ignore query difficulty. This uniform design causes redundant computation and latency for simple queries, while coarse retrieval for complex, multi-hop reasoning can miss key information. Such single-step retrieval severely limits the trade-off between efficiency and cognitive depth. We propose AdaVideoRAG, an adaptive RAG framework for long-video understanding. A lightweight intent classifier dynamically selects suitable retrieval schemes according to query complexity from the simplest to the most sophisticated. We design an Omni-Knowledge Indexing module that extracts and organizes multi-modal information into three databases: (1) a text base built from clip captions, ASR, and OCR; (2) a visual base; and (3) a knowledge graph for deep semantic understanding. This supports hierarchical knowledge access, from naive retrieval to graph-based retrieval, balancing resource cost and reasoning ability. To evaluate deep understanding, we further construct the HiVU benchmark. Experiments show that AdaVideoRAG significantly improves both efficiency and accuracy on long-video QA tasks and can be seamlessly plugged into existing MLLMs through lightweight APIs, establishing a new paradigm for adaptive retrieval-augmented video analysis.
>
---
#### [replaced 145] "It's trained by non-disabled people": Evaluating How Image Quality Affects Product Captioning with VLMs
- **分类: cs.HC; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08917v2](https://arxiv.org/pdf/2511.08917v2)**

> **作者:** Kapil Garg; Xinru Tang; Jimin Heo; Dwayne R. Morgan; Darren Gergle; Erik B. Sudderth; Anne Marie Piper
>
> **备注:** Paper under review
>
> **摘要:** Vision-Language Models (VLMs) are increasingly used by blind and low-vision (BLV) people to identify and understand products in their everyday lives, such as food, personal products, and household goods. Despite their prevalence, we lack an empirical understanding of how common image quality issues, like blur and misframing of items, affect the accuracy of VLM-generated captions and whether resulting captions meet BLV people's information needs. Grounded in a survey with 86 BLV people, we systematically evaluate how image quality issues affect captions generated by VLMs. We show that the best model recognizes products in images with no quality issues with 98% accuracy, but drops to 75% accuracy overall when quality issues are present, worsening considerably as issues compound. We discuss the need for model evaluations that center on disabled people's experiences throughout the process and offer concrete recommendations for HCI and ML researchers to make VLMs more reliable for BLV people.
>
---
#### [replaced 146] RefDrone: A Challenging Benchmark for Referring Expression Comprehension in Drone Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.00392v3](https://arxiv.org/pdf/2502.00392v3)**

> **作者:** Zhichao Sun; Yepeng Liu; Zhiling Su; Huachao Zhu; Yuliang Gu; Yuda Zou; Zelong Liu; Gui-Song Xia; Bo Du; Yongchao Xu
>
> **摘要:** Drones have become prevalent robotic platforms with diverse applications, showing significant potential in Embodied Artificial Intelligence (Embodied AI). Referring Expression Comprehension (REC) enables drones to locate objects based on natural language expressions, a crucial capability for Embodied AI. Despite advances in REC for ground-level scenes, aerial views introduce unique challenges including varying viewpoints, occlusions and scale variations. To address this gap, we introduce RefDrone, a REC benchmark for drone scenes. RefDrone reveals three key challenges in REC: 1) multi-scale and small-scale target detection; 2) multi-target and no-target samples; 3) complex environment with rich contextual expressions. To efficiently construct this dataset, we develop RDAgent (referring drone annotation framework with multi-agent system), a semi-automated annotation tool for REC tasks. RDAgent ensures high-quality contextual expressions and reduces annotation cost. Furthermore, we propose Number GroundingDINO (NGDINO), a novel method designed to handle multi-target and no-target cases. NGDINO explicitly learns and utilizes the number of objects referred to in the expression. Comprehensive experiments with state-of-the-art REC methods demonstrate that NGDINO achieves superior performance on both the proposed RefDrone and the existing gRefCOCO datasets. The dataset and code are be publicly at https://github.com/sunzc-sunny/refdrone.
>
---
#### [replaced 147] Explainable Cross-Disease Reasoning for Cardiovascular Risk Assessment from LDCT
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.06625v3](https://arxiv.org/pdf/2511.06625v3)**

> **作者:** Yifei Zhang; Jiashuo Zhang; Mojtaba Safari; Xiaofeng Yang; Liang Zhao
>
> **摘要:** Low-dose chest computed tomography (LDCT) inherently captures both pulmonary and cardiac structures, offering a unique opportunity for joint assessment of lung and cardiovascular health. However, most existing approaches treat these domains as independent tasks, overlooking their physiological interplay and shared imaging biomarkers. We propose an Explainable Cross-Disease Reasoning Framework that enables interpretable cardiopulmonary risk assessment from a single LDCT scan. The framework introduces an agentic reasoning process that emulates clinical diagnostic thinking-first perceiving pulmonary findings, then reasoning through established medical knowledge, and finally deriving a cardiovascular judgment with explanatory rationale. It integrates three synergistic components: a pulmonary perception module that summarizes lung abnormalities, a knowledge-guided reasoning module that infers their cardiovascular implications, and a cardiac representation module that encodes structural biomarkers. Their outputs are fused to produce a holistic cardiovascular risk prediction that is both accurate and physiologically grounded. Experiments on the NLST cohort demonstrate that the proposed framework achieves state-of-the-art performance for CVD screening and mortality prediction, outperforming single-disease and purely image-based baselines. Beyond quantitative gains, the framework provides human-verifiable reasoning that aligns with cardiological understanding, revealing coherent links between pulmonary abnormalities and cardiac stress mechanisms. Overall, this work establishes a unified and explainable paradigm for cardiovascular analysis from LDCT, bridging the gap between image-based prediction and mechanism-based medical interpretation.
>
---
#### [replaced 148] DAGLFNet: Deep Feature Attention Guided Global and Local Feature Fusion for Pseudo-Image Point Cloud Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.10471v2](https://arxiv.org/pdf/2510.10471v2)**

> **作者:** Chuang Chen; Yi Lin; Bo Wang; Jing Hu; Xi Wu; Wenyi Ge
>
> **摘要:** Environmental perception systems are crucial for high-precision mapping and autonomous navigation, with LiDAR serving as a core sensor providing accurate 3D point cloud data. Efficiently processing unstructured point clouds while extracting structured semantic information remains a significant challenge. In recent years, numerous pseudo-image-based representation methods have emerged to balance efficiency and performance by fusing 3D point clouds with 2D grids. However, the fundamental inconsistency between the pseudo-image representation and the original 3D information critically undermines 2D-3D feature fusion, posing a primary obstacle for coherent information fusion and leading to poor feature discriminability. This work proposes DAGLFNet, a pseudo-image-based semantic segmentation framework designed to extract discriminative features. It incorporates three key components: first, a Global-Local Feature Fusion Encoding (GL-FFE) module to enhance intra-set local feature correlation and capture global contextual information; second, a Multi-Branch Feature Extraction (MB-FE) network to capture richer neighborhood information and improve the discriminability of contour features; and third, a Feature Fusion via Deep Feature-guided Attention (FFDFA) mechanism to refine cross-channel feature fusion precision. Experimental evaluations demonstrate that DAGLFNet achieves mean Intersection-over-Union (mIoU) scores of 69.9% and 78.7% on the validation sets of SemanticKITTI and nuScenes, respectively. The method achieves an excellent balance between accuracy and efficiency.
>
---
#### [replaced 149] AdaTok: Adaptive Token Compression with Object-Aware Representations for Efficient Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14169v2](https://arxiv.org/pdf/2511.14169v2)**

> **作者:** Xinliang Zhang; Lei Zhu; Hangzhou He; Shuang Zeng; Ourui Fu; Jiakui Hu; Zhengjian Yao; Yanye Lu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated substantial value in unified text-image understanding and reasoning, primarily by converting images into sequences of patch-level tokens that align with their architectural paradigm. However, patch-level tokenization leads to a quadratic growth in image tokens, burdening MLLMs' understanding and reasoning with enormous computation and memory. Additionally, the traditional patch-wise scanning tokenization workflow misaligns with the human vision cognition system, further leading to hallucination and computational redundancy. To address this issue, we propose an object-level token merging strategy for Adaptive Token compression, revealing the consistency with human vision system. The experiments are conducted on multiple comprehensive benchmarks, which show that our approach averagely, utilizes only 10% tokens while achieving almost 96% of the vanilla model's performance. More extensive experimental results in comparison with relevant works demonstrate the superiority of our method in balancing compression ratio and performance. Our code will be available.
>
---
#### [replaced 150] DICE: Distilling Classifier-Free Guidance into Text Embeddings
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.03726v2](https://arxiv.org/pdf/2502.03726v2)**

> **作者:** Zhenyu Zhou; Defang Chen; Can Wang; Chun Chen; Siwei Lyu
>
> **备注:** AAAI 2026 (Oral)
>
> **摘要:** Text-to-image diffusion models are capable of generating high-quality images, but suboptimal pre-trained text representations often result in these images failing to align closely with the given text prompts. Classifier-free guidance (CFG) is a popular and effective technique for improving text-image alignment in the generative process. However, CFG introduces significant computational overhead. In this paper, we present DIstilling CFG by sharpening text Embeddings (DICE) that replaces CFG in the sampling process with half the computational complexity while maintaining similar generation quality. DICE distills a CFG-based text-to-image diffusion model into a CFG-free version by refining text embeddings to replicate CFG-based directions. In this way, we avoid the computational drawbacks of CFG, enabling high-quality, well-aligned image generation at a fast sampling speed. Furthermore, examining the enhancement pattern, we identify the underlying mechanism of DICE that sharpens specific components of text embeddings to preserve semantic information while enhancing fine-grained details. Extensive experiments on multiple Stable Diffusion v1.5 variants, SDXL, and PixArt-$α$ demonstrate the effectiveness of our method. Code is available at https://github.com/zju-pi/dice.
>
---
#### [replaced 151] ReefNet: A Large scale, Taxonomically Enriched Dataset and Benchmark for Hard Coral Classification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.16822v2](https://arxiv.org/pdf/2510.16822v2)**

> **作者:** Yahia Battach; Abdulwahab Felemban; Faizan Farooq Khan; Yousef A. Radwan; Xiang Li; Fabio Marchese; Sara Beery; Burton H. Jones; Francesca Benzoni; Mohamed Elhoseiny
>
> **摘要:** Coral reefs are rapidly declining due to anthropogenic pressures such as climate change, underscoring the urgent need for scalable, automated monitoring. We introduce ReefNet, a large public coral reef image dataset with point-label annotations mapped to the World Register of Marine Species (WoRMS). ReefNet aggregates imagery from 76 curated CoralNet sources and an additional site from Al Wajh in the Red Sea, totaling approximately 925000 genus-level hard coral annotations with expert-verified labels. Unlike prior datasets, which are often limited by size, geography, or coarse labels and are not ML-ready, ReefNet offers fine-grained, taxonomically mapped labels at a global scale to WoRMS. We propose two evaluation settings: (i) a within-source benchmark that partitions each source's images for localized evaluation, and (ii) a cross-source benchmark that withholds entire sources to test domain generalization. We analyze both supervised and zero-shot classification performance on ReefNet and find that while supervised within-source performance is promising, supervised performance drops sharply across domains, and performance is low across the board for zero-shot models, especially for rare and visually similar genera. This provides a challenging benchmark intended to catalyze advances in domain generalization and fine-grained coral classification. We will release our dataset, benchmarking code, and pretrained models to advance robust, domain-adaptive, global coral reef monitoring and conservation.
>
---
#### [replaced 152] From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.01347v3](https://arxiv.org/pdf/2503.01347v3)**

> **作者:** Ruikun Zhang; Yan Yang; Liyuan Pan
>
> **摘要:** Spatial transcriptomics (ST) measures gene expression at fine-grained spatial resolution, offering insights into tissue molecular landscapes. Previous methods for spatial gene expression prediction typically crop spots of interest from histopathology slide images, and train models to map each spot to a corresponding gene expression profile. However, these methods inherently lose the spatial resolution in gene expression: 1) each spot often contains multiple cells with distinct gene expression profiles; 2) spots are typically defined at fixed spatial resolutions, limiting the ability to predict gene expression at varying scales. To address these limitations, this paper presents PixNet, a dense prediction network capable of predicting spatially resolved gene expression across spots of varying sizes and scales directly from histopathology slide images. Different from previous methods that map individual spots to gene expression values, we generate a spatially dense continuous gene expression map from the histopathology slide image, and aggregate values within spots of interest to predict the gene expression. Our PixNet outperforms state-of-the-art methods on four common ST datasets in multiple spatial scales. The source code will be publicly available.
>
---
#### [replaced 153] Comparative Study of UNet-based Architectures for Liver Tumor Segmentation in Multi-Phase Contrast-Enhanced Computed Tomography
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.25522v4](https://arxiv.org/pdf/2510.25522v4)**

> **作者:** Doan-Van-Anh Ly; Thi-Thu-Hien Pham; Thanh-Hai Le
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Segmentation of liver structures in multi-phase contrast-enhanced computed tomography (CECT) plays a crucial role in computer-aided diagnosis and treatment planning for liver diseases, including tumor detection. In this study, we investigate the performance of UNet-based architectures for liver tumor segmentation, starting from the original UNet and extending to UNet3+ with various backbone networks. We evaluate ResNet, Transformer-based, and State-space (Mamba) backbones, all initialized with pretrained weights. Surprisingly, despite the advances in modern architecture, ResNet-based models consistently outperform Transformer- and Mamba-based alternatives across multiple evaluation metrics. To further improve segmentation quality, we introduce attention mechanisms into the backbone and observe that incorporating the Convolutional Block Attention Module (CBAM) yields the best performance. ResNetUNet3+ with CBAM module not only produced the best overlap metrics with a Dice score of 0.755 and IoU of 0.662, but also achieved the most precise boundary delineation, evidenced by the lowest HD95 distance of 77.911. The model's superiority was further cemented by its leading overall accuracy of 0.925 and specificity of 0.926, showcasing its robust capability in accurately identifying both lesion and healthy tissue. To further enhance interpretability, Grad-CAM visualizations were employed to highlight the region's most influential predictions, providing insights into its decision-making process. These findings demonstrate that classical ResNet architecture, when combined with modern attention modules, remain highly competitive for medical image segmentation tasks, offering a promising direction for liver tumor detection in clinical practice.
>
---
#### [replaced 154] CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12919v2](https://arxiv.org/pdf/2511.12919v2)**

> **作者:** Dexin Zuo; Ang Li; Wei Wang; Wenxian Yu; Danping Zou
>
> **备注:** 7 pages, accepted by AAAI 2026 (oral)
>
> **摘要:** Object 6D pose estimation, a crucial task for robotics and augmented reality applications, becomes particularly challenging when dealing with novel objects whose 3D models are not readily available. To reduce dependency on 3D models, recent studies have explored one-reference-based pose estimation, which requires only a single reference view instead of a complete 3D model. However, existing methods that rely on real-valued coordinate regression suffer from limited global consistency due to the local nature of convolutional architectures and face challenges in symmetric or occluded scenarios owing to a lack of uncertainty modeling. We present CoordAR, a novel autoregressive framework for one-reference 6D pose estimation of unseen objects. CoordAR formulates 3D-3D correspondences between the reference and query views as a map of discrete tokens, which is obtained in an autoregressive and probabilistic manner. To enable accurate correspondence regression, CoordAR introduces 1) a novel coordinate map tokenization that enables probabilistic prediction over discretized 3D space; 2) a modality-decoupled encoding strategy that separately encodes RGB appearance and coordinate cues; and 3) an autoregressive transformer decoder conditioned on both position-aligned query features and the partially generated token sequence. With these novel mechanisms, CoordAR significantly outperforms existing methods on multiple benchmarks and demonstrates strong robustness to symmetry, occlusion, and other challenges in real-world tests.
>
---
#### [replaced 155] AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.11025v2](https://arxiv.org/pdf/2511.11025v2)**

> **作者:** Jirong Zha; Yuxuan Fan; Tianyu Zhang; Geng Chen; Yingfeng Chen; Chen Gao; Xinlei Chen
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown promise in single-agent vision tasks, yet benchmarks for evaluating multi-agent collaborative perception remain scarce. This gap is critical, as multi-drone systems provide enhanced coverage, robustness, and collaboration compared to single-sensor setups. Existing multi-image benchmarks mainly target basic perception tasks using high-quality single-agent images, thus failing to evaluate MLLMs in more complex, egocentric collaborative scenarios, especially under real-world degraded perception conditions.To address these challenges, we introduce AirCopBench, the first comprehensive benchmark designed to evaluate MLLMs in embodied aerial collaborative perception under challenging perceptual conditions. AirCopBench includes 14.6k+ questions derived from both simulator and real-world data, spanning four key task dimensions: Scene Understanding, Object Understanding, Perception Assessment, and Collaborative Decision, across 14 task types. We construct the benchmark using data from challenging degraded-perception scenarios with annotated collaborative events, generating large-scale questions through model-, rule-, and human-based methods under rigorous quality control. Evaluations on 40 MLLMs show significant performance gaps in collaborative perception tasks, with the best model trailing humans by 24.38% on average and exhibiting inconsistent results across tasks. Fine-tuning experiments further confirm the feasibility of sim-to-real transfer in aerial collaborative perception and reasoning.
>
---
#### [replaced 156] SPAGS: Sparse-View Articulated Object Reconstruction from Single State via Planar Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17092v2](https://arxiv.org/pdf/2511.17092v2)**

> **作者:** Di Wu; Liu Liu; Xueyu Yuan; Qiaojun Yu; Wenxiao Chen; Ruilong Yan; Yiming Tang; Liangtu Song
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Articulated objects are ubiquitous in daily environments, and their 3D reconstruction holds great significance across various fields. However, existing articulated object reconstruction methods typically require costly inputs such as multi-stage and multi-view observations. To address the limitations, we propose a category-agnostic articulated object reconstruction framework via planar Gaussian Splatting, which only uses sparse-view RGB images from a single state. Specifically, we first introduce a Gaussian information field to perceive the optimal sparse viewpoints from candidate camera poses. Then we compress 3D Gaussians into planar Gaussians to facilitate accurate estimation of normal and depth. The planar Gaussians are optimized in a coarse-to-fine manner through depth smooth regularization and few-shot diffusion. Moreover, we introduce a part segmentation probability for each Gaussian primitive and update them by back-projecting part segmentation masks of renderings. Extensive experimental results demonstrate that our method achieves higher-fidelity part-level surface reconstruction on both synthetic and real-world data than existing methods. Codes will be made publicly available.
>
---
#### [replaced 157] Motion-R1: Enhancing Motion Generation with Decomposed Chain-of-Thought and RL Binding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.10353v4](https://arxiv.org/pdf/2506.10353v4)**

> **作者:** Runqi Ouyang; Haoyun Li; Zhenyuan Zhang; Xiaofeng Wang; Zeyu Zhang; Zheng Zhu; Guan Huang; Sirui Han; Xingang Wang
>
> **摘要:** Text-to-Motion generation has become a fundamental task in human-machine interaction, enabling the synthesis of realistic human motions from natural language descriptions. Although recent advances in large language models and reinforcement learning have contributed to high-quality motion generation, two major challenges remain. Existing approaches often fail to capture the temporal and causal complexities inherent in natural language, leading to oversimplified or incoherent motions. Additionally, RL-based methods are frequently overly complex, hindering their scalability and adaptability across various motion generation tasks. To address these challenges, we propose Motion-R1, a novel framework that combines decomposed Chain-of-Thought reasoning with reinforcement learning to enhance both the quality and interpretability of generated motions. Specifically, we introduce the Decomposed CoT Data Engine, which leverages an automated pipeline to synthesize high-quality reasoning data, allowing the model to better capture the temporal dependencies and causal relationships of human motion. We also propose RL Binding, a reinforcement learning strategy that incorporates multi-modal text-motion alignment into the RL reward function, guiding the model to produce motions that are both semantically accurate and motionally realistic. Extensive experiments across benchmark datasets demonstrate that Motion-R1 achieves state-of-the-art performance, with a 3.5% improvement in MM-Dist on HumanML3D and improvements in R-Precision and FID on KIT-ML and BABEL, surpassing existing methods across key metrics and highlighting its superior capability in handling complex motion generation tasks. Project page: https://motion-r1.github.io/.
>
---
#### [replaced 158] Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16301v2](https://arxiv.org/pdf/2511.16301v2)**

> **作者:** Minseok Seo; Mark Hamilton; Changick Kim
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** We present \textbf{Upsample Anything}, a lightweight test-time optimization (TTO) framework that restores low-resolution features to high-resolution, pixel-wise outputs without any training. Although Vision Foundation Models demonstrate strong generalization across diverse downstream tasks, their representations are typically downsampled by 14x/16x (e.g., ViT), which limits their direct use in pixel-level applications. Existing feature upsampling approaches depend on dataset-specific retraining or heavy implicit optimization, restricting scalability and generalization. Upsample Anything addresses these issues through a simple per-image optimization that learns an anisotropic Gaussian kernel combining spatial and range cues, effectively bridging Gaussian Splatting and Joint Bilateral Upsampling. The learned kernel acts as a universal, edge-aware operator that transfers seamlessly across architectures and modalities, enabling precise high-resolution reconstruction of features, depth, or probability maps. It runs in only $\approx0.419 \text{s}$ per 224x224 image and achieves state-of-the-art performance on semantic segmentation, depth estimation, and both depth and probability map upsampling. \textbf{Project page:} \href{https://seominseok0429.github.io/Upsample-Anything/}{https://seominseok0429.github.io/Upsample-Anything/}
>
---
#### [replaced 159] Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.15065v2](https://arxiv.org/pdf/2511.15065v2)**

> **作者:** Cheng Yang; Haiyuan Wan; Yiran Peng; Xin Cheng; Zhaoyang Yu; Jiayi Zhang; Junchi Yu; Xinlei Yu; Xiawu Zheng; Dongzhan Zhou; Chenglin Wu
>
> **摘要:** Video Models have achieved remarkable success in high-fidelity video generation with coherent motion dynamics. Analogous to the development from text generation to text-based reasoning in language modeling, the development of video models motivates us to ask: Can video models reason via video generation? Compared with the discrete text corpus, video grounds reasoning in explicit spatial layouts and temporal continuity, which serves as an ideal substrate for spatial reasoning. In this work, we explore the reasoning via video paradigm and introduce VR-Bench -- a comprehensive benchmark designed to systematically evaluate video models' reasoning capabilities. Grounded in maze-solving tasks that inherently require spatial planning and multi-step reasoning, VR-Bench contains 7,920 procedurally generated videos across five maze types and diverse visual styles. Our empirical analysis demonstrates that SFT can efficiently elicit the reasoning ability of video model. Video models exhibit stronger spatial perception during reasoning, outperforming leading VLMs and generalizing well across diverse scenarios, tasks, and levels of complexity. We further discover a test-time scaling effect, where diverse sampling during inference improves reasoning reliability by 10--20%. These findings highlight the unique potential and scalability of reasoning via video for spatial reasoning tasks.
>
---
