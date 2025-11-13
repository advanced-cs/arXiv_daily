# 计算机视觉 cs.CV

- **最新发布 94 篇**

- **更新 82 篇**

## 最新发布

#### [new 001] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images
- **分类: cs.CV**

- **简介: 论文提出WDT-MD，用于眼底图像中微动脉瘤的检测，解决扩散模型易复制输入、误检其他病灶、正常结构重建差等问题，通过小波变换与Transformer结合，引入噪声条件与伪正常合成，提升检测精度。**

- **链接: []()**

> **作者:** Yifei Sun; Yuzhi He; Junhao Jia; Jinhong Wang; Ruiquan Ge; Changmiao Wang; Hongxia Xu
>
> **备注:** 9 pages, 6 figures, 8 tables, accepted by AAAI 2026
>
> **摘要:** Microaneurysms (MAs), the earliest pathognomonic signs of Diabetic Retinopathy (DR), present as sub-60 $μm$ lesions in fundus images with highly variable photometric and morphological characteristics, rendering manual screening not only labor-intensive but inherently error-prone. While diffusion-based anomaly detection has emerged as a promising approach for automated MA screening, its clinical application is hindered by three fundamental limitations. First, these models often fall prey to "identity mapping", where they inadvertently replicate the input image. Second, they struggle to distinguish MAs from other anomalies, leading to high false positives. Third, their suboptimal reconstruction of normal features hampers overall performance. To address these challenges, we propose a Wavelet Diffusion Transformer framework for MA Detection (WDT-MD), which features three key innovations: a noise-encoded image conditioning mechanism to avoid "identity mapping" by perturbing image conditions during training; pseudo-normal pattern synthesis via inpainting to introduce pixel-level supervision, enabling discrimination between MAs and other anomalies; and a wavelet diffusion Transformer architecture that combines the global modeling capability of diffusion Transformers with multi-scale wavelet analysis to enhance reconstruction of normal retinal features. Comprehensive experiments on the IDRiD and e-ophtha MA datasets demonstrate that WDT-MD outperforms state-of-the-art methods in both pixel-level and image-level MA detection. This advancement holds significant promise for improving early DR screening.
>
---
#### [new 002] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference
- **分类: cs.CV**

- **简介: 该论文针对CLIP模型对抗攻击脆弱性问题，提出方向正交对抗反击（DOC），通过正交梯度与动量更新增强反击多样性，提升对各类对抗扰动的泛化防御能力，同时保持清洁样本性能。**

- **链接: []()**

> **作者:** Chengze Jiang; Minjing Dong; Xinli Shi; Jie Gui
>
> **备注:** Accepted to AAAI-2026 Oral
>
> **摘要:** Vision-language pre-training models (VLPs) demonstrate strong multimodal understanding and zero-shot generalization, yet remain vulnerable to adversarial examples, raising concerns about their reliability. Recent work, Test-Time Counterattack (TTC), improves robustness by generating perturbations that maximize the embedding deviation of adversarial inputs using PGD, pushing them away from their adversarial representations. However, due to the fundamental difference in optimization objectives between adversarial attacks and counterattacks, generating counterattacks solely based on gradients with respect to the adversarial input confines the search to a narrow space. As a result, the counterattacks could overfit limited adversarial patterns and lack the diversity to fully neutralize a broad range of perturbations. In this work, we argue that enhancing the diversity and coverage of counterattacks is crucial to improving adversarial robustness in test-time defense. Accordingly, we propose Directional Orthogonal Counterattack (DOC), which augments counterattack optimization by incorporating orthogonal gradient directions and momentum-based updates. This design expands the exploration of the counterattack space and increases the diversity of perturbations, which facilitates the discovery of more generalizable counterattacks and ultimately improves the ability to neutralize adversarial perturbations. Meanwhile, we present a directional sensitivity score based on averaged cosine similarity to boost DOC by improving example discrimination and adaptively modulating the counterattack strength. Extensive experiments on 16 datasets demonstrate that DOC improves adversarial robustness under various attacks while maintaining competitive clean accuracy. Code is available at https://github.com/bookman233/DOC.
>
---
#### [new 003] An ICTM-RMSAV Framework for Bias-Field Aware Image Segmentation under Poisson and Multiplicative Noise
- **分类: cs.CV; math.OC**

- **简介: 该论文针对含泊松与乘性噪声及强度不均匀的图像分割问题，提出ICTM-RMSAV框架，融合I散度与自适应TV正则化去噪，并引入空间自适应权重与偏置场校正，提升分割精度与鲁棒性。**

- **链接: []()**

> **作者:** Xinyu Wang; Wenjun Yao; Fanghui Song; Zhichang Guo
>
> **摘要:** Image segmentation is a core task in image processing, yet many methods degrade when images are heavily corrupted by noise and exhibit intensity inhomogeneity. Within the iterative-convolution thresholding method (ICTM) framework, we propose a variational segmentation model that integrates denoising terms. Specifically, the denoising component consists of an I-divergence term and an adaptive total-variation (TV) regularizer, making the model well suited to images contaminated by Gamma--distributed multiplicative noise and Poisson noise. A spatially adaptive weight derived from a gray-level indicator guides diffusion differently across regions of varying intensity. To further address intensity inhomogeneity, we estimate a smoothly varying bias field, which improves segmentation accuracy. Regions are represented by characteristic functions, with contour length encoded accordingly. For efficient optimization, we couple ICTM with a relaxed modified scalar auxiliary variable (RMSAV) scheme. Extensive experiments on synthetic and real-world images with intensity inhomogeneity and diverse noise types show that the proposed model achieves superior accuracy and robustness compared with competing approaches.
>
---
#### [new 004] Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文针对移动红外小目标检测任务，提出TDCNet，通过新型时序差分卷积（TDC）融合运动线索与时空特征，并结合TDC引导的注意力机制，有效抑制背景干扰，提升检测精度。**

- **链接: []()**

> **作者:** Houzhang Fang; Shukai Guo; Qiuhuan Chen; Yi Chang; Luxin Yan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Moving infrared small target detection (IRSTD) plays a critical role in practical applications, such as surveillance of unmanned aerial vehicles (UAVs) and UAV-based search system. Moving IRSTD still remains highly challenging due to weak target features and complex background interference. Accurate spatio-temporal feature modeling is crucial for moving target detection, typically achieved through either temporal differences or spatio-temporal (3D) convolutions. Temporal difference can explicitly leverage motion cues but exhibits limited capability in extracting spatial features, whereas 3D convolution effectively represents spatio-temporal features yet lacks explicit awareness of motion dynamics along the temporal dimension. In this paper, we propose a novel moving IRSTD network (TDCNet), which effectively extracts and enhances spatio-temporal features for accurate target detection. Specifically, we introduce a novel temporal difference convolution (TDC) re-parameterization module that comprises three parallel TDC blocks designed to capture contextual dependencies across different temporal ranges. Each TDC block fuses temporal difference and 3D convolution into a unified spatio-temporal convolution representation. This re-parameterized module can effectively capture multi-scale motion contextual features while suppressing pseudo-motion clutter in complex backgrounds, significantly improving detection performance. Moreover, we propose a TDC-guided spatio-temporal attention mechanism that performs cross-attention between the spatio-temporal features from the TDC-based backbone and a parallel 3D backbone. This mechanism models their global semantic dependencies to refine the current frame's features. Extensive experiments on IRSTD-UAV and public infrared datasets demonstrate that our TDCNet achieves state-of-the-art detection performance in moving target detection.
>
---
#### [new 005] Efficient and Effective In-context Demonstration Selection with Coreset
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型的上下文学习中示范样本选择效率低、效果差的问题，提出CoDR框架，通过聚类剪枝构建多样核集，并设计双检索机制，实现高效且有效的示范样本选择。**

- **链接: []()**

> **作者:** Zihua Wang; Jiarui Wang; Haiyang Xu; Ming Yan; Fei Huang; Xu Yang; Xiu-Shen Wei; Siya Mi; Yu Zhang
>
> **备注:** This paper is accepted by AAAI26
>
> **摘要:** In-context learning (ICL) has emerged as a powerful paradigm for Large Visual Language Models (LVLMs), enabling them to leverage a few examples directly from input contexts. However, the effectiveness of this approach is heavily reliant on the selection of demonstrations, a process that is NP-hard. Traditional strategies, including random, similarity-based sampling and infoscore-based sampling, often lead to inefficiencies or suboptimal performance, struggling to balance both efficiency and effectiveness in demonstration selection. In this paper, we propose a novel demonstration selection framework named Coreset-based Dual Retrieval (CoDR). We show that samples within a diverse subset achieve a higher expected mutual information. To implement this, we introduce a cluster-pruning method to construct a diverse coreset that aligns more effectively with the query while maintaining diversity. Additionally, we develop a dual retrieval mechanism that enhances the selection process by achieving global demonstration selection while preserving efficiency. Experimental results demonstrate that our method significantly improves the ICL performance compared to the existing strategies, providing a robust solution for effective and efficient demonstration selection.
>
---
#### [new 006] Taming Object Hallucinations with Verified Atomic Confidence Estimation
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型中的物体幻觉问题，提出TACO框架，通过原子化查询、自一致性与自置信度校准，无需外部模型即可提升回答真实性与置信度校准效果。**

- **链接: []()**

> **作者:** Jiarui Liu; Weihao Xuan; Zhijing Jin; Mona Diab
>
> **摘要:** Multimodal Large Language Models (MLLMs) often suffer from hallucinations, particularly errors in object existence, attributes, or relations, which undermine their reliability. We introduce TACO (Verified Atomic Confidence Estimation), a simple framework that mitigates hallucinations through self-verification and confidence calibration without relying on external vision experts. TACO decomposes responses into atomic queries, paraphrases them to reduce sensitivity to wording, and estimates confidence using self-consistency (black-box) or self-confidence (gray-box) aggregation, before refining answers with a language model. Experiments on five benchmarks (POPE, MME, HallusionBench, AMBER, and MM-Hal Bench) with two MLLMs (\texttt{LLaVA-1.5-7B} and \texttt{CogVLM2}) show that TACO consistently outperforms direct prompting and Visual Contrastive Decoding, reduces systematic biases, and improves confidence calibration, demonstrating its effectiveness in enhancing the faithfulness of MLLMs.
>
---
#### [new 007] Time-to-Move: Training-Free Motion Controlled Video Generation via Dual-Clock Denoising
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.MM**

- **简介: 论文提出Time-to-Move（TTM），一种无需训练的视频生成框架，通过双时钟去噪实现精准运动与外观控制，利用用户简易操作生成的粗略动画作为运动引导，兼容任意扩散模型，突破文本提示的局限。**

- **链接: []()**

> **作者:** Assaf Singer; Noam Rotstein; Amir Mann; Ron Kimmel; Or Litany
>
> **摘要:** Diffusion-based video generation can create realistic videos, yet existing image- and text-based conditioning fails to offer precise motion control. Prior methods for motion-conditioned synthesis typically require model-specific fine-tuning, which is computationally expensive and restrictive. We introduce Time-to-Move (TTM), a training-free, plug-and-play framework for motion- and appearance-controlled video generation with image-to-video (I2V) diffusion models. Our key insight is to use crude reference animations obtained through user-friendly manipulations such as cut-and-drag or depth-based reprojection. Motivated by SDEdit's use of coarse layout cues for image editing, we treat the crude animations as coarse motion cues and adapt the mechanism to the video domain. We preserve appearance with image conditioning and introduce dual-clock denoising, a region-dependent strategy that enforces strong alignment in motion-specified regions while allowing flexibility elsewhere, balancing fidelity to user intent with natural dynamics. This lightweight modification of the sampling process incurs no additional training or runtime cost and is compatible with any backbone. Extensive experiments on object and camera motion benchmarks show that TTM matches or exceeds existing training-based baselines in realism and motion control. Beyond this, TTM introduces a unique capability: precise appearance control through pixel-level conditioning, exceeding the limits of text-only prompting. Visit our project page for video examples and code: https://time-to-move.github.io/.
>
---
#### [new 008] LLM-Guided Probabilistic Fusion for Label-Efficient Document Layout Analysis
- **分类: cs.CV**

- **简介: 该论文面向文档布局分析任务，提出LLM引导的概率融合框架，利用文本预训练LLM的结构先验与视觉检测结果进行自适应加权融合，显著降低标注数据需求，在仅5%标签下达到SOTA性能，且支持隐私保护部署。**

- **链接: []()**

> **作者:** Ibne Farabi Shihab; Sanjeda Akter; Anuj Sharma
>
> **摘要:** Document layout understanding remains data-intensive despite advances in semi-supervised learning. We present a framework that enhances semi-supervised detection by fusing visual predictions with structural priors from text-pretrained LLMs via principled probabilistic weighting. Given unlabeled documents, an OCR-LLM pipeline infers hierarchical regions which are combined with teacher detector outputs through inverse-variance fusion to generate refined pseudo-labels.Our method demonstrates consistent gains across model scales. With a lightweight SwiftFormer backbone (26M params), we achieve 88.2$\pm$0.3 AP using only 5\% labels on PubLayNet. When applied to document-pretrained LayoutLMv3 (133M params), our fusion framework reaches 89.7$\pm$0.4 AP, surpassing both LayoutLMv3 with standard semi-supervised learning (89.1$\pm$0.4 AP, p=0.02) and matching UDOP~\cite{udop} (89.8 AP) which requires 100M+ pages of multimodal pretraining. This demonstrates that LLM structural priors are complementary to both lightweight and pretrained architectures. Key findings include: (1) learned instance-adaptive gating improves over fixed weights by +0.9 AP with data-dependent PAC bounds correctly predicting convergence; (2) open-source LLMs enable privacy-preserving deployment with minimal loss (Llama-3-70B: 87.1 AP lightweight, 89.4 AP with LayoutLMv3); (3) LLMs provide targeted semantic disambiguation (18.7\% of cases, +3.8 AP gain) beyond simple text heuristics.Total system cost includes \$12 for GPT-4o-mini API or 17 GPU-hours for local Llama-3-70B per 50K pages, amortized across training runs.
>
---
#### [new 009] Boosting Adversarial Transferability via Ensemble Non-Attention
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出NAMEA攻击方法，通过融合非注意力区域梯度提升跨架构对抗样本的迁移性，解决异构模型（CNN与ViT）梯度差异大导致攻击效果差的问题，首次实现非注意力区域的梯度解耦与元学习融合。**

- **链接: []()**

> **作者:** Yipeng Zou; Qin Liu; Jie Wu; Yu Peng; Guo Chen; Hui Zhou; Guanghui Ye
>
> **摘要:** Ensemble attacks integrate the outputs of surrogate models with diverse architectures, which can be combined with various gradient-based attacks to improve adversarial transferability. However, previous work shows unsatisfactory attack performance when transferring across heterogeneous model architectures. The main reason is that the gradient update directions of heterogeneous surrogate models differ widely, making it hard to reduce the gradient variance of ensemble models while making the best of individual model. To tackle this challenge, we design a novel ensemble attack, NAMEA, which for the first time integrates the gradients from the non-attention areas of ensemble models into the iterative gradient optimization process. Our design is inspired by the observation that the attention areas of heterogeneous models vary sharply, thus the non-attention areas of ViTs are likely to be the focus of CNNs and vice versa. Therefore, we merge the gradients respectively from the attention and non-attention areas of ensemble models so as to fuse the transfer information of CNNs and ViTs. Specifically, we pioneer a new way of decoupling the gradients of non-attention areas from those of attention areas, while merging gradients by meta-learning. Empirical evaluations on ImageNet dataset indicate that NAMEA outperforms AdaEA and SMER, the state-of-the-art ensemble attacks by an average of 15.0% and 9.6%, respectively. This work is the first attempt to explore the power of ensemble non-attention in boosting cross-architecture transferability, providing new insights into launching ensemble attacks.
>
---
#### [new 010] Assessing Identity Leakage in Talking Face Generation: Metrics and Evaluation Framework
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对说话人脸生成中的身份泄露问题，提出一种通用评估框架，通过三种测试场景和新指标（如唇部同步偏差）量化音频驱动与参考图像间的非预期关联，提升生成真实性评估的可靠性。**

- **链接: []()**

> **作者:** Dogucan Yaman; Fevziye Irem Eyiokur; Hazım Kemal Ekenel; Alexander Waibel
>
> **摘要:** Inpainting-based talking face generation aims to preserve video details such as pose, lighting, and gestures while modifying only lip motion, often using an identity reference image to maintain speaker consistency. However, this mechanism can introduce lip leaking, where generated lips are influenced by the reference image rather than solely by the driving audio. Such leakage is difficult to detect with standard metrics and conventional test setup. To address this, we propose a systematic evaluation methodology to analyze and quantify lip leakage. Our framework employs three complementary test setups: silent-input generation, mismatched audio-video pairing, and matched audio-video synthesis. We also introduce derived metrics including lip-sync discrepancy and silent-audio-based lip-sync scores. In addition, we study how different identity reference selections affect leakage, providing insights into reference design. The proposed methodology is model-agnostic and establishes a more reliable benchmark for future research in talking face generation.
>
---
#### [new 011] OUGS: Active View Selection via Object-aware Uncertainty Estimation in 3DGS
- **分类: cs.CV; cs.CG; cs.GR; cs.HC**

- **简介: 论文提出OUGS，面向3D高斯泼溅（3DGS）的主动视图选择任务，解决背景干扰导致的对象重建效率低问题。通过物理参数传播构建对象感知不确定性模型，结合语义掩码精准聚焦目标对象，提升重建质量与效率。**

- **链接: []()**

> **作者:** Haiyi Li; Qi Chen; Denis Kalkofen; Hsiang-Ting Chen
>
> **备注:** 11 pages (10 main + 1 appendix), 7 figures, 3 tables. Preprint, under review for Eurographics 2026
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have achieved state-of-the-art results for novel view synthesis. However, efficiently capturing high-fidelity reconstructions of specific objects within complex scenes remains a significant challenge. A key limitation of existing active reconstruction methods is their reliance on scene-level uncertainty metrics, which are often biased by irrelevant background clutter and lead to inefficient view selection for object-centric tasks. We present OUGS, a novel framework that addresses this challenge with a more principled, physically-grounded uncertainty formulation for 3DGS. Our core innovation is to derive uncertainty directly from the explicit physical parameters of the 3D Gaussian primitives (e.g., position, scale, rotation). By propagating the covariance of these parameters through the rendering Jacobian, we establish a highly interpretable uncertainty model. This foundation allows us to then seamlessly integrate semantic segmentation masks to produce a targeted, object-aware uncertainty score that effectively disentangles the object from its environment. This allows for a more effective active view selection strategy that prioritizes views critical to improving object fidelity. Experimental evaluations on public datasets demonstrate that our approach significantly improves the efficiency of the 3DGS reconstruction process and achieves higher quality for targeted objects compared to existing state-of-the-art methods, while also serving as a robust uncertainty estimator for the global scene.
>
---
#### [new 012] T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection
- **分类: cs.CV**

- **简介: T-Rex-Omni面向开放集目标检测，引入负视觉提示抑制语义混淆的干扰物，提出无训练的NNC模块与NNH损失，实现正负提示联合建模，显著提升零样本检测性能，尤其在长尾场景下表现突出。**

- **链接: []()**

> **作者:** Jiazhou Zhou; Qing Jiang; Kanghao Chen; Lutao Jiang; Yuanhuiyi Lyu; Ying-Cong Chen; Lei Zhang
>
> **备注:** Accepted by AAAI 2026. Main paper: 7 pages with 4 figures; Appendix: 8 pages with 7 figures
>
> **摘要:** Object detection methods have evolved from closed-set to open-set paradigms over the years. Current open-set object detectors, however, remain constrained by their exclusive reliance on positive indicators based on given prompts like text descriptions or visual exemplars. This positive-only paradigm experiences consistent vulnerability to visually similar but semantically different distractors. We propose T-Rex-Omni, a novel framework that addresses this limitation by incorporating negative visual prompts to negate hard negative distractors. Specifically, we first introduce a unified visual prompt encoder that jointly processes positive and negative visual prompts. Next, a training-free Negating Negative Computing (NNC) module is proposed to dynamically suppress negative responses during the probability computing stage. To further boost performance through fine-tuning, our Negating Negative Hinge (NNH) loss enforces discriminative margins between positive and negative embeddings. T-Rex-Omni supports flexible deployment in both positive-only and joint positive-negative inference modes, accommodating either user-specified or automatically generated negative examples. Extensive experiments demonstrate remarkable zero-shot detection performance, significantly narrowing the performance gap between visual-prompted and text-prompted methods while showing particular strength in long-tailed scenarios (51.2 AP_r on LVIS-minival). This work establishes negative prompts as a crucial new dimension for advancing open-set visual recognition systems.
>
---
#### [new 013] Hand Held Multi-Object Tracking Dataset in American Football
- **分类: cs.CV**

- **简介: 该论文构建了首个美式橄榄球多目标跟踪数据集，解决现有数据集缺失导致方法难以公平评估的问题，并验证了微调检测与重识别模型可显著提升密集场景下的跟踪精度。**

- **链接: []()**

> **作者:** Rintaro Otsubo; Kanta Sawafuji; Hideo Saito
>
> **摘要:** Multi-Object Tracking (MOT) plays a critical role in analyzing player behavior from videos, enabling performance evaluation. Current MOT methods are often evaluated using publicly available datasets. However, most of these focus on everyday scenarios such as pedestrian tracking or are tailored to specific sports, including soccer and basketball. Despite the inherent challenges of tracking players in American football, such as frequent occlusion and physical contact, no standardized dataset has been publicly available, making fair comparisons between methods difficult. To address this gap, we constructed the first dedicated detection and tracking dataset for the American football players and conducted a comparative evaluation of various detection and tracking methods. Our results demonstrate that accurate detection and tracking can be achieved even in crowded scenarios. Fine-tuning detection models improved performance over pre-trained models. Furthermore, when these fine-tuned detectors and re-identification models were integrated into tracking systems, we observed notable improvements in tracking accuracy compared to existing approaches. This work thus enables robust detection and tracking of American football players in challenging, high-density scenarios previously underserved by conventional methods.
>
---
#### [new 014] SIFT-Graph: Benchmarking Multimodal Defense Against Image Adversarial Attacks With Robust Feature Graph
- **分类: cs.CV**

- **简介: SIFT-Graph提出一种多模态防御框架，结合SIFT关键点与图注意力网络，提取鲁棒结构特征，融合至视觉模型以增强对抗攻击下的鲁棒性，同时保持高干净准确率。**

- **链接: []()**

> **作者:** Jingjie He; Weijie Liang; Zihan Shan; Matthew Caesar
>
> **备注:** Accepted by ICCV2025 Workshop, short paper
>
> **摘要:** Adversarial attacks expose a fundamental vulnerability in modern deep vision models by exploiting their dependence on dense, pixel-level representations that are highly sensitive to imperceptible perturbations. Traditional defense strategies typically operate within this fragile pixel domain, lacking mechanisms to incorporate inherently robust visual features. In this work, we introduce SIFT-Graph, a multimodal defense framework that enhances the robustness of traditional vision models by aggregating structurally meaningful features extracted from raw images using both handcrafted and learned modalities. Specifically, we integrate Scale-Invariant Feature Transform keypoints with a Graph Attention Network to capture scale and rotation invariant local structures that are resilient to perturbations. These robust feature embeddings are then fused with traditional vision model, such as Vision Transformer and Convolutional Neural Network, to form a unified, structure-aware and perturbation defensive model. Preliminary results demonstrate that our method effectively improves the visual model robustness against gradient-based white box adversarial attacks, while incurring only a marginal drop in clean accuracy.
>
---
#### [new 015] Towards Trustworthy Dermatology MLLMs: A Benchmark and Multimodal Evaluator for Diagnostic Narratives
- **分类: cs.CV**

- **简介: 该论文面向皮肤病诊断叙事生成任务，解决评估不可靠问题，提出DermBench基准与DermEval评估器，实现对多模态大模型生成文本的临床可解释、自动化、高精度评估。**

- **链接: []()**

> **作者:** Yuhao Shen; Jiahe Qian; Shuping Zhang; Zhangtianyi Chen; Tao Lu; Juexiao Zhou
>
> **摘要:** Multimodal large language models (LLMs) are increasingly used to generate dermatology diagnostic narratives directly from images. However, reliable evaluation remains the primary bottleneck for responsible clinical deployment. We introduce a novel evaluation framework that combines DermBench, a meticulously curated benchmark, with DermEval, a robust automatic evaluator, to enable clinically meaningful, reproducible, and scalable assessment. We build DermBench, which pairs 4,000 real-world dermatology images with expert-certified diagnostic narratives and uses an LLM-based judge to score candidate narratives across clinically grounded dimensions, enabling consistent and comprehensive evaluation of multimodal models. For individual case assessment, we train DermEval, a reference-free multimodal evaluator. Given an image and a generated narrative, DermEval produces a structured critique along with an overall score and per-dimension ratings. This capability enables fine-grained, per-case analysis, which is critical for identifying model limitations and biases. Experiments on a diverse dataset of 4,500 cases demonstrate that DermBench and DermEval achieve close alignment with expert ratings, with mean deviations of 0.251 and 0.117 (out of 5), respectively, providing reliable measurement of diagnostic ability and trustworthiness across different multimodal LLMs.
>
---
#### [new 016] Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Network
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向几何深度学习，解决传统方法仅用静态单子空间表征导致的结构捕捉不足问题，提出拓扑驱动的多子空间融合框架，通过自适应加权与Fréchet均值融合，提升非欧氏空间中的判别力与鲁棒性。**

- **链接: []()**

> **作者:** Xuan Yu; Tianyang Xu
>
> **备注:** 9 pages, 2 figures, accepted at AAAI 2026
>
> **摘要:** Grassmannian manifold offers a powerful carrier for geometric representation learning by modelling high-dimensional data as low-dimensional subspaces. However, existing approaches predominantly rely on static single-subspace representations, neglecting the dynamic interplay between multiple subspaces critical for capturing complex geometric structures. To address this limitation, we propose a topology-driven multi-subspace fusion framework that enables adaptive subspace collaboration on the Grassmannian. Our solution introduces two key innovations: (1) Inspired by the Kolmogorov-Arnold representation theorem, an adaptive multi-subspace modelling mechanism is proposed that dynamically selects and weights task-relevant subspaces via topological convergence analysis, and (2) a multi-subspace interaction block that fuses heterogeneous geometric representations through Fréchet mean optimisation on the manifold. Theoretically, we establish the convergence guarantees of adaptive subspaces under a projection metric topology, ensuring stable gradient-based optimisation. Practically, we integrate Riemannian batch normalisation and mutual information regularisation to enhance discriminability and robustness. Extensive experiments on 3D action recognition (HDM05, FPHA), EEG classification (MAMEM-SSVEPII), and graph tasks demonstrate state-of-the-art performance. Our work not only advances geometric deep learning but also successfully adapts the proven multi-channel interaction philosophy of Euclidean networks to non-Euclidean domains, achieving superior discriminability and interpretability.
>
---
#### [new 017] Machines Serve Human: A Novel Variable Human-machine Collaborative Compression Framework
- **分类: cs.CV**

- **简介: 该论文提出Diff-FCHM框架，首次以机器视觉为导向重构人机协同压缩，解决传统方法依赖人眼感知导致的冗余问题，通过可变比特率策略与扩散先验融合，实现机器与人类视觉的高效高质量压缩。**

- **链接: []()**

> **作者:** Zifu Zhang; Shengxi Li; Xiancheng Sun; Mai Xu; Zhengyuan Liu; Jingyuan Xia
>
> **摘要:** Human-machine collaborative compression has been receiving increasing research efforts for reducing image/video data, serving as the basis for both human perception and machine intelligence. Existing collaborative methods are dominantly built upon the de facto human-vision compression pipeline, witnessing deficiency on complexity and bit-rates when aggregating the machine-vision compression. Indeed, machine vision solely focuses on the core regions within the image/video, requiring much less information compared with the compressed information for human vision. In this paper, we thus set out the first successful attempt by a novel collaborative compression method based on the machine-vision-oriented compression, instead of human-vision pipeline. In other words, machine vision serves as the basis for human vision within collaborative compression. A plug-and-play variable bit-rate strategy is also developed for machine vision tasks. Then, we propose to progressively aggregate the semantics from the machine-vision compression, whilst seamlessly tailing the diffusion prior to restore high-fidelity details for human vision, thus named as diffusion-prior based feature compression for human and machine visions (Diff-FCHM). Experimental results verify the consistently superior performances of our Diff-FCHM, on both machine-vision and human-vision compression with remarkable margins. Our code will be released upon acceptance.
>
---
#### [new 018] Ultra-Light Test-Time Adaptation for Vision--Language Models
- **分类: cs.CV**

- **简介: 该论文提出UL-TTA，一种无训练、无反传的视觉-语言模型在线自适应方法，仅调整logit层参数（原型、先验、温度），在域偏移下提升准确率与校准性，显著降低计算开销。**

- **链接: []()**

> **作者:** Byunghyun Kim
>
> **备注:** 7 pages
>
> **摘要:** Vision-Language Models (VLMs) such as CLIP achieve strong zero-shot recognition by comparing image embeddings to text-derived class prototypes. However, under domain shift, they suffer from feature drift, class-prior mismatch, and severe miscalibration. Existing test-time adaptation (TTA) methods often require backpropagation through large backbones, covariance estimation, or heavy memory/state, which is problematic for streaming and edge scenarios. We propose Ultra-Light Test-Time Adaptation (UL-TTA), a fully training-free and backprop-free framework that freezes the backbone and adapts only logit-level parameters: class prototypes, class priors, and temperature. UL-TTA performs an online EM-style procedure with (i) selective sample filtering to use only confident predictions, (ii) closed-form Bayesian updates for prototypes and priors anchored by text and Dirichlet priors, (iii) decoupled temperatures for prediction vs. calibration, and (iv) lightweight guards (norm clipping, prior KL constraints, smoothed temperature) to prevent drift in long streams. Across large-scale cross-domain and OOD benchmarks (PACS, Office-Home, DomainNet, Terra Incognita, ImageNet-R/A/V2/Sketch; ~726K test samples) and strong TTA baselines including Tent, T3A, CoTTA, SAR, Tip-Adapter, and FreeTTA, UL-TTA consistently improves top-1 accuracy (e.g., +4.7 points over zero-shot CLIP on average) while reducing ECE by 20-30%, with less than 8% latency overhead. Long-stream experiments up to 200K samples show no collapse. Our results demonstrate that logit-level Bayesian adaptation is sufficient to obtain state-of-the-art accuracy-calibration trade-offs for VLMs under domain shift, without updating any backbone parameters.
>
---
#### [new 019] DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals for Detection and Binarization
- **分类: cs.CV**

- **简介: 该论文提出DKDS基准数据集，面向古日文草书文档的降质与印章干扰问题，构建了文本与印章检测、文档二值化两大任务，提供基线模型与开源代码，推动鲁棒OCR研究。**

- **链接: []()**

> **作者:** Rui-Yang Ju; Kohei Yamashita; Hirotaka Kameko; Shinsuke Mori
>
> **摘要:** Kuzushiji, a pre-modern Japanese cursive script, can currently be read and understood by only a few thousand trained experts in Japan. With the rapid development of deep learning, researchers have begun applying Optical Character Recognition (OCR) techniques to transcribe Kuzushiji into modern Japanese. Although existing OCR methods perform well on clean pre-modern Japanese documents written in Kuzushiji, they often fail to consider various types of noise, such as document degradation and seals, which significantly affect recognition accuracy. To the best of our knowledge, no existing dataset specifically addresses these challenges. To address this gap, we introduce the Degraded Kuzushiji Documents with Seals (DKDS) dataset as a new benchmark for related tasks. We describe the dataset construction process, which required the assistance of a trained Kuzushiji expert, and define two benchmark tracks: (1) text and seal detection and (2) document binarization. For the text and seal detection track, we provide baseline results using multiple versions of the You Only Look Once (YOLO) models for detecting Kuzushiji characters and seals. For the document binarization track, we present baseline results from traditional binarization algorithms, traditional algorithms combined with K-means clustering, and Generative Adversarial Network (GAN)-based methods. The DKDS dataset and the implementation code for baseline methods are available at https://ruiyangju.github.io/DKDS.
>
---
#### [new 020] Rethinking generative image pretraining: How far are we from scaling up next-pixel prediction?
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究自回归像素预测在图像生成与分类中的扩展规律，发现两者最优缩放策略不同，且计算量而非数据量是主要瓶颈，预测五年内可实现高分辨率像素级建模。**

- **链接: []()**

> **作者:** Xinchen Yan; Chen Liang; Lijun Yu; Adams Wei Yu; Yifeng Lu; Quoc V. Le
>
> **摘要:** This paper investigates the scaling properties of autoregressive next-pixel prediction, a simple, end-to-end yet under-explored framework for unified vision models. Starting with images at resolutions of 32x32, we train a family of Transformers using IsoFlops profiles across compute budgets up to 7e19 FLOPs and evaluate three distinct target metrics: next-pixel prediction objective, ImageNet classification accuracy, and generation quality measured by Fr'echet Distance. First, optimal scaling strategy is critically task-dependent. At a fixed 32x32 resolution alone, the optimal scaling properties for image classification and image generation diverge, where generation optimal setup requires the data size grow three to five times faster than for the classification optimal setup. Second, as image resolution increases, the optimal scaling strategy indicates that the model size must grow much faster than data size. Surprisingly, by projecting our findings, we discover that the primary bottleneck is compute rather than the amount of training data. As compute continues to grow four to five times annually, we forecast the feasibility of pixel-by-pixel modeling of images within the next five years.
>
---
#### [new 021] BronchOpt : Vision-Based Pose Optimization with Fine-Tuned Foundation Models for Accurate Bronchoscopy Navigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出BronchOpt框架，通过微调基础模型实现内窥镜图像与CT的跨域2D-3D配准，解决术中呼吸运动导致的定位偏差问题，并构建首个合成基准数据集，实现无需域适配的高精度导航。**

- **链接: []()**

> **作者:** Hongchao Shu; Roger D. Soberanis-Mukul; Jiru Xu; Hao Ding; Morgan Ringel; Mali Shen; Saif Iftekar Sayed; Hedyeh Rafii-Tari; Mathias Unberath
>
> **摘要:** Accurate intra-operative localization of the bronchoscope tip relative to patient anatomy remains challenging due to respiratory motion, anatomical variability, and CT-to-body divergence that cause deformation and misalignment between intra-operative views and pre-operative CT. Existing vision-based methods often fail to generalize across domains and patients, leading to residual alignment errors. This work establishes a generalizable foundation for bronchoscopy navigation through a robust vision-based framework and a new synthetic benchmark dataset that enables standardized and reproducible evaluation. We propose a vision-based pose optimization framework for frame-wise 2D-3D registration between intra-operative endoscopic views and pre-operative CT anatomy. A fine-tuned modality- and domain-invariant encoder enables direct similarity computation between real endoscopic RGB frames and CT-rendered depth maps, while a differentiable rendering module iteratively refines camera poses through depth consistency. To enhance reproducibility, we introduce the first public synthetic benchmark dataset for bronchoscopy navigation, addressing the lack of paired CT-endoscopy data. Trained exclusively on synthetic data distinct from the benchmark, our model achieves an average translational error of 2.65 mm and a rotational error of 0.19 rad, demonstrating accurate and stable localization. Qualitative results on real patient data further confirm strong cross-domain generalization, achieving consistent frame-wise 2D-3D alignment without domain-specific adaptation. Overall, the proposed framework achieves robust, domain-invariant localization through iterative vision-based optimization, while the new benchmark provides a foundation for standardized progress in vision-based bronchoscopy navigation.
>
---
#### [new 022] RF-DETR: Neural Architecture Search for Real-Time Detection Transformers
- **分类: cs.CV**

- **简介: RF-DETR提出一种轻量级检测Transformer，通过权值共享的神经架构搜索，在不重新训练的情况下快速优化目标数据集的精度-延迟权衡，提升开放词汇目标检测的实时性能与泛化能力。**

- **链接: []()**

> **作者:** Isaac Robinson; Peter Robicheaux; Matvei Popov; Deva Ramanan; Neehar Peri
>
> **备注:** Project Page: https://rfdetr.roboflow.com/
>
> **摘要:** Open-vocabulary detectors achieve impressive performance on COCO, but often fail to generalize to real-world datasets with out-of-distribution classes not typically found in their pre-training. Rather than simply fine-tuning a heavy-weight vision-language model (VLM) for new domains, we introduce RF-DETR, a light-weight specialist detection transformer that discovers accuracy-latency Pareto curves for any target dataset with weight-sharing neural architecture search (NAS). Our approach fine-tunes a pre-trained base network on a target dataset and evaluates thousands of network configurations with different accuracy-latency tradeoffs without re-training. Further, we revisit the "tunable knobs" for NAS to improve the transferability of DETRs to diverse target domains. Notably, RF-DETR significantly improves on prior state-of-the-art real-time methods on COCO and Roboflow100-VL. RF-DETR (nano) achieves 48.0 AP on COCO, beating D-FINE (nano) by 5.3 AP at similar latency, and RF-DETR (2x-large) outperforms GroundingDINO (tiny) by 1.2 AP on Roboflow100-VL while running 20x as fast. To the best of our knowledge, RF-DETR (2x-large) is the first real-time detector to surpass 60 AP on COCO. Our code is at https://github.com/roboflow/rf-detr
>
---
#### [new 023] vMFCoOp: Towards Equilibrium on a Unified Hyperspherical Manifold for Prompting Biomedical VLMs
- **分类: cs.CV**

- **简介: vMFCoOp提出在超球面流形上对齐LLM与CLIP的语义偏差，通过von Mises-Fisher分布建模统一语义锚点，解决生物医学VLM提示学习中的模态鸿沟与泛化不足问题，提升少样本分类性能。**

- **链接: []()**

> **作者:** Minye Shao; Sihan Guo; Xinrun Li; Xingyu Miao; Haoran Duan; Yang Long
>
> **备注:** Accepted as an Oral Presentation at AAAI 2026 Main Technical Track (this version is not peer-reviewed; it is the extended version)
>
> **摘要:** Recent advances in context optimization (CoOp) guided by large language model (LLM)-distilled medical semantic priors offer a scalable alternative to manual prompt engineering and full fine-tuning for adapting biomedical CLIP-based vision-language models (VLMs). However, prompt learning in this context is challenged by semantic misalignment between LLMs and CLIP variants due to divergent training corpora and model architectures; it further lacks scalability across continuously evolving families of foundation models. More critically, pairwise multimodal alignment via conventional Euclidean-space optimization lacks the capacity to model unified representations or apply localized geometric constraints, which tends to amplify modality gaps in complex biomedical imaging and destabilize few-shot adaptation. In this work, we propose vMFCoOp, a framework that inversely estimates von Mises-Fisher (vMF) distributions on a shared Hyperspherical Manifold, aligning semantic biases between arbitrary LLMs and CLIP backbones via Unified Semantic Anchors to achieve robust biomedical prompting and superior few-shot classification. Grounded in three complementary constraints, vMFCoOp demonstrates consistent improvements across 14 medical datasets, 12 medical imaging modalities, and 13 anatomical regions, outperforming state-of-the-art methods in accuracy, generalization, and clinical applicability. This work will be continuously expanded to encompass more downstream applications, and the corresponding resources are intended to be shared through https://github.com/VinyehShaw/UniEqui.
>
---
#### [new 024] Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding
- **分类: cs.CV**

- **简介: 该论文提出一种潜空间匿名化方法，针对视频基础模型在特征提取中泄露隐私信息的问题，设计轻量级适配器AAM，在不微调模型前提下消除敏感属性（如肤色、性别），同时保留任务性能，实现隐私与效用的平衡。**

- **链接: []()**

> **作者:** Joseph Fioresi; Ishan Rajendrakumar Dave; Mubarak Shah
>
> **摘要:** We introduce a novel formulation of visual privacy preservation for video foundation models that operates entirely in the latent space. While spatio-temporal features learned by foundation models have deepened general understanding of video content, sharing or storing these extracted visual features for downstream tasks inadvertently reveals sensitive personal information like skin color, gender, or clothing. Current privacy preservation methods focus on input-pixel-level anonymization, which requires retraining the entire utility video model and results in task-specific anonymization, making them unsuitable for recent video foundational models. To address these challenges, we introduce a lightweight Anonymizing Adapter Module (AAM) that removes private information from video features while retaining general task utility. AAM can be applied in a plug-and-play fashion to frozen video encoders, minimizing the computational burden of finetuning and re-extracting features. Our framework employs three newly designed training objectives: (1) a clip-level self-supervised privacy objective to reduce mutual information between static clips, (2) a co-training objective to retain utility across seen tasks, and (3) a latent consistency loss for generalization on unseen tasks. Our extensive evaluations demonstrate a significant 35% reduction in privacy leakage while maintaining near-baseline utility performance across various downstream tasks: Action Recognition (Kinetics400, UCF101, HMDB51), Temporal Action Detection (THUMOS14), and Anomaly Detection (UCF-Crime). We also provide an analysis on anonymization for sensitive temporal attribute recognition. Additionally, we propose new protocols for assessing gender bias in action recognition models, showing that our method effectively mitigates such biases and promotes more equitable video understanding.
>
---
#### [new 025] FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向图像生成任务，解决分形生成模型（FGM）输出多样性不足的问题，首次引入Hausdorff维度（HD）作为多样性度量，提出可学习HD估计、HD驱动的训练调度与推理采样策略，在保持画质前提下提升多样性39%。**

- **链接: []()**

> **作者:** Haowei Zhang; Yuanpei Zhao; Jizhe Zhou; Mao Li
>
> **备注:** 12 pages, AAAI-26
>
> **摘要:** Improving the diversity of generated results while maintaining high visual quality remains a significant challenge in image generation tasks. Fractal Generative Models (FGMs) are efficient in generating high-quality images, but their inherent self-similarity limits the diversity of output images. To address this issue, we propose a novel approach based on the Hausdorff Dimension (HD), a widely recognized concept in fractal geometry used to quantify structural complexity, which aids in enhancing the diversity of generated outputs. To incorporate HD into FGM, we propose a learnable HD estimation method that predicts HD directly from image embeddings, addressing computational cost concerns. However, simply introducing HD into a hybrid loss is insufficient to enhance diversity in FGMs due to: 1) degradation of image quality, and 2) limited improvement in generation diversity. To this end, during training, we adopt an HD-based loss with a monotonic momentum-driven scheduling strategy to progressively optimize the hyperparameters, obtaining optimal diversity without sacrificing visual quality. Moreover, during inference, we employ HD-guided rejection sampling to select geometrically richer outputs. Extensive experiments on the ImageNet dataset demonstrate that our FGM-HD framework yields a 39\% improvement in output diversity compared to vanilla FGMs, while preserving comparable image quality. To our knowledge, this is the very first work introducing HD into FGM. Our method effectively enhances the diversity of generated outputs while offering a principled theoretical contribution to FGM development.
>
---
#### [new 026] CADIC: Continual Anomaly Detection Based on Incremental Coreset
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CADIC框架，用于持续异常检测任务，解决传统方法因任务专属内存导致的灵活性差问题。通过共享统一核心集增量更新嵌入，实现高效知识累积与高精度检测，在多个数据集上表现最优。**

- **链接: []()**

> **作者:** Gen Yang; Zhipeng Deng; Junfeng Man
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** The primary objective of Continual Anomaly Detection (CAD) is to learn the normal patterns of new tasks under dynamic data distribution assumptions while mitigating catastrophic forgetting. Existing embedding-based CAD approaches continuously update a memory bank with new embeddings to adapt to sequential tasks. However, these methods require constructing class-specific sub-memory banks for each task, which restricts their flexibility and scalability. To address this limitation, we propose a novel CAD framework where all tasks share a unified memory bank. During training, the method incrementally updates embeddings within a fixed-size coreset, enabling continuous knowledge acquisition from sequential tasks without task-specific memory fragmentation. In the inference phase, anomaly scores are computed via a nearest-neighbor matching mechanism, achieving state-of-the-art detection accuracy. We validate the method through comprehensive experiments on MVTec AD and Visa datasets. Results show that our approach outperforms existing baselines, achieving average image-level AUROC scores of 0.972 (MVTec AD) and 0.891 (Visa). Notably, on a real-world electronic paper dataset, it demonstrates 100% accuracy in anomaly sample detection, confirming its robustness in practical scenarios. The implementation will be open-sourced on GitHub.
>
---
#### [new 027] Harnessing Diffusion-Generated Synthetic Images for Fair Image Classification
- **分类: cs.CV**

- **简介: 该论文针对图像分类中的偏差问题，提出用微调扩散模型（如LoRA、DreamBooth+聚类）生成更精准的平衡合成数据，提升公平性。方法在高偏差场景下优于主流去偏技术。**

- **链接: []()**

> **作者:** Abhipsa Basu; Aviral Gupta; Abhijnya Bhat; R. Venkatesh Babu
>
> **摘要:** Image classification systems often inherit biases from uneven group representation in training data. For example, in face datasets for hair color classification, blond hair may be disproportionately associated with females, reinforcing stereotypes. A recent approach leverages the Stable Diffusion model to generate balanced training data, but these models often struggle to preserve the original data distribution. In this work, we explore multiple diffusion-finetuning techniques, e.g., LoRA and DreamBooth, to generate images that more accurately represent each training group by learning directly from their samples. Additionally, in order to prevent a single DreamBooth model from being overwhelmed by excessive intra-group variations, we explore a technique of clustering images within each group and train a DreamBooth model per cluster. These models are then used to generate group-balanced data for pretraining, followed by fine-tuning on real data. Experiments on multiple benchmarks demonstrate that the studied finetuning approaches outperform vanilla Stable Diffusion on average and achieve results comparable to SOTA debiasing techniques like Group-DRO, while surpassing them as the dataset bias severity increases.
>
---
#### [new 028] HitoMi-Cam: A Shape-Agnostic Person Detection Method Using the Spectral Characteristics of Clothing
- **分类: cs.CV**

- **简介: 论文提出HitoMi-Cam，一种基于衣物光谱反射特性的形状无关人体检测方法，解决CNN在非训练姿态下性能下降问题，在无GPU边缘设备上实现23.2 fps实时检测，AP达93.5%，显著优于CNN，适用于灾害救援等复杂场景。**

- **链接: []()**

> **作者:** Shuji Ono
>
> **备注:** 37 pages, 21 figures, 9 tables. Published in MDPI Journal of Imaging. Includes 1 supplementary video file (ancillary file)
>
> **摘要:** While convolutional neural network (CNN)-based object detection is widely used, it exhibits a shape dependency that degrades performance for postures not included in the training data. Building upon our previous simulation study published in this journal, this study implements and evaluates the spectral-based approach on physical hardware to address this limitation. Specifically, this paper introduces HitoMi-Cam, a lightweight and shape-agnostic person detection method that uses the spectral reflectance properties of clothing. The author implemented the system on a resource-constrained edge device without a GPU to assess its practical viability. The results indicate that a processing speed of 23.2 frames per second (fps) (253x190 pixels) is achievable, suggesting that the method can be used for real-time applications. In a simulated search and rescue scenario where the performance of CNNs declines, HitoMi-Cam achieved an average precision (AP) of 93.5%, surpassing that of the compared CNN models (best AP of 53.8%). Throughout all evaluation scenarios, the occurrence of false positives remained minimal. This study positions the HitoMi-Cam method not as a replacement for CNN-based detectors but as a complementary tool under specific conditions. The results indicate that spectral-based person detection can be a viable option for real-time operation on edge devices in real-world environments where shapes are unpredictable, such as disaster rescue.
>
---
#### [new 029] 4KDehazeFlow: Ultra-High-Definition Image Dehazing via Flow Matching
- **分类: cs.CV**

- **简介: 4KDehazeFlow提出一种基于流匹配的超高清去雾方法，通过可学习3D LUT与RK4 ODE求解器实现高效、低失真去雾，兼容任意网络，显著提升密集雾霾场景下的图像质量与色彩保真度。**

- **链接: []()**

> **作者:** Xingchi Chen; Pu Wang; Xuerui Li; Chaopeng Li; Juxiang Zhou; Jianhou Gan; Dianjie Lu; Guijuan Zhang; Wenqi Ren; Zhuoran Zheng
>
> **摘要:** Ultra-High-Definition (UHD) image dehazing faces challenges such as limited scene adaptability in prior-based methods and high computational complexity with color distortion in deep learning approaches. To address these issues, we propose 4KDehazeFlow, a novel method based on Flow Matching and the Haze-Aware vector field. This method models the dehazing process as a progressive optimization of continuous vector field flow, providing efficient data-driven adaptive nonlinear color transformation for high-quality dehazing. Specifically, our method has the following advantages: 1) 4KDehazeFlow is a general method compatible with various deep learning networks, without relying on any specific network architecture. 2) We propose a learnable 3D lookup table (LUT) that encodes haze transformation parameters into a compact 3D mapping matrix, enabling efficient inference through precomputed mappings. 3) We utilize a fourth-order Runge-Kutta (RK4) ordinary differential equation (ODE) solver to stably solve the dehazing flow field through an accurate step-by-step iterative method, effectively suppressing artifacts. Extensive experiments show that 4KDehazeFlow exceeds seven state-of-the-art methods. It delivers a 2dB PSNR increase and better performance in dense haze and color fidelity.
>
---
#### [new 030] DreamPose3D: Hallucinative Diffusion with Prompt Learning for 3D Human Pose Estimation
- **分类: cs.CV; cs.AI**

- **简介: DreamPose3D提出一种基于扩散模型的3D人体姿态估计方法，通过动作提示学习与时空幻觉解码，提升模糊场景下的时序一致性与关节关系建模，显著优于现有方法。**

- **链接: []()**

> **作者:** Jerrin Bright; Yuhao Chen; John S. Zelek
>
> **摘要:** Accurate 3D human pose estimation remains a critical yet unresolved challenge, requiring both temporal coherence across frames and fine-grained modeling of joint relationships. However, most existing methods rely solely on geometric cues and predict each 3D pose independently, which limits their ability to resolve ambiguous motions and generalize to real-world scenarios. Inspired by how humans understand and anticipate motion, we introduce DreamPose3D, a diffusion-based framework that combines action-aware reasoning with temporal imagination for 3D pose estimation. DreamPose3D dynamically conditions the denoising process using task-relevant action prompts extracted from 2D pose sequences, capturing high-level intent. To model the structural relationships between joints effectively, we introduce a representation encoder that incorporates kinematic joint affinity into the attention mechanism. Finally, a hallucinative pose decoder predicts temporally coherent 3D pose sequences during training, simulating how humans mentally reconstruct motion trajectories to resolve ambiguity in perception. Extensive experiments on benchmarked Human3.6M and MPI-3DHP datasets demonstrate state-of-the-art performance across all metrics. To further validate DreamPose3D's robustness, we tested it on a broadcast baseball dataset, where it demonstrated strong performance despite ambiguous and noisy 2D inputs, effectively handling temporal consistency and intent-driven motion variations.
>
---
#### [new 031] Enhancing Rotation-Invariant 3D Learning with Global Pose Awareness and Attention Mechanisms
- **分类: cs.CV**

- **简介: 该论文面向3D点云旋转不变学习，解决因局部特征丢失全局姿态信息而导致的对称结构混淆问题。提出SiPF与RIAttnConv，引入全局“影子”参考点和注意力机制，提升旋转下细粒度空间判别能力。**

- **链接: []()**

> **作者:** Jiaxun Guo; Manar Amayri; Nizar Bouguila; Xin Liu; Wentao Fan
>
> **备注:** 14 pages, 6 gigures,AAAI 2026
>
> **摘要:** Recent advances in rotation-invariant (RI) learning for 3D point clouds typically replace raw coordinates with handcrafted RI features to ensure robustness under arbitrary rotations. However, these approaches often suffer from the loss of global pose information, making them incapable of distinguishing geometrically similar but spatially distinct structures. We identify that this limitation stems from the restricted receptive field in existing RI methods, leading to Wing-tip feature collapse, a failure to differentiate symmetric components (e.g., left and right airplane wings) due to indistinguishable local geometries. To overcome this challenge, we introduce the Shadow-informed Pose Feature (SiPF), which augments local RI descriptors with a globally consistent reference point (referred to as the 'shadow') derived from a learned shared rotation. This mechanism enables the model to preserve global pose awareness while maintaining rotation invariance. We further propose Rotation-invariant Attention Convolution (RIAttnConv), an attention-based operator that integrates SiPFs into the feature aggregation process, thereby enhancing the model's capacity to distinguish structurally similar components. Additionally, we design a task-adaptive shadow locating module based on the Bingham distribution over unit quaternions, which dynamically learns the optimal global rotation for constructing consistent shadows. Extensive experiments on 3D classification and part segmentation benchmarks demonstrate that our approach substantially outperforms existing RI methods, particularly in tasks requiring fine-grained spatial discrimination under arbitrary rotations.
>
---
#### [new 032] Case Study: Transformer-Based Solution for the Automatic Digitization of Gas Plants
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文面向燃气电厂数字化，提出基于Transformer的AI方案，自动从P&ID图纸中提取设计数据与拓扑结构，融合OCR、视觉大模型与关系推理，实现91%文本与93%组件识别准确率。**

- **链接: []()**

> **作者:** I. Bailo; F. Buonora; G. Ciarfaglia; L. T. Consoli; A. Evangelista; M. Gabusi; M. Ghiani; C. Petracca Ciavarella; F. Picariello; F. Sarcina; F. Tuosto; V. Zullo; L. Airoldi; G. Bruno; D. D. Gobbo; S. Pezzenati; G. A. Tona
>
> **摘要:** The energy transition is a key theme of the last decades to determine a future of eco-sustainability, and an area of such importance cannot disregard digitization, innovation and the new technological tools available. This is the context in which the Generative Artificial Intelligence models described in this paper are positioned, developed by Engineering Ingegneria Informatica SpA in order to automate the plant structures acquisition of SNAM energy infrastructure, a leading gas transportation company in Italy and Europe. The digitization of a gas plant consists in registering all its relevant information through the interpretation of the related documentation. The aim of this work is therefore to design an effective solution based on Artificial Intelligence techniques to automate the extraction of the information necessary for the digitization of a plant, in order to streamline the daily work of MGM users. The solution received the P&ID of the plant as input, each one in pdf format, and uses OCR, Vision LLM, Object Detection, Relational Reasoning and optimization algorithms to return an output consisting of two sets of information: a structured overview of the relevant design data and the hierarchical framework of the plant. To achieve convincing results, we extend a state-of-the-art model for Scene Graph Generation introducing a brand new Transformer architecture with the aim of deepening the analysis of the complex relations between the plant's components. The synergistic use of the listed AI-based technologies allowed to overcome many obstacles arising from the high variety of data, due to the lack of standardization. An accuracy of 91\% has been achieved in the extraction of textual information relating to design data. Regarding the plants topology, 93\% of components are correctly identified and the hierarchical structure is extracted with an accuracy around 80\%.
>
---
#### [new 033] Revisiting Cross-Architecture Distillation: Adaptive Dual-Teacher Transfer for Lightweight Video Models
- **分类: cs.CV**

- **简介: 该论文针对视频动作识别中ViT与轻量CNN间的性能差距，提出双教师自适应知识蒸馏框架，通过动态加权与残差特征学习，高效迁移ViT知识至CNN，显著提升轻量模型精度。**

- **链接: []()**

> **作者:** Ying Peng; Hongsen Ye; Changxin Huang; Xiping Hu; Jian Chen; Runhao Zeng
>
> **备注:** 2 figures, 7 tables
>
> **摘要:** Vision Transformers (ViTs) have achieved strong performance in video action recognition, but their high computational cost limits their practicality. Lightweight CNNs are more efficient but suffer from accuracy gaps. Cross-Architecture Knowledge Distillation (CAKD) addresses this by transferring knowledge from ViTs to CNNs, yet existing methods often struggle with architectural mismatch and overlook the value of stronger homogeneous CNN teachers. To tackle these challenges, we propose a Dual-Teacher Knowledge Distillation framework that leverages both a heterogeneous ViT teacher and a homogeneous CNN teacher to collaboratively guide a lightweight CNN student. We introduce two key components: (1) Discrepancy-Aware Teacher Weighting, which dynamically fuses the predictions from ViT and CNN teachers by assigning adaptive weights based on teacher confidence and prediction discrepancy with the student, enabling more informative and effective supervision; and (2) a Structure Discrepancy-Aware Distillation strategy, where the student learns the residual features between ViT and CNN teachers via a lightweight auxiliary branch, focusing on transferable architectural differences without mimicking all of ViT's high-dimensional patterns. Extensive experiments on benchmarks including HMDB51, EPIC-KITCHENS-100, and Kinetics-400 demonstrate that our method consistently outperforms state-of-the-art distillation approaches, achieving notable performance improvements with a maximum accuracy gain of 5.95% on HMDB51.
>
---
#### [new 034] From Structure to Detail: Hierarchical Distillation for Efficient Diffusion Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出分层蒸馏（HD）框架，解决扩散模型单步推理中结构与细节难以兼顾的问题。通过轨迹蒸馏构建结构骨架，再用自适应加权判别器（AWD）精细优化细节，实现单步生成媲美多步模型的高保真效果。**

- **链接: []()**

> **作者:** Hanbo Cheng; Peng Wang; Kaixiang Lei; Qi Li; Zhen Zou; Pengfei Hu; Jun Du
>
> **摘要:** The inference latency of diffusion models remains a critical barrier to their real-time application. While trajectory-based and distribution-based step distillation methods offer solutions, they present a fundamental trade-off. Trajectory-based methods preserve global structure but act as a "lossy compressor", sacrificing high-frequency details. Conversely, distribution-based methods can achieve higher fidelity but often suffer from mode collapse and unstable training. This paper recasts them from independent paradigms into synergistic components within our novel Hierarchical Distillation (HD) framework. We leverage trajectory distillation not as a final generator, but to establish a structural ``sketch", providing a near-optimal initialization for the subsequent distribution-based refinement stage. This strategy yields an ideal initial distribution that enhances the ceiling of overall performance. To further improve quality, we introduce and refine the adversarial training process. We find standard discriminator structures are ineffective at refining an already high-quality generator. To overcome this, we introduce the Adaptive Weighted Discriminator (AWD), tailored for the HD pipeline. By dynamically allocating token weights, AWD focuses on local imperfections, enabling efficient detail refinement. Our approach demonstrates state-of-the-art performance across diverse tasks. On ImageNet $256\times256$, our single-step model achieves an FID of 2.26, rivaling its 250-step teacher. It also achieves promising results on the high-resolution text-to-image MJHQ benchmark, proving its generalizability. Our method establishes a robust new paradigm for high-fidelity, single-step diffusion models.
>
---
#### [new 035] Improve Contrastive Clustering Performance by Multiple Fusing-Augmenting ViT Blocks
- **分类: cs.CV**

- **简介: 该论文面向图像聚类任务，提出多融合增强ViT模块（MFAVBs），通过显式融合正样本对特征并迭代增强，提升对比聚类性能，结合CLIP预训练特征预处理，在七大数据集上超越SOTA方法。**

- **链接: []()**

> **作者:** Cheng Wang; Shuisheng Zhou; Fengjiao Peng; Jin Sheng; Feng Ye; Yinli Dong
>
> **摘要:** In the field of image clustering, the widely used contrastive learning networks improve clustering performance by maximizing the similarity between positive pairs and the dissimilarity of negative pairs of the inputs. Extant contrastive learning networks, whose two encoders often implicitly interact with each other by parameter sharing or momentum updating, may not fully exploit the complementarity and similarity of the positive pairs to extract clustering features from input data. To explicitly fuse the learned features of positive pairs, we design a novel multiple fusing-augmenting ViT blocks (MFAVBs) based on the excellent feature learning ability of Vision Transformers (ViT). Firstly, two preprocessed augmentions as positive pairs are separately fed into two shared-weight ViTs, then their output features are fused to input into a larger ViT. Secondly, the learned features are split into a pair of new augmented positive samples and passed to the next FAVBs, enabling multiple fusion and augmention through MFAVBs operations. Finally, the learned features are projected into both instance-level and clustering-level spaces to calculate the cross-entropy loss, followed by parameter updates by backpropagation to finalize the training process. To further enhance ability of the model to distinguish between similar images, our input data for the network we propose is preprocessed augmentions with features extracted from the CLIP pretrained model. Our experiments on seven public datasets demonstrate that MFAVBs serving as the backbone for contrastive clustering outperforms the state-of-the-art techniques in terms of clustering performance.
>
---
#### [new 036] Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对大视觉语言模型（LVLMs）中的对象幻觉问题，提出Owl框架，通过因果图建模视觉与文本注意力交互，引入VTACR指标识别模态失衡，并设计双路径注意力干预与对比解码，显著降低幻觉并保持理解能力。**

- **链接: []()**

> **作者:** Liu Yu; Zhonghao Chen; Ping Kuang; Zhikun Feng; Fan Zhou; Lan Wang; Gillian Dobbie
>
> **备注:** 9 pages, published to AAAI 2026
>
> **摘要:** Object hallucination remains a critical challenge in Large Vision-Language Models (LVLMs), where models generate content inconsistent with visual inputs. Existing language-decoder based mitigation approaches often regulate visual or textual attention independently, overlooking their interaction as two key causal factors. To address this, we propose Owl (Bi-mOdal attention reWeighting for Layer-wise hallucination mitigation), a causally-grounded framework that models hallucination process via a structural causal graph, treating decomposed visual and textual attentions as mediators. We introduce VTACR (Visual-to-Textual Attention Contribution Ratio), a novel metric that quantifies the modality contribution imbalance during decoding. Our analysis reveals that hallucinations frequently occur in low-VTACR scenarios, where textual priors dominate and visual grounding is weakened. To mitigate this, we design a fine-grained attention intervention mechanism that dynamically adjusts token- and layer-wise attention guided by VTACR signals. Finally, we propose a dual-path contrastive decoding strategy: one path emphasizes visually grounded predictions, while the other amplifies hallucinated ones -- letting visual truth shine and hallucination collapse. Experimental results on the POPE and CHAIR benchmarks show that Owl achieves significant hallucination reduction, setting a new SOTA in faithfulness while preserving vision-language understanding capability. Our code is available at https://github.com/CikZ2023/OWL
>
---
#### [new 037] Asymmetric Cross-Modal Knowledge Distillation: Bridging Modalities with Weak Semantic Consistency
- **分类: cs.CV**

- **简介: 该论文提出Asymmetric Cross-Modal Knowledge Distillation（ACKD），解决弱语义一致性下跨模态知识迁移难题，构建SemBridge框架，通过动态匹配与最优传输实现高效知识传递，并在遥感图像分类任务中取得SOTA效果。**

- **链接: []()**

> **作者:** Riling Wei; Kelu Yao; Chuanguang Yang; Jin Wang; Zhuoyan Gao; Chao Li
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** Cross-modal Knowledge Distillation has demonstrated promising performance on paired modalities with strong semantic connections, referred to as Symmetric Cross-modal Knowledge Distillation (SCKD). However, implementing SCKD becomes exceedingly constrained in real-world scenarios due to the limited availability of paired modalities. To this end, we investigate a general and effective knowledge learning concept under weak semantic consistency, dubbed Asymmetric Cross-modal Knowledge Distillation (ACKD), aiming to bridge modalities with limited semantic overlap. Nevertheless, the shift from strong to weak semantic consistency improves flexibility but exacerbates challenges in knowledge transmission costs, which we rigorously verified based on optimal transport theory. To mitigate the issue, we further propose a framework, namely SemBridge, integrating a Student-Friendly Matching module and a Semantic-aware Knowledge Alignment module. The former leverages self-supervised learning to acquire semantic-based knowledge and provide personalized instruction for each student sample by dynamically selecting the relevant teacher samples. The latter seeks the optimal transport path by employing Lagrangian optimization. To facilitate the research, we curate a benchmark dataset derived from two modalities, namely Multi-Spectral (MS) and asymmetric RGB images, tailored for remote sensing scene classification. Comprehensive experiments exhibit that our framework achieves state-of-the-art performance compared with 7 existing approaches on 6 different model architectures across various datasets.
>
---
#### [new 038] SPEED-Q: Staged Processing with Enhanced Distillation towards Efficient Low-bit On-device VLM Quantization
- **分类: cs.CV**

- **简介: 该论文提出SPEED-Q，首次实现小规模亿级VLM的低比特权重量化，解决视觉与语言模块敏感度差异及训练不稳问题，通过分阶段自适应与增强蒸馏，在2-bit下精度提升6倍，实现高效边缘部署。**

- **链接: []()**

> **作者:** Tianyu Guo; Shanwei Zhao; Shiai Zhu; Chenguang Ma
>
> **摘要:** Deploying Vision-Language Models (VLMs) on edge devices (e.g., smartphones and robots) is crucial for enabling low-latency and privacy-preserving intelligent applications. Given the resource constraints of these devices, quantization offers a promising solution by improving memory efficiency and reducing bandwidth requirements, thereby facilitating the deployment of VLMs. However, existing research has rarely explored aggressive quantization on VLMs, particularly for the models ranging from 1B to 2B parameters, which are more suitable for resource-constrained edge devices. In this paper, we propose SPEED-Q, a novel Staged Processing with Enhanced Distillation framework for VLM low-bit weight-only quantization that systematically addresses the following two critical obstacles: (1) significant discrepancies in quantization sensitivity between vision (ViT) and language (LLM) components in VLMs; (2) training instability arising from the reduced numerical precision inherent in low-bit quantization. In SPEED-Q, a staged sensitivity adaptive mechanism is introduced to effectively harmonize performance across different modalities. We further propose a distillation-enhanced quantization strategy to stabilize the training process and reduce data dependence. Together, SPEED-Q enables accurate, stable, and data-efficient quantization of complex VLMs. SPEED-Q is the first framework tailored for quantizing entire small-scale billion-parameter VLMs to low bits. Extensive experiments across multiple benchmarks demonstrate that SPEED-Q achieves up to 6x higher accuracy than existing quantization methods under 2-bit settings and consistently outperforms prior on-device VLMs under both 2-bit and 4-bit settings. Our code and models are available at https://github.com/antgroup/SPEED-Q.
>
---
#### [new 039] Enriching Knowledge Distillation with Cross-Modal Teacher Fusion
- **分类: cs.CV**

- **简介: 该论文属于知识蒸馏任务，旨在解决单模态教师模型知识多样性不足的问题。提出RichKD框架，融合CLIP的视觉-语言多模态信息，增强学生模型的监督信号，提升蒸馏质量与鲁棒性。**

- **链接: []()**

> **作者:** Amir M. Mansourian; Amir Mohammad Babaei; Shohreh Kasaei
>
> **备注:** 11 pages, 5 figures, 8 tables
>
> **摘要:** Multi-teacher knowledge distillation (KD), a more effective technique than traditional single-teacher methods, transfers knowledge from expert teachers to a compact student model using logit or feature matching. However, most existing approaches lack knowledge diversity, as they rely solely on unimodal visual information, overlooking the potential of cross-modal representations. In this work, we explore the use of CLIP's vision-language knowledge as a complementary source of supervision for KD, an area that remains largely underexplored. We propose a simple yet effective framework that fuses the logits and features of a conventional teacher with those from CLIP. By incorporating CLIP's multi-prompt textual guidance, the fused supervision captures both dataset-specific and semantically enriched visual cues. Beyond accuracy, analysis shows that the fused teacher yields more confident and reliable predictions, significantly increasing confident-correct cases while reducing confidently wrong ones. Moreover, fusion with CLIP refines the entire logit distribution, producing semantically meaningful probabilities for non-target classes, thereby improving inter-class consistency and distillation quality. Despite its simplicity, the proposed method, Enriching Knowledge Distillation (RichKD), consistently outperforms most existing baselines across multiple benchmarks and exhibits stronger robustness under distribution shifts and input corruptions.
>
---
#### [new 040] DT-NVS: Diffusion Transformers for Novel View Synthesis
- **分类: cs.CV; cs.AI**

- **简介: DT-NVS提出一种基于Transformer的3D扩散模型，实现从单张真实场景图像生成多样新视角，突破传统方法对小幅度运动或人工场景的限制，首次在非对齐真实视频数据上训练，提升泛化能力。**

- **链接: []()**

> **作者:** Wonbong Jang; Jonathan Tremblay; Lourdes Agapito
>
> **备注:** 14 pages
>
> **摘要:** Generating novel views of a natural scene, e.g., every-day scenes both indoors and outdoors, from a single view is an under-explored problem, even though it is an organic extension to the object-centric novel view synthesis. Existing diffusion-based approaches focus rather on small camera movements in real scenes or only consider unnatural object-centric scenes, limiting their potential applications in real-world settings. In this paper we move away from these constrained regimes and propose a 3D diffusion model trained with image-only losses on a large-scale dataset of real-world, multi-category, unaligned, and casually acquired videos of everyday scenes. We propose DT-NVS, a 3D-aware diffusion model for generalized novel view synthesis that exploits a transformer-based architecture backbone. We make significant contributions to transformer and self-attention architectures to translate images to 3d representations, and novel camera conditioning strategies to allow training on real-world unaligned datasets. In addition, we introduce a novel training paradigm swapping the role of reference frame between the conditioning image and the sampled noisy input. We evaluate our approach on the 3D task of generalized novel view synthesis from a single input image and show improvements over state-of-the-art 3D aware diffusion models and deterministic approaches, while generating diverse outputs.
>
---
#### [new 041] Adaptive graph Kolmogorov-Arnold network for 3D human pose estimation
- **分类: cs.CV**

- **简介: 该论文提出PoseKAN，一种自适应图Kolmogorov-Arnold网络，用于单图像2D到3D人体姿态估计。通过可学习边函数与多跳聚合，突破GCN的局部感知与频谱偏差限制，提升对遮挡和深度模糊的建模能力。**

- **链接: []()**

> **作者:** Abu Taib Mohammed Shahjahan; A. Ben Hamza
>
> **摘要:** Graph convolutional network (GCN)-based methods have shown strong performance in 3D human pose estimation by leveraging the natural graph structure of the human skeleton. However, their local receptive field limits their ability to capture long-range dependencies essential for handling occlusions and depth ambiguities. They also exhibit spectral bias, which prioritizes low-frequency components while struggling to model high-frequency details. In this paper, we introduce PoseKAN, an adaptive graph Kolmogorov-Arnold Network (KAN), framework that extends KANs to graph-based learning for 2D-to-3D pose lifting from a single image. Unlike GCNs that use fixed activation functions, KANs employ learnable functions on graph edges, allowing data-driven, adaptive feature transformations. This enhances the model's adaptability and expressiveness, making it more expressive in learning complex pose variations. Our model employs multi-hop feature aggregation, ensuring the body joints can leverage information from both local and distant neighbors, leading to improved spatial awareness. It also incorporates residual PoseKAN blocks for deeper feature refinement, and a global response normalization for improved feature selectivity and contrast. Extensive experiments on benchmark datasets demonstrate the competitive performance of our model against state-of-the-art methods.
>
---
#### [new 042] GRACE: Designing Generative Face Video Codec via Agile Hardware-Centric Workflow
- **分类: cs.CV**

- **简介: 该论文面向边缘计算，提出一种基于FPGA的生成式人脸视频编码器（AGC）部署方案，解决其高功耗与部署困难问题，通过量化、融合与软硬件协同设计，实现能效提升24.9倍，单像素重构仅需11.7μJ。**

- **链接: []()**

> **作者:** Rui Wan; Qi Zheng; Ruoyu Zhang; Bu Chen; Jiaming Liu; Min Li; Minge Jing; Jinjia Zhou; Yibo Fan
>
> **摘要:** The Animation-based Generative Codec (AGC) is an emerging paradigm for talking-face video compression. However, deploying its intricate decoder on resource and power-constrained edge devices presents challenges due to numerous parameters, the inflexibility to adapt to dynamically evolving algorithms, and the high power consumption induced by extensive computations and data transmission. This paper for the first time proposes a novel field programmable gate arrays (FPGAs)-oriented AGC deployment scheme for edge-computing video services. Initially, we analyze the AGC algorithm and employ network compression methods including post-training static quantization and layer fusion techniques. Subsequently, we design an overlapped accelerator utilizing the co-processor paradigm to perform computations through software-hardware co-design. The hardware processing unit comprises engines such as convolution, grid sampling, upsample, etc. Parallelization optimization strategies like double-buffered pipelines and loop unrolling are employed to fully exploit the resources of FPGA. Ultimately, we establish an AGC FPGA prototype on the PYNQ-Z1 platform using the proposed scheme, achieving \textbf{24.9$\times$} and \textbf{4.1$\times$} higher energy efficiency against commercial Central Processing Unit (CPU) and Graphic Processing Unit (GPU), respectively. Specifically, only \textbf{11.7} microjoules ($\upmu$J) are required for one pixel reconstructed by this FPGA system.
>
---
#### [new 043] VietMEAgent: Culturally-Aware Few-Shot Multimodal Explanation for Vietnamese Visual Question Answering
- **分类: cs.CV**

- **简介: 论文提出VietMEAgent，面向越南语视觉问答的少样本多模态可解释框架，解决文化知识缺失与解释性不足问题，通过文化知识库与程序生成模块，实现融合视觉证据与人文理据的透明推理，支持文化敏感的AI应用。**

- **链接: []()**

> **作者:** Hai-Dang Nguyen; Minh-Anh Dang; Minh-Tan Le; Minh-Tuan Le
>
> **备注:** 7 pages, 3 figures, 3 tables, FAIR 2025 conference
>
> **摘要:** Contemporary Visual Question Answering (VQA) systems remain constrained when confronted with culturally specific content, largely because cultural knowledge is under-represented in training corpora and the reasoning process is not rendered interpretable to end users. This paper introduces VietMEAgent, a multimodal explainable framework engineered for Vietnamese cultural understanding. The method integrates a cultural object detection backbone with a structured program generation layer, yielding a pipeline in which answer prediction and explanation are tightly coupled. A curated knowledge base of Vietnamese cultural entities serves as an explicit source of background information, while a dual-modality explanation module combines attention-based visual evidence with structured, human-readable textual rationales. We further construct a Vietnamese Cultural VQA dataset sourced from public repositories and use it to demonstrate the practicality of programming-based methodologies for cultural AI. The resulting system provides transparent explanations that disclose both the computational rationale and the underlying cultural context, supporting education and cultural preservation with an emphasis on interpretability and cultural sensitivity.
>
---
#### [new 044] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition
- **分类: cs.CV**

- **简介: 该论文面向零样本骨架动作识别任务，解决现有方法对齐脆弱、分类僵化问题，提出Flora模型，通过邻域感知语义调准与开放流分类器，实现鲁棒的跨模态对齐与细粒度决策。**

- **链接: []()**

> **作者:** Yang Chen; Miaoge Li; Zhijie Rao; Deze Zeng; Song Guo; Jingcai Guo
>
> **备注:** Code is available at https://github.com/cseeyangchen/Flora
>
> **摘要:** Recognizing unseen skeleton action categories remains highly challenging due to the absence of corresponding skeletal priors. Existing approaches generally follow an "align-then-classify" paradigm but face two fundamental issues, i.e., (i) fragile point-to-point alignment arising from imperfect semantics, and (ii) rigid classifiers restricted by static decision boundaries and coarse-grained anchors. To address these issues, we propose a novel method for zero-shot skeleton action recognition, termed $\texttt{$\textbf{Flora}$}$, which builds upon $\textbf{F}$lexib$\textbf{L}$e neighb$\textbf{O}$r-aware semantic attunement and open-form dist$\textbf{R}$ibution-aware flow cl$\textbf{A}$ssifier. Specifically, we flexibly attune textual semantics by incorporating neighboring inter-class contextual cues to form direction-aware regional semantics, coupled with a cross-modal geometric consistency objective that ensures stable and robust point-to-region alignment. Furthermore, we employ noise-free flow matching to bridge the modality distribution gap between semantic and skeleton latent embeddings, while a condition-free contrastive regularization enhances discriminability, leading to a distribution-aware classifier with fine-grained decision boundaries achieved through token-level velocity predictions. Extensive experiments on three benchmark datasets validate the effectiveness of our method, showing particularly impressive performance even when trained with only 10\% of the seen data. Code is available at https://github.com/cseeyangchen/Flora.
>
---
#### [new 045] Composition-Incremental Learning for Compositional Generalization
- **分类: cs.CV**

- **简介: 该论文提出Composition-Incremental Learning（CompIL）任务，解决模型在持续学习新组合时保持 compositional generalization 能力的问题，构建了两个基准，并提出基于视觉合成与语言原语蒸馏的伪回放框架以提升持续学习效果。**

- **链接: []()**

> **作者:** Zhen Li; Yuwei Wu; Chenchen Jing; Che Sun; Chuanhao Li; Yunde Jia
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Compositional generalization has achieved substantial progress in computer vision on pre-collected training data. Nonetheless, real-world data continually emerges, with possible compositions being nearly infinite, long-tailed, and not entirely visible. Thus, an ideal model is supposed to gradually improve the capability of compositional generalization in an incremental manner. In this paper, we explore Composition-Incremental Learning for Compositional Generalization (CompIL) in the context of the compositional zero-shot learning (CZSL) task, where models need to continually learn new compositions, intending to improve their compositional generalization capability progressively. To quantitatively evaluate CompIL, we develop a benchmark construction pipeline leveraging existing datasets, yielding MIT-States-CompIL and C-GQA-CompIL. Furthermore, we propose a pseudo-replay framework utilizing a visual synthesizer to synthesize visual representations of learned compositions and a linguistic primitive distillation mechanism to maintain aligned primitive representations across the learning process. Extensive experiments demonstrate the effectiveness of the proposed framework.
>
---
#### [new 046] Classifying Histopathologic Glioblastoma Sub-regions with EfficientNet
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对胶质母细胞瘤（GBM）组织切片的六类亚区分类任务，采用EfficientNet架构实现自动化病理识别，旨在提升病理分析的效率与一致性。在BraTS-Path 2024数据集上验证，EfficientNet-B1/B4表现最优，但模型泛化能力仍待提升。**

- **链接: []()**

> **作者:** Sanyukta Adap; Ujjwal Baid; Spyridon Bakas
>
> **摘要:** Glioblastoma (GBM) is the most common aggressive, fast-growing brain tumor, with a grim prognosis. Despite clinical diagnostic advancements, there have not been any substantial improvements to patient prognosis. Histopathological assessment of excised tumors is the first line of clinical diagnostic routine. We hypothesize that automated, robust, and accurate identification of distinct histological sub-regions within GBM could contribute to morphologically understanding this disease at scale. In this study, we designed a four-step deep learning approach to classify six (6) histopathological regions and quantitatively evaluated it on the BraTS-Path 2024 challenge dataset, which includes digitized Hematoxylin \& Eosin (H\&E) stained GBM tissue sections annotated for six distinct regions. We used the challenge's publicly available training dataset to develop and evaluate the effectiveness of several variants of EfficientNet architectures (i.e., B0, B1, B2, B3, B4). EfficientNet-B1 and EfficientNet-B4 achieved the best performance, achieving an F1 score of 0.98 in a 5-fold cross-validation configuration using the BraTS-Path training set. The quantitative performance evaluation of our proposed approach with EfficientNet-B1 on the BraTS-Path hold-out validation data and the final hidden testing data yielded F1 scores of 0.546 and 0.517, respectively, for the associated 6-class classification task. The difference in the performance on training, validation, and testing data highlights the challenge of developing models that generalize well to new data, which is crucial for clinical applications. The source code of the proposed approach can be found at the GitHub repository of Indiana University Division of Computational Pathology: https://github.com/IUCompPath/brats-path-2024-enet.
>
---
#### [new 047] WiCV at CVPR 2025: The Women in Computer Vision Workshop
- **分类: cs.CV**

- **简介: 本文是WiCV@CVPR 2025 workshop的总结报告，旨在记录并评估其在促进女性与少数群体在计算机视觉领域参与度方面的成效，涵盖论文录用、导师配对、参会规模与资助情况，为后续多样性倡议提供参考。**

- **链接: []()**

> **作者:** Estefania Talavera; Deblina Bhattacharjee; Himangi Mittal; Mengwei Ren; Karen Sanchez; Carla Muntean; JungEun Kim; Mona Jalal
>
> **摘要:** The Women in Computer Vision Workshop (WiCV@CVPR 2025) was held in conjunction with the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025) in Nashville, Tennessee, United States. This report presents an overview of the workshop program, participation statistics, mentorship outcomes, and historical trends from previous WiCV editions. The goal is to document the impact and evolution of WiCV as a reference for future editions and for other initiatives aimed at advancing diversity, equity, and inclusion within the AI and computer vision communities. WiCV@CVPR 2025 marked the 16th edition of this long-standing event dedicated to increasing the visibility, inclusion, and professional growth of women and underrepresented minorities in the computer vision community. This year's workshop featured 14 accepted papers in the CVPR Workshop Proceedings out of 32 full-paper submissions. Five of these were selected for oral presentations, while all 14 were also presented as posters, along with 36 extended abstract posters accepted from 62 short-paper submissions, which are not included in the proceedings. The mentoring program matched 80 mentees with 37 mentors from both academia and industry. The 2025 edition attracted over 100 onsite participants, fostering rich technical and networking interactions across all career stages. Supported by 10 sponsors and approximately $44,000 USD in travel grants and diversity awards, WiCV continued its mission to empower emerging researchers and amplify diverse voices in computer vision.
>
---
#### [new 048] HOTFLoc++: End-to-End Hierarchical LiDAR Place Recognition, Re-Ranking, and 6-DoF Metric Localisation in Forests
- **分类: cs.CV; cs.RO**

- **简介: HOTFLoc++提出一种端到端LiDAR定位框架，用于森林等复杂环境中的场景识别、重排序与6自由度精确定位。通过八叉树变换器提取多尺度特征，结合可学习几何验证模块，显著提升鲁棒性与效率，Recall@1达90.7%以上。**

- **链接: []()**

> **作者:** Ethan Griffiths; Maryam Haghighat; Simon Denman; Clinton Fookes; Milad Ramezani
>
> **备注:** 9 pages, 2 figures. Submitted to RA-L
>
> **摘要:** This article presents HOTFLoc++, an end-to-end framework for LiDAR place recognition, re-ranking, and 6-DoF metric localisation in forests. Leveraging an octree-based transformer, our approach extracts hierarchical local descriptors at multiple granularities to increase robustness to clutter, self-similarity, and viewpoint changes in challenging scenarios, including ground-to-ground and ground-to-aerial in forest and urban environments. We propose a learnable multi-scale geometric verification module to reduce re-ranking failures in the presence of degraded single-scale correspondences. Our coarse-to-fine registration approach achieves comparable or lower localisation errors to baselines, with runtime improvements of two orders of magnitude over RANSAC for dense point clouds. Experimental results on public datasets show the superiority of our approach compared to state-of-the-art methods, achieving an average Recall@1 of 90.7% on CS-Wild-Places: an improvement of 29.6 percentage points over baselines, while maintaining high performance on single-source benchmarks with an average Recall@1 of 91.7% and 96.0% on Wild-Places and MulRan, respectively. Our method achieves under 2 m and 5 degrees error for 97.2% of 6-DoF registration attempts, with our multi-scale re-ranking module reducing localisation errors by ~2$\times$ on average. The code will be available upon acceptance.
>
---
#### [new 049] USF-Net: A Unified Spatiotemporal Fusion Network for Ground-Based Remote Sensing Cloud Image Sequence Extrapolation
- **分类: cs.CV**

- **简介: 该论文提出USF-Net，用于地面遥感云序列外推任务，解决现有方法特征提取静态、时序建模弱和计算开销大等问题，通过自适应大核卷积与低复杂度注意力机制，实现高效高精度的时空联合预测，并发布新数据集ASI-CIS。**

- **链接: []()**

> **作者:** Penghui Niu; Taotao Cai; Jiashuai She; Yajuan Zhang; Junhua Gua; Ping Zhanga; Jungong Hane; Jianxin Li
>
> **摘要:** Ground-based remote sensing cloud image sequence extrapolation is a key research area in the development of photovoltaic power systems. However, existing approaches exhibit several limitations:(1)they primarily rely on static kernels to augment feature information, lacking adaptive mechanisms to extract features at varying resolutions dynamically;(2)temporal guidance is insufficient, leading to suboptimal modeling of long-range spatiotemporal dependencies; and(3)the quadratic computational cost of attention mechanisms is often overlooked, limiting efficiency in practical deployment. To address these challenges, we propose USF-Net, a Unified Spatiotemporal Fusion Network that integrates adaptive large-kernel convolutions and a low-complexity attention mechanism, combining temporal flow information within an encoder-decoder framework. Specifically, the encoder employs three basic layers to extract features. Followed by the USTM, which comprises:(1)a SiB equipped with a SSM that dynamically captures multi-scale contextual information, and(2)a TiB featuring a TAM that effectively models long-range temporal dependencies while maintaining computational efficiency. In addition, a DSM with a TGM is introduced to enable unified modeling of temporally guided spatiotemporal dependencies. On the decoder side, a DUM is employed to address the common "ghosting effect." It utilizes the initial temporal state as an attention operator to preserve critical motion signatures. As a key contribution, we also introduce and release the ASI-CIS dataset. Extensive experiments on ASI-CIS demonstrate that USF-Net significantly outperforms state-of-the-art methods, establishing a superior balance between prediction accuracy and computational efficiency for ground-based cloud extrapolation. The dataset and source code will be available at https://github.com/she1110/ASI-CIS.
>
---
#### [new 050] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对半监督医学图像分割中的误差传播问题，提出DualFete框架，引入师生反馈机制，让学生反馈伪标签扰动以修正教师标签，并设计双教师反馈模型提升纠错能力。**

- **链接: []()**

> **作者:** Le Yi; Wei Huang; Lei Zhang; Kefu Zhao; Yan Wang; Zizhou Wang
>
> **备注:** Accepted by Proceedings of the AAAI Conference on Artificial Intelligence 40 (AAAI-26)
>
> **摘要:** The teacher-student paradigm has emerged as a canonical framework in semi-supervised learning. When applied to medical image segmentation, the paradigm faces challenges due to inherent image ambiguities, making it particularly vulnerable to erroneous supervision. Crucially, the student's iterative reconfirmation of these errors leads to self-reinforcing bias. While some studies attempt to mitigate this bias, they often rely on external modifications to the conventional teacher-student framework, overlooking its intrinsic potential for error correction. In response, this work introduces a feedback mechanism into the teacher-student framework to counteract error reconfirmations. Here, the student provides feedback on the changes induced by the teacher's pseudo-labels, enabling the teacher to refine these labels accordingly. We specify that this interaction hinges on two key components: the feedback attributor, which designates pseudo-labels triggering the student's update, and the feedback receiver, which determines where to apply this feedback. Building on this, a dual-teacher feedback model is further proposed, which allows more dynamics in the feedback loop and fosters more gains by resolving disagreements through cross-teacher supervision while avoiding consistent errors. Comprehensive evaluations on three medical image benchmarks demonstrate the method's effectiveness in addressing error propagation in semi-supervised medical image segmentation.
>
---
#### [new 051] PAN: A World Model for General, Interactable, and Long-Horizon World Simulation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: PAN提出一种通用可交互的长时序世界模型，通过LLM驱动的潜变量动态与视频扩散解码器，实现自然语言动作引导的高保真未来状态预测，解决现有模型缺乏因果控制与泛化能力的问题。**

- **链接: []()**

> **作者:** PAN Team; Jiannan Xiang; Yi Gu; Zihan Liu; Zeyu Feng; Qiyue Gao; Yiyan Hu; Benhao Huang; Guangyi Liu; Yichi Yang; Kun Zhou; Davit Abrahamyan; Arif Ahmad; Ganesh Bannur; Junrong Chen; Kimi Chen; Mingkai Deng; Ruobing Han; Xinqi Huang; Haoqiang Kang; Zheqi Li; Enze Ma; Hector Ren; Yashowardhan Shinde; Rohan Shingre; Ramsundar Tanikella; Kaiming Tao; Dequan Yang; Xinle Yu; Cong Zeng; Binglin Zhou; Hector Liu; Zhiting Hu; Eric P. Xing
>
> **摘要:** A world model enables an intelligent agent to imagine, predict, and reason about how the world evolves in response to its actions, and accordingly to plan and strategize. While recent video generation models produce realistic visual sequences, they typically operate in the prompt-to-full-video manner without causal control, interactivity, or long-horizon consistency required for purposeful reasoning. Existing world modeling efforts, on the other hand, often focus on restricted domains (e.g., physical, game, or 3D-scene dynamics) with limited depth and controllability, and struggle to generalize across diverse environments and interaction formats. In this work, we introduce PAN, a general, interactable, and long-horizon world model that predicts future world states through high-quality video simulation conditioned on history and natural language actions. PAN employs the Generative Latent Prediction (GLP) architecture that combines an autoregressive latent dynamics backbone based on a large language model (LLM), which grounds simulation in extensive text-based knowledge and enables conditioning on language-specified actions, with a video diffusion decoder that reconstructs perceptually detailed and temporally coherent visual observations, to achieve a unification between latent space reasoning (imagination) and realizable world dynamics (reality). Trained on large-scale video-action pairs spanning diverse domains, PAN supports open-domain, action-conditioned simulation with coherent, long-term dynamics. Extensive experiments show that PAN achieves strong performance in action-conditioned world simulation, long-horizon forecasting, and simulative reasoning compared to other video generators and world models, taking a step towards general world models that enable predictive simulation of future world states for reasoning and acting.
>
---
#### [new 052] A Multi-Drone Multi-View Dataset and Deep Learning Framework for Pedestrian Detection and Tracking
- **分类: cs.CV; cs.IT; cs.LG; cs.RO; eess.IV**

- **简介: 该论文提出MATRIX数据集与深度学习框架，解决动态多无人机视角下行人检测与跟踪难题，通过BEV特征融合与实时校准，在复杂遮挡环境中实现约90%精度，显著优于静态相机方法。**

- **链接: []()**

> **作者:** Kosta Dakic; Kanchana Thilakarathna; Rodrigo N. Calheiros; Teng Joon Lim
>
> **备注:** Introduction of the MATRIX Dataset, featuring synchronized footage from eight drones in an urban environment with comprehensive annotations for detection and tracking, available at https://github.com/KostaDakic/MATRIX/tree/main
>
> **摘要:** Multi-drone surveillance systems offer enhanced coverage and robustness for pedestrian tracking, yet existing approaches struggle with dynamic camera positions and complex occlusions. This paper introduces MATRIX (Multi-Aerial TRacking In compleX environments), a comprehensive dataset featuring synchronized footage from eight drones with continuously changing positions, and a novel deep learning framework for multi-view detection and tracking. Unlike existing datasets that rely on static cameras or limited drone coverage, MATRIX provides a challenging scenario with 40 pedestrians and a significant architectural obstruction in an urban environment. Our framework addresses the unique challenges of dynamic drone-based surveillance through real-time camera calibration, feature-based image registration, and multi-view feature fusion in bird's-eye-view (BEV) representation. Experimental results demonstrate that while static camera methods maintain over 90\% detection and tracking precision and accuracy metrics in a simplified MATRIX environment without an obstruction, 10 pedestrians and a much smaller observational area, their performance significantly degrades in the complex environment. Our proposed approach maintains robust performance with $\sim$90\% detection and tracking accuracy, as well as successfully tracks $\sim$80\% of trajectories under challenging conditions. Transfer learning experiments reveal strong generalization capabilities, with the pretrained model achieving much higher detection and tracking accuracy performance compared to training the model from scratch. Additionally, systematic camera dropout experiments reveal graceful performance degradation, demonstrating practical robustness for real-world deployments where camera failures may occur. The MATRIX dataset and framework provide essential benchmarks for advancing dynamic multi-view surveillance systems.
>
---
#### [new 053] RS-Net: Context-Aware Relation Scoring for Dynamic Scene Graph Generation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出RS-Net，用于动态场景图生成（DSGG），解决关系稀疏导致的长尾分布预测难题。通过融合空间与时间上下文对物体对进行关系评分，提升关系预测精度与召回率，可无缝集成至现有模型。**

- **链接: []()**

> **作者:** Hae-Won Jo; Yeong-Jun Cho
>
> **摘要:** Dynamic Scene Graph Generation (DSGG) models how object relations evolve over time in videos. However, existing methods are trained only on annotated object pairs and lack guidance for non-related pairs, making it difficult to identify meaningful relations during inference. In this paper, we propose Relation Scoring Network (RS-Net), a modular framework that scores the contextual importance of object pairs using both spatial interactions and long-range temporal context. RS-Net consists of a spatial context encoder with learnable context tokens and a temporal encoder that aggregates video-level information. The resulting relation scores are integrated into a unified triplet scoring mechanism to enhance relation prediction. RS-Net can be easily integrated into existing DSGG models without architectural changes. Experiments on the Action Genome dataset show that RS-Net consistently improves both Recall and Precision across diverse baselines, with notable gains in mean Recall, highlighting its ability to address the long-tailed distribution of relations. Despite the increased number of parameters, RS-Net maintains competitive efficiency, achieving superior performance over state-of-the-art methods.
>
---
#### [new 054] Negative Entity Suppression for Zero-Shot Captioning with Synthetic Images
- **分类: cs.CV**

- **简介: 该论文针对零样本图像描述（ZIC）中的幻觉问题，提出负实体抑制（NES）方法，利用合成图像增强检索一致性，过滤并抑制与图像无关的实体，提升跨域泛化能力，降低幻觉率，达到新SOTA。**

- **链接: []()**

> **作者:** Zimao Lu; Hui Xu; Bing Liu; Ke Wang
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Text-only training provides an attractive approach to address data scarcity challenges in zero-shot image captioning (ZIC), avoiding the expense of collecting paired image-text annotations. However, although these approaches perform well within training domains, they suffer from poor cross-domain generalization, often producing hallucinated content when encountering novel visual environments. Retrieval-based methods attempt to mitigate this limitation by leveraging external knowledge, but they can paradoxically exacerbate hallucination when retrieved captions contain entities irrelevant to the inputs. We introduce the concept of negative entities--objects that appear in generated caption but are absent from the input--and propose Negative Entity Suppression (NES) to tackle this challenge. NES seamlessly integrates three stages: (1) it employs synthetic images to ensure consistent image-to-text retrieval across both training and inference; (2) it filters negative entities from retrieved content to enhance accuracy; and (3) it applies attention-level suppression using identified negative entities to further minimize the impact of hallucination-prone features. Evaluation across multiple benchmarks demonstrates that NES maintains competitive in-domain performance while improving cross-domain transfer and reducing hallucination rates, achieving new state-of-the-art results in ZIC. Our code is available at https://github.com/nidongpinyinme/NESCap.
>
---
#### [new 055] Spatial Information Bottleneck for Interpretable Visual Recognition
- **分类: cs.CV**

- **简介: 该论文提出Spatial Information Bottleneck（S-IB），通过信息论框架优化梯度反传（VJP）的时空结构，解耦前景与背景特征，提升视觉解释的可解释性与鲁棒性，无需调参即可通用提升六种解释方法效果。**

- **链接: []()**

> **作者:** Kaixiang Shu; Kai Meng; Junqin Luo
>
> **摘要:** Deep neural networks typically learn spatially entangled representations that conflate discriminative foreground features with spurious background correlations, thereby undermining model interpretability and robustness. We propose a novel understanding framework for gradient-based attribution from an information-theoretic perspective. We prove that, under mild conditions, the Vector-Jacobian Products (VJP) computed during backpropagation form minimal sufficient statistics of input features with respect to class labels. Motivated by this finding, we propose an encoding-decoding perspective : forward propagation encodes inputs into class space, while VJP in backpropagation decodes this encoding back to feature space. Therefore, we propose Spatial Information Bottleneck (S-IB) to spatially disentangle information flow. By maximizing mutual information between foreground VJP and inputs while minimizing mutual information in background regions, S-IB encourages networks to encode information only in class-relevant spatial regions. Since post-hoc explanation methods fundamentally derive from VJP computations, directly optimizing VJP's spatial structure during training improves visualization quality across diverse explanation paradigms. Experiments on five benchmarks demonstrate universal improvements across six explanation methods, achieving better foreground concentration and background suppression without method-specific tuning, alongside consistent classification accuracy gains.
>
---
#### [new 056] DensiCrafter: Physically-Constrained Generation and Fabrication of Self-Supporting Hollow Structures
- **分类: cs.CV; cs.AI**

- **简介: DensiCrafter针对3D生成模型忽视物理可制造性的问题，提出一种无需重训练的密度场优化框架，通过物理约束损失函数自动生成轻量化、自支撑中空结构，实现材料减重43%并保障可打印性。**

- **链接: []()**

> **作者:** Shengqi Dang; Fu Chai; Jiaxin Li; Chao Yuan; Wei Ye; Nan Cao
>
> **摘要:** The rise of 3D generative models has enabled automatic 3D geometry and texture synthesis from multimodal inputs (e.g., text or images). However, these methods often ignore physical constraints and manufacturability considerations. In this work, we address the challenge of producing 3D designs that are both lightweight and self-supporting. We present DensiCrafter, a framework for generating lightweight, self-supporting 3D hollow structures by optimizing the density field. Starting from coarse voxel grids produced by Trellis, we interpret these as continuous density fields to optimize and introduce three differentiable, physically constrained, and simulation-free loss terms. Additionally, a mass regularization penalizes unnecessary material, while a restricted optimization domain preserves the outer surface. Our method seamlessly integrates with pretrained Trellis-based models (e.g., Trellis, DSO) without any architectural changes. In extensive evaluations, we achieve up to 43% reduction in material mass on the text-to-3D task. Compared to state-of-the-art baselines, our method could improve the stability and maintain high geometric fidelity. Real-world 3D-printing experiments confirm that our hollow designs can be reliably fabricated and could be self-supporting.
>
---
#### [new 057] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery
- **分类: cs.CV; cs.AI**

- **简介: 论文提出PressTrack-HMR，面向多人群体人体网格重建任务，解决压力信号混叠难以分离的问题，通过跟踪检测分离个体压力信号并重建三维人体网格，构建MIP数据集，实现隐私友好的多人体动作感知。**

- **链接: []()**

> **作者:** Jiayue Yuan; Fangting Xie; Guangwen Ouyang; Changhai Ma; Ziyu Wu; Heyu Ding; Quan Wan; Yi Ke; Yuchen Wu; Xiaohui Cai
>
> **备注:** Accepeted by AAAI26
>
> **摘要:** Multi-person global human mesh recovery (HMR) is crucial for understanding crowd dynamics and interactions. Traditional vision-based HMR methods sometimes face limitations in real-world scenarios due to mutual occlusions, insufficient lighting, and privacy concerns. Human-floor tactile interactions offer an occlusion-free and privacy-friendly alternative for capturing human motion. Existing research indicates that pressure signals acquired from tactile mats can effectively estimate human pose in single-person scenarios. However, when multiple individuals walk randomly on the mat simultaneously, how to distinguish intermingled pressure signals generated by different persons and subsequently acquire individual temporal pressure data remains a pending challenge for extending pressure-based HMR to the multi-person situation. In this paper, we present \textbf{PressTrack-HMR}, a top-down pipeline that recovers multi-person global human meshes solely from pressure signals. This pipeline leverages a tracking-by-detection strategy to first identify and segment each individual's pressure signal from the raw pressure data, and subsequently performs HMR for each extracted individual signal. Furthermore, we build a multi-person interaction pressure dataset \textbf{MIP}, which facilitates further research into pressure-based human motion analysis in multi-person scenarios. Experimental results demonstrate that our method excels in multi-person HMR using pressure data, with 89.2~$mm$ MPJPE and 112.6~$mm$ WA-MPJPE$_{100}$, and these showcase the potential of tactile mats for ubiquitous, privacy-preserving multi-person action recognition. Our dataset \& code are available at https://github.com/Jiayue-Yuan/PressTrack-HMR.
>
---
#### [new 058] Deep Learning for Metabolic Rate Estimation from Biosignals: A Comparative Study of Architectures and Signal Selection
- **分类: cs.CV**

- **简介: 该论文研究利用深度学习从生物信号估算代谢率，系统比较不同神经架构与信号组合的性能，发现通气量最有效，Transformer模型表现最优，并揭示个体差异需自适应建模。**

- **链接: []()**

> **作者:** Sarvenaz Babakhani; David Remy; Alina Roitberg
>
> **备注:** Accepted at the MPI Workshop, BMVC 2025. 17 pages, 6 figures. Code available at https://github.com/Sarvibabakhani/deeplearning-biosignals-ee
>
> **摘要:** Energy expenditure estimation aims to infer human metabolic rate from physiological signals such as heart rate, respiration, or accelerometer data, and has been studied primarily with classical regression methods. The few existing deep learning approaches rarely disentangle the role of neural architecture from that of signal choice. In this work, we systematically evaluate both aspects. We compare classical baselines with newer neural architectures across single signals, signal pairs, and grouped sensor inputs for diverse physical activities. Our results show that minute ventilation is the most predictive individual signal, with a transformer model achieving the lowest root mean square error (RMSE) of 0.87 W/kg across all activities. Paired and grouped signals, such as those from the Hexoskin smart shirt (five signals), offer good alternatives for faster models like CNN and ResNet with attention. Per-activity evaluation revealed mixed outcomes: notably better results in low-intensity activities (RMSE down to 0.29 W/kg; NRMSE = 0.04), while higher-intensity tasks showed larger RMSE but more comparable normalized errors. Finally, subject-level analysis highlights strong inter-individual variability, motivating the need for adaptive modeling strategies. Our code and models will be publicly available at https://github.com/Sarvibabakhani/deeplearning-biosignals-ee .
>
---
#### [new 059] Predict and Resist: Long-Term Accident Anticipation under Sensor Noise
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向自动驾驶安全，解决传感器噪声下事故预测的延迟与误报问题。提出融合扩散去噪与时间感知强化学习的框架，提升噪声环境下的预测准确性与响应时效性。**

- **链接: []()**

> **作者:** Xingcheng Liu; Bin Rao; Yanchen Guan; Chengyue Wang; Haicheng Liao; Jiaxun Zhang; Chengyu Lin; Meixin Zhu; Zhenning Li
>
> **备注:** accepted by the Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** Accident anticipation is essential for proactive and safe autonomous driving, where even a brief advance warning can enable critical evasive actions. However, two key challenges hinder real-world deployment: (1) noisy or degraded sensory inputs from weather, motion blur, or hardware limitations, and (2) the need to issue timely yet reliable predictions that balance early alerts with false-alarm suppression. We propose a unified framework that integrates diffusion-based denoising with a time-aware actor-critic model to address these challenges. The diffusion module reconstructs noise-resilient image and object features through iterative refinement, preserving critical motion and interaction cues under sensor degradation. In parallel, the actor-critic architecture leverages long-horizon temporal reasoning and time-weighted rewards to determine the optimal moment to raise an alert, aligning early detection with reliability. Experiments on three benchmark datasets (DAD, CCD, A3D) demonstrate state-of-the-art accuracy and significant gains in mean time-to-accident, while maintaining robust performance under Gaussian and impulse noise. Qualitative analyses further show that our model produces earlier, more stable, and human-aligned predictions in both routine and highly complex traffic scenarios, highlighting its potential for real-world, safety-critical deployment.
>
---
#### [new 060] SasMamba: A Lightweight Structure-Aware Stride State Space Model for 3D Human Pose Estimation
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 论文面向3D人体姿态估计任务，针对现有SSM方法破坏骨骼空间结构的问题，提出SAS-SSM模块，通过结构感知卷积与步幅扫描策略，保留时空结构并实现线性复杂度建模，构建轻量模型SasMamba，性能优越且参数更少。**

- **链接: []()**

> **作者:** Hu Cui; Wenqiang Hua; Renjing Huang; Shurui Jia; Tessai Hayama
>
> **备注:** 8pages, WACV2026 accepted
>
> **摘要:** Recently, the Mamba architecture based on State Space Models (SSMs) has gained attention in 3D human pose estimation due to its linear complexity and strong global modeling capability. However, existing SSM-based methods typically apply manually designed scan operations to flatten detected 2D pose sequences into purely temporal sequences, either locally or globally. This approach disrupts the inherent spatial structure of human poses and entangles spatial and temporal features, making it difficult to capture complex pose dependencies. To address these limitations, we propose the Skeleton Structure-Aware Stride SSM (SAS-SSM), which first employs a structure-aware spatiotemporal convolution to dynamically capture essential local interactions between joints, and then applies a stride-based scan strategy to construct multi-scale global structural representations. This enables flexible modeling of both local and global pose information while maintaining linear computational complexity. Built upon SAS-SSM, our model SasMamba achieves competitive 3D pose estimation performance with significantly fewer parameters compared to existing hybrid models. The source code is available at https://hucui2022.github.io/sasmamba_proj/.
>
---
#### [new 061] Dense Cross-Scale Image Alignment With Fully Spatial Correlation and Just Noticeable Difference Guidance
- **分类: cs.CV**

- **简介: 该论文针对无监督图像对齐任务，提出一种稠密跨尺度对齐模型，利用全空间相关性与JND引导，提升精度并降低计算复杂度，实现精度与效率的灵活平衡。**

- **链接: []()**

> **作者:** Jinkun You; Jiaxue Li; Jie Zhang; Yicong Zhou
>
> **摘要:** Existing unsupervised image alignment methods exhibit limited accuracy and high computational complexity. To address these challenges, we propose a dense cross-scale image alignment model. It takes into account the correlations between cross-scale features to decrease the alignment difficulty. Our model supports flexible trade-offs between accuracy and efficiency by adjusting the number of scales utilized. Additionally, we introduce a fully spatial correlation module to further improve accuracy while maintaining low computational costs. We incorporate the just noticeable difference to encourage our model to focus on image regions more sensitive to distortions, eliminating noticeable alignment errors. Extensive quantitative and qualitative experiments demonstrate that our method surpasses state-of-the-art approaches.
>
---
#### [new 062] Consistency Change Detection Framework for Unsupervised Remote Sensing Change Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对无监督遥感变化检测中生成器过拟合问题，提出一致性变化检测框架（CCDF），引入循环一致性和语义一致性模块，提升重建精度与变化检测性能，无需标注数据即可有效识别地表变化区域。**

- **链接: []()**

> **作者:** Yating Liu; Yan Lu
>
> **备注:** 2025 IEEE International Conference on Multimedia and Expo (ICME)
>
> **摘要:** Unsupervised remote sensing change detection aims to monitor and analyze changes from multi-temporal remote sensing images in the same geometric region at different times, without the need for labeled training data. Previous unsupervised methods attempt to achieve style transfer across multi-temporal remote sensing images through reconstruction by a generator network, and then capture the unreconstructable areas as the changed regions. However, it often leads to poor performance due to generator overfitting. In this paper, we propose a novel Consistency Change Detection Framework (CCDF) to address this challenge. Specifically, we introduce a Cycle Consistency (CC) module to reduce the overfitting issues in the generator-based reconstruction. Additionally, we propose a Semantic Consistency (SC) module to enable detail reconstruction. Extensive experiments demonstrate that our method outperforms other state-of-the-art approaches.
>
---
#### [new 063] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection
- **分类: cs.CV**

- **简介: 论文针对多视角3D目标检测中PETR模型量化导致精度骤降的问题，提出FQ-PETR框架，通过LiDAR引导位置编码、双查表非线性近似和量化前数值稳定化，实现W8A8全量化下接近浮点精度，延迟降低75%。**

- **链接: []()**

> **作者:** Jiangyong Yu; Changyong Shu; Sifan Zhou; Zichen Yu; Xing Hu; Yan Chen; Dawei Yang
>
> **摘要:** Camera-based multi-view 3D detection is crucial for autonomous driving. PETR and its variants (PETRs) excel in benchmarks but face deployment challenges due to high computational cost and memory footprint. Quantization is an effective technique for compressing deep neural networks by reducing the bit width of weights and activations. However, directly applying existing quantization methods to PETRs leads to severe accuracy degradation. This issue primarily arises from two key challenges: (1) significant magnitude disparity between multi-modal features-specifically, image features and camera-ray positional embeddings (PE), and (2) the inefficiency and approximation error of quantizing non-linear operators, which commonly rely on hardware-unfriendly computations. In this paper, we propose FQ-PETR, a fully quantized framework for PETRs, featuring three key innovations: (1) Quantization-Friendly LiDAR-ray Position Embedding (QFPE): Replacing multi-point sampling with LiDAR-prior-guided single-point sampling and anchor-based embedding eliminates problematic non-linearities (e.g., inverse-sigmoid) and aligns PE scale with image features, preserving accuracy. (2) Dual-Lookup Table (DULUT): This algorithm approximates complex non-linear functions using two cascaded linear LUTs, achieving high fidelity with minimal entries and no specialized hardware. (3) Quantization After Numerical Stabilization (QANS): Performing quantization after softmax numerical stabilization mitigates attention distortion from large inputs. On PETRs (e.g. PETR, StreamPETR, PETRv2, MV2d), FQ-PETR under W8A8 achieves near-floating-point accuracy (1% degradation) while reducing latency by up to 75%, significantly outperforming existing PTQ and QAT baselines.
>
---
#### [new 064] Neural B-frame Video Compression with Bi-directional Reference Harmonization
- **分类: cs.CV**

- **简介: 该论文提出BRHVC，解决神经B帧视频压缩中双向参考帧贡献不平衡问题，通过双向运动收敛与上下文融合，提升运动补偿精度与参考信息利用效率，性能超越传统编码与现有神经方法。**

- **链接: []()**

> **作者:** Yuxi Liu; Dengchao Jin; Shuai Huo; Jiawen Gu; Chao Zhou; Huihui Bai; Ming Lu; Zhan Ma
>
> **摘要:** Neural video compression (NVC) has made significant progress in recent years, while neural B-frame video compression (NBVC) remains underexplored compared to P-frame compression. NBVC can adopt bi-directional reference frames for better compression performance. However, NBVC's hierarchical coding may complicate continuous temporal prediction, especially at some hierarchical levels with a large frame span, which could cause the contribution of the two reference frames to be unbalanced. To optimize reference information utilization, we propose a novel NBVC method, termed Bi-directional Reference Harmonization Video Compression (BRHVC), with the proposed Bi-directional Motion Converge (BMC) and Bi-directional Contextual Fusion (BCF). BMC converges multiple optical flows in motion compression, leading to more accurate motion compensation on a larger scale. Then BCF explicitly models the weights of reference contexts under the guidance of motion compensation accuracy. With more efficient motions and contexts, BRHVC can effectively harmonize bi-directional references. Experimental results indicate that our BRHVC outperforms previous state-of-the-art NVC methods, even surpassing the traditional coding, VTM-RA (under random access configuration), on the HEVC datasets. The source code is released at https://github.com/kwai/NVC.
>
---
#### [new 065] DBINDS - Can Initial Noise from Diffusion Model Inversion Help Reveal AI-Generated Videos?
- **分类: cs.CV**

- **简介: 论文提出DBINDS，用于检测AI生成视频。通过扩散模型逆向恢复初始噪声，发现其在真实与生成视频中存在系统差异，构建INDS特征并用LightGBM分类，实现跨生成器的强泛化能力。**

- **链接: []()**

> **作者:** Yanlin Wu; Xiaogang Yuan; Dezhi An
>
> **备注:** Preprint. Submitted to IEEE Transactions on Dependable and Secure Computing (TDSC) on 16 September 2025
>
> **摘要:** AI-generated video has advanced rapidly and poses serious challenges to content security and forensic analysis. Existing detectors rely mainly on pixel-level visual cues and generalize poorly to unseen generators. We propose DBINDS, a diffusion-model-inversion based detector that analyzes latent-space dynamics rather than pixels. We find that initial noise sequences recovered by diffusion inversion differ systematically between real and generated videos. Building on this, DBINDS forms an Initial Noise Difference Sequence (INDS) and extracts multi-domain, multi-scale features. With feature optimization and a LightGBM classifier tuned by Bayesian search, DBINDS (trained on a single generator) achieves strong cross-generator performance on GenVidBench, demonstrating good generalization and robustness in limited-data settings.
>
---
#### [new 066] MACEval: A Multi-Agent Continual Evaluation Network for Large Models
- **分类: cs.CV**

- **简介: MACEval提出一种多智能体持续评估网络，解决大模型评估中数据污染、人工依赖与静态基准问题，实现自动化、低成本、可扩展的动态评估，支持长期性能追踪与现有基准集成。**

- **链接: []()**

> **作者:** Zijian Chen; Yuze Sun; Yuan Tian; Wenjun Zhang; Guangtao Zhai
>
> **备注:** 38 pages, 12 figures
>
> **摘要:** Hundreds of benchmarks dedicated to evaluating large models from multiple perspectives have been presented over the past few years. Albeit substantial efforts, most of them remain closed-ended and are prone to overfitting due to the potential data contamination in the ever-growing training corpus of large models, thereby undermining the credibility of the evaluation. Moreover, the increasing scale and scope of current benchmarks with transient metrics, as well as the heavily human-dependent curation procedure, pose significant challenges for timely maintenance and adaptation to gauge the advancing capabilities of large models. In this paper, we introduce MACEval, a \Multi-Agent Continual Evaluation network for dynamic evaluation of large models, and define a new set of metrics to quantify performance longitudinally and sustainably. MACEval adopts an interactive and autonomous evaluation mode that employs role assignment, in-process data generation, and evaluation routing through a cascaded agent network. Extensive experiments on 9 open-ended tasks with 23 participating large models demonstrate that MACEval is (1) human-free and automatic, mitigating laborious result processing with inter-agent judgment guided; (2) efficient and economical, reducing a considerable amount of data and overhead to obtain similar results compared to related benchmarks; and (3) flexible and scalable, migrating or integrating existing benchmarks via customized evaluation topologies. We hope that MACEval can broaden future directions of large model evaluation.
>
---
#### [new 067] PIFF: A Physics-Informed Generative Flow Model for Real-Time Flood Depth Mapping
- **分类: cs.CV**

- **简介: PIFF提出一种物理信息生成流模型，用于实时洪水深度预测，解决传统方法效率低问题。通过融合DEM、降雨时序与简化水动力先验，实现从地形与降雨到洪水图的高效映射，替代昂贵数值模拟。**

- **链接: []()**

> **作者:** ChunLiang Wu; Tsunhua Yang; Hungying Chen
>
> **摘要:** Flood mapping is crucial for assessing and mitigating flood impacts, yet traditional methods like numerical modeling and aerial photography face limitations in efficiency and reliability. To address these challenges, we propose PIFF, a physics-informed, flow-based generative neural network for near real-time flood depth estimation. Built on an image-to-image generative framework, it efficiently maps Digital Elevation Models (DEM) to flood depth predictions. The model is conditioned on a simplified inundation model (SPM) that embeds hydrodynamic priors into the training process. Additionally, a transformer-based rainfall encoder captures temporal dependencies in precipitation. Integrating physics-informed constraints with data-driven learning, PIFF captures the causal relationships between rainfall, topography, SPM, and flooding, replacing costly simulations with accurate, real-time flood maps. Using a 26 km study area in Tainan, Taiwan, with 182 rainfall scenarios ranging from 24 mm to 720 mm over 24 hours, our results demonstrate that PIFF offers an effective, data-driven alternative for flood prediction and response.
>
---
#### [new 068] AuthSig: Safeguarding Scanned Signatures Against Unauthorized Reuse in Paperless Workflows
- **分类: cs.CV; cs.AI**

- **简介: 论文提出AuthSig，解决静态扫描签名易被复制滥用的问题，通过生成模型与隐式水印绑定身份信息，实现“一签一用”，并用关键点增强数据提升鲁棒性，可在多种失真下保持98%以上水印提取准确率。**

- **链接: []()**

> **作者:** RuiQiang Zhang; Zehua Ma; Guanjie Wang; Chang Liu; Hengyi Wang; Weiming Zhang
>
> **摘要:** With the deepening trend of paperless workflows, signatures as a means of identity authentication are gradually shifting from traditional ink-on-paper to electronic formats.Despite the availability of dynamic pressure-sensitive and PKI-based digital signatures, static scanned signatures remain prevalent in practice due to their convenience. However, these static images, having almost lost their authentication attributes, cannot be reliably verified and are vulnerable to malicious copying and reuse. To address these issues, we propose AuthSig, a novel static electronic signature framework based on generative models and watermark, which binds authentication information to the signature image. Leveraging the human visual system's insensitivity to subtle style variations, AuthSig finely modulates style embeddings during generation to implicitly encode watermark bits-enforcing a One Signature, One Use policy.To overcome the scarcity of handwritten signature data and the limitations of traditional augmentation methods, we introduce a keypoint-driven data augmentation strategy that effectively enhances style diversity to support robust watermark embedding. Experimental results show that AuthSig achieves over 98% extraction accuracy under both digital-domain distortions and signature-specific degradations, and remains effective even in print-scan scenarios.
>
---
#### [new 069] Improving VisNet for Object Recognition
- **分类: cs.CV**

- **简介: 该论文针对物体识别任务，改进生物启发的VisNet模型，引入RBF神经元、马氏距离学习和视网膜预处理，提升其对变换不变特征的提取能力，在多个数据集上显著提高识别与对称性分类准确率。**

- **链接: []()**

> **作者:** Mehdi Fatan Serj; C. Alejandro Parraga; Xavier Otazu
>
> **摘要:** Object recognition plays a fundamental role in how biological organisms perceive and interact with their environment. While the human visual system performs this task with remarkable efficiency, reproducing similar capabilities in artificial systems remains challenging. This study investigates VisNet, a biologically inspired neural network model, and several enhanced variants incorporating radial basis function neurons, Mahalanobis distance based learning, and retinal like preprocessing for both general object recognition and symmetry classification. By leveraging principles of Hebbian learning and temporal continuity associating temporally adjacent views to build invariant representations. VisNet and its extensions capture robust and transformation invariant features. Experimental results across multiple datasets, including MNIST, CIFAR10, and custom symmetric object sets, show that these enhanced VisNet variants substantially improve recognition accuracy compared with the baseline model. These findings underscore the adaptability and biological relevance of VisNet inspired architectures, offering a powerful and interpretable framework for visual recognition in both neuroscience and artificial intelligence. Keywords: VisNet, Object Recognition, Symmetry Detection, Hebbian Learning, RBF Neurons, Mahalanobis Distance, Biologically Inspired Models, Invariant Representations
>
---
#### [new 070] FSampler: Training Free Acceleration of Diffusion Sampling via Epsilon Extrapolation
- **分类: cs.LG; cs.CV**

- **简介: FSampler是一种无训练、与采样器无关的加速框架，通过二至四阶外推历史噪声信号（epsilon）减少扩散模型采样中的函数调用次数，在保持采样规则不变的前提下，显著提升推理效率。**

- **链接: []()**

> **作者:** Michael A. Vladimir
>
> **备注:** 10 pages; diffusion models; accelerated sampling; ODE solvers; epsilon extrapolation; training free inference
>
> **摘要:** FSampler is a training free, sampler agnostic execution layer that accelerates diffusion sampling by reducing the number of function evaluations (NFE). FSampler maintains a short history of denoising signals (epsilon) from recent real model calls and extrapolates the next epsilon using finite difference predictors at second order, third order, or fourth order, falling back to lower order when history is insufficient. On selected steps the predicted epsilon substitutes the model call while keeping each sampler's update rule unchanged. Predicted epsilons are validated for finiteness and magnitude; a learning stabilizer rescales predictions on skipped steps to correct drift, and an optional gradient estimation stabilizer compensates local curvature. Protected windows, periodic anchors, and a cap on consecutive skips bound deviation over the trajectory. Operating at the sampler level, FSampler integrates with Euler/DDIM, DPM++ 2M/2S, LMS/AB2, and RES family exponential multistep methods and drops into standard workflows. FLUX.1 dev, Qwen Image, and Wan 2.2, FSampler reduces time by 8 to 22% and model calls by 15 to 25% at high fidelity (Structural Similarity Index (SSIM) 0.95 to 0.99), without altering sampler formulas. With an aggressive adaptive gate, reductions can reach 45 to 50% fewer model calls at lower fidelity (SSIM 0.73 to 0.74).
>
---
#### [new 071] OG-PCL: Efficient Sparse Point Cloud Processing for Human Activity Recognition
- **分类: eess.SP; cs.CV**

- **简介: 该论文针对毫米波雷达稀疏点云的人体活动识别任务，提出轻量级OG-PCL网络，通过三视图并行结构与占用门控卷积（OGConv）提升精度与效率，在RadHAR数据集上以0.83M参数达到91.75%准确率。**

- **链接: []()**

> **作者:** Jiuqi Yan; Chendong Xu; Dongyu Liu
>
> **摘要:** Human activity recognition (HAR) with millimeter-wave (mmWave) radar offers a privacy-preserving and robust alternative to camera- and wearable-based approaches. In this work, we propose the Occupancy-Gated Parallel-CNN Bi-LSTM (OG-PCL) network to process sparse 3D radar point clouds produced by mmWave sensing. Designed for lightweight deployment, the parameter size of the proposed OG-PCL is only 0.83M and achieves 91.75 accuracy on the RadHAR dataset, outperforming those existing baselines such as 2D CNN, PointNet, and 3D CNN methods. We validate the advantages of the tri-view parallel structure in preserving spatial information across three dimensions while maintaining efficiency through ablation studies. We further introduce the Occupancy-Gated Convolution (OGConv) block and demonstrate the necessity of its occupancy compensation mechanism for handling sparse point clouds. The proposed OG-PCL thus offers a compact yet accurate framework for real-time radar-based HAR on lightweight platforms.
>
---
#### [new 072] History-Aware Reasoning for GUI Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文针对GUI代理历史感知不足的问题，提出History-Aware Reasoning（HAR）框架，通过反思学习与混合奖励机制增强短时记忆，使代理能利用交互历史进行连贯推理，提升长任务自动化性能。**

- **链接: []()**

> **作者:** Ziwei Wang; Leyang Yang; Xiaoxuan Tang; Sheng Zhou; Dajun Chen; Wei Jiang; Yong Li
>
> **备注:** Paper accepted to AAAI 2026
>
> **摘要:** Advances in Multimodal Large Language Models have significantly enhanced Graphical User Interface (GUI) automation. Equipping GUI agents with reliable episodic reasoning capabilities is essential for bridging the gap between users' concise task descriptions and the complexities of real-world execution. Current methods integrate Reinforcement Learning (RL) with System-2 Chain-of-Thought, yielding notable gains in reasoning enhancement. For long-horizon GUI tasks, historical interactions connect each screen to the goal-oriented episode chain, and effectively leveraging these clues is crucial for the current decision. However, existing native GUI agents exhibit weak short-term memory in their explicit reasoning, interpreting the chained interactions as discrete screen understanding, i.e., unawareness of the historical interactions within the episode. This history-agnostic reasoning challenges their performance in GUI automation. To alleviate this weakness, we propose a History-Aware Reasoning (HAR) framework, which encourages an agent to reflect on its own errors and acquire episodic reasoning knowledge from them via tailored strategies that enhance short-term memory in long-horizon interaction. The framework mainly comprises constructing a reflective learning scenario, synthesizing tailored correction guidelines, and designing a hybrid RL reward function. Using the HAR framework, we develop a native end-to-end model, HAR-GUI-3B, which alters the inherent reasoning mode from history-agnostic to history-aware, equipping the GUI agent with stable short-term memory and reliable perception of screen details. Comprehensive evaluations across a range of GUI-related benchmarks demonstrate the effectiveness and generalization of our method.
>
---
#### [new 073] Expand Your SCOPE: Semantic Cognition over Potential-Based Exploration for Embodied Visual Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对具身视觉导航任务，提出SCOPE框架，利用视觉-语言模型挖掘前沿信息构建时空势能图，结合自反思机制提升决策可靠性，实现零样本下更精准的长程导航。**

- **链接: []()**

> **作者:** Ningnan Wang; Weihuang Chen; Liming Chen; Haoxuan Ji; Zhongyu Guo; Xuchong Zhang; Hongbin Sun
>
> **摘要:** Embodied visual navigation remains a challenging task, as agents must explore unknown environments with limited knowledge. Existing zero-shot studies have shown that incorporating memory mechanisms to support goal-directed behavior can improve long-horizon planning performance. However, they overlook visual frontier boundaries, which fundamentally dictate future trajectories and observations, and fall short of inferring the relationship between partial visual observations and navigation goals. In this paper, we propose Semantic Cognition Over Potential-based Exploration (SCOPE), a zero-shot framework that explicitly leverages frontier information to drive potential-based exploration, enabling more informed and goal-relevant decisions. SCOPE estimates exploration potential with a Vision-Language Model and organizes it into a spatio-temporal potential graph, capturing boundary dynamics to support long-horizon planning. In addition, SCOPE incorporates a self-reconsideration mechanism that revisits and refines prior decisions, enhancing reliability and reducing overconfident errors. Experimental results on two diverse embodied navigation tasks show that SCOPE outperforms state-of-the-art baselines by 4.6\% in accuracy. Further analysis demonstrates that its core components lead to improved calibration, stronger generalization, and higher decision quality.
>
---
#### [new 074] RadHARSimulator V2: Video to Doppler Generator
- **分类: eess.SP; cs.CV**

- **简介: 该论文提出RadHARSimulator V2，实现从视频到雷达多普勒谱的生成，解决现有HAR仿真灵活性不足问题。结合计算机视觉与雷达仿真模块，构建端到端模拟系统，并提出新型神经网络用于动作识别。**

- **链接: []()**

> **作者:** Weicheng Gao
>
> **备注:** 19 pages, 16 figures, 8 tables
>
> **摘要:** Radar-based human activity recognition (HAR) still lacks a comprehensive simulation method. Existing software is developed based on models or motion-captured data, resulting in limited flexibility. To address this issue, a simulator that directly generates Doppler spectra from recorded video footage (RadHARSimulator V2) is presented in this paper. Both computer vision and radar modules are included in the simulator. In computer vision module, the real-time model for object detection with global nearest neighbor is first used to detect and track human targets in the video. Then, the high-resolution network is used to estimate two-dimensional poses of the detected human targets. Next, the three-dimensional poses of the detected human targets are obtained by nearest matching method. Finally, smooth temporal three-dimensional pose estimation is achieved through Kalman filtering. In radar module, pose interpolation and smoothing are first achieved through the Savitzky-Golay method. Second, the delay model and the mirror method are used to simulate echoes in both free-space and through-the-wall scenarios. Then, range-time map is generated using pulse compression, moving target indication, and DnCNN. Next, Doppler-time map (DTM) is generated using short-time Fourier transform and DnCNN again. Finally, the ridge features on the DTM are extracted using the maximum local energy method. In addition, a hybrid parallel-serial neural network architecture is proposed for radar-based HAR. Numerical experiments are conducted and analyzed to demonstrate the effectiveness of the designed simulator and the proposed network model. The open-source code of this work can be found in: https://github.com/JoeyBGOfficial/RadHARSimulatorV2-Video-to-Doppler-Generator.
>
---
#### [new 075] SAMora: Enhancing SAM through Hierarchical Self-Supervised Pre-Training for Medical Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出SAMora，面向医学图像分割任务，解决SAM在少量标注数据下性能不足的问题。通过层级自监督预训练与HL-Attn模块，融合图像、块、像素级知识，显著提升分割精度并减少90%微调迭代。**

- **链接: []()**

> **作者:** Shuhang Chen; Hangjie Yuan; Pengwei Liu; Hanxue Gu; Tao Feng; Dong Ni
>
> **摘要:** The Segment Anything Model (SAM) has demonstrated significant potential in medical image segmentation. Yet, its performance is limited when only a small amount of labeled data is available, while there is abundant valuable yet often overlooked hierarchical information in medical data. To address this limitation, we draw inspiration from self-supervised learning and propose SAMora, an innovative framework that captures hierarchical medical knowledge by applying complementary self-supervised learning objectives at the image, patch, and pixel levels. To fully exploit the complementarity of hierarchical knowledge within LoRAs, we introduce HL-Attn, a hierarchical fusion module that integrates multi-scale features while maintaining their distinct characteristics. SAMora is compatible with various SAM variants, including SAM2, SAMed, and H-SAM. Experimental results on the Synapse, LA, and PROMISE12 datasets demonstrate that SAMora outperforms existing SAM variants. It achieves state-of-the-art performance in both few-shot and fully supervised settings while reducing fine-tuning epochs by 90%. The code is available at https://github.com/ShChen233/SAMora.
>
---
#### [new 076] 3D-TDA - Topological feature extraction from 3D images for Alzheimer's disease classification
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出基于持久同调的3D-TDA方法，从脑部MRI中提取拓扑特征，用于阿尔茨海默病分类。相比深度学习模型，其无需复杂预处理，在ADNI数据集上实现更高准确率，适用于小样本场景。**

- **链接: []()**

> **作者:** Faisal Ahmed; Taymaz Akan; Fatih Gelir; Owen T. Carmichael; Elizabeth A. Disbrow; Steven A. Conrad; Mohammad A. N. Bhuiyan
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Now that disease-modifying therapies for Alzheimer disease have been approved by regulatory agencies, the early, objective, and accurate clinical diagnosis of AD based on the lowest-cost measurement modalities possible has become an increasingly urgent need. In this study, we propose a novel feature extraction method using persistent homology to analyze structural MRI of the brain. This approach converts topological features into powerful feature vectors through Betti functions. By integrating these feature vectors with a simple machine learning model like XGBoost, we achieve a computationally efficient machine learning model. Our model outperforms state-of-the-art deep learning models in both binary and three-class classification tasks for ADNI 3D MRI disease diagnosis. Using 10-fold cross-validation, our model achieved an average accuracy of 97.43 percent and sensitivity of 99.09 percent for binary classification. For three-class classification, it achieved an average accuracy of 95.47 percent and sensitivity of 94.98 percent. Unlike many deep learning models, our approach does not require data augmentation or extensive preprocessing, making it particularly suitable for smaller datasets. Topological features differ significantly from those commonly extracted using convolutional filters and other deep learning machinery. Because it provides an entirely different type of information from machine learning models, it has the potential to combine topological features with other models later on.
>
---
#### [new 077] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation
- **分类: cs.HC; cs.CV; cs.MM**

- **简介: 该论文提出一种零样本多模态框架Plug-and-Play Clarifier，用于解决第一人称视角下语言、视觉与手势模糊导致的意图歧义问题，通过三模块协同提升小模型意图理解与交互准确性。**

- **链接: []()**

> **作者:** Sicheng Yang; Yukai Huang; Weitong Cai; Shitong Sun; You He; Jiankang Deng; Hang Zhang; Jifei Song; Zhensong Zhang
>
> **备注:** 16 pages, 9 figures, AAAI 2026
>
> **摘要:** The performance of egocentric AI agents is fundamentally limited by multimodal intent ambiguity. This challenge arises from a combination of underspecified language, imperfect visual data, and deictic gestures, which frequently leads to task failure. Existing monolithic Vision-Language Models (VLMs) struggle to resolve these multimodal ambiguous inputs, often failing silently or hallucinating responses. To address these ambiguities, we introduce the Plug-and-Play Clarifier, a zero-shot and modular framework that decomposes the problem into discrete, solvable sub-tasks. Specifically, our framework consists of three synergistic modules: (1) a text clarifier that uses dialogue-driven reasoning to interactively disambiguate linguistic intent, (2) a vision clarifier that delivers real-time guidance feedback, instructing users to adjust their positioning for improved capture quality, and (3) a cross-modal clarifier with grounding mechanism that robustly interprets 3D pointing gestures and identifies the specific objects users are pointing to. Extensive experiments demonstrate that our framework improves the intent clarification performance of small language models (4--8B) by approximately 30%, making them competitive with significantly larger counterparts. We also observe consistent gains when applying our framework to these larger models. Furthermore, our vision clarifier increases corrective guidance accuracy by over 20%, and our cross-modal clarifier improves semantic answer accuracy for referential grounding by 5%. Overall, our method provides a plug-and-play framework that effectively resolves multimodal ambiguity and significantly enhances user experience in egocentric interaction.
>
---
#### [new 078] Stabilizing Direct Training of Spiking Neural Networks: Membrane Potential Initialization and Threshold-robust Surrogate Gradient
- **分类: cs.NE; cs.CV**

- **简介: 该论文针对脉冲神经网络（SNN）直接训练中的时序协变量偏移和阈值敏感梯度问题，提出MP-Init初始化膜电位与TrSG鲁棒替代梯度方法，显著提升训练稳定性与精度。**

- **链接: []()**

> **作者:** Hyunho Kook; Byeongho Yu; Jeong Min Oh; Eunhyeok Park
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Recent advancements in the direct training of Spiking Neural Networks (SNNs) have demonstrated high-quality outputs even at early timesteps, paving the way for novel energy-efficient AI paradigms. However, the inherent non-linearity and temporal dependencies in SNNs introduce persistent challenges, such as temporal covariate shift (TCS) and unstable gradient flow with learnable neuron thresholds. In this paper, we present two key innovations: MP-Init (Membrane Potential Initialization) and TrSG (Threshold-robust Surrogate Gradient). MP-Init addresses TCS by aligning the initial membrane potential with its stationary distribution, while TrSG stabilizes gradient flow with respect to threshold voltage during training. Extensive experiments validate our approach, achieving state-of-the-art accuracy on both static and dynamic image datasets. The code is available at: https://github.com/kookhh0827/SNN-MP-Init-TRSG
>
---
#### [new 079] SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 论文提出SpatialActor，面向机器人操作任务，解决深度噪声下语义与几何纠缠导致的定位不准问题，通过解耦语义与几何表示，融合多源几何信息与低层空间线索，提升操作鲁棒性与泛化能力。**

- **链接: []()**

> **作者:** Hao Shi; Bin Xie; Yingfei Liu; Yang Yue; Tiancai Wang; Haoqiang Fan; Xiangyu Zhang; Gao Huang
>
> **备注:** AAAI 2026 Oral | Project Page: https://shihao1895.github.io/SpatialActor
>
> **摘要:** Robotic manipulation requires precise spatial understanding to interact with objects in the real world. Point-based methods suffer from sparse sampling, leading to the loss of fine-grained semantics. Image-based methods typically feed RGB and depth into 2D backbones pre-trained on 3D auxiliary tasks, but their entangled semantics and geometry are sensitive to inherent depth noise in real-world that disrupts semantic understanding. Moreover, these methods focus on high-level geometry while overlooking low-level spatial cues essential for precise interaction. We propose SpatialActor, a disentangled framework for robust robotic manipulation that explicitly decouples semantics and geometry. The Semantic-guided Geometric Module adaptively fuses two complementary geometry from noisy depth and semantic-guided expert priors. Also, a Spatial Transformer leverages low-level spatial cues for accurate 2D-3D mapping and enables interaction among spatial features. We evaluate SpatialActor on multiple simulation and real-world scenarios across 50+ tasks. It achieves state-of-the-art performance with 87.4% on RLBench and improves by 13.9% to 19.4% under varying noisy conditions, showing strong robustness. Moreover, it significantly enhances few-shot generalization to new tasks and maintains robustness under various spatial perturbations. Project Page: https://shihao1895.github.io/SpatialActor
>
---
#### [new 080] IFG: Internet-Scale Guidance for Functional Grasping Generation
- **分类: cs.RO; cs.AI; cs.CV; cs.GR; cs.LG**

- **简介: 该论文提出IFG框架，结合互联网规模视觉模型的语义理解与仿真驱动的力闭合抓取生成，实现无需人工标注的实时高精度3D抓取规划，解决机器人抓取中语义与几何精度脱节的问题。**

- **链接: []()**

> **作者:** Ray Muxin Liu; Mingxuan Li; Kenneth Shaw; Deepak Pathak
>
> **备注:** Website at https://ifgrasping.github.io/
>
> **摘要:** Large Vision Models trained on internet-scale data have demonstrated strong capabilities in segmenting and semantically understanding object parts, even in cluttered, crowded scenes. However, while these models can direct a robot toward the general region of an object, they lack the geometric understanding required to precisely control dexterous robotic hands for 3D grasping. To overcome this, our key insight is to leverage simulation with a force-closure grasping generation pipeline that understands local geometries of the hand and object in the scene. Because this pipeline is slow and requires ground-truth observations, the resulting data is distilled into a diffusion model that operates in real-time on camera point clouds. By combining the global semantic understanding of internet-scale models with the geometric precision of a simulation-based locally-aware force-closure, \our achieves high-performance semantic grasping without any manually collected training data. For visualizations of this please visit our website at https://ifgrasping.github.io/
>
---
#### [new 081] MAP-VLA: Memory-Augmented Prompting for Vision-Language-Action Model in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向机器人长时程操作任务，解决预训练VLA模型缺乏记忆导致的性能不足问题，提出MAP-VLA框架，通过可学习的演示记忆提示实现即插即用的动态增强，显著提升任务成功率。**

- **链接: []()**

> **作者:** Runhao Li; Wenkai Guo; Zhenyu Wu; Changyuan Wang; Haoyuan Deng; Zhenyu Weng; Yap-Peng Tan; Ziwei Wang
>
> **摘要:** Pre-trained Vision-Language-Action (VLA) models have achieved remarkable success in improving robustness and generalization for end-to-end robotic manipulation. However, these models struggle with long-horizon tasks due to their lack of memory and reliance solely on immediate sensory inputs. To address this limitation, we propose Memory-Augmented Prompting for Vision-Language-Action model (MAP-VLA), a novel framework that empowers pre-trained VLA models with demonstration-derived memory prompts to augment action generation for long-horizon robotic manipulation tasks. To achieve this, MAP-VLA first constructs a memory library from historical demonstrations, where each memory unit captures information about a specific stage of a task. These memory units are implemented as learnable soft prompts optimized through prompt tuning. Then, during real-time task execution, MAP-VLA retrieves relevant memory through trajectory similarity matching and dynamically integrates it into the VLA model for augmented action generation. Importantly, this prompt tuning and retrieval augmentation approach operates as a plug-and-play module for a frozen VLA model, offering a lightweight and flexible solution to improve task performance. Experimental results show that MAP-VLA delivers up to 7.0% absolute performance gains in the simulation benchmark and 25.0% on real robot evaluations for long-horizon tasks, surpassing the current state-of-the-art methods.
>
---
#### [new 082] UniMM-V2X: MoE-Enhanced Multi-Level Fusion for End-to-End Cooperative Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: UniMM-V2X提出一种端到端协同自动驾驶框架，通过多级融合与MoE架构，实现感知、预测与规划的协同优化，解决单体智能感知受限与任务脱节问题，在DAIR-V2X上显著提升性能。**

- **链接: []()**

> **作者:** Ziyi Song; Chen Xia; Chenbing Wang; Haibao Yu; Sheng Zhou; Zhisheng Niu
>
> **摘要:** Autonomous driving holds transformative potential but remains fundamentally constrained by the limited perception and isolated decision-making with standalone intelligence. While recent multi-agent approaches introduce cooperation, they often focus merely on perception-level tasks, overlooking the alignment with downstream planning and control, or fall short in leveraging the full capacity of the recent emerging end-to-end autonomous driving. In this paper, we present UniMM-V2X, a novel end-to-end multi-agent framework that enables hierarchical cooperation across perception, prediction, and planning. At the core of our framework is a multi-level fusion strategy that unifies perception and prediction cooperation, allowing agents to share queries and reason cooperatively for consistent and safe decision-making. To adapt to diverse downstream tasks and further enhance the quality of multi-level fusion, we incorporate a Mixture-of-Experts (MoE) architecture to dynamically enhance the BEV representations. We further extend MoE into the decoder to better capture diverse motion patterns. Extensive experiments on the DAIR-V2X dataset demonstrate our approach achieves state-of-the-art (SOTA) performance with a 39.7% improvement in perception accuracy, a 7.2% reduction in prediction error, and a 33.2% improvement in planning performance compared with UniV2X, showcasing the strength of our MoE-enhanced multi-level cooperative paradigm.
>
---
#### [new 083] Augment to Augment: Diverse Augmentations Enable Competitive Ultra-Low-Field MRI Enhancement
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文针对超低场MRI图像质量差的问题，利用有限配对数据，在无外部数据约束下，通过多样化任务自适应数据增强提升图像重建质量，显著改善SSIM指标，实现接近高场MRI的视觉效果。**

- **链接: []()**

> **作者:** Felix F Zimmermann
>
> **备注:** MICCAI 2025 ULF-EnC Challenge
>
> **摘要:** Ultra-low-field (ULF) MRI promises broader accessibility but suffers from low signal-to-noise ratio (SNR), reduced spatial resolution, and contrasts that deviate from high-field standards. Imageto- image translation can map ULF images to a high-field appearance, yet efficacy is limited by scarce paired training data. Working within the ULF-EnC challenge constraints (50 paired 3D volumes; no external data), we study how task-adapted data augmentations impact a standard deep model for ULF image enhancement. We show that strong, diverse augmentations, including auxiliary tasks on high-field data, substantially improve fidelity. Our submission ranked third by brain-masked SSIM on the public validation leaderboard and fourth by the official score on the final test leaderboard. Code is available at https://github.com/fzimmermann89/low-field-enhancement.
>
---
#### [new 084] MicroEvoEval: A Systematic Evaluation Framework for Image-Based Microstructure Evolution Prediction
- **分类: cond-mat.mtrl-sci; cs.CV; cs.LG**

- **简介: 论文提出MicroEvoEval框架，首次系统评估图像驱动的微观结构演化预测模型，解决现有研究忽视物理保真与长期误差传播的问题，通过14种模型对比，揭示现代架构在精度、效率与物理一致性上的优势。**

- **链接: []()**

> **作者:** Qinyi Zhang; Duanyu Feng; Ronghui Han; Yangshuai Wang; Hao Wang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Simulating microstructure evolution (MicroEvo) is vital for materials design but demands high numerical accuracy, efficiency, and physical fidelity. Although recent studies on deep learning (DL) offer a promising alternative to traditional solvers, the field lacks standardized benchmarks. Existing studies are flawed due to a lack of comparing specialized MicroEvo DL models with state-of-the-art spatio-temporal architectures, an overemphasis on numerical accuracy over physical fidelity, and a failure to analyze error propagation over time. To address these gaps, we introduce MicroEvoEval, the first comprehensive benchmark for image-based microstructure evolution prediction. We evaluate 14 models, encompassing both domain-specific and general-purpose architectures, across four representative MicroEvo tasks with datasets specifically structured for both short- and long-term assessment. Our multi-faceted evaluation framework goes beyond numerical accuracy and computational cost, incorporating a curated set of structure-preserving metrics to assess physical fidelity. Our extensive evaluations yield several key insights. Notably, we find that modern architectures (e.g., VMamba), not only achieve superior long-term stability and physical fidelity but also operate with an order-of-magnitude greater computational efficiency. The results highlight the necessity of holistic evaluation and identify these modern architectures as a highly promising direction for developing efficient and reliable surrogate models in data-driven materials science.
>
---
#### [new 085] Fast $k$-means clustering in Riemannian manifolds via Fréchet maps: Applications to large-dimensional SPD matrices
- **分类: cs.LG; cs.CV; math.DG**

- **简介: 该论文提出一种基于Fréchet映射的快速k-means聚类方法，将高维黎曼流形（如SPD矩阵）数据映射到低维欧氏空间，从而高效应用标准聚类算法，显著降低计算成本并保持高精度。**

- **链接: []()**

> **作者:** Ji Shi; Nicolas Charon; Andreas Mang; Demetrio Labate; Robert Azencott
>
> **备注:** 32 pages, 5 figures, 5 tables
>
> **摘要:** We introduce a novel, efficient framework for clustering data on high-dimensional, non-Euclidean manifolds that overcomes the computational challenges associated with standard intrinsic methods. The key innovation is the use of the $p$-Fréchet map $F^p : \mathcal{M} \to \mathbb{R}^\ell$ -- defined on a generic metric space $\mathcal{M}$ -- which embeds the manifold data into a lower-dimensional Euclidean space $\mathbb{R}^\ell$ using a set of reference points $\{r_i\}_{i=1}^\ell$, $r_i \in \mathcal{M}$. Once embedded, we can efficiently and accurately apply standard Euclidean clustering techniques such as k-means. We rigorously analyze the mathematical properties of $F^p$ in the Euclidean space and the challenging manifold of $n \times n$ symmetric positive definite matrices $\mathit{SPD}(n)$. Extensive numerical experiments using synthetic and real $\mathit{SPD}(n)$ data demonstrate significant performance gains: our method reduces runtime by up to two orders of magnitude compared to intrinsic manifold-based approaches, all while maintaining high clustering accuracy, including scenarios where existing alternative methods struggle or fail.
>
---
#### [new 086] "It's trained by non-disabled people": Evaluating How Image Quality Affects Product Captioning with VLMs
- **分类: cs.HC; cs.CV**

- **简介: 该论文评估图像质量（如模糊、构图不佳）对视觉语言模型（VLMs）生成产品描述准确性的影响，聚焦盲低视人群的信息需求，发现质量下降显著降低识别准确率，呼吁面向残障用户优化模型评估与设计。**

- **链接: []()**

> **作者:** Kapil Garg; Xinru Tang; Jimin Heo; Dwayne R. Morgan; Darren Gergle; Erik B. Sudderth; Anne Marie Piper
>
> **备注:** Paper under review
>
> **摘要:** Vision-Language Models (VLMs) are increasingly used by blind and low-vision (BLV) people to identify and understand products in their everyday lives, such as food, personal products, and household goods. Despite their prevalence, we lack an empirical understanding of how common image quality issues, like blur and misframing of items, affect the accuracy of VLM-generated captions and whether resulting captions meet BLV people's information needs. Grounded in a survey with 86 BLV people, we systematically evaluate how image quality issues affect captions generated by VLMs. We show that the best model recognizes products in images with no quality issues with 98% accuracy, but drops to 75% accuracy overall when quality issues are present, worsening considerably as issues compound. We discuss the need for model evaluations that center on disabled people's experiences throughout the process and offer concrete recommendations for HCI and ML researchers to make VLMs more reliable for BLV people.
>
---
#### [new 087] SPIDER: Scalable Physics-Informed Dexterous Retargeting
- **分类: cs.RO; cs.CV**

- **简介: SPIDER提出一种物理驱动的灵巧重定向框架，将人类运动数据转化为机器人可执行的动态可行轨迹，解决人机本体差异与数据稀缺问题，实现跨9种机器人、6类数据的高效规模化生成，显著提升策略学习效率。**

- **链接: []()**

> **作者:** Chaoyi Pan; Changhao Wang; Haozhi Qi; Zixi Liu; Homanga Bharadhwaj; Akash Sharma; Tingfan Wu; Guanya Shi; Jitendra Malik; Francois Hogan
>
> **备注:** Project website: https://jc-bao.github.io/spider-project/
>
> **摘要:** Learning dexterous and agile policy for humanoid and dexterous hand control requires large-scale demonstrations, but collecting robot-specific data is prohibitively expensive. In contrast, abundant human motion data is readily available from motion capture, videos, and virtual reality, which could help address the data scarcity problem. However, due to the embodiment gap and missing dynamic information like force and torque, these demonstrations cannot be directly executed on robots. To bridge this gap, we propose Scalable Physics-Informed DExterous Retargeting (SPIDER), a physics-based retargeting framework to transform and augment kinematic-only human demonstrations to dynamically feasible robot trajectories at scale. Our key insight is that human demonstrations should provide global task structure and objective, while large-scale physics-based sampling with curriculum-style virtual contact guidance should refine trajectories to ensure dynamical feasibility and correct contact sequences. SPIDER scales across diverse 9 humanoid/dexterous hand embodiments and 6 datasets, improving success rates by 18% compared to standard sampling, while being 10X faster than reinforcement learning (RL) baselines, and enabling the generation of a 2.4M frames dynamic-feasible robot dataset for policy learning. As a universal physics-based retargeting method, SPIDER can work with diverse quality data and generate diverse and high-quality data to enable efficient policy learning with methods like RL.
>
---
#### [new 088] Moving pattern-based modeling using a new type of interval ARX model
- **分类: stat.ME; cs.CV; eess.SY**

- **简介: 该论文提出一种新型区间ARX模型（IARX），用于处理区间数据，解决传统ARX模型无法有效建模不确定数据的问题，并将其应用于移动模式建模，在烧结过程中验证了其鲁棒性与优越性。**

- **链接: []()**

> **作者:** Changping Sun
>
> **摘要:** In this paper,firstly,to overcome the shortcoming of traditional ARX model, a new operator between an interval number and a real matrix is defined, and then it is applied to the traditional ARX model to get a new type of structure interval ARX model that can deal with interval data, which is defined as interval ARX model (IARX). Secondly,the IARX model is applied to moving pattern-based modeling. Finally,to verify the validity of the proposed modeling method,it is applied to a sintering process. The simulation results show the moving pattern-based modeling using the new type of interval ARX model is robust to variation in parameters of the model, and the performance of the modeling using the proposed IARX is superior to that of the previous work.
>
---
#### [new 089] ROI-based Deep Image Compression with Implicit Bit Allocation
- **分类: eess.IV; cs.CV; cs.IT; cs.MM**

- **简介: 该论文面向深度图像压缩任务，解决显式比特分配破坏熵模型的问题，提出隐式比特分配方法，通过MGFE模块与双解码器实现ROI自适应编码，在COCO2017上显著提升率失真性能。**

- **链接: []()**

> **作者:** Kai Hu; Han Wang; Renhe Liu; Zhilin Li; Shenghui Song; Yu Liu
>
> **备注:** 10 pages, 10 figures, journal
>
> **摘要:** Region of Interest (ROI)-based image compression has rapidly developed due to its ability to maintain high fidelity in important regions while reducing data redundancy. However, existing compression methods primarily apply masks to suppress background information before quantization. This explicit bit allocation strategy, which uses hard gating, significantly impacts the statistical distribution of the entropy model, thereby limiting the coding performance of the compression model. In response, this work proposes an efficient ROI-based deep image compression model with implicit bit allocation. To better utilize ROI masks for implicit bit allocation, this paper proposes a novel Mask-Guided Feature Enhancement (MGFE) module, comprising a Region-Adaptive Attention (RAA) block and a Frequency-Spatial Collaborative Attention (FSCA) block. This module allows for flexible bit allocation across different regions while enhancing global and local features through frequencyspatial domain collaboration. Additionally, we use dual decoders to separately reconstruct foreground and background images, enabling the coding network to optimally balance foreground enhancement and background quality preservation in a datadriven manner. To the best of our knowledge, this is the first work to utilize implicit bit allocation for high-quality regionadaptive coding. Experiments on the COCO2017 dataset show that our implicit-based image compression method significantly outperforms explicit bit allocation approaches in rate-distortion performance, achieving optimal results while maintaining satisfactory visual quality in the reconstructed background regions.
>
---
#### [new 090] SMF-VO: Direct Ego-Motion Estimation via Sparse Motion Fields
- **分类: cs.RO; cs.CV**

- **简介: 论文提出SMF-VO，一种轻量级视觉里程计方法，直接从稀疏光流估计相机瞬时速度，规避传统位姿估计与地图优化，提升实时性。适用于资源受限设备，实现超100 FPS的CPU推理。**

- **链接: []()**

> **作者:** Sangheon Yang; Yeongin Yoon; Hong Mo Jung; Jongwoo Lim
>
> **摘要:** Traditional Visual Odometry (VO) and Visual Inertial Odometry (VIO) methods rely on a 'pose-centric' paradigm, which computes absolute camera poses from the local map thus requires large-scale landmark maintenance and continuous map optimization. This approach is computationally expensive, limiting their real-time performance on resource-constrained devices. To overcome these limitations, we introduce Sparse Motion Field Visual Odometry (SMF-VO), a lightweight, 'motion-centric' framework. Our approach directly estimates instantaneous linear and angular velocity from sparse optical flow, bypassing the need for explicit pose estimation or expensive landmark tracking. We also employed a generalized 3D ray-based motion field formulation that works accurately with various camera models, including wide-field-of-view lenses. SMF-VO demonstrates superior efficiency and competitive accuracy on benchmark datasets, achieving over 100 FPS on a Raspberry Pi 5 using only a CPU. Our work establishes a scalable and efficient alternative to conventional methods, making it highly suitable for mobile robotics and wearable devices.
>
---
#### [new 091] A Finite Difference Approximation of Second Order Regularization of Neural-SDFs
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文针对神经SDF学习中曲率正则化计算昂贵的问题，提出用有限差分近似二阶导数，替代高阶自动微分，显著降低内存与训练开销，同时保持重建精度，实现高效可扩展的曲率感知SDF学习。**

- **链接: []()**

> **作者:** Haotian Yin; Aleksander Plocharski; Michal Jan Wlodarczyk; Przemyslaw Musialski
>
> **备注:** SIGGRAPH Asia Technical Communications, 6 pages, 6 figures, preprint
>
> **摘要:** We introduce a finite-difference framework for curvature regularization in neural signed distance field (SDF) learning. Existing approaches enforce curvature priors using full Hessian information obtained via second-order automatic differentiation, which is accurate but computationally expensive. Others reduced this overhead by avoiding explicit Hessian assembly, but still required higher-order differentiation. In contrast, our method replaces these operations with lightweight finite-difference stencils that approximate second derivatives using the well known Taylor expansion with a truncation error of O(h^2), and can serve as drop-in replacements for Gaussian curvature and rank-deficiency losses. Experiments demonstrate that our finite-difference variants achieve reconstruction fidelity comparable to their automatic-differentiation counterparts, while reducing GPU memory usage and training time by up to a factor of two. Additional tests on sparse, incomplete, and non-CAD data confirm that the proposed formulation is robust and general, offering an efficient and scalable alternative for curvature-aware SDF learning.
>
---
#### [new 092] BayesQ: Uncertainty-Guided Bayesian Quantization
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: BayesQ是一种不确定性引导的后训练量化框架，首次基于后验期望损失优化量化，通过高斯后验建模、码本设计与混合精度分配，在保持低比特下显著提升模型精度，适用于ResNet和BERT等模型。**

- **链接: []()**

> **作者:** Ismail Lamaakal; Chaymae Yahyati; Yassine Maleh; Khalid El Makkaoui; Ibrahim Ouahbi
>
> **摘要:** We present BayesQ, an uncertainty-guided post-training quantization framework that is the first to optimize quantization under the posterior expected loss. BayesQ fits a lightweight Gaussian posterior over weights (diagonal Laplace by default; optional K-FAC/low-rank), whitens by the posterior covariance, designs codebooks to minimize posterior-expected distortion, and allocates mixed precision via a greedy knapsack that maximizes marginal expected-loss reduction per bit under a global budget. For scalar quantizers, posterior-expected MSE yields closed-form tables; task-aware proxies are handled by short Monte Carlo on a small calibration set. An optional calibration-only distillation aligns the quantized model with the posterior predictive teacher. At matched average bits/weight of 3.0/3.5/4.0, BayesQ improves over strong PTQ baselines on ResNet-50 (ImageNet) and BERT-base (GLUE) e.g., vs. GPTQ by $+1.5/+0.7/+0.3$ top-1 percentage points on RN50 and $+1.1/+0.4/+0.2$ GLUE points on BERT, while requiring one-time preprocessing comparable to a GPTQ pass. BayesQ reframes low-bit quantization as uncertainty-aware risk minimization in a practical, post-training pipeline.
>
---
#### [new 093] Fluence Map Prediction with Deep Learning: A Transformer-based Approach
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出一种基于Swin-UNETR的深度学习方法，实现从CT与轮廓直接预测IMRT九束流强图，替代传统耗时优化，提升自动化、准确性与效率，临床剂量验证结果优异。**

- **链接: []()**

> **作者:** Ujunwa Mgboh; Rafi Sultan; Dongxiao Zhu; Joshua Kim
>
> **摘要:** Accurate fluence map prediction is essential in intensity-modulated radiation therapy (IMRT) to maximize tumor coverage while minimizing dose to healthy tissues. Conventional optimization is time-consuming and dependent on planner expertise. This study presents a deep learning framework that accelerates fluence map generation while maintaining clinical quality. An end-to-end 3D Swin-UNETR network was trained to predict nine-beam fluence maps directly from volumetric CT images and anatomical contours using 99 prostate IMRT cases (79 for training and 20 for testing). The transformer-based model employs hierarchical self-attention to capture both local anatomical structures and long-range spatial dependencies. Predicted fluence maps were imported into the Eclipse Treatment Planning System for dose recalculation, and model performance was evaluated using beam-wise fluence correlation, spatial gamma analysis, and dose-volume histogram (DVH) metrics. The proposed model achieved an average R^2 of 0.95 +/- 0.02, MAE of 0.035 +/- 0.008, and gamma passing rate of 85 +/- 10 percent (3 percent / 3 mm) on the test set, with no significant differences observed in DVH parameters between predicted and clinical plans. The Swin-UNETR framework enables fully automated, inverse-free fluence map prediction directly from anatomical inputs, enhancing spatial coherence, accuracy, and efficiency while offering a scalable and consistent solution for automated IMRT plan generation.
>
---
#### [new 094] Spatio-Temporal Data Enhanced Vision-Language Model for Traffic Scene Understanding
- **分类: cs.MM; cs.CV**

- **简介: 该论文提出ST-CLIP模型，首次将时空数据融入视觉-语言模型，提升交通场景理解（TSU）性能。通过动态时空上下文表示与多方面提示学习，解决传统方法忽略时空信息与场景关联的问题。**

- **链接: []()**

> **作者:** Jingtian Ma; Jingyuan Wang; Wayne Xin Zhao; Guoping Liu; Xiang Wen
>
> **摘要:** Nowadays, navigation and ride-sharing apps have collected numerous images with spatio-temporal data. A core technology for analyzing such images, associated with spatiotemporal information, is Traffic Scene Understanding (TSU), which aims to provide a comprehensive description of the traffic scene. Unlike traditional spatio-temporal data analysis tasks, the dependence on both spatio-temporal and visual-textual data introduces distinct challenges to TSU task. However, recent research often treats TSU as a common image understanding task, ignoring the spatio-temporal information and overlooking the interrelations between different aspects of the traffic scene. To address these issues, we propose a novel SpatioTemporal Enhanced Model based on CILP (ST-CLIP) for TSU. Our model uses the classic vision-language model, CLIP, as the backbone, and designs a Spatio-temporal Context Aware Multiaspect Prompt (SCAMP) learning method to incorporate spatiotemporal information into TSU. The prompt learning method consists of two components: A dynamic spatio-temporal context representation module that extracts representation vectors of spatio-temporal data for each traffic scene image, and a bi-level ST-aware multi-aspect prompt learning module that integrates the ST-context representation vectors into word embeddings of prompts for the CLIP model. The second module also extracts low-level visual features and image-wise high-level semantic features to exploit interactive relations among different aspects of traffic scenes. To the best of our knowledge, this is the first attempt to integrate spatio-temporal information into visionlanguage models to facilitate TSU task. Experiments on two realworld datasets demonstrate superior performance in the complex scene understanding scenarios with a few-shot learning strategy.
>
---
## 更新

#### [replaced 001] Distribution-Aware Tensor Decomposition for Compression of Convolutional Neural Networks
- **分类: cs.LG; cs.CV**

- **链接: []()**

> **作者:** Alper Kalle; Theo Rudkiewicz; Mohamed-Oumar Ouerfelli; Mohamed Tamaazousti
>
> **备注:** Corrected typos in references
>
> **摘要:** Neural networks are widely used for image-related tasks but typically demand considerable computing power. Once a network has been trained, however, its memory- and compute-footprint can be reduced by compression. In this work, we focus on compression through tensorization and low-rank representations. Whereas classical approaches search for a low-rank approximation by minimizing an isotropic norm such as the Frobenius norm in weight-space, we use data-informed norms that measure the error in function space. Concretely, we minimize the change in the layer's output distribution, which can be expressed as $\lVert (W - \widetilde{W}) Σ^{1/2}\rVert_F$ where $Σ^{1/2}$ is the square root of the covariance matrix of the layer's input and $W$, $\widetilde{W}$ are the original and compressed weights. We propose new alternating least square algorithms for the two most common tensor decompositions (Tucker-2 and CPD) that directly optimize the new norm. Unlike conventional compression pipelines, which almost always require post-compression fine-tuning, our data-informed approach often achieves competitive accuracy without any fine-tuning. We further show that the same covariance-based norm can be transferred from one dataset to another with only a minor accuracy drop, enabling compression even when the original training dataset is unavailable. Experiments on several CNN architectures (ResNet-18/50, and GoogLeNet) and datasets (ImageNet, FGVC-Aircraft, Cifar10, and Cifar100) confirm the advantages of the proposed method.
>
---
#### [replaced 002] DG-DETR: Toward Domain Generalized Detection Transformer
- **分类: cs.CV**

- **链接: []()**

> **作者:** Seongmin Hwang; Daeyoung Han; Moongu Jeon
>
> **备注:** Accepted by Pattern Recognition Letters (DOI: https://doi.org/10.1016/j.patrec.2025.11.023)
>
> **摘要:** End-to-end Transformer-based detectors (DETRs) have demonstrated strong detection performance. However, domain generalization (DG) research has primarily focused on convolutional neural network (CNN)-based detectors, while paying little attention to enhancing the robustness of DETRs. In this letter, we introduce a Domain Generalized DEtection TRansformer (DG-DETR), a simple, effective, and plug-and-play method that improves out-of-distribution (OOD) robustness for DETRs. Specifically, we propose a novel domain-agnostic query selection strategy that removes domain-induced biases from object queries via orthogonal projection onto the instance-specific style space. Additionally, we leverage a wavelet decomposition to disentangle features into domain-invariant and domain-specific components, enabling synthesis of diverse latent styles while preserving the semantic features of objects. Experimental results validate the effectiveness of DG-DETR. Our code is available at https://github.com/sminhwang/DG-DETR.
>
---
#### [replaced 003] Exploring the Adversarial Robustness of Face Forgery Detection with Decision-based Black-box Attacks
- **分类: cs.CV; cs.CY**

- **链接: []()**

> **作者:** Zhaoyu Chen; Bo Li; Kaixun Jiang; Shuang Wu; Shouhong Ding; Wenqiang Zhang
>
> **备注:** Accepted by Knowledge-Based Systems
>
> **摘要:** Face forgery generation technologies generate vivid faces, which have raised public concerns about security and privacy. Many intelligent systems, such as electronic payment and identity verification, rely on face forgery detection. Although face forgery detection has successfully distinguished fake faces, recent studies have demonstrated that face forgery detectors are very vulnerable to adversarial examples. Meanwhile, existing attacks rely on network architectures or training datasets instead of the predicted labels, which leads to a gap in attacking deployed applications. To narrow this gap, we first explore the decision-based attacks on face forgery detection. We identify challenges in directly applying existing decision-based attacks, such as perturbation initialization failure and reduced image quality. To overcome these issues, we propose cross-task perturbation to handle initialization failures by utilizing the high correlation of face features on different tasks. Additionally, inspired by the use of frequency cues in face forgery detection, we introduce the frequency decision-based attack. This attack involves adding perturbations in the frequency domain while constraining visual quality in the spatial domain. Finally, extensive experiments demonstrate that our method achieves state-of-the-art attack performance on FaceForensics++, CelebDF, and industrial APIs, with high query efficiency and guaranteed image quality. Further, the fake faces by our method can pass face forgery detection and face recognition, which exposes the security problems of face forgery detectors.
>
---
#### [replaced 004] Domain Adaptation from Generated Multi-Weather Images for Unsupervised Maritime Object Classification
- **分类: cs.CV**

- **链接: []()**

> **作者:** Dan Song; Shumeng Huo; Wenhui Li; Lanjun Wang; Chao Xue; An-An Liu
>
> **摘要:** The classification and recognition of maritime objects are crucial for enhancing maritime safety, monitoring, and intelligent sea environment prediction. However, existing unsupervised methods for maritime object classification often struggle with the long-tail data distributions in both object categories and weather conditions. In this paper, we construct a dataset named AIMO produced by large-scale generative models with diverse weather conditions and balanced object categories, and collect a dataset named RMO with real-world images where long-tail issue exists. We propose a novel domain adaptation approach that leverages AIMO (source domain) to address the problem of limited labeled data, unbalanced distribution and domain shift in RMO (target domain), enhance the generalization of source features with the Vision-Language Models such as CLIP, and propose a difficulty score for curriculum learning to optimize training process. Experimental results shows that the proposed method significantly improves the classification accuracy, particularly for samples within rare object categories and weather conditions. Datasets and codes will be publicly available at https://github.com/honoria0204/AIMO.
>
---
#### [replaced 005] DcMatch: Unsupervised Multi-Shape Matching with Dual-Level Consistency
- **分类: cs.CV**

- **链接: []()**

> **作者:** Tianwei Ye; Yong Ma; Xiaoguang Mei
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Establishing point-to-point correspondences across multiple 3D shapes is a fundamental problem in computer vision and graphics. In this paper, we introduce DcMatch, a novel unsupervised learning framework for non-rigid multi-shape matching. Unlike existing methods that learn a canonical embedding from a single shape, our approach leverages a shape graph attention network to capture the underlying manifold structure of the entire shape collection. This enables the construction of a more expressive and robust shared latent space, leading to more consistent shape-to-universe correspondences via a universe predictor. Simultaneously, we represent these correspondences in both the spatial and spectral domains and enforce their alignment in the shared universe space through a novel cycle consistency loss. This dual-level consistency fosters more accurate and coherent mappings. Extensive experiments on several challenging benchmarks demonstrate that our method consistently outperforms previous state-of-the-art approaches across diverse multi-shape matching scenarios.
>
---
#### [replaced 006] SFFR: Spatial-Frequency Feature Reconstruction for Multispectral Aerial Object Detection
- **分类: cs.CV**

- **链接: []()**

> **作者:** Xin Zuo; Yuchen Qu; Haibo Zhan; Jifeng Shen; Wankou Yang
>
> **备注:** 11 pages,8 figures, accepted by IEEE TGRS
>
> **摘要:** Recent multispectral object detection methods have primarily focused on spatial-domain feature fusion based on CNNs or Transformers, while the potential of frequency-domain feature remains underexplored. In this work, we propose a novel Spatial and Frequency Feature Reconstruction method (SFFR) method, which leverages the spatial-frequency feature representation mechanisms of the Kolmogorov-Arnold Network (KAN) to reconstruct complementary representations in both spatial and frequency domains prior to feature fusion. The core components of SFFR are the proposed Frequency Component Exchange KAN (FCEKAN) module and Multi-Scale Gaussian KAN (MSGKAN) module. The FCEKAN introduces an innovative selective frequency component exchange strategy that effectively enhances the complementarity and consistency of cross-modal features based on the frequency feature of RGB and IR images. The MSGKAN module demonstrates excellent nonlinear feature modeling capability in the spatial domain. By leveraging multi-scale Gaussian basis functions, it effectively captures the feature variations caused by scale changes at different UAV flight altitudes, significantly enhancing the model's adaptability and robustness to scale variations. It is experimentally validated that our proposed FCEKAN and MSGKAN modules are complementary and can effectively capture the frequency and spatial semantic features respectively for better feature fusion. Extensive experiments on the SeaDroneSee, DroneVehicle and DVTOD datasets demonstrate the superior performance and significant advantages of the proposed method in UAV multispectral object perception task. Code will be available at https://github.com/qchenyu1027/SFFR.
>
---
#### [replaced 007] Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations
- **分类: cs.CV**

- **链接: []()**

> **作者:** Kiran Shahi; Anup Bagale
>
> **备注:** https://github.com/kiranshahi/pneumonia-analysis
>
> **摘要:** This study proposes a weakly supervised deep learning framework for pneumonia classification and localization from chest X-rays, utilizing Grad-CAM explanations. Instead of costly pixel-level annotations, our approach uses image-level labels to generate clinically meaningful heatmaps that highlight regions affected by pneumonia. We evaluate seven pre-trained architectures and the Vision Transformer under identical training conditions, using focal loss and patient-wise splits to prevent data leakage. Experimental results suggest that all models achieved high accuracy (96-98%), with ResNet-18 and EfficientNet-B0 showing the best overall performance and MobileNet-V2 providing an efficient lightweight alternative. Grad-CAM heatmap visualizations confirm that the proposed models focus on clinically relevant lung regions, supporting the use of interpretable AI for radiological diagnostics. This work highlights the potential of weakly supervised, explainable models that enhance the transparency of pneumonia screening and clinical trust in AI-assisted screening.
>
---
#### [replaced 008] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models
- **分类: cs.CV**

- **链接: []()**

> **作者:** Xiao An; Jiaxing Sun; Zihan Gui; Wei He
>
> **备注:** Accepted by NeurIPS 2025 Track on Datasets and Benchmarks
>
> **摘要:** The rapid advancement of Large Vision-Language Models (VLMs), both general-domain models and those specifically tailored for remote sensing, has demonstrated exceptional perception and reasoning capabilities in Earth observation tasks. However, a benchmark for systematically evaluating their capabilities in this domain is still lacking. To bridge this gap, we propose CHOICE, an extensive benchmark designed to objectively evaluate the hierarchical remote sensing capabilities of VLMs. Focusing on 2 primary capability dimensions essential to remote sensing: perception and reasoning, we further categorize 6 secondary dimensions and 23 leaf tasks to ensure a well-rounded assessment coverage. CHOICE guarantees the quality of all 10,507 problems through a rigorous process of data collection from 50 globally distributed cities, question construction and quality control. The newly curated data and the format of multiple-choice questions with definitive answers allow for an objective and straightforward performance assessment. Our evaluation of 3 proprietary and 21 open-source VLMs highlights their critical limitations within this specialized context. We hope that CHOICE will serve as a valuable resource and offer deeper insights into the challenges and potential of VLMs in the field of remote sensing. We will release CHOICE at [this https URL](https://github.com/ShawnAn-WHU/CHOICE).
>
---
#### [replaced 009] TiS-TSL: Image-Label Supervised Surgical Video Stereo Matching via Time-Switchable Teacher-Student Learning
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Rui Wang; Ying Zhou; Hao Wang; Wenwei Zhang; Qiang Li; Zhiwei Wang
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Stereo matching in minimally invasive surgery (MIS) is essential for next-generation navigation and augmented reality. Yet, dense disparity supervision is nearly impossible due to anatomical constraints, typically limiting annotations to only a few image-level labels acquired before the endoscope enters deep body cavities. Teacher-Student Learning (TSL) offers a promising solution by leveraging a teacher trained on sparse labels to generate pseudo labels and associated confidence maps from abundant unlabeled surgical videos. However, existing TSL methods are confined to image-level supervision, providing only spatial confidence and lacking temporal consistency estimation. This absence of spatio-temporal reliability results in unstable disparity predictions and severe flickering artifacts across video frames. To overcome these challenges, we propose TiS-TSL, a novel time-switchable teacher-student learning framework for video stereo matching under minimal supervision. At its core is a unified model that operates in three distinct modes: Image-Prediction (IP), Forward Video-Prediction (FVP), and Backward Video-Prediction (BVP), enabling flexible temporal modeling within a single architecture. Enabled by this unified model, TiS-TSL adopts a two-stage learning strategy. The Image-to-Video (I2V) stage transfers sparse image-level knowledge to initialize temporal modeling. The subsequent Video-to-Video (V2V) stage refines temporal disparity predictions by comparing forward and backward predictions to calculate bidirectional spatio-temporal consistency. This consistency identifies unreliable regions across frames, filters noisy video-level pseudo labels, and enforces temporal coherence. Experimental results on two public datasets demonstrate that TiS-TSL exceeds other image-based state-of-the-arts by improving TEPE and EPE by at least 2.11% and 4.54%, respectively.
>
---
#### [replaced 010] Differentiable, Bit-shifting, and Scalable Quantization without training neural network from scratch
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: []()**

> **作者:** Zia Badar
>
> **摘要:** Quantization of neural networks provides benefits of inference in less compute and memory requirements. Previous work in quantization lack two important aspects which this work provides. First almost all previous work in quantization used a non-differentiable approach and for learning; the derivative is usually set manually in backpropogation which make the learning ability of algorithm questionable, our approach is not just differentiable, we also provide proof of convergence of our approach to the optimal neural network. Second previous work in shift/logrithmic quantization either have avoided activation quantization along with weight quantization or achieved less accuracy. Learning logrithmic quantize values of form $2^n$ requires the quantization function can scale to more than 1 bit quantization which is another benifit of our quantization that it provides $n$ bits quantization as well. Our approach when tested with image classification task using imagenet dataset, resnet18 and weight quantization only achieves less than 1 percent accuracy compared to full precision accuracy while taking only 15 epochs to train using shift bit quantization and achieves comparable to SOTA approaches accuracy in both weight and activation quantization using shift bit quantization in 15 training epochs with slightly higher(only higher cpu instructions) inference cost compared to 1 bit quantization(without logrithmic quantization) and not requiring any higher precision multiplication.
>
---
#### [replaced 011] LMSeg: An end-to-end geometric message-passing network on barycentric dual graphs for large-scale landscape mesh segmentation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zexian Huang; Kourosh Khoshelham; Martin Tomko
>
> **摘要:** Semantic segmentation of large-scale 3D landscape meshes is critical for geospatial analysis in complex environments, yet existing approaches face persistent challenges of scalability, end-to-end trainability, and accurate segmentation of small and irregular objects. To address these issues, we introduce the BudjBim Wall (BBW) dataset, a large-scale annotated mesh dataset derived from high-resolution LiDAR scans of the UNESCO World Heritage-listed Budj Bim cultural landscape in Victoria, Australia. The BBW dataset captures historic dry-stone wall structures that are difficult to detect under vegetation occlusion, supporting research in underrepresented cultural heritage contexts. Building on this dataset, we propose LMSeg, a deep graph message-passing network for semantic segmentation of large-scale meshes. LMSeg employs a barycentric dual graph representation of mesh faces and introduces the Geometry Aggregation+ (GA+) module, a learnable softmax-based operator that adaptively combines neighborhood features and captures high-frequency geometric variations. A hierarchical-local dual pooling integrates hierarchical and local geometric aggregation to balance global context with fine-detail preservation. Experiments on three large-scale benchmarks (SUM, H3D, and BBW) show that LMSeg achieves 75.1% mIoU on SUM, 78.4% O.A. on H3D, and 62.4% mIoU on BBW, using only 2.4M lightweight parameters. In particular, LMSeg demonstrates accurate segmentation across both urban and natural scenes-capturing small-object classes such as vehicles and high vegetation in complex city environments, while also reliably detecting dry-stone walls in dense, occluded rural landscapes. Together, the BBW dataset and LMSeg provide a practical and extensible method for advancing 3D mesh segmentation in cultural heritage, environmental monitoring, and urban applications.
>
---
#### [replaced 012] Faithful Contouring: Near-Lossless 3D Voxel Representation Free from Iso-surface
- **分类: cs.CV; cs.GR**

- **链接: []()**

> **作者:** Yihao Luo; Xianglong He; Chuanyu Pan; Yiwen Chen; Jiaqi Wu; Yangguang Li; Wanli Ouyang; Yuanming Hu; Guang Yang; ChoonHwai Yap
>
> **摘要:** Accurate and efficient voxelized representations of 3D meshes are the foundation of 3D reconstruction and generation. However, existing representations based on iso-surface heavily rely on water-tightening or rendering optimization, which inevitably compromise geometric fidelity. We propose Faithful Contouring, a sparse voxelized representation that supports 2048+ resolutions for arbitrary meshes, requiring neither converting meshes to field functions nor extracting the isosurface during remeshing. It achieves near-lossless fidelity by preserving sharpness and internal structures, even for challenging cases with complex geometry and topology. The proposed method also shows flexibility for texturing, manipulation, and editing. Beyond representation, we design a dual-mode autoencoder for Faithful Contouring, enabling scalable and detail-preserving shape reconstruction. Extensive experiments show that Faithful Contouring surpasses existing methods in accuracy and efficiency for both representation and reconstruction. For direct representation, it achieves distance errors at the $10^{-5}$ level; for mesh reconstruction, it yields a 93\% reduction in Chamfer Distance and a 35\% improvement in F-score over strong baselines, confirming superior fidelity as a representation for 3D learning tasks.
>
---
#### [replaced 013] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer
- **分类: cs.CV**

- **链接: []()**

> **作者:** Kaiyi Xu; Junchao Gong; Zhiwang Zhou; Zhangrui Li; Yuandong Pu; Yihao Liu; Ben Fei; Fenghua Ling; Wenlong Zhang; Lei Bei
>
> **备注:** Accepted by AAAI-26 Oral
>
> **摘要:** With the advancement of meteorological instruments, abundant data has become available. Current approaches are typically focus on single-variable, single-region tasks and primarily rely on deterministic modeling. This limits unified synthesis across variables and regions, overlooks cross-variable complementarity and often leads to over-smoothed results. To address above challenges, we introduce SynWeather, the first dataset designed for Unified Multi-region and Multi-variable Weather Observation Data Synthesis. SynWeather covers four representative regions: the Continental United States, Europe, East Asia, and Tropical Cyclone regions, as well as provides high-resolution observations of key weather variables, including Composite Radar Reflectivity, Hourly Precipitation, Visible Light, and Microwave Brightness Temperature. In addition, we introduce SynWeatherDiff, a general and probabilistic weather synthesis model built upon the Diffusion Transformer framework to address the over-smoothed problem. Experiments on the SynWeather dataset demonstrate the effectiveness of our network compared with both task-specific and general models.
>
---
#### [replaced 014] EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement
- **分类: eess.IV; cs.CV**

- **链接: []()**

> **作者:** Wenjie Cai; Qingguo Meng; Zhenyu Wang; Xingbo Dong; Zhe Jin
>
> **摘要:** Event cameras offer significant potential for Low-light Image Enhancement (LLIE), yet existing fusion approaches are constrained by a fundamental dilemma: early fusion struggles with modality heterogeneity, while late fusion severs crucial feature correlations. To address these limitations, we propose EvRWKV, a novel framework that enables continuous cross-modal interaction through dual-domain processing, which mainly includes a Cross-RWKV Module to capture fine-grained temporal and cross-modal dependencies, and an Event Image Spectral Fusion Enhancer (EISFE) module to perform joint adaptive frequency-domain denoising and spatial-domain alignment. This continuous interaction maintains feature consistency from low-level textures to high-level semantics. Extensive experiments on the real-world SDE and SDSD datasets demonstrate that EvRWKV significantly outperforms only image-based methods by 1.79 dB and 1.85 dB in PSNR, respectively. To further validate the practical utility of our method for downstream applications, we evaluated its impact on semantic segmentation. Experiments demonstrate that images enhanced by EvRWKV lead to a significant 35.44% improvement in mIoU.
>
---
#### [replaced 015] Knowledge-Guided Brain Tumor Segmentation via Synchronized Visual-Semantic-Topological Prior Fusion
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Mingda Zhang; Kaiwen Pan
>
> **备注:** 37 pages, 6 figures
>
> **摘要:** Background: Brain tumor segmentation requires precise delineation of hierarchical structures from multi-sequence MRI. However, existing deep learning methods primarily rely on visual features, showing insufficient discriminative power in ambiguous boundary regions. Moreover, they lack explicit integration of medical domain knowledge such as anatomical semantics and geometric topology. Methods: We propose a knowledge-guided framework, Synchronized Tri-modal Prior Fusion (STPF), that explicitly integrates three heterogeneous knowledge priors: pathology-driven differential features (T1ce-T1, T2-FLAIR, T1/T2) encoding contrast patterns; unsupervised semantic descriptions transformed into voxel-level guidance via spatialization operators; and geometric constraints extracted through persistent homology analysis. A dual-level fusion architecture dynamically allocates prior weights at the voxel level based on confidence and at the sample level through hypernetwork-generated conditional vectors. Furthermore, nested output heads structurally ensure the hierarchical constraint ET subset TC subset WT. Results: STPF achieves a mean Dice coefficient of 0.868 on the BraTS 2020 dataset, surpassing the best baseline by 2.6 percentage points (3.09% relative improvement). Notably, five-fold cross-validation yields coefficients of variation between 0.23% and 0.33%, demonstrating stable performance. Additionally, ablation experiments show that removing topological and semantic priors leads to performance degradation of 2.8% and 3.5%, respectively. Conclusions: By explicitly integrating medical knowledge priors - anatomical semantics and geometric constraints - STPF improves segmentation accuracy in ambiguous boundary regions while demonstrating generalization capability and clinical deployment potential.
>
---
#### [replaced 016] LBMamba: Locally Bi-directional Mamba
- **分类: cs.CV**

- **链接: []()**

> **作者:** Jingwei Zhang; Xi Han; Hong Qin; Mahdi S. Hosseini; Dimitris Samaras
>
> **备注:** Accepted to TMLR
>
> **摘要:** Mamba, a State Space Model (SSM) that accelerates training by recasting recurrence as a parallel scan, has recently emerged as a linearly-scaling alternative to self-attention. Because of its unidirectional nature, each state in Mamba only has information of its previous states and is blind to states after. Current Mamba-based computer-vision methods typically overcome this by augmenting Mamba's global forward scan with a global backward scan, forming a bi-directional scan to restore a full receptive field. However, this operation doubles the computational load, eroding much of the efficiency advantage that originally Mamba have. To eliminate this extra scans, we introduce LBMamba, a locally bi-directional SSM block that embeds a lightweight locally backward scan inside the forward scan and executes it in per-thread registers. Building on LBMamba, we present LBVim, a backbone that alternates scan directions every two layers to recover a global receptive field without extra backward sweeps. We validate our approach on both natural images and whole slide images (WSIs) and show that it constantly offers a superior performance-throughput trade-off. Under the same throughput, LBVim achieves 0.8% to 1.6% higher top-1 accuracy on the ImageNet-1K classification dataset, 0.6% to 2.7% higher mIoU on the ADE20K semantic segmentation dataset, 0.9% higher APb and 1.1% higher APm on the COCO detection dataset. Our method also boosts the accuracy of four SOTA Mamba models, namely VMamba, LocalVim, PlainMamba and Adventurer, by 0.5% to 3.4%. We integrate LBMamba into the SOTA pathology multiple instance learning (MIL) model, MambaMIL, which is unidirectional. Experiments on 3 public WSI classification datasets show that our method achieves a relative improvement of up to 3.06% better AUC, 3.39% better F1, 1.67% better accuracy. Our code is available at https://github.com/cvlab-stonybrook/LBMamba.
>
---
#### [replaced 017] evMLP: An Efficient Event-Driven MLP Architecture for Vision
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zhentan Zheng
>
> **摘要:** Deep neural networks have achieved remarkable results in computer vision tasks. In the early days, Convolutional Neural Networks (CNNs) were the mainstream architecture. In recent years, Vision Transformers (ViTs) have become increasingly popular. In addition, exploring applications of multi-layer perceptrons (MLPs) has provided new perspectives for research into vision model architectures. In this paper, we present evMLP accompanied by a simple event-driven local update mechanism. The proposed evMLP can independently process patches on images or feature maps via MLPs. We define changes between consecutive frames as ``events''. Under the event-driven local update mechanism, evMLP selectively processes patches where events occur. For sequential image data (e.g., video processing), this approach improves computational performance by avoiding redundant computations. Through ImageNet image classification experiments, evMLP attains accuracy competitive with state-of-the-art models. More significantly, experimental results on multiple video datasets demonstrate that evMLP reduces computational cost via its event-driven local update mechanism while maintaining output consistency with its non-event-driven baseline. The code and pre-trained models are available at https://github.com/i-evi/evMLP.
>
---
#### [replaced 018] Background Invariance Testing According to Semantic Proximity
- **分类: cs.CV; cs.LG**

- **链接: []()**

> **作者:** Zukang Liao; Min Chen
>
> **摘要:** In many applications, machine-learned (ML) models are required to hold some invariance qualities, such as rotation, size, and intensity invariance. Among these, testing for background invariance presents a significant challenge due to the vast and complex data space it encompasses. To evaluate invariance qualities, we first use a visualization-based testing framework which allows human analysts to assess and make informed decisions about the invariance properties of ML models. We show that such informative testing framework is preferred as ML models with the same global statistics (e.g., accuracy scores) can behave differently and have different visualized testing patterns. However, such human analysts might not lead to consistent decisions without a systematic sampling approach to select representative testing suites. In this work, we present a technical solution for selecting background scenes according to their semantic proximity to a target image that contains a foreground object being tested. We construct an ontology for storing knowledge about relationships among different objects using association analysis. This ontology enables an efficient and meaningful search for background scenes of different semantic distances to a target image, enabling the selection of a test suite that is both diverse and reasonable. Compared with other testing techniques, e.g., random sampling, nearest neighbors, or other sampled test suites by visual-language models (VLMs), our method achieved a superior balance between diversity and consistency of human annotations, thereby enhancing the reliability and comprehensiveness of background invariance testing.
>
---
#### [replaced 019] A Bayesian Approach to Segmentation with Noisy Labels via Spatially Correlated Distributions
- **分类: eess.IV; cs.CV; cs.LG; stat.ML**

- **链接: []()**

> **作者:** Ryu Tadokoro; Tsukasa Takagi; Shin-ichi Maeda
>
> **摘要:** In semantic segmentation, the accuracy of models heavily depends on the high-quality annotations. However, in many practical scenarios, such as medical imaging and remote sensing, obtaining true annotations is not straightforward and usually requires significant human labor. Relying on human labor often introduces annotation errors, including mislabeling, omissions, and inconsistency between annotators. In the case of remote sensing, differences in procurement time can lead to misaligned ground-truth annotations. These label errors are not independently distributed, and instead usually appear in spatially connected regions where adjacent pixels are more likely to share the same errors.To address these issues, we propose an approximate Bayesian estimation based on a probabilistic model that assumes training data include label errors, incorporating the tendency for these errors to occur with spatial correlations between adjacent pixels. However, Bayesian inference for such spatially correlated discrete variables is notoriously intractable. To overcome this fundamental challenge, we introduce a novel class of probabilistic models, which we term the ELBO-Computable Correlated Discrete Distribution (ECCD). By representing the discrete dependencies through a continuous latent Gaussian field with a Kac-Murdock-Szegö (KMS) structured covariance, our framework enables scalable and efficient variational inference for problems previously considered computationally prohibitive. Through experiments on multiple segmentation tasks, we confirm that leveraging the spatial correlation of label errors significantly improves performance. Notably, in specific tasks such as lung segmentation, the proposed method achieves performance comparable to training with clean labels under moderate noise levels. Code is available at https://github.com/pfnet-research/Bayesian_SpatialCorr.
>
---
#### [replaced 020] Prompt-OT: An Optimal Transport Regularization Paradigm for Knowledge Preservation in Vision-Language Model Adaptation
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: []()**

> **作者:** Xiwen Chen; Wenhui Zhu; Peijie Qiu; Hao Wang; Huayu Li; Haiyu Wu; Aristeidis Sotiras; Yalin Wang; Abolfazl Razi
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Vision-language models (VLMs) such as CLIP demonstrate strong performance but struggle when adapted to downstream tasks. Prompt learning has emerged as an efficient and effective strategy to adapt VLMs while preserving their pre-trained knowledge. However, existing methods still lead to overfitting and degrade zero-shot generalization. To address this challenge, we propose an optimal transport (OT)-guided prompt learning framework that mitigates forgetting by preserving the structural consistency of feature distributions between pre-trained and fine-tuned models. Unlike conventional point-wise constraints, OT naturally captures cross-instance relationships and expands the feasible parameter space for prompt tuning, allowing a better trade-off between adaptation and generalization. Our approach enforces joint constraints on both vision and text representations, ensuring a holistic feature alignment. Extensive experiments on benchmark datasets demonstrate that our simple yet effective method can outperform existing prompt learning strategies in base-to-novel generalization, cross-dataset evaluation, and domain generalization without additional augmentation or ensemble techniques. The code is available at https://github.com/ChongQingNoSubway/Prompt-OT
>
---
#### [replaced 021] Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models
- **分类: cs.CV; cs.GR**

- **链接: []()**

> **作者:** Ronghuan Wu; Wanchao Su; Jing Liao
>
> **备注:** Project Page: https://chat2svg.github.io/
>
> **摘要:** Scalable Vector Graphics (SVG) has become the de facto standard for vector graphics in digital design, offering resolution independence and precise control over individual elements. Despite their advantages, creating high-quality SVG content remains challenging, as it demands technical expertise with professional editing software and a considerable time investment to craft complex shapes. Recent text-to-SVG generation methods aim to make vector graphics creation more accessible, but they still encounter limitations in shape regularity, generalization ability, and expressiveness. To address these challenges, we introduce Chat2SVG, a hybrid framework that combines the strengths of Large Language Models (LLMs) and image diffusion models for text-to-SVG generation. Our approach first uses an LLM to generate semantically meaningful SVG templates from basic geometric primitives. Guided by image diffusion models, a dual-stage optimization pipeline refines paths in latent space and adjusts point coordinates to enhance geometric complexity. Extensive experiments show that Chat2SVG outperforms existing methods in visual fidelity, path regularity, and semantic alignment. Additionally, our system enables intuitive editing through natural language instructions, making professional vector graphics creation accessible to all users.
>
---
#### [replaced 022] Foam Segmentation in Wastewater Treatment Plants: A Federated Learning Approach with Segment Anything Model 2
- **分类: cs.CV; cs.DC; cs.LG**

- **链接: []()**

> **作者:** Mehmet Batuhan Duman; Alejandro Carnero; Cristian Martín; Daniel Garrido; Manuel Díaz
>
> **备注:** 36 pages, 14 figures, 3 tables, 4 algorithms. This work is part of the Zerovision project. Code available at: https://github.com/ertis-research/zerovision
>
> **摘要:** Foam formation in Wastewater Treatment Plants (WTPs) is a major challenge that can reduce treatment efficiency and increase costs. The ability to automatically examine changes in real-time with respect to the percentage of foam can be of great benefit to the plant. However, large amounts of labeled data are required to train standard Machine Learning (ML) models. The development of these systems is slow due to the scarcity and heterogeneity of labeled data. Additionally, the development is often hindered by the fact that different WTPs do not share their data due to privacy concerns. This paper proposes a new framework to address these challenges by combining Federated Learning (FL) with the state-of-the-art base model for image segmentation, Segment Anything Model 2 (SAM2). The FL paradigm enables collaborative model training across multiple WTPs without centralizing sensitive operational data, thereby ensuring privacy. The framework accelerates training convergence and improves segmentation performance even with limited local datasets by leveraging SAM2's strong pre-trained weights for initialization. The methodology involves fine-tuning SAM2 on distributed clients (edge nodes) using the Flower framework, where a central Fog server orchestrates the process by aggregating model weights without accessing private data. The model was trained and validated using various data collections, including real-world images captured at a WTPs in Granada, Spain, a synthetically generated foam dataset, and images from publicly available datasets to improve generalization. This research offers a practical, scalable, and privacy-aware solution for automatic foam tracking in WTPs. The findings highlight the significant potential of integrating large-scale foundational models into FL systems to solve real-world industrial challenges characterized by distributed and sensitive data.
>
---
#### [replaced 023] Rethinking Pan-sharpening: A New Training Process for Full-Resolution Generalization
- **分类: cs.CV**

- **链接: []()**

> **作者:** Ran Zhang; Xuanhua He; Li Xueheng; Ke Cao; Liu Liu; Wenbo Xu; Fang Jiabin; Yang Qize; Jie Zhang
>
> **摘要:** The field of pan-sharpening has recently seen a trend towards increasingly large and complex models, often trained on single, specific satellite datasets. This one-dataset, one-model approach leads to high computational overhead and impractical deployment. More critically, it overlooks a core challenge: poor generalization from reduced-resolution (RR) training to real-world full-resolution (FR) data. In response to this issue, we challenge this paradigm. We introduce a multiple-in-one training strategy, where a single, compact model is trained simultaneously on three distinct satellite datasets (WV2, WV3, and GF2). Our experiments show the primary benefit of this unified strategy is a significant and universal boost in FR generalization (QNR) across all tested models, directly addressing this overlooked problem. This paradigm also inherently solves the one-model-per-dataset challenge, and we support it with a highly reproducible, dependency-free codebase for true usability. Finally, we propose PanTiny, a lightweight framework designed specifically for this new, robust paradigm. We demonstrate it achieves a superior performance-to-efficiency balance, proving that principled, simple and robust design is more effective than brute-force scaling in this practical setting. Our work advocates for a community-wide shift towards creating efficient, deployable, and truly generalizable models for pan-sharpening. The code is open-sourced at https://github.com/Zirconium233/PanTiny.
>
---
#### [replaced 024] vS-Graphs: Tightly Coupling Visual SLAM and 3D Scene Graphs Exploiting Hierarchical Scene Understanding
- **分类: cs.RO; cs.CV**

- **链接: []()**

> **作者:** Ali Tourani; Saad Ejaz; Hriday Bavle; Miguel Fernandez-Cortizas; David Morilla-Cabello; Jose Luis Sanchez-Lopez; Holger Voos
>
> **备注:** 19 pages, 10 figures, 5 tables
>
> **摘要:** Current Visual Simultaneous Localization and Mapping (VSLAM) systems often struggle to create maps that are both semantically rich and easily interpretable. While incorporating semantic scene knowledge aids in building richer maps with contextual associations among mapped objects, representing them in structured formats, such as scene graphs, has not been widely addressed, resulting in complex map comprehension and limited scalability. This paper introduces vS-Graphs, a novel real-time VSLAM framework that integrates vision-based scene understanding with map reconstruction and comprehensible graph-based representation. The framework infers structural elements (i.e., rooms and floors) from detected building components (i.e., walls and ground surfaces) and incorporates them into optimizable 3D scene graphs. This solution enhances the reconstructed map's semantic richness, comprehensibility, and localization accuracy. Extensive experiments on standard benchmarks and real-world datasets demonstrate that vS-Graphs achieves an average of 15.22% accuracy gain across all tested datasets compared to state-of-the-art VSLAM methods. Furthermore, the proposed framework achieves environment-driven semantic entity detection accuracy comparable to that of precise LiDAR-based frameworks, using only visual features. The code is publicly available at https://github.com/snt-arg/visual_sgraphs and is actively being improved. Moreover, a web page containing more media and evaluation outcomes is available on https://snt-arg.github.io/vsgraphs-results/.
>
---
#### [replaced 025] Trustworthy Pedestrian Trajectory Prediction via Pattern-Aware Interaction Modeling
- **分类: cs.CV**

- **链接: []()**

> **作者:** Kaiyuan Zhai; Juan Chen; Chao Wang; Zeyi Xu; Guoming Tang
>
> **摘要:** Accurate and reliable pedestrian trajectory prediction is critical for the application of intelligent applications, yet achieving trustworthy prediction remains highly challenging due to the complexity of interactions among pedestrians. Previous methods often adopt black-box modeling of pedestrian interactions. Despite their strong performance, such opaque modeling limits the reliability of predictions in real-world deployments. To address this issue, we propose InSyn (Interaction-Synchronization Network), a novel Transformer-based model that explicitly captures diverse interaction patterns (e.g., walking in sync or conflicting) while effectively modeling direction-sensitive social behaviors. Additionally, we introduce a training strategy, termed Seq-Start of Seq (SSOS), designed to alleviate the common issue of initial-step divergence in numerical time-series prediction. Experiments on the ETH and UCY datasets demonstrate that our model not only outperforms recent black-box baselines in prediction accuracy, especially under high-density scenarios, but also provides transparent interaction modeling, as shown in the case study. Furthermore, the SSOS strategy proves to be effective in improving sequential prediction performance, reducing the initial-step prediction error by approximately 6.58%. Code is avaliable at https://github.com/rickzky1001/InSyn
>
---
#### [replaced 026] Synth-Align: Improving Trustworthiness in Vision-Language Model with Synthetic Preference Data Alignment
- **分类: cs.CV**

- **链接: []()**

> **作者:** Robert Wijaya; Ngoc-Bao Nguyen; Ngai-Man Cheung
>
> **摘要:** Large Vision-Language Models (LVLMs) have shown promising capabilities in understanding and generating information by integrating both visual and textual data. However, current models are still prone to hallucinations, which degrade the performance and greatly harm the user experience in real-world applications. Post-training alignment, particularly preference-tuning, is intended to align model outputs and behaviors (safety, instruction-following, style), ensuring robustness and adaptability to a wide range of tasks. The use of synthetic data for alignment, particularly in multimodal settings, remains under explored. Existing approaches typically use a strong model or a ground-truth model (CLIP) to determine positive and negative image-text data points. This paper proposes SynthAlign, a pipeline to generate and collect synthetic human-preference image-text data with optimal control built specifically for post-training alignment with DPO. At the core of the framework is the utilization of reward models as a proxy of human preference. A series of evaluation and benchmarking is provided to validate the effectiveness of the proposed framework and the resulting dataset. Notably, our framework enhanced LLaVA-1.5-7B achieved substantial POPE improvements: 87.6\% accuracy and 97.8\% precision, MMHal-Bench score increased from 2.36 to 3.49, and hallucination rate decreased from 51.0\% to 25.0\% (a 50.98\% relative reduction).
>
---
#### [replaced 027] SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM
- **分类: cs.RO; cs.CV**

- **链接: []()**

> **作者:** Samuel Cerezo; Gaetano Meli; Tomás Berriel Martins; Kirill Safronov; Javier Civera
>
> **备注:** 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
>
> **摘要:** Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
>
---
#### [replaced 028] Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships
- **分类: cs.CV; cs.AI; cs.IR**

- **链接: []()**

> **作者:** Futa Waseda; Antonio Tejero-de-Pablos; Isao Echizen
>
> **备注:** WACV 2026 Accepted
>
> **摘要:** Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. This work pioneers defense strategies against multimodal attacks, providing insights for building robust VLMs from both optimization and data perspectives.
>
---
#### [replaced 029] PET2Rep: Towards Vision-Language Model-Drived Automated Radiology Report Generation for Positron Emission Tomography
- **分类: eess.IV; cs.CV**

- **链接: []()**

> **作者:** Yichi Zhang; Wenbo Zhang; Zehui Ling; Gang Feng; Sisi Peng; Deshu Chen; Yuchen Liu; Hongwei Zhang; Shuqi Wang; Lanlan Li; Limei Han; Yuan Cheng; Zixin Hu; Yuan Qi; Le Xue
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Positron emission tomography (PET) is a cornerstone of modern oncologic and neurologic imaging, distinguished by its unique ability to illuminate dynamic metabolic processes that transcend the anatomical focus of traditional imaging technologies. Radiology reports are essential for clinical decision making, yet their manual creation is labor-intensive and time-consuming. Recent advancements of vision-language models (VLMs) have shown strong potential in medical applications, presenting a promising avenue for automating report generation. However, existing applications of VLMs in the medical domain have predominantly focused on structural imaging modalities, while the unique characteristics of molecular PET imaging have largely been overlooked. To bridge the gap, we introduce PET2Rep, a large-scale comprehensive benchmark for evaluation of general and medical VLMs for radiology report generation for PET images. PET2Rep stands out as the first dedicated dataset for PET report generation with metabolic information, uniquely capturing whole-body image-report pairs that cover dozens of organs to fill the critical gap in existing benchmarks and mirror real-world clinical comprehensiveness. In addition to widely recognized natural language generation metrics, we introduce a series of clinical efficacy metrics to evaluate the quality of radiotracer uptake pattern description in key organs in generated reports. We conduct a head-to-head comparison of 30 cutting-edge general-purpose and medical-specialized VLMs. The results show that the current state-of-the-art VLMs perform poorly on PET report generation task, falling considerably short of fulfilling practical needs. Moreover, we identify several key insufficiency that need to be addressed to advance the development in medical applications.
>
---
#### [replaced 030] The Visual Counter Turing Test (VCT2): A Benchmark for Evaluating AI-Generated Image Detection and the Visual AI Index (VAI)
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Nasrin Imanpour; Abhilekh Borah; Shashwat Bajpai; Subhankar Ghosh; Sainath Reddy Sankepally; Hasnat Md Abdullah; Nishoak Kosaraju; Shreyas Dixit; Ashhar Aziz; Shwetangshu Biswas; Vinija Jain; Aman Chadha; Song Wang; Amit Sheth; Amitava Das
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** The rapid progress and widespread availability of text-to-image (T2I) generative models have heightened concerns about the misuse of AI-generated visuals, particularly in the context of misinformation campaigns. Existing AI-generated image detection (AGID) methods often overfit to known generators and falter on outputs from newer or unseen models. We introduce the Visual Counter Turing Test (VCT2), a comprehensive benchmark of 166,000 images, comprising both real and synthetic prompt-image pairs produced by six state-of-the-art T2I systems: Stable Diffusion 2.1, SDXL, SD3 Medium, SD3.5 Large, DALL.E 3, and Midjourney 6. We curate two distinct subsets: COCOAI, featuring structured captions from MS COCO, and TwitterAI, containing narrative-style tweets from The New York Times. Under a unified zero-shot evaluation, we benchmark 17 leading AGID models and observe alarmingly low detection accuracy, 58% on COCOAI and 58.34% on TwitterAI. To transcend binary classification, we propose the Visual AI Index (VAI), an interpretable, prompt-agnostic realism metric based on twelve low-level visual features, enabling us to quantify and rank the perceptual quality of generated outputs with greater nuance. Correlation analysis reveals a moderate inverse relationship between VAI and detection accuracy: Pearson of -0.532 on COCOAI and -0.503 on TwitterAI, suggesting that more visually realistic images tend to be harder to detect, a trend observed consistently across generators. We release COCOAI, TwitterAI, and all codes to catalyze future advances in generalized AGID and perceptual realism assessment.
>
---
#### [replaced 031] A Simple and Effective Reinforcement Learning Method for Text-to-Image Diffusion Fine-tuning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: []()**

> **作者:** Shashank Gupta; Chaitanya Ahuja; Tsung-Yu Lin; Sreya Dutta Roy; Harrie Oosterhuis; Maarten de Rijke; Satya Narayan Shukla
>
> **摘要:** Reinforcement learning (RL)-based fine-tuning has emerged as a powerful approach for aligning diffusion models with black-box objectives. Proximal policy optimization (PPO) is the most popular choice of method for policy optimization. While effective in terms of performance, PPO is highly sensitive to hyper-parameters and involves substantial computational overhead. REINFORCE, on the other hand, mitigates some computational complexities such as high memory overhead and sensitive hyper-parameter tuning, but has suboptimal performance due to high-variance and sample inefficiency. While the variance of the REINFORCE can be reduced by sampling multiple actions per input prompt and using a baseline correction term, it still suffers from sample inefficiency. To address these challenges, we systematically analyze the efficiency-effectiveness trade-off between REINFORCE and PPO, and propose leave-one-out PPO (LOOP), a novel RL for diffusion fine-tuning method. LOOP combines variance reduction techniques from REINFORCE, such as sampling multiple actions per input prompt and a baseline correction term, with the robustness and sample efficiency of PPO via clipping and importance sampling. Our results demonstrate that LOOP effectively improves diffusion models on various black-box objectives, and achieves a better balance between computational efficiency and performance.
>
---
#### [replaced 032] CART: Compositional Auto-Regressive Transformer for Image Generation
- **分类: cs.CV; cs.LG**

- **链接: []()**

> **作者:** Siddharth Roheda; Rohit Chowdhury; Aniruddha Bala; Rohan Jaiswal
>
> **备注:** figures compressed to meet arxiv size limit
>
> **摘要:** We propose a novel Auto-Regressive (AR) image generation approach that models images as hierarchical compositions of interpretable visual layers. While AR models have achieved transformative success in language modeling, replicating this success in vision tasks remains challenging due to inherent spatial dependencies in images. Addressing the unique challenges of vision tasks, our method (CART) adds image details iteratively via semantically meaningful decompositions. We demonstrate the flexibility and generality of CART by applying it across three distinct decomposition strategies: (i) Base-Detail Decomposition (Mumford-Shah smoothness), (ii) Intrinsic Decomposition (albedo/shading), and (iii) Specularity Decomposition (diffuse/specular). This next-detail strategy outperforms traditional next-token and next-scale approaches, improving controllability, semantic interpretability, and resolution scalability. Experiments show CART generates visually compelling results while enabling structured image manipulation, opening new directions for controllable generative modeling via physically or perceptually motivated image factorization.
>
---
#### [replaced 033] Survival Modeling from Whole Slide Images via Patch-Level Graph Clustering and Mixture Density Experts
- **分类: cs.CV**

- **链接: []()**

> **作者:** Ardhendu Sekhar; Vasu Soni; Keshav Aske; Garima Jain; Pranav Jeevan; Amit Sethi
>
> **摘要:** We propose a modular framework for predicting cancer specific survival directly from whole slide pathology images (WSIs). The framework consists of four key stages designed to capture prognostic and morphological heterogeneity. First, a Quantile Based Patch Filtering module selects prognostically informative tissue regions through quantile thresholding. Second, Graph Regularized Patch Clustering models phenotype level variations using a k nearest neighbor graph that enforces spatial and morphological coherence. Third, Hierarchical Feature Aggregation learns both intra and inter cluster dependencies to represent multiscale tumor organization. Finally, an Expert Guided Mixture Density Model estimates complex survival distributions via Gaussian mixtures, enabling fine grained risk prediction. Evaluated on TCGA LUAD, TCGA KIRC, and TCGA BRCA cohorts, our model achieves concordance indices of 0.653 ,0.719 ,and 0.733 respectively, surpassing existing state of the art approaches in survival prediction from WSIs.
>
---
#### [replaced 034] FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yiming Yang; Hongbin Lin; Yueru Luo; Suzhong Fu; Chao Zheng; Xinrui Yan; Shuqi Mei; Kun Tang; Shuguang Cui; Zhen Li
>
> **摘要:** Lane segment topology reasoning provides comprehensive bird's-eye view (BEV) road scene understanding, which can serve as a key perception module in planning-oriented end-to-end autonomous driving systems. Existing lane topology reasoning methods often fall short in effectively leveraging temporal information to enhance detection and reasoning performance. Recently, stream-based temporal propagation method has demonstrated promising results by incorporating temporal cues at both the query and BEV levels. However, it remains limited by over-reliance on historical queries, vulnerability to pose estimation failures, and insufficient temporal propagation. To overcome these limitations, we propose FASTopoWM, a novel fast-slow lane segment topology reasoning framework augmented with latent world models. To reduce the impact of pose estimation failures, this unified framework enables parallel supervision of both historical and newly initialized queries, facilitating mutual reinforcement between the fast and slow systems. Furthermore, we introduce latent query and BEV world models conditioned on the action latent to propagate the state representations from past observations to the current timestep. This design substantially improves the performance of temporal perception within the slow pipeline. Extensive experiments on the OpenLane-V2 benchmark demonstrate that FASTopoWM outperforms state-of-the-art methods in both lane segment detection (37.4% v.s. 33.6% on mAP) and centerline perception (46.3% v.s. 41.5% on OLS).
>
---
#### [replaced 035] Multi-scale Cascaded Foundation Model for Whole-body Organs-at-risk Segmentation
- **分类: eess.IV; cs.CV**

- **链接: []()**

> **作者:** Rui Hao; Dayu Tan; Qiankun Li; Chunhou Zheng; Weimin Zhong; Zhigang Zeng
>
> **摘要:** Accurate segmentation of organs-at-risk (OARs) is vital for safe and precise radiotherapy and surgery. Most existing studies segment only a limited set of organs or regions, lacking a systematic treatment of OARs segmentation. We present a Multi-scale Cascaded Fusion Network (MCFNet) that aggregates features across multiple scales and resolutions. MCFNet consists of a Sharp Extraction Backbone for the downsampling path and a Flexible Connection Backbone for skip-connection fusion, strengthening representation learning in both stages. This design improves boundary localization and preserves fine structures while maintaining computational efficiency, enabling reliable performance even on low-resolution inputs. Experiments on an NVIDIA A6000 GPU using 36,131 image-mask pairs from 671 patients across 10 datasets show consistent robustness and strong cross-dataset generalization. An adaptive loss-aggregation strategy further stabilizes optimization and yields additional gains in accuracy and training efficiency. Through extensive validation, MCFNet outperforms existing methods, excelling in organ segmentation and providing reliable image-guided support for computer-aided diagnosis. Our solution aims to improve the precision and safety of radiotherapy and surgery while supporting personalized treatment, advancing modern medical technology. The code has been made available on GitHub: https://github.com/Henry991115/MCFNet.
>
---
#### [replaced 036] Redundant Queries in DETR-Based 3D Detection Methods: Unnecessary and Prunable
- **分类: cs.CV**

- **链接: []()**

> **作者:** Lizhen Xu; Zehao Wu; Wenzhao Qiu; Shanmin Pang; Xiuxiu Bai; Kuizhi Mei; Jianru Xue
>
> **备注:** AAAI 2026
>
> **摘要:** Query-based models are extensively used in 3D object detection tasks, with a wide range of pre-trained checkpoints readily available online. However, despite their popularity, these models often require an excessive number of object queries, far surpassing the actual number of objects to detect. The redundant queries result in unnecessary computational and memory costs. In this paper, we find that not all queries contribute equally -- a significant portion of queries have a much smaller impact compared to others. Based on this observation, we propose an embarrassingly simple approach called \bd{G}radually \bd{P}runing \bd{Q}ueries (GPQ), which prunes queries incrementally based on their classification scores. It is straightforward to implement in any query-based method, as it can be seamlessly integrated as a fine-tuning step using an existing checkpoint after training. With GPQ, users can easily generate multiple models with fewer queries, starting from a checkpoint with an excessive number of queries. Experiments on various advanced 3D detectors show that GPQ effectively reduces redundant queries while maintaining performance. Using our method, model inference on desktop GPUs can be accelerated by up to 1.31x. Moreover, after deployment on edge devices, it achieves up to a 67.86\% reduction in FLOPs and a 76.38\% decrease in inference time. The code will be available at https://github.com/iseri27/Gpq.
>
---
#### [replaced 037] Robust Bayesian Scene Reconstruction with Retrieval-Augmented Priors for Precise Grasping and Planning
- **分类: cs.CV; cs.RO**

- **链接: []()**

> **作者:** Herbert Wright; Weiming Zhi; Martin Matak; Matthew Johnson-Roberson; Tucker Hermans
>
> **摘要:** Constructing 3D representations of object geometry is critical for many robotics tasks, particularly manipulation problems. These representations must be built from potentially noisy partial observations. In this work, we focus on the problem of reconstructing a multi-object scene from a single RGBD image using a fixed camera. Traditional scene representation methods generally cannot infer the geometry of unobserved regions of the objects in the image. Attempts have been made to leverage deep learning to train on a dataset of known objects and representations, and then generalize to new observations. However, this can be brittle to noisy real-world observations and objects not contained in the dataset, and do not provide well-calibrated reconstruction confidences. We propose BRRP, a reconstruction method that leverages preexisting mesh datasets to build an informative prior during robust probabilistic reconstruction. We introduce the concept of a retrieval-augmented prior, where we retrieve relevant components of our prior distribution from a database of objects during inference. The resulting prior enables estimation of the geometry of occluded portions of the in-scene objects. Our method produces a distribution over object shape that can be used for reconstruction and measuring uncertainty. We evaluate our method in both simulated scenes and in the real world. We demonstrate the robustness of our method against deep learning-only approaches while being more accurate than a method without an informative prior. Through real-world experiments, we particularly highlight the capability of BRRP to enable successful dexterous manipulation in clutter.
>
---
#### [replaced 038] What's Producible May Not Be Reachable: Measuring the Steerability of Generative Models
- **分类: cs.LG; cs.AI; cs.CV; cs.HC**

- **链接: []()**

> **作者:** Keyon Vafa; Sarah Bentley; Jon Kleinberg; Sendhil Mullainathan
>
> **摘要:** How should we evaluate the quality of generative models? Many existing metrics focus on a model's producibility, i.e. the quality and breadth of outputs it can generate. However, the actual value from using a generative model stems not just from what it can produce but whether a user with a specific goal can produce an output that satisfies that goal. We refer to this property as steerability. In this paper, we first introduce a mathematical decomposition for quantifying steerability independently from producibility. Steerability is more challenging to evaluate than producibility because it requires knowing a user's goals. We address this issue by creating a benchmark task that relies on one key idea: sample an output from a generative model and ask users to reproduce it. We implement this benchmark in user studies of text-to-image and large language models. Despite the ability of these models to produce high-quality outputs, they all perform poorly on steerability. These results suggest that we need to focus on improving the steerability of generative models. We show such improvements are indeed possible: simple image-based steering mechanisms achieve more than 2x improvement on this benchmark.
>
---
#### [replaced 039] LangPose: Language-Aligned Motion for Robust 3D Human Pose Estimation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Longyun Liao; Rong Zheng
>
> **备注:** Accepted by WACV2026. Please find the supplementary material under the "Ancillary files"
>
> **摘要:** 2D-to-3D human pose lifting is an ill-posed problem due to depth ambiguity and occlusion. Existing methods relying on spatial and temporal consistency alone are insufficient to resolve these problems especially in the presence of significant occlusions or high dynamic actions. Semantic information, however, offers a complementary signal that can help disambiguate such cases. To this end, we propose LangPose, a framework that leverages action knowledge by aligning motion embeddings with text embeddings of fine-grained action labels. LangPose operates in two stages: pretraining and fine-tuning. In the pretraining stage, the model simultaneously learns to recognize actions and reconstruct 3D poses from masked and noisy 2D poses. During the fine-tuning stage, the model is further refined using real-world 3D human pose estimation datasets without action labels. Additionally, our framework incorporates masked body parts and masked time windows in motion modeling, encouraging the model to leverage semantic information when spatial and temporal consistency is unreliable. Experiments demonstrate the effectiveness of LangPose, achieving SOTA level performance in 3D pose estimation on public datasets, including Human3.6M and MPI-INF-3DHP. Specifically, LangPose achieves an MPJPE of 36.7mm on Human3.6M with detected 2D poses as input and 15.5mm on MPI-INF-3DHP with ground-truth 2D poses as input.
>
---
#### [replaced 040] MOSformer: Momentum encoder-based inter-slice fusion transformer for medical image segmentation
- **分类: eess.IV; cs.CV**

- **链接: []()**

> **作者:** De-Xing Huang; Xiao-Hu Zhou; Mei-Jiang Gui; Xiao-Liang Xie; Shi-Qi Liu; Shuang-Yi Wang; Zhen-Qiu Feng; Zhi-Chao Lai; Zeng-Guang Hou
>
> **备注:** Accepted by Biomimetic Intelligence and Robotics. 13 pages, 9 figures, 8 tables
>
> **摘要:** Medical image segmentation takes an important position in various clinical applications. 2.5D-based segmentation models bridge the computational efficiency of 2D-based models with the spatial perception capabilities of 3D-based models. However, existing 2.5D-based models primarily adopt a single encoder to extract features of target and neighborhood slices, failing to effectively fuse inter-slice information, resulting in suboptimal segmentation performance. In this study, a novel momentum encoder-based inter-slice fusion transformer (MOSformer) is proposed to overcome this issue by leveraging inter-slice information from multi-scale feature maps extracted by different encoders. Specifically, dual encoders are employed to enhance feature distinguishability among different slices. One of the encoders is moving-averaged to maintain consistent slice representations. Moreover, an inter-slice fusion transformer (IF-Trans) module is developed to fuse inter-slice multi-scale features. MOSformer is evaluated on three benchmark datasets (Synapse, ACDC, and AMOS), achieving a new state-of-the-art with 85.63%, 92.19%, and 85.43% DSC, respectively. These results demonstrate MOSformer's competitiveness in medical image segmentation.
>
---
#### [replaced 041] LayerPeeler: Autoregressive Peeling for Layer-wise Image Vectorization
- **分类: cs.CV; cs.GR**

- **链接: []()**

> **作者:** Ronghuan Wu; Wanchao Su; Jing Liao
>
> **备注:** Project Page: https://layerpeeler.github.io/
>
> **摘要:** Image vectorization is a powerful technique that converts raster images into vector graphics, enabling enhanced flexibility and interactivity. However, popular image vectorization tools struggle with occluded regions, producing incomplete or fragmented shapes that hinder editability. While recent advancements have explored optimization-based and learning-based layer-wise image vectorization, these methods face limitations in vectorization quality and flexibility. In this paper, we introduce LayerPeeler, a novel layer-wise image vectorization approach that addresses these challenges through a progressive simplification paradigm. The key to LayerPeeler's success lies in its autoregressive peeling strategy: by identifying and removing the topmost non-occluded layers while recovering underlying content, we generate vector graphics with complete paths and coherent layer structures. Our method leverages vision-language models to construct a layer graph that captures occlusion relationships among elements, enabling precise detection and description for non-occluded layers. These descriptive captions are used as editing instructions for a finetuned image diffusion model to remove the identified layers. To ensure accurate removal, we employ localized attention control that precisely guides the model to target regions while faithfully preserving the surrounding content. To support this, we contribute a large-scale dataset specifically designed for layer peeling tasks. Extensive quantitative and qualitative experiments demonstrate that LayerPeeler significantly outperforms existing techniques, producing vectorization results with superior path semantics, geometric regularity, and visual fidelity.
>
---
#### [replaced 042] Self-Supervised Implicit Attention Priors for Point Cloud Reconstruction
- **分类: cs.CV**

- **链接: []()**

> **作者:** Kyle Fogarty; Chenyue Cai; Jing Yang; Zhilin Guo; Cengiz Öztireli
>
> **备注:** Accepted at 3DV 2026
>
> **摘要:** Recovering high-quality surfaces from irregular point cloud is ill-posed unless strong geometric priors are available. We introduce an implicit self-prior approach that distills a shape-specific prior directly from the input point cloud itself and embeds it within an implicit neural representation. This is achieved by jointly training a small dictionary of learnable embeddings with an implicit distance field; at every query location, the field attends to the dictionary via cross-attention, enabling the network to capture and reuse repeating structures and long-range correlations inherent to the shape. Optimized solely with self-supervised point cloud reconstruction losses, our approach requires no external training data. To effectively integrate this learned prior while preserving input fidelity, the trained field is then sampled to extract densely distributed points and analytic normals via automatic differentiation. We integrate the resulting dense point cloud and corresponding normals into a robust implicit moving least squares (RIMLS) formulation. We show this hybrid strategy preserves fine geometric details in the input data, while leveraging the learned prior to regularize sparse regions. Experiments show that our method outperforms both classical and learning-based approaches in generating high-fidelity surfaces with superior detail preservation and robustness to common data degradations.
>
---
#### [replaced 043] Improving Adversarial Transferability with Neighbourhood Gradient Information
- **分类: cs.CV; cs.CR**

- **链接: []()**

> **作者:** Haijing Guo; Jiafeng Wang; Zhaoyu Chen; Kaixun Jiang; Lingyi Hong; Pinxue Guo; Jinglun Li; Wenqiang Zhang
>
> **备注:** Accepted by Applied Soft Computing
>
> **摘要:** Deep neural networks (DNNs) are known to be susceptible to adversarial examples, leading to significant performance degradation. In black-box attack scenarios, a considerable attack performance gap between the surrogate model and the target model persists. This work focuses on enhancing the transferability of adversarial examples to narrow this performance gap. We observe that the gradient information around the clean image, i.e., Neighbourhood Gradient Information (NGI), can offer high transferability.Based on this insight, we introduce NGI-Attack, incorporating Example Backtracking and Multiplex Mask strategies to exploit this gradient information and enhance transferability. Specifically, we first adopt Example Backtracking to accumulate Neighbourhood Gradient Information as the initial momentum term. Then, we utilize Multiplex Mask to form a multi-way attack strategy that forces the network to focus on non-discriminative regions, which can obtain richer gradient information during only a few iterations. Extensive experiments demonstrate that our approach significantly enhances adversarial transferability. Especially, when attacking numerous defense models, we achieve an average attack success rate of 95.2%. Notably, our method can seamlessly integrate with any off-the-shelf algorithm, enhancing their attack performance without incurring extra time costs.
>
---
#### [replaced 044] OpenWorldSAM: Extending SAM2 for Universal Image Segmentation with Language Prompts
- **分类: cs.CV**

- **链接: []()**

> **作者:** Shiting Xiao; Rishabh Kabra; Yuhang Li; Donghyun Lee; Joao Carreira; Priyadarshini Panda
>
> **摘要:** The ability to segment objects based on open-ended language prompts remains a critical challenge, requiring models to ground textual semantics into precise spatial masks while handling diverse and unseen categories. We present OpenWorldSAM, a framework that extends the prompt-driven Segment Anything Model v2 (SAM2) to open-vocabulary scenarios by integrating multi-modal embeddings extracted from a lightweight vision-language model (VLM). Our approach is guided by four key principles: i) Unified prompting: OpenWorldSAM supports a diverse range of prompts, including category-level and sentence-level language descriptions, providing a flexible interface for various segmentation tasks. ii) Efficiency: By freezing the pre-trained components of SAM2 and the VLM, we train only 4.5 million parameters on the COCO-stuff dataset, achieving remarkable resource efficiency. iii) Instance Awareness: We enhance the model's spatial understanding through novel positional tie-breaker embeddings and cross-attention layers, enabling effective segmentation of multiple instances. iv) Generalization: OpenWorldSAM exhibits strong zero-shot capabilities, generalizing well on unseen categories and an open vocabulary of concepts without additional training. Extensive experiments demonstrate that OpenWorldSAM achieves state-of-the-art performance in open-vocabulary semantic, instance, and panoptic segmentation across multiple benchmarks. Code is available at https://github.com/GinnyXiao/OpenWorldSAM.
>
---
#### [replaced 045] A Synthetic Benchmark for Collaborative 3D Semantic Occupancy Prediction in V2X Autonomous Driving
- **分类: cs.CV**

- **链接: []()**

> **作者:** Hanlin Wu; Pengfei Lin; Ehsan Javanmardi; Naren Bao; Bo Qian; Hao Si; Manabu Tsukada
>
> **摘要:** 3D semantic occupancy prediction is an emerging perception paradigm in autonomous driving, providing a voxel-level representation of both geometric details and semantic categories. However, the perception capability of a single vehicle is inherently constrained by occlusion, restricted sensor range, and narrow viewpoints. To address these limitations, collaborative perception enables the exchange of complementary information, thereby enhancing the completeness and accuracy. In the absence of a dedicated dataset for collaborative 3D semantic occupancy prediction, we augment an existing collaborative perception dataset by replaying it in CARLA with a high-resolution semantic voxel sensor to provide dense and comprehensive occupancy annotations. In addition, we establish benchmarks with varying prediction ranges designed to systematically assess the impact of spatial extent on collaborative prediction. We further develop a baseline model that performs inter-agent feature fusion via spatial alignment and attention aggregation. Experimental results demonstrate that our baseline model consistently outperforms single-agent models, with increasing gains observed as the prediction range expands.
>
---
#### [replaced 046] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: []()**

> **作者:** Jiwoo Shin; Byeonghu Na; Mina Kang; Wonhyeok Choi; Il-Chul Moon
>
> **备注:** Accepted at NeurIPS 2025 Workshop on Generative and Protective AI for Content Creation
>
> **摘要:** Recent advances in text-to-image generative models have raised concerns about their potential to produce harmful content when provided with malicious input text prompts. To address this issue, two main approaches have emerged: (1) fine-tuning the model to unlearn harmful concepts and (2) training-free guidance methods that leverage negative prompts. However, we observe that combining these two orthogonal approaches often leads to marginal or even degraded defense performance. This observation indicates a critical incompatibility between two paradigms, which hinders their combined effectiveness. In this work, we address this issue by proposing a conceptually simple yet experimentally robust method: replacing the negative prompts used in training-free methods with implicit negative embeddings obtained through concept inversion. Our method requires no modification to either approach and can be easily integrated into existing pipelines. We experimentally validate its effectiveness on nudity and violence benchmarks, demonstrating consistent improvements in defense success rate while preserving the core semantics of input prompts.
>
---
#### [replaced 047] FS-DAG: Few Shot Domain Adapting Graph Networks for Visually Rich Document Understanding
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: []()**

> **作者:** Amit Agarwal; Srikant Panda; Kulbhushan Pachauri
>
> **备注:** Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025), Industry Track, pages 100-114
>
> **摘要:** In this work, we propose Few Shot Domain Adapting Graph (FS-DAG), a scalable and efficient model architecture for visually rich document understanding (VRDU) in few-shot settings. FS-DAG leverages domain-specific and language/vision specific backbones within a modular framework to adapt to diverse document types with minimal data. The model is robust to practical challenges such as handling OCR errors, misspellings, and domain shifts, which are critical in real-world deployments. FS-DAG is highly performant with less than 90M parameters, making it well-suited for complex real-world applications for Information Extraction (IE) tasks where computational resources are limited. We demonstrate FS-DAG's capability through extensive experiments for information extraction task, showing significant improvements in convergence speed and performance compared to state-of-the-art methods. Additionally, this work highlights the ongoing progress in developing smaller, more efficient models that do not compromise on performance. Code : https://github.com/oracle-samples/fs-dag
>
---
#### [replaced 048] LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: []()**

> **作者:** Randall Balestriero; Yann LeCun
>
> **摘要:** Learning manipulable representations of the world and its dynamics is central to AI. Joint-Embedding Predictive Architectures (JEPAs) offer a promising blueprint, but lack of practical guidance and theory has led to ad-hoc R&D. We present a comprehensive theory of JEPAs and instantiate it in {\bf LeJEPA}, a lean, scalable, and theoretically grounded training objective. First, we identify the isotropic Gaussian as the optimal distribution that JEPAs' embeddings should follow to minimize downstream prediction risk. Second, we introduce a novel objective--{\bf Sketched Isotropic Gaussian Regularization} (SIGReg)--to constrain embeddings to reach that ideal distribution. Combining the JEPA predictive loss with SIGReg yields LeJEPA with numerous theoretical and practical benefits: (i) single trade-off hyperparameter, (ii) linear time and memory complexity, (iii) stability across hyper-parameters, architectures (ResNets, ViTs, ConvNets) and domains, (iv) heuristics-free, e.g., no stop-gradient, no teacher-student, no hyper-parameter schedulers, and (v) distributed training-friendly implementation requiring only $\approx$50 lines of code. Our empirical validation covers 10+ datasets, 60+ architectures, all with varying scales and domains. As an example, using imagenet-1k for pretraining and linear evaluation with frozen backbone, LeJEPA reaches 79\% with a ViT-H/14. We hope that the simplicity and theory-friendly ecosystem offered by LeJEPA will reestablish self-supervised pre-training as a core pillar of AI research (\href{https://github.com/rbalestr-lab/lejepa}{GitHub repo}).
>
---
#### [replaced 049] Procedure Learning via Regularized Gromov-Wasserstein Optimal Transport
- **分类: cs.CV**

- **链接: []()**

> **作者:** Syed Ahmed Mahmood; Ali Shah Ali; Umer Ahmed; Fawad Javed Fateh; M. Zeeshan Zia; Quoc-Huy Tran
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** We study self-supervised procedure learning, which discovers key steps and their order from a set of unlabeled videos. Previous methods typically learn frame-to-frame correspondences between videos before determining key steps and their order. However, their performance often suffers from order variations, background/redundant frames, and repeated actions. To overcome these challenges, we propose a self-supervised framework, which utilizes a fused Gromov-Wasserstein optimal transport with a structural prior for frame-to-frame mapping. However, optimizing only for the above temporal alignment may lead to degenerate solutions, where all frames are mapped to a small cluster in the embedding space and thus every video is assigned to just one key step. To address that issue, we integrate a contrastive regularization, which maps different frames to various points, avoiding trivial solutions. Finally, extensive experiments on egocentric and third-person benchmarks demonstrate our superior performance over prior works, including OPEL which relies on a classical Kantorovich optimal transport with an optimality prior.
>
---
#### [replaced 050] Raw Data Matters: Enhancing Prompt Tuning by Internal Augmentation on Vision-Language Models
- **分类: cs.CV**

- **链接: []()**

> **作者:** Haoyang Li; Liang Wang; Chao Wang; Siyu Zhou; Jing Jiang; Yan Peng; Guodong Long
>
> **备注:** 16 pages, 6 figures, 15 tables
>
> **摘要:** For CLIP-based prompt tuning, introducing more data as additional knowledge for enhancing fine-tuning process is proved to be an effective approach. Existing data amplification strategies for prompt tuning typically rely on external knowledge (e.g., large language models or pre-structured knowledge bases), resulting in higher costs for data collection and processing, while generally ignoring further utilization of features in image modality. To address this, we propose Augmentation-driven Prompt Tuning (AugPT), a self-contained distillation-based prompt tuning approach using only internal augmentation on raw dataset to better exploit known features. Specifically, AugPT employs self-supervised augmentation on unlabeled images in the training set, and introduces a novel gating mechanism based on consensus test, reusing the pre-trained prompt tuning backbone model to spontaneously filter noisy samples, further enhancing the quality of augmented views. Extensive experiments validate that AugPT simultaneously enhances model performance and generalization capability without using appended external knowledge. The code of AugPT is available at: https://github.com/JREion/AugPT .
>
---
#### [replaced 051] Bridging Synthetic and Real-World Domains: A Human-in-the-Loop Weakly-Supervised Framework for Industrial Toxic Emission Segmentation
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Yida Tao; Yen-Chia Hsu
>
> **摘要:** Industrial smoke segmentation is critical for air-quality monitoring and environmental protection but is often hampered by the high cost and scarcity of pixel-level annotations in real-world settings. We introduce CEDANet, a human-in-the-loop, class-aware domain adaptation framework that uniquely integrates weak, citizen-provided video-level labels with adversarial feature alignment. Specifically, we refine pseudo-labels generated by a source-trained segmentation model using citizen votes, and employ class-specific domain discriminators to transfer rich source-domain representations to the industrial domain. Comprehensive experiments on SMOKE5K and custom IJmond datasets demonstrate that CEDANet achieves an F1-score of 0.414 and a smoke-class IoU of 0.261 with citizen feedback, vastly outperforming the baseline model, which scored 0.083 and 0.043 respectively. This represents a five-fold increase in F1-score and a six-fold increase in smoke-class IoU. Notably, CEDANet with citizen-constrained pseudo-labels achieves performance comparable to the same architecture trained on limited 100 fully annotated images with F1-score of 0.418 and IoU of 0.264, demonstrating its ability to reach small-sampled fully supervised-level accuracy without target-domain annotations. Our research validates the scalability and cost-efficiency of combining citizen science with weakly supervised domain adaptation, offering a practical solution for complex, data-scarce environmental monitoring applications.
>
---
#### [replaced 052] DiffRegCD: Integrated Registration and Change Detection with Diffusion Features
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Seyedehanita Madani; Rama Chellappa; Vishal M. Patel
>
> **备注:** 10 pages, 8 figures. Accepted to WACV 2026
>
> **摘要:** Change detection (CD) is fundamental to computer vision and remote sensing, supporting applications in environmental monitoring, disaster response, and urban development. Most CD models assume co-registered inputs, yet real-world imagery often exhibits parallax, viewpoint shifts, and long temporal gaps that cause severe misalignment. Traditional two stage methods that first register and then detect, as well as recent joint frameworks (e.g., BiFA, ChangeRD), still struggle under large displacements, relying on regression only flow, global homographies, or synthetic perturbations. We present DiffRegCD, an integrated framework that unifies dense registration and change detection in a single model. DiffRegCD reformulates correspondence estimation as a Gaussian smoothed classification task, achieving sub-pixel accuracy and stable training. It leverages frozen multi-scale features from a pretrained denoising diffusion model, ensuring robustness to illumination and viewpoint variation. Supervision is provided through controlled affine perturbations applied to standard CD datasets, yielding paired ground truth for both flow and change detection without pseudo labels. Extensive experiments on aerial (LEVIR-CD, DSIFN-CD, WHU-CD, SYSU-CD) and ground level (VL-CMU-CD) datasets show that DiffRegCD consistently surpasses recent baselines and remains reliable under wide temporal and geometric variation, establishing diffusion features and classification based correspondence as a strong foundation for unified change detection.
>
---
#### [replaced 053] Geo-Registration of Terrestrial LiDAR Point Clouds with Satellite Images without GNSS
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Xinyu Wang; Muhammad Ibrahim; Haitian Wang; Atif Mansoor; Xiuping Jia; Ajmal Mian
>
> **备注:** Submitted to IEEE Transactions on Geoscience & Remote Sensing. Under reviewing now
>
> **摘要:** Accurate geo-registration of LiDAR point clouds remains a significant challenge in urban environments where Global Navigation Satellite System (GNSS) signals are denied or degraded. Existing methods typically rely on real-time GNSS and Inertial Measurement Unit (IMU) data, which require pre-calibration and assume stable signals. However, this assumption often fails in dense cities, resulting in localization errors. To address this, we propose a structured geo-registration method that accurately aligns LiDAR point clouds with satellite images, enabling frame-wise geo-registration and city-scale 3D reconstruction without prior localization. Our method uses a pre-trained Point Transformer to segment road points, then extracts road skeletons and intersections from the point cloud and the satellite image. Global alignment is achieved through rigid transformation using corresponding intersection points, followed by local non-rigid refinement with radial basis function (RBF) interpolation. Elevation discrepancies are corrected using terrain data from the Shuttle Radar Topography Mission (SRTM). To evaluate geo-registration accuracy, we measure the absolute distances between the roads extracted from the two modalities. Our method is validated on the KITTI benchmark and a newly collected dataset of Perth, Western Australia. On KITTI, our method achieves a mean planimetric alignment error of 0.69m, representing 50% improvement over the raw KITTI data. On Perth dataset, it achieves a mean planimetric error of 2.17m from GNSS values extracted from Google Maps, corresponding to 57.4% improvement over rigid alignment. Elevation correlation improved by 30.5% (KITTI) and 55.8% (Perth). A demonstration video is available at: https://youtu.be/0wkACAB-O6E.
>
---
#### [replaced 054] Mapping Hidden Heritage: Self-supervised Pre-training for Archaeological Stone Wall Mapping in Historic Landscapes Using High-Resolution DEM Derivatives
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zexian Huang; Mashnoon Islam; Brian Armstrong; Billy Bell; Kourosh Khoshelham; Martin Tomko
>
> **摘要:** Historic dry-stone walls hold significant cultural and environmental importance, serving as historical markers and contributing to ecosystem preservation and wildfire management during dry seasons in Australia. However, many of these stone structures in remote or vegetated landscapes remain undocumented due to limited accessibility and the high cost of manual mapping. Deep learning-based segmentation offers a scalable approach for automated mapping of such features, but challenges remain: the visual occlusion of low-lying walls by dense vegetation and the scarcity of labeled training data. This study presents DINO-CV, a self-supervised cross-view pre-training framework based on knowledge distillation, designed for accurate mapping of dry-stone walls using high-resolution Digital Elevation Models (DEMs) derived from airborne LiDAR. By learning invariant structural representations across multiple DEM-derived views, specifically Multi-directional Hillshade (MHS) and Visualization for Archaeological Topography (VAT), DINO-CV addresses both occlusion and data scarcity challenges. Applied to the Budj Bim Cultural Landscape (Victoria, Australia), a UNESCO World Heritage site, the approach achieves a mean Intersection over Union (mIoU) of 68.6% on test areas and maintains 63.8% mIoU when fine-tuned with only 10% labeled data. These results demonstrate the potential of self-supervised learning on high-resolution DEM derivatives for large-scale, automated mapping of cultural heritage features in complex and vegetated environments. Beyond archaeology, this approach offers a scalable solution for environmental monitoring and heritage preservation across inaccessible or environmentally sensitive regions.
>
---
#### [replaced 055] UltraSam: A Foundation Model for Ultrasound using Large Open-Access Segmentation Datasets
- **分类: eess.IV; cs.CV**

- **链接: []()**

> **作者:** Adrien Meyer; Aditya Murali; Farahdiba Zarin; Didier Mutter; Nicolas Padoy
>
> **备注:** 7 pages, 3 figures, 3 tables
>
> **摘要:** Purpose: Automated ultrasound image analysis is challenging due to anatomical complexity and limited annotated data. To tackle this, we take a data-centric approach, assembling the largest public ultrasound segmentation dataset and training a versatile visual foundation model tailored for ultrasound. Methods: We compile US-43d, a large-scale collection of 43 open-access ultrasound datasets with over 280,000 images and segmentation masks for more than 50 anatomical structures. We then introduce UltraSam, an adaptation of the Segment Anything Model (SAM) that is trained on US-43d and supports both point- and box-prompts. Finally, we introduce a new use case for SAM-style models by using UltraSam as a model initialization that can be fine-tuned for various downstream analysis tasks, demonstrating UltraSam's foundational capabilities. Results: UltraSam achieves vastly improved performance over existing SAM-style models for prompt-based segmentation on three diverse public datasets. Moreover, an UltraSam-initialized Vision Transformer surpasses ImageNet-, SAM-, and MedSAM-initialized models in various downstream segmentation and classification tasks, highlighting UltraSam's effectiveness as a foundation model. Conclusion: We compile US-43d, a large-scale unified ultrasound dataset, and introduce UltraSam, a powerful multi-purpose SAM-style model for ultrasound images. We release our code and pretrained models at https://github.com/CAMMA-public/UltraSam and invite the community to further this effort by contributing high-quality datasets.
>
---
#### [replaced 056] MAUGIF: Mechanism-Aware Unsupervised General Image Fusion via Dual Cross-Image Autoencoders
- **分类: cs.CV**

- **链接: []()**

> **作者:** Kunjing Yang; Zhiwei Wang; Minru Bai
>
> **摘要:** Image fusion aims to integrate structural and complementary information from multi-source images. However, existing fusion methods are often either highly task-specific, or general frameworks that apply uniform strategies across diverse tasks, ignoring their distinct fusion mechanisms. To address this issue, we propose a mechanism-aware unsupervised general image fusion (MAUGIF) method based on dual cross-image autoencoders. Initially, we introduce a classification of additive and multiplicative fusion according to the inherent mechanisms of different fusion tasks. Then, dual encoders map source images into a shared latent space, capturing common content while isolating modality-specific details. During the decoding phase, dual decoders act as feature injectors, selectively reintegrating the unique characteristics of each modality into the shared content for reconstruction. The modality-specific features are injected into the source image in the fusion process, generating the fused image that integrates information from both modalities. The architecture of decoders varies according to their fusion mechanisms, enhancing both performance and interpretability. Extensive experiments are conducted on diverse fusion tasks to validate the effectiveness and generalization ability of our method. The code is available at https://anonymous.4open.science/r/MAUGIF.
>
---
#### [replaced 057] The Power of Many: Synergistic Unification of Diverse Augmentations for Efficient Adversarial Robustness
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Wang Yu-Hang; Shiwei Li; Jianxiang Liao; Li Bohan; Jian Liu; Wenfei Yin
>
> **备注:** Due to flaws in the theoretical derivation, the paper is being withdrawn and rewritten
>
> **摘要:** Adversarial perturbations pose a significant threat to deep learning models. Adversarial Training (AT), the predominant defense method, faces challenges of high computational costs and a degradation in standard performance. While data augmentation offers an alternative path, existing techniques either yield limited robustness gains or incur substantial training overhead. Therefore, developing a defense mechanism that is both highly efficient and strongly robust is of paramount importance.In this work, we first conduct a systematic analysis of existing augmentation techniques, revealing that the synergy among diverse strategies -- rather than any single method -- is crucial for enhancing robustness. Based on this insight, we propose the Universal Adversarial Augmenter (UAA) framework, which is characterized by its plug-and-play nature and training efficiency. UAA decouples the expensive perturbation generation process from model training by pre-computing a universal transformation offline, which is then used to efficiently generate unique adversarial perturbations for each sample during training.Extensive experiments conducted on multiple benchmarks validate the effectiveness of UAA. The results demonstrate that UAA establishes a new state-of-the-art (SOTA) for data-augmentation-based adversarial defense strategies , without requiring the online generation of adversarial examples during training. This framework provides a practical and efficient pathway for building robust models,Our code is available in the supplementary materials.
>
---
#### [replaced 058] Mapping Reduced Accessibility to WASH Facilities in Rohingya Refugee Camps with Sub-Meter Imagery
- **分类: cs.CV**

- **链接: []()**

> **作者:** Kyeongjin Ahn; YongHun Suh; Sungwon Han; Jeasurk Yang; Hannes Taubenböck; Meeyoung Cha
>
> **备注:** 23 pages, 13 figures, 2 tables
>
> **摘要:** Access to Water, Sanitation, and Hygiene (WASH) services remains a major public health concern in refugee camps. This study introduces a remote sensing-driven framework to quantify WASH accessibility-specifically to water pumps, latrines, and bathing cubicles-in the Rohingya camps of Cox's Bazar, one of the world's most densely populated displacement settings. Detecting refugee shelters in such emergent camps presents substantial challenges, primarily due to their dense spatial configuration and irregular geometric patterns. Using sub-meter satellite images, we develop a semi-supervised segmentation framework that achieves an F1-score of 76.4% in detecting individual refugee shelters. Applying the framework across multi-year data reveals declining WASH accessibility, driven by rapid refugee population growth and reduced facility availability, rising from 25 people per facility in 2022 to 29.4 in 2025. Gender-disaggregated analysis further shows that women and girls experience reduced accessibility, in scenarios with inadequate safety-related segregation in WASH facilities. These findings suggest the importance of demand-responsive allocation strategies that can identify areas with under-served populations-such as women and girls-and ensure that limited infrastructure serves the greatest number of people in settings with fixed or shrinking budgets. We also discuss the value of high-resolution remote sensing and machine learning to detect inequality and inform equitable resource planning in complex humanitarian environments.
>
---
#### [replaced 059] VPN: Visual Prompt Navigation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Shuo Feng; Zihan Wang; Yuchen Li; Rui Kong; Hengyi Cai; Shuaiqiang Wang; Gim Hee Lee; Piji Li; Shuqiang Jiang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** While natural language is commonly used to guide embodied agents, the inherent ambiguity and verbosity of language often hinder the effectiveness of language-guided navigation in complex environments. To this end, we propose Visual Prompt Navigation (VPN), a novel paradigm that guides agents to navigate using only user-provided visual prompts within 2D top-view maps. This visual prompt primarily focuses on marking the visual navigation trajectory on a top-down view of a scene, offering intuitive and spatially grounded guidance without relying on language instructions. It is more friendly for non-expert users and reduces interpretive ambiguity. We build VPN tasks in both discrete and continuous navigation settings, constructing two new datasets, R2R-VP and R2R-CE-VP, by extending existing R2R and R2R-CE episodes with corresponding visual prompts. Furthermore, we introduce VPNet, a dedicated baseline network to handle the VPN tasks, with two data augmentation strategies: view-level augmentation (altering initial headings and prompt orientations) and trajectory-level augmentation (incorporating diverse trajectories from large-scale 3D scenes), to enhance navigation performance. Extensive experiments evaluate how visual prompt forms, top-view map formats, and data augmentation strategies affect the performance of visual prompt navigation. The code is available at https://github.com/farlit/VPN.
>
---
#### [replaced 060] A Mixture-of-Experts Framework with Log-Logistic Components for Survival Analysis on Histopathology Images
- **分类: cs.CV**

- **链接: []()**

> **作者:** Ardhendu Sekhar; Vasu Soni; Keshav Aske; Shivam Madnoorkar; Pranav Jeevan; Amit Sethi
>
> **摘要:** We propose a modular framework for predicting cancer specific survival from whole slide pathology images (WSIs). The method integrates four components: (i) Quantile Gated Patch Selection via quantile based thresholding to isolate prognostically informative tissue regions; (ii) Graph Guided Clustering using a k nearest neighbor graph to capture phenotype level heterogeneity through spatial and morphological coherence; (iii) Hierarchical Context Attention to learn intra and inter cluster interactions; and (iv) an Expert Driven Mixture of Log logistics framework to estimate complex survival distributions using Log logistics distributions. The model attains a concordance index of 0.644 on TCGA LUAD, 0.751 on TCGA KIRC, and 0.752 on TCGA BRCA respectively, outperforming existing state of the art approaches.
>
---
#### [replaced 061] EAGLE: Episodic Appearance- and Geometry-aware Memory for Unified 2D-3D Visual Query Localization in Egocentric Vision
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yifei Cao; Yu Liu; Guolong Wang; Zhu Liu; Kai Wang; Xianjie Zhang; Jizhe Yu; Xun Tu
>
> **备注:** 13 Pages, accepted by AAAI-2026
>
> **摘要:** Egocentric visual query localization is vital for embodied AI and VR/AR, yet remains challenging due to camera motion, viewpoint changes, and appearance variations. We present EAGLE, a novel framework that leverages episodic appearance- and geometry-aware memory to achieve unified 2D-3D visual query localization in egocentric vision. Inspired by avian memory consolidation, EAGLE synergistically integrates segmentation guided by an appearance-aware meta-learning memory (AMM), with tracking driven by a geometry-aware localization memory (GLM). This memory consolidation mechanism, through structured appearance and geometry memory banks, stores high-confidence retrieval samples, effectively supporting both long- and short-term modeling of target appearance variations. This enables precise contour delineation with robust spatial discrimination, leading to significantly improved retrieval accuracy. Furthermore, by integrating the VQL-2D output with a visual geometry grounded Transformer (VGGT), we achieve a efficient unification of 2D and 3D tasks, enabling rapid and accurate back-projection into 3D space. Our method achieves state-ofthe-art performance on the Ego4D-VQ benchmark.
>
---
#### [replaced 062] Learning More by Seeing Less: Structure First Learning for Efficient, Transferable, and Human-Aligned Vision
- **分类: cs.CV**

- **链接: []()**

> **作者:** Tianqin Li; George Liu; Tai Sing Lee
>
> **摘要:** Despite remarkable progress in computer vision, modern recognition systems remain fundamentally limited by their dependence on rich, redundant visual inputs. In contrast, humans can effortlessly understand sparse, minimal representations like line drawings, suggesting that structure, rather than appearance, underlies efficient visual understanding. In this work, we propose a novel structure-first learning paradigm that uses line drawings as an initial training modality to induce more compact and generalizable visual representations. We demonstrate that models trained with this approach develop a stronger shape bias, more focused attention, and greater data efficiency across classification, detection, and segmentation tasks. Notably, these models also exhibit lower intrinsic dimensionality, requiring significantly fewer principal components to capture representational variance, which mirrors observations of low-dimensional, efficient representations in the human brain. Beyond performance improvements, structure-first learning produces more compressible representations, enabling better distillation into lightweight student models. Students distilled from teachers trained on line drawings consistently outperform those trained from color-supervised teachers, highlighting the benefits of structurally compact knowledge. Together, our results support the view that structure-first visual learning fosters efficiency, generalization, and human-aligned inductive biases, offering a simple yet powerful strategy for building more robust and adaptable vision systems.
>
---
#### [replaced 063] GAITGen: Disentangled Motion-Pathology Impaired Gait Generative Model -- Bringing Motion Generation to the Clinical Domain
- **分类: cs.CV**

- **链接: []()**

> **作者:** Vida Adeli; Soroush Mehraban; Majid Mirmehdi; Alan Whone; Benjamin Filtjens; Amirhossein Dadashzadeh; Alfonso Fasano; Andrea Iaboni; Babak Taati
>
> **备注:** Accepted at the IEEE/CVF winter conference on applications of computer vision (WACV 2026)
>
> **摘要:** Gait analysis is crucial for the diagnosis and monitoring of movement disorders like Parkinson's Disease. While computer vision models have shown potential for objectively evaluating parkinsonian gait, their effectiveness is limited by scarce clinical datasets and the challenge of collecting large and well-labelled data, impacting model accuracy and risk of bias. To address these gaps, we propose GAITGen, a novel framework that generates realistic gait sequences conditioned on specified pathology severity levels. GAITGen employs a Conditional Residual Vector Quantized Variational Autoencoder to learn disentangled representations of motion dynamics and pathology-specific factors, coupled with Mask and Residual Transformers for conditioned sequence generation. GAITGen generates realistic, diverse gait sequences across severity levels, enriching datasets and enabling large-scale model training in parkinsonian gait analysis. Experiments on our new PD-GaM (real) dataset demonstrate that GAITGen outperforms adapted state-of-the-art models in both reconstruction fidelity and generation quality, accurately capturing critical pathology-specific gait features. A clinical user study confirms the realism and clinical relevance of our generated sequences. Moreover, incorporating GAITGen-generated data into downstream tasks improves parkinsonian gait severity estimation, highlighting its potential for advancing clinical gait analysis.
>
---
#### [replaced 064] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yuetong Liu; Yunqiu Xu; Yang Wei; Xiuli Bi; Bin Xiao
>
> **备注:** 18 pages, 20 figures, Accepted by AAAI 2026
>
> **摘要:** Restoring nighttime images affected by multiple adverse weather conditions is a practical yet under-explored research problem, as multiple weather conditions often coexist in the real world alongside various lighting effects at night. This paper first explores the challenging multi-weather nighttime image restoration task, where various types of weather degradations are intertwined with flare effects. To support the research, we contribute the AllWeatherNight dataset, featuring large-scale high-quality nighttime images with diverse compositional degradations, synthesized using our introduced illumination-aware degradation generation. Moreover, we present ClearNight, a unified nighttime image restoration framework, which effectively removes complex degradations in one go. Specifically, ClearNight extracts Retinex-based dual priors and explicitly guides the network to focus on uneven illumination regions and intrinsic texture contents respectively, thereby enhancing restoration effectiveness in nighttime scenarios. In order to better represent the common and unique characters of multiple weather degradations, we introduce a weather-aware dynamic specific-commonality collaboration method, which identifies weather degradations and adaptively selects optimal candidate units associated with specific weather types. Our ClearNight achieves state-of-the-art performance on both synthetic and real-world images. Comprehensive ablation experiments validate the necessity of AllWeatherNight dataset as well as the effectiveness of ClearNight. Project Page: https://henlyta.github.io/ClearNight/
>
---
#### [replaced 065] HumanDreamer-X: Photorealistic Single-image Human Avatars Reconstruction via Gaussian Restoration
- **分类: cs.CV**

- **链接: []()**

> **作者:** Boyuan Wang; Runqi Ouyang; Xiaofeng Wang; Zheng Zhu; Guosheng Zhao; Chaojun Ni; Xiaopei Zhang; Guan Huang; Yijie Ren; Lihong Liu; Xingang Wang
>
> **备注:** Project Page: https://humandreamer-x.github.io/
>
> **摘要:** Single-image human reconstruction is vital for digital human modeling applications but remains an extremely challenging task. Current approaches rely on generative models to synthesize multi-view images for subsequent 3D reconstruction and animation. However, directly generating multiple views from a single human image suffers from geometric inconsistencies, resulting in issues like fragmented or blurred limbs in the reconstructed models. To tackle these limitations, we introduce \textbf{HumanDreamer-X}, a novel framework that integrates multi-view human generation and reconstruction into a unified pipeline, which significantly enhances the geometric consistency and visual fidelity of the reconstructed 3D models. In this framework, 3D Gaussian Splatting serves as an explicit 3D representation to provide initial geometry and appearance priority. Building upon this foundation, \textbf{HumanFixer} is trained to restore 3DGS renderings, which guarantee photorealistic results. Furthermore, we delve into the inherent challenges associated with attention mechanisms in multi-view human generation, and propose an attention modulation strategy that effectively enhances geometric details identity consistency across multi-view. Experimental results demonstrate that our approach markedly improves generation and reconstruction PSNR quality metrics by 16.45% and 12.65%, respectively, achieving a PSNR of up to 25.62 dB, while also showing generalization capabilities on in-the-wild data and applicability to various human reconstruction backbone models.
>
---
#### [replaced 066] Surgical AI Copilot: Energy-Based Fourier Gradient Low-Rank Adaptation for Surgical LLM Agent Reasoning and Planning
- **分类: cs.CV**

- **链接: []()**

> **作者:** Jiayuan Huang; Runlong He; Danyal Zaman Khan; Evangelos B. Mazomenos; Danail Stoyanov; Hani Marcus; Linzhe Jiang; Matthew J Clarkson; Mobarak I. Hoque
>
> **备注:** 11 pages
>
> **摘要:** Image-guided surgery demands adaptive, real-time decision support, yet static AI models struggle with structured task planning and providing interactive guidance. Large language models (LLMs)-powered agents offer a promising solution by enabling dynamic task planning and predictive decision support. Despite recent advances, the absence of surgical agent datasets and robust parameter-efficient fine-tuning techniques limits the development of LLM agents capable of complex intraoperative reasoning. In this paper, we introduce Surgical AI Copilot, an LLM agent for image-guided pituitary surgery, capable of conversation, planning, and task execution in response to queries involving tasks such as MRI tumor segmentation, endoscope anatomy segmentation, overlaying preoperative imaging with intraoperative views, instrument tracking, and surgical visual question answering (VQA). To enable structured agent planning, we develop the PitAgent dataset, a surgical context-aware planning dataset covering surgical tasks like workflow analysis, instrument localization, anatomical segmentation, and query-based reasoning. Additionally, we propose DEFT-GaLore, a Deterministic Energy-based Fourier Transform (DEFT) gradient projection technique for efficient low-rank adaptation of recent LLMs (e.g., LLaMA 3.2, Qwen 2.5), enabling their use as surgical agent planners. We extensively validate our agent's performance and the proposed adaptation technique against other state-of-the-art low-rank adaptation methods on agent planning and prompt generation tasks, including a zero-shot surgical VQA benchmark, demonstrating the significant potential for truly efficient and scalable surgical LLM agents in real-time operative settings.
>
---
#### [replaced 067] Sim-to-Real: An Unsupervised Noise Layer for Screen-Camera Watermarking Robustness
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yufeng Wu; Xin Liao; Baowei Wang; Han Fang; Xiaoshuai Wu; Mingyue Chen; Guiling Wang
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** Unauthorized screen capturing and dissemination pose severe security threats such as data leakage and information theft. Several studies propose robust watermarking methods to track the copyright of Screen-Camera (SC) images, facilitating post-hoc certification against infringement. These techniques typically employ heuristic mathematical modeling or supervised neural network fitting as the noise layer, to enhance watermarking robustness against SC. However, both strategies cannot fundamentally achieve an effective approximation of SC noise. Mathematical simulation suffers from biased approximations due to the incomplete decomposition of the noise and the absence of interdependence among the noise components. Supervised networks require paired data to train the noise-fitting model, and it is difficult for the model to learn all the features of the noise. To address the above issues, we propose Simulation-to-Real (S2R). Specifically, an unsupervised noise layer employs unpaired data to learn the discrepancy between the modeled simulated noise distribution and the real-world SC noise distribution, rather than directly learning the mapping from sharp images to real-world images. Learning this transformation from simulation to reality is inherently simpler, as it primarily involves bridging the gap in noise distributions, instead of the complex task of reconstructing fine-grained image details. Extensive experimental results validate the efficacy of the proposed method, demonstrating superior watermark robustness and generalization compared to state-of-the-art methods.
>
---
#### [replaced 068] SASG-DA: Sparse-Aware Semantic-Guided Diffusion Augmentation For Myoelectric Gesture Recognition
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: []()**

> **作者:** Chen Liu; Can Han; Weishi Xu; Yaqi Wang; Dahong Qian
>
> **备注:** Under review
>
> **摘要:** Surface electromyography (sEMG)-based gesture recognition plays a critical role in human-machine interaction (HMI), particularly for rehabilitation and prosthetic control. However, sEMG-based systems often suffer from the scarcity of informative training data, leading to overfitting and poor generalization in deep learning models. Data augmentation offers a promising approach to increasing the size and diversity of training data, where faithfulness and diversity are two critical factors to effectiveness. However, promoting untargeted diversity can result in redundant samples with limited utility. To address these challenges, we propose a novel diffusion-based data augmentation approach, Sparse-Aware Semantic-Guided Diffusion Augmentation (SASG-DA). To enhance generation faithfulness, we introduce the Semantic Representation Guidance (SRG) mechanism by leveraging fine-grained, task-aware semantic representations as generation conditions. To enable flexible and diverse sample generation, we propose a Gaussian Modeling Semantic Sampling (GMSS) strategy, which models the semantic representation distribution and allows stochastic sampling to produce both faithful and diverse samples. To enhance targeted diversity, we further introduce a Sparse-Aware Semantic Sampling (SASS) strategy to explicitly explore underrepresented regions, improving distribution coverage and sample utility. Extensive experiments on benchmark sEMG datasets, Ninapro DB2, DB4, and DB7, demonstrate that SASG-DA significantly outperforms existing augmentation methods. Overall, our proposed data augmentation approach effectively mitigates overfitting and improves recognition performance and generalization by offering both faithful and diverse samples.
>
---
#### [replaced 069] Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark
- **分类: cs.CV**

- **链接: []()**

> **作者:** Saad Lahrichi; Jake Bova; Jesse Johnson; Jordan Malof
>
> **备注:** 8 pages, 6 figures, accepted at WACV 2026
>
> **摘要:** Recent research has demonstrated the potential of deep neural networks (DNNs) to accurately predict wildfire spread on a given day based upon high-dimensional explanatory data from a single preceding day, or from a time series of T preceding days. For the first time, we investigate a large number of existing data-driven wildfire modeling strategies under controlled conditions, revealing the best modeling strategies and resulting in models that achieve state-of-the-art (SOTA) accuracy for both single-day and multi-day input scenarios, as evaluated on a large public benchmark for next-day wildfire spread, termed the WildfireSpreadTS (WSTS) benchmark. Consistent with prior work, we found that models using time-series input obtained the best overall accuracy, suggesting this is an important future area of research. Furthermore, we create a new benchmark, WSTS+, by incorporating four additional years of historical wildfire data into the WSTS benchmark. Our benchmark doubles the number of unique years of historical data, expands its geographic scope, and, to our knowledge, represents the largest public benchmark for time-series-based wildfire spread prediction.
>
---
#### [replaced 070] TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Yiming Yang; Yueru Luo; Bingkun He; Hongbin Lin; Suzhong Fu; Chao Zheng; Zhipeng Cao; Erlong Li; Chao Yan; Shuguang Cui; Zhen Li
>
> **摘要:** Lane segment topology reasoning constructs a comprehensive road network by capturing the topological relationships between lane segments and their semantic types. This enables end-to-end autonomous driving systems to perform road-dependent maneuvers such as turning and lane changing. However, the limitations in consistent positional embedding and temporal multiple attribute learning in existing methods hinder accurate roadnet reconstruction. To address these issues, we propose TopoStreamer, an end-to-end temporal perception model for lane segment topology reasoning. Specifically, TopoStreamer introduces three key improvements: streaming attribute constraints, dynamic lane boundary positional encoding, and lane segment denoising. The streaming attribute constraints enforce temporal consistency in both centerline and boundary coordinates, along with their classifications. Meanwhile, dynamic lane boundary positional encoding enhances the learning of up-to-date positional information within queries, while lane segment denoising helps capture diverse lane segment patterns, ultimately improving model performance. Additionally, we assess the accuracy of existing models using a lane boundary classification metric, which serves as a crucial measure for lane-changing scenarios in autonomous driving. On the OpenLane-V2 dataset, TopoStreamer demonstrates significant improvements over state-of-the-art methods, achieving substantial performance gains of +3.0% mAP in lane segment perception and +1.7% OLS in centerline perception tasks.
>
---
#### [replaced 071] DMAT: An End-to-End Framework for Joint Atmospheric Turbulence Mitigation and Object Detection
- **分类: cs.CV**

- **链接: []()**

> **作者:** Paul Hill; Zhiming Liu; Alin Achim; Dave Bull; Nantheera Anantrasirichai
>
> **备注:** Accepted to WACV2026
>
> **摘要:** Atmospheric Turbulence (AT) degrades the clarity and accuracy of surveillance imagery, posing challenges not only for visualization quality but also for object classification and scene tracking. Deep learning-based methods have been proposed to improve visual quality, but spatio-temporal distortions remain a significant issue. Although deep learning-based object detection performs well under normal conditions, it struggles to operate effectively on sequences distorted by atmospheric turbulence. In this paper, we propose a novel framework that learns to compensate for distorted features while simultaneously improving visualization and object detection. This end-to-end training strategy leverages and exchanges knowledge of low-level distorted features in the AT mitigator with semantic features extracted in the object detector. Specifically, in the AT mitigator a 3D Mamba-based structure is used to handle the spatio-temporal displacements and blurring caused by turbulence. Optimization is achieved through back-propagation in both the AT mitigator and object detector. Our proposed DMAT outperforms state-of-the-art AT mitigation and object detection systems up to a 15% improvement on datasets corrupted by generated turbulence.
>
---
#### [replaced 072] GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: []()**

> **作者:** Hanrui Wang; Ching-Chun Chang; Chun-Shien Lu; Christopher Leckie; Isao Echizen
>
> **备注:** IEEE Transactions on Information Forensics and Security
>
> **摘要:** Deep neural networks are highly vulnerable to adversarial examples, which are inputs with small, carefully crafted perturbations that cause misclassification--making adversarial attacks a critical tool for evaluating robustness. Existing black-box methods typically entail a trade-off between precision and flexibility: pixel-sparse attacks (e.g., single- or few-pixel attacks) provide fine-grained control but lack adaptability, whereas patch- or frequency-based attacks improve efficiency or transferability, but at the cost of producing larger and less precise perturbations. We present GreedyPixel, a fine-grained black-box attack method that performs brute-force-style, per-pixel greedy optimization guided by a surrogate-derived priority map and refined by means of query feedback. It evaluates each coordinate directly without any gradient information, guaranteeing monotonic loss reduction and convergence to a coordinate-wise optimum, while also yielding near white-box-level precision and pixel-wise sparsity and perceptual quality. On the CIFAR-10 and ImageNet datasets, spanning convolutional neural networks (CNNs) and Transformer models, GreedyPixel achieved state-of-the-art success rates with visually imperceptible perturbations, effectively bridging the gap between black-box practicality and white-box performance. The implementation is available at https://github.com/azrealwang/greedypixel.
>
---
#### [replaced 073] GaussianArt: Unified Modeling of Geometry and Motion for Articulated Objects
- **分类: cs.CV**

- **链接: []()**

> **作者:** Licheng Shen; Saining Zhang; Honghan Li; Peilin Yang; Zihao Huang; Zongzheng Zhang; Hao Zhao
>
> **备注:** 3DV 2026 Project Page: https://sainingzhang.github.io/project/gaussianart/
>
> **摘要:** Reconstructing articulated objects is essential for building digital twins of interactive environments. However, prior methods typically decouple geometry and motion by first reconstructing object shape in distinct states and then estimating articulation through post-hoc alignment. This separation complicates the reconstruction pipeline and restricts scalability, especially for objects with complex, multi-part articulation. We introduce a unified representation that jointly models geometry and motion using articulated 3D Gaussians. This formulation improves robustness in motion decomposition and supports articulated objects with up to 20 parts, significantly outperforming prior approaches that often struggle beyond 2--3 parts due to brittle initialization. To systematically assess scalability and generalization, we propose MPArt-90, a new benchmark consisting of 90 articulated objects across 20 categories, each with diverse part counts and motion configurations. Extensive experiments show that our method consistently achieves superior accuracy in part-level geometry reconstruction and motion estimation across a broad range of object types. We further demonstrate applicability to downstream tasks such as robotic simulation and human-scene interaction modeling, highlighting the potential of unified articulated representations in scalable physical modeling.
>
---
#### [replaced 074] PISA-Bench: The PISA Index as a Multilingual and Multimodal Metric for the Evaluation of Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Patrick Haller; Fabio Barth; Jonas Golde; Georg Rehm; Alan Akbik
>
> **备注:** 8 pages, 11 tables and figures
>
> **摘要:** Vision-language models (VLMs) have demonstrated remarkable progress in multimodal reasoning. However, existing benchmarks remain limited in terms of high-quality, human-verified examples. Many current datasets rely on synthetically generated content by large language models (LLMs). Furthermore, most datasets are limited to English, as manual quality assurance of translated samples is time-consuming and costly. To fill this gap, we introduce PISA-Bench, a multilingual benchmark derived from English examples of the expert-created PISA tests, a unified framework for the assessment of student competencies in over eighty countries. Each example consists of human-extracted instructions, questions, answer options, and images, enriched with question type categories, and has been translated from English into five additional languages (Spanish, German, Chinese, French, and Italian), resulting in a fully parallel corpus covering six languages. We evaluate state-of-the-art vision-language models on PISA-Bench and find that especially small models (<20B parameters) fail to achieve high test scores. We further find substantial performance degradation on non-English splits as well as high error-rates when models are tasked with spatial and geometric reasoning. By releasing the dataset and evaluation framework, we provide a resource for advancing research on multilingual multimodal reasoning.
>
---
#### [replaced 075] Adjacent-view Transformers for Supervised Surround-view Depth Estimation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Xianda Guo; Wenjie Yuan; Yunpeng Zhang; Tian Yang; Chenming Zhang; Zheng Zhu; Qin Zou; Long Chen
>
> **备注:** This paper has been accepted by IROS 2025
>
> **摘要:** Depth estimation has been widely studied and serves as the fundamental step of 3D perception for robotics and autonomous driving. Though significant progress has been made in monocular depth estimation in the past decades, these attempts are mainly conducted on the KITTI benchmark with only front-view cameras, which ignores the correlations across surround-view cameras. In this paper, we propose an Adjacent-View Transformer for Supervised Surround-view Depth estimation (AVT-SSDepth), to jointly predict the depth maps across multiple surrounding cameras. Specifically, we employ a global-to-local feature extraction module that combines CNN with transformer layers for enriched representations. Further, the adjacent-view attention mechanism is proposed to enable the intra-view and inter-view feature propagation. The former is achieved by the self-attention module within each view, while the latter is realized by the adjacent attention module, which computes the attention across multi-cameras to exchange the multi-scale representations across surroundview feature maps. In addition, AVT-SSDepth has strong crossdataset generaliza- tion. Extensive experiments show that our method achieves superior performance over existing state-ofthe-art methods on both DDAD and nuScenes datasets. Code is available at https://github.com/XiandaGuo/SSDepth.
>
---
#### [replaced 076] ACDC: The Adverse Conditions Dataset with Correspondences for Robust Semantic Driving Scene Perception
- **分类: cs.CV**

- **链接: []()**

> **作者:** Christos Sakaridis; Haoran Wang; Ke Li; René Zurbrügg; Arpit Jadon; Wim Abbeloos; Daniel Olmeda Reino; Luc Van Gool; Dengxin Dai
>
> **备注:** IEEE T-PAMI 2025. Extended version of original conference paper published in ICCV 2021
>
> **摘要:** Level-5 driving automation requires a robust visual perception system that can parse input images under any condition. However, existing driving datasets for dense semantic perception are either dominated by images captured under normal conditions or are small in scale. To address this, we introduce ACDC, the Adverse Conditions Dataset with Correspondences for training and testing methods for diverse semantic perception tasks on adverse visual conditions. ACDC consists of a large set of 8012 images, half of which (4006) are equally distributed between four common adverse conditions: fog, nighttime, rain, and snow. Each adverse-condition image comes with a high-quality pixel-level panoptic annotation, a corresponding image of the same scene under normal conditions, and a binary mask that distinguishes between intra-image regions of clear and uncertain semantic content. 1503 of the corresponding normal-condition images feature panoptic annotations, raising the total annotated images to 5509. ACDC supports the standard tasks of semantic segmentation, object detection, instance segmentation, and panoptic segmentation, as well as the newly introduced uncertainty-aware semantic segmentation. A detailed empirical study demonstrates the challenges that the adverse domains of ACDC pose to state-of-the-art supervised and unsupervised approaches and indicates the value of our dataset in steering future progress in the field. Our dataset and benchmark are publicly available at https://acdc.vision.ee.ethz.ch
>
---
#### [replaced 077] An Instance-Aware Prompting Framework for Training-free Camouflaged Object Segmentation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Chao Yin; Jide Li; Hang Yao; Xiaoqiang Li
>
> **备注:** under review
>
> **摘要:** Training-free Camouflaged Object Segmentation (COS) seeks to segment camouflaged objects without task-specific training, by automatically generating visual prompts to guide the Segment Anything Model (SAM). However, existing pipelines mostly yield semantic-level prompts, which drive SAM to coarse semantic masks and struggle to handle multiple discrete camouflaged instances effectively. To address this critical limitation, we propose an \textbf{I}nstance-\textbf{A}ware \textbf{P}rompting \textbf{F}ramework (IAPF) tailored for the first training-free COS that upgrades prompt granularity from semantic to instance-level while keeping all components frozen. The centerpiece is an Instance Mask Generator that (i) leverages a detector-agnostic enumerator to produce precise instance-level box prompts for the foreground tag, and (ii) introduces the Single-Foreground Multi-Background Prompting (SFMBP) strategy to sample region-constrained point prompts within each box prompt, enabling SAM to output instance masks. The pipeline is supported by a simple text prompt generator that produces image-specific tags and a self-consistency vote across synonymous task-generic prompts to stabilize inference. Extensive evaluations on three COS benchmarks, two CIS benchmarks, and two downstream datasets demonstrate state-of-the-art performance among training-free methods. Code will be released upon acceptance.
>
---
#### [replaced 078] ArchCAD-400K: A Large-Scale CAD drawings Dataset and New Baseline for Panoptic Symbol Spotting
- **分类: cs.CV**

- **链接: []()**

> **作者:** Ruifeng Luo; Zhengjie Liu; Tianxiao Cheng; Jie Wang; Tongjie Wang; Xingguang Wei; Haomin Wang; YanPeng Li; Fu Chai; Fei Cheng; Shenglong Ye; Wenhai Wang; Yanting Zhang; Yu Qiao; Hongjie Zhang; Xianzhong Zhao
>
> **摘要:** Recognizing symbols in architectural CAD drawings is critical for various advanced engineering applications. In this paper, we propose a novel CAD data annotation engine that leverages intrinsic attributes from systematically archived CAD drawings to automatically generate high-quality annotations, thus significantly reducing manual labeling efforts. Utilizing this engine, we construct ArchCAD-400K, a large-scale CAD dataset consisting of 413,062 chunks from 5538 highly standardized drawings, making it over 26 times larger than the largest existing CAD dataset. ArchCAD-400K boasts an extended drawing diversity and broader categories, offering line-grained annotations. Furthermore, we present a new baseline model for panoptic symbol spotting, termed Dual-Pathway Symbol Spotter (DPSS). It incorporates an adaptive fusion module to enhance primitive features with complementary image features, achieving state-of-the-art performance and enhanced robustness. Extensive experiments validate the effectiveness of DPSS, demonstrating the value of ArchCAD-400K and its potential to drive innovation in architectural design and construction.
>
---
#### [replaced 079] RL-U$^2$Net: A Dual-Branch UNet with Reinforcement Learning-Assisted Multimodal Feature Fusion for Accurate 3D Whole-Heart Segmentation
- **分类: eess.IV; cs.CV**

- **链接: []()**

> **作者:** Jierui Qu; Jianchun Zhao
>
> **摘要:** Accurate whole-heart segmentation is a critical component in the precise diagnosis and interventional planning of cardiovascular diseases. Integrating complementary information from modalities such as computed tomography (CT) and magnetic resonance imaging (MRI) can significantly enhance segmentation accuracy and robustness. However, existing multi-modal segmentation methods face several limitations: severe spatial inconsistency between modalities hinders effective feature fusion; fusion strategies are often static and lack adaptability; and the processes of feature alignment and segmentation are decoupled and inefficient. To address these challenges, we propose a dual-branch U-Net architecture enhanced by reinforcement learning for feature alignment, termed RL-U$^2$Net, designed for precise and efficient multi-modal 3D whole-heart segmentation. The model employs a dual-branch U-shaped network to process CT and MRI patches in parallel, and introduces a novel RL-XAlign module between the encoders. The module employs a cross-modal attention mechanism to capture semantic correspondences between modalities and a reinforcement-learning agent learns an optimal rotation strategy that consistently aligns anatomical pose and texture features. The aligned features are then reconstructed through their respective decoders. Finally, an ensemble-learning-based decision module integrates the predictions from individual patches to produce the final segmentation result. Experimental results on the publicly available MM-WHS 2017 dataset demonstrate that the proposed RL-U$^2$Net outperforms existing state-of-the-art methods, achieving Dice coefficients of 93.1% on CT and 87.0% on MRI, thereby validating the effectiveness and superiority of the proposed approach.
>
---
#### [replaced 080] RAFT - A Domain Adaptation Framework for RGB & LiDAR Semantic Segmentation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Edward Humes; Xiaomin Lin; Boxun Hu; Rithvik Jonna; Tinoosh Mohsenin
>
> **备注:** Submitted to RA-L
>
> **摘要:** Image segmentation is a powerful computer vision technique for scene understanding. However, real-world deployment is stymied by the need for high-quality, meticulously labeled datasets. Synthetic data provides high-quality labels while reducing the need for manual data collection and annotation. However, deep neural networks trained on synthetic data often face the Syn2Real problem, leading to poor performance in real-world deployments. To mitigate the aforementioned gap in image segmentation, we propose RAFT, a novel framework for adapting image segmentation models using minimal labeled real-world data through data and feature augmentations, as well as active learning. To validate RAFT, we perform experiments on the synthetic-to-real "SYNTHIA->Cityscapes" and "GTAV->Cityscapes" benchmarks. We managed to surpass the previous state of the art, HALO. SYNTHIA->Cityscapes experiences an improvement in mIoU* upon domain adaptation of 2.1%/79.9%, and GTAV->Cityscapes experiences a 0.4%/78.2% improvement in mIoU. Furthermore, we test our approach on the real-to-real benchmark of "Cityscapes->ACDC", and again surpass HALO, with a gain in mIoU upon adaptation of 1.3%/73.2%. Finally, we examine the effect of the allocated annotation budget and various components of RAFT upon the final transfer mIoU.
>
---
#### [replaced 081] DI3CL: Contrastive Learning With Dynamic Instances and Contour Consistency for SAR Land-Cover Classification Foundation Model
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zhongle Ren; Hui Ding; Kai Wang; Biao Hou; Xingyu Luo; Weibin Li; Licheng Jiao
>
> **备注:** 18 pages, 10 figures;Submitted to IEEE Transactions on Image Processing (TIP); In peer review
>
> **摘要:** Although significant advances have been achieved in SAR land-cover classification, recent methods remain predominantly focused on supervised learning, which relies heavily on extensive labeled datasets. This dependency not only limits scalability and generalization but also restricts adaptability to diverse application scenarios. In this paper, a general-purpose foundation model for SAR land-cover classification is developed, serving as a robust cornerstone to accelerate the development and deployment of various downstream models. Specifically, a Dynamic Instance and Contour Consistency Contrastive Learning (DI3CL) pre-training framework is presented, which incorporates a Dynamic Instance (DI) module and a Contour Consistency (CC) module. DI module enhances global contextual awareness by enforcing local consistency across different views of the same region. CC module leverages shallow feature maps to guide the model to focus on the geometric contours of SAR land-cover objects, thereby improving structural discrimination. Additionally, to enhance robustness and generalization during pre-training, a large-scale and diverse dataset named SARSense, comprising 460,532 SAR images, is constructed to enable the model to capture comprehensive and representative features. To evaluate the generalization capability of our foundation model, we conducted extensive experiments across a variety of SAR land-cover classification tasks, including SAR land-cover mapping, water body detection, and road extraction. The results consistently demonstrate that the proposed DI3CL outperforms existing methods. Our code and pre-trained weights are publicly available at: https://github.com/SARpre-train/DI3CL.
>
---
#### [replaced 082] FlowLensing: Simulating Gravitational Lensing with Flow Matching
- **分类: astro-ph.IM; cs.CV**

- **链接: []()**

> **作者:** Hamees Sayed; Pranath Reddy; Michael W. Toomey; Sergei Gleyzer
>
> **备注:** 6 pages, 2 figures, 3 tables
>
> **摘要:** Gravitational lensing is one of the most powerful probes of dark matter, yet creating high-fidelity lensed images at scale remains a bottleneck. Existing tools rely on ray-tracing or forward-modeling pipelines that, while precise, are prohibitively slow. We introduce FlowLensing, a Diffusion Transformer-based compact and efficient flow-matching model for strong gravitational lensing simulation. FlowLensing operates in both discrete and continuous regimes, handling classes such as different dark matter models as well as continuous model parameters ensuring physical consistency. By enabling scalable simulations, our model can advance dark matter studies, specifically for probing dark matter substructure in cosmological surveys. We find that our model achieves a speedup of over 200$\times$ compared to classical simulators for intensive dark matter models, with high fidelity and low inference latency. FlowLensing enables rapid, scalable, and physically consistent image synthesis, offering a practical alternative to traditional forward-modeling pipelines.
>
---
