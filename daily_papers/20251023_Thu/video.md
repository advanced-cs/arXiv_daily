# 计算机视觉 cs.CV

- **最新发布 87 篇**

- **更新 82 篇**

## 最新发布

#### [new 001] HAD: Hierarchical Asymmetric Distillation to Bridge Spatio-Temporal Gaps in Event-Based Object Tracking
- **分类: cs.CV**

- **简介: 该论文针对事件相机与RGB相机在时空特性上的不对称性，提出层级异构知识蒸馏框架HAD，用于提升多模态目标跟踪性能。通过分层对齐策略，在保留学生模型高效性的同时，有效融合双模态优势，显著改善高动态、高速运动等复杂场景下的跟踪效果。**

- **链接: [http://arxiv.org/pdf/2510.19560v1](http://arxiv.org/pdf/2510.19560v1)**

> **作者:** Yao Deng; Xian Zhong; Wenxuan Liu; Zhaofei Yu; Jingling Yuan; Tiejun Huang
>
> **摘要:** RGB cameras excel at capturing rich texture details with high spatial resolution, whereas event cameras offer exceptional temporal resolution and a high dynamic range (HDR). Leveraging their complementary strengths can substantially enhance object tracking under challenging conditions, such as high-speed motion, HDR environments, and dynamic background interference. However, a significant spatio-temporal asymmetry exists between these two modalities due to their fundamentally different imaging mechanisms, hindering effective multi-modal integration. To address this issue, we propose {Hierarchical Asymmetric Distillation} (HAD), a multi-modal knowledge distillation framework that explicitly models and mitigates spatio-temporal asymmetries. Specifically, HAD proposes a hierarchical alignment strategy that minimizes information loss while maintaining the student network's computational efficiency and parameter compactness. Extensive experiments demonstrate that HAD consistently outperforms state-of-the-art methods, and comprehensive ablation studies further validate the effectiveness and necessity of each designed component. The code will be released soon.
>
---
#### [new 002] Background Fades, Foreground Leads: Curriculum-Guided Background Pruning for Efficient Foreground-Centric Collaborative Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对车载协同感知中的带宽受限问题，提出FadeLead框架。通过课程学习策略，将背景上下文融入前景特征中，实现高效前景主导的特征共享，提升感知性能。**

- **链接: [http://arxiv.org/pdf/2510.19250v1](http://arxiv.org/pdf/2510.19250v1)**

> **作者:** Yuheng Wu; Xiangbo Gao; Quang Tau; Zhengzhong Tu; Dongman Lee
>
> **摘要:** Collaborative perception enhances the reliability and spatial coverage of autonomous vehicles by sharing complementary information across vehicles, offering a promising solution to long-tail scenarios that challenge single-vehicle perception. However, the bandwidth constraints of vehicular networks make transmitting the entire feature map impractical. Recent methods, therefore, adopt a foreground-centric paradigm, transmitting only predicted foreground-region features while discarding the background, which encodes essential context. We propose FadeLead, a foreground-centric framework that overcomes this limitation by learning to encapsulate background context into compact foreground features during training. At the core of our design is a curricular learning strategy that leverages background cues early on but progressively prunes them away, forcing the model to internalize context into foreground representations without transmitting background itself. Extensive experiments on both simulated and real-world benchmarks show that FadeLead outperforms prior methods under different bandwidth settings, underscoring the effectiveness of context-enriched foreground sharing.
>
---
#### [new 003] Reasoning Like Experts: Leveraging Multimodal Large Language Models for Drawing-based Psychoanalysis
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出PICK框架，利用多模态大模型进行绘画心理分析，聚焦房树人测验。通过分层解析绘画结构并注入专业知识，实现对心理状态的专家级推理，解决了MLLM在主观情感领域应用不足的问题，提升了心理分析的准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.19451v1](http://arxiv.org/pdf/2510.19451v1)**

> **作者:** Xueqi Ma; Yanbei Jiang; Sarah Erfani; James Bailey; Weifeng Liu; Krista A. Ehinger; Jey Han Lau
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance across various objective multimodal perception tasks, yet their application to subjective, emotionally nuanced domains, such as psychological analysis, remains largely unexplored. In this paper, we introduce PICK, a multi-step framework designed for Psychoanalytical Image Comprehension through hierarchical analysis and Knowledge injection with MLLMs, specifically focusing on the House-Tree-Person (HTP) Test, a widely used psychological assessment in clinical practice. First, we decompose drawings containing multiple instances into semantically meaningful sub-drawings, constructing a hierarchical representation that captures spatial structure and content across three levels: single-object level, multi-object level, and whole level. Next, we analyze these sub-drawings at each level with a targeted focus, extracting psychological or emotional insights from their visual cues. We also introduce an HTP knowledge base and design a feature extraction module, trained with reinforcement learning, to generate a psychological profile for single-object level analysis. This profile captures both holistic stylistic features and dynamic object-specific features (such as those of the house, tree, or person), correlating them with psychological states. Finally, we integrate these multi-faceted information to produce a well-informed assessment that aligns with expert-level reasoning. Our approach bridges the gap between MLLMs and specialized expert domains, offering a structured and interpretable framework for understanding human mental states through visual expression. Experimental results demonstrate that the proposed PICK significantly enhances the capability of MLLMs in psychological analysis. It is further validated as a general framework through extensions to emotion understanding tasks.
>
---
#### [new 004] VGD: Visual Geometry Gaussian Splatting for Feed-Forward Surround-view Driving Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出VGD框架，针对前馈式环视自动驾驶场景重建中几何一致性与新视角质量难以兼顾的问题。通过轻量级几何分支提取几何先验，结合高斯头预测渲染参数，并利用多尺度特征联合优化语义细节，实现高效高保真重建。**

- **链接: [http://arxiv.org/pdf/2510.19578v1](http://arxiv.org/pdf/2510.19578v1)**

> **作者:** Junhong Lin; Kangli Wang; Shunzhou Wang; Songlin Fan; Ge Li; Wei Gao
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Feed-forward surround-view autonomous driving scene reconstruction offers fast, generalizable inference ability, which faces the core challenge of ensuring generalization while elevating novel view quality. Due to the surround-view with minimal overlap regions, existing methods typically fail to ensure geometric consistency and reconstruction quality for novel views. To tackle this tension, we claim that geometric information must be learned explicitly, and the resulting features should be leveraged to guide the elevating of semantic quality in novel views. In this paper, we introduce \textbf{Visual Gaussian Driving (VGD)}, a novel feed-forward end-to-end learning framework designed to address this challenge. To achieve generalizable geometric estimation, we design a lightweight variant of the VGGT architecture to efficiently distill its geometric priors from the pre-trained VGGT to the geometry branch. Furthermore, we design a Gaussian Head that fuses multi-scale geometry tokens to predict Gaussian parameters for novel view rendering, which shares the same patch backbone as the geometry branch. Finally, we integrate multi-scale features from both geometry and Gaussian head branches to jointly supervise a semantic refinement model, optimizing rendering quality through feature-consistent learning. Experiments on nuScenes demonstrate that our approach significantly outperforms state-of-the-art methods in both objective metrics and subjective quality under various settings, which validates VGD's scalability and high-fidelity surround-view reconstruction.
>
---
#### [new 005] [De|Re]constructing VLMs' Reasoning in Counting
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）在计数任务中的推理能力，针对其在对象数量、空间布局和干扰项影响下的表现不佳问题，通过控制实验与层分析揭示错误源于最后一层表示到输出空间的映射错误。提出仅微调输出层的方法，提升准确率最高达21%，并在真实数据集上验证有效性。**

- **链接: [http://arxiv.org/pdf/2510.19555v1](http://arxiv.org/pdf/2510.19555v1)**

> **作者:** Simone Alghisi; Gabriel Roccabruna; Massimo Rizzoli; Seyed Mahed Mousavi; Giuseppe Riccardi
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Vision-Language Models (VLMs) have recently gained attention due to their competitive performance on multiple downstream tasks, achieved by following user-input instructions. However, VLMs still exhibit several limitations in visual reasoning, such as difficulties in identifying relations (e.g., spatial, temporal, and among objects), understanding temporal sequences (e.g., frames), and counting objects. In this work, we go beyond score-level benchmark evaluations of VLMs by investigating the underlying causes of their failures and proposing a targeted approach to improve their reasoning capabilities. We study the reasoning skills of seven state-of-the-art VLMs in the counting task under controlled experimental conditions. Our experiments show that VLMs are highly sensitive to the number and type of objects, their spatial arrangement, and the co-occurrence of distractors. A layer-wise analysis reveals that errors are due to incorrect mapping of the last-layer representation into the output space. Our targeted training shows that fine-tuning just the output layer improves accuracy by up to 21%. We corroborate these findings by achieving consistent improvements on real-world datasets.
>
---
#### [new 006] Ninja Codes: Neurally Generated Fiducial Markers for Stealthy 6-DoF Tracking
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出Ninja Codes，一种通过神经网络生成的隐蔽式标记，用于实现6-DoF跟踪。旨在解决传统标记显眼、影响美观的问题。工作包括设计编码器将图像转为视觉上几乎不可见的标记，并联合训练检测网络，实现打印后在普通纸张上被摄像头可靠识别，适用于AR、机器人等场景。**

- **链接: [http://arxiv.org/pdf/2510.18976v1](http://arxiv.org/pdf/2510.18976v1)**

> **作者:** Yuichiro Takeuchi; Yusuke Imoto; Shunya Kato
>
> **备注:** 11 pages, 12 figures
>
> **摘要:** In this paper we describe Ninja Codes, neurally-generated fiducial markers that can be made to naturally blend into various real-world environments. An encoder network converts arbitrary images into Ninja Codes by applying visually modest alterations; the resulting codes, printed and pasted onto surfaces, can provide stealthy 6-DoF location tracking for a wide range of applications including augmented reality, robotics, motion-based user interfaces, etc. Ninja Codes can be printed using off-the-shelf color printers on regular printing paper, and can be detected using any device equipped with a modern RGB camera and capable of running inference. Using an end-to-end process inspired by prior work on deep steganography, we jointly train a series of network modules that perform the creation and detection of Ninja Codes. Through experiments, we demonstrate Ninja Codes' ability to provide reliable location tracking under common indoor lighting conditions, while successfully concealing themselves within diverse environmental textures. We expect Ninja Codes to offer particular value in scenarios where the conspicuous appearances of conventional fiducial markers make them undesirable for aesthetic and other reasons.
>
---
#### [new 007] CBDiff:Conditional Bernoulli Diffusion Models for Image Forgery Localization
- **分类: cs.CV**

- **简介: 该论文聚焦图像伪造定位任务，针对现有方法生成单一确定性结果、精度与可靠性不足的问题，提出条件伯努利扩散模型CBDiff。通过引入伯努利噪声和时间步交叉注意力机制，生成多组多样且可信的伪造区域预测，有效建模伪造分布的不确定性，显著提升定位性能。**

- **链接: [http://arxiv.org/pdf/2510.19597v1](http://arxiv.org/pdf/2510.19597v1)**

> **作者:** Zhou Lei; Pan Gang; Wang Jiahao; Sun Di
>
> **摘要:** Image Forgery Localization (IFL) is a crucial task in image forensics, aimed at accurately identifying manipulated or tampered regions within an image at the pixel level. Existing methods typically generate a single deterministic localization map, which often lacks the precision and reliability required for high-stakes applications such as forensic analysis and security surveillance. To enhance the credibility of predictions and mitigate the risk of errors, we introduce an advanced Conditional Bernoulli Diffusion Model (CBDiff). Given a forged image, CBDiff generates multiple diverse and plausible localization maps, thereby offering a richer and more comprehensive representation of the forgery distribution. This approach addresses the uncertainty and variability inherent in tampered regions. Furthermore, CBDiff innovatively incorporates Bernoulli noise into the diffusion process to more faithfully reflect the inherent binary and sparse properties of forgery masks. Additionally, CBDiff introduces a Time-Step Cross-Attention (TSCAttention), which is specifically designed to leverage semantic feature guidance with temporal steps to improve manipulation detection. Extensive experiments on eight publicly benchmark datasets demonstrate that CBDiff significantly outperforms existing state-of-the-art methods, highlighting its strong potential for real-world deployment.
>
---
#### [new 008] Uncertainty evaluation of segmentation models for Earth observation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对遥感图像语义分割中的不确定性评估问题，提出并评测了多种可扩展的像素级不确定性估计方法。研究对比了不同模型与度量在两个遥感数据集上的表现，旨在提升预测可靠性，识别错误区域与噪声输入。**

- **链接: [http://arxiv.org/pdf/2510.19586v1](http://arxiv.org/pdf/2510.19586v1)**

> **作者:** Melanie Rey; Andriy Mnih; Maxim Neumann; Matt Overlan; Drew Purves
>
> **摘要:** This paper investigates methods for estimating uncertainty in semantic segmentation predictions derived from satellite imagery. Estimating uncertainty for segmentation presents unique challenges compared to standard image classification, requiring scalable methods producing per-pixel estimates. While most research on this topic has focused on scene understanding or medical imaging, this work benchmarks existing methods specifically for remote sensing and Earth observation applications. Our evaluation focuses on the practical utility of uncertainty measures, testing their ability to identify prediction errors and noise-corrupted input image regions. Experiments are conducted on two remote sensing datasets, PASTIS and ForTy, selected for their differences in scale, geographic coverage, and label confidence. We perform an extensive evaluation featuring several models, such as Stochastic Segmentation Networks and ensembles, in combination with a number of neural architectures and uncertainty metrics. We make a number of practical recommendations based on our findings.
>
---
#### [new 009] olmOCR 2: Unit Test Rewards for Document OCR
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出olmOCR 2，用于将数字化印刷文档（如PDF）转换为结构化文本。针对复杂布局下OCR精度不足的问题，采用基于可验证奖励的强化学习，利用大量二元单元测试作为奖励信号，训练7B视觉语言模型。通过合成数据生成管道提升测试覆盖度，显著提升数学公式、表格和多栏布局的识别性能。**

- **链接: [http://arxiv.org/pdf/2510.19817v1](http://arxiv.org/pdf/2510.19817v1)**

> **作者:** Jake Poznanski; Luca Soldaini; Kyle Lo
>
> **备注:** https://olmocr.allen.ai/
>
> **摘要:** We present olmOCR 2, the latest in our family of powerful OCR systems for converting digitized print documents, like PDFs, into clean, naturally ordered plain text. olmOCR 2 is powered by olmOCR-2-7B-1025, a specialized, 7B vision language model (VLM) trained using reinforcement learning with verifiable rewards (RLVR), where our rewards are a diverse set of binary unit tests. To scale unit test creation, we develop a pipeline for generating synthetic documents with diverse and challenging layouts, known ground-truth HTML source code, and extracted test cases. We show that RL training on these test cases results in state-of-the-art performance on olmOCR-Bench, our English-language OCR benchmark, with the largest improvements in math formula conversion, table parsing, and multi-column layouts compared to previous versions. We release our model, data and code under permissive open licenses.
>
---
#### [new 010] Towards Single-Source Domain Generalized Object Detection via Causal Visual Prompts
- **分类: cs.CV**

- **简介: 该论文聚焦单源域泛化目标检测（SDGOD）任务，针对模型因依赖表面特征导致的虚假关联问题，提出Cauvis方法。通过交叉注意力提示模块与双分支适配器，分离因果与虚假特征，增强域不变表示，显著提升模型在未知域的泛化能力与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.19487v1](http://arxiv.org/pdf/2510.19487v1)**

> **作者:** Chen Li; Huiying Xu; Changxin Gao; Zeyu Wang; Yun Liu; Xinzhong Zhu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Single-source Domain Generalized Object Detection (SDGOD), as a cutting-edge research topic in computer vision, aims to enhance model generalization capability in unseen target domains through single-source domain training. Current mainstream approaches attempt to mitigate domain discrepancies via data augmentation techniques. However, due to domain shift and limited domain-specific knowledge, models tend to fall into the pitfall of spurious correlations. This manifests as the model's over-reliance on simplistic classification features (e.g., color) rather than essential domain-invariant representations like object contours. To address this critical challenge, we propose the Cauvis (Causal Visual Prompts) method. First, we introduce a Cross-Attention Prompts module that mitigates bias from spurious features by integrating visual prompts with cross-attention. To address the inadequate domain knowledge coverage and spurious feature entanglement in visual prompts for single-domain generalization, we propose a dual-branch adapter that disentangles causal-spurious features while achieving domain adaptation via high-frequency feature extraction. Cauvis achieves state-of-the-art performance with 15.9-31.4% gains over existing domain generalization methods on SDGOD datasets, while exhibiting significant robustness advantages in complex interference environments.
>
---
#### [new 011] PoseCrafter: Extreme Pose Estimation with Hybrid Video Synthesis
- **分类: cs.CV**

- **简介: 该论文针对稀疏重叠图像对的极端位姿估计难题，提出PoseCrafter方法。通过融合视频插值与位姿条件的新视角合成实现清晰中间帧生成，并设计基于特征匹配的筛选器（FMS）高效选择适配位姿估计的帧，显著提升小/无重叠情况下的位姿估计性能。**

- **链接: [http://arxiv.org/pdf/2510.19527v1](http://arxiv.org/pdf/2510.19527v1)**

> **作者:** Qing Mao; Tianxin Huang; Yu Zhu; Jinqiu Sun; Yanning Zhang; Gim Hee Lee
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Pairwise camera pose estimation from sparsely overlapping image pairs remains a critical and unsolved challenge in 3D vision. Most existing methods struggle with image pairs that have small or no overlap. Recent approaches attempt to address this by synthesizing intermediate frames using video interpolation and selecting key frames via a self-consistency score. However, the generated frames are often blurry due to small overlap inputs, and the selection strategies are slow and not explicitly aligned with pose estimation. To solve these cases, we propose Hybrid Video Generation (HVG) to synthesize clearer intermediate frames by coupling a video interpolation model with a pose-conditioned novel view synthesis model, where we also propose a Feature Matching Selector (FMS) based on feature correspondence to select intermediate frames appropriate for pose estimation from the synthesized results. Extensive experiments on Cambridge Landmarks, ScanNet, DL3DV-10K, and NAVI demonstrate that, compared to existing SOTA methods, PoseCrafter can obviously enhance the pose estimation performances, especially on examples with small or no overlap.
>
---
#### [new 012] OmniMotion-X: Versatile Multimodal Whole-Body Motion Generation
- **分类: cs.CV**

- **简介: 该论文提出OmniMotion-X，一个统一的多模态全身动作生成框架，解决文本、音乐、语音等多源条件下的动作生成与控制问题。通过引入参考动作和渐进式混合训练策略，结合自回归扩散变换器与大规模标准化数据集OmniMoCap-X，实现高质量、一致且可控的长时动作生成。**

- **链接: [http://arxiv.org/pdf/2510.19789v1](http://arxiv.org/pdf/2510.19789v1)**

> **作者:** Guowei Xu; Yuxuan Bian; Ailing Zeng; Mingyi Shi; Shaoli Huang; Wen Li; Lixin Duan; Qiang Xu
>
> **摘要:** This paper introduces OmniMotion-X, a versatile multimodal framework for whole-body human motion generation, leveraging an autoregressive diffusion transformer in a unified sequence-to-sequence manner. OmniMotion-X efficiently supports diverse multimodal tasks, including text-to-motion, music-to-dance, speech-to-gesture, and global spatial-temporal control scenarios (e.g., motion prediction, in-betweening, completion, and joint/trajectory-guided synthesis), as well as flexible combinations of these tasks. Specifically, we propose the use of reference motion as a novel conditioning signal, substantially enhancing the consistency of generated content, style, and temporal dynamics crucial for realistic animations. To handle multimodal conflicts, we introduce a progressive weak-to-strong mixed-condition training strategy. To enable high-quality multimodal training, we construct OmniMoCap-X, the largest unified multimodal motion dataset to date, integrating 28 publicly available MoCap sources across 10 distinct tasks, standardized to the SMPL-X format at 30 fps. To ensure detailed and consistent annotations, we render sequences into videos and use GPT-4o to automatically generate structured and hierarchical captions, capturing both low-level actions and high-level semantics. Extensive experimental evaluations confirm that OmniMotion-X significantly surpasses existing methods, demonstrating state-of-the-art performance across multiple multimodal tasks and enabling the interactive generation of realistic, coherent, and controllable long-duration motions.
>
---
#### [new 013] LyTimeT: Towards Robust and Interpretable State-Variable Discovery
- **分类: cs.CV**

- **简介: 该论文提出LyTimeT框架，用于从高维视频中提取鲁棒、可解释的动力系统状态变量。针对背景运动等干扰因素导致的变量提取困难问题，采用时空注意力与稳定性约束结合的方法，实现准确长期预测与物理可解释性。**

- **链接: [http://arxiv.org/pdf/2510.19716v1](http://arxiv.org/pdf/2510.19716v1)**

> **作者:** Kuai Yu; Crystal Su; Xiang Liu; Judah Goldfeder; Mingyuan Shao; Hod Lipson
>
> **摘要:** Extracting the true dynamical variables of a system from high-dimensional video is challenging due to distracting visual factors such as background motion, occlusions, and texture changes. We propose LyTimeT, a two-phase framework for interpretable variable extraction that learns robust and stable latent representations of dynamical systems. In Phase 1, LyTimeT employs a spatio-temporal TimeSformer-based autoencoder that uses global attention to focus on dynamically relevant regions while suppressing nuisance variation, enabling distraction-robust latent state learning and accurate long-horizon video prediction. In Phase 2, we probe the learned latent space, select the most physically meaningful dimensions using linear correlation analysis, and refine the transition dynamics with a Lyapunov-based stability regularizer to enforce contraction and reduce error accumulation during roll-outs. Experiments on five synthetic benchmarks and four real-world dynamical systems, including chaotic phenomena, show that LyTimeT achieves mutual information and intrinsic dimension estimates closest to ground truth, remains invariant under background perturbations, and delivers the lowest analytical mean squared error among CNN-based (TIDE) and transformer-only baselines. Our results demonstrate that combining spatio-temporal attention with stability constraints yields predictive models that are not only accurate but also physically interpretable.
>
---
#### [new 014] PruneHal: Reducing Hallucinations in Multi-modal Large Language Models through Adaptive KV Cache Pruning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大模型中的幻觉问题，提出无需训练的PruneHal方法，通过自适应KV缓存剪枝增强模型对关键视觉信息的关注，减少冗余视觉令牌干扰，有效抑制幻觉，且无额外计算开销，适用于多种解码策略。**

- **链接: [http://arxiv.org/pdf/2510.19183v1](http://arxiv.org/pdf/2510.19183v1)**

> **作者:** Fengyuan Sun; Hui Chen; Xinhao Xu; Dandan Zheng; Jingdong Chen; Jun Zhou; Jungong Han; Guiguang Ding
>
> **摘要:** While multi-modal large language models (MLLMs) have made significant progress in recent years, the issue of hallucinations remains a major challenge. To mitigate this phenomenon, existing solutions either introduce additional data for further training or incorporate external or internal information during inference. However, these approaches inevitably introduce extra computational costs. In this paper, we observe that hallucinations in MLLMs are strongly associated with insufficient attention allocated to visual tokens. In particular, the presence of redundant visual tokens disperses the model's attention, preventing it from focusing on the most informative ones. As a result, critical visual cues are often under-attended, which in turn exacerbates the occurrence of hallucinations. Building on this observation, we propose \textbf{PruneHal}, a training-free, simple yet effective method that leverages adaptive KV cache pruning to enhance the model's focus on critical visual information, thereby mitigating hallucinations. To the best of our knowledge, we are the first to apply token pruning for hallucination mitigation in MLLMs. Notably, our method don't require additional training and incurs nearly no extra inference cost. Moreover, PruneHal is model-agnostic and can be seamlessly integrated with different decoding strategies, including those specifically designed for hallucination mitigation. We evaluate PruneHal on several widely used hallucination evaluation benchmarks using four mainstream MLLMs, achieving robust and outstanding results that highlight the effectiveness and superiority of our method. Our code will be publicly available.
>
---
#### [new 015] Enhancing Early Alzheimer Disease Detection through Big Data and Ensemble Few-Shot Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于早期阿尔茨海默病检测任务，针对标注数据稀缺、疾病复杂及隐私限制问题，提出基于预训练CNN与集成少样本学习的混合模型。通过融合多种编码器及双损失机制，提升医学图像特征表达与分类精度，在两个数据集上分别达到99.72%和99.86%的准确率。**

- **链接: [http://arxiv.org/pdf/2510.19282v1](http://arxiv.org/pdf/2510.19282v1)**

> **作者:** Safa Ben Atitallah; Maha Driss; Wadii Boulila; Anis Koubaa
>
> **摘要:** Alzheimer disease is a severe brain disorder that causes harm in various brain areas and leads to memory damage. The limited availability of labeled medical data poses a significant challenge for accurate Alzheimer disease detection. There is a critical need for effective methods to improve the accuracy of Alzheimer disease detection, considering the scarcity of labeled data, the complexity of the disease, and the constraints related to data privacy. To address this challenge, our study leverages the power of big data in the form of pre-trained Convolutional Neural Networks (CNNs) within the framework of Few-Shot Learning (FSL) and ensemble learning. We propose an ensemble approach based on a Prototypical Network (ProtoNet), a powerful method in FSL, integrating various pre-trained CNNs as encoders. This integration enhances the richness of features extracted from medical images. Our approach also includes a combination of class-aware loss and entropy loss to ensure a more precise classification of Alzheimer disease progression levels. The effectiveness of our method was evaluated using two datasets, the Kaggle Alzheimer dataset and the ADNI dataset, achieving an accuracy of 99.72% and 99.86%, respectively. The comparison of our results with relevant state-of-the-art studies demonstrated that our approach achieved superior accuracy and highlighted its validity and potential for real-world applications in early Alzheimer disease detection.
>
---
#### [new 016] Re-Activating Frozen Primitives for 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对3D高斯泼溅（3D-GS）在复杂场景中出现的过重建伪影问题，提出ReAct-GS方法。通过重要性感知的重新激活准则与自适应参数扰动机制，解决梯度稀释与原始体素冻结问题，有效提升细节还原能力，显著改善视图合成质量，并可兼容其他3D-GS变体。**

- **链接: [http://arxiv.org/pdf/2510.19653v1](http://arxiv.org/pdf/2510.19653v1)**

> **作者:** Yuxin Cheng; Binxiao Huang; Wenyong Zhou; Taiqiang Wu; Zhengwu Liu; Graziano Chesi; Ngai Wong
>
> **摘要:** 3D Gaussian Splatting (3D-GS) achieves real-time photorealistic novel view synthesis, yet struggles with complex scenes due to over-reconstruction artifacts, manifesting as local blurring and needle-shape distortions. While recent approaches attribute these issues to insufficient splitting of large-scale Gaussians, we identify two fundamental limitations: gradient magnitude dilution during densification and the primitive frozen phenomenon, where essential Gaussian densification is inhibited in complex regions while suboptimally scaled Gaussians become trapped in local optima. To address these challenges, we introduce ReAct-GS, a method founded on the principle of re-activation. Our approach features: (1) an importance-aware densification criterion incorporating $\alpha$-blending weights from multiple viewpoints to re-activate stalled primitive growth in complex regions, and (2) a re-activation mechanism that revitalizes frozen primitives through adaptive parameter perturbations. Comprehensive experiments across diverse real-world datasets demonstrate that ReAct-GS effectively eliminates over-reconstruction artifacts and achieves state-of-the-art performance on standard novel view synthesis metrics while preserving intricate geometric details. Additionally, our re-activation mechanism yields consistent improvements when integrated with other 3D-GS variants such as Pixel-GS, demonstrating its broad applicability.
>
---
#### [new 017] Explainable Face Presentation Attack Detection via Ensemble-CAM
- **分类: cs.CV**

- **简介: 该论文针对深度学习驱动的面部活体检测系统缺乏可解释性的问题，提出Ensemble-CAM方法，通过视觉化方式揭示模型判断真假人脸的关键区域，提升系统的透明度与可信度，属于可解释性研究范畴。**

- **链接: [http://arxiv.org/pdf/2510.19695v1](http://arxiv.org/pdf/2510.19695v1)**

> **作者:** Rashik Shadman; M G Sarwar Murshed; Faraz Hussain
>
> **摘要:** Presentation attacks represent a critical security threat where adversaries use fake biometric data, such as face, fingerprint, or iris images, to gain unauthorized access to protected systems. Various presentation attack detection (PAD) systems have been designed leveraging deep learning (DL) models to mitigate this type of threat. Despite their effectiveness, most of the DL models function as black boxes - their decisions are opaque to their users. The purpose of explainability techniques is to provide detailed information about the reason behind the behavior or decision of DL models. In particular, visual explanation is necessary to better understand the decisions or predictions of DL-based PAD systems and determine the key regions due to which a biometric image is considered real or fake by the system. In this work, a novel technique, Ensemble-CAM, is proposed for providing visual explanations for the decisions made by deep learning-based face PAD systems. Our goal is to improve DL-based face PAD systems by providing a better understanding of their behavior. Our provided visual explanations will enhance the transparency and trustworthiness of DL-based face PAD systems.
>
---
#### [new 018] Exploring "Many in Few" and "Few in Many" Properties in Long-Tailed, Highly-Imbalanced IC Defect Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对集成电路（IC）缺陷分类中的长尾、高度不平衡数据问题，提出新数据集IC-Defect-14和ReCAME-Net模型。解决真实工业场景下样本分布极端不均与类间相似性高的挑战，通过多专家框架、注意力机制与知识蒸馏等技术提升小类缺陷识别性能。**

- **链接: [http://arxiv.org/pdf/2510.19463v1](http://arxiv.org/pdf/2510.19463v1)**

> **作者:** Hao-Chiang Shao; Chun-Hao Chang; Yu-Hsien Lin; Chia-Wen Lin; Shao-Yun Fang; Yan-Hsiu Liu
>
> **摘要:** Despite significant advancements in deep classification techniques and in-lab automatic optical inspection models for long-tailed or highly imbalanced data, applying these approaches to real-world IC defect classification tasks remains challenging. This difficulty stems from two primary factors. First, real-world conditions, such as the high yield-rate requirements in the IC industry, result in data distributions that are far more skewed than those found in general public imbalanced datasets. Consequently, classifiers designed for open imbalanced datasets often fail to perform effectively in real-world scenarios. Second, real-world samples exhibit a mix of class-specific attributes and class-agnostic, domain-related features. This complexity adds significant difficulty to the classification process, particularly for highly imbalanced datasets. To address these challenges, this paper introduces the IC-Defect-14 dataset, a large, highly imbalanced IC defect image dataset sourced from AOI systems deployed in real-world IC production lines. This dataset is characterized by its unique "intra-class clusters" property, which presents two major challenges: large intra-class diversity and high inter-class similarity. These characteristics, rarely found simultaneously in existing public datasets, significantly degrade the performance of current state-of-the-art classifiers for highly imbalanced data. To tackle this challenge, we propose ReCAME-Net, which follows a multi-expert classifier framework and integrates a regional channel attention module, metric learning losses, a hard category mining strategy, and a knowledge distillation procedure. Extensive experimental evaluations demonstrate that ReCAME-Net outperforms previous state-of-the-art models on the IC-Defect-14 dataset while maintaining comparable performance and competitiveness on general public datasets.
>
---
#### [new 019] I Spy With My Model's Eye: Visual Search as a Behavioural Test for MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文将视觉搜索范式引入多模态大模型评估，测试其感知能力。针对模型视觉处理的黑箱问题，通过控制变量实验检验模型是否具备人类类似的“突出效应”及多重特征整合能力，发现先进模型在颜色、大小搜索中表现类人，并受自然场景先验影响。研究提出以认知心理学为基础的诊断工具，揭示模型内在感知机制。**

- **链接: [http://arxiv.org/pdf/2510.19678v1](http://arxiv.org/pdf/2510.19678v1)**

> **作者:** John Burden; Jonathan Prunty; Ben Slater; Matthieu Tehenan; Greg Davis; Lucy Cheke
>
> **备注:** Preprint
>
> **摘要:** Multimodal large language models (MLLMs) achieve strong performance on vision-language tasks, yet their visual processing is opaque. Most black-box evaluations measure task accuracy, but reveal little about underlying mechanisms. Drawing on cognitive psychology, we adapt classic visual search paradigms -- originally developed to study human perception -- to test whether MLLMs exhibit the ``pop-out'' effect, where salient visual features are detected independently of distractor set size. Using controlled experiments targeting colour, size and lighting features, we find that advanced MLLMs exhibit human-like pop-out effects in colour or size-based disjunctive (single feature) search, as well as capacity limits for conjunctive (multiple feature) search. We also find evidence to suggest that MLLMs, like humans, incorporate natural scene priors such as lighting direction into object representations. We reinforce our findings using targeted fine-tuning and mechanistic interpretability analyses. Our work shows how visual search can serve as a cognitively grounded diagnostic tool for evaluating perceptual capabilities in MLLMs.
>
---
#### [new 020] Space Object Detection using Multi-frame Temporal Trajectory Completion Method
- **分类: cs.CV**

- **简介: 该论文针对地球静止轨道（GEO）空间目标检测难题，提出基于多帧时序轨迹补全的方法。通过小波变换增强目标高频特征，结合匈牙利算法实现跨帧最优匹配，并设计时序一致性滤波与轨迹精修流程，有效提升检测精度，F₁达90.14%。**

- **链接: [http://arxiv.org/pdf/2510.19220v1](http://arxiv.org/pdf/2510.19220v1)**

> **作者:** Xiaoqing Lan; Biqiao Xin; Bingshu Wang; Han Zhang; Laixian Zhang
>
> **摘要:** Space objects in Geostationary Earth Orbit (GEO) present significant detection challenges in optical imaging due to weak signals, complex stellar backgrounds, and environmental interference. In this paper, we enhance high-frequency features of GEO targets while suppressing background noise at the single-frame level through wavelet transform. Building on this, we propose a multi-frame temporal trajectory completion scheme centered on the Hungarian algorithm for globally optimal cross-frame matching. To effectively mitigate missing and false detections, a series of key steps including temporal matching and interpolation completion, temporal-consistency-based noise filtering, and progressive trajectory refinement are designed in the post-processing pipeline. Experimental results on the public SpotGEO dataset demonstrate the effectiveness of the proposed method, achieving an F_1 score of 90.14%.
>
---
#### [new 021] PRGCN: A Graph Memory Network for Cross-Sequence Pattern Reuse in 3D Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文针对单目3D人体姿态估计中的深度模糊问题，提出PRGCN框架，通过图记忆库实现跨序列运动模式的检索与复用，结合双流架构融合局部时序与全局关系特征，提升姿态估计精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.19475v1](http://arxiv.org/pdf/2510.19475v1)**

> **作者:** Zhuoyang Xie; Yibo Zhao; Hui Huang; Riwei Wang; Zan Gao
>
> **备注:** 29 pages, 6 figures, 6 tables
>
> **摘要:** Monocular 3D human pose estimation remains a fundamentally ill-posed inverse problem due to the inherent depth ambiguity in 2D-to-3D lifting. While contemporary video-based methods leverage temporal context to enhance spatial reasoning, they operate under a critical paradigm limitation: processing each sequence in isolation, thereby failing to exploit the strong structural regularities and repetitive motion patterns that pervade human movement across sequences. This work introduces the Pattern Reuse Graph Convolutional Network (PRGCN), a novel framework that formalizes pose estimation as a problem of pattern retrieval and adaptation. At its core, PRGCN features a graph memory bank that learns and stores a compact set of pose prototypes, encoded as relational graphs, which are dynamically retrieved via an attention mechanism to provide structured priors. These priors are adaptively fused with hard-coded anatomical constraints through a memory-driven graph convolution, ensuring geometrical plausibility. To underpin this retrieval process with robust spatiotemporal features, we design a dual-stream hybrid architecture that synergistically combines the linear-complexity, local temporal modeling of Mamba-based state-space models with the global relational capacity of self-attention. Extensive evaluations on Human3.6M and MPI-INF-3DHP benchmarks demonstrate that PRGCN establishes a new state-of-the-art, achieving an MPJPE of 37.1mm and 13.4mm, respectively, while exhibiting enhanced cross-domain generalization capability. Our work posits that the long-overlooked mechanism of cross-sequence pattern reuse is pivotal to advancing the field, shifting the paradigm from per-sequence optimization towards cumulative knowledge learning.
>
---
#### [new 022] Curvilinear Structure-preserving Unpaired Cross-domain Medical Image Translation
- **分类: cs.CV**

- **简介: 该论文针对无配对医学图像翻译中细长结构（如微血管）易失真的问题，提出CST框架，通过引入拓扑监督的曲线结构提取模块，增强几何一致性。可集成于CycleGAN、UNSB等模型，显著提升光学相干断层扫描血管成像、眼底彩图和冠状动脉造影等多模态图像翻译的结构保真度，实现更可靠的跨域医学图像生成。**

- **链接: [http://arxiv.org/pdf/2510.19679v1](http://arxiv.org/pdf/2510.19679v1)**

> **作者:** Zihao Chen; Yi Zhou; Xudong Jiang; Li Chen; Leopold Schmetterer; Bingyao Tan; Jun Cheng
>
> **摘要:** Unpaired image-to-image translation has emerged as a crucial technique in medical imaging, enabling cross-modality synthesis, domain adaptation, and data augmentation without costly paired datasets. Yet, existing approaches often distort fine curvilinear structures, such as microvasculature, undermining both diagnostic reliability and quantitative analysis. This limitation is consequential in ophthalmic and vascular imaging, where subtle morphological changes carry significant clinical meaning. We propose Curvilinear Structure-preserving Translation (CST), a general framework that explicitly preserves fine curvilinear structures during unpaired translation by integrating structure consistency into the training. Specifically, CST augments baseline models with a curvilinear extraction module for topological supervision. It can be seamlessly incorporated into existing methods. We integrate it into CycleGAN and UNSB as two representative backbones. Comprehensive evaluation across three imaging modalities: optical coherence tomography angiography, color fundus and X-ray coronary angiography demonstrates that CST improves translation fidelity and achieves state-of-the-art performance. By reinforcing geometric integrity in learned mappings, CST establishes a principled pathway toward curvilinear structure-aware cross-domain translation in medical imaging.
>
---
#### [new 023] Addressing the Depth-of-Field Constraint: A New Paradigm for High Resolution Multi-Focus Image Fusion
- **分类: cs.CV**

- **简介: 该论文针对多聚焦图像融合（MFIF）任务，解决光学镜头景深有限导致的局部清晰问题。提出VAEEDOF方法，结合蒸馏变分自编码器与多图融合模块，提升重建质量与效率。构建真实感4K合成数据集MattingMFIF缓解数据稀缺与域差距问题，实现高质量、无伪影的融合结果。**

- **链接: [http://arxiv.org/pdf/2510.19581v1](http://arxiv.org/pdf/2510.19581v1)**

> **作者:** Luca Piano; Peng Huanwen; Radu Ciprian Bilcu
>
> **摘要:** Multi-focus image fusion (MFIF) addresses the depth-of-field (DOF) limitations of optical lenses, where only objects within a specific range appear sharp. Although traditional and deep learning methods have advanced the field, challenges persist, including limited training data, domain gaps from synthetic datasets, and difficulties with regions lacking information. We propose VAEEDOF, a novel MFIF method that uses a distilled variational autoencoder for high-fidelity, efficient image reconstruction. Our fusion module processes up to seven images simultaneously, enabling robust fusion across diverse focus points. To address data scarcity, we introduce MattingMFIF, a new syntetic 4K dataset, simulating realistic DOF effects from real photographs. Our method achieves state-of-the-art results, generating seamless artifact-free fused images and bridging the gap between synthetic and real-world scenarios, offering a significant step forward in addressing complex MFIF challenges. The code, and weights are available here:
>
---
#### [new 024] A Matter of Time: Revealing the Structure of Time in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.IR; cs.MM**

- **简介: 该论文研究视觉语言模型（VLMs）对时间的感知能力，旨在揭示其嵌入空间中时间结构。提出TIME10k基准数据集与评估方法，发现时间信息存在于低维非线性流形上，并据此构建显式“时间线”表示，提升时间推理性能且高效。属于多模态时间理解任务。**

- **链接: [http://arxiv.org/pdf/2510.19559v1](http://arxiv.org/pdf/2510.19559v1)**

> **作者:** Nidham Tekaya; Manuela Waldner; Matthias Zeppelzauer
>
> **摘要:** Large-scale vision-language models (VLMs) such as CLIP have gained popularity for their generalizable and expressive multimodal representations. By leveraging large-scale training data with diverse textual metadata, VLMs acquire open-vocabulary capabilities, solving tasks beyond their training scope. This paper investigates the temporal awareness of VLMs, assessing their ability to position visual content in time. We introduce TIME10k, a benchmark dataset of over 10,000 images with temporal ground truth, and evaluate the time-awareness of 37 VLMs by a novel methodology. Our investigation reveals that temporal information is structured along a low-dimensional, non-linear manifold in the VLM embedding space. Based on this insight, we propose methods to derive an explicit ``timeline'' representation from the embedding space. These representations model time and its chronological progression and thereby facilitate temporal reasoning tasks. Our timeline approaches achieve competitive to superior accuracy compared to a prompt-based baseline while being computationally efficient. All code and data are available at https://tekayanidham.github.io/timeline-page/.
>
---
#### [new 025] DaMo: Data Mixing Optimizer in Fine-tuning Multimodal LLMs for Mobile Phone Agents
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型在手机智能体中多任务协同效率低的问题，提出DaMo数据混合优化器，通过可训练网络预测最优数据配比，提升模型性能。构建首个专用评测基准PhoneAgentBench，实验证明其显著提升多任务表现与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.19336v1](http://arxiv.org/pdf/2510.19336v1)**

> **作者:** Kai Shi; Jun Yang; Ni Yang; Binqiang Pan; Qingsong Xie; Chao Zhang; Zhenyu Yang; Tianhuang Su; Haonan Lu
>
> **摘要:** Mobile Phone Agents (MPAs) have emerged as a promising research direction due to their broad applicability across diverse scenarios. While Multimodal Large Language Models (MLLMs) serve as the foundation for MPAs, their effectiveness in handling multiple mobile phone tasks simultaneously remains limited. Although multitask supervised fine-tuning (SFT) is widely adopted for multitask learning, existing approaches struggle to determine optimal training data compositions for peak performance. To address this challenge, we propose DaMo (Data Mixture Optimizer) - a novel solution employing a trainable network that predicts optimal data mixtures by forecasting downstream task performance for any given dataset ratio. To support comprehensive evaluation, we introduce PhoneAgentBench, the first specialized benchmark to evaluate MLLMs on multimodal mobile phone tasks, comprising 1235 QA pairs spanning diverse real-world industrial mobile application scenarios. Demonstrating strong predictive capability (R^2=0.81) in small-scale pilot experiments, DaMo efficiently extrapolates optimal data mixing configurations. Our results show DaMo achieves a 3.38% performance improvement on PhoneAgentBench compared to alternative methods. Furthermore, extensive experiments across established benchmarks including BFCL-v3, MME-Reasoning, MME-Perception, and OCRBench reveal DaMo's superior generalization, outperforming other approaches by 2.57% in terms of average score. When used solely for MLLM optimization on the BFCL-v3 task, DaMo improves the metrics by 12.47% than other methods. Notably, DaMo maintains robust scalability, preserving its effectiveness when applied to other model architectures. The code and dataset are available at https://github.com/OPPO-Mente-Lab/DaMo.git
>
---
#### [new 026] MedReason-R1: Learning to Reason for CT Diagnosis with Reinforcement Learning and Local Zoom
- **分类: cs.CV**

- **简介: 该论文针对医学CT图像诊断中通用视觉语言模型表现不佳的问题，提出MedReason-R1模型。通过构建84K样本的CT-RATE-VQA数据集，并引入局部放大与强化学习推理框架，实现从粗到细的诊断推理，显著提升诊断准确率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.19626v1](http://arxiv.org/pdf/2510.19626v1)**

> **作者:** Yifan Li; Fenghe Tang; Yingtai Li; Shaohua Kevin Zhou
>
> **备注:** The code, checkpoints, and dataset are available at: https://github.com/Leevan001/MedReason-R1
>
> **摘要:** General-purpose large Vision-Language Models (VLMs) demonstrate strong capabilities in generating detailed descriptions for natural images. However, their performance in the medical domain remains suboptimal, even for relatively straightforward tasks, primarily due to the lack of large-scale, high-quality, specialized medical imaging datasets and the neglect of the diagnostic process that progresses from coarse to fine-grained. To address the first issue, we construct the CT-RATE-VQA dataset, which has 84K QA pairs. For the second issue, we propose MedReason-R1, a medical VLM with explicit reasoning process for disease diagnosis. MedReason-R1 incorporates a novel strategy that embeds zoom-in disease region-of-interest areas into the image, highlighting the crucial role of both global localization and disease-specific details in enhancing the model's diagnostic performance. Furthermore, we introduce the GRPO reinforcement learning framework to MedReason-R1, which enables effective reasoning without relying on costly manual annotations. Compared to recent general-purpose and medical VLMs, MedReason-R1 achieves state-of-the-art performance in CT disease diagnosis while retaining generalization. The code, checkpoints, and dataset are available at: https://github.com/Leevan001/MedReason-R1
>
---
#### [new 027] SFGFusion: Surface Fitting Guided 3D Object Detection with 4D Radar and Camera Fusion
- **分类: cs.CV**

- **简介: 该论文聚焦于自动驾驶中的3D目标检测任务，针对4D雷达点云稀疏、分辨率低导致几何表征差的问题，提出SFGFusion方法。通过图像与雷达联合估计物体二次曲面参数，生成稠密伪点云并引导特征转换至鸟瞰图空间，提升多模态融合效果与深度预测精度，显著增强检测性能。**

- **链接: [http://arxiv.org/pdf/2510.19215v1](http://arxiv.org/pdf/2510.19215v1)**

> **作者:** Xiaozhi Li; Huijun Di; Jian Li; Feng Liu; Wei Liang
>
> **备注:** Submitted to Pattern Recognition
>
> **摘要:** 3D object detection is essential for autonomous driving. As an emerging sensor, 4D imaging radar offers advantages as low cost, long-range detection, and accurate velocity measurement, making it highly suitable for object detection. However, its sparse point clouds and low resolution limit object geometric representation and hinder multi-modal fusion. In this study, we introduce SFGFusion, a novel camera-4D imaging radar detection network guided by surface fitting. By estimating quadratic surface parameters of objects from image and radar data, the explicit surface fitting model enhances spatial representation and cross-modal interaction, enabling more reliable prediction of fine-grained dense depth. The predicted depth serves two purposes: 1) in an image branch to guide the transformation of image features from perspective view (PV) to a unified bird's-eye view (BEV) for multi-modal fusion, improving spatial mapping accuracy; and 2) in a surface pseudo-point branch to generate dense pseudo-point cloud, mitigating the radar point sparsity. The original radar point cloud is also encoded in a separate radar branch. These two point cloud branches adopt a pillar-based method and subsequently transform the features into the BEV space. Finally, a standard 2D backbone and detection head are used to predict object labels and bounding boxes from BEV features. Experimental results show that SFGFusion effectively fuses camera and 4D radar features, achieving superior performance on the TJ4DRadSet and view-of-delft (VoD) object detection benchmarks.
>
---
#### [new 028] Is This Tracker On? A Benchmark Protocol for Dynamic Tracking
- **分类: cs.CV**

- **简介: 该论文提出ITTO基准，用于评估点跟踪方法在真实场景下的性能。针对现有基准缺乏运动复杂性、遮挡和物体多样性的问题，构建了包含高质量标注的动态追踪数据集，并对主流算法进行分析，揭示其在遮挡后重识别等方面的缺陷，推动更鲁棒的跟踪模型发展。**

- **链接: [http://arxiv.org/pdf/2510.19819v1](http://arxiv.org/pdf/2510.19819v1)**

> **作者:** Ilona Demler; Saumya Chauhan; Georgia Gkioxari
>
> **备注:** Project page: https://glab-caltech.github.io/ITTO/
>
> **摘要:** We introduce ITTO, a challenging new benchmark suite for evaluating and diagnosing the capabilities and limitations of point tracking methods. Our videos are sourced from existing datasets and egocentric real-world recordings, with high-quality human annotations collected through a multi-stage pipeline. ITTO captures the motion complexity, occlusion patterns, and object diversity characteristic of real-world scenes -- factors that are largely absent in current benchmarks. We conduct a rigorous analysis of state-of-the-art tracking methods on ITTO, breaking down performance along key axes of motion complexity. Our findings reveal that existing trackers struggle with these challenges, particularly in re-identifying points after occlusion, highlighting critical failure modes. These results point to the need for new modeling approaches tailored to real-world dynamics. We envision ITTO as a foundation testbed for advancing point tracking and guiding the development of more robust tracking algorithms.
>
---
#### [new 029] Mitigating representation bias caused by missing pixels in methane plume detection
- **分类: cs.CV**

- **简介: 该论文针对卫星图像中因云层导致的非随机缺失像素（MNAR）问题，研究其在甲烷泄漏检测任务中引发的表示偏差。提出加权重采样与多重插补方法，消除标签与图像覆盖率间的虚假关联，提升低覆盖率图像中的漏检率，显著降低偏差且不损害模型性能。**

- **链接: [http://arxiv.org/pdf/2510.19478v1](http://arxiv.org/pdf/2510.19478v1)**

> **作者:** Julia Wąsala; Joannes D. Maasakkers; Ilse Aben; Rochelle Schneider; Holger Hoos; Mitra Baratchi
>
> **备注:** Accepted at the MACLEAN workshop at ECML-PKDD 2025
>
> **摘要:** Most satellite images have systematically missing pixels (i.e., missing data not at random (MNAR)) due to factors such as clouds. If not addressed, these missing pixels can lead to representation bias in automated feature extraction models. In this work, we show that spurious association between the label and the number of missing values in methane plume detection can cause the model to associate the coverage (i.e., the percentage of valid pixels in an image) with the label, subsequently under-detecting plumes in low-coverage images. We evaluate multiple imputation approaches to remove the dependence between the coverage and a label. Additionally, we propose a weighted resampling scheme during training that removes the association between the label and the coverage by enforcing class balance in each coverage bin. Our results show that both resampling and imputation can significantly reduce the representation bias without hurting balanced accuracy, precision, or recall. Finally, we evaluate the capability of the debiased models using these techniques in an operational scenario and demonstrate that the debiased models have a higher chance of detecting plumes in low-coverage images.
>
---
#### [new 030] Beyond sparse denoising in frames: minimax estimation with a scattering transform
- **分类: cs.CV**

- **简介: 该论文研究图像去噪任务，针对传统稀疏表示在复杂几何结构（如分段C^α曲线）下表现不佳的问题，提出基于散射变换的联合ℓ¹范数优化去噪方法。通过分析不同子集散射系数的ℓ¹范数，捕捉多种几何正则性，实验证明其可达到所有α≤2下的极小极大最优界，为调和分析与深度网络去噪建立了数学桥梁。**

- **链接: [http://arxiv.org/pdf/2510.19612v1](http://arxiv.org/pdf/2510.19612v1)**

> **作者:** Nathanaël Cuvelle--Magar; Stéphane Mallat
>
> **摘要:** A considerable amount of research in harmonic analysis has been devoted to non-linear estimators of signals contaminated by additive Gaussian noise. They are implemented by thresholding coefficients in a frame, which provide a sparse signal representation, or by minimising their $\ell^1$ norm. However, sparse estimators in frames are not sufficiently rich to adapt to complex signal regularities. For cartoon images whose edges are piecewise $\bf C^\alpha$ curves, wavelet, curvelet and Xlet frames are suboptimal if the Lipschitz exponent $\alpha \leq 2$ is an unknown parameter. Deep convolutional neural networks have recently obtained much better numerical results, which reach the minimax asymptotic bounds for all $\alpha$. Wavelet scattering coefficients have been introduced as simplified convolutional neural network models. They are computed by transforming the modulus of wavelet coefficients with a second wavelet transform. We introduce a denoising estimator by jointly minimising and maximising the $\ell^1$ norms of different subsets of scattering coefficients. We prove that these $\ell^1$ norms capture different types of geometric image regularity. Numerical experiments show that this denoising estimator reaches the minimax asymptotic bound for cartoon images for all Lipschitz exponents $\alpha \leq 2$. We state this numerical result as a mathematical conjecture. It provides a different harmonic analysis approach to suppress noise from signals, and to specify the geometric regularity of functions. It also opens a mathematical bridge between harmonic analysis and denoising estimators with deep convolutional network.
>
---
#### [new 031] Predicting before Reconstruction: A generative prior framework for MRI acceleration
- **分类: cs.CV**

- **简介: 该论文提出一种生成先验框架，用于加速磁共振成像（MRI）。针对传统MRI采集时间长的问题，通过生成模型预测目标对比图像作为先验，指导高倍率欠采样数据的重建。在多组数据上验证，显著优于现有方法，推动从图像重建向预测成像的范式转变。**

- **链接: [http://arxiv.org/pdf/2510.19472v1](http://arxiv.org/pdf/2510.19472v1)**

> **作者:** Juhyung Park; Rokgi Hong; Roh-Eul Yoo; Jaehyeon Koo; Se Young Chun; Seung Hong Choi; Jongho Lee
>
> **备注:** 33 pages, 8figures
>
> **摘要:** Recent advancements in artificial intelligence have created transformative capabilities in image synthesis and generation, enabling diverse research fields to innovate at revolutionary speed and spectrum. In this study, we leverage this generative power to introduce a new paradigm for accelerating Magnetic Resonance Imaging (MRI), introducing a shift from image reconstruction to proactive predictive imaging. Despite being a cornerstone of modern patient care, MRI's lengthy acquisition times limit clinical throughput. Our novel framework addresses this challenge by first predicting a target contrast image, which then serves as a data-driven prior for reconstructing highly under-sampled data. This informative prior is predicted by a generative model conditioned on diverse data sources, such as other contrast images, previously scanned images, acquisition parameters, patient information. We demonstrate this approach with two key applications: (1) reconstructing FLAIR images using predictions from T1w and/or T2w scans, and (2) reconstructing T1w images using predictions from previously acquired T1w scans. The framework was evaluated on internal and multiple public datasets (total 14,921 scans; 1,051,904 slices), including multi-channel k-space data, for a range of high acceleration factors (x4, x8 and x12). The results demonstrate that our prediction-prior reconstruction method significantly outperforms other approaches, including those with alternative or no prior information. Through this framework we introduce a fundamental shift from image reconstruction towards a new paradigm of predictive imaging.
>
---
#### [new 032] Rethinking Driving World Model as Synthetic Data Generator for Perception Tasks
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶感知任务中合成数据利用效率低的问题，提出Dream4Drive框架，通过3D感知引导的视频生成，实现多视角罕见场景的高效合成。该方法显著提升下游感知模型性能，尤其在有限训练周期下表现突出，并发布DriveObj3D大尺度3D资产数据集。**

- **链接: [http://arxiv.org/pdf/2510.19195v1](http://arxiv.org/pdf/2510.19195v1)**

> **作者:** Kai Zeng; Zhanqian Wu; Kaixin Xiong; Xiaobao Wei; Xiangyu Guo; Zhenxin Zhu; Kalok Ho; Lijun Zhou; Bohan Zeng; Ming Lu; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Wentao Zhang
>
> **摘要:** Recent advancements in driving world models enable controllable generation of high-quality RGB videos or multimodal videos. Existing methods primarily focus on metrics related to generation quality and controllability. However, they often overlook the evaluation of downstream perception tasks, which are $\mathbf{really\ crucial}$ for the performance of autonomous driving. Existing methods usually leverage a training strategy that first pretrains on synthetic data and finetunes on real data, resulting in twice the epochs compared to the baseline (real data only). When we double the epochs in the baseline, the benefit of synthetic data becomes negligible. To thoroughly demonstrate the benefit of synthetic data, we introduce Dream4Drive, a novel synthetic data generation framework designed for enhancing the downstream perception tasks. Dream4Drive first decomposes the input video into several 3D-aware guidance maps and subsequently renders the 3D assets onto these guidance maps. Finally, the driving world model is fine-tuned to produce the edited, multi-view photorealistic videos, which can be used to train the downstream perception models. Dream4Drive enables unprecedented flexibility in generating multi-view corner cases at scale, significantly boosting corner case perception in autonomous driving. To facilitate future research, we also contribute a large-scale 3D asset dataset named DriveObj3D, covering the typical categories in driving scenarios and enabling diverse 3D-aware video editing. We conduct comprehensive experiments to show that Dream4Drive can effectively boost the performance of downstream perception models under various training epochs. Project: $\href{https://wm-research.github.io/Dream4Drive/}{this\ https\ URL}$
>
---
#### [new 033] Robust Driving QA through Metadata-Grounded Context and Task-Specific Prompts
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对自动驾驶中的高阶视觉问答任务，提出两阶段视觉语言模型框架。通过多视角图像与历史帧输入，结合链式思维与自一致性推理提升准确性；第二阶段引入场景元数据和任务专用提示，显著增强鲁棒性与性能，在噪声环境下仍保持96%准确率。**

- **链接: [http://arxiv.org/pdf/2510.19001v1](http://arxiv.org/pdf/2510.19001v1)**

> **作者:** Seungjun Yu; Junsung Park; Youngsun Lim; Hyunjung Shim
>
> **摘要:** We present a two-phase vision-language QA system for autonomous driving that answers high-level perception, prediction, and planning questions. In Phase-1, a large multimodal LLM (Qwen2.5-VL-32B) is conditioned on six-camera inputs, a short temporal window of history, and a chain-of-thought prompt with few-shot exemplars. A self-consistency ensemble (multiple sampled reasoning chains) further improves answer reliability. In Phase-2, we augment the prompt with nuScenes scene metadata (object annotations, ego-vehicle state, etc.) and category-specific question instructions (separate prompts for perception, prediction, planning tasks). In experiments on a driving QA benchmark, our approach significantly outperforms the baseline Qwen2.5 models. For example, using 5 history frames and 10-shot prompting in Phase-1 yields 65.1% overall accuracy (vs.62.61% with zero-shot); applying self-consistency raises this to 66.85%. Phase-2 achieves 67.37% overall. Notably, the system maintains 96% accuracy under severe visual corruption. These results demonstrate that carefully engineered prompts and contextual grounding can greatly enhance high-level driving QA with pretrained vision-language models.
>
---
#### [new 034] MobiAct: Efficient MAV Action Recognition Using MobileNetV4 with Contrastive Learning and Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文针对资源受限的微型飞行器（MAV）动作识别任务，提出轻量级框架MobiAct。通过MobileNetV4骨干网络、分阶段正交知识蒸馏与无参数注意力机制，实现高精度低功耗识别，显著提升推理速度与效率。**

- **链接: [http://arxiv.org/pdf/2510.19273v1](http://arxiv.org/pdf/2510.19273v1)**

> **作者:** Zhang Nengbo; Ho Hann Woei
>
> **摘要:** Accurate and efficient recognition of Micro Air Vehicle (MAV) motion is essential for enabling real-time perception and coordination in autonomous aerial swarm. However, most existing approaches rely on large, computationally intensive models that are unsuitable for resource-limited MAV platforms, which results in a trade-off between recognition accuracy and inference speed. To address these challenges, this paper proposes a lightweight MAV action recognition framework, MobiAct, designed to achieve high accuracy with low computational cost. Specifically, MobiAct adopts MobileNetV4 as the backbone network and introduces a Stage-wise Orthogonal Knowledge Distillation (SOKD) strategy to effectively transfer MAV motion features from a teacher network (ResNet18) to a student network, thereby enhancing knowledge transfer efficiency. Furthermore, a parameter-free attention mechanism is integrated into the architecture to improve recognition accuracy without increasing model complexity. In addition, a hybrid loss training strategy is developed to combine multiple loss objectives, which ensures stable and robust optimization during training. Experimental results demonstrate that the proposed MobiAct achieves low-energy and low-computation MAV action recognition, while maintaining the fastest action decoding speed among compared methods. Across all three self-collected datasets, MobiAct achieves an average recognition accuracy of 92.12%, while consuming only 136.16 pJ of energy and processing recognition at a rate of 8.84 actions per second. Notably, MobiAct decodes actions up to 2 times faster than the leading method, with highly comparable recognition accuracy, highlighting its superior efficiency in MAV action recognition.
>
---
#### [new 035] Class-Aware Prototype Learning with Negative Contrast for Test-Time Adaptation of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型在测试时分布偏移下的性能下降问题，提出CPL-NC框架。通过类感知原型缓存与负对比学习，动态维护类别原型并增强类间区分度，仅优化文本原型以提升泛化能力，在15个基准上显著优于现有TTA方法。**

- **链接: [http://arxiv.org/pdf/2510.19802v1](http://arxiv.org/pdf/2510.19802v1)**

> **作者:** Xiaozhen Qiao; Jingkai Zhao; Yuqiu Jiang; Xianda Guo; Zhe Sun; Hongyuan Zhang; Xuelong Li
>
> **摘要:** Vision-Language Models (VLMs) demonstrate impressive zero-shot generalization through large-scale image-text pretraining, yet their performance can drop once the deployment distribution diverges from the training distribution. To address this, Test-Time Adaptation (TTA) methods update models using unlabeled target data. However, existing approaches often ignore two key challenges: prototype degradation in long-tailed distributions and confusion between semantically similar classes. To tackle these issues, we propose \textbf{C}lass-Aware \textbf{P}rototype \textbf{L}earning with \textbf{N}egative \textbf{C}ontrast(\textbf{CPL-NC}), a lightweight TTA framework designed specifically for VLMs to enhance generalization under distribution shifts. CPL-NC introduces a \textit{Class-Aware Prototype Cache} Module that dynamically adjusts per-class capacity based on test-time frequency and activation history, with a rejuvenation mechanism for inactive classes to retain rare-category knowledge. Additionally, a \textit{Negative Contrastive Learning} Mechanism identifies and constrains hard visual-textual negatives to improve class separability. The framework employs asymmetric optimization, refining only textual prototypes while anchoring on stable visual features. Experiments on 15 benchmarks show that CPL-NC consistently outperforms prior TTA methods across both ResNet-50 and ViT-B/16 backbones.
>
---
#### [new 036] Pico-Banana-400K: A Large-Scale Dataset for Text-Guided Image Editing
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出Pico-Banana-400K数据集，用于文本引导图像编辑任务。针对现有数据集规模小、质量低、缺乏真实图像的问题，利用纳米香蕉模型生成40万张高质量编辑对，涵盖单轮、多轮、偏好与指令重写等复杂场景，提升模型训练与评估的多样性与真实性。**

- **链接: [http://arxiv.org/pdf/2510.19808v1](http://arxiv.org/pdf/2510.19808v1)**

> **作者:** Yusu Qian; Eli Bocek-Rivele; Liangchen Song; Jialing Tong; Yinfei Yang; Jiasen Lu; Wenze Hu; Zhe Gan
>
> **摘要:** Recent advances in multimodal models have demonstrated remarkable text-guided image editing capabilities, with systems like GPT-4o and Nano-Banana setting new benchmarks. However, the research community's progress remains constrained by the absence of large-scale, high-quality, and openly accessible datasets built from real images. We introduce Pico-Banana-400K, a comprehensive 400K-image dataset for instruction-based image editing. Our dataset is constructed by leveraging Nano-Banana to generate diverse edit pairs from real photographs in the OpenImages collection. What distinguishes Pico-Banana-400K from previous synthetic datasets is our systematic approach to quality and diversity. We employ a fine-grained image editing taxonomy to ensure comprehensive coverage of edit types while maintaining precise content preservation and instruction faithfulness through MLLM-based quality scoring and careful curation. Beyond single turn editing, Pico-Banana-400K enables research into complex editing scenarios. The dataset includes three specialized subsets: (1) a 72K-example multi-turn collection for studying sequential editing, reasoning, and planning across consecutive modifications; (2) a 56K-example preference subset for alignment research and reward model training; and (3) paired long-short editing instructions for developing instruction rewriting and summarization capabilities. By providing this large-scale, high-quality, and task-rich resource, Pico-Banana-400K establishes a robust foundation for training and benchmarking the next generation of text-guided image editing models.
>
---
#### [new 037] Advances in 4D Representation: Geometry, Motion, and Interaction
- **分类: cs.CV**

- **简介: 该论文聚焦4D表示学习任务，旨在建模随时间演化的3D几何、运动与交互。通过分类讨论几何、运动、交互三类核心表示，分析主流与新兴方法，提出选择与定制4D表示的指导原则，并探讨大模型与数据集在该领域的应用与局限。**

- **链接: [http://arxiv.org/pdf/2510.19255v1](http://arxiv.org/pdf/2510.19255v1)**

> **作者:** Mingrui Zhao; Sauradip Nag; Kai Wang; Aditya Vora; Guangda Ji; Peter Chun; Ali Mahdavi-Amiri; Hao Zhang
>
> **备注:** 21 pages. Project Page: https://mingrui-zhao.github.io/4DRep-GMI/
>
> **摘要:** We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations\/}, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based on three key pillars: geometry, motion, and interaction. Our discourse will not only encompass the most popular representations of today, such as neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS), but also bring attention to relatively under-explored representations in the 4D context, such as structured models and long-range motions. Throughout our survey, we will reprise the role of large language models (LLMs) and video foundational models (VFMs) in a variety of 4D applications, while steering our discussion towards their current limitations and how they can be addressed. We also provide a dedicated coverage on what 4D datasets are currently available, as well as what is lacking, in driving the subfield forward. Project page:https://mingrui-zhao.github.io/4DRep-GMI/
>
---
#### [new 038] BrainMCLIP: Brain Image Decoding with Multi-Layer feature Fusion of CLIP
- **分类: cs.CV**

- **简介: 该论文提出BrainMCLIP，用于从fMRI解码图像，解决传统方法依赖参数庞大的VAE管道、忽略CLIP中间层信息的问题。通过融合CLIP多层特征并遵循视觉系统功能层级，实现高效高精度图像重建，显著减少参数量，提升语义与细节表现。**

- **链接: [http://arxiv.org/pdf/2510.19332v1](http://arxiv.org/pdf/2510.19332v1)**

> **作者:** Tian Xia; Zihan Ma; Xinlong Wang; Qing Liu; Xiaowei He; Tianming Liu; Yudan Ren
>
> **摘要:** Decoding images from fMRI often involves mapping brain activity to CLIP's final semantic layer. To capture finer visual details, many approaches add a parameter-intensive VAE-based pipeline. However, these approaches overlook rich object information within CLIP's intermediate layers and contradicts the brain's functionally hierarchical. We introduce BrainMCLIP, which pioneers a parameter-efficient, multi-layer fusion approach guided by human visual system's functional hierarchy, eliminating the need for such a separate VAE pathway. BrainMCLIP aligns fMRI signals from functionally distinct visual areas (low-/high-level) to corresponding intermediate and final CLIP layers, respecting functional hierarchy. We further introduce a Cross-Reconstruction strategy and a novel multi-granularity loss. Results show BrainMCLIP achieves highly competitive performance, particularly excelling on high-level semantic metrics where it matches or surpasses SOTA(state-of-the-art) methods, including those using VAE pipelines. Crucially, it achieves this with substantially fewer parameters, demonstrating a reduction of 71.7\%(Table.\ref{tab:compare_clip_vae}) compared to top VAE-based SOTA methods, by avoiding the VAE pathway. By leveraging intermediate CLIP features, it effectively captures visual details often missed by CLIP-only approaches, striking a compelling balance between semantic accuracy and detail fidelity without requiring a separate VAE pipeline.
>
---
#### [new 039] $Δ$t-Mamba3D: A Time-Aware Spatio-Temporal State-Space Model for Breast Cancer Risk Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对乳腺癌风险预测中的纵向影像分析任务，解决高分辨率、非均匀时间间隔影像序列建模难题。提出Δt-Mamba3D模型，通过连续时间感知的扫描机制和多尺度3D融合模块，高效捕捉时空特征，显著提升预测性能并保持线性计算复杂度。**

- **链接: [http://arxiv.org/pdf/2510.19003v1](http://arxiv.org/pdf/2510.19003v1)**

> **作者:** Zhengbo Zhou; Dooman Arefan; Margarita Zuley; Shandong Wu
>
> **摘要:** Longitudinal analysis of sequential radiological images is hampered by a fundamental data challenge: how to effectively model a sequence of high-resolution images captured at irregular time intervals. This data structure contains indispensable spatial and temporal cues that current methods fail to fully exploit. Models often compromise by either collapsing spatial information into vectors or applying spatio-temporal models that are computationally inefficient and incompatible with non-uniform time steps. We address this challenge with Time-Aware $\Delta$t-Mamba3D, a novel state-space architecture adapted for longitudinal medical imaging. Our model simultaneously encodes irregular inter-visit intervals and rich spatio-temporal context while remaining computationally efficient. Its core innovation is a continuous-time selective scanning mechanism that explicitly integrates the true time difference between exams into its state transitions. This is complemented by a multi-scale 3D neighborhood fusion module that robustly captures spatio-temporal relationships. In a comprehensive breast cancer risk prediction benchmark using sequential screening mammogram exams, our model shows superior performance, improving the validation c-index by 2-5 percentage points and achieving higher 1-5 year AUC scores compared to established variants of recurrent, transformer, and state-space models. Thanks to its linear complexity, the model can efficiently process long and complex patient screening histories of mammograms, forming a new framework for longitudinal image analysis.
>
---
#### [new 040] PCP-GAN: Property-Constrained Pore-scale image reconstruction via conditional Generative Adversarial Networks
- **分类: cs.CV; cs.LG; physics.geo-ph**

- **简介: 该论文提出PCP-GAN框架，通过条件生成对抗网络实现孔隙尺度图像的可控生成。针对天然岩心样本稀疏与代表性不足的问题，模型联合控制孔隙度与深度参数，生成符合真实地质特征且具高代表性的孔隙图像，显著提升数字岩心物理模拟精度，适用于碳封存、地热能等应用。**

- **链接: [http://arxiv.org/pdf/2510.19465v1](http://arxiv.org/pdf/2510.19465v1)**

> **作者:** Ali Sadeghkhani; Brandon Bennett; Masoud Babaei; Arash Rabbani
>
> **摘要:** Obtaining truly representative pore-scale images that match bulk formation properties remains a fundamental challenge in subsurface characterization, as natural spatial heterogeneity causes extracted sub-images to deviate significantly from core-measured values. This challenge is compounded by data scarcity, where physical samples are only available at sparse well locations. This study presents a multi-conditional Generative Adversarial Network (cGAN) framework that generates representative pore-scale images with precisely controlled properties, addressing both the representativeness challenge and data availability constraints. The framework was trained on thin section samples from four depths (1879.50-1943.50 m) of a carbonate formation, simultaneously conditioning on porosity values and depth parameters within a single unified model. This approach captures both universal pore network principles and depth-specific geological characteristics, from grainstone fabrics with interparticle-intercrystalline porosity to crystalline textures with anhydrite inclusions. The model achieved exceptional porosity control (R^2=0.95) across all formations with mean absolute errors of 0.0099-0.0197. Morphological validation confirmed preservation of critical pore network characteristics including average pore radius, specific surface area, and tortuosity, with statistical differences remaining within acceptable geological tolerances. Most significantly, generated images demonstrated superior representativeness with dual-constraint errors of 1.9-11.3% compared to 36.4-578% for randomly extracted real sub-images. This capability provides transformative tools for subsurface characterization, particularly valuable for carbon storage, geothermal energy, and groundwater management applications where knowing the representative morphology of the pore space is critical for implementing digital rock physics.
>
---
#### [new 041] Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research
- **分类: cs.CV**

- **简介: 该论文针对纸质心电图（ECG）无法用于现代自动化诊断的问题，提出一种开源、全自动的数字化框架。通过处理扫描或拍摄的纸张ECG图像，将其转换为可分析的数字信号，有效应对常见伪影与畸变，在大规模数据集上超越现有技术，助力回顾性研究与AI诊断普及。**

- **链接: [http://arxiv.org/pdf/2510.19590v1](http://arxiv.org/pdf/2510.19590v1)**

> **作者:** Elias Stenhede; Agnar Martin Bjørnstad; Arian Ranjbar
>
> **摘要:** Millions of clinical ECGs exist only as paper scans, making them unusable for modern automated diagnostics. We introduce a fully automated, modular framework that converts scanned or photographed ECGs into digital signals, suitable for both clinical and research applications. The framework is validated on 37,191 ECG images with 1,596 collected at Akershus University Hospital, where the algorithm obtains a mean signal-to-noise ratio of 19.65 dB on scanned papers with common artifacts. It is further evaluated on the Emory Paper Digitization ECG Dataset, comprising 35,595 images, including images with perspective distortion, wrinkles, and stains. The model improves on the state-of-the-art in all subcategories. The full software is released as open-source, promoting reproducibility and further development. We hope the software will contribute to unlocking retrospective ECG archives and democratize access to AI-driven diagnostics.
>
---
#### [new 042] Can You Trust What You See? Alpha Channel No-Box Attacks on Video Object Detection
- **分类: cs.CV; cs.CR**

- **简介: 该论文针对视频目标检测模型的无框（no-box）对抗攻击问题，提出α-Cloak攻击方法。通过操控RGBA视频的透明度通道（alpha channel），将恶意视频与正常视频融合，实现视觉隐蔽的攻击，无需模型信息且无感知痕迹。在多个先进模型上实现100%攻击成功率，揭示了视频感知系统中α通道的安全隐患。**

- **链接: [http://arxiv.org/pdf/2510.19574v1](http://arxiv.org/pdf/2510.19574v1)**

> **作者:** Ariana Yi; Ce Zhou; Liyang Xiao; Qiben Yan
>
> **摘要:** As object detection models are increasingly deployed in cyber-physical systems such as autonomous vehicles (AVs) and surveillance platforms, ensuring their security against adversarial threats is essential. While prior work has explored adversarial attacks in the image domain, those attacks in the video domain remain largely unexamined, especially in the no-box setting. In this paper, we present {\alpha}-Cloak, the first no-box adversarial attack on object detectors that operates entirely through the alpha channel of RGBA videos. {\alpha}-Cloak exploits the alpha channel to fuse a malicious target video with a benign video, resulting in a fused video that appears innocuous to human viewers but consistently fools object detectors. Our attack requires no access to model architecture, parameters, or outputs, and introduces no perceptible artifacts. We systematically study the support for alpha channels across common video formats and playback applications, and design a fusion algorithm that ensures visual stealth and compatibility. We evaluate {\alpha}-Cloak on five state-of-the-art object detectors, a vision-language model, and a multi-modal large language model (Gemini-2.0-Flash), demonstrating a 100% attack success rate across all scenarios. Our findings reveal a previously unexplored vulnerability in video-based perception systems, highlighting the urgent need for defenses that account for the alpha channel in adversarial settings.
>
---
#### [new 043] DARE: A Deformable Adaptive Regularization Estimator for Learning-Based Medical Image Registration
- **分类: cs.CV; cs.NA; math.NA**

- **简介: 该论文针对医学图像配准任务，解决深度学习方法忽视正则化导致的非物理变形问题。提出DARE框架，通过自适应调节弹性正则化，结合应变与剪切能量，并引入折叠抑制机制，提升配准精度与解剖合理性。**

- **链接: [http://arxiv.org/pdf/2510.19353v1](http://arxiv.org/pdf/2510.19353v1)**

> **作者:** Ahsan Raza Siyal; Markus Haltmeier; Ruth Steiger; Malik Galijasevic; Elke Ruth Gizewski; Astrid Ellen Grams
>
> **摘要:** Deformable medical image registration is a fundamental task in medical image analysis. While deep learning-based methods have demonstrated superior accuracy and computational efficiency compared to traditional techniques, they often overlook the critical role of regularization in ensuring robustness and anatomical plausibility. We propose DARE (Deformable Adaptive Regularization Estimator), a novel registration framework that dynamically adjusts elastic regularization based on the gradient norm of the deformation field. Our approach integrates strain and shear energy terms, which are adaptively modulated to balance stability and flexibility. To ensure physically realistic transformations, DARE includes a folding-prevention mechanism that penalizes regions with negative deformation Jacobian. This strategy mitigates non-physical artifacts such as folding, avoids over-smoothing, and improves both registration accuracy and anatomical plausibility
>
---
#### [new 044] How to Evaluate Monocular Depth Estimation?
- **分类: cs.CV**

- **简介: 该论文针对单目深度估计的评估问题，指出现有指标对曲率扰动不敏感，缺乏与人类判断的一致性。为此，提出基于相对表面法向的新度量，结合可视化工具与组合度量方法，提升评估的人类感知一致性。**

- **链接: [http://arxiv.org/pdf/2510.19814v1](http://arxiv.org/pdf/2510.19814v1)**

> **作者:** Siyang Wu; Jack Nugent; Willow Yang; Jia Deng
>
> **摘要:** Monocular depth estimation is an important task with rapid progress, but how to evaluate it remains an open question, as evidenced by a lack of standardization in existing literature and a large selection of evaluation metrics whose trade-offs and behaviors are not well understood. This paper contributes a novel, quantitative analysis of existing metrics in terms of their sensitivity to various types of perturbations of ground truth, emphasizing comparison to human judgment. Our analysis reveals that existing metrics are severely under-sensitive to curvature perturbation such as making flat surfaces wavy. To remedy this, we introduce a new metric based on relative surface normals, along with new depth visualization tools and a principled method to create composite metrics with better human alignment. Code and data are available at: https://github.com/princeton-vl/evalmde.
>
---
#### [new 045] Decomposed Attention Fusion in MLLMs for Training-Free Video Reasoning Segmentation
- **分类: cs.CV**

- **简介: 该论文针对训练-free视频推理分割任务，提出DecAF方法，通过分解注意力融合机制，优化噪声大的原始注意力图，提升与物体区域的对齐性，并结合SAM2实现精细分割。无需微调模型，性能媲美训练型方法。**

- **链接: [http://arxiv.org/pdf/2510.19592v1](http://arxiv.org/pdf/2510.19592v1)**

> **作者:** Su Ho Han; Jeongseok Hyun; Pilhyeon Lee; Minho Shim; Dongyoon Wee; Seon Joo Kim
>
> **备注:** Project page: https://www.jshyun.me/projects/decaf
>
> **摘要:** Multimodal large language models (MLLMs) demonstrate strong video understanding by attending to visual tokens relevant to textual queries. To directly adapt this for localization in a training-free manner, we cast video reasoning segmentation as a video QA task and extract attention maps via rollout mechanism. However, raw attention maps are noisy and poorly aligned with object regions. We propose Decomposed Attention Fusion (DecAF), which refines these maps through two mechanisms: (1) contrastive object-background fusion and (2) complementary video-frame fusion. This method suppresses irrelevant activations and enhances object-focused cues, enabling direct conversion of attention maps into coarse segmentation masks. In addition, we introduce attention-guided SAM2 prompting for obtaining fine-grained masks. Unlike existing methods that jointly train MLLMs with SAM, our method operates entirely without retraining. DecAF outperforms training-free methods and achieves performance comparable to training-based methods on both referring and reasoning VOS benchmarks. The code will be available at https://github.com/HYUNJS/DecAF.
>
---
#### [new 046] Adaptive Distribution-aware Quantization for Mixed-Precision Neural Networks
- **分类: cs.CV**

- **简介: 该论文针对资源受限设备上的神经网络部署问题，提出自适应分布感知量化（ADQ）框架。解决权重分布非均匀和静态码本不匹配难题，通过动态码本更新与敏感度引导的混合精度分配，实现高效低比特量化。实验表明其在ImageNet上显著提升精度。**

- **链接: [http://arxiv.org/pdf/2510.19760v1](http://arxiv.org/pdf/2510.19760v1)**

> **作者:** Shaohang Jia; Zhiyong Huang; Zhi Yu; Mingyang Hou; Shuai Miao; Han Yang
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Quantization-Aware Training (QAT) is a critical technique for deploying deep neural networks on resource-constrained devices. However, existing methods often face two major challenges: the highly non-uniform distribution of activations and the static, mismatched codebooks used in weight quantization. To address these challenges, we propose Adaptive Distribution-aware Quantization (ADQ), a mixed-precision quantization framework that employs a differentiated strategy. The core of ADQ is a novel adaptive weight quantization scheme comprising three key innovations: (1) a quantile-based initialization method that constructs a codebook closely aligned with the initial weight distribution; (2) an online codebook adaptation mechanism based on Exponential Moving Average (EMA) to dynamically track distributional shifts; and (3) a sensitivity-informed strategy for mixed-precision allocation. For activations, we integrate a hardware-friendly non-uniform-to-uniform mapping scheme. Comprehensive experiments validate the effectiveness of our method. On ImageNet, ADQ enables a ResNet-18 to achieve 71.512% Top-1 accuracy with an average bit-width of only 2.81 bits, outperforming state-of-the-art methods under comparable conditions. Furthermore, detailed ablation studies on CIFAR-10 systematically demonstrate the individual contributions of each innovative component, validating the rationale and effectiveness of our design.
>
---
#### [new 047] D2D: Detector-to-Differentiable Critic for Improved Numeracy in Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成中的计数不准问题，提出D2D框架，将非可微的检测器转化为可微批评者，通过软化检测输出引导生成过程，显著提升对象数量准确性，同时保持图像质量与低计算开销。**

- **链接: [http://arxiv.org/pdf/2510.19278v1](http://arxiv.org/pdf/2510.19278v1)**

> **作者:** Nobline Yoo; Olga Russakovsky; Ye Zhu
>
> **备注:** 24 pages, 14 figures
>
> **摘要:** Text-to-image (T2I) diffusion models have achieved strong performance in semantic alignment, yet they still struggle with generating the correct number of objects specified in prompts. Existing approaches typically incorporate auxiliary counting networks as external critics to enhance numeracy. However, since these critics must provide gradient guidance during generation, they are restricted to regression-based models that are inherently differentiable, thus excluding detector-based models with superior counting ability, whose count-via-enumeration nature is non-differentiable. To overcome this limitation, we propose Detector-to-Differentiable (D2D), a novel framework that transforms non-differentiable detection models into differentiable critics, thereby leveraging their superior counting ability to guide numeracy generation. Specifically, we design custom activation functions to convert detector logits into soft binary indicators, which are then used to optimize the noise prior at inference time with pre-trained T2I models. Our extensive experiments on SDXL-Turbo, SD-Turbo, and Pixart-DMD across four benchmarks of varying complexity (low-density, high-density, and multi-object scenarios) demonstrate consistent and substantial improvements in object counting accuracy (e.g., boosting up to 13.7% on D2D-Small, a 400-prompt, low-density benchmark), with minimal degradation in overall image quality and computational overhead.
>
---
#### [new 048] A Training-Free Framework for Open-Vocabulary Image Segmentation and Recognition with EfficientNet and CLIP
- **分类: cs.CV**

- **简介: 该论文提出一种无训练的开放词汇图像分割与识别框架，结合EfficientNetB0与CLIP，通过无监督分割和视觉-语言对齐实现新类别识别。解决开放词汇下模型泛化难题，无需微调即可在多个基准上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2510.19333v1](http://arxiv.org/pdf/2510.19333v1)**

> **作者:** Ying Dai; Wei Yu Chen
>
> **摘要:** This paper presents a novel training-free framework for open-vocabulary image segmentation and object recognition (OVSR), which leverages EfficientNetB0, a convolutional neural network, for unsupervised segmentation and CLIP, a vision-language model, for open-vocabulary object recognition. The proposed framework adopts a two stage pipeline: unsupervised image segmentation followed by segment-level recognition via vision-language alignment. In the first stage, pixel-wise features extracted from EfficientNetB0 are decomposed using singular value decomposition to obtain latent representations, which are then clustered using hierarchical clustering to segment semantically meaningful regions. The number of clusters is adaptively determined by the distribution of singular values. In the second stage, the segmented regions are localized and encoded into image embeddings using the Vision Transformer backbone of CLIP. Text embeddings are precomputed using CLIP's text encoder from category-specific prompts, including a generic something else prompt to support open set recognition. The image and text embeddings are concatenated and projected into a shared latent feature space via SVD to enhance cross-modal alignment. Recognition is performed by computing the softmax over the similarities between the projected image and text embeddings. The proposed method is evaluated on standard benchmarks, including COCO, ADE20K, and PASCAL VOC, achieving state-of-the-art performance in terms of Hungarian mIoU, precision, recall, and F1-score. These results demonstrate the effectiveness, flexibility, and generalizability of the proposed framework.
>
---
#### [new 049] A Novel Approach to Breast Cancer Segmentation using U-Net Model with Attention Mechanisms and FedProx
- **分类: cs.CV; cs.AI; 68U10, 68T07, 68T45, 92C55; I.4.6; I.2.10; I.5.4; J.3**

- **简介: 该论文属于医学图像分割任务，旨在解决非独立同分布（non-IID）超声乳腺癌数据下模型精度与隐私保护的难题。通过结合改进的注意力U-Net与联邦近端（FedProx）方法，提升了肿瘤边界分割准确率，实现96%的高精度，兼顾隐私安全与模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.19118v1](http://arxiv.org/pdf/2510.19118v1)**

> **作者:** Eyad Gad; Mustafa Abou Khatwa; Mustafa A. Elattar; Sahar Selim
>
> **摘要:** Breast cancer is a leading cause of death among women worldwide, emphasizing the need for early detection and accurate diagnosis. As such Ultrasound Imaging, a reliable and cost-effective tool, is used for this purpose, however the sensitive nature of medical data makes it challenging to develop accurate and private artificial intelligence models. A solution is Federated Learning as it is a promising technique for distributed machine learning on sensitive medical data while preserving patient privacy. However, training on non-Independent and non-Identically Distributed (non-IID) local datasets can impact the accuracy and generalization of the trained model, which is crucial for accurate tumour boundary delineation in BC segmentation. This study aims to tackle this challenge by applying the Federated Proximal (FedProx) method to non-IID Ultrasonic Breast Cancer Imaging datasets. Moreover, we focus on enhancing tumour segmentation accuracy by incorporating a modified U-Net model with attention mechanisms. Our approach resulted in a global model with 96% accuracy, demonstrating the effectiveness of our method in enhancing tumour segmentation accuracy while preserving patient privacy. Our findings suggest that FedProx has the potential to be a promising approach for training precise machine learning models on non-IID local medical datasets.
>
---
#### [new 050] AegisRF: Adversarial Perturbations Guided with Sensitivity for Protecting Intellectual Property of Neural Radiance Fields
- **分类: cs.CV**

- **简介: 该论文针对NeRFs的知识产权保护问题，提出AegisRF框架。通过引入可学习的敏感度场，自适应约束几何扰动，在保持渲染质量的同时，向预渲染输出注入对抗性扰动，有效干扰未经授权的下游任务应用。**

- **链接: [http://arxiv.org/pdf/2510.19371v1](http://arxiv.org/pdf/2510.19371v1)**

> **作者:** Woo Jae Kim; Kyu Beom Han; Yoonki Cho; Youngju Na; Junsik Jung; Sooel Son; Sung-eui Yoon
>
> **备注:** BMVC 2025
>
> **摘要:** As Neural Radiance Fields (NeRFs) have emerged as a powerful tool for 3D scene representation and novel view synthesis, protecting their intellectual property (IP) from unauthorized use is becoming increasingly crucial. In this work, we aim to protect the IP of NeRFs by injecting adversarial perturbations that disrupt their unauthorized applications. However, perturbing the 3D geometry of NeRFs can easily deform the underlying scene structure and thus substantially degrade the rendering quality, which has led existing attempts to avoid geometric perturbations or restrict them to explicit spaces like meshes. To overcome this limitation, we introduce a learnable sensitivity to quantify the spatially varying impact of geometric perturbations on rendering quality. Building upon this, we propose AegisRF, a novel framework that consists of a Perturbation Field, which injects adversarial perturbations into the pre-rendering outputs (color and volume density) of NeRF models to fool an unauthorized downstream target model, and a Sensitivity Field, which learns the sensitivity to adaptively constrain geometric perturbations, preserving rendering quality while disrupting unauthorized use. Our experimental evaluations demonstrate the generalized applicability of AegisRF across diverse downstream tasks and modalities, including multi-view image classification and voxel-based 3D localization, while maintaining high visual fidelity. Codes are available at https://github.com/wkim97/AegisRF.
>
---
#### [new 051] X-Ego: Acquiring Team-Level Tactical Situational Awareness via Cross-Egocentric Contrastive Video Representation Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对多智能体战术决策中的团队态势感知问题，提出X-Ego-CS数据集与交叉第一人称对比学习（CECL）方法。通过同步捕捉各玩家视角视频与动作轨迹，利用对比学习对齐队友视觉流，提升单视角下对队友与对手位置的预测能力，推动电竞环境中跨视角多智能体协作研究。**

- **链接: [http://arxiv.org/pdf/2510.19150v1](http://arxiv.org/pdf/2510.19150v1)**

> **作者:** Yunzhe Wang; Soham Hans; Volkan Ustun
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Human team tactics emerge from each player's individual perspective and their ability to anticipate, interpret, and adapt to teammates' intentions. While advances in video understanding have improved the modeling of team interactions in sports, most existing work relies on third-person broadcast views and overlooks the synchronous, egocentric nature of multi-agent learning. We introduce X-Ego-CS, a benchmark dataset consisting of 124 hours of gameplay footage from 45 professional-level matches of the popular e-sports game Counter-Strike 2, designed to facilitate research on multi-agent decision-making in complex 3D environments. X-Ego-CS provides cross-egocentric video streams that synchronously capture all players' first-person perspectives along with state-action trajectories. Building on this resource, we propose Cross-Ego Contrastive Learning (CECL), which aligns teammates' egocentric visual streams to foster team-level tactical situational awareness from an individual's perspective. We evaluate CECL on a teammate-opponent location prediction task, demonstrating its effectiveness in enhancing an agent's ability to infer both teammate and opponent positions from a single first-person view using state-of-the-art video encoders. Together, X-Ego-CS and CECL establish a foundation for cross-egocentric multi-agent benchmarking in esports. More broadly, our work positions gameplay understanding as a testbed for multi-agent modeling and tactical learning, with implications for spatiotemporal reasoning and human-AI teaming in both virtual and real-world domains. Code and dataset are available at https://github.com/HATS-ICT/x-ego.
>
---
#### [new 052] UniHPR: Unified Human Pose Representation via Singular Value Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文提出UniHPR，一种基于奇异值对比学习的统一人体姿态表示方法，旨在对齐图像、2D/3D姿态等多模态人体姿态表示。通过新设计的对比损失函数，提升跨模态对齐效果，在2D/3D姿态估计与检索任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2510.19078v1](http://arxiv.org/pdf/2510.19078v1)**

> **作者:** Zhongyu Jiang; Wenhao Chai; Lei Li; Zhuoran Zhou; Cheng-Yen Yang; Jenq-Neng Hwang
>
> **摘要:** In recent years, there has been a growing interest in developing effective alignment pipelines to generate unified representations from different modalities for multi-modal fusion and generation. As an important component of Human-Centric applications, Human Pose representations are critical in many downstream tasks, such as Human Pose Estimation, Action Recognition, Human-Computer Interaction, Object tracking, etc. Human Pose representations or embeddings can be extracted from images, 2D keypoints, 3D skeletons, mesh models, and lots of other modalities. Yet, there are limited instances where the correlation among all of those representations has been clearly researched using a contrastive paradigm. In this paper, we propose UniHPR, a unified Human Pose Representation learning pipeline, which aligns Human Pose embeddings from images, 2D and 3D human poses. To align more than two data representations at the same time, we propose a novel singular value-based contrastive learning loss, which better aligns different modalities and further boosts performance. To evaluate the effectiveness of the aligned representation, we choose 2D and 3D Human Pose Estimation (HPE) as our evaluation tasks. In our evaluation, with a simple 3D human pose decoder, UniHPR achieves remarkable performance metrics: MPJPE 49.9mm on the Human3.6M dataset and PA-MPJPE 51.6mm on the 3DPW dataset with cross-domain evaluation. Meanwhile, we are able to achieve 2D and 3D pose retrieval with our unified human pose representations in Human3.6M dataset, where the retrieval error is 9.24mm in MPJPE.
>
---
#### [new 053] Online Handwritten Signature Verification Based on Temporal-Spatial Graph Attention Transformer
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动态手写签名验证任务，旨在解决用户间差异与伪造风险导致的高精度验证难题。提出TS-GATR模型，结合图注意力网络与门控循环单元，通过时空图结构建模签名的动态特征与依赖关系，显著提升验证性能。**

- **链接: [http://arxiv.org/pdf/2510.19321v1](http://arxiv.org/pdf/2510.19321v1)**

> **作者:** Hai-jie Yuan; Heng Zhang; Fei Yin
>
> **摘要:** Handwritten signature verification is a crucial aspect of identity authentication, with applications in various domains such as finance and e-commerce. However, achieving high accuracy in signature verification remains challenging due to intra-user variability and the risk of forgery. This paper introduces a novel approach for dynamic signature verification: the Temporal-Spatial Graph Attention Transformer (TS-GATR). TS-GATR combines the Graph Attention Network (GAT) and the Gated Recurrent Unit (GRU) to model both spatial and temporal dependencies in signature data. TS-GATR enhances verification performance by representing signatures as graphs, where each node captures dynamic features (e.g. position, velocity, pressure), and by using attention mechanisms to model their complex relationships. The proposed method further employs a Dual-Graph Attention Transformer (DGATR) module, which utilizes k-step and k-nearest neighbor adjacency graphs to model local and global spatial features, respectively. To capture long-term temporal dependencies, the model integrates GRU, thereby enhancing its ability to learn dynamic features during signature verification. Comprehensive experiments conducted on benchmark datasets such as MSDS and DeepSignDB show that TS-GATR surpasses current state-of-the-art approaches, consistently achieving lower Equal Error Rates (EER) across various scenarios.
>
---
#### [new 054] XBench: A Comprehensive Benchmark for Visual-Language Explanations in Chest Radiography
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出XBench，首个针对胸部X光图像中文本-视觉解释可解释性的系统性基准。针对视觉语言模型（VLMs）在医学影像中接地能力不足的问题，通过跨注意力与相似性定位图评估模型对病灶区域的定位准确性，揭示了模型在小或弥散病灶上表现差、特定数据预训练提升对齐度等关键发现，强调需加强可解释性评估以推动临床应用。**

- **链接: [http://arxiv.org/pdf/2510.19599v1](http://arxiv.org/pdf/2510.19599v1)**

> **作者:** Haozhe Luo; Shelley Zixin Shu; Ziyu Zhou; Sebastian Otalora; Mauricio Reyes
>
> **摘要:** Vision-language models (VLMs) have recently shown remarkable zero-shot performance in medical image understanding, yet their grounding ability, the extent to which textual concepts align with visual evidence, remains underexplored. In the medical domain, however, reliable grounding is essential for interpretability and clinical adoption. In this work, we present the first systematic benchmark for evaluating cross-modal interpretability in chest X-rays across seven CLIP-style VLM variants. We generate visual explanations using cross-attention and similarity-based localization maps, and quantitatively assess their alignment with radiologist-annotated regions across multiple pathologies. Our analysis reveals that: (1) while all VLM variants demonstrate reasonable localization for large and well-defined pathologies, their performance substantially degrades for small or diffuse lesions; (2) models that are pretrained on chest X-ray-specific datasets exhibit improved alignment compared to those trained on general-domain data. (3) The overall recognition ability and grounding ability of the model are strongly correlated. These findings underscore that current VLMs, despite their strong recognition ability, still fall short in clinically reliable grounding, highlighting the need for targeted interpretability benchmarks before deployment in medical practice. XBench code is available at https://github.com/Roypic/Benchmarkingattention
>
---
#### [new 055] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型在机器人场景中多视角空间推理能力不足的问题，提出MV-RoboBench基准，涵盖1.7k问答数据，评估模型在空间理解与机器人执行任务中的表现。结果表明现有模型远低于人类水平，且单视角性能无法保证多视角任务成功。**

- **链接: [http://arxiv.org/pdf/2510.19400v1](http://arxiv.org/pdf/2510.19400v1)**

> **作者:** Zhiyuan Feng; Zhaolu Kang; Qijie Wang; Zhiying Du; Jiongrui Yan; Shubin Shi; Chengbo Yuan; Huizhi Liang; Yu Deng; Qixiu Li; Rushuai Yang; Arctanx An; Leqi Zheng; Weijie Wang; Shawn Chen; Sicheng Xu; Yaobo Liang; Jiaolong Yang; Baining Guo
>
> **备注:** The project and benchmark are publicly available at https://github.com/microsoft/MV-RoboBench
>
> **摘要:** Vision-language models (VLMs) are essential to Embodied AI, enabling robots to perceive, reason, and act in complex environments. They also serve as the foundation for the recent Vision-Language-Action (VLA) models. Yet most evaluations of VLMs focus on single-view settings, leaving their ability to integrate multi-view information underexplored. At the same time, multi-camera setups are increasingly standard in robotic platforms, as they provide complementary perspectives to mitigate occlusion and depth ambiguity. Whether VLMs can effectively leverage such multi-view inputs for robotic reasoning therefore remains an open question. To bridge this gap, we introduce MV-RoboBench, a benchmark specifically designed to evaluate the multi-view spatial reasoning capabilities of VLMs in robotic manipulation. MV-RoboBench consists of 1.7k manually curated QA items across eight subtasks, divided into two primary categories: spatial understanding and robotic execution. We evaluate a diverse set of existing VLMs, including both open-source and closed-source models, along with enhanced versions incorporating CoT-inspired techniques. The results show that state-of-the-art models remain far below human performance, underscoring the substantial challenges VLMs face in multi-view robotic perception. Additionally, our analysis uncovers two key findings: (i) spatial intelligence and robotic task execution are positively correlated in multi-view robotic scenarios; and (ii) strong performance on existing general-purpose single-view spatial understanding benchmarks does not reliably translate to success in the robotic spatial tasks assessed by our benchmark. We release MV-RoboBench as an open resource to foster progress in spatially grounded VLMs and VLAs, providing not only data but also a standardized evaluation protocol for multi-view embodied reasoning.
>
---
#### [new 056] Video Consistency Distance: Enhancing Temporal Consistency for Image-to-Video Generation via Reward-Based Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文针对图像到视频生成任务中视频时序不一致的问题，提出视频一致性距离（VCD）度量方法。通过在频域分析帧特征，增强模型对时序一致性的建模能力，并基于奖励机制进行微调，显著提升生成视频的时序连贯性，同时保持其他质量指标。**

- **链接: [http://arxiv.org/pdf/2510.19193v1](http://arxiv.org/pdf/2510.19193v1)**

> **作者:** Takehiro Aoshima; Yusuke Shinohara; Park Byeongseon
>
> **备注:** 17 pages
>
> **摘要:** Reward-based fine-tuning of video diffusion models is an effective approach to improve the quality of generated videos, as it can fine-tune models without requiring real-world video datasets. However, it can sometimes be limited to specific performances because conventional reward functions are mainly aimed at enhancing the quality across the whole generated video sequence, such as aesthetic appeal and overall consistency. Notably, the temporal consistency of the generated video often suffers when applying previous approaches to image-to-video (I2V) generation tasks. To address this limitation, we propose Video Consistency Distance (VCD), a novel metric designed to enhance temporal consistency, and fine-tune a model with the reward-based fine-tuning framework. To achieve coherent temporal consistency relative to a conditioning image, VCD is defined in the frequency space of video frame features to capture frame information effectively through frequency-domain analysis. Experimental results across multiple I2V datasets demonstrate that fine-tuning a video generation model with VCD significantly enhances temporal consistency without degrading other performance compared to the previous method.
>
---
#### [new 057] Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism
- **分类: cs.CV**

- **简介: 该论文针对异构多智能体协同感知中因传感器与模型差异导致的领域鸿沟问题，提出生成式通信机制GenComm。通过特征生成与轻量级空间对齐，实现无需重训练的无缝协作，显著降低计算开销与参数量，提升系统可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.19618v1](http://arxiv.org/pdf/2510.19618v1)**

> **作者:** Junfei Zhou; Penglin Dai; Quanmin Wei; Bingyi Liu; Xiao Wu; Jianping Wang
>
> **备注:** 19 pages, 10 figures, accepted to NeurIPS 2025
>
> **摘要:** Multi-agent collaboration enhances the perception capabilities of individual agents through information sharing. However, in real-world applications, differences in sensors and models across heterogeneous agents inevitably lead to domain gaps during collaboration. Existing approaches based on adaptation and reconstruction fail to support pragmatic heterogeneous collaboration due to two key limitations: (1) Intrusive retraining of the encoder or core modules disrupts the established semantic consistency among agents; and (2) accommodating new agents incurs high computational costs, limiting scalability. To address these challenges, we present a novel Generative Communication mechanism (GenComm) that facilitates seamless perception across heterogeneous multi-agent systems through feature generation, without altering the original network, and employs lightweight numerical alignment of spatial information to efficiently integrate new agents at minimal cost. Specifically, a tailored Deformable Message Extractor is designed to extract spatial message for each collaborator, which is then transmitted in place of intermediate features. The Spatial-Aware Feature Generator, utilizing a conditional diffusion model, generates features aligned with the ego agent's semantic space while preserving the spatial information of the collaborators. These generated features are further refined by a Channel Enhancer before fusion. Experiments conducted on the OPV2V-H, DAIR-V2X and V2X-Real datasets demonstrate that GenComm outperforms existing state-of-the-art methods, achieving an 81\% reduction in both computational cost and parameter count when incorporating new agents. Our code is available at https://github.com/jeffreychou777/GenComm.
>
---
#### [new 058] Multi-modal Co-learning for Earth Observation: Enhancing single-modality models via modality collaboration
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对地球观测中单模态模型在推理时因数据缺失导致性能下降的问题，提出一种通用的多模态协同学习框架。通过对比与模态判别学习，使模型分离共享与特有信息，在仅使用训练时部分模态数据的情况下，显著提升单模态预测性能，适用于多种遥感任务。**

- **链接: [http://arxiv.org/pdf/2510.19579v1](http://arxiv.org/pdf/2510.19579v1)**

> **作者:** Francisco Mena; Dino Ienco; Cassio F. Dantas; Roberto Interdonato; Andreas Dengel
>
> **备注:** Accepted at the Machine Learning journal, CfP: Discovery Science 2024
>
> **摘要:** Multi-modal co-learning is emerging as an effective paradigm in machine learning, enabling models to collaboratively learn from different modalities to enhance single-modality predictions. Earth Observation (EO) represents a quintessential domain for multi-modal data analysis, wherein diverse remote sensors collect data to sense our planet. This unprecedented volume of data introduces novel challenges. Specifically, the access to the same sensor modalities at both training and inference stages becomes increasingly complex based on real-world constraints affecting remote sensing platforms. In this context, multi-modal co-learning presents a promising strategy to leverage the vast amount of sensor-derived data available at the training stage to improve single-modality models for inference-time deployment. Most current research efforts focus on designing customized solutions for either particular downstream tasks or specific modalities available at the inference stage. To address this, we propose a novel multi-modal co-learning framework capable of generalizing across various tasks without targeting a specific modality for inference. Our approach combines contrastive and modality discriminative learning together to guide single-modality models to structure the internal model manifold into modality-shared and modality-specific information. We evaluate our framework on four EO benchmarks spanning classification and regression tasks across different sensor modalities, where only one of the modalities available during training is accessible at inference time. Our results demonstrate consistent predictive improvements over state-of-the-art approaches from the recent machine learning and computer vision literature, as well as EO-specific methods. The obtained findings validate our framework in the single-modality inference scenarios across a diverse range of EO applications.
>
---
#### [new 059] MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对动态场景重建中现有3D高斯溅射方法性能不稳定的问题，提出MoE-GS框架，通过多专家协同与体积感知像素路由机制，实现更一致的高质量渲染。结合高效推理策略与知识蒸馏，提升性能与效率，是首个将混合专家引入动态高斯溅射的工作。**

- **链接: [http://arxiv.org/pdf/2510.19210v1](http://arxiv.org/pdf/2510.19210v1)**

> **作者:** In-Hwan Jin; Hyeongju Mun; Joonsoo Kim; Kugjin Yun; Kyeongbo Kong
>
> **摘要:** Recent advances in dynamic scene reconstruction have significantly benefited from 3D Gaussian Splatting, yet existing methods show inconsistent performance across diverse scenes, indicating no single approach effectively handles all dynamic challenges. To overcome these limitations, we propose Mixture of Experts for Dynamic Gaussian Splatting (MoE-GS), a unified framework integrating multiple specialized experts via a novel Volume-aware Pixel Router. Our router adaptively blends expert outputs by projecting volumetric Gaussian-level weights into pixel space through differentiable weight splatting, ensuring spatially and temporally coherent results. Although MoE-GS improves rendering quality, the increased model capacity and reduced FPS are inherent to the MoE architecture. To mitigate this, we explore two complementary directions: (1) single-pass multi-expert rendering and gate-aware Gaussian pruning, which improve efficiency within the MoE framework, and (2) a distillation strategy that transfers MoE performance to individual experts, enabling lightweight deployment without architectural changes. To the best of our knowledge, MoE-GS is the first approach incorporating Mixture-of-Experts techniques into dynamic Gaussian splatting. Extensive experiments on the N3V and Technicolor datasets demonstrate that MoE-GS consistently outperforms state-of-the-art methods with improved efficiency. Video demonstrations are available at https://anonymous.4open.science/w/MoE-GS-68BA/.
>
---
#### [new 060] Seabed-Net: A multi-task network for joint bathymetry estimation and seabed classification from remote sensing imagery in shallow waters
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Seabed-Net，一个联合估计浅水深度与海底分类的多任务网络。针对现有方法孤立处理二者导致性能受限的问题，通过双分支编码器、注意力融合与动态权重平衡，实现深度与分类协同优化，在两个海岸站点上显著降低误差并提升精度与空间一致性。**

- **链接: [http://arxiv.org/pdf/2510.19329v1](http://arxiv.org/pdf/2510.19329v1)**

> **作者:** Panagiotis Agrafiotis; Begüm Demir
>
> **备注:** Submitted to ISPRS Journal of Photogrammetry and Remote Sensing
>
> **摘要:** Accurate, detailed, and regularly updated bathymetry, coupled with complex semantic content, is essential for under-mapped shallow-water environments facing increasing climatological and anthropogenic pressures. However, existing approaches that derive either depth or seabed classes from remote sensing imagery treat these tasks in isolation, forfeiting the mutual benefits of their interaction and hindering the broader adoption of deep learning methods. To address these limitations, we introduce Seabed-Net, a unified multi-task framework that simultaneously predicts bathymetry and pixel-based seabed classification from remote sensing imagery of various resolutions. Seabed-Net employs dual-branch encoders for bathymetry estimation and pixel-based seabed classification, integrates cross-task features via an Attention Feature Fusion module and a windowed Swin-Transformer fusion block, and balances objectives through dynamic task uncertainty weighting. In extensive evaluations at two heterogeneous coastal sites, it consistently outperforms traditional empirical models and traditional machine learning regression methods, achieving up to 75\% lower RMSE. It also reduces bathymetric RMSE by 10-30\% compared to state-of-the-art single-task and multi-task baselines and improves seabed classification accuracy up to 8\%. Qualitative analyses further demonstrate enhanced spatial consistency, sharper habitat boundaries, and corrected depth biases in low-contrast regions. These results confirm that jointly modeling depth with both substrate and seabed habitats yields synergistic gains, offering a robust, open solution for integrated shallow-water mapping. Code and pretrained weights are available at https://github.com/pagraf/Seabed-Net.
>
---
#### [new 061] PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对详细图像描述的评估难题，提出PoSh框架，利用场景图引导大模型作为评判者，实现可复现、可解释的细粒度错误定位。构建新数据集DOCENT，验证PoSh优于现有指标，且能有效指导模型训练，推动视觉语言模型在复杂图像描述上的发展。**

- **链接: [http://arxiv.org/pdf/2510.19060v1](http://arxiv.org/pdf/2510.19060v1)**

> **作者:** Amith Ananthram; Elias Stengel-Eskin; Lorena A. Bradford; Julia Demarest; Adam Purvis; Keith Krut; Robert Stein; Rina Elster Pantalony; Mohit Bansal; Kathleen McKeown
>
> **备注:** 24 pages, 9 figures. Metric/benchmark available at https://github.com/amith-ananthram/posh
>
> **摘要:** While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $\rho$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.
>
---
#### [new 062] Vision-Based Mistake Analysis in Procedural Activities: A Review of Advances and Challenges
- **分类: cs.CV**

- **简介: 该论文聚焦视觉引导的程序性任务错误分析，旨在检测与预测结构化操作中的执行错误。通过综述计算机视觉技术在动作识别、预测与理解方面的进展，系统梳理了方法分类、数据集与评估指标，指出视角差异、类内变异等挑战，并提出未来方向如神经符号推理与反事实建模。**

- **链接: [http://arxiv.org/pdf/2510.19292v1](http://arxiv.org/pdf/2510.19292v1)**

> **作者:** Konstantinos Bacharidis; Antonis A. Argyros
>
> **备注:** 21pages, 6 figures, 2 tables
>
> **摘要:** Mistake analysis in procedural activities is a critical area of research with applications spanning industrial automation, physical rehabilitation, education and human-robot collaboration. This paper reviews vision-based methods for detecting and predicting mistakes in structured tasks, focusing on procedural and executional errors. By leveraging advancements in computer vision, including action recognition, anticipation and activity understanding, vision-based systems can identify deviations in task execution, such as incorrect sequencing, use of improper techniques, or timing errors. We explore the challenges posed by intra-class variability, viewpoint differences and compositional activity structures, which complicate mistake detection. Additionally, we provide a comprehensive overview of existing datasets, evaluation metrics and state-of-the-art methods, categorizing approaches based on their use of procedural structure, supervision levels and learning strategies. Open challenges, such as distinguishing permissible variations from true mistakes and modeling error propagation are discussed alongside future directions, including neuro-symbolic reasoning and counterfactual state modeling. This work aims to establish a unified perspective on vision-based mistake analysis in procedural activities, highlighting its potential to enhance safety, efficiency and task performance across diverse domains.
>
---
#### [new 063] Multi-Camera Worker Tracking in Logistics Warehouse Considering Wide-Angle Distortion
- **分类: cs.CV**

- **简介: 该论文属于多摄像头人员追踪任务，旨在解决宽视角相机畸变导致的定位不准问题。通过19个吊顶安装的广角摄像头，基于脚部位置对齐实现跨相机坐标校准，有效降低图像畸变影响，提升追踪精度超20%，并验证了外观特征利用的有效性。**

- **链接: [http://arxiv.org/pdf/2510.19432v1](http://arxiv.org/pdf/2510.19432v1)**

> **作者:** Yuki Mori; Kazuma Kano; Yusuke Asai; Shin Katayama; Kenta Urano; Takuro Yonezawa; Nobuo Kawaguchi
>
> **摘要:** With the spread of e-commerce, the logistics market is growing around the world. Therefore, improving the efficiency of warehouse operations is essential. To achieve this, various approaches have been explored, and among them, the use of digital twins is gaining attention. To make this approach possible, it is necessary to accurately collect the positions of workers in a warehouse and reflect them in a virtual space. However, a single camera has limitations in its field of view, therefore sensing with multiple cameras is necessary. In this study, we explored a method to track workers using 19 wide-angle cameras installed on the ceiling, looking down at the floor of the logistics warehouse. To understand the relationship between the camera coordinates and the actual positions in the warehouse, we performed alignment based on the floor surface. However, due to the characteristics of wide-angle cameras, significant distortion occurs at the edges of the image, particularly in the vertical direction. To address this, the detected worker positions from each camera were aligned based on foot positions, reducing the effects of image distortion, and enabling accurate position alignment across cameras. As a result, we confirmed an improvement of over 20% in tracking accuracy. Furthermore, we compared multiple methods for utilizing appearance features and validated the effectiveness of the proposed approach.
>
---
#### [new 064] MoAlign: Motion-Centric Representation Alignment for Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对文本到视频扩散模型生成视频时运动不连贯、物理不合理的问题，提出MoAlign框架。通过从预训练视频编码器中学习解耦的运动子空间，并对齐扩散模型的潜在特征，使生成视频具备更真实的运动动态，显著提升物理常识性与时间一致性。**

- **链接: [http://arxiv.org/pdf/2510.19022v1](http://arxiv.org/pdf/2510.19022v1)**

> **作者:** Aritra Bhowmik; Denis Korzhenkov; Cees G. M. Snoek; Amirhossein Habibian; Mohsen Ghafoorian
>
> **摘要:** Text-to-video diffusion models have enabled high-quality video synthesis, yet often fail to generate temporally coherent and physically plausible motion. A key reason is the models' insufficient understanding of complex motions that natural videos often entail. Recent works tackle this problem by aligning diffusion model features with those from pretrained video encoders. However, these encoders mix video appearance and dynamics into entangled features, limiting the benefit of such alignment. In this paper, we propose a motion-centric alignment framework that learns a disentangled motion subspace from a pretrained video encoder. This subspace is optimized to predict ground-truth optical flow, ensuring it captures true motion dynamics. We then align the latent features of a text-to-video diffusion model to this new subspace, enabling the generative model to internalize motion knowledge and generate more plausible videos. Our method improves the physical commonsense in a state-of-the-art video diffusion model, while preserving adherence to textual prompts, as evidenced by empirical evaluations on VideoPhy, VideoPhy2, VBench, and VBench-2.0, along with a user study.
>
---
#### [new 065] The Intricate Dance of Prompt Complexity, Quality, Diversity, and Consistency in T2I Models
- **分类: cs.CV**

- **简介: 该论文研究文本到图像模型中提示复杂度对生成数据质量、多样性与一致性的影响。针对提示复杂度如何影响合成数据实用性的关键问题，提出新评估框架并进行大规模实验，发现复杂提示降低多样性与一致性但缩小与真实数据的分布差异；提示扩展方法在多样性和美学上优于真实数据。**

- **链接: [http://arxiv.org/pdf/2510.19557v1](http://arxiv.org/pdf/2510.19557v1)**

> **作者:** Xiaofeng Zhang; Aaron Courville; Michal Drozdzal; Adriana Romero-Soriano
>
> **摘要:** Text-to-image (T2I) models offer great potential for creating virtually limitless synthetic data, a valuable resource compared to fixed and finite real datasets. Previous works evaluate the utility of synthetic data from T2I models on three key desiderata: quality, diversity, and consistency. While prompt engineering is the primary means of interacting with T2I models, the systematic impact of prompt complexity on these critical utility axes remains underexplored. In this paper, we first conduct synthetic experiments to motivate the difficulty of generalization w.r.t. prompt complexity and explain the observed difficulty with theoretical derivations. Then, we introduce a new evaluation framework that can compare the utility of real data and synthetic data, and present a comprehensive analysis of how prompt complexity influences the utility of synthetic data generated by commonly used T2I models. We conduct our study across diverse datasets, including CC12M, ImageNet-1k, and DCI, and evaluate different inference-time intervention methods. Our synthetic experiments show that generalizing to more general conditions is harder than the other way round, since the former needs an estimated likelihood that is not learned by diffusion models. Our large-scale empirical experiments reveal that increasing prompt complexity results in lower conditional diversity and prompt consistency, while reducing the synthetic-to-real distribution shift, which aligns with the synthetic experiments. Moreover, current inference-time interventions can augment the diversity of the generations at the expense of moving outside the support of real data. Among those interventions, prompt expansion, by deliberately using a pre-trained language model as a likelihood estimator, consistently achieves the highest performance in both image diversity and aesthetics, even higher than that of real data.
>
---
#### [new 066] From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出政策世界模型（PWM），解决自动驾驶中世界模型与规划脱节的问题。通过整合建模与规划，利用无动作未来状态预测提升规划性能，实现协同状态-动作预测。引入动态并行令牌生成机制，仅用前视摄像头即达到领先效果。**

- **链接: [http://arxiv.org/pdf/2510.19654v1](http://arxiv.org/pdf/2510.19654v1)**

> **作者:** Zhida Zhao; Talas Fu; Yifan Wang; Lijun Wang; Huchuan Lu
>
> **备注:** Accepted by NuerIPS 2025 (Poster)
>
> **摘要:** Despite remarkable progress in driving world models, their potential for autonomous systems remains largely untapped: the world models are mostly learned for world simulation and decoupled from trajectory planning. While recent efforts aim to unify world modeling and planning in a single framework, the synergistic facilitation mechanism of world modeling for planning still requires further exploration. In this work, we introduce a new driving paradigm named Policy World Model (PWM), which not only integrates world modeling and trajectory planning within a unified architecture, but is also able to benefit planning using the learned world knowledge through the proposed action-free future state forecasting scheme. Through collaborative state-action prediction, PWM can mimic the human-like anticipatory perception, yielding more reliable planning performance. To facilitate the efficiency of video forecasting, we further introduce a dynamically enhanced parallel token generation mechanism, equipped with a context-guided tokenizer and an adaptive dynamic focal loss. Despite utilizing only front camera input, our method matches or exceeds state-of-the-art approaches that rely on multi-view and multi-modal inputs. Code and model weights will be released at https://github.com/6550Zhao/Policy-World-Model.
>
---
#### [new 067] SCEESR: Semantic-Control Edge Enhancement for Diffusion-Based Super-Resolution
- **分类: cs.CV**

- **简介: 该论文针对真实图像超分辨率任务，解决一阶段扩散模型结构失真与计算成本高的问题。提出SCEESR框架，利用控制网引入语义边缘引导，结合混合损失函数，在单次推理中提升结构准确性和视觉质量，实现高效高质的超分辨率重建。**

- **链接: [http://arxiv.org/pdf/2510.19272v1](http://arxiv.org/pdf/2510.19272v1)**

> **作者:** Yun Kai Zhuang
>
> **备注:** 10 pages, 5 figures, 3 tables
>
> **摘要:** Real-world image super-resolution (Real-ISR) must handle complex degradations and inherent reconstruction ambiguities. While generative models have improved perceptual quality, a key trade-off remains with computational cost. One-step diffusion models offer speed but often produce structural inaccuracies due to distillation artifacts. To address this, we propose a novel SR framework that enhances a one-step diffusion model using a ControlNet mechanism for semantic edge guidance. This integrates edge information to provide dynamic structural control during single-pass inference. We also introduce a hybrid loss combining L2, LPIPS, and an edge-aware AME loss to optimize for pixel accuracy, perceptual quality, and geometric precision. Experiments show our method effectively improves structural integrity and realism while maintaining the efficiency of one-step generation, achieving a superior balance between output quality and inference speed. The results of test datasets will be published at https://drive.google.com/drive/folders/1amddXQ5orIyjbxHgGpzqFHZ6KTolinJF?usp=drive_link and the related code will be published at https://github.com/ARBEZ-ZEBRA/SCEESR.
>
---
#### [new 068] CARES: Context-Aware Resolution Selector for VLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出CARES，一种上下文感知的图像分辨率选择器，用于降低视觉语言模型（VLMs）的计算开销。针对高分辨率输入导致计算和延迟过高的问题，CARES通过轻量级模型预测任务所需的最低分辨率，在保持性能前提下最高减少80%计算量。**

- **链接: [http://arxiv.org/pdf/2510.19496v1](http://arxiv.org/pdf/2510.19496v1)**

> **作者:** Moshe Kimhi; Nimrod Shabtay; Raja Giryes; Chaim Baskin; Eli Schwartz
>
> **摘要:** Large vision-language models (VLMs) commonly process images at native or high resolution to remain effective across tasks. This inflates visual tokens ofter to 97-99% of total tokens, resulting in high compute and latency, even when low-resolution images would suffice. We introduce \emph{CARES}-a \textbf{C}ontext-\textbf{A}ware \textbf{R}esolution \textbf{S}elector, a lightweight preprocessing module that, given an image-query pair, predicts the \emph{minimal} sufficient input resolution. CARES uses a compact VLM (350M) to extract features and predict when a target pretrained VLM's response converges to its peak ability to answer correctly. Though trained as a discrete classifier over a set of optional resolutions, CARES interpolates continuous resolutions at inference for fine-grained control. Across five multimodal benchmarks spanning documents and natural images, as well as diverse target VLMs, CARES preserves task performance while reducing compute by up to 80%.
>
---
#### [new 069] Advancing Brain Tumor Segmentation via Attention-based 3D U-Net Architecture and Digital Image Processing
- **分类: cs.CV; 68U10, 68T07, 68T45; I.4.6; I.2.10; I.5.4; J.3**

- **简介: 该论文属于医学图像分割任务，旨在提升脑肿瘤在MRI影像中的分割精度。针对边界模糊、形状不规则及数据不平衡问题，提出融合注意力机制的3D U-Net模型，并结合数字图像处理技术优化训练，显著提升了分割性能。**

- **链接: [http://arxiv.org/pdf/2510.19109v1](http://arxiv.org/pdf/2510.19109v1)**

> **作者:** Eyad Gad; Seif Soliman; M. Saeed Darweesh
>
> **摘要:** In the realm of medical diagnostics, rapid advancements in Artificial Intelligence (AI) have significantly yielded remarkable improvements in brain tumor segmentation. Encoder-Decoder architectures, such as U-Net, have played a transformative role by effectively extracting meaningful representations in 3D brain tumor segmentation from Magnetic resonance imaging (MRI) scans. However, standard U-Net models encounter challenges in accurately delineating tumor regions, especially when dealing with irregular shapes and ambiguous boundaries. Additionally, training robust segmentation models on high-resolution MRI data, such as the BraTS datasets, necessitates high computational resources and often faces challenges associated with class imbalance. This study proposes the integration of the attention mechanism into the 3D U-Net model, enabling the model to capture intricate details and prioritize informative regions during the segmentation process. Additionally, a tumor detection algorithm based on digital image processing techniques is utilized to address the issue of imbalanced training data and mitigate bias. This study aims to enhance the performance of brain tumor segmentation, ultimately improving the reliability of diagnosis. The proposed model is thoroughly evaluated and assessed on the BraTS 2020 dataset using various performance metrics to accomplish this goal. The obtained results indicate that the model outperformed related studies, exhibiting dice of 0.975, specificity of 0.988, and sensitivity of 0.995, indicating the efficacy of the proposed model in improving brain tumor segmentation, offering valuable insights for reliable diagnosis in clinical settings.
>
---
#### [new 070] FootFormer: Estimating Stability from Visual Input
- **分类: cs.CV**

- **简介: 该论文提出FootFormer，一种从视觉输入联合预测人体运动动力学的跨模态方法，旨在准确估计足压分布、足接触图及重心（CoM），并实现最优的稳定性预测指标（如重心-支撑面关系）。**

- **链接: [http://arxiv.org/pdf/2510.19170v1](http://arxiv.org/pdf/2510.19170v1)**

> **作者:** Keaton Kraiger; Jingjing Li; Skanda Bharadwaj; Jesse Scott; Robert T. Collins; Yanxi Liu
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** We propose FootFormer, a cross-modality approach for jointly predicting human motion dynamics directly from visual input. On multiple datasets, FootFormer achieves statistically significantly better or equivalent estimates of foot pressure distributions, foot contact maps, and center of mass (CoM), as compared with existing methods that generate one or two of those measures. Furthermore, FootFormer achieves SOTA performance in estimating stability-predictive components (CoP, CoM, BoS) used in classic kinesiology metrics. Code and data are available at https://github.com/keatonkraiger/Vision-to-Stability.git.
>
---
#### [new 071] Unified Reinforcement and Imitation Learning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出统一强化与模仿学习（RIL）框架，用于训练轻量级视觉语言模型。针对大模型在资源受限环境不适用的问题，RIL结合强化学习与对抗模仿学习，使小模型既能模仿大模型的文本生成，又能通过奖励信号持续优化，显著提升性能，缩小与顶尖闭源模型的差距。**

- **链接: [http://arxiv.org/pdf/2510.19307v1](http://arxiv.org/pdf/2510.19307v1)**

> **作者:** Byung-Kwan Lee; Ryo Hachiuma; Yong Man Ro; Yu-Chiang Frank Wang; Yueh-Hua Wu
>
> **备注:** NeurIPS 2025, Project page: https://byungkwanlee.github.io/RIL-page
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress, yet their large scale often renders them impractical for resource-constrained environments. This paper introduces Unified Reinforcement and Imitation Learning (RIL), a novel and efficient training algorithm designed to create powerful, lightweight VLMs. RIL distinctively combines the strengths of reinforcement learning with adversarial imitation learning. This enables smaller student VLMs not only to mimic the sophisticated text generation of large teacher models but also to systematically improve their generative capabilities through reinforcement signals. Key to our imitation framework is an LLM-based discriminator that adeptly distinguishes between student and teacher outputs, complemented by guidance from multiple large teacher VLMs to ensure diverse learning. This unified learning strategy, leveraging both reinforcement and imitation, empowers student models to achieve significant performance gains, making them competitive with leading closed-source VLMs. Extensive experiments on diverse vision-language benchmarks demonstrate that RIL significantly narrows the performance gap with state-of-the-art open- and closed-source VLMs and, in several instances, surpasses them.
>
---
#### [new 072] Malaria Detection from Blood Cell Images Using XceptionNet
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决疟疾人工诊断效率低、易出错的问题。通过对比六种深度卷积网络，发现XceptionNet在疟原虫感染红细胞图像分类中表现最优，准确率达97.55%，验证了深度学习在自动、可靠疟疾检测中的可行性。**

- **链接: [http://arxiv.org/pdf/2510.19182v1](http://arxiv.org/pdf/2510.19182v1)**

> **作者:** Warisa Nusrat; Mostafijur Rahman; Ayatullah Faruk Mollah
>
> **摘要:** Malaria, which primarily spreads with the bite of female anopheles mosquitos, often leads to death of people - specifically children in the age-group of 0-5 years. Clinical experts identify malaria by observing RBCs in blood smeared images with a microscope. Lack of adequate professional knowledge and skills, and most importantly manual involvement may cause incorrect diagnosis. Therefore, computer aided automatic diagnosis stands as a preferred substitute. In this paper, well-demonstrated deep networks have been applied to extract deep intrinsic features from blood cell images and thereafter classify them as malaria infected or healthy cells. Among the six deep convolutional networks employed in this work viz. AlexNet, XceptionNet, VGG-19, Residual Attention Network, DenseNet-121 and Custom-CNN. Residual Attention Network and XceptionNet perform relatively better than the rest on a publicly available malaria cell image dataset. They yield an average accuracy of 97.28% and 97.55% respectively, that surpasses other related methods on the same dataset. These findings highly encourage the reality of deep learning driven method for automatic and reliable detection of malaria while minimizing direct manual involvement.
>
---
#### [new 073] Dimensionality Reduction for Remote Sensing Data Analysis: A Systematic Review of Methods and Applications
- **分类: cs.CV**

- **简介: 该论文属于遥感数据分析任务，针对高维数据带来的稀疏性与“维度灾难”问题，系统综述了特征提取类降维方法。通过梳理现有技术及其在数据压缩、融合、可视化等环节的应用，为后续研究提供指导并指出未充分探索的方向。**

- **链接: [http://arxiv.org/pdf/2510.18935v1](http://arxiv.org/pdf/2510.18935v1)**

> **作者:** Nathan Mankovich; Kai-Hendrik Cohrs; Homer Durand; Vasileios Sitokonstantinou; Tristan Williams; Gustau Camps-Valls
>
> **摘要:** Earth observation involves collecting, analyzing, and processing an ever-growing mass of data. Automatically harvesting information is crucial for addressing significant societal, economic, and environmental challenges, ranging from environmental monitoring to urban planning and disaster management. However, the high dimensionality of these data poses challenges in terms of sparsity, inefficiency, and the curse of dimensionality, which limits the effectiveness of machine learning models. Dimensionality reduction (DR) techniques, specifically feature extraction, address these challenges by preserving essential data properties while reducing complexity and enhancing tasks such as data compression, cleaning, fusion, visualization, anomaly detection, and prediction. This review provides a handbook for leveraging DR across the RS data value chain and identifies opportunities for under-explored DR algorithms and their application in future research.
>
---
#### [new 074] Exploring Scale Shift in Crowd Localization under the Context of Domain Generalization
- **分类: cs.CV**

- **简介: 该论文聚焦于人群定位任务中的尺度偏移问题，针对领域泛化场景下因训练与测试数据头尺度分布差异导致的性能下降。通过构建基准ScaleBench、理论分析及提出Catto算法，系统研究并缓解了尺度偏移的影响，揭示了其复杂性与重要性。**

- **链接: [http://arxiv.org/pdf/2510.19330v1](http://arxiv.org/pdf/2510.19330v1)**

> **作者:** Juncheng Wang; Lei Shang; Ziqi Liu; Wang Lu; Xixu Hu; Zhe Hu; Jindong Wang; Shujun Wang
>
> **摘要:** Crowd localization plays a crucial role in visual scene understanding towards predicting each pedestrian location in a crowd, thus being applicable to various downstream tasks. However, existing approaches suffer from significant performance degradation due to discrepancies in head scale distributions (scale shift) between training and testing data, a challenge known as domain generalization (DG). This paper aims to comprehend the nature of scale shift within the context of domain generalization for crowd localization models. To this end, we address four critical questions: (i) How does scale shift influence crowd localization in a DG scenario? (ii) How can we quantify this influence? (iii) What causes this influence? (iv) How to mitigate the influence? Initially, we conduct a systematic examination of how crowd localization performance varies with different levels of scale shift. Then, we establish a benchmark, ScaleBench, and reproduce 20 advanced DG algorithms to quantify the influence. Through extensive experiments, we demonstrate the limitations of existing algorithms and underscore the importance and complexity of scale shift, a topic that remains insufficiently explored. To deepen our understanding, we provide a rigorous theoretical analysis on scale shift. Building on these insights, we further propose an effective algorithm called Causal Feature Decomposition and Anisotropic Processing (Catto) to mitigate the influence of scale shift in DG settings. Later, we also provide extensive analytical experiments, revealing four significant insights for future research. Our results emphasize the importance of this novel and applicable research direction, which we term Scale Shift Domain Generalization.
>
---
#### [new 075] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning
- **分类: cs.CV**

- **简介: 该论文针对视频时刻检索任务，解决数据稀缺、边界模糊与细粒度语义区分困难问题。提出零依赖的两阶段框架AMR，通过数据增强与冷启动+蒸馏训练，提升边界与语义判别能力，有效避免知识遗忘并增强实际场景泛化性。**

- **链接: [http://arxiv.org/pdf/2510.19622v1](http://arxiv.org/pdf/2510.19622v1)**

> **作者:** Zhengxuan Wei; Jiajin Tang; Sibei Yang
>
> **备注:** This work is accepted by ICCV 2025
>
> **摘要:** Existing Moment Retrieval methods face three critical bottlenecks: (1) data scarcity forces models into shallow keyword-feature associations; (2) boundary ambiguity in transition regions between adjacent events; (3) insufficient discrimination of fine-grained semantics (e.g., distinguishing ``kicking" vs. ``throwing" a ball). In this paper, we propose a zero-external-dependency Augmented Moment Retrieval framework, AMR, designed to overcome local optima caused by insufficient data annotations and the lack of robust boundary and semantic discrimination capabilities. AMR is built upon two key insights: (1) it resolves ambiguous boundary information and semantic confusion in existing annotations without additional data (avoiding costly manual labeling), and (2) it preserves boundary and semantic discriminative capabilities enhanced by training while generalizing to real-world scenarios, significantly improving performance. Furthermore, we propose a two-stage training framework with cold-start and distillation adaptation. The cold-start stage employs curriculum learning on augmented data to build foundational boundary/semantic awareness. The distillation stage introduces dual query sets: Original Queries maintain DETR-based localization using frozen Base Queries from the cold-start model, while Active Queries dynamically adapt to real-data distributions. A cross-stage distillation loss enforces consistency between Original and Base Queries, preventing knowledge forgetting while enabling real-world generalization. Experiments on multiple benchmarks show that AMR achieves improved performance over prior state-of-the-art approaches.
>
---
#### [new 076] MetaCluster: Enabling Deep Compression of Kolmogorov-Arnold Network
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对Kolmogorov-Arnold网络（KAN）参数量大、存储成本高的问题，提出MetaCluster框架。通过引入轻量级元学习器引导系数向量聚集于低维流形，再用K-means聚类压缩为共享中心点，实现高达80倍的参数压缩，且不损失精度。属于模型压缩任务。**

- **链接: [http://arxiv.org/pdf/2510.19105v1](http://arxiv.org/pdf/2510.19105v1)**

> **作者:** Matthew Raffel; Adwaith Renjith; Lizhong Chen
>
> **摘要:** Kolmogorov-Arnold Networks (KANs) replace scalar weights with per-edge vectors of basis coefficients, thereby boosting expressivity and accuracy but at the same time resulting in a multiplicative increase in parameters and memory. We propose MetaCluster, a framework that makes KANs highly compressible without sacrificing accuracy. Specifically, a lightweight meta-learner, trained jointly with the KAN, is used to map low-dimensional embedding to coefficient vectors, shaping them to lie on a low-dimensional manifold that is amenable to clustering. We then run K-means in coefficient space and replace per-edge vectors with shared centroids. Afterwards, the meta-learner can be discarded, and a brief fine-tuning of the centroid codebook recovers any residual accuracy loss. The resulting model stores only a small codebook and per-edge indices, exploiting the vector nature of KAN parameters to amortize storage across multiple coefficients. On MNIST, CIFAR-10, and CIFAR-100, across standard KANs and ConvKANs using multiple basis functions, MetaCluster achieves a reduction of up to 80$\times$ in parameter storage, with no loss in accuracy. Code will be released upon publication.
>
---
#### [new 077] GRASPLAT: Enabling dexterous grasping through novel view synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对多指机器人灵巧抓取任务，解决因缺乏高质量3D数据导致的抓取失败问题。提出GRASPLAT框架，仅用RGB图像训练，通过3D高斯点云合成新视角图像，结合光度损失优化抓取姿态预测，显著提升抓取成功率。**

- **链接: [http://arxiv.org/pdf/2510.19200v1](http://arxiv.org/pdf/2510.19200v1)**

> **作者:** Matteo Bortolon; Nuno Ferreira Duarte; Plinio Moreno; Fabio Poiesi; José Santos-Victor; Alessio Del Bue
>
> **备注:** Accepted IROS 2025
>
> **摘要:** Achieving dexterous robotic grasping with multi-fingered hands remains a significant challenge. While existing methods rely on complete 3D scans to predict grasp poses, these approaches face limitations due to the difficulty of acquiring high-quality 3D data in real-world scenarios. In this paper, we introduce GRASPLAT, a novel grasping framework that leverages consistent 3D information while being trained solely on RGB images. Our key insight is that by synthesizing physically plausible images of a hand grasping an object, we can regress the corresponding hand joints for a successful grasp. To achieve this, we utilize 3D Gaussian Splatting to generate high-fidelity novel views of real hand-object interactions, enabling end-to-end training with RGB data. Unlike prior methods, our approach incorporates a photometric loss that refines grasp predictions by minimizing discrepancies between rendered and real images. We conduct extensive experiments on both synthetic and real-world grasping datasets, demonstrating that GRASPLAT improves grasp success rates up to 36.9% over existing image-based methods. Project page: https://mbortolon97.github.io/grasplat/
>
---
#### [new 078] Spatio-temporal Sign Language Representation and Translation
- **分类: cs.CL; cs.CV**

- **简介: 该论文参与WMT-SLT 2022任务，将瑞士德语手语视频翻译为德语文本。针对传统SLT模型忽略时序特征的问题，提出端到端的时空特征表示与翻译模型，统一学习空间与时间信息，提升泛化能力。虽在开发集表现良好，测试集性能显著下降。**

- **链接: [http://arxiv.org/pdf/2510.19413v1](http://arxiv.org/pdf/2510.19413v1)**

> **作者:** Yasser Hamidullah; Josef van Genabith; Cristina España-Bonet
>
> **摘要:** This paper describes the DFKI-MLT submission to the WMT-SLT 2022 sign language translation (SLT) task from Swiss German Sign Language (video) into German (text). State-of-the-art techniques for SLT use a generic seq2seq architecture with customized input embeddings. Instead of word embeddings as used in textual machine translation, SLT systems use features extracted from video frames. Standard approaches often do not benefit from temporal features. In our participation, we present a system that learns spatio-temporal feature representations and translation in a single model, resulting in a real end-to-end architecture expected to better generalize to new data sets. Our best system achieved $5\pm1$ BLEU points on the development set, but the performance on the test dropped to $0.11\pm0.06$ BLEU points.
>
---
#### [new 079] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对长时序、高记忆需求的具身智能任务，提出Memo框架，通过周期性摘要标记实现高效记忆压缩与检索。解决了Transformer在长上下文下内存与计算开销大的问题，提升了模型在复杂环境中的泛化性与实时性表现。**

- **链接: [http://arxiv.org/pdf/2510.19732v1](http://arxiv.org/pdf/2510.19732v1)**

> **作者:** Gunshi Gupta; Karmesh Yadav; Zsolt Kira; Yarin Gal; Rahaf Aljundi
>
> **备注:** Accepted for Spotlight Presentation at NeurIPS 2025
>
> **摘要:** To enable embodied agents to operate effectively over extended timeframes, it is crucial to develop models that form and access memories to stay contextualized in their environment. In the current paradigm of training transformer-based policies for embodied sequential decision-making tasks, visual inputs often overwhelm the context limits of transformers, while humans can maintain and utilize a lifetime of experience compressed as memories. Significant compression is possible in principle, as much of the input is irrelevant and can be abstracted. However, existing approaches predominantly focus on either recurrent models with fixed-size memory or transformers with full-context reliance. In this work, we propose Memo, a transformer-based architecture and training recipe for reinforcement learning (RL) on memory-intensive, long-horizon tasks. Memo incorporates the creation and retrieval of memory by interleaving periodic summarization tokens with the inputs of a model during training. We demonstrate Memo's effectiveness on a gridworld meta-RL benchmark and a multi-object navigation task in photo-realistic indoor settings. Memo outperforms naive long-context transformer baselines while being more compute and storage efficient. Additionally, Memo generalizes better to longer contexts at inference time and remains robust in streaming settings, where historical context must be truncated to fit inference constraints.
>
---
#### [new 080] From See to Shield: ML-Assisted Fine-Grained Access Control for Visual Data
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 该论文针对视觉数据中敏感区域的细粒度访问控制问题，提出基于机器学习的混合加密架构。通过自动检测、重评估与选择性加密敏感对象，结合对称加密与属性基加密，实现高效安全的数据共享。实验表明系统在检测精度与响应速度上均有显著提升。**

- **链接: [http://arxiv.org/pdf/2510.19418v1](http://arxiv.org/pdf/2510.19418v1)**

> **作者:** Mete Harun Akcay; Buse Gul Atli; Siddharth Prakash Rao; Alexandros Bakas
>
> **备注:** 10 pages, 3 figures, 6 tables. In submission
>
> **摘要:** As the volume of stored data continues to grow, identifying and protecting sensitive information within large repositories becomes increasingly challenging, especially when shared with multiple users with different roles and permissions. This work presents a system architecture for trusted data sharing with policy-driven access control, enabling selective protection of sensitive regions while maintaining scalability. The proposed architecture integrates four core modules that combine automated detection of sensitive regions, post-correction, key management, and access control. Sensitive regions are secured using a hybrid scheme that employs symmetric encryption for efficiency and Attribute-Based Encryption for policy enforcement. The system supports efficient key distribution and isolates key storage to strengthen overall security. To demonstrate its applicability, we evaluate the system on visual datasets, where Privacy-Sensitive Objects in images are automatically detected, reassessed, and selectively encrypted prior to sharing in a data repository. Experimental results show that our system provides effective PSO detection, increases macro-averaged F1 score (5%) and mean Average Precision (10%), and maintains an average policy-enforced decryption time of less than 1 second per image. These results demonstrate the effectiveness, efficiency and scalability of our proposed solution for fine-grained access control.
>
---
#### [new 081] Automated Morphological Analysis of Neurons in Fluorescence Microscopy Using YOLOv8
- **分类: eess.IV; cs.CV; q-bio.QM**

- **简介: 该论文针对荧光显微镜下神经元形态分析自动化问题，提出基于YOLOv8的实例分割与测量流水线。利用人工标注数据训练模型，实现高精度分割（>97%）和形态特征提取（准确率75.32%），显著减少人工干预，提升分析效率与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.19455v1](http://arxiv.org/pdf/2510.19455v1)**

> **作者:** Banan Alnemri; Arwa Basbrain
>
> **备注:** 7 pages, 2 figures and 2 tables
>
> **摘要:** Accurate segmentation and precise morphological analysis of neuronal cells in fluorescence microscopy images are crucial steps in neuroscience and biomedical imaging applications. However, this process is labor-intensive and time-consuming, requiring significant manual effort and expertise to ensure reliable outcomes. This work presents a pipeline for neuron instance segmentation and measurement based on a high-resolution dataset of stem-cell-derived neurons. The proposed method uses YOLOv8, trained on manually annotated microscopy images. The model achieved high segmentation accuracy, exceeding 97%. In addition, the pipeline utilized both ground truth and predicted masks to extract biologically significant features, including cell length, width, area, and grayscale intensity values. The overall accuracy of the extracted morphological measurements reached 75.32%, further supporting the effectiveness of the proposed approach. This integrated framework offers a valuable tool for automated analysis in cell imaging and neuroscience research, reducing the need for manual annotation and enabling scalable, precise quantification of neuron morphology.
>
---
#### [new 082] A Survey on Cache Methods in Diffusion Models: Toward Efficient Multi-Modal Generation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文聚焦于扩散模型的高效生成任务，针对其计算开销大、延迟高的问题，提出并系统综述了“缓存”（Diffusion Caching）这一无需训练、适配性强的加速方法。通过复用扩散过程中的冗余计算，实现跨步特征重用与层间调度，提升推理效率，推动实时多模态生成发展。**

- **链接: [http://arxiv.org/pdf/2510.19755v1](http://arxiv.org/pdf/2510.19755v1)**

> **作者:** Jiacheng Liu; Xinyu Wang; Yuqi Lin; Zhikai Wang; Peiru Wang; Peiliang Cai; Qinming Zhou; Zhengan Yan; Zexuan Yan; Zhengyi Shi; Chang Zou; Yue Ma; Linfeng Zhang
>
> **备注:** 22 pages,2 figures
>
> **摘要:** Diffusion Models have become a cornerstone of modern generative AI for their exceptional generation quality and controllability. However, their inherent \textit{multi-step iterations} and \textit{complex backbone networks} lead to prohibitive computational overhead and generation latency, forming a major bottleneck for real-time applications. Although existing acceleration techniques have made progress, they still face challenges such as limited applicability, high training costs, or quality degradation. Against this backdrop, \textbf{Diffusion Caching} offers a promising training-free, architecture-agnostic, and efficient inference paradigm. Its core mechanism identifies and reuses intrinsic computational redundancies in the diffusion process. By enabling feature-level cross-step reuse and inter-layer scheduling, it reduces computation without modifying model parameters. This paper systematically reviews the theoretical foundations and evolution of Diffusion Caching and proposes a unified framework for its classification and analysis. Through comparative analysis of representative methods, we show that Diffusion Caching evolves from \textit{static reuse} to \textit{dynamic prediction}. This trend enhances caching flexibility across diverse tasks and enables integration with other acceleration techniques such as sampling optimization and model distillation, paving the way for a unified, efficient inference framework for future multimodal and interactive applications. We argue that this paradigm will become a key enabler of real-time and efficient generative AI, injecting new vitality into both theory and practice of \textit{Efficient Generative Intelligence}.
>
---
#### [new 083] FrogDeepSDM: Improving Frog Counting and Occurrence Prediction Using Multimodal Data and Pseudo-Absence Imputation
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对物种分布建模中数据稀疏与不完整问题，提出FrogDeepSDM模型，融合多模态数据与伪缺失值填补技术，提升蛙类数量预测与栖息地分布预测精度。通过数据平衡与特征选择优化，构建集成模型，显著降低误差，实现高效精准的生物多样性监测。**

- **链接: [http://arxiv.org/pdf/2510.19305v1](http://arxiv.org/pdf/2510.19305v1)**

> **作者:** Chirag Padubidri; Pranesh Velmurugan; Andreas Lanitis; Andreas Kamilaris
>
> **摘要:** Monitoring species distribution is vital for conservation efforts, enabling the assessment of environmental impacts and the development of effective preservation strategies. Traditional data collection methods, including citizen science, offer valuable insights but remain limited in coverage and completeness. Species Distribution Modelling (SDM) helps address these gaps by using occurrence data and environmental variables to predict species presence across large regions. In this study, we enhance SDM accuracy for frogs (Anura) by applying deep learning and data imputation techniques using data from the "EY - 2022 Biodiversity Challenge." Our experiments show that data balancing significantly improved model performance, reducing the Mean Absolute Error (MAE) from 189 to 29 in frog counting tasks. Feature selection identified key environmental factors influencing occurrence, optimizing inputs while maintaining predictive accuracy. The multimodal ensemble model, integrating land cover, NDVI, and other environmental inputs, outperformed individual models and showed robust generalization across unseen regions. The fusion of image and tabular data improved both frog counting and habitat classification, achieving 84.9% accuracy with an AUC of 0.90. This study highlights the potential of multimodal learning and data preprocessing techniques such as balancing and imputation to improve predictive ecological modeling when data are sparse or incomplete, contributing to more precise and scalable biodiversity monitoring.
>
---
#### [new 084] Learning To Defer To A Population With Limited Demonstrations
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文研究学习去延迟（L2D）任务，针对专家数据稀缺问题，提出基于元学习的半监督框架，利用少量示范生成专家嵌入，构建伪标签训练数据，并实现实时适应新专家。实验表明，模型可快速逼近最优性能，显著提升数据效率与系统实用性。**

- **链接: [http://arxiv.org/pdf/2510.19351v1](http://arxiv.org/pdf/2510.19351v1)**

> **作者:** Nilesh Ramgolam; Gustavo Carneiro; Hsiang-Ting; Chen
>
> **备注:** Accepted to IEEE DICTA 2025 (poster). 7 pages, 2 figures
>
> **摘要:** This paper addresses the critical data scarcity that hinders the practical deployment of learning to defer (L2D) systems to the population. We introduce a context-aware, semi-supervised framework that uses meta-learning to generate expert-specific embeddings from only a few demonstrations. We demonstrate the efficacy of a dual-purpose mechanism, where these embeddings are used first to generate a large corpus of pseudo-labels for training, and subsequently to enable on-the-fly adaptation to new experts at test-time. The experiment results on three different datasets confirm that a model trained on these synthetic labels rapidly approaches oracle-level performance, validating the data efficiency of our approach. By resolving a key training bottleneck, this work makes adaptive L2D systems more practical and scalable, paving the way for human-AI collaboration in real-world environments. To facilitate reproducibility and address implementation details not covered in the main text, we provide our source code and training configurations at https://github.com/nil123532/learning-to-defer-to-a-population-with-limited-demonstrations.
>
---
#### [new 085] GigaBrain-0: A World Model-Powered Vision-Language-Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GigaBrain-0，一种基于世界模型生成数据的视觉-语言-动作（VLA）基础模型，旨在解决真实机器人数据收集成本高、难以泛化的问题。通过生成多样化仿真数据，减少对真实数据依赖，提升跨任务泛化能力与策略鲁棒性，支持复杂操作任务。同时推出轻量版GigaBrain-0-Small，适用于边缘设备。**

- **链接: [http://arxiv.org/pdf/2510.19430v1](http://arxiv.org/pdf/2510.19430v1)**

> **作者:** GigaBrain Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jie Li; Jiagang Zhu; Lv Feng; Peng Li; Qiuping Deng; Runqi Ouyang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yang Wang; Yifan Li; Yilong Li; Yiran Ding; Yuan Xu; Yun Ye; Yukun Zhou; Zhehao Dong; Zhenan Wang; Zhichao Liu; Zheng Zhu
>
> **备注:** https://gigabrain0.github.io/
>
> **摘要:** Training Vision-Language-Action (VLA) models for generalist robots typically requires large-scale real-world robot data, which is expensive and time-consuming to collect. The inefficiency of physical data collection severely limits the scalability, and generalization capacity of current VLA systems. To address this challenge, we introduce GigaBrain-0, a novel VLA foundation model empowered by world model-generated data (e.g., video generation, real2real transfer, human transfer, view transfer, sim2real transfer data). By leveraging world models to generate diverse data at scale, GigaBrain-0 significantly reduces reliance on real robot data while improving cross-task generalization. Our approach further improves policy robustness through RGBD input modeling and embodied Chain-of-Thought (CoT) supervision, enabling the model to reason about spatial geometry, object states, and long-horizon dependencies during task execution. This leads to substantial gains in real-world performance on dexterous, long-horizon, and mobile manipulation tasks. Extensive experiments demonstrate that GigaBrain-0 achieves superior generalization across variations in appearances (e.g., textures, colors), object placements, and camera viewpoints. Additionally, we present GigaBrain-0-Small, an optimized lightweight variant designed to run efficiently on devices such as the NVIDIA Jetson AGX Orin.
>
---
#### [new 086] $\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出$\nabla$-SDF，用于在线、大尺度下高效重建连续可微的欧氏符号距离函数（SDF）。针对传统方法在效率与精度间的权衡问题，结合梯度增强八叉树显式结构与神经隐式残差，实现非截断SDF的高精度、低内存、实时计算，显著提升机器人感知与规划性能。**

- **链接: [http://arxiv.org/pdf/2510.18999v1](http://arxiv.org/pdf/2510.18999v1)**

> **作者:** Zhirui Dai; Qihao Qian; Tianxing Fan; Nikolay Atanasov
>
> **摘要:** Estimation of signed distance functions (SDFs) from point cloud data has been shown to benefit many robot autonomy capabilities, including localization, mapping, motion planning, and control. Methods that support online and large-scale SDF reconstruction tend to rely on discrete volumetric data structures, which affect the continuity and differentiability of the SDF estimates. Recently, using implicit features, neural network methods have demonstrated high-fidelity and differentiable SDF reconstruction but they tend to be less efficient, can experience catastrophic forgetting and memory limitations in large environments, and are often restricted to truncated SDFs. This work proposes $\nabla$-SDF, a hybrid method that combines an explicit prior obtained from gradient-augmented octree interpolation with an implicit neural residual. Our method achieves non-truncated (Euclidean) SDF reconstruction with computational and memory efficiency comparable to volumetric methods and differentiability and accuracy comparable to neural network methods. Extensive experiments demonstrate that \methodname{} outperforms the state of the art in terms of accuracy and efficiency, providing a scalable solution for downstream tasks in robotics and computer vision.
>
---
#### [new 087] Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark
- **分类: cs.CL; cs.AI; cs.CV; cs.DL**

- **简介: 该论文提出从混合语言历史文献中检测拉丁文片段的多模态任务，针对复杂版式文档，构建了724页标注数据集，评估大模型性能。研究首次系统分析了主流模型在该任务中的能力与局限，证明了现代模型实现可靠拉丁文检测的可行性。**

- **链接: [http://arxiv.org/pdf/2510.19585v1](http://arxiv.org/pdf/2510.19585v1)**

> **作者:** Yu Wu; Ke Shu; Jonas Fischer; Lidia Pivovarova; David Rosson; Eetu Mäkelä; Mikko Tolonen
>
> **备注:** Under review. Both the dataset and code will be published
>
> **摘要:** This paper presents a novel task of extracting Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary models is achievable. Our study provides the first comprehensive analysis of these models' capabilities and limits for this task.
>
---
## 更新

#### [replaced 001] How many samples to label for an application given a foundation model? Chest X-ray classification study
- **分类: cs.CV; 68T07 (Primary) 68T45, 62H30, 62P10 (Secondary)**

- **链接: [http://arxiv.org/pdf/2510.11553v2](http://arxiv.org/pdf/2510.11553v2)**

> **作者:** Nikolay Nechaev; Evgeniia Przhezdzetskaia; Viktor Gombolevskiy; Dmitry Umerenkov; Dmitry Dylov
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Chest X-ray classification is vital yet resource-intensive, typically demanding extensive annotated data for accurate diagnosis. Foundation models mitigate this reliance, but how many labeled samples are required remains unclear. We systematically evaluate the use of power-law fits to predict the training size necessary for specific ROC-AUC thresholds. Testing multiple pathologies and foundation models, we find XrayCLIP and XraySigLIP achieve strong performance with significantly fewer labeled examples than a ResNet-50 baseline. Importantly, learning curve slopes from just 50 labeled cases accurately forecast final performance plateaus. Our results enable practitioners to minimize annotation costs by labeling only the essential samples for targeted performance.
>
---
#### [replaced 002] Spiking Neural Networks Need High Frequency Information
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18608v3](http://arxiv.org/pdf/2505.18608v3)**

> **作者:** Yuetong Fang; Deming Zhou; Ziqing Wang; Hongwei Ren; ZeCui Zeng; Lusong Li; Shibo Zhou; Renjing Xu
>
> **摘要:** Spiking Neural Networks promise brain-inspired and energy-efficient computation by transmitting information through binary (0/1) spikes. Yet, their performance still lags behind that of artificial neural networks, often assumed to result from information loss caused by sparse and binary activations. In this work, we challenge this long-standing assumption and reveal a previously overlooked frequency bias: spiking neurons inherently suppress high-frequency components and preferentially propagate low-frequency information. This frequency-domain imbalance, we argue, is the root cause of degraded feature representation in SNNs. Empirically, on Spiking Transformers, adopting Avg-Pooling (low-pass) for token mixing lowers performance to 76.73% on Cifar-100, whereas replacing it with Max-Pool (high-pass) pushes the top-1 accuracy to 79.12%. Accordingly, we introduce Max-Former that restores high-frequency signals through two frequency-enhancing operators: (1) extra Max-Pool in patch embedding, and (2) Depth-Wise Convolution in place of self-attention. Notably, Max-Former attains 82.39% top-1 accuracy on ImageNet using only 63.99M parameters, surpassing Spikformer (74.81%, 66.34M) by +7.58%. Extending our insight beyond transformers, our Max-ResNet-18 achieves state-of-the-art performance on convolution-based benchmarks: 97.17% on CIFAR-10 and 83.06\% on CIFAR-100. We hope this simple yet effective solution inspires future research to explore the distinctive nature of spiking neural networks. Code is available: https://github.com/bic-L/MaxFormer.
>
---
#### [replaced 003] FeatureFool: Zero-Query Fooling of Video Models via Feature Map
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.18362v2](http://arxiv.org/pdf/2510.18362v2)**

> **作者:** Duoxun Tang; Xi Xiao; Guangwu Hu; Kangkang Sun; Xiao Yang; Dongyang Chen; Qing Li; Yongjie Yin; Jiyao Wang
>
> **摘要:** The vulnerability of deep neural networks (DNNs) has been preliminarily verified. Existing black-box adversarial attacks usually require multi-round interaction with the model and consume numerous queries, which is impractical in the real-world and hard to scale to recently emerged Video-LLMs. Moreover, no attack in the video domain directly leverages feature maps to shift the clean-video feature space. We therefore propose FeatureFool, a stealthy, video-domain, zero-query black-box attack that utilizes information extracted from a DNN to alter the feature space of clean videos. Unlike query-based methods that rely on iterative interaction, FeatureFool performs a zero-query attack by directly exploiting DNN-extracted information. This efficient approach is unprecedented in the video domain. Experiments show that FeatureFool achieves an attack success rate above 70\% against traditional video classifiers without any queries. Benefiting from the transferability of the feature map, it can also craft harmful content and bypass Video-LLM recognition. Additionally, adversarial videos generated by FeatureFool exhibit high quality in terms of SSIM, PSNR, and Temporal-Inconsistency, making the attack barely perceptible. This paper may contain violent or explicit content.
>
---
#### [replaced 004] ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.18795v2](http://arxiv.org/pdf/2510.18795v2)**

> **作者:** Xiaoxing Hu; Kaicheng Yang; Ziyang Gong; Qi Ming; Zonghao Guo; Xiang An; Ziyong Feng; Junchi Yan; Xue Yang
>
> **备注:** 17 pages, 5 fiugres
>
> **摘要:** The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP.
>
---
#### [replaced 005] FairGen: Controlling Sensitive Attributes for Fair Generations in Diffusion Models via Adaptive Latent Guidance
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01872v2](http://arxiv.org/pdf/2503.01872v2)**

> **作者:** Mintong Kang; Vinayshekhar Bannihatti Kumar; Shamik Roy; Abhishek Kumar; Sopan Khosla; Balakrishnan Murali Narayanaswamy; Rashmi Gangadharaiah
>
> **备注:** EMNLP 2025 Main Conference (Camera Ready)
>
> **摘要:** Text-to-image diffusion models often exhibit biases toward specific demographic groups, such as generating more males than females when prompted to generate images of engineers, raising ethical concerns and limiting their adoption. In this paper, we tackle the challenge of mitigating generation bias towards any target attribute value (e.g., "male" for "gender") in diffusion models while preserving generation quality. We propose FairGen, an adaptive latent guidance mechanism which controls the generation distribution during inference. In FairGen, a latent guidance module dynamically adjusts the diffusion process to enforce specific attributes, while a memory module tracks the generation statistics and steers latent guidance to align with the targeted fair distribution of the attribute values. Furthermore, we address the limitations of existing datasets by introducing the Holistic Bias Evaluation (HBE) benchmark, which covers diverse domains and incorporates complex prompts to assess bias more comprehensively. Extensive evaluations on HBE and Stable Bias datasets demonstrate that FairGen outperforms existing bias mitigation approaches, achieving substantial bias reduction (e.g., 68.5% gender bias reduction on Stable Diffusion 2). Ablation studies highlight FairGen's ability to flexibly control the output distribution at any user-specified granularity, ensuring adaptive and targeted bias mitigation.
>
---
#### [replaced 006] SparseWorld: A Flexible, Adaptive, and Efficient 4D Occupancy World Model Powered by Sparse and Dynamic Queries
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17482v2](http://arxiv.org/pdf/2510.17482v2)**

> **作者:** Chenxu Dang; Haiyan Liu; Guangjun Bao; Pei An; Xinyue Tang; An Pan; Jie Ma; Bingchuan Sun; Yan Wang
>
> **备注:** Under Review
>
> **摘要:** Semantic occupancy has emerged as a powerful representation in world models for its ability to capture rich spatial semantics. However, most existing occupancy world models rely on static and fixed embeddings or grids, which inherently limit the flexibility of perception. Moreover, their "in-place classification" over grids exhibits a potential misalignment with the dynamic and continuous nature of real scenarios.In this paper, we propose SparseWorld, a novel 4D occupancy world model that is flexible, adaptive, and efficient, powered by sparse and dynamic queries. We propose a Range-Adaptive Perception module, in which learnable queries are modulated by the ego vehicle states and enriched with temporal-spatial associations to enable extended-range perception. To effectively capture the dynamics of the scene, we design a State-Conditioned Forecasting module, which replaces classification-based forecasting with regression-guided formulation, precisely aligning the dynamic queries with the continuity of the 4D environment. In addition, We specifically devise a Temporal-Aware Self-Scheduling training strategy to enable smooth and efficient training. Extensive experiments demonstrate that SparseWorld achieves state-of-the-art performance across perception, forecasting, and planning tasks. Comprehensive visualizations and ablation studies further validate the advantages of SparseWorld in terms of flexibility, adaptability, and efficiency. The code is available at https://github.com/MSunDYY/SparseWorld.
>
---
#### [replaced 007] Probing Perceptual Constancy in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.10273v2](http://arxiv.org/pdf/2502.10273v2)**

> **作者:** Haoran Sun; Bingyang Wang; Suyang Yu; Yijiang Li; Qingying Gao; Haiyun Lyu; Hokin Deng; Dezhi Luo
>
> **备注:** Accepted by ICML 2025 Workshop Building Physically Plausible World Models
>
> **摘要:** Perceptual constancy is the ability to maintain stable perceptions of objects despite changes in sensory input, such as variations in distance, angle, or lighting. This ability is crucial for visual understanding in a dynamic world. Here, we explored such ability in current Vision Language Models (VLMs). In this study, we evaluated 155 VLMs using 236 experiments across three domains: color, size, and shape constancy. The experiments included single-image and video adaptations of classic cognitive tasks, along with novel tasks in in-the-wild conditions. We found significant variability in VLM performance across these domains, with model performance in shape constancy clearly dissociated from that of color and size constancy.
>
---
#### [replaced 008] Kolmogorov-Arnold Attention: Is Learnable Attention Better For Vision Transformers?
- **分类: cs.LG; cs.CV; 68T07; I.2.6; I.5.1; I.5.5; I.5.4; I.4.10**

- **链接: [http://arxiv.org/pdf/2503.10632v3](http://arxiv.org/pdf/2503.10632v3)**

> **作者:** Subhajit Maity; Killian Hitsman; Xin Li; Aritra Dutta
>
> **备注:** Preprint, Appendix included
>
> **摘要:** Kolmogorov-Arnold networks (KANs) are a remarkable innovation that consists of learnable activation functions, with the potential to capture more complex relationships from data. Presently, KANs are deployed by replacing multilayer perceptrons (MLPs) in deep networks, including advanced architectures such as vision Transformers (ViTs). This work asks whether KAN could learn token interactions. In this paper, we design the first learnable attention called Kolmogorov-Arnold Attention (KArAt) for ViTs that can operate on any basis, ranging from Fourier, Wavelets, Splines, to Rational Functions. However, learnable activations in the attention cause a memory explosion. To remedy this, we propose a modular version of KArAt that uses a low-rank approximation. By adopting the Fourier basis, Fourier-KArAt and its variants, in some cases, outperform their traditional softmax counterparts, or show comparable performance on CIFAR-10, CIFAR-100, and ImageNet-1K. We also deploy Fourier KArAt to ConViT and Swin-Transformer, and use it in detection and segmentation with ViT-Det. We dissect the performance of these architectures by analyzing their loss landscapes, weight distributions, optimizer paths, attention visualizations, and transferability to other datasets. KArAt's learnable activation yields a better attention score across all ViTs, indicating improved token-to-token interactions and contributing to enhanced inference. Still, its generalizability does not scale with larger ViTs. However, many factors, including the present computing interface, affect the relative performance of parameter- and memory-heavy KArAts. We note that the goal of this paper is not to produce efficient attention or challenge the traditional activations; by designing KArAt, we are the first to show that attention can be learned and encourage researchers to explore KArAt in conjunction with more advanced architectures.
>
---
#### [replaced 009] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12138v5](http://arxiv.org/pdf/2502.12138v5)**

> **作者:** Shangzan Zhang; Jianyuan Wang; Yinghao Xu; Nan Xue; Christian Rupprecht; Xiaowei Zhou; Yujun Shen; Gordon Wetzstein
>
> **摘要:** We present FLARE, a feed-forward model designed to infer high-quality camera poses and 3D geometry from uncalibrated sparse-view images (i.e., as few as 2-8 inputs), which is a challenging yet practical setting in real-world applications. Our solution features a cascaded learning paradigm with camera pose serving as the critical bridge, recognizing its essential role in mapping 3D structures onto 2D image planes. Concretely, FLARE starts with camera pose estimation, whose results condition the subsequent learning of geometric structure and appearance, optimized through the objectives of geometry reconstruction and novel-view synthesis. Utilizing large-scale public datasets for training, our method delivers state-of-the-art performance in the tasks of pose estimation, geometry reconstruction, and novel view synthesis, while maintaining the inference efficiency (i.e., less than 0.5 seconds). The project page and code can be found at: https://zhanghe3z.github.io/FLARE/
>
---
#### [replaced 010] Team Westwood Solution for MIDOG 2025 Challenge: An Ensemble-CNN-Based Approach For Mitosis Detection And Classification
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02600v2](http://arxiv.org/pdf/2509.02600v2)**

> **作者:** Tengyou Xu; Haochen Yang; Xiang 'Anthony' Chen; Hongyan Gu; Mohammad Haeri
>
> **备注:** 3 pages, 2 figures
>
> **摘要:** This abstract presents our solution (Team Westwood) for mitosis detection and atypical mitosis classification in the MItosis DOmain Generalization (MIDOG) 2025 challenge. For mitosis detection, we trained an nnUNetV2 for initial mitosis candidate screening with high sensitivity, followed by a random forest classifier ensembling predictions of three convolutional neural networks (CNNs): EfficientNet-b3, EfficientNet-b5, and EfficientNetV2-s. For the atypical mitosis classification, we trained another random forest classifier ensembling the predictions of three CNNs: EfficientNet-b3, EfficientNet-b5, and InceptionV3. On the preliminary test set, our solution achieved an F1 score of 0.7450 for track 1 mitosis detection, and a balanced accuracy of 0.8722 for track 2 atypical mitosis classification. On the final test set, our solution achieved an F1 score of 0.6972 for track 1 mitosis detection, and a balanced accuracy of 0.8242 for track 2 atypical mitosis classification.
>
---
#### [replaced 011] One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.15591v3](http://arxiv.org/pdf/2506.15591v3)**

> **作者:** Yujing Sun; Lingchen Sun; Shuaizheng Liu; Rongyuan Wu; Zhengqiang Zhang; Lei Zhang
>
> **备注:** Accepted by Neurips2025
>
> **摘要:** It is a challenging problem to reproduce rich spatial details while maintaining temporal consistency in real-world video super-resolution (Real-VSR), especially when we leverage pre-trained generative models such as stable diffusion (SD) for realistic details synthesis. Existing SD-based Real-VSR methods often compromise spatial details for temporal coherence, resulting in suboptimal visual quality. We argue that the key lies in how to effectively extract the degradation-robust temporal consistency priors from the low-quality (LQ) input video and enhance the video details while maintaining the extracted consistency priors. To achieve this, we propose a Dual LoRA Learning (DLoRAL) paradigm to train an effective SD-based one-step diffusion model, achieving realistic frame details and temporal consistency simultaneously. Specifically, we introduce a Cross-Frame Retrieval (CFR) module to aggregate complementary information across frames, and train a Consistency-LoRA (C-LoRA) to learn robust temporal representations from degraded inputs. After consistency learning, we fix the CFR and C-LoRA modules and train a Detail-LoRA (D-LoRA) to enhance spatial details while aligning with the temporal space defined by C-LoRA to keep temporal coherence. The two phases alternate iteratively for optimization, collaboratively delivering consistent and detail-rich outputs. During inference, the two LoRA branches are merged into the SD model, allowing efficient and high-quality video restoration in a single diffusion step. Experiments show that DLoRAL achieves strong performance in both accuracy and speed. Code and models are available at https://github.com/yjsunnn/DLoRAL.
>
---
#### [replaced 012] VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01822v2](http://arxiv.org/pdf/2412.01822v2)**

> **作者:** Byung-Kwan Lee; Ryo Hachiuma; Yu-Chiang Frank Wang; Yong Man Ro; Yueh-Hua Wu
>
> **备注:** CVPR 2025, Project page: https://byungkwanlee.github.io/VLsI-page/
>
> **摘要:** The recent surge in high-quality visual instruction tuning samples from closed-source vision-language models (VLMs) such as GPT-4V has accelerated the release of open-source VLMs across various model sizes. However, scaling VLMs to improve performance using larger models brings significant computational challenges, especially for deployment on resource-constrained devices like mobile platforms and robots. To address this, we propose VLsI: Verbalized Layers-to-Interactions, a new VLM family in 2B and 7B model sizes, which prioritizes efficiency without compromising accuracy. VLsI leverages a unique, layer-wise distillation process, introducing intermediate "verbalizers" that map features from each layer to natural language space, allowing smaller VLMs to flexibly align with the reasoning processes of larger VLMs. This approach mitigates the training instability often encountered in output imitation and goes beyond typical final-layer tuning by aligning the small VLMs' layer-wise progression with that of the large ones. We validate VLsI across ten challenging vision-language benchmarks, achieving notable performance gains (11.0% for 2B and 17.4% for 7B) over GPT-4V without the need for model scaling, merging, or architectural changes.
>
---
#### [replaced 013] Training-Free Label Space Alignment for Universal Domain Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.17452v2](http://arxiv.org/pdf/2509.17452v2)**

> **作者:** Dujin Lee; Sojung An; Jungmyung Wi; Kuniaki Saito; Donghyun Kim
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** Universal domain adaptation (UniDA) transfers knowledge from a labeled source domain to an unlabeled target domain, where label spaces may differ and the target domain may contain private classes. Previous UniDA methods primarily focused on visual space alignment but often struggled with visual ambiguities due to content differences, which limited their robustness and generalizability. To overcome this, we introduce a novel approach that leverages the strong \textit{zero-shot capabilities} of recent vision-language foundation models (VLMs) like CLIP, concentrating solely on label space alignment to enhance adaptation stability. CLIP can generate task-specific classifiers based only on label names. However, adapting CLIP to UniDA is challenging because the label space is not fully known in advance. In this study, we first utilize generative vision-language models to identify unknown categories in the target domain. Noise and semantic ambiguities in the discovered labels -- such as those similar to source labels (e.g., synonyms, hypernyms, hyponyms) -- complicate label alignment. To address this, we propose a training-free label-space alignment method for UniDA (\ours). Our method aligns label spaces instead of visual spaces by filtering and refining noisy labels between the domains. We then construct a \textit{universal classifier} that integrates both shared knowledge and target-private class information, thereby improving generalizability under domain shifts. The results reveal that the proposed method considerably outperforms existing UniDA techniques across key DomainBed benchmarks, delivering an average improvement of \textcolor{blue}{+7.9\%}in H-score and \textcolor{blue}{+6.1\%} in H$^3$-score. Furthermore, incorporating self-training further enhances performance and achieves an additional (\textcolor{blue}{+1.6\%}) increment in both H- and H$^3$-scores.
>
---
#### [replaced 014] Discretized Gaussian Representation for Tomographic Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.04844v4](http://arxiv.org/pdf/2411.04844v4)**

> **作者:** Shaokai Wu; Yuxiang Lu; Yapan Guo; Wei Ji; Suizhi Huang; Fengyu Yang; Shalayiding Sirejiding; Qichen He; Jing Tong; Yanbiao Ji; Yue Ding; Hongtao Lu
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Computed Tomography (CT) enables detailed cross-sectional imaging but continues to face challenges in balancing reconstruction quality and computational efficiency. While deep learning-based methods have significantly improved image quality and noise reduction, they typically require large-scale training data and intensive computation. Recent advances in scene reconstruction, such as Neural Radiance Fields and 3D Gaussian Splatting, offer alternative perspectives but are not well-suited for direct volumetric CT reconstruction. In this work, we propose Discretized Gaussian Representation (DGR), a novel framework that reconstructs the 3D volume directly using a set of discretized Gaussian functions in an end-to-end manner. To further enhance efficiency, we introduce Fast Volume Reconstruction, a highly parallelized technique that aggregates Gaussian contributions into the voxel grid with minimal overhead. Extensive experiments on both real-world and synthetic datasets demonstrate that DGR achieves superior reconstruction quality and runtime performance across various CT reconstruction scenarios. Our code is publicly available at https://github.com/wskingdom/DGR.
>
---
#### [replaced 015] Latent Diffusion Models with Masked AutoEncoders
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09984v3](http://arxiv.org/pdf/2507.09984v3)**

> **作者:** Junho Lee; Jeongwoo Shin; Hyungwook Choi; Joonseok Lee
>
> **摘要:** In spite of the remarkable potential of Latent Diffusion Models (LDMs) in image generation, the desired properties and optimal design of the autoencoders have been underexplored. In this work, we analyze the role of autoencoders in LDMs and identify three key properties: latent smoothness, perceptual compression quality, and reconstruction quality. We demonstrate that existing autoencoders fail to simultaneously satisfy all three properties, and propose Variational Masked AutoEncoders (VMAEs), taking advantage of the hierarchical features maintained by Masked AutoEncoders. We integrate VMAEs into the LDM framework, introducing Latent Diffusion Models with Masked AutoEncoders (LDMAEs). Our code is available at https://github.com/isno0907/ldmae.
>
---
#### [replaced 016] MMLA: Multi-Environment, Multi-Species, Low-Altitude Drone Dataset
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07744v3](http://arxiv.org/pdf/2504.07744v3)**

> **作者:** Jenna Kline; Samuel Stevens; Guy Maalouf; Camille Rondeau Saint-Jean; Dat Nguyen Ngoc; Majid Mirmehdi; David Guerin; Tilo Burghardt; Elzbieta Pastucha; Blair Costelloe; Matthew Watson; Thomas Richardson; Ulrik Pagh Schultz Lundquist
>
> **备注:** Accepted at CVPR Workshop, CV4Animals 2025
>
> **摘要:** Real-time wildlife detection in drone imagery supports critical ecological and conservation monitoring. However, standard detection models like YOLO often fail to generalize across locations and struggle with rare species, limiting their use in automated drone deployments. We present MMLA, a novel multi-environment, multi-species, low-altitude drone dataset collected across three sites (Ol Pejeta Conservancy and Mpala Research Centre in Kenya, and The Wilds in Ohio), featuring six species (zebras, giraffes, onagers, and African wild dogs). The dataset contains 811K annotations from 37 high-resolution videos. Baseline YOLO models show performance disparities across locations while fine-tuning YOLOv11m on MMLA improves mAP50 to 82%, a 52-point gain over baseline. Our results underscore the need for diverse training data to enable robust animal detection in autonomous drone systems.
>
---
#### [replaced 017] kabr-tools: Automated Framework for Multi-Species Behavioral Monitoring
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.02030v2](http://arxiv.org/pdf/2510.02030v2)**

> **作者:** Jenna Kline; Maksim Kholiavchenko; Samuel Stevens; Nina van Tiel; Alison Zhong; Namrata Banerji; Alec Sheets; Sowbaranika Balasubramaniam; Isla Duporge; Matthew Thompson; Elizabeth Campolongo; Jackson Miliko; Neil Rosser; Tanya Berger-Wolf; Charles V. Stewart; Daniel I. Rubenstein
>
> **备注:** 31 pages
>
> **摘要:** A comprehensive understanding of animal behavior ecology depends on scalable approaches to quantify and interpret complex, multidimensional behavioral patterns. Traditional field observations are often limited in scope, time-consuming, and labor-intensive, hindering the assessment of behavioral responses across landscapes. To address this, we present kabr-tools (Kenyan Animal Behavior Recognition Tools), an open-source package for automated multi-species behavioral monitoring. This framework integrates drone-based video with machine learning systems to extract behavioral, social, and spatial metrics from wildlife footage. Our pipeline leverages object detection, tracking, and behavioral classification systems to generate key metrics, including time budgets, behavioral transitions, social interactions, habitat associations, and group composition dynamics. Compared to ground-based methods, drone-based observations significantly improved behavioral granularity, reducing visibility loss by 15% and capturing more transitions with higher accuracy and continuity. We validate kabr-tools through three case studies, analyzing 969 behavioral sequences, surpassing the capacity of traditional methods for data capture and annotation. We found that, like Plains zebras, vigilance in Grevy's zebras decreases with herd size, but, unlike Plains zebras, habitat has a negligible impact. Plains and Grevy's zebras exhibit strong behavioral inertia, with rare transitions to alert behaviors and observed spatial segregation between Grevy's zebras, Plains zebras, and giraffes in mixed-species herds. By enabling automated behavioral monitoring at scale, kabr-tools offers a powerful tool for ecosystem-wide studies, advancing conservation, biodiversity research, and ecological monitoring.
>
---
#### [replaced 018] Learning Spatially Adaptive $\ell_1$-Norms Weights for Convolutional Synthesis Regularization
- **分类: cs.LG; cs.CV; math.OC**

- **链接: [http://arxiv.org/pdf/2503.09483v4](http://arxiv.org/pdf/2503.09483v4)**

> **作者:** Andreas Kofler; Luca Calatroni; Christoph Kolbitsch; Kostas Papafitsoros
>
> **备注:** Accepted for publication in the proceedings of the EUSIPCO 2025 conference; corrected typo in equation (3)
>
> **摘要:** We propose an unrolled algorithm approach for learning spatially adaptive parameter maps in the framework of convolutional synthesis-based $\ell_1$ regularization. More precisely, we consider a family of pre-trained convolutional filters and estimate deeply parametrized spatially varying parameters applied to the sparse feature maps by means of unrolling a FISTA algorithm to solve the underlying sparse estimation problem. The proposed approach is evaluated for image reconstruction of low-field MRI and compared to spatially adaptive and non-adaptive analysis-type procedures relying on Total Variation regularization and to a well-established model-based deep learning approach. We show that the proposed approach produces visually and quantitatively comparable results with the latter approaches and at the same time remains highly interpretable. In particular, the inferred parameter maps quantify the local contribution of each filter in the reconstruction, which provides valuable insight into the algorithm mechanism and could potentially be used to discard unsuited filters.
>
---
#### [replaced 019] Deep Linear Probe Generators for Weight Space Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.10811v2](http://arxiv.org/pdf/2410.10811v2)**

> **作者:** Jonathan Kahana; Eliahu Horwitz; Imri Shuval; Yedid Hoshen
>
> **备注:** ICLR 2025. Project page: https://vision.huji.ac.il/probegen
>
> **摘要:** Weight space learning aims to extract information about a neural network, such as its training dataset or generalization error. Recent approaches learn directly from model weights, but this presents many challenges as weights are high-dimensional and include permutation symmetries between neurons. An alternative approach, Probing, represents a model by passing a set of learned inputs (probes) through the model, and training a predictor on top of the corresponding outputs. Although probing is typically not used as a stand alone approach, our preliminary experiment found that a vanilla probing baseline worked surprisingly well. However, we discover that current probe learning strategies are ineffective. We therefore propose Deep Linear Probe Generators (ProbeGen), a simple and effective modification to probing approaches. ProbeGen adds a shared generator module with a deep linear architecture, providing an inductive bias towards structured probes thus reducing overfitting. While simple, ProbeGen performs significantly better than the state-of-the-art and is very efficient, requiring between 30 to 1000 times fewer FLOPs than other top approaches.
>
---
#### [replaced 020] AmorLIP: Efficient Language-Image Pretraining via Amortization
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18983v2](http://arxiv.org/pdf/2505.18983v2)**

> **作者:** Haotian Sun; Yitong Li; Yuchen Zhuang; Niao He; Hanjun Dai; Bo Dai
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) has demonstrated strong zero-shot performance across diverse downstream text-image tasks. Existing CLIP methods typically optimize a contrastive objective using negative samples drawn from each minibatch. To achieve robust representation learning, these methods require extremely large batch sizes and escalate computational demands to hundreds or even thousands of GPUs. Prior approaches to mitigate this issue often compromise downstream performance, prolong training duration, or face scalability challenges with very large datasets. To overcome these limitations, we propose AmorLIP, an efficient CLIP pretraining framework that amortizes expensive computations involved in contrastive learning through lightweight neural networks, which substantially improves training efficiency and performance. Leveraging insights from a spectral factorization of energy-based models, we introduce novel amortization objectives along with practical techniques to improve training stability. Extensive experiments across 38 downstream tasks demonstrate the superior zero-shot classification and retrieval capabilities of AmorLIP, consistently outperforming standard CLIP baselines with substantial relative improvements of up to 12.24%.
>
---
#### [replaced 021] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18876v2](http://arxiv.org/pdf/2510.18876v2)**

> **作者:** Haochen Wang; Yuhao Wang; Tao Zhang; Yikang Zhou; Yanwei Li; Jiacong Wang; Jiani Zheng; Ye Tian; Jiahao Meng; Zilong Huang; Guangcan Mai; Anran Wang; Yunhai Tong; Zhuochen Wang; Xiangtai Li; Zhaoxiang Zhang
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at holistic understanding, they struggle in capturing the dense world with complex scenes, requiring fine-grained analysis of intricate details and object inter-relationships. Region-level MLLMs have been a promising step. However, previous attempts are generally optimized to understand given regions in isolation, neglecting crucial global contexts. To address this, we introduce Grasp Any Region (GAR) for comprehen- sive region-level visual understanding. Empowered by an effective RoI-aligned feature replay technique, GAR supports (1) precise perception by leveraging necessary global contexts, and (2) modeling interactions between multiple prompts. Together, it then naturally achieves (3) advanced compositional reasoning to answer specific free-form questions about any region, shifting the paradigm from passive description to active dialogue. Moreover, we construct GAR-Bench, which not only provides a more accurate evaluation of single-region comprehension, but also, more importantly, measures interactions and complex reasoning across multiple regions. Extensive experiments have demonstrated that GAR-1B not only maintains the state-of-the-art captioning capabilities, e.g., outperforming DAM-3B +4.5 on DLC-Bench, but also excels at modeling relationships between multiple prompts with advanced comprehension capabilities, even surpassing InternVL3-78B on GAR-Bench-VQA. More importantly, our zero-shot GAR-8B even outperforms in-domain VideoRefer-7B on VideoRefer-BenchQ, indicating its strong capabilities can be easily transferred to videos.
>
---
#### [replaced 022] MINGLE: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11883v3](http://arxiv.org/pdf/2505.11883v3)**

> **作者:** Zihuan Qiu; Yi Xu; Chiyuan He; Fanman Meng; Linfeng Xu; Qingbo Wu; Hongliang Li
>
> **备注:** accepted by NeurIPS 2025
>
> **摘要:** Continual model merging integrates independently fine-tuned models sequentially without access to the original training data, offering a scalable and efficient solution for continual learning. However, existing methods face two critical challenges: parameter interference among tasks, which leads to catastrophic forgetting, and limited adaptability to evolving test distributions. To address these issues, we introduce the task of Test-Time Continual Model Merging (TTCMM), which leverages a small set of unlabeled test samples during inference to alleviate parameter conflicts and handle distribution shifts. We propose MINGLE, a novel framework for TTCMM. MINGLE employs a mixture-of-experts architecture with parameter-efficient, low-rank experts, which enhances adaptability to evolving test distributions while dynamically merging models to mitigate conflicts. To further reduce forgetting, we propose Null-Space Constrained Gating, which restricts gating updates to subspaces orthogonal to prior task representations, thereby suppressing activations on old tasks and preserving past knowledge. We further introduce an Adaptive Relaxation Strategy that adjusts constraint strength dynamically based on interference signals observed during test-time adaptation, striking a balance between stability and adaptability. Extensive experiments on standard continual merging benchmarks demonstrate that MINGLE achieves robust generalization, significantly reduces forgetting, and consistently surpasses previous state-of-the-art methods by 7-9% on average across diverse task orders. Our code is available at: https://github.com/zihuanqiu/MINGLE
>
---
#### [replaced 023] Variable Rate Image Compression via N-Gram Context based Swin-transformer
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2510.00058v2](http://arxiv.org/pdf/2510.00058v2)**

> **作者:** Priyanka Mudgal
>
> **备注:** Accepted at ISVC 2025
>
> **摘要:** This paper presents an N-gram context-based Swin Transformer for learned image compression. Our method achieves variable-rate compression with a single model. By incorporating N-gram context into the Swin Transformer, we overcome its limitation of neglecting larger regions during high-resolution image reconstruction due to its restricted receptive field. This enhancement expands the regions considered for pixel restoration, thereby improving the quality of high-resolution reconstructions. Our method increases context awareness across neighboring windows, leading to a -5.86\% improvement in BD-Rate over existing variable-rate learned image compression techniques. Additionally, our model improves the quality of regions of interest (ROI) in images, making it particularly beneficial for object-focused applications in fields such as manufacturing and industrial vision systems.
>
---
#### [replaced 024] PixelWorld: How Far Are We from Perceiving Everything as Pixels?
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19339v3](http://arxiv.org/pdf/2501.19339v3)**

> **作者:** Zhiheng Lyu; Xueguang Ma; Wenhu Chen
>
> **摘要:** Recent agentic language models increasingly need to interact with real-world environments that contain tightly intertwined visual and textual information, often through raw camera pixels rather than separately processed images and tokenized text. This shift highlights the need for a unified perception paradigm. To investigate this idea, we explore Perceive Everything as Pixels (PEAP) and introduce PixelWorld, a benchmark that renders natural-language, tabular, mathematical, and diagrammatic inputs into a shared pixel space. Experiments across multiple benchmarks show that PEAP achieves comparable performance to token-based approaches on semantic understanding tasks, suggesting that vision transformers can partially capture global textual semantics without explicit tokenization. In contrast, reasoning-intensive tasks such as mathematics and code show notable performance degradation, although Chain-of-Thought prompting helps mitigate this gap by compensating for missing symbolic structure. We further find that when visual and textual information are closely integrated, representing everything as pixels simplifies preprocessing and avoids cross-modal misalignment. PixelWorld thus provides a systematic and practical framework for evaluating unified vision--language models and facilitates further exploration of pixel-based multimodal learning.
>
---
#### [replaced 025] Towards foundational LiDAR world models with efficient latent flow matching
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.23434v2](http://arxiv.org/pdf/2506.23434v2)**

> **作者:** Tianran Liu; Shengwen Zhao; Nicholas Rhinehart
>
> **备注:** Accepted to the Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025), 25 pages, 13 figures
>
> **摘要:** LiDAR-based world models offer more structured and geometry-aware representations than their image-based counterparts. However, existing LiDAR world models are narrowly trained; each model excels only in the domain for which it was built. Can we develop LiDAR world models that exhibit strong transferability across multiple domains? We conduct the first systematic domain transfer study across three demanding scenarios: (i) outdoor to indoor generalization, (ii) sparse-beam & dense-beam adaptation, and (iii) non-semantic to semantic transfer. Given different amounts of fine-tuning data, our experiments show that a single pre-trained model can achieve up to 11% absolute improvement (83% relative) over training from scratch and outperforms training from scratch in 30/36 of our comparisons. This transferability of dynamic learning significantly reduces the reliance on manually annotated data for semantic occupancy forecasting: our method exceed the previous semantic occupancy forecasting models with only 5% of the labeled training data required by prior models. We also observed inefficiencies of current LiDAR world models, mainly through their under-compression of LiDAR data and inefficient training objectives. To address this, we propose a latent conditional flow matching (CFM)-based frameworks that achieves state-of-the-art reconstruction accuracy using only half the training data and a compression ratio 6 times higher than that of prior methods. Our model achieves SOTA performance on future-trajectory-conditioned semantic occupancy forecasting while being 23x more computationally efficient (a 28x FPS speedup); and achieves SOTA performance on semantic occupancy forecasting while being 2x more computationally efficient (a 1.1x FPS speedup).
>
---
#### [replaced 026] Concept-Guided Interpretability via Neural Chunking
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11576v3](http://arxiv.org/pdf/2505.11576v3)**

> **作者:** Shuchen Wu; Stephan Alaniz; Shyamgopal Karthik; Peter Dayan; Eric Schulz; Zeynep Akata
>
> **摘要:** Neural networks are often described as black boxes, reflecting the significant challenge of understanding their internal workings and interactions. We propose a different perspective that challenges the prevailing view: rather than being inscrutable, neural networks exhibit patterns in their raw population activity that mirror regularities in the training data. We refer to this as the Reflection Hypothesis and provide evidence for this phenomenon in both simple recurrent neural networks (RNNs) and complex large language models (LLMs). Building on this insight, we propose to leverage our cognitive tendency of chunking to segment high-dimensional neural population dynamics into interpretable units that reflect underlying concepts. We propose three methods to extract recurring chunks on a neural population level, complementing each other based on label availability and neural data dimensionality. Discrete sequence chunking (DSC) learns a dictionary of entities in a lower-dimensional neural space; population averaging (PA) extracts recurring entities that correspond to known labels; and unsupervised chunk discovery (UCD) can be used when labels are absent. We demonstrate the effectiveness of these methods in extracting concept-encoding entities agnostic to model architectures. These concepts can be both concrete (words), abstract (POS tags), or structural (narrative schema). Additionally, we show that extracted chunks play a causal role in network behavior, as grafting them leads to controlled and predictable changes in the model's behavior. Our work points to a new direction for interpretability, one that harnesses both cognitive principles and the structure of naturalistic data to reveal the hidden computations of complex learning systems, gradually transforming them from black boxes into systems we can begin to understand.
>
---
#### [replaced 027] Towards Depth Foundation Model: Recent Trends in Vision-Based Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11540v2](http://arxiv.org/pdf/2507.11540v2)**

> **作者:** Zhen Xu; Hongyu Zhou; Sida Peng; Haotong Lin; Haoyu Guo; Jiahao Shao; Peishan Yang; Qinglin Yang; Sheng Miao; Xingyi He; Yifan Wang; Yue Wang; Ruizhen Hu; Yiyi Liao; Xiaowei Zhou; Hujun Bao
>
> **摘要:** Depth estimation is a fundamental task in 3D computer vision, crucial for applications such as 3D reconstruction, free-viewpoint rendering, robotics, autonomous driving, and AR/VR technologies. Traditional methods relying on hardware sensors like LiDAR are often limited by high costs, low resolution, and environmental sensitivity, limiting their applicability in real-world scenarios. Recent advances in vision-based methods offer a promising alternative, yet they face challenges in generalization and stability due to either the low-capacity model architectures or the reliance on domain-specific and small-scale datasets. The emergence of scaling laws and foundation models in other domains has inspired the development of "depth foundation models": deep neural networks trained on large datasets with strong zero-shot generalization capabilities. This paper surveys the evolution of deep learning architectures and paradigms for depth estimation across the monocular, stereo, multi-view, and monocular video settings. We explore the potential of these models to address existing challenges and provide a comprehensive overview of large-scale datasets that can facilitate their development. By identifying key architectures and training strategies, we aim to highlight the path towards robust depth foundation models, offering insights into their future research and applications.
>
---
#### [replaced 028] Video-R1: Reinforcing Video Reasoning in MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21776v4](http://arxiv.org/pdf/2503.21776v4)**

> **作者:** Kaituo Feng; Kaixiong Gong; Bohao Li; Zonghao Guo; Yibing Wang; Tianshuo Peng; Junfei Wu; Xiaoying Zhang; Benyou Wang; Xiangyu Yue
>
> **备注:** NeurIPS 2025, Project page: https://github.com/tulerfeng/Video-R1
>
> **摘要:** Inspired by DeepSeek-R1's success in eliciting reasoning abilities through rule-based reinforcement learning (RL), we introduce Video-R1 as the first attempt to systematically explore the R1 paradigm for incentivizing video reasoning within multimodal large language models (MLLMs). However, directly applying RL training with the GRPO algorithm to video reasoning presents two primary challenges: (i) a lack of temporal modeling for video reasoning, and (ii) the scarcity of high-quality video-reasoning data. To address these issues, we first propose the T-GRPO algorithm, which encourages models to utilize temporal information in videos for reasoning. Additionally, instead of relying solely on video data, we incorporate high-quality image-reasoning data into the training process. We have constructed two datasets: Video-R1-CoT-165k for SFT cold start and Video-R1-260k for RL training, both comprising image and video data. Experimental results demonstrate that Video-R1 achieves significant improvements on video reasoning benchmarks such as VideoMMMU and VSI-Bench, as well as on general video benchmarks including MVBench and TempCompass, etc. Notably, Video-R1-7B attains a 37.1% accuracy on video spatial reasoning benchmark VSI-bench, surpassing the commercial proprietary model GPT-4o. All code, models, and data are released in: https://github.com/tulerfeng/Video-R1.
>
---
#### [replaced 029] PAGE-4D: Disentangled Pose and Geometry Estimation for 4D Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.17568v2](http://arxiv.org/pdf/2510.17568v2)**

> **作者:** Kaichen Zhou; Yuhan Wang; Grace Chen; Xinhai Chang; Gaspard Beaudouin; Fangneng Zhan; Paul Pu Liang; Mengyu Wang
>
> **摘要:** Recent 3D feed-forward models, such as the Visual Geometry Grounded Transformer (VGGT), have shown strong capability in inferring 3D attributes of static scenes. However, since they are typically trained on static datasets, these models often struggle in real-world scenarios involving complex dynamic elements, such as moving humans or deformable objects like umbrellas. To address this limitation, we introduce PAGE-4D, a feedforward model that extends VGGT to dynamic scenes, enabling camera pose estimation, depth prediction, and point cloud reconstruction -- all without post-processing. A central challenge in multi-task 4D reconstruction is the inherent conflict between tasks: accurate camera pose estimation requires suppressing dynamic regions, while geometry reconstruction requires modeling them. To resolve this tension, we propose a dynamics-aware aggregator that disentangles static and dynamic information by predicting a dynamics-aware mask -- suppressing motion cues for pose estimation while amplifying them for geometry reconstruction. Extensive experiments show that PAGE-4D consistently outperforms the original VGGT in dynamic scenarios, achieving superior results in camera pose estimation, monocular and video depth estimation, and dense point map reconstruction.
>
---
#### [replaced 030] The Photographer Eye: Teaching Multimodal Large Language Models to Understand Image Aesthetics like Photographers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.18582v2](http://arxiv.org/pdf/2509.18582v2)**

> **作者:** Daiqing Qi; Handong Zhao; Jing Shi; Simon Jenni; Yifei Fan; Franck Dernoncourt; Scott Cohen; Sheng Li
>
> **摘要:** While editing directly from life, photographers have found it too difficult to see simultaneously both the blue and the sky. Photographer and curator, Szarkowski insightfully revealed one of the notable gaps between general and aesthetic visual understanding: while the former focuses on identifying the factual element in an image (sky), the latter transcends such object identification, viewing it instead as an aesthetic component--a pure color block (blue). Such fundamental distinctions between general (detection, localization, etc.) and aesthetic (color, lighting, composition, etc.) visual understanding present a significant challenge for Multimodal Large Language Models (MLLMs). Although some recent works have made initial explorations, they are often limited to general and basic aesthetic commonsense. As a result, they frequently fall short in real-world scenarios (Fig. 1), which require extensive expertise--including photographic techniques, photo pre/post-processing knowledge, and more, to provide a detailed analysis and description. To fundamentally enhance the aesthetics understanding of MLLMs, we first introduce a novel dataset, PhotoCritique, derived from extensive discussions among professional photographers and enthusiasts, and characterized by the large scale, expertise, and diversity. Then, to better learn visual aesthetics from PhotoCritique, we furthur propose a novel model, PhotoEye, featuring a languageguided multi-view vision fusion mechanism to understand image aesthetics from multiple perspectives. Finally, we present a novel benchmark, PhotoBench, a comprehensive and professional benchmark for aesthetic visual understanding. On existing benchmarks and PhotoBench, our model demonstrates clear advantages over existing models.
>
---
#### [replaced 031] ScaleNet: Scaling up Pretrained Neural Networks with Incremental Parameters
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.18431v2](http://arxiv.org/pdf/2510.18431v2)**

> **作者:** Zhiwei Hao; Jianyuan Guo; Li Shen; Kai Han; Yehui Tang; Han Hu; Yunhe Wang
>
> **备注:** accepted to IEEE Transactions on Image Processing (TIP)
>
> **摘要:** Recent advancements in vision transformers (ViTs) have demonstrated that larger models often achieve superior performance. However, training these models remains computationally intensive and costly. To address this challenge, we introduce ScaleNet, an efficient approach for scaling ViT models. Unlike conventional training from scratch, ScaleNet facilitates rapid model expansion with negligible increases in parameters, building on existing pretrained models. This offers a cost-effective solution for scaling up ViTs. Specifically, ScaleNet achieves model expansion by inserting additional layers into pretrained ViTs, utilizing layer-wise weight sharing to maintain parameters efficiency. Each added layer shares its parameter tensor with a corresponding layer from the pretrained model. To mitigate potential performance degradation due to shared weights, ScaleNet introduces a small set of adjustment parameters for each layer. These adjustment parameters are implemented through parallel adapter modules, ensuring that each instance of the shared parameter tensor remains distinct and optimized for its specific function. Experiments on the ImageNet-1K dataset demonstrate that ScaleNet enables efficient expansion of ViT models. With a 2$\times$ depth-scaled DeiT-Base model, ScaleNet achieves a 7.42% accuracy improvement over training from scratch while requiring only one-third of the training epochs, highlighting its efficiency in scaling ViTs. Beyond image classification, our method shows significant potential for application in downstream vision areas, as evidenced by the validation in object detection task.
>
---
#### [replaced 032] LBL: Logarithmic Barrier Loss Function for One-class Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2307.10753v2](http://arxiv.org/pdf/2307.10753v2)**

> **作者:** Xiaofeng Guo; Ziyang Jiang; Tianlei Wang; Shichen Zhang; Dinghan Hu; Jiuwen Cao
>
> **摘要:** One-class classification (OCC) aims to train a classifier solely on target data and attracts increasing attention due to its applicability in practice. Despite OCC has obtained many advances, it still lacks the effective OCC loss functions for deep learning. In this paper, a novel logarithmic barrier function based OCC loss (LBL) that assigns large gradients to margin samples and thus derives more compact hypersphere is first proposed by approximating the OCC objective smoothly. But the optimization of LBL may be instability especially when samples lie on the boundary leading to the infinity value. To address this issue, a smoother LBLSig loss is further proposed by utilizing a unilateral relaxation Sigmoid function. Experiments on different networks demonstrate the effectiveness of the proposed LBL and LBLSig. The source code can be found at https://github.com/ML-HDU/LBL_LBLSig.
>
---
#### [replaced 033] Chiron-o1: Igniting Multimodal Large Language Models towards Generalizable Medical Reasoning via Mentor-Intern Collaborative Search
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16962v2](http://arxiv.org/pdf/2506.16962v2)**

> **作者:** Haoran Sun; Yankai Jiang; Wenjie Lou; Yujie Zhang; Wenjie Li; Lilong Wang; Mianxin Liu; Lei Liu; Xiaosong Wang
>
> **摘要:** Multimodal large language models (MLLMs) have begun to demonstrate robust reasoning capabilities on general tasks, yet their application in the medical domain remains in its early stages. Constructing chain-of-thought (CoT) training data is essential for bolstering the reasoning abilities of medical MLLMs. However, existing approaches exhibit a deficiency in offering a comprehensive framework for searching and evaluating effective reasoning paths towards critical diagnosis. To address this challenge, we propose Mentor-Intern Collaborative Search (MICS), a novel reasoning-path searching scheme to generate rigorous and effective medical CoT data. MICS first leverages mentor models to initialize the reasoning, one step at a time, then prompts each intern model to continue the thinking along those initiated paths, and finally selects the optimal reasoning path according to the overall reasoning performance of multiple intern models. The reasoning performance is determined by an MICS-Score, which assesses the quality of generated reasoning paths. Eventually, we construct MMRP, a multi-task medical reasoning dataset with ranked difficulty, and Chiron-o1, a new medical MLLM devised via a curriculum learning strategy, with robust visual question-answering and generalizable reasoning capabilities. Extensive experiments demonstrate that Chiron-o1, trained on our CoT dataset constructed using MICS, achieves state-of-the-art performance across a list of medical visual question answering and reasoning benchmarks. Codes are available at https://github.com/manglu097/Chiron-o1
>
---
#### [replaced 034] Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14475v2](http://arxiv.org/pdf/2503.14475v2)**

> **作者:** Umar Farooq; Jean-Yves Guillemaut; Adrian Hilton; Marco Volino
>
> **摘要:** The field of Novel View Synthesis has been revolutionized by 3D Gaussian Splatting (3DGS), which enables high-quality scene reconstruction that can be rendered in real-time. 3DGS-based techniques typically suffer from high GPU memory and disk storage requirements which limits their practical application on consumer-grade devices. We propose Opti3DGS, a novel frequency-modulated coarse-to-fine optimization framework that aims to minimize the number of Gaussian primitives used to represent a scene, thus reducing memory and storage demands. Opti3DGS leverages image frequency modulation, initially enforcing a coarse scene representation and progressively refining it by modulating frequency details in the training images. On the baseline 3DGS, we demonstrate an average reduction of 62% in Gaussians, a 40% reduction in the training GPU memory requirements and a 20% reduction in optimization time without sacrificing the visual quality. Furthermore, we show that our method integrates seamlessly with many 3DGS-based techniques, consistently reducing the number of Gaussian primitives while maintaining, and often improving, visual quality. Additionally, Opti3DGS inherently produces a level-of-detail scene representation at no extra cost, a natural byproduct of the optimization pipeline. Results and code will be made publicly available.
>
---
#### [replaced 035] Where are we with calibration under dataset shift in image classification?
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.07780v2](http://arxiv.org/pdf/2507.07780v2)**

> **作者:** Mélanie Roschewitz; Raghav Mehta; Fabio de Sousa Ribeiro; Ben Glocker
>
> **备注:** Code available at https://github.com/biomedia-mira/calibration_under_shifts. Published in TMLR, October 2025 (https://openreview.net/forum?id=1NYKXlRU2H)
>
> **摘要:** We conduct an extensive study on the state of calibration under real-world dataset shift for image classification. Our work provides important insights on the choice of post-hoc and in-training calibration techniques, and yields practical guidelines for all practitioners interested in robust calibration under shift. We compare various post-hoc calibration methods, and their interactions with common in-training calibration strategies (e.g., label smoothing), across a wide range of natural shifts, on eight different classification tasks across several imaging domains. We find that: (i) simultaneously applying entropy regularisation and label smoothing yield the best calibrated raw probabilities under dataset shift, (ii) post-hoc calibrators exposed to a small amount of semantic out-of-distribution data (unrelated to the task) are most robust under shift, (iii) recent calibration methods specifically aimed at increasing calibration under shifts do not necessarily offer significant improvements over simpler post-hoc calibration methods, (iv) improving calibration under shifts often comes at the cost of worsening in-distribution calibration. Importantly, these findings hold for randomly initialised classifiers, as well as for those finetuned from foundation models, the latter being consistently better calibrated compared to models trained from scratch. Finally, we conduct an in-depth analysis of ensembling effects, finding that (i) applying calibration prior to ensembling (instead of after) is more effective for calibration under shifts, (ii) for ensembles, OOD exposure deteriorates the ID-shifted calibration trade-off, (iii) ensembling remains one of the most effective methods to improve calibration robustness and, combined with finetuning from foundation models, yields best calibration results overall.
>
---
#### [replaced 036] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12323v2](http://arxiv.org/pdf/2506.12323v2)**

> **作者:** Janet Wang; Yunbei Zhang; Zhengming Ding; Jihun Hamm
>
> **备注:** NeurIPS 2025
>
> **摘要:** Paucity of medical data severely limits the generalizability of diagnostic ML models, as the full spectrum of disease variability can not be represented by a small clinical dataset. To address this, diffusion models (DMs) have been considered as a promising avenue for synthetic image generation and augmentation. However, they frequently produce medically inaccurate images, deteriorating the model performance. Expert domain knowledge is critical for synthesizing images that correctly encode clinical information, especially when data is scarce and quality outweighs quantity. Existing approaches for incorporating human feedback, such as reinforcement learning (RL) and Direct Preference Optimization (DPO), rely on robust reward functions or demand labor-intensive expert evaluations. Recent progress in Multimodal Large Language Models (MLLMs) reveals their strong visual reasoning capabilities, making them adept candidates as evaluators. In this work, we propose a novel framework, coined MAGIC (Medically Accurate Generation of Images through AI-Expert Collaboration), that synthesizes clinically accurate skin disease images for data augmentation. Our method creatively translates expert-defined criteria into actionable feedback for image synthesis of DMs, significantly improving clinical accuracy while reducing the direct human workload. Experiments demonstrate that our method greatly improves the clinical quality of synthesized skin disease images, with outputs aligning with dermatologist assessments. Additionally, augmenting training data with these synthesized images improves diagnostic accuracy by +9.02% on a challenging 20-condition skin disease classification task, and by +13.89% in the few-shot setting.
>
---
#### [replaced 037] CNeuroMod-THINGS, a densely-sampled fMRI dataset for visual neuroscience
- **分类: q-bio.NC; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09024v4](http://arxiv.org/pdf/2507.09024v4)**

> **作者:** Marie St-Laurent; Basile Pinsard; Oliver Contier; Elizabeth DuPre; Katja Seeliger; Valentina Borghesani; Julie A. Boyle; Lune Bellec; Martin N. Hebart
>
> **备注:** 16 pages manuscript, 5 figures, 9 pages supplementary material
>
> **摘要:** Data-hungry neuro-AI modelling requires ever larger neuroimaging datasets. CNeuroMod-THINGS meets this need by capturing neural representations for a wide set of semantic concepts using well-characterized images in a new densely-sampled, large-scale fMRI dataset. Importantly, CNeuroMod-THINGS exploits synergies between two existing projects: the THINGS initiative (THINGS) and the Courtois Project on Neural Modelling (CNeuroMod). THINGS has developed a common set of thoroughly annotated images broadly sampling natural and man-made objects which is used to acquire a growing collection of large-scale multimodal neural responses. Meanwhile, CNeuroMod is acquiring hundreds of hours of fMRI data from a core set of participants during controlled and naturalistic tasks, including visual tasks like movie watching and videogame playing. For CNeuroMod-THINGS, four CNeuroMod participants each completed 33-36 sessions of a continuous recognition paradigm using approximately 4000 images from the THINGS stimulus set spanning 720 categories. We report behavioural and neuroimaging metrics that showcase the quality of the data. By bridging together large existing resources, CNeuroMod-THINGS expands our capacity to model broad slices of the human visual experience.
>
---
#### [replaced 038] ComDrive: Comfort-Oriented End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.05051v2](http://arxiv.org/pdf/2410.05051v2)**

> **作者:** Junming Wang; Xingyu Zhang; Zebin Xing; Songen Gu; Xiaoyang Guo; Yang Hu; Ziying Song; Qian Zhang; Xiaoxiao Long; Wei Yin
>
> **备注:** IROS 2025
>
> **摘要:** We propose ComDrive: the first comfort-oriented end-to-end autonomous driving system to generate temporally consistent and comfortable trajectories. Recent studies have demonstrated that imitation learning-based planners and learning-based trajectory scorers can effectively generate and select safety trajectories that closely mimic expert demonstrations. However, such trajectory planners and scorers face the challenge of generating temporally inconsistent and uncomfortable trajectories. To address these issues, ComDrive first extracts 3D spatial representations through sparse perception, which then serves as conditional inputs. These inputs are used by a Conditional Denoising Diffusion Probabilistic Model (DDPM)-based motion planner to generate temporally consistent multi-modal trajectories. A dual-stream adaptive trajectory scorer subsequently selects the most comfortable trajectory from these candidates to control the vehicle. Experiments demonstrate that ComDrive achieves state-of-the-art performance in both comfort and safety, outperforming UniAD by 17% in driving comfort and reducing collision rates by 25% compared to SparseDrive. More results are available on our project page: https://jmwang0117.github.io/ComDrive/.
>
---
#### [replaced 039] OmniNWM: Omniscient Driving Navigation World Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.18313v2](http://arxiv.org/pdf/2510.18313v2)**

> **作者:** Bohan Li; Zhuang Ma; Dalong Du; Baorui Peng; Zhujin Liang; Zhenqiang Liu; Chao Ma; Yueming Jin; Hao Zhao; Wenjun Zeng; Xin Jin
>
> **备注:** https://arlo0o.github.io/OmniNWM/
>
> **摘要:** Autonomous driving world models are expected to work effectively across three core dimensions: state, action, and reward. Existing models, however, are typically restricted to limited state modalities, short video sequences, imprecise action control, and a lack of reward awareness. In this paper, we introduce OmniNWM, an omniscient panoramic navigation world model that addresses all three dimensions within a unified framework. For state, OmniNWM jointly generates panoramic videos of RGB, semantics, metric depth, and 3D occupancy. A flexible forcing strategy enables high-quality long-horizon auto-regressive generation. For action, we introduce a normalized panoramic Plucker ray-map representation that encodes input trajectories into pixel-level signals, enabling highly precise and generalizable control over panoramic video generation. Regarding reward, we move beyond learning reward functions with external image-based models: instead, we leverage the generated 3D occupancy to directly define rule-based dense rewards for driving compliance and safety. Extensive experiments demonstrate that OmniNWM achieves state-of-the-art performance in video generation, control accuracy, and long-horizon stability, while providing a reliable closed-loop evaluation framework through occupancy-grounded rewards. Project page is available at https://github.com/Arlo0o/OmniNWM.
>
---
#### [replaced 040] Breaking the Discretization Barrier of Continuous Physics Simulation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.17955v2](http://arxiv.org/pdf/2509.17955v2)**

> **作者:** Fan Xu; Hao Wu; Nan Wang; Lilan Peng; Kun Wang; Wei Gong; Xibin Zhao
>
> **摘要:** The modeling of complicated time-evolving physical dynamics from partial observations is a long-standing challenge. Particularly, observations can be sparsely distributed in a seemingly random or unstructured manner, making it difficult to capture highly nonlinear features in a variety of scientific and engineering problems. However, existing data-driven approaches are often constrained by fixed spatial and temporal discretization. While some researchers attempt to achieve spatio-temporal continuity by designing novel strategies, they either overly rely on traditional numerical methods or fail to truly overcome the limitations imposed by discretization. To address these, we propose CoPS, a purely data-driven methods, to effectively model continuous physics simulation from partial observations. Specifically, we employ multiplicative filter network to fuse and encode spatial information with the corresponding observations. Then we customize geometric grids and use message-passing mechanism to map features from original spatial domain to the customized grids. Subsequently, CoPS models continuous-time dynamics by designing multi-scale graph ODEs, while introducing a Markov-based neural auto-correction module to assist and constrain the continuous extrapolations. Comprehensive experiments demonstrate that CoPS advances the state-of-the-art methods in space-time continuous modeling across various scenarios.
>
---
#### [replaced 041] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09271v4](http://arxiv.org/pdf/2503.09271v4)**

> **作者:** Chiara Cappellino; Gianluca Mancusi; Matteo Mosconi; Angelo Porrello; Simone Calderara; Rita Cucchiara
>
> **备注:** Accepted at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Open-Vocabulary object detectors can generalize to an unrestricted set of categories through simple textual prompting. However, adapting these models to rare classes or reinforcing their abilities on multiple specialized domains remains essential. While recent methods rely on monolithic adaptation strategies with a single set of weights, we embrace modular deep learning. We introduce DitHub, a framework designed to build and maintain a library of efficient adaptation modules. Inspired by Version Control Systems, DitHub manages expert modules as branches that can be fetched and merged as needed. This modular approach allows us to conduct an in-depth exploration of the compositional properties of adaptation modules, marking the first such study in Object Detection. Our method achieves state-of-the-art performance on the ODinW-13 benchmark and ODinW-O, a newly introduced benchmark designed to assess class reappearance. For more details, visit our project page: https://aimagelab.github.io/DitHub/
>
---
#### [replaced 042] RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration
- **分类: cs.AI; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.25271v3](http://arxiv.org/pdf/2509.25271v3)**

> **作者:** Xiuyuan Chen; Jian Zhao; Yuchen Yuan; Tianle Zhang; Huilin Zhou; Zheng Zhu; Ping Hu; Linghe Kong; Chi Zhang; Weiran Huang; Xuelong Li
>
> **摘要:** Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.
>
---
#### [replaced 043] QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00711v2](http://arxiv.org/pdf/2506.00711v2)**

> **作者:** Wei Dai; Peilin Chen; Chanakya Ekbote; Paul Pu Liang
>
> **备注:** Accepted as Oral at NeurIPS 2025. Revision after camera ready
>
> **摘要:** Clinical decision-making routinely demands reasoning over heterogeneous data, yet existing multimodal language models (MLLMs) remain largely vision-centric and fail to generalize across clinical specialties. To bridge this gap, we introduce QoQ-Med-7B/32B, the first open generalist clinical foundation model that jointly reasons across medical images, time-series signals, and text reports. QoQ-Med is trained with Domain-aware Relative Policy Optimization (DRPO), a novel reinforcement-learning objective that hierarchically scales normalized rewards according to domain rarity and modality difficulty, mitigating performance imbalance caused by skewed clinical data distributions. Trained on 2.61 million instruction tuning pairs spanning 9 clinical domains, we show that DRPO training boosts diagnostic performance by 43% in macro-F1 on average across all visual domains as compared to other critic-free training methods like GRPO. Furthermore, with QoQ-Med trained on intensive segmentation data, it is able to highlight salient regions related to the diagnosis, with an IoU 10x higher than open models while reaching the performance of OpenAI o4-mini. To foster reproducibility and downstream research, we release (i) the full model weights, (ii) the modular training pipeline, and (iii) all intermediate reasoning traces at https://github.com/DDVD233/QoQ_Med.
>
---
#### [replaced 044] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10483v3](http://arxiv.org/pdf/2504.10483v3)**

> **作者:** Xingjian Leng; Jaskirat Singh; Yunzhong Hou; Zhenchang Xing; Saining Xie; Liang Zheng
>
> **摘要:** In this paper we tackle a fundamental question: "Can we train latent diffusion models together with the variational auto-encoder (VAE) tokenizer in an end-to-end manner?" Traditional deep-learning wisdom dictates that end-to-end training is often preferable when possible. However, for latent diffusion transformers, it is observed that end-to-end training both VAE and diffusion-model using standard diffusion-loss is ineffective, even causing a degradation in final performance. We show that while diffusion loss is ineffective, end-to-end training can be unlocked through the representation-alignment (REPA) loss -- allowing both VAE and diffusion model to be jointly tuned during the training process. Despite its simplicity, the proposed training recipe (REPA-E) shows remarkable performance; speeding up diffusion model training by over 17x and 45x over REPA and vanilla training recipes, respectively. Interestingly, we observe that end-to-end tuning with REPA-E also improves the VAE itself; leading to improved latent space structure and downstream generation performance. In terms of final performance, our approach sets a new state-of-the-art; achieving FID of 1.12 and 1.69 with and without classifier-free guidance on ImageNet 256 x 256. Code is available at https://end2end-diffusion.github.io.
>
---
#### [replaced 045] Intelligent Software System for Low-Cost, Brightfield Segmentation: Algorithmic Implementation for Cytometric Auto-Analysis
- **分类: q-bio.QM; cs.CV; eess.IV; q-bio.CB**

- **链接: [http://arxiv.org/pdf/2509.11354v4](http://arxiv.org/pdf/2509.11354v4)**

> **作者:** Surajit Das; Pavel Zun
>
> **摘要:** Bright-field microscopy, a cost-effective solution for live-cell culture, is often the only resource available, along with standard CPUs, for many low-budget labs. The inherent chal- lenges of bright-field images - their noisiness, low contrast, and dynamic morphology - coupled with a lack of GPU resources and complex software interfaces, hinder the desired research output. This article presents a novel microscopy image analysis frame- work designed for low-budget labs equipped with a standard CPU desktop. The Python-based program enables cytometric analysis of live, unstained cells in culture through an advanced computer vision and machine learning pipeline. Crucially, the framework operates on label-free data, requiring no manually annotated training data or training phase. It is accessible via a user-friendly, cross-platform GUI that requires no programming skills, while also providing a scripting interface for programmatic control and integration by developers. The end-to-end workflow performs semantic and instance segmentation, feature extraction, analysis, evaluation, and automated report generation. Its modular archi- tecture supports easy maintenance and flexible integration while supporting both single-image and batch processing. Validated on several unstained cell types from the public dataset of livecells, the framework demonstrates superior accuracy and reproducibility compared to contemporary tools like Cellpose and StarDist. Its competitive segmentation speed on a CPU-based platform highlights its significant potential for basic research and clinical applications - particularly in cell transplantation for personalised medicine and muscle regeneration therapies. The access to the application is available for reproducibility
>
---
#### [replaced 046] On the Effectiveness of Methods and Metrics for Explainable AI in Remote Sensing Image Scene Classification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05916v3](http://arxiv.org/pdf/2507.05916v3)**

> **作者:** Jonas Klotz; Tom Burgert; Begüm Demir
>
> **备注:** The code of this work will be publicly available at https://git.tu-berlin.de/rsim/xai4rs Accepted at IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
>
> **摘要:** The development of explainable artificial intelligence (xAI) methods for scene classification problems has attracted great attention in remote sensing (RS). Most xAI methods and the related evaluation metrics in RS are initially developed for natural images considered in computer vision (CV), and their direct usage in RS may not be suitable. To address this issue, in this paper, we investigate the effectiveness of explanation methods and metrics in the context of RS image scene classification. In detail, we methodologically and experimentally analyze ten explanation metrics spanning five categories (faithfulness, robustness, localization, complexity, randomization), applied to five established feature attribution methods (Occlusion, LIME, GradCAM, LRP, and DeepLIFT) across three RS datasets. Our methodological analysis identifies key limitations in both explanation methods and metrics. The performance of perturbation-based methods, such as Occlusion and LIME, heavily depends on perturbation baselines and spatial characteristics of RS scenes. Gradient-based approaches like GradCAM struggle when multiple labels are present in the same image, while some relevance propagation methods (LRP) can distribute relevance disproportionately relative to the spatial extent of classes. Analogously, we find limitations in evaluation metrics. Faithfulness metrics share the same problems as perturbation-based methods. Localization metrics and complexity metrics are unreliable for classes with a large spatial extent. In contrast, robustness metrics and randomization metrics consistently exhibit greater stability. Our experimental results support these methodological findings. Based on our analysis, we provide guidelines for selecting explanation methods, metrics, and hyperparameters in the context of RS image scene classification.
>
---
#### [replaced 047] Learning Differential Pyramid Representation for Tone Mapping
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.01463v2](http://arxiv.org/pdf/2412.01463v2)**

> **作者:** Qirui Yang; Yinbo Li; Yihao Liu; Peng-Tao Jiang; Fangpu Zhang; Qihua Cheng; Huanjing Yue; Jingyu Yang
>
> **摘要:** Existing tone mapping methods operate on downsampled inputs and rely on handcrafted pyramids to recover high-frequency details. These designs typically fail to preserve fine textures and structural fidelity in complex HDR scenes. Furthermore, most methods lack an effective mechanism to jointly model global tone consistency and local contrast enhancement, leading to globally flat or locally inconsistent outputs such as halo artifacts. We present the Differential Pyramid Representation Network (DPRNet), an end-to-end framework for high-fidelity tone mapping. At its core is a learnable differential pyramid that generalizes traditional Laplacian and Difference-of-Gaussian pyramids through content-aware differencing operations across scales. This allows DPRNet to adaptively capture high-frequency variations under diverse luminance and contrast conditions. To enforce perceptual consistency, DPRNet incorporates global tone perception and local tone tuning modules operating on downsampled inputs, enabling efficient yet expressive tone adaptation. Finally, an iterative detail enhancement module progressively restores the full-resolution output in a coarse-to-fine manner, reinforcing structure and sharpness. Experiments show that DPRNet achieves state-of-the-art results, improving PSNR by 2.39 dB on the 4K HDR+ dataset and 3.01 dB on the 4K HDRI Haven dataset, while producing perceptually coherent, detail-preserving results. \textit{We provide an anonymous online demo at https://xxxxxxdprnet.github.io/DPRNet/.
>
---
#### [replaced 048] FST.ai 2.0: An Explainable AI Ecosystem for Fair, Fast, and Inclusive Decision-Making in Olympic and Paralympic Taekwondo
- **分类: cs.AI; cs.CV; cs.LG; stat.ML; 68T01; I.2.8**

- **链接: [http://arxiv.org/pdf/2510.18193v2](http://arxiv.org/pdf/2510.18193v2)**

> **作者:** Keivan Shariatmadar; Ahmad Osman; Ramin Ray; Kisam Kim
>
> **备注:** 23 pages, 12 figures
>
> **摘要:** Fair, transparent, and explainable decision-making remains a critical challenge in Olympic and Paralympic combat sports. This paper presents \emph{FST.ai 2.0}, an explainable AI ecosystem designed to support referees, coaches, and athletes in real time during Taekwondo competitions and training. The system integrates {pose-based action recognition} using graph convolutional networks (GCNs), {epistemic uncertainty modeling} through credal sets, and {explainability overlays} for visual decision support. A set of {interactive dashboards} enables human--AI collaboration in referee evaluation, athlete performance analysis, and Para-Taekwondo classification. Beyond automated scoring, FST.ai~2.0 incorporates modules for referee training, fairness monitoring, and policy-level analytics within the World Taekwondo ecosystem. Experimental validation on competition data demonstrates an {85\% reduction in decision review time} and {93\% referee trust} in AI-assisted decisions. The framework thus establishes a transparent and extensible pipeline for trustworthy, data-driven officiating and athlete assessment. By bridging real-time perception, explainable inference, and governance-aware design, FST.ai~2.0 represents a step toward equitable, accountable, and human-aligned AI in sports.
>
---
#### [replaced 049] LookUp3D: Data-Driven 3D Scanning
- **分类: cs.CV; cs.GR; eess.IV**

- **链接: [http://arxiv.org/pdf/2405.14882v2](http://arxiv.org/pdf/2405.14882v2)**

> **作者:** Giancarlo Pereira; Yidan Gao; Yurii Piadyk; David Fouhey; Claudio T Silva; Daniele Panozzo
>
> **备注:** Giancarlo Pereira, Yidan Gao, and Yurii Piadyk are joint first authors with equal contribution. 11 pages of main paper, 9 pages of supplemental text (all combined into a single document)
>
> **摘要:** High speed, high-resolution, and accurate 3D scanning would open doors to many new applications in graphics, robotics, science, and medicine by enabling the accurate scanning of deformable objects during interactions. Past attempts to use structured light, time-of-flight, and stereo in high-speed settings have usually required tradeoffs in resolution or inaccuracy. In this paper, we introduce a method that enables, for the first time, 3D scanning at 450 frames per second at 1~Megapixel, or 1,450 frames per second at 0.4~Megapixel in an environment with controlled lighting. The key idea is to use a per-pixel lookup table that maps colors to depths, which is built using a linear stage. Imperfections, such as lens-distortion and sensor defects are baked into the calibration. We describe our method and test it on a novel hardware prototype. We compare the system with both ground-truth geometry as well as commercially available dynamic sensors like the Microsoft Kinect and Intel Realsense. Our results show the system acquiring geometry of objects undergoing high-speed deformations and oscillations and demonstrate the ability to recover physical properties from the reconstructions.
>
---
#### [replaced 050] Learning What Matters: Steering Diffusion via Spectrally Anisotropic Forward Noise
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.09660v3](http://arxiv.org/pdf/2510.09660v3)**

> **作者:** Luca Scimeca; Thomas Jiralerspong; Berton Earnshaw; Jason Hartford; Yoshua Bengio
>
> **摘要:** Diffusion Probabilistic Models (DPMs) have achieved strong generative performance, yet their inductive biases remain largely implicit. In this work, we aim to build inductive biases into the training and sampling of diffusion models to better accommodate the target distribution of the data to model. We introduce an anisotropic noise operator that shapes these biases by replacing the isotropic forward covariance with a structured, frequency-diagonal covariance. This operator unifies band-pass masks and power-law weightings, allowing us to emphasize or suppress designated frequency bands, while keeping the forward process Gaussian. We refer to this as spectrally anisotropic Gaussian diffusion (SAGD). In this work, we derive the score relation for anisotropic covariances and show that, under full support, the learned score converges to the true data score as $t\!\to\!0$, while anisotropy reshapes the probability-flow path from noise to data. Empirically, we show the induced anisotropy outperforms standard diffusion across several vision datasets, and enables selective omission: learning while ignoring known corruptions confined to specific bands. Together, these results demonstrate that carefully designed anisotropic forward noise provides a simple, yet principled, handle to tailor inductive bias in DPMs.
>
---
#### [replaced 051] Context-Aware Pseudo-Label Scoring for Zero-Shot Video Summarization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17501v3](http://arxiv.org/pdf/2510.17501v3)**

> **作者:** Yuanli Wu; Long Zhang; Yue Du; Bin Li
>
> **摘要:** We propose a rubric-guided, pseudo-labeled, and prompt-driven zero-shot video summarization framework that bridges large language models with structured semantic reasoning. A small subset of human annotations is converted into high-confidence pseudo labels and organized into dataset-adaptive rubrics defining clear evaluation dimensions such as thematic relevance, action detail, and narrative progression. During inference, boundary scenes, including the opening and closing segments, are scored independently based on their own descriptions, while intermediate scenes incorporate concise summaries of adjacent segments to assess narrative continuity and redundancy. This design enables the language model to balance local salience with global coherence without any parameter tuning. Across three benchmarks, the proposed method achieves stable and competitive results, with F1 scores of 57.58 on SumMe, 63.05 on TVSum, and 53.79 on QFVS, surpassing zero-shot baselines by +0.85, +0.84, and +0.37, respectively. These outcomes demonstrate that rubric-guided pseudo labeling combined with contextual prompting effectively stabilizes LLM-based scoring and establishes a general, interpretable, and training-free paradigm for both generic and query-focused video summarization.
>
---
#### [replaced 052] Fast MRI for All: Bridging Access Gaps by Training without Raw Data
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.13022v3](http://arxiv.org/pdf/2411.13022v3)**

> **作者:** Yaşar Utku Alçalar; Merve Gülle; Mehmet Akçakaya
>
> **备注:** Neural Information Processing Systems (NeurIPS), 2025 (Spotlight)
>
> **摘要:** Physics-driven deep learning (PD-DL) approaches have become popular for improved reconstruction of fast magnetic resonance imaging (MRI) scans. Though PD-DL offers higher acceleration rates than existing clinical fast MRI techniques, their use has been limited outside specialized MRI centers. A key challenge is generalization to rare pathologies or different populations, noted in multiple studies, with fine-tuning on target populations suggested for improvement. However, current approaches for PD-DL training require access to raw k-space measurements, which is typically only available at specialized MRI centers that have research agreements for such data access. This is especially an issue for rural and under-resourced areas, where commercial MRI scanners only provide access to a final reconstructed image. To tackle these challenges, we propose Compressibility-inspired Unsupervised Learning via Parallel Imaging Fidelity (CUPID) for high-quality PD-DL training using only routine clinical reconstructed images exported from an MRI scanner. CUPID evaluates output quality with a compressibility-based approach while ensuring that the output stays consistent with the clinical parallel imaging reconstruction through well-designed perturbations. Our results show CUPID achieves similar quality to established PD-DL training that requires k-space data while outperforming compressed sensing (CS) and diffusion-based generative methods. We further demonstrate its effectiveness in a zero-shot training setup for retrospectively and prospectively sub-sampled acquisitions, attesting to its minimal training burden. As an approach that radically deviates from existing strategies, CUPID presents an opportunity to provide broader access to fast MRI for remote and rural populations in an attempt to reduce the obstacles associated with this expensive imaging modality.
>
---
#### [replaced 053] Advancing Image Super-resolution Techniques in Remote Sensing: A Comprehensive Survey
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23248v3](http://arxiv.org/pdf/2505.23248v3)**

> **作者:** Yunliang Qi; Meng Lou; Yimin Liu; Lu Li; Zhen Yang; Wen Nie
>
> **备注:** Accepted by ISPRS Journal of Photogrammetry and Remote Sensing
>
> **摘要:** Remote sensing image super-resolution (RSISR) is a crucial task in remote sensing image processing, aiming to reconstruct high-resolution (HR) images from their low-resolution (LR) counterparts. Despite the growing number of RSISR methods proposed in recent years, a systematic and comprehensive review of these methods is still lacking. This paper presents a thorough review of RSISR algorithms, covering methodologies, datasets, and evaluation metrics. We provide an in-depth analysis of RSISR methods, categorizing them into supervised, unsupervised, and quality evaluation approaches, to help researchers understand current trends and challenges. Our review also discusses the strengths, limitations, and inherent challenges of these techniques. Notably, our analysis reveals significant limitations in existing methods, particularly in preserving fine-grained textures and geometric structures under large-scale degradation. Based on these findings, we outline future research directions, highlighting the need for domain-specific architectures and robust evaluation protocols to bridge the gap between synthetic and real-world RSISR scenarios.
>
---
#### [replaced 054] Weakly Supervised Food Image Segmentation using Vision Transformers and Segment Anything Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.19028v2](http://arxiv.org/pdf/2509.19028v2)**

> **作者:** Ioannis Sarafis; Alexandros Papadopoulos; Anastasios Delopoulos
>
> **备注:** Accepted for presentation at the 20th International Workshop on Semantic and Social Media Adaptation & Personalization (SMAP 2025)
>
> **摘要:** In this paper, we propose a weakly supervised semantic segmentation approach for food images which takes advantage of the zero-shot capabilities and promptability of the Segment Anything Model (SAM) along with the attention mechanisms of Vision Transformers (ViTs). Specifically, we use class activation maps (CAMs) from ViTs to generate prompts for SAM, resulting in masks suitable for food image segmentation. The ViT model, a Swin Transformer, is trained exclusively using image-level annotations, eliminating the need for pixel-level annotations during training. Additionally, to enhance the quality of the SAM-generated masks, we examine the use of image preprocessing techniques in combination with single-mask and multi-mask SAM generation strategies. The methodology is evaluated on the FoodSeg103 dataset, generating an average of 2.4 masks per image (excluding background), and achieving an mIoU of 0.54 for the multi-mask scenario. We envision the proposed approach as a tool to accelerate food image annotation tasks or as an integrated component in food and nutrition tracking applications.
>
---
#### [replaced 055] Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23566v3](http://arxiv.org/pdf/2505.23566v3)**

> **作者:** Yu Li; Jin Jiang; Jianhua Zhu; Shuai Peng; Baole Wei; Yuxuan Zhou; Liangcai Gao
>
> **备注:** Accepted by NeurIPS 2025 as a spotlight
>
> **摘要:** Handwritten Mathematical Expression Recognition (HMER) remains a persistent challenge in Optical Character Recognition (OCR) due to the inherent freedom of symbol layouts and variability in handwriting styles. Prior methods have faced performance bottlenecks by proposing isolated architectural modifications, making them difficult to integrate coherently into a unified framework. Meanwhile, recent advances in pretrained vision-language models (VLMs) have demonstrated strong cross-task generalization, offering a promising foundation for developing unified solutions. In this paper, we introduce Uni-MuMER, which fully fine-tunes a VLM for the HMER task without modifying its architecture, effectively injecting domain-specific knowledge into a generalist framework. Our method integrates three data-driven tasks: Tree-Aware Chain-of-Thought (Tree-CoT) for structured spatial reasoning, Error-Driven Learning (EDL) for reducing confusion among visually similar characters, and Symbol Counting (SC) for improving recognition consistency in long expressions. Experiments on the CROHME and HME100K datasets show that Uni-MuMER achieves super state-of-the-art performance, outperforming the best lightweight specialized model SSAN by 16.31\% and the top-performing VLM Gemini2.5-flash by 24.42\% under zero-shot setting. Our datasets, models, and code are open-sourced at: {https://github.com/BFlameSwift/Uni-MuMER
>
---
#### [replaced 056] Towards Enhanced Image Generation Via Multi-modal Chain of Thought in Unified Generative Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.01298v2](http://arxiv.org/pdf/2503.01298v2)**

> **作者:** Yi Wang; Mushui Liu; Wanggui He; Hanyang Yuan; Longxiang Zhang; Ziwei Huang; Guanghao Zhang; Wenkai Fang; Haoze Jiang; Shengxuming Zhang; Dong She; Jinlong Liu; Weilong Dai; Mingli Song; Hao Jiang; Jie Song
>
> **摘要:** Unified generative models have shown remarkable performance in text and image generation. For image synthesis tasks, they adopt straightforward text-to-image (T2I) generation. However, direct T2I generation limits the models in handling complex compositional instructions, which frequently occur in real-world scenarios. Although this issue is vital, existing works mainly focus on improving the basic image generation capability of the models. While such improvements help to some extent, they still fail to adequately resolve the problem. Inspired by Chain of Thought (CoT) solving complex problems step by step, this work aims to introduce CoT into unified generative models to address the challenges of complex image generation that direct T2I generation cannot effectively solve, thereby endowing models with enhanced image generation ability. To achieve this, we first propose Functionality-oriented eXperts (FoXperts), an expert-parallel architecture in our model FoX, which assigns experts by function. FoXperts disentangles potential conflicts in mainstream modality-oriented designs and provides a solid foundation for CoT. When introducing CoT, the first question is how to design it for complex image generation. To this end, we emulate a human-like artistic workflow -- planning, acting, reflection, and correction -- and propose the Multimodal Chain of Thought (MCoT) approach, as the data involves both text and image. To address the subsequent challenge -- designing an effective MCoT training paradigm -- we develop a multi-task joint training scheme that equips the model with all capabilities required for each MCoT step in a disentangled manner. This paradigm avoids the difficulty of collecting consistent multi-step data tuples. Extensive experiments show that FoX consistently outperforms existing unified models on various T2I benchmarks, delivering notable improvements in complex image generation.
>
---
#### [replaced 057] See through the Dark: Learning Illumination-affined Representations for Nighttime Occupancy Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20641v3](http://arxiv.org/pdf/2505.20641v3)**

> **作者:** Yuan Wu; Zhiqiang Yan; Yigong Zhang; Xiang Li; Jian Yang
>
> **摘要:** Occupancy prediction aims to estimate the 3D spatial distribution of occupied regions along with their corresponding semantic labels. Existing vision-based methods perform well on daytime benchmarks but struggle in nighttime scenarios due to limited visibility and challenging lighting conditions. To address these challenges, we propose LIAR, a novel framework that learns illumination-affined representations. LIAR first introduces Selective Low-light Image Enhancement (SLLIE), which leverages the illumination priors from daytime scenes to adaptively determine whether a nighttime image is genuinely dark or sufficiently well-lit, enabling more targeted global enhancement. Building on the illumination maps generated by SLLIE, LIAR further incorporates two illumination-aware components: 2D Illumination-guided Sampling (2D-IGS) and 3D Illumination-driven Projection (3D-IDP), to respectively tackle local underexposure and overexposure. Specifically, 2D-IGS modulates feature sampling positions according to illumination maps, assigning larger offsets to darker regions and smaller ones to brighter regions, thereby alleviating feature degradation in underexposed areas. Subsequently,3D-IDP enhances semantic understanding in overexposed regions by constructing illumination intensity fields and supplying refined residual queries to the BEV context refinement process. Extensive experiments on both real and synthetic datasets demonstrate the superior performance of LIAR under challenging nighttime scenarios. The source code and pretrained models are available [here](https://github.com/yanzq95/LIAR).
>
---
#### [replaced 058] WikiVideo: Article Generation from Multiple Videos
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.00939v2](http://arxiv.org/pdf/2504.00939v2)**

> **作者:** Alexander Martin; Reno Kriz; William Gantt Walden; Kate Sanders; Hannah Recknor; Eugene Yang; Francis Ferraro; Benjamin Van Durme
>
> **备注:** Repo can be found here: https://github.com/alexmartin1722/wikivideo
>
> **摘要:** We introduce the task of grounded article generation with the goal of creating a Wikipedia-style article from multiple diverse videos about real-world events -- from natural disasters to political elections -- where all the information in the article is supported by video evidence. Videos are intuitive sources for retrieval-augmented generation (RAG), but most contemporary RAG workflows focus heavily on text while existing methods for video-based summarization focus on low-level scene understanding rather than high-level event semantics. To close this gap, we introduce WikiVideo, a benchmark consisting of expert-written articles and densely annotated videos that provide evidence for articles' claims, facilitating the integration of video into RAG pipelines and enabling the creation of in-depth content that is grounded in multimodal sources. We further propose Collaborative Article Generation (CAG), a novel interactive method for article creation from multiple videos. CAG leverages an iterative interaction between an r1-style reasoning model and a VideoLLM to draw higher-level inferences about the target event than is possible with VideoLLMs alone, which fixate on low-level visual features. We benchmark state-of-the-art VideoLLMs and CAG in both oracle retrieval and RAG settings and find that CAG consistently outperforms alternative methods, while suggesting intriguing avenues for future work.
>
---
#### [replaced 059] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20120v3](http://arxiv.org/pdf/2502.20120v3)**

> **作者:** QingYuan Jiang; Longfei Huang; Yang Yang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Multimodal learning (MML) is significantly constrained by modality imbalance, leading to suboptimal performance in practice. While existing approaches primarily focus on balancing the learning of different modalities to address this issue, they fundamentally overlook the inherent disproportion in model classification ability, which serves as the primary cause of this phenomenon. In this paper, we propose a novel multimodal learning approach to dynamically balance the classification ability of weak and strong modalities by incorporating the principle of boosting. Concretely, we first propose a sustained boosting algorithm in multimodal learning by simultaneously optimizing the classification and residual errors. Subsequently, we introduce an adaptive classifier assignment strategy to dynamically facilitate the classification performance of the weak modality. Furthermore, we theoretically analyze the convergence property of the cross-modal gap function, ensuring the effectiveness of the proposed boosting scheme. To this end, the classification ability of strong and weak modalities is expected to be balanced, thereby mitigating the imbalance issue. Empirical experiments on widely used datasets reveal the superiority of our method through comparison with various state-of-the-art (SOTA) multimodal learning baselines. The source code is available at https://github.com/njustkmg/NeurIPS25-AUG.
>
---
#### [replaced 060] Rethinking Backbone Design for Lightweight 3D Object Detection in LiDAR
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00744v2](http://arxiv.org/pdf/2508.00744v2)**

> **作者:** Adwait Chandorkar; Hasan Tercan; Tobias Meisen
>
> **备注:** Best Paper Award at the Embedded Vision Workshop ICCV 2025
>
> **摘要:** Recent advancements in LiDAR-based 3D object detection have significantly accelerated progress toward the realization of fully autonomous driving in real-world environments. Despite achieving high detection performance, most of the approaches still rely on a VGG-based or ResNet-based backbone for feature exploration, which increases the model complexity. Lightweight backbone design is well-explored for 2D object detection, but research on 3D object detection still remains limited. In this work, we introduce Dense Backbone, a lightweight backbone that combines the benefits of high processing speed, lightweight architecture, and robust detection accuracy. We adapt multiple SoTA 3d object detectors, such as PillarNet, with our backbone and show that with our backbone, these models retain most of their detection capability at a significantly reduced computational cost. To our knowledge, this is the first dense-layer-based backbone tailored specifically for 3D object detection from point cloud data. DensePillarNet, our adaptation of PillarNet, achieves a 29% reduction in model parameters and a 28% reduction in latency with just a 2% drop in detection accuracy on the nuScenes test set. Furthermore, Dense Backbone's plug-and-play design allows straightforward integration into existing architectures, requiring no modifications to other network components.
>
---
#### [replaced 061] Semi-off-Policy Reinforcement Learning for Vision-Language Slow-Thinking Reasoning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16814v2](http://arxiv.org/pdf/2507.16814v2)**

> **作者:** Junhao Shen; Haiteng Zhao; Yuzhe Gu; Songyang Gao; Kuikun Liu; Haian Huang; Jianfei Gao; Dahua Lin; Wenwei Zhang; Kai Chen
>
> **摘要:** Enhancing large vision-language models (LVLMs) with visual slow-thinking reasoning is crucial for solving complex multimodal tasks. However, since LVLMs are mainly trained with vision-language alignment, it is difficult to adopt on-policy reinforcement learning (RL) to develop the slow thinking ability because the rollout space is restricted by its initial abilities. Off-policy RL offers a way to go beyond the current policy, but directly distilling trajectories from external models may cause visual hallucinations due to mismatched visual perception abilities across models. To address these issues, this paper proposes SOPHIA, a simple and scalable Semi-Off-Policy RL for vision-language slow-tHInking reAsoning. SOPHIA builds a semi-off-policy behavior model by combining on-policy visual understanding from a trainable LVLM with off-policy slow-thinking reasoning from a language model, assigns outcome-based rewards to reasoning, and propagates visual rewards backward. Then LVLM learns slow-thinking reasoning ability from the obtained reasoning trajectories using propagated rewards via off-policy RL algorithms. Extensive experiments with InternVL2.5 and InternVL3.0 with 8B and 38B sizes show the effectiveness of SOPHIA. Notably, SOPHIA improves InternVL3.0-38B by 8.50% in average, reaching state-of-the-art performance among open-source LVLMs on multiple multimodal reasoning benchmarks, and even outperforms some closed-source models (e.g., GPT-4.1) on the challenging MathVision and OlympiadBench, achieving 49.08% and 49.95% pass@1 accuracy, respectively. Analysis shows SOPHIA outperforms supervised fine-tuning and direct on-policy RL methods, offering a better policy initialization for further on-policy training.
>
---
#### [replaced 062] VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.15530v2](http://arxiv.org/pdf/2510.15530v2)**

> **作者:** Zehao Ni; Yonghao He; Lingfeng Qian; Jilei Mao; Fa Fu; Wei Sui; Hu Su; Junran Peng; Zhipeng Wang; Bin He
>
> **摘要:** In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator.
>
---
#### [replaced 063] EgoBlind: Towards Egocentric Visual Assistance for the Blind
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.08221v3](http://arxiv.org/pdf/2503.08221v3)**

> **作者:** Junbin Xiao; Nanxin Huang; Hao Qiu; Zhulin Tao; Xun Yang; Richang Hong; Meng Wang; Angela Yao
>
> **备注:** NeurIPS'25 (D&B Track)
>
> **摘要:** We present EgoBlind, the first egocentric VideoQA dataset collected from blind individuals to evaluate the assistive capabilities of contemporary multimodal large language models (MLLMs). EgoBlind comprises 1,392 first-person videos from the daily lives of blind and visually impaired individuals. It also features 5,311 questions directly posed or verified by the blind to reflect their in-situation needs for visual assistance. Each question has an average of 3 manually annotated reference answers to reduce subjectiveness. Using EgoBlind, we comprehensively evaluate 16 advanced MLLMs and find that all models struggle. The best performers achieve an accuracy near 60\%, which is far behind human performance of 87.4\%. To guide future advancements, we identify and summarize major limitations of existing MLLMs in egocentric visual assistance for the blind and explore heuristic solutions for improvement. With these efforts, we hope that EgoBlind will serve as a foundation for developing effective AI assistants to enhance the independence of the blind and visually impaired. Data and code are available at https://github.com/doc-doc/EgoBlind.
>
---
#### [replaced 064] Adversarial Attacks on LiDAR-Based Tracking Across Road Users: Robustness Evaluation and Target-Aware Black-Box Method
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.20893v3](http://arxiv.org/pdf/2410.20893v3)**

> **作者:** Shengjing Tian; Xiantong Zhao; Yuhao Bian; Yinan Han; Bin Liu
>
> **摘要:** In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.
>
---
#### [replaced 065] 3D Visual Illusion Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13061v4](http://arxiv.org/pdf/2505.13061v4)**

> **作者:** Chengtang Yao; Zhidan Liu; Jiaxi Zeng; Lidong Yu; Yuwei Wu; Yunde Jia
>
> **备注:** NeurIPS 2025, Project: https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation
>
> **摘要:** 3D visual illusion is a perceptual phenomenon where a two-dimensional plane is manipulated to simulate three-dimensional spatial relationships, making a flat artwork or object look three-dimensional in the human visual system. In this paper, we reveal that the machine visual system is also seriously fooled by 3D visual illusions, including monocular and binocular depth estimation. In order to explore and analyze the impact of 3D visual illusion on depth estimation, we collect a large dataset containing almost 3k scenes and 200k images to train and evaluate SOTA monocular and binocular depth estimation methods. We also propose a 3D visual illusion depth estimation framework that uses common sense from the vision language model to adaptively fuse depth from binocular disparity and monocular depth. Experiments show that SOTA monocular, binocular, and multi-view depth estimation approaches are all fooled by various 3D visual illusions, while our method achieves SOTA performance.
>
---
#### [replaced 066] Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20755v4](http://arxiv.org/pdf/2505.20755v4)**

> **作者:** Yifei Wang; Weimin Bai; Colin Zhang; Debing Zhang; Weijian Luo; He Sun
>
> **摘要:** In this paper, we unify more than 10 existing one-step diffusion distillation approaches, such as Diff-Instruct, DMD, SIM, SiD, $f$-distill, etc, inside a theory-driven framework which we name the \textbf{\emph{Uni-Instruct}}. Uni-Instruct is motivated by our proposed diffusion expansion theory of the $f$-divergence family. Then we introduce key theories that overcome the intractability issue of the original expanded $f$-divergence, resulting in an equivalent yet tractable loss that effectively trains one-step diffusion models by minimizing the expanded $f$-divergence family. The novel unification introduced by Uni-Instruct not only offers new theoretical contributions that help understand existing approaches from a high-level perspective but also leads to state-of-the-art one-step diffusion generation performances. On the CIFAR10 generation benchmark, Uni-Instruct achieves record-breaking Frechet Inception Distance (FID) values of \textbf{\emph{1.46}} for unconditional generation and \textbf{\emph{1.38}} for conditional generation. On the ImageNet-$64\times 64$ generation benchmark, Uni-Instruct achieves a new SoTA one-step generation FID of \textbf{\emph{1.02}}, which outperforms its 79-step teacher diffusion with a significant improvement margin of 1.33 (1.02 vs 2.35). We also apply Uni-Instruct on broader tasks like text-to-3D generation. For text-to-3D generation, Uni-Instruct gives decent results, which slightly outperforms previous methods, such as SDS and VSD, in terms of both generation quality and diversity. Both the solid theoretical and empirical contributions of Uni-Instruct will potentially help future studies on one-step diffusion distillation and knowledge transferring of diffusion models.
>
---
#### [replaced 067] Vectorization of Persistence Diagrams for Topological Data Analysis in R and Python Using TDAvec Package
- **分类: math.AT; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17340v3](http://arxiv.org/pdf/2411.17340v3)**

> **作者:** Aleksei Luchinsky; Umar Islambekov
>
> **备注:** 9 pages, 3 figures, 3 tables; minor changes: two more vectorizations are described
>
> **摘要:** Persistent homology is a widely-used tool in topological data analysis (TDA) for understanding the underlying shape of complex data. By constructing a filtration of simplicial complexes from data points, it captures topological features such as connected components, loops, and voids across multiple scales. These features are encoded in persistence diagrams (PDs), which provide a concise summary of the data's topological structure. However, the non-Hilbert nature of the space of PDs poses challenges for their direct use in machine learning applications. To address this, kernel methods and vectorization techniques have been developed to transform PDs into machine-learning-compatible formats. In this paper, we introduce a new software package designed to streamline the vectorization of PDs, offering an intuitive workflow and advanced functionalities. We demonstrate the necessity of the package through practical examples and provide a detailed discussion on its contributions to applied TDA. Definitions of all vectorization summaries used in the package are included in the appendix.
>
---
#### [replaced 068] Brain3D: Generating 3D Objects from fMRI
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.15239v5](http://arxiv.org/pdf/2405.15239v5)**

> **作者:** Yuankun Yang; Li Zhang; Ziyang Xie; Zhiyuan Yuan; Jianfeng Feng; Xiatian Zhu; Yu-Gang Jiang
>
> **备注:** IJCV 2025
>
> **摘要:** Understanding the hidden mechanisms behind human's visual perception is a fundamental question in neuroscience. To that end, investigating into the neural responses of human mind activities, such as functional Magnetic Resonance Imaging (fMRI), has been a significant research vehicle. However, analyzing fMRI signals is challenging, costly, daunting, and demanding for professional training. Despite remarkable progress in fMRI analysis, existing approaches are limited to generating 2D images and far away from being biologically meaningful and practically useful. Under this insight, we propose to generate visually plausible and functionally more comprehensive 3D outputs decoded from brain signals, enabling more sophisticated modeling of fMRI data. Conceptually, we reformulate this task as a {\em fMRI conditioned 3D object generation} problem. We design a novel 3D object representation learning method, Brain3D, that takes as input the fMRI data of a subject who was presented with a 2D image, and yields as output the corresponding 3D object images. The key capabilities of this model include tackling the noises with high-level semantic signals and a two-stage architecture design for progressive high-level information integration. Extensive experiments validate the superior capability of our model over previous state-of-the-art 3D object generation methods. Importantly, we show that our model captures the distinct functionalities of each region of human vision system as well as their intricate interplay relationships, aligning remarkably with the established discoveries in neuroscience. Further, preliminary evaluations indicate that Brain3D can successfully identify the disordered brain regions in simulated scenarios, such as V1, V2, V3, V4, and the medial temporal lobe (MTL) within the human visual system. Our data and code will be available at https://brain-3d.github.io/.
>
---
#### [replaced 069] Saccade crossing avoidance as a visual search strategy
- **分类: q-bio.NC; cs.CV; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2508.18404v2](http://arxiv.org/pdf/2508.18404v2)**

> **作者:** Alex Szorkovszky; Rujeena Mathema; Pedro Lencastre; Pedro Lind; Anis Yazidi
>
> **备注:** Main text: 12 pages, 4 figures; Supplementary info: 13 pages, 9 figures
>
> **摘要:** Although visual search appears largely random, several oculomotor biases exist such that the likelihoods of saccade directions and lengths depend on the previous scan path. Compared to the most recent fixations, the impact of the longer path history is more difficult to quantify. Using the step-selection framework commonly used in movement ecology, and analyzing data from 45-second viewings of "Where's Waldo?", we report a new memory-dependent effect that also varies significantly between individuals, which we term self-crossing avoidance. This is a tendency for saccades to avoid crossing those earlier in the scan path, and is most evident when both have small amplitudes. We show this by comparing real data to synthetic data generated from a memoryless approximation of the spatial statistics (i.e. a Markovian nonparametric model with a matching distribution of saccade lengths over time). Maximum likelihood fitting indicates that this effect is strongest when including the last $\approx 7$ seconds of a scan path. The effect size is comparable to well-known forms of history dependence such as inhibition of return. A parametric probabilistic model including a self-crossing penalty term was able to reproduce joint statistics of saccade lengths and self-crossings. We also quantified individual strategic differences, and their consistency over the six images viewed per participant, using mixed-effect regressions. Participants with a higher tendency to avoid crossings displayed smaller saccade lengths and shorter fixation durations on average, but did not display more horizontal, vertical, forward or reverse saccades. Together, these results indicate that the avoidance of crossings is a local orienting strategy that facilitates and complements inhibition of return, and hence exploration of visual scenes.
>
---
#### [replaced 070] Investigating the Relationship between the Weighted Figure of Merit and Rosin's Measure
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05749v3](http://arxiv.org/pdf/2506.05749v3)**

> **作者:** Bimal Kumar Ray
>
> **摘要:** Many studies have been conducted to solve the problem of approximating a digital boundary by piece straight-line segments for the further processing required in computer vision applications. The authors of these studies compared their schemes to determine the best one. The initial measure used to assess the goodness of fit of a polygonal approximation was the figure of merit. Later,it was noted that this measure was not an appropriate metric for a valid reason which is why Rosin-through mathematical analysis-introduced a measure called merit. However,this measure involves an optimal scheme of polygonal approximation,so it is time-consuming to compute it to assess the goodness of fit of an approximation. This led many researchers to use a weighted figure of merit as a substitute for Rosin's measure to compare sub optimal schemes. An attempt is made in this communication to investigate whether the two measures-weighted figure of merit and Rosin's measure-are related so that one can be used instead of the other, and toward this end, theoretical analysis, experimental investigation and statistical analysis are carried out. The mathematical formulas for the weighted figure of merit and Rosin's measure are analyzed, and through proof of theorems,it is found that the two measures are theoretically independent of each other. The graphical analysis of experiments carried out using a public dataset supports the results of the theoretical analysis. The statistical analysis via Pearson's correlation coefficient and non-linear correlation measure also revealed that the two measures are uncorrelated. This analysis leads one to conclude that if a suboptimal scheme is found to be better (worse) than some other suboptimal scheme,as indicated by Rosin's measure,then the same conclusion cannot be drawn using a weighted figure of merit,so one cannot use a weighted figure of merit instead of Rosin's measure.
>
---
#### [replaced 071] Unfolding Generative Flows with Koopman Operators: Fast and Interpretable Sampling
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22304v2](http://arxiv.org/pdf/2506.22304v2)**

> **作者:** Erkan Turan; Aristotelis Siozopoulos; Louis Martinez; Julien Gaubil; Emery Pierson; Maks Ovsjanikov
>
> **摘要:** Continuous Normalizing Flows (CNFs) enable elegant generative modeling but remain bottlenecked by slow sampling: producing a single sample requires solving a nonlinear ODE with hundreds of function evaluations. Recent approaches such as Rectified Flow and OT-CFM accelerate sampling by straightening trajectories, yet the learned dynamics remain nonlinear black boxes, limiting both efficiency and interpretability. We propose a fundamentally different perspective: globally linearizing flow dynamics via Koopman theory. By lifting Conditional Flow Matching (CFM) into a higher-dimensional Koopman space, we represent its evolution with a single linear operator. This yields two key benefits. First, sampling becomes one-step and parallelizable, computed in closed form via the matrix exponential. Second, the Koopman operator provides a spectral blueprint of generation, enabling novel interpretability through its eigenvalues and modes. We derive a practical, simulation-free training objective that enforces infinitesimal consistency with the teacher's dynamics and show that this alignment preserves fidelity along the full generative path, distinguishing our method from boundary-only distillation. Empirically, our approach achieves competitive sample quality with dramatic speedups, while uniquely enabling spectral analysis of generative flows.
>
---
#### [replaced 072] ImagerySearch: Adaptive Test-Time Search for Video Generation Beyond Semantic Dependency Constraints
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14847v2](http://arxiv.org/pdf/2510.14847v2)**

> **作者:** Meiqi Wu; Jiashu Zhu; Xiaokun Feng; Chubin Chen; Chen Zhu; Bingze Song; Fangyuan Mao; Jiahong Wu; Xiangxiang Chu; Kaiqi Huang
>
> **摘要:** Video generation models have achieved remarkable progress, particularly excelling in realistic scenarios; however, their performance degrades notably in imaginative scenarios. These prompts often involve rarely co-occurring concepts with long-distance semantic relationships, falling outside training distributions. Existing methods typically apply test-time scaling for improving video quality, but their fixed search spaces and static reward designs limit adaptability to imaginative scenarios. To fill this gap, we propose ImagerySearch, a prompt-guided adaptive test-time search strategy that dynamically adjusts both the inference search space and reward function according to semantic relationships in the prompt. This enables more coherent and visually plausible videos in challenging imaginative settings. To evaluate progress in this direction, we introduce LDT-Bench, the first dedicated benchmark for long-distance semantic prompts, consisting of 2,839 diverse concept pairs and an automated protocol for assessing creative generation capabilities. Extensive experiments show that ImagerySearch consistently outperforms strong video generation baselines and existing test-time scaling approaches on LDT-Bench, and achieves competitive improvements on VBench, demonstrating its effectiveness across diverse prompt types. We will release LDT-Bench and code to facilitate future research on imaginative video generation.
>
---
#### [replaced 073] Flexible-length Text Infilling for Discrete Diffusion Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13579v2](http://arxiv.org/pdf/2506.13579v2)**

> **作者:** Andrew Zhang; Anushka Sivakumar; Chiawei Tang; Chris Thomas
>
> **备注:** Major edit of methodology section. Matches EMNLP camera-ready version
>
> **摘要:** Discrete diffusion models are a new class of text generators that offer advantages such as bidirectional context use, parallelizable generation, and flexible prompting compared to autoregressive models. However, a critical limitation of discrete diffusion models is their inability to perform flexible-length or flexible-position text infilling without access to ground-truth positional data. We introduce \textbf{DDOT} (\textbf{D}iscrete \textbf{D}iffusion with \textbf{O}ptimal \textbf{T}ransport Position Coupling), the first discrete diffusion model to overcome this challenge. DDOT jointly denoises token values and token positions, employing a novel sample-level Optimal Transport (OT) coupling. This coupling preserves relative token ordering while dynamically adjusting the positions and length of infilled segments, a capability previously missing in text diffusion. Our method is orthogonal to existing discrete text diffusion methods and is compatible with various pretrained text denoisers. Extensive experiments on text infilling benchmarks such as One-Billion-Word and Yelp demonstrate that DDOT outperforms naive diffusion baselines. Furthermore, DDOT achieves performance on par with state-of-the-art non-autoregressive models and enables significant improvements in training efficiency and flexibility.
>
---
#### [replaced 074] With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.16895v2](http://arxiv.org/pdf/2506.16895v2)**

> **作者:** Fabian Gröger; Shuo Wen; Huyen Le; Maria Brbić
>
> **备注:** NeurIPS 2025 camera-ready
>
> **摘要:** Multimodal models have demonstrated powerful capabilities in complex tasks requiring multimodal alignment, including zero-shot classification and cross-modal retrieval. However, existing models typically rely on millions of paired multimodal samples, which are prohibitively expensive or infeasible to obtain in many domains. In this work, we explore the feasibility of building multimodal models with limited amount of paired data by aligning pretrained unimodal foundation models. We show that high-quality alignment is possible with as few as tens of thousands of paired samples$\unicode{x2013}$less than $1\%$ of the data typically used in the field. To achieve this, we introduce STRUCTURE, an effective regularization technique that preserves the neighborhood geometry of the latent space of unimodal encoders. Additionally, we show that aligning last layers is often suboptimal and demonstrate the benefits of aligning the layers with the highest representational similarity across modalities. These two components can be readily incorporated into existing alignment methods, yielding substantial gains across 24 zero-shot image classification and retrieval benchmarks, with average relative improvement of $51.6\%$ in classification and $91.8\%$ in retrieval tasks. Our results highlight the effectiveness and broad applicability of our framework for limited-sample multimodal learning and offer a promising path forward for resource-constrained domains.
>
---
#### [replaced 075] Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.15568v5](http://arxiv.org/pdf/2508.15568v5)**

> **作者:** Youjia Zhang; Youngeun Kim; Young-Geun Choi; Hongyeob Kim; Huiling Liu; Sungeun Hong
>
> **摘要:** Test-time adaptation (TTA) enhances the zero-shot robustness under distribution shifts by leveraging unlabeled test data during inference. Despite notable advances, several challenges still limit its broader applicability. First, most methods rely on backpropagation or iterative optimization, which limits scalability and hinders real-time deployment. Second, they lack explicit modeling of class-conditional feature distributions. This modeling is crucial for producing reliable decision boundaries and calibrated predictions, but it remains underexplored due to the lack of both source data and supervision at test time. In this paper, we propose ADAPT, an Advanced Distribution-Aware and backPropagation-free Test-time adaptation method. We reframe TTA as a Gaussian probabilistic inference task by modeling class-conditional likelihoods using gradually updated class means and a shared covariance matrix. This enables closed-form, training-free inference. To correct potential likelihood bias, we introduce lightweight regularization guided by CLIP priors and a historical knowledge bank. ADAPT requires no source data, no gradient updates, and no full access to target data, supporting both online and transductive settings. Extensive experiments across diverse benchmarks demonstrate that our method achieves state-of-the-art performance under a wide range of distribution shifts with superior scalability and robustness.
>
---
#### [replaced 076] Chimera: Compositional Image Generation using Part-based Concepting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.18083v2](http://arxiv.org/pdf/2510.18083v2)**

> **作者:** Shivam Singh; Yiming Chen; Agneet Chatterjee; Amit Raj; James Hays; Yezhou Yang; Chitta Baral
>
> **摘要:** Personalized image generative models are highly proficient at synthesizing images from text or a single image, yet they lack explicit control for composing objects from specific parts of multiple source images without user specified masks or annotations. To address this, we introduce Chimera, a personalized image generation model that generates novel objects by combining specified parts from different source images according to textual instructions. To train our model, we first construct a dataset from a taxonomy built on 464 unique (part, subject) pairs, which we term semantic atoms. From this, we generate 37k prompts and synthesize the corresponding images with a high-fidelity text-to-image model. We train a custom diffusion prior model with part-conditional guidance, which steers the image-conditioning features to enforce both semantic identity and spatial layout. We also introduce an objective metric PartEval to assess the fidelity and compositional accuracy of generation pipelines. Human evaluations and our proposed metric show that Chimera outperforms other baselines by 14% in part alignment and compositional accuracy and 21% in visual quality.
>
---
#### [replaced 077] MUG-V 10B: High-efficiency Training Pipeline for Large Video Generation Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17519v2](http://arxiv.org/pdf/2510.17519v2)**

> **作者:** Yongshun Zhang; Zhongyi Fan; Yonghang Zhang; Zhangzikang Li; Weifeng Chen; Zhongwei Feng; Chaoyue Wang; Peng Hou; Anxiang Zeng
>
> **备注:** Technical Report; Project Page: https://github.com/Shopee-MUG/MUG-V
>
> **摘要:** In recent years, large-scale generative models for visual content (\textit{e.g.,} images, videos, and 3D objects/scenes) have made remarkable progress. However, training large-scale video generation models remains particularly challenging and resource-intensive due to cross-modal text-video alignment, the long sequences involved, and the complex spatiotemporal dependencies. To address these challenges, we present a training framework that optimizes four pillars: (i) data processing, (ii) model architecture, (iii) training strategy, and (iv) infrastructure for large-scale video generation models. These optimizations delivered significant efficiency gains and performance improvements across all stages of data preprocessing, video compression, parameter scaling, curriculum-based pretraining, and alignment-focused post-training. Our resulting model, MUG-V 10B, matches recent state-of-the-art video generators overall and, on e-commerce-oriented video generation tasks, surpasses leading open-source baselines in human evaluations. More importantly, we open-source the complete stack, including model weights, Megatron-Core-based large-scale training code, and inference pipelines for video generation and enhancement. To our knowledge, this is the first public release of large-scale video generation training code that exploits Megatron-Core to achieve high training efficiency and near-linear multi-node scaling, details are available in https://github.com/Shopee-MUG/MUG-V.
>
---
#### [replaced 078] ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding
- **分类: cs.CV; cs.MM; Multimedia**

- **链接: [http://arxiv.org/pdf/2412.12718v2](http://arxiv.org/pdf/2412.12718v2)**

> **作者:** Zhenxing Zhang; Yaxiong Wang; Lechao Cheng; Zhun Zhong; Dan Guo; Meng Wang
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** We present ASAP, a new framework for detecting and grounding multi-modal media manipulation (DGM4).Upon thorough examination, we observe that accurate fine-grained cross-modal semantic alignment between the image and text is vital for accurately manipulation detection and grounding. While existing DGM4 methods pay rare attention to the cross-modal alignment, hampering the accuracy of manipulation detecting to step further. To remedy this issue, this work targets to advance the semantic alignment learning to promote this task. Particularly, we utilize the off-the-shelf Multimodal Large-Language Models (MLLMs) and Large Language Models (LLMs) to construct paired image-text pairs, especially for the manipulated instances. Subsequently, a cross-modal alignment learning is performed to enhance the semantic alignment. Besides the explicit auxiliary clues, we further design a Manipulation-Guided Cross Attention (MGCA) to provide implicit guidance for augmenting the manipulation perceiving. With the grounding truth available during training, MGCA encourages the model to concentrate more on manipulated components while downplaying normal ones, enhancing the model's ability to capture manipulations. Extensive experiments are conducted on the DGM4 dataset, the results demonstrate that our model can surpass the comparison method with a clear margin.
>
---
#### [replaced 079] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24625v3](http://arxiv.org/pdf/2505.24625v3)**

> **作者:** Duo Zheng; Shijia Huang; Yanyang Li; Liwei Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Previous research has investigated the application of Multimodal Large Language Models (MLLMs) in understanding 3D scenes by interpreting them as videos. These approaches generally depend on comprehensive 3D data inputs, such as point clouds or reconstructed Bird's-Eye View (BEV) maps. In our research, we advance this field by enhancing the capability of MLLMs to understand and reason in 3D spaces directly from video data, without the need for additional 3D input. We propose a novel and efficient method called the Video-3D Geometry Large Language Model (VG LLM). Our approach utilizes a 3D visual geometry encoder to extract 3D prior information from video sequences. This information is then integrated with visual tokens and input into the MLLM. Extensive experiments have shown that our method has achieved substantial improvements in various tasks related to 3D scene understanding and spatial reasoning, all directly learned from video sources. Impressively, our 4B model, which does not rely on explicit 3D data inputs, achieves competitive results compared to existing state-of-the-art methods, and even surpasses the Gemini-1.5-Pro in the VSI-Bench evaluations.
>
---
#### [replaced 080] MsEdF: A Multi-stream Encoder-decoder Framework for Remote Sensing Image Captioning
- **分类: cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09282v3](http://arxiv.org/pdf/2502.09282v3)**

> **作者:** Swadhin Das; Raksha Sharma
>
> **摘要:** Remote sensing images contain complex spatial patterns and semantic structures, which makes the captioning model difficult to accurately describe. Encoder-decoder architectures have become the widely used approach for RSIC by translating visual content into descriptive text. However, many existing methods rely on a single-stream architecture, which weakens the model to accurately describe the image. Such single-stream architectures typically struggle to extract diverse spatial features or capture complex semantic relationships, limiting their effectiveness in scenes with high intraclass similarity or contextual ambiguity. In this work, we propose a novel Multi-stream Encoder-decoder Framework (MsEdF) which improves the performance of RSIC by optimizing both the spatial representation and language generation of encoder-decoder architecture. The encoder fuses information from two complementary image encoders, thereby promoting feature diversity through the integration of multiscale and structurally distinct cues. To improve the capture of context-aware descriptions, we refine the input sequence's semantic modeling on the decoder side using a stacked GRU architecture with an element-wise aggregation scheme. Experiments on three benchmark RSIC datasets show that MsEdF outperforms several baseline models.
>
---
#### [replaced 081] SAM 2++: Tracking Anything at Any Granularity
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.18822v2](http://arxiv.org/pdf/2510.18822v2)**

> **作者:** Jiaming Zhang; Cheng Liang; Yichun Yang; Chenkai Zeng; Yutao Cui; Xinwen Zhang; Xin Zhou; Kai Ma; Gangshan Wu; Limin Wang
>
> **备注:** update results
>
> **摘要:** Video tracking aims at finding the specific target in subsequent frames given its initial state. Due to the varying granularity of target states across different tasks, most existing trackers are tailored to a single task and heavily rely on custom-designed modules within the individual task, which limits their generalization and leads to redundancy in both model design and parameters. To unify video tracking tasks, we present SAM 2++, a unified model towards tracking at any granularity, including masks, boxes, and points. First, to extend target granularity, we design task-specific prompts to encode various task inputs into general prompt embeddings, and a unified decoder to unify diverse task results into a unified form pre-output. Next, to satisfy memory matching, the core operation of tracking, we introduce a task-adaptive memory mechanism that unifies memory across different granularities. Finally, we introduce a customized data engine to support tracking training at any granularity, producing a large and diverse video tracking dataset with rich annotations at three granularities, termed Tracking-Any-Granularity, which represents a comprehensive resource for training and benchmarking on unified tracking. Comprehensive experiments on multiple benchmarks confirm that SAM 2++ sets a new state of the art across diverse tracking tasks at different granularities, establishing a unified and robust tracking framework.
>
---
#### [replaced 082] Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow
- **分类: cs.MA; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.21789v2](http://arxiv.org/pdf/2509.21789v2)**

> **作者:** Xinlei Yu; Chengming Xu; Guibin Zhang; Yongbo He; Zhangquan Chen; Zhucun Xue; Jiangning Zhang; Yue Liao; Xiaobin Hu; Yu-Gang Jiang; Shuicheng Yan
>
> **摘要:** Multi-Agent System (MAS) powered by Visual Language Models (VLMs) enables challenging tasks but suffers from a novel failure term, multi-agent visual hallucination snowballing, where hallucinations are seeded in a single agent and amplified by following ones due to the over-reliance on textual flow to relay visual information. Through turn-, layer-, and token-wise attention analyses, we provide detailed insights into the essence of hallucination snowballing regarding the reduction of visual attention allocation. It leads us to identify a subset of vision tokens with a unimodal attention peak in middle layers that best preserve visual evidence but gradually diminish in deeper agent turns, resulting in the visual hallucination snowballing in MAS. Thus, we propose ViF, a lightweight, plug-and-play mitigation paradigm that relays inter-agent messages with Visual Flow powered by the selected visual relay tokens and applies attention reallocation to amplify this pattern. The experiment results demonstrate that our method markedly reduces hallucination snowballing, consistently improving the performance across eight benchmarks based on four common MAS structures and ten base models. The source code is publicly available at: https://github.com/YU-deep/ViF.git.
>
---
