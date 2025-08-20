# 计算机视觉 cs.CV

- **最新发布 118 篇**

- **更新 67 篇**

## 最新发布

#### [new 001] MIRAGE: Towards AI-Generated Image Detection in the Wild
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.13223v1](http://arxiv.org/pdf/2508.13223v1)**

> **作者:** Cheng Xia; Manxi Lin; Jiexiang Tan; Xiaoxiong Du; Yang Qiu; Junjun Zheng; Xiangheng Kong; Yuning Jiang; Bo Zheng
>
> **摘要:** The spreading of AI-generated images (AIGI), driven by advances in generative AI, poses a significant threat to information security and public trust. Existing AIGI detectors, while effective against images in clean laboratory settings, fail to generalize to in-the-wild scenarios. These real-world images are noisy, varying from ``obviously fake" images to realistic ones derived from multiple generative models and further edited for quality control. We address in-the-wild AIGI detection in this paper. We introduce Mirage, a challenging benchmark designed to emulate the complexity of in-the-wild AIGI. Mirage is constructed from two sources: (1) a large corpus of Internet-sourced AIGI verified by human experts, and (2) a synthesized dataset created through the collaboration between multiple expert generators, closely simulating the realistic AIGI in the wild. Building on this benchmark, we propose Mirage-R1, a vision-language model with heuristic-to-analytic reasoning, a reflective reasoning mechanism for AIGI detection. Mirage-R1 is trained in two stages: a supervised-fine-tuning cold start, followed by a reinforcement learning stage. By further adopting an inference-time adaptive thinking strategy, Mirage-R1 is able to provide either a quick judgment or a more robust and accurate conclusion, effectively balancing inference speed and performance. Extensive experiments show that our model leads state-of-the-art detectors by 5% and 10% on Mirage and the public benchmark, respectively. The benchmark and code will be made publicly available.
>
---
#### [new 002] The 9th AI City Challenge
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文介绍第九届AI City Challenge，聚焦计算机视觉与AI在交通、工业自动化和公共安全中的应用。解决多摄像头3D跟踪、视频问答、空间推理及边缘设备高效检测等问题，通过四个赛道推动技术进步与公平评测。**

- **链接: [http://arxiv.org/pdf/2508.13564v1](http://arxiv.org/pdf/2508.13564v1)**

> **作者:** Zheng Tang; Shuo Wang; David C. Anastasiu; Ming-Ching Chang; Anuj Sharma; Quan Kong; Norimasa Kobori; Munkhjargal Gochoo; Ganzorig Batnasan; Munkh-Erdene Otgonbold; Fady Alnajjar; Jun-Wei Hsieh; Tomasz Kornuta; Xiaolong Li; Yilin Zhao; Han Zhang; Subhashree Radhakrishnan; Arihant Jain; Ratnesh Kumar; Vidya N. Murali; Yuxing Wang; Sameer Satish Pusegaonkar; Yizhou Wang; Sujit Biswas; Xunlei Wu; Zhedong Zheng; Pranamesh Chakraborty; Rama Chellappa
>
> **备注:** Summary of the 9th AI City Challenge Workshop in conjunction with ICCV 2025
>
> **摘要:** The ninth AI City Challenge continues to advance real-world applications of computer vision and AI in transportation, industrial automation, and public safety. The 2025 edition featured four tracks and saw a 17% increase in participation, with 245 teams from 15 countries registered on the evaluation server. Public release of challenge datasets led to over 30,000 downloads to date. Track 1 focused on multi-class 3D multi-camera tracking, involving people, humanoids, autonomous mobile robots, and forklifts, using detailed calibration and 3D bounding box annotations. Track 2 tackled video question answering in traffic safety, with multi-camera incident understanding enriched by 3D gaze labels. Track 3 addressed fine-grained spatial reasoning in dynamic warehouse environments, requiring AI systems to interpret RGB-D inputs and answer spatial questions that combine perception, geometry, and language. Both Track 1 and Track 3 datasets were generated in NVIDIA Omniverse. Track 4 emphasized efficient road object detection from fisheye cameras, supporting lightweight, real-time deployment on edge devices. The evaluation framework enforced submission limits and used a partially held-out test set to ensure fair benchmarking. Final rankings were revealed after the competition concluded, fostering reproducibility and mitigating overfitting. Several teams achieved top-tier results, setting new benchmarks in multiple tasks.
>
---
#### [new 003] Learnable SMPLify: A Neural Solution for Optimization-Free Human Pose Inverse Kinematics
- **分类: cs.CV**

- **简介: 该论文针对3D人体姿态估计中的逆运动学问题，提出Learnable SMPLify框架，用单次回归神经网络替代SMPLify的迭代优化过程，显著提升速度并保持精度，支持序列推理与插件式后处理。**

- **链接: [http://arxiv.org/pdf/2508.13562v1](http://arxiv.org/pdf/2508.13562v1)**

> **作者:** Yuchen Yang; Linfeng Dong; Wei Wang; Zhihang Zhong; Xiao Sun
>
> **摘要:** In 3D human pose and shape estimation, SMPLify remains a robust baseline that solves inverse kinematics (IK) through iterative optimization. However, its high computational cost limits its practicality. Recent advances across domains have shown that replacing iterative optimization with data-driven neural networks can achieve significant runtime improvements without sacrificing accuracy. Motivated by this trend, we propose Learnable SMPLify, a neural framework that replaces the iterative fitting process in SMPLify with a single-pass regression model. The design of our framework targets two core challenges in neural IK: data construction and generalization. To enable effective training, we propose a temporal sampling strategy that constructs initialization-target pairs from sequential frames. To improve generalization across diverse motions and unseen poses, we propose a human-centric normalization scheme and residual learning to narrow the solution space. Learnable SMPLify supports both sequential inference and plug-in post-processing to refine existing image-based estimators. Extensive experiments demonstrate that our method establishes itself as a practical and simple baseline: it achieves nearly 200x faster runtime compared to SMPLify, generalizes well to unseen 3DPW and RICH, and operates in a model-agnostic manner when used as a plug-in tool on LucidAction. The code is available at https://github.com/Charrrrrlie/Learnable-SMPLify.
>
---
#### [new 004] Uncertainty-Aware Learning Policy for Reliable Pulmonary Nodule Detection on Chest X-Ray
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对肺结节检测任务，解决医疗AI因知识不足导致诊断不确定性的问题。通过引入医生背景知识，提出不确定性感知学习策略，在胸部X光图像上提升敏感性并降低不确定性。**

- **链接: [http://arxiv.org/pdf/2508.13236v1](http://arxiv.org/pdf/2508.13236v1)**

> **作者:** Hyeonjin Choi; Jinse Kim; Dong-yeon Yoo; Ju-sung Sun; Jung-won Lee
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Early detection and rapid intervention of lung cancer are crucial. Nonetheless, ensuring an accurate diagnosis is challenging, as physicians' ability to interpret chest X-rays varies significantly depending on their experience and degree of fatigue. Although medical AI has been rapidly advancing to assist in diagnosis, physicians' trust in such systems remains limited, preventing widespread clinical adoption. This skepticism fundamentally stems from concerns about its diagnostic uncertainty. In clinical diagnosis, physicians utilize extensive background knowledge and clinical experience. In contrast, medical AI primarily relies on repetitive learning of the target lesion to generate diagnoses based solely on that data. In other words, medical AI does not possess sufficient knowledge to render a diagnosis, leading to diagnostic uncertainty. Thus, this study suggests an Uncertainty-Aware Learning Policy that can address the issue of knowledge deficiency by learning the physicians' background knowledge alongside the Chest X-ray lesion information. We used 2,517 lesion-free images and 656 nodule images, all obtained from Ajou University Hospital. The proposed model attained 92% (IoU 0.2 / FPPI 2) with a 10% enhancement in sensitivity compared to the baseline model while also decreasing entropy as a measure of uncertainty by 0.2.
>
---
#### [new 005] EAvatar: Expression-Aware Head Avatar Reconstruction with Generative Geometry Priors
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.13537v1](http://arxiv.org/pdf/2508.13537v1)**

> **作者:** Shikun Zhang; Cunjian Chen; Yiqun Wang; Qiuhong Ke; Yong Li
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** High-fidelity head avatar reconstruction plays a crucial role in AR/VR, gaming, and multimedia content creation. Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated effectiveness in modeling complex geometry with real-time rendering capability and are now widely used in high-fidelity head avatar reconstruction tasks. However, existing 3DGS-based methods still face significant challenges in capturing fine-grained facial expressions and preserving local texture continuity, especially in highly deformable regions. To mitigate these limitations, we propose a novel 3DGS-based framework termed EAvatar for head reconstruction that is both expression-aware and deformation-aware. Our method introduces a sparse expression control mechanism, where a small number of key Gaussians are used to influence the deformation of their neighboring Gaussians, enabling accurate modeling of local deformations and fine-scale texture transitions. Furthermore, we leverage high-quality 3D priors from pretrained generative models to provide a more reliable facial geometry, offering structural guidance that improves convergence stability and shape accuracy during training. Experimental results demonstrate that our method produces more accurate and visually coherent head reconstructions with improved expression controllability and detail fidelity.
>
---
#### [new 006] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving
- **分类: cs.CV**

- **简介: 论文提出Prune2Drive框架，用于加速自动驾驶中多视角视觉语言模型的推理。针对高分辨率图像导致的计算开销问题，通过多样性感知的token剪枝和视图自适应控制器，在不重训练模型的前提下显著提升速度并降低内存占用，同时保持任务性能。**

- **链接: [http://arxiv.org/pdf/2508.13305v1](http://arxiv.org/pdf/2508.13305v1)**

> **作者:** Minhao Xiong; Zichen Wen; Zhuangcheng Gu; Xuyang Liu; Rui Zhang; Hengrui Kang; Jiabing Yang; Junyuan Zhang; Weijia Li; Conghui He; Yafei Wang; Linfeng Zhang
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Vision-Language Models (VLMs) have emerged as a promising paradigm in autonomous driving (AD), offering a unified framework for perception, reasoning, and decision-making by jointly modeling visual inputs and natural language instructions. However, their deployment is hindered by the significant computational overhead incurred when processing high-resolution, multi-view images, a standard setup in AD systems with six or more synchronized cameras. This overhead stems from the large number of visual tokens generated during encoding, increasing inference latency and memory consumption due to the quadratic complexity of self-attention. To address these challenges, we propose Prune2Drive, a plug-and-play visual token pruning framework for multi-view VLMs in autonomous driving. Prune2Drive introduces two core innovations: (i) a diversity-aware token selection mechanism inspired by farthest point sampling, which prioritizes semantic and spatial coverage across views rather than relying solely on attention scores, and (ii) a view-adaptive pruning controller that learns optimal pruning ratios for each camera view based on their importance to downstream driving tasks. Unlike prior methods, Prune2Drive does not require model retraining or access to attention maps, making it compatible with modern efficient attention implementations. Extensive experiments on two large-scale multi-view driving benchmarks, DriveLM and DriveLMM-o1, show that Prune2Drive achieves significant speedups and memory savings while maintaining or improving task performance. When retaining only 10% of the visual tokens, our method achieves a 6.40$\times$ speedup in the prefilling phase and consumes 13.4% of the original FLOPs, with only a 3% performance drop on the DriveLM benchmark.
>
---
#### [new 007] Mitigating Easy Option Bias in Multiple-Choice Question Answering
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文研究视觉问答任务中因选项偏置导致的模型误判问题。作者发现模型仅凭视觉和选项即可正确作答，提出GroundAttack工具生成难负样本以消除此偏差，并验证新标注数据能更真实评估模型能力。**

- **链接: [http://arxiv.org/pdf/2508.13428v1](http://arxiv.org/pdf/2508.13428v1)**

> **作者:** Hao Zhang; Chen Li; Basura Fernando
>
> **备注:** Under review
>
> **摘要:** In this early study, we observe an Easy-Options Bias (EOB) issue in some multiple-choice Visual Question Answering (VQA) benchmarks such as MMStar, RealWorldQA, SEED-Bench, Next-QA, STAR benchmark and Video-MME. This bias allows vision-language models (VLMs) to select the correct answer using only the vision (V) and options (O) as inputs, without the need for the question (Q). Through grounding experiments, we attribute the bias to an imbalance in visual relevance: the correct answer typically aligns more closely with the visual contents than the negative options in feature space, creating a shortcut for VLMs to infer the answer via simply vision-option similarity matching. To fix this, we introduce GroundAttack, a toolkit that automatically generates hard negative options as visually plausible as the correct answer. We apply it to the NExT-QA and MMStar datasets, creating new EOB-free annotations. On these EOB-free annotations, current VLMs approach to random accuracies under (V+O) settings, and drop to non-saturated accuracies under (V+Q+O) settings, providing a more realistic evaluation of VLMs' QA ability. Codes and new annotations will be released soon.
>
---
#### [new 008] SCRNet: Spatial-Channel Regulation Network for Medical Ultrasound Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学超声图像分割任务，解决传统CNN忽略长程依赖、Transformer忽视局部信息的问题。提出FAM模块融合卷积与交叉注意力机制，结合SCRM增强特征选择能力，并构建SCRNet框架，在分割性能上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2508.13899v1](http://arxiv.org/pdf/2508.13899v1)**

> **作者:** Weixin Xu; Ziliang Wang
>
> **备注:** 8 pagegs
>
> **摘要:** Medical ultrasound image segmentation presents a formidable challenge in the realm of computer vision. Traditional approaches rely on Convolutional Neural Networks (CNNs) and Transformer-based methods to address the intricacies of medical image segmentation. Nevertheless, inherent limitations persist, as CNN-based methods tend to disregard long-range dependencies, while Transformer-based methods may overlook local contextual information. To address these deficiencies, we propose a novel Feature Aggregation Module (FAM) designed to process two input features from the preceding layer. These features are seamlessly directed into two branches of the Convolution and Cross-Attention Parallel Module (CCAPM) to endow them with different roles in each of the two branches to help establish a strong connection between the two input features. This strategy enables our module to focus concurrently on both long-range dependencies and local contextual information by judiciously merging convolution operations with cross-attention mechanisms. Moreover, by integrating FAM within our proposed Spatial-Channel Regulation Module (SCRM), the ability to discern salient regions and informative features warranting increased attention is enhanced. Furthermore, by incorporating the SCRM into the encoder block of the UNet architecture, we introduce a novel framework dubbed Spatial-Channel Regulation Network (SCRNet). The results of our extensive experiments demonstrate the superiority of SCRNet, which consistently achieves state-of-the-art (SOTA) performance compared to existing methods.
>
---
#### [new 009] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment
- **分类: cs.CV**

- **简介: 论文提出Noisy Alignment（NA）方法，用于攻击自监督对比学习中的数据 poisoning 问题。通过抑制中毒图像中的噪声成分，实现更有效的后门攻击，同时保持干净数据准确性，并对抗常见防御机制。**

- **链接: [http://arxiv.org/pdf/2508.14015v1](http://arxiv.org/pdf/2508.14015v1)**

> **作者:** Tuo Chen; Jie Gui; Minjing Dong; Ju Jia; Lanting Fang; Jian Liu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Self-supervised contrastive learning (CL) effectively learns transferable representations from unlabeled data containing images or image-text pairs but suffers vulnerability to data poisoning backdoor attacks (DPCLs). An adversary can inject poisoned images into pretraining datasets, causing compromised CL encoders to exhibit targeted misbehavior in downstream tasks. Existing DPCLs, however, achieve limited efficacy due to their dependence on fragile implicit co-occurrence between backdoor and target object and inadequate suppression of discriminative features in backdoored images. We propose Noisy Alignment (NA), a DPCL method that explicitly suppresses noise components in poisoned images. Inspired by powerful training-controllable CL attacks, we identify and extract the critical objective of noisy alignment, adapting it effectively into data-poisoning scenarios. Our method implements noisy alignment by strategically manipulating contrastive learning's random cropping mechanism, formulating this process as an image layout optimization problem with theoretically derived optimal parameters. The resulting method is simple yet effective, achieving state-of-the-art performance compared to existing DPCLs, while maintaining clean-data accuracy. Furthermore, Noisy Alignment demonstrates robustness against common backdoor defenses. Codes can be found at https://github.com/jsrdcht/Noisy-Alignment.
>
---
#### [new 010] VisionLaw: Inferring Interpretable Intrinsic Dynamics from Visual Observations via Bilevel Optimization
- **分类: cs.CV**

- **简介: 论文提出VisionLaw框架，通过双层优化从视觉观测中推断可解释的物体内在动力学。解决现有方法依赖人工先验或缺乏可解释性的问题，利用LLM生成并演化本构定律，结合视觉仿真评估一致性，实现高效且泛化能力强的动力学建模。**

- **链接: [http://arxiv.org/pdf/2508.13792v1](http://arxiv.org/pdf/2508.13792v1)**

> **作者:** Jailing Lin; Shu Jiang; Qingyuan Zeng; Zhenzhong Wang; Min Jiang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** The intrinsic dynamics of an object governs its physical behavior in the real world, playing a critical role in enabling physically plausible interactive simulation with 3D assets. Existing methods have attempted to infer the intrinsic dynamics of objects from visual observations, but generally face two major challenges: one line of work relies on manually defined constitutive priors, making it difficult to generalize to complex scenarios; the other models intrinsic dynamics using neural networks, resulting in limited interpretability and poor generalization. To address these challenges, we propose VisionLaw, a bilevel optimization framework that infers interpretable expressions of intrinsic dynamics from visual observations. At the upper level, we introduce an LLMs-driven decoupled constitutive evolution strategy, where LLMs are prompted as a knowledgeable physics expert to generate and revise constitutive laws, with a built-in decoupling mechanism that substantially reduces the search complexity of LLMs. At the lower level, we introduce a vision-guided constitutive evaluation mechanism, which utilizes visual simulation to evaluate the consistency between the generated constitutive law and the underlying intrinsic dynamics, thereby guiding the upper-level evolution. Experiments on both synthetic and real-world datasets demonstrate that VisionLaw can effectively infer interpretable intrinsic dynamics from visual observations. It significantly outperforms existing state-of-the-art methods and exhibits strong generalization for interactive simulation in novel scenarios.
>
---
#### [new 011] STER-VLM: Spatio-Temporal With Enhanced Reference Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 论文提出STER-VLM框架，用于高效精准的交通场景分析。针对现有视觉语言模型计算资源消耗大、时空理解粗粒度的问题，通过分步处理空间时间信息、优选关键帧、参考驱动理解及优化提示词等技术提升性能，在多个数据集和AI City Challenge中验证有效。**

- **链接: [http://arxiv.org/pdf/2508.13470v1](http://arxiv.org/pdf/2508.13470v1)**

> **作者:** Tinh-Anh Nguyen-Nhu; Triet Dao Hoang Minh; Dat To-Thanh; Phuc Le-Gia; Tuan Vo-Lan; Tien-Huy Nguyen
>
> **备注:** ICCV Workshop 2025
>
> **摘要:** Vision-language models (VLMs) have emerged as powerful tools for enabling automated traffic analysis; however, current approaches often demand substantial computational resources and struggle with fine-grained spatio-temporal understanding. This paper introduces STER-VLM, a computationally efficient framework that enhances VLM performance through (1) caption decomposition to tackle spatial and temporal information separately, (2) temporal frame selection with best-view filtering for sufficient temporal information, and (3) reference-driven understanding for capturing fine-grained motion and dynamic context and (4) curated visual/textual prompt techniques. Experimental results on the WTS \cite{kong2024wts} and BDD \cite{BDD} datasets demonstrate substantial gains in semantic richness and traffic scene interpretation. Our framework is validated through a decent test score of 55.655 in the AI City Challenge 2025 Track 2, showing its effectiveness in advancing resource-efficient and accurate traffic analysis for real-world applications.
>
---
#### [new 012] RCGNet: RGB-based Category-Level 6D Object Pose Estimation with Geometric Guidance
- **分类: cs.CV**

- **简介: 论文提出RCGNet，解决仅用RGB图像进行类别级6D物体位姿估计的问题。通过Transformer预测并融合几何特征，结合几何引导算法提升几何信息表达，并用RANSAC-PnP算法处理尺度变化，实现高效高精度位姿估计。**

- **链接: [http://arxiv.org/pdf/2508.13623v1](http://arxiv.org/pdf/2508.13623v1)**

> **作者:** Sheng Yu; Di-Hua Zhai; Yuanqing Xia
>
> **备注:** Accepted by IROS2025
>
> **摘要:** While most current RGB-D-based category-level object pose estimation methods achieve strong performance, they face significant challenges in scenes lacking depth information. In this paper, we propose a novel category-level object pose estimation approach that relies solely on RGB images. This method enables accurate pose estimation in real-world scenarios without the need for depth data. Specifically, we design a transformer-based neural network for category-level object pose estimation, where the transformer is employed to predict and fuse the geometric features of the target object. To ensure that these predicted geometric features faithfully capture the object's geometry, we introduce a geometric feature-guided algorithm, which enhances the network's ability to effectively represent the object's geometric information. Finally, we utilize the RANSAC-PnP algorithm to compute the object's pose, addressing the challenges associated with variable object scales in pose estimation. Experimental results on benchmark datasets demonstrate that our approach is not only highly efficient but also achieves superior accuracy compared to previous RGB-based methods. These promising results offer a new perspective for advancing category-level object pose estimation using RGB images.
>
---
#### [new 013] A Lightweight Dual-Mode Optimization for Generative Face Video Coding
- **分类: cs.CV; eess.IV**

- **简介: 论文提出轻量级双模式优化方法，用于生成式人脸视频编码（GFVC），解决模型参数大、计算成本高的问题。通过结构重设计和两阶段通道剪枝策略，在大幅降低复杂度的同时保持高质量重建，优于VVC标准。**

- **链接: [http://arxiv.org/pdf/2508.13547v1](http://arxiv.org/pdf/2508.13547v1)**

> **作者:** Zihan Zhang; Shanzhi Yin; Bolin Chen; Ru-Ling Liao; Shiqi Wang; Yan Ye
>
> **摘要:** Generative Face Video Coding (GFVC) achieves superior rate-distortion performance by leveraging the strong inference capabilities of deep generative models. However, its practical deployment is hindered by large model parameters and high computational costs. To address this, we propose a lightweight GFVC framework that introduces dual-mode optimization - combining architectural redesign and operational refinement - to reduce complexity whilst preserving reconstruction quality. Architecturally, we replace traditional 3 x 3 convolutions with slimmer and more efficient layers, reducing complexity without compromising feature expressiveness. Operationally, we develop a two-stage adaptive channel pruning strategy: (1) soft pruning during training identifies redundant channels via learnable thresholds, and (2) hard pruning permanently eliminates these channels post-training using a derived mask. This dual-phase approach ensures both training stability and inference efficiency. Experimental results demonstrate that the proposed lightweight dual-mode optimization for GFVC can achieve 90.4% parameter reduction and 88.9% computation saving compared to the baseline, whilst achieving superior performance compared to state-of-the-art video coding standard Versatile Video Coding (VVC) in terms of perceptual-level quality metrics. As such, the proposed method is expected to enable efficient GFVC deployment in resource-constrained environments such as mobile edge devices.
>
---
#### [new 014] AIM 2025 challenge on Inverse Tone Mapping Report: Methods and Results
- **分类: cs.CV; eess.IV**

- **简介: 该论文报告了AIM 2025挑战赛中逆色调映射（ITM）任务的成果，旨在从单张LDR图像重建HDR图像。通过67支队伍提交的319个结果，分析了提升感知保真度和数值一致性的方法，并确立了性能基准。**

- **链接: [http://arxiv.org/pdf/2508.13479v1](http://arxiv.org/pdf/2508.13479v1)**

> **作者:** Chao Wang; Francesco Banterle; Bin Ren; Radu Timofte; Xin Lu; Yufeng Peng; Chengjie Ge; Zhijing Sun; Ziang Zhou; Zihao Li; Zishun Liao; Qiyu Kang; Xueyang Fu; Zheng-Jun Zha; Zhijing Sun; Xingbo Wang; Kean Liu; Senyan Xu; Yang Qiu; Yifan Ding; Gabriel Eilertsen; Jonas Unger; Zihao Wang; Ke Wu; Jinshan Pan; Zhen Liu; Zhongyang Li; Shuaicheng Liu; S. M Nadim Uddin
>
> **摘要:** This paper presents a comprehensive review of the AIM 2025 Challenge on Inverse Tone Mapping (ITM). The challenge aimed to push forward the development of effective ITM algorithms for HDR image reconstruction from single LDR inputs, focusing on perceptual fidelity and numerical consistency. A total of \textbf{67} participants submitted \textbf{319} valid results, from which the best five teams were selected for detailed analysis. This report consolidates their methodologies and performance, with the lowest PU21-PSNR among the top entries reaching 29.22 dB. The analysis highlights innovative strategies for enhancing HDR reconstruction quality and establishes strong benchmarks to guide future research in inverse tone mapping.
>
---
#### [new 015] Color Spike Data Generation via Bio-inspired Neuron-like Encoding with an Artificial Photoreceptor Layer
- **分类: cs.CV**

- **简介: 论文提出一种生物启发的脉冲编码方法，通过人工光感受器层生成含颜色和亮度信息的视觉脉冲数据，提升脉冲神经网络的信息容量与性能，解决传统SNN因脉冲数据信息有限导致的性能不足问题。**

- **链接: [http://arxiv.org/pdf/2508.13558v1](http://arxiv.org/pdf/2508.13558v1)**

> **作者:** Hsieh Ching-Teng; Wang Yuan-Kai
>
> **备注:** 14 pages, 11 figures
>
> **摘要:** In recent years, neuromorphic computing and spiking neural networks (SNNs) have ad-vanced rapidly through integration with deep learning. However, the performance of SNNs still lags behind that of convolutional neural networks (CNNs), primarily due to the limited information capacity of spike-based data. Although some studies have attempted to improve SNN performance by training them with non-spiking inputs such as static images, this approach deviates from the original intent of neuromorphic computing, which emphasizes spike-based information processing. To address this issue, we propose a Neuron-like Encoding method that generates spike data based on the intrinsic operational principles and functions of biological neurons. This method is further enhanced by the incorporation of an artificial pho-toreceptor layer, enabling spike data to carry both color and luminance information, thereby forming a complete visual spike signal. Experimental results using the Integrate-and-Fire neuron model demonstrate that this biologically inspired approach effectively increases the information content of spike signals and improves SNN performance, all while adhering to neuromorphic principles. We believe this concept holds strong potential for future development and may contribute to overcoming current limitations in neuro-morphic computing, facilitating broader applications of SNNs.
>
---
#### [new 016] Online 3D Gaussian Splatting Modeling with Novel View Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14014v1](http://arxiv.org/pdf/2508.14014v1)**

> **作者:** Byeonggwon Lee; Junkyu Park; Khang Truong Giang; Soohwan Song
>
> **摘要:** This study addresses the challenge of generating online 3D Gaussian Splatting (3DGS) models from RGB-only frames. Previous studies have employed dense SLAM techniques to estimate 3D scenes from keyframes for 3DGS model construction. However, these methods are limited by their reliance solely on keyframes, which are insufficient to capture an entire scene, resulting in incomplete reconstructions. Moreover, building a generalizable model requires incorporating frames from diverse viewpoints to achieve broader scene coverage. However, online processing restricts the use of many frames or extensive training iterations. Therefore, we propose a novel method for high-quality 3DGS modeling that improves model completeness through adaptive view selection. By analyzing reconstruction quality online, our approach selects optimal non-keyframes for additional training. By integrating both keyframes and selected non-keyframes, the method refines incomplete regions from diverse viewpoints, significantly enhancing completeness. We also present a framework that incorporates an online multi-view stereo approach, ensuring consistency in 3D information throughout the 3DGS modeling process. Experimental results demonstrate that our method outperforms state-of-the-art methods, delivering exceptional performance in complex outdoor scenes.
>
---
#### [new 017] YOLO11-CR: a Lightweight Convolution-and-Attention Framework for Accurate Fatigue Driving Detection
- **分类: cs.CV; eess.IV**

- **简介: 论文提出YOLO11-CR模型，用于实时疲劳驾驶检测任务。针对视觉方法中小目标和遮挡检测不准的问题，引入卷积与注意力融合模块（CAFM）和矩形校准模块（RCM），提升特征表达和定位精度，在DSM数据集上表现优于基线模型。**

- **链接: [http://arxiv.org/pdf/2508.13205v1](http://arxiv.org/pdf/2508.13205v1)**

> **作者:** Zhebin Jin; Ligang Dong
>
> **摘要:** Driver fatigue detection is of paramount importance for intelligent transportation systems due to its critical role in mitigating road traffic accidents. While physiological and vehicle dynamics-based methods offer accuracy, they are often intrusive, hardware-dependent, and lack robustness in real-world environments. Vision-based techniques provide a non-intrusive and scalable alternative, but still face challenges such as poor detection of small or occluded objects and limited multi-scale feature modeling. To address these issues, this paper proposes YOLO11-CR, a lightweight and efficient object detection model tailored for real-time fatigue detection. YOLO11-CR introduces two key modules: the Convolution-and-Attention Fusion Module (CAFM), which integrates local CNN features with global Transformer-based context to enhance feature expressiveness; and the Rectangular Calibration Module (RCM), which captures horizontal and vertical contextual information to improve spatial localization, particularly for profile faces and small objects like mobile phones. Experiments on the DSM dataset demonstrated that YOLO11-CR achieves a precision of 87.17%, recall of 83.86%, mAP@50 of 88.09%, and mAP@50-95 of 55.93%, outperforming baseline models significantly. Ablation studies further validate the effectiveness of the CAFM and RCM modules in improving both sensitivity and localization accuracy. These results demonstrate that YOLO11-CR offers a practical and high-performing solution for in-vehicle fatigue monitoring, with strong potential for real-world deployment and future enhancements involving temporal modeling, multi-modal data integration, and embedded optimization.
>
---
#### [new 018] CLoE: Curriculum Learning on Endoscopic Images for Robust MES Classification
- **分类: cs.CV; cs.LG**

- **简介: 论文提出CLoE框架，用于内镜图像的MES分类任务，解决标签噪声和序数特性问题。通过图像质量代理标注置信度构建课程学习策略，并结合ResizeMix增强，提升模型鲁棒性与准确率。**

- **链接: [http://arxiv.org/pdf/2508.13280v1](http://arxiv.org/pdf/2508.13280v1)**

> **作者:** Zeynep Ozdemir; Hacer Yalim Keles; Omer Ozgur Tanriover
>
> **备注:** 16 pages, 4 figures, 9 tables
>
> **摘要:** Estimating disease severity from endoscopic images is essential in assessing ulcerative colitis, where the Mayo Endoscopic Subscore (MES) is widely used to grade inflammation. However, MES classification remains challenging due to label noise from inter-observer variability and the ordinal nature of the score, which standard models often ignore. We propose CLoE, a curriculum learning framework that accounts for both label reliability and ordinal structure. Image quality, estimated via a lightweight model trained on Boston Bowel Preparation Scale (BBPS) labels, is used as a proxy for annotation confidence to order samples from easy (clean) to hard (noisy). This curriculum is further combined with ResizeMix augmentation to improve robustness. Experiments on the LIMUC and HyperKvasir datasets, using both CNNs and Transformers, show that CLoE consistently improves performance over strong supervised and self-supervised baselines. For instance, ConvNeXt-Tiny reaches 82.5\% accuracy and a QWK of 0.894 on LIMUC with low computational cost. These results highlight the potential of difficulty-aware training strategies for improving ordinal classification under label uncertainty. Code will be released at https://github.com/zeynepozdemir/CLoE.
>
---
#### [new 019] Calibrating Biased Distribution in VFM-derived Latent Space via Cross-Domain Geometric Consistency
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.13518v1](http://arxiv.org/pdf/2508.13518v1)**

> **作者:** Yanbiao Ma; Wei Dai; Bowei Liu; Jiayi Chen; Wenke Huang; Guancheng Wan; Zhiwu Lu; Junchi Yan
>
> **备注:** 15 pages, CVPR Oral
>
> **摘要:** Despite the fast progress of deep learning, one standing challenge is the gap of the observed training samples and the underlying true distribution. There are multiple reasons for the causing of this gap e.g. sampling bias, noise etc. In the era of foundation models, we show that when leveraging the off-the-shelf (vision) foundation models (e.g., CLIP, DINOv2) for feature extraction, the geometric shapes of the resulting feature distributions exhibit remarkable transferability across domains and datasets. To verify its practical usefulness, we embody our geometric knowledge-guided distribution calibration framework in two popular and challenging settings: federated learning and long-tailed recognition. In the federated setting, we devise a technique of acquiring the global geometric shape under privacy constraints, then leverage this knowledge to generate new samples for clients, in the aim of bridging the gap between local and global observations. In long-tailed learning, it utilizes the geometric knowledge transferred from sample-rich categories to recover the true distribution for sample-scarce tail classes. Comprehensive experiments show that our proposed geometric knowledge-guided distribution calibration effectively overcomes information deficits caused by data heterogeneity and sample imbalance, with boosted performance across benchmarks.
>
---
#### [new 020] Bridging Clear and Adverse Driving Conditions
- **分类: cs.CV**

- **简介: 论文针对自动驾驶在恶劣天气下性能下降的问题，提出一种混合域适应方法，通过生成逼真恶劣条件图像来提升模型鲁棒性。工作包括设计多种数据生成管道、改进GAN与扩散模型结合方案，并在ACDC数据集上验证，显著提升语义分割精度。**

- **链接: [http://arxiv.org/pdf/2508.13592v1](http://arxiv.org/pdf/2508.13592v1)**

> **作者:** Yoel Shapiro; Yahia Showgan; Koustav Mullick
>
> **摘要:** Autonomous Driving (AD) systems exhibit markedly degraded performance under adverse environmental conditions, such as low illumination and precipitation. The underrepresentation of adverse conditions in AD datasets makes it challenging to address this deficiency. To circumvent the prohibitive cost of acquiring and annotating adverse weather data, we propose a novel Domain Adaptation (DA) pipeline that transforms clear-weather images into fog, rain, snow, and nighttime images. Here, we systematically develop and evaluate several novel data-generation pipelines, including simulation-only, GAN-based, and hybrid diffusion-GAN approaches, to synthesize photorealistic adverse images from labelled clear images. We leverage an existing DA GAN, extend it to support auxiliary inputs, and develop a novel training recipe that leverages both simulated and real images. The simulated images facilitate exact supervision by providing perfectly matched image pairs, while the real images help bridge the simulation-to-real (sim2real) gap. We further introduce a method to mitigate hallucinations and artifacts in Stable-Diffusion Image-to-Image (img2img) outputs by blending them adaptively with their progenitor images. We finetune downstream models on our synthetic data and evaluate them on the Adverse Conditions Dataset with Correspondences (ACDC). We achieve 1.85 percent overall improvement in semantic segmentation, and 4.62 percent on nighttime, demonstrating the efficacy of our hybrid method for robust AD perception under challenging conditions.
>
---
#### [new 021] Two-Factor Authentication Smart Entryway Using Modified LBPH Algorithm
- **分类: cs.CV**

- **简介: 论文提出了一种基于改进LBPH算法的双因素智能门禁系统，解决疫情下人脸遮挡导致识别困难的问题。通过人脸识别与密码验证结合，实现自动门控、陌生人报警及远程Telegram控制，准确率约70%，用户接受度高。**

- **链接: [http://arxiv.org/pdf/2508.13617v1](http://arxiv.org/pdf/2508.13617v1)**

> **作者:** Zakiah Ayop; Wan Mohamad Hariz Bin Wan Mohamad Rosdi; Looi Wei Hua; Syarulnaziah Anawar; Nur Fadzilah Othman
>
> **摘要:** Face mask detection has become increasingly important recently, particularly during the COVID-19 pandemic. Many face detection models have been developed in smart entryways using IoT. However, there is a lack of IoT development on face mask detection. This paper proposes a two-factor authentication system for smart entryway access control using facial recognition and passcode verification and an automation process to alert the owner and activate the surveillance system when a stranger is detected and controls the system remotely via Telegram on a Raspberry Pi platform. The system employs the Local Binary Patterns Histograms for the full face recognition algorithm and modified LBPH algorithm for occluded face detection. On average, the system achieved an Accuracy of approximately 70%, a Precision of approximately 80%, and a Recall of approximately 83.26% across all tested users. The results indicate that the system is capable of conducting face recognition and mask detection, automating the operation of the remote control to register users, locking or unlocking the door, and notifying the owner. The sample participants highly accept it for future use in the user acceptance test.
>
---
#### [new 022] 2D Gaussians Meet Visual Tokenizer
- **分类: cs.CV**

- **简介: 该论文提出Visual Gaussian Quantization（VGQ），用于图像生成中的视觉分词任务，解决传统量化方法忽略几何结构的问题。通过将2D高斯分布引入代码本量化框架，显式建模位置、旋转和尺度等结构信息，提升重建质量与结构保真度。**

- **链接: [http://arxiv.org/pdf/2508.13515v1](http://arxiv.org/pdf/2508.13515v1)**

> **作者:** Yiang Shi; Xiaoyang Guo; Wei Yin; Mingkai Jia; Qian Zhang; Xiaolin Hu; Wenyu Liu; Xinggang Wan
>
> **摘要:** The image tokenizer is a critical component in AR image generation, as it determines how rich and structured visual content is encoded into compact representations. Existing quantization-based tokenizers such as VQ-GAN primarily focus on appearance features like texture and color, often neglecting geometric structures due to their patch-based design. In this work, we explored how to incorporate more visual information into the tokenizer and proposed a new framework named Visual Gaussian Quantization (VGQ), a novel tokenizer paradigm that explicitly enhances structural modeling by integrating 2D Gaussians into traditional visual codebook quantization frameworks. Our approach addresses the inherent limitations of naive quantization methods such as VQ-GAN, which struggle to model structured visual information due to their patch-based design and emphasis on texture and color. In contrast, VGQ encodes image latents as 2D Gaussian distributions, effectively capturing geometric and spatial structures by directly modeling structure-related parameters such as position, rotation and scale. We further demonstrate that increasing the density of 2D Gaussians within the tokens leads to significant gains in reconstruction fidelity, providing a flexible trade-off between token efficiency and visual richness. On the ImageNet 256x256 benchmark, VGQ achieves strong reconstruction quality with an rFID score of 1.00. Furthermore, by increasing the density of 2D Gaussians within the tokens, VGQ gains a significant boost in reconstruction capability and achieves a state-of-the-art reconstruction rFID score of 0.556 and a PSNR of 24.93, substantially outperforming existing methods. Codes will be released soon.
>
---
#### [new 023] DAASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出DAASH框架，用于生成高效且隐蔽的对抗样本。针对现有方法在感知一致性上的不足，通过多阶段组合Lp约束攻击方法，优化误分类与感知失真，显著提升攻击成功率和视觉质量，并具有良好泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.13309v1](http://arxiv.org/pdf/2508.13309v1)**

> **作者:** Abdullah Al Nomaan Nafi; Habibur Rahaman; Zafaryab Haider; Tanzim Mahfuz; Fnu Suya; Swarup Bhunia; Prabuddha Chakraborty
>
> **摘要:** Numerous techniques have been proposed for generating adversarial examples in white-box settings under strict Lp-norm constraints. However, such norm-bounded examples often fail to align well with human perception, and only recently have a few methods begun specifically exploring perceptually aligned adversarial examples. Moreover, it remains unclear whether insights from Lp-constrained attacks can be effectively leveraged to improve perceptual efficacy. In this paper, we introduce DAASH, a fully differentiable meta-attack framework that generates effective and perceptually aligned adversarial examples by strategically composing existing Lp-based attack methods. DAASH operates in a multi-stage fashion: at each stage, it aggregates candidate adversarial examples from multiple base attacks using learned, adaptive weights and propagates the result to the next stage. A novel meta-loss function guides this process by jointly minimizing misclassification loss and perceptual distortion, enabling the framework to dynamically modulate the contribution of each base attack throughout the stages. We evaluate DAASH on adversarially trained models across CIFAR-10, CIFAR-100, and ImageNet. Despite relying solely on Lp-constrained based methods, DAASH significantly outperforms state-of-the-art perceptual attacks such as AdvAD -- achieving higher attack success rates (e.g., 20.63\% improvement) and superior visual quality, as measured by SSIM, LPIPS, and FID (improvements $\approx$ of 11, 0.015, and 5.7, respectively). Furthermore, DAASH generalizes well to unseen defenses, making it a practical and strong baseline for evaluating robustness without requiring handcrafted adaptive attacks for each new defense.
>
---
#### [new 024] AIM 2025 Rip Current Segmentation (RipSeg) Challenge Report
- **分类: cs.CV; cs.AI; I.4.0; I.4.9**

- **简介: 该论文介绍AIM 2025 RipSeg挑战赛，聚焦于 rip current 的单类实例分割任务，旨在提升海滩安全。工作包括构建多样化数据集、设计评估指标与竞赛框架，并分析参赛方法性能，推动自动检测技术发展。**

- **链接: [http://arxiv.org/pdf/2508.13401v1](http://arxiv.org/pdf/2508.13401v1)**

> **作者:** Andrei Dumitriu; Florin Miron; Florin Tatui; Radu Tudor Ionescu; Radu Timofte; Aakash Ralhan; Florin-Alexandru Vasluianu; Shenyang Qian; Mitchell Harley; Imran Razzak; Yang Song; Pu Luo; Yumei Li; Cong Xu; Jinming Chai; Kexin Zhang; Licheng Jiao; Lingling Li; Siqi Yu; Chao Zhang; Kehuan Song; Fang Liu; Puhua Chen; Xu Liu; Jin Hu; Jinyang Xu; Biao Liu
>
> **备注:** Challenge report paper from AIM2025 Workshop at ICCVW 2025
>
> **摘要:** This report presents an overview of the AIM 2025 RipSeg Challenge, a competition designed to advance techniques for automatic rip current segmentation in still images. Rip currents are dangerous, fast-moving flows that pose a major risk to beach safety worldwide, making accurate visual detection an important and underexplored research task. The challenge builds on RipVIS, the largest available rip current dataset, and focuses on single-class instance segmentation, where precise delineation is critical to fully capture the extent of rip currents. The dataset spans diverse locations, rip current types, and camera orientations, providing a realistic and challenging benchmark. In total, $75$ participants registered for this first edition, resulting in $5$ valid test submissions. Teams were evaluated on a composite score combining $F_1$, $F_2$, $AP_{50}$, and $AP_{[50:95]}$, ensuring robust and application-relevant rankings. The top-performing methods leveraged deep learning architectures, domain adaptation techniques, pretrained models, and domain generalization strategies to improve performance under diverse conditions. This report outlines the dataset details, competition framework, evaluation metrics, and final results, providing insights into the current state of rip current segmentation. We conclude with a discussion of key challenges, lessons learned from the submissions, and future directions for expanding RipSeg.
>
---
#### [new 025] In-hoc Concept Representations to Regularise Deep Learning in Medical Imaging
- **分类: cs.CV**

- **简介: 论文提出LCRReg，一种基于潜在概念表示的正则化方法，用于提升医学图像深度学习模型在分布偏移下的泛化能力，解决模型依赖伪相关而非临床特征的问题。通过小规模辅助数据合成概念样本，引导CNN激活特定语义子空间，无需密集概念标注。**

- **链接: [http://arxiv.org/pdf/2508.13880v1](http://arxiv.org/pdf/2508.13880v1)**

> **作者:** Valentina Corbetta; Floris Six Dijkstra; Regina Beets-Tan; Hoel Kervadec; Kristoffer Wickstrøm; Wilson Silva
>
> **备注:** 13 pages, 13 figures, 2 tables, accepted at PHAROS-AFE-AIMI Workshop in conjunction with the International Conference on Computer Vision (ICCV), 2025. This is the submitted manuscript with added link to the github repo, funding acknowledgments and author names and affiliations, and a correction to numbers in Table 1. Final version not published yet
>
> **摘要:** Deep learning models in medical imaging often achieve strong in-distribution performance but struggle to generalise under distribution shifts, frequently relying on spurious correlations instead of clinically meaningful features. We introduce LCRReg, a novel regularisation approach that leverages Latent Concept Representations (LCRs) (e.g., Concept Activation Vectors (CAVs)) to guide models toward semantically grounded representations. LCRReg requires no concept labels in the main training set and instead uses a small auxiliary dataset to synthesise high-quality, disentangled concept examples. We extract LCRs for predefined relevant features, and incorporate a regularisation term that guides a Convolutional Neural Network (CNN) to activate within latent subspaces associated with those concepts. We evaluate LCRReg across synthetic and real-world medical tasks. On a controlled toy dataset, it significantly improves robustness to injected spurious correlations and remains effective even in multi-concept and multiclass settings. On the diabetic retinopathy binary classification task, LCRReg enhances performance under both synthetic spurious perturbations and out-of-distribution (OOD) generalisation. Compared to baselines, including multitask learning, linear probing, and post-hoc concept-based models, LCRReg offers a lightweight, architecture-agnostic strategy for improving model robustness without requiring dense concept supervision. Code is available at the following link: https://github.com/Trustworthy-AI-UU-NKI/lcr\_regularization
>
---
#### [new 026] PersonaVlog: Personalized Multimodal Vlog Generation with Multi-Agent Collaboration and Iterative Self-Correction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13602v1](http://arxiv.org/pdf/2508.13602v1)**

> **作者:** Xiaolu Hou; Bing Ma; Jiaxiang Cheng; Xuhua Ren; Kai Yu; Wenyue Li; Tianxiang Zheng; Qinglin Lu
>
> **备注:** Project Page: https://personavlog-paper.github.io/
>
> **摘要:** With the growing demand for short videos and personalized content, automated Video Log (Vlog) generation has become a key direction in multimodal content creation. Existing methods mostly rely on predefined scripts, lacking dynamism and personal expression. Therefore, there is an urgent need for an automated Vlog generation approach that enables effective multimodal collaboration and high personalization. To this end, we propose PersonaVlog, an automated multimodal stylized Vlog generation framework that can produce personalized Vlogs featuring videos, background music, and inner monologue speech based on a given theme and reference image. Specifically, we propose a multi-agent collaboration framework based on Multimodal Large Language Models (MLLMs). This framework efficiently generates high-quality prompts for multimodal content creation based on user input, thereby improving the efficiency and creativity of the process. In addition, we incorporate a feedback and rollback mechanism that leverages MLLMs to evaluate and provide feedback on generated results, thereby enabling iterative self-correction of multimodal content. We also propose ThemeVlogEval, a theme-based automated benchmarking framework that provides standardized metrics and datasets for fair evaluation. Comprehensive experiments demonstrate the significant advantages and potential of our framework over several baselines, highlighting its effectiveness and great potential for generating automated Vlogs.
>
---
#### [new 027] Generative Model-Based Feature Attention Module for Video Action Analysis
- **分类: cs.CV**

- **简介: 论文提出一种基于生成模型的特征注意力模块，用于视频动作分析任务，解决现有方法忽视特征语义、精度不足的问题。通过学习帧与段级时序特征语义关系，提升动作识别与检测性能，在两个基准数据集上验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.13565v1](http://arxiv.org/pdf/2508.13565v1)**

> **作者:** Guiqin Wang; Peng Zhao; Cong Zhao; Jing Huang; Siyan Guo; Shusen Yang
>
> **摘要:** Video action analysis is a foundational technology within the realm of intelligent video comprehension, particularly concerning its application in Internet of Things(IoT). However, existing methodologies overlook feature semantics in feature extraction and focus on optimizing action proposals, thus these solutions are unsuitable for widespread adoption in high-performance IoT applications due to the limitations in precision, such as autonomous driving, which necessitate robust and scalable intelligent video analytics analysis. To address this issue, we propose a novel generative attention-based model to learn the relation of feature semantics. Specifically, by leveraging the differences of actions' foreground and background, our model simultaneously learns the frame- and segment-dependencies of temporal action feature semantics, which takes advantage of feature semantics in the feature extraction effectively. To evaluate the effectiveness of our model, we conduct extensive experiments on two benchmark video task, action recognition and action detection. In the context of action detection tasks, we substantiate the superiority of our approach through comprehensive validation on widely recognized datasets. Moreover, we extend the validation of the effectiveness of our proposed method to a broader task, video action recognition. Our code is available at https://github.com/Generative-Feature-Model/GAF.
>
---
#### [new 028] GaitCrafter: Diffusion Model for Biometric Preserving Gait Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.13300v1](http://arxiv.org/pdf/2508.13300v1)**

> **作者:** Sirshapan Mitra; Yogesh S. Rawat
>
> **摘要:** Gait recognition is a valuable biometric task that enables the identification of individuals from a distance based on their walking patterns. However, it remains limited by the lack of large-scale labeled datasets and the difficulty of collecting diverse gait samples for each individual while preserving privacy. To address these challenges, we propose GaitCrafter, a diffusion-based framework for synthesizing realistic gait sequences in the silhouette domain. Unlike prior works that rely on simulated environments or alternative generative models, GaitCrafter trains a video diffusion model from scratch, exclusively on gait silhouette data. Our approach enables the generation of temporally consistent and identity-preserving gait sequences. Moreover, the generation process is controllable-allowing conditioning on various covariates such as clothing, carried objects, and view angle. We show that incorporating synthetic samples generated by GaitCrafter into the gait recognition pipeline leads to improved performance, especially under challenging conditions. Additionally, we introduce a mechanism to generate novel identities-synthetic individuals not present in the original dataset-by interpolating identity embeddings. These novel identities exhibit unique, consistent gait patterns and are useful for training models while maintaining privacy of real subjects. Overall, our work takes an important step toward leveraging diffusion models for high-quality, controllable, and privacy-aware gait data generation.
>
---
#### [new 029] Distilled-3DGS:Distilled 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于新视角合成任务，针对3D高斯泼溅（3DGS）模型参数量大、存储成本高的问题，提出知识蒸馏框架，通过多教师模型指导轻量学生模型训练，并引入结构相似性损失提升几何一致性，实现高质量且高效的渲染。**

- **链接: [http://arxiv.org/pdf/2508.14037v1](http://arxiv.org/pdf/2508.14037v1)**

> **作者:** Lintao Xiang; Xinkai Chen; Jianhuang Lai; Guangcong Wang
>
> **备注:** Project page: https://distilled3dgs.github.io Code: https://github.com/lt-xiang/Distilled-3DGS
>
> **摘要:** 3D Gaussian Splatting (3DGS) has exhibited remarkable efficacy in novel view synthesis (NVS). However, it suffers from a significant drawback: achieving high-fidelity rendering typically necessitates a large number of 3D Gaussians, resulting in substantial memory consumption and storage requirements. To address this challenge, we propose the first knowledge distillation framework for 3DGS, featuring various teacher models, including vanilla 3DGS, noise-augmented variants, and dropout-regularized versions. The outputs of these teachers are aggregated to guide the optimization of a lightweight student model. To distill the hidden geometric structure, we propose a structural similarity loss to boost the consistency of spatial geometric distributions between the student and teacher model. Through comprehensive quantitative and qualitative evaluations across diverse datasets, the proposed Distilled-3DGS, a simple yet effective framework without bells and whistles, achieves promising rendering results in both rendering quality and storage efficiency compared to state-of-the-art methods. Project page: https://distilled3dgs.github.io . Code: https://github.com/lt-xiang/Distilled-3DGS .
>
---
#### [new 030] Applications of Small Language Models in Medical Imaging Classification with a Focus on Prompt Strategies
- **分类: cs.CV**

- **简介: 论文研究小语言模型在医学影像分类任务中的应用，解决资源受限医疗环境中大模型成本高、隐私差的问题。通过对比不同小模型和提示策略，在NIH胸部X光数据集上验证了优化提示可显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.13378v1](http://arxiv.org/pdf/2508.13378v1)**

> **作者:** Yiting Wang; Ziwei Wang; Jiachen Zhong; Di Zhu; Weiyi Li
>
> **备注:** Under Review
>
> **摘要:** Large language models (LLMs) have shown remarkable capabilities in natural language processing and multi-modal understanding. However, their high computational cost, limited accessibility, and data privacy concerns hinder their adoption in resource-constrained healthcare environments. This study investigates the performance of small language models (SLMs) in a medical imaging classification task, comparing different models and prompt designs to identify the optimal combination for accuracy and usability. Using the NIH Chest X-ray dataset, we evaluate multiple SLMs on the task of classifying chest X-ray positions (anteroposterior [AP] vs. posteroanterior [PA]) under three prompt strategies: baseline instruction, incremental summary prompts, and correction-based reflective prompts. Our results show that certain SLMs achieve competitive accuracy with well-crafted prompts, suggesting that prompt engineering can substantially enhance SLM performance in healthcare applications without requiring deep AI expertise from end users.
>
---
#### [new 031] Hierarchical Vision-Language Retrieval of Educational Metaverse Content in Agriculture
- **分类: cs.CV**

- **简介: 论文提出了一种层次化视觉语言模型，用于农业教育元宇宙内容的检索任务，解决用户难以搜索匹配兴趣场景的问题。构建了457个农业虚拟博物馆数据集，并实现基于自然语言查询的高效检索，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.13713v1](http://arxiv.org/pdf/2508.13713v1)**

> **作者:** Ali Abdari; Alex Falcon; Giuseppe Serra
>
> **备注:** Accepted for publication at the 23rd International Conference on Image Analysis and Processing (ICIAP 2025)
>
> **摘要:** Every day, a large amount of educational content is uploaded online across different areas, including agriculture and gardening. When these videos or materials are grouped meaningfully, they can make learning easier and more effective. One promising way to organize and enrich such content is through the Metaverse, which allows users to explore educational experiences in an interactive and immersive environment. However, searching for relevant Metaverse scenarios and finding those matching users' interests remains a challenging task. A first step in this direction has been done recently, but existing datasets are small and not sufficient for training advanced models. In this work, we make two main contributions: first, we introduce a new dataset containing 457 agricultural-themed virtual museums (AgriMuseums), each enriched with textual descriptions; and second, we propose a hierarchical vision-language model to represent and retrieve relevant AgriMuseums using natural language queries. In our experimental setting, the proposed method achieves up to about 62\% R@1 and 78\% MRR, confirming its effectiveness, and it also leads to improvements on existing benchmarks by up to 6\% R@1 and 11\% MRR. Moreover, an extensive evaluation validates our design choices. Code and dataset are available at https://github.com/aliabdari/Agricultural_Metaverse_Retrieval .
>
---
#### [new 032] Beyond Simple Edits: Composed Video Retrieval with Dense Modifications
- **分类: cs.CV**

- **简介: 论文提出Dense-WebVid-CoVR数据集和新模型，解决视频检索中细粒度组合修改的难题。通过跨注意力融合视觉与文本信息，实现精准对齐，显著提升检索性能。**

- **链接: [http://arxiv.org/pdf/2508.14039v1](http://arxiv.org/pdf/2508.14039v1)**

> **作者:** Omkar Thawakar; Dmitry Demidov; Ritesh Thawkar; Rao Muhammad Anwer; Mubarak Shah; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** Accepted to ICCV-2025
>
> **摘要:** Composed video retrieval is a challenging task that strives to retrieve a target video based on a query video and a textual description detailing specific modifications. Standard retrieval frameworks typically struggle to handle the complexity of fine-grained compositional queries and variations in temporal understanding limiting their retrieval ability in the fine-grained setting. To address this issue, we introduce a novel dataset that captures both fine-grained and composed actions across diverse video segments, enabling more detailed compositional changes in retrieved video content. The proposed dataset, named Dense-WebVid-CoVR, consists of 1.6 million samples with dense modification text that is around seven times more than its existing counterpart. We further develop a new model that integrates visual and textual information through Cross-Attention (CA) fusion using grounded text encoder, enabling precise alignment between dense query modifications and target videos. The proposed model achieves state-of-the-art results surpassing existing methods on all metrics. Notably, it achieves 71.3\% Recall@1 in visual+text setting and outperforms the state-of-the-art by 3.4\%, highlighting its efficacy in terms of leveraging detailed video descriptions and dense modification texts. Our proposed dataset, code, and model are available at :https://github.com/OmkarThawakar/BSE-CoVR
>
---
#### [new 033] Exploration of Deep Learning Based Recognition for Urdu Text
- **分类: cs.CV**

- **简介: 该论文属于 Urdu 文本识别任务，旨在解决因连写特性导致的字符分割困难问题。作者提出基于卷积神经网络的组件分类方法，通过生成字符组合数据并提取连写部件，构建两级神经网络模型，实现高精度组件识别。**

- **链接: [http://arxiv.org/pdf/2508.13245v1](http://arxiv.org/pdf/2508.13245v1)**

> **作者:** Sumaiya Fazal; Sheeraz Ahmed
>
> **摘要:** Urdu is a cursive script language and has similarities with Arabic and many other South Asian languages. Urdu is difficult to classify due to its complex geometrical and morphological structure. Character classification can be processed further if segmentation technique is efficient, but due to context sensitivity in Urdu, segmentation-based recognition often results with high error rate. Our proposed approach for Urdu optical character recognition system is a component-based classification relying on automatic feature learning technique called convolutional neural network. CNN is trained and tested on Urdu text dataset, which is generated through permutation process of three characters and further proceeds to discarding unnecessary images by applying connected component technique in order to obtain ligature only. Hierarchical neural network is implemented with two levels to deal with three degrees of character permutations and component classification Our model successfully achieved 0.99% for component classification.
>
---
#### [new 034] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis
- **分类: cs.CV**

- **简介: 论文提出PhysGM，用于从单张图像快速生成高保真4D物理模拟。解决现有方法依赖预重建3D表示和不稳定物理建模的问题，通过联合预测3D高斯表示与物理属性，并利用参考视频优化，实现高效、真实的4D合成。**

- **链接: [http://arxiv.org/pdf/2508.13911v1](http://arxiv.org/pdf/2508.13911v1)**

> **作者:** Chunji Lv; Zequn Chen; Donglin Di; Weinan Zhang; Hao Li; Wei Chen; Changsheng Li
>
> **摘要:** While physics-grounded 3D motion synthesis has seen significant progress, current methods face critical limitations. They typically rely on pre-reconstructed 3D Gaussian Splatting (3DGS) representations, while physics integration depends on either inflexible, manually defined physical attributes or unstable, optimization-heavy guidance from video models. To overcome these challenges, we introduce PhysGM, a feed-forward framework that jointly predicts a 3D Gaussian representation and its physical properties from a single image, enabling immediate, physical simulation and high-fidelity 4D rendering. We first establish a base model by jointly optimizing for Gaussian reconstruction and probabilistic physics prediction. The model is then refined with physically plausible reference videos to enhance both rendering fidelity and physics prediction accuracy. We adopt the Direct Preference Optimization (DPO) to align its simulations with reference videos, circumventing Score Distillation Sampling (SDS) optimization which needs back-propagating gradients through the complex differentiable simulation and rasterization. To facilitate the training, we introduce a new dataset PhysAssets of over 24,000 3D assets, annotated with physical properties and corresponding guiding videos. Experimental results demonstrate that our method effectively generates high-fidelity 4D simulations from a single image in one minute. This represents a significant speedup over prior works while delivering realistic rendering results. Our project page is at:https://hihixiaolv.github.io/PhysGM.github.io/
>
---
#### [new 035] Multi-view Clustering via Bi-level Decoupling and Consistency Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.13499v1](http://arxiv.org/pdf/2508.13499v1)**

> **作者:** Shihao Dong; Yuhui Zheng; Huiying Xu; Xinzhong Zhu
>
> **摘要:** Multi-view clustering has shown to be an effective method for analyzing underlying patterns in multi-view data. The performance of clustering can be improved by learning the consistency and complementarity between multi-view features, however, cluster-oriented representation learning is often overlooked. In this paper, we propose a novel Bi-level Decoupling and Consistency Learning framework (BDCL) to further explore the effective representation for multi-view data to enhance inter-cluster discriminability and intra-cluster compactness of features in multi-view clustering. Our framework comprises three modules: 1) The multi-view instance learning module aligns the consistent information while preserving the private features between views through reconstruction autoencoder and contrastive learning. 2) The bi-level decoupling of features and clusters enhances the discriminability of feature space and cluster space. 3) The consistency learning module treats the different views of the sample and their neighbors as positive pairs, learns the consistency of their clustering assignments, and further compresses the intra-cluster space. Experimental results on five benchmark datasets demonstrate the superiority of the proposed method compared with the SOTA methods. Our code is published on https://github.com/LouisDong95/BDCL.
>
---
#### [new 036] RotBench: Evaluating Multimodal Large Language Models on Identifying Image Rotation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出RotBench基准，评估多模态大模型在识别图像0°、90°、180°、270°旋转上的能力。发现多数模型难以区分90°与270°旋转，且微调和提示策略改进有限，揭示其空间推理能力距人类仍有差距。**

- **链接: [http://arxiv.org/pdf/2508.13968v1](http://arxiv.org/pdf/2508.13968v1)**

> **作者:** Tianyi Niu; Jaemin Cho; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** 20 pages. Code and data: https://github.com/tianyiniu/RotBench
>
> **摘要:** We investigate to what extent Multimodal Large Language Models (MLLMs) can accurately identify the orientation of input images rotated 0{\deg}, 90{\deg}, 180{\deg}, and 270{\deg}. This task demands robust visual reasoning capabilities to detect rotational cues and contextualize spatial relationships within images, regardless of their orientation. To evaluate MLLMs on these abilities, we introduce RotBench -- a 350-image manually-filtered benchmark comprising lifestyle, portrait, and landscape images. Despite the relatively simple nature of this task, we show that several state-of-the-art open and proprietary MLLMs, including GPT-5, o3, and Gemini-2.5-Pro, do not reliably identify rotation in input images. Providing models with auxiliary information -- including captions, depth maps, and more -- or using chain-of-thought prompting offers only small and inconsistent improvements. Our results indicate that most models are able to reliably identify right-side-up (0{\deg}) images, while certain models are able to identify upside-down (180{\deg}) images. None can reliably distinguish between 90{\deg} and 270{\deg}. Simultaneously showing the image rotated in different orientations leads to moderate performance gains for reasoning models, while a modified setup using voting improves the performance of weaker models. We further show that fine-tuning does not improve models' ability to distinguish 90{\deg} and 270{\deg} rotations, despite substantially improving the identification of 180{\deg} images. Together, these results reveal a significant gap between MLLMs' spatial reasoning capabilities and human perception in identifying rotation.
>
---
#### [new 037] ViT-FIQA: Assessing Face Image Quality using Vision Transformers
- **分类: cs.CV**

- **简介: 论文提出ViT-FIQA，用视觉Transformer评估人脸图像质量。解决传统方法依赖CNN、忽视ViT潜力的问题。通过可学习质量标记，结合全局自注意力机制，同时预测图像质量与提取人脸特征，在多个基准上表现优异。**

- **链接: [http://arxiv.org/pdf/2508.13957v1](http://arxiv.org/pdf/2508.13957v1)**

> **作者:** Andrea Atzori; Fadi Boutros; Naser Damer
>
> **备注:** Accepted at the IEEE/CVF International Conference on Computer Vision Workshops 2025 (ICCVW 2025)
>
> **摘要:** Face Image Quality Assessment (FIQA) aims to predict the utility of a face image for face recognition (FR) systems. State-of-the-art FIQA methods mainly rely on convolutional neural networks (CNNs), leaving the potential of Vision Transformer (ViT) architectures underexplored. This work proposes ViT-FIQA, a novel approach that extends standard ViT backbones, originally optimized for FR, through a learnable quality token designed to predict a scalar utility score for any given face image. The learnable quality token is concatenated with the standard image patch tokens, and the whole sequence is processed via global self-attention by the ViT encoders to aggregate contextual information across all patches. At the output of the backbone, ViT-FIQA branches into two heads: (1) the patch tokens are passed through a fully connected layer to learn discriminative face representations via a margin-penalty softmax loss, and (2) the quality token is fed into a regression head to learn to predict the face sample's utility. Extensive experiments on challenging benchmarks and several FR models, including both CNN- and ViT-based architectures, demonstrate that ViT-FIQA consistently achieves top-tier performance. These results underscore the effectiveness of transformer-based architectures in modeling face image utility and highlight the potential of ViTs as a scalable foundation for future FIQA research https://cutt.ly/irHlzXUC.
>
---
#### [new 038] Self-Aware Adaptive Alignment: Enabling Accurate Perception for Intelligent Transportation Systems
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13823v1](http://arxiv.org/pdf/2508.13823v1)**

> **作者:** Tong Xiang; Hongxia Zhao; Fenghua Zhu; Yuanyuan Chen; Yisheng Lv
>
> **备注:** Domain adaptation, Virtual Reality, Object Detection
>
> **摘要:** Achieving top-notch performance in Intelligent Transportation detection is a critical research area. However, many challenges still need to be addressed when it comes to detecting in a cross-domain scenario. In this paper, we propose a Self-Aware Adaptive Alignment (SA3), by leveraging an efficient alignment mechanism and recognition strategy. Our proposed method employs a specified attention-based alignment module trained on source and target domain datasets to guide the image-level features alignment process, enabling the local-global adaptive alignment between the source domain and target domain. Features from both domains, whose channel importance is re-weighted, are fed into the region proposal network, which facilitates the acquisition of salient region features. Also, we introduce an instance-to-image level alignment module specific to the target domain to adaptively mitigate the domain gap. To evaluate the proposed method, extensive experiments have been conducted on popular cross-domain object detection benchmarks. Experimental results show that SA3 achieves superior results to the previous state-of-the-art methods.
>
---
#### [new 039] Enhancing Robustness of Implicit Neural Representations Against Weight Perturbations
- **分类: cs.CV**

- **简介: 论文研究隐式神经表示（INRs）在权重扰动下的鲁棒性问题，提出一种新损失函数以最小化扰动前后损失差异，显著提升重建质量，在多模态任务中PSNR提升达7.5 dB。**

- **链接: [http://arxiv.org/pdf/2508.13481v1](http://arxiv.org/pdf/2508.13481v1)**

> **作者:** Wenyong Zhou; Yuxin Cheng; Zhengwu Liu; Taiqiang Wu; Chen Zhang; Ngai Wong
>
> **备注:** 4 pages, 7 figures
>
> **摘要:** Implicit Neural Representations (INRs) encode discrete signals in a continuous manner using neural networks, demonstrating significant value across various multimedia applications. However, the vulnerability of INRs presents a critical challenge for their real-world deployments, as the network weights might be subjected to unavoidable perturbations. In this work, we investigate the robustness of INRs for the first time and find that even minor perturbations can lead to substantial performance degradation in the quality of signal reconstruction. To mitigate this issue, we formulate the robustness problem in INRs by minimizing the difference between loss with and without weight perturbations. Furthermore, we derive a novel robust loss function to regulate the gradient of the reconstruction loss with respect to weights, thereby enhancing the robustness. Extensive experiments on reconstruction tasks across multiple modalities demonstrate that our method achieves up to a 7.5~dB improvement in peak signal-to-noise ratio (PSNR) values compared to original INRs under noisy conditions.
>
---
#### [new 040] ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans
- **分类: cs.CV; cs.RO; 68T45**

- **简介: 论文提出ResPlan，一个包含1.7万张住宅平面图的大规模数据集，用于推动空间智能研究。它解决现有数据集规模小、多样性不足的问题，提供高保真度和结构多样性的标注数据，支持机器人、AI生成、VR等应用。**

- **链接: [http://arxiv.org/pdf/2508.14006v1](http://arxiv.org/pdf/2508.14006v1)**

> **作者:** Mohamed Abouagour; Eleftherios Garyfallidis
>
> **备注:** 18 pages, 3 figures, 4 tables
>
> **摘要:** We introduce ResPlan, a large-scale dataset of 17,000 detailed, structurally rich, and realistic residential floor plans, created to advance spatial AI research. Each plan includes precise annotations of architectural elements (walls, doors, windows, balconies) and functional spaces (such as kitchens, bedrooms, and bathrooms). ResPlan addresses key limitations of existing datasets such as RPLAN (Wu et al., 2019) and MSD (van Engelenburg et al., 2024) by offering enhanced visual fidelity and greater structural diversity, reflecting realistic and non-idealized residential layouts. Designed as a versatile, general-purpose resource, ResPlan supports a wide range of applications including robotics, reinforcement learning, generative AI, virtual and augmented reality, simulations, and game development. Plans are provided in both geometric and graph-based formats, enabling direct integration into simulation engines and fast 3D conversion. A key contribution is an open-source pipeline for geometry cleaning, alignment, and annotation refinement. Additionally, ResPlan includes structured representations of room connectivity, supporting graph-based spatial reasoning tasks. Finally, we present comparative analyses with existing benchmarks and outline several open benchmark tasks enabled by ResPlan. Ultimately, ResPlan offers a significant advance in scale, realism, and usability, providing a robust foundation for developing and benchmarking next-generation spatial intelligence systems.
>
---
#### [new 041] Physics-Based 3D Simulation for Synthetic Data Generation and Failure Analysis in Packaging Stability Assessment
- **分类: cs.CV**

- **简介: 论文提出物理驱动的3D仿真系统，用于包装稳定性评估，解决传统实验成本高、效率低问题。通过虚拟环境模拟不同包装布局与材料，训练神经网络预测堆垛安全性，提升分析精度与环保性。**

- **链接: [http://arxiv.org/pdf/2508.13989v1](http://arxiv.org/pdf/2508.13989v1)**

> **作者:** Samuel Seligardi; Pietro Musoni; Eleonora Iotti; Gianluca Contesso; Alessandro Dal Palù
>
> **摘要:** The design and analysis of pallet setups are essential for ensuring safety of packages transportation. With rising demands in the logistics sector, the development of automated systems utilizing advanced technologies has become increasingly crucial. Moreover, the widespread use of plastic wrapping has motivated researchers to investigate eco-friendly alternatives that still adhere to safety standards. We present a fully controllable and accurate physical simulation system capable of replicating the behavior of moving pallets. It features a 3D graphics-based virtual environment that supports a wide range of configurations, including variable package layouts, different wrapping materials, and diverse dynamic conditions. This innovative approach reduces the need for physical testing, cutting costs and environmental impact while improving measurement accuracy for analyzing pallet dynamics. Additionally, we train a deep neural network to evaluate the rendered videos generated by our simulator, as a crash-test predictor for pallet configurations, further enhancing the system's utility in safety analysis.
>
---
#### [new 042] ROVR-Open-Dataset: A Large-Scale Depth Dataset for Autonomous Driving
- **分类: cs.CV**

- **简介: 论文提出ROVR-Open-Dataset，一个大规模、多样化的单目深度估计数据集，旨在解决现有数据集多样性不足和深度密度低的问题，支持自动驾驶场景下的深度估计研究与模型训练。**

- **链接: [http://arxiv.org/pdf/2508.13977v1](http://arxiv.org/pdf/2508.13977v1)**

> **作者:** Xianda Guo; Ruijun Zhang; Yiqun Duan; Ruilin Wang; Keyuan Zhou; Wenzhao Zheng; Wenke Huang; Gangwei Xu; Mike Horton; Yuan Si; Hao Zhao; Long Chen
>
> **摘要:** Depth estimation is a fundamental task for 3D scene understanding in autonomous driving, robotics, and augmented reality. Existing depth datasets, such as KITTI, nuScenes, and DDAD, have advanced the field but suffer from limitations in diversity and scalability. As benchmark performance on these datasets approaches saturation, there is an increasing need for a new generation of large-scale, diverse, and cost-efficient datasets to support the era of foundation models and multi-modal learning. To address these challenges, we introduce a large-scale, diverse, frame-wise continuous dataset for depth estimation in dynamic outdoor driving environments, comprising 20K video frames to evaluate existing methods. Our lightweight acquisition pipeline ensures broad scene coverage at low cost, while sparse yet statistically sufficient ground truth enables robust training. Compared to existing datasets, ours presents greater diversity in driving scenarios and lower depth density, creating new challenges for generalization. Benchmark experiments with standard monocular depth estimation models validate the dataset's utility and highlight substantial performance gaps in challenging conditions, establishing a new platform for advancing depth estimation research.
>
---
#### [new 043] RED.AI Id-Pattern: First Results of Stone Deterioration Patterns with Multi-Agent Systems
- **分类: cs.CV; cs.MA; I.2.11; I.5.4**

- **简介: 该论文属于计算机视觉与人工智能在文化遗产保护中的应用任务，旨在解决传统石材劣化诊断耗时费力的问题。作者构建了一个由五类专家AI代理组成的多智能体系统，通过协作自动识别石材病害模式，并在28张复杂图像上验证了其显著优于基础模型的效果。**

- **链接: [http://arxiv.org/pdf/2508.13872v1](http://arxiv.org/pdf/2508.13872v1)**

> **作者:** Daniele Corradetti; José Delgado Rodrigues
>
> **备注:** 11 pages, 1 figure, 1 table. Contribution for REEACH 2025 Symposium
>
> **摘要:** The Id-Pattern system within the RED.AI project (Reabilita\c{c}\~ao Estrutural Digital atrav\'es da AI) consists of an agentic system designed to assist in the identification of stone deterioration patterns. Traditional methodologies, based on direct observation by expert teams, are accurate but costly in terms of time and resources. The system developed here introduces and evaluates a multi-agent artificial intelligence (AI) system, designed to simulate collaboration between experts and automate the diagnosis of stone pathologies from visual evidence. The approach is based on a cognitive architecture that orchestrates a team of specialized AI agents which, in this specific case, are limited to five: a lithologist, a pathologist, an environmental expert, a conservator-restorer, and a diagnostic coordinator. To evaluate the system we selected 28 difficult images involving multiple deterioration patterns. Our first results showed a huge boost on all metrics of our system compared to the foundational model.
>
---
#### [new 044] CORENet: Cross-Modal 4D Radar Denoising Network with LiDAR Supervision for Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CORENet，用于自动驾驶中的4D雷达去噪任务。针对雷达点云稀疏噪声大的问题，利用LiDAR监督提取特征，提升检测鲁棒性，且训练时用LiDAR、推理时仅用雷达。**

- **链接: [http://arxiv.org/pdf/2508.13485v1](http://arxiv.org/pdf/2508.13485v1)**

> **作者:** Fuyang Liu; Jilin Mei; Fangyuan Mao; Chen Min; Yan Xing; Yu Hu
>
> **备注:** 8 pages, 5 figures, Accepted to IROS 2025
>
> **摘要:** 4D radar-based object detection has garnered great attention for its robustness in adverse weather conditions and capacity to deliver rich spatial information across diverse driving scenarios. Nevertheless, the sparse and noisy nature of 4D radar point clouds poses substantial challenges for effective perception. To address the limitation, we present CORENet, a novel cross-modal denoising framework that leverages LiDAR supervision to identify noise patterns and extract discriminative features from raw 4D radar data. Designed as a plug-and-play architecture, our solution enables seamless integration into voxel-based detection frameworks without modifying existing pipelines. Notably, the proposed method only utilizes LiDAR data for cross-modal supervision during training while maintaining full radar-only operation during inference. Extensive evaluation on the challenging Dual-Radar dataset, which is characterized by elevated noise level, demonstrates the effectiveness of our framework in enhancing detection robustness. Comprehensive experiments validate that CORENet achieves superior performance compared to existing mainstream approaches.
>
---
#### [new 045] A Fully Transformer Based Multimodal Framework for Explainable Cancer Image Segmentation Using Radiology Reports
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Med-CTX框架，用于乳腺癌超声图像分割任务，通过融合临床报告与图像信息，实现高精度分割、不确定性估计和可解释性说明，提升诊断可信度。**

- **链接: [http://arxiv.org/pdf/2508.13796v1](http://arxiv.org/pdf/2508.13796v1)**

> **作者:** Enobong Adahada; Isabel Sassoon; Kate Hone; Yongmin Li
>
> **摘要:** We introduce Med-CTX, a fully transformer based multimodal framework for explainable breast cancer ultrasound segmentation. We integrate clinical radiology reports to boost both performance and interpretability. Med-CTX achieves exact lesion delineation by using a dual-branch visual encoder that combines ViT and Swin transformers, as well as uncertainty aware fusion. Clinical language structured with BI-RADS semantics is encoded by BioClinicalBERT and combined with visual features utilising cross-modal attention, allowing the model to provide clinically grounded, model generated explanations. Our methodology generates segmentation masks, uncertainty maps, and diagnostic rationales all at once, increasing confidence and transparency in computer assisted diagnosis. On the BUS-BRA dataset, Med-CTX achieves a Dice score of 99% and an IoU of 95%, beating existing baselines U-Net, ViT, and Swin. Clinical text plays a key role in segmentation accuracy and explanation quality, as evidenced by ablation studies that show a -5.4% decline in Dice score and -31% in CIDEr. Med-CTX achieves good multimodal alignment (CLIP score: 85%) and increased confi dence calibration (ECE: 3.2%), setting a new bar for trustworthy, multimodal medical architecture.
>
---
#### [new 046] HumanPCR: Probing MLLM Capabilities in Diverse Human-Centric Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13692v1](http://arxiv.org/pdf/2508.13692v1)**

> **作者:** Keliang Li; Hongze Shen; Hao Shi; Ruibing Hou; Hong Chang; Jie Huang; Chenghao Jia; Wen Wang; Yiling Wu; Dongmei Jiang; Shiguang Shan; Xilin Chen
>
> **摘要:** The aspiration for artificial general intelligence, fueled by the rapid progress of multimodal models, demands human-comparable performance across diverse environments. We propose HumanPCR, an evaluation suite for probing MLLMs' capacity about human-related visual contexts across three hierarchical levels: Perception, Comprehension, and Reasoning (denoted by Human-P, Human-C, and Human-R, respectively). Human-P and Human-C feature over 6,000 human-verified multiple choice questions, assessing massive tasks of 9 dimensions, including but not limited to essential skills frequently overlooked by existing benchmarks. Human-R offers a challenging manually curated video reasoning test that requires integrating multiple visual evidences, proactively extracting context beyond question cues, and applying human-like expertise. Each question includes human-annotated Chain-of-Thought (CoT) rationales with key visual evidence to support further research. Extensive evaluations on over 30 state-of-the-art models exhibit significant challenges in human-centric visual understanding, particularly in tasks involving detailed space perception, temporal understanding, and mind modeling. Moreover, analysis of Human-R reveals the struggle of models in extracting essential proactive visual evidence from diverse human scenes and their faulty reliance on query-guided retrieval. Even with advanced techniques like scaling visual contexts and test-time thinking yield only limited benefits. We hope HumanPCR and our findings will advance the development, evaluation, and human-centric application of multimodal models.
>
---
#### [new 047] Temporal-Conditional Referring Video Object Segmentation with Noise-Free Text-to-Video Diffusion Model
- **分类: cs.CV**

- **简介: 论文提出一种新的RVOS方法，通过改进分割头设计和利用无噪声文本到视频扩散模型提升边界分割精度，结合TCMR模块增强特征提取，显著改善了视频目标分割性能。**

- **链接: [http://arxiv.org/pdf/2508.13584v1](http://arxiv.org/pdf/2508.13584v1)**

> **作者:** Ruixin Zhang; Jiaqing Fan; Yifan Liao; Qian Qiao; Fanzhang Li
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Referring Video Object Segmentation (RVOS) aims to segment specific objects in a video according to textual descriptions. We observe that recent RVOS approaches often place excessive emphasis on feature extraction and temporal modeling, while relatively neglecting the design of the segmentation head. In fact, there remains considerable room for improvement in segmentation head design. To address this, we propose a Temporal-Conditional Referring Video Object Segmentation model, which innovatively integrates existing segmentation methods to effectively enhance boundary segmentation capability. Furthermore, our model leverages a text-to-video diffusion model for feature extraction. On top of this, we remove the traditional noise prediction module to avoid the randomness of noise from degrading segmentation accuracy, thereby simplifying the model while improving performance. Finally, to overcome the limited feature extraction capability of the VAE, we design a Temporal Context Mask Refinement (TCMR) module, which significantly improves segmentation quality without introducing complex designs. We evaluate our method on four public RVOS benchmarks, where it consistently achieves state-of-the-art performance.
>
---
#### [new 048] Diversity-enhanced Collaborative Mamba for Semi-supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对半监督医学图像分割任务，解决标注数据稀缺问题。提出DCMamba框架，从数据、网络和特征三方面增强多样性：引入patch级弱强混合增强、多样扫描协作模块和不确定加权对比学习，显著提升分割性能。**

- **链接: [http://arxiv.org/pdf/2508.13712v1](http://arxiv.org/pdf/2508.13712v1)**

> **作者:** Shumeng Li; Jian Zhang; Lei Qi; Luping Zhou; Yinghuan Shi; Yang Gao
>
> **摘要:** Acquiring high-quality annotated data for medical image segmentation is tedious and costly. Semi-supervised segmentation techniques alleviate this burden by leveraging unlabeled data to generate pseudo labels. Recently, advanced state space models, represented by Mamba, have shown efficient handling of long-range dependencies. This drives us to explore their potential in semi-supervised medical image segmentation. In this paper, we propose a novel Diversity-enhanced Collaborative Mamba framework (namely DCMamba) for semi-supervised medical image segmentation, which explores and utilizes the diversity from data, network, and feature perspectives. Firstly, from the data perspective, we develop patch-level weak-strong mixing augmentation with Mamba's scanning modeling characteristics. Moreover, from the network perspective, we introduce a diverse-scan collaboration module, which could benefit from the prediction discrepancies arising from different scanning directions. Furthermore, from the feature perspective, we adopt an uncertainty-weighted contrastive learning mechanism to enhance the diversity of feature representation. Experiments demonstrate that our DCMamba significantly outperforms other semi-supervised medical image segmentation methods, e.g., yielding the latest SSM-based method by 6.69% on the Synapse dataset with 20% labeled data.
>
---
#### [new 049] Revisiting MLLM Token Technology through the Lens of Classical Visual Coding
- **分类: cs.CV**

- **简介: 论文将视觉编码原理引入MLLM令牌技术研究，通过统一框架对比分析令牌化、压缩与推理，旨在提升多模态模型效率与视觉编解码性能，推动二者协同发展。**

- **链接: [http://arxiv.org/pdf/2508.13460v1](http://arxiv.org/pdf/2508.13460v1)**

> **作者:** Jinming Liu; Junyan Lin; Yuntao Wei; Kele Shao; Keda Tao; Jianguo Huang; Xudong Yang; Zhibo Chen; Huan Wang; Xin Jin
>
> **摘要:** Classical visual coding and Multimodal Large Language Model (MLLM) token technology share the core objective - maximizing information fidelity while minimizing computational cost. Therefore, this paper reexamines MLLM token technology, including tokenization, token compression, and token reasoning, through the established principles of long-developed visual coding area. From this perspective, we (1) establish a unified formulation bridging token technology and visual coding, enabling a systematic, module-by-module comparative analysis; (2) synthesize bidirectional insights, exploring how visual coding principles can enhance MLLM token techniques' efficiency and robustness, and conversely, how token technology paradigms can inform the design of next-generation semantic visual codecs; (3) prospect for promising future research directions and critical unsolved challenges. In summary, this study presents the first comprehensive and structured technology comparison of MLLM token and visual coding, paving the way for more efficient multimodal models and more powerful visual codecs simultaneously.
>
---
#### [new 050] SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成任务，解决生成图像与文本提示对齐不佳的问题。提出SAGA方法，通过学习信号对齐分布，在无需训练的情况下提升生成图像的准确性与可控性，并支持多模态条件输入。**

- **链接: [http://arxiv.org/pdf/2508.13866v1](http://arxiv.org/pdf/2508.13866v1)**

> **作者:** Paul Grimal; Michaël Soumm; Hervé Le Borgne; Olivier Ferret; Akihiro Sugimoto
>
> **摘要:** State-of-the-art text-to-image models produce visually impressive results but often struggle with precise alignment to text prompts, leading to missing critical elements or unintended blending of distinct concepts. We propose a novel approach that learns a high-success-rate distribution conditioned on a target prompt, ensuring that generated images faithfully reflect the corresponding prompts. Our method explicitly models the signal component during the denoising process, offering fine-grained control that mitigates over-optimization and out-of-distribution artifacts. Moreover, our framework is training-free and seamlessly integrates with both existing diffusion and flow matching architectures. It also supports additional conditioning modalities -- such as bounding boxes -- for enhanced spatial alignment. Extensive experiments demonstrate that our approach outperforms current state-of-the-art methods. The code is available at https://github.com/grimalPaul/gsn-factory.
>
---
#### [new 051] MR6D: Benchmarking 6D Pose Estimation for Mobile Robots
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MR6D数据集，用于移动机器人在工业环境中的6D姿态估计任务。针对现有数据集多聚焦家用小物体、忽略移动平台特有挑战的问题，MR6D包含92个真实场景和16个大尺寸物体，涵盖远距离视角、复杂遮挡等难点，揭示当前方法性能不足，推动移动端专用姿态估计研究。**

- **链接: [http://arxiv.org/pdf/2508.13775v1](http://arxiv.org/pdf/2508.13775v1)**

> **作者:** Anas Gouda; Shrutarv Awasthi; Christian Blesing; Lokeshwaran Manohar; Frank Hoffmann; Alice Kirchheim
>
> **备注:** accepted CVPR 2025 Workshop on Recovering 6D Object Pose (R6D)
>
> **摘要:** Existing 6D pose estimation datasets primarily focus on small household objects typically handled by robot arm manipulators, limiting their relevance to mobile robotics. Mobile platforms often operate without manipulators, interact with larger objects, and face challenges such as long-range perception, heavy self-occlusion, and diverse camera perspectives. While recent models generalize well to unseen objects, evaluations remain confined to household-like settings that overlook these factors. We introduce MR6D, a dataset designed for 6D pose estimation for mobile robots in industrial environments. It includes 92 real-world scenes featuring 16 unique objects across static and dynamic interactions. MR6D captures the challenges specific to mobile platforms, including distant viewpoints, varied object configurations, larger object sizes, and complex occlusion/self-occlusion patterns. Initial experiments reveal that current 6D pipelines underperform in these settings, with 2D segmentation being another hurdle. MR6D establishes a foundation for developing and evaluating pose estimation methods tailored to the demands of mobile robotics. The dataset is available at https://huggingface.co/datasets/anas-gouda/mr6d.
>
---
#### [new 052] Shape-from-Template with Generalised Camera
- **分类: cs.CV**

- **简介: 论文提出基于广义相机模型的非刚性形状恢复方法（Shape-from-Template），解决多视角2D关键点对应下3D形状重建问题，通过三种优化策略提升精度，适用于医学影像等场景。**

- **链接: [http://arxiv.org/pdf/2508.13791v1](http://arxiv.org/pdf/2508.13791v1)**

> **作者:** Agniva Sengupta; Stefan Zachow
>
> **备注:** Pre-print of the IMAVIS article: https://www.sciencedirect.com/science/article/abs/pii/S0262885625001672 Code and data in: https://git.zib.de/asengupta/sft-generalised
>
> **摘要:** This article presents a new method for non-rigidly registering a 3D shape to 2D keypoints observed by a constellation of multiple cameras. Non-rigid registration of a 3D shape to observed 2D keypoints, i.e., Shape-from-Template (SfT), has been widely studied using single images, but SfT with information from multiple-cameras jointly opens new directions for extending the scope of known use-cases such as 3D shape registration in medical imaging and registration from hand-held cameras, to name a few. We represent such multi-camera setup with the generalised camera model; therefore any collection of perspective or orthographic cameras observing any deforming object can be registered. We propose multiple approaches for such SfT: the first approach where the corresponded keypoints lie on a direction vector from a known 3D point in space, the second approach where the corresponded keypoints lie on a direction vector from an unknown 3D point in space but with known orientation w.r.t some local reference frame, and a third approach where, apart from correspondences, the silhouette of the imaged object is also known. Together, these form the first set of solutions to the SfT problem with generalised cameras. The key idea behind SfT with generalised camera is the improved reconstruction accuracy from estimating deformed shape while utilising the additional information from the mutual constraints between multiple views of a deformed object. The correspondence-based approaches are solved with convex programming while the silhouette-based approach is an iterative refinement of the results from the convex solutions. We demonstrate the accuracy of our proposed methods on many synthetic and real data
>
---
#### [new 053] RICO: Two Realistic Benchmarks and an In-Depth Analysis for Incremental Learning in Object Detection
- **分类: cs.CV**

- **简介: 论文提出RICO基准，解决增量学习在目标检测中因简化评估导致的性能误判问题。通过两个真实场景基准D-RICO和EC-RICO，揭示现有方法在适应性和记忆保持上的不足，并指出其根源在于教师模型弱、任务多样性难处理及模型塑性不足。**

- **链接: [http://arxiv.org/pdf/2508.13878v1](http://arxiv.org/pdf/2508.13878v1)**

> **作者:** Matthias Neuwirth-Trapp; Maarten Bieshaar; Danda Pani Paudel; Luc Van Gool
>
> **备注:** Accepted to ICCV Workshops 2025
>
> **摘要:** Incremental Learning (IL) trains models sequentially on new data without full retraining, offering privacy, efficiency, and scalability. IL must balance adaptability to new data with retention of old knowledge. However, evaluations often rely on synthetic, simplified benchmarks, obscuring real-world IL performance. To address this, we introduce two Realistic Incremental Object Detection Benchmarks (RICO): Domain RICO (D-RICO) features domain shifts with a fixed class set, and Expanding-Classes RICO (EC-RICO) integrates new domains and classes per IL step. Built from 14 diverse datasets covering real and synthetic domains, varying conditions (e.g., weather, time of day), camera sensors, perspectives, and labeling policies, both benchmarks capture challenges absent in existing evaluations. Our experiments show that all IL methods underperform in adaptability and retention, while replaying a small amount of previous data already outperforms all methods. However, individual training on the data remains superior. We heuristically attribute this gap to weak teachers in distillation, single models' inability to manage diverse tasks, and insufficient plasticity. Our code will be made publicly available.
>
---
#### [new 054] DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13669v1](http://arxiv.org/pdf/2508.13669v1)**

> **作者:** Dengxian Gong; Shunping Ji
>
> **备注:** Under review
>
> **摘要:** The automated extraction of complete and precise road network graphs from remote sensing imagery remains a critical challenge in geospatial computer vision. Segmentation-based approaches, while effective in pixel-level recognition, struggle to maintain topology fidelity after vectorization postprocessing. Graph-growing methods build more topologically faithful graphs but suffer from computationally prohibitive iterative ROI cropping. Graph-generating methods first predict global static candidate road network vertices, and then infer possible edges between vertices. They achieve fast topology-aware inference, but limits the dynamic insertion of vertices. To address these challenges, we propose DeH4R, a novel hybrid model that combines graph-generating efficiency and graph-growing dynamics. This is achieved by decoupling the task into candidate vertex detection, adjacent vertex prediction, initial graph contruction, and graph expansion. This architectural innovation enables dynamic vertex (edge) insertions while retaining fast inference speed and enhancing both topology fidelity and spatial consistency. Comprehensive evaluations on CityScale and SpaceNet benchmarks demonstrate state-of-the-art (SOTA) performance. DeH4R outperforms the prior SOTA graph-growing method RNGDet++ by 4.62 APLS and 10.18 IoU on CityScale, while being approximately 10 $\times$ faster. The code will be made publicly available at https://github.com/7777777FAN/DeH4R.
>
---
#### [new 055] DIME-Net: A Dual-Illumination Adaptive Enhancement Network Based on Retinex and Mixture-of-Experts
- **分类: cs.CV**

- **简介: 论文提出DIME-Net，解决复杂光照下图像退化问题，通过双光照自适应增强框架，结合Retinex理论与专家混合机制，实现低光和逆光图像的统一增强，提升图像质量与下游任务性能。**

- **链接: [http://arxiv.org/pdf/2508.13921v1](http://arxiv.org/pdf/2508.13921v1)**

> **作者:** Ziang Wang; Xiaoqin Wang; Dingyi Wang; Qiang Li; Shushan Qiao
>
> **备注:** Accepted at ACM Multimedia 2025 (ACM MM 2025)
>
> **摘要:** Image degradation caused by complex lighting conditions such as low-light and backlit scenarios is commonly encountered in real-world environments, significantly affecting image quality and downstream vision tasks. Most existing methods focus on a single type of illumination degradation and lack the ability to handle diverse lighting conditions in a unified manner. To address this issue, we propose a dual-illumination enhancement framework called DIME-Net. The core of our method is a Mixture-of-Experts illumination estimator module, where a sparse gating mechanism adaptively selects suitable S-curve expert networks based on the illumination characteristics of the input image. By integrating Retinex theory, this module effectively performs enhancement tailored to both low-light and backlit images. To further correct illumination-induced artifacts and color distortions, we design a damage restoration module equipped with Illumination-Aware Cross Attention and Sequential-State Global Attention mechanisms. In addition, we construct a hybrid illumination dataset, MixBL, by integrating existing datasets, allowing our model to achieve robust illumination adaptability through a single training process. Experimental results show that DIME-Net achieves competitive performance on both synthetic and real-world low-light and backlit datasets without any retraining. These results demonstrate its generalization ability and potential for practical multimedia applications under diverse and complex illumination conditions.
>
---
#### [new 056] Timestep-Compressed Attack on Spiking Neural Networks through Timestep-Level Backpropagation
- **分类: cs.CV; cs.NE**

- **链接: [http://arxiv.org/pdf/2508.13812v1](http://arxiv.org/pdf/2508.13812v1)**

> **作者:** Donghwa Kang; Doohyun Kim; Sang-Ki Ko; Jinkyu Lee; Hyeongboo Baek; Brent ByungHoon Kang
>
> **备注:** 8 pages
>
> **摘要:** State-of-the-art (SOTA) gradient-based adversarial attacks on spiking neural networks (SNNs), which largely rely on extending FGSM and PGD frameworks, face a critical limitation: substantial attack latency from multi-timestep processing, rendering them infeasible for practical real-time applications. This inefficiency stems from their design as direct extensions of ANN paradigms, which fail to exploit key SNN properties. In this paper, we propose the timestep-compressed attack (TCA), a novel framework that significantly reduces attack latency. TCA introduces two components founded on key insights into SNN behavior. First, timestep-level backpropagation (TLBP) is based on our finding that global temporal information in backpropagation to generate perturbations is not critical for an attack's success, enabling per-timestep evaluation for early stopping. Second, adversarial membrane potential reuse (A-MPR) is motivated by the observation that initial timesteps are inefficiently spent accumulating membrane potential, a warm-up phase that can be pre-calculated and reused. Our experiments on VGG-11 and ResNet-17 with the CIFAR-10/100 and CIFAR10-DVS datasets show that TCA significantly reduces the required attack latency by up to 56.6% and 57.1% compared to SOTA methods in white-box and black-box settings, respectively, while maintaining a comparable attack success rate.
>
---
#### [new 057] Distribution-Aware Hadamard Quantization for Hardware-Efficient Implicit Neural Representations
- **分类: cs.CV**

- **简介: 论文针对隐式神经表示（INRs）硬件效率低的问题，提出DHQ量化方法，通过Hadamard变换统一权重与激活分布，实现权重和激活联合量化，在FPGA上显著降低延迟、能耗与资源占用。**

- **链接: [http://arxiv.org/pdf/2508.13478v1](http://arxiv.org/pdf/2508.13478v1)**

> **作者:** Wenyong Zhou; Jiachen Ren; Taiqiang Wu; Yuxin Cheng; Zhengwu Liu; Ngai Wong
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Implicit Neural Representations (INRs) encode discrete signals using Multi-Layer Perceptrons (MLPs) with complex activation functions. While INRs achieve superior performance, they depend on full-precision number representation for accurate computation, resulting in significant hardware overhead. Previous INR quantization approaches have primarily focused on weight quantization, offering only limited hardware savings due to the lack of activation quantization. To fully exploit the hardware benefits of quantization, we propose DHQ, a novel distribution-aware Hadamard quantization scheme that targets both weights and activations in INRs. Our analysis shows that the weights in the first and last layers have distributions distinct from those in the intermediate layers, while the activations in the last layer differ significantly from those in the preceding layers. Instead of customizing quantizers individually, we utilize the Hadamard transformation to standardize these diverse distributions into a unified bell-shaped form, supported by both empirical evidence and theoretical analysis, before applying a standard quantizer. To demonstrate the practical advantages of our approach, we present an FPGA implementation of DHQ that highlights its hardware efficiency. Experiments on diverse image reconstruction tasks show that DHQ outperforms previous quantization methods, reducing latency by 32.7\%, energy consumption by 40.1\%, and resource utilization by up to 98.3\% compared to full-precision counterparts.
>
---
#### [new 058] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos
- **分类: cs.CV**

- **简介: 论文提出LongSplat，用于从随意拍摄的长视频中实现鲁棒的3D高斯点渲染。解决相机位姿未知、运动不规则和场景庞大导致的渲染质量差、位姿漂移等问题，通过联合优化、鲁棒位姿估计和八叉树锚点机制提升效果与效率。**

- **链接: [http://arxiv.org/pdf/2508.14041v1](http://arxiv.org/pdf/2508.14041v1)**

> **作者:** Chin-Yang Lin; Cheng Sun; Fu-En Yang; Min-Hung Chen; Yen-Yu Lin; Yu-Lun Liu
>
> **备注:** ICCV 2025. Project page: https://linjohnss.github.io/longsplat/
>
> **摘要:** LongSplat addresses critical challenges in novel view synthesis (NVS) from casually captured long videos characterized by irregular camera motion, unknown camera poses, and expansive scenes. Current methods often suffer from pose drift, inaccurate geometry initialization, and severe memory limitations. To address these issues, we introduce LongSplat, a robust unposed 3D Gaussian Splatting framework featuring: (1) Incremental Joint Optimization that concurrently optimizes camera poses and 3D Gaussians to avoid local minima and ensure global consistency; (2) a robust Pose Estimation Module leveraging learned 3D priors; and (3) an efficient Octree Anchor Formation mechanism that converts dense point clouds into anchors based on spatial density. Extensive experiments on challenging benchmarks demonstrate that LongSplat achieves state-of-the-art results, substantially improving rendering quality, pose accuracy, and computational efficiency compared to prior approaches. Project page: https://linjohnss.github.io/longsplat/
>
---
#### [new 059] TalkVid: A Large-Scale Diversified Dataset for Audio-Driven Talking Head Synthesis
- **分类: cs.CV**

- **简介: 论文提出TalkVid数据集，解决音频驱动人脸合成模型在种族、语言和年龄多样性上的泛化不足问题。该数据集包含1244小时视频，7729名独特说话者，并构建了分层评估集TalkVid-Bench，用于揭示模型在子群体中的性能差异。**

- **链接: [http://arxiv.org/pdf/2508.13618v1](http://arxiv.org/pdf/2508.13618v1)**

> **作者:** Shunian Chen; Hejin Huang; Yexin Liu; Zihan Ye; Pengcheng Chen; Chenghao Zhu; Michael Guan; Rongsheng Wang; Junying Chen; Guanbin Li; Ser-Nam Lim; Harry Yang; Benyou Wang
>
> **摘要:** Audio-driven talking head synthesis has achieved remarkable photorealism, yet state-of-the-art (SOTA) models exhibit a critical failure: they lack generalization to the full spectrum of human diversity in ethnicity, language, and age groups. We argue that this generalization gap is a direct symptom of limitations in existing training data, which lack the necessary scale, quality, and diversity. To address this challenge, we introduce TalkVid, a new large-scale, high-quality, and diverse dataset containing 1244 hours of video from 7729 unique speakers. TalkVid is curated through a principled, multi-stage automated pipeline that rigorously filters for motion stability, aesthetic quality, and facial detail, and is validated against human judgments to ensure its reliability. Furthermore, we construct and release TalkVid-Bench, a stratified evaluation set of 500 clips meticulously balanced across key demographic and linguistic axes. Our experiments demonstrate that a model trained on TalkVid outperforms counterparts trained on previous datasets, exhibiting superior cross-dataset generalization. Crucially, our analysis on TalkVid-Bench reveals performance disparities across subgroups that are obscured by traditional aggregate metrics, underscoring its necessity for future research. Code and data can be found in https://github.com/FreedomIntelligence/TalkVid
>
---
#### [new 060] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup
- **分类: cs.CV**

- **简介: 该论文提出DictAS框架，解决少样本异常分割中跨类别泛化问题。通过自监督字典构建与查找机制，仅用少量正常图像作为提示，实现无需重训练的异常检测，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.13560v1](http://arxiv.org/pdf/2508.13560v1)**

> **作者:** Zhen Qu; Xian Tao; Xinyi Gong; ShiChen Qu; Xiaopei Zhang; Xingang Wang; Fei Shen; Zhengtao Zhang; Mukesh Prasad; Guiguang Ding
>
> **备注:** Accepted by ICCV 2025, Project: https://github.com/xiaozhen228/DictAS
>
> **摘要:** Recent vision-language models (e.g., CLIP) have demonstrated remarkable class-generalizable ability to unseen classes in few-shot anomaly segmentation (FSAS), leveraging supervised prompt learning or fine-tuning on seen classes. However, their cross-category generalization largely depends on prior knowledge of real seen anomaly samples. In this paper, we propose a novel framework, namely DictAS, which enables a unified model to detect visual anomalies in unseen object categories without any retraining on the target data, only employing a few normal reference images as visual prompts. The insight behind DictAS is to transfer dictionary lookup capabilities to the FSAS task for unseen classes via self-supervised learning, instead of merely memorizing the normal and abnormal feature patterns from the training set. Specifically, DictAS mainly consists of three components: (1) **Dictionary Construction** - to simulate the index and content of a real dictionary using features from normal reference images. (2) **Dictionary Lookup** - to retrieve queried region features from the dictionary via a sparse lookup strategy. When a query feature cannot be retrieved, it is classified as an anomaly. (3) **Query Discrimination Regularization**- to enhance anomaly discrimination by making abnormal features harder to retrieve from the dictionary. To achieve this, Contrastive Query Constraint and Text Alignment Constraint are further proposed. Extensive experiments on seven public industrial and medical datasets demonstrate that DictAS consistently outperforms state-of-the-art FSAS methods.
>
---
#### [new 061] Forecasting Smog Events Using ConvLSTM: A Spatio-Temporal Approach for Aerosol Index Prediction in South Asia
- **分类: cs.CV**

- **简介: 该论文属于时空预测任务，旨在解决南亚地区雾霾事件实时预报难题。作者利用ConvLSTM模型和Sentinel-5P数据，基于紫外气溶胶指数预测气溶胶浓度，实现五日间隔的高精度预测。**

- **链接: [http://arxiv.org/pdf/2508.13891v1](http://arxiv.org/pdf/2508.13891v1)**

> **作者:** Taimur Khan
>
> **摘要:** The South Asian Smog refers to the recurring annual air pollution events marked by high contaminant levels, reduced visibility, and significant socio-economic impacts, primarily affecting the Indo-Gangetic Plains (IGP) from November to February. Over the past decade, increased air pollution sources such as crop residue burning, motor vehicles, and changing weather patterns have intensified these smog events. However, real-time forecasting systems for increased particulate matter concentrations are still not established at regional scale. The Aerosol Index, closely tied to smog formation and a key component in calculating the Air Quality Index (AQI), reflects particulate matter concentrations. This study forecasts aerosol events using Sentinel-5P air constituent data (2019-2023) and a Convolutional Long-Short Term Memory (ConvLSTM) neural network, which captures spatial and temporal correlations more effectively than previous models. Using the Ultraviolet (UV) Aerosol Index at 340-380 nm as the predictor, results show the Aerosol Index can be forecasted at five-day intervals with a Mean Squared Error of ~0.0018, loss of ~0.3995, and Structural Similarity Index of ~0.74. While effective, the model can be improved by integrating additional data and refining its architecture.
>
---
#### [new 062] DiffIER: Optimizing Diffusion Models with Iterative Error Reduction
- **分类: cs.CV**

- **简介: 论文提出DiffIER方法，针对扩散模型在条件生成中因指导权重敏感导致质量不稳定的问题，通过迭代优化每步误差减少训练-推理差距，提升文本到图像、超分辨率等任务的生成质量。**

- **链接: [http://arxiv.org/pdf/2508.13628v1](http://arxiv.org/pdf/2508.13628v1)**

> **作者:** Ao Chen; Lihe Ding; Tianfan Xue
>
> **摘要:** Diffusion models have demonstrated remarkable capabilities in generating high-quality samples and enhancing performance across diverse domains through Classifier-Free Guidance (CFG). However, the quality of generated samples is highly sensitive to the selection of the guidance weight. In this work, we identify a critical ``training-inference gap'' and we argue that it is the presence of this gap that undermines the performance of conditional generation and renders outputs highly sensitive to the guidance weight. We quantify this gap by measuring the accumulated error during the inference stage and establish a correlation between the selection of guidance weight and minimizing this gap. Furthermore, to mitigate this gap, we propose DiffIER, an optimization-based method for high-quality generation. We demonstrate that the accumulated error can be effectively reduced by an iterative error minimization at each step during inference. By introducing this novel plug-and-play optimization framework, we enable the optimization of errors at every single inference step and enhance generation quality. Empirical results demonstrate that our proposed method outperforms baseline approaches in conditional generation tasks. Furthermore, the method achieves consistent success in text-to-image generation, image super-resolution, and text-to-speech generation, underscoring its versatility and potential for broad applications in future research.
>
---
#### [new 063] Bridging the Gap: Doubles Badminton Analysis with Singles-Trained Models
- **分类: cs.CV**

- **简介: 论文将单人训练的模型迁移用于双人羽毛球分析，解决 doubles 数据稀缺和跟踪难题。通过关键点提取、对比学习嵌入和Transformer分类器，实现双人场景下的击球识别，为双打研究奠定基础。**

- **链接: [http://arxiv.org/pdf/2508.13507v1](http://arxiv.org/pdf/2508.13507v1)**

> **作者:** Seungheon Baek; Jinhyuk Yun
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Badminton is known as one of the fastest racket sports in the world. Despite doubles matches being more prevalent in international tournaments than singles, previous research has mainly focused on singles due to the challenges in data availability and multi-person tracking. To address this gap, we designed an approach that transfers singles-trained models to doubles analysis. We extracted keypoints from the ShuttleSet single matches dataset using ViT-Pose and embedded them through a contrastive learning framework based on ST-GCN. To improve tracking stability, we incorporated a custom multi-object tracking algorithm that resolves ID switching issues from fast and overlapping player movements. A Transformer-based classifier then determines shot occurrences based on the learned embeddings. Our findings demonstrate the feasibility of extending pose-based shot recognition to doubles badminton, broadening analytics capabilities. This work establishes a foundation for doubles-specific datasets to enhance understanding of this predominant yet understudied format of the fast racket sport.
>
---
#### [new 064] Evaluating Open-Source Vision Language Models for Facial Emotion Recognition against Traditional Deep Learning Models
- **分类: cs.CV; cs.AI**

- **简介: 论文研究面部情绪识别任务，比较开源视觉语言模型与传统深度学习模型在低质量数据上的表现。通过引入图像修复管道提升性能，发现传统模型显著优于VLMs，并提供计算成本分析，为部署提供依据。**

- **链接: [http://arxiv.org/pdf/2508.13524v1](http://arxiv.org/pdf/2508.13524v1)**

> **作者:** Vamsi Krishna Mulukutla; Sai Supriya Pavarala; Srinivasa Raju Rudraraju; Sridevi Bonthu
>
> **摘要:** Facial Emotion Recognition (FER) is crucial for applications such as human-computer interaction and mental health diagnostics. This study presents the first empirical comparison of open-source Vision-Language Models (VLMs), including Phi-3.5 Vision and CLIP, against traditional deep learning models VGG19, ResNet-50, and EfficientNet-B0 on the challenging FER-2013 dataset, which contains 35,887 low-resolution grayscale images across seven emotion classes. To address the mismatch between VLM training assumptions and the noisy nature of FER data, we introduce a novel pipeline that integrates GFPGAN-based image restoration with FER evaluation. Results show that traditional models, particularly EfficientNet-B0 (86.44%) and ResNet-50 (85.72%), significantly outperform VLMs like CLIP (64.07%) and Phi-3.5 Vision (51.66%), highlighting the limitations of VLMs in low-quality visual tasks. In addition to performance evaluation using precision, recall, F1-score, and accuracy, we provide a detailed computational cost analysis covering preprocessing, training, inference, and evaluation phases, offering practical insights for deployment. This work underscores the need for adapting VLMs to noisy environments and provides a reproducible benchmark for future research in emotion recognition.
>
---
#### [new 065] Enhancing Targeted Adversarial Attacks on Large Vision-Language Models through Intermediate Projector Guidance
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型（VLM）的定向对抗攻击任务，解决现有方法因依赖全局图像表示而难以实现细粒度操控的问题。提出IPGA方法，通过攻击Q-Former中间投影模块，实现更精确的视觉语义扰动，并引入RQA保持无关内容不变，显著提升攻击效果与跨模型迁移性。**

- **链接: [http://arxiv.org/pdf/2508.13739v1](http://arxiv.org/pdf/2508.13739v1)**

> **作者:** Yiming Cao; Yanjie Li; Kaisheng Liang; Yuni Lai; Bin Xiao
>
> **摘要:** Targeted adversarial attacks are essential for proactively identifying security flaws in Vision-Language Models before real-world deployment. However, current methods perturb images to maximize global similarity with the target text or reference image at the encoder level, collapsing rich visual semantics into a single global vector. This limits attack granularity, hindering fine-grained manipulations such as modifying a car while preserving its background. Furthermore, these methods largely overlook the projector module, a critical semantic bridge between the visual encoder and the language model in VLMs, thereby failing to disrupt the full vision-language alignment pipeline within VLMs and limiting attack effectiveness. To address these issues, we propose the Intermediate Projector Guided Attack (IPGA), the first method to attack using the intermediate stage of the projector module, specifically the widely adopted Q-Former, which transforms global image embeddings into fine-grained visual features. This enables more precise control over adversarial perturbations by operating on semantically meaningful visual tokens rather than a single global representation. Specifically, IPGA leverages the Q-Former pretrained solely on the first vision-language alignment stage, without LLM fine-tuning, which improves both attack effectiveness and transferability across diverse VLMs. Furthermore, we propose Residual Query Alignment (RQA) to preserve unrelated visual content, thereby yielding more controlled and precise adversarial manipulations. Extensive experiments show that our attack method consistently outperforms existing methods in both standard global image captioning tasks and fine-grained visual question-answering tasks in black-box environment. Additionally, IPGA successfully transfers to multiple commercial VLMs, including Google Gemini and OpenAI GPT.
>
---
#### [new 066] GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.14036v1](http://arxiv.org/pdf/2508.14036v1)**

> **作者:** Ken Deng; Yunhan Yang; Jingxiang Sun; Xihui Liu; Yebin Liu; Ding Liang; Yan-Pei Cao
>
> **备注:** https://detailgen3d.github.io/GeoSAM2/
>
> **摘要:** Modern 3D generation methods can rapidly create shapes from sparse or single views, but their outputs often lack geometric detail due to computational constraints. We present DetailGen3D, a generative approach specifically designed to enhance these generated 3D shapes. Our key insight is to model the coarse-to-fine transformation directly through data-dependent flows in latent space, avoiding the computational overhead of large-scale 3D generative models. We introduce a token matching strategy that ensures accurate spatial correspondence during refinement, enabling local detail synthesis while preserving global structure. By carefully designing our training data to match the characteristics of synthesized coarse shapes, our method can effectively enhance shapes produced by various 3D generation and reconstruction approaches, from single-view to sparse multi-view inputs. Extensive experiments demonstrate that DetailGen3D achieves high-fidelity geometric detail synthesis while maintaining efficiency in training.
>
---
#### [new 067] Automated Assessment of Aesthetic Outcomes in Facial Plastic Surgery
- **分类: cs.CV**

- **简介: 该论文提出了一种基于计算机视觉的美学评估框架，用于量化面部整形手术后的效果。任务是客观评估术后美学变化，解决传统主观评价的局限性。工作包括构建大规模配对术前术后图像数据集、开发多模态分析算法，并验证其在对称性、年龄感知和鼻部形态上的显著改善。**

- **链接: [http://arxiv.org/pdf/2508.13363v1](http://arxiv.org/pdf/2508.13363v1)**

> **作者:** Pegah Varghaei; Kiran Abraham-Aggarwal; Manoj T. Abraham; Arun Ross
>
> **摘要:** We introduce a scalable, interpretable computer-vision framework for quantifying aesthetic outcomes of facial plastic surgery using frontal photographs. Our pipeline leverages automated landmark detection, geometric facial symmetry computation, deep-learning-based age estimation, and nasal morphology analysis. To perform this study, we first assemble the largest curated dataset of paired pre- and post-operative facial images to date, encompassing 7,160 photographs from 1,259 patients. This dataset includes a dedicated rhinoplasty-only subset consisting of 732 images from 366 patients, 96.2% of whom showed improvement in at least one of the three nasal measurements with statistically significant group-level change. Among these patients, the greatest statistically significant improvements (p < 0.001) occurred in the alar width to face width ratio (77.0%), nose length to face height ratio (41.5%), and alar width to intercanthal ratio (39.3%). Among the broader frontal-view cohort, comprising 989 rigorously filtered subjects, 71.3% exhibited significant enhancements in global facial symmetry or perceived age (p < 0.01). Importantly, our analysis shows that patient identity remains consistent post-operatively, with True Match Rates of 99.5% and 99.6% at a False Match Rate of 0.01% for the rhinoplasty-specific and general patient cohorts, respectively. Additionally, we analyze inter-practitioner variability in improvement rates. By providing reproducible, quantitative benchmarks and a novel dataset, our pipeline facilitates data-driven surgical planning, patient counseling, and objective outcome evaluation across practices.
>
---
#### [new 068] AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes
- **分类: cs.CV; eess.IV**

- **简介: 论文提出AdaptiveAE，一种基于强化学习的自适应曝光策略，用于动态场景下的HDR成像。解决传统方法忽视快门速度与ISO交互及运动模糊的问题，通过融合运动模糊和噪声模拟的训练机制，优化曝光组合，在有限时间内提升HDR重建质量。**

- **链接: [http://arxiv.org/pdf/2508.13503v1](http://arxiv.org/pdf/2508.13503v1)**

> **作者:** Tianyi Xu; Fan Zhang; Boxin Shi; Tianfan Xue; Yujin Wang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Mainstream high dynamic range imaging techniques typically rely on fusing multiple images captured with different exposure setups (shutter speed and ISO). A good balance between shutter speed and ISO is crucial for achieving high-quality HDR, as high ISO values introduce significant noise, while long shutter speeds can lead to noticeable motion blur. However, existing methods often overlook the complex interaction between shutter speed and ISO and fail to account for motion blur effects in dynamic scenes. In this work, we propose AdaptiveAE, a reinforcement learning-based method that optimizes the selection of shutter speed and ISO combinations to maximize HDR reconstruction quality in dynamic environments. AdaptiveAE integrates an image synthesis pipeline that incorporates motion blur and noise simulation into our training procedure, leveraging semantic information and exposure histograms. It can adaptively select optimal ISO and shutter speed sequences based on a user-defined exposure time budget, and find a better exposure schedule than traditional solutions. Experimental results across multiple datasets demonstrate that it achieves the state-of-the-art performance.
>
---
#### [new 069] MINR: Efficient Implicit Neural Representations for Multi-Image Encoding
- **分类: cs.CV**

- **简介: 论文提出MINR，用于多图高效隐式神经表示。针对单图独立建模导致的参数冗余问题，通过共享中间层并保留输入输出层特异性，结合投影层捕捉图像独特特征，显著减少参数量（最多60%），同时保持重建与超分性能。**

- **链接: [http://arxiv.org/pdf/2508.13471v1](http://arxiv.org/pdf/2508.13471v1)**

> **作者:** Wenyong Zhou; Taiqiang Wu; Zhengwu Liu; Yuxin Cheng; Chen Zhang; Ngai Wong
>
> **备注:** 4 pages, 4 figures
>
> **摘要:** Implicit Neural Representations (INRs) aim to parameterize discrete signals through implicit continuous functions. However, formulating each image with a separate neural network~(typically, a Multi-Layer Perceptron (MLP)) leads to computational and storage inefficiencies when encoding multi-images. To address this issue, we propose MINR, sharing specific layers to encode multi-image efficiently. We first compare the layer-wise weight distributions for several trained INRs and find that corresponding intermediate layers follow highly similar distribution patterns. Motivated by this, we share these intermediate layers across multiple images while preserving the input and output layers as input-specific. In addition, we design an extra novel projection layer for each image to capture its unique features. Experimental results on image reconstruction and super-resolution tasks demonstrate that MINR can save up to 60\% parameters while maintaining comparable performance. Particularly, MINR scales effectively to handle 100 images, maintaining an average peak signal-to-noise ratio (PSNR) of 34 dB. Further analysis of various backbones proves the robustness of the proposed MINR.
>
---
#### [new 070] Unleashing Semantic and Geometric Priors for 3D Scene Completion
- **分类: cs.CV**

- **简介: 该论文针对3D语义场景补全任务，解决现有方法因耦合编码器导致语义与几何信息冲突的问题。提出FoundationSSC框架，通过源级和路径级双解耦设计，分离并优化语义与几何先验，结合轴向感知融合模块提升性能，在SemanticKITTI和SSCBench-KITTI-360上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2508.13601v1](http://arxiv.org/pdf/2508.13601v1)**

> **作者:** Shiyuan Chen; Wei Sui; Bohao Zhang; Zeyd Boukhers; John See; Cong Yang
>
> **备注:** 9 pages, 5 figures, 6 tables
>
> **摘要:** Camera-based 3D semantic scene completion (SSC) provides dense geometric and semantic perception for autonomous driving and robotic navigation. However, existing methods rely on a coupled encoder to deliver both semantic and geometric priors, which forces the model to make a trade-off between conflicting demands and limits its overall performance. To tackle these challenges, we propose FoundationSSC, a novel framework that performs dual decoupling at both the source and pathway levels. At the source level, we introduce a foundation encoder that provides rich semantic feature priors for the semantic branch and high-fidelity stereo cost volumes for the geometric branch. At the pathway level, these priors are refined through specialised, decoupled pathways, yielding superior semantic context and depth distributions. Our dual-decoupling design produces disentangled and refined inputs, which are then utilised by a hybrid view transformation to generate complementary 3D features. Additionally, we introduce a novel Axis-Aware Fusion (AAF) module that addresses the often-overlooked challenge of fusing these features by anisotropically merging them into a unified representation. Extensive experiments demonstrate the advantages of FoundationSSC, achieving simultaneous improvements in both semantic and geometric metrics, surpassing prior bests by +0.23 mIoU and +2.03 IoU on SemanticKITTI. Additionally, we achieve state-of-the-art performance on SSCBench-KITTI-360, with 21.78 mIoU and 48.61 IoU. The code will be released upon acceptance.
>
---
#### [new 071] Unsupervised Urban Tree Biodiversity Mapping from Street-Level Imagery Using Spatially-Aware Visual Clustering
- **分类: cs.CV; cs.LG**

- **简介: 论文提出一种无监督方法，通过街景图像与空间种植模式结合，实现城市树木物种多样性的精细映射，解决传统调查成本高、监督学习泛化差的问题，可推广至缺乏详细数据的城市。**

- **链接: [http://arxiv.org/pdf/2508.13814v1](http://arxiv.org/pdf/2508.13814v1)**

> **作者:** Diaa Addeen Abuhani; Marco Seccaroni; Martina Mazzarello; Imran Zualkernan; Fabio Duarte; Carlo Ratti
>
> **备注:** 26 pages, 7 figures, Nature Format
>
> **摘要:** Urban tree biodiversity is critical for climate resilience, ecological stability, and livability in cities, yet most municipalities lack detailed knowledge of their canopies. Field-based inventories provide reliable estimates of Shannon and Simpson diversity but are costly and time-consuming, while supervised AI methods require labeled data that often fail to generalize across regions. We introduce an unsupervised clustering framework that integrates visual embeddings from street-level imagery with spatial planting patterns to estimate biodiversity without labels. Applied to eight North American cities, the method recovers genus-level diversity patterns with high fidelity, achieving low Wasserstein distances to ground truth for Shannon and Simpson indices and preserving spatial autocorrelation. This scalable, fine-grained approach enables biodiversity mapping in cities lacking detailed inventories and offers a pathway for continuous, low-cost monitoring to support equitable access to greenery and adaptive management of urban ecosystems.
>
---
#### [new 072] Towards Efficient Vision State Space Models via Token Merging
- **分类: cs.CV**

- **简介: 论文针对视觉状态空间模型（SSM）的计算效率问题，提出一种名为MaMe的令牌合并策略。该方法通过状态转移参数衡量令牌重要性并保留序列信息，在保持性能的同时显著提升效率，适用于图像、视频和音频等多种任务。**

- **链接: [http://arxiv.org/pdf/2508.13599v1](http://arxiv.org/pdf/2508.13599v1)**

> **作者:** Jinyoung Park; Minseok Son; Changick Kim
>
> **备注:** under review
>
> **摘要:** State Space Models (SSMs) have emerged as powerful architectures in computer vision, yet improving their computational efficiency remains crucial for practical and scalable deployment.While token reduction serves as an effective approach for model efficiency, applying it to SSMs requires careful consideration of their unique sequential modeling capabilities.In this work, we propose MaMe, a token-merging strategy tailored for SSM-based vision models.MaMe addresses two key challenges: quantifying token importance and preserving sequential properties. Our approach leverages the state transition parameter $\mathbf{\Delta}$ as an informativeness measure and introduces strategic token arrangements to preserve sequential information flow.Extensive experiments demonstrate that MaMe achieves superior efficiency-performance trade-offs for both fine-tuned and off-the-shelf models. Particularly, our approach maintains robustness even under aggressive token reduction where existing methods undergo significant performance degradation.Beyond image classification, MaMe shows strong generalization capabilities across video and audio domains, establishing an effective approach for enhancing efficiency in diverse SSM applications.
>
---
#### [new 073] Self-Supervised Sparse Sensor Fusion for Long Range Perception
- **分类: cs.CV**

- **简介: 论文提出自监督稀疏传感器融合方法，解决长距离感知难题。针对高速公路上需250米感知距离的问题，设计高效3D特征编码与自监督预训练，提升检测精度与LiDAR预测性能。**

- **链接: [http://arxiv.org/pdf/2508.13995v1](http://arxiv.org/pdf/2508.13995v1)**

> **作者:** Edoardo Palladin; Samuel Brucker; Filippo Ghilotti; Praveen Narayanan; Mario Bijelic; Felix Heide
>
> **摘要:** Outside of urban hubs, autonomous cars and trucks have to master driving on intercity highways. Safe, long-distance highway travel at speeds exceeding 100 km/h demands perception distances of at least 250 m, which is about five times the 50-100m typically addressed in city driving, to allow sufficient planning and braking margins. Increasing the perception ranges also allows to extend autonomy from light two-ton passenger vehicles to large-scale forty-ton trucks, which need a longer planning horizon due to their high inertia. However, most existing perception approaches focus on shorter ranges and rely on Bird's Eye View (BEV) representations, which incur quadratic increases in memory and compute costs as distance grows. To overcome this limitation, we built on top of a sparse representation and introduced an efficient 3D encoding of multi-modal and temporal features, along with a novel self-supervised pre-training scheme that enables large-scale learning from unlabeled camera-LiDAR data. Our approach extends perception distances to 250 meters and achieves an 26.6% improvement in mAP in object detection and a decrease of 30.5% in Chamfer Distance in LiDAR forecasting compared to existing methods, reaching distances up to 250 meters. Project Page: https://light.princeton.edu/lrs4fusion/
>
---
#### [new 074] InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing
- **分类: cs.CV**

- **简介: 论文提出InfiniteTalk，解决音频驱动视频生成中面部与身体动作不协调的问题。通过稀疏帧视频配音新范式，实现长序列、全身同步的高质量语音动画生成。**

- **链接: [http://arxiv.org/pdf/2508.14033v1](http://arxiv.org/pdf/2508.14033v1)**

> **作者:** Shaoshu Yang; Zhe Kong; Feng Gao; Meng Cheng; Xiangyu Liu; Yong Zhang; Zhuoliang Kang; Wenhan Luo; Xunliang Cai; Ran He; Xiaoming Wei
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Recent breakthroughs in video AIGC have ushered in a transformative era for audio-driven human animation. However, conventional video dubbing techniques remain constrained to mouth region editing, resulting in discordant facial expressions and body gestures that compromise viewer immersion. To overcome this limitation, we introduce sparse-frame video dubbing, a novel paradigm that strategically preserves reference keyframes to maintain identity, iconic gestures, and camera trajectories while enabling holistic, audio-synchronized full-body motion editing. Through critical analysis, we identify why naive image-to-video models fail in this task, particularly their inability to achieve adaptive conditioning. Addressing this, we propose InfiniteTalk, a streaming audio-driven generator designed for infinite-length long sequence dubbing. This architecture leverages temporal context frames for seamless inter-chunk transitions and incorporates a simple yet effective sampling strategy that optimizes control strength via fine-grained reference frame positioning. Comprehensive evaluations on HDTF, CelebV-HQ, and EMTD datasets demonstrate state-of-the-art performance. Quantitative metrics confirm superior visual realism, emotional coherence, and full-body motion synchronization.
>
---
#### [new 075] Mitigating Cross-Image Information Leakage in LVLMs for Multi-Image Tasks
- **分类: cs.CV; cs.AI**

- **简介: 论文研究多图任务中大型视觉语言模型的跨图信息泄露问题，提出无需训练的FOCUS解码策略，通过逐图掩码与对比优化提升多图推理准确性。**

- **链接: [http://arxiv.org/pdf/2508.13744v1](http://arxiv.org/pdf/2508.13744v1)**

> **作者:** Yeji Park; Minyoung Lee; Sanghyuk Chun; Junsuk Choe
>
> **备注:** Source code is available at https://github.com/yejipark-m/FOCUS
>
> **摘要:** Large Vision-Language Models (LVLMs) demonstrate strong performance on single-image tasks. However, we observe that their performance degrades significantly when handling multi-image inputs. This occurs because visual cues from different images become entangled in the model's output. We refer to this phenomenon as cross-image information leakage. To address this issue, we propose FOCUS, a training-free and architecture-agnostic decoding strategy that mitigates cross-image information leakage during inference. FOCUS sequentially masks all but one image with random noise, guiding the model to focus on the single clean image. We repeat this process across all target images to obtain logits under partially masked contexts. These logits are aggregated and then contrastively refined using a noise-only reference input, which suppresses the leakage and yields more accurate outputs. FOCUS consistently improves performance across four multi-image benchmarks and diverse LVLM families. This demonstrates that FOCUS offers a general and practical solution for enhancing multi-image reasoning without additional training or architectural modifications.
>
---
#### [new 076] Structured Prompting and Multi-Agent Knowledge Distillation for Traffic Video Interpretation and Risk Inference
- **分类: cs.CV; cs.AI; cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.13439v1](http://arxiv.org/pdf/2508.13439v1)**

> **作者:** Yunxiang Yang; Ningning Xu; Jidong J. Yang
>
> **备注:** 16 pages, 10 figures, 1 table
>
> **摘要:** Comprehensive highway scene understanding and robust traffic risk inference are vital for advancing Intelligent Transportation Systems (ITS) and autonomous driving. Traditional approaches often struggle with scalability and generalization, particularly under the complex and dynamic conditions of real-world environments. To address these challenges, we introduce a novel structured prompting and knowledge distillation framework that enables automatic generation of high-quality traffic scene annotations and contextual risk assessments. Our framework orchestrates two large Vision-Language Models (VLMs): GPT-4o and o3-mini, using a structured Chain-of-Thought (CoT) strategy to produce rich, multi-perspective outputs. These outputs serve as knowledge-enriched pseudo-annotations for supervised fine-tuning of a much smaller student VLM. The resulting compact 3B-scale model, named VISTA (Vision for Intelligent Scene and Traffic Analysis), is capable of understanding low-resolution traffic videos and generating semantically faithful, risk-aware captions. Despite its significantly reduced parameter count, VISTA achieves strong performance across established captioning metrics (BLEU-4, METEOR, ROUGE-L, and CIDEr) when benchmarked against its teacher models. This demonstrates that effective knowledge distillation and structured multi-agent supervision can empower lightweight VLMs to capture complex reasoning capabilities. The compact architecture of VISTA facilitates efficient deployment on edge devices, enabling real-time risk monitoring without requiring extensive infrastructure upgrades.
>
---
#### [new 077] OmniTry: Virtual Try-On Anything without Masks
- **分类: cs.CV**

- **简介: 论文提出OmniTry，一个无需掩码的虚拟试穿框架，扩展VTON任务至任意穿戴物（如饰品），解决配对数据稀缺问题，通过两阶段训练实现精准定位与外观一致性。**

- **链接: [http://arxiv.org/pdf/2508.13632v1](http://arxiv.org/pdf/2508.13632v1)**

> **作者:** Yutong Feng; Linlin Zhang; Hengyuan Cao; Yiming Chen; Xiaoduan Feng; Jian Cao; Yuxiong Wu; Bin Wang
>
> **摘要:** Virtual Try-ON (VTON) is a practical and widely-applied task, for which most of existing works focus on clothes. This paper presents OmniTry, a unified framework that extends VTON beyond garment to encompass any wearable objects, e.g., jewelries and accessories, with mask-free setting for more practical application. When extending to various types of objects, data curation is challenging for obtaining paired images, i.e., the object image and the corresponding try-on result. To tackle this problem, we propose a two-staged pipeline: For the first stage, we leverage large-scale unpaired images, i.e., portraits with any wearable items, to train the model for mask-free localization. Specifically, we repurpose the inpainting model to automatically draw objects in suitable positions given an empty mask. For the second stage, the model is further fine-tuned with paired images to transfer the consistency of object appearance. We observed that the model after the first stage shows quick convergence even with few paired samples. OmniTry is evaluated on a comprehensive benchmark consisting of 12 common classes of wearable objects, with both in-shop and in-the-wild images. Experimental results suggest that OmniTry shows better performance on both object localization and ID-preservation compared with existing methods. The code, model weights, and evaluation benchmark of OmniTry will be made publicly available at https://omnitry.github.io/.
>
---
#### [new 078] OmViD: Omni-supervised active learning for video action detection
- **分类: cs.CV**

- **简介: 论文提出OmViD，一种多监督主动学习方法，用于视频动作检测任务。针对标注成本高问题，通过自适应选择不同粒度的标注类型（如标签、框、掩码），并利用3D超像素生成伪标签，显著降低标注成本且保持检测性能。**

- **链接: [http://arxiv.org/pdf/2508.13983v1](http://arxiv.org/pdf/2508.13983v1)**

> **作者:** Aayush Rana; Akash Kumar; Vibhav Vineet; Yogesh S Rawat
>
> **备注:** ICCVW'25
>
> **摘要:** Video action detection requires dense spatio-temporal annotations, which are both challenging and expensive to obtain. However, real-world videos often vary in difficulty and may not require the same level of annotation. This paper analyzes the appropriate annotation types for each sample and their impact on spatio-temporal video action detection. It focuses on two key aspects: 1) how to obtain varying levels of annotation for videos, and 2) how to learn action detection from different annotation types. The study explores video-level tags, points, scribbles, bounding boxes, and pixel-level masks. First, a simple active learning strategy is proposed to estimate the necessary annotation type for each video. Then, a novel spatio-temporal 3D-superpixel approach is introduced to generate pseudo-labels from these annotations, enabling effective training. The approach is validated on UCF101-24 and JHMDB-21 datasets, significantly cutting annotation costs with minimal performance loss.
>
---
#### [new 079] Vision Transformers for Kidney Stone Image Classification: A Comparative Study with CNNs
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决肾结石图像分类难题。通过对比Vision Transformer（ViT）与CNN模型，发现ViT在复杂图像条件下表现更优，提升了分类准确率和F1分数，证明其更适合肾结石图像分析。**

- **链接: [http://arxiv.org/pdf/2508.13461v1](http://arxiv.org/pdf/2508.13461v1)**

> **作者:** Ivan Reyes-Amezcua; Francisco Lopez-Tiro; Clement Larose; Andres Mendez-Vazquez; Gilberto Ochoa-Ruiz; Christian Daul
>
> **摘要:** Kidney stone classification from endoscopic images is critical for personalized treatment and recurrence prevention. While convolutional neural networks (CNNs) have shown promise in this task, their limited ability to capture long-range dependencies can hinder performance under variable imaging conditions. This study presents a comparative analysis between Vision Transformers (ViTs) and CNN-based models, evaluating their performance on two ex vivo datasets comprising CCD camera and flexible ureteroscope images. The ViT-base model pretrained on ImageNet-21k consistently outperformed a ResNet50 baseline across multiple imaging conditions. For instance, in the most visually complex subset (Section patches from endoscopic images), the ViT model achieved 95.2% accuracy and 95.1% F1-score, compared to 64.5% and 59.3% with ResNet50. In the mixed-view subset from CCD-camera images, ViT reached 87.1% accuracy versus 78.4% with CNN. These improvements extend across precision and recall as well. The results demonstrate that ViT-based architectures provide superior classification performance and offer a scalable alternative to conventional CNNs for kidney stone image analysis.
>
---
#### [new 080] GazeProphet: Software-Only Gaze Prediction for VR Foveated Rendering
- **分类: cs.CV**

- **简介: 论文提出GazeProphet，一种无需眼动硬件的VR注视预测方法，通过融合视觉Transformer与LSTM捕捉场景与 gaze 动态，实现高精度未来注视点预测，解决硬件依赖问题，提升VR视锥渲染效率。**

- **链接: [http://arxiv.org/pdf/2508.13546v1](http://arxiv.org/pdf/2508.13546v1)**

> **作者:** Farhaan Ebadulla; Chiraag Mudlapur; Gaurav BV
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Foveated rendering significantly reduces computational demands in virtual reality applications by concentrating rendering quality where users focus their gaze. Current approaches require expensive hardware-based eye tracking systems, limiting widespread adoption due to cost, calibration complexity, and hardware compatibility constraints. This paper presents GazeProphet, a software-only approach for predicting gaze locations in VR environments without requiring dedicated eye tracking hardware. The approach combines a Spherical Vision Transformer for processing 360-degree VR scenes with an LSTM-based temporal encoder that captures gaze sequence patterns. A multi-modal fusion network integrates spatial scene features with temporal gaze dynamics to predict future gaze locations with associated confidence estimates. Experimental evaluation on a comprehensive VR dataset demonstrates that GazeProphet achieves a median angular error of 3.83 degrees, outperforming traditional saliency-based baselines by 24% while providing reliable confidence calibration. The approach maintains consistent performance across different spatial regions and scene types, enabling practical deployment in VR systems without additional hardware requirements. Statistical analysis confirms the significance of improvements across all evaluation metrics. These results show that software-only gaze prediction can work for VR foveated rendering, making this performance boost more accessible to different VR platforms and apps.
>
---
#### [new 081] DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool Interleaved Vision-Language Model
- **分类: cs.CV**

- **简介: 论文提出DianJin-OCR-R1，通过推理与工具交替的视觉语言模型提升OCR准确性。针对大模型易幻觉和通用性不足问题，该方法结合专家模型结果进行反思推理，显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2508.13238v1](http://arxiv.org/pdf/2508.13238v1)**

> **作者:** Qian Chen; Xianyin Zhang; Lifan Guo; Feng Chen; Chi Zhang
>
> **摘要:** Recent advances in large vision-language models (LVLMs) have enabled a new paradigm of end-to-end document image parsing, excelling in Optical Character Recognition (OCR) tasks such as text, table, and formula recognition. However, generative LVLMs, similarly to large language models (LLMs), are prone to hallucinations--generating words that do not exist in input images. Furthermore, LVLMs are designed for general purposes and tend to be less effective on OCR tasks compared to expert models that are trained on domain-specific datasets. In this paper, we propose DianJin-OCR-R1, a reasoning-enhanced framework designed to address these limitations through training reasoning-and-tool interleaved VLMs. Given a recognition instruction, our DianJin-OCR-R1 model first recognizes the content in the input image by its own OCR capabilities, and then calls other tools (i.e., other expert models) to obtain their results as references, finally looks again the image and rethinks about the reasoning process to provide the final recognized content. Since architectures of expert models are tailored for specific OCR tasks, which makes them less prone to hallucinations, their results can help VLMs mitigate hallucinations. Additionally, expert models are typically smaller in scale and easy to iterate, enabling performance improvements for VLMs at a lower cost. We evaluate our model on ReST and OmniDocBench, and experimental results show that our DianJin-OCR-R1 models consistently outperform their non-reasoning counterparts and expert OCR models, which proves the effectiveness of our method.
>
---
#### [new 082] EDTalk++: Full Disentanglement for Controllable Talking Head Synthesis
- **分类: cs.CV**

- **简介: 该论文提出EDTalk++，用于可控人脸说话头合成任务。针对现有方法难以解耦多类面部运动且缺乏跨模态适应性的问题，设计四模块分离嘴型、姿态、眼神和表情，并通过正交基共享视觉先验，实现独立控制与音频驱动生成。**

- **链接: [http://arxiv.org/pdf/2508.13442v1](http://arxiv.org/pdf/2508.13442v1)**

> **作者:** Shuai Tan; Bin Ji
>
> **备注:** 17 pages,15 figures. arXiv admin note: substantial text overlap with arXiv:2404.01647
>
> **摘要:** Achieving disentangled control over multiple facial motions and accommodating diverse input modalities greatly enhances the application and entertainment of the talking head generation. This necessitates a deep exploration of the decoupling space for facial features, ensuring that they a) operate independently without mutual interference and b) can be preserved to share with different modal inputs, both aspects often neglected in existing methods. To address this gap, this paper proposes EDTalk++, a novel full disentanglement framework for controllable talking head generation. Our framework enables individual manipulation of mouth shape, head pose, eye movement, and emotional expression, conditioned on video or audio inputs. Specifically, we employ four lightweight modules to decompose the facial dynamics into four distinct latent spaces representing mouth, pose, eye, and expression, respectively. Each space is characterized by a set of learnable bases whose linear combinations define specific motions. To ensure independence and accelerate training, we enforce orthogonality among bases and devise an efficient training strategy to allocate motion responsibilities to each space without relying on external knowledge. The learned bases are then stored in corresponding banks, enabling shared visual priors with audio input. Furthermore, considering the properties of each space, we propose an Audio-to-Motion module for audio-driven talking head synthesis. Experiments are conducted to demonstrate the effectiveness of EDTalk++.
>
---
#### [new 083] FLAIR: Frequency- and Locality-Aware Implicit Neural Representations
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FLAIR，一种新型隐式神经表示方法，用于解决现有INRs在频率选择、空间定位和稀疏性上的不足。通过RC-GAUSS激活函数和WEGE编码机制，提升图像和3D重建质量。**

- **链接: [http://arxiv.org/pdf/2508.13544v1](http://arxiv.org/pdf/2508.13544v1)**

> **作者:** Sukhun Ko; Dahyeon Kye; Kyle Min; Chanho Eom; Jihyong Oh
>
> **备注:** Please visit our project page at https://cmlab-korea.github.io/FLAIR/
>
> **摘要:** Implicit Neural Representations (INRs) leverage neural networks to map coordinates to corresponding signals, enabling continuous and compact representations. This paradigm has driven significant advances in various vision tasks. However, existing INRs lack frequency selectivity, spatial localization, and sparse representations, leading to an over-reliance on redundant signal components. Consequently, they exhibit spectral bias, tending to learn low-frequency components early while struggling to capture fine high-frequency details. To address these issues, we propose FLAIR (Frequency- and Locality-Aware Implicit Neural Representations), which incorporates two key innovations. The first is RC-GAUSS, a novel activation designed for explicit frequency selection and spatial localization under the constraints of the time-frequency uncertainty principle (TFUP). The second is Wavelet-Energy-Guided Encoding (WEGE), which leverages the discrete wavelet transform (DWT) to compute energy scores and explicitly guide frequency information to the network. Our method consistently outperforms existing INRs in 2D image representation and restoration, as well as 3D reconstruction.
>
---
#### [new 084] FAMNet: Integrating 2D and 3D Features for Micro-expression Recognition via Multi-task Learning and Hierarchical Attention
- **分类: cs.CV**

- **简介: 论文提出FAMNet模型，用于微表情识别任务，解决微表情持续时间短、强度低导致特征提取困难的问题。通过多任务学习和分层注意力机制融合2D与3D CNN，提升细粒度时空特征提取能力，显著改善识别性能。**

- **链接: [http://arxiv.org/pdf/2508.13483v1](http://arxiv.org/pdf/2508.13483v1)**

> **作者:** Liangyu Fu; Xuecheng Wu; Danlei Huang; Xinyi Yin
>
> **备注:** 8 pages, 6 figures. Accepted to IJCNN 2025
>
> **摘要:** Micro-expressions recognition (MER) has essential application value in many fields, but the short duration and low intensity of micro-expressions (MEs) bring considerable challenges to MER. The current MER methods in deep learning mainly include three data loading methods: static images, dynamic image sequence, and a combination of the two streams. How to effectively extract MEs' fine-grained and spatiotemporal features has been difficult to solve. This paper proposes a new MER method based on multi-task learning and hierarchical attention, which fully extracts MEs' omni-directional features by merging 2D and 3D CNNs. The fusion model consists of a 2D CNN AMNet2D and a 3D CNN AMNet3D, with similar structures consisting of a shared backbone network Resnet18 and attention modules. During training, the model adopts different data loading methods to adapt to two specific networks respectively, jointly trains on the tasks of MER and facial action unit detection (FAUD), and adopts the parameter hard sharing for information association, which further improves the effect of the MER task, and the final fused model is called FAMNet. Extensive experimental results show that our proposed FAMNet significantly improves task performance. On the SAMM, CASME II and MMEW datasets, FAMNet achieves 83.75% (UAR) and 84.03% (UF1). Furthermore, on the challenging CAS(ME)$^3$ dataset, FAMNet achieves 51% (UAR) and 43.42% (UF1).
>
---
#### [new 085] MMIS-Net for Retinal Fluid Segmentation and Detection
- **分类: eess.IV; cs.CV**

- **简介: 论文提出MMIS-Net用于视网膜液体分割与检测任务，解决多源医学图像数据中标签不一致和模态差异问题。通过相似性融合块和one-hot标签空间，整合10个数据集提升模型泛化能力，在公开测试集上表现优异。**

- **链接: [http://arxiv.org/pdf/2508.13936v1](http://arxiv.org/pdf/2508.13936v1)**

> **作者:** Nchongmaje Ndipenocha; Alina Mirona; Kezhi Wanga; Yongmin Li
>
> **摘要:** Purpose: Deep learning methods have shown promising results in the segmentation, and detection of diseases in medical images. However, most methods are trained and tested on data from a single source, modality, organ, or disease type, overlooking the combined potential of other available annotated data. Numerous small annotated medical image datasets from various modalities, organs, and diseases are publicly available. In this work, we aim to leverage the synergistic potential of these datasets to improve performance on unseen data. Approach: To this end, we propose a novel algorithm called MMIS-Net (MultiModal Medical Image Segmentation Network), which features Similarity Fusion blocks that utilize supervision and pixel-wise similarity knowledge selection for feature map fusion. Additionally, to address inconsistent class definitions and label contradictions, we created a one-hot label space to handle classes absent in one dataset but annotated in another. MMIS-Net was trained on 10 datasets encompassing 19 organs across 2 modalities to build a single model. Results: The algorithm was evaluated on the RETOUCH grand challenge hidden test set, outperforming large foundation models for medical image segmentation and other state-of-the-art algorithms. We achieved the best mean Dice score of 0.83 and an absolute volume difference of 0.035 for the fluids segmentation task, as well as a perfect Area Under the Curve of 1 for the fluid detection task. Conclusion: The quantitative results highlight the effectiveness of our proposed model due to the incorporation of Similarity Fusion blocks into the network's backbone for supervision and similarity knowledge selection, and the use of a one-hot label space to address label class inconsistencies and contradictions.
>
---
#### [new 086] State of Abdominal CT Datasets: A Critical Review of Bias, Clinical Relevance, and Real-world Applicability
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学AI数据集评估任务，旨在解决现有腹部CT数据集存在的偏倚与临床适用性问题。作者系统分析46个公开数据集，发现高冗余和地域集中问题，并提出多中心合作、标准化协议等改进策略，以提升AI模型的公平性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.13626v1](http://arxiv.org/pdf/2508.13626v1)**

> **作者:** Saeide Danaei; Zahra Dehghanian; Elahe Meftah; Nariman Naderi; Seyed Amir Ahmad Safavi-Naini; Faeze Khorasanizade; Hamid R. Rabiee
>
> **备注:** Preprint. Submitted to IEEE Journal of Biomedical and Health Informatics (under review). 10 pages, 3 figures, 5 tables
>
> **摘要:** This systematic review critically evaluates publicly available abdominal CT datasets and their suitability for artificial intelligence (AI) applications in clinical settings. We examined 46 publicly available abdominal CT datasets (50,256 studies). Across all 46 datasets, we found substantial redundancy (59.1\% case reuse) and a Western/geographic skew (75.3\% from North America and Europe). A bias assessment was performed on the 19 datasets with >=100 cases; within this subset, the most prevalent high-risk categories were domain shift (63\%) and selection bias (57\%), both of which may undermine model generalizability across diverse healthcare environments -- particularly in resource-limited settings. To address these challenges, we propose targeted strategies for dataset improvement, including multi-institutional collaboration, adoption of standardized protocols, and deliberate inclusion of diverse patient populations and imaging technologies. These efforts are crucial in supporting the development of more equitable and clinically robust AI models for abdominal imaging.
>
---
#### [new 087] PediDemi -- A Pediatric Demyelinating Lesion Segmentation Dataset
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出PedDemi数据集，用于儿科脱髓鞘病变分割任务。针对儿童脱髓鞘疾病数据稀缺问题，收集13例患儿MRI及标注，包含ADEM和MS病例。通过模型评估验证数据质量与实用性，推动精准医疗研究。**

- **链接: [http://arxiv.org/pdf/2508.13239v1](http://arxiv.org/pdf/2508.13239v1)**

> **作者:** Maria Popa; Gabriela Adriana Visa
>
> **摘要:** Demyelinating disorders of the central nervous system may have multiple causes, the most common are infections, autoimmune responses, genetic or vascular etiology. Demyelination lesions are characterized by areas were the myelin sheath of the nerve fibers are broken or destroyed. Among autoimmune disorders, Multiple Sclerosis (MS) is the most well-known Among these disorders, Multiple Sclerosis (MS) is the most well-known and aggressive form. Acute Disseminated Encephalomyelitis (ADEM) is another type of demyelinating disease, typically with a better prognosis. Magnetic Resonance Imaging (MRI) is widely used for diagnosing and monitoring disease progression by detecting lesions. While both adults and children can be affected, there is a significant lack of publicly available datasets for pediatric cases and demyelinating disorders beyond MS. This study introduces, for the first time, a publicly available pediatric dataset for demyelinating lesion segmentation. The dataset comprises MRI scans from 13 pediatric patients diagnosed with demyelinating disorders, including 3 with ADEM. In addition to lesion segmentation masks, the dataset includes extensive patient metadata, such as diagnosis, treatment, personal medical background, and laboratory results. To assess the quality of the dataset and demonstrate its relevance, we evaluate a state-of-the-art lesion segmentation model trained on an existing MS dataset. The results underscore the importance of diverse datasets
>
---
#### [new 088] Towards Understanding and Harnessing the Transferability of Prognostic Knowledge in Computational Pathology
- **分类: eess.IV; cs.CV**

- **简介: 论文研究计算病理学中预后知识的可迁移性，解决罕见癌症样本少、无法利用其他癌症预后知识的问题。构建了13种癌症数据集UNI2-h-DSS，分析迁移因素并提出MoE-PKT方法提升预后预测性能。**

- **链接: [http://arxiv.org/pdf/2508.13482v1](http://arxiv.org/pdf/2508.13482v1)**

> **作者:** Pei Liu; Luping Ji; Jiaxiang Gou; Xiangxiang Zeng
>
> **备注:** 15 pages (13 figures and 5 tables)
>
> **摘要:** Whole-Slide Image (WSI) is an important tool for evaluating the prognosis of cancer patients. Present WSI-based prognosis studies generally follow a conventional paradigm -- cancer-specific model development -- where one cancer disease corresponds to one model and this model cannot make use of the prognostic knowledge from others. Despite its notable success in recent years, this paradigm has inherent limitations and has always been struggling with practical requirements: (i) scaling to the rare tumor diseases with very limited samples and (ii) benefiting from the generalizable prognostic knowledge in other cancers. To this end, this paper presents the first systematic study on Prognostic Knowledge Transfer in Pathology, called Path-PKT. It comprises three main parts. (1) We curate a large dataset (UNI2-h-DSS) with 13 cancers and use it to evaluate the transferability of prognostic knowledge between different cancers computationally. (2) We design experiments to understand what factors affect knowledge transfer and what causes positive transfers. (3) Motivated by empirical findings, we propose a new baseline approach (MoE-PKT) with a routing mechanism to utilize the generalizable prognostic knowledge in other cancers. Finally, we show the transferability of source models to rare tumor diseases. This study could lay solid foundations for the study of knowledge transfer in WSI-based cancer prognosis. Source code is available at https://github.com/liupei101/Path-PKT.
>
---
#### [new 089] Real-Time, Population-Based Reconstruction of 3D Bone Models via Very-Low-Dose Protocols
- **分类: eess.IV; cs.CV**

- **简介: 论文提出SSR-KD框架，通过低剂量双平面X射线实现30秒内重建高精度骨模型（误差<1mm），解决CT依赖与手动标注耗时问题，支持术中导航与术前规划，提升骨模型临床实用性。**

- **链接: [http://arxiv.org/pdf/2508.13947v1](http://arxiv.org/pdf/2508.13947v1)**

> **作者:** Yiqun Lin; Haoran Sun; Yongqing Li; Rabia Aslam; Lung Fung Tse; Tiange Cheng; Chun Sing Chui; Wing Fung Yau; Victorine R. Le Meur; Meruyert Amangeldy; Kiho Cho; Yinyu Ye; James Zou; Wei Zhao; Xiaomeng Li
>
> **摘要:** Patient-specific bone models are essential for designing surgical guides and preoperative planning, as they enable the visualization of intricate anatomical structures. However, traditional CT-based approaches for creating bone models are limited to preoperative use due to the low flexibility and high radiation exposure of CT and time-consuming manual delineation. Here, we introduce Semi-Supervised Reconstruction with Knowledge Distillation (SSR-KD), a fast and accurate AI framework to reconstruct high-quality bone models from biplanar X-rays in 30 seconds, with an average error under 1.0 mm, eliminating the dependence on CT and manual work. Additionally, high tibial osteotomy simulation was performed by experts on reconstructed bone models, demonstrating that bone models reconstructed from biplanar X-rays have comparable clinical applicability to those annotated from CT. Overall, our approach accelerates the process, reduces radiation exposure, enables intraoperative guidance, and significantly improves the practicality of bone models, offering transformative applications in orthopedics.
>
---
#### [new 090] MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出MM-BrowseComp，一个评估AI代理多模态网页浏览能力的基准。针对现有文本主导基准忽略图像、视频等多模态信息的问题，设计了224个手写挑战题，要求模型融合图文信息进行推理。实验表明当前顶尖模型在该任务上表现不佳，凸显多模态能力不足。**

- **链接: [http://arxiv.org/pdf/2508.13186v1](http://arxiv.org/pdf/2508.13186v1)**

> **作者:** Shilong Li; Xingyuan Bu; Wenjie Wang; Jiaheng Liu; Jun Dong; Haoyang He; Hao Lu; Haozhe Zhang; Chenchen Jing; Zhen Li; Chuanhao Li; Jiayi Tian; Chenchen Zhang; Tianhao Peng; Yancheng He; Jihao Gu; Yuanxing Zhang; Jian Yang; Ge Zhang; Wenhao Huang; Wangchunshu Zhou; Zhaoxiang Zhang; Ruizhe Ding; Shilei Wen
>
> **备注:** The first two authors contribute equally, 26 pages, repo at https://github.com/MMBrowseComp/MM-BrowseComp
>
> **摘要:** AI agents with advanced reasoning and tool use capabilities have demonstrated impressive performance in web browsing for deep search. While existing benchmarks such as BrowseComp evaluate these browsing abilities, they primarily focus on textual information, overlooking the prevalence of multimodal content. To bridge this gap, we introduce MM-BrowseComp, a novel benchmark comprising 224 challenging, hand-crafted questions specifically designed to assess agents' multimodal retrieval and reasoning capabilities. These questions often incorporate images in prompts, and crucial information encountered during the search and reasoning process may also be embedded within images or videos on webpages. Consequently, methods relying solely on text prove insufficient for our benchmark. Additionally, we provide a verified checklist for each question, enabling fine-grained analysis of multimodal dependencies and reasoning paths. Our comprehensive evaluation of state-of-the-art models on MM-BrowseComp reveals that even top models like OpenAI o3 with tools achieve only 29.02\% accuracy, highlighting the suboptimal multimodal capabilities and lack of native multimodal reasoning in current models.
>
---
#### [new 091] Multimodal Data Storage and Retrieval for Embodied AI: A Survey
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于Embodied AI的数据管理任务，旨在解决多模态数据存储与检索难题。通过系统分析五种存储架构和五种检索范式，识别出物理接地缺失、跨模态整合等瓶颈，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2508.13901v1](http://arxiv.org/pdf/2508.13901v1)**

> **作者:** Yihao Lu; Hao Tang
>
> **摘要:** Embodied AI (EAI) agents continuously interact with the physical world, generating vast, heterogeneous multimodal data streams that traditional management systems are ill-equipped to handle. In this survey, we first systematically evaluate five storage architectures (Graph Databases, Multi-Model Databases, Data Lakes, Vector Databases, and Time-Series Databases), focusing on their suitability for addressing EAI's core requirements, including physical grounding, low-latency access, and dynamic scalability. We then analyze five retrieval paradigms (Fusion Strategy-Based Retrieval, Representation Alignment-Based Retrieval, Graph-Structure-Based Retrieval, Generation Model-Based Retrieval, and Efficient Retrieval-Based Optimization), revealing a fundamental tension between achieving long-term semantic coherence and maintaining real-time responsiveness. Based on this comprehensive analysis, we identify key bottlenecks, spanning from the foundational Physical Grounding Gap to systemic challenges in cross-modal integration, dynamic adaptation, and open-world generalization. Finally, we outline a forward-looking research agenda encompassing physics-aware data models, adaptive storage-retrieval co-optimization, and standardized benchmarking, to guide future research toward principled data management solutions for EAI. Our survey is based on a comprehensive review of more than 180 related studies, providing a rigorous roadmap for designing the robust, high-performance data management frameworks essential for the next generation of autonomous embodied systems.
>
---
#### [new 092] A Surveillance Based Interactive Robot
- **分类: cs.RO; cs.AI; cs.CV; I.2.9; I.2.10; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.13319v1](http://arxiv.org/pdf/2508.13319v1)**

> **作者:** Kshitij Kavimandan; Pooja Mangal; Devanshi Mehta
>
> **备注:** 4 pages, 5 figures
>
> **摘要:** We build a mobile surveillance robot that streams video in real time and responds to speech so a user can monitor and steer it from a phone or browser. The system uses two Raspberry Pi 4 units: a front unit on a differential drive base with camera, mic, and speaker, and a central unit that serves the live feed and runs perception. Video is sent with FFmpeg. Objects in the scene are detected using YOLOv3 to support navigation and event awareness. For voice interaction, we use Python libraries for speech recognition, multilingual translation, and text-to-speech, so the robot can take spoken commands and read back responses in the requested language. A Kinect RGB-D sensor provides visual input and obstacle cues. In indoor tests the robot detects common objects at interactive frame rates on CPU, recognises commands reliably, and translates them to actions without manual control. The design relies on off-the-shelf hardware and open software, making it easy to reproduce. We discuss limits and practical extensions, including sensor fusion with ultrasonic range data, GPU acceleration, and adding face and text recognition.
>
---
#### [new 093] Model-based Multi-object Visual Tracking: Identification and Standard Model Limitations
- **分类: eess.SY; cs.CV; cs.SY**

- **简介: 论文研究多目标视觉跟踪任务，针对行人跟踪问题，采用雷达中的点目标模型（SPO）与PMBM滤波器。通过分析MOT-17数据集识别模型局限性，并提出改进方向以提升模型精度。**

- **链接: [http://arxiv.org/pdf/2508.13647v1](http://arxiv.org/pdf/2508.13647v1)**

> **作者:** Jan Krejčí; Oliver Kost; Yuxuan Xia; Lennart Svensson; Ondřej Straka
>
> **备注:** Submitted to FUSION 2025 conference
>
> **摘要:** This paper uses multi-object tracking methods known from the radar tracking community to address the problem of pedestrian tracking using 2D bounding box detections. The standard point-object (SPO) model is adopted, and the posterior density is computed using the Poisson multi-Bernoulli mixture (PMBM) filter. The selection of the model parameters rooted in continuous time is discussed, including the birth and survival probabilities. Some parameters are selected from the first principles, while others are identified from the data, which is, in this case, the publicly available MOT-17 dataset. Although the resulting PMBM algorithm yields promising results, a mismatch between the SPO model and the data is revealed. The model-based approach assumes that modifying the problematic components causing the SPO model-data mismatch will lead to better model-based algorithms in future developments.
>
---
#### [new 094] MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出MimicFunc框架，通过功能对应关系从单个RGB-D人类视频中模仿工具操作，解决机器人在几何差异下难以泛化工具操作的问题。该方法利用关键点抽象构建功能坐标系，实现一次学习即可适配新工具，无需繁琐的遥操作数据收集。**

- **链接: [http://arxiv.org/pdf/2508.13534v1](http://arxiv.org/pdf/2508.13534v1)**

> **作者:** Chao Tang; Anxing Xiao; Yuhong Deng; Tianrun Hu; Wenlong Dong; Hanbo Zhang; David Hsu; Hong Zhang
>
> **备注:** Accepted to CoRL 2025
>
> **摘要:** Imitating tool manipulation from human videos offers an intuitive approach to teaching robots, while also providing a promising and scalable alternative to labor-intensive teleoperation data collection for visuomotor policy learning. While humans can mimic tool manipulation behavior by observing others perform a task just once and effortlessly transfer the skill to diverse tools for functionally equivalent tasks, current robots struggle to achieve this level of generalization. A key challenge lies in establishing function-level correspondences, considering the significant geometric variations among functionally similar tools, referred to as intra-function variations. To address this challenge, we propose MimicFunc, a framework that establishes functional correspondences with function frame, a function-centric local coordinate frame constructed with keypoint-based abstraction, for imitating tool manipulation skills. Experiments demonstrate that MimicFunc effectively enables the robot to generalize the skill from a single RGB-D human video to manipulating novel tools for functionally equivalent tasks. Furthermore, leveraging MimicFunc's one-shot generalization capability, the generated rollouts can be used to train visuomotor policies without requiring labor-intensive teleoperation data collection for novel objects. Our code and video are available at https://sites.google.com/view/mimicfunc.
>
---
#### [new 095] UNICON: UNIfied CONtinual Learning for Medical Foundational Models
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14024v1](http://arxiv.org/pdf/2508.14024v1)**

> **作者:** Mohammad Areeb Qazi; Munachiso S Nwadike; Ibrahim Almakky; Mohammad Yaqub; Numan Saeed
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** Foundational models are trained on extensive datasets to capture the general trends of a domain. However, in medical imaging, the scarcity of data makes pre-training for every domain, modality, or task challenging. Continual learning offers a solution by fine-tuning a model sequentially on different domains or tasks, enabling it to integrate new knowledge without requiring large datasets for each training phase. In this paper, we propose UNIfied CONtinual Learning for Medical Foundational Models (UNICON), a framework that enables the seamless adaptation of foundation models to diverse domains, tasks, and modalities. Unlike conventional adaptation methods that treat these changes in isolation, UNICON provides a unified, perpetually expandable framework. Through careful integration, we show that foundation models can dynamically expand across imaging modalities, anatomical regions, and clinical objectives without catastrophic forgetting or task interference. Empirically, we validate our approach by adapting a chest CT foundation model initially trained for classification to a prognosis and segmentation task. Our results show improved performance across both additional tasks. Furthermore, we continually incorporated PET scans and achieved a 5\% improvement in Dice score compared to respective baselines. These findings establish that foundation models are not inherently constrained to their initial training scope but can evolve, paving the way toward generalist AI models for medical imaging.
>
---
#### [new 096] Automated Cervical Cancer Detection through Visual Inspection with Acetic Acid in Resource-Poor Settings with Lightweight Deep Learning Models Deployed on an Android Device
- **分类: eess.IV; cs.CV; cs.LG; 68T07, 92C55, 68T45; I.4.9; J.3; I.2.10; I.2.6**

- **简介: 论文提出轻量级深度学习模型，用于在资源匮乏地区自动检测宫颈癌。任务为基于醋酸视觉检查（VIA）的自动化分析，解决人工判读主观性强、专业人员短缺问题。工作包括设计EfficientDet-Lite3与MobileNet-V2模型并部署于安卓设备，实现高精度即时筛查。**

- **链接: [http://arxiv.org/pdf/2508.13253v1](http://arxiv.org/pdf/2508.13253v1)**

> **作者:** Leander Melroy Maben; Keerthana Prasad; Shyamala Guruvare; Vidya Kudva; P C Siddalingaswamy
>
> **摘要:** Cervical cancer is among the most commonly occurring cancer among women and claims a huge number of lives in low and middle-income countries despite being relatively easy to treat. Several studies have shown that public screening programs can bring down cervical cancer incidence and mortality rates significantly. While several screening tests are available, visual inspection with acetic acid (VIA) presents itself as the most viable option for low-resource settings due to the affordability and simplicity of performing the test. VIA requires a trained medical professional to interpret the test and is subjective in nature. Automating VIA using AI eliminates subjectivity and would allow shifting of the task to less trained health workers. Task shifting with AI would help further expedite screening programs in low-resource settings. In our work, we propose a lightweight deep learning algorithm that includes EfficientDet-Lite3 as the Region of Interest (ROI) detector and a MobileNet- V2 based model for classification. These models would be deployed on an android-based device that can operate remotely and provide almost instant results without the requirement of highly-trained medical professionals, labs, sophisticated infrastructure, or internet connectivity. The classification model gives an accuracy of 92.31%, a sensitivity of 98.24%, and a specificity of 88.37% on the test dataset and presents itself as a promising automated low-resource screening approach.
>
---
#### [new 097] PreSem-Surf: RGB-D Surface Reconstruction with Progressive Semantic Modeling and SG-MLP Pre-Rendering Mechanism
- **分类: cs.GR; cs.AI; cs.CV; eess.IV**

- **简介: 论文提出PreSem-Surf，用于RGB-D序列的高质量表面重建任务。针对传统方法重建慢、细节易受噪声干扰的问题，引入SG-MLP预渲染与渐进语义建模，提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2508.13228v1](http://arxiv.org/pdf/2508.13228v1)**

> **作者:** Yuyan Ye; Hang Xu; Yanghang Huang; Jiali Huang; Qian Weng
>
> **备注:** 2025 International Joint Conference on Neural Networks (IJCNN 2025)
>
> **摘要:** This paper proposes PreSem-Surf, an optimized method based on the Neural Radiance Field (NeRF) framework, capable of reconstructing high-quality scene surfaces from RGB-D sequences in a short time. The method integrates RGB, depth, and semantic information to improve reconstruction performance. Specifically, a novel SG-MLP sampling structure combined with PR-MLP (Preconditioning Multilayer Perceptron) is introduced for voxel pre-rendering, allowing the model to capture scene-related information earlier and better distinguish noise from local details. Furthermore, progressive semantic modeling is adopted to extract semantic information at increasing levels of precision, reducing training time while enhancing scene understanding. Experiments on seven synthetic scenes with six evaluation metrics show that PreSem-Surf achieves the best performance in C-L1, F-score, and IoU, while maintaining competitive results in NC, Accuracy, and Completeness, demonstrating its effectiveness and practical applicability.
>
---
#### [new 098] Sketch3DVE: Sketch-based 3D-Aware Scene Video Editing
- **分类: cs.GR; cs.CV**

- **简介: 论文提出Sketch3DVE，解决视频中3D场景结构编辑难题，尤其在视角变化大时保持一致性。通过草图控制几何、点云编辑与3D-aware掩码传播，实现精确局部编辑并生成逼真结果。**

- **链接: [http://arxiv.org/pdf/2508.13797v1](http://arxiv.org/pdf/2508.13797v1)**

> **作者:** Feng-Lin Liu; Shi-Yang Li; Yan-Pei Cao; Hongbo Fu; Lin Gao
>
> **备注:** SIGGRAPH 2025
>
> **摘要:** Recent video editing methods achieve attractive results in style transfer or appearance modification. However, editing the structural content of 3D scenes in videos remains challenging, particularly when dealing with significant viewpoint changes, such as large camera rotations or zooms. Key challenges include generating novel view content that remains consistent with the original video, preserving unedited regions, and translating sparse 2D inputs into realistic 3D video outputs. To address these issues, we propose Sketch3DVE, a sketch-based 3D-aware video editing method to enable detailed local manipulation of videos with significant viewpoint changes. To solve the challenge posed by sparse inputs, we employ image editing methods to generate edited results for the first frame, which are then propagated to the remaining frames of the video. We utilize sketching as an interaction tool for precise geometry control, while other mask-based image editing methods are also supported. To handle viewpoint changes, we perform a detailed analysis and manipulation of the 3D information in the video. Specifically, we utilize a dense stereo method to estimate a point cloud and the camera parameters of the input video. We then propose a point cloud editing approach that uses depth maps to represent the 3D geometry of newly edited components, aligning them effectively with the original 3D scene. To seamlessly merge the newly edited content with the original video while preserving the features of unedited regions, we introduce a 3D-aware mask propagation strategy and employ a video diffusion model to produce realistic edited videos. Extensive experiments demonstrate the superiority of Sketch3DVE in video editing. Homepage and code: http://http://geometrylearning.com/Sketch3DVE/
>
---
#### [new 099] RISE: Enhancing VLM Image Annotation with Self-Supervised Reasoning
- **分类: cs.LG; cs.CV**

- **简介: 论文提出RISE框架，解决VLM在复杂图像标注中推理不足的问题。通过自监督闭环生成高质量推理链，并结合监督与强化微调，提升标注准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2508.13229v1](http://arxiv.org/pdf/2508.13229v1)**

> **作者:** Suhang Hu; Wei Hu; Yuhang Su; Fan Zhang
>
> **摘要:** Vision-Language Models (VLMs) struggle with complex image annotation tasks, such as emotion classification and context-driven object detection, which demand sophisticated reasoning. Standard Supervised Fine-Tuning (SFT) focuses solely on annotation outcomes, ignoring underlying rationales, while Visual Reinforcement Fine-Tuning (Visual-RFT) produces inconsistent Chains of Thought (CoTs) due to the absence of high-quality, verified CoTs during pre-training. We introduce RISE (Reason-Inspire-Strengthen-Expertise), a two-stage framework to overcome these limitations. In the Reason stage (RISE-CoT), a reinforcement learning-driven "annotation-reasoning-annotation" closed-loop generates visually grounded, logically consistent CoTs by verifying their ability to reconstruct original annotations without direct leakage. The Inspire and Strengthen stage (RISE-R1) leverages a high-quality CoT subset, filtered by RISE-CoT rewards, for supervised fine-tuning, followed by reinforcement fine-tuning to produce interpretable reasoning and accurate annotations, achieving Expertise in complex visual tasks. Evaluated on complex and simple image annotation tasks, RISE-trained Qwen2-VL-2B outperforms SFT and Visual-RFT, achieving robust performance and enhanced explainability. RISE offers a self-supervised solution for advancing VLM reasoning without requiring manually annotated CoTs.
>
---
#### [new 100] Is-NeRF: In-scattering Neural Radiance Field for Blurred Images
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13808v1](http://arxiv.org/pdf/2508.13808v1)**

> **作者:** Nan Luo; Chenglin Ye; Jiaxu Li; Gang Liu; Bo Wan; Di Wang; Lupeng Liu; Jun Xiao
>
> **摘要:** Neural Radiance Fields (NeRF) has gained significant attention for its prominent implicit 3D representation and realistic novel view synthesis capabilities. Available works unexceptionally employ straight-line volume rendering, which struggles to handle sophisticated lightpath scenarios and introduces geometric ambiguities during training, particularly evident when processing motion-blurred images. To address these challenges, this work proposes a novel deblur neural radiance field, Is-NeRF, featuring explicit lightpath modeling in real-world environments. By unifying six common light propagation phenomena through an in-scattering representation, we establish a new scattering-aware volume rendering pipeline adaptable to complex lightpaths. Additionally, we introduce an adaptive learning strategy that enables autonomous determining of scattering directions and sampling intervals to capture finer object details. The proposed network jointly optimizes NeRF parameters, scattering parameters, and camera motions to recover fine-grained scene representations from blurry images. Comprehensive evaluations demonstrate that it effectively handles complex real-world scenarios, outperforming state-of-the-art approaches in generating high-fidelity images with accurate geometric details.
>
---
#### [new 101] subCellSAM: Zero-Shot (Sub-)Cellular Segmentation for Hit Validation in Drug Discovery
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出subCellSAM，用于药物发现中的细胞及亚细胞结构分割任务。针对传统方法需大量调参或微调的问题，作者利用零样本分割和上下文学习策略，通过自提示机制实现无需特定数据集调整的精准分割，显著提升hit验证效率。**

- **链接: [http://arxiv.org/pdf/2508.13701v1](http://arxiv.org/pdf/2508.13701v1)**

> **作者:** Jacob Hanimann; Daniel Siegismund; Mario Wieser; Stephan Steigele
>
> **备注:** Accepted at DAGM German Conference on Pattern Recognition (GCPR) 2025
>
> **摘要:** High-throughput screening using automated microscopes is a key driver in biopharma drug discovery, enabling the parallel evaluation of thousands of drug candidates for diseases such as cancer. Traditional image analysis and deep learning approaches have been employed to analyze these complex, large-scale datasets, with cell segmentation serving as a critical step for extracting relevant structures. However, both strategies typically require extensive manual parameter tuning or domain-specific model fine-tuning. We present a novel method that applies a segmentation foundation model in a zero-shot setting (i.e., without fine-tuning), guided by an in-context learning strategy. Our approach employs a three-step process for nuclei, cell, and subcellular segmentation, introducing a self-prompting mechanism that encodes morphological and topological priors using growing masks and strategically placed foreground/background points. We validate our method on both standard cell segmentation benchmarks and industry-relevant hit validation assays, demonstrating that it accurately segments biologically relevant structures without the need for dataset-specific tuning.
>
---
#### [new 102] Image2Net: Datasets, Benchmark and Hybrid Framework to Convert Analog Circuit Diagrams into Netlists
- **分类: cs.AR; cs.AI; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.13157v1](http://arxiv.org/pdf/2508.13157v1)**

> **作者:** Haohang Xu; Chengjie Liu; Qihang Wang; Wenhao Huang; Yongjian Xu; Weiyu Chen; Anlan Peng; Zhijun Li; Bo Li; Lei Qi; Jun Yang; Yuan Du; Li Du
>
> **备注:** 10 pages, 12 figures, 6 tables
>
> **摘要:** Large Language Model (LLM) exhibits great potential in designing of analog integrated circuits (IC) because of its excellence in abstraction and generalization for knowledge. However, further development of LLM-based analog ICs heavily relies on textual description of analog ICs, while existing analog ICs are mostly illustrated in image-based circuit diagrams rather than text-based netlists. Converting circuit diagrams to netlists help LLMs to enrich the knowledge of analog IC. Nevertheless, previously proposed conversion frameworks face challenges in further application because of limited support of image styles and circuit elements. Up to now, it still remains a challenging task to effectively convert complex circuit diagrams into netlists. To this end, this paper constructs and opensources a new dataset with rich styles of circuit diagrams as well as balanced distribution of simple and complex analog ICs. And a hybrid framework, named Image2Net, is proposed for practical conversion from circuit diagrams to netlists. The netlist edit distance (NED) is also introduced to precisely assess the difference between the converted netlists and ground truth. Based on our benchmark, Image2Net achieves 80.77\% successful rate, which is 34.62\%-45.19\% higher than previous works. Specifically, the proposed work shows 0.116 averaged NED, which is 62.1\%-69.6\% lower than state-of-the-arts.
>
---
#### [new 103] Latent Interpolation Learning Using Diffusion Models for Cardiac Volume Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对心脏MRI中稀疏切片导致的三维体积重建难题，提出CaLID框架。通过扩散模型实现数据驱动的非线性插值，高效重构完整心脏体积，无需额外标注，提升精度与速度，适用于时空动态建模。**

- **链接: [http://arxiv.org/pdf/2508.13826v1](http://arxiv.org/pdf/2508.13826v1)**

> **作者:** Niklas Bubeck; Suprosanna Shit; Chen Chen; Can Zhao; Pengfei Guo; Dong Yang; Georg Zitzlsberger; Daguang Xu; Bernhard Kainz; Daniel Rueckert; Jiazhen Pan
>
> **摘要:** Cardiac Magnetic Resonance (CMR) imaging is a critical tool for diagnosing and managing cardiovascular disease, yet its utility is often limited by the sparse acquisition of 2D short-axis slices, resulting in incomplete volumetric information. Accurate 3D reconstruction from these sparse slices is essential for comprehensive cardiac assessment, but existing methods face challenges, including reliance on predefined interpolation schemes (e.g., linear or spherical), computational inefficiency, and dependence on additional semantic inputs such as segmentation labels or motion data. To address these limitations, we propose a novel \textbf{Ca}rdiac \textbf{L}atent \textbf{I}nterpolation \textbf{D}iffusion (CaLID) framework that introduces three key innovations. First, we present a data-driven interpolation scheme based on diffusion models, which can capture complex, non-linear relationships between sparse slices and improves reconstruction accuracy. Second, we design a computationally efficient method that operates in the latent space and speeds up 3D whole-heart upsampling time by a factor of 24, reducing computational overhead compared to previous methods. Third, with only sparse 2D CMR images as input, our method achieves SOTA performance against baseline methods, eliminating the need for auxiliary input such as morphological guidance, thus simplifying workflows. We further extend our method to 2D+T data, enabling the effective modeling of spatiotemporal dynamics and ensuring temporal coherence. Extensive volumetric evaluations and downstream segmentation tasks demonstrate that CaLID achieves superior reconstruction quality and efficiency. By addressing the fundamental limitations of existing approaches, our framework advances the state of the art for spatio and spatiotemporal whole-heart reconstruction, offering a robust and clinically practical solution for cardiovascular imaging.
>
---
#### [new 104] Learning to See Through Flare
- **分类: eess.IV; cs.CV**

- **简介: 论文提出NeuSee框架，解决激光耀斑导致视觉系统失效的问题。通过联合学习衍射光学元件与频率域Mamba-GAN网络，在全可见光谱下实现高保真图像恢复，显著提升抗激光干扰能力。**

- **链接: [http://arxiv.org/pdf/2508.13907v1](http://arxiv.org/pdf/2508.13907v1)**

> **作者:** Xiaopeng Peng; Heath Gemar; Erin Fleet; Kyle Novak; Abbie Watnik; Grover Swartzlander
>
> **备注:** accepted by ICCVW 2025
>
> **摘要:** Machine vision systems are susceptible to laser flare, where unwanted intense laser illumination blinds and distorts its perception of the environment through oversaturation or permanent damage to sensor pixels. We introduce NeuSee, the first computational imaging framework for high-fidelity sensor protection across the full visible spectrum. It jointly learns a neural representation of a diffractive optical element (DOE) and a frequency-space Mamba-GAN network for image restoration. NeuSee system is adversarially trained end-to-end on 100K unique images to suppress the peak laser irradiance as high as $10^6$ times the sensor saturation threshold $I_{\textrm{sat}}$, the point at which camera sensors may experience damage without the DOE. Our system leverages heterogeneous data and model parallelism for distributed computing, integrating hyperspectral information and multiple neural networks for realistic simulation and image restoration. NeuSee takes into account open-world scenes with dynamically varying laser wavelengths, intensities, and positions, as well as lens flare effects, unknown ambient lighting conditions, and sensor noises. It outperforms other learned DOEs, achieving full-spectrum imaging and laser suppression for the first time, with a 10.1\% improvement in restored image quality.
>
---
#### [new 105] Benchmarking GPT-5 for Zero-Shot Multimodal Medical Reasoning in Radiology and Radiation Oncology
- **分类: eess.IV; cs.CV**

- **简介: 该论文评估GPT-5在放射学和放疗领域的零样本多模态医学推理能力，解决大模型在高风险医疗场景中性能提升问题。通过三项任务对比GPT-5与GPT-4o，结果表明GPT-5在图像理解与物理计算上显著优于前者，尤其在胸部、肺部和脑部区域表现突出。**

- **链接: [http://arxiv.org/pdf/2508.13192v1](http://arxiv.org/pdf/2508.13192v1)**

> **作者:** Mingzhe Hu; Zach Eidex; Shansong Wang; Mojtaba Safari; Qiang Li; Xiaofeng Yang
>
> **摘要:** Radiology, radiation oncology, and medical physics require decision-making that integrates medical images, textual reports, and quantitative data under high-stakes conditions. With the introduction of GPT-5, it is critical to assess whether recent advances in large multimodal models translate into measurable gains in these safety-critical domains. We present a targeted zero-shot evaluation of GPT-5 and its smaller variants (GPT-5-mini, GPT-5-nano) against GPT-4o across three representative tasks. We present a targeted zero-shot evaluation of GPT-5 and its smaller variants (GPT-5-mini, GPT-5-nano) against GPT-4o across three representative tasks: (1) VQA-RAD, a benchmark for visual question answering in radiology; (2) SLAKE, a semantically annotated, multilingual VQA dataset testing cross-modal grounding; and (3) a curated Medical Physics Board Examination-style dataset of 150 multiple-choice questions spanning treatment planning, dosimetry, imaging, and quality assurance. Across all datasets, GPT-5 achieved the highest accuracy, with substantial gains over GPT-4o up to +20.00% in challenging anatomical regions such as the chest-mediastinal, +13.60% in lung-focused questions, and +11.44% in brain-tissue interpretation. On the board-style physics questions, GPT-5 attained 90.7% accuracy (136/150), exceeding the estimated human passing threshold, while GPT-4o trailed at 78.0%. These results demonstrate that GPT-5 delivers consistent and often pronounced performance improvements over GPT-4o in both image-grounded reasoning and domain-specific numerical problem-solving, highlighting its potential to augment expert workflows in medical imaging and therapeutic physics.
>
---
#### [new 106] Hierarchy-Consistent Learning and Adaptive Loss Balancing for Hierarchical Multi-Label Classification
- **分类: cs.LG; cs.CV**

- **简介: 论文提出HCAL模型解决层次多标签分类中的结构一致性与损失权重不平衡问题，通过原型对比学习和自适应损失加权机制提升准确率并降低层级违反率。**

- **链接: [http://arxiv.org/pdf/2508.13452v1](http://arxiv.org/pdf/2508.13452v1)**

> **作者:** Ruobing Jiang; Mengzhe Liu; Haobing Liu; Yanwei Yu
>
> **备注:** 10 pages, 7 figures, accepted by CIKM 2025
>
> **摘要:** Hierarchical Multi-Label Classification (HMC) faces critical challenges in maintaining structural consistency and balancing loss weighting in Multi-Task Learning (MTL). In order to address these issues, we propose a classifier called HCAL based on MTL integrated with prototype contrastive learning and adaptive task-weighting mechanisms. The most significant advantage of our classifier is semantic consistency including both prototype with explicitly modeling label and feature aggregation from child classes to parent classes. The other important advantage is an adaptive loss-weighting mechanism that dynamically allocates optimization resources by monitoring task-specific convergence rates. It effectively resolves the "one-strong-many-weak" optimization bias inherent in traditional MTL approaches. To further enhance robustness, a prototype perturbation mechanism is formulated by injecting controlled noise into prototype to expand decision boundaries. Additionally, we formalize a quantitative metric called Hierarchical Violation Rate (HVR) as to evaluate hierarchical consistency and generalization. Extensive experiments across three datasets demonstrate both the higher classification accuracy and reduced hierarchical violation rate of the proposed classifier over baseline models.
>
---
#### [new 107] Colon Polyps Detection from Colonoscopy Images Using Deep Learning
- **分类: eess.IV; cs.CV**

- **简介: 论文属于医学图像目标检测任务，旨在提升结肠息肉早期识别准确率。通过改进YOLOv5模型并利用Kvasir-SEG数据集训练，YOLOv5l表现最优，mAP达85.1%，为结直肠癌筛查提供有效工具。**

- **链接: [http://arxiv.org/pdf/2508.13188v1](http://arxiv.org/pdf/2508.13188v1)**

> **作者:** Md Al Amin; Bikash Kumar Paul
>
> **备注:** 17 Pages
>
> **摘要:** Colon polyps are precursors to colorectal cancer, a leading cause of cancer-related mortality worldwide. Early detection is critical for improving patient outcomes. This study investigates the application of deep learning-based object detection for early polyp identification using colonoscopy images. We utilize the Kvasir-SEG dataset, applying extensive data augmentation and splitting the data into training (80\%), validation (20\% of training), and testing (20\%) sets. Three variants of the YOLOv5 architecture (YOLOv5s, YOLOv5m, YOLOv5l) are evaluated. Experimental results show that YOLOv5l outperforms the other variants, achieving a mean average precision (mAP) of 85.1\%, with the highest average Intersection over Union (IoU) of 0.86. These findings demonstrate that YOLOv5l provides superior detection performance for colon polyp localization, offering a promising tool for enhancing colorectal cancer screening accuracy.
>
---
#### [new 108] Augmenting cobots for sheet-metal SMEs with 3D object recognition and localisation
- **分类: cs.RO; cs.CV**

- **简介: 论文探讨如何通过3D物体识别与定位技术增强协作机器人（cobots）能力，解决中小型企业钣金车间高混合低批量生产中的自动化难题，提升生产效率并优化技工使用。**

- **链接: [http://arxiv.org/pdf/2508.13964v1](http://arxiv.org/pdf/2508.13964v1)**

> **作者:** Martijn Cramer; Yanming Wu; David De Schepper; Eric Demeester
>
> **备注:** 13 pages, 25 figures
>
> **摘要:** Due to high-mix-low-volume production, sheet-metal workshops today are challenged by small series and varying orders. As standard automation solutions tend to fall short, SMEs resort to repetitive manual labour impacting production costs and leading to tech-skilled workforces not being used to their full potential. The COOCK+ ROBUST project aims to transform cobots into mobile and reconfigurable production assistants by integrating existing technologies, including 3D object recognition and localisation. This article explores both the opportunities and challenges of enhancing cobotic systems with these technologies in an industrial setting, outlining the key steps involved in the process. Additionally, insights from a past project, carried out by the ACRO research unit in collaboration with an industrial partner, serves as a concrete implementation example throughout.
>
---
#### [new 109] MME-SCI: A Comprehensive and Challenging Science Benchmark for Multimodal Large Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出MME-SCI基准，用于评估多模态大模型在科学领域的推理能力。针对现有基准在多语言、多模态覆盖和知识粒度上的不足，构建了包含4学科、5语言的1019个QA对，并测试了16个开源和4个闭源模型，揭示了当前模型在各领域的能力短板。**

- **链接: [http://arxiv.org/pdf/2508.13938v1](http://arxiv.org/pdf/2508.13938v1)**

> **作者:** Jiacheng Ruan; Dan Jiang; Xian Gao; Ting Liu; Yuzhuo Fu; Yangyang Kang
>
> **备注:** 9 pages, 6 figures, work in progress
>
> **摘要:** Recently, multimodal large language models (MLLMs) have achieved significant advancements across various domains, and corresponding evaluation benchmarks have been continuously refined and improved. In this process, benchmarks in the scientific domain have played an important role in assessing the reasoning capabilities of MLLMs. However, existing benchmarks still face three key challenges: 1) Insufficient evaluation of models' reasoning abilities in multilingual scenarios; 2) Inadequate assessment of MLLMs' comprehensive modality coverage; 3) Lack of fine-grained annotation of scientific knowledge points. To address these gaps, we propose MME-SCI, a comprehensive and challenging benchmark. We carefully collected 1,019 high-quality question-answer pairs, which involve 3 distinct evaluation modes. These pairs cover four subjects, namely mathematics, physics, chemistry, and biology, and support five languages: Chinese, English, French, Spanish, and Japanese. We conducted extensive experiments on 16 open-source models and 4 closed-source models, and the results demonstrate that MME-SCI is widely challenging for existing MLLMs. For instance, under the Image-only evaluation mode, o4-mini achieved accuracy of only 52.11%, 24.73%, 36.57%, and 29.80% in mathematics, physics, chemistry, and biology, respectively, indicating a significantly higher difficulty level compared to existing benchmarks. More importantly, using MME-SCI's multilingual and fine-grained knowledge attributes, we analyzed existing models' performance in depth and identified their weaknesses in specific domains. The Data and Evaluation Code are available at https://github.com/JCruan519/MME-SCI.
>
---
#### [new 110] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation
- **分类: cs.AI; cs.CV**

- **简介: 论文针对图表转代码生成任务，解决监督微调性能瓶颈问题。提出多粒度结构化强化学习方法MSRL，结合文本与视觉奖励机制，显著提升生成代码的准确性与结构合理性。**

- **链接: [http://arxiv.org/pdf/2508.13587v1](http://arxiv.org/pdf/2508.13587v1)**

> **作者:** Lei Chen; Xuanle Zhao; Zhixiong Zeng; Jing Huang; Liming Zheng; Yufeng Zhong; Lin Ma
>
> **备注:** technical report
>
> **摘要:** While reinforcement learning (RL) has proven highly effective for general reasoning in vision-language models, its application to tasks requiring in-depth understanding of information-rich images and generation of structured outputs remains underexplored. Chart-to-code generation exemplifies this challenge, demanding complex reasoning over visual charts to generate structured code. Supervised fine-tuning (SFT) alone is often insufficient, highlighting the need for effective RL strategies that appropriately reward structured outputs. We systematically investigate the performance plateau in SFT through large-scale experiments and propose Multimodal Structured Reinforcement Learning (MSRL) for chart-to-code generation, which substantially breaks through this plateau. We construct the largest training corpus to date, containing 3 million chart-code pairs from real-world arXiv tables to mitigate simplistic patterns of prior synthetic data. Despite reaching state-of-the-art performance, our experiments show that scaling SFT data eventually hits a plateau where further increases yield negligible improvements. Our MSRL method leverages a multi-granularity structured reward system using multimodal textual and visual feedback. At the textual level, rule-based rewards validate fine-grained code details. At the visual level, model-based rewards assess structural similarity by rendering generated code into images and employing an evaluator model. We implement this within a two-stage curriculum for training stability. Results demonstrate that MSRL significantly breaks the SFT plateau, improving high-level metrics by 6.2% and 9.9% on ChartMimic and ReachQA benchmarks respectively, achieving competitive performance with advanced closed-source models.
>
---
#### [new 111] BERT-VQA: Visual Question Answering on Plots
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉问答任务中的子问题——图表问答。旨在解决模型如何理解图表内容并回答相关问题。作者提出BERT-VQA模型，基于VisualBERT架构，结合预训练ResNet图像编码器进行训练与评估，发现跨模态模块并非必需，揭示了该任务的挑战性及模型设计的局限性。**

- **链接: [http://arxiv.org/pdf/2508.13184v1](http://arxiv.org/pdf/2508.13184v1)**

> **作者:** Tai Vu; Robert Yang
>
> **摘要:** Visual question answering has been an exciting challenge in the field of natural language understanding, as it requires deep learning models to exchange information from both vision and language domains. In this project, we aim to tackle a subtask of this problem, namely visual question answering on plots. To achieve this, we developed BERT-VQA, a VisualBERT-based model architecture with a pretrained ResNet 101 image encoder, along with a potential addition of joint fusion. We trained and evaluated this model against a baseline that consisted of a LSTM, a CNN, and a shallow classifier. The final outcome disproved our core hypothesis that the cross-modality module in VisualBERT is essential in aligning plot components with question phrases. Therefore, our work provided valuable insights into the difficulty of the plot question answering challenge as well as the appropriateness of different model architectures in solving this problem.
>
---
#### [new 112] Deep Biomechanically-Guided Interpolation for Keypoint-Based Brain Shift Registration
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决神经外科中脑移位补偿问题。通过构建生物力学引导的深度学习框架，从稀疏关键点预测密集且物理合理的脑变形场，显著降低误差并保持高效推理。**

- **链接: [http://arxiv.org/pdf/2508.13762v1](http://arxiv.org/pdf/2508.13762v1)**

> **作者:** Tiago Assis; Ines P. Machado; Benjamin Zwick; Nuno C. Garcia; Reuben Dorent
>
> **备注:** Accepted at COLlaborative Intelligence and Autonomy in Image-guided Surgery (COLAS) Workshop - MICCAI 2025
>
> **摘要:** Accurate compensation of brain shift is critical for maintaining the reliability of neuronavigation during neurosurgery. While keypoint-based registration methods offer robustness to large deformations and topological changes, they typically rely on simple geometric interpolators that ignore tissue biomechanics to create dense displacement fields. In this work, we propose a novel deep learning framework that estimates dense, physically plausible brain deformations from sparse matched keypoints. We first generate a large dataset of synthetic brain deformations using biomechanical simulations. Then, a residual 3D U-Net is trained to refine standard interpolation estimates into biomechanically guided deformations. Experiments on a large set of simulated displacement fields demonstrate that our method significantly outperforms classical interpolators, reducing by half the mean square error while introducing negligible computational overhead at inference time. Code available at: \href{https://github.com/tiago-assis/Deep-Biomechanical-Interpolator}{https://github.com/tiago-assis/Deep-Biomechanical-Interpolator}.
>
---
#### [new 113] A Novel Attention-Augmented Wavelet YOLO System for Real-time Brain Vessel Segmentation on Transcranial Color-coded Doppler
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出AAW-YOLO模型，用于实时自动分割经颅彩色多普勒（TCCD）图像中的脑血管结构，解决人工操作依赖性强、效率低的问题。基于自建数据集，模型在精度和速度上表现优异，适用于临床筛查与资源受限环境。**

- **链接: [http://arxiv.org/pdf/2508.13875v1](http://arxiv.org/pdf/2508.13875v1)**

> **作者:** Wenxuan Zhang; Shuai Li; Xinyi Wang; Yu Sun; Hongyu Kang; Pui Yuk Chryste Wan; Yong-Ping Zheng; Sai-Kit Lam
>
> **摘要:** The Circle of Willis (CoW), vital for ensuring consistent blood flow to the brain, is closely linked to ischemic stroke. Accurate assessment of the CoW is important for identifying individuals at risk and guiding appropriate clinical management. Among existing imaging methods, Transcranial Color-coded Doppler (TCCD) offers unique advantages due to its radiation-free nature, affordability, and accessibility. However, reliable TCCD assessments depend heavily on operator expertise for identifying anatomical landmarks and performing accurate angle correction, which limits its widespread adoption. To address this challenge, we propose an AI-powered, real-time CoW auto-segmentation system capable of efficiently capturing cerebral arteries. No prior studies have explored AI-driven cerebrovascular segmentation using TCCD. In this work, we introduce a novel Attention-Augmented Wavelet YOLO (AAW-YOLO) network tailored for TCCD data, designed to provide real-time guidance for brain vessel segmentation in the CoW. We prospectively collected TCCD data comprising 738 annotated frames and 3,419 labeled artery instances to establish a high-quality dataset for model training and evaluation. The proposed AAW-YOLO demonstrated strong performance in segmenting both ipsilateral and contralateral CoW vessels, achieving an average Dice score of 0.901, IoU of 0.823, precision of 0.882, recall of 0.926, and mAP of 0.953, with a per-frame inference speed of 14.199 ms. This system offers a practical solution to reduce reliance on operator experience in TCCD-based cerebrovascular screening, with potential applications in routine clinical workflows and resource-constrained settings. Future research will explore bilateral modeling and larger-scale validation.
>
---
#### [new 114] A Comprehensive Re-Evaluation of Biometric Modality Properties in the Modern Era
- **分类: cs.LG; cs.CV**

- **简介: 论文属于生物特征识别评估任务，旨在解决传统评价框架过时的问题。通过专家调查与数据集不确定性分析，重新评估各类生物特征模态的属性，揭示技术进步与安全漏洞带来的变化，并为未来研究指明方向。**

- **链接: [http://arxiv.org/pdf/2508.13874v1](http://arxiv.org/pdf/2508.13874v1)**

> **作者:** Rouqaiah Al-Refai; Pankaja Priya Ramasamy; Ragini Ramesh; Patricia Arias-Cabarcos; Philipp Terhörst
>
> **摘要:** The rapid advancement of authentication systems and their increasing reliance on biometrics for faster and more accurate user verification experience, highlight the critical need for a reliable framework to evaluate the suitability of biometric modalities for specific applications. Currently, the most widely known evaluation framework is a comparative table from 1998, which no longer adequately captures recent technological developments or emerging vulnerabilities in biometric systems. To address these challenges, this work revisits the evaluation of biometric modalities through an expert survey involving 24 biometric specialists. The findings indicate substantial shifts in property ratings across modalities. For example, face recognition, shows improved ratings due to technological progress, while fingerprint, shows decreased reliability because of emerging vulnerabilities and attacks. Further analysis of expert agreement levels across rated properties highlighted the consistency of the provided evaluations and ensured the reliability of the ratings. Finally, expert assessments are compared with dataset-level uncertainty across 55 biometric datasets, revealing strong alignment in most modalities and underscoring the importance of integrating empirical evidence with expert insight. Moreover, the identified expert disagreements reveal key open challenges and help guide future research toward resolving them.
>
---
#### [new 115] InnerGS: Internal Scenes Rendering via Factorized 3D Gaussian Splatting
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出InnerGS，用于从稀疏切片数据中重建物体内部结构。任务为内部场景渲染，解决现有方法仅关注外部表面的问题。工作包括通过内嵌3D高斯分布建模连续体密度，无需相机位姿，兼容多种数据模态。**

- **链接: [http://arxiv.org/pdf/2508.13287v1](http://arxiv.org/pdf/2508.13287v1)**

> **作者:** Shuxin Liang; Yihan Xiao; Wenlu Tang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently gained popularity for efficient scene rendering by representing scenes as explicit sets of anisotropic 3D Gaussians. However, most existing work focuses primarily on modeling external surfaces. In this work, we target the reconstruction of internal scenes, which is crucial for applications that require a deep understanding of an object's interior. By directly modeling a continuous volumetric density through the inner 3D Gaussian distribution, our model effectively reconstructs smooth and detailed internal structures from sparse sliced data. Our approach eliminates the need for camera poses, is plug-and-play, and is inherently compatible with any data modalities. We provide cuda implementation at: https://github.com/Shuxin-Liang/InnerGS.
>
---
#### [new 116] Susceptibility Distortion Correction of Diffusion MRI with a single Phase-Encoding Direction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于扩散MRI图像处理任务，旨在解决单相位编码方向下 susceptibility 引起的几何和强度失真问题。作者提出一种基于深度学习的方法，仅用单次采集数据即可实现高效、准确的畸变校正，性能媲美传统需双方向数据的方法。**

- **链接: [http://arxiv.org/pdf/2508.13340v1](http://arxiv.org/pdf/2508.13340v1)**

> **作者:** Sedigheh Dargahi; Sylvain Bouix; Christian Desrosier
>
> **摘要:** Diffusion MRI (dMRI) is a valuable tool to map brain microstructure and connectivity by analyzing water molecule diffusion in tissue. However, acquiring dMRI data requires to capture multiple 3D brain volumes in a short time, often leading to trade-offs in image quality. One challenging artifact is susceptibility-induced distortion, which introduces significant geometric and intensity deformations. Traditional correction methods, such as topup, rely on having access to blip-up and blip-down image pairs, limiting their applicability to retrospective data acquired with a single phase encoding direction. In this work, we propose a deep learning-based approach to correct susceptibility distortions using only a single acquisition (either blip-up or blip-down), eliminating the need for paired acquisitions. Experimental results show that our method achieves performance comparable to topup, demonstrating its potential as an efficient and practical alternative for susceptibility distortion correction in dMRI.
>
---
#### [new 117] ROVER: Robust Loop Closure Verification with Trajectory Prior in Repetitive Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM中的回环验证任务，旨在解决重复环境中因外观相似导致的误检问题。作者提出ROVER方法，利用历史轨迹作为先验约束，通过轨迹一致性评分验证回环真伪，提升鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2508.13488v1](http://arxiv.org/pdf/2508.13488v1)**

> **作者:** Jingwen Yu; Jiayi Yang; Anjun Hu; Jiankun Wang; Ping Tan; Hong Zhang
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Loop closure detection is important for simultaneous localization and mapping (SLAM), which associates current observations with historical keyframes, achieving drift correction and global relocalization. However, a falsely detected loop can be fatal, and this is especially difficult in repetitive environments where appearance-based features fail due to the high similarity. Therefore, verification of a loop closure is a critical step in avoiding false positive detections. Existing works in loop closure verification predominantly focus on learning invariant appearance features, neglecting the prior knowledge of the robot's spatial-temporal motion cue, i.e., trajectory. In this letter, we propose ROVER, a loop closure verification method that leverages the historical trajectory as a prior constraint to reject false loops in challenging repetitive environments. For each loop candidate, it is first used to estimate the robot trajectory with pose-graph optimization. This trajectory is then submitted to a scoring scheme that assesses its compliance with the trajectory without the loop, which we refer to as the trajectory prior, to determine if the loop candidate should be accepted. Benchmark comparisons and real-world experiments demonstrate the effectiveness of the proposed method. Furthermore, we integrate ROVER into state-of-the-art SLAM systems to verify its robustness and efficiency. Our source code and self-collected dataset are available at https://github.com/jarvisyjw/ROVER.
>
---
#### [new 118] Comparing Conditional Diffusion Models for Synthesizing Contrast-Enhanced Breast MRI from Pre-Contrast Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在用预对比MRI合成增强MRI，以减少对比剂使用。通过22种扩散模型变体比较，引入肿瘤感知损失和分割掩码条件，提升病灶保真度，验证了生成图像的临床潜力。**

- **链接: [http://arxiv.org/pdf/2508.13776v1](http://arxiv.org/pdf/2508.13776v1)**

> **作者:** Sebastian Ibarra; Javier del Riego; Alessandro Catanese; Julian Cuba; Julian Cardona; Nataly Leon; Jonathan Infante; Karim Lekadir; Oliver Diaz; Richard Osuala
>
> **备注:** 13 pages, 5 figures, submitted and accepted to MICCAI Deepbreath workshop 2025
>
> **摘要:** Dynamic contrast-enhanced (DCE) MRI is essential for breast cancer diagnosis and treatment. However, its reliance on contrast agents introduces safety concerns, contraindications, increased cost, and workflow complexity. To this end, we present pre-contrast conditioned denoising diffusion probabilistic models to synthesize DCE-MRI, introducing, evaluating, and comparing a total of 22 generative model variants in both single-breast and full breast settings. Towards enhancing lesion fidelity, we introduce both tumor-aware loss functions and explicit tumor segmentation mask conditioning. Using a public multicenter dataset and comparing to respective pre-contrast baselines, we observe that subtraction image-based models consistently outperform post-contrast-based models across five complementary evaluation metrics. Apart from assessing the entire image, we also separately evaluate the region of interest, where both tumor-aware losses and segmentation mask inputs improve evaluation metrics. The latter notably enhance qualitative results capturing contrast uptake, albeit assuming access to tumor localization inputs that are not guaranteed to be available in screening settings. A reader study involving 2 radiologists and 4 MRI technologists confirms the high realism of the synthetic images, indicating an emerging clinical potential of generative contrast-enhancement. We share our codebase at https://github.com/sebastibar/conditional-diffusion-breast-MRI.
>
---
## 更新

#### [replaced 001] Enhancing Cost Efficiency in Active Learning with Candidate Set Query
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06209v2](http://arxiv.org/pdf/2502.06209v2)**

> **作者:** Yeho Gwon; Sehyun Hwang; Hoyoung Kim; Jungseul Ok; Suha Kwak
>
> **备注:** Accepted to TMLR
>
> **摘要:** This paper introduces a cost-efficient active learning (AL) framework for classification, featuring a novel query design called candidate set query. Unlike traditional AL queries requiring the oracle to examine all possible classes, our method narrows down the set of candidate classes likely to include the ground-truth class, significantly reducing the search space and labeling cost. Moreover, we leverage conformal prediction to dynamically generate small yet reliable candidate sets, adapting to model enhancement over successive AL rounds. To this end, we introduce an acquisition function designed to prioritize data points that offer high information gain at lower cost. Empirical evaluations on CIFAR-10, CIFAR-100, and ImageNet64x64 demonstrate the effectiveness and scalability of our framework. Notably, it reduces labeling cost by 48% on ImageNet64x64. The project page can be found at https://yehogwon.github.io/csq-al.
>
---
#### [replaced 002] ReservoirTTA: Prolonged Test-time Adaptation for Evolving and Recurring Domains
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14511v2](http://arxiv.org/pdf/2505.14511v2)**

> **作者:** Guillaume Vray; Devavrat Tomar; Xufeng Gao; Jean-Philippe Thiran; Evan Shelhamer; Behzad Bozorgtabar
>
> **摘要:** This paper introduces ReservoirTTA, a novel plug-in framework designed for prolonged test-time adaptation (TTA) in scenarios where the test domain continuously shifts over time, including cases where domains recur or evolve gradually. At its core, ReservoirTTA maintains a reservoir of domain-specialized models -- an adaptive test-time model ensemble -- that both detects new domains via online clustering over style features of incoming samples and routes each sample to the appropriate specialized model, and thereby enables domain-specific adaptation. This multi-model strategy overcomes key limitations of single model adaptation, such as catastrophic forgetting, inter-domain interference, and error accumulation, ensuring robust and stable performance on sustained non-stationary test distributions. Our theoretical analysis reveals key components that bound parameter variance and prevent model collapse, while our plug-in TTA module mitigates catastrophic forgetting of previously encountered domains. Extensive experiments on the classification corruption benchmarks, including ImageNet-C and CIFAR-10/100-C, as well as the Cityscapes$\rightarrow$ACDC semantic segmentation task, covering recurring and continuously evolving domain shifts, demonstrate that ReservoirTTA significantly improves adaptation accuracy and maintains stable performance across prolonged, recurring shifts, outperforming state-of-the-art methods. Our code is publicly available at https://github.com/LTS5/ReservoirTTA.
>
---
#### [replaced 003] SlotMatch: Distilling Temporally Consistent Object-Centric Representations for Unsupervised Video Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.03411v2](http://arxiv.org/pdf/2508.03411v2)**

> **作者:** Diana-Nicoleta Grigore; Neelu Madan; Andreas Mogelmose; Thomas B. Moeslund; Radu Tudor Ionescu
>
> **摘要:** Unsupervised video segmentation is a challenging computer vision task, especially due to the lack of supervisory signals coupled with the complexity of visual scenes. To overcome this challenge, state-of-the-art models based on slot attention often have to rely on large and computationally expensive neural architectures. To this end, we propose a simple knowledge distillation framework that effectively transfers object-centric representations to a lightweight student. The proposed framework, called SlotMatch, aligns corresponding teacher and student slots via the cosine similarity, requiring no additional distillation objectives or auxiliary supervision. The simplicity of SlotMatch is confirmed via theoretical and empirical evidence, both indicating that integrating additional losses is redundant. We conduct experiments on two datasets to compare the state-of-the-art teacher model, SlotContrast, with our distilled student. The results show that our student based on SlotMatch matches and even outperforms its teacher, while using 3.6x less parameters and running 1.9x faster. Moreover, our student surpasses previous unsupervised video segmentation models.
>
---
#### [replaced 004] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.12332v4](http://arxiv.org/pdf/2505.12332v4)**

> **作者:** Qianyue Hu; Junyan Wu; Wei Lu; Xiangyang Luo
>
> **摘要:** Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning. Audio samples of VoiceCloak are available at https://voice-cloak.github.io/VoiceCloak/.
>
---
#### [replaced 005] A Versatile Pathology Co-pilot via Reasoning Enhanced Multimodal Large Language Model
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17303v2](http://arxiv.org/pdf/2507.17303v2)**

> **作者:** Zhe Xu; Ziyi Liu; Junlin Hou; Jiabo Ma; Cheng Jin; Yihui Wang; Zhixuan Chen; Zhengyu Zhang; Fuxiang Huang; Zhengrui Guo; Fengtao Zhou; Yingxue Xu; Xi Wang; Ronald Cheong Kin Chan; Li Liang; Hao Chen
>
> **摘要:** Multimodal large language models (MLLMs) have emerged as powerful tools for computational pathology, offering unprecedented opportunities to integrate pathological images with language context for comprehensive diagnostic analysis. These models hold particular promise for automating complex tasks that traditionally require expert interpretation of pathologists. However, current MLLM approaches in pathology demonstrate significantly constrained reasoning capabilities, primarily due to their reliance on expensive chain-of-thought annotations. Additionally, existing methods remain limited to simplex application of visual question answering (VQA) at the region-of-interest (ROI) level, failing to address the full spectrum of diagnostic needs such as ROI classification, detection, segmentation, whole-slide-image (WSI) classification and VQA in clinical practice. In this study, we present SmartPath-R1, a versatile MLLM capable of simultaneously addressing both ROI-level and WSI-level tasks while demonstrating robust pathological reasoning capability. Our framework combines scale-dependent supervised fine-tuning and task-aware reinforcement fine-tuning, which circumvents the requirement for chain-of-thought supervision by leveraging the intrinsic knowledge within MLLM. Furthermore, SmartPath-R1 integrates multiscale and multitask analysis through a mixture-of-experts mechanism, enabling dynamic processing for diverse tasks. We curate a large-scale dataset comprising 2.3M ROI samples and 188K WSI samples for training and evaluation. Extensive experiments across 72 tasks validate the effectiveness and superiority of the proposed approach. This work represents a significant step toward developing versatile, reasoning-enhanced AI systems for precision pathology.
>
---
#### [replaced 006] Blending 3D Geometry and Machine Learning for Multi-View Stereopsis
- **分类: cs.CV; cs.AI; cs.CG; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03470v3](http://arxiv.org/pdf/2505.03470v3)**

> **作者:** Vibhas Vats; Md. Alimoor Reza; David Crandall; Soon-heung Jung
>
> **备注:** A pre-print -- accepted at Neurocomputing. arXiv admin note: substantial text overlap with arXiv:2310.19583
>
> **摘要:** Traditional multi-view stereo (MVS) methods primarily depend on photometric and geometric consistency constraints. In contrast, modern learning-based algorithms often rely on the plane sweep algorithm to infer 3D geometry, applying explicit geometric consistency (GC) checks only as a post-processing step, with no impact on the learning process itself. In this work, we introduce GC MVSNet plus plus, a novel approach that actively enforces geometric consistency of reference view depth maps across multiple source views (multi view) and at various scales (multi scale) during the learning phase (see Fig. 1). This integrated GC check significantly accelerates the learning process by directly penalizing geometrically inconsistent pixels, effectively halving the number of training iterations compared to other MVS methods. Furthermore, we introduce a densely connected cost regularization network with two distinct block designs simple and feature dense optimized to harness dense feature connections for enhanced regularization. Extensive experiments demonstrate that our approach achieves a new state of the art on the DTU and BlendedMVS datasets and secures second place on the Tanks and Temples benchmark. To our knowledge, GC MVSNet plus plus is the first method to enforce multi-view, multi-scale supervised geometric consistency during learning. Our code is available.
>
---
#### [replaced 007] Benchmarking Federated Learning for Semantic Datasets: Federated Scene Graph Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.10436v3](http://arxiv.org/pdf/2412.10436v3)**

> **作者:** SeungBum Ha; Taehwan Lee; Jiyoun Lim; Sung Whan Yoon
>
> **备注:** This work has been accepted for publication in Pattern Recognition Letters
>
> **摘要:** Federated learning (FL) enables decentralized training while preserving data privacy, yet existing FL benchmarks address relatively simple classification tasks, where each sample is annotated with a one-hot label. However, little attention has been paid to demonstrating an FL benchmark that handles complicated semantics, where each sample encompasses diverse semantic information, such as relations between objects. Because the existing benchmarks are designed to distribute data in a narrow view of a single semantic, managing the complicated semantic heterogeneity across clients when formalizing FL benchmarks is non-trivial. In this paper, we propose a benchmark process to establish an FL benchmark with controllable semantic heterogeneity across clients: two key steps are (i) data clustering with semantics and (ii) data distributing via controllable semantic heterogeneity across clients. As a proof of concept, we construct a federated PSG benchmark, demonstrating the efficacy of the existing PSG methods in an FL setting with controllable semantic heterogeneity of scene graphs. We also present the effectiveness of our benchmark by applying robust federated learning algorithms to data heterogeneity to show increased performance. To our knowledge, this is the first benchmark framework that enables federated learning and its evaluation for multi-semantic vision tasks under the controlled semantic heterogeneity. Our code is available at https://github.com/Seung-B/FL-PSG.
>
---
#### [replaced 008] Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2404.07846v4](http://arxiv.org/pdf/2404.07846v4)**

> **作者:** Junyi Li; Zhilu Zhang; Wangmeng Zuo
>
> **备注:** AAAI 2025 Camera Ready, update Fig.4
>
> **摘要:** Blind-spot networks (BSN) have been prevalent neural architectures in self-supervised image denoising (SSID). However, most existing BSNs are conducted with convolution layers. Although transformers have shown the potential to overcome the limitations of convolutions in many image restoration tasks, the attention mechanisms may violate the blind-spot requirement, thereby restricting their applicability in BSN. To this end, we propose to analyze and redesign the channel and spatial attentions to meet the blind-spot requirement. Specifically, channel self-attention may leak the blind-spot information in multi-scale architectures, since the downsampling shuffles the spatial feature into channel dimensions. To alleviate this problem, we divide the channel into several groups and perform channel attention separately. For spatial selfattention, we apply an elaborate mask to the attention matrix to restrict and mimic the receptive field of dilated convolution. Based on the redesigned channel and window attentions, we build a Transformer-based Blind-Spot Network (TBSN), which shows strong local fitting and global perspective abilities. Furthermore, we introduce a knowledge distillation strategy that distills TBSN into smaller denoisers to improve computational efficiency while maintaining performance. Extensive experiments on real-world image denoising datasets show that TBSN largely extends the receptive field and exhibits favorable performance against state-of-theart SSID methods.
>
---
#### [replaced 009] RadGPT: Constructing 3D Image-Text Tumor Datasets
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.04678v2](http://arxiv.org/pdf/2501.04678v2)**

> **作者:** Pedro R. A. S. Bassi; Mehmet Can Yavuz; Kang Wang; Xiaoxi Chen; Wenxuan Li; Sergio Decherchi; Andrea Cavalli; Yang Yang; Alan Yuille; Zongwei Zhou
>
> **摘要:** Cancers identified in CT scans are usually accompanied by detailed radiology reports, but publicly available CT datasets often lack these essential reports. This absence limits their usefulness for developing accurate report generation AI. To address this gap, we present AbdomenAtlas 3.0, the first public, high-quality abdominal CT dataset with detailed, expert-reviewed radiology reports. All reports are paired with per-voxel masks and they describe liver, kidney and pancreatic tumors. AbdomenAtlas 3.0 has 9,262 triplets of CT, mask and report--3,955 with tumors. These CT scans come from 17 public datasets. Besides creating the reports for these datasets, we expanded their number of tumor masks by 4.2x, identifying 3,011 new tumor cases. Notably, the reports in AbdomenAtlas 3.0 are more standardized, and generated faster than traditional human-made reports. They provide details like tumor size, location, attenuation and surgical resectability. These reports were created by 12 board-certified radiologists using our proposed RadGPT, a novel framework that converted radiologist-revised tumor segmentation masks into structured and narrative reports. Besides being a dataset creation tool, RadGPT can also become a fully-automatic, segmentation-assisted report generation method. We benchmarked this method and 5 state-of-the-art report generation vision-language models. Our results show that segmentation strongly improves tumor detection in AI-made reports.
>
---
#### [replaced 010] SRMA-Mamba: Spatial Reverse Mamba Attention Network for Pathological Liver Segmentation in MRI Volumes
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.12410v2](http://arxiv.org/pdf/2508.12410v2)**

> **作者:** Jun Zeng; Yannan Huang; Elif Keles; Halil Ertugrul Aktas; Gorkem Durak; Nikhil Kumar Tomar; Quoc-Huy Trinh; Deepak Ranjan Nayak; Ulas Bagci; Debesh Jha
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Liver Cirrhosis plays a critical role in the prognosis of chronic liver disease. Early detection and timely intervention are critical in significantly reducing mortality rates. However, the intricate anatomical architecture and diverse pathological changes of liver tissue complicate the accurate detection and characterization of lesions in clinical settings. Existing methods underutilize the spatial anatomical details in volumetric MRI data, thereby hindering their clinical effectiveness and explainability. To address this challenge, we introduce a novel Mamba-based network, SRMA-Mamba, designed to model the spatial relationships within the complex anatomical structures of MRI volumes. By integrating the Spatial Anatomy-Based Mamba module (SABMamba), SRMA-Mamba performs selective Mamba scans within liver cirrhotic tissues and combines anatomical information from the sagittal, coronal, and axial planes to construct a global spatial context representation, enabling efficient volumetric segmentation of pathological liver structures. Furthermore, we introduce the Spatial Reverse Attention module (SRMA), designed to progressively refine cirrhotic details in the segmentation map, utilizing both the coarse segmentation map and hierarchical encoding features. Extensive experiments demonstrate that SRMA-Mamba surpasses state-of-the-art methods, delivering exceptional performance in 3D pathological liver segmentation. Our code is available for public: https://github.com/JunZengz/SRMA-Mamba.
>
---
#### [replaced 011] ResFlow: Fine-tuning Residual Optical Flow for Event-based High Temporal Resolution Motion Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09105v2](http://arxiv.org/pdf/2412.09105v2)**

> **作者:** Qianang Zhou; Zhiyu Zhu; Junhui Hou; Yongjian Deng; Youfu Li; Junlin Xiong
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Event cameras hold significant promise for high-temporal-resolution (HTR) motion estimation. However, estimating event-based HTR optical flow faces two key challenges: the absence of HTR ground-truth data and the intrinsic sparsity of event data. Most existing approaches rely on the flow accumulation paradigms to indirectly supervise intermediate flows, often resulting in accumulation errors and optimization difficulties. To address these challenges, we propose a residual-based paradigm for estimating HTR optical flow with event data. Our approach separates HTR flow estimation into two stages: global linear motion estimation and HTR residual flow refinement. The residual paradigm effectively mitigates the impacts of event sparsity on optimization and is compatible with any LTR algorithm. Next, to address the challenge posed by the absence of HTR ground truth, we incorporate novel learning strategies. Specifically, we initially employ a shared refiner to estimate the residual flows, enabling both LTR supervision and HTR inference. Subsequently, we introduce regional noise to simulate the residual patterns of intermediate flows, facilitating the adaptation from LTR supervision to HTR inference. Additionally, we show that the noise-based strategy supports in-domain self-supervised training. Comprehensive experimental results demonstrate that our approach achieves state-of-the-art accuracy in both LTR and HTR metrics, highlighting its effectiveness and superiority.
>
---
#### [replaced 012] MMHMER:Multi-viewer and Multi-task for Handwritten Mathematical Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05557v2](http://arxiv.org/pdf/2502.05557v2)**

> **作者:** Kehua Chen; Haoyang Shen; Lifan Zhong; Mingyi Chen
>
> **备注:** 7 pages;2 figures
>
> **摘要:** Handwritten Mathematical Expression Recognition (HMER) methods have made remarkable progress, with most existing HMER approaches based on either a hybrid CNN/RNN-based with GRU architecture or Transformer architectures. Each of these has its strengths and weaknesses. Leveraging different model structures as viewers and effectively integrating their diverse capabilities presents an intriguing avenue for exploration. This involves addressing two key challenges: 1) How to fuse these two methods effectively, and 2) How to achieve higher performance under an appropriate level of complexity. This paper proposes an efficient CNN-Transformer multi-viewer, multi-task approach to enhance the model's recognition performance. Our MMHMER model achieves 63.96%, 62.51%, and 65.46% ExpRate on CROHME14, CROHME16, and CROHME19, outperforming Posformer with an absolute gain of 1.28%, 1.48%, and 0.58%. The main contribution of our approach is that we propose a new multi-view, multi-task framework that can effectively integrate the strengths of CNN and Transformer. By leveraging the feature extraction capabilities of CNN and the sequence modeling capabilities of Transformer, our model can better handle the complexity of handwritten mathematical expressions.
>
---
#### [replaced 013] Diffusion Noise Feature: Accurate and Fast Generated Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.02625v3](http://arxiv.org/pdf/2312.02625v3)**

> **作者:** Yichi Zhang; Xiaogang Xu
>
> **备注:** Accepted by ECAI 2025
>
> **摘要:** Generative models now produce images with such stunning realism that they can easily deceive the human eye. While this progress unlocks vast creative potential, it also presents significant risks, such as the spread of misinformation. Consequently, detecting generated images has become a critical research challenge. However, current detection methods are often plagued by low accuracy and poor generalization. In this paper, to address these limitations and enhance the detection of generated images, we propose a novel representation, Diffusion Noise Feature (DNF). Derived from the inverse process of diffusion models, DNF effectively amplifies the subtle, high-frequency artifacts that act as fingerprints of artificial generation. Our key insight is that real and generated images exhibit distinct DNF signatures, providing a robust basis for differentiation. By training a simple classifier such as ResNet-50 on DNF, our approach achieves remarkable accuracy, robustness, and generalization in detecting generated images, including those from unseen generators or with novel content. Extensive experiments across four training datasets and five test sets confirm that DNF establishes a new state-of-the-art in generated image detection. The code is available at https://github.com/YichiCS/Diffusion-Noise-Feature.
>
---
#### [replaced 014] RAPNet: A Receptive-Field Adaptive Convolutional Neural Network for Pansharpening
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.10461v3](http://arxiv.org/pdf/2507.10461v3)**

> **作者:** Tao Tang; Chengxu Yang
>
> **备注:** Accepted by the 6th International Conference on Artificial Intelligence and Electromechanical Automation (AIEA 2025). 5 pages, 6 figures
>
> **摘要:** Pansharpening refers to the process of integrating a high resolution panchromatic (PAN) image with a lower resolution multispectral (MS) image to generate a fused product, which is pivotal in remote sensing. Despite the effectiveness of CNNs in addressing this challenge, they are inherently constrained by the uniform application of convolutional kernels across all spatial positions, overlooking local content variations. To overcome this issue, we introduce RAPNet, a new architecture that leverages content-adaptive convolution. At its core, RAPNet employs the Receptive-field Adaptive Pansharpening Convolution (RAPConv), designed to produce spatially adaptive kernels responsive to local feature context, thereby enhancing the precision of spatial detail extraction. Additionally, the network integrates the Pansharpening Dynamic Feature Fusion (PAN-DFF) module, which incorporates an attention mechanism to achieve an optimal balance between spatial detail enhancement and spectral fidelity. Comprehensive evaluations on publicly available datasets confirm that RAPNet delivers superior performance compared to existing approaches, as demonstrated by both quantitative metrics and qualitative assessments. Ablation analyses further substantiate the effectiveness of the proposed adaptive components.
>
---
#### [replaced 015] Disentangled Representation Learning with the Gromov-Monge Gap
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2407.07829v3](http://arxiv.org/pdf/2407.07829v3)**

> **作者:** Théo Uscidda; Luca Eyring; Karsten Roth; Fabian Theis; Zeynep Akata; Marco Cuturi
>
> **备注:** ICLR 2025
>
> **摘要:** Learning disentangled representations from unlabelled data is a fundamental challenge in machine learning. Solving it may unlock other problems, such as generalization, interpretability, or fairness. Although remarkably challenging to solve in theory, disentanglement is often achieved in practice through prior matching. Furthermore, recent works have shown that prior matching approaches can be enhanced by leveraging geometrical considerations, e.g., by learning representations that preserve geometric features of the data, such as distances or angles between points. However, matching the prior while preserving geometric features is challenging, as a mapping that fully preserves these features while aligning the data distribution with the prior does not exist in general. To address these challenges, we introduce a novel approach to disentangled representation learning based on quadratic optimal transport. We formulate the problem using Gromov-Monge maps that transport one distribution onto another with minimal distortion of predefined geometric features, preserving them as much as can be achieved. To compute such maps, we propose the Gromov-Monge-Gap (GMG), a regularizer quantifying whether a map moves a reference distribution with minimal geometry distortion. We demonstrate the effectiveness of our approach for disentanglement across four standard benchmarks, outperforming other methods leveraging geometric considerations.
>
---
#### [replaced 016] Advancing Toward Robust and Scalable Fingerprint Orientation Estimation: From Gradients to Deep Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2010.11563v2](http://arxiv.org/pdf/2010.11563v2)**

> **作者:** Amit Kumar Trivedi; Jasvinder Pal Singh
>
> **摘要:** The study identifies a clear evolution from traditional methods to more advanced machine learning approaches. Current algorithms face persistent challenges, including degraded image quality, damaged ridge structures, and background noise, which impact performance. To overcome these limitations, future research must focus on developing efficient algorithms with lower computational complexity while maintaining robust performance across varied conditions. Hybrid methods that combine the simplicity and efficiency of gradient-based techniques with the adaptability and robustness of machine learning are particularly promising for advancing fingerprint recognition systems. Fingerprint orientation estimation plays a crucial role in improving the reliability and accuracy of biometric systems. This study highlights the limitations of current approaches and underscores the importance of designing next-generation algorithms that can operate efficiently across diverse application domains. By addressing these challenges, future developments could enhance the scalability, reliability, and applicability of biometric systems, paving the way for broader use in security and identification technologies.
>
---
#### [replaced 017] Unlocking the Potential of MLLMs in Referring Expression Segmentation via a Light-weight Mask Decoder
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.04107v3](http://arxiv.org/pdf/2508.04107v3)**

> **作者:** Jingchao Wang; Zhijian Wu; Dingjiang Huang; Yefeng Zheng; Hong Wang
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Reference Expression Segmentation (RES) aims to segment image regions specified by referring expressions and has become popular with the rise of multimodal large models (MLLMs). While MLLMs excel in semantic understanding, their token-generation paradigm struggles with pixel-level dense prediction. Existing RES methods either couple MLLMs with the parameter-heavy Segment Anything Model (SAM) with 632M network parameters or adopt SAM-free lightweight pipelines that sacrifice accuracy. To address the trade-off between performance and cost, we specifically propose MLLMSeg, a novel framework that fully exploits the inherent visual detail features encoded in the MLLM vision encoder without introducing an extra visual encoder. Besides, we propose a detail-enhanced and semantic-consistent feature fusion module (DSFF) that fully integrates the detail-related visual feature with the semantic-related feature output by the large language model (LLM) of MLLM. Finally, we establish a light-weight mask decoder with only 34M network parameters that optimally leverages detailed spatial features from the visual encoder and semantic features from the LLM to achieve precise mask prediction. Extensive experiments demonstrate that our method generally surpasses both SAM-based and SAM-free competitors, striking a better balance between performance and cost. Code is available at https://github.com/jcwang0602/MLLMSeg.
>
---
#### [replaced 018] DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10486v2](http://arxiv.org/pdf/2504.10486v2)**

> **作者:** Zeren Jiang; Shaofei Wang; Siyu Tang
>
> **备注:** 17 pages, 9 figures, ICCV 2025 Findings Oral, Project pages: https://jzr99.github.io/DNF-Avatar/
>
> **摘要:** Creating relightable and animatable human avatars from monocular videos is a rising research topic with a range of applications, e.g. virtual reality, sports, and video games. Previous works utilize neural fields together with physically based rendering (PBR), to estimate geometry and disentangle appearance properties of human avatars. However, one drawback of these methods is the slow rendering speed due to the expensive Monte Carlo ray tracing. To tackle this problem, we proposed to distill the knowledge from implicit neural fields (teacher) to explicit 2D Gaussian splatting (student) representation to take advantage of the fast rasterization property of Gaussian splatting. To avoid ray-tracing, we employ the split-sum approximation for PBR appearance. We also propose novel part-wise ambient occlusion probes for shadow computation. Shadow prediction is achieved by querying these probes only once per pixel, which paves the way for real-time relighting of avatars. These techniques combined give high-quality relighting results with realistic shadow effects. Our experiments demonstrate that the proposed student model achieves comparable or even better relighting results with our teacher model while being 370 times faster at inference time, achieving a 67 FPS rendering speed.
>
---
#### [replaced 019] Active contours driven by local and global intensity fitting energy with application to SAR image segmentation and its fast solvers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.11849v2](http://arxiv.org/pdf/2312.11849v2)**

> **作者:** Guangming Liu
>
> **备注:** 21 pages,28 figures
>
> **摘要:** In this paper, we propose a novel variational active contour model based on Aubert-Aujol (AA) denoising model, which hybrides geodesic active contour (GAC) model with active contours without edges (ACWE) model and can be used to segment images corrupted by multiplicative gamma noise. We transform the proposed model into classic ROF model by adding a proximity term. [26] is submitted on 29-Aug-2013, and our early edition ever submitted to TGRS on 12-Jun-2012, Venkatakrishnan et al. [27] proposed their 'pnp algorithm' on 29-May-2013, so Venkatakrishnan and we proposed the 'pnp algorithm'almost simultaneously. Inspired by a fast denosing algorithm proposed by Jia-Zhao recently, we propose two fast fixed point algorithms to solve SAR image segmentation question. Experimental results for real SAR images show that the proposed image segmentation model can efficiently stop the contours at weak or blurred edges, and can automatically detect the exterior and interior boundaries of images with multiplicative gamma noise. The proposed fast fixed point algorithms are robustness to initialization contour, and can further reduce about 15% of the time needed for algorithm proposed by Goldstein-Osher.
>
---
#### [replaced 020] Identify, Isolate, and Purge: Mitigating Hallucinations in LVLMs via Self-Evolving Distillation
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04680v2](http://arxiv.org/pdf/2507.04680v2)**

> **作者:** Wenhao Li; Xiu Su; Jingyi Wu; Feng Yang; Yang Liu; Yi Chen; Shan You; Chang Xu
>
> **备注:** In Figure 2, the correlation coefficient and the scatter plot do not match. I calculated this correlation using two sets of settings. I used the scatter plot from setting A, but accidentally wrote the correlation coefficient, r, from setting B
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable advancements in numerous areas such as multimedia. However, hallucination issues significantly limit their credibility and application potential. Existing mitigation methods typically rely on external tools or the comparison of multi-round inference, which significantly increase inference time. In this paper, we propose \textbf{SE}lf-\textbf{E}volving \textbf{D}istillation (\textbf{SEED}), which identifies hallucinations within the inner knowledge of LVLMs, isolates and purges them, and then distills the purified knowledge back into the model, enabling self-evolution. Furthermore, we identified that traditional distillation methods are prone to inducing void spaces in the output space of LVLMs. To address this issue, we propose a Mode-Seeking Evolving approach, which performs distillation to capture the dominant modes of the purified knowledge distribution, thereby avoiding the chaotic results that could emerge from void spaces. Moreover, we introduce a Hallucination Elimination Adapter, which corrects the dark knowledge of the original model by learning purified knowledge. Extensive experiments on multiple benchmarks validate the superiority of our SEED, demonstrating substantial improvements in mitigating hallucinations for representative LVLM models such as LLaVA-1.5 and InternVL2. Remarkably, the F1 score of LLaVA-1.5 on the hallucination evaluation metric POPE-Random improved from 81.3 to 88.3.
>
---
#### [replaced 021] Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction
- **分类: cs.CV; I.4.5**

- **链接: [http://arxiv.org/pdf/2504.07961v2](http://arxiv.org/pdf/2504.07961v2)**

> **作者:** Zeren Jiang; Chuanxia Zheng; Iro Laina; Diane Larlus; Andrea Vedaldi
>
> **备注:** 17 pages, 6 figures, ICCV 2025 Highlight, Project page: https://geo4d.github.io/
>
> **摘要:** We introduce Geo4D, a method to repurpose video diffusion models for monocular 3D reconstruction of dynamic scenes. By leveraging the strong dynamic priors captured by large-scale pre-trained video models, Geo4D can be trained using only synthetic data while generalizing well to real data in a zero-shot manner. Geo4D predicts several complementary geometric modalities, namely point, disparity, and ray maps. We propose a new multi-modal alignment algorithm to align and fuse these modalities, as well as a sliding window approach at inference time, thus enabling robust and accurate 4D reconstruction of long videos. Extensive experiments across multiple benchmarks show that Geo4D significantly surpasses state-of-the-art video depth estimation methods.
>
---
#### [replaced 022] AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22884v2](http://arxiv.org/pdf/2503.22884v2)**

> **作者:** Yi-Ting Shen; Sungmin Eum; Doheon Lee; Rohit Shete; Chiao-Yi Wang; Heesung Kwon; Shuvra S. Bhattacharyya
>
> **备注:** ICCV 2025
>
> **摘要:** Composed pose retrieval (CPR) enables users to search for human poses by specifying a reference pose and a transition description, but progress in this field is hindered by the scarcity and inconsistency of annotated pose transitions. Existing CPR datasets rely on costly human annotations or heuristic-based rule generation, both of which limit scalability and diversity. In this work, we introduce AutoComPose, the first framework that leverages multimodal large language models (MLLMs) to automatically generate rich and structured pose transition descriptions. Our method enhances annotation quality by structuring transitions into fine-grained body part movements and introducing mirrored/swapped variations, while a cyclic consistency constraint ensures logical coherence between forward and reverse transitions. To advance CPR research, we construct and release two dedicated benchmarks, AIST-CPR and PoseFixCPR, supplementing prior datasets with enhanced attributes. Extensive experiments demonstrate that training retrieval models with AutoComPose yields superior performance over human-annotated and heuristic-based methods, significantly reducing annotation costs while improving retrieval quality. Our work pioneers the automatic annotation of pose transitions, establishing a scalable foundation for future CPR research.
>
---
#### [replaced 023] C2PSA-Enhanced YOLOv11 Architecture: A Novel Approach for Small Target Detection in Cotton Disease Diagnosis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.12219v2](http://arxiv.org/pdf/2508.12219v2)**

> **作者:** Kaiyuan Wang; Jixing Liu; Xiaobo Cai
>
> **摘要:** This study presents a deep learning-based optimization of YOLOv11 for cotton disease detection, developing an intelligent monitoring system. Three key challenges are addressed: (1) low precision in early spot detection (35% leakage rate for sub-5mm2 spots), (2) performance degradation in field conditions (25% accuracy drop), and (3) high error rates (34.7%) in multi-disease scenarios. The proposed solutions include: C2PSA module for enhanced small-target feature extraction; Dynamic category weighting to handle sample imbalance; Improved data augmentation via Mosaic-MixUp scaling. Experimental results on a 4,078-image dataset show: mAP50: 0.820 (+8.0% improvement); mAP50-95: 0.705 (+10.5% improvement); Inference speed: 158 FPS. The mobile-deployed system enables real-time disease monitoring and precision treatment in agricultural applications.
>
---
#### [replaced 024] Stereo-based 3D Anomaly Object Detection for Autonomous Driving: A New Dataset and Baseline
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09214v2](http://arxiv.org/pdf/2507.09214v2)**

> **作者:** Shiyi Mu; Zichong Gu; Hanqi Lyu; Yilin Gao; Shugong Xu
>
> **备注:** under review
>
> **摘要:** 3D detection technology is widely used in the field of autonomous driving, with its application scenarios gradually expanding from enclosed highways to open conventional roads. For rare anomaly categories that appear on the road, 3D detection models trained on closed sets often misdetect or fail to detect anomaly objects. To address this risk, it is necessary to enhance the generalization ability of 3D detection models for targets of arbitrary shapes and to possess the capability to filter out anomalies. The generalization of 3D detection is limited by two factors: the coupled training of 2D and 3D, and the insufficient diversity in the scale distribution of training samples. This paper proposes a Stereo-based 3D Anomaly object Detection (S3AD) algorithm, which decouples the training strategy of 3D and 2D to release the generalization ability for arbitrary 3D foreground detection, and proposes an anomaly scoring algorithm based on foreground confidence prediction, achieving target-level anomaly scoring. In order to further verify and enhance the generalization of anomaly detection, we use a 3D rendering method to synthesize two augmented reality binocular stereo 3D detection datasets which named KITTI-AR. KITTI-AR extends upon KITTI by adding 97 new categories, totaling 6k pairs of stereo images. The KITTI-AR-ExD subset includes 39 common categories as extra training data to address the sparse sample distribution issue. Additionally, 58 rare categories form the KITTI-AR-OoD subset, which are not used in training to simulate zero-shot scenarios in real-world settings, solely for evaluating 3D anomaly detection. Finally, the performance of the algorithm and the dataset is verified in the experiments. (Code and dataset can be obtained at https://github.com/shiyi-mu/S3AD-Code).
>
---
#### [replaced 025] WIPES: Wavelet-based Visual Primitives
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.12615v2](http://arxiv.org/pdf/2508.12615v2)**

> **作者:** Wenhao Zhang; Hao Zhu; Delong Wu; Di Kang; Linchao Bao; Xun Cao; Zhan Ma
>
> **备注:** IEEE/CVF International Conference on Computer Vision 2025
>
> **摘要:** Pursuing a continuous visual representation that offers flexible frequency modulation and fast rendering speed has recently garnered increasing attention in the fields of 3D vision and graphics. However, existing representations often rely on frequency guidance or complex neural network decoding, leading to spectrum loss or slow rendering. To address these limitations, we propose WIPES, a universal Wavelet-based vIsual PrimitivES for representing multi-dimensional visual signals. Building on the spatial-frequency localization advantages of wavelets, WIPES effectively captures both the low-frequency "forest" and the high-frequency "trees." Additionally, we develop a wavelet-based differentiable rasterizer to achieve fast visual rendering. Experimental results on various visual tasks, including 2D image representation, 5D static and 6D dynamic novel view synthesis, demonstrate that WIPES, as a visual primitive, offers higher rendering quality and faster inference than INR-based methods, and outperforms Gaussian-based representations in rendering quality.
>
---
#### [replaced 026] Beyond the Horizon: Decoupling Multi-View UAV Action Recognition via Partial Order Transfer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20530v2](http://arxiv.org/pdf/2504.20530v2)**

> **作者:** Wenxuan Liu; Zhuo Zhou; Xuemei Jia; Siyuan Yang; Wenxin Huang; Xian Zhong; Chia-Wen Lin
>
> **备注:** 11 pages
>
> **摘要:** Action recognition in unmanned aerial vehicles (UAVs) poses unique challenges due to significant view variations along the vertical spatial axis. Unlike traditional ground-based settings, UAVs capture actions at a wide range of altitudes, resulting in considerable appearance discrepancies. We introduce a multi-view formulation tailored to varying UAV altitudes and empirically observe a partial order among views, where recognition accuracy consistently decreases as altitude increases. This observation motivates a novel approach that explicitly models the hierarchical structure of UAV views to improve recognition performance across altitudes. To this end, we propose the Partial Order Guided Multi-View Network (POG-MVNet), designed to address drastic view variations by effectively leveraging view-dependent information across different altitude levels. The framework comprises three key components: a View Partition (VP) module, which uses the head-to-body ratio to group views by altitude; an Order-aware Feature Decoupling (OFD) module, which disentangles action-relevant and view-specific features under partial order guidance; and an Action Partial Order Guide (APOG), which uses the partial order to transfer informative knowledge from easier views to more challenging ones. We conduct experiments on Drone-Action, MOD20, and UAV, demonstrating that POG-MVNet significantly outperforms competing methods. For example, POG-MVNet achieves a 4.7% improvement on Drone-Action and a 3.5% improvement on UAV compared to state-of-the-art methods ASAT and FAR. Code will be released soon.
>
---
#### [replaced 027] SBP-YOLO:A Lightweight Real-Time Model for Detecting Speed Bumps and Potholes
- **分类: cs.CV; cs.AI; 68T45; I.4.8; C.3**

- **链接: [http://arxiv.org/pdf/2508.01339v2](http://arxiv.org/pdf/2508.01339v2)**

> **作者:** Chuanqi Liang; Jie Fu; Miao Yu; Lei Luo
>
> **备注:** 14pages,10figures
>
> **摘要:** Reliable and real-time detection of road speed bumps and potholes is crucial for anticipatory perception in advanced suspension systems, enabling timely and adaptive damping control. Achieving high accuracy and efficiency on embedded platforms remains challenging due to limited computational resources and the small scale of distant targets. This paper presents SBP-YOLO, a lightweight and high-speed detection framework tailored for bump and pothole recognition. Based on YOLOv11n, the model integrates GhostConv and VoVGSCSPC modules into the backbone and neck to reduce computation while enhancing multi-scale semantic features. To improve small-object detection, a P2-level branch is introduced with a lightweight and efficient detection head LEDH mitigating the added computational overhead without compromising accuracy. A hybrid training strategy combining NWD loss, backbone-level knowledge distillation, and Albumentations-driven augmentation further enhances localization precision and robustness. Experiments show that SBP-YOLO achieves 87.0 percent mAP, outperforming the YOLOv11n baseline by 5.8 percent. After TensorRT FP16 quantization, it runs at 139.5 FPS on Jetson AGX Xavier, delivering a 12.4 percent speedup over the P2-enhanced YOLOv11. These results validate the effectiveness of the proposed method for fast and low-latency road condition perception in embedded suspension control systems.
>
---
#### [replaced 028] Fusing Echocardiography Images and Medical Records for Continuous Patient Stratification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2401.07796v3](http://arxiv.org/pdf/2401.07796v3)**

> **作者:** Nathan Painchaud; Jérémie Stym-Popper; Pierre-Yves Courand; Nicolas Thome; Pierre-Marc Jodoin; Nicolas Duchateau; Olivier Bernard
>
> **备注:** 13 pages + 2 pages of supplementary material, accepted for publication in IEEE TUFFC
>
> **摘要:** Deep learning enables automatic and robust extraction of cardiac function descriptors from echocardiographic sequences, such as ejection fraction or strain. These descriptors provide fine-grained information that physicians consider, in conjunction with more global variables from the clinical record, to assess patients' condition. Drawing on novel Transformer models applied to tabular data, we propose a method that considers all descriptors extracted from medical records and echocardiograms to learn the representation of a cardiovascular pathology with a difficult-to-characterize continuum, namely hypertension. Our method first projects each variable into its own representation space using modality-specific approaches. These standardized representations of multimodal data are then fed to a Transformer encoder, which learns to merge them into a comprehensive representation of the patient through the task of predicting a clinical rating. This stratification task is formulated as an ordinal classification to enforce a pathological continuum in the representation space. We observe the major trends along this continuum on a cohort of 239 hypertensive patients, providing unprecedented details in the description of hypertension's impact on various cardiac function descriptors. Our analysis shows that i) the XTab foundation model's architecture allows to reach outstanding performance (96.8% AUROC) even with limited data (less than 200 training samples), ii) stratification across the population is reproducible between trainings (within 5.7% mean absolute error), and iii) patterns emerge in descriptors, some of which align with established physiological knowledge about hypertension, while others could pave the way for a more comprehensive understanding of this pathology. Code is available at https://github.com/creatis-myriad/didactic.
>
---
#### [replaced 029] Assessment of Using Synthetic Data in Brain Tumor Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11922v2](http://arxiv.org/pdf/2508.11922v2)**

> **作者:** Aditi Jahagirdar; Sameer Joshi
>
> **备注:** Updates include improved references, clearer table column title, and minor language corrections
>
> **摘要:** Manual brain tumor segmentation from MRI scans is challenging due to tumor heterogeneity, scarcity of annotated data, and class imbalance in medical imaging datasets. Synthetic data generated by generative models has the potential to mitigate these issues by improving dataset diversity. This study investigates, as a proof of concept, the impact of incorporating synthetic MRI data, generated using a pre-trained GAN model, into training a U-Net segmentation network. Experiments were conducted using real data from the BraTS 2020 dataset, synthetic data generated with the medigan library, and hybrid datasets combining real and synthetic samples in varying proportions. While overall quantitative performance (Dice coefficient, IoU, precision, recall, accuracy) was comparable between real-only and hybrid-trained models, qualitative inspection suggested that hybrid datasets, particularly with 40% real and 60% synthetic data, improved whole tumor boundary delineation. However, region-wise accuracy for the tumor core and the enhancing tumor remained lower, indicating a persistent class imbalance. The findings support the feasibility of synthetic data as an augmentation strategy for brain tumor segmentation, while highlighting the need for larger-scale experiments, volumetric data consistency, and mitigating class imbalance in future work.
>
---
#### [replaced 030] Boosting Adversarial Transferability for Hyperspectral Image Classification Using 3D Structure-invariant Transformation and Weighted Intermediate Feature Divergence
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.10459v2](http://arxiv.org/pdf/2506.10459v2)**

> **作者:** Chun Liu; Bingqian Zhu; Tao Xu; Zheng Zheng; Zheng Li; Wei Yang; Zhigang Han; Jiayao Wang
>
> **摘要:** Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, which pose security challenges to hyperspectral image (HSI) classification based on DNNs. Numerous adversarial attack methods have been designed in the domain of natural images. However, different from natural images, HSIs contains high-dimensional rich spectral information, which presents new challenges for generating adversarial examples. Based on the specific characteristics of HSIs, this paper proposes a novel method to enhance the transferability of the adversarial examples for HSI classification using 3D structure-invariant transformation and weighted intermediate feature divergence. While keeping the HSIs structure invariant, the proposed method divides the image into blocks in both spatial and spectral dimensions. Then, various transformations are applied on each block to increase input diversity and mitigate the overfitting to substitute models. Moreover, a weighted intermediate feature divergence loss is also designed by leveraging the differences between the intermediate features of original and adversarial examples. It constrains the perturbation direction by enlarging the feature maps of the original examples, and assigns different weights to different feature channels to destroy the features that have a greater impact on HSI classification. Extensive experiments demonstrate that the adversarial examples generated by the proposed method achieve more effective adversarial transferability on three public HSI datasets. Furthermore, the method maintains robust attack performance even under defense strategies.
>
---
#### [replaced 031] WHALES: A Multi-Agent Scheduling Dataset for Enhanced Cooperation in Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.13340v3](http://arxiv.org/pdf/2411.13340v3)**

> **作者:** Yinsong Wang; Siwei Chen; Ziyi Song; Sheng Zhou
>
> **摘要:** Cooperative perception research is hindered by the limited availability of datasets that capture the complexity of real-world Vehicle-to-Everything (V2X) interactions, particularly under dynamic communication constraints. To address this gap, we introduce WHALES (Wireless enhanced Autonomous vehicles with Large number of Engaged agents), the first large-scale V2X dataset explicitly designed to benchmark communication-aware agent scheduling and scalable cooperative perception. WHALES introduces a new benchmark that enables state-of-the-art (SOTA) research in communication-aware cooperative perception, featuring an average of 8.4 cooperative agents per scene and 2.01 million annotated 3D objects across diverse traffic scenarios. It incorporates detailed communication metadata to emulate real-world communication bottlenecks, enabling rigorous evaluation of scheduling strategies. To further advance the field, we propose the Coverage-Aware Historical Scheduler (CAHS), a novel scheduling baseline that selects agents based on historical viewpoint coverage, improving perception performance over existing SOTA methods. WHALES bridges the gap between simulated and real-world V2X challenges, providing a robust framework for exploring perception-scheduling co-design, cross-data generalization, and scalability limits. The WHALES dataset and code are available at https://github.com/chensiweiTHU/WHALES.
>
---
#### [replaced 032] Enhancing Visual Reliance in Text Generation: A Bayesian Perspective on Mitigating Hallucination in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19498v2](http://arxiv.org/pdf/2505.19498v2)**

> **作者:** Nanxing Hu; Xiaoyue Duan; Jinchao Zhang; Guoliang Kang
>
> **摘要:** Large Vision-Language Models (LVLMs) usually generate texts which satisfy context coherence but don't match the visual input. Such a hallucination issue hinders LVLMs' applicability in the real world. The key to solving hallucination in LVLM is to make the text generation rely more on the visual content. Most previous works choose to enhance/adjust the features/output of a specific modality (i.e., visual or textual) to alleviate hallucinations in LVLM, which do not explicitly or systematically enhance the visual reliance. In this paper, we comprehensively investigate the factors which may degenerate the visual reliance in text generation of LVLM from a Bayesian perspective. Based on our observations, we propose to mitigate hallucination in LVLM from three aspects. Firstly, we observe that not all visual tokens are informative in generating meaningful texts. We propose to evaluate and remove redundant visual tokens to avoid their disturbance. Secondly, LVLM may encode inappropriate prior information, making it lean toward generating unexpected words. We propose a simple yet effective way to rectify the prior from a Bayesian perspective. Thirdly, we observe that starting from certain steps, the posterior of next-token prediction conditioned on visual tokens may collapse to a prior distribution which does not depend on any informative visual tokens at all. Thus, we propose to stop further text generation to avoid hallucination. Extensive experiments on three benchmarks including POPE, CHAIR, and MME demonstrate that our method can consistently mitigate the hallucination issue of LVLM and performs favorably against previous state-of-the-arts.
>
---
#### [replaced 033] Vision Backbone Efficient Selection for Image Classification in Low-Data Regimes
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.08592v2](http://arxiv.org/pdf/2410.08592v2)**

> **作者:** Joris Guerin; Shray Bansal; Amirreza Shaban; Paulo Mann; Harshvardhan Gazula
>
> **备注:** 16 pages, 8 figures, Accepted at BMVC 2025
>
> **摘要:** Transfer learning has become an essential tool in modern computer vision, allowing practitioners to leverage backbones, pretrained on large datasets, to train successful models from limited annotated data. Choosing the right backbone is crucial, especially for small datasets, since final performance depends heavily on the quality of the initial feature representations. While prior work has conducted benchmarks across various datasets to identify universal top-performing backbones, we demonstrate that backbone effectiveness is highly dataset-dependent, especially in low-data scenarios where no single backbone consistently excels. To overcome this limitation, we introduce dataset-specific backbone selection as a new research direction and investigate its practical viability in low-data regimes. Since exhaustive evaluation is computationally impractical for large backbone pools, we formalize Vision Backbone Efficient Selection (VIBES) as the problem of searching for high-performing backbones under computational constraints. We define the solution space, propose several heuristics, and demonstrate VIBES feasibility for low-data image classification by performing experiments on four diverse datasets. Our results show that even simple search strategies can find well-suited backbones within a pool of over $1300$ pretrained models, outperforming generic benchmark recommendations within just ten minutes of search time on a single GPU (NVIDIA RTX A5000).
>
---
#### [replaced 034] SAR image segmentation algorithms based on I-divergence-TV model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.09365v2](http://arxiv.org/pdf/2312.09365v2)**

> **作者:** Guangming Liu
>
> **备注:** 22 pages,28 figures
>
> **摘要:** In this paper, we propose a novel variational active contour model based on I-divergence-TV model to segment Synthetic aperture radar (SAR) images with multiplicative gamma noise, which hybrides edge-based model with region-based model. The proposed model can efficiently stop the contours at weak or blurred edges, and can automatically detect the exterior and interior boundaries of images. We incorporate the global convex segmentation method and split Bregman technique into the proposed model, and propose a fast fixed point algorithm to solve the global convex segmentation question[25]. [25] is submitted on 29-Aug-2013, and our early edition ever submitted to TGRS on 12-Jun-2012, Venkatakrishnan et al. [26] proposed their 'pnp algorithm' on 29-May-2013, so Venkatakrishnan and we proposed the 'pnp algorithm' almost simultaneously. Experimental results for synthetic images and real SAR images show that the proposed fast fixed point algorithm is robust and efficient compared with the state-of-the-art approach.
>
---
#### [replaced 035] Hyperspectral Image Generation with Unmixing Guided Diffusion Model
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.02601v3](http://arxiv.org/pdf/2506.02601v3)**

> **作者:** Shiyu Shen; Bin Pan; Ziye Zhang; Zhenwei Shi
>
> **摘要:** We address hyperspectral image (HSI) synthesis, a problem that has garnered growing interest yet remains constrained by the conditional generative paradigms that limit sample diversity. While diffusion models have emerged as a state-of-the-art solution for high-fidelity image generation, their direct extension from RGB to hyperspectral domains is challenged by the high spectral dimensionality and strict physical constraints inherent to HSIs. To overcome the challenges, we introduce a diffusion framework explicitly guided by hyperspectral unmixing. The approach integrates two collaborative components: (i) an unmixing autoencoder that projects generation from the image domain into a low-dimensional abundance manifold, thereby reducing computational burden while maintaining spectral fidelity; and (ii) an abundance diffusion process that enforces non-negativity and sum-to-one constraints, ensuring physical consistency of the synthesized data. We further propose two evaluation metrics tailored to hyperspectral characteristics. Comprehensive experiments, assessed with both conventional measures and the proposed metrics, demonstrate that our method produces HSIs with both high quality and diversity, advancing the state of the art in hyperspectral data generation.
>
---
#### [replaced 036] Fully Automated Segmentation of Fiber Bundles in Anatomic Tracing Data
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.12942v2](http://arxiv.org/pdf/2508.12942v2)**

> **作者:** Kyriaki-Margarita Bintsi; Yaël Balbastre; Jingjing Wu; Julia F. Lehman; Suzanne N. Haber; Anastasia Yendiki
>
> **备注:** Accepted at CDMRI, MICCAI 2025
>
> **摘要:** Anatomic tracer studies are critical for validating and improving diffusion MRI (dMRI) tractography. However, large-scale analysis of data from such studies is hampered by the labor-intensive process of annotating fiber bundles manually on histological slides. Existing automated methods often miss sparse bundles or require complex post-processing across consecutive sections, limiting their flexibility and generalizability. We present a streamlined, fully automated framework for fiber bundle segmentation in macaque tracer data, based on a U-Net architecture with large patch sizes, foreground aware sampling, and semisupervised pre-training. Our approach eliminates common errors such as mislabeling terminals as bundles, improves detection of sparse bundles by over 20% and reduces the False Discovery Rate (FDR) by 40% compared to the state-of-the-art, all while enabling analysis of standalone slices. This new framework will facilitate the automated analysis of anatomic tracing data at a large scale, generating more ground-truth data that can be used to validate and optimize dMRI tractography methods.
>
---
#### [replaced 037] MR-EEGWaveNet: Multiresolutional EEGWaveNet for Seizure Detection from Long EEG Recordings
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.17972v2](http://arxiv.org/pdf/2505.17972v2)**

> **作者:** Kazi Mahmudul Hassan; Xuyang Zhao; Hidenori Sugano; Toshihisa Tanaka
>
> **备注:** 33 pages, 10 figures, 18 tables
>
> **摘要:** Feature engineering for generalized seizure detection models remains a significant challenge. Recently proposed models show variable performance depending on the training data and remain ineffective at accurately distinguishing artifacts from seizure data. In this study, we propose a novel end-to-end model, "Multiresolutional EEGWaveNet (MR-EEGWaveNet)," which efficiently distinguishes seizure events from background electroencephalogram (EEG) and artifacts/noise by capturing both temporal dependencies across different time frames and spatial relationships between channels. The model has three modules: convolution, feature extraction, and predictor. The convolution module extracts features through depth-wise and spatio-temporal convolution. The feature extraction module individually reduces the feature dimension extracted from EEG segments and their sub-segments. Subsequently, the extracted features are concatenated into a single vector for classification using a fully connected classifier called the predictor module. In addition, an anomaly score-based post-classification processing technique is introduced to reduce the false-positive rates of the model. Experimental results are reported and analyzed using different parameter settings and datasets (Siena (public) and Juntendo (private)). The proposed MR-EEGWaveNet significantly outperformed the conventional non-multiresolution approach, improving the F1 scores from 0.177 to 0.336 on Siena and 0.327 to 0.488 on Juntendo, with precision gains of 15.9% and 20.62%, respectively.
>
---
#### [replaced 038] Slot Attention with Re-Initialization and Self-Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.23755v2](http://arxiv.org/pdf/2507.23755v2)**

> **作者:** Rongzhen Zhao; Yi Zhao; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Unlike popular solutions based on dense feature maps, Object-Centric Learning (OCL) represents visual scenes as sub-symbolic object-level feature vectors, termed slots, which are highly versatile for tasks involving visual modalities. OCL typically aggregates object superpixels into slots by iteratively applying competitive cross attention, known as Slot Attention, with the slots as the query. However, once initialized, these slots are reused naively, causing redundant slots to compete with informative ones for representing objects. This often results in objects being erroneously segmented into parts. Additionally, mainstream methods derive supervision signals solely from decoding slots into the input's reconstruction, overlooking potential supervision based on internal information. To address these issues, we propose Slot Attention with re-Initialization and self-Distillation (DIAS): $\emph{i)}$ We reduce redundancy in the aggregated slots and re-initialize extra aggregation to update the remaining slots; $\emph{ii)}$ We drive the bad attention map at the first aggregation iteration to approximate the good at the last iteration to enable self-distillation. Experiments demonstrate that DIAS achieves state-of-the-art on OCL tasks like object discovery and recognition, while also improving advanced visual prediction and reasoning. Our source code and model checkpoints are available on https://github.com/Genera1Z/DIAS.
>
---
#### [replaced 039] HouseCrafter: Lifting Floorplans to 3D Scenes with 2D Diffusion Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.20077v2](http://arxiv.org/pdf/2406.20077v2)**

> **作者:** Hieu T. Nguyen; Yiwen Chen; Vikram Voleti; Varun Jampani; Huaizu Jiang
>
> **摘要:** We introduce HouseCrafter, a novel approach that can lift a floorplan into a complete large 3D indoor scene (e.g., a house). Our key insight is to adapt a 2D diffusion model, which is trained on web-scale images, to generate consistent multi-view color (RGB) and depth (D) images across different locations of the scene. Specifically, the RGB-D images are generated autoregressively in a batch-wise manner along sampled locations based on the floorplan, where previously generated images are used as condition to the diffusion model to produce images at nearby locations. The global floorplan and attention design in the diffusion model ensures the consistency of the generated images, from which a 3D scene can be reconstructed. Through extensive evaluation on the 3D-Front dataset, we demonstrate that HouseCraft can generate high-quality house-scale 3D scenes. Ablation studies also validate the effectiveness of different design choices. We will release our code and model weights. Project page: https://neu-vi.github.io/houseCrafter/
>
---
#### [replaced 040] Rapid Urban Visibility Hotspots: Quantifying Building Vertex Visibility from Connected Vehicle Trajectories using Spatial Indexing
- **分类: eess.SY; cs.CV; cs.SY; stat.CO**

- **链接: [http://arxiv.org/pdf/2506.03365v2](http://arxiv.org/pdf/2506.03365v2)**

> **作者:** Artur Grigorev; Adriana-Simona Mihaita
>
> **摘要:** Effective placement of Out-of-Home advertising and street furniture requires accurate identification of locations offering maximum visual exposure to target audiences, particularly vehicular traffic. Traditional site selection methods often rely on static traffic counts or subjective assessments. This research introduces a data-driven methodology to objectively quantify location visibility by analyzing large-scale connected vehicle trajectory data (sourced from Compass IoT) within urban environments. We model the dynamic driver field-of-view using a forward-projected visibility area for each vehicle position derived from interpolated trajectories. By integrating this with building vertex locations extracted from OpenStreetMap, we quantify the cumulative visual exposure, or ``visibility count'', for thousands of potential points of interest near roadways. The analysis reveals that visibility is highly concentrated, identifying specific ``visual hotspots'' that receive disproportionately high exposure compared to average locations. The core technical contribution involves the construction of a BallTree spatial index over building vertices. This enables highly efficient (O(logN) complexity) radius queries to determine which vertices fall within the viewing circles of millions of trajectory points across numerous trips, significantly outperforming brute-force geometric checks. Analysis reveals two key findings: 1) Visibility is highly concentrated, identifying distinct 'visual hotspots' receiving disproportionately high exposure compared to average locations. 2) The aggregated visibility counts across vertices conform to a Log-Normal distribution.
>
---
#### [replaced 041] Regional quality estimation for echocardiography using deep learning
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.00591v5](http://arxiv.org/pdf/2408.00591v5)**

> **作者:** Gilles Van De Vyver; Svein-Erik Måsøy; Håvard Dalen; Bjørnar Leangen Grenne; Espen Holte; Sindre Hellum Olaisen; John Nyberg; Andreas Østvik; Lasse Løvstakken; Erik Smistad
>
> **摘要:** Automatic estimation of cardiac ultrasound image quality can be beneficial for guiding operators and ensuring the accuracy of clinical measurements. Previous work often fails to distinguish the view correctness of the echocardiogram from the image quality. Additionally, previous studies only provide a global image quality value, which limits their practical utility. In this work, we developed and compared three methods to estimate image quality: 1) classic pixel-based metrics like the generalized contrast-to-noise ratio (gCNR) on myocardial segments as region of interest and left ventricle lumen as background, obtained using a U-Net segmentation 2) local image coherence derived from a U-Net model that predicts coherence from B-Mode images 3) a deep convolutional network that predicts the quality of each region directly in an end-to-end fashion. We evaluate each method against manual regional image quality annotations by three experienced cardiologists. The results indicate poor performance of the gCNR metric, with Spearman correlation to the annotations of rho = 0.24. The end-to-end learning model obtains the best result, rho = 0.69, comparable to the inter-observer correlation, rho = 0.63. Finally, the coherence-based method, with rho = 0.58, outperformed the classical metrics and is more generic than the end-to-end approach. The image quality prediction tool is available as an open source Python library at https://github.com/GillesVanDeVyver/arqee.
>
---
#### [replaced 042] ContrastAlign: Toward Robust BEV Feature Alignment via Contrastive Learning for Multi-Modal 3D Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.16873v3](http://arxiv.org/pdf/2405.16873v3)**

> **作者:** Ziying Song; Hongyu Pan; Feiyang Jia; Yongchang Zhang; Lin Liu; Lei Yang; Shaoqing Xu; Peiliang Wu; Caiyan Jia; Zheng Zhang; Yadan Luo
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** In the field of 3D object detection tasks, fusing heterogeneous features from LiDAR and camera sensors into a unified Bird's Eye View (BEV) representation is a widely adopted paradigm. However, existing methods often suffer from imprecise sensor calibration, leading to feature misalignment in LiDAR-camera BEV fusion. Moreover, such inaccuracies cause errors in depth estimation for the camera branch, aggravating misalignment between LiDAR and camera BEV features. In this work, we propose a novel ContrastAlign approach that utilizes contrastive learning to enhance the alignment of heterogeneous modalities, thereby improving the robustness of the fusion process. Specifically, our approach comprises three key components: (1) the L-Instance module, which extracts LiDAR instance features within the LiDAR BEV features; (2) the C-Instance module, which predicts camera instance features through Region of Interest (RoI) pooling on the camera BEV features; (3) the InstanceFusion module, which employs contrastive learning to generate consistent instance features across heFterogeneous modalities. Subsequently, we use graph matching to calculate the similarity between the neighboring camera instance features and the similarity instance features to complete the alignment of instance features. Our method achieves SOTA performance, with an mAP of 71.5%, surpassing GraphBEV by 1.4% on the nuScenes val set. Importantly, our method excels BEVFusion under conditions with spatial & temporal misalignment noise, improving mAP by 1.4% and 11.1% on nuScenes dataset. Notably, on the Argoverse2 dataset, ContrastAlign outperforms GraphBEV by 1.0% in mAP, indicating that the farther the distance, the more severe the feature misalignment and the more effective.
>
---
#### [replaced 043] Segment Anything in Pathology Images with Natural Language
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.20988v2](http://arxiv.org/pdf/2506.20988v2)**

> **作者:** Zhixuan Chen; Junlin Hou; Liqi Lin; Yihui Wang; Yequan Bie; Xi Wang; Yanning Zhou; Ronald Cheong Kin Chan; Hao Chen
>
> **摘要:** Pathology image segmentation is crucial in computational pathology for analyzing histological features relevant to cancer diagnosis and prognosis. However, current methods face major challenges in clinical applications due to limited annotated data and restricted category definitions. To address these limitations, we propose PathSegmentor, the first text-prompted segmentation foundation model designed specifically for pathology images. We also introduce PathSeg, the largest and most comprehensive dataset for pathology segmentation, built from 21 public sources and containing 275k image-mask-label triples across 160 diverse categories. With PathSegmentor, users can perform semantic segmentation using natural language prompts, eliminating the need for laborious spatial inputs such as points or boxes. Extensive experiments demonstrate that PathSegmentor outperforms specialized models with higher accuracy and broader applicability, while maintaining a compact architecture. It significantly surpasses existing spatial- and text-prompted models by 0.145 and 0.429 in overall Dice scores, respectively, showing strong robustness in segmenting complex structures and generalizing to external datasets. Moreover, PathSegmentor's outputs enhance the interpretability of diagnostic models through feature importance estimation and imaging biomarker discovery, offering pathologists evidence-based support for clinical decision-making. This work advances the development of explainable AI in precision oncology.
>
---
#### [replaced 044] MAViS: A Multi-Agent Framework for Long-Sequence Video Storytelling
- **分类: cs.CV; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.08487v2](http://arxiv.org/pdf/2508.08487v2)**

> **作者:** Qian Wang; Ziqi Huang; Ruoxi Jia; Paul Debevec; Ning Yu
>
> **备注:** Video Generation Agent
>
> **摘要:** Despite recent advances, long-sequence video generation frameworks still suffer from significant limitations: poor assistive capability, suboptimal visual quality, and limited expressiveness. To mitigate these limitations, we propose MAViS, an end-to-end multi-agent collaborative framework for long-sequence video storytelling. MAViS orchestrates specialized agents across multiple stages, including script writing, shot designing, character modeling, keyframe generation, video animation, and audio generation. In each stage, agents operate under the 3E Principle -- Explore, Examine, and Enhance -- to ensure the completeness of intermediate outputs. Considering the capability limitations of current generative models, we propose the Script Writing Guidelines to optimize compatibility between scripts and generative tools. Experimental results demonstrate that MAViS achieves state-of-the-art performance in assistive capability, visual quality, and video expressiveness. Its modular framework further enables scalability with diverse generative models and tools. With just a brief user prompt, MAViS is capable of producing high-quality, expressive long-sequence video storytelling, enriching inspirations and creativity for users. To the best of our knowledge, MAViS is the only framework that provides multimodal design output -- videos with narratives and background music.
>
---
#### [replaced 045] Dataset Condensation with Color Compensation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.01139v2](http://arxiv.org/pdf/2508.01139v2)**

> **作者:** Huyu Wu; Duo Su; Junjie Hou; Guang Li
>
> **摘要:** Dataset condensation always faces a constitutive trade-off: balancing performance and fidelity under extreme compression. Existing methods struggle with two bottlenecks: image-level selection methods (Coreset Selection, Dataset Quantization) suffer from inefficiency condensation, while pixel-level optimization (Dataset Distillation) introduces semantic distortion due to over-parameterization. With empirical observations, we find that a critical problem in dataset condensation is the oversight of color's dual role as an information carrier and a basic semantic representation unit. We argue that improving the colorfulness of condensed images is beneficial for representation learning. Motivated by this, we propose DC3: a Dataset Condensation framework with Color Compensation. After a calibrated selection strategy, DC3 utilizes the latent diffusion model to enhance the color diversity of an image rather than creating a brand-new one. Extensive experiments demonstrate the superior performance and generalization of DC3 that outperforms SOTA methods across multiple benchmarks. To the best of our knowledge, besides focusing on downstream tasks, DC3 is the first research to fine-tune pre-trained diffusion models with condensed datasets. The FID results prove that training networks with our high-quality datasets is feasible without model collapse or other degradation issues. Code and generated data are available at https://github.com/528why/Dataset-Condensation-with-Color-Compensation.
>
---
#### [replaced 046] LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2308.09908v5](http://arxiv.org/pdf/2308.09908v5)**

> **作者:** Zhenrong Zhang; Jianan Liu; Yuxuan Xia; Tao Huang; Qing-Long Han; Hongbin Liu
>
> **摘要:** Online multi-object tracking (MOT) plays a pivotal role in autonomous systems. The state-of-the-art approaches usually employ a tracking-by-detection method, and data association plays a critical role. This paper proposes a learning and graph-optimized (LEGO) modular tracker to improve data association performance in the existing literature. The proposed LEGO tracker integrates graph optimization and self-attention mechanisms, which efficiently formulate the association score map, facilitating the accurate and efficient matching of objects across time frames. To further enhance the state update process, the Kalman filter is added to ensure consistent tracking by incorporating temporal coherence in the object states. Our proposed method utilizing LiDAR alone has shown exceptional performance compared to other online tracking approaches, including LiDAR-based and LiDAR-camera fusion-based methods. LEGO ranked 1st at the time of submitting results to KITTI object tracking evaluation ranking board and remains 2nd at the time of submitting this paper, among all online trackers in the KITTI MOT benchmark for cars1
>
---
#### [replaced 047] Rethinking Weight-Averaged Model-merging
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.09263v5](http://arxiv.org/pdf/2411.09263v5)**

> **作者:** Hu Wang; Congbo Ma; Ibrahim Almakky; Ian Reid; Gustavo Carneiro; Mohammad Yaqub
>
> **摘要:** Model merging, particularly through weight averaging, has shown surprising effectiveness in saving computations and improving model performance without any additional training. However, the interpretability of why and how this technique works remains unclear. In this work, we reinterpret weight-averaged model merging through the lens of interpretability and provide empirical insights into the underlying mechanisms that govern its behavior. We approach the problem from three perspectives: (1) we analyze the learned weight structures and demonstrate that model weights encode structured representations that help explain the compatibility of weight averaging; (2) we compare averaging in weight space and feature space across diverse model architectures (CNNs and ViTs) and datasets, aiming to expose under which circumstances what combination paradigm will work more effectively; (3) we study the effect of parameter scaling on prediction stability, highlighting how weight averaging acts as a form of regularization that contributes to robustness. By framing these analyses in an interpretability context, our work contributes to a more transparent and systematic understanding of model merging for stakeholders interested in the safety and reliability of untrained model combination methods. The code is available at https://github.com/billhhh/Rethink-Merge.
>
---
#### [replaced 048] FreqDGT: Frequency-Adaptive Dynamic Graph Networks with Transformer for Cross-subject EEG Emotion Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22807v3](http://arxiv.org/pdf/2506.22807v3)**

> **作者:** Yueyang Li; Shengyu Gong; Weiming Zeng; Nizhuan Wang; Wai Ting Siok
>
> **摘要:** Electroencephalography (EEG) serves as a reliable and objective signal for emotion recognition in affective brain-computer interfaces, offering unique advantages through its high temporal resolution and ability to capture authentic emotional states that cannot be consciously controlled. However, cross-subject generalization remains a fundamental challenge due to individual variability, cognitive traits, and emotional responses. We propose FreqDGT, a frequency-adaptive dynamic graph transformer that systematically addresses these limitations through an integrated framework. FreqDGT introduces frequency-adaptive processing (FAP) to dynamically weight emotion-relevant frequency bands based on neuroscientific evidence, employs adaptive dynamic graph learning (ADGL) to learn input-specific brain connectivity patterns, and implements multi-scale temporal disentanglement network (MTDN) that combines hierarchical temporal transformers with adversarial feature disentanglement to capture both temporal dynamics and ensure cross-subject robustness. Comprehensive experiments demonstrate that FreqDGT significantly improves cross-subject emotion recognition accuracy, confirming the effectiveness of integrating frequency-adaptive, spatial-dynamic, and temporal-hierarchical modeling while ensuring robustness to individual differences. The code is available at https://github.com/NZWANG/FreqDGT.
>
---
#### [replaced 049] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10082v5](http://arxiv.org/pdf/2506.10082v5)**

> **作者:** Chenjian Gao; Lihe Ding; Xin Cai; Zhanpeng Huang; Zibin Wang; Tianfan Xue
>
> **备注:** 9 pages
>
> **摘要:** Video editing using diffusion models has achieved remarkable results in generating high-quality edits for videos. However, current methods often rely on large-scale pretraining, limiting flexibility for specific edits. First-frame-guided editing provides control over the first frame, but lacks flexibility over subsequent frames. To address this, we propose a mask-based LoRA (Low-Rank Adaptation) tuning method that adapts pretrained Image-to-Video (I2V) models for flexible video editing. Our key innovation is using a spatiotemporal mask to strategically guide the LoRA fine-tuning process. This teaches the model two distinct skills: first, to interpret the mask as a command to either preserve content from the source video or generate new content in designated regions. Second, for these generated regions, LoRA learns to synthesize either temporally consistent motion inherited from the video or novel appearances guided by user-provided reference frames. This dual-capability LoRA grants users control over the edit's entire temporal evolution, allowing complex transformations like an object rotating or a flower blooming. Experimental results show our method achieves superior video editing performance compared to baseline methods. Project Page: https://cjeen.github.io/LoRAEdit
>
---
#### [replaced 050] Spatially-guided Temporal Aggregation for Robust Event-RGB Optical Flow Estimation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00838v2](http://arxiv.org/pdf/2501.00838v2)**

> **作者:** Qianang Zhou; Junhui Hou; Meiyi Yang; Yongjian Deng; Youfu Li; Junlin Xiong
>
> **备注:** 11 pages, 8 figures, under review
>
> **摘要:** Current optical flow methods exploit the stable appearance of frame (or RGB) data to establish robust correspondences across time. Event cameras, on the other hand, provide high-temporal-resolution motion cues and excel in challenging scenarios. These complementary characteristics underscore the potential of integrating frame and event data for optical flow estimation. However, most cross-modal approaches fail to fully utilize the complementary advantages, relying instead on simply stacking information. This study introduces a novel approach that uses a spatially dense modality to guide the aggregation of the temporally dense event modality, achieving effective cross-modal fusion. Specifically, we propose an event-enhanced frame representation that preserves the rich texture of frames and the basic structure of events. We use the enhanced representation as the guiding modality and employ events to capture temporally dense motion information. The robust motion features derived from the guiding modality direct the aggregation of motion information from events. To further enhance fusion, we propose a transformer-based module that complements sparse event motion features with spatially rich frame information and enhances global information propagation. Additionally, a mix-fusion encoder is designed to extract comprehensive spatiotemporal contextual features from both modalities. Extensive experiments on the MVSEC and DSEC-Flow datasets demonstrate the effectiveness of our framework. Leveraging the complementary strengths of frames and events, our method achieves leading performance on the DSEC-Flow dataset. Compared to the event-only model, frame guidance improves accuracy by 10\%. Furthermore, it outperforms the state-of-the-art fusion-based method with a 4\% accuracy gain and a 45\% reduction in inference time.
>
---
#### [replaced 051] A global optimization SAR image segmentation model can be easily transformed to a general ROF denoising model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.08376v2](http://arxiv.org/pdf/2312.08376v2)**

> **作者:** Guangming Liu
>
> **备注:** 28 pages,49 figures
>
> **摘要:** In this paper, we propose a novel locally statistical active contour model (LACM) based on Aubert-Aujol (AA) denoising model and variational level set method, which can be used for SAR images segmentation with intensity inhomogeneity. Then we transform the proposed model into a global optimization model by using convex relaxation technique. Firstly, we apply the Split Bregman technique to transform the global optimization model into two alternating optimization processes of Shrink operator and Laplace operator, which is called SB_LACM model. Moreover, we propose two fast models to solve the global optimization model , which are more efficient than the SB_LACM model. The first model is: we add the proximal function to transform the global optimization model to a general ROF model[29], which can be solved by a fast denoising algorithm proposed by R.-Q.Jia, and H.Zhao; [29] is submitted on 29-Aug-2013, and our early edition ever submitted to TGRS on 12-Jun-2012, Venkatakrishnan et al. [30] proposed their 'pnp algorithm' on 29-May-2013, so Venkatakrishnan and we proposed the 'pnp algorithm' almost simultaneously. Thus we obtain a fast segmentation algorithm with global optimization solver that does not involve partial differential equations or difference equation, and only need simple difference computation. The second model is: we use a different splitting approach than one model to transform the global optimization model into a differentiable term and a general ROF model term, which can be solved by the same technique as the first model. Experiments using some challenging synthetic images and Envisat SAR images demonstrate the superiority of our proposed models with respect to the state-of-the-art models.
>
---
#### [replaced 052] Upsample What Matters: Region-Adaptive Latent Sampling for Accelerated Diffusion Transformers
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.08422v2](http://arxiv.org/pdf/2507.08422v2)**

> **作者:** Wongi Jeong; Kyungryeol Lee; Hoigi Seo; Se Young Chun
>
> **摘要:** Diffusion transformers have emerged as an alternative to U-net-based diffusion models for high-fidelity image and video generation, offering superior scalability. However, their heavy computation remains a major obstacle to real-world deployment. Existing acceleration methods primarily exploit the temporal dimension such as reusing cached features across diffusion timesteps. Here, we propose Region-Adaptive Latent Upsampling (RALU), a training-free framework that accelerates inference along spatial dimension. RALU performs mixed-resolution sampling across three stages: 1) low-resolution denoising latent diffusion to efficiently capture global semantic structure, 2) region-adaptive upsampling on specific regions prone to artifacts at full-resolution, and 3) all latent upsampling at full-resolution for detail refinement. To stabilize generations across resolution transitions, we leverage noise-timestep rescheduling to adapt the noise level across varying resolutions. Our method significantly reduces computation while preserving image quality by achieving up to 7.0$\times$ speed-up on FLUX and 3.0$\times$ on Stable Diffusion 3 with minimal degradation. Furthermore, RALU is complementary to existing temporal accelerations such as caching methods, thus can be seamlessly integrated to further reduce inference latency without compromising generation quality.
>
---
#### [replaced 053] Always Skip Attention
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01996v3](http://arxiv.org/pdf/2505.01996v3)**

> **作者:** Yiping Ji; Hemanth Saratchandran; Peyman Moghadam; Simon Lucey
>
> **备注:** This work has just been accepted by ICCV 2025
>
> **摘要:** We highlight a curious empirical result within modern Vision Transformers (ViTs). Specifically, self-attention catastrophically fails to train unless it is used in conjunction with a skip connection. This is in contrast to other elements of a ViT that continue to exhibit good performance (albeit suboptimal) when skip connections are removed. Further, we show that this critical dependence on skip connections is a relatively new phenomenon, with previous deep architectures (\eg, CNNs) exhibiting good performance in their absence. In this paper, we theoretically characterize that the self-attention mechanism is fundamentally ill-conditioned and is, therefore, uniquely dependent on skip connections for regularization. Additionally, we propose Token Graying -- a simple yet effective complement (to skip connections) that further improves the condition of input tokens. We validate our approach in both supervised and self-supervised training methods.
>
---
#### [replaced 054] MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation
- **分类: eess.IV; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.02458v3](http://arxiv.org/pdf/2410.02458v3)**

> **作者:** Gurucharan Marthi Krishna Kumar; Aman Chadha; Janine Mendola; Amir Shmuel
>
> **备注:** Accepted to the CVAMD Workshop (Computer Vision for Automated Medical Diagnosis) at the 2025 IEEE/CVF International Conference on Computer Vision (ICCVW 2025)
>
> **摘要:** Large Language Models (LLMs), known for their versatility in textual data, are increasingly being explored for their potential to enhance medical image segmentation, a crucial task for accurate diagnostic imaging. This study explores enhancing Vision Transformers (ViTs) for medical image segmentation by integrating pre-trained LLM transformer blocks. Our approach, which incorporates a frozen LLM transformer block into the encoder of a ViT-based model, leads to substantial improvements in segmentation performance across various medical imaging modalities. We propose a Hybrid Attention Mechanism that combines global and local feature learning with a Multi-Scale Fusion Block for aggregating features across different scales. The enhanced model shows significant performance gains, including an average Dice score increase from 0.74 to 0.79 and improvements in accuracy, precision, and the Jaccard Index. These results demonstrate the effectiveness of LLM-based transformers in refining medical image segmentation, highlighting their potential to significantly boost model accuracy and robustness. The source code and our implementation are available at: https://github.com/AS-Lab/Marthi-et-al-2025-MedVisionLlama-Pre-Trained-LLM-Layers-to-Enhance-Medical-Image-Segmentation
>
---
#### [replaced 055] Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2303.15564v3](http://arxiv.org/pdf/2303.15564v3)**

> **作者:** Tao Sun; Lu Pang; Weimin Lyu; Chao Chen; Haibin Ling
>
> **摘要:** Deep neural networks are vulnerable to backdoor attacks, where an adversary manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which is impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for local attacks and black-box models. The true label of every test image needs to be recovered on the fly from a suspicious model regardless of image benignity. We consider test-time image purification that incapacitates local triggers while keeping semantic contents intact. Due to diverse trigger patterns and sizes, the heuristic trigger search can be unscalable. We circumvent such barrier by leveraging the strong reconstruction power of generative models, and propose Blind Defense with Masked AutoEncoder (BDMAE). BDMAE detects possible local triggers using image structural similarity and label consistency between the test image and MAE restorations. The detection results are then refined by considering trigger topology. Finally, we fuse MAE restorations adaptively into a purified image for making prediction. Extensive experiments under different backdoor settings validate its effectiveness and generalizability.
>
---
#### [replaced 056] Automatic Image Colorization with Convolutional Neural Networks and Generative Adversarial Networks
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.05068v2](http://arxiv.org/pdf/2508.05068v2)**

> **作者:** Changyuan Qiu; Hangrui Cao; Qihan Ren; Ruiyu Li; Yuqing Qiu
>
> **备注:** All authors have equal authorship and equal contribution, ranked in alphabetic order. First version of this paper was completed and published in 2021
>
> **摘要:** Image colorization, the task of adding colors to grayscale images, has been the focus of significant research efforts in computer vision in recent years for its various application areas such as color restoration and automatic animation colorization [15, 1]. The colorization problem is challenging as it is highly ill-posed with two out of three image dimensions lost, resulting in large degrees of freedom. However, semantics of the scene as well as the surface texture could provide important cues for colors: the sky is typically blue, the clouds are typically white and the grass is typically green, and there are huge amounts of training data available for learning such priors since any colored image could serve as a training data point [20]. Colorization is initially formulated as a regression task[5], which ignores the multi-modal nature of color prediction. In this project, we explore automatic image colorization via classification and adversarial learning. We will build our models on prior works, apply modifications for our specific scenario and make comparisons.
>
---
#### [replaced 057] Vector-Quantized Vision Foundation Models for Object-Centric Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20263v5](http://arxiv.org/pdf/2502.20263v5)**

> **作者:** Rongzhen Zhao; Vivienne Wang; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Object-Centric Learning (OCL) aggregates image or video feature maps into object-level feature vectors, termed \textit{slots}. It's self-supervision of reconstructing the input from slots struggles with complex object textures, thus Vision Foundation Model (VFM) representations are used as the aggregation input and reconstruction target. Existing methods leverage VFM representations in diverse ways yet fail to fully exploit their potential. In response, we propose a unified architecture, Vector-Quantized VFMs for OCL (VQ-VFM-OCL, or VVO). The key to our unification is simply shared quantizing VFM representations in OCL aggregation and decoding. Experiments show that across different VFMs, aggregators and decoders, our VVO consistently outperforms baselines in object discovery and recognition, as well as downstream visual prediction and reasoning. We also mathematically analyze why VFM representations facilitate OCL aggregation and why their shared quantization as reconstruction targets strengthens OCL supervision. Our source code and model checkpoints are available on https://github.com/Genera1Z/VQ-VFM-OCL.
>
---
#### [replaced 058] BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification with Swin-HAFNet
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14318v2](http://arxiv.org/pdf/2506.14318v2)**

> **作者:** Amirreza Fateh; Yasin Rezvani; Sara Moayedi; Sadjad Rezvani; Fatemeh Fateh; Mansoor Fateh
>
> **摘要:** Accurate segmentation and classification of brain tumors from Magnetic Resonance Imaging (MRI) remain key challenges in medical image analysis. This is primarily due to the lack of high-quality, balanced, and diverse datasets. In this work, we present a newly developed MRI dataset named BRISC designed specifically for brain tumor segmentation and classification tasks. The dataset comprises 6,000 contrast-enhanced T1-weighted MRI scans annotated by certified radiologists and physicians. It includes three major tumor types, namely glioma, meningioma, and pituitary, as well as non-tumorous cases. Each sample includes high-resolution labels and is categorized across axial, sagittal, and coronal imaging planes to facilitate robust model development and cross-view generalization. To demonstrate the utility of the dataset, we propose a transformer-based segmentation model and benchmark it against established baselines. In this work, we propose a transformer-based model designed for both segmentation and classification of brain tumors, leveraging multi-scale feature representations from a Swin Transformer backbone. The model is benchmarked against established baselines to demonstrate the utility of the dataset, enabling accurate segmentation and robust classification across four diagnostic categories: glioma, meningioma, pituitary, and non-tumorous cases. In this work, our proposed transformer-based model demonstrates superior performance in both segmentation and classification tasks for brain tumor analysis. For the segmentation task, the method achieves the highest weighted mean Intersection-over-Union (IoU) of 82.3\%, with improvements observed across all tumor categories. For the classification task, the model attains an accuracy of 99.63\%, effectively distinguishing between glioma, meningioma, pituitary, and non-tumorous cases. https://www.kaggle.com/datasets/briscdataset/brisc2025/
>
---
#### [replaced 059] MCN-SLAM: Multi-Agent Collaborative Neural SLAM with Hybrid Implicit Neural Scene Representation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18678v2](http://arxiv.org/pdf/2506.18678v2)**

> **作者:** Tianchen Deng; Guole Shen; Xun Chen; Shenghai Yuan; Hongming Shen; Guohao Peng; Zhenyu Wu; Jingchuan Wang; Lihua Xie; Danwei Wang; Hesheng Wang; Weidong Chen
>
> **摘要:** Neural implicit scene representations have recently shown promising results in dense visual SLAM. However, existing implicit SLAM algorithms are constrained to single-agent scenarios, and fall difficulties in large-scale scenes and long sequences. Existing NeRF-based multi-agent SLAM frameworks cannot meet the constraints of communication bandwidth. To this end, we propose the first distributed multi-agent collaborative neural SLAM framework with hybrid scene representation, distributed camera tracking, intra-to-inter loop closure, and online distillation for multiple submap fusion. A novel triplane-grid joint scene representation method is proposed to improve scene reconstruction. A novel intra-to-inter loop closure method is designed to achieve local (single-agent) and global (multi-agent) consistency. We also design a novel online distillation method to fuse the information of different submaps to achieve global consistency. Furthermore, to the best of our knowledge, there is no real-world dataset for NeRF-based/GS-based SLAM that provides both continuous-time trajectories groundtruth and high-accuracy 3D meshes groundtruth. To this end, we propose the first real-world Dense slam (DES) dataset covering both single-agent and multi-agent scenarios, ranging from small rooms to large-scale outdoor scenes, with high-accuracy ground truth for both 3D mesh and continuous-time camera trajectory. This dataset can advance the development of the research in both SLAM, 3D reconstruction, and visual foundation model. Experiments on various datasets demonstrate the superiority of the proposed method in both mapping, tracking, and communication. The dataset and code will open-source on https://github.com/dtc111111/mcnslam.
>
---
#### [replaced 060] Towards Vision Zero: The TUM Traffic Accid3nD Dataset
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12095v2](http://arxiv.org/pdf/2503.12095v2)**

> **作者:** Walter Zimmer; Ross Greer; Daniel Lehmberg; Marc Pavel; Holger Caesar; Xingcheng Zhou; Ahmed Ghita; Mohan Trivedi; Rui Song; Hu Cao; Akshay Gopalkrishnan; Alois C. Knoll
>
> **备注:** Accepted for the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)
>
> **摘要:** Even though a significant amount of work has been done to increase the safety of transportation networks, accidents still occur regularly. They must be understood as unavoidable and sporadic outcomes of traffic networks. No public dataset contains 3D annotations of real-world accidents recorded from roadside camera and LiDAR sensors. We present the TUM Traffic Accid3nD (TUMTraf-Accid3nD) dataset, a collection of real-world highway accidents in different weather and lighting conditions. It contains vehicle crashes at high-speed driving with 2,634,233 labeled 2D bounding boxes, instance masks, and 3D bounding boxes with track IDs. In total, the dataset contains 111,945 labeled image and point cloud frames recorded from four roadside cameras and LiDARs at 25 Hz. The dataset contains six object classes and is provided in the OpenLABEL format. We propose an accident detection model that combines a rule-based approach with a learning-based one. Experiments and ablation studies on our dataset show the robustness of our proposed method. The dataset, model, and code are available on our website: https://accident-dataset.github.io.
>
---
#### [replaced 061] MA-CBP: A Criminal Behavior Prediction Framework Based on Multi-Agent Asynchronous Collaboration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06189v2](http://arxiv.org/pdf/2508.06189v2)**

> **作者:** Cheng Liu; Daou Zhang; Tingxu Liu; Yuhan Wang; Jinyang Chen; Yuexuan Li; Xinying Xiao; Chenbo Xin; Ziru Wang; Weichao Wu
>
> **摘要:** With the acceleration of urbanization, criminal behavior in public scenes poses an increasingly serious threat to social security. Traditional anomaly detection methods based on feature recognition struggle to capture high-level behavioral semantics from historical information, while generative approaches based on Large Language Models (LLMs) often fail to meet real-time requirements. To address these challenges, we propose MA-CBP, a criminal behavior prediction framework based on multi-agent asynchronous collaboration. This framework transforms real-time video streams into frame-level semantic descriptions, constructs causally consistent historical summaries, and fuses adjacent image frames to perform joint reasoning over long- and short-term contexts. The resulting behavioral decisions include key elements such as event subjects, locations, and causes, enabling early warning of potential criminal activity. In addition, we construct a high-quality criminal behavior dataset that provides multi-scale language supervision, including frame-level, summary-level, and event-level semantic annotations. Experimental results demonstrate that our method achieves superior performance on multiple datasets and offers a promising solution for risk warning in urban public safety scenarios.
>
---
#### [replaced 062] EmoSEM: Segment and Explain Emotion Stimuli in Visual Art
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14658v3](http://arxiv.org/pdf/2504.14658v3)**

> **作者:** Jing Zhang; Dan Guo; Zhangbin Li; Meng Wang
>
> **摘要:** This paper focuses on a key challenge in visual emotion understanding: given an art image, the model pinpoints pixel regions that trigger a specific human emotion, and generates linguistic explanations for it. Despite advances in general segmentation, pixel-level emotion understanding still faces a dual challenge: first, the subjectivity of emotion limits general segmentation models like SAM to adapt to emotion-oriented segmentation tasks; and second, the abstract nature of art expression makes it hard for captioning models to balance pixel-level semantics and emotion reasoning. To solve the above problems, this paper proposes the Emotion stimuli Segmentation and Explanation Model (EmoSEM) model to endow the segmentation framework with emotion comprehension capability. First, to enable the model to perform segmentation under the guidance of emotional intent well, we introduce an emotional prompt with a learnable mask token as the conditional input for segmentation decoding. Then, we design an emotion projector to establish the association between emotion and visual features. Next, more importantly, to address emotion-visual stimuli alignment, we develop a lightweight prefix adapter, a module that fuses the learned emotional mask with the corresponding emotion into a unified representation compatible with the language model. Finally, we input the joint visual, mask, and emotional tokens into the language model and output the emotional explanations. It ensures that the generated interpretations remain semantically and emotionally coherent with the visual stimuli. Our method realizes end-to-end modeling from low-level pixel features to high-level emotion interpretation, delivering the first interpretable fine-grained framework for visual emotion analysis. Extensive experiments validate the effectiveness of our model. Code will be made publicly available.
>
---
#### [replaced 063] UltraDfeGAN: Detail-Enhancing Generative Adversarial Networks for High-Fidelity Functional Ultrasound Synthesis
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2507.03341v2](http://arxiv.org/pdf/2507.03341v2)**

> **作者:** Zhuo Li; Xuhang Chen; Shuqiang Wang; Bin Yuan; Nou Sotheany; Ngeth Rithea
>
> **摘要:** Functional ultrasound (fUS) is a neuroimaging technique known for its high spatiotemporal resolution, enabling non-invasive observation of brain activity through neurovascular coupling. Despite its potential in clinical applications such as neonatal monitoring and intraoperative guidance, the development of fUS faces challenges related to data scarcity and limitations in generating realistic fUS images. This paper explores the use of a generative adversarial network (GAN) framework tailored for fUS image synthesis. The proposed method incorporates architectural enhancements, including feature enhancement modules and normalization techniques, aiming to improve the fidelity and physiological plausibility of generated images. The study evaluates the performance of the framework against existing generative models, demonstrating its capability to produce high-quality fUS images under various experimental conditions. Additionally, the synthesized images are assessed for their utility in downstream tasks, showing improvements in classification accuracy when used for data augmentation. Experimental results are based on publicly available fUS datasets, highlighting the framework's effectiveness in addressing data limitations.
>
---
#### [replaced 064] Vehicle detection from GSV imagery: Predicting travel behaviour for cycling and motorcycling using Computer Vision
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.12794v2](http://arxiv.org/pdf/2508.12794v2)**

> **作者:** Kyriaki; Kokka; Rahul Goel; Ali Abbas; Kerry A. Nice; Luca Martial; SM Labib; Rihuan Ke; Carola Bibiane Schönlieb; James Woodcock
>
> **摘要:** Transportation influence health by shaping exposure to physical activity, air pollution and injury risk. Comparative data on cycling and motorcycling behaviours is scarce, particularly at a global scale. Street view imagery, such as Google Street View (GSV), combined with computer vision, is a valuable resource for efficiently capturing travel behaviour data. This study demonstrates a novel approach using deep learning on street view images to estimate cycling and motorcycling levels across diverse cities worldwide. We utilized data from 185 global cities. The data on mode shares of cycling and motorcycling estimated using travel surveys or censuses. We used GSV images to detect cycles and motorcycles in sampled locations, using 8000 images per city. The YOLOv4 model, fine-tuned using images from six cities, achieved a mean average precision of 89% for detecting cycles and motorcycles. A global prediction model was developed using beta regression with city-level mode shares as outcome, with log transformed explanatory variables of counts of GSV-detected images with cycles and motorcycles, while controlling for population density. We found strong correlations between GSV motorcycle counts and motorcycle mode share (0.78) and moderate correlations between GSV cycle counts and cycling mode share (0.51). Beta regression models predicted mode shares with $R^2$ values of 0.614 for cycling and 0.612 for motorcycling, achieving median absolute errors (MDAE) of 1.3% and 1.4%, respectively. Scatterplots demonstrated consistent prediction accuracy, though cities like Utrecht and Cali were outliers. The model was applied to 60 cities globally for which we didn't have recent mode share data. We provided estimates for some cities in the Middle East, Latin America and East Asia. With computer vision, GSV images capture travel modes and activity, providing insights alongside traditional data sources.
>
---
#### [replaced 065] Image Augmentation Agent for Weakly Supervised Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20439v3](http://arxiv.org/pdf/2412.20439v3)**

> **作者:** Wangyu Wu; Xianglin Qiu; Siqi Song; Zhenhong Chen; Xiaowei Huang; Fei Ma; Jimin Xiao
>
> **备注:** Accepted at Neurocomputing 2025
>
> **摘要:** Weakly-supervised semantic segmentation (WSSS) has achieved remarkable progress using only image-level labels. However, most existing WSSS methods focus on designing new network structures and loss functions to generate more accurate dense labels, overlooking the limitations imposed by fixed datasets, which can constrain performance improvements. We argue that more diverse trainable images provides WSSS richer information and help model understand more comprehensive semantic pattern. Therefore in this paper, we introduce a novel approach called Image Augmentation Agent (IAA) which shows that it is possible to enhance WSSS from data generation perspective. IAA mainly design an augmentation agent that leverages large language models (LLMs) and diffusion models to automatically generate additional images for WSSS. In practice, to address the instability in prompt generation by LLMs, we develop a prompt self-refinement mechanism. It allow LLMs to re-evaluate the rationality of generated prompts to produce more coherent prompts. Additionally, we insert an online filter into diffusion generation process to dynamically ensure the quality and balance of generated images. Experimental results show that our method significantly surpasses state-of-the-art WSSS approaches on the PASCAL VOC 2012 and MS COCO 2014 datasets.
>
---
#### [replaced 066] Unsupervised Anomaly Detection Using Diffusion Trend Analysis for Display Inspection
- **分类: cs.CV; cs.LG; 68T45 (Primary) 68T27 (Secondary); I.2.10**

- **链接: [http://arxiv.org/pdf/2407.09578v3](http://arxiv.org/pdf/2407.09578v3)**

> **作者:** Eunwoo Kim; Un Yang; Cheol Lae Roh; Stefano Ermon
>
> **备注:** Published in the SID Digest of Technical Papers 2025 (Volume 56, Issue 1)
>
> **摘要:** Reconstruction-based anomaly detection via denoising diffusion model has limitations in determining appropriate noise parameters that can degrade anomalies while preserving normal characteristics. Also, normal regions can fluctuate considerably during reconstruction, resulting in false detection. In this paper, we propose a method to detect anomalies by analysis of reconstruction trend depending on the degree of degradation, effectively solving the both problems that impede practical application in display inspection.
>
---
#### [replaced 067] A Study of the Framework and Real-World Applications of Language Embedding for 3D Scene Understanding
- **分类: cs.GR; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05064v2](http://arxiv.org/pdf/2508.05064v2)**

> **作者:** Mahmoud Chick Zaouali; Todd Charter; Yehor Karpichev; Brandon Haworth; Homayoun Najjaran
>
> **摘要:** Gaussian Splatting has rapidly emerged as a transformative technique for real-time 3D scene representation, offering a highly efficient and expressive alternative to Neural Radiance Fields (NeRF). Its ability to render complex scenes with high fidelity has enabled progress across domains such as scene reconstruction, robotics, and interactive content creation. More recently, the integration of Large Language Models (LLMs) and language embeddings into Gaussian Splatting pipelines has opened new possibilities for text-conditioned generation, editing, and semantic scene understanding. Despite these advances, a comprehensive overview of this emerging intersection has been lacking. This survey presents a structured review of current research efforts that combine language guidance with 3D Gaussian Splatting, detailing theoretical foundations, integration strategies, and real-world use cases. We highlight key limitations such as computational bottlenecks, generalizability, and the scarcity of semantically annotated 3D Gaussian data and outline open challenges and future directions for advancing language-guided 3D scene understanding using Gaussian Splatting.
>
---
