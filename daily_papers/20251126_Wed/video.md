# 计算机视觉 cs.CV

- **最新发布 219 篇**

- **更新 125 篇**

## 最新发布

#### [new 001] GFT-GCN: Privacy-Preserving 3D Face Mesh Recognition with Spectral Diffusion
- **分类: cs.CV**

- **简介: 该论文提出GFT-GCN框架，解决3D人脸认证中生物特征模板隐私保护问题。通过结合图傅里叶变换与图卷积网络提取谱特征，并引入不可逆、可重置的谱扩散机制，实现隐私保护。采用轻量客户端-服务器架构，确保原始数据不外泄，有效平衡了识别性能与安全性。**

- **链接: [https://arxiv.org/pdf/2511.19958v1](https://arxiv.org/pdf/2511.19958v1)**

> **作者:** Hichem Felouat; Hanrui Wang; Isao Echizen
>
> **备注:** 13 pages, 8 figures, WACV 2026
>
> **摘要:** 3D face recognition offers a robust biometric solution by capturing facial geometry, providing resilience to variations in illumination, pose changes, and presentation attacks. Its strong spoof resistance makes it suitable for high-security applications, but protecting stored biometric templates remains critical. We present GFT-GCN, a privacy-preserving 3D face recognition framework that combines spectral graph learning with diffusion-based template protection. Our approach integrates the Graph Fourier Transform (GFT) and Graph Convolutional Networks (GCN) to extract compact, discriminative spectral features from 3D face meshes. To secure these features, we introduce a spectral diffusion mechanism that produces irreversible, renewable, and unlinkable templates. A lightweight client-server architecture ensures that raw biometric data never leaves the client device. Experiments on the BU-3DFE and FaceScape datasets demonstrate high recognition accuracy and strong resistance to reconstruction attacks. Results show that GFT-GCN effectively balances privacy and performance, offering a practical solution for secure 3D face authentication.
>
---
#### [new 002] Automated Monitoring of Cultural Heritage Artifacts Using Semantic Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像语义分割任务，旨在解决文化遗产雕像与纪念碑中微裂纹的自动化检测问题。通过对比不同CNN编码器的U-Net模型，在OmniCrack30k数据集上进行定量评估，并在真实场景下进行泛化能力的定性分析，验证了模型对未见文物场景的良好适应性。**

- **链接: [https://arxiv.org/pdf/2511.20541v1](https://arxiv.org/pdf/2511.20541v1)**

> **作者:** Andrea Ranieri; Giorgio Palmieri; Silvia Biasotti
>
> **备注:** Keywords: Cultural Heritage, Monitoring, Deep Learning, U-Nets, Semantic Segmentation
>
> **摘要:** This paper addresses the critical need for automated crack detection in the preservation of cultural heritage through semantic segmentation. We present a comparative study of U-Net architectures, using various convolutional neural network (CNN) encoders, for pixel-level crack identification on statues and monuments. A comparative quantitative evaluation is performed on the test set of the OmniCrack30k dataset [1] using popular segmentation metrics including Mean Intersection over Union (mIoU), Dice coefficient, and Jaccard index. This is complemented by an out-of-distribution qualitative evaluation on an unlabeled test set of real-world cracked statues and monuments. Our findings provide valuable insights into the capabilities of different CNN- based encoders for fine-grained crack segmentation. We show that the models exhibit promising generalization capabilities to unseen cultural heritage contexts, despite never having been explicitly trained on images of statues or monuments.
>
---
#### [new 003] HistoSpeckle-Net: Mutual Information-Guided Deep Learning for high-fidelity reconstruction of complex OrganAMNIST images via perturbed Multimode Fibers
- **分类: cs.CV; physics.optics**

- **简介: 该论文针对多模光纤成像中复杂医学图像重建难题，提出HistoSpeckle-Net模型。通过引入基于直方图的互信息损失与三尺度特征精炼模块，实现低数据依赖下的高保真重建，有效提升结构相似性与统计一致性，显著改善在光纤弯曲扰动下的鲁棒性，推动MMF成像向临床应用迈进。**

- **链接: [https://arxiv.org/pdf/2511.20245v1](https://arxiv.org/pdf/2511.20245v1)**

> **作者:** Jawaria Maqbool; M. Imran Cheema
>
> **摘要:** Existing deep learning methods in multimode fiber (MMF) imaging often focus on simpler datasets, limiting their applicability to complex, real-world imaging tasks. These models are typically data-intensive, a challenge that becomes more pronounced when dealing with diverse and complex images. In this work, we propose HistoSpeckle-Net, a deep learning architecture designed to reconstruct structurally rich medical images from MMF speckles. To build a clinically relevant dataset, we develop an optical setup that couples laser light through a spatial light modulator (SLM) into an MMF, capturing output speckle patterns corresponding to input OrganAMNIST images. Unlike previous MMF imaging approaches, which have not considered the underlying statistics of speckles and reconstructed images, we introduce a distribution-aware learning strategy. We employ a histogram-based mutual information loss to enhance model robustness and reduce reliance on large datasets. Our model includes a histogram computation unit that estimates smooth marginal and joint histograms for calculating mutual information loss. It also incorporates a unique Three-Scale Feature Refinement Module, which leads to multiscale Structural Similarity Index Measure (SSIM) loss computation. Together, these two loss functions enhance both the structural fidelity and statistical alignment of the reconstructed images. Our experiments on the complex OrganAMNIST dataset demonstrate that HistoSpeckle-Net achieves higher fidelity than baseline models such as U-Net and Pix2Pix. It gives superior performance even with limited training samples and across varying fiber bending conditions. By effectively reconstructing complex anatomical features with reduced data and under fiber perturbations, HistoSpeckle-Net brings MMF imaging closer to practical deployment in real-world clinical environments.
>
---
#### [new 004] Leveraging Foundation Models for Histological Grading in Cutaneous Squamous Cell Carcinoma using PathFMTools
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对皮肤鳞状细胞癌（cSCC）的组织学分级任务，解决基础模型在病理图像处理中适应难、可解释性差的问题。提出PathFMTools工具包，评估CONCH与MUSK模型在440例WSI上的表现，验证了利用基础模型嵌入训练小型专用模型的有效性，推动其临床应用。**

- **链接: [https://arxiv.org/pdf/2511.19751v1](https://arxiv.org/pdf/2511.19751v1)**

> **作者:** Abdul Rahman Diab; Emily E. Karn; Renchin Wu; Emily S. Ruiz; William Lotter
>
> **备注:** Proceedings of the 5th Machine Learning for Health (ML4H) Symposium (2025)
>
> **摘要:** Despite the promise of computational pathology foundation models, adapting them to specific clinical tasks remains challenging due to the complexity of whole-slide image (WSI) processing, the opacity of learned features, and the wide range of potential adaptation strategies. To address these challenges, we introduce PathFMTools, a lightweight, extensible Python package that enables efficient execution, analysis, and visualization of pathology foundation models. We use this tool to interface with and evaluate two state-of-the-art vision-language foundation models, CONCH and MUSK, on the task of histological grading in cutaneous squamous cell carcinoma (cSCC), a critical criterion that informs cSCC staging and patient management. Using a cohort of 440 cSCC H&E WSIs, we benchmark multiple adaptation strategies, demonstrating trade-offs across prediction approaches and validating the potential of using foundation model embeddings to train small specialist models. These findings underscore the promise of pathology foundation models for real-world clinical applications, with PathFMTools enabling efficient analysis and validation.
>
---
#### [new 005] The Determinant Ratio Matrix Approach to Solving 3D Matching and 2D Orthographic Projection Alignment Tasks
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对3D姿态估计中的全姿态（EnP）与正交投影（OnP）任务，提出基于行列式比矩阵（DRaM）的解析解法。解决了无噪声情况下3D-3D与3D-2D正交投影对齐的最小二乘问题，并揭示了其可追溯至高斯时代且可推广至N维欧氏空间的理论深度。**

- **链接: [https://arxiv.org/pdf/2511.19511v1](https://arxiv.org/pdf/2511.19511v1)**

> **作者:** Andrew J. Hanson; Sonya M. Hanson
>
> **备注:** 12 pages of main text, 3 figures, 31 pages total (including references and 2 appendices, one with algorithm-defining source code)
>
> **摘要:** Pose estimation is a general problem in computer vision with wide applications. The relative orientation of a 3D reference object can be determined from a 3D rotated version of that object, or from a projection of the rotated object to a 2D planar image. This projection can be a perspective projection (the PnP problem) or an orthographic projection (the OnP problem). We restrict our attention here to the OnP problem and the full 3D pose estimation task (the EnP problem). Here we solve the least squares systems for both the error-free EnP and OnP problems in terms of the determinant ratio matrix (DRaM) approach. The noisy-data case can be addressed with a straightforward rotation correction scheme. While the SVD and optimal quaternion eigensystem methods solve the noisy EnP 3D-3D alignment exactly, the noisy 3D-2D orthographic (OnP) task has no known comparable closed form, and can be solved by DRaM-class methods. We note that while previous similar work has been presented in the literature exploiting both the QR decomposition and the Moore-Penrose pseudoinverse transformations, here we place these methods in a larger context that has not previously been fully recognized in the absence of the corresponding DRaM solution. We term this class of solutions as the DRaM family, and conduct comparisons of the behavior of the families of solutions for the EnP and OnP rotation estimation problems. Overall, this work presents both a new solution to the 3D and 2D orthographic pose estimation problems and provides valuable insight into these classes of problems. With hindsight, we are able to show that our DRaM solutions to the exact EnP and OnP problems possess derivations that could have been discovered in the time of Gauss, and in fact generalize to all analogous N-dimensional Euclidean pose estimation problems.
>
---
#### [new 006] PuzzlePoles: Cylindrical Fiducial Markers Based on the PuzzleBoard Pattern
- **分类: cs.CV**

- **简介: 该论文提出PuzzlePole，一种基于PuzzleBoard图案的圆柱形视觉标记。针对自主系统中环境感知的校准与定位需求，解决传统标记在多视角下识别不可靠、易受遮挡影响的问题。通过利用PuzzleBoard的组合结构，实现360°视角下的高精度定位与姿态估计，提升系统鲁棒性与部署灵活性。**

- **链接: [https://arxiv.org/pdf/2511.19448v1](https://arxiv.org/pdf/2511.19448v1)**

> **作者:** Juri Zach; Peer Stelldinger
>
> **摘要:** Reliable perception of the environment is a key enabler for autonomous systems, where calibration and localization tasks often rely on robust visual markers. We introduce the PuzzlePole, a new type of fiducial markers derived from the recently proposed PuzzleBoard calibration pattern. The PuzzlePole is a cylindrical marker, enabling reliable recognition and pose estimation from 360° viewing direction. By leveraging the unique combinatorial structure of the PuzzleBoard pattern, PuzzlePoles provide a high accuracy in localization and orientation while being robust to occlusions. The design offers flexibility for deployment in diverse autonomous systems scenarios, ranging from robot navigation and SLAM to tangible interfaces.
>
---
#### [new 007] Image Diffusion Models Exhibit Emergent Temporal Propagation in Videos
- **分类: cs.CV**

- **简介: 该论文研究图像扩散模型在视频中的时序传播能力，旨在解决零样本视频目标分割与跟踪问题。通过重新解释自注意力图为语义传播核，结合测试时优化策略，提出DRIFT框架，利用预训练扩散模型与SAM引导的掩码优化，实现优异的零样本跟踪性能。**

- **链接: [https://arxiv.org/pdf/2511.19936v1](https://arxiv.org/pdf/2511.19936v1)**

> **作者:** Youngseo Kim; Dohyun Kim; Geohee Han; Paul Hongsuck Seo
>
> **摘要:** Image diffusion models, though originally developed for image generation, implicitly capture rich semantic structures that enable various recognition and localization tasks beyond synthesis. In this work, we investigate their self-attention maps can be reinterpreted as semantic label propagation kernels, providing robust pixel-level correspondences between relevant image regions. Extending this mechanism across frames yields a temporal propagation kernel that enables zero-shot object tracking via segmentation in videos. We further demonstrate the effectiveness of test-time optimization strategies-DDIM inversion, textual inversion, and adaptive head weighting-in adapting diffusion features for robust and consistent label propagation. Building on these findings, we introduce DRIFT, a framework for object tracking in videos leveraging a pretrained image diffusion model with SAM-guided mask refinement, achieving state-of-the-art zero-shot performance on standard video object segmentation benchmarks.
>
---
#### [new 008] VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild
- **分类: cs.CV**

- **简介: 该论文针对真实场景下人脸几何重建中拓扑不一致、泛化性差的问题，提出VGGTFace方法。利用3D基础模型VGGT生成点图，并结合Pixel3DMM注入像素对齐的UV坐标以恢复拓扑结构，进而通过拓扑感知束调整融合多视角信息，实现快速高精度重建。**

- **链接: [https://arxiv.org/pdf/2511.20366v1](https://arxiv.org/pdf/2511.20366v1)**

> **作者:** Xin Ming; Yuxuan Han; Tianyu Huang; Feng Xu
>
> **摘要:** Reconstructing topologically consistent facial geometry is crucial for the digital avatar creation pipelines. Existing methods either require tedious manual efforts, lack generalization to in-the-wild data, or are constrained by the limited expressiveness of 3D Morphable Models. To address these limitations, we propose VGGTFace, an automatic approach that innovatively applies the 3D foundation model, \emph{i.e.} VGGT, for topologically consistent facial geometry reconstruction from in-the-wild multi-view images captured by everyday users. Our key insight is that, by leveraging VGGT, our method naturally inherits strong generalization ability and expressive power from its large-scale training and point map representation. However, it is unclear how to reconstruct a topologically consistent mesh from VGGT, as the topology information is missing in its prediction. To this end, we augment VGGT with Pixel3DMM for injecting topology information via pixel-aligned UV values. In this manner, we convert the pixel-aligned point map of VGGT to a point cloud with topology. Tailored to this point cloud with known topology, we propose a novel Topology-Aware Bundle Adjustment strategy to fuse them, where we construct a Laplacian energy for the Bundle Adjustment objective. Our method achieves high-quality reconstruction in 10 seconds for 16 views on a single NVIDIA RTX 4090. Experiments demonstrate state-of-the-art results on benchmarks and impressive generalization to in-the-wild data. Code is available at https://github.com/grignarder/vggtface.
>
---
#### [new 009] GS-Checker: Tampering Localization for 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对3D高斯溅射（3DGS）内容易被恶意篡改的问题，提出GS-Checker方法，通过在3D高斯参数中嵌入篡改属性，结合3D对比机制与循环优化策略，实现无需3D标签的篡改区域定位，有效提升3DGS模型的可信性。**

- **链接: [https://arxiv.org/pdf/2511.20354v1](https://arxiv.org/pdf/2511.20354v1)**

> **作者:** Haoliang Han; Ziyuan Luo; Jun Qi; Anderson Rocha; Renjie Wan
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Recent advances in editing technologies for 3D Gaussian Splatting (3DGS) have made it simple to manipulate 3D scenes. However, these technologies raise concerns about potential malicious manipulation of 3D content. To avoid such malicious applications, localizing tampered regions becomes crucial. In this paper, we propose GS-Checker, a novel method for locating tampered areas in 3DGS models. Our approach integrates a 3D tampering attribute into the 3D Gaussian parameters to indicate whether the Gaussian has been tampered. Additionally, we design a 3D contrastive mechanism by comparing the similarity of key attributes between 3D Gaussians to seek tampering cues at 3D level. Furthermore, we introduce a cyclic optimization strategy to refine the 3D tampering attribute, enabling more accurate tampering localization. Notably, our approach does not require expensive 3D labels for supervision. Extensive experimental results demonstrate the effectiveness of our proposed method to locate the tampered 3DGS area.
>
---
#### [new 010] SkillSight: Efficient First-Person Skill Assessment with Gaze
- **分类: cs.CV**

- **简介: 该论文提出SkillSight，用于高效的第一人称技能评估。针对智能眼镜上持续视频处理功耗高的问题，提出结合眼动与视频的两阶段框架，先联合建模，再蒸馏出仅需眼动数据的轻量模型。实验证明眼动对技能理解至关重要，所提方法在保持高精度的同时降低73倍功耗，推动了真实场景下的AI辅助学习。**

- **链接: [https://arxiv.org/pdf/2511.19629v1](https://arxiv.org/pdf/2511.19629v1)**

> **作者:** Chi Hsuan Wu; Kumar Ashutosh; Kristen Grauman
>
> **摘要:** Egocentric perception on smart glasses could transform how we learn new skills in the physical world, but automatic skill assessment remains a fundamental technical challenge. We introduce SkillSight for power-efficient skill assessment from first-person data. Central to our approach is the hypothesis that skill level is evident not only in how a person performs an activity (video), but also in how they direct their attention when doing so (gaze). Our two-stage framework first learns to jointly model gaze and egocentric video when predicting skill level, then distills a gaze-only student model. At inference, the student model requires only gaze input, drastically reducing power consumption by eliminating continuous video processing. Experiments on three datasets spanning cooking, music, and sports establish, for the first time, the valuable role of gaze in skill understanding across diverse real-world settings. Our SkillSight teacher model achieves state-of-the-art performance, while our gaze-only student variant maintains high accuracy using 73x less power than competing methods. These results pave the way for in-the-wild AI-supported skill learning.
>
---
#### [new 011] The Image as Its Own Reward: Reinforcement Learning with Adversarial Reward for Image Generation
- **分类: cs.CV**

- **简介: 该论文针对图像生成中奖励函数不可靠的问题，提出基于对抗性奖励的强化学习框架Adv-GRPO。通过将图像自身作为奖励，利用视觉基础模型提供密集视觉信号，避免奖励黑客，提升图像质量与美学表现，实现风格可控生成。**

- **链接: [https://arxiv.org/pdf/2511.20256v1](https://arxiv.org/pdf/2511.20256v1)**

> **作者:** Weijia Mao; Hao Chen; Zhenheng Yang; Mike Zheng Shou
>
> **摘要:** A reliable reward function is essential for reinforcement learning (RL) in image generation. Most current RL approaches depend on pre-trained preference models that output scalar rewards to approximate human preferences. However, these rewards often fail to capture human perception and are vulnerable to reward hacking, where higher scores do not correspond to better images. To address this, we introduce Adv-GRPO, an RL framework with an adversarial reward that iteratively updates both the reward model and the generator. The reward model is supervised using reference images as positive samples and can largely avoid being hacked. Unlike KL regularization that constrains parameter updates, our learned reward directly guides the generator through its visual outputs, leading to higher-quality images. Moreover, while optimizing existing reward functions can alleviate reward hacking, their inherent biases remain. For instance, PickScore may degrade image quality, whereas OCR-based rewards often reduce aesthetic fidelity. To address this, we take the image itself as a reward, using reference images and vision foundation models (e.g., DINO) to provide rich visual rewards. These dense visual signals, instead of a single scalar, lead to consistent gains across image quality, aesthetics, and task-specific metrics. Finally, we show that combining reference samples with foundation-model rewards enables distribution transfer and flexible style customization. In human evaluation, our method outperforms Flow-GRPO and SD3, achieving 70.0% and 72.4% win rates in image quality and aesthetics, respectively. Code and models have been released.
>
---
#### [new 012] Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对真实世界中单目视频下乒乓球3D轨迹与旋转估计难题，提出两阶段框架：前端用新数据集TTHQ训练2D检测，后端基于物理合成数据训练鲁棒的3D还原模型，有效应对实际中的检测缺失与帧率变化，实现高精度、实用化的3D运动分析。**

- **链接: [https://arxiv.org/pdf/2511.20250v1](https://arxiv.org/pdf/2511.20250v1)**

> **作者:** Daniel Kienzle; Katja Ludwig; Julian Lorenz; Shin'ichi Satoh; Rainer Lienhart
>
> **摘要:** Obtaining the precise 3D motion of a table tennis ball from standard monocular videos is a challenging problem, as existing methods trained on synthetic data struggle to generalize to the noisy, imperfect ball and table detections of the real world. This is primarily due to the inherent lack of 3D ground truth trajectories and spin annotations for real-world video. To overcome this, we propose a novel two-stage pipeline that divides the problem into a front-end perception task and a back-end 2D-to-3D uplifting task. This separation allows us to train the front-end components with abundant 2D supervision from our newly created TTHQ dataset, while the back-end uplifting network is trained exclusively on physically-correct synthetic data. We specifically re-engineer the uplifting model to be robust to common real-world artifacts, such as missing detections and varying frame rates. By integrating a ball detector and a table keypoint detector, our approach transforms a proof-of-concept uplifting method into a practical, robust, and high-performing end-to-end application for 3D table tennis trajectory and spin analysis.
>
---
#### [new 013] ScenarioCLIP: Pretrained Transferable Visual Language Models and Action-Genome Dataset for Natural Scene Analysis
- **分类: cs.CV**

- **简介: 该论文针对真实场景中多对象、多关系的复杂结构分析问题，提出ScenarioCLIP模型与Action-Genome数据集。通过引入关系标注和聚焦区域，增强视觉语言模型对场景的细粒度理解能力。在自建数据集上预训练并微调，显著提升零样本与微调性能，推动自然场景分析发展。**

- **链接: [https://arxiv.org/pdf/2511.20274v1](https://arxiv.org/pdf/2511.20274v1)**

> **作者:** Advik Sinha; Saurabh Atreya; Aashutosh A; Sk Aziz Ali; Abhijit Das
>
> **摘要:** Until recently, the general corpus of CLIP-type fundamental models has widely explored either the retrieval of short descriptions or the classification of objects in the scene as SINGLE-object image classification task. The same holds for retrieving the image embedding (image retrieval task) given a text prompt. However, real-world scene images exhibit rich compositional structure involving multiple objects and actions. The latest methods in the CLIP-based literature improve class-level discrimination by mining harder negative image-text pairs and by refining permanent text prompts, often using LLMs. However, these improvements remain confined to predefined class lists and do not explicitly model relational or compositional structure. PyramidCLIP partially addresses this gap by aligning global and local visual features, yet it still lacks explicit modeling of inter-object relations. Hence, to further leverage this aspect for scene analysis, the proposed ScenarioCLIP model accepts input texts, grounded relations, and input images, along with focused regions highlighting relations. The proposed model is pretrained on curated scenario data, and finetuned for specialized downstream tasks, such as cross-modal retrieval and fine-grained visual understanding tasks. To address the lack of domain-specific datasets, we generate a novel dataset by extending image-text pairs from existing diverse indoor and outdoor scenario datasets that are publicly available. We used a pipeline of existing language models to ground action, object, and relations, filled by manual and automatic curation. We established a comprehensive benchmark for several scenario-based tasks and compared it with many baseline methods. ScenarioCLIP demonstrates robust zero-shot and finetune performance on various domain-specific tasks. Our code and dataset are available at https://github.com/scenario-clip/ScenarioCLIP
>
---
#### [new 014] SFA: Scan, Focus, and Amplify toward Guidance-aware Answering for Video TextVQA
- **分类: cs.CV**

- **简介: 该论文针对视频文本视觉问答（Video TextVQA）任务，解决模型在复杂场景下感知与理解多变文本、整合时空语义及聚焦关键信息的难题。提出SFA框架，通过扫描、聚焦、放大三步，引导视频大模型注意力至关键文本线索，实现无需训练的精准回答，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.20190v1](https://arxiv.org/pdf/2511.20190v1)**

> **作者:** Haibin He; Qihuang Zhong; Juhua Liu; Bo Du; Peng Wang; Jing Zhang
>
> **摘要:** Video text-based visual question answering (Video TextVQA) task aims to answer questions about videos by leveraging the visual text appearing within the videos. This task poses significant challenges, requiring models to accurately perceive and comprehend scene text that varies in scale, orientation, and clarity across frames, while effectively integrating temporal and semantic context to generate precise answers. Moreover, the model must identify question-relevant textual cues and filter out redundant or irrelevant information to ensure answering is guided by the most relevant and informative cues. To address these challenges, we propose SFA, a training-free framework and the first Video-LLM-based method tailored for Video TextVQA, motivated by the human process of answering questions. By adaptively scanning video frames, selectively focusing on key regions, and directly amplifying them, SFA effectively guides the Video-LLM's attention toward essential cues, enabling it to generate more accurate answers. SFA achieves new state-of-the-art results across several public Video TextVQA datasets and surpasses previous methods by a substantial margin, demonstrating its effectiveness and generalizability.
>
---
#### [new 015] WPT: World-to-Policy Transfer via Online World Model Distillation
- **分类: cs.CV**

- **简介: 该论文提出WPT框架，解决世界模型与策略间在线迁移的效率与实时性问题。通过在线蒸馏，将世界模型预测能力注入轻量策略，实现高精度、低延迟的决策。在开放与闭环任务中均超越现有方法，推理速度提升4.9倍。**

- **链接: [https://arxiv.org/pdf/2511.20095v1](https://arxiv.org/pdf/2511.20095v1)**

> **作者:** Guangfeng Jiang; Yueru Luo; Jun Liu; Yi Huang; Yiyao Zhu; Zhan Qu; Dave Zhenyu Chen; Bingbing Liu; Xu Yan
>
> **摘要:** Recent years have witnessed remarkable progress in world models, which primarily aim to capture the spatio-temporal correlations between an agent's actions and the evolving environment. However, existing approaches often suffer from tight runtime coupling or depend on offline reward signals, resulting in substantial inference overhead or hindering end-to-end optimization. To overcome these limitations, we introduce WPT, a World-to-Policy Transfer training paradigm that enables online distillation under the guidance of an end-to-end world model. Specifically, we develop a trainable reward model that infuses world knowledge into a teacher policy by aligning candidate trajectories with the future dynamics predicted by the world model. Subsequently, we propose policy distillation and world reward distillation to transfer the teacher's reasoning ability into a lightweight student policy, enhancing planning performance while preserving real-time deployability. Extensive experiments on both open-loop and closed-loop benchmarks show that our WPT achieves state-of-the-art performance with a simple policy architecture: it attains a 0.11 collision rate (open-loop) and achieves a 79.23 driving score (closed-loop) surpassing both world-model-based and imitation-learning methods in accuracy and safety. Moreover, the student sustains up to 4.9x faster inference, while retaining most of the gains.
>
---
#### [new 016] Modular Deep Learning Framework for Assistive Perception: Gaze, Affect, and Speaker Identification
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对辅助感知中多模态信息融合难题，提出模块化深度学习框架，分别解决眼状态检测、表情识别与语音说话人识别任务。通过独立训练CNN、CNN与LSTM模型，在多数据集上实现高精度，验证了轻量级模块在资源受限设备中实时融合的可行性。**

- **链接: [https://arxiv.org/pdf/2511.20474v1](https://arxiv.org/pdf/2511.20474v1)**

> **作者:** Akshit Pramod Anchan; Jewelith Thomas; Sritama Roy
>
> **备注:** 10 pages, 9 figures, and 3 tables
>
> **摘要:** Developing comprehensive assistive technologies requires the seamless integration of visual and auditory perception. This research evaluates the feasibility of a modular architecture inspired by core functionalities of perceptive systems like 'Smart Eye.' We propose and benchmark three independent sensing modules: a Convolutional Neural Network (CNN) for eye state detection (drowsiness/attention), a deep CNN for facial expression recognition, and a Long Short-Term Memory (LSTM) network for voice-based speaker identification. Utilizing the Eyes Image, FER2013, and customized audio datasets, our models achieved accuracies of 93.0%, 97.8%, and 96.89%, respectively. This study demonstrates that lightweight, domain-specific models can achieve high fidelity on discrete tasks, establishing a validated foundation for future real-time, multimodal integration in resource-constrained assistive devices.
>
---
#### [new 017] iMontage: Unified, Versatile, Highly Dynamic Many-to-many Image Generation
- **分类: cs.CV**

- **简介: 该论文提出iMontage，一个统一的多对多图像生成框架。针对视频模型动态受限的问题，通过引入图像数据多样性，使预训练视频模型能生成多样且连贯的图像序列。工作包括轻量适配策略、数据优化与训练范式，实现多种图像生成与编辑任务，保持上下文一致并拓展动态范围。**

- **链接: [https://arxiv.org/pdf/2511.20635v1](https://arxiv.org/pdf/2511.20635v1)**

> **作者:** Zhoujie Fu; Xianfang Zeng; Jinghong Lan; Xinyao Liao; Cheng Chen; Junyi Chen; Jiacheng Wei; Wei Cheng; Shiyu Liu; Yunuo Chen; Gang Yu; Guosheng Lin
>
> **摘要:** Pre-trained video models learn powerful priors for generating high-quality, temporally coherent content. While these models excel at temporal coherence, their dynamics are often constrained by the continuous nature of their training data. We hypothesize that by injecting the rich and unconstrained content diversity from image data into this coherent temporal framework, we can generate image sets that feature both natural transitions and a far more expansive dynamic range. To this end, we introduce iMontage, a unified framework designed to repurpose a powerful video model into an all-in-one image generator. The framework consumes and produces variable-length image sets, unifying a wide array of image generation and editing tasks. To achieve this, we propose an elegant and minimally invasive adaptation strategy, complemented by a tailored data curation process and training paradigm. This approach allows the model to acquire broad image manipulation capabilities without corrupting its invaluable original motion priors. iMontage excels across several mainstream many-in-many-out tasks, not only maintaining strong cross-image contextual consistency but also generating scenes with extraordinary dynamics that surpass conventional scopes. Find our homepage at: https://kr1sjfu.github.io/iMontage-web/.
>
---
#### [new 018] Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding
- **分类: cs.CV**

- **简介: 该论文针对视频理解中的程序性推理任务，解决抽象描述与视觉细节对齐困难的问题。提出TSS框架，引入可观察的“状态”作为视觉锚点，通过分层预训练强化状态-步骤-任务的结构化表示，显著提升任务识别、步骤预测等下游性能。**

- **链接: [https://arxiv.org/pdf/2511.20073v1](https://arxiv.org/pdf/2511.20073v1)**

> **作者:** Jinghan Zhao; Yifei Huang; Feng Lu
>
> **备注:** Accepted by AAAI 2026. 15 pages, 12 figures
>
> **摘要:** Learning procedural-aware video representations is a key step towards building agents that can reason about and execute complex tasks. Existing methods typically address this problem by aligning visual content with textual descriptions at the task and step levels to inject procedural semantics into video representations. However, due to their high level of abstraction, 'task' and 'step' descriptions fail to form a robust alignment with the concrete, observable details in visual data. To address this, we introduce 'states', i.e., textual snapshots of object configurations, as a visually-grounded semantic layer that anchors abstract procedures to what a model can actually see. We formalize this insight in a novel Task-Step-State (TSS) framework, where tasks are achieved via steps that drive transitions between observable states. To enforce this structure, we propose a progressive pre-training strategy that unfolds the TSS hierarchy, forcing the model to ground representations in states while associating them with steps and high-level tasks. Extensive experiments on the COIN and CrossTask datasets show that our method outperforms baseline models on multiple downstream tasks, including task recognition, step recognition, and next step prediction. Ablation studies show that introducing state supervision is a key driver of performance gains across all tasks. Additionally, our progressive pretraining strategy proves more effective than standard joint training, as it better enforces the intended hierarchical structure.
>
---
#### [new 019] Multi Head Attention Enhanced Inception v3 for Cardiomegaly Detection
- **分类: cs.CV**

- **简介: 该论文针对心血管疾病中的心影增大（cardiomegaly）自动检测问题，提出一种融合多头注意力机制的Inception V3深度学习模型。通过高质量标注的X光图像数据集，结合图像预处理与注意力增强，提升了特征提取与诊断精度，实验显示模型在准确率、召回率等指标上均表现优异，具有临床应用潜力。**

- **链接: [https://arxiv.org/pdf/2511.20101v1](https://arxiv.org/pdf/2511.20101v1)**

> **作者:** Abishek Karthik; Pandiyaraju V
>
> **摘要:** The healthcare industry has been revolutionized significantly by novel imaging technologies, not just in the diagnosis of cardiovascular diseases but also by the visualization of structural abnormalities like cardiomegaly. This article explains an integrated approach to the use of deep learning tools and attention mechanisms for automatic detection of cardiomegaly using X-ray images. The initiation of the project is grounded on a strong Data Collection phase and gathering the data of annotated X-ray images of various types. Then, while the Preprocessing module fine-tunes image quality, it is feasible to utilize the best out of the data quality in the proposed system. In our proposed system, the process is a CNN configuration leveraging the inception V3 model as one of the key blocks. Besides, we also employ a multilayer attention mechanism to enhance the strength. The most important feature of the method is the multi-head attention mechanism that can learn features automatically. By exact selective focusing on only some regions of input, the model can thus identify cardiomegaly in a sensitive manner. Attention rating is calculated, duplicated, and applied to enhance representation of main data, and therefore there is a successful diagnosis. The Evaluation stage will be extremely strict and it will thoroughly evaluate the model based on such measures as accuracy and precision. This will validate that the model can identify cardiomegaly and will also show the clinical significance of this method. The model has accuracy of 95.6, precision of 95.2, recall of 96.2, sensitivity of 95.7, specificity of 96.1 and an Area Under Curve(AUC) of 96.0 and their respective graphs are plotted for visualisation.
>
---
#### [new 020] VQ-VA World: Towards High-Quality Visual Question-Visual Answering
- **分类: cs.CV**

- **简介: 该论文研究视觉问答生成图像的任务（VQ-VA），旨在让开源模型具备如闭源系统般生成图像回答的能力。提出VQ-VA World数据框架，通过智能爬虫构建180万高质量图文数据，并发布Human-curated基准IntelligentBench。实验表明，基于该数据训练的LightFusion模型性能显著提升，大幅缩小与领先闭源系统差距。**

- **链接: [https://arxiv.org/pdf/2511.20573v1](https://arxiv.org/pdf/2511.20573v1)**

> **作者:** Chenhui Gou; Zilong Chen; Zeyu Wang; Feng Li; Deyao Zhu; Zicheng Duan; Kunchang Li; Chaorui Deng; Hongyi Yuan; Haoqi Fan; Cihang Xie; Jianfei Cai; Hamid Rezatofighi
>
> **摘要:** This paper studies Visual Question-Visual Answering (VQ-VA): generating an image, rather than text, in response to a visual question -- an ability that has recently emerged in proprietary systems such as NanoBanana and GPT-Image. To also bring this capability to open-source models, we introduce VQ-VA World, a data-centric framework built around an agentic pipeline for large-scale, targeted data construction. Leveraging web-scale deployment, this pipeline crawls a massive amount of ~1.8M high-quality, interleaved image-text samples for model training. For evaluation, we further release IntelligentBench, a human-curated benchmark that systematically assesses VQ-VA along the aspects of world knowledge, design knowledge, and reasoning. Training with VQ-VA World data yields strong empirical gains: it helps LightFusion attain 53.06 on IntelligentBench, substantially surpassing the best prior open-source baselines (i.e., 7.78 from vanilla LightFusion; 1.94 from UniWorld-V1), and significantly narrowing the gap toward leading proprietary systems (e.g., 81.67 from NanoBanana; 82.64 from GPT-Image). By releasing the full suite of model weights, datasets, and pipelines, we hope to stimulate future research on VQ-VA.
>
---
#### [new 021] SONIC: Spectral Optimization of Noise for Inpainting with Consistency
- **分类: cs.CV**

- **简介: 该论文提出SONIC，一种无需训练的图像修复方法。针对通用文本到图像模型在修复任务中效果不佳的问题，通过优化初始噪声（在频域进行线性近似），使其生成结果与未掩码区域一致，仅需数十步优化即可显著提升修复质量，超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.19985v1](https://arxiv.org/pdf/2511.19985v1)**

> **作者:** Seungyeon Baek; Erqun Dong; Shadan Namazifard; Mark J. Matthews; Kwang Moo Yi
>
> **摘要:** We propose a novel training-free method for inpainting with off-the-shelf text-to-image models. While guidance-based methods in theory allow generic models to be used for inverse problems such as inpainting, in practice, their effectiveness is limited, leading to the necessity of specialized inpainting-specific models. In this work, we argue that the missing ingredient for training-free inpainting is the optimization (guidance) of the initial seed noise. We propose to optimize the initial seed noise to approximately match the unmasked parts of the data - with as few as a few tens of optimization steps. We then apply conventional training-free inpainting methods on top of our optimized initial seed noise. Critically, we propose two core ideas to effectively implement this idea: (i) to avoid the costly unrolling required to relate the initial noise and the generated outcome, we perform linear approximation; and (ii) to stabilize the optimization, we optimize the initial seed noise in the spectral domain. We demonstrate the effectiveness of our method on various inpainting tasks, outperforming the state of the art. Project page: https://ubc-vision.github.io/sonic/
>
---
#### [new 022] While recognizing actions, LMMs struggle to detect core interaction events
- **分类: cs.CV; cs.AI; q-bio.NC**

- **简介: 该论文研究视频中动作交互事件的检测任务，旨在解决大模型在识别动作时缺乏对物理接触时刻与位置的精准感知问题。研究构建了首个包含2万+交互事件标注的大规模数据集，通过对比Qwen-2.5VL和GPT-4o模型在定位接触/释放事件上的表现，发现其虽能描述动作，却无法准确判断事件起止帧与空间位置，表明模型缺乏对视觉输入的深层感知锚定。**

- **链接: [https://arxiv.org/pdf/2511.20162v1](https://arxiv.org/pdf/2511.20162v1)**

> **作者:** Daniel Harari; Michael Sidorov; Liel David; Chen Shterental; Abrham Kahsay Gebreselasie; Muhammad Haris Khan
>
> **摘要:** Large multi-modal models (LMMs) show increasing performance in realistic visual tasks for images and, more recently, for videos. For example, given a video sequence, such models are able to describe in detail objects, the surroundings and dynamic actions. In this study, we explored the extent to which these models ground their semantic understanding in the actual visual input. Specifically, given sequences of hands interacting with objects, we asked models when and where the interaction begins or ends. For this purpose, we introduce a first of its kind, large-scale dataset with more than 20K annotated interactions on videos from the Something-Something-V2 dataset. 250 AMTurk human annotators labeled core interaction events, particularly when and where objects and agents become attached ('contact') or detached ('release'). We asked two LMMs (Qwen-2.5VL and GPT-4o) to locate these events in short videos, each with a single event. The results show that although the models can reliably name the target objects, identify the action and provide coherent reasoning, they consistently fail to identify the frame where the interaction begins or ends and cannot localize the event within the scene. Our findings suggest that in struggling to pinpoint the moment and location of physical contact that defines the interaction, the models lack the perceptual grounding required for deeper understanding of dynamic scenes.
>
---
#### [new 023] OncoVision: Integrating Mammography and Clinical Data through Attention-Driven Multimodal AI for Enhanced Breast Cancer Diagnosis
- **分类: cs.CV**

- **简介: 该论文提出OncoVision，一种融合乳腺钼靶影像与临床数据的多模态AI系统，用于提升乳腺癌诊断准确性。针对传统诊断中影像与临床信息割裂、依赖经验导致的偏差问题，其通过注意力机制联合分割病灶并预测临床特征，采用双晚融合策略增强决策可靠性。系统可生成带置信度与可视化热图的结构化报告，支持实时辅助诊断，助力基层医疗普及。**

- **链接: [https://arxiv.org/pdf/2511.19667v1](https://arxiv.org/pdf/2511.19667v1)**

> **作者:** Istiak Ahmed; Galib Ahmed; K. Shahriar Sanjid; Md. Tanzim Hossain; Md. Nishan Khan; Md. Misbah Khan; Md. Arifur Rahman; Sheikh Anisul Haque; Sharmin Akhtar Rupa; Mohammed Mejbahuddin Mia; Mahmud Hasan Mostofa Kamal; Md. Mostafa Kamal Sarker; M. Monir Uddin
>
> **摘要:** OncoVision is a multimodal AI pipeline that combines mammography images and clinical data for better breast cancer diagnosis. Employing an attention-based encoder-decoder backbone, it jointly segments four ROIs - masses, calcifications, axillary findings, and breast tissues - with state-of-the-art accuracy and robustly predicts ten structured clinical features: mass morphology, calcification type, ACR breast density, and BI-RADS categories. To fuse imaging and clinical insights, we developed two late-fusion strategies. By utilizing complementary multimodal data, late fusion strategies improve diagnostic precision and reduce inter-observer variability. Operationalized as a secure, user-friendly web application, OncoVision produces structured reports with dual-confidence scoring and attention-weighted visualizations for real-time diagnostic support to improve clinician trust and facilitate medical teaching. It can be easily incorporated into the clinic, making screening available in underprivileged areas around the world, such as rural South Asia. Combining accurate segmentation with clinical intuition, OncoVision raises the bar for AI-based mammography, offering a scalable and equitable solution to detect breast cancer at an earlier stage and enhancing treatment through timely interventions.
>
---
#### [new 024] CounterVQA: Evaluating and Improving Counterfactual Reasoning in Vision-Language Models for Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦视频理解中的反事实推理任务，针对视觉语言模型在复杂因果链推理上的薄弱表现，提出CounterVQA基准与CFGPT后训练方法，有效提升模型对假设情境下替代结果的推断能力。**

- **链接: [https://arxiv.org/pdf/2511.19923v1](https://arxiv.org/pdf/2511.19923v1)**

> **作者:** Yuefei Chen; Jiang Liu; Xiaodong Lin; Ruixiang Tang
>
> **摘要:** Vision Language Models (VLMs) have recently shown significant advancements in video understanding, especially in feature alignment, event reasoning, and instruction-following tasks. However, their capability for counterfactual reasoning, inferring alternative outcomes under hypothetical conditions, remains underexplored. This capability is essential for robust video understanding, as it requires identifying underlying causal structures and reasoning about unobserved possibilities, rather than merely recognizing observed patterns. To systematically evaluate this capability, we introduce CounterVQA, a video-based benchmark featuring three progressive difficulty levels that assess different aspects of counterfactual reasoning. Through comprehensive evaluation of both state-of-the-art open-source and closed-source models, we uncover a substantial performance gap: while these models achieve reasonable accuracy on simple counterfactual questions, performance degrades significantly on complex multi-hop causal chains. To address these limitations, we develop a post-training method, CFGPT, that enhances a model's visual counterfactual reasoning ability by distilling its counterfactual reasoning capability from the language modality, yielding consistent improvements across all CounterVQA difficulty levels. Dataset and code will be further released.
>
---
#### [new 025] Map-World: Masked Action planning and Path-Integral World Model for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶多模态路径规划问题，提出MAP-World框架。通过掩码动作规划生成多样轨迹，结合路径加权世界模型，实现无需锚点或强化学习的端到端多模态规划，提升决策多样性与计算效率。**

- **链接: [https://arxiv.org/pdf/2511.20156v1](https://arxiv.org/pdf/2511.20156v1)**

> **作者:** Bin Hu; Zijian Lu; Haicheng Liao; Chengran Yuan; Bin Rao; Yongkang Li; Guofa Li; Zhiyong Cui; Cheng-zhong Xu; Zhenning Li
>
> **摘要:** Motion planning for autonomous driving must handle multiple plausible futures while remaining computationally efficient. Recent end-to-end systems and world-model-based planners predict rich multi-modal trajectories, but typically rely on handcrafted anchors or reinforcement learning to select a single best mode for training and control. This selection discards information about alternative futures and complicates optimization. We propose MAP-World, a prior-free multi-modal planning framework that couples masked action planning with a path-weighted world model. The Masked Action Planning (MAP) module treats future ego motion as masked sequence completion: past waypoints are encoded as visible tokens, future waypoints are represented as mask tokens, and a driving-intent path provides a coarse scaffold. A compact latent planning state is expanded into multiple trajectory queries with injected noise, yielding diverse, temporally consistent modes without anchor libraries or teacher policies. A lightweight world model then rolls out future BEV semantics conditioned on each candidate trajectory. During training, semantic losses are computed as an expectation over modes, using trajectory probabilities as discrete path weights, so the planner learns from the full distribution of plausible futures instead of a single selected path. On NAVSIM, our method matches anchor-based approaches and achieves state-of-the-art performance among world-model-based methods, while avoiding reinforcement learning and maintaining real-time inference latency.
>
---
#### [new 026] Block Cascading: Training Free Acceleration of Block-Causal Video Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对块因果视频生成中的速度与质量权衡问题，提出训练免费的Block Cascading方法。通过允许部分去噪的前序块启动后续块生成，实现并行化推理，显著提升速度（最高2倍），同时保持生成质量，消除上下文切换时的缓存开销。**

- **链接: [https://arxiv.org/pdf/2511.20426v1](https://arxiv.org/pdf/2511.20426v1)**

> **作者:** Hmrishav Bandyopadhyay; Nikhil Pinnaparaju; Rahim Entezari; Jim Scott; Yi-Zhe Song; Varun Jampani
>
> **摘要:** Block-causal video generation faces a stark speed-quality trade-off: small 1.3B models manage only 16 FPS while large 14B models crawl at 4.5 FPS, forcing users to choose between responsiveness and quality. Block Cascading significantly mitigates this trade-off through training-free parallelization. Our key insight: future video blocks do not need fully denoised current blocks to begin generation. By starting block generation with partially denoised context from predecessors, we transform sequential pipelines into parallel cascades where multiple blocks denoise simultaneously. With 5 GPUs exploiting temporal parallelism, we achieve ~2x acceleration across all model scales: 1.3B models accelerate from 16 to 30 FPS, 14B models from 4.5 to 12.5 FPS. Beyond inference speed, Block Cascading eliminates overhead from KV-recaching (of ~200ms) during context switches for interactive generation. Extensive evaluations validated against multiple block-causal pipelines demonstrate no significant loss in generation quality when switching from block-causal to Block Cascading pipelines for inference. Project Page: https://hmrishavbandy.github.io/block_cascading_page/
>
---
#### [new 027] Intelligent Image Search Algorithms Fusing Visual Large Models
- **分类: cs.CV**

- **简介: 该论文针对细粒度图像检索任务，解决传统方法在状态识别与零样本搜索上的不足。提出DetVLM框架，融合YOLO检测与视觉大模型，通过两阶段机制实现高效组件筛选与状态验证，支持状态搜索与零样本检索，显著提升准确率。**

- **链接: [https://arxiv.org/pdf/2511.19920v1](https://arxiv.org/pdf/2511.19920v1)**

> **作者:** Kehan Wang; Tingqiong Cui; Yang Zhang; Yu Chen; Shifeng Wu; Zhenzhang Li
>
> **备注:** 31 pages,7 figures
>
> **摘要:** Fine-grained image retrieval, which aims to find images containing specific object components and assess their detailed states, is critical in fields like security and industrial inspection. However, conventional methods face significant limitations: manual features (e.g., SIFT) lack robustness; deep learning-based detectors (e.g., YOLO) can identify component presence but cannot perform state-specific retrieval or zero-shot search; Visual Large Models (VLMs) offer semantic and zero-shot capabilities but suffer from poor spatial grounding and high computational cost, making them inefficient for direct retrieval. To bridge these gaps, this paper proposes DetVLM, a novel intelligent image search framework that synergistically fuses object detection with VLMs. The framework pioneers a search-enhancement paradigm via a two-stage pipeline: a YOLO detector first conducts efficient, high-recall component-level screening to determine component presence; then, a VLM acts as a recall-enhancement unit, performing secondary verification for components missed by the detector. This architecture directly enables two advanced capabilities: 1) State Search: Guided by task-specific prompts, the VLM refines results by verifying component existence and executing sophisticated state judgments (e.g., "sun visor lowered"), allowing retrieval based on component state. 2) Zero-shot Search: The framework leverages the VLM's inherent zero-shot capability to recognize and retrieve images containing unseen components or attributes (e.g., "driver wearing a mask") without any task-specific training. Experiments on a vehicle component dataset show DetVLM achieves a state-of-the-art overall retrieval accuracy of 94.82\%, significantly outperforming detection-only baselines. It also attains 94.95\% accuracy in zero-shot search for driver mask-wearing and over 90\% average accuracy in state search tasks.
>
---
#### [new 028] LungEvaty: A Scalable, Open-Source Transformer-based Deep Learning Model for Lung Cancer Risk Prediction in LDCT Screening
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LungEvaty，一个基于Transformer的开源深度学习模型，用于低剂量CT筛查中肺癌风险预测。针对现有方法依赖像素标注或碎片化分析导致可扩展性差、性能受限的问题，模型以全肺为输入，无需区域监督，直接学习大规模数据中的病理特征，实现高效准确的风险评估，支持未来多模态与纵向研究。**

- **链接: [https://arxiv.org/pdf/2511.20116v1](https://arxiv.org/pdf/2511.20116v1)**

> **作者:** Johannes Brandt; Maulik Chevli; Rickmer Braren; Georgios Kaissis; Philip Müller; Daniel Rueckert
>
> **摘要:** Lung cancer risk estimation is gaining increasing importance as more countries introduce population-wide screening programs using low-dose CT (LDCT). As imaging volumes grow, scalable methods that can process entire lung volumes efficiently are essential to tap into the full potential of these large screening datasets. Existing approaches either over-rely on pixel-level annotations, limiting scalability, or analyze the lung in fragments, weakening performance. We present LungEvaty, a fully transformer-based framework for predicting 1-6 year lung cancer risk from a single LDCT scan. The model operates on whole-lung inputs, learning directly from large-scale screening data to capture comprehensive anatomical and pathological cues relevant for malignancy risk. Using only imaging data and no region supervision, LungEvaty matches state-of-the-art performance, refinable by an optional Anatomically Informed Attention Guidance (AIAG) loss that encourages anatomically focused attention. In total, LungEvaty was trained on more than 90,000 CT scans, including over 28,000 for fine-tuning and 6,000 for evaluation. The framework offers a simple, data-efficient, and fully open-source solution that provides an extensible foundation for future research in longitudinal and multimodal lung cancer risk prediction.
>
---
#### [new 029] Vision--Language Enhanced Foundation Model for Semi-supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医疗图像分割中标注数据稀缺的问题，提出视觉-语言增强的半监督分割框架VESSA。通过引入视觉-语言模型，利用少量标注样本生成高质量伪标签，动态优化学生模型，显著提升有限标注下的分割性能。**

- **链接: [https://arxiv.org/pdf/2511.19759v1](https://arxiv.org/pdf/2511.19759v1)**

> **作者:** Jiaqi Guo; Mingzhen Li; Hanyu Su; Santiago López; Lexiaozi Fan; Daniel Kim; Aggelos Katsaggelos
>
> **摘要:** Semi-supervised learning (SSL) has emerged as an effective paradigm for medical image segmentation, reducing the reliance on extensive expert annotations. Meanwhile, vision-language models (VLMs) have demonstrated strong generalization and few-shot capabilities across diverse visual domains. In this work, we integrate VLM-based segmentation into semi-supervised medical image segmentation by introducing a Vision-Language Enhanced Semi-supervised Segmentation Assistant (VESSA) that incorporates foundation-level visual-semantic understanding into SSL frameworks. Our approach consists of two stages. In Stage 1, the VLM-enhanced segmentation foundation model VESSA is trained as a reference-guided segmentation assistant using a template bank containing gold-standard exemplars, simulating learning from limited labeled data. Given an input-template pair, VESSA performs visual feature matching to extract representative semantic and spatial cues from exemplar segmentations, generating structured prompts for a SAM2-inspired mask decoder to produce segmentation masks. In Stage 2, VESSA is integrated into a state-of-the-art SSL framework, enabling dynamic interaction with the student model: as student predictions become more refined, they are fed back to VESSA as prompts, allowing it to generate higher-quality pseudo-labels and stronger guidance. Extensive experiments across multiple segmentation datasets and domains show that VESSA-augmented SSL significantly enhances segmentation accuracy, outperforming state-of-the-art baselines under extremely limited annotation conditions.
>
---
#### [new 030] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning
- **分类: cs.CV**

- **简介: 该论文提出GroundingAgent，一种无需微调的视觉定位框架。针对现有方法依赖标注数据、泛化能力差的问题，利用预训练模型与代理式推理，通过迭代分析实现零样本视觉定位，准确率达65.1%，并具备高可解释性。**

- **链接: [https://arxiv.org/pdf/2511.19516v1](https://arxiv.org/pdf/2511.19516v1)**

> **作者:** Liqin Luo; Guangyao Chen; Xiawu Zheng; Yongxing Dai; Yixiong Zou; Yonghong Tian
>
> **备注:** AAAI 2025
>
> **摘要:** Visual grounding, the task of linking textual queries to specific regions within images, plays a pivotal role in vision-language integration. Existing methods typically rely on extensive task-specific annotations and fine-tuning, limiting their ability to generalize effectively to novel or out-of-distribution scenarios. To address these limitations, we introduce GroundingAgent, a novel agentic visual grounding framework that operates without any task-specific fine-tuning. GroundingAgent employs a structured, iterative reasoning mechanism that integrates pretrained open-vocabulary object detectors, multimodal large language models (MLLMs), and large language models (LLMs) to progressively refine candidate regions through joint semantic and spatial analyses. Remarkably, GroundingAgent achieves an average zero-shot grounding accuracy of 65.1 % on widely-used benchmarks (RefCOCO, RefCOCO+, RefCOCOg), entirely without fine-tuning. Furthermore, by substituting MLLM-generated captions with the original query texts, the accuracy at the selection stage alone reaches approximately 90 %, closely matching supervised performance and underscoring the critical role of LLM reasoning capabilities. GroundingAgent also offers strong interpretability, transparently illustrating each reasoning step and providing clear insights into its decision-making process.
>
---
#### [new 031] ShapeGen: Towards High-Quality 3D Shape Synthesis
- **分类: cs.CV**

- **简介: 该论文聚焦图像到3D形状生成任务，旨在解决现有方法生成的3D资产细节不足、表面过于平滑及结构破碎等问题。提出ShapeGen框架，通过改进3D表示与监督、提升分辨率及引入线性变压器，显著提升生成质量，实现更高质量的3D资产合成，推动其在实际应用中的集成与普及。**

- **链接: [https://arxiv.org/pdf/2511.20624v1](https://arxiv.org/pdf/2511.20624v1)**

> **作者:** Yangguang Li; Xianglong He; Zi-Xin Zou; Zexiang Liu; Wanli Ouyang; Ding Liang; Yan-Pei Cao
>
> **备注:** Accepted to SIGGRAPH Asia 2025
>
> **摘要:** Inspired by generative paradigms in image and video, 3D shape generation has made notable progress, enabling the rapid synthesis of high-fidelity 3D assets from a single image. However, current methods still face challenges, including the lack of intricate details, overly smoothed surfaces, and fragmented thin-shell structures. These limitations leave the generated 3D assets still one step short of meeting the standards favored by artists. In this paper, we present ShapeGen, which achieves high-quality image-to-3D shape generation through 3D representation and supervision improvements, resolution scaling up, and the advantages of linear transformers. These advancements allow the generated assets to be seamlessly integrated into 3D pipelines, facilitating their widespread adoption across various applications. Through extensive experiments, we validate the impact of these improvements on overall performance. Ultimately, thanks to the synergistic effects of these enhancements, ShapeGen achieves a significant leap in image-to-3D generation, establishing a new state-of-the-art performance.
>
---
#### [new 032] Blind Adaptive Local Denoising for CEST Imaging
- **分类: cs.CV**

- **简介: 该论文针对CEST MRI中因硬件限制和复杂协议导致的异方差噪声问题，提出盲适应局部去噪（BALD）方法。通过利用数据自相似性实现噪声均衡化，并结合局部SVD变换与两阶段去噪，有效保留分子信号，提升定量分析与癌症检测性能。**

- **链接: [https://arxiv.org/pdf/2511.20081v1](https://arxiv.org/pdf/2511.20081v1)**

> **作者:** Chu Chen; Aitor Artola; Yang Liu; Se Weon Park; Raymond H. Chan; Jean-Michel Morel; Kannie W. Y. Chan
>
> **摘要:** Chemical Exchange Saturation Transfer (CEST) MRI enables molecular-level visualization of low-concentration metabolites by leveraging proton exchange dynamics. However, its clinical translation is hindered by inherent challenges: spatially varying noise arising from hardware limitations, and complex imaging protocols introduce heteroscedasticity in CEST data, perturbing the accuracy of quantitative contrast mapping such as amide proton transfer (APT) imaging. Traditional denoising methods are not designed for this complex noise and often alter the underlying information that is critical for biomedical analysis. To overcome these limitations, we propose a new Blind Adaptive Local Denoising (BALD) method. BALD exploits the self-similar nature of CEST data to derive an adaptive variance-stabilizing transform that equalizes the noise distributions across CEST pixels without prior knowledge of noise characteristics. Then, BALD performs two-stage denoising on a linear transformation of data to disentangle molecular signals from noise. A local SVD decomposition is used as a linear transform to prevent spatial and spectral denoising artifacts. We conducted extensive validation experiments on multiple phantoms and \textit{in vivo} CEST scans. In these experiments, BALD consistently outperformed state-of-the-art CEST denoisers in both denoising metrics and downstream tasks such as molecular concentration maps estimation and cancer detection.
>
---
#### [new 033] Leveraging Unlabeled Scans for NCCT Image Segmentation in Early Stroke Diagnosis: A Semi-Supervised GAN Approach
- **分类: cs.CV**

- **简介: 该论文针对早期缺血性卒中诊断中非增强CT（NCCT）图像难以识别微小梗死灶的问题，提出一种半监督GAN方法。通过结合少量标注数据与大量未标注扫描，利用多损失函数提升分割精度，旨在提高早期卒中检测效率，减少人工标注负担。**

- **链接: [https://arxiv.org/pdf/2511.19576v1](https://arxiv.org/pdf/2511.19576v1)**

> **作者:** Maria Thoma; Michalis A. Savelonas; Dimitris K. Iakovidis
>
> **摘要:** Ischemic stroke is a time-critical medical emergency where rapid diagnosis is essential for improving patient outcomes. Non-contrast computed tomography (NCCT) serves as the frontline imaging tool, yet it often fails to reveal the subtle ischemic changes present in the early, hyperacute phase. This limitation can delay crucial interventions. To address this diagnostic challenge, we introduce a semi-supervised segmentation method using generative adversarial networks (GANs) to accurately delineate early ischemic stroke regions. The proposed method employs an adversarial framework to effectively learn from a limited number of annotated NCCT scans, while simultaneously leveraging a larger pool of unlabeled scans. By employing Dice loss, cross-entropy loss, a feature matching loss and a self-training loss, the model learns to identify and delineate early infarcts, even when they are faint or their size is small. Experiments on the publicly available Acute Ischemic Stroke Dataset (AISD) demonstrate the potential of the proposed method to enhance diagnostic capabilities, reduce the burden of manual annotation, and support more efficient clinical decision-making in stroke care.
>
---
#### [new 034] Restora-Flow: Mask-Guided Image Restoration with Flow Matching
- **分类: cs.CV**

- **简介: 该论文提出Restora-Flow，一种无需训练的图像修复方法，针对流匹配模型在修复任务中存在处理慢、结果过平滑的问题。通过退化掩码引导采样并引入轨迹修正机制，提升修复质量与速度，在图像补全、超分和去噪任务上表现优越。**

- **链接: [https://arxiv.org/pdf/2511.20152v1](https://arxiv.org/pdf/2511.20152v1)**

> **作者:** Arnela Hadzic; Franz Thaler; Lea Bogensperger; Simon Johannes Joham; Martin Urschler
>
> **备注:** Accepted for WACV 2026
>
> **摘要:** Flow matching has emerged as a promising generative approach that addresses the lengthy sampling times associated with state-of-the-art diffusion models and enables a more flexible trajectory design, while maintaining high-quality image generation. This capability makes it suitable as a generative prior for image restoration tasks. Although current methods leveraging flow models have shown promising results in restoration, some still suffer from long processing times or produce over-smoothed results. To address these challenges, we introduce Restora-Flow, a training-free method that guides flow matching sampling by a degradation mask and incorporates a trajectory correction mechanism to enforce consistency with degraded inputs. We evaluate our approach on both natural and medical datasets across several image restoration tasks involving a mask-based degradation, i.e., inpainting, super-resolution and denoising. We show superior perceptual quality and processing time compared to diffusion and flow matching-based reference methods.
>
---
#### [new 035] Proxy-Free Gaussian Splats Deformation with Splat-Based Surface Estimation
- **分类: cs.CV**

- **简介: 该论文提出SpLap，一种无需代理的高斯点云变形方法。针对现有方法依赖代理或忽略表面结构的问题，提出基于交集定义邻接关系的表面感知图，并结合自适应高斯核，提升变形保真度与渲染质量。在多个数据集上验证了其优越性。**

- **链接: [https://arxiv.org/pdf/2511.19542v1](https://arxiv.org/pdf/2511.19542v1)**

> **作者:** Jaeyeong Kim; Seungwoo Yoo; Minhyuk Sung
>
> **备注:** 17 pages, Accepted to 3DV 2026 (IEEE/CVF International Conference on 3D Vision)
>
> **摘要:** We introduce SpLap, a proxy-free deformation method for Gaussian splats (GS) based on a Laplacian operator computed from our novel surface-aware splat graph. Existing approaches to GS deformation typically rely on deformation proxies such as cages or meshes, but they suffer from dependency on proxy quality and additional computational overhead. An alternative is to directly apply Laplacian-based deformation techniques by treating splats as point clouds. However, this often fail to properly capture surface information due to lack of explicit structure. To address this, we propose a novel method that constructs a surface-aware splat graph, enabling the Laplacian operator derived from it to support more plausible deformations that preserve details and topology. Our key idea is to leverage the spatial arrangement encoded in splats, defining neighboring splats not merely by the distance between their centers, but by their intersections. Furthermore, we introduce a Gaussian kernel adaptation technique that preserves surface structure under deformation, thereby improving rendering quality after deformation. In our experiments, we demonstrate the superior performance of our method compared to both proxy-based and proxy-free baselines, evaluated on 50 challenging objects from the ShapeNet, Objaverse, and Sketchfab datasets, as well as the NeRF-Synthetic dataset. Code is available at https://github.com/kjae0/SpLap.
>
---
#### [new 036] STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction
- **分类: cs.CV**

- **简介: 该论文聚焦于单目视频重建高保真可驱动3D头部动画的任务。针对现有方法绑定僵硬、细节与遮挡区域重建差的问题，提出STAvatar：通过UV自适应软绑定实现动态密度控制，结合时序聚类与融合感知误差的密度优化策略，显著提升细节与遮挡区重建效果。**

- **链接: [https://arxiv.org/pdf/2511.19854v1](https://arxiv.org/pdf/2511.19854v1)**

> **作者:** Jiankuo Zhao; Xiangyu Zhu; Zidu Wang; Zhen Lei
>
> **备注:** 17 pages, 14 figures
>
> **摘要:** Reconstructing high-fidelity and animatable 3D head avatars from monocular videos remains a challenging yet essential task. Existing methods based on 3D Gaussian Splatting typically bind Gaussians to mesh triangles and model deformations solely via Linear Blend Skinning, which results in rigid motion and limited expressiveness. Moreover, they lack specialized strategies to handle frequently occluded regions (e.g., mouth interiors, eyelids). To address these limitations, we propose STAvatar, which consists of two key components: (1) a UV-Adaptive Soft Binding framework that leverages both image-based and geometric priors to learn per-Gaussian feature offsets within the UV space. This UV representation supports dynamic resampling, ensuring full compatibility with Adaptive Density Control (ADC) and enhanced adaptability to shape and textural variations. (2) a Temporal ADC strategy, which first clusters structurally similar frames to facilitate more targeted computation of the densification criterion. It further introduces a novel fused perceptual error as clone criterion to jointly capture geometric and textural discrepancies, encouraging densification in regions requiring finer details. Extensive experiments on four benchmark datasets demonstrate that STAvatar achieves state-of-the-art reconstruction performance, especially in capturing fine-grained details and reconstructing frequently occluded regions. The code will be publicly available.
>
---
#### [new 037] Exo2EgoSyn: Unlocking Foundation Video Generation Models for Exocentric-to-Egocentric Video Synthesis
- **分类: cs.CV**

- **简介: 该论文提出Exo2EgoSyn，解决基础视频生成模型仅支持同视角生成的问题。针对第三人称到第一人称视角的视频合成任务，通过视图对齐、多视角条件融合与姿态感知潜空间注入，实现无需重训练的高质量跨视角视频生成。**

- **链接: [https://arxiv.org/pdf/2511.20186v1](https://arxiv.org/pdf/2511.20186v1)**

> **作者:** Mohammad Mahdi; Yuqian Fu; Nedko Savov; Jiancheng Pan; Danda Pani Paudel; Luc Van Gool
>
> **摘要:** Foundation video generation models such as WAN 2.2 exhibit strong text- and image-conditioned synthesis abilities but remain constrained to the same-view generation setting. In this work, we introduce Exo2EgoSyn, an adaptation of WAN 2.2 that unlocks Exocentric-to-Egocentric(Exo2Ego) cross-view video synthesis. Our framework consists of three key modules. Ego-Exo View Alignment(EgoExo-Align) enforces latent-space alignment between exocentric and egocentric first-frame representations, reorienting the generative space from the given exo view toward the ego view. Multi-view Exocentric Video Conditioning (MultiExoCon) aggregates multi-view exocentric videos into a unified conditioning signal, extending WAN2.2 beyond its vanilla single-image or text conditioning. Furthermore, Pose-Aware Latent Injection (PoseInj) injects relative exo-to-ego camera pose information into the latent state, guiding geometry-aware synthesis across viewpoints. Together, these modules enable high-fidelity ego view video generation from third-person observations without retraining from scratch. Experiments on ExoEgo4D validate that Exo2EgoSyn significantly improves Ego2Exo synthesis, paving the way for scalable cross-view video generation with foundation models. Source code and models will be released publicly.
>
---
#### [new 038] MAPS: Preserving Vision-Language Representations via Module-Wise Proximity Scheduling for Better Vision-Language-Action Generalization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在微调时破坏预训练视觉-语言表征、损害泛化能力的问题，提出MAPS框架。通过模块级近邻调度，分阶段放松不同模块的约束，平衡稳定性与适应性，无需额外参数或数据，显著提升多场景下的性能。**

- **链接: [https://arxiv.org/pdf/2511.19878v1](https://arxiv.org/pdf/2511.19878v1)**

> **作者:** Chengyue Huang; Mellon M. Zhang; Robert Azarcon; Glen Chou; Zsolt Kira
>
> **摘要:** Vision-Language-Action (VLA) models inherit strong priors from pretrained Vision-Language Models (VLMs), but naive fine-tuning often disrupts these representations and harms generalization. Existing fixes -- freezing modules or applying uniform regularization -- either overconstrain adaptation or ignore the differing roles of VLA components. We present MAPS (Module-Wise Proximity Scheduling), the first robust fine-tuning framework for VLAs. Through systematic analysis, we uncover an empirical order in which proximity constraints should be relaxed to balance stability and flexibility. MAPS linearly schedules this relaxation, enabling visual encoders to stay close to their pretrained priors while action-oriented language layers adapt more freely. MAPS introduces no additional parameters or data, and can be seamlessly integrated into existing VLAs. Across MiniVLA-VQ, MiniVLA-OFT, OpenVLA-OFT, and challenging benchmarks such as SimplerEnv, CALVIN, LIBERO, as well as real-world evaluations on the Franka Emika Panda platform, MAPS consistently boosts both in-distribution and out-of-distribution performance (up to +30%). Our findings highlight empirically guided proximity to pretrained VLMs as a simple yet powerful principle for preserving broad generalization in VLM-to-VLA transfer.
>
---
#### [new 039] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout
- **分类: cs.CV**

- **简介: 该论文针对自回归视频生成中时序长度受限、控制响应慢和无法实现连续转场的问题，提出$\infty$-RoPE框架。通过块相对位置编码、KV缓存刷新和RoPE裁剪，实现无限时长、可控且具有电影级转场的视频生成，无需训练即可显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.20649v1](https://arxiv.org/pdf/2511.20649v1)**

> **作者:** Hidir Yesiltepe; Tuna Han Salih Meral; Adil Kaan Akan; Kaan Oktay; Pinar Yanardag
>
> **备注:** Project Page: https://infinity-rope.github.io/
>
> **摘要:** Current autoregressive video diffusion models are constrained by three core bottlenecks: (i) the finite temporal horizon imposed by the base model's 3D Rotary Positional Embedding (3D-RoPE), (ii) slow prompt responsiveness in maintaining fine-grained action control during long-form rollouts, and (iii) the inability to realize discontinuous cinematic transitions within a single generation stream. We introduce $\infty$-RoPE, a unified inference-time framework that addresses all three limitations through three interconnected components: Block-Relativistic RoPE, KV Flush, and RoPE Cut. Block-Relativistic RoPE reformulates temporal encoding as a moving local reference frame, where each newly generated latent block is rotated relative to the base model's maximum frame horizon while earlier blocks are rotated backward to preserve relative temporal geometry. This relativistic formulation eliminates fixed temporal positions, enabling continuous video generation far beyond the base positional limits. To obtain fine-grained action control without re-encoding, KV Flush renews the KV cache by retaining only two latent frames, the global sink and the last generated latent frame, thereby ensuring immediate prompt responsiveness. Finally, RoPE Cut introduces controlled discontinuities in temporal RoPE coordinates, enabling multi-cut scene transitions within a single continuous rollout. Together, these components establish $\infty$-RoPE as a training-free foundation for infinite-horizon, controllable, and cinematic video diffusion. Comprehensive experiments show that $\infty$-RoPE consistently surpasses previous autoregressive models in overall VBench scores.
>
---
#### [new 040] EmoFeedback2: Reinforcement of Continuous Emotional Image Generation via LVLM-based Reward and Textual Feedback
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对连续情感图像生成（C-EICG）中情感反馈缺失与情感保真度不足的问题，提出EmoFeedback2框架。通过微调大视觉语言模型提供情感奖励与文本反馈，实现生成-理解-反馈的强化循环，提升图像情感连续性与精准度。**

- **链接: [https://arxiv.org/pdf/2511.19982v1](https://arxiv.org/pdf/2511.19982v1)**

> **作者:** Jingyang Jia; Kai Shu; Gang Yang; Long Xing; Xun Chen; Aiping Liu
>
> **摘要:** Continuous emotional image generation (C-EICG) is emerging rapidly due to its ability to produce images aligned with both user descriptions and continuous emotional values. However, existing approaches lack emotional feedback from generated images, limiting the control of emotional continuity. Additionally, their simple alignment between emotions and naively generated texts fails to adaptively adjust emotional prompts according to image content, leading to insufficient emotional fidelity. To address these concerns, we propose a novel generation-understanding-feedback reinforcement paradigm (EmoFeedback2) for C-EICG, which exploits the reasoning capability of the fine-tuned large vision-language model (LVLM) to provide reward and textual feedback for generating high-quality images with continuous emotions. Specifically, we introduce an emotion-aware reward feedback strategy, where the LVLM evaluates the emotional values of generated images and computes the reward against target emotions, guiding the reinforcement fine-tuning of the generative model and enhancing the emotional continuity of images. Furthermore, we design a self-promotion textual feedback framework, in which the LVLM iteratively analyzes the emotional content of generated images and adaptively produces refinement suggestions for the next-round prompt, improving the emotional fidelity with fine-grained content. Extensive experimental results demonstrate that our approach effectively generates high-quality images with the desired emotions, outperforming existing state-of-the-art methods in our custom dataset. The code and dataset will be released soon.
>
---
#### [new 041] Navigating Gigapixel Pathology Images with Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文针对病理学中巨型图像（gigapixel）的智能分析难题，提出GIANT框架与MultiPathQA基准。通过让大模型像病理医生一样迭代导航全切片图像，显著提升医学图像理解性能，超越传统方法及专用模型，推动通用多模态模型在病理诊断中的应用。**

- **链接: [https://arxiv.org/pdf/2511.19652v1](https://arxiv.org/pdf/2511.19652v1)**

> **作者:** Thomas A. Buckley; Kian R. Weihrauch; Katherine Latham; Andrew Z. Zhou; Padmini A. Manrai; Arjun K. Manrai
>
> **摘要:** Despite being widely used to support clinical care, general-purpose large multimodal models (LMMs) have generally shown poor or inconclusive performance in medical image interpretation, particularly in pathology, where gigapixel images are used. However, prior studies have used either low-resolution thumbnails or random patches, which likely underestimated model performance. Here, we ask whether LMMs can be adapted to reason coherently and accurately in the evaluation of such images. In this study, we introduce Gigapixel Image Agent for Navigating Tissue (GIANT), the first framework that allows LMMs to iteratively navigate whole-slide images (WSIs) like a pathologist. Accompanying GIANT, we release MultiPathQA, a new benchmark, which comprises 934 WSI-level questions, encompassing five clinically-relevant tasks ranging from cancer diagnosis to open-ended reasoning. MultiPathQA also includes 128 questions, authored by two professional pathologists, requiring direct slide interpretation. Using MultiPathQA, we show that our simple agentic system substantially outperforms conventional patch- and thumbnail-based baselines, approaching or surpassing the performance of specialized models trained on millions of images. For example, on pathologist-authored questions, GPT-5 with GIANT achieves 62.5% accuracy, outperforming specialist pathology models such as TITAN (43.8%) and SlideChat (37.5%). Our findings reveal the strengths and limitations of current foundation models and ground future development of LMMs for expert reasoning in pathology.
>
---
#### [new 042] OmniRefiner: Reinforcement-Guided Local Diffusion Refinement
- **分类: cs.CV**

- **简介: 该论文针对参考图像引导的图像精修任务，解决现有扩散模型在细节保留上的不足。提出OmniRefiner框架，通过双阶段优化：先联合输入草图与参考图进行全局一致精修，再用强化学习增强局部细节准确性，显著提升视觉一致性与细节保真度。**

- **链接: [https://arxiv.org/pdf/2511.19990v1](https://arxiv.org/pdf/2511.19990v1)**

> **作者:** Yaoli Liu; Ziheng Ouyang; Shengtao Lou; Yiren Song
>
> **摘要:** Reference-guided image generation has progressed rapidly, yet current diffusion models still struggle to preserve fine-grained visual details when refining a generated image using a reference. This limitation arises because VAE-based latent compression inherently discards subtle texture information, causing identity- and attribute-specific cues to vanish. Moreover, post-editing approaches that amplify local details based on existing methods often produce results inconsistent with the original image in terms of lighting, texture, or shape. To address this, we introduce \ourMthd{}, a detail-aware refinement framework that performs two consecutive stages of reference-driven correction to enhance pixel-level consistency. We first adapt a single-image diffusion editor by fine-tuning it to jointly ingest the draft image and the reference image, enabling globally coherent refinement while maintaining structural fidelity. We then apply reinforcement learning to further strengthen localized editing capability, explicitly optimizing for detail accuracy and semantic consistency. Extensive experiments demonstrate that \ourMthd{} significantly improves reference alignment and fine-grained detail preservation, producing faithful and visually coherent edits that surpass both open-source and commercial models on challenging reference-guided restoration benchmarks.
>
---
#### [new 043] Prompting Lipschitz-constrained network for multiple-in-one sparse-view CT reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对稀疏视角CT重建中深层网络难以保证Lipschitz约束及多视图需多模型导致存储成本高的问题，提出LipNet与PromptCT框架。通过显式构建Lipschitz约束网络并引入提示模块，实现单模型处理多种稀疏采样配置，提升重建质量并降低存储开销。**

- **链接: [https://arxiv.org/pdf/2511.20296v1](https://arxiv.org/pdf/2511.20296v1)**

> **作者:** Baoshun Shi; Ke Jiang; Qiusheng Lian; Xinran Yu; Huazhu Fu
>
> **摘要:** Despite significant advancements in deep learning-based sparse-view computed tomography (SVCT) reconstruction algorithms, these methods still encounter two primary limitations: (i) It is challenging to explicitly prove that the prior networks of deep unfolding algorithms satisfy Lipschitz constraints due to their empirically designed nature. (ii) The substantial storage costs of training a separate model for each setting in the case of multiple views hinder practical clinical applications. To address these issues, we elaborate an explicitly provable Lipschitz-constrained network, dubbed LipNet, and integrate an explicit prompt module to provide discriminative knowledge of different sparse sampling settings, enabling the treatment of multiple sparse view configurations within a single model. Furthermore, we develop a storage-saving deep unfolding framework for multiple-in-one SVCT reconstruction, termed PromptCT, which embeds LipNet as its prior network to ensure the convergence of its corresponding iterative algorithm. In simulated and real data experiments, PromptCT outperforms benchmark reconstruction algorithms in multiple-in-one SVCT reconstruction, achieving higher-quality reconstructions with lower storage costs. On the theoretical side, we explicitly demonstrate that LipNet satisfies boundary property, further proving its Lipschitz continuity and subsequently analyzing the convergence of the proposed iterative algorithms. The data and code are publicly available at https://github.com/shibaoshun/PromptCT.
>
---
#### [new 044] One Attention, One Scale: Phase-Aligned Rotary Positional Embeddings for Mixed-Resolution Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文针对扩散模型中混合分辨率生成时旋转位置编码（RoPE）因线性插值导致的注意力崩溃问题。提出无需训练的跨分辨率相位对齐注意力（CRPA），通过统一查询步长下的位置编码索引，消除相位混淆，稳定所有头与层，实现高效高保真图像与视频生成。**

- **链接: [https://arxiv.org/pdf/2511.19778v1](https://arxiv.org/pdf/2511.19778v1)**

> **作者:** Haoyu Wu; Jingyi Xu; Qiaomu Miao; Dimitris Samaras; Hieu Le
>
> **摘要:** We identify a core failure mode that occurs when using the usual linear interpolation on rotary positional embeddings (RoPE) for mixed-resolution denoising with Diffusion Transformers. When tokens from different spatial grids are mixed, the attention mechanism collapses. The issue is structural. Linear coordinate remapping forces a single attention head to compare RoPE phases sampled at incompatible rates, creating phase aliasing that destabilizes the score landscape. Pretrained DiTs are especially brittle-many heads exhibit extremely sharp, periodic phase selectivity-so even tiny cross-rate inconsistencies reliably cause blur, artifacts, or full collapse. To this end, our main contribution is Cross-Resolution Phase-Aligned Attention (CRPA), a training-free drop-in fix that eliminates this failure at its source. CRPA modifies only the RoPE index map for each attention call: all Q/K positions are expressed on the query's stride so that equal physical distances always induce identical phase increments. This restores the precise phase patterns that DiTs rely on. CRPA is fully compatible with pretrained DiTs, stabilizes all heads and layers uniformly. We demonstrate that CRPA enables high-fidelity and efficient mixed-resolution generation, outperforming previous state-of-the-art methods on image and video generation.
>
---
#### [new 045] Learning to Generate Human-Human-Object Interactions from Textual Descriptions
- **分类: cs.CV**

- **简介: 该论文提出人类-人类-物体交互（HHOI）建模任务，旨在生成多人群体在场景中与物体的复杂互动。针对缺乏专用数据的问题，构建了新数据集并利用生成模型合成数据。基于分数驱动扩散模型，联合建模人-物与人-人交互，实现文本到完整HHOI的统一生成，支持多主体交互生成。**

- **链接: [https://arxiv.org/pdf/2511.20446v1](https://arxiv.org/pdf/2511.20446v1)**

> **作者:** Jeonghyeon Na; Sangwon Baik; Inhee Lee; Junyoung Lee; Hanbyul Joo
>
> **备注:** Project Page: https://tlb-miss.github.io/hhoi/
>
> **摘要:** The way humans interact with each other, including interpersonal distances, spatial configuration, and motion, varies significantly across different situations. To enable machines to understand such complex, context-dependent behaviors, it is essential to model multiple people in relation to the surrounding scene context. In this paper, we present a novel research problem to model the correlations between two people engaged in a shared interaction involving an object. We refer to this formulation as Human-Human-Object Interactions (HHOIs). To overcome the lack of dedicated datasets for HHOIs, we present a newly captured HHOIs dataset and a method to synthesize HHOI data by leveraging image generative models. As an intermediary, we obtain individual human-object interaction (HOIs) and human-human interaction (HHIs) from the HHOIs, and with these data, we train an text-to-HOI and text-to-HHI model using score-based diffusion model. Finally, we present a unified generative framework that integrates the two individual model, capable of synthesizing complete HHOIs in a single advanced sampling process. Our method extends HHOI generation to multi-human settings, enabling interactions involving more than two individuals. Experimental results show that our method generates realistic HHOIs conditioned on textual descriptions, outperforming previous approaches that focus only on single-human HOIs. Furthermore, we introduce multi-human motion generation involving objects as an application of our framework.
>
---
#### [new 046] A Reason-then-Describe Instruction Interpreter for Controllable Video Generation
- **分类: cs.CV**

- **简介: 该论文针对可控视频生成中用户指令与模型输出不匹配的问题，提出ReaDe框架。通过“先推理后描述”机制，将模糊、复杂的指令转化为精确的生成规范，提升生成视频的准确性和可控性。采用两阶段训练优化，实现高保真、强泛化的指令解析。**

- **链接: [https://arxiv.org/pdf/2511.20563v1](https://arxiv.org/pdf/2511.20563v1)**

> **作者:** Shengqiong Wu; Weicai Ye; Yuanxing Zhang; Jiahao Wang; Quande Liu; Xintao Wang; Pengfei Wan; Kun Gai; Hao Fei; Tat-Seng Chua
>
> **备注:** 27 pages, 13 figures, 13 tables, Project Page: https://sqwu.top/ReaDe/
>
> **摘要:** Diffusion Transformers have significantly improved video fidelity and temporal coherence, however, practical controllability remains limited. Concise, ambiguous, and compositionally complex user inputs contrast with the detailed prompts used in training, yielding an intent-output mismatch. We propose ReaDe, a universal, model-agnostic interpreter that converts raw instructions into precise, actionable specifications for downstream video generators. ReaDe follows a reason-then-describe paradigm: it first analyzes the user request to identify core requirements and resolve ambiguities, then produces detailed guidance that enables faithful, controllable generation. We train ReaDe via a two-stage optimization: (i) reasoning-augmented supervision imparts analytic parsing with stepwise traces and dense captions, and (ii) a multi-dimensional reward assigner enables stable, feedback-driven refinement for natural-style captions. Experiments across single- and multi-condition scenarios show consistent gains in instruction fidelity, caption accuracy, and downstream video quality, with strong generalization to reasoning-intensive and unseen inputs. ReaDe offers a practical route to aligning controllable video generation with accurately interpreted user intent. Project Page: https://sqwu.top/ReaDe/.
>
---
#### [new 047] DesignPref: Capturing Personal Preferences in Visual Design Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文针对视觉设计生成中个体偏好差异问题，提出DesignPref数据集（12k设计对比，20名设计师标注）。研究发现设计师间偏好分歧大（Krippendorff's alpha=0.25），传统多数投票方法不准确。通过个性化微调与RAG集成，证明少量个性化数据即可显著提升个体偏好预测效果，为个性化设计生成提供新范式。**

- **链接: [https://arxiv.org/pdf/2511.20513v1](https://arxiv.org/pdf/2511.20513v1)**

> **作者:** Yi-Hao Peng; Jeffrey P. Bigham; Jason Wu
>
> **摘要:** Generative models, such as large language models and text-to-image diffusion models, are increasingly used to create visual designs like user interfaces (UIs) and presentation slides. Finetuning and benchmarking these generative models have often relied on datasets of human-annotated design preferences. Yet, due to the subjective and highly personalized nature of visual design, preference varies widely among individuals. In this paper, we study this problem by introducing DesignPref, a dataset of 12k pairwise comparisons of UI design generation annotated by 20 professional designers with multi-level preference ratings. We found that among trained designers, substantial levels of disagreement exist (Krippendorff's alpha = 0.25 for binary preferences). Natural language rationales provided by these designers indicate that disagreements stem from differing perceptions of various design aspect importance and individual preferences. With DesignPref, we demonstrate that traditional majority-voting methods for training aggregated judge models often do not accurately reflect individual preferences. To address this challenge, we investigate multiple personalization strategies, particularly fine-tuning or incorporating designer-specific annotations into RAG pipelines. Our results show that personalized models consistently outperform aggregated baseline models in predicting individual designers' preferences, even when using 20 times fewer examples. Our work provides the first dataset to study personalized visual design evaluation and support future research into modeling individual design taste.
>
---
#### [new 048] 3D-Aware Multi-Task Learning with Cross-View Correlations for Dense Scene Understanding
- **分类: cs.CV**

- **简介: 该论文研究多任务学习中的密集场景理解任务，旨在解决现有方法在2D空间建模跨任务关系导致缺乏3D一致性的难题。提出一种轻量级跨视图模块（CvM），通过引入视图间代价体实现几何一致性，增强网络对3D结构的感知，提升分割与深度估计等任务性能。**

- **链接: [https://arxiv.org/pdf/2511.20646v1](https://arxiv.org/pdf/2511.20646v1)**

> **作者:** Xiaoye Wang; Chen Tang; Xiangyu Yue; Wei-Hong Li
>
> **备注:** 3D-aware Multi-task Learning, Cross-view Correlations, Code will be available at https://github.com/WeiHongLee/CrossView3DMTL
>
> **摘要:** This paper addresses the challenge of training a single network to jointly perform multiple dense prediction tasks, such as segmentation and depth estimation, i.e., multi-task learning (MTL). Current approaches mainly capture cross-task relations in the 2D image space, often leading to unstructured features lacking 3D-awareness. We argue that 3D-awareness is vital for modeling cross-task correlations essential for comprehensive scene understanding. We propose to address this problem by integrating correlations across views, i.e., cost volume, as geometric consistency in the MTL network. Specifically, we introduce a lightweight Cross-view Module (CvM), shared across tasks, to exchange information across views and capture cross-view correlations, integrated with a feature from MTL encoder for multi-task predictions. This module is architecture-agnostic and can be applied to both single and multi-view data. Extensive results on NYUv2 and PASCAL-Context demonstrate that our method effectively injects geometric consistency into existing MTL methods to improve performance.
>
---
#### [new 049] Fewer Tokens, Greater Scaling: Self-Adaptive Visual Bases for Efficient and Expansive Representation Learning
- **分类: cs.CV**

- **简介: 该论文研究视觉模型中令牌数量与模型容量的关系，旨在降低视觉表示学习的计算开销。针对冗余令牌问题，提出自适应正交过滤模块，动态生成紧凑正交基。发现大模型可更少令牌保持语义，揭示“令牌-模型”缩放规律，并构建视觉长上下文数据集。**

- **链接: [https://arxiv.org/pdf/2511.19515v1](https://arxiv.org/pdf/2511.19515v1)**

> **作者:** Shawn Young; Xingyu Zeng; Lijian Xu
>
> **摘要:** This paper investigates the fundamental relationship between model capacity and the minimal number of visual tokens required to preserve image semantics. Inspired by the Minimum Description Length principle, we reinterpret image tokens as vectors in a visual semantic space and define the intrinsic semantic complexity of an image as the smallest set of basis vectors needed to span this space. Building on this perspective, we propose Orthogonal Filtering, a lightweight module that adaptively clusters redundant tokens into a compact set of orthogonal bases. Through extensive experiments across a range of ViT models, we reveal a consistent token, model scaling law: larger models require significantly fewer tokens to span visual semantic space. Besides, we also contribute a visual long-context dataset.
>
---
#### [new 050] Context-Aware Token Pruning and Discriminative Selective Attention for Transformer Tracking
- **分类: cs.CV**

- **简介: 该论文针对一阶段Transformer跟踪器中背景与干扰项导致的判别力下降问题，提出CPDATrack框架。通过可学习模块估计搜索区域令牌与目标的相关性，动态剪枝低信息背景令牌，并设计判别性选择注意力机制，在早期阻断背景到模板的注意力，后期仅让高概率目标令牌参与注意力计算，有效提升跟踪精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.19928v1](https://arxiv.org/pdf/2511.19928v1)**

> **作者:** Janani Kugarajeevan; Thanikasalam Kokul; Amirthalingam Ramanan; Subha Fernando
>
> **摘要:** One-stream Transformer-based trackers have demonstrated remarkable performance by concatenating template and search region tokens, thereby enabling joint attention across all tokens. However, enabling an excessive proportion of background search tokens to attend to the target template tokens weakens the tracker's discriminative capability. Several token pruning methods have been proposed to mitigate background interference; however, they often remove tokens near the target, leading to the loss of essential contextual information and degraded tracking performance. Moreover, the presence of distractors within the search tokens further reduces the tracker's ability to accurately identify the target. To address these limitations, we propose CPDATrack, a novel tracking framework designed to suppress interference from background and distractor tokens while enhancing computational efficiency. First, a learnable module is integrated between two designated encoder layers to estimate the probability of each search token being associated with the target. Based on these estimates, less-informative background tokens are pruned from the search region while preserving the contextual cues surrounding the target. To further suppress background interference, a discriminative selective attention mechanism is employed that fully blocks search-to-template attention in the early layers. In the subsequent encoder layers, high-probability target tokens are selectively extracted from a localized region to attend to the template tokens, thereby reducing the influence of background and distractor tokens. The proposed CPDATrack achieves state-of-the-art performance across multiple benchmarks, particularly on GOT-10k, where it attains an average overlap of 75.1 percent.
>
---
#### [new 051] DeLightMono: Enhancing Self-Supervised Monocular Depth Estimation in Endoscopy by Decoupling Uneven Illumination
- **分类: cs.CV**

- **简介: 该论文针对内窥镜图像中不均匀光照导致的单目深度估计性能下降问题，提出DeLight-Mono框架。通过解耦光照、反射与深度成分，并设计自监督联合优化损失，有效提升低光照区域的深度估计精度，显著改善了自监督单目深度估计在内窥镜场景下的表现。**

- **链接: [https://arxiv.org/pdf/2511.20058v1](https://arxiv.org/pdf/2511.20058v1)**

> **作者:** Mingyang Ou; Haojin Li; Yifeng Zhang; Ke Niu; Zhongxi Qiu; Heng Li; Jiang Liu
>
> **摘要:** Self-supervised monocular depth estimation serves as a key task in the development of endoscopic navigation systems. However, performance degradation persists due to uneven illumination inherent in endoscopic images, particularly in low-intensity regions. Existing low-light enhancement techniques fail to effectively guide the depth network. Furthermore, solutions from other fields, like autonomous driving, require well-lit images, making them unsuitable and increasing data collection burdens. To this end, we present DeLight-Mono - a novel self-supervised monocular depth estimation framework with illumination decoupling. Specifically, endoscopic images are represented by a designed illumination-reflectance-depth model, and are decomposed with auxiliary networks. Moreover, a self-supervised joint-optimizing framework with novel losses leveraging the decoupled components is proposed to mitigate the effects of uneven illumination on depth estimation. The effectiveness of the proposed methods was rigorously verified through extensive comparisons and an ablation study performed on two public datasets.
>
---
#### [new 052] RubricRL: Simple Generalizable Rewards for Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成中的偏好对齐问题，提出RubricRL框架。通过动态构建可解释的多维度视觉评判清单（如物体正确性、属性准确性等），结合多模态判别器与自适应加权，实现灵活、可调控的奖励设计，提升生成结果的忠实度与泛化性。**

- **链接: [https://arxiv.org/pdf/2511.20651v1](https://arxiv.org/pdf/2511.20651v1)**

> **作者:** Xuelu Feng; Yunsheng Li; Ziyu Wan; Zixuan Gao; Junsong Yuan; Dongdong Chen; Chunming Qiao
>
> **摘要:** Reinforcement learning (RL) has recently emerged as a promising approach for aligning text-to-image generative models with human preferences. A key challenge, however, lies in designing effective and interpretable rewards. Existing methods often rely on either composite metrics (e.g., CLIP, OCR, and realism scores) with fixed weights or a single scalar reward distilled from human preference models, which can limit interpretability and flexibility. We propose RubricRL, a simple and general framework for rubric-based reward design that offers greater interpretability, composability, and user control. Instead of using a black-box scalar signal, RubricRL dynamically constructs a structured rubric for each prompt--a decomposable checklist of fine-grained visual criteria such as object correctness, attribute accuracy, OCR fidelity, and realism--tailored to the input text. Each criterion is independently evaluated by a multimodal judge (e.g., o4-mini), and a prompt-adaptive weighting mechanism emphasizes the most relevant dimensions. This design not only produces interpretable and modular supervision signals for policy optimization (e.g., GRPO or PPO), but also enables users to directly adjust which aspects to reward or penalize. Experiments with an autoregressive text-to-image model demonstrate that RubricRL improves prompt faithfulness, visual detail, and generalizability, while offering a flexible and extensible foundation for interpretable RL alignment across text-to-image architectures.
>
---
#### [new 053] Thinking in 360°: Humanoid Visual Search in the Wild
- **分类: cs.CV**

- **简介: 该论文研究人形机器人在360°全景图像中进行视觉搜索的任务，旨在解决传统方法忽视物理具身与三维交互的问题。提出人形视觉搜索框架，构建H* Bench基准，验证并提升模型性能，揭示路径搜索的高难度源于空间常识需求。**

- **链接: [https://arxiv.org/pdf/2511.20351v1](https://arxiv.org/pdf/2511.20351v1)**

> **作者:** Heyang Yu; Yinan Han; Xiangyu Zhang; Baiqiao Yin; Bowen Chang; Xiangyu Han; Xinhao Liu; Jing Zhang; Marco Pavone; Chen Feng; Saining Xie; Yiming Li
>
> **摘要:** Humans rely on the synergistic control of head (cephalomotor) and eye (oculomotor) to efficiently search for visual information in 360°. However, prior approaches to visual search are limited to a static image, neglecting the physical embodiment and its interaction with the 3D world. How can we develop embodied visual search agents as efficient as humans while bypassing the constraints imposed by real-world hardware? To this end, we propose humanoid visual search where a humanoid agent actively rotates its head to search for objects or paths in an immersive world represented by a 360° panoramic image. To study visual search in visually-crowded real-world scenarios, we build H* Bench, a new benchmark that moves beyond household scenes to challenging in-the-wild scenes that necessitate advanced visual-spatial reasoning capabilities, such as transportation hubs, large-scale retail spaces, urban streets, and public institutions. Our experiments first reveal that even top-tier proprietary models falter, achieving only ~30% success in object and path search. We then use post-training techniques to enhance the open-source Qwen2.5-VL, increasing its success rate by over threefold for both object search (14.83% to 47.38%) and path search (6.44% to 24.94%). Notably, the lower ceiling of path search reveals its inherent difficulty, which we attribute to the demand for sophisticated spatial commonsense. Our results not only show a promising path forward but also quantify the immense challenge that remains in building MLLM agents that can be seamlessly integrated into everyday human life.
>
---
#### [new 054] MambaEye: A Size-Agnostic Visual Encoder with Causal Sequential Processing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MambaEye，一种无输入尺寸依赖的视觉编码器，解决传统模型无法适应任意图像分辨率的问题。通过因果序列处理与相对位移嵌入，实现对任意分辨率和扫描模式的自适应，结合扩散启发式损失函数，提升高分辨率下性能，保持线性复杂度。**

- **链接: [https://arxiv.org/pdf/2511.19963v1](https://arxiv.org/pdf/2511.19963v1)**

> **作者:** Changho Choi; Minho Kim; Jinkyu Kim
>
> **备注:** Code will be released in github
>
> **摘要:** Despite decades of progress, a truly input-size agnostic visual encoder-a fundamental characteristic of human vision-has remained elusive. We address this limitation by proposing \textbf{MambaEye}, a novel, causal sequential encoder that leverages the low complexity and causal-process based pure Mamba2 backbone. Unlike previous Mamba-based vision encoders that often employ bidirectional processing, our strictly unidirectional approach preserves the inherent causality of State Space Models, enabling the model to generate a prediction at any point in its input sequence. A core innovation is our use of relative move embedding, which encodes the spatial shift between consecutive patches, providing a strong inductive bias for translation invariance and making the model inherently adaptable to arbitrary image resolutions and scanning patterns. To achieve this, we introduce a novel diffusion-inspired loss function that provides dense, step-wise supervision, training the model to build confidence as it gathers more visual evidence. We demonstrate that MambaEye exhibits robust performance across a wide range of image resolutions, especially at higher resolutions such as $1536^2$ on the ImageNet-1K classification task. This feat is achieved while maintaining linear time and memory complexity relative to the number of patches.
>
---
#### [new 055] Explainable Visual Anomaly Detection via Concept Bottleneck Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉异常检测（VAD）中现有方法解释性不足的问题，提出基于概念瓶颈模型的可解释方法CONVAD。通过构建概念数据集、改进CBM架构并设计合成异常数据管道，实现既可定位异常又可提供语义化解释，提升模型可解释性与可信度。**

- **链接: [https://arxiv.org/pdf/2511.20088v1](https://arxiv.org/pdf/2511.20088v1)**

> **作者:** Arianna Stropeni; Valentina Zaccaria; Francesco Borsatti; Davide Dalle Pezze; Manuel Barusco; Gian Antonio Susto
>
> **摘要:** In recent years, Visual Anomaly Detection (VAD) has gained significant attention due to its ability to identify anomalous images using only normal images during training. Many VAD models work without supervision but are still able to provide visual explanations by highlighting the anomalous regions within an image. However, although these visual explanations can be helpful, they lack a direct and semantically meaningful interpretation for users. To address this limitation, we propose extending Concept Bottleneck Models (CBMs) to the VAD setting. By learning meaningful concepts, the network can provide human-interpretable descriptions of anomalies, offering a novel and more insightful way to explain them. Our contributions are threefold: (i) we develop a Concept Dataset to support research on CBMs for VAD; (ii) we improve the CBM architecture to generate both concept-based and visual explanations, bridging semantic and localization interpretability; and (iii) we introduce a pipeline for synthesizing artificial anomalies, preserving the VAD paradigm of minimizing dependence on rare anomalous samples. Our approach, Concept-Aware Visual Anomaly Detection (CONVAD), achieves performance comparable to classic VAD methods while providing richer, concept-driven explanations that enhance interpretability and trust in VAD systems.
>
---
#### [new 056] MajutsuCity: Language-driven Aesthetic-adaptive City Generation with Controllable 3D Assets and Layouts
- **分类: cs.CV**

- **简介: 该论文提出MajutsuCity，一个语言驱动的3D城市生成框架，旨在解决现有方法在风格多样性、结构一致性和可控性间的平衡难题。通过四阶段管线与可交互的MajutsuAgent，实现风格自适应、对象级编辑，并构建了高质量多模态数据集与评估指标，显著提升生成效果，达到新基准。**

- **链接: [https://arxiv.org/pdf/2511.20415v1](https://arxiv.org/pdf/2511.20415v1)**

> **作者:** Zilong Huang; Jun He; Xiaobin Huang; Ziyi Xiong; Yang Luo; Junyan Ye; Weijia Li; Yiping Chen; Ting Han
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Generating realistic 3D cities is fundamental to world models, virtual reality, and game development, where an ideal urban scene must satisfy both stylistic diversity, fine-grained, and controllability. However, existing methods struggle to balance the creative flexibility offered by text-based generation with the object-level editability enabled by explicit structural representations. We introduce MajutsuCity, a natural language-driven and aesthetically adaptive framework for synthesizing structurally consistent and stylistically diverse 3D urban scenes. MajutsuCity represents a city as a composition of controllable layouts, assets, and materials, and operates through a four-stage pipeline. To extend controllability beyond initial generation, we further integrate MajutsuAgent, an interactive language-grounded editing agent} that supports five object-level operations. To support photorealistic and customizable scene synthesis, we also construct MajutsuDataset, a high-quality multimodal dataset} containing 2D semantic layouts and height maps, diverse 3D building assets, and curated PBR materials and skyboxes, each accompanied by detailed annotations. Meanwhile, we develop a practical set of evaluation metrics, covering key dimensions such as structural consistency, scene complexity, material fidelity, and lighting atmosphere. Extensive experiments demonstrate MajutsuCity reduces layout FID by 83.7% compared with CityDreamer and by 20.1% over CityCraft. Our method ranks first across all AQS and RDR scores, outperforming existing methods by a clear margin. These results confirm MajutsuCity as a new state-of-the-art in geometric fidelity, stylistic adaptability, and semantic controllability for 3D city generation. We expect our framework can inspire new avenues of research in 3D city generation. Our dataset and code will be released at https://github.com/LongHZ140516/MajutsuCity.
>
---
#### [new 057] Supervise Less, See More: Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting
- **分类: cs.CV**

- **简介: 该论文针对病理图像中核实例分割任务，提出无需训练和标注的SPROUT框架。通过构建组织学先验引导的参考原型，利用部分最优传输对齐特征，生成正负点提示，驱动SAM模型实现精准分割，突破了传统方法依赖密集标注与微调的局限。**

- **链接: [https://arxiv.org/pdf/2511.19953v1](https://arxiv.org/pdf/2511.19953v1)**

> **作者:** Wen Zhang; Qin Ren; Wenjing Liu; Haibin Ling; Chenyu You
>
> **备注:** Preprint; 40 pages, 25 figures, 18 tables
>
> **摘要:** Accurate nuclear instance segmentation is a pivotal task in computational pathology, supporting data-driven clinical insights and facilitating downstream translational applications. While large vision foundation models have shown promise for zero-shot biomedical segmentation, most existing approaches still depend on dense supervision and computationally expensive fine-tuning. Consequently, training-free methods present a compelling research direction, yet remain largely unexplored. In this work, we introduce SPROUT, a fully training- and annotation-free prompting framework for nuclear instance segmentation. SPROUT leverages histology-informed priors to construct slide-specific reference prototypes that mitigate domain gaps. These prototypes progressively guide feature alignment through a partial optimal transport scheme. The resulting foreground and background features are transformed into positive and negative point prompts, enabling the Segment Anything Model (SAM) to produce precise nuclear delineations without any parameter updates. Extensive experiments across multiple histopathology benchmarks demonstrate that SPROUT achieves competitive performance without supervision or retraining, establishing a novel paradigm for scalable, training-free nuclear instance segmentation in pathology.
>
---
#### [new 058] Image-Free Timestep Distillation via Continuous-Time Consistency with Trajectory-Sampled Pairs
- **分类: cs.CV**

- **简介: 该论文针对扩散模型生成效率问题，提出无需外部数据的轨迹采样自蒸馏方法TBCM。通过直接从教师模型生成轨迹提取潜在表示，实现高效、低资源的连续时间一致性蒸馏，显著提升生成速度与内存效率，同时保持高质量生成结果。**

- **链接: [https://arxiv.org/pdf/2511.20410v1](https://arxiv.org/pdf/2511.20410v1)**

> **作者:** Bao Tang; Shuai Zhang; Yueting Zhu; Jijun Xiang; Xin Yang; Li Yu; Wenyu Liu; Xinggang Wang
>
> **摘要:** Timestep distillation is an effective approach for improving the generation efficiency of diffusion models. The Consistency Model (CM), as a trajectory-based framework, demonstrates significant potential due to its strong theoretical foundation and high-quality few-step generation. Nevertheless, current continuous-time consistency distillation methods still rely heavily on training data and computational resources, hindering their deployment in resource-constrained scenarios and limiting their scalability to diverse domains. To address this issue, we propose Trajectory-Backward Consistency Model (TBCM), which eliminates the dependence on external training data by extracting latent representations directly from the teacher model's generation trajectory. Unlike conventional methods that require VAE encoding and large-scale datasets, our self-contained distillation paradigm significantly improves both efficiency and simplicity. Moreover, the trajectory-extracted samples naturally bridge the distribution gap between training and inference, thereby enabling more effective knowledge transfer. Empirically, TBCM achieves 6.52 FID and 28.08 CLIP scores on MJHQ-30k under one-step generation, while reducing training time by approximately 40% compared to Sana-Sprint and saving a substantial amount of GPU memory, demonstrating superior efficiency without sacrificing quality. We further reveal the diffusion-generation space discrepancy in continuous-time consistency distillation and analyze how sampling strategies affect distillation performance, offering insights for future distillation research. GitHub Link: https://github.com/hustvl/TBCM.
>
---
#### [new 059] What You See is (Usually) What You Get: Multimodal Prototype Networks that Abstain from Expensive Modalities
- **分类: cs.CV**

- **简介: 该论文针对物种识别任务，解决多模态神经网络可解释性差、基因数据采集成本高的问题。提出成本感知的多模态原型网络，通过加权融合多模态原型，智能决定是否使用昂贵的基因数据，实现高精度且节省资源的物种分类。**

- **链接: [https://arxiv.org/pdf/2511.19752v1](https://arxiv.org/pdf/2511.19752v1)**

> **作者:** Muchang Bahng; Charlie Berens; Jon Donnelly; Eric Chen; Chaofan Chen; Cynthia Rudin
>
> **备注:** 19 pages. 16 figures. 10 tables
>
> **摘要:** Species detection is important for monitoring the health of ecosystems and identifying invasive species, serving a crucial role in guiding conservation efforts. Multimodal neural networks have seen increasing use for identifying species to help automate this task, but they have two major drawbacks. First, their black-box nature prevents the interpretability of their decision making process. Second, collecting genetic data is often expensive and requires invasive procedures, often necessitating researchers to capture or kill the target specimen. We address both of these problems by extending prototype networks (ProtoPNets), which are a popular and interpretable alternative to traditional neural networks, to the multimodal, cost-aware setting. We ensemble prototypes from each modality, using an associated weight to determine how much a given prediction relies on each modality. We further introduce methods to identify cases for which we do not need the expensive genetic information to make confident predictions. We demonstrate that our approach can intelligently allocate expensive genetic data for fine-grained distinctions while using abundant image data for clearer visual classifications and achieving comparable accuracy to models that consistently use both modalities.
>
---
#### [new 060] DINO-Tok: Adapting DINO for Visual Tokenizers
- **分类: cs.CV**

- **简介: 该论文提出DINO-Tok，一种基于DINO的视觉分词器，旨在解决现有分词器在高维潜在空间中语义与重建保真度难以平衡的问题。通过融合浅层细节与深层语义特征，并引入全局PCA重加权机制稳定向量量化，显著提升图像重建性能，达到当前最优水平。**

- **链接: [https://arxiv.org/pdf/2511.20565v1](https://arxiv.org/pdf/2511.20565v1)**

> **作者:** Mingkai Jia; Mingxiao Li; Liaoyuan Fan; Tianxing Shi; Jiaxin Guo; Zeming Li; Xiaoyang Guo; Xiao-Xiao Long; Qian Zhang; Ping Tan; Wei Yin
>
> **摘要:** Recent advances in visual generation have highlighted the rise of Latent Generative Models (LGMs), which rely on effective visual tokenizers to bridge pixels and semantics. However, existing tokenizers are typically trained from scratch and struggle to balance semantic representation and reconstruction fidelity, particularly in high-dimensional latent spaces. In this work, we introduce DINO-Tok, a DINO-based visual tokenizer that unifies hierarchical representations into an information-complete latent space. By integrating shallow features that retain fine-grained details with deep features encoding global semantics, DINO-Tok effectively bridges pretrained representations and visual generation. We further analyze the challenges of vector quantization (VQ) in this high-dimensional space, where key information is often lost and codebook collapse occurs. We thus propose a global PCA reweighting mechanism to stabilize VQ and preserve essential information across dimensions. On ImageNet 256$\times$256, DINO-Tok achieves state-of-the-art reconstruction performance, reaching 28.54 PSNR for autoencoding and 23.98 PSNR for VQ-based modeling, significantly outperforming prior tokenizers and comparable to billion-level data trained models (such as Hunyuan and Wan). These results demonstrate that adapting powerful pretrained vision models like DINO for tokenization enables semantically aligned and high-fidelity latent representations, enabling next-generation visual generative models. Code will be publicly available at https://github.com/MKJia/DINO-Tok.
>
---
#### [new 061] MedROV: Towards Real-Time Open-Vocabulary Detection Across Diverse Medical Imaging Modalities
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MedROV，首个实时医学影像开放词汇目标检测模型。针对传统方法闭集限制与数据稀缺问题，构建大规模多模态数据集Omnis，采用伪标签策略与跨模态学习，实现对已知及未知结构的高效检测，显著提升性能并达到70 FPS实时速度。**

- **链接: [https://arxiv.org/pdf/2511.20650v1](https://arxiv.org/pdf/2511.20650v1)**

> **作者:** Tooba Tehreem Sheikh; Jean Lahoud; Rao Muhammad Anwer; Fahad Shahbaz Khan; Salman Khan; Hisham Cholakkal
>
> **摘要:** Traditional object detection models in medical imaging operate within a closed-set paradigm, limiting their ability to detect objects of novel labels. Open-vocabulary object detection (OVOD) addresses this limitation but remains underexplored in medical imaging due to dataset scarcity and weak text-image alignment. To bridge this gap, we introduce MedROV, the first Real-time Open Vocabulary detection model for medical imaging. To enable open-vocabulary learning, we curate a large-scale dataset, Omnis, with 600K detection samples across nine imaging modalities and introduce a pseudo-labeling strategy to handle missing annotations from multi-source datasets. Additionally, we enhance generalization by incorporating knowledge from a large pre-trained foundation model. By leveraging contrastive learning and cross-modal representations, MedROV effectively detects both known and novel structures. Experimental results demonstrate that MedROV outperforms the previous state-of-the-art foundation model for medical image detection with an average absolute improvement of 40 mAP50, and surpasses closed-set detectors by more than 3 mAP50, while running at 70 FPS, setting a new benchmark in medical detection. Our source code, dataset, and trained model are available at https://github.com/toobatehreem/MedROV.
>
---
#### [new 062] From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像篡改定位（IML）中标注成本与定位精度的矛盾，提出弱监督框架BoxPromptIML。通过粗粒度区域标注降低标注成本，利用轻量学生模型与固定教师模型知识蒸馏实现高效定位，并借鉴人类记忆机制设计动态特征融合模块，显著提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.20359v1](https://arxiv.org/pdf/2511.20359v1)**

> **作者:** Zhiqing Guo; Dongdong Xi; Songlin Li; Gaobo Yang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Image manipulation localization (IML) faces a fundamental trade-off between minimizing annotation cost and achieving fine-grained localization accuracy. Existing fully-supervised IML methods depend heavily on dense pixel-level mask annotations, which limits scalability to large datasets or real-world deployment.In contrast, the majority of existing weakly-supervised IML approaches are based on image-level labels, which greatly reduce annotation effort but typically lack precise spatial localization. To address this dilemma, we propose BoxPromptIML, a novel weakly-supervised IML framework that effectively balances annotation cost and localization performance. Specifically, we propose a coarse region annotation strategy, which can generate relatively accurate manipulation masks at lower cost. To improve model efficiency and facilitate deployment, we further design an efficient lightweight student model, which learns to perform fine-grained localization through knowledge distillation from a fixed teacher model based on the Segment Anything Model (SAM). Moreover, inspired by the human subconscious memory mechanism, our feature fusion module employs a dual-guidance strategy that actively contextualizes recalled prototypical patterns with real-time observational cues derived from the input. Instead of passive feature extraction, this strategy enables a dynamic process of knowledge recollection, where long-term memory is adapted to the specific context of the current image, significantly enhancing localization accuracy and robustness. Extensive experiments across both in-distribution and out-of-distribution datasets show that BoxPromptIML outperforms or rivals fully-supervised models, while maintaining strong generalization, low annotation cost, and efficient deployment characteristics.
>
---
#### [new 063] Reading Between the Lines: Abstaining from VLM-Generated OCR Errors via Latent Representation Probes
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型（VLM）在场景文本视觉问答（STVQA）中因OCR错误导致的可靠性问题，提出通过探测模型内部表示来实现不确定时的拒绝回答。工作包括设计三种基于隐藏状态与注意力模式的轻量级探测器，实验表明其显著提升抽象准确率，揭示中间层表示更适合作为置信度信号。**

- **链接: [https://arxiv.org/pdf/2511.19806v1](https://arxiv.org/pdf/2511.19806v1)**

> **作者:** Jihan Yao; Achin Kulshrestha; Nathalie Rauschmayr; Reed Roberts; Banghua Zhu; Yulia Tsvetkov; Federico Tombari
>
> **摘要:** As VLMs are deployed in safety-critical applications, their ability to abstain from answering when uncertain becomes crucial for reliability, especially in Scene Text Visual Question Answering (STVQA) tasks. For example, OCR errors like misreading "50 mph" as "60 mph" could cause severe traffic accidents. This leads us to ask: Can VLMs know when they can't see? Existing abstention methods suggest pessimistic answers: they either rely on miscalibrated output probabilities or require semantic agreement unsuitable for OCR tasks. However, this failure may indicate we are looking in the wrong place: uncertainty signals could be hidden in VLMs' internal representations. Building on this insight, we propose Latent Representation Probing (LRP): training lightweight probes on hidden states or attention patterns. We explore three probe designs: concatenating representations across all layers, aggregating attention over visual tokens, and ensembling single layer probes by majority vote. Experiments on four benchmarks across image and video modalities show LRP improves abstention accuracy by 7.6\% over best baselines. Our analysis reveals: probes generalize across various uncertainty sources and datasets, and optimal signals emerge from intermediate rather than final layers. This establishes a principled framework for building deployment-ready AI systems by detecting confidence signals from internal states rather than unreliable outputs.
>
---
#### [new 064] MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对生成模型多偏好对齐中的“对齐税”问题，提出MapReduce LoRA与RaTE方法。通过并行训练并融合偏好特定的LoRA专家，以及学习可组合的奖励嵌入，在文本到图像、视频及语言任务中实现跨模态的高效多目标优化，显著提升生成质量与对齐效果。**

- **链接: [https://arxiv.org/pdf/2511.20629v1](https://arxiv.org/pdf/2511.20629v1)**

> **作者:** Chieh-Yun Chen; Zhonghao Wang; Qi Chen; Zhifan Ye; Min Shi; Yue Zhao; Yinan Zhao; Hui Qu; Wei-An Lin; Yiru Shen; Ajinkya Kale; Irfan Essa; Humphrey Shi
>
> **摘要:** Reinforcement learning from human feedback (RLHF) with reward models has advanced alignment of generative models to human aesthetic and perceptual preferences. However, jointly optimizing multiple rewards often incurs an alignment tax, improving one dimension while degrading others. To address this, we introduce two complementary methods: MapReduce LoRA and Reward-aware Token Embedding (RaTE). MapReduce LoRA trains preference-specific LoRA experts in parallel and iteratively merges them to refine a shared base model; RaTE learns reward-specific token embeddings that compose at inference for flexible preference control. Experiments on Text-to-Image generation (Stable Diffusion 3.5 Medium and FLUX.1-dev) show improvements of 36.1%, 4.6%, and 55.7%, and 32.7%, 4.3%, and 67.1% on GenEval, PickScore, and OCR, respectively. On Text-to-Video generation (HunyuanVideo), visual and motion quality improve by 48.1% and 90.0%, respectively. On the language task, Helpful Assistant, with Llama-2 7B, helpful and harmless improve by 43.4% and 136.7%, respectively. Our framework sets a new state-of-the-art multi-preference alignment recipe across modalities.
>
---
#### [new 065] A Storage-Efficient Feature for 3D Concrete Defect Segmentation to Replace Normal Vector
- **分类: cs.CV**

- **简介: 该论文针对3D混凝土缺陷分割中点云数据存储与计算开销大的问题，提出一种名为“相对角”的新特征，替代传统法向量。该特征仅需单维存储，通过熵评估有效保留损伤区域信息，使模型在保持性能的同时减少27.6%存储和83%输入通道，显著提升资源受限设备的处理效率。**

- **链接: [https://arxiv.org/pdf/2511.19760v1](https://arxiv.org/pdf/2511.19760v1)**

> **作者:** Linxin Hua; Jianghua Deng; Ye Lu
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Point cloud reconstruction of damage offers an effective solution to image-based methods vulnerable to background noise, yet its application is constrained by the high volume of 3D data. This study proposes a new feature, relative angle, computed as the angle between the normal vector of a point and the average normal vector of its parent point cloud. This single-dimensional feature provides directionality information equivalent to normal vectors for concrete surface defect characteristics. Through entropy-based feature evaluation, this study demonstrates the ability of relative angle to filter out redundant information in undamaged sections while retaining effective information in damaged sections. By training and testing with PointNet++, models based on the relative angles achieved similar performance to that of models based on normal vectors while delivering 27.6% storage reduction and 83% input channel compression. This novel feature has the potential to enable larger-batch execution on resource-constrained hardware without the necessity of architectural modifications to models.
>
---
#### [new 066] Concept-Aware Batch Sampling Improves Language-Image Pretraining
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言预训练中的数据筛选问题，提出Concept-Aware Batch Sampling（CABS）框架。通过构建包含1.28亿对图像文本的DataConcept数据集，实现在线、基于概念的动态批处理采样，支持多样性与高频物体优化，显著提升CLIP/SigLIP模型性能，为下游任务提供灵活高效的数据策略。**

- **链接: [https://arxiv.org/pdf/2511.20643v1](https://arxiv.org/pdf/2511.20643v1)**

> **作者:** Adhiraj Ghosh; Vishaal Udandarao; Thao Nguyen; Matteo Farina; Mehdi Cherti; Jenia Jitsev; Sewoong Oh; Elisa Ricci; Ludwig Schmidt; Matthias Bethge
>
> **备注:** Tech Report
>
> **摘要:** What data should a vision-language model be trained on? To answer this question, many data curation efforts center on the quality of a dataset. However, most of these existing methods are (i) offline, i.e. they produce a static dataset from a set of predetermined filtering criteria, and (ii) concept-agnostic, i.e. they use model-based filters which induce additional data biases. In this work, we go beyond such offline, concept-agnostic methods and advocate for more flexible, task-adaptive online concept-based curation. Our first contribution is DataConcept, a collection of 128M web-crawled image-text pairs annotated with fine-grained details about their concept composition. Building on DataConcept, we introduce Concept-Aware Batch Sampling (CABS), a simple yet effective batch sampling framework that flexibly constructs batches on-the-fly based on specific target distributions. We propose two variants: (i) Diversity Maximization (CABS-DM) to curate batches with a broad coverage of available concepts, and (ii) Frequency Maximization (CABS-FM) to curate batches with high object multiplicity. Through extensive evaluations across 28 benchmarks, we demonstrate that our CABS method significantly benefits CLIP/SigLIP model classes and yields highly performant models. Overall, CABS represents a strong open-source alternative to proprietary online data curation algorithms, enabling practitioners to define custom concept distributions that optimize for specific downstream tasks.
>
---
#### [new 067] BRIC: Bridging Kinematic Plans and Physical Control at Test Time
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对长时序人体动作生成中运动规划与物理控制间的执行偏差问题，提出BRIC框架。通过测试时动态调整物理控制器并引入轻量级引导机制，实现对扩散模型生成动作的实时修正，提升动作的物理合理性与环境适应性，有效解决执行漂移问题。**

- **链接: [https://arxiv.org/pdf/2511.20431v1](https://arxiv.org/pdf/2511.20431v1)**

> **作者:** Dohun Lim; Minji Kim; Jaewoon Lim; Sungchan Kim
>
> **摘要:** We propose BRIC, a novel test-time adaptation (TTA) framework that enables long-term human motion generation by resolving execution discrepancies between diffusion-based kinematic motion planners and reinforcement learning-based physics controllers. While diffusion models can generate diverse and expressive motions conditioned on text and scene context, they often produce physically implausible outputs, leading to execution drift during simulation. To address this, BRIC dynamically adapts the physics controller to noisy motion plans at test time, while preserving pre-trained skills via a loss function that mitigates catastrophic forgetting. In addition, BRIC introduces a lightweight test-time guidance mechanism that steers the diffusion model in the signal space without updating its parameters. By combining both adaptation strategies, BRIC ensures consistent and physically plausible long-term executions across diverse environments in an effective and efficient manner. We validate the effectiveness of BRIC on a variety of long-term tasks, including motion composition, obstacle avoidance, and human-scene interaction, achieving state-of-the-art performance across all tasks.
>
---
#### [new 068] TaCo: Capturing Spatio-Temporal Semantic Consistency in Remote Sensing Change Detection
- **分类: cs.CV**

- **简介: 该论文针对遥感变化检测任务，解决现有方法仅依赖掩码监督导致的时间语义不一致问题。提出TaCo框架，通过文本引导的过渡生成器和时空语义联合约束，建模双时相间的语义演变，提升变化检测的语义一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.20306v1](https://arxiv.org/pdf/2511.20306v1)**

> **作者:** Han Guo; Chenyang Liu; Haotian Zhang; Bowen Chen; Zhengxia Zou; Zhenwei Shi
>
> **摘要:** Remote sensing change detection (RSCD) aims to identify surface changes across bi-temporal satellite images. Most previous methods rely solely on mask supervision, which effectively guides spatial localization but provides limited constraints on the temporal semantic transitions. Consequently, they often produce spatially coherent predictions while still suffering from unresolved semantic inconsistencies. To address this limitation, we propose TaCo, a spatio-temporal semantic consistent network, which enriches the existing mask-supervised framework with a spatio-temporal semantic joint constraint. TaCo conceptualizes change as a semantic transition between bi-temporal states, in which one temporal feature representation can be derived from the other via dedicated transition features. To realize this, we introduce a Text-guided Transition Generator that integrates textual semantics with bi-temporal visual features to construct the cross-temporal transition features. In addition, we propose a spatio-temporal semantic joint constraint consisting of bi-temporal reconstruct constraints and a transition constraint: the former enforces alignment between reconstructed and original features, while the latter enhances discrimination for changes. This design can yield substantial performance gains without introducing any additional computational overhead during inference. Extensive experiments on six public datasets, spanning both binary and semantic change detection tasks, demonstrate that TaCo consistently achieves SOTA performance.
>
---
#### [new 069] Efficient Transferable Optimal Transport via Min-Sliced Transport Plans
- **分类: cs.CV**

- **简介: 该论文研究最优传输中的可迁移性问题，针对计算成本高、难以跨分布复用的问题，提出min-STP框架。通过理论分析与高效批量算法，证明优化切片在分布微小变化下仍具稳定性，实现一次训练多任务复用，显著提升点云对齐与生成建模的效率。**

- **链接: [https://arxiv.org/pdf/2511.19741v1](https://arxiv.org/pdf/2511.19741v1)**

> **作者:** Xinran Liu; Elaheh Akbari; Rocio Diaz Martin; Navid NaderiAlizadeh; Soheil Kolouri
>
> **摘要:** Optimal Transport (OT) offers a powerful framework for finding correspondences between distributions and addressing matching and alignment problems in various areas of computer vision, including shape analysis, image generation, and multimodal tasks. The computation cost of OT, however, hinders its scalability. Slice-based transport plans have recently shown promise for reducing the computational cost by leveraging the closed-form solutions of 1D OT problems. These methods optimize a one-dimensional projection (slice) to obtain a conditional transport plan that minimizes the transport cost in the ambient space. While efficient, these methods leave open the question of whether learned optimal slicers can transfer to new distribution pairs under distributional shift. Understanding this transferability is crucial in settings with evolving data or repeated OT computations across closely related distributions. In this paper, we study the min-Sliced Transport Plan (min-STP) framework and investigate the transferability of optimized slicers: can a slicer trained on one distribution pair yield effective transport plans for new, unseen pairs? Theoretically, we show that optimized slicers remain close under slight perturbations of the data distributions, enabling efficient transfer across related tasks. To further improve scalability, we introduce a minibatch formulation of min-STP and provide statistical guarantees on its accuracy. Empirically, we demonstrate that the transferable min-STP achieves strong one-shot matching performance and facilitates amortized training for point cloud alignment and flow-based generative modeling.
>
---
#### [new 070] Cross-Domain Generalization of Multimodal LLMs for Global Photovoltaic Assessment
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文针对分布式光伏系统全球普查中因缺乏标注数据导致的跨区域识别难题，提出基于多模态大模型的跨域泛化方法。通过结构化提示与微调，实现检测、定位与量化一体化，显著提升在未见区域的性能稳定性，解决了传统视觉模型泛化能力差的问题。**

- **链接: [https://arxiv.org/pdf/2511.19537v1](https://arxiv.org/pdf/2511.19537v1)**

> **作者:** Muhao Guo; Yang Weng
>
> **备注:** 5 pages, 7 figures
>
> **摘要:** The rapid expansion of distributed photovoltaic (PV) systems poses challenges for power grid management, as many installations remain undocumented. While satellite imagery provides global coverage, traditional computer vision (CV) models such as CNNs and U-Nets require extensive labeled data and fail to generalize across regions. This study investigates the cross-domain generalization of a multimodal large language model (LLM) for global PV assessment. By leveraging structured prompts and fine-tuning, the model integrates detection, localization, and quantification within a unified schema. Cross-regional evaluation using the $Δ$F1 metric demonstrates that the proposed model achieves the smallest performance degradation across unseen regions, outperforming conventional CV and transformer baselines. These results highlight the robustness of multimodal LLMs under domain shift and their potential for scalable, transferable, and interpretable global PV mapping.
>
---
#### [new 071] Object-Centric Vision Token Pruning for Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型中视觉令牌冗余导致计算效率低的问题，提出OC-VTP方法。通过轻量级预训练的物体中心化视觉令牌剪枝器，直接且保证地选择最具代表性的视觉令牌，提升推理效率同时保持高精度，无需微调现有模型，具有良好的可解释性。**

- **链接: [https://arxiv.org/pdf/2511.20439v1](https://arxiv.org/pdf/2511.20439v1)**

> **作者:** Guangyuan Li; Rongzhen Zhao; Jinhong Deng; Yanbo Wang; Joni Pajarinen
>
> **摘要:** In Vision Language Models (VLMs), vision tokens are quantity-heavy yet information-dispersed compared with language tokens, thus consume too much unnecessary computation. Pruning redundant vision tokens for high VLM inference efficiency has been continuously studied but all existing methods resort to indirect and non-guaranteed ways. We propose OC-VTP, a direct and guaranteed approach to select the most representative vision tokens for high-efficiency yet accuracy-preserving VLM inference. Our OC-VTP requires merely light-weight pre-training of a small object-centric vision token pruner, which can then be inserted into existing VLMs, without fine-tuning of any models on any datasets. It is gauranteed that the most representative vision tokens are kept by minimizing the error in reconstructing the original unpruned tokens from the selected ones. Across any vision pruning ratios, i.e., inference efficiency, our OC-VTP consistently helps mainstream VLMs to preserve the highest inference accuracy. Our pruning also demonstrates interesting interpretability. Our codes are available at https://github.com/GarryLarry010131/OC-VTP.
>
---
#### [new 072] LocateAnything3D: Vision-Language 3D Detection with Chain-of-Sight
- **分类: cs.CV**

- **简介: 该论文提出LocateAnything3D，解决视觉语言模型在3D多物体检测中的缺失问题。通过构建链式视线（CoS）序列，将3D检测转为逐令牌预测任务，先2D定位后推断距离、尺寸与姿态，实现开放词汇与零样本泛化，在Omni3D上达49.89 AP_3D，显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20648v1](https://arxiv.org/pdf/2511.20648v1)**

> **作者:** Yunze Man; Shihao Wang; Guowen Zhang; Johan Bjorck; Zhiqi Li; Liang-Yan Gui; Jim Fan; Jan Kautz; Yu-Xiong Wang; Zhiding Yu
>
> **备注:** Tech report. Project page: https://nvlabs.github.io/LocateAnything3D/
>
> **摘要:** To act in the world, a model must name what it sees and know where it is in 3D. Today's vision-language models (VLMs) excel at open-ended 2D description and grounding, yet multi-object 3D detection remains largely missing from the VLM toolbox. We present LocateAnything3D, a VLM-native recipe that casts 3D detection as a next-token prediction problem. The key is a short, explicit Chain-of-Sight (CoS) sequence that mirrors how human reason from images: find an object in 2D, then infer its distance, size, and pose. The decoder first emits 2D detections as a visual chain-of-thought, then predicts 3D boxes under an easy-to-hard curriculum: across objects, a near-to-far order reduces early ambiguity and matches ego-centric utility; within each object, a center-from-camera, dimensions, and rotation factorization ranks information by stability and learnability. This VLM-native interface preserves open-vocabulary and visual-prompting capability without specialized heads. On the challenging Omni3D benchmark, our model achieves state-of-the-art results, with 49.89 AP_3D, surpassing the previous best by +15.51 absolute improvement even when the baseline is given ground-truth 2D boxes. It also generalizes zero-shot to held-out categories with strong robustness. By turning 3D detection into a disciplined next-token problem, LocateAnything3D offers a practical foundation for models to perceive in 3D.
>
---
#### [new 073] Zoo3D: Zero-Shot 3D Object Detection at Scene Level
- **分类: cs.CV**

- **简介: 该论文提出Zoo3D，首个无需训练的3D物体检测框架，解决开放词汇下未知物体识别难题。通过2D实例掩码图聚类生成3D框，并结合视图选择与共识掩码实现零样本语义标注。支持点云与图像输入，在ScanNet200和ARKitScenes上达到领先性能，验证了训练自由方法在真实场景中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.20253v1](https://arxiv.org/pdf/2511.20253v1)**

> **作者:** Andrey Lemeshko; Bulat Gabdullin; Nikita Drozdov; Anton Konushin; Danila Rukhovich; Maksim Kolodiazhnyi
>
> **摘要:** 3D object detection is fundamental for spatial understanding. Real-world environments demand models capable of recognizing diverse, previously unseen objects, which remains a major limitation of closed-set methods. Existing open-vocabulary 3D detectors relax annotation requirements but still depend on training scenes, either as point clouds or images. We take this a step further by introducing Zoo3D, the first training-free 3D object detection framework. Our method constructs 3D bounding boxes via graph clustering of 2D instance masks, then assigns semantic labels using a novel open-vocabulary module with best-view selection and view-consensus mask generation. Zoo3D operates in two modes: the zero-shot Zoo3D$_0$, which requires no training at all, and the self-supervised Zoo3D$_1$, which refines 3D box prediction by training a class-agnostic detector on Zoo3D$_0$-generated pseudo labels. Furthermore, we extend Zoo3D beyond point clouds to work directly with posed and even unposed images. Across ScanNet200 and ARKitScenes benchmarks, both Zoo3D$_0$ and Zoo3D$_1$ achieve state-of-the-art results in open-vocabulary 3D object detection. Remarkably, our zero-shot Zoo3D$_0$ outperforms all existing self-supervised methods, hence demonstrating the power and adaptability of training-free, off-the-shelf approaches for real-world 3D understanding. Code is available at https://github.com/col14m/zoo3d .
>
---
#### [new 074] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion
- **分类: cs.CV**

- **简介: 该论文针对域自适应点云补全（DA PCC）任务，解决源域与目标域间几何语义差异导致的性能下降问题。提出DAPointMamba框架，通过三类跨域对齐模块，实现局部对齐、空间一致性和全局语义对齐，兼具高效线性复杂度与强域适应能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20278v1](https://arxiv.org/pdf/2511.20278v1)**

> **作者:** Yinghui Li; Qianyu Zhou; Di Shao; Hao Yang; Ye Zhu; Richard Dazeley; Xuequan Lu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Domain adaptive point cloud completion (DA PCC) aims to narrow the geometric and semantic discrepancies between the labeled source and unlabeled target domains. Existing methods either suffer from limited receptive fields or quadratic complexity due to using CNNs or vision Transformers. In this paper, we present the first work that studies the adaptability of State Space Models (SSMs) in DA PCC and find that directly applying SSMs to DA PCC will encounter several challenges: directly serializing 3D point clouds into 1D sequences often disrupts the spatial topology and local geometric features of the target domain. Besides, the overlook of designs in the learning domain-agnostic representations hinders the adaptation performance. To address these issues, we propose a novel framework, DAPointMamba for DA PCC, that exhibits strong adaptability across domains and has the advantages of global receptive fields and efficient linear complexity. It has three novel modules. In particular, Cross-Domain Patch-Level Scanning introduces patch-level geometric correspondences, enabling effective local alignment. Cross-Domain Spatial SSM Alignment further strengthens spatial consistency by modulating patch features based on cross-domain similarity, effectively mitigating fine-grained structural discrepancies. Cross-Domain Channel SSM Alignment actively addresses global semantic gaps by interleaving and aligning feature channels. Extensive experiments on both synthetic and real-world benchmarks demonstrate that our DAPointMamba outperforms state-of-the-art methods with less computational complexity and inference latency.
>
---
#### [new 075] ReDirector: Creating Any-Length Video Retakes with Rotary Camera Encoding
- **分类: cs.CV**

- **简介: 该论文提出ReDirector，解决动态拍摄视频重拍中相机轨迹与长度不一致的问题。通过修正RoPE使用方式，引入相机条件的旋转编码（RoCE），实现跨视角、多长度视频的精准重拍，提升相机可控性、几何一致性与视频质量。**

- **链接: [https://arxiv.org/pdf/2511.19827v1](https://arxiv.org/pdf/2511.19827v1)**

> **作者:** Byeongjun Park; Byung-Hoon Kim; Hyungjin Chung; Jong Chul Ye
>
> **备注:** Project page: https://byeongjun-park.github.io/ReDirector/
>
> **摘要:** We present ReDirector, a novel camera-controlled video retake generation method for dynamically captured variable-length videos. In particular, we rectify a common misuse of RoPE in previous works by aligning the spatiotemporal positions of the input video and the target retake. Moreover, we introduce Rotary Camera Encoding (RoCE), a camera-conditioned RoPE phase shift that captures and integrates multi-view relationships within and across the input and target videos. By integrating camera conditions into RoPE, our method generalizes to out-of-distribution camera trajectories and video lengths, yielding improved dynamic object localization and static background preservation. Extensive experiments further demonstrate significant improvements in camera controllability, geometric consistency, and video quality across various trajectories and lengths.
>
---
#### [new 076] Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Agent0-VL，一种自进化视觉语言代理，旨在解决多模态推理中依赖人工标注、评估易幻觉的问题。通过整合工具使用于推理与自评估，实现证据驱动的自我修正，无需外部奖励，显著提升几何与科学图像分析能力。**

- **链接: [https://arxiv.org/pdf/2511.19900v1](https://arxiv.org/pdf/2511.19900v1)**

> **作者:** Jiaqi Liu; Kaiwen Xiong; Peng Xia; Yiyang Zhou; Haonian Ji; Lu Feng; Siwei Han; Mingyu Ding; Huaxiu Yao
>
> **摘要:** Vision-language agents have achieved remarkable progress in a variety of multimodal reasoning tasks; however, their learning remains constrained by the limitations of human-annotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely text-based self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a self-evolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-external-reward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on geometric problem solving and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the base model. Our code is available at \href{https://github.com/aiming-lab/Agent0/Agent0-VL}{this https URL}.
>
---
#### [new 077] Training-Free Generation of Diverse and High-Fidelity Images via Prompt Semantic Space Optimization
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对文本生成图像模型中图像多样性不足的问题，提出无需训练的TPSO方法。通过优化提示词嵌入空间，探索低频语义区域，在不降低图像质量的前提下显著提升生成多样性，适用于多种扩散模型。**

- **链接: [https://arxiv.org/pdf/2511.19811v1](https://arxiv.org/pdf/2511.19811v1)**

> **作者:** Debin Meng; Chen Jin; Zheng Gao; Yanran Li; Ioannis Patras; Georgios Tzimiropoulos
>
> **备注:** under review
>
> **摘要:** Image diversity remains a fundamental challenge for text-to-image diffusion models. Low-diversity models tend to generate repetitive outputs, increasing sampling redundancy and hindering both creative exploration and downstream applications. A primary cause is that generation often collapses toward a strong mode in the learned distribution. Existing attempts to improve diversity, such as noise resampling, prompt rewriting, or steering-based guidance, often still collapse to dominant modes or introduce distortions that degrade image quality. In light of this, we propose Token-Prompt embedding Space Optimization (TPSO), a training-free and model-agnostic module. TPSO introduces learnable parameters to explore underrepresented regions of the token embedding space, reducing the tendency of the model to repeatedly generate samples from strong modes of the learned distribution. At the same time, the prompt-level space provides a global semantic constraint that regulates distribution shifts, preventing quality degradation while maintaining high fidelity. Extensive experiments on MS-COCO and three diffusion backbones show that TPSO significantly enhances generative diversity, improving baseline performance from 1.10 to 4.18 points, without sacrificing image quality. Code will be released upon acceptance.
>
---
#### [new 078] Dance Style Classification using Laban-Inspired and Frequency-Domain Motion Features
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于舞蹈风格分类任务，旨在解决不同舞蹈风格间动作相似导致识别困难的问题。通过提取视频中的姿态数据，结合莱班运动分析的时空特征与傅里叶变换的频域特征，构建轻量级可解释的运动表征，实现高效精准的舞蹈风格分类。**

- **链接: [https://arxiv.org/pdf/2511.20469v1](https://arxiv.org/pdf/2511.20469v1)**

> **作者:** Ben Hamscher; Arnold Brosch; Nicolas Binninger; Maksymilian Jan Dejna; Kira Maag
>
> **摘要:** Dance is an essential component of human culture and serves as a tool for conveying emotions and telling stories. Identifying and distinguishing dance genres based on motion data is a complex problem in human activity recognition, as many styles share similar poses, gestures, and temporal motion patterns. This work presents a lightweight framework for classifying dance styles that determines motion characteristics based on pose estimates extracted from videos. We propose temporal-spatial descriptors inspired by Laban Movement Analysis. These features capture local joint dynamics such as velocity, acceleration, and angular movement of the upper body, enabling a structured representation of spatial coordination. To further encode rhythmic and periodic aspects of movement, we integrate Fast Fourier Transform features that characterize movement patterns in the frequency domain. The proposed approach achieves robust classification of different dance styles with low computational effort, as complex model architectures are not required, and shows that interpretable motion representations can effectively capture stylistic nuances.
>
---
#### [new 079] Bootstrapping Physics-Grounded Video Generation through VLM-Guided Iterative Self-Refinement
- **分类: cs.CV**

- **简介: 该论文针对视频生成中物理一致性不足的问题，提出一种无需训练的迭代自精炼框架。利用视觉语言模型提供物理反馈，通过多模态思维链逐步优化生成提示，提升视频与真实物理规律的契合度。在PhyIQ基准上，Physics-IQ得分从56.31提升至62.38。**

- **链接: [https://arxiv.org/pdf/2511.20280v1](https://arxiv.org/pdf/2511.20280v1)**

> **作者:** Yang Liu; Xilin Zhao; Peisong Wen; Siran Dai; Qingming Huang
>
> **备注:** ICCV 2025 Physics-IQ Challenge Third Place Solution
>
> **摘要:** Recent progress in video generation has led to impressive visual quality, yet current models still struggle to produce results that align with real-world physical principles. To this end, we propose an iterative self-refinement framework that leverages large language models and vision-language models to provide physics-aware guidance for video generation. Specifically, we introduce a multimodal chain-of-thought (MM-CoT) process that refines prompts based on feedback from physical inconsistencies, progressively enhancing generation quality. This method is training-free and plug-and-play, making it readily applicable to a wide range of video generation models. Experiments on the PhyIQ benchmark show that our method improves the Physics-IQ score from 56.31 to 62.38. We hope this work serves as a preliminary exploration of physics-consistent video generation and may offer insights for future research.
>
---
#### [new 080] New York Smells: A Large Multimodal Dataset for Olfaction
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出纽约气味数据集（New York Smells），解决机器缺乏自然场景下多模态嗅觉数据的问题。包含7000对图像与气味数据，用于跨模态检索、气味识别与细粒度分类任务。实验表明视觉信息可促进嗅觉表征学习，所学特征优于传统手工特征。**

- **链接: [https://arxiv.org/pdf/2511.20544v1](https://arxiv.org/pdf/2511.20544v1)**

> **作者:** Ege Ozguroglu; Junbang Liang; Ruoshi Liu; Mia Chiquier; Michael DeTienne; Wesley Wei Qian; Alexandra Horowitz; Andrew Owens; Carl Vondrick
>
> **备注:** Project website at https://smell.cs.columbia.edu
>
> **摘要:** While olfaction is central to how animals perceive the world, this rich chemical sensory modality remains largely inaccessible to machines. One key bottleneck is the lack of diverse, multimodal olfactory training data collected in natural settings. We present New York Smells, a large dataset of paired image and olfactory signals captured ``in the wild.'' Our dataset contains 7,000 smell-image pairs from 3,500 distinct objects across indoor and outdoor environments, with approximately 70$\times$ more objects than existing olfactory datasets. Our benchmark has three tasks: cross-modal smell-to-image retrieval, recognizing scenes, objects, and materials from smell alone, and fine-grained discrimination between grass species. Through experiments on our dataset, we find that visual data enables cross-modal olfactory representation learning, and that our learned olfactory representations outperform widely-used hand-crafted features.
>
---
#### [new 081] Multi-Context Fusion Transformer for Pedestrian Crossing Intention Prediction in Urban Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对城市环境中自动驾驶车辆的行人过街意图预测任务，提出多上下文融合Transformer（MFT）模型。通过融合行为、环境、定位和车辆运动四维上下文信息，采用渐进式注意力机制实现跨上下文深度交互，提升预测精度。在三个公开数据集上取得领先性能。**

- **链接: [https://arxiv.org/pdf/2511.20011v1](https://arxiv.org/pdf/2511.20011v1)**

> **作者:** Yuanzhe Li; Hang Zhong; Steffen Müller
>
> **摘要:** Pedestrian crossing intention prediction is essential for autonomous vehicles to improve pedestrian safety and reduce traffic accidents. However, accurate pedestrian intention prediction in urban environments remains challenging due to the multitude of factors affecting pedestrian behavior. In this paper, we propose a multi-context fusion Transformer (MFT) that leverages diverse numerical contextual attributes across four key dimensions, encompassing pedestrian behavior context, environmental context, pedestrian localization context and vehicle motion context, to enable accurate pedestrian intention prediction. MFT employs a progressive fusion strategy, where mutual intra-context attention enables reciprocal interactions within each context, thereby facilitating feature sequence fusion and yielding a context token as a context-specific representation. This is followed by mutual cross-context attention, which integrates features across contexts with a global CLS token serving as a compact multi-context representation. Finally, guided intra-context attention refines context tokens within each context through directed interactions, while guided cross-context attention strengthens the global CLS token to promote multi-context fusion via guided information propagation, yielding deeper and more efficient integration. Experimental results validate the superiority of MFT over state-of-the-art methods, achieving accuracy rates of 73%, 93%, and 90% on the JAADbeh, JAADall, and PIE datasets, respectively. Extensive ablation studies are further conducted to investigate the effectiveness of the network architecture and contribution of different input context. Our code is open-source: https://github.com/ZhongHang0307/Multi-Context-Fusion-Transformer.
>
---
#### [new 082] PixelDiT: Pixel Diffusion Transformers for Image Generation
- **分类: cs.CV**

- **简介: 该论文提出PixelDiT，一种直接在像素空间进行图像生成的单阶段扩散Transformer模型，解决传统方法依赖预训练自编码器导致的重建损失与误差累积问题。通过双层级架构实现全局语义与细节纹理的协同建模，在ImageNet上达到1.61 FID，且在文本到图像生成任务中表现接近最优的潜空间模型。**

- **链接: [https://arxiv.org/pdf/2511.20645v1](https://arxiv.org/pdf/2511.20645v1)**

> **作者:** Yongsheng Yu; Wei Xiong; Weili Nie; Yichen Sheng; Shiqiu Liu; Jiebo Luo
>
> **摘要:** Latent-space modeling has been the standard for Diffusion Transformers (DiTs). However, it relies on a two-stage pipeline where the pretrained autoencoder introduces lossy reconstruction, leading to error accumulation while hindering joint optimization. To address these issues, we propose PixelDiT, a single-stage, end-to-end model that eliminates the need for the autoencoder and learns the diffusion process directly in the pixel space. PixelDiT adopts a fully transformer-based architecture shaped by a dual-level design: a patch-level DiT that captures global semantics and a pixel-level DiT that refines texture details, enabling efficient training of a pixel-space diffusion model while preserving fine details. Our analysis reveals that effective pixel-level token modeling is essential to the success of pixel diffusion. PixelDiT achieves 1.61 FID on ImageNet 256x256, surpassing existing pixel generative models by a large margin. We further extend PixelDiT to text-to-image generation and pretrain it at the 1024x1024 resolution in pixel space. It achieves 0.74 on GenEval and 83.5 on DPG-bench, approaching the best latent diffusion models.
>
---
#### [new 083] OmniAlpha: A Sequence-to-Sequence Framework for Unified Multi-Task RGBA Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出OmniAlpha，首个统一的序列到序列RGBA生成与编辑框架。针对现有模型在RGBA操作中任务分散、缺乏多任务统一的问题，提出新型RoPE结构与AlphaLayers数据集，实现21项任务联合训练，显著提升生成质量与效率，验证了统一模型在多层图像处理中的优越性。**

- **链接: [https://arxiv.org/pdf/2511.20211v1](https://arxiv.org/pdf/2511.20211v1)**

> **作者:** Hao Yu; Jiabo Zhan; Zile Wang; Jinglin Wang; Huaisong Zhang; Hongyu Li; Xinrui Chen; Yongxian Wei; Chun Yuan
>
> **摘要:** Generative models have excelled in RGB synthesis, but real-world applications require RGBA manipulation. This has led to a fragmented landscape: specialized, single-task models handle alpha but lack versatility, while unified multi-task frameworks are confined to the RGB domain. To bridge this critical gap, we propose OmniAlpha, the first unified, multi-task generative framework for sequence-to-sequence RGBA image generation and editing. Its architecture features MSRoPE-BiL, a novel RoPE method with a bi-directionally extendable layer axis for its Diffusion Transformer (DiT) backbone, enabling the concurrent processing of multiple input and target RGBA layers. To power this framework, we introduce AlphaLayers, a new dataset of 1,000 high-quality, multi-layer triplets, built via a novel automated synthesis and filter pipeline. Jointly training OmniAlpha on this dataset across a comprehensive suite of 21 diverse tasks, extensive experiments demonstrate that our unified approach consistently outperforms strong, specialized baselines. Most notably, OmniAlpha achieves a dramatic 84.8% relative reduction in SAD for mask-free matting on AIM-500 and wins over 90% of human preferences in layer-conditioned completion. Our work proves that a unified, multi-task model can learn a superior shared representation for RGBA, paving the way for more powerful, layer-aware generative systems.
>
---
#### [new 084] IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants
- **分类: cs.CV; cs.AI; cs.HC; cs.RO**

- **简介: 该论文提出IndEgo数据集，用于工业场景下的第一人称助手研究。针对协作任务中多模态理解与错误检测难题，构建了包含3460段第一人称和1092段第三人称视频的数据集，涵盖多种工业任务，提供丰富标注与基准测试，推动协作任务理解、错误检测与推理问答的模型发展。**

- **链接: [https://arxiv.org/pdf/2511.19684v1](https://arxiv.org/pdf/2511.19684v1)**

> **作者:** Vivek Chavan; Yasmina Imgrund; Tung Dao; Sanwantri Bai; Bosong Wang; Ze Lu; Oliver Heimann; Jörg Krüger
>
> **备注:** Accepted to NeurIPS 2025 D&B Track. Project Page: https://indego-dataset.github.io/
>
> **摘要:** We introduce IndEgo, a multimodal egocentric and exocentric dataset addressing common industrial tasks, including assembly/disassembly, logistics and organisation, inspection and repair, woodworking, and others. The dataset contains 3,460 egocentric recordings (approximately 197 hours), along with 1,092 exocentric recordings (approximately 97 hours). A key focus of the dataset is collaborative work, where two workers jointly perform cognitively and physically intensive tasks. The egocentric recordings include rich multimodal data and added context via eye gaze, narration, sound, motion, and others. We provide detailed annotations (actions, summaries, mistake annotations, narrations), metadata, processed outputs (eye gaze, hand pose, semi-dense point cloud), and benchmarks on procedural and non-procedural task understanding, Mistake Detection, and reasoning-based Question Answering. Baseline evaluations for Mistake Detection, Question Answering and collaborative task understanding show that the dataset presents a challenge for the state-of-the-art multimodal models. Our dataset is available at: https://huggingface.co/datasets/FraunhoferIPK/IndEgo
>
---
#### [new 085] On the Utility of Foundation Models for Fast MRI: Vision-Language-Guided Image Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像重建任务，旨在解决欠采样MRI重建中细节丢失与感知质量差的问题。提出基于视觉-语言基础模型的语义分布引导框架，利用高阶语义先验提升重建质量，通过对比学习对齐语义特征，显著改善图像细节与主观感知效果。**

- **链接: [https://arxiv.org/pdf/2511.19641v1](https://arxiv.org/pdf/2511.19641v1)**

> **作者:** Ruimin Feng; Xingxin He; Ronald Mercer; Zachary Stewart; Fang Liu
>
> **摘要:** Purpose: To investigate whether a vision-language foundation model can enhance undersampled MRI reconstruction by providing high-level contextual information beyond conventional priors. Methods: We proposed a semantic distribution-guided reconstruction framework that uses a pre-trained vision-language foundation model to encode both the reconstructed image and auxiliary information into high-level semantic features. A contrastive objective aligns the reconstructed representation with the target semantic distribution, ensuring consistency with high-level perceptual cues. The proposed objective works with various deep learning-based reconstruction methods and can flexibly incorporate semantic priors from multimodal sources. To test the effectiveness of these semantic priors, we evaluated reconstruction results guided by priors derived from either image-only or image-language auxiliary information. Results: Experiments on knee and brain datasets demonstrate that semantic priors from images preserve fine anatomical structures and achieve superior perceptual quality, as reflected in lower LPIPS values, higher Tenengrad scores, and improved scores in the reader study, compared with conventional regularization. The image-language information further expands the semantic distribution and enables high-level control over reconstruction attributes. Across all evaluations, the contrastive objective consistently guided the reconstructed features toward the desired semantic distributions while maintaining data fidelity, demonstrating the effectiveness of the proposed optimization framework. Conclusion: The study highlights that vision-language foundation models can improve undersampled MRI reconstruction through semantic-space optimization.
>
---
#### [new 086] A Training-Free Approach for Multi-ID Customization via Attention Adjustment and Spatial Control
- **分类: cs.CV**

- **简介: 该论文研究多身份图像定制任务，旨在无训练条件下融合多人身份生成一致图像。针对复制粘贴问题与文本控制弱的挑战，提出ID解耦交叉注意力与空间控制策略，构建IDBench基准，实现高质量、高可控性生成。**

- **链接: [https://arxiv.org/pdf/2511.20401v1](https://arxiv.org/pdf/2511.20401v1)**

> **作者:** Jiawei Lin; Guanlong Jiao; Jianjin Xu
>
> **摘要:** Multi-ID customization is an interesting topic in computer vision and attracts considerable attention recently. Given the ID images of multiple individuals, its purpose is to generate a customized image that seamlessly integrates them while preserving their respective identities. Compared to single-ID customization, multi-ID customization is much more difficult and poses two major challenges. First, since the multi-ID customization model is trained to reconstruct an image from the cropped person regions, it often encounters the copy-paste issue during inference, leading to lower quality. Second, the model also suffers from inferior text controllability. The generated result simply combines multiple persons into one image, regardless of whether it is aligned with the input text. In this work, we propose MultiID to tackle this challenging task in a training-free manner. Since the existing single-ID customization models have less copy-paste issue, our key idea is to adapt these models to achieve multi-ID customization. To this end, we present an ID-decoupled cross-attention mechanism, injecting distinct ID embeddings into the corresponding image regions and thus generating multi-ID outputs. To enhance the generation controllability, we introduce three critical strategies, namely the local prompt, depth-guided spatial control, and extended self-attention, making the results more consistent with the text prompts and ID images. We also carefully build a benchmark, called IDBench, for evaluation. The extensive qualitative and quantitative results demonstrate the effectiveness of MultiID in solving the aforementioned two challenges. Its performance is comparable or even better than the training-based multi-ID customization methods.
>
---
#### [new 087] Maritime Small Object Detection from UAVs using Deep Learning with Altitude-Aware Dynamic Tiling
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对无人机在海上搜救中检测小目标困难的问题，提出一种高度感知的动态切片方法。通过根据飞行高度自适应调整图像切片大小和数量，提升小目标检测精度与推理速度。实验表明，该方法在保持高检测性能的同时，显著降低计算开销，适用于复杂多变的海上搜救场景。**

- **链接: [https://arxiv.org/pdf/2511.19728v1](https://arxiv.org/pdf/2511.19728v1)**

> **作者:** Sakib Ahmed; Oscar Pizarro
>
> **备注:** This is the author's accepted version of an article that has been published by IEEE. The final published version is available at IEEE Xplore
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are crucial in Search and Rescue (SAR) missions due to their ability to monitor vast maritime areas. However, small objects often remain difficult to detect from high altitudes due to low object-to-background pixel ratios. We propose an altitude-aware dynamic tiling method that scales and adaptively subdivides the image into tiles for enhanced small object detection. By integrating altitude-dependent scaling with an adaptive tiling factor, we reduce unnecessary computation while maintaining detection performance. Tested on the SeaDronesSee dataset [1] with YOLOv5 [2] and Slicing Aided Hyper Inference (SAHI) framework [3], our approach improves Mean Average Precision (mAP) for small objects by 38% compared to a baseline and achieves more than double the inference speed compared to static tiling. This approach enables more efficient and accurate UAV-based SAR operations under diverse conditions.
>
---
#### [new 088] Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究统一多模态模型中理解与生成的关系，旨在解决“理解是否真正指导生成”这一核心问题。通过构建UniSandbox评估框架与合成数据集，发现存在显著的理解-生成差距。研究揭示链式思维（CoT）可有效弥合差距，并提出自训练方法实现隐式推理，为未来模型设计提供新思路。**

- **链接: [https://arxiv.org/pdf/2511.20561v1](https://arxiv.org/pdf/2511.20561v1)**

> **作者:** Yuwei Niu; Weiyang Jin; Jiaqi Liao; Chaoran Feng; Peng Jin; Bin Lin; Zongjian Li; Bin Zhu; Weihao Yu; Li Yuan
>
> **摘要:** Recent years have witnessed significant progress in Unified Multimodal Models, yet a fundamental question remains: Does understanding truly inform generation? To investigate this, we introduce UniSandbox, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: reasoning generation and knowledge transfer. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned knowledge, and also discover that query-based architectures inherently exhibit latent CoT-like properties that affect this transfer. UniSandbox provides preliminary insights for designing future unified architectures and training strategies that truly bridge the gap between understanding and generation. Code and data are available at https://github.com/PKU-YuanGroup/UniSandBox
>
---
#### [new 089] Coupled Physics-Gated Adaptation: Spatially Decoding Volumetric Photochemical Conversion in Complex 3D-Printed Objects
- **分类: cs.CV**

- **简介: 该论文提出耦合物理门控自适应（C-PGA）框架，解决复杂3D打印物体中光化学转化的体积预测难题。针对传统视觉模型无法建模光学与材料物理耦合的问题，利用多模态融合与动态特征调制，从3D视觉数据精准预测非可视化的化学状态，实现虚拟化学表征。**

- **链接: [https://arxiv.org/pdf/2511.19913v1](https://arxiv.org/pdf/2511.19913v1)**

> **作者:** Maryam Eftekharifar; Churun Zhang; Jialiang Wei; Xudong Cao; Hossein Heidari
>
> **摘要:** We present a framework that pioneers the prediction of photochemical conversion in complex three-dimensionally printed objects, introducing a challenging new computer vision task: predicting dense, non-visual volumetric physical properties from 3D visual data. This approach leverages the largest-ever optically printed 3D specimen dataset, comprising a large family of parametrically designed complex minimal surface structures that have undergone terminal chemical characterisation. Conventional vision models are ill-equipped for this task, as they lack an inductive bias for the coupled, non-linear interactions of optical physics (diffraction, absorption) and material physics (diffusion, convection) that govern the final chemical state. To address this, we propose Coupled Physics-Gated Adaptation (C-PGA), a novel multimodal fusion architecture. Unlike standard concatenation, C-PGA explicitly models physical coupling by using sparse geometrical and process parameters (e.g., surface transport, print layer height) as a Query to dynamically gate and adapt the dense visual features via feature-wise linear modulation (FiLM). This mechanism spatially modulates dual 3D visual streams-extracted by parallel 3D-CNNs processing raw projection stacks and their diffusion-diffraction corrected counterparts allowing the model to recalibrate its visual perception based on the physical context. This approach offers a breakthrough in virtual chemical characterisation, eliminating the need for traditional post-print measurements and enabling precise control over the chemical conversion state.
>
---
#### [new 090] DRL-Guided Neural Batch Sampling for Semi-Supervised Pixel-Level Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文针对工业视觉检测中缺陷样本稀缺导致的异常检测难题，提出一种半监督深度强化学习框架。通过强化学习引导的神经批采样器，自适应选择关键图像块，结合自编码器与预测器，有效利用少量标注数据学习正常与异常模式，提升对细微缺陷的检测精度与定位能力。**

- **链接: [https://arxiv.org/pdf/2511.20270v1](https://arxiv.org/pdf/2511.20270v1)**

> **作者:** Amirhossein Khadivi Noghredeh; Abdollah Safari; Fatemeh Ziaeetabar; Firoozeh Haghighi
>
> **摘要:** Anomaly detection in industrial visual inspection is challenging due to the scarcity of defective samples. Most existing methods rely on unsupervised reconstruction using only normal data, often resulting in overfitting and poor detection of subtle defects. We propose a semi-supervised deep reinforcement learning framework that integrates a neural batch sampler, an autoencoder, and a predictor. The RL-based sampler adaptively selects informative patches by balancing exploration and exploitation through a composite reward. The autoencoder generates loss profiles highlighting abnormal regions, while the predictor performs segmentation in the loss-profile space. This interaction enables the system to effectively learn both normal and defective patterns with limited labeled data. Experiments on the MVTec AD dataset demonstrate that our method achieves higher accuracy and better localization of subtle anomalies than recent state-of-the-art approaches while maintaining low complexity, yielding an average improvement of 0.15 in F1_max and 0.06 in AUC, with a maximum gain of 0.37 in F1_max in the best case.
>
---
#### [new 091] HybriDLA: Hybrid Generation for Document Layout Analysis
- **分类: cs.CV**

- **简介: 该论文针对复杂现代文档布局分析任务，解决传统方法在多样元素数量和复杂布局下性能不足的问题。提出HybriDLA框架，融合扩散与自回归解码，并设计多尺度特征融合编码器，显著提升检测精度，达83.5% mAP，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.19919v1](https://arxiv.org/pdf/2511.19919v1)**

> **作者:** Yufan Chen; Omar Moured; Ruiping Liu; Junwei Zheng; Kunyu Peng; Jiaming Zhang; Rainer Stiefelhagen
>
> **备注:** Accepted by AAAI 2026 (Oral). Project page at https://yufanchen96.github.io/projects/HybriDLA
>
> **摘要:** Conventional document layout analysis (DLA) traditionally depends on empirical priors or a fixed set of learnable queries executed in a single forward pass. While sufficient for early-generation documents with a small, predetermined number of regions, this paradigm struggles with contemporary documents, which exhibit diverse element counts and increasingly complex layouts. To address challenges posed by modern documents, we present HybriDLA, a novel generative framework that unifies diffusion and autoregressive decoding within a single layer. The diffusion component iteratively refines bounding-box hypotheses, whereas the autoregressive component injects semantic and contextual awareness, enabling precise region prediction even in highly varied layouts. To further enhance detection quality, we design a multi-scale feature-fusion encoder that captures both fine-grained and high-level visual cues. This architecture elevates performance to 83.5% mean Average Precision (mAP). Extensive experiments on the DocLayNet and M$^6$Doc benchmarks demonstrate that HybriDLA sets a state-of-the-art performance, outperforming previous approaches. All data and models will be made publicly available at https://yufanchen96.github.io/projects/HybriDLA.
>
---
#### [new 092] StableTrack: Stabilizing Multi-Object Tracking on Low-Frequency Detections
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对低频检测下的多目标跟踪（MOT）问题，提出StableTrack方法。通过两阶段匹配策略与基于边界框的新型距离度量，提升跨帧关联精度，并融合视觉特征与卡尔曼滤波，显著改善低频检测下的跟踪稳定性，在MOT17上实现11.6%的HOTA提升。**

- **链接: [https://arxiv.org/pdf/2511.20418v1](https://arxiv.org/pdf/2511.20418v1)**

> **作者:** Matvei Shelukhan; Timur Mamedov; Karina Kvanchiani
>
> **摘要:** Multi-object tracking (MOT) is one of the most challenging tasks in computer vision, where it is important to correctly detect objects and associate these detections across frames. Current approaches mainly focus on tracking objects in each frame of a video stream, making it almost impossible to run the model under conditions of limited computing resources. To address this issue, we propose StableTrack, a novel approach that stabilizes the quality of tracking on low-frequency detections. Our method introduces a new two-stage matching strategy to improve the cross-frame association between low-frequency detections. We propose a novel Bbox-Based Distance instead of the conventional Mahalanobis distance, which allows us to effectively match objects using the Re-ID model. Furthermore, we integrate visual tracking into the Kalman Filter and the overall tracking pipeline. Our method outperforms current state-of-the-art trackers in the case of low-frequency detections, achieving $\textit{11.6%}$ HOTA improvement at $\textit{1}$ Hz on MOT17-val, while keeping up with the best approaches on the standard MOT17, MOT20, and DanceTrack benchmarks with full-frequency detections.
>
---
#### [new 093] RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models
- **分类: cs.CV**

- **简介: 该论文针对零样本开放词汇语义分割任务，解决现有方法参数与计算开销大、泛化能力不足的问题。提出RADSeg框架，基于聚类式视觉基础模型RADIO，通过自相关递归注意力等技术，实现高精度、低延迟、低参数量的分割，显著优于此前复杂模型组合。**

- **链接: [https://arxiv.org/pdf/2511.19704v1](https://arxiv.org/pdf/2511.19704v1)**

> **作者:** Omar Alama; Darshil Jariwala; Avigyan Bhattacharya; Seungchan Kim; Wenshan Wang; Sebastian Scherer
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) underpins many vision and robotics tasks that require generalizable semantic understanding. Existing approaches either rely on limited segmentation training data, which hinders generalization, or apply zero-shot heuristics to vision-language models (e.g CLIP), while the most competitive approaches combine multiple models to improve performance at the cost of high computational and memory demands. In this work, we leverage an overlooked agglomerative vision foundation model, RADIO, to improve zero-shot OVSS along three key axes simultaneously: mIoU, latency, and parameter efficiency. We present the first comprehensive study of RADIO for zero-shot OVSS and enhance its performance through self-correlating recursive attention, self-correlating global aggregation, and computationally efficient mask refinement. Our approach, RADSeg, achieves 6-30% mIoU improvement in the base ViT class while being 3.95x faster and using 2.5x fewer parameters. Surprisingly, RADSeg-base (105M) outperforms previous combinations of huge vision models (850-1350M) in mIoU, achieving state-of-the-art accuracy with substantially lower computational and memory cost.
>
---
#### [new 094] Hybrid Convolution and Frequency State Space Network for Image Compression
- **分类: cs.CV**

- **简介: 该论文针对图像压缩中局部细节与长程结构建模的平衡问题，提出HCFSSNet混合架构。结合CNN捕捉高频细节与频率感知状态空间模型建模低频信息，通过VFSS块和FSTAM模块实现高效比特分配与侧信息建模，显著降低参数量并提升率失真性能。**

- **链接: [https://arxiv.org/pdf/2511.20151v1](https://arxiv.org/pdf/2511.20151v1)**

> **作者:** Haodong Pan; Hao Wei; Yusong Wang; Nanning Zheng; Caigui Jiang
>
> **备注:** 36 pages, 8 figures
>
> **摘要:** Learned image compression (LIC) has recently benefited from Transformer based and state space model (SSM) based architectures. Convolutional neural networks (CNNs) effectively capture local high frequency details, whereas Transformers and SSMs provide strong long range modeling capabilities but may cause structural information loss or ignore frequency characteristics that are crucial for compression. In this work we propose HCFSSNet, a Hybrid Convolution and Frequency State Space Network for LIC. HCFSSNet uses CNNs to extract local high frequency structures and introduces a Vision Frequency State Space (VFSS) block that models long range low frequency information. The VFSS block combines an Omni directional Neighborhood State Space (VONSS) module, which scans features horizontally, vertically and diagonally, with an Adaptive Frequency Modulation Module (AFMM) that applies content adaptive weighting of discrete cosine transform frequency components for more efficient bit allocation. To further reduce redundancy in the entropy model, we integrate AFMM with a Swin Transformer to form a Frequency Swin Transformer Attention Module (FSTAM) for frequency aware side information modeling. Experiments on the Kodak, Tecnick and CLIC Professional Validation datasets show that HCFSSNet achieves competitive rate distortion performance compared with recent SSM based codecs such as MambaIC, while using significantly fewer parameters. On Kodak, Tecnick and CLIC, HCFSSNet reduces BD rate over the VTM anchor by 18.06, 24.56 and 22.44 percent, respectively, providing an efficient and interpretable hybrid architecture for future learned image compression systems.
>
---
#### [new 095] Boosting Reasoning in Large Multimodal Models via Activation Replay
- **分类: cs.CV**

- **简介: 该论文针对后训练大视觉语言模型（LMMs）推理能力不足的问题，提出无需训练的Activation Replay方法。通过重放输入中低熵激活，提升模型在数学、视觉代理和视频理解等任务中的推理表现，有效缓解RLVR带来的推理覆盖狭窄问题。**

- **链接: [https://arxiv.org/pdf/2511.19972v1](https://arxiv.org/pdf/2511.19972v1)**

> **作者:** Yun Xing; Xiaobin Hu; Qingdong He; Jiangning Zhang; Shuicheng Yan; Shijian Lu; Yu-Gang Jiang
>
> **备注:** 11 figures, 10 tables
>
> **摘要:** Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective approach to incentivizing reasoning capability in Large Multimodal Models (LMMs), while the underlying mechanisms behind this post-training paradigm are poorly understood. We begin by exploring how input activations are affected by RLVR through the perspective of logit lens. Our systematic investigations across multiple post-trained LMMs suggest that RLVR shifts low-entropy activations unexpectedly, while high-entropy ones are less affected. We further demonstrate that such phenomena are associated with LMM reasoning by controlled experiments, suggesting a potentially beneficial role of modulating low-entropy activations. To this end, we propose Activation Replay, a novel simple yet effective training-free approach that boosts multimodal reasoning of post-trained LMMs without requiring expensive policy optimization. Our design involves manipulation of visual tokens at test time, replaying low-entropy activations from the input context of base LMMs to regulating the RLVR counterparts. Activation Replay triggers better reasoning across diverse scenarios, including mathematics, o3-like visual agents, and video reasoning. We further show that Activation Replay boosts Pass@K and mitigates narrower reasoning coverage of RLVR. Our design is compared against alternative choices, such as replaying high-entropy activations instead of low-entropy ones, or direct cross-model intervention instead of manipulating input tokens, demonstrating the superiority of our implementation. Codes will be made publicly available.
>
---
#### [new 096] ACIT: Attention-Guided Cross-Modal Interaction Transformer for Pedestrian Crossing Intention Prediction
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中行人过街意图预测任务，解决多模态数据融合难题。提出ACIT模型，通过三对视觉与运动模态的交叉注意力机制，实现跨模态特征互补，结合Transformer捕捉时序依赖，显著提升预测精度。**

- **链接: [https://arxiv.org/pdf/2511.20020v1](https://arxiv.org/pdf/2511.20020v1)**

> **作者:** Yuanzhe Li; Steffen Müller
>
> **摘要:** Predicting pedestrian crossing intention is crucial for autonomous vehicles to prevent pedestrian-related collisions. However, effectively extracting and integrating complementary cues from different types of data remains one of the major challenges. This paper proposes an attention-guided cross-modal interaction Transformer (ACIT) for pedestrian crossing intention prediction. ACIT leverages six visual and motion modalities, which are grouped into three interaction pairs: (1) Global semantic map and global optical flow, (2) Local RGB image and local optical flow, and (3) Ego-vehicle speed and pedestrian's bounding box. Within each visual interaction pair, a dual-path attention mechanism enhances salient regions within the primary modality through intra-modal self-attention and facilitates deep interactions with the auxiliary modality (i.e., optical flow) via optical flow-guided attention. Within the motion interaction pair, cross-modal attention is employed to model the cross-modal dynamics, enabling the effective extraction of complementary motion features. Beyond pairwise interactions, a multi-modal feature fusion module further facilitates cross-modal interactions at each time step. Furthermore, a Transformer-based temporal feature aggregation module is introduced to capture sequential dependencies. Experimental results demonstrate that ACIT outperforms state-of-the-art methods, achieving accuracy rates of 70% and 89% on the JAADbeh and JAADall datasets, respectively. Extensive ablation studies are further conducted to investigate the contribution of different modules of ACIT.
>
---
#### [new 097] Exploring State-of-the-art models for Early Detection of Forest Fires
- **分类: cs.CV**

- **简介: 该论文针对森林火灾早期检测任务，解决现有方法因数据集不足导致漏检的问题。提出一个基于游戏模拟器生成的烟雾与火苗图像数据集，结合公开数据，评估YOLOv7与检测变压器模型在图像分类与定位上的性能，以提升早期预警能力。**

- **链接: [https://arxiv.org/pdf/2511.20096v1](https://arxiv.org/pdf/2511.20096v1)**

> **作者:** Sharjeel Ahmed; Daim Armaghan; Fatima Naweed; Umair Yousaf; Ahmad Zubair; Murtaza Taj
>
> **摘要:** There have been many recent developments in the use of Deep Learning Neural Networks for fire detection. In this paper, we explore an early warning system for detection of forest fires. Due to the lack of sizeable datasets and models tuned for this task, existing methods suffer from missed detection. In this work, we first propose a dataset for early identification of forest fires through visual analysis. Unlike existing image corpuses that contain images of wide-spread fire, our dataset consists of multiple instances of smoke plumes and fire that indicates the initiation of fire. We obtained this dataset synthetically by utilising game simulators such as Red Dead Redemption 2. We also combined our dataset with already published images to obtain a more comprehensive set. Finally, we compared image classification and localisation methods on the proposed dataset. More specifically we used YOLOv7 (You Only Look Once) and different models of detection transformer.
>
---
#### [new 098] VGGT4D: Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出VGGT4D，解决动态4D场景重建中动态物体与静态背景难分离的问题。基于3D基础模型VGGT，挖掘其全局注意力层中的动态线索，通过梯度相似性与投影梯度优化，生成精准掩码，实现无需训练的单次推理，显著提升动态分割、位姿估计与稠密重建性能。**

- **链接: [https://arxiv.org/pdf/2511.19971v1](https://arxiv.org/pdf/2511.19971v1)**

> **作者:** Yu Hu; Chong Cheng; Sicheng Yu; Xiaoyang Guo; Hao Wang
>
> **摘要:** Reconstructing dynamic 4D scenes is challenging, as it requires robust disentanglement of dynamic objects from the static background. While 3D foundation models like VGGT provide accurate 3D geometry, their performance drops markedly when moving objects dominate. Existing 4D approaches often rely on external priors, heavy post-optimization, or require fine-tuning on 4D datasets. In this paper, we propose VGGT4D, a training-free framework that extends the 3D foundation model VGGT for robust 4D scene reconstruction. Our approach is motivated by the key finding that VGGT's global attention layers already implicitly encode rich, layer-wise dynamic cues. To obtain masks that decouple static and dynamic elements, we mine and amplify global dynamic cues via gram similarity and aggregate them across a temporal window. To further sharpen mask boundaries, we introduce a refinement strategy driven by projection gradient. We then integrate these precise masks into VGGT's early-stage inference, effectively mitigating motion interference in both pose estimation and geometric reconstruction. Across six datasets, our method achieves superior performance in dynamic object segmentation, camera pose estimation, and dense reconstruction. It also supports single-pass inference on sequences longer than 500 frames.
>
---
#### [new 099] Clair Obscur: an Illumination-Aware Method for Real-World Image Vectorization
- **分类: cs.CV**

- **简介: 该论文针对真实世界图像矢量化任务，解决现有方法在复杂图像上易产生碎片化形状、语义不连贯的问题。提出COVec框架，首次在矢量域引入光照感知的内在图像分解，分离出反射率、阴影与光照层，并通过语义引导初始化和两阶段优化，实现更高保真度与可编辑性。**

- **链接: [https://arxiv.org/pdf/2511.20034v1](https://arxiv.org/pdf/2511.20034v1)**

> **作者:** Xingyue Lin; Shuai Peng; Xiangyu Xie; Jianhua Zhu; Yuxuan Zhou; Liangcai Gao
>
> **摘要:** Image vectorization aims to convert raster images into editable, scalable vector representations while preserving visual fidelity. Existing vectorization methods struggle to represent complex real-world images, often producing fragmented shapes at the cost of semantic conciseness. In this paper, we propose COVec, an illumination-aware vectorization framework inspired by the Clair-Obscur principle of light-shade contrast. COVec is the first to introduce intrinsic image decomposition in the vector domain, separating an image into albedo, shade, and light layers in a unified vector representation. A semantic-guided initialization and two-stage optimization refine these layers with differentiable rendering. Experiments on various datasets demonstrate that COVec achieves higher visual fidelity and significantly improved editability compared to existing methods.
>
---
#### [new 100] History-Augmented Contrastive Meta-Learning for Unsupervised Blind Super-Resolution of Planetary Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文针对行星遥感图像的无监督盲超分辨率任务，解决因未知退化导致缺乏真值图像的问题。提出HACBSR框架，通过对比核采样与历史模型增强的对比学习，实现无需真值和先验核的高质量重建，显著提升在多倍上采样下的性能。**

- **链接: [https://arxiv.org/pdf/2511.20045v1](https://arxiv.org/pdf/2511.20045v1)**

> **作者:** Huijia Zhao; Jie Lu; Yunqing Jiang; Xiao-Ping Lu; Kaichang Di
>
> **备注:** 13pages
>
> **摘要:** Planetary remote sensing images are affected by diverse and unknown degradations caused by imaging environments and hardware constraints. These factors limit image quality and hinder supervised blind super-resolution due to the lack of ground-truth images. This work presents History-Augmented Contrastive Blind Super-Resolution (HACBSR), an unsupervised framework for blind super-resolution that operates without ground-truth images and external kernel priors. HACBSR comprises two components: (1) a contrastive kernel sampling mechanism with kernel similarity control to mitigate distribution bias from Gaussian sampling, and (2) a history-augmented contrastive learning that uses historical models to generate negative samples to enable less greedy optimization and to induce strong convexity without ground-truth. A convergence analysis of the history-augmented contrastive learning is given in the Appendix. To support evaluation in planetary applications, we introduce Ceres-50, a dataset with diverse geological features simulated degradation patterns. Experiments show that HACBSR achieves competitive performance compared with state-of-the-art unsupervised methods across multiple upscaling factors. The code is available at https://github.com/2333repeat/HACBSR, and the dataset is available at https://github.com/2333repeat/Ceres-50.
>
---
#### [new 101] CountXplain: Interpretable Cell Counting with Prototype-Based Density Map Estimation
- **分类: cs.CV**

- **简介: 该论文针对生物医学图像中细胞计数的可解释性问题，提出基于原型的密度图估计方法。通过引入原型层学习细胞与背景的代表性视觉模式，生成可解释的计数结果，在保持计数精度的同时提升模型透明度，增强了临床可信度。**

- **链接: [https://arxiv.org/pdf/2511.19686v1](https://arxiv.org/pdf/2511.19686v1)**

> **作者:** Abdurahman Ali Mohammed; Wallapak Tavanapong; Catherine Fonder; Donald S. Sakaguchi
>
> **备注:** Medical Imaging with Deep Learning 2025
>
> **摘要:** Cell counting in biomedical imaging is pivotal for various clinical applications, yet the interpretability of deep learning models in this domain remains a significant challenge. We propose a novel prototype-based method for interpretable cell counting via density map estimation. Our approach integrates a prototype layer into the density estimation network, enabling the model to learn representative visual patterns for both cells and background artifacts. The learned prototypes were evaluated through a survey of biologists, who confirmed the relevance of the visual patterns identified, further validating the interpretability of the model. By generating interpretations that highlight regions in the input image most similar to each prototype, our method offers a clear understanding of how the model identifies and counts cells. Extensive experiments on two public datasets demonstrate that our method achieves interpretability without compromising counting effectiveness. This work provides researchers and clinicians with a transparent and reliable tool for cell counting, potentially increasing trust and accelerating the adoption of deep learning in critical biomedical applications. Code is available at https://github.com/NRT-D4/CountXplain.
>
---
#### [new 102] Vision-Language Models for Automated 3D PET/CT Report Generation
- **分类: cs.CV**

- **简介: 该论文针对肿瘤学中PET/CT报告生成自动化任务，解决专家短缺与报告差异性问题。提出PETRG-3D框架，融合3D双模态编码与风格自适应提示，构建多中心淋巴瘤数据集与公开基准，引入临床导向评估指标，显著提升报告生成的准确性与临床实用性。**

- **链接: [https://arxiv.org/pdf/2511.20145v1](https://arxiv.org/pdf/2511.20145v1)**

> **作者:** Wenpei Jiao; Kun Shang; Hui Li; Ke Yan; Jiajin Zhang; Guangjie Yang; Lijuan Guo; Yan Wan; Xing Yang; Dakai Jin; Zhaoheng Xie
>
> **摘要:** Positron emission tomography/computed tomography (PET/CT) is essential in oncology, yet the rapid expansion of scanners has outpaced the availability of trained specialists, making automated PET/CT report generation (PETRG) increasingly important for reducing clinical workload. Compared with structural imaging (e.g., X-ray, CT, and MRI), functional PET poses distinct challenges: metabolic patterns vary with tracer physiology, and whole-body 3D contextual information is required rather than local-region interpretation. To advance PETRG, we propose PETRG-3D, an end-to-end 3D dual-branch framework that separately encodes PET and CT volumes and incorporates style-adaptive prompts to mitigate inter-hospital variability in reporting practices. We construct PETRG-Lym, a multi-center lymphoma dataset collected from four hospitals (824 reports w/ 245,509 paired PET/CT slices), and construct AutoPET-RG-Lym, a publicly accessible PETRG benchmark derived from open imaging data but equipped with new expert-written, clinically validated reports (135 cases). To assess clinical utility, we introduce PETRG-Score, a lymphoma-specific evaluation protocol that jointly measures metabolic and structural findings across curated anatomical regions. Experiments show that PETRG-3D substantially outperforms existing methods on both natural language metrics (e.g., +31.49\% ROUGE-L) and clinical efficacy metrics (e.g., +8.18\% PET-All), highlighting the benefits of volumetric dual-modality modeling and style-aware prompting. Overall, this work establishes a foundation for future PET/CT-specific models emphasizing disease-aware reasoning and clinically reliable evaluation. Codes, models, and AutoPET-RG-Lym will be released.
>
---
#### [new 103] PRADA: Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images
- **分类: cs.CV**

- **简介: 该论文针对自回归（AR）图像生成的检测与溯源问题，提出PRADA方法。通过分析图像生成时条件概率与无条件概率的比值，利用其独特特征实现对AR生成图像的可靠检测与源模型归属，有效区分真实图像与多种生成模型产出的图像。**

- **链接: [https://arxiv.org/pdf/2511.20068v1](https://arxiv.org/pdf/2511.20068v1)**

> **作者:** Simon Damm; Jonas Ricker; Henning Petzka; Asja Fischer
>
> **摘要:** Autoregressive (AR) image generation has recently emerged as a powerful paradigm for image synthesis. Leveraging the generation principle of large language models, they allow for efficiently generating deceptively real-looking images, further increasing the need for reliable detection methods. However, to date there is a lack of work specifically targeting the detection of images generated by AR image generators. In this work, we present PRADA (Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images), a simple and interpretable approach that can reliably detect AR-generated images and attribute them to their respective source model. The key idea is to inspect the ratio of a model's conditional and unconditional probability for the autoregressive token sequence representing a given image. Whenever an image is generated by a particular model, its probability ratio shows unique characteristics which are not present for images generated by other models or real images. We exploit these characteristics for threshold-based attribution and detection by calibrating a simple, model-specific score function. Our experimental evaluation shows that PRADA is highly effective against eight class-to-image and four text-to-image models.
>
---
#### [new 104] Studying Maps at Scale: A Digital Investigation of Cartography and the Evolution of Figuration
- **分类: cs.CV; cs.CL; cs.DL**

- **简介: 该论文属大规模地图文化遗产研究任务，旨在解决传统地图研究忽视文化语义与历史演变的问题。通过整合超百万地图数据，运用语义分割与目标检测技术，分析地图的地理结构、符号系统及政治文化关联，揭示地图作为象征性文化产物的演化规律与传播机制。**

- **链接: [https://arxiv.org/pdf/2511.19538v1](https://arxiv.org/pdf/2511.19538v1)**

> **作者:** Remi Petitpierre
>
> **备注:** PhD thesis, EPFL. 396 pages, 156 figures
>
> **摘要:** This thesis presents methods and datasets to investigate cartographic heritage on a large scale and from a cultural perspective. Heritage institutions worldwide have digitized more than one million maps, and automated techniques now enable large-scale recognition and extraction of map content. Yet these methods have engaged little with the history of cartography, or the view that maps are semantic-symbolic systems, and cultural objects reflecting political and epistemic expectations. This work leverages a diverse corpus of 771,561 map records and 99,715 digitized images aggregated from 38 digital catalogs. After normalization, the dataset includes 236,925 contributors and spans six centuries, from 1492 to 1948. These data make it possible to chart geographic structures and the global chronology of map publication. The spatial focus of cartography is analyzed in relation to political dynamics, evidencing links between Atlantic maritime charting, the triangular trade, and colonial expansion. Further results document the progression of national, domestic focus and the impact of military conflicts on publication volumes. The research introduces semantic segmentation techniques and object detection models for the generic recognition of land classes and cartographic signs, trained on annotated data and synthetic images. The analysis of land classes shows that maps are designed images whose framing and composition emphasize features through centering and semantic symmetries. The study of cartographic figuration encodes 63 M signs and 25 M fragments into a latent visual space, revealing figurative shifts such as the replacement of relief hachures by terrain contours and showing that signs tend to form locally consistent systems. Analyses of collaboration and diffusion highlight the role of legitimacy, larger actors, and major cities in the spread of figurative norms and semiotic cultures.
>
---
#### [new 105] SelfMOTR: Revisiting MOTR with Self-Generating Detection Priors
- **分类: cs.CV**

- **简介: 该论文针对端到端多目标跟踪中检测性能差与检测关联冲突的问题，提出SelfMOTR。通过挖掘MOTR模型隐含的强检测能力，自生成检测先验，提升跟踪性能，在DanceTrack上达到先进水平。**

- **链接: [https://arxiv.org/pdf/2511.20279v1](https://arxiv.org/pdf/2511.20279v1)**

> **作者:** Fabian Gülhan; Emil Mededovic; Yuli Wu; Johannes Stegmaier
>
> **备注:** 11 pages, 5 figures, 10 tables
>
> **摘要:** Despite progress toward end-to-end tracking with transformer architectures, poor detection performance and the conflict between detection and association in a joint architecture remain critical concerns. Recent approaches aim to mitigate these issues by (i) employing advanced denoising or label assignment strategies, or (ii) incorporating detection priors from external object detectors via distillation or anchor proposal techniques. Inspired by the success of integrating detection priors and by the key insight that MOTR-like models are secretly strong detection models, we introduce SelfMOTR, a novel tracking transformer that relies on self-generated detection priors. Through extensive analysis and ablation studies, we uncover and demonstrate the hidden detection capabilities of MOTR-like models, and present a practical set of tools for leveraging them effectively. On DanceTrack, SelfMOTR achieves strong performance, competing with recent state-of-the-art end-to-end tracking methods.
>
---
#### [new 106] Personalized Reward Modeling for Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像生成中用户偏好难以量化评估的问题，提出PIGReward模型。通过自举策略构建用户个性化评价维度，结合思维链推理实现动态评分与可解释反馈，支持用户特定提示优化。研究还构建了个性化基准PIGBench，验证了方法在准确性与可解释性上的优越性。**

- **链接: [https://arxiv.org/pdf/2511.19458v1](https://arxiv.org/pdf/2511.19458v1)**

> **作者:** Jeongeun Lee; Ryang Heo; Dongha Lee
>
> **摘要:** Recent text-to-image (T2I) models generate semantically coherent images from textual prompts, yet evaluating how well they align with individual user preferences remains an open challenge. Conventional evaluation methods, general reward functions or similarity-based metrics, fail to capture the diversity and complexity of personal visual tastes. In this work, we present PIGReward, a personalized reward model that dynamically generates user-conditioned evaluation dimensions and assesses images through CoT reasoning. To address the scarcity of user data, PIGReward adopt a self-bootstrapping strategy that reasons over limited reference data to construct rich user contexts, enabling personalization without user-specific training. Beyond evaluation, PIGReward provides personalized feedback that drives user-specific prompt optimization, improving alignment between generated images and individual intent. We further introduce PIGBench, a per-user preference benchmark capturing diverse visual interpretations of shared prompts. Extensive experiments demonstrate that PIGReward surpasses existing methods in both accuracy and interpretability, establishing a scalable and reasoning-based foundation for personalized T2I evaluation and optimization. Taken together, our findings highlight PIGReward as a robust steptoward individually aligned T2I generation.
>
---
#### [new 107] CREward: A Type-Specific Creativity Reward Model
- **分类: cs.CV**

- **简介: 该论文提出CREward，首个针对图像创作中几何、材质、纹理三类特性的类型化创造力奖励模型。旨在解决传统创造力评估忽视类型差异的问题。通过人类标注与LVLM对齐分析，构建可应用于评估、解释与生成的奖励模型，推动创造性内容的精准生成与设计启发。**

- **链接: [https://arxiv.org/pdf/2511.19995v1](https://arxiv.org/pdf/2511.19995v1)**

> **作者:** Jiyeon Han; Ali Mahdavi-Amiri; Hao Zhang; Haedong Jeong
>
> **摘要:** Creativity is a complex phenomenon. When it comes to representing and assessing creativity, treating it as a single undifferentiated quantity would appear naive and underwhelming. In this work, we learn the \emph{first type-specific creativity reward model}, coined CREward, which spans three creativity ``axes," geometry, material, and texture, to allow us to view creativity through the lens of the image formation pipeline. To build our reward model, we first conduct a human benchmark evaluation to capture human perception of creativity for each type across various creative images. We then analyze the correlation between human judgments and predictions by large vision-language models (LVLMs), confirming that LVLMs exhibit strong alignment with human perception. Building on this observation, we collect LVLM-generated labels to train our CREward model that is applicable to both evaluation and generation of creative images. We explore three applications of CREward: creativity assessment, explainable creativity, and creative sample acquisition for both human design inspiration and guiding creative generation through low-rank adaptation.
>
---
#### [new 108] Rectified SpaAttn: Revisiting Attention Sparsity for Efficient Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视频生成中扩散Transformer的注意力计算复杂度高问题，提出Rectified SpaAttn方法。通过修正注意力分配偏差，提升稀疏注意力与全注意力的对齐度，在保持生成质量的同时实现显著加速。**

- **链接: [https://arxiv.org/pdf/2511.19835v1](https://arxiv.org/pdf/2511.19835v1)**

> **作者:** Xuewen Liu; Zhikai Li; Jing Zhang; Mengjuan Chen; Qingyi Gu
>
> **备注:** Code at https://github.com/BienLuky/Rectified-SpaAttn
>
> **摘要:** Diffusion Transformers dominate video generation, but the quadratic complexity of attention computation introduces substantial latency. Attention sparsity reduces computational costs by focusing on critical tokens while ignoring non-critical tokens. However, existing methods suffer from severe performance degradation. In this paper, we revisit attention sparsity and reveal that existing methods induce systematic biases in attention allocation: (1) excessive focus on critical tokens amplifies their attention weights; (2) complete neglect of non-critical tokens causes the loss of relevant attention weights. To address these issues, we propose Rectified SpaAttn, which rectifies attention allocation with implicit full attention reference, thereby enhancing the alignment between sparse and full attention maps. Specifically: (1) for critical tokens, we show that their bias is proportional to the sparse attention weights, with the ratio governed by the amplified weights. Accordingly, we propose Isolated-Pooling Attention Reallocation, which calculates accurate rectification factors by reallocating multimodal pooled weights. (2) for non-critical tokens, recovering attention weights from the pooled query-key yields attention gains but also introduces pooling errors. Therefore, we propose Gain-Aware Pooling Rectification, which ensures that the rectified gain consistently surpasses the induced error. Moreover, we customize and integrate the Rectified SpaAttn kernel using Triton, achieving up to 3.33 and 2.08 times speedups on HunyuanVideo and Wan 2.1, respectively, while maintaining high generation quality. We release Rectified SpaAttn as open-source at https://github.com/BienLuky/Rectified-SpaAttn .
>
---
#### [new 109] STARFlow-V: End-to-End Video Generative Modeling with Normalizing Flow
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出STARFlow-V，一种基于归一化流的端到端视频生成模型，解决视频生成中时序误差累积与采样效率低的问题。通过全局-局部潜空间架构与流-分数匹配机制，实现高效、一致的自回归视频生成，并支持多模态生成任务。**

- **链接: [https://arxiv.org/pdf/2511.20462v1](https://arxiv.org/pdf/2511.20462v1)**

> **作者:** Jiatao Gu; Ying Shen; Tianrong Chen; Laurent Dinh; Yuyang Wang; Miguel Angel Bautista; David Berthelot; Josh Susskind; Shuangfei Zhai
>
> **备注:** 21 pages
>
> **摘要:** Normalizing flows (NFs) are end-to-end likelihood-based generative models for continuous data, and have recently regained attention with encouraging progress on image generation. Yet in the video generation domain, where spatiotemporal complexity and computational cost are substantially higher, state-of-the-art systems almost exclusively rely on diffusion-based models. In this work, we revisit this design space by presenting STARFlow-V, a normalizing flow-based video generator with substantial benefits such as end-to-end learning, robust causal prediction, and native likelihood estimation. Building upon the recently proposed STARFlow, STARFlow-V operates in the spatiotemporal latent space with a global-local architecture which restricts causal dependencies to a global latent space while preserving rich local within-frame interactions. This eases error accumulation over time, a common pitfall of standard autoregressive diffusion model generation. Additionally, we propose flow-score matching, which equips the model with a light-weight causal denoiser to improve the video generation consistency in an autoregressive fashion. To improve the sampling efficiency, STARFlow-V employs a video-aware Jacobi iteration scheme that recasts inner updates as parallelizable iterations without breaking causality. Thanks to the invertible structure, the same model can natively support text-to-video, image-to-video as well as video-to-video generation tasks. Empirically, STARFlow-V achieves strong visual fidelity and temporal consistency with practical sampling throughput relative to diffusion-based baselines. These results present the first evidence, to our knowledge, that NFs are capable of high-quality autoregressive video generation, establishing them as a promising research direction for building world models. Code and generated samples are available at https://github.com/apple/ml-starflow.
>
---
#### [new 110] Back to the Feature: Explaining Video Classifiers with Video Counterfactual Explanations
- **分类: cs.CV**

- **简介: 该论文针对视频分类器的可解释性问题，提出Back To The Feature（BTTF）框架，生成物理合理、时序连贯的视频反事实解释。通过双阶段优化与渐进式去噪策略，实现对视频分类器决策机制的忠实、对比性解释，有效提升视频反事实生成的质量与效率。**

- **链接: [https://arxiv.org/pdf/2511.20295v1](https://arxiv.org/pdf/2511.20295v1)**

> **作者:** Chao Wang; Chengan Che; Xinyue Chen; Sophia Tsoka; Luis C. Garcia-Peraza-Herrera
>
> **摘要:** Counterfactual explanations (CFEs) are minimal and semantically meaningful modifications of the input of a model that alter the model predictions. They highlight the decisive features the model relies on, providing contrastive interpretations for classifiers. State-of-the-art visual counterfactual explanation methods are designed to explain image classifiers. The generation of CFEs for video classifiers remains largely underexplored. For the counterfactual videos to be useful, they have to be physically plausible, temporally coherent, and exhibit smooth motion trajectories. Existing CFE image-based methods, designed to explain image classifiers, lack the capacity to generate temporally coherent, smooth and physically plausible video CFEs. To address this, we propose Back To The Feature (BTTF), an optimization framework that generates video CFEs. Our method introduces two novel features, 1) an optimization scheme to retrieve the initial latent noise conditioned by the first frame of the input video, 2) a two-stage optimization strategy to enable the search for counterfactual videos in the vicinity of the input video. Both optimization processes are guided solely by the target classifier, ensuring the explanation is faithful. To accelerate convergence, we also introduce a progressive optimization strategy that incrementally increases the number of denoising steps. Extensive experiments on video datasets such as Shape-Moving (motion classification), MEAD (emotion classification), and NTU RGB+D (action classification) show that our BTTF effectively generates valid, visually similar and realistic counterfactual videos that provide concrete insights into the classifier's decision-making mechanism.
>
---
#### [new 111] PhysChoreo: Physics-Controllable Video Generation with Part-Aware Semantic Grounding
- **分类: cs.CV**

- **简介: 该论文提出PhysChoreo，面向视频生成任务，解决现有模型物理可控性与真实性不足的问题。通过部件感知的物理属性重建与可编辑的物理模拟，实现从单图生成具丰富动态行为和物理真实感的视频，显著提升可控性与逼真度。**

- **链接: [https://arxiv.org/pdf/2511.20562v1](https://arxiv.org/pdf/2511.20562v1)**

> **作者:** Haoze Zhang; Tianyu Huang; Zichen Wan; Xiaowei Jin; Hongzhi Zhang; Hui Li; Wangmeng Zuo
>
> **摘要:** While recent video generation models have achieved significant visual fidelity, they often suffer from the lack of explicit physical controllability and plausibility. To address this, some recent studies attempted to guide the video generation with physics-based rendering. However, these methods face inherent challenges in accurately modeling complex physical properties and effectively control ling the resulting physical behavior over extended temporal sequences. In this work, we introduce PhysChoreo, a novel framework that can generate videos with diverse controllability and physical realism from a single image. Our method consists of two stages: first, it estimates the static initial physical properties of all objects in the image through part-aware physical property reconstruction. Then, through temporally instructed and physically editable simulation, it synthesizes high-quality videos with rich dynamic behaviors and physical realism. Experimental results show that PhysChoreo can generate videos with rich behaviors and physical realism, outperforming state-of-the-art methods on multiple evaluation metrics.
>
---
#### [new 112] FLaTEC: Frequency-Disentangled Latent Triplanes for Efficient Compression of LiDAR Point Clouds
- **分类: cs.CV**

- **简介: 该论文针对LiDAR点云压缩中低频与高频成分难以平衡的问题，提出FLaTEC模型。通过频率解耦的隐式三平面表示，分离并高效压缩低频结构与高频细节，结合注意力机制恢复3D相关性，显著提升压缩效率与重建质量。**

- **链接: [https://arxiv.org/pdf/2511.20065v1](https://arxiv.org/pdf/2511.20065v1)**

> **作者:** Xiaoge Zhang; Zijie Wu; Mingtao Feng; Zichen Geng; Mehwish Nasim; Saeed Anwar; Ajmal Mian
>
> **摘要:** Point cloud compression methods jointly optimize bitrates and reconstruction distortion. However, balancing compression ratio and reconstruction quality is difficult because low-frequency and high-frequency components contribute differently at the same resolution. To address this, we propose FLaTEC, a frequency-aware compression model that enables the compression of a full scan with high compression ratios. Our approach introduces a frequency-aware mechanism that decouples low-frequency structures and high-frequency textures, while hybridizing latent triplanes as a compact proxy for point cloud. Specifically, we convert voxelized embeddings into triplane representations to reduce sparsity, computational cost, and storage requirements. We then devise a frequency-disentangling technique that extracts compact low-frequency content while collecting high-frequency details across scales. The decoupled low-frequency and high-frequency components are stored in binary format. During decoding, full-spectrum signals are progressively recovered via a modulation block. Additionally, to compensate for the loss of 3D correlation, we introduce an efficient frequency-based attention mechanism that fosters local connectivity and outputs arbitrary resolution points. Our method achieves state-of-the-art rate-distortion performance and outperforms the standard codecs by 78\% and 94\% in BD-rate on both SemanticKITTI and Ford datasets.
>
---
#### [new 113] MotionV2V: Editing Motion in a Video
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **简介: 该论文提出MotionV2V，针对视频编辑中精准控制运动的难题，通过编辑输入视频的稀疏轨迹实现运动修改。提出生成“运动反事实”数据集，微调运动条件视频扩散模型，支持任意时间点起始的自然运动编辑，在用户研究中优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20640v1](https://arxiv.org/pdf/2511.20640v1)**

> **作者:** Ryan Burgert; Charles Herrmann; Forrester Cole; Michael S Ryoo; Neal Wadhwa; Andrey Voynov; Nataniel Ruiz
>
> **摘要:** While generative video models have achieved remarkable fidelity and consistency, applying these capabilities to video editing remains a complex challenge. Recent research has explored motion controllability as a means to enhance text-to-video generation or image animation; however, we identify precise motion control as a promising yet under-explored paradigm for editing existing videos. In this work, we propose modifying video motion by directly editing sparse trajectories extracted from the input. We term the deviation between input and output trajectories a "motion edit" and demonstrate that this representation, when coupled with a generative backbone, enables powerful video editing capabilities. To achieve this, we introduce a pipeline for generating "motion counterfactuals", video pairs that share identical content but distinct motion, and we fine-tune a motion-conditioned video diffusion architecture on this dataset. Our approach allows for edits that start at any timestamp and propagate naturally. In a four-way head-to-head user study, our model achieves over 65 percent preference against prior work. Please see our project page: https://ryanndagreat.github.io/MotionV2V
>
---
#### [new 114] CodeV: Code with Images for Faithful Visual Reasoning via Tool-Aware Policy Optimization
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型在视觉推理中“工具使用不忠实”问题，提出CodeV框架。通过代码化工具调用与面向工具的策略优化（TAPO），以中间输出为监督信号，提升工具使用的准确性与可解释性。实验表明，CodeV在保持高准确率的同时显著提高忠实度，适用于多模态推理与数学任务。**

- **链接: [https://arxiv.org/pdf/2511.19661v1](https://arxiv.org/pdf/2511.19661v1)**

> **作者:** Xinhai Hou; Shaoyuan Xu; Manan Biyani; Mayan Li; Jia Liu; Todd C. Hollon; Bryan Wang
>
> **摘要:** Agentic vision-language models are increasingly trained to "think with images" by calling image operations. However, we show that high final-answer accuracy often hides unfaithful visual reasoning: models may invoke tools on irrelevant regions or ignore tool outputs entirely, yet still guess the correct answer. In this work, we first propose a faithfulness evaluation protocol that measures whether intermediate visual tool outputs (e.g., crops) actually contain the queried evidence. This reveals that recent visual agents achieve high final-answer accuracy but exhibit low rates of faithful tool-use on visual search benchmarks. We then introduce CodeV, a code-based visual agent trained with Tool-Aware Policy Optimization (TAPO). TAPO is a process-level RL framework that augments GRPO with dense rewards defined directly on visual tool inputs and outputs, rather than on chain-of-thought tokens, making supervision easier to verify and less susceptible to reward hacking. CodeV represents visual tools as executable Python code, and TAPO assigns step-wise rewards based solely on the question and tool output, encouraging both necessary and evidence-consistent tool use. In a two-stage SFT+RL pipeline, CodeV achieves competitive or superior accuracy while substantially increasing faithful tool-use rates on related visual search benchmarks. Beyond visual search, CodeV attains strong performance on a range of multimodal reasoning and math benchmarks, suggesting that explicitly supervising intermediate tool behavior is crucial for building trustworthy, agentic visual reasoning systems.
>
---
#### [new 115] Patch-Level Glioblastoma Subregion Classification with a Contrastive Learning-Based Encoder
- **分类: cs.CV**

- **简介: 该论文针对胶质母细胞瘤病理异质性带来的诊断困难，提出基于对比学习的ViT编码器进行病灶区域分类。通过微调预训练ViT模型，在BraTS-Path 2025挑战中实现高MCC与F1分数，有效提升了自动化病理分析性能，为后续研究提供可靠基线。**

- **链接: [https://arxiv.org/pdf/2511.20221v1](https://arxiv.org/pdf/2511.20221v1)**

> **作者:** Juexin Zhang; Qifeng Zhong; Ying Weng; Ke Chen
>
> **备注:** Accepted by the International Brain Tumor Segmentation (BraTS) challenge organized at MICCAI 2025 conference
>
> **摘要:** The significant molecular and pathological heterogeneity of glioblastoma, an aggressive brain tumor, complicates diagnosis and patient stratification. While traditional histopathological assessment remains the standard, deep learning offers a promising path toward objective and automated analysis of whole slide images. For the BraTS-Path 2025 Challenge, we developed a method that fine-tunes a pre-trained Vision Transformer (ViT) encoder with a dedicated classification head on the official training dataset. Our model's performance on the online validation set, evaluated via the Synapse platform, yielded a Matthews Correlation Coefficient (MCC) of 0.7064 and an F1-score of 0.7676. On the final test set, the model achieved an MCC of 0.6509 and an F1-score of 0.5330, which secured our team second place in the BraTS-Pathology 2025 Challenge. Our results establish a solid baseline for ViT-based histopathological analysis, and future efforts will focus on bridging the performance gap observed on the unseen validation data.
>
---
#### [new 116] Robust 3D Brain MRI Inpainting with Random Masking Augmentation
- **分类: cs.CV**

- **简介: 该论文针对脑肿瘤MRI图像中数据集偏差问题，提出一种基于U-Net的3D图像修复方法。通过随机掩码增强策略提升模型泛化能力，有效恢复健康组织。在BraTS-Inpainting 2025挑战中表现优异，获第一名，优于历届冠军方案。**

- **链接: [https://arxiv.org/pdf/2511.20202v1](https://arxiv.org/pdf/2511.20202v1)**

> **作者:** Juexin Zhang; Ying Weng; Ke Chen
>
> **备注:** Accepted by the International Brain Tumor Segmentation (BraTS) challenge organized at MICCAI 2025 conference
>
> **摘要:** The ASNR-MICCAI BraTS-Inpainting Challenge was established to mitigate dataset biases that limit deep learning models in the quantitative analysis of brain tumor MRI. This paper details our submission to the 2025 challenge, a novel deep learning framework for synthesizing healthy tissue in 3D scans. The core of our method is a U-Net architecture trained to inpaint synthetically corrupted regions, enhanced with a random masking augmentation strategy to improve generalization. Quantitative evaluation confirmed the efficacy of our approach, yielding an SSIM of 0.873$\pm$0.004, a PSNR of 24.996$\pm$4.694, and an MSE of 0.005$\pm$0.087 on the validation set. On the final online test set, our method achieved an SSIM of 0.919$\pm$0.088, a PSNR of 26.932$\pm$5.057, and an RMSE of 0.052$\pm$0.026. This performance secured first place in the BraTS-Inpainting 2025 challenge and surpassed the winning solutions from the 2023 and 2024 competitions on the official leaderboard.
>
---
#### [new 117] Rethinking Vision Transformer Depth via Structural Reparameterization
- **分类: cs.CV**

- **简介: 该论文针对视觉Transformer计算开销大的问题，提出基于结构重参数化的分支方法，在训练中构建并行分支，推理时精确合并为浅层模型。成功将ViT-Tiny从12层压缩至3层，保持精度并提升37%推理速度，挑战了深度堆叠的必要性，推动高效视觉Transformer设计。**

- **链接: [https://arxiv.org/pdf/2511.19718v1](https://arxiv.org/pdf/2511.19718v1)**

> **作者:** Chengwei Zhou; Vipin Chaudhary; Gourav Datta
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** The computational overhead of Vision Transformers in practice stems fundamentally from their deep architectures, yet existing acceleration strategies have primarily targeted algorithmic-level optimizations such as token pruning and attention speedup. This leaves an underexplored research question: can we reduce the number of stacked transformer layers while maintaining comparable representational capacity? To answer this, we propose a branch-based structural reparameterization technique that operates during the training phase. Our approach leverages parallel branches within transformer blocks that can be systematically consolidated into streamlined single-path models suitable for inference deployment. The consolidation mechanism works by gradually merging branches at the entry points of nonlinear components, enabling both feed-forward networks (FFN) and multi-head self-attention (MHSA) modules to undergo exact mathematical reparameterization without inducing approximation errors at test time. When applied to ViT-Tiny, the framework successfully reduces the original 12-layer architecture to 6, 4, or as few as 3 layers while maintaining classification accuracy on ImageNet-1K. The resulting compressed models achieve inference speedups of up to 37% on mobile CPU platforms. Our findings suggest that the conventional wisdom favoring extremely deep transformer stacks may be unnecessarily restrictive, and point toward new opportunities for constructing efficient vision transformers.
>
---
#### [new 118] Prune-Then-Plan: Step-Level Calibration for Stable Frontier Exploration in Embodied Question Answering
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对具身问答（EQA）中的不稳定探索问题，提出Prune-Then-Plan框架。通过霍尔姆-邦尼费罗尼启发的剪枝去除不可信前缘选择，再由覆盖导向规划器决策，实现步级校准。有效缓解了大视觉语言模型导致的前沿振荡，提升导航效率与答案质量，在多个数据集上显著优于基线。**

- **链接: [https://arxiv.org/pdf/2511.19768v1](https://arxiv.org/pdf/2511.19768v1)**

> **作者:** Noah Frahm; Prakrut Patel; Yue Zhang; Shoubin Yu; Mohit Bansal; Roni Sengupta
>
> **备注:** webpage: https://noahfrahm.github.io/Prune-Then-Plan-project-page/
>
> **摘要:** Large vision-language models (VLMs) have improved embodied question answering (EQA) agents by providing strong semantic priors for open-vocabulary reasoning. However, when used directly for step-level exploration, VLMs often exhibit frontier oscillations, unstable back-and-forth movements caused by overconfidence and miscalibration, leading to inefficient navigation and degraded answer quality. We propose Prune-Then-Plan, a simple and effective framework that stabilizes exploration through step-level calibration. Instead of trusting raw VLM scores, our method prunes implausible frontier choices using a Holm-Bonferroni inspired pruning procedure and then delegates final decisions to a coverage-based planner. This separation converts overconfident predictions into conservative, interpretable actions by relying on human-level judgments to calibrate the step-level behavior of VLMs. Integrated into the 3D-Mem EQA framework, our approach achieves relative improvements of up to 49% and 33% in visually grounded SPL and LLM-Match metrics respectively over baselines. Overall, our method achieves better scene coverage under equal exploration budgets on both OpenEQA and EXPRESS-Bench datasets.
>
---
#### [new 119] VKnowU: Evaluating Visual Knowledge Understanding in Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文聚焦多模态大模型的视觉理解能力，针对其缺乏对物理与社会常识的深层理解问题，提出VKnowU基准评估视觉知识。通过构建新数据集VKnowQA和VideoKnow+模型，引入结构化推理与知识奖励机制，显著提升模型在复杂场景下的理解性能，推动更通用的多模态理解发展。**

- **链接: [https://arxiv.org/pdf/2511.20272v1](https://arxiv.org/pdf/2511.20272v1)**

> **作者:** Tianxiang Jiang; Sheng Xia; Yicheng Xu; Linquan Wu; Xiangyu Zeng; Limin Wang; Yu Qiao; Yi Wang
>
> **备注:** Data & Code: this https URL
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have become adept at recognizing objects, they often lack the intuitive, human-like understanding of the world's underlying physical and social principles. This high-level vision-grounded semantics, which we term visual knowledge, forms a bridge between perception and reasoning, yet remains an underexplored area in current MLLMs. To systematically evaluate this capability, we present VKnowU, a comprehensive benchmark featuring 1,680 questions in 1,249 videos, covering 8 core types of visual knowledge spanning both world-centric (e.g., intuitive physics) and human-centric (e.g., subjective intentions). Evaluation of 23 SOTA MLLMs reveals that leading models still fall short of human performance, with particularly notable gaps in the world-centric. To bridge this gap, we introduce a new dataset, VKnowQA, and VideoKnow+, a baseline model that explicitly incorporates visual knowledge into MLLMs. VideoKnow+ follows a structured See-Think-Answer paradigm and adopts reinforcement learning with visual knowledge reward, achieving a +3.7% improvement on VKnowU and consistent gains on MVBench, Video-MME, and MMVU. Our work highlights visual knowledge as a missing cornerstone for developing more generalizable MLLMs that can not only see but also truly understand our physical and social worlds.
>
---
#### [new 120] Flash-DMD: Towards High-Fidelity Few-Step Image Generation with Efficient Distillation and Joint Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对扩散模型生成速度慢、蒸馏训练成本高及强化学习微调不稳定的难题，提出Flash-DMD框架。通过高效时间步蒸馏与联合强化学习训练，显著降低训练成本，提升生成质量与稳定性，实现少步采样下的高保真图像生成。**

- **链接: [https://arxiv.org/pdf/2511.20549v1](https://arxiv.org/pdf/2511.20549v1)**

> **作者:** Guanjie Chen; Shirui Huang; Kai Liu; Jianchen Zhu; Xiaoye Qu; Peng Chen; Yu Cheng; Yifu Sun
>
> **摘要:** Diffusion Models have emerged as a leading class of generative models, yet their iterative sampling process remains computationally expensive. Timestep distillation is a promising technique to accelerate generation, but it often requires extensive training and leads to image quality degradation. Furthermore, fine-tuning these distilled models for specific objectives, such as aesthetic appeal or user preference, using Reinforcement Learning (RL) is notoriously unstable and easily falls into reward hacking. In this work, we introduce Flash-DMD, a novel framework that enables fast convergence with distillation and joint RL-based refinement. Specifically, we first propose an efficient timestep-aware distillation strategy that significantly reduces training cost with enhanced realism, outperforming DMD2 with only $2.1\%$ its training cost. Second, we introduce a joint training scheme where the model is fine-tuned with an RL objective while the timestep distillation training continues simultaneously. We demonstrate that the stable, well-defined loss from the ongoing distillation acts as a powerful regularizer, effectively stabilizing the RL training process and preventing policy collapse. Extensive experiments on score-based and flow matching models show that our proposed Flash-DMD not only converges significantly faster but also achieves state-of-the-art generation quality in the few-step sampling regime, outperforming existing methods in visual quality, human preference, and text-image alignment metrics. Our work presents an effective paradigm for training efficient, high-fidelity, and stable generative models. Codes are coming soon.
>
---
#### [new 121] Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs
- **分类: cs.CV**

- **简介: 该论文研究持续视觉指令微调（CVIT）中安全对齐的多模态大模型，针对模型在持续学习中出现任务遗忘与安全性能下降的问题，提出和谐参数适配（HPA）框架，通过参数分区、平衡选择与正交更新，有效兼顾任务性能与安全性，显著提升模型稳定性。**

- **链接: [https://arxiv.org/pdf/2511.20158v1](https://arxiv.org/pdf/2511.20158v1)**

> **作者:** Ziqi Wang; Chang Che; Qi Wang; Hui Ma; Zenglin Shi; Cees G. M. Snoek; Meng Wang
>
> **摘要:** While continual visual instruction tuning (CVIT) has shown promise in adapting multimodal large language models (MLLMs), existing studies predominantly focus on models without safety alignment. This critical oversight ignores the fact that real-world MLLMs inherently require such mechanisms to mitigate potential risks. In this work, we shift our focus to CVIT for safety-aligned MLLMs and observe that during continual adaptation, the model not only suffers from task forgetting but also exhibits degradation in its safety. Achieving a harmonious balance between safety and task performance remains a crucial challenge. To address this, we propose Harmonious Parameter Adaptation (HPA), a post-training framework composed of focusing-based parameter partition, harmoniously balanced parameter selection, and orthogonal parameter adjustment. Specifically, HPA partitions parameters into two types based on their focus on safety or task performance, and selects the focused ones to preserve from a balanced perspective. In addition, HPA imposes orthogonality constraints on parameter updates to further alleviate catastrophic forgetting. Extensive experiments on the CVIT benchmark and safety evaluation datasets demonstrate that HPA better maintains high safety and mitigates forgetting than existing baselines.
>
---
#### [new 122] WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出安全关键推理新任务，针对自动驾驶中单视角难以应对复杂风险的问题，构建了包含3.5万条多视图问答数据的WaymoQA数据集。通过多视角输入与分阶段推理框架，提升模型在高危场景下的决策能力，实验表明其有效改善现有多模态大模型的推理性能。**

- **链接: [https://arxiv.org/pdf/2511.20022v1](https://arxiv.org/pdf/2511.20022v1)**

> **作者:** Seungjun Yu; Seonho Lee; Namho Kim; Jaeyo Shin; Junsung Park; Wonjeong Ryu; Raehyuk Jung; Hyunjung Shim
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have shown strong understanding of driving scenes, drawing interest in their application to autonomous driving. However, high-level reasoning in safety-critical scenarios, where avoiding one traffic risk can create another, remains a major challenge. Such reasoning is often infeasible with only a single front view and requires a comprehensive view of the environment, which we achieve through multi-view inputs. We define Safety-Critical Reasoning as a new task that leverages multi-view inputs to address this challenge. Then, we distill Safety-Critical Reasoning into two stages: first resolve the immediate risk, then mitigate the decision-induced downstream risks. To support this, we introduce WaymoQA, a dataset of 35,000 human-annotated question-answer pairs covering complex, high-risk driving scenarios. The dataset includes multiple-choice and open-ended formats across both image and video modalities. Experiments reveal that existing MLLMs underperform in safety-critical scenarios compared to normal scenes, but fine-tuning with WaymoQA significantly improves their reasoning ability, highlighting the effectiveness of our dataset in developing safer and more reasoning-capable driving agents.
>
---
#### [new 123] SG-OIF: A Stability-Guided Online Influence Framework for Reliable Vision Data
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对深度视觉模型中训练样本影响力在线估计难题，提出SG-OIF框架。通过算法稳定性引导，结合轻量级锚点IHVP与模块化曲率后端，实现实时、可靠的影响力计算，在噪声标签和分布外检测任务上达到SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.19466v1](https://arxiv.org/pdf/2511.19466v1)**

> **作者:** Penghao Rao; Runmin Jiang; Min Xu
>
> **摘要:** Approximating training-point influence on test predictions is critical for deploying deep-learning vision models, essential for locating noisy data. Though the influence function was proposed for attributing how infinitesimal up-weighting or removal of individual training examples affects model outputs, its implementation is still challenging in deep-learning vision models: inverse-curvature computations are expensive, and training non-stationarity invalidates static approximations. Prior works use iterative solvers and low-rank surrogates to reduce cost, but offline computation lags behind training dynamics, and missing confidence calibration yields fragile rankings that misidentify critical examples. To address these challenges, we introduce a Stability-Guided Online Influence Framework (SG-OIF), the first framework that treats algorithmic stability as a real-time controller, which (i) maintains lightweight anchor IHVPs via stochastic Richardson and preconditioned Neumann; (ii) proposes modular curvature backends to modulate per-example influence scores using stability-guided residual thresholds, anomaly gating, and confidence. Experimental results show that SG-OIF achieves SOTA (State-Of-The-Art) on noise-label and out-of-distribution detection tasks across multiple datasets with various corruption. Notably, our approach achieves 91.1\% accuracy in the top 1\% prediction samples on the CIFAR-10 (20\% asym), and gets 99.8\% AUPR score on MNIST, effectively demonstrating that this framework is a practical controller for online influence estimation.
>
---
#### [new 124] HBridge: H-Shape Bridging of Heterogeneous Experts for Unified Multimodal Understanding and Generation
- **分类: cs.CV**

- **简介: 该论文针对统一多模态理解与生成任务，解决现有对称架构因模态差异导致的效率与质量瓶颈。提出非对称H型结构HBridge，通过选择性桥接中间层、减少注意力共享，并引入语义重建令牌，增强跨模态一致性，显著提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2511.20520v1](https://arxiv.org/pdf/2511.20520v1)**

> **作者:** Xiang Wang; Zhifei Zhang; He Zhang; Zhe Lin; Yuqian Zhou; Qing Liu; Shiwei Zhang; Yijun Li; Shaoteng Liu; Haitian Zheng; Jason Kuen; Yuehuan Wang; Changxin Gao; Nong Sang
>
> **摘要:** Recent unified models integrate understanding experts (e.g., LLMs) with generative experts (e.g., diffusion models), achieving strong multimodal performance. However, recent advanced methods such as BAGEL and LMFusion follow the Mixture-of-Transformers (MoT) paradigm, adopting a symmetric design that mirrors one expert to another for convenient initialization and fusion, which remains suboptimal due to inherent modality discrepancies. In this work, we propose HBridge, an asymmetric H-shaped architecture that enables heterogeneous experts to optimally leverage pretrained priors from their respective modality domains. Unlike prior dense fusion strategies that straightforwardly connect all layers between experts via shared attention, HBridge selectively bridges intermediate layers, reducing over 40% attention sharing, which improves efficiency and enhances generation quality. Shallow and deep layers, which capture modality-specific representations, are decoupled, while mid-layer bridging promotes semantic alignment. To further strengthen cross-modal coherence, we introduce semantic reconstruction tokens that explicitly guide the generative expert to reconstruct visual semantic tokens of the target image. Extensive experiments across multiple benchmarks demonstrate the effectiveness and superior performance of HBridge, establishing a new paradigm for unified multimodal generation.
>
---
#### [new 125] Vidi2: Large Multimodal Models for Video Understanding and Creation
- **分类: cs.CV**

- **简介: 该论文提出Vidi2，一种用于视频理解与生成的大型多模态模型。针对视频中细粒度时空定位（STG）与问答（Video QA）难题，提出端到端的时空定位能力，并构建新基准VUE-STG与VUE-TR-V2，显著提升评估质量。实验表明，Vidi2在多项任务上超越主流商业与开源模型。**

- **链接: [https://arxiv.org/pdf/2511.19529v1](https://arxiv.org/pdf/2511.19529v1)**

> **作者:** Vidi Team; Celong Liu; Chia-Wen Kuo; Chuang Huang; Dawei Du; Fan Chen; Guang Chen; Haoji Zhang; Haojun Zhao; Lingxi Zhang; Lu Guo; Lusha Li; Longyin Wen; Qihang Fan; Qingyu Chen; Rachel Deng; Sijie Zhu; Stuart Siew; Tong Jin; Weiyan Tao; Wen Zhong; Xiaohui Shen; Xin Gu; Zhenfang Chen; Zuhua Lin
>
> **摘要:** Video has emerged as the primary medium for communication and creativity on the Internet, driving strong demand for scalable, high-quality video production. Vidi models continue to evolve toward next-generation video creation and have achieved state-of-the-art performance in multimodal temporal retrieval (TR). In its second release, Vidi2 advances video understanding with fine-grained spatio-temporal grounding (STG) and extends its capability to video question answering (Video QA), enabling comprehensive multimodal reasoning. Given a text query, Vidi2 can identify not only the corresponding timestamps but also the bounding boxes of target objects within the output time ranges. This end-to-end spatio-temporal grounding capability enables potential applications in complex editing scenarios, such as plot or character understanding, automatic multi-view switching, and intelligent, composition-aware reframing and cropping. To enable comprehensive evaluation of STG in practical settings, we introduce a new benchmark, VUE-STG, which offers four key improvements over existing STG datasets: 1) Video duration: spans from roughly 10s to 30 mins, enabling long-context reasoning; 2) Query format: queries are mostly converted into noun phrases while preserving sentence-level expressiveness; 3) Annotation quality: all ground-truth time ranges and bounding boxes are manually annotated with high accuracy; 4) Evaluation metric: a refined vIoU/tIoU/vIoU-Intersection scheme. In addition, we upgrade the previous VUE-TR benchmark to VUE-TR-V2, achieving a more balanced video-length distribution and more user-style queries. Remarkably, the Vidi2 model substantially outperforms leading proprietary systems, such as Gemini 3 Pro (Preview) and GPT-5, on both VUE-TR-V2 and VUE-STG, while achieving competitive results with popular open-source models with similar scale on video QA benchmarks.
>
---
#### [new 126] FREE: Uncertainty-Aware Autoregression for Parallel Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文针对扩散Transformer（DiT）生成慢的问题，提出FREE框架，通过轻量级草稿模型与并行验证实现无损加速。利用顶层特征的时序一致性进行特征级自回归，并引入不确定性引导的松弛策略，动态调整接受率，显著提升推理速度，实验表明最快达2.25倍加速，同时保持高质量生成。**

- **链接: [https://arxiv.org/pdf/2511.20390v1](https://arxiv.org/pdf/2511.20390v1)**

> **作者:** Xinwan Wen; Bowen Li; Jiajun Luo; Ye Li; Zhi Wang
>
> **摘要:** Diffusion Transformers (DiTs) achieve state-of-the-art generation quality but require long sequential denoising trajectories, leading to high inference latency. Recent speculative inference methods enable lossless parallel sampling in U-Net-based diffusion models via a drafter-verifier scheme, but their acceleration is limited on DiTs due to insufficient draft accuracy during verification. To address this limitation, we analyze the DiTs' feature dynamics and find the features of the final transformer layer (top-block) exhibit strong temporal consistency and rich semantic abstraction. Based on this insight, we propose FREE, a novel framework that employs a lightweight drafter to perform feature-level autoregression with parallel verification, guaranteeing lossless acceleration with theoretical and empirical support. Meanwhile, prediction variance (uncertainty) of DiTs naturally increases in later denoising steps, reducing acceptance rates under speculative sampling. To mitigate this effect, we further introduce an uncertainty-guided relaxation strategy, forming FREE (relax), which dynamically adjusts the acceptance probability in response to uncertainty levels. Experiments on ImageNet-$512^2$ show that FREE achieves up to $1.86 \times$ acceleration, and FREE (relax) further reaches $2.25 \times$ speedup while maintaining high perceptual and quantitative fidelity in generation quality.
>
---
#### [new 127] Pedestrian Crossing Intention Prediction Using Multimodal Fusion Network
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶中行人过街意图预测任务，解决行为多样性和上下文依赖带来的预测难题。提出一种多模态融合网络，结合视觉与运动特征，通过Transformer模块提取，利用深度引导注意力、模态与时间注意力机制，有效融合多源信息，显著提升预测性能。**

- **链接: [https://arxiv.org/pdf/2511.20008v1](https://arxiv.org/pdf/2511.20008v1)**

> **作者:** Yuanzhe Li; Steffen Müller
>
> **摘要:** Pedestrian crossing intention prediction is essential for the deployment of autonomous vehicles (AVs) in urban environments. Ideal prediction provides AVs with critical environmental cues, thereby reducing the risk of pedestrian-related collisions. However, the prediction task is challenging due to the diverse nature of pedestrian behavior and its dependence on multiple contextual factors. This paper proposes a multimodal fusion network that leverages seven modality features from both visual and motion branches, aiming to effectively extract and integrate complementary cues across different modalities. Specifically, motion and visual features are extracted from the raw inputs using multiple Transformer-based extraction modules. Depth-guided attention module leverages depth information to guide attention towards salient regions in another modality through comprehensive spatial feature interactions. To account for the varying importance of different modalities and frames, modality attention and temporal attention are designed to selectively emphasize informative modalities and effectively capture temporal dependencies. Extensive experiments on the JAAD dataset validate the effectiveness of the proposed network, achieving superior performance compared to the baseline methods.
>
---
#### [new 128] ADNet: A Large-Scale and Extensible Multi-Domain Benchmark for Anomaly Detection Across 380 Real-World Categories
- **分类: cs.CV**

- **简介: 该论文提出ADNet，一个涵盖380个真实世界类别的大规模多领域异常检测基准，解决现有数据集覆盖范围窄、难以评估跨场景泛化能力的问题。通过标准化图像与标注，支持多模态异常检测任务，并提出Dinomaly-m模型，在多类别设置下显著提升性能，为异常检测模型的可扩展性研究提供基础。**

- **链接: [https://arxiv.org/pdf/2511.20169v1](https://arxiv.org/pdf/2511.20169v1)**

> **作者:** Hai Ling; Jia Guo; Zhulin Tao; Yunkang Cao; Donglin Di; Hongyan Xu; Xiu Su; Yang Song; Lei Fan
>
> **摘要:** Anomaly detection (AD) aims to identify defects using normal-only training data. Existing anomaly detection benchmarks (e.g., MVTec-AD with 15 categories) cover only a narrow range of categories, limiting the evaluation of cross-context generalization and scalability. We introduce ADNet, a large-scale, multi-domain benchmark comprising 380 categories aggregated from 49 publicly available datasets across Electronics, Industry, Agrifood, Infrastructure, and Medical domains. The benchmark includes a total of 196,294 RGB images, consisting of 116,192 normal samples for training and 80,102 test images, of which 60,311 are anomalous. All images are standardized with MVTec-style pixel-level annotations and structured text descriptions spanning both spatial and visual attributes, enabling multimodal anomaly detection tasks. Extensive experiments reveal a clear scalability challenge: existing state-of-the-art methods achieve 90.6% I-AUROC in one-for-one settings but drop to 78.5% when scaling to all 380 categories in a multi-class setting. To address this, we propose Dinomaly-m, a context-guided Mixture-of-Experts extension of Dinomaly that expands decoder capacity without increasing inference cost. It achieves 83.2% I-AUROC and 93.1% P-AUROC, demonstrating superior performance over existing approaches. ADNet is designed as a standardized and extensible benchmark, supporting the community in expanding anomaly detection datasets across diverse domains and providing a scalable foundation for future anomaly detection foundation models. Dataset: https://grainnet.github.io/ADNet
>
---
#### [new 129] Look Where It Matters: Training-Free Ultra-HR Remote Sensing VQA via Adaptive Zoom Search
- **分类: cs.CV**

- **简介: 该论文针对超高清遥感图像视觉问答（Ultra-HR RS-VQA）任务，解决模型因全图编码耗内存、缩放预处理丢失细节的问题。提出无训练的ZoomSearch框架，通过自适应多分支缩放搜索定位关键区域，并布局感知重组装，实现高效精准问答。**

- **链接: [https://arxiv.org/pdf/2511.20460v1](https://arxiv.org/pdf/2511.20460v1)**

> **作者:** Yunqi Zhou; Chengjie Jiang; Chun Yuan; Jing Li
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** With advances in satellite constellations, sensor technologies, and imaging pipelines, ultra-high-resolution (Ultra-HR) remote sensing imagery is becoming increasingly widespread. However, current remote sensing foundation models are ill-suited to such inputs: full-image encoding exhausts token and memory budgets, while resize-based preprocessing loses fine-grained and answer-critical details. In this context, guiding the model look where it matters before prediction becomes crucial. Therefore, we present ZoomSearch, a training-free, plug-and-play pipeline that decouples 'where to look' from 'how to answer' for Ultra-HR Remote Sensing Visual Question Answering (RS-VQA). ZoomSearch combines Adaptive Multi-Branch Zoom Search, which performs a hierarchical search over image patches to localize query-relevant regions, with Layout-Aware Patch Reassembly, which reorganizes the selected patches into a compact, layout-faithful canvas. We conduct comprehensive experiments on Ultra-HR RS-VQA benchmarks MME-RealWorld-RS and LRS-VQA, comparing against (i) strong general foundation models, (ii) remote sensing foundation models, (iii) Ultra-HR RS-VQA methods, and (iv) plug-and-play search-based VQA methods. When integrated with LLaVA-ov, ZoomSearch attains state-of-the-art accuracy across diverse tasks, improving the LLaVA-ov baseline by 26.3% on LRS-VQA and 114.8\% on MME-RealWorld-RS. Meanwhile, it achieves much higher inference efficiency, outperforming prior search-based methods by 20%~44% in speed.
>
---
#### [new 130] Large Language Model Aided Birt-Hogg-Dube Syndrome Diagnosis with Multimodal Retrieval-Augmented Generation
- **分类: cs.CV**

- **简介: 该论文针对罕见病BHD诊断中影像样本少、类别区分度低的问题，提出BHD-RAG框架。通过构建DCLDs多模态语料库，利用检索增强生成技术，结合医学知识与CT影像，提升诊断准确性，实现精准、可解释的辅助诊断。**

- **链接: [https://arxiv.org/pdf/2511.19834v1](https://arxiv.org/pdf/2511.19834v1)**

> **作者:** Haoqing Li; Jun Shi; Xianmeng Chen; Qiwei Jia; Rui Wang; Wei Wei; Hong An; Xiaowen Hu
>
> **摘要:** Deep learning methods face dual challenges of limited clinical samples and low inter-class differentiation among Diffuse Cystic Lung Diseases (DCLDs) in advancing Birt-Hogg-Dube syndrome (BHD) diagnosis via Computed Tomography (CT) imaging. While Multimodal Large Language Models (MLLMs) demonstrate diagnostic potential fo such rare diseases, the absence of domain-specific knowledge and referable radiological features intensify hallucination risks. To address this problem, we propose BHD-RAG, a multimodal retrieval-augmented generation framework that integrates DCLD-specific expertise and clinical precedents with MLLMs to improve BHD diagnostic accuracy. BHDRAG employs: (1) a specialized agent generating imaging manifestation descriptions of CT images to construct a multimodal corpus of DCLDs cases. (2) a cosine similarity-based retriever pinpointing relevant imagedescription pairs for query images, and (3) an MLLM synthesizing retrieved evidence with imaging data for diagnosis. BHD-RAG is validated on the dataset involving four types of DCLDs, achieving superior accuracy and generating evidence-based descriptions closely aligned with expert insights.
>
---
#### [new 131] VeriSciQA: An Auto-Verified Dataset for Scientific Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文针对科学图表问答（SVQA）任务，解决开源模型因缺乏高质量数据而表现不佳的问题。提出“生成-验证”框架，通过跨模态一致性检查生成并筛选高可信度的QA对，构建了20,351条标注的VeriSciQA数据集，显著提升开源模型性能。**

- **链接: [https://arxiv.org/pdf/2511.19899v1](https://arxiv.org/pdf/2511.19899v1)**

> **作者:** Yuyi Li; Daoyuan Chen; Zhen Wang; Yutong Lu; Yaliang Li
>
> **摘要:** Large Vision-Language Models (LVLMs) show promise for scientific applications, yet open-source models still struggle with Scientific Visual Question Answering (SVQA), namely answering questions about figures from scientific papers. A key bottleneck lies in the lack of public, large-scale, high-quality SVQA datasets. Although recent work uses LVLMs to synthesize data at scale, we identify systematic errors in their resulting QA pairs, stemming from LVLMs' inherent limitations and information asymmetry between figures and text. To address these challenges, we propose a verification-centric Generate-then-Verify framework that first generates QA pairs with figure-associated textual context, then applies cross-modal consistency checks against figures along with auxiliary filters to eliminate erroneous pairs. We instantiate this framework to curate VeriSciQA, a dataset of 20,351 QA pairs spanning 20 scientific domains and 12 figure types. VeriSciQA poses a challenging benchmark for open-source models, with a substantial accuracy gap between the leading open-source models (64%) and a proprietary model (82%). Moreover, models fine-tuned on VeriSciQA achieve consistent improvements on SVQA benchmarks, with performance gains that scale with data size and surpass models trained on existing datasets. Human evaluation further validates the superior correctness of VeriSciQA. Together, these evidences demonstrate that continued data expansion by our scalable framework can further advance SVQA capability in the open-source community.
>
---
#### [new 132] Perceptual Taxonomy: Evaluating and Guiding Hierarchical Scene Reasoning in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出Perceptual Taxonomy，针对视觉语言模型在物理场景理解中缺乏深层属性推理的问题，构建包含84个细粒度属性的标注数据集与多类型问答基准。通过真实与合成图像上的实验，揭示当前模型在属性驱动推理上表现不佳，且提示学习可有效提升性能。**

- **链接: [https://arxiv.org/pdf/2511.19526v1](https://arxiv.org/pdf/2511.19526v1)**

> **作者:** Jonathan Lee; Xingrui Wang; Jiawei Peng; Luoxin Ye; Zehan Zheng; Tiezheng Zhang; Tao Wang; Wufei Ma; Siyi Chen; Yu-Cheng Chou; Prakhar Kaushik; Alan Yuille
>
> **摘要:** We propose Perceptual Taxonomy, a structured process of scene understanding that first recognizes objects and their spatial configurations, then infers task-relevant properties such as material, affordance, function, and physical attributes to support goal-directed reasoning. While this form of reasoning is fundamental to human cognition, current vision-language benchmarks lack comprehensive evaluation of this ability and instead focus on surface-level recognition or image-text alignment. To address this gap, we introduce Perceptual Taxonomy, a benchmark for physically grounded visual reasoning. We annotate 3173 objects with four property families covering 84 fine-grained attributes. Using these annotations, we construct a multiple-choice question benchmark with 5802 images across both synthetic and real domains. The benchmark contains 28033 template-based questions spanning four types (object description, spatial reasoning, property matching, and taxonomy reasoning), along with 50 expert-crafted questions designed to evaluate models across the full spectrum of perceptual taxonomy reasoning. Experimental results show that leading vision-language models perform well on recognition tasks but degrade by 10 to 20 percent on property-driven questions, especially those requiring multi-step reasoning over structured attributes. These findings highlight a persistent gap in structured visual understanding and the limitations of current models that rely heavily on pattern matching. We also show that providing in-context reasoning examples from simulated scenes improves performance on real-world and expert-curated questions, demonstrating the effectiveness of perceptual-taxonomy-guided prompting.
>
---
#### [new 133] Distilling Cross-Modal Knowledge via Feature Disentanglement
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对跨模态知识蒸馏中模态间表征不一致导致的知识迁移困难问题，提出频域解耦的跨模态知识蒸馏方法。通过分离低频与高频特征，分别施加强对齐与宽松对齐，并引入尺度一致性损失与共享分类器，实现更有效的跨模态知识传递。**

- **链接: [https://arxiv.org/pdf/2511.19887v1](https://arxiv.org/pdf/2511.19887v1)**

> **作者:** Junhong Liu; Yuan Zhang; Tao Huang; Wenchao Xu; Renyu Yang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Knowledge distillation (KD) has proven highly effective for compressing large models and enhancing the performance of smaller ones. However, its effectiveness diminishes in cross-modal scenarios, such as vision-to-language distillation, where inconsistencies in representation across modalities lead to difficult knowledge transfer. To address this challenge, we propose frequency-decoupled cross-modal knowledge distillation, a method designed to decouple and balance knowledge transfer across modalities by leveraging frequency-domain features. We observed that low-frequency features exhibit high consistency across different modalities, whereas high-frequency features demonstrate extremely low cross-modal similarity. Accordingly, we apply distinct losses to these features: enforcing strong alignment in the low-frequency domain and introducing relaxed alignment for high-frequency features. We also propose a scale consistency loss to address distributional shifts between modalities, and employ a shared classifier to unify feature spaces. Extensive experiments across multiple benchmark datasets show our method substantially outperforms traditional KD and state-of-the-art cross-modal KD approaches. Code is available at https://github.com/Johumliu/FD-CMKD.
>
---
#### [new 134] GigaWorld-0: World Models as Data Engine to Empower Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GigaWorld-0，一个用于具身智能的统一世界模型数据引擎。针对真实交互数据稀缺且难获取的问题，构建视频与3D生成协同框架，实现高保真、物理可信、指令对齐的模拟数据生成。通过高效训练框架支持大规模训练，使VLA模型在无真实交互情况下显著提升机器人任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19861v1](https://arxiv.org/pdf/2511.19861v1)**

> **作者:** GigaWorld Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jiagang Zhu; Kerui Li; Mengyuan Xu; Qiuping Deng; Siting Wang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yankai Wang; Yu Cao; Yifan Chang; Yuan Xu; Yun Ye; Yang Wang; Yukun Zhou; Zhengyuan Zhang; Zhehao Dong; Zheng Zhu
>
> **备注:** Project Page: https://gigaworld0.github.io/
>
> **摘要:** World models are emerging as a foundational paradigm for scalable, data-efficient embodied AI. In this work, we present GigaWorld-0, a unified world model framework designed explicitly as a data engine for Vision-Language-Action (VLA) learning. GigaWorld-0 integrates two synergistic components: GigaWorld-0-Video, which leverages large-scale video generation to produce diverse, texture-rich, and temporally coherent embodied sequences under fine-grained control of appearance, camera viewpoint, and action semantics; and GigaWorld-0-3D, which combines 3D generative modeling, 3D Gaussian Splatting reconstruction, physically differentiable system identification, and executable motion planning to ensure geometric consistency and physical realism. Their joint optimization enables the scalable synthesis of embodied interaction data that is visually compelling, spatially coherent, physically plausible, and instruction-aligned. Training at scale is made feasible through our efficient GigaTrain framework, which exploits FP8-precision and sparse attention to drastically reduce memory and compute requirements. We conduct comprehensive evaluations showing that GigaWorld-0 generates high-quality, diverse, and controllable data across multiple dimensions. Critically, VLA model (e.g., GigaBrain-0) trained on GigaWorld-0-generated data achieve strong real-world performance, significantly improving generalization and task success on physical robots without any real-world interaction during training.
>
---
#### [new 135] Scale Where It Matters: Training-Free Localized Scaling for Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对文本生成图像中的扩散模型，提出无需训练的局部化测试时缩放（LoTTS）方法。针对现有方法全局计算效率低、忽略图像质量空间异质性的问题，通过对比注意力信号定位缺陷区域，并局部修复以提升质量与一致性，显著降低计算成本。**

- **链接: [https://arxiv.org/pdf/2511.19917v1](https://arxiv.org/pdf/2511.19917v1)**

> **作者:** Qin Ren; Yufei Wang; Lanqing Guo; Wen Zhang; Zhiwen Fan; Chenyu You
>
> **摘要:** Diffusion models have become the dominant paradigm in text-to-image generation, and test-time scaling (TTS) further improves quality by allocating more computation during inference. However, existing TTS methods operate at the full-image level, overlooking the fact that image quality is often spatially heterogeneous. This leads to unnecessary computation on already satisfactory regions and insufficient correction of localized defects. In this paper, we explore a new direction - Localized TTS - that adaptively resamples defective regions while preserving high-quality regions, thereby substantially reducing the search space. This paradigm poses two central challenges: accurately localizing defects and maintaining global consistency. We propose LoTTS, the first fully training-free framework for localized TTS. For defect localization, LoTTS contrasts cross- and self-attention signals under quality-aware prompts (e.g., high-quality vs. low-quality) to identify defective regions, and then refines them into coherent masks. For consistency, LoTTS perturbs only defective regions and denoises them locally, ensuring that corrections remain confined while the rest of the image remains undisturbed. Extensive experiments on SD2.1, SDXL, and FLUX demonstrate that LoTTS achieves state-of-the-art performance: it consistently improves both local quality and global fidelity, while reducing GPU cost by 2-4x compared to Best-of-N sampling. These findings establish localized TTS as a promising new direction for scaling diffusion models at inference time.
>
---
#### [new 136] Pistachio: Towards Synthetic, Balanced, and Long-Form Video Anomaly Benchmarks
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文提出Pistachio，一个基于生成式方法的视频异常检测与理解基准。针对现有数据集缺乏场景多样性、异常覆盖不均及时间复杂度不足的问题，构建了可控、平衡且长时序的合成视频数据集，支持动态多事件异常分析，推动视频异常理解研究发展。**

- **链接: [https://arxiv.org/pdf/2511.19474v1](https://arxiv.org/pdf/2511.19474v1)**

> **作者:** Jie Li; Hongyi Cai; Mingkang Dong; Muxin Pu; Shan You; Fei Wang; Tao Huang
>
> **摘要:** Automatically detecting abnormal events in videos is crucial for modern autonomous systems, yet existing Video Anomaly Detection (VAD) benchmarks lack the scene diversity, balanced anomaly coverage, and temporal complexity needed to reliably assess real-world performance. Meanwhile, the community is increasingly moving toward Video Anomaly Understanding (VAU), which requires deeper semantic and causal reasoning but remains difficult to benchmark due to the heavy manual annotation effort it demands. In this paper, we introduce Pistachio, a new VAD/VAU benchmark constructed entirely through a controlled, generation-based pipeline. By leveraging recent advances in video generation models, Pistachio provides precise control over scenes, anomaly types, and temporal narratives, effectively eliminating the biases and limitations of Internet-collected datasets. Our pipeline integrates scene-conditioned anomaly assignment, multi-step storyline generation, and a temporally consistent long-form synthesis strategy that produces coherent 41-second videos with minimal human intervention. Extensive experiments demonstrate the scale, diversity, and complexity of Pistachio, revealing new challenges for existing methods and motivating future research on dynamic and multi-event anomaly understanding.
>
---
#### [new 137] AD-R1: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving with Impartial World Models
- **分类: cs.CV**

- **简介: 该论文针对端到端自动驾驶中的安全与长尾事件难题，提出基于公正世界模型的闭环强化学习框架。通过反事实数据合成构建能真实预测危险的模型，使其在策略优化中充当内部批评者，显著提升失败预测能力，有效减少安全违规，推动更安全的自动驾驶系统发展。**

- **链接: [https://arxiv.org/pdf/2511.20325v1](https://arxiv.org/pdf/2511.20325v1)**

> **作者:** Tianyi Yan; Tao Tang; Xingtai Gui; Yongkang Li; Jiasen Zhesng; Weiyao Huang; Lingdong Kong; Wencheng Han; Xia Zhou; Xueyang Zhang; Yifei Zhan; Kun Zhan; Cheng-zhong Xu; Jianbing Shen
>
> **摘要:** End-to-end models for autonomous driving hold the promise of learning complex behaviors directly from sensor data, but face critical challenges in safety and handling long-tail events. Reinforcement Learning (RL) offers a promising path to overcome these limitations, yet its success in autonomous driving has been elusive. We identify a fundamental flaw hindering this progress: a deep seated optimistic bias in the world models used for RL. To address this, we introduce a framework for post-training policy refinement built around an Impartial World Model. Our primary contribution is to teach this model to be honest about danger. We achieve this with a novel data synthesis pipeline, Counterfactual Synthesis, which systematically generates a rich curriculum of plausible collisions and off-road events. This transforms the model from a passive scene completer into a veridical forecaster that remains faithful to the causal link between actions and outcomes. We then integrate this Impartial World Model into our closed-loop RL framework, where it serves as an internal critic. During refinement, the agent queries the critic to ``dream" of the outcomes for candidate actions. We demonstrate through extensive experiments, including on a new Risk Foreseeing Benchmark, that our model significantly outperforms baselines in predicting failures. Consequently, when used as a critic, it enables a substantial reduction in safety violations in challenging simulations, proving that teaching a model to dream of danger is a critical step towards building truly safe and intelligent autonomous agents.
>
---
#### [new 138] Modality-Balanced Collaborative Distillation for Multi-Modal Domain Generalization
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多模态领域泛化（MMDG）中加权平均（WA）因模态收敛速度差异导致的早期偏差问题，提出MBCD框架。通过自适应模态丢弃、梯度一致性约束与基于WA的教师蒸馏，实现模态协同优化，提升模型在未见域上的泛化性能。**

- **链接: [https://arxiv.org/pdf/2511.20258v1](https://arxiv.org/pdf/2511.20258v1)**

> **作者:** Xiaohan Wang; Zhangtao Cheng; Ting Zhong; Leiting Chen; Fan Zhou
>
> **摘要:** Weight Averaging (WA) has emerged as a powerful technique for enhancing generalization by promoting convergence to a flat loss landscape, which correlates with stronger out-of-distribution performance. However, applying WA directly to multi-modal domain generalization (MMDG) is challenging: differences in optimization speed across modalities lead WA to overfit to faster-converging ones in early stages, suppressing the contribution of slower yet complementary modalities, thereby hindering effective modality fusion and skewing the loss surface toward sharper, less generalizable minima. To address this issue, we propose MBCD, a unified collaborative distillation framework that retains WA's flatness-inducing advantages while overcoming its shortcomings in multi-modal contexts. MBCD begins with adaptive modality dropout in the student model to curb early-stage bias toward dominant modalities. A gradient consistency constraint then aligns learning signals between uni-modal branches and the fused representation, encouraging coordinated and smoother optimization. Finally, a WA-based teacher conducts cross-modal distillation by transferring fused knowledge to each uni-modal branch, which strengthens cross-modal interactions and steer convergence toward flatter solutions. Extensive experiments on MMDG benchmarks show that MBCD consistently outperforms existing methods, achieving superior accuracy and robustness across diverse unseen domains.
>
---
#### [new 139] LiMT: A Multi-task Liver Image Benchmark Dataset
- **分类: cs.CV**

- **简介: 该论文提出LiMT多任务肝脏影像基准数据集，旨在解决现有数据集任务单一、限制CAD技术发展的难题。通过构建包含150例增强CT的多任务数据集，支持肝肿瘤分割、多标签分类与检测，促进任务间关联研究，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19889v1](https://arxiv.org/pdf/2511.19889v1)**

> **作者:** Zhe Liu; Kai Han; Siqi Ma; Yan Zhu; Jun Chen; Chongwen Lyu; Xinyi Qiu; Chengxuan Qian; Yuqing Song; Yi Liu; Liyuan Tian; Yang Ji; Yuefeng Li
>
> **备注:** IEEE Journal of Biomedical and Health Informatics
>
> **摘要:** Computer-aided diagnosis (CAD) technology can assist clinicians in evaluating liver lesions and intervening with treatment in time. Although CAD technology has advanced in recent years, the application scope of existing datasets remains relatively limited, typically supporting only single tasks, which has somewhat constrained the development of CAD technology. To address the above limitation, in this paper, we construct a multi-task liver dataset (LiMT) used for liver and tumor segmentation, multi-label lesion classification, and lesion detection based on arterial phase-enhanced computed tomography (CT), potentially providing an exploratory solution that is able to explore the correlation between tasks and does not need to worry about the heterogeneity between task-specific datasets during training. The dataset includes CT volumes from 150 different cases, comprising four types of liver diseases as well as normal cases. Each volume has been carefully annotated and calibrated by experienced clinicians. This public multi-task dataset may become a valuable resource for the medical imaging research community in the future. In addition, this paper not only provides relevant baseline experimental results but also reviews existing datasets and methods related to liver-related tasks. Our dataset is available at https://drive.google.com/drive/folders/1l9HRK13uaOQTNShf5pwgSz3OTanWjkag?usp=sharing.
>
---
#### [new 140] Low-Resolution Editing is All You Need for High-Resolution Editing
- **分类: cs.CV**

- **简介: 该论文提出高分辨率图像编辑任务，针对现有方法仅支持低分辨率（≤1K）的局限，设计测试时优化框架，通过分块优化、细节迁移与同步策略，实现高质量高分辨率编辑，推动高分辨率内容生成发展。**

- **链接: [https://arxiv.org/pdf/2511.19945v1](https://arxiv.org/pdf/2511.19945v1)**

> **作者:** Junsung Lee; Hyunsoo Lee; Yong Jae Lee; Bohyung Han
>
> **备注:** 14 pages, 8 figures, 2 tables
>
> **摘要:** High-resolution content creation is rapidly emerging as a central challenge in both the vision and graphics communities. While images serve as the most fundamental modality for visual expression, content generation that aligns with the user intent requires effective, controllable high-resolution image manipulation mechanisms. However, existing approaches remain limited to low-resolution settings, typically supporting only up to 1K resolution. In this work, we introduce the task of high-resolution image editing and propose a test-time optimization framework to address it. Our method performs patch-wise optimization on high-resolution source images, followed by a fine-grained detail transfer module and a novel synchronization strategy to maintain consistency across patches. Extensive experiments show that our method produces high-quality edits, facilitating the way toward high-resolution content creation.
>
---
#### [new 141] Text-guided Controllable Diffusion for Realistic Camouflage Images Generation
- **分类: cs.CV**

- **简介: 该论文研究可控文本引导的伪装图像生成任务，旨在解决现有方法忽视伪装物体与背景逻辑关系导致结果不自然的问题。提出CT-CIG框架，通过大视觉语言模型生成高质量文本提示，结合轻量控制器与频率交互优化模块，实现语义一致、逼真的伪装图像生成。**

- **链接: [https://arxiv.org/pdf/2511.20218v1](https://arxiv.org/pdf/2511.20218v1)**

> **作者:** Yuhang Qian; Haiyan Chen; Wentong Li; Ningzhong Liu; Jie Qin
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Camouflage Images Generation (CIG) is an emerging research area that focuses on synthesizing images in which objects are harmoniously blended and exhibit high visual consistency with their surroundings. Existing methods perform CIG by either fusing objects into specific backgrounds or outpainting the surroundings via foreground object-guided diffusion. However, they often fail to obtain natural results because they overlook the logical relationship between camouflaged objects and background environments. To address this issue, we propose CT-CIG, a Controllable Text-guided Camouflage Images Generation method that produces realistic and logically plausible camouflage images. Leveraging Large Visual Language Models (VLM), we design a Camouflage-Revealing Dialogue Mechanism (CRDM) to annotate existing camouflage datasets with high-quality text prompts. Subsequently, the constructed image-prompt pairs are utilized to finetune Stable Diffusion, incorporating a lightweight controller to guide the location and shape of camouflaged objects for enhanced camouflage scene fitness. Moreover, we design a Frequency Interaction Refinement Module (FIRM) to capture high-frequency texture features, facilitating the learning of complex camouflage patterns. Extensive experiments, including CLIPScore evaluation and camouflage effectiveness assessment, demonstrate the semantic alignment of our generated text prompts and CT-CIG's ability to produce photorealistic camouflage images.
>
---
#### [new 142] Blinking Beyond EAR: A Stable Eyelid Angle Metric for Driver Drowsiness Detection and Data Augmentation
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文针对驾驶员困倦检测任务，解决传统眼睑状态评估方法在视角变化下不稳定的问题。提出新型眼睑角度（ELA）度量，具备几何稳定性与可重复性，用于精准捕捉眨眼时序特征。基于ELA生成可控的合成数据，增强训练集多样性，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.19519v1](https://arxiv.org/pdf/2511.19519v1)**

> **作者:** Mathis Wolter; Julie Stephany Berrio Perez; Mao Shan
>
> **备注:** 8 pages, 5 figures, 3 tables
>
> **摘要:** Detecting driver drowsiness reliably is crucial for enhancing road safety and supporting advanced driver assistance systems (ADAS). We introduce the Eyelid Angle (ELA), a novel, reproducible metric of eye openness derived from 3D facial landmarks. Unlike conventional binary eye state estimators or 2D measures, such as the Eye Aspect Ratio (EAR), the ELA provides a stable geometric description of eyelid motion that is robust to variations in camera angle. Using the ELA, we design a blink detection framework that extracts temporal characteristics, including the closing, closed, and reopening durations, which are shown to correlate with drowsiness levels. To address the scarcity and risk of collecting natural drowsiness data, we further leverage ELA signals to animate rigged avatars in Blender 3D, enabling the creation of realistic synthetic datasets with controllable noise, camera viewpoints, and blink dynamics. Experimental results in public driver monitoring datasets demonstrate that the ELA offers lower variance under viewpoint changes compared to EAR and achieves accurate blink detection. At the same time, synthetic augmentation expands the diversity of training data for drowsiness recognition. Our findings highlight the ELA as both a reliable biometric measure and a powerful tool for generating scalable datasets in driver state monitoring.
>
---
#### [new 143] A Physics-Informed Loss Function for Boundary-Consistent and Robust Artery Segmentation in DSA Sequences
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对数字减影血管造影（DSA）中脑动脉分割任务，解决传统损失函数忽略边界几何与物理一致性导致分割碎片化的问题。提出物理信息损失（PIL），基于材料物理中的位错理论建模边界交互，引入物理正则化项，提升分割的平滑性与结构一致性。在多个网络架构和数据集上验证，显著优于经典损失函数。**

- **链接: [https://arxiv.org/pdf/2511.20501v1](https://arxiv.org/pdf/2511.20501v1)**

> **作者:** Muhammad Irfan; Nasir Rahim; Khalid Mahmood Malik
>
> **摘要:** Accurate extraction and segmentation of the cerebral arteries from digital subtraction angiography (DSA) sequences is essential for developing reliable clinical management models of complex cerebrovascular diseases. Conventional loss functions often rely solely on pixel-wise overlap, overlooking the geometric and physical consistency of vascular boundaries, which can lead to fragmented or unstable vessel predictions. To overcome this limitation, we propose a novel \textit{Physics-Informed Loss} (PIL) that models the interaction between the predicted and ground-truth boundaries as an elastic process inspired by dislocation theory in materials physics. This formulation introduces a physics-based regularization term that enforces smooth contour evolution and structural consistency, allowing the network to better capture fine vascular geometry. The proposed loss is integrated into several segmentation architectures, including U-Net, U-Net++, SegFormer, and MedFormer, and evaluated on two public benchmarks: DIAS and DSCA. Experimental results demonstrate that PIL consistently outperforms conventional loss functions such as Cross-Entropy, Dice, Active Contour, and Surface losses, achieving superior sensitivity, F1 score, and boundary coherence. These findings confirm that the incorporation of physics-based boundary interactions into deep neural networks improves both the precision and robustness of vascular segmentation in dynamic angiographic imaging. The implementation of the proposed method is publicly available at https://github.com/irfantahir301/Physicsis_loss.
>
---
#### [new 144] IrisNet: Infrared Image Status Awareness Meta Decoder for Infrared Small Targets Detection
- **分类: cs.CV**

- **简介: 该论文针对红外小目标检测任务，解决复杂背景下低信噪比与静态模型适应性差的问题。提出IrisNet，通过图像-解码器变换器动态生成解码器参数，结合高频信息增强感知，实现跨场景自适应检测，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.20319v1](https://arxiv.org/pdf/2511.20319v1)**

> **作者:** Xuelin Qian; Jiaming Lu; Zixuan Wang; Wenxuan Wang; Zhongling Huang; Dingwen Zhang; Junwei Han
>
> **备注:** 10pages,5figures
>
> **摘要:** Infrared Small Target Detection (IRSTD) faces significant challenges due to low signal-to-noise ratios, complex backgrounds, and the absence of discernible target features. While deep learning-based encoder-decoder frameworks have advanced the field, their static pattern learning suffers from pattern drift across diverse scenarios (\emph{e.g.}, day/night variations, sky/maritime/ground domains), limiting robustness. To address this, we propose IrisNet, a novel meta-learned framework that dynamically adapts detection strategies to the input infrared image status. Our approach establishes a dynamic mapping between infrared image features and entire decoder parameters via an image-to-decoder transformer. More concretely, we represent the parameterized decoder as a structured 2D tensor preserving hierarchical layer correlations and enable the transformer to model inter-layer dependencies through self-attention while generating adaptive decoding patterns via cross-attention. To further enhance the perception ability of infrared images, we integrate high-frequency components to supplement target-position and scene-edge information. Experiments on NUDT-SIRST, NUAA-SIRST, and IRSTD-1K datasets demonstrate the superiority of our IrisNet, achieving state-of-the-art performance.
>
---
#### [new 145] Temporal-Visual Semantic Alignment: A Unified Architecture for Transferring Spatial Priors from Vision Models to Zero-Shot Temporal Tasks
- **分类: cs.CV**

- **简介: 该论文提出TimeArtist框架，解决时间序列与视觉语义间缺乏语义对齐的问题。针对零样本时序任务中视觉先验迁移不足的挑战，通过自监督预训练与投影对齐，实现时间序列到图像的高质量生成，建立时序动态与视觉语义的统一跨模态框架。**

- **链接: [https://arxiv.org/pdf/2511.19856v1](https://arxiv.org/pdf/2511.19856v1)**

> **作者:** Xiangkai Ma; Han Zhang; Wenzhong Li; Sanglu Lu
>
> **摘要:** Large Multimodal Models (LMMs) have achieved remarkable progress in aligning and generating content across text and image modalities. However, the potential of using non-visual, continuous sequential, as a conditioning signal for high-fidelity image generation remains largely unexplored. Furthermore, existing methods that convert series into "pseudo-images" for temporal forecasting fail to establish semantic-level alignment. In this paper, we propose TimeArtist, a temporal-visual conversion framework that pioneers semantic-level alignment between time series fluctuations and visual concepts. It pioneers a "warmup-align" paradigm: first, a dual-autoencoder and shared quantizer are self-supervised trained on large-scale datasets to learn modality-shared representations. Then, the encoders and quantizer are frozen, and a projection is introduced to align temporal and visual samples at the representation level. TimeArtist establishes a versatile cross-modal framework, enabling high-quality, diverse image generation directly from time series, while capturing temporal fluctuation patterns to render images as styles transfer. Extensive experiments show that TimeArtist achieves satisfactory performance in image generation metrics, while also attaining superior results in zero-shot temporal tasks. Our work establishes a new paradigm for cross-modal generation, bridging the gap between temporal dynamics and visual semantics.
>
---
#### [new 146] SKEL-CF: Coarse-to-Fine Biomechanical Skeleton and Surface Mesh Recovery
- **分类: cs.CV**

- **简介: 该论文提出SKEL-CF框架，解决3D人体姿态与形状估计中生物力学真实性的不足问题。针对SKEL模型参数估计困难，采用粗到细的Transformer架构，并构建SKEL对齐数据集4DHuman-SKEL，结合相机建模提升精度。在MOYO数据集上显著优于现有方法，实现了更精准、生物力学一致的人体重建。**

- **链接: [https://arxiv.org/pdf/2511.20157v1](https://arxiv.org/pdf/2511.20157v1)**

> **作者:** Da Li; Ji-Ping Jin; Xuanlong Yu; Wei Liu; Xiaodong Cun; Kai Chen; Rui Fan; Jiangang Kong; Shen Xi
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Parametric 3D human models such as SMPL have driven significant advances in human pose and shape estimation, yet their simplified kinematics limit biomechanical realism. The recently proposed SKEL model addresses this limitation by re-rigging SMPL with an anatomically accurate skeleton. However, estimating SKEL parameters directly remains challenging due to limited training data, perspective ambiguities, and the inherent complexity of human articulation. We introduce SKEL-CF, a coarse-to-fine framework for SKEL parameter estimation. SKEL-CF employs a transformer-based encoder-decoder architecture, where the encoder predicts coarse camera and SKEL parameters, and the decoder progressively refines them in successive layers. To ensure anatomically consistent supervision, we convert the existing SMPL-based dataset 4DHuman into a SKEL-aligned version, 4DHuman-SKEL, providing high-quality training data for SKEL estimation. In addition, to mitigate depth and scale ambiguities, we explicitly incorporate camera modeling into the SKEL-CF pipeline and demonstrate its importance across diverse viewpoints. Extensive experiments validate the effectiveness of the proposed design. On the challenging MOYO dataset, SKEL-CF achieves 85.0 MPJPE / 51.4 PA-MPJPE, significantly outperforming the previous SKEL-based state-of-the-art HSMR (104.5 / 79.6). These results establish SKEL-CF as a scalable and anatomically faithful framework for human motion analysis, bridging the gap between computer vision and biomechanics. Our implementation is available on the project page: https://pokerman8.github.io/SKEL-CF/.
>
---
#### [new 147] Realizing Fully-Integrated, Low-Power, Event-Based Pupil Tracking with Neuromorphic Hardware
- **分类: cs.CV**

- **简介: 该论文针对可穿戴设备中低功耗、高频率眼动追踪难题，提出首个全集成、电池供电的事件基瞳孔中心追踪系统。基于Speck2f芯片，结合脉冲神经网络与低功耗微控制器，实现100Hz实时追踪，功耗低于5mW/眼，解决了嵌入式场景下能效与精度的矛盾。**

- **链接: [https://arxiv.org/pdf/2511.20175v1](https://arxiv.org/pdf/2511.20175v1)**

> **作者:** Federico Paredes-Valles; Yoshitaka Miyatani; Kirk Y. W. Scheper
>
> **备注:** 17 pages, 14 figures, 3 tables
>
> **摘要:** Eye tracking is fundamental to numerous applications, yet achieving robust, high-frequency tracking with ultra-low power consumption remains challenging for wearable platforms. While event-based vision sensors offer microsecond resolution and sparse data streams, they have lacked fully integrated, low-power processing solutions capable of real-time inference. In this work, we present the first battery-powered, wearable pupil-center-tracking system with complete on-device integration, combining event-based sensing and neuromorphic processing on the commercially available Speck2f system-on-chip with lightweight coordinate decoding on a low-power microcontroller. Our solution features a novel uncertainty-quantifying spiking neural network with gated temporal decoding, optimized for strict memory and bandwidth constraints, complemented by systematic deployment mechanisms that bridge the reality gap. We validate our system on a new multi-user dataset and demonstrate a wearable prototype with dual neuromorphic devices achieving robust binocular pupil tracking at 100 Hz with an average power consumption below 5 mW per eye. Our work demonstrates that end-to-end neuromorphic computing enables practical, always-on eye tracking for next-generation energy-efficient wearable systems.
>
---
#### [new 148] GHR-VQA: Graph-guided Hierarchical Relational Reasoning for Video Question Answering
- **分类: cs.CV**

- **简介: 该论文针对视频问答任务，提出GHR-VQA框架，通过构建以人类为中心的层次化图结构，捕捉视频中人与物体的复杂交互。利用图神经网络实现跨帧关系推理，增强对时空动态的理解，显著提升对象关系推理能力，在AGQA数据集上超越现有方法7.3%。**

- **链接: [https://arxiv.org/pdf/2511.20201v1](https://arxiv.org/pdf/2511.20201v1)**

> **作者:** Dionysia Danai Brilli; Dimitrios Mallis; Vassilis Pitsikalis; Petros Maragos
>
> **摘要:** We propose GHR-VQA, Graph-guided Hierarchical Relational Reasoning for Video Question Answering (Video QA), a novel human-centric framework that incorporates scene graphs to capture intricate human-object interactions within video sequences. Unlike traditional pixel-based methods, each frame is represented as a scene graph and human nodes across frames are linked to a global root, forming the video-level graph and enabling cross-frame reasoning centered on human actors. The video-level graphs are then processed by Graph Neural Networks (GNNs), transforming them into rich, context-aware embeddings for efficient processing. Finally, these embeddings are integrated with question features in a hierarchical network operating across different abstraction levels, enhancing both local and global understanding of video content. This explicit human-rooted structure enhances interpretability by decomposing actions into human-object interactions and enables a more profound understanding of spatiotemporal dynamics. We validate our approach on the Action Genome Question Answering (AGQA) dataset, achieving significant performance improvements, including a 7.3% improvement in object-relation reasoning over the state of the art.
>
---
#### [new 149] AlignBench: Benchmarking Fine-Grained Image-Text Alignment with Synthetic Image-Caption Pairs
- **分类: cs.CV**

- **简介: 该论文提出AlignBench，一个用于评估细粒度图像-文本对齐的基准。针对现有方法依赖规则扰动或简短描述、难以衡量精细对齐的问题，构建由多种模型生成的详细图文对，并人工标注正确性。通过评估多类解码器型视觉语言模型，发现CLIP模型对组合推理仍不敏感，检测器存在过评分与自偏好问题。**

- **链接: [https://arxiv.org/pdf/2511.20515v1](https://arxiv.org/pdf/2511.20515v1)**

> **作者:** Kuniaki Saito; Risa Shinoda; Shohei Tanaka; Tosho Hirasawa; Fumio Okura; Yoshitaka Ushiku
>
> **备注:** Project Page: https://dahlian00.github.io/AlignBench/
>
> **摘要:** Assessing image-text alignment models such as CLIP is crucial for bridging visual and linguistic representations. Yet existing benchmarks rely on rule-based perturbations or short captions, limiting their ability to measure fine-grained alignment. We introduce AlignBench, a benchmark that provides a new indicator of image-text alignment by evaluating detailed image-caption pairs generated by diverse image-to-text and text-to-image models. Each sentence is annotated for correctness, enabling direct assessment of VLMs as alignment evaluators. Benchmarking a wide range of decoder-based VLMs reveals three key findings: (i) CLIP-based models, even those tailored for compositional reasoning, remain nearly blind; (ii) detectors systematically over-score early sentences; and (iii) they show strong self-preference, favoring their own outputs and harming detection performance. Our project page will be available at https://dahlian00.github.io/AlignBench/.
>
---
#### [new 150] MapRF: Weakly Supervised Online HD Map Construction via NeRF-Guided Self-Training
- **分类: cs.CV**

- **简介: 该论文提出MapRF，一种弱监督在线高精地图构建方法。针对现有方法依赖昂贵3D标注的问题，利用2D图像标签与NeRF引导的自训练机制，生成高质量伪标签并迭代优化地图网络，通过射线匹配缓解误差累积，实现低成本、可扩展的HD地图构建。**

- **链接: [https://arxiv.org/pdf/2511.19527v1](https://arxiv.org/pdf/2511.19527v1)**

> **作者:** Hongyu Lyu; Thomas Monninger; Julie Stephany Berrio Perez; Mao Shan; Zhenxing Ming; Stewart Worrall
>
> **摘要:** Autonomous driving systems benefit from high-definition (HD) maps that provide critical information about road infrastructure. The online construction of HD maps offers a scalable approach to generate local maps from on-board sensors. However, existing methods typically rely on costly 3D map annotations for training, which limits their generalization and scalability across diverse driving environments. In this work, we propose MapRF, a weakly supervised framework that learns to construct 3D maps using only 2D image labels. To generate high-quality pseudo labels, we introduce a novel Neural Radiance Fields (NeRF) module conditioned on map predictions, which reconstructs view-consistent 3D geometry and semantics. These pseudo labels are then iteratively used to refine the map network in a self-training manner, enabling progressive improvement without additional supervision. Furthermore, to mitigate error accumulation during self-training, we propose a Map-to-Ray Matching strategy that aligns map predictions with camera rays derived from 2D labels. Extensive experiments on the Argoverse 2 and nuScenes datasets demonstrate that MapRF achieves performance comparable to fully supervised methods, attaining around 75% of the baseline while surpassing several approaches using only 2D labels. This highlights the potential of MapRF to enable scalable and cost-effective online HD map construction for autonomous driving.
>
---
#### [new 151] Tracking and Segmenting Anything in Any Modality
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文聚焦视频理解中的跟踪与分割任务，针对现有方法在多模态输入和多任务学习中存在跨模态分布差异与特征表示差异的问题，提出SATA框架。通过解耦的专家混合机制与任务感知的多目标跟踪管道，实现任意模态下多种任务的统一建模，显著提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19475v1](https://arxiv.org/pdf/2511.19475v1)**

> **作者:** Tianlu Zhang; Qiang Zhang; Guiguang Ding; Jungong Han
>
> **备注:** Accpetd by AAAI 2026
>
> **摘要:** Tracking and segmentation play essential roles in video understanding, providing basic positional information and temporal association of objects within video sequences. Despite their shared objective, existing approaches often tackle these tasks using specialized architectures or modality-specific parameters, limiting their generalization and scalability. Recent efforts have attempted to unify multiple tracking and segmentation subtasks from the perspectives of any modality input or multi-task inference. However, these approaches tend to overlook two critical challenges: the distributional gap across different modalities and the feature representation gap across tasks. These issues hinder effective cross-task and cross-modal knowledge sharing, ultimately constraining the development of a true generalist model. To address these limitations, we propose a universal tracking and segmentation framework named SATA, which unifies a broad spectrum of tracking and segmentation subtasks with any modality input. Specifically, a Decoupled Mixture-of-Expert (DeMoE) mechanism is presented to decouple the unified representation learning task into the modeling process of cross-modal shared knowledge and specific information, thus enabling the model to maintain flexibility while enhancing generalization. Additionally, we introduce a Task-aware Multi-object Tracking (TaMOT) pipeline to unify all the task outputs as a unified set of instances with calibrated ID information, thereby alleviating the degradation of task-specific knowledge during multi-task training. SATA demonstrates superior performance on 18 challenging tracking and segmentation benchmarks, offering a novel perspective for more generalizable video understanding.
>
---
#### [new 152] DOGE: Differentiable Bezier Graph Optimization for Road Network Extraction
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对遥感图像中道路网络矢量提取任务，解决传统方法依赖难获取的曲线真值、难以建模弯曲道路的问题。提出可微分贝塞尔图（DOGE）框架，从分割掩码直接学习参数化贝塞尔图，通过可微渲染与拓扑优化交替迭代，实现几何与拓扑联合优化，显著提升道路提取精度。**

- **链接: [https://arxiv.org/pdf/2511.19850v1](https://arxiv.org/pdf/2511.19850v1)**

> **作者:** Jiahui Sun; Junran Lu; Jinhui Yin; Yishuo Xu; Yuanqi Li; Yanwen Guo
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Automatic extraction of road networks from aerial imagery is a fundamental task, yet prevailing methods rely on polylines that struggle to model curvilinear geometry. We maintain that road geometry is inherently curve-based and introduce the Bézier Graph, a differentiable parametric curve-based representation. The primary obstacle to this representation is to obtain the difficult-to-construct vector ground-truth (GT). We sidestep this bottleneck by reframing the task as a global optimization problem over the Bézier Graph. Our framework, DOGE, operationalizes this paradigm by learning a parametric Bézier Graph directly from segmentation masks, eliminating the need for curve GT. DOGE holistically optimizes the graph by alternating between two complementary modules: DiffAlign continuously optimizes geometry via differentiable rendering, while TopoAdapt uses discrete operators to refine its topology. Our method sets a new state-of-the-art on the large-scale SpaceNet and CityScale benchmarks, presenting a new paradigm for generating high-fidelity vector maps of road networks. We will release our code and related data.
>
---
#### [new 153] TReFT: Taming Rectified Flow Models For One-Step Image Translation
- **分类: cs.CV**

- **简介: 该论文针对图像翻译任务中矩形流模型依赖多步去噪、无法实时推理的问题，提出TReFT方法。通过利用预训练模型的预测速度并引入轻量设计，实现单步推理下的稳定对抗训练，显著提升效率，同时保持与当前最优方法相当的性能。**

- **链接: [https://arxiv.org/pdf/2511.20307v1](https://arxiv.org/pdf/2511.20307v1)**

> **作者:** Shengqian Li; Ming Gao; Yi Liu; Zuzeng Lin; Feng Wang; Feng Dai
>
> **摘要:** Rectified Flow (RF) models have advanced high-quality image and video synthesis via optimal transport theory. However, when applied to image-to-image translation, they still depend on costly multi-step denoising, hindering real-time applications. Although the recent adversarial training paradigm, CycleGAN-Turbo, works in pretrained diffusion models for one-step image translation, we find that directly applying it to RF models leads to severe convergence issues. In this paper, we analyze these challenges and propose TReFT, a novel method to Tame Rectified Flow models for one-step image Translation. Unlike previous works, TReFT directly uses the velocity predicted by pretrained DiT or UNet as output-a simple yet effective design that tackles the convergence issues under adversarial training with one-step inference. This design is mainly motivated by a novel observation that, near the end of the denoising process, the velocity predicted by pretrained RF models converges to the vector from origin to the final clean image, a property we further justify through theoretical analysis. When applying TReFT to large pretrained RF models such as SD3.5 and FLUX, we introduce memory-efficient latent cycle-consistency and identity losses during training, as well as lightweight architectural simplifications for faster inference. Pretrained RF models finetuned with TReFT achieve performance comparable to sota methods across multiple image translation datasets while enabling real-time inference.
>
---
#### [new 154] HunyuanOCR Technical Report
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出轻量级多模态模型HunyuanOCR（1B参数），面向OCR任务，解决传统方法依赖预处理、效率低及通用性差的问题。通过端到端架构与数据驱动+强化学习策略，实现文本检测、识别、信息抽取与翻译等多功能统一，性能超越现有方案，在ICDAR 2025挑战赛中夺冠。**

- **链接: [https://arxiv.org/pdf/2511.19575v1](https://arxiv.org/pdf/2511.19575v1)**

> **作者:** Hunyuan Vision Team; Pengyuan Lyu; Xingyu Wan; Gengluo Li; Shangpin Peng; Weinong Wang; Liang Wu; Huawen Shen; Yu Zhou; Canhui Tang; Qi Yang; Qiming Peng; Bin Luo; Hower Yang; Houwen Peng; Hongming Yang; Senhao Xie; Binghong Wu; Mana Yang; Sergey Wang; Raccoon Liu; Dick Zhu; Jie Jiang; Linus; Han Hu; Chengquan Zhang
>
> **摘要:** This paper presents HunyuanOCR, a commercial-grade, open-source, and lightweight (1B parameters) Vision-Language Model (VLM) dedicated to OCR tasks. The architecture comprises a Native Vision Transformer (ViT) and a lightweight LLM connected via an MLP adapter. HunyuanOCR demonstrates superior performance, outperforming commercial APIs, traditional pipelines, and larger models (e.g., Qwen3-VL-4B). Specifically, it surpasses current public solutions in perception tasks (Text Spotting, Parsing) and excels in semantic tasks (IE, Text Image Translation), securing first place in the ICDAR 2025 DIMT Challenge (Small Model Track). Furthermore, it achieves state-of-the-art (SOTA) results on OCRBench among VLMs with fewer than 3B parameters. HunyuanOCR achieves breakthroughs in three key aspects: 1) Unifying Versatility and Efficiency: We implement comprehensive support for core capabilities including spotting, parsing, IE, VQA, and translation within a lightweight framework. This addresses the limitations of narrow "OCR expert models" and inefficient "General VLMs". 2) Streamlined End-to-End Architecture: Adopting a pure end-to-end paradigm eliminates dependencies on pre-processing modules (e.g., layout analysis). This fundamentally resolves error propagation common in traditional pipelines and simplifies system deployment. 3) Data-Driven and RL Strategies: We confirm the critical role of high-quality data and, for the first time in the industry, demonstrate that Reinforcement Learning (RL) strategies yield significant performance gains in OCR tasks. HunyuanOCR is officially open-sourced on HuggingFace. We also provide a high-performance deployment solution based on vLLM, placing its production efficiency in the top tier. We hope this model will advance frontier research and provide a solid foundation for industrial applications.
>
---
#### [new 155] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型（LVLMs）的对抗攻击，解决现有方法难以精确控制图像局部语义的问题。针对patch-token表示中语义纠缠导致操控不可控的难题，提出V-Attack方法，通过操纵Transformer中的值特征（V），利用自值增强与文本引导的值操纵模块，实现对特定概念的精准语义篡改，显著提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2511.20223v1](https://arxiv.org/pdf/2511.20223v1)**

> **作者:** Sen Nie; Jie Zhang; Jianxin Yan; Shiguang Shan; Xilin Chen
>
> **备注:** 21 pages
>
> **摘要:** Adversarial attacks have evolved from simply disrupting predictions on conventional task-specific models to the more complex goal of manipulating image semantics on Large Vision-Language Models (LVLMs). However, existing methods struggle with controllability and fail to precisely manipulate the semantics of specific concepts in the image. We attribute this limitation to semantic entanglement in the patch-token representations on which adversarial attacks typically operate: global context aggregated by self-attention in the vision encoder dominates individual patch features, making them unreliable handles for precise local semantic manipulation. Our systematic investigation reveals a key insight: value features (V) computed within the transformer attention block serve as much more precise handles for manipulation. We show that V suppresses global-context channels, allowing it to retain high-entropy, disentangled local semantic information. Building on this discovery, we propose V-Attack, a novel method designed for precise local semantic attacks. V-Attack targets the value features and introduces two core components: (1) a Self-Value Enhancement module to refine V's intrinsic semantic richness, and (2) a Text-Guided Value Manipulation module that leverages text prompts to locate source concept and optimize it toward a target concept. By bypassing the entangled patch features, V-Attack achieves highly effective semantic control. Extensive experiments across diverse LVLMs, including LLaVA, InternVL, DeepseekVL and GPT-4o, show that V-Attack improves the attack success rate by an average of 36% over state-of-the-art methods, exposing critical vulnerabilities in modern visual-language understanding. Our code and data are available https://github.com/Summu77/V-Attack.
>
---
#### [new 156] Vision-Language Memory for Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文聚焦视频理解中的空间推理任务，针对视觉-语言模型在3D理解上的语义-几何错位及缺乏持久记忆的问题，提出VLM²模型。通过双记忆机制（工作记忆与情景记忆），实现基于2D视频的视图一致、3D感知的持续推理，显著提升长时程空间理解能力，在多个基准上达到当前最优性能。**

- **链接: [https://arxiv.org/pdf/2511.20644v1](https://arxiv.org/pdf/2511.20644v1)**

> **作者:** Zuntao Liu; Yi Du; Taimeng Fu; Shaoshu Su; Cherie Ho; Chen Wang
>
> **摘要:** Spatial reasoning is a critical capability for intelligent robots, yet current vision-language models (VLMs) still fall short of human-level performance in video-based spatial reasoning. This gap mainly stems from two challenges: a semantic-geometric misalignment that prevents consistent 3D understanding, and the absence of persistent memory to retain 3D representation and understanding over time. To address these limitations, we present VLM$^2$, a Vision-Language Model with persistent Memory for spatial reasoning with a view-consistent, 3D-aware representation purely from 2D video. Specifically, to enhance long-horizon reasoning, we incorporate a dual-memory module, consisting of a working memory that operates as a sliding window to focus on immediate context, and an episodic memory that consolidates and stores critical long-term information. This design enables efficient and long-horizon spatial reasoning with a fixed computational cost. Extensive experiments on multiple benchmarks show that VLM$^2$ achieves state-of-the-art performance among video-only models, significantly advancing the frontier of visual-spatial intelligence.
>
---
#### [new 157] HiCoGen: Hierarchical Compositional Text-to-Image Generation in Diffusion Models via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文针对扩散模型在复杂文本生成图像时概念遗漏与组合性差的问题，提出HiCoGen框架。通过LLM分解提示词，采用链式合成逐步生成图像，并引入基于强化学习的分层奖励机制与去衰减随机性调度，提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2511.19965v1](https://arxiv.org/pdf/2511.19965v1)**

> **作者:** Hongji Yang; Yucheng Zhou; Wencheng Han; Runzhou Tao; Zhongying Qiu; Jianfei Yang; Jianbing Shen
>
> **备注:** 9 pages
>
> **摘要:** Recent advances in diffusion models have demonstrated impressive capability in generating high-quality images for simple prompts. However, when confronted with complex prompts involving multiple objects and hierarchical structures, existing models struggle to accurately follow instructions, leading to issues such as concept omission, confusion, and poor compositionality. To address these limitations, we propose a Hierarchical Compositional Generative framework (HiCoGen) built upon a novel Chain of Synthesis (CoS) paradigm. Instead of monolithic generation, HiCoGen first leverages a Large Language Model (LLM) to decompose complex prompts into minimal semantic units. It then synthesizes these units iteratively, where the image generated in each step provides crucial visual context for the next, ensuring all textual concepts are faithfully constructed into the final scene. To further optimize this process, we introduce a reinforcement learning (RL) framework. Crucially, we identify that the limited exploration of standard diffusion samplers hinders effective RL. We theoretically prove that sample diversity is maximized by concentrating stochasticity in the early generation stages and, based on this insight, propose a novel Decaying Stochasticity Schedule to enhance exploration. Our RL algorithm is then guided by a hierarchical reward mechanism that jointly evaluates the image at the global, subject, and relationship levels. We also construct HiCoPrompt, a new text-to-image benchmark with hierarchical prompts for rigorous evaluation. Experiments show our approach significantly outperforms existing methods in both concept coverage and compositional accuracy.
>
---
#### [new 158] Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对数字孪生中的3D场景重建任务，解决传统LiDAR-camera融合方法依赖复杂校准、难以表征玻璃等材料的问题。提出纯相机方案：基于多视角图像用3D高斯溅射重建几何，结合视觉模型提取材质掩码，将材质标签投影至网格并赋予物理材质属性，实现高保真传感器模拟。**

- **链接: [https://arxiv.org/pdf/2511.20348v1](https://arxiv.org/pdf/2511.20348v1)**

> **作者:** João Malheiro Silva; Andy Huynh; Tong Duy Son; Holger Caesar
>
> **备注:** 8 pages, 5 figures. Submitted to IEEE Intelligent Vehicles Symposium (IV) 2026 for possible publication
>
> **摘要:** 3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
>
---
#### [new 159] Motion Marionette: Rethinking Rigid Motion Transfer via Prior Guidance
- **分类: cs.CV**

- **简介: 该论文提出Motion Marionette，解决单目视频到单视图图像的零样本刚性运动迁移问题。针对现有方法依赖外部先验导致泛化与时序一致性权衡的问题，提出基于空间-时间变换的内部先验，通过统一3D表示与速度场建模，实现高效、可控且视觉一致的视频生成。**

- **链接: [https://arxiv.org/pdf/2511.19909v1](https://arxiv.org/pdf/2511.19909v1)**

> **作者:** Haoxuan Wang; Jiachen Tao; Junyi Wu; Gaowen Liu; Ramana Rao Kompella; Yan Yan
>
> **摘要:** We present Motion Marionette, a zero-shot framework for rigid motion transfer from monocular source videos to single-view target images. Previous works typically employ geometric, generative, or simulation priors to guide the transfer process, but these external priors introduce auxiliary constraints that lead to trade-offs between generalizability and temporal consistency. To address these limitations, we propose guiding the motion transfer process through an internal prior that exclusively captures the spatial-temporal transformations and is shared between the source video and any transferred target video. Specifically, we first lift both the source video and the target image into a unified 3D representation space. Motion trajectories are then extracted from the source video to construct a spatial-temporal (SpaT) prior that is independent of object geometry and semantics, encoding relative spatial variations over time. This prior is further integrated with the target object to synthesize a controllable velocity field, which is subsequently refined using Position-Based Dynamics to mitigate artifacts and enhance visual coherence. The resulting velocity field can be flexibly employed for efficient video production. Empirical results demonstrate that Motion Marionette generalizes across diverse objects, produces temporally consistent videos that align well with the source motion, and supports controllable video generation.
>
---
#### [new 160] GazeProphetV2: Head-Movement-Based Gaze Prediction Enabling Efficient Foveated Rendering on Mobile VR
- **分类: cs.CV**

- **简介: 该论文提出GazeProphetV2，一种基于头动信息的VR注视预测方法，旨在提升移动VR中的渲染效率。通过融合注视历史、头动数据与场景内容，利用门控融合与跨模态注意力机制，实现更精准的未来帧注视预测，有效支持低硬件成本下的视点渲染优化。**

- **链接: [https://arxiv.org/pdf/2511.19988v1](https://arxiv.org/pdf/2511.19988v1)**

> **作者:** Farhaan Ebadulla; Chiraag Mudlpaur; Shreya Chaurasia; Gaurav BV
>
> **摘要:** Predicting gaze behavior in virtual reality environments remains a significant challenge with implications for rendering optimization and interface design. This paper introduces a multimodal approach to VR gaze prediction that combines temporal gaze patterns, head movement data, and visual scene information. By leveraging a gated fusion mechanism with cross-modal attention, the approach learns to adaptively weight gaze history, head movement, and scene content based on contextual relevance. Evaluations using a dataset spanning 22 VR scenes with 5.3M gaze samples demonstrate improvements in predictive accuracy when combining modalities compared to using individual data streams alone. The results indicate that integrating past gaze trajectories with head orientation and scene content enhances prediction accuracy across 1-3 future frames. Cross-scene generalization testing shows consistent performance with 93.1% validation accuracy and temporal consistency in predicted gaze trajectories. These findings contribute to understanding attention mechanisms in virtual environments while suggesting potential applications in rendering optimization, interaction design, and user experience evaluation. The approach represents a step toward more efficient virtual reality systems that can anticipate user attention patterns without requiring expensive eye tracking hardware.
>
---
#### [new 161] UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文针对视频扩散模型在长视频生成中难以外推的问题，提出UltraViCo方法。通过分析注意力分散机制，设计训练无关的常数衰减策略抑制超长序列注意力，同时解决重复与质量下降问题，显著提升外推能力至4倍，且可无缝适配多种下游任务。**

- **链接: [https://arxiv.org/pdf/2511.20123v1](https://arxiv.org/pdf/2511.20123v1)**

> **作者:** Min Zhao; Hongzhou Zhu; Yingze Wang; Bokai Yan; Jintao Zhang; Guande He; Ling Yang; Chongxuan Li; Jun Zhu
>
> **备注:** Project page: https://thu-ml.github.io/UltraViCo.github.io/
>
> **摘要:** Despite advances, video diffusion transformers still struggle to generalize beyond their training length, a challenge we term video length extrapolation. We identify two failure modes: model-specific periodic content repetition and a universal quality degradation. Prior works attempt to solve repetition via positional encodings, overlooking quality degradation and achieving only limited extrapolation. In this paper, we revisit this challenge from a more fundamental view: attention maps, which directly govern how context influences outputs. We identify that both failure modes arise from a unified cause: attention dispersion, where tokens beyond the training window dilute learned attention patterns. This leads to quality degradation and repetition emerges as a special case when this dispersion becomes structured into periodic attention patterns, induced by harmonic properties of positional encodings. Building on this insight, we propose UltraViCo, a training-free, plug-and-play method that suppresses attention for tokens beyond the training window via a constant decay factor. By jointly addressing both failure modes, we outperform a broad set of baselines largely across models and extrapolation ratios, pushing the extrapolation limit from 2x to 4x. Remarkably, it improves Dynamic Degree and Imaging Quality by 233% and 40.5% over the previous best method at 4x extrapolation. Furthermore, our method generalizes seamlessly to downstream tasks such as controllable video synthesis and editing.
>
---
#### [new 162] SAM-MI: A Mask-Injected Framework for Enhancing Open-Vocabulary Semantic Segmentation with SAM
- **分类: cs.CV**

- **简介: 该论文针对开放词汇语义分割（OVSS）中SAM模型过分割及掩码与标签硬结合的问题，提出SAM-MI框架。通过文本引导稀疏点提示加速掩码生成，浅层掩码聚合缓解过分割，解耦掩码注入分别在高低频层面融合掩码指导，显著提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2511.20027v1](https://arxiv.org/pdf/2511.20027v1)**

> **作者:** Lin Chen; Yingjian Zhu; Qi Yang; Xin Niu; Kun Ding; Shiming Xiang
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) aims to segment and recognize objects universally. Trained on extensive high-quality segmentation data, the segment anything model (SAM) has demonstrated remarkable universal segmentation capabilities, offering valuable support for OVSS. Although previous methods have made progress in leveraging SAM for OVSS, there are still some challenges: (1) SAM's tendency to over-segment and (2) hard combinations between fixed masks and labels. This paper introduces a novel mask-injected framework, SAM-MI, which effectively integrates SAM with OVSS models to address these challenges. Initially, SAM-MI employs a Text-guided Sparse Point Prompter to sample sparse prompts for SAM instead of previous dense grid-like prompts, thus significantly accelerating the mask generation process. The framework then introduces Shallow Mask Aggregation (SMAgg) to merge partial masks to mitigate the SAM's over-segmentation issue. Finally, Decoupled Mask Injection (DMI) incorporates SAM-generated masks for guidance at low-frequency and high-frequency separately, rather than directly combining them with labels. Extensive experiments on multiple benchmarks validate the superiority of SAM-MI. Notably, the proposed method achieves a 16.7% relative improvement in mIoU over Grounded-SAM on the MESS benchmark, along with a 1.6$\times$ speedup. We hope SAM-MI can serve as an alternative methodology to effectively equip the OVSS model with SAM.
>
---
#### [new 163] 4DWorldBench: A Comprehensive Evaluation Framework for 3D/4D World Generation Models
- **分类: cs.CV**

- **简介: 该论文提出4DWorldBench，针对3D/4D世界生成模型缺乏统一评估的问题，构建涵盖感知质量、条件对齐、物理真实性和时空一致性的多维评估框架，融合多模态条件与LLM/MLLM评判，实现更全面、客观的模型评价，推动从“视觉生成”向“世界生成”的演进。**

- **链接: [https://arxiv.org/pdf/2511.19836v1](https://arxiv.org/pdf/2511.19836v1)**

> **作者:** Yiting Lu; Wei Luo; Peiyan Tu; Haoran Li; Hanxin Zhu; Zihao Yu; Xingrui Wang; Xinyi Chen; Xinge Peng; Xin Li; Zhibo Chen
>
> **摘要:** World Generation Models are emerging as a cornerstone of next-generation multimodal intelligence systems. Unlike traditional 2D visual generation, World Models aim to construct realistic, dynamic, and physically consistent 3D/4D worlds from images, videos, or text. These models not only need to produce high-fidelity visual content but also maintain coherence across space, time, physics, and instruction control, enabling applications in virtual reality, autonomous driving, embodied intelligence, and content creation. However, prior benchmarks emphasize different evaluation dimensions and lack a unified assessment of world-realism capability. To systematically evaluate World Models, we introduce the 4DWorldBench, which measures models across four key dimensions: Perceptual Quality, Condition-4D Alignment, Physical Realism, and 4D Consistency. The benchmark covers tasks such as Image-to-3D/4D, Video-to-4D, Text-to-3D/4D. Beyond these, we innovatively introduce adaptive conditioning across multiple modalities, which not only integrates but also extends traditional evaluation paradigms. To accommodate different modality-conditioned inputs, we map all modality conditions into a unified textual space during evaluation, and further integrate LLM-as-judge, MLLM-as-judge, and traditional network-based methods. This unified and adaptive design enables more comprehensive and consistent evaluation of alignment, physical realism, and cross-modal coherence. Preliminary human studies further demonstrate that our adaptive tool selection achieves closer agreement with subjective human judgments. We hope this benchmark will serve as a foundation for objective comparisons and improvements, accelerating the transition from "visual generation" to "world generation." Our project can be found at https://yeppp27.github.io/4DWorldBench.github.io/.
>
---
#### [new 164] Multiscale Vector-Quantized Variational Autoencoder for Endoscopic Image Synthesis
- **分类: cs.CV**

- **简介: 该论文针对无线胶囊内镜图像数据稀缺问题，提出多尺度向量量化变分自编码器（MSVQ-VAE），用于生成包含各类异常的合成内镜图像。通过条件生成实现异常类型可控，提升数据多样性，有效支持临床决策系统训练，实验表明生成数据可媲美真实数据性能。**

- **链接: [https://arxiv.org/pdf/2511.19578v1](https://arxiv.org/pdf/2511.19578v1)**

> **作者:** Dimitrios E. Diamantis; Dimitris K. Iakovidis
>
> **备注:** Accepted in IEEE Int. Conf. Imaging Systems and Techniques (IST 2025), Strasburg, France
>
> **摘要:** Gastrointestinal (GI) imaging via Wireless Capsule Endoscopy (WCE) generates a large number of images requiring manual screening. Deep learning-based Clinical Decision Support (CDS) systems can assist screening, yet their performance relies on the existence of large, diverse, training medical datasets. However, the scarcity of such data, due to privacy constraints and annotation costs, hinders CDS development. Generative machine learning offers a viable solution to combat this limitation. While current Synthetic Data Generation (SDG) methods, such as Generative Adversarial Networks and Variational Autoencoders have been explored, they often face challenges with training stability and capturing sufficient visual diversity, especially when synthesizing abnormal findings. This work introduces a novel VAE-based methodology for medical image synthesis and presents its application for the generation of WCE images. The novel contributions of this work include a) multiscale extension of the Vector Quantized VAE model, named as Multiscale Vector Quantized Variational Autoencoder (MSVQ-VAE); b) unlike other VAE-based SDG models for WCE image generation, MSVQ-VAE is used to seamlessly introduce abnormalities into normal WCE images; c) it enables conditional generation of synthetic images, enabling the introduction of different types of abnormalities into the normal WCE images; d) it performs experiments with a variety of abnormality types, including polyps, vascular and inflammatory conditions. The utility of the generated images for CDS is assessed via image classification. Comparative experiments demonstrate that training a CDS classifier using the abnormal images generated by the proposed methodology yield comparable results with a classifier trained with only real data. The generality of the proposed methodology promises its applicability to various domains related to medical multimedia.
>
---
#### [new 165] CrossEarth-Gate: Fisher-Guided Adaptive Tuning Engine for Efficient Adaptation of Cross-Domain Remote Sensing Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对遥感图像语义分割中的跨域适应问题，提出CrossEarth-Gate方法。通过构建空间、语义、频率多维度模块工具箱，并引入Fisher信息引导的自适应选择机制，动态激活关键模块，提升模型在复杂域偏移下的适应效率与性能，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20302v1](https://arxiv.org/pdf/2511.20302v1)**

> **作者:** Shilei Cao; Ziyang Gong; Hehai Lin; Yang Liu; Jiashun Cheng; Xiaoxing Hu; Haoyuan Liang; Guowen Li; Chengwei Qin; Hong Cheng; Xue Yang; Juepeng Zheng; Haohuan Fu
>
> **摘要:** In Remote Sensing (RS), Parameter-Efficient Fine-Tuning (PEFT) has emerged as a key approach to activate the generalizable representation ability of foundation models for downstream tasks. However, existing specialized PEFT methods often fail when applied to large-scale Earth observation tasks, as they are unable to fully handle the multifaceted and unpredictable domain gaps (\eg, spatial, semantic, and frequency shifts) inherent in RS data. To overcome this, we propose CrossEarth-Gate, which introduces two primary contributions. First, we establish a comprehensive RS module toolbox to address multifaceted domain gaps, comprising spatial, semantic, and frequency modules. Second, we develop a Fisher-guided adaptive selection mechanism that operates on this toolbox. This selection is guided by Fisher Information to quantify each module's importance by measuring its contribution to the task-specific gradient flow. It dynamically activates only the most critical modules at the appropriate layers, guiding the gradient flow to maximize adaptation effectiveness and efficiency. Comprehensive experiments validate the efficacy and generalizability of our method, where CrossEarth-Gate achieves state-of-the-art performance across 16 cross-domain benchmarks for RS semantic segmentation. The code of the work will be released.
>
---
#### [new 166] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对开放世界具身智能中闭环评估难的问题，提出Wanderland框架，通过多传感器捕获与几何精确重建，实现高保真仿真。解决了视觉与几何模拟真实差距大、图像单模态方法性能差等问题，构建了可信赖的导航评估与3D重建基准。**

- **链接: [https://arxiv.org/pdf/2511.20620v1](https://arxiv.org/pdf/2511.20620v1)**

> **作者:** Xinhao Liu; Jiaqi Li; Youming Deng; Ruxin Chen; Yingjia Zhang; Yifei Ma; Li Guo; Yiming Li; Jing Zhang; Chen Feng
>
> **摘要:** Reproducible closed-loop evaluation remains a major bottleneck in Embodied AI such as visual navigation. A promising path forward is high-fidelity simulation that combines photorealistic sensor rendering with geometrically grounded interaction in complex, open-world urban environments. Although recent video-3DGS methods ease open-world scene capturing, they are still unsuitable for benchmarking due to large visual and geometric sim-to-real gaps. To address these challenges, we introduce Wanderland, a real-to-sim framework that features multi-sensor capture, reliable reconstruction, accurate geometry, and robust view synthesis. Using this pipeline, we curate a diverse dataset of indoor-outdoor urban scenes and systematically demonstrate how image-only pipelines scale poorly, how geometry quality impacts novel view synthesis, and how all of these adversely affect navigation policy learning and evaluation reliability. Beyond serving as a trusted testbed for embodied navigation, Wanderland's rich raw sensor data further allows benchmarking of 3D reconstruction and novel view synthesis models. Our work establishes a new foundation for reproducible research in open-world embodied AI. Project website is at https://ai4ce.github.io/wanderland/.
>
---
#### [new 167] AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend
- **分类: cs.CV**

- **简介: 该论文提出AMB3R，一种用于密集3D重建的前馈模型，通过紧凑的体素场景表示实现度量尺度下的高精度重建。解决了多视图重建中几何推理效率与精度难题，无需微调即可拓展至视觉里程计和大规模SfM，性能超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20343v1](https://arxiv.org/pdf/2511.20343v1)**

> **作者:** Hengyi Wang; Lourdes Agapito
>
> **备注:** Project page: https://hengyiwang.github.io/projects/amber
>
> **摘要:** We present AMB3R, a multi-view feed-forward model for dense 3D reconstruction on a metric-scale that addresses diverse 3D vision tasks. The key idea is to leverage a sparse, yet compact, volumetric scene representation as our backend, enabling geometric reasoning with spatial compactness. Although trained solely for multi-view reconstruction, we demonstrate that AMB3R can be seamlessly extended to uncalibrated visual odometry (online) or large-scale structure from motion without the need for task-specific fine-tuning or test-time optimization. Compared to prior pointmap-based models, our approach achieves state-of-the-art performance in camera pose, depth, and metric-scale estimation, 3D reconstruction, and even surpasses optimization-based SLAM and SfM methods with dense reconstruction priors on common benchmarks.
>
---
#### [new 168] Towards Efficient VLMs: Information-Theoretic Driven Compression via Adaptive Structural Pruning
- **分类: cs.CV; cs.AI; cs.IT; cs.LG**

- **简介: 该论文针对视觉语言模型（VLMs）部署效率低的问题，提出基于信息论的自适应结构压缩方法InfoPrune。通过引入熵基有效秩与KS距离，量化注意力头重要性，实现兼顾信息保留与结构稀疏性的剪枝。结合训练与免训练压缩策略，在多个数据集上显著降低计算量，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2511.19518v1](https://arxiv.org/pdf/2511.19518v1)**

> **作者:** Zhaoqi Xu; Yingying Zhang; Jian Li; Jianwei Guo; Qiannan Zhu; Hua Huang
>
> **摘要:** Recent advances in vision-language models (VLMs) have shown remarkable performance across multimodal tasks, yet their ever-growing scale poses severe challenges for deployment and efficiency. Existing compression methods often rely on heuristic importance metrics or empirical pruning rules, lacking theoretical guarantees about information preservation. In this work, we propose InfoPrune, an information-theoretic framework for adaptive structural compression of VLMs. Grounded in the Information Bottleneck principle, we formulate pruning as a trade-off between retaining task-relevant semantics and discarding redundant dependencies. To quantify the contribution of each attention head, we introduce an entropy-based effective rank (eRank) and employ the Kolmogorov--Smirnov (KS) distance to measure the divergence between original and compressed structures. This yields a unified criterion that jointly considers structural sparsity and informational efficiency. Building on this foundation, we further design two complementary schemes: (1) a training-based head pruning guided by the proposed information loss objective, and (2) a training-free FFN compression via adaptive low-rank approximation. Extensive experiments on VQAv2, TextVQA, and GQA demonstrate that InfoPrune achieves up to 3.2x FLOP reduction and 1.8x acceleration with negligible performance degradation, establishing a theoretically grounded and practically effective step toward efficient multimodal large models.
>
---
#### [new 169] Unleashing the Power of Vision-Language Models for Long-Tailed Multi-Label Visual Recognition
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对长尾多标签图像识别任务，解决类别不平衡导致模型偏袒头部类、尾部类表现差的问题。提出CAPNET框架，通过视觉-语言模型的文本编码器显式建模标签相关性，结合图卷积与可学习提示词，优化训练与推理，提升尾部类识别性能。**

- **链接: [https://arxiv.org/pdf/2511.20641v1](https://arxiv.org/pdf/2511.20641v1)**

> **作者:** Wei Tang; Zuo-Zheng Wang; Kun Zhang; Tong Wei; Min-Ling Zhang
>
> **摘要:** Long-tailed multi-label visual recognition poses a significant challenge, as images typically contain multiple labels with highly imbalanced class distributions, leading to biased models that favor head classes while underperforming on tail classes. Recent efforts have leveraged pre-trained vision-language models, such as CLIP, alongside long-tailed learning techniques to exploit rich visual-textual priors for improved performance. However, existing methods often derive semantic inter-class relationships directly from imbalanced datasets, resulting in unreliable correlations for tail classes due to data scarcity. Moreover, CLIP's zero-shot paradigm is optimized for single-label image-text matching, making it suboptimal for multi-label tasks. To address these issues, we propose the correlation adaptation prompt network (CAPNET), a novel end-to-end framework that explicitly models label correlations from CLIP's textual encoder. The framework incorporates a graph convolutional network for label-aware propagation and learnable soft prompts for refined embeddings. It utilizes a distribution-balanced Focal loss with class-aware re-weighting for optimized training under imbalance. Moreover, it improves generalization through test-time ensembling and realigns visual-textual modalities using parameter-efficient fine-tuning to avert overfitting on tail classes without compromising head class performance. Extensive experiments and ablation studies on benchmarks including VOC-LT, COCO-LT, and NUS-WIDE demonstrate that CAPNET achieves substantial improvements over state-of-the-art methods, validating its effectiveness for real-world long-tailed multi-label visual recognition.
>
---
#### [new 170] Alzheimers Disease Progression Prediction Based on Manifold Mapping of Irregularly Sampled Longitudinal Data
- **分类: cs.CV**

- **简介: 该论文针对阿尔茨海默病（AD）进展预测任务，解决不规则采样纵向sMRI数据建模难题。提出R-TNAG框架，通过流形映射保留数据几何结构，结合时序感知神经微分方程与注意力门控机制，实现对不规则时间间隔数据的连续动态建模，提升预测准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.20154v1](https://arxiv.org/pdf/2511.20154v1)**

> **作者:** Xin Hong; Ying Shi; Yinhao Li; Yen-Wei Chen
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** The uncertainty of clinical examinations frequently leads to irregular observation intervals in longitudinal imaging data, posing challenges for modeling disease progression.Most existing imaging-based disease prediction models operate in Euclidean space, which assumes a flat representation of data and fails to fully capture the intrinsic continuity and nonlinear geometric structure of irregularly sampled longitudinal images. To address the challenge of modeling Alzheimers disease (AD) progression from irregularly sampled longitudinal structural Magnetic Resonance Imaging (sMRI) data, we propose a Riemannian manifold mapping, a Time-aware manifold Neural ordinary differential equation, and an Attention-based riemannian Gated recurrent unit (R-TNAG) framework. Our approach first projects features extracted from high-dimensional sMRI into a manifold space to preserve the intrinsic geometry of disease progression. On this representation, a time-aware Neural Ordinary Differential Equation (TNODE) models the continuous evolution of latent states between observations, while an Attention-based Riemannian Gated Recurrent Unit (ARGRU) adaptively integrates historical and current information to handle irregular intervals. This joint design improves temporal consistency and yields robust AD trajectory prediction under irregular sampling.Experimental results demonstrate that the proposed method consistently outperforms state-of-the-art models in both disease status prediction and cognitive score regression. Ablation studies verify the contributions of each module, highlighting their complementary roles in enhancing predictive accuracy. Moreover, the model exhibits stable performance across varying sequence lengths and missing data rates, indicating strong temporal generalizability. Cross-dataset validation further confirms its robustness and applicability in diverse clinical settings.
>
---
#### [new 171] Evaluating the Performance of Deep Learning Models in Whole-body Dynamic 3D Posture Prediction During Load-reaching Activities
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对动态负重搬运中全身三维姿态预测任务，提出基于Transformer和BLSTM的时序模型，利用手部位置、动作类型等输入预测后续75%运动过程。创新性引入约束肢体长度的新损失函数，显著降低误差，其中Transformer模型表现最优，较BLSTM提升58%精度。**

- **链接: [https://arxiv.org/pdf/2511.20615v1](https://arxiv.org/pdf/2511.20615v1)**

> **作者:** Seyede Niloofar Hosseini; Ali Mojibi; Mahdi Mohseni; Navid Arjmand; Alireza Taheri
>
> **备注:** 10 pages, 6 figures, 7 tables
>
> **摘要:** This study aimed to explore the application of deep neural networks for whole-body human posture prediction during dynamic load-reaching activities. Two time-series models were trained using bidirectional long short-term memory (BLSTM) and transformer architectures. The dataset consisted of 3D full-body plug-in gait dynamic coordinates from 20 normal-weight healthy male individuals each performing 204 load-reaching tasks from different load positions while adapting various lifting and handling techniques. The model inputs consisted of the 3D position of the hand-load position, lifting (stoop, full-squat and semi-squat) and handling (one- and two-handed) techniques, body weight and height, and the 3D coordinate data of the body posture from the first 25% of the task duration. These inputs were used by the models to predict body coordinates during the remaining 75% of the task period. Moreover, a novel method was proposed to improve the accuracy of the previous and present posture prediction networks by enforcing constant body segment lengths through the optimization of a new cost function. The results indicated that the new cost function decreased the prediction error of the models by approximately 8% and 21% for the arm and leg models, respectively. We indicated that utilizing the transformer architecture, with a root-mean-square-error of 47.0 mm, exhibited ~58% more accurate long-term performance than the BLSTM-based model. This study merits the use of neural networks that capture time series dependencies in 3D motion frames, providing a unique approach for understanding and predict motion dynamics during manual material handling activities.
>
---
#### [new 172] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLMs）中的幻觉问题，提出无需训练的Vision-Guided Attention（VGA）方法。通过利用视觉令牌语义构建精准视觉定位，引导模型关注相关视觉区域，并在生成时动态抑制已描述区域，有效减少幻觉。方法低延迟、兼容高效注意力机制，在多个基准上实现先进去幻觉性能。**

- **链接: [https://arxiv.org/pdf/2511.20032v1](https://arxiv.org/pdf/2511.20032v1)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng; Zhixing Tan
>
> **备注:** Under Review
>
> **摘要:** Visual attention serves as the primary mechanism through which MLLMs interpret visual information; however, its limited localization capability often leads to hallucinations. We observe that although MLLMs can accurately extract visual semantics from visual tokens, they fail to fully leverage this advantage during subsequent inference. To address this limitation, we propose Vision-Guided Attention (VGA), a training-free method that first constructs precise visual grounding by exploiting the semantic content of visual tokens, and then uses this grounding to guide the model's focus toward relevant visual regions. In image captioning, VGA further refines this guidance dynamically during generation by suppressing regions that have already been described. In VGA, each token undergoes only a single forward pass, introducing a negligible latency overhead of just 4.36\%. In addition, VGA is fully compatible with efficient attention implementations such as FlashAttention. Extensive experiments across diverse MLLMs and multiple hallucination benchmarks demonstrate that VGA achieves state-of-the-art dehallucination performance. Further analysis confirms that explicit visual guidance plays a crucial role in enhancing the visual understanding capabilities of MLLMs.
>
---
#### [new 173] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对视觉语言模型在细粒度图像理解任务中的性能瓶颈，提出CropVLM方法。通过强化学习实现动态“局部放大”，无需标注框即可提升模型对细节的感知能力。该方法可通用适配各类VLM，显著改善高分辨率理解任务表现，尤其在域外场景下有效，且不需微调原模型。**

- **链接: [https://arxiv.org/pdf/2511.19820v1](https://arxiv.org/pdf/2511.19820v1)**

> **作者:** Miguel Carvalho; Helder Dias; Bruno Martins
>
> **摘要:** Vision-Language Models (VLMs) often struggle with tasks that require fine-grained image understanding, such as scene-text recognition or document analysis, due to perception limitations and visual fragmentation. To address these challenges, we introduce CropVLM as an external low-cost method for boosting performance, enabling VLMs to dynamically ''zoom in'' on relevant image regions, enhancing their ability to capture fine details. CropVLM is trained using reinforcement learning, without using human-labeled bounding boxes as a supervision signal, and without expensive synthetic evaluations. The model is trained once and can be paired with both open-source and proprietary VLMs to improve their performance. Our approach delivers significant improvements on tasks that require high-resolution image understanding, notably for benchmarks that are out-of-domain for the target VLM, without modifying or fine-tuning the VLM, thus avoiding catastrophic forgetting.
>
---
#### [new 174] ShelfRectNet: Single View Shelf Image Rectification with Homography Estimation
- **分类: cs.CV**

- **简介: 该论文针对单视角货架图像校正任务，解决从任意角度拍摄的货架图像因透视畸变导致的识别困难问题。提出ShelfRectNet框架，基于ConvNeXt的深度学习模型，通过4点参数化单应性矩阵估计实现图像校正，并引入合成单应性增强数据以提升泛化能力，显著提高校正精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.20335v1](https://arxiv.org/pdf/2511.20335v1)**

> **作者:** Onur Berk Tore; Ibrahim Samil Yalciner; Server Calap
>
> **摘要:** Estimating homography from a single image remains a challenging yet practically valuable task, particularly in domains like retail, where only one viewpoint is typically available for shelf monitoring and product alignment. In this paper, we present a deep learning framework that predicts a 4-point parameterized homography matrix to rectify shelf images captured from arbitrary angles. Our model leverages a ConvNeXt-based backbone for enhanced feature representation and adopts normalized coordinate regression for improved stability. To address data scarcity and promote generalization, we introduce a novel augmentation strategy by modeling and sampling synthetic homographies. Our method achieves a mean corner error of 1.298 pixels on the test set. When compared with both classical computer vision and deep learning-based approaches, our method demonstrates competitive performance in both accuracy and inference speed. Together, these results establish our approach as a robust and efficient solution for realworld single-view rectification. To encourage further research in this domain, we will make our dataset, ShelfRectSet, and code publicly available
>
---
#### [new 175] Face, Whole-Person, and Object Classification in a Unified Space Via The Interleaved Multi-Domain Identity Curriculum
- **分类: cs.CV**

- **简介: 该论文提出一种统一嵌入空间的多任务识别方法（IMIC），解决视觉基础模型在多任务微调中灾难性遗忘问题。通过交错训练策略，实现人脸、人体与物体的联合分类，在不损害分布外泛化能力的前提下，显著提升多任务性能。**

- **链接: [https://arxiv.org/pdf/2511.19846v1](https://arxiv.org/pdf/2511.19846v1)**

> **作者:** Thomas M Metz; Matthew Q Hill; Alice J O'Toole
>
> **摘要:** Vision foundation models can perform generalized object classification in zero-shot mode, and face/person recognition when they are fine-tuned. However, fine-tuned models suffer from catastrophic forgetting. We create models that perform four tasks (object recognition, face recognition from high- and low-quality images, and person recognition from whole-body images) in a single embedding space -- without incurring substantial catastrophic forgetting. To accomplish this, we introduce two variants of the Interleaved Multi-Domain Identity Curriculum (IMIC): a gradient-coupled, interleaving training schedule that fine-tunes a foundation backbone simultaneously on all four tasks. The IMIC method proved effective with three foundation model bases: DINOv3, CLIP, and EVA-02. Two of these (EVA-02 and CLIP) performed comparably with domain experts on all four tasks concurrently and were more accurate than humans at multi-tasking across face, body, and object datasets. Further, we demonstrate that our approach does not substantially harm out-of-distribution generalization, thus maintaining a key property of foundation models. Analysis of the most accurate model variants (EVA-02 + IMIC A and B) showed linearly separable representations of the four tasks in the unified embedding space, but with substantial sharing of features across tasks. Fewer than 100 PCs calculated from any one task could perform all other tasks with nearly zero performance degradation.
>
---
#### [new 176] Think First, Assign Next (ThiFAN-VQA): A Two-stage Chain-of-Thought Framework for Post-Disaster Damage Assessment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对灾后损伤评估中的视觉问答任务，解决现有方法在数据稀缺、答案空间固定及生成幻觉等问题。提出两阶段框架ThiFAN-VQA，结合链式思维与信息检索，提升推理可解释性与答案准确性，实现零样本下的高效、可靠损伤评估。**

- **链接: [https://arxiv.org/pdf/2511.19557v1](https://arxiv.org/pdf/2511.19557v1)**

> **作者:** Ehsan Karimi; Nhut Le; Maryam Rahnemoonfar
>
> **摘要:** Timely and accurate assessment of damages following natural disasters is essential for effective emergency response and recovery. Recent AI-based frameworks have been developed to analyze large volumes of aerial imagery collected by Unmanned Aerial Vehicles, providing actionable insights rapidly. However, creating and annotating data for training these models is costly and time-consuming, resulting in datasets that are limited in size and diversity. Furthermore, most existing approaches rely on traditional classification-based frameworks with fixed answer spaces, restricting their ability to provide new information without additional data collection or model retraining. Using pre-trained generative models built on in-context learning (ICL) allows for flexible and open-ended answer spaces. However, these models often generate hallucinated outputs or produce generic responses that lack domain-specific relevance. To address these limitations, we propose ThiFAN-VQA, a two-stage reasoning-based framework for visual question answering (VQA) in disaster scenarios. ThiFAN-VQA first generates structured reasoning traces using chain-of-thought (CoT) prompting and ICL to enable interpretable reasoning under limited supervision. A subsequent answer selection module evaluates the generated responses and assigns the most coherent and contextually accurate answer, effectively improve the model performance. By integrating a custom information retrieval system, domain-specific prompting, and reasoning-guided answer selection, ThiFAN-VQA bridges the gap between zero-shot and supervised methods, combining flexibility with consistency. Experiments on FloodNet and RescueNet-VQA, UAV-based datasets from flood- and hurricane-affected regions, demonstrate that ThiFAN-VQA achieves superior accuracy, interpretability, and adaptability for real-world post-disaster damage assessment tasks.
>
---
#### [new 177] Advancing Image Classification with Discrete Diffusion Classification Modeling
- **分类: cs.CV**

- **简介: 该论文针对高不确定性下图像分类性能下降的问题，提出离散扩散分类建模（DiDiCM）框架。通过扩散过程建模输入图像条件下的类别后验分布，支持概率或离散标签的迭代预测，显著提升在噪声图像和小样本场景下的分类准确率。**

- **链接: [https://arxiv.org/pdf/2511.20263v1](https://arxiv.org/pdf/2511.20263v1)**

> **作者:** Omer Belhasin; Shelly Golan; Ran El-Yaniv; Michael Elad
>
> **摘要:** Image classification is a well-studied task in computer vision, and yet it remains challenging under high-uncertainty conditions, such as when input images are corrupted or training data are limited. Conventional classification approaches typically train models to directly predict class labels from input images, but this might lead to suboptimal performance in such scenarios. To address this issue, we propose Discrete Diffusion Classification Modeling (DiDiCM), a novel framework that leverages a diffusion-based procedure to model the posterior distribution of class labels conditioned on the input image. DiDiCM supports diffusion-based predictions either on class probabilities or on discrete class labels, providing flexibility in computation and memory trade-offs. We conduct a comprehensive empirical study demonstrating the superior performance of DiDiCM over standard classifiers, showing that a few diffusion iterations achieve higher classification accuracy on the ImageNet dataset compared to baselines, with accuracy gains increasing as the task becomes more challenging. We release our code at https://github.com/omerb01/didicm .
>
---
#### [new 178] MHB: Multimodal Handshape-aware Boundary Detection for Continuous Sign Language Recognition
- **分类: cs.CV**

- **简介: 该论文针对连续手语识别中的符号边界检测问题，提出多模态方法。利用3D骨骼特征和预训练的手形分类器，融合动态与形态信息，提升边界检测精度，并用于后续手语识别，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.19907v1](https://arxiv.org/pdf/2511.19907v1)**

> **作者:** Mingyu Zhao; Zhanfu Yang; Yang Zhou; Zhaoyang Xia; Can Jin; Xiaoxiao He; Carol Neidle; Dimitris N. Metaxas
>
> **摘要:** This paper presents a multimodal approach for continuous sign recognition that first uses machine learning to detect the start and end frames of signs in videos of American Sign Language (ASL) sentences, and then recognizes the segmented signs. For improved robustness, we use 3D skeletal features extracted from sign language videos to capture the convergence of sign properties and their dynamics, which tend to cluster at sign boundaries. Another focus of this work is the incorporation of information from 3D handshape for boundary detection. To detect handshapes normally expected at the beginning and end of signs, we pretrain a handshape classifier for 87 linguistically defined canonical handshape categories using a dataset that we created by integrating and normalizing several existing datasets. A multimodal fusion module is then used to unify the pretrained sign video segmentation framework and the handshape classification models. Finally, the estimated boundaries are used for sign recognition, where the recognition model is trained on a large database containing both citation-form isolated signs and signs pre-segmented (based on manual annotations) from continuous signing, as such signs often differ in certain respects. We evaluate our method on the ASLLRP corpus and demonstrate significant improvements over previous work.
>
---
#### [new 179] Lightweight Transformer Framework for Weakly Supervised Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对弱监督语义分割任务，解决噪声标签下边界模糊、小物体漏检问题。提出轻量级CrispFormer框架，通过边界分支、不确定性引导精炼器和动态多尺度融合，提升分割精度与鲁棒性，无需修改主干网络或复杂后处理。**

- **链接: [https://arxiv.org/pdf/2511.19765v1](https://arxiv.org/pdf/2511.19765v1)**

> **作者:** Ali Torabi; Sanjog Gaihre; Yaqoob Majeed
>
> **摘要:** Weakly supervised semantic segmentation (WSSS) must learn dense masks from noisy, under-specified cues. We revisit the SegFormer decoder and show that three small, synergistic changes make weak supervision markedly more effective-without altering the MiT backbone or relying on heavy post-processing. Our method, CrispFormer, augments the decoder with: (1) a boundary branch that supervises thin object contours using a lightweight edge head and a boundary-aware loss; (2) an uncertainty-guided refiner that predicts per-pixel aleatoric uncertainty and uses it to weight losses and gate a residual correction of the segmentation logits; and (3) a dynamic multi-scale fusion layer that replaces static concatenation with spatial softmax gating over multi-resolution features, optionally modulated by uncertainty. The result is a single-pass model that preserves crisp boundaries, selects appropriate scales per location, and resists label noise from weak cues. Integrated into a standard WSSS pipeline (seed, student, and EMA relabeling), CrispFormer consistently improves boundary F-score, small-object recall, and mIoU over SegFormer baselines trained on the same seeds, while adding minimal compute. Our decoder-centric formulation is simple to implement, broadly compatible with existing SegFormer variants, and offers a reproducible path to higher-fidelity masks from image-level supervision.
>
---
#### [new 180] The Consistency Critic: Correcting Inconsistencies in Generated Images via Reference-Guided Attentive Alignment
- **分类: cs.CV**

- **简介: 该论文针对图像生成中细粒度细节不一致问题，提出参考引导的后处理方法ImageCritic。通过构建三元组数据集，设计注意力对齐损失与细节编码器，实现多轮局部修正。可集成至智能体框架，自动检测并修复生成图像中的不一致性，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.20614v1](https://arxiv.org/pdf/2511.20614v1)**

> **作者:** Ziheng Ouyang; Yiren Song; Yaoli Liu; Shihao Zhu; Qibin Hou; Ming-Ming Cheng; Mike Zheng Shou
>
> **备注:** Project page: https://ouyangziheng.github.io/ImageCritic-Page/
>
> **摘要:** Previous works have explored various customized generation tasks given a reference image, but they still face limitations in generating consistent fine-grained details. In this paper, our aim is to solve the inconsistency problem of generated images by applying a reference-guided post-editing approach and present our ImageCritic. We first construct a dataset of reference-degraded-target triplets obtained via VLM-based selection and explicit degradation, which effectively simulates the common inaccuracies or inconsistencies observed in existing generation models. Furthermore, building on a thorough examination of the model's attention mechanisms and intrinsic representations, we accordingly devise an attention alignment loss and a detail encoder to precisely rectify inconsistencies. ImageCritic can be integrated into an agent framework to automatically detect inconsistencies and correct them with multi-round and local editing in complex scenarios. Extensive experiments demonstrate that ImageCritic can effectively resolve detail-related issues in various customized generation scenarios, providing significant improvements over existing methods.
>
---
#### [new 181] PromptMoG: Enhancing Diversity in Long-Prompt Image Generation via Prompt Embedding Mixture-of-Gaussian Sampling
- **分类: cs.CV**

- **简介: 该论文针对长提示图像生成中多样性下降的问题，提出PromptMoG方法。通过在提示嵌入空间采用高斯混合采样，提升生成多样性并保持语义一致性。研究构建了LPD-Bench基准，验证了现有模型在长提示下多样性显著降低，并提出无需训练的改进方案，有效缓解了保真度与多样性之间的矛盾。**

- **链接: [https://arxiv.org/pdf/2511.20251v1](https://arxiv.org/pdf/2511.20251v1)**

> **作者:** Bo-Kai Ruan; Teng-Fang Hsiao; Ling Lo; Yi-Lun Wu; Hong-Han Shuai
>
> **备注:** Technical Report
>
> **摘要:** Recent advances in text-to-image (T2I) generation have achieved remarkable visual outcomes through large-scale rectified flow models. However, how these models behave under long prompts remains underexplored. Long prompts encode rich content, spatial, and stylistic information that enhances fidelity but often suppresses diversity, leading to repetitive and less creative outputs. In this work, we systematically study this fidelity-diversity dilemma and reveal that state-of-the-art models exhibit a clear drop in diversity as prompt length increases. To enable consistent evaluation, we introduce LPD-Bench, a benchmark designed for assessing both fidelity and diversity in long-prompt generation. Building on our analysis, we develop a theoretical framework that increases sampling entropy through prompt reformulation and propose a training-free method, PromptMoG, which samples prompt embeddings from a Mixture-of-Gaussians in the embedding space to enhance diversity while preserving semantics. Extensive experiments on four state-of-the-art models, SD3.5-Large, Flux.1-Krea-Dev, CogView4, and Qwen-Image, demonstrate that PromptMoG consistently improves long-prompt generation diversity without semantic drifting.
>
---
#### [new 182] XiCAD: Camera Activation Detection in the Da Vinci Xi User Interface
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对达芬奇Xi手术系统中摄像头激活状态检测任务，旨在从手术视频中自动识别摄像头图块位置及其激活状态。通过基于ResNet18的轻量级模型，利用标注数据训练并验证，实现高精度实时检测，为手术数据分析提供关键元数据支持。**

- **链接: [https://arxiv.org/pdf/2511.20254v1](https://arxiv.org/pdf/2511.20254v1)**

> **作者:** Alexander C. Jenke; Gregor Just; Claas de Boer; Martin Wagner; Sebastian Bodenstedt; Stefanie Speidel
>
> **摘要:** Purpose: Robot-assisted minimally invasive surgery relies on endoscopic video as the sole intraoperative visual feedback. The DaVinci Xi system overlays a graphical user interface (UI) that indicates the state of each robotic arm, including the activation of the endoscope arm. Detecting this activation provides valuable metadata such as camera movement information, which can support downstream surgical data science tasks including tool tracking, skill assessment, or camera control automation. Methods: We developed a lightweight pipeline based on a ResNet18 convolutional neural network to automatically identify the position of the camera tile and its activation state within the DaVinci Xi UI. The model was fine-tuned on manually annotated data from the SurgToolLoc dataset and evaluated across three public datasets comprising over 70,000 frames. Results: The model achieved F1-scores between 0.993 and 1.000 for the binary detection of active cameras and correctly localized the camera tile in all cases without false multiple-camera detections. Conclusion: The proposed pipeline enables reliable, real-time extraction of camera activation metadata from surgical videos, facilitating automated preprocessing and analysis for diverse downstream applications. All code, trained models, and annotations are publicly available.
>
---
#### [new 183] Diverse Video Generation with Determinantal Point Process-Guided Policy Optimization
- **分类: cs.CV**

- **简介: 该论文针对文本到视频生成中多样性不足的问题，提出DPP-GRPO框架，结合确定性点过程与组相对策略优化，显式鼓励生成多样化视频。方法不依赖特定模型，可提升视觉、运镜和场景结构的多样性，同时保持提示一致性和质量，在多个基准上验证有效。**

- **链接: [https://arxiv.org/pdf/2511.20647v1](https://arxiv.org/pdf/2511.20647v1)**

> **作者:** Tahira Kazimi; Connor Dunlop; Pinar Yanardag
>
> **备注:** Project webpage: https://diverse-video.github.io/
>
> **摘要:** While recent text-to-video (T2V) diffusion models have achieved impressive quality and prompt alignment, they often produce low-diversity outputs when sampling multiple videos from a single text prompt. We tackle this challenge by formulating it as a set-level policy optimization problem, with the goal of training a policy that can cover the diverse range of plausible outcomes for a given prompt. To address this, we introduce DPP-GRPO, a novel framework for diverse video generation that combines Determinantal Point Processes (DPPs) and Group Relative Policy Optimization (GRPO) theories to enforce explicit reward on diverse generations. Our objective turns diversity into an explicit signal by imposing diminishing returns on redundant samples (via DPP) while supplies groupwise feedback over candidate sets (via GRPO). Our framework is plug-and-play and model-agnostic, and encourages diverse generations across visual appearance, camera motions, and scene structure without sacrificing prompt fidelity or perceptual quality. We implement our method on WAN and CogVideoX, and show that our method consistently improves video diversity on state-of-the-art benchmarks such as VBench, VideoScore, and human preference studies. Moreover, we release our code and a new benchmark dataset of 30,000 diverse prompts to support future research.
>
---
#### [new 184] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning
- **分类: cs.CV; cs.MA**

- **简介: 该论文针对视频理解任务，解决现有方法中工具调用机制静态、缺乏动态协作的问题。提出VideoChat-M1系统，基于多智能体强化学习的协同策略规划框架，通过策略生成、执行与通信实现动态优化，显著提升复杂视频理解性能，在多个基准上达到SOTA。**

- **链接: [https://arxiv.org/pdf/2511.19524v1](https://arxiv.org/pdf/2511.19524v1)**

> **作者:** Boyu Chen; Zikang Wang; Zhengrong Yue; Kainan Yan; Chenyun Yu; Yi Huang; Zijun Liu; Yafei Wen; Xiaoxin Chen; Yang Liu; Peng Li; Yali Wang
>
> **备注:** 21 pages, 9 figures
>
> **摘要:** By leveraging tool-augmented Multimodal Large Language Models (MLLMs), multi-agent frameworks are driving progress in video understanding. However, most of them adopt static and non-learnable tool invocation mechanisms, which limit the discovery of diverse clues essential for robust perception and reasoning regarding temporally or spatially complex videos. To address this challenge, we propose a novel Multi-agent system for video understanding, namely VideoChat-M1. Instead of using a single or fixed policy, VideoChat-M1 adopts a distinct Collaborative Policy Planning (CPP) paradigm with multiple policy agents, which comprises three key processes. (1) Policy Generation: Each agent generates its unique tool invocation policy tailored to the user's query; (2) Policy Execution: Each agent sequentially invokes relevant tools to execute its policy and explore the video content; (3) Policy Communication: During the intermediate stages of policy execution, agents interact with one another to update their respective policies. Through this collaborative framework, all agents work in tandem, dynamically refining their preferred policies based on contextual insights from peers to effectively respond to the user's query. Moreover, we equip our CPP paradigm with a concise Multi-Agent Reinforcement Learning (MARL) method. Consequently, the team of policy agents can be jointly optimized to enhance VideoChat-M1's performance, guided by both the final answer reward and intermediate collaborative process feedback. Extensive experiments demonstrate that VideoChat-M1 achieves SOTA performance across eight benchmarks spanning four tasks. Notably, on LongVideoBench, our method outperforms the SOTA model Gemini 2.5 pro by 3.6% and GPT-4o by 15.6%.
>
---
#### [new 185] MFM-point: Multi-scale Flow Matching for Point Cloud Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对点云生成任务，解决点基于方法生成质量与可扩展性不足的问题。提出MFM-Point多尺度流匹配框架，采用粗到细生成策略，结合结构化下采样与上采样，有效保持几何结构与分布一致性，显著提升生成质量与效率，达到当前最优点基方法性能，并挑战主流表示基方法。**

- **链接: [https://arxiv.org/pdf/2511.20041v1](https://arxiv.org/pdf/2511.20041v1)**

> **作者:** Petr Molodyk; Jaemoo Choi; David W. Romero; Ming-Yu Liu; Yongxin Chen
>
> **摘要:** In recent years, point cloud generation has gained significant attention in 3D generative modeling. Among existing approaches, point-based methods directly generate point clouds without relying on other representations such as latent features, meshes, or voxels. These methods offer low training cost and algorithmic simplicity, but often underperform compared to representation-based approaches. In this paper, we propose MFM-Point, a multi-scale Flow Matching framework for point cloud generation that substantially improves the scalability and performance of point-based methods while preserving their simplicity and efficiency. Our multi-scale generation algorithm adopts a coarse-to-fine generation paradigm, enhancing generation quality and scalability without incurring additional training or inference overhead. A key challenge in developing such a multi-scale framework lies in preserving the geometric structure of unordered point clouds while ensuring smooth and consistent distributional transitions across resolutions. To address this, we introduce a structured downsampling and upsampling strategy that preserves geometry and maintains alignment between coarse and fine resolutions. Our experimental results demonstrate that MFM-Point achieves best-in-class performance among point-based methods and challenges the best representation-based methods. In particular, MFM-point demonstrates strong results in multi-category and high-resolution generation tasks.
>
---
#### [new 186] INTERLACE: Interleaved Layer Pruning and Efficient Adaptation in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型（VLM）层冗余问题，提出INTERLACE框架，通过分析连续三层的局部冗余，剪枝前两层中最冗余的一层，仅微调剩余层并冻结第三层，实现高效适配。在1%数据上微调一epoch即达88.9%性能保留，显著提升剪枝后模型表现。**

- **链接: [https://arxiv.org/pdf/2511.19676v1](https://arxiv.org/pdf/2511.19676v1)**

> **作者:** Parsa Madinei; Ryan Solgi; Ziqi Wen; Jonathan Skaza; Miguel Eckstein; Ramtin Pedarsani
>
> **摘要:** We introduce INTERLACE, a novel framework that prunes redundant layers in VLMs while maintaining performance through sample-efficient finetuning. Existing layer pruning methods lead to significant performance drop when applied to VLMs. Instead, we analyze triplets of consecutive layers to identify local redundancy, removing the most redundant of the first two layers, finetune the remaining layer to compensate for the lost capacity, and freeze the third layer to serve as a stable anchor during finetuning. We found that this interleaved finetune-freeze design enables rapid convergence with minimal data after pruning. By finetuning only a subset of layers on just 1% of the FineVision dataset for one epoch, Interlace achieves 88.9% average performance retention after dropping 25% of the network, achieving SOTA performance. Our code is available at: https://github.com/pmadinei/Interlace.git
>
---
#### [new 187] Reasoning-VLA: A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中视觉-语言-动作（VLA）模型推理慢、泛化能力差的问题，提出Reasoning-VLA框架。通过可学习的动作查询与增强的视觉语言特征交互，实现并行连续动作生成，并整合多数据集构建标准化训练数据，结合监督与强化学习，显著提升性能、泛化性与推理速度。**

- **链接: [https://arxiv.org/pdf/2511.19912v1](https://arxiv.org/pdf/2511.19912v1)**

> **作者:** Dapeng Zhang; Zhenlong Yuan; Zhangquan Chen; Chih-Ting Liao; Yinda Chen; Fei Shen; Qingguo Zhou; Tat-Seng Chua
>
> **摘要:** Vision-Language-Action (VLA) models have recently shown strong decision-making capabilities in autonomous driving. However, existing VLAs often struggle with achieving efficient inference and generalizing to novel autonomous vehicle configurations and driving scenarios. In this paper, we propose Reasoning-VLA, a general and fast action-generation VLA framework. The proposed model employs a set of learnable action queries, initialized via Gaussian sampling from ground-truth trajectories within the training corpus. These learnable queries interact with reasoning-enhanced vision-language features to generate continuous action trajectories in parallel. To promote robust generalization, we consolidate eight publicly available autonomous driving datasets into a standardized, Chain-of-Thought reasoning-based, and easy-to-use data format for model training. Leveraging both supervised learning and reinforcement learning fine-tuning, extensive empirical evaluations across multiple benchmarks demonstrate that Reasoning-VLA achieves state-of-the-art performance, superior generalization capability, and the excellent inference speed reported to date.
>
---
#### [new 188] 3D Motion Perception of Binocular Vision Target with PID-CNN
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对双目视觉目标的三维运动感知任务，解决实时获取三维坐标、速度与加速度的问题。提出一种17层小规模PID-CNN网络，通过特征复用提升性能，利用模拟随机运动球数据训练，实现接近图像分辨率上限的预测精度，并探讨了高维卷积与PID机制在计算效率与记忆注意力中的潜力。**

- **链接: [https://arxiv.org/pdf/2511.20332v1](https://arxiv.org/pdf/2511.20332v1)**

> **作者:** Shi Jiazhao; Pan Pan; Shi Haotian
>
> **备注:** 7 pages, 9 figures, 2 tables
>
> **摘要:** This article trained a network for perceiving three-dimensional motion information of binocular vision target, which can provide real-time three-dimensional coordinate, velocity, and acceleration, and has a basic spatiotemporal perception capability. Understood the ability of neural networks to fit nonlinear problems from the perspective of PID. Considered a single-layer neural network as using a second-order difference equation and a nonlinearity to describe a local problem. Multilayer networks gradually transform the raw representation to the desired representation through multiple such combinations. Analysed some reference principles for designing neural networks. Designed a relatively small PID convolutional neural network, with a total of 17 layers and 413 thousand parameters. Implemented a simple but practical feature reuse method by concatenation and pooling. The network was trained and tested using the simulated randomly moving ball datasets, and the experimental results showed that the prediction accuracy was close to the upper limit that the input image resolution can represent. Analysed the experimental results and errors, as well as the existing shortcomings and possible directions for improvement. Finally, discussed the advantages of high-dimensional convolution in improving computational efficiency and feature space utilization. As well as the potential advantages of using PID information to implement memory and attention mechanisms.
>
---
#### [new 189] ChessMamba: Structure-Aware Interleaving of State Spaces for Change Detection in Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文针对多时相遥感图像变化检测任务，解决异质性与时空错位导致的局部结构破坏问题。提出ChessMamba框架，通过棋盘式交错扫描序列化多时相特征，并结合多膨胀卷积实现结构感知融合，提升变化定位精度。**

- **链接: [https://arxiv.org/pdf/2511.19882v1](https://arxiv.org/pdf/2511.19882v1)**

> **作者:** Lei Ding; Tong Liu; Xuanguang Liu; Xiangyun Liu; Haitao Guo; Jun Lu
>
> **摘要:** Change detection (CD) in multitemporal remote sensing imagery presents significant challenges for fine-grained recognition, owing to heterogeneity and spatiotemporal misalignment. However, existing methodologies based on vision transformers or state-space models typically disrupt local structural consistency during temporal serialization, obscuring discriminative cues under misalignment and hindering reliable change localization. To address this, we introduce ChessMamba, a structure-aware framework leveraging interleaved state-space modeling for robust CD with multi-temporal inputs. ChessMamba integrates a SpatialMamba encoder with a lightweight cross-source interaction module, featuring two key innovations: (i) Chessboard interleaving with snake scanning order, which serializes multi-temporal features into a unified sequence within a single forward pass, thereby shortening interaction paths and enabling direct comparison for accurate change localization; and (ii) Structure-aware fusion via multi-dilated convolutions, selectively capturing center-and-corner neighborhood contexts within each mono-temporal. Comprehensive evaluations on three CD tasks, including binary CD, semantic CD and multimodal building damage assessment, demonstrate that ChessMamba effectively fuses heterogeneous features and achieves substantial accuracy improvements over state-of-the-art methods.The relevant code will be available at: github.com/DingLei14/ChessMamba.
>
---
#### [new 190] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos
- **分类: cs.CV**

- **简介: 该论文提出Mistake Attribution（MATT）任务，旨在细粒度理解第一人称视频中的人类错误。针对现有方法缺乏精细输出的问题，论文构建了MisEngine数据引擎，生成大规模标注数据集EPIC-KITCHENS-M和Ego4D-M，并提出MisFormer模型，统一解决错误的语义、时间、空间定位问题，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20525v1](https://arxiv.org/pdf/2511.20525v1)**

> **作者:** Yayuan Li; Aadit Jain; Filippos Bellos; Jason J. Corso
>
> **备注:** 11 pages, 4 figures, 6 tables
>
> **摘要:** We introduce Mistake Attribution (MATT), a task for fine-grained understanding of human mistakes in egocentric video. Unlike prior mistake understanding work, which lacks fine-grained output, MATT concretely attributes mistakes to the input instruction text or the attempt video. MATT determines what part of the instruction is violated (semantic role), when the deviation becomes irreversible (the Point-of-No-Return, PNR), and where the mistake appears in the PNR frame. We develop MisEngine, a data engine that automatically constructs attribution-rich mistake samples from existing datasets and inherits their annotations. Applied to large egocentric corpora, MisEngine yields EPIC-KITCHENS-M and Ego4D-M, two datasets that are up to two orders of magnitude larger than prior mistake datasets. We then present MisFormer, a unified attention-based model for mistake attribution across semantic (what), temporal (when), and spatial (where) dimensions, trained using MisEngine supervision. Experiments on our new datasets and prior benchmarks show that MisFormer outperforms strong video-language, temporal localization, hand-object interaction, and mistake-detection baselines.
>
---
#### [new 191] On the Feasibility of Hijacking MLLMs' Decision Chain via One Perturbation
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文研究多模态大模型决策链的对抗攻击问题。针对传统攻击仅影响单次决策的局限，提出通过单一扰动操控模型连续输出，实现对多个预设目标的精准误导。为此，设计语义感知通用扰动（SAUPs）与优化算法，并构建新数据集RIST进行验证，实证三模型在单帧扰动下70%成功率，揭示严重安全风险。**

- **链接: [https://arxiv.org/pdf/2511.20002v1](https://arxiv.org/pdf/2511.20002v1)**

> **作者:** Changyue Li; Jiaying Li; Youliang Yuan; Jiaming He; Zhicong Huang; Pinjia He
>
> **摘要:** Conventional adversarial attacks focus on manipulating a single decision of neural networks. However, real-world models often operate in a sequence of decisions, where an isolated mistake can be easily corrected, but cascading errors can lead to severe risks. This paper reveals a novel threat: a single perturbation can hijack the whole decision chain. We demonstrate the feasibility of manipulating a model's outputs toward multiple, predefined outcomes, such as simultaneously misclassifying "non-motorized lane" signs as "motorized lane" and "pedestrian" as "plastic bag". To expose this threat, we introduce Semantic-Aware Universal Perturbations (SAUPs), which induce varied outcomes based on the semantics of the inputs. We overcome optimization challenges by developing an effective algorithm, which searches for perturbations in normalized space with a semantic separation strategy. To evaluate the practical threat of SAUPs, we present RIST, a new real-world image dataset with fine-grained semantic annotations. Extensive experiments on three multimodal large language models demonstrate their vulnerability, achieving a 70% attack success rate when controlling five distinct targets using just an adversarial frame.
>
---
#### [new 192] Single Image to High-Quality 3D Object via Latent Features
- **分类: cs.CV**

- **简介: 该论文针对单图生成高质量3D物体的任务，解决现有方法难以兼顾速度、细节与保真度的问题。提出LatentDreamer框架，利用预训练变分自编码器将3D几何映射到潜在特征，通过潜空间逐步生成粗略形状、精细结构与真实纹理，在70秒内完成高保真生成，仅需少量训练即可达到先进水平。**

- **链接: [https://arxiv.org/pdf/2511.19512v1](https://arxiv.org/pdf/2511.19512v1)**

> **作者:** Huanning Dong; Yinuo Huang; Fan Li; Ping Kuang
>
> **摘要:** 3D assets are essential in the digital age. While automatic 3D generation, such as image-to-3d, has made significant strides in recent years, it often struggles to achieve fast, detailed, and high-fidelity generation simultaneously. In this work, we introduce LatentDreamer, a novel framework for generating 3D objects from single images. The key to our approach is a pre-trained variational autoencoder that maps 3D geometries to latent features, which greatly reducing the difficulty of 3D generation. Starting from latent features, the pipeline of LatentDreamer generates coarse geometries, refined geometries, and realistic textures sequentially. The 3D objects generated by LatentDreamer exhibit high fidelity to the input images, and the entire generation process can be completed within a short time (typically in 70 seconds). Extensive experiments show that with only a small amount of training, LatentDreamer demonstrates competitive performance compared to contemporary approachs.
>
---
#### [new 193] It Hears, It Sees too: Multi-Modal LLM for Depression Detection By Integrating Visual Understanding into Audio Language Models
- **分类: cs.MM; cs.CV; cs.LG; eess.AS**

- **简介: 该论文针对抑郁症检测任务，解决传统语言模型无法处理视听非语言线索的问题。提出一种融合音频与视觉理解的多模态大模型，通过时间戳级对齐实现跨模态动态建模，提升检测精度并降低资源需求，实验验证其优于单模态及现有多模态方法。**

- **链接: [https://arxiv.org/pdf/2511.19877v1](https://arxiv.org/pdf/2511.19877v1)**

> **作者:** Xiangyu Zhao; Yaling Shen; Yiwen Jiang; Zimu Wang; Jiahe Liu; Maxmartwell H Cheng; Guilherme C Oliveira; Robert Desimone; Dominic Dwyer; Zongyuan Ge
>
> **摘要:** Depression is one of the most prevalent mental health disorders globally. In recent years, multi-modal data, such as speech, video, and transcripts, has been increasingly used to develop AI-assisted depression assessment systems. Large language models have further advanced this field due to their strong language understanding and generalization capabilities. However, conventional LLMs remain text-centric and cannot process the rich non-verbal cues found in audio and visual modalities, which are critical components in mental health evaluation. While multi-modal LLMs offer a promising direction, few are tailored for psychological applications. In this study, we propose a novel multi-modal LLM framework for depression detection. Our approach augments an audio language model with visual understanding and aligns audio-visual features at the timestamp level. This fine-grained alignment improves modeling of temporal dynamics across modalities while reducing the need for extensive training data and computational resources. Experiments on the DAIC-WoZ dataset demonstrate that our model outperforms both single-modality approaches and previous multi-modal methods. Moreover, the proposed framework can be extended to incorporate additional physiological signals, paving the way for broader clinical applications beyond mental health.
>
---
#### [new 194] SPQR: A Standardized Benchmark for Modern Safety Alignment Methods in Text-to-Image Diffusion Models
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对文本到图像扩散模型的安全对齐问题，提出SPQR基准。研究发现现有安全方法在良性微调后易失效，因此构建了综合评估安全、实用性与鲁棒性的单一分数基准，以标准化评测模型在部署后仍能保持安全的能力。**

- **链接: [https://arxiv.org/pdf/2511.19558v1](https://arxiv.org/pdf/2511.19558v1)**

> **作者:** Mohammed Talha Alam; Nada Saadi; Fahad Shamshad; Nils Lukas; Karthik Nandakumar; Fahkri Karray; Samuele Poppi
>
> **备注:** 20 pages, 8 figures, 10 tables
>
> **摘要:** Text-to-image diffusion models can emit copyrighted, unsafe, or private content. Safety alignment aims to suppress specific concepts, yet evaluations seldom test whether safety persists under benign downstream fine-tuning routinely applied after deployment (e.g., LoRA personalization, style/domain adapters). We study the stability of current safety methods under benign fine-tuning and observe frequent breakdowns. As true safety alignment must withstand even benign post-deployment adaptations, we introduce the SPQR benchmark (Safety-Prompt adherence-Quality-Robustness). SPQR is a single-scored metric that provides a standardized and reproducible framework to evaluate how well safety-aligned diffusion models preserve safety, utility, and robustness under benign fine-tuning, by reporting a single leaderboard score to facilitate comparisons. We conduct multilingual, domain-specific, and out-of-distribution analyses, along with category-wise breakdowns, to identify when safety alignment fails after benign fine-tuning, ultimately showcasing SPQR as a concise yet comprehensive benchmark for T2I safety alignment techniques for T2I models.
>
---
#### [new 195] Merging without Forgetting: Continual Fusion of Task-Specific Models via Optimal Transport
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究多任务模型合并问题，针对现有方法因参数插值导致特征分布偏移、任务知识丢失的问题，提出基于最优传输的OTMF框架。通过挖掘共享掩码对齐语义几何，保留任务特异性并融合可迁移成分，支持持续增量融合，实现高效高精度的模型合并。**

- **链接: [https://arxiv.org/pdf/2511.19561v1](https://arxiv.org/pdf/2511.19561v1)**

> **作者:** Zecheng Pan; Zhikang Chen; Ding Li; Min Zhang; Sen Cui; Hongshuo Jin; Luqi Tao; Yi Yang; Deheng Ye; Yu Zhang; Tingting Zhu; Tianling Ren
>
> **摘要:** Merging models fine-tuned for different tasks into a single unified model has become an increasingly important direction for building versatile, efficient multi-task systems. Existing approaches predominantly rely on parameter interpolation in weight space, which we show introduces significant distribution shift in the feature space and undermines task-specific knowledge. In this paper, we propose OTMF (Optimal Transport-based Masked Fusion), a novel model merging framework rooted in optimal transport theory to address the distribution shift that arises from naive parameter interpolation. Instead of directly aggregating features or weights, OTMF aligns the semantic geometry of task-specific models by discovering common masks applied to task vectors through optimal transport plans. These masks selectively extract transferable and task-agnostic components while preserving the unique structural identities of each task. To ensure scalability in real-world settings, OTMF further supports a continual fusion paradigm that incrementally integrates each new task vector without revisiting previous ones, maintaining a bounded memory footprint and enabling efficient fusion across a growing number of tasks. We conduct comprehensive experiments on multiple vision and language benchmarks, and results show that OTMF achieves state-of-the-art performance in terms of both accuracy and efficiency. These findings highlight the practical and theoretical value of our approach to model merging.
>
---
#### [new 196] Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Splatblox，用于户外复杂环境中的机器人自主导航。针对植被密集、障碍物不规则等问题，融合RGB与LiDAR数据，基于高斯点阵构建可通行性感知的欧氏有符号距离场，实现几何与语义联合建模，支持实时路径规划与避障，显著提升导航成功率与效率。**

- **链接: [https://arxiv.org/pdf/2511.18525v1](https://arxiv.org/pdf/2511.18525v1)**

> **作者:** Samarth Chopra; Jing Liang; Gershom Seneviratne; Yonghan Lee; Jaehoon Choi; Jianyu An; Stephen Cheng; Dinesh Manocha
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: https://splatblox.github.io
>
---
#### [new 197] A Multi-Stage Deep Learning Framework with PKCP-MixUp Augmentation for Pediatric Liver Tumor Diagnosis Using Multi-Phase Contrast-Enhanced CT
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文针对儿科肝肿瘤非侵入性诊断难题，提出多阶段深度学习框架，结合PKCP-MixUp数据增强与多期增强CT，实现肿瘤检测与良恶性、病理分型分类。解决了数据稀缺、模型可解释性不足等问题，显著提升诊断准确性与临床可用性。**

- **链接: [https://arxiv.org/pdf/2511.19478v1](https://arxiv.org/pdf/2511.19478v1)**

> **作者:** Wanqi Wang; Chun Yang; Jianbo Shao; Yaokai Zhang; Xuehua Peng; Jin Sun; Chao Xiong; Long Lu; Lianting Hu
>
> **摘要:** Pediatric liver tumors are one of the most common solid tumors in pediatrics, with differentiation of benign or malignant status and pathological classification critical for clinical treatment. While pathological examination is the gold standard, the invasive biopsy has notable limitations: the highly vascular pediatric liver and fragile tumor tissue raise complication risks such as bleeding; additionally, young children with poor compliance require anesthesia for biopsy, increasing medical costs or psychological trauma. Although many efforts have been made to utilize AI in clinical settings, most researchers have overlooked its importance in pediatric liver tumors. To establish a non-invasive examination procedure, we developed a multi-stage deep learning (DL) framework for automated pediatric liver tumor diagnosis using multi-phase contrast-enhanced CT. Two retrospective and prospective cohorts were enrolled. We established a novel PKCP-MixUp data augmentation method to address data scarcity and class imbalance. We also trained a tumor detection model to extract ROIs, and then set a two-stage diagnosis pipeline with three backbones with ROI-masked images. Our tumor detection model has achieved high performance (mAP=0.871), and the first stage classification model between benign and malignant tumors reached an excellent performance (AUC=0.989). Final diagnosis models also exhibited robustness, including benign subtype classification (AUC=0.915) and malignant subtype classification (AUC=0.979). We also conducted multi-level comparative analyses, such as ablation studies on data and training pipelines, as well as Shapley-Value and CAM interpretability analyses. This framework fills the pediatric-specific DL diagnostic gap, provides actionable insights for CT phase selection and model design, and paves the way for precise, accessible pediatric liver tumor diagnosis.
>
---
#### [new 198] Frequency Bias Matters: Diving into Robust and Generalized Deep Image Forgery Detection
- **分类: cs.CR; cs.CV**

- **简介: 该论文聚焦于深度图像伪造检测任务，针对检测器在未知生成模型和噪声环境下的泛化性与鲁棒性不足问题，提出从频率视角分析并解决频率偏差。通过两步频率对齐方法，实现同时增强检测器鲁棒性与作为攻击手段的双重效果，验证了其在多种场景下的有效性。**

- **链接: [https://arxiv.org/pdf/2511.19886v1](https://arxiv.org/pdf/2511.19886v1)**

> **作者:** Chi Liu; Tianqing Zhu; Wanlei Zhou; Wei Zhao
>
> **备注:** Accepted for publication in IEEE Transactions on Dependable and Secure Computing
>
> **摘要:** As deep image forgery powered by AI generative models, such as GANs, continues to challenge today's digital world, detecting AI-generated forgeries has become a vital security topic. Generalizability and robustness are two critical concerns of a forgery detector, determining its reliability when facing unknown GANs and noisy samples in an open world. Although many studies focus on improving these two properties, the root causes of these problems have not been fully explored, and it is unclear if there is a connection between them. Moreover, despite recent achievements in addressing these issues from image forensic or anti-forensic aspects, a universal method that can contribute to both sides simultaneously remains practically significant yet unavailable. In this paper, we provide a fundamental explanation of these problems from a frequency perspective. Our analysis reveals that the frequency bias of a DNN forgery detector is a possible cause of generalization and robustness issues. Based on this finding, we propose a two-step frequency alignment method to remove the frequency discrepancy between real and fake images, offering double-sided benefits: it can serve as a strong black-box attack against forgery detectors in the anti-forensic context or, conversely, as a universal defense to improve detector reliability in the forensic context. We also develop corresponding attack and defense implementations and demonstrate their effectiveness, as well as the effect of the frequency alignment method, in various experimental settings involving twelve detectors, eight forgery models, and five metrics.
>
---
#### [new 199] The Selective Disk Bispectrum and Its Inversion, with Application to Multi-Reference Alignment
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对图像形状分析中的旋转不变性问题，提出可逆的快速选择性盘形双谱方法。解决了传统双谱因不可逆和高复杂度导致的应用瓶颈，实现了旋转图像的多参考对齐，为旋转不变形状学习提供了高效、可逆的表示工具。**

- **链接: [https://arxiv.org/pdf/2511.19706v1](https://arxiv.org/pdf/2511.19706v1)**

> **作者:** Adele Myers; Nina Miolane
>
> **摘要:** In many computer vision and shape analysis tasks, practitioners are interested in learning from the shape of the object in an image, while disregarding the object's orientation. To this end, it is valuable to define a rotation-invariant representation of images, retaining all information about that image, but disregarding the way an object is rotated in the frame. To be practical for learning tasks, this representation must be computationally efficient for large datasets and invertible, so the representation can be visualized in image space. To this end, we present the selective disk bispectrum: a fast, rotation-invariant representation for image shape analysis. While the translational bispectrum has long been used as a translational invariant representation for 1-D and 2-D signals, its extension to 2-D (disk) rotational invariance on images has been hindered by the absence of an invertible formulation and its cubic complexity. In this work, we derive an explicit inverse for the disk bispectrum, which allows us to define a "selective" disk bispectrum, which only uses the minimal number of coefficients needed for faithful shape recovery. We show that this representation enables multi-reference alignment for rotated images-a task previously intractable for disk bispectrum methods. These results establish the disk bispectrum as a practical and theoretically grounded tool for learning on rotation-invariant shape data.
>
---
#### [new 200] Development of a fully deep learning model to improve the reproducibility of sector classification systems for predicting unerupted maxillary canine likelihood of impaction
- **分类: eess.IV; cs.CV; q-bio.QM**

- **简介: 该论文属于医学影像智能分析任务，旨在解决口腔正畸中未萌上颌尖牙位置分类的可重复性问题。研究对比三种分类系统，利用深度学习模型（DenseNet121）自动分类，提升诊断一致性，结果显示AI模型具备76.8%准确率，可有效辅助临床决策。**

- **链接: [https://arxiv.org/pdf/2511.20493v1](https://arxiv.org/pdf/2511.20493v1)**

> **作者:** Marzio Galdi; Davide Cannatà; Flavia Celentano; Luigia Rizzo; Domenico Rossi; Tecla Bocchino; Stefano Martina
>
> **摘要:** Objectives. The aim of the present study was to develop a fully deep learning model to reduce the intra- and inter-operator reproducibility of sector classification systems for predicting unerupted maxillary canine likelihood of impaction. Methods. Three orthodontists (Os) and three general dental practitioners (GDPs) classified the position of unerupted maxillary canines on 306 radiographs (T0) according to the three different sector classification systems (5-, 4-, and 3-sector classification system). The assessment was repeated after four weeks (T1). Intra- and inter-observer agreement were evaluated with Cohen's K and Fleiss K, and between group differences with a z-test. The same radiographs were tested on different artificial intelligence (AI) models, pre-trained on an extended dataset of 1,222 radiographs. The best-performing model was identified based on its sensitivity and precision. Results. The 3-sector system was found to be the classification method with highest reproducibility, with an agreement (Cohen's K values) between observations (T0 versus T1) for each examiner ranged from 0.80 to 0.92, and an overall agreement of 0.85 [95% confidence interval (CI) = 0.83-0.87]. The overall inter-observer agreement (Fleiss K) ranged from 0.69 to 0.7. The educational background did not affect either intra- or inter-observer agreement (p>0.05). DenseNet121 proved to be the best-performing model in allocating impacted canines in the three different classes, with an overall accuracy of 76.8%. Conclusion. AI models can be designed to automatically classify the position of unerupted maxillary canines.
>
---
#### [new 201] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics
- **分类: q-bio.QM; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对空间转录组学中基因表达预测任务，解决现有方法对生物知识利用不足、依赖样本检索及模态对齐不佳的问题。提出DKAN模型，通过双路径对比对齐与基因语义增强，融合病理图像与基因表达数据，实现高效跨模态特征整合，显著提升预测性能。**

- **链接: [https://arxiv.org/pdf/2511.17685v1](https://arxiv.org/pdf/2511.17685v1)**

> **作者:** Wei Zhang; Jiajun Chu; Xinci Liu; Chen Tong; Xinyue Li
>
> **备注:** AAAI 2026 Oral, extended version
>
> **摘要:** Spatial Transcriptomics (ST) is a technology that measures gene expression profiles within tissue sections while retaining spatial context. It reveals localized gene expression patterns and tissue heterogeneity, both of which are essential for understanding disease etiology. However, its high cost has driven efforts to predict spatial gene expression from whole slide images. Despite recent advancements, current methods still face significant limitations, such as under-exploitation of high-level biological context, over-reliance on exemplar retrievals, and inadequate alignment of heterogeneous modalities. To address these challenges, we propose DKAN, a novel Dual-path Knowledge-Augmented contrastive alignment Network that predicts spatially resolved gene expression by integrating histopathological images and gene expression profiles through a biologically informed approach. Specifically, we introduce an effective gene semantic representation module that leverages the external gene database to provide additional biological insights, thereby enhancing gene expression prediction. Further, we adopt a unified, one-stage contrastive learning paradigm, seamlessly combining contrastive learning and supervised learning to eliminate reliance on exemplars, complemented with an adaptive weighting mechanism. Additionally, we propose a dual-path contrastive alignment module that employs gene semantic features as dynamic cross-modal coordinators to enable effective heterogeneous feature integration. Through extensive experiments across three public ST datasets, DKAN demonstrates superior performance over state-of-the-art models, establishing a new benchmark for spatial gene expression prediction and offering a powerful tool for advancing biological and clinical research.
>
---
#### [new 202] On-Demand Multi-Task Sparsity for Efficient Large-Model Deployment on Edge Devices
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对边缘设备上大模型多任务部署的效率问题，提出一种按需多任务稀疏框架。通过将权重分解为可重用的块粒度单元并对齐稀疏结构，实现任务间参数高效复用，显著降低频繁切换任务时的输入输出开销与冷启动延迟。实验表明，该方法平均提速6.6倍。**

- **链接: [https://arxiv.org/pdf/2511.19986v1](https://arxiv.org/pdf/2511.19986v1)**

> **作者:** Lianming Huang; Haibo Hu; Qiao Li; Nan Guan; Chun Jason Xue
>
> **摘要:** Sparsity is essential for deploying large models on resource constrained edge platforms. However, optimizing sparsity patterns for individual tasks in isolation ignores the significant I/O overhead incurred during frequent task switching. We introduce an on-demand multi-task sparsity framework specifically designed to minimize switching costs by maximizing parameter reuse. Unlike monolithic approaches, we decompose weights into reusable block-granular units and align sparse structures across tasks to maximize overlap. By dynamically loading only the small differential set of blocks required for the next task, our method effectively mitigates the cold-start latency inherent in traditional monolithic approaches.Experiments on a real-world autonomous driving platform demonstrate that our framework achieves superior switching efficiency, accelerating task switching by over 6.6X on average compared to existing sparsity methods.
>
---
#### [new 203] Beyond Binary Classification: A Semi-supervised Approach to Generalized AI-generated Image Detection
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **简介: 该论文针对AI生成图像检测任务，解决现有方法在跨架构（如GAN到扩散模型）下泛化能力差的问题。通过分析不同架构生成图像的内在差异，提出半监督的TriDetect方法，利用聚类与跨视图一致性挖掘潜在架构特征，提升对未知生成器的检测性能。**

- **链接: [https://arxiv.org/pdf/2511.19499v1](https://arxiv.org/pdf/2511.19499v1)**

> **作者:** Hong-Hanh Nguyen-Le; Van-Tuan Tran; Dinh-Thuc Nguyen; Nhien-An Le-Khac
>
> **备注:** Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - 2025
>
> **摘要:** The rapid advancement of generators (e.g., StyleGAN, Midjourney, DALL-E) has produced highly realistic synthetic images, posing significant challenges to digital media authenticity. These generators are typically based on a few core architectural families, primarily Generative Adversarial Networks (GANs) and Diffusion Models (DMs). A critical vulnerability in current forensics is the failure of detectors to achieve cross-generator generalization, especially when crossing architectural boundaries (e.g., from GANs to DMs). We hypothesize that this gap stems from fundamental differences in the artifacts produced by these \textbf{distinct architectures}. In this work, we provide a theoretical analysis explaining how the distinct optimization objectives of the GAN and DM architectures lead to different manifold coverage behaviors. We demonstrate that GANs permit partial coverage, often leading to boundary artifacts, while DMs enforce complete coverage, resulting in over-smoothing patterns. Motivated by this analysis, we propose the \textbf{Tri}archy \textbf{Detect}or (TriDetect), a semi-supervised approach that enhances binary classification by discovering latent architectural patterns within the "fake" class. TriDetect employs balanced cluster assignment via the Sinkhorn-Knopp algorithm and a cross-view consistency mechanism, encouraging the model to learn fundamental architectural distincts. We evaluate our approach on two standard benchmarks and three in-the-wild datasets against 13 baselines to demonstrate its generalization capability to unseen generators.
>
---
#### [new 204] Redefining Radar Segmentation: Simultaneous Static-Moving Segmentation and Ego-Motion Estimation using Radar Point Clouds
- **分类: eess.SP; cs.CV**

- **简介: 该论文提出一种基于雷达点云的神经网络方法，同时实现静态与动态物体分割及自运动估计。针对传统雷达分割忽视动静分类、依赖复杂预处理的问题，该方法直接从原始点云中提取特征，无需聚合或补偿，首次实现双任务联合求解，验证了其在真实数据上的有效性与应用潜力。**

- **链接: [https://arxiv.org/pdf/2511.20003v1](https://arxiv.org/pdf/2511.20003v1)**

> **作者:** Simin Zhu; Satish Ravindran; Alexander Yarovoy; Francesco Fioranelli
>
> **备注:** 16 pages, 9 figures, under review at IEEE Transactions on Radar Systems
>
> **摘要:** Conventional radar segmentation research has typically focused on learning category labels for different moving objects. Although fundamental differences between radar and optical sensors lead to differences in the reliability of predicting accurate and consistent category labels, a review of common radar perception tasks in automotive reveals that determining whether an object is moving or static is a prerequisite for most tasks. To fill this gap, this study proposes a neural network based solution that can simultaneously segment static and moving objects from radar point clouds. Furthermore, since the measured radial velocity of static objects is correlated with the motion of the radar, this approach can also estimate the instantaneous 2D velocity of the moving platform or vehicle (ego motion). However, despite performing dual tasks, the proposed method employs very simple yet effective building blocks for feature extraction: multi layer perceptrons (MLPs) and recurrent neural networks (RNNs). In addition to being the first of its kind in the literature, the proposed method also demonstrates the feasibility of extracting the information required for the dual task directly from unprocessed point clouds, without the need for cloud aggregation, Doppler compensation, motion compensation, or any other intermediate signal processing steps. To measure its performance, this study introduces a set of novel evaluation metrics and tests the proposed method using a challenging real world radar dataset, RadarScenes. The results show that the proposed method not only performs well on the dual tasks, but also has broad application potential in other radar perception tasks.
>
---
#### [new 205] DLADiff: A Dual-Layer Defense Framework against Fine-Tuning and Zero-Shot Customization of Diffusion Models
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对扩散模型在微调和零样本定制中泄露人脸隐私的问题，提出双层防御框架DLADiff。第一层通过双代理模型与动态微调防御微调攻击，第二层有效阻止零样本生成。实验表明，该方法在防御两类攻击上均显著优于现有技术。**

- **链接: [https://arxiv.org/pdf/2511.19910v1](https://arxiv.org/pdf/2511.19910v1)**

> **作者:** Jun Jia; Hongyi Miao; Yingjie Zhou; Linhan Cao; Yanwei Jiang; Wangqiu Zhou; Dandan Zhu; Hua Yang; Wei Sun; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** With the rapid advancement of diffusion models, a variety of fine-tuning methods have been developed, enabling high-fidelity image generation with high similarity to the target content using only 3 to 5 training images. More recently, zero-shot generation methods have emerged, capable of producing highly realistic outputs from a single reference image without altering model weights. However, technological advancements have also introduced significant risks to facial privacy. Malicious actors can exploit diffusion model customization with just a few or even one image of a person to create synthetic identities nearly identical to the original identity. Although research has begun to focus on defending against diffusion model customization, most existing defense methods target fine-tuning approaches and neglect zero-shot generation defenses. To address this issue, this paper proposes Dual-Layer Anti-Diffusion (DLADiff) to defense both fine-tuning methods and zero-shot methods. DLADiff contains a dual-layer protective mechanism. The first layer provides effective protection against unauthorized fine-tuning by leveraging the proposed Dual-Surrogate Models (DSUR) mechanism and Alternating Dynamic Fine-Tuning (ADFT), which integrates adversarial training with the prior knowledge derived from pre-fine-tuned models. The second layer, though simple in design, demonstrates strong effectiveness in preventing image generation through zero-shot methods. Extensive experimental results demonstrate that our method significantly outperforms existing approaches in defending against fine-tuning of diffusion models and achieves unprecedented performance in protecting against zero-shot generation.
>
---
#### [new 206] VibraVerse: A Large-Scale Geometry-Acoustics Alignment Dataset for Physically-Consistent Multimodal Learning
- **分类: cs.AI; cs.CV; cs.GR; cs.RO**

- **简介: 论文提出VibraVerse，一个大规模几何-声学对齐数据集，解决现有多模态学习缺乏物理一致性的问题。通过建模物体几何、材料属性与振动发声的因果关系，构建从形状到声音的可解释映射，并设计CLASP框架实现跨模态物理一致对齐，推动可解释的声学引导感知。**

- **链接: [https://arxiv.org/pdf/2511.20422v1](https://arxiv.org/pdf/2511.20422v1)**

> **作者:** Bo Pang; Chenxi Xu; Jierui Ren; Guoping Wang; Sheng Li
>
> **摘要:** Understanding the physical world requires perceptual models grounded in physical laws rather than mere statistical correlations. However, existing multimodal learning frameworks, focused on vision and language, lack physical consistency and overlook the intrinsic causal relationships among an object's geometry, material, vibration modes, and the sounds it produces. We introduce VibraVerse, a large-scale geometry-acoustics alignment dataset that explicitly bridges the causal chain from 3D geometry -> physical attributes -> modal parameters -> acoustic signals. Each 3D model has explicit physical properties (density, Young's modulus, Poisson's ratio) and volumetric geometry, from which modal eigenfrequencies and eigenvectors are computed for impact sound synthesis under controlled excitations. To establish this coherence, we introduce CLASP, a contrastive learning framework for cross-modal alignment that preserves the causal correspondence between an object's physical structure and its acoustic response. This framework enforces physically consistent alignment across modalities, ensuring that every sample is coherent, traceable to the governing equations, and embedded within a unified representation space spanning shape, image, and sound. Built upon VibraVerse, we define a suite of benchmark tasks for geometry-to-sound prediction, sound-guided shape reconstruction, and cross-modal representation learning. Extensive validations on these tasks demonstrate that models trained on VibraVerse exhibit superior accuracy, interpretability, and generalization across modalities. These results establish VibraVerse as a benchmark for physically consistent and causally interpretable multimodal learning, providing a foundation for sound-guided embodied perception and a deeper understanding of the physical world. The dataset will be open-sourced.
>
---
#### [new 207] Latent Diffusion Inversion Requires Understanding the Latent Space
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究隐空间生成模型（如LDM）的隐私泄露问题，针对模型逆向攻击中忽略编码器-解码器几何结构的问题，发现隐空间不同维度对记忆化贡献不均。提出基于维度重要性排序的方法，移除低贡献维度后显著提升成员推断攻击性能，验证了隐空间几何对隐私风险的关键影响。**

- **链接: [https://arxiv.org/pdf/2511.20592v1](https://arxiv.org/pdf/2511.20592v1)**

> **作者:** Mingxing Rao; Bowen Qu; Daniel Moyer
>
> **备注:** 14 pages, 4 figures, 4 tables
>
> **摘要:** The recovery of training data from generative models (``model inversion'') has been extensively studied for diffusion models in the data domain. The encoder/decoder pair and corresponding latent codes have largely been ignored by inversion techniques applied to latent space generative models, e.g., Latent Diffusion models (LDMs). In this work we describe two key findings: (1) The diffusion model exhibits non-uniform memorization across latent codes, tending to overfit samples located in high-distortion regions of the decoder pullback metric. (2) Even within a single latent code, different dimensions contribute unequally to memorization. We introduce a principled method to rank latent dimensions by their per-dimensional contribution to the decoder pullback metric, identifying those most responsible for memorization. Empirically, removing less-memorizing dimensions when computing attack statistics for score-based membership inference attacker significantly improves performance, with average AUROC gains of 2.7\% and substantial increases in TPR@1\%FPR (6.42\%) across diverse datasets including CIFAR-10, CelebA, ImageNet-1K, Pokémon, MS-COCO, and Flickr. This indicates stronger confidence in identifying members under extremely low false-positive tolerance. Our results highlight the overlooked influence of the auto-encoder geometry on LDM memorization and provide a new perspective for analyzing privacy risks in diffusion-based generative models.
>
---
#### [new 208] Zero-Shot Transfer Capabilities of the Sundial Foundation Model for Leaf Area Index Forecasting
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究遥感时间序列的零样本预测任务，旨在解决农业监测中叶面积指数（LAI）预测缺乏通用模型的问题。通过对比统计基线、监督LSTM与Sundial基础模型，发现Sundial在长时序上下文下可超越训练好的LSTM，首次证明通用基础模型无需微调即可优于专用模型。**

- **链接: [https://arxiv.org/pdf/2511.20004v1](https://arxiv.org/pdf/2511.20004v1)**

> **作者:** Peining Zhang; Hongchen Qin; Haochen Zhang; Ziqi Guo; Guiling Wang; Jinbo Bi
>
> **摘要:** This work investigates the zero-shot forecasting capability of time-series foundation models for Leaf Area Index (LAI) forecasting in agricultural monitoring. Using the HiQ dataset (U.S., 2000-2022), we systematically compare statistical baselines, a fully supervised LSTM, and the Sundial foundation model under multiple evaluation protocols. We find that Sundial, in the zero-shot setting, can outperform a fully trained LSTM provided that the input context window is sufficiently long-specifically, when covering more than one or two full seasonal cycles. This demonstrates, for the first time, that a general-purpose foundation model can surpass specialized supervised models on remote-sensing time series prediction without any task-specific tuning. These results highlight the strong potential of pretrained time-series foundation models to serve as effective plug-and-play forecasters in agricultural and environmental applications.
>
---
#### [new 209] Beyond Generation: Multi-Hop Reasoning for Factual Accuracy in Vision-Language Models
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对视觉语言模型（VLMs）事实性错误问题，提出基于知识图谱的多跳推理框架。通过图像-标题任务，实现视觉实体识别、知识图谱遍历与事实修正，提升生成准确性。实验表明，该方法显著改善事实准确率，推动VLMs在多模态推理上的发展。**

- **链接: [https://arxiv.org/pdf/2511.20531v1](https://arxiv.org/pdf/2511.20531v1)**

> **作者:** Shamima Hossain
>
> **备注:** Accepted as poster at NewInML Workshop ICML, 2025
>
> **摘要:** Visual Language Models (VLMs) are powerful generative tools but often produce factually inaccurate outputs due to a lack of robust reasoning capabilities. While extensive research has been conducted on integrating external knowledge for reasoning in large language models (LLMs), such efforts remain underexplored in VLMs, where the challenge is compounded by the need to bridge multiple modalities seamlessly. This work introduces a framework for knowledge-guided reasoning in VLMs, leveraging structured knowledge graphs for multi-hop verification using image-captioning task to illustrate our framework. Our approach enables systematic reasoning across multiple steps, including visual entity recognition, knowledge graph traversal, and fact-based caption refinement. We evaluate the framework using hierarchical, triple-based and bullet-point based knowledge representations, analyzing their effectiveness in factual accuracy and logical inference. Empirical results show that our approach improves factual accuracy by approximately 31% on preliminary experiments on a curated dataset of mixtures from Google Landmarks v2, Conceptual captions and Coco captions revealing key insights into reasoning patterns and failure modes. This work demonstrates the potential of integrating external knowledge for advancing reasoning in VLMs, paving the way for more reliable and knowledgable multimodal systems.
>
---
#### [new 210] ArtiBench and ArtiBrain: Benchmarking Generalizable Vision-Language Articulated Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉语言引导的可动物体操作中的泛化难题，提出ArtiBench基准与ArtiBrain框架。通过多场景、多层次任务评估，揭示了跨部件、跨实例的挑战；ArtiBrain融合高层推理与自适应控制，利用视觉语言模型分解任务，结合几何关键帧与扩散模型实现精准、可解释的操作，并通过可行动能记忆库提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.20330v1](https://arxiv.org/pdf/2511.20330v1)**

> **作者:** Yuhan Wu; Tiantian Wei; Shuo Wang; ZhiChao Wang; Yanyong Zhang; Daniel Cremers; Yan Xia
>
> **摘要:** Interactive articulated manipulation requires long-horizon, multi-step interactions with appliances while maintaining physical consistency. Existing vision-language and diffusion-based policies struggle to generalize across parts, instances, and categories. We first introduce ArtiBench, a five-level benchmark covering kitchen, storage, office, and tool environments. ArtiBench enables structured evaluation from cross-part and cross-instance variation to long-horizon multi-object tasks, revealing the core generalization challenges of articulated object manipulation. Building on this benchmark, we propose ArtiBrain, a modular framework that unifies high-level reasoning with adaptive low-level control. ArtiBrain uses a VLM-based Task Reasoner (GPT-4.1) to decompose and validate subgoals, and employs a Hybrid Controller that combines geometry-aware keyframe execution with affordance-guided diffusion for precise and interpretable manipulation. An Affordance Memory Bank continually accumulates successful execution episodes and propagates part-level actionable affordances to unseen articulated parts and configurations. Extensive experiments on ArtiBench show that our ArtiBrain significantly outperforms state-of-the-art multimodal and diffusion-based methods in robustness and generalization. Code and dataset will be released upon acceptance.
>
---
#### [new 211] Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对视觉语言模型（VLMs）在多步视觉推理中工具使用能力弱的问题，提出VISTA-Gym训练环境，通过标准化接口支持工具集成与强化学习。基于此，训练出VISTA-R1模型，实现工具与推理的协同，显著提升多任务视觉问答性能。**

- **链接: [https://arxiv.org/pdf/2511.19773v1](https://arxiv.org/pdf/2511.19773v1)**

> **作者:** Meng Lu; Ran Xu; Yi Fang; Wenxuan Zhang; Yue Yu; Gaurav Srivastava; Yuchen Zhuang; Mohamed Elhoseiny; Charles Fleming; Carl Yang; Zhengzhong Tu; Yang Xie; Guanghua Xiao; Hanrui Wang; Di Jin; Wenqi Shi; Xuan Wang
>
> **备注:** 17 pages, 9 figures, work in progress
>
> **摘要:** While recent vision-language models (VLMs) demonstrate strong image understanding, their ability to "think with images", i.e., to reason through multi-step visual interactions, remains limited. We introduce VISTA-Gym, a scalable training environment for incentivizing tool-integrated visual reasoning capabilities in VLMs. VISTA-Gym unifies diverse real-world multimodal reasoning tasks (7 tasks from 13 datasets in total) with a standardized interface for visual tools (e.g., grounding, parsing), executable interaction loops, verifiable feedback signals, and efficient trajectory logging, enabling visual agentic reinforcement learning at scale. While recent VLMs exhibit strong text-only reasoning, both proprietary and open-source models still struggle with tool selection, invocation, and coordination. With VISTA-Gym, we train VISTA-R1 to interleave tool-use with agentic reasoning via multi-turn trajectory sampling and end-to-end reinforcement learning. Extensive experiments across 11 public reasoning-intensive VQA benchmarks show that VISTA-R1-8B outperforms state-of-the-art baselines with similar sizes by 9.51%-18.72%, demonstrating VISTA-Gym as an effective training ground to unlock the tool-integrated reasoning capabilities for VLMs.
>
---
#### [new 212] Shortcut Invariance: Targeted Jacobian Regularization in Disentangled Latent Space
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文针对深度学习中的“捷径学习”问题，提出基于解耦潜在空间的靶向雅可比正则化方法。通过识别与标签强相关的简单位置特征，注入定向噪声使分类器对这些捷径信号不敏感，从而提升分布外泛化能力。属于鲁棒机器学习任务，旨在改善模型在未见数据上的表现。**

- **链接: [https://arxiv.org/pdf/2511.19525v1](https://arxiv.org/pdf/2511.19525v1)**

> **作者:** Shivam Pal; Sakshi Varshney; Piyush Rai
>
> **摘要:** Deep neural networks are prone to learning shortcuts, spurious and easily learned correlations in training data that cause severe failures in out-of-distribution (OOD) generalization. A dominant line of work seeks robustness by learning a robust representation, often explicitly partitioning the latent space into core and spurious components; this approach can be complex, brittle, and difficult to scale. We take a different approach, instead of a robust representation, we learn a robust function. We present a simple and effective training method that renders the classifier functionally invariant to shortcut signals. Our method operates within a disentangled latent space, which is essential as it isolates spurious and core features into distinct dimensions. This separation enables the identification of candidate shortcut features by their strong correlation with the label, used as a proxy for semantic simplicity. The classifier is then desensitized to these features by injecting targeted, anisotropic latent noise during training. We analyze this as targeted Jacobian regularization, which forces the classifier to ignore spurious features and rely on more complex, core semantic signals. The result is state-of-the-art OOD performance on established shortcut learning benchmarks.
>
---
#### [new 213] Fara-7B: An Efficient Agentic Model for Computer Use
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对计算机使用代理（CUA）缺乏高质量训练数据的问题，提出FaraGen合成数据生成系统，构建多步骤网页任务数据集。基于此数据训练出小型高效模型Fara-7B，仅用截图与坐标实现端上运行，在多个基准上超越同类模型，并优于更大模型，验证了可扩展数据生成对高效代理模型的关键作用。**

- **链接: [https://arxiv.org/pdf/2511.19663v1](https://arxiv.org/pdf/2511.19663v1)**

> **作者:** Ahmed Awadallah; Yash Lara; Raghav Magazine; Hussein Mozannar; Akshay Nambi; Yash Pandya; Aravind Rajeswaran; Corby Rosset; Alexey Taymanov; Vibhav Vineet; Spencer Whitehead; Andrew Zhao
>
> **摘要:** Progress in computer use agents (CUAs) has been constrained by the absence of large and high-quality datasets that capture how humans interact with a computer. While LLMs have thrived on abundant textual data, no comparable corpus exists for CUA trajectories. To address these gaps, we introduce FaraGen, a novel synthetic data generation system for multi-step web tasks. FaraGen can propose diverse tasks from frequently used websites, generate multiple solution attempts, and filter successful trajectories using multiple verifiers. It achieves high throughput, yield, and diversity for multi-step web tasks, producing verified trajectories at approximately $1 each. We use this data to train Fara-7B, a native CUA model that perceives the computer using only screenshots, executes actions via predicted coordinates, and is small enough to run on-device. We find that Fara-7B outperforms other CUA models of comparable size on benchmarks like WebVoyager, Online-Mind2Web, and WebTailBench -- our novel benchmark that better captures under-represented web tasks in pre-existing benchmarks. Furthermore, Fara-7B is competitive with much larger frontier models, illustrating key benefits of scalable data generation systems in advancing small efficient agentic models. We are making Fara-7B open-weight on Microsoft Foundry and HuggingFace, and we are releasing WebTailBench.
>
---
#### [new 214] Not Quite Anything: Overcoming SAMs Limitations for 3D Medical Imaging
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对基础分割模型SAM在脑MRI图像上因结构边界模糊、对比度低导致性能下降的问题，提出一种无需微调的组合式方法。通过将SAM输出作为额外输入通道，结合轻量级3D U-Net生成提示并平滑边缘，实现高效、准确的基底节分割，用于研究儿童强迫症的炎症相关体积变化。**

- **链接: [https://arxiv.org/pdf/2511.19471v1](https://arxiv.org/pdf/2511.19471v1)**

> **作者:** Keith Moore
>
> **备注:** Preprint; Paper accepted at AIAS 2025
>
> **摘要:** Foundation segmentation models such as SAM and SAM-2 perform well on natural images but struggle with brain MRIs where structures like the caudate and thalamus lack sharp boundaries and have low contrast. Rather than fine tune these models (for example MedSAM), we propose a compositional alternative where the foundation model output is treated as an additional input channel and passed alongside the MRI to highlight regions of interest. We generate SAM-2 prompts by using a lightweight 3D U-Net that was previously trained on MRI segmentation. The U-Net may have been trained on a different dataset, so its guesses are often imprecise but usually in the correct region. The edges of the resulting foundation model guesses are smoothed to improve alignment with the MRI. We also test prompt free segmentation using DINO attention maps in the same framework. This has-a architecture avoids modifying foundation weights and adapts to domain shift without retraining the foundation model. It reaches about 96 percent volume accuracy on basal ganglia segmentation, which is sufficient for our study of longitudinal volume change. The approach is fast, label efficient, and robust to out of distribution scans. We apply it to study inflammation linked changes in sudden onset pediatric OCD.
>
---
#### [new 215] Optimization of Sums of Bivariate Functions: An Introduction to Relaxation-Based Methods for the Case of Finite Domains
- **分类: math.OC; cs.CV; stat.ML**

- **简介: 该论文研究有限域上多变量函数的优化问题，针对可表示为双变量函数之和的模型。提出基于松弛、ℓ²逼近与熵正则化的可解方法，涵盖线性规划、坐标上升等算法。通过理论分析与实验验证，揭示其在图着色、信号重建等任务中的应用潜力。**

- **链接: [https://arxiv.org/pdf/2511.20607v1](https://arxiv.org/pdf/2511.20607v1)**

> **作者:** Nils Müller
>
> **备注:** 59 pages, 7 figures
>
> **摘要:** We study the optimization of functions with $n>2$ arguments that have a representation as a sum of several functions that have only $2$ of the $n$ arguments each, termed sums of bivariates, on finite domains. The complexity of optimizing sums of bivariates is shown to be NP-equivalent and it is shown that there exists free lunch in the optimization of sums of bivariates. Based on measure-valued extensions of the objective function, so-called relaxations, $\ell^2$-approximation, and entropy-regularization, we derive several tractable problem formulations solvable with linear programming, coordinate ascent as well as with closed-form solutions. The limits of applying tractable versions of such relaxations to sums of bivariates are investigated using general results for reconstructing measures from their bivariate marginals. Experiments in which the derived algorithms are applied to random functions, vertex coloring, and signal reconstruction problems provide insights into qualitatively different function classes that can be modeled as sums of bivariates.
>
---
#### [new 216] PhysDNet: Physics-Guided Decomposition Network of Side-Scan Sonar Imagery
- **分类: physics.ao-ph; cs.CV**

- **简介: 该论文针对侧扫声呐图像因反射率、地形和衰减耦合导致视依赖性强的问题，提出PhysDNet网络，通过物理引导的多分支结构将图像分解为反射率、地形和传播损失三个可解释分量，实现自监督训练，提升图像物理一致性与下游任务性能。**

- **链接: [https://arxiv.org/pdf/2511.19539v1](https://arxiv.org/pdf/2511.19539v1)**

> **作者:** Can Lei; Hayat Rajani; Nuno Gracias; Rafael Garcia; Huigang Wang
>
> **备注:** This work was previously submitted in error as arXiv:2509.11255v2
>
> **摘要:** Side-scan sonar (SSS) imagery is widely used for seafloor mapping and underwater remote sensing, yet the measured intensity is strongly influenced by seabed reflectivity, terrain elevation, and acoustic path loss. This entanglement makes the imagery highly view-dependent and reduces the robustness of downstream analysis. In this letter, we present PhysDNet, a physics-guided multi-branch network that decouples SSS images into three interpretable fields: seabed reflectivity, terrain elevation, and propagation loss. By embedding the Lambertian reflection model, PhysDNet reconstructs sonar intensity from these components, enabling self-supervised training without ground-truth annotations. Experiments show that the decomposed representations preserve stable geological structures, capture physically consistent illumination and attenuation, and produce reliable shadow maps. These findings demonstrate that physics-guided decomposition provides a stable and interpretable domain for SSS analysis, improving both physical consistency and downstream tasks such as registration and shadow interpretation.
>
---
#### [new 217] Terminal Velocity Matching
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文提出终端速度匹配（TVM），一种用于高保真一至几步生成建模的扩散模型方法。针对传统流匹配在初始时间约束的局限，TVM改在终态正则化，证明其可上界2-Wasserstein距离。通过轻量架构改进与融合注意力核，实现稳定单阶段训练与高效计算，在ImageNet上达先进性能。**

- **链接: [https://arxiv.org/pdf/2511.19797v1](https://arxiv.org/pdf/2511.19797v1)**

> **作者:** Linqi Zhou; Mathias Parger; Ayaan Haque; Jiaming Song
>
> **备注:** Code available at: https://github.com/lumalabs/tvm
>
> **摘要:** We propose Terminal Velocity Matching (TVM), a generalization of flow matching that enables high-fidelity one- and few-step generative modeling. TVM models the transition between any two diffusion timesteps and regularizes its behavior at its terminal time rather than at the initial time. We prove that TVM provides an upper bound on the $2$-Wasserstein distance between data and model distributions when the model is Lipschitz continuous. However, since Diffusion Transformers lack this property, we introduce minimal architectural changes that achieve stable, single-stage training. To make TVM efficient in practice, we develop a fused attention kernel that supports backward passes on Jacobian-Vector Products, which scale well with transformer architectures. On ImageNet-256x256, TVM achieves 3.29 FID with a single function evaluation (NFE) and 1.99 FID with 4 NFEs. It similarly achieves 4.32 1-NFE FID and 2.94 4-NFE FID on ImageNet-512x512, representing state-of-the-art performance for one/few-step models from scratch.
>
---
#### [new 218] CostNav: A Navigation Benchmark for Cost-Aware Evaluation of Embodied Agents
- **分类: cs.AI; cs.CE; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出CostNav，一个面向具身智能体的成本感知导航基准，旨在解决现有研究仅关注任务成功率而忽视商业可行性的问题。通过构建包含硬件、能源、维护等成本与收益的经济模型，揭示了任务成功与商业盈利间的根本差异，并指出碰撞导致的维护成本是主要亏损来源，为导航算法的经济优化提供量化评估框架。**

- **链接: [https://arxiv.org/pdf/2511.20216v1](https://arxiv.org/pdf/2511.20216v1)**

> **作者:** Haebin Seong; Sungmin Kim; Minchan Kim; Yongjun Cho; Myunchul Joe; Suhwan Choi; Jaeyoon Jung; Jiyong Youn; Yoonshik Kim; Samwoo Seong; Yubeen Park; Youngjae Yu; Yunsung Lee
>
> **摘要:** Existing navigation benchmarks focus on task success metrics while overlooking economic viability -- critical for commercial deployment of autonomous delivery robots. We introduce \emph{CostNav}, a \textbf{Micro-Navigation Economic Testbed} that evaluates embodied agents through comprehensive cost-revenue analysis aligned with real-world business operations. CostNav models the complete economic lifecycle including hardware, training, energy, maintenance costs, and delivery revenue with service-level agreements, using industry-derived parameters. \textbf{To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability}, revealing that optimizing for task success fundamentally differs from optimizing for economic deployment. Our cost model uses parameters derived from industry data sources (energy rates, delivery service pricing), and we project from a reduced-scale simulation to realistic deliveries. Under this projection, the baseline achieves 43.0\% SLA compliance but is \emph{not} commercially viable: yielding a loss of \$30.009 per run with no finite break-even point, because operating costs are dominated by collision-induced maintenance, which accounts for 99.7\% of per-run costs and highlights collision avoidance as a key optimization target. We demonstrate a learning-based on-device navigation baseline and establish a foundation for evaluating rule-based navigation, imitation learning, and cost-aware RL training. CostNav bridges the gap between navigation research and commercial deployment, enabling data-driven decisions about economic trade-offs across navigation paradigms.
>
---
#### [new 219] Learning Massively Multitask World Models for Continuous Control
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文提出一种多任务世界模型Newt，解决在线强化学习在连续控制中难以规模化的问题。通过200个多样化任务的预训练与在线联合优化，实现高效多任务学习与快速适应新任务，推动通用控制发展。**

- **链接: [https://arxiv.org/pdf/2511.19584v1](https://arxiv.org/pdf/2511.19584v1)**

> **作者:** Nicklas Hansen; Hao Su; Xiaolong Wang
>
> **备注:** Webpage: https://www.nicklashansen.com/NewtWM
>
> **摘要:** General-purpose control demands agents that act across many tasks and embodiments, yet research on reinforcement learning (RL) for continuous control remains dominated by single-task or offline regimes, reinforcing a view that online RL does not scale. Inspired by the foundation model recipe (large-scale pretraining followed by light RL) we ask whether a single agent can be trained on hundreds of tasks with online interaction. To accelerate research in this direction, we introduce a new benchmark with 200 diverse tasks spanning many domains and embodiments, each with language instructions, demonstrations, and optionally image observations. We then present \emph{Newt}, a language-conditioned multitask world model that is first pretrained on demonstrations to acquire task-aware representations and action priors, and then jointly optimized with online interaction across all tasks. Experiments show that Newt yields better multitask performance and data-efficiency than a set of strong baselines, exhibits strong open-loop control, and enables rapid adaptation to unseen tasks. We release our environments, demonstrations, code for training and evaluation, as well as 200+ checkpoints.
>
---
## 更新

#### [replaced 001] Dream-IF: Dynamic Relative EnhAnceMent for Image Fusion
- **分类: cs.CV**

- **简介: 该论文针对多模态图像融合任务，解决传统方法将增强与融合分离导致质量下降的问题。提出Dream-IF框架，通过动态量化模态相对主导性，实现跨模态协同增强，并引入提示编码捕捉退化特征，提升融合与增强效果。**

- **链接: [https://arxiv.org/pdf/2503.10109v2](https://arxiv.org/pdf/2503.10109v2)**

> **作者:** Xingxin Xu; Bing Cao; Dongdong Li; Qinghua Hu; Pengfei Zhu
>
> **摘要:** Image fusion aims to integrate comprehensive information from images acquired through multiple sources. However, images captured by diverse sensors often encounter various degradations that can negatively affect fusion quality. Traditional fusion methods generally treat image enhancement and fusion as separate processes, overlooking the inherent correlation between them; notably, the dominant regions in one modality of a fused image often indicate areas where the other modality might benefit from enhancement. Inspired by this observation, we introduce the concept of dominant regions for image enhancement and present a Dynamic Relative EnhAnceMent framework for Image Fusion (Dream-IF). This framework quantifies the relative dominance of each modality across different layers and leverages this information to facilitate reciprocal cross-modal enhancement. By integrating the relative dominance derived from image fusion, our approach supports not only image restoration but also a broader range of image enhancement applications. Furthermore, we employ prompt-based encoding to capture degradation-specific details, which dynamically steer the restoration process and promote coordinated enhancement in both multi-modal image fusion and image enhancement scenarios. Extensive experimental results demonstrate that Dream-IF consistently outperforms its counterparts. The code is publicly available.\footnote{ https://github.com/jehovahxu/Dream-IF
>
---
#### [replaced 002] PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model
- **分类: cs.CV**

- **简介: 该论文提出PaddleOCR-VL，面向多语言文档解析任务，解决现有模型在多语言支持、复杂元素识别与资源消耗之间的平衡问题。通过0.9B超轻量视觉语言模型，融合动态分辨率视觉编码与高效语言模型，实现109语言下精准的文本、表格、公式等元素识别，兼具高精度与低资源开销。**

- **链接: [https://arxiv.org/pdf/2510.14528v4](https://arxiv.org/pdf/2510.14528v4)**

> **作者:** Cheng Cui; Ting Sun; Suyin Liang; Tingquan Gao; Zelun Zhang; Jiaxuan Liu; Xueqing Wang; Changda Zhou; Hongen Liu; Manhui Lin; Yue Zhang; Yubo Zhang; Handong Zheng; Jing Zhang; Jun Zhang; Yi Liu; Dianhai Yu; Yanjun Ma
>
> **备注:** Github Repo: https://github.com/PaddlePaddle/PaddleOCR
>
> **摘要:** In this report, we propose PaddleOCR-VL, a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios. Code is available at https://github.com/PaddlePaddle/PaddleOCR .
>
---
#### [replaced 003] Target-aware Image Editing via Cycle-consistent Constraints
- **分类: cs.CV**

- **简介: 该论文针对文本驱动图像编辑任务，解决现有方法中间态构建目标无关导致编辑不一致的问题。提出FlowCycle框架，通过可学习噪声参数化污染过程，并利用循环一致性约束实现目标感知的中间状态生成，提升编辑质量与源图保真度。**

- **链接: [https://arxiv.org/pdf/2510.20212v2](https://arxiv.org/pdf/2510.20212v2)**

> **作者:** Yanghao Wang; Zhen Wang; Long Chen
>
> **摘要:** Recent advances in pre-trained text-to-image flow models have enabled remarkable progress in text-based image editing. Mainstream approaches always adopt a corruption-then-restoration paradigm, where the source image is first corrupted into an ``intermediate state'' and then restored to the target image under the prompt guidance. However, current methods construct this intermediate state in a target-agnostic manner, i.e., they primarily focus on realizing source image reconstruction while neglecting the semantic gaps towards the specific editing target. This design inherently results in limited editability or inconsistency when the desired modifications substantially deviate from the source. In this paper, we argue that the intermediate state should be target-aware, i.e., selectively corrupting editing-relevant contents while preserving editing-irrelevant ones. To this end, we propose FlowCycle, a novel inversion-free and flow-based editing framework that parameterizes corruption with learnable noises and optimizes them through a cycle-consistent process. By iteratively editing the source to the target and recovering back to the source with dual consistency constraints, FlowCycle learns to produce a target-aware intermediate state, enabling faithful modifications while preserving source consistency. Extensive ablations have demonstrated that FlowCycle achieves superior editing quality and consistency over state-of-the-art methods.
>
---
#### [replaced 004] From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出政策世界模型（PWM），解决自动驾驶中世界模型与规划分离的问题。通过动作无关的未来状态预测，实现状态-动作协同预测，提升规划可靠性。引入动态并行令牌生成机制，仅用前视摄像头即达到领先性能。**

- **链接: [https://arxiv.org/pdf/2510.19654v2](https://arxiv.org/pdf/2510.19654v2)**

> **作者:** Zhida Zhao; Talas Fu; Yifan Wang; Lijun Wang; Huchuan Lu
>
> **备注:** Accepted by NuerIPS 2025 (Poster)
>
> **摘要:** Despite remarkable progress in driving world models, their potential for autonomous systems remains largely untapped: the world models are mostly learned for world simulation and decoupled from trajectory planning. While recent efforts aim to unify world modeling and planning in a single framework, the synergistic facilitation mechanism of world modeling for planning still requires further exploration. In this work, we introduce a new driving paradigm named Policy World Model (PWM), which not only integrates world modeling and trajectory planning within a unified architecture, but is also able to benefit planning using the learned world knowledge through the proposed action-free future state forecasting scheme. Through collaborative state-action prediction, PWM can mimic the human-like anticipatory perception, yielding more reliable planning performance. To facilitate the efficiency of video forecasting, we further introduce a dynamically enhanced parallel token generation mechanism, equipped with a context-guided tokenizer and an adaptive dynamic focal loss. Despite utilizing only front camera input, our method matches or exceeds state-of-the-art approaches that rely on multi-view and multi-modal inputs. Code and model weights will be released at https://github.com/6550Zhao/Policy-World-Model.
>
---
#### [replaced 005] Optimizing DINOv2 with Registers for Face Anti-Spoofing
- **分类: cs.CV**

- **简介: 该论文针对人脸识别系统中的活体检测问题，提出基于DINOv2与寄存器的伪造攻击检测方法。通过引入寄存器增强特征提取能力，抑制注意力机制中的干扰，有效识别真实与伪造人脸间的细微差异，在物理-数字攻击场景下实现高精度反欺骗检测。**

- **链接: [https://arxiv.org/pdf/2510.17201v2](https://arxiv.org/pdf/2510.17201v2)**

> **作者:** Mika Feng; Pierre Gallin-Martel; Koichi Ito; Takafumi Aoki
>
> **备注:** ICCV 2025 Workshop FAS
>
> **摘要:** Face recognition systems are designed to be robust against variations in head pose, illumination, and image blur during capture. However, malicious actors can exploit these systems by presenting a face photo of a registered user, potentially bypassing the authentication process. Such spoofing attacks must be detected prior to face recognition. In this paper, we propose a DINOv2-based spoofing attack detection method to discern minute differences between live and spoofed face images. Specifically, we employ DINOv2 with registers to extract generalizable features and to suppress perturbations in the attention mechanism, which enables focused attention on essential and minute features. We demonstrate the effectiveness of the proposed method through experiments conducted on the dataset provided by ``The 6th Face Anti-Spoofing Workshop: Unified Physical-Digital Attacks Detection@ICCV2025'' and SiW dataset. The project page is available at: https://gsisaoki.github.io/FAS-DINOv2-ICCVW/ .
>
---
#### [replaced 006] Zero-Shot Video Translation via Token Warping
- **分类: cs.CV**

- **简介: 该论文提出TokenWarping框架，用于零样本视频翻译任务。针对现有方法在时序一致性与局部结构保持上的不足，通过光流引导的查询、键、值令牌变形，增强自注意力机制中的特征聚合与时序连贯性，无需额外训练，可无缝集成现有文本到图像编辑方法，显著提升视频生成质量。**

- **链接: [https://arxiv.org/pdf/2402.12099v4](https://arxiv.org/pdf/2402.12099v4)**

> **作者:** Haiming Zhu; Yangyang Xu; Jun Yu; Shengfeng He
>
> **备注:** Code is available at: https://github.com/Alex-Zhu1/TokenWarping
>
> **摘要:** With the revolution of generative AI, video-related tasks have been widely studied. However, current state-of-the-art video models still lag behind image models in visual quality and user control over generated content. In this paper, we introduce TokenWarping, a novel framework for temporally coherent video translation. Existing diffusion-based video editing approaches rely solely on key and value patches in self-attention to ensure temporal consistency, often sacrificing the preservation of local and structural regions. Critically, these methods overlook the significance of the query patches in achieving accurate feature aggregation and temporal coherence. In contrast, TokenWarping leverages complementary token priors by constructing temporal correlations across different frames. Our method begins by extracting optical flows from source videos. During the denoising process of the diffusion model, these optical flows are used to warp the previous frame's query, key, and value patches, aligning them with the current frame's patches. By directly warping the query patches, we enhance feature aggregation in self-attention, while warping the key and value patches ensures temporal consistency across frames. This token warping imposes explicit constraints on the self-attention layer outputs, effectively ensuring temporally coherent translation. Our framework does not require any additional training or fine-tuning and can be seamlessly integrated with existing text-to-image editing methods. We conduct extensive experiments on various video translation tasks, demonstrating that TokenWarping surpasses state-of-the-art methods both qualitatively and quantitatively. Video demonstrations can be found on our project webpage: https://alex-zhu1.github.io/TokenWarping/. Code is available at: https://github.com/Alex-Zhu1/TokenWarping.
>
---
#### [replaced 007] Are Image-to-Video Models Good Zero-Shot Image Editors?
- **分类: cs.CV**

- **简介: 该论文研究视频扩散模型作为零样本图像编辑器的潜力。针对提示错位、冗余时序潜变量和后期帧模糊问题，提出IF-Edit框架，通过思维链提示增强、时序潜变量丢弃和自一致后优化，实现高效精准的指令驱动图像编辑，在推理类任务上表现优异。**

- **链接: [https://arxiv.org/pdf/2511.19435v2](https://arxiv.org/pdf/2511.19435v2)**

> **作者:** Zechuan Zhang; Zhenyuan Chen; Zongxin Yang; Yi Yang
>
> **备注:** technical report
>
> **摘要:** Large-scale video diffusion models show strong world simulation and temporal reasoning abilities, but their use as zero-shot image editors remains underexplored. We introduce IF-Edit, a tuning-free framework that repurposes pretrained image-to-video diffusion models for instruction-driven image editing. IF-Edit addresses three key challenges: prompt misalignment, redundant temporal latents, and blurry late-stage frames. It includes (1) a chain-of-thought prompt enhancement module that transforms static editing instructions into temporally grounded reasoning prompts; (2) a temporal latent dropout strategy that compresses frame latents after the expert-switch point, accelerating denoising while preserving semantic and temporal coherence; and (3) a self-consistent post-refinement step that sharpens late-stage frames using a short still-video trajectory. Experiments on four public benchmarks, covering non-rigid editing, physical and temporal reasoning, and general instruction edits, show that IF-Edit performs strongly on reasoning-centric tasks while remaining competitive on general-purpose edits. Our study provides a systematic view of video diffusion models as image editors and highlights a simple recipe for unified video-image generative reasoning.
>
---
#### [replaced 008] Segmentation-Aware Generative Reinforcement Network (GRN) for Tissue Layer Segmentation in 3-D Ultrasound Images for Chronic Low-back Pain (cLBP) Assessment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出一种用于3D超声图像组织层分割的生成式强化网络（GRN），旨在减少标注数据需求。通过引入分割感知的联合训练与分割引导增强技术，实现高效、半监督学习，显著降低标注成本，同时保持高分割精度，适用于慢性腰痛评估。**

- **链接: [https://arxiv.org/pdf/2501.17690v4](https://arxiv.org/pdf/2501.17690v4)**

> **作者:** Zixue Zeng; Xiaoyan Zhao; Matthew Cartier; Tong Yu; Jing Wang; Xin Meng; Zhiyu Sheng; Maryam Satarpour; John M Cormack; Allison Bean; Ryan Nussbaum; Maya Maurer; Emily Landis-Walkenhorst; Dinesh Kumbhare; Kang Kim; Ajay Wasan; Jiantao Pu
>
> **摘要:** We introduce a novel segmentation-aware joint training framework called generative reinforcement network (GRN) that integrates segmentation loss feedback to optimize both image generation and segmentation performance in a single stage. An image enhancement technique called segmentation-guided enhancement (SGE) is also developed, where the generator produces images tailored specifically for the segmentation model. Two variants of GRN were also developed, including GRN for sample-efficient learning (GRN-SEL) and GRN for semi-supervised learning (GRN-SSL). GRN's performance was evaluated using a dataset of 69 fully annotated 3D ultrasound scans from 29 subjects. The annotations included six anatomical structures: dermis, superficial fat, superficial fascial membrane (SFM), deep fat, deep fascial membrane (DFM), and muscle. Our results show that GRN-SEL with SGE reduces labeling efforts by up to 70% while achieving a 1.98% improvement in the Dice Similarity Coefficient (DSC) compared to models trained on fully labeled datasets. GRN-SEL alone reduces labeling efforts by 60%, GRN-SSL with SGE decreases labeling requirements by 70%, and GRN-SSL alone by 60%, all while maintaining performance comparable to fully supervised models. These findings suggest the effectiveness of the GRN framework in optimizing segmentation performance with significantly less labeled data, offering a scalable and efficient solution for ultrasound image analysis and reducing the burdens associated with data annotation.
>
---
#### [replaced 009] Advancing Limited-Angle CT Reconstruction Through Diffusion-Based Sinogram Completion
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对有限角度CT重建中因缺失投影角度导致的图像伪影问题，提出基于扩散模型的sinogram补全方法。通过MR-SDE完成投影域数据填充，并结合知识蒸馏与伪逆约束加速重建，最后通过后处理模块提升图像质量，显著改善了重建精度与视觉效果。**

- **链接: [https://arxiv.org/pdf/2505.19385v2](https://arxiv.org/pdf/2505.19385v2)**

> **作者:** Jiaqi Guo; Santiago Lopez-Tapia; Aggelos K. Katsaggelos
>
> **备注:** Accepted at the 2025 IEEE International Conference on Image Processing (Oral)
>
> **摘要:** Limited Angle Computed Tomography (LACT) often faces significant challenges due to missing angular information. Unlike previous methods that operate in the image domain, we propose a new method that focuses on sinogram inpainting. We leverage MR-SDEs, a variant of diffusion models that characterize the diffusion process with mean-reverting stochastic differential equations, to fill in missing angular data at the projection level. Furthermore, by combining distillation with constraining the output of the model using the pseudo-inverse of the inpainting matrix, the diffusion process is accelerated and done in a step, enabling efficient and accurate sinogram completion. A subsequent post-processing module back-projects the inpainted sinogram into the image domain and further refines the reconstruction, effectively suppressing artifacts while preserving critical structural details. Quantitative experimental results demonstrate that the proposed method achieves state-of-the-art performance in both perceptual and fidelity quality, offering a promising solution for LACT reconstruction in scientific and clinical applications.
>
---
#### [replaced 010] Adversarial Robustness for Unified Multi-Modal Encoders via Efficient Calibration
- **分类: cs.CV**

- **简介: 该论文研究统一多模态编码器的对抗鲁棒性问题，针对其在对抗攻击下性能显著下降的缺陷，提出无需修改预训练主干的高效校准框架。通过模态专属投影头与对抗样本训练，实现跨模态鲁棒性提升，显著增强音频、点云等非视觉模态的稳定性，同时保持原有性能。**

- **链接: [https://arxiv.org/pdf/2505.11895v2](https://arxiv.org/pdf/2505.11895v2)**

> **作者:** Chih-Ting Liao; Zhangquan Chen; Chunlei Meng; Tzu-Yu Huang; Xin Cao; Xu Zheng
>
> **摘要:** Recent unified multi-modal encoders align a wide range of modalities into a shared representation space, enabling diverse cross-modal tasks. Despite their impressive capabilities, the robustness of these models under adversarial perturbations remains underexplored, which is a critical concern for safety-sensitive applications. In this work, we present the first comprehensive study of adversarial vulnerability in unified multi-modal encoders. We find that even mild adversarial perturbations lead to substantial performance drops across all modalities. Non-visual inputs, such as audio and point clouds, are especially fragile, while visual inputs like images and videos also degrade significantly. To address this, we propose an efficient adversarial calibration framework that improves robustness across modalities without modifying pretrained encoders or semantic centers, ensuring compatibility with existing foundation models. Our method introduces modality-specific projection heads trained solely on adversarial examples, while keeping the backbone and embeddings frozen. We explore three training objectives: fixed-center cross-entropy, clean-to-adversarial L2 alignment, and clean-adversarial InfoNCE, and we introduce a regularization strategy to ensure modality-consistent alignment under attack. Experiments on six modalities and three Bind-style models show that our method improves adversarial robustness by up to 47.3 percent at epsilon = 4/255, while preserving or even improving clean zero-shot and retrieval performance with less than 1 percent trainable parameters.
>
---
#### [replaced 011] ABM-LoRA: Activation Boundary Matching for Fast Convergence in Low-Rank Adaptation
- **分类: cs.CV**

- **简介: 该论文针对低秩适配（LoRA）因随机初始化导致梯度更新偏离最优空间、收敛慢的问题，提出激活边界匹配（ABM-LoRA）初始化方法。通过对齐预训练模型与适配器的激活边界，提升梯度投影效率，减少信息损失，加速收敛。在语言理解、对话生成和视觉识别任务中均取得显著性能提升。**

- **链接: [https://arxiv.org/pdf/2511.19145v2](https://arxiv.org/pdf/2511.19145v2)**

> **作者:** Dongha Lee; Jinhee Park; Minjun Kim; Junseok Kwon
>
> **备注:** 16 pages, 5 figures, under review
>
> **摘要:** We propose Activation Boundary Matching for Low-Rank Adaptation (ABM-LoRA), a principled initialization strategy that substantially accelerates the convergence of low-rank adapters. While LoRA offers high parameter efficiency, its random initialization restricts gradient updates to a mismatched tangent space, causing significant information loss and hindering early convergence. Our ABM-LoRA addresses this by aligning the adapter's activation boundaries with those of the pretrained model before downstream training, thereby maximizing the projection of full-parameter gradients into the adapter subspace. This alignment sharply reduces information loss at initialization, yields a lower starting loss, and accelerates convergence. We demonstrate ABM-LoRA's effectiveness across diverse architectures and tasks: language understanding (T5-Base on GLUE), dialogue generation (LLaMA2-7B on WizardLM), and vision recognition (ViT-B/16 on VTAB-1K). On VTAB-1K, it achieves the highest accuracy among all methods, with strong gains on structured reasoning tasks requiring geometric understanding.
>
---
#### [replaced 012] High Resolution UDF Meshing via Iterative Networks
- **分类: cs.GR; cs.CV**

- **简介: 该论文针对高分辨率无符号距离场（UDF）网格化难题，提出一种迭代神经网络方法。传统方法在单次采样中忽略邻域信息，导致表面缺失与噪声。本文通过多轮迭代，逐步传播邻域信息，融合表面、距离与梯度特征，有效提升复杂几何的网格精度与完整性，实现高质量高分辨率网格生成。**

- **链接: [https://arxiv.org/pdf/2509.17212v2](https://arxiv.org/pdf/2509.17212v2)**

> **作者:** Federico Stella; Nicolas Talabot; Hieu Le; Pascal Fua
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Unsigned Distance Fields (UDFs) are a natural implicit representation for open surfaces but, unlike Signed Distance Fields (SDFs), are challenging to triangulate into explicit meshes. This is especially true at high resolutions where neural UDFs exhibit higher noise levels, which makes it hard to capture fine details. Most current techniques perform within single voxels without reference to their neighborhood, resulting in missing surface and holes where the UDF is ambiguous or noisy. We show that this can be remedied by performing several passes and by reasoning on previously extracted surface elements to incorporate neighborhood information. Our key contribution is an iterative neural network that does this and progressively improves surface recovery within each voxel by spatially propagating information from increasingly distant neighbors. Unlike single-pass methods, our approach integrates newly detected surfaces, distance values, and gradients across multiple iterations, effectively correcting errors and stabilizing extraction in challenging regions. Experiments on diverse 3D models demonstrate that our method produces significantly more accurate and complete meshes than existing approaches, particularly for complex geometries, enabling UDF surface extraction at higher resolutions where traditional methods fail.
>
---
#### [replaced 013] Natural Image Stitching Using Depth Maps
- **分类: cs.CV**

- **简介: 该论文属于图像拼接任务，针对非平面场景下手持拍摄因视差导致的拼接失真问题。提出基于深度图的拼接方法，通过优化特征匹配、利用对极几何建立像素对应关系，并设计模块消除映射伪影，实现高精度重叠区对齐与视角一致的非重叠区结果。**

- **链接: [https://arxiv.org/pdf/2202.06276v3](https://arxiv.org/pdf/2202.06276v3)**

> **作者:** Tianli Liao; Nan Li
>
> **备注:** accept by Signal Processing: Image Communication
>
> **摘要:** Natural image stitching aims to create a single, natural-looking mosaic from overlapped images that capture the same 3D scene from different viewing positions. Challenges inevitably arise when the scene is non-planar and captured by handheld cameras since parallax is non-negligible in such cases. In this paper, we propose a novel image stitching method using depth maps, which generates accurate alignment mosaics against parallax. Firstly, we construct a robust fitting method to filter out the outliers in feature matches and estimate the epipolar geometry between input images. Then, we utilize epipolar geometry to establish pixel-to-pixel correspondences between the input images and render the warped images using the proposed optimal warping. In the rendering stage, we introduce several modules to solve the mapping artifacts in the warping results and generate the final mosaic. Experimental results on three challenging datasets demonstrate that the depth maps of input images enable our method to provide much more accurate alignment in the overlapping region and view-consistent results in the non-overlapping region. We believe our method will continue to work under the rapid progress of monocular depth estimation. The source code is available at https://github.com/tlliao/NIS_depths.
>
---
#### [replaced 014] Orientation Matters: Making 3D Generative Models Orientation-Aligned
- **分类: cs.CV**

- **简介: 该论文针对3D生成模型方向不一致的问题，提出方向对齐的3D生成任务。构建了14,832个模型的Objaverse-OA数据集，通过微调多视图扩散与3D VAE模型，实现跨类别一致的方向生成，提升下游应用如零样本姿态估计和旋转操作的性能。**

- **链接: [https://arxiv.org/pdf/2506.08640v2](https://arxiv.org/pdf/2506.08640v2)**

> **作者:** Yichong Lu; Yuzhuo Tian; Zijin Jiang; Yikun Zhao; Yuanbo Yang; Hao Ouyang; Haoji Hu; Huimin Yu; Yujun Shen; Yiyi Liao
>
> **备注:** Accepted by NeurIPS 2025. Project Page: https://xdimlab.github.io/Orientation_Matters
>
> **摘要:** Humans intuitively perceive object shape and orientation from a single image, guided by strong priors about canonical poses. However, existing 3D generative models often produce misaligned results due to inconsistent training data, limiting their usability in downstream tasks. To address this gap, we introduce the task of orientation-aligned 3D object generation: producing 3D objects from single images with consistent orientations across categories. To facilitate this, we construct Objaverse-OA, a dataset of 14,832 orientation-aligned 3D models spanning 1,008 categories. Leveraging Objaverse-OA, we fine-tune two representative 3D generative models based on multi-view diffusion and 3D variational autoencoder frameworks to produce aligned objects that generalize well to unseen objects across various categories. Experimental results demonstrate the superiority of our method over post-hoc alignment approaches. Furthermore, we showcase downstream applications enabled by our aligned object generation, including zero-shot object orientation estimation via analysis-by-synthesis and efficient arrow-based object rotation manipulation.
>
---
#### [replaced 015] ManipShield: A Unified Framework for Image Manipulation Detection, Localization and Explanation
- **分类: cs.CV**

- **简介: 该论文针对AI生成图像操纵检测难题，提出ManipBench大尺度基准与ManipShield统一框架。解决现有方法在多样性、覆盖性和可解释性上的不足，通过多模态大模型实现检测、定位与解释一体化，显著提升泛化能力与性能。**

- **链接: [https://arxiv.org/pdf/2511.14259v2](https://arxiv.org/pdf/2511.14259v2)**

> **作者:** Zitong Xu; Huiyu Duan; Xiaoyu Wang; Zhaolin Cai; Kaiwei Zhang; Qiang Hu; Jing Liu; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** With the rapid advancement of generative models, powerful image editing methods now enable diverse and highly realistic image manipulations that far surpass traditional deepfake techniques, posing new challenges for manipulation detection. Existing image manipulation detection and localization (IMDL) benchmarks suffer from limited content diversity, narrow generative-model coverage, and insufficient interpretability, which hinders the generalization and explanation capabilities of current manipulation detection methods. To address these limitations, we introduce \textbf{ManipBench}, a large-scale benchmark for image manipulation detection and localization focusing on AI-edited images. ManipBench contains over 450K manipulated images produced by 25 state-of-the-art image editing models across 12 manipulation categories, among which 100K images are further annotated with bounding boxes, judgment cues, and textual explanations to support interpretable detection. Building upon ManipBench, we propose \textbf{ManipShield}, an all-in-one model based on a Multimodal Large Language Model (MLLM) that leverages contrastive LoRA fine-tuning and task-specific decoders to achieve unified image manipulation detection, localization, and explanation. Extensive experiments on ManipBench and several public datasets demonstrate that ManipShield achieves state-of-the-art performance and exhibits strong generality to unseen manipulation models. Both ManipBench and ManipShield will be released upon publication.
>
---
#### [replaced 016] Time-step Mixup for Efficient Spiking Knowledge Transfer from Appearance to Event Domain
- **分类: cs.CV**

- **简介: 该论文针对事件相机与脉冲神经网络中跨模态知识迁移难题，提出时间步混合（TMKT）策略，通过在不同时间步混合RGB与事件数据，并引入模态感知辅助损失，缓解模态差异，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2509.12959v2](https://arxiv.org/pdf/2509.12959v2)**

> **作者:** Yuqi Xie; Shuhan Ye; Yi Yu; Chong Wang; Qixin Zhang; Jiazhen Xu; Le Shen; Yuanbin Qian; Jiangbo Qian; Guoqi Li
>
> **摘要:** The integration of event cameras and spiking neural networks holds great promise for energy-efficient visual processing. However, the limited availability of event data and the sparse nature of DVS outputs pose challenges for effective training. Although some prior work has attempted to transfer semantic knowledge from RGB datasets to DVS, they often overlook the significant distribution gap between the two modalities. In this paper, we propose Time-step Mixup knowledge transfer (TMKT), a novel fine-grained mixing strategy that exploits the asynchronous nature of SNNs by interpolating RGB and DVS inputs at various time-steps. To enable label mixing in cross-modal scenarios, we further introduce modality-aware auxiliary learning objectives. These objectives support the time-step mixup process and enhance the model's ability to discriminate effectively across different modalities. Our approach enables smoother knowledge transfer, alleviates modality shift during training, and achieves superior performance in spiking image classification tasks. Extensive experiments demonstrate the effectiveness of our method across multiple datasets. The code will be released after the double-blind review process.
>
---
#### [replaced 017] LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preference
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LikePhys，一种无需训练的评估方法，用于衡量视频扩散模型对直观物理理解的能力。针对生成中物理正确性与视觉外观难分离的问题，利用去噪目标作为似然替代，通过对比有效与无效视频对，构建普适性偏好误差（PPE）指标，系统评估多场景下模型物理理解能力及影响因素。**

- **链接: [https://arxiv.org/pdf/2510.11512v2](https://arxiv.org/pdf/2510.11512v2)**

> **作者:** Jianhao Yuan; Fabio Pizzati; Francesco Pinto; Lars Kunze; Ivan Laptev; Paul Newman; Philip Torr; Daniele De Martini
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** Intuitive physics understanding in video diffusion models plays an essential role in building general-purpose physically plausible world simulators, yet accurately evaluating such capacity remains a challenging task due to the difficulty in disentangling physics correctness from visual appearance in generation. To the end, we introduce LikePhys, a training-free method that evaluates intuitive physics in video diffusion models by distinguishing physically valid and impossible videos using the denoising objective as an ELBO-based likelihood surrogate on a curated dataset of valid-invalid pairs. By testing on our constructed benchmark of twelve scenarios spanning over four physics domains, we show that our evaluation metric, Plausibility Preference Error (PPE), demonstrates strong alignment with human preference, outperforming state-of-the-art evaluator baselines. We then systematically benchmark intuitive physics understanding in current video diffusion models. Our study further analyses how model design and inference settings affect intuitive physics understanding and highlights domain-specific capacity variations across physical laws. Empirical results show that, despite current models struggling with complex and chaotic dynamics, there is a clear trend of improvement in physics understanding as model capacity and inference settings scale.
>
---
#### [replaced 018] FAPE-IR: Frequency-Aware Planning and Execution Framework for All-in-One Image Restoration
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多类型图像退化下的统一修复任务，提出FAPE-IR框架。通过冻结的多模态大模型生成频率感知修复计划，指导基于LoRA-MoE的扩散执行器动态选择高低频专家，结合对抗训练与频率正则化，实现高效、可解释的端到端修复，显著提升复杂退化场景下的性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.14099v2](https://arxiv.org/pdf/2511.14099v2)**

> **作者:** Jingren Liu; Shuning Xu; Qirui Yang; Yun Wang; Xiangyu Chen; Zhong Ji
>
> **摘要:** All-in-One Image Restoration (AIO-IR) aims to develop a unified model that can handle multiple degradations under complex conditions. However, existing methods often rely on task-specific designs or latent routing strategies, making it hard to adapt to real-world scenarios with various degradations. We propose FAPE-IR, a Frequency-Aware Planning and Execution framework for image restoration. It uses a frozen Multimodal Large Language Model (MLLM) as a planner to analyze degraded images and generate concise, frequency-aware restoration plans. These plans guide a LoRA-based Mixture-of-Experts (LoRA-MoE) module within a diffusion-based executor, which dynamically selects high- or low-frequency experts, complemented by frequency features of the input image. To further improve restoration quality and reduce artifacts, we introduce adversarial training and a frequency regularization loss. By coupling semantic planning with frequency-based restoration, FAPE-IR offers a unified and interpretable solution for all-in-one image restoration. Extensive experiments show that FAPE-IR achieves state-of-the-art performance across seven restoration tasks and exhibits strong zero-shot generalization under mixed degradations.
>
---
#### [replaced 019] VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大模型（MLLMs）在视频理解中对复杂编排内容认知能力不足的问题，提出VidComposition基准。通过982个精心制作的视频与1706道多选题，评估模型对镜头语言、叙事结构等组合层面的理解。实验揭示当前模型与人类存在显著差距，为后续研究提供方向。**

- **链接: [https://arxiv.org/pdf/2411.10979v5](https://arxiv.org/pdf/2411.10979v5)**

> **作者:** Yolo Y. Tang; Junjia Guo; Hang Hua; Susan Liang; Mingqian Feng; Xinyang Li; Rui Mao; Chao Huang; Jing Bi; Zeliang Zhang; Pooyan Fazli; Chenliang Xu
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** The advancement of Multimodal Large Language Models (MLLMs) has enabled significant progress in multimodal understanding, expanding their capacity to analyze video content. However, existing evaluation benchmarks for MLLMs primarily focus on abstract video comprehension, lacking a detailed assessment of their ability to understand video compositions, the nuanced interpretation of how visual elements combine and interact within highly compiled video contexts. We introduce VidComposition, a new benchmark specifically designed to evaluate the video composition understanding capabilities of MLLMs using carefully curated compiled videos and cinematic-level annotations. VidComposition includes 982 videos with 1706 multiple-choice questions, covering various compositional aspects such as camera movement, angle, shot size, narrative structure, character actions and emotions, etc. Our comprehensive evaluation of 33 open-source and proprietary MLLMs reveals a significant performance gap between human and model capabilities. This highlights the limitations of current MLLMs in understanding complex, compiled video compositions and offers insights into areas for further improvement. The leaderboard and evaluation code are available at https://yunlong10.github.io/VidComposition/
>
---
#### [replaced 020] A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding
- **分类: cs.CV**

- **简介: 该论文针对无人机视角下异常理解任务，提出A2Seek基准数据集与A2Seek-R1推理框架。解决现有方法在动态视角、复杂场景下性能下降问题，通过多粒度标注、图思维微调与飞行模拟注意力机制，显著提升异常检测与定位精度，实现更深入的因果推理。**

- **链接: [https://arxiv.org/pdf/2505.21962v2](https://arxiv.org/pdf/2505.21962v2)**

> **作者:** Mengjingcheng Mo; Xinyang Tong; Mingpi Tan; Jiaxu Leng; Jiankang Zheng; Yiran Liu; Haosheng Chen; Ji Gan; Weisheng Li; Xinbo Gao
>
> **摘要:** While unmanned aerial vehicles (UAVs) offer wide-area, high-altitude coverage for anomaly detection, they face challenges such as dynamic viewpoints, scale variations, and complex scenes. Existing datasets and methods, mainly designed for fixed ground-level views, struggle to adapt to these conditions, leading to significant performance drops in drone-view scenarios. To bridge this gap, we introduce A2Seek (Aerial Anomaly Seek), a large-scale, reasoning-centric benchmark dataset for aerial anomaly understanding. This dataset covers various scenarios and environmental conditions, providing high-resolution real-world aerial videos with detailed annotations, including anomaly categories, frame-level timestamps, region-level bounding boxes, and natural language explanations for causal reasoning. Building on this dataset, we propose A2Seek-R1, a novel reasoning framework that generalizes R1-style strategies to aerial anomaly understanding, enabling a deeper understanding of "Where" anomalies occur and "Why" they happen in aerial frames. To this end, A2Seek-R1 first employs a graph-of-thought (GoT)-guided supervised fine-tuning approach to activate the model's latent reasoning capabilities on A2Seek. Then, we introduce Aerial Group Relative Policy Optimization (A-GRPO) to design rule-based reward functions tailored to aerial scenarios. Furthermore, we propose a novel "seeking" mechanism that simulates UAV flight behavior by directing the model's attention to informative regions. Extensive experiments demonstrate that A2Seek-R1 achieves up to a 22.04% improvement in AP for prediction accuracy and a 13.9% gain in mIoU for anomaly localization, exhibiting strong generalization across complex environments and out-of-distribution scenarios. Our dataset and code are released at https://2-mo.github.io/A2Seek/.
>
---
#### [replaced 021] Beyond Fully Supervised Pixel Annotations: Scribble-Driven Weakly-Supervised Framework for Image Manipulation Localization
- **分类: cs.CV**

- **简介: 该论文针对图像篡改定位任务，解决依赖大量像素标注导致的标注成本高问题。提出基于草图标注的弱监督框架，构建首个草图标注数据集，设计自监督学习、先验感知调制与门控融合模块，提升模型在弱监督下的定位性能。**

- **链接: [https://arxiv.org/pdf/2507.13018v2](https://arxiv.org/pdf/2507.13018v2)**

> **作者:** Songlin Li; Guofeng Yu; Zhiqing Guo; Yunfeng Diao; Dan Ma; Gaobo Yang
>
> **摘要:** Deep learning-based image manipulation localization (IML) methods have achieved remarkable performance in recent years, but typically rely on large-scale pixel-level annotated datasets. To address the challenge of acquiring high-quality annotations, some recent weakly supervised methods utilize image-level labels to segment manipulated regions. However, the performance is still limited due to insufficient supervision signals. In this study, we explore a form of weak supervision that improves the annotation efficiency and detection performance, namely scribble annotation supervision. We re-annotate mainstream IML datasets with scribble labels and propose the first scribble-based IML (Sc-IML) dataset. Additionally, we propose the first scribble-based weakly supervised IML framework. Specifically, we employ self-supervised training with a structural consistency loss to encourage the model to produce consistent predictions under multi-scale and augmented inputs. In addition, we propose a prior-aware feature modulation module (PFMM) that adaptively integrates prior information from both manipulated and authentic regions for dynamic feature adjustment, further enhancing feature discriminability and prediction consistency in complex scenes. We also propose a gated adaptive fusion module (GAFM) that utilizes gating mechanisms to regulate information flow during feature fusion, guiding the model toward emphasizing potential manipulated regions. Finally, we propose a confidence-aware entropy minimization loss (${\mathcal{L}}_{ {CEM }}$). This loss dynamically regularizes predictions in weakly annotated or unlabeled regions based on model uncertainty, effectively suppressing unreliable predictions. Experimental results show that our method outperforms existing fully supervised approaches in terms of average performance both in-distribution and out-of-distribution.
>
---
#### [replaced 022] GigaBrain-0: A World Model-Powered Vision-Language-Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GigaBrain-0，一种基于世界模型生成数据的视觉-语言-动作（VLA）基础模型，旨在解决通用机器人训练中真实数据收集成本高、可扩展性差的问题。通过生成多样化仿真数据，减少对真实数据依赖，提升跨任务泛化与政策鲁棒性，尤其在复杂操作任务中表现优异。**

- **链接: [https://arxiv.org/pdf/2510.19430v2](https://arxiv.org/pdf/2510.19430v2)**

> **作者:** GigaBrain Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jie Li; Jiagang Zhu; Lv Feng; Peng Li; Qiuping Deng; Runqi Ouyang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yang Wang; Yifan Li; Yilong Li; Yiran Ding; Yuan Xu; Yun Ye; Yukun Zhou; Zhehao Dong; Zhenan Wang; Zhichao Liu; Zheng Zhu
>
> **备注:** https://gigabrain0.github.io/
>
> **摘要:** Training Vision-Language-Action (VLA) models for generalist robots typically requires large-scale real-world robot data, which is expensive and time-consuming to collect. The inefficiency of physical data collection severely limits the scalability, and generalization capacity of current VLA systems. To address this challenge, we introduce GigaBrain-0, a novel VLA foundation model empowered by world model-generated data (e.g., video generation, real2real transfer, human transfer, view transfer, sim2real transfer data). By leveraging world models to generate diverse data at scale, GigaBrain-0 significantly reduces reliance on real robot data while improving cross-task generalization. Our approach further improves policy robustness through RGBD input modeling and embodied Chain-of-Thought (CoT) supervision, enabling the model to reason about spatial geometry, object states, and long-horizon dependencies during task execution. This leads to substantial gains in real-world performance on dexterous, long-horizon, and mobile manipulation tasks. Extensive experiments demonstrate that GigaBrain-0 achieves superior generalization across variations in appearances (e.g., textures, colors), object placements, and camera viewpoints. Additionally, we present GigaBrain-0-Small, an optimized lightweight variant designed to run efficiently on devices such as the NVIDIA Jetson AGX Orin.
>
---
#### [replaced 023] E$^{3}$NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images
- **分类: cs.CV**

- **简介: 该论文针对从模糊图像重建清晰神经辐射场（NeRF）的任务，提出E³NeRF框架。通过融合模糊图像与事件流，引入模糊渲染损失和事件渲染损失，利用事件流中的时空信息分离并优化时空模糊，提升重建效率与质量，尤其在高速非均匀运动和低光场景下表现优异。**

- **链接: [https://arxiv.org/pdf/2408.01840v2](https://arxiv.org/pdf/2408.01840v2)**

> **作者:** Yunshan Qi; Jia Li; Yifan Zhao; Yu Zhang; Lin Zhu
>
> **摘要:** Neural Radiance Fields (NeRF) achieves impressive novel view rendering performance by learning implicit 3D representation from sparse view images. However, it is difficult to reconstruct a sharp NeRF from blurry input that often occurs in the wild. To solve this problem, we propose a novel Efficient Event-Enhanced NeRF (E$^{3}$NeRF), reconstructing sharp NeRF by utilizing both blurry images and corresponding event streams. A blur rendering loss and an event rendering loss are introduced, which guide the NeRF training via modeling the physical image motion blur process and event generation process, respectively. To improve the efficiency of the framework, we further leverage the latent spatial-temporal blur information in the event stream to evenly distribute training over temporal blur and focus training on spatial blur. Moreover, a camera pose estimation framework for real-world data is built with the guidance of the events, generalizing the method to more practical applications. Compared to previous image-based and event-based NeRF works, our framework makes more profound use of the internal relationship between events and images. Extensive experiments on both synthetic data and real-world data demonstrate that E\textsuperscript{3}NeRF can effectively learn a sharp NeRF from blurry images, especially for high-speed non-uniform motion and low-light scenes.
>
---
#### [replaced 024] Ming-Flash-Omni: A Sparse, Unified Architecture for Multimodal Perception and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Ming-Flash-Omni，一种稀疏统一的多模态架构，解决高效多模态感知与生成难题。基于稀疏MoE的1000亿参数模型，仅61亿激活，显著提升计算效率与模型容量。实现顶尖的跨模态理解与生成能力，涵盖语音识别、图像生成与编辑、生成分割等任务，统一架构下达成多项性能突破。**

- **链接: [https://arxiv.org/pdf/2510.24821v2](https://arxiv.org/pdf/2510.24821v2)**

> **作者:** Inclusion AI; :; Bowen Ma; Cheng Zou; Canxiang Yan; Chunxiang Jin; Chunjie Shen; Chenyu Lian; Dandan Zheng; Fudong Wang; Furong Xu; GuangMing Yao; Jun Zhou; Jingdong Chen; Jianing Li; Jianxin Sun; Jiajia Liu; Jian Sha; Jianjiang Zhu; Jianping Jiang; Jun Peng; Kaixiang Ji; Kaimeng Ren; Libin Wang; Lixiang Ru; Longhua Tan; Lu Ma; Lan Wang; Mochen Bai; Ning Gao; Qingpei Guo; Qinglong Zhang; Qiang Xu; Rui Liu; Ruijie Xiong; Ruobing Zheng; Sirui Gao; Tao Zhang; Tianqi Li; Tinghao Liu; Weilong Chai; Xinyu Xiao; Xiaomei Wang; Xiaolong Wang; Xiao Lu; Xiaoyu Li; Xingning Dong; Xuzheng Yu; Yi Yuan; Yuting Gao; Yuting Xiao; Yunxiao Sun; Yipeng Chen; Yifan Mao; Yifei Wu; Yongjie Lyu; Ziping Ma; Zhiqiang Fang; Zhihao Qiu; Ziyuan Huang; Zizheng Yang; Zhengyu He
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** We propose Ming-Flash-Omni, an upgraded version of Ming-Omni, built upon a sparser Mixture-of-Experts (MoE) variant of Ling-Flash-2.0 with 100 billion total parameters, of which only 6.1 billion are active per token. This architecture enables highly efficient scaling (dramatically improving computational efficiency while significantly expanding model capacity) and empowers stronger unified multimodal intelligence across vision, speech, and language, representing a key step toward Artificial General Intelligence (AGI). Compared to its predecessor, the upgraded version exhibits substantial improvements across multimodal understanding and generation. We significantly advance speech recognition capabilities, achieving state-of-the-art performance in contextual ASR and highly competitive results in dialect-aware ASR. In image generation, Ming-Flash-Omni introduces high-fidelity text rendering and demonstrates marked gains in scene consistency and identity preservation during image editing. Furthermore, Ming-Flash-Omni introduces generative segmentation, a capability that not only achieves strong standalone segmentation performance but also enhances spatial control in image generation and improves editing consistency. Notably, Ming-Flash-Omni achieves state-of-the-art results in text-to-image generation and generative segmentation, and sets new records on all 12 contextual ASR benchmarks, all within a single unified architecture.
>
---
#### [replaced 025] Stitch-a-Demo: Video Demonstrations from Multistep Descriptions
- **分类: cs.CV**

- **简介: 该论文提出Stitch-a-Demo，解决多步骤文本描述生成连贯视频演示的任务。针对现有方法无法处理多步指令导致视频不连贯的问题，提出基于检索的视频组装方法，通过弱监督训练和难例挖掘，实现准确且视觉一致的视频生成，在真实教学视频上表现优异。**

- **链接: [https://arxiv.org/pdf/2503.13821v2](https://arxiv.org/pdf/2503.13821v2)**

> **作者:** Chi Hsuan Wu; Kumar Ashutosh; Kristen Grauman
>
> **摘要:** When obtaining visual illustrations from text descriptions, today's methods take a description with a single text context - a caption, or an action description - and retrieve or generate the matching visual context. However, prior work does not permit visual illustration of multistep descriptions, e.g. a cooking recipe or a gardening instruction manual, and simply handling each step description in isolation would result in an incoherent demonstration. We propose Stitch-a-Demo, a novel retrieval-based method to assemble a video demonstration from a multistep description. The resulting video contains clips, possibly from different sources, that accurately reflect all the step descriptions, while being visually coherent. We formulate a training pipeline that creates large-scale weakly supervised data containing diverse procedures and injects hard negatives that promote both correctness and coherence. Validated on in-the-wild instructional videos, Stitch-a-Demo achieves state-of-the-art performance, with gains up to 29% as well as dramatic wins in a human preference study.
>
---
#### [replaced 026] Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation
- **分类: cs.CV**

- **简介: 该论文针对生成视频中身份一致性的难题，提出轻量级可插拔框架Stand-In。通过引入条件图像分支与受限自注意力机制，仅用约1%额外参数和2000对数据，实现高质量视频生成与身份精准控制，兼容多种AIGC任务。**

- **链接: [https://arxiv.org/pdf/2508.07901v3](https://arxiv.org/pdf/2508.07901v3)**

> **作者:** Bowen Xue; Zheng-Peng Duan; Qixin Yan; Wenjing Wang; Hao Liu; Chun-Le Guo; Chongyi Li; Chen Li; Jing Lyu
>
> **摘要:** Generating high-fidelity human videos that match user-specified identities is important yet challenging in the field of generative AI. Existing methods often rely on an excessive number of training parameters and lack compatibility with other AIGC tools. In this paper, we propose Stand-In, a lightweight and plug-and-play framework for identity preservation in video generation. Specifically, we introduce a conditional image branch into the pre-trained video generation model. Identity control is achieved through restricted self-attentions with conditional position mapping. Thanks to these designs, which greatly preserve the pre-trained prior of the video generation model, our approach is able to outperform other full-parameter training methods in video quality and identity preservation, even with just $\sim$1% additional parameters and only 2000 training pairs. Moreover, our framework can be seamlessly integrated for other tasks, such as subject-driven video generation, pose-referenced video generation, stylization, and face swapping.
>
---
#### [replaced 027] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLMs）参数高效合并难题，提出RobustMerge方法。通过保持方向鲁棒性，实现无需训练的高效模型融合，提升跨任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2502.17159v4](https://arxiv.org/pdf/2502.17159v4)**

> **作者:** Fanhu Zeng; Haiyang Guo; Fei Zhu; Li Shen; Hao Tang
>
> **备注:** NeurIPS 2025 (Spotlight) Fix some typos
>
> **摘要:** Fine-tuning pre-trained models with custom data leads to numerous expert models on specific tasks. Merging models into one universal model to empower multi-task ability refraining from data leakage has gained popularity. With the expansion in data and model size, parameter-efficient tuning becomes the common practice for obtaining task-specific models efficiently. However, few methods are dedicated to efficient merging, and existing methods designed for full fine-tuning merging fail under efficient merging. To address the issue, we analyze from low-rank decomposition and reveal that direction robustness during merging is crucial for merging efficient modules. We furthermore uncover that compensating for the gap between stark singular values contributes to direction robustness. Therefore, we propose RobustMerge, a training-free parameter-efficient merging method with complementary parameter adaptation to maintain direction robustness. Specifically, we (1) prune parameters and scale coefficients from inter-parameter relation for singular values to maintain direction stability away from task interference, and (2) perform cross-task normalization to enhance unseen task generalization. We establish a benchmark consisting of diverse multimodal tasks, on which we conduct experiments to certify the outstanding performance and generalizability of our method. Additional studies and extensive analyses further showcase the effectiveness. Code is available at https://github.com/AuroraZengfh/RobustMerge.
>
---
#### [replaced 028] LoRA-based methods on Unet for transfer learning in Subarachnoid Hematoma Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对动脉瘤性蛛网膜下腔出血（SAH）的医学图像分割任务，解决小样本下模型性能不足问题。提出基于CP-LoRA和DoRA的参数高效微调方法，在Unet架构上实现跨血肿类型迁移学习，显著优于传统微调，且参数更少，效果更优。**

- **链接: [https://arxiv.org/pdf/2508.01772v3](https://arxiv.org/pdf/2508.01772v3)**

> **作者:** Cristian Minoccheri; Matthew Hodgman; Haoyuan Ma; Rameez Merchant; Emily Wittrup; Craig Williamson; Kayvan Najarian
>
> **摘要:** Aneurysmal subarachnoid hemorrhage (SAH) is a life-threatening neurological emergency with mortality rates exceeding 30%. Transfer learning from related hematoma types represents a potentially valuable but underexplored approach. Although Unet architectures remain the gold standard for medical image segmentation due to their effectiveness on limited datasets, Low-Rank Adaptation (LoRA) methods for parameter-efficient transfer learning have been rarely applied to convolutional neural networks in medical imaging contexts. We implemented a Unet architecture pre-trained on computed tomography scans from 124 traumatic brain injury patients across multiple institutions, then fine-tuned on 30 aneurysmal SAH patients from the University of Michigan Health System using 3-fold cross-validation. We developed a novel CP-LoRA method based on tensor CP-decomposition and introduced DoRA variants (DoRA-C, convDoRA, CP-DoRA) that decompose weight matrices into magnitude and directional components. We compared these approaches against existing LoRA methods (LoRA-C, convLoRA) and standard fine-tuning strategies across different modules on a multi-view Unet model. LoRA-based methods consistently outperformed standard Unet fine-tuning. Performance varied by hemorrhage volume, with all methods showing improved accuracy for larger volumes. CP-LoRA achieved comparable performance to existing methods while using significantly fewer parameters. Over-parameterization with higher ranks consistently yielded better performance than strictly low-rank adaptations. This study demonstrates that transfer learning between hematoma types is feasible and that LoRA-based methods significantly outperform conventional Unet fine-tuning for aneurysmal SAH segmentation.
>
---
#### [replaced 029] Can Modern Vision Models Understand the Difference Between an Object and a Look-alike?
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型是否能区分真实物体与外观相似物。针对人类具备但模型欠缺的“形似非实”判断能力，构建了RoLA数据集，通过对比真实与仿制品样本，探索CLIP模型在嵌入空间中区分二者的能力，并利用方向向量提升跨模态检索与图像描述性能。**

- **链接: [https://arxiv.org/pdf/2511.19200v2](https://arxiv.org/pdf/2511.19200v2)**

> **作者:** Itay Cohen; Ethan Fetaya; Amir Rosenfeld
>
> **摘要:** Recent advances in computer vision have yielded models with strong performance on recognition benchmarks; however, significant gaps remain in comparison to human perception. One subtle ability is to judge whether an image looks like a given object without being an instance of that object. We study whether vision-language models such as CLIP capture this distinction. We curated a dataset named RoLA (Real or Lookalike) of real and lookalike exemplars (e.g., toys, statues, drawings, pareidolia) across multiple categories, and first evaluate a prompt-based baseline with paired "real"/"lookalike" prompts. We then estimate a direction in CLIP's embedding space that moves representations between real and lookalike. Applying this direction to image and text embeddings improves discrimination in cross-modal retrieval on Conceptual12M, and also enhances captions produced by a CLIP prefix captioner.
>
---
#### [replaced 030] RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows
- **分类: cs.MA; cs.CL; cs.CV**

- **简介: 该论文针对胸部X光片（CXR）解读中推理不可解释、多模态信息融合不足、工具间矛盾无法解决等问题，提出RadAgents框架。通过模拟放射科医生工作流程，整合多模态推理与视觉-文本对齐的验证机制，实现可审计、一致且临床可信的智能诊断。**

- **链接: [https://arxiv.org/pdf/2509.20490v2](https://arxiv.org/pdf/2509.20490v2)**

> **作者:** Kai Zhang; Corey D Barrett; Jangwon Kim; Lichao Sun; Tara Taghavi; Krishnaram Kenthapadi
>
> **备注:** ML4H'25; Work in progress
>
> **摘要:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework that couples clinical priors with task-aware multimodal reasoning and encodes a radiologist-style workflow into a modular, auditable pipeline. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.
>
---
#### [replaced 031] FaVChat: Hierarchical Prompt-Query Guided Facial Video Understanding with Data-Efficient GRPO
- **分类: cs.CV**

- **简介: 该论文针对视频理解中细粒度面部特征识别困难的问题，提出FaVChat模型。通过多层级提示引导特征提取与数据高效强化学习（GRPO），提升对细微表情、情绪等的感知能力，显著增强模型在少量数据下的表现，实现更精准的细粒度面部视频理解。**

- **链接: [https://arxiv.org/pdf/2503.09158v3](https://arxiv.org/pdf/2503.09158v3)**

> **作者:** Fufangchen Zhao; Xuerui Qiu; Linrui Xu; Ming Li; Wenhao Jiang; Jinkai Zheng; Hehe Fan; Jian Gao; Danfeng Yan
>
> **摘要:** Multi-modal large language models (MLLMs) have shown strong capability in video understanding but still struggle with fine-grained visual comprehension, as pure visual encoders often lose subtle cues essential for precise reasoning. To address this limitation, we propose FaVChat, a Video-MLLM specifically designed for fine-grained facial understanding. FaVChat introduces a multi-level prompt-guided feature extraction mechanism that progressively captures task-relevant information from three complementary stages: low-level transformer layers for textures and motion, medium-level learnable queries for discriminative regions, and high-level adaptive feature weighting for semantic alignment. These enriched features are dynamically fused and fed into the LLM to enable more accurate fine-grained reasoning. To further enhance the model's ability to capture fine-grained facial attributes and maximize the utility of limited data, we propose Date-Efficient GRPO, a novel data-efficient reinforcement learning (RL) algorithm that maximizes the utility of each training sample through per-instance utility estimation and dynamic lifecycle scheduling. Extensive zero-shot evaluations across emotion recognition, explainable reasoning, and textual expression analysis demonstrate that FaVChat achieves finer-grained understanding, stronger accuracy, and better generalization than existing Video-MLLMs, even when trained with only 10K RL samples.
>
---
#### [replaced 032] OceanGym: A Benchmark Environment for Underwater Embodied Agents
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出OceanGym，首个面向水下具身智能体的综合性基准环境，旨在解决水下感知与决策难题。通过整合多模态大模型，构建涵盖八类任务的仿真平台，推动智能体在低可见度、强流等复杂环境下实现自主探索与长程目标达成，填补了水下智能研究空白。**

- **链接: [https://arxiv.org/pdf/2509.26536v2](https://arxiv.org/pdf/2509.26536v2)**

> **作者:** Yida Xue; Mingjun Mao; Xiangyuan Ru; Yuqi Zhu; Baochang Ren; Shuofei Qiao; Mengru Wang; Shumin Deng; Xinyu An; Ningyu Zhang; Ying Chen; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and adaptability in ocean underwater environments. By providing a high-fidelity, rigorously designed platform, OceanGym establishes a testbed for developing robust embodied AI and transferring these capabilities to real-world autonomous ocean underwater vehicles, marking a decisive step toward intelligent agents capable of operating in one of Earth's last unexplored frontiers. The code and data are available at https://github.com/OceanGPT/OceanGym.
>
---
#### [replaced 033] GMT: Effective Global Framework for Multi-Camera Multi-Target Tracking
- **分类: cs.CV**

- **简介: 该论文针对多摄像头多目标跟踪任务，解决传统两阶段框架中多视角信息利用不足的问题。提出GMT全局框架，通过跨视角特征对齐与全局轨迹关联，实现端到端的多视图联合跟踪，显著提升精度，并构建了大规模高质量数据集VisionTrack。**

- **链接: [https://arxiv.org/pdf/2407.01007v2](https://arxiv.org/pdf/2407.01007v2)**

> **作者:** Yihao Zhen; Mingyue Xu; Qiang Wang; Baojie Fan; Jiahua Dong; Tinghui Zhao; Huijie Fan
>
> **摘要:** Multi-Camera Multi-Target (MCMT) tracking aims to locate and associate the same targets across multiple camera views. Existing methods typically adopt a two-stage framework, involving single-camera tracking followed by inter-camera tracking. However, in this paradigm, multi-view information is used only to recover missed matches in the first stage, providing a limited contribution to overall tracking. To address this issue, we propose GMT, a global MCMT tracking framework that jointly exploits intra-view and inter-view cues for tracking. Specifically, instead of assigning trajectories independently for each view, we integrate the same historical targets across different views as global trajectories, thereby reformulating the two-stage tracking as a unified global-level trajectory-target association process. We introduce a Cross-View Feature Consistency Enhancement (CFCE) module to align visual and spatial features across views, providing a consistent feature space for global trajectory modeling. With these aligned features, the Global Trajectory Association (GTA) module associates new detections with existing global trajectories, enabling direct use of multi-view information. Compared to the two-stage framework, GMT achieves significant improvements on existing datasets, with gains of up to 21.3 percent in CVMA and 17.2 percent in CVIDF1. Furthermore, we introduce VisionTrack, a high-quality, large-scale MCMT dataset providing significantly greater diversity than existing datasets. Our code and dataset will be released.
>
---
#### [replaced 034] Achieving detailed medial temporal lobe segmentation with upsampled isotropic training from implicit neural representation
- **分类: cs.CV**

- **简介: 该论文针对阿尔茨海默病研究中海马旁回亚区分割难题，提出基于隐式神经表示的多模态图像上采样方法，将各向异性T2w MRI与各向同性T1w MRI融合，构建高分辨率训练数据，提升nnU-Net模型对MTL亚区的分割精度。实验表明，该方法显著增强了形态学指标的区分度与稳定性，无需额外标注即可提高生物标志物可靠性。**

- **链接: [https://arxiv.org/pdf/2508.17171v2](https://arxiv.org/pdf/2508.17171v2)**

> **作者:** Yue Li; Pulkit Khandelwal; Rohit Jena; Long Xie; Michael Duong; Amanda E. Denning; Christopher A. Brown; Laura E. M. Wisse; Sandhitsu R. Das; David A. Wolk; Paul A. Yushkevich
>
> **摘要:** Imaging biomarkers in magnetic resonance imaging (MRI) are important tools for diagnosing, tracking and treating Alzheimer's disease (AD). Neurofibrillary tau pathology in AD is closely linked to neurodegeneration and generally follows a pattern of spread in the brain, with early stages involving subregions of the medial temporal lobe (MTL). Accurate segmentation of MTL subregions is needed to extract granular biomarkers of AD progression. MTL subregions are often imaged using T2-weighted (T2w) MRI scans that are highly anisotropic due to constraints of MRI physics and image acquisition, making it difficult to reliably model MTL subregions geometrically and extract morphological measures, such as thickness. In this study, we used an implicit neural representation method to combine isotropic T1-weighted (T1w) and anisotropic T2w MRI to upsample an atlas set of expert-annotated MTL subregions, establishing a multi-modality, high-resolution training set of isotropic data for automatic segmentation with the nnU-Net framework. In an independent test set, the morphological measures extracted using this isotropic model showed stronger effect sizes than models trained on anisotropic in distinguishing participants with mild cognitive impairment (MCI) and cognitively unimpaired individuals. In test-retest analysis, morphological measures extracted using the isotropic model had greater stability. This study demonstrates improved reliability of MRI-derived MTL subregion biomarkers without additional atlas annotation effort, which may more accurately quantify and track the relationship between AD pathology and brain atrophy for monitoring disease progression.
>
---
#### [replaced 035] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution
- **分类: cs.CV; physics.ao-ph**

- **简介: 该论文提出Cloud4D，一个基于深度学习的四维云场重建框架，利用同步地基相机实现25米空间、5秒时间分辨率的三维液态水含量反演，并估计水平风场。解决了高分辨率云观测难题，显著提升时空分辨率与精度，优于现有卫星数据。**

- **链接: [https://arxiv.org/pdf/2511.19431v2](https://arxiv.org/pdf/2511.19431v2)**

> **作者:** Jacob Lin; Edward Gryspeerdt; Ronald Clark
>
> **备注:** NeurIPS 2025 Spotlight, project page: https://cloud4d.jacob-lin.com/
>
> **摘要:** There has been great progress in improving numerical weather prediction and climate models using machine learning. However, most global models act at a kilometer-scale, making it challenging to model individual clouds and factors such as extreme precipitation, wind gusts, turbulence, and surface irradiance. Therefore, there is a need to move towards higher-resolution models, which in turn require high-resolution real-world observations that current instruments struggle to obtain. We present Cloud4D, the first learning-based framework that reconstructs a physically consistent, four-dimensional cloud state using only synchronized ground-based cameras. Leveraging a homography-guided 2D-to-3D transformer, Cloud4D infers the full 3D distribution of liquid water content at 25 m spatial and 5 s temporal resolution. By tracking the 3D liquid water content retrievals over time, Cloud4D additionally estimates horizontal wind vectors. Across a two-month deployment comprising six skyward cameras, our system delivers an order-of-magnitude improvement in space-time resolution relative to state-of-the-art satellite measurements, while retaining single-digit relative error ($<10\%$) against collocated radar measurements. Code and data are available on our project page https://cloud4d.jacob-lin.com/.
>
---
#### [replaced 036] ConceptGuard: Proactive Safety in Text-and-Image-to-Video Generation through Multimodal Risk Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本与图像生成视频中的多模态安全风险问题，提出ConceptGuard框架。通过对比检测与语义抑制，实现对潜在危险内容的主动识别与规避。构建了ConceptRisk数据集和T2VSafetyBench-TI2V基准，验证其在风险检测与安全生成上的优越性。**

- **链接: [https://arxiv.org/pdf/2511.18780v2](https://arxiv.org/pdf/2511.18780v2)**

> **作者:** Ruize Ma; Minghong Cai; Yilei Jiang; Jiaming Han; Yi Feng; Yingshui Tan; Xiaoyong Zhu; Bo Zhang; Bo Zheng; Xiangyu Yue
>
> **摘要:** Recent progress in video generative models has enabled the creation of high-quality videos from multimodal prompts that combine text and images. While these systems offer enhanced controllability, they also introduce new safety risks, as harmful content can emerge from individual modalities or their interaction. Existing safety methods are often text-only, require prior knowledge of the risk category, or operate as post-generation auditors, struggling to proactively mitigate such compositional, multimodal risks. To address this challenge, we present ConceptGuard, a unified safeguard framework for proactively detecting and mitigating unsafe semantics in multimodal video generation. ConceptGuard operates in two stages: First, a contrastive detection module identifies latent safety risks by projecting fused image-text inputs into a structured concept space; Second, a semantic suppression mechanism steers the generative process away from unsafe concepts by intervening in the prompt's multimodal conditioning. To support the development and rigorous evaluation of this framework, we introduce two novel benchmarks: ConceptRisk, a large-scale dataset for training on multimodal risks, and T2VSafetyBench-TI2V, the first benchmark adapted from T2VSafetyBench for the Text-and-Image-to-Video (TI2V) safety setting. Comprehensive experiments on both benchmarks show that ConceptGuard consistently outperforms existing baselines, achieving state-of-the-art results in both risk detection and safe video generation.Our code is available at https://github.com/Ruize-Ma/ConceptGuard.
>
---
#### [replaced 037] Multi-view Surface Reconstruction Using Normal and Reflectance Cues
- **分类: cs.CV**

- **简介: 该论文属于3D表面重建任务，旨在解决复杂材质下高保真细节重建难题。提出融合多视角法向与反射率信息的框架，通过联合重参数化将二者表示为模拟光照下的辐射量，兼容传统与神经渲染重建流程，显著提升细粒度结构恢复与遮挡处理能力。**

- **链接: [https://arxiv.org/pdf/2506.04115v2](https://arxiv.org/pdf/2506.04115v2)**

> **作者:** Robin Bruneau; Baptiste Brument; Yvain Quéau; Jean Mélou; François Bernard Lauze; Jean-Denis Durou; Lilian Calvet
>
> **备注:** 22 pages, 15 figures, 11 tables. A thorough qualitative and quantitive study is available in the supplementary material at https://drive.google.com/file/d/1KDfCKediXNP5Os954TL_QldaUWS0nKcD/view?usp=drive_link. Accepted to IJCV
>
> **摘要:** Achieving high-fidelity 3D surface reconstruction while preserving fine details remains challenging, especially in the presence of materials with complex reflectance properties and without a dense-view setup. In this paper, we introduce a versatile framework that incorporates multi-view normal and optionally reflectance maps into radiance-based surface reconstruction. Our approach employs a pixel-wise joint re-parametrization of reflectance and surface normals, representing them as a vector of radiances under simulated, varying illumination. This formulation enables seamless incorporation into standard surface reconstruction pipelines, such as traditional multi-view stereo (MVS) frameworks or modern neural volume rendering (NVR) ones. Combined with the latter, our approach achieves state-of-the-art performance on multi-view photometric stereo (MVPS) benchmark datasets, including DiLiGenT-MV, LUCES-MV and Skoltech3D. In particular, our method excels in reconstructing fine-grained details and handling challenging visibility conditions. The present paper is an extended version of the earlier conference paper by Brument et al. (in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024), featuring an accelerated and more robust algorithm as well as a broader empirical evaluation. The code and data relative to this article is available at https://github.com/RobinBruneau/RNb-NeuS2.
>
---
#### [replaced 038] Embodied Crowd Counting
- **分类: cs.CV**

- **简介: 该论文提出新型任务“具身人群计数”（ECC），针对传统方法因被动摄像头导致的遮挡问题。构建大规模交互式仿真数据集ECCD，引入真实人群分布先验；设计零样本导航方法ZECC，结合多模态大模型的粗细粒度导航与法线分析，实现高效精准计数。**

- **链接: [https://arxiv.org/pdf/2503.08367v2](https://arxiv.org/pdf/2503.08367v2)**

> **作者:** Runling Long; Yunlong Wang; Jia Wan; Xiang Deng; Xinting Zhu; Weili Guan; Antoni B. Chan; Liqiang Nie
>
> **摘要:** Occlusion is one of the fundamental challenges in crowd counting. In the community, various data-driven approaches have been developed to address this issue, yet their effectiveness is limited. This is mainly because most existing crowd counting datasets on which the methods are trained are based on passive cameras, restricting their ability to fully sense the environment. Recently, embodied navigation methods have shown significant potential in precise object detection in interactive scenes. These methods incorporate active camera settings, holding promise in addressing the fundamental issues in crowd counting. However, most existing methods are designed for indoor navigation, showing unknown performance in analyzing complex object distribution in large scale scenes, such as crowds. Besides, most existing embodied navigation datasets are indoor scenes with limited scale and object quantity, preventing them from being introduced into dense crowd analysis. Based on this, a novel task, Embodied Crowd Counting (ECC), is proposed. We first build up an interactive simulator, Embodied Crowd Counting Dataset (ECCD), which enables large scale scenes and large object quantity. A prior probability distribution that approximates realistic crowd distribution is introduced to generate crowds. Then, a zero-shot navigation method (ZECC) is proposed. This method contains a MLLM driven coarse-to-fine navigation mechanism, enabling active Z-axis exploration, and a normal-line-based crowd distribution analysis method for fine counting. Experimental results against baselines show that the proposed method achieves the best trade-off between counting accuracy and navigation cost.
>
---
#### [replaced 039] Cross-Layer Vision Smoothing: Enhancing Visual Understanding via Sustained Focus on Key Objects in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对大视觉语言模型中关键物体注意力过短的问题，提出跨层视觉平滑（CLVS）方法。通过引入视觉记忆机制，使模型在多层推理中持续关注关键对象，提升视觉理解能力。实验验证了其在多个任务上的有效性与通用性。**

- **链接: [https://arxiv.org/pdf/2509.12897v2](https://arxiv.org/pdf/2509.12897v2)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng; Zhixing Tan
>
> **备注:** Under Review
>
> **摘要:** Large Vision-Language Models (LVLMs) can accurately locate key objects in images, yet their attention to these objects tends to be very brief. Motivated by the hypothesis that sustained focus on key objects can improve LVLMs' visual capabilities, we propose Cross-Layer Vision Smoothing (CLVS). The core idea of CLVS is to incorporate a vision memory that smooths the attention distribution across layers. Specifically, we initialize this vision memory with position-unbiased visual attention in the first layer. In subsequent layers, the model's visual attention jointly considers the vision memory from previous layers, while the memory is updated iteratively, thereby maintaining smooth attention on key objects. Given that visual understanding primarily occurs in the early and middle layers of the model, we use uncertainty as an indicator of completed visual understanding and terminate the smoothing process accordingly. Experiments on four benchmarks across three LVLMs confirm the effectiveness and generalizability of our method. CLVS achieves state-of-the-art overall performance across a variety of visual understanding tasks and attains comparable results to the leading approaches on image captioning benchmarks.
>
---
#### [replaced 040] Deep Hybrid Model for Region of Interest Detection in Omnidirectional Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对360°视频中的兴趣区域（ROI）检测任务，旨在通过深度混合模型预测用户关注区域，以优化视频流传输与观看体验。工作包括视频帧预处理、构建并训练混合显著性模型，以及后处理输出ROI，最终在360RAT数据集上验证性能。**

- **链接: [https://arxiv.org/pdf/2511.18856v2](https://arxiv.org/pdf/2511.18856v2)**

> **作者:** Sana Alamgeer; Mylene Farias; Marcelo Carvalho
>
> **摘要:** The main goal of the project is to design a new model that predicts regions of interest in 360$^{\circ}$ videos. The region of interest (ROI) plays an important role in 360$^{\circ}$ video streaming. For example, ROIs are used to predict view-ports, intelligently cut the videos for live streaming, etc so that less bandwidth is used. Detecting view-ports in advance helps reduce the movement of the head while streaming and watching a video via the head-mounted device. Whereas, intelligent cuts of the videos help improve the efficiency of streaming the video to users and enhance the quality of their viewing experience. This report illustrates the secondary task to identify ROIs, in which, we design, train, and test a hybrid saliency model. In this work, we refer to saliency regions to represent the regions of interest. The method includes the processes as follows: preprocessing the video to obtain frames, developing a hybrid saliency model for predicting the region of interest, and finally post-processing the output predictions of the hybrid saliency model to obtain the output region of interest for each frame. Then, we compare the performance of the proposed method with the subjective annotations of the 360RAT dataset.
>
---
#### [replaced 041] Optimally Deep Networks -- Adapting Model Depth to Datasets for Superior Efficiency
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对深度神经网络模型过大、计算资源浪费的问题，提出最优深度网络（ODN），通过渐进式深度扩展策略，根据数据集复杂度动态调整模型深度。在保持高精度的同时，显著降低内存占用与计算成本，提升边缘设备部署效率。**

- **链接: [https://arxiv.org/pdf/2510.10764v4](https://arxiv.org/pdf/2510.10764v4)**

> **作者:** Shaharyar Ahmed Khan Tareen; Filza Khan Tareen
>
> **备注:** 6 pages, 4 figures, 1 table, 2 equations, 1 algorithm
>
> **摘要:** Deep neural networks (DNNs) have provided brilliant performance across various tasks. However, this success often comes at the cost of unnecessarily large model sizes, high computational demands, and substantial memory footprints. Typically, powerful architectures are trained at full depths but not all datasets or tasks require such high model capacity. Training big and deep architectures on relatively low-complexity datasets frequently leads to wasted computation, unnecessary energy consumption, and excessive memory usage, which in turn makes deployment of models on resource-constrained devices impractical. To address this problem, we introduce the concept of Optimally Deep Networks (ODNs), which provides a balance between model depth and task complexity. Specifically, we propose a NAS like training strategy called progressive depth expansion, which begins by training neural networks at shallower depths and incrementally increases their depth as the earlier blocks converge, continuing this process until the target accuracy is reached. ODNs use only the optimal depth for the tasks at hand, removing redundant layers. This cuts down future training and inference costs, lowers the model memory footprint, enhances computational efficiency, and facilitates deployment on edge devices. Empirical results show that the optimal depths of ResNet-18 and ResNet-34 for MNIST and SVHN, achieve up to 98.64 % and 96.44 % reduction in memory footprint, while maintaining a competitive accuracy of 99.31 % and 96.08 %, respectively.
>
---
#### [replaced 042] Enhancing Medical Image Analysis through Geometric and Photometric transformations
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对医疗图像数据少的问题，研究几何与光度变换等数据增强技术。通过在皮肤癌分类和视网膜血管分割任务中应用增强方法，显著提升CNN和U-Net模型的准确率与分割效果，验证了数据增强对医学图像分析的有效性。**

- **链接: [https://arxiv.org/pdf/2501.13643v2](https://arxiv.org/pdf/2501.13643v2)**

> **作者:** Khadija Rais; Mohamed Amroune; Mohamed Yassine Haouam; Abdelmadjid Benmachiche
>
> **摘要:** Medical image analysis suffers from a lack of labeled data due to several challenges including patient privacy and lack of experts. Although some AI models only perform well with large amounts of data, we will move to data augmentation where there is a solution to improve the performance of our models and increase the dataset size through traditional or advanced techniques. In this paper, we evaluate the effectiveness of data augmentation techniques on two different medical image datasets. In the first step, we applied some transformation techniques to the skin cancer dataset containing benign and malignant classes. Then, we trained the convolutional neural network (CNN) on the dataset before and after augmentation, which significantly improved test accuracy from 90.74% to 96.88% and decreased test loss from 0.7921 to 0.1468 after augmentation. In the second step, we used the Mixup technique by mixing two random images and their corresponding masks using the retina and blood vessels dataset, then we trained the U-net model and obtained the Dice coefficient which increased from 0 before augmentation to 0.4163 after augmentation. The result shows the effect of using data augmentation to increase the dataset size on the classification and segmentation performance.
>
---
#### [replaced 043] SafeFix: Targeted Model Repair via Controlled Image Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对视觉识别模型在罕见语义子群体上的系统性错误问题，提出SafeFix修复框架。通过可解释的故障归因定位问题属性，利用条件文生图模型生成语义准确的合成图像，并用大视觉语言模型过滤以保持分布一致性，最终通过重训练提升模型对罕见案例的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2508.08701v2](https://arxiv.org/pdf/2508.08701v2)**

> **作者:** Ouyang Xu; Baoming Zhang; Ruiyu Mao; Yunhui Guo
>
> **摘要:** Deep learning models for visual recognition often exhibit systematic errors due to underrepresented semantic subpopulations. Although existing debugging frameworks can pinpoint these failures by identifying key failure attributes, repairing the model effectively remains difficult. Current solutions often rely on manually designed prompts to generate synthetic training images -- an approach prone to distribution shift and semantic errors. To overcome these challenges, we introduce a model repair module that builds on an interpretable failure attribution pipeline. Our approach uses a conditional text-to-image model to generate semantically faithful and targeted images for failure cases. To preserve the quality and relevance of the generated samples, we further employ a large vision-language model (LVLM) to filter the outputs, enforcing alignment with the original data distribution and maintaining semantic consistency. By retraining vision models with this rare-case-augmented synthetic dataset, we significantly reduce errors associated with rare cases. Our experiments demonstrate that this targeted repair strategy improves model robustness without introducing new bugs. Code is available at https://github.com/oxu2/SafeFix
>
---
#### [replaced 044] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLMs）对视角理解能力不足的问题，提出首个系统性基准MMPerspective。通过10项任务评估模型在视角感知、推理与鲁棒性三方面表现，涵盖2,711张图像和5,083个问答对。实验发现模型在深层推理与空间一致性上存在明显短板，揭示了架构、规模与提示策略的影响。**

- **链接: [https://arxiv.org/pdf/2505.20426v5](https://arxiv.org/pdf/2505.20426v5)**

> **作者:** Yolo Y. Tang; Pinxin Liu; Zhangyun Tan; Mingqian Feng; Rui Mao; Chao Huang; Jing Bi; Yunzhong Xiao; Susan Liang; Hang Hua; Ali Vosoughi; Luchuan Song; Zeliang Zhang; Chenliang Xu
>
> **备注:** Accepted to NeurIPS 2025 DB Track. Rating: 5,5,5,5
>
> **摘要:** Understanding perspective is fundamental to human visual perception, yet the extent to which multimodal large language models (MLLMs) internalize perspective geometry remains unclear. We introduce MMPerspective, the first benchmark specifically designed to systematically evaluate MLLMs' understanding of perspective through 10 carefully crafted tasks across three complementary dimensions: Perspective Perception, Reasoning, and Robustness. Our benchmark comprises 2,711 real-world and synthetic image instances with 5,083 question-answer pairs that probe key capabilities, such as vanishing point perception and counting, perspective type reasoning, line relationship understanding in 3D space, invariance to perspective-preserving transformations, etc. Through a comprehensive evaluation of 43 state-of-the-art MLLMs, we uncover significant limitations: while models demonstrate competence on surface-level perceptual tasks, they struggle with compositional reasoning and maintaining spatial consistency under perturbations. Our analysis further reveals intriguing patterns between model architecture, scale, and perspective capabilities, highlighting both robustness bottlenecks and the benefits of chain-of-thought prompting. MMPerspective establishes a valuable testbed for diagnosing and advancing spatial understanding in vision-language systems. Resources available at: https://yunlong10.github.io/MMPerspective/
>
---
#### [replaced 045] ExDDV: A New Dataset for Explainable Deepfake Detection in Video
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文提出ExDDV，首个面向视频深度伪造可解释检测的数据集。针对现有检测模型缺乏可解释性的问题，构建包含5.4K视频及文本描述与点击标注的多模态数据集，评估视觉-语言模型在定位与描述伪造痕迹上的表现，验证文本与点击监督对提升模型可解释性的重要性。**

- **链接: [https://arxiv.org/pdf/2503.14421v2](https://arxiv.org/pdf/2503.14421v2)**

> **作者:** Vlad Hondru; Eduard Hogea; Darian Onchis; Radu Tudor Ionescu
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** The ever growing realism and quality of generated videos makes it increasingly harder for humans to spot deepfake content, who need to rely more and more on automatic deepfake detectors. However, deepfake detectors are also prone to errors, and their decisions are not explainable, leaving humans vulnerable to deepfake-based fraud and misinformation. To this end, we introduce ExDDV, the first dataset and benchmark for Explainable Deepfake Detection in Video. ExDDV comprises around 5.4K real and deepfake videos that are manually annotated with text descriptions (to explain the artifacts) and clicks (to point out the artifacts). We evaluate a number of vision-language models on ExDDV, performing experiments with various fine-tuning and in-context learning strategies. Our results show that text and click supervision are both required to develop robust explainable models for deepfake videos, which are able to localize and describe the observed artifacts. Our novel dataset and code to reproduce the results are available at https://github.com/vladhondru25/ExDDV.
>
---
#### [replaced 046] Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对静态与动态场景的3D重建任务，解决现有方法中高冗余、位置受限及难以利用时序信息的问题。提出Fuse-and-Refine模块，通过混合体素表示融合多视角输入，在3D空间中精炼高斯原语，实现高效、低冗余的在线重建，支持交互式实时运行。**

- **链接: [https://arxiv.org/pdf/2503.14698v2](https://arxiv.org/pdf/2503.14698v2)**

> **作者:** Yiming Wang; Lucy Chai; Xuan Luo; Michael Niemeyer; Manuel Lagunas; Stephen Lombardi; Siyu Tang; Tiancheng Sun
>
> **备注:** NeurIPS 2025, Previously titled "SplatVoxel: History-Aware Novel View Streaming without Temporal Training", Project Page: https://19reborn.github.io/SplatVoxel/
>
> **摘要:** Recent advances in feed-forward 3D Gaussian Splatting have led to rapid improvements in efficient scene reconstruction from sparse views. However, most existing approaches construct Gaussian primitives directly aligned with the pixels in one or more of the input images. This leads to redundancies in the representation when input views overlap and constrains the position of the primitives to lie along the input rays without full flexibility in 3D space. Moreover, these pixel-aligned approaches do not naturally generalize to dynamic scenes, where effectively leveraging temporal information requires resolving both redundant and newly appearing content across frames. To address these limitations, we introduce a novel Fuse-and-Refine module that enhances existing feed-forward models by merging and refining the primitives in a canonical 3D space. At the core of our method is an efficient hybrid Splat-Voxel representation: from an initial set of pixel-aligned Gaussian primitives, we aggregate local features into a coarse-to-fine voxel hierarchy, and then use a sparse voxel transformer to process these voxel features and generate refined Gaussian primitives. By fusing and refining an arbitrary number of inputs into a consistent set of primitives, our representation effectively reduces redundancy and naturally adapts to temporal frames, enabling history-aware online reconstruction of dynamic scenes. Our approach achieves state-of-the-art performance in both static and streaming scene reconstructions while running at interactive rates (15 fps with 350ms delay) on a single H100 GPU.
>
---
#### [replaced 047] Panoptic Captioning: An Equivalence Bridge for Image and Text
- **分类: cs.CV**

- **简介: 该论文提出panoptic captioning任务，旨在生成涵盖图像中所有实体、位置、属性、关系及全局状态的最小文本等价描述。针对现有模型表现不佳，提出PancapEngine数据引擎与PancapChain方法，通过分步生成与高质量数据提升性能，超越主流模型。**

- **链接: [https://arxiv.org/pdf/2505.16334v3](https://arxiv.org/pdf/2505.16334v3)**

> **作者:** Kun-Yu Lin; Hongjun Wang; Weining Ren; Kai Han
>
> **备注:** NeurIPS 2025; Project page: https://visual-ai.github.io/pancap/
>
> **摘要:** This work introduces panoptic captioning, a novel task striving to seek the minimum text equivalent of images, which has broad potential applications. We take the first step towards panoptic captioning by formulating it as a task of generating a comprehensive textual description for an image, which encapsulates all entities, their respective locations and attributes, relationships among entities, as well as global image state. Through an extensive evaluation, our work reveals that state-of-the-art Multi-modal Large Language Models (MLLMs) have limited performance in solving panoptic captioning. To address this, we propose an effective data engine named PancapEngine to produce high-quality data and a novel method named PancapChain to improve panoptic captioning. Specifically, our PancapEngine first detects diverse categories of entities in images by an elaborate detection suite, and then generates required panoptic captions using entity-aware prompts. Additionally, our PancapChain explicitly decouples the challenging panoptic captioning task into multiple stages and generates panoptic captions step by step. More importantly, we contribute a comprehensive metric named PancapScore and a human-curated test set for reliable model evaluation. Experiments show that our PancapChain-13B model can beat state-of-the-art open-source MLLMs like InternVL-2.5-78B and even surpass proprietary models like GPT-4o and Gemini-2.0-Pro, demonstrating the effectiveness of our data engine and method. Project page: https://visual-ai.github.io/pancap/
>
---
#### [replaced 048] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文提出广义开放词汇3D场景理解（GOV-3D）任务，旨在超越传统对象类别限制，理解细粒度语义属性。针对现有方法在抽象词汇理解上的不足，构建了包含八类语言属性的基准OpenScan，评估并揭示当前模型在复杂语义理解上的局限性，推动更全面的3D场景理解研究。**

- **链接: [https://arxiv.org/pdf/2408.11030v4](https://arxiv.org/pdf/2408.11030v4)**

> **作者:** Youjun Zhao; Jiaying Lin; Shuquan Ye; Qianshi Pang; Rynson W. H. Lau
>
> **备注:** Accepted by AAAI 2026. Project Page: https://youjunzhao.github.io/OpenScan/
>
> **摘要:** Open-vocabulary 3D scene understanding (OV-3D) aims to localize and classify novel objects beyond the closed set of object classes. However, existing approaches and benchmarks primarily focus on the open vocabulary problem within the context of object classes, which is insufficient in providing a holistic evaluation to what extent a model understands the 3D scene. In this paper, we introduce a more challenging task called Generalized Open-Vocabulary 3D Scene Understanding (GOV-3D) to explore the open vocabulary problem beyond object classes. It encompasses an open and diverse set of generalized knowledge, expressed as linguistic queries of fine-grained and object-specific attributes. To this end, we contribute a new benchmark named \textit{OpenScan}, which consists of 3D object attributes across eight representative linguistic aspects, including affordance, property, and material. We further evaluate state-of-the-art OV-3D methods on our OpenScan benchmark and discover that these methods struggle to comprehend the abstract vocabularies of the GOV-3D task, a challenge that cannot be addressed simply by scaling up object classes during training. We highlight the limitations of existing methodologies and explore promising directions to overcome the identified shortcomings.
>
---
#### [replaced 049] InstaDA: Augmenting Instance Segmentation Data with Dual-Agent System
- **分类: cs.CV**

- **简介: 该论文针对实例分割数据标注难、类别不平衡问题，提出无需训练的Dual-Agent系统InstaDA。通过文本代理（T-Agent）与图像代理（I-Agent）协同，利用大语言模型与扩散模型生成多样化新样本，提升数据多样性与分布均衡性。实验表明，该方法在LVIS 1.0上显著提升性能，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2509.02973v2](https://arxiv.org/pdf/2509.02973v2)**

> **作者:** Xianbao Hou; Yonghao He; Zeyd Boukhers; John See; Hu Su; Wei Sui; Cong Yang
>
> **摘要:** Acquiring high-quality instance segmentation data is challenging due to the labor-intensive nature of the annotation process and significant class imbalances within datasets. Recent studies have utilized the integration of Copy-Paste and diffusion models to create more diverse datasets. However, these studies often lack deep collaboration between large language models (LLMs) and diffusion models, and underutilize the rich information within the existing training data. To address these limitations, we propose InstaDA, a novel, training-free Dual-Agent system designed to augment instance segmentation datasets. First, we introduce a Text-Agent (T-Agent) that enhances data diversity through collaboration between LLMs and diffusion models. This agent features a novel Prompt Rethink mechanism, which iteratively refines prompts based on the generated images. This process not only fosters collaboration but also increases image utilization and optimizes the prompts themselves. Additionally, we present an Image-Agent (I-Agent) aimed at enriching the overall data distribution. This agent augments the training set by generating new instances conditioned on the training images. To ensure practicality and efficiency, both agents operate as independent and automated workflows, enhancing usability. Experiments conducted on the LVIS 1.0 validation set indicate that InstaDA achieves significant improvements, with an increase of +4.0 in box average precision (AP) and +3.3 in mask AP compared to the baseline. Furthermore, it outperforms the leading model, DiverGen, by +0.3 in box AP and +0.1 in mask AP, with a notable +0.7 gain in box AP on common categories and mask AP gains of +0.2 on common categories and +0.5 on frequent categories.
>
---
#### [replaced 050] 3D-Guided Scalable Flow Matching for Generating Volumetric Tissue Spatial Transcriptomics from Serial Histology
- **分类: cs.CV**

- **简介: 该论文提出HoloTea，一种3D-aware流匹配框架，用于从连续组织切片生成体素级空间转录组数据。针对现有方法忽略3D结构或无法扩展的问题，通过跨切面形态对齐与轻量ControlNet融合，引入3D一致先验和全局注意力机制，实现高效高精度的3D基因表达重建。**

- **链接: [https://arxiv.org/pdf/2511.14613v2](https://arxiv.org/pdf/2511.14613v2)**

> **作者:** Mohammad Vali Sanian; Arshia Hemmat; Amirhossein Vahidi; Jonas Maaskola; Jimmy Tsz Hang Lee; Stanislaw Makarchuk; Yeliz Demirci; Nana-Jane Chipampe; Muzlifah Haniffa; Omer Bayraktar; Lassi Paavolainen; Mohammad Lotfollahi
>
> **备注:** 19 pages
>
> **摘要:** A scalable and robust 3D tissue transcriptomics profile can enable a holistic understanding of tissue organization and provide deeper insights into human biology and disease. Most predictive algorithms that infer ST directly from histology treat each section independently and ignore 3D structure, while existing 3D-aware approaches are not generative and do not scale well. We present Holographic Tissue Expression Inpainting and Analysis (HoloTea), a 3D-aware flow-matching framework that imputes spot-level gene expression from H&E while explicitly using information from adjacent sections. Our key idea is to retrieve morphologically corresponding spots on neighboring slides in a shared feature space and fuse this cross section context into a lightweight ControlNet, allowing conditioning to follow anatomical continuity. To better capture the count nature of the data, we introduce a 3D-consistent prior for flow matching that combines a learned zero-inflated negative binomial (ZINB) prior with a spatial-empirical prior constructed from neighboring sections. A global attention block introduces 3D H&E scaling linearly with the number of spots in the slide, enabling training and inference on large 3D ST datasets. Across three spatial transcriptomics datasets spanning different tissue types and resolutions, HoloTea consistently improves 3D expression accuracy and generalization compared to 2D and 3D baselines. We envision HoloTea advancing the creation of accurate 3D virtual tissues, ultimately accelerating biomarker discovery and deepening our understanding of disease.
>
---
#### [replaced 051] Unified Text-Image-to-Video Generation: A Training-Free Approach to Flexible Visual Conditioning
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对文本图像到视频生成任务，解决现有方法依赖微调、条件固定且资源消耗大的问题。提出无需训练的FlexTI2V方法，通过隐空间反演与随机局部块替换，实现任意数量图像在任意位置的灵活视觉条件控制，并动态调节条件强度，兼容多种模型架构。**

- **链接: [https://arxiv.org/pdf/2505.20629v2](https://arxiv.org/pdf/2505.20629v2)**

> **作者:** Bolin Lai; Sangmin Lee; Xu Cao; Xiang Li; James M. Rehg
>
> **备注:** 18 pages, 10 figures, 8 tables
>
> **摘要:** Text-image-to-video (TI2V) generation is a critical problem for controllable video generation using both semantic and visual conditions. Most existing methods typically add visual conditions to text-to-video (T2V) foundation models by finetuning, which is costly in resources and only limited to a few pre-defined conditioning settings. To tackle these constraints, we introduce a unified formulation for TI2V generation with flexible visual conditioning. Furthermore, we propose an innovative training-free approach, dubbed FlexTI2V, that can condition T2V foundation models on an arbitrary amount of images at arbitrary positions. Specifically, we firstly invert the condition images to noisy representation in a latent space. Then, in the denoising process of T2V models, our method uses a novel random patch swapping strategy to incorporate visual features into video representations through local image patches. To balance creativity and fidelity, we use a dynamic control mechanism to adjust the strength of visual conditioning to each video frame. Extensive experiments validate that our method surpasses previous training-free image conditioning methods by a notable margin. Our method can also generalize to both UNet-based and transformer-based architectures.
>
---
#### [replaced 052] SatSAM2: Motion-Constrained Video Object Tracking in Satellite Imagery using Promptable SAM2 and Kalman Priors
- **分类: cs.CV**

- **简介: 该论文提出SatSAM2，一种零样本卫星视频目标跟踪方法，解决现有方法泛化差、易丢失目标的问题。通过引入基于卡尔曼滤波的运动约束模块和状态机，利用时序运动信息抑制漂移。构建了大规模合成基准MVOT，实验表明其在多个数据集上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.18264v2](https://arxiv.org/pdf/2511.18264v2)**

> **作者:** Ruijie Fan; Junyan Ye; Huan Chen; Zilong Huang; Xiaolei Wang; Weijia Li
>
> **摘要:** Existing satellite video tracking methods often struggle with generalization, requiring scenario-specific training to achieve satisfactory performance, and are prone to track loss in the presence of occlusion. To address these challenges, we propose SatSAM2, a zero-shot satellite video tracker built on SAM2, designed to adapt foundation models to the remote sensing domain. SatSAM2 introduces two core modules: a Kalman Filter-based Constrained Motion Module (KFCMM) to exploit temporal motion cues and suppress drift, and a Motion-Constrained State Machine (MCSM) to regulate tracking states based on motion dynamics and reliability. To support large-scale evaluation, we propose MatrixCity Video Object Tracking (MVOT), a synthetic benchmark containing 1,500+ sequences and 157K annotated frames with diverse viewpoints, illumination, and occlusion conditions. Extensive experiments on two satellite tracking benchmarks and MVOT show that SatSAM2 outperforms both traditional and foundation model-based trackers, including SAM2 and its variants. Notably, on the OOTB dataset, SatSAM2 achieves a 5.84% AUC improvement over state-of-the-art methods. Our code and dataset will be publicly released to encourage further research.
>
---
#### [replaced 053] CGCE: Classifier-Guided Concept Erasure in Generative Models
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文提出CGCE框架，解决生成模型中安全概念被恶意重生成的问题。针对现有方法在安全与生成质量间的权衡，CGCE通过轻量级文本分类器在推理时动态修正提示，实现鲁棒且无损的概念擦除，适用于多种文生图/视频模型。**

- **链接: [https://arxiv.org/pdf/2511.05865v2](https://arxiv.org/pdf/2511.05865v2)**

> **作者:** Viet Nguyen; Vishal M. Patel
>
> **备注:** 26 pages, 17 figures
>
> **摘要:** Recent advancements in large-scale generative models have enabled the creation of high-quality images and videos, but have also raised significant safety concerns regarding the generation of unsafe content. To mitigate this, concept erasure methods have been developed to remove undesirable concepts from pre-trained models. However, existing methods remain vulnerable to adversarial attacks that can regenerate the erased content. Moreover, achieving robust erasure often degrades the model's generative quality for safe, unrelated concepts, creating a difficult trade-off between safety and performance. To address this challenge, we introduce Classifier-Guided Concept Erasure (CGCE), an efficient plug-and-play framework that provides robust concept erasure for diverse generative models without altering their original weights. CGCE uses a lightweight classifier operating on text embeddings to first detect and then refine prompts containing undesired concepts. This approach is highly scalable, allowing for multi-concept erasure by aggregating guidance from several classifiers. By modifying only unsafe embeddings at inference time, our method prevents harmful content generation while preserving the model's original quality on benign prompts. Extensive experiments show that CGCE achieves state-of-the-art robustness against a wide range of red-teaming attacks. Our approach also maintains high generative utility, demonstrating a superior balance between safety and performance. We showcase the versatility of CGCE through its successful application to various modern T2I and T2V models, establishing it as a practical and effective solution for safe generative AI.
>
---
#### [replaced 054] StrCGAN: A Generative Framework for Stellar Image Restoration
- **分类: cs.CV; astro-ph.IM; astro-ph.SR**

- **简介: 该论文提出StrCGAN，一种用于天文图像增强的生成模型，旨在提升小望远镜低分辨率星像质量。针对传统方法扭曲星体形态的问题，引入多光谱融合与天体物理正则化模块，利用多源星图数据指导训练，实现更真实、高保真的星像重建。**

- **链接: [https://arxiv.org/pdf/2509.19805v3](https://arxiv.org/pdf/2509.19805v3)**

> **作者:** Shantanusinh Parmar; Silas Janke
>
> **摘要:** We introduce StrCGAN (Stellar Cyclic GAN), a generative model designed to enhance low-resolution astrophotography images. Our goal is to reconstruct high fidelity ground truth like representations of stellar objects, a task that is challenging due to the limited resolution and quality of small-telescope observations such as the MobilTelesco dataset. Traditional models such as CycleGAN provide a foundation for image to image translation but often distort the morphology of stars and produce barely resembling images. To overcome these limitations, we extend the CycleGAN framework with some key innovations: multi-spectral fusion to align optical and near infrared (NIR) domains, and astrophysical regularization modules to preserve stellar morphology. Ground truth references from multi-mission all sky surveys spanning optical to NIR guide the training process, ensuring that reconstructions remain consistent across spectral bands. Together, these components allow StrCGAN to generate reconstructions that are visually sharper outperforming standard GAN models in the task of astrophysical image enhancement.
>
---
#### [replaced 055] Probabilistic Hyper-Graphs using Multiple Randomly Masked Autoencoders for Semi-supervised Multi-modal Multi-task Learning
- **分类: cs.CV**

- **简介: 该论文提出PHG-MAE，一种基于多模态随机掩码自编码器的半监督多任务学习模型，通过构建概率超图统一神经图与自编码器框架，实现模态级掩码采样与推理时集成，提升性能与一致性，并支持轻量知识蒸馏，适用于无人机等多解释场景。**

- **链接: [https://arxiv.org/pdf/2510.10068v2](https://arxiv.org/pdf/2510.10068v2)**

> **作者:** Pîrvu Mihai-Cristian; Marius Leordeanu
>
> **备注:** Submitted to Neurocomputing
>
> **摘要:** The computer vision domain has greatly benefited from an abundance of data across many modalities to improve on various visual tasks. Recently, there has been a lot of focus on self-supervised pre-training methods through Masked Autoencoders (MAE) \cite{he2022masked,bachmann2022multimae}, usually used as a first step before optimizing for a downstream task, such as classification or regression. This is very useful as it doesn't require any manually labeled data. In this work, we introduce Probabilistic Hyper-Graphs using Masked Autoencoders (PHG-MAE): a novel model that unifies the classical work on neural graphs \cite{leordeanu2021semi} with the modern approach of masked autoencoders under a common theoretical framework. Through random masking of entire modalities, not just patches, the model samples from the distribution of hyper-edges on each forward pass. Additionally, the model adapts the standard MAE algorithm by combining pre-training and fine-tuning into a single training loop. Moreover, our approach enables the creation of inference-time ensembles which, through aggregation, boost the final prediction performance and consistency. Lastly, we show that we can apply knowledge distillation on top of the ensembles with little loss in performance, even with models that have fewer than 1M parameters. While our work mostly focuses on outdoor UAV scenes that contain multiple world interpretations and modalities, the same steps can be followed in other similar domains, such as autonomous driving or indoor robotics. In order to streamline the process of integrating external pre-trained experts for computer vision multi-modal multi-task learning (MTL) scenarios, we developed a data-pipeline software. Using this tool, we have created and released a fully-automated extension of the Dronescapes dataset. All the technical details, code and reproduction steps are publicly released.
>
---
#### [replaced 056] OmniLens++: Blind Lens Aberration Correction via Large LensLib Pre-Training and Latent PSF Representation
- **分类: eess.IV; cs.CV; cs.LG; physics.optics**

- **简介: 该论文针对盲镜头像畸变校正任务，解决现有方法在数据规模和退化先验缺失上的局限。提出OmniLens++框架，通过扩展镜头库多样性与引入潜空间点扩散函数表示（LPR），提升模型泛化能力，实现对真实与合成镜头退化的高效校正。**

- **链接: [https://arxiv.org/pdf/2511.17126v3](https://arxiv.org/pdf/2511.17126v3)**

> **作者:** Qi Jiang; Xiaolong Qian; Yao Gao; Lei Sun; Kailun Yang; Zhonghua Yi; Wenyong Li; Ming-Hsuan Yang; Luc Van Gool; Kaiwei Wang
>
> **备注:** The source code and datasets will be made publicly available at https://github.com/zju-jiangqi/OmniLens2
>
> **摘要:** Emerging deep-learning-based lens library pre-training (LensLib-PT) pipeline offers a new avenue for blind lens aberration correction by training a universal neural network, demonstrating strong capability in handling diverse unknown optical degradations. This work proposes the OmniLens++ framework, which resolves two challenges that hinder the generalization ability of existing pipelines: the difficulty of scaling data and the absence of prior guidance characterizing optical degradation. To improve data scalability, we expand the design specifications to increase the degradation diversity of the lens source, and we sample a more uniform distribution by quantifying the spatial-variation patterns and severity of optical degradation. In terms of model design, to leverage the Point Spread Functions (PSFs), which intuitively describe optical degradation, as guidance in a blind paradigm, we propose the Latent PSF Representation (LPR). The VQVAE framework is introduced to learn latent features of LensLib's PSFs, which is assisted by modeling the optical degradation process to constrain the learning of degradation priors. Experiments on diverse aberrations of real-world lenses and synthetic LensLib show that OmniLens++ exhibits state-of-the-art generalization capacity in blind aberration correction. Beyond performance, the AODLibpro is verified as a scalable foundation for more effective training across diverse aberrations, and LPR can further tap the potential of large-scale LensLib. The source code and datasets will be made publicly available at https://github.com/zju-jiangqi/OmniLens2.
>
---
#### [replaced 057] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对多模态情感分析中的模态竞争问题，提出PaSE框架。通过原型对齐校准与谢林值优化均衡机制，增强模态协作、缓解主导模态压制弱模态的问题，提升融合效果。**

- **链接: [https://arxiv.org/pdf/2511.17585v2](https://arxiv.org/pdf/2511.17585v2)**

> **作者:** Kang He; Boyu Chen; Yuzhe Ding; Fei Li; Chong Teng; Donghong Ji
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multimodal Sentiment Analysis (MSA) seeks to understand human emotions by integrating textual, acoustic, and visual signals. Although multimodal fusion is designed to leverage cross-modal complementarity, real-world scenarios often exhibit modality competition: dominant modalities tend to overshadow weaker ones, leading to suboptimal performance. In this paper, we propose PaSE, a novel Prototype-aligned Calibration and Shapley-optimized Equilibrium framework, which enhances collaboration while explicitly mitigating modality competition. PaSE first applies Prototype-guided Calibration Learning (PCL) to refine unimodal representations and align them through an Entropic Optimal Transport mechanism that ensures semantic consistency. To further stabilize optimization, we introduce a Dual-Phase Optimization strategy. A prototype-gated fusion module is first used to extract shared representations, followed by Shapley-based Gradient Modulation (SGM), which adaptively adjusts gradients according to the contribution of each modality. Extensive experiments on IEMOCAP, MOSI, and MOSEI confirm that PaSE achieves the superior performance and effectively alleviates modality competition.
>
---
#### [replaced 058] AlignCVC: Aligning Cross-View Consistency for Single-Image-to-3D Generation
- **分类: cs.CV**

- **简介: 该论文针对单图像生成3D任务中的多视角一致性（CVC）问题，提出AlignCVC框架。通过分布对齐策略，同时优化生成与重建的多视角分布，提升一致性。采用软-硬协同对齐机制，显著加速推理至4步，且可无缝集成各类模型。**

- **链接: [https://arxiv.org/pdf/2506.23150v2](https://arxiv.org/pdf/2506.23150v2)**

> **作者:** Xinyue Liang; Zhiyuan Ma; Lingchen Sun; Yanjun Guo; Lei Zhang
>
> **摘要:** Single-image-to-3D models typically follow a sequential generation and reconstruction workflow. However, intermediate multi-view images synthesized by pre-trained generation models often lack cross-view consistency (CVC), significantly degrading 3D reconstruction performance. While recent methods attempt to refine CVC by feeding reconstruction results back into the multi-view generator, these approaches struggle with noisy and unstable reconstruction outputs that limit effective CVC improvement. We introduce AlignCVC, a novel framework that fundamentally re-frames single-image-to-3D generation through distribution alignment rather than relying on strict regression losses. Our key insight is to align both generated and reconstructed multi-view distributions toward the ground-truth multi-view distribution, establishing a principled foundation for improved CVC. Observing that generated images exhibit weak CVC while reconstructed images display strong CVC due to explicit rendering, we propose a soft-hard alignment strategy with distinct objectives for generation and reconstruction models. This approach not only enhances generation quality but also dramatically accelerates inference to as few as 4 steps. As a plug-and-play paradigm, our method, namely AlignCVC, seamlessly integrates various multi-view generation models with 3D reconstruction models. Extensive experiments demonstrate the effectiveness and efficiency of AlignCVC for single-image-to-3D generation.
>
---
#### [replaced 059] How Animals Dance (When You're Not Looking)
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出一种生成音乐同步、舞步结构感知的动物舞蹈视频的方法。针对舞蹈视频生成中缺乏长程结构控制的问题，引入“舞步模式”作为高层控制信号，通过图优化确定关键帧布局，并结合视频扩散模型生成中间帧，实现仅用6个关键帧生成30秒多样动物舞蹈视频。**

- **链接: [https://arxiv.org/pdf/2505.23738v2](https://arxiv.org/pdf/2505.23738v2)**

> **作者:** Xiaojuan Wang; Aleksander Holynski; Brian Curless; Ira Kemelmacher; Steve Seitz
>
> **备注:** Project page: https://how-animals-dance.github.io/
>
> **摘要:** We present a framework for generating music-synchronized, choreography aware animal dance videos. Our framework introduces choreography patterns -- structured sequences of motion beats that define the long-range structure of a dance -- as a novel high-level control signal for dance video generation. These patterns can be automatically estimated from human dance videos. Starting from a few keyframes representing distinct animal poses, generated via text-to-image prompting or GPT-4o, we formulate dance synthesis as a graph optimization problem that seeks the optimal keyframe structure to satisfy a specified choreography pattern of beats. We also introduce an approach for mirrored pose image generation, essential for capturing symmetry in dance. In-between frames are synthesized using an video diffusion model. With as few as six input keyframes, our method can produce up to 30 seconds dance videos across a wide range of animals and music tracks.
>
---
#### [replaced 060] IVY-FAKE: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对AIGC检测中解释性不足的问题，提出IVY-FAKE统一框架与基准。构建了超10万样本的多维标注数据集，设计基于GRPO的可解释检测模型Ivy-xDetector，显著提升图像与视频伪造内容的检测性能与推理透明度。**

- **链接: [https://arxiv.org/pdf/2506.00979v2](https://arxiv.org/pdf/2506.00979v2)**

> **作者:** Changjiang Jiang; Wenhui Dong; Zhonghao Zhang; Chenyang Si; Fengchang Yu; Wei Peng; Xinbin Yuan; Yifei Bi; Ming Zhao; Zian Zhou; Caifeng Shan
>
> **备注:** 30 pages
>
> **摘要:** The rapid development of Artificial Intelligence Generated Content (AIGC) techniques has enabled the creation of high-quality synthetic content, but it also raises significant security concerns. Current detection methods face two major limitations: (1) the lack of multidimensional explainable datasets for generated images and videos. Existing open-source datasets (e.g., WildFake, GenVideo) rely on oversimplified binary annotations, which restrict the explainability and trustworthiness of trained detectors. (2) Prior MLLM-based forgery detectors (e.g., FakeVLM) exhibit insufficiently fine-grained interpretability in their step-by-step reasoning, which hinders reliable localization and explanation. To address these challenges, we introduce Ivy-Fake, the first large-scale multimodal benchmark for explainable AIGC detection. It consists of over 106K richly annotated training samples (images and videos) and 5,000 manually verified evaluation examples, sourced from multiple generative models and real world datasets through a carefully designed pipeline to ensure both diversity and quality. Furthermore, we propose Ivy-xDetector, a reinforcement learning model based on Group Relative Policy Optimization (GRPO), capable of producing explainable reasoning chains and achieving robust performance across multiple synthetic content detection benchmarks. Extensive experiments demonstrate the superiority of our dataset and confirm the effectiveness of our approach. Notably, our method improves performance on GenImage from 86.88% to 96.32%, surpassing prior state-of-the-art methods by a clear margin.
>
---
#### [replaced 061] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MeshSplat，一种基于高斯溅射的通用稀疏视图表面重建方法。针对输入视图过少时几何恢复不准确的问题，利用2DGS作为桥梁，结合像素对齐的2DGS预测与自监督学习，提升重建精度。通过加权切比雪夫距离损失和法向对齐网络优化深度与姿态，实现高性能稀疏视图网格重建。**

- **链接: [https://arxiv.org/pdf/2508.17811v2](https://arxiv.org/pdf/2508.17811v2)**

> **作者:** Hanzhi Chang; Ruijie Zhu; Wenjie Chang; Mulin Yu; Yanzhe Liang; Jiahao Lu; Zhuoyuan Li; Tianzhu Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Surface reconstruction has been widely studied in computer vision and graphics. However, existing surface reconstruction works struggle to recover accurate scene geometry when the input views are extremely sparse. To address this issue, we propose MeshSplat, a generalizable sparse-view surface reconstruction framework via Gaussian Splatting. Our key idea is to leverage 2DGS as a bridge, which connects novel view synthesis to learned geometric priors and then transfers these priors to achieve surface reconstruction. Specifically, we incorporate a feed-forward network to predict per-view pixel-aligned 2DGS, which enables the network to synthesize novel view images and thus eliminates the need for direct 3D ground-truth supervision. To improve the accuracy of 2DGS position and orientation prediction, we propose a Weighted Chamfer Distance Loss to regularize the depth maps, especially in overlapping areas of input views, and also a normal prediction network to align the orientation of 2DGS with normal vectors predicted by a monocular normal estimator. Extensive experiments validate the effectiveness of our proposed improvement, demonstrating that our method achieves state-of-the-art performance in generalizable sparse-view mesh reconstruction tasks. Project Page: https://hanzhichang.github.io/meshsplat_web
>
---
#### [replaced 062] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文针对视频到音频生成任务，解决现有方法因目标耦合导致的感知维度失衡与人类偏好不符问题。提出PrismAudio框架，通过四类分解式思维链与多维奖励机制实现可解释的强化学习优化，并引入Fast-GRPO加速训练及AudioCanvas基准测试，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.18833v2](https://arxiv.org/pdf/2511.18833v2)**

> **作者:** Huadai Liu; Kaicheng Luo; Wen Wang; Qian Chen; Peiwen Sun; Rongjie Huang; Xiangang Li; Jieping Ye; Wei Xue
>
> **备注:** Preprint
>
> **摘要:** Video-to-Audio (V2A) generation requires balancing four critical perceptual dimensions: semantic consistency, audio-visual temporal synchrony, aesthetic quality, and spatial accuracy; yet existing methods suffer from objective entanglement that conflates competing goals in single loss functions and lack human preference alignment. We introduce PrismAudio, the first framework to integrate Reinforcement Learning into V2A generation with specialized Chain-of-Thought (CoT) planning. Our approach decomposes monolithic reasoning into four specialized CoT modules (Semantic, Temporal, Aesthetic, and Spatial CoT), each paired with targeted reward functions. This CoT-reward correspondence enables multidimensional RL optimization that guides the model to jointly generate better reasoning across all perspectives, solving the objective entanglement problem while preserving interpretability. To make this optimization computationally practical, we propose Fast-GRPO, which employs hybrid ODE-SDE sampling that dramatically reduces the training overhead compared to existing GRPO implementations. We also introduce AudioCanvas, a rigorous benchmark that is more distributionally balanced and covers more realistically diverse and challenging scenarios than existing datasets, with 300 single-event classes and 501 multi-event samples. Experimental results demonstrate that PrismAudio achieves state-of-the-art performance across all four perceptual dimensions on both the in-domain VGGSound test set and out-of-domain AudioCanvas benchmark. The project page is available at https://PrismAudio-Project.github.io.
>
---
#### [replaced 063] HunyuanVideo 1.5 Technical Report
- **分类: cs.CV**

- **简介: 该论文提出HunyuanVideo 1.5，一个仅83亿参数的轻量级开源视频生成模型。针对高质视频生成中参数量大、推理成本高的问题，通过数据优化、SSTA架构、双语编码与超分网络等设计，实现文本/图像到视频的高效生成，达成开源领域新最优性能，推动视频生成技术普及。**

- **链接: [https://arxiv.org/pdf/2511.18870v2](https://arxiv.org/pdf/2511.18870v2)**

> **作者:** Bing Wu; Chang Zou; Changlin Li; Duojun Huang; Fang Yang; Hao Tan; Jack Peng; Jianbing Wu; Jiangfeng Xiong; Jie Jiang; Linus; Patrol; Peizhen Zhang; Peng Chen; Penghao Zhao; Qi Tian; Songtao Liu; Weijie Kong; Weiyan Wang; Xiao He; Xin Li; Xinchi Deng; Xuefei Zhe; Yang Li; Yanxin Long; Yuanbo Peng; Yue Wu; Yuhong Liu; Zhenyu Wang; Zuozhuo Dai; Bo Peng; Coopers Li; Gu Gong; Guojian Xiao; Jiahe Tian; Jiaxin Lin; Jie Liu; Jihong Zhang; Jiesong Lian; Kaihang Pan; Lei Wang; Lin Niu; Mingtao Chen; Mingyang Chen; Mingzhe Zheng; Miles Yang; Qiangqiang Hu; Qi Yang; Qiuyong Xiao; Runzhou Wu; Ryan Xu; Rui Yuan; Shanshan Sang; Shisheng Huang; Siruis Gong; Shuo Huang; Weiting Guo; Xiang Yuan; Xiaojia Chen; Xiawei Hu; Wenzhi Sun; Xiele Wu; Xianshun Ren; Xiaoyan Yuan; Xiaoyue Mi; Yepeng Zhang; Yifu Sun; Yiting Lu; Yitong Li; You Huang; Yu Tang; Yixuan Li; Yuhang Deng; Yuan Zhou; Zhichao Hu; Zhiguang Liu; Zhihe Yang; Zilin Yang; Zhenzhi Lu; Zixiang Zhou; Zhao Zhong
>
> **摘要:** We present HunyuanVideo 1.5, a lightweight yet powerful open-source video generation model that achieves state-of-the-art visual quality and motion coherence with only 8.3 billion parameters, enabling efficient inference on consumer-grade GPUs. This achievement is built upon several key components, including meticulous data curation, an advanced DiT architecture featuring selective and sliding tile attention (SSTA), enhanced bilingual understanding through glyph-aware text encoding, progressive pre-training and post-training, and an efficient video super-resolution network. Leveraging these designs, we developed a unified framework capable of high-quality text-to-video and image-to-video generation across multiple durations and resolutions. Extensive experiments demonstrate that this compact and proficient model establishes a new state-of-the-art among open-source video generation models. By releasing the code and model weights, we provide the community with a high-performance foundation that lowers the barrier to video creation and research, making advanced video generation accessible to a broader audience. All open-source assets are publicly available at https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.
>
---
#### [replaced 064] Generative AI for Cel-Animation: A Survey
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于生成式AI在传统赛璐珞动画中的应用综述，旨在解决动画制作中人工成本高、效率低的问题。通过分析GenAI在分镜、关键帧、补间、上色等环节的自动化应用，探讨其提升创作效率与可及性的潜力，并指出视觉一致性与伦理挑战，展望未来发展方向。**

- **链接: [https://arxiv.org/pdf/2501.06250v5](https://arxiv.org/pdf/2501.06250v5)**

> **作者:** Yolo Y. Tang; Junjia Guo; Pinxin Liu; Zhiyuan Wang; Hang Hua; Jia-Xing Zhong; Yunzhong Xiao; Chao Huang; Luchuan Song; Susan Liang; Yizhi Song; Liu He; Jing Bi; Mingqian Feng; Xinyang Li; Zeliang Zhang; Chenliang Xu
>
> **备注:** Accepted by ICCV 2025 AISTORY Workshop
>
> **摘要:** Traditional Celluloid (Cel) Animation production pipeline encompasses multiple essential steps, including storyboarding, layout design, keyframe animation, inbetweening, and colorization, which demand substantial manual effort, technical expertise, and significant time investment. These challenges have historically impeded the efficiency and scalability of Cel-Animation production. The rise of generative artificial intelligence (GenAI), encompassing large language models, multimodal models, and diffusion models, offers innovative solutions by automating tasks such as inbetween frame generation, colorization, and storyboard creation. This survey explores how GenAI integration is revolutionizing traditional animation workflows by lowering technical barriers, broadening accessibility for a wider range of creators through tools like AniDoc, ToonCrafter, and AniSora, and enabling artists to focus more on creative expression and artistic innovation. Despite its potential, challenges like visual consistency, stylistic coherence, and ethical considerations persist. Additionally, this paper explores future directions and advancements in AI-assisted animation. For further exploration and resources, please visit our GitHub repository: https://github.com/yunlong10/Awesome-AI4Animation
>
---
#### [replaced 065] HoliSafe: Holistic Safety Benchmarking and Modeling for Vision-Language Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型（VLM）安全问题，提出全景安全基准与建模框架HoliSafe。解决现有方法对图文交互危害覆盖不足及依赖数据调优的缺陷。构建涵盖五类图文组合的HoliSafe-Bench，并设计可插拔的视觉守卫模块（VGM），实现安全增强与决策可解释性，显著提升VLM安全性。**

- **链接: [https://arxiv.org/pdf/2506.04704v5](https://arxiv.org/pdf/2506.04704v5)**

> **作者:** Youngwan Lee; Kangsan Kim; Kwanyong Park; Ilcahe Jung; Soojin Jang; Seanie Lee; Yong-Ju Lee; Sung Ju Hwang
>
> **备注:** Project page: https://youngwanlee.github.io/holisafe
>
> **摘要:** Despite emerging efforts to enhance the safety of Vision-Language Models (VLMs), current approaches face two main shortcomings. 1) Existing safety-tuning datasets and benchmarks only partially consider how image-text interactions can yield harmful content, often overlooking contextually unsafe outcomes from seemingly benign pairs. This narrow coverage leaves VLMs vulnerable to jailbreak attacks in unseen configurations. 2) Prior methods rely primarily on data-centric tuning, with limited architectural innovations to intrinsically strengthen safety. We address these gaps by introducing a holistic safety dataset and benchmark, \textbf{HoliSafe}, that spans all five safe/unsafe image-text combinations, providing a more robust basis for both training and evaluation (HoliSafe-Bench). We further propose a novel modular framework for enhancing VLM safety with a visual guard module (VGM) designed to assess the harmfulness of input images for VLMs. This module endows VLMs with a dual functionality: they not only learn to generate safer responses but can also provide an interpretable harmfulness classification to justify their refusal decisions. A significant advantage of this approach is its modularity; the VGM is designed as a plug-in component, allowing for seamless integration with diverse pre-trained VLMs across various scales. Experiments show that Safe-VLM with VGM, trained on our HoliSafe, achieves state-of-the-art safety performance across multiple VLM benchmarks. Additionally, the HoliSafe-Bench itself reveals critical vulnerabilities in existing VLM models. We hope that HoliSafe and VGM will spur further research into robust and interpretable VLM safety, expanding future avenues for multimodal alignment.
>
---
#### [replaced 066] FlagEval Findings Report: A Preliminary Evaluation of Large Reasoning Models on Automatically Verifiable Textual and Visual Questions
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对大推理模型（LRM）的评估问题，提出无污染的中等规模评测方案，并发布视觉语言推理基准ROME。旨在评估模型在文本与视觉线索下的自动可验证推理能力，推动多模态推理模型的客观评测。**

- **链接: [https://arxiv.org/pdf/2509.17177v3](https://arxiv.org/pdf/2509.17177v3)**

> **作者:** Bowen Qin; Chen Yue; Fang Yin; Hui Wang; JG Yao; Jiakang Liu; Jing-Shu Zheng; Miguel Hu Chen; Richeng Xuan; Shibei Meng; Shiqi Zhou; Teng Dai; Tong-Shuai Ren; Wei Cui; Xi Yang; Xialin Du; Xiaojing Xu; Xue Sun; Xuejing Li; Yaming Liu; Yesheng Liu; Ying Liu; Yonghua Lin; Yu Zhao; Yunduo Zhang; Yuwen Luo; Zheqi He; Zhiyuan He; Zhongyuan Wang
>
> **备注:** Project homepage: https://flageval-baai.github.io/LRM-Eval/ This work will also be presented at NeurIPS 2025 Workshop on Foundations of Reasoning in Language Models (FoRLM); update with trials on Gemini 3 Pro
>
> **摘要:** We conduct a moderate-scale contamination-free (to some extent) evaluation of current large reasoning models (LRMs) with some preliminary findings. We also release ROME, our evaluation benchmark for vision language models intended to test reasoning from visual clues. We attach links to the benchmark, evaluation data, and other updates on this website: https://flageval-baai.github.io/LRM-Eval/
>
---
#### [replaced 067] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对大视觉语言模型空间理解能力弱的问题，提出自监督强化学习框架Spatial-SSRL。通过从普通图像中自动构建五种无需标注的预训练任务，实现无需人工监督的空间结构学习，显著提升模型在多场景下的空间推理能力，为大规模增强视觉模型空间智能提供可行路径。**

- **链接: [https://arxiv.org/pdf/2510.27606v2](https://arxiv.org/pdf/2510.27606v2)**

> **作者:** Yuhong Liu; Beichen Zhang; Yuhang Zang; Yuhang Cao; Long Xing; Xiaoyi Dong; Haodong Duan; Dahua Lin; Jiaqi Wang
>
> **备注:** preprint
>
> **摘要:** Spatial understanding remains a weakness of Large Vision-Language Models (LVLMs). Existing supervised fine-tuning (SFT) and recent reinforcement learning with verifiable rewards (RLVR) pipelines depend on costly supervision, specialized tools, or constrained environments that limit scale. We introduce Spatial-SSRL, a self-supervised RL paradigm that derives verifiable signals directly from ordinary RGB or RGB-D images. Spatial-SSRL automatically formulates five pretext tasks that capture 2D and 3D spatial structure: shuffled patch reordering, flipped patch recognition, cropped patch inpainting, regional depth ordering, and relative 3D position prediction. These tasks provide ground-truth answers that are easy to verify and require no human or LVLM annotation. Training on our tasks substantially improves spatial reasoning while preserving general visual capabilities. On seven spatial understanding benchmarks in both image and video settings, Spatial-SSRL delivers average accuracy gains of 4.63% (3B) and 3.89% (7B) over the Qwen2.5-VL baselines. Our results show that simple, intrinsic supervision enables RLVR at scale and provides a practical route to stronger spatial intelligence in LVLMs.
>
---
#### [replaced 068] DBGroup: Dual-Branch Point Grouping for Weakly Supervised 3D Semantic Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文提出DBGroup，一种两阶段弱监督3D语义实例分割框架，解决标注成本高、依赖专家标注的问题。通过多视图图像提取语义与掩码线索，生成伪标签并进行精细化优化，结合自训练与实例掩码过滤，实现高效高质量分割，性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10003v2](https://arxiv.org/pdf/2511.10003v2)**

> **作者:** Xuexun Liu; Xiaoxu Xu; Qiudan Zhang; Lin Ma; Xu Wang
>
> **摘要:** Weakly supervised 3D instance segmentation is essential for 3D scene understanding, especially as the growing scale of data and high annotation costs associated with fully supervised approaches. Existing methods primarily rely on two forms of weak supervision: one-thing-one-click annotations and bounding box annotations, both of which aim to reduce labeling efforts. However, these approaches still encounter limitations, including labor-intensive annotation processes, high complexity, and reliance on expert annotators. To address these challenges, we propose \textbf{DBGroup}, a two-stage weakly supervised 3D instance segmentation framework that leverages scene-level annotations as a more efficient and scalable alternative. In the first stage, we introduce a Dual-Branch Point Grouping module to generate pseudo labels guided by semantic and mask cues extracted from multi-view images. To further improve label quality, we develop two refinement strategies: Granularity-Aware Instance Merging and Semantic Selection and Propagation. The second stage involves multi-round self-training on an end-to-end instance segmentation network using the refined pseudo-labels. Additionally, we introduce an Instance Mask Filter strategy to address inconsistencies within the pseudo labels. Extensive experiments demonstrate that DBGroup achieves competitive performance compared to sparse-point-level supervised 3D instance segmentation methods, while surpassing state-of-the-art scene-level supervised 3D semantic segmentation approaches. Code is available at https://github.com/liuxuexun/DBGroup.
>
---
#### [replaced 069] Yanyun-3: Enabling Cross-Platform Strategy Game Operation with Vision-Language Models
- **分类: cs.AI; cs.CV**

- **简介: 该论文针对跨平台策略游戏自动化难题，提出Yanyun-3框架，利用视觉语言模型实现界面理解与动作执行。通过引入组合粒度的数据组织方法，优化多模态数据融合，显著提升泛化能力与推理效率，无需平台特化调优即可完成核心任务。**

- **链接: [https://arxiv.org/pdf/2511.12937v2](https://arxiv.org/pdf/2511.12937v2)**

> **作者:** Guoyan Wang; Yanyan Huang; Chunlin Chen; Lifeng Wang; Yuxiang Sun
>
> **备注:** 32 pages, 13 figures
>
> **摘要:** Cross-platform strategy game automation remains a challenge due to diverse user interfaces and dynamic battlefield environments. Existing Vision--Language Models (VLMs) struggle with generalization across heterogeneous platforms and lack precision in interface understanding and action execution. We introduce Yanyun-3, a VLM-based agent that integrates Qwen2.5-VL for visual reasoning and UI-TARS for interface execution. We propose a novel data organization principle -- combination granularity -- to distinguish intra-sample fusion and inter-sample mixing of multimodal data (static images, multi-image sequences, and videos). The model is fine-tuned using QLoRA on a curated dataset across three strategy game platforms. The optimal strategy (M*V+S) achieves a 12.98x improvement in BLEU-4 score and a 63% reduction in inference time compared to full fusion. Yanyun-3 successfully executes core tasks (e.g., target selection, resource allocation) across platforms without platform-specific tuning. Our findings demonstrate that structured multimodal data organization significantly enhances VLM performance in embodied tasks. Yanyun-3 offers a generalizable framework for GUI automation, with broader implications for robotics and autonomous systems.
>
---
#### [replaced 070] Prompt Guiding Multi-Scale Adaptive Sparse Representation-driven Network for Low-Dose CT MAR
- **分类: cs.CV**

- **简介: 该论文针对低剂量CT金属伪影去除（LDMAR）任务，解决现有方法忽视多尺度信息及需为不同剂量训练多个模型的问题。提出PMSRNet，通过提示引导的自适应阈值生成与多尺度系数融合，实现单模型跨剂量泛化，显著提升重建质量与可解释性。**

- **链接: [https://arxiv.org/pdf/2504.19687v3](https://arxiv.org/pdf/2504.19687v3)**

> **作者:** Baoshun Shi; Bing Chen; Shaolei Zhang; Huazhu Fu; Zhanli Hu
>
> **摘要:** Low-dose CT (LDCT) is capable of reducing X-ray radiation exposure, but it will potentially degrade image quality, even yields metal artifacts at the case of metallic implants. For simultaneous LDCT reconstruction and metal artifact reduction (LDMAR), existing deep learning-based efforts face two main limitations: i) the network design neglects multi-scale and within-scale information; ii) training a distinct model for each dose necessitates significant storage space for multiple doses. To fill these gaps, we propose a prompt guiding multi-scale adaptive sparse representation-driven network, abbreviated as PMSRNet, for LDMAR task. Specifically, we construct PMSRNet inspired from multi-scale sparsifying frames, and it can simultaneously employ within-scale characteristics and cross-scale complementarity owing to an elaborated prompt guiding scale-adaptive threshold generator (PSATG) and a built multi-scale coefficient fusion module (MSFuM). The PSATG can adaptively capture multiple contextual information to generate more faithful thresholds, achieved by fusing features from local, regional, and global levels. Furthermore, we elaborate a model interpretable dual domain LDMAR framework called PDuMSRNet, and train single model with a prompt guiding strategy for multiple dose levels. We build a prompt guiding module, whose input contains dose level, metal mask and input instance, to provide various guiding information, allowing a single model to accommodate various CT dose settings. Extensive experiments at various dose levels demonstrate that the proposed methods outperform the state-of-the-art LDMAR methods.
>
---
#### [replaced 071] KEPT: Knowledge-Enhanced Prediction of Trajectories from Consecutive Driving Frames with Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶中短时轨迹预测任务，解决现有视觉语言模型对驾驶场景理解不足、轨迹预测不可靠的问题。提出KEPT框架，融合时频空视频编码器与知识检索增强生成，通过三阶段微调提升模型空间感知与预测能力，实现更可信的轨迹预测。**

- **链接: [https://arxiv.org/pdf/2509.02966v2](https://arxiv.org/pdf/2509.02966v2)**

> **作者:** Yujin Wang; Tianyi Wang; Quanfeng Liu; Wenxian Fan; Junfeng Jiao; Christian Claudel; Yunbing Yan; Bingzhao Gao; Jianqiang Wang; Hong Chen
>
> **摘要:** Accurate short-horizon trajectory prediction is crucial for safe and reliable autonomous driving. However, existing vision-language models (VLMs) often fail to accurately understand driving scenes and generate trustworthy trajectories. To address this challenge, this paper introduces KEPT, a knowledge-enhanced VLM framework that predicts ego trajectories directly from consecutive front-view driving frames. KEPT integrates a temporal frequency-spatial fusion (TFSF) video encoder, which is trained via self-supervised learning with hard-negative mining, with a k-means & HNSW retrieval-augmented generation (RAG) pipeline. Retrieved prior knowledge is added into chain-of-thought (CoT) prompts with explicit planning constraints, while a triple-stage fine-tuning paradigm aligns the VLM backbone to enhance spatial perception and trajectory prediction capabilities. Evaluated on nuScenes dataset, KEPT achieves the best open-loop performance compared with baseline methods. Ablation studies on fine-tuning stages, Top-K value of RAG, different retrieval strategies, vision encoders, and VLM backbones are conducted to demonstrate the effectiveness of KEPT. These results indicate that KEPT offers a promising, data-efficient way toward trustworthy trajectory prediction in autonomous driving.
>
---
#### [replaced 072] Temporally Compressed 3D Gaussian Splatting for Dynamic Scenes
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对动态场景三维重建中的内存与效率问题，提出时空压缩的3D高斯点云渲染方法（TC3DGS）。通过基于时间相关性的剪枝、梯度感知的混合精度量化及轨迹插值，实现高达67倍的压缩比，显著降低存储与计算开销，同时保持高质量视觉效果。**

- **链接: [https://arxiv.org/pdf/2412.05700v2](https://arxiv.org/pdf/2412.05700v2)**

> **作者:** Saqib Javed; Ahmad Jarrar Khan; Corentin Dumery; Chen Zhao; Mathieu Salzmann
>
> **备注:** Accepted at British Machine Vision Conference (BMVC) 2025
>
> **摘要:** Recent advancements in high-fidelity dynamic scene reconstruction have leveraged dynamic 3D Gaussians and 4D Gaussian Splatting for realistic scene representation. However, to make these methods viable for real-time applications such as AR/VR, gaming, and rendering on low-power devices, substantial reductions in memory usage and improvements in rendering efficiency are required. While many state-of-the-art methods prioritize lightweight implementations, they struggle in handling {scenes with complex motions or long sequences}. In this work, we introduce Temporally Compressed 3D Gaussian Splatting (TC3DGS), a novel technique designed specifically to effectively compress dynamic 3D Gaussian representations. TC3DGS selectively prunes Gaussians based on their temporal relevance and employs gradient-aware mixed-precision quantization to dynamically compress Gaussian parameters. In addition, TC3DGS exploits an adapted version of the Ramer-Douglas-Peucker algorithm to further reduce storage by interpolating Gaussian trajectories across frames. Our experiments on multiple datasets demonstrate that TC3DGS achieves up to 67$\times$ compression with minimal or no degradation in visual quality. More results and videos are provided in the supplementary. Project Page: https://ahmad-jarrar.github.io/tc-3dgs/
>
---
#### [replaced 073] Localizing Knowledge in Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文研究扩散变压器（DiT）中知识的层间分布，旨在解决生成模型可解释性与可控性问题。提出一种模型无关的知识定位方法，识别知识编码的特定块，并应用于个性化与知识删减，实现高效、精准的模型编辑。**

- **链接: [https://arxiv.org/pdf/2505.18832v2](https://arxiv.org/pdf/2505.18832v2)**

> **作者:** Arman Zarei; Samyadeep Basu; Keivan Rezaei; Zihao Lin; Sayan Nag; Soheil Feizi
>
> **摘要:** Understanding how knowledge is distributed across the layers of generative models is crucial for improving interpretability, controllability, and adaptation. While prior work has explored knowledge localization in UNet-based architectures, Diffusion Transformer (DiT)-based models remain underexplored in this context. In this paper, we propose a model- and knowledge-agnostic method to localize where specific types of knowledge are encoded within the DiT blocks. We evaluate our method on state-of-the-art DiT-based models, including PixArt-alpha, FLUX, and SANA, across six diverse knowledge categories. We show that the identified blocks are both interpretable and causally linked to the expression of knowledge in generated outputs. Building on these insights, we apply our localization framework to two key applications: model personalization and knowledge unlearning. In both settings, our localized fine-tuning approach enables efficient and targeted updates, reducing computational cost, improving task-specific performance, and better preserving general model behavior with minimal interference to unrelated or surrounding content. Overall, our findings offer new insights into the internal structure of DiTs and introduce a practical pathway for more interpretable, efficient, and controllable model editing.
>
---
#### [replaced 074] DVP-MVS: Synergize Depth-Edge and Visibility Prior for Multi-View Stereo
- **分类: cs.CV**

- **简介: 该论文针对多视图立体（MVS）任务，解决patch变形中因边缘误跳和可见性遮挡导致的不稳定性问题。提出DVP-MVS，融合深度-边缘对齐与跨视图先验，通过精细边界引导和可见性恢复，实现鲁棒、可见性感知的变形，提升重建精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2412.11578v3](https://arxiv.org/pdf/2412.11578v3)**

> **作者:** Zhenlong Yuan; Jinguo Luo; Fei Shen; Zhaoxin Li; Cong Liu; Tianlu Mao; Zhaoqi Wang
>
> **摘要:** Patch deformation-based methods have recently exhibited substantial effectiveness in multi-view stereo, due to the incorporation of deformable and expandable perception to reconstruct textureless areas. However, such approaches typically focus on exploring correlative reliable pixels to alleviate match ambiguity during patch deformation, but ignore the deformation instability caused by mistaken edge-skipping and visibility occlusion, leading to potential estimation deviation. To remedy the above issues, we propose DVP-MVS, which innovatively synergizes depth-edge aligned and cross-view prior for robust and visibility-aware patch deformation. Specifically, to avoid unexpected edge-skipping, we first utilize Depth Anything V2 followed by the Roberts operator to initialize coarse depth and edge maps respectively, both of which are further aligned through an erosion-dilation strategy to generate fine-grained homogeneous boundaries for guiding patch deformation. In addition, we reform view selection weights as visibility maps and restore visible areas by cross-view depth reprojection, then regard them as cross-view prior to facilitate visibility-aware patch deformation. Finally, we improve propagation and refinement with multi-view geometry consistency by introducing aggregated visible hemispherical normals based on view selection and local projection depth differences based on epipolar lines, respectively. Extensive evaluations on ETH3D and Tanks & Temples benchmarks demonstrate that our method can achieve state-of-the-art performance with excellent robustness and generalization.
>
---
#### [replaced 075] Comparison of Generative Learning Methods for Turbulence Surrogates
- **分类: physics.flu-dyn; cs.CV**

- **简介: 该论文研究生成式模型在湍流代理建模中的应用，旨在降低高成本湍流模拟的计算负担。针对2D卡门涡街及实验涡流数据，比较VAE、DCGAN和DDPM三种模型，评估其对湍流统计特性与空间结构的捕捉能力。结果表明DCGAN在速度与精度上表现最优，具备高效训练与推理优势。**

- **链接: [https://arxiv.org/pdf/2411.16417v2](https://arxiv.org/pdf/2411.16417v2)**

> **作者:** Claudia Drygala; Edmund Ross; Francesca di Mare; Hanno Gottschalk
>
> **摘要:** Numerical simulations of turbulent flows present significant challenges in fluid dynamics due to their complexity and high computational cost. High resolution techniques such as Direct Numerical Simulation (DNS) and Large Eddy Simulation (LES) are generally not computationally affordable, particularly for technologically relevant problems. Recent advances in machine learning, specifically in generative probabilistic models, offer promising alternatives as surrogates for turbulence. This paper investigates the application of three generative models - Variational Autoencoders (VAE), Deep Convolutional Generative Adversarial Networks (DCGAN), and Denoising Diffusion Probabilistic Models (DDPM) - in simulating a von Kármán vortex street around a fixed cylinder projected into 2D, as well as a real-world experimental dataset of the wake flow of a cylinder array. Training data was obtained by means of LES in the simulated case and Particle Image Velocimetry (PIV) in the experimental case. We evaluate each model's ability to capture the statistical properties and spatial structures of the turbulent flow. Our results demonstrate that DDPM and DCGAN effectively replicate all flow distributions, highlighting their potential as efficient and accurate tools for turbulence surrogacy. We find a strong argument for DCGAN, as although they are more difficult to train (due to problems such as mode collapse), they show the fastest inference and training time, require less data to train compared to VAE and DDPM, and provide the results most closely aligned with the input stream. In contrast, VAE train quickly (and can generate samples quickly) but do not produce adequate results, and DDPM, whilst effective, are significantly slower at both, inference and training time.
>
---
#### [replaced 076] Zero-Shot Anomaly Detection with Dual-Branch Prompt Selection
- **分类: cs.CV**

- **简介: 该论文针对零样本异常检测（ZSAD）在域偏移下泛化能力差的问题，提出PILOT框架。通过双分支提示学习动态融合可学习提示与语义属性，并引入无标签测试时自适应策略，利用高置信度伪标签更新提示参数，显著提升模型在新域下的异常检测与定位性能。**

- **链接: [https://arxiv.org/pdf/2508.00777v3](https://arxiv.org/pdf/2508.00777v3)**

> **作者:** Zihan Wang; Samira Ebrahimi Kahou; Narges Armanfard
>
> **备注:** Accepted at BMVC 2025
>
> **摘要:** Zero-shot anomaly detection (ZSAD) enables identifying and localizing defects in unseen categories by relying solely on generalizable features rather than requiring any labeled examples of anomalies. However, existing ZSAD methods, whether using fixed or learned prompts, struggle under domain shifts because their training data are derived from limited training domains and fail to generalize to new distributions. In this paper, we introduce PILOT, a framework designed to overcome these challenges through two key innovations: (1) a novel dual-branch prompt learning mechanism that dynamically integrates a pool of learnable prompts with structured semantic attributes, enabling the model to adaptively weight the most relevant anomaly cues for each input image; and (2) a label-free test-time adaptation strategy that updates the learnable prompt parameters using high-confidence pseudo-labels from unlabeled test data. Extensive experiments on 13 industrial and medical benchmarks demonstrate that PILOT achieves state-of-the-art performance in both anomaly detection and localization under domain shift.
>
---
#### [replaced 077] FedPromo: Federated Lightweight Proxy Models at the Edge Bring New Domains to Foundation Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出FedPromo框架，解决大模型在资源受限客户端上联邦学习的计算瓶颈问题。通过轻量级代理模型与知识蒸馏结合，实现高效跨域适应，保护隐私的同时降低开销，显著提升性能。任务为边缘端个性化图像分类。**

- **链接: [https://arxiv.org/pdf/2508.03356v2](https://arxiv.org/pdf/2508.03356v2)**

> **作者:** Matteo Caligiuri; Francesco Barbato; Donald Shenaj; Umberto Michieli; Pietro Zanuttigh
>
> **备注:** 8 pages (main document) + 13 pages (suppl. mat.), 4 figures (main) + 11 figures (suppl. mat.), 6 tables (main) + 5 tables (suppl. mat.) + 4 algorithms (suppl. mat.)
>
> **摘要:** Federated Learning (FL) is an established paradigm for training deep learning models on decentralized data. However, as the size of the models grows, conventional FL approaches often require significant computational resources on client devices, which may not be feasible. We introduce FedPromo, a novel framework that enables efficient adaptation of large-scale foundation models stored on a central server to new domains encountered only by remote clients. Instead of directly training the large model on client devices, FedPromo optimizes lightweight proxy models via FL, significantly reducing computational overhead while maintaining privacy. Our method follows a two-stage process: first, server-side knowledge distillation aligns the representations of a large-scale foundation model (e.g., a transformer) with those of a compact counterpart (e.g., a CNN). Then, the compact model encoder is deployed to client devices, where trainable classifiers are learned locally. These classifiers are subsequently aggregated and seamlessly transferred back to the foundation model, facilitating personalized adaptation without requiring direct access to user data. Through novel regularization strategies, our framework enables decentralized multi-domain learning, balancing performance, privacy, and resource efficiency. Extensive experiments on five image classification benchmarks demonstrate that FedPromo outperforms existing methods while assuming limited-resource clients.
>
---
#### [replaced 078] LiHi-GS: LiDAR-Supervised Gaussian Splatting for Highway Driving Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中的高速公路场景重建任务，解决现有方法忽视高速场景与未充分利用LiDAR深度信息的问题。提出LiHi-GS，通过LiDAR监督提升3D场景重建精度，并支持LiDAR渲染，实现更真实的动态场景合成与编辑。**

- **链接: [https://arxiv.org/pdf/2412.15447v3](https://arxiv.org/pdf/2412.15447v3)**

> **作者:** Pou-Chun Kung; Xianling Zhang; Katherine A. Skinner; Nikita Jaipuria
>
> **备注:** RA-L 2025
>
> **摘要:** Photorealistic 3D scene reconstruction plays an important role in autonomous driving, enabling the generation of novel data from existing datasets to simulate safety-critical scenarios and expand training data without additional acquisition costs. Gaussian Splatting (GS) facilitates real-time, photorealistic rendering with an explicit 3D Gaussian representation of the scene, providing faster processing and more intuitive scene editing than the implicit Neural Radiance Fields (NeRFs). While extensive GS research has yielded promising advancements in autonomous driving applications, they overlook two critical aspects: First, existing methods mainly focus on low-speed and feature-rich urban scenes and ignore the fact that highway scenarios play a significant role in autonomous driving. Second, while LiDARs are commonplace in autonomous driving platforms, existing methods learn primarily from images and use LiDAR only for initial estimates or without precise sensor modeling, thus missing out on leveraging the rich depth information LiDAR offers and limiting the ability to synthesize LiDAR data. In this paper, we propose a novel GS method for dynamic scene synthesis and editing with improved scene reconstruction through LiDAR supervision and support for LiDAR rendering. Unlike prior works that are tested mostly on urban datasets, to the best of our knowledge, we are the first to focus on the more challenging and highly relevant highway scenes for autonomous driving, with sparse sensor views and monotone backgrounds. Visit our project page at: https://umautobots.github.io/lihi_gs
>
---
#### [replaced 079] TK-Mamba: Marrying KAN With Mamba for Text-Driven 3D Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对3D医学图像分割任务，解决高维数据与复杂空间依赖带来的计算效率低、上下文建模弱的问题。提出TK-Mamba框架，融合Mamba与KAN，引入3D-GR-KAN实现高效非线性变换，并设计双分支文本驱动策略，利用医学文本增强语义对齐，提升分割精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2505.18525v2](https://arxiv.org/pdf/2505.18525v2)**

> **作者:** Haoyu Yang; Yutong Guan; Meixing Shi; Yuxiang Cai; Jintao Chen; Sun Bing; Wenhui Lei; Mianxin Liu; Xiaoming Shi; Yankai Jiang; Jianwei Yin
>
> **摘要:** 3D medical image segmentation is important for clinical diagnosis and treatment but faces challenges from high-dimensional data and complex spatial dependencies. Traditional single-modality networks, such as CNNs and Transformers, are often limited by computational inefficiency and constrained contextual modeling in 3D settings. To alleviate these limitations, we propose TK-Mamba, a multimodal framework that fuses the linear-time Mamba with Kolmogorov-Arnold Networks (KAN) to form an efficient hybrid backbone. Our approach is characterized by two primary technical contributions. Firstly, we introduce the novel 3D-Group-Rational KAN (3D-GR-KAN), which marks the first application of KAN in 3D medical imaging, providing a superior and computationally efficient nonlinear feature transformation crucial for complex volumetric structures. Secondly, we devise a dual-branch text-driven strategy using Pubmedclip's embeddings. This strategy significantly enhances segmentation robustness and accuracy by simultaneously capturing inter-organ semantic relationships to mitigate label inconsistencies and aligning image features with anatomical texts. By combining this advanced backbone and vision-language knowledge, TK-Mamba offers a unified and scalable solution for both multi-organ and tumor segmentation. Experiments on multiple datasets demonstrate that our framework achieves state-of-the-art performance in both organ and tumor segmentation tasks, surpassing existing methods in both accuracy and efficiency. Our code is publicly available at https://github.com/yhy-whu/TK-Mamba
>
---
#### [replaced 080] WeatherDiffusion: Controllable Weather Editing in Intrinsic Space
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出WeatherDiffusion，一种基于扩散模型的可控天气编辑框架。针对传统像素空间编辑控制性差的问题，通过逆渲染生成材质、几何、光照等内在图，并结合文本提示与CLIP空间插值实现精细天气控制。构建了包含3.8万张合成与1.8万张真实图像的数据集，显著提升复杂户外场景下的天气编辑效果，适用于自动驾驶等下游任务。**

- **链接: [https://arxiv.org/pdf/2508.06982v2](https://arxiv.org/pdf/2508.06982v2)**

> **作者:** Yixin Zhu; Zuoliang Zhu; Jian Yang; Miloš Hašan; Jin Xie; Beibei Wang
>
> **摘要:** We present WeatherDiffusion, a diffusion-based framework for controllable weather editing in intrinsic space. Our framework includes two components based on diffusion priors: an inverse renderer that estimates material properties, scene geometry, and lighting as intrinsic maps from an input image, and a forward renderer that utilizes these geometry and material maps along with a text prompt that describes specific weather conditions to generate a final image. The intrinsic maps enhance controllability compared to traditional pixel-space editing approaches.We propose an intrinsic map-aware attention mechanism that improves spatial correspondence and decomposition quality in large outdoor scenes. For forward rendering, we leverage CLIP-space interpolation of weather prompts to achieve fine-grained weather control. We also introduce a synthetic and a real-world dataset, containing 38k and 18k images under various weather conditions, each with intrinsic map annotations. WeatherDiffusion outperforms state-of-the-art pixel-space editing approaches, weather restoration methods, and rendering-based methods, showing promise for downstream tasks such as autonomous driving, enhancing the robustness of detection and segmentation in challenging weather scenarios.
>
---
#### [replaced 081] Leveraging Unlabeled Data from Unknown Sources via Dual-Path Guidance for Deepfake Face Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对深度伪造人脸检测中缺乏标注数据的问题，提出双路径引导网络（DPGNet），通过文本引导跨域对齐和课程驱动伪标签生成，有效利用未知来源的无标签数据，解决不同生成模型间域差异与真实/伪造语义混淆问题，提升检测泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.09022v3](https://arxiv.org/pdf/2508.09022v3)**

> **作者:** Zhiqiang Yang; Renshuai Tao; Chunjie Zhang; guodong yang; Xiaolong Zheng; Yao Zhao
>
> **备注:** 11pages,4figures
>
> **摘要:** Existing deepfake detection methods heavily rely on static labeled datasets. However, with the proliferation of generative models, real-world scenarios are flooded with massive amounts of unlabeled fake face data from unknown sources. This presents a critical dilemma: detectors relying solely on existing data face generalization failure, while manual labeling for this new stream is infeasible due to the high realism of fakes. A more fundamental challenge is that, unlike typical unsupervised learning tasks where categories are clearly defined, real and fake faces share the same semantics, which leads to a decline in the performance of traditional unsupervised strategies. Therefore, there is an urgent need for a new paradigm designed specifically for this scenario to effectively utilize these unlabeled data. Accordingly, this paper proposes a dual-path guided network (DPGNet) to address two key challenges: (1) bridging the domain differences between faces generated by different generative models; and (2) utilizing unlabeled image samples. The method comprises two core modules: text-guided cross-domain alignment, which uses learnable cues to unify visual and textual embeddings into a domain-invariant feature space; and curriculum-driven pseudo-label generation, which dynamically utilizes unlabeled samples. Extensive experiments on multiple mainstream datasets show that DPGNet significantly outperforms existing techniques,, highlighting its effectiveness in addressing the challenges posed by the deepfakes using unlabeled data.
>
---
#### [replaced 082] CLIP-IT: CLIP-based Pairing for Histology Images Classification
- **分类: cs.CV**

- **简介: 该论文提出CLIP-IT，用于组织学图像分类任务。针对现有视觉语言模型依赖大量配对数据的问题，利用未配对的病理报告通过CLIP模型检索相关文本，构建伪配对数据，以低开销实现多模态知识迁移，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2504.16181v5](https://arxiv.org/pdf/2504.16181v5)**

> **作者:** Banafsheh Karimian; Giulia Avanzato; Soufian Belharbi; Alexis Guichemerre; Luke McCaffrey; Mohammadhadi Shateri; Eric Granger
>
> **摘要:** Multimodal learning has shown promise in medical imaging, combining complementary modalities like images and text. Vision-language models (VLMs) capture rich diagnostic cues but often require large paired datasets and prompt- or text-based inference, limiting their practicality due to annotation cost, privacy, and compute demands. Crucially, available free unpaired external text, like pathology reports, can still provide complementary diagnostic cues if semantically relevant content is retrievable per image. To address this, we introduce CLIP-IT, a novel framework that relies on rich unpaired text reports. Specifically, CLIP-IT uses a CLIP model pre-trained on histology image-text pairs from a separate dataset to retrieve the most relevant unpaired textual report for each image in the downstream unimodal dataset. These reports, sourced from the same disease domain and tissue type, form pseudo-pairs that reflect shared clinical semantics rather than exact alignment. Knowledge from these texts is distilled into the vision model during training, while LoRA-based adaptation mitigates the semantic gap between unaligned modalities. At inference, only the vision model is used, keeping overhead low while still benefiting from multimodal training without requiring paired data in the downstream dataset. Experiments on histology image datasets confirm that CLIP-IT consistently improves classification accuracy over both unimodal and multimodal CLIP-based baselines in most cases, without the burden of per-dataset paired annotation or inference-time complexity.
>
---
#### [replaced 083] Scalable FPGA Framework for Real-Time Denoising in High-Throughput Imaging: A DRAM-Optimized Pipeline using High-Level Synthesis
- **分类: cs.AR; cs.CV; cs.DC; eess.IV; eess.SP; physics.ins-det**

- **简介: 该论文针对高通量成像中实时去噪难题，提出基于FPGA的可扩展预处理框架。利用高层次综合与DRAM优化流水线，实现低延迟帧减法与平均，满足PRISM等系统实时性需求，降低后续分析数据量。**

- **链接: [https://arxiv.org/pdf/2508.14917v2](https://arxiv.org/pdf/2508.14917v2)**

> **作者:** Weichien Liao
>
> **备注:** FPGA-based denoising pipeline for PRISM-scale imaging. Real-time frame subtraction and averaging via burst-mode AXI4 and DRAM buffering. Benchmarked against CPU/GPU workflows; scalable across multi-bank FPGA setups. Acknowledgements revised for consistency with journal submission; scientific content remains unchanged
>
> **摘要:** High-throughput imaging workflows, such as Parallel Rapid Imaging with Spectroscopic Mapping (PRISM), generate data at rates that exceed conventional real-time processing capabilities. We present a scalable FPGA-based preprocessing pipeline for real-time denoising, implemented via High-Level Synthesis (HLS) and optimized for DRAM-backed buffering. Our architecture performs frame subtraction and averaging directly on streamed image data, minimizing latency through burst-mode AXI4 interfaces. The resulting kernel operates below the inter-frame interval, enabling inline denoising and reducing dataset size for downstream CPU/GPU analysis. Validated under PRISM-scale acquisition, this modular FPGA framework offers a practical solution for latency-sensitive imaging workflows in spectroscopy and microscopy.
>
---
#### [replaced 084] Exploring Convolutional Neural Networks for Rice Grain Classification: An Explainable AI Approach
- **分类: cs.CV**

- **简介: 该论文针对水稻品种自动分类任务，旨在解决人工分类效率低、易出错的问题。提出基于CNN的自动分类框架，结合LIME与SHAP实现模型可解释性，显著提升分类准确率并揭示决策依据。**

- **链接: [https://arxiv.org/pdf/2505.05513v4](https://arxiv.org/pdf/2505.05513v4)**

> **作者:** Muhammad Junaid Asif; Hamza Khan; Rabia Tehseen; Rana Fayyaz Ahmad; Mujtaba Asad; Syed Tahir Hussain Rizvi; Shazia Saqib
>
> **摘要:** Rice is an essential staple food worldwide that is important in promoting international trade, economic growth, and nutrition. Asian countries such as China, India, Pakistan, Thailand, Vietnam, and Indonesia are notable for their significant contribution to the cultivation and utilization of rice. These nations are also known for cultivating different rice grains, including short and long grains. These sizes are further classified as basmati, jasmine, kainat saila, ipsala, arborio, etc., catering to diverse culinary preferences and cultural traditions. For both local and international trade, inspecting and maintaining the quality of rice grains to satisfy customers and preserve a country's reputation is necessary. Manual quality check and classification is quite a laborious and time-consuming process. It is also highly prone to mistakes. Therefore, an automatic solution must be proposed for the effective and efficient classification of different varieties of rice grains. This research paper presents an automatic framework based on a convolutional neural network (CNN) for classifying different varieties of rice grains. We evaluated the proposed model based on performance metrics such as accuracy, recall, precision, and F1-Score. The CNN model underwent rigorous training and validation, achieving a remarkable accuracy rate and a perfect area under each class's Receiver Operating Characteristic (ROC) curve. The confusion matrix analysis confirmed the model's effectiveness in distinguishing between the different rice varieties, indicating minimal misclassifications. Additionally, the integration of explainability techniques such as LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) provided valuable insights into the model's decision-making process, revealing how specific features of the rice grains influenced classification outcomes.
>
---
#### [replaced 085] Detecting Cultural Differences in News Video Thumbnails via Computational Aesthetics
- **分类: cs.CY; cs.CV**

- **简介: 该论文属于跨文化视觉风格分析任务，旨在检测中美新闻视频缩略图的文化差异。通过内容聚类与美学特征对比，分析2,400个来自中美YouTube频道的缩略图，发现美式缩略图更正式、细节更丰富、构图更随意，而中式更自然、色彩更鲜艳，揭示了文化偏好对视觉呈现的影响。**

- **链接: [https://arxiv.org/pdf/2505.21912v2](https://arxiv.org/pdf/2505.21912v2)**

> **作者:** Marvin Limpijankit; John Kender
>
> **备注:** ICWSM'24 Workshop
>
> **摘要:** We propose a two-step approach for detecting differences in the style of images across sources of differing cultural affinity, where images are first clustered into finer visual themes based on content before their aesthetic features are compared. We test this approach on 2,400 YouTube video thumbnails taken equally from two U.S. and two Chinese YouTube channels, and relating equally to COVID-19 and the Ukraine conflict. Our results suggest that while Chinese thumbnails are less formal and more candid, U.S. channels tend to use more deliberate, proper photographs as thumbnails. In particular, U.S. thumbnails are less colorful, more saturated, darker, more finely detailed, less symmetric, sparser, less varied, and more up close and personal than Chinese thumbnails. We suggest that most of these differences reflect cultural preferences, and that our methods and observations can serve as a baseline against which suspected visual propaganda can be computed and compared.
>
---
#### [replaced 086] Multi-modal Generative AI: Multi-modal LLMs, Diffusions, and the Unification
- **分类: cs.AI; cs.CV**

- **简介: 该论文聚焦多模态生成人工智能，旨在统一理解与生成任务。针对多模态大模型与扩散模型的分离问题，系统综述其架构与建模方法，探索融合二者的关键设计与策略，提出统一框架，并总结常用数据集，指明未来研究方向。**

- **链接: [https://arxiv.org/pdf/2409.14993v3](https://arxiv.org/pdf/2409.14993v3)**

> **作者:** Xin Wang; Yuwei Zhou; Bin Huang; Hong Chen; Wenwu Zhu
>
> **备注:** 21 pages, 10 figures, 3 tables
>
> **摘要:** Multi-modal generative AI (Artificial Intelligence) has attracted increasing attention from both academia and industry. Particularly, two dominant families of techniques have emerged: i) Multi-modal large language models (LLMs) demonstrate impressive ability for multi-modal understanding; and ii) Diffusion models exhibit remarkable multi-modal powers in terms of multi-modal generation. Therefore, this paper provides a comprehensive overview of multi-modal generative AI, including multi-modal LLMs, diffusions, and the unification for understanding and generation. To lay a solid foundation for unified models, we first provide a detailed review of both multi-modal LLMs and diffusion models respectively, including their probabilistic modeling procedure, multi-modal architecture design, and advanced applications to image/video LLMs as well as text-to-image/video generation. Furthermore, we explore the emerging efforts toward unified models for understanding and generation. To achieve the unification of understanding and generation, we investigate key designs including autoregressive-based and diffusion-based modeling, as well as dense and Mixture-of-Experts (MoE) architectures. We then introduce several strategies for unified models, analyzing their potential advantages and disadvantages. In addition, we summarize the common datasets widely used for multi-modal generative AI pretraining. Last but not least, we present several challenging future research directions which may contribute to the ongoing advancement of multi-modal generative AI.
>
---
#### [replaced 087] When to Think and When to Look: Uncertainty-Guided Lookback
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态视觉语言模型中的测试时思考（test-time thinking）问题，旨在提升视觉推理能力。针对“思考越多越好的误区”，提出基于不确定性的自适应回溯策略，通过短回溯提示与广度搜索，增强图像关联性，显著提升多个基准上的性能，实现新基准突破。**

- **链接: [https://arxiv.org/pdf/2511.15613v2](https://arxiv.org/pdf/2511.15613v2)**

> **作者:** Jing Bi; Filippos Bellos; Junjia Guo; Yayuan Li; Chao Huang; Yolo Y. Tang; Luchuan Song; Susan Liang; Zhongfei Mark Zhang; Jason J. Corso; Chenliang Xu
>
> **摘要:** Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets.
>
---
#### [replaced 088] M2SVid: End-to-End Inpainting and Refinement for Monocular-to-Stereo Video Conversion
- **分类: cs.CV**

- **简介: 该论文针对单目视频转立体视频任务，解决深度重投影导致的右视图缺失区域（遮挡）问题。提出M2SVid模型，基于Stable Video Diffusion，利用左视图、扭曲右视图和遮挡掩码作为条件，通过改进注意力机制实现端到端的高质量右视图生成与修复，显著提升质量与速度。**

- **链接: [https://arxiv.org/pdf/2505.16565v2](https://arxiv.org/pdf/2505.16565v2)**

> **作者:** Nina Shvetsova; Goutam Bhat; Prune Truong; Hilde Kuehne; Federico Tombari
>
> **备注:** To be published at 3DV 2026, project webpage https://m2svid.github.io/
>
> **摘要:** We tackle the problem of monocular-to-stereo video conversion and propose a novel architecture for inpainting and refinement of the warped right view obtained by depth-based reprojection of the input left view. We extend the Stable Video Diffusion (SVD) model to utilize the input left video, the warped right video, and the disocclusion masks as conditioning input to generate a high-quality right camera view. In order to effectively exploit information from neighboring frames for inpainting, we modify the attention layers in SVD to compute full attention for discoccluded pixels. Our model is trained to generate the right view video in an end-to-end manner without iterative diffusion steps by minimizing image space losses to ensure high-quality generation. Our approach outperforms previous state-of-the-art methods, being ranked best 2.6x more often than the second-place method in a user study, while being 6x faster.
>
---
#### [replaced 089] VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation
- **分类: cs.CV**

- **简介: 该论文针对图像与视频生成中对齐人类偏好的难题，提出VisionReward框架。通过分层视觉评估与线性加权实现细粒度、可解释的偏好学习，并设计多维一致性策略优化生成质量。实验表明其在预测准确性和生成效果上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2412.21059v3](https://arxiv.org/pdf/2412.21059v3)**

> **作者:** Jiazheng Xu; Yu Huang; Jiale Cheng; Yuanming Yang; Jiajun Xu; Yuan Wang; Wenbo Duan; Shen Yang; Qunlin Jin; Shurun Li; Jiayan Teng; Zhuoyi Yang; Wendi Zheng; Xiao Liu; Dan Zhang; Ming Ding; Xiaohan Zhang; Xiaotao Gu; Shiyu Huang; Minlie Huang; Jie Tang; Yuxiao Dong
>
> **备注:** 27 pages
>
> **摘要:** Visual generative models have achieved remarkable progress in synthesizing photorealistic images and videos, yet aligning their outputs with human preferences across critical dimensions remains a persistent challenge. Though reinforcement learning from human feedback offers promise for preference alignment, existing reward models for visual generation face limitations, including black-box scoring without interpretability and potentially resultant unexpected biases. We present VisionReward, a general framework for learning human visual preferences in both image and video generation. Specifically, we employ a hierarchical visual assessment framework to capture fine-grained human preferences, and leverages linear weighting to enable interpretable preference learning. Furthermore, we propose a multi-dimensional consistent strategy when using VisionReward as a reward model during preference optimization for visual generation. Experiments show that VisionReward can significantly outperform existing image and video reward models on both machine metrics and human evaluation. Notably, VisionReward surpasses VideoScore by 17.2% in preference prediction accuracy, and text-to-video models with VisionReward achieve a 31.6% higher pairwise win rate compared to the same models using VideoScore. All code and datasets are provided at https://github.com/THUDM/VisionReward.
>
---
#### [replaced 090] LightMem: Lightweight and Efficient Memory-Augmented Generation
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文针对大语言模型在动态环境中难以有效利用历史交互信息的问题，提出轻量高效的LightMem记忆系统。受人类记忆模型启发，将记忆分为感知、短期和长期三阶段，实现高效的信息压缩、组织与离线更新。实验表明，LightMem显著提升问答准确率，大幅降低计算开销与API调用次数。**

- **链接: [https://arxiv.org/pdf/2510.18866v2](https://arxiv.org/pdf/2510.18866v2)**

> **作者:** Jizhan Fang; Xinle Deng; Haoming Xu; Ziyan Jiang; Yuqi Tang; Ziwen Xu; Shumin Deng; Yunzhi Yao; Mengru Wang; Shuofei Qiao; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Despite their remarkable capabilities, Large Language Models (LLMs) struggle to effectively leverage historical interaction information in dynamic and complex environments. Memory systems enable LLMs to move beyond stateless interactions by introducing persistent information storage, retrieval, and utilization mechanisms. However, existing memory systems often introduce substantial time and computational overhead. To this end, we introduce a new memory system called LightMem, which strikes a balance between the performance and efficiency of memory systems. Inspired by the Atkinson-Shiffrin model of human memory, LightMem organizes memory into three complementary stages. First, cognition-inspired sensory memory rapidly filters irrelevant information through lightweight compression and groups information according to their topics. Next, topic-aware short-term memory consolidates these topic-based groups, organizing and summarizing content for more structured access. Finally, long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference. On LongMemEval and LoCoMo, using GPT and Qwen backbones, LightMem consistently surpasses strong baselines, improving QA accuracy by up to 7.7% / 29.3%, reducing total token usage by up to 38x / 20.9x and API calls by up to 30x / 55.5x, while purely online test-time costs are even lower, achieving up to 106x / 117x token reduction and 159x / 310x fewer API calls. The code is available at https://github.com/zjunlp/LightMem.
>
---
#### [replaced 091] SD-MVS: Segmentation-Driven Deformation Multi-View Stereo with Spherical Refinement and EM optimization
- **分类: cs.CV**

- **简介: 该论文针对纹理缺失区域的3D重建难题，提出SD-MVS方法。利用SAM分割语义实例，驱动像素级变形优化匹配与传播；结合球坐标与梯度下降改进深度与法向优化；采用EM算法自适应调参。在ETH3D和Tanks and Temples上实现更优重建效果且效率更高。**

- **链接: [https://arxiv.org/pdf/2401.06385v2](https://arxiv.org/pdf/2401.06385v2)**

> **作者:** Zhenlong Yuan; Jiakai Cao; Zhaoxin Li; Hao Jiang; Zhaoqi Wang
>
> **备注:** Published to AAAI2024
>
> **摘要:** In this paper, we introduce Segmentation-Driven Deformation Multi-View Stereo (SD-MVS), a method that can effectively tackle challenges in 3D reconstruction of textureless areas. We are the first to adopt the Segment Anything Model (SAM) to distinguish semantic instances in scenes and further leverage these constraints for pixelwise patch deformation on both matching cost and propagation. Concurrently, we propose a unique refinement strategy that combines spherical coordinates and gradient descent on normals and pixelwise search interval on depths, significantly improving the completeness of reconstructed 3D model. Furthermore, we adopt the Expectation-Maximization (EM) algorithm to alternately optimize the aggregate matching cost and hyperparameters, effectively mitigating the problem of parameters being excessively dependent on empirical tuning. Evaluations on the ETH3D high-resolution multi-view stereo benchmark and the Tanks and Temples dataset demonstrate that our method can achieve state-of-the-art results with less time consumption.
>
---
#### [replaced 092] Video-R4: Reinforcing Text-Rich Video Reasoning with Visual Rumination
- **分类: cs.CV**

- **简介: 该论文针对文本丰富的视频理解任务，解决模型因单次扫描导致的细粒度证据遗漏问题。提出Video-R4框架，通过迭代视觉沉思（选择帧、局部放大、重编码、更新推理）提升推理能力。构建两个数据集，采用多阶段训练方法，使模型在多个视频与文档问答任务上达到领先效果。**

- **链接: [https://arxiv.org/pdf/2511.17490v2](https://arxiv.org/pdf/2511.17490v2)**

> **作者:** Yolo Y. Tang; Daiki Shimada; Hang Hua; Chao Huang; Jing Bi; Rogerio Feris; Chenliang Xu
>
> **摘要:** Understanding text-rich videos requires reading small, transient textual cues that often demand repeated inspection. Yet most video QA models rely on single-pass perception over fixed frames, leading to hallucinations and failures on fine-grained evidence. Inspired by how humans pause, zoom, and re-read critical regions, we introduce Video-R4 (Reinforcing Text-Rich Video Reasoning with Visual Rumination), a video reasoning LMM that performs visual rumination: iteratively selecting frames, zooming into informative regions, re-encoding retrieved pixels, and updating its reasoning state. We construct two datasets with executable rumination trajectories: Video-R4-CoT-17k for supervised practice and Video-R4-RL-30k for reinforcement learning. We propose a multi-stage rumination learning framework that progressively finetunes a 7B LMM to learn atomic and mixing visual operations via SFT and GRPO-based RL. Video-R4-7B achieves state-of-the-art results on M4-ViteVQA and further generalizes to multi-page document QA, slides QA, and generic video QA, demonstrating that iterative rumination is an effective paradigm for pixel-grounded multimodal reasoning. Project Page: https://yunlong10.github.io/Video-R4/
>
---
#### [replaced 093] MAPo : Motion-Aware Partitioning of Deformable 3D Gaussian Splatting for High-Fidelity Dynamic Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对动态场景重建中变形3D高斯溅射因统一模型无法捕捉复杂运动导致的模糊与细节丢失问题，提出MAPo框架。通过动态评分分区，对高动态区域进行时序递归分割并独立建模，低动态区域视为静态，结合跨帧一致性损失提升连续性，显著改善渲染质量。**

- **链接: [https://arxiv.org/pdf/2508.19786v2](https://arxiv.org/pdf/2508.19786v2)**

> **作者:** Han Jiao; Jiakai Sun; Yexing Xu; Lei Zhao; Wei Xing; Huaizhong Lin
>
> **摘要:** 3D Gaussian Splatting, known for enabling high-quality static scene reconstruction with fast rendering, is increasingly being applied to multi-view dynamic scene reconstruction. A common strategy involves learning a deformation field to model the temporal changes of a canonical set of 3D Gaussians. However, these deformation-based methods often produce blurred renderings and lose fine motion details in highly dynamic regions due to the inherent limitations of a single, unified model in representing diverse motion patterns. To address these challenges, we introduce Motion-Aware Partitioning of Deformable 3D Gaussian Splatting (MAPo), a novel framework for high-fidelity dynamic scene reconstruction. Its core is a dynamic score-based partitioning strategy that distinguishes between high- and low-dynamic 3D Gaussians. For high-dynamic 3D Gaussians, we recursively partition them temporally and duplicate their deformation networks for each new temporal segment, enabling specialized modeling to capture intricate motion details. Concurrently, low-dynamic 3DGs are treated as static to reduce computational costs. However, this temporal partitioning strategy for high-dynamic 3DGs can introduce visual discontinuities across frames at the partition boundaries. To address this, we introduce a cross-frame consistency loss, which not only ensures visual continuity but also further enhances rendering quality. Extensive experiments demonstrate that MAPo achieves superior rendering quality compared to baselines while maintaining comparable computational costs, particularly in regions with complex or rapid motions.
>
---
#### [replaced 094] ContextFlow: Training-Free Video Object Editing via Adaptive Context Enrichment
- **分类: cs.CV**

- **简介: 该论文针对训练-free视频对象编辑任务，解决对象编辑中保真度与时序一致性差的问题。提出ContextFlow框架，通过高阶流求解器与自适应上下文增强机制，动态融合多路径信息，并基于响应性指标定位关键层，实现精准、高效编辑，显著提升结果质量。**

- **链接: [https://arxiv.org/pdf/2509.17818v2](https://arxiv.org/pdf/2509.17818v2)**

> **作者:** Yiyang Chen; Xuanhua He; Xiujun Ma; Yue Ma
>
> **备注:** The project page is at https://yychen233.github.io/ContextFlow-page
>
> **摘要:** Training-free video object editing aims to achieve precise object-level manipulation, including object insertion, swapping, and deletion. However, it faces significant challenges in maintaining fidelity and temporal consistency. Existing methods, often designed for U-Net architectures, suffer from two primary limitations: inaccurate inversion due to first-order solvers, and contextual conflicts caused by crude "hard" feature replacement. These issues are more challenging in Diffusion Transformers (DiTs), where the unsuitability of prior layer-selection heuristics makes effective guidance challenging. To address these limitations, we introduce ContextFlow, a novel training-free framework for DiT-based video object editing. In detail, we first employ a high-order Rectified Flow solver to establish a robust editing foundation. The core of our framework is Adaptive Context Enrichment (for specifying what to edit), a mechanism that addresses contextual conflicts. Instead of replacing features, it enriches the self-attention context by concatenating Key-Value pairs from parallel reconstruction and editing paths, empowering the model to dynamically fuse information. Additionally, to determine where to apply this enrichment (for specifying where to edit), we propose a systematic, data-driven analysis to identify task-specific vital layers. Based on a novel Guidance Responsiveness Metric, our method pinpoints the most influential DiT blocks for different tasks (e.g., insertion, swapping), enabling targeted and highly effective guidance. Extensive experiments show that ContextFlow significantly outperforms existing training-free methods and even surpasses several state-of-the-art training-based approaches, delivering temporally coherent, high-fidelity results.
>
---
#### [replaced 095] Personalized Generative Low-light Image Denoising and Enhancement
- **分类: cs.CV**

- **简介: 该论文针对低光图像噪声与模糊问题，提出基于个性化图库的扩散模型DiffPGD。通过构建身份一致的物理属性缓冲区，作为先验知识融入模型，实现无需微调的高质量去噪与增强，显著提升低信噪比下的图像恢复效果。**

- **链接: [https://arxiv.org/pdf/2412.14327v3](https://arxiv.org/pdf/2412.14327v3)**

> **作者:** Xijun Wang; Prateek Chennuri; Dilshan Godaliyadda; Yu Yuan; Bole Ma; Xingguang Zhang; Hamid R. Sheikh; Stanley Chan
>
> **摘要:** Modern cameras' performance in low-light conditions remains suboptimal due to fundamental limitations in photon shot noise and sensor read noise. Generative image restoration methods have shown promising results compared to traditional approaches, but they suffer from hallucinatory content generation when the signal-to-noise ratio (SNR) is low. Leveraging the availability of personalized photo galleries of the users, we introduce Diffusion-based Personalized Generative Denoising (DiffPGD), a new approach that builds a customized diffusion model for individual users. Our key innovation lies in the development of an identity-consistent physical buffer that extracts the physical attributes of the person from the gallery. This ID-consistent physical buffer serves as a robust prior that can be seamlessly integrated into the diffusion model to restore degraded images without the need for fine-tuning. Over a wide range of low-light testing scenarios, we show that DiffPGD achieves superior image denoising and enhancement performance compared to existing diffusion-based denoising approaches. Our project page can be found at \href{https://genai-restore.github.io/DiffPGD/}{\textcolor{purple}{\textbf{https://genai-restore.github.io/DiffPGD/}}}.
>
---
#### [replaced 096] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search
- **分类: cs.CV**

- **简介: 该论文聚焦于人本开放未来任务发现（HOTD）任务，旨在让大模型在动态多变的人类意图下自动发现能减轻人类负担的未来任务。为此，提出HOTD-Bench基准和CMAST搜索框架，通过多智能体协作与可扩展树状搜索，显著提升任务发现性能，并有效增强现有大模型能力。**

- **链接: [https://arxiv.org/pdf/2511.18929v2](https://arxiv.org/pdf/2511.18929v2)**

> **作者:** Zijian Song; Xiaoxin Lin; Tao Pu; Zhenlong Yuan; Guangrun Wang; Liang Lin
>
> **备注:** accepted to AAAI 2026, 10 pages, 9 figures
>
> **摘要:** Recent progress in robotics and embodied AI is largely driven by Large Multimodal Models (LMMs). However, a key challenge remains underexplored: how can we advance LMMs to discover tasks that directly assist humans in open-future scenarios, where human intentions are highly concurrent and dynamic. In this work, we formalize the problem of Human-centric Open-future Task Discovery (HOTD), focusing particularly on identifying tasks that reduce human effort across multiple plausible futures. To facilitate this study, we propose an HOTD-Bench, which features over 2K real-world videos, a semi-automated annotation pipeline, and a simulation-based protocol tailored for open-set future evaluation. Additionally, we propose the Collaborative Multi-Agent Search Tree (CMAST) framework, which decomposes the complex reasoning through a multi-agent system and structures the reasoning process through a scalable search tree module. In our experiments, CMAST achieves the best performance on the HOTD-Bench, significantly surpassing existing LMMs. It also integrates well with existing LMMs, consistently improving performance.
>
---
#### [replaced 097] MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequence
- **分类: cs.CV**

- **简介: 该论文提出MovieDreamer，一种用于长视频生成的分层框架，解决扩散模型在长序列中叙事连贯性与角色一致性差的问题。通过结合自回归模型与扩散渲染，利用多模态剧本增强视觉一致性，实现高质量、长时长电影级视频生成。**

- **链接: [https://arxiv.org/pdf/2407.16655v3](https://arxiv.org/pdf/2407.16655v3)**

> **作者:** Canyu Zhao; Mingyu Liu; Wen Wang; Weihua Chen; Fan Wang; Hao Chen; Bo Zhang; Chunhua Shen
>
> **备注:** 30 pages, 22 figures
>
> **摘要:** Recent advancements in video generation have primarily leveraged diffusion models for short-duration content. However, these approaches often fall short in modeling complex narratives and maintaining character consistency over extended periods, which is essential for long-form video production like movies. We propose MovieDreamer, a novel hierarchical framework that integrates the strengths of autoregressive models with diffusion-based rendering to pioneer long-duration video generation with intricate plot progressions and high visual fidelity. Our approach utilizes autoregressive models for global narrative coherence, predicting sequences of visual tokens that are subsequently transformed into high-quality video frames through diffusion rendering. This method is akin to traditional movie production processes, where complex stories are factorized down into manageable scene capturing. Further, we employ a multimodal script that enriches scene descriptions with detailed character information and visual style, enhancing continuity and character identity across scenes. We present extensive experiments across various movie genres, demonstrating that our approach not only achieves superior visual and narrative quality but also effectively extends the duration of generated content significantly beyond current capabilities. Homepage: https://aim-uofa.github.io/MovieDreamer/.
>
---
#### [replaced 098] Endoshare: A Publicly Available, Surgeons-Friendly Solution to De-Identify and Manage Surgical Videos
- **分类: cs.CV**

- **简介: 该论文针对手术视频共享中的格式异构与隐私问题，提出Endoshare系统，实现视频合并、标准化与去标识化。通过用户中心设计与多轮测试，验证其易用性与有效性，为外科训练与研究提供公开可用的解决方案。**

- **链接: [https://arxiv.org/pdf/2510.20087v2](https://arxiv.org/pdf/2510.20087v2)**

> **作者:** Lorenzo Arboit; Dennis N. Schneider; Britty Baby; Vinkle Srivastav; Pietro Mascagni; Nicolas Padoy
>
> **备注:** 13 pages, 6 figures. Source-available software: https://camma-public.github.io/Endoshare/
>
> **摘要:** Video-based assessment and surgical data science can advance surgical training, research, and quality improvement, yet adoption remains limited by heterogeneous recording formats and privacy concerns linked to video sharing. This work develops, evaluates, and publicly releases Endoshare, a surgeon-friendly application that merges, standardizes, and de-identifies endoscopic videos. Development followed an iterative, user-centered software life cycle. In the analysis phase, an internal survey of four clinicians and four computer scientists, based on 10 usability heuristics, identified early requirements and guided a cross-platform, privacy-by-design architecture. Prototype testing reported high usability for clinicians (4.68 +/- 0.40 out of 5) and for computer scientists (4.03 +/- 0.51 out of 5), with the lowest score (4.00 +/- 0.93 out of 5) relating to label clarity, prompting interface refinement to streamline case selection, video merging, automated out-of-body removal, and filename pseudonymization. In the testing phase, ten surgeons completed an external survey combining the same heuristics with Technology Acceptance Model constructs, reporting high perceived usefulness (5.07 +/- 1.75 out of 7), ease of use (5.15 +/- 1.71 out of 7), heuristic usability (4.38 +/- 0.48 out of 5), and strong recommendation likelihood (9.20 +/- 0.79 out of 10). A performance assessment across different hardware and configurations showed that processing time increased proportionally with video duration and was consistently lower in fast mode. Endoshare is a publicly available solution to manage surgical videos, with potential to support training, research, and quality improvement. Compliance certification and broader interoperability validation are needed to establish it as a reliable tool for surgical video management. The software is available at https://camma-public.github.io/Endoshare
>
---
#### [replaced 099] From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images
- **分类: cs.CV**

- **简介: 该论文属于空间转录组学中的基因表达预测任务。针对传统方法因固定大小斑点导致空间分辨率损失的问题，提出PixNet模型，直接从病理切片图像生成连续的高密度基因表达图，并按需聚合不同尺度斑点的表达值，实现多尺度精准预测。**

- **链接: [https://arxiv.org/pdf/2503.01347v4](https://arxiv.org/pdf/2503.01347v4)**

> **作者:** Ruikun Zhang; Yan Yang; Liyuan Pan
>
> **摘要:** Spatial transcriptomics (ST) measures gene expression at fine-grained spatial resolution, offering insights into tissue molecular landscapes. Previous methods for spatial gene expression prediction typically crop spots of interest from histopathology slide images, and train models to map each spot to a corresponding gene expression profile. However, these methods inherently lose the spatial resolution in gene expression: 1) each spot often contains multiple cells with distinct gene expression profiles; 2) spots are typically defined at fixed spatial resolutions, limiting the ability to predict gene expression at varying scales. To address these limitations, this paper presents PixNet, a dense prediction network capable of predicting spatially resolved gene expression across spots of varying sizes and scales directly from histopathology slide images. Different from previous methods that map individual spots to gene expression values, we generate a spatially dense continuous gene expression map from the histopathology slide image, and aggregate values within spots of interest to predict the gene expression. Our PixNet outperforms state-of-the-art methods on four common ST datasets in multiple spatial scales. The source code will be publicly available.
>
---
#### [replaced 100] SuperQuadricOcc: Multi-Layer Gaussian Approximation of Superquadrics for Real-Time Self-Supervised Occupancy Estimation
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中的语义占用估计任务，解决高内存占用与实时性差的问题。提出SuperQuadricOcc，用多层球面网格近似超立方体，实现高效自监督训练。相比传统高斯方法，内存减少75%，推理速度提升124%，精度提升5.9%，且仅需空间标签。**

- **链接: [https://arxiv.org/pdf/2511.17361v2](https://arxiv.org/pdf/2511.17361v2)**

> **作者:** Seamie Hayes; Reenu Mohandas; Tim Brophy; Alexandre Boulch; Ganesh Sistu; Ciaran Eising
>
> **摘要:** Semantic occupancy estimation enables comprehensive scene understanding for automated driving, providing dense spatial and semantic information essential for perception and planning. While Gaussian representations have been widely adopted in self-supervised occupancy estimation, the deployment of a large number of Gaussian primitives drastically increases memory requirements and is not suitable for real-time inference. In contrast, superquadrics permit reduced primitive count and lower memory requirements due to their diverse shape set. However, implementation into a self-supervised occupancy model is nontrivial due to the absence of a superquadric rasterizer to enable model supervision. Our proposed method, SuperQuadricOcc, employs a superquadric-based scene representation. By leveraging a multi-layer icosphere-tessellated Gaussian approximation of superquadrics, we enable Gaussian rasterization for supervision during training. On the Occ3D dataset, SuperQuadricOcc achieves a 75% reduction in memory footprint, 124% faster inference, and a 5.9% improvement in mIoU compared to previous Gaussian-based methods, without the use of temporal labels. To our knowledge, this is the first occupancy model to enable real-time inference while maintaining competitive performance. The use of superquadrics reduces the number of primitives required for scene modeling by 84% relative to Gaussian-based approaches. Finally, evaluation against prior methods is facilitated by our fast superquadric voxelization module. The code will be made available at https://github.com/seamie6/SuperQuadricOcc.
>
---
#### [replaced 101] Click2Graph: Interactive Panoptic Video Scene Graphs from a Single Click
- **分类: cs.CV**

- **简介: 该论文提出Click2Graph，面向交互式全景视频场景图生成（PVSG）任务，解决现有方法无法融合用户引导与时空语义推理的问题。通过单次点击实现目标分割、跟踪及关系推断，引入动态交互发现与联合分类模块，实现可控、可解释的视频理解。**

- **链接: [https://arxiv.org/pdf/2511.15948v2](https://arxiv.org/pdf/2511.15948v2)**

> **作者:** Raphael Ruschel; Hardikkumar Prajapati; Awsafur Rahman; B. S. Manjunath
>
> **摘要:** State-of-the-art Video Scene Graph Generation (VSGG) systems provide structured visual understanding but operate as closed, feed-forward pipelines with no ability to incorporate human guidance. In contrast, promptable segmentation models such as SAM2 enable precise user interaction but lack semantic or relational reasoning. We introduce Click2Graph, the first interactive framework for Panoptic Video Scene Graph Generation (PVSG) that unifies visual prompting with spatial, temporal, and semantic understanding. From a single user cue, such as a click or bounding box, Click2Graph segments and tracks the subject across time, autonomously discovers interacting objects, and predicts <subject, object, predicate> triplets to form a temporally consistent scene graph. Our framework introduces two key components: a Dynamic Interaction Discovery Module that generates subject-conditioned object prompts, and a Semantic Classification Head that performs joint entity and predicate reasoning. Experiments on the OpenPVSG benchmark demonstrate that Click2Graph establishes a strong foundation for user-guided PVSG, showing how human prompting can be combined with panoptic grounding and relational inference to enable controllable and interpretable video scene understanding.
>
---
#### [replaced 102] Harnessing Vision-Language Models for Time Series Anomaly Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对时间序列异常检测任务，解决传统方法缺乏视觉-时序理解能力的问题。提出两阶段框架：先用轻量级ViT4TS定位候选异常，再用VLM4TS结合全局上下文与视觉理解能力精炼检测结果。无需时间序列训练，显著提升准确率与效率。**

- **链接: [https://arxiv.org/pdf/2506.06836v2](https://arxiv.org/pdf/2506.06836v2)**

> **作者:** Zelin He; Sarah Alnegheimish; Matthew Reimherr
>
> **备注:** Accepted at AAAI 2026 (Oral)
>
> **摘要:** Time-series anomaly detection (TSAD) has played a vital role in a variety of fields, including healthcare, finance, and sensor-based condition monitoring. Prior methods, which mainly focus on training domain-specific models on numerical data, lack the visual-temporal understanding capacity that human experts have to identify contextual anomalies. To fill this gap, we explore a solution based on vision language models (VLMs). Recent studies have shown the ability of VLMs for visual understanding tasks, yet their direct application to time series has fallen short on both accuracy and efficiency. To harness the power of VLMs for TSAD, we propose a two-stage solution, with (1) ViT4TS, a vision-screening stage built on a relatively lightweight pre-trained vision encoder, which leverages 2D time series representations to accurately localize candidate anomalies; (2) VLM4TS, a VLM-based stage that integrates global temporal context and VLM's visual understanding capacity to refine the detection upon the candidates provided by ViT4TS. We show that without any time-series training, VLM4TS outperforms time-series pre-trained and from-scratch baselines in most cases, yielding a 24.6% improvement in F1-max score over the best baseline. Moreover, VLM4TS also consistently outperforms existing language model-based TSAD methods and is on average 36x more efficient in token usage.
>
---
#### [replaced 103] Disc3D: Automatic Curation of High-Quality 3D Dialog Data via Discriminative Object Referring
- **分类: cs.CV**

- **简介: 该论文针对3D多模态大模型缺乏高质量对话数据的问题，提出Disc3D自动构建管道，解决视角与对象指代模糊性。通过规则约束与多模态模型协同，实现无监督、可扩展的数据生成，产出超200万样本的高质量3D对话数据集，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.18817v2](https://arxiv.org/pdf/2511.18817v2)**

> **作者:** Siyuan Wei; Chunjie Wang; Xiao Liu; Xiaosheng Yan; Zhishan Zhou; Rui Huang
>
> **备注:** 8 pages
>
> **摘要:** 3D Multi-modal Large Language Models (MLLMs) still lag behind their 2D peers, largely because large-scale, high-quality 3D scene-dialogue datasets remain scarce. Prior efforts hinge on expensive human annotation and leave two key ambiguities unresolved: viewpoint ambiguity, where spatial language presumes unknown camera poses, and object referring ambiguity, where non-exclusive descriptions blur the line between targets and distractors. We therefore present a fully automated pipeline that converts raw 3D scans into unambiguous, high-quality dialogue data at a fraction of the previous cost. By synergizing rule-based constraints with 2D MLLMs and LLMs, the pipeline enables controllable, scalable generation without human intervention. The pipeline comprises four stages: (1) meta-annotation collection harvesting object-, frame-, and scene-level captions, (2) scene graph construction with relation correction to capture proximal object relations, (3) discriminative object referring that generates exclusive and compact descriptions, and (4) multi-task data generation synthesizing diverse dialogues. Our pipeline systematically mitigates inherent flaws in source datasets and produces the final Disc3D dataset, over 2 million samples in 25K hybrid 3D scenes, spanning scene, view, and object captioning, visual grounding, and five object-centric QA tasks. Extensive experiments demonstrate that training with Disc3D yields consistent, significant improvements on both public benchmarks and our multifaceted Disc3D-QA tasks. Code, data, and models will be publicly available.
>
---
#### [replaced 104] FMPlug: Plug-In Foundation Flow-Matching Priors for Inverse Problems
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **简介: 该论文提出FMPlug框架，用于解决图像超分辨率与高斯去模糊等逆问题。针对传统方法依赖领域特定先验的问题，利用观测与目标间的相似性及生成流的高斯性，通过时间自适应预热和高斯正则化，增强通用基础流匹配先验，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2508.00721v2](https://arxiv.org/pdf/2508.00721v2)**

> **作者:** Yuxiang Wan; Ryan Devera; Wenjie Zhang; Ju Sun
>
> **摘要:** We present FMPlug, a novel plug-in framework that enhances foundation flow-matching (FM) priors for solving ill-posed inverse problems. Unlike traditional approaches that rely on domain-specific or untrained priors, FMPlug smartly leverages two simple but powerful insights: the similarity between observed and desired objects and the Gaussianity of generative flows. By introducing a time-adaptive warm-up strategy and sharp Gaussianity regularization, FMPlug unlocks the true potential of domain-agnostic foundation models. Our method beats state-of-the-art methods that use foundation FM priors by significant margins, on image super-resolution and Gaussian deblurring.
>
---
#### [replaced 105] CardioComposer: Leveraging Differentiable Geometry for Compositional Control of Anatomical Diffusion Models
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出CardioComposer，一种基于可微几何的3D心血管解剖生成框架。针对生成模型中几何可控性与真实感的权衡问题，利用可微的椭球体基元实现对解剖结构尺寸、形状和位置的解耦控制，通过体素级几何矩损失在扩散采样中引导生成，支持多结构组合控制，并适用于多种含非凸结构的解剖系统。**

- **链接: [https://arxiv.org/pdf/2509.08015v2](https://arxiv.org/pdf/2509.08015v2)**

> **作者:** Karim Kadry; Shoaib Goraya; Ajay Manicka; Abdalla Abdelwahed; Naravich Chutisilp; Farhad Nezami; Elazer Edelman
>
> **备注:** 10 pages, 16 figures
>
> **摘要:** Generative models of 3D cardiovascular anatomy can synthesize informative structures for clinical research and medical device evaluation, but face a trade-off between geometric controllability and realism. We propose CardioComposer: a programmable, inference-time framework for generating multi-class anatomical label maps based on interpretable ellipsoidal primitives. These primitives represent geometric attributes such as the size, shape, and position of discrete substructures. We specifically develop differentiable measurement functions based on voxel-wise geometric moments, enabling loss-based gradient guidance during diffusion model sampling. We demonstrate that these losses can constrain individual geometric attributes in a disentangled manner and provide compositional control over multiple substructures. Finally, we show that our method is compatible with a wide array of anatomical systems containing non-convex substructures, spanning cardiac, vascular, and skeletal organs.
>
---
#### [replaced 106] AutoFocus-IL: VLM-based Saliency Maps for Data-Efficient Visual Imitation Learning without Extra Human Annotations
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AutoFocus-IL，一种基于视觉语言模型的视觉模仿学习方法，旨在提升数据效率与泛化能力。针对现有方法依赖人工标注或眼动数据的问题，该方法利用VLM自动生成时序显著性图，引导策略关注任务相关特征，抑制干扰因素，从而在无需额外人类标注的情况下实现更优性能。**

- **链接: [https://arxiv.org/pdf/2511.18617v2](https://arxiv.org/pdf/2511.18617v2)**

> **作者:** Litian Gong; Fatemeh Bahrani; Yutai Zhou; Amin Banayeeanzade; Jiachen Li; Erdem Bıyık
>
> **备注:** 8 pages, 6 figures. Code and datasets available at http://autofocus-il.github.io/
>
> **摘要:** AutoFocus-IL is a simple yet effective method to improve data efficiency and generalization in visual imitation learning by guiding policies to attend to task-relevant features rather than distractors and spurious correlations. Although saliency regularization has emerged as a promising way to achieve this, existing approaches typically require costly supervision such as human gaze data or manual saliency annotations. In contrast, AutoFocus-IL leverages vision-language models (VLMs) to automatically identify and track key objects in demonstrations, generating temporal saliency maps that highlight causal visual signals while suppressing distractors. These maps are then used to regularize behavior cloning policies, yielding stronger alignment between visual attention and task-relevant cues. Experiments in both the CARLA simulator and real-robot manipulation tasks demonstrate that AutoFocus-IL not only outperforms standard behavior cloning but also surpasses state-of-the-art baselines that assume privileged access to human supervision, such as gaze data. Code, datasets, and trained policy videos are available at https://AutoFocus-IL.github.io/.
>
---
#### [replaced 107] Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文针对大语言模型在长周期稀疏奖励任务中的探索-利用平衡问题，提出SPEAR方法。通过自模仿学习与渐进式熵调控，分阶段促进探索与利用，显著提升成功率，且开销极小，适用于各类代理强化学习任务。**

- **链接: [https://arxiv.org/pdf/2509.22601v3](https://arxiv.org/pdf/2509.22601v3)**

> **作者:** Yulei Qin; Xiaoyu Tan; Zhengbao He; Gang Li; Haojia Lin; Zongyi Li; Zihan Xu; Yuchen Shi; Siqi Cai; Renting Rui; Shaofei Cai; Yuzheng Cai; Xuan Zhang; Sheng Ye; Ke Li; Xing Sun
>
> **备注:** 45 pages, 14 figures
>
> **摘要:** Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent's own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL, where a replay buffer stores good experience for off-policy update, by gradually steering the policy entropy across stages. Specifically, the proposed curriculum scheduling harmonizes intrinsic reward shaping and self-imitation to 1) expedite exploration via frequent tool interactions at the beginning, and 2) strengthen exploitation of successful tactics upon convergence towards familiarity with the environment. We also combine bag-of-tricks of industrial RL optimizations for a strong baseline Dr.BoT to demonstrate our effectiveness. In ALFWorld and WebShop, SPEAR increases the success rates of GRPO/GiGPO/Dr.BoT by up to 16.1%/5.1%/8.6% and 20.7%/11.8%/13.9%, respectively. In AIME24 and AIME25, SPEAR boosts Dr.BoT by up to 3.8% and 6.1%, respectively. Such gains incur only 10%-25% extra theoretical complexity and negligible runtime overhead in practice, demonstrating the plug-and-play scalability of SPEAR.
>
---
#### [replaced 108] Metis-HOME: Hybrid Optimized Mixture-of-Experts for Multimodal Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大模型在复杂推理与通用理解间的权衡问题，提出Metis-HOME框架。通过构建“思考”与“非思考”双专家分支及轻量路由机制，实现高效推理与通用能力的协同提升。**

- **链接: [https://arxiv.org/pdf/2510.20519v2](https://arxiv.org/pdf/2510.20519v2)**

> **作者:** Xiaohan Lan; Fanfan Liu; Haibo Qiu; Siqi Yang; Delian Ruan; Peng Shi; Lin Ma
>
> **摘要:** Inspired by recent advancements in LLM reasoning, the field of multimodal reasoning has seen remarkable progress, achieving significant performance gains on intricate tasks such as mathematical problem-solving. Despite this progress, current multimodal large reasoning models exhibit two key limitations. They tend to employ computationally expensive reasoning even for simple queries, leading to inefficiency. Furthermore, this focus on specialized reasoning often impairs their broader, more general understanding capabilities. In this paper, we propose Metis-HOME: a Hybrid Optimized Mixture-of-Experts framework designed to address this trade-off. Metis-HOME enables a ''Hybrid Thinking'' paradigm by structuring the original dense model into two distinct expert branches: a thinking branch tailored for complex, multi-step reasoning, and a non-thinking branch optimized for rapid, direct inference on tasks like general VQA and OCR. A lightweight, trainable router dynamically allocates queries to the most suitable expert. We instantiate Metis-HOME by adapting the Qwen2.5-VL-7B into an MoE architecture. Comprehensive evaluations reveal that our approach not only substantially enhances complex reasoning abilities but also improves the model's general capabilities, reversing the degradation trend observed in other reasoning-specialized models. Our work establishes a new paradigm for building powerful and versatile MLLMs, effectively resolving the prevalent reasoning-vs-generalization dilemma. Code and weights are available at https://github.com/MM-Thinking/Metis-HOME.
>
---
#### [replaced 109] STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control
- **分类: cs.CV**

- **简介: 该论文针对边缘场景重建中的高效数据聚合问题，提出STT-GS框架。针对传统方法忽视视图质量贡献差异的问题，设计了以高斯溅射质量为导向的优化目标，通过样本-传输策略与联合客户端选择及功率控制，实现低开销下的高质量重建。**

- **链接: [https://arxiv.org/pdf/2510.13186v3](https://arxiv.org/pdf/2510.13186v3)**

> **作者:** Zhen Li; Xibin Jin; Guoliang Li; Shuai Wang; Miaowen Wen; Huseyin Arslan; Derrick Wing Kwan Ng; Chengzhong Xu
>
> **摘要:** Edge Gaussian splatting (EGS), which aggregates data from distributed clients (e.g., drones) and trains a global GS model at the edge (e.g., ground server), is an emerging paradigm for scene reconstruction in low-altitude economy. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead.Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments reveal that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. The GS-oriented objective can be accurately predicted with low sampling ratios (e.g., 10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
>
---
#### [replaced 110] Rethinking the Learning Paradigm for Facial Expression Recognition
- **分类: cs.CV**

- **简介: 该论文聚焦于面部表情识别（FER）任务，针对真实数据集因主观标注和类别相似性导致的模糊标注问题，提出摒弃传统将模糊标注转为精确一热编码的做法，主张采用弱监督策略直接利用原始模糊标注训练模型，以更合理地建模真实场景下的表达识别。**

- **链接: [https://arxiv.org/pdf/2209.15402v3](https://arxiv.org/pdf/2209.15402v3)**

> **作者:** Weijie Wang; Nicu Sebe; Bruno Lepri
>
> **摘要:** Due to the subjective crowdsourcing annotations and the inherent inter-class similarity of facial expressions, the real-world Facial Expression Recognition (FER) datasets usually exhibit ambiguous annotation. To simplify the learning paradigm, most previous methods convert ambiguous annotation results into precise one-hot annotations and train FER models in an end-to-end supervised manner. In this paper, we rethink the existing training paradigm and propose that it is better to use weakly supervised strategies to train FER models with original ambiguous annotation.
>
---
#### [replaced 111] Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对3D重建中视点规划效率与鲁棒性不足的问题，提出Hestia框架。通过分层结构、面感知设计、多样数据集和近似贪婪策略，实现高效、实时的五自由度视点选择，显著提升覆盖率与重建精度，适用于真实场景。**

- **链接: [https://arxiv.org/pdf/2508.01014v3](https://arxiv.org/pdf/2508.01014v3)**

> **作者:** Cheng-You Lu; Zhuoli Zhuang; Nguyen Thanh Trung Le; Da Xiao; Yu-Cheng Chang; Thomas Do; Srinath Sridhar; Chin-teng Lin
>
> **备注:** Accepted to the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Advances in 3D reconstruction and novel view synthesis have enabled efficient and photorealistic rendering. However, images for reconstruction are still either largely manual or constrained by simple preplanned trajectories. To address this issue, recent works propose generalizable next-best-view planners that do not require online learning. Nevertheless, robustness and performance remain limited across various shapes. Hence, this study introduces Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction (Hestia), which addresses the shortcomings of the reinforcement learning-based generalizable approaches for five-degree-of-freedom viewpoint prediction. Hestia systematically improves the planners through four components: a more diverse dataset to promote robustness, a hierarchical structure to manage the high-dimensional continuous action search space, a close-greedy strategy to mitigate spurious correlations, and a face-aware design to avoid overlooking geometry. Experimental results show that Hestia achieves non-marginal improvements, with at least a 4% gain in coverage ratio, while reducing Chamfer Distance by 50% and maintaining real-time inference. In addition, Hestia outperforms prior methods by at least 12% in coverage ratio with a 5-image budget and remains robust to object placement variations. Finally, we demonstrate that Hestia, as a next-best-view planner, is feasible for the real-world application. Our project page is https://johnnylu305.github.io/hestia web.
>
---
#### [replaced 112] RoPECraft: Training-Free Motion Transfer with Trajectory-Guided RoPE Optimization on Diffusion Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出RoPECraft，一种无需训练的扩散模型视频运动迁移方法。通过修改旋转位置编码（RoPE）中的复数张量，将参考视频的运动信息嵌入生成过程，并在去噪过程中通过流匹配优化轨迹对齐，同时利用傅里叶相位正则化抑制高频伪影，实现高质量、忠实于文本的运动迁移。**

- **链接: [https://arxiv.org/pdf/2505.13344v2](https://arxiv.org/pdf/2505.13344v2)**

> **作者:** Ahmet Berke Gokmen; Yigit Ekin; Bahri Batuhan Bilecen; Aysegul Dundar
>
> **备注:** https://berkegokmen1.github.io/RoPECraft/
>
> **摘要:** We propose RoPECraft, a training-free video motion transfer method for diffusion transformers that operates solely by modifying their rotary positional embeddings (RoPE). We first extract dense optical flow from a reference video, and utilize the resulting motion offsets to warp the complex-exponential tensors of RoPE, effectively encoding motion into the generation process. These embeddings are then further optimized during denoising time steps via trajectory alignment between the predicted and target velocities using a flow-matching objective. To keep the output faithful to the text prompt and prevent duplicate generations, we incorporate a regularization term based on the phase components of the reference video's Fourier transform, projecting the phase angles onto a smooth manifold to suppress high-frequency artifacts. Experiments on benchmarks reveal that RoPECraft outperforms all recently published methods, both qualitatively and quantitatively.
>
---
#### [replaced 113] FastGS: Training 3D Gaussian Splatting in 100 Seconds
- **分类: cs.CV**

- **简介: 该论文针对3D高斯点云渲染训练效率低的问题，提出FastGS框架。通过基于多视角一致性的自适应密度调节策略，无需预算限制，显著提升训练速度，实现100秒内完成训练，在多个任务中均展现高效与通用性。**

- **链接: [https://arxiv.org/pdf/2511.04283v2](https://arxiv.org/pdf/2511.04283v2)**

> **作者:** Shiwei Ren; Tianci Wen; Yongchun Fang; Biao Lu
>
> **备注:** Project page: https://fastgs.github.io/
>
> **摘要:** The dominant 3D Gaussian splatting (3DGS) acceleration methods fail to properly regulate the number of Gaussians during training, causing redundant computational time overhead. In this paper, we propose FastGS, a novel, simple, and general acceleration framework that fully considers the importance of each Gaussian based on multi-view consistency, efficiently solving the trade-off between training time and rendering quality. We innovatively design a densification and pruning strategy based on multi-view consistency, dispensing with the budgeting mechanism. Extensive experiments on Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets demonstrate that our method significantly outperforms the state-of-the-art methods in training speed, achieving a 3.32$\times$ training acceleration and comparable rendering quality compared with DashGaussian on the Mip-NeRF 360 dataset and a 15.45$\times$ acceleration compared with vanilla 3DGS on the Deep Blending dataset. We demonstrate that FastGS exhibits strong generality, delivering 2-7$\times$ training acceleration across various tasks, including dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and simultaneous localization and mapping. The project page is available at https://fastgs.github.io/
>
---
#### [replaced 114] Shape-Adapting Gated Experts: Dynamic Expert Routing for Colonoscopic Lesion Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对胃镜图像中细胞形态与尺度多样带来的分割难题，提出SAGE框架，通过动态专家路由实现输入自适应的视觉推理。采用双路径结构与分层门控机制，融合CNN与Transformer优势，显著提升医学图像分割精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.18493v2](https://arxiv.org/pdf/2511.18493v2)**

> **作者:** Gia Huy Thai; Hoang-Nguyen Vu; Anh-Minh Phan; Quang-Thinh Ly; Tram Dinh; Thi-Ngoc-Truc Nguyen; Nhat Ho
>
> **摘要:** The substantial diversity in cell scale and form remains a primary challenge in computer-aided cancer detection on gigapixel Whole Slide Images (WSIs), attributable to cellular heterogeneity. Existing CNN-Transformer hybrids rely on static computation graphs with fixed routing, which consequently causes redundant computation and limits their adaptability to input variability. We propose Shape-Adapting Gated Experts (SAGE), an input-adaptive framework that enables dynamic expert routing in heterogeneous visual networks. SAGE reconfigures static backbones into dynamically routed expert architectures. SAGE's dual-path design features a backbone stream that preserves representation and selectively activates an expert path through hierarchical gating. This gating mechanism operates at multiple hierarchical levels, performing a two-level, hierarchical selection between shared and specialized experts to modulate model logits for Top-K activation. Our Shape-Adapting Hub (SA-Hub) harmonizes structural and semantic representations across the CNN and the Transformer module, effectively bridging diverse modules. Embodied as SAGE-UNet, our model achieves superior segmentation on three medical benchmarks: EBHI, DigestPath, and GlaS, yielding state-of-the-art Dice Scores of 95.57%, 95.16%, and 94.17%, respectively, and robustly generalizes across domains by adaptively balancing local refinement and global context. SAGE provides a scalable foundation for dynamic expert routing, enabling flexible visual reasoning.
>
---
#### [replaced 115] MHR: Momentum Human Rig
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出MHR，一种结合骨架与形状解耦的参数化人体模型，融合现代骨骼系统与非线性姿态校正机制，解决人体动画中表达力与解剖合理性不足的问题，支持在AR/VR及图形管线中的高效集成。**

- **链接: [https://arxiv.org/pdf/2511.15586v3](https://arxiv.org/pdf/2511.15586v3)**

> **作者:** Aaron Ferguson; Ahmed A. A. Osman; Berta Bescos; Carsten Stoll; Chris Twigg; Christoph Lassner; David Otte; Eric Vignola; Fabian Prada; Federica Bogo; Igor Santesteban; Javier Romero; Jenna Zarate; Jeongseok Lee; Jinhyung Park; Jinlong Yang; John Doublestein; Kishore Venkateshan; Kris Kitani; Ladislav Kavan; Marco Dal Farra; Matthew Hu; Matthew Cioffi; Michael Fabris; Michael Ranieri; Mohammad Modarres; Petr Kadlecek; Rawal Khirodkar; Rinat Abdrashitov; Romain Prévost; Roman Rajbhandari; Ronald Mallet; Russell Pearsall; Sandy Kao; Sanjeev Kumar; Scott Parrish; Shoou-I Yu; Shunsuke Saito; Takaaki Shiratori; Te-Li Wang; Tony Tung; Yichen Xu; Yuan Dong; Yuhua Chen; Yuanlu Xu; Yuting Ye; Zhongshi Jiang
>
> **摘要:** We present MHR, a parametric human body model that combines the decoupled skeleton/shape paradigm of ATLAS with a flexible, modern rig and pose corrective system inspired by the Momentum library. Our model enables expressive, anatomically plausible human animation, supporting non-linear pose correctives, and is designed for robust integration in AR/VR and graphics pipelines.
>
---
#### [replaced 116] Adapting Vision-Language Models for Evaluating World Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对世界模型视频生成结果的细粒度评估难题，提出UNIVERSE——一个基于视觉语言模型的统一评估框架。通过适配VLM以支持动作与角色识别的多格式评估，在数据与计算受限下实现高效、语义感知的自动评价，实验与人类判断高度一致，显著提升评估准确性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2506.17967v2](https://arxiv.org/pdf/2506.17967v2)**

> **作者:** Mariya Hendriksen; Tabish Rashid; David Bignell; Raluca Georgescu; Abdelhak Lemkhenter; Katja Hofmann; Sam Devlin; Sarah Parisot
>
> **备注:** NeurIPS LAW 2025 (Oral)
>
> **摘要:** World models - generative models that simulate environment dynamics conditioned on past observations and actions - are gaining prominence in planning, simulation, and embodied AI. However, evaluating their rollouts remains a fundamental challenge, requiring fine-grained, temporally grounded assessment of action alignment and semantic consistency - capabilities not captured by existing metrics. Vision-Language Models (VLMs) have shown promise as automatic evaluators of generative content due to their strong multimodal reasoning abilities. Yet, their use in fine-grained, temporally sensitive evaluation tasks remains limited and requires targeted adaptation. We introduce an evaluation protocol targeting two recognition tasks - action recognition and character recognition - each assessed across binary, multiple-choice, and open-ended formats. To support this, we present UNIVERSE (UNIfied Vision-language Evaluator for Rollouts in Simulated Environments), a VLM-based evaluator for video world model rollouts adapted under data and compute constraints. In our extensive experiments totaling over 5,154 GPU-days, we explore full, partial, and parameter-efficient adaptation methods across various task formats, context lengths, sampling methods, and data compositions. The resulting unified evaluator achieves parity with task-specific checkpoints. Human studies across seven diverse environments confirm strong alignment with human judgments, establishing UNIVERSE as a lightweight, adaptable, and semantics-aware evaluator for video world models.
>
---
#### [replaced 117] Alternating Perception-Reasoning for Hallucination-Resistant Video Understanding
- **分类: cs.CV**

- **简介: 该论文针对视频理解任务中的幻觉问题，提出感知-推理交替框架（Video-PLR）。通过循环式感知与精确时间戳描述缓解证据不足，结合事实感知评估器（FAE）提供反幻觉奖励，提升推理可靠性。在3B和7B模型规模上均达到当前最佳性能，且数据效率高。**

- **链接: [https://arxiv.org/pdf/2511.18463v2](https://arxiv.org/pdf/2511.18463v2)**

> **作者:** Bowei Pu; Chuanbin Liu; Yifan Ge; Peicheng Zhou; Yiwei Sun; Zhiying Lu; Jiankang Wang; Hongtao Xie
>
> **备注:** 32 pages, 36 figures
>
> **摘要:** Sufficient visual perception is the foundation of video reasoning. Nevertheless, existing Video Reasoning LLMs suffer from perception shortcuts, relying on a flawed single-step perception paradigm. This paradigm describes the video and then conducts reasoning, which runs the risk of insufficient evidence and emergent hallucinations. To address these issues, we introduce a new framework that integrates a loop-based paradigm with an anti-hallucination reward. First, to address the insufficient evidence, we introduce the Perception Loop Reasoning (PLR) paradigm. Instead of describing the video at once, each loop requires the model to describe a video segment with precise timestamps, analyze this segment, and decide the next action. Second, for the risk of hallucinations, the Factual-Aware Evaluator (FAE) evaluates each perception result as a reliable anti-hallucination reward. This reward encourages the model to provide sufficient and precise video evidence. Our FAE, which performs comparably to GPT-4o, is tuned on our AnetHallu-117K, a large-scale hallucination judgment preference dataset. Extensive experiments show that our Video-PLR achieves the state-of-the-art in both 3B and 7B parameter scales and has the best data efficiency. Our code, models, and datasets are released on: https://github.com/BoweiPu/VideoPLR.
>
---
#### [replaced 118] LayerComposer: Multi-Human Personalized Generation via Layered Canvas
- **分类: cs.CV**

- **简介: 该论文针对多人物个性化图像生成中空间控制弱、扩展性差的问题，提出LayerComposer框架。通过分层画布实现人物的交互式布局与缩放，结合透明潜在剪枝与分层交叉引用训练，提升生成质量与效率，显著改善空间控制、构图一致性与身份保真度。**

- **链接: [https://arxiv.org/pdf/2510.20820v3](https://arxiv.org/pdf/2510.20820v3)**

> **作者:** Guocheng Gordon Qian; Ruihang Zhang; Tsai-Shien Chen; Yusuf Dalva; Anujraaj Argo Goyal; Willi Menapace; Ivan Skorokhodov; Meng Dong; Arpit Sahni; Daniil Ostashev; Ju Hu; Sergey Tulyakov; Kuan-Chieh Jackson Wang
>
> **备注:** 17 pages including appendix, preprint. Project page: https://snap-research.github.io/layercomposer/
>
> **摘要:** Despite their impressive visual fidelity, existing personalized image generators lack interactive control over spatial composition and scale poorly to multiple humans. To address these limitations, we present LayerComposer, an interactive and scalable framework for multi-human personalized generation. Inspired by professional image-editing software, LayerComposer provides intuitive reference-based human injection, allowing users to place and resize multiple subjects directly on a layered digital canvas to guide personalized generation. The core of our approach is the layered canvas, a novel representation where each subject is placed on a distinct layer, enabling interactive and occlusion-free composition. We further introduce a transparent latent pruning mechanism that improves scalability by decoupling computational cost from the number of subjects, and a layerwise cross-reference training strategy that mitigates copy-paste artifacts. Extensive experiments demonstrate that LayerComposer achieves superior spatial control, coherent composition, and identity preservation compared to state-of-the-art methods in multi-human personalized image generation.
>
---
#### [replaced 119] X-ReID: Multi-granularity Information Interaction for Video-Based Visible-Infrared Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对视频可见光-红外行人重识别（VVI-ReID）任务，解决模态差异与时空信息利用难题。提出X-ReID框架，通过跨模态原型协作对齐特征，并设计多粒度信息交互机制，融合短时、长时及跨模态信息，提升序列级表示能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17964v2](https://arxiv.org/pdf/2511.17964v2)**

> **作者:** Chenyang Yu; Xuehu Liu; Pingping Zhang; Huchuan Lu
>
> **备注:** Accepted by AAAI2026. More modifications may be performed
>
> **摘要:** Large-scale vision-language models (e.g., CLIP) have recently achieved remarkable performance in retrieval tasks, yet their potential for Video-based Visible-Infrared Person Re-Identification (VVI-ReID) remains largely unexplored. The primary challenges are narrowing the modality gap and leveraging spatiotemporal information in video sequences. To address the above issues, in this paper, we propose a novel cross-modality feature learning framework named X-ReID for VVI-ReID. Specifically, we first propose a Cross-modality Prototype Collaboration (CPC) to align and integrate features from different modalities, guiding the network to reduce the modality discrepancy. Then, a Multi-granularity Information Interaction (MII) is designed, incorporating short-term interactions from adjacent frames, long-term cross-frame information fusion, and cross-modality feature alignment to enhance temporal modeling and further reduce modality gaps. Finally, by integrating multi-granularity information, a robust sequence-level representation is achieved. Extensive experiments on two large-scale VVI-ReID benchmarks (i.e., HITSZ-VCM and BUPTCampus) demonstrate the superiority of our method over state-of-the-art methods. The source code is released at https://github.com/AsuradaYuci/X-ReID.
>
---
#### [replaced 120] Panoramic Distortion-Aware Tokenization for Person Detection and Localization in Overhead Fisheye Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对航拍鱼眼图像中人物检测与定位任务，解决人物旋转和小目标漏检问题。通过将鱼眼图重映射为等距柱状全景图，利用全景几何特性，提出畸变感知的分块令牌化方法，增强对小人物的检测能力，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2503.14228v3](https://arxiv.org/pdf/2503.14228v3)**

> **作者:** Nobuhiko Wakai; Satoshi Sato; Yasunori Ishii; Takayoshi Yamashita
>
> **摘要:** Person detection in overhead fisheye images is challenging due to person rotation and small persons. Prior work has mainly addressed person rotation, leaving the small-person problem underexplored. We remap fisheye images to equirectangular panoramas to handle rotation and exploit panoramic geometry to handle small persons more effectively. Conventional detection methods tend to favor larger persons because they dominate the attention maps, causing smaller persons to be missed. In hemispherical equirectangular panoramas, we find that apparent person height decreases approximately linearly with the vertical angle near the top of the image. Using this finding, we introduce panoramic distortion-aware tokenization to enhance the detection of small persons. This tokenization procedure divides panoramic features using self-similar figures that enable the determination of optimal divisions without gaps, and we leverage the maximum significance values in each tile of the token groups to preserve the significance areas of smaller persons. We propose a transformer-based person detection and localization method that combines panoramic-image remapping and the tokenization procedure. Extensive experiments demonstrated that our method outperforms conventional methods on large-scale datasets.
>
---
#### [replaced 121] Learning Hierarchical Sparse Transform Coding of 3DGS
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对3DGS模型压缩任务，解决现有方法因缺乏分析-合成变换导致熵编码负担过重的问题。提出SHTC方法，通过稀疏引导的分层变换编码，结合KLT与轻量神经变换，实现高效压缩，显著提升率-失真性能与解码速度。**

- **链接: [https://arxiv.org/pdf/2505.22908v2](https://arxiv.org/pdf/2505.22908v2)**

> **作者:** Hao Xu; Xiaolin Wu; Xi Zhang
>
> **备注:** Our code will be released at \href{https://github.com/hxu160/SHTC_for_3DGS_compression}{here}
>
> **摘要:** 3D Gaussian Splatting (3DGS) supports fast, high quality, novel view synthesis but has a heavy memory footprint, making the compression of its model crucial. Current state-of-the-art (SOTA) 3DGS compression methods adopt an anchor-based architecture that pairs the Scaffold-GS representation with conditional entropy coding. However, these methods forego the analysis-synthesis transform, a vital mechanism in visual data compression. As a result, redundancy remains intact in the signal and its removal is left to the entropy coder, which computationally overburdens the entropy coding module, increasing coding latency. Even with added complexity thorough redundancy removal is a task unsuited to an entropy coder. To fix this critical omission, we introduce a Sparsity-guided Hierarchical Transform Coding (SHTC) method, the first study on the end-to-end learned neural transform coding of 3DGS. SHTC applies KLT to decorrelate intra-anchor attributes, followed by quantization and entropy coding, and then compresses KLT residuals with a low-complexity, scene-adaptive neural transform. Aided by the sparsity prior and deep unfolding technique, the learned transform uses only a few trainable parameters, reducing the memory usage. Overall, SHTC achieves an appreciably improved R-D performance and at the same time higher decoding speed over SOTA. Its prior-guided, parameter-efficient design may also inspire low-complexity neural image and video codecs. Our code will be released at https://github.com/hxu160/SHTC_for_3DGS_compression.
>
---
#### [replaced 122] Rethinking Two-Stage Referring-by-Tracking in Referring Multi-Object Tracking: Make it Strong Again
- **分类: cs.CV**

- **简介: 该论文针对Referring Multi-Object Tracking（RMOT）任务，解决两阶段Referring-by-Tracking（RBT）框架因特征构建粗糙和对应关系建模脆弱而性能落后的问题。提出FlexHook框架，通过采样式特征构建与语言条件注入的C-Hook，以及基于成对对应关系的PCD，显著提升模型性能，首次实现两阶段RBT全面超越当前最优方法。**

- **链接: [https://arxiv.org/pdf/2503.07516v4](https://arxiv.org/pdf/2503.07516v4)**

> **作者:** Weize Li; Yunhao Du; Qixiang Yin; Zhicheng Zhao; Fei Su
>
> **摘要:** Referring Multi-Object Tracking (RMOT) aims to track multiple objects specified by natural language expressions in videos. With the recent significant progress of one-stage methods, the two-stage Referring-by-Tracking (RBT) paradigm has gradually lost its popularity. However, its lower training cost and flexible incremental deployment remain irreplaceable. Rethinking existing two-stage RBT frameworks, we identify two fundamental limitations: the overly heuristic feature construction and fragile correspondence modeling. To address these issues, we propose FlexHook, a novel two-stage RBT framework. In FlexHook, the proposed Conditioning Hook (C-Hook) redefines the feature construction by a sampling-based strategy and language-conditioned cue injection. Then, we introduce a Pairwise Correspondence Decoder (PCD) that replaces CLIP-based similarity matching with active correspondence modeling, yielding a more flexible and robust strategy. Extensive experiments on multiple benchmarks (Refer-KITTI/v2, Refer-Dance, and LaMOT) demonstrate that FlexHook becomes the first two-stage RBT approach to comprehensively outperform current state-of-the-art methods. Code can be found in the Supplementary Materials.
>
---
#### [replaced 123] SplatCo: Structure-View Collaborative Gaussian Splatting for Detail-Preserving Rendering of Large-Scale Unbounded Scenes
- **分类: cs.CV**

- **简介: 论文提出SplatCo，一种用于大规模无界场景高保真渲染的结构-视图协同高斯溅射框架。针对复杂户外场景中全局一致性与局部细节难以兼顾的问题，创新性地引入跨结构融合与跨视图协同训练机制，显著提升重建质量与多视角一致性。**

- **链接: [https://arxiv.org/pdf/2505.17951v3](https://arxiv.org/pdf/2505.17951v3)**

> **作者:** Haihong Xiao; Jianan Zou; Yuxin Zhou; Ying He; Wenxiong Kang
>
> **摘要:** We present SplatCo, a structure-view collaborative Gaussian splatting framework for high-fidelity rendering of complex outdoor environments. SplatCo builds upon two novel components: (1) a cross-structure collaboration module that combines global tri-plane representations, which capture coarse scene layouts, with local context grid features that represent fine surface details. This fusion is achieved through a novel hierarchical compensation strategy, ensuring both global consistency and local detail preservation; and (2) a cross-view assisted training strategy that enhances multi-view consistency by synchronizing gradient updates across viewpoints, applying visibility-aware densification, and pruning overfitted or inaccurate Gaussians based on structural consistency. Through joint optimization of structural representation and multi-view coherence, SplatCo effectively reconstructs fine-grained geometric structures and complex textures in large-scale scenes. Comprehensive evaluations on 13 diverse large-scale scenes, including Mill19, MatrixCity, Tanks & Temples, WHU, and custom aerial captures, demonstrate that SplatCo consistently achieves higher reconstruction quality than state-of-the-art methods, with PSNR improvements of 1-2 dB and SSIM gains of 0.1 to 0.2. These results establish a new benchmark for high-fidelity rendering of large-scale unbounded scenes. Code and additional information are available at https://github.com/SCUT-BIP-Lab/SplatCo.
>
---
#### [replaced 124] The Early Bird Identifies the Worm: You Can't Beat a Head Start in Long-Term Body Re-ID (ECHO-BID)
- **分类: cs.CV**

- **简介: 该论文研究长时序人体重识别任务，旨在解决因衣物变化导致的识别困难。通过将EVA-02等视觉基础模型进行领域迁移学习，提出ECHO-BID框架，利用少量数据实现卓越性能，验证了大模型“先发优势”与小规模领域数据的有效性。**

- **链接: [https://arxiv.org/pdf/2507.17640v2](https://arxiv.org/pdf/2507.17640v2)**

> **作者:** Thomas M. Metz; Matthew Q. Hill; Alice J. O'Toole
>
> **摘要:** A wide range of model-based approaches to long-term person re-identification have been proposed. Whether these models perform more accurately than direct domain transfer learning applied to extensively trained large-scale foundation models is not known. We applied domain transfer learning for long-term person re-id to four vision foundation models (CLIP, DINOv2, AIMv2, and EVA-02). Domain-adapted versions of all four models %CLIP-L, DINOv2-L, AIMv2-L, and EVA-02-L surpassed existing state-of-the-art models by a large margin in highly unconstrained viewing environments. Decision score fusion of the four models improved performance over any individual model. Of the individual models, the EVA-02 foundation model provided the best ``head start'' to long-term re-id, surpassing other models on three of the four performance metrics by substantial margins. Accordingly, we introduce $\textbf{E}$va $\textbf{C}$lothes-Change from $\textbf{H}$idden $\textbf{O}$bjects - $\textbf{B}$ody $\textbf{ID}$entification (ECHO-BID), a class of long-term re-id models built on the object-pretrained EVA-02 Large backbones. Ablation experiments varying backbone size, scale of object classification pretraining, and transfer learning protocol indicated that model size and the use of a smaller, but more challenging transfer learning protocol are critical features in performance. We conclude that foundation models provide a head start to domain transfer learning and support state-of-the-art performance with modest amounts of domain data. The limited availability of long-term re-id data makes this approach advantageous.
>
---
#### [replaced 125] NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文针对面部情绪识别（FER）模型泛化能力差的问题，提出NeuroGaze-Distill框架。通过脑电（EEG）数据导出静态情感原型与抑郁相关几何先验，引导图像模型学习更鲁棒的特征表示，提升跨数据集性能，无需部署时使用非视觉信号。**

- **链接: [https://arxiv.org/pdf/2509.11916v2](https://arxiv.org/pdf/2509.11916v2)**

> **作者:** Zilin Li; Weiwei Xu; Xuanqi Zhao; Yiran Zhu
>
> **备注:** Preprint. Vision-only deployment; EEG used to form static prototypes. Includes appendix, 7 figures and 3 tables. Considering submission to ICLR 2026. Revision note: This version corrects inaccuracies in the authors' institutional affiliations. No technical content has been modified
>
> **摘要:** Facial emotion recognition (FER) models trained only on pixels often fail to generalize across datasets because facial appearance is an indirect and biased proxy for underlying affect. We present NeuroGaze-Distill, a cross-modal distillation framework that transfers brain-informed priors into an image-only FER student via static Valence/Arousal (V/A) prototypes and a depression-inspired geometric prior (D-Geo). A teacher trained on EEG topographic maps from DREAMER (with MAHNOB-HCI as unlabeled support) produces a consolidated 5x5 V/A prototype grid that is frozen and reused; no EEG-face pairing and no non-visual signals at deployment are required. The student (ResNet-18/50) is trained on FERPlus with conventional CE/KD and two lightweight regularizers: (i) Proto-KD (cosine) aligns student features to the static prototypes; (ii) D-Geo softly shapes the embedding geometry in line with affective findings often reported in depression research (e.g., anhedonia-like contraction in high-valence regions). We evaluate both within-domain (FERPlus validation) and cross-dataset protocols (AffectNet-mini; optional CK+), reporting standard 8-way scores alongside present-only Macro-F1 and balanced accuracy to fairly handle label-set mismatch. Ablations attribute consistent gains to prototypes and D-Geo, and favor 5x5 over denser grids for stability. The method is simple, deployable, and improves robustness without architectural complexity.
>
---
